from io import BytesIO

import pytest
from PIL import Image

from app.api.adapters.detection_request import adapt_detection_request, parse_history
from app.schemas.detection import DetectionMode, DetectionTaskType
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.detection_application import DetectionApplicationService
from app.services.followup_application import FollowupApplicationService, FollowupContextError
from app.services.input_builder import DetectionInputBuilder
from app.services.remote_image_downloader import DownloadedRemoteImage
from app.services.session_service import SessionService


def valid_png_bytes(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(output, format="PNG")
    return output.getvalue()


class StubWebpageExtractor:
    def extract(self, url):
        raise AssertionError(f"unexpected webpage extraction: {url}")


class StubRemoteImageDownloader:
    def download(self, url):
        output = BytesIO()
        Image.new("RGB", (2, 2), color="white").save(output, format="PNG")
        return DownloadedRemoteImage(output.getvalue(), ".png", "image/png", url)


class StubOrchestrator:
    def __init__(self):
        self.calls = []

    def detect(self, **kwargs):
        self.calls.append(kwargs)
        mode = "multimodal" if kwargs.get("text") and (kwargs.get("image_content") or kwargs.get("image_url")) else "text_only"
        return {
            "mode": mode,
            "label": "Cannot_Determine",
            "summary": "Insufficient evidence.",
            "question": kwargs["question"] or "",
            "evidence": [],
            "raw": {},
        }


class StubAnnotatedOrchestrator:
    def __init__(self, annotated_path):
        self.annotated_path = annotated_path

    def detect_task(self, _task):
        artifacts = {"annotated_image_path": str(self.annotated_path)}
        return {
            "mode": "image_only",
            "label": "伪造图像",
            "summary": "检测到局部伪造。",
            "question": "",
            "image_verdict": {
                "label": "伪造图像",
                "artifacts": dict(artifacts),
                "suspicious_regions": [{"region": "1,1,3,3"}],
            },
            "evidence": [],
            "raw": {"image_forensics": {"artifacts": dict(artifacts)}},
        }


class StubFollowupAnalysis:
    def __init__(self):
        self.calls = []

    def explain(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": "Explanation from the existing result."}


def make_session(tmp_path):
    conversation_store = ConversationHistoryStore(log_dir=tmp_path)
    artifact_store = ArtifactStore(tmp_path)
    return SessionService(conversation_store, artifact_store), artifact_store


def test_parse_history_accepts_only_json_arrays():
    assert parse_history('[{"role":"user","content":"hello"}]')[0]["role"] == "user"
    assert parse_history('{"role":"user"}') == []
    assert parse_history("not-json") == []


def test_detection_request_adapter_normalizes_http_fields():
    request = adapt_detection_request(
        task="multimodal_fact_checking",
        text="claim",
        image_content=None,
        image_filename=None,
        image_content_type=None,
        image_url="  https://example.com/image.jpg  ",
        webpage_url=None,
        question=None,
        language="zh",
        history="[]",
        conversation_id="conversation",
    )

    assert request.image_url == "https://example.com/image.jpg"
    assert request.task == DetectionTaskType.MULTIMODAL_FACT_CHECKING
    assert request.history == []


def test_detection_application_runs_pipeline_and_persists_session(tmp_path):
    session_service, artifact_store = make_session(tmp_path)
    orchestrator = StubOrchestrator()
    application = DetectionApplicationService(
        orchestrator,
        DetectionInputBuilder(
            StubWebpageExtractor(),
            artifact_store,
            remote_image_downloader=StubRemoteImageDownloader(),
        ),
        session_service,
    )
    request = adapt_detection_request(
        task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        text="claim",
        image_content=None,
        image_filename=None,
        image_content_type=None,
        image_url="https://example.com/image.jpg",
        webpage_url=None,
        question="is it true?",
        language="en",
        history='[{"role":"system","content":"context"}]',
        conversation_id="application-test",
    )

    result = application.detect(request)

    assert result["conversation_id"] == "application-test"
    assert len(result["conversation_history"]) == 2
    assert orchestrator.calls[0]["history"] == request.history
    assert set(result["raw"]["timings"]) == {
        "session_preparation_seconds",
        "input_preparation_seconds",
        "orchestration_seconds",
        "application_seconds",
    }
    assert session_service.snapshot("application-test")["latest_result"]["label"] == "Cannot_Determine"


def test_detection_application_persists_full_resolution_annotation(tmp_path):
    session_service, artifact_store = make_session(tmp_path / "conversations")
    annotated_path = tmp_path / "annotated.png"
    Image.new("RGB", (20, 10), color="white").save(annotated_path)
    application = DetectionApplicationService(
        StubAnnotatedOrchestrator(annotated_path),
        DetectionInputBuilder(StubWebpageExtractor(), artifact_store),
        session_service,
    )
    request = adapt_detection_request(
        task=DetectionTaskType.DEEPFAKE_DETECTION,
        text=None,
        image_content=valid_png_bytes(),
        image_filename="input.png",
        image_content_type="image/png",
        image_url=None,
        webpage_url=None,
        question=None,
        language="zh",
        history="[]",
        conversation_id="annotated-preview",
    )

    result = application.detect(request)

    url = result["image_verdict"]["artifacts"]["annotated_image_url"]
    assert url.startswith("/conversations/annotated-preview/assets/images/")
    assert result["raw"]["image_forensics"]["artifacts"]["annotated_image_url"] == url
    stored_name = result["image_verdict"]["artifacts"]["annotated_image_asset"]["stored_filename"]
    stored_path = session_service.resolve_asset("annotated-preview", f"images/{stored_name}")
    assert stored_path is not None
    assert Image.open(stored_path).size == (20, 10)
    latest = session_service.snapshot("annotated-preview")["latest_result"]
    assert latest["image_verdict"]["artifacts"]["annotated_image_url"] == url


def test_followup_application_rejects_session_without_detection(tmp_path):
    session_service, _ = make_session(tmp_path)
    application = FollowupApplicationService(session_service, analysis_service=object())

    with pytest.raises(FollowupContextError, match="没有可追问"):
        application.followup(
            question="why?",
            conversation_id="empty-session",
            language="en",
            orchestrator=StubOrchestrator(),
        )


def test_followup_application_explains_without_replacing_latest_result(tmp_path):
    session_service, _ = make_session(tmp_path)
    original_result = {
        "mode": "text_only",
        "label": "Mostly_True",
        "summary": "Supported.",
    }
    session_service.append_detection(
        "followup-explain",
        {"mode": "text_only", "text": "claim"},
        original_result,
    )
    analysis = StubFollowupAnalysis()
    application = FollowupApplicationService(session_service, analysis)

    result = application.followup(
        question="Why?",
        conversation_id="followup-explain",
        language="en",
        orchestrator=StubOrchestrator(),
    )

    assert result["label"] == "Mostly_True"
    assert result["conversation_history"][-1]["content"] == "Explanation from the existing result."
    assert session_service.snapshot("followup-explain")["latest_result"] == original_result
    assert len(analysis.calls) == 1


def test_followup_application_reruns_detection_on_explicit_request(tmp_path):
    session_service, _ = make_session(tmp_path)
    session_service.append_detection(
        "followup-rerun",
        {"mode": "text_only", "text": "claim", "language": "zh"},
        {"mode": "text_only", "label": "Cannot_Determine", "summary": "Unclear."},
    )
    orchestrator = StubOrchestrator()
    application = FollowupApplicationService(session_service, StubFollowupAnalysis())

    result = application.followup(
        question="请从头重新检测",
        conversation_id="followup-rerun",
        language="zh",
        orchestrator=orchestrator,
    )

    assert result["followup"] == {"question": "请从头重新检测", "rerun_detection": True}
    assert orchestrator.calls[0]["text"] == "claim"
    assert "mode" not in orchestrator.calls[0]
    assert len(result["conversation_history"]) == 4


def test_failed_followup_rerun_preserves_the_saved_detection(tmp_path):
    class FailingOrchestrator:
        def detect(self, **_kwargs):
            return {"mode": "text_only", "label": None, "decision_status": "technical_error"}

    session_service, _ = make_session(tmp_path)
    original_result = {"mode": "text_only", "label": "True", "summary": "Supported."}
    session_service.append_detection("rerun-failure", {"text": "claim"}, original_result)
    application = FollowupApplicationService(session_service, StubFollowupAnalysis())

    result = application.followup(
        question="请重新检测",
        conversation_id="rerun-failure",
        language="zh",
        orchestrator=FailingOrchestrator(),
    )

    assert result["decision_status"] == "technical_error"
    snapshot = session_service.snapshot("rerun-failure")
    assert snapshot["latest_result"] == original_result
    assert len(snapshot["conversation_history"]) == 2


def test_followup_application_preserves_deepfake_task_on_rerun(tmp_path):
    session_service, _ = make_session(tmp_path)
    session_service.append_detection(
        "followup-deepfake",
        {
            "task": "deepfake_detection",
            "mode": "image_only",
            "image_url": "https://example.com/image.jpg",
            "language": "zh",
        },
        {"mode": "image_only", "label": "Cannot_Determine", "summary": "Unclear."},
    )
    orchestrator = StubOrchestrator()
    application = FollowupApplicationService(session_service, StubFollowupAnalysis())

    application.followup(
        question="请从头重新检测",
        conversation_id="followup-deepfake",
        language="zh",
        orchestrator=orchestrator,
    )

    assert orchestrator.calls[0]["mode"] == DetectionMode.IMAGE_ONLY


def test_followup_rerun_reuses_conversation_image_path(tmp_path):
    session_service, artifact_store = make_session(tmp_path)
    artifact = artifact_store.save_image(
        "followup-local-image",
        b"saved-image",
        filename="image.png",
        content_type="image/png",
        source_url="https://example.com/image.png",
    )
    session_service.append_detection(
        "followup-local-image",
        {
            "task": "deepfake_detection",
            "mode": "image_only",
            "image_url": "https://example.com/image.png",
            "image_urls": ["https://example.com/image.png"],
            "image_asset": artifact_store.artifact_to_legacy_asset(artifact),
            "image_assets": [artifact_store.artifact_to_legacy_asset(artifact)],
            "language": "zh",
        },
        {"mode": "image_only", "label": "Cannot_Determine", "summary": "Unclear."},
    )
    orchestrator = StubOrchestrator()

    FollowupApplicationService(session_service, StubFollowupAnalysis()).followup(
        question="请重新检测",
        conversation_id="followup-local-image",
        language="zh",
        orchestrator=orchestrator,
    )

    call = orchestrator.calls[0]
    assert call["image_path"] == artifact.local_path
    assert call["image_url"] is None
    assert call["image_urls"] == []
