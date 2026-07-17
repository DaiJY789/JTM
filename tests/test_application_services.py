import pytest

from app.api.adapters.detection_request import adapt_detection_request, parse_history
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.detection_application import DetectionApplicationService
from app.services.followup_application import FollowupApplicationService, FollowupContextError
from app.services.input_builder import DetectionInputBuilder
from app.services.session_service import SessionService


class StubWebpageExtractor:
    def extract(self, url):
        raise AssertionError(f"unexpected webpage extraction: {url}")


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
    assert request.history == []


def test_detection_application_runs_pipeline_and_persists_session(tmp_path):
    session_service, artifact_store = make_session(tmp_path)
    orchestrator = StubOrchestrator()
    application = DetectionApplicationService(
        orchestrator,
        DetectionInputBuilder(StubWebpageExtractor(), artifact_store),
        session_service,
    )
    request = adapt_detection_request(
        text="claim",
        image_content=None,
        image_filename=None,
        image_content_type=None,
        image_url=None,
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
    assert session_service.snapshot("application-test")["latest_result"]["label"] == "Cannot_Determine"


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
