import json
import re
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.api.dependencies import get_detection_orchestrator
from app.core.config import settings
from app.core.logging import log_io_event, setup_logging
from app.main import app
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.remote_image_downloader import DownloadedRemoteImage
from app.services.session_service import SessionService


def test_same_day_logging_setups_append_to_one_file(tmp_path):
    try:
        first = setup_logging(str(tmp_path / "app.log"))
        log_io_event("request_input", "request-1", "/detect", {"text": "测试输入"})
        second = setup_logging(str(tmp_path / "app.log"))
        log_io_event("request_output", "request-2", "/detect", {"label": "True"})

        assert first == second
        assert re.fullmatch(r"app_\d{4}-\d{2}-\d{2}\.log", first.name)
        content = first.read_text(encoding="utf-8")
        assert "测试输入" in content
        assert "request_output" in content
        assert first.stat().st_mode & 0o777 == 0o600
    finally:
        setup_logging(settings.log_file)


def test_logging_rolls_over_to_new_date_without_restart(tmp_path, monkeypatch):
    day = ["2026-09-22"]
    try:
        with monkeypatch.context() as date_patch:
            date_patch.setattr("app.core.logging._local_day", lambda: day[0])
            first = setup_logging(str(tmp_path / "app.log"))
            log_io_event("request_input", "request-1", "/detect/mmd", {"text": "第一天"})
            day[0] = "2026-09-23"
            log_io_event("request_output", "request-2", "/detect/mmd", {"label": "第二天"})

        second = tmp_path / "app_2026-09-23.log"
        assert first.name == "app_2026-09-22.log"
        assert "第一天" in first.read_text(encoding="utf-8")
        assert "第二天" not in first.read_text(encoding="utf-8")
        assert "第二天" in second.read_text(encoding="utf-8")
        assert second.stat().st_mode & 0o777 == 0o600
    finally:
        setup_logging(settings.log_file)


def test_detect_route_logs_input_and_full_output(monkeypatch, tmp_path):
    records = []

    class StubOrchestrator:
        def detect(self, **_kwargs):
            return {
                "mode": "multimodal",
                "label": "Mixture",
                "summary": "真假信息混合。",
                "confidence": 0.6,
                "raw": {},
            }

    store = ConversationHistoryStore(log_dir=tmp_path / "conversations")
    artifacts = ArtifactStore(store.log_dir)
    image = BytesIO()
    Image.new("RGB", (4, 4), "white").save(image, format="PNG")

    class StubRemoteImageDownloader:
        def download(self, url):
            return DownloadedRemoteImage(image.getvalue(), ".png", "image/png", url)

    monkeypatch.setattr(routes, "artifact_store", artifacts)
    monkeypatch.setattr(routes, "session_service", SessionService(store, artifacts))
    monkeypatch.setattr(routes, "remote_image_downloader", StubRemoteImageDownloader())
    monkeypatch.setattr(routes, "log_io_event", lambda event, request_id, endpoint, payload: records.append((event, request_id, endpoint, payload)))
    app.dependency_overrides[get_detection_orchestrator] = lambda: StubOrchestrator()
    try:
        response = TestClient(app).post(
            "/detect",
            data={
                "task": "multimodal_fact_checking",
                "text": "待核查的新闻文字",
                "image_url": "https://example.com/image.jpg",
                "language": "zh",
            },
        )
    finally:
        app.dependency_overrides.pop(get_detection_orchestrator, None)

    assert response.status_code == 200
    assert [event for event, *_ in records] == ["request_input", "request_output"]
    assert records[0][1] == records[1][1]
    assert records[0][3]["text"] == "待核查的新闻文字"
    assert records[0][3]["image_url"] == "https://example.com/image.jpg"
    assert records[1][3]["response"]["label"] == "Mixture"
    assert records[1][3]["response"]["summary"] == "真假信息混合。"
    json.dumps(records[1][3], ensure_ascii=False)


def test_detect_route_logs_uploaded_image_metadata_without_binary(monkeypatch, tmp_path):
    records = []

    class StubOrchestrator:
        def detect(self, **_kwargs):
            return {"mode": "image_only", "label": "真实图像", "summary": "未见明显异常。", "raw": {}}

    store = ConversationHistoryStore(log_dir=tmp_path / "conversations")
    artifacts = ArtifactStore(store.log_dir)
    monkeypatch.setattr(routes, "artifact_store", artifacts)
    monkeypatch.setattr(routes, "session_service", SessionService(store, artifacts))
    monkeypatch.setattr(routes, "log_io_event", lambda event, request_id, endpoint, payload: records.append((event, payload)))
    app.dependency_overrides[get_detection_orchestrator] = lambda: StubOrchestrator()
    image = BytesIO()
    Image.new("RGB", (4, 4), "white").save(image, format="PNG")
    content = image.getvalue()
    try:
        response = TestClient(app).post(
            "/detect",
            data={"task": "deepfake_detection", "language": "zh"},
            files={"image": ("sample.png", content, "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_detection_orchestrator, None)

    assert response.status_code == 200
    image_record = records[0][1]["images"][0]
    assert image_record["filename"] == "sample.png"
    assert image_record["size_bytes"] == len(content)
    assert len(image_record["sha256"]) == 64
    assert "image_content" not in json.dumps(records[0][1])


def test_detect_route_logs_validation_error(monkeypatch, tmp_path):
    records = []
    store = ConversationHistoryStore(log_dir=tmp_path / "conversations")
    artifacts = ArtifactStore(store.log_dir)
    monkeypatch.setattr(routes, "artifact_store", artifacts)
    monkeypatch.setattr(routes, "session_service", SessionService(store, artifacts))
    monkeypatch.setattr(routes, "log_io_event", lambda event, request_id, endpoint, payload: records.append((event, request_id, payload)))

    response = TestClient(app).post(
        "/detect", data={"task": "multimodal_fact_checking"},
    )

    assert response.status_code == 400
    assert [record[0] for record in records] == ["request_input", "request_error"]
    assert records[0][1] == records[1][1]
    assert records[1][2]["status_code"] == 400


def test_followup_route_logs_input_and_output(monkeypatch):
    records = []

    class StubFollowupApplication:
        def followup(self, **_kwargs):
            return {"mode": "multimodal", "label": "True", "summary": "追问解释。", "raw": {}}

    monkeypatch.setattr(routes, "followup_application", StubFollowupApplication())
    monkeypatch.setattr(routes, "log_io_event", lambda event, request_id, endpoint, payload: records.append((event, request_id, endpoint, payload)))

    response = TestClient(app).post(
        "/followup", data={"question": "为什么？", "conversation_id": "test-log", "language": "zh"},
    )

    assert response.status_code == 200
    assert [record[0] for record in records] == ["request_input", "request_output"]
    assert records[0][1] == records[1][1]
    assert records[0][2] == "/followup"
    assert records[0][3]["question"] == "为什么？"
    assert records[1][3]["response"]["summary"] == "追问解释。"


def test_conversation_followup_uses_path_id_and_logs_endpoint(monkeypatch):
    records = []
    calls = []

    class StubFollowupApplication:
        def followup(self, **kwargs):
            calls.append(kwargs)
            return {
                "mode": "multimodal",
                "label": "True",
                "summary": "基于会话继续解释。",
                "conversation_id": kwargs["conversation_id"],
                "raw": {},
            }

    monkeypatch.setattr(routes, "followup_application", StubFollowupApplication())
    monkeypatch.setattr(
        routes,
        "log_io_event",
        lambda event, request_id, endpoint, payload: records.append(
            (event, request_id, endpoint, payload)
        ),
    )

    response = TestClient(app).post(
        "/conversations/conversation-123/followup",
        data={"question": "证据是什么？", "language": "zh"},
    )

    assert response.status_code == 200
    assert calls[0]["conversation_id"] == "conversation-123"
    assert [record[0] for record in records] == ["request_input", "request_output"]
    assert {record[2] for record in records} == {
        "/conversations/conversation-123/followup"
    }
    assert records[0][3]["conversation_id"] == "conversation-123"
