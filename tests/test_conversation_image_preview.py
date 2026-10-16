from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.main import app
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.conversation_image_preview import (
    ConversationImagePreviewError,
    ConversationImagePreviewService,
)
from app.services.session_service import SessionService


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), "teal").save(output, format="PNG")
    return output.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.content


def make_service(tmp_path, requester, host_validator=lambda _host: True):
    store = ConversationHistoryStore(log_dir=tmp_path / "conversations")
    artifacts = ArtifactStore(store.log_dir)
    session = SessionService(store, artifacts)
    session.append_detection(
        "preview-test",
        {
            "text": "待核查文字",
            "image_url": "https://images.example/news.png",
            "image_urls": ["https://images.example/news.png"],
            "image_artifacts": [
                {
                    "source_url": "https://images.example/news.png",
                    "role": "remote_image",
                }
            ],
        },
        {"mode": "multimodal", "label": "True", "summary": "完成"},
    )
    return ConversationImagePreviewService(
        session,
        artifacts,
        requester=requester,
        host_validator=host_validator,
    )


def test_remote_history_image_is_downloaded_and_cached(tmp_path):
    calls = []

    def requester(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(png_bytes())

    service = make_service(tmp_path, requester)
    first = service.get("preview-test", 0)
    second = service.get("preview-test", 0)

    assert first.path == second.path
    assert first.path.is_file()
    assert first.media_type == "image/png"
    assert first.path.read_bytes() == png_bytes()
    assert len(calls) == 1
    assert calls[0][1]["allow_redirects"] is False


def test_remote_history_image_must_belong_to_conversation(tmp_path):
    service = make_service(tmp_path, lambda *_args, **_kwargs: FakeResponse(png_bytes()))

    with pytest.raises(ConversationImagePreviewError, match="未找到"):
        service.get("preview-test", 1)


def test_remote_history_image_rejects_non_public_host(tmp_path):
    called = False

    def requester(*_args, **_kwargs):
        nonlocal called
        called = True
        return FakeResponse(png_bytes())

    service = make_service(tmp_path, requester, host_validator=lambda _host: False)

    with pytest.raises(ConversationImagePreviewError, match="私有网络"):
        service.get("preview-test", 0)
    assert called is False


def test_conversation_image_endpoint_serves_cached_remote_preview(tmp_path, monkeypatch):
    store = ConversationHistoryStore(log_dir=tmp_path / "conversations")
    artifacts = ArtifactStore(store.log_dir)
    session = SessionService(store, artifacts)
    source_url = "https://images.example/cached.png"
    session.append_detection(
        "cached-preview",
        {
            "image_url": source_url,
            "image_artifacts": [{"source_url": source_url, "role": "remote_image"}],
        },
        {"mode": "image_only", "label": "真实图像", "summary": "完成"},
    )
    artifacts.save_remote_preview("cached-preview", source_url, png_bytes(), ".png")
    monkeypatch.setattr(routes, "session_service", session)
    monkeypatch.setattr(routes, "artifact_store", artifacts)

    response = TestClient(app).get("/conversations/cached-preview/images/0")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, max-age=86400"
    assert response.content == png_bytes()


def test_remote_history_image_reuses_submit_time_conversation_copy(tmp_path):
    store = ConversationHistoryStore(log_dir=tmp_path / "conversations")
    artifacts = ArtifactStore(store.log_dir)
    session = SessionService(store, artifacts)
    source_url = "https://images.example/persisted.png"
    artifact = artifacts.save_image(
        "persisted-preview",
        png_bytes(),
        filename="persisted.png",
        content_type="image/png",
        source_url=source_url,
    )
    session.append_detection(
        "persisted-preview",
        {
            "image_url": source_url,
            "image_urls": [source_url],
            "image_assets": [artifacts.artifact_to_legacy_asset(artifact)],
            "image_artifacts": [artifact.model_dump(mode="json")],
        },
        {"mode": "image_only", "label": "真实图像", "summary": "完成"},
    )

    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("submit-time image copy should be reused")

    preview = ConversationImagePreviewService(
        session,
        artifacts,
        requester=unexpected_download,
        host_validator=lambda _host: True,
    ).get("persisted-preview", 0)

    assert preview.path == artifacts.resolve(
        "persisted-preview",
        f"images/{artifact.metadata['stored_filename']}",
    )
    assert preview.path.read_bytes() == png_bytes()
