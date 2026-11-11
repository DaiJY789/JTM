import json
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_detection_orchestrator
from app.api import routes
from app.main import app
from app.schemas.detection import DetectionResponse
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.followup_application import should_rerun_detection
from app.services.remote_image_downloader import DownloadedRemoteImage
from app.services.session_service import SessionService
from app.services.webpage_extraction import WebpageExtractionResult


def valid_png_bytes(color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(output, format="PNG")
    return output.getvalue()


class StubRemoteImageDownloader:
    def download(self, url):
        return DownloadedRemoteImage(
            valid_png_bytes("white"),
            ".png",
            "image/png",
            url,
        )


def test_detection_response_accepts_multimodal_payload():
    response = DetectionResponse.model_validate(
        {
            "mode": "multimodal",
            "type": "事实核查",
            "label": "Cross_Modal_Mismatch",
            "summary": "图文内容存在不一致。",
            "confidence": 0.8,
            "text_verdict": {
                "label": "Mostly_True",
                "appraisalResults": "主要事实有证据支持。",
                "claims": [
                    {
                        "claim": "新闻称图片展示了某地事件。",
                        "type": "event",
                        "entities": ["某地"],
                        "search_query": "某地 事件 图片",
                    }
                ],
            },
            "image_verdict": {
                "label": "Real",
                "deepfake_risk": "low",
                "manipulation_risk": "low",
                "visual_description": "图片展示了一处街景。",
                "evidence": ["未见明显生成痕迹。"],
                "reasoning": "视觉线索较自然。",
            },
            "crossmodal_verdict": {
                "label": "Mismatch",
                "text_elements": {"location": "某地", "event": "新闻文本描述的事件"},
                "visual_elements": {"scene": "街景", "location_clues": ["路牌"]},
                "matched_elements": ["事件类型"],
                "mismatched_elements": ["地点"],
                "missing_elements": ["具体时间"],
                "reasoning": "图片地点与文本地点不一致。",
                "limitations": ["无法确认拍摄时间"],
            },
            "evidence": [{"type": "visual_clue", "content": "地点标识不一致。"}],
            "claims": [{"claim": "新闻称图片展示了某地事件。", "type": "event"}],
            "report": {"risk_level": "high", "summary": "综合融合报告。"},
            "raw": {},
        }
    )

    assert response.mode == "multimodal"
    assert response.type == "事实核查"
    assert response.text_verdict is not None
    assert response.image_verdict is not None
    assert response.crossmodal_verdict is not None
    assert response.crossmodal_verdict.text_elements["location"] == "某地"
    assert response.report["risk_level"] == "high"
    assert response.claims[0].claim == "新闻称图片展示了某地事件。"


def test_detect_openapi_documents_request_examples():
    operation = app.openapi()["paths"]["/detect"]["post"]
    multipart = operation["requestBody"]["content"]["multipart/form-data"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/DetectionResponse"
    )
    assert "Unified endpoint" in operation["description"]
    assert set(multipart["examples"]) >= {"image_only", "multimodal", "webpage"}
    assert "text_only" not in multipart["examples"]


def test_split_detection_openapi_exposes_task_specific_fields():
    schema = app.openapi()
    mmd_body = schema["paths"]["/detect/mmd"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    deepfake_body = schema["paths"]["/detect/deepfake"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    mmd_fields = schema["components"]["schemas"][mmd_body["$ref"].rsplit("/", 1)[-1]]["properties"]
    deepfake_fields = schema["components"]["schemas"][deepfake_body["$ref"].rsplit("/", 1)[-1]]["properties"]

    assert {"text", "images", "image_urls", "webpage_url"} <= mmd_fields.keys()
    assert {"image", "image_url", "webpage_url"} <= deepfake_fields.keys()
    assert {"task", "text", "images", "image_urls"}.isdisjoint(deepfake_fields)
    assert "task" not in mmd_fields


def test_conversation_followup_openapi_uses_conversation_path_parameter():
    operation = app.openapi()["paths"]["/conversations/{conversation_id}/followup"]["post"]
    parameters = {item["name"]: item for item in operation["parameters"]}
    body_schema = operation["requestBody"]["content"]["application/x-www-form-urlencoded"]["schema"]
    body_fields = app.openapi()["components"]["schemas"][body_schema["$ref"].rsplit("/", 1)[-1]]["properties"]

    assert parameters["conversation_id"]["in"] == "path"
    assert parameters["conversation_id"]["required"] is True
    assert set(body_fields) == {"question", "language"}
    assert "/followup" in app.openapi()["paths"]


def test_webpage_extraction_endpoint_returns_preview_without_detection(monkeypatch):
    class StubWebpageExtractor:
        def extract(self, url):
            return WebpageExtractionResult(
                url=url,
                final_url="https://example.com/final-news",
                title="网页新闻标题",
                text="网页新闻正文。",
                image_url="https://example.com/one.jpg",
                images=["https://example.com/one.jpg", "https://example.com/two.jpg"],
                metadata={"image_count": 2},
            )

    monkeypatch.setattr(routes, "webpage_extraction_service", StubWebpageExtractor())
    response = TestClient(app).post(
        "/webpages/extract",
        data={"webpage_url": "https://example.com/news"},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "网页新闻标题"
    assert response.json()["text"] == "网页新闻正文。"
    assert len(response.json()["images"]) == 2


def test_split_detection_routes_dispatch_to_matching_tasks(monkeypatch, tmp_path):
    from app.services.artifact_store import ArtifactStore
    from app.services.conversation_history import ConversationHistoryStore
    from app.services.session_service import SessionService

    calls = []

    class StubOrchestrator:
        def detect(self, **kwargs):
            calls.append(kwargs)
            return {
                "mode": "image_only" if kwargs["task_type"].value == "deepfake_detection" else "multimodal",
                "label": "Real" if kwargs["task_type"].value == "deepfake_detection" else "True",
                "summary": "检测完成。",
                "raw": {},
            }

    store = ConversationHistoryStore(log_dir=tmp_path / "conversations")
    artifacts = ArtifactStore(store.log_dir)
    logged_endpoints = []
    monkeypatch.setattr(routes, "artifact_store", artifacts)
    monkeypatch.setattr(routes, "session_service", SessionService(store, artifacts))
    monkeypatch.setattr(routes, "log_io_event", lambda event, request_id, endpoint, payload: logged_endpoints.append((event, endpoint)))
    app.dependency_overrides[get_detection_orchestrator] = lambda: StubOrchestrator()
    try:
        client = TestClient(app)
        mmd_response = client.post(
            "/detect/mmd",
            data={"text": "这是一条包含两张图片的新闻。"},
            files=[
                ("images", ("one.png", valid_png_bytes("red"), "image/png")),
                ("images", ("two.png", valid_png_bytes("blue"), "image/png")),
            ],
        )
        deepfake_response = client.post(
            "/detect/deepfake",
            files={"image": ("sample.png", valid_png_bytes("green"), "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_detection_orchestrator, None)

    assert mmd_response.status_code == 200
    assert deepfake_response.status_code == 200
    assert calls[0]["task_type"].value == "multimodal_fact_checking"
    assert len(calls[0]["image_contents"]) == 2
    assert calls[1]["task_type"].value == "deepfake_detection"
    assert len(calls[1]["image_contents"]) == 1
    assert logged_endpoints == [
        ("request_input", "/detect/mmd"),
        ("request_output", "/detect/mmd"),
        ("request_input", "/detect/deepfake"),
        ("request_output", "/detect/deepfake"),
    ]


@pytest.mark.parametrize(
    ("endpoint", "selection", "expected"),
    [
        ("/detect", None, ["https://example.com/image.jpg", "https://example.com/other.jpg"]),
        ("/detect/mmd", ["https://example.com/other.jpg"], ["https://example.com/other.jpg"]),
        ("/detect/mmd", [], []),
        ("/detect/deepfake", ["https://example.com/other.jpg"], ["https://example.com/other.jpg"]),
    ],
)
def test_detect_accepts_webpage_url_and_passes_extracted_multimodal_input(monkeypatch, tmp_path, endpoint, selection, expected):
    first_image = "https://example.com/image.jpg" if selection is None else "http://127.0.0.1/private.jpg"

    class StubWebpageExtractor:
        def extract(self, url):
            return WebpageExtractionResult(
                url=url,
                final_url="https://example.com/news",
                title="网页新闻标题",
                text="网页正文描述了一张配图。",
                image_url=first_image,
                images=[first_image, "https://example.com/other.jpg"],
                metadata={"text_length": 11, "image_count": 2},
            )

    class StubOrchestrator:
        def __init__(self):
            self.calls = []

        def detect(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "mode": "multimodal",
                "type": "图文一致",
                "label": "均无异常",
                "summary": "网页图文信息暂未发现明显异常。",
                "confidence": 0.7,
                "evidence": [],
                "claims": [],
                "raw": {},
                "user_answer": {"judgment": "网页图文信息暂未发现明显异常。"},
            }

    stub_orchestrator = StubOrchestrator()
    downloaded_urls = []

    class TrackingImageDownloader(StubRemoteImageDownloader):
        def download(self, url):
            downloaded_urls.append(url)
            assert url != "http://127.0.0.1/private.jpg"
            return super().download(url)

    artifact_store = ArtifactStore(tmp_path)
    monkeypatch.setattr(routes, "artifact_store", artifact_store)
    monkeypatch.setattr(routes, "session_service", SessionService(ConversationHistoryStore(log_dir=tmp_path), artifact_store))
    monkeypatch.setattr(routes, "webpage_extraction_service", StubWebpageExtractor())
    monkeypatch.setattr(routes, "remote_image_downloader", TrackingImageDownloader())
    app.dependency_overrides[get_detection_orchestrator] = lambda: stub_orchestrator

    try:
        client = TestClient(app)
        data = {"language": "zh", "webpage_url": "https://example.com/news", "conversation_id": "test-webpage-detect"}
        if selection is not None:
            data["webpage_image_urls"] = json.dumps(selection)
        response = client.post(endpoint, data=data)
    finally:
        app.dependency_overrides.pop(get_detection_orchestrator, None)

    assert response.status_code == 200
    payload = response.json()
    call = stub_orchestrator.calls[0]
    assert call["text"].startswith("网页新闻标题")
    assert call["image_url"] is None
    assert len(call["image_contents"]) == len(expected)
    assert call["image_urls"] == []
    assert downloaded_urls == expected
    assert payload["raw"]["webpage"]["final_url"] == "https://example.com/news"
    assert payload["raw"]["webpage"]["text"] == "网页正文描述了一张配图。"
    assert payload["raw"]["webpage"]["images"] == expected
    assets = payload["conversation_history"][0]["meta"]["image_assets"]
    assert [image["source_url"] for image in assets] == expected
    assert payload["evidence"][0]["type"] == "webpage_source"


def test_detect_accepts_multiple_uploaded_images(monkeypatch):
    class StubOrchestrator:
        def __init__(self):
            self.calls = []

        def detect(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "mode": "multimodal",
                "type": "multimodal_misinformation",
                "label": "True",
                "summary": "多张图片已进入检测。",
                "confidence": 0.7,
                "evidence": [],
                "claims": [],
                "raw": {},
            }

    stub = StubOrchestrator()
    monkeypatch.setattr(routes, "remote_image_downloader", StubRemoteImageDownloader())
    app.dependency_overrides[get_detection_orchestrator] = lambda: stub
    try:
        response = TestClient(app).post(
            "/detect",
            data={
                "task": "multimodal_fact_checking",
                "text": "这是一条包含多张配图的新闻。",
                "conversation_id": "test-multiple-image-upload",
            },
            files=[
                ("images", ("one.png", valid_png_bytes("red"), "image/png")),
                ("images", ("two.png", valid_png_bytes("blue"), "image/png")),
            ],
        )
    finally:
        app.dependency_overrides.pop(get_detection_orchestrator, None)

    assert response.status_code == 200
    call = stub.calls[0]
    assert len(call["image_contents"]) == 2
    assert call["image_urls"] == []
    assert call["task_type"].value == "multimodal_fact_checking"


def test_detect_accepts_multiple_remote_image_urls(monkeypatch):
    class StubOrchestrator:
        def __init__(self):
            self.calls = []

        def detect(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "mode": "multimodal",
                "type": "multimodal_misinformation",
                "label": "Mostly True",
                "summary": "远程多图已进入检测。",
                "confidence": 0.6,
                "evidence": [],
                "claims": [],
                "raw": {},
            }

    stub = StubOrchestrator()
    monkeypatch.setattr(routes, "remote_image_downloader", StubRemoteImageDownloader())
    app.dependency_overrides[get_detection_orchestrator] = lambda: stub
    try:
        response = TestClient(app).post(
            "/detect",
            data={
                "task": "multimodal_fact_checking",
                "text": "这是一条包含远程多图的新闻。",
                "conversation_id": "test-multiple-image-urls",
                "image_urls": [
                    "https://example.com/one.jpg",
                    "https://example.com/two.jpg",
                ],
            },
        )
    finally:
        app.dependency_overrides.pop(get_detection_orchestrator, None)

    assert response.status_code == 200
    call = stub.calls[0]
    assert len(call["image_contents"]) == 2
    assert call["image_urls"] == []
    assert call["task_type"].value == "multimodal_fact_checking"


def test_conversation_endpoints_return_history_shape():
    client = TestClient(app)

    snapshot = client.get("/conversations/test-api-conversation").json()
    listing = client.get("/conversations").json()

    assert snapshot["conversation_id"] == "test-api-conversation"
    assert snapshot["conversation_history"] == []
    assert snapshot["conversation"]["history_count"] == 0
    assert not any(item["conversation_id"] == "test-api-conversation" for item in listing["conversations"])


def test_followup_rerun_intent_detection():
    assert should_rerun_detection("不要依赖已有结果，重新检测一下")
    assert should_rerun_detection("please rerun the tools from scratch")
    assert not should_rerun_detection("为什么你认为这张图是真的？")
    assert not should_rerun_detection("不用重新检测，只解释一下原因")
