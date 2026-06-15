from fastapi.testclient import TestClient

from app.main import app
from app.api.detect_routes import should_rerun_detection
from app.schemas.detection import DetectionResponse


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
            "alignment_verdict": {
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
            "raw": {},
        }
    )

    assert response.mode == "multimodal"
    assert response.type == "事实核查"
    assert response.text_verdict is not None
    assert response.image_verdict is not None
    assert response.alignment_verdict is not None
    assert response.alignment_verdict.text_elements["location"] == "某地"
    assert response.claims[0].claim == "新闻称图片展示了某地事件。"


def test_detect_openapi_documents_request_examples():
    operation = app.openapi()["paths"]["/detect"]["post"]
    multipart = operation["requestBody"]["content"]["multipart/form-data"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/DetectionResponse"
    )
    assert "Unified endpoint" in operation["description"]
    assert set(multipart["examples"]) >= {"text_only", "image_only", "multimodal"}


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
