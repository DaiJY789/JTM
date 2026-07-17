import json

from app.services.comprehensive_fusion import ComprehensiveFusionService
from app.services.crossmodal import CrossmodalConsistencyPipeline


class StubImageService:
    config = type(
        "Config",
        (),
        {"vision_verdict_timeout_seconds": 3, "qwen_vision_model": "qwen3-vl-plus"},
    )()

    def build_image_content_part(self, image_content=None, image_url=None):
        if image_content or str(image_url or "").startswith("https://"):
            return {"type": "image_url", "image_url": {"url": image_url or "data:image/png;base64,AA=="}}
        return None


def test_crossmodal_pipeline_normalizes_consistency_payload():
    service = CrossmodalConsistencyPipeline(StubImageService())
    payload = {
        "label": "Inconsistent",
        "consistency_score": 0.2,
        "text_elements": {"location": "北京", "event": "暴雨"},
        "visual_elements": {"scene": "晴天街景"},
        "matched_elements": ["城市街景"],
        "mismatched_elements": ["天气"],
        "missing_elements": ["新闻事件"],
        "reasoning": "图片不能支持暴雨新闻。",
        "limitations": ["无法确认拍摄时间"],
    }

    result = service._normalize_result(payload, "", "zh")

    assert result["label"] == "Inconsistent"
    assert result["consistency_score"] == 0.2
    assert result["text_elements"]["location"] == "北京"
    assert result["visual_elements"]["scene"] == "晴天街景"
    assert result["mismatched_elements"] == ["天气"]


def test_crossmodal_prompt_is_limited_to_consistency_check():
    service = CrossmodalConsistencyPipeline(StubImageService())

    prompt = service._build_prompt("新闻文本", "图文一致吗？", "zh")

    assert "唯一任务" in prompt
    assert "不要判断文本事实真假" in prompt
    assert "后续综合融合模块" in prompt


def test_crossmodal_pipeline_rejects_invalid_image_without_model_call():
    class ModelFactory:
        def qwen_vision_client(self, timeout=None):
            raise AssertionError("model should not be called")

    image_service = StubImageService()
    image_service.model_factory = ModelFactory()
    service = CrossmodalConsistencyPipeline(image_service)

    result = service.detect("新闻文本", image_url="blob:http://localhost/image", language="zh")

    assert result["label"] == "Partially_Consistent"
    assert result["reasoning"]


def test_crossmodal_pipeline_falls_back_when_model_call_fails():
    class Completion:
        def create(self, **kwargs):
            raise RuntimeError("invalid image")

    class Client:
        chat = type("Chat", (), {"completions": Completion()})()

    class ModelFactory:
        def qwen_vision_client(self, timeout=None):
            return Client()

    image_service = StubImageService()
    image_service.model_factory = ModelFactory()
    service = CrossmodalConsistencyPipeline(image_service)

    result = service.detect("新闻文本", image_url="https://example.com/image.jpg", language="zh")

    assert result["label"] == "Partially_Consistent"
    assert "图文一致性模型调用失败" in result["reasoning"]


def test_comprehensive_fusion_uses_multimodal_report_payload():
    model_report = {
        "label": "图像异常",
        "type": "图文不一致",
        "confidence": 0.88,
        "summary": "图片存在异常且不能支持新闻文本。",
        "risk_level": "high",
        "user_answer": {
            "judgment": "综合判断存在较高风险。",
            "reason_analysis": "文本证据有限，图片异常且图文冲突。",
            "text_analysis": "文本暂未发现明确异常。",
            "visual_analysis": "图片存在合成迹象。",
            "crossmodal_analysis": "图片场景与文本地点冲突。",
            "notes": [],
        },
        "limitations": ["无法确认原始发布来源"],
    }
    calls = []

    def evaluator(prompt, image_content, image_url, language):
        calls.append((prompt, image_content, image_url, language))
        return json.dumps(model_report, ensure_ascii=False)

    service = ComprehensiveFusionService(StubImageService(), evaluator=evaluator)
    result = service.generate_report(
        text="北京发生暴雨",
        image_content=b"image",
        image_url=None,
        question="是否可信？",
        textual_result={"label": "Mostly_True", "summary": "文本有证据支持", "confidence": 0.7},
        visual_result={"label": "合成图像", "summary": "发现合成迹象", "confidence": 0.9},
        crossmodal_result={"label": "Inconsistent", "consistency_score": 0.2},
        language="zh",
    )

    assert result["label"] == "图像异常"
    assert result["type"] == "图文不一致"
    assert result["summary"] == model_report["summary"]
    assert result["decision_trace"]["model_report_used"] is True
    assert calls[0][1] == b"image"
    assert "直接重新观察原图" in calls[0][0]


def test_comprehensive_fusion_has_deterministic_fallback():
    service = ComprehensiveFusionService(StubImageService(), evaluator=lambda *_: "")

    result = service.generate_report(
        text="新闻文本",
        image_content=b"image",
        image_url=None,
        question=None,
        textual_result={"label": "Completely_False", "confidence": 0.8},
        visual_result={"label": "真实图像", "confidence": 0.8},
        crossmodal_result={"label": "Consistent", "consistency_score": 0.8},
        language="zh",
    )

    assert result["label"] == "文本异常"
    assert result["type"] == "图文一致"
    assert result["decision_trace"]["model_report_used"] is False


def test_comprehensive_fusion_calls_vision_model_with_original_image():
    calls = []
    payload = {
        "label": "均无异常",
        "type": "图文一致",
        "confidence": 0.82,
        "summary": "综合判断暂未发现异常。",
        "risk_level": "low",
        "user_answer": {"judgment": "整体可信。", "reason_analysis": "三路结果相互支持。", "notes": []},
        "limitations": [],
    }

    class Completion:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = type("Message", (), {"content": json.dumps(payload, ensure_ascii=False)})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class ModelFactory:
        def qwen_vision_client(self, timeout=None):
            return type("Client", (), {"chat": type("Chat", (), {"completions": Completion()})()})()

    image_service = StubImageService()
    image_service.model_factory = ModelFactory()
    service = ComprehensiveFusionService(image_service)

    result = service.generate_report(
        text="新闻文本",
        image_content=b"original-image",
        image_url=None,
        question=None,
        textual_result={"label": "Mostly_True"},
        visual_result={"label": "真实图像"},
        crossmodal_result={"label": "Consistent"},
        language="zh",
    )

    assert result["summary"] == payload["summary"]
    user_content = calls[0]["messages"][1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[1]["type"] == "text"
