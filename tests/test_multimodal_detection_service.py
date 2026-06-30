import time

from app.services.multimodal import MultimodalDetectionService, MultimodalVerdictService
from app.services.multimodal.alignment import MultimodalAlignmentService


class StubService:
    def detect(self, *args, **kwargs):
        return {}


def make_service():
    return MultimodalDetectionService(StubService(), StubService(), StubService())


def test_extract_alignment_elements():
    service = make_service()
    analysis = """Label: Mismatch
Matched_Elements:
- 人物数量一致
Mismatched_Elements:
- 文本称地点为北京，图片未体现北京
Reasoning: 图文缺少关键地点对应
"""
    assert service._extract_alignment_label(analysis) == "Inconsistent"
    assert service._extract_list_field(analysis, "Matched_Elements") == ["人物数量一致"]
    assert service._extract_list_field(analysis, "Mismatched_Elements") == ["文本称地点为北京，图片未体现北京"]


def test_alignment_service_normalizes_json_payload():
    service = MultimodalAlignmentService(StubService())
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


def test_alignment_service_rejects_invalid_image_url_without_model_call():
    class ModelFactory:
        def qwen_vision_client(self, timeout=None):
            raise AssertionError("model should not be called for invalid image_url")

    class ImageServiceStub:
        config = type("Config", (), {"vision_verdict_timeout_seconds": 3, "qwen_vision_model": "qwen3-vl-plus"})()
        model_factory = ModelFactory()

    service = MultimodalAlignmentService(ImageServiceStub())

    result = service.evaluate("新闻文本", image_url="blob:http://localhost/image", language="zh")

    assert result["label"] == "Partially_Consistent"
    assert result["reasoning"]


def test_alignment_service_falls_back_when_model_rejects_image_url():
    class Completion:
        def create(self, **kwargs):
            raise RuntimeError("The provided URL does not appear to be valid.")

    class Client:
        chat = type("Chat", (), {"completions": Completion()})()

    class ModelFactory:
        def qwen_vision_client(self, timeout=None):
            return Client()

    class ImageServiceStub:
        config = type("Config", (), {"vision_verdict_timeout_seconds": 3, "qwen_vision_model": "qwen3-vl-plus"})()
        model_factory = ModelFactory()

    service = MultimodalAlignmentService(ImageServiceStub())

    result = service.evaluate("新闻文本", image_url="https://example.com/image.jpg", language="zh")

    assert result["label"] == "Partially_Consistent"
    assert "图文一致性模型调用失败" in result["reasoning"]
    assert result["limitations"]


def test_multimodal_verdict_prioritizes_image_false():
    service = MultimodalVerdictService(evaluator=lambda *_: "")

    result = service.evaluate(
        {"label": "完全真实", "summary": "文本可信"},
        {"label": "完全AI生成", "summary": "图片为AI生成"},
        {"label": "Consistent", "consistency_score": 0.9},
        "zh",
    )

    assert result["label"] == "图像异常"
    assert result["type"] == "图文一致"
    assert result["decision_trace"]["fusion_label"] == "Image_Anomaly"


def test_multimodal_detection_returns_structured_alignment():
    class TextStub:
        def detect(self, *args, **kwargs):
            return {
                "label": "完全真实",
                "summary": "文本可信",
                "confidence": 0.8,
                "text_verdict": {"label": "完全真实", "appraisalResults": "文本可信"},
                "evidence": [{"type": "search_result", "content": "支持文本"}],
                "claims": [{"claim": "北京发生活动"}],
                "user_answer": {"analysis": "文本证据支持。"},
            }

    class ImageStub:
        def detect(self, *args, **kwargs):
            return {
                "label": "完全真实",
                "summary": "图片真实",
                "confidence": 0.75,
                "image_verdict": {"label": "完全真实", "suspicious_regions": []},
                "evidence": [{"type": "visual_tool_evidence", "content": "图片无伪造"}],
                "user_answer": {"analysis": "图片未见伪造。"},
            }

    class AlignmentStub:
        def evaluate(self, *args, **kwargs):
            return {
                "label": "Inconsistent",
                "consistency_score": 0.25,
                "text_elements": {"location": "北京"},
                "visual_elements": {"scene": "海边"},
                "matched_elements": [],
                "mismatched_elements": ["地点"],
                "missing_elements": [],
                "reasoning": "图片地点不支持文本。",
                "limitations": [],
            }

    service = MultimodalDetectionService(
        TextStub(),
        ImageStub(),
        StubService(),
        alignment_service=AlignmentStub(),
        verdict_service=MultimodalVerdictService(evaluator=lambda *_: ""),
    )

    result = service.detect("北京发生活动", image_url="https://example.com/a.jpg", language="zh")

    assert result["label"] == "均无异常"
    assert result["type"] == "图文不一致"
    assert result["alignment_verdict"]["text_elements"]["location"] == "北京"
    assert result["alignment_verdict"]["mismatched_elements"] == ["地点"]
    assert result["user_answer"]["judgment"] == "综合判断为均无异常，图文一致性为图文不一致。"
    assert any(item["type"] == "image_text_alignment" for item in result["evidence"])


def test_multimodal_detection_runs_three_branches_in_parallel():
    class SlowTextStub:
        def detect(self, *args, **kwargs):
            time.sleep(0.18)
            return {"label": "完全真实", "confidence": 0.8}

    class SlowImageStub:
        def detect(self, *args, **kwargs):
            time.sleep(0.18)
            return {"label": "真实图像", "confidence": 0.8}

    class SlowAlignmentStub:
        def evaluate(self, *args, **kwargs):
            time.sleep(0.18)
            return {"label": "Consistent", "consistency_score": 0.8}

    service = MultimodalDetectionService(
        SlowTextStub(),
        SlowImageStub(),
        StubService(),
        alignment_service=SlowAlignmentStub(),
        verdict_service=MultimodalVerdictService(evaluator=lambda *_: ""),
    )

    started = time.perf_counter()
    result = service.detect("北京发生活动", image_url="https://example.com/a.jpg", language="zh")
    elapsed = time.perf_counter() - started

    assert result["label"] == "均无异常"
    assert result["type"] == "图文一致"
    assert elapsed < 0.42
