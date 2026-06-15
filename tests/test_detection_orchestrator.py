from app.schemas.detection import DetectionMode
from app.services.detection_orchestrator import DetectionOrchestrator


class StubService:
    def __init__(self):
        self.calls = []

    def detect(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"mode": "stub"}


def make_orchestrator():
    stub = StubService()
    return DetectionOrchestrator(stub, stub, stub)


def test_auto_routes_text_only_without_image():
    orchestrator = make_orchestrator()
    assert orchestrator.resolve_mode("新闻文本", False, "", DetectionMode.AUTO) == DetectionMode.TEXT_ONLY


def test_auto_routes_generic_image_question_to_image_only():
    orchestrator = make_orchestrator()
    assert orchestrator.resolve_mode("", True, "这张图片是真的假的？", DetectionMode.AUTO) == DetectionMode.IMAGE_ONLY


def test_auto_routes_short_news_title_with_image_to_multimodal():
    orchestrator = make_orchestrator()
    assert orchestrator.resolve_mode("特朗普被捕", True, "", DetectionMode.AUTO) == DetectionMode.MULTIMODAL


def test_explicit_mode_wins():
    orchestrator = make_orchestrator()
    assert orchestrator.resolve_mode("新闻文本", True, "", DetectionMode.IMAGE_ONLY) == DetectionMode.IMAGE_ONLY


def test_text_only_sets_default_question_zh():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(text="新闻文本", language="zh")

    assert result["question"] == "请判断这段文字的真实性。"
    assert orchestrator.text_detection_service.calls[0][0][0] == "新闻文本"


def test_text_only_sets_default_question_en():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(text="A news claim", language="en")

    assert result["question"] == "Please determine the authenticity of this text."


def test_image_only_sets_default_question_zh():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(image_content=b"image bytes", language="zh")

    assert result["question"] == "请判断这张图片是真实图像，伪造图像还是合成图像。"
    assert orchestrator.image_detection_service.calls[0][0][2] == "请判断这张图片是真实图像，伪造图像还是合成图像。"


def test_multimodal_sets_default_question_zh():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(text="新闻文本", image_content=b"image bytes", language="zh")

    assert result["question"] == "请判断这段文字和这张图片是否真实可信，并分析图文是否一致。"
    assert orchestrator.multimodal_detection_service.calls[0][1]["question"] == "请判断这段文字和这张图片是否真实可信，并分析图文是否一致。"


def test_explicit_question_is_preserved():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(text="新闻文本", question="请重点判断标题是否真实", language="zh")

    assert result["question"] == "请重点判断标题是否真实"
