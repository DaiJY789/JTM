from io import BytesIO
import time

from app.services.vision.image import ImageDetectionService
from app.services.vision.forensic_pipeline import ImageForensicPipeline
from app.prompts.image import IMAGE_FORENSIC_PROMPT_EN
from config import Settings

from PIL import Image


class StubImageService:
    config = Settings(vision_output_dir="outputs/test_vision")


def test_extract_image_label():
    service = ImageDetectionService(StubImageService())
    assert service._extract_label("Label: Deepfake_Suspected\nReasoning: ...") == "合成图像"
    assert service._extract_label("Label: 局部篡改\nReasoning: ...") == "伪造图像"
    assert service._extract_label("Label: AI_Manipulated\nReasoning: ...") == "合成图像"


def test_extract_image_field():
    service = ImageDetectionService(StubImageService())
    assert service._extract_field("Deepfake_Risk: high\nReasoning: ...", "Deepfake_Risk") == "high"


def test_extract_image_evidence_list():
    service = ImageDetectionService(StubImageService())
    analysis = """Label: Manipulated
Evidence:
- 光照方向不一致
- 面部边缘模糊
Reasoning: 存在可疑视觉线索
"""
    assert service._extract_list_field(analysis, "Evidence") == ["光照方向不一致", "面部边缘模糊"]


def test_materialize_image_preserves_original_jpeg_bytes(tmp_path):
    image_service = StubImageService()
    image_service.config = Settings(vision_output_dir=str(tmp_path / "vision"))
    service = ImageDetectionService(image_service)
    buffer = BytesIO()
    Image.new("RGB", (12, 12), color="white").save(buffer, format="JPEG")
    content = buffer.getvalue()

    path = service._materialize_image(content, None)

    assert path.endswith(".jpg")
    assert open(path, "rb").read() == content


def test_image_error_result_is_cannot_determine(tmp_path):
    image_service = StubImageService()
    image_service.config = Settings(vision_output_dir=str(tmp_path / "vision"))
    service = ImageDetectionService(image_service)

    result = service.detect(image_content=b"not an image", language="zh")

    assert result["label"] == "Cannot_Determine"
    assert result["image_verdict"]["label"] == "Cannot_Determine"


class StubVisionTool:
    def __init__(self, tool_name):
        self.tool_name = tool_name

    def __call__(self, args_str):
        if self.tool_name == "Spectrum_Grid_Analyzer":
            return {
                "success": True,
                "status": "success",
                "description": "spectrum ok",
                "metadata": {"tool_name": self.tool_name, "execution_time_ms": 1},
                "result": {
                    "forensics_report": {"method": "DFT", "summary": "freq evidence"},
                    "evidence_scores": {"peak_density_score": 0.2},
                    "artifacts": {},
                },
            }
        return {
            "success": True,
            "status": "success",
            "description": "noise ok",
            "metadata": {"tool_name": self.tool_name, "execution_time_ms": 1},
            "result": {
                "forensics_report": {
                    "method": "Noise",
                    "summary": "noise evidence",
                    "top_suspicious_regions": [{"region": "x:10-30, y:20-50", "score": 0.8}],
                },
                "evidence_scores": {"noise_inconsistency_score": 0.82},
                "artifacts": {},
            },
        }


class SlowVisionTool(StubVisionTool):
    def __init__(self, tool_name, delay=0.2):
        super().__init__(tool_name)
        self.delay = delay

    def __call__(self, args_str):
        time.sleep(self.delay)
        return super().__call__(args_str)


def test_image_forensic_pipeline_runs_multi_round_and_marks_region(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 80), color="white").save(image_path)
    responses = iter(
        [
            '{"action": "call_tool", "tool": "Noise_Analyzer", "parameters": {}, "why": "核验噪声不一致"}',
            '{"label": "伪造图像", "probability": 0.82, "confidence": 0.76, '
            '"forgery_types": ["splicing"], "suspicious_regions": [{"region": "10,20,30,50", "score": 0.8, "reason": "噪声异常"}], '
            '"tool_evidence": [], "limitations": [], "final_summary": "检测到伪造。"}',
        ]
    )
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision"), vision_max_rounds=3),
        tools={
            "Spectrum_Grid_Analyzer": StubVisionTool("Spectrum_Grid_Analyzer"),
            "Noise_Analyzer": StubVisionTool("Noise_Analyzer"),
        },
        evaluator=lambda prompt: next(responses),
    )

    result = pipeline.detect(str(image_path), language="zh")

    assert result["label"] == "伪造图像"
    assert "verdict" not in result
    assert "consistency_check" not in result
    assert "tool_evidence" in result
    assert result["suspicious_regions"][0]["region"] == "10,20,30,50"
    assert result["artifacts"]["annotated_image_path"]
    annotated = Image.open(result["artifacts"]["annotated_image_path"]).convert("RGB")
    assert annotated.getpixel((10, 20)) == (255, 0, 0)
    assert annotated.getpixel((30, 50)) == (255, 0, 0)
    assert len(result["tool_calls"]) == 2


def test_image_forensic_pipeline_runs_initial_tools_in_parallel(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 80), color="white").save(image_path)
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision"), vision_max_rounds=2),
        tools={
            "SRM_Filter": SlowVisionTool("SRM_Filter", delay=0.2),
            "Spectrum_Grid_Analyzer": SlowVisionTool("Spectrum_Grid_Analyzer", delay=0.2),
        },
        evaluator=lambda prompt: '{"label": "真实图像", "probability": 0.1, "confidence": 0.7, "forgery_types": ["无"], "final_summary": "图片更可能真实。"}',
    )

    started = time.perf_counter()
    result = pipeline.detect(str(image_path), language="zh")
    elapsed = time.perf_counter() - started

    assert [item["tool"] for item in result["tool_calls"]] == ["SRM_Filter", "Spectrum_Grid_Analyzer"]
    assert elapsed < 0.35


def test_image_forensic_pipeline_parses_structured_regions():
    pipeline = ImageForensicPipeline.__new__(ImageForensicPipeline)

    assert pipeline._parse_region({"x0": 10, "y0": 20, "x1": 30, "y1": 50}) == "10,20,30,50"
    assert pipeline._parse_region({"x": 10, "y": 20, "w": 20, "h": 30}) == "10,20,30,50"
    assert pipeline._parse_region({"region": {"x0": 10, "y0": 20, "x1": 30, "y1": 50}}) == "10,20,30,50"


def test_image_forensic_pipeline_uses_cfa_score_in_fallback_probability():
    pipeline = ImageForensicPipeline.__new__(ImageForensicPipeline)
    tool_calls = [
        {
            "tool": "CFA_Validator",
            "summary": {
                "evidence_scores": {"cfa_inconsistency_score": 0.73},
            },
        }
    ]

    assert pipeline._tamper_score(tool_calls) == 0.73


def test_image_forensic_pipeline_ignores_cfa_score_for_non_camera_source():
    pipeline = ImageForensicPipeline.__new__(ImageForensicPipeline)
    tool_calls = [
        {
            "tool": "CFA_Validator",
            "summary": {
                "evidence_scores": {"cfa_inconsistency_score": 0.73},
            },
        }
    ]

    assert pipeline._tamper_score(tool_calls, {"likely_camera_source": False}) == 0.0
    assert pipeline._tamper_score(tool_calls, {"likely_camera_source": True}) == 0.73


def test_image_forensic_pipeline_gates_conditional_tools():
    pipeline = ImageForensicPipeline.__new__(ImageForensicPipeline)
    pipeline.tool_names = {"Noise_Analyzer", "Facial_Biological_Auditor", "PatchDense_Scanner", "CFA_Validator"}
    tool_calls = []
    image_context = {"likely_camera_source": False, "face_hint": False, "copy_move_hint": False}

    assert pipeline._select_next_tool({"tool": "Facial_Biological_Auditor", "why": "check details"}, tool_calls, image_context) == "Noise_Analyzer"
    assert pipeline._select_next_tool({"tool": "Facial_Biological_Auditor", "why": "visible face in the image"}, tool_calls, image_context) == "Facial_Biological_Auditor"
    assert pipeline._select_next_tool({"tool": "PatchDense_Scanner", "why": "suspected copy-move cloned region"}, tool_calls, image_context) == "PatchDense_Scanner"
    assert pipeline._select_next_tool({"tool": "CFA_Validator", "why": "check CFA"}, tool_calls, image_context) == "Noise_Analyzer"
    assert pipeline._select_next_tool({"tool": "CFA_Validator", "why": "check CFA"}, tool_calls, {"likely_camera_source": True}) == "CFA_Validator"


def test_image_forensic_pipeline_selects_first_tool_by_context():
    pipeline = ImageForensicPipeline.__new__(ImageForensicPipeline)
    pipeline.tool_names = {"Spectrum_Grid_Analyzer", "SRM_Filter", "PatchDense_Scanner"}

    assert pipeline._choose_first_tool({"copy_move_hint": True}) == "PatchDense_Scanner"
    assert pipeline._choose_first_tool({"face_hint": True}) == "Spectrum_Grid_Analyzer"
    assert pipeline._choose_first_tool({}) == "SRM_Filter"


def test_image_forensic_pipeline_calibrates_model_probability_and_confidence():
    pipeline = ImageForensicPipeline.__new__(ImageForensicPipeline)
    tool_calls = [
        {
            "tool": "Spectrum_Grid_Analyzer",
            "summary": {
                "success": True,
                "status": "success",
                "evidence_scores": {"peak_density_score": 0.1},
            },
        }
    ]

    probability = pipeline._calibrate_probability("真实图像", 0.9, tool_calls, [], {})
    confidence = pipeline._calibrate_confidence("合成图像", 0.95, tool_calls, [])

    assert probability <= 0.45
    assert confidence <= 0.72


def test_image_forensic_pipeline_prompt_exposes_gate_context(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision")),
        tools={"Spectrum_Grid_Analyzer": StubVisionTool("Spectrum_Grid_Analyzer"), "CFA_Validator": StubVisionTool("CFA_Validator")},
        evaluator=lambda prompt: '{"label": "真实图像"}',
    )
    context = pipeline._build_image_context(str(image_path), question="这张图片是真的假的？")
    prompt = pipeline._build_interaction_prompt([], "这张图片是真的假的？", "zh", False, context)

    assert "条件工具规则" in prompt
    assert "CFA_Validator" not in pipeline._eligible_remaining_tools([], context)


def test_english_image_prompt_uses_english_labels_only():
    assert "Real_Image|Forged_Image|Synthetic_Image" in IMAGE_FORENSIC_PROMPT_EN
    assert "完全真实" not in IMAGE_FORENSIC_PROMPT_EN
    assert "局部篡改" not in IMAGE_FORENSIC_PROMPT_EN
    assert "AI操纵" not in IMAGE_FORENSIC_PROMPT_EN
    assert "存在伪造" not in IMAGE_FORENSIC_PROMPT_EN
    assert "AI生成" not in IMAGE_FORENSIC_PROMPT_EN
    assert "完全AI生成" not in IMAGE_FORENSIC_PROMPT_EN


def test_image_forensic_pipeline_keeps_regions_for_synthetic_image(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 80), color="white").save(image_path)
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision"), vision_max_rounds=1),
        tools={"Spectrum_Grid_Analyzer": StubVisionTool("Spectrum_Grid_Analyzer")},
        evaluator=lambda prompt: (
            '{"label": "合成图像", "probability": 0.8, "confidence": 0.7, '
            '"forgery_types": ["ai_generated"], '
            '"suspicious_regions": [{"region": "10,20,30,50", "score": 0.8, "reason": "模型误给区域"}], '
            '"tool_evidence": [], "limitations": [], "final_summary": "整图为AI生成。"}'
        ),
    )

    result = pipeline.detect(str(image_path), language="zh")

    assert result["label"] == "合成图像"
    assert result["suspicious_regions"][0]["region"] == "10,20,30,50"
    assert result["forgery_types"] == ["无"]


def test_image_forensic_pipeline_adds_region_for_forged_label_without_tool_coordinates(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 60), color="white").save(image_path)
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision"), vision_max_rounds=1),
        tools={"Spectrum_Grid_Analyzer": StubVisionTool("Spectrum_Grid_Analyzer")},
        evaluator=lambda prompt: (
            '{"label": "伪造图像", "probability": 0.7, "confidence": 0.65, '
            '"forgery_types": ["unknown"], "suspicious_regions": [], '
            '"tool_evidence": [], "limitations": [], "final_summary": "检测到伪造。"}'
        ),
    )

    result = pipeline.detect(str(image_path), language="zh")

    assert result["label"] == "伪造图像"
    assert result["suspicious_regions"]
    assert result["suspicious_regions"][0]["region"] == "20,15,60,45"
    assert result["suspicious_regions"][0]["region_reliability"] == "model_estimated_fallback"


def test_image_forensic_pipeline_preserves_model_user_answer(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 60), color="white").save(image_path)
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision"), vision_max_rounds=1),
        tools={"Spectrum_Grid_Analyzer": StubVisionTool("Spectrum_Grid_Analyzer")},
        evaluator=lambda prompt: (
            '{"label": "真实图像", "probability": 0.2, "confidence": 0.7, '
            '"forgery_types": ["无"], "suspicious_regions": [], '
            '"tool_evidence": [], "limitations": [], "final_summary": "图片更可能真实。", '
            '"user_answer": {"judgment": "图片更可能真实。", "reason_analysis": "画面结构自然，辅助检测未发现稳定异常。", "notes": ["工具结果仅作辅助。"]}}'
        ),
    )

    result = pipeline.detect(str(image_path), language="zh")

    assert result["user_answer"]["judgment"] == "图片更可能真实。"
    assert result["user_answer"]["reason_analysis"] == "画面结构自然，辅助检测未发现稳定异常。"
    assert result["user_answer"]["notes"] == ["工具结果仅作辅助。"]
    assert result["user_answer"]["conclusion"] == "图片更可能真实。"
    assert result["user_answer"]["analysis"] == "画面结构自然，辅助检测未发现稳定异常。"


def test_image_forensic_pipeline_uses_user_answer_forgery_regions(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 60), color="white").save(image_path)
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision"), vision_max_rounds=1),
        tools={"Spectrum_Grid_Analyzer": StubVisionTool("Spectrum_Grid_Analyzer")},
        evaluator=lambda prompt: (
            '{"label": "伪造图像", "probability": 0.7, "confidence": 0.65, '
            '"forgery_types": ["拼接"], "suspicious_regions": [], '
            '"tool_evidence": [], '
            '"user_answer": {"judgment": "图片存在局部伪造。", "reason_analysis": "主体边缘与背景不一致。", '
            '"notes": ["区域为综合判断结果。"], '
            '"forgery_regions": [{"region": "5,6,30,40", "score": 0.74, "reason": "主体边缘异常"}]}}'
        ),
    )

    result = pipeline.detect(str(image_path), language="zh")

    assert result["suspicious_regions"][0]["region"] == "5,6,30,40"
    assert result["user_answer"]["forgery_regions"][0]["region"] == "5,6,30,40"


def test_image_forensic_pipeline_keeps_only_one_forgery_type(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 60), color="white").save(image_path)
    pipeline = ImageForensicPipeline(
        config=Settings(vision_output_dir=str(tmp_path / "vision"), vision_max_rounds=1),
        tools={"Spectrum_Grid_Analyzer": StubVisionTool("Spectrum_Grid_Analyzer")},
        evaluator=lambda prompt: (
            '{"label": "伪造图像", "probability": 0.7, "confidence": 0.65, '
            '"forgery_types": ["拼接", "复制移动"], '
            '"user_answer": {"judgment": "图片存在伪造。", "reason_analysis": "局部区域异常。", '
            '"forgery_regions": [{"region": "5,6,30,40", "score": 0.74, "reason": "局部异常"}]}}'
        ),
    )

    result = pipeline.detect(str(image_path), language="zh")

    assert result["forgery_types"] == ["拼接"]


class StubQwenCompletion:
    def __init__(self, captured):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.update(kwargs)
        message = type("Message", (), {"content": '{"label": "Completely_Real"}'})
        choice = type("Choice", (), {"message": message})
        return type("Completion", (), {"choices": [choice]})


class StubQwenClient:
    def __init__(self, captured):
        self.chat = type("Chat", (), {"completions": StubQwenCompletion(captured)})()


class StubModelFactory:
    def __init__(self):
        self.captured = {}

    def qwen_vision_client(self, timeout=None):
        self.captured["timeout"] = timeout
        return StubQwenClient(self.captured)


def test_image_forensic_pipeline_calls_qwen_with_base64_image(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    model_factory = StubModelFactory()
    pipeline = ImageForensicPipeline(
        config=Settings(qwen_vision_model="qwen3-vl-plus", vision_verdict_timeout_seconds=9),
        model_factory=model_factory,
        tools={},
    )

    raw = pipeline._call_model("prompt text", "en", str(image_path))

    assert raw == '{"label": "Completely_Real"}'
    assert model_factory.captured["model"] == "qwen3-vl-plus"
    assert model_factory.captured["timeout"] == 9
    user_content = model_factory.captured["messages"][1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_content[1] == {"type": "text", "text": "prompt text"}
