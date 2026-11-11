import json

import pytest

from app.agents.MFC_Agent import MFCDetectionAgent
from app.core.config import Settings


class FakeImageService:
    def __init__(self, model_factory=None):
        self.model_factory = model_factory

    def build_image_content_part(self, image_content=None, image_url=None):
        if not image_content and not image_url:
            return None
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


def mfc_payload():
    return {
        "label": "Mostly False",
        "confidence": 0.82,
        "type": "C2_contextual_inconsistency",
        "atomic_points": [
            {
                "claim": "图片展示了文字所述事件。",
                "visual_observation": "图片中的地点标识与文字冲突。",
                "search_query": "事件 地点 现场图片",
                "point_type": "crossmodal_claim",
            }
        ],
        "fine_grained_checks": {
            "T1_fabrication": {"assessment": "absent", "severity": 0, "evidence_basis": "none", "analysis": "未见明确虚构。"},
            "T2_distortion": {"assessment": "confirmed", "severity": 2, "evidence_basis": "text_internal", "analysis": "文字扩大了图片可支持的事实。"},
            "T3_misleading_context": {"assessment": "confirmed", "severity": 3, "evidence_basis": "visual_input", "analysis": "时间地点无法对应。"},
            "V1_synthetic_visual_content": {"assessment": "absent", "severity": 0, "evidence_basis": "visual_input", "analysis": "未见生成痕迹。"},
            "V2_visual_editing": {"assessment": "absent", "severity": 0, "evidence_basis": "visual_input", "analysis": "未见编辑痕迹。"},
            "C1_semantic_inconsistency": {"assessment": "confirmed", "severity": 2, "evidence_basis": "visual_input", "analysis": "场景与事件明确冲突。"},
            "C2_contextual_inconsistency": {"assessment": "confirmed", "severity": 3, "evidence_basis": "visual_input", "analysis": "图片语境不匹配。"},
            "C3_pragmatic_inconsistency": {"assessment": "absent", "severity": 0, "evidence_basis": "none", "analysis": "未见讽刺。"},
        },
        "user_answer": {
            "judgment": "该内容整体上基本虚假。",
            "analysis": "图片可能真实，但其语境不能支持文字中的核心事件。",
            "notes": ["未使用外部新闻来源核验。"],
        },
    }


def test_mfc_agent_returns_label_and_user_explanation_in_one_call():
    calls = []

    def evaluator(prompt, image_content, image_url, language):
        calls.append((prompt, image_content, image_url, language))
        return json.dumps(mfc_payload(), ensure_ascii=False)

    agent = MFCDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator)
    result = agent.detect(text="新闻声称图片展示某事件。", image_content=b"image", language="zh")

    assert len(calls) == 1
    assert result["status"] == "success"
    assert result["label"] == "Mostly False"
    assert result["type"] == "C2_contextual_inconsistency"
    assert result["user_answer"]["judgment"] == "该内容整体判断为基本虚假。"
    assert result["fine_grained_checks"]["C2_contextual_inconsistency"]["severity"] == 3
    assert result["fine_grained_checks"]["C2_contextual_inconsistency"]["assessment"] == "confirmed"
    assert len(result["fine_grained_checks"]) == 8
    assert result["atomic_points"][0]["search_query"] == "事件 地点 现场图片"
    assert result["timings"]["model_calls"] == 1
    assert "五类之一" in calls[0][0]
    assert "八个细粒度分析维度" in calls[0][0]
    assert "T1_fabrication" in calls[0][0]
    assert "C3_pragmatic_inconsistency" in calls[0][0]
    assert "原子点解析" in calls[0][0]
    assert "检测流程（必须按顺序完成）" in calls[0][0]
    assert '"confidence_label"' not in calls[0][0]
    assert '"fine_grained_checks"' in calls[0][0]
    assert '"user_answer"' in calls[0][0]
    assert '"current_date"' in calls[0][0]
    assert '"selected_image_count": 1' in calls[0][0]
    assert "verdict" not in result
    assert "explanation" not in result
    assert set(result["user_answer"]) == {"judgment", "analysis", "notes"}


@pytest.mark.parametrize("dimension", MFCDetectionAgent.SIGNAL_KEYS)
def test_mfc_primary_type_preserves_the_model_choice(dimension):
    agent = MFCDetectionAgent(FakeImageService(), Settings())
    data = mfc_payload()
    # Even a lower-severity active dimension can be the main issue for a core claim.
    data["fine_grained_checks"][dimension] = {
        "assessment": "suspected", "severity": 1,
        "evidence_basis": "visual_input", "analysis": "对核心主张最关键的线索。",
    }
    data["type"] = dimension
    assert agent._normalize_result(data, "zh")["type"] == dimension


@pytest.mark.parametrize("invalid", [None, "bad_key", "V2_visual_editing", "none"])
def test_mfc_primary_type_falls_back_only_to_active_dimensions(invalid):
    agent = MFCDetectionAgent(FakeImageService(), Settings())
    data = mfc_payload()
    data["type"] = invalid
    result = agent._normalize_result(data, "zh")
    selected = result["fine_grained_checks"][result["type"]]
    assert selected["assessment"] == "confirmed"
    assert selected["severity"] == 3


def test_mfc_primary_type_distinguishes_no_risk_from_unverifiable_text():
    agent = MFCDetectionAgent(FakeImageService(), Settings())
    data = mfc_payload()
    data["label"] = "True"
    data["type"] = "V1_synthetic_visual_content"
    data["fine_grained_checks"] = {
        key: {"assessment": "absent", "analysis": "未发现风险。"}
        for key in agent.SIGNAL_KEYS
    }
    assert agent._normalize_result(data, "zh", has_image=False)["type"] == "none"
    data["fine_grained_checks"]["T1_fabrication"] = {"assessment": "not_verifiable"}
    assert agent._normalize_result(data, "zh", has_image=False)["type"] == "unknown"
    data["fine_grained_checks"]["T2_distortion"] = {
        "assessment": "confirmed", "severity": 2, "analysis": "夸大了事实。",
    }
    assert agent._normalize_result(data, "zh", has_image=False)["type"] == "T2_distortion"


def test_mfc_agent_normalizes_legacy_output_to_new_contract():
    data = mfc_payload()
    data["verdict"] = data.pop("label")
    answer = data.pop("user_answer")
    data["explanation"] = {
        "verdict_statement": answer["judgment"],
        "analysis": answer["analysis"],
        "limitations": answer["notes"],
    }
    agent = MFCDetectionAgent(FakeImageService(), Settings(), evaluator=lambda *_: json.dumps(data))
    result = agent.detect(text="新闻文字", image_content=b"image", language="zh")

    assert result["status"] == "success"
    assert result["label"] == "Mostly False"
    assert result["user_answer"]["analysis"] == answer["analysis"]
    assert result["user_answer"]["notes"] == answer["notes"]
    assert "verdict" not in result
    assert "explanation" not in result


def test_mfc_agent_supports_text_only_input():
    calls = []

    def evaluator(prompt, image_content, image_url, language):
        calls.append((prompt, image_content, image_url, language))
        data = mfc_payload()
        for key in (
            "V1_synthetic_visual_content",
            "V2_visual_editing",
            "C1_semantic_inconsistency",
            "C2_contextual_inconsistency",
            "C3_pragmatic_inconsistency",
        ):
            data["fine_grained_checks"][key] = {
                "assessment": "not_verifiable",
                "severity": 0,
                "evidence_basis": "none",
                "analysis": "未提供图片。",
            }
        return json.dumps(data, ensure_ascii=False)

    agent = MFCDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator)
    result = agent.detect(text="待核查的文本主张。", language="zh")

    assert result["status"] == "success"
    assert result["timings"]["input_image_count"] == 0
    assert calls[0][1] is None
    assert calls[0][2] is None
    assert "纯文本事实核查智能体" in calls[0][0]
    assert "三维核查" in calls[0][0]
    assert "input_scope" not in calls[0][0]
    assert "visual_observation" not in calls[0][0]
    assert "V1_synthetic_visual_content" not in calls[0][0]
    for key in (
        "V1_synthetic_visual_content",
        "V2_visual_editing",
        "C1_semantic_inconsistency",
        "C2_contextual_inconsistency",
        "C3_pragmatic_inconsistency",
    ):
        assert result["fine_grained_checks"][key]["assessment"] == "not_verifiable"
        assert result["fine_grained_checks"][key]["severity"] == 0
        assert result["fine_grained_checks"][key]["evidence_basis"] == "none"


def test_mfc_agent_generates_explanation_fallback_without_second_call():
    calls = []

    def evaluator(*args):
        calls.append(args)
        return json.dumps({"label": "Mixture", "confidence": 0.4, "fine_grained_checks": {}}, ensure_ascii=False)

    agent = MFCDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator)
    result = agent.detect(text="待判断内容", image_url="https://example.com/image.jpg", language="zh")

    assert len(calls) == 1
    assert result["label"] == "Mixture"
    assert result["decision_status"] == "low_confidence"
    assert result["confidence"] == 0.4
    assert result["user_answer"]["analysis"]
    assert result["user_answer"]["judgment"]


def test_mfc_agent_rejects_invalid_verdict_as_technical_error():
    agent = MFCDetectionAgent(
        FakeImageService(),
        Settings(),
        evaluator=lambda *_: json.dumps({"label": "Cannot_Determine"}),
    )

    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "error"
    assert result["label"] is None
    assert result["confidence"] is None
    assert result["timings"]["model_calls"] == 3


def test_mfc_agent_timeout_returns_no_verdict():
    def timed_out(*_):
        raise TimeoutError("Request timed out.")

    agent = MFCDetectionAgent(FakeImageService(), Settings(), evaluator=timed_out)

    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "error"
    assert result["label"] is None
    assert result["confidence"] is None
    assert result["error"] == "Request timed out."
    assert result["timings"]["model_calls"] == 3
    assert "不输出五分类标签" in result["user_answer"]["analysis"]


def test_mfc_agent_recovers_after_transient_model_failure():
    calls = []

    def evaluator(*_):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("Temporary timeout")
        return json.dumps(mfc_payload(), ensure_ascii=False)

    agent = MFCDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator)
    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "success"
    assert result["label"] == "Mostly False"
    assert result["timings"]["model_calls"] == 2
    assert result["timings"]["model_attempts"][0]["error"] == "Temporary timeout"


def test_mfc_agent_returns_no_verdict_for_truncated_json():
    agent = MFCDetectionAgent(
        FakeImageService(),
        Settings(),
        evaluator=lambda *_: '{"label":"Mostly True","confidence":',
    )

    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "error"
    assert result["label"] is None
    assert result["confidence"] is None


def test_mfc_agent_retries_empty_object_then_accepts_valid_json():
    outputs = iter(["{}", json.dumps(mfc_payload(), ensure_ascii=False)])
    prompts = []

    def evaluator(prompt, *_):
        prompts.append(prompt)
        return next(outputs)

    agent = MFCDetectionAgent(
        FakeImageService(),
        Settings(mfc_invalid_output_max_retries=2),
        evaluator=evaluator,
    )

    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "success"
    assert result["label"] == "Mostly False"
    assert result["timings"]["model_calls"] == 2
    assert result["timings"]["invalid_output_retries"] == 1
    assert len(result["timings"]["model_attempts"]) == 2
    assert result["timings"]["model_attempts"][0]["output_characters"] == 2
    assert "不要返回 {}" in prompts[1]


def test_mfc_agent_stops_after_two_invalid_output_retries():
    calls = []

    def evaluator(*args):
        calls.append(args)
        return "{}"

    agent = MFCDetectionAgent(
        FakeImageService(),
        Settings(mfc_invalid_output_max_retries=2),
        evaluator=evaluator,
    )

    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "error"
    assert result["label"] is None
    assert len(calls) == 3
    assert result["timings"]["model_calls"] == 3
    assert result["timings"]["invalid_output_retries"] == 2
    assert [item["output_characters"] for item in result["timings"]["model_attempts"]] == [
        2,
        2,
        2,
    ]


def test_mfc_agent_uses_json_response_format_for_production_client():
    captured = {}

    class Completion:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": json.dumps(mfc_payload(), ensure_ascii=False)})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class Factory:
        def qwen_vision_client(self, timeout=None, max_retries=2):
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
            return type("Client", (), {"chat": type("Chat", (), {"completions": Completion()})()})()

    factory = Factory()
    agent = MFCDetectionAgent(
        FakeImageService(factory),
        Settings(mfc_enable_thinking=False),
        model_factory=factory,
    )
    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "success"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 1800
    assert captured["model"] == "qwen3.7-flash"
    assert captured["extra_body"] == {"enable_thinking": False}
    assert captured["timeout"] == 60
    assert captured["max_retries"] == 0


def test_mfc_agent_sends_thinking_budget_when_enabled():
    captured = {}

    class Completion:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type(
                "Message",
                (),
                {
                    "content": json.dumps(mfc_payload(), ensure_ascii=False),
                    "reasoning_content": "internal reasoning",
                },
            )()
            usage = type(
                "Usage",
                (),
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "completion_tokens_details": type(
                        "Details", (), {"reasoning_tokens": 80}
                    )(),
                },
            )()
            choice = type(
                "Choice", (), {"message": message, "finish_reason": "stop"}
            )()
            return type("Response", (), {"choices": [choice], "usage": usage})()

    class Factory:
        def qwen_vision_client(self, timeout=None, max_retries=2):
            return type("Client", (), {"chat": type("Chat", (), {"completions": Completion()})()})()

    factory = Factory()
    agent = MFCDetectionAgent(
        FakeImageService(factory),
        Settings(mfc_enable_thinking=True, mfc_thinking_budget=1024),
        model_factory=factory,
    )

    result = agent.detect(text="待判断内容", image_content=b"image", language="zh")

    assert result["status"] == "success"
    assert captured["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 1024,
    }
    assert result["timings"]["thinking_characters"] == len("internal reasoning")
    assert result["timings"]["reasoning_tokens"] == 80
    assert result["timings"]["finish_reason"] == "stop"


def test_mfc_agent_sends_multiple_images_in_order_to_final_model():
    captured = {}

    class Completion:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": json.dumps(mfc_payload(), ensure_ascii=False)})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class Factory:
        def qwen_vision_client(self, timeout=None, max_retries=2):
            return type("Client", (), {"chat": type("Chat", (), {"completions": Completion()})()})()

    factory = Factory()
    agent = MFCDetectionAgent(
        FakeImageService(factory),
        Settings(mfc_max_images=4),
        model_factory=factory,
    )

    result = agent.detect(
        text="待判断多图新闻",
        image_contents=[b"image-1", b"image-2"],
        image_urls=["https://example.com/image-3.jpg"],
        image_labels=["Image 2", "Image 7", "Image 9"],
        language="zh",
    )

    content = captured["messages"][1]["content"]
    assert [item["text"] for item in content if item["type"] == "text"][:3] == [
        "[Image 2]",
        "[Image 7]",
        "[Image 9]",
    ]
    assert sum(item["type"] == "image_url" for item in content) == 3
    assert '"selected_image_count": 3' in content[-1]["text"]
    assert '"image_labels": ["Image 2", "Image 7", "Image 9"]' in content[-1]["text"]
    assert result["timings"]["input_image_count"] == 3


def test_mfc_agent_keeps_model_false_and_confidence_with_only_model_knowledge():
    data = mfc_payload()
    data["label"] = "False"
    data["confidence"] = 0.95
    data["claim_assessments"] = [
        {
            "claim": "某事件已经发生。",
            "importance": "core",
            "status": "contradicted",
            "evidence_basis": "model_knowledge",
            "evidence": "模型已有知识与声明不一致。",
        }
    ]
    data["fine_grained_checks"] = {}
    agent = MFCDetectionAgent(
        FakeImageService(), Settings(), evaluator=lambda *_: json.dumps(data, ensure_ascii=False)
    )

    result = agent.detect(text="某事件已经发生。", image_content=b"image", language="zh")

    assert result["label"] == "False"
    assert "model_verdict" not in result
    assert "verdict_adjustments" not in result
    assert result["decision_status"] == "confident"
    assert result["confidence"] == 0.95
    assert "模型已有知识" in "".join(result["user_answer"]["notes"])


def test_mfc_agent_repairs_missing_comma_between_json_objects():
    agent = MFCDetectionAgent(FakeImageService(), Settings(), evaluator=lambda *_: "")
    parsed = agent._parse_json('{"label":"Mixture","signals":{} "user_explanation":{}}')

    assert parsed["label"] == "Mixture"


def test_mfc_agent_keeps_model_label_and_confidence_when_evidence_is_insufficient():
    data = mfc_payload()
    data["label"] = "True"
    data["claim_assessments"] = [
        {
            "claim": "某事件已经发生。",
            "importance": "core",
            "status": "not_verifiable",
            "evidence_basis": "none",
            "evidence": "图片无法验证该事件。",
        }
    ]
    data["fine_grained_checks"] = {}
    agent = MFCDetectionAgent(
        FakeImageService(), Settings(), evaluator=lambda *_: json.dumps(data, ensure_ascii=False)
    )

    result = agent.detect(text="某事件已经发生。", image_content=b"image", language="zh")

    assert result["label"] == "True"
    assert "raw_confidence" not in result
    assert "confidence_calibration" not in result
    assert result["decision_status"] == "confident"
    assert result["confidence"] == data["confidence"]
    assert result["crossmodal_assessment"]["label"] == "Cannot_Determine"


def test_mfc_agent_does_not_change_true_label_for_grounded_minor_errors():
    data = mfc_payload()
    data["label"] = "True"
    data["claim_assessments"] = [
        {
            "claim": "核心事件确实发生。",
            "importance": "core",
            "status": "supported",
            "evidence_basis": "external_search",
            "evidence": "S1为官方通报。",
        },
        {
            "claim": "影响覆盖全部地区。",
            "importance": "supporting",
            "status": "contradicted",
            "evidence_basis": "external_search",
            "evidence": "S2显示仅影响局部地区。",
        },
    ]
    data["fine_grained_checks"] = {}
    agent = MFCDetectionAgent(
        FakeImageService(), Settings(), evaluator=lambda *_: json.dumps(data, ensure_ascii=False)
    )

    result = agent.detect(text="核心事件发生并影响全部地区。", image_content=b"image")

    assert result["label"] == "True"
    assert result["diagnostics"]["model_verdict"] == "True"
    assert result["diagnostics"]["verdict_adjustments"] == []
    assert result["confidence"] == data["confidence"]


@pytest.mark.parametrize("label", MFCDetectionAgent.VERDICTS)
@pytest.mark.parametrize("has_image", [False, True])
@pytest.mark.parametrize("confidence", [0.0, 1.0, 0.987654321])
def test_mfc_and_tfc_preserve_every_legal_label_and_exact_model_confidence(label, has_image, confidence):
    data = mfc_payload()
    data.update(label=label, confidence=confidence)
    agent = MFCDetectionAgent(
        FakeImageService(), Settings(), evaluator=lambda *_: json.dumps(data),
    )
    result = agent.detect(text="待核查主张", image_content=b"image" if has_image else None)
    assert result["status"] == "success"
    assert result["label"] == label
    assert result["confidence"] == confidence
    assert result["diagnostics"]["verdict_adjustments"] == []


@pytest.mark.parametrize(("raw_label", "expected"), [
    ("mostly_true", "Mostly True"), ("mixed", "Mixture"),
    ("completely_false", "False"), ("大部分真实", "Mostly True"),
    ("真伪混合", "Mixture"),
])
def test_mfc_maps_only_label_aliases_without_changing_model_confidence(raw_label, expected):
    agent = MFCDetectionAgent(FakeImageService(), Settings())
    result = agent._normalize_result({"label": raw_label, "confidence": 0.91234567}, "zh")
    assert result["label"] == expected
    assert result["confidence"] == 0.91234567


@pytest.mark.parametrize("value", [None, "0.9", True, -0.1, 1.1, float("nan"), float("inf")])
def test_mfc_does_not_invent_or_clip_missing_or_invalid_confidence(value):
    agent = MFCDetectionAgent(FakeImageService(), Settings())
    result = agent._normalize_result({"label": "True", "confidence": value}, "zh")
    assert result["label"] == "True"
    assert result["confidence"] is None


def test_mfc_preserves_explicit_model_decision_status_without_confidence_calibration():
    agent = MFCDetectionAgent(FakeImageService(), Settings())
    result = agent._normalize_result({
        "label": "True", "confidence": 0.91234567, "decision_status": "insufficient_evidence",
    }, "zh")
    assert result["decision_status"] == "insufficient_evidence"
    assert result["confidence"] == 0.91234567


def test_mfc_agent_deduplicates_reposts_and_marks_source_tier():
    agent = MFCDetectionAgent(FakeImageService(), Settings())
    batches = [
        (
            {"query_id": "Q1", "strategy": "whole_post", "query": "事件"},
            {
                "status": "success",
                "results": [
                    {
                        "title": "政府发布事件调查通报",
                        "url": "https://agency.gov/report",
                        "snippet": "政府发布事件调查通报，确认事件仅影响局部地区并已经恢复。",
                    },
                    {
                        "title": "政府发布事件调查通报转载",
                        "url": "https://copy.example/repost",
                        "snippet": "政府发布事件调查通报，确认事件仅影响局部地区并已经恢复。",
                    },
                ],
                "images": [],
                "limitations": [],
            },
        )
    ]

    result = agent._compact_search_results(batches)

    assert len(result["results"]) == 1
    assert result["results"][0]["source_tier"] == "official_candidate"
