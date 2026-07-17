from app.services.textual.textual_verdict import TextualVerdictService


class StubConfig:
    text_verdict_timeout_seconds = 12
    deepseek_model = "deepseek-chat"


class StubTextVerdictContext:
    config = StubConfig()
    model_factory = object()


def test_text_verdict_uses_model_label_without_evidence_gate():
    service = TextualVerdictService(
        StubTextVerdictContext(),
        evaluator=lambda prompt, timeout: """
        {"label": "Completely_True", "confidence": 0.8, "summary": "Supported by model judgement",
         "reasoning": "The model judged the available context.", "evidence_assessment": "insufficient",
         "claim_assessments": []}
        """,
    )
    result = service.evaluate(
        text="A claim",
        claims=[{"claim": "A claim", "evidence": []}],
        evidence=[{"claim_index": 0, "claim": "A claim", "items": []}],
        language="en",
    )

    assert result["label"] == "Completely_True"
    assert result["type"] == "fact_checking"
    assert result["confidence"] == 0.8
    assert result["label_policy"]["model_judged_evidence_sufficiency"] is True


def test_text_verdict_passes_evidence_into_prompt():
    captured = {}

    def evaluator(prompt, timeout):
        captured["prompt"] = prompt
        return '{"label": "Mostly_False", "summary": "Refuted", "reasoning": "Evidence refutes it."}'

    service = TextualVerdictService(StubTextVerdictContext(), evaluator=evaluator)
    result = service.evaluate(
        text="A claim",
        claims=[{"claim": "A claim", "evidence": [{"title": "Report", "snippet": "Evidence"}]}],
        evidence=[{"claim_index": 0, "claim": "A claim", "items": [{"title": "Report"}]}],
        language="en",
    )

    assert result["label"] == "Mostly_False"
    assert '"evidence"' in captured["prompt"]
    assert "Report" in captured["prompt"]


def test_text_verdict_normalizes_chinese_label():
    service = TextualVerdictService(
        StubTextVerdictContext(),
        evaluator=lambda prompt, timeout: '{"label": "大部分虚假", "summary": "证据反驳"}',
    )

    result = service.evaluate(
        text="一条声明",
        claims=[{"claim": "一条声明", "evidence": [{"title": "报道"}]}],
        evidence=[{"claim_index": 0, "claim": "一条声明", "items": [{"title": "报道"}]}],
        language="zh",
    )

    assert result["label"] == "Mostly_False"


def test_text_verdict_accepts_chinese_cannot_determine_label():
    service = TextualVerdictService(
        StubTextVerdictContext(),
        evaluator=lambda prompt, timeout: '{"label": "无法判断", "reasoning": "证据不足"}',
    )

    result = service.evaluate(
        text="一条无法验证的声明",
        claims=[{"claim": "一条无法验证的声明", "evidence": []}],
        evidence=[{"claim_index": 0, "claim": "一条无法验证的声明", "items": []}],
        language="zh",
    )

    assert result["label"] == "Cannot_Determine"
    assert result["type"] == "谣言检测"
    assert result["user_answer"]["conclusion"] == "最终标签：无法判断。置信度：未知。"
