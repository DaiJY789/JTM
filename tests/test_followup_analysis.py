from types import SimpleNamespace

from app.core.config import Settings
from app.services.followup_analysis import FollowupAnalysisService


class StubCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="底图较可能是真实拍摄，但现有证据不足以完全排除局部编辑。"
                    )
                )
            ]
        )


class StubModelFactory:
    def __init__(self):
        self.completions = StubCompletions()
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )

    def qwen_text_client(self, timeout=None):
        self.timeout = timeout
        return self.client


def test_followup_uses_current_mmd_model_and_plain_text_mode():
    config = Settings(
        qwen_text_model="legacy-text-model",
        mmd_vision_model="qwen3.7-flash",
        mmd_enable_thinking=False,
        text_verdict_timeout_seconds=42,
    )
    factory = StubModelFactory()
    service = FollowupAnalysisService(config, factory)

    output = service._call_model("prompt", "zh")

    assert output == "底图较可能是真实拍摄，但现有证据不足以完全排除局部编辑。"
    assert factory.timeout == 42
    call = factory.completions.calls[0]
    assert call["model"] == "qwen3.7-flash"
    assert call["model"] != config.qwen_text_model
    assert "response_format" not in call
    assert call["extra_body"] == {"enable_thinking": False}


def test_followup_prompt_forbids_claiming_new_tool_runs():
    service = FollowupAnalysisService(Settings(), StubModelFactory())

    prompt = service._build_prompt(
        "除了文字外，底图是否真实？",
        {"task": "deepfake_detection", "has_image": True},
        {"label": "伪造图像", "summary": "文字疑似后期叠加。"},
        [],
        "zh",
    )

    assert "不重新执行检测" in prompt
    assert "不编造新的工具结果" in prompt
    assert "直接输出自然语言回答" in prompt
    assert "不要输出 JSON" in prompt
    assert '"followup_question": "除了文字外，底图是否真实？"' in prompt


def test_followup_explain_returns_plain_text_without_updates():
    service = FollowupAnalysisService(Settings(), StubModelFactory())

    explanation = service.explain(
        "除了文字外，底图是否真实？",
        {"task": "deepfake_detection", "has_image": True},
        {"label": "伪造图像", "summary": "文字疑似后期叠加。"},
        [],
        "zh",
    )

    assert explanation["content"] == "底图较可能是真实拍摄，但现有证据不足以完全排除局部编辑。"
    assert explanation["raw_output"] == explanation["content"]
    assert explanation["updated_fields"] == {}
