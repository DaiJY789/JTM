from types import SimpleNamespace

import pytest

from app.agents.models.anthropic_compat import AnthropicMessagesClient, RoutedChatClient


class StubResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_anthropic_client_converts_multimodal_openai_request():
    calls = []

    def requester(url, **kwargs):
        calls.append((url, kwargs))
        return StubResponse(
            {
                "content": [
                    {"type": "thinking", "thinking": "分析"},
                    {"type": "text", "text": '{"verdict":"True"}'},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 8},
            }
        )

    client = AnthropicMessagesClient(
        auth_token="secret",
        base_url="https://example.com/apps/anthropic",
        model="qwen3.8-flash",
        timeout=30,
        requester=requester,
    )
    completion = client.chat.completions.create(
        model="ignored-primary-model",
        messages=[
            {"role": "system", "content": "只输出 JSON"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,YWJj"},
                    },
                    {"type": "text", "text": "判断图片"},
                ],
            },
        ],
        max_tokens=1800,
        temperature=0,
        extra_body={"enable_thinking": True, "thinking_budget": 1024},
    )

    url, request = calls[0]
    assert url == "https://example.com/apps/anthropic/v1/messages"
    assert request["headers"]["x-api-key"] == "secret"
    assert request["json"]["model"] == "qwen3.8-flash"
    assert request["json"]["system"] == "只输出 JSON"
    assert request["json"]["messages"][0]["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"},
    }
    assert request["json"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "temperature" not in request["json"]
    assert completion.choices[0].message.content == '{"verdict":"True"}'
    assert completion.choices[0].message.reasoning_content == "分析"
    assert completion.usage.prompt_tokens == 20


class StubCompletions:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def stub_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_auto_router_falls_back_after_primary_error():
    primary = StubCompletions(error=RuntimeError("primary unavailable"))
    fallback_result = SimpleNamespace(ok=True)
    fallback = StubCompletions(result=fallback_result)
    client = RoutedChatClient(stub_client(primary), stub_client(fallback), "auto")

    assert client.chat.completions.create(model="anything") is fallback_result
    assert primary.calls == 1
    assert fallback.calls == 1


def test_anthropic_router_does_not_call_primary():
    primary = StubCompletions(error=AssertionError("primary must not run"))
    fallback_result = SimpleNamespace(ok=True)
    fallback = StubCompletions(result=fallback_result)
    client = RoutedChatClient(stub_client(primary), stub_client(fallback), "anthropic")

    assert client.chat.completions.create(model="anything") is fallback_result
    assert primary.calls == 0


def test_auto_router_uses_fallback_when_primary_is_not_configured():
    fallback_result = SimpleNamespace(ok=True)
    fallback = StubCompletions(result=fallback_result)
    client = RoutedChatClient(None, stub_client(fallback), "auto")

    assert client.chat.completions.create(model="anything") is fallback_result
    assert fallback.calls == 1


def test_anthropic_router_requires_fallback_configuration():
    primary = StubCompletions()
    client = RoutedChatClient(stub_client(primary), None, "anthropic")

    with pytest.raises(RuntimeError, match="not configured"):
        client.chat.completions.create(model="anything")


def test_anthropic_client_rejects_malformed_token():
    with pytest.raises(ValueError, match="whitespace or non-ASCII"):
        AnthropicMessagesClient(
            auth_token="token with space",
            base_url="https://example.com/apps/anthropic",
            model="qwen3.8-flash",
        )
