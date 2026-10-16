from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse

import requests


DATA_IMAGE_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$", re.DOTALL)


class AnthropicMessagesClient:
    """Expose an OpenAI-like chat.completions surface over Anthropic Messages."""

    def __init__(
        self,
        *,
        auth_token: str,
        base_url: str,
        model: str,
        timeout: float | None = None,
        requester: Callable[..., Any] | None = None,
    ):
        normalized_token = auth_token.strip()
        if not normalized_token:
            raise ValueError("ANTHROPIC_AUTH_TOKEN is not configured")
        if not normalized_token.isascii() or any(char.isspace() for char in normalized_token):
            raise ValueError("ANTHROPIC_AUTH_TOKEN contains whitespace or non-ASCII characters")
        normalized_base_url = base_url.strip().rstrip("/")
        parsed_base_url = urlparse(normalized_base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("ANTHROPIC_BASE_URL is not configured")
        self.auth_token = normalized_token
        self.base_url = normalized_base_url
        self.model = model.strip() or "qwen3.8-flash"
        self.timeout = timeout
        self.requester = requester or requests.post
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        system, messages = self._convert_messages(kwargs.get("messages") or [])
        max_tokens = max(1, int(kwargs.get("max_tokens") or 1024))
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        extra_body = kwargs.get("extra_body") if isinstance(kwargs.get("extra_body"), dict) else {}
        thinking_enabled = bool(extra_body.get("enable_thinking"))
        if thinking_enabled:
            budget = max(1, int(extra_body.get("thinking_budget") or 1024))
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(budget, max(1, max_tokens - 1)),
            }
        else:
            payload["thinking"] = {"type": "disabled"}
            if kwargs.get("temperature") is not None:
                payload["temperature"] = kwargs["temperature"]

        response = self.requester(
            self._messages_url(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.auth_token,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._to_openai_response(response.json())

    def _messages_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            content = message.get("content", "")
            if role == "system":
                text = self._text_content(content)
                if text:
                    system_parts.append(text)
                continue
            converted.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": self._convert_content(content),
                }
            )
        if not converted:
            converted.append({"role": "user", "content": ""})
        return "\n\n".join(system_parts), converted

    def _convert_content(self, content: Any) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")
        blocks: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                blocks.append({"type": "text", "text": str(item.get("text") or "")})
                continue
            if item_type != "image_url":
                continue
            image_url = item.get("image_url")
            url = str(image_url.get("url") if isinstance(image_url, dict) else image_url or "")
            match = DATA_IMAGE_RE.match(url)
            if match:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": match.group(1),
                            "data": match.group(2),
                        },
                    }
                )
            elif url.startswith(("http://", "https://")):
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                )
        return blocks

    def _text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")
        return "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()

    def _to_openai_response(self, payload: dict[str, Any]) -> Any:
        content = payload.get("content") if isinstance(payload.get("content"), list) else []
        text = "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        reasoning = "".join(
            str(block.get("thinking") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "thinking"
        )
        usage_payload = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        message = SimpleNamespace(content=text, reasoning_content=reasoning)
        choice = SimpleNamespace(
            message=message,
            finish_reason=str(payload.get("stop_reason") or ""),
        )
        usage = SimpleNamespace(
            prompt_tokens=int(usage_payload.get("input_tokens") or 0),
            completion_tokens=int(usage_payload.get("output_tokens") or 0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class RoutedChatClient:
    """Route OpenAI-shaped calls to DashScope, Anthropic, or automatic fallback."""

    def __init__(self, primary: Any | None, fallback: Any | None, provider: str):
        self.primary = primary
        self.fallback = fallback
        self.provider = provider
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        if self.provider == "anthropic":
            if self.fallback is None:
                raise RuntimeError("Anthropic-compatible model provider is not configured")
            return self.fallback.chat.completions.create(**kwargs)
        if self.provider == "dashscope":
            if self.primary is None:
                raise RuntimeError("DashScope model provider is not configured")
            return self.primary.chat.completions.create(**kwargs)
        if self.primary is None:
            if self.fallback is None:
                raise RuntimeError("No model provider is configured")
            return self.fallback.chat.completions.create(**kwargs)
        try:
            return self.primary.chat.completions.create(**kwargs)
        except Exception as primary_error:
            if self.fallback is None:
                raise
            try:
                return self.fallback.chat.completions.create(**kwargs)
            except Exception as fallback_error:
                raise RuntimeError(
                    "Both model providers failed: "
                    f"dashscope={type(primary_error).__name__}, "
                    f"anthropic={type(fallback_error).__name__}"
                ) from fallback_error


__all__ = ["AnthropicMessagesClient", "RoutedChatClient"]
