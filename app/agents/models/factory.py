from openai import OpenAI

from app.agents.models.anthropic_compat import AnthropicMessagesClient, RoutedChatClient
from app.core.config import Settings, apply_runtime_environment, settings


class ModelFactory:
    def __init__(self, config: Settings = settings):
        self.config = config
        apply_runtime_environment(config)

    def qwen_vision_client(
        self,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> OpenAI | RoutedChatClient:
        primary = None
        if self.config.model_provider != "anthropic" and self.config.dashscope_api_key:
            primary = OpenAI(
                api_key=self.config.dashscope_api_key,
                base_url=self.config.dashscope_base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        return RoutedChatClient(primary, self._anthropic_client(timeout), self.config.model_provider)

    def qwen_text_client(self, timeout: float | None = None) -> OpenAI | RoutedChatClient:
        primary = None
        if self.config.model_provider != "anthropic" and self.config.dashscope_api_key:
            primary = OpenAI(
                api_key=self.config.dashscope_api_key,
                base_url=self.config.dashscope_base_url,
                timeout=timeout,
            )
        return RoutedChatClient(primary, self._anthropic_client(timeout), self.config.model_provider)

    def _anthropic_client(self, timeout: float | None) -> AnthropicMessagesClient | None:
        if not (self.config.anthropic_auth_token and self.config.anthropic_base_url):
            return None
        return AnthropicMessagesClient(
            auth_token=self.config.anthropic_auth_token,
            base_url=self.config.anthropic_base_url,
            model=self.config.anthropic_model,
            timeout=timeout,
        )
