from openai import OpenAI

from app.core.config import Settings, apply_runtime_environment, settings


class ModelFactory:
    def __init__(self, config: Settings = settings):
        self.config = config
        apply_runtime_environment(config)

    def deepseek_client(self, timeout: float | None = None) -> OpenAI:
        return OpenAI(
            api_key=self.config.deepseek_api_key,
            base_url=self.config.deepseek_base_url,
            timeout=timeout,
        )

    def qwen_vision_client(self, timeout: float | None = None) -> OpenAI:
        return OpenAI(
            api_key=self.config.dashscope_api_key,
            base_url=self.config.dashscope_base_url,
            timeout=timeout,
        )
