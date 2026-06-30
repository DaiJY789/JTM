"""Centralized runtime configuration.

Secrets must be supplied through environment variables, not committed in source.
Business modules should import settings from here instead of reading env vars
directly.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def optional_timeout_from_env(name: str) -> float | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value or raw_value.lower() in {"none", "null", "false", "0"}:
        return None
    return float(raw_value)


@dataclass(frozen=True)
class Settings:
    bocha_api_key: str = os.getenv("BOCHA_API_KEY", "")

    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_vision_model: str = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-plus")

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", str(3600 * 24 * 30)))

    claim_max_count: int = int(os.getenv("CLAIM_MAX_COUNT", "3"))
    claim_extraction_timeout_seconds: float | None = optional_timeout_from_env("CLAIM_EXTRACTION_TIMEOUT_SECONDS")
    text_verdict_timeout_seconds: float | None = optional_timeout_from_env("TEXT_VERDICT_TIMEOUT_SECONDS")
    search_timeout_seconds: float | None = optional_timeout_from_env("SEARCH_TIMEOUT_SECONDS")
    vision_max_rounds: int = int(os.getenv("VISION_MAX_ROUNDS", "3"))
    vision_verdict_timeout_seconds: float | None = optional_timeout_from_env("VISION_VERDICT_TIMEOUT_SECONDS")
    vision_output_dir: str = os.getenv("VISION_OUTPUT_DIR", "outputs/vision")
    vision_external_tool_timeout_seconds: float = float(os.getenv("VISION_EXTERNAL_TOOL_TIMEOUT_SECONDS", "120"))
    vision_trufor_command: str = os.getenv("VISION_TRUFOR_COMMAND", "")
    vision_catnet_command: str = os.getenv("VISION_CATNET_COMMAND", "")
    vision_dire_command: str = os.getenv("VISION_DIRE_COMMAND", "")

    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "5000"))
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")


settings = Settings()


def apply_runtime_environment(config: Settings = settings) -> None:
    """Populate environment variables required by third-party libraries."""
    os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api_key
    os.environ["DASHSCOPE_API_KEY"] = config.dashscope_api_key


__all__ = ["Settings", "apply_runtime_environment", "settings"]
