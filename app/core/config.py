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


def env_value(name: str, default: str = "") -> str:
    """Prefer MFC settings while accepting existing MMD environment variables."""
    value = os.getenv(name)
    if value is None and name.startswith("MFC_"):
        value = os.getenv("MMD_" + name.removeprefix("MFC_"))
    return default if value is None else value


def optional_timeout_from_env(name: str, default: float) -> float | None:
    raw_value = env_value(name).strip()
    if not raw_value:
        return default
    if raw_value.lower() in {"none", "null", "false", "0"}:
        return None
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def bool_from_env(name: str, default: bool) -> bool:
    raw_value = env_value(name).strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "on"}


def model_provider_from_env(name: str = "MODEL_PROVIDER", default: str = "auto") -> str:
    value = os.getenv(name, default).strip().lower()
    aliases = {"primary": "dashscope", "fallback": "anthropic"}
    value = aliases.get(value, value)
    if value not in {"auto", "dashscope", "anthropic"}:
        raise ValueError(f"{name} must be one of: auto, dashscope, anthropic")
    return value


@dataclass(frozen=True)
class Settings:
    bocha_api_key: str = os.getenv("BOCHA_API_KEY", "")
    bocha_search_url: str = os.getenv("BOCHA_SEARCH_URL", "https://api.bochaai.com/v1/web-search")

    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_provider: str = model_provider_from_env()
    anthropic_auth_token: str = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "qwen3.8-flash")
    qwen_text_model: str = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")
    qwen_vision_model: str = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-plus")
    mfc_vision_model: str = env_value("MFC_VISION_MODEL", "qwen3.7-flash")
    mfc_enable_thinking: bool = bool_from_env("MFC_ENABLE_THINKING", False)
    mfc_thinking_budget: int = int(env_value("MFC_THINKING_BUDGET", "1024"))
    mfc_detection_timeout_seconds: float | None = optional_timeout_from_env("MFC_DETECTION_TIMEOUT_SECONDS", 60)
    mfc_agent_timeout_seconds: float | None = optional_timeout_from_env("MFC_AGENT_TIMEOUT_SECONDS", 120)
    mfc_max_tool_rounds: int = int(env_value("MFC_MAX_TOOL_ROUNDS", "3"))
    mfc_invalid_output_max_retries: int = int(
        env_value("MFC_INVALID_OUTPUT_MAX_RETRIES", "2")
    )
    ifd_model_failure_max_retries: int = int(os.getenv("IFD_MODEL_FAILURE_MAX_RETRIES", "2"))
    mfc_max_tokens: int = int(env_value("MFC_MAX_TOKENS", "1800"))
    mfc_image_max_side: int = int(env_value("MFC_IMAGE_MAX_SIDE", "768"))
    mfc_max_images: int = int(env_value("MFC_MAX_IMAGES", "4"))
    mfc_ocr_timeout_seconds: float | None = optional_timeout_from_env("MFC_OCR_TIMEOUT_SECONDS", 30)
    mfc_ocr_max_tokens: int = int(env_value("MFC_OCR_MAX_TOKENS", "900"))
    mfc_ocr_image_max_side: int = int(env_value("MFC_OCR_IMAGE_MAX_SIDE", "1600"))
    mfc_ocr_enabled: bool = bool_from_env("MFC_OCR_ENABLED", True)
    mfc_external_evidence_enabled: bool = bool_from_env("MFC_EXTERNAL_EVIDENCE_ENABLED", True)
    mfc_search_query_count: int = int(env_value("MFC_SEARCH_QUERY_COUNT", "3"))
    mfc_search_results_per_query: int = int(env_value("MFC_SEARCH_RESULTS_PER_QUERY", "3"))
    mfc_search_total_results: int = int(env_value("MFC_SEARCH_TOTAL_RESULTS", "6"))
    mfc_search_snippet_characters: int = int(env_value("MFC_SEARCH_SNIPPET_CHARACTERS", "250"))

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", str(3600 * 24 * 30)))

    text_verdict_timeout_seconds: float | None = optional_timeout_from_env("TEXT_VERDICT_TIMEOUT_SECONDS", 90)
    search_timeout_seconds: float | None = optional_timeout_from_env("SEARCH_TIMEOUT_SECONDS", 15)
    vision_max_rounds: int = int(os.getenv("VISION_MAX_ROUNDS", "3"))
    vision_verdict_timeout_seconds: float | None = optional_timeout_from_env("VISION_VERDICT_TIMEOUT_SECONDS", 120)
    vision_output_dir: str = os.getenv("VISION_OUTPUT_DIR", "outputs/vision")

    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "5000"))
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")


settings = Settings()


def apply_runtime_environment(config: Settings = settings) -> None:
    """Populate environment variables required by third-party libraries."""
    os.environ["DASHSCOPE_API_KEY"] = config.dashscope_api_key


__all__ = [
    "Settings",
    "apply_runtime_environment",
    "model_provider_from_env",
    "settings",
]
