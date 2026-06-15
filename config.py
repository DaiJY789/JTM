"""Centralized runtime configuration.

This project intentionally keeps API keys in this file per current project
requirements. Business modules should import settings from here instead of
hard-coding credentials locally.
"""

from dataclasses import dataclass
import os
from pathlib import Path


# Legacy defaults kept for backwards compatibility with existing imports.
selected_language = "zh"
selected_model = "自研"
selected_constract = "deepseek"
query = ""
constract_model_prompt = "您是虚假新闻检测任务的助理。你需要鉴定以下新闻内容是否正确，并根据你所知道的情况生成你的判断解释。"


def optional_timeout_from_env(name: str) -> float | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value or raw_value.lower() in {"none", "null", "false", "0"}:
        return None
    return float(raw_value)


def default_c2pa_command() -> str:
    local_tool = Path(__file__).resolve().parent / ".venv" / "bin" / "c2patool"
    executable = str(local_tool) if local_tool.exists() else "c2patool"
    return f"{executable} {{image_path}} --crjson"


@dataclass(frozen=True)
class Settings:
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "dcc98b22d5f7d413979a175ff7d75b721c5992a3ee1e2363020b2bbdf4f82404")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "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")

    bocha_api_key: str = os.getenv("BOCHA_API_KEY", "sk-adeb0353f3434158934a841ac0f8d4a4")
    abstract_ip_api_key: str = os.getenv("ABSTRACT_IP_API_KEY", "f962beaf785545d0a90b76d105996cdb")
    abstract_exchange_api_key: str = os.getenv("ABSTRACT_EXCHANGE_API_KEY", "3599c1514c554f6da2ae8916151e5770")
    abstract_holiday_api_key: str = os.getenv("ABSTRACT_HOLIDAY_API_KEY", "fa1dbc3bf696406db3412a7d034ba531")
    calendarific_api_key: str = os.getenv("CALENDARIFIC_API_KEY", "O7fuldKDDJgpl9ckui8lduMWNjx4bCDf")
    amap_api_key: str = os.getenv("AMAP_API_KEY", "e7ff59e1de1573456145f0d736cc3a2f")
    visual_crossing_api_key: str = os.getenv("VISUAL_CROSSING_API_KEY", "6LX8E4LWX7ZFSCCA5S3JLFVPS")

    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "sk-642e9f7c136c4f039be9f9b1a9fd46cf")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "sk-c884df06a4954672b1e1996f48271de1")
    dashscope_base_url: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_vision_model: str = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-plus")

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", str(3600 * 24 * 30)))
    clear_cache_password: str = os.getenv("CLEAR_CACHE_PASSWORD", "hit1920")

    default_language: str = os.getenv("DEFAULT_LANGUAGE", "zh")
    claim_max_count: int = int(os.getenv("CLAIM_MAX_COUNT", "3"))
    claim_extraction_timeout_seconds: float | None = optional_timeout_from_env("CLAIM_EXTRACTION_TIMEOUT_SECONDS")
    text_verdict_timeout_seconds: float | None = optional_timeout_from_env("TEXT_VERDICT_TIMEOUT_SECONDS")
    search_timeout_seconds: float | None = optional_timeout_from_env("SEARCH_TIMEOUT_SECONDS")
    vision_max_rounds: int = int(os.getenv("VISION_MAX_ROUNDS", "3"))
    vision_verdict_timeout_seconds: float | None = optional_timeout_from_env("VISION_VERDICT_TIMEOUT_SECONDS")
    vision_output_dir: str = os.getenv("VISION_OUTPUT_DIR", "outputs/vision")
    vision_external_tool_timeout_seconds: float = float(os.getenv("VISION_EXTERNAL_TOOL_TIMEOUT_SECONDS", "120"))
    vision_c2pa_command: str = os.getenv("VISION_C2PA_COMMAND", default_c2pa_command())
    vision_trufor_command: str = os.getenv("VISION_TRUFOR_COMMAND", "")
    vision_catnet_command: str = os.getenv("VISION_CATNET_COMMAND", "")
    vision_dire_command: str = os.getenv("VISION_DIRE_COMMAND", "")

    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "5000"))
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")


settings = Settings()


def apply_runtime_environment(config: Settings = settings) -> None:
    """Populate environment variables required by third-party libraries."""
    os.environ["SERPAPI_API_KEY"] = config.serpapi_api_key
    os.environ["TAVILY_API_KEY"] = config.tavily_api_key
    os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api_key
    os.environ["DASHSCOPE_API_KEY"] = config.dashscope_api_key
