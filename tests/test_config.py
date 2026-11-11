import pytest

from app.core.config import bool_from_env, env_value, model_provider_from_env, optional_timeout_from_env


def test_timeout_uses_safe_default_when_environment_is_unset(monkeypatch):
    monkeypatch.delenv("TEST_TIMEOUT_SECONDS", raising=False)

    assert optional_timeout_from_env("TEST_TIMEOUT_SECONDS", 30) == 30


def test_timeout_rejects_negative_values(monkeypatch):
    monkeypatch.setenv("TEST_TIMEOUT_SECONDS", "-1")

    with pytest.raises(ValueError, match="greater than zero"):
        optional_timeout_from_env("TEST_TIMEOUT_SECONDS", 30)


def test_model_provider_accepts_aliases(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_PROVIDER", "fallback")

    assert model_provider_from_env("TEST_MODEL_PROVIDER") == "anthropic"


def test_model_provider_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="auto, dashscope, anthropic"):
        model_provider_from_env("TEST_MODEL_PROVIDER")


def test_mfc_environment_keeps_legacy_settings_and_prefers_new_names(monkeypatch):
    monkeypatch.delenv("MFC_VISION_MODEL", raising=False)
    monkeypatch.setenv("MMD_VISION_MODEL", "existing-model")
    assert env_value("MFC_VISION_MODEL", "default-model") == "existing-model"
    monkeypatch.setenv("MFC_VISION_MODEL", "new-model")
    assert env_value("MFC_VISION_MODEL", "default-model") == "new-model"


def test_mfc_timeout_and_boolean_settings_accept_legacy_names(monkeypatch):
    monkeypatch.delenv("MFC_AGENT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("MMD_AGENT_TIMEOUT_SECONDS", "37")
    assert optional_timeout_from_env("MFC_AGENT_TIMEOUT_SECONDS", 120) == 37
    monkeypatch.delenv("MFC_ENABLE_THINKING", raising=False)
    monkeypatch.setenv("MMD_ENABLE_THINKING", "true")
    assert bool_from_env("MFC_ENABLE_THINKING", False) is True
    monkeypatch.setenv("MFC_ENABLE_THINKING", "false")
    assert bool_from_env("MFC_ENABLE_THINKING", True) is False
