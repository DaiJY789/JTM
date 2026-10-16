import pytest

from app.core.config import model_provider_from_env, optional_timeout_from_env


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
