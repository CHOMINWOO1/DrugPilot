from __future__ import annotations

from dataclasses import dataclass

import pytest

from drugpilot.config import Settings
from drugpilot.domain.models import RunMode
from drugpilot.providers import llm as llm_module


@dataclass(frozen=True)
class _FakeProvider:
    provider: str
    key_marker: str


@pytest.fixture
def fake_provider_constructors(monkeypatch: pytest.MonkeyPatch) -> list[_FakeProvider]:
    """Replace both SDK adapters so policy tests cannot create a real client."""

    constructed: list[_FakeProvider] = []

    def fake_openai(api_key: str) -> _FakeProvider:
        provider = _FakeProvider("openai", f"length:{len(api_key)}")
        constructed.append(provider)
        return provider

    def fake_anthropic(api_key: str) -> _FakeProvider:
        provider = _FakeProvider("anthropic", f"length:{len(api_key)}")
        constructed.append(provider)
        return provider

    monkeypatch.setattr(llm_module, "OpenAIProvider", fake_openai)
    monkeypatch.setattr(llm_module, "AnthropicProvider", fake_anthropic)
    return constructed


def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_mock_without_keys_uses_rules_only(
    monkeypatch: pytest.MonkeyPatch,
    fake_provider_constructors: list[_FakeProvider],
) -> None:
    _clear_keys(monkeypatch)
    settings = Settings(_env_file=None, mode=RunMode.MOCK)
    registry = llm_module.ProviderRegistry(settings)

    assert settings.effective_mode is RunMode.MOCK
    assert registry.reviewer_backend == "deterministic-rules"
    assert registry.for_reviewer() is None
    assert fake_provider_constructors == []


def test_openai_key_auto_hybrid_enables_same_read_only_reviewer_provider(
    monkeypatch: pytest.MonkeyPatch,
    fake_provider_constructors: list[_FakeProvider],
) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-only")
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        auto_activate_openai=True,
        openai_ma3_reviewer_enabled=True,
    )
    registry = llm_module.ProviderRegistry(settings)

    assert settings.effective_mode is RunMode.HYBRID
    assert settings.openai_auto_activated is True
    assert registry.reviewer_backend == "openai-read-only"
    assert registry.openai is not None
    assert registry.for_reviewer() is registry.openai
    assert registry.for_reviewer() is registry.for_component("MA1")
    assert [provider.provider for provider in fake_provider_constructors] == ["openai"]


def test_openai_reviewer_disabled_falls_back_to_rules(
    monkeypatch: pytest.MonkeyPatch,
    fake_provider_constructors: list[_FakeProvider],
) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-disabled")
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        auto_activate_openai=True,
        openai_ma3_reviewer_enabled=False,
    )
    registry = llm_module.ProviderRegistry(settings)

    assert settings.effective_mode is RunMode.HYBRID
    assert registry.component_backend == "openai"
    assert registry.reviewer_backend == "deterministic-rules"
    assert registry.for_reviewer() is None
    assert [provider.provider for provider in fake_provider_constructors] == ["openai"]


def test_anthropic_is_preferred_when_both_keys_are_configured(
    monkeypatch: pytest.MonkeyPatch,
    fake_provider_constructors: list[_FakeProvider],
) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        auto_activate_openai=True,
        openai_ma3_reviewer_enabled=True,
    )
    registry = llm_module.ProviderRegistry(settings)

    assert settings.effective_mode is RunMode.HYBRID
    assert registry.reviewer_backend == "anthropic"
    assert registry.anthropic_reviewer is not None
    assert registry.for_reviewer() is registry.anthropic_reviewer
    assert registry.for_reviewer() is not registry.openai
    assert [provider.provider for provider in fake_provider_constructors] == [
        "openai",
        "anthropic",
    ]

