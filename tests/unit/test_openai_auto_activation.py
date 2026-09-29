from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode
from drugpilot.providers.llm import MockLLMProvider, OpenAIProvider, ProviderRegistry
from drugpilot.storage import SQLiteRunStore


def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_no_key_preserves_safe_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_keys(monkeypatch)
    settings = Settings(_env_file=None, mode=RunMode.MOCK)
    registry = ProviderRegistry(settings)

    assert settings.effective_mode is RunMode.MOCK
    assert settings.openai_auto_activated is False
    assert registry.component_backend == "mock"
    assert isinstance(registry.for_component("ma1"), MockLLMProvider)


def test_openai_key_alone_auto_activates_real_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-do-not-use")
    settings = Settings(_env_file=None, mode=RunMode.MOCK)
    registry = ProviderRegistry(settings)

    assert settings.mode is RunMode.MOCK
    assert settings.effective_mode is RunMode.HYBRID
    assert settings.openai_auto_activated is True
    assert registry.component_backend == "openai"
    assert isinstance(registry.for_component("planner"), OpenAIProvider)


def test_auto_activation_can_be_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-do-not-use")
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        auto_activate_openai=False,
    )

    assert settings.effective_mode is RunMode.MOCK
    assert ProviderRegistry(settings).component_backend == "mock"


def test_blank_key_is_not_treated_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    settings = Settings(_env_file=None, mode=RunMode.MOCK)

    assert settings.openai_api_key is None
    assert settings.effective_mode is RunMode.MOCK


class ParsedAnswer(BaseModel):
    answer: str


class FakeResponses:
    def __init__(self) -> None:
        self.request: dict | None = None

    async def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            status="completed",
            error=None,
            output=[],
            output_text=ParsedAnswer(answer="validated").model_dump_json(),
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=9,
                total_tokens=21,
                output_tokens_details=SimpleNamespace(reasoning_tokens=4),
            ),
        )


@pytest.mark.asyncio
async def test_openai_provider_uses_responses_structured_output() -> None:
    responses = FakeResponses()
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    fallback = ParsedAnswer(answer="fixture")

    result = await provider.structured(
        component="ma1",
        model="gpt-test",
        system="bounded worker",
        prompt="return a validated answer",
        response_model=ParsedAnswer,
        fallback=fallback,
        max_output_tokens=100,
    )

    assert result.data == ParsedAnswer(answer="validated")
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 5
    assert result.usage.reasoning_tokens == 4
    assert responses.request is not None
    response_format = responses.request["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    assert response_format["name"] == "ParsedAnswer"


@pytest.mark.asyncio
async def test_health_reports_effective_backend_without_exposing_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _clear_provider_keys(monkeypatch)
    secret = "sk-test-never-return-this"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "auto-activation.sqlite3",
        mock_step_delay_ms=0,
    )
    app = create_app(
        settings=settings,
        store=SQLiteRunStore(settings.database_path),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "hybrid"
    assert payload["configured_mode"] == "mock"
    assert payload["effective_mode"] == "hybrid"
    assert payload["mode_auto_activated"] is True
    assert payload["llm_backend"] == "openai"
    assert payload["openai_configured"] is True
    assert secret not in response.text
    assert secret not in repr(settings)
    assert secret not in str(settings.model_dump())
