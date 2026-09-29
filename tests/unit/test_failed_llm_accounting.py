from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent, MA3_REVIEWER_SYSTEM_PROMPT
from drugpilot.domain.models import (
    BudgetPolicy,
    RunMode,
    RunSnapshot,
    RunStage,
    TokenUsage,
)
from drugpilot.orchestration.budget import BudgetExceeded
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.providers import (
    AnthropicProvider,
    LLMProviderCallError,
    LLMResult,
    usage_for_failed_call,
)
from drugpilot.providers.llm import (
    OpenAIProvider,
    _anthropic_usage,
    _openai_usage,
    conservative_structured_input_tokens,
    rough_tokens,
)
from drugpilot.providers.structured_outputs import (
    DiseaseIntakeSuggestionOutput,
    HypothesisNarrativeOverlayOutput,
    PlannerNarrativeOverlayOutput,
    ReviewerAssessmentOutput,
    ReviewerRoutingAssessmentOutput,
)
from drugpilot.storage.sqlite import SQLiteRunStore


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["openai", "anthropic"])
@pytest.mark.parametrize("failure", ["500", "timeout"])
async def test_sdk_transport_is_attempted_exactly_once(vendor, failure):
    import httpx
    calls = []
    def handler(request):
        calls.append(request)
        if failure == "timeout": raise httpx.ReadTimeout("test-only timeout", request=request)
        return httpx.Response(500, json={"error":{"type":"server_error","message":"test-only"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        if vendor == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key="test-only", http_client=http, max_retries=2)
            provider = OpenAIProvider(client=client)
        else:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key="test-only", http_client=http, max_retries=2)
            provider = AnthropicProvider(client=client)
        assert provider.client.max_retries == 0 and client.max_retries == 2
        with pytest.raises(LLMProviderCallError):
            await provider.structured(component="test-only", model="test-only", system="isolated",
                prompt="isolated", response_model=Answer, fallback=Answer(answer="unused"), max_output_tokens=10)
    assert len(calls) == 1


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    source_id: str
    caveats: list[str]


class SchemaHeavyAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    evidence: list[EvidenceItem]
    uncertainties: list[str]
    next_steps: list[str]


def test_conservative_structured_bound_includes_schema_bytes() -> None:
    system = "bounded worker"
    prompt = "return the required object"

    small = conservative_structured_input_tokens(
        system=system,
        prompt=prompt,
        response_model=Answer,
    )
    heavy = conservative_structured_input_tokens(
        system=system,
        prompt=prompt,
        response_model=SchemaHeavyAnswer,
    )

    assert small > rough_tokens(system + prompt)
    assert heavy > small
    assert heavy > len((system + prompt).encode("utf-8"))


@pytest.mark.parametrize(
    "response_model",
    [
        DiseaseIntakeSuggestionOutput,
        HypothesisNarrativeOverlayOutput,
        PlannerNarrativeOverlayOutput,
        ReviewerAssessmentOutput,
    ],
)
def test_authorized_live_response_schemas_leave_room_under_single_input_cap(
    response_model,
) -> None:
    bound = conservative_structured_input_tokens(
        system="bounded structured-output worker",
        prompt="{}",
        response_model=response_model,
    )

    assert bound < 250_000


def _provider_response(
    *,
    output_text: str,
    status: str = "completed",
    refusal: bool = False,
) -> SimpleNamespace:
    output = []
    if refusal:
        output = [
            SimpleNamespace(
                content=[SimpleNamespace(type="refusal")],
            )
        ]
    return SimpleNamespace(
        status=status,
        error=None,
        output=output,
        output_text=output_text,
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )


class ReturnedResponses:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.request = None
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        self.request = kwargs
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _provider_response(output_text="{not-json"),
        _provider_response(output_text='{"answer":"partial"}', status="incomplete"),
        _provider_response(output_text="", refusal=True),
    ],
    ids=["local-validation", "incomplete", "refusal"],
)
async def test_openai_response_failures_retain_usage(response) -> None:
    responses = ReturnedResponses(response)
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))

    with pytest.raises(LLMProviderCallError) as caught:
        await provider.structured(
            component="MA1",
            model="gpt-5.6-luna",
            system="bounded worker",
            prompt="return the object",
            response_model=Answer,
            fallback=Answer(answer="fallback"),
            max_output_tokens=100,
        )

    usage = caught.value.usage
    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 5
    assert usage.reasoning_tokens == 2
    assert responses.request["text"]["format"]["strict"] is True


class HTTPFailureResponses:
    async def create(self, **_kwargs):
        raise RuntimeError("transport failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("detail,expected", [
    ("max_output_tokens", "OUTPUT_LIMIT"),
    ("content_filter", "PROVIDER_INCOMPLETE"),
    ("untrusted-secret-text", "PROVIDER_INCOMPLETE"),
])
async def test_incomplete_reason_is_bounded_and_usage_is_retained(detail, expected):
    response = _provider_response(output_text='{"answer":"partial"}', status="incomplete")
    response.incomplete_details = SimpleNamespace(reason=detail)
    responses = ReturnedResponses(response)
    with pytest.raises(LLMProviderCallError) as caught:
        await OpenAIProvider(client=SimpleNamespace(responses=responses)).structured(
            component="unit", model="gpt-5.6-sol", system="isolated", prompt="isolated",
            response_model=Answer, fallback=Answer(answer="unused"), max_output_tokens=100)
    assert caught.value.reason_code == expected
    assert caught.value.usage.total_tokens == 18
    assert responses.calls == 1


def test_provider_reason_code_cannot_expose_untrusted_text():
    error = LLMProviderCallError("safe message", usage=None, reason_code="untrusted-secret-text")
    assert error.reason_code == "PROVIDER_CALL_FAILED"


@pytest.mark.asyncio
async def test_openai_transport_failure_carries_conservative_nonzero_usage() -> None:
    provider = OpenAIProvider(
        client=SimpleNamespace(responses=HTTPFailureResponses())
    )

    with pytest.raises(LLMProviderCallError) as caught:
        await provider.structured(
            component="Planner",
            model="gpt-5.6-luna",
            system="bounded worker",
            prompt="return the object",
            response_model=Answer,
            fallback=Answer(answer="fallback"),
            max_output_tokens=100,
        )

    marker = usage_for_failed_call(
        caught.value,
        component="Planner",
        model="gpt-5.6-luna",
    )
    assert marker is not None
    assert marker.input_tokens == conservative_structured_input_tokens(
        system="bounded worker",
        prompt="return the object",
        response_model=Answer,
    )
    assert marker.output_tokens == 100
    assert marker.total_tokens > 0
    assert marker.estimated is True


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"input_tokens": 1, "total_tokens": 2},
        {"output_tokens": 1, "total_tokens": 2},
        {"input_tokens": 1, "output_tokens": 1},
        {"input_tokens": 0, "output_tokens": 1, "total_tokens": 1},
        {"input_tokens": True, "output_tokens": 1, "total_tokens": 2},
        {"input_tokens": -1, "output_tokens": 1, "total_tokens": 0},
        {"input_tokens": 1.0, "output_tokens": 1, "total_tokens": 2},
        {"input_tokens": "1", "output_tokens": 1, "total_tokens": 2},
        {"input_tokens": 1, "output_tokens": True, "total_tokens": 2},
        {"input_tokens": 1, "output_tokens": -1, "total_tokens": 0},
        {"input_tokens": 1, "output_tokens": 1.0, "total_tokens": 2},
        {"input_tokens": 1, "output_tokens": "1", "total_tokens": 2},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": True},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": -1},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2.0},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": "2"},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 100},
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": True},
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": -1},
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1.0},
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": "1"},
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "output_tokens_details": "malformed",
        },
    ],
    ids=lambda value: repr(value),
)
def test_openai_usage_parser_rejects_missing_or_non_strict_counters(usage) -> None:
    assert (
        _openai_usage(
            SimpleNamespace(usage=usage),
            component="MA1",
            model="gpt-5.6-luna",
        )
        is None
    )


def test_openai_usage_parser_accepts_positive_input_and_zero_output() -> None:
    usage = _openai_usage(
        SimpleNamespace(
            usage={
                "input_tokens": 1,
                "output_tokens": 0,
                "total_tokens": 1,
                "output_tokens_details": None,
            }
        ),
        component="MA1",
        model="gpt-5.6-luna",
    )

    assert usage is not None
    assert usage.input_tokens == 1
    assert usage.output_tokens == 0
    assert usage.reasoning_tokens == 0


class RaisingUsageProperty:
    @property
    def usage(self):
        raise RuntimeError("malformed provider usage property")


@pytest.mark.parametrize(
    ("parser", "component", "model"),
    [
        (_openai_usage, "MA1", "gpt-5.6-luna"),
        (_anthropic_usage, "MA3-G0-Reviewer", "claude-opus-4-1"),
    ],
)
def test_usage_parsers_are_exception_safe_for_raising_properties(
    parser,
    component: str,
    model: str,
) -> None:
    assert parser(RaisingUsageProperty(), component=component, model=model) is None


@pytest.mark.asyncio
async def test_openai_completed_response_with_malformed_usage_is_conservative() -> None:
    system = "bounded worker"
    prompt = "return the object"
    responses = ReturnedResponses(
        SimpleNamespace(
            status="completed",
            error=None,
            output=[],
            output_text=Answer(answer="validated").model_dump_json(),
            usage={},
        )
    )
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))

    result = await provider.structured(
        component="MA1",
        model="gpt-5.6-luna",
        system=system,
        prompt=prompt,
        response_model=Answer,
        fallback=Answer(answer="fallback"),
        max_output_tokens=100,
    )

    assert result.data == Answer(answer="validated")
    expected_input = conservative_structured_input_tokens(
        system=system,
        prompt=prompt,
        response_model=Answer,
    )
    assert expected_input > rough_tokens(system + prompt)
    assert result.usage.input_tokens == expected_input
    assert result.usage.output_tokens == 100
    assert result.usage.estimated is True


class ReturnedMessages:
    async def create(self, **_kwargs):
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="{not-json")],
            usage=SimpleNamespace(
                input_tokens=17,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=2,
                output_tokens=6,
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        )


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"input_tokens": 1},
        {"output_tokens": 1},
        {"input_tokens": 0, "output_tokens": 1},
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": 1.0, "output_tokens": 1},
        {"input_tokens": "1", "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": True},
        {"input_tokens": 1, "output_tokens": -1},
        {"input_tokens": 1, "output_tokens": 1.0},
        {"input_tokens": 1, "output_tokens": "1"},
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_creation_input_tokens": True,
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_creation_input_tokens": -1,
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": 1.0,
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": "1",
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": True},
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "output_tokens_details": "malformed",
        },
    ],
    ids=lambda value: repr(value),
)
def test_anthropic_usage_parser_rejects_missing_or_non_strict_counters(usage) -> None:
    assert (
        _anthropic_usage(
            SimpleNamespace(usage=usage),
            component="MA3-G0-Reviewer",
            model="claude-opus-4-1",
        )
        is None
    )


def test_anthropic_usage_parser_accepts_strict_optional_counters() -> None:
    usage = _anthropic_usage(
        SimpleNamespace(
            usage={
                "input_tokens": 1,
                "output_tokens": 0,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
                "output_tokens_details": None,
            }
        ),
        component="MA3-G0-Reviewer",
        model="claude-opus-4-1",
    )

    assert usage is not None
    assert usage.input_tokens == 6
    assert usage.output_tokens == 0
    assert usage.reasoning_tokens == 0


class AnthropicTransportFailureMessages:
    def __init__(self) -> None:
        self.calls = 0
        self.request = None

    async def create(self, **kwargs):
        self.calls += 1
        self.request = kwargs
        raise RuntimeError("transport failed")


@pytest.mark.asyncio
async def test_anthropic_transport_failure_carries_conservative_nonzero_usage() -> None:
    system = "read only"
    prompt = "return the assessment"
    provider = AnthropicProvider(
        client=SimpleNamespace(messages=AnthropicTransportFailureMessages())
    )
    fallback = ReviewerAssessmentOutput(
        summary="fallback",
        additional_findings=[],
        uncertainties=[],
    )

    with pytest.raises(LLMProviderCallError) as caught:
        await provider.structured(
            component="MA3-G0-Reviewer",
            model="claude-opus-4-1",
            system=system,
            prompt=prompt,
            response_model=ReviewerAssessmentOutput,
            fallback=fallback,
            max_output_tokens=100,
        )

    usage = caught.value.usage
    assert usage is not None
    assert usage.input_tokens == conservative_structured_input_tokens(
        system=system,
        prompt=prompt,
        response_model=ReviewerAssessmentOutput,
    )
    assert usage.output_tokens == 100
    assert usage.estimated is True


class AnthropicMalformedUsageMessages:
    async def create(self, **_kwargs):
        assessment = ReviewerAssessmentOutput(
            summary="validated",
            additional_findings=[],
            uncertainties=[],
        )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=assessment.model_dump_json())],
            usage={},
        )


@pytest.mark.asyncio
async def test_anthropic_completed_response_with_malformed_usage_is_conservative() -> None:
    system = "read only"
    prompt = "return the assessment"
    provider = AnthropicProvider(
        client=SimpleNamespace(messages=AnthropicMalformedUsageMessages())
    )
    fallback = ReviewerAssessmentOutput(
        summary="fallback",
        additional_findings=[],
        uncertainties=[],
    )

    result = await provider.structured(
        component="MA3-G0-Reviewer",
        model="claude-opus-4-1",
        system=system,
        prompt=prompt,
        response_model=ReviewerAssessmentOutput,
        fallback=fallback,
        max_output_tokens=100,
    )

    assert result.data.summary == "validated"
    expected_input = conservative_structured_input_tokens(
        system=system,
        prompt=prompt,
        response_model=ReviewerAssessmentOutput,
    )
    assert expected_input > rough_tokens(system + prompt)
    assert result.usage.input_tokens == expected_input
    assert result.usage.output_tokens == 100
    assert result.usage.estimated is True


@pytest.mark.asyncio
async def test_anthropic_validation_failure_retain_usage() -> None:
    provider = AnthropicProvider(
        client=SimpleNamespace(messages=ReturnedMessages())
    )
    fallback = ReviewerAssessmentOutput(
        summary="fallback",
        additional_findings=[],
        uncertainties=[],
    )

    with pytest.raises(LLMProviderCallError) as caught:
        await provider.structured(
            component="MA3-G0-Reviewer",
            model="claude-opus-4-1",
            system="read only",
            prompt="return the assessment",
            response_model=ReviewerAssessmentOutput,
            fallback=fallback,
            max_output_tokens=100,
        )

    usage = caught.value.usage
    assert usage is not None
    assert usage.input_tokens == 22
    assert usage.output_tokens == 5
    assert usage.reasoning_tokens == 1


class FailingProvider:
    def __init__(self, usage: TokenUsage | None) -> None:
        self.usage = usage
        self.calls = 0

    async def structured(self, **_kwargs):
        self.calls += 1
        raise LLMProviderCallError("provider response rejected", usage=self.usage)


class OneProviderRegistry:
    def __init__(self, provider: FailingProvider) -> None:
        self.provider = provider

    def for_component(self, _component: str) -> FailingProvider:
        return self.provider


def _orchestrator_for_provider(
    *,
    store: SQLiteRunStore,
    provider: FailingProvider,
    auto_activated: bool,
) -> DrugPilotOrchestrator:
    orchestrator = object.__new__(DrugPilotOrchestrator)
    orchestrator.settings = SimpleNamespace(
        effective_mode=RunMode.HYBRID,
        openai_auto_activated=auto_activated,
    )
    orchestrator.store = store
    orchestrator.providers = OneProviderRegistry(provider)

    async def ignore_trace(*_args, **_kwargs) -> None:
        return None

    orchestrator._trace_event = ignore_trace
    return orchestrator


@pytest.mark.asyncio
async def test_fatal_paid_failure_is_priced_and_persisted_once(tmp_path, brief) -> None:
    store = SQLiteRunStore(tmp_path / "fatal-paid.sqlite3")
    snapshot = RunSnapshot(
        run_id="run_fatal_paid",
        mode=RunMode.HYBRID,
        stage=RunStage.MA1,
        status_message="MA1",
        brief=brief,
    )
    await store.save_snapshot(snapshot)
    provider = FailingProvider(
        TokenUsage(
            component="MA1",
            model="gpt-5.6-luna",
            input_tokens=100,
            output_tokens=20,
            reasoning_tokens=5,
        )
    )
    orchestrator = _orchestrator_for_provider(
        store=store,
        provider=provider,
        auto_activated=True,
    )

    with pytest.raises(LLMProviderCallError):
        await orchestrator._llm_enhance(
            snapshot.run_id,
            component="MA1",
            model="gpt-5.6-luna",
            response_model=Answer,
            fallback=Answer(answer="fallback"),
            prompt="return the object",
            max_output_tokens=100,
        )

    restored = await store.get_snapshot(snapshot.run_id)
    assert restored is not None
    assert len(restored.token_usage) == 1
    assert restored.token_usage[0].estimated_cost_usd == pytest.approx(0.00005)
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_success_without_exact_usage_is_not_adopted_before_durable_stop(
    tmp_path,
    brief,
) -> None:
    store = SQLiteRunStore(tmp_path / "success-unknown-usage.sqlite3")
    snapshot = RunSnapshot(
        run_id="run_success_unknown_usage",
        mode=RunMode.HYBRID,
        stage=RunStage.MA1,
        status_message="MA1",
        brief=brief,
    )
    await store.save_snapshot(snapshot)
    responses = ReturnedResponses(
        SimpleNamespace(
            status="completed",
            error=None,
            output=[],
            output_text=Answer(answer="live result").model_dump_json(),
            usage={},
        )
    )
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    orchestrator = _orchestrator_for_provider(
        store=store,
        provider=provider,
        auto_activated=False,
    )

    with pytest.raises(BudgetExceeded, match="live LLM usage is estimated"):
        await orchestrator._llm_enhance(
            snapshot.run_id,
            component="MA1",
            model="gpt-5.6-luna",
            response_model=Answer,
            fallback=Answer(answer="fallback"),
            prompt="return the object",
            max_output_tokens=100,
        )

    restored = await store.get_snapshot(snapshot.run_id)
    assert restored is not None
    assert len(restored.token_usage) == 1
    recorded = restored.token_usage[0]
    assert recorded.estimated is True
    assert recorded.input_tokens == conservative_structured_input_tokens(
        system=(
            "You are a bounded DrugPilot worker. Follow the supplied JSON schema, "
            "preserve provenance, and never invent a tool result."
        ),
        prompt="return the object",
        response_model=Answer,
    )
    assert recorded.total_tokens > 100
    assert recorded.estimated_cost_usd > 0
    assert responses.calls == 1

    with pytest.raises(BudgetExceeded, match="prior live LLM usage is estimated"):
        await orchestrator._llm_enhance(
            snapshot.run_id,
            component="Planner",
            model="gpt-5.6-luna",
            response_model=Answer,
            fallback=Answer(answer="fallback"),
            prompt="do not dispatch",
            max_output_tokens=100,
        )
    assert responses.calls == 1


@pytest.mark.asyncio
async def test_graph_preflight_reserves_full_schema_bound_before_dispatch(
    tmp_path,
    brief,
) -> None:
    system = (
        "You are a bounded DrugPilot worker. Follow the supplied JSON schema, "
        "preserve provenance, and never invent a tool result."
    )
    prompt = "return the object"
    reserved_input = conservative_structured_input_tokens(
        system=system,
        prompt=prompt,
        response_model=SchemaHeavyAnswer,
    )
    assert reserved_input > 1_000
    bounded_brief = brief.model_copy(
        update={
            "budget": brief.budget.model_copy(
                update={"max_single_input_tokens": reserved_input - 1},
                deep=True,
            )
        },
        deep=True,
    )
    store = SQLiteRunStore(tmp_path / "schema-preflight.sqlite3")
    snapshot = RunSnapshot(
        run_id="run_schema_preflight",
        mode=RunMode.HYBRID,
        stage=RunStage.MA1,
        status_message="MA1",
        brief=bounded_brief,
    )
    await store.save_snapshot(snapshot)
    unused = SchemaHeavyAnswer(
        summary="unused",
        evidence=[],
        uncertainties=[],
        next_steps=[],
    )
    responses = ReturnedResponses(_provider_response(output_text=unused.model_dump_json()))
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    orchestrator = _orchestrator_for_provider(
        store=store,
        provider=provider,
        auto_activated=False,
    )

    with pytest.raises(BudgetExceeded, match="single input cap"):
        await orchestrator._llm_enhance(
            snapshot.run_id,
            component="MA1",
            model="gpt-5.6-luna",
            response_model=SchemaHeavyAnswer,
            fallback=unused,
            prompt=prompt,
            max_output_tokens=100,
        )

    assert responses.calls == 0


@pytest.mark.asyncio
async def test_unknown_http_usage_is_persisted_then_immediately_stops_run(
    tmp_path,
    brief,
) -> None:
    store = SQLiteRunStore(tmp_path / "unknown-http.sqlite3")
    limited_brief = brief.model_copy(
        update={
            "budget": BudgetPolicy(
                max_run_tokens=100_000,
                hard_max_run_tokens=120_000,
                max_llm_calls=5,
                max_single_input_tokens=50_000,
                max_run_cost_usd=1,
            )
        },
        deep=True,
    )
    snapshot = RunSnapshot(
        run_id="run_unknown_http",
        mode=RunMode.HYBRID,
        stage=RunStage.MA1,
        status_message="MA1",
        brief=limited_brief,
    )
    await store.save_snapshot(snapshot)
    provider = FailingProvider(None)
    orchestrator = _orchestrator_for_provider(
        store=store,
        provider=provider,
        auto_activated=False,
    )

    with pytest.raises(BudgetExceeded, match="live LLM usage is estimated"):
        await orchestrator._llm_enhance(
            snapshot.run_id,
            component="MA1-Intake",
            model="gpt-5.6-luna",
            response_model=Answer,
            fallback=Answer(answer="fallback"),
            prompt="return the object",
            max_output_tokens=100,
            failure_is_fatal=False,
        )

    restored = await store.get_snapshot(snapshot.run_id)
    assert restored is not None
    assert len(restored.token_usage) == 1
    recorded = restored.token_usage[0]
    assert recorded.input_tokens == conservative_structured_input_tokens(
        system=(
            "You are a bounded DrugPilot worker. Follow the supplied JSON schema, "
            "preserve provenance, and never invent a tool result."
        ),
        prompt="return the object",
        response_model=Answer,
    )
    assert recorded.output_tokens == 100
    assert recorded.total_tokens > 0
    assert recorded.estimated_cost_usd > 0
    assert recorded.estimated is True

    with pytest.raises(BudgetExceeded, match="prior live LLM usage is estimated"):
        await orchestrator._llm_enhance(
            snapshot.run_id,
            component="MA1",
            model="gpt-5.6-luna",
            response_model=Answer,
            fallback=Answer(answer="fallback"),
            prompt="return the object",
            max_output_tokens=100,
        )
    assert provider.calls == 1


class ReviewerFailureProvider:
    def __init__(self, usage: TokenUsage) -> None:
        self.usage = usage

    async def structured(self, **_kwargs):
        raise LLMProviderCallError("reviewer response rejected", usage=self.usage)


class InvalidReviewerData(BaseModel):
    unrelated: str


class ReviewerConversionFailureProvider:
    def __init__(self, usage: TokenUsage) -> None:
        self.usage = usage

    async def structured(self, **_kwargs):
        return LLMResult(
            data=InvalidReviewerData(unrelated="not an assessment"),
            usage=self.usage,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_factory",
    [ReviewerFailureProvider, ReviewerConversionFailureProvider],
    ids=["provider-failure", "dto-domain-conversion"],
)
async def test_ma3_records_usage_once_before_optional_reviewer_fallback(
    brief,
    provider_factory,
) -> None:
    packet = MA1Agent().run(brief)
    paid = TokenUsage(
        component="MA3-G0-Reviewer",
        model="gpt-5.6-luna",
        input_tokens=31,
        output_tokens=9,
    )
    seen: list[TokenUsage] = []
    assurance = MA3AssuranceAgent(
        reviewer_model="gpt-5.6-luna",
        reviewer_provider=provider_factory(paid),
    )

    async def record(usage: TokenUsage) -> None:
        seen.append(usage)

    verdict = await assurance.audit_g0(brief, packet, on_usage=record)

    assert len(seen) == 1
    assert seen[0].total_tokens == 40
    assert verdict.reviewer_invoked is False
    assert "REVIEWER_UNAVAILABLE" in {item.code for item in verdict.findings}


@pytest.mark.asyncio
async def test_ma3_failed_call_postflight_is_not_swallowed(brief) -> None:
    packet = MA1Agent().run(brief)
    paid = TokenUsage(
        component="MA3-G0-Reviewer",
        model="gpt-5.6-luna",
        input_tokens=31,
        output_tokens=9,
    )
    assurance = MA3AssuranceAgent(
        reviewer_model="gpt-5.6-luna",
        reviewer_provider=ReviewerFailureProvider(paid),
    )

    async def deny_postflight(_usage: TokenUsage) -> None:
        raise BudgetExceeded("postflight denied")

    with pytest.raises(BudgetExceeded, match="postflight denied"):
        await assurance.audit_g0(brief, packet, on_usage=deny_postflight)


@pytest.mark.asyncio
async def test_ma3_unknown_live_usage_is_durably_recorded_before_stop(
    tmp_path,
    brief,
) -> None:
    bounded_brief = brief.model_copy(
        update={
            "budget": brief.budget.model_copy(
                update={"max_llm_calls": 5},
                deep=True,
            )
        },
        deep=True,
    )
    packet = MA1Agent().run(bounded_brief)
    store = SQLiteRunStore(tmp_path / "reviewer-unknown-usage.sqlite3")
    snapshot = RunSnapshot(
        run_id="run_reviewer_unknown_usage",
        mode=RunMode.HYBRID,
        stage=RunStage.G0,
        status_message="G0",
        brief=bounded_brief,
    )
    await store.save_snapshot(snapshot)
    unused_provider = FailingProvider(None)
    orchestrator = _orchestrator_for_provider(
        store=store,
        provider=unused_provider,
        auto_activated=False,
    )
    messages = AnthropicTransportFailureMessages()
    assurance = MA3AssuranceAgent(
        reviewer_model="claude-opus-4-1",
        reviewer_provider=AnthropicProvider(
            client=SimpleNamespace(messages=messages)
        ),
    )
    reservations: list[int] = []
    async def persist_and_check(usage: TokenUsage) -> None:
        await orchestrator._record_llm_usage(snapshot.run_id, usage)

    async def reserve_and_preflight(
        model: str,
        prompt: str,
        max_output_tokens: int,
    ) -> None:
        reservations.append(
                conservative_structured_input_tokens(
                    system=MA3_REVIEWER_SYSTEM_PROMPT,
                prompt=prompt,
                response_model=ReviewerRoutingAssessmentOutput,
            )
        )
        await orchestrator._preflight_llm(
            snapshot.run_id,
            model,
            prompt,
            max_output_tokens,
        )

    with pytest.raises(BudgetExceeded, match="live LLM usage is estimated"):
        await assurance.audit_g0(
            bounded_brief,
            packet,
            on_usage=persist_and_check,
            before_review=reserve_and_preflight,
        )

    restored = await store.get_snapshot(snapshot.run_id)
    assert restored is not None
    assert len(restored.token_usage) == 1
    recorded = restored.token_usage[0]
    assert recorded.model == "claude-opus-4-1"
    assert recorded.estimated is True
    assert reservations == [recorded.input_tokens]
    assert recorded.total_tokens > 4_000
    assert recorded.estimated_cost_usd > 0
    assert messages.calls == 1
    assert unused_provider.calls == 0
