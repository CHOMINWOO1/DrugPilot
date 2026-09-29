from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from drugpilot.agents.evidence_semantics import EvidenceMeaningBatch

from drugpilot.providers.llm import LLMProviderCallError, OpenAIProvider
from drugpilot.providers.structured_outputs import (
    DiseaseIntakeSuggestionOutput,
    HypothesisNarrativeOverlayOutput,
    PlannerNarrativeOverlayOutput,
    ReviewerAssessmentOutput,
    ReviewerRoutingAssessmentOutput,
)


LIVE_OUTPUTS = [
    EvidenceMeaningBatch(records=[]),
    ReviewerRoutingAssessmentOutput(summary="Typed routing without verdict authority.", additional_findings=[], uncertainties=[]),
    DiseaseIntakeSuggestionOutput(
        suggested_normalized_name=None,
        explanation=None,
        warnings=[],
        field_mismatches=[],
    ),
    HypothesisNarrativeOverlayOutput(hypotheses=[]),
    PlannerNarrativeOverlayOutput(tasks=[], source_suggestions=[]),
    ReviewerAssessmentOutput(
        summary="No additional concern.",
        additional_findings=[],
        uncertainties=[],
    ),
]


def _assert_strict_schema(value: object) -> None:
    if isinstance(value, dict):
        assert "default" not in value
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
            properties = value.get("properties", {})
            assert set(value.get("required", [])) == set(properties)
        if "additionalProperties" in value:
            assert value["additionalProperties"] is False
        for item in value.values():
            _assert_strict_schema(item)
    elif isinstance(value, list):
        for item in value:
            _assert_strict_schema(item)


@pytest.mark.parametrize("output", LIVE_OUTPUTS)
def test_all_live_dtos_are_openai_strict_schema_compatible(output: BaseModel) -> None:
    _assert_strict_schema(type(output).model_json_schema())


class FakeResponses:
    def __init__(
        self,
        *,
        output_text: str,
        status: str = "completed",
        output: list[Any] | None = None,
    ) -> None:
        self.output_text = output_text
        self.status = status
        self.output = output or []
        self.create_request: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.create_request = kwargs
        return SimpleNamespace(
            status=self.status,
            error=None,
            output=self.output,
            output_text=self.output_text,
            usage=SimpleNamespace(
                input_tokens=21,
                output_tokens=13,
                total_tokens=34,
                output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", LIVE_OUTPUTS)
async def test_all_live_dtos_dispatch_as_strict_schema(expected: BaseModel) -> None:
    responses = FakeResponses(output_text=expected.model_dump_json())
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))

    result = await provider.structured(
        component="strict-test",
        model="gpt-test",
        system="bounded worker",
        prompt="return the required object",
        response_model=type(expected),
        fallback=expected,
        max_output_tokens=100,
    )

    assert result.data == expected
    assert responses.create_request is not None
    response_format = responses.create_request["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    assert response_format["name"] == type(expected).__name__
    _assert_strict_schema(response_format["schema"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_text", "status", "match"),
    [
        ("", "completed", "no JSON output"),
        ("{not-json", "completed", "failed local validation"),
        ('{"hypotheses":[{"target":{}}]}', "completed", "failed local validation"),
        ('{"hypotheses":[]}', "incomplete", "non-completed status"),
    ],
)
async def test_strict_output_failure_is_usage_bearing_and_never_uses_fallback(
    output_text: str,
    status: str,
    match: str,
) -> None:
    responses = FakeResponses(output_text=output_text, status=status)
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    fallback = HypothesisNarrativeOverlayOutput(hypotheses=[])

    with pytest.raises(LLMProviderCallError, match=match) as captured:
        await provider.structured(
            component="ma1",
            model="gpt-test",
            system="bounded worker",
            prompt="return the overlay",
            response_model=HypothesisNarrativeOverlayOutput,
            fallback=fallback,
            max_output_tokens=100,
        )

    assert captured.value.usage is not None
    assert captured.value.usage.input_tokens == 21
    assert captured.value.usage.output_tokens == 10


@pytest.mark.asyncio
async def test_strict_output_refusal_fails_closed() -> None:
    refusal = SimpleNamespace(
        content=[SimpleNamespace(type="refusal")],
    )
    responses = FakeResponses(
        output_text="",
        output=[refusal],
    )
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    fallback = ReviewerAssessmentOutput(
        summary="fallback",
        additional_findings=[],
        uncertainties=[],
    )

    with pytest.raises(LLMProviderCallError, match="refused"):
        await provider.structured(
            component="ma3",
            model="gpt-test",
            system="read-only reviewer",
            prompt="return the assessment",
            response_model=ReviewerAssessmentOutput,
            fallback=fallback,
            max_output_tokens=100,
        )
