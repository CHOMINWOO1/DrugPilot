from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode, RunSnapshot, RunStage, TokenUsage
from drugpilot.orchestration.graph import (
    DrugPilotOrchestrator,
    _intake_suggestion_from_output,
    _ma1_narrative_fallback,
    _planner_overlay_from_output,
)
from drugpilot.providers import LLMProviderCallError, LLMResult
from drugpilot.providers.structured_outputs import (
    DiseaseIntakeSuggestionOutput,
    HypothesisNarrativeOverlayOutput,
    IntakeFieldMismatchOutput,
    IntakeValueOutput,
    PlannerNarrativeOverlayOutput,
    PlannerTaskNarrativeOutput,
)
from drugpilot.storage.sqlite import SQLiteRunStore


def test_intake_value_pairs_round_trip_without_authority_fields() -> None:
    output = DiseaseIntakeSuggestionOutput(
        suggested_normalized_name="Acute myeloid leukemia",
        explanation="Unverified normalization suggestion only.",
        warnings=[],
        field_mismatches=[
            IntakeFieldMismatchOutput(
                code="FIELD_MISMATCH",
                message="Disease and goal wording differ.",
                fields=["disease", "research_goal"],
                values=[
                    IntakeValueOutput(field="disease", value="AML"),
                    IntakeValueOutput(field="research_goal", value="leukemia"),
                ],
            )
        ],
    )

    converted = _intake_suggestion_from_output(output)

    assert converted.field_mismatches[0].values == {
        "disease": "AML",
        "research_goal": "leukemia",
    }
    with pytest.raises(ValidationError):
        DiseaseIntakeSuggestionOutput.model_validate(
            {
                **output.model_dump(),
                "status": "valid",
                "resolved_disease_id": "EFO_FAKE",
                "targets": ["FLT3"],
            }
        )


def test_intake_duplicate_value_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate field names"):
        IntakeFieldMismatchOutput(
            code="FIELD_MISMATCH",
            message="Duplicate value fields are ambiguous.",
            fields=["disease"],
            values=[
                IntakeValueOutput(field="disease", value="AML"),
                IntakeValueOutput(field="disease", value="other"),
            ],
        )


class ReturnedProvider:
    def __init__(self, data) -> None:
        self.data = data

    async def structured(self, **_kwargs) -> LLMResult:
        return LLMResult(
            data=self.data,
            usage=TokenUsage(
                component="test",
                model="gpt-5.6-luna",
                input_tokens=20,
                output_tokens=5,
            ),
        )


class OneProviderRegistry:
    def __init__(self, provider: ReturnedProvider) -> None:
        self.provider = provider

    def for_component(self, _component: str) -> ReturnedProvider:
        return self.provider


async def _orchestrator(
    tmp_path,
    brief,
    data,
    *,
    auto_activated: bool,
) -> tuple[DrugPilotOrchestrator, list[str]]:
    store = SQLiteRunStore(tmp_path / f"boundary-{id(data)}.sqlite3")
    snapshot = RunSnapshot(
        run_id=f"run_boundary_{id(data)}",
        mode=RunMode.HYBRID,
        stage=RunStage.MA1,
        status_message="MA1",
        brief=brief,
    )
    await store.save_snapshot(snapshot)
    orchestrator = object.__new__(DrugPilotOrchestrator)
    orchestrator.settings = SimpleNamespace(
        effective_mode=RunMode.HYBRID,
        openai_auto_activated=auto_activated,
    )
    orchestrator.store = store
    orchestrator.providers = OneProviderRegistry(ReturnedProvider(data))
    events: list[str] = []

    async def trace(*args: Any, **_kwargs: Any) -> None:
        events.append(str(args[1]))

    orchestrator._trace_event = trace
    return orchestrator, events


@pytest.mark.asyncio
async def test_invalid_intake_advisory_uses_deterministic_nonfatal_fallback(
    tmp_path,
    brief,
) -> None:
    invalid = DiseaseIntakeSuggestionOutput.model_construct(
        suggested_normalized_name="x" * 301,
        explanation=None,
        warnings=[],
        field_mismatches=[],
    )
    fallback = DiseaseIntakeSuggestionOutput(
        suggested_normalized_name=None,
        explanation=None,
        warnings=[],
        field_mismatches=[],
    )
    orchestrator, events = await _orchestrator(
        tmp_path,
        brief,
        invalid,
        auto_activated=True,
    )

    result = await orchestrator._llm_enhance(
        f"run_boundary_{id(invalid)}",
        component="MA1-Intake",
        model="gpt-5.6-luna",
        response_model=DiseaseIntakeSuggestionOutput,
        fallback=fallback,
        prompt="return advisory JSON",
        max_output_tokens=100,
        failure_is_fatal=False,
        postprocess=_intake_suggestion_from_output,
    )

    assert result.suggested_normalized_name is None
    assert result.warnings == []
    assert "llm_fallback" in events
    restored = await orchestrator.store.get_snapshot(f"run_boundary_{id(invalid)}")
    assert restored is not None
    assert len(restored.token_usage) == 1


@pytest.mark.asyncio
async def test_invalid_ma1_ids_are_fatal_even_in_explicit_hybrid(
    tmp_path,
    brief,
) -> None:
    settings = Settings(
        mode=RunMode.MOCK,
        public_evidence_enabled=False,
        dynamic_target_discovery_enabled=False,
        dynamic_target_fallback_enabled=True,
        demo_fixture_fallback_enabled=True,
    )
    immutable = MA1Agent(settings).run(brief)
    invalid = _ma1_narrative_fallback(immutable)
    invalid.hypotheses[0].hypothesis_id = "hyp_unknown"
    fallback = _ma1_narrative_fallback(immutable)
    orchestrator, events = await _orchestrator(
        tmp_path,
        brief,
        invalid,
        auto_activated=False,
    )

    with pytest.raises(LLMProviderCallError, match="authority boundary"):
        await orchestrator._llm_enhance(
            f"run_boundary_{id(invalid)}",
            component="MA1",
            model="gpt-5.6-luna",
            response_model=HypothesisNarrativeOverlayOutput,
            fallback=fallback,
            prompt="return narrative JSON",
            max_output_tokens=1_000,
            postprocess=lambda output: orchestrator._reconcile_ma1_narrative(
                immutable,
                output,
            ),
            postprocess_failure_is_fatal=True,
        )

    assert "llm_call_failed" in events
    assert "llm_fallback" not in events


@pytest.mark.asyncio
async def test_invalid_planner_duplicates_are_fatal(
    tmp_path,
    brief,
) -> None:
    duplicate = PlannerTaskNarrativeOutput(
        task_id="task_1",
        acceptance_criteria=[],
        tool_purposes=[],
    )
    invalid = PlannerNarrativeOverlayOutput.model_construct(
        tasks=[duplicate, duplicate.model_copy(deep=True)],
        source_suggestions=[],
    )
    fallback = PlannerNarrativeOverlayOutput(tasks=[], source_suggestions=[])
    orchestrator, events = await _orchestrator(
        tmp_path,
        brief,
        invalid,
        auto_activated=False,
    )

    with pytest.raises(LLMProviderCallError, match="authority boundary"):
        await orchestrator._llm_enhance(
            f"run_boundary_{id(invalid)}",
            component="Planner",
            model="gpt-5.6-luna",
            response_model=PlannerNarrativeOverlayOutput,
            fallback=fallback,
            prompt="return planner JSON",
            max_output_tokens=1_000,
            postprocess=_planner_overlay_from_output,
            postprocess_failure_is_fatal=True,
        )

    assert "llm_call_failed" in events
    assert "llm_fallback" not in events
