from __future__ import annotations

import pytest
from pydantic import ValidationError

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode
from drugpilot.orchestration.graph import (
    DrugPilotOrchestrator,
    PlannerNarrativeOverlay,
    PlannerTaskNarrative,
    _apply_planner_narrative_overlay,
    _planner_execution_contract,
)
from drugpilot.providers.structured_outputs import PlannerNarrativeOverlayOutput


def _fixture_ma1(brief):
    return MA1Agent(
        Settings(
            mode=RunMode.MOCK,
            public_evidence_enabled=False,
            dynamic_target_discovery_enabled=False,
            dynamic_target_fallback_enabled=True,
            demo_fixture_fallback_enabled=True,
        )
    ).run(brief)


def test_planner_overlay_schema_rejects_executable_fields() -> None:
    with pytest.raises(ValidationError):
        PlannerNarrativeOverlay.model_validate(
            {
                "tasks": [],
                "total_token_budget": 99_999_999,
                "approved_tools": ["unapproved_tool"],
            }
        )

    with pytest.raises(ValidationError):
        PlannerNarrativeOverlay.model_validate(
            {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "acceptance_criteria": [],
                        "tool_purposes": {},
                        "allowed_operations": ["arbitrary_operation"],
                    }
                ]
            }
        )


def test_narrative_merge_preserves_the_entire_execution_contract(brief) -> None:
    hypotheses = _fixture_ma1(brief)
    deterministic = PlannerAgent().run(hypotheses, brief)
    task = deterministic.tasks[1]
    assignment = task.tools[0]
    overlay = PlannerNarrativeOverlay(
        tasks=[
            PlannerTaskNarrative(
                task_id=task.task_id,
                acceptance_criteria=["Explain why the selected route is auditable."],
                tool_purposes={
                    assignment.tool_name: "Describe the persisted routing evidence.",
                    "unapproved_tool": "This unknown tool must be ignored.",
                },
            ),
            PlannerTaskNarrative(
                task_id="task_not_in_the_deterministic_plan",
                acceptance_criteria=["This task must be ignored."],
            ),
        ]
    )

    merged = _apply_planner_narrative_overlay(deterministic, overlay)

    assert _planner_execution_contract(merged) == _planner_execution_contract(
        deterministic
    )
    assert len(merged.tasks) == len(deterministic.tasks)
    merged_task = next(item for item in merged.tasks if item.task_id == task.task_id)
    assert merged_task.name == task.name
    assert set(task.acceptance_criteria) <= set(merged_task.acceptance_criteria)
    assert "Explain why the selected route is auditable." in (
        merged_task.acceptance_criteria
    )
    merged_tool = next(
        item for item in merged_task.tools if item.tool_name == assignment.tool_name
    )
    assert merged_tool.allowed_operations == assignment.allowed_operations
    assert merged_tool.required == assignment.required
    assert "Describe the persisted routing evidence." in merged_tool.purpose


@pytest.mark.asyncio
async def test_run_planner_uses_overlay_without_replacing_deterministic_plan(
    brief,
    tmp_path,
    monkeypatch,
) -> None:
    settings = Settings(
        mode=RunMode.MOCK,
        database_path=tmp_path / "planner-guard.sqlite3",
        public_evidence_enabled=False,
        dynamic_target_discovery_enabled=False,
        dynamic_target_fallback_enabled=True,
        demo_fixture_fallback_enabled=True,
        mock_step_delay_ms=0,
    )
    orchestrator = DrugPilotOrchestrator(settings=settings)
    hypotheses = MA1Agent(settings=settings).run(brief)
    hypotheses.metadata["transport_trace"] = (
        "DO_NOT_SEND_PLANNER_TRANSPORT_TRACE" * 100_000
    )
    for hypothesis in hypotheses.hypotheses:
        for evidence in hypothesis.evidence:
            evidence.metadata["transport_trace"] = (
                "DO_NOT_SEND_PLANNER_TRANSPORT_TRACE" * 100_000
            )
    g0 = await MA3AssuranceAgent().audit_g0(brief, hypotheses)
    deterministic = orchestrator.planner.run(hypotheses, brief)

    async def no_op(*args, **kwargs):
        return None

    async def narrative_only(*args, **kwargs):
        assert kwargs["response_model"] is PlannerNarrativeOverlayOutput
        prompt = kwargs["prompt"]
        assert "AUDITED_HYPOTHESIS_CONTEXT:" in prompt
        assert "DO_NOT_SEND_PLANNER_TRANSPORT_TRACE" not in prompt
        for hypothesis in hypotheses.hypotheses:
            assert hypothesis.hypothesis_id in prompt
            for claim in hypothesis.claims:
                assert claim.claim_id in prompt
        fallback = kwargs["fallback"].model_copy(deep=True)
        fallback.tasks[0].acceptance_criteria.append(
            "LLM-supplied explanatory criterion."
        )
        return kwargs["postprocess"](fallback)

    monkeypatch.setattr(orchestrator, "_transition", no_op)
    monkeypatch.setattr(orchestrator, "_artifact_transition", no_op)
    monkeypatch.setattr(orchestrator, "_llm_enhance", narrative_only)

    result = await orchestrator._run_planner(
        {
            "run_id": "run_planner_guard",
            "brief": brief,
            "hypothesis_packet": hypotheses,
            "g0_verdict": g0,
            "rework_loops": 0,
        }
    )
    final_plan = result["plan_spec"]

    assert _planner_execution_contract(final_plan) == _planner_execution_contract(
        deterministic
    )
    assert "LLM-supplied explanatory criterion." in (
        final_plan.tasks[0].acceptance_criteria
    )
