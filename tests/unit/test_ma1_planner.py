from __future__ import annotations

import asyncio

import pytest

from drugpilot.agents.ma1 import MA1Agent, MockEvidenceRetriever
from drugpilot.agents.planner import APPROVED_TOOLS, PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    BudgetPolicy,
    CandidateRoute,
    ExcludedHypothesis,
    HypothesisTrack,
    ProjectConstraints,
    ReservedControlCompound,
    NamespacedRegistryIdentifier,
    ResearchBrief,
    RunMode,
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        mode=RunMode.MOCK,
        database_path=tmp_path / "drugpilot.sqlite3",
        mock_step_delay_ms=0,
        max_run_tokens=1_000_000,
        hard_max_run_tokens=5_000_000,
        candidates_per_route=20,
    )


@pytest.fixture
def aml_brief() -> ResearchBrief:
    return ResearchBrief(
        brief_id="brief_test_aml",
        research_goal=(
            "Identify evidence-linked, falsifiable targets and dual-route drug "
            "candidates for acute myeloid leukemia."
        ),
        disease="Acute myeloid leukemia",
        constraints=ProjectConstraints(
            target_cell_type="AML blast",
            candidates_per_route=20,
            max_ma1_iterations=3,
            forbidden_targets=["BCL2"],
        ),
        budget=BudgetPolicy(max_run_tokens=300_000),
        human_priorities=["mechanistic diversity", "traceable evidence"],
    )


def test_ma1_runs_bounded_pipeline_with_separate_track_ranks(
    settings: Settings,
    aml_brief: ResearchBrief,
) -> None:
    packet = MA1Agent(settings=settings).run(aml_brief)

    assert packet.iterations == 3
    assert {hypothesis.track for hypothesis in packet.hypotheses} == {
        HypothesisTrack.LITERATURE,
        HypothesisTrack.TBIO,
    }
    for track in HypothesisTrack:
        ranks = [item.rank for item in packet.hypotheses if item.track is track]
        assert ranks == list(range(1, len(ranks) + 1))

    assert "BCL2" not in {item.target.symbol for item in packet.hypotheses}
    excluded = {item.target_symbol: item.reason_codes for item in packet.excluded}
    assert excluded["BCL2"] == ["FORBIDDEN_TARGET"]

    for hypothesis in packet.hypotheses:
        evidence_ids = {item.evidence_id for item in hypothesis.evidence}
        assert hypothesis.falsification_test
        assert any(item.supports == "counter" for item in hypothesis.evidence)
        assert all(item.metadata["mock"] is True for item in hypothesis.evidence)
        for claim in hypothesis.claims:
            assert claim.evidence_ids
            assert set(claim.evidence_ids) <= evidence_ids
            assert set(claim.counter_evidence_ids) <= evidence_ids


def test_ma1_is_deterministic_and_preserves_prior_exclusions_non_destructively(
    settings: Settings,
    aml_brief: ResearchBrief,
) -> None:
    agent = MA1Agent(settings=settings, evidence_retriever=MockEvidenceRetriever())
    prior = ExcludedHypothesis(
        target_symbol="FLT3",
        reason_codes=["HUMAN_EXCLUSION"],
        source_iteration=1,
    )
    context = {"excluded": [prior]}

    first = agent.run(aml_brief, revision_context=context)
    second = agent.run(aml_brief, revision_context=context)

    # Scientific content remains deterministic; execution timestamps must
    # describe each new packet, not reuse the brief creation timestamp.
    assert first.model_dump(exclude={"generated_at"}) == second.model_dump(exclude={"generated_at"})
    assert aml_brief.created_at <= first.generated_at <= second.generated_at
    assert context["excluded"] == [prior]
    assert "FLT3" not in {item.target.symbol for item in first.hypotheses}
    assert first.excluded[0].reason_codes == ["HUMAN_EXCLUSION"]


def test_ma1_async_wrapper(settings: Settings, aml_brief: ResearchBrief) -> None:
    packet = asyncio.run(MA1Agent(settings=settings).arun(aml_brief))
    assert packet.brief_id == aml_brief.brief_id


def test_mock_component_fixture_does_not_relax_live_fallback_boundary(
    tmp_path,
    aml_brief: ResearchBrief,
) -> None:
    shared = {
        "_env_file": None,
        "database_path": tmp_path / "isolated.sqlite3",
        "dynamic_target_discovery_enabled": True,
        "dynamic_target_fallback_enabled": False,
        "demo_fixture_fallback_enabled": False,
        "mock_step_delay_ms": 0,
    }
    mock_packet = MA1Agent(
        settings=Settings(mode=RunMode.MOCK, **shared)
    ).run(aml_brief)
    assert mock_packet.hypotheses
    assert mock_packet.metadata["target_fallback_used"] is True
    assert mock_packet.metadata["demo_fixture_fallback_authorized"] is False

    live_packet = MA1Agent(
        settings=Settings(
            mode=RunMode.LIVE,
            openai_api_key="test-key-never-used",
            **shared,
        )
    ).run(aml_brief)
    assert live_packet.hypotheses == []
    assert live_packet.metadata["target_fallback_used"] is False


def test_planner_preserves_structured_reserved_assay_controls(
    settings: Settings,
    aml_brief: ResearchBrief,
) -> None:
    control = ReservedControlCompound(
        name="Quizartinib positive control",
        registry_ids=[
            NamespacedRegistryIdentifier(
                namespace="ChEMBL",
                identifier="CHEMBL576982",
            )
        ],
    )
    brief = aml_brief.model_copy(
        update={
            "constraints": aml_brief.constraints.model_copy(
                update={"reserved_control_compounds": [control]},
                deep=True,
            )
        },
        deep=True,
    )
    hypotheses = MA1Agent(settings=settings).run(brief)

    plan = PlannerAgent(settings=settings).run(hypotheses, brief)

    assert plan.reserved_control_compounds == [control]
    ranking_task = next(task for task in plan.tasks if "Rank policy" in task.name)
    assert "Reserved assay controls remain control-only, excluded, and unranked" in (
        ranking_task.acceptance_criteria
    )
    candidate_judge = next(
        tool
        for tool in ranking_task.tools
        if tool.tool_name == "candidate_judge"
    )
    assert "screen_paper_native_v5_primary_source" in (
        candidate_judge.allowed_operations
    )
    assert "screen_paper_native_v6_primary_source" in (
        candidate_judge.allowed_operations
    )


def test_ma1_revision_keeps_existing_hypotheses_immutable(
    settings: Settings,
    aml_brief: ResearchBrief,
) -> None:
    agent = MA1Agent(settings=settings)
    original = agent.run(aml_brief)
    snapshot = original.model_dump()
    revised = agent.run(aml_brief, revision_context=original)

    assert original.model_dump() == snapshot
    original_ids = {item.hypothesis_id for item in original.hypotheses}
    assert original_ids <= {item.hypothesis_id for item in revised.hypotheses}
    assert len({item.hypothesis_id for item in revised.hypotheses}) == len(
        revised.hypotheses
    )


def test_planner_builds_strict_dual_route_dag_with_closed_tool_list(
    settings: Settings,
    aml_brief: ResearchBrief,
) -> None:
    packet = MA1Agent(settings=settings).run(aml_brief)
    plan = PlannerAgent(settings=settings).run(packet, aml_brief)

    assert plan.revision == 0
    assert plan.approved_tools == list(APPROVED_TOOLS)
    assert plan.total_token_budget == 300_000
    assert sum(task.max_tokens for task in plan.tasks) == plan.total_token_budget
    assert {task.route for task in plan.tasks if task.route is not None} == {
        CandidateRoute.REPURPOSING,
        CandidateRoute.DE_NOVO,
    }
    assert {tool.tool_name for task in plan.tasks for tool in task.tools} == set(
        APPROVED_TOOLS
    )
    assert any("LLM calls" in condition for condition in plan.stop_conditions)
    assert any("G1 PASS" in condition for condition in plan.stop_conditions)

    known: set[str] = set()
    for task in plan.tasks:
        assert set(task.depends_on) <= known
        known.add(task.task_id)

    chemistry = next(
        task for task in plan.tasks if task.name == "Validate and standardize molecular structures"
    )
    assert chemistry.dependency_policy == "any"
    assert len(chemistry.depends_on) == 2
    assert all(
        task.dependency_policy == "all"
        for task in plan.tasks
        if task.task_id != chemistry.task_id
    )


def test_planner_revision_context_increments_and_is_visible(
    settings: Settings,
    aml_brief: ResearchBrief,
) -> None:
    packet = MA1Agent(settings=settings).run(aml_brief)
    planner = PlannerAgent(settings=settings)
    initial = planner.run(packet, aml_brief)
    revised = planner.run(
        packet,
        aml_brief,
        revision_context={
            "previous_plan": initial,
            "required_actions": ["Re-run ADMET after unit normalization"],
            "additional_stop_conditions": ["Stop if the corrected unit is still ambiguous"],
        },
    )

    assert revised.revision == 1
    assert revised.plan_id != initial.plan_id
    assert any(
        "Revision directive retained" in criterion
        for task in revised.tasks
        for criterion in task.acceptance_criteria
    )
    assert revised.stop_conditions[-1] == "Stop if the corrected unit is still ambiguous"


def test_planner_rejects_packet_from_another_brief(
    settings: Settings,
    aml_brief: ResearchBrief,
) -> None:
    packet = MA1Agent(settings=settings).run(aml_brief)
    other = aml_brief.model_copy(update={"brief_id": "brief_other"})
    with pytest.raises(ValueError, match="does not belong"):
        PlannerAgent(settings=settings).run(packet, other)
