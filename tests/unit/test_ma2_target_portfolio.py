from __future__ import annotations

import asyncio

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    FaultType,
    GateVerdictCode,
    ScientificExecutionStatus,
)
from drugpilot.tools.executable import ScientificToolUnavailable
from drugpilot.tools.fixture_gateway import DeterministicFixtureGateway


class OneTargetUnavailableGateway(DeterministicFixtureGateway):
    def __init__(self, unavailable_symbol: str) -> None:
        self.unavailable_symbol = unavailable_symbol

    def normalize_target(self, target):
        if target.symbol == self.unavailable_symbol:
            raise ScientificToolUnavailable("target-specific executable profile missing")
        return super().normalize_target(target)


def _pipeline(brief):
    # This is an isolated deterministic portfolio component test.  Do not let
    # a developer server's live ``.env`` silently disable its fixture catalog.
    settings = Settings(
        _env_file=None,
        dynamic_target_discovery_enabled=False,
        dynamic_target_fallback_enabled=True,
        demo_fixture_fallback_enabled=True,
    )
    hypotheses = MA1Agent(settings=settings).run(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    return hypotheses, plan


def test_portfolio_attempts_every_unique_identity_resolved_target(brief) -> None:
    hypotheses, plan = _pipeline(brief)

    packet = MA2Agent(ma2_portfolio_enabled=True).run(plan, hypotheses)

    assert packet.portfolio_enabled is True
    assert packet.g0_approved_target_count == len(hypotheses.hypotheses)
    assert packet.eligible_target_count == len(hypotheses.hypotheses)
    assert packet.attempted_target_count == packet.eligible_target_count
    assert packet.completed_target_count == packet.eligible_target_count
    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len(packet.candidates) == (
        packet.eligible_target_count * plan.candidates_per_route
    )
    assert all(item.target_rank is not None for item in packet.candidates)
    assert sorted(item.portfolio_rank for item in packet.candidates) == list(
        range(1, len(packet.candidates) + 1)
    )
    assert packet.metadata["portfolio_ranking"][
        "cross_target_raw_docking_comparison"
    ] is False

    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.PASS, [
        (item.code, item.message) for item in verdict.findings
    ]


def test_partial_target_failure_is_retained_and_g1_abstains(brief) -> None:
    hypotheses, plan = _pipeline(brief)
    unavailable_symbol = hypotheses.hypotheses[-1].target.symbol

    packet = MA2Agent(
        gateway=OneTargetUnavailableGateway(unavailable_symbol),
        ma2_portfolio_enabled=True,
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.PARTIAL
    assert packet.completed_target_count == packet.eligible_target_count - 1
    failed = next(
        item
        for item in packet.target_executions
        if item.target.symbol == unavailable_symbol
    )
    assert failed.attempted is True
    assert failed.status == "unavailable"
    assert failed.reason_codes
    assert "PORTFOLIO_PARTIAL_TARGET_COVERAGE" in packet.abstain_reason_codes

    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.ABSTAIN, [
        (item.code, item.message) for item in verdict.findings
    ]
    assert "PORTFOLIO_PARTIAL_EXECUTION" in {
        item.code for item in verdict.findings
    }


def test_g1_blocks_target_scoped_route_attestation_tampering(brief) -> None:
    hypotheses, plan = _pipeline(brief)
    packet = MA2Agent(ma2_portfolio_enabled=True).run(
        plan, hypotheses
    ).model_copy(deep=True)
    summary = next(item for item in packet.target_executions if item.attempted)
    summary.route_decision["selected_route"] = (
        "de_novo" if summary.route.value == "repurposing" else "repurposing"
    )

    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )

    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "PORTFOLIO_ROUTE_POLICY_VIOLATION" in {
        item.code for item in verdict.findings
    }


def test_raw_docking_change_does_not_change_cross_target_portfolio_score(brief) -> None:
    hypotheses, plan = _pipeline(brief)
    agent = MA2Agent(ma2_portfolio_enabled=True)
    packet = agent.run(plan, hypotheses)
    original = packet.candidates[0]
    tampered = original.model_copy(
        update={
            "metrics": original.metrics.model_copy(
                update={"docking_score_kcal_mol": -99.0}
            )
        },
        deep=True,
    )
    from drugpilot.tools.portfolio_ranking import portfolio_components, portfolio_score

    target_cards = [
        item
        for item in packet.candidates
        if item.metadata["portfolio_target_key"]
        == original.metadata["portfolio_target_key"]
        and item.status != "excluded"
    ]
    kwargs = {
        "hypothesis_score": original.metadata["portfolio_hypothesis_score"],
        "eligible_count_for_target": len(target_cards),
        "target_best_score": max(item.score for item in target_cards),
    }
    assert portfolio_score(portfolio_components(original, **kwargs)) == portfolio_score(
        portfolio_components(tampered, **kwargs)
    )


def test_candidate_scoped_rework_reuses_unaffected_target_artifacts(brief) -> None:
    hypotheses, plan = _pipeline(brief)
    agent = MA2Agent(ma2_portfolio_enabled=True)
    faulty = agent.run(plan, hypotheses, fault=FaultType.G1_INVALID_SMILES)
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, faulty)
    )
    assert verdict.verdict is GateVerdictCode.REVISE
    assert verdict.rework_order is not None
    affected_id = verdict.rework_order.affected_candidate_ids[0]
    affected_key = next(
        item.metadata["portfolio_target_key"]
        for item in faulty.candidates
        if item.candidate_id == affected_id
    )
    untouched_cards = {
        item.candidate_id: tuple(item.tool_trace_ids)
        for item in faulty.candidates
        if item.metadata["portfolio_target_key"] != affected_key
    }
    untouched_call_ids = {
        call_id
        for summary in faulty.target_executions
        if summary.target_key != affected_key
        for call_id in summary.tool_call_ids
    }

    repaired = agent.rework(
        plan,
        hypotheses,
        faulty,
        verdict.rework_order,
    )

    repaired_by_id = {item.candidate_id: item for item in repaired.candidates}
    assert all(
        tuple(repaired_by_id[candidate_id].tool_trace_ids) == trace_ids
        for candidate_id, trace_ids in untouched_cards.items()
    )
    assert untouched_call_ids.issubset(set(repaired.reused_artifact_ids))
    assert set(untouched_cards).issubset(set(repaired.reused_artifact_ids))
