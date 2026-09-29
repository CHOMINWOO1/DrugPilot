from __future__ import annotations

from copy import deepcopy

import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    AuditFinding,
    GateName,
    GateVerdictCode,
    HypothesisPacket,
    Severity,
)


def _fixture_ma1() -> MA1Agent:
    """Keep this isolated G0 component suite explicit about fixture use."""

    return MA1Agent(
        settings=Settings(
            _env_file=None,
            dynamic_target_discovery_enabled=False,
            dynamic_target_fallback_enabled=True,
            demo_fixture_fallback_enabled=True,
        )
    )


@pytest.mark.parametrize(
    "code",
    [
        "UNAPPROVED_TOOL_OPERATION",
        "PLAN_TASK_TOOL_MISMATCH",
        "INACTIVE_ROUTE_TOOL_EXECUTED",
        "FORBIDDEN_DRUG_COMPUTED_AFTER_EXCLUSION",
        "MISSING_PLAN_TASK_TRACE",
    ],
)
def test_critical_process_governance_findings_block_immediately(code: str) -> None:
    finding = AuditFinding(
        gate=GateName.G1,
        category="execution",
        severity=Severity.CRITICAL,
        code=code,
        message="A critical execution-policy invariant was violated.",
        failed_node="ma2.tool_dispatch",
    )

    verdict = MA3AssuranceAgent()._decide(GateName.G1, [finding], [])

    assert verdict.verdict is GateVerdictCode.BLOCK
    assert verdict.rework_order is None


@pytest.mark.asyncio
async def test_g0_passes_evidence_linked_hypotheses(brief):
    packet = _fixture_ma1().run(brief)
    before = packet.model_dump_json()

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)

    assert verdict.verdict is GateVerdictCode.PASS
    assert packet.model_dump_json() == before, "MA3 must be read-only"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("removed_role", "expected_code"),
    [
        (
            "direct_target_direction_primary",
            "RETAINED_DIRECT_TARGET_DIRECTION_PRIMARY_MISSING",
        ),
        (
            "primary_target_disease_association",
            "RETAINED_PRIMARY_TARGET_DISEASE_ASSOCIATION_MISSING",
        ),
    ],
)
async def test_g0_independently_rejects_retained_missing_primary_role_linkage(
    brief,
    removed_role: str,
    expected_code: str,
):
    packet = _fixture_ma1().run(brief).model_copy(deep=True)
    hypothesis = packet.hypotheses[0]
    assignment = hypothesis.target.metadata["claim_evidence_assignment"]
    assignment["claim_units"] = [
        unit
        for unit in assignment["claim_units"]
        if unit["role"] != removed_role
    ]
    before = packet.model_dump_json()

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)

    findings = [
        finding
        for finding in verdict.findings
        if finding.code == expected_code
        and finding.artifact_id == hypothesis.hypothesis_id
    ]
    assert len(findings) == 1
    assert findings[0].severity is Severity.HIGH
    assert findings[0].failed_node == "ma1.claim_evidence_assignment"
    assert verdict.verdict is GateVerdictCode.REVISE
    assert packet.model_dump_json() == before, "MA3 must remain read-only"


@pytest.mark.asyncio
async def test_g0_does_not_trust_assignment_without_reconciled_evidence_link(brief):
    packet = _fixture_ma1().run(brief).model_copy(deep=True)
    hypothesis = packet.hypotheses[0]
    assignment = hypothesis.target.metadata["claim_evidence_assignment"]
    direct_unit = next(
        unit
        for unit in assignment["claim_units"]
        if unit["role"] == "direct_target_direction_primary"
    )
    direct_record = next(
        item
        for item in hypothesis.evidence
        if item.evidence_id == direct_unit["evidence_id"]
    )
    direct_record.metadata["claim_linkage_roles"] = []

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)

    assert "RETAINED_DIRECT_TARGET_DIRECTION_PRIMARY_MISSING" in {
        finding.code for finding in verdict.findings
    }


@pytest.mark.asyncio
async def test_g0_blocks_missing_provenance(brief):
    packet = _fixture_ma1().run(brief)
    broken = packet.model_copy(deep=True)
    broken.hypotheses[0].evidence = []

    verdict = await MA3AssuranceAgent().audit_g0(brief, broken)

    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "MISSING_PROVENANCE" in {finding.code for finding in verdict.findings}


@pytest.mark.asyncio
async def test_g0_distinguishes_completed_context_only_counter_search_from_unavailable(
    brief,
):
    packet = _fixture_ma1().run(brief).model_copy(deep=True)
    hypothesis = packet.hypotheses[0]
    hypothesis.evidence = [
        item for item in hypothesis.evidence if item.supports != "counter"
    ]
    hypothesis.target.metadata["counter_evidence_search"] = {
        "status": "completed_with_results",
        "raw_search_status": "completed_with_results",
        "real_counter_count": 0,
        "retrieved_resistance_context_count": 3,
        "fixture_used": False,
    }

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)
    codes = {finding.code for finding in verdict.findings}

    assert "COUNTER_EVIDENCE_NOT_FOUND" in codes
    assert "COUNTER_EVIDENCE_SEARCH_UNAVAILABLE" not in codes


@pytest.mark.asyncio
async def test_g0_rejects_missing_track_ranks(brief):
    packet = _fixture_ma1().run(brief).model_copy(deep=True)
    for hypothesis in packet.hypotheses:
        hypothesis.rank = None

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)

    rank_control = next(
        validation
        for validation in verdict.validations
        if validation.name == "rank_integrity_control"
    )
    assert rank_control.passed is False
    assert verdict.verdict is GateVerdictCode.REVISE


@pytest.mark.asyncio
async def test_g0_zero_hypotheses_cannot_vacuously_pass_diversity(brief):
    packet = HypothesisPacket(brief_id=brief.brief_id, hypotheses=[])

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)

    diversity = next(
        validation
        for validation in verdict.validations
        if validation.name == "target_diversity_control"
    )
    assert diversity.passed is False
    assert diversity.details["hypothesis_count"] == 0
    assert diversity.details["minimum_required"] == 1
    assert verdict.verdict is GateVerdictCode.REVISE


@pytest.mark.asyncio
async def test_g1_pass_and_direction_revision(brief):
    hypotheses = _fixture_ma1().run(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    candidates = MA2Agent().run(plan, hypotheses)
    assurance = MA3AssuranceAgent()

    passing = await assurance.audit_g1(brief, hypotheses, plan, candidates)
    assert passing.verdict is GateVerdictCode.PASS

    broken = candidates.model_copy(deep=True)
    broken.candidates[0].mechanism = "activate FLT3"
    revised = await assurance.audit_g1(brief, hypotheses, plan, broken)
    assert revised.verdict is GateVerdictCode.REVISE
    assert revised.rework_order is not None
    assert revised.rework_order.target_stage == "ma2"


@pytest.mark.asyncio
async def test_g1_records_conditional_admet_risk_without_failing_gate(brief):
    hypotheses = _fixture_ma1().run(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    candidates = MA2Agent().run(plan, hypotheses).model_copy(deep=True)
    candidate = next(
        item for item in candidates.candidates if item.status != "excluded"
    )

    scores = {
        "hepatotoxicity": 0.95,
        "herg_risk": 0.20,
        "ames_risk": 0.10,
        "cyp3a4_inhibition": 0.30,
        "clearance_risk": 0.25,
        "oral_bioavailability": 0.80,
    }
    predictive_flags = {
        "hepatotoxicity_high",
        "herg_risk_high",
        "ames_risk_high",
        "cyp3a4_inhibition_high",
        "clearance_risk_high",
    }
    candidate.metrics = candidate.metrics.model_copy(
        update={"admet_scores": scores}
    )
    assert candidate.admet_execution is not None
    candidate.admet_execution = candidate.admet_execution.model_copy(
        update={"scores": scores}
    )
    candidate.safety_flags = sorted(
        (set(candidate.safety_flags) - predictive_flags)
        | {"hepatotoxicity_high"}
    )
    candidate.status = "conditional"
    candidate.metadata["admet_decision_policy"] = (
        "prediction_as_warning_not_confirmatory_exclusion_v1"
    )
    candidate.metadata["admet_prediction_requires_experimental_confirmation"] = True

    verdict = await MA3AssuranceAgent().audit_g1(
        brief, hypotheses, plan, candidates
    )

    requirements = [
        finding
        for finding in verdict.findings
        if finding.code == "ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION"
        and finding.artifact_id == candidate.candidate_id
    ]
    assert len(requirements) == 1
    assert requirements[0].severity is Severity.MEDIUM
    assert "DILI probability (hepatotoxicity)=0.950" in requirements[0].message
    assert "human-hepatocyte" in requirements[0].message
    assert "not confirmatory safety exclusions" in requirements[0].message
    assert verdict.verdict is GateVerdictCode.PASS
    assert verdict.rework_order is None


@pytest.mark.asyncio
async def test_g1_does_not_treat_uncalibrated_descriptor_proxy_as_domain_proof(
    brief,
):
    hypotheses = _fixture_ma1().run(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    candidates = MA2Agent().run(plan, hypotheses).model_copy(deep=True)
    candidate = next(
        item for item in candidates.candidates if item.status != "excluded"
    )
    candidate.metrics = candidate.metrics.model_copy(update={"ood_score": 0.95})
    candidate.metadata["applicability_assessment"] = {
        "schema_version": "1.0",
        "method": "mw-clogp-tpsa-descriptor-distance-proxy-v1",
        "score": 0.95,
        "status": "high_descriptor_distance",
        "calibrated": False,
        "model_domain_claim": False,
        "reason_code": "UNCERTAINTY_NOT_CALIBRATED",
    }

    verdict = await MA3AssuranceAgent().audit_g1(
        brief,
        hypotheses,
        plan,
        candidates,
    )

    proxy_findings = [
        finding
        for finding in verdict.findings
        if finding.code == "OOD_PROXY_HIGH_REQUIRES_REVIEW"
        and finding.artifact_id == candidate.candidate_id
    ]
    assert len(proxy_findings) == 1
    assert proxy_findings[0].severity is Severity.MEDIUM
    assert "not proof" in proxy_findings[0].message
    assert not any(
        finding.code == "OOD_HIGH"
        and finding.artifact_id == candidate.candidate_id
        for finding in verdict.findings
    )
    assert verdict.verdict is GateVerdictCode.PASS
    assert verdict.rework_order is None


@pytest.mark.asyncio
async def test_g1_accepts_traceable_structural_exclusions_at_full_20_plus_20(brief):
    full_brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={"candidates_per_route": 20}
            )
        }
    )
    hypotheses = _fixture_ma1().run(full_brief)
    plan = PlannerAgent().run(hypotheses, full_brief)
    candidates = MA2Agent().run(plan, hypotheses)

    excluded = [candidate for candidate in candidates.candidates if candidate.status == "excluded"]
    assert excluded, "the full fixture should exercise MA2's exclusion path"
    assert all(candidate.exclusion_reasons for candidate in excluded)

    verdict = await MA3AssuranceAgent().audit_g1(
        full_brief, hypotheses, plan, candidates
    )

    assert verdict.verdict is GateVerdictCode.PASS


@pytest.mark.asyncio
async def test_g1_independently_blocks_tampered_route_attestation(brief):
    hypotheses = _fixture_ma1().run(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    candidates = MA2Agent().run(plan, hypotheses).model_copy(deep=True)
    candidates.metadata["route_decision"]["selected_route"] = "de_novo"
    candidates.metadata["route_decision"]["selected_routes"] = ["de_novo"]

    verdict = await MA3AssuranceAgent().audit_g1(
        brief, hypotheses, plan, candidates
    )

    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "ROUTE_POLICY_VIOLATION" in {
        finding.code for finding in verdict.findings
    }


@pytest.mark.asyncio
async def test_g1_accepts_a_traceable_forbidden_drug_exclusion_but_blocks_promotion(
    brief,
):
    constrained = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={"forbidden_drugs": ["RP-FLT3-001"]}
            )
        },
        deep=True,
    )
    hypotheses = _fixture_ma1().run(constrained)
    plan = PlannerAgent().run(hypotheses, constrained)
    candidates = MA2Agent().run(plan, hypotheses)
    excluded = next(
        item for item in candidates.candidates if item.name == "RP-FLT3-001"
    )

    passing = await MA3AssuranceAgent().audit_g1(
        constrained, hypotheses, plan, candidates
    )
    assert passing.verdict is GateVerdictCode.PASS
    assert excluded.rank is None

    tampered = candidates.model_copy(deep=True)
    promoted = next(
        item for item in tampered.candidates if item.candidate_id == excluded.candidate_id
    )
    promoted.status = "priority"
    promoted.rank = 1
    promoted.metadata["policy_eligible"] = True

    blocked = await MA3AssuranceAgent().audit_g1(
        constrained, hypotheses, plan, tampered
    )
    assert blocked.verdict is GateVerdictCode.BLOCK
    assert "FORBIDDEN_DRUG_NOT_EXCLUDED" in {
        finding.code for finding in blocked.findings
    }
