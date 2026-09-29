"""Isolated policy contracts; no live research evidence or paid calls."""
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma3 import (
    MA3AssuranceAgent, _required_candidate_direction, _scoped_evidence_roles,
)
from drugpilot.agents.hypothesis_projection import build_hypothesis_projection
from drugpilot.domain.models import (
    AuditFinding, GateName, GateVerdictCode, HypothesisPacket, RunStage,
    Severity, ValidationResult,
)
from drugpilot.orchestration.next_actions import build_next_action_packet
from drugpilot.orchestration.automatic_resolution import summarize
from test_reviewer_provider import _fixture_ma1


def finding(*, deterministic=False, severity=Severity.HIGH, **kwargs):
    return AuditFinding(gate=GateName.G0, category="scientific", severity=severity,
        code="CONCERN" if deterministic else "REVIEWER_CONCERN",
        message="Synthetic contract concern, not research evidence.",
        artifact_id="hyp_test", claim_ids=["claim_test"], deterministic=deterministic,
        **kwargs)


@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.CRITICAL])
def test_unconfirmed_reviewer_severity_is_advisory_and_original_is_preserved(severity):
    raw = finding(severity=severity, failed_node="ma1.proposer")
    before = raw.model_dump_json()
    result = MA3AssuranceAgent()._decide(GateName.G0, [raw], [])
    saved = result.findings[0]
    assert result.verdict is GateVerdictCode.PASS and result.rework_order is None
    assert saved.reported_severity is severity
    assert saved.severity is saved.effective_severity is Severity.MEDIUM
    assert saved.confirmation_status == "unconfirmed"
    assert saved.confirmation_finding_ids == []
    assert saved.message == raw.message and raw.model_dump_json() == before


@pytest.mark.parametrize("code,node", [
    ("MISSING_PROVENANCE", "ma1.proposer"), ("OOD_HIGH", None),
    ("PLAN_UNVERIFIED_CONCERN", "planner"), ("UNAPPROVED_TOOL", None),
])
def test_reviewer_cannot_trigger_control_flow_through_code_or_node(code, node):
    raw = finding(failed_node=node).model_copy(update={"code": code})
    result = MA3AssuranceAgent()._decide(GateName.G0, [raw], [])
    assert result.verdict is GateVerdictCode.PASS


def test_confirmed_matching_rule_keeps_high_and_rework_contains_rule():
    rule = finding(deterministic=True, failed_node="ma1.proposer")
    opinion = finding(failed_node="planner")
    result = MA3AssuranceAgent()._decide(GateName.G0, [opinion, rule], [])
    assert result.verdict is GateVerdictCode.REVISE
    assert result.findings[1] is rule
    assert result.findings[0].confirmation_finding_ids == [rule.finding_id]
    assert result.findings[0].confirmation_status == "confirmed"
    assert result.findings[0].effective_severity is Severity.HIGH


def test_mixed_unconfirmed_and_real_violation_retains_authoritative_rework():
    opinion = finding(failed_node="planner")
    rule = finding(deterministic=True, failed_node="ma1.proposer").model_copy(
        update={"code": "OTHER_RULE", "message": "Actual rule violation."})
    result = MA3AssuranceAgent()._decide(GateName.G0, [opinion, rule], [])
    assert result.verdict is GateVerdictCode.REVISE
    assert result.rework_order.failed_node == "ma1.proposer"
    assert result.rework_order.required_actions == ["Actual rule violation."]


@pytest.mark.parametrize("changed", [{"artifact_id": "hyp_other"},
    {"claim_ids": ["claim_other"]}, {"gate": GateName.G1}, {"category": "schema"}])
def test_confirmation_is_exact_artifact_claim_gate_and_category_bound(changed):
    rule = finding(deterministic=True).model_copy(update=changed)
    result = MA3AssuranceAgent()._decide(GateName.G0, [rule, finding()], [])
    assert result.findings[1].confirmation_status == "unconfirmed"
    assert result.findings[1].effective_severity is Severity.MEDIUM


def test_self_declared_confirmation_cannot_escalate_gate():
    raw = finding(confirmation_status="confirmed", confirmation_finding_ids=["invented"],
                  effective_severity=Severity.CRITICAL)
    result = MA3AssuranceAgent()._decide(GateName.G0, [raw], [])
    assert result.verdict is GateVerdictCode.PASS
    assert result.findings[0].confirmation_status == "unconfirmed"


def test_failed_fixed_validation_still_requires_revision():
    validation = ValidationResult(name="fixed_invariant", passed=False, control_type="positive")
    result = MA3AssuranceAgent()._decide(GateName.G0, [finding()], [validation])
    assert result.verdict is GateVerdictCode.REVISE
    assert result.rework_order.required_actions == ["Repeat failed validation."]


def test_historical_findings_round_trip_without_new_fields_or_severity_changes():
    historical = finding()
    raw = historical.model_dump(mode="json")
    assert "confirmation_status" not in raw and "reported_severity" not in raw
    restored = AuditFinding.model_validate(raw)
    assert restored.model_dump(mode="json") == raw
    assert restored.severity is Severity.HIGH


def test_advisory_cannot_create_action_or_automatic_hold(brief):
    verdict = MA3AssuranceAgent()._decide(GateName.G0, [finding()], [])
    packet = build_next_action_packet(run_id="run_test", trigger_stage=RunStage.G0,
        brief=brief, verdict=verdict, candidates=[], checkpoint=None, terminal_message="Advisory only")
    assert packet is None
    outcome, reasons, remaining = summarize(SimpleNamespace(g1_verdict=verdict), [])
    assert outcome == "READY_FOR_REVIEW" and remaining == []
    assert "EXTERNAL_REVIEW_CONCERN_UNRESOLVED" not in reasons


@pytest.mark.parametrize("projection", [False, True])
def test_repeated_publication_roles_remain_hypothesis_claim_and_scope_local(brief, projection):
    original = _fixture_ma1(brief).hypotheses[0]
    general = original.model_copy(deep=True)
    scoped = original.model_copy(deep=True, update={"hypothesis_id": "hyp_scoped_test"})
    shared = general.evidence[0]
    scoped.evidence = [shared.model_copy(deep=True)]
    general.evidence = [shared]
    general.target.metadata["claim_evidence_assignment"] = {"claim_units": []}
    general.target.metadata.pop("molecular_hypothesis_scope", None)
    shared.metadata["linked_claim_ids"] = []
    shared.metadata["claim_linkage_roles"] = ["context_unlinked"]
    scoped.target.metadata["molecular_hypothesis_scope"] = {"label": "FLT3:ITD"}
    scoped.target.metadata["claim_evidence_assignment"] = {"claim_units": [{
        "claim_id": "claim_scoped", "evidence_id": shared.evidence_id,
        "role": "primary_target_disease_association", "molecular_scope_label": "FLT3:ITD"}]}
    scoped.evidence[0].metadata["linked_claim_ids"] = ["claim_scoped"]
    scoped.evidence[0].metadata["claim_linkage_roles"] = ["primary_target_disease_association"]
    packet = HypothesisPacket(brief_id=brief.brief_id, hypotheses=[general, scoped])
    before = packet.model_dump_json()
    rows = _scoped_evidence_roles({"hypothesis_packet": build_hypothesis_projection(packet) if projection else packet})
    assert len(rows) == 2
    assert rows[0]["molecular_scope"] is None and rows[0]["claim_roles"] == []
    assert rows[1]["molecular_scope"] == "FLT3:ITD"
    assert rows[1]["claim_roles"][0]["claim_id"] == "claim_scoped"
    assert packet.model_dump_json() == before


@pytest.mark.parametrize("requested", ["modulate", "unspecified"])
def test_broad_direction_is_bound_to_g0_hypothesis_not_candidate_assertion(brief, requested):
    hypotheses = _fixture_ma1(brief)
    brief.constraints.intervention_direction = requested
    h = hypotheses.hypotheses[0]
    candidate = SimpleNamespace(target=h.target, metadata={"direction_requested": "activate",
        "portfolio_hypothesis_id": h.hypothesis_id})
    assert _required_candidate_direction(brief, candidate, hypotheses) == "inhibit"
    candidate.metadata["portfolio_hypothesis_id"] = "not_a_retained_hypothesis"
    assert _required_candidate_direction(brief, candidate, hypotheses) is None


def test_ambiguous_opposite_directions_require_exact_hypothesis_link(brief):
    hypotheses = _fixture_ma1(brief)
    h = hypotheses.hypotheses[0]
    hypotheses.hypotheses = [h, h.model_copy(update={"hypothesis_id": "hyp_opposite", "intervention_direction": "activate"})]
    brief.constraints.intervention_direction = "modulate"
    candidate = SimpleNamespace(target=h.target, metadata={})
    assert _required_candidate_direction(brief, candidate, hypotheses) is None


@pytest.mark.parametrize("requested", ["modulate", "unspecified"])
def test_g0_independently_rejects_opposite_direction_with_intact_linkage(requested):
    from drugpilot.agents.ma1 import _expand_directional_hypotheses
    from drugpilot.agents.ma3 import _retained_primary_role_is_linked
    from test_qa_direction_resolution import brief, build, record
    b = brief(requested)
    h = _expand_directional_hypotheses(b, build(b, [record(
        "FLT3 inhibition reduced viability in acute myeloid leukemia cells.", requested)]))[0]
    assert h.intervention_direction == "inhibit"
    assert _retained_primary_role_is_linked(h, "direct_target_direction_primary", b)
    h.intervention_direction = "activate"
    assert not _retained_primary_role_is_linked(h, "direct_target_direction_primary", b)
    packet = HypothesisPacket(brief_id=b.brief_id, hypotheses=[h])
    findings = MA3AssuranceAgent()._audit_g0(b, packet)
    assert "RETAINED_DIRECT_TARGET_DIRECTION_PRIMARY_MISSING" in {f.code for f in findings}
