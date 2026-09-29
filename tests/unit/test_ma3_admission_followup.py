"""Read-only typed followup planning; synthetic inputs never enter real runs."""
from copy import deepcopy
from hashlib import sha256

import pytest

from drugpilot.domain.models import (
    AuditFinding, Claim, EvidenceRef, GateName, GateVerdict, GateVerdictCode,
    Hypothesis, HypothesisPacket, HypothesisTrack, ReworkOrder, Severity, TargetCandidate,
)
from drugpilot.orchestration.g0_followup import (
    build_plan, digest, evidence_work_fingerprint, ma3_followup_revision_context,
)


def _proposal(brief, symbol):
    text = f"Study of {symbol} in acute myeloid leukemia cells."
    return Hypothesis(hypothesis_id=f"hyp_{symbol}", track=HypothesisTrack.LITERATURE,
        disease=brief.disease, target=TargetCandidate(symbol=symbol), intervention_direction="inhibit",
        mechanism="Pending assessment", falsification_test="Measure a controlled outcome.",
        claims=[Claim(text="Unlinked prospective claim")], evidence=[EvidenceRef(
            source="PubMed", external_id=f"unit-{symbol}", title="Synthetic source",
            supports="context", excerpt_hash=sha256(text.encode()).hexdigest(),
            metadata={"excerpt": text, "provenance_class": "external_live"})])


def _packet(brief, statuses):
    proposals = [_proposal(brief, symbol) for symbol in statuses]
    report = {"policy": "ma3-source-bound-v1", "entries": [
        {"hypothesis_id": h.hypothesis_id, "target_symbol": h.target.symbol,
         "status": statuses[h.target.symbol], "reason": "Typed assessment",
         "original_proposal": h.model_dump(mode="json")} for h in proposals]}
    report["report_sha256"] = digest(report)
    return HypothesisPacket(brief_id=brief.brief_id, hypotheses=[], metadata={
        "admission_policy": "ma3-source-bound-v1", "ma3_admission_report": report,
        "rejected_screening_trace": {"entries": [{"target": {"symbol": h.target.symbol},
            "hypothesis_id": h.hypothesis_id, "reason_codes": ["MISSING_DIRECT_TARGET_DIRECTION_PRIMARY",
            "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION"]} for h in proposals]},
    })


def _reseal(packet):
    report = packet.metadata["ma3_admission_report"]
    report.pop("report_sha256", None)
    report["report_sha256"] = digest(report)


def _verdict():
    return GateVerdict(gate=GateName.G0, verdict=GateVerdictCode.REVISE,
        findings=[AuditFinding(gate=GateName.G0, category="evidence_provenance", severity=Severity.HIGH,
            code="NO_HYPOTHESIS", message="No approved hypotheses.", deterministic=True)],
        rationale="Source-bound review completed.", reviewer_model="unit",
        rework_order=ReworkOrder(gate=GateName.G0, target_stage="ma1", failed_node="ma1.proposer",
                                required_actions=["Review bounded source gaps."]))


@pytest.mark.parametrize("statuses", [
    {"FLT3": "rejected"}, {"FLT3": "hold"}, {"FLT3": "rejected", "BCL2": "hold"},
])
def test_scientific_rejection_and_hold_never_replay_legacy_critic_failures(brief, statuses):
    packet = _packet(brief, statuses)
    original = packet.model_dump_json()
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=4)
    assert not plan["eligible"] and not plan["tasks"]
    assert plan["reason_code"] == "MA3_ADMISSION_NO_RETRIEVABLE_EVIDENCE_GAP"
    assert plan["legacy_critic_trace_cannot_override_ma3"] is True
    assert packet.model_dump_json() == original


def test_mixed_report_only_schedules_needs_evidence_existing_target(brief):
    packet = _packet(brief, {"FLT3": "needs_evidence", "BCL2": "rejected", "IDH1": "hold"})
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=4)
    assert plan["eligible"]
    assert {task["target"] for task in plan["tasks"]} == {"FLT3"}
    assert {task["role"] for task in plan["tasks"]} == {"association", "direction"}
    assert all({path["source"] for path in task["source_paths"]} <= {"PubMed", "Europe PMC"}
               for task in plan["tasks"])


@pytest.mark.parametrize("error,role", [
    ("DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED", "direction"),
    ("PRIMARY_ASSOCIATION_NOT_ESTABLISHED", "association"),
])
def test_known_validation_error_only_requests_missing_role(brief, error, role):
    packet = _packet(brief, {"FLT3": "needs_evidence"})
    packet.metadata["ma3_admission_report"]["entries"][0]["validation_errors"] = [error]
    _reseal(packet)
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)
    assert {task["role"] for task in plan["tasks"]} == {role}


def test_missing_primary_role_uses_assessment_role_set(brief):
    packet = _packet(brief, {"FLT3": "needs_evidence"})
    row = packet.metadata["ma3_admission_report"]["entries"][0]
    row.update(validation_errors=["REQUIRED_PRIMARY_ROLE_MISSING"],
               assessment={"claims": [{"role": "direct_target_direction_primary"}]})
    _reseal(packet)
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)
    assert {task["role"] for task in plan["tasks"]} == {"association"}


def test_no_legacy_rejected_trace_needed_for_typed_source_gap(brief):
    packet = _packet(brief, {"FLT3": "needs_evidence"})
    packet.metadata["rejected_screening_trace"]["entries"] = []
    assert build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)["eligible"]


def test_report_integrity_and_proposal_identity_fail_closed(brief):
    packet = _packet(brief, {"FLT3": "hold"})
    packet.metadata["ma3_admission_report"]["entries"][0]["status"] = "needs_evidence"
    assert build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)["reason_code"] == "MA3_ADMISSION_REPORT_INVALID"
    _reseal(packet)
    packet.metadata["ma3_admission_report"]["entries"][0]["target_symbol"] = "INVENTED"
    _reseal(packet)
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)
    assert not plan["eligible"] and not plan["tasks"]
    assert plan["reason_code"] == "MA3_ADMISSION_REPORT_INVALID"


def test_missing_report_is_not_permission_to_replay_pending(brief):
    packet = _packet(brief, {"FLT3": "needs_evidence"})
    packet.metadata.pop("ma3_admission_report")
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)
    assert not plan["eligible"]
    assert plan["reason_code"] == "MA3_ADMISSION_REVIEW_NOT_COMPLETED"


def test_source_fingerprint_ignores_review_status_and_generated_reason(brief):
    packet = _packet(brief, {"FLT3": "needs_evidence"})
    original = evidence_work_fingerprint(brief, packet)
    row = packet.metadata["ma3_admission_report"]["entries"][0]
    row.update(status="rejected", reason="Different model wording", validation_errors=["OTHER"])
    _reseal(packet)
    assert evidence_work_fingerprint(brief, packet) == original
    row["original_proposal"]["evidence"][0]["excerpt_hash"] = "new-source-hash"
    _reseal(packet)
    assert evidence_work_fingerprint(brief, packet) != original


def test_pending_and_reviewed_original_source_fingerprints_match(brief):
    reviewed = _packet(brief, {"FLT3": "needs_evidence"})
    pending = reviewed.model_copy(deep=True)
    pending.provisional_hypotheses = [Hypothesis.model_validate(
        pending.metadata.pop("ma3_admission_report")["entries"][0]["original_proposal"])]
    assert evidence_work_fingerprint(brief, pending) == evidence_work_fingerprint(brief, reviewed)


def test_existing_source_attempt_budget_and_forbidden_controls_still_apply(brief):
    packet = _packet(brief, {"FLT3": "needs_evidence"})
    assert not build_plan(brief, packet, _verdict(), rework_loops=2, target_limit=1)["eligible"]
    first = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)
    packet.metadata["g0_followup_history"] = [{"attempted_task_keys": [task["task_key"] for task in first["tasks"]]}]
    assert not build_plan(brief, packet, _verdict(), rework_loops=1, target_limit=1)["eligible"]
    restricted = brief.model_copy(deep=True)
    restricted.constraints.forbidden_targets = ["FLT3"]
    assert not build_plan(restricted, packet, _verdict(), rework_loops=0, target_limit=1)["tasks"]


def test_ephemeral_retry_context_restores_only_typed_original_target_without_approval(brief):
    packet = _packet(brief, {"FLT3": "needs_evidence", "BCL2": "rejected", "IDH1": "hold"})
    original = packet.model_dump_json()
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)
    context = ma3_followup_revision_context(packet, plan)
    assert packet.model_dump_json() == original
    assert not context.hypotheses and not context.provisional_hypotheses
    excluded = {item.target_symbol.casefold(): item.reason_codes for item in context.excluded}
    assert "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY" in excluded["flt3"]
    assert excluded["bcl2"] == ["MA3_SCIENTIFIC_REJECTED"]
    assert excluded["idh1"] == ["MA3_ADMISSION_HOLD"]
    assert context.metadata["ma3_retry_context"]["targets"] == ["flt3"]
    assert context.metadata["ma1_rejected_screening_history"]["entries"][0]["target"]["symbol"] == "FLT3"


def test_typed_needs_evidence_target_is_actually_retrieved_after_catalog_window_moves(brief):
    from drugpilot.agents.ma1 import MA1Agent, TargetCatalogResult
    from drugpilot.config import Settings
    from drugpilot.domain.models import RunMode
    packet = _packet(brief, {"FLT3": "needs_evidence", "BCL2": "rejected"})
    plan = build_plan(brief, packet, _verdict(), rework_loops=0, target_limit=1)
    context = ma3_followup_revision_context(packet, plan)
    calls = []
    class Retriever:
        def retrieve(self, current_brief, seed, iteration):
            calls.append(seed.symbol)
            return _proposal(current_brief, seed.symbol).evidence
    agent = MA1Agent(settings=Settings(_env_file=None, mode=RunMode.MOCK,
        ma1_provisional_admission_enabled=True), evidence_retriever=Retriever())
    agent.g0_followup_targets = ["flt3"]
    agent._prepare_target_catalog = lambda current_brief, **kwargs: TargetCatalogResult(
        catalog={track: () for track in HypothesisTrack}, disease_resolution_status="resolved")
    output = agent.run(brief, context)
    assert calls == ["FLT3"]
    assert output.metadata["prior_evidence_retry_execution_summary"]["attempted_count"] == 1
    assert output.metadata["prior_evidence_retry_catalog_seed_restored_symbols"] == ["flt3"]
    assert output.hypotheses == []
    assert any(item.target.symbol == "FLT3" for item in output.provisional_hypotheses)
