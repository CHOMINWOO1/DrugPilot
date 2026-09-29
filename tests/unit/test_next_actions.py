from __future__ import annotations

import base64
from hashlib import sha256

import pytest

from drugpilot.domain.models import (
    AuditFinding,
    CandidateCard,
    CandidateRoute,
    GateName,
    GateVerdict,
    GateVerdictCode,
    ExperimentalResultRecord,
    HumanDecisionCode,
    HumanGateDecision,
    ReworkOrder,
    RunMode,
    RunSnapshot,
    RunStage,
    Severity,
    TargetCandidate,
)
from drugpilot.config import Settings
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.orchestration.next_actions import build_next_action_packet


@pytest.mark.parametrize("message", [
    "The excerpt describes a breast cancer experiment, not the requested lung cancer model.",
    "Two evidence IDs repeat the same publication, same experiment and viability result.",
    "The measured response was blocked by fibroblasts; this is not favorable inhibition.",
])
def test_untyped_reviewer_interpretation_defects_do_not_prescribe_experiments(message):
    from drugpilot.orchestration.next_actions import _kind_for
    finding = AuditFinding(gate=GateName.G0, category="scientific", severity=Severity.HIGH,
        code="REVIEWER_GENERAL_CONTEXT_PROBLEM", message=message, deterministic=False)
    assert _kind_for(finding) == "data_required"


def test_typed_remediation_is_not_based_on_incidental_experimental_words():
    from drugpilot.orchestration.next_actions import _kind_for
    finding = AuditFinding(gate=GateName.G0, category="scientific", severity=Severity.HIGH,
        code="REVIEWER_GENERAL_CONTEXT_PROBLEM", message="An experiment is misclassified.",
        deterministic=False, remediation_kind="computational_rework")
    assert _kind_for(finding) == "computational_rework"
    finding.message = "Artifact hash mismatch in an experiment."
    assert _kind_for(finding) == "integrity_block"


def _candidate(candidate_id: str, name: str) -> CandidateCard:
    return CandidateCard(
        candidate_id=candidate_id,
        route=CandidateRoute.REPURPOSING,
        name=name,
        target=TargetCandidate(symbol="JAK2"),
        mechanism="inhibit",
        smiles="CCO",
        score=0.8,
        rank=1,
    )


def test_g1_findings_become_mixed_actions_without_synthetic_results(brief) -> None:
    findings = [
        AuditFinding(
            finding_id="finding_momo",
            gate=GateName.G1,
            category="safety",
            severity=Severity.MEDIUM,
            code="ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION",
            message="Momelotinib DILI and hERG predictions require experimental confirmation.",
        ),
        AuditFinding(
            finding_id="finding_fedra",
            gate=GateName.G1,
            category="safety",
            severity=Severity.MEDIUM,
            code="ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION",
            message="Fedratinib hERG prediction requires experimental confirmation.",
        ),
        AuditFinding(
            finding_id="finding_contract",
            gate=GateName.G1,
            category="schema",
            severity=Severity.HIGH,
            code="REVIEWER_PROSPECTIVE_CONTRACT_ADDS_UNAUTHORIZED_EXPOSURE_RULES",
            message=(
                "Reviewer contract adds unauthorized exposure rules and a "
                "cell-viability exposure selection requirement."
            ),
        ),
        AuditFinding(
            finding_id="finding_score",
            gate=GateName.G1,
            category="model_appropriateness",
            severity=Severity.MEDIUM,
            code="REVIEWER_TARGET_LEVEL_EVIDENCE_EMBEDDED_IN_CANDIDATE_SCORES",
            message="Target-level evidence is embedded in candidate scores.",
        ),
    ]
    verdict = GateVerdict(
        gate=GateName.G1,
        verdict=GateVerdictCode.ABSTAIN,
        findings=findings,
        rationale="Additional action is required.",
        reviewer_model="test-reviewer",
    )

    packet = build_next_action_packet(
        run_id="run_test",
        trigger_stage=RunStage.G1,
        brief=brief,
        verdict=verdict,
        candidates=[
            _candidate("cand_momo", "Momelotinib"),
            _candidate("cand_fedra", "Fedratinib"),
        ],
        checkpoint=None,
        terminal_message="G1 did not pass.",
    )

    assert packet.disposition == "MIXED_ACTIONS_REQUIRED"
    assert packet.approval_allowed is False
    assert packet.human_gate_mode == "action_required_hold"
    assert {item.kind for item in packet.actions} == {
        "experiment_required",
        "computational_rework",
    }
    assert {item.candidate_id for item in packet.experiment_requests} == {
        "cand_momo",
        "cand_fedra",
    }
    assert all(item.synthetic_results_forbidden for item in packet.experiment_requests)
    assert all(item.result_status == "awaiting_results" for item in packet.experiment_requests)
    serialized = packet.model_dump_json().casefold()
    assert "measured_value" in serialized
    assert '"measured_value":' not in serialized
    assert "fresh g1 pass" in serialized


def test_unclassified_stop_still_produces_blocking_operator_action(brief) -> None:
    packet = build_next_action_packet(
        run_id="run_failed",
        trigger_stage=RunStage.FAILED,
        brief=brief,
        verdict=None,
        candidates=[],
        checkpoint=None,
        terminal_message="Unexpected provider response.",
    )

    assert len(packet.actions) == 1
    assert packet.actions[0].kind == "operator_review"
    assert packet.actions[0].blocking is True
    assert packet.resume_eligible is False


def test_exhausted_g1_rework_routes_to_action_required_human_gate(brief) -> None:
    verdict = GateVerdict(
        gate=GateName.G1,
        verdict=GateVerdictCode.REVISE,
        rationale="Experimental confirmation is required.",
        reviewer_model="test-reviewer",
        rework_order=ReworkOrder(
            gate=GateName.G1,
            target_stage="ma2",
            failed_node="g1.audit",
            required_actions=["Attach real assay results"],
        ),
    )
    orchestrator = object.__new__(DrugPilotOrchestrator)

    route = DrugPilotOrchestrator._route_g1(
        orchestrator,
        {
            "g1_verdict": verdict,
            "brief": brief,
            "rework_loops": brief.constraints.max_rework_loops,
        },
    )

    assert route == "human_gate"


@pytest.mark.asyncio
async def test_server_rejects_approve_while_action_packet_is_open(tmp_path, brief) -> None:
    orchestrator = DrugPilotOrchestrator(
        settings=Settings(_env_file=None, database_path=tmp_path / "actions.sqlite3")
    )
    packet = build_next_action_packet(
        run_id="run_locked",
        trigger_stage=RunStage.G1,
        brief=brief,
        verdict=None,
        candidates=[],
        checkpoint=None,
        terminal_message="Additional work is required.",
    )
    await orchestrator.store.save_snapshot(
        RunSnapshot(
            run_id="run_locked",
            mode=RunMode.LIVE,
            stage=RunStage.HUMAN_GATE,
            status_message="Action required.",
            brief=brief,
            next_action_packet=packet,
        )
    )

    with pytest.raises(ValueError, match="APPROVE is locked"):
        await orchestrator.apply_human_decision(
            HumanGateDecision(
                run_id="run_locked",
                decision=HumanDecisionCode.APPROVE,
                researcher="Researcher",
                rationale="Approve despite unresolved work.",
            )
        )
    orchestrator.close()


@pytest.mark.asyncio
async def test_real_experimental_evidence_is_hash_bound_before_reaudit(tmp_path, brief) -> None:
    orchestrator = DrugPilotOrchestrator(
        settings=Settings(_env_file=None, database_path=tmp_path / "evidence.sqlite3")
    )
    finding = AuditFinding(
        finding_id="finding_fedra_assay",
        gate=GateName.G1,
        category="safety",
        severity=Severity.MEDIUM,
        code="ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION",
        message="Fedratinib hERG prediction requires experimental confirmation.",
    )
    verdict = GateVerdict(
        gate=GateName.G1,
        verdict=GateVerdictCode.ABSTAIN,
        findings=[finding],
        rationale="Observed assay results are required.",
        reviewer_model="test-reviewer",
    )
    packet = build_next_action_packet(
        run_id="run_evidence",
        trigger_stage=RunStage.G1,
        brief=brief,
        verdict=verdict,
        candidates=[_candidate("cand_fedra", "Fedratinib")],
        checkpoint=None,
        terminal_message="G1 did not pass.",
    )
    action = next(item for item in packet.actions if item.kind == "experiment_required")
    request = packet.experiment_requests[0]
    await orchestrator.store.save_snapshot(
        RunSnapshot(
            run_id="run_evidence",
            mode=RunMode.LIVE,
            stage=RunStage.HUMAN_GATE,
            status_message="Action required.",
            brief=brief,
            g1_verdict=verdict,
            next_action_packet=packet,
        )
    )
    source = b"candidate_id,replicate,value\ncand_fedra,R1,1.7\n"
    source_digest = sha256(source).hexdigest()
    row = ExperimentalResultRecord(
        candidate_id="cand_fedra",
        assay_name="hERG patch clamp",
        replicate_id="R1",
        dose=1.0,
        dose_unit="uM",
        timepoint="30 min",
        endpoint="current inhibition",
        measured_value=1.7,
        value_unit="percent",
        control="vehicle",
        uncertainty=0.2,
        source_artifact_sha256=source_digest,
    )

    attached = await orchestrator.submit_action_evidence(
        "run_evidence",
        action_id=action.action_id,
        experiment_request_id=request.request_id,
        evidence_type="experimental_result",
        submitted_by="Laboratory A",
        notes="Observed hERG result with one replicate for contract validation.",
        source_filename="fedratinib-herg.csv",
        media_type="text/csv",
        source_content_base64=base64.b64encode(source).decode("ascii"),
        observed_results=[row],
        observed_not_synthetic_attestation=True,
    )

    submission = attached.action_evidence_submissions[0]
    assert submission.source_artifact_sha256 == source_digest
    assert attached.next_action_progress is not None
    assert attached.next_action_progress.action_status[action.action_id] == "evidence_attached"
    assert attached.next_action_progress.approval_allowed is False

    completed = await orchestrator.resolve_next_action(
        "run_evidence",
        action_id=action.action_id,
        supporting_submission_ids=[submission.submission_id],
        resolved_by="Research lead",
        rationale="The matching observed result and source artifact were reviewed.",
    )

    assert completed.next_action_progress is not None
    assert completed.next_action_progress.ready_for_reaudit is True
    assert completed.next_action_progress.approval_allowed is False
    assert completed.human_decision is None
    assert await orchestrator.store.verify_ledger("run_evidence") is True
    orchestrator.close()


@pytest.mark.asyncio
async def test_experimental_row_cannot_claim_a_different_source_hash(tmp_path, brief) -> None:
    orchestrator = DrugPilotOrchestrator(
        settings=Settings(_env_file=None, database_path=tmp_path / "hash.sqlite3")
    )
    finding = AuditFinding(
        finding_id="finding_hash",
        gate=GateName.G1,
        category="safety",
        severity=Severity.MEDIUM,
        code="ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION",
        message="Fedratinib requires experimental confirmation.",
    )
    verdict = GateVerdict(
        gate=GateName.G1,
        verdict=GateVerdictCode.ABSTAIN,
        findings=[finding],
        rationale="Observed assay results are required.",
        reviewer_model="test-reviewer",
    )
    packet = build_next_action_packet(
        run_id="run_hash",
        trigger_stage=RunStage.G1,
        brief=brief,
        verdict=verdict,
        candidates=[_candidate("cand_fedra", "Fedratinib")],
        checkpoint=None,
        terminal_message="G1 did not pass.",
    )
    action = next(item for item in packet.actions if item.kind == "experiment_required")
    request = packet.experiment_requests[0]
    await orchestrator.store.save_snapshot(
        RunSnapshot(
            run_id="run_hash",
            mode=RunMode.LIVE,
            stage=RunStage.HUMAN_GATE,
            status_message="Action required.",
            brief=brief,
            g1_verdict=verdict,
            next_action_packet=packet,
        )
    )
    row = ExperimentalResultRecord(
        candidate_id="cand_fedra",
        assay_name="hERG patch clamp",
        replicate_id="R1",
        dose=1.0,
        dose_unit="uM",
        timepoint="30 min",
        endpoint="current inhibition",
        measured_value=1.7,
        value_unit="percent",
        control="vehicle",
        uncertainty=0.2,
        source_artifact_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="server-computed attachment hash"):
        await orchestrator.submit_action_evidence(
            "run_hash",
            action_id=action.action_id,
            experiment_request_id=request.request_id,
            evidence_type="experimental_result",
            submitted_by="Laboratory A",
            notes="This row intentionally claims the wrong source.",
            source_filename="fedratinib-herg.csv",
            media_type="text/csv",
            source_content_base64=base64.b64encode(b"real source").decode("ascii"),
            observed_results=[row],
            observed_not_synthetic_attestation=True,
        )
    orchestrator.close()
