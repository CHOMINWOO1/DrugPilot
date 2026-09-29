from __future__ import annotations

import asyncio
import base64
import sqlite3
from hashlib import sha256
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from drugpilot.config import Settings
from drugpilot.domain.models import (
    AuditFinding, ExperimentalResultRecord, GateName, GateVerdictCode,
    HumanGateDecision, RunStage, Severity,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.orchestration.human_review import HumanReviewService
from drugpilot.orchestration.next_actions import _kind_for


@pytest.fixture
async def review_run(tmp_path, brief):
    o = DrugPilotOrchestrator(settings=Settings(_env_file=None,
        database_path=tmp_path / "review.sqlite3", auto_activate_openai=False,
        OPENAI_API_KEY=None, ANTHROPIC_API_KEY=None, mock_step_delay_ms=0,
        public_evidence_enabled=False, demo_fixture_fallback_enabled=True))
    created = await o.create_run(brief)
    task = o._tasks.get(created.run_id)
    if task:
        await task
    s = await o.store.get_snapshot(created.run_id)
    assert s.stage is RunStage.HUMAN_GATE
    assert s.g1_verdict.verdict is GateVerdictCode.PASS
    yield o, HumanReviewService(o), s
    o.close()


async def start_and_wait(service, o, s, key="request-rules-0001", mode="rules", **kwargs):
    result = await service.start(s.run_id, request_id=key,
        base_verdict_id=s.g1_verdict.verdict_id, researcher="Test researcher",
        rationale="Re-evaluate stored evidence after a generic code correction.",
        mode=mode, **kwargs)
    task = o._tasks.get(s.run_id)
    if task:
        await task
    return await o.store.get_snapshot(s.run_id)


class IsolatedSourceCollector:
    calls = 0

    async def collect(self, snapshot):
        from drugpilot.domain.models import SourceAcquisitionReport, SourceDocumentReview
        self.calls += 1
        candidate = snapshot.candidate_packet.candidates[0]
        return SourceAcquisitionReport(run_id=snapshot.run_id,
            candidate_packet_id=snapshot.candidate_packet.packet_id,
            candidate_packet_sha256=sha256(snapshot.candidate_packet.model_dump_json().encode()).hexdigest(),
            documents=[SourceDocumentReview(candidate_id=candidate.candidate_id, candidate_name=candidate.name,
                status="unavailable", reason_code="TEST_ONLY_SOURCE_UNAVAILABLE")])


@pytest.mark.asyncio
async def test_source_mode_configuration_idempotency_lineage_and_science_immutability(review_run):
    o, _, s = review_run
    collector = IsolatedSourceCollector()
    service = HumanReviewService(o, source_enricher=collector)
    with pytest.raises(ValueError, match="PUBLIC_LIVE_NO_FIXTURE_REQUIRED"):
        await start_and_wait(service, o, s, mode="sources")
    o.settings.public_evidence_enabled = True
    o.settings.demo_fixture_fallback_enabled = False
    before = await o.store.artifact_manifest(s.run_id)
    result = await start_and_wait(service, o, s, mode="sources")
    assert result.g1_reaudits[-1].status == "completed"
    report, record = result.public_source_reports[-1], result.g1_reaudits[-1]
    assert report.request_id == record.request_id
    assert report.request_sha256 == record.request_sha256 and report.source_sha256 == record.source_sha256
    assert report.input_sha256 == record.input_sha256
    assert report.base_verdict_sha256 == sha256(s.g1_verdict.model_dump_json().encode()).hexdigest()
    assert record.source_report_ids == [report.report_id]
    assert result.candidate_packet == s.candidate_packet and result.g1_verdict == s.g1_verdict
    assert result.token_usage == s.token_usage and result.human_decision is None
    assert (await o.store.read_artifact(s.run_id, report.report_id, "SourceAcquisitionReport")) == report.model_dump(mode="json")
    assert await o.store.verify_ledger(s.run_id)
    after = {a["artifact_id"]:a["sha256"] for a in await o.store.artifact_manifest(s.run_id)}
    assert all(after[a["artifact_id"]] == a["sha256"] for a in before)
    await start_and_wait(service, o, s, mode="sources")
    assert collector.calls == 1
    with pytest.raises(ValueError, match="SOURCE_REFRESH_COOLDOWN"):
        await start_and_wait(service, o, result, mode="sources", key="source-second-request")


@pytest.mark.asyncio
async def test_failed_source_collection_leaves_no_report_or_scientific_mutation(review_run):
    o, _, s = review_run
    o.settings.public_evidence_enabled = True
    o.settings.demo_fixture_fallback_enabled = False
    class FailingCollector:
        async def collect(self, snapshot):
            raise RuntimeError("source unavailable")
    result = await start_and_wait(HumanReviewService(o, source_enricher=FailingCollector()), o, s, mode="sources")
    assert result.g1_reaudits[-1].status == "failed" and not result.public_source_reports
    assert result.candidate_packet == s.candidate_packet and result.g1_verdict == s.g1_verdict
    assert result.token_usage == s.token_usage and await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [None, "unbound", "request_hash", "base_verdict_hash"])
async def test_failed_refresh_keeps_previous_success_in_reviewer_input(review_run, monkeypatch, tamper):
    from drugpilot.domain.models import G1ReauditRecord
    from drugpilot.orchestration.human_review import input_digest
    o, service, s = review_run
    before = input_digest(s)
    collector = IsolatedSourceCollector()
    successful = await collector.collect(s)
    successful.documents[0].status = "available_for_review"
    latest = await collector.collect(s)
    for index, report in enumerate((successful, latest)):
        owner = G1ReauditRecord(request_id=f"test-sources-request-{index}", request_sha256="1"*64,
            input_sha256="2"*64, source_sha256="3"*64, researcher="Test", rationale="Isolated ingest lineage",
            mode="sources", status="completed", previous_verdict_id=s.g1_verdict.verdict_id,
            verdict_id=s.g1_verdict.verdict_id, source_report_ids=[report.report_id])
        report.request_id, report.request_sha256 = owner.request_id, owner.request_sha256
        report.input_sha256, report.source_sha256 = owner.input_sha256, owner.source_sha256
        report.base_verdict_id = owner.previous_verdict_id
        report.base_verdict_sha256 = sha256(s.g1_verdict.model_dump_json().encode()).hexdigest()
        if tamper == "unbound": owner.source_report_ids = []
        if tamper == "request_hash": report.request_sha256 = "9"*64
        if tamper == "base_verdict_hash": report.base_verdict_sha256 = "9"*64
        digest = await o.store.save_artifact(s.run_id, owner.record_id, "G1ReauditRecord", owner)
        await o._emit(s, s.stage, "test_source_owner", "Isolated source owner",
            artifact_ids=[owner.record_id], metadata={"artifact_sha256":{owner.record_id:digest}})
        s.g1_reaudits.append(owner)
        digest = await o.store.save_artifact(s.run_id, report.report_id, "SourceAcquisitionReport", report)
        await o._emit(s, s.stage, "test_source_report", "Isolated source report",
            artifact_ids=[report.report_id], metadata={"artifact_sha256":{report.report_id:digest}})
    s.public_source_reports = [successful, latest]
    await o.store.save_snapshot(s)
    assert before != input_digest(s)
    captured = []
    async def audit(*args, **kwargs):
        captured.extend(kwargs["supplemental_evidence"].public_source_reports)
        return s.g1_verdict.model_copy(update={"verdict_id":"verdict_test_sources"})
    o.ma3.reviewer_provider = object()
    monkeypatch.setattr(o.ma3, "audit_g1", audit)
    result = await start_and_wait(service, o, s, mode="external", allow_paid=True)
    if tamper:
        assert result.g1_reaudits[-1].status == "failed" and not captured
        assert result.g1_verdict == s.g1_verdict
        return
    assert result.g1_reaudits[-1].status == "completed"
    assert [r.report_id for r in captured] == [successful.report_id, latest.report_id]


@pytest.mark.asyncio
async def test_reaudit_preserves_candidate_and_prior_verdict_with_new_checkpoint(review_run):
    o, service, s = review_run
    candidate_before = s.candidate_packet.model_dump_json()
    before_manifest = await o.store.artifact_manifest(s.run_id)
    result = await start_and_wait(service, o, s)
    assert result.g1_reaudits[-1].status == "completed"
    assert result.g1_verdict.verdict_id != s.g1_verdict.verdict_id
    assert result.g1_verdict.verdict is GateVerdictCode.PASS
    assert result.candidate_packet.model_dump_json() == candidate_before
    assert result.token_usage == s.token_usage
    assert result.human_decision is None
    after = {a["artifact_id"]: a["sha256"] for a in await o.store.artifact_manifest(s.run_id)}
    assert all(after[a["artifact_id"]] == a["sha256"] for a in before_manifest)
    checkpoint = await o.store.latest_stage_checkpoint(s.run_id)
    assert checkpoint.g1_verdict.verdict_id == result.g1_verdict.verdict_id
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_idempotent_replay_and_unchanged_input_do_not_reaudit(review_run):
    o, service, s = review_run
    result = await start_and_wait(service, o, s)
    replay = await start_and_wait(service, o, s)
    assert len(replay.g1_reaudits) == 1
    assert replay.g1_verdict.verdict_id == result.g1_verdict.verdict_id
    with pytest.raises(ValueError, match="INPUTS_UNCHANGED"):
        await start_and_wait(service, o, result, key="different-request-02")
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        await service.start(s.run_id, request_id="request-rules-0001",
            base_verdict_id=s.g1_verdict.verdict_id, researcher="Changed", rationale="Different request", mode="rules")


@pytest.mark.asyncio
@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.LOW])
async def test_rules_only_cannot_erase_previous_external_concern(review_run, severity):
    o, service, s = review_run
    concern = AuditFinding(gate=GateName.G1, category="model_appropriateness",
        severity=severity, code="REVIEWER_UNRESOLVED_CONTEXT",
        message="Independent review still requires source context.", deterministic=False)
    s.g1_verdict = o.ma3._decide(GateName.G1, [concern], [], reviewer_invoked=True)
    digest = await o.store.save_artifact(s.run_id, s.g1_verdict.verdict_id, "GateVerdict", s.g1_verdict)
    await o._emit(s, s.stage, "test_external_concern", "Test-only external concern",
        artifact_ids=[s.g1_verdict.verdict_id], metadata={"artifact_sha256": {s.g1_verdict.verdict_id: digest}})
    await o.store.save_snapshot(s)
    await o.materialize_next_actions(s.run_id)
    result = await start_and_wait(service, o, s)
    assert result.g1_reaudits[-1].status == "completed"
    assert "REVIEWER_UNRESOLVED_CONTEXT" in result.g1_reaudits[-1].remaining_finding_codes
    assert result.g1_verdict.verdict is GateVerdictCode.PASS
    assert result.next_action_packet is None
    assert not any(v.name == "external_review_freshness" and not v.passed for v in result.g1_verdict.validations)
    retained = next(f for f in result.g1_verdict.findings if f.code == "REVIEWER_UNRESOLVED_CONTEXT")
    assert retained.confirmation_status == "unconfirmed"
    assert retained.reported_severity is severity


@pytest.mark.asyncio
async def test_note_keeps_gate_open_and_final_hold_closes_it(review_run):
    o, service, s = review_run
    noted = await service.note(s.run_id, researcher="Lead", note="Await original assay report.")
    assert noted.stage is RunStage.HUMAN_GATE and noted.human_decision is None
    assert len(noted.review_notes) == 1
    assert (await service.status(s.run_id))["can_attach"]
    await o.apply_human_decision(HumanGateDecision(run_id=s.run_id, decision="HOLD",
        researcher="Lead", rationale="Close this review."))
    assert not (await service.status(s.run_id))["open_gate"]
    with pytest.raises(ValueError, match="HUMAN_GATE_CLOSED"):
        await service.note(s.run_id, researcher="Lead", note="Reopen silently")


@pytest.mark.asyncio
async def test_paid_reaudit_requires_consent_and_refuses_stale_verdict(review_run):
    o, service, s = review_run
    o.ma3.reviewer_provider = object()
    with pytest.raises(ValueError, match="PAID_REVIEW_NOT_AUTHORIZED"):
        await start_and_wait(service, o, s, mode="external")
    with pytest.raises(ValueError, match="STALE_VERDICT"):
        await service.start(s.run_id, request_id="stale-request-0001", base_verdict_id="old",
            researcher="Lead", rationale="Stale browser", mode="rules")
    assert not (await o.store.get_snapshot(s.run_id)).g1_reaudits


@pytest.mark.asyncio
async def test_failure_preserves_original_gate_and_blocks_cost_only_retry(review_run, monkeypatch):
    o, service, s = review_run
    async def fail(*args, **kwargs):
        raise RuntimeError("provider failure without secret details")
    o.ma3.reviewer_provider = object()
    monkeypatch.setattr(o.ma3, "audit_g1", fail)
    result = await start_and_wait(service, o, s, mode="external", allow_paid=True)
    assert result.g1_reaudits[-1].status == "failed"
    assert result.g1_reaudits[-1].error_code == "RuntimeError"
    assert result.g1_verdict == s.g1_verdict
    assert not (await service.status(s.run_id))["modes"]["external"]["eligible"]
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_unavailable_external_review_cannot_return_authoritative_pass(review_run, monkeypatch):
    o, service, s = review_run
    async def unavailable(*args, **kwargs):
        finding = AuditFinding(gate=GateName.G1, category="model_appropriateness",
            severity=Severity.MEDIUM, code="REVIEWER_UNAVAILABLE", deterministic=False,
            message="External reviewer was unavailable.")
        return o.ma3._decide(GateName.G1, [finding], [], reviewer_invoked=False)
    o.ma3.reviewer_provider = object()
    monkeypatch.setattr(o.ma3, "audit_g1", unavailable)
    result = await start_and_wait(service, o, s, mode="external", allow_paid=True)
    assert result.g1_verdict.verdict is GateVerdictCode.REVISE
    assert any(v.name == "external_review_freshness" and not v.passed for v in result.g1_verdict.validations)
    assert result.next_action_packet.approval_allowed is False
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_running_reaudit_rejects_decision_and_notes(review_run, monkeypatch):
    o, service, s = review_run
    entered, finish = asyncio.Event(), asyncio.Event()
    async def audit(*args, **kwargs):
        entered.set()
        await finish.wait()
        return s.g1_verdict.model_copy(update={"verdict_id": "verdict_new_for_concurrency"})
    o.ma3.reviewer_provider = object()
    monkeypatch.setattr(o.ma3, "audit_g1", audit)
    await service.start(s.run_id, request_id="concurrent-review-01", base_verdict_id=s.g1_verdict.verdict_id,
        researcher="Lead", rationale="Concurrency test", mode="external", allow_paid=True)
    await entered.wait()
    with pytest.raises(ValueError, match="RUN_ACTIVE"):
        await o.apply_human_decision(HumanGateDecision(run_id=s.run_id, decision="REJECT", researcher="Lead", rationale="Racing decision"))
    with pytest.raises(ValueError, match="RUN_ACTIVE"):
        await service.note(s.run_id, researcher="Lead", note="Racing note")
    task = o._tasks[s.run_id]
    finish.set()
    await task
    assert (await o.store.get_snapshot(s.run_id)).human_decision is None


@pytest.mark.parametrize("code,kind", [
    ("ASSAY_VARIANT_CONTEXT_UNKNOWN", "data_required"),
    ("CANDIDATE_PRIMARY_PAPER_NATIVE_V6_ATTESTATION_INVALID", "computational_rework"),
    ("REVIEWER_KNOWN_DRUG_COUNT_INCLUDES_NON_ATOMIC_INTERVENTION_LABELS", "computational_rework"),
    ("ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION", "experiment_required"),
])
def test_followup_classifies_by_actual_action(code, kind):
    assert _kind_for(AuditFinding(gate=GateName.G1, category="safety", severity=Severity.MEDIUM,
        code=code, message="Additional supporting information required.")) == kind


@pytest.mark.parametrize("field", ["dose", "measured_value", "uncertainty"])
def test_observed_values_reject_nan_infinity(field):
    raw = dict(candidate_id="candidate", assay_name="assay", replicate_id="R1", dose=1,
        dose_unit="nM", timepoint="1h", endpoint="binding", measured_value=2,
        value_unit="percent", control="vehicle", uncertainty=0.1, source_artifact_sha256="0"*64)
    raw[field] = float("nan")
    with pytest.raises(ValidationError):
        ExperimentalResultRecord(**raw)


@pytest.mark.asyncio
async def test_request_transaction_rolls_back_before_any_review_call(review_run):
    o, service, s = review_run
    before = await o.store.artifact_manifest(s.run_id)
    with o.store._connect() as connection:
        connection.execute("CREATE TRIGGER fail_review_request BEFORE UPDATE ON runs BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        await start_and_wait(service, o, s)
    with o.store._connect() as connection:
        connection.execute("DROP TRIGGER fail_review_request")
    persisted = await o.store.get_snapshot(s.run_id)
    assert persisted.g1_reaudits == []
    assert await o.store.artifact_manifest(s.run_id) == before
    assert await o.store.verify_ledger(s.run_id)
    assert (await start_and_wait(service, o, s)).g1_reaudits[-1].status == "completed"


@pytest.mark.asyncio
async def test_result_transaction_failure_leaves_no_partial_verdict_or_orphan(review_run):
    o, service, s = review_run
    before = await o.store.artifact_manifest(s.run_id)
    with o.store._connect() as connection:
        connection.execute("CREATE TRIGGER fail_review_checkpoint BEFORE INSERT ON stage_checkpoints BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    result = await start_and_wait(service, o, s)
    assert result.g1_reaudits[-1].status == "failed"
    assert result.g1_verdict == s.g1_verdict
    after = await o.store.artifact_manifest(s.run_id)
    before_ids = {a["artifact_id"] for a in before}
    assert all(a["artifact_type"] == "G1ReauditRecord" for a in after if a["artifact_id"] not in before_ids)
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_broker_error_after_commit_cannot_undo_completed_review(review_run, monkeypatch):
    o, service, s = review_run
    original = o.broker.publish
    async def fail_completion(event):
        if event.event_type == "g1_reaudit_completed":
            raise RuntimeError("UI connection interrupted")
        await original(event)
    monkeypatch.setattr(o.broker, "publish", fail_completion)
    result = await start_and_wait(service, o, s)
    assert result.g1_reaudits[-1].status == "completed"
    assert result.g1_verdict.verdict_id != s.g1_verdict.verdict_id
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_claim_reread_preserves_concurrent_review_note(review_run, monkeypatch):
    o, service, s = review_run
    from drugpilot.domain.models import ReviewNote
    original = o.store.claim_live_execution_lease
    async def claim(*args, **kwargs):
        current = await o.store.get_snapshot(s.run_id)
        current.review_notes.append(ReviewNote(researcher="Other process", note="Earlier review context"))
        await o.store.save_snapshot(current)
        return await original(*args, **kwargs)
    monkeypatch.setattr(o.store, "claim_live_execution_lease", claim)
    result = await start_and_wait(service, o, s)
    assert result.review_notes[0].note == "Earlier review context"


@pytest.mark.asyncio
async def test_claim_reread_rejects_concurrent_final_decision(review_run, monkeypatch):
    o, service, s = review_run
    original = o.store.claim_live_execution_lease
    async def claim(*args, **kwargs):
        current = await o.store.get_snapshot(s.run_id)
        current.human_decision = HumanGateDecision(run_id=s.run_id, decision="HOLD", researcher="Other process", rationale="Close review")
        await o.store.save_snapshot(current)
        return await original(*args, **kwargs)
    monkeypatch.setattr(o.store, "claim_live_execution_lease", claim)
    with pytest.raises(ValueError, match="HUMAN_GATE_CLOSED"):
        await start_and_wait(service, o, s)
    assert not (await o.store.get_snapshot(s.run_id)).g1_reaudits
