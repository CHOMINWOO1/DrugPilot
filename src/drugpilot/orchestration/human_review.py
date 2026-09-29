"""Researcher-controlled review of stopped runs with immutable audit history."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
from datetime import timedelta

from drugpilot.agents.ma3_reviewer_projection import ReviewerProjectionMA3AssuranceAgent
from drugpilot.domain.models import (
    ExecutionEvent, G1ReauditRecord, GateName, GateVerdict, GateVerdictCode, NextActionProgress, ReviewNote,
    RunMode, RunStage, SourceAcquisitionReport, SupplementalReviewBundle, ValidationResult, WorkflowCheckpoint, new_id, utc_now,
)
from drugpilot.orchestration.next_actions import build_next_action_packet
from drugpilot.tools.review_source_enrichment import PublicSourceEnricher, LIMITS as SOURCE_LIMITS


def _digest(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    names = ("agents/ma1.py", "agents/evidence_semantics.py", "agents/hypothesis_projection.py",
             "agents/variant_search_scope.py",
             "agents/molecular_scope.py", "agents/document_result_units.py", "tools/pharos_on_demand.py",
             "tools/disease_targets.py",
             "orchestration/semantic_interpretation.py",
             "orchestration/evidence_carryover.py",
             "agents/ma2.py", "agents/ma3.py", "agents/ma3_reviewer_projection.py",
             "tools/assay_variant_alignment.py",
             "tools/route_policy.py", "orchestration/next_actions.py",
             "orchestration/human_review.py", "orchestration/graph.py", "domain/models.py", "storage/sqlite.py",
             "orchestration/automatic_resolution.py",
             "orchestration/research_selection.py",
             "providers/llm.py", "providers/structured_outputs.py", "config.py",
             "orchestration/g0_followup.py",
             "orchestration/goal_focus.py", "domain/goal_intent.py", "domain/cell_context.py",
             "tools/context_router.py",
             "tools/evidence_gap.py",
             "tools/assay_followup.py",
             "tools/assay_relations.py",
             "tools/assay_identity.py",
             "tools/assay_applicability.py",
             "tools/assay_fulltext.py",
             "tools/review_source_enrichment.py", "tools/europe_pmc_oa_fulltext.py", "tools/public_biology.py")
    return _digest({name: sha256((root / name).read_bytes()).hexdigest() for name in names})


def input_digest(snapshot) -> str:
    return _digest({name: getattr(snapshot, name) and (
        getattr(snapshot, name).model_dump(mode="json")
        if hasattr(getattr(snapshot, name), "model_dump")
        else [item.model_dump(mode="json") for item in getattr(snapshot, name)]
    ) for name in ("brief", "hypothesis_packet", "g0_verdict", "plan_spec",
                   "candidate_packet", "action_evidence_submissions", "action_resolutions", "public_source_reports")})


def review_input_digest(snapshot, mode):
    if mode != "sources":
        return input_digest(snapshot)
    return _digest({"candidate_packet": snapshot.candidate_packet.model_dump(mode="json") if snapshot.candidate_packet else None,
                    "findings": sorted((f.code, f.artifact_id or "") for f in snapshot.g1_verdict.findings if not f.resolved)
                    if snapshot.g1_verdict else []})


async def ensure_review_idle(orchestrator, run_id):
    await orchestrator._cancellation_checkpoint(run_id)
    snapshot = await orchestrator.store.get_snapshot(run_id)
    if snapshot is None:
        raise KeyError(run_id)
    task = orchestrator._tasks.get(run_id)
    lease = await orchestrator.store.live_execution_lease_status(run_id, now=utc_now())
    if (task is not None and not task.done()) or (lease and lease["state"] == "active"):
        raise ValueError("RUN_ACTIVE: wait for the current operation")
    if any(item.status == "running" for item in snapshot.g1_reaudits):
        raise ValueError("REAUDIT_RECOVERY_REQUIRED: reconcile the interrupted review first")
    if not await orchestrator.store.verify_ledger(run_id):
        raise ValueError("LEDGER_INTEGRITY_FAILED")
    return snapshot


def serialized_review_mutation(method):
    """Serialize decisions and evidence writes against queued/running re-audits."""
    @wraps(method)
    async def wrapped(self, run_id, *args, **kwargs):
        key = run_id if isinstance(run_id, str) else run_id.run_id
        async with review_mutation(self, key) as token:
            kwargs["_review_lease_token"] = token
            return await method(self, run_id, *args, **kwargs)
    return wrapped


@asynccontextmanager
async def review_mutation(orchestrator, run_id):
    async with orchestrator._task_registry_lock:
        await ensure_review_idle(orchestrator, run_id)
        token = new_id("review_write")
        await orchestrator.store.claim_live_execution_lease(
            run_id, lease_token=token, now=utc_now(),
            lease_ttl_seconds=orchestrator.settings.beta_live_lease_ttl_seconds,
            max_active_live_runs=20)
        try:
            yield token
        finally:
            await orchestrator.store.release_live_execution_lease(
                run_id, lease_token=token, reason="review_write_finished", now=utc_now())


class HumanReviewService:
    def __init__(self, orchestrator, *, source_enricher=None):
        self.o = orchestrator
        self.source_enricher = source_enricher

    async def run_automatic_inline(self, run_id, token):
        """Native live-workflow step; ownership is the existing workflow lease."""
        from drugpilot.orchestration.automatic_resolution import current_report, run_automatic
        s = await self.o.store.get_snapshot(run_id)
        if any(r.mode == "automatic" for r in s.g1_reaudits):
            if any(r.status == "running" for r in s.g1_reaudits):
                raise ValueError("AUTOMATIC_RECOVERY_REQUIRED")
            report = current_report(s)
            if report is None:
                raise ValueError("AUTOMATIC_COMPLETION_NOT_VERIFIED")
            return report
        if not await self.o.store.verify_ledger(run_id):
            raise ValueError("LEDGER_INTEGRITY_FAILED")
        record = G1ReauditRecord(request_id=new_id("automatic_followup"),
            request_sha256=_digest({"run_id":run_id,"scope":"native_workflow_automatic_followup",
                "paid_enabled":self.o.settings.automatic_resolution_external_review_enabled}),
            input_sha256=input_digest(s), source_sha256=source_digest(),
            researcher="DrugPilot 자동 보완", rationale="가용 원문 경로·규칙 재검사·필요 시 예산 내 외부 재감사 후 자동 결론",
            mode="automatic", previous_verdict_id=s.g1_verdict.verdict_id, usage_start_index=len(s.token_usage))
        s.status_message = "자동 보완 진행 중 · 공개 원문·규칙·필요 시 예산 내 재감사"
        await self._record(s, record, token)
        return await run_automatic(self, run_id, record, token,
            allow_paid=self.o.settings.automatic_resolution_external_review_enabled)

    async def status(self, run_id):
        o = self.o
        s = await o.store.get_snapshot(run_id)
        if s is None:
            raise KeyError(run_id)
        lease = await o.store.live_execution_lease_status(run_id, now=utc_now())
        active = bool(lease and lease["state"] == "active")
        pending = any(r.status == "running" for r in s.g1_reaudits)
        open_gate = s.stage is RunStage.HUMAN_GATE and s.human_decision is None
        artifacts = bool(s.g0_verdict and s.g0_verdict.verdict is GateVerdictCode.PASS
                         and s.hypothesis_packet and s.plan_spec and s.candidate_packet
                         and s.g1_verdict)
        source = source_digest()
        cancelled = s.stage in {RunStage.CANCEL_REQUESTED, RunStage.CANCELLED}
        reason = ("RUN_CANCELLED" if cancelled else "RUN_ACTIVE" if active else "REAUDIT_RECOVERY_REQUIRED" if pending
                  else "HUMAN_GATE_CLOSED" if not open_gate else
                  "G0_PASS_AND_G1_ARTIFACTS_REQUIRED" if not artifacts else None)
        modes = {}
        for mode in ("rules", "external", "sources", "automatic"):
            inputs = review_input_digest(s, mode)
            repeated = any(r.mode == mode and r.input_sha256 == inputs
                           and r.source_sha256 == source and r.status in {"running", "completed", "failed"}
                           and (mode != "sources" or utc_now() < r.started_at + timedelta(seconds=SOURCE_LIMITS["refresh_cooldown_seconds"]))
                           for r in s.g1_reaudits)
            if mode == "automatic":
                repeated = repeated or any(r.source_sha256 == source and r.final_input_sha256 == input_digest(s)
                    and r.final_verdict_id == s.g1_verdict.verdict_id for r in s.automatic_resolution_reports)
            configured = (o.settings.public_evidence_enabled and not o.settings.demo_fixture_fallback_enabled
                          if mode in {"sources", "automatic"} else mode == "rules" or o.ma3.reviewer_provider is not None)
            if mode == "automatic":
                configured = configured and o.settings.automatic_resolution_enabled and s.mode is RunMode.LIVE
            modes[mode] = {"eligible": not reason and not repeated and configured,
                           "reason_code": reason or (("SOURCE_REFRESH_COOLDOWN" if mode == "sources" else "INPUTS_UNCHANGED") if repeated else
                              ("PUBLIC_LIVE_NO_FIXTURE_REQUIRED" if mode in {"sources", "automatic"} else "REVIEWER_UNAVAILABLE") if not configured else None)}
        return {"run_id": run_id, "open_gate": open_gate, "busy": active,
                "can_attach": open_gate and not active and not pending,
                "can_recover": pending and not active and not cancelled,
                "base_verdict_id": s.g1_verdict.verdict_id if s.g1_verdict else None,
                "modes": modes, "budget": s.brief.budget.model_dump(mode="json"),
                "recorded_cost_usd": sum(u.estimated_cost_usd for u in s.token_usage),
                "source_acquisition_limits": dict(SOURCE_LIMITS),
                "submission_handling": "recorded_for_review_not_automatic_scientific_validation"}

    async def note(self, run_id, *, researcher, note):
        o = self.o
        async with review_mutation(o, run_id) as token:
            s = await o.store.get_snapshot(run_id)
            if s.stage is not RunStage.HUMAN_GATE or s.human_decision is not None:
                raise ValueError("HUMAN_GATE_CLOSED")
            record = ReviewNote(researcher=researcher, note=note)
            digest = sha256(record.model_dump_json().encode()).hexdigest()
            s.review_notes.append(record)
            event = ExecutionEvent(run_id=run_id, stage=s.stage, event_type="review_note_recorded",
                message="임시 보류 메모 저장 · 자료 제출과 재감사는 계속 가능합니다.",
                node="human_gate.review_note", artifact_ids=[record.note_id],
                metadata={"artifact_sha256": {record.note_id: digest}})
            s.events.append(event)
            await o.store.commit_review_update(s, artifacts=[(record.note_id, "ReviewNote", record)],
                                               events=[event], lease_token=token)
            await o.broker.publish(event)
            return s

    async def recover(self, run_id):
        """Retire an interrupted request, never replay a possibly paid call."""
        o = self.o
        async with o._task_registry_lock:
            await o._cancellation_checkpoint(run_id)
            s = await o.store.get_snapshot(run_id)
            if s is None:
                raise KeyError(run_id)
            lease = await o.store.live_execution_lease_status(run_id, now=utc_now())
            task = o._tasks.get(run_id)
            if (lease and lease["state"] == "active") or (task and not task.done()):
                raise ValueError("RUN_ACTIVE")
            if not await o.store.verify_ledger(run_id):
                raise ValueError("LEDGER_INTEGRITY_FAILED")
            if lease and lease["state"] == "expired_recovery_required":
                await o.store.retire_expired_live_execution_lease(
                    run_id, expected_stage=s.stage.value, now=utc_now())
            token = new_id("review_recovery")
            await o.store.claim_live_execution_lease(run_id, lease_token=token, now=utc_now(),
                lease_ttl_seconds=o.settings.beta_live_lease_ttl_seconds, max_active_live_runs=20)
            try:
                s = await o.store.get_snapshot(run_id)
                for r in list(s.g1_reaudits):
                    if r.status == "running":
                        failed = r.model_copy(update={"record_id": new_id("review_record"),
                            "status": "failed", "error_code": "INTERRUPTED_ACCOUNTING_REVIEW_REQUIRED",
                            "finished_at": utc_now(), "usage_end_index": len(s.token_usage)})
                        await self._record(s, failed, token)
                return s
            finally:
                await o.store.release_live_execution_lease(run_id, lease_token=token,
                    reason="review_recovery_finished", now=utc_now())

    async def start(self, run_id, *, request_id, base_verdict_id, researcher,
                    rationale, mode, allow_paid=False):
        o = self.o
        payload_hash = _digest({"run_id": run_id, "base_verdict_id": base_verdict_id,
            "researcher": researcher, "rationale": rationale, "mode": mode, "allow_paid": allow_paid})
        async with o._task_registry_lock:
            await o._cancellation_checkpoint(run_id)
            s = await o.store.get_snapshot(run_id)
            if s is None:
                raise KeyError(run_id)
            old = next((r for r in s.g1_reaudits if r.request_id == request_id), None)
            if old:
                if old.request_sha256 != payload_hash:
                    raise ValueError("IDEMPOTENCY_CONFLICT")
                return s
            await ensure_review_idle(o, run_id)
            st = await self.status(run_id)
            if mode not in st["modes"] or not st["modes"][mode]["eligible"]:
                raise ValueError(st["modes"].get(mode, {}).get("reason_code") or "INVALID_REAUDIT_MODE")
            if s.g1_verdict.verdict_id != base_verdict_id:
                raise ValueError("STALE_VERDICT: refresh before re-audit")
            if mode == "external" and not allow_paid:
                raise ValueError("PAID_REVIEW_NOT_AUTHORIZED")
            token = new_id("review_lease")
            await o.store.claim_live_execution_lease(run_id, lease_token=token, now=utc_now(),
                lease_ttl_seconds=o.settings.beta_live_lease_ttl_seconds,
                max_active_live_runs=o.settings.beta_max_active_live_runs)
            try:
                # Re-read after the cross-process lease claim.
                latest = await o.store.get_snapshot(run_id)
                if latest.stage is not RunStage.HUMAN_GATE or latest.human_decision is not None:
                    raise ValueError("HUMAN_GATE_CLOSED")
                if latest.g1_reaudits != s.g1_reaudits:
                    raise ValueError("STALE_REVIEW_REQUESTS")
                if review_input_digest(latest, mode) != review_input_digest(s, mode) or latest.g1_verdict.verdict_id != base_verdict_id:
                    raise ValueError("STALE_REVIEW_INPUTS")
                s = latest
                r = G1ReauditRecord(request_id=request_id, request_sha256=payload_hash,
                    input_sha256=review_input_digest(s, mode), source_sha256=source_digest(),
                    researcher=researcher, rationale=rationale, mode=mode,
                    previous_verdict_id=base_verdict_id,
                    submission_ids=[x.submission_id for x in s.action_evidence_submissions],
                    usage_start_index=len(s.token_usage))
                if mode == "sources":
                    s.status_message = "공개 원문 자동 보강 중 · 기존 후보·판정 보존 · 모델 과금 없음"
                if mode == "automatic":
                    s.status_message = "자동 보완 진행 중 · 가용 경로 점검 후 보류·진행 불가·검토 준비로 종료"
                await self._record(s, r, token)
                task = asyncio.create_task(self._execute(run_id, r, token, allow_paid=allow_paid), name=f"drugpilot:{run_id}:reaudit")
                o._track_task(run_id, task)
            except BaseException:
                await o.store.release_live_execution_lease(run_id, lease_token=token,
                    reason="reaudit_start_failed", now=utc_now())
                raise
            return s

    def _prepare_record(self, s, record):
        digest = sha256(record.model_dump_json().encode()).hexdigest()
        s.g1_reaudits = [r for r in s.g1_reaudits if r.request_id != record.request_id] + [record]
        event = ExecutionEvent(run_id=s.run_id, stage=s.stage, event_type=f"g1_reaudit_{record.status}",
            message=f"{ {'sources':'공개 원문 자동 보강','automatic':'자동 보완·판정'}.get(record.mode, 'G1 재감사')} {record.status} · 기존 산출물과 감사 이력 보존",
            node="human_gate.reaudit", artifact_ids=[record.record_id],
            metadata={"artifact_sha256": {record.record_id: digest}})
        s.events.append(event)
        return event

    async def _record(self, s, record, token):
        event = self._prepare_record(s, record)
        await self.o.store.commit_review_update(s,
            artifacts=[(record.record_id, "G1ReauditRecord", record)], events=[event], lease_token=token)
        await self.o.broker.publish(event)

    async def _execute(self, run_id, record, token, *, allow_paid=False):
        from drugpilot.orchestration.automatic_resolution import run_automatic
        o = self.o
        o._workflow_lease_tokens[run_id] = token
        heartbeat = asyncio.create_task(o._lease_heartbeat(run_id, lease_token=token))
        operation = (run_automatic(self, run_id, record, token, allow_paid=allow_paid) if record.mode == "automatic"
                     else self._collect_sources(run_id, record, token) if record.mode == "sources"
                     else self._audit(run_id, record, token))
        audit = asyncio.create_task(operation)
        try:
            done, _ = await asyncio.wait({audit, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if audit not in done:
                await heartbeat
            await audit
        except BaseException as exc:
            audit.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await audit
            s = await o.store.get_snapshot(run_id)
            if s.stage in {RunStage.CANCEL_REQUESTED, RunStage.CANCELLED}:
                await o._finish_cancellation(run_id)
                return
            if any(r.request_id == record.request_id and r.status == "completed" for r in s.g1_reaudits):
                # A UI event delivery failure after commit cannot undo a result.
                return
            failed = record.model_copy(update={"record_id": new_id("review_record"),
                "status": "failed", "error_code": type(exc).__name__,
                "finished_at": utc_now(), "usage_end_index": len(s.token_usage)})
            s.status_message = "검토 작업 중단 · 이전 G1 결과 유지. 사용량과 중단 사유를 확인하십시오."
            await self._record(s, failed, token)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat
            latest = await o.store.get_snapshot(run_id)
            if latest and latest.stage is RunStage.CANCEL_REQUESTED:
                await o._finish_cancellation(run_id)
            await o.store.release_live_execution_lease(run_id, lease_token=token,
                reason="reaudit_finished", now=utc_now())
            if o._workflow_lease_tokens.get(run_id) == token:
                o._workflow_lease_tokens.pop(run_id, None)
            await o.broker.close(run_id)

    async def _collect_sources(self, run_id, record, token, *, enricher=None):
        o = self.o
        s = await o.store.get_snapshot(run_id)
        if review_input_digest(s, "sources") != record.input_sha256 or source_digest() != record.source_sha256:
            raise ValueError("SOURCE_REVIEW_INPUT_CHANGED")
        report = await (enricher or self.source_enricher or PublicSourceEnricher()).collect(s)
        latest = await o.store.get_snapshot(run_id)
        if review_input_digest(latest, "sources") != record.input_sha256 or source_digest() != record.source_sha256:
            raise ValueError("SOURCE_REVIEW_INPUT_CHANGED")
        s = latest
        if (report.run_id != run_id or report.candidate_packet_id != s.candidate_packet.packet_id
                or report.candidate_packet_sha256 != sha256(s.candidate_packet.model_dump_json().encode()).hexdigest()):
            raise ValueError("SOURCE_REPORT_BINDING_MISMATCH")
        report.request_id = record.request_id
        report.request_sha256 = record.request_sha256
        report.input_sha256 = record.input_sha256
        report.source_sha256 = record.source_sha256
        report.base_verdict_id = s.g1_verdict.verdict_id
        report.base_verdict_sha256 = sha256(s.g1_verdict.model_dump_json().encode()).hexdigest()
        s.public_source_reports.append(report)
        completed = record.model_copy(update={"record_id":new_id("review_record"), "status":"completed",
            "source_report_ids":[report.report_id],
            "verdict_id":s.g1_verdict.verdict_id, "finished_at":utc_now(), "usage_end_index":len(s.token_usage),
            "remaining_finding_codes":sorted({f.code for f in s.g1_verdict.findings if not f.resolved})})
        ready = sum(d.status == "available_for_review" for d in report.documents)
        s.status_message = f"공개 원문 자동 보강 완료 · 확보 {ready}/{len(report.documents)}건 · 기존 G1 유지 · 재감사 가능"
        event = ExecutionEvent(run_id=run_id, stage=s.stage, event_type="public_source_review_collected",
            message=s.status_message, node="human_gate.public_sources", artifact_ids=[report.report_id],
            metadata={"artifact_sha256":{report.report_id:sha256(report.model_dump_json().encode()).hexdigest()},
                "document_count":len(report.documents), "available_count":ready,
                "http_calls":sum(len(d.calls) for d in report.documents), "model_calls":0})
        s.events.append(event)
        completed_event = self._prepare_record(s, completed)
        await o.store.commit_review_update(s, artifacts=[(report.report_id,"SourceAcquisitionReport",report),
            (completed.record_id,"G1ReauditRecord",completed)], events=[event,completed_event], lease_token=token)
        await o.broker.publish(event)
        await o.broker.publish(completed_event)

    async def _audit(self, run_id, record, token, *, code_checks=None):
        o = self.o
        s = await o.store.get_snapshot(run_id)
        previous = s.g1_verdict
        previous_checkpoint = await o.store.latest_stage_checkpoint(run_id)
        if input_digest(s) != record.input_sha256 or source_digest() != record.source_sha256:
            raise ValueError("REAUDIT_INPUT_CHANGED")
        agent = o.ma3 if record.mode == "external" else ReviewerProjectionMA3AssuranceAgent(
            settings=o.settings, policy=o.ma3.policy)
        # Latest attempt plus newest successful document per candidate. A later
        # transient outage must not hide a previously verified source context.
        selected = {s.public_source_reports[-1].report_id} if s.public_source_reports else set()
        covered = set()
        for report in reversed(s.public_source_reports):
            fresh = {d.candidate_id for d in report.documents if d.status == "available_for_review"} - covered
            if fresh:
                selected.add(report.report_id)
                covered.update(fresh)
        reports = [r for r in s.public_source_reports if r.report_id in selected]
        for report in reports:
            stored = await o.store.read_artifact(run_id, report.report_id, "SourceAcquisitionReport")
            owners = [r for r in s.g1_reaudits if r.mode == "sources" and r.status == "completed"
                      and report.report_id in r.source_report_ids and r.request_id == report.request_id]
            if len(owners) != 1:
                raise ValueError("SOURCE_REPORT_REQUEST_BINDING_FAILED")
            owner = owners[0]
            stored_owner = await o.store.read_artifact(run_id, owner.record_id, "G1ReauditRecord")
            base = await o.store.read_artifact(run_id, owner.previous_verdict_id, "GateVerdict")
            if (not base or not stored_owner or G1ReauditRecord.model_validate(stored_owner) != owner
                    or not all((report.request_sha256, report.input_sha256, report.source_sha256, report.base_verdict_sha256))
                    or report.request_sha256 != owner.request_sha256 or report.input_sha256 != owner.input_sha256
                    or report.source_sha256 != owner.source_sha256 or report.base_verdict_id != owner.previous_verdict_id
                    or report.base_verdict_sha256 != sha256(GateVerdict.model_validate(base).model_dump_json().encode()).hexdigest()):
                raise ValueError("SOURCE_REPORT_REQUEST_BINDING_FAILED")
            if (not stored or SourceAcquisitionReport.model_validate(stored) != report or report.run_id != run_id
                    or report.candidate_packet_id != s.candidate_packet.packet_id
                    or report.candidate_packet_sha256 != sha256(s.candidate_packet.model_dump_json().encode()).hexdigest()):
                raise ValueError("SOURCE_REPORT_INTEGRITY_FAILED")
        bundle = SupplementalReviewBundle(submissions=s.action_evidence_submissions,
            resolutions=s.action_resolutions, public_source_reports=reports, automatic_code_checks=code_checks or [])
        verdict = await agent.audit_g1(s.brief, s.hypothesis_packet, s.plan_spec,
            s.candidate_packet, supplemental_evidence=bundle,
            on_usage=lambda u: o._record_llm_usage(run_id, u),
            before_review=lambda m, p, n: o._preflight_llm(run_id, m, p, n))
        if not verdict.reviewer_invoked:
            # An unavailable or deliberately disabled reviewer cannot silently
            # erase an earlier external concern.
            retained = [f for f in previous.findings if not f.deterministic and not f.resolved
                        and (f.code, f.artifact_id) not in {(x.code, x.artifact_id) for x in verdict.findings}]
            pending_external = [f for f in previous.findings if not f.deterministic and not f.resolved
                                and f.confirmation_status != "unconfirmed"]
            validations = list(verdict.validations)
            if pending_external or record.mode == "external":
                # A low-severity retained finding alone would let _decide return
                # PASS. A rules-only check has not completed its external review,
                # so explicitly record the missing workflow validation instead
                # of changing scientific finding severities or inventing evidence.
                validations.append(ValidationResult(name="external_review_freshness", passed=False,
                    control_type="repurposing", details={
                        "scope": "human_review_workflow_not_scientific_measurement",
                        "reason_code": "EXTERNAL_REAUDIT_REQUIRED",
                        "remaining_external_finding_ids": [f.finding_id for f in pending_external],
                        "message": ("The requested external review did not complete; approval remains locked."
                                    if record.mode == "external" else
                                    "Previous external concerns require a fresh external review before approval.")}))
            verdict = agent._decide(GateName.G1, [*verdict.findings, *retained],
                validations, reviewer_invoked=False,
                reviewer_note=("외부 재감사 미완료. 규칙 검사 결과와 이전 지적을 보존하며 승인을 잠급니다."
                               if record.mode == "external" else
                               "규칙 재검사 완료. 이전 외부 감사 항목은 새 외부 검토 전까지 유지됩니다."))
        # Usage callbacks wrote the authoritative snapshot during the call.
        latest = await o.store.get_snapshot(run_id)
        if input_digest(latest) != record.input_sha256 or source_digest() != record.source_sha256:
            raise ValueError("REAUDIT_INPUT_CHANGED")
        s = latest
        artifacts = [(verdict.verdict_id, "GateVerdict", verdict)]
        s.g1_verdict = verdict
        if verdict.verdict is GateVerdictCode.PASS:
            s.next_action_packet = None
            s.next_action_progress = None
        else:
            old_packet, old_progress = s.next_action_packet, s.next_action_progress
            packet = build_next_action_packet(run_id=run_id, trigger_stage=RunStage.G1,
                brief=s.brief, verdict=verdict, candidates=s.candidate_packet.candidates,
                checkpoint=previous_checkpoint, terminal_message=verdict.rationale)
            old_findings = {f.finding_id: (f.code, f.artifact_id) for f in previous.findings}
            new_findings = {f.finding_id: (f.code, f.artifact_id) for f in verdict.findings}
            old_actions = {tuple(old_findings.get(i) for i in a.finding_ids): a
                           for a in (old_packet.actions if old_packet else [])}
            for a in packet.actions:
                prior = old_actions.get(tuple(new_findings.get(i) for i in a.finding_ids))
                if prior and prior.kind == a.kind:
                    a.action_id = prior.action_id
            if old_packet:
                packet.preserve_artifact_ids.append(old_packet.packet_id)
                for req in packet.experiment_requests:
                    prior = next((x for x in old_packet.experiment_requests
                                  if x.candidate_id == req.candidate_id
                                  and set(x.required_assays) == set(req.required_assays)), None)
                    if prior:
                        req.request_id = prior.request_id
            artifacts.append((packet.packet_id, "NextActionPacket", packet))
            s.next_action_packet = packet
            progress = NextActionProgress(packet_id=packet.packet_id,
                action_status={a.action_id: ("evidence_attached" if any(
                    x.action_id == a.action_id for x in s.action_evidence_submissions) else "pending") for a in packet.actions},
                experiment_status={r.request_id: ("results_attached" if any(
                    x.experiment_request_id == r.request_id for x in s.action_evidence_submissions) else "awaiting_results") for r in packet.experiment_requests},
                submission_ids=list(old_progress.submission_ids) if old_progress else [],
                resolution_ids=list(old_progress.resolution_ids) if old_progress else [])
            artifacts.append((progress.progress_id, "NextActionProgress", progress))
            s.next_action_progress = progress
        remaining = {(f.code, f.artifact_id) for f in verdict.findings if not f.resolved}
        completed = record.model_copy(update={"record_id": new_id("review_record"),
            "status": "completed", "verdict_id": verdict.verdict_id,
            "cleared_finding_codes": sorted({f.code for f in previous.findings
                if not f.resolved and (f.code, f.artifact_id) not in remaining}),
            "remaining_finding_codes": sorted({code for code, _ in remaining}),
            "finished_at": utc_now(), "usage_end_index": len(s.token_usage)})
        s.status_message = f"G1 재감사 {verdict.verdict.value} · 제출 자료 {len(bundle.submissions)}건 · 연구자 검토 대기"
        completed_event = self._prepare_record(s, completed)
        artifacts.append((completed.record_id, "G1ReauditRecord", completed))
        manifest = await o.store.artifact_manifest(run_id)
        digests = {x["artifact_id"]: x["sha256"] for x in manifest}
        digests.update({i: sha256(payload.model_dump_json().encode()).hexdigest() for i, _, payload in artifacts})
        checkpoint = WorkflowCheckpoint(run_id=run_id, completed_node="g1", stage=s.stage,
            brief=s.brief, hypothesis_packet=s.hypothesis_packet, g0_verdict=s.g0_verdict,
            plan_spec=s.plan_spec, candidate_packet=s.candidate_packet, g1_verdict=verdict,
            rework_order=verdict.rework_order, rework_loops=s.rework_loops,
            artifact_sha256=digests)
        event = ExecutionEvent(run_id=run_id, stage=s.stage, event_type="stage_checkpoint_committed",
            message="새 G1 감사와 연구자 검토 상태를 함께 저장했습니다.", node="human_gate.reaudit",
            artifact_ids=sorted(checkpoint.artifact_sha256),
            metadata={"checkpoint_id": checkpoint.checkpoint_id,
                      "checkpoint_sha256": o.store.checkpoint_digest(checkpoint),
                      "checkpoint_schema_version": checkpoint.schema_version,
                      "workflow_version": checkpoint.workflow_version,
                      "completed_node": "g1", "artifact_sha256": checkpoint.artifact_sha256})
        s.events.append(event)
        await o.store.commit_review_update(s, artifacts=artifacts,
            events=[completed_event, event], lease_token=token, checkpoint=checkpoint)
        await o.broker.publish(completed_event)
        await o.broker.publish(event)
