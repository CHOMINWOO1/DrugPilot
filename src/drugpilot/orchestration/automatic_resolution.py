"""Bounded, source-preserving follow-up; completion is not automatic approval."""
from __future__ import annotations

import asyncio
from hashlib import sha256

from drugpilot.domain.models import (
    AutomaticResolutionReport, ExecutionEvent, G1ReauditRecord, GateVerdictCode,
    new_id, utc_now,
)
from drugpilot.tools.review_source_enrichment import PublicSourceEnricher, LIMITS as SOURCE_LIMITS
from drugpilot.tools.route_policy import derive_route_evidence

LIMITS = {"source_rounds": 1, "rules_reviews": 1, "external_reviews": 1,
          "elapsed_seconds": 600, "retry_on_ambiguous_paid_response": False}
COUNT_CODE = "REVIEWER_KNOWN_DRUG_COUNT_INCLUDES_NON_ATOMIC_INTERVENTION_LABELS"


def current_report(snapshot):
    """Only a completed parent bound to the current verdict can unlock approval."""
    parent = next((r for r in reversed(snapshot.g1_reaudits) if r.mode == "automatic"), None)
    if parent is None or parent.status != "completed" or snapshot.g1_verdict is None:
        return None
    return next((r for r in reversed(snapshot.automatic_resolution_reports)
        if r.request_id == parent.request_id and r.source_sha256 == parent.source_sha256
        and r.final_verdict_id == parent.verdict_id == snapshot.g1_verdict.verdict_id), None)


def approval_locked(snapshot):
    if not any(r.mode == "automatic" for r in snapshot.g1_reaudits):
        return False
    report = current_report(snapshot)
    return report is None or report.outcome != "READY_FOR_REVIEW"


def code_checks(snapshot):
    """Recompute derived counts from original records, never edit those records."""
    if not any(f.code == COUNT_CODE and not f.resolved for f in snapshot.g1_verdict.findings):
        return []
    checks = []
    for hypothesis in snapshot.hypothesis_packet.hypotheses:
        result = derive_route_evidence(hypothesis, hypothesis.target)
        checks.append({"check":"atomic_named_drug_recalculation", "target":hypothesis.target.symbol,
            "hypothesis_id":hypothesis.hypothesis_id,
            "hypothesis_sha256":sha256(hypothesis.model_dump_json().encode()).hexdigest(),
            "source_evidence_ids":result["known_drug_evidence_ids"],
            "recomputed_count":result["structured_known_drug_count"],
            "count_basis":result["structured_known_drug_count_basis"],
            "accepted_examples":result["known_drug_examples"],
            "excluded_labels":result["excluded_unqualified_drug_labels"],
            "scope":"Derived recalculation only; historical artifacts are unchanged. An external concern is not automatically cleared."})
    return checks


def summarize(snapshot, steps):
    findings = [f for f in snapshot.g1_verdict.findings if not f.resolved and f.severity.value != "INFO"
                and (f.deterministic or f.confirmation_status is None)]
    reasons = []
    if any(s.get("status") == "failed" for s in steps):
        reasons.append("AUTOMATED_STEP_FAILED_NO_AUTOMATIC_REPLAY")
    if any(f.code == "ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION" for f in findings):
        reasons.append("EXPERIMENTAL_VALIDATION_UNAVAILABLE")
    if any(f.code.startswith("ASSAY_VARIANT_") for f in findings):
        reasons.append("EXACT_ASSAY_CONTEXT_NOT_VERIFIED")
    if any(not f.deterministic for f in findings):
        reasons.append("EXTERNAL_REVIEW_CONCERN_UNRESOLVED")
    if snapshot.g1_verdict.verdict is GateVerdictCode.BLOCK:
        return "BLOCKED", ["AUDIT_REJECTED_AUTOMATED_PROGRESSION", *reasons], findings
    if snapshot.g1_verdict.verdict is GateVerdictCode.PASS and not findings and not reasons:
        return "READY_FOR_REVIEW", ["AUTOMATED_CHECKS_COMPLETE_HUMAN_APPROVAL_NOT_GIVEN"], findings
    return "HOLD", [*reasons, "CONFIGURED_PATHS_EXHAUSTED_OR_BUDGET_BOUND"], findings


async def run_automatic(service, run_id, parent, token, *, allow_paid=False):
    """One durable request owns bounded child source/rules/external operations.

    No retries after an ambiguous model result, no candidate substitution, no
    generated experiment, and no mutation of HumanGateDecision.
    """
    from drugpilot.orchestration.human_review import _digest, input_digest, review_input_digest, source_digest
    o = service.o
    initial = await o.store.get_snapshot(run_id)
    candidate_hash = sha256(initial.candidate_packet.model_dump_json().encode()).hexdigest()
    checks, steps, source_ids = [], [], []
    try:
        checks = code_checks(initial)
    except (ValueError, TypeError, KeyError) as exc:
        steps.append({"step":"code_recalculation", "status":"failed", "reason_code":type(exc).__name__})

    async def child(mode):
        current = await o.store.get_snapshot(run_id)
        if sha256(current.candidate_packet.model_dump_json().encode()).hexdigest() != candidate_hash or source_digest() != parent.source_sha256:
            raise ValueError("AUTOMATIC_INPUT_OR_CODE_CHANGED")
        record = G1ReauditRecord(request_id="auto-step-" + sha256((parent.request_id + ":" + mode).encode()).hexdigest(),
            request_sha256=_digest({"parent":parent.request_id, "mode":mode, "allow_paid":allow_paid}),
            input_sha256=review_input_digest(current, mode), source_sha256=source_digest(),
            researcher="DrugPilot 자동 보완", rationale=f"자동 보완의 {mode} 단계 · 원본 근거 보존",
            mode=mode, previous_verdict_id=current.g1_verdict.verdict_id,
            usage_start_index=len(current.token_usage))
        try:
            await service._record(current, record, token)
            if mode == "sources":
                await service._collect_sources(run_id, record, token,
                    enricher=service.source_enricher or PublicSourceEnricher(ncbi_fallback=True))
            else:
                await service._audit(run_id, record, token, code_checks=checks)
        except BaseException as exc:
            current = await o.store.get_snapshot(run_id)
            committed = next((r for r in current.g1_reaudits if r.request_id == record.request_id and r.status == "completed"), None)
            if committed is None:
                if not any(r.request_id == record.request_id for r in current.g1_reaudits):
                    raise
                failed = record.model_copy(update={"record_id":new_id("review_record"), "status":"failed",
                    "error_code":type(exc).__name__, "finished_at":utc_now(), "usage_end_index":len(current.token_usage)})
                await service._record(current, failed, token)
                raise
        current = await o.store.get_snapshot(run_id)
        result = next(r for r in current.g1_reaudits if r.request_id == record.request_id)
        steps.append({"step":mode, "request_id":record.request_id, "status":result.status,
            "verdict_id":result.verdict_id, "source_report_ids":result.source_report_ids})
        source_ids.extend(result.source_report_ids)

    try:
        async with asyncio.timeout(LIMITS["elapsed_seconds"]):
            await child("sources")
            await child("rules")
            current = await o.store.get_snapshot(run_id)
            reports = [r for r in current.public_source_reports if r.report_id in source_ids]
            new_context = any(d.passages for r in reports for d in r.documents if d.identity_verified)
            need_external = bool(new_context or checks) and current.g1_verdict.verdict is not GateVerdictCode.BLOCK
            if need_external and allow_paid and o.settings.automatic_resolution_external_review_enabled and o.ma3.reviewer_provider is not None:
                await child("external")
            else:
                steps.append({"step":"external", "status":"skipped",
                    "reason_code":"NO_NEW_REVIEW_CONTEXT" if not need_external else "PAID_REVIEW_NOT_AUTHORIZED_OR_CONFIGURED"})
    except asyncio.CancelledError:
        # Interrupted/possibly paid work is never replayed as a fresh request.
        raise
    except Exception as exc:
        steps.append({"step":"automatic", "status":"failed", "reason_code":type(exc).__name__})

    current = await o.store.get_snapshot(run_id)
    if sha256(current.candidate_packet.model_dump_json().encode()).hexdigest() != candidate_hash or source_digest() != parent.source_sha256:
        raise ValueError("AUTOMATIC_CANDIDATE_CHANGED")
    outcome, reasons, findings = summarize(current, steps)
    report = AutomaticResolutionReport(run_id=run_id, request_id=parent.request_id,
        source_sha256=parent.source_sha256, candidate_packet_sha256=candidate_hash,
        initial_verdict_id=initial.g1_verdict.verdict_id, final_verdict_id=current.g1_verdict.verdict_id,
        final_input_sha256=input_digest(current),
        outcome=outcome, reason_codes=reasons, steps=steps, source_report_ids=source_ids, code_checks=checks,
        remaining_findings=[{"code":f.code,"candidate_id":f.artifact_id,"severity":f.severity.value,
                             "message":f.message} for f in findings],
        candidate_outcomes=[{"candidate_id":c.candidate_id, "name":c.name,
            "outcome":outcome,
            "finding_codes":[f.code for f in findings if f.artifact_id in {None,c.candidate_id,current.candidate_packet.packet_id}]} for c in current.candidate_packet.candidates],
        usage_start_index=parent.usage_start_index, usage_end_index=len(current.token_usage),
        limits={**LIMITS, "source_limits":dict(SOURCE_LIMITS), "existing_run_cost_cap_usd":current.brief.budget.max_run_cost_usd,
                "paid_authorized":allow_paid})
    current.automatic_resolution_reports.append(report)
    labels = {"HOLD":"보류", "BLOCKED":"진행 불가", "READY_FOR_REVIEW":"검토 준비 완료"}
    current.status_message = f"자동 분석 완료 · {labels[outcome]} · 가용 경로 점검 종료 · 자료 추가는 선택 사항"
    completed = parent.model_copy(update={"record_id":new_id("review_record"), "status":"completed",
        "verdict_id":current.g1_verdict.verdict_id, "source_report_ids":source_ids,
        "remaining_finding_codes":sorted({f.code for f in findings}), "finished_at":utc_now(),
        "usage_end_index":len(current.token_usage)})
    event = ExecutionEvent(run_id=run_id, stage=current.stage, event_type="automatic_resolution_completed",
        node="orchestrator.automatic_resolution", message=current.status_message, artifact_ids=[report.report_id],
        metadata={"artifact_sha256":{report.report_id:sha256(report.model_dump_json().encode()).hexdigest()},
                  "outcome":outcome, "reason_codes":reasons})
    current.events.append(event)
    completed_event = service._prepare_record(current, completed)
    await o.store.commit_review_update(current, artifacts=[(report.report_id,"AutomaticResolutionReport",report),
        (completed.record_id,"G1ReauditRecord",completed)], events=[event,completed_event], lease_token=token)
    await o.broker.publish(event)
    await o.broker.publish(completed_event)
    return report
