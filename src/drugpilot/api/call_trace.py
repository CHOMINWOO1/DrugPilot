from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field

from drugpilot.domain.models import ExecutionEvent, RunSnapshot, TokenUsage


CallKind = Literal["model", "public_data", "scientific_tool"]


class CallTraceItem(BaseModel):
    """Read-only projection of one persisted model/tool execution record.

    ``granularity`` is deliberately explicit.  MA2 scientific tools are
    recorded one invocation at a time, whereas some public-data adapters only
    persist an aggregate source outcome or the records observed downstream.
    The API must not present the latter as an exact HTTP request ledger.
    """

    call_id: str
    kind: CallKind
    agent: str
    node: str
    name: str
    operation: str
    status: str
    cached: bool = False
    fixture: bool = False
    granularity: Literal[
        "exact", "aggregate", "observed_records", "selection_decision"
    ]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    task_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    records_observed: int | None = Field(default=None, ge=0)
    source_version: str | None = None
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CallTraceSummary(BaseModel):
    total_calls: int = 0
    model_calls: int = 0
    public_data_calls: int = 0
    scientific_tool_calls: int = 0
    successful: int = 0
    failed: int = 0
    cached: int = 0
    fixture: int = 0


class RunCallTrace(BaseModel):
    run_id: str
    summary: CallTraceSummary
    model_calls: list[CallTraceItem] = Field(default_factory=list)
    public_data_calls: list[CallTraceItem] = Field(default_factory=list)
    scientific_tool_calls: list[CallTraceItem] = Field(default_factory=list)


_FAILURE_STATUSES = {
    "accounting_unknown",
    "conflict",
    "error",
    "failed",
    "timeout",
    "blocked",
    "unavailable",
}
_SKIP_STATUSES = {
    "skipped",
    "not_configured",
    "skipped_missing_identifier",
    "skipped_call_budget",
    "skipped_time_budget",
    "policy_rejected",
}
_SUCCESS_STATUSES = {
    "success",
    "ok",
    "live",
    "cache",
    "partial",
    "fixture",
    "completed",
    "completed_with_conflict",
    "completed_with_source_issues",
    "completed_with_quarantine",
    "reused_existing",
}


def build_run_call_trace(snapshot: RunSnapshot) -> RunCallTrace:
    """Normalize persisted run data into a compact, audit-friendly call view."""

    model_calls = _model_calls(snapshot)
    public_calls = _public_data_calls(snapshot)
    scientific_calls = _scientific_tool_calls(snapshot)
    all_calls = [*model_calls, *public_calls, *scientific_calls]
    summary = CallTraceSummary(
        total_calls=len(all_calls),
        model_calls=len(model_calls),
        public_data_calls=len(public_calls),
        scientific_tool_calls=len(scientific_calls),
        successful=sum(_is_success(item.status) for item in all_calls),
        failed=sum(_is_failure(item.status) for item in all_calls),
        cached=sum(item.cached for item in all_calls),
        fixture=sum(item.fixture for item in all_calls),
    )
    return RunCallTrace(
        run_id=snapshot.run_id,
        summary=summary,
        model_calls=model_calls,
        public_data_calls=public_calls,
        scientific_tool_calls=scientific_calls,
    )


def _model_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    usages = list(snapshot.token_usage)
    used_usage_indexes: set[int] = set()
    events = list(snapshot.events)
    calls: list[CallTraceItem] = []

    for index, event in enumerate(events):
        if event.event_type != "llm_call_started":
            continue
        trace = _interaction(event)
        model = str(trace.get("model") or _model_from_receiver(trace) or "unknown")
        component = _component_from_event(event, trace)
        terminal = _find_model_terminal(events, index + 1, event.node)
        status = {
            "llm_call_completed": "success",
            "llm_fallback": "fallback",
            "llm_call_failed": "failed",
        }.get(terminal.event_type if terminal else "", "running")
        usage_index, usage = _take_usage(usages, used_usage_indexes, component, model)
        if usage_index is not None:
            used_usage_indexes.add(usage_index)
        if _is_estimated_live_usage(usage):
            status = "accounting_unknown"
        finished = terminal.created_at if terminal else None
        calls.append(
            CallTraceItem(
                call_id=event.event_id,
                kind="model",
                agent=component,
                node=event.node or f"{component.casefold()}.llm",
                name=model,
                operation="structured_generation",
                status=status,
                granularity="exact",
                started_at=event.created_at,
                finished_at=finished,
                duration_ms=_duration_ms(event.created_at, finished),
                trace_ids=[
                    item.event_id for item in (event, terminal) if item is not None
                ],
                detail=(
                    "Conservative usage bound persisted; exact live provider usage "
                    "is unknown and the run was stopped."
                    if _is_estimated_live_usage(usage)
                    else (terminal.message if terminal else event.message)
                ),
                metadata=_usage_metadata(usage),
            )
        )

    # MA3 reviewer calls currently persist usage and the final GateVerdict but
    # do not emit the same start/completion pair used by MA1 and Planner.
    # Keep them visible while marking the record as an aggregate projection.
    # Estimated non-mock usage is never presented as a successful review: it
    # is a conservative bound whose exact provider usage remains unknown.
    for index, usage in enumerate(usages):
        if index in used_usage_indexes:
            continue
        calls.append(
            CallTraceItem(
                call_id=_stable_id(
                    "model",
                    snapshot.run_id,
                    str(index),
                    usage.component,
                    usage.model,
                ),
                kind="model",
                agent=usage.component,
                node=f"{usage.component.casefold()}.reviewer",
                name=usage.model,
                operation="structured_review",
                status=(
                    "accounting_unknown"
                    if _is_estimated_live_usage(usage)
                    else "success"
                ),
                granularity="aggregate",
                detail=(
                    "Conservative usage bound persisted; exact live provider usage "
                    "is unknown and the run was stopped."
                    if _is_estimated_live_usage(usage)
                    else "Persisted token usage; no paired start/completion event is available."
                ),
                metadata=_usage_metadata(usage),
            )
        )
    return calls


def _scientific_tool_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    packet = snapshot.candidate_packet
    if packet is None:
        return []
    task_names = {
        task.task_id: task.name for task in (snapshot.plan_spec.tasks if snapshot.plan_spec else [])
    }
    candidate_links: dict[str, list[str]] = defaultdict(list)
    for candidate in packet.candidates:
        for trace_id in candidate.tool_trace_ids:
            candidate_links[trace_id].append(candidate.candidate_id)

    calls: list[CallTraceItem] = []
    for call in packet.tool_calls:
        fixture = _is_fixture_version(call.source_version)
        calls.append(
            CallTraceItem(
                call_id=call.call_id,
                kind="scientific_tool",
                agent="MA2",
                node=f"ma2.{call.tool_name}",
                name=call.tool_name,
                operation=call.operation,
                status="success" if call.success else "failed",
                cached=call.cached,
                fixture=fixture,
                granularity="exact",
                started_at=call.started_at,
                finished_at=call.finished_at,
                duration_ms=_duration_ms(call.started_at, call.finished_at),
                task_id=call.task_id,
                candidate_ids=sorted(set(candidate_links.get(call.call_id, []))),
                trace_ids=[call.call_id],
                source_version=call.source_version,
                detail=call.error or task_names.get(call.task_id or ""),
                metadata={
                    "input_hash": call.input_hash,
                    "output_hash": call.output_hash,
                    "execution_backend": (
                        "fixture_computation" if fixture else "scientific_tool"
                    ),
                    **_bounded_scientific_failure_metadata(call.provenance),
                },
            )
        )
    return calls


_PAPER_NATIVE_V6_FAILURE_REASON_CODES = frozenset(
    {
        "PAPER_NATIVE_V6_REPURPOSING_CONTRACT_UNAVAILABLE",
        "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT",
        "PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID",
    }
)


def _bounded_scientific_failure_metadata(
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expose bounded machine fields, never arbitrary exception provenance."""

    if not isinstance(provenance, Mapping):
        return {}
    reason_code = provenance.get("reason_code")
    if reason_code not in _PAPER_NATIVE_V6_FAILURE_REASON_CODES:
        return {}

    def bounded_count(value: object, *, minimum: int = 0) -> int | None:
        return (
            value
            if type(value) is int and minimum <= value <= 12
            else None
        )

    projection = provenance.get("wrapper_atomic_screen_projection")
    projection_hash = None
    if isinstance(projection, Mapping):
        candidate_hash = projection.get("attestation_sha256")
        if (
            isinstance(candidate_hash, str)
            and len(candidate_hash) == 64
            and all(character in "0123456789abcdef" for character in candidate_hash)
        ):
            projection_hash = candidate_hash
    grade = provenance.get("evidence_grade")
    return {
        "reason_code": reason_code,
        "support_candidate_count": bounded_count(
            provenance.get("support_candidate_count")
        ),
        "minimum_verified_candidate_count": bounded_count(
            provenance.get("minimum_verified_candidate_count"), minimum=1
        ),
        "evidence_grade": grade if grade in {"B", "C"} else None,
        "promotion_eligible": (
            False if provenance.get("promotion_eligible") is False else None
        ),
        "rejection_count": bounded_count(provenance.get("rejection_count")),
        "wrapper_screen_attestation_sha256": projection_hash,
    }


def _public_data_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    packet = snapshot.hypothesis_packet
    calls: list[CallTraceItem] = []
    if packet is not None:
        calls.extend(_target_discovery_calls(snapshot))
        calls.extend(_target_identity_calls(snapshot))
        calls.extend(_evidence_failure_calls(snapshot))
        calls.extend(_evidence_observation_calls(snapshot))
        calls.extend(_pharos_routing_calls(snapshot))
        calls.extend(_context_router_calls(snapshot))
    calls.extend(_candidate_enrichment_calls(snapshot))
    return calls


def _pharos_routing_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    """Project the Orchestrator's explicit-only Pharos decision and queries."""

    packet = snapshot.hypothesis_packet
    assert packet is not None
    routing = packet.metadata.get("pharos_routing")
    if not isinstance(routing, dict):
        return []

    status = str(routing.get("status") or "unknown")
    selected = routing.get("selected_targets")
    selected_targets = selected if isinstance(selected, list) else []
    raw_outcomes = routing.get("outcomes")
    outcomes = raw_outcomes if isinstance(raw_outcomes, list) else []
    calls = [
        CallTraceItem(
            call_id=_stable_id("pharos-selection", snapshot.run_id, packet.packet_id),
            kind="public_data",
            agent="Orchestrator",
            node="ma1.source_router.pharos",
            name="Pharos",
            operation="source_selection",
            status=status,
            granularity="selection_decision",
            trace_ids=[packet.packet_id],
            records_observed=0,
            detail=str(routing.get("reason") or "") or None,
            metadata={
                "queried": any(
                    isinstance(item, dict) and item.get("queried") is True
                    for item in outcomes
                ),
                "requested": bool(routing.get("requested")),
                "matched_terms": routing.get("matched_terms", []),
                "policy_owner": routing.get("policy_owner"),
                "selection_policy": routing.get("selection_policy"),
                "selected_targets": selected_targets,
                "records_added": int(routing.get("records_added") or 0),
                "records_reused": int(routing.get("records_reused") or 0),
                "records_conflicted": int(
                    routing.get("records_conflicted") or 0
                ),
                "context_only": True,
            },
        )
    ]

    for index, raw in enumerate(outcomes):
        if not isinstance(raw, dict):
            continue
        evidence_ids = [str(value) for value in raw.get("evidence_ids", [])]
        provenance = [str(value) for value in raw.get("provenance_classes", [])]
        observed_provenance = [
            str(value) for value in raw.get("observed_provenance_classes", [])
        ]
        request_provenance = observed_provenance or provenance
        cached = bool(request_provenance) and set(request_provenance) == {
            "external_cache"
        }
        hypothesis_id = str(raw.get("hypothesis_id") or "")
        symbol = str(raw.get("symbol") or "")
        calls.append(
            CallTraceItem(
                call_id=_stable_id(
                    "pharos-query",
                    snapshot.run_id,
                    hypothesis_id,
                    symbol,
                    str(index),
                ),
                kind="public_data",
                agent="MA1",
                node="ma1.source_router.pharos",
                name="Pharos",
                operation="retrieve_target_development_context",
                status=str(raw.get("status") or "unknown"),
                cached=cached,
                fixture=False,
                granularity="aggregate",
                trace_ids=[
                    value
                    for value in (packet.packet_id, hypothesis_id, *evidence_ids)
                    if value
                ],
                records_observed=int(raw.get("records_observed") or 0),
                detail=str(raw.get("reason") or "") or None,
                metadata={
                    "queried": bool(raw.get("queried")),
                    "target_symbol": symbol or None,
                    "track": raw.get("track"),
                    "records_added": int(raw.get("records_added") or 0),
                    "records_reused": int(raw.get("records_reused") or 0),
                    "records_conflicted": int(
                        raw.get("records_conflicted") or 0
                    ),
                    "evidence_ids": evidence_ids,
                    "reused_evidence_ids": raw.get(
                        "reused_evidence_ids", []
                    ),
                    "conflicting_evidence_ids": raw.get(
                        "conflicting_evidence_ids", []
                    ),
                    "provenance_classes": provenance,
                    "observed_provenance_classes": observed_provenance,
                    "effective_tdl": raw.get("effective_tdl"),
                    "tdl_resolution_status": raw.get(
                        "tdl_resolution_status"
                    ),
                    "context_only": True,
                    "failure_category": (
                        _failure_category(str(raw.get("reason") or ""))
                        if _is_failure(str(raw.get("status") or ""))
                        else None
                    ),
                },
            )
        )
    return calls


def _target_discovery_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    packet = snapshot.hypothesis_packet
    assert packet is not None
    calls: list[CallTraceItem] = []
    for index, raw in enumerate(packet.metadata.get("target_source_statuses", [])):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "unknown")
        status = str(raw.get("status") or "unknown")
        operations = [str(value) for value in raw.get("operations", [])]
        failure_codes = [str(value) for value in raw.get("failure_codes", [])]
        legacy_observations = int(raw.get("records_observed") or 0)
        unique_value = raw.get("unique_records_observed")
        unique_observations = (
            int(unique_value)
            if isinstance(unique_value, (int, float))
            and not isinstance(unique_value, bool)
            and unique_value >= 0
            else legacy_observations
        )
        operation_value = raw.get("operation_records_observed")
        operation_observations = (
            int(operation_value)
            if isinstance(operation_value, (int, float))
            and not isinstance(operation_value, bool)
            and operation_value >= 0
            else legacy_observations
        )
        screened_value = raw.get("records_screened")
        screened_observations = (
            int(screened_value)
            if isinstance(screened_value, (int, float))
            and not isinstance(screened_value, bool)
            and screened_value >= 0
            else None
        )
        operation_metrics = [
            dict(value)
            for value in raw.get("operation_metrics", [])
            if isinstance(value, dict)
        ]
        calls.append(
            CallTraceItem(
                call_id=_stable_id("target", snapshot.run_id, source, str(index)),
                kind="public_data",
                agent="MA1",
                node="ma1.target_discovery",
                name=source,
                operation=", ".join(operations) or "target_discovery",
                status=status,
                cached=status == "cache" or int(raw.get("cache_hits") or 0) > 0,
                granularity="aggregate",
                records_observed=unique_observations,
                trace_ids=[packet.packet_id],
                detail="; ".join(failure_codes) or None,
                metadata={
                    "cache_hits": int(raw.get("cache_hits") or 0),
                    "operations": operations,
                    "count_semantics": (
                        "records_observed_is_unique_source_records; "
                        "operation observations are preserved separately"
                    ),
                    "legacy_records_observed": legacy_observations,
                    "operation_records_observed": operation_observations,
                    "unique_records_observed": unique_observations,
                    "records_screened": screened_observations,
                    "operation_metrics": operation_metrics,
                    "failure_codes": failure_codes,
                    "failure_categories": sorted(
                        {_failure_category(value) for value in failure_codes}
                    ),
                },
            )
        )
        # ``partial`` means that at least one source operation succeeded and
        # at least one genuinely failed.  Keep the successful aggregate above,
        # but also expose one failure aggregate instead of silently counting
        # the whole source as a success.
        if status.casefold() == "partial" and failure_codes:
            calls.append(
                CallTraceItem(
                    call_id=_stable_id(
                        "target-failure", snapshot.run_id, source, str(index)
                    ),
                    kind="public_data",
                    agent="MA1",
                    node="ma1.target_discovery",
                    name=source,
                    operation="target_discovery_failed_operation",
                    status="error",
                    granularity="aggregate",
                    records_observed=0,
                    trace_ids=[packet.packet_id],
                    detail="; ".join(failure_codes),
                    metadata={
                        "parent_status": "partial",
                        "failure_codes": failure_codes,
                        "failure_categories": sorted(
                            {_failure_category(value) for value in failure_codes}
                        ),
                    },
                )
            )
    return calls


def _target_identity_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    packet = snapshot.hypothesis_packet
    assert packet is not None
    calls: list[CallTraceItem] = []
    for hypothesis in packet.hypotheses:
        identity = hypothesis.target.metadata.get("identity", {})
        sources = identity.get("sources", []) if isinstance(identity, dict) else []
        for index, raw in enumerate(sources):
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source") or "unknown")
            provenance = str(raw.get("provenance_class") or "")
            fixture = provenance == "fixture_fallback" or source.casefold() == "fixture"
            status = str(raw.get("status") or "unknown")
            error_type = str(raw.get("error_type") or "")
            calls.append(
                CallTraceItem(
                    call_id=_stable_id(
                        "identity", snapshot.run_id, hypothesis.hypothesis_id, source, str(index)
                    ),
                    kind="public_data",
                    agent="MA1",
                    node="ma1.target_identity",
                    name=source,
                    operation="resolve_target_identity",
                    status=status,
                    cached=bool(raw.get("cache_hit")),
                    fixture=fixture,
                    granularity="exact",
                    trace_ids=[hypothesis.hypothesis_id],
                    records_observed=1 if raw.get("status") == "ok" else 0,
                    detail=error_type or None,
                    metadata={
                        "target_symbol": hypothesis.target.symbol,
                        "query": raw.get("query"),
                        "endpoint": raw.get("endpoint"),
                        "matched_by": raw.get("matched_by"),
                        "provenance_class": provenance,
                        "error_type": error_type or None,
                        "failure_category": (
                            _failure_category(error_type)
                            if _is_failure(status)
                            else None
                        ),
                    },
                )
            )
    return calls


def _evidence_observation_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    packet = snapshot.hypothesis_packet
    assert packet is not None
    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for hypothesis in packet.hypotheses:
        for evidence in hypothesis.evidence:
            if evidence.metadata.get("context_only"):
                # These records are projected from the bounded context-router
                # summary below, so do not count them twice here.
                continue
            provenance = str(evidence.metadata.get("provenance_class") or "unknown")
            grouped[(hypothesis.hypothesis_id, evidence.source, provenance)].append(evidence)

    calls: list[CallTraceItem] = []
    for (hypothesis_id, source, provenance), records in sorted(grouped.items()):
        cached = provenance == "external_cache" or any(
            bool(item.metadata.get("cache_hit")) for item in records
        )
        fixture = provenance.startswith("fixture") or any(
            bool(item.metadata.get("mock")) for item in records
        )
        status = "fixture" if fixture else ("cache" if cached else "success")
        calls.append(
            CallTraceItem(
                call_id=_stable_id(
                    "evidence", snapshot.run_id, hypothesis_id, source, provenance
                ),
                kind="public_data",
                agent="MA1",
                node="ma1.evidence_retrieval",
                name=source,
                operation="retrieve_evidence",
                status=status,
                cached=cached,
                fixture=fixture,
                granularity="observed_records",
                trace_ids=[hypothesis_id, *[item.evidence_id for item in records]],
                records_observed=len(records),
                detail="Observed records grouped from the immutable HypothesisPacket.",
                metadata={
                    "provenance_class": provenance,
                    "queries": sorted(
                        {
                            str(item.metadata.get("query"))
                            for item in records
                            if item.metadata.get("query")
                        }
                    ),
                    "retrieval_endpoints": sorted(
                        {
                            str(item.metadata.get("retrieval_endpoint"))
                            for item in records
                            if item.metadata.get("retrieval_endpoint")
                        }
                    ),
                },
            )
        )
    return calls


def _evidence_failure_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    """Expose source outcomes that preceded fallback or partial retrieval.

    An empty query result and an unavailable local configuration are not
    transport failures.  They remain visible as ``empty`` and
    ``not_configured`` respectively, while network/TLS/schema/HTTP failures
    are recorded as ``error``.  Partial source failures are included even when
    another source returned usable evidence, so a successful fallback cannot
    mask the upstream problem.
    """

    packet = snapshot.hypothesis_packet
    assert packet is not None
    outcomes: dict[tuple[str, str, str], tuple[str, bool]] = {}
    for hypothesis in packet.hypotheses:
        for evidence in hypothesis.evidence:
            for field, fallback_followed in (
                ("fallback_reason", True),
                ("partial_source_failures", False),
            ):
                raw = evidence.metadata.get(field)
                if not isinstance(raw, dict):
                    continue
                for source, reason in raw.items():
                    reason_text = str(reason)
                    status = _status_for_source_reason(str(source), reason_text)
                    display_source = (
                        "Public biology"
                        if str(source).casefold() == "configuration"
                        else str(source)
                    )
                    outcomes[(hypothesis.hypothesis_id, display_source, status)] = (
                        reason_text,
                        fallback_followed,
                    )

    return [
        CallTraceItem(
            call_id=_stable_id(
                "evidence-outcome", snapshot.run_id, hypothesis_id, source, status
            ),
            kind="public_data",
            agent="MA1",
            node="ma1.evidence_retrieval",
            name=source,
            operation="retrieve_evidence",
            status=status,
            granularity="aggregate",
            trace_ids=[hypothesis_id],
            records_observed=0,
            detail=reason,
            metadata={
                "fallback_followed": fallback_followed,
                "failure_category": _failure_category(reason),
            },
        )
        for (hypothesis_id, source, status), (reason, fallback_followed) in sorted(
            outcomes.items()
        )
    ]


def _context_router_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    packet = snapshot.hypothesis_packet
    assert packet is not None
    context = packet.metadata.get("context_routing")
    if not isinstance(context, dict):
        return []
    exact_outcomes = context.get("source_outcomes")
    if isinstance(exact_outcomes, list):
        return _context_outcome_calls(snapshot, packet.packet_id, exact_outcomes)

    # Backward-compatible projection for snapshots persisted before individual
    # outcomes were retained.  Do not copy the global routing status onto each
    # source: it cannot identify which source failed.
    outcomes = context.get("source_outcome_counts", {})
    records = context.get("source_record_counts", {})
    if not isinstance(outcomes, dict) or not isinstance(records, dict):
        return []
    calls: list[CallTraceItem] = []
    sources = sorted(set(outcomes) | set(records))
    for source in sources:
        observed = int(records.get(source) or 0)
        calls.append(
            CallTraceItem(
                call_id=_stable_id("context", snapshot.run_id, str(source)),
                kind="public_data",
                agent="MA1",
                node="ma1.context_router",
                name=str(source),
                operation="retrieve_context",
                status="success" if observed else "unknown",
                granularity="aggregate",
                duration_ms=(
                    float(context.get("elapsed_seconds") or 0.0) * 1_000
                    if len(sources) == 1
                    else None
                ),
                trace_ids=[packet.packet_id],
                records_observed=observed,
                detail=(
                    "Legacy context-only source aggregate; per-source status was not persisted."
                ),
                metadata={
                    "outcome_count": int(outcomes.get(source) or 0),
                    "context_only": True,
                    "source_calls_used": context.get("source_calls_used"),
                    "planned_source_calls": context.get("planned_source_calls"),
                },
            )
        )
    status_counts = context.get("status_counts", {})
    if isinstance(status_counts, dict) and int(status_counts.get("error") or 0) > 0:
        calls.append(
            CallTraceItem(
                call_id=_stable_id("context-error", snapshot.run_id, packet.packet_id),
                kind="public_data",
                agent="MA1",
                node="ma1.context_router",
                name="Context router",
                operation="retrieve_context",
                status="error",
                granularity="aggregate",
                trace_ids=[packet.packet_id],
                records_observed=0,
                detail="Legacy snapshot recorded source errors without per-source attribution.",
                metadata={
                    "error_outcome_count": int(status_counts.get("error") or 0),
                    "legacy_projection": True,
                },
            )
        )
    return calls


def _context_outcome_calls(
    snapshot: RunSnapshot,
    packet_id: str,
    outcomes: list[Any],
) -> list[CallTraceItem]:
    calls: list[CallTraceItem] = []
    for index, raw in enumerate(outcomes):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "unknown")
        status = str(raw.get("status") or "unknown")
        queried = bool(raw.get("queried"))
        target_symbol = str(raw.get("target_symbol") or "")
        hypothesis_id = str(raw.get("hypothesis_id") or "")
        error_type = str(raw.get("error_type") or "")
        detail = str(raw.get("detail") or "")
        call_index = raw.get("call_index")
        calls.append(
            CallTraceItem(
                call_id=_stable_id(
                    "context",
                    snapshot.run_id,
                    hypothesis_id,
                    target_symbol,
                    source,
                    str(call_index if call_index is not None else index),
                ),
                kind="public_data",
                agent="MA1",
                node="ma1.context_router",
                name=source,
                operation="retrieve_context",
                status=status,
                cached=bool(raw.get("cache_hit")),
                granularity="exact",
                duration_ms=float(raw.get("elapsed_seconds") or 0.0) * 1_000,
                trace_ids=[value for value in (packet_id, hypothesis_id) if value],
                records_observed=int(raw.get("record_count") or 0),
                detail="; ".join(value for value in (error_type, detail) if value) or None,
                metadata={
                    "queried": queried,
                    "adapter_status": raw.get("adapter_status"),
                    "call_index": call_index,
                    "target_symbol": target_symbol or None,
                    "track": raw.get("track"),
                    "rejected_record_count": int(
                        raw.get("rejected_record_count") or 0
                    ),
                    "error_type": error_type or None,
                    "failure_category": (
                        _failure_category(error_type) if _is_failure(status) else None
                    ),
                },
            )
        )
    return calls


def _candidate_enrichment_calls(snapshot: RunSnapshot) -> list[CallTraceItem]:
    packet = snapshot.candidate_packet
    if packet is None:
        return []
    calls: list[CallTraceItem] = []
    for index, raw in enumerate(packet.metadata.get("public_enrichment_queries", [])):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "public_chemistry_enricher")
        candidate_id = str(raw.get("candidate_id") or "")
        status = str(raw.get("status") or "unknown")
        error_type = str(raw.get("error_type") or "")
        routing_decision = raw.get("record_type") == "source_routing_decision"
        operation = str(
            raw.get("operation")
            or ("source_selection" if routing_decision else "candidate_enrichment")
        )
        detail = str(raw.get("selection_reason") or error_type or "")
        calls.append(
            CallTraceItem(
                call_id=_stable_id(
                    "chemistry", snapshot.run_id, candidate_id, source, str(index)
                ),
                kind="public_data",
                agent="MA2",
                node="ma2.public_enrichment",
                name=source,
                operation=operation,
                status=status,
                cached=bool(raw.get("cache_hit")),
                granularity="selection_decision" if routing_decision else "aggregate",
                candidate_ids=[candidate_id] if candidate_id else [],
                trace_ids=[packet.packet_id],
                records_observed=int(raw.get("record_count") or 0),
                detail=detail or None,
                metadata={
                    "queried": bool(raw.get("queried")),
                    "selected": raw.get("selected"),
                    "record_type": raw.get("record_type"),
                    "selection_reason": raw.get("selection_reason"),
                    "candidate_name": raw.get("candidate_name"),
                    "route": raw.get("route"),
                    "error_type": error_type or None,
                    "failure_category": (
                        _failure_category(error_type) if _is_failure(status) else None
                    ),
                },
            )
        )
    return calls


def _interaction(event: ExecutionEvent) -> dict[str, Any]:
    interaction = event.metadata.get("interaction")
    return interaction if isinstance(interaction, dict) else {}


def _model_from_receiver(trace: dict[str, Any]) -> str | None:
    receiver = str(trace.get("receiver") or "")
    return receiver.split("LLM ·", maxsplit=1)[-1].strip() if "LLM ·" in receiver else None


def _component_from_event(event: ExecutionEvent, trace: dict[str, Any]) -> str:
    sender = str(trace.get("sender") or "").strip()
    if sender and not sender.startswith("LLM ·"):
        return sender
    node = event.node or "unknown.llm"
    return node.split(".", maxsplit=1)[0].upper()


def _find_model_terminal(
    events: list[ExecutionEvent], start: int, node: str | None
) -> ExecutionEvent | None:
    for event in events[start:]:
        if event.node != node:
            continue
        if event.event_type == "llm_call_started":
            return None
        if event.event_type in {"llm_call_completed", "llm_fallback", "llm_call_failed"}:
            return event
    return None


def _take_usage(
    usages: list[TokenUsage],
    used: set[int],
    component: str,
    model: str,
) -> tuple[int | None, TokenUsage | None]:
    for index, usage in enumerate(usages):
        if index in used:
            continue
        if usage.component.casefold() == component.casefold() and usage.model == model:
            return index, usage
    return None, None


def _usage_metadata(usage: TokenUsage | None) -> dict[str, Any]:
    if usage is None:
        return {}
    estimated_live = _is_estimated_live_usage(usage)
    if estimated_live:
        accounting_status = "unknown"
    elif usage.estimated:
        accounting_status = "estimated_mock"
    else:
        accounting_status = "exact"
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "estimated": usage.estimated,
        "accounting_status": accounting_status,
        "usage_bound": "conservative_upper_bound" if estimated_live else None,
        "actual_usage_known": not usage.estimated,
    }


def _is_estimated_live_usage(usage: TokenUsage | None) -> bool:
    return bool(
        usage is not None
        and usage.estimated
        and not usage.model.startswith("mock:")
    )


def _duration_ms(started: datetime | None, finished: datetime | None) -> float | None:
    if started is None or finished is None:
        return None
    return max(0.0, round((finished - started).total_seconds() * 1_000, 3))


def _is_fixture_version(source_version: str | None) -> bool:
    value = (source_version or "").casefold()
    return "fixture" in value or value.startswith("mock")


def _is_success(status: str) -> bool:
    return status.casefold() in _SUCCESS_STATUSES


def _is_failure(status: str) -> bool:
    return status.casefold() in _FAILURE_STATUSES


def _status_for_source_reason(source: str, reason: str) -> str:
    value = reason.casefold().strip()
    if value in {"empty", "empty_result", "no_result", "no_results"}:
        return "empty"
    if (
        source.casefold() == "configuration"
        or "not_configured" in value
        or "not configured" in value
        or "no supported public biology source" in value
    ):
        return "not_configured"
    if value.startswith("skipped"):
        return "skipped"
    return "error"


def _failure_category(reason: str) -> str:
    value = reason.casefold().strip()
    if value in {"empty", "empty_result", "no_result", "no_results"}:
        return "no_results"
    if (
        "not_configured" in value
        or "not configured" in value
        or "no supported public biology source" in value
    ):
        return "configuration"
    if value.startswith("skipped"):
        return "skipped"
    if value.startswith("upstream_http_") or "httpstatus" in value:
        return "upstream_http"
    if any(
        token in value
        for token in (
            "connect",
            "network",
            "timeout",
            "tls",
            "ssl",
            "certificate",
            "proxy",
            "dns",
        )
    ):
        return "connectivity"
    return "adapter_error"


def _stable_id(prefix: str, *values: str) -> str:
    digest = sha256("|".join(values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


__all__ = [
    "CallTraceItem",
    "CallTraceSummary",
    "RunCallTrace",
    "build_run_call_trace",
]
