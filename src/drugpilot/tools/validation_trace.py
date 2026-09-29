"""Canonical, identifier-free validation traces for public-source preflights."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Sequence


_EPMC_POLICY = "epmc_oa_fulltext_one_hop_v1"
_EPMC_EXCERPT_POLICY = "europe_pmc_oa_jats_results_v1"
_EMPTY_LEDGER_SHA256 = sha256(b"[]").hexdigest()
_MAX_SUCCESSFUL_STAGES = 8


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class EPMCNotExecutedReason(StrEnum):
    NO_ELIGIBLE_PARENT = "no_eligible_parent"
    NOT_REQUIRED = "not_required"
    UPSTREAM_EVIDENCE_UNAVAILABLE = "upstream_evidence_unavailable"


def canonical_epmc_not_executed_trace(
    reason: EPMCNotExecutedReason,
) -> dict[str, Any]:
    """Return the canonical enabled zero-call Europe PMC OA trace."""

    if not isinstance(reason, EPMCNotExecutedReason):
        raise TypeError("reason must be EPMCNotExecutedReason")
    trace: dict[str, Any] = {
        "schema_version": "epmc_oa_fulltext_one_hop_trace/v1",
        "policy": _EPMC_POLICY,
        "excerpt_selection_policy": _EPMC_EXCERPT_POLICY,
        "enabled": True,
        "status": "not_executed",
        "stop_reason": reason.value,
        "trigger": {
            "trigger_scope": "pre_escalation_terminal_zero_call",
            "reason_code": reason.value,
        },
        "bounds": {
            "parent_fulltexts": 8,
            "references": 6,
            "child_fulltexts": 2,
            "added_http_calls": 12,
            "added_wire_attempts": 12,
            "bytes_per_document": 4 * 1024 * 1024,
            "aggregate_response_bytes": 12 * 1024 * 1024,
            "elapsed_seconds": 20.0,
            "elapsed_acceptance_deadline_seconds": 20.0,
            "citation_hops": 1,
        },
        "observed": {
            "added_http_call_count": 0,
            "added_logical_call_count": 0,
            "added_wire_attempt_count": 0,
            "response_bytes_total": 0,
            "parent_fulltexts_attempted": 0,
            "parent_fulltexts_parsed": 0,
            "external_jats_doctypes_removed": 0,
            "references_retained": 0,
            "cross_parent_duplicate_references_dropped": 0,
            "child_metadata_checked": 0,
            "child_fulltexts_attempted": 0,
            "child_fulltexts_parsed": 0,
            "result_record_count": 0,
            "elapsed_seconds": 0.0,
        },
        "eligibility_rejections": {},
        "error_reason_codes": [],
        "reference_audits": [],
        "discovery_parents": [],
        "http_calls": [],
        "http_call_ledger_sha256": _EMPTY_LEDGER_SHA256,
        "official_europe_pmc_only": True,
        "forward_citations_followed": False,
        "publisher_or_pdf_calls": 0,
        "fixture_calls": 0,
        "retry_policy": "one_wire_attempt_per_ledger_entry_no_retries",
        "elapsed_limit_semantics": (
            "request_admission_and_response_acceptance_deadline"
        ),
        "elapsed_budget_semantics": "admission_deadline",
        "exact_wall_clock_cap_claimed": False,
        "exact_transport_wall_clock_deadline_claimed": False,
        "response_body_limit_semantics": {
            "document_bytes": "accepted_body_bytes",
            "aggregate_bytes": "accepted_or_bounded_prefix_bytes_read",
            "live_transport": (
                "content_length_precheck_or_stream_read_limit_plus_one_sentinel"
            ),
            "unknown_length_stream_overflow_sentinel_bytes": 1,
            "exact_upstream_network_transfer_bytes_claimed": False,
            "injected_buffered_client_stream_cap_claimed": False,
        },
        "raw_xml_stored": False,
        "raw_reference_payload_stored": False,
        "full_text_stored": False,
    }
    trace["trace_sha256"] = _canonical_sha256(trace)
    return trace


class FailureComponent(StrEnum):
    MA1_PUBLIC_EVIDENCE = "ma1_public_evidence"
    CANDIDATE_SOURCE = "candidate_source"


class OperationStage(StrEnum):
    TARGET_NORMALIZATION = "target_normalization"
    TOOL_READINESS = "tool_readiness"
    MECHANISM_ACTIVITY = "mechanism_activity"
    PAPER_NATIVE_PROMOTION = "paper_native_promotion"
    WRAPPER_VALIDATION = "wrapper_validation"
    MA1_DISCOVERY = "ma1_discovery"
    MA1_EVIDENCE = "ma1_evidence"
    MA1_CLAIM_ASSIGNMENT = "ma1_claim_assignment"


class FailureReason(StrEnum):
    BOUNDED_EXCEPTION = "bounded_exception"
    CONTRACT_VALIDATION_FAILED = "contract_validation_failed"
    STRICT_SUPPORT_UNAVAILABLE = "strict_support_unavailable"
    UPSTREAM_RESPONSE_MALFORMED = "upstream_response_malformed"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"


_COMPONENT_STAGE_ORDER = {
    FailureComponent.CANDIDATE_SOURCE: (
        OperationStage.TARGET_NORMALIZATION,
        OperationStage.TOOL_READINESS,
        OperationStage.MECHANISM_ACTIVITY,
        OperationStage.PAPER_NATIVE_PROMOTION,
        OperationStage.WRAPPER_VALIDATION,
    ),
    FailureComponent.MA1_PUBLIC_EVIDENCE: (
        OperationStage.MA1_DISCOVERY,
        OperationStage.MA1_EVIDENCE,
        OperationStage.MA1_CLAIM_ASSIGNMENT,
    ),
}


@dataclass(frozen=True, slots=True)
class BoundedOperationFailure:
    """Immutable failure projection containing no scientific identifiers."""

    component: FailureComponent
    failed_stage: OperationStage
    reason: FailureReason
    successful_stages: tuple[OperationStage, ...]
    error_type_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": "bounded_operation_failure/v1",
            "component": self.component.value,
            "failed_stage": self.failed_stage.value,
            "reason": self.reason.value,
            "successful_stages": [item.value for item in self.successful_stages],
            "successful_stage_count": len(self.successful_stages),
            "error_type_sha256": self.error_type_sha256,
        }
        body["projection_sha256"] = _canonical_sha256(body)
        return body


def bounded_operation_failure(
    *,
    component: FailureComponent,
    failed_stage: OperationStage,
    reason: FailureReason,
    successful_stages: Sequence[OperationStage] = (),
    exception: BaseException | None = None,
) -> BoundedOperationFailure:
    """Build a fail-closed, bounded projection from allowlisted values only."""

    if not isinstance(component, FailureComponent):
        raise TypeError("component must be FailureComponent")
    if not isinstance(failed_stage, OperationStage):
        raise TypeError("failed_stage must be OperationStage")
    if not isinstance(reason, FailureReason):
        raise TypeError("reason must be FailureReason")
    stages = tuple(successful_stages)
    component_order = _COMPONENT_STAGE_ORDER[component]
    if (
        len(stages) > _MAX_SUCCESSFUL_STAGES
        or any(not isinstance(item, OperationStage) for item in stages)
        or len(stages) != len(set(stages))
        or failed_stage in stages
        or failed_stage not in component_order
        or stages != component_order[: len(stages)]
        or len(stages) > component_order.index(failed_stage)
    ):
        raise ValueError("successful_stages must be the bounded prior-stage prefix")
    error_type_sha256 = (
        sha256(type(exception).__name__.encode("utf-8")).hexdigest()
        if exception is not None
        else None
    )
    return BoundedOperationFailure(
        component=component,
        failed_stage=failed_stage,
        reason=reason,
        successful_stages=stages,
        error_type_sha256=error_type_sha256,
    )


__all__ = [
    "BoundedOperationFailure",
    "EPMCNotExecutedReason",
    "FailureComponent",
    "FailureReason",
    "OperationStage",
    "bounded_operation_failure",
    "canonical_epmc_not_executed_trace",
]
