from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
import json

import pytest

from drugpilot.tools.validation_trace import (
    EPMCNotExecutedReason,
    FailureComponent,
    FailureReason,
    OperationStage,
    bounded_operation_failure,
    canonical_epmc_not_executed_trace,
)


def _canonical(value):
    return sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("reason", list(EPMCNotExecutedReason))
def test_canonical_zero_call_epmc_trace(reason):
    trace = canonical_epmc_not_executed_trace(reason)
    assert trace["enabled"] is True
    assert trace["status"] == "not_executed"
    assert trace["stop_reason"] == reason.value
    assert trace["http_calls"] == []
    assert trace["http_call_ledger_sha256"] == _canonical([])
    assert trace["observed"]["added_http_call_count"] == 0
    assert trace["observed"]["added_logical_call_count"] == 0
    assert trace["observed"]["added_wire_attempt_count"] == 0
    assert trace["observed"]["response_bytes_total"] == 0
    assert trace["raw_xml_stored"] is False
    assert trace["raw_reference_payload_stored"] is False
    assert trace["full_text_stored"] is False
    digest = trace.pop("trace_sha256")
    assert digest == _canonical(trace)


def test_epmc_reason_is_strict_enum():
    with pytest.raises(TypeError):
        canonical_epmc_not_executed_trace("secret or PMID 123")  # type: ignore[arg-type]


def test_bounded_failure_preserves_only_allowlisted_stages_and_type_hash():
    failure = bounded_operation_failure(
        component=FailureComponent.CANDIDATE_SOURCE,
        failed_stage=OperationStage.PAPER_NATIVE_PROMOTION,
        reason=FailureReason.STRICT_SUPPORT_UNAVAILABLE,
        successful_stages=(
            OperationStage.TARGET_NORMALIZATION,
            OperationStage.TOOL_READINESS,
            OperationStage.MECHANISM_ACTIVITY,
        ),
        exception=RuntimeError("CHEMBL123 PMID 999 secret raw exception"),
    )
    body = failure.as_dict()
    serialized = json.dumps(body)
    assert body["successful_stage_count"] == 3
    assert body["successful_stages"][-1] == "mechanism_activity"
    assert body["error_type_sha256"] == sha256(b"RuntimeError").hexdigest()
    assert "CHEMBL" not in serialized
    assert "PMID" not in serialized
    assert "secret" not in serialized
    digest = body.pop("projection_sha256")
    assert digest == _canonical(body)


def test_bounded_failure_is_immutable():
    failure = bounded_operation_failure(
        component=FailureComponent.MA1_PUBLIC_EVIDENCE,
        failed_stage=OperationStage.MA1_EVIDENCE,
        reason=FailureReason.UPSTREAM_UNAVAILABLE,
        successful_stages=(OperationStage.MA1_DISCOVERY,),
    )
    with pytest.raises(FrozenInstanceError):
        failure.reason = FailureReason.BOUNDED_EXCEPTION  # type: ignore[misc]


@pytest.mark.parametrize("stages", [
    (OperationStage.TARGET_NORMALIZATION, OperationStage.TARGET_NORMALIZATION),
    (OperationStage.PAPER_NATIVE_PROMOTION,),
    tuple([OperationStage.TARGET_NORMALIZATION] * 9),
])
def test_bounded_failure_rejects_duplicate_failed_or_oversize_stages(stages):
    with pytest.raises(ValueError):
        bounded_operation_failure(
            component=FailureComponent.CANDIDATE_SOURCE,
            failed_stage=OperationStage.PAPER_NATIVE_PROMOTION,
            reason=FailureReason.CONTRACT_VALIDATION_FAILED,
            successful_stages=stages,
        )


def test_bounded_failure_rejects_free_form_component_stage_and_reason():
    for field, value in (
        ("component", "candidate CHEMBL123"),
        ("failed_stage", "PMID 123"),
        ("reason", "raw exception text"),
    ):
        kwargs = {
            "component": FailureComponent.CANDIDATE_SOURCE,
            "failed_stage": OperationStage.WRAPPER_VALIDATION,
            "reason": FailureReason.BOUNDED_EXCEPTION,
        }
        kwargs[field] = value
        with pytest.raises(TypeError):
            bounded_operation_failure(**kwargs)  # type: ignore[arg-type]


def test_bounded_failure_rejects_cross_component_or_out_of_order_stages():
    with pytest.raises(ValueError):
        bounded_operation_failure(
            component=FailureComponent.CANDIDATE_SOURCE,
            failed_stage=OperationStage.MA1_EVIDENCE,
            reason=FailureReason.BOUNDED_EXCEPTION,
        )
    with pytest.raises(ValueError):
        bounded_operation_failure(
            component=FailureComponent.CANDIDATE_SOURCE,
            failed_stage=OperationStage.WRAPPER_VALIDATION,
            reason=FailureReason.CONTRACT_VALIDATION_FAILED,
            successful_stages=(OperationStage.MECHANISM_ACTIVITY,),
        )
