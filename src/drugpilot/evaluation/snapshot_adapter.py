from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from drugpilot.api.call_trace import build_run_call_trace
from drugpilot.domain.models import (
    CandidateCard,
    RunSnapshot,
    ScientificExecutionStatus,
    TargetCandidate,
)

from .io import BenchmarkArtifact, query_digest
from .models import (
    CandidatePrediction,
    CasePrediction,
    EvaluationMode,
    EvaluationQuery,
    EvaluationRun,
    ExecutionAttempt,
    ExcludedItem,
    RetrievalIsolationProvenance,
    TargetPrediction,
)


SNAPSHOT_ADAPTER_VERSION = "run-snapshot-readonly-v1"


class SnapshotConversionError(ValueError):
    """Persisted run data cannot be truthfully mapped to an evaluation case."""


def load_run_snapshot(path: str | Path) -> RunSnapshot:
    """Load a RunSnapshot JSON export without opening or mutating SQLite.

    Direct RunSnapshot JSON, ``{"snapshot": {...}}`` API wrappers, and
    ``{"snapshot_json": "..."}`` database-export wrappers are accepted.
    The function never connects to a database or invokes a runtime service.
    """

    source = Path(path).resolve()
    raw: Any = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "snapshot" in raw:
        raw = raw["snapshot"]
    if isinstance(raw, dict) and "snapshot_json" in raw:
        raw = raw["snapshot_json"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    return RunSnapshot.model_validate(raw)


def snapshot_digest(snapshot: RunSnapshot) -> str:
    payload = json.dumps(
        snapshot.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RunSnapshotAdapter:
    """Read-only projection from a persisted run to one label-free result."""

    def __init__(self, *, code_revision: str | None = None) -> None:
        self.code_revision = code_revision

    def convert(
        self,
        snapshot: RunSnapshot,
        query: EvaluationQuery,
    ) -> CasePrediction:
        """Convert using only persisted state and the public benchmark query."""

        _validate_public_input(snapshot, query)
        target_predictions = _target_predictions(snapshot)
        candidate_predictions, exclusions = _candidate_predictions(snapshot)
        attempts = _execution_attempts(snapshot)
        public_trace = build_run_call_trace(snapshot).public_data_calls
        source_ids = _unique(
            item.name for item in public_trace if item.name.strip()
        )
        artifact_digests = _artifact_digests(snapshot)
        runtime_ms = max(
            0.0,
            (snapshot.updated_at - snapshot.created_at).total_seconds() * 1_000,
        )
        return CasePrediction(
            case_id=query.case_id,
            targets=target_predictions,
            candidates=candidate_predictions,
            exclusions=exclusions,
            execution_attempts=attempts,
            runtime_ms=round(runtime_ms, 6),
            provenance=RetrievalIsolationProvenance(
                system_version=SNAPSHOT_ADAPTER_VERSION,
                input_digest=query_digest(query),
                code_revision=self.code_revision,
                retrieval_artifact_digests=artifact_digests,
                retrieved_source_ids=source_ids,
                label_fields_received=(),
                benchmark_payload_accessed=False,
                source_run_id=snapshot.run_id,
                source_run_stage=snapshot.stage.value,
                source_run_mode=snapshot.mode.value,
                source_run_status_message=snapshot.status_message,
                source_run_error=snapshot.error,
                source_snapshot_digest=snapshot_digest(snapshot),
            ),
        )


def build_snapshot_evaluation_run(
    benchmark: BenchmarkArtifact,
    case_snapshots: Mapping[str, RunSnapshot],
    *,
    mode: EvaluationMode,
    run_id: str | None = None,
    code_revision: str | None = None,
) -> EvaluationRun:
    """Build a complete offline EvaluationRun from case-keyed snapshots.

    This scorer-side coordinator sees the benchmark artifact only to create
    each case's ``public_query``.  ``RunSnapshotAdapter`` never receives a
    BenchmarkCase or BenchmarkLabels object.
    """

    expected = {case.case_id for case in benchmark.dataset.cases}
    observed = set(case_snapshots)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SnapshotConversionError(
            f"snapshot mapping must exactly cover benchmark cases; missing={missing}, extra={extra}"
        )
    adapter = RunSnapshotAdapter(code_revision=code_revision)
    predictions = tuple(
        adapter.convert(case_snapshots[case.case_id], case.public_query())
        for case in benchmark.dataset.cases
    )
    stable_material = "|".join(
        (
            benchmark.digest,
            mode.value,
            *(snapshot_digest(case_snapshots[case.case_id]) for case in benchmark.dataset.cases),
        )
    )
    stable_suffix = hashlib.sha256(stable_material.encode("utf-8")).hexdigest()[:16]
    return EvaluationRun(
        run_id=run_id or f"eval_snapshot_{stable_suffix}",
        mode=mode,
        benchmark_dataset_id=benchmark.dataset.dataset_id,
        benchmark_version=benchmark.dataset.version,
        benchmark_digest=benchmark.digest,
        predictions=predictions,
    )


def _validate_public_input(snapshot: RunSnapshot, query: EvaluationQuery) -> None:
    if _text_key(snapshot.brief.disease) != _text_key(query.disease):
        raise SnapshotConversionError(
            "snapshot disease does not match the label-free benchmark query"
        )
    if _text_key(snapshot.brief.research_goal) != _text_key(query.research_goal):
        raise SnapshotConversionError(
            "snapshot research goal does not match the label-free benchmark query"
        )
    requested_depth = next(
        (
            item.value.strip().casefold()
            for item in query.attributes
            if item.name.strip().casefold() == "research_depth"
        ),
        None,
    )
    if (
        requested_depth is not None
        and snapshot.brief.research_depth is not None
        and requested_depth != snapshot.brief.research_depth.casefold()
    ):
        raise SnapshotConversionError(
            "snapshot research depth does not match the label-free benchmark query"
        )


def _target_predictions(snapshot: RunSnapshot) -> tuple[TargetPrediction, ...]:
    hypothesis_by_id = {
        item.hypothesis_id: item
        for item in (
            snapshot.hypothesis_packet.hypotheses
            if snapshot.hypothesis_packet is not None
            else ()
        )
    }
    packet = snapshot.candidate_packet
    summaries = packet.target_executions if packet is not None else []
    predictions: list[TargetPrediction] = []
    seen: set[str] = set()
    for index, summary in enumerate(summaries, start=1):
        target_id, identifiers = _target_identifiers(summary.target, summary.target_key)
        primary = _text_key(target_id)
        if primary in seen:
            continue
        seen.add(primary)
        hypothesis = hypothesis_by_id.get(summary.hypothesis_id)
        raw_status = summary.status
        normalized_status = "ineligible" if raw_status == "duplicate" else raw_status
        predictions.append(
            TargetPrediction(
                target_id=target_id,
                identifiers=identifiers,
                rank=(hypothesis.rank if hypothesis and hypothesis.rank else index),
                attempted=summary.attempted,
                raw_status=raw_status,
                status=normalized_status,
            )
        )
    if predictions or snapshot.hypothesis_packet is None:
        return tuple(predictions)
    for index, hypothesis in enumerate(snapshot.hypothesis_packet.hypotheses, start=1):
        target_id, identifiers = _target_identifiers(hypothesis.target)
        primary = _text_key(target_id)
        if primary in seen:
            continue
        seen.add(primary)
        predictions.append(
            TargetPrediction(
                target_id=target_id,
                identifiers=identifiers,
                rank=hypothesis.rank or index,
                attempted=False,
                raw_status="not_executed",
                status="ineligible",
            )
        )
    return tuple(predictions)


def _candidate_predictions(
    snapshot: RunSnapshot,
) -> tuple[tuple[CandidatePrediction, ...], tuple[ExcludedItem, ...]]:
    packet = snapshot.candidate_packet
    if packet is None:
        return (), ()
    eligible = [item for item in packet.candidates if item.status != "excluded"]
    portfolio_ranks = [item.portfolio_rank for item in eligible]
    has_global_portfolio_ranks = (
        bool(eligible)
        and all(rank is not None for rank in portfolio_ranks)
        and len(set(portfolio_ranks)) == len(portfolio_ranks)
    )
    if has_global_portfolio_ranks:
        global_rank = {
            item.candidate_id: int(item.portfolio_rank or 0) for item in eligible
        }
    else:
        ordered = sorted(
            eligible,
            key=lambda item: (
                -(item.portfolio_score if item.portfolio_score is not None else item.score),
                _text_key(item.target.symbol),
                item.target_rank or item.rank or 10**9,
                _text_key(item.candidate_id),
            ),
        )
        global_rank = {
            item.candidate_id: index for index, item in enumerate(ordered, start=1)
        }

    predictions: list[CandidatePrediction] = []
    exclusions: list[ExcludedItem] = []
    for card in packet.candidates:
        target_id, _ = _target_identifiers(card.target)
        predictions.append(
            CandidatePrediction(
                candidate_id=card.candidate_id,
                target_id=target_id,
                identifiers=_candidate_identifiers(card),
                rank=(None if card.status == "excluded" else global_rank[card.candidate_id]),
                source_rank=card.rank,
                target_rank=card.target_rank,
                portfolio_rank=card.portfolio_rank,
                status=card.status,
            )
        )
        if card.status == "excluded":
            exclusions.append(
                ExcludedItem(
                    item_type="candidate",
                    identifier=card.candidate_id,
                    target_id=target_id,
                    reason_codes=tuple(card.exclusion_reasons or ("EXCLUDED_WITHOUT_REASON",)),
                )
            )
    for summary in packet.target_executions:
        if summary.status not in {"ineligible", "duplicate"}:
            continue
        target_id, _ = _target_identifiers(summary.target, summary.target_key)
        exclusions.append(
            ExcludedItem(
                item_type="target",
                identifier=target_id,
                reason_codes=tuple(summary.reason_codes or (summary.status.upper(),)),
            )
        )
    return tuple(predictions), tuple(exclusions)


def _execution_attempts(snapshot: RunSnapshot) -> tuple[ExecutionAttempt, ...]:
    packet = snapshot.candidate_packet
    candidate_targets = {
        item.candidate_id: item.target.symbol for item in (packet.candidates if packet else ())
    }
    attempts: list[ExecutionAttempt] = []
    if packet is not None:
        trace_candidates: dict[str, list[str]] = {}
        for card in packet.candidates:
            for trace_id in card.tool_trace_ids:
                trace_candidates.setdefault(trace_id, []).append(card.candidate_id)
        for call in packet.tool_calls:
            candidate_ids = tuple(_unique(trace_candidates.get(call.call_id, ())))
            target_id = _single_target(candidate_ids, candidate_targets)
            raw_status = call.execution_status.value
            attempts.append(
                ExecutionAttempt(
                    name=call.tool_name,
                    kind="tool",
                    status=_normalized_attempt_status(raw_status, call.error),
                    target_id=target_id,
                    raw_status=raw_status,
                    operation=call.operation,
                    call_id=call.call_id,
                    duration_ms=max(
                        0.0,
                        (call.finished_at - call.started_at).total_seconds() * 1_000,
                    ),
                    cached=call.cached,
                    fixture=call.fixture,
                    source_version=call.source_version,
                    candidate_ids=candidate_ids,
                    granularity="exact",
                    counts_toward_failure_rate=True,
                )
            )

    public_trace = build_run_call_trace(snapshot).public_data_calls
    for call in public_trace:
        candidate_ids = tuple(_unique(call.candidate_ids))
        attempts.append(
            ExecutionAttempt(
                name=call.name,
                kind="source",
                status=_normalized_attempt_status(call.status, call.detail),
                target_id=_single_target(candidate_ids, candidate_targets),
                raw_status=call.status,
                operation=call.operation,
                call_id=call.call_id,
                duration_ms=call.duration_ms,
                cached=call.cached,
                fixture=call.fixture,
                source_version=call.source_version,
                candidate_ids=candidate_ids,
                granularity=call.granularity,
                counts_toward_failure_rate=call.granularity in {"exact", "aggregate"},
            )
        )
    return tuple(attempts)


def _target_identifiers(
    target: TargetCandidate,
    target_key: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    primary = target.symbol.strip() or target.uniprot_id or target.ensembl_id
    if not primary:
        raise SnapshotConversionError("persisted target has no stable identifier")
    values = (
        target.uniprot_id,
        target.ensembl_id,
        target.target_name,
        target_key,
    )
    return primary, _unique(value for value in values if value and _text_key(value) != _text_key(primary))


_SCALAR_IDENTIFIER_KEYS = frozenset(
    {
        "drug_id",
        "chembl_id",
        "molecule_chembl_id",
        "pubchem_cid",
        "drugcentral_id",
        "resolved_inchikey",
        "standard_inchi_key",
        "inchikey",
        "source_candidate_id",
    }
)
_COLLECTION_IDENTIFIER_KEYS = frozenset(
    {"aliases", "identifiers", "candidate_identifiers"}
)


def _candidate_identifiers(card: CandidateCard) -> tuple[str, ...]:
    values: list[str] = [card.name, card.smiles]
    if card.inchikey:
        values.append(card.inchikey)
    if card.chemistry_execution is not None:
        if card.chemistry_execution.inchikey:
            values.append(card.chemistry_execution.inchikey)
        if card.chemistry_execution.canonical_smiles:
            values.append(card.chemistry_execution.canonical_smiles)
    for key, value in card.metadata.items():
        normalized_key = str(key).strip().casefold()
        if normalized_key in _SCALAR_IDENTIFIER_KEYS and value is not None:
            values.append(str(value))
        elif normalized_key in _COLLECTION_IDENTIFIER_KEYS and isinstance(
            value, (list, tuple, set, frozenset)
        ):
            values.extend(str(item) for item in value if item is not None)
    return _unique(value for value in values if value.strip())


def _artifact_digests(snapshot: RunSnapshot) -> tuple[str, ...]:
    values: list[str] = []
    packet = snapshot.candidate_packet
    if packet is not None:
        values.extend(call.output_hash for call in packet.tool_calls)
        for card in packet.candidates:
            values.extend(
                evidence.excerpt_hash
                for evidence in card.evidence
                if evidence.excerpt_hash is not None
            )
    if snapshot.hypothesis_packet is not None:
        for hypothesis in snapshot.hypothesis_packet.hypotheses:
            values.extend(
                evidence.excerpt_hash
                for evidence in hypothesis.evidence
                if evidence.excerpt_hash is not None
            )
    return _unique(
        value.casefold()
        for value in values
        if re.fullmatch(r"[0-9A-Fa-f]{64}", value or "")
    )


def _normalized_attempt_status(raw_status: str, detail: str | None) -> str:
    value = raw_status.strip().casefold()
    combined = f"{value} {(detail or '').casefold()}"
    if "timeout" in combined:
        return "timeout"
    if value in {"unavailable", "not_available"}:
        return "unavailable"
    if value in {
        "failed",
        "failure",
        "error",
        "blocked",
        "fallback",
        "running",
        ScientificExecutionStatus.PARTIAL.value,
    }:
        return "failed"
    if value in {
        "skipped",
        "not_configured",
        "skipped_missing_identifier",
        "skipped_call_budget",
        "skipped_time_budget",
        "policy_rejected",
    } or value.startswith("skipped"):
        return "skipped"
    return "completed"


def _single_target(
    candidate_ids: tuple[str, ...],
    candidate_targets: Mapping[str, str],
) -> str | None:
    targets = _unique(
        candidate_targets[item]
        for item in candidate_ids
        if item in candidate_targets
    )
    return targets[0] if len(targets) == 1 else None


def _unique(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        key = _text_key(value)
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


__all__ = [
    "RunSnapshotAdapter",
    "SNAPSHOT_ADAPTER_VERSION",
    "SnapshotConversionError",
    "build_snapshot_evaluation_run",
    "load_run_snapshot",
    "snapshot_digest",
]
