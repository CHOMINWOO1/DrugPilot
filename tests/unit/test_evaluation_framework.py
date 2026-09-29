from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from drugpilot.evaluation import (
    BenchmarkRunner,
    CasePrediction,
    EvaluationMode,
    EvaluationRun,
    LeakageError,
    load_benchmark,
)
from drugpilot.evaluation.io import query_digest
from drugpilot.evaluation.models import (
    CandidatePrediction,
    ExecutionAttempt,
    ExcludedItem,
    RetrievalIsolationProvenance,
    TargetPrediction,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "evaluation" / "demo_v1" / "benchmark.json"
CHECKSUM = ROOT / "data" / "evaluation" / "demo_v1" / "benchmark.sha256"


def _provenance(case, *, retrieved_digests=(), payload_accessed=False):
    return RetrievalIsolationProvenance(
        system_version="test-system-v1",
        input_digest=query_digest(case.public_query()),
        code_revision="test-revision",
        retrieval_artifact_digests=tuple(retrieved_digests),
        retrieved_source_ids=("Open Targets", "ChEMBL"),
        benchmark_payload_accessed=payload_accessed,
    )


def _predictions(artifact) -> tuple[CasePrediction, ...]:
    aml, tnbc = artifact.dataset.cases
    return (
        CasePrediction(
            case_id=aml.case_id,
            targets=(
                TargetPrediction(target_id="FLT3", rank=1, attempted=True, status="completed"),
                TargetPrediction(target_id="BCL2", rank=2, attempted=True, status="no_candidates"),
                TargetPrediction(target_id="IDH1", rank=3, attempted=True, status="completed"),
                TargetPrediction(target_id="DNMT3A", rank=4, attempted=True, status="failed"),
            ),
            candidates=(
                CandidatePrediction(
                    candidate_id="cand-1",
                    target_id="FLT3",
                    identifiers=("quizartinib",),
                    rank=1,
                    status="priority",
                ),
                CandidatePrediction(
                    candidate_id="cand-decoy",
                    target_id="BCL2",
                    identifiers=("demo decoy",),
                    rank=2,
                ),
                CandidatePrediction(
                    candidate_id="cand-3",
                    target_id="IDH1",
                    identifiers=("ivosidenib",),
                    rank=3,
                ),
            ),
            execution_attempts=(
                ExecutionAttempt(name="ChEMBL", kind="source", status="completed"),
                ExecutionAttempt(name="RDKit", kind="tool", status="completed"),
                ExecutionAttempt(name="Vina", kind="tool", status="unavailable"),
                ExecutionAttempt(name="DrugBank", kind="source", status="skipped"),
            ),
            runtime_ms=12.5,
            provenance=_provenance(aml),
        ),
        CasePrediction(
            case_id=tnbc.case_id,
            targets=(
                TargetPrediction(target_id="TACSTD2", rank=1, attempted=True, status="completed"),
                TargetPrediction(target_id="TOP1", rank=2, attempted=True, status="completed"),
                TargetPrediction(target_id="BRCA1", rank=3, attempted=True, status="no_candidates"),
            ),
            candidates=(
                CandidatePrediction(
                    candidate_id="cand-sg",
                    target_id="TACSTD2",
                    identifiers=("sacituzumab govitecan",),
                    rank=1,
                    status="priority",
                ),
            ),
            execution_attempts=(
                ExecutionAttempt(name="Open Targets", kind="source", status="completed"),
                ExecutionAttempt(name="ADMET-AI", kind="tool", status="completed"),
            ),
            runtime_ms=7.5,
            provenance=_provenance(tnbc),
        ),
    )


def _run(artifact, mode=EvaluationMode.DETERMINISTIC) -> EvaluationRun:
    return EvaluationRun(
        run_id=f"run-{mode.value}",
        mode=mode,
        benchmark_dataset_id=artifact.dataset.dataset_id,
        benchmark_version=artifact.dataset.version,
        benchmark_digest=artifact.digest,
        predictions=_predictions(artifact),
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )


def test_versioned_demo_dataset_loads_with_checksum_and_is_immutable() -> None:
    artifact = load_benchmark(BENCHMARK, checksum_path=CHECKSUM)

    assert artifact.dataset.classification.value == "demo_non_production"
    assert artifact.dataset.labels_are_illustrative is True
    assert artifact.dataset.production_approved is False
    assert len(artifact.digest) == 64
    with pytest.raises(ValidationError):
        artifact.dataset.version = "silently-mutated"  # type: ignore[misc]


def test_metrics_cover_recovery_precision_enrichment_failures_and_runtime() -> None:
    artifact = load_benchmark(BENCHMARK, checksum_path=CHECKSUM)
    report = BenchmarkRunner(k_values=(1, 3)).evaluate(artifact, _run(artifact))
    aml = report.case_scores[0]
    tnbc = report.case_scores[1]

    assert aml.metric("target_recall_at_k", 1).value == 0.5
    assert aml.metric("target_recall_at_k", 3).value == 1.0
    assert aml.metric("known_hit_recovery_at_k", 1).value == 0.5
    assert aml.metric("known_hit_recovery_at_k", 3).value == 1.0
    assert aml.metric("target_precision_at_k", 3).value == pytest.approx(2 / 3, abs=1e-6)
    assert aml.metric("target_enrichment_at_k", 1).value == 4.0
    assert aml.metric("candidate_enrichment_at_k", 1).value == 10.0
    assert aml.metric("target_branch_coverage").value == 1.0
    assert aml.metric("source_tool_failure_rate").value == pytest.approx(1 / 3, abs=1e-6)
    assert aml.metric("source_failure_rate").value == 0.0
    assert aml.metric("tool_failure_rate").value == 0.5
    assert report.metric("runtime_ms_total").value == 20.0

    assert tnbc.metric("target_precision_at_k", 1).value is None
    assert "not exhaustive" in (
        tnbc.metric("target_precision_at_k", 1).reason_unavailable or ""
    )
    # Aggregate precision uses only cases where labels permit the metric.
    assert report.metric("target_precision_at_k", 1).cases_scored == 1


def test_false_exclusion_matches_known_hit_alias() -> None:
    artifact = load_benchmark(BENCHMARK, checksum_path=CHECKSUM)
    run = _run(artifact)
    aml = run.predictions[0].model_copy(
        update={
            "exclusions": (
                ExcludedItem(
                    item_type="target",
                    identifier="IDH1",
                    reason_codes=("TEST",),
                ),
                ExcludedItem(
                    item_type="candidate",
                    identifier="ivosidenib",
                    target_id="IDH1",
                    reason_codes=("TEST",),
                ),
            )
        },
        deep=True,
    )
    changed = run.model_copy(
        update={"predictions": (aml, run.predictions[1])},
        deep=True,
    )

    report = BenchmarkRunner(k_values=(1,)).evaluate(artifact, changed)

    assert report.case_scores[0].metric("target_false_exclusion_rate").value == 0.5
    assert report.case_scores[0].metric("candidate_false_exclusion_rate").value == 0.5


def test_label_digest_or_benchmark_access_fails_closed() -> None:
    artifact = load_benchmark(BENCHMARK, checksum_path=CHECKSUM)
    run = _run(artifact)
    case = artifact.dataset.cases[0]
    poisoned = run.predictions[0].model_copy(
        update={
            "provenance": _provenance(
                case,
                retrieved_digests=(artifact.label_digest_for(case.case_id),),
            )
        },
        deep=True,
    )
    changed = run.model_copy(
        update={"predictions": (poisoned, run.predictions[1])},
        deep=True,
    )

    with pytest.raises(LeakageError, match="benchmark/label digest"):
        BenchmarkRunner().evaluate(artifact, changed)

    accessed = poisoned.model_copy(
        update={"provenance": _provenance(case, payload_accessed=True)},
        deep=True,
    )
    changed = run.model_copy(
        update={"predictions": (accessed, run.predictions[1])},
        deep=True,
    )
    with pytest.raises(LeakageError, match="accessed"):
        BenchmarkRunner().evaluate(artifact, changed)


def test_offline_mode_comparison_needs_no_provider_credentials() -> None:
    artifact = load_benchmark(BENCHMARK, checksum_path=CHECKSUM)
    baseline = _run(artifact, EvaluationMode.DETERMINISTIC)
    llm = _run(artifact, EvaluationMode.LLM)
    hermes = _run(artifact, EvaluationMode.HERMES)

    comparison = BenchmarkRunner(k_values=(1, 3)).compare(
        artifact,
        (baseline, llm, hermes),
    )

    assert {item.mode for item in comparison.reports} == {
        EvaluationMode.DETERMINISTIC,
        EvaluationMode.LLM,
        EvaluationMode.HERMES,
    }
    assert len(comparison.comparisons) == 2
    assert all(
        all(delta.delta == 0 for delta in item.deltas_from_baseline)
        for item in comparison.comparisons
    )


def test_checksum_mismatch_rejects_modified_or_unpinned_dataset() -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        load_benchmark(BENCHMARK, expected_digest="0" * 64)


def test_execute_boundary_passes_only_label_free_queries() -> None:
    artifact = load_benchmark(BENCHMARK, checksum_path=CHECKSUM)

    class CapturingExecutor:
        mode = EvaluationMode.DETERMINISTIC

        def __init__(self) -> None:
            self.seen = []

        def predict(self, query):
            self.seen.append(query)
            return CasePrediction(
                case_id=query.case_id,
                provenance=RetrievalIsolationProvenance(
                    system_version="capture-test-v1",
                    input_digest=query_digest(query),
                ),
            )

    executor = CapturingExecutor()
    result = BenchmarkRunner(k_values=(1,)).execute(
        artifact,
        executor,
        run_id="eval-boundary-test",
    )

    assert len(executor.seen) == len(artifact.dataset.cases)
    assert all(not hasattr(query, "labels") for query in executor.seen)
    assert result.report.retrieval_isolation_verified is True
