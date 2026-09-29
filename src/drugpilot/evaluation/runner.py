from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic import Field, model_validator

from .io import BenchmarkArtifact, query_digest
from .metrics import AggregateMetric, CaseScore, aggregate_scores, score_case
from .models import (
    CasePrediction,
    EvaluationMode,
    EvaluationQuery,
    EvaluationRun,
    FrozenModel,
)


class LeakageError(ValueError):
    """Raised when a prediction cannot prove the scorer/runtime boundary."""


class BenchmarkExecutor(Protocol):
    """Adapter boundary for deterministic, LLM, or Hermes implementations."""

    mode: EvaluationMode

    def predict(self, query: EvaluationQuery) -> CasePrediction:
        """Predict from the label-free public query only."""


class EvaluationReport(FrozenModel):
    schema_version: str = "1.0"
    evaluation_run_id: str
    mode: EvaluationMode
    benchmark_dataset_id: str
    benchmark_version: str
    benchmark_digest: str
    case_scores: tuple[CaseScore, ...]
    aggregate_metrics: tuple[AggregateMetric, ...]
    retrieval_isolation_verified: bool

    def metric(self, name: str, k: int | None = None) -> AggregateMetric:
        return next(
            item
            for item in self.aggregate_metrics
            if item.metric == name and item.k == k
        )


class EvaluationResult(FrozenModel):
    run: EvaluationRun
    report: EvaluationReport


class MetricDelta(FrozenModel):
    metric: str
    k: int | None = Field(default=None, ge=1)
    baseline_value: float
    comparison_value: float
    delta: float


class ModeComparison(FrozenModel):
    mode: EvaluationMode
    run_id: str
    deltas_from_baseline: tuple[MetricDelta, ...]


class EvaluationComparison(FrozenModel):
    benchmark_dataset_id: str
    benchmark_version: str
    benchmark_digest: str
    baseline_mode: EvaluationMode
    baseline_run_id: str
    reports: tuple[EvaluationReport, ...]
    comparisons: tuple[ModeComparison, ...]

    @model_validator(mode="after")
    def validate_modes(self) -> "EvaluationComparison":
        modes = [item.mode for item in self.reports]
        if len(set(modes)) != len(modes):
            raise ValueError("comparison accepts one report per mode")
        return self


class BenchmarkRunner:
    """Execute or score version-pinned benchmark artifacts offline.

    ``execute`` passes only :class:`EvaluationQuery` to the adapter.  For
    reproducible comparisons, ``evaluate`` also accepts previously recorded
    EvaluationRun JSON artifacts, so LLM and Hermes runs can be compared later
    without API credentials being present in the scorer environment.
    """

    def __init__(self, *, k_values: Sequence[int] = (1, 3, 5, 10)) -> None:
        self.k_values = tuple(k_values)

    def execute(
        self,
        benchmark: BenchmarkArtifact,
        executor: BenchmarkExecutor,
        *,
        run_id: str | None = None,
    ) -> EvaluationResult:
        predictions: list[CasePrediction] = []
        for case in benchmark.dataset.cases:
            query = case.public_query()
            started = time.perf_counter()
            prediction = executor.predict(query)
            elapsed_ms = (time.perf_counter() - started) * 1_000
            predictions.append(
                prediction.model_copy(update={"runtime_ms": elapsed_ms}, deep=True)
            )
        run = EvaluationRun(
            run_id=run_id or f"eval_{uuid.uuid4().hex[:16]}",
            mode=executor.mode,
            benchmark_dataset_id=benchmark.dataset.dataset_id,
            benchmark_version=benchmark.dataset.version,
            benchmark_digest=benchmark.digest,
            predictions=tuple(predictions),
        )
        return EvaluationResult(run=run, report=self.evaluate(benchmark, run))

    def evaluate(
        self,
        benchmark: BenchmarkArtifact,
        run: EvaluationRun,
    ) -> EvaluationReport:
        self._validate_run(benchmark, run)
        by_case = {item.case_id: item for item in run.predictions}
        scores = tuple(
            score_case(case, by_case[case.case_id], k_values=self.k_values)
            for case in benchmark.dataset.cases
        )
        return EvaluationReport(
            evaluation_run_id=run.run_id,
            mode=run.mode,
            benchmark_dataset_id=benchmark.dataset.dataset_id,
            benchmark_version=benchmark.dataset.version,
            benchmark_digest=benchmark.digest,
            case_scores=scores,
            aggregate_metrics=aggregate_scores(scores),
            retrieval_isolation_verified=True,
        )

    def compare(
        self,
        benchmark: BenchmarkArtifact,
        runs: Sequence[EvaluationRun],
        *,
        baseline_mode: EvaluationMode = EvaluationMode.DETERMINISTIC,
    ) -> EvaluationComparison:
        if not runs:
            raise ValueError("at least one evaluation run is required")
        reports = tuple(self.evaluate(benchmark, run) for run in runs)
        baseline = next(
            (item for item in reports if item.mode is baseline_mode),
            None,
        )
        if baseline is None:
            raise ValueError(f"comparison requires a {baseline_mode.value} baseline")
        comparisons: list[ModeComparison] = []
        for report in reports:
            if report is baseline:
                continue
            deltas: list[MetricDelta] = []
            for base_metric in baseline.aggregate_metrics:
                other = next(
                    (
                        item
                        for item in report.aggregate_metrics
                        if item.metric == base_metric.metric and item.k == base_metric.k
                    ),
                    None,
                )
                if (
                    other is None
                    or base_metric.value is None
                    or other.value is None
                ):
                    continue
                deltas.append(
                    MetricDelta(
                        metric=base_metric.metric,
                        k=base_metric.k,
                        baseline_value=base_metric.value,
                        comparison_value=other.value,
                        delta=round(other.value - base_metric.value, 6),
                    )
                )
            comparisons.append(
                ModeComparison(
                    mode=report.mode,
                    run_id=report.evaluation_run_id,
                    deltas_from_baseline=tuple(deltas),
                )
            )
        return EvaluationComparison(
            benchmark_dataset_id=benchmark.dataset.dataset_id,
            benchmark_version=benchmark.dataset.version,
            benchmark_digest=benchmark.digest,
            baseline_mode=baseline_mode,
            baseline_run_id=baseline.evaluation_run_id,
            reports=reports,
            comparisons=tuple(comparisons),
        )

    @staticmethod
    def _validate_run(benchmark: BenchmarkArtifact, run: EvaluationRun) -> None:
        dataset = benchmark.dataset
        if run.benchmark_dataset_id != dataset.dataset_id:
            raise ValueError("evaluation run references a different benchmark dataset")
        if run.benchmark_version != dataset.version:
            raise ValueError("evaluation run references a different benchmark version")
        if run.benchmark_digest != benchmark.digest:
            raise ValueError("evaluation run benchmark digest mismatch")
        expected = {case.case_id for case in dataset.cases}
        observed = {prediction.case_id for prediction in run.predictions}
        if observed != expected:
            raise ValueError(
                "evaluation run must contain exactly one prediction for every benchmark case"
            )
        cases = {case.case_id: case for case in dataset.cases}
        for prediction in run.predictions:
            case = cases[prediction.case_id]
            query_hash = query_digest(case.public_query())
            provenance = prediction.provenance
            if provenance.input_digest != query_hash:
                raise LeakageError(
                    f"{case.case_id}: input digest is not the label-free query digest"
                )
            if provenance.label_fields_received:
                raise LeakageError(
                    f"{case.case_id}: executor attested that label fields were received"
                )
            if provenance.benchmark_payload_accessed:
                raise LeakageError(
                    f"{case.case_id}: executor accessed the scorer-side benchmark payload"
                )
            forbidden_digests = {
                benchmark.digest,
                benchmark.label_digest_for(case.case_id),
            }
            overlaps = forbidden_digests.intersection(
                provenance.retrieval_artifact_digests
            )
            if overlaps:
                raise LeakageError(
                    f"{case.case_id}: retrieval provenance contains benchmark/label digest"
                )
            if any(
                _reserved_benchmark_source(source_id)
                for source_id in provenance.retrieved_source_ids
            ):
                raise LeakageError(
                    f"{case.case_id}: benchmark labels were declared as a retrieval source"
                )


def _reserved_benchmark_source(source_id: str) -> bool:
    normalized = source_id.strip().casefold().replace("_", "-")
    return normalized.startswith(("benchmark-label", "evaluation-label", "gold-label"))


__all__ = [
    "BenchmarkExecutor",
    "BenchmarkRunner",
    "EvaluationComparison",
    "EvaluationReport",
    "EvaluationResult",
    "LeakageError",
    "MetricDelta",
    "ModeComparison",
]
