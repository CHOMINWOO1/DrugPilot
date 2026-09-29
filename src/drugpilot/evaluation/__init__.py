"""Offline, retrieval-isolated evaluation for DrugPilot result artifacts.

The evaluation package deliberately has no dependency on the runtime graph or
scientific gateways.  Benchmark labels stay on the scorer side of the
boundary; executors receive only :class:`EvaluationQuery` objects.
"""

from .io import BenchmarkArtifact, benchmark_digest, load_benchmark, query_digest
from .models import (
    BenchmarkCase,
    BenchmarkClassification,
    BenchmarkDataset,
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
from .runner import (
    BenchmarkExecutor,
    BenchmarkRunner,
    EvaluationComparison,
    EvaluationReport,
    EvaluationResult,
    LeakageError,
)
from .snapshot_adapter import (
    RunSnapshotAdapter,
    SnapshotConversionError,
    build_snapshot_evaluation_run,
    load_run_snapshot,
    snapshot_digest,
)

__all__ = [
    "BenchmarkArtifact",
    "BenchmarkCase",
    "BenchmarkClassification",
    "BenchmarkDataset",
    "BenchmarkExecutor",
    "BenchmarkRunner",
    "CandidatePrediction",
    "CasePrediction",
    "EvaluationComparison",
    "EvaluationMode",
    "EvaluationQuery",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationRun",
    "ExecutionAttempt",
    "ExcludedItem",
    "LeakageError",
    "RetrievalIsolationProvenance",
    "RunSnapshotAdapter",
    "SnapshotConversionError",
    "TargetPrediction",
    "benchmark_digest",
    "build_snapshot_evaluation_run",
    "load_benchmark",
    "load_run_snapshot",
    "query_digest",
    "snapshot_digest",
]
