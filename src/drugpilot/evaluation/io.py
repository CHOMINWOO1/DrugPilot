from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .models import BenchmarkDataset, BenchmarkLabels, EvaluationQuery


def _canonical_bytes(value: object) -> bytes:
    if isinstance(value, BaseException):
        raise TypeError("exceptions are not canonical evaluation payloads")
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)  # type: ignore[union-attr]
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def benchmark_digest(dataset: BenchmarkDataset) -> str:
    return hashlib.sha256(_canonical_bytes(dataset)).hexdigest()


def query_digest(query: EvaluationQuery) -> str:
    return hashlib.sha256(_canonical_bytes(query)).hexdigest()


def labels_digest(labels: BenchmarkLabels) -> str:
    return hashlib.sha256(_canonical_bytes(labels)).hexdigest()


@dataclass(frozen=True)
class BenchmarkArtifact:
    dataset: BenchmarkDataset
    digest: str
    source_path: Path

    def label_digest_for(self, case_id: str) -> str:
        case = next(
            (item for item in self.dataset.cases if item.case_id == case_id),
            None,
        )
        if case is None:
            raise KeyError(case_id)
        return labels_digest(case.labels)


def load_benchmark(
    path: str | Path,
    *,
    expected_digest: str | None = None,
    checksum_path: str | Path | None = None,
) -> BenchmarkArtifact:
    """Load and digest a benchmark without exposing it to runtime retrieval.

    A supplied expected digest or checksum sidecar pins both labels and public
    inputs.  Sidecars use the conventional ``<sha256>  <filename>`` format.
    """

    source = Path(path).resolve()
    dataset = BenchmarkDataset.model_validate_json(source.read_text(encoding="utf-8"))
    digest = benchmark_digest(dataset)
    pinned = expected_digest
    if checksum_path is not None:
        checksum = Path(checksum_path).resolve().read_text(encoding="utf-8").strip()
        pinned_from_file = checksum.split()[0].lower() if checksum else ""
        if pinned and pinned.lower() != pinned_from_file:
            raise ValueError("expected digest disagrees with checksum sidecar")
        pinned = pinned_from_file
    if pinned is not None and digest != pinned.lower():
        raise ValueError(
            f"benchmark digest mismatch: expected {pinned.lower()}, calculated {digest}"
        )
    return BenchmarkArtifact(dataset=dataset, digest=digest, source_path=source)


__all__ = [
    "BenchmarkArtifact",
    "benchmark_digest",
    "labels_digest",
    "load_benchmark",
    "query_digest",
]
