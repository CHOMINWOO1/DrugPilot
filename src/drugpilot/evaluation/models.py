from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FrozenModel(BaseModel):
    """Deeply structured evaluation model with immutable model containers.

    All collection fields in this package are tuples rather than mutable lists
    or dictionaries.  Pydantic's frozen flag therefore protects the complete
    benchmark object rather than only its top-level attributes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationMode(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HERMES = "hermes"


class BenchmarkClassification(str, Enum):
    DEMO_NON_PRODUCTION = "demo_non_production"
    VALIDATED_PRODUCTION = "validated_production"


class QueryAttribute(FrozenModel):
    name: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=1_000)


class EvaluationQuery(FrozenModel):
    """The only case payload that a benchmark executor is allowed to see."""

    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1, max_length=160)
    disease: str = Field(min_length=1, max_length=500)
    research_goal: str = Field(min_length=1, max_length=2_000)
    attributes: tuple[QueryAttribute, ...] = ()


class KnownCandidateHit(FrozenModel):
    hit_id: str = Field(min_length=1, max_length=200)
    target_id: str = Field(min_length=1, max_length=120)
    identifiers: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "KnownCandidateHit":
        if len({_norm(item) for item in self.identifiers}) != len(self.identifiers):
            raise ValueError("known-hit identifiers must be unique")
        return self


class BenchmarkLabels(FrozenModel):
    """Scorer-only labels.  These must never enter an EvaluationQuery."""

    relevant_target_ids: tuple[str, ...] = ()
    known_candidate_hits: tuple[KnownCandidateHit, ...] = ()
    expected_target_branches: tuple[str, ...] = ()
    target_labels_complete: bool = False
    candidate_labels_complete: bool = False
    target_universe_size: int | None = Field(default=None, ge=1)
    candidate_universe_size: int | None = Field(default=None, ge=1)
    annotation_note: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_labels(self) -> "BenchmarkLabels":
        _ensure_unique(self.relevant_target_ids, "relevant target")
        _ensure_unique(self.expected_target_branches, "expected target branch")
        hit_ids = tuple(item.hit_id for item in self.known_candidate_hits)
        _ensure_unique(hit_ids, "known hit")
        if (
            self.target_universe_size is not None
            and self.target_universe_size < len(self.relevant_target_ids)
        ):
            raise ValueError("target universe cannot be smaller than relevant labels")
        if (
            self.candidate_universe_size is not None
            and self.candidate_universe_size < len(self.known_candidate_hits)
        ):
            raise ValueError("candidate universe cannot be smaller than known hits")
        return self


class BenchmarkCase(FrozenModel):
    case_id: str = Field(min_length=1, max_length=160)
    disease: str = Field(min_length=1, max_length=500)
    research_goal: str = Field(min_length=1, max_length=2_000)
    attributes: tuple[QueryAttribute, ...] = ()
    labels: BenchmarkLabels

    def public_query(self) -> EvaluationQuery:
        """Return a new label-free query for the system under evaluation."""

        return EvaluationQuery(
            case_id=self.case_id,
            disease=self.disease,
            research_goal=self.research_goal,
            attributes=self.attributes,
        )


class BenchmarkDataset(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=160)
    version: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=500)
    classification: BenchmarkClassification
    description: str = Field(min_length=1, max_length=4_000)
    created_at: datetime
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1)
    labels_are_illustrative: bool
    production_approved: bool

    @model_validator(mode="after")
    def validate_dataset_boundary(self) -> "BenchmarkDataset":
        _ensure_unique(tuple(item.case_id for item in self.cases), "case")
        if self.classification is BenchmarkClassification.DEMO_NON_PRODUCTION:
            if not self.labels_are_illustrative or self.production_approved:
                raise ValueError(
                    "demo datasets require illustrative labels and cannot be production approved"
                )
        if self.classification is BenchmarkClassification.VALIDATED_PRODUCTION:
            if self.labels_are_illustrative or not self.production_approved:
                raise ValueError(
                    "validated datasets require reviewed labels and production approval"
                )
        return self


class TargetPrediction(FrozenModel):
    target_id: str = Field(min_length=1, max_length=160)
    identifiers: tuple[str, ...] = ()
    rank: int | None = Field(default=None, ge=1)
    attempted: bool = False
    raw_status: str | None = Field(default=None, max_length=120)
    status: Literal[
        "completed",
        "no_candidates",
        "partial",
        "unavailable",
        "failed",
        "ineligible",
        "excluded",
    ]

    @model_validator(mode="after")
    def validate_rank(self) -> "TargetPrediction":
        if self.status == "excluded" and self.rank is not None:
            raise ValueError("excluded targets cannot be ranked")
        _ensure_unique(self.identifiers, "target identifier")
        return self


class CandidatePrediction(FrozenModel):
    candidate_id: str = Field(min_length=1, max_length=200)
    target_id: str = Field(min_length=1, max_length=160)
    identifiers: tuple[str, ...] = ()
    rank: int | None = Field(default=None, ge=1)
    source_rank: int | None = Field(default=None, ge=1)
    target_rank: int | None = Field(default=None, ge=1)
    portfolio_rank: int | None = Field(default=None, ge=1)
    status: Literal["priority", "conditional", "excluded"] = "conditional"

    @model_validator(mode="after")
    def validate_rank(self) -> "CandidatePrediction":
        if self.status == "excluded" and self.rank is not None:
            raise ValueError("excluded candidates cannot be ranked")
        if self.status != "excluded" and self.rank is None:
            raise ValueError("non-excluded candidates require a rank")
        _ensure_unique(self.identifiers, "candidate identifier")
        return self


class ExcludedItem(FrozenModel):
    item_type: Literal["target", "candidate"]
    identifier: str = Field(min_length=1, max_length=200)
    target_id: str | None = Field(default=None, max_length=160)
    reason_codes: tuple[str, ...] = Field(min_length=1)


class ExecutionAttempt(FrozenModel):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["source", "tool"]
    status: Literal["completed", "failed", "timeout", "unavailable", "skipped"]
    target_id: str | None = Field(default=None, max_length=160)
    raw_status: str | None = Field(default=None, max_length=120)
    operation: str | None = Field(default=None, max_length=300)
    call_id: str | None = Field(default=None, max_length=200)
    duration_ms: float | None = Field(default=None, ge=0)
    cached: bool = False
    fixture: bool = False
    source_version: str | None = Field(default=None, max_length=1_000)
    candidate_ids: tuple[str, ...] = ()
    granularity: Literal[
        "exact", "aggregate", "observed_records", "selection_decision"
    ] = "exact"
    counts_toward_failure_rate: bool = True


class RetrievalIsolationProvenance(FrozenModel):
    """Attestation checked by the scorer before any metric is calculated."""

    system_version: str = Field(min_length=1, max_length=500)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_revision: str | None = Field(default=None, max_length=200)
    retrieval_artifact_digests: tuple[str, ...] = ()
    retrieved_source_ids: tuple[str, ...] = ()
    label_fields_received: tuple[str, ...] = ()
    benchmark_payload_accessed: bool = False
    source_run_id: str | None = Field(default=None, max_length=200)
    source_run_stage: str | None = Field(default=None, max_length=120)
    source_run_mode: str | None = Field(default=None, max_length=120)
    source_run_status_message: str | None = Field(default=None, max_length=2_000)
    source_run_error: str | None = Field(default=None, max_length=4_000)
    source_snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_digests(self) -> "RetrievalIsolationProvenance":
        for value in self.retrieval_artifact_digests:
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("retrieval artifact digests must be lowercase SHA-256")
        return self


class CasePrediction(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1, max_length=160)
    targets: tuple[TargetPrediction, ...] = ()
    candidates: tuple[CandidatePrediction, ...] = ()
    exclusions: tuple[ExcludedItem, ...] = ()
    execution_attempts: tuple[ExecutionAttempt, ...] = ()
    runtime_ms: float = Field(default=0.0, ge=0)
    provenance: RetrievalIsolationProvenance

    @model_validator(mode="after")
    def validate_predictions(self) -> "CasePrediction":
        _ensure_unique(tuple(item.target_id for item in self.targets), "predicted target")
        _ensure_unique(tuple(item.candidate_id for item in self.candidates), "candidate")
        return self


class EvaluationRun(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1, max_length=200)
    mode: EvaluationMode
    benchmark_dataset_id: str = Field(min_length=1, max_length=160)
    benchmark_version: str = Field(min_length=1, max_length=80)
    benchmark_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions: tuple[CasePrediction, ...] = Field(min_length=1)
    generated_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def validate_cases(self) -> "EvaluationRun":
        _ensure_unique(tuple(item.case_id for item in self.predictions), "prediction case")
        return self


def _norm(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _ensure_unique(values: tuple[str, ...], label: str) -> None:
    if len({_norm(item) for item in values}) != len(values):
        raise ValueError(f"{label} values must be unique")


__all__ = [
    "BenchmarkCase",
    "BenchmarkClassification",
    "BenchmarkDataset",
    "BenchmarkLabels",
    "CandidatePrediction",
    "CasePrediction",
    "EvaluationMode",
    "EvaluationQuery",
    "EvaluationRun",
    "ExecutionAttempt",
    "ExcludedItem",
    "KnownCandidateHit",
    "QueryAttribute",
    "RetrievalIsolationProvenance",
    "TargetPrediction",
]
