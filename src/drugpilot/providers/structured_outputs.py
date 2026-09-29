"""Strict, authority-limited response models for external LLM calls.

These are wire contracts, not persisted scientific artifacts.  Every field is
required (nullable where appropriate), every object forbids extra properties,
and dictionary-shaped fields are represented as key/value arrays.  This keeps
the schemas inside OpenAI Structured Outputs' strict subset while preventing a
model from rewriting deterministic identifiers, evidence, scores, metadata,
permissions, budgets, or gate verdicts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


def _bounded_text(value: str, *, minimum: int, maximum: int, label: str) -> str:
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{label} length must be between {minimum} and {maximum}")
    return value


class StrictLLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntakeValueOutput(StrictLLMOutput):
    field: str
    value: str | None


class IntakeWarningOutput(StrictLLMOutput):
    code: str
    message: str
    fields: list[str]

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _bounded_text(value, minimum=2, maximum=100, label="warning code")

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=2, maximum=1_500, label="warning message"
        )

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str]) -> list[str]:
        if len(value) > 8:
            raise ValueError("warning fields must contain at most 8 entries")
        return value


class IntakeFieldMismatchOutput(StrictLLMOutput):
    code: str
    message: str
    fields: list[str]
    values: list[IntakeValueOutput]

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _bounded_text(value, minimum=2, maximum=100, label="mismatch code")

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=2, maximum=1_500, label="mismatch message"
        )

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str]) -> list[str]:
        if len(value) > 8:
            raise ValueError("mismatch fields must contain at most 8 entries")
        return value

    @model_validator(mode="after")
    def reject_duplicate_values(self) -> "IntakeFieldMismatchOutput":
        names = [item.field for item in self.values]
        if len(names) != len(set(names)):
            raise ValueError("mismatch values contain duplicate field names")
        return self


class DiseaseIntakeSuggestionOutput(StrictLLMOutput):
    suggested_normalized_name: str | None
    explanation: str | None
    warnings: list[IntakeWarningOutput]
    field_mismatches: list[IntakeFieldMismatchOutput]

    @field_validator("suggested_normalized_name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is not None:
            return _bounded_text(
                value, minimum=0, maximum=300, label="suggested disease name"
            )
        return value

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str | None) -> str | None:
        if value is not None:
            return _bounded_text(
                value, minimum=0, maximum=2_000, label="intake explanation"
            )
        return value

    @field_validator("warnings")
    @classmethod
    def validate_warnings(
        cls, value: list[IntakeWarningOutput]
    ) -> list[IntakeWarningOutput]:
        if len(value) > 12:
            raise ValueError("intake output contains more than 12 warnings")
        return value

    @field_validator("field_mismatches")
    @classmethod
    def validate_mismatches(
        cls, value: list[IntakeFieldMismatchOutput]
    ) -> list[IntakeFieldMismatchOutput]:
        if len(value) > 12:
            raise ValueError("intake output contains more than 12 field mismatches")
        return value


class ClaimNarrativeOutput(StrictLLMOutput):
    claim_id: str
    text: str

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=1, maximum=160, label="claim narrative ID"
        )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=2, maximum=2_000, label="claim narrative"
        )


class HypothesisNarrativeOutput(StrictLLMOutput):
    hypothesis_id: str
    mechanism: str
    falsification_test: str
    claims: list[ClaimNarrativeOutput]

    @field_validator("hypothesis_id")
    @classmethod
    def validate_hypothesis_id(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=1, maximum=160, label="hypothesis narrative ID"
        )

    @field_validator("mechanism", "falsification_test")
    @classmethod
    def validate_narrative(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=2, maximum=2_000, label="hypothesis narrative"
        )

    @field_validator("claims")
    @classmethod
    def validate_claim_count(
        cls, value: list[ClaimNarrativeOutput]
    ) -> list[ClaimNarrativeOutput]:
        if len(value) > 64:
            raise ValueError("hypothesis narrative contains more than 64 claims")
        return value


class HypothesisNarrativeOverlayOutput(StrictLLMOutput):
    hypotheses: list[HypothesisNarrativeOutput]

    @field_validator("hypotheses")
    @classmethod
    def validate_hypothesis_count(
        cls, value: list[HypothesisNarrativeOutput]
    ) -> list[HypothesisNarrativeOutput]:
        if len(value) > 64:
            raise ValueError("narrative overlay contains more than 64 hypotheses")
        return value


class ToolPurposeOutput(StrictLLMOutput):
    tool_name: str
    purpose: str


class PlannerTaskNarrativeOutput(StrictLLMOutput):
    task_id: str
    acceptance_criteria: list[str]
    tool_purposes: list[ToolPurposeOutput]

    @field_validator("acceptance_criteria")
    @classmethod
    def validate_criteria(cls, value: list[str]) -> list[str]:
        if len(value) > 24:
            raise ValueError("planner task contains more than 24 criteria")
        return value

    @model_validator(mode="after")
    def reject_duplicate_tools(self) -> "PlannerTaskNarrativeOutput":
        names = [item.tool_name for item in self.tool_purposes]
        if len(names) != len(set(names)):
            raise ValueError("planner task contains duplicate tool names")
        return self


class SourceCoverageSuggestionOutput(StrictLLMOutput):
    source_name: str
    recommend_use: bool
    priority: Literal["high", "medium", "low"]
    rationale: str
    query_focus: str | None

    @field_validator("source_name")
    @classmethod
    def validate_source_name(cls, value: str) -> str:
        return _bounded_text(value, minimum=1, maximum=120, label="source name")

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=2, maximum=1_200, label="source rationale"
        )

    @field_validator("query_focus")
    @classmethod
    def validate_query_focus(cls, value: str | None) -> str | None:
        if value is not None:
            return _bounded_text(
                value, minimum=0, maximum=800, label="source query focus"
            )
        return value


class PlannerNarrativeOverlayOutput(StrictLLMOutput):
    tasks: list[PlannerTaskNarrativeOutput]
    source_suggestions: list[SourceCoverageSuggestionOutput]

    @field_validator("tasks")
    @classmethod
    def validate_task_count(
        cls, value: list[PlannerTaskNarrativeOutput]
    ) -> list[PlannerTaskNarrativeOutput]:
        if len(value) > 32:
            raise ValueError("planner output contains more than 32 tasks")
        return value

    @field_validator("source_suggestions")
    @classmethod
    def validate_source_count(
        cls, value: list[SourceCoverageSuggestionOutput]
    ) -> list[SourceCoverageSuggestionOutput]:
        if len(value) > 40:
            raise ValueError("planner output contains more than 40 source suggestions")
        return value

    @model_validator(mode="after")
    def reject_duplicate_ids(self) -> "PlannerNarrativeOverlayOutput":
        task_ids = [item.task_id for item in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("planner output contains duplicate task IDs")
        source_names = [item.source_name for item in self.source_suggestions]
        if len(source_names) != len(set(source_names)):
            raise ValueError("planner output contains duplicate source names")
        return self


class ReviewerFindingOutput(StrictLLMOutput):
    category: Literal[
        "scientific",
        "model_appropriateness",
        "execution",
        "evidence_provenance",
        "safety",
        "schema",
    ]
    severity: Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    code: str
    message: str
    artifact_id: str | None
    claim_ids: list[str]
    failed_node: str | None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _bounded_text(value, minimum=2, maximum=80, label="review code")

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _bounded_text(
            value, minimum=5, maximum=1_500, label="review message"
        )


class ReviewerAssessmentOutput(StrictLLMOutput):
    summary: str
    additional_findings: list[ReviewerFindingOutput]
    uncertainties: list[str]

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _bounded_text(value, minimum=1, maximum=2_000, label="review summary")

    @field_validator("additional_findings")
    @classmethod
    def validate_finding_count(
        cls, value: list[ReviewerFindingOutput]
    ) -> list[ReviewerFindingOutput]:
        if len(value) > 20:
            raise ValueError("review output contains more than 20 findings")
        return value

    @field_validator("uncertainties")
    @classmethod
    def validate_uncertainty_count(cls, value: list[str]) -> list[str]:
        if len(value) > 20:
            raise ValueError("review output contains more than 20 uncertainties")
        return value


class ReviewerRoutingFindingOutput(ReviewerFindingOutput):
    remediation_kind: Literal["computational_rework", "data_required", "experiment_required",
        "tool_configuration", "resource_required", "researcher_decision", "integrity_block",
        "operator_review"] | None


class ReviewerRoutingAssessmentOutput(ReviewerAssessmentOutput):
    additional_findings: list[ReviewerRoutingFindingOutput]
