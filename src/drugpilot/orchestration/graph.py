from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
from contextlib import suppress
from hashlib import sha256
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3_REVIEWER_SYSTEM_PROMPT
from drugpilot.agents.ma3_reviewer_projection import (
    ReviewerProjectionMA3AssuranceAgent,
)
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings, get_settings
from drugpilot.domain.models import (
    CandidatePacket,
    ActionEvidenceSubmission,
    ActionResolution,
    DiseaseIntakeAssessment,
    DiseaseIntakeLLMSuggestion,
    ExecutionEvent,
    EvidenceAttachmentArtifact,
    ExperimentalResultRecord,
    FaultType,
    GateVerdict,
    GateVerdictCode,
    HumanGateDecision,
    HypothesisPacket,
    Hypothesis,
    IntakeFieldMismatch,
    IntakeWarning,
    NextActionPacket,
    NextActionProgress,
    PlanSpec,
    ResearchBrief,
    ReworkOrder,
    RunMode,
    RunSnapshot,
    RunStage,
    TokenUsage,
    WorkflowCheckpoint,
    new_id,
    utc_now,
)
from drugpilot.orchestration.budget import BudgetExceeded, BudgetGuard
from drugpilot.orchestration.dependency import WorkflowDependencyGraph
from drugpilot.orchestration.events import EventBroker
from drugpilot.orchestration.explanations import (
    audit_handoff,
    candidate_handoff,
    hypothesis_handoff,
    interaction,
    plan_handoff,
)
from drugpilot.orchestration.goal_focus import (
    apply_goal_focus,
    derive_goal_focus,
    routing_priorities_for_focus,
)
from drugpilot.orchestration.next_actions import build_next_action_packet
from drugpilot.orchestration.ma1_review_context import (
    apply_ma1_review_goal_focus, needs_review_context_view,
    restore_review_context, review_context_view, review_hypotheses,
)
from drugpilot.orchestration.human_review import serialized_review_mutation
from drugpilot.orchestration.research_selection import eligible_candidate_ids
from drugpilot.domain.cell_context import classify_cell_context
from drugpilot.orchestration.tool_coverage import (
    SourceCoverageLLMSuggestion,
    apply_llm_source_suggestions,
    source_coverage_execution_contract,
)
from drugpilot.providers import (
    LLMProviderCallError,
    ProviderRegistry,
    usage_for_failed_call,
)
from drugpilot.providers.llm import conservative_structured_input_tokens
from drugpilot.providers.structured_outputs import (
    DiseaseIntakeSuggestionOutput,
    HypothesisNarrativeOverlayOutput,
    HypothesisNarrativeOutput,
    ClaimNarrativeOutput,
    PlannerNarrativeOverlayOutput,
    PlannerTaskNarrativeOutput,
    ReviewerRoutingAssessmentOutput,
    ToolPurposeOutput,
)
from drugpilot.storage import CheckpointError, SQLiteRunStore
from drugpilot.storage.sqlite import RunCancellationRequested
from drugpilot.tools.fixture_gateway import DeterministicFixtureGateway
from drugpilot.tools.live_repurposing_gateway import (
    PAPER_NATIVE_V5_REQUIRED_POLICY,
    PAPER_NATIVE_V6_REQUIRED_POLICY,
    PAIRED_V4_REQUIRED_POLICY,
    PublicRepurposingGateway,
)
from drugpilot.tools.protocols import CandidateToolGateway
from drugpilot.tools.route_policy import ROUTE_POLICY_VERSION
from drugpilot.tools.route_policy import (
    select_hypothesis_for_priority,
    select_route_from_evidence,
)


class _UnavailableDockingBackend:
    backend_kind = "live"
    fixture = False
    last_result = None

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        self.source_version = f"docking=unavailable:{reason_code}"

    def readiness(self, target=None) -> dict[str, Any]:
        return {
            "tool": "AutoDock Vina",
            "operation": "score_binding_pose",
            "status": "unavailable",
            "backend": "not_configured",
            "version": None,
            "fixture": False,
            "required": False,
            "reason": self.detail,
            "reason_code": self.reason_code,
            "provenance": {"target_specific_check": target is not None},
        }

    def dock(self, *, smiles: str, target) -> float:
        from drugpilot.tools.executable import ScientificToolUnavailable

        raise ScientificToolUnavailable(f"{self.reason_code}: {self.detail}")


class _UnavailableADMETBackend:
    backend_kind = "live"
    fixture = False
    last_result = None

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        self.source_version = f"admet=unavailable:{reason_code}"

    def readiness(self) -> dict[str, Any]:
        return {
            "tool": "ADMET-AI",
            "operation": "predict_six_endpoint_panel",
            "status": "unavailable",
            "backend": "not_configured",
            "version": None,
            "fixture": False,
            "required": True,
            "reason": self.detail,
            "reason_code": self.reason_code,
            "provenance": {},
        }

    def predict(self, *, smiles: str) -> dict[str, float]:
        from drugpilot.tools.executable import ScientificToolUnavailable

        raise ScientificToolUnavailable(f"{self.reason_code}: {self.detail}")


def build_ma2_gateway(
    settings: Settings,
    *,
    repurposing_evidence_policy: str | None = None,
    public_gateway_factory=None,
    scientific_gateway_factory=None,
    docking_factory=None,
    admet_factory=None,
    profile_registry_factory=None,
) -> CandidateToolGateway:
    """Select the MA2 trust boundary from explicit configuration only.

    LLM activation never changes this choice. Full scientific execution is a
    second explicit opt-in. Each missing executable/model/profile is retained
    as typed unavailable capability rather than replaced by fixture output.
    """

    if (
        settings.ma2_backend_mode == "fixture"
        and repurposing_evidence_policy
        in {PAPER_NATIVE_V5_REQUIRED_POLICY, PAPER_NATIVE_V6_REQUIRED_POLICY}
    ):
        raise ValueError(
            "Exact paper-native policies require a non-fixture live MA2 backend."
        )
    if settings.ma2_backend_mode == "fixture":
        return DeterministicFixtureGateway()
    public_gateway_factory = public_gateway_factory or PublicRepurposingGateway
    dataset_path = (
        settings.drugcentral_dataset_path
        if settings.drugcentral_dataset_path is not None
        else settings.database_path.parent / "drugcentral-not-configured.tsv"
    )
    delegate = None
    build_status: dict[str, Any] = {
        "backend_mode": "live",
        "fixture": False,
        "scientific_tools_requested": settings.ma2_live_scientific_tools_enabled,
        "scientific_delegate_configured": False,
        "component_errors": [],
    }
    if settings.ma2_live_scientific_tools_enabled:
        if scientific_gateway_factory is None:
            from drugpilot.tools.live_ma2_gateway import LiveScientificGateway

            scientific_gateway_factory = LiveScientificGateway
        if admet_factory is None:
            from drugpilot.tools.admet_ai_adapter import ADMETAIAdapter

            admet_factory = ADMETAIAdapter
        if docking_factory is None:
            from drugpilot.tools.vina_adapter import VinaDockingAdapter

            docking_factory = VinaDockingAdapter
        if profile_registry_factory is None:
            from drugpilot.tools.vina_adapter import DockingProfileRegistry

            profile_registry_factory = DockingProfileRegistry.from_json

        try:
            admet = admet_factory(
                python_executable=settings.admet_python_executable,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            admet = _UnavailableADMETBackend(
                "ADMET_INITIALIZATION_UNAVAILABLE",
                reason,
            )
            build_status["component_errors"].append(
                {"component": "admet", "reason": reason}
            )

        profile_path = settings.docking_profile_registry_path
        if profile_path is None or not profile_path.is_file():
            docking = _UnavailableDockingBackend(
                "DOCKING_PROFILE_REGISTRY_UNAVAILABLE",
                "A configured docking profile registry file is required.",
            )
            build_status["component_errors"].append(
                {
                    "component": "docking",
                    "reason": "DOCKING_PROFILE_REGISTRY_UNAVAILABLE",
                }
            )
        else:
            try:
                profiles = profile_registry_factory(profile_path)
                docking = docking_factory(
                    profiles=profiles,
                    vina_executable=settings.vina_binary,
                    python_executable=settings.admet_python_executable,
                    allow_experimental_profiles=(
                        settings.ma2_live_allow_experimental_docking_profiles
                    ),
                    runtime_variant=settings.build_variant,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                docking = _UnavailableDockingBackend(
                    "DOCKING_INITIALIZATION_UNAVAILABLE",
                    reason,
                )
                build_status["component_errors"].append(
                    {"component": "docking", "reason": reason}
                )
        try:
            scientific_gateway_kwargs: dict[str, Any] = {
                "docking": docking,
                "admet": admet,
                "timeout_seconds": settings.ma2_chembl_timeout_seconds,
                "operation_timeout_seconds": (
                    settings.ma2_chembl_operation_timeout_seconds
                ),
                "max_http_attempts": settings.ma2_chembl_max_attempts,
                "retry_backoff_seconds": (
                    settings.ma2_chembl_retry_backoff_seconds
                ),
                "python_executable": settings.admet_python_executable,
            }
            # The new policy changes the delegate's retrieval contract, so it
            # must cross that boundary explicitly.  Default, near-token and
            # paired-v4 construction retain the legacy factory call shape.
            if repurposing_evidence_policy in {
                PAPER_NATIVE_V5_REQUIRED_POLICY,
                PAPER_NATIVE_V6_REQUIRED_POLICY,
            }:
                scientific_gateway_kwargs["repurposing_evidence_policy"] = (
                    repurposing_evidence_policy
                )
            delegate = scientific_gateway_factory(
                **scientific_gateway_kwargs,
            )
            build_status["scientific_delegate_configured"] = True
        except (OSError, RuntimeError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            build_status["component_errors"].append(
                {"component": "scientific_delegate", "reason": reason}
            )
            close = getattr(admet, "close", None)
            if callable(close):
                close()

    public_gateway_kwargs: dict[str, Any] = {
        "drugcentral_dataset_path": dataset_path,
        "drugcentral_dataset_version": settings.drugcentral_dataset_version,
        "timeout_seconds": min(settings.public_evidence_timeout_seconds, 8.0),
        "delegate": delegate,
    }
    # Preserve compatibility with injected legacy factories on the default
    # non-policy path. Run-local exact-policy gateways receive the new keyword.
    if repurposing_evidence_policy is not None:
        public_gateway_kwargs["repurposing_evidence_policy"] = (
            repurposing_evidence_policy
        )
    gateway = public_gateway_factory(
        **public_gateway_kwargs,
    )
    build_status["repurposing_evidence_policy"] = repurposing_evidence_policy
    gateway.build_status = build_status
    return gateway


class WorkflowState(TypedDict, total=False):
    run_id: str
    brief: ResearchBrief
    hypothesis_packet: HypothesisPacket
    g0_verdict: GateVerdict
    plan_spec: PlanSpec
    candidate_packet: CandidatePacket
    g1_verdict: GateVerdict
    rework_order: ReworkOrder | None
    rework_loops: int
    fault_injected: bool
    disease_intake_assessment: DiseaseIntakeAssessment
    intake_abstain_recommended: bool
    route_abstain_recommended: bool
    abstain_reason: str
    _resume_from: str
    _resume_checkpoint_id: str


_COMPREHENSIVE_PHAROS_ROUTING_HINT = (
    "Pharos druggability [deterministic comprehensive coverage]"
)
_EXACT_REPURPOSING_EVIDENCE_POLICIES = (
    PAIRED_V4_REQUIRED_POLICY,
    PAPER_NATIVE_V5_REQUIRED_POLICY,
    PAPER_NATIVE_V6_REQUIRED_POLICY,
)


def _repurposing_evidence_policy_for_priorities(
    human_priorities: list[str],
) -> str | None:
    """Select one exact run-local evidence policy or fail on contradiction."""

    selected = [
        policy
        for policy in _EXACT_REPURPOSING_EVIDENCE_POLICIES
        if policy in human_priorities
    ]
    if len(selected) > 1:
        raise ValueError(
            "Conflicting exact repurposing evidence policies were requested: "
            + ", ".join(selected)
        )
    return selected[0] if selected else None


def _research_depth_routing_priorities(
    brief: ResearchBrief,
    goal_focus: object,
    settings: Settings,
) -> list[str]:
    """Add broad-source hints only from deterministic applicability policy."""

    priorities = routing_priorities_for_focus(brief.human_priorities, goal_focus)
    if (
        settings.evidence_budget_for(brief.research_depth).depth == "comprehensive"
        and brief.constraints.druggability_required
        and "Pharos" in brief.data_policy.allowed_sources
        and _COMPREHENSIVE_PHAROS_ROUTING_HINT not in priorities
    ):
        priorities.append(_COMPREHENSIVE_PHAROS_ROUTING_HINT)
    return priorities


class PlannerTaskNarrative(BaseModel):
    """Non-executable Planner text that an LLM may suggest."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=24)
    tool_purposes: dict[str, str] = Field(default_factory=dict)


class PlannerNarrativeOverlay(BaseModel):
    """Narrow schema that cannot carry PlanSpec permissions or budgets."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[PlannerTaskNarrative] = Field(default_factory=list, max_length=32)
    # The model may annotate known registry entries, but this schema has no
    # source-admission, tool-permission, operation, or budget field.
    source_suggestions: list[SourceCoverageLLMSuggestion] = Field(
        default_factory=list,
        max_length=40,
    )


def _planner_narrative_fallback(plan: PlanSpec) -> PlannerNarrativeOverlay:
    return PlannerNarrativeOverlay(
        tasks=[
            PlannerTaskNarrative(
                task_id=task.task_id,
                acceptance_criteria=list(task.acceptance_criteria),
                tool_purposes={tool.tool_name: tool.purpose for tool in task.tools},
            )
            for task in plan.tasks
        ]
    )


def _planner_output_fallback(plan: PlanSpec) -> PlannerNarrativeOverlayOutput:
    """Build the required strict wire shape from the deterministic PlanSpec."""

    overlay = _planner_narrative_fallback(plan)
    return PlannerNarrativeOverlayOutput(
        tasks=[
            PlannerTaskNarrativeOutput(
                task_id=task.task_id,
                acceptance_criteria=list(task.acceptance_criteria),
                tool_purposes=[
                    ToolPurposeOutput(tool_name=name, purpose=purpose)
                    for name, purpose in task.tool_purposes.items()
                ],
            )
            for task in overlay.tasks
        ],
        source_suggestions=[],
    )


def _planner_overlay_from_output(
    output: PlannerNarrativeOverlayOutput,
) -> PlannerNarrativeOverlay:
    """Convert the strict wire DTO without granting executable authority."""

    task_ids = [item.task_id for item in output.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Planner narrative output contains duplicate task IDs")
    source_names = [item.source_name for item in output.source_suggestions]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Planner narrative output contains duplicate source names")

    tasks: list[PlannerTaskNarrative] = []
    for item in output.tasks:
        tool_names = [tool.tool_name for tool in item.tool_purposes]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(
                f"Planner task {item.task_id} contains duplicate tool names"
            )
        tasks.append(
            PlannerTaskNarrative(
                task_id=item.task_id,
                acceptance_criteria=list(item.acceptance_criteria),
                tool_purposes={
                    tool.tool_name: tool.purpose for tool in item.tool_purposes
                },
            )
        )

    return PlannerNarrativeOverlay(
        tasks=tasks,
        source_suggestions=[
            SourceCoverageLLMSuggestion.model_validate(item.model_dump())
            for item in output.source_suggestions
        ],
    )


def _bounded_narrative(value: object, *, max_chars: int = 1_200) -> str:
    return " ".join(str(value).split())[:max_chars].strip()


def _apply_planner_narrative_overlay(
    deterministic: PlanSpec,
    overlay: PlannerNarrativeOverlay,
) -> PlanSpec:
    """Merge explanatory text without changing the executable plan contract.

    Task names remain deterministic because MA2's current task lookup also uses
    them as dispatch hints.  Deterministic acceptance criteria and tool purposes
    are retained; bounded LLM text may only be appended to those descriptions.
    Unknown task/tool identifiers are ignored.
    """

    suggestions = {item.task_id: item for item in overlay.tasks}
    tasks = []
    for task in deterministic.tasks:
        suggestion = suggestions.get(task.task_id)
        if suggestion is None:
            tasks.append(task.model_copy(deep=True))
            continue

        criteria = list(task.acceptance_criteria)
        for raw in suggestion.acceptance_criteria:
            text = _bounded_narrative(raw)
            if text and text not in criteria:
                criteria.append(text)
            if len(criteria) >= len(task.acceptance_criteria) + 8:
                break

        tools = []
        for assignment in task.tools:
            extra = _bounded_narrative(
                suggestion.tool_purposes.get(assignment.tool_name, "")
            )
            purpose = assignment.purpose
            if extra and extra != purpose:
                purpose = f"{purpose} Additional planning context: {extra}"
            tools.append(assignment.model_copy(update={"purpose": purpose}, deep=True))

        tasks.append(
            task.model_copy(
                update={
                    "acceptance_criteria": criteria,
                    "tools": tools,
                },
                deep=True,
            )
        )
    source_coverage = deterministic.source_coverage
    if source_coverage is not None:
        source_coverage = apply_llm_source_suggestions(
            source_coverage,
            overlay.source_suggestions,
        )
    return deterministic.model_copy(
        update={"tasks": tasks, "source_coverage": source_coverage},
        deep=True,
    )


def _planner_execution_contract(plan: PlanSpec) -> dict[str, Any]:
    """Return every field that can affect dispatch, permissions, or limits."""

    return {
        "schema_version": plan.schema_version,
        "plan_id": plan.plan_id,
        "hypothesis_packet_id": plan.hypothesis_packet_id,
        "revision": plan.revision,
        "approved_tools": list(plan.approved_tools),
        "stop_conditions": list(plan.stop_conditions),
        "candidates_per_route": plan.candidates_per_route,
        "total_token_budget": plan.total_token_budget,
        "research_depth": plan.research_depth,
        "discovery_priority": plan.discovery_priority,
        "requested_discovery_priority": plan.requested_discovery_priority,
        "discovery_priority_derivation": plan.discovery_priority_derivation,
        "forbidden_drugs": list(plan.forbidden_drugs),
        "forbidden_drug_ids": list(plan.forbidden_drug_ids),
        "forbidden_drug_classes": list(plan.forbidden_drug_classes),
        "reserved_control_compounds": [
            item.model_dump(mode="json")
            for item in plan.reserved_control_compounds
        ],
        "source_coverage": source_coverage_execution_contract(plan.source_coverage),
        "created_at": plan.created_at,
        "tasks": [
            {
                "task_id": task.task_id,
                # Task names are included because MA2 currently uses them as
                # deterministic lookup hints when binding tool traces.
                "name": task.name,
                "route": task.route,
                "depends_on": list(task.depends_on),
                "dependency_policy": task.dependency_policy,
                "model_name": task.model_name,
                "max_tokens": task.max_tokens,
                "max_retries": task.max_retries,
                "tools": [
                    {
                        "tool_name": tool.tool_name,
                        "required": tool.required,
                        "allowed_operations": list(tool.allowed_operations),
                    }
                    for tool in task.tools
                ],
            }
            for task in plan.tasks
        ],
    }


_LABEL_TOKEN = re.compile(r"[a-z0-9]+")
_LABEL_STOPWORDS = frozenset(
    {"a", "an", "and", "disease", "of", "the", "type", "unspecified"}
)
_DISEASE_CONTEXT_MARKERS: dict[str, tuple[str, ...]] = {
    "AML": ("aml", "acute myeloid leukemia", "acute myelogenous leukemia"),
    "ALL": ("acute lymphoblastic leukemia", "acute lymphocytic leukemia"),
    "CML": ("cml", "chronic myeloid leukemia", "chronic myelogenous leukemia"),
    "CLL": ("cll", "chronic lymphocytic leukemia"),
    "TNBC": ("tnbc", "triple negative breast cancer"),
    "NSCLC": ("nsclc", "non small cell lung cancer"),
    "SCLC": ("sclc", "small cell lung cancer"),
    "HCC": ("hcc", "hepatocellular carcinoma"),
    "CRC": ("crc", "colorectal cancer"),
    "GBM": ("gbm", "glioblastoma"),
    "PDAC": ("pdac", "pancreatic ductal adenocarcinoma"),
}


def _label_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _LABEL_TOKEN.findall(value.casefold().replace("-", " "))
        if token not in _LABEL_STOPWORDS
    )


def _labels_consistent(query: str, resolved: str) -> bool:
    query_tokens = _label_tokens(query)
    resolved_tokens = _label_tokens(resolved)
    if not query_tokens or not resolved_tokens:
        return False
    if query_tokens == resolved_tokens:
        return True
    query_text = " ".join(query_tokens)
    resolved_text = " ".join(resolved_tokens)
    if query_text in resolved_text or resolved_text in query_text:
        return True
    initialism = "".join(token[0] for token in resolved_tokens if token)
    if len(query_tokens) == 1 and query_tokens[0] == initialism:
        return True
    overlap = set(query_tokens) & set(resolved_tokens)
    return len(overlap) >= 2 and (
        len(overlap) / min(len(set(query_tokens)), len(set(resolved_tokens))) >= 0.6
    )


def _context_marker_groups(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = " " + " ".join(_LABEL_TOKEN.findall(value.casefold())) + " "
    groups: set[str] = set()
    for group, markers in _DISEASE_CONTEXT_MARKERS.items():
        if any(f" {marker} " in normalized for marker in markers):
            groups.add(group)
    return groups


def _deterministic_brief_mismatches(
    brief: ResearchBrief,
    resolved_name: str | None,
) -> list[IntakeFieldMismatch]:
    disease_groups = _context_marker_groups(
        " ".join(part for part in (brief.disease, resolved_name) if part)
    )
    if not disease_groups:
        return []
    mismatches: list[IntakeFieldMismatch] = []
    for field, value in (
        ("research_goal", brief.research_goal),
        ("constraints.target_cell_type", brief.constraints.target_cell_type),
    ):
        field_groups = _context_marker_groups(value)
        if field_groups and field_groups.isdisjoint(disease_groups):
            mismatches.append(
                IntakeFieldMismatch(
                    code="BRIEF_DISEASE_CONTEXT_MISMATCH",
                    message=(
                        f"{field} contains a disease context that conflicts with "
                        "the externally resolved disease; researcher clarification is required."
                    ),
                    fields=["disease", field],
                    values={"disease": brief.disease, field: value},
                )
            )
    return mismatches


def _derive_disease_intake_assessment(
    brief: ResearchBrief,
    packet: HypothesisPacket,
    *,
    demo_fixture_allowed: bool,
) -> DiseaseIntakeAssessment:
    """Convert immutable source results into the authoritative intake verdict."""

    metadata = packet.metadata
    raw_resolution = metadata.get("resolved_disease")
    resolution = raw_resolution if isinstance(raw_resolution, dict) else None
    source_status = str(
        metadata.get("disease_resolution_status") or "not_attempted"
    )
    match_method = str(resolution.get("match_method", "")) if resolution else ""
    externally_resolved = bool(
        resolution
        and source_status == "resolved"
        and match_method
        in {
            "provided_id",
            "exact_name",
            "exact_synonym",
            "unique_high_similarity",
        }
        and resolution.get("disease_id")
        and resolution.get("disease_name")
    )
    fixture_used = bool(metadata.get("target_fallback_used"))
    explicit_demo = bool(demo_fixture_allowed and fixture_used)

    warnings: list[IntakeWarning] = []
    mismatches: list[IntakeFieldMismatch] = []
    reason_codes: list[str] = []
    resolved_id = str(resolution["disease_id"]) if externally_resolved else None
    resolved_name = str(resolution["disease_name"]) if externally_resolved else None

    if externally_resolved and brief.disease_id and not _labels_consistent(
        brief.disease,
        resolved_name or "",
    ):
        mismatches.append(
            IntakeFieldMismatch(
                code="DISEASE_ID_LABEL_MISMATCH",
                message=(
                    "The supplied disease label is inconsistent with the name returned "
                    "for the supplied ontology identifier."
                ),
                fields=["disease", "disease_id"],
                values={
                    "disease": brief.disease,
                    "disease_id": brief.disease_id,
                    "resolved_disease_name": resolved_name,
                },
            )
        )
        reason_codes.append("DISEASE_ID_LABEL_MISMATCH")

    if externally_resolved:
        contextual = _deterministic_brief_mismatches(brief, resolved_name)
        mismatches.extend(contextual)
        if contextual:
            reason_codes.append("BRIEF_FIELD_MISMATCH")

    if explicit_demo and not externally_resolved:
        warnings.append(
            IntakeWarning(
                code="DEMO_FIXTURE_MODE",
                message=(
                    "External disease verification did not succeed; explicitly enabled "
                    "fixture output is for demonstration only."
                ),
                fields=["disease"],
            )
        )
        return DiseaseIntakeAssessment(
            status="warning",
            disease_resolved=False,
            external_verification_status="demo_fixture",
            query_disease=brief.disease,
            requested_disease_id=brief.disease_id,
            explanation=(
                "Demo fixture mode was explicitly enabled. The disease remains unverified "
                "and the generated records must not be interpreted as scientific evidence."
            ),
            abstain_recommended=False,
            reason_codes=["DEMO_FIXTURE_MODE"],
            warnings=warnings,
            field_mismatches=mismatches,
        )

    if not externally_resolved:
        unavailable = source_status in {"source_unavailable", "not_attempted"}
        code = (
            "EVIDENCE_SOURCE_UNAVAILABLE" if unavailable else "UNRESOLVED_DISEASE"
        )
        warnings.append(
            IntakeWarning(
                code=code,
                message=(
                    "The external disease resolver was unavailable; retry verification "
                    "before generating hypotheses."
                    if unavailable
                    else "The disease name was not resolved by the authoritative external source."
                ),
                fields=["disease", "disease_id"],
            )
        )
        return DiseaseIntakeAssessment(
            status="abstain",
            disease_resolved=False,
            external_verification_status=(
                "source_unavailable" if unavailable else "unresolved"
            ),
            query_disease=brief.disease,
            requested_disease_id=brief.disease_id,
            explanation=(
                "No externally verified disease identity is available. The workflow "
                "stops before G0 and candidate generation."
            ),
            abstain_recommended=True,
            reason_codes=[code],
            warnings=warnings,
            field_mismatches=mismatches,
        )

    if reason_codes:
        return DiseaseIntakeAssessment(
            status="abstain",
            disease_resolved=True,
            external_verification_status="resolved",
            query_disease=brief.disease,
            requested_disease_id=brief.disease_id,
            resolved_disease_id=resolved_id,
            resolved_disease_name=resolved_name,
            explanation=(
                "The disease identifier resolved, but conflicting ResearchBrief fields "
                "require researcher clarification before hypothesis generation."
            ),
            abstain_recommended=True,
            reason_codes=list(dict.fromkeys(reason_codes)),
            warnings=warnings,
            field_mismatches=mismatches,
        )

    if fixture_used:
        warnings.append(
            IntakeWarning(
                code="DEMO_FIXTURE_FALLBACK",
                message="Fixture targets supplement this explicitly configured demo run.",
                fields=["disease"],
            )
        )
    if match_method == "unique_high_similarity":
        warnings.append(
            IntakeWarning(
                code="DISEASE_LABEL_TYPO_CORRECTED",
                message=(
                    "The disease label was conservatively corrected to the single "
                    "high-similarity official Open Targets result. Review the resolved "
                    "name before interpreting downstream results."
                ),
                fields=["disease"],
            )
        )
    return DiseaseIntakeAssessment(
        status=(
            "warning"
            if fixture_used or match_method == "unique_high_similarity"
            else "valid"
        ),
        disease_resolved=True,
        external_verification_status="resolved",
        query_disease=brief.disease,
        requested_disease_id=brief.disease_id,
        resolved_disease_id=resolved_id,
        resolved_disease_name=resolved_name,
        explanation=(
            "The disease identity was verified by the external resolver."
            if not fixture_used
            else "The disease identity was verified; fixture supplementation remains demo-only."
        ),
        abstain_recommended=False,
        reason_codes=["DEMO_FIXTURE_FALLBACK"] if fixture_used else [],
        warnings=warnings,
        field_mismatches=mismatches,
    )


def _merge_disease_intake_suggestion(
    deterministic: DiseaseIntakeAssessment,
    suggestion: DiseaseIntakeLLMSuggestion,
) -> DiseaseIntakeAssessment:
    """Merge narrative fields only; deterministic status and IDs stay frozen."""

    llm_used = bool(
        suggestion.suggested_normalized_name
        or suggestion.explanation
        or suggestion.warnings
        or suggestion.field_mismatches
    )
    warnings = [*deterministic.warnings, *suggestion.warnings][:20]
    mismatches = [
        *deterministic.field_mismatches,
        *suggestion.field_mismatches,
    ][:20]
    explanation = deterministic.explanation
    if suggestion.explanation:
        advisory_prefix = " LLM advisory (unverified; cannot change intake status): "
        remaining = max(0, 2_000 - len(explanation) - len(advisory_prefix))
        if remaining:
            explanation = (
                explanation
                + advisory_prefix
                + suggestion.explanation[:remaining]
            )
    return deterministic.model_copy(
        update={
            "suggested_normalized_name": suggestion.suggested_normalized_name,
            "explanation": explanation,
            "warnings": warnings,
            "field_mismatches": mismatches,
            "llm_suggestion_used": llm_used,
        },
        deep=True,
    )


def _intake_suggestion_from_output(
    output: DiseaseIntakeSuggestionOutput,
) -> DiseaseIntakeLLMSuggestion:
    """Convert the strict wire DTO to the existing advisory-only domain model."""

    mismatches: list[IntakeFieldMismatch] = []
    for item in output.field_mismatches:
        value_fields = [value.field for value in item.values]
        if len(value_fields) != len(set(value_fields)):
            raise ValueError(
                f"Intake mismatch {item.code} contains duplicate value fields"
            )
        mismatches.append(
            IntakeFieldMismatch(
                code=item.code,
                message=item.message,
                fields=list(item.fields),
                values={value.field: value.value for value in item.values},
            )
        )
    return DiseaseIntakeLLMSuggestion(
        suggested_normalized_name=output.suggested_normalized_name,
        explanation=output.explanation,
        warnings=[
            IntakeWarning(
                code=item.code,
                message=item.message,
                fields=list(item.fields),
            )
            for item in output.warnings
        ],
        field_mismatches=mismatches,
    )


def _ma1_narrative_fallback(
    packet: HypothesisPacket,
) -> HypothesisNarrativeOverlayOutput:
    """Project an immutable packet into the only fields an LLM may rewrite."""

    return HypothesisNarrativeOverlayOutput(
        hypotheses=[
            HypothesisNarrativeOutput(
                hypothesis_id=hypothesis.hypothesis_id,
                mechanism=hypothesis.mechanism,
                falsification_test=hypothesis.falsification_test,
                claims=[
                    ClaimNarrativeOutput(claim_id=claim.claim_id, text=claim.text)
                    for claim in hypothesis.claims
                ],
            )
            for hypothesis in packet.hypotheses
        ]
    )


MA1_KOREAN_NARRATIVE_INSTRUCTION = (
    "mechanism, falsification_test 및 모든 claim text는 자연스럽고 명확한 "
    "한국어로 작성하세요. 유전자·단백질 기호, 약물명, assay명, "
    "데이터 소스명, 외부 식별자와 표준 과학 단위는 입력에 나온 표기를 "
    "그대로 유지하고 번역·음역·정규화·수정하지 마세요.\n"
    "근거 경계: 각 claim text는 그 claim의 동결된 evidence_ids 및 "
    "counter_evidence_ids가 가리키는 제목·발췌·메타데이터가 직접 뒷받침하는 "
    "내용만 기술하세요. 근거가 연관성, 유병률 또는 예후만 다루면 이를 "
    "표적 억제, 인산화 감소, 세포 생존율 감소, 선택성 또는 치료 반응이 이미 "
    "확인된 결과로 바꾸지 마세요. 제목만 기록된 근거는 제목의 의미 범위를 "
    "넘어 확장하지 말고, 직접 근거가 없으면 불확실성을 명시하세요.\n"
    "가설과 관찰의 구분: mechanism은 검증할 전향적 가설로 표현하세요. "
    "falsification_test는 앞으로 수행할 검증 계획으로 표현하고 이미 확인된 "
    "결과처럼 쓰지 마세요. 연구자 지정 조건에 없는 결과값이나 실험 조건을 "
    "새로 만들지 마세요.\n"
    "운영 조건 보존: 아래 RESEARCHER_OPERATIONAL_REQUIREMENTS의 각 문자열은 "
    "수치, 농도·노출, 채취 시간, assay/readout·phosphosite, normalization, "
    "대조군, 반복 수, 효과 임계값, 불확실성·통계 기준을 포함할 수 있습니다. "
    "비어 있지 않다면 각 문자열을 falsification_test에 원문 그대로 한 번씩 "
    "포함하고, 그 조건에서 어떤 결과가 가설을 기각하는지 명시하세요."
)

_MA1_OPERATIONAL_PRIORITY_PATTERN = re.compile(
    r"(?:"
    r"\bn\s*=\s*\d+\b|"
    r"\d+(?:\.\d+)?\s*(?:%|fold|x|nM|uM|µM|μM|mM|h|hr|hrs|hour|hours|"
    r"min|mins|minute|minutes|day|days|percent)(?!\w)|"
    r"\d+\s+(?:independent\s+)?(?:biological\s+)?replicates?\b|"
    r"(?-i:\bp(?:-|_)?[A-Z][A-Z0-9_-]*\b)|"
    r"\b(?:endpoint|readout|phosphosite|normalization|normalisation|comparator|"
    r"isogenic|replicate|replicates|threshold|confidence interval|p-value|"
    r"statistical criterion|vehicle control|matched control|target-engagement control)\b|"
    r"\b(?:measure|quantify|test|compare|evaluate)\w*\b.{0,80}\b(?:viability|"
    r"proliferation|apoptosis|phosphorylation|target engagement)\b|"
    r"인산화\s*부위|정규화|대조군|비교군|동질유전자|반복\s*(?:수|횟수|\d+회)?|"
    r"임계값|신뢰구간|p\s*값|통계\s*(?:기준|검정)|표적\s*결합\s*대조|"
    r"(?:생존율|증식|세포사멸|인산화).{0,30}(?:측정|정량|검사)|"
    r"(?:측정|정량|검사).{0,30}(?:생존율|증식|세포사멸|인산화)"
    r")",
    re.IGNORECASE,
)


def _ma1_operational_priorities(brief: ResearchBrief) -> tuple[str, ...]:
    """Select researcher-authored experimental conditions without rewriting them."""

    selected: list[str] = []
    for raw in brief.human_priorities:
        priority = raw.strip()
        if (
            priority
            and _MA1_OPERATIONAL_PRIORITY_PATTERN.search(priority)
            and priority not in selected
        ):
            selected.append(priority)
    return tuple(selected)


def _ma1_narrative_prompt_suffix(
    brief: ResearchBrief,
    immutable_packet: HypothesisPacket,
    operational_requirements: tuple[str, ...],
) -> str:
    """Build the strict narrative-only payload with explicit evidence boundaries."""

    return (
        "\nReturn only the permitted narrative overlay for the supplied immutable "
        "HypothesisPacket. Include every supplied hypothesis_id exactly once and, "
        "within it, every supplied claim_id exactly once. You may refine only "
        "mechanism, falsification_test, and claim text. The output schema has no "
        "field for disease, track, target, rank, score, evidence, confidence, "
        "metadata, provenance, citations, tool results, or external identifiers; "
        "never attempt to return or alter them.\n"
        "RESEARCHER_OPERATIONAL_REQUIREMENTS:\n"
        + json.dumps(list(operational_requirements), ensure_ascii=False)
        + "\nRESEARCH_BRIEF:\n"
        + brief.model_dump_json()
        + "\nNARRATIVE_EVIDENCE_VIEW:\n"
        + json.dumps(
            _ma1_narrative_evidence_view(immutable_packet),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


_MA1_NARRATIVE_SOURCE_TEXT_MAX_CHARS = 6_000


def _ma1_narrative_source_text(value: object) -> dict[str, object]:
    """Expose a complete bounded source field, or only its immutable digest.

    Narrative generation never needs transport traces, complete API responses,
    or full-text documents. Silently slicing a source sentence could change its
    meaning, so an oversized value is omitted rather than truncated.
    """

    text = str(value or "").strip()
    if not text:
        return {}
    result: dict[str, object] = {
        "sha256": sha256(text.encode("utf-8")).hexdigest(),
        "character_count": len(text),
    }
    if len(text) <= _MA1_NARRATIVE_SOURCE_TEXT_MAX_CHARS:
        result["text"] = text
    else:
        result["omitted_reason"] = "OVERSIZED_SOURCE_TEXT_NOT_PARTIALLY_QUOTED"
    return result


def _ma1_narrative_evidence_view(packet: HypothesisPacket) -> dict[str, object]:
    """Project claim-linked material required for the narrative-only overlay.

    The complete packet is persisted and hash-bound before this view is built.
    This projection has no gate authority: it removes repeated retrieval
    telemetry while retaining every frozen hypothesis/claim identifier and all
    linked supporting or counter-evidence identities. Reconciliation applies
    returned prose to the untouched packet and freezes every scientific field.
    """

    packet_json = packet.model_dump_json()
    hypotheses: list[dict[str, object]] = []
    for hypothesis in packet.hypotheses:
        linked_ids = {
            evidence_id
            for claim in hypothesis.claims
            for evidence_id in (*claim.evidence_ids, *claim.counter_evidence_ids)
        }
        evidence_view: list[dict[str, object]] = []
        for evidence in hypothesis.evidence:
            if evidence.evidence_id not in linked_ids:
                continue
            metadata = evidence.metadata
            source_text = _ma1_narrative_source_text(
                metadata.get("excerpt") or metadata.get("detail")
            )
            evidence_view.append(
                {
                    key: value
                    for key, value in {
                        "evidence_id": evidence.evidence_id,
                        "source": evidence.source,
                        "external_id": evidence.external_id,
                        "title": evidence.title,
                        "supports": evidence.supports,
                        "excerpt_hash": evidence.excerpt_hash,
                        "source_text": source_text or None,
                        "publication_types": metadata.get("publication_types"),
                        "claim_linkage_roles": metadata.get("claim_linkage_roles"),
                        "direct_support_eligible": metadata.get(
                            "direct_support_eligible"
                        ),
                        "direct_support_rejection_codes": metadata.get(
                            "direct_support_rejection_codes"
                        ),
                    }.items()
                    if value not in (None, "", [], {})
                }
            )
        hypotheses.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "target_symbol": hypothesis.target.symbol,
                "mechanism": hypothesis.mechanism,
                "falsification_test": hypothesis.falsification_test,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "evidence_ids": list(claim.evidence_ids),
                        "counter_evidence_ids": list(claim.counter_evidence_ids),
                    }
                    for claim in hypothesis.claims
                ],
                "claim_linked_evidence": evidence_view,
            }
        )
    return {
        "schema_version": "ma1_narrative_evidence_view_v1",
        "authority_scope": "narrative_only_no_gate_or_evidence_mutation",
        "source_packet_id": packet.packet_id,
        "source_packet_sha256": sha256(packet_json.encode("utf-8")).hexdigest(),
        "source_packet_retained_outside_prompt": True,
        "hypotheses": hypotheses,
    }


def _ensure_ma1_operational_requirements(
    falsification_test: str,
    requirements: tuple[str, ...],
) -> str:
    """Freeze researcher-authored quantitative rules against narrative drift.

    When an executable operational requirement is present, the model's free
    prose is not authoritative for the falsification rule.  Reusing any part
    of that prose can introduce a weaker trailing quantifier even when the
    exact researcher text also appears earlier.  Emit one deterministic,
    bounded wrapper around the original strings instead.
    """

    if not requirements:
        return falsification_test
    combined = (
        "아래 연구자 지정 원문 조건을 후보별 전향 검증의 정량 판정 규칙으로 "
        "그대로 적용한다. 모델 서술로 조건을 완화·대체·추가하지 않는다."
        "\n연구자 지정 검증 조건(원문):\n"
        + "\n".join(f"- {item}" for item in requirements)
    )
    if len(combined) > 2_000:
        raise ValueError(
            "MA1 falsification narrative cannot preserve researcher operational "
            "requirements within the 2000-character strict DTO boundary"
        )
    return combined


_NARRATIVE_ATTESTATION_PREVIEW_CHARS = 600
_MA1_FROZEN_AUTHORITY_FIELDS = [
    "schema_version",
    "packet_id",
    "brief_id",
    "hypothesis_id",
    "claim_id",
    "track",
    "disease",
    "target",
    "intervention_direction",
    "claim.evidence_ids",
    "claim.counter_evidence_ids",
    "claim.confidence",
    "evidence",
    "score",
    "rank",
    "revision",
    "excluded",
    "iterations",
    "generated_at",
]


def _narrative_text_attestation(value: str) -> dict[str, object]:
    """Return a bounded preview plus a full-text digest for one narrative value."""

    truncated = len(value) > _NARRATIVE_ATTESTATION_PREVIEW_CHARS
    if truncated:
        preview = value[: _NARRATIVE_ATTESTATION_PREVIEW_CHARS - 1] + "…"
    else:
        preview = value
    return {
        "text": preview,
        "sha256": sha256(value.encode("utf-8")).hexdigest(),
        "character_count": len(value),
        "truncated": truncated,
    }


def _ma1_narrative_change(
    *,
    hypothesis_id: str,
    field: str,
    before: str,
    after: str,
    claim_id: str | None = None,
) -> dict[str, object]:
    change: dict[str, object] = {
        "hypothesis_id": hypothesis_id,
        "field": field,
        "before": _narrative_text_attestation(before),
        "after": _narrative_text_attestation(after),
    }
    if claim_id is not None:
        change["claim_id"] = claim_id
    return change


def _canonical_payload_sha256(payload: Any) -> str:
    """Hash a JSON-compatible scientific payload deterministically."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _hypothesis_scientific_sha256(hypothesis: Hypothesis) -> str:
    """Hash hypothesis content without its mutable revision counter."""

    return _canonical_payload_sha256(
        hypothesis.model_dump(mode="json", exclude={"revision"})
    )


def _hypothesis_packet_scientific_sha256(packet: HypothesisPacket) -> str:
    """Hash packet science/provenance without artifact-version identity fields."""

    # Packet metadata contains artifact identities such as the independently
    # persisted DiseaseIntakeAssessment ID. Those IDs change between attempts
    # even when the hypotheses do not, so the scientific digest is deliberately
    # limited to the hypothesis/exclusion content and bounded iteration result.
    payload = {
        "brief_id": packet.brief_id,
        "hypotheses": [
            hypothesis.model_dump(mode="json", exclude={"revision"})
            for hypothesis in packet.hypotheses
        ],
        "excluded": [item.model_dump(mode="json") for item in packet.excluded],
        "iterations": packet.iterations,
    }
    if packet.provisional_hypotheses:
        payload["provisional_hypotheses"] = [
            h.model_dump(mode="json", exclude={"revision"})
            for h in packet.provisional_hypotheses
        ]
    return _canonical_payload_sha256(payload)


def _hypothesis_packet_revision_payload_sha256(
    packet: HypothesisPacket,
) -> str:
    """Hash every persisted field except the revision identity/lineage itself."""

    payload = packet.model_dump(
        mode="json",
        exclude={"packet_id"},
    )
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("revision_lineage", None)
    return _canonical_payload_sha256(payload)


def _hypothesis_packet_screening_trace_sha256(
    packet: HypothesisPacket,
) -> str | None:
    """Hash the bounded MA1 screening audit trace outside the science digest."""

    trace = packet.metadata.get("rejected_screening_trace")
    if isinstance(trace, dict):
        return _canonical_payload_sha256(trace)
    recorded = packet.metadata.get("screening_trace_sha256")
    if isinstance(recorded, str) and re.fullmatch(r"[0-9a-f]{64}", recorded):
        return recorded
    return None


def _ma1_completion_message(
    packet: HypothesisPacket,
    *,
    intake_abstain_recommended: bool,
) -> str:
    if packet.provisional_hypotheses:
        return (f"MA1 submitted {len(packet.provisional_hypotheses)} provisional "
                "hypotheses for source-bound MA3 review; none is approved yet.")
    if packet.hypotheses:
        return "MA1 produced a track-balanced hypothesis portfolio."
    if intake_abstain_recommended:
        # Preserve the genuine intake-failure message used by existing runs.
        return "MA1 withheld hypotheses because disease intake was not verified."
    return (
        "MA1 screened candidates but retained zero hypotheses after evidence "
        "and constraint review."
    )


def _version_reworked_hypothesis_packet(
    packet: HypothesisPacket,
    previous: HypothesisPacket,
    order: ReworkOrder,
) -> HypothesisPacket:
    """Create an immutable MA1 revision linked to its persisted parent packet.

    MA1's deterministic base identifier intentionally ignores narrative and
    enrichment payloads. Reusing that identifier after a G0 rework would
    therefore attempt to overwrite the first immutable artifact. A rework is
    instead a new artifact version, even when the requested correction produces
    no scientific delta; ``scientific_changed`` makes that outcome explicit.
    """

    previous_by_id = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in previous.hypotheses
    }
    revised_hypotheses: list[Hypothesis] = []
    for hypothesis in packet.hypotheses:
        prior = previous_by_id.get(hypothesis.hypothesis_id)
        if prior is None:
            revised_hypotheses.append(hypothesis.model_copy(deep=True))
            continue
        changed = (
            _hypothesis_scientific_sha256(hypothesis)
            != _hypothesis_scientific_sha256(prior)
        )
        revision = max(
            hypothesis.revision,
            prior.revision + (1 if changed else 0),
        )
        revised_hypotheses.append(
            hypothesis.model_copy(update={"revision": revision}, deep=True)
        )

    revised_packet = packet.model_copy(
        update={"hypotheses": revised_hypotheses},
        deep=True,
    )
    parent_sha256 = _hypothesis_packet_scientific_sha256(previous)
    scientific_sha256 = _hypothesis_packet_scientific_sha256(revised_packet)
    parent_revision_payload_sha256 = (
        _hypothesis_packet_revision_payload_sha256(previous)
    )
    revision_payload_sha256 = _hypothesis_packet_revision_payload_sha256(
        revised_packet
    )
    parent_screening_trace_sha256 = (
        _hypothesis_packet_screening_trace_sha256(previous)
    )
    screening_trace_sha256 = _hypothesis_packet_screening_trace_sha256(
        revised_packet
    )
    previous_lineage = previous.metadata.get("revision_lineage")
    previous_revision = (
        previous_lineage.get("revision_number", 0)
        if isinstance(previous_lineage, dict)
        else 0
    )
    try:
        revision_number = max(0, int(previous_revision)) + 1
    except (TypeError, ValueError):
        revision_number = 1
    packet_id = (
        "hp_"
        + sha256(
            (
                f"{previous.packet_id}|{order.order_id}|{revision_number}|"
                f"{revision_payload_sha256}"
            ).encode("utf-8")
        ).hexdigest()[:12]
    )
    lineage = {
        "revision_number": revision_number,
        "parent_packet_id": previous.packet_id,
        "rework_order_id": order.order_id,
        "gate": order.gate.value,
        "target_stage": order.target_stage,
        "failed_node": order.failed_node,
        "scientific_changed": scientific_sha256 != parent_sha256,
        "parent_scientific_payload_sha256": parent_sha256,
        "scientific_payload_sha256": scientific_sha256,
        "parent_screening_trace_sha256": parent_screening_trace_sha256,
        "screening_trace_sha256": screening_trace_sha256,
        "audit_trace_changed": (
            screening_trace_sha256 != parent_screening_trace_sha256
        ),
        "parent_revision_payload_sha256": parent_revision_payload_sha256,
        "revision_payload_sha256": revision_payload_sha256,
    }
    return revised_packet.model_copy(
        update={
            "packet_id": packet_id,
            "metadata": {
                **revised_packet.metadata,
                "revision_lineage": lineage,
            },
        },
        deep=True,
    )


def _resume_briefs_are_compatible(
    checkpoint_brief: ResearchBrief,
    snapshot_brief: ResearchBrief,
) -> bool:
    """Accept only audited, non-decreasing operational budget carry-over.

    A technical resume may raise the single-call input or dollar ceiling while
    reusing a scientific-stage checkpoint.  If a later technical interruption
    occurs, the checkpoint still contains the earlier ceilings.  Every
    scientific/user field and every other budget control must remain identical.
    """

    if checkpoint_brief.brief_id != snapshot_brief.brief_id:
        return False
    if checkpoint_brief.model_dump_json() == snapshot_brief.model_dump_json():
        return True

    checkpoint_payload = checkpoint_brief.model_dump(mode="json")
    snapshot_payload = snapshot_brief.model_dump(mode="json")
    checkpoint_budget = checkpoint_payload.pop("budget")
    snapshot_budget = snapshot_payload.pop("budget")
    if checkpoint_payload != snapshot_payload:
        return False
    for field in ("max_run_tokens", "hard_max_run_tokens", "max_llm_calls"):
        if checkpoint_budget[field] != snapshot_budget[field]:
            return False
    return bool(
        snapshot_budget["max_single_input_tokens"]
        >= checkpoint_budget["max_single_input_tokens"]
        and snapshot_budget["max_run_cost_usd"]
        >= checkpoint_budget["max_run_cost_usd"]
    )


class DrugPilotOrchestrator:
    """Deterministic control layer around independent worker components."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: SQLiteRunStore | None = None,
        broker: EventBroker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or SQLiteRunStore(self.settings.database_path)
        self.broker = broker or EventBroker()
        self.providers = ProviderRegistry(self.settings)
        self.dependency = WorkflowDependencyGraph()
        from drugpilot.tools.ncbi_rate_limit import get_process_ncbi_rate_limiter

        self._ncbi_rate_limiter = get_process_ncbi_rate_limiter()
        evidence_retriever = None
        target_discovery = None
        identity_resolver = None
        self.context_router = None
        pharos_retriever = None
        evidence_budget = self.settings.evidence_budget
        if self.settings.public_evidence_enabled:
            # Public evidence is intentionally independent of LLM credentials.
            # The resilient wrapper preserves provenance and falls back to
            # explicit fixtures when a keyless public service is unavailable.
            from drugpilot.tools.evidence import ResilientEvidenceRetriever
            from drugpilot.tools.public_biology import PublicBiologyRetriever

            evidence_retriever = ResilientEvidenceRetriever(
                public=PublicBiologyRetriever(
                    # The selected research-depth profile expands evidence
                    # breadth without changing allow-lists, provenance, or the
                    # fail-closed/no-fixture boundary.
                    timeout_seconds=evidence_budget.core_timeout_seconds,
                    max_records_per_source=(
                        evidence_budget.core_records_per_source
                    ),
                    max_source_calls=(
                        evidence_budget.core_source_calls_per_target
                    ),
                    max_elapsed_seconds=(
                        evidence_budget.core_elapsed_seconds_per_target
                    ),
                    max_attempts=evidence_budget.core_max_attempts,
                    retry_backoff_seconds=(
                        evidence_budget.retry_backoff_seconds
                    ),
                    # Pharos remains an available MA1 adapter, but its public
                    # endpoint has variable latency.  The interactive graph
                    # enables only the bounded core set by default; a caller
                    # can construct PublicBiologyRetriever with Pharos in its
                    # explicit enabled_sources list for an on-demand query.
                    enabled_sources=(
                        "PubMed",
                        "Europe PMC",
                        "Open Targets",
                        "UniProt",
                        "DGIdb",
                    ),
                    ncbi_api_key=self.settings.ncbi_api_key,
                    ncbi_tool=self.settings.ncbi_tool,
                    ncbi_email=self.settings.ncbi_email,
                    ncbi_rate_limiter=self._ncbi_rate_limiter,
                    # Retrieve real limitation/counter-signal literature from
                    # Europe PMC.  Explicit textual signals are required before
                    # a record can be linked as counter-evidence.
                    counter_evidence_enabled=True,
                    epmc_oa_fulltext_one_hop_v1=(
                        self.settings.epmc_oa_fulltext_one_hop_v1
                    ),
                    ma3_source_review_enabled=self.settings.ma1_provisional_admission_enabled,
                ),
                # Research runs never manufacture support or counter records.
                # Source failure remains visible and fails closed instead.
                fixture_fallback_enabled=False,
            )
            pharos_retriever = PublicBiologyRetriever(
                timeout_seconds=evidence_budget.core_timeout_seconds,
                max_records_per_source=evidence_budget.core_records_per_source,
                max_source_calls=1,
                max_elapsed_seconds=(
                    evidence_budget.core_elapsed_seconds_per_target
                ),
                max_attempts=evidence_budget.core_max_attempts,
                retry_backoff_seconds=evidence_budget.retry_backoff_seconds,
                fallback=None,
                enabled_sources=("Pharos",),
            )
            if self.settings.dynamic_target_discovery_enabled:
                from drugpilot.tools.disease_targets import DiseaseTargetDiscovery

                target_discovery = DiseaseTargetDiscovery(
                    timeout_seconds=evidence_budget.core_timeout_seconds,
                    max_targets=self.settings.dynamic_target_max_candidates,
                    pubmed_max_records=evidence_budget.target_pubmed_records,
                    max_pubmed_supplements=(
                        evidence_budget.target_pubmed_supplements
                    ),
                    target_lookup_limit=evidence_budget.target_lookup_limit,
                    max_attempts=evidence_budget.core_max_attempts,
                    retry_backoff_seconds=(
                        evidence_budget.retry_backoff_seconds
                    ),
                    ncbi_api_key=self.settings.ncbi_api_key,
                    ncbi_tool=self.settings.ncbi_tool,
                    ncbi_email=self.settings.ncbi_email,
                    ncbi_rate_limiter=self._ncbi_rate_limiter,
                )
            if self.settings.target_identity_resolution_enabled:
                from drugpilot.tools.target_identity import TargetIdentityResolver

                identity_resolver = TargetIdentityResolver(
                    timeout_seconds=evidence_budget.core_timeout_seconds,
                )
            if self.settings.dynamic_context_enabled:
                from drugpilot.tools.context_router import DiseaseTargetContextRouter
                from drugpilot.tools.public_context import MA1PublicContextRetriever

                self.context_router = DiseaseTargetContextRouter(
                    MA1PublicContextRetriever(
                        timeout_seconds=evidence_budget.context_timeout_seconds,
                        string_caller_identity=self.settings.ncbi_tool,
                        cellxgene_sidecar_url=self.settings.cellxgene_service_url,
                        biogrid_access_key=self.settings.biogrid_access_key,
                    )
                )
        from drugpilot.tools.pharos_on_demand import OnDemandPharosRouter

        self.pharos_router = OnDemandPharosRouter(
            pharos_retriever,
            max_targets_per_track=(
                self.settings.pharos_on_demand_targets_per_track
            ),
        )
        self.ma1 = MA1Agent(
            self.settings,
            evidence_retriever=evidence_retriever,
            target_discovery=target_discovery,
            identity_resolver=identity_resolver,
        )
        self.planner = PlannerAgent(self.settings)
        candidate_enricher = None
        enrichment_per_route = 0
        # The restricted-source router is always present.  Its adapters return
        # typed ``not_configured`` outcomes when no licensed/local source is
        # available, which lets MA2 distinguish "not checked" from a clean
        # lookup while keeping the default offline demonstration runnable.
        from drugpilot.tools.restricted_candidate_enrichment import (
            RoutedRestrictedCandidateEnricher,
        )
        from drugpilot.tools.restricted_sources import (
            DrugBankLicensedAdapter,
            DrugCentralRestrictedAdapter,
            GooglePatentsBigQueryAdapter,
        )

        restricted_candidate_enricher = RoutedRestrictedCandidateEnricher(
            drugcentral=DrugCentralRestrictedAdapter(
                dataset_path=self.settings.drugcentral_dataset_path,
                dataset_version=self.settings.drugcentral_dataset_version,
            ),
            drugbank=DrugBankLicensedAdapter(
                api_key=self.settings.drugbank_api_key,
                timeout_seconds=min(
                    self.settings.public_evidence_timeout_seconds,
                    5.0,
                ),
            ),
            google_patents=GooglePatentsBigQueryAdapter(
                project=self.settings.google_cloud_project,
            ),
        )
        if (
            self.settings.public_evidence_enabled
            and self.settings.public_candidate_enrichment_enabled
            and self.settings.public_candidate_enrichment_per_route > 0
        ):
            from drugpilot.tools.public_chemistry import PublicChemistryEnricher

            candidate_enricher = PublicChemistryEnricher(
                timeout_seconds=min(
                    self.settings.public_evidence_timeout_seconds,
                    5.0,
                ),
                drugcentral_dataset_path=self.settings.drugcentral_dataset_path,
                drugcentral_dataset_version=self.settings.drugcentral_dataset_version,
            )
            enrichment_per_route = self.settings.public_candidate_enrichment_per_route
        self.ma2 = MA2Agent(
            gateway=build_ma2_gateway(self.settings),
            ma2_portfolio_enabled=self.settings.ma2_portfolio_enabled,
            max_refinement_iterations=(
                0 if self.settings.ma2_backend_mode == "live" else 1
            ),
            evidence_enricher=candidate_enricher,
            restricted_evidence_enricher=restricted_candidate_enricher,
            enrichment_per_route=enrichment_per_route,
        )
        # Preserve the public component attributes for health checks and test
        # injection, but normal runs build depth-bound local wrappers.  The
        # sentinels let an explicitly injected test double continue to win
        # without making production profile selection mutable or global.
        self._configured_ma1 = self.ma1
        self._configured_ma1_target_discovery = self.ma1.target_discovery
        self._configured_ma1_identity_resolver = self.ma1.identity_resolver
        self._configured_ma1_retriever = self.ma1.proposer.retriever
        self._configured_context_router = self.context_router
        self._configured_pharos_router = self.pharos_router
        self._configured_ma2 = self.ma2
        reviewer_provider = self.providers.for_reviewer()
        reviewer_backend = self.providers.reviewer_backend
        reviewer = (
            self.settings.reviewer_model
            if reviewer_backend == "anthropic"
            else self.settings.sol_model
            if reviewer_backend == "openai-read-only"
            else "deterministic-rules-v1"
        )
        self.ma3 = ReviewerProjectionMA3AssuranceAgent(
            reviewer_model=reviewer,
            reviewer_provider=reviewer_provider,
            settings=self.settings,
        )
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._workflow_lease_tokens: dict[str, str] = {}
        self._task_registry_lock = asyncio.Lock()
        self.graph = self._build_graph()

    def close(self) -> None:
        """Release process-backed scientific adapters owned by the workflow."""

        self.ma2.close()

    def active_run_count(self) -> int:
        """Return the bounded in-process admission count for this server."""

        return sum(not task.done() for task in self._tasks.values())

    async def _cancellation_checkpoint(self, run_id: str) -> None:
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot and snapshot.stage in {RunStage.CANCEL_REQUESTED, RunStage.CANCELLED}:
            raise RunCancellationRequested("RUN_CANCELLED: no further work will be dispatched")

    async def cancellation_status(self, run_id: str) -> dict[str, Any]:
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        task = self._tasks.get(run_id)
        owned = bool(task is not None and not task.done())
        active = snapshot.stage in {RunStage.CREATED, RunStage.BRIEF_VALIDATED,
            RunStage.MA1, RunStage.G0, RunStage.PLANNER, RunStage.MA2, RunStage.G1}
        active = active or any(r.status == "running" for r in snapshot.g1_reaudits)
        pending = snapshot.stage is RunStage.CANCEL_REQUESTED
        cancelled = snapshot.stage is RunStage.CANCELLED
        return {"run_id": run_id, "eligible": owned and active and not pending and not cancelled,
            "active_in_process": owned, "current_stage": snapshot.stage.value,
            "reason_code": "CANCELLATION_REQUESTED" if pending else "RUN_CANCELLED" if cancelled
                else "RUN_NOT_OWNED_BY_THIS_SERVER" if not owned else "RUN_NOT_ACTIVE" if not active else "CANCELLATION_AVAILABLE",
            "in_flight_call_may_be_billed": True,
            "cancellation_requested_at": snapshot.cancellation_requested_at,
            "cancelled_at": snapshot.cancelled_at}

    async def cancel(self, run_id: str) -> RunSnapshot:
        async with self._task_registry_lock:
            status = await self.cancellation_status(run_id)
            if status["current_stage"] in {RunStage.CANCEL_REQUESTED.value, RunStage.CANCELLED.value}:
                return await self.store.get_snapshot(run_id)
            if not status["eligible"]:
                raise ValueError(status["reason_code"])
            snapshot, event = await self.store.set_cancellation(run_id,
                lease_token=self._workflow_lease_tokens.get(run_id))
            if event is not None:
                await self.broker.publish(event)
            return snapshot

    async def _finish_cancellation(self, run_id: str) -> None:
        snapshot, event = await self.store.set_cancellation(run_id, finished=True,
            lease_token=self._workflow_lease_tokens.get(run_id))
        if event is not None:
            await self.broker.publish(event)
        await self.broker.close(run_id)

    async def recover_pending_cancellations(self) -> None:
        """Finish durable user stops after restart, never replay an unknown call."""
        for run_id in await self.store.pending_cancellation_run_ids():
            task = self._tasks.get(run_id)
            if task and not task.done():
                continue
            lease = await self.store.live_execution_lease_status(run_id, now=utc_now())
            if lease and lease["state"] == "active":
                continue
            if not await self.store.verify_ledger(run_id):
                continue
            try:
                snapshot, event = await self.store.set_cancellation(run_id, finished=True, interrupted=True)
            except ValueError:
                continue  # A competing owner renewed its lease before the transaction.
            if event:
                await self.broker.publish(event)
            await self.broker.close(run_id)

    async def settle_pending_cancellations_after_restart(self) -> None:
        """Wait out surviving leases without dispatching any scientific work."""
        while await self.store.pending_cancellation_run_ids():
            await self.recover_pending_cancellations()
            if await self.store.pending_cancellation_run_ids():
                await asyncio.sleep(30)

    def _ma1_for_brief(self, brief: ResearchBrief) -> MA1Agent:
        """Build a run-local MA1 whose limits are bound to the persisted brief."""

        if (
            self.ma1 is not self._configured_ma1
            or self.ma1.target_discovery
            is not self._configured_ma1_target_discovery
            or self.ma1.identity_resolver
            is not self._configured_ma1_identity_resolver
            or self.ma1.proposer.retriever is not self._configured_ma1_retriever
        ):
            return self.ma1

        budget = self.settings.evidence_budget_for(brief.research_depth)
        run_settings = self.settings.model_copy(
            update={"dynamic_target_max_candidates": budget.target_max_candidates},
            deep=True,
        )
        evidence_retriever = None
        target_discovery = None
        identity_resolver = None
        if self.settings.public_evidence_enabled:
            from drugpilot.tools.disease_targets import DiseaseTargetDiscovery
            from drugpilot.tools.evidence import ResilientEvidenceRetriever
            from drugpilot.tools.public_biology import PublicBiologyRetriever
            from drugpilot.tools.target_identity import TargetIdentityResolver

            evidence_retriever = ResilientEvidenceRetriever(
                public=PublicBiologyRetriever(
                    timeout_seconds=budget.core_timeout_seconds,
                    max_records_per_source=budget.core_records_per_source,
                    max_source_calls=budget.core_source_calls_per_target,
                    max_elapsed_seconds=budget.core_elapsed_seconds_per_target,
                    max_attempts=budget.core_max_attempts,
                    retry_backoff_seconds=budget.retry_backoff_seconds,
                    enabled_sources=(
                        "PubMed",
                        "Europe PMC",
                        "Open Targets",
                        "UniProt",
                        "DGIdb",
                    ),
                    ncbi_api_key=self.settings.ncbi_api_key,
                    ncbi_tool=self.settings.ncbi_tool,
                    ncbi_email=self.settings.ncbi_email,
                    ncbi_rate_limiter=self._ncbi_rate_limiter,
                    counter_evidence_enabled=True,
                    epmc_oa_fulltext_one_hop_v1=(
                        self.settings.epmc_oa_fulltext_one_hop_v1
                    ),
                    ma3_source_review_enabled=self.settings.ma1_provisional_admission_enabled,
                ),
                fixture_fallback_enabled=False,
            )
            if self.settings.dynamic_target_discovery_enabled:
                target_discovery = DiseaseTargetDiscovery(
                    timeout_seconds=budget.core_timeout_seconds,
                    max_targets=budget.target_max_candidates,
                    pubmed_max_records=budget.target_pubmed_records,
                    max_pubmed_supplements=budget.target_pubmed_supplements,
                    target_lookup_limit=budget.target_lookup_limit,
                    max_attempts=budget.core_max_attempts,
                    retry_backoff_seconds=budget.retry_backoff_seconds,
                    ncbi_api_key=self.settings.ncbi_api_key,
                    ncbi_tool=self.settings.ncbi_tool,
                    ncbi_email=self.settings.ncbi_email,
                    ncbi_rate_limiter=self._ncbi_rate_limiter,
                )
            if self.settings.target_identity_resolution_enabled:
                identity_resolver = TargetIdentityResolver(
                    timeout_seconds=budget.core_timeout_seconds,
                )
        return MA1Agent(
            run_settings,
            evidence_retriever=evidence_retriever,
            target_discovery=target_discovery,
            identity_resolver=identity_resolver,
        )

    def _context_router_for_brief(self, brief: ResearchBrief):
        """Return a run-local context router so caches cannot cross profiles."""

        if self.context_router is not self._configured_context_router:
            return self.context_router
        if not (
            self.settings.public_evidence_enabled
            and self.settings.dynamic_context_enabled
        ):
            return None
        from drugpilot.tools.context_router import DiseaseTargetContextRouter
        from drugpilot.tools.public_context import MA1PublicContextRetriever

        budget = self.settings.evidence_budget_for(brief.research_depth)
        return DiseaseTargetContextRouter(
            MA1PublicContextRetriever(
                timeout_seconds=budget.context_timeout_seconds,
                string_caller_identity=self.settings.ncbi_tool,
                cellxgene_sidecar_url=self.settings.cellxgene_service_url,
                biogrid_access_key=self.settings.biogrid_access_key,
            )
        )

    def _pharos_router_for_brief(self, brief: ResearchBrief):
        """Return a run-local Pharos router with the selected target cap."""

        if self.pharos_router is not self._configured_pharos_router:
            return self.pharos_router
        from drugpilot.tools.pharos_on_demand import OnDemandPharosRouter

        budget = self.settings.evidence_budget_for(brief.research_depth)
        retriever = None
        if self.settings.public_evidence_enabled and budget.pharos_targets_per_track:
            from drugpilot.tools.public_biology import PublicBiologyRetriever

            retriever = PublicBiologyRetriever(
                timeout_seconds=budget.core_timeout_seconds,
                max_records_per_source=budget.core_records_per_source,
                max_source_calls=1,
                max_elapsed_seconds=budget.core_elapsed_seconds_per_target,
                max_attempts=budget.core_max_attempts,
                retry_backoff_seconds=budget.retry_backoff_seconds,
                fallback=None,
                enabled_sources=("Pharos",),
            )
        return OnDemandPharosRouter(
            retriever,
            max_targets_per_track=budget.pharos_targets_per_track,
        )

    def _ma2_for_brief(self, brief: ResearchBrief) -> MA2Agent:
        """Create a fully run-local MA2 tool boundary.

        Live gateways and enrichment adapters retain request caches and last
        provenance internally.  Reusing them across concurrent depth runs
        could cross-label otherwise deterministic outputs, so normal runs get
        independent instances and close them after each MA2 stage.
        """

        if self.ma2 is not self._configured_ma2:
            return self.ma2
        budget = self.settings.evidence_budget_for(brief.research_depth)
        from drugpilot.tools.restricted_candidate_enrichment import (
            RoutedRestrictedCandidateEnricher,
        )
        from drugpilot.tools.restricted_sources import (
            DrugBankLicensedAdapter,
            DrugCentralRestrictedAdapter,
            GooglePatentsBigQueryAdapter,
        )

        restricted_candidate_enricher = RoutedRestrictedCandidateEnricher(
            drugcentral=DrugCentralRestrictedAdapter(
                dataset_path=self.settings.drugcentral_dataset_path,
                dataset_version=self.settings.drugcentral_dataset_version,
            ),
            drugbank=DrugBankLicensedAdapter(
                api_key=self.settings.drugbank_api_key,
                timeout_seconds=min(
                    self.settings.public_evidence_timeout_seconds,
                    5.0,
                ),
            ),
            google_patents=GooglePatentsBigQueryAdapter(
                project=self.settings.google_cloud_project,
            ),
        )
        candidate_enricher = None
        if budget.candidate_enrichment_per_route > 0:
            from drugpilot.tools.public_chemistry import PublicChemistryEnricher

            candidate_enricher = PublicChemistryEnricher(
                timeout_seconds=min(
                    self.settings.public_evidence_timeout_seconds,
                    5.0,
                ),
                drugcentral_dataset_path=self.settings.drugcentral_dataset_path,
                drugcentral_dataset_version=(
                    self.settings.drugcentral_dataset_version
                ),
            )
        repurposing_evidence_policy = _repurposing_evidence_policy_for_priorities(
            list(brief.human_priorities)
        )
        return MA2Agent(
            gateway=build_ma2_gateway(
                self.settings,
                repurposing_evidence_policy=repurposing_evidence_policy,
            ),
            max_refinement_iterations=(
                0 if self.settings.ma2_backend_mode == "live" else 1
            ),
            evidence_enricher=candidate_enricher,
            restricted_evidence_enricher=restricted_candidate_enricher,
            enrichment_per_route=budget.candidate_enrichment_per_route,
            ma2_portfolio_enabled=self.settings.ma2_portfolio_enabled,
        )

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("resume_entry", self._resume_entry)
        graph.add_node(
            "validate_brief",
            self._checkpointed_node("validate_brief", self._validate_brief),
        )
        graph.add_node("ma1", self._checkpointed_node("ma1", self._run_ma1))
        graph.add_node("intake_guard", self._guard_disease_intake)
        graph.add_node("g0", self._checkpointed_node("g0", self._run_g0))
        graph.add_node(
            "planner", self._checkpointed_node("planner", self._run_planner)
        )
        graph.add_node("route_guard", self._guard_ma2_route)
        graph.add_node("ma2", self._checkpointed_node("ma2", self._run_ma2))
        graph.add_node("g1", self._checkpointed_node("g1", self._run_g1))
        graph.add_node("human_gate", self._await_human)
        graph.add_node("abstain", self._abstain)
        graph.add_node("block", self._block)

        graph.add_edge(START, "resume_entry")
        graph.add_conditional_edges(
            "resume_entry",
            self._route_resume_entry,
            {
                "validate_brief": "validate_brief",
                "ma1": "ma1",
                "intake_guard": "intake_guard",
                "planner": "planner",
                "route_guard": "route_guard",
                "ma2": "ma2",
                "g1": "g1",
                "human_gate": "human_gate",
                "abstain": "abstain",
                "block": "block",
            },
        )
        graph.add_edge("validate_brief", "ma1")
        graph.add_edge("ma1", "intake_guard")
        graph.add_conditional_edges(
            "intake_guard",
            self._route_disease_intake,
            {
                "g0": "g0",
                "abstain": "abstain",
            },
        )
        graph.add_conditional_edges(
            "g0",
            self._route_g0,
            {
                "planner": "planner",
                "ma1": "ma1",
                "human_gate": "human_gate",
                "abstain": "abstain",
                "block": "block",
            },
        )
        graph.add_edge("planner", "route_guard")
        graph.add_conditional_edges(
            "route_guard",
            self._route_ma2_guard,
            {
                "ma2": "ma2",
                "abstain": "abstain",
            },
        )
        graph.add_edge("ma2", "g1")
        graph.add_conditional_edges(
            "g1",
            self._route_g1,
            {
                "human_gate": "human_gate",
                "ma2": "ma2",
                "planner": "planner",
                "abstain": "abstain",
                "block": "block",
            },
        )
        graph.add_edge("human_gate", END)
        graph.add_edge("abstain", END)
        graph.add_edge("block", END)
        return graph.compile()

    @staticmethod
    async def _resume_entry(state: WorkflowState) -> WorkflowState:
        """Graph entry marker; routing uses only persisted structured state."""

        return {}

    @staticmethod
    def _route_resume_entry(state: WorkflowState) -> str:
        destination = state.get("_resume_from", "validate_brief")
        supported = {
            "validate_brief",
            "ma1",
            "intake_guard",
            "planner",
            "route_guard",
            "ma2",
            "g1",
            "human_gate",
            "abstain",
            "block",
        }
        if destination not in supported:
            raise ValueError("checkpoint selected an unsupported workflow node")
        return destination

    def _checkpointed_node(self, node: str, handler):
        async def run(state: WorkflowState) -> WorkflowState:
            await self._cancellation_checkpoint(state["run_id"])
            delta = await handler(state)
            await self._cancellation_checkpoint(state["run_id"])
            merged: WorkflowState = {**state, **delta}
            await self._commit_workflow_checkpoint(node, merged)
            return delta

        return run

    async def _commit_workflow_checkpoint(
        self,
        completed_node: str,
        state: WorkflowState,
    ) -> None:
        """Atomically checkpoint a successfully completed, resumable node."""

        run_id = state["run_id"]
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        manifest = await self.store.artifact_manifest(run_id)
        artifact_sha256 = {
            str(item["artifact_id"]): str(item["sha256"]) for item in manifest
        }
        expected_artifact_ids = {state["brief"].brief_id}
        for artifact in (
            state.get("disease_intake_assessment"),
            state.get("hypothesis_packet"),
            state.get("g0_verdict"),
            state.get("plan_spec"),
            state.get("candidate_packet"),
            state.get("g1_verdict"),
        ):
            if artifact is None:
                continue
            artifact_id = next(
                (
                    getattr(artifact, field)
                    for field in (
                        "assessment_id",
                        "packet_id",
                        "verdict_id",
                        "plan_id",
                    )
                    if getattr(artifact, field, None)
                ),
                None,
            )
            if artifact_id:
                expected_artifact_ids.add(artifact_id)
        if not expected_artifact_ids <= set(artifact_sha256):
            raise ValueError(
                "checkpoint cannot commit before required artifacts are durable"
            )

        checkpoint = WorkflowCheckpoint(
            run_id=run_id,
            completed_node=completed_node,
            stage=snapshot.stage,
            brief=state["brief"],
            hypothesis_packet=state.get("hypothesis_packet"),
            g0_verdict=state.get("g0_verdict"),
            plan_spec=state.get("plan_spec"),
            candidate_packet=state.get("candidate_packet"),
            g1_verdict=state.get("g1_verdict"),
            rework_order=state.get("rework_order"),
            rework_loops=state.get("rework_loops", 0),
            fault_injected=state.get("fault_injected", False),
            disease_intake_assessment=state.get("disease_intake_assessment"),
            intake_abstain_recommended=state.get(
                "intake_abstain_recommended", False
            ),
            route_abstain_recommended=state.get(
                "route_abstain_recommended", False
            ),
            abstain_reason=state.get("abstain_reason"),
            artifact_sha256=artifact_sha256,
        )
        digest = self.store.checkpoint_digest(checkpoint)
        event = ExecutionEvent(
            event_id=f"evt_{checkpoint.checkpoint_id}",
            run_id=run_id,
            stage=snapshot.stage,
            event_type="stage_checkpoint_committed",
            message=f"Durable checkpoint committed after {completed_node}.",
            node="orchestrator.checkpoint",
            artifact_ids=sorted(artifact_sha256),
            metadata={
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_sha256": digest,
                "checkpoint_schema_version": checkpoint.schema_version,
                "workflow_version": checkpoint.workflow_version,
                "completed_node": completed_node,
                "artifact_sha256": artifact_sha256,
                "resumable": True,
            },
            created_at=checkpoint.created_at,
        )
        snapshot.events.append(event)
        await self.store.commit_stage_checkpoint(checkpoint, event, snapshot)
        await self.broker.publish(event)

    async def create_run(
        self,
        brief: ResearchBrief,
        *,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
        admission_controlled: bool = False,
    ) -> RunSnapshot:
        if (idempotency_key is None) != (request_sha256 is None):
            raise ValueError(
                "idempotency key and request digest must be supplied together"
            )
        run_id = new_id("run")
        # Direct callers and legacy snapshots may omit the newly introduced
        # field. Persist an explicit effective value for every newly created
        # run without mutating the caller's model or global Settings.
        if brief.research_depth is None:
            brief = brief.model_copy(
                update={"research_depth": self.settings.evidence_research_depth},
                deep=True,
            )
        goal_focus = derive_goal_focus(brief)
        snapshot = RunSnapshot(
            run_id=run_id,
            mode=self.settings.effective_mode,
            stage=RunStage.CREATED,
            status_message="Research brief accepted; workflow queued.",
            brief=brief,
        )
        lease_token = (
            new_id("lease")
            if admission_controlled
            and self.settings.effective_mode is RunMode.LIVE
            else None
        )
        if admission_controlled:
            artifact_digest = sha256(
                brief.model_dump_json().encode("utf-8")
            ).hexdigest()
            focus_digest = sha256(
                goal_focus.model_dump_json().encode("utf-8")
            ).hexdigest()
            event = ExecutionEvent(
                run_id=run_id,
                stage=RunStage.CREATED,
                event_type="run_created",
                message="Research brief accepted and persisted.",
                node="research_brief",
                artifact_ids=[brief.brief_id, goal_focus.focus_id],
                metadata={
                    "artifact_sha256": {
                        brief.brief_id: artifact_digest,
                        goal_focus.focus_id: focus_digest,
                    },
                    "goal_focus": goal_focus.model_dump(mode="json"),
                    "research_depth": brief.research_depth,
                    "research_depth_contract": self.settings.evidence_budget_for(
                        brief.research_depth
                    ).as_dict(),
                    "admission_controlled": True,
                    "idempotent_admission": idempotency_key is not None,
                    "request_sha256": request_sha256,
                },
            )
            snapshot.events.append(event)
            await self.store.commit_admitted_run(
                snapshot,
                artifacts=[
                    (
                        brief.brief_id,
                        "ResearchBrief",
                        brief,
                        brief.schema_version,
                    ),
                    (
                        goal_focus.focus_id,
                        "GoalFocus",
                        goal_focus,
                        goal_focus.schema_version,
                    ),
                ],
                event=event,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                admitted_at=snapshot.created_at,
                live_execution=lease_token is not None,
                lease_token=lease_token,
                lease_ttl_seconds=self.settings.beta_live_lease_ttl_seconds,
                max_active_live_runs=self.settings.beta_max_active_live_runs,
                max_new_runs_per_hour=self.settings.beta_max_new_runs_per_hour,
            )
            try:
                await self.broker.publish(event)
            except Exception:
                if lease_token is not None:
                    await self.store.expire_live_execution_lease_for_recovery(
                        run_id,
                        lease_token=lease_token,
                        reason="admission_event_publish_failed",
                        now=utc_now(),
                    )
                raise
        else:
            await self.store.save_snapshot(snapshot)
            artifact_digest = await self.store.save_artifact(
                run_id, brief.brief_id, "ResearchBrief", brief, brief.schema_version
            )
            focus_digest = await self.store.save_artifact(
                run_id,
                goal_focus.focus_id,
                "GoalFocus",
                goal_focus,
                goal_focus.schema_version,
            )
            await self._emit(
                snapshot,
                RunStage.CREATED,
                "run_created",
                "Research brief accepted and persisted.",
                node="research_brief",
                artifact_ids=[brief.brief_id, goal_focus.focus_id],
                metadata={
                    "artifact_sha256": {
                        brief.brief_id: artifact_digest,
                        goal_focus.focus_id: focus_digest,
                    },
                    "goal_focus": goal_focus.model_dump(mode="json"),
                    "research_depth": brief.research_depth,
                    "research_depth_contract": self.settings.evidence_budget_for(
                        brief.research_depth
                    ).as_dict(),
                },
            )
        workflow = (
            self._execute_with_lease(run_id, brief, lease_token=lease_token)
            if lease_token is not None
            else self._execute(run_id, brief)
        )
        task: asyncio.Task[None] | None = None
        try:
            task = asyncio.create_task(workflow, name=f"drugpilot:{run_id}")
            self._track_task(run_id, task)
        except Exception:
            if task is None:
                workflow.close()
            else:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if lease_token is not None:
                await self.store.expire_live_execution_lease_for_recovery(
                    run_id,
                    lease_token=lease_token,
                    reason="create_task_registration_failed",
                    now=utc_now(),
                )
            raise
        return snapshot

    def _track_task(self, run_id: str, task: asyncio.Task[None]) -> None:
        """Register a task without letting an older callback remove its successor."""

        self._tasks[run_id] = task
        task.add_done_callback(
            lambda completed, tracked_run_id=run_id: self._remove_task_if_current(
                tracked_run_id, completed
            )
        )

    def _remove_task_if_current(
        self,
        run_id: str,
        completed: asyncio.Task[None],
    ) -> None:
        if self._tasks.get(run_id) is completed:
            self._tasks.pop(run_id, None)

    async def _execute(
        self,
        run_id: str,
        brief: ResearchBrief,
        resume_state: WorkflowState | None = None,
    ) -> None:
        state: WorkflowState = resume_state or {
            "run_id": run_id,
            "brief": brief,
            "rework_loops": 0,
            "fault_injected": False,
            "rework_order": None,
            "_resume_from": "validate_brief",
        }
        try:
            await self.graph.ainvoke(state, config={"recursion_limit": 30})
            await self._cancellation_checkpoint(run_id)
        except RunCancellationRequested:
            await self._finish_cancellation(run_id)
        except BudgetExceeded as exc:
            await self._terminal(run_id, RunStage.ABSTAINED, str(exc), error=str(exc))
        except Exception as exc:  # noqa: BLE001 - persist safe failure state
            await self._terminal(
                run_id,
                RunStage.FAILED,
                "Workflow stopped after an internal error.",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _lease_heartbeat(
        self,
        run_id: str,
        *,
        lease_token: str,
    ) -> None:
        while True:
            await asyncio.sleep(self.settings.beta_live_lease_heartbeat_seconds)
            renewed = await self.store.renew_live_execution_lease(
                run_id,
                lease_token=lease_token,
                now=utc_now(),
                lease_ttl_seconds=self.settings.beta_live_lease_ttl_seconds,
            )
            if not renewed:
                raise RuntimeError("durable live execution lease ownership was lost")

    async def _execute_with_lease(
        self,
        run_id: str,
        brief: ResearchBrief,
        *,
        lease_token: str,
        resume_state: WorkflowState | None = None,
    ) -> None:
        """Run one paid workflow while heartbeating and finally releasing its lease."""

        self._workflow_lease_tokens[run_id] = lease_token
        workflow = asyncio.create_task(
            self._execute(run_id, brief, resume_state),
            name=f"drugpilot-workflow:{run_id}",
        )
        heartbeat = asyncio.create_task(
            self._lease_heartbeat(run_id, lease_token=lease_token),
            name=f"drugpilot-lease:{run_id}",
        )
        release_reason = "workflow_task_finished"
        try:
            done, _ = await asyncio.wait(
                {workflow, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if workflow in done:
                await workflow
                current = await self.store.get_snapshot(run_id)
                if current and current.stage is RunStage.CANCELLED:
                    release_reason = "user_cancelled"
            else:
                release_reason = "lease_heartbeat_lost"
                workflow.cancel()
                with suppress(asyncio.CancelledError):
                    await workflow
                await heartbeat
        except asyncio.CancelledError:
            release_reason = "workflow_task_cancelled"
            workflow.cancel()
            with suppress(asyncio.CancelledError):
                await workflow
            raise
        finally:
            heartbeat.cancel()
            # If the heartbeat was the completed task, its ownership-loss
            # exception was already propagated above. Do not let awaiting it
            # again mask the workflow outcome or prevent best-effort release.
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat
            await self.store.release_live_execution_lease(
                run_id,
                lease_token=lease_token,
                reason=release_reason,
                now=utc_now(),
            )
            if self._workflow_lease_tokens.get(run_id) == lease_token:
                self._workflow_lease_tokens.pop(run_id, None)

    async def _validate_brief(self, state: WorkflowState) -> WorkflowState:
        brief = state["brief"]
        if brief.constraints.candidates_per_route > 100:
            raise ValueError("candidate route limit exceeds policy")
        await self._transition(
            state["run_id"],
            RunStage.BRIEF_VALIDATED,
            "Research brief schema and policy checks passed.",
            "brief_validated",
            node="research_brief",
        )
        return {}

    async def _run_ma1(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        research_budget = self.settings.evidence_budget_for(
            state["brief"].research_depth
        )
        previous = state.get("hypothesis_packet")
        revision_context = previous
        reworking = state.get("rework_order") is not None
        if reworking:
            await self._record_rework(run_id, state["rework_order"])
        await self._transition(
            run_id,
            RunStage.MA1,
            "MA1 Intake → Proposer → Critic → Router → Selector running.",
            "stage_started",
            node="ma1",
            metadata={
                "interaction": interaction(
                    "Orchestrator",
                    "MA1",
                    "task_assignment",
                    f"{state['brief'].disease} 연구목표에서 근거 연결 가설 포트폴리오 생성 요청",
                    basis=[
                        state["brief"].research_goal,
                        f"연구 깊이 {research_budget.depth}",
                        f"허용 데이터 소스 {len(state['brief'].data_policy.allowed_sources)}종",
                        f"최대 MA1 반복 {state['brief'].constraints.max_ma1_iterations}회",
                    ],
                    backend=self._component_backend("MA1"),
                    model=(
                        self.settings.terra_model
                        if self.settings.effective_mode is not RunMode.MOCK
                        else None
                    ),
                    artifact_ids=[state["brief"].brief_id],
                )
            },
        )
        run_ma1 = self._ma1_for_brief(state["brief"])
        from drugpilot.orchestration.g0_followup import enabled as g0_followup_enabled, build_plan, FollowupRetriever
        followup_retriever = None
        prior_evidence = {}
        if reworking and previous is not None and state.get("g0_verdict") and g0_followup_enabled(self):
            followup_plan = build_plan(state["brief"], previous, state["g0_verdict"],
                rework_loops=state.get("rework_loops", 0),
                target_limit=self.settings.ma1_supplementary_evidence_retry_budget)
            if not followup_plan["eligible"]:
                raise ValueError("G0 follow-up is not eligible")
            from drugpilot.orchestration.evidence_carryover import load_prior_bundles, POLICY_NAME
            from drugpilot.orchestration.semantic_interpretation import interpretation_contract
            if self.settings.ma1_provisional_admission_enabled:
                from drugpilot.orchestration.ma1_source_bundles import load_source_bundles
                from drugpilot.orchestration.g0_followup import ma3_followup_revision_context
                prior_evidence, prior_lineage = await load_source_bundles(self.store, run_id, state["brief"])
                revision_context = ma3_followup_revision_context(previous, followup_plan)
            else:
                prior_evidence, prior_lineage = await load_prior_bundles(self.store, run_id, state["brief"],
                    contract=interpretation_contract(), models=[self.settings.terra_model,self.settings.sol_model])
            await self._trace_event(run_id, "g0_prior_evidence_restored",
                "이전 회차의 검증된 원문을 보존 · 모델 불일치 재추첨 없음 · 과학 게이트 유지.",
                node="ma1.g0_followup", metadata={"policy":POLICY_NAME,"bundles":prior_lineage,
                    "record_counts":{k:len(v) for k,v in prior_evidence.items()},
                    "new_source_calls":0,"new_evidence_created":False})
            followup_retriever = FollowupRetriever(run_ma1.proposer.retriever, followup_plan, prior_evidence)
            run_ma1.proposer.retriever = followup_retriever
            run_ma1.g0_followup_targets = sorted({
                *(
                    t["target"].casefold()
                    for t in followup_plan["tasks"]
                    if t["status"] == "scheduled"
                ),
                *(
                    target.casefold()
                    for target in followup_plan.get(
                        "reclassification_targets", []
                    )
                ),
            })
        if ((self.settings.ma1_semantic_interpretation_enabled or self.settings.ma1_provisional_admission_enabled)
                and self.settings.effective_mode is RunMode.LIVE
                and self.settings.public_evidence_enabled
                and run_ma1 is not self.ma1):
            from drugpilot.orchestration.semantic_interpretation import interpret_target
            loop = asyncio.get_running_loop()
            # MA1's synchronous public retrieval runs in a worker thread. All
            # metering, provider I/O and durable progress stay on the owner loop.
            if not self.settings.ma1_provisional_admission_enabled:
                run_ma1.proposer.evidence_interpreter = lambda brief, seed, iteration, evidence: (
                    asyncio.run_coroutine_threadsafe(
                        interpret_target(self, run_id, brief, seed, iteration, evidence,
                            **({"preserved_evidence":prior_evidence.get(seed.symbol.casefold(), ())}
                               if prior_evidence else {})), loop
                    ).result()
                )
            else:
                from drugpilot.orchestration.ma1_source_bundles import persist_source_bundle
                run_ma1.proposer.evidence_interpreter = lambda brief, seed, iteration, evidence: (
                    asyncio.run_coroutine_threadsafe(
                        persist_source_bundle(self, run_id, brief, seed, iteration, evidence), loop
                    ).result()
                )
            run_ma1.proposer.on_target_started = lambda seed, iteration: (
                asyncio.run_coroutine_threadsafe(self._trace_event(
                    run_id, "ma1_target_retrieval_started",
                    f"{seed.symbol}: 공개 근거 조회 시작 (MA1 반복 {iteration}).",
                    node="ma1.evidence_retrieval", metadata={"target": seed.symbol,
                        "iteration": iteration, "status": "retrieval_started"}), loop
                ).result()
            )
        packet = await asyncio.to_thread(run_ma1.run, state["brief"], revision_context)
        if self.settings.ma1_provisional_admission_enabled:
            packet.metadata["semantic_authority"] = "MA3_source_bound_admission"
            packet.metadata["ma1_semantic_pair_skipped"] = True
        if followup_retriever is not None:
            history = [*previous.metadata.get("g0_followup_history", []), followup_retriever.history_entry()]
            packet = packet.model_copy(update={"metadata": {**packet.metadata,
                "g0_followup_history": history}}, deep=True)
            await self._trace_event(run_id, "g0_followup_sources_completed",
                "G0 원인별 공개 근거 재조회 완료 · 새 근거도 MA1/G0 검증 필수.",
                node="ma1.g0_followup", metadata=history[-1])
        deterministic_intake = _derive_disease_intake_assessment(
            state["brief"],
            packet,
            demo_fixture_allowed=self.settings.demo_fixture_fallback_enabled,
        )
        intake_suggestion = await self._llm_enhance(
            run_id,
            component="MA1-Intake",
            model=self.settings.terra_model,
            response_model=DiseaseIntakeSuggestionOutput,
            fallback=DiseaseIntakeSuggestionOutput(
                suggested_normalized_name=None,
                explanation=None,
                warnings=[],
                field_mismatches=[],
            ),
            prompt=(
                "Explain the deterministic disease-intake result and, if useful, suggest "
                "a disease label for the researcher to confirm. You may report field "
                "mismatches or warnings only. The response schema intentionally cannot "
                "set verification status, ontology IDs, targets, evidence, scores, or a "
                "gate verdict. Never describe a suggested name as verified.\n"
                "RESEARCH_BRIEF:\n"
                + state["brief"].model_dump_json()
                + "\nDETERMINISTIC_EXTERNAL_ASSESSMENT:\n"
                + deterministic_intake.model_dump_json()
                + "\nSOURCE_METADATA:\n"
                + str(
                    {
                        "resolved_disease": packet.metadata.get("resolved_disease"),
                        "source_statuses": packet.metadata.get(
                            "target_source_statuses", []
                        ),
                        "discovery_error": packet.metadata.get(
                            "target_discovery_error"
                        ),
                    }
                )
            ),
            max_output_tokens=2_000,
            failure_is_fatal=False,
            postprocess=_intake_suggestion_from_output,
        )
        intake_assessment = _merge_disease_intake_suggestion(
            deterministic_intake,
            intake_suggestion,
        )
        # No earlier or prior-revision hypotheses may survive an intake
        # abstention.  The external decision remains authoritative even when a
        # future LLM proposes a plausible normalization.
        if intake_assessment.abstain_recommended:
            packet = packet.model_copy(
                update={
                    "hypotheses": [],
                    "provisional_hypotheses": [],
                    "excluded": [],
                    "metadata": {
                        **packet.metadata,
                        "target_sources_scope": "retained_only",
                        "target_sources": {},
                        **({"ma1_provisional_hypothesis_count": 0,
                            "ma1_submitted_hypothesis_ids": [],
                            "intake_withheld_provisional_count": len(packet.provisional_hypotheses)}
                           if packet.metadata.get("admission_policy") == "ma3-source-bound-v1" else {}),
                    },
                },
                deep=True,
            )
        packet = packet.model_copy(
            update={
                "metadata": {
                    **packet.metadata,
                    "intake_validation": intake_assessment.model_dump(mode="json"),
                }
            },
            deep=True,
        )
        goal_focus = derive_goal_focus(state["brief"])
        packet, goal_focus = apply_ma1_review_goal_focus(packet, goal_focus)
        # Canonical routing hints are derived from exact brief cues.  They are
        # used only by context/source routers; the immutable ResearchBrief and
        # MA1 scientific evidence request remain unchanged.
        routing_brief = state["brief"].model_copy(
            update={
                "human_priorities": _research_depth_routing_priorities(
                    state["brief"],
                    goal_focus,
                    self.settings,
                )
            },
            deep=True,
        )
        if not intake_assessment.abstain_recommended:
            packet = await self._enrich_ma1_pharos(run_id, packet, routing_brief)
            packet = await self._enrich_ma1_context(run_id, packet, routing_brief)
        packet = packet.model_copy(
            update={
                "metadata": {
                    **packet.metadata,
                    "research_depth_contract": research_budget.as_dict(),
                }
            },
            deep=True,
        )
        if (
            state["brief"].fault_injection is FaultType.G0_MISSING_EVIDENCE
            and not state.get("fault_injected", False)
            and review_hypotheses(packet)
        ):
            packet = packet.model_copy(deep=True)
            review_hypotheses(packet)[0].evidence = []
            fault_injected = True
        else:
            fault_injected = state.get("fault_injected", False)
        # Timestamp this newly assembled packet, not the immutable input brief.
        packet = packet.model_copy(update={"generated_at": utc_now()}, deep=True)
        if self.settings.effective_mode is RunMode.LIVE:
            from drugpilot.orchestration.semantic_interpretation import _persist_bound
            draft_id = "ma1draft_" + sha256(packet.model_dump_json().encode()).hexdigest()[:32]
            await _persist_bound(self, run_id, draft_id, "MA1ScreeningDraft", packet)
        if intake_assessment.abstain_recommended or not packet.hypotheses:
            immutable_packet = packet.model_copy(deep=True)
            if not intake_assessment.abstain_recommended:
                await self._trace_event(run_id, "ma1_narrative_skipped",
                    ("Provisional hypotheses await MA3 scientific review; no approval narrative generated."
                     if packet.provisional_hypotheses else
                     "No retained hypotheses to narrate; full packet proceeds to G0 unchanged."),
                    node="ma1.narrative", metadata={"reason": (
                        "provisional_review_pending" if packet.provisional_hypotheses else "no_retained_hypotheses"),
                        "model_calls":0,"scientific_audit_skipped":False})
        else:
            immutable_packet = packet.model_copy(deep=True)
            narrative_fallback = _ma1_narrative_fallback(immutable_packet)
            operational_requirements = _ma1_operational_priorities(state["brief"])
            llm_packet = await self._llm_enhance(
                run_id,
                component="MA1",
                model=self.settings.terra_model,
                response_model=HypothesisNarrativeOverlayOutput,
                fallback=narrative_fallback,
                prompt=(
                    MA1_KOREAN_NARRATIVE_INSTRUCTION
                    + _ma1_narrative_prompt_suffix(
                        state["brief"],
                        immutable_packet,
                        operational_requirements,
                    )
                ),
                max_output_tokens=24_000,
                postprocess=lambda output: self._reconcile_ma1_narrative(
                    immutable_packet,
                    output,
                    attestation_model=(
                        self.settings.terra_model
                        if output is not narrative_fallback
                        else None
                    ),
                    attestation_component="MA1",
                    researcher_operational_requirements=(
                        operational_requirements
                        if output is not narrative_fallback
                        else ()
                    ),
                ),
                postprocess_failure_is_fatal=True,
            )
            packet = llm_packet
        if reworking and previous is not None:
            packet = _version_reworked_hypothesis_packet(
                packet,
                previous,
                state["rework_order"],
            )
        evidence_records = [
            evidence
            for hypothesis in review_hypotheses(packet)
            for evidence in hypothesis.evidence
        ]
        provenance_counts: dict[str, int] = {}
        for evidence in evidence_records:
            provenance = str(
                evidence.metadata.get("provenance_class")
                or ("external_live" if evidence.metadata.get("live") else "fixture")
            )
            provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
        handoff = hypothesis_handoff(packet)
        if packet.metadata.get("admission_policy") == "ma3-source-bound-v1":
            handoff = interaction("MA1", "MA3 · G0", "artifact_handoff",
                f"잠정 가설 {len(packet.provisional_hypotheses)}개를 과학적 심사에 제출 · 승인 가설 {len(packet.hypotheses)}개",
                basis=["원문 출처·인용 무결성 확인", "가설 타당성 및 작용 방향 판단은 MA3 담당",
                       "잠정 가설은 MA2 실행 대상이 아님"],
                evidence_ids=[item.evidence_id for item in evidence_records], artifact_ids=[packet.packet_id],
                details={"hypothesis_count": len(packet.hypotheses),
                         "provisional_hypothesis_count": len(packet.provisional_hypotheses),
                         "excluded_count": len(packet.excluded), "iterations": packet.iterations,
                         "scientific_approval_pending": True})
        handoff["backend"] = self._component_backend("MA1")
        if self.settings.effective_mode is not RunMode.MOCK:
            handoff["model"] = self.settings.terra_model
        await self._artifact_transition(
            run_id,
            RunStage.MA1,
            (
                "Disease intake verified."
                if not intake_assessment.abstain_recommended
                else "Disease intake requires clarification; downstream execution is withheld."
            ),
            "disease_intake_assessed",
            "ma1.intake_guard",
            intake_assessment.assessment_id,
            "DiseaseIntakeAssessment",
            intake_assessment,
            metadata={
                "intake_validation": intake_assessment.model_dump(mode="json"),
            },
        )
        await self._artifact_transition(
            run_id,
            RunStage.MA1,
            _ma1_completion_message(
                packet,
                intake_abstain_recommended=(
                    intake_assessment.abstain_recommended
                ),
            ),
            "ma1_completed",
            "ma1.selector",
            packet.packet_id,
            "HypothesisPacket",
            packet,
            metadata={
                "evidence_sources": sorted({item.source for item in evidence_records}),
                "evidence_provenance_counts": provenance_counts,
                "public_evidence_enabled": self.settings.public_evidence_enabled,
                "research_depth": research_budget.depth,
                "research_depth_contract": research_budget.as_dict(),
                "target_discovery": {
                    "resolved_disease": packet.metadata.get("resolved_disease"),
                    "source_statuses": packet.metadata.get("target_source_statuses", []),
                    "fallback_used": packet.metadata.get("target_fallback_used", False),
                    "external_seed_count": packet.metadata.get(
                        "external_target_seed_count", 0
                    ),
                },
                "context_routing": packet.metadata.get("context_routing"),
                "goal_focus": packet.metadata.get("goal_focus"),
                "intake_validation": packet.metadata.get("intake_validation"),
                "revision_lineage": packet.metadata.get("revision_lineage"),
                "target_sources_scope": packet.metadata.get(
                    "target_sources_scope"
                ),
                "screened_target_source_count_total": packet.metadata.get(
                    "screened_target_source_count_total", 0
                ),
                "rejected_screening_entry_count_total": (
                    packet.metadata.get("rejected_screening_trace", {}).get(
                        "entry_count_total", 0
                    )
                    if isinstance(
                        packet.metadata.get("rejected_screening_trace"), dict
                    )
                    else 0
                ),
                "screening_trace_sha256": packet.metadata.get(
                    "screening_trace_sha256"
                ),
                "interaction": handoff,
            },
        )
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot:
            snapshot.ma1_iterations = max(
                snapshot.ma1_iterations, packet.iterations
            )
            snapshot.rework_loops = state.get("rework_loops", 0) + (
                1 if reworking else 0
            )
            await self.store.save_snapshot(snapshot)
        return {
            "hypothesis_packet": packet,
            "disease_intake_assessment": intake_assessment,
            "intake_abstain_recommended": (
                intake_assessment.abstain_recommended
            ),
            "fault_injected": fault_injected,
            "rework_order": None,
            "rework_loops": state.get("rework_loops", 0) + (1 if reworking else 0),
        }

    async def _guard_disease_intake(self, state: WorkflowState) -> WorkflowState:
        assessment = state.get("disease_intake_assessment")
        if assessment is None:
            raw = state["hypothesis_packet"].metadata.get("intake_validation")
            try:
                assessment = DiseaseIntakeAssessment.model_validate(raw)
            except Exception:
                await self._trace_event(
                    state["run_id"],
                    "disease_intake_guard_failed",
                    "Disease intake assessment is missing or invalid; workflow will abstain.",
                    node="ma1.intake_guard",
                    metadata={
                        "reason_codes": ["MISSING_INTAKE_ASSESSMENT"],
                        "abstain_recommended": True,
                    },
                )
                return {
                    "intake_abstain_recommended": True,
                    "abstain_reason": (
                        "Disease intake verification is missing; human clarification is required."
                    ),
                }

        assert assessment is not None
        await self._trace_event(
            state["run_id"],
            (
                "disease_intake_guard_abstain"
                if assessment.abstain_recommended
                else "disease_intake_guard_passed"
            ),
            (
                "Disease intake did not pass external verification; workflow will abstain."
                if assessment.abstain_recommended
                else "Disease intake guard accepted the externally verified identity."
            ),
            node="ma1.intake_guard",
            metadata={
                "intake_validation": assessment.model_dump(mode="json"),
                "reason_codes": assessment.reason_codes,
                "abstain_recommended": assessment.abstain_recommended,
            },
        )
        return {
            "intake_abstain_recommended": assessment.abstain_recommended,
            "abstain_reason": (
                assessment.explanation if assessment.abstain_recommended else ""
            ),
        }

    @staticmethod
    def _route_disease_intake(state: WorkflowState) -> str:
        return "abstain" if state.get("intake_abstain_recommended", True) else "g0"

    async def _run_g0(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        packet = state["hypothesis_packet"]
        await self._transition(
            run_id,
            RunStage.G0,
            "MA3 G0 read-only hypothesis audit running.",
            "stage_started",
            node="g0.audit",
            metadata={
                "interaction": interaction(
                    "MA1",
                    "MA3 · G0",
                    "audit_request",
                    "가설의 근거 추적성·반증 가능성·포트폴리오 무결성 독립 검사 요청",
                    basis=[
                        f"잠정 가설 {len(packet.provisional_hypotheses)}개 · 승인 대상 {len(packet.hypotheses)}개",
                        f"가설 패킷 {state['hypothesis_packet'].packet_id}",
                    ],
                    backend="deterministic_rules_with_optional_external_reviewer",
                    artifact_ids=[state["hypothesis_packet"].packet_id],
                )
            },
        )
        if packet.provisional_hypotheses:
            from drugpilot.orchestration.hypothesis_admission import review_submission
            packet = await review_submission(self, state["brief"], packet, run_id)
        verdict = await self.ma3.audit_g0(
            state["brief"],
            packet,
            on_usage=lambda usage: self._record_llm_usage(run_id, usage),
            before_review=lambda model, prompt, max_output: self._preflight_llm(
                run_id, model, prompt, max_output
            ),
        )
        audit_trace = audit_handoff(verdict)
        audit_trace["backend"] = (
            "external_reviewer_plus_rules"
            if verdict.reviewer_invoked
            else "deterministic_rules"
        )
        await self._artifact_transition(
            run_id,
            RunStage.G0,
            f"G0 verdict: {verdict.verdict.value}",
            "gate_verdict",
            "g0.audit",
            verdict.verdict_id,
            "GateVerdict",
            verdict,
            metadata={
                "gate": "G0",
                "verdict": verdict.verdict.value,
                "interaction": audit_trace,
            },
        )
        from drugpilot.orchestration.g0_followup import enabled as g0_followup_enabled, build_plan
        if g0_followup_enabled(self) and verdict.verdict is not GateVerdictCode.PASS:
            plan = build_plan(state["brief"], packet, verdict,
                rework_loops=state.get("rework_loops", 0),
                target_limit=self.settings.ma1_supplementary_evidence_retry_budget)
            await self._artifact_transition(run_id, RunStage.G0,
                "G0 자동 보완 계획 · " + plan["reason_code"], "g0_followup_planned",
                "g0.followup", plan["plan_id"], "G0FollowupPlan", plan,
                metadata={"followup_plan": plan})
        return {"hypothesis_packet": packet, "g0_verdict": verdict, "rework_order": verdict.rework_order}

    async def _run_planner(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        reworking = state.get("rework_order") is not None
        if reworking:
            await self._record_rework(run_id, state["rework_order"])
        await self._transition(
            run_id,
            RunStage.PLANNER,
            "Planner is compiling approved tasks, tools, criteria and stop conditions.",
            "stage_started",
            node="planner",
            metadata={
                "interaction": interaction(
                    "MA3 · G0",
                    "Planner",
                    "task_assignment",
                    "감사를 통과한 가설만 제한된 도구·예산·중단 조건을 갖춘 실행 명세로 변환 요청",
                    basis=[
                        f"G0 판정 {state['g0_verdict'].verdict.value}",
                        state["g0_verdict"].rationale,
                    ],
                    decision=state["g0_verdict"].verdict.value,
                    backend=self._component_backend("Planner"),
                    model=(
                        self.settings.terra_model
                        if self.settings.effective_mode is not RunMode.MOCK
                        else None
                    ),
                    artifact_ids=[state["hypothesis_packet"].packet_id],
                )
            },
        )
        deterministic_plan = await asyncio.to_thread(
            self.planner.run,
            state["hypothesis_packet"],
            state["brief"],
            revision_context=state.get("rework_order"),
        )
        narrative = await self._llm_enhance(
            run_id,
            component="Planner",
            model=self.settings.terra_model,
            response_model=PlannerNarrativeOverlayOutput,
            fallback=_planner_output_fallback(deterministic_plan),
            prompt=(
                "Suggest only supplementary acceptance-criterion wording and explanatory "
                "tool-purpose text for the supplied deterministic PlanSpec. Also review the "
                "source_coverage registry and provide priority, rationale, and query-focus "
                "suggestions for as many registered sources as scientifically useful. The "
                "deterministic broad-coverage policy already admits every allowed, applicable, "
                "configured source: your recommendation cannot add, remove, authorize, or "
                "dispatch a source. Every non-admitted source retains its recorded rejection "
                "reason. Return only task IDs and source names already present. Do not propose "
                "task names, tools, operations, models, dependencies, numerical limits, budgets, "
                "retries, stop conditions, candidate counts, policy constraints, identifiers, "
                "or scientific results.\nDETERMINISTIC_PLAN:\n"
                + deterministic_plan.model_dump_json()
                + "\nAUDITED_HYPOTHESIS_CONTEXT:\n"
                + json.dumps(
                    _ma1_narrative_evidence_view(state["hypothesis_packet"]),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            max_output_tokens=12_000,
            postprocess=_planner_overlay_from_output,
            postprocess_failure_is_fatal=True,
        )
        plan = _apply_planner_narrative_overlay(
            deterministic_plan,
            narrative,
        )
        if _planner_execution_contract(plan) != _planner_execution_contract(
            deterministic_plan
        ):
            raise ValueError("Planner narrative overlay changed the executable PlanSpec")
        handoff = plan_handoff(plan)
        handoff["backend"] = self._component_backend("Planner")
        if self.settings.effective_mode is not RunMode.MOCK:
            handoff["model"] = self.settings.terra_model
        await self._artifact_transition(
            run_id,
            RunStage.PLANNER,
            "PlanSpec approved by schema and policy checks.",
            "planner_completed",
            "planner",
            plan.plan_id,
            "PlanSpec",
            plan,
            metadata={
                "interaction": handoff,
                "discovery_priority": {
                    "requested": plan.requested_discovery_priority,
                    "effective": plan.discovery_priority,
                    "derivation": plan.discovery_priority_derivation,
                },
                "source_coverage": (
                    {
                        "strategy": plan.source_coverage.strategy,
                        "registry_source_count": plan.source_coverage.registry_source_count,
                        "eligible_source_count": plan.source_coverage.eligible_source_count,
                        "admitted_source_count": plan.source_coverage.admitted_source_count,
                        "eligible_coverage_ratio": plan.source_coverage.eligible_coverage_ratio,
                        "llm_overlay_applied": plan.source_coverage.llm_overlay_applied,
                        "rejected_llm_suggestion_count": len(
                            plan.source_coverage.llm_rejected_suggestions
                        ),
                    }
                    if plan.source_coverage is not None
                    else None
                ),
            },
        )
        return {
            "plan_spec": plan,
            "rework_order": None,
            "rework_loops": state.get("rework_loops", 0) + (1 if reworking else 0),
        }

    async def _guard_ma2_route(self, state: WorkflowState) -> WorkflowState:
        """Honor persisted route-policy abstention before any MA2 tool call."""

        if self.settings.ma2_portfolio_enabled:
            previews: list[dict[str, Any]] = []
            seen_targets: set[str] = set()
            for item in state["hypothesis_packet"].hypotheses:
                target = item.target
                target_key = str(
                    target.uniprot_id or target.ensembl_id or target.symbol
                ).strip().upper()
                if not target_key or target_key in seen_targets:
                    continue
                seen_targets.add(target_key)
                decision = select_route_from_evidence(item, target)
                fallback = decision.get("fallback", {})
                recommends_abstain = bool(
                    isinstance(fallback, dict)
                    and fallback.get("abstain_recommended")
                )
                previews.append(
                    {
                        "hypothesis_id": item.hypothesis_id,
                        "target_key": target_key,
                        "selected_route": decision.get("selected_route"),
                        "abstain_recommended": recommends_abstain,
                        "reason": (
                            fallback.get("reason")
                            if isinstance(fallback, dict)
                            else None
                        ),
                    }
                )
            runnable_count = sum(
                not item["abstain_recommended"] for item in previews
            )
            abstain = runnable_count == 0
            reason_codes = (
                ["ROUTE_EVIDENCE_INSUFFICIENT"] if abstain else []
            )
            await self._trace_event(
                state["run_id"],
                "route_guard_abstain" if abstain else "route_guard_passed",
                (
                    "No portfolio target has sufficient evidence for route assignment; "
                    "MA2 execution is withheld."
                    if abstain
                    else "Portfolio route guard admitted evidence-qualified targets; "
                    "ambiguous targets remain recorded but are not executed."
                ),
                node="planner.route_guard",
                metadata={
                    "portfolio_enabled": True,
                    "abstain_recommended": abstain,
                    "reason_codes": reason_codes,
                    "runnable_target_count": runnable_count,
                    "route_decision_previews": previews,
                },
            )
            return {
                "route_abstain_recommended": abstain,
                "abstain_reason": (
                    "No portfolio target has sufficient route evidence; additional "
                    "evidence or researcher clarification is required."
                    if abstain
                    else ""
                ),
            }

        hypothesis, selection = select_hypothesis_for_priority(
            state["hypothesis_packet"].hypotheses,
            state["plan_spec"].discovery_priority,
        )
        route_decision = select_route_from_evidence(hypothesis, hypothesis.target)
        fallback = route_decision.get("fallback", {})
        route_recommends_abstain = bool(
            isinstance(fallback, dict) and fallback.get("abstain_recommended")
        )
        selection_recommends_abstain = bool(
            selection.get("abstain_recommended_in_live")
        )
        abstain = route_recommends_abstain or selection_recommends_abstain
        reason_codes = []
        if route_recommends_abstain:
            reason_codes.append("ROUTE_EVIDENCE_INSUFFICIENT")
        if selection_recommends_abstain:
            reason_codes.append("DISCOVERY_PRIORITY_EVIDENCE_MISMATCH")
        await self._trace_event(
            state["run_id"],
            "route_guard_abstain" if abstain else "route_guard_passed",
            (
                "Route policy recommends abstention; MA2 candidate generation is withheld."
                if abstain
                else "Route policy guard approved one evidence-qualified MA2 route."
            ),
            node="planner.route_guard",
            metadata={
                "abstain_recommended": abstain,
                "reason_codes": reason_codes,
                "hypothesis_selection": selection,
                "route_decision_preview": route_decision,
            },
        )
        return {
            "route_abstain_recommended": abstain,
            "abstain_reason": (
                "Route evidence is insufficient for candidate generation; additional "
                "evidence or researcher clarification is required."
                if abstain
                else ""
            ),
        }

    @staticmethod
    def _route_ma2_guard(state: WorkflowState) -> str:
        return "abstain" if state.get("route_abstain_recommended", True) else "ma2"

    async def _run_ma2(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        run_ma2 = self._ma2_for_brief(state["brief"])
        research_budget = self.settings.evidence_budget_for(
            state["brief"].research_depth
        )
        if state["plan_spec"].research_depth != research_budget.depth:
            raise ValueError("PlanSpec research depth differs from ResearchBrief")
        coverage = state["plan_spec"].source_coverage
        if (
            coverage is None
            or coverage.budget.research_depth != research_budget.depth
            or coverage.budget.candidate_enrichment_per_route
            != research_budget.candidate_enrichment_per_route
        ):
            raise ValueError("PlanSpec source budget differs from ResearchBrief")
        order = state.get("rework_order")
        previous = state.get("candidate_packet")
        reworking = order is not None and previous is not None
        metadata: dict[str, Any] = {
            "route_policy": ROUTE_POLICY_VERSION,
            "research_depth": research_budget.depth,
            "candidate_enrichment_per_route": (
                research_budget.candidate_enrichment_per_route
            ),
            "interaction": interaction(
                "Planner",
                "MA2",
                "task_assignment",
                "승인된 PlanSpec에 따라 MA1 근거로 단일 후보 경로를 선택한 뒤 후보 생성·계산 요청",
                basis=[
                    f"계획 작업 {len(state['plan_spec'].tasks)}개",
                    f"선택 경로 후보 {state['plan_spec'].candidates_per_route}건",
                    f"승인 도구 {len(state['plan_spec'].approved_tools)}종",
                ],
                backend="deterministic_tool_executor",
                artifact_ids=[
                    state["plan_spec"].plan_id,
                    state["hypothesis_packet"].packet_id,
                ],
            ),
        }
        if reworking and order:
            await self._record_rework(run_id, order)
            metadata.update(
                {
                    "invalidated_nodes": self.dependency.invalidated_nodes(order),
                    "reused_nodes": self.dependency.reusable_nodes(order),
                }
            )
        await self._transition(
            run_id,
            RunStage.MA2,
            "MA2 is selecting an evidence-governed route, then executing that route.",
            "stage_started",
            node="ma2",
            metadata=metadata,
        )
        owns_run_ma2 = run_ma2 is not self.ma2
        loop = asyncio.get_running_loop()
        previous_checkpoint_callback = getattr(run_ma2, "before_operation", None)
        run_ma2.before_operation = lambda: asyncio.run_coroutine_threadsafe(
            self._cancellation_checkpoint(run_id), loop
        ).result()
        try:
            if reworking and order and previous:
                packet = await asyncio.to_thread(
                    run_ma2.rework,
                    state["plan_spec"],
                    state["hypothesis_packet"],
                    previous,
                    order,
                    allowed_sources=state["brief"].data_policy.allowed_sources,
                )
            else:
                fault = (
                    FaultType.NONE
                    if state.get("fault_injected", False)
                    else state["brief"].fault_injection
                )
                packet = await asyncio.to_thread(
                    run_ma2.run,
                    state["plan_spec"],
                    state["hypothesis_packet"],
                    fault=fault,
                    allowed_sources=state["brief"].data_policy.allowed_sources,
                )
        finally:
            run_ma2.before_operation = previous_checkpoint_callback
            if owns_run_ma2:
                run_ma2.close()
        # MA2's CandidatePacket is an immutable scientific-computation
        # artifact.  It is intentionally not passed through a generative
        # rewrite: docking/ADMET/activity values, units, identifiers and tool
        # traces remain byte-for-byte outputs of the approved tool gateway and
        # deterministic Judge.  A future tool-calling LLM may choose approved
        # calls, but still must not author their numeric results.
        candidate_evidence = [
            evidence
            for candidate in packet.candidates
            for evidence in candidate.evidence
        ]
        candidate_provenance_counts: dict[str, int] = {}
        for evidence in candidate_evidence:
            provenance = str(
                evidence.metadata.get("provenance_class")
                or ("external_live" if evidence.metadata.get("live") else "fixture")
            )
            candidate_provenance_counts[provenance] = (
                candidate_provenance_counts.get(provenance, 0) + 1
            )
        source_query_status_counts: dict[str, int] = {}
        for query in packet.metadata.get("public_enrichment_queries", []):
            status = str(query.get("status", "unknown"))
            source_query_status_counts[status] = (
                source_query_status_counts.get(status, 0) + 1
            )
        metadata.update(
            {
                "route_decision": packet.metadata.get("route_decision"),
                "selected_route": packet.metadata.get("route_decision", {}).get(
                    "selected_route"
                ),
                "routes": packet.metadata.get("route_decision", {}).get(
                    "selected_routes", []
                ),
                "evidence_sources": sorted(
                    {evidence.source for evidence in candidate_evidence}
                ),
                "evidence_provenance_counts": candidate_provenance_counts,
                "public_source_query_status_counts": source_query_status_counts,
                "public_candidate_enrichment_enabled": bool(
                    run_ma2.evidence_enricher is not None
                    and run_ma2.enrichment_per_route > 0
                ),
                "interaction": candidate_handoff(packet),
            }
        )
        await self._artifact_transition(
            run_id,
            RunStage.MA2,
            f"MA2 produced {len(packet.candidates)} ranked candidates.",
            "ma2_completed",
            "ma2.judge",
            packet.packet_id,
            "CandidatePacket",
            packet,
            metadata=metadata,
        )
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot:
            snapshot.rework_loops = state.get("rework_loops", 0) + (
                1 if reworking else 0
            )
            await self.store.save_snapshot(snapshot)
        return {
            "candidate_packet": packet,
            "fault_injected": state.get("fault_injected", False)
            or state["brief"].fault_injection is not FaultType.NONE,
            "rework_order": None,
            "rework_loops": state.get("rework_loops", 0) + (1 if reworking else 0),
        }

    async def _run_g1(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        await self._transition(
            run_id,
            RunStage.G1,
            "MA3 G1 candidate and process audit running.",
            "stage_started",
            node="g1.audit",
            metadata={
                "interaction": interaction(
                    "MA2 · Judge",
                    "MA3 · G1",
                    "audit_request",
                    "후보 화학·활성·안전성·계획/도구 추적 무결성 독립 검사 요청",
                    basis=[
                        f"후보 {len(state['candidate_packet'].candidates)}개",
                        f"도구 추적 {len(state['candidate_packet'].tool_calls)}건",
                        f"PlanSpec {state['plan_spec'].plan_id}",
                    ],
                    backend="deterministic_rules_with_optional_external_reviewer",
                    artifact_ids=[
                        state["candidate_packet"].packet_id,
                        state["plan_spec"].plan_id,
                    ],
                )
            },
        )
        verdict = await self.ma3.audit_g1(
            state["brief"],
            state["hypothesis_packet"],
            state["plan_spec"],
            state["candidate_packet"],
            on_usage=lambda usage: self._record_llm_usage(run_id, usage),
            before_review=lambda model, prompt, max_output: self._preflight_llm(
                run_id, model, prompt, max_output
            ),
        )
        audit_trace = audit_handoff(verdict)
        audit_trace["backend"] = (
            "external_reviewer_plus_rules"
            if verdict.reviewer_invoked
            else "deterministic_rules"
        )
        await self._artifact_transition(
            run_id,
            RunStage.G1,
            f"G1 verdict: {verdict.verdict.value}",
            "gate_verdict",
            "g1.audit",
            verdict.verdict_id,
            "GateVerdict",
            verdict,
            metadata={
                "gate": "G1",
                "verdict": verdict.verdict.value,
                "interaction": audit_trace,
            },
        )
        return {"g1_verdict": verdict, "rework_order": verdict.rework_order}

    def _route_g0(self, state: WorkflowState) -> str:
        verdict = state["g0_verdict"].verdict
        if verdict is GateVerdictCode.PASS:
            return "planner"
        if verdict in {GateVerdictCode.REVISE, GateVerdictCode.REPLAN}:
            from drugpilot.orchestration.g0_followup import enabled as g0_followup_enabled, build_plan
            if g0_followup_enabled(self):
                plan = build_plan(state["brief"], state["hypothesis_packet"], state["g0_verdict"],
                    rework_loops=state.get("rework_loops", 0),
                    target_limit=self.settings.ma1_supplementary_evidence_retry_budget)
                if not plan["eligible"]:
                    return "human_gate"
            elif not state["hypothesis_packet"].hypotheses:
                # Without a verified follow-up plan, zero retained hypotheses
                # cannot authorize another paid proposer pass.
                return "human_gate"
            return "ma1" if self._can_rework(state) else "human_gate"
        if verdict is GateVerdictCode.ABSTAIN:
            return "human_gate"
        return "block"

    def _route_g1(self, state: WorkflowState) -> str:
        verdict = state["g1_verdict"].verdict
        if verdict is GateVerdictCode.PASS:
            return "human_gate"
        if verdict is GateVerdictCode.REVISE:
            return "ma2" if self._can_rework(state) else "human_gate"
        if verdict is GateVerdictCode.REPLAN:
            return "planner" if self._can_rework(state) else "human_gate"
        if verdict is GateVerdictCode.ABSTAIN:
            return "human_gate"
        return "block"

    @staticmethod
    def _can_rework(state: WorkflowState) -> bool:
        packet = state.get("hypothesis_packet")
        if packet is not None:
            lineage = packet.metadata.get("revision_lineage")
            if (
                isinstance(lineage, dict)
                and lineage.get("scientific_changed") is False
            ):
                # One auditable rework attempt is enough when it reproduced the
                # same scientific packet.  Repeating it would spend model/source
                # budget without introducing a new hypothesis or evidence link.
                return False
        return state.get("rework_loops", 0) < state["brief"].constraints.max_rework_loops

    async def _await_human(self, state: WorkflowState) -> WorkflowState:
        from drugpilot.orchestration.automatic_resolution import approval_locked, current_report
        snapshot = await self.store.get_snapshot(state["run_id"])
        auto = current_report(snapshot) if snapshot else None
        if (snapshot and snapshot.mode is RunMode.LIVE and snapshot.g1_verdict
                and snapshot.candidate_packet and snapshot.g0_verdict
                and snapshot.g0_verdict.verdict is GateVerdictCode.PASS
                and (snapshot.g1_verdict.verdict is not GateVerdictCode.PASS
                     or any(not f.resolved and f.severity.value != "INFO" for f in snapshot.g1_verdict.findings))
                and self.settings.automatic_resolution_enabled and self.settings.public_evidence_enabled
                and not self.settings.demo_fixture_fallback_enabled):
            from drugpilot.orchestration.human_review import HumanReviewService
            token = self._workflow_lease_tokens.get(state["run_id"])
            if not token:
                raise ValueError("AUTOMATIC_WORKFLOW_LEASE_REQUIRED")
            auto = await HumanReviewService(self).run_automatic_inline(state["run_id"], token)
            snapshot = await self.store.get_snapshot(state["run_id"])
        clean_pass = bool(
            snapshot
            and snapshot.g1_verdict
            and snapshot.g1_verdict.verdict is GateVerdictCode.PASS
            and not approval_locked(snapshot)
        )
        if not clean_pass:
            await self._materialize_next_action_packet(
                state["run_id"],
                trigger_stage=snapshot.stage if snapshot else RunStage.G1,
                terminal_message=(
                    snapshot.status_message
                    if snapshot
                    else "Automated audit requires additional action."
                ),
            )
        zero_retained = bool(snapshot and snapshot.hypothesis_packet
                             and not snapshot.hypothesis_packet.hypotheses)
        await self._transition(
            state["run_id"],
            RunStage.HUMAN_GATE,
            snapshot.status_message if auto else (
                "All automated gates passed; awaiting researcher decision." if clean_pass else
                "가설 0개 · 추가로 실행할 유효한 자동 보강 작업이 없어 보류되었습니다." if zero_retained else
                "Automated audit requires action; APPROVE is locked pending a fresh G1 PASS."),
            "human_gate_ready" if clean_pass else "action_required_human_gate",
            node="human_gate",
        )
        await self.broker.close(state["run_id"])
        return {}

    async def _abstain(self, state: WorkflowState) -> WorkflowState:
        reason = state.get("abstain_reason")
        packet = state.get("hypothesis_packet")
        if packet is not None:
            lineage = packet.metadata.get("revision_lineage")
            if (
                isinstance(lineage, dict)
                and lineage.get("scientific_changed") is False
                and reason in {
                    None,
                    "Evidence or model confidence is insufficient; human review required.",
                }
            ):
                reason = (
                    "MA1 rework reproduced the same scientific packet; execution "
                    "stopped to avoid a cost-only retry without new evidence."
                )
        await self._terminal(
            state["run_id"],
            RunStage.ABSTAINED,
            reason
            or "Evidence or model confidence is insufficient; human review required.",
        )
        return {}

    async def _block(self, state: WorkflowState) -> WorkflowState:
        await self._terminal(
            state["run_id"],
            RunStage.BLOCKED,
            "A blocking provenance, policy, or safety defect stopped execution.",
        )
        return {}

    @serialized_review_mutation
    async def apply_human_decision(self, decision: HumanGateDecision, *, _review_lease_token: str | None = None) -> RunSnapshot:
        snapshot = await self.store.get_snapshot(decision.run_id)
        if snapshot is None:
            raise KeyError(decision.run_id)
        if snapshot.stage is not RunStage.HUMAN_GATE:
            raise ValueError("human decision is allowed only at the Human Gate")
        from drugpilot.orchestration.automatic_resolution import approval_locked
        if decision.decision.value == "APPROVE" and approval_locked(snapshot):
            raise ValueError("APPROVE is locked: automatic completion is unverified or HOLD/BLOCKED")
        if (
            snapshot.next_action_packet is not None
            and not snapshot.next_action_packet.approval_allowed
            and decision.decision.value == "APPROVE"
        ):
            raise ValueError(
                "APPROVE is locked while required actions remain unresolved; "
                "record HOLD or REJECT and satisfy the packet re-entry conditions"
            )
        valid_ids = {
            candidate.candidate_id
            for candidate in (snapshot.candidate_packet.candidates if snapshot.candidate_packet else [])
        }
        if not set(decision.selected_candidate_ids) <= valid_ids:
            raise ValueError("human decision references an unknown candidate")
        eligible_ids = eligible_candidate_ids(snapshot)
        if decision.decision.value == "APPROVE":
            if not snapshot.g0_verdict or snapshot.g0_verdict.verdict is not GateVerdictCode.PASS:
                raise ValueError("APPROVE requires G0 PASS")
            if not snapshot.g1_verdict or snapshot.g1_verdict.verdict is not GateVerdictCode.PASS:
                raise ValueError("APPROVE requires a fresh G1 PASS")
            if not decision.selected_candidate_ids:
                raise ValueError("APPROVE requires at least one selected candidate")
            if not set(decision.selected_candidate_ids) <= eligible_ids:
                raise ValueError("excluded candidates cannot be approved")
        snapshot.human_decision = decision
        snapshot.stage = (
            RunStage.COMPLETED
            if decision.decision.value == "APPROVE"
            else RunStage.ABSTAINED
            if decision.decision.value == "HOLD"
            else RunStage.BLOCKED
        )
        snapshot.status_message = (
            "Action-required HOLD recorded; complete the next-action packet before re-entry."
            if snapshot.next_action_packet is not None
            and decision.decision.value == "HOLD"
            else f"Human decision recorded: {decision.decision.value}."
        )
        decision_digest = sha256(decision.model_dump_json().encode()).hexdigest()
        event = ExecutionEvent(
            run_id=snapshot.run_id, stage=snapshot.stage, event_type="human_decision",
            message=snapshot.status_message,
            node="human_gate",
            artifact_ids=[decision.decision_id],
            metadata={
                "decision": decision.decision.value,
                "artifact_sha256": {
                    decision.decision_id: decision_digest,
                },
            },
        )
        snapshot.events.append(event)
        await self.store.commit_review_update(snapshot,
            artifacts=[(decision.decision_id, "HumanGateDecision", decision)],
            events=[event], lease_token=_review_lease_token)
        await self.broker.publish(event)
        return snapshot

    async def resume(self, run_id: str) -> RunSnapshot:
        async with self._task_registry_lock:
            await self._cancellation_checkpoint(run_id)
            snapshot = await self.store.get_snapshot(run_id)
            if snapshot is None:
                raise KeyError(run_id)
            active = self._tasks.get(run_id)
            lease = None
            if snapshot.mode is RunMode.LIVE:
                lease = await self.store.live_execution_lease_status(
                    run_id, now=utc_now()
                )
                if lease is not None and lease["state"] == "active":
                    raise ValueError("RUN_ACTIVE")
            if (
                active is not None
                and not active.done()
                and snapshot.mode is not RunMode.LIVE
            ):
                return snapshot
            terminal_semantics = bool(
                snapshot.human_decision is not None
                or snapshot.stage
                in {RunStage.HUMAN_GATE, RunStage.COMPLETED, RunStage.BLOCKED}
            )
            if (
                snapshot.mode is RunMode.LIVE
                and lease is not None
                and lease["state"] == "expired_recovery_required"
                and terminal_semantics
            ):
                if not await self.store.verify_ledger(run_id):
                    raise ValueError(
                        "LEDGER_INTEGRITY_FAILED: terminal lease was not reconciled"
                    )
                retired = await self.store.retire_expired_live_execution_lease(
                    run_id,
                    expected_stage=snapshot.stage.value,
                    now=utc_now(),
                )
                if not retired:
                    raise ValueError("LEASE_RECONCILIATION_CONFLICT")
                # This endpoint is also the explicit cleanup action advertised
                # by resume-status. A verified stale terminal lease is retired
                # without reopening or starting any workflow, regardless of
                # which persisted terminal/human semantic applies.
                return snapshot
            if snapshot.human_decision is not None:
                raise ValueError(
                    "a persisted human decision cannot be reopened by resume"
                )
            if snapshot.stage is RunStage.HUMAN_GATE:
                # Waiting for a researcher is not an interrupted execution.
                # Never bypass or replay the human gate.
                return snapshot
            if snapshot.stage in {
                RunStage.COMPLETED,
                RunStage.BLOCKED,
            }:
                raise ValueError("a terminal run cannot be resumed")
            resumable_stages = {
                RunStage.CREATED,
                RunStage.BRIEF_VALIDATED,
                RunStage.MA1,
                RunStage.G0,
                RunStage.PLANNER,
                RunStage.MA2,
                RunStage.G1,
                RunStage.FAILED,
                RunStage.ABSTAINED,
            }
            if snapshot.stage not in resumable_stages:
                raise ValueError("run is not in a resumable stage")

            try:
                checkpoint = await self.store.latest_stage_checkpoint(run_id)
            except CheckpointError as exc:
                raise ValueError(
                    "checkpoint verification failed; the run was not resumed"
                ) from exc

            if checkpoint is None:
                if any(
                    item is not None
                    for item in (
                        snapshot.hypothesis_packet,
                        snapshot.g0_verdict,
                        snapshot.plan_spec,
                        snapshot.candidate_packet,
                        snapshot.g1_verdict,
                    )
                ):
                    raise ValueError(
                        "completed artifacts exist without a compatible stage checkpoint"
                    )
                resume_state: WorkflowState = {
                    "run_id": run_id,
                    "brief": snapshot.brief,
                    "rework_loops": snapshot.rework_loops,
                    "fault_injected": False,
                    "rework_order": None,
                    "_resume_from": "validate_brief",
                }
                checkpoint_id = None
                checkpoint_digest = None
                reused_artifacts: dict[str, str] = {}
            else:
                if not _resume_briefs_are_compatible(
                    checkpoint.brief, snapshot.brief
                ):
                    raise ValueError(
                        "checkpoint Research Brief does not match the persisted run"
                    )
                resume_state = self._state_from_checkpoint(checkpoint)
                resume_state["_resume_from"] = self._node_after_checkpoint(
                    checkpoint, resume_state
                )
                resume_state["_resume_checkpoint_id"] = checkpoint.checkpoint_id
                checkpoint_id = checkpoint.checkpoint_id
                checkpoint_digest = self.store.checkpoint_digest(checkpoint)
                reused_artifacts = dict(checkpoint.artifact_sha256)

            budget_resume_override: dict[str, int | float] | None = None
            if (
                isinstance(snapshot.error, str)
                and snapshot.error.startswith(
                    "single input cap would be exceeded"
                )
                and self.settings.max_single_input_tokens
                > snapshot.brief.budget.max_single_input_tokens
            ):
                previous_limit = snapshot.brief.budget.max_single_input_tokens
                updated_budget = snapshot.brief.budget.model_copy(
                    update={
                        "max_single_input_tokens": (
                            self.settings.max_single_input_tokens
                        )
                    }
                )
                updated_brief = snapshot.brief.model_copy(
                    update={"budget": updated_budget}
                )
                snapshot.brief = updated_brief
                resume_state["brief"] = updated_brief
                budget_resume_override = {
                    "previous_max_single_input_tokens": previous_limit,
                    "effective_max_single_input_tokens": (
                        updated_budget.max_single_input_tokens
                    ),
                }
            elif (
                isinstance(snapshot.error, str)
                and snapshot.error.startswith(
                    "run cost cap would be exceeded"
                )
                and self.settings.max_run_cost_usd
                > snapshot.brief.budget.max_run_cost_usd
            ):
                previous_limit = snapshot.brief.budget.max_run_cost_usd
                updated_budget = snapshot.brief.budget.model_copy(
                    update={
                        "max_run_cost_usd": self.settings.max_run_cost_usd
                    }
                )
                updated_brief = snapshot.brief.model_copy(
                    update={"budget": updated_budget}
                )
                snapshot.brief = updated_brief
                resume_state["brief"] = updated_brief
                budget_resume_override = {
                    "previous_max_run_cost_usd": previous_limit,
                    "effective_max_run_cost_usd": (
                        updated_budget.max_run_cost_usd
                    ),
                }

            lease_token: str | None = None
            if snapshot.mode is RunMode.LIVE:
                if not await self.store.verify_ledger(run_id):
                    raise ValueError(
                        "LEDGER_INTEGRITY_FAILED: explicit recovery was refused"
                    )
                lease_token = new_id("lease")
                await self.store.claim_live_execution_lease(
                    run_id,
                    lease_token=lease_token,
                    now=utc_now(),
                    lease_ttl_seconds=self.settings.beta_live_lease_ttl_seconds,
                    max_active_live_runs=self.settings.beta_max_active_live_runs,
                )

            try:
                snapshot.error = None
                snapshot.status_message = (
                    f"Run queued from {resume_state['_resume_from']}; completed "
                    "checkpoint artifacts will be reused."
                )
                await self._emit(
                    snapshot,
                    snapshot.stage,
                    "workflow_resume_queued",
                    snapshot.status_message,
                    node="orchestrator.resume",
                    metadata={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_sha256": checkpoint_digest,
                        "resume_from_node": resume_state["_resume_from"],
                        "reused_artifact_sha256": reused_artifacts,
                        "reused_completed_scientific_stages": bool(checkpoint),
                        "budget_resume_override": budget_resume_override,
                        "durable_execution_lease": lease_token is not None,
                    },
                )
                await self.store.save_snapshot(snapshot)
                workflow = (
                    self._execute_with_lease(
                        run_id,
                        snapshot.brief,
                        lease_token=lease_token,
                        resume_state=resume_state,
                    )
                    if lease_token is not None
                    else self._execute(run_id, snapshot.brief, resume_state)
                )
                task: asyncio.Task[None] | None = None
                try:
                    task = asyncio.create_task(
                        workflow, name=f"drugpilot:{run_id}:resume"
                    )
                    self._track_task(run_id, task)
                except Exception:
                    if task is None:
                        workflow.close()
                    else:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                    raise
                return snapshot
            except Exception:
                if lease_token is not None:
                    await self.store.expire_live_execution_lease_for_recovery(
                        run_id,
                        lease_token=lease_token,
                        reason="resume_task_registration_failed",
                        now=utc_now(),
                    )
                raise

    async def resume_status(self, run_id: str) -> dict[str, Any]:
        """Return a safe UI contract for an interrupted run's resume action."""

        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        if snapshot.stage in {RunStage.CANCEL_REQUESTED, RunStage.CANCELLED}:
            return {"run_id": run_id, "eligible": False, "reason_code": "RUN_CANCELLED",
                "reason": "User cancellation cannot be reopened by resume.",
                "active_in_process": bool((task := self._tasks.get(run_id)) and not task.done()),
                "current_stage": snapshot.stage.value, "resume_from_node": None,
                "checkpoint_completed_node": None, "checkpoint_integrity": "not_required"}
        lease_contract: dict[str, str | None] = {}
        if snapshot.mode is RunMode.LIVE:
            lease = await self.store.live_execution_lease_status(
                run_id, now=utc_now()
            )
            lease_contract["execution_lease_state"] = (
                lease["state"] if lease is not None else None
            )
            if lease is not None and lease["state"] == "active":
                return {
                    "run_id": run_id,
                    "eligible": False,
                    "reason_code": "RUN_ACTIVE",
                    "reason": "The workflow owns an active durable execution lease.",
                    "active_in_process": bool(
                        (task := self._tasks.get(run_id)) is not None
                        and not task.done()
                    ),
                    "execution_lease_state": "active",
                    "current_stage": snapshot.stage.value,
                    "resume_from_node": None,
                    "checkpoint_completed_node": None,
                    "checkpoint_integrity": "not_checked_while_active",
                }
            if (
                lease is not None
                and lease["state"] == "expired_recovery_required"
                and (
                    snapshot.human_decision is not None
                    or snapshot.stage
                    in {RunStage.HUMAN_GATE, RunStage.COMPLETED, RunStage.BLOCKED}
                )
            ):
                return {
                    "run_id": run_id,
                    "eligible": True,
                    "reason_code": "EXPIRED_TERMINAL_LEASE_RECONCILIATION_REQUIRED",
                    "reason": "The stale lease can be retired after explicit integrity verification; no workflow will start.",
                    "active_in_process": False,
                    "execution_lease_state": "expired_recovery_required",
                    "current_stage": snapshot.stage.value,
                    "resume_from_node": None,
                    "checkpoint_completed_node": None,
                    "checkpoint_integrity": "verification_required_on_reconcile",
                }
        active = self._tasks.get(run_id)
        if active is not None and not active.done():
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": False,
                "reason_code": "RUN_ACTIVE",
                "reason": "The workflow is currently active in this server process.",
                "active_in_process": True,
                "current_stage": snapshot.stage.value,
                "resume_from_node": None,
                "checkpoint_completed_node": None,
                "checkpoint_integrity": "not_checked_while_active",
            }
        if snapshot.human_decision is not None:
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": False,
                "reason_code": "HUMAN_DECISION_ALREADY_RECORDED",
                "reason": "A researcher decision is already persisted and cannot be reopened by resume.",
                "active_in_process": False,
                "current_stage": snapshot.stage.value,
                "resume_from_node": None,
                "checkpoint_completed_node": None,
                "checkpoint_integrity": "not_required",
            }
        if snapshot.stage is RunStage.HUMAN_GATE:
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": False,
                "reason_code": "AWAITING_HUMAN_DECISION",
                "reason": "The automated workflow is complete and awaits a researcher decision.",
                "active_in_process": False,
                "current_stage": snapshot.stage.value,
                "resume_from_node": None,
                "checkpoint_completed_node": "g1",
                "checkpoint_integrity": "not_required",
            }
        if snapshot.stage in {RunStage.COMPLETED, RunStage.BLOCKED}:
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": False,
                "reason_code": "RUN_TERMINAL",
                "reason": "The run is terminal and cannot be resumed.",
                "active_in_process": False,
                "current_stage": snapshot.stage.value,
                "resume_from_node": None,
                "checkpoint_completed_node": None,
                "checkpoint_integrity": "not_required",
            }
        try:
            checkpoint = await self.store.latest_stage_checkpoint(run_id)
        except CheckpointError:
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": False,
                "reason_code": "CHECKPOINT_INVALID",
                "reason": "The persisted checkpoint failed compatibility or integrity verification.",
                "active_in_process": False,
                "current_stage": snapshot.stage.value,
                "resume_from_node": None,
                "checkpoint_completed_node": None,
                "checkpoint_integrity": "invalid",
            }
        if checkpoint is None:
            has_completed_artifacts = any(
                item is not None
                for item in (
                    snapshot.hypothesis_packet,
                    snapshot.g0_verdict,
                    snapshot.plan_spec,
                    snapshot.candidate_packet,
                    snapshot.g1_verdict,
                )
            )
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": not has_completed_artifacts,
                "reason_code": (
                    "CHECKPOINT_REQUIRED"
                    if has_completed_artifacts
                    else "RESTART_FROM_BRIEF"
                ),
                "reason": (
                    "Completed artifacts exist without a compatible checkpoint."
                    if has_completed_artifacts
                    else "No scientific stage completed; execution can restart from the Research Brief."
                ),
                "active_in_process": False,
                "current_stage": snapshot.stage.value,
                "resume_from_node": (
                    None if has_completed_artifacts else "validate_brief"
                ),
                "checkpoint_completed_node": None,
                "checkpoint_integrity": "missing",
            }
        if not _resume_briefs_are_compatible(checkpoint.brief, snapshot.brief):
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": False,
                "reason_code": "CHECKPOINT_BRIEF_MISMATCH",
                "reason": "The checkpoint Research Brief does not match the persisted run.",
                "active_in_process": False,
                "current_stage": snapshot.stage.value,
                "resume_from_node": None,
                "checkpoint_completed_node": checkpoint.completed_node,
                "checkpoint_integrity": "invalid",
            }
        try:
            state = self._state_from_checkpoint(checkpoint)
            resume_from = self._node_after_checkpoint(checkpoint, state)
        except ValueError:
            return {
                **lease_contract,
                "run_id": run_id,
                "eligible": False,
                "reason_code": "CHECKPOINT_INVALID",
                "reason": "The checkpoint does not identify a valid next workflow node.",
                "active_in_process": False,
                "current_stage": snapshot.stage.value,
                "resume_from_node": None,
                "checkpoint_completed_node": checkpoint.completed_node,
                "checkpoint_integrity": "invalid",
            }
        return {
            **lease_contract,
            "run_id": run_id,
            "eligible": True,
            "reason_code": "CHECKPOINT_READY",
            "reason": "A verified checkpoint can resume from the first incomplete node.",
            "active_in_process": False,
            "current_stage": snapshot.stage.value,
            "resume_from_node": resume_from,
            "checkpoint_completed_node": checkpoint.completed_node,
            "checkpoint_integrity": "verified",
        }

    @staticmethod
    def _state_from_checkpoint(
        checkpoint: WorkflowCheckpoint,
    ) -> WorkflowState:
        state: WorkflowState = {
            "run_id": checkpoint.run_id,
            "brief": checkpoint.brief,
            "rework_loops": checkpoint.rework_loops,
            "fault_injected": checkpoint.fault_injected,
            "rework_order": checkpoint.rework_order,
            "intake_abstain_recommended": (
                checkpoint.intake_abstain_recommended
            ),
            "route_abstain_recommended": (
                checkpoint.route_abstain_recommended
            ),
        }
        optional = {
            "hypothesis_packet": checkpoint.hypothesis_packet,
            "g0_verdict": checkpoint.g0_verdict,
            "plan_spec": checkpoint.plan_spec,
            "candidate_packet": checkpoint.candidate_packet,
            "g1_verdict": checkpoint.g1_verdict,
            "disease_intake_assessment": checkpoint.disease_intake_assessment,
            "abstain_reason": checkpoint.abstain_reason,
        }
        state.update({key: value for key, value in optional.items() if value is not None})
        return state

    def _node_after_checkpoint(
        self,
        checkpoint: WorkflowCheckpoint,
        state: WorkflowState,
    ) -> str:
        if checkpoint.completed_node == "validate_brief":
            return "ma1"
        if checkpoint.completed_node == "ma1":
            return "intake_guard"
        if checkpoint.completed_node == "g0":
            if state.get("g0_verdict") is None:
                raise ValueError("G0 checkpoint is missing its gate verdict")
            return self._route_g0(state)
        if checkpoint.completed_node == "planner":
            return "route_guard"
        if checkpoint.completed_node == "ma2":
            return "g1"
        if checkpoint.completed_node == "g1":
            if state.get("g1_verdict") is None:
                raise ValueError("G1 checkpoint is missing its gate verdict")
            return self._route_g1(state)
        raise ValueError("checkpoint completed node is not resumable")

    async def _enrich_ma1_pharos(
        self,
        run_id: str,
        packet: HypothesisPacket,
        brief: ResearchBrief,
    ) -> HypothesisPacket:
        """Apply the Orchestrator-owned deterministic Pharos source policy."""

        if needs_review_context_view(packet):
            enriched = await self._enrich_ma1_pharos(run_id, review_context_view(packet), brief)
            return restore_review_context(packet, enriched)
        evidence_budget = self.settings.evidence_budget_for(brief.research_depth)
        pharos_router = self._pharos_router_for_brief(brief)
        result = await asyncio.to_thread(pharos_router.route, packet, brief)
        comprehensive_routing = (
            evidence_budget.depth == "comprehensive"
            and
            _COMPREHENSIVE_PHAROS_ROUTING_HINT in brief.human_priorities
        )
        decision = dict(result.decision)
        routed_packet = result.packet
        if comprehensive_routing:
            decision.update(
                {
                    "request_field": (
                        "ResearchBrief.research_depth + "
                        "ProjectConstraints.druggability_required"
                    ),
                    "request_origin": "deterministic_comprehensive_profile",
                    "human_priority_required": False,
                }
            )
            routed_packet = result.packet.model_copy(deep=True)
            for hypothesis in routed_packet.hypotheses:
                for evidence in hypothesis.evidence:
                    if evidence.source != "Pharos":
                        continue
                    evidence.metadata = {
                        **evidence.metadata,
                        "routing_policy": (
                            "deterministic_comprehensive_druggability_coverage"
                        ),
                        "human_priority_required": False,
                    }
            routed_packet.metadata = {
                **routed_packet.metadata,
                "pharos_routing": decision,
            }
        status = str(decision.get("status") or "skipped")
        outcomes = decision.get("outcomes")
        safe_outcomes = outcomes if isinstance(outcomes, list) else []
        evidence_ids = [
            str(evidence_id)
            for outcome in safe_outcomes
            if isinstance(outcome, dict)
            for evidence_id in outcome.get("evidence_ids", [])
        ]
        selected = decision.get("selected_targets")
        safe_selected = selected if isinstance(selected, list) else []
        reason = str(decision.get("reason") or "no routing reason recorded")
        queried_count = sum(
            1
            for outcome in safe_outcomes
            if isinstance(outcome, dict) and outcome.get("queried") is True
        )
        await self._trace_event(
            run_id,
            (
                "pharos_routing_skipped"
                if status == "skipped"
                else "pharos_routing_completed"
            ),
            (
                "Orchestrator recorded why the configured Pharos lookup was not executed."
                if status == "skipped"
                else "Orchestrator completed bounded on-demand Pharos lookups."
            ),
            node="ma1.source_router.pharos",
            metadata={
                "source_routing_decision": decision,
                "interaction": interaction(
                    "Orchestrator · Source Router",
                    "MA1 · Proposer/Critic",
                    "source_routing_decision",
                    (
                        "결정론적 적용성·설정·예산 기준에서 Pharos를 호출하지 않음"
                        if status == "skipped"
                        else (
                            "포괄적 druggability 정책에 따라 상위 표적의 Pharos 맥락 조회"
                            if comprehensive_routing
                            else "연구자 요청에 따라 상위 표적의 Pharos 맥락을 제한 조회"
                        )
                    ),
                    basis=[
                        f"결정 상태 {status}",
                        f"선택 표적 {len(safe_selected)}개 · 실제 조회 {queried_count}회",
                        f"추가 레코드 {decision.get('records_added', 0)}개",
                        reason,
                        "context-only · claim evidence links unchanged",
                        "결정 주체 Orchestrator deterministic policy",
                    ],
                    decision=status,
                    backend=(
                        "deterministic_source_policy"
                        if status == "skipped"
                        else "pharos_public_api"
                    ),
                    evidence_ids=evidence_ids,
                    artifact_ids=[packet.packet_id],
                    details={
                        "requested": decision.get("requested", False),
                        "request_origin": decision.get("request_origin"),
                        "matched_terms": decision.get("matched_terms", []),
                        "selected_targets": safe_selected,
                        "outcomes": safe_outcomes,
                        "reason": reason,
                    },
                ),
            },
        )
        return routed_packet

    async def _enrich_ma1_context(
        self,
        run_id: str,
        packet: HypothesisPacket,
        brief: ResearchBrief,
    ) -> HypothesisPacket:
        if needs_review_context_view(packet):
            enriched = await self._enrich_ma1_context(run_id, review_context_view(packet), brief)
            return restore_review_context(packet, enriched)
        context_router = self._context_router_for_brief(brief)
        evidence_budget = self.settings.evidence_budget_for(brief.research_depth)
        goal_focus = derive_goal_focus(brief)
        requested_families = [
            family
            for family in goal_focus.source_families
            if family in {"single_cell", "expression", "network"}
        ]
        # An explicit target-cell value is the authoritative classification.
        # Canonical routing priorities may include the phrase "single-cell
        # context" to select an atlas source family; that routing label must
        # not relabel a named cell-resolved input as an explicit single-cell
        # experiment.
        cell_context_kind = classify_cell_context(
            brief.constraints.target_cell_type
        )
        if cell_context_kind == "none":
            cell_context_kind = classify_cell_context(
                None,
                brief.human_priorities,
            )
        if (
            context_router is None
            or not packet.hypotheses
            or evidence_budget.context_targets_per_track <= 0
        ):
            skip_reason = (
                "router_not_configured"
                if context_router is None
                else "no_retained_hypotheses"
                if not packet.hypotheses
                else "context_budget_disabled"
            )
            safe = packet.model_copy(deep=True)
            safe.metadata = {
                **safe.metadata,
                "context_routing": {
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "requested_source_families": requested_families,
                    "executed_source_families": [],
                    "unexecuted_source_families": requested_families,
                    "contract_satisfied": not requested_families,
                    "source_calls_used": 0,
                    "planned_source_calls": 0,
                    "target_cell_context_kind": cell_context_kind,
                    "context_only": True,
                },
            }
            await self._trace_event(
                run_id,
                "context_routing_skipped",
                "MA1 context routing was not executed; the selected-versus-called contract was recorded.",
                node="ma1.context_router",
                metadata={
                    "skip_reason": skip_reason,
                    "requested_source_families": requested_families,
                    "contract_satisfied": not requested_families,
                    "target_cell_context_kind": cell_context_kind,
                },
            )
            return safe

        from drugpilot.tools.context_router import ContextRoutingBudget
        from drugpilot.tools.ma1_context_enrichment import (
            enrich_hypothesis_packet_with_context,
        )

        budget = ContextRoutingBudget(
            max_targets_per_track=evidence_budget.context_targets_per_track,
            max_source_calls=evidence_budget.context_source_calls,
            max_elapsed_seconds=evidence_budget.context_elapsed_seconds,
            limit_per_source=evidence_budget.context_records_per_source,
        )
        gencode_ids = {
            hypothesis.target.symbol: hypothesis.target.ensembl_id
            for hypothesis in packet.hypotheses
            if hypothesis.target.ensembl_id
        }
        try:
            routing = await asyncio.to_thread(
                context_router.route_hypothesis_packet,
                packet,
                budget=budget,
                gencode_ids_by_symbol=gencode_ids,
                target_cell_type=brief.constraints.target_cell_type,
                human_priorities=brief.human_priorities,
                allowed_sources=brief.data_policy.allowed_sources,
            )
            enriched = enrich_hypothesis_packet_with_context(packet, routing)
            context_audit = dict(enriched.metadata.get("context_routing", {}))
            executed_families = sorted(
                {
                    outcome.family
                    for target in routing.targets
                    for outcome in target.sources
                    if outcome.queried
                }
            )
            unexecuted_families = sorted(
                set(requested_families) - set(executed_families)
            )
            context_audit.update(
                {
                    "research_depth": evidence_budget.depth,
                    "budget": budget.model_dump(mode="json"),
                    "requested_source_families": requested_families,
                    "executed_source_families": executed_families,
                    "unexecuted_source_families": unexecuted_families,
                    "contract_satisfied": not unexecuted_families,
                    "target_cell_context_kind": cell_context_kind,
                }
            )
            enriched.metadata = {
                **enriched.metadata,
                "context_routing": context_audit,
            }
            source_counts = enriched.metadata.get("context_routing", {}).get(
                "source_record_counts", {}
            )
            await self._trace_event(
                run_id,
                "context_routing_completed",
                "MA1 context router completed bounded track-specific public DB lookups.",
                node="ma1.context_router",
                metadata={
                    "interaction": interaction(
                        "MA1 · Context Router",
                        "MA1 · Proposer/Critic",
                        "context_enrichment",
                        "상위 표적에 트랙별 공개 DB 맥락을 추가하되 인과 근거로 승격하지 않음",
                        basis=[
                            f"소스 호출 {routing.source_calls_used}/{routing.planned_source_calls}회",
                            f"조회 표적 {len(routing.targets)}개",
                            f"시간/호출 예산 소진 {routing.budget_exhausted}",
                            "모든 추가 레코드 supports=context · causal inference disabled",
                        ],
                        backend="public_context_adapters",
                        evidence_ids=[
                            item.evidence_id
                            for target in routing.targets
                            for item in target.evidence
                        ],
                        artifact_ids=[packet.packet_id],
                        details={
                            "source_record_counts": source_counts,
                            "requested_source_families": requested_families,
                            "executed_source_families": executed_families,
                            "unexecuted_source_families": unexecuted_families,
                            "contract_satisfied": not unexecuted_families,
                            "research_depth": evidence_budget.depth,
                            "budget": budget.model_dump(mode="json"),
                            "stop_reason": routing.stop_reason,
                            "target_cell_type": brief.constraints.target_cell_type,
                            "human_priorities": list(brief.human_priorities),
                            "source_decisions": [
                                {
                                    "target": target.target.symbol,
                                    "source": outcome.source,
                                    "status": outcome.status,
                                    "queried": outcome.queried,
                                    "reason": outcome.detail,
                                }
                                for target in routing.targets
                                for outcome in target.sources
                            ],
                        },
                    )
                },
            )
            return enriched
        except Exception as exc:
            safe = packet.model_copy(deep=True)
            safe.metadata = {
                **safe.metadata,
                "context_routing": {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "requested_source_families": requested_families,
                    "executed_source_families": [],
                    "unexecuted_source_families": requested_families,
                    "contract_satisfied": not requested_families,
                    "target_cell_context_kind": cell_context_kind,
                    "context_only": True,
                },
            }
            await self._trace_event(
                run_id,
                "context_routing_failed",
                "MA1 context lookup failed; primary evidence workflow continues.",
                node="ma1.context_router",
                metadata={
                    "interaction": interaction(
                        "MA1 · Context Router",
                        "Orchestrator",
                        "source_failure",
                        "맥락 전용 조회 실패를 주 근거 실패와 분리하고 기존 패킷을 보존",
                        basis=[f"오류 유형 {type(exc).__name__}"],
                        backend="public_context_adapters",
                        artifact_ids=[packet.packet_id],
                    )
                },
            )
            return safe

    @staticmethod
    def _reconcile_ma1_narrative(
        immutable: HypothesisPacket,
        proposed: HypothesisNarrativeOverlayOutput,
        *,
        attestation_model: str | None = None,
        attestation_component: str = "MA1",
        researcher_operational_requirements: tuple[str, ...] = (),
    ) -> HypothesisPacket:
        """Merge an exact narrative overlay while freezing all authority fields.

        Missing, unknown, or duplicate hypothesis/claim identifiers are an
        invalid live response, not a reason to silently accept a fallback.  A
        live model may also receive a bounded narrative-only change attestation;
        deterministic and mock calls omit it and preserve metadata byte-for-byte.
        """

        proposed_ids = [item.hypothesis_id for item in proposed.hypotheses]
        if len(proposed_ids) != len(set(proposed_ids)):
            raise ValueError("MA1 narrative overlay contains duplicate hypothesis IDs")
        expected_ids = [item.hypothesis_id for item in immutable.hypotheses]
        proposed_id_set = set(proposed_ids)
        expected_id_set = set(expected_ids)
        if proposed_id_set != expected_id_set:
            missing = sorted(expected_id_set - proposed_id_set)
            unknown = sorted(proposed_id_set - expected_id_set)
            raise ValueError(
                "MA1 narrative hypothesis ID mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        proposed_by_id = {item.hypothesis_id: item for item in proposed.hypotheses}

        reconciled: list[Hypothesis] = []
        changed_fields: list[dict[str, object]] = []
        for base in immutable.hypotheses:
            candidate = proposed_by_id[base.hypothesis_id]
            falsification_test = _ensure_ma1_operational_requirements(
                candidate.falsification_test,
                researcher_operational_requirements,
            )
            proposed_claim_ids = [item.claim_id for item in candidate.claims]
            if len(proposed_claim_ids) != len(set(proposed_claim_ids)):
                raise ValueError(
                    "MA1 narrative overlay contains duplicate claim IDs for "
                    f"{base.hypothesis_id}"
                )
            expected_claim_ids = [item.claim_id for item in base.claims]
            proposed_claim_set = set(proposed_claim_ids)
            expected_claim_set = set(expected_claim_ids)
            if proposed_claim_set != expected_claim_set:
                missing = sorted(expected_claim_set - proposed_claim_set)
                unknown = sorted(proposed_claim_set - expected_claim_set)
                raise ValueError(
                    f"MA1 narrative claim ID mismatch for {base.hypothesis_id}: "
                    f"missing={missing}, unknown={unknown}"
                )
            claims_by_id = {item.claim_id: item for item in candidate.claims}
            for base_claim in base.claims:
                proposed_text = claims_by_id[base_claim.claim_id].text
                if proposed_text != base_claim.text:
                    changed_fields.append(
                        _ma1_narrative_change(
                            hypothesis_id=base.hypothesis_id,
                            claim_id=base_claim.claim_id,
                            field="claim.text",
                            before=base_claim.text,
                            after=proposed_text,
                        )
                    )
            claims = [
                base_claim.model_copy(
                    update={"text": claims_by_id[base_claim.claim_id].text},
                    deep=True,
                )
                for base_claim in base.claims
            ]
            if candidate.mechanism != base.mechanism:
                changed_fields.append(
                    _ma1_narrative_change(
                        hypothesis_id=base.hypothesis_id,
                        field="mechanism",
                        before=base.mechanism,
                        after=candidate.mechanism,
                    )
                )
            if falsification_test != base.falsification_test:
                changed_fields.append(
                    _ma1_narrative_change(
                        hypothesis_id=base.hypothesis_id,
                        field="falsification_test",
                        before=base.falsification_test,
                        after=falsification_test,
                    )
                )
            reconciled.append(
                base.model_copy(
                    update={
                        "mechanism": candidate.mechanism,
                        "falsification_test": falsification_test,
                        "claims": claims,
                    },
                    deep=True,
                )
            )
        updates: dict[str, object] = {"hypotheses": reconciled}
        if attestation_model is not None:
            updates["metadata"] = {
                **immutable.metadata,
                "llm_narrative_attestation": {
                    "schema_version": "1.0",
                    "component": attestation_component,
                    "model": attestation_model,
                    "authority_scope": "narrative_only",
                    "changed_field_count": len(changed_fields),
                    "changed_fields": changed_fields,
                    "frozen_authority_fields": list(_MA1_FROZEN_AUTHORITY_FIELDS),
                },
            }
        return immutable.model_copy(update=updates, deep=True)

    async def _llm_enhance(
        self,
        run_id: str,
        *,
        component: str,
        model: str,
        response_model,
        fallback,
        prompt: str,
        max_output_tokens: int,
        failure_is_fatal: bool = True,
        postprocess: Callable[[Any], Any] | None = None,
        postprocess_failure_is_fatal: bool = False,
    ):
        await self._cancellation_checkpoint(run_id)
        if self.settings.effective_mode is RunMode.MOCK:
            return postprocess(fallback) if postprocess is not None else fallback
        provider = self.providers.for_component(component)
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        system = (
            "You are a bounded DrugPilot worker. Follow the supplied JSON schema, "
            "preserve provenance, and never invent a tool result."
        )
        conservative_input_tokens = conservative_structured_input_tokens(
            system=system,
            prompt=prompt,
            response_model=response_model,
        )
        BudgetGuard(snapshot.brief.budget).preflight(
            snapshot.token_usage,
            model=model,
            input_tokens=conservative_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        await self._trace_event(
            run_id,
            "llm_call_started",
            f"{component}가 {model} 구조화 출력을 요청했습니다.",
            node=f"{component.lower()}.llm",
            metadata={
                "interaction": interaction(
                    component,
                    f"LLM · {model}",
                    "structured_generation_request",
                    "검증된 입력과 고정 JSON 스키마를 사용한 구조화 결과 생성 요청",
                    basis=[
                        "도구 계산값과 외부 식별자 임의 생성 금지",
                        f"최대 출력 {max_output_tokens:,} tokens",
                        "원문 프롬프트와 비공개 내부 추론은 실행 로그에 저장하지 않음",
                    ],
                    backend="openai_api",
                    model=model,
                )
            },
        )
        postprocess_failed = False
        try:
            result = await provider.structured(
                component=component,
                model=model,
                system=system,
                prompt=prompt,
                response_model=response_model,
                fallback=fallback,
                max_output_tokens=max_output_tokens,
            )
            try:
                data = (
                    postprocess(result.data)
                    if postprocess is not None
                    else result.data
                )
            except Exception as exc:
                postprocess_failed = True
                raise LLMProviderCallError(
                    "LLM output failed the domain authority boundary",
                    usage=result.usage,
                    reason_code="DOMAIN_AUTHORITY_REJECTED",
                ) from exc
        except Exception as exc:
            failed_usage = usage_for_failed_call(
                exc,
                component=component,
                model=model,
                conservative_input_tokens=conservative_input_tokens,
                conservative_max_output_tokens=max_output_tokens,
            )
            if failed_usage is not None:
                # Record and price the attempted/paid call before either the
                # deterministic fallback or fatal workflow path.  Missing
                # exact provider usage becomes a conservative nonzero bound;
                # BudgetGuard then stops the run after durable persistence.
                await self._record_llm_usage(run_id, failed_usage)
            # An explicitly configured hybrid run may fail open for diagnostics.
            # Key-only auto activation is strict: an invalid key, quota error or
            # provider outage must be visible instead of being reported as a
            # successful OpenAI-backed run that actually used fixture output.
            fallback_allowed = not (
                postprocess_failed and postprocess_failure_is_fatal
            ) and (
                not failure_is_fatal
                or (
                    self.settings.effective_mode is RunMode.HYBRID
                    and not self.settings.openai_auto_activated
                )
            )
            if fallback_allowed:
                await self._trace_event(
                    run_id,
                    "llm_fallback",
                    f"{component} LLM 호출 실패로 권한이 제한된 로컬 결과를 사용했습니다.",
                    node=f"{component.lower()}.llm",
                    metadata={
                        "reason_code": exc.reason_code if isinstance(exc, LLMProviderCallError) else "LOCAL_PROCESSING_FAILURE",
                        "error_type": type(exc).__name__,
                        "interaction": interaction(
                            f"LLM · {model}",
                            component,
                            "fallback",
                            (
                                "선택적 서술 보조 실패는 외부 검증 판정을 변경하지 않고 "
                                "결정론적 산출물로 복구"
                                if not failure_is_fatal
                                else "명시적 HYBRID 정책에 따라 검증된 결정론적 산출물로 복구"
                            ),
                            basis=[f"오류 유형 {type(exc).__name__}"],
                            backend="deterministic_fallback",
                            model=model,
                        )
                    },
                )
                return (
                    postprocess(fallback)
                    if postprocess is not None
                    else fallback
                )
            await self._trace_event(
                run_id,
                "llm_call_failed",
                f"{component} LLM 호출이 실패해 워크플로우를 중단합니다.",
                node=f"{component.lower()}.llm",
                metadata={
                    "reason_code": exc.reason_code if isinstance(exc, LLMProviderCallError) else "LOCAL_PROCESSING_FAILURE",
                    "error_type": type(exc).__name__,
                    "interaction": interaction(
                        f"LLM · {model}",
                        "Orchestrator",
                        "error",
                        "키 자동 활성화 실행에서는 실패를 숨기지 않고 종료 정책으로 전달",
                        basis=[f"오류 유형 {type(exc).__name__}"],
                        backend="openai_api",
                        model=model,
                    )
                },
            )
            raise
        await self._record_llm_usage(run_id, result.usage)
        await self._trace_event(
            run_id,
            "llm_call_completed",
            f"{component}가 {model} 구조화 결과를 수신했습니다.",
            node=f"{component.lower()}.llm",
            metadata={
                "interaction": interaction(
                    f"LLM · {model}",
                    component,
                    "structured_generation_response",
                    "스키마 검증을 통과한 구조화 결과를 후속 에이전트 산출물로 채택",
                    basis=[
                        f"입력 {result.usage.input_tokens:,} tokens",
                        f"출력 {result.usage.output_tokens:,} tokens",
                        f"추론 계수 {result.usage.reasoning_tokens:,} tokens",
                    ],
                    backend="openai_api",
                    model=model,
                )
            },
        )
        return data

    def _component_backend(self, component: str) -> str:
        if self.settings.effective_mode is RunMode.MOCK:
            return "deterministic_local"
        return "openai_api" if component in {"MA1", "Planner"} else "local_tools"

    async def _trace_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        *,
        node: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._cancellation_checkpoint(run_id)
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        await self._emit(
            snapshot,
            snapshot.stage,
            event_type,
            message,
            node=node,
            metadata=metadata,
        )
        await self.store.save_snapshot(snapshot)

    async def _record_llm_usage(self, run_id: str, usage: TokenUsage) -> None:
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        guard = BudgetGuard(snapshot.brief.budget)
        priced_usage = guard.price_usage(usage)
        # Persist actual provider usage before enforcing the postflight cap so
        # a paid call is never omitted from the accounting record.
        snapshot = await self.store.append_token_usage(run_id, priced_usage)
        guard.check(snapshot.token_usage)

    async def _preflight_llm(
        self,
        run_id: str,
        model: str,
        prompt: str,
        max_output_tokens: int,
    ) -> None:
        await self._cancellation_checkpoint(run_id)
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        BudgetGuard(snapshot.brief.budget).preflight(
            snapshot.token_usage,
            model=model,
            input_tokens=conservative_structured_input_tokens(
                system=MA3_REVIEWER_SYSTEM_PROMPT,
                prompt=prompt,
                response_model=ReviewerRoutingAssessmentOutput,
            ),
            max_output_tokens=max_output_tokens,
        )

    async def _record_rework(self, run_id: str, order: ReworkOrder) -> None:
        await self._transition(
            run_id,
            RunStage.G0 if order.gate.value == "G0" else RunStage.G1,
            f"Scoped rework ordered from {order.failed_node}.",
            "rework_order",
            node=order.failed_node,
            metadata={
                "target_stage": order.target_stage,
                "invalidated_nodes": self.dependency.invalidated_nodes(order),
                "reused_nodes": self.dependency.reusable_nodes(order),
                "interaction": interaction(
                    f"MA3 · {order.gate.value}",
                    order.target_stage.upper(),
                    "selective_rework",
                    f"{order.failed_node}에서 발견된 결함만 선택적으로 재작업 요청",
                    basis=order.required_actions,
                    decision=(
                        "REVISE" if order.target_stage in {"ma1", "ma2"} else "REPLAN"
                    ),
                    artifact_ids=order.preserve_artifact_ids,
                    details={
                        "invalidated_nodes": self.dependency.invalidated_nodes(order),
                        "reused_nodes": self.dependency.reusable_nodes(order),
                        "affected_claim_ids": order.affected_claim_ids,
                        "affected_candidate_ids": order.affected_candidate_ids,
                    },
                ),
            },
        )

    async def _artifact_transition(
        self,
        run_id: str,
        stage: RunStage,
        message: str,
        event_type: str,
        node: str,
        artifact_id: str,
        artifact_type: str,
        artifact: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        field_map = {
            "HypothesisPacket": "hypothesis_packet",
            "PlanSpec": "plan_spec",
            "CandidatePacket": "candidate_packet",
        }
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        field = field_map.get(artifact_type)
        if field:
            setattr(snapshot, field, artifact)
        elif artifact_type == "GateVerdict":
            setattr(snapshot, "g0_verdict" if artifact.gate.value == "G0" else "g1_verdict", artifact)
        snapshot.stage = stage
        snapshot.status_message = message
        event = ExecutionEvent(run_id=run_id, stage=stage, event_type=event_type,
            message=message, node=node, artifact_ids=[artifact_id], metadata=metadata or {})
        artifact_digest = await self.store.commit_artifact_event(
            snapshot, artifact_id, artifact_type, artifact, event)
        event.metadata["artifact_sha256"] = {artifact_id: artifact_digest}
        await self.broker.publish(event)

    async def _transition(
        self,
        run_id: str,
        stage: RunStage,
        message: str,
        event_type: str,
        *,
        node: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        snapshot.stage = stage
        snapshot.status_message = message
        await self._emit(
            snapshot,
            stage,
            event_type,
            message,
            node=node,
            metadata=metadata,
        )
        await self.store.save_snapshot(snapshot)

    async def _terminal(
        self,
        run_id: str,
        stage: RunStage,
        message: str,
        error: str | None = None,
    ) -> None:
        try:
            await self._terminal_write(run_id, stage, message, error)
        except RunCancellationRequested:
            # Cancellation may race with terminal artifact/event persistence.
            # Its durable fence must win even inside an exception handler.
            await self._finish_cancellation(run_id)

    async def _terminal_write(
        self, run_id: str, stage: RunStage, message: str, error: str | None = None,
    ) -> None:
        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            return
        if snapshot.stage in {RunStage.CANCEL_REQUESTED, RunStage.CANCELLED}:
            await self._finish_cancellation(run_id)
            return
        trigger_stage = snapshot.stage
        if stage in {RunStage.ABSTAINED, RunStage.BLOCKED, RunStage.FAILED}:
            await self._materialize_next_action_packet(
                run_id,
                trigger_stage=trigger_stage,
                terminal_message=error or message,
            )
            snapshot = await self.store.get_snapshot(run_id)
            if snapshot is None:
                return
        snapshot.stage = stage
        snapshot.status_message = message
        snapshot.error = error
        await self._emit(
            snapshot,
            stage,
            "workflow_terminal",
            message,
            node=stage.value,
            metadata={"error": error} if error else None,
        )
        await self.store.save_snapshot(snapshot)
        await self.broker.close(run_id)

    async def _materialize_next_action_packet(
        self,
        run_id: str,
        *,
        trigger_stage: RunStage,
        terminal_message: str,
        _review_lease_token: str | None = None,
    ) -> NextActionPacket | None:
        if _review_lease_token is not None:
            return await self._materialize_next_action_packet_owned(
                run_id, trigger_stage=trigger_stage, terminal_message=terminal_message,
                _review_lease_token=_review_lease_token)
        # The user-cancel endpoint owns this same lock. Keep legacy multi-
        # artifact handoff writes intact before allowing its cancellation fence.
        async with self._task_registry_lock:
            return await self._materialize_next_action_packet_owned(
                run_id, trigger_stage=trigger_stage, terminal_message=terminal_message,
                _review_lease_token=self._workflow_lease_tokens.get(run_id))

    async def _materialize_next_action_packet_owned(
        self, run_id: str, *, trigger_stage: RunStage, terminal_message: str,
        _review_lease_token: str | None = None,
    ) -> NextActionPacket | None:
        """Persist one idempotent handoff without invoking a model or a fixture."""

        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        existing_packet = snapshot.next_action_packet
        if (
            existing_packet is not None
            and "classifier_version" in existing_packet.model_fields_set
            and existing_packet.classifier_version == "next-actions-r3"
        ):
            return snapshot.next_action_packet
        try:
            checkpoint = await self.store.latest_stage_checkpoint(run_id)
        except CheckpointError:
            checkpoint = None
        verdict = snapshot.g1_verdict or snapshot.g0_verdict
        packet = build_next_action_packet(
            run_id=run_id,
            trigger_stage=trigger_stage,
            brief=snapshot.brief,
            verdict=verdict,
            candidates=(
                snapshot.candidate_packet.candidates
                if snapshot.candidate_packet is not None
                else []
            ),
            checkpoint=checkpoint,
            terminal_message=terminal_message,
        )
        if packet is None:
            return None
        if existing_packet is not None:
            old_actions = {tuple(a.finding_ids): a for a in existing_packet.actions}
            for action in packet.actions:
                prior = old_actions.get(tuple(action.finding_ids))
                if prior and prior.kind == action.kind:
                    action.action_id = prior.action_id
            for request in packet.experiment_requests:
                prior = next((r for r in existing_packet.experiment_requests
                    if r.candidate_id == request.candidate_id and r.source_finding_ids == request.source_finding_ids), None)
                if prior:
                    request.request_id = prior.request_id
            packet.preserve_artifact_ids = list(
                dict.fromkeys(
                    [*packet.preserve_artifact_ids, existing_packet.packet_id]
                )
            )
        digest = sha256(packet.model_dump_json().encode()).hexdigest()
        artifacts = [(packet.packet_id, "NextActionPacket", packet)]
        snapshot.next_action_packet = packet
        if snapshot.next_action_progress:
            old = snapshot.next_action_progress
            progress = NextActionProgress(packet_id=packet.packet_id,
                action_status={a.action_id: old.action_status.get(a.action_id, "pending") for a in packet.actions},
                experiment_status={r.request_id: old.experiment_status.get(r.request_id, "awaiting_results") for r in packet.experiment_requests},
                submission_ids=list(old.submission_ids), resolution_ids=list(old.resolution_ids))
            progress.all_blocking_actions_completed = all(
                progress.action_status.get(a.action_id) == "completed" for a in packet.actions if a.blocking)
            progress.ready_for_reaudit = progress.all_blocking_actions_completed and all(
                progress.experiment_status.get(r.request_id) in {"submission_reviewed", "validated"}
                for r in packet.experiment_requests)
            snapshot.next_action_progress = progress
            artifacts.append((progress.progress_id, "NextActionProgress", progress))
        event = ExecutionEvent(
            run_id=run_id, stage=trigger_stage, event_type="next_action_packet_created",
            message=packet.summary,
            node="orchestrator.next_actions",
            artifact_ids=[i for i, _, _ in artifacts],
            metadata={
                "disposition": packet.disposition,
                "action_count": len(packet.actions),
                "experiment_request_count": len(packet.experiment_requests),
                "approval_allowed": packet.approval_allowed,
                "classifier_version": packet.classifier_version,
                "supersedes_packet_id": (
                    existing_packet.packet_id if existing_packet else None
                ),
                "artifact_sha256": {i: sha256(p.model_dump_json().encode()).hexdigest() for i, _, p in artifacts},
            },
        )
        snapshot.events.append(event)
        if _review_lease_token:
            await self.store.commit_review_update(snapshot, artifacts=artifacts,
                events=[event], lease_token=_review_lease_token)
        else:
            for i, kind, payload in artifacts:
                await self.store.save_artifact(run_id, i, kind, payload)
            await self.store.append_event(event)
            await self.store.save_snapshot(snapshot)
        await self.broker.publish(event)
        return packet

    @serialized_review_mutation
    async def materialize_next_actions(self, run_id: str, *, _review_lease_token: str | None = None) -> RunSnapshot:
        """Backfill the typed action handoff for an already stopped run."""

        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        active = self._tasks.get(run_id)
        if active is not None and not active.done():
            raise ValueError("next actions cannot be backfilled while the run is active")
        if snapshot.stage not in {
            RunStage.HUMAN_GATE,
            RunStage.ABSTAINED,
            RunStage.BLOCKED,
            RunStage.FAILED,
        }:
            raise ValueError("next actions are available only for a stopped or review-stage run")
        await self._materialize_next_action_packet(
            run_id,
            trigger_stage=(
                RunStage.G1
                if snapshot.g1_verdict is not None
                else RunStage.G0
                if snapshot.g0_verdict is not None
                else snapshot.stage
            ),
            terminal_message=snapshot.error or snapshot.status_message,
            _review_lease_token=_review_lease_token,
        )
        if (
            snapshot.stage is RunStage.ABSTAINED
            and snapshot.human_decision is None
            and snapshot.g1_verdict is not None
            and snapshot.g1_verdict.verdict is not GateVerdictCode.PASS
        ):
            await self._transition(
                run_id,
                RunStage.HUMAN_GATE,
                "Automated audit requires action; APPROVE is locked pending a fresh G1 PASS.",
                "action_required_human_gate",
                node="human_gate",
                metadata={"backfilled_from_stage": RunStage.ABSTAINED.value},
            )
        refreshed = await self.store.get_snapshot(run_id)
        if refreshed is None:
            raise KeyError(run_id)
        return refreshed

    @staticmethod
    def _required_evidence_type(action_kind: str) -> str:
        return {
            "computational_rework": "code_validation",
            "experiment_required": "experimental_result",
            "data_required": "source_document",
            "tool_configuration": "tool_readiness",
            "researcher_decision": "researcher_policy",
            "resource_required": "resource_approval",
            "integrity_block": "integrity_report",
            "operator_review": "researcher_policy",
        }[action_kind]

    @staticmethod
    def _action_progress(snapshot: RunSnapshot) -> NextActionProgress:
        packet = snapshot.next_action_packet
        if packet is None:
            raise ValueError("the run has no next-action packet")
        existing = snapshot.next_action_progress
        if existing is not None and existing.packet_id == packet.packet_id:
            return existing
        return NextActionProgress(
            packet_id=packet.packet_id,
            action_status={item.action_id: "pending" for item in packet.actions},
            experiment_status={
                item.request_id: "awaiting_results"
                for item in packet.experiment_requests
            },
        )

    @serialized_review_mutation
    async def submit_action_evidence(
        self,
        run_id: str,
        *,
        action_id: str,
        experiment_request_id: str | None,
        evidence_type: str,
        submitted_by: str,
        notes: str,
        source_filename: str,
        media_type: str,
        source_content_base64: str,
        observed_results: list[ExperimentalResultRecord],
        observed_not_synthetic_attestation: bool,
        _review_lease_token: str | None = None,
    ) -> RunSnapshot:
        """Attach a real source file and typed rows without interpreting results."""

        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        packet = snapshot.next_action_packet
        if packet is None:
            raise ValueError("the run has no next-action packet")
        if snapshot.stage is not RunStage.HUMAN_GATE or snapshot.human_decision is not None:
            raise ValueError("action evidence is accepted only at an open action-required Human Gate")
        action = next((item for item in packet.actions if item.action_id == action_id), None)
        if action is None:
            raise ValueError("unknown next-action id")
        expected_type = self._required_evidence_type(action.kind)
        if evidence_type != expected_type:
            raise ValueError(
                f"{action.kind} requires evidence_type={expected_type}"
            )
        linked_requests = [
            item
            for item in packet.experiment_requests
            if set(item.source_finding_ids) & set(action.finding_ids)
        ]
        if action.kind == "experiment_required":
            request = next(
                (
                    item
                    for item in linked_requests
                    if item.request_id == experiment_request_id
                ),
                None,
            )
            if request is None:
                raise ValueError("experimental evidence does not match this action request")
            if request.candidate_id and any(
                row.candidate_id != request.candidate_id for row in observed_results
            ):
                raise ValueError("observed rows must match the requested candidate")
        elif experiment_request_id is not None:
            raise ValueError("only experimental actions accept an experiment request id")
        try:
            raw_content = base64.b64decode(source_content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("source attachment is not valid base64") from exc
        if not raw_content:
            raise ValueError("source attachment cannot be empty")
        if len(raw_content) > 5_000_000:
            raise ValueError("source attachment exceeds the 5 MB limit")
        source_digest = sha256(raw_content).hexdigest()
        if any(
            row.source_artifact_sha256 != source_digest for row in observed_results
        ):
            raise ValueError(
                "observed row source hashes must equal the server-computed attachment hash"
            )
        safe_filename = source_filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not safe_filename:
            raise ValueError("source filename is required")
        duplicate = next(
            (
                item
                for item in snapshot.action_evidence_submissions
                if item.action_id == action_id
                and item.source_artifact_sha256 == source_digest
            ),
            None,
        )
        if duplicate is not None:
            old_attachment = await self.store.read_artifact(run_id, duplicate.attachment_id, "EvidenceAttachmentArtifact")
            if (duplicate.observed_results != observed_results
                or duplicate.experiment_request_id != experiment_request_id
                or duplicate.evidence_type != evidence_type
                or duplicate.submitted_by != submitted_by or duplicate.notes != notes
                or duplicate.source_filename != safe_filename
                or duplicate.observed_not_synthetic_attestation != observed_not_synthetic_attestation
                or old_attachment is None or old_attachment["media_type"] != media_type):
                raise ValueError("The same source file already has different submission data; review the existing submission before retrying")
            return snapshot
        attachment = EvidenceAttachmentArtifact(
            filename=safe_filename,
            media_type=media_type,
            byte_count=len(raw_content),
            sha256=source_digest,
            content_base64=source_content_base64,
        )
        submission = ActionEvidenceSubmission(
            run_id=run_id,
            packet_id=packet.packet_id,
            action_id=action_id,
            experiment_request_id=experiment_request_id,
            evidence_type=evidence_type,
            submitted_by=submitted_by,
            notes=notes,
            attachment_id=attachment.attachment_id,
            source_filename=safe_filename,
            source_artifact_sha256=source_digest,
            source_byte_count=len(raw_content),
            observed_results=observed_results,
            observed_not_synthetic_attestation=(
                observed_not_synthetic_attestation
            ),
        )
        attachment_digest = sha256(attachment.model_dump_json().encode()).hexdigest()
        submission_digest = sha256(submission.model_dump_json().encode()).hexdigest()
        current = self._action_progress(snapshot)
        action_status = dict(current.action_status)
        action_status[action_id] = "evidence_attached"
        experiment_status = dict(current.experiment_status)
        if experiment_request_id:
            experiment_status[experiment_request_id] = "results_attached"
        progress = NextActionProgress(
            packet_id=packet.packet_id,
            action_status=action_status,
            experiment_status=experiment_status,
            submission_ids=[*current.submission_ids, submission.submission_id],
            resolution_ids=list(current.resolution_ids),
        )
        progress_digest = sha256(progress.model_dump_json().encode()).hexdigest()
        snapshot.action_evidence_submissions.append(submission)
        snapshot.next_action_progress = progress
        event = ExecutionEvent(
            run_id=run_id, stage=RunStage.HUMAN_GATE, event_type="action_evidence_attached",
            message=f"Traceable evidence attached for action {action_id}; scientific review remains required.",
            node="human_gate.action_evidence",
            artifact_ids=[
                attachment.attachment_id,
                submission.submission_id,
                progress.progress_id,
            ],
            metadata={
                "action_id": action_id,
                "evidence_type": evidence_type,
                "source_artifact_sha256": source_digest,
                "artifact_sha256": {
                    attachment.attachment_id: attachment_digest,
                    submission.submission_id: submission_digest,
                    progress.progress_id: progress_digest,
                },
            },
        )
        snapshot.events.append(event)
        await self.store.commit_review_update(snapshot, artifacts=[
            (attachment.attachment_id, "EvidenceAttachmentArtifact", attachment),
            (submission.submission_id, "ActionEvidenceSubmission", submission),
            (progress.progress_id, "NextActionProgress", progress),
        ], events=[event], lease_token=_review_lease_token)
        await self.broker.publish(event)
        return snapshot

    @serialized_review_mutation
    async def resolve_next_action(
        self,
        run_id: str,
        *,
        action_id: str,
        supporting_submission_ids: list[str],
        resolved_by: str,
        rationale: str,
        _review_lease_token: str | None = None,
    ) -> RunSnapshot:
        """Record signed completion; never converts completion into approval."""

        snapshot = await self.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        packet = snapshot.next_action_packet
        if packet is None:
            raise ValueError("the run has no next-action packet")
        if snapshot.stage is not RunStage.HUMAN_GATE or snapshot.human_decision is not None:
            raise ValueError("actions can be resolved only at an open Human Gate")
        action = next((item for item in packet.actions if item.action_id == action_id), None)
        if action is None:
            raise ValueError("unknown next-action id")
        current = self._action_progress(snapshot)
        if current.action_status.get(action_id) == "completed":
            return snapshot
        supporting = [
            item
            for item in snapshot.action_evidence_submissions
            if item.submission_id in set(supporting_submission_ids)
        ]
        if len(supporting) != len(set(supporting_submission_ids)) or not supporting:
            raise ValueError("every supporting submission must exist in this run")
        if any(item.action_id != action_id for item in supporting):
            raise ValueError("supporting submissions must belong to the resolved action")
        expected_type = self._required_evidence_type(action.kind)
        if any(item.evidence_type != expected_type for item in supporting):
            raise ValueError("supporting evidence has the wrong type for this action")
        linked_request_ids = {
            item.request_id
            for item in packet.experiment_requests
            if set(item.source_finding_ids) & set(action.finding_ids)
        }
        if action.kind == "experiment_required" and not any(
            item.experiment_request_id in linked_request_ids
            and item.observed_results
            and item.observed_not_synthetic_attestation
            for item in supporting
        ):
            raise ValueError("experimental action requires matching observed result rows")
        resolution = ActionResolution(
            run_id=run_id,
            packet_id=packet.packet_id,
            action_id=action_id,
            supporting_submission_ids=list(dict.fromkeys(supporting_submission_ids)),
            resolved_by=resolved_by,
            rationale=rationale,
        )
        resolution_digest = sha256(resolution.model_dump_json().encode()).hexdigest()
        action_status = dict(current.action_status)
        action_status[action_id] = "completed"
        experiment_status = dict(current.experiment_status)
        for request_id in linked_request_ids:
            if any(item.experiment_request_id == request_id for item in supporting):
                experiment_status[request_id] = "submission_reviewed"
        blocking_ids = {item.action_id for item in packet.actions if item.blocking}
        all_completed = all(
            action_status.get(item_id) == "completed" for item_id in blocking_ids
        )
        experiments_reviewed = all(
            experiment_status.get(item.request_id) in {"submission_reviewed", "validated"}
            for item in packet.experiment_requests
        )
        progress = NextActionProgress(
            packet_id=packet.packet_id,
            action_status=action_status,
            experiment_status=experiment_status,
            submission_ids=list(current.submission_ids),
            resolution_ids=[*current.resolution_ids, resolution.resolution_id],
            all_blocking_actions_completed=all_completed,
            ready_for_reaudit=all_completed and experiments_reviewed,
            approval_allowed=False,
            updated_at=utc_now(),
        )
        progress_digest = sha256(progress.model_dump_json().encode()).hexdigest()
        snapshot.action_resolutions.append(resolution)
        snapshot.next_action_progress = progress
        snapshot.status_message = (
            "All required actions are complete; a fresh G1 re-audit is required before approval."
            if progress.ready_for_reaudit
            else "Action completion recorded; unresolved actions remain."
        )
        event = ExecutionEvent(
            run_id=run_id, stage=RunStage.HUMAN_GATE, event_type="next_action_resolved",
            message=snapshot.status_message,
            node="human_gate.action_resolution",
            artifact_ids=[resolution.resolution_id, progress.progress_id],
            metadata={
                "action_id": action_id,
                "ready_for_reaudit": progress.ready_for_reaudit,
                "approval_allowed": False,
                "artifact_sha256": {
                    resolution.resolution_id: resolution_digest,
                    progress.progress_id: progress_digest,
                },
            },
        )
        snapshot.events.append(event)
        await self.store.commit_review_update(snapshot, artifacts=[
            (resolution.resolution_id, "ActionResolution", resolution),
            (progress.progress_id, "NextActionProgress", progress),
        ], events=[event], lease_token=_review_lease_token)
        await self.broker.publish(event)
        return snapshot

    async def _emit(
        self,
        snapshot: RunSnapshot,
        stage: RunStage,
        event_type: str,
        message: str,
        *,
        node: str | None = None,
        artifact_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = ExecutionEvent(
            run_id=snapshot.run_id,
            stage=stage,
            event_type=event_type,
            message=message,
            node=node,
            artifact_ids=artifact_ids or [],
            metadata=metadata or {},
        )
        snapshot.events.append(event)
        await self.store.append_event(event)
        await self.broker.publish(event)
