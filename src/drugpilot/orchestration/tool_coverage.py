"""Deterministic broad-coverage planning with a non-authoritative LLM overlay.

The source registry, ResearchBrief allow-list, applicability rules, installed
configuration, and runtime feature flags are the authority.  A model may add
priority and query-focus prose for known sources, but cannot admit a source,
grant an operation, or change a call/record/token budget.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Literal, Sequence

from pydantic import Field

from drugpilot.config import EvidenceResearchBudget, Settings, get_settings
from drugpilot.domain.models import (
    HypothesisPacket,
    HypothesisTrack,
    ResearchBrief,
    SourceCoverageBudget,
    SourceCoverageDecision,
    SourceCoverageManifest,
    SourceCoverageSuggestionRejection,
    StrictModel,
)
from drugpilot.domain.cell_context import classify_cell_context
from drugpilot.tools.source_catalog import (
    RUNTIME_SOURCES,
    SourceConfiguration,
    source_status_inventory,
)


class SourceCoverageLLMSuggestion(StrictModel):
    """Narrative-only recommendation for one deterministic registry entry."""

    source_name: str = Field(min_length=1, max_length=120)
    recommend_use: bool = True
    priority: Literal["high", "medium", "low"] = "medium"
    rationale: str = Field(min_length=2, max_length=1_200)
    query_focus: str | None = Field(default=None, max_length=800)


_SOURCE_FAMILIES: dict[str, str] = {
    "pubmed": "literature",
    "europe_pmc": "literature",
    "biorxiv": "literature",
    "medrxiv": "literature",
    "uniprot": "target_identity",
    "open_targets": "target_identity",
    "pharos": "target_identity",
    "dgidb": "drug_interaction",
    "clinical_trials": "clinical",
    "gtex": "expression",
    "archs4": "expression",
    "string": "network",
    "biogrid": "network",
    "reactome": "pathway",
    "kegg": "pathway",
    "quickgo": "ontology",
    "cellxgene": "single_cell",
    "human_cell_atlas": "single_cell",
    "pubchem": "chemical_identity",
    "chembl": "measured_activity",
    "bindingdb": "measured_activity",
    "drugcentral": "measured_activity",
    "rcsb_pdb": "structure",
    "alphafold_db": "structure",
    "drugbank": "regulatory_safety",
    "google_patents": "prior_art",
}

_CORE_MA1 = frozenset(
    {"pubmed", "europe_pmc", "uniprot", "open_targets", "dgidb"}
)
_PREPRINT = frozenset({"biorxiv", "medrxiv"})
_CONTEXT = frozenset(
    {
        "gtex",
        "archs4",
        "string",
        "biogrid",
        "reactome",
        "kegg",
        "quickgo",
        "clinical_trials",
        "cellxgene",
        "human_cell_atlas",
    }
)
_SINGLE_CELL = frozenset({"cellxgene", "human_cell_atlas"})
_TBIO_CONTEXT = frozenset(
    {"gtex", "archs4", "string", "biogrid", "reactome", "kegg", "quickgo"}
)
_OPTIONAL_MA2_ENRICHMENT = frozenset(
    {"bindingdb", "pubchem", "rcsb_pdb", "alphafold_db", "google_patents"}
)


def build_broad_source_coverage(
    brief: ResearchBrief,
    packet: HypothesisPacket,
    settings: Settings,
) -> SourceCoverageManifest:
    """Admit every applicable and runnable source within fixed policies.

    This is a unique-source coverage plan, not a promise that every source will
    return a record.  Route activity and the existing per-stage call/time/record
    guards remain authoritative at execution time.
    """

    evidence_budget = settings.evidence_budget_for(brief.research_depth)
    inventory = source_status_inventory(_source_configuration(settings))
    status_by_id = {
        str(item["source_id"]): item
        for item in inventory.get("runtime", [])
        if isinstance(item, dict) and item.get("source_id")
    }
    allowed = {value.strip().casefold() for value in brief.data_policy.allowed_sources}
    decisions: list[SourceCoverageDecision] = []

    for source in RUNTIME_SOURCES:
        status = status_by_id.get(source.source_id, {})
        allowed_by_brief = (
            source.name.casefold() in allowed or source.source_id.casefold() in allowed
        )
        applicable, applicability_reason, execution_condition = _applicability(
            source.source_id,
            brief,
            packet,
        )
        configured = bool(status.get("configured", False))
        adapter = str(status["adapter"]) if status.get("adapter") else None
        enabled_by_runtime_policy = _runtime_policy_enabled(
            source.source_id,
            settings,
            evidence_budget,
        )

        if not allowed_by_brief:
            decision_status = "policy_excluded"
            reason_code = "NOT_IN_RESEARCH_BRIEF_ALLOW_LIST"
            reason = "ResearchBrief.data_policy.allowed_sources does not admit this source."
        elif not applicable:
            decision_status = "not_applicable"
            reason_code = "SOURCE_NOT_APPLICABLE_TO_RESEARCH_CONTEXT"
            reason = applicability_reason
        elif adapter is None:
            decision_status = "adapter_unavailable"
            reason_code = "SOURCE_ADAPTER_UNAVAILABLE"
            reason = "The deterministic source registry has no executable adapter."
        elif not configured:
            decision_status = "not_configured"
            reason_code = "SOURCE_NOT_CONFIGURED"
            reason = str(
                status.get("requirement")
                or status.get("configuration_state")
                or "The source requires operator configuration."
            )
        elif not enabled_by_runtime_policy:
            decision_status = "policy_excluded"
            reason_code = "RUNTIME_SOURCE_POLICY_DISABLED"
            reason = "The installed adapter is disabled by the immutable runtime policy."
        else:
            decision_status = "admitted"
            reason_code = "BROAD_COVERAGE_ADMITTED"
            reason = applicability_reason

        decisions.append(
            SourceCoverageDecision(
                source_id=source.source_id,
                source_name=source.name,
                stages=list(source.stages),
                family=_SOURCE_FAMILIES[source.source_id],
                status=decision_status,
                admitted=decision_status == "admitted",
                allowed_by_brief=allowed_by_brief,
                applicable=applicable,
                configured=configured,
                enabled_by_runtime_policy=enabled_by_runtime_policy,
                adapter=adapter,
                reason_code=reason_code,
                reason=reason,
                execution_condition=execution_condition,
            )
        )

    eligible = [
        item
        for item in decisions
        if item.allowed_by_brief
        and item.applicable
        and item.configured
        and item.enabled_by_runtime_policy
        and item.adapter is not None
    ]
    admitted = [item for item in decisions if item.admitted]
    return SourceCoverageManifest(
        registry_digest=source_registry_digest(),
        decisions=decisions,
        registry_source_count=len(decisions),
        allowed_source_count=sum(item.allowed_by_brief for item in decisions),
        applicable_source_count=sum(item.applicable for item in decisions),
        eligible_source_count=len(eligible),
        admitted_source_count=len(admitted),
        eligible_coverage_ratio=(len(admitted) / len(eligible) if eligible else 1.0),
        budget=SourceCoverageBudget(
            research_depth=evidence_budget.depth,
            max_run_tokens=min(brief.budget.max_run_tokens, settings.max_run_tokens),
            max_llm_calls=min(brief.budget.max_llm_calls, settings.max_llm_calls),
            max_run_cost_usd=min(
                brief.budget.max_run_cost_usd,
                settings.max_run_cost_usd,
            ),
            max_core_source_calls_per_target=(
                evidence_budget.core_source_calls_per_target
            ),
            max_core_records_per_source=evidence_budget.core_records_per_source,
            max_core_elapsed_seconds_per_target=(
                evidence_budget.core_elapsed_seconds_per_target
            ),
            max_core_attempts=evidence_budget.core_max_attempts,
            max_context_source_calls=(
                evidence_budget.context_source_calls
                if settings.dynamic_context_enabled
                else 0
            ),
            max_context_records_per_source=(
                evidence_budget.context_records_per_source
                if settings.dynamic_context_enabled
                else 0
            ),
            max_context_elapsed_seconds=(
                evidence_budget.context_elapsed_seconds
                if settings.dynamic_context_enabled
                else 0
            ),
            max_context_targets_per_track=(
                evidence_budget.context_targets_per_track
                if settings.dynamic_context_enabled
                else 0
            ),
            target_discovery_pubmed_records=evidence_budget.target_pubmed_records,
            target_discovery_pubmed_supplements=(
                evidence_budget.target_pubmed_supplements
            ),
            target_discovery_lookup_limit=evidence_budget.target_lookup_limit,
            target_discovery_max_candidates=evidence_budget.target_max_candidates,
            pharos_targets_per_track=evidence_budget.pharos_targets_per_track,
            candidate_enrichment_per_route=(
                evidence_budget.candidate_enrichment_per_route
                if settings.public_candidate_enrichment_enabled
                else 0
            ),
        ),
    )


def apply_llm_source_suggestions(
    manifest: SourceCoverageManifest,
    suggestions: Sequence[SourceCoverageLLMSuggestion],
) -> SourceCoverageManifest:
    """Attach bounded model prose without changing deterministic admission."""

    decisions = [item.model_copy(deep=True) for item in manifest.decisions]
    by_name: dict[str, SourceCoverageDecision] = {}
    for item in decisions:
        by_name[item.source_name.casefold()] = item
        by_name[item.source_id.casefold()] = item

    rejections: list[SourceCoverageSuggestionRejection] = []
    seen: set[str] = set()
    applied = False
    for suggestion in suggestions:
        key = suggestion.source_name.strip().casefold()
        decision = by_name.get(key)
        if decision is None:
            unknown_key = f"unknown:{key}"
            if unknown_key in seen:
                rejections.append(
                    SourceCoverageSuggestionRejection(
                        source_name=suggestion.source_name,
                        reason_code="DUPLICATE_LLM_SOURCE_SUGGESTION",
                        reason="Only the first suggestion for a source label is retained.",
                    )
                )
                continue
            seen.add(unknown_key)
            rejections.append(
                SourceCoverageSuggestionRejection(
                    source_name=suggestion.source_name,
                    reason_code="UNKNOWN_SOURCE_SUGGESTION",
                    reason="The model suggestion is outside the deterministic source registry.",
                )
            )
            continue
        canonical_key = decision.source_id.casefold()
        if canonical_key in seen:
            rejections.append(
                SourceCoverageSuggestionRejection(
                    source_name=suggestion.source_name,
                    reason_code="DUPLICATE_LLM_SOURCE_SUGGESTION",
                    reason="Only the first suggestion for a registered source is retained.",
                )
            )
            continue
        seen.add(canonical_key)

        applied = True
        recommendation = "recommended" if suggestion.recommend_use else "not_recommended"
        guard_note: str | None = None
        if suggestion.recommend_use and not decision.admitted:
            guard_note = (
                "Recommendation rejected for dispatch; deterministic decision remains "
                f"{decision.status}/{decision.reason_code}."
            )
            rejections.append(
                SourceCoverageSuggestionRejection(
                    source_name=decision.source_name,
                    reason_code=decision.reason_code,
                    reason=guard_note,
                )
            )
        elif not suggestion.recommend_use and decision.admitted:
            guard_note = (
                "Broad-coverage policy retained this eligible source; the model cannot "
                "remove deterministic coverage."
            )

        updated = decision.model_copy(
            update={
                "llm_recommendation": recommendation,
                "llm_priority": suggestion.priority,
                "llm_rationale": _bounded(suggestion.rationale, 1_200),
                "llm_query_focus": _bounded(suggestion.query_focus, 800),
                "llm_guard_note": guard_note,
            },
            deep=True,
        )
        for index, current in enumerate(decisions):
            if current.source_id == decision.source_id:
                decisions[index] = updated
                break
        by_name[decision.source_name.casefold()] = updated
        by_name[decision.source_id.casefold()] = updated

    return manifest.model_copy(
        update={
            "decisions": decisions,
            "llm_overlay_applied": applied,
            "llm_rejected_suggestions": rejections[:50],
        },
        deep=True,
    )


def source_coverage_execution_contract(
    manifest: SourceCoverageManifest | None,
) -> dict[str, object] | None:
    """Projection of coverage fields that an LLM must never change."""

    if manifest is None:
        return None
    return {
        "strategy": manifest.strategy,
        "registry_digest": manifest.registry_digest,
        "registry_source_count": manifest.registry_source_count,
        "allowed_source_count": manifest.allowed_source_count,
        "applicable_source_count": manifest.applicable_source_count,
        "eligible_source_count": manifest.eligible_source_count,
        "admitted_source_count": manifest.admitted_source_count,
        "eligible_coverage_ratio": manifest.eligible_coverage_ratio,
        "budget": manifest.budget.model_dump(mode="json"),
        "decisions": [
            {
                "source_id": item.source_id,
                "source_name": item.source_name,
                "stages": list(item.stages),
                "family": item.family,
                "status": item.status,
                "admitted": item.admitted,
                "allowed_by_brief": item.allowed_by_brief,
                "applicable": item.applicable,
                "configured": item.configured,
                "enabled_by_runtime_policy": item.enabled_by_runtime_policy,
                "adapter": item.adapter,
                "reason_code": item.reason_code,
                "execution_condition": item.execution_condition,
            }
            for item in manifest.decisions
        ],
    }


def source_registry_digest() -> str:
    payload = [
        {
            "source_id": item.source_id,
            "name": item.name,
            "stages": list(item.stages),
            "access": item.access,
            "endpoint": item.endpoint,
        }
        for item in RUNTIME_SOURCES
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_source_coverage_manifest(
    brief: ResearchBrief,
    manifest: SourceCoverageManifest | None,
    settings: Settings | None = None,
) -> list[tuple[str, str]]:
    """Independently verify manifest integrity without trusting model prose."""

    if manifest is None:
        return [
            (
                "SOURCE_COVERAGE_MANIFEST_MISSING",
                "PlanSpec has no deterministic source-coverage manifest.",
            )
        ]

    # Legacy callers validate registry/policy integrity without knowing which
    # operator ceilings built the manifest.  MA3 always supplies Settings and
    # therefore performs the stricter exact-profile recomputation.
    exact_profile_validation = settings is not None
    settings = settings or get_settings()
    issues: list[tuple[str, str]] = []
    expected_by_id = {item.source_id: item for item in RUNTIME_SOURCES}
    decisions_by_id = {item.source_id: item for item in manifest.decisions}
    decision_ids = [item.source_id for item in manifest.decisions]
    if len(decision_ids) != len(set(decision_ids)):
        issues.append(
            (
                "SOURCE_COVERAGE_MANIFEST_INVALID",
                "Source coverage contains duplicate registry identifiers.",
            )
        )
    if set(decisions_by_id) != set(expected_by_id):
        issues.append(
            (
                "SOURCE_COVERAGE_REGISTRY_MISMATCH",
                "Source coverage does not exactly match the deterministic runtime registry.",
            )
        )
    if manifest.registry_digest != source_registry_digest():
        issues.append(
            (
                "SOURCE_COVERAGE_REGISTRY_MISMATCH",
                "Source coverage registry digest does not match the installed registry.",
            )
        )
    for source_id, decision in decisions_by_id.items():
        expected = expected_by_id.get(source_id)
        if expected is None:
            continue
        if decision.source_name != expected.name or decision.stages != list(expected.stages):
            issues.append(
                (
                    "SOURCE_COVERAGE_REGISTRY_MISMATCH",
                    f"Coverage identity for {source_id} differs from the installed registry.",
                )
            )

    allowed = {value.strip().casefold() for value in brief.data_policy.allowed_sources}
    for item in manifest.decisions:
        expected_allowed = (
            item.source_name.casefold() in allowed or item.source_id.casefold() in allowed
        )
        if item.allowed_by_brief != expected_allowed:
            issues.append(
                (
                    "SOURCE_COVERAGE_POLICY_MISMATCH",
                    f"Allow-list decision for {item.source_name} does not match ResearchBrief.",
                )
            )
        eligible = (
            item.allowed_by_brief
            and item.applicable
            and item.configured
            and item.enabled_by_runtime_policy
            and item.adapter is not None
        )
        if eligible != item.admitted:
            issues.append(
                (
                    "SOURCE_COVERAGE_GAP",
                    (
                        f"Broad coverage must admit every eligible source; {item.source_name} "
                        f"has eligible={eligible}, admitted={item.admitted}."
                    ),
                )
            )
        if item.llm_recommendation == "recommended" and not item.admitted:
            if not item.llm_guard_note:
                issues.append(
                    (
                        "UNGUARDED_LLM_SOURCE_SUGGESTION",
                        f"Rejected LLM suggestion for {item.source_name} has no guard record.",
                    )
                )
        if item.llm_recommendation == "not_recommended" and item.admitted:
            if not item.llm_guard_note:
                issues.append(
                    (
                        "UNGUARDED_LLM_SOURCE_SUGGESTION",
                        f"LLM de-prioritization for {item.source_name} has no non-binding guard record.",
                    )
                )

    eligible_count = sum(
        item.allowed_by_brief
        and item.applicable
        and item.configured
        and item.enabled_by_runtime_policy
        and item.adapter is not None
        for item in manifest.decisions
    )
    admitted_count = sum(item.admitted for item in manifest.decisions)
    expected_ratio = admitted_count / eligible_count if eligible_count else 1.0
    expected_counts = {
        "registry_source_count": len(manifest.decisions),
        "allowed_source_count": sum(
            item.allowed_by_brief for item in manifest.decisions
        ),
        "applicable_source_count": sum(item.applicable for item in manifest.decisions),
        "eligible_source_count": eligible_count,
        "admitted_source_count": admitted_count,
    }
    for field_name, expected in expected_counts.items():
        if getattr(manifest, field_name) != expected:
            issues.append(
                (
                    "SOURCE_COVERAGE_COUNT_MISMATCH",
                    f"{field_name} does not match the deterministic decision rows.",
                )
            )
    if abs(manifest.eligible_coverage_ratio - expected_ratio) > 1e-9:
        issues.append(
            (
                "SOURCE_COVERAGE_COUNT_MISMATCH",
                "eligible_coverage_ratio does not match admitted/eligible decisions.",
            )
        )
    if manifest.budget.max_run_tokens > brief.budget.max_run_tokens:
        issues.append(
            (
                "SOURCE_COVERAGE_BUDGET_ESCALATION",
                "Coverage token budget exceeds the immutable ResearchBrief budget.",
            )
        )
    if manifest.budget.max_llm_calls > brief.budget.max_llm_calls:
        issues.append(
            (
                "SOURCE_COVERAGE_BUDGET_ESCALATION",
                "Coverage LLM-call budget exceeds the immutable ResearchBrief budget.",
            )
        )
    if manifest.budget.max_run_cost_usd > brief.budget.max_run_cost_usd + 1e-9:
        issues.append(
            (
                "SOURCE_COVERAGE_BUDGET_ESCALATION",
                "Coverage cost budget exceeds the immutable ResearchBrief budget.",
            )
        )
    pharos_decision = decisions_by_id.get("pharos")
    if pharos_decision is not None:
        pharos_profile_enabled = manifest.budget.pharos_targets_per_track > 0
        if pharos_decision.enabled_by_runtime_policy != pharos_profile_enabled:
            issues.append(
                (
                    "SOURCE_COVERAGE_POLICY_MISMATCH",
                    (
                        "Pharos runtime admission does not match the persisted "
                        "research-depth target budget."
                    ),
                )
            )
    expected_profile = settings.evidence_budget_for(brief.research_depth)
    if (
        exact_profile_validation
        and manifest.budget.research_depth != expected_profile.depth
    ):
        issues.append(
            (
                "SOURCE_COVERAGE_DEPTH_MISMATCH",
                "Coverage research depth does not match the immutable ResearchBrief profile.",
            )
        )
    expected_profile_budget = {
        "max_core_source_calls_per_target": (
            expected_profile.core_source_calls_per_target
        ),
        "max_core_records_per_source": expected_profile.core_records_per_source,
        "max_core_elapsed_seconds_per_target": (
            expected_profile.core_elapsed_seconds_per_target
        ),
        "max_core_attempts": expected_profile.core_max_attempts,
        "max_context_source_calls": (
            expected_profile.context_source_calls
            if settings.dynamic_context_enabled
            else 0
        ),
        "max_context_records_per_source": (
            expected_profile.context_records_per_source
            if settings.dynamic_context_enabled
            else 0
        ),
        "max_context_elapsed_seconds": (
            expected_profile.context_elapsed_seconds
            if settings.dynamic_context_enabled
            else 0
        ),
        "max_context_targets_per_track": (
            expected_profile.context_targets_per_track
            if settings.dynamic_context_enabled
            else 0
        ),
        "target_discovery_pubmed_records": (
            expected_profile.target_pubmed_records
        ),
        "target_discovery_pubmed_supplements": (
            expected_profile.target_pubmed_supplements
        ),
        "target_discovery_lookup_limit": expected_profile.target_lookup_limit,
        "target_discovery_max_candidates": expected_profile.target_max_candidates,
        "pharos_targets_per_track": expected_profile.pharos_targets_per_track,
        "candidate_enrichment_per_route": (
            expected_profile.candidate_enrichment_per_route
            if settings.public_candidate_enrichment_enabled
            else 0
        ),
    }
    for field_name, expected in expected_profile_budget.items():
        if (
            exact_profile_validation
            and getattr(manifest.budget, field_name) != expected
        ):
            issues.append(
                (
                    "SOURCE_COVERAGE_BUDGET_MISMATCH",
                    (
                        f"{field_name} does not match the deterministic "
                        f"{expected_profile.depth} profile."
                    ),
                )
            )
    return issues


def _source_configuration(settings: Settings) -> SourceConfiguration:
    drugcentral_path = settings.drugcentral_dataset_path
    archs4_path = settings.archs4_dataset_path
    return SourceConfiguration(
        public_evidence_enabled=settings.public_evidence_enabled,
        public_candidate_enrichment_enabled=(
            settings.public_evidence_enabled
            and settings.public_candidate_enrichment_enabled
            and settings.public_candidate_enrichment_per_route > 0
        ),
        drugcentral_dataset_configured=drugcentral_path is not None,
        drugcentral_dataset_available=bool(
            drugcentral_path is not None and drugcentral_path.is_file()
        ),
        archs4_dataset_configured=archs4_path is not None,
        archs4_dataset_available=bool(
            archs4_path is not None and archs4_path.is_file()
        ),
        cellxgene_sidecar_configured=bool(settings.cellxgene_service_url),
        biogrid_credential_configured=bool(settings.biogrid_access_key),
        google_patents_project_configured=bool(settings.google_cloud_project),
        drugbank_credential_configured=bool(settings.drugbank_api_key),
    )


def _runtime_policy_enabled(
    source_id: str,
    settings: Settings,
    evidence_budget: EvidenceResearchBudget | None = None,
) -> bool:
    if source_id == "pharos":
        return bool(
            settings.public_evidence_enabled
            and (
                evidence_budget is None
                or evidence_budget.pharos_targets_per_track > 0
            )
        )
    if source_id in _CORE_MA1 | _PREPRINT:
        return settings.public_evidence_enabled
    if source_id in _CONTEXT:
        return settings.public_evidence_enabled and settings.dynamic_context_enabled
    if source_id in _OPTIONAL_MA2_ENRICHMENT:
        return (
            settings.ma2_backend_mode == "live"
            and settings.public_candidate_enrichment_enabled
            and settings.public_candidate_enrichment_per_route > 0
        )
    if source_id in {"chembl", "drugcentral", "drugbank"}:
        return settings.ma2_backend_mode == "live"
    return False


def _applicability(
    source_id: str,
    brief: ResearchBrief,
    packet: HypothesisPacket,
) -> tuple[bool, str, str]:
    tracks = {item.track for item in packet.hypotheses}
    has_literature = HypothesisTrack.LITERATURE in tracks
    has_tbio = HypothesisTrack.TBIO in tracks
    text = " ".join(
        [
            brief.research_goal,
            brief.rationale or "",
            brief.constraints.target_cell_type or "",
            *brief.human_priorities,
        ]
    )
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    cell_context_kind = classify_cell_context(
        brief.constraints.target_cell_type,
        brief.human_priorities,
    )
    single_cell = cell_context_kind in {"single_cell", "cell_resolved"}
    expression = has_tbio or "expression" in normalized or "발현" in text
    network = has_tbio or "network" in normalized or "interaction" in normalized or "상호작용" in text
    has_uniprot = any(item.target.uniprot_id for item in packet.hypotheses)

    if source_id in _CORE_MA1:
        return (
            True,
            "Core disease, target-identity, literature, or drug-interaction evidence.",
            "Query for each evidence-qualified target, subject to the MA1 source budget.",
        )
    if source_id in _PREPRINT:
        return (
            has_literature or bool(packet.hypotheses),
            "Preprint coverage can expose recent disease-target evidence not yet indexed elsewhere.",
            "Query as non-causal literature context; retain publication status and provenance.",
        )
    if source_id == "pharos":
        applicable = brief.constraints.druggability_required or any(
            cue in normalized for cue in ("pharos", "tcrd", "tdl", "druggability")
        )
        return (
            applicable,
            "Target-development and druggability context is requested by the brief."
            if applicable
            else "No target-development or druggability context was requested.",
            "Query evidence-qualified targets only; Pharos metadata cannot create a target.",
        )
    if source_id == "clinical_trials":
        return (
            True,
            "Disease-relevant intervention and translational-status context.",
            "Query disease/target context; use trial records as route evidence, not efficacy proof.",
        )
    if source_id in _SINGLE_CELL:
        return (
            single_cell,
            "Explicit cell-type or single-cell context is present."
            if single_cell
            else "Single-cell sources require explicit cell-type or single-cell context.",
            "Query selected targets and retain results as context-only evidence.",
        )
    if source_id == "archs4":
        return (
            expression,
            "Tbio or explicit expression context is present."
            if expression
            else "ARCHS4 requires Tbio or explicit expression context.",
            "Query expression context for evidence-qualified targets only.",
        )
    if source_id in {"string", "biogrid"}:
        return (
            network,
            "Tbio or explicit interaction/network context is present."
            if network
            else "Network sources require Tbio or explicit interaction context.",
            "Query interaction context; do not convert network proximity into causal support.",
        )
    if source_id in {"gtex", "reactome", "kegg"}:
        return (
            has_tbio,
            "A Tbio-track hypothesis requires tissue/pathway context."
            if has_tbio
            else "This source is limited to Tbio-track contextual enrichment.",
            "Query selected Tbio targets and retain results as context-only evidence.",
        )
    if source_id == "quickgo":
        applicable = has_tbio and has_uniprot
        return (
            applicable,
            "A Tbio hypothesis and resolved UniProt identifier are available."
            if applicable
            else "QuickGO requires both a Tbio hypothesis and resolved UniProt identifier.",
            "Query exact UniProt targets and retain ontology annotations as context.",
        )

    # MA2 sources are admitted prospectively because the route guard runs after
    # the Planner.  Their branch conditions prevent inactive-route execution.
    ma2_conditions = {
        "chembl": "Query exact-target measured ligands for active repurposing or de novo seed branches.",
        "bindingdb": "Query exact-structure/exact-target activity when candidate enrichment is active.",
        "drugcentral": "Query exact-target known-drug interactions for active repurposing branches.",
        "pubchem": "Resolve candidate chemical identity or structure when an identifier/name needs enrichment.",
        "rcsb_pdb": "Query experimentally resolved structures only when structural assessment is applicable.",
        "alphafold_db": "Query predicted target structure only when structural assessment is applicable.",
        "drugbank": "Query identity, regulatory, and contraindication context before expensive computation.",
        "google_patents": "Query prior-art context for de novo candidates after a stable identity is available.",
    }
    if source_id in ma2_conditions:
        return (
            True,
            "Candidate discovery, identity, safety, structure, or prior-art coverage may apply after route selection.",
            ma2_conditions[source_id],
        )
    return False, "No deterministic applicability rule is registered.", "Do not query."


def _bounded(value: object, limit: int) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())[:limit].strip()
    return normalized or None


__all__ = [
    "SourceCoverageLLMSuggestion",
    "apply_llm_source_suggestions",
    "build_broad_source_coverage",
    "source_coverage_execution_contract",
    "source_registry_digest",
    "validate_source_coverage_manifest",
]
