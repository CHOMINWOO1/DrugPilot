"""Pure evidence-governed MA2 route selection.

The policy distinguishes *strong*, *weak*, and *absent* repurposing evidence.
A large drug--gene interaction count is only weak context: it cannot by itself
route a target to repurposing.  Strong evidence requires a disease-aware signal
such as an Open Targets known-drug score, an approved/clinical tractability
annotation combined with target evidence, or a disease-relevant clinical drug
whose target action is corroborated by DGIdb.

Generic chemical identity records (for example a PubChem hit) and keyword-only
literature mentions are deliberately ignored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from drugpilot.domain.models import CandidateRoute, EvidenceRef, Hypothesis, TargetCandidate


ROUTE_POLICY_VERSION = "ma1-target-knownness-v6"
HYPOTHESIS_SELECTION_POLICY_VERSION = "ma1-hypothesis-priority-v2"
KNOWN_DRUG_REPURPOSING_THRESHOLD = 0.20
WEAK_KNOWN_DRUG_THRESHOLD = 0.02
UNDEREXPLORED_KNOWN_DRUG_MAX = 0.02
MIN_DE_NOVO_ASSOCIATION_SCORE = 0.10

_CANONICAL_TDL = {
    "tclin": "Tclin",
    "tchem": "Tchem",
    "tbio": "Tbio",
    "tdark": "Tdark",
}

_UNFAVOURABLE_TRIAL_STATUSES = {
    "terminated",
    "withdrawn",
    "suspended",
    "no longer available",
    "temporarily not available",
}
_ACTIVE_OR_COMPLETED_TRIAL_STATUSES = {
    "active not recruiting",
    "approved for marketing",
    "available",
    "completed",
    "enrolling by invitation",
    "not yet recruiting",
    "recruiting",
}
_ACTIVE_TRIAL_STATUSES = {
    "active not recruiting",
    "enrolling by invitation",
    "not yet recruiting",
    "recruiting",
}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def _unit_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0.0 <= parsed <= 1.0 else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _canonical_tdl(value: Any) -> str | None:
    return _CANONICAL_TDL.get(str(value or "").strip().casefold())


def _track_tdl_contract(
    hypothesis: Hypothesis,
    metadata: Mapping[str, Any],
    verified_tdl: str | None,
) -> dict[str, Any]:
    """Keep work allocation and scientific TDL in separate namespaces."""

    if verified_tdl in {"Tclin", "Tchem"}:
        effective_work_track = "literature"
        basis = "effective_tdl"
    elif verified_tdl in {"Tbio", "Tdark"}:
        effective_work_track = "tbio"
        basis = "effective_tdl"
    else:
        effective_work_track = hypothesis.track.value
        basis = "hypothesis_track_fallback"

    raw = metadata.get("track_tdl_contract")
    raw_policy = raw.get("policy") if isinstance(raw, Mapping) else None
    return {
        "policy": str(
            raw_policy or "separate-work-track-and-scientific-tdl-v1"
        ),
        "hypothesis_track": hypothesis.track.value,
        "hypothesis_track_semantics": "original_ma1_evidence_supply_track",
        "effective_work_track": effective_work_track,
        "effective_work_track_basis": basis,
        "effective_tdl": verified_tdl,
        "effective_tdl_semantics": (
            "authoritative_scientific_target_development_level"
        ),
        "namespaces_distinct": True,
        "context_source_plan_uses": "effective_work_track",
        "ma2_candidate_route_uses": "effective_tdl",
    }


def _target_routed(record: EvidenceRef, symbol: str) -> bool:
    query = str(record.metadata.get("query") or "")
    if not query:
        return False
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", query, re.I))


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _dedupe_text(values: Sequence[str], *, limit: int = 12) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", value.strip())
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _normalise_phrase(value: Any) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", str(value or "").casefold()).strip()


def _normalise_drug_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _condition_matches_disease(condition: Any, disease: str) -> bool:
    condition_text = _normalise_phrase(condition)
    disease_text = _normalise_phrase(disease)
    if not condition_text or not disease_text:
        return False
    if disease_text in condition_text or condition_text in disease_text:
        return True
    disease_tokens = [token for token in disease_text.split() if len(token) > 1]
    if disease_tokens and all(token in condition_text.split() for token in disease_tokens):
        return True
    acronym = "".join(token[0] for token in disease_tokens)
    return len(acronym) >= 2 and re.search(
        rf"(?<![a-z0-9]){re.escape(acronym)}(?![a-z0-9])",
        condition_text,
    ) is not None


def _phase_number(phases: Sequence[Any]) -> int:
    maximum = 0
    for phase in phases:
        numbers = [int(value) for value in re.findall(r"[1-4]", str(phase))]
        if numbers:
            maximum = max(maximum, *numbers)
    return maximum


def _interaction_action_text(interaction: Mapping[str, Any]) -> str:
    values: list[str] = []
    for item in _mapping_sequence(interaction.get("interaction_types")):
        values.extend(
            str(item.get(key) or "")
            for key in ("type", "directionality")
        )
    raw = interaction.get("interaction_types")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values.extend(str(item) for item in raw if not isinstance(item, Mapping))
    return _normalise_phrase(" ".join(values))


def _action_alignment(action_text: str, direction: str) -> str:
    if not action_text:
        return "unknown"
    inhibitory = {
        "antagonist",
        "blocker",
        "inhibitor",
        "inhibitory",
        "negative modulator",
        "suppressor",
    }
    activating = {
        "activator",
        "agonist",
        "inducer",
        "positive modulator",
        "stimulator",
    }
    has_inhibitory = any(term in action_text for term in inhibitory)
    has_activating = any(term in action_text for term in activating)
    if direction == "inhibit":
        return "aligned" if has_inhibitory else ("conflicting" if has_activating else "unknown")
    if direction == "activate":
        return "aligned" if has_activating else ("conflicting" if has_inhibitory else "unknown")
    return "aligned" if has_inhibitory or has_activating else "unknown"


def _true_tractability_labels(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in _mapping_sequence(summary.get("tractability"))
        if item.get("value") is True and str(item.get("label") or "").strip()
    ]


def _evidence_mode(hypothesis: Hypothesis, target: TargetCandidate) -> str:
    metadata = target.metadata if isinstance(target.metadata, Mapping) else {}
    external = any(
        str(item.metadata.get("provenance_class") or "").casefold()
        in {"external_live", "external_cache", "external_mixed"}
        or item.metadata.get("mock") is False
        for item in hypothesis.evidence
    )
    fixture = (
        str(metadata.get("target_discovery_source") or "").casefold()
        == "fixture_fallback"
        or any(
            item.metadata.get("mock") is True
            or str(item.metadata.get("provenance_class") or "").casefold()
            == "fixture_fallback"
            for item in hypothesis.evidence
        )
    )
    if external:
        return "external"
    if fixture:
        return "mock_fixture"
    return "unknown"


def _is_atomic_named_drug_label(
    value: Any,
    *,
    identifier_backed: bool = False,
) -> bool:
    """Accept one named medicine, not a modality or combination label."""

    name = str(value or "").strip()
    if not name:
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    non_drug_controls = {
        "placebo",
        "best supportive care",
        "standard of care",
        "investigator choice",
        "chemotherapy",
        "immunotherapy",
        "radiotherapy",
        "radiation therapy",
        "targeted therapy",
    }
    if normalized in non_drug_controls or re.search(
        r"\b(?:placebo|best supportive care|standard of care|investigator choice|chemotherapy|immunotherapy|radiotherapy|radiation therapy|targeted therapy)\b",
        normalized,
    ):
        return False
    # A source may encode a whole regimen in one DRUG label.  Without an
    # identifier-backed decomposition, counting that string as one atomic
    # medicine is misleading.  Preserve it as rejected context instead.
    if re.search(
        r"(?:,|;|\+|&|\\|\||\band\b|\bwith\b|\bplus\b|\bor\b|\bversus\b|\bvs\.?\b)",
        name,
        re.I,
    ):
        return False
    # Slash-separated labels are ambiguous: trial registries commonly use them
    # for combinations, while an identifier-backed drug record may use a slash
    # for two aliases of the same molecule.  Only the latter is accepted.
    return "/" not in name or identifier_backed


def _is_named_drug_intervention(item: Mapping[str, Any]) -> bool:
    return bool(
        str(item.get("type") or "").strip().casefold() == "drug"
        and _is_atomic_named_drug_label(item.get("name"))
    )


def derive_route_evidence(
    hypothesis: Hypothesis,
    target: TargetCandidate,
) -> dict[str, Any]:
    """Derive target knownness from MA1 metadata and structured evidence."""

    metadata = target.metadata if isinstance(target.metadata, Mapping) else {}
    raw_summary = metadata.get("evidence_summary")
    summary = raw_summary if isinstance(raw_summary, Mapping) else {}
    known_drug_score = _unit_score(summary.get("known_drug_score"))
    association_score = _unit_score(metadata.get("association_score"))
    if association_score is None:
        association_score = _unit_score(summary.get("overall_association_score"))
    metadata_known_drug_count = _nonnegative_int(metadata.get("known_drug_count"))

    structured_sources: list[str] = []
    drug_examples: list[str] = []
    evidence_ids: list[str] = []
    dgidb_interaction_count = 0
    dgidb_interactions_observed = 0
    dgidb_record_observed = False
    dgidb_drug_names: list[str] = []
    direction_aligned_drugs: list[str] = []
    direction_conflicting_drugs: list[str] = []
    clinical_trial_drugs: list[str] = []
    disease_relevant_trial_drugs: list[str] = []
    qualifying_trial_drugs: list[str] = []
    trial_summaries: list[dict[str, Any]] = []
    excluded_unqualified_drug_labels: list[str] = []
    verified_tdl: str | None = None
    verified_tdl_source: str | None = None
    observed_tdl_claims: list[dict[str, Any]] = []
    tdl_source_claims: list[dict[str, Any]] = []
    tdl_resolution_status = "unavailable"
    tdl_conflict_detected = False
    partial_source_failures: dict[str, str] = {}

    for record in hypothesis.evidence:
        failures = record.metadata.get("partial_source_failures")
        if isinstance(failures, Mapping):
            for source, failure in failures.items():
                partial_source_failures[str(source)] = str(failure)

    for record in hypothesis.evidence:
        source = record.source.strip()
        source_key = source.casefold()
        if source_key == "dgidb" and _target_routed(record, target.symbol):
            dgidb_record_observed = True
            interactions = _mapping_sequence(record.metadata.get("interactions"))
            raw_names = [
                str(item.get("drug_name") or "").strip()
                for item in interactions
                if item.get("drug_name") or item.get("drug_id")
            ]
            names = [
                str(item.get("drug_name") or "").strip()
                for item in interactions
                if _is_atomic_named_drug_label(
                    item.get("drug_name"),
                    identifier_backed=bool(item.get("drug_id")),
                )
            ]
            excluded_unqualified_drug_labels.extend(
                str(item.get("drug_name") or "").strip()
                for item in interactions
                if str(item.get("drug_name") or "").strip()
                and not _is_atomic_named_drug_label(
                    item.get("drug_name"),
                    identifier_backed=bool(item.get("drug_id")),
                )
            )
            count = _nonnegative_int(record.metadata.get("interaction_count")) or 0
            if count > 0 or interactions:
                dgidb_interaction_count = max(
                    dgidb_interaction_count,
                    count,
                    len(interactions),
                )
                dgidb_interactions_observed = max(
                    dgidb_interactions_observed,
                    len(interactions),
                )
                structured_sources.append("DGIdb")
                drug_examples.extend(names)
                dgidb_drug_names.extend(names)
                for item in interactions:
                    name = str(item.get("drug_name") or "").strip()
                    if not _is_atomic_named_drug_label(
                        name,
                        identifier_backed=bool(item.get("drug_id")),
                    ):
                        continue
                    alignment = _action_alignment(
                        _interaction_action_text(item),
                        hypothesis.intervention_direction,
                    )
                    if alignment == "aligned":
                        direction_aligned_drugs.append(name)
                    elif alignment == "conflicting":
                        direction_conflicting_drugs.append(name)
                evidence_ids.append(record.evidence_id)
            continue

        if source_key == "clinicaltrials.gov" and _target_routed(record, target.symbol):
            study_type = str(record.metadata.get("study_type") or "").casefold()
            if study_type and study_type != "interventional":
                continue
            interventions = _mapping_sequence(record.metadata.get("interventions"))
            raw_names = [
                str(item.get("name") or "").strip()
                for item in interventions
                if str(item.get("type") or "").strip().casefold() == "drug"
                and str(item.get("name") or "").strip()
            ]
            names = [
                str(item.get("name") or "").strip()
                for item in interventions
                if _is_named_drug_intervention(item)
            ]
            excluded_unqualified_drug_labels.extend(
                name for name in raw_names
                if not _is_atomic_named_drug_label(name)
            )
            if names:
                clinical_trial_drugs.extend(names)
                structured_sources.append("ClinicalTrials.gov")
                drug_examples.extend(names)
                evidence_ids.append(record.evidence_id)
                conditions = list(record.metadata.get("conditions") or [])
                disease_relevant = any(
                    _condition_matches_disease(condition, hypothesis.disease)
                    for condition in conditions
                )
                phases = list(record.metadata.get("phases") or [])
                phase = _phase_number(phases)
                status = _normalise_phrase(record.metadata.get("overall_status"))
                favourable_status = status not in _UNFAVOURABLE_TRIAL_STATUSES
                status_supported = status in _ACTIVE_OR_COMPLETED_TRIAL_STATUSES
                if disease_relevant:
                    disease_relevant_trial_drugs.extend(names)
                if disease_relevant and favourable_status and (phase > 0 or status_supported):
                    qualifying_trial_drugs.extend(names)
                trial_summaries.append(
                    {
                        "evidence_id": record.evidence_id,
                        "nct_id": record.external_id,
                        "phase": phase,
                        "status": status or None,
                        "disease_relevant": disease_relevant,
                        "favourable_status": favourable_status,
                        "drug_names": names,
                    }
                )
            continue

        if source_key == "pharos" and (
            _target_routed(record, target.symbol)
            or record.external_id.casefold() == target.symbol.casefold()
        ):
            tdl = _canonical_tdl(record.metadata.get("tdl"))
            if tdl is not None:
                observed_tdl_claims.append(
                    {
                        "source": "Pharos",
                        "value": tdl,
                        "evidence_id": record.evidence_id,
                        "provenance_class": record.metadata.get(
                            "provenance_class"
                        ),
                    }
                )

    # Prefer the orchestrator-authored reconciliation contract.  It retains
    # every source claim and explicitly marks unresolved same-priority
    # conflicts, so evidence iteration order can never choose a TDL.
    raw_reconciliation = metadata.get("tdl_reconciliation")
    raw_source_claims = metadata.get("tdl_source_claims")
    if isinstance(raw_source_claims, Sequence) and not isinstance(
        raw_source_claims, (str, bytes, bytearray)
    ):
        tdl_source_claims = [
            dict(item) for item in raw_source_claims if isinstance(item, Mapping)
        ]
    else:
        tdl_source_claims = observed_tdl_claims

    if isinstance(raw_reconciliation, Mapping):
        tdl_resolution_status = str(
            raw_reconciliation.get("status") or "unavailable"
        )
        tdl_conflict_detected = bool(
            raw_reconciliation.get("conflict_detected")
        )
        candidate_tdl = _canonical_tdl(
            raw_reconciliation.get("effective_tdl")
            or metadata.get("effective_tdl")
        )
        candidate_source = str(
            raw_reconciliation.get("effective_tdl_source")
            or metadata.get("effective_tdl_source")
            or ""
        ).strip()
        if (
            tdl_resolution_status in {"resolved", "resolved_with_source_conflict"}
            and candidate_tdl is not None
            and candidate_source
        ):
            verified_tdl = candidate_tdl
            verified_tdl_source = candidate_source
    else:
        legacy_claims = list(observed_tdl_claims)
        metadata_tdl_source = str(metadata.get("tdl_source") or "").strip()
        metadata_tdl = _canonical_tdl(target.tdl)
        if (
            metadata_tdl is not None
            and metadata_tdl_source.casefold()
            in {"pharos", "pharos api", "target central resource database"}
        ):
            legacy_claims.append(
                {
                    "source": metadata_tdl_source,
                    "value": metadata_tdl,
                    "evidence_id": None,
                    "provenance_class": "persisted_verified_metadata",
                }
            )
        tdl_source_claims = legacy_claims
        legacy_values = {
            str(item["value"])
            for item in legacy_claims
            if _canonical_tdl(item.get("value")) is not None
        }
        tdl_conflict_detected = len(legacy_values) > 1
        if len(legacy_values) == 1:
            verified_tdl = next(iter(legacy_values))
            verified_tdl_source = " + ".join(
                sorted({str(item["source"]) for item in legacy_claims})
            )
            tdl_resolution_status = "resolved_legacy_unique_source_claim"
        elif tdl_conflict_detected:
            tdl_resolution_status = "unresolved_legacy_source_conflict"

    unique_examples = _dedupe_text(drug_examples)
    unique_dgidb_drugs = _dedupe_text(dgidb_drug_names, limit=100)
    unique_aligned_drugs = _dedupe_text(direction_aligned_drugs, limit=100)
    unique_conflicting_drugs = _dedupe_text(direction_conflicting_drugs, limit=100)
    unique_trial_drugs = _dedupe_text(clinical_trial_drugs, limit=100)
    unique_relevant_trial_drugs = _dedupe_text(
        disease_relevant_trial_drugs,
        limit=100,
    )
    unique_qualifying_trial_drugs = _dedupe_text(qualifying_trial_drugs, limit=100)
    unique_excluded_unqualified_labels = _dedupe_text(
        excluded_unqualified_drug_labels,
        limit=100,
    )
    dgidb_by_key = {
        _normalise_drug_name(name): name
        for name in unique_dgidb_drugs
        if _normalise_drug_name(name)
    }
    qualifying_by_key = {
        _normalise_drug_name(name): name
        for name in unique_qualifying_trial_drugs
        if _normalise_drug_name(name)
    }
    aligned_keys = {
        _normalise_drug_name(name)
        for name in unique_aligned_drugs
        if _normalise_drug_name(name)
    }
    overlap_keys = sorted(set(dgidb_by_key) & set(qualifying_by_key))
    cross_source_drug_overlap = [dgidb_by_key[key] for key in overlap_keys]
    direction_aligned_overlap = [
        dgidb_by_key[key]
        for key in overlap_keys
        if key in aligned_keys
    ]
    late_stage_trial_count = sum(
        item["disease_relevant"]
        and item["favourable_status"]
        and item["phase"] >= 2
        for item in trial_summaries
    )
    active_trial_count = sum(
        item["disease_relevant"]
        and item["status"] in _ACTIVE_TRIAL_STATUSES
        for item in trial_summaries
    )
    active_or_completed_trial_count = sum(
        item["disease_relevant"]
        and item["status"] in _ACTIVE_OR_COMPLETED_TRIAL_STATUSES
        for item in trial_summaries
    )
    tractability = _true_tractability_labels(summary)
    approved_drug_tractability = any(
        "approved drug" in _normalise_phrase(item.get("label"))
        for item in tractability
    )
    clinical_stage_tractability = any(
        "clinical" in _normalise_phrase(item.get("label"))
        for item in tractability
    )
    context_outcomes = _mapping_sequence(metadata.get("context_source_outcomes"))
    clinical_outcome = next(
        (
            item
            for item in context_outcomes
            if str(item.get("source") or "").casefold() == "clinicaltrials.gov"
        ),
        None,
    )
    if trial_summaries:
        clinical_trials_coverage = "ok"
    elif clinical_outcome is not None:
        clinical_trials_coverage = str(
            clinical_outcome.get("status") or "unknown"
        ).casefold()
    else:
        clinical_trials_coverage = "unknown"
    if dgidb_interaction_count > 0:
        dgidb_coverage = "ok"
    elif dgidb_record_observed:
        dgidb_coverage = "empty"
    else:
        failure = next(
            (
                value
                for source, value in partial_source_failures.items()
                if source.casefold() == "dgidb"
            ),
            None,
        )
        dgidb_coverage = (
            "empty" if failure == "empty_result" else "error" if failure else "unknown"
        )
    dgidb_overlap_verifiable = bool(
        dgidb_coverage == "empty"
        or (
            dgidb_interaction_count > 0
            and dgidb_interaction_count <= dgidb_interactions_observed
        )
    )
    evidence_mode = _evidence_mode(hypothesis, target)
    if evidence_mode == "mock_fixture":
        known_drug_coverage_status = (
            "fixture_assumed" if bool(summary) else "fixture_incomplete"
        )
        known_drug_coverage_complete = bool(summary)
    else:
        known_drug_coverage_complete = bool(
            summary
            and dgidb_coverage in {"ok", "empty"}
            and (dgidb_coverage == "empty" or dgidb_overlap_verifiable)
        )
        known_drug_coverage_status = (
            "complete" if known_drug_coverage_complete else "incomplete"
        )
    observed_drug_keys = {
        key
        for name in [*dgidb_drug_names, *clinical_trial_drugs]
        if (key := _normalise_drug_name(name))
    }
    structured_known_drug_count = len(observed_drug_keys)
    # DGIdb's ``interaction_count`` is a count of graph edges, not a count of
    # unique medicines.  Keep it separately for coverage/provenance, but never
    # promote it into this named-drug cardinality.
    known_drug_count = (
        structured_known_drug_count
        if structured_known_drug_count > 0
        else metadata_known_drug_count
    )
    track_tdl_contract = _track_tdl_contract(
        hypothesis,
        metadata,
        verified_tdl,
    )
    return {
        "known_drug_score": known_drug_score,
        "metadata_known_drug_count": metadata_known_drug_count,
        "structured_known_drug_count": structured_known_drug_count,
        "structured_known_drug_count_basis": (
            "unique_normalized_named_drugs_in_observed_dgidb_and_trial_rows"
        ),
        "known_drug_count": known_drug_count,
        "known_drug_count_basis": (
            "observed_unique_named_drugs"
            if structured_known_drug_count > 0
            else "target_metadata_named_drug_count"
            if metadata_known_drug_count is not None
            else "unavailable"
        ),
        "known_drug_sources": sorted(set(structured_sources), key=str.casefold),
        "known_drug_examples": unique_examples,
        "excluded_unqualified_drug_labels": unique_excluded_unqualified_labels,
        "known_drug_evidence_ids": list(dict.fromkeys(evidence_ids)),
        "dgidb_interaction_count": dgidb_interaction_count,
        "dgidb_interactions_observed": dgidb_interactions_observed,
        "dgidb_record_observed": dgidb_record_observed,
        "dgidb_overlap_verifiable": dgidb_overlap_verifiable,
        "clinical_trial_drug_count": len(unique_trial_drugs),
        "disease_relevant_trial_count": sum(
            item["disease_relevant"] for item in trial_summaries
        ),
        "qualifying_trial_count": sum(
            item["disease_relevant"]
            and item["favourable_status"]
            and (item["phase"] > 0 or item["status"] in _ACTIVE_OR_COMPLETED_TRIAL_STATUSES)
            for item in trial_summaries
        ),
        "late_stage_trial_count": late_stage_trial_count,
        "active_trial_count": active_trial_count,
        "active_or_completed_trial_count": active_or_completed_trial_count,
        "trial_summaries": trial_summaries,
        "disease_relevant_trial_drugs": unique_relevant_trial_drugs,
        "qualifying_trial_drugs": unique_qualifying_trial_drugs,
        "cross_source_drug_overlap": cross_source_drug_overlap,
        "direction_aligned_overlap": direction_aligned_overlap,
        "direction_aligned_drugs": unique_aligned_drugs,
        "direction_conflicting_drugs": unique_conflicting_drugs,
        "approved_drug_tractability": approved_drug_tractability,
        "clinical_stage_tractability": clinical_stage_tractability,
        "association_score": association_score,
        "known_drug_source_complete": known_drug_coverage_complete,
        "known_drug_coverage_status": known_drug_coverage_status,
        "source_coverage": {
            "open_targets_known_drug_summary": "ok" if summary else "unknown",
            "dgidb": dgidb_coverage,
            "clinical_trials": clinical_trials_coverage,
            "partial_source_failures": partial_source_failures,
        },
        "coverage_requirements": {
            "required_for_underexplored": [
                "open_targets_known_drug_summary",
                "dgidb_target_interactions",
            ],
            "optional_context": ["clinical_trials_disease_context"],
            "interpretation": (
                "ClinicalTrials.gov is optional negative coverage for Tbio/novel-target "
                "routing because that context route may intentionally skip trials; positive "
                "trial evidence is still qualified when present."
            ),
        },
        "verified_tdl": verified_tdl,
        "verified_tdl_source": verified_tdl_source,
        "effective_tdl": verified_tdl,
        "effective_tdl_source": verified_tdl_source,
        "tdl_source_claims": tdl_source_claims,
        "tdl_resolution_status": tdl_resolution_status,
        "tdl_conflict_detected": tdl_conflict_detected,
        "hypothesis_track": hypothesis.track.value,
        "effective_work_track": track_tdl_contract["effective_work_track"],
        "effective_work_track_basis": track_tdl_contract[
            "effective_work_track_basis"
        ],
        "track_tdl_contract": track_tdl_contract,
        "support_evidence_count": _route_support_count(hypothesis),
        "evidence_sources": sorted({item.source for item in hypothesis.evidence}),
        "evidence_mode": evidence_mode,
        "drug_name_match_method": "exact_casefold_alphanumeric_name_v1",
        "drug_name_match_limitations": [
            "salt forms, brand names, synonyms, and identifiers are not yet resolved",
            "DGIdb interactions are bounded; an overlap absence is not verifiable when total count exceeds observed records",
        ],
    }


def _route_support_count(hypothesis: Hypothesis) -> int:
    if "ma3_admission" in hypothesis.target.metadata:
        from drugpilot.agents.hypothesis_admission import admitted_support_ids
        return len(admitted_support_ids(hypothesis))
    return sum(item.supports == "support" for item in hypothesis.evidence)


def select_route_from_evidence(
    hypothesis: Hypothesis,
    target: TargetCandidate,
) -> dict[str, Any]:
    """Return the complete persisted route-decision contract."""

    evidence = derive_route_evidence(hypothesis, target)
    known_drug_score = evidence["known_drug_score"]
    known_drug_count = evidence["known_drug_count"]
    association_score = evidence["association_score"]
    verified_tdl = str(evidence["verified_tdl"] or "").casefold()
    evidence_supported = evidence["support_evidence_count"] > 0 and (
        association_score is None
        or association_score >= MIN_DE_NOVO_ASSOCIATION_SCORE
    )
    strong_reasons: list[str] = []
    if verified_tdl == "tclin":
        strong_reasons.append("PHAROS_TCLIN")
    if (
        known_drug_score is not None
        and known_drug_score >= KNOWN_DRUG_REPURPOSING_THRESHOLD
        and evidence_supported
    ):
        strong_reasons.append("DISEASE_TARGET_KNOWN_DRUG_SCORE")
    if (
        evidence["approved_drug_tractability"]
        and evidence_supported
        and (
            evidence["dgidb_interaction_count"] > 0
            or evidence["qualifying_trial_count"] > 0
        )
    ):
        strong_reasons.append("APPROVED_TRACTABILITY_WITH_TARGET_CONTEXT")
    if evidence["direction_aligned_overlap"]:
        strong_reasons.append("DISEASE_TRIAL_TARGET_ACTION_OVERLAP")
    if (
        evidence["late_stage_trial_count"] > 0
        and evidence["direction_aligned_overlap"]
        and evidence_supported
    ):
        strong_reasons.append("LATE_STAGE_DISEASE_TRIAL_WITH_ALIGNED_TARGET_ACTION")

    weak_reasons: list[str] = []
    if evidence["dgidb_interaction_count"] > 0:
        weak_reasons.append("DGIDB_TARGET_INTERACTIONS")
    if evidence["disease_relevant_trial_count"] > 0:
        weak_reasons.append("DISEASE_RELEVANT_TRIAL_CONTEXT")
    if verified_tdl == "tchem":
        weak_reasons.append("PHAROS_TCHEM")
    if evidence["clinical_stage_tractability"]:
        weak_reasons.append("CLINICAL_TRACTABILITY")
    if evidence["approved_drug_tractability"]:
        weak_reasons.append("APPROVED_DRUG_TRACTABILITY")
    if (
        known_drug_score is not None
        and known_drug_score > WEAK_KNOWN_DRUG_THRESHOLD
    ):
        weak_reasons.append("NONZERO_DISEASE_TARGET_KNOWN_DRUG_SCORE")
    if known_drug_count is not None and known_drug_count > 0:
        weak_reasons.append("UNQUALIFIED_KNOWN_DRUG_COUNT")

    if strong_reasons:
        drug_evidence_tier = "strong"
        tier_reason_codes = strong_reasons
    elif weak_reasons:
        drug_evidence_tier = "weak"
        tier_reason_codes = weak_reasons
    else:
        drug_evidence_tier = "none"
        tier_reason_codes = ["NO_QUALIFIED_REPURPOSING_EVIDENCE"]

    completeness_components = {
        "association_score": association_score is not None,
        "known_drug_summary": bool(
            isinstance(target.metadata.get("evidence_summary"), Mapping)
        ),
        "tractability_assessment": bool(
            isinstance(target.metadata.get("evidence_summary"), Mapping)
            and "tractability" in target.metadata.get("evidence_summary", {})
        ),
        "target_drug_source": evidence["dgidb_interaction_count"] > 0,
        "disease_trial_source": bool(evidence["trial_summaries"]),
        "target_action_annotation": bool(
            evidence["direction_aligned_drugs"]
            or evidence["direction_conflicting_drugs"]
        ),
    }
    evidence_completeness_score = round(
        sum(completeness_components.values()) / len(completeness_components),
        3,
    )
    repurposing_signal_score = _clamp(
        (0.35 if verified_tdl == "tclin" else 0.0)
        + (0.30 if known_drug_score is not None and known_drug_score >= KNOWN_DRUG_REPURPOSING_THRESHOLD else 0.0)
        + (0.25 if evidence["approved_drug_tractability"] else 0.0)
        + (0.25 if evidence["direction_aligned_overlap"] else 0.0)
        + (0.15 if evidence["late_stage_trial_count"] else 0.0)
        + (0.05 if evidence["dgidb_interaction_count"] else 0.0)
        + (0.05 if verified_tdl == "tchem" else 0.0)
    )
    explicit_underexplored_signal = bool(
        verified_tdl in {"tbio", "tdark"}
        or (
            evidence["known_drug_source_complete"]
            and not evidence["approved_drug_tractability"]
            and not evidence["clinical_stage_tractability"]
            and (
                known_drug_score is None
                or known_drug_score <= UNDEREXPLORED_KNOWN_DRUG_MAX
            )
            and not evidence["direction_aligned_overlap"]
            and evidence["late_stage_trial_count"] == 0
        )
    )

    fallback_used = False
    fallback_reason: str | None = None
    if drug_evidence_tier == "strong":
        route = CandidateRoute.REPURPOSING
        status = "repurposing_evidence_strong"
        source_text = ", ".join(evidence["known_drug_sources"])
        rationale = (
            "Strong disease-aware target-level repurposing evidence"
            + (f" from {source_text}" if source_text else "")
            + " satisfies at least one pre-specified approval, clinical-stage, "
            "known-drug, or target-action criterion; de novo generation is suppressed. "
            "This route decision does not claim that the subsequently retrieved "
            "candidates have disease-specific or clinical evidence."
        )
        confidence = _clamp(
            0.70 + 0.18 * repurposing_signal_score + 0.10 * evidence_completeness_score,
            maximum=0.98,
        )
    elif (
        drug_evidence_tier in {"weak", "none"}
        and explicit_underexplored_signal
        and evidence_supported
    ):
        route = CandidateRoute.DE_NOVO
        status = (
            "underexplored_with_weak_drug_context"
            if drug_evidence_tier == "weak"
            else "underexplored_without_drug_evidence"
        )
        rationale = (
            "The target has biological support but lacks strong, disease- and "
            "action-qualified repurposing evidence. Unqualified interaction counts "
            "do not override underexplored status, so de novo design is selected."
        )
        confidence = _clamp(
            0.60
            + 0.15 * (association_score or 0.0)
            + 0.10 * evidence_completeness_score,
            maximum=0.95,
        )
    else:
        fallback_used = True
        route = (
            CandidateRoute.DE_NOVO
            if evidence["effective_work_track"] == "tbio"
            else CandidateRoute.REPURPOSING
        )
        status = "ambiguous_drug_evidence"
        fallback_reason = (
            "Repurposing evidence was not strong and the underexplored criteria were "
            "not fully established; "
            f"the deterministic {evidence['effective_work_track']} work-allocation "
            f"track selected {route.value} only as a low-confidence mock tie-break. "
            f"The original MA1 {hypothesis.track.value} hypothesis track remains "
            "provenance and is not interpreted as a scientific TDL."
        )
        rationale = fallback_reason
        confidence = 0.42 if evidence["evidence_mode"] == "mock_fixture" else 0.35

    factors = [
        {
            "name": "known_drug_score",
            "value": known_drug_score,
            "source": "MA1.target.metadata.evidence_summary",
            "interpretation": "Open Targets routing heuristic; not an efficacy probability",
        },
        {
            "name": "known_drug_count",
            "value": known_drug_count,
            "source": "MA1 target metadata + structured target-routed evidence",
            "interpretation": (
                "Unique normalized named drugs from observed rows when available; "
                "DGIdb total interaction edges are not counted as drugs"
            ),
        },
        {
            "name": "structured_known_drug_evidence",
            "value": evidence["structured_known_drug_count"],
            "source": evidence["known_drug_sources"],
            "interpretation": (
                "Unique named drugs in observed DGIdb interactions and target-routed "
                "ClinicalTrials DRUG interventions; "
                "disease, stage, status, and action qualification are evaluated separately"
            ),
        },
        {
            "name": "drug_evidence_tier",
            "value": drug_evidence_tier,
            "source": "deterministic evidence qualification",
            "interpretation": tier_reason_codes,
        },
        {
            "name": "disease_trial_target_action_overlap",
            "value": evidence["direction_aligned_overlap"],
            "source": "ClinicalTrials.gov x DGIdb",
            "interpretation": "Same named drug, relevant disease trial, and aligned target action",
        },
        {
            "name": "approved_or_clinical_tractability",
            "value": {
                "approved": evidence["approved_drug_tractability"],
                "clinical": evidence["clinical_stage_tractability"],
            },
            "source": "Open Targets tractability",
            "interpretation": "Stage evidence is combined with disease/target support",
        },
        {
            "name": "verified_target_development_level",
            "value": evidence["verified_tdl"],
            "source": evidence["verified_tdl_source"],
            "interpretation": "Used only when explicitly sourced from Pharos/TCRD",
        },
        {
            "name": "association_score",
            "value": association_score,
            "source": "MA1.target.metadata",
            "interpretation": "Minimum support check for an underexplored de novo target",
        },
        {
            "name": "support_evidence_count",
            "value": evidence["support_evidence_count"],
            "source": "MA1.Hypothesis.evidence",
            "interpretation": "At least one linked support record is required",
        },
        {
            "name": "ma1_hypothesis_track",
            "value": hypothesis.track.value,
            "source": "MA1.Hypothesis.track",
            "interpretation": (
                "Original evidence-supply provenance; never interpreted as a "
                "scientific target-development level"
            ),
        },
        {
            "name": "effective_work_allocation_track",
            "value": evidence["effective_work_track"],
            "source": evidence["effective_work_track_basis"],
            "interpretation": (
                "Context/source allocation namespace, deterministically reconciled "
                "from effective TDL when available"
            ),
        },
    ]
    return {
        "policy_version": ROUTE_POLICY_VERSION,
        "hypothesis_id": hypothesis.hypothesis_id,
        "target_symbol": target.symbol,
        "selected_route": route.value,
        "selected_routes": [route.value],
        "decision": route.value.upper(),
        "rationale": rationale,
        "factors": factors,
        **evidence,
        "drug_evidence_tier": drug_evidence_tier,
        "tier_reason_codes": tier_reason_codes,
        "repurposing_signal_score": round(repurposing_signal_score, 3),
        "evidence_completeness_score": evidence_completeness_score,
        "evidence_completeness_components": completeness_components,
        "target_novelty": status,
        "target_status": status,
        "thresholds": {
            "known_drug_repurposing_min": KNOWN_DRUG_REPURPOSING_THRESHOLD,
            "weak_known_drug_min": WEAK_KNOWN_DRUG_THRESHOLD,
            "underexplored_known_drug_max": UNDEREXPLORED_KNOWN_DRUG_MAX,
            "de_novo_association_min": MIN_DE_NOVO_ASSOCIATION_SCORE,
        },
        "confidence": round(confidence, 3),
        "confidence_interpretation": (
            "routing evidence completeness; not biological or clinical success probability"
        ),
        "route_evidence_scope": "target_hypothesis_level",
        "candidate_specific_route_evidence_claimed": False,
        "fallback": {
            "used": fallback_used,
            "reason": fallback_reason,
            "requires_evidence_request": fallback_used,
            "disposition": (
                "mock_low_confidence_fallback"
                if fallback_used and evidence["evidence_mode"] == "mock_fixture"
                else "evidence_request_or_abstain"
                if fallback_used
                else "not_applicable"
            ),
            "abstain_recommended": bool(
                fallback_used and evidence["evidence_mode"] != "mock_fixture"
            ),
            "execution_allowed_in_mock": bool(
                fallback_used and evidence["evidence_mode"] == "mock_fixture"
            ),
        },
        "policy_eligible": True,
        "attestation": "exactly_one_route_selected_from_persisted_ma1_evidence",
    }


def select_hypothesis_for_priority(
    hypotheses: Sequence[Hypothesis],
    priority: str = "auto",
) -> tuple[Hypothesis, dict[str, Any]]:
    """Select an evidence-qualified hypothesis without forcing its MA2 route.

    MA1 rank remains authoritative in ``auto`` mode.  An explicit discovery
    priority searches the ranked portfolio for a hypothesis whose independently
    computed route already matches that priority.  If none qualifies, the
    function falls back to MA1 rank order and records that recovery; a live
    caller can use the persisted abstain recommendation instead of executing it.
    """

    if not hypotheses:
        raise ValueError("hypothesis selection requires at least one hypothesis")
    if priority not in {"auto", "novel_target", "repurposing_target"}:
        raise ValueError(f"unsupported discovery priority: {priority}")

    # A goal-focus marker is written only after an exact target-symbol match to
    # a claim-linked MA1 support record.  It may therefore break an ``auto``
    # tie/order, but cannot introduce a target or make an unsupported target
    # eligible.  Explicit structured discovery priorities remain authoritative.
    ranked = sorted(
        hypotheses,
        key=lambda item: (
            (
                not bool(item.target.metadata.get("goal_focus_primary_candidate"))
                if priority == "auto"
                else False
            ),
            item.rank if item.rank is not None else 1_000_000,
            -item.score,
            item.hypothesis_id,
        ),
    )
    evaluated: list[tuple[Hypothesis, dict[str, Any], bool]] = []
    previews: list[dict[str, Any]] = []
    for hypothesis in ranked:
        decision = select_route_from_evidence(hypothesis, hypothesis.target)
        if priority == "novel_target":
            eligible = bool(
                decision["selected_route"] == CandidateRoute.DE_NOVO.value
                and decision["fallback"]["used"] is False
                and decision["support_evidence_count"] > 0
                and decision["evidence_completeness_score"] >= 0.33
            )
        elif priority == "repurposing_target":
            eligible = bool(
                decision["selected_route"] == CandidateRoute.REPURPOSING.value
                and decision["drug_evidence_tier"] == "strong"
                and decision["fallback"]["used"] is False
                and decision["evidence_completeness_score"] >= 0.33
            )
        else:
            eligible = hypothesis is ranked[0]
        evaluated.append((hypothesis, decision, eligible))
        previews.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "target_symbol": hypothesis.target.symbol,
                "ma1_rank": hypothesis.rank,
                "ma1_score": hypothesis.score,
                "preview_route": decision["selected_route"],
                "drug_evidence_tier": decision["drug_evidence_tier"],
                "fallback_used": decision["fallback"]["used"],
                "evidence_completeness_score": decision[
                    "evidence_completeness_score"
                ],
                "priority_eligible": eligible,
                "goal_focus_primary_candidate": bool(
                    hypothesis.target.metadata.get("goal_focus_primary_candidate")
                ),
            }
        )

    fallback_to_auto = False
    if priority == "auto":
        selected, selected_decision, _ = evaluated[0]
        reason = (
            "EXPLICIT_EVIDENCE_BACKED_TARGET_FOCUS"
            if selected.target.metadata.get("goal_focus_primary_candidate")
            else "MA1_TOP_RANKED_AUTO"
        )
    else:
        match = next((item for item in evaluated if item[2]), None)
        if match is None:
            selected, selected_decision, _ = evaluated[0]
            fallback_to_auto = True
            reason = "NO_EVIDENCE_QUALIFIED_PRIORITY_MATCH_FALLBACK_AUTO"
        else:
            selected, selected_decision, _ = match
            reason = (
                "EVIDENCE_QUALIFIED_UNDEREXPLORED_HYPOTHESIS"
                if priority == "novel_target"
                else "EVIDENCE_QUALIFIED_REPURPOSING_HYPOTHESIS"
            )

    any_external = any(
        decision["evidence_mode"] == "external"
        for _, decision, _ in evaluated
    )
    return selected, {
        "policy_version": HYPOTHESIS_SELECTION_POLICY_VERSION,
        "requested_priority": priority,
        "selected_hypothesis_id": selected.hypothesis_id,
        "selected_target_symbol": selected.target.symbol,
        "selected_ma1_rank": selected.rank,
        "selected_preview_route": selected_decision["selected_route"],
        "selection_reason": reason,
        "fallback_to_auto": fallback_to_auto,
        "evidence_request_required": bool(fallback_to_auto),
        "abstain_recommended_in_live": bool(fallback_to_auto and any_external),
        "considered_hypotheses": previews,
        "attestation": "priority_selects_only_an_independently_route_qualified_hypothesis",
    }


__all__ = [
    "KNOWN_DRUG_REPURPOSING_THRESHOLD",
    "HYPOTHESIS_SELECTION_POLICY_VERSION",
    "MIN_DE_NOVO_ASSOCIATION_SCORE",
    "ROUTE_POLICY_VERSION",
    "UNDEREXPLORED_KNOWN_DRUG_MAX",
    "WEAK_KNOWN_DRUG_THRESHOLD",
    "derive_route_evidence",
    "select_route_from_evidence",
    "select_hypothesis_for_priority",
]
