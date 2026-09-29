from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from drugpilot.domain.models import CandidateCard
from drugpilot.tools.candidate_evidence_identity import (
    evidence_scoring_cardinality,
)
from drugpilot.tools.activity_comparability import (
    ASSAY_COMPARABILITY_POLICY,
    comparability_summary,
    derive_activity_rankings,
    measurement_evidence_quality,
)


PORTFOLIO_RANKING_VERSION = "target-normalized-assay-governed-v5"
PORTFOLIO_WEIGHTS: dict[str, float] = {
    "target_disease_evidence": 0.35,
    "within_target_rank": 0.25,
    "measured_activity_quality": 0.15,
    "safety": 0.12,
    "applicability": 0.08,
    "provenance_completeness": 0.05,
}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def target_key(card: CandidateCard) -> str:
    target = card.target
    if target.uniprot_id:
        return f"uniprot:{target.uniprot_id.strip().upper()}"
    if target.ensembl_id:
        return f"ensembl:{target.ensembl_id.strip().upper()}"
    return f"symbol:{target.symbol.strip().upper()}"


def provenance_completeness(card: CandidateCard) -> tuple[float, dict[str, Any]]:
    """Score declared provenance dimensions without rewarding missing coverage."""

    typed_results = (
        card.chemistry_execution,
        card.docking_execution,
        card.admet_execution,
    )
    typed_execution = sum(item is not None for item in typed_results) / len(
        typed_results
    )
    scoring_evidence = [
        item for item in card.evidence if item.source and item.retrieved_at
    ]
    evidence_cardinality = evidence_scoring_cardinality(scoring_evidence)
    linked_evidence = int(
        evidence_cardinality["unique_scientific_record_count"]
    )
    evidence_linkage = min(1.0, linked_evidence / 2.0)

    safety_screen = card.metadata.get("restricted_safety_screen")
    safety_coverage = (
        1.0
        if isinstance(safety_screen, Mapping)
        and safety_screen.get("coverage") == "complete"
        else 0.0
    )
    regulatory_status = str(card.metadata.get("regulatory_status") or "")
    regulatory_coverage = (
        1.0
        if regulatory_status
        and not regulatory_status.startswith("unknown")
        and regulatory_status != "not_assessed"
        else 0.0
    )
    dimensions = {
        "typed_execution_provenance": round(_clamp(typed_execution), 6),
        "linked_scientific_evidence": round(_clamp(evidence_linkage), 6),
        "restricted_safety_coverage": safety_coverage,
        "regulatory_status_coverage": regulatory_coverage,
    }
    value = sum(dimensions.values()) / len(dimensions)
    if card.metadata.get("missing_values_imputed") is True:
        value *= 0.5
    details = {
        "schema_version": "provenance_completeness_v2",
        "method": "equal_declared_coverage_dimensions_no_missing_credit_v1",
        "dimensions": dimensions,
        "missing_coverage_scores_zero": True,
        "missing_values_imputed": (
            card.metadata.get("missing_values_imputed") is True
        ),
        "value": round(_clamp(value), 6),
    }
    return details["value"], details


def portfolio_components(
    card: CandidateCard,
    *,
    hypothesis_score: float,
    eligible_count_for_target: int,
    target_best_score: float,
) -> dict[str, float]:
    """Return cross-target-comparable factors without using raw docking values.

    Docking and route-specific scientific outputs may influence ``score``
    inside one target.  Across targets only the candidate-to-target-best score
    ratio is used.  This prevents Vina scores from unrelated receptor boxes
    becoming a false common scale while avoiding the artificial 1.0/0.5/0.0
    gaps produced by an ordinal rank percentile for nearly tied candidates.
    """

    if card.status == "excluded" or card.target_rank is None:
        factors = {key: 0.0 for key in PORTFOLIO_WEIGHTS}
        factors["measured_activity_quality"] = round(
            _clamp(measurement_evidence_quality(card)),
            6,
        )
        return factors

    if eligible_count_for_target <= 1:
        within_target = 1.0
    elif target_best_score > 0.0:
        within_target = float(card.score) / float(target_best_score)
    else:
        # Live candidate scores are bounded to [0, 1].  A target whose entire
        # eligible set scored zero contains no continuous separation signal.
        # Treat the candidates as tied rather than inventing an ordinal gap.
        within_target = 1.0

    activity_quality = measurement_evidence_quality(card)

    risks = card.metrics.admet_scores
    risk_keys = (
        "hepatotoxicity",
        "herg_risk",
        "ames_risk",
        "cyp3a4_inhibition",
        "clearance_risk",
    )
    safety = 0.0
    if all(key in risks for key in risk_keys):
        safety = 1.0 - sum(float(risks[key]) for key in risk_keys) / len(risk_keys)
        safety -= min(0.30, 0.05 * len(card.safety_flags))

    applicability = (
        1.0 - float(card.metrics.ood_score)
        if card.metrics.ood_score is not None
        else 0.0
    )
    provenance, _provenance_details = provenance_completeness(card)

    return {
        "target_disease_evidence": round(_clamp(float(hypothesis_score)), 6),
        "within_target_rank": round(_clamp(within_target), 6),
        "measured_activity_quality": round(_clamp(activity_quality), 6),
        "safety": round(_clamp(safety), 6),
        "applicability": round(_clamp(applicability), 6),
        "provenance_completeness": round(_clamp(provenance), 6),
    }


def portfolio_score(components: Mapping[str, float]) -> float:
    return round(
        _clamp(
            sum(
                PORTFOLIO_WEIGHTS[name] * float(components.get(name, 0.0))
                for name in PORTFOLIO_WEIGHTS
            )
        ),
        6,
    )


def rank_portfolio(
    candidates: Sequence[CandidateCard],
    *,
    hypothesis_scores: Mapping[str, float],
) -> list[CandidateCard]:
    activity_rankings = derive_activity_rankings(candidates)
    activity_summary = comparability_summary(activity_rankings)
    eligible_counts: dict[str, int] = {}
    target_best_scores: dict[str, float] = {}
    for card in candidates:
        key = str(card.metadata.get("portfolio_target_key") or target_key(card))
        if card.status != "excluded" and card.target_rank is not None:
            eligible_counts[key] = eligible_counts.get(key, 0) + 1
            target_best_scores[key] = max(
                target_best_scores.get(key, 0.0),
                float(card.score),
            )

    scored: list[CandidateCard] = []
    for card in candidates:
        activity_ranking = activity_rankings[card.candidate_id]
        key = str(card.metadata.get("portfolio_target_key") or target_key(card))
        factors = portfolio_components(
            card,
            hypothesis_score=float(hypothesis_scores.get(key, 0.0)),
            eligible_count_for_target=eligible_counts.get(key, 0),
            target_best_score=target_best_scores.get(key, 0.0),
        )
        score = None if card.status == "excluded" else portfolio_score(factors)
        scored.append(
            card.model_copy(
                update={
                    "portfolio_score": score,
                    "metadata": {
                        **card.metadata,
                        "portfolio_ranking": {
                            "version": PORTFOLIO_RANKING_VERSION,
                            "factors": factors,
                            "provenance_completeness": (
                                provenance_completeness(card)[1]
                            ),
                            "weights": dict(PORTFOLIO_WEIGHTS),
                            "raw_docking_compared_across_targets": False,
                            "within_target_normalization": (
                                "continuous_score_ratio_to_target_best"
                            ),
                            "target_best_score": round(
                                target_best_scores.get(key, 0.0),
                                6,
                            ),
                            "activity_comparability_policy": (
                                ASSAY_COMPARABILITY_POLICY
                            ),
                            "measured_activity_quality_scope": (
                                "value_independent_measurement_evidence_quality"
                            ),
                            "comparative_potency_directly_scored": False,
                            "comparative_potency_used_within_target_rank": (
                                activity_ranking.get(
                                    "comparative_potency_used"
                                )
                                is True
                            ),
                            "cross_assay_numeric_comparison": False,
                            "pchembl_used_for_ranking": False,
                            "activity_comparability": activity_summary,
                            "evidence_cardinality": evidence_scoring_cardinality(
                                card.evidence
                            ),
                        },
                    },
                },
                deep=True,
            )
        )

    ordered = sorted(
        scored,
        key=lambda card: (
            card.status == "excluded",
            -(card.portfolio_score or 0.0),
            str(card.metadata.get("portfolio_target_key", "")),
            card.target_rank or 10**9,
            card.name.casefold(),
            card.candidate_id,
        ),
    )
    portfolio_rank = 0
    ranked: list[CandidateCard] = []
    for card in ordered:
        rank: int | None = None
        if card.status != "excluded":
            portfolio_rank += 1
            rank = portfolio_rank
        ranked.append(card.model_copy(update={"portfolio_rank": rank}, deep=True))
    return ranked


def ranking_attestation() -> dict[str, Any]:
    return {
        "version": PORTFOLIO_RANKING_VERSION,
        "weights": dict(PORTFOLIO_WEIGHTS),
        "within_target_normalization": (
            "continuous_score_ratio_to_target_best"
        ),
        "cross_target_raw_docking_comparison": False,
        "activity_comparability_policy": ASSAY_COMPARABILITY_POLICY,
        "measured_activity_quality_scope": (
            "value_independent_measurement_evidence_quality"
        ),
        "comparative_potency_directly_scored": False,
        "cross_assay_numeric_comparison": False,
        "unit_conversion_applied": False,
        "pchembl_used_for_ranking": False,
        "missing_value_policy": "explicit_zero_with_provenance_penalty_no_neutral_imputation",
    }
