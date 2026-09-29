from __future__ import annotations

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateCard,
    CandidateMetrics,
    CandidatePacket,
    CandidateRoute,
    EvidenceRef,
    ScientificBackendKind,
    Severity,
    TargetCandidate,
)
from drugpilot.tools.activity_comparability import (
    ASSAY_COMPARABILITY_POLICY,
    CHEMBL_ASSAY_CONTEXT_SCHEMA,
    comparability_summary,
    derive_activity_rankings,
)
from drugpilot.tools.portfolio_ranking import (
    portfolio_components,
    provenance_completeness,
    rank_portfolio,
    target_key,
)


def test_live_score_recalculates_from_exact_disclosed_component_values() -> None:
    card = _card("cand_rounding", name="Rounding", assay_id="ASSAY_A", value=1.0)
    card = card.model_copy(
        update={
            "safety_flags": ["risk_a", "risk_b", "risk_c"],
            "metrics": card.metrics.model_copy(
                update={
                    "docking_score_kcal_mol": None,
                    "synthetic_accessibility": 2.87389,
                    "admet_scores": {
                        "hepatotoxicity": 0.487865,
                        "herg_risk": 0.487865,
                        "ames_risk": 0.487865,
                        "cyp3a4_inhibition": 0.487865,
                        "clearance_risk": 0.487865,
                        "oral_bioavailability": 0.8,
                    },
                }
            ),
        },
        deep=True,
    )

    breakdown = MA2Agent._live_score_breakdown(card)
    numerator = sum(
        item["weight"] * item["value"]
        for item in breakdown["components"]
    )
    denominator = sum(item["weight"] for item in breakdown["components"])
    expected = round(numerator / denominator - 0.12, 6)

    assert abs(breakdown["numerator"] - numerator) <= 1e-9
    assert breakdown["final_score"] == expected


def test_provenance_completeness_gives_no_credit_for_missing_safety_coverage() -> None:
    card = _card("cand_coverage", name="Coverage", assay_id="ASSAY_A", value=1.0)
    card = card.model_copy(
        update={
            "metadata": {
                **card.metadata,
                "regulatory_status": "unknown_no_matching_record",
                "restricted_safety_screen": {"coverage": "partial"},
            }
        },
        deep=True,
    )

    value, details = provenance_completeness(card)

    assert details["dimensions"]["restricted_safety_coverage"] == 0.0
    assert details["dimensions"]["regulatory_status_coverage"] == 0.0
    assert details["missing_coverage_scores_zero"] is True
    assert value < 1.0


def test_portfolio_preserves_single_target_v6_quota_projection_metadata() -> None:
    projection = {
        "schema_version": "paper_native_v6_group_projection_v1",
        "projection_sha256": "sha",
    }
    cards = [
        _card(f"cand_{index}", name=f"Candidate {index}", assay_id=f"A{index}", value=1.0)
        .model_copy(
            update={
                "metadata": {
                    "portfolio_target_key": "uniprot:P12345",
                    "paper_native_v6_group_projection": projection,
                }
            },
            deep=True,
        )
        for index in range(3)
    ]
    source_metadata = {
        "candidate_yield": {
            "requested_eligible_count": 3,
            "paper_native_v6_group_projection": projection,
        },
        "candidate_backfill": {
            "requested_eligible_count": 3,
            "paper_native_v6_group_projection": projection,
        },
    }

    preserved = MA2Agent._portfolio_paper_v5_group_metadata(
        cards,
        {"uniprot:P12345": source_metadata},
    )

    assert preserved == source_metadata


TARGET = TargetCandidate(
    symbol="GENE",
    uniprot_id="P12345",
    metadata={"chembl_target_id": "CHEMBL_TARGET"},
)


def _card(
    candidate_id: str,
    *,
    name: str,
    assay_id: str,
    value: float,
    unit: str = "nM",
    relation: str = "=",
    context_status: str = "completed",
    binding_status: str = "exact",
    description: str = "Identical assay description",
    pchembl: float = 8.0,
) -> CandidateCard:
    evidence_id = f"ev_{candidate_id}"
    record = (
        {
            "assay_chembl_id": assay_id,
            "target_chembl_id": "CHEMBL_TARGET",
            "description": {"excerpt": description},
        }
        if context_status == "completed"
        else None
    )
    evidence = EvidenceRef(
        evidence_id=evidence_id,
        source="ChEMBL",
        external_id=f"activity_{candidate_id}",
        title=f"Measured activity for {name}",
        metadata={
            "target_accession": "P12345",
            "target_gene": "GENE",
            "target_chembl_id": "CHEMBL_TARGET",
            "assay_chembl_id": assay_id,
            "standard_type": "IC50",
            "standard_relation": relation,
            "standard_value": value,
            "standard_units": unit,
            "pchembl_value": pchembl,
            "activity_properties": [],
            "chembl_assay_context": {
                "schema_version": CHEMBL_ASSAY_CONTEXT_SCHEMA,
                "retrieval_status": context_status,
                "binding_status": binding_status,
                "assay_chembl_id": assay_id,
                "record": record,
                "missing_values_imputed": False,
                "inference_used": False,
                "unit_conversion_applied": False,
                "description_similarity_used": False,
            },
        },
    )
    return CandidateCard(
        candidate_id=candidate_id,
        route=CandidateRoute.REPURPOSING,
        name=name,
        target=TARGET,
        mechanism="measured target activity",
        smiles="CCO",
        activities=[
            ActivityMeasurement(
                endpoint="IC50",
                relation=relation,
                value=value,
                unit=unit,
                assay_type="single protein format",
                source_evidence_id=evidence_id,
            )
        ],
        metrics=CandidateMetrics(
            docking_score_kcal_mol=-8.0,
            synthetic_accessibility=3.0,
            admet_scores={
                "hepatotoxicity": 0.2,
                "herg_risk": 0.2,
                "ames_risk": 0.2,
                "cyp3a4_inhibition": 0.2,
                "clearance_risk": 0.2,
                "oral_bioavailability": 0.8,
            },
            ood_score=0.1,
        ),
        evidence=[evidence],
        status="conditional",
    )


def _score_and_order(cards: list[CandidateCard]) -> tuple[dict[str, float], list[str]]:
    rankings = derive_activity_rankings(cards)
    scores = {
        card.candidate_id: MA2Agent._score_live_without_imputation(
            card,
            rankings[card.candidate_id],
        )
        for card in cards
    }
    order = [
        card.candidate_id
        for card in sorted(
            cards,
            key=lambda item: (
                -scores[item.candidate_id],
                item.route.value,
                item.name,
                item.candidate_id,
            ),
        )
    ]
    return scores, order


def test_distinct_assay_values_are_retained_but_score_and_rank_invariant() -> None:
    first = [
        _card("cand_a", name="A", assay_id="ASSAY_A", value=0.16, pchembl=9.8),
        _card("cand_b", name="B", assay_id="ASSAY_B", value=236.0, pchembl=6.6),
    ]
    swapped = [
        _card("cand_a", name="A", assay_id="ASSAY_A", value=236.0, pchembl=6.6),
        _card("cand_b", name="B", assay_id="ASSAY_B", value=0.16, pchembl=9.8),
    ]

    rankings = derive_activity_rankings(first)
    assert all(item["policy"] == ASSAY_COMPARABILITY_POLICY for item in rankings.values())
    assert all(item["measurement_retained"] is True for item in rankings.values())
    assert all(item["comparative_potency"] is None for item in rankings.values())
    assert all(item["comparative_potency_used"] is False for item in rankings.values())
    assert all(item["numeric_value_used_for_ranking"] is False for item in rankings.values())
    assert all(item["pchembl_used_for_ranking"] is False for item in rankings.values())
    assert all("DISTINCT_ASSAY_IDS" in item["reason_codes"] for item in rankings.values())

    first_scores, first_order = _score_and_order(first)
    swapped_scores, swapped_order = _score_and_order(swapped)
    assert first_scores == swapped_scores
    assert first_order == swapped_order == ["cand_a", "cand_b"]


def test_same_exact_assay_allows_only_within_assay_numeric_potency() -> None:
    cards = [
        _card("cand_a", name="A", assay_id="ASSAY_SHARED", value=1.0),
        _card("cand_b", name="B", assay_id="ASSAY_SHARED", value=10.0),
    ]

    rankings = derive_activity_rankings(cards)

    assert rankings["cand_a"]["comparative_potency"] == 1.0
    assert rankings["cand_b"]["comparative_potency"] == 0.0
    assert all(
        item["comparative_potency_used"] is True
        for item in rankings.values()
    )
    assert rankings["cand_a"]["activity_component_value"] == 1.0
    assert rankings["cand_b"]["activity_component_value"] == 0.5
    scores, order = _score_and_order(cards)
    assert scores["cand_a"] > scores["cand_b"]
    assert order == ["cand_a", "cand_b"]


def test_unit_or_relation_mismatch_never_triggers_conversion_or_inference() -> None:
    unit_mismatch = [
        _card("cand_a", name="A", assay_id="ASSAY_SHARED", value=1.0, unit="nM"),
        _card("cand_b", name="B", assay_id="ASSAY_SHARED", value=0.001, unit="uM"),
    ]
    unit_rankings = derive_activity_rankings(unit_mismatch)
    assert all(item["comparative_potency"] is None for item in unit_rankings.values())
    assert all("UNIT_MISMATCH" in item["reason_codes"] for item in unit_rankings.values())
    assert all(item["unit_conversion_applied"] is False for item in unit_rankings.values())

    normalized_source = _card(
        "cand_b",
        name="B",
        assay_id="ASSAY_SHARED",
        value=1.0,
        unit="nM",
    )
    normalized_evidence = normalized_source.evidence[0].model_copy(
        update={
            "metadata": {
                **normalized_source.evidence[0].metadata,
                "standard_units": "nm",
            }
        },
        deep=True,
    )
    normalized_source = normalized_source.model_copy(
        update={"evidence": [normalized_evidence]},
        deep=True,
    )
    explicit_unit_rankings = derive_activity_rankings(
        [
            _card(
                "cand_a",
                name="A",
                assay_id="ASSAY_SHARED",
                value=1.0,
                unit="nM",
            ),
            normalized_source,
        ]
    )
    assert (
        "ACTIVITY_UNIT_BINDING_MISMATCH"
        in explicit_unit_rankings["cand_b"]["reason_codes"]
    )
    assert all(
        item["comparative_potency"] is None
        for item in explicit_unit_rankings.values()
    )

    relation_mismatch = [
        _card("cand_a", name="A", assay_id="ASSAY_SHARED", value=1.0),
        _card(
            "cand_b",
            name="B",
            assay_id="ASSAY_SHARED",
            value=0.5,
            relation="<",
        ),
    ]
    relation_rankings = derive_activity_rankings(relation_mismatch)
    assert all(
        item["comparative_potency"] is None
        for item in relation_rankings.values()
    )
    assert "RELATION_NOT_EXACT" in relation_rankings["cand_b"]["reason_codes"]
    assert all(
        item["missing_value_inference_used"] is False
        for item in relation_rankings.values()
    )


def test_description_similarity_and_partial_context_cannot_authorize_comparison() -> None:
    same_description_distinct_assays = [
        _card("cand_a", name="A", assay_id="ASSAY_A", value=1.0),
        _card("cand_b", name="B", assay_id="ASSAY_B", value=2.0),
    ]
    rankings = derive_activity_rankings(same_description_distinct_assays)
    assert all(item["comparative_potency"] is None for item in rankings.values())
    assert all(item["description_similarity_used"] is False for item in rankings.values())

    partial_context = [
        _card("cand_a", name="A", assay_id="ASSAY_SHARED", value=1.0),
        _card(
            "cand_b",
            name="B",
            assay_id="ASSAY_SHARED",
            value=2.0,
            context_status="partial",
            binding_status="unresolved",
        ),
    ]
    partial_rankings = derive_activity_rankings(partial_context)
    assert partial_rankings["cand_b"]["measurement_retained"] is True
    assert partial_rankings["cand_b"]["comparative_potency"] is None
    assert "ASSAY_CONTEXT_NOT_COMPLETED" in partial_rankings["cand_b"]["reason_codes"]
    assert partial_rankings["cand_a"]["measurement_evidence_quality"] == 1.0
    assert partial_rankings["cand_b"]["measurement_evidence_quality"] == 0.5
    scores, _order = _score_and_order(partial_context)
    assert scores["cand_a"] > scores["cand_b"]


def test_portfolio_rank_is_invariant_to_distinct_assay_value_and_pchembl_swap() -> None:
    first = [
        _card("cand_a", name="A", assay_id="ASSAY_A", value=0.16, pchembl=9.8),
        _card("cand_b", name="B", assay_id="ASSAY_B", value=236.0, pchembl=6.6),
    ]
    swapped = [
        _card("cand_a", name="A", assay_id="ASSAY_A", value=236.0, pchembl=6.6),
        _card("cand_b", name="B", assay_id="ASSAY_B", value=0.16, pchembl=9.8),
    ]
    first = [
        card.model_copy(
            update={
                "target_rank": index,
                "metadata": {
                    **card.metadata,
                    "portfolio_target_key": target_key(card),
                },
            },
            deep=True,
        )
        for index, card in enumerate(first, start=1)
    ]
    swapped = [
        card.model_copy(
            update={
                "target_rank": index,
                "metadata": {
                    **card.metadata,
                    "portfolio_target_key": target_key(card),
                },
            },
            deep=True,
        )
        for index, card in enumerate(swapped, start=1)
    ]
    hypothesis_scores = {target_key(first[0]): 0.8}

    first_ranked = rank_portfolio(first, hypothesis_scores=hypothesis_scores)
    swapped_ranked = rank_portfolio(swapped, hypothesis_scores=hypothesis_scores)

    assert [card.candidate_id for card in first_ranked] == [
        card.candidate_id for card in swapped_ranked
    ]
    assert {
        card.candidate_id: (card.portfolio_score, card.portfolio_rank)
        for card in first_ranked
    } == {
        card.candidate_id: (card.portfolio_score, card.portfolio_rank)
        for card in swapped_ranked
    }
    assert all(
        card.metadata["portfolio_ranking"]["pchembl_used_for_ranking"] is False
        for card in first_ranked
    )


def _auditable_packet(cards: list[CandidateCard]) -> CandidatePacket:
    rankings = derive_activity_rankings(cards)
    scored: list[CandidateCard] = []
    for card in cards:
        ranking = rankings[card.candidate_id]
        breakdown = MA2Agent._live_score_breakdown(card, ranking)
        scored.append(
            card.model_copy(
                update={
                    "score": breakdown["final_score"],
                    "metadata": {
                        **card.metadata,
                        "activity_ranking": ranking,
                        "activity_score_breakdown": breakdown,
                    },
                },
                deep=True,
            )
        )
    return CandidatePacket(
        plan_id="plan_activity_audit",
        candidates=scored,
        backend_kind=ScientificBackendKind.LIVE,
        metadata={"activity_comparability": comparability_summary(rankings)},
    )


def test_duplicate_physical_evidence_rows_do_not_inflate_live_or_portfolio_score() -> None:
    original = _card("cand_unique", name="A", assay_id="ASSAY_A", value=1.0)
    duplicate_row = original.evidence[0].model_copy(
        update={
            "evidence_id": "ev_same_scientific_record_other_row",
            "metadata": {
                **original.evidence[0].metadata,
                "source_row": 99,
            },
        },
        deep=True,
    )
    duplicated = original.model_copy(
        update={"evidence": [*original.evidence, duplicate_row]},
        deep=True,
    )

    original_breakdown = MA2Agent._live_score_breakdown(original)
    duplicate_breakdown = MA2Agent._live_score_breakdown(duplicated)

    assert duplicate_breakdown["final_score"] == original_breakdown["final_score"]
    cardinality = duplicate_breakdown["evidence_cardinality"]
    assert cardinality["policy_version"] == (
        "source-external-id-title-excerpt-support-v1"
    )
    assert cardinality["raw_record_count"] == 2
    assert cardinality["unique_scientific_record_count"] == 1
    assert cardinality["duplicate_physical_row_count"] == 1
    assert len(cardinality["raw_record_identity_sha256s"]) == 2
    assert len(cardinality["unique_scientific_record_identity_sha256s"]) == 1
    assert cardinality["raw_records_preserved"] is True
    assert cardinality["duplicates_contribute_to_score"] is False
    kwargs = {
        "hypothesis_score": 0.8,
        "eligible_count_for_target": 1,
        "target_best_score": original.score,
    }
    assert portfolio_components(duplicated, **kwargs) == portfolio_components(
        original,
        **kwargs,
    )


def test_nearly_tied_within_target_scores_remain_nearly_tied_in_portfolio() -> None:
    cards = [
        _card("cand_a", name="A", assay_id="ASSAY_A", value=1.0),
        _card("cand_b", name="B", assay_id="ASSAY_B", value=2.0),
        _card("cand_c", name="C", assay_id="ASSAY_C", value=3.0),
    ]
    live_scores = (0.685420, 0.684721, 0.679380)
    cards = [
        card.model_copy(
            update={
                "score": score,
                "target_rank": index,
                "metadata": {
                    **card.metadata,
                    "portfolio_target_key": target_key(card),
                },
            },
            deep=True,
        )
        for index, (card, score) in enumerate(
            zip(cards, live_scores, strict=True),
            start=1,
        )
    ]

    ranked = rank_portfolio(
        cards,
        hypothesis_scores={target_key(cards[0]): 0.8},
    )
    factors = {
        card.candidate_id: card.metadata["portfolio_ranking"]["factors"][
            "within_target_rank"
        ]
        for card in ranked
    }

    assert factors["cand_a"] == 1.0
    assert 0.998 < factors["cand_b"] < 1.0
    assert 0.99 < factors["cand_c"] < factors["cand_b"]
    assert max(factors.values()) - min(factors.values()) < 0.02
    assert all(
        card.metadata["portfolio_ranking"]["within_target_normalization"]
        == "continuous_score_ratio_to_target_best"
        for card in ranked
    )


def test_ma3_independently_accepts_exact_policy_and_flags_tampering_high() -> None:
    packet = _auditable_packet(
        [
            _card("cand_a", name="A", assay_id="ASSAY_A", value=1.0),
            _card("cand_b", name="B", assay_id="ASSAY_B", value=2.0),
        ]
    )
    auditor = object.__new__(MA3AssuranceAgent)

    assert auditor._audit_activity_comparability(packet) == []

    original = packet.candidates[0]
    tampered_ranking = {
        **original.metadata["activity_ranking"],
        "comparative_potency": 1.0,
        "comparative_potency_used": True,
        "numeric_value_used_for_ranking": True,
    }
    tampered = original.model_copy(
        update={
            "metadata": {
                **original.metadata,
                "activity_ranking": tampered_ranking,
            }
        },
        deep=True,
    )
    tampered_packet = packet.model_copy(
        update={"candidates": [tampered, packet.candidates[1]]},
        deep=True,
    )

    findings = auditor._audit_activity_comparability(tampered_packet)

    assert any(
        finding.code == "CROSS_ASSAY_ACTIVITY_COMPARISON_POLICY_VIOLATION"
        and finding.severity is Severity.HIGH
        and finding.artifact_id == "cand_a"
        for finding in findings
    )
