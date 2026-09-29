from __future__ import annotations

import pytest
from pydantic import ValidationError

from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    Claim,
    EvidenceRef,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    RunMode,
    TargetCandidate,
)
from drugpilot.orchestration.graph import (
    _COMPREHENSIVE_PHAROS_ROUTING_HINT,
    PlannerNarrativeOverlay,
    _apply_planner_narrative_overlay,
    _planner_execution_contract,
    _research_depth_routing_priorities,
)
from drugpilot.orchestration.goal_focus import derive_goal_focus
from drugpilot.orchestration.tool_coverage import (
    SourceCoverageLLMSuggestion,
    validate_source_coverage_manifest,
)


def _packet(brief_id: str) -> HypothesisPacket:
    evidence = EvidenceRef(
        evidence_id="ev_live_target",
        source="Open Targets",
        external_id="ENSG00000122025",
        title="FLT3 target association",
        metadata={"live": True, "provenance_class": "external_live"},
    )
    hypotheses = []
    for rank, track in enumerate(
        (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO),
        start=1,
    ):
        hypotheses.append(
            Hypothesis(
                hypothesis_id=f"hyp_{track.value}",
                track=track,
                disease="Acute myeloid leukemia",
                target=TargetCandidate(
                    symbol="FLT3",
                    ensembl_id="ENSG00000122025",
                    uniprot_id="P36888",
                    identity_status="resolved",
                ),
                mechanism="Test FLT3 inhibition.",
                intervention_direction="inhibit",
                falsification_test="Reject if two models show no directional effect.",
                claims=[
                    Claim(
                        text="FLT3 is associated with AML.",
                        evidence_ids=[evidence.evidence_id],
                        confidence=0.8,
                    )
                ],
                evidence=[evidence],
                score=0.8,
                rank=rank,
            )
        )
    return HypothesisPacket(brief_id=brief_id, hypotheses=hypotheses)


def _live_settings(tmp_path) -> Settings:
    drugcentral = tmp_path / "drugcentral.tsv"
    drugcentral.write_text("STRUCT_ID\n", encoding="utf-8")
    return Settings(
        mode=RunMode.MOCK,
        database_path=tmp_path / "coverage.sqlite3",
        ma2_backend_mode="live",
        public_evidence_enabled=True,
        dynamic_context_enabled=True,
        dynamic_context_max_calls=30,
        public_candidate_enrichment_enabled=True,
        public_candidate_enrichment_per_route=5,
        drugcentral_dataset_path=drugcentral,
        drugcentral_dataset_version="test",
        cellxgene_service_url="http://cellxgene-sidecar.invalid",
        biogrid_access_key="configured-for-test",
        drugbank_api_key="configured-for-test",
        google_cloud_project="configured-for-test",
    )


def test_planner_admits_every_allowed_applicable_configured_source(
    brief,
    tmp_path,
) -> None:
    settings = _live_settings(tmp_path)
    plan = PlannerAgent(settings=settings).run(_packet(brief.brief_id), brief)
    manifest = plan.source_coverage

    assert manifest is not None
    eligible = [
        item
        for item in manifest.decisions
        if item.allowed_by_brief
        and item.applicable
        and item.configured
        and item.enabled_by_runtime_policy
        and item.adapter is not None
    ]
    assert eligible
    assert all(item.admitted for item in eligible)
    assert manifest.admitted_source_count == len(eligible)
    assert manifest.eligible_coverage_ratio == 1.0
    assert validate_source_coverage_manifest(brief, manifest) == []


def test_llm_suggestions_cannot_add_or_remove_source_admission(
    brief,
    tmp_path,
) -> None:
    settings = _live_settings(tmp_path)
    deterministic = PlannerAgent(settings=settings).run(
        _packet(brief.brief_id),
        brief,
    )
    assert deterministic.source_coverage is not None
    pubmed = next(
        item
        for item in deterministic.source_coverage.decisions
        if item.source_name == "PubMed"
    )
    assert pubmed.admitted

    restricted_brief = brief.model_copy(
        update={
            "data_policy": brief.data_policy.model_copy(
                update={
                    "allowed_sources": [
                        item
                        for item in brief.data_policy.allowed_sources
                        if item != "Google Patents"
                    ]
                }
            )
        },
        deep=True,
    )
    deterministic = PlannerAgent(settings=settings).run(
        _packet(restricted_brief.brief_id),
        restricted_brief,
    )
    overlay = PlannerNarrativeOverlay(
        source_suggestions=[
            SourceCoverageLLMSuggestion(
                source_name="PubMed",
                recommend_use=False,
                priority="low",
                rationale="The model would deprioritize it.",
            ),
            SourceCoverageLLMSuggestion(
                source_name="Google Patents",
                recommend_use=True,
                priority="high",
                rationale="The model wants a prior-art search.",
            ),
            SourceCoverageLLMSuggestion(
                source_name="Invented Science Tool",
                recommend_use=True,
                priority="high",
                rationale="This source is not registered.",
            ),
        ]
    )

    merged = _apply_planner_narrative_overlay(deterministic, overlay)

    assert _planner_execution_contract(merged) == _planner_execution_contract(
        deterministic
    )
    assert merged.source_coverage is not None
    merged_pubmed = next(
        item
        for item in merged.source_coverage.decisions
        if item.source_name == "PubMed"
    )
    patents = next(
        item
        for item in merged.source_coverage.decisions
        if item.source_name == "Google Patents"
    )
    assert merged_pubmed.admitted
    assert merged_pubmed.llm_recommendation == "not_recommended"
    assert merged_pubmed.llm_guard_note
    assert not patents.admitted
    assert patents.reason_code == "NOT_IN_RESEARCH_BRIEF_ALLOW_LIST"
    assert patents.llm_guard_note
    rejection_codes = {
        item.reason_code for item in merged.source_coverage.llm_rejected_suggestions
    }
    assert "NOT_IN_RESEARCH_BRIEF_ALLOW_LIST" in rejection_codes
    assert "UNKNOWN_SOURCE_SUGGESTION" in rejection_codes
    assert validate_source_coverage_manifest(restricted_brief, merged.source_coverage) == []


def test_coverage_overlay_schema_rejects_permissions_and_numeric_limits() -> None:
    with pytest.raises(ValidationError):
        PlannerNarrativeOverlay.model_validate(
            {
                "tasks": [],
                "source_suggestions": [
                    {
                        "source_name": "PubMed",
                        "recommend_use": True,
                        "priority": "high",
                        "rationale": "Search broadly.",
                        "max_records": 10_000,
                        "allowed_operations": ["arbitrary_operation"],
                    }
                ],
            }
        )


def test_manifest_validator_detects_an_eligible_coverage_gap(
    brief,
    tmp_path,
) -> None:
    plan = PlannerAgent(settings=_live_settings(tmp_path)).run(
        _packet(brief.brief_id),
        brief,
    )
    assert plan.source_coverage is not None
    decisions = [item.model_copy(deep=True) for item in plan.source_coverage.decisions]
    index = next(index for index, item in enumerate(decisions) if item.admitted)
    decisions[index] = decisions[index].model_copy(
        update={
            "status": "policy_excluded",
            "admitted": False,
            "reason_code": "TAMPERED",
            "reason": "Tampered after deterministic planning.",
        }
    )
    tampered = plan.source_coverage.model_copy(update={"decisions": decisions})

    codes = {code for code, _ in validate_source_coverage_manifest(brief, tampered)}
    assert "SOURCE_COVERAGE_GAP" in codes
    assert "SOURCE_COVERAGE_COUNT_MISMATCH" in codes


def test_comprehensive_profile_routes_applicable_pharos_without_llm_authority(
    brief,
    tmp_path,
) -> None:
    comprehensive = _live_settings(tmp_path)
    priorities = _research_depth_routing_priorities(
        brief,
        derive_goal_focus(brief),
        comprehensive,
    )

    assert _COMPREHENSIVE_PHAROS_ROUTING_HINT in priorities

    focused = comprehensive.model_copy(update={"evidence_research_depth": "focused"})
    focused_priorities = _research_depth_routing_priorities(
        brief,
        derive_goal_focus(brief),
        focused,
    )
    assert _COMPREHENSIVE_PHAROS_ROUTING_HINT not in focused_priorities
