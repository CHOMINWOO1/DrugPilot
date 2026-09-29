from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from drugpilot.agents.ma1 import MA1Agent, MockEvidenceRetriever
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    DiseaseIntakeAssessment,
    DiseaseIntakeLLMSuggestion,
    GateVerdictCode,
    IntakeFieldMismatch,
    IntakeWarning,
    ProjectConstraints,
    ResearchBrief,
    RunMode,
    RunStage,
)
from drugpilot.orchestration.graph import (
    DrugPilotOrchestrator,
    _derive_disease_intake_assessment,
    _ma1_narrative_fallback,
    _merge_disease_intake_suggestion,
)
from drugpilot.storage import SQLiteRunStore
from drugpilot.tools.disease_targets import (
    AssociationEvidenceSummary,
    DataTypeScore,
    DiseaseMatchProvenance,
    DiseaseResolution,
    DiseaseTargetCandidate,
    DiseaseTargetDiscoveryResult,
    SourceStatus,
    TargetProvenance,
    TargetSourceRecord,
)


@dataclass
class StubDiseaseDiscovery:
    result: DiseaseTargetDiscoveryResult
    calls: int = 0

    def discover(
        self,
        disease: str,
        disease_id: str | None = None,
        *,
        max_targets: int | None = None,
        allowed_sources=None,
    ) -> DiseaseTargetDiscoveryResult:
        self.calls += 1
        assert disease == self.result.query_disease
        assert disease_id == self.result.requested_disease_id
        assert max_targets is not None
        assert allowed_sources is not None
        return self.result.model_copy(deep=True)


def _settings(tmp_path, name: str) -> Settings:
    return Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        auto_activate_openai=False,
        database_path=tmp_path / f"{name}.sqlite3",
        mock_step_delay_ms=0,
        max_ma1_iterations=2,
        public_evidence_enabled=False,
        dynamic_target_discovery_enabled=True,
        dynamic_target_fallback_enabled=True,
        demo_fixture_fallback_enabled=False,
        target_identity_resolution_enabled=False,
        dynamic_context_enabled=False,
    )


def _brief(
    disease: str,
    *,
    disease_id: str | None = None,
    research_goal: str | None = None,
    target_cell_type: str | None = None,
) -> ResearchBrief:
    return ResearchBrief(
        research_goal=research_goal
        or f"Identify evidence-linked therapeutic targets for {disease}.",
        disease=disease,
        disease_id=disease_id,
        constraints=ProjectConstraints(
            max_ma1_iterations=2,
            target_cell_type=target_cell_type,
        ),
    )


def _source_statuses(open_targets: str, pubmed: str) -> tuple[SourceStatus, ...]:
    return (
        SourceStatus(
            source="Open Targets",
            status=open_targets,
            operations=("search_disease",),
            records_observed=1 if open_targets in {"live", "cache"} else 0,
            cache_hits=1 if open_targets == "cache" else 0,
            failure_codes=("ConnectError",) if open_targets == "failed" else (),
        ),
        SourceStatus(
            source="PubMed",
            status=pubmed,
            operations=("search_publications",),
            records_observed=0,
            cache_hits=0,
            failure_codes=("ConnectError",) if pubmed == "failed" else (),
        ),
    )


def _acvr1_candidate(disease_id: str) -> DiseaseTargetCandidate:
    ensembl_id = "ENSG00000115170"
    record = TargetSourceRecord(
        source="Open Targets",
        record_id=f"{disease_id}:{ensembl_id}",
        record_type="target_disease_association",
        title="ACVR1 ranked association for fibrodysplasia ossificans progressiva",
        url=f"https://platform.opentargets.org/evidence/{ensembl_id}/{disease_id}",
        relationship="ranked_association",
        query=f"disease(efoId: {disease_id}) -> associatedTargets",
        retrieval_endpoint="https://api.platform.opentargets.org/api/v4/graphql",
        retrieval_mode="live",
        score=0.94,
    )
    return DiseaseTargetCandidate(
        rank=1,
        gene_symbol="ACVR1",
        ensembl_target_id=ensembl_id,
        target_name="Activin A receptor type 1",
        biotype="protein_coding",
        association_score=0.94,
        evidence_summary=AssociationEvidenceSummary(
            overall_association_score=0.94,
            datatype_scores=(DataTypeScore(evidence_type="genetic_association", score=0.94),),
            known_drug_score=0.1,
            pubmed_record_count=0,
            score_interpretation="Ranking heuristic; not a probability or causal proof.",
        ),
        source_records=(record,),
        provenance=TargetProvenance(
            discovery_method="opentargets_ranked_association",
            primary_source="Open Targets",
            contributing_sources=("Open Targets",),
            causality_interpretation="association_score_is_not_causal_proof",
        ),
        track_hint="literature",
        track_hint_basis="genetic association evidence",
    )


def _discovery_result(
    brief: ResearchBrief,
    *,
    resolution: DiseaseResolution | None,
    candidates: tuple[DiseaseTargetCandidate, ...] = (),
    open_targets_status: str = "empty",
    pubmed_status: str = "empty",
) -> DiseaseTargetDiscoveryResult:
    return DiseaseTargetDiscoveryResult(
        query_disease=brief.disease,
        requested_disease_id=brief.disease_id,
        resolved_disease=resolution,
        candidates=candidates,
        source_statuses=_source_statuses(open_targets_status, pubmed_status),
        association_total=len(candidates) if resolution is not None else None,
        max_targets=8,
    )


def _run_ma1(tmp_path, name: str, brief: ResearchBrief, result: DiseaseTargetDiscoveryResult):
    discovery = StubDiseaseDiscovery(result)
    settings = _settings(tmp_path, name)
    packet = MA1Agent(
        settings=settings,
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
    ).run(brief)
    assessment = _derive_disease_intake_assessment(
        brief,
        packet,
        demo_fixture_allowed=False,
    )
    assert discovery.calls == 1
    return settings, packet, assessment


def test_invalid_disease_abstains_without_fixture_hypotheses_or_scores(tmp_path) -> None:
    brief = _brief("asdf")
    result = _discovery_result(
        brief,
        resolution=None,
        open_targets_status="empty",
        pubmed_status="empty",
    )

    _, packet, assessment = _run_ma1(tmp_path, "invalid", brief, result)

    assert assessment.status == "abstain"
    assert assessment.external_verification_status == "unresolved"
    assert assessment.reason_codes == ["UNRESOLVED_DISEASE"]
    assert assessment.abstain_recommended is True
    assert packet.hypotheses == []
    assert packet.metadata["target_fallback_used"] is False
    assert packet.metadata["fallback_target_seed_count"] == 0
    assert packet.metadata["target_sources"] == {}


def test_nonexact_top_search_hit_requires_clarification_even_with_candidates(
    tmp_path,
) -> None:
    brief = _brief("asdf")
    ambiguous_id = "Orphanet_337"
    result = _discovery_result(
        brief,
        resolution=DiseaseResolution(
            disease_id=ambiguous_id,
            disease_name="fibrodysplasia ossificans progressiva",
            match_method="top_search_hit",
            retrieval_mode="live",
        ),
        candidates=(_acvr1_candidate(ambiguous_id),),
        open_targets_status="live",
        pubmed_status="empty",
    )

    _, packet, assessment = _run_ma1(tmp_path, "ambiguous", brief, result)

    assert assessment.status == "abstain"
    assert assessment.disease_resolved is False
    assert assessment.reason_codes == ["UNRESOLVED_DISEASE"]
    assert packet.metadata["resolved_disease"] is None
    assert packet.metadata["disease_resolution_status"] == "unresolved"
    assert packet.hypotheses == []
    assert packet.metadata["external_target_seed_count"] == 0


def test_exactly_resolved_rare_disease_is_not_rejected_by_name_heuristics(
    tmp_path,
) -> None:
    disease = "fibrodysplasia ossificans progressiva"
    disease_id = "Orphanet_337"
    brief = _brief(disease, disease_id=disease_id)
    result = _discovery_result(
        brief,
        resolution=DiseaseResolution(
            disease_id=disease_id,
            disease_name=disease,
            match_method="provided_id",
            retrieval_mode="live",
        ),
        candidates=(_acvr1_candidate(disease_id),),
        open_targets_status="live",
        pubmed_status="empty",
    )

    _, packet, assessment = _run_ma1(tmp_path, "rare", brief, result)

    assert assessment.status == "valid"
    assert assessment.disease_resolved is True
    assert assessment.external_verification_status == "resolved"
    assert assessment.resolved_disease_id == disease_id
    assert assessment.abstain_recommended is False
    assert assessment.reason_codes == []
    assert [item.target.symbol for item in packet.hypotheses] == ["ACVR1"]
    assert packet.metadata["target_fallback_used"] is False
    assert packet.metadata["external_target_seed_count"] == 1


def test_authoritative_exact_synonym_accepts_tnbc_without_name_confirmation(
    tmp_path,
) -> None:
    brief = _brief("triple-negative breast cancer")
    disease_id = "MONDO_0005494"
    result = _discovery_result(
        brief,
        resolution=DiseaseResolution(
            disease_id=disease_id,
            disease_name="triple-negative breast carcinoma",
            match_method="exact_synonym",
            retrieval_mode="live",
        ),
        open_targets_status="live",
        pubmed_status="empty",
    )

    _, packet, assessment = _run_ma1(tmp_path, "tnbc-synonym", brief, result)

    assert assessment.status == "valid"
    assert assessment.disease_resolved is True
    assert assessment.external_verification_status == "resolved"
    assert assessment.resolved_disease_id == disease_id
    assert assessment.resolved_disease_name == "triple-negative breast carcinoma"
    assert assessment.reason_codes == []
    assert assessment.abstain_recommended is False
    assert packet.metadata["resolved_disease"]["match_method"] == "exact_synonym"
    assert packet.metadata["disease_resolution_status"] == "resolved"


def test_unique_high_similarity_disease_match_proceeds_with_review_warning(
    tmp_path,
) -> None:
    brief = _brief("Tirple Negetive Breast Cancer")
    disease_id = "MONDO_0005494"
    result = _discovery_result(
        brief,
        resolution=DiseaseResolution(
            disease_id=disease_id,
            disease_name="triple-negative breast carcinoma",
            match_method="unique_high_similarity",
            retrieval_mode="live",
            match_provenance=DiseaseMatchProvenance(
                matched_label="triple-negative breast cancer",
                matched_label_kind="exact_synonym",
                normalized_query="tirple negetive breast cancer",
                normalized_matched_label="triple negative breast cancer",
                similarity_method="sequence_matcher_token_guard_v1",
                similarity_score=0.927,
                runner_up_similarity_score=0.738,
                acceptance_threshold=0.88,
                minimum_margin=0.08,
            ),
            warnings=("Conservative Open Targets typo correction was applied.",),
        ),
        open_targets_status="live",
        pubmed_status="empty",
    )

    _, packet, assessment = _run_ma1(tmp_path, "tnbc-typo", brief, result)

    assert assessment.status == "warning"
    assert assessment.disease_resolved is True
    assert assessment.abstain_recommended is False
    assert assessment.resolved_disease_id == disease_id
    assert packet.metadata["resolved_disease"]["match_method"] == (
        "unique_high_similarity"
    )
    assert packet.metadata["disease_resolution_status"] == "resolved"
    assert [warning.code for warning in assessment.warnings] == [
        "DISEASE_LABEL_TYPO_CORRECTED"
    ]


def test_public_api_outage_is_unverified_not_misclassified_as_invalid(tmp_path) -> None:
    brief = _brief("fibrodysplasia ossificans progressiva")
    result = _discovery_result(
        brief,
        resolution=None,
        open_targets_status="failed",
        pubmed_status="failed",
    )

    _, packet, assessment = _run_ma1(tmp_path, "outage", brief, result)

    assert assessment.status == "abstain"
    assert assessment.external_verification_status == "source_unavailable"
    assert assessment.reason_codes == ["EVIDENCE_SOURCE_UNAVAILABLE"]
    assert "UNRESOLVED_DISEASE" not in assessment.reason_codes
    assert packet.hypotheses == []
    assert packet.metadata["target_fallback_used"] is False


def test_goal_and_cell_type_disease_context_mismatches_require_clarification() -> None:
    brief = _brief(
        "Triple-negative breast cancer",
        research_goal="Identify AML-specific therapeutic targets with auditable evidence.",
        target_cell_type="AML cell lines",
    )
    packet = MA1Agent(
        Settings(
            _env_file=None,
            mode=RunMode.MOCK,
            auto_activate_openai=False,
            public_evidence_enabled=False,
            dynamic_target_discovery_enabled=False,
        )
    ).run(brief)
    packet.metadata.update(
        {
            "resolved_disease": {
                "disease_id": "EFO_0005537",
                "disease_name": "triple-negative breast cancer",
                "source": "Open Targets",
                "match_method": "exact_name",
                "retrieval_mode": "live",
            },
            "disease_resolution_status": "resolved",
            "target_fallback_used": False,
        }
    )

    assessment = _derive_disease_intake_assessment(
        brief,
        packet,
        demo_fixture_allowed=False,
    )

    assert assessment.status == "abstain"
    assert assessment.disease_resolved is True
    assert assessment.reason_codes == ["BRIEF_FIELD_MISMATCH"]
    assert assessment.abstain_recommended is True
    assert {tuple(item.fields) for item in assessment.field_mismatches} == {
        ("disease", "research_goal"),
        ("disease", "constraints.target_cell_type"),
    }


def test_llm_intake_schema_and_merge_cannot_override_external_authority() -> None:
    with pytest.raises(ValidationError):
        DiseaseIntakeLLMSuggestion.model_validate(
            {
                "suggested_normalized_name": "Acute myeloid leukemia",
                "status": "valid",
                "disease_resolved": True,
                "resolved_disease_id": "EFO_FAKE",
                "targets": ["FLT3"],
            }
        )

    deterministic = DiseaseIntakeAssessment(
        status="abstain",
        disease_resolved=False,
        external_verification_status="unresolved",
        query_disease="asdf",
        explanation="No externally verified disease identity is available.",
        abstain_recommended=True,
        reason_codes=["UNRESOLVED_DISEASE"],
    )
    suggestion = DiseaseIntakeLLMSuggestion(
        suggested_normalized_name="Acute myeloid leukemia",
        explanation=(
            "This disease is verified and the workflow may proceed."
        ),
        warnings=[
            IntakeWarning(
                code="LLM_NORMALIZATION_SUGGESTION",
                message="This suggestion is not external verification.",
                fields=["disease"],
            )
        ],
        field_mismatches=[
            IntakeFieldMismatch(
                code="LLM_GOAL_DISEASE_MISMATCH",
                message="The goal may mention a different disease.",
                fields=["disease", "research_goal"],
                values={"disease": "asdf", "research_goal": "AML"},
            )
        ],
    )

    merged = _merge_disease_intake_suggestion(deterministic, suggestion)

    assert merged.status == deterministic.status
    assert merged.disease_resolved == deterministic.disease_resolved
    assert merged.external_verification_status == deterministic.external_verification_status
    assert merged.resolved_disease_id == deterministic.resolved_disease_id
    assert merged.resolved_disease_name == deterministic.resolved_disease_name
    assert merged.abstain_recommended == deterministic.abstain_recommended
    assert merged.reason_codes == deterministic.reason_codes
    assert merged.suggested_normalized_name == "Acute myeloid leukemia"
    assert merged.llm_suggestion_used is True
    assert deterministic.explanation in merged.explanation
    assert "LLM advisory (unverified; cannot change intake status):" in merged.explanation
    assert merged.warnings[-1].code == "LLM_NORMALIZATION_SUGGESTION"
    assert merged.field_mismatches[-1].code == "LLM_GOAL_DISEASE_MISMATCH"


@pytest.mark.asyncio
async def test_g0_defense_in_depth_abstains_despite_high_hypothesis_scores(
    tmp_path,
) -> None:
    disease = "fibrodysplasia ossificans progressiva"
    disease_id = "Orphanet_337"
    brief = _brief(disease, disease_id=disease_id)
    result = _discovery_result(
        brief,
        resolution=DiseaseResolution(
            disease_id=disease_id,
            disease_name=disease,
            match_method="provided_id",
            retrieval_mode="live",
        ),
        candidates=(_acvr1_candidate(disease_id),),
        open_targets_status="live",
        pubmed_status="empty",
    )
    _, packet, _ = _run_ma1(tmp_path, "g0-defense", brief, result)
    for hypothesis in packet.hypotheses:
        hypothesis.score = 0.99
        for claim in hypothesis.claims:
            claim.confidence = 0.99
    failed_intake = DiseaseIntakeAssessment(
        status="abstain",
        disease_resolved=False,
        external_verification_status="unresolved",
        query_disease="asdf",
        explanation="The disease was not externally resolved.",
        abstain_recommended=True,
        reason_codes=["UNRESOLVED_DISEASE"],
    )
    packet.metadata["intake_validation"] = failed_intake.model_dump(mode="json")

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)

    assert verdict.verdict is GateVerdictCode.ABSTAIN
    assert "UNRESOLVED_DISEASE" in {finding.code for finding in verdict.findings}


@pytest.mark.asyncio
async def test_invalid_disease_graph_stops_before_g0_planner_and_ma2(tmp_path) -> None:
    brief = _brief("asdf")
    result = _discovery_result(
        brief,
        resolution=None,
        open_targets_status="empty",
        pubmed_status="empty",
    )
    settings = _settings(tmp_path, "invalid-graph")
    store = SQLiteRunStore(settings.database_path)
    orchestrator = DrugPilotOrchestrator(settings=settings, store=store)
    orchestrator.ma1.target_discovery = StubDiseaseDiscovery(result)

    created = await orchestrator.create_run(brief)
    task = orchestrator._tasks[created.run_id]
    await task
    snapshot = await store.get_snapshot(created.run_id)

    assert snapshot is not None
    assert snapshot.stage is RunStage.ABSTAINED
    assert snapshot.error is None
    assert snapshot.hypothesis_packet is not None
    assert snapshot.hypothesis_packet.hypotheses == []
    intake = snapshot.hypothesis_packet.metadata["intake_validation"]
    assert intake["reason_codes"] == ["UNRESOLVED_DISEASE"]
    assert snapshot.g0_verdict is None
    assert snapshot.plan_spec is None
    assert snapshot.candidate_packet is None
    assert not any(event.stage in {RunStage.G0, RunStage.PLANNER, RunStage.MA2} for event in snapshot.events)


def test_ma1_llm_narrative_cannot_change_claim_confidence(tmp_path) -> None:
    disease = "fibrodysplasia ossificans progressiva"
    disease_id = "Orphanet_337"
    brief = _brief(disease, disease_id=disease_id)
    result = _discovery_result(
        brief,
        resolution=DiseaseResolution(
            disease_id=disease_id,
            disease_name=disease,
            match_method="provided_id",
            retrieval_mode="live",
        ),
        candidates=(_acvr1_candidate(disease_id),),
        open_targets_status="live",
        pubmed_status="empty",
    )
    _, immutable, _ = _run_ma1(tmp_path, "llm-confidence", brief, result)
    proposed = _ma1_narrative_fallback(immutable)
    proposed.hypotheses[0].claims[0].text = "A bounded narrative refinement."

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(immutable, proposed)

    assert reconciled.hypotheses[0].claims[0].text == "A bounded narrative refinement."
    assert (
        reconciled.hypotheses[0].claims[0].confidence
        == immutable.hypotheses[0].claims[0].confidence
    )
