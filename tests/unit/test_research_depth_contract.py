from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    Claim,
    EvidenceRef,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
    TargetCandidate,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.orchestration.tool_coverage import validate_source_coverage_manifest
from drugpilot.tools.ncbi_rate_limit import NCBIRateLimiter


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_path=tmp_path / "depth.sqlite3",
        evidence_research_depth="comprehensive",
        public_evidence_enabled=True,
        public_evidence_max_records_per_source=8,
        public_evidence_max_source_calls_per_target=5,
        public_evidence_max_elapsed_seconds_per_target=60,
        public_evidence_max_attempts=2,
        dynamic_target_max_candidates=8,
        target_discovery_pubmed_max_records=40,
        target_discovery_max_pubmed_supplements=6,
        target_discovery_target_lookup_limit=8,
        dynamic_context_targets_per_track=2,
        dynamic_context_max_calls=36,
        dynamic_context_max_seconds=90,
        dynamic_context_limit_per_source=5,
        pharos_on_demand_targets_per_track=1,
        public_candidate_enrichment_enabled=True,
        public_candidate_enrichment_per_route=5,
    )


def _brief(depth: str | None) -> ResearchBrief:
    return ResearchBrief(
        brief_id=f"brief_depth_{depth or 'legacy'}",
        research_goal="Identify evidence-linked FLT3 candidates with auditable tools.",
        disease="Acute myeloid leukemia",
        research_depth=depth,
        constraints=ProjectConstraints(
            candidates_per_route=5,
            max_ma1_iterations=2,
            druggability_required=True,
        ),
    )


def _packet(brief: ResearchBrief) -> HypothesisPacket:
    evidence = EvidenceRef(
        evidence_id="ev_depth",
        source="Open Targets",
        external_id="ENSG00000122025",
        title="FLT3 association",
        metadata={"live": True, "provenance_class": "external_live"},
    )
    hypothesis = Hypothesis(
        hypothesis_id="hyp_depth",
        track=HypothesisTrack.LITERATURE,
        disease=brief.disease,
        target=TargetCandidate(
            symbol="FLT3",
            ensembl_id="ENSG00000122025",
            uniprot_id="P36888",
            identity_status="resolved",
        ),
        mechanism="Test FLT3 inhibition.",
        intervention_direction="inhibit",
        falsification_test="Reject after replicated directional failure.",
        claims=[
            Claim(
                text="FLT3 is associated with AML.",
                evidence_ids=[evidence.evidence_id],
                confidence=0.8,
            )
        ],
        evidence=[evidence],
        score=0.8,
        rank=1,
    )
    return HypothesisPacket(brief_id=brief.brief_id, hypotheses=[hypothesis])


def test_depth_profiles_resolve_to_distinct_bounded_contracts(tmp_path) -> None:
    settings = _settings(tmp_path)

    focused = settings.evidence_budget_for("focused")
    standard = settings.evidence_budget_for("standard")
    comprehensive = settings.evidence_budget_for("comprehensive")

    assert (focused.core_records_per_source, standard.core_records_per_source, comprehensive.core_records_per_source) == (2, 5, 8)
    assert (focused.target_pubmed_records, standard.target_pubmed_records, comprehensive.target_pubmed_records) == (10, 25, 40)
    assert (focused.context_source_calls, standard.context_source_calls, comprehensive.context_source_calls) == (8, 20, 36)
    assert (focused.candidate_enrichment_per_route, standard.candidate_enrichment_per_route, comprehensive.candidate_enrichment_per_route) == (1, 3, 5)
    assert (focused.pharos_targets_per_track, standard.pharos_targets_per_track, comprehensive.pharos_targets_per_track) == (0, 1, 1)
    assert settings.evidence_research_depth == "comprehensive"


def test_planner_binds_depth_and_exact_budget_to_plan(tmp_path) -> None:
    settings = _settings(tmp_path)
    brief = _brief("standard")

    plan = PlannerAgent(settings).run(_packet(brief), brief)

    assert plan.research_depth == "standard"
    assert plan.source_coverage is not None
    assert plan.source_coverage.budget.research_depth == "standard"
    assert plan.source_coverage.budget.max_core_records_per_source == 5
    assert plan.source_coverage.budget.target_discovery_pubmed_records == 25
    assert plan.source_coverage.budget.max_context_source_calls == 20
    assert plan.source_coverage.budget.candidate_enrichment_per_route == 3
    assert validate_source_coverage_manifest(
        brief, plan.source_coverage, settings
    ) == []


def test_depth_bound_components_are_run_local_and_do_not_mutate_settings(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    orchestrator = DrugPilotOrchestrator(settings=settings)

    def build(depth: str):
        brief = _brief(depth)
        ma1 = orchestrator._ma1_for_brief(brief)
        context = orchestrator._context_router_for_brief(brief)
        pharos = orchestrator._pharos_router_for_brief(brief)
        ma2 = orchestrator._ma2_for_brief(brief)
        return brief, ma1, context, pharos, ma2

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            focused, standard, comprehensive = list(
                pool.map(build, ("focused", "standard", "comprehensive"))
            )

        focused_public = focused[1].proposer.retriever.public
        standard_public = standard[1].proposer.retriever.public
        comprehensive_public = comprehensive[1].proposer.retriever.public
        assert len({id(focused_public), id(standard_public), id(comprehensive_public)}) == 3
        assert [
            focused_public.max_records_per_source,
            standard_public.max_records_per_source,
            comprehensive_public.max_records_per_source,
        ] == [2, 5, 8]
        assert [
            focused[1].target_discovery.pubmed_max_records,
            standard[1].target_discovery.pubmed_max_records,
            comprehensive[1].target_discovery.pubmed_max_records,
        ] == [10, 25, 40]
        assert len({id(focused[2]), id(standard[2]), id(comprehensive[2])}) == 3
        assert [
            focused[3].max_targets_per_track,
            standard[3].max_targets_per_track,
            comprehensive[3].max_targets_per_track,
        ] == [0, 1, 1]
        assert [
            focused[4].enrichment_per_route,
            standard[4].enrichment_per_route,
            comprehensive[4].enrichment_per_route,
        ] == [1, 3, 5]
        assert len({id(focused[4].gateway), id(standard[4].gateway), id(comprehensive[4].gateway)}) == 3
        assert len({id(focused[4].evidence_enricher), id(standard[4].evidence_enricher), id(comprehensive[4].evidence_enricher)}) == 3
        assert settings.evidence_research_depth == "comprehensive"
        assert orchestrator.ma2.enrichment_per_route == 5
    finally:
        orchestrator.close()


def test_legacy_brief_marker_uses_server_default_without_rewriting_model(
    tmp_path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"evidence_research_depth": "standard"}, deep=True
    )
    legacy = _brief(None)

    resolved = settings.evidence_budget_for(legacy.research_depth)

    assert legacy.research_depth is None
    assert resolved.depth == "standard"


def test_manifest_tampering_cannot_change_run_depth_or_enrichment_budget(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    brief = _brief("focused")
    plan = PlannerAgent(settings).run(_packet(brief), brief)
    assert plan.source_coverage is not None
    tampered_budget = plan.source_coverage.budget.model_copy(
        update={
            "research_depth": "comprehensive",
            "candidate_enrichment_per_route": 5,
        }
    )
    tampered = plan.source_coverage.model_copy(
        update={"budget": tampered_budget}, deep=True
    )

    codes = {
        code
        for code, _ in validate_source_coverage_manifest(brief, tampered, settings)
    }

    assert "SOURCE_COVERAGE_DEPTH_MISMATCH" in codes
    assert "SOURCE_COVERAGE_BUDGET_MISMATCH" in codes


def test_focused_manifest_does_not_admit_zero_budget_pharos(tmp_path) -> None:
    settings = _settings(tmp_path)
    brief = _brief("focused")

    plan = PlannerAgent(settings).run(_packet(brief), brief)

    assert plan.source_coverage is not None
    pharos = next(
        item
        for item in plan.source_coverage.decisions
        if item.source_id == "pharos"
    )
    assert plan.source_coverage.budget.pharos_targets_per_track == 0
    assert pharos.applicable is True
    assert pharos.enabled_by_runtime_policy is False
    assert pharos.admitted is False
    assert pharos.reason_code == "RUNTIME_SOURCE_POLICY_DISABLED"
    assert validate_source_coverage_manifest(
        brief, plan.source_coverage, settings
    ) == []

    tampered_decisions = [item.model_copy(deep=True) for item in plan.source_coverage.decisions]
    index = next(i for i, item in enumerate(tampered_decisions) if item.source_id == "pharos")
    tampered_decisions[index] = tampered_decisions[index].model_copy(
        update={
            "enabled_by_runtime_policy": True,
            "status": "admitted",
            "admitted": True,
        }
    )
    tampered = plan.source_coverage.model_copy(
        update={
            "decisions": tampered_decisions,
            "admitted_source_count": plan.source_coverage.admitted_source_count + 1,
        },
        deep=True,
    )
    codes = {
        code
        for code, _ in validate_source_coverage_manifest(brief, tampered, settings)
    }
    assert "SOURCE_COVERAGE_POLICY_MISMATCH" in codes


def test_process_rate_limiter_spaces_concurrent_ncbi_admissions() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = NCBIRateLimiter(clock=clock, sleeper=sleeper)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(
            pool.map(
                lambda _: limiter.acquire(api_key_configured=False),
                range(3),
            )
        )

    assert sleeps == [0.34, 0.34]
    assert now[0] == 0.68
