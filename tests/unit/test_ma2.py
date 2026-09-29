from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    CandidateRoute,
    EvidenceRef,
    FaultType,
    GateName,
    GateVerdictCode,
    HypothesisPacket,
    PlanSpec,
    ReworkOrder,
    ResearchBrief,
    RunMode,
)
from drugpilot.tools.public_chemistry import SourceEnrichment
from drugpilot.tools.fixture_gateway import DeterministicFixtureGateway
from drugpilot.tools.restricted_candidate_enrichment import (
    RoutedRestrictedCandidateEnricher,
)
from drugpilot.tools.route_policy import _is_atomic_named_drug_label


def test_atomic_named_drug_label_requires_id_for_slash_alias() -> None:
    assert _is_atomic_named_drug_label("BAY 43-9006/Sorafenib") is False
    assert (
        _is_atomic_named_drug_label(
            "BAY 43-9006/Sorafenib",
            identifier_backed=True,
        )
        is True
    )
    assert _is_atomic_named_drug_label("sorafenib plus azacitidine") is False
    assert _is_atomic_named_drug_label("Matching placebo") is False


class StubPublicEnricher:
    def __init__(self) -> None:
        self.requests = []
        self.allowed_sources = []

    def enrich(self, request, *, allowed_sources=None):
        self.requests.append(request)
        self.allowed_sources.append(
            tuple(allowed_sources) if allowed_sources is not None else None
        )
        return SimpleNamespace(
            evidence=[
                EvidenceRef(
                    evidence_id=f"ev_public_{len(self.requests)}",
                    source="PubChem",
                    external_id=f"CID-{len(self.requests)}",
                    title=f"Public identity record for {request.candidate_name}",
                    supports="context",
                    metadata={
                        "provenance_class": "external_live",
                        "live": True,
                        "mock": False,
                        "canonical_smiles": request.smiles,
                    },
                )
            ],
            sources=[
                SourceEnrichment(
                    source="PubChem",
                    status="ok",
                    queried=True,
                    record_count=1,
                )
            ],
        )


class WithdrawnFixtureGateway(DeterministicFixtureGateway):
    def retrieve_repurposing(self, target, *, direction, limit):
        seeds = list(
            super().retrieve_repurposing(target, direction=direction, limit=limit)
        )
        first = seeds[0]
        seeds[0] = replace(
            first,
            metadata={**dict(first.metadata), "regulatory_status": "withdrawn"},
        )
        return seeds


@pytest.fixture
def pipeline(tmp_path, brief: ResearchBrief) -> tuple[ResearchBrief, HypothesisPacket, PlanSpec]:
    settings = Settings(
        mode=RunMode.MOCK,
        database_path=tmp_path / "ma2.sqlite3",
        mock_step_delay_ms=0,
        candidates_per_route=20,
    )
    hypotheses = MA1Agent(settings=settings).run(brief)
    plan = PlannerAgent(settings=settings).run(hypotheses, brief)
    return brief, hypotheses, plan


def test_policy_selected_single_route_candidate_packet_is_complete_and_traceable(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    brief, hypotheses, plan = pipeline

    packet = MA2Agent().run(plan, hypotheses)

    assert len(packet.candidates) == plan.candidates_per_route
    assert sum(card.route is CandidateRoute.REPURPOSING for card in packet.candidates) == 3
    assert sum(card.route is CandidateRoute.DE_NOVO for card in packet.candidates) == 0
    assert [card.rank for card in packet.candidates] == list(range(1, 4))
    assert packet.metadata["route_decision"]["selected_routes"] == ["repurposing"]
    assert packet.metadata["route_decision"]["fallback"]["used"] is True
    assert packet.metadata["route_decision"]["fallback"][
        "requires_evidence_request"
    ] is True
    assert {card.mechanism.split()[0] for card in packet.candidates} == {
        brief.constraints.intervention_direction
    }
    assert all(card.target.symbol == card.target.symbol.upper() for card in packet.candidates)
    assert all(len(card.metrics.admet_scores) == 6 for card in packet.candidates)
    assert all(card.metrics.docking_score_kcal_mol is not None for card in packet.candidates)
    assert all(card.evidence for card in packet.candidates)
    assert all(
        activity.unit in {"pM", "nM", "uM", "mM"}
        for card in packet.candidates
        for activity in card.activities
    )

    call_ids = {call.call_id for call in packet.tool_calls}
    task_ids = {task.task_id for task in plan.tasks}
    assert all(set(card.tool_trace_ids) <= call_ids for card in packet.candidates)
    assert all(call.task_id in task_ids for call in packet.tool_calls)
    assert {call.tool_name for call in packet.tool_calls} <= set(plan.approved_tools)
    assert {"repurposing_search", "docking", "admet_predictor"} <= {
        call.tool_name for call in packet.tool_calls
    }
    assert "de_novo_generator" not in {call.tool_name for call in packet.tool_calls}


def test_public_enrichment_is_bounded_per_route_and_does_not_change_rank(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    enricher = StubPublicEnricher()

    baseline = MA2Agent().run(plan, hypotheses)
    packet = MA2Agent(
        evidence_enricher=enricher,
        enrichment_per_route=1,
    ).run(plan, hypotheses)

    assert len(enricher.requests) == 1
    assert {request.uniprot_id for request in enricher.requests} == {"P36888"}
    assert all(request.inchikey is None for request in enricher.requests)
    assert all(request.chemical_query[0] == "smiles" for request in enricher.requests)
    assert [item.rank for item in packet.candidates] == [item.rank for item in baseline.candidates]
    enriched = [
        item
        for item in packet.candidates
        if any(ref.source == "PubChem" for ref in item.evidence)
    ]
    assert {item.route for item in enriched} == {CandidateRoute.REPURPOSING}
    assert len(packet.metadata["public_enrichment_queries"]) == 1
    assert {
        item["status"] for item in packet.metadata["public_enrichment_queries"]
    } == {"ok"}
    assert all(
        ref.metadata.get("provenance_class") == "external_live"
        for item in enriched
        for ref in item.evidence
        if ref.source == "PubChem"
    )


def test_ma2_propagates_immutable_allowed_sources_to_candidate_enricher(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    enricher = StubPublicEnricher()

    MA2Agent(
        evidence_enricher=enricher,
        enrichment_per_route=1,
    ).run(
        plan,
        hypotheses,
        allowed_sources=["PubChem"],
    )

    assert enricher.allowed_sources == [("PubChem",)]


def test_restricted_sources_are_route_selected_and_missing_configuration_is_explicit(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline

    packet = MA2Agent(
        restricted_evidence_enricher=RoutedRestrictedCandidateEnricher(),
        enrichment_per_route=1,
    ).run(plan, hypotheses)

    queries = packet.metadata["public_enrichment_queries"]
    routed = [item for item in queries if item.get("record_type") == "source_routing_decision"]
    summaries = [
        item
        for item in queries
        if item.get("operation")
        and item.get("record_type") != "source_routing_decision"
    ]
    assert {item["source"] for item in routed} == {
        "DrugCentral",
        "DrugBank",
        "Google Patents",
    }
    assert any(
        item["source"] == "Google Patents"
        and item["selected"] is False
        and item["status"] == "skipped"
        for item in routed
    )
    assert {item["source"] for item in summaries} == {"DrugCentral", "DrugBank"}
    assert {item["status"] for item in summaries} == {"not_configured"}
    assert all(item["queried"] is False for item in summaries)


def test_configurable_twenty_candidate_de_novo_closed_loop_refinement(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    plan = plan.model_copy(update={"candidates_per_route": 20})
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target.tdl = "Tbio"
    primary.target.metadata["association_score"] = 0.72
    primary.target.metadata["evidence_summary"] = {
        "known_drug_score": None,
        "overall_association_score": 0.72,
    }

    packet = MA2Agent(max_refinement_iterations=1).run(plan, hypotheses)

    assert len(packet.candidates) == 20
    assert packet.metadata["route_decision"]["selected_route"] == "de_novo"
    assert packet.generation_iterations == 2
    assert any(call.operation == "refine_candidate" for call in packet.tool_calls)
    assert len({card.smiles for card in packet.candidates if card.route is CandidateRoute.DE_NOVO}) == 20


def test_known_drug_signal_selects_only_repurposing_without_fallback(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target.metadata["association_score"] = 0.84
    primary.target.metadata["evidence_summary"] = {
        "known_drug_score": 0.76,
        "overall_association_score": 0.84,
    }

    packet = MA2Agent().run(plan, hypotheses)
    decision = packet.metadata["route_decision"]

    assert decision["selected_route"] == "repurposing"
    assert decision["target_novelty"] == "repurposing_evidence_strong"
    assert decision["drug_evidence_tier"] == "strong"
    assert decision["fallback"]["used"] is False
    assert decision["known_drug_score"] == 0.76
    assert {card.route for card in packet.candidates} == {CandidateRoute.REPURPOSING}


def test_structured_target_drug_evidence_overrides_null_ot_score_to_repurposing(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    brief, hypotheses, plan = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target.metadata["association_score"] = 0.72
    primary.target.metadata["evidence_summary"] = {
        "known_drug_score": None,
        "overall_association_score": 0.72,
    }

    before = MA2Agent().run(plan, hypotheses)
    assert before.metadata["route_decision"]["selected_route"] == "de_novo"

    primary.evidence.extend(
        [
            EvidenceRef(
                source="DGIdb",
                external_id="hgnc:3765",
                title=f"DGIdb drug-gene context: {primary.target.symbol}",
                supports="context",
                metadata={
                    "query": primary.target.symbol,
                    "interaction_count": 420,
                    "interactions": [
                        {
                            "drug_name": "QUIZARTINIB",
                            "drug_id": "chembl:576982",
                            "interaction_types": [
                                {
                                    "type": "inhibitor",
                                    "directionality": "inhibitory",
                                }
                            ],
                        },
                        {
                            "drug_name": "Ivosidenib",
                            "drug_id": "chembl:3989516",
                            "interaction_types": [
                                {
                                    "type": "inhibitor",
                                    "directionality": "inhibitory",
                                }
                            ],
                        }
                    ],
                },
            ),
            EvidenceRef(
                source="ClinicalTrials.gov",
                external_id="NCT01234567",
                title=f"{primary.target.symbol}-directed AML study",
                supports="context",
                metadata={
                    "query": f'"{primary.target.symbol}" AND "{brief.disease}"',
                    "study_type": "INTERVENTIONAL",
                    "overall_status": "COMPLETED",
                    "phases": ["PHASE2"],
                    "conditions": [brief.disease],
                    "interventions": [
                        {"name": "Ivosidenib", "type": "DRUG"},
                        {"name": "Placebo", "type": "DRUG"},
                        {"name": "Matching placebo", "type": "DRUG"},
                        {"name": "Placebo oral tablet", "type": "DRUG"},
                        {"name": "Chemotherapy", "type": "DRUG"},
                        {
                            "name": "sorafenib, vorinostat and bortezomib",
                            "type": "DRUG",
                        },
                        {"name": "sorafenib plus azacitidine", "type": "DRUG"},
                        {"name": "sorafenib or azacitidine", "type": "DRUG"},
                        {"name": "sorafenib/azacitidine", "type": "DRUG"},
                    ],
                },
            ),
            EvidenceRef(
                source="PubChem",
                external_id="CID-FAKE",
                title="Chemical identity only",
                supports="context",
                metadata={
                    "query": primary.target.symbol,
                    "interaction_count": 999,
                    "drug_name": "MUST-NOT-COUNT",
                },
            ),
        ]
    )

    packet = MA2Agent().run(plan, hypotheses)
    decision = packet.metadata["route_decision"]

    assert decision["selected_route"] == "repurposing"
    assert decision["known_drug_score"] is None
    assert decision["known_drug_count"] == 2
    assert decision["structured_known_drug_count"] == 2
    assert decision["dgidb_interaction_count"] == 420
    assert decision["known_drug_count_basis"] == "observed_unique_named_drugs"
    assert decision["known_drug_sources"] == ["ClinicalTrials.gov", "DGIdb"]
    assert set(decision["known_drug_examples"]) == {"QUIZARTINIB", "Ivosidenib"}
    assert decision["excluded_unqualified_drug_labels"] == [
        "Placebo",
        "Matching placebo",
        "Placebo oral tablet",
        "Chemotherapy",
        "sorafenib, vorinostat and bortezomib",
        "sorafenib plus azacitidine",
        "sorafenib or azacitidine",
        "sorafenib/azacitidine",
    ]
    assert "MUST-NOT-COUNT" not in decision["known_drug_examples"]
    assert decision["drug_evidence_tier"] == "strong"
    assert decision["direction_aligned_overlap"] == ["Ivosidenib"]
    assert decision["late_stage_trial_count"] == 1
    assert decision["fallback"]["used"] is False
    assert {card.route for card in packet.candidates} == {CandidateRoute.REPURPOSING}

    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.PASS


def test_large_unqualified_dgidb_count_does_not_force_repurposing(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    """DGIdb breadth is target context, not disease-specific efficacy evidence."""

    brief, hypotheses, plan = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target.metadata["association_score"] = 0.72
    primary.target.metadata["evidence_summary"] = {
        "known_drug_score": None,
        "overall_association_score": 0.72,
        "tractability": [],
    }
    primary.evidence.append(
        EvidenceRef(
            source="DGIdb",
            external_id="hgnc:underexplored",
            title=f"DGIdb drug-gene context: {primary.target.symbol}",
            supports="context",
            metadata={
                "query": primary.target.symbol,
                "interaction_count": 420,
                "interactions": [
                    {
                        "drug_name": "Unqualified interaction",
                        "drug_id": "chembl:context-only",
                        "interaction_types": [],
                    }
                ],
            },
        )
    )

    packet = MA2Agent().run(plan, hypotheses)
    decision = packet.metadata["route_decision"]

    assert decision["selected_route"] == "de_novo"
    assert decision["drug_evidence_tier"] == "weak"
    assert decision["structured_known_drug_count"] == 1
    assert decision["dgidb_interaction_count"] == 420
    assert decision["fallback"]["used"] is False
    assert {card.route for card in packet.candidates} == {CandidateRoute.DE_NOVO}
    assert not any(call.tool_name == "repurposing_search" for call in packet.tool_calls)
    assert any(call.tool_name == "de_novo_generator" for call in packet.tool_calls)

    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.PASS


def test_live_incomplete_negative_coverage_cannot_justify_de_novo(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target.metadata["target_discovery_source"] = "external_dynamic"
    primary.target.metadata["association_score"] = 0.72
    primary.target.metadata["evidence_summary"] = {
        "known_drug_score": None,
        "overall_association_score": 0.72,
        "tractability": [],
    }
    for evidence in primary.evidence:
        evidence.metadata = {
            **evidence.metadata,
            "mock": False,
            "provenance_class": "external_live",
        }

    packet = MA2Agent().run(plan, hypotheses)
    decision = packet.metadata["route_decision"]

    assert decision["evidence_mode"] == "external"
    assert decision["known_drug_coverage_status"] == "incomplete"
    assert decision["known_drug_source_complete"] is False
    assert decision["selected_route"] == "repurposing"
    assert decision["fallback"]["used"] is True
    assert decision["fallback"]["abstain_recommended"] is True


def test_live_ngly1_observed_empty_dgidb_allows_de_novo_when_trials_are_optional(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target = primary.target.model_copy(update={"symbol": "NGLY1"})
    primary.target.metadata.update(
        {
            "target_discovery_source": "external_dynamic",
            "association_score": 0.72,
            "evidence_summary": {
                "known_drug_score": None,
                "overall_association_score": 0.72,
                "tractability": [],
            },
            "context_source_outcomes": [
                {
                    "source": "ClinicalTrials.gov",
                    "status": "skipped",
                    "queried": False,
                    "record_count": 0,
                }
            ],
        }
    )
    for evidence in primary.evidence:
        evidence.metadata = {
            **evidence.metadata,
            "mock": False,
            "provenance_class": "external_live",
        }
    primary.evidence.append(
        EvidenceRef(
            source="DGIdb",
            external_id="hgnc:empty-observed",
            title="DGIdb drug-gene context: NGLY1",
            supports="context",
            metadata={
                "query": "NGLY1",
                "interaction_count": 0,
                "interactions": [],
                "mock": False,
                "provenance_class": "external_live",
            },
        )
    )

    packet = MA2Agent().run(plan, hypotheses)
    decision = packet.metadata["route_decision"]

    assert decision["dgidb_record_observed"] is True
    assert decision["source_coverage"]["dgidb"] == "empty"
    assert decision["source_coverage"]["clinical_trials"] == "skipped"
    assert decision["known_drug_coverage_status"] == "complete"
    assert decision["target_symbol"] == "NGLY1"
    assert decision["selected_route"] == "de_novo"
    assert decision["fallback"]["used"] is False


def test_late_stage_trial_for_different_drug_is_not_strong_overlap_evidence(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    brief, hypotheses, plan = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target.metadata["association_score"] = 0.72
    primary.target.metadata["evidence_summary"] = {
        "known_drug_score": None,
        "overall_association_score": 0.72,
        "tractability": [],
    }
    primary.evidence.extend(
        [
            EvidenceRef(
                source="DGIdb",
                external_id="hgnc:action-a",
                title=f"DGIdb context for {primary.target.symbol}",
                supports="context",
                metadata={
                    "query": primary.target.symbol,
                    "interaction_count": 1,
                    "interactions": [
                        {
                            "drug_name": "Drug A",
                            "interaction_types": [
                                {
                                    "type": "inhibitor",
                                    "directionality": "inhibitory",
                                }
                            ],
                        }
                    ],
                },
            ),
            EvidenceRef(
                source="ClinicalTrials.gov",
                external_id="NCT-DIFFERENT-DRUG",
                title="Late-stage trial for another compound",
                supports="context",
                metadata={
                    "query": f'"{primary.target.symbol}" AND "{brief.disease}"',
                    "study_type": "INTERVENTIONAL",
                    "overall_status": "COMPLETED",
                    "phases": ["PHASE3"],
                    "conditions": [brief.disease],
                    "interventions": [{"name": "Drug B", "type": "DRUG"}],
                },
            ),
        ]
    )

    packet = MA2Agent().run(plan, hypotheses)
    decision = packet.metadata["route_decision"]

    assert decision["late_stage_trial_count"] == 1
    assert decision["active_trial_count"] == 0
    assert decision["active_or_completed_trial_count"] == 1
    assert decision["direction_aligned_drugs"] == ["Drug A"]
    assert decision["direction_aligned_overlap"] == []
    assert decision["drug_evidence_tier"] == "weak"
    assert "LATE_STAGE_DISEASE_TRIAL_WITH_ALIGNED_TARGET_ACTION" not in decision[
        "tier_reason_codes"
    ]


def test_aml_idh1_approved_small_molecule_context_remains_repurposing(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    brief, hypotheses, plan = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    primary = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]
    primary.target = primary.target.model_copy(
        update={
            "symbol": "IDH1",
            "metadata": {
                **primary.target.metadata,
                "association_score": 0.79,
                "evidence_summary": {
                    "known_drug_score": None,
                    "overall_association_score": 0.79,
                    "tractability": [
                        {
                            "label": "Approved Drug",
                            "modality": "SM",
                            "value": True,
                        }
                    ],
                },
            },
        }
    )
    primary.evidence.append(
        EvidenceRef(
            source="DGIdb",
            external_id="hgnc:5382",
            title="DGIdb drug-gene context: IDH1",
            supports="context",
            metadata={
                "query": "IDH1",
                "interaction_count": 420,
                "interactions": [
                    {"drug_name": "OLAPARIB", "interaction_types": []}
                ],
            },
        )
    )

    packet = MA2Agent().run(plan, hypotheses)
    decision = packet.metadata["route_decision"]

    assert decision["selected_route"] == "repurposing"
    assert decision["drug_evidence_tier"] == "strong"
    assert "APPROVED_TRACTABILITY_WITH_TARGET_CONTEXT" in decision["tier_reason_codes"]
    assert decision["fallback"]["used"] is False
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.PASS


def test_novel_target_priority_selects_lower_ranked_qualified_de_novo_hypothesis(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    brief, hypotheses, _ = pipeline
    hypotheses = hypotheses.model_copy(deep=True)
    ordered = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )
    top, underexplored = ordered[:2]
    top.target.metadata["association_score"] = 0.84
    top.target.metadata["evidence_summary"] = {
        "known_drug_score": 0.76,
        "overall_association_score": 0.84,
        "tractability": [],
    }
    underexplored.target.metadata["association_score"] = 0.72
    underexplored.target.metadata["evidence_summary"] = {
        "known_drug_score": None,
        "overall_association_score": 0.72,
        "tractability": [],
    }
    brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={"discovery_priority": "novel_target"}
            )
        },
        deep=True,
    )
    plan = PlannerAgent().run(hypotheses, brief)

    packet = MA2Agent().run(plan, hypotheses)
    selection = packet.metadata["primary_hypothesis_selection"]

    assert selection["requested_priority"] == "novel_target"
    assert selection["selected_hypothesis_id"] == underexplored.hypothesis_id
    assert selection["selected_hypothesis_id"] != top.hypothesis_id
    assert selection["selected_preview_route"] == "de_novo"
    assert selection["fallback_to_auto"] is False
    assert packet.metadata["route_decision"]["selected_route"] == "de_novo"
    assert {card.target.symbol for card in packet.candidates} == {
        underexplored.target.symbol
    }

    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.PASS


def test_novel_target_priority_falls_back_to_auto_when_only_track_fallbacks_exist(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    brief, hypotheses, _ = pipeline
    brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={"discovery_priority": "novel_target"}
            )
        },
        deep=True,
    )
    plan = PlannerAgent().run(hypotheses, brief)
    top = sorted(
        hypotheses.hypotheses,
        key=lambda item: (item.rank or 1_000_000, -item.score, item.hypothesis_id),
    )[0]

    packet = MA2Agent().run(plan, hypotheses)
    selection = packet.metadata["primary_hypothesis_selection"]

    assert selection["selected_hypothesis_id"] == top.hypothesis_id
    assert selection["fallback_to_auto"] is True
    assert selection["evidence_request_required"] is True
    assert selection["abstain_recommended_in_live"] is False
    assert selection["selection_reason"] == (
        "NO_EVIDENCE_QUALIFIED_PRIORITY_MATCH_FALLBACK_AUTO"
    )

    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.PASS


def test_forbidden_drug_is_excluded_before_docking_and_never_ranked(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    plan = plan.model_copy(update={"forbidden_drugs": ["rp-flt3-001"]})

    packet = MA2Agent().run(plan, hypotheses)
    forbidden = next(card for card in packet.candidates if card.name == "RP-FLT3-001")

    assert forbidden.status == "excluded"
    assert forbidden.rank is None
    assert forbidden.score == 0.0
    assert forbidden.metadata["policy_eligible"] is False
    assert "forbidden_drug" in forbidden.safety_flags
    assert forbidden.exclusion_reasons == ["forbidden_drug_name:rp-flt3-001"]
    assert sum(call.tool_name == "docking" for call in packet.tool_calls) == 2
    assert sum(call.tool_name == "admet_predictor" for call in packet.tool_calls) == 2
    assert packet.metadata["exclusion_summary"]["forbidden_or_withdrawn_count"] == 1


def test_forbidden_registry_id_from_source_evidence_is_screened_precompute(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    plan = plan.model_copy(
        update={"forbidden_drug_ids": ["CHEMBL-FIXTURE-00001"]}
    )

    packet = MA2Agent().run(plan, hypotheses)
    forbidden = next(card for card in packet.candidates if card.name == "RP-FLT3-001")

    assert forbidden.status == "excluded"
    assert forbidden.rank is None
    assert forbidden.exclusion_reasons == [
        "forbidden_drug_id:CHEMBL-FIXTURE-00001"
    ]
    assert forbidden.metadata["source_evidence"][0]["drug_id"] == (
        "CHEMBL-FIXTURE-00001"
    )
    assert sum(call.tool_name == "docking" for call in packet.tool_calls) == 2
    assert sum(call.tool_name == "admet_predictor" for call in packet.tool_calls) == 2


def test_withdrawn_seed_is_automatically_excluded_precompute(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline

    packet = MA2Agent(gateway=WithdrawnFixtureGateway()).run(plan, hypotheses)
    withdrawn = next(card for card in packet.candidates if card.name == "RP-FLT3-001")

    assert withdrawn.status == "excluded"
    assert withdrawn.rank is None
    assert withdrawn.exclusion_reasons == [
        "prohibited_regulatory_status:withdrawn"
    ]
    assert "prohibited_regulatory_status" in withdrawn.safety_flags
    assert sum(call.tool_name == "docking" for call in packet.tool_calls) == 2


@pytest.mark.parametrize(
    ("fault", "assertion"),
    [
        (
            FaultType.G1_IC50_UNIT,
            lambda card, packet, plan: card.activities[0].value > 1_000_000,
        ),
        (
            FaultType.G1_INVALID_SMILES,
            lambda card, packet, plan: card.smiles == "NOT_A_SMILES?",
        ),
        (
            FaultType.G1_DIRECTION_MISMATCH,
            lambda card, packet, plan: not card.mechanism.startswith("inhibit"),
        ),
        (
            FaultType.G1_PLAN_TRACE_MISMATCH,
            lambda card, packet, plan: any(
                call.task_id not in {task.task_id for task in plan.tasks}
                for call in packet.tool_calls
            ),
        ),
    ],
)
def test_supported_g1_fault_injections_are_observable(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
    fault: FaultType,
    assertion,
) -> None:
    _, hypotheses, plan = pipeline

    packet = MA2Agent().run(plan, hypotheses, fault=fault)
    repurposing = next(card for card in packet.candidates if card.route is CandidateRoute.REPURPOSING)

    assert assertion(repurposing, packet, plan)


def test_activity_unit_rework_is_selective_and_reuses_unaffected_artifacts(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    agent = MA2Agent()
    faulty = agent.run(plan, hypotheses, fault=FaultType.G1_IC50_UNIT)
    affected = next(
        card
        for card in faulty.candidates
        if card.route is CandidateRoute.REPURPOSING and card.activities[0].value > 1_000_000
    )
    unaffected = next(card for card in faulty.candidates if card.candidate_id != affected.candidate_id)
    old_calls = {call.call_id: call for call in faulty.tool_calls}
    before_expensive_calls = sum(
        call.tool_name in {"docking", "admet_predictor"} for call in faulty.tool_calls
    )
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.activity_normalization",
        affected_candidate_ids=[affected.candidate_id],
        required_actions=["Correct the IC50 unit and activity scale."],
        preserve_artifact_ids=unaffected.tool_trace_ids,
    )

    repaired = agent.rework(plan, hypotheses, faulty, order)

    fixed = next(card for card in repaired.candidates if card.candidate_id == affected.candidate_id)
    assert fixed.activities[0].unit == "nM"
    assert fixed.activities[0].value < 1_000_000
    assert unaffected.candidate_id in repaired.reused_artifact_ids
    assert set(unaffected.tool_trace_ids) <= set(repaired.reused_artifact_ids)
    assert all(call_id in {call.call_id for call in repaired.tool_calls} for call_id in old_calls)
    after_expensive_calls = sum(
        call.tool_name in {"docking", "admet_predictor"} for call in repaired.tool_calls
    )
    assert after_expensive_calls == before_expensive_calls
    assert any(call.operation == "restore_activity_units" for call in repaired.tool_calls)


def test_invalid_smiles_rework_reruns_only_affected_candidate_evaluation(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    agent = MA2Agent()
    faulty = agent.run(plan, hypotheses, fault=FaultType.G1_INVALID_SMILES)
    affected = next(card for card in faulty.candidates if card.smiles == "NOT_A_SMILES?")
    old_docking = sum(call.tool_name == "docking" for call in faulty.tool_calls)
    old_admet = sum(call.tool_name == "admet_predictor" for call in faulty.tool_calls)
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.standardize",
        affected_candidate_ids=[affected.candidate_id],
        required_actions=["Restore and validate the candidate SMILES structure."],
    )

    repaired = agent.rework(plan, hypotheses, faulty, order)
    fixed = next(card for card in repaired.candidates if card.candidate_id == affected.candidate_id)

    assert fixed.smiles != "NOT_A_SMILES?"
    assert sum(call.tool_name == "docking" for call in repaired.tool_calls) == old_docking + 1
    assert sum(call.tool_name == "admet_predictor" for call in repaired.tool_calls) == old_admet + 1
    assert len(repaired.reused_artifact_ids) >= len(faulty.candidates) - 1


def test_plan_trace_rework_repairs_task_ids_without_rerunning_scientific_tools(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
) -> None:
    _, hypotheses, plan = pipeline
    agent = MA2Agent()
    faulty = agent.run(plan, hypotheses, fault=FaultType.G1_PLAN_TRACE_MISMATCH)
    scientific_call_count = sum(
        call.tool_name != "candidate_judge" for call in faulty.tool_calls
    )
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.tool_dispatch",
        required_actions=["Repair the Plan-Trace task mismatch."],
    )

    repaired = agent.rework(plan, hypotheses, faulty, order)

    valid_task_ids = {task.task_id for task in plan.tasks}
    assert all(call.task_id in valid_task_ids for call in repaired.tool_calls)
    assert sum(call.tool_name != "candidate_judge" for call in repaired.tool_calls) == scientific_call_count
    assert any(call.operation == "repair_plan_trace" for call in repaired.tool_calls)


@pytest.mark.parametrize(
    ("fault", "expected_code"),
    [
        (FaultType.G1_IC50_UNIT, "ACTIVITY_SCALE_OUTLIER"),
        (FaultType.G1_INVALID_SMILES, "INVALID_SMILES"),
        (FaultType.G1_DIRECTION_MISMATCH, "DIRECTION_MISMATCH"),
        (FaultType.G1_PLAN_TRACE_MISMATCH, "UNKNOWN_PLAN_TASK"),
    ],
)
def test_faults_are_detected_by_g1_and_ma2_rework_clears_them(
    pipeline: tuple[ResearchBrief, HypothesisPacket, PlanSpec],
    fault: FaultType,
    expected_code: str,
) -> None:
    brief, hypotheses, plan = pipeline
    ma2 = MA2Agent()
    ma3 = MA3AssuranceAgent()
    faulty = ma2.run(plan, hypotheses, fault=fault)

    verdict = asyncio.run(ma3.audit_g1(brief, hypotheses, plan, faulty))

    assert verdict.verdict is GateVerdictCode.REVISE
    assert expected_code in {finding.code for finding in verdict.findings}
    assert verdict.rework_order is not None

    repaired = ma2.rework(plan, hypotheses, faulty, verdict.rework_order)
    second_verdict = asyncio.run(ma3.audit_g1(brief, hypotheses, plan, repaired))

    assert expected_code not in {finding.code for finding in second_verdict.findings}
    assert second_verdict.verdict is GateVerdictCode.PASS
