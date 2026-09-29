from __future__ import annotations

from typing import Any

from drugpilot.agents.planner import PlannerAgent
from drugpilot.domain.models import (
    Claim,
    EvidenceRef,
    ExcludedHypothesis,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
    TargetCandidate,
)
from drugpilot.orchestration.goal_focus import (
    apply_goal_focus,
    derive_goal_focus,
    routing_priorities_for_focus,
)
from drugpilot.tools.context_router import (
    ContextRoutingBudget,
    DiseaseTargetContextRouter,
    context_request_from_hypothesis_packet,
)
from drugpilot.tools.public_context import ContextSourceResult, MA1ContextResult
from drugpilot.tools.route_policy import select_hypothesis_for_priority


class RecordingContextRetriever:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def lookup(self, request: Any) -> MA1ContextResult:
        source = request.sources[0]
        self.sources.append(source)
        return MA1ContextResult(
            lookup=request,
            evidence=[],
            sources=[
                ContextSourceResult(
                    source=source,
                    status="empty",
                    queried=True,
                    record_count=0,
                )
            ],
        )


def _brief(goal: str, *, priority: str = "auto") -> ResearchBrief:
    return ResearchBrief(
        brief_id="brief-goal-focus",
        research_goal=goal,
        disease="test disease",
        constraints=ProjectConstraints(
            discovery_priority=priority,
            max_ma1_iterations=1,
        ),
    )


def _hypothesis(symbol: str, rank: int, *, linked: bool = True) -> Hypothesis:
    evidence = EvidenceRef(
        source="PubMed",
        external_id=f"pmid-{symbol}",
        title=f"Evidence for {symbol}",
        supports="support",
    )
    return Hypothesis(
        hypothesis_id=f"hyp-{symbol}",
        track=HypothesisTrack.LITERATURE,
        disease="test disease",
        target=TargetCandidate(symbol=symbol),
        mechanism="Evidence-derived test mechanism",
        intervention_direction="inhibit",
        falsification_test="Reject after a prespecified negative result.",
        claims=[
            Claim(
                text=f"{symbol} is testable",
                evidence_ids=[evidence.evidence_id] if linked else [],
            )
        ],
        evidence=[evidence],
        score=0.9 - rank / 10,
        rank=rank,
    )


def _packet() -> HypothesisPacket:
    return HypothesisPacket(
        packet_id="hp-goal-focus",
        brief_id="brief-goal-focus",
        hypotheses=[_hypothesis("ABL1", 1), _hypothesis("TP53", 2)],
    )


def _executable_scope_packet(*, flt3_linked: bool = True) -> HypothesisPacket:
    return HypothesisPacket(
        packet_id="hp-executable-scope",
        brief_id="brief-goal-focus",
        hypotheses=[
            _hypothesis("FLT3", 1, linked=flt3_linked),
            _hypothesis("SRSF2", 2),
        ],
        metadata={
            "target_sources": {
                "FLT3": {"target_source": "external_dynamic"},
                "SRSF2": {"target_source": "external_dynamic"},
            }
        },
    )


def _routed_sources(goal: str) -> set[str]:
    brief = _brief(goal)
    focus = derive_goal_focus(brief)
    packet = HypothesisPacket(
        brief_id=brief.brief_id,
        hypotheses=[_hypothesis("TP53", 1)],
    )
    request = context_request_from_hypothesis_packet(
        packet,
        budget=ContextRoutingBudget(max_source_calls=10),
        human_priorities=routing_priorities_for_focus([], focus),
        allowed_sources=[
            "Human Cell Atlas", "CELLxGENE Census", "ARCHS4", "BioGRID"
        ],
    )
    retriever = RecordingContextRetriever()
    DiseaseTargetContextRouter(retriever).route(request)
    return set(retriever.sources)


def test_same_disease_network_vs_single_cell_goal_changes_source_plan() -> None:
    network = _routed_sources(
        "Map the explicit protein interaction network for this disease target."
    )
    single_cell = _routed_sources(
        "Resolve this disease target in a single-cell transcriptomic context."
    )

    assert "BioGRID" in network
    assert "Human Cell Atlas" not in network
    assert {"Human Cell Atlas", "CELLxGENE Census", "ARCHS4"} <= single_cell
    assert "BioGRID" not in single_cell


def test_cell_line_target_is_not_misclassified_as_single_cell() -> None:
    brief = _brief("Evaluate direct target inhibition in AML cell-line models.")
    brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={"target_cell_type": "AML cell lines"},
                deep=True,
            )
        },
        deep=True,
    )

    focus = derive_goal_focus(brief)

    assert any(cue.category == "cell_line" for cue in focus.cues)
    assert all(cue.category != "single_cell" for cue in focus.cues)
    assert "single_cell" not in focus.source_families
    assert "cell-line model context" in focus.routing_priorities


def test_isogenic_cell_line_protocol_does_not_route_to_single_cell_atlas() -> None:
    brief = _brief("Evaluate direct FLT3 inhibition in AML.")
    brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={
                    "target_cell_type": (
                        "MOLM-13 with molecularly confirmed FLT3-ITD, paired "
                        "with an isogenic FLT3-ITD-corrected comparator "
                        "generated from the same parental background"
                    )
                },
                deep=True,
            ),
            "human_priorities": [
                "Measure apoptosis in FLT3-ITD AML cells and compare the "
                "isogenic corrected comparator."
            ],
        },
        deep=True,
    )

    focus = derive_goal_focus(brief)

    assert any(cue.category == "cell_line" for cue in focus.cues)
    assert all(cue.category not in {"single_cell", "cell_context"} for cue in focus.cues)
    assert "single_cell" not in focus.source_families


def test_named_primary_cell_type_is_cell_resolved_not_implicitly_single_cell() -> None:
    brief = _brief("Evaluate a disease-relevant target in primary cells.")
    brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={"target_cell_type": "synovial fibroblasts"},
                deep=True,
            )
        },
        deep=True,
    )

    focus = derive_goal_focus(brief)

    assert any(cue.category == "cell_context" for cue in focus.cues)
    assert all(cue.category != "single_cell" for cue in focus.cues)
    assert focus.source_families == ["single_cell"]
    assert "cell-resolved context" in focus.routing_priorities


def test_explicit_target_focus_selects_only_existing_evidence_backed_hypothesis() -> None:
    packet = _packet()
    focus = derive_goal_focus(
        _brief("Prioritize target: TP53 while preserving the evidence requirements.")
    )
    focused, resolved = apply_goal_focus(packet, focus)
    selected, decision = select_hypothesis_for_priority(focused.hypotheses, "auto")

    assert selected.target.symbol == "TP53"
    assert decision["selection_reason"] == "EXPLICIT_EVIDENCE_BACKED_TARGET_FOCUS"
    assert resolved.resolved_target_mentions == ["TP53"]
    assert {item.target.symbol for item in focused.hypotheses} == {"ABL1", "TP53"}
    assert resolved.target_scope_mode == "portfolio"
    assert resolved.executable_target_mentions == []


def test_executable_target_only_clause_deterministically_filters_portfolio() -> None:
    focus = derive_goal_focus(
        _brief(
            "Use executable target FLT3 only and require auditable candidate evidence."
        )
    )
    focused, resolved = apply_goal_focus(_executable_scope_packet(), focus)

    assert focus.target_scope_mode == "explicit_only"
    assert focus.executable_target_mentions == ["FLT3"]
    assert {item.target.symbol for item in focused.hypotheses} == {"FLT3"}
    assert resolved.resolved_target_mentions == ["FLT3"]
    assert focused.hypotheses[0].rank == 1
    assert focused.metadata["target_sources"] == {
        "FLT3": {"target_source": "external_dynamic"}
    }
    assert focused.metadata["goal_focus_scope_enforced"] is True
    assert focused.metadata["goal_focus_executable_targets"] == ["FLT3"]
    assert focused.metadata["goal_focus_retained_symbols"] == ["flt3"]
    assert focused.metadata["goal_focus_scope_excluded_symbols"] == ["SRSF2"]
    assert any(
        item.target_symbol == "SRSF2"
        and item.reason_codes == ["OUTSIDE_EXECUTABLE_TARGET_SCOPE"]
        for item in focused.excluded
    )


def test_direct_target_action_only_clause_is_executable_scope() -> None:
    focus = derive_goal_focus(
        _brief(
            "Evaluate direct JAK2 inhibition only in myelofibrosis and identify "
            "evidence-supported candidates."
        )
    )

    assert focus.target_scope_mode == "explicit_only"
    assert focus.executable_target_mentions == ["JAK2"]
    assert "JAK2" in focus.explicit_target_mentions


def test_target_mechanism_role_word_is_not_parsed_as_a_gene_symbol() -> None:
    focus = derive_goal_focus(
        _brief(
            "Executable target JAK2 only. Use the exact human JAK2 single-protein "
            "INHIBITOR mechanism."
        )
    )

    assert focus.target_scope_mode == "explicit_only"
    assert focus.executable_target_mentions == ["JAK2"]
    assert focus.explicit_target_mentions == ["JAK2"]


def test_target_action_without_only_does_not_close_portfolio_scope() -> None:
    focus = derive_goal_focus(
        _brief("Evaluate JAK2 inhibition in myelofibrosis alongside other mechanisms.")
    )

    assert focus.target_scope_mode == "portfolio"
    assert focus.executable_target_mentions == []


def test_plural_disease_targets_phrase_does_not_create_explicit_target() -> None:
    brief = _brief(
        "Discover and compare a portfolio of independently supported molecular "
        "targets for acute myeloid leukemia."
    )
    brief = brief.model_copy(
        update={
            "disease": "Acute myeloid leukemia",
            "human_priorities": [
                "compare multiple independently supported AML targets"
            ],
        },
        deep=True,
    )

    focus = derive_goal_focus(brief)

    assert focus.target_scope_mode == "portfolio"
    assert focus.explicit_target_mentions == []
    assert focus.executable_target_mentions == []


def test_singular_target_label_still_parses_exact_symbol() -> None:
    focus = derive_goal_focus(
        _brief("Prioritize JAK2 target while preserving the portfolio.")
    )

    assert focus.target_scope_mode == "portfolio"
    assert focus.explicit_target_mentions == ["JAK2"]


def test_korean_plural_disease_targets_phrase_does_not_create_explicit_target() -> None:
    focus = derive_goal_focus(
        _brief("여러 AML 표적들을 비교하고 독립 근거를 평가한다.")
    )

    assert focus.target_scope_mode == "portfolio"
    assert focus.explicit_target_mentions == []


def test_korean_singular_target_label_with_particle_still_parses_symbol() -> None:
    focus = derive_goal_focus(
        _brief("JAK2 표적은 우선 검토하되 포트폴리오 범위는 유지한다.")
    )

    assert focus.target_scope_mode == "portfolio"
    assert focus.explicit_target_mentions == ["JAK2"]


def test_restrict_target_list_is_closed_and_does_not_depend_on_llm() -> None:
    focus = derive_goal_focus(
        _brief(
            "Restrict executable targets to FLT3 and BCL2 for this audited run."
        )
    )

    assert focus.target_scope_mode == "explicit_only"
    assert focus.executable_target_mentions == ["FLT3", "BCL2"]
    assert {"FLT3", "BCL2"} <= set(focus.explicit_target_mentions)
    assert any(
        cue.reason == "closed executable/restrictive target-scope clause matched"
        for cue in focus.cues
    )


def test_executable_scope_without_claim_linked_support_fails_closed() -> None:
    focus = derive_goal_focus(
        _brief("Use executable target FLT3 only for this evidence-governed run.")
    )
    focused, resolved = apply_goal_focus(
        _executable_scope_packet(flt3_linked=False),
        focus,
    )

    assert focused.hypotheses == []
    assert resolved.resolved_target_mentions == []
    assert resolved.rejected_target_mentions == [
        {"symbol": "FLT3", "reason": "no claim-linked support evidence"}
    ]
    exclusions = {
        item.target_symbol: item.reason_codes for item in focused.excluded
    }
    assert exclusions["FLT3"] == [
        "EXECUTABLE_TARGET_SCOPE_EVIDENCE_UNQUALIFIED"
    ]
    assert exclusions["SRSF2"] == ["OUTSIDE_EXECUTABLE_TARGET_SCOPE"]


def test_screened_explicit_target_is_not_reported_as_never_produced() -> None:
    packet = HypothesisPacket(
        brief_id="brief_screened_scope",
        hypotheses=[],
        excluded=[
            ExcludedHypothesis(
                target_symbol="ALK",
                reason_codes=["UNSUPPORTED_CLAIM"],
                source_iteration=1,
            )
        ],
        metadata={
            "ma1_discovery_candidate_trace": [
                {"target_symbol": "ALK", "screening_allocation": "scheduled_for_ma1"}
            ],
            "screened_target_sources": [
                {"target_symbol": "ALK", "disposition": "rejected"}
            ],
        },
    )
    focus = derive_goal_focus(
        _brief("Use executable target ALK only for this run.")
    )

    focused, resolved = apply_goal_focus(packet, focus)

    assert focused.hypotheses == []
    assert resolved.resolved_target_mentions == []
    assert resolved.rejected_target_mentions == [
        {
            "symbol": "ALK",
            "reason": "screened by MA1 but no claim-linked retained hypothesis",
        }
    ]
    assert focused.metadata["goal_focus_observed_symbols"] == ["alk"]
    assert focused.metadata["goal_focus_discovered_symbols"] == ["alk"]
    assert focused.metadata["goal_focus_screened_symbols"] == ["alk"]


def test_current_and_cumulative_screening_scopes_are_not_conflated() -> None:
    packet = HypothesisPacket(
        brief_id="brief_screening_scope",
        hypotheses=[],
        excluded=[],
        metadata={
            "screened_target_sources": [
                {"target_symbol": "CALR", "disposition": "rejected"}
            ],
            "ma1_screening_history_sources": [
                {"target_symbol": "JAK2", "disposition": "rejected"},
                {"target_symbol": "CALR", "disposition": "rejected"},
            ],
            "ma1_screened_candidate_count": 1,
        },
    )

    focused, _ = apply_goal_focus(
        packet,
        derive_goal_focus(_brief("Compare a broad portfolio.")),
    )

    assert focused.metadata["goal_focus_screened_symbols"] == ["calr"]
    assert focused.metadata["goal_focus_screening_history_symbols"] == [
        "calr",
        "jak2",
    ]
    assert focused.metadata["goal_focus_screened_symbols_semantics"] == (
        "current_revision_unique_targets"
    )


def test_executable_scope_is_reapplied_without_duplicate_rework_exclusions() -> None:
    focus = derive_goal_focus(
        _brief("Use executable target FLT3 only throughout every rework iteration.")
    )
    first, _ = apply_goal_focus(_executable_scope_packet(), focus)
    rework_packet = _executable_scope_packet()
    rework_packet.excluded = [item.model_copy(deep=True) for item in first.excluded]

    reworked, _ = apply_goal_focus(rework_packet, focus)

    assert {item.target.symbol for item in reworked.hypotheses} == {"FLT3"}
    scope_exclusions = [
        item
        for item in reworked.excluded
        if item.target_symbol == "SRSF2"
        and "OUTSIDE_EXECUTABLE_TARGET_SCOPE" in item.reason_codes
    ]
    assert len(scope_exclusions) == 1


def test_unknown_or_unlabelled_goal_does_not_fabricate_or_reorder_targets() -> None:
    packet = _packet()
    before = [(item.target.symbol, item.rank) for item in packet.hypotheses]
    focus = derive_goal_focus(
        _brief("Explore robust mechanisms without naming a candidate gene symbol.")
    )
    focused, resolved = apply_goal_focus(packet, focus)

    assert resolved.explicit_target_mentions == []
    assert resolved.target_scope_mode == "portfolio"
    assert resolved.executable_target_mentions == []
    assert resolved.resolved_target_mentions == []
    assert [(item.target.symbol, item.rank) for item in focused.hypotheses] == before


def test_generic_executable_target_discovery_without_symbol_keeps_portfolio() -> None:
    packet = _packet()
    focus = derive_goal_focus(
        _brief(
            "Identify executable targets for this disease using auditable evidence."
        )
    )
    focused, resolved = apply_goal_focus(packet, focus)

    assert resolved.target_scope_mode == "portfolio"
    assert resolved.executable_target_mentions == []
    assert {item.target.symbol for item in focused.hypotheses} == {"ABL1", "TP53"}
    assert focused.metadata["goal_focus_scope_enforced"] is False


def test_goal_strategy_changes_effective_priority_only_when_structured_value_is_auto() -> None:
    packet = _packet()
    repurposing_brief = _brief(
        "Prioritize drug repurposing using disease-aware known-drug evidence."
    )
    focused, _ = apply_goal_focus(packet, derive_goal_focus(repurposing_brief))
    plan = PlannerAgent().run(focused, repurposing_brief)

    assert plan.requested_discovery_priority == "auto"
    assert plan.discovery_priority == "repurposing_target"
    assert plan.discovery_priority_derivation == "explicit_goal_repurposing_cue_when_auto"

    structured = _brief(
        "Prioritize drug repurposing using disease-aware known-drug evidence.",
        priority="novel_target",
    )
    structured_packet, _ = apply_goal_focus(packet, derive_goal_focus(structured))
    structured_plan = PlannerAgent().run(structured_packet, structured)
    assert structured_plan.discovery_priority == "novel_target"
    assert structured_plan.discovery_priority_derivation == "research_brief_structured_constraint"
