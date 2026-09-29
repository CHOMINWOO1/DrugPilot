from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from drugpilot.config import Settings
from drugpilot.domain.models import (
    Claim, EvidenceRef, Hypothesis, HypothesisPacket, HypothesisTrack,
    ResearchBrief, RunMode, TargetCandidate,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.orchestration.goal_focus import GoalFocus
from drugpilot.orchestration.ma1_review_context import (
    apply_ma1_review_goal_focus, review_context_view, restore_review_context,
)
from drugpilot.tools.context_router import (
    ContextRoutingBudget, ContextRoutingResult, ContextSourceOutcome,
    ContextTarget, TargetContextBundle,
)


def _packet(*symbols: str) -> HypothesisPacket:
    return HypothesisPacket(brief_id="brief_review_context", hypotheses=[],
        provisional_hypotheses=[Hypothesis(
            hypothesis_id=f"hyp_{symbol}", track=HypothesisTrack.LITERATURE,
            disease="acute myeloid leukemia", target=TargetCandidate(symbol=symbol,
                metadata={"ma1_admission": {"status": "provisional_pending_ma3"},
                          "direction_resolution": {"status": "direction_unresolved", "resolved_direction": None}}),
            intervention_direction="modulate", mechanism="Prospective source-bound hypothesis.",
            falsification_test="Measure a controlled endpoint.", claims=[Claim(text="Pending source assessment.")],
            evidence=[EvidenceRef(evidence_id=f"ev_{symbol}", source="PubMed", external_id="test",
                                  title="Test source", supports="context")], rank=i + 1,
        ) for i, symbol in enumerate(symbols)],
        metadata={"admission_policy": "ma3-source-bound-v1", "ma1_executable_hypothesis_count": 0,
                  "target_sources": {symbol: {"identity_status": "resolved"} for symbol in symbols}})


def test_review_context_roundtrip_keeps_scientific_approval_and_direction_unchanged() -> None:
    original = _packet("FLT3")
    original_json = original.model_dump_json()
    view = review_context_view(original)
    assert len(view.hypotheses) == 1 and not view.provisional_hypotheses
    view.hypotheses[0].evidence.append(EvidenceRef(
        source="Pharos", external_id="context", title="Context only", supports="context"))
    restored = restore_review_context(original, view)
    assert not restored.hypotheses and len(restored.provisional_hypotheses) == 1
    assert restored.provisional_hypotheses[0].claims == original.provisional_hypotheses[0].claims
    assert restored.provisional_hypotheses[0].target.metadata["direction_resolution"]["resolved_direction"] is None
    assert "context_view_only_not_scientific_approval" not in restored.metadata
    assert original.model_dump_json() == original_json


@pytest.mark.parametrize("mutation", ["claim", "direction", "identity", "evidence", "support", "admission",
                                    "target_symbol", "target_ensembl_id", "target_uniprot_id", "original_support"])
def test_context_router_cannot_promote_or_rewrite_scientific_submission(mutation: str) -> None:
    original = _packet("FLT3")
    view = review_context_view(original)
    hypothesis = view.hypotheses[0]
    if mutation == "claim":
        hypothesis.claims[0].evidence_ids = ["ev_FLT3"]
    elif mutation == "direction":
        hypothesis.intervention_direction = "inhibit"
    elif mutation == "identity":
        hypothesis.hypothesis_id = "forged"
    elif mutation == "evidence":
        hypothesis.evidence[0].external_id = "forged"
    elif mutation == "support":
        hypothesis.evidence.append(EvidenceRef(source="Pharos", external_id="new", title="Forged primary"))
    elif mutation.startswith("target_"):
        setattr(hypothesis.target, mutation.removeprefix("target_"), "forged")
    elif mutation == "original_support":
        hypothesis.evidence[0].supports = "support"
    else:
        hypothesis.target.metadata["ma1_admission"]["status"] = "approved"
    with pytest.raises(ValueError):
        restore_review_context(original, view)


def test_explicit_scope_retains_unlinked_pending_match_without_faking_scientific_support() -> None:
    packet = _packet("FLT3", "BCL2")
    focus = GoalFocus(focus_id="focus", brief_id=packet.brief_id,
        target_scope_mode="explicit_only", executable_target_mentions=["FLT3"])
    result, updated_focus = apply_ma1_review_goal_focus(packet, focus)
    assert not result.hypotheses
    assert [item.target.symbol for item in result.provisional_hypotheses] == ["FLT3"]
    assert not result.provisional_hypotheses[0].claims[0].evidence_ids
    assert updated_focus.resolved_target_mentions == []
    assert result.metadata["goal_focus_retained_symbols"] == []
    assert result.metadata["goal_focus_provisional_symbols"] == ["flt3"]
    assert result.metadata["target_sources"] == {"FLT3": {"identity_status": "resolved"}}
    assert any(item.target_symbol == "BCL2" for item in result.excluded)


@pytest.mark.asyncio
async def test_pharos_router_sees_pending_targets_but_result_never_promotes_them(tmp_path) -> None:
    orchestrator = DrugPilotOrchestrator(settings=Settings(
        _env_file=None, mode=RunMode.MOCK, database_path=tmp_path / "unused.sqlite3",
        public_evidence_enabled=False, ma1_provisional_admission_enabled=True))
    seen = []
    class Router:
        def route(self, packet, brief):
            seen.extend(item.target.symbol for item in packet.hypotheses)
            copied = packet.model_copy(deep=True)
            copied.hypotheses[0].evidence.append(EvidenceRef(
                source="Pharos", external_id="tdl", title="TDL context", supports="context"))
            return SimpleNamespace(packet=copied, decision={"status": "completed", "outcomes": [], "records_added": 1})
    orchestrator._pharos_router_for_brief = lambda brief: Router()
    orchestrator._trace_event = AsyncMock()
    result = await orchestrator._enrich_ma1_pharos("test", _packet("FLT3"),
        ResearchBrief(research_goal="Investigate AML", disease="acute myeloid leukemia"))
    assert seen == ["FLT3"]
    assert not result.hypotheses
    assert result.provisional_hypotheses[0].evidence[-1].source == "Pharos"


@pytest.mark.asyncio
async def test_context_router_sees_pending_targets_without_claim_link_changes(tmp_path) -> None:
    orchestrator = DrugPilotOrchestrator(settings=Settings(
        _env_file=None, mode=RunMode.MOCK, database_path=tmp_path / "unused.sqlite3",
        public_evidence_enabled=False, ma1_provisional_admission_enabled=True))
    seen = []
    class Router:
        def route_hypothesis_packet(self, packet, **kwargs):
            seen.extend(item.target.symbol for item in packet.hypotheses)
            return ContextRoutingResult(disease="acute myeloid leukemia", budget=ContextRoutingBudget(),
                targets=[TargetContextBundle(target=ContextTarget(symbol="FLT3", track=HypothesisTrack.LITERATURE,
                    hypothesis_id="hyp_FLT3", rank=1), sources=[ContextSourceOutcome(
                        source="bioRxiv", family="preprint", status="ok", queried=True, record_count=1)],
                    evidence=[EvidenceRef(source="bioRxiv", external_id="context", title="Context result", supports="context",
                        metadata={"context_only": True, "causal_inference_allowed": False, "counter_evidence_allowed": False})])],
                source_calls_used=1, planned_source_calls=1, elapsed_seconds=0.1, budget_exhausted=False)
    orchestrator._context_router_for_brief = lambda brief: Router()
    orchestrator._trace_event = AsyncMock()
    result = await orchestrator._enrich_ma1_context("test", _packet("FLT3"),
        ResearchBrief(research_goal="Investigate AML", disease="acute myeloid leukemia"))
    assert seen == ["FLT3"]
    assert not result.hypotheses
    assert result.metadata["context_routing"]["status"] != "skipped"
    assert not result.provisional_hypotheses[0].claims[0].evidence_ids


@pytest.mark.asyncio
async def test_intake_abstention_withholds_pending_submissions_too(tmp_path) -> None:
    orchestrator = DrugPilotOrchestrator(settings=Settings(
        _env_file=None, mode=RunMode.MOCK, database_path=tmp_path / "unused.sqlite3",
        public_evidence_enabled=False, ma1_provisional_admission_enabled=True))
    packet = _packet("FLT3")
    orchestrator._ma1_for_brief = lambda brief: SimpleNamespace(run=lambda brief, previous: packet)
    orchestrator._transition = AsyncMock()
    orchestrator._trace_event = AsyncMock()
    orchestrator._artifact_transition = AsyncMock()
    orchestrator.store.get_snapshot = AsyncMock(return_value=None)
    async def enhance(*args, **kwargs):
        result = kwargs["fallback"]
        return kwargs["postprocess"](result) if kwargs.get("postprocess") else result
    orchestrator._llm_enhance = enhance
    state = {"run_id": "test", "brief": ResearchBrief(
        research_goal="Investigate AML", disease="acute myeloid leukemia")}
    result = await orchestrator._run_ma1(state)
    assert result["intake_abstain_recommended"] is True
    assert result["hypothesis_packet"].hypotheses == []
    assert result["hypothesis_packet"].provisional_hypotheses == []
    assert result["hypothesis_packet"].metadata["intake_withheld_provisional_count"] == 1
