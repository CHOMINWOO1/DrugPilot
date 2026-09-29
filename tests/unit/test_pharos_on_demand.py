from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.agents.ma1 import Critic
from drugpilot.api.call_trace import build_run_call_trace
from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import (
    Claim,
    EvidenceRef,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    ResearchBrief,
    RunMode,
    RunSnapshot,
    RunStage,
    TargetCandidate,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.storage import SQLiteRunStore
from drugpilot.tools.pharos_on_demand import (
    OnDemandPharosRouter,
    requested_pharos_terms,
)
from drugpilot.tools.context_router import context_request_from_hypothesis_packet
from drugpilot.tools.route_policy import derive_route_evidence


class _RecordingPharosRetriever:
    def __init__(
        self,
        *,
        fail: bool = False,
        tdl: str = "Tbio",
        provenance_class: str = "external_live",
    ) -> None:
        self.calls: list[str] = []
        self.fail = fail
        self.tdl = tdl
        self.provenance_class = provenance_class

    def retrieve(self, brief, seed, iteration):
        del brief, iteration
        self.calls.append(seed.symbol)
        if self.fail:
            raise TimeoutError("simulated endpoint timeout")
        content_hash = f"content-hash-{seed.symbol.casefold()}"
        trust_metadata: dict[str, object]
        if self.provenance_class == "external_cache":
            trust_metadata = {
                "live": False,
                "cache_hit": True,
                "retrieved_live": True,
                "origin_provenance_class": "external_live",
                "origin_content_hash": content_hash,
            }
        elif "fixture" in self.provenance_class or "mock" in self.provenance_class:
            trust_metadata = {
                "backend_kind": "fixture",
                "fixture": True,
                "live": False,
                "cache_hit": False,
            }
        else:
            trust_metadata = {"live": True, "cache_hit": False}
        return [
            EvidenceRef(
                evidence_id=f"ev_pharos_{seed.symbol.casefold()}",
                source="Pharos",
                external_id=seed.symbol,
                title=f"Pharos target development record: {seed.symbol}",
                url=f"https://pharos.nih.gov/targets/{seed.symbol}",
                supports="context",
                metadata={
                    "provenance_class": self.provenance_class,
                    "retrieval_endpoint": "https://pharos-api.ncats.io/graphql",
                    "query": seed.symbol,
                    "tdl": self.tdl,
                    "content_hash": content_hash,
                    **trust_metadata,
                },
            )
        ]


class _ConflictingPharosRetriever:
    def __init__(self, *, cached_conflict: bool = False) -> None:
        self.cached_conflict = cached_conflict

    def retrieve(self, brief, seed, iteration):
        del brief, iteration
        return [
            EvidenceRef(
                evidence_id=f"ev_pharos_{seed.symbol.casefold()}_live",
                source="Pharos",
                external_id=seed.symbol,
                title=f"Live Pharos target development record: {seed.symbol}",
                supports="context",
                metadata={
                    "provenance_class": "external_live",
                    "query": seed.symbol,
                    "tdl": "Tclin",
                },
            ),
            EvidenceRef(
                evidence_id=f"ev_pharos_{seed.symbol.casefold()}_conflict",
                source="Pharos",
                external_id=seed.symbol,
                title=f"Conflicting Pharos target development record: {seed.symbol}",
                supports="context",
                metadata={
                    "provenance_class": (
                        "external_cache"
                        if self.cached_conflict
                        else "external_live"
                    ),
                    "query": seed.symbol,
                    "tdl": "Tbio",
                },
            ),
        ]


class _ChangedSameIdPharosRetriever(_RecordingPharosRetriever):
    def retrieve(self, brief, seed, iteration):
        records = super().retrieve(brief, seed, iteration)
        return [
            records[0].model_copy(
                update={"title": f"Changed content for {seed.symbol}"},
                deep=True,
            )
        ]


class _OriginlessCachePharosRetriever(_RecordingPharosRetriever):
    def __init__(self, *, tdl: str = "Tbio") -> None:
        super().__init__(tdl=tdl, provenance_class="external_cache")

    def retrieve(self, brief, seed, iteration):
        records = super().retrieve(brief, seed, iteration)
        for record in records:
            record.metadata.pop("origin_provenance_class", None)
            record.metadata.pop("origin_content_hash", None)
        return records


class _FixtureMarkedLivePharosRetriever(_RecordingPharosRetriever):
    def __init__(self, *, tdl: str = "Tbio") -> None:
        super().__init__(tdl=tdl, provenance_class="external_live")

    def retrieve(self, brief, seed, iteration):
        records = super().retrieve(brief, seed, iteration)
        for record in records:
            record.metadata.update(
                {
                    "backend_kind": "fixture",
                    "fixture": True,
                    "live": False,
                }
            )
        return records


class _EmptyPharosRetriever:
    last_failures = {"Pharos": "empty_result"}

    def retrieve(self, brief, seed, iteration):
        del brief, seed, iteration
        raise RuntimeError("No public biology evidence returned")


def _hypothesis(
    symbol: str,
    track: HypothesisTrack,
    *,
    rank: int,
    score: float,
) -> Hypothesis:
    base_id = f"ev_base_{symbol.casefold()}_{track.value}"
    return Hypothesis(
        hypothesis_id=f"hyp_{symbol.casefold()}_{track.value}",
        track=track,
        disease="acute myeloid leukemia",
        target=TargetCandidate(symbol=symbol, target_name=symbol),
        mechanism="A bounded mechanism.",
        intervention_direction="inhibit",
        falsification_test="Reject after a prespecified negative experiment.",
        claims=[
            Claim(
                claim_id=f"claim_{symbol.casefold()}_{track.value}",
                text="Evidence-linked claim.",
                evidence_ids=[base_id],
            )
        ],
        evidence=[
            EvidenceRef(
                evidence_id=base_id,
                source="PubMed",
                external_id=f"PMID-{symbol}",
                title=f"Primary evidence for {symbol}",
                supports="support",
            )
        ],
        score=score,
        rank=rank,
    )


def _packet() -> HypothesisPacket:
    return HypothesisPacket(
        packet_id="hp_pharos_test",
        brief_id="brief_pharos_test",
        hypotheses=[
            _hypothesis("FLT3", HypothesisTrack.LITERATURE, rank=1, score=0.92),
            _hypothesis("KIT", HypothesisTrack.LITERATURE, rank=2, score=0.88),
            _hypothesis("BCL2", HypothesisTrack.TBIO, rank=1, score=0.90),
        ],
    )


def _brief(*priorities: str, allowed_sources: list[str] | None = None) -> ResearchBrief:
    values = {
        "brief_id": "brief_pharos_test",
        "research_goal": "Identify an evidence-linked therapeutic target hypothesis.",
        "disease": "acute myeloid leukemia",
        "human_priorities": list(priorities),
    }
    if allowed_sources is not None:
        values["data_policy"] = {"allowed_sources": allowed_sources}
    return ResearchBrief.model_validate(values)


@pytest.mark.parametrize(
    "priority",
    [
        "Use Pharos target context",
        "Check TDL",
        "assess target development level",
        "표적 개발 수준을 확인",
        "약물화 가능성 검토",
    ],
)
def test_explicit_priority_terms_trigger_deterministic_pharos_policy(priority) -> None:
    assert requested_pharos_terms([priority])


def test_optional_druggability_without_priority_records_skip() -> None:
    retriever = _RecordingPharosRetriever()
    brief = _brief("prioritize primary literature")
    brief.constraints.druggability_required = False
    result = OnDemandPharosRouter(retriever).route(
        _packet(),
        brief,
    )

    assert retriever.calls == []
    assert result.decision["status"] == "skipped"
    assert result.decision["policy_owner"] == "Orchestrator"
    assert result.decision["requested"] is False
    assert "not requested" in str(result.decision["reason"])
    assert result.packet.metadata["pharos_routing"] == result.decision


def test_structured_druggability_requirement_triggers_without_prose():
    retriever = _RecordingPharosRetriever()
    original = _packet()
    before = original.model_dump_json()
    result = OnDemandPharosRouter(retriever).route(original, _brief())
    assert result.decision["requested"] is True
    assert result.decision["matched_terms"] == []
    assert result.decision["request_fields"] == ["ResearchBrief.constraints.druggability_required"]
    assert "ResearchBrief.constraints.druggability_required" in result.decision["reason"]
    assert "human priority" not in result.decision["reason"]
    assert retriever.calls
    assert result.decision["claim_links_modified"] is False
    assert original.model_dump_json() == before


@pytest.mark.parametrize("disabled", ["source", "adapter", "quota", "no_targets"])
def test_structured_druggability_never_bypasses_source_or_execution_limits(disabled):
    retriever = _RecordingPharosRetriever()
    brief, packet = _brief(), _packet()
    if disabled == "source": brief.data_policy.allowed_sources = ["PubMed"]
    if disabled == "no_targets": packet.hypotheses = []
    result = OnDemandPharosRouter(None if disabled == "adapter" else retriever,
        max_targets_per_track=0 if disabled == "quota" else 1).route(packet, brief)
    assert result.decision["requested"] is True
    assert result.decision["status"] == "skipped"
    assert retriever.calls == []


def test_explicit_request_queries_only_top_target_per_track_and_keeps_context_only() -> None:
    retriever = _RecordingPharosRetriever()
    packet = _packet()
    claim_links_before = [
        (list(claim.evidence_ids), list(claim.counter_evidence_ids))
        for hypothesis in packet.hypotheses
        for claim in hypothesis.claims
    ]

    result = OnDemandPharosRouter(retriever, max_targets_per_track=1).route(
        packet,
        _brief("Pharos TDL and druggability context"),
    )

    assert retriever.calls == ["FLT3", "BCL2"]
    assert result.decision["status"] == "completed"
    assert result.decision["records_added"] == 2
    assert [item["symbol"] for item in result.decision["selected_targets"]] == [
        "FLT3",
        "BCL2",
    ]
    pharos_records = [
        evidence
        for hypothesis in result.packet.hypotheses
        for evidence in hypothesis.evidence
        if evidence.source == "Pharos"
    ]
    assert len(pharos_records) == 2
    assert all(item.supports == "context" for item in pharos_records)
    assert {
        item.metadata["provenance_class"] for item in pharos_records
    } == {"external_live"}
    assert claim_links_before == [
        (list(claim.evidence_ids), list(claim.counter_evidence_ids))
        for hypothesis in result.packet.hypotheses
        for claim in hypothesis.claims
    ]
    assert all(
        evidence.source != "Pharos"
        for hypothesis in packet.hypotheses
        for evidence in hypothesis.evidence
    )


def test_live_pharos_tdl_replaces_track_placeholder_with_auditable_effective_tdl() -> None:
    packet = _packet()
    packet.hypotheses[0].target.tdl = "Tbio"
    result = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(packet, _brief("Pharos TDL context"))
    target = result.packet.hypotheses[0].target

    assert target.tdl == "Tclin"
    assert target.metadata["effective_tdl"] == "Tclin"
    assert target.metadata["effective_tdl_source"] == "Pharos"
    assert target.metadata["tdl_source"] == "Pharos"
    assert target.metadata["ma1_supply_category"] == {
        "hypothesis_track": "literature",
        "semantics": "original_ma1_evidence_supply_track",
        "legacy_track_derived_label": "Tbio",
        "legacy_track_derived_label_source": None,
        "legacy_track_derived_label_is_scientific_tdl": False,
    }
    assert target.metadata["track_tdl_contract"] == {
        "policy": "separate-work-track-and-scientific-tdl-v1",
        "hypothesis_track": "literature",
        "hypothesis_track_semantics": "original_ma1_evidence_supply_track",
        "effective_work_track": "literature",
        "effective_work_track_basis": "effective_tdl",
        "effective_tdl": "Tclin",
        "effective_tdl_semantics": (
            "authoritative_scientific_target_development_level"
        ),
        "namespaces_distinct": True,
        "context_source_plan_uses": "effective_work_track",
        "ma2_candidate_route_uses": "effective_tdl",
    }
    reconciliation = target.metadata["tdl_reconciliation"]
    assert reconciliation["status"] == "resolved"
    assert reconciliation["conflict_detected"] is False
    assert reconciliation["conflict_hidden"] is False
    assert reconciliation["effective_evidence_ids"] == ["ev_pharos_flt3"]
    claim = target.metadata["tdl_source_claims"][0]
    assert claim["source"] == "Pharos"
    assert claim["value"] == "Tclin"
    assert claim["provenance_class"] == "external_live"
    assert claim["evidence_id"] == "ev_pharos_flt3"

    route_evidence = derive_route_evidence(
        result.packet.hypotheses[0],
        target,
    )
    assert route_evidence["verified_tdl"] == "Tclin"
    assert route_evidence["effective_tdl"] == "Tclin"
    assert route_evidence["tdl_resolution_status"] == "resolved"
    assert route_evidence["tdl_conflict_detected"] is False
    assert route_evidence["effective_work_track"] == "literature"


def test_same_priority_pharos_conflict_is_preserved_and_not_routed_as_tdl() -> None:
    result = OnDemandPharosRouter(_ConflictingPharosRetriever()).route(
        _packet(),
        _brief("Pharos TDL context"),
    )
    hypothesis = result.packet.hypotheses[0]
    target = hypothesis.target

    assert target.tdl is None
    assert "tdl_source" not in target.metadata
    assert target.metadata["effective_tdl"] is None
    assert {item["value"] for item in target.metadata["tdl_source_claims"]} == {
        "Tbio",
        "Tclin",
    }
    reconciliation = target.metadata["tdl_reconciliation"]
    assert reconciliation["status"] == "unresolved_same_priority_conflict"
    assert reconciliation["conflict_detected"] is True
    assert reconciliation["conflict_hidden"] is False

    route_evidence = derive_route_evidence(hypothesis, target)
    assert route_evidence["verified_tdl"] is None
    assert route_evidence["effective_tdl"] is None
    assert route_evidence["tdl_conflict_detected"] is True
    assert route_evidence["tdl_resolution_status"] == (
        "unresolved_same_priority_conflict"
    )


def test_live_tdl_precedes_cached_conflict_without_hiding_source_claim() -> None:
    result = OnDemandPharosRouter(
        _ConflictingPharosRetriever(cached_conflict=True)
    ).route(_packet(), _brief("Pharos TDL context"))
    hypothesis = result.packet.hypotheses[0]
    target = hypothesis.target

    assert target.tdl == "Tclin"
    reconciliation = target.metadata["tdl_reconciliation"]
    assert reconciliation["status"] == "resolved_with_source_conflict"
    assert reconciliation["conflict_detected"] is True
    assert reconciliation["source_values"] == ["Tbio", "Tclin"]
    assert len(target.metadata["tdl_source_claims"]) == 2

    route_evidence = derive_route_evidence(hypothesis, target)
    assert route_evidence["verified_tdl"] == "Tclin"
    assert route_evidence["tdl_conflict_detected"] is True


def test_repeated_pharos_enrichment_keeps_effective_tdl_and_does_not_duplicate_claims() -> None:
    retriever = _RecordingPharosRetriever(tdl="Tclin")
    packet = _packet()
    packet.hypotheses[0].target.tdl = "Tbio"
    first = OnDemandPharosRouter(retriever).route(
        packet,
        _brief("Pharos TDL context"),
    )
    # Mirror the orchestration layer's comprehensive-coverage annotation.  It
    # is retrieval policy state, not a change to the immutable Pharos record.
    for hypothesis in first.packet.hypotheses:
        for evidence in hypothesis.evidence:
            if evidence.source == "Pharos":
                evidence.supports = "context"
                evidence.metadata = {
                    **evidence.metadata,
                    "source_supports": "context",
                    "partial_source_failures": {"Europe PMC": "timeout"},
                    "routing_policy": (
                        "deterministic_comprehensive_druggability_coverage"
                    ),
                    "human_priority_required": False,
                    "claim_linkage_status": "reconciled",
                    "claim_linkage_policy": (
                        "atomic-publication-intervention-endpoint-linkage-v2"
                    ),
                    "linked_claim_ids": [],
                    "claim_linkage_roles": ["context_unlinked"],
                    "claim_linkage_scope": "unlinked_context",
                    "direct_support_eligible": False,
                    "direct_support_rejection_codes": [],
                    "verified_direct_primary": False,
                    "verified_direct_primary_rejection_codes": [],
                }
    second = OnDemandPharosRouter(
        _RecordingPharosRetriever(
            tdl="Tclin",
            provenance_class="external_cache",
        )
    ).route(
        first.packet,
        _brief("Pharos TDL context"),
    )
    target = second.packet.hypotheses[0].target

    assert target.tdl == "Tclin"
    assert target.metadata["tdl_reconciliation"]["status"] == "resolved"
    assert len(target.metadata["tdl_source_claims"]) == 1
    assert (
        target.metadata["ma1_supply_category"]["legacy_track_derived_label"]
        == "Tbio"
    )
    assert second.decision["status"] == "completed"
    assert second.decision["records_added"] == 0
    assert second.decision["records_reused"] == 2
    assert second.decision["records_conflicted"] == 0
    assert {item["status"] for item in second.decision["outcomes"]} == {
        "reused_existing"
    }
    flt3_outcome = target.metadata["pharos_source_outcome"]
    assert flt3_outcome["evidence_ids"] == ["ev_pharos_flt3"]
    assert flt3_outcome["reused_evidence_ids"] == ["ev_pharos_flt3"]
    assert flt3_outcome["provenance_classes"] == ["external_live"]
    assert flt3_outcome["observed_provenance_classes"] == ["external_cache"]
    pharos_record = next(
        item
        for item in second.packet.hypotheses[0].evidence
        if item.evidence_id == "ev_pharos_flt3"
    )
    assert pharos_record.metadata["claim_linkage_status"] == "reconciled"
    assert pharos_record.metadata["claim_linkage_roles"] == ["context_unlinked"]
    assert pharos_record.metadata["source_supports"] == "context"
    assert pharos_record.metadata["partial_source_failures"] == {
        "Europe PMC": "timeout"
    }

    trace = build_run_call_trace(
        RunSnapshot(
            run_id="run_pharos_rework",
            mode=RunMode.MOCK,
            stage=RunStage.MA1,
            status_message="MA1 rework complete",
            brief=_brief("Pharos TDL context"),
            hypothesis_packet=second.packet,
        )
    )
    queries = [
        item
        for item in trace.public_data_calls
        if item.operation == "retrieve_target_development_context"
    ]
    assert {item.status for item in queries} == {"reused_existing"}
    assert all(item.cached is True for item in queries)
    assert all(item.metadata["records_reused"] == 1 for item in queries)
    assert all(item.metadata["evidence_ids"] for item in queries)


def test_same_payload_fixture_cannot_reuse_live_pharos_record() -> None:
    first = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(_packet(), _brief("Pharos TDL context"))
    second = OnDemandPharosRouter(
        _FixtureMarkedLivePharosRetriever(tdl="Tclin")
    ).route(first.packet, _brief("Pharos TDL context"))

    assert second.decision["status"] == "failed"
    assert second.decision["records_added"] == 0
    assert second.decision["records_reused"] == 0
    assert second.decision["records_conflicted"] == 2
    assert {item["status"] for item in second.decision["outcomes"]} == {
        "conflict"
    }
    for outcome in second.decision["outcomes"]:
        conflict = outcome["conflicts"][0]
        assert conflict["existing_provenance_class"] == "external_live"
        assert conflict["incoming_provenance_class"] == "external_live"
        assert conflict["existing_content_hash"] != conflict["incoming_content_hash"]


def test_originless_cache_cannot_reuse_live_pharos_record() -> None:
    first = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(_packet(), _brief("Pharos TDL context"))
    second = OnDemandPharosRouter(
        _OriginlessCachePharosRetriever(tdl="Tclin")
    ).route(first.packet, _brief("Pharos TDL context"))

    assert second.decision["status"] == "failed"
    assert second.decision["records_reused"] == 0
    assert second.decision["records_conflicted"] == 2


@pytest.mark.parametrize(
    "source_supports",
    ["support", None, "context"],
    ids=[
        "source_role_changed",
        "source_role_missing_after_reclassification",
        "stale_source_role_without_reconciliation_contract",
    ],
)
def test_changed_or_missing_source_supports_conflicts_for_pharos_rework(
    source_supports: str | None,
) -> None:
    first = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(_packet(), _brief("Pharos TDL context"))
    for hypothesis in first.packet.hypotheses:
        records = [item for item in hypothesis.evidence if item.source == "Pharos"]
        if not records:
            continue
        record = records[0]
        record.supports = "support"
        if source_supports is not None:
            record.metadata["source_supports"] = source_supports

    second = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(first.packet, _brief("Pharos TDL context"))

    assert second.decision["status"] == "failed"
    assert second.decision["records_reused"] == 0
    assert second.decision["records_conflicted"] == 2


def test_context_source_polarity_cannot_be_promoted_for_pharos_rework() -> None:
    first = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(_packet(), _brief("Pharos TDL context"))
    for hypothesis in first.packet.hypotheses:
        records = [item for item in hypothesis.evidence if item.source == "Pharos"]
        if not records:
            continue
        record = records[0]
        record.supports = "support"
        record.metadata = {
            **record.metadata,
            "source_supports": "context",
            "claim_linkage_status": "reconciled",
            "claim_linkage_policy": "atomic-publication-intervention-endpoint-linkage-v2",
            "linked_claim_ids": [f"claim_{hypothesis.hypothesis_id}"],
            "claim_linkage_roles": ["direct_target_direction_primary"],
            "claim_linkage_scope": "claim_specific",
        }

    second = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(first.packet, _brief("Pharos TDL context"))

    assert second.decision["status"] == "failed"
    assert second.decision["records_reused"] == 0
    assert second.decision["records_conflicted"] == 2


def test_mixed_claim_and_context_roles_cannot_reuse_pharos_record() -> None:
    first = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(_packet(), _brief("Pharos TDL context"))
    for hypothesis in first.packet.hypotheses:
        records = [item for item in hypothesis.evidence if item.source == "Pharos"]
        if not records:
            continue
        record = records[0]
        record.metadata = {
            **record.metadata,
            "source_supports": "context",
            "claim_linkage_status": "reconciled",
            "claim_linkage_policy": "atomic-publication-intervention-endpoint-linkage-v2",
            "linked_claim_ids": [f"claim_{hypothesis.hypothesis_id}"],
            "claim_linkage_roles": [
                "direct_target_direction_primary",
                "context_unlinked",
            ],
            "claim_linkage_scope": "claim_specific",
        }

    second = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(first.packet, _brief("Pharos TDL context"))

    assert second.decision["status"] == "failed"
    assert second.decision["records_reused"] == 0
    assert second.decision["records_conflicted"] == 2


def test_tclin_reconciles_tbio_supply_to_literature_work_without_mutating_track() -> None:
    result = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(_packet(), _brief("Pharos TDL context"))
    hypothesis = next(
        item for item in result.packet.hypotheses if item.target.symbol == "BCL2"
    )

    assert hypothesis.track is HypothesisTrack.TBIO
    assert hypothesis.target.tdl == "Tclin"
    contract = hypothesis.target.metadata["track_tdl_contract"]
    assert contract["hypothesis_track"] == "tbio"
    assert contract["effective_work_track"] == "literature"
    assert contract["namespaces_distinct"] is True

    request = context_request_from_hypothesis_packet(result.packet)
    projected = next(item for item in request.targets if item.symbol == "BCL2")
    assert projected.track is HypothesisTrack.TBIO
    assert projected.effective_work_track is HypothesisTrack.LITERATURE
    assert projected.work_track_basis == "effective_tdl"

    route_evidence = derive_route_evidence(hypothesis, hypothesis.target)
    assert route_evidence["verified_tdl"] == "Tclin"
    assert route_evidence["hypothesis_track"] == "tbio"
    assert route_evidence["effective_work_track"] == "literature"

    critique = Critic().review(_brief("Pharos TDL context"), hypothesis)
    assert critique.accepted is True
    assert "TBIO_CLASSIFICATION_MISSING" not in critique.reason_codes


def test_same_id_with_changed_immutable_pharos_content_is_a_conflict() -> None:
    first = OnDemandPharosRouter(
        _RecordingPharosRetriever(tdl="Tclin")
    ).route(_packet(), _brief("Pharos TDL context"))
    second = OnDemandPharosRouter(
        _ChangedSameIdPharosRetriever(tdl="Tbio")
    ).route(first.packet, _brief("Pharos TDL context"))

    assert second.decision["status"] == "failed"
    assert second.decision["records_added"] == 0
    assert second.decision["records_reused"] == 0
    assert second.decision["records_conflicted"] == 2
    assert {item["status"] for item in second.decision["outcomes"]} == {
        "conflict"
    }
    flt3 = second.packet.hypotheses[0]
    assert flt3.target.tdl is None
    assert flt3.target.metadata["tdl_source_claims"][0]["evidence_id"] == (
        "ev_pharos_flt3"
    )
    reconciliation = flt3.target.metadata["tdl_reconciliation"]
    assert reconciliation["status"] == "unresolved_record_integrity_conflict"
    assert reconciliation["conflict_detected"] is True
    assert reconciliation["conflict_hidden"] is False
    assert reconciliation["source_values"] == ["Tbio", "Tclin"]
    assert reconciliation["record_integrity_conflict_count"] == 1
    conflict = reconciliation["record_integrity_conflicts"][0]
    assert conflict["existing_tdl"] == "Tclin"
    assert conflict["incoming_tdl"] == "Tbio"
    assert conflict["existing_content_hash"] != conflict["incoming_content_hash"]
    assert conflict["disposition"] == (
        "incoming_record_quarantined_existing_record_preserved"
    )
    assert [
        item.evidence_id for item in flt3.evidence if item.source == "Pharos"
    ] == ["ev_pharos_flt3"]


def test_data_policy_can_forbid_explicit_pharos_request() -> None:
    retriever = _RecordingPharosRetriever()
    result = OnDemandPharosRouter(retriever).route(
        _packet(),
        _brief("Check Pharos TDL", allowed_sources=["PubMed"]),
    )

    assert retriever.calls == []
    assert result.decision["status"] == "skipped"
    assert "data policy excludes Pharos" in str(result.decision["reason"])


def test_failed_query_records_error_without_fixture_evidence() -> None:
    retriever = _RecordingPharosRetriever(fail=True)
    result = OnDemandPharosRouter(retriever).route(
        _packet(),
        _brief("druggability with Pharos"),
    )

    assert result.decision["status"] == "failed"
    assert result.decision["records_added"] == 0
    assert {item["status"] for item in result.decision["outcomes"]} == {"error"}
    assert all(
        evidence.source != "Pharos"
        for hypothesis in result.packet.hypotheses
        for evidence in hypothesis.evidence
    )


def test_no_target_match_is_empty_not_transport_error() -> None:
    result = OnDemandPharosRouter(_EmptyPharosRetriever()).route(
        _packet(),
        _brief("Pharos TDL context"),
    )

    assert result.decision["status"] == "partial"
    assert {item["status"] for item in result.decision["outcomes"]} == {"empty"}
    assert all(
        item["source_failures"] == {"Pharos": "empty_result"}
        for item in result.decision["outcomes"]
    )


@pytest.mark.asyncio
async def test_orchestrator_emits_source_routing_interaction_trace(tmp_path) -> None:
    orchestrator = DrugPilotOrchestrator(
        settings=Settings(
            database_path=tmp_path / "pharos-trace.sqlite3",
            public_evidence_enabled=False,
        )
    )
    retriever = _RecordingPharosRetriever()
    orchestrator.pharos_router = OnDemandPharosRouter(retriever)
    orchestrator._trace_event = AsyncMock()

    enriched = await orchestrator._enrich_ma1_pharos(
        "run_pharos_test",
        _packet(),
        _brief("표적 개발 수준과 Pharos 확인"),
    )

    assert enriched.metadata["pharos_routing"]["status"] == "completed"
    call = orchestrator._trace_event.await_args
    assert call.args[1] == "pharos_routing_completed"
    metadata = call.kwargs["metadata"]
    assert metadata["source_routing_decision"]["policy_owner"] == "Orchestrator"
    assert metadata["interaction"]["kind"] == "source_routing_decision"
    assert metadata["interaction"]["backend"] == "pharos_public_api"
    assert len(metadata["interaction"]["evidence_ids"]) == 2


def test_run_call_trace_surfaces_pharos_selection_and_bounded_query_outcomes() -> None:
    result = OnDemandPharosRouter(_RecordingPharosRetriever()).route(
        _packet(),
        _brief("Pharos druggability context"),
    )
    snapshot = RunSnapshot(
        run_id="run_pharos_calls",
        mode=RunMode.MOCK,
        stage=RunStage.MA1,
        status_message="MA1 complete",
        brief=_brief("Pharos druggability context"),
        hypothesis_packet=result.packet,
    )

    trace = build_run_call_trace(snapshot)
    pharos = [item for item in trace.public_data_calls if item.name == "Pharos"]

    assert len(pharos) == 3
    selection = next(item for item in pharos if item.operation == "source_selection")
    assert selection.agent == "Orchestrator"
    assert selection.granularity == "selection_decision"
    assert selection.metadata["requested"] is True
    queries = [
        item
        for item in pharos
        if item.operation == "retrieve_target_development_context"
    ]
    assert len(queries) == 2
    assert {item.metadata["target_symbol"] for item in queries} == {"FLT3", "BCL2"}
    assert all(item.metadata["queried"] is True for item in queries)
    assert all(item.metadata["context_only"] is True for item in queries)


def test_run_call_trace_surfaces_default_pharos_skip_without_query_call() -> None:
    brief = _brief("primary literature only")
    brief.constraints.druggability_required = False
    result = OnDemandPharosRouter(_RecordingPharosRetriever()).route(
        _packet(),
        brief,
    )
    snapshot = RunSnapshot(
        run_id="run_pharos_skip",
        mode=RunMode.MOCK,
        stage=RunStage.MA1,
        status_message="MA1 complete",
        brief=brief,
        hypothesis_packet=result.packet,
    )

    trace = build_run_call_trace(snapshot)
    pharos = [item for item in trace.public_data_calls if item.name == "Pharos"]

    assert len(pharos) == 1
    assert pharos[0].operation == "source_selection"
    assert pharos[0].status == "skipped"
    assert pharos[0].metadata["queried"] is False


@pytest.mark.asyncio
async def test_get_run_calls_returns_pharos_selection_and_query_trace(tmp_path) -> None:
    brief = _brief("Pharos TDL 확인")
    result = OnDemandPharosRouter(_RecordingPharosRetriever()).route(
        _packet(),
        brief,
    )
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "pharos-calls-api.sqlite3",
        public_evidence_enabled=False,
    )
    store = SQLiteRunStore(settings.database_path)
    snapshot = RunSnapshot(
        run_id="run_pharos_calls_api",
        mode=RunMode.MOCK,
        stage=RunStage.MA1,
        status_message="MA1 complete",
        brief=brief,
        hypothesis_packet=result.packet,
    )
    await store.save_snapshot(snapshot)
    app = create_app(settings=settings, store=store)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/runs/run_pharos_calls_api/calls")

    response.raise_for_status()
    pharos = [
        item
        for item in response.json()["public_data_calls"]
        if item["name"] == "Pharos"
    ]
    assert [item["operation"] for item in pharos] == [
        "source_selection",
        "retrieve_target_development_context",
        "retrieve_target_development_context",
    ]
    assert pharos[0]["metadata"]["policy_owner"] == "Orchestrator"
