from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from drugpilot.config import Settings
from drugpilot.domain.models import (
    Claim,
    DataSourcePolicy,
    EvidenceRef,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
    RunMode,
    TargetCandidate,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.tools.context_router import (
    ContextRoutingBudget,
    ContextRoutingResult,
    ContextSourceOutcome,
    ContextTarget,
    ContextTargetSelection,
    TargetContextBundle,
)
from drugpilot.tools.ma1_context_enrichment import (
    CONTEXT_ROUTING_METADATA_KEY,
    ContextTargetMatchError,
    enrich_hypothesis_packet_with_context,
)


def _base_evidence(symbol: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"ev_base_{symbol.lower()}",
        source="PubMed",
        external_id=f"PMID-{symbol}",
        title=f"Primary evidence for {symbol}",
        supports="support",
    )


@pytest.mark.asyncio
async def test_explicit_cell_resolved_kind_survives_single_cell_routing_hint(
    tmp_path,
) -> None:
    orchestrator = DrugPilotOrchestrator(
        settings=Settings(
            mode=RunMode.MOCK,
            database_path=tmp_path / "cell-context.sqlite3",
            mock_step_delay_ms=0,
        )
    )
    orchestrator._trace_event = AsyncMock()  # type: ignore[method-assign]
    brief = ResearchBrief(
        brief_id="brief_cell_resolved_contract",
        research_goal="Evaluate disease-relevant targets in synovial fibroblasts.",
        disease="Rheumatoid arthritis",
        constraints=ProjectConstraints(target_cell_type="synovial fibroblasts"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed", "Human Cell Atlas", "CELLxGENE Census"]
        ),
        human_priorities=["single-cell context"],
    )

    result = await orchestrator._enrich_ma1_context(
        "run_cell_resolved_contract",
        HypothesisPacket(
            brief_id=brief.brief_id,
            hypotheses=[],
        ),
        brief,
    )

    contract = result.metadata["context_routing"]
    assert contract["requested_source_families"] == ["single_cell"]
    assert contract["target_cell_context_kind"] == "cell_resolved"
    assert contract["contract_satisfied"] is False


def _hypothesis(
    symbol: str,
    track: HypothesisTrack,
    *,
    hypothesis_id: str | None = None,
    disease: str = "test disease",
    extra_evidence: list[EvidenceRef] | None = None,
) -> Hypothesis:
    base = _base_evidence(symbol)
    return Hypothesis(
        hypothesis_id=hypothesis_id or f"hyp_{symbol.lower()}_{track.value}",
        track=track,
        disease=disease,
        target=TargetCandidate(symbol=symbol),
        mechanism="A bounded, testable mechanism.",
        intervention_direction="inhibit",
        falsification_test="Reject after the prespecified negative experiment.",
        claims=[
            Claim(
                claim_id=f"claim_{symbol.lower()}",
                text=f"Mechanistic claim for {symbol}",
                evidence_ids=[base.evidence_id],
                counter_evidence_ids=[f"counter_{symbol.lower()}"],
            )
        ],
        evidence=[base, *(extra_evidence or [])],
        score=0.8,
        rank=1,
    )


def _packet(*hypotheses: Hypothesis) -> HypothesisPacket:
    return HypothesisPacket(
        packet_id="hp_context_test",
        brief_id="brief_context_test",
        hypotheses=list(hypotheses),
        metadata={"preexisting": {"labels": ["keep"]}},
    )


def _context_record(
    evidence_id: str,
    source: str,
    *,
    supports: str = "context",
    context_only: bool = True,
    causal_allowed: bool = False,
    counter_allowed: bool = False,
    title: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source=source,
        external_id=f"external-{evidence_id}",
        title=title or f"Context record {evidence_id}",
        supports=supports,
        metadata={
            "context_only": context_only,
            "causal_inference_allowed": causal_allowed,
            "counter_evidence_allowed": counter_allowed,
            "provenance_class": "external_live",
            "retrieval_adapter": "FakeContextAdapter",
            "retrieval_endpoint": "https://example.test/context",
        },
    )


def _outcome(
    source: str,
    family: str,
    status: str = "ok",
    *,
    queried: bool = True,
    record_count: int = 1,
) -> ContextSourceOutcome:
    return ContextSourceOutcome(
        source=source,
        family=family,
        status=status,
        queried=queried,
        record_count=record_count,
    )


def _bundle(
    symbol: str,
    track: HypothesisTrack,
    evidence: list[EvidenceRef],
    *,
    hypothesis_id: str | None = None,
    sources: list[ContextSourceOutcome] | None = None,
) -> TargetContextBundle:
    default_source = "bioRxiv" if track is HypothesisTrack.LITERATURE else "STRING"
    default_family = "preprint" if track is HypothesisTrack.LITERATURE else "network"
    return TargetContextBundle(
        target=ContextTarget(
            symbol=symbol,
            track=track,
            hypothesis_id=hypothesis_id,
            rank=1,
        ),
        sources=sources
        or [_outcome(default_source, default_family, record_count=len(evidence))],
        evidence=evidence,
    )


def _routing(
    *bundles: TargetContextBundle,
    disease: str = "test disease",
    budget_exhausted: bool = True,
) -> ContextRoutingResult:
    selections = [
        ContextTargetSelection(
            target=bundle.target.model_copy(deep=True),
            status="selected",
            detail="test selection",
        )
        for bundle in bundles
    ]
    return ContextRoutingResult(
        disease=disease,
        budget=ContextRoutingBudget(max_source_calls=8),
        target_selection=selections,
        targets=list(bundles),
        source_calls_used=2,
        planned_source_calls=3,
        elapsed_seconds=0.25,
        budget_exhausted=budget_exhausted,
        stop_reason="call_budget" if budget_exhausted else None,
    )


def _claim_links(packet: HypothesisPacket) -> list[tuple[list[str], list[str]]]:
    return [
        (list(claim.evidence_ids), list(claim.counter_evidence_ids))
        for hypothesis in packet.hypotheses
        for claim in hypothesis.claims
    ]


def test_deep_copy_merges_by_id_or_track_symbol_into_evidence_only() -> None:
    literature = _hypothesis(
        "TP53", HypothesisTrack.LITERATURE, hypothesis_id="hyp_literature"
    )
    tbio = _hypothesis("GPR132", HypothesisTrack.TBIO, hypothesis_id="hyp_tbio")
    packet = _packet(literature, tbio)
    input_json = packet.model_dump_json()
    claims_before = _claim_links(packet)
    routing = _routing(
        _bundle(
            "TP53",
            HypothesisTrack.LITERATURE,
            [_context_record("ctx_preprint", "bioRxiv")],
            hypothesis_id="hyp_literature",
            sources=[
                _outcome("bioRxiv", "preprint"),
                _outcome(
                    "ClinicalTrials.gov",
                    "clinical",
                    "skipped_call_budget",
                    queried=False,
                    record_count=0,
                ),
            ],
        ),
        _bundle(
            "gpr132",
            HypothesisTrack.TBIO,
            [_context_record("ctx_network", "STRING")],
            # No hypothesis ID exercises the exact (track, casefolded symbol)
            # fallback without weakening an explicit-ID mismatch.
            hypothesis_id=None,
        ),
    )

    enriched = enrich_hypothesis_packet_with_context(packet, routing)

    assert enriched is not packet
    assert packet.model_dump_json() == input_json
    assert [record.evidence_id for record in enriched.hypotheses[0].evidence] == [
        "ev_base_tp53",
        "ctx_preprint",
    ]
    assert [record.evidence_id for record in enriched.hypotheses[1].evidence] == [
        "ev_base_gpr132",
        "ctx_network",
    ]
    assert _claim_links(enriched) == claims_before
    assert all(
        context_id not in evidence_ids and context_id not in counter_ids
        for evidence_ids, counter_ids in _claim_links(enriched)
        for context_id in ("ctx_preprint", "ctx_network")
    )

    audit = enriched.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["status"] == "budget_exhausted"
    assert audit["source_calls_used"] == 2
    assert audit["planned_source_calls"] == 3
    assert audit["budget_exhausted"] is True
    assert audit["stop_reason"] == "call_budget"
    assert audit["status_counts"] == {"ok": 2, "skipped_call_budget": 1}
    assert audit["source_outcome_counts"] == {
        "ClinicalTrials.gov": 1,
        "STRING": 1,
        "bioRxiv": 1,
    }
    assert [
        (item["source"], item["status"], item["queried"])
        for item in audit["source_outcomes"]
    ] == [
        ("bioRxiv", "ok", True),
        ("ClinicalTrials.gov", "skipped_call_budget", False),
        ("STRING", "ok", True),
    ]
    assert audit["source_outcomes"][0]["target_symbol"] == "TP53"
    assert audit["source_outcomes"][0]["hypothesis_id"] == "hyp_literature"
    assert audit["source_record_counts"] == {"STRING": 1, "bioRxiv": 1}
    assert audit["source_added_counts"] == {"STRING": 1, "bioRxiv": 1}
    assert audit["source_counts"] == {
        "outcomes": {
            "ClinicalTrials.gov": 1,
            "STRING": 1,
            "bioRxiv": 1,
        },
        "records_observed": {"STRING": 1, "bioRxiv": 1},
        "records_added": {"STRING": 1, "bioRxiv": 1},
    }
    assert audit["records_added"] == 2
    assert audit["claim_links_modified"] is False
    assert enriched.metadata["preexisting"] == {"labels": ["keep"]}

    enriched.metadata["preexisting"]["labels"].append("output-only")
    assert packet.metadata["preexisting"] == {"labels": ["keep"]}


def test_unsafe_records_are_quarantined_and_duplicate_ids_are_removed() -> None:
    shared_existing = EvidenceRef(
        evidence_id="shared_id",
        source="PubMed",
        external_id="PMID-SHARED",
        title="Existing primary record",
        supports="support",
    )
    repeated_existing = shared_existing.model_copy(deep=True)
    hypothesis = _hypothesis(
        "TP53",
        HypothesisTrack.LITERATURE,
        extra_evidence=[shared_existing, repeated_existing],
    )
    packet = _packet(hypothesis)
    valid = _context_record("ctx_valid", "bioRxiv")
    unsafe_support = _context_record(
        "ctx_support", "bioRxiv", supports="support", context_only=False
    )
    unsafe_flag = _context_record(
        "ctx_bad_flag", "bioRxiv", counter_allowed=True
    )
    claim_collision = _context_record("ev_base_tp53", "bioRxiv")
    duplicate_conflict = _context_record("shared_id", "bioRxiv")
    routing = _routing(
        _bundle(
            "TP53",
            HypothesisTrack.LITERATURE,
            [
                valid,
                valid.model_copy(deep=True),
                unsafe_support,
                unsafe_flag,
                claim_collision,
                duplicate_conflict,
            ],
            hypothesis_id=hypothesis.hypothesis_id,
        ),
        budget_exhausted=False,
    )
    claims_before = _claim_links(packet)

    enriched = enrich_hypothesis_packet_with_context(packet, routing)

    ids = [record.evidence_id for record in enriched.hypotheses[0].evidence]
    assert ids == ["ev_base_tp53", "shared_id", "ctx_valid"]
    assert len(ids) == len(set(ids))
    assert _claim_links(enriched) == claims_before
    assert packet.hypotheses[0].evidence[-1].evidence_id == "shared_id"
    assert len(packet.hypotheses[0].evidence) == 3

    audit = enriched.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["status"] == "completed_with_quarantine"
    assert audit["records_added"] == 1
    assert audit["records_reused"] == 2
    assert audit["reused_record_ids"] == ["shared_id", "ctx_valid"]
    assert audit["records_conflicted"] == 1
    assert audit["conflicting_record_ids"] == ["shared_id"]
    assert audit["duplicate_ids_removed"] == 0
    assert audit["records_quarantined"] == 4
    assert audit["quarantine_reason_counts"] == {
        "claim_link_id_collision": 1,
        "duplicate_id_conflict": 1,
        "non_context_or_unsafe_flags": 2,
    }
    assert len(audit["quarantined_records"]) == 4
    assert {item["evidence_id"] for item in audit["quarantined_records"]} == {
        "ctx_support",
        "ctx_bad_flag",
        "ev_base_tp53",
        "shared_id",
    }
    conflict_audit = next(
        item
        for item in audit["quarantined_records"]
        if item["reason"] == "duplicate_id_conflict"
    )
    assert conflict_audit["disposition"] == (
        "incoming_conflicting_record_rejected_existing_record_preserved"
    )
    assert conflict_audit["existing_content_hash"] != (
        conflict_audit["incoming_content_hash"]
    )


def test_preexisting_same_id_content_conflict_is_the_only_duplicate_removed() -> None:
    first = _context_record("ctx_conflict", "bioRxiv", title="First content")
    changed = _context_record("ctx_conflict", "bioRxiv", title="Changed content")
    hypothesis = _hypothesis(
        "TP53",
        HypothesisTrack.LITERATURE,
        extra_evidence=[first, changed],
    )
    enriched = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = enriched.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["records_reused"] == 0
    assert audit["records_conflicted"] == 1
    assert audit["duplicate_ids_removed"] == 1
    assert audit["records_quarantined"] == 1
    assert audit["quarantined_records"][0]["disposition"] == (
        "later_preexisting_instance_removed_first_record_preserved"
    )
    assert [
        item.evidence_id for item in enriched.hypotheses[0].evidence
    ].count("ctx_conflict") == 1


@pytest.mark.parametrize(
    ("bundle_id", "bundle_symbol"),
    [
        ("missing_hypothesis", "TP53"),
        ("hyp_tp53_literature", "WRONG_SYMBOL"),
    ],
)
def test_explicit_hypothesis_id_must_exist_and_agree_with_target(
    bundle_id: str,
    bundle_symbol: str,
) -> None:
    packet = _packet(
        _hypothesis(
            "TP53",
            HypothesisTrack.LITERATURE,
            hypothesis_id="hyp_tp53_literature",
        )
    )
    before = packet.model_dump_json()
    routing = _routing(
        _bundle(
            bundle_symbol,
            HypothesisTrack.LITERATURE,
            [_context_record("ctx_one", "bioRxiv")],
            hypothesis_id=bundle_id,
        )
    )

    with pytest.raises(ContextTargetMatchError):
        enrich_hypothesis_packet_with_context(packet, routing)

    assert packet.model_dump_json() == before


def test_track_symbol_fallback_rejects_an_ambiguous_destination() -> None:
    packet = _packet(
        _hypothesis(
            "TP53", HypothesisTrack.LITERATURE, hypothesis_id="hyp_duplicate_a"
        ),
        _hypothesis(
            "tp53", HypothesisTrack.LITERATURE, hypothesis_id="hyp_duplicate_b"
        ),
    )
    routing = _routing(
        _bundle(
            "TP53",
            HypothesisTrack.LITERATURE,
            [_context_record("ctx_one", "bioRxiv")],
            hypothesis_id=None,
        )
    )

    with pytest.raises(ContextTargetMatchError, match="exactly one"):
        enrich_hypothesis_packet_with_context(packet, routing)


def test_routing_disease_must_match_the_packet() -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    packet = _packet(hypothesis)
    routing = _routing(
        _bundle(
            "TP53",
            HypothesisTrack.LITERATURE,
            [_context_record("ctx_one", "bioRxiv")],
            hypothesis_id=hypothesis.hypothesis_id,
        ),
        disease="different disease",
    )

    with pytest.raises(ContextTargetMatchError, match="disease"):
        enrich_hypothesis_packet_with_context(packet, routing)


def test_reapplying_the_same_result_is_idempotent_for_evidence_and_claims() -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    packet = _packet(hypothesis)
    routing = _routing(
        _bundle(
            "TP53",
            HypothesisTrack.LITERATURE,
            [_context_record("ctx_one", "bioRxiv")],
            hypothesis_id=hypothesis.hypothesis_id,
        )
    )

    first = enrich_hypothesis_packet_with_context(packet, routing)
    second = enrich_hypothesis_packet_with_context(first, routing)

    assert [item.evidence_id for item in second.hypotheses[0].evidence] == [
        "ev_base_tp53",
        "ctx_one",
    ]
    assert _claim_links(first) == _claim_links(second)
    assert second.metadata[CONTEXT_ROUTING_METADATA_KEY]["records_added"] == 0
    assert second.metadata[CONTEXT_ROUTING_METADATA_KEY]["records_reused"] == 1
    assert second.metadata[CONTEXT_ROUTING_METADATA_KEY]["reused_record_ids"] == [
        "ctx_one"
    ]
    assert second.metadata[CONTEXT_ROUTING_METADATA_KEY]["records_conflicted"] == 0
    assert second.metadata[CONTEXT_ROUTING_METADATA_KEY]["duplicate_ids_removed"] == 0
    assert second.metadata[CONTEXT_ROUTING_METADATA_KEY]["records_quarantined"] == 0
    assert second.metadata[CONTEXT_ROUTING_METADATA_KEY]["status"] == (
        "budget_exhausted"
    )


def test_retrieval_session_metadata_change_reuses_same_immutable_context_record() -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    base = _context_record("ctx_one", "bioRxiv")
    live = base.model_copy(
        update={
            "excerpt_hash": "excerpt-hash-a",
            "metadata": {
                **base.metadata,
                "excerpt": "Bounded source excerpt.",
                "source_location": "abstract:chars:0-23",
                "content_hash": "content-hash-a",
                "partial_source_failures": {"Europe PMC": "timeout"},
            },
        },
        deep=True,
    )
    cached = live.model_copy(
        update={
            "metadata": {
                **live.metadata,
                "provenance_class": "external_cache",
                "live": False,
                "cache_hit": True,
                "retrieved_live": True,
                "origin_provenance_class": "external_live",
                "origin_content_hash": "content-hash-a",
                "partial_source_failures": {"DGIdb": "empty_result"},
            }
        },
        deep=True,
    )
    first = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [live],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )
    stored = next(
        item
        for item in first.hypotheses[0].evidence
        if item.evidence_id == "ctx_one"
    )
    stored.supports = "context"
    stored.metadata = {
        **stored.metadata,
        "source_supports": "context",
        "claim_linkage_status": "reconciled",
        "claim_linkage_policy": "atomic-publication-intervention-endpoint-linkage-v2",
        "linked_claim_ids": [],
        "claim_linkage_roles": ["context_unlinked"],
        "claim_linkage_scope": "unlinked_context",
        "direct_support_eligible": False,
        "direct_support_rejection_codes": [],
        "verified_direct_primary": False,
        "verified_direct_primary_rejection_codes": [],
    }

    second = enrich_hypothesis_packet_with_context(
        first,
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [cached],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = second.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["status"] == "completed"
    assert audit["records_added"] == 0
    assert audit["records_reused"] == 1
    assert audit["records_conflicted"] == 0
    assert audit["records_quarantined"] == 0
    assert [item.evidence_id for item in second.hypotheses[0].evidence].count(
        "ctx_one"
    ) == 1
    reused = next(
        item
        for item in second.hypotheses[0].evidence
        if item.evidence_id == "ctx_one"
    )
    assert reused.excerpt_hash == "excerpt-hash-a"
    assert reused.metadata["content_hash"] == "content-hash-a"
    assert reused.metadata["partial_source_failures"] == {
        "Europe PMC": "timeout"
    }
    assert reused.metadata["claim_linkage_status"] == "reconciled"
    assert reused.metadata["source_supports"] == "context"


@pytest.mark.parametrize(
    "source_supports",
    ["support", None, "context"],
    ids=[
        "source_role_changed",
        "source_role_missing_after_reclassification",
        "stale_source_role_without_reconciliation_contract",
    ],
)
def test_changed_or_missing_source_supports_conflicts_after_role_reconciliation(
    source_supports: str | None,
) -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    raw = _context_record("ctx_source_role_guard", "bioRxiv")
    first = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [raw],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )
    stored = next(
        item
        for item in first.hypotheses[0].evidence
        if item.evidence_id == "ctx_source_role_guard"
    )
    stored.supports = "support"
    stored.metadata = {
        **stored.metadata,
        "claim_linkage_status": "reconciled",
    }
    if source_supports is not None:
        stored.metadata["source_supports"] = source_supports

    second = enrich_hypothesis_packet_with_context(
        first,
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [raw],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = second.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["records_reused"] == 0
    assert audit["records_conflicted"] == 1
    assert audit["records_quarantined"] == 1


def test_context_source_polarity_cannot_be_promoted_to_claim_support() -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    raw = _context_record("ctx_source_promotion_guard", "bioRxiv")
    first = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [raw],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )
    stored = next(
        item
        for item in first.hypotheses[0].evidence
        if item.evidence_id == raw.evidence_id
    )
    stored.supports = "support"
    stored.metadata = {
        **stored.metadata,
        "source_supports": "context",
        "claim_linkage_status": "reconciled",
        "claim_linkage_policy": "atomic-publication-intervention-endpoint-linkage-v2",
        "linked_claim_ids": ["claim_tp53_direct"],
        "claim_linkage_roles": ["direct_target_direction_primary"],
        "claim_linkage_scope": "claim_specific",
    }

    second = enrich_hypothesis_packet_with_context(
        first,
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [raw],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = second.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["records_reused"] == 0
    assert audit["records_conflicted"] == 1
    assert audit["records_quarantined"] == 1


def test_mixed_claim_and_context_roles_cannot_reuse_context_record() -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    raw = _context_record("ctx_mixed_role_guard", "bioRxiv")
    first = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [raw],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )
    stored = next(
        item
        for item in first.hypotheses[0].evidence
        if item.evidence_id == raw.evidence_id
    )
    stored.metadata = {
        **stored.metadata,
        "source_supports": "context",
        "claim_linkage_status": "reconciled",
        "claim_linkage_policy": "atomic-publication-intervention-endpoint-linkage-v2",
        "linked_claim_ids": ["claim_tp53_direct"],
        "claim_linkage_roles": [
            "direct_target_direction_primary",
            "context_unlinked",
        ],
        "claim_linkage_scope": "claim_specific",
    }

    second = enrich_hypothesis_packet_with_context(
        first,
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [raw],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = second.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["records_reused"] == 0
    assert audit["records_conflicted"] == 1
    assert audit["records_quarantined"] == 1


@pytest.mark.parametrize(
    "boundary_metadata",
    [
        {"fixture": True},
        {"mock": True},
        {"backend_kind": "fixture"},
    ],
    ids=["fixture_flag", "mock_flag", "fixture_backend_kind"],
)
def test_same_payload_fixture_cannot_reuse_live_context_record(
    boundary_metadata: dict[str, object],
) -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    base = _context_record("ctx_trust_boundary", "bioRxiv")
    live = base.model_copy(
        update={
            "excerpt_hash": "excerpt-hash-trust",
            "metadata": {
                **base.metadata,
                "content_hash": "content-hash-trust",
                "live": True,
                "cache_hit": False,
            },
        },
        deep=True,
    )
    first = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [live],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )
    fixture = live.model_copy(
        update={
            "metadata": {
                **live.metadata,
                **boundary_metadata,
            }
        },
        deep=True,
    )

    second = enrich_hypothesis_packet_with_context(
        first,
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [fixture],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = second.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["records_reused"] == 0
    assert audit["records_conflicted"] == 1
    assert audit["records_quarantined"] == 1
    conflict = audit["quarantined_records"][0]
    assert conflict["reason"] == "duplicate_id_conflict"
    assert conflict["existing_content_hash"] != conflict["incoming_content_hash"]
    stored = next(
        item
        for item in second.hypotheses[0].evidence
        if item.evidence_id == "ctx_trust_boundary"
    )
    assert stored.metadata["provenance_class"] == "external_live"


@pytest.mark.parametrize(
    "origin_metadata",
    [
        {},
        {
            "origin_provenance_class": "external_live",
            "origin_content_hash": "wrong-content-hash",
        },
    ],
    ids=["origin_missing", "origin_hash_mismatch"],
)
def test_unverified_cache_cannot_reuse_live_context_record(
    origin_metadata: dict[str, object],
) -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    base = _context_record("ctx_originless_cache", "bioRxiv")
    live = base.model_copy(
        update={
            "metadata": {
                **base.metadata,
                "content_hash": "content-hash-originless",
                "live": True,
                "cache_hit": False,
            }
        },
        deep=True,
    )
    first = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [live],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )
    cache_without_origin = live.model_copy(
        update={
            "metadata": {
                **live.metadata,
                "provenance_class": "external_cache",
                "live": False,
                "cache_hit": True,
                "retrieved_live": True,
                **origin_metadata,
            }
        },
        deep=True,
    )

    second = enrich_hypothesis_packet_with_context(
        first,
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [cache_without_origin],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = second.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["records_reused"] == 0
    assert audit["records_conflicted"] == 1
    assert audit["records_quarantined"] == 1


@pytest.mark.parametrize("changed_field", ["excerpt_hash", "content_hash"])
def test_source_excerpt_or_content_hash_change_remains_a_conflict(
    changed_field: str,
) -> None:
    hypothesis = _hypothesis("TP53", HypothesisTrack.LITERATURE)
    base = _context_record("ctx_hash_guard", "bioRxiv")
    live = base.model_copy(
        update={
            "excerpt_hash": "excerpt-hash-a",
            "metadata": {
                **base.metadata,
                "excerpt": "Bounded source excerpt.",
                "source_location": "abstract:chars:0-23",
                "content_hash": "content-hash-a",
            },
        },
        deep=True,
    )
    first = enrich_hypothesis_packet_with_context(
        _packet(hypothesis),
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [live],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )
    stored = next(
        item
        for item in first.hypotheses[0].evidence
        if item.evidence_id == "ctx_hash_guard"
    )
    stored.metadata = {
        **stored.metadata,
        "claim_linkage_status": "reconciled",
        "claim_linkage_policy": "claim-specific-primary-direction-excerpt-v1",
        "linked_claim_ids": [],
        "claim_linkage_roles": ["context_unlinked"],
        "claim_linkage_scope": "unlinked_context",
    }
    changed = (
        live.model_copy(update={"excerpt_hash": "excerpt-hash-b"}, deep=True)
        if changed_field == "excerpt_hash"
        else live.model_copy(
            update={
                "metadata": {
                    **live.metadata,
                    "content_hash": "content-hash-b",
                }
            },
            deep=True,
        )
    )

    second = enrich_hypothesis_packet_with_context(
        first,
        _routing(
            _bundle(
                "TP53",
                HypothesisTrack.LITERATURE,
                [changed],
                hypothesis_id=hypothesis.hypothesis_id,
            ),
            budget_exhausted=False,
        ),
    )

    audit = second.metadata[CONTEXT_ROUTING_METADATA_KEY]
    assert audit["records_added"] == 0
    assert audit["records_reused"] == 0
    assert audit["records_conflicted"] == 1
    assert audit["records_quarantined"] == 1
    conflict = audit["quarantined_records"][0]
    assert conflict["reason"] == "duplicate_id_conflict"
    assert conflict["existing_content_hash"] != conflict["incoming_content_hash"]
