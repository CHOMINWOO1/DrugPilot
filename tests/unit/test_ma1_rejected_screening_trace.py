from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from drugpilot.agents.ma1 import (
    MA1Agent,
    MockEvidenceRetriever,
    TargetCatalogResult,
    TargetSeed,
    _reconcile_discovery_trace_with_proposal_budget,
)
from drugpilot.config import Settings
from drugpilot.domain.models import (
    DataSourcePolicy,
    EvidenceRef,
    ExcludedHypothesis,
    GateName,
    HypothesisPacket,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
    ReworkOrder,
    RunMode,
)
from drugpilot.orchestration.graph import (
    DrugPilotOrchestrator,
    _hypothesis_packet_scientific_sha256,
    _hypothesis_packet_screening_trace_sha256,
    _ma1_completion_message,
    _version_reworked_hypothesis_packet,
)


class _CountingMockRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.delegate = MockEvidenceRetriever()

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        self.calls.append(seed.symbol)
        return self.delegate.retrieve(brief, seed, iteration)


def test_discovery_trace_reports_retry_budget_reallocation_honestly() -> None:
    trace = (
        {
            "target_symbol": "FRESH1",
            "scope_disposition": "portfolio_candidate",
            "screening_allocation": "scheduled_for_ma1",
            "reason_codes": [],
        },
        {
            "target_symbol": "FRESH2",
            "scope_disposition": "portfolio_candidate",
            "screening_allocation": "scheduled_for_ma1",
            "reason_codes": [],
        },
        {
            "target_symbol": "RETRY1",
            "scope_disposition": "portfolio_candidate",
            "screening_allocation": "previous_revision_screened_result_preserved",
            "reason_codes": ["MA1_PREVIOUS_REVISION_SCREENING_PRESERVED"],
        },
    )
    proposal_catalog = {
        HypothesisTrack.LITERATURE: (
            TargetSeed(
                symbol="FRESH1",
                target_name="fresh target 1",
                track=HypothesisTrack.LITERATURE,
                priority=0.9,
            ),
            TargetSeed(
                symbol="RETRY1",
                target_name="retry target 1",
                track=HypothesisTrack.LITERATURE,
                priority=0.8,
            ),
            TargetSeed(
                symbol="FRESH2",
                target_name="fresh target 2",
                track=HypothesisTrack.LITERATURE,
                priority=0.7,
            ),
        ),
        HypothesisTrack.TBIO: (),
    }

    reconciled = {
        item["target_symbol"]: item
        for item in _reconcile_discovery_trace_with_proposal_budget(
            trace,
            proposal_catalog,
            max_iterations=2,
            retry_symbols={"retry1"},
        )
    }

    assert reconciled["FRESH1"]["screening_allocation"] == "scheduled_for_ma1"
    assert reconciled["RETRY1"]["screening_allocation"] == (
        "scheduled_for_ma1_evidence_retry"
    )
    assert reconciled["FRESH2"]["screening_allocation"] == (
        "not_scheduled_evidence_retry_budget"
    )
    assert reconciled["FRESH2"]["reason_codes"] == [
        "MA1_SCREENING_BUDGET_REALLOCATED_TO_EVIDENCE_RETRY"
    ]


class _BoundedTraceRetriever:
    def __init__(self, record_count: int = 20) -> None:
        self.record_count = record_count
        self.calls: list[str] = []

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        self.calls.append(seed.symbol)
        records: list[EvidenceRef] = []
        for index in range(self.record_count):
            metadata = {
                "pmid": str(100_000 + index),
                "doi": f"10.1000/trace.{index}",
                "query": (
                    "BCL2 AND AML api_key=must-not-survive"
                    if index == 0
                    else "BCL2 AND AML"
                ),
                "retrieval_endpoint": (
                    "https://example.test/search?api_key=must-not-survive"
                ),
                "excerpt": "E" * 400,
                "content_hash": f"content-hash-{index}",
                "content_availability": "abstract_excerpt",
                "abstract_retrieval_status": "available",
                "provenance_class": "external_live",
                "full_abstract": "FULL_ABSTRACT_SENTINEL_ONLY",
            }
            if index == 0:
                metadata["partial_source_failures"] = {
                    "Europe PMC": "TimeoutError"
                }
                metadata["epmc_oa_fulltext_one_hop_trace"] = {
                    "policy": "epmc_oa_fulltext_one_hop_v1",
                    "enabled": True,
                    "status": "completed_with_errors_without_evidence",
                    "stop_reason": "parent_candidates_exhausted",
                    "trigger": {"trigger_scope": "pubmed_pmcid_cross_source"},
                    "cross_source_escalation": True,
                    "requested_pmcids": ["PMC100000"],
                    "resolved_exact_pmcids": ["PMC100000"],
                    "observed": {
                        "added_http_call_count": 1,
                        "parent_fulltexts_attempted": 1,
                        "parent_fulltexts_parsed": 0,
                        "result_record_count": 0,
                    },
                    "error_reason_codes": ["license_not_allowed"],
                    "query": "SECRET_QUERY_MUST_NOT_SURVIVE",
                }
            records.append(
                EvidenceRef(
                    evidence_id=f"ev_trace_{index}",
                    source="PubMed",
                    external_id=str(100_000 + index),
                    title=f"Bounded trace record {index}",
                    supports="context",
                    excerpt_hash=f"excerpt-hash-{index}",
                    metadata=metadata,
                )
            )
        return records


class _SameExternalTraceRetriever(_BoundedTraceRetriever):
    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        records = super().retrieve(brief, seed, iteration)
        return [
            record.model_copy(
                update={
                    "external_id": "shared-external-record",
                    "metadata": {
                        **record.metadata,
                        "pmid": "100000",
                        "doi": "10.1000/shared-trace",
                        "provenance_class": f"external_variant_{index}",
                    },
                },
                deep=True,
            )
            for index, record in enumerate(records)
        ]


def _settings(tmp_path) -> Settings:
    return Settings(
        mode=RunMode.MOCK,
        database_path=tmp_path / "drugpilot.sqlite3",
        mock_step_delay_ms=0,
        max_run_tokens=1_000_000,
        hard_max_run_tokens=5_000_000,
    )


def _brief(*, forbidden_targets: Sequence[str] = ()) -> ResearchBrief:
    return ResearchBrief(
        brief_id="brief_rejected_trace",
        research_goal="Evaluate evidence-linked AML target hypotheses.",
        disease="Acute myeloid leukemia",
        disease_id="EFO_0000222",
        constraints=ProjectConstraints(
            intervention_direction="inhibit",
            max_ma1_iterations=3,
            forbidden_targets=list(forbidden_targets),
        ),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed", "Open Targets"]
        ),
    )


def _retry_attempt_ledger(entry_count: int = 1) -> dict:
    entries = [
        {"target_symbol": f"tgt{index:02d}", "attempt_count": 1}
        for index in range(entry_count)
    ]
    return {
        "schema_version": "ma1_evidence_retry_attempt_ledger/v1",
        "authoritative": True,
        "entry_limit": 64,
        "entry_count_total": entry_count,
        "entry_count_recorded": entry_count,
        "entry_count_omitted": 0,
        "entries": entries,
    }


@pytest.mark.parametrize(
    "case",
    [
        "not_mapping",
        "wrong_schema",
        "not_authoritative",
        "wrong_limit",
        "declared_total_mismatch",
        "recorded_mismatch",
        "omitted",
        "non_mapping_entry",
        "empty_symbol",
        "unnormalized_symbol",
        "duplicate_symbol",
        "zero_count",
        "negative_count",
        "boolean_count",
        "string_count",
    ],
)
def test_incoming_retry_attempt_ledger_fails_closed_when_malformed(
    tmp_path,
    case: str,
) -> None:
    ledger: object = _retry_attempt_ledger()
    if case == "not_mapping":
        ledger = []
    else:
        assert isinstance(ledger, dict)
        if case == "wrong_schema":
            ledger["schema_version"] = "unknown"
        elif case == "not_authoritative":
            ledger["authoritative"] = False
        elif case == "wrong_limit":
            ledger["entry_limit"] = 63
        elif case == "declared_total_mismatch":
            ledger["entry_count_total"] = 2
        elif case == "recorded_mismatch":
            ledger["entry_count_recorded"] = 0
        elif case == "omitted":
            ledger["entry_count_omitted"] = 1
        elif case == "non_mapping_entry":
            ledger["entries"] = ["bad"]
        elif case == "empty_symbol":
            ledger["entries"][0]["target_symbol"] = ""
        elif case == "unnormalized_symbol":
            ledger["entries"][0]["target_symbol"] = " TGT00 "
        elif case == "duplicate_symbol":
            ledger = _retry_attempt_ledger(2)
            ledger["entries"][1]["target_symbol"] = "tgt00"
        elif case == "zero_count":
            ledger["entries"][0]["attempt_count"] = 0
        elif case == "negative_count":
            ledger["entries"][0]["attempt_count"] = -1
        elif case == "boolean_count":
            ledger["entries"][0]["attempt_count"] = True
        elif case == "string_count":
            ledger["entries"][0]["attempt_count"] = "1"
    prior = HypothesisPacket(
        brief_id=_brief().brief_id,
        hypotheses=[],
        metadata={"ma1_evidence_retry_attempt_ledger": ledger},
    )

    with pytest.raises(
        ValueError,
        match="invalid persisted MA1 evidence-retry attempt ledger",
    ):
        MA1Agent(settings=_settings(tmp_path)).run(
            _brief(), revision_context=prior
        )


def test_retry_attempt_ledger_accepts_exactly_64_and_rejects_65(tmp_path) -> None:
    exact = HypothesisPacket(
        brief_id=_brief().brief_id,
        hypotheses=[],
        metadata={
            "ma1_evidence_retry_attempt_ledger": _retry_attempt_ledger(64)
        },
    )
    packet = MA1Agent(settings=_settings(tmp_path)).run(
        _brief(), revision_context=exact
    )
    ledger = packet.metadata["ma1_evidence_retry_attempt_ledger"]
    assert ledger["entry_count_total"] == 64
    assert ledger["entry_count_recorded"] == 64
    assert ledger["entry_count_omitted"] == 0

    over = HypothesisPacket(
        brief_id=_brief().brief_id,
        hypotheses=[],
        metadata={
            "ma1_evidence_retry_attempt_ledger": _retry_attempt_ledger(65)
        },
    )
    with pytest.raises(
        ValueError,
        match="invalid persisted MA1 evidence-retry attempt ledger",
    ):
        MA1Agent(settings=_settings(tmp_path)).run(
            _brief(), revision_context=over
        )


def _seed(symbol: str, track: HypothesisTrack) -> TargetSeed:
    numeric = sum((index + 1) * ord(char) for index, char in enumerate(symbol))
    return TargetSeed(
        symbol=symbol,
        target_name=f"{symbol} target",
        track=track,
        priority=0.9,
        ensembl_id=f"ENSG{numeric % 10**11:011d}",
        uniprot_id=f"P{numeric % 10**5:05d}",
        identity_status="resolved",
        identity_sources=("UniProt",),
        discovery_source="external_dynamic",
    )


def _catalog(*, literature: Sequence[str], tbio: Sequence[str]) -> TargetCatalogResult:
    return TargetCatalogResult(
        catalog={
            HypothesisTrack.LITERATURE: tuple(
                _seed(symbol, HypothesisTrack.LITERATURE) for symbol in literature
            ),
            HypothesisTrack.TBIO: tuple(
                _seed(symbol, HypothesisTrack.TBIO) for symbol in tbio
            ),
        },
        disease_resolution_status="resolved",
        external_seed_count=len(literature) + len(tbio),
    )


def test_g0_rework_reopens_only_evidence_retry_and_calls_flt3_once(
    tmp_path,
) -> None:
    forbidden = ["IDH1", "DNMT3A", "TET2", "CEBPA", "KIT"]
    brief = _brief(forbidden_targets=forbidden)
    retriever = _CountingMockRetriever()
    agent = MA1Agent(settings=_settings(tmp_path), evidence_retriever=retriever)
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(
            literature=["FLT3", "IDH1", "DNMT3A"],
            tbio=["TET2", "CEBPA", "KIT"],
        )
    )
    prior = [
        ExcludedHypothesis(
            target_symbol="FLT3",
            reason_codes=["MISSING_EVIDENCE", "BROKEN_EVIDENCE_LINK"],
            source_iteration=1,
        ),
        *[
            ExcludedHypothesis(
                target_symbol=symbol,
                reason_codes=["FORBIDDEN_TARGET"],
                source_iteration=1,
            )
            for symbol in forbidden
        ],
    ]

    packet = agent.run(brief, revision_context={"excluded": prior})

    assert retriever.calls == ["FLT3"]
    assert packet.metadata["prior_claim_evidence_retry_symbols"] == ["flt3"]
    assert [item.target.symbol for item in packet.hypotheses] == ["FLT3"]
    assert {item.target_symbol for item in packet.excluded} == set(forbidden)
    assert packet.metadata["target_sources_scope"] == "retained_only"
    assert set(packet.metadata["target_sources"]) == {"FLT3"}


def test_rank_three_retry_is_prioritized_and_attempted_before_early_break(
    tmp_path,
) -> None:
    retriever = _CountingMockRetriever()
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=retriever,
    )
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(
            literature=["FLT3", "BCL2", "IDH2"],
            tbio=["LILRB4", "CD300A"],
        )
    )
    prior = ExcludedHypothesis(
        target_symbol="IDH2",
        reason_codes=["MISSING_EVIDENCE"],
        source_iteration=3,
    )

    packet = agent.run(_brief(), revision_context={"excluded": [prior]})

    assert retriever.calls[0] == "IDH2"
    assert retriever.calls.count("IDH2") == 1
    assert "IDH2" in {item.target.symbol for item in packet.hypotheses}
    assert prior not in packet.excluded
    assert packet.metadata["prior_claim_evidence_retry_symbols"] == ["idh2"]
    assert packet.metadata["prior_evidence_retry_statuses"] == [
        {"target_symbol": "idh2", "status": "attempted"}
    ]


def test_proposer_skips_prior_disposition_and_screens_next_scheduled_seed() -> None:
    proposer = MA1Agent(evidence_retriever=_CountingMockRetriever()).proposer
    catalog = _catalog(
        literature=["FLT3", "BCL2", "IDH2"],
        tbio=[],
    ).catalog

    proposals = proposer.propose(
        _brief(),
        iteration=2,
        excluded_symbols={"flt3"},
        accepted_symbols=set(),
        catalog=catalog,
    )

    assert [item.target.symbol for item in proposals] == ["BCL2"]


def test_catalog_missing_retry_preserves_exclusion_and_records_not_attempted(
    tmp_path,
) -> None:
    retriever = _CountingMockRetriever()
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=retriever,
    )
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(
            literature=["FLT3"],
            tbio=["LILRB4"],
        )
    )
    prior = ExcludedHypothesis(
        target_symbol="IDH2",
        reason_codes=["UNSUPPORTED_CLAIM"],
        source_iteration=2,
    )

    packet = agent.run(_brief(), revision_context={"excluded": [prior]})

    assert "IDH2" not in retriever.calls
    assert prior in packet.excluded
    assert packet.metadata["prior_claim_evidence_retry_symbols"] == []
    assert packet.metadata["prior_evidence_retry_catalog_missing_symbols"] == [
        "idh2"
    ]
    assert packet.metadata["prior_evidence_retry_statuses"] == [
        {"target_symbol": "idh2", "status": "catalog_missing"}
    ]


def test_retry_restores_canonical_seed_from_bounded_screening_history(
    tmp_path,
) -> None:
    retriever = _CountingMockRetriever()
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=retriever,
    )
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(
            literature=["FLT3"],
            tbio=["LILRB4"],
        )
    )
    prior = ExcludedHypothesis(
        target_symbol="IDH2",
        reason_codes=["UNSUPPORTED_CLAIM"],
        source_iteration=2,
    )
    prior_packet = HypothesisPacket(
        brief_id=_brief().brief_id,
        hypotheses=[],
        excluded=[prior],
        metadata={
            "ma1_screening_history_sources": [
                {
                    "target_symbol": "IDH2",
                    "target_source": "prior_revision_evidence_retry",
                }
            ],
            "ma1_screening_history_source_count_total": 1,
            "ma1_rejected_screening_history": {
                "entry_count_total": 1,
                "entries": [
                    {
                        "hypothesis_id": "hyp_prior_idh2",
                        "target": {
                            "symbol": "IDH2",
                            "target_name": "isocitrate dehydrogenase 2",
                            "track": "literature",
                            "ensembl_id": "ENSG00000182054",
                            "uniprot_id": "P48735",
                            "identity_status": "resolved",
                            "identity_sources": ["Open Targets", "UniProt"],
                        },
                    }
                ],
            }
        },
    )

    packet = agent.run(_brief(), revision_context=prior_packet)

    assert retriever.calls[0] == "IDH2"
    assert packet.metadata["prior_evidence_retry_restored_symbols"] == ["idh2"]
    assert packet.metadata["prior_evidence_retry_catalog_missing_symbols"] == []
    assert packet.metadata["prior_evidence_retry_state_records"][0][
        "present_in_current_discovery_trace"
    ] is False
    assert packet.metadata["prior_evidence_retry_state_records"][0][
        "current_catalog_priority_available"
    ] is False
    retry_ledger = packet.metadata["ma1_evidence_retry_attempt_ledger"]
    assert retry_ledger["carry_forward_source"] == "legacy_history_bootstrap"
    assert retry_ledger["legacy_history_bootstrap_used"] is True
    assert retry_ledger["current_revision_incremented_symbols"] == ["idh2"]
    assert retry_ledger["entries"] == [
        {"target_symbol": "idh2", "attempt_count": 2}
    ]
    assert packet.metadata["prior_evidence_retry_statuses"] == [
        {"target_symbol": "idh2", "status": "attempted"}
    ]


def test_scheduled_but_unattempted_retry_does_not_increment_ledger(
    tmp_path,
) -> None:
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=_CountingMockRetriever(),
    )
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(
            literature=["FLT3"],
            tbio=[],
        )
    )
    agent.proposer.propose = lambda *args, **kwargs: []  # type: ignore[method-assign]
    prior = HypothesisPacket(
        brief_id=_brief().brief_id,
        hypotheses=[],
        excluded=[
            ExcludedHypothesis(
                target_symbol="IDH2",
                reason_codes=["UNSUPPORTED_CLAIM"],
                source_iteration=1,
            )
        ],
        metadata={
            "ma1_evidence_retry_attempt_ledger": {
                **_retry_attempt_ledger(),
                "entries": [
                    {"target_symbol": "idh2", "attempt_count": 2}
                ],
            },
            "ma1_rejected_screening_history": {
                "entry_count_total": 1,
                "entries": [
                    {
                        "hypothesis_id": "hyp_prior_idh2",
                        "target": {
                            "symbol": "IDH2",
                            "target_name": "isocitrate dehydrogenase 2",
                            "track": "literature",
                            "ensembl_id": "ENSG00000182054",
                            "uniprot_id": "P48735",
                            "identity_status": "resolved",
                            "identity_sources": ["Open Targets", "UniProt"],
                        },
                    }
                ],
            },
        },
    )

    packet = agent.run(_brief(), revision_context=prior)

    assert packet.metadata["prior_evidence_retry_scheduled_symbols"] == [
        "idh2"
    ]
    assert packet.metadata["prior_claim_evidence_retry_symbols"] == []
    assert packet.metadata["prior_evidence_retry_unattempted_symbols"] == [
        "idh2"
    ]
    ledger = packet.metadata["ma1_evidence_retry_attempt_ledger"]
    assert ledger["current_revision_incremented_symbols"] == []
    assert ledger["entries"] == [
        {"target_symbol": "idh2", "attempt_count": 2}
    ]


@pytest.mark.parametrize(
    ("reason_codes", "brief_forbidden"),
    [
        ([], False),
        (["FORBIDDEN_TARGET"], False),
        (["MISSING_EVIDENCE", "HUMAN_EXCLUSION"], False),
        (["UNSUPPORTED_CLAIM", "SCOPE_EXCLUSION"], False),
        (["MISSING_EVIDENCE"], True),
    ],
)
def test_empty_mixed_hard_or_current_scope_exclusion_never_reopens(
    tmp_path,
    reason_codes: list[str],
    brief_forbidden: bool,
) -> None:
    retriever = _CountingMockRetriever()
    brief = _brief(forbidden_targets=["FLT3"] if brief_forbidden else [])
    agent = MA1Agent(settings=_settings(tmp_path), evidence_retriever=retriever)
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(literature=["FLT3"], tbio=[])
    )
    prior = ExcludedHypothesis(
        target_symbol="FLT3",
        reason_codes=reason_codes,
        source_iteration=1,
    )

    packet = agent.run(brief, revision_context={"excluded": [prior]})

    assert retriever.calls == []
    assert packet.hypotheses == []
    assert packet.metadata["prior_claim_evidence_retry_symbols"] == []
    assert packet.excluded == [prior]


def test_rejected_trace_is_bounded_sanitized_deterministic_and_separates_operations(
    tmp_path,
) -> None:
    retriever = _BoundedTraceRetriever(record_count=20)
    brief = _brief(forbidden_targets=["BCL2"])
    agent = MA1Agent(settings=_settings(tmp_path), evidence_retriever=retriever)
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(literature=["BCL2"], tbio=[])
    )

    first = agent.run(brief)
    second = agent.run(brief)

    trace = first.metadata["rejected_screening_trace"]
    assert first.metadata["screening_trace_sha256"] == second.metadata[
        "screening_trace_sha256"
    ]
    assert first.metadata["screening_trace_sha256"] == (
        _hypothesis_packet_screening_trace_sha256(first)
    )
    assert first.metadata["target_sources_scope"] == "retained_only"
    assert first.metadata["target_sources"] == {}
    screened_source = first.metadata["screened_target_sources"][0]
    assert screened_source["disposition"] == "rejected"
    assert screened_source["direction_triage"]["status"] == (
        "unknown_no_direct_primary"
    )
    assert trace["entry_count_total"] == 1
    entry = trace["entries"][0]
    assert entry["target"]["symbol"] == "BCL2"
    assert "FORBIDDEN_TARGET" in entry["reason_codes"]
    assert entry["evidence_count_total"] == 20
    assert entry["unique_external_record_count_total"] == 20
    assert entry["unique_external_record_count_recorded"] == 12
    assert entry["operation_observation_count_total"] >= 1
    assert entry["operation_observation_linkage_policy"] == (
        "claim-linked-visible-external-record-retrieval-first-v2; "
        "aggregate-source-telemetry-secondary"
    )
    assert entry["unique_external_records"]
    assert entry["operation_observations"]
    visible_ids = {
        record["external_id"] for record in entry["unique_external_records"]
    }
    observed_ids = {
        item["external_id"]
        for item in entry["operation_observations"]
        if item.get("observation_kind") == "evidence_record_retrieval"
    }
    assert visible_ids <= observed_ids
    failure_observations = [
        item
        for item in entry["operation_observations"]
        if item.get("observation_kind") == "evidence_source_failure"
    ]
    assert failure_observations
    assert all("endpoint" not in item for item in failure_observations)
    assert all("query" not in item for item in failure_observations)
    assert all(
        item["endpoint_attribution"]
        == "not_available_from_aggregate_failure"
        for item in failure_observations
    )
    assert "claim_assignment" in entry
    assert entry["direction_triage"] == screened_source["direction_triage"]
    assert entry["direction_triage"]["basis"] == (
        "no_linked_direct_target_direction_primary"
    )
    assert entry["direction_triage"]["inference_forbidden"] is True
    first_unit = entry["unique_external_records"][0]["evidence_units"][0]
    assert first_unit["oa_fulltext_trace"] == {
        "policy": "epmc_oa_fulltext_one_hop_v1",
        "enabled": True,
        "status": "completed_with_errors_without_evidence",
        "stop_reason": "parent_candidates_exhausted",
        "trigger_scope": "pubmed_pmcid_cross_source",
        "cross_source_escalation": True,
        "requested_pmcids": ["PMC100000"],
        "resolved_exact_pmcids": ["PMC100000"],
        "http_calls": 1,
        "parent_fulltexts_attempted": 1,
        "parent_fulltexts_parsed": 0,
        "result_record_count": 0,
        "error_reason_codes": ["license_not_allowed"],
    }
    for record in entry["unique_external_records"]:
        for unit in record["evidence_units"]:
            assert len(unit.get("bounded_excerpt", "")) <= 320
            assert "?" not in unit.get("endpoint", "")
    serialized = json.dumps(trace, ensure_ascii=False)
    assert "must-not-survive" not in serialized
    assert "FULL_ABSTRACT_SENTINEL_ONLY" not in serialized
    assert "SECRET_QUERY_MUST_NOT_SURVIVE" not in serialized
    assert "[redacted]" in serialized


def test_same_external_record_caps_provenance_classes_and_evidence_units(
    tmp_path,
) -> None:
    retriever = _SameExternalTraceRetriever(record_count=20)
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=retriever,
    )
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(literature=["BCL2"], tbio=[])
    )

    packet = agent.run(_brief(forbidden_targets=["BCL2"]))

    entry = packet.metadata["rejected_screening_trace"]["entries"][0]
    assert entry["unique_external_record_count_total"] == 1
    record = entry["unique_external_records"][0]
    assert record["evidence_unit_count_total"] == 20
    assert record["evidence_unit_count_recorded"] == 4
    assert record["evidence_unit_count_omitted"] == 16
    assert len(record["evidence_units"]) == 4
    assert record["provenance_class_count_total"] == 20
    assert record["provenance_class_count_recorded"] == 8
    assert record["provenance_class_count_omitted"] == 12
    assert len(record["provenance_classes"]) == 8


def test_pubmed_and_europe_pmc_same_pmid_count_as_one_publication(
    tmp_path,
) -> None:
    class DuplicatePublicationRetriever:
        def retrieve(self, brief, seed, iteration):
            return [
                EvidenceRef(
                    evidence_id="ev_pubmed_shared",
                    source="PubMed",
                    external_id="34810129",
                    title="Shared publication",
                    supports="context",
                    metadata={
                        "pmid": "34810129",
                        "doi": "10.1000/shared",
                        "provenance_class": "external_live",
                    },
                ),
                EvidenceRef(
                    evidence_id="ev_epmc_shared",
                    source="Europe PMC",
                    external_id="34810129",
                    title="Shared publication",
                    supports="context",
                    metadata={
                        "pmid": "34810129",
                        "doi": "10.1000/shared",
                        "provenance_class": "external_live",
                    },
                ),
            ]

    brief = _brief(forbidden_targets=["BCL2"]).model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["PubMed", "Europe PMC", "Open Targets"]
            )
        },
        deep=True,
    )
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=DuplicatePublicationRetriever(),
    )
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(literature=["BCL2"], tbio=[])
    )

    packet = agent.run(brief)
    source = packet.metadata["screened_target_sources"][0]
    entry = packet.metadata["rejected_screening_trace"]["entries"][0]
    record = entry["unique_external_records"][0]

    assert source["evidence_count"] == 1
    assert source["evidence_unit_count"] == 2
    assert entry["unique_external_record_count_total"] == 1
    assert entry["evidence_count_total"] == 1
    assert entry["evidence_unit_count_total"] == 2
    assert record["sources"] == ["PubMed", "Europe PMC"]
    assert record["evidence_unit_count_total"] == 2


def test_rework_preserves_bounded_prior_screening_history(tmp_path) -> None:
    brief = _brief(forbidden_targets=["BCL2"])
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=_BoundedTraceRetriever(record_count=2),
    )
    agent._prepare_target_catalog = (  # type: ignore[method-assign]
        lambda brief, per_track_limit: _catalog(literature=["BCL2"], tbio=[])
    )

    first = agent.run(brief)
    revised = agent.run(brief, revision_context=first)

    assert revised.metadata["ma1_revision_attempt"] == 1
    assert revised.metadata["ma1_screening_history_source_count_total"] == 1
    assert revised.metadata["ma1_screening_history_sources"][0][
        "record_origin"
    ] == "prior_revision_screening"
    history = revised.metadata["ma1_rejected_screening_history"]
    assert history["entry_count_total"] == 1
    assert history["entries"][0]["record_origin"] == "prior_revision_screening"
    provenance = revised.metadata["ma1_exclusion_provenance"][0]
    assert provenance["record_origin"] == (
        "prior_packet_screened_preserved_not_rescreened"
    )
    assert provenance["scientific_rejection_replayed"] is False


def test_screening_trace_changes_audit_lineage_without_changing_scientific_digest() -> None:
    parent = HypothesisPacket(
        packet_id="hp_parent",
        brief_id="brief_lineage",
        hypotheses=[],
        metadata={
            "rejected_screening_trace": {
                "schema_version": "1.0",
                "entries": [{"target": {"symbol": "FLT3"}}],
            }
        },
    )
    revised = parent.model_copy(
        update={
            "metadata": {
                "rejected_screening_trace": {
                    "schema_version": "1.0",
                    "entries": [{"target": {"symbol": "BCL2"}}],
                }
            }
        },
        deep=True,
    )
    order = ReworkOrder(
        gate=GateName.G0,
        target_stage="ma1",
        failed_node="g0.audit",
        required_actions=["Retry evidence linkage."],
    )

    assert _hypothesis_packet_scientific_sha256(parent) == (
        _hypothesis_packet_scientific_sha256(revised)
    )
    versioned = _version_reworked_hypothesis_packet(revised, parent, order)
    lineage = versioned.metadata["revision_lineage"]
    assert lineage["scientific_changed"] is False
    assert lineage["audit_trace_changed"] is True
    assert lineage["parent_screening_trace_sha256"] != lineage[
        "screening_trace_sha256"
    ]


def test_valid_zero_hypotheses_message_differs_from_genuine_intake_failure() -> None:
    packet = HypothesisPacket(brief_id="brief_zero", hypotheses=[])

    assert _ma1_completion_message(
        packet, intake_abstain_recommended=False
    ) == (
        "MA1 screened candidates but retained zero hypotheses after evidence "
        "and constraint review."
    )
    assert _ma1_completion_message(
        packet, intake_abstain_recommended=True
    ) == "MA1 withheld hypotheses because disease intake was not verified."


def test_scientifically_identical_rework_is_not_repeated() -> None:
    packet = HypothesisPacket(
        brief_id="brief_noop_rework",
        hypotheses=[],
        metadata={
            "revision_lineage": {
                "scientific_changed": False,
                "audit_trace_changed": True,
            }
        },
    )
    brief = _brief().model_copy(
        update={
            "constraints": _brief().constraints.model_copy(
                update={"max_rework_loops": 3},
                deep=True,
            )
        },
        deep=True,
    )

    assert DrugPilotOrchestrator._can_rework(
        {
            "brief": brief,
            "hypothesis_packet": packet,
            "rework_loops": 1,
        }
    ) is False
