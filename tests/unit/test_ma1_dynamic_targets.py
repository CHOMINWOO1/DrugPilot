from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pytest

from drugpilot.agents.ma1 import (
    MA1Agent,
    MockEvidenceRetriever,
    TargetSeed,
    _select_supplementary_evidence_retry_catalog,
)
from drugpilot.config import Settings
from drugpilot.domain.models import (
    ExcludedHypothesis,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
    RunMode,
)
from drugpilot.tools.disease_targets import (
    AssociationEvidenceSummary,
    DataTypeScore,
    DiseaseResolution,
    DiseaseTargetCandidate,
    DiseaseTargetDiscoveryResult,
    SourceStatus,
    TargetProvenance,
    TargetSourceRecord,
    TractabilityAssessment,
)
from drugpilot.tools.target_identity import (
    CanonicalTargetIdentity,
    IdentitySourceResult,
    TargetIdentityQuery,
    TargetIdentityResolution,
)


@dataclass
class FakeTargetDiscovery:
    result: DiseaseTargetDiscoveryResult | None = None
    error: Exception | None = None
    calls: int = 0
    arguments: tuple[str, str | None, int | None] | None = None
    allowed_sources: tuple[str, ...] | None = None

    def discover(
        self,
        disease: str,
        disease_id: str | None = None,
        *,
        max_targets: int | None = None,
        allowed_sources=None,
    ) -> DiseaseTargetDiscoveryResult:
        self.calls += 1
        self.arguments = (disease, disease_id, max_targets)
        self.allowed_sources = (
            tuple(allowed_sources) if allowed_sources is not None else None
        )
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result.model_copy(deep=True)


@dataclass
class WindowedTargetDiscovery:
    candidates: tuple[DiseaseTargetCandidate, ...]
    requested_limits: list[int]

    def discover(
        self,
        disease: str,
        disease_id: str | None = None,
        *,
        max_targets: int | None = None,
        allowed_sources=None,
    ) -> DiseaseTargetDiscoveryResult:
        del disease, disease_id, allowed_sources
        limit = max_targets if max_targets is not None else len(self.candidates)
        self.requested_limits.append(limit)
        return _result(self.candidates[:limit]).model_copy(
            update={"max_targets": limit},
            deep=True,
        )


class EmptyEvidenceRetriever:
    def retrieve(self, brief, seed, iteration):
        del brief, seed, iteration
        return ()


class RecordingEmptyEvidenceRetriever:
    def __init__(self) -> None:
        self.seeds: list[TargetSeed] = []

    def retrieve(self, brief, seed, iteration):
        del brief, iteration
        self.seeds.append(seed)
        return ()


class FakeIdentityResolver:
    def __init__(self, identities: Mapping[str, tuple[str, str, str]]) -> None:
        self.identities = dict(identities)
        self.calls: list[tuple[str, str | None]] = []
        self.allowed_sources: list[tuple[str, ...] | None] = []

    def resolve(
        self,
        gene_symbol: str,
        ensembl_id: str | None = None,
        *,
        allowed_sources=None,
    ) -> TargetIdentityResolution:
        self.calls.append((gene_symbol, ensembl_id))
        self.allowed_sources.append(
            tuple(allowed_sources) if allowed_sources is not None else None
        )
        canonical_symbol, canonical_ensembl, uniprot = self.identities[
            gene_symbol.upper()
        ]
        return TargetIdentityResolution(
            query=TargetIdentityQuery(
                gene_symbol=gene_symbol,
                ensembl_id=ensembl_id,
            ),
            status="resolved",
            identity=CanonicalTargetIdentity(
                canonical_symbol=canonical_symbol,
                ensembl_id=canonical_ensembl,
                uniprot_accession=uniprot,
                uniprot_entry_name=f"{canonical_symbol}_HUMAN",
                protein_name=f"Canonical {canonical_symbol} protein",
            ),
            sources=[
                IdentitySourceResult(
                    source="Open Targets",
                    status="ok",
                    queried=True,
                    provenance_class="external_live",
                    query=canonical_ensembl,
                    matched_by="ensembl_id",
                    canonical_symbol=canonical_symbol,
                    ensembl_id=canonical_ensembl,
                    endpoint="https://api.platform.opentargets.org/api/v4/graphql",
                ),
                IdentitySourceResult(
                    source="UniProt",
                    status="ok",
                    queried=True,
                    provenance_class="external_live",
                    query=canonical_symbol,
                    matched_by="gene_symbol",
                    canonical_symbol=canonical_symbol,
                    ensembl_id=canonical_ensembl,
                    uniprot_accession=uniprot,
                    endpoint="https://rest.uniprot.org/uniprotkb/search",
                ),
            ],
        )


@dataclass
class StaticIdentityResolver:
    resolution: TargetIdentityResolution
    calls: list[tuple[str, str | None]]

    def resolve(
        self,
        gene_symbol: str,
        ensembl_id: str | None = None,
        *,
        allowed_sources=None,
    ) -> TargetIdentityResolution:
        del allowed_sources
        self.calls.append((gene_symbol, ensembl_id))
        return self.resolution.model_copy(deep=True)


def _settings(tmp_path, **updates: object) -> Settings:
    values: dict[str, object] = {
        "mode": RunMode.MOCK,
        "database_path": tmp_path / "dynamic-ma1.sqlite3",
        "mock_step_delay_ms": 0,
        "max_ma1_iterations": 2,
        "dynamic_target_discovery_enabled": True,
        "dynamic_target_max_candidates": 8,
        "dynamic_target_fallback_enabled": True,
        # This suite contains explicit fallback-behaviour demonstrations.  The
        # production/default setting is false and is covered by intake tests.
        "demo_fixture_fallback_enabled": True,
        "target_identity_resolution_enabled": True,
    }
    values.update(updates)
    return Settings(**values)


def _brief() -> ResearchBrief:
    return ResearchBrief(
        brief_id="brief_dynamic_colon",
        research_goal=(
            "Identify disease-specific and falsifiable therapeutic targets for "
            "microsatellite-stable colorectal cancer."
        ),
        disease="microsatellite-stable colorectal cancer",
        disease_id="EFO_0021782",
        constraints=ProjectConstraints(max_ma1_iterations=2),
    )


def _source_record(
    *,
    symbol: str,
    ensembl_id: str,
    disease_id: str,
    relationship: str = "ranked_association",
    source: str = "Open Targets",
    record_id: str | None = None,
) -> TargetSourceRecord:
    if relationship == "ranked_association":
        return TargetSourceRecord(
            source="Open Targets",
            record_id=record_id or f"{disease_id}:{ensembl_id}",
            record_type="target_disease_association",
            title=f"{symbol} association with colorectal cancer",
            url=f"https://platform.opentargets.org/evidence/{ensembl_id}/{disease_id}",
            relationship="ranked_association",
            query=f"disease(efoId: {disease_id}) -> associatedTargets",
            retrieval_endpoint="https://api.platform.opentargets.org/api/v4/graphql",
            retrieval_mode="live",
            score=0.8,
            metadata={"score_scope": "association ranking heuristic"},
        )
    assert source == "PubMed"
    return TargetSourceRecord(
        source="PubMed",
        record_id=record_id or "12345678",
        record_type="publication_metadata",
        title=f"{symbol} in colorectal cancer",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{record_id or '12345678'}/",
        relationship="co_mention_context_only",
        query=f'"{symbol}" AND "colorectal cancer"',
        retrieval_endpoint=(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ),
        retrieval_mode="live",
        metadata={"evidence_scope": "title co-mention only"},
    )


def _candidate(
    *,
    rank: int,
    symbol: str,
    ensembl_id: str,
    track: str,
    score: float | None,
    include_pubmed: bool = False,
) -> DiseaseTargetCandidate:
    disease_id = "EFO_0021782"
    records = [
        _source_record(
            symbol=symbol,
            ensembl_id=ensembl_id,
            disease_id=disease_id,
        )
    ]
    if include_pubmed:
        records.append(
            _source_record(
                symbol=symbol,
                ensembl_id=ensembl_id,
                disease_id=disease_id,
                relationship="co_mention_context_only",
                source="PubMed",
                record_id=f"9000000{rank}",
            )
        )
    datatype = (
        (DataTypeScore(evidence_type="literature", score=0.7),)
        if track == "literature"
        else (DataTypeScore(evidence_type="genetic_association", score=0.75),)
    )
    return DiseaseTargetCandidate(
        rank=rank,
        gene_symbol=symbol,
        ensembl_target_id=ensembl_id,
        target_name=f"Discovered {symbol}",
        biotype="protein_coding",
        association_score=score,
        evidence_summary=AssociationEvidenceSummary(
            overall_association_score=score,
            datatype_scores=datatype,
            known_drug_score=None,
            tractability=(
                TractabilityAssessment(label="High Quality Ligand", modality="SM", value=True),
            ),
            pubmed_record_count=1 if include_pubmed else 0,
            score_interpretation="Ranking heuristic, not causal proof.",
        ),
        source_records=tuple(records),
        provenance=TargetProvenance(
            discovery_method="opentargets_ranked_association",
            primary_source="Open Targets",
            contributing_sources=(
                ("Open Targets", "PubMed")
                if include_pubmed
                else ("Open Targets",)
            ),
            causality_interpretation="association_score_is_not_causal_proof",
        ),
        track_hint=track,
        track_hint_basis=f"Test evidence supports the {track} supply track.",
    )


def _result(candidates: tuple[DiseaseTargetCandidate, ...]) -> DiseaseTargetDiscoveryResult:
    return DiseaseTargetDiscoveryResult(
        query_disease="microsatellite-stable colorectal cancer",
        requested_disease_id="EFO_0021782",
        resolved_disease=DiseaseResolution(
            disease_id="EFO_0021782",
            disease_name="microsatellite-stable colorectal cancer",
            description="A colorectal cancer subtype.",
            match_method="provided_id",
            retrieval_mode="live",
        ),
        candidates=candidates,
        source_statuses=(
            SourceStatus(
                source="Open Targets",
                status="live",
                operations=("resolve_disease_id", "associated_targets"),
                records_observed=len(candidates) + 1,
                cache_hits=0,
            ),
            SourceStatus(
                source="PubMed",
                status="live",
                operations=("search_publications", "publication_metadata"),
                records_observed=sum(
                    candidate.evidence_summary.pubmed_record_count
                    for candidate in candidates
                ),
                cache_hits=0,
            ),
        ),
        association_total=len(candidates),
        max_targets=8,
    )


def _static_ret_identity_resolution(kind: str) -> TargetIdentityResolution:
    if kind == "partial":
        return TargetIdentityResolution(
            query=TargetIdentityQuery(gene_symbol="RET"),
            status="partial",
            identity=CanonicalTargetIdentity(
                canonical_symbol="RET",
                ensembl_id="ENSG00000165731",
            ),
            sources=[
                IdentitySourceResult(
                    source="Open Targets",
                    status="ok",
                    queried=True,
                    provenance_class="external_live",
                    query="RET",
                    matched_by="gene_symbol",
                    canonical_symbol="RET",
                    ensembl_id="ENSG00000165731",
                )
            ],
        )
    if kind == "fixture":
        return TargetIdentityResolution(
            query=TargetIdentityQuery(gene_symbol="RET"),
            status="fixture",
            identity=CanonicalTargetIdentity(
                canonical_symbol="RET",
                ensembl_id="ENSG00000165731",
                uniprot_accession="P07949",
            ),
            sources=[
                IdentitySourceResult(
                    source="fixture",
                    status="ok",
                    queried=False,
                    provenance_class="fixture_fallback",
                    query="RET",
                    matched_by="fixture",
                    canonical_symbol="RET",
                    ensembl_id="ENSG00000165731",
                    uniprot_accession="P07949",
                )
            ],
        )
    assert kind == "mismatch"
    return TargetIdentityResolution(
        query=TargetIdentityQuery(gene_symbol="RET"),
        status="resolved",
        identity=CanonicalTargetIdentity(
            canonical_symbol="ALK",
            ensembl_id="ENSG00000171094",
            uniprot_accession="Q9UM73",
        ),
        sources=[
            IdentitySourceResult(
                source="UniProt",
                status="ok",
                queried=True,
                provenance_class="external_live",
                query="ALK",
                matched_by="gene_symbol",
                canonical_symbol="ALK",
                ensembl_id="ENSG00000171094",
                uniprot_accession="Q9UM73",
            )
        ],
    )


def test_non_aml_disease_uses_dynamic_balanced_targets_and_normalized_ids(
    tmp_path,
) -> None:
    candidates = (
        _candidate(
            rank=1,
            symbol="esr1",
            ensembl_id="ENSG00000091831",
            track="literature",
            score=0.91,
            include_pubmed=True,
        ),
        _candidate(
            rank=2,
            symbol="PGR",
            ensembl_id="ENSG00000082175",
            track="literature",
            score=0.83,
            include_pubmed=True,
        ),
        _candidate(
            rank=3,
            symbol="CDK4",
            ensembl_id="ENSG00000135446",
            track="tbio",
            score=0.79,
        ),
        _candidate(
            rank=4,
            symbol="AURKA",
            ensembl_id="ENSG00000087586",
            track="tbio",
            score=0.74,
        ),
    )
    discovery = FakeTargetDiscovery(result=_result(candidates))
    identity = FakeIdentityResolver(
        {
            "ESR1": ("ESR1", "ENSG00000091831", "P03372"),
            "PGR": ("PGR", "ENSG00000082175", "P06401"),
            "CDK4": ("CDK4", "ENSG00000135446", "P11802"),
            "AURKA": ("AURKA", "ENSG00000087586", "O14965"),
        }
    )
    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
        identity_resolver=identity,
    )

    brief = _brief()
    packet = agent.run(brief)

    assert discovery.calls == 1
    assert discovery.arguments == (
        "microsatellite-stable colorectal cancer",
        "EFO_0021782",
        8,
    )
    assert discovery.allowed_sources == tuple(brief.data_policy.allowed_sources)
    assert len(identity.calls) == 4
    assert identity.allowed_sources == [
        tuple(brief.data_policy.allowed_sources)
    ] * 4
    assert ("esr1", "ENSG00000091831") in identity.calls
    assert {item.target.symbol for item in packet.hypotheses} == {
        "ESR1",
        "PGR",
        "CDK4",
        "AURKA",
    }
    assert not {"STAT3", "MAPK1", "PIK3CA", "GPR132"} & {
        item.target.symbol for item in packet.hypotheses
    }
    assert {
        track: sum(item.track is track for item in packet.hypotheses)
        for track in HypothesisTrack
    } == {HypothesisTrack.LITERATURE: 2, HypothesisTrack.TBIO: 2}

    for hypothesis in packet.hypotheses:
        assert hypothesis.target.ensembl_id is not None
        assert hypothesis.target.uniprot_id is not None
        assert hypothesis.target.identity_status == "resolved"
        assert hypothesis.target.identity_sources == ["Open Targets", "UniProt"]
        assert hypothesis.target.metadata["target_discovery_source"] == "external_dynamic"
        ranked_records = [
            item
            for item in hypothesis.evidence
            if item.metadata.get("relationship") == "ranked_association"
        ]
        assert ranked_records and all(item.supports == "context" for item in ranked_records)
        assert all(
            item.metadata.get("claim_linkage_roles")
            == ["ranked_association_discovery_context_unlinked"]
            for item in ranked_records
        )
        assert any(item.supports == "counter" for item in hypothesis.evidence)

    esr1 = next(item for item in packet.hypotheses if item.target.symbol == "ESR1")
    pubmed_context = [
        item
        for item in esr1.evidence
        if item.source == "PubMed"
        and item.metadata.get("relationship") == "co_mention_context_only"
    ]
    assert pubmed_context
    assert all(item.supports == "context" for item in pubmed_context)
    assert all(
        item.metadata["causality_scope"] == "context only; no causality inferred"
        for item in pubmed_context
    )
    claim_support_ids = {
        evidence_id for claim in esr1.claims for evidence_id in claim.evidence_ids
    }
    assert not claim_support_ids & {item.evidence_id for item in pubmed_context}

    assert packet.metadata["resolved_disease"]["disease_id"] == "EFO_0021782"
    assert len(packet.metadata["target_source_statuses"]) == 2
    assert packet.metadata["target_fallback_used"] is False
    assert all(
        item["target_source"] == "external_dynamic"
        for item in packet.metadata["target_sources"].values()
    )


def test_external_shortage_is_filled_with_disjoint_explicit_fixture_targets(
    tmp_path,
) -> None:
    external = _candidate(
        rank=1,
        symbol="ERBB2",
        ensembl_id="ENSG00000141736",
        track="literature",
        score=0.87,
        include_pubmed=True,
    )
    # Duplicate identity in another track must not create cross-track leakage.
    duplicate = external.model_copy(
        update={"rank": 2, "track_hint": "tbio", "track_hint_basis": "duplicate"},
        deep=True,
    )
    discovery = FakeTargetDiscovery(result=_result((external, duplicate)))
    settings = _settings(tmp_path, target_identity_resolution_enabled=False)

    packet = MA1Agent(
        settings=settings,
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
    ).run(_brief())

    assert discovery.calls == 1
    symbols_by_track = {
        track: {item.target.symbol for item in packet.hypotheses if item.track is track}
        for track in HypothesisTrack
    }
    assert symbols_by_track[HypothesisTrack.LITERATURE].isdisjoint(
        symbols_by_track[HypothesisTrack.TBIO]
    )
    assert "ERBB2" in symbols_by_track[HypothesisTrack.LITERATURE]
    assert "ERBB2" not in symbols_by_track[HypothesisTrack.TBIO]
    assert all(len(symbols) == 2 for symbols in symbols_by_track.values())

    sources = packet.metadata["target_sources"]
    assert sources["ERBB2"]["target_source"] == "external_dynamic"
    fixture_symbols = {
        symbol
        for symbol, metadata in sources.items()
        if metadata["target_source"] == "fixture_fallback"
    }
    assert len(fixture_symbols) == 3
    assert packet.metadata["target_fallback_used"] is True
    for hypothesis in packet.hypotheses:
        if hypothesis.target.symbol in fixture_symbols:
            assert hypothesis.target.identity_status == "fixture"
            assert hypothesis.target.identity_sources == ["fixture"]
            assert hypothesis.target.metadata["requires_live_verification"] is True


def test_tbio_heavy_external_results_are_reallocated_without_generic_fallback(
    tmp_path,
) -> None:
    candidates = (
        _candidate(
            rank=1,
            symbol="KRAS",
            ensembl_id="ENSG00000133703",
            track="tbio",
            score=0.94,
        ),
        _candidate(
            rank=2,
            symbol="BRAF",
            ensembl_id="ENSG00000157764",
            track="tbio",
            score=0.90,
        ),
        _candidate(
            rank=3,
            symbol="PIK3CA",
            ensembl_id="ENSG00000121879",
            track="tbio",
            score=0.86,
        ),
        _candidate(
            rank=4,
            symbol="PTEN",
            ensembl_id="ENSG00000171862",
            track="tbio",
            score=0.82,
        ),
    )
    discovery = FakeTargetDiscovery(result=_result(candidates))
    packet = MA1Agent(
        settings=_settings(tmp_path, target_identity_resolution_enabled=False),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
    ).run(_brief())

    assert discovery.calls == 1
    assert {item.target.symbol for item in packet.hypotheses} == {
        "KRAS",
        "BRAF",
        "PIK3CA",
        "PTEN",
    }
    assert {
        track: sum(item.track is track for item in packet.hypotheses)
        for track in HypothesisTrack
    } == {HypothesisTrack.LITERATURE: 2, HypothesisTrack.TBIO: 2}
    literature = [
        item for item in packet.hypotheses if item.track is HypothesisTrack.LITERATURE
    ]
    assert {item.target.symbol for item in literature} == {"PIK3CA", "PTEN"}
    assert all(
        item.target.metadata["original_track_hint"] == "tbio"
        and item.target.metadata["assignment_basis"] == "balanced_reallocation"
        and item.target.metadata["assigned_track"] == "literature"
        for item in literature
    )
    tbio = [item for item in packet.hypotheses if item.track is HypothesisTrack.TBIO]
    assert all(
        item.target.metadata["assignment_basis"] == "explicit_track_hint"
        for item in tbio
    )
    assert packet.metadata["external_target_seed_count"] == 4
    assert packet.metadata["fallback_target_seed_count"] == 0


@pytest.mark.parametrize("max_iterations", [1, 3])
def test_screening_budget_is_independent_from_ma1_round_count(
    tmp_path,
    max_iterations: int,
) -> None:
    symbols = [
        "TGT1",
        "TGT2",
        "TGT3",
        "TGT4",
        "TGT5",
        "TGT6",
        "TGT7",
        "TGT8",
    ]
    candidates = tuple(
        _candidate(
            rank=index,
            symbol=symbol,
            ensembl_id=f"ENSG{index:011d}",
            track="literature" if index % 2 else "tbio",
            score=0.95 - index / 100,
            include_pubmed=index % 2 == 1,
        )
        for index, symbol in enumerate(symbols, start=1)
    )
    brief = _brief().model_copy(
        update={
            "constraints": ProjectConstraints(max_ma1_iterations=max_iterations),
        },
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(
            tmp_path,
            max_ma1_iterations=max_iterations,
            target_identity_resolution_enabled=False,
        ),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=FakeTargetDiscovery(result=_result(candidates)),
    ).run(brief)

    screened = {
        item["target_symbol"] for item in packet.metadata["screened_target_sources"]
    }
    trace = packet.metadata["ma1_discovery_candidate_trace"]

    assert screened == set(symbols)
    assert len(trace) == 8
    assert all(
        item["screening_allocation"] == "scheduled_for_ma1" for item in trace
    )
    assert all(
        item["direction_alignment_status"]
        == "not_established_at_discovery"
        for item in trace
    )
    assert all(
        item["direction_evaluation_status"] == "not_assessed_search_stage"
        and item["execution_disposition"] == "search_only_not_executable"
        for item in trace
    )
    assert packet.metadata["ma1_screening_candidate_budget"] == 8
    assert packet.metadata["ma1_scheduled_target_count"] == 8
    assert packet.metadata["ma1_screened_target_count"] == 8
    assert packet.metadata["ma1_search_candidate_count"] == len(trace)
    assert packet.metadata["ma1_screened_candidate_count"] == len(screened)
    assert packet.metadata["ma1_executable_hypothesis_count"] == len(
        packet.hypotheses
    )
    assert packet.metadata["ma1_executable_admission_policy"] == (
        "critic_acceptance_requires_linked_direct_target_direction_primary_"
        "and_primary_target_disease_association_v1"
    )
    assert packet.metadata["ma1_pool_contract"] == {
        "schema_version": "search-screen-executable-pools-v1",
        "search_pool_role": "search_only_not_executable",
        "search_stage_direction_semantics": "not_assessed_search_stage",
        "screening_direction_triage_policy": (
            "persisted_claim_linkage_only_no_biological_inference_v1"
        ),
        "executable_pool_role": "retained_hypotheses_only",
        "search_candidate_count": len(trace),
        "screened_candidate_count": len(screened),
        "executable_hypothesis_count": len(packet.hypotheses),
    }
    assert packet.metadata["target_fallback_used"] is False
    assert all(
        item["target_source"] == "external_dynamic"
        for item in packet.metadata["target_sources"].values()
    )


def test_odd_screening_budget_never_exceeds_global_candidate_cap(tmp_path) -> None:
    symbols = ["TGT1", "TGT2", "TGT3", "TGT4", "TGT5", "TGT6"]
    candidates = tuple(
        _candidate(
            rank=index,
            symbol=symbol,
            ensembl_id=f"ENSG{index:011d}",
            track="tbio",
            score=0.95 - index / 100,
        )
        for index, symbol in enumerate(symbols, start=1)
    )
    discovery = FakeTargetDiscovery(result=_result(candidates))
    brief = _brief().model_copy(
        update={"constraints": ProjectConstraints(max_ma1_iterations=1)},
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(
            tmp_path,
            max_ma1_iterations=1,
            dynamic_target_max_candidates=5,
            target_identity_resolution_enabled=False,
        ),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
    ).run(brief)

    assert discovery.arguments == (
        "microsatellite-stable colorectal cancer",
        "EFO_0021782",
        5,
    )
    assert packet.metadata["ma1_screening_candidate_budget"] == 5
    assert packet.metadata["external_target_seed_count"] == 5
    assert packet.metadata["ma1_scheduled_target_count"] == 5
    assert packet.metadata["ma1_screened_target_count"] == 5
    assert {
        item["target_symbol"] for item in packet.metadata["screened_target_sources"]
    } == set(symbols[:5])


def test_fixture_fallback_uses_effective_intake_iteration_cap(tmp_path) -> None:
    brief = _brief().model_copy(
        update={"constraints": ProjectConstraints(max_ma1_iterations=3)},
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(
            tmp_path,
            max_ma1_iterations=1,
            dynamic_target_discovery_enabled=False,
            target_identity_resolution_enabled=False,
        ),
        evidence_retriever=MockEvidenceRetriever(),
    ).run(brief)

    assert packet.iterations == 1
    assert packet.metadata["fallback_target_seed_count"] == 2
    assert packet.metadata["ma1_scheduled_target_count"] == 2
    assert packet.metadata["ma1_screened_target_count"] == 2
    assert len(packet.hypotheses) == 2


def test_discovery_failure_calls_once_and_uses_only_labelled_fallback(tmp_path) -> None:
    discovery = FakeTargetDiscovery(error=RuntimeError("Open Targets unavailable"))
    packet = MA1Agent(
        settings=_settings(tmp_path, target_identity_resolution_enabled=False),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
    ).run(_brief())

    assert discovery.calls == 1
    assert packet.metadata["target_discovery_error"] == "RuntimeError"
    assert packet.metadata["resolved_disease"] is None
    assert packet.metadata["target_source_statuses"] == []
    assert packet.metadata["target_fallback_used"] is True
    assert packet.hypotheses
    assert all(
        item.target.metadata["target_discovery_source"] == "fixture_fallback"
        for item in packet.hypotheses
    )


@pytest.mark.parametrize(
    ("symbol", "ensembl_id", "uniprot_id"),
    (
        ("FLT3", "ENSG00000122025", "P36888"),
        ("ALK", "ENSG00000171094", "Q9UM73"),
        ("RET", "ENSG00000165731", "P07949"),
    ),
)
def test_explicit_scope_gets_first_proposal_opportunity_from_exact_external_identity(
    tmp_path,
    symbol: str,
    ensembl_id: str,
    uniprot_id: str,
) -> None:
    distractors = (
        _candidate(
            rank=1,
            symbol="ESR1",
            ensembl_id="ENSG00000091831",
            track="literature",
            score=0.96,
        ),
        _candidate(
            rank=2,
            symbol="PGR",
            ensembl_id="ENSG00000082175",
            track="literature",
            score=0.93,
        ),
        _candidate(
            rank=3,
            symbol="CDK4",
            ensembl_id="ENSG00000135446",
            track="tbio",
            score=0.90,
        ),
        _candidate(
            rank=4,
            symbol="AURKA",
            ensembl_id="ENSG00000087586",
            track="tbio",
            score=0.87,
        ),
        _candidate(
            rank=5,
            symbol="KRAS",
            ensembl_id="ENSG00000133703",
            track="tbio",
            score=0.84,
        ),
    )
    scoped = _candidate(
        rank=6,
        symbol=symbol,
        ensembl_id=ensembl_id,
        track="literature",
        score=0.70,
        include_pubmed=True,
    )
    discovery = FakeTargetDiscovery(result=_result((*distractors, scoped)))
    identity = FakeIdentityResolver(
        {symbol: (symbol, ensembl_id, uniprot_id)}
    )
    brief = _brief().model_copy(
        update={
            "brief_id": f"brief_explicit_{symbol.casefold()}",
            "research_goal": (
                f"Use executable target {symbol} only and require auditable "
                "candidate evidence."
            ),
        },
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
        identity_resolver=identity,
    ).run(brief)

    assert discovery.calls == 1
    assert identity.calls == [(symbol, ensembl_id)]
    assert {item.target.symbol for item in packet.hypotheses} == {symbol}
    assert packet.hypotheses[0].target.uniprot_id == uniprot_id
    assert packet.hypotheses[0].target.metadata["discovery_rank"] == 6
    assert {
        item["target_symbol"] for item in packet.metadata["screened_target_sources"]
    } == {symbol}
    assert packet.metadata["ma1_executable_target_scope_applied"] is True
    assert packet.metadata["ma1_executable_target_scope_symbols"] == [symbol]
    assert packet.metadata["ma1_scope_candidate_pool_count"] == 6
    assert packet.metadata["ma1_scope_candidate_match_count"] == 1
    assert packet.metadata["ma1_scope_identity_match_count"] == 1
    trace = packet.metadata["ma1_discovery_candidate_trace"]
    assert len(trace) == 6
    assert [item["target_symbol"] for item in trace if item["scope_disposition"] == "scope_match"] == [symbol]
    assert sum(item["scope_disposition"] == "outside_explicit_scope" for item in trace) == 5
    assert all(item["scientific_rejection"] is False for item in trace)
    assert packet.metadata["ma1_scope_evidence_missing_fail_closed"] is False
    assert packet.metadata["external_target_seed_count"] == 1
    assert packet.metadata["fallback_target_seed_count"] == 0
    assert packet.metadata["target_fallback_used"] is False


def test_explicit_scope_without_exact_external_identity_never_injects_fallback(
    tmp_path,
) -> None:
    discovery = FakeTargetDiscovery(
        result=_result(
            (
                _candidate(
                    rank=1,
                    symbol="FLT3",
                    ensembl_id="ENSG00000122025",
                    track="literature",
                    score=0.91,
                ),
                _candidate(
                    rank=2,
                    symbol="ALK",
                    ensembl_id="ENSG00000171094",
                    track="tbio",
                    score=0.86,
                ),
            )
        )
    )
    identity = FakeIdentityResolver({})
    brief = _brief().model_copy(
        update={
            "brief_id": "brief_explicit_ret_absent",
            "research_goal": (
                "Use executable target RET only and require auditable candidate evidence."
            ),
        },
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
        identity_resolver=identity,
    ).run(brief)

    assert identity.calls == [("RET", None)]
    assert packet.hypotheses == []
    assert packet.excluded == []
    assert packet.metadata["ma1_scope_candidate_pool_count"] == 2
    assert packet.metadata["ma1_scope_candidate_match_count"] == 0
    assert packet.metadata["ma1_scope_identity_match_count"] == 0
    assert packet.metadata["ma1_request_scope_identity_lane_used"] is True
    assert packet.metadata["ma1_request_scope_identity_query_count"] == 1
    assert packet.metadata["ma1_request_scope_identity_seed_count"] == 0
    identity_trace = packet.metadata["ma1_request_scope_identity_trace"]
    assert len(identity_trace) == 1
    assert identity_trace[0]["requested_symbol"] == "RET"
    assert identity_trace[0]["origin"] == "explicit_request_scope_identity"
    assert identity_trace[0]["query_attempted"] is True
    assert identity_trace[0]["disposition"] == "rejected"
    assert identity_trace[0]["reason_code"] == (
        "EXPLICIT_TARGET_IDENTITY_NOT_RESOLVED"
    )
    assert identity_trace[0]["scientific_support_created"] is False
    assert {
        item["target_symbol"]: item["reason_codes"]
        for item in packet.metadata["ma1_discovery_candidate_trace"]
    } == {
        "FLT3": ["OUTSIDE_EXECUTABLE_TARGET_SCOPE"],
        "ALK": ["OUTSIDE_EXECUTABLE_TARGET_SCOPE"],
    }
    assert packet.metadata["ma1_scope_evidence_missing_fail_closed"] is True
    assert packet.metadata["external_target_seed_count"] == 0
    assert packet.metadata["fallback_target_seed_count"] == 0
    assert packet.metadata["target_fallback_used"] is False
    assert packet.metadata["screened_target_source_count_total"] == 0


def test_explicit_scope_exact_identity_gets_screened_without_invented_evidence(
    tmp_path,
) -> None:
    discovery = FakeTargetDiscovery(
        result=_result(
            (
                _candidate(
                    rank=1,
                    symbol="FLT3",
                    ensembl_id="ENSG00000122025",
                    track="literature",
                    score=0.91,
                ),
            )
        )
    )
    identity = FakeIdentityResolver(
        {"RET": ("RET", "ENSG00000165731", "P07949")}
    )
    brief = _brief().model_copy(
        update={
            "brief_id": "brief_explicit_ret_identity_only",
            "research_goal": (
                "Use executable target RET only and require auditable "
                "candidate evidence."
            ),
        },
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=discovery,
        identity_resolver=identity,
    ).run(brief)

    assert identity.calls == [("RET", None)]
    assert packet.hypotheses == []
    assert len(packet.excluded) == 1
    assert packet.excluded[0].target_symbol == "RET"
    assert "MISSING_EVIDENCE" in packet.excluded[0].reason_codes
    assert "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY" in (
        packet.excluded[0].reason_codes
    )
    assert "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION" in (
        packet.excluded[0].reason_codes
    )
    assert packet.metadata["ma1_scope_candidate_match_count"] == 0
    assert packet.metadata["ma1_scope_identity_match_count"] == 1
    assert packet.metadata["ma1_request_scope_identity_seed_count"] == 1
    assert packet.metadata["target_fallback_used"] is False
    identity_trace = packet.metadata["ma1_request_scope_identity_trace"]
    assert len(identity_trace) == 1
    assert identity_trace[0]["disposition"] == "resolved"
    assert identity_trace[0]["scheduling_disposition"] == "scheduled_for_ma1"
    assert identity_trace[0]["scientific_support_created"] is False
    screened = packet.metadata["screened_target_sources"]
    assert len(screened) == 1
    assert screened[0]["target_symbol"] == "RET"
    assert screened[0]["target_source"] == (
        "explicit_request_scope_identity"
    )
    ledger = packet.metadata["ma1_explicit_scope_conservation_ledger"]
    assert ledger["conservation_valid"] is True
    assert ledger["collections"]["requested"]["symbols"] == ["RET"]
    assert ledger["collections"]["identity_resolved"]["symbols"] == [
        "RET"
    ]
    assert ledger["collections"]["screened"]["symbols"] == ["RET"]
    assert ledger["collections"]["retained"]["symbols"] == []
    assert ledger["collections"]["excluded"]["symbols"] == ["RET"]


@pytest.mark.parametrize(
    ("identity_kind", "expected_reason"),
    [
        ("partial", "EXPLICIT_TARGET_IDENTITY_NOT_RESOLVED"),
        ("fixture", "EXPLICIT_TARGET_FIXTURE_IDENTITY_REJECTED"),
        ("mismatch", "EXPLICIT_TARGET_CANONICAL_SYMBOL_MISMATCH"),
    ],
)
def test_explicit_request_identity_lane_rejects_non_exact_or_fixture_identity(
    tmp_path, identity_kind, expected_reason,
) -> None:
    discovery = FakeTargetDiscovery(result=_result(()))
    identity = StaticIdentityResolver(
        _static_ret_identity_resolution(identity_kind), []
    )
    brief = _brief().model_copy(
        update={
            "brief_id": f"brief_explicit_ret_{identity_kind}",
            "research_goal": (
                "Use executable target RET only and require auditable "
                "candidate evidence."
            ),
        },
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=discovery,
        identity_resolver=identity,
    ).run(brief)

    assert identity.calls == [("RET", None)]
    assert packet.hypotheses == []
    assert packet.excluded == []
    assert packet.metadata["ma1_scope_identity_match_count"] == 0
    assert packet.metadata["ma1_request_scope_identity_seed_count"] == 0
    trace = packet.metadata["ma1_request_scope_identity_trace"]
    assert len(trace) == 1
    assert trace[0]["reason_code"] == expected_reason
    assert trace[0]["seed_created"] is False
    assert trace[0]["scientific_support_created"] is False


def test_explicit_forbidden_target_is_rejected_before_identity_query(
    tmp_path,
) -> None:
    identity = FakeIdentityResolver(
        {"RET": ("RET", "ENSG00000165731", "P07949")}
    )
    base = _brief()
    brief = base.model_copy(
        update={
            "brief_id": "brief_explicit_ret_forbidden",
            "research_goal": (
                "Use executable target RET only and require auditable "
                "candidate evidence."
            ),
            "constraints": base.constraints.model_copy(
                update={"forbidden_targets": ["RET"]}, deep=True
            ),
        },
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=FakeTargetDiscovery(result=_result(())),
        identity_resolver=identity,
    ).run(brief)

    assert identity.calls == []
    assert packet.hypotheses == []
    assert packet.excluded == []
    trace = packet.metadata["ma1_request_scope_identity_trace"]
    assert len(trace) == 1
    assert trace[0]["query_attempted"] is False
    assert trace[0]["reason_code"] == "EXPLICIT_TARGET_FORBIDDEN"


def test_explicit_identity_is_queried_when_disease_resolution_failed_without_creating_support(
    tmp_path,
) -> None:
    identity = FakeIdentityResolver(
        {"RET": ("RET", "ENSG00000165731", "P07949")}
    )
    brief = _brief().model_copy(
        update={
            "brief_id": "brief_explicit_ret_unresolved_disease",
            "research_goal": (
                "Use executable target RET only and require auditable "
                "candidate evidence."
            ),
        },
        deep=True,
    )

    packet = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=FakeTargetDiscovery(error=RuntimeError("offline")),
        identity_resolver=identity,
    ).run(brief)

    assert identity.calls == [("RET", None)]
    assert packet.metadata["disease_resolution_status"] == (
        "source_unavailable"
    )
    assert packet.metadata["ma1_scope_identity_match_count"] == 1
    trace = packet.metadata["ma1_request_scope_identity_trace"]
    assert len(trace) == 1
    assert trace[0]["query_attempted"] is True
    assert trace[0]["identity_status"] == "resolved"
    assert trace[0]["canonical_symbol"] == "RET"
    assert trace[0]["ensembl_id"] == "ENSG00000165731"
    assert trace[0]["uniprot_id"] == "P07949"
    assert trace[0]["scientific_support_created"] is False
    assert trace[0]["reason_code"] is None
    assert packet.metadata["target_fallback_used"] is False


def test_multiple_explicit_identity_queries_preserve_order_and_create_no_support(
    tmp_path,
) -> None:
    identity = FakeIdentityResolver(
        {
            "FLT3": ("FLT3", "ENSG00000122025", "P36888"),
            "BCL2": ("BCL2", "ENSG00000171791", "P10415"),
        }
    )
    retriever = RecordingEmptyEvidenceRetriever()
    brief = _brief().model_copy(
        update={
            "brief_id": "brief_explicit_flt3_bcl2_order",
            "research_goal": (
                "Restrict executable targets to FLT3 and BCL2 for this "
                "audited run."
            ),
        },
        deep=True,
    )

    agent = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=retriever,
        target_discovery=FakeTargetDiscovery(result=_result(())),
        identity_resolver=identity,
    )
    packet = agent.run(brief)

    assert identity.calls == [("FLT3", None), ("BCL2", None)]
    assert [item.symbol for item in retriever.seeds] == ["FLT3", "BCL2"]
    assert all(item.discovery_evidence == () for item in retriever.seeds)
    assert all(item.priority == 0.0 for item in retriever.seeds)
    assert all(
        item.discovery_source == "explicit_request_scope_identity"
        for item in retriever.seeds
    )
    assert [item.track for item in retriever.seeds] == [
        HypothesisTrack.LITERATURE,
        HypothesisTrack.TBIO,
    ]
    rebuilt = agent.proposer._build_hypothesis(brief, retriever.seeds[0], 1)
    assert rebuilt.target.tdl is None
    assert rebuilt.target.metadata["association_score"] is None
    assert [item["requested_symbol"] for item in packet.metadata[
        "ma1_request_scope_identity_trace"
    ]] == ["FLT3", "BCL2"]
    assert {item.target_symbol for item in packet.excluded} == {
        "FLT3",
        "BCL2",
    }
    ledger = packet.metadata["ma1_explicit_scope_conservation_ledger"]
    assert ledger["collections"]["requested"]["symbols"] == [
        "FLT3",
        "BCL2",
    ]
    assert ledger["collections"]["screened"]["symbols"] == [
        "FLT3",
        "BCL2",
    ]
    assert ledger["unexpected_generated_symbols"] == []


def test_rework_schedules_fresh_ranked_targets_before_identical_retries(tmp_path) -> None:
    symbols = ["FLT3", "IDH1", "DNMT3A", "CEBPA", "TET2", "SRSF2", "RUNX1", "KIT"]
    candidates = tuple(
        _candidate(
            rank=index,
            symbol=symbol,
            ensembl_id=f"ENSG{index:011d}",
            track="literature" if index % 2 else "tbio",
            score=0.95 - index / 100,
            include_pubmed=index % 2 == 1,
        )
        for index, symbol in enumerate(symbols, start=1)
    )
    discovery = FakeTargetDiscovery(result=_result(candidates))
    prior_symbols = symbols[:4]
    prior = [
        ExcludedHypothesis(
            target_symbol=symbol,
            reason_codes=["UNSUPPORTED_CLAIM"],
            source_iteration=1,
        )
        for symbol in prior_symbols
    ]

    packet = MA1Agent(
        settings=_settings(tmp_path),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
    ).run(_brief(), revision_context={"excluded": prior})

    screened = {
        item["target_symbol"] for item in packet.metadata["screened_target_sources"]
    }
    assert {"TET2", "SRSF2", "RUNX1", "KIT"} <= screened
    trace = {
        item["target_symbol"]: item
        for item in packet.metadata["ma1_discovery_candidate_trace"]
    }
    assert all(
        trace[symbol]["screening_allocation"] == "scheduled_for_ma1"
        for symbol in ("TET2", "SRSF2", "RUNX1", "KIT")
    )
    exclusion_provenance = {
        item["target_symbol"]: item
        for item in packet.metadata["ma1_exclusion_provenance"]
    }
    for symbol in prior_symbols:
        if symbol not in screened:
            assert trace[symbol]["screening_allocation"] == (
                "not_scheduled_evidence_retry_budget"
            )
            assert trace[symbol]["reason_codes"] == [
                "MA1_SCREENING_BUDGET_REALLOCATED_TO_EVIDENCE_RETRY"
            ]
            assert (
                exclusion_provenance[symbol]["record_origin"]
                == "prior_packet_preserved_not_rescreened"
            )
            assert (
                exclusion_provenance[symbol]["scientific_rejection_replayed"]
                is False
            )


def test_rework_expansion_screens_fresh_eight_before_restored_retries(
    tmp_path,
) -> None:
    prior_symbols = [f"OLD{index}" for index in range(1, 9)]
    fresh_symbols = [f"NEW{index}" for index in range(1, 9)]
    all_symbols = [*prior_symbols, *fresh_symbols]
    candidates = tuple(
        _candidate(
            rank=index,
            symbol=symbol,
            ensembl_id=f"ENSG{index:011d}",
            track="literature" if index % 2 else "tbio",
            score=0.99 - index / 100,
        )
        for index, symbol in enumerate(all_symbols, start=1)
    )
    discovery = WindowedTargetDiscovery(
        candidates=candidates,
        requested_limits=[],
    )
    agent = MA1Agent(
        settings=_settings(
            tmp_path,
            target_identity_resolution_enabled=False,
        ),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=discovery,
    )

    first = agent.run(_brief())
    second = agent.run(_brief(), revision_context=first)

    first_screened = {
        item["target_symbol"]
        for item in first.metadata["screened_target_sources"]
    }
    second_screened = {
        item["target_symbol"]
        for item in second.metadata["screened_target_sources"]
    }
    cumulative_screened = {
        item["target_symbol"]
        for item in second.metadata["ma1_screening_history_sources"]
    }

    assert discovery.requested_limits == [8, 16]
    assert first_screened == set(prior_symbols)
    retry_screened = second_screened & first_screened
    fresh_screened = second_screened - first_screened
    assert fresh_screened == set(fresh_symbols)
    assert len(retry_screened) == 1
    assert cumulative_screened == set(all_symbols)
    assert second.metadata["ma1_screening_history_source_count_total"] == 17
    assert second.metadata["ma1_screening_candidate_budget"] == 8
    assert second.metadata["ma1_total_screening_candidate_budget"] == 9
    assert second.metadata["ma1_scheduled_target_count"] == 9
    assert second.metadata["ma1_screened_target_count"] == 9
    normalized_prior_symbols = {symbol.casefold() for symbol in prior_symbols}
    assert set(second.metadata["prior_evidence_retry_restored_symbols"]) == (
        normalized_prior_symbols
    )
    assert set(
        second.metadata["prior_evidence_retry_catalog_seed_restored_symbols"]
    ) == normalized_prior_symbols
    assert second.metadata[
        "prior_evidence_retry_restored_symbols_semantics"
    ].endswith("restoration does not mean scheduled or attempted")
    assert set(
        second.metadata["prior_evidence_retry_budget_unscheduled_symbols"]
    ) == normalized_prior_symbols - {
        symbol.casefold() for symbol in retry_screened
    }
    assert second.metadata["prior_evidence_retry_execution_summary"] == {
        "semantics": (
            "catalog seed restoration only reconstructs a bounded target identity; "
            "evidence retrieval occurs only for attempted symbols; supplementary "
            "retry slots never replace fresh-target slots"
        ),
        "selection_policy": (
            "bounded_supplementary_least_prior_attempts_then_available_"
            "catalog_priority_then_stable_symbol_v2"
        ),
        "configured_supplementary_budget": 1,
        "effective_supplementary_budget": 1,
        "requested_count": 8,
        "catalog_available_count": 8,
        "catalog_seed_restored_count": 8,
        "catalog_missing_count": 0,
        "scheduled_count": 1,
        "attempted_count": 1,
        "budget_not_scheduled_count": 7,
        "scheduled_not_attempted_count": 0,
    }
    retry_states = second.metadata["prior_evidence_retry_state_records"]
    assert all(item["prior_retry_attempt_count"] == 0 for item in retry_states)
    assert all(
        item["catalog_seed_status"] == "restored_from_screening_history"
        for item in retry_states
    )
    assert sum(item["execution_status"] == "attempted" for item in retry_states) == 1
    assert sum(
        item["execution_status"] == "budget_not_scheduled"
        for item in retry_states
    ) == 7


def test_rework_retry_metadata_distinguishes_restored_seed_from_execution(
    tmp_path,
) -> None:
    first_symbols = [f"OLD{index}" for index in range(1, 9)]
    second_symbols = [f"MID{index}" for index in range(1, 9)]
    third_symbols = [f"NEW{index}" for index in range(1, 9)]
    all_symbols = [*first_symbols, *second_symbols, *third_symbols]
    candidates = tuple(
        _candidate(
            rank=index,
            symbol=symbol,
            ensembl_id=f"ENSG{index:011d}",
            track="literature" if index % 2 else "tbio",
            score=0.99 - index / 100,
        )
        for index, symbol in enumerate(all_symbols, start=1)
    )
    discovery = WindowedTargetDiscovery(
        candidates=candidates,
        requested_limits=[],
    )
    agent = MA1Agent(
        settings=_settings(
            tmp_path,
            target_identity_resolution_enabled=False,
        ),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=discovery,
    )

    first = agent.run(_brief())
    second = agent.run(_brief(), revision_context=first)
    third = agent.run(_brief(), revision_context=second)

    summary = third.metadata["prior_evidence_retry_execution_summary"]
    assert discovery.requested_limits == [8, 16, 24]
    assert summary["requested_count"] == 16
    assert summary["catalog_seed_restored_count"] == 16
    assert summary["scheduled_count"] == 1
    assert summary["attempted_count"] == 1
    assert summary["budget_not_scheduled_count"] == 15
    assert {
        item["target_symbol"]
        for item in third.metadata["prior_evidence_retry_state_records"]
    } == {symbol.casefold() for symbol in [*first_symbols, *second_symbols]}
    assert {
        item["catalog_seed_status"]
        for item in third.metadata["prior_evidence_retry_state_records"]
    } == {"restored_from_screening_history"}
    assert {
        item["execution_status"]
        for item in third.metadata["prior_evidence_retry_state_records"]
    } == {"attempted", "budget_not_scheduled"}
    second_retry_symbols = {
        item["target_symbol"]
        for item in second.metadata["screened_target_sources"]
    } - set(second_symbols)
    third_screened_symbols = {
        item["target_symbol"]
        for item in third.metadata["screened_target_sources"]
    }
    third_retry_symbols = third_screened_symbols - set(third_symbols)
    assert len(second_retry_symbols) == 1
    assert len(third_retry_symbols) == 1
    assert third_retry_symbols <= set([*first_symbols, *second_symbols])
    assert second_retry_symbols.isdisjoint(third_retry_symbols)
    assert set(third_symbols) <= third_screened_symbols


def test_operator_can_disable_supplementary_retry_without_reducing_fresh_window(
    tmp_path,
) -> None:
    prior_symbols = [f"OLD{index}" for index in range(1, 9)]
    fresh_symbols = [f"NEW{index}" for index in range(1, 9)]
    candidates = tuple(
        _candidate(
            rank=index,
            symbol=symbol,
            ensembl_id=f"ENSG{index:011d}",
            track="literature" if index % 2 else "tbio",
            score=0.99 - index / 100,
        )
        for index, symbol in enumerate(
            [*prior_symbols, *fresh_symbols], start=1
        )
    )
    agent = MA1Agent(
        settings=_settings(
            tmp_path,
            target_identity_resolution_enabled=False,
            ma1_supplementary_evidence_retry_budget=0,
        ),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=WindowedTargetDiscovery(
            candidates=candidates,
            requested_limits=[],
        ),
    )

    first = agent.run(_brief())
    second = agent.run(_brief(), revision_context=first)

    assert {
        item["target_symbol"]
        for item in second.metadata["screened_target_sources"]
    } == set(fresh_symbols)
    assert second.metadata[
        "ma1_supplementary_evidence_retry_budget_configured"
    ] == 0
    assert second.metadata[
        "ma1_supplementary_evidence_retry_budget_effective"
    ] == 0
    assert second.metadata["ma1_screening_candidate_budget"] == 8
    assert second.metadata["ma1_total_screening_candidate_budget"] == 8
    assert second.metadata["ma1_scheduled_target_count"] == 8
    assert second.metadata["prior_evidence_retry_execution_summary"][
        "attempted_count"
    ] == 0


def test_retry_aging_survives_five_reworks_and_screening_history_truncation(
    tmp_path,
) -> None:
    symbols = [f"TGT{index:02d}" for index in range(1, 25)]
    candidates = tuple(
        _candidate(
            rank=index,
            symbol=symbol,
            ensembl_id=f"ENSG{index:011d}",
            track="literature" if index % 2 else "tbio",
            score=0.99 - index / 100,
        )
        for index, symbol in enumerate(symbols, start=1)
    )
    agent = MA1Agent(
        settings=_settings(
            tmp_path,
            target_identity_resolution_enabled=False,
        ),
        evidence_retriever=EmptyEvidenceRetriever(),
        target_discovery=WindowedTargetDiscovery(
            candidates=candidates,
            requested_limits=[],
        ),
    )
    brief = _brief().model_copy(
        update={
            "constraints": ProjectConstraints(
                max_ma1_iterations=2,
                max_rework_loops=5,
            )
        },
        deep=True,
    )

    packet = agent.run(brief)
    retry_sequence: list[str] = []
    for _ in range(5):
        packet = agent.run(brief, revision_context=packet)
        attempted = packet.metadata["prior_claim_evidence_retry_symbols"]
        assert len(attempted) == 1
        retry_sequence.extend(attempted)

    assert len(set(retry_sequence)) == 5
    assert packet.metadata["ma1_screening_history_source_count_total"] > 24
    assert packet.metadata[
        "ma1_screening_history_source_count_recorded"
    ] == 24
    ledger = packet.metadata["ma1_evidence_retry_attempt_ledger"]
    assert ledger["authoritative"] is True
    assert ledger["carry_forward_source"] == "carried_forward_authoritative"
    assert ledger["legacy_history_bootstrap_used"] is False
    assert ledger["entry_count_omitted"] == 0
    assert ledger["current_revision_incremented_symbols"] == [
        retry_sequence[-1]
    ]
    ledger_counts = {
        item["target_symbol"]: item["attempt_count"]
        for item in ledger["entries"]
    }
    assert {symbol: ledger_counts[symbol] for symbol in retry_sequence} == {
        symbol: 1 for symbol in retry_sequence
    }


def test_max_supplementary_policy_can_rotate_twenty_unique_retries() -> None:
    seeds = tuple(
        TargetSeed(
            symbol=f"TGT{index:02d}",
            target_name=f"target {index}",
            track=(
                HypothesisTrack.LITERATURE
                if index % 2
                else HypothesisTrack.TBIO
            ),
            priority=1.0 - index / 100,
            discovery_source="external_dynamic",
        )
        for index in range(1, 21)
    )
    catalog = {
        track: tuple(seed for seed in seeds if seed.track is track)
        for track in HypothesisTrack
    }
    retry_symbols = {seed.symbol.casefold() for seed in seeds}
    attempt_counts: dict[str, int] = {}
    attempted_across_reworks: set[str] = set()
    for _ in range(5):
        selected = _select_supplementary_evidence_retry_catalog(
            catalog,
            retry_symbols,
            prior_attempt_counts=attempt_counts,
            retry_limit=4,
        )
        attempted = {
            seed.symbol.casefold()
            for track_seeds in selected.values()
            for seed in track_seeds
        }
        assert len(attempted) == 4
        assert attempted.isdisjoint(attempted_across_reworks)
        attempted_across_reworks.update(attempted)
        for symbol in attempted:
            attempt_counts[symbol] = attempt_counts.get(symbol, 0) + 1

    assert len(attempted_across_reworks) == 20
    assert set(attempt_counts.values()) == {1}


def test_explicit_scope_rejects_stale_non_scope_prior_before_selector(tmp_path) -> None:
    candidates = (
        _candidate(
            rank=1,
            symbol="ALK",
            ensembl_id="ENSG00000171094",
            track="literature",
            score=0.91,
        ),
        _candidate(
            rank=2,
            symbol="RET",
            ensembl_id="ENSG00000165731",
            track="tbio",
            score=0.88,
        ),
    )
    discovery = FakeTargetDiscovery(result=_result(candidates))
    identity = FakeIdentityResolver(
        {
            "ALK": ("ALK", "ENSG00000171094", "Q9UM73"),
            "RET": ("RET", "ENSG00000165731", "P07949"),
        }
    )
    agent = MA1Agent(
        settings=_settings(
            tmp_path,
            dynamic_target_fallback_enabled=False,
            demo_fixture_fallback_enabled=False,
        ),
        evidence_retriever=MockEvidenceRetriever(),
        target_discovery=discovery,
        identity_resolver=identity,
    )
    prior = agent.run(_brief())
    assert {item.target.symbol for item in prior.hypotheses} == {"ALK", "RET"}
    brief = _brief().model_copy(
        update={
            "brief_id": "brief_explicit_ret_rework",
            "research_goal": "Use executable target RET only throughout rework.",
        },
        deep=True,
    )

    packet = agent.run(brief, revision_context=prior)

    assert {item.target.symbol for item in packet.hypotheses} == {"RET"}
    assert any(
        item.target_symbol == "ALK"
        and item.reason_codes == ["OUTSIDE_EXECUTABLE_TARGET_SCOPE"]
        for item in packet.excluded
    )
    assert packet.metadata["ma1_scope_prior_exclusion_count"] == 1
    assert packet.metadata["ma1_scope_selector_exclusion_count"] == 0
    selector_ranked = agent.selector.select(
        prior.hypotheses,
        allowed_target_keys=frozenset({"ret"}),
    )
    assert {item.target.symbol for item in selector_ranked} == {"RET"}
