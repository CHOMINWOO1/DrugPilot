from __future__ import annotations

from typing import Any

from drugpilot.domain.models import (
    Claim,
    EvidenceRef,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    TargetCandidate,
)
from drugpilot.tools.context_router import (
    ContextRoutingBudget,
    ContextRoutingRequest,
    ContextTarget,
    DiseaseTargetContextRouter,
    context_request_from_hypothesis_packet,
)
from drugpilot.tools.public_context import (
    CLINICAL_TRIALS_SEARCH_URL,
    EUROPE_PMC_SEARCH_URL,
    GTEX_GENE_SEARCH_URL,
    KEGG_HUMAN_GENE_FIND_URL,
    MA1ContextResult,
    MA1PublicContextRetriever,
    QUICKGO_ANNOTATION_URL,
    REACTOME_SEARCH_URL,
    STRING_NETWORK_URL,
    ContextSourceResult,
)
from drugpilot.tools.public_expression_interaction import (
    ARCHS4_CORRELATION_URL,
    BIOGRID_INTERACTIONS_URL,
)
from drugpilot.tools.public_cell_data import (
    CELLXGENE_SIDECAR_PATH,
    HCA_PROJECTS_URL,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeResponse:
    def __init__(self, payload: Any = None, *, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakeContextClient:
    """HTTP fake that exercises the real MA1 public-context adapters."""

    def __init__(self, *, clock: FakeClock | None = None, seconds_per_call: float = 0) -> None:
        self.clock = clock
        self.seconds_per_call = seconds_per_call
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self.clock is not None:
            self.clock.advance(self.seconds_per_call)
        if url == EUROPE_PMC_SEARCH_URL:
            query = kwargs["params"]["query"]
            server = "bioRxiv" if 'PUBLISHER:"bioRxiv"' in query else "medRxiv"
            return FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": f"10.1101/{server.lower()}.001",
                                "title": f"{server} routed context",
                            }
                        ]
                    }
                }
            )
        if url == HCA_PROJECTS_URL:
            return FakeResponse(
                {
                    "hits": [
                        {
                            "entryId": "hca-project-1",
                            "projects": [
                                {
                                    "projectId": "hca-project-1",
                                    "projectTitle": "Cell-context project",
                                }
                            ],
                        }
                    ]
                }
            )
        if url.endswith(CELLXGENE_SIDECAR_PATH):
            return FakeResponse(
                {
                    "schema_version": "1.0",
                    "census_version": "test-census",
                    "records": [
                        {
                            "dataset_id": "cxg-dataset-1",
                            "dataset_title": "Cell-context dataset",
                            "matched_cell_count": 42,
                            "gene_symbol": kwargs["params"].get("gene_symbol"),
                            "disease": [kwargs["params"].get("disease")],
                            "cell_type": [kwargs["params"].get("cell_type")],
                        }
                    ],
                }
            )
        if url == CLINICAL_TRIALS_SEARCH_URL:
            return FakeResponse(
                {
                    "studies": [
                        {
                            "protocolSection": {
                                "identificationModule": {
                                    "nctId": "NCT00000001",
                                    "briefTitle": "Routed target trial",
                                },
                                "statusModule": {"overallStatus": "RECRUITING"},
                                "designModule": {"studyType": "INTERVENTIONAL"},
                                "conditionsModule": {"conditions": ["test disease"]},
                                "armsInterventionsModule": {"interventions": []},
                            }
                        }
                    ]
                }
            )
        if url == GTEX_GENE_SEARCH_URL:
            query = kwargs["params"]["geneId"]
            return FakeResponse(
                {
                    "data": [
                        {
                            "gencodeId": query,
                            "geneSymbol": "GPR132",
                            "geneName": "G protein-coupled receptor 132",
                        }
                    ]
                }
            )
        if url == STRING_NETWORK_URL:
            symbol = kwargs["params"]["identifiers"]
            return FakeResponse(
                [
                    {
                        "stringId_A": f"9606.{symbol}",
                        "stringId_B": "9606.PARTNER",
                        "preferredName_A": symbol,
                        "preferredName_B": "PARTNER",
                        "score": 0.91,
                    }
                ]
            )
        if url == BIOGRID_INTERACTIONS_URL:
            return FakeResponse(
                {
                    "1001": {
                        "BIOGRID_INTERACTION_ID": "1001",
                        "OFFICIAL_SYMBOL_A": kwargs["params"]["geneList"],
                        "OFFICIAL_SYMBOL_B": "PARTNER",
                        "EXPERIMENTAL_SYSTEM": "Two-hybrid",
                    }
                }
            )
        if url == REACTOME_SEARCH_URL:
            return FakeResponse(
                {
                    "results": [
                        {
                            "entries": [
                                {
                                    "stId": "R-HSA-0001",
                                    "name": "Routed pathway",
                                    "type": "Pathway",
                                }
                            ]
                        }
                    ]
                }
            )
        if url.startswith(f"{KEGG_HUMAN_GENE_FIND_URL}/"):
            return FakeResponse(text="hsa:0001\tGPR132; routed target\n")
        if url == QUICKGO_ANNOTATION_URL:
            return FakeResponse(
                {
                    "results": [
                        {
                            "geneProductId": kwargs["params"]["geneProductId"],
                            "symbol": "GPR132",
                            "goId": "GO:0004930",
                            "evidenceCode": "ECO:0000269",
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected URL: {url}")

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self.clock is not None:
            self.clock.advance(self.seconds_per_call)
        if url == ARCHS4_CORRELATION_URL:
            return FakeResponse(
                {
                    "gene": kwargs["json"]["gene"],
                    "searchterm": kwargs["json"]["meta"],
                    "mean_log_expression": 1.7,
                    "positive_correlated_genes": [],
                    "negative_correlated_genes": [],
                    "samples": ["GSM1"],
                }
            )
        raise AssertionError(f"unexpected POST URL: {url}")


def _request(*, max_calls: int = 10, max_seconds: float = 30) -> ContextRoutingRequest:
    return ContextRoutingRequest(
        disease="test disease",
        targets=[
            ContextTarget(
                symbol="MAPK1",
                track=HypothesisTrack.LITERATURE,
                rank=2,
                score=0.99,
            ),
            ContextTarget(
                symbol="TP53",
                track=HypothesisTrack.LITERATURE,
                rank=1,
                score=0.8,
            ),
            ContextTarget(
                symbol="SLC15A4",
                track=HypothesisTrack.TBIO,
                rank=2,
                score=0.99,
            ),
            ContextTarget(
                symbol="GPR132",
                track=HypothesisTrack.TBIO,
                rank=1,
                score=0.7,
                uniprot_id="Q9UNW8",
                gencode_id="ENSG00000183484.12",
            ),
        ],
        budget=ContextRoutingBudget(
            max_targets_per_track=1,
            max_source_calls=max_calls,
            max_elapsed_seconds=max_seconds,
            limit_per_source=1,
        ),
    )


def test_routes_only_top_targets_to_track_specific_sources_with_provenance() -> None:
    client = FakeContextClient()
    retriever = MA1PublicContextRetriever(
        client=client,
        biogrid_access_key="A" * 32,
    )
    result = DiseaseTargetContextRouter(retriever).route(_request())

    assert result.source_calls_used == 10
    assert result.planned_source_calls == 10
    assert result.budget_exhausted is False
    assert [item.target.symbol for item in result.targets] == ["TP53", "GPR132"]
    selection = {item.target.symbol: item.status for item in result.target_selection}
    assert selection == {
        "MAPK1": "outside_top_target_limit",
        "TP53": "selected",
        "SLC15A4": "outside_top_target_limit",
        "GPR132": "selected",
    }

    by_symbol = {item.target.symbol: item for item in result.targets}
    assert [item.source for item in by_symbol["TP53"].sources] == [
        "bioRxiv",
        "Human Cell Atlas",
        "CELLxGENE Census",
        "medRxiv",
        "ClinicalTrials.gov",
        "ARCHS4",
        "BioGRID",
    ]
    assert [item.source for item in by_symbol["GPR132"].sources] == [
        "GTEx",
        "ARCHS4",
        "STRING",
        "BioGRID",
        "Reactome",
        "KEGG",
        "QuickGO",
        "Human Cell Atlas",
        "CELLxGENE Census",
    ]
    queried = [item for bundle in result.targets for item in bundle.sources if item.queried]
    skipped = [item for bundle in result.targets for item in bundle.sources if not item.queried]
    assert all(item.status == "ok" for item in queried)
    assert all(item.status == "skipped_not_applicable" for item in skipped)
    assert all(record.supports == "context" for bundle in result.targets for record in bundle.evidence)
    assert all(
        provenance.context_only
        and not provenance.causal_inference_allowed
        and not provenance.counter_evidence_allowed
        for bundle in result.targets
        for provenance in bundle.provenance
    )

    serialized_calls = repr(client.calls)
    assert "MAPK1" not in serialized_calls
    assert "SLC15A4" not in serialized_calls
    assert "TP53" in serialized_calls
    assert "GPR132" in serialized_calls
    gtex_call = next(call for call in client.calls if call[0] == GTEX_GENE_SEARCH_URL)
    assert gtex_call[1]["params"]["geneId"] == "ENSG00000183484.12"


def test_effective_work_track_controls_source_plan_without_changing_hypothesis_track() -> None:
    client = FakeContextClient()
    request = ContextRoutingRequest(
        disease="test disease",
        targets=[
            ContextTarget(
                symbol="GPR132",
                track=HypothesisTrack.TBIO,
                effective_work_track=HypothesisTrack.LITERATURE,
                work_track_basis="effective_tdl",
                rank=1,
            )
        ],
        budget=ContextRoutingBudget(max_source_calls=8, limit_per_source=1),
    )

    result = DiseaseTargetContextRouter(
        MA1PublicContextRetriever(client=client)
    ).route(request)
    target = result.targets[0].target

    assert target.track is HypothesisTrack.TBIO
    assert target.effective_work_track is HypothesisTrack.LITERATURE
    assert [item.source for item in result.targets[0].sources] == [
        "bioRxiv",
        "Human Cell Atlas",
        "CELLxGENE Census",
        "medRxiv",
        "ClinicalTrials.gov",
        "ARCHS4",
        "BioGRID",
    ]
    assert all(
        item.source not in {"GTEx", "STRING", "Reactome", "KEGG", "QuickGO"}
        for item in result.targets[0].sources
    )


def test_comprehensive_budget_covers_every_applicable_source_for_two_targets_per_track() -> None:
    client = FakeContextClient()
    base_request = _request(max_calls=36, max_seconds=90)
    request = base_request.model_copy(
        update={
            "targets": [
                item.model_copy(
                    update={
                        "uniprot_id": item.uniprot_id or "Q9H000",
                        "gencode_id": item.gencode_id or "ENSG00000100000.1",
                    },
                    deep=True,
                )
                for item in base_request.targets
            ],
            "target_cell_type": "tumor cell",
            "human_priorities": ["single-cell expression and interaction network"],
            "budget": ContextRoutingBudget(
                max_targets_per_track=2,
                max_source_calls=36,
                max_elapsed_seconds=90,
                limit_per_source=5,
            ),
        },
        deep=True,
    )
    retriever = MA1PublicContextRetriever(
        client=client,
        cellxgene_sidecar_url="https://cell-sidecar.example.test",
        biogrid_access_key="A" * 32,
    )

    result = DiseaseTargetContextRouter(retriever).route(request)

    assert {item.target.symbol for item in result.targets} == {
        "MAPK1",
        "TP53",
        "SLC15A4",
        "GPR132",
    }
    # 7 Literature sources x 2 targets + 9 Tbio sources x 2 targets.
    assert result.planned_source_calls == 32
    assert result.source_calls_used == 32
    assert result.budget_exhausted is False
    assert result.stop_reason is None
    outcomes = [item for bundle in result.targets for item in bundle.sources]
    assert all(item.queried for item in outcomes)
    assert not [
        item
        for item in outcomes
        if item.status in {"skipped_call_budget", "skipped_time_budget"}
    ]
    assert all(item.record_count <= 5 for item in outcomes)
    assert all(
        record.metadata["context_only"] is True
        and record.metadata["causal_inference_allowed"] is False
        for bundle in result.targets
        for record in bundle.evidence
    )


def test_call_budget_stops_new_sources_and_preserves_balanced_track_order() -> None:
    client = FakeContextClient()
    result = DiseaseTargetContextRouter(
        MA1PublicContextRetriever(client=client)
    ).route(_request(max_calls=3))

    assert result.source_calls_used == 3
    assert len(client.calls) == 3
    assert result.budget_exhausted is True
    assert result.stop_reason == "call_budget"
    # Round-robin scheduling gives both tracks context before returning to the
    # literature track under a tight shared budget.
    assert [url for url, _ in client.calls] == [
        EUROPE_PMC_SEARCH_URL,
        GTEX_GENE_SEARCH_URL,
        EUROPE_PMC_SEARCH_URL,
    ]
    all_outcomes = [item for bundle in result.targets for item in bundle.sources]
    assert sum(item.queried for item in all_outcomes) == 3
    assert all(
        item.status in {"skipped_call_budget", "skipped_not_applicable"}
        for item in all_outcomes
        if not item.queried
    )
    assert any(item.status == "skipped_call_budget" for item in all_outcomes)


def test_time_budget_stops_after_in_flight_source_boundary() -> None:
    clock = FakeClock()
    client = FakeContextClient(clock=clock, seconds_per_call=0.6)
    retriever = MA1PublicContextRetriever(client=client, clock=clock)
    result = DiseaseTargetContextRouter(retriever, clock=clock).route(
        _request(max_seconds=0.5)
    )

    assert len(client.calls) == 1
    assert result.source_calls_used == 1
    assert result.elapsed_seconds == 0.6
    assert result.budget_exhausted is True
    assert result.stop_reason == "time_budget"
    assert sum(
        item.status == "skipped_time_budget"
        for bundle in result.targets
        for item in bundle.sources
    ) == 9


def test_missing_uniprot_skips_quickgo_without_spending_a_call() -> None:
    client = FakeContextClient()
    request = ContextRoutingRequest(
        disease="test disease",
        targets=[ContextTarget(symbol="GPR132", track=HypothesisTrack.TBIO, rank=1)],
        budget=ContextRoutingBudget(max_source_calls=8, limit_per_source=1),
    )
    result = DiseaseTargetContextRouter(
        MA1PublicContextRetriever(client=client)
    ).route(request)

    outcomes = {item.source: item for item in result.targets[0].sources}
    assert result.source_calls_used == 6
    assert outcomes["QuickGO"].status == "skipped_missing_identifier"
    assert outcomes["QuickGO"].queried is False
    assert all(url != QUICKGO_ANNOTATION_URL for url, _ in client.calls)


def test_cell_sources_are_admitted_only_for_explicit_brief_cell_context() -> None:
    client = FakeContextClient()
    retriever = MA1PublicContextRetriever(
        client=client,
        cellxgene_sidecar_url="https://cell-sidecar.example.test",
    )
    request = ContextRoutingRequest(
        disease="test disease",
        targets=[
            ContextTarget(
                symbol="TP53",
                track=HypothesisTrack.LITERATURE,
                rank=1,
            )
        ],
        target_cell_type="AML blast",
        human_priorities=["single-cell expression context"],
        budget=ContextRoutingBudget(max_source_calls=10, limit_per_source=1),
    )

    result = DiseaseTargetContextRouter(retriever).route(request)
    outcomes = {item.source: item for item in result.targets[0].sources}

    assert outcomes["Human Cell Atlas"].status == "ok"
    assert outcomes["CELLxGENE Census"].status == "ok"
    assert outcomes["ARCHS4"].status == "ok"
    assert outcomes["BioGRID"].status == "skipped_not_applicable"
    assert "cell-resolved or single-cell context" in outcomes["Human Cell Atlas"].detail
    cell_calls = [
        kwargs
        for url, kwargs in client.calls
        if url == HCA_PROJECTS_URL or url.endswith(CELLXGENE_SIDECAR_PATH)
    ]
    assert len(cell_calls) == 2
    assert "AML blast" in repr(cell_calls)
    assert all(
        record.supports == "context"
        and record.metadata["context_only"] is True
        and record.metadata["causal_inference_allowed"] is False
        for record in result.targets[0].evidence
        if record.source in {"Human Cell Atlas", "CELLxGENE Census", "ARCHS4"}
    )


def test_cell_line_model_does_not_query_single_cell_atlases() -> None:
    client = FakeContextClient()
    request = ContextRoutingRequest(
        disease="acute myeloid leukemia",
        targets=[
            ContextTarget(
                symbol="IDH1",
                track=HypothesisTrack.LITERATURE,
                rank=1,
            )
        ],
        target_cell_type="AML cell lines",
        budget=ContextRoutingBudget(max_source_calls=10, limit_per_source=1),
    )

    result = DiseaseTargetContextRouter(
        MA1PublicContextRetriever(
            client=client,
            cellxgene_sidecar_url="https://cell-sidecar.example.test",
        )
    ).route(request)
    outcomes = {item.source: item for item in result.targets[0].sources}

    assert outcomes["Human Cell Atlas"].status == "skipped_not_applicable"
    assert outcomes["CELLxGENE Census"].status == "skipped_not_applicable"
    assert outcomes["Human Cell Atlas"].queried is False
    assert outcomes["CELLxGENE Census"].queried is False
    assert "cell-line model" in outcomes["Human Cell Atlas"].detail
    assert all(
        url != HCA_PROJECTS_URL and not url.endswith(CELLXGENE_SIDECAR_PATH)
        for url, _ in client.calls
    )


def test_tbio_biogrid_without_key_is_not_configured_and_performs_no_io() -> None:
    client = FakeContextClient()
    request = ContextRoutingRequest(
        disease="test disease",
        targets=[
            ContextTarget(
                symbol="GPR132",
                track=HypothesisTrack.TBIO,
                rank=1,
            )
        ],
        budget=ContextRoutingBudget(max_source_calls=4, limit_per_source=1),
    )

    result = DiseaseTargetContextRouter(
        MA1PublicContextRetriever(client=client)
    ).route(request)
    outcomes = {item.source: item for item in result.targets[0].sources}

    assert outcomes["ARCHS4"].status == "ok"
    assert outcomes["BioGRID"].status == "not_configured"
    assert outcomes["BioGRID"].queried is False
    assert "BIOGRID_ACCESS_KEY" in outcomes["BioGRID"].detail
    assert all(url != BIOGRID_INTERACTIONS_URL for url, _ in client.calls)


class UnsafeContextRetriever:
    """Returns a wrongly classified record to verify the fail-closed boundary."""

    def lookup(self, request: Any) -> MA1ContextResult:
        source = request.sources[0]
        return MA1ContextResult(
            lookup=request,
            evidence=[
                EvidenceRef(
                    source=source,
                    external_id="unsafe-1",
                    title="Must not enter the context bundle",
                    supports="support",
                    metadata={"context_only": False},
                )
            ],
            sources=[
                ContextSourceResult(
                    source=source,
                    status="ok",
                    queried=True,
                    record_count=1,
                )
            ],
        )


def test_non_context_record_is_quarantined_not_relabelled() -> None:
    request = ContextRoutingRequest(
        disease="test disease",
        targets=[ContextTarget(symbol="TP53", track=HypothesisTrack.LITERATURE, rank=1)],
        budget=ContextRoutingBudget(max_source_calls=1),
    )
    result = DiseaseTargetContextRouter(UnsafeContextRetriever()).route(request)

    bundle = result.targets[0]
    assert bundle.evidence == []
    assert bundle.provenance == []
    assert bundle.sources[0].status == "policy_rejected"
    assert bundle.sources[0].rejected_record_count == 1
    assert "support" not in result.model_dump_json(exclude={"safety_note"})


def _hypothesis(
    symbol: str,
    track: HypothesisTrack,
    rank: int,
    *,
    uniprot_id: str | None = None,
) -> Hypothesis:
    evidence = EvidenceRef(
        source="fixture",
        external_id=f"fixture-{symbol}",
        title="Fixture hypothesis evidence",
    )
    return Hypothesis(
        hypothesis_id=f"hyp-{symbol}",
        track=track,
        disease="test disease",
        target=TargetCandidate(symbol=symbol, uniprot_id=uniprot_id),
        mechanism="A testable mechanism",
        intervention_direction="inhibit",
        falsification_test="Reject after a prespecified negative experiment.",
        claims=[Claim(text="Test claim", evidence_ids=[evidence.evidence_id])],
        evidence=[evidence],
        score=0.8,
        rank=rank,
    )


def test_hypothesis_packet_adapter_preserves_packet_and_maps_optional_ids() -> None:
    packet = HypothesisPacket(
        brief_id="brief-1",
        hypotheses=[
            _hypothesis("TP53", HypothesisTrack.LITERATURE, 1, uniprot_id="P04637"),
            _hypothesis("GPR132", HypothesisTrack.TBIO, 1, uniprot_id="Q9UNW8"),
        ],
    )
    before = packet.model_dump_json()

    request = context_request_from_hypothesis_packet(
        packet,
        budget=ContextRoutingBudget(max_source_calls=2),
        gencode_ids_by_symbol={"gpr132": "ENSG00000183484.12"},
    )

    assert packet.model_dump_json() == before
    assert request.disease == "test disease"
    assert request.targets[1].hypothesis_id == "hyp-GPR132"
    assert request.targets[1].uniprot_id == "Q9UNW8"
    assert request.targets[1].gencode_id == "ENSG00000183484.12"
