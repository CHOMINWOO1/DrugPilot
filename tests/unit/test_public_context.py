from __future__ import annotations

from typing import Any

from drugpilot.tools.public_context import (
    CLINICAL_TRIALS_SEARCH_URL,
    EUROPE_PMC_SEARCH_URL,
    GTEX_GENE_SEARCH_URL,
    KEGG_HUMAN_GENE_FIND_URL,
    MA1ContextLookup,
    MA1PublicContextRetriever,
    QUICKGO_ANNOTATION_URL,
    REACTOME_SEARCH_URL,
    STRING_NETWORK_URL,
    SUPPORTED_CONTEXT_SOURCES,
)


class FakeResponse:
    def __init__(self, payload: Any = None, *, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakeContextClient:
    def __init__(self, *, fail_url: str | None = None) -> None:
        self.fail_url = fail_url
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if url == self.fail_url:
            raise TimeoutError("simulated source timeout")
        if url == EUROPE_PMC_SEARCH_URL:
            query = kwargs["params"]["query"]
            server = "bioRxiv" if 'PUBLISHER:"bioRxiv"' in query else "medRxiv"
            return FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "doi": f"10.1101/{server.lower()}.001",
                                "id": f"PPR-{server}",
                                "title": f"{server} TP53 AML study",
                                "firstPublicationDate": "2026-01-02",
                                "authorString": "Researcher A",
                            }
                        ]
                    }
                }
            )
        if url == GTEX_GENE_SEARCH_URL:
            return FakeResponse(
                {
                    "data": [
                        {
                            "gencodeId": "ENSG00000141510.18",
                            "geneSymbol": "TP53",
                            "geneName": "tumor protein p53",
                            "chromosome": "chr17",
                            "start": 7661779,
                            "end": 7687538,
                            "strand": "-",
                        }
                    ]
                }
            )
        if url == STRING_NETWORK_URL:
            return FakeResponse(
                [
                    {
                        "stringId_A": "9606.ENSP00000269305",
                        "stringId_B": "9606.ENSP00000418960",
                        "preferredName_A": "TP53",
                        "preferredName_B": "MDM2",
                        "score": 0.999,
                        "experimental": 0.88,
                    }
                ]
            )
        if url == REACTOME_SEARCH_URL:
            return FakeResponse(
                {
                    "results": [
                        {
                            "entries": [
                                {
                                    "stId": "R-HSA-5633007",
                                    "name": "Regulation of TP53 Activity",
                                    "type": "Pathway",
                                    "species": ["Homo sapiens"],
                                }
                            ]
                        }
                    ]
                }
            )
        if url.startswith(KEGG_HUMAN_GENE_FIND_URL):
            return FakeResponse(text="hsa:7157\tTP53, P53; tumor protein p53\n")
        if url == QUICKGO_ANNOTATION_URL:
            return FakeResponse(
                {
                    "results": [
                        {
                            "geneProductId": "UniProtKB:P04637",
                            "symbol": "TP53",
                            "goId": "GO:0003677",
                            "goAspect": "molecular_function",
                            "evidenceCode": "ECO:0000269",
                            "reference": "PMID:123456",
                            "assignedBy": "UniProt",
                        }
                    ]
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
                                    "briefTitle": "TP53-directed AML trial",
                                },
                                "statusModule": {"overallStatus": "RECRUITING"},
                                "designModule": {
                                    "studyType": "INTERVENTIONAL",
                                    "phases": ["PHASE2"],
                                },
                                "conditionsModule": {"conditions": ["AML"]},
                                "armsInterventionsModule": {
                                    "interventions": [
                                        {"name": "Drug A", "type": "DRUG"}
                                    ]
                                },
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected URL: {url}")


def _lookup(**updates: Any) -> MA1ContextLookup:
    values: dict[str, Any] = {
        "disease": "acute myeloid leukemia",
        "gene_symbol": "TP53",
        "uniprot_id": "P04637",
        "gencode_id": "ENSG00000141510.18",
        "limit_per_source": 1,
    }
    values.update(updates)
    return MA1ContextLookup(**values)


def test_all_keyless_context_adapters_normalize_context_only_records() -> None:
    client = FakeContextClient()
    retriever = MA1PublicContextRetriever(client=client, timeout_seconds=1.25)

    result = retriever.lookup(_lookup())

    assert [summary.source for summary in result.sources] == list(
        SUPPORTED_CONTEXT_SOURCES
    )
    assert all(summary.status == "ok" for summary in result.sources)
    assert {record.source for record in result.evidence} == set(
        SUPPORTED_CONTEXT_SOURCES
    )
    assert len(result.evidence) == len(SUPPORTED_CONTEXT_SOURCES)
    for record in result.evidence:
        assert record.supports == "context"
        assert record.url
        assert record.metadata["provenance_class"] == "external_live"
        assert record.metadata["query"]
        assert record.metadata["context_only"] is True
        assert record.metadata["causal_inference_allowed"] is False
        assert record.metadata["counter_evidence_allowed"] is False
        assert record.metadata["retrieval_endpoint"]
    assert all(call[1]["timeout"] == 1.25 for call in client.calls)


def test_preprint_queries_are_server_scoped_through_europe_pmc() -> None:
    client = FakeContextClient()
    retriever = MA1PublicContextRetriever(client=client)

    result = retriever.lookup(_lookup(sources=["bioRxiv", "medRxiv"]))

    preprints = {record.source: record for record in result.evidence}
    assert 'PUBLISHER:"bioRxiv"' in preprints["bioRxiv"].metadata["query"]
    assert 'PUBLISHER:"medRxiv"' in preprints["medRxiv"].metadata["query"]
    assert all(record.metadata["is_preprint"] for record in preprints.values())
    assert all(
        record.metadata["access_path"] == "Europe PMC federation"
        for record in preprints.values()
    )
    assert sum(url == EUROPE_PMC_SEARCH_URL for url, _ in client.calls) == 2


def test_cache_is_per_source_and_provenance_changes_to_external_cache() -> None:
    client = FakeContextClient()
    retriever = MA1PublicContextRetriever(client=client)
    request = _lookup(sources=["GTEx", "STRING"])

    first = retriever.lookup(request)
    first_call_count = len(client.calls)
    second = retriever.lookup(request)

    assert first_call_count == 2
    assert len(client.calls) == first_call_count
    assert all(not summary.cache_hit for summary in first.sources if summary.status == "ok")
    cached_summaries = [summary for summary in second.sources if summary.status == "ok"]
    assert len(cached_summaries) == 2
    assert all(summary.cache_hit for summary in cached_summaries)
    assert all(
        record.metadata["provenance_class"] == "external_cache"
        for record in second.evidence
    )
    assert all(record.metadata["cache_hit"] is True for record in second.evidence)
    assert all(
        record.metadata["origin_provenance_class"] == "external_live"
        for record in second.evidence
    )
    assert all(
        record.metadata["origin_content_hash"] == record.excerpt_hash
        for record in second.evidence
    )


def test_one_source_failure_is_isolated_and_does_not_become_counter_evidence() -> None:
    client = FakeContextClient(fail_url=STRING_NETWORK_URL)
    retriever = MA1PublicContextRetriever(client=client)

    result = retriever.lookup(_lookup(sources=["STRING", "KEGG"]))

    statuses = {summary.source: summary for summary in result.sources}
    assert statuses["STRING"].status == "error"
    assert statuses["STRING"].error_type == "TimeoutError"
    assert statuses["KEGG"].status == "ok"
    assert [record.source for record in result.evidence] == ["KEGG"]
    assert result.evidence[0].supports == "context"
    assert "simulated source timeout" not in result.model_dump_json()


def test_quickgo_is_explicitly_skipped_without_uniprot_identifier() -> None:
    client = FakeContextClient()
    retriever = MA1PublicContextRetriever(client=client)

    result = retriever.lookup(
        _lookup(uniprot_id=None, sources=["QuickGO", "ClinicalTrials.gov"])
    )

    statuses = {summary.source: summary for summary in result.sources}
    assert statuses["QuickGO"].status == "skipped"
    assert statuses["QuickGO"].queried is False
    assert statuses["QuickGO"].detail == "uniprot_id is required"
    assert statuses["ClinicalTrials.gov"].status == "ok"
    assert all(url != QUICKGO_ANNOTATION_URL for url, _ in client.calls)


def test_registry_is_adapter_truth_not_connectivity_or_gated_source_claim() -> None:
    registry = MA1PublicContextRetriever(client=FakeContextClient()).registry()

    assert [entry.source for entry in registry] == list(SUPPORTED_CONTEXT_SOURCES)
    assert all(entry.runtime_state == "adapter_ready" for entry in registry)
    assert all(entry.connectivity_status == "not_tested" for entry in registry)
    assert all(entry.requires_api_key is False for entry in registry)
    assert {entry.access_mode for entry in registry} == {
        "live_api",
        "federated_api",
    }
    serialized = " ".join(entry.model_dump_json() for entry in registry)
    for gated_source in (
        "Human Cell Atlas",
        "CELLxGENE Census",
        "ARCHS4",
        "BioGRID",
        "Google Patents",
        "DrugBank",
    ):
        assert gated_source not in serialized
