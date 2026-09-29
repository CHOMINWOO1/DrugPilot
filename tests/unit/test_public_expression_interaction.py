from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError
import pytest

from drugpilot.tools.public_expression_interaction import (
    ARCHS4_CORRELATION_URL,
    ARCHS4_DOCUMENTATION_URL,
    ARCHS4_DOWNLOAD_URL,
    BIOGRID_ACCESS_KEY_ENV,
    BIOGRID_DOCUMENTATION_URL,
    BIOGRID_INTERACTIONS_URL,
    ExpressionInteractionLookup,
    PublicExpressionInteractionRetriever,
    SUPPORTED_EXPRESSION_INTERACTION_SOURCES,
)


VALID_BIOGRID_KEY = "A" * 32


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        request_method: str = "GET",
        request_url: str = "https://example.test",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request(request_method, request_url)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(
                self.status_code,
                request=self.request,
                json={"error": "upstream failure"},
            )
            raise httpx.HTTPStatusError(
                f"upstream returned {self.status_code}",
                request=self.request,
                response=response,
            )

    def json(self) -> Any:
        return self._payload


class FakeExpressionInteractionClient:
    def __init__(
        self,
        *,
        failing_url: str | None = None,
        http_status_by_url: dict[str, int] | None = None,
        empty: bool = False,
    ) -> None:
        self.failing_url = failing_url
        self.http_status_by_url = http_status_by_url or {}
        self.empty = empty
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._respond("GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._respond("POST", url, kwargs)

    def _respond(
        self,
        method: str,
        url: str,
        kwargs: dict[str, Any],
    ) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if url == self.failing_url:
            raise httpx.ReadTimeout(
                "simulated upstream timeout",
                request=httpx.Request(method, url),
            )
        status = self.http_status_by_url.get(url, 200)

        if url == ARCHS4_CORRELATION_URL:
            if self.empty:
                payload: Any = {
                    "gene": kwargs["json"]["gene"],
                    "samples": [],
                    "positive_correlated_genes": [],
                    "negative_correlated_genes": [],
                }
            else:
                meta = kwargs["json"]["meta"]
                payload = {
                    "gene": kwargs["json"]["gene"],
                    "searchterm": meta,
                    "mean_log_expression": 8.75,
                    "positive_correlated_genes": [
                        {
                            "gene": "MDM2",
                            "correlation": 0.82,
                            "mean_log_expression": 4.2,
                        },
                        {
                            "gene": "CDKN1A",
                            "correlation": 0.73,
                            "mean_log_expression": 6.1,
                        },
                    ],
                    "negative_correlated_genes": [
                        {
                            "gene": "MKI67",
                            "correlation": -0.55,
                            "mean_log_expression": 3.8,
                        }
                    ],
                    "samples": ["GSM100", "GSM101", "GSM102"],
                }
            return FakeResponse(
                payload,
                status_code=status,
                request_method=method,
                request_url=url,
            )

        if url == BIOGRID_INTERACTIONS_URL:
            payload = (
                {}
                if self.empty
                else {
                    "101": {
                        "BIOGRID_INTERACTION_ID": 101,
                        "ENTREZ_GENE_A": 7157,
                        "ENTREZ_GENE_B": 4193,
                        "BIOGRID_ID_A": 113409,
                        "BIOGRID_ID_B": 110778,
                        "OFFICIAL_SYMBOL_A": "TP53",
                        "OFFICIAL_SYMBOL_B": "MDM2",
                        "EXPERIMENTAL_SYSTEM": "Affinity Capture-Western",
                        "EXPERIMENTAL_SYSTEM_TYPE": "physical",
                        "PUBMED_AUTHOR": "Momand J (1992)",
                        "PUBMED_ID": 1535557,
                        "ORGANISM_A": 9606,
                        "ORGANISM_B": 9606,
                        "THROUGHPUT": "Low Throughput",
                        "QUANTITATION": "-",
                        "MODIFICATION": "-",
                        "QUALIFICATIONS": "-",
                        "TAGS": "-",
                        "SOURCEDB": "BIOGRID",
                    },
                    "102": {
                        "BIOGRID_INTERACTION_ID": 102,
                        "OFFICIAL_SYMBOL_A": "TP53",
                        "OFFICIAL_SYMBOL_B": "ATM",
                        "EXPERIMENTAL_SYSTEM": "Reconstituted Complex",
                        "EXPERIMENTAL_SYSTEM_TYPE": "physical",
                        "PUBMED_ID": 12345678,
                        "ORGANISM_A": 9606,
                        "ORGANISM_B": 9606,
                        "THROUGHPUT": "Low Throughput",
                        "SOURCEDB": "BIOGRID",
                    },
                }
            )
            return FakeResponse(
                payload,
                status_code=status,
                request_method=method,
                request_url=url,
            )
        raise AssertionError(f"unexpected URL: {url}")


def _lookup(**updates: Any) -> ExpressionInteractionLookup:
    values: dict[str, Any] = {
        "gene_symbol": "TP53",
        "disease": "acute myeloid leukemia",
        "cell_type": "myeloblast",
        "tissue": "bone marrow",
        "species": "human",
        "limit_per_source": 2,
    }
    values.update(updates)
    return ExpressionInteractionLookup(**values)


def test_configured_adapters_query_gene_disease_cell_and_tissue_context() -> None:
    client = FakeExpressionInteractionClient()
    retriever = PublicExpressionInteractionRetriever(
        client=client,
        biogrid_access_key=VALID_BIOGRID_KEY,
        timeout_seconds=1.25,
    )

    result = retriever.lookup(_lookup())

    assert [summary.source for summary in result.sources] == list(
        SUPPORTED_EXPRESSION_INTERACTION_SOURCES
    )
    assert all(summary.status == "ok" for summary in result.sources)
    summaries = {summary.source: summary for summary in result.sources}
    assert summaries["ARCHS4"].query_count == 3
    assert summaries["ARCHS4"].record_count == 3
    assert summaries["BioGRID"].query_count == 1
    assert summaries["BioGRID"].record_count == 2

    archs4_calls = [call for call in client.calls if call[1] == ARCHS4_CORRELATION_URL]
    assert [call[2]["json"]["meta"] for call in archs4_calls] == [
        "acute myeloid leukemia",
        "myeloblast",
        "bone marrow",
    ]
    assert all(call[0] == "POST" for call in archs4_calls)
    assert all(call[2]["json"]["gene"] == "TP53" for call in archs4_calls)
    assert all(call[2]["json"]["species"] == "human" for call in archs4_calls)

    biogrid_call = next(call for call in client.calls if call[1] == BIOGRID_INTERACTIONS_URL)
    assert biogrid_call[0] == "GET"
    assert biogrid_call[2]["params"] == {
        "accesskey": VALID_BIOGRID_KEY,
        "format": "json",
        "geneList": "TP53",
        "searchNames": "true",
        "includeInteractors": "true",
        "includeInteractorInteractions": "false",
        "interSpeciesExcluded": "true",
        "taxId": 9606,
        "max": 2,
    }
    assert all(call[2]["timeout"] == 1.25 for call in client.calls)


def test_records_are_context_only_and_have_auditable_live_provenance() -> None:
    result = PublicExpressionInteractionRetriever(
        client=FakeExpressionInteractionClient(),
        biogrid_access_key=VALID_BIOGRID_KEY,
    ).lookup(_lookup(limit_per_source=1))

    assert {record.source for record in result.evidence} == {"ARCHS4", "BioGRID"}
    for record in result.evidence:
        assert record.supports == "context"
        assert record.url
        assert record.metadata["provenance_class"] == "external_live"
        assert record.metadata["context_only"] is True
        assert record.metadata["causal_inference_allowed"] is False
        assert record.metadata["counter_evidence_allowed"] is False
        assert record.metadata["read_only"] is True
        assert record.metadata["retrieval_endpoint"]
        assert record.metadata["query"]

    archs4 = next(record for record in result.evidence if record.source == "ARCHS4")
    assert archs4.metadata["context_kind"] == "disease"
    assert archs4.metadata["positive_correlated_genes"][0]["gene"] == "MDM2"
    assert archs4.metadata["sample_count"] == 3
    assert len(archs4.metadata["sample_ids"]) == 1

    biogrid = next(record for record in result.evidence if record.source == "BioGRID")
    assert biogrid.external_id == "101"
    assert biogrid.metadata["experimental_system_type"] == "physical"
    assert biogrid.metadata["pubmed_id"] == 1535557
    assert biogrid.metadata["server_filter_scope"] == "gene_symbol_and_tax_id"
    assert biogrid.metadata["disease_context"] == "acute myeloid leukemia"


def test_missing_biogrid_key_is_not_configured_and_performs_no_biogrid_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BIOGRID_ACCESS_KEY_ENV, raising=False)
    client = FakeExpressionInteractionClient()
    retriever = PublicExpressionInteractionRetriever(client=client)

    result = retriever.lookup(_lookup())
    summaries = {summary.source: summary for summary in result.sources}
    registry = {entry.source: entry for entry in retriever.registry()}

    assert summaries["ARCHS4"].status == "ok"
    assert summaries["BioGRID"].status == "not_configured"
    assert summaries["BioGRID"].queried is False
    assert summaries["BioGRID"].error_type == "missing_access_key"
    assert BIOGRID_ACCESS_KEY_ENV in (summaries["BioGRID"].detail or "")
    assert all(call[1] != BIOGRID_INTERACTIONS_URL for call in client.calls)
    assert registry["BioGRID"].runtime_state == "not_configured"
    assert registry["BioGRID"].requires_api_key is True
    assert registry["ARCHS4"].runtime_state == "adapter_ready"


def test_invalid_biogrid_key_is_configuration_error_not_remote_failure() -> None:
    client = FakeExpressionInteractionClient()
    result = PublicExpressionInteractionRetriever(
        client=client,
        biogrid_access_key="too-short",
    ).lookup(_lookup(sources=["BioGRID"]))

    summaries = {summary.source: summary for summary in result.sources}
    assert summaries["ARCHS4"].status == "skipped"
    assert summaries["BioGRID"].status == "not_configured"
    assert summaries["BioGRID"].queried is False
    assert summaries["BioGRID"].error_type == "invalid_access_key_configuration"
    assert client.calls == []


def test_biogrid_key_never_enters_result_or_cached_provenance() -> None:
    client = FakeExpressionInteractionClient()
    retriever = PublicExpressionInteractionRetriever(
        client=client,
        biogrid_access_key=VALID_BIOGRID_KEY,
    )
    request = _lookup(sources=["BioGRID"])

    first = retriever.lookup(request)
    first_call_count = len(client.calls)
    second = retriever.lookup(request)

    assert first_call_count == 1
    assert len(client.calls) == first_call_count
    assert first.sources[1].cache_hit is False
    assert second.sources[1].cache_hit is True
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
    assert VALID_BIOGRID_KEY not in first.model_dump_json()
    assert VALID_BIOGRID_KEY not in second.model_dump_json()


def test_source_failure_is_isolated_and_transport_error_is_classified() -> None:
    client = FakeExpressionInteractionClient(failing_url=ARCHS4_CORRELATION_URL)
    result = PublicExpressionInteractionRetriever(
        client=client,
        biogrid_access_key=VALID_BIOGRID_KEY,
    ).lookup(_lookup())

    summaries = {summary.source: summary for summary in result.sources}
    assert summaries["ARCHS4"].status == "error"
    assert summaries["ARCHS4"].error_type == "read_timeout"
    assert summaries["BioGRID"].status == "ok"
    assert {record.source for record in result.evidence} == {"BioGRID"}


def test_upstream_http_status_is_reported_distinctly() -> None:
    client = FakeExpressionInteractionClient(
        http_status_by_url={BIOGRID_INTERACTIONS_URL: 401}
    )
    result = PublicExpressionInteractionRetriever(
        client=client,
        biogrid_access_key=VALID_BIOGRID_KEY,
    ).lookup(_lookup(sources=["BioGRID"]))

    biogrid = result.sources[1]
    assert biogrid.status == "error"
    assert biogrid.error_type == "upstream_http_401"
    assert VALID_BIOGRID_KEY not in result.model_dump_json()


def test_gene_only_lookup_skips_archs4_but_can_query_biogrid() -> None:
    client = FakeExpressionInteractionClient()
    result = PublicExpressionInteractionRetriever(
        client=client,
        biogrid_access_key=VALID_BIOGRID_KEY,
    ).lookup(
        ExpressionInteractionLookup(
            gene_symbol=" TP53 ",
            species="human",
            limit_per_source=1,
        )
    )

    summaries = {summary.source: summary for summary in result.sources}
    assert result.lookup.gene_symbol == "TP53"
    assert summaries["ARCHS4"].status == "skipped"
    assert summaries["ARCHS4"].queried is False
    assert summaries["BioGRID"].status == "ok"
    assert [call[1] for call in client.calls] == [BIOGRID_INTERACTIONS_URL]


def test_empty_results_are_distinct_from_errors_and_are_cached() -> None:
    client = FakeExpressionInteractionClient(empty=True)
    retriever = PublicExpressionInteractionRetriever(
        client=client,
        biogrid_access_key=VALID_BIOGRID_KEY,
    )
    request = _lookup(disease="AML", cell_type=None, tissue=None)

    first = retriever.lookup(request)
    call_count = len(client.calls)
    second = retriever.lookup(request)

    assert all(summary.status == "empty" for summary in first.sources)
    assert all(summary.error_type is None for summary in first.sources)
    assert second.evidence == []
    assert all(summary.cache_hit for summary in second.sources)
    assert len(client.calls) == call_count


def test_registry_records_official_access_conditions_without_health_claim() -> None:
    registry = {
        entry.source: entry
        for entry in PublicExpressionInteractionRetriever(
            client=FakeExpressionInteractionClient(),
            biogrid_access_key=VALID_BIOGRID_KEY,
        ).registry()
    }

    assert registry["ARCHS4"].endpoint == ARCHS4_CORRELATION_URL
    assert registry["ARCHS4"].documentation_url == ARCHS4_DOCUMENTATION_URL
    assert registry["ARCHS4"].bulk_download_url == ARCHS4_DOWNLOAD_URL
    assert registry["ARCHS4"].access_mode == "public_api"
    assert registry["ARCHS4"].requires_api_key is False
    assert registry["BioGRID"].endpoint == BIOGRID_INTERACTIONS_URL
    assert registry["BioGRID"].documentation_url == BIOGRID_DOCUMENTATION_URL
    assert registry["BioGRID"].access_mode == "credentialed_api"
    assert registry["BioGRID"].credential_environment_variable == BIOGRID_ACCESS_KEY_ENV
    assert all(entry.connectivity_status == "not_tested" for entry in registry.values())


def test_lookup_contract_normalizes_contexts_and_rejects_invalid_fields() -> None:
    request = _lookup(
        disease=" AML ",
        cell_type="aml",
        tissue=" bone marrow ",
        sources=["ARCHS4", "ARCHS4", "BioGRID"],
    )
    assert request.contexts == (("disease", "AML"), ("tissue", "bone marrow"))
    assert request.sources == ["ARCHS4", "BioGRID"]

    with pytest.raises(ValidationError):
        ExpressionInteractionLookup(gene_symbol="   ")
    with pytest.raises(ValidationError):
        ExpressionInteractionLookup(gene_symbol="TP53", secret_option=True)
    with pytest.raises(ValidationError):
        ExpressionInteractionLookup(gene_symbol="TP53", species="rat")
