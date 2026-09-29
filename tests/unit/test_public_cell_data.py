from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from drugpilot.tools.public_cell_data import (
    CELLXGENE_SIDECAR_PATH,
    HCA_PROJECTS_URL,
    CellDataLookup,
    PublicCellDataRetriever,
    SUPPORTED_CELL_DATA_SOURCES,
)


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakeCellDataClient:
    def __init__(
        self,
        *,
        failing_urls: set[str] | None = None,
        hca_hits: list[dict[str, Any]] | None = None,
        sidecar_payload: Any | None = None,
    ) -> None:
        self.failing_urls = failing_urls or set()
        self.hca_hits = _hca_hits() if hca_hits is None else hca_hits
        self.sidecar_payload = sidecar_payload or _sidecar_payload()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if url in self.failing_urls:
            raise TimeoutError("private transport detail must not leak")
        if url == HCA_PROJECTS_URL:
            return FakeResponse(
                {
                    "pagination": {"total": len(self.hca_hits)},
                    "termFacets": {},
                    "hits": self.hca_hits,
                }
            )
        if url.endswith(CELLXGENE_SIDECAR_PATH):
            return FakeResponse(self.sidecar_payload)
        raise AssertionError(f"unexpected URL: {url}")


def _hca_hits() -> list[dict[str, Any]]:
    return [
        {
            "entryId": "hca-project-1",
            "projects": [
                {
                    "projectId": "hca-project-1",
                    "projectTitle": "Glioblastoma single-cell atlas",
                    "projectShortname": "GBM atlas",
                    "projectDescription": "Public project-level metadata.",
                    "estimatedCellCount": 42000,
                    "dataUseRestriction": "NRES",
                    "accessible": True,
                    "publications": [
                        {
                            "publicationTitle": "A glioblastoma atlas",
                            "doi": "10.1000/example",
                            "publicationUrl": "https://doi.org/10.1000/example",
                            "officialHcaPublication": False,
                        }
                    ],
                }
            ],
            "samples": [
                {"disease": ["glioblastoma"], "organ": ["brain"]}
            ],
            "specimens": [{"disease": ["glioblastoma"]}],
            "donorOrganisms": [
                {"genusSpecies": ["Homo sapiens"], "disease": ["glioblastoma"]}
            ],
            "cellSuspensions": [
                {"selectedCellType": ["malignant cell"]}
            ],
        }
    ]


def _sidecar_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "census_version": "2025-11-08",
        "provenance": {
            "census_uri": (
                "s3://cellxgene-census-public-us-west-2/"
                "cell-census/2025-11-08/soma/"
            ),
            "census_build_date": "2025-11-08",
            "cellxgene_census_version": "1.17.0",
            "queried_at": "2026-08-18T02:00:00Z",
            "ignored_secret": "must not be retained",
        },
        "records": [
            {
                "dataset_id": "cxg-dataset-1",
                "dataset_title": "Glioblastoma Census dataset",
                "collection_id": "cxg-collection-1",
                "matched_cell_count": 231,
                "disease": ["glioblastoma"],
                "disease_ontology_term_id": ["MONDO:0018177"],
                "cell_type": ["malignant cell"],
                "cell_type_ontology_term_id": ["CL:0001064"],
                "gene_symbol": "TP53",
                "feature_id": "ENSG00000141510",
                "gene_present": True,
                "assay": ["10x 3' v3"],
                "tissue": ["brain"],
                "citation": "Dataset citation",
                "untrusted_extra": "must not be retained",
            }
        ],
    }


def _lookup(**updates: Any) -> CellDataLookup:
    values: dict[str, Any] = {
        "disease": "glioblastoma",
        "gene_symbol": "TP53",
        "cell_type": "malignant cell",
        "limit_per_source": 3,
        "census_version": "2025-11-08",
    }
    values.update(updates)
    return CellDataLookup(**values)


def _retriever(client: FakeCellDataClient, **kwargs: Any) -> PublicCellDataRetriever:
    return PublicCellDataRetriever(
        client=client,
        cellxgene_sidecar_url=kwargs.pop(
            "cellxgene_sidecar_url", "http://127.0.0.1:8787/"
        ),
        timeout_seconds=kwargs.pop("timeout_seconds", 1.25),
        **kwargs,
    )


def test_registry_distinguishes_keyless_public_and_sidecar_configuration() -> None:
    unconfigured = PublicCellDataRetriever(client=FakeCellDataClient()).registry()
    configured = _retriever(FakeCellDataClient()).registry()

    assert [entry.source for entry in unconfigured] == list(
        SUPPORTED_CELL_DATA_SOURCES
    )
    hca, census = unconfigured
    assert hca.runtime_state == "adapter_ready"
    assert hca.configuration_state == "keyless_public"
    assert hca.supported_inputs == ["disease", "cell_type"]
    assert hca.unsupported_inputs == ["gene_symbol"]
    assert census.runtime_state == "configuration_required"
    assert census.configuration_state == "not_configured"
    assert census.endpoint is None
    assert all(not entry.requires_credentials for entry in unconfigured)
    assert all(entry.connectivity_status == "not_tested" for entry in unconfigured)

    configured_census = configured[1]
    assert configured_census.runtime_state == "adapter_ready"
    assert configured_census.configuration_state == "configured"
    assert configured_census.endpoint == (
        "http://127.0.0.1:8787" + CELLXGENE_SIDECAR_PATH
    )
    assert "not an official CELLxGENE REST endpoint" in configured_census.note


def test_hca_uses_official_exact_filters_and_normalizes_context_only_record() -> None:
    client = FakeCellDataClient()
    result = _retriever(client).lookup(
        _lookup(sources=["Human Cell Atlas"])
    )

    assert len(client.calls) == 1
    url, kwargs = client.calls[0]
    assert url == HCA_PROJECTS_URL
    filters = json.loads(kwargs["params"]["filters"])
    assert filters == {
        "genusSpecies": {"is": ["Homo sapiens"]},
        "accessible": {"is": [True]},
        "sampleDisease": {"is": ["glioblastoma"]},
        "selectedCellType": {"is": ["malignant cell"]},
    }
    assert kwargs["params"]["catalog"] == "dcp60"
    assert kwargs["params"]["size"] == 3
    assert kwargs["timeout"] == 1.25
    assert kwargs["headers"] == {"Accept": "application/json"}
    assert "Authorization" not in kwargs["headers"]

    summary = result.sources[0]
    assert summary.status == "ok"
    assert summary.provenance_class == "external_live"
    assert summary.ignored_inputs == ["gene_symbol"]
    record = result.evidence[0]
    assert record.source == "Human Cell Atlas"
    assert record.external_id == "hca-project-1"
    assert record.url.endswith("/hca-project-1")
    assert record.supports == "context"
    assert record.metadata["sample_diseases"] == ["glioblastoma"]
    assert record.metadata["selected_cell_types"] == ["malignant cell"]
    assert record.metadata["species"] == ["Homo sapiens"]
    assert record.metadata["gene_lookup_supported"] is False
    assert record.metadata["differential_expression_claim_allowed"] is False
    assert record.metadata["credentials_sent"] is False
    assert "contributors" not in record.metadata


def test_configured_cellxgene_sidecar_get_contract_and_provenance_allowlist() -> None:
    client = FakeCellDataClient()
    result = _retriever(client).lookup(
        _lookup(sources=["CELLxGENE Census"])
    )

    assert len(client.calls) == 1
    url, kwargs = client.calls[0]
    assert url == "http://127.0.0.1:8787" + CELLXGENE_SIDECAR_PATH
    assert kwargs["params"] == {
        "organism": "homo_sapiens",
        "census_version": "2025-11-08",
        "limit": 3,
        "disease": "glioblastoma",
        "gene_symbol": "TP53",
        "cell_type": "malignant cell",
    }
    assert kwargs["headers"] == {"Accept": "application/json"}
    assert kwargs["follow_redirects"] is False
    assert "Authorization" not in kwargs["headers"]

    summary = result.sources[1]
    assert summary.status == "ok"
    assert summary.provenance_class == "external_live"
    record = result.evidence[0]
    assert record.source == "CELLxGENE Census"
    assert record.external_id == "cxg-dataset-1"
    assert record.metadata["census_version"] == "2025-11-08"
    assert record.metadata["matched_cell_count"] == 231
    assert record.metadata["diseases"] == ["glioblastoma"]
    assert record.metadata["cell_types"] == ["malignant cell"]
    assert record.metadata["gene_symbol"] == "TP53"
    assert record.metadata["feature_id"] == "ENSG00000141510"
    assert record.metadata["access_path"] == "configured_sidecar_http"
    assert record.metadata["direct_upstream_call"] is False
    assert "ignored_secret" not in record.metadata["sidecar_provenance"]
    assert "untrusted_extra" not in record.metadata


def test_gene_only_skips_hca_but_queries_configured_census_sidecar() -> None:
    client = FakeCellDataClient()
    result = _retriever(client).lookup(
        _lookup(disease=None, cell_type=None, gene_symbol="TP53")
    )

    statuses = {summary.source: summary for summary in result.sources}
    assert statuses["Human Cell Atlas"].status == "skipped"
    assert statuses["Human Cell Atlas"].queried is False
    assert statuses["Human Cell Atlas"].ignored_inputs == ["gene_symbol"]
    assert "no gene/expression lookup" in statuses["Human Cell Atlas"].detail
    assert statuses["CELLxGENE Census"].status == "ok"
    assert [url for url, _ in client.calls] == [
        "http://127.0.0.1:8787" + CELLXGENE_SIDECAR_PATH
    ]
    assert client.calls[0][1]["params"] == {
        "organism": "homo_sapiens",
        "census_version": "2025-11-08",
        "limit": 3,
        "gene_symbol": "TP53",
    }


def test_unconfigured_sidecar_is_visible_and_does_not_block_hca() -> None:
    client = FakeCellDataClient()
    retriever = _retriever(client, cellxgene_sidecar_url=None)

    result = retriever.lookup(_lookup())

    statuses = {summary.source: summary for summary in result.sources}
    assert statuses["Human Cell Atlas"].status == "ok"
    assert statuses["CELLxGENE Census"].status == "not_configured"
    assert statuses["CELLxGENE Census"].queried is False
    assert statuses["CELLxGENE Census"].provenance_class == "not_applicable"
    assert [url for url, _ in client.calls] == [HCA_PROJECTS_URL]


def test_one_source_timeout_is_isolated_and_error_text_is_not_exposed() -> None:
    sidecar_endpoint = "http://127.0.0.1:8787" + CELLXGENE_SIDECAR_PATH
    client = FakeCellDataClient(failing_urls={HCA_PROJECTS_URL})
    result = _retriever(client).lookup(_lookup())

    statuses = {summary.source: summary for summary in result.sources}
    assert statuses["Human Cell Atlas"].status == "error"
    assert statuses["Human Cell Atlas"].error_type == "TimeoutError"
    assert statuses["CELLxGENE Census"].status == "ok"
    assert [url for url, _ in client.calls] == [HCA_PROJECTS_URL, sidecar_endpoint]
    assert "private transport detail" not in result.model_dump_json()
    assert [record.source for record in result.evidence] == ["CELLxGENE Census"]


def test_per_source_cache_marks_provenance_and_caches_empty_results() -> None:
    client = FakeCellDataClient(hca_hits=[])
    retriever = _retriever(client)
    request = _lookup()

    first = retriever.lookup(request)
    first_call_count = len(client.calls)
    second = retriever.lookup(request)

    assert first_call_count == 2
    assert len(client.calls) == first_call_count
    first_statuses = {summary.source: summary for summary in first.sources}
    second_statuses = {summary.source: summary for summary in second.sources}
    assert first_statuses["Human Cell Atlas"].status == "empty"
    assert first_statuses["Human Cell Atlas"].cache_hit is False
    assert second_statuses["Human Cell Atlas"].status == "empty"
    assert second_statuses["Human Cell Atlas"].cache_hit is True
    assert second_statuses["Human Cell Atlas"].provenance_class == "external_cache"
    assert second_statuses["CELLxGENE Census"].cache_hit is True
    census_record = second.evidence[0]
    assert census_record.metadata["provenance_class"] == "external_cache"
    assert census_record.metadata["cache_hit"] is True
    assert census_record.metadata["live"] is False
    assert census_record.metadata["retrieved_live"] is True
    assert census_record.metadata["origin_provenance_class"] == "external_live"
    assert census_record.metadata["origin_content_hash"] == census_record.excerpt_hash


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": "2.0", "census_version": "latest", "records": []},
        {"schema_version": "1.0", "records": []},
        {"schema_version": "1.0", "census_version": "latest", "records": {}},
    ],
)
def test_malformed_sidecar_contract_becomes_source_error(payload: Any) -> None:
    client = FakeCellDataClient(sidecar_payload=payload)
    result = _retriever(client).lookup(
        _lookup(sources=["CELLxGENE Census"])
    )

    census = result.sources[1]
    assert census.status == "error"
    assert census.error_type == "ValueError"
    assert result.evidence == []


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1:8787",
        "127.0.0.1:8787",
        "http://user:secret@127.0.0.1:8787",
        "http://127.0.0.1:8787?token=secret",
        "http://127.0.0.1:8787/#fragment",
    ],
)
def test_sidecar_url_rejects_non_http_credentials_and_embedded_state(url: str) -> None:
    with pytest.raises(ValueError):
        PublicCellDataRetriever(client=FakeCellDataClient(), cellxgene_sidecar_url=url)


def test_lookup_normalizes_inputs_and_requires_a_query_dimension() -> None:
    lookup = CellDataLookup(
        disease="  glioblastoma  ",
        sources=["Human Cell Atlas", "Human Cell Atlas"],
    )

    assert lookup.disease == "glioblastoma"
    assert lookup.sources == ["Human Cell Atlas"]
    with pytest.raises(ValidationError):
        CellDataLookup(disease=" ", gene_symbol=None, cell_type="  ")
    with pytest.raises(ValidationError):
        CellDataLookup(disease="glioblastoma", census_version="  ")
