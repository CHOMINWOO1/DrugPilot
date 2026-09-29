from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from drugpilot.tools.target_identity import (
    CanonicalTargetIdentity,
    FixtureTargetIdentity,
    OPEN_TARGETS_GRAPHQL_URL,
    TargetIdentityQuery,
    TargetIdentityResolver,
    UNIPROT_SEARCH_URL,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeIdentityClient:
    def __init__(
        self,
        *,
        fail_open_targets: bool = False,
        fail_uniprot: bool = False,
        empty_open_targets: bool = False,
        empty_uniprot: bool = False,
    ) -> None:
        self.fail_open_targets = fail_open_targets
        self.fail_uniprot = fail_uniprot
        self.empty_open_targets = empty_open_targets
        self.empty_uniprot = empty_uniprot
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        assert url == OPEN_TARGETS_GRAPHQL_URL
        self.calls.append(("POST", url, kwargs))
        if self.fail_open_targets:
            raise httpx.ConnectTimeout("Open Targets offline", request=httpx.Request("POST", url))
        variables = kwargs["json"]["variables"]
        if self.empty_open_targets:
            if "ensemblId" in variables:
                return FakeResponse({"data": {"target": None}})
            return FakeResponse({"data": {"search": {"hits": []}}})
        if "ensemblId" in variables:
            return FakeResponse(
                {
                    "data": {
                        "target": {
                            "id": "ENSG00000122025",
                            "approvedSymbol": "FLT3",
                            "approvedName": "fms related receptor tyrosine kinase 3",
                            "biotype": "protein_coding",
                        }
                    }
                }
            )
        return FakeResponse(
            {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": "ENSG00000122025",
                                "name": "FLT3",
                                "entity": "target",
                                "description": "fms related receptor tyrosine kinase 3",
                            }
                        ]
                    }
                }
            }
        )


    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        assert url == UNIPROT_SEARCH_URL
        self.calls.append(("GET", url, kwargs))
        if self.fail_uniprot:
            raise httpx.ConnectTimeout("UniProt offline", request=httpx.Request("GET", url))
        if self.empty_uniprot:
            return FakeResponse({"results": []})
        return FakeResponse(
            {
                "results": [
                    {
                        "primaryAccession": "P36888",
                        "uniProtkbId": "FLT3_HUMAN",
                        "entryType": "UniProtKB reviewed (Swiss-Prot)",
                        "proteinDescription": {
                            "recommendedName": {
                                "fullName": {
                                    "value": "Receptor-type tyrosine-protein kinase FLT3"
                                }
                            }
                        },
                        "genes": [{"geneName": {"value": "FLT3"}}],
                        "organism": {
                            "scientificName": "Homo sapiens",
                            "taxonId": 9606,
                        },
                        "uniProtKBCrossReferences": [
                            {
                                "database": "Ensembl",
                                "id": "ENST00000241453.12",
                                "properties": [
                                    {"key": "GeneId", "value": "ENSG00000122025.16"}
                                ],
                            }
                        ],
                    }
                ]
            }
        )


class ConflictingEnsemblClient(FakeIdentityClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        response = super().get(url, **kwargs)
        payload = response.json()
        references = payload["results"][0]["uniProtKBCrossReferences"]
        references[0]["properties"][0]["value"] = "ENSG00000999999.1"
        return FakeResponse(payload)


class MixedReviewStatusClient(FakeIdentityClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        response = super().get(url, **kwargs)
        reviewed = response.json()["results"][0]
        unreviewed = deepcopy(reviewed)
        unreviewed["primaryAccession"] = "A00001"
        unreviewed["uniProtkbId"] = "FLT3_TREMBL"
        unreviewed["entryType"] = "UniProtKB unreviewed (TrEMBL)"
        return FakeResponse({"results": [unreviewed, reviewed]})


def test_resolves_canonical_human_identity_from_both_sources() -> None:
    client = FakeIdentityClient()
    result = TargetIdentityResolver(client=client, timeout_seconds=1.25).resolve("flt3")

    assert result.status == "resolved"
    assert result.identity is not None
    assert result.identity.canonical_symbol == "FLT3"
    assert result.identity.ensembl_id == "ENSG00000122025"
    assert result.identity.uniprot_accession == "P36888"
    assert result.identity.uniprot_entry_name == "FLT3_HUMAN"
    assert result.identity.taxon_id == 9606
    assert result.ready_for_structure_lookup is True
    assert {source.source: source.status for source in result.sources} == {
        "Open Targets": "ok",
        "UniProt": "ok",
    }
    assert all(call[2]["timeout"] == 1.25 for call in client.calls)
    assert all(source.provenance_class == "external_live" for source in result.sources)


def test_conflicting_ensembl_ids_disable_structure_lookup() -> None:
    result = TargetIdentityResolver(
        client=ConflictingEnsemblClient(), timeout_seconds=1
    ).resolve("FLT3")

    assert result.status == "partial"
    assert result.identity is not None
    assert result.identity.ensembl_id == "ENSG00000122025"
    assert result.identity.uniprot_accession is None
    assert result.ready_for_structure_lookup is False
    assert "Open Targets and UniProt Ensembl identifiers disagree" in result.warnings


def test_reviewed_swissprot_entry_is_preferred_over_unreviewed_trembl() -> None:
    result = TargetIdentityResolver(
        client=MixedReviewStatusClient(), timeout_seconds=1
    ).resolve("FLT3")

    assert result.status == "resolved"
    assert result.identity is not None
    assert result.identity.uniprot_accession == "P36888"
    assert result.identity.uniprot_entry_name == "FLT3_HUMAN"


def test_optional_ensembl_id_uses_exact_open_targets_lookup() -> None:
    client = FakeIdentityClient()
    resolver = TargetIdentityResolver(client=client, timeout_seconds=1)

    result = resolver.resolve("FLT3", "ensg00000122025.16")

    assert result.status == "resolved"
    assert result.query.ensembl_id == "ENSG00000122025"
    open_targets_call = next(call for call in client.calls if call[0] == "POST")
    assert open_targets_call[2]["json"]["variables"] == {
        "ensemblId": "ENSG00000122025"
    }
    open_targets_source = next(
        source for source in result.sources if source.source == "Open Targets"
    )
    assert open_targets_source.matched_by == "ensembl_id"


def test_open_targets_failure_is_isolated_and_uniprot_still_resolves() -> None:
    client = FakeIdentityClient(fail_open_targets=True)
    result = TargetIdentityResolver(client=client, timeout_seconds=1).resolve("FLT3")

    assert result.status == "resolved"
    assert result.identity is not None
    assert result.identity.ensembl_id == "ENSG00000122025"
    assert result.identity.uniprot_accession == "P36888"
    summaries = {source.source: source for source in result.sources}
    assert summaries["Open Targets"].status == "error"
    assert summaries["Open Targets"].error_type == "ConnectTimeout"
    assert summaries["UniProt"].status == "ok"
    assert result.warnings == ["Open Targets unavailable: ConnectTimeout"]


def test_uniprot_failure_preserves_partial_open_targets_identity() -> None:
    client = FakeIdentityClient(fail_uniprot=True)
    result = TargetIdentityResolver(client=client, timeout_seconds=1).resolve("FLT3")

    assert result.status == "partial"
    assert result.identity is not None
    assert result.identity.ensembl_id == "ENSG00000122025"
    assert result.identity.uniprot_accession is None
    summaries = {source.source: source for source in result.sources}
    assert summaries["Open Targets"].status == "ok"
    assert summaries["UniProt"].status == "error"
    assert result.ready_for_structure_lookup is False


def test_empty_services_return_unresolved_without_copying_input_as_an_id() -> None:
    client = FakeIdentityClient(empty_open_targets=True, empty_uniprot=True)
    result = TargetIdentityResolver(client=client, timeout_seconds=1).resolve(
        "UNKNOWN1", "ENSG00000999999"
    )

    assert result.status == "unresolved"
    assert result.identity is None
    assert all(source.status == "empty" for source in result.sources)
    assert not any(source.ensembl_id for source in result.sources)
    assert not any(source.uniprot_accession for source in result.sources)


def test_explicit_fixture_mapping_is_never_mislabelled_as_live() -> None:
    fixture = FixtureTargetIdentity(
        identity=CanonicalTargetIdentity(
            canonical_symbol="FLT3",
            ensembl_id="ENSG00000122025",
            uniprot_accession="P36888",
            uniprot_entry_name="FLT3_HUMAN",
            protein_name="Offline FLT3 demonstration record",
        ),
        fixture_label="DrugPilot offline target registry v1",
    )
    client = FakeIdentityClient(empty_open_targets=True, empty_uniprot=True)
    result = TargetIdentityResolver(
        client=client,
        timeout_seconds=1,
        fixture_mapping={"FLT3": fixture},
    ).resolve("FLT3")

    assert result.status == "fixture"
    assert result.identity == fixture.identity
    fixture_source = result.sources[-1]
    assert fixture_source.source == "fixture"
    assert fixture_source.queried is False
    assert fixture_source.provenance_class == "fixture_fallback"
    assert "Offline fixture used" in result.warnings[-1]


def test_cache_avoids_duplicate_requests_and_changes_provenance() -> None:
    client = FakeIdentityClient()
    resolver = TargetIdentityResolver(client=client, timeout_seconds=1)

    first = resolver.resolve("FLT3")
    call_count = len(client.calls)
    second = resolver.resolve("FLT3")

    assert call_count == 2
    assert len(client.calls) == call_count
    assert all(source.provenance_class == "external_live" for source in first.sources)
    assert all(source.provenance_class == "external_cache" for source in second.sources)
    assert all(source.cache_hit is True for source in second.sources)
    assert all(source.queried is False for source in second.sources)


def test_models_are_strict_and_reject_invalid_identifiers() -> None:
    with pytest.raises(ValidationError):
        TargetIdentityQuery(gene_symbol="FLT3", ensembl_id="ENSG_NOT_REAL")
    with pytest.raises(ValidationError):
        TargetIdentityQuery(gene_symbol="FLT3", unexpected=True)
