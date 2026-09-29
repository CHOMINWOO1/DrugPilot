from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from drugpilot.tools.public_chemistry import (
    ChEMBLAdapter,
    ChemistryLookup,
    PublicChemistryEnricher,
)


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakePublicChemistryClient:
    def __init__(
        self,
        *,
        failing_sources: set[str] | None = None,
        http_status_sources: dict[str, int] | None = None,
        exceptions_by_source: dict[str, Exception] | None = None,
    ) -> None:
        self.failing_sources = failing_sources or set()
        self.http_status_sources = http_status_sources or {}
        self.exceptions_by_source = exceptions_by_source or {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    @staticmethod
    def _source(url: str) -> str:
        if "chembl" in url:
            return "ChEMBL"
        if "pubchem" in url:
            return "PubChem"
        if "bindingdb" in url:
            return "BindingDB"
        if "rcsb" in url:
            return "RCSB PDB"
        if "alphafold" in url:
            return "AlphaFold DB"
        raise AssertionError(f"unexpected URL: {url}")

    def _respond(self, method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        source = self._source(url)
        self.calls.append((method, url, kwargs))
        if source in self.exceptions_by_source:
            raise self.exceptions_by_source[source]
        if source in self.failing_sources:
            raise httpx.ConnectTimeout(f"{source} unavailable")
        if source in self.http_status_sources:
            status_code = self.http_status_sources[source]
            request = httpx.Request(method, url)
            response = httpx.Response(status_code, request=request)
            raise httpx.HTTPStatusError(
                f"upstream HTTP {status_code}",
                request=request,
                response=response,
            )
        if source == "ChEMBL":
            return FakeResponse(
                {
                    "molecules": [
                        {
                            "molecule_chembl_id": "CHEMBL123",
                            "pref_name": "Quizartinib",
                            "molecule_type": "Small molecule",
                            "max_phase": 3,
                            "first_approval": None,
                            "molecule_structures": {
                                "canonical_smiles": "COC1=NC=CC(=C1)C(=O)N",
                                "standard_inchi_key": "FAKEINCHIKEY123",
                            },
                        }
                    ]
                }
            )
        if source == "PubChem":
            return FakeResponse(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 24889392,
                                "IUPACName": "quizartinib",
                                "CanonicalSMILES": "COC1=NC=CC(=C1)C(=O)N",
                                "IsomericSMILES": "COC1=NC=CC(=C1)C(=O)N",
                                "InChIKey": "FAKEINCHIKEY123",
                                "MolecularFormula": "C29H32ClN5O2",
                                "MolecularWeight": "560.1",
                            }
                        ]
                    }
                }
            )
        if source == "BindingDB":
            return FakeResponse(
                {
                    "getLindsByUniprotsResponse": {
                        "affinities": [
                            {
                                "monomerid": "BDBM501",
                                "smile": "COC1=NC=CC(=C1)C(=O)N",
                                "affinity_type": "Ki",
                                "affinity": "12 nM",
                            }
                        ]
                    }
                }
            )
        if source == "RCSB PDB":
            return FakeResponse(
                {
                    "result_set": [
                        {"identifier": "4RT7", "score": 1.0},
                        {"identifier": "6JQR", "score": 0.97},
                    ]
                }
            )
        return FakeResponse(
            [
                {
                    "entryId": "AF-P36888-F1",
                    "uniprotAccession": "P36888",
                    "uniprotDescription": "Receptor-type tyrosine-protein kinase FLT3",
                    "gene": "FLT3",
                    "organismScientificName": "Homo sapiens",
                    "taxId": 9606,
                    "globalMetricValue": 84.2,
                    "latestVersion": 6,
                    "cifUrl": "https://example.test/AF-P36888-F1.cif",
                    "pdbUrl": "https://example.test/AF-P36888-F1.pdb",
                    "paeDocUrl": "https://example.test/AF-P36888-F1-pae.json",
                }
            ]
        )

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._respond("GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._respond("POST", url, kwargs)


def _lookup() -> ChemistryLookup:
    return ChemistryLookup(
        candidate_name="quizartinib",
        inchikey="FAKEINCHIKEY123",
        smiles="COC1=NC=CC(=C1)C(=O)N",
        uniprot_id="P36888",
        limit_per_source=2,
    )


def test_enricher_returns_attachable_evidence_from_all_configured_sources() -> None:
    client = FakePublicChemistryClient()
    result = PublicChemistryEnricher(client=client, timeout_seconds=1.25).enrich(_lookup())

    assert {item.source for item in result.evidence} == {
        "ChEMBL",
        "PubChem",
        "BindingDB",
        "RCSB PDB",
        "AlphaFold DB",
    }
    assert len(result.evidence) == 6
    assert all(item.supports == "context" for item in result.evidence)
    assert all(item.metadata["provenance_class"] == "external_live" for item in result.evidence)
    assert all(item.metadata["mock"] is False for item in result.evidence)
    summaries = {summary.source: summary for summary in result.sources}
    assert summaries["DrugCentral"].status == "not_configured"
    assert all(
        summary.status == "ok"
        for summary in result.sources
        if summary.source != "DrugCentral"
    )
    assert all(call[2]["timeout"] == 1.25 for call in client.calls)
    bindingdb = next(item for item in result.evidence if item.source == "BindingDB")
    assert bindingdb.metadata["smiles"] == "COC1=NC=CC(=C1)C(=O)N"


def test_chembl_routes_identifiers_to_documented_query_semantics() -> None:
    client = FakePublicChemistryClient()
    adapter = ChEMBLAdapter(client=client, timeout_seconds=1)

    name_records, _ = adapter.lookup(ChemistryLookup(candidate_name="quizartinib"))
    inchikey_records, _ = adapter.lookup(ChemistryLookup(inchikey="FAKEINCHIKEY123"))
    smiles_records, _ = adapter.lookup(
        ChemistryLookup(smiles="COC1=NC=CC(=C1)C(=O)N")
    )

    assert len(name_records) == len(inchikey_records) == len(smiles_records) == 1
    name_call, inchikey_call, smiles_call = client.calls
    assert name_call[1] == ChEMBLAdapter.endpoint
    assert name_call[2]["params"] == {"q": "quizartinib", "limit": 3}
    assert inchikey_call[1] == ChEMBLAdapter.molecule_endpoint
    assert inchikey_call[2]["params"] == {
        "molecule_structures__standard_inchi_key": "FAKEINCHIKEY123",
        "limit": 3,
    }
    assert smiles_call[1] == ChEMBLAdapter.molecule_endpoint
    assert smiles_call[2]["params"] == {
        "molecule_structures__canonical_smiles__connectivity": (
            "COC1=NC=CC(=C1)C(=O)N"
        ),
        "limit": 3,
    }
    assert name_records[0].metadata["query_semantics"] == "full_text_name_search"
    assert inchikey_records[0].metadata["query_semantics"] == "exact_standard_inchi_key"
    assert smiles_records[0].metadata["query_semantics"] == "smiles_connectivity"


def test_cache_avoids_duplicate_requests_and_marks_cached_provenance() -> None:
    client = FakePublicChemistryClient()
    enricher = PublicChemistryEnricher(client=client, timeout_seconds=1)

    first = enricher.enrich(_lookup())
    first_call_count = len(client.calls)
    second = enricher.enrich(_lookup())

    assert first_call_count == 5
    assert len(client.calls) == first_call_count
    assert not any(summary.cache_hit for summary in first.sources)
    assert all(
        summary.cache_hit
        for summary in second.sources
        if summary.source != "DrugCentral"
    )
    assert all(item.metadata["cache_hit"] is True for item in second.evidence)
    assert all(item.metadata["provenance_class"] == "external_cache" for item in second.evidence)


def test_one_source_failure_is_isolated_from_other_public_sources() -> None:
    client = FakePublicChemistryClient(failing_sources={"RCSB PDB"})
    result = PublicChemistryEnricher(client=client, timeout_seconds=1).enrich(_lookup())

    summaries = {summary.source: summary for summary in result.sources}
    assert summaries["RCSB PDB"].status == "error"
    assert summaries["RCSB PDB"].error_type == "connect_timeout"
    assert summaries["ChEMBL"].status == "ok"
    assert summaries["PubChem"].status == "ok"
    assert summaries["BindingDB"].status == "ok"
    assert summaries["AlphaFold DB"].status == "ok"
    assert "RCSB PDB" not in {item.source for item in result.evidence}


def test_upstream_http_500_is_not_mislabelled_as_a_connection_failure() -> None:
    client = FakePublicChemistryClient(http_status_sources={"ChEMBL": 500})
    result = PublicChemistryEnricher(client=client, timeout_seconds=1).enrich(_lookup())

    chembl = next(summary for summary in result.sources if summary.source == "ChEMBL")
    assert chembl.status == "error"
    assert chembl.error_type == "upstream_http_500"
    assert any(item.source == "PubChem" for item in result.evidence)


def test_transport_failures_are_classified_for_operator_action() -> None:
    client = FakePublicChemistryClient(
        exceptions_by_source={
            "ChEMBL": httpx.ReadTimeout(
                "upstream did not respond",
                request=httpx.Request("GET", ChEMBLAdapter.endpoint),
            ),
            "RCSB PDB": httpx.ConnectError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                "self-signed certificate in certificate chain",
                request=httpx.Request(
                    "POST", "https://search.rcsb.org/rcsbsearch/v2/query"
                ),
            ),
        }
    )

    result = PublicChemistryEnricher(client=client, timeout_seconds=1).enrich(_lookup())
    summaries = {summary.source: summary for summary in result.sources}

    assert summaries["ChEMBL"].error_type == "read_timeout"
    assert summaries["RCSB PDB"].error_type == "tls_certificate_verification"
    assert summaries["PubChem"].status == "ok"
    assert summaries["BindingDB"].status == "ok"
    assert summaries["AlphaFold DB"].status == "ok"


def test_target_only_lookup_skips_chemical_sources_explicitly() -> None:
    client = FakePublicChemistryClient()
    request = ChemistryLookup(uniprot_id="P36888", limit_per_source=1)

    result = PublicChemistryEnricher(client=client).enrich(request)
    summaries = {summary.source: summary for summary in result.sources}

    assert summaries["ChEMBL"].status == "skipped"
    assert summaries["PubChem"].status == "skipped"
    assert summaries["BindingDB"].status == "ok"
    assert summaries["DrugCentral"].status == "not_configured"
    assert summaries["RCSB PDB"].status == "ok"
    assert summaries["AlphaFold DB"].status == "ok"
    assert len(client.calls) == 3


def test_drugcentral_uses_configured_local_tsv_and_records_file_provenance(tmp_path) -> None:
    dataset = tmp_path / "drug.target.interaction.tsv"
    dataset.write_text(
        "STRUCT_ID\tDRUG_NAME\tACCESSION\tGENE\tACTION_TYPE\tACT_TYPE\tACT_VALUE\tACT_UNIT\tSMILES\tINCHIKEY\n"
        "DC001\tquizartinib\tP36888\tFLT3\tINHIBITOR\tIC50\t4.2\tnM\t"
        "COC1=NC=CC(=C1)C(=O)N\tFAKEINCHIKEY123\n",
        encoding="utf-8",
    )
    client = FakePublicChemistryClient()
    enricher = PublicChemistryEnricher(
        client=client,
        drugcentral_dataset_path=dataset,
        drugcentral_dataset_version="test-snapshot",
    )

    result = enricher.enrich(_lookup())
    drugcentral = [item for item in result.evidence if item.source == "DrugCentral"]
    registry = {entry.source: entry for entry in enricher.registry()}

    assert len(drugcentral) == 1
    assert drugcentral[0].external_id == "DC001"
    assert drugcentral[0].metadata["provenance_class"] == "external_local_dataset"
    assert drugcentral[0].metadata["dataset_filename"] == dataset.name
    assert drugcentral[0].metadata["dataset_version"] == "test-snapshot"
    assert registry["DrugCentral"].access_mode == "local_dataset"
    assert registry["DrugCentral"].runtime_state == "ready"


def test_registry_does_not_claim_unconfigured_drugcentral_is_active() -> None:
    registry = {
        entry.source: entry
        for entry in PublicChemistryEnricher(client=FakePublicChemistryClient()).registry()
    }

    assert registry["DrugCentral"].runtime_state == "not_configured"
    assert registry["DrugCentral"].access_mode == "local_dataset"
    assert registry["BindingDB"].runtime_state == "ready"


def test_lookup_contract_rejects_empty_or_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ChemistryLookup()
    with pytest.raises(ValidationError):
        ChemistryLookup(candidate_name="quizartinib", secret_option=True)
