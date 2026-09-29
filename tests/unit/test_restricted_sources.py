from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping

import httpx
import pytest
from pydantic import ValidationError

from drugpilot.tools.restricted_sources import (
    BigQueryExecution,
    DrugBankLicensedAdapter,
    DrugCentralRestrictedAdapter,
    GOOGLE_PATENTS_TABLE,
    GooglePatentsBigQueryAdapter,
    RestrictedSourceQuery,
)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        url: str = "https://example.test/read-only",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("GET", url)
        self.response = httpx.Response(status_code, request=self.request)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=self.response,
            )

    def json(self) -> Any:
        return self._payload


class FakeDrugBankClient:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected DrugBank HTTP call")
        return self.responses.popleft()


class FakeBigQueryExecutor:
    def __init__(
        self,
        execution: BigQueryExecution | None = None,
        error: Exception | None = None,
    ) -> None:
        self.execution = execution or BigQueryExecution()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        maximum_bytes_billed: int,
        location: str,
    ) -> BigQueryExecution:
        self.calls.append(
            {
                "sql": sql,
                "parameters": dict(parameters),
                "maximum_bytes_billed": maximum_bytes_billed,
                "location": location,
            }
        )
        if self.error:
            raise self.error
        return self.execution


def test_query_contract_requires_inputs_for_each_operation() -> None:
    with pytest.raises(ValidationError):
        RestrictedSourceQuery(operation="repurposing")
    with pytest.raises(ValidationError):
        RestrictedSourceQuery(operation="identity_regulatory")
    with pytest.raises(ValidationError):
        RestrictedSourceQuery(operation="contraindication")
    with pytest.raises(ValidationError):
        RestrictedSourceQuery(operation="patent_novelty")

    query = RestrictedSourceQuery(
        operation="patent_novelty",
        novelty_terms=("  FLT3 inhibitor ", "flt3 INHIBITOR", "AML"),
    )
    assert query.novelty_terms == ("FLT3 inhibitor", "AML")


def test_missing_configuration_never_calls_or_fabricates_live_evidence() -> None:
    drugbank_client = FakeDrugBankClient()
    bigquery = FakeBigQueryExecutor()
    cases = (
        (
            DrugCentralRestrictedAdapter(),
            RestrictedSourceQuery(
                operation="identity_regulatory", candidate_name="quizartinib"
            ),
            "local_dataset_not_configured",
        ),
        (
            DrugBankLicensedAdapter(client=drugbank_client),
            RestrictedSourceQuery(
                operation="identity_regulatory", candidate_name="quizartinib"
            ),
            "licensed_api_key_not_configured",
        ),
        (
            GooglePatentsBigQueryAdapter(executor=bigquery),
            RestrictedSourceQuery(
                operation="patent_novelty", novelty_terms=("FLT3",)
            ),
            "google_cloud_project_not_configured",
        ),
    )

    for adapter, request, error_type in cases:
        result = adapter.query(request)
        assert result.summary.status == "not_configured"
        assert result.summary.queried is False
        assert result.summary.error_type == error_type
        assert result.evidence == []

    assert drugbank_client.calls == []
    assert bigquery.calls == []


def test_drugcentral_reuses_local_interaction_adapter_and_caches(tmp_path: Path) -> None:
    dataset = tmp_path / "drug.target.interaction.tsv"
    dataset.write_text(
        "STRUCT_ID\tDRUG_NAME\tACCESSION\tGENE\tACTION_TYPE\tACT_TYPE\tACT_VALUE\tACT_UNIT\n"
        "DC001\tquizartinib\tP36888\tFLT3\tINHIBITOR\tIC50\t4.2\tnM\n",
        encoding="utf-8",
    )
    adapter = DrugCentralRestrictedAdapter(
        dataset_path=dataset,
        dataset_version="official-test-snapshot",
    )
    request = RestrictedSourceQuery(
        operation="repurposing", target_uniprot="P36888", limit=3
    )

    first = adapter.query(request)
    second = adapter.query(request)

    assert first.summary.status == "ok"
    assert first.summary.cache_hit is False
    assert first.evidence[0].metadata["local_adapter_reused"] is True
    assert first.evidence[0].metadata["provenance_class"] == "external_local_dataset"
    assert first.evidence[0].metadata["live"] is False
    assert second.summary.cache_hit is True
    assert second.evidence[0].metadata["provenance_class"] == "external_cache"
    assert second.evidence[0].metadata["origin_provenance_class"] == (
        "external_local_dataset"
    )
    assert second.evidence[0].metadata["live"] is False


def test_drugcentral_reads_explicit_regulatory_and_contraindication_fields(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "drugcentral.regulatory.tsv"
    dataset.write_text(
        "STRUCT_ID\tDRUG_NAME\tINCHIKEY\tAPPROVAL_STATUS\tWITHDRAWN\tCONTRAINDICATION\n"
        "DC009\tunsafe-drug\tTESTKEY\tWithdrawn\tyes\tSevere liver disease\n",
        encoding="utf-8",
    )
    adapter = DrugCentralRestrictedAdapter(
        dataset_path=dataset, dataset_version="licensed-export-v1"
    )

    identity = adapter.query(
        RestrictedSourceQuery(
            operation="identity_regulatory", candidate_name="unsafe-drug"
        )
    )
    contraindication = adapter.query(
        RestrictedSourceQuery(
            operation="contraindication", drug_id="DC009", condition="liver"
        )
    )
    unrelated_condition = adapter.query(
        RestrictedSourceQuery(
            operation="contraindication", drug_id="DC009", condition="kidney"
        )
    )

    assert identity.summary.status == "ok"
    assert identity.evidence[0].supports == "counter"
    assert identity.evidence[0].metadata["withdrawn"] is True
    assert identity.evidence[0].metadata["regulatory_status"] == "Withdrawn"
    assert contraindication.summary.status == "ok"
    assert contraindication.evidence[0].metadata["contraindication_present"] is True
    assert "Severe liver disease" not in str(contraindication.model_dump())
    assert contraindication.evidence[0].metadata["absence_is_not_negative_evidence"]
    assert unrelated_condition.summary.status == "empty"
    assert unrelated_condition.evidence == []


def test_drugcentral_does_not_treat_interaction_tsv_as_safety_clearance(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interaction-only.tsv"
    dataset.write_text(
        "STRUCT_ID\tDRUG_NAME\tACCESSION\nDC001\tquizartinib\tP36888\n",
        encoding="utf-8",
    )
    result = DrugCentralRestrictedAdapter(dataset_path=dataset).query(
        RestrictedSourceQuery(
            operation="contraindication", candidate_name="quizartinib"
        )
    )

    assert result.summary.status == "not_configured"
    assert result.summary.queried is False
    assert result.summary.error_type == (
        "drugcentral_contraindication_fields_not_present_in_dataset"
    )
    assert result.evidence == []


def test_drugcentral_identity_uses_available_stable_fields_by_precedence(
    tmp_path: Path,
) -> None:
    cases = (
        (
            "id-only.tsv",
            "STRUCT_ID\tAPPROVAL_STATUS\n5306\tApproved\n",
            "5306",
        ),
        (
            "inchikey-only.tsv",
            "INCHIKEY\tAPPROVAL_STATUS\nEXACT-KEY\tApproved\n",
            "row-2",
        ),
        (
            "name-only.tsv",
            "DRUG_NAME\tAPPROVAL_STATUS\nCandidate　Alpha\tApproved\n",
            "row-2",
        ),
    )

    for filename, contents, expected_external_id in cases:
        dataset = tmp_path / filename
        dataset.write_text(contents, encoding="utf-8")
        result = DrugCentralRestrictedAdapter(dataset_path=dataset).query(
            RestrictedSourceQuery(
                operation="identity_regulatory",
                candidate_name="Candidate   Alpha",
                drug_id="5306",
                inchikey="EXACT-KEY",
            )
        )

        assert result.summary.status == "ok"
        assert [item.external_id for item in result.evidence] == [
            expected_external_id
        ]


def test_drugcentral_stable_id_conflict_cannot_fall_back_to_exact_name(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "conflicting-identities.tsv"
    dataset.write_text(
        "STRUCT_ID\tDRUG_NAME\tINCHIKEY\tAPPROVAL_STATUS\n"
        "9999\tCandidate Alpha\tEXACT-KEY\tApproved\n"
        "5306\tCandidate Alpha\tDIFFERENT-KEY\tApproved\n",
        encoding="utf-8",
    )

    result = DrugCentralRestrictedAdapter(dataset_path=dataset).query(
        RestrictedSourceQuery(
            operation="identity_regulatory",
            candidate_name="Candidate Alpha",
            drug_id="5306",
            inchikey="EXACT-KEY",
        )
    )

    assert result.summary.status == "empty"
    assert result.summary.queried is True
    assert result.evidence == []


def test_drugbank_identity_records_withdrawal_and_never_exposes_key() -> None:
    secret = "licensed-secret-key"
    client = FakeDrugBankClient(
        FakeResponse(
            {
                "drugbank_id": "DB00001",
                "name": "Examplemab",
                "groups": ["Approved", "Withdrawn"],
                "availability_by_region": [
                    {
                        "region": "us",
                        "max_phase": 4,
                        "marketed_prescription": False,
                        "pre_market_cancelled": False,
                        "post_market_cancelled": True,
                    }
                ],
                "identifiers": {"inchikey": "EXAMPLE-KEY"},
                "pharmacology": {"indication_description": "licensed narrative"},
            }
        )
    )
    adapter = DrugBankLicensedAdapter(api_key=secret, client=client)
    request = RestrictedSourceQuery(
        operation="identity_regulatory", drug_id="DB00001"
    )

    first = adapter.query(request)
    second = adapter.query(request)

    assert first.summary.status == "ok"
    assert first.evidence[0].metadata["withdrawn"] is True
    assert first.evidence[0].metadata["cancelled_regions"] == ("us",)
    assert first.evidence[0].supports == "counter"
    assert client.calls[0]["headers"]["Authorization"] == secret
    assert secret not in str(first.model_dump())
    assert "licensed narrative" not in str(first.model_dump())
    assert len(client.calls) == 1
    assert second.summary.cache_hit is True
    assert second.evidence[0].metadata["live"] is False


def test_drugbank_resolves_exact_name_then_queries_contraindications() -> None:
    client = FakeDrugBankClient(
        FakeResponse(
            [
                {
                    "drugbank_id": "DB00316",
                    "name": "Acetaminophen",
                    "synonyms": ["Paracetamol"],
                },
                {"drugbank_id": "DB99999", "name": "Not an exact match"},
            ]
        ),
        FakeResponse(
            [
                {
                    "regions": "US",
                    "route": [],
                    "dose_form": [],
                    "hypersensitivity": [],
                    "patient_conditions": [
                        {
                            "name": "Severe, active Liver Disease",
                            "drugbank_id": "DBCOND0112340",
                        }
                    ],
                }
            ]
        ),
    )
    result = DrugBankLicensedAdapter(api_key="test-key", client=client).query(
        RestrictedSourceQuery(
            operation="contraindication",
            candidate_name="Paracetamol",
            condition="liver",
        )
    )

    assert result.summary.status == "ok"
    assert result.evidence[0].metadata["patient_conditions"] == (
        "Severe, active Liver Disease",
    )
    assert result.evidence[0].supports == "counter"
    assert len(client.calls) == 2
    assert client.calls[0]["params"]["q"] == 'name:"Paracetamol"'
    assert client.calls[1]["url"].endswith("/DB00316/contraindications")


def test_drugbank_repurposing_filters_non_target_bonds() -> None:
    client = FakeDrugBankClient(
        FakeResponse(
            [
                {
                    "type": "Target",
                    "drug": {"drugbank_id": "DB01234", "name": "Candidate A"},
                    "bio_entity": {
                        "bio_entity_id": "BE0001",
                        "name": "FLT3",
                    },
                    "known_action": "yes",
                    "actions": ["inhibitor"],
                },
                {
                    "type": "Carrier",
                    "drug": {"drugbank_id": "DB09999", "name": "Carrier ligand"},
                    "bio_entity": {"bio_entity_id": "BE0002", "name": "Albumin"},
                },
            ]
        )
    )
    result = DrugBankLicensedAdapter(api_key="test-key", client=client).query(
        RestrictedSourceQuery(operation="repurposing", target_uniprot="P36888")
    )

    assert [item.external_id for item in result.evidence] == ["DB01234"]
    assert result.evidence[0].metadata["clinical_efficacy_inferred"] is False
    assert result.evidence[0].metadata["regulatory_status_checked"] is False


def test_drugbank_http_failure_is_error_not_empty_or_not_configured() -> None:
    client = FakeDrugBankClient(FakeResponse({}, status_code=401))
    result = DrugBankLicensedAdapter(api_key="bad-key", client=client).query(
        RestrictedSourceQuery(
            operation="identity_regulatory", drug_id="DB00001"
        )
    )

    assert result.summary.status == "error"
    assert result.summary.queried is True
    assert result.summary.error_type == "upstream_http_401"
    assert result.evidence == []


def test_google_patents_uses_parameters_budget_and_truthful_novelty_scope() -> None:
    malicious_term = "kinase' OR 1=1 --"
    executor = FakeBigQueryExecutor(
        BigQueryExecution(
            rows=[
                {
                    "publication_number": "US-1234567-A1",
                    "title": "Selective FLT3 inhibitors",
                    "url": "https://patents.google.com/patent/US1234567A1/en",
                    "country": "United States",
                    "publication_description": "Application",
                    "top_terms": ["FLT3", "kinase inhibitor"],
                }
            ],
            job_id="job-test-1",
            total_bytes_processed=123_456,
        )
    )
    adapter = GooglePatentsBigQueryAdapter(
        project="billing-project",
        executor=executor,
        maximum_bytes_billed=1_000_000,
    )
    request = RestrictedSourceQuery(
        operation="patent_novelty",
        novelty_terms=("FLT3", malicious_term),
        limit=5,
    )

    first = adapter.query(request)
    second = adapter.query(request)

    call = executor.calls[0]
    assert GOOGLE_PATENTS_TABLE in call["sql"]
    assert malicious_term not in call["sql"]
    assert call["parameters"] == {
        "terms": ("FLT3", malicious_term),
        "limit": 5,
    }
    assert call["maximum_bytes_billed"] == 1_000_000
    assert call["location"] == "US"
    assert first.summary.status == "ok"
    assert first.evidence[0].metadata["provenance_class"] == "external_live"
    assert first.evidence[0].metadata["potential_prior_art_context_only"] is True
    assert first.evidence[0].metadata["legal_novelty_determination"] is False
    assert first.evidence[0].metadata["total_bytes_processed"] == 123_456
    assert len(executor.calls) == 1
    assert second.summary.cache_hit is True
    assert second.evidence[0].metadata["provenance_class"] == "external_cache"
    assert second.evidence[0].metadata["live"] is False


def test_google_patents_provider_cache_is_reported_as_cache() -> None:
    executor = FakeBigQueryExecutor(
        BigQueryExecution(
            rows=[
                {
                    "publication_number": "WO-2024000001-A1",
                    "title": "A therapeutic composition",
                }
            ],
            provider_cache_hit=True,
        )
    )
    result = GooglePatentsBigQueryAdapter(
        project="billing-project", executor=executor
    ).query(
        RestrictedSourceQuery(
            operation="patent_novelty", novelty_terms=("therapeutic",)
        )
    )

    assert result.summary.cache_hit is True
    assert result.evidence[0].metadata["provenance_class"] == "external_cache"
    assert result.evidence[0].metadata["live"] is False


def test_google_patents_missing_adc_is_not_configured_not_live() -> None:
    class DefaultCredentialsError(Exception):
        pass

    executor = FakeBigQueryExecutor(error=DefaultCredentialsError("no ADC"))
    result = GooglePatentsBigQueryAdapter(
        project="billing-project", executor=executor
    ).query(
        RestrictedSourceQuery(operation="patent_novelty", novelty_terms=("FLT3",))
    )

    assert result.summary.status == "not_configured"
    assert result.summary.queried is False
    assert result.summary.error_type == (
        "google_application_default_credentials_not_configured"
    )
    assert result.evidence == []
