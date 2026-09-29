from __future__ import annotations

from drugpilot.tools.source_catalog import (
    SourceConfiguration,
    source_status_inventory,
)


def _runtime_by_id(inventory: dict[str, object]) -> dict[str, dict[str, object]]:
    runtime = inventory["runtime"]
    assert isinstance(runtime, list)
    return {str(item["source_id"]): item for item in runtime}


def test_registry_covers_every_report_source_without_reachability_claims():
    inventory = source_status_inventory()
    runtime = _runtime_by_id(inventory)

    assert len(runtime) == 26
    assert len(inventory["evaluation"]) == 3
    assert inventory["summary"]["runtime_total"] == 26
    assert inventory["summary"]["evaluation_total"] == 3
    assert inventory["summary"]["network_reachability_checked"] is False
    assert all(item["reachability"] == "not_checked" for item in runtime.values())

    assert runtime["pubmed"]["status"] == "connected_adapter"
    assert runtime["pubmed"]["default_queried"] is True
    assert runtime["pharos"]["status"] == "available_not_default"
    assert runtime["pharos"]["default_queried"] is False
    assert (
        runtime["pharos"]["configuration_state"]
        == "explicit_human_priority_trigger"
    )
    assert runtime["pharos"]["adapter"] == "PublicBiologyRetriever"
    assert "ResearchBrief.human_priorities" in runtime["pharos"]["note"]
    assert runtime["biorxiv"]["status"] == "federated"
    assert runtime["biorxiv"]["default_queried"] is False
    assert runtime["chembl"]["status"] == "available_not_default"
    assert runtime["gtex"]["status"] == "available_not_default"
    assert runtime["gtex"]["adapter"] == "MA1PublicContextRetriever"
    assert runtime["human_cell_atlas"]["status"] == "available_not_default"
    assert runtime["human_cell_atlas"]["adapter"] == "PublicCellDataRetriever"
    assert runtime["archs4"]["status"] == "available_not_default"
    assert runtime["archs4"]["adapter"] == "PublicExpressionInteractionRetriever"
    assert runtime["cellxgene"]["status"] == "not_configured"
    assert runtime["cellxgene"]["adapter"] == "PublicCellDataRetriever"
    assert runtime["biogrid"]["adapter"] == "PublicExpressionInteractionRetriever"
    assert runtime["google_patents"]["adapter"] == "GooglePatentsBigQueryAdapter"
    assert runtime["drugbank"]["adapter"] == "DrugBankLicensedAdapter"
    assert runtime["drugcentral"]["status"] == "not_configured"
    assert {item["status"] for item in inventory["evaluation"]} == {
        "evaluation_only"
    }


def test_registry_distinguishes_disabled_missing_and_configured_sources():
    missing = source_status_inventory(
        SourceConfiguration(
            public_evidence_enabled=False,
            drugcentral_dataset_configured=True,
            drugcentral_dataset_available=False,
            biogrid_credential_configured=True,
        )
    )
    runtime = _runtime_by_id(missing)

    assert runtime["pubmed"]["status"] == "available_not_default"
    assert runtime["pubmed"]["configuration_state"] == "disabled_by_policy"
    assert runtime["drugcentral"]["status"] == "not_configured"
    assert runtime["drugcentral"]["configuration_state"] == "dataset_missing"
    assert runtime["biogrid"]["status"] == "available_not_default"
    assert runtime["biogrid"]["configured"] is True

    ready = source_status_inventory(
        SourceConfiguration(
            drugcentral_dataset_configured=True,
            drugcentral_dataset_available=True,
        )
    )
    assert _runtime_by_id(ready)["drugcentral"]["status"] == (
        "available_not_default"
    )
