from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode
from drugpilot.storage import SQLiteRunStore


@pytest.mark.asyncio
async def test_source_api_reports_configuration_but_never_secrets_or_paths(tmp_path):
    drugcentral = tmp_path / "drugcentral.tsv"
    drugcentral.write_text("struct_id\tname\n1\tExample\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "source-api.sqlite3",
        public_evidence_enabled=True,
        biogrid_access_key="biogrid-secret-value",
        drugbank_api_key="drugbank-secret-value",
        drugcentral_dataset_path=drugcentral,
    )
    app = create_app(
        settings=settings,
        store=SQLiteRunStore(settings.database_path),
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/sources")
        response.raise_for_status()
        payload = response.json()

        health = await client.get("/api/health")
        health.raise_for_status()

    runtime = {item["source_id"]: item for item in payload["runtime"]}
    assert len(runtime) == 26
    assert len(payload["evaluation"]) == 3
    assert runtime["pubmed"]["adapter"] == "PublicBiologyRetriever"
    assert runtime["chembl"]["status"] == "connected_adapter"
    assert runtime["chembl"]["default_queried"] is True
    assert runtime["drugcentral"]["configuration_state"] == "enabled"
    assert runtime["drugcentral"]["status"] == "connected_adapter"
    assert runtime["biogrid"]["configuration_state"] == "credential_configured"
    assert runtime["biogrid"]["status"] == "available_not_default"
    assert runtime["biogrid"]["adapter"] == "PublicExpressionInteractionRetriever"
    assert runtime["drugbank"]["configuration_state"] == "credential_configured"
    assert runtime["drugbank"]["status"] == "available_not_default"
    assert runtime["drugbank"]["adapter"] == "DrugBankLicensedAdapter"
    health_payload = health.json()
    assert health_payload["source_registry"]["endpoint"] == "/api/sources"
    public_http = health_payload["public_http_transport"]
    assert public_http["certificate_verification"] is True
    assert public_http["trust_backend"] in {
        "system_truststore",
        "environment_ca",
        "certifi",
        "python_default",
    }
    assert public_http["proxy_mode"] in {"environment", "explicit", "disabled"}
    assert public_http["max_attempts"] == 2
    assert public_http["reachability_probed"] is False
    assert "recent_failures" not in public_http
    evidence_research = health_payload["evidence_research"]
    assert evidence_research["depth"] == "comprehensive"
    assert evidence_research["core_records_per_source"] == 8
    assert evidence_research["core_source_calls_per_target"] == 5
    assert evidence_research["core_elapsed_seconds_per_target"] == 60.0
    assert evidence_research["target_pubmed_records"] == 40
    assert evidence_research["context_targets_per_track"] == 2
    assert evidence_research["context_source_calls"] == 36
    assert evidence_research["context_elapsed_seconds"] == 90.0
    assert evidence_research["context_records_per_source"] == 5
    assert evidence_research["server_default"] == "comprehensive"
    assert evidence_research["available_profiles"] == [
        "focused",
        "standard",
        "comprehensive",
    ]
    assert evidence_research["resolved_profiles"]["standard"][
        "core_records_per_source"
    ] == 5
    assert evidence_research["resolved_profiles"]["standard"][
        "candidate_enrichment_per_route"
    ] == 3
    assert health_payload["dynamic_context_routing"] == {
        "enabled": True,
        "research_depth": "comprehensive",
        "profile_scope": "server_default_only",
        "targets_per_track": 2,
        "max_source_calls": 36,
        "max_elapsed_seconds": 90.0,
        "records_per_source": 5,
        "context_only": True,
    }

    serialized = json.dumps(payload)
    assert "biogrid-secret-value" not in serialized
    assert "drugbank-secret-value" not in serialized
    assert str(drugcentral) not in serialized
