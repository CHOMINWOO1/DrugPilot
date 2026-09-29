from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode
from drugpilot.storage import SQLiteRunStore
from tests.unit.test_scientific_tool_health import _ready_report


def test_new_optional_receptor_paths_normalize_to_none(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "settings.sqlite3",
        receptor_python_executable=" ",
        receptor_rdkit_root="",
        receptor_openmm_plugin_dir="\t",
        receptor_meeko_receptor_executable=" ",
        receptor_meeko_ligand_executable="",
        receptor_p2rank_executable=" ",
        receptor_p2rank_jar_path="",
        receptor_java_executable="\t",
    )
    assert settings.receptor_builder_enabled is False
    assert settings.receptor_python_executable is None
    assert settings.receptor_rdkit_root is None
    assert settings.receptor_openmm_plugin_dir is None
    assert settings.receptor_meeko_receptor_executable is None
    assert settings.receptor_meeko_ligand_executable is None
    assert settings.receptor_p2rank_executable is None
    assert settings.receptor_p2rank_jar_path is None
    assert settings.receptor_java_executable is None


@pytest.mark.asyncio
async def test_api_separates_data_sources_tools_and_profile_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _ = _ready_report(tmp_path)
    monkeypatch.setattr(
        "drugpilot.api.app.probe_scientific_tools",
        lambda **kwargs: report,
    )
    registry = tmp_path / "experimental-profiles.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profiles": [
                    {
                        "profile_id": "experimental",
                        "validated": True,
                        "profile_status": "experimental_smoke_only",
                        "production_ready": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "health.sqlite3",
        ma2_backend_mode="live",
        ma2_live_scientific_tools_enabled=False,
        receptor_builder_enabled=True,
        docking_profile_registry_path=registry,
        public_evidence_enabled=True,
        public_candidate_enrichment_enabled=True,
        public_candidate_enrichment_per_route=1,
        mock_step_delay_ms=0,
    )
    app = create_app(settings=settings, store=SQLiteRunStore(settings.database_path))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/health")
    response.raise_for_status()
    payload = response.json()

    assert "ChEMBL" in payload["data_backends"]
    assert "RDKit BRICS" not in payload["data_backends"]
    assert "ADMET-AI" not in payload["data_backends"]
    assert "AutoDock Vina" not in payload["data_backends"]
    assert "fixture chemistry computation" not in payload["data_backends"]
    assert payload["scientific_tools"]["local_paths_exposed"] is False
    assert payload["receptor_profiles"]["profile_count"] == 1
    assert payload["receptor_profiles"]["accepted_profile_count"] == 0
    assert payload["receptor_profiles"]["builder_ready"] is True
    assert payload["receptor_profiles"]["production_ready"] is False
