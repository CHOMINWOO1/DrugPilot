from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import (
    ProjectConstraints,
    ReadinessStatus,
    ResearchBrief,
    RunMode,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator, build_ma2_gateway
from drugpilot.storage import SQLiteRunStore
from drugpilot.tools.live_repurposing_gateway import (
    PAIRED_V4_REQUIRED_POLICY,
    PublicRepurposingGateway,
)


class _FakeADMET:
    source_version = "admet-test-v1"
    fixture = False

    def __init__(self, *, python_executable=None) -> None:
        self.python_executable = python_executable
        self.close_calls = 0

    def readiness(self) -> dict[str, object]:
        return {
            "tool": "ADMET-AI",
            "operation": "predict_six_endpoint_panel",
            "status": "completed",
            "backend": "test-live-admet",
            "version": self.source_version,
            "fixture": False,
            "required": True,
            "reason": None,
            "provenance": {},
        }

    def close(self) -> None:
        self.close_calls += 1


class _FakeDocking:
    source_version = "vina-test-v1"
    fixture = False

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def readiness(self, target=None) -> dict[str, object]:
        return {
            "tool": "AutoDock Vina",
            "operation": "score_binding_pose",
            "status": "completed",
            "backend": "test-live-docking",
            "version": self.source_version,
            "fixture": False,
            "required": False,
            "reason": None,
            "provenance": {"target_specific_check": target is not None},
        }


class _FakeScientificGateway:
    backend_kind = "live"
    fixture = False

    def __init__(self, *, docking, admet, **kwargs) -> None:
        self.docking = docking
        self.admet = admet
        self.kwargs = kwargs

    @property
    def source_version(self) -> str:
        return f"scientific-test;{self.docking.source_version};{self.admet.source_version}"

    def readiness(self, target=None) -> dict[str, object]:
        return {
            "backend_kind": "live",
            "fixture": False,
            "status": "completed",
            "source_version": self.source_version,
            "tools": [
                {
                    "tool": "ChEMBL",
                    "operation": "retrieve_measured_target_ligands",
                    "status": "completed",
                    "backend": "official_live_rest_api",
                    "version": "test",
                    "fixture": False,
                    "required": True,
                    "reason": None,
                },
                {
                    "tool": "RDKit BRICS",
                    "operation": "generate_candidates_from_measured_ligands",
                    "status": "completed",
                    "backend": "isolated_local_process",
                    "version": "test",
                    "fixture": False,
                    "required": True,
                    "reason": None,
                },
                self.docking.readiness(target),
                self.admet.readiness(),
            ],
        }


def _live_settings(tmp_path: Path, **overrides: object) -> Settings:
    dataset = tmp_path / "drugcentral.tsv"
    dataset.write_text("STRUCT_ID\tDRUG_NAME\tGENE\n1\tExample\tGENE\n", encoding="utf-8")
    profile = tmp_path / "profiles.json"
    profile.write_text("{}", encoding="utf-8")
    values: dict[str, object] = {
        "_env_file": None,
        "mode": RunMode.MOCK,
        "database_path": tmp_path / "live.sqlite3",
        "ma2_backend_mode": "live",
        "ma2_live_scientific_tools_enabled": True,
        "drugcentral_dataset_path": dataset,
        "docking_profile_registry_path": profile,
        "public_evidence_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def test_blank_optional_scientific_paths_normalize_to_unconfigured(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "blank.sqlite3",
        drugcentral_dataset_path="   ",
        vina_binary="",
        docking_profile_registry_path="  ",
        admet_python_executable="\t",
    )

    assert settings.drugcentral_dataset_path is None
    assert settings.vina_binary is None
    assert settings.docking_profile_registry_path is None
    assert settings.admet_python_executable is None


def test_opt_in_builder_composes_public_gateway_with_live_scientific_delegate(
    tmp_path: Path,
) -> None:
    settings = _live_settings(tmp_path)

    gateway = build_ma2_gateway(
        settings,
        scientific_gateway_factory=_FakeScientificGateway,
        docking_factory=_FakeDocking,
        admet_factory=_FakeADMET,
        profile_registry_factory=lambda path: {"registry": path.name},
    )

    assert isinstance(gateway, PublicRepurposingGateway)
    assert gateway.fixture is False
    assert isinstance(gateway.delegate, _FakeScientificGateway)
    assert gateway.build_status["scientific_delegate_configured"] is True
    assert gateway.delegate.kwargs["python_executable"] is None
    assert gateway.delegate.kwargs["timeout_seconds"] == 30.0
    assert gateway.delegate.kwargs["operation_timeout_seconds"] == 120.0
    assert gateway.delegate.kwargs["max_http_attempts"] == 2
    assert gateway.delegate.kwargs["retry_backoff_seconds"] == 0.25
    assert gateway.delegate.docking.kwargs["allow_experimental_profiles"] is False
    readiness = gateway.readiness(None)
    assert all(item.fixture is False for item in readiness.capabilities)
    assert all(
        item.status is ReadinessStatus.READY
        for item in readiness.capabilities
        if item.required
    )


def test_live_builder_passes_optional_paired_v4_policy_only_when_requested(
    tmp_path: Path,
) -> None:
    settings = _live_settings(tmp_path)

    default_gateway = build_ma2_gateway(
        settings,
        scientific_gateway_factory=_FakeScientificGateway,
        docking_factory=_FakeDocking,
        admet_factory=_FakeADMET,
        profile_registry_factory=lambda path: {"registry": path.name},
    )
    paired_gateway = build_ma2_gateway(
        settings,
        repurposing_evidence_policy=PAIRED_V4_REQUIRED_POLICY,
        scientific_gateway_factory=_FakeScientificGateway,
        docking_factory=_FakeDocking,
        admet_factory=_FakeADMET,
        profile_registry_factory=lambda path: {"registry": path.name},
    )

    assert default_gateway.repurposing_evidence_policy is None
    assert default_gateway.build_status["repurposing_evidence_policy"] is None
    assert (
        paired_gateway.repurposing_evidence_policy
        == PAIRED_V4_REQUIRED_POLICY
    )
    assert (
        paired_gateway.build_status["repurposing_evidence_policy"]
        == PAIRED_V4_REQUIRED_POLICY
    )


def test_run_local_gateway_activates_policy_by_exact_priority_membership(
    tmp_path: Path,
) -> None:
    settings = _live_settings(
        tmp_path,
        ma2_live_scientific_tools_enabled=False,
        public_candidate_enrichment_enabled=False,
    )
    orchestrator = DrugPilotOrchestrator(settings=settings)
    constraints = ProjectConstraints(candidates_per_route=3)
    exact = ResearchBrief(
        brief_id="brief_exact_paired_v4",
        research_goal="Evaluate an auditable repurposing cohort.",
        disease="Generic disease",
        constraints=constraints,
        human_priorities=[PAIRED_V4_REQUIRED_POLICY],
    )
    near = exact.model_copy(
        update={
            "brief_id": "brief_near_paired_v4",
            "human_priorities": [f"{PAIRED_V4_REQUIRED_POLICY}_extra"],
        },
        deep=True,
    )
    exact_ma2 = None
    near_ma2 = None
    try:
        assert orchestrator.ma2.gateway.repurposing_evidence_policy is None
        exact_ma2 = orchestrator._ma2_for_brief(exact)
        near_ma2 = orchestrator._ma2_for_brief(near)

        assert (
            exact_ma2.gateway.repurposing_evidence_policy
            == PAIRED_V4_REQUIRED_POLICY
        )
        assert near_ma2.gateway.repurposing_evidence_policy is None
        assert orchestrator.ma2.gateway.repurposing_evidence_policy is None
    finally:
        if exact_ma2 is not None:
            exact_ma2.close()
        if near_ma2 is not None:
            near_ma2.close()
        orchestrator.close()


def test_chembl_operation_budget_cannot_be_shorter_than_one_request(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="operation_timeout_seconds"):
        _live_settings(
            tmp_path,
            ma2_chembl_timeout_seconds=30,
            ma2_chembl_operation_timeout_seconds=20,
        )


def test_live_builder_accepts_extended_chembl_operation_budget(
    tmp_path: Path,
) -> None:
    settings = _live_settings(
        tmp_path,
        ma2_chembl_operation_timeout_seconds=600,
    )
    gateway = build_ma2_gateway(
        settings,
        scientific_gateway_factory=_FakeScientificGateway,
        docking_factory=_FakeDocking,
        admet_factory=_FakeADMET,
        profile_registry_factory=lambda path: {"registry": path.name},
    )
    assert gateway.delegate.kwargs["operation_timeout_seconds"] == 600.0
    assert gateway.fixture is False


def test_missing_docking_profile_is_typed_optional_unavailable_not_fixture(
    tmp_path: Path,
) -> None:
    settings = _live_settings(tmp_path, docking_profile_registry_path=None)
    gateway = build_ma2_gateway(
        settings,
        scientific_gateway_factory=_FakeScientificGateway,
        admet_factory=_FakeADMET,
    )

    readiness = gateway.readiness(None)
    docking = next(
        item for item in readiness.capabilities if item.tool_name == "docking"
    )
    assert docking.required is False
    assert docking.fixture is False
    assert docking.status is ReadinessStatus.NOT_CONFIGURED
    assert gateway.build_status["component_errors"] == [
        {
            "component": "docking",
            "reason": "DOCKING_PROFILE_REGISTRY_UNAVAILABLE",
        }
    ]


def test_ma2_close_releases_live_admet_worker_once(tmp_path: Path) -> None:
    gateway = build_ma2_gateway(
        _live_settings(tmp_path),
        scientific_gateway_factory=_FakeScientificGateway,
        docking_factory=_FakeDocking,
        admet_factory=_FakeADMET,
        profile_registry_factory=lambda path: {"registry": path.name},
    )
    admet = gateway.delegate.admet

    MA2Agent(gateway=gateway, max_refinement_iterations=0).close()

    assert admet.close_calls == 1


def test_experimental_docking_override_rejected_for_production_variant(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="test/demo/development"):
        _live_settings(
            tmp_path,
            build_variant="production",
            ma2_live_allow_experimental_docking_profiles=True,
        )


@pytest.mark.asyncio
async def test_health_reports_live_ma2_without_unconditional_fixture_claim(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "health.sqlite3",
        ma2_backend_mode="live",
        ma2_live_scientific_tools_enabled=False,
        drugcentral_dataset_path=tmp_path / "missing.tsv",
        public_evidence_enabled=False,
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
    assert payload["ma2"]["configured_mode"] == "live"
    assert payload["ma2"]["fixture"] is False
    assert payload["ma2"]["fail_closed"] is True
    assert payload["ma2"]["fixture_fallback_authorized"] is False
    assert payload["ma2"]["readiness"] == "unavailable"
    assert payload["ma2"]["max_refinement_iterations"] == 0
    assert "fixture chemistry computation" not in payload["data_backends"]
    assert "fixture computation" not in payload["candidate_backend"]
    assert "live candidate execution" in payload["data_backend"]
