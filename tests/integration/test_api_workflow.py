from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode
from drugpilot.storage import SQLiteRunStore


async def wait_for_stage(client: AsyncClient, run_id: str, stages: set[str], timeout: float = 5) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        snapshot = response.json()
        if snapshot["stage"] in stages:
            return snapshot
        await asyncio.sleep(0.02)
    pytest.fail(f"run {run_id} did not reach {stages}")


@pytest.fixture
def app(tmp_path):
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "integration.sqlite3",
        mock_step_delay_ms=0,
        candidates_per_route=3,
        max_ma1_iterations=2,
        max_rework_loops=2,
        public_evidence_enabled=False,
        demo_fixture_fallback_enabled=True,
        ma2_backend_mode="fixture",
        ma2_portfolio_enabled=False,
    )
    return create_app(settings=settings, store=SQLiteRunStore(settings.database_path))


@pytest.mark.asyncio
async def test_normal_web_workflow_and_human_gate(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "mock"
        assert health.json()["orchestrator_llm"] is False
        assert health.json()["dynamic_target_discovery"][
            "supplementary_evidence_retry_budget"
        ] == 1

        created = await client.post(
            "/api/runs",
            json={
                "research_goal": "Identify auditable FLT3 inhibitors through repurposing and de novo routes.",
                "disease": "Acute myeloid leukemia",
                "research_depth": "standard",
                "target_cell_type": "MOLM-13",
                "intervention_direction": "inhibit",
                "candidates_per_route": 3,
                "max_ma1_iterations": 2,
                "fault_injection": "none",
            },
        )
        assert created.status_code == 202, created.text
        assert created.json()["brief"]["research_depth"] == "standard"
        run_id = created.json()["run_id"]
        snapshot = await wait_for_stage(client, run_id, {"human_gate", "failed"})
        assert snapshot["stage"] == "human_gate", snapshot.get("error")
        assert snapshot["g0_verdict"]["verdict"] == "PASS"
        assert snapshot["g1_verdict"]["verdict"] == "PASS"
        assert snapshot["plan_spec"]["research_depth"] == "standard"
        assert snapshot["plan_spec"]["source_coverage"]["budget"][
            "research_depth"
        ] == "standard"
        assert snapshot["hypothesis_packet"]["metadata"][
            "research_depth_contract"
        ]["depth"] == "standard"
        assert len(snapshot["candidate_packet"]["candidates"]) == 3
        assert snapshot["candidate_packet"]["metadata"]["route_decision"][
            "selected_routes"
        ] == ["repurposing"]

        selected = [snapshot["candidate_packet"]["candidates"][0]["candidate_id"]]
        decided = await client.post(
            f"/api/runs/{run_id}/human-decision",
            json={
                "decision": "APPROVE",
                "researcher": "Integration Tester",
                "rationale": "All automated gates passed; approve one candidate for wet-lab review.",
                "selected_candidate_ids": selected,
            },
        )
        assert decided.status_code == 200, decided.text
        assert decided.json()["stage"] == "completed"
        reloaded = await client.get(f"/api/runs/{run_id}")
        assert reloaded.status_code == 200
        assert reloaded.json()["human_decision"]["decision"] == "APPROVE"
        assert reloaded.json()["human_decision"]["selected_candidate_ids"] == selected
        assert reloaded.json()["human_decision"]["researcher"] == "Integration Tester"
        summaries = await client.get("/api/run-summaries")
        assert summaries.status_code == 200
        listed = next(
            item for item in summaries.json() if item["run_id"] == run_id
        )
        assert listed["stage"] == "completed"
        assert listed["brief"]["disease"] == "Acute myeloid leukemia"
        assert listed["human_decision"] == {"decision": "APPROVE"}
        assert listed["list_summary"] is True
        assert "candidate_packet" not in listed
        assert "events" not in listed
        assert "token_usage" not in listed
        manifest = await client.get(f"/api/runs/{run_id}/artifacts")
        assert manifest.json()["ledger_valid"] is True
        assert any(
            artifact["artifact_type"] == "HumanGateDecision"
            for artifact in manifest.json()["artifacts"]
        )


@pytest.mark.asyncio
async def test_portfolio_enabled_api_attempts_every_g0_eligible_target(tmp_path):
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "portfolio-integration.sqlite3",
        mock_step_delay_ms=0,
        candidates_per_route=2,
        max_ma1_iterations=2,
        max_rework_loops=2,
        public_evidence_enabled=False,
        demo_fixture_fallback_enabled=True,
        ma2_portfolio_enabled=True,
    )
    portfolio_app = create_app(
        settings=settings,
        store=SQLiteRunStore(settings.database_path),
    )
    transport = ASGITransport(app=portfolio_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/runs",
            json={
                "research_goal": "Evaluate all evidence-qualified AML targets before ranking candidates.",
                "disease": "Acute myeloid leukemia",
                "target_cell_type": "MOLM-13",
                "intervention_direction": "inhibit",
                "candidates_per_route": 2,
                "max_ma1_iterations": 2,
            },
        )
        assert created.status_code == 202, created.text
        snapshot = await wait_for_stage(
            client,
            created.json()["run_id"],
            {"human_gate", "abstained", "blocked", "failed"},
        )
        assert snapshot["stage"] == "human_gate", snapshot.get("error")
        packet = snapshot["candidate_packet"]
        assert packet["portfolio_enabled"] is True
        assert packet["eligible_target_count"] >= 2
        assert packet["attempted_target_count"] == packet["eligible_target_count"]
        assert packet["completed_target_count"] == packet["eligible_target_count"]
        assert len(packet["target_executions"]) == packet["g0_approved_target_count"]
        assert all(
            item["attempted"]
            for item in packet["target_executions"]
            if item["execution_eligible"]
        )
        assert snapshot["g1_verdict"]["verdict"] == "PASS"
        primary_route = packet["metadata"]["route_decision"]["selected_route"]
        alternative = next(
            (
                candidate
                for candidate in packet["candidates"]
                if candidate["status"] != "excluded"
                and candidate["route"] != primary_route
            ),
            None,
        )
        assert alternative is not None, "fixture portfolio should exercise both routes"
        decided = await client.post(
            f"/api/runs/{snapshot['run_id']}/human-decision",
            json={
                "decision": "APPROVE",
                "researcher": "Portfolio Integration Tester",
                "rationale": "Approve an audited candidate from a non-primary target route.",
                "selected_candidate_ids": [alternative["candidate_id"]],
            },
        )
        assert decided.status_code == 200, decided.text
        assert decided.json()["stage"] == "completed"


@pytest.mark.asyncio
async def test_human_approval_requires_nonexcluded_candidates(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/runs",
            json={
                "research_goal": "Identify auditable FLT3 inhibitors for approval validation.",
                "disease": "Acute myeloid leukemia",
                "candidates_per_route": 3,
            },
        )
        run_id = created.json()["run_id"]
        snapshot = await wait_for_stage(client, run_id, {"human_gate", "failed"})
        assert snapshot["stage"] == "human_gate"
        invalid = await client.post(
            f"/api/runs/{run_id}/human-decision",
            json={
                "decision": "APPROVE",
                "researcher": "Integration Tester",
                "rationale": "Deliberately invalid empty selection.",
                "selected_candidate_ids": [],
            },
        )
        assert invalid.status_code == 409


@pytest.mark.asyncio
async def test_api_persists_forbidden_drug_policy_and_human_gate_rejects_match(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/runs",
            json={
                "research_goal": "Identify FLT3 candidates while enforcing drug exclusions.",
                "disease": "Acute myeloid leukemia",
                "candidates_per_route": 3,
                "forbidden_drugs": ["RP-FLT3-001"],
                "forbidden_drug_ids": ["CHEMBL-BLOCKED"],
                "forbidden_drug_classes": ["prohibited class"],
                "reserved_control_compounds": [
                    {
                        "name": "Positive assay control",
                        "registry_ids": [
                            {
                                "namespace": "ChEMBL",
                                "identifier": "CHEMBL99999999",
                            }
                        ],
                    }
                ],
            },
        )
        assert created.status_code == 202, created.text
        snapshot = await wait_for_stage(
            client, created.json()["run_id"], {"human_gate", "failed", "blocked"}
        )
        assert snapshot["stage"] == "human_gate", snapshot.get("error")
        assert snapshot["brief"]["constraints"]["forbidden_drugs"] == [
            "RP-FLT3-001"
        ]
        assert snapshot["plan_spec"]["forbidden_drug_ids"] == ["CHEMBL-BLOCKED"]
        expected_control = [
            {
                "name": "Positive assay control",
                "registry_ids": [
                    {
                        "namespace": "ChEMBL",
                        "identifier": "CHEMBL99999999",
                    }
                ],
            }
        ]
        assert snapshot["brief"]["constraints"]["reserved_control_compounds"] == (
            expected_control
        )
        assert snapshot["plan_spec"]["reserved_control_compounds"] == expected_control
        excluded = next(
            item
            for item in snapshot["candidate_packet"]["candidates"]
            if item["name"] == "RP-FLT3-001"
        )
        assert excluded["status"] == "excluded"
        assert excluded["rank"] is None

        rejected = await client.post(
            f"/api/runs/{snapshot['run_id']}/human-decision",
            json={
                "decision": "APPROVE",
                "researcher": "Integration Tester",
                "rationale": "Deliberately try to approve an excluded drug.",
                "selected_candidate_ids": [excluded["candidate_id"]],
            },
        )
        assert rejected.status_code == 409


@pytest.mark.asyncio
async def test_fault_detection_and_selective_rework(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/runs",
            json={
                "research_goal": "Find FLT3 candidates and exercise IC50 audit recovery for AML.",
                "disease": "Acute myeloid leukemia",
                "intervention_direction": "inhibit",
                "candidates_per_route": 3,
                "max_ma1_iterations": 2,
                "max_rework_loops": 2,
                "fault_injection": "g1_ic50_unit",
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        snapshot = await wait_for_stage(
            client, run_id, {"human_gate", "abstained", "blocked", "failed"}
        )
        assert snapshot["stage"] == "human_gate", snapshot.get("error")
        assert snapshot["rework_loops"] >= 1
        assert snapshot["g1_verdict"]["verdict"] == "PASS"
        event_types = [event["event_type"] for event in snapshot["events"]]
        assert "rework_order" in event_types
        ma2_events = [
            event for event in snapshot["events"] if event["event_type"] == "ma2_completed"
        ]
        assert len(ma2_events) >= 2
        assert snapshot["candidate_packet"]["reused_artifact_ids"]
