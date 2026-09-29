from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import GateVerdict, GateVerdictCode, RunMode
from drugpilot.storage import SQLiteRunStore


async def _wait_for_human_gate(
    client: AsyncClient,
    run_id: str,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    terminal = {"human_gate", "failed", "abstained", "blocked"}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        snapshot = response.json()
        if snapshot["stage"] in terminal:
            return snapshot
        await asyncio.sleep(0.02)
    pytest.fail(f"run {run_id} did not reach the human gate")


def _event(
    events: list[Any],
    event_type: str,
    *,
    node: str,
) -> Any:
    matches = [
        event
        for event in events
        if event.event_type == event_type and event.node == node
    ]
    assert len(matches) == 1, (event_type, node, len(matches))
    return matches[0]


def _assert_validated_pass(verdict: GateVerdict, gate_event: Any) -> None:
    """A PASS must be the result of persisted checks, not a stage default."""

    assert verdict.verdict is GateVerdictCode.PASS
    assert verdict.validations
    assert all(validation.passed for validation in verdict.validations)
    assert all(validation.name.strip() for validation in verdict.validations)
    assert all(validation.control_type for validation in verdict.validations)
    assert all(isinstance(validation.details, dict) for validation in verdict.validations)
    assert verdict.rationale.strip()
    assert verdict.rationale.strip().upper() != "PASS"

    trace = gate_event.metadata["interaction"]
    assert trace["kind"] == "audit_decision"
    assert trace["decision"] == "PASS"
    assert trace["explanation_scope"] == "evidence_and_decision_summary"
    assert verdict.rationale in trace["basis"]
    assert (
        f"검증 {len(verdict.validations)}개 중 실패 0개"
        in trace["basis"]
    )
    assert trace["details"]["failed_validations"] == []


@pytest.mark.asyncio
async def test_mock_run_persists_explanatory_handoffs_and_gate_basis(tmp_path) -> None:
    """Exercise the real graph and re-read its trace from the SQLite snapshot."""

    store = SQLiteRunStore(tmp_path / "explanations.sqlite3")
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        auto_activate_openai=False,
        database_path=store.path,
        mock_step_delay_ms=0,
        candidates_per_route=2,
        max_ma1_iterations=2,
        max_rework_loops=1,
        public_evidence_enabled=False,
        demo_fixture_fallback_enabled=True,
        public_candidate_enrichment_enabled=False,
        dynamic_target_discovery_enabled=False,
        target_identity_resolution_enabled=False,
        dynamic_context_enabled=False,
    )
    app = create_app(settings=settings, store=store)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/runs",
            json={
                "research_goal": (
                    "Identify evidence-linked FLT3 inhibitors and preserve every "
                    "scientific decision handoff for an audit demonstration."
                ),
                "disease": "Acute myeloid leukemia",
                "target_cell_type": "MOLM-13",
                "intervention_direction": "inhibit",
                "candidates_per_route": 2,
                "max_ma1_iterations": 2,
                "max_rework_loops": 1,
                "fault_injection": "none",
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        response_snapshot = await _wait_for_human_gate(client, run_id)
        assert response_snapshot["stage"] == "human_gate", response_snapshot.get("error")

    # Re-read from SQLite rather than relying on the in-memory API response.
    persisted = await store.get_snapshot(run_id)
    assert persisted is not None
    assert persisted.stage.value == "human_gate"
    assert persisted.g0_verdict is not None
    assert persisted.g1_verdict is not None

    ma1 = _event(persisted.events, "ma1_completed", node="ma1.selector")
    g0 = _event(persisted.events, "gate_verdict", node="g0.audit")
    planner = _event(persisted.events, "planner_completed", node="planner")
    ma2 = _event(persisted.events, "ma2_completed", node="ma2.judge")
    g1 = _event(persisted.events, "gate_verdict", node="g1.audit")

    expected_handoffs = [
        (ma1, "MA1", "MA3 · G0", "artifact_handoff"),
        (g0, "MA3 · G0", "Planner", "audit_decision"),
        (planner, "Planner", "MA2", "plan_dispatch"),
        (ma2, "MA2 · Judge", "MA3 · G1", "artifact_handoff"),
        (g1, "MA3 · G1", "Human Gate", "audit_decision"),
    ]
    event_positions = {event.event_id: index for index, event in enumerate(persisted.events)}
    for event, sender, receiver, kind in expected_handoffs:
        trace = event.metadata.get("interaction")
        assert isinstance(trace, dict)
        assert trace["sender"] == sender
        assert trace["receiver"] == receiver
        assert trace["kind"] == kind
        assert trace["summary"].strip()
        assert trace["basis"]
        assert trace["artifact_ids"]
        assert trace["explanation_scope"] == "evidence_and_decision_summary"

    assert [event_positions[event.event_id] for event, *_ in expected_handoffs] == sorted(
        event_positions[event.event_id] for event, *_ in expected_handoffs
    )
    _assert_validated_pass(persisted.g0_verdict, g0)
    _assert_validated_pass(persisted.g1_verdict, g1)

    manifest = await store.artifact_manifest(run_id)
    assert sum(item["artifact_type"] == "GateVerdict" for item in manifest) == 2
    assert await store.verify_ledger(run_id) is True
