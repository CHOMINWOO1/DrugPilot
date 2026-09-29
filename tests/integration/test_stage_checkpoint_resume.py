from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import (
    GateName,
    GateVerdict,
    GateVerdictCode,
    HumanGateDecision,
    ResearchBrief,
    ReworkOrder,
    RunMode,
    RunStage,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.storage import SQLiteRunStore


class _InterruptingSyncAgent:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError(f"simulated process interruption in {self.label}")

    def close(self) -> None:
        return None


class _ForbiddenSyncAgent:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(f"completed {self.label} stage was rerun")


class _InterruptingReviewer:
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    async def audit_g0(self, *args, **kwargs):
        return await self.delegate.audit_g0(*args, **kwargs)

    async def audit_g1(self, *args, **kwargs):
        raise RuntimeError("simulated process interruption after MA2")


class _SameIdChangingMA1:
    """Reproduce an MA1 worker returning changed science under a reused ID."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0
        self.first_packet_id: str | None = None

    def run(self, brief, revision_context=None):
        self.calls += 1
        packet = self.delegate.run(brief, revision_context)
        if self.calls == 1:
            self.first_packet_id = packet.packet_id
            return packet
        assert revision_context is not None
        assert packet.hypotheses
        packet = packet.model_copy(
            update={"packet_id": revision_context.packet_id},
            deep=True,
        )
        packet.hypotheses[0].mechanism += " Reworked after the G0 finding."
        return packet


class _ReviseOnceG0Reviewer:
    def __init__(self) -> None:
        self.calls = 0
        self.order_id: str | None = None

    async def audit_g0(self, *args, **kwargs) -> GateVerdict:
        self.calls += 1
        if self.calls == 1:
            order = ReworkOrder(
                gate=GateName.G0,
                target_stage="ma1",
                failed_node="ma1.proposer",
                required_actions=["Revise the mechanism narrative."],
            )
            self.order_id = order.order_id
            return GateVerdict(
                gate=GateName.G0,
                verdict=GateVerdictCode.REVISE,
                rationale="Focused regression requests one MA1 revision.",
                rework_order=order,
                reviewer_model="focused-regression",
            )
        return GateVerdict(
            gate=GateName.G0,
            verdict=GateVerdictCode.PASS,
            rationale="The revised packet passes the focused regression gate.",
            reviewer_model="focused-regression",
        )

    async def audit_g1(self, *args, **kwargs):
        raise AssertionError("G1 is outside this focused regression")


def _settings(tmp_path, name: str) -> Settings:
    return Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / f"{name}.sqlite3",
        mock_step_delay_ms=0,
        candidates_per_route=2,
        max_ma1_iterations=2,
        max_rework_loops=1,
        public_evidence_enabled=False,
        demo_fixture_fallback_enabled=True,
        ma2_backend_mode="fixture",
        ma2_portfolio_enabled=False,
    )


async def _wait_for_owned_task(
    orchestrator: DrugPilotOrchestrator,
    run_id: str,
) -> None:
    task = orchestrator._tasks.get(run_id)
    if task is not None:
        await task
    await asyncio.sleep(0)


def _simulate_process_loss(orchestrator: DrugPilotOrchestrator) -> None:
    async def leave_in_progress_stage(*args, **kwargs) -> None:
        return None

    orchestrator._terminal = leave_in_progress_stage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interrupted_stage", "latest_completed_node"),
    [
        ("ma1", "validate_brief"),
        ("planner", "g0"),
        ("ma2", "planner"),
    ],
)
async def test_process_restart_resumes_first_incomplete_stage_without_reusing_agents(
    tmp_path,
    brief: ResearchBrief,
    interrupted_stage: str,
    latest_completed_node: str,
) -> None:
    settings = _settings(tmp_path, f"resume-{interrupted_stage}")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    interrupter = _InterruptingSyncAgent(interrupted_stage)
    if interrupted_stage == "ma1":
        first.ma1 = interrupter
    elif interrupted_stage == "planner":
        first.planner = interrupter
    else:
        first.ma2 = interrupter
    _simulate_process_loss(first)

    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    interrupted = await store.get_snapshot(created.run_id)
    assert interrupted is not None
    assert interrupted.stage is RunStage(interrupted_stage)
    assert interrupter.calls == 1
    checkpoint = await store.latest_stage_checkpoint(created.run_id)
    assert checkpoint is not None
    assert checkpoint.completed_node == latest_completed_node
    active_probe = asyncio.create_task(asyncio.Event().wait())
    first._tasks[created.run_id] = active_probe
    active_status = await first.resume_status(created.run_id)
    assert active_status["eligible"] is False
    assert active_status["reason_code"] == "RUN_ACTIVE"
    assert active_status["active_in_process"] is True
    active_probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active_probe
    first._tasks.pop(created.run_id, None)
    interrupted_status = await first.resume_status(created.run_id)
    assert interrupted_status["eligible"] is True
    assert interrupted_status["reason_code"] == "CHECKPOINT_READY"
    assert interrupted_status["checkpoint_completed_node"] == latest_completed_node
    checkpoint_artifact_ids = set(checkpoint.artifact_sha256)
    original_packet_id = (
        interrupted.hypothesis_packet.packet_id
        if interrupted.hypothesis_packet is not None
        else None
    )
    original_plan_id = (
        interrupted.plan_spec.plan_id if interrupted.plan_spec is not None else None
    )
    first.close()

    # A fresh orchestrator/store pair represents a new server process.  Any
    # already completed expensive agent is replaced with a bomb so the test
    # proves the checkpoint, rather than normal deterministic coincidence,
    # supplied its output.
    restarted_store = SQLiteRunStore(settings.database_path)
    restarted = DrugPilotOrchestrator(settings=settings, store=restarted_store)
    forbidden_ma1 = _ForbiddenSyncAgent("MA1")
    forbidden_planner = _ForbiddenSyncAgent("Planner")
    if interrupted_stage in {"planner", "ma2"}:
        restarted.ma1 = forbidden_ma1
    if interrupted_stage == "ma2":
        restarted.planner = forbidden_planner

    queued = await restarted.resume(created.run_id)
    assert queued.run_id == created.run_id
    await _wait_for_owned_task(restarted, created.run_id)
    completed = await restarted_store.get_snapshot(created.run_id)
    assert completed is not None
    assert completed.stage is RunStage.HUMAN_GATE
    assert forbidden_ma1.calls == 0
    assert forbidden_planner.calls == 0
    if original_packet_id is not None:
        assert completed.hypothesis_packet is not None
        assert completed.hypothesis_packet.packet_id == original_packet_id
    if original_plan_id is not None:
        assert completed.plan_spec is not None
        assert completed.plan_spec.plan_id == original_plan_id
    manifest = await restarted_store.checkpoint_manifest(created.run_id)
    assert sum(item["completed_node"] == "validate_brief" for item in manifest) == 1
    assert checkpoint_artifact_ids <= {
        item["artifact_id"]
        for item in await restarted_store.artifact_manifest(created.run_id)
    }
    assert await restarted_store.verify_ledger(created.run_id)
    restarted.close()


@pytest.mark.asyncio
async def test_g0_ma1_rework_versions_immutable_packet_and_resume_uses_revision(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "g0-ma1-immutable-revision")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    changing_ma1 = _SameIdChangingMA1(first.ma1)
    reviewer = _ReviseOnceG0Reviewer()
    first.ma1 = changing_ma1
    first.ma3 = reviewer
    first.planner = _InterruptingSyncAgent("planner-after-revised-g0")
    _simulate_process_loss(first)

    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    interrupted = await store.get_snapshot(created.run_id)
    assert interrupted is not None
    assert interrupted.stage is RunStage.PLANNER
    assert interrupted.hypothesis_packet is not None
    assert changing_ma1.calls == 2
    assert reviewer.calls == 2
    original_packet_id = changing_ma1.first_packet_id
    revised_packet = interrupted.hypothesis_packet
    assert original_packet_id is not None
    assert revised_packet.packet_id != original_packet_id
    lineage = revised_packet.metadata["revision_lineage"]
    assert lineage == {
        "revision_number": 1,
        "parent_packet_id": original_packet_id,
        "rework_order_id": reviewer.order_id,
        "gate": "G0",
        "target_stage": "ma1",
        "failed_node": "ma1.proposer",
        "scientific_changed": True,
        "parent_scientific_payload_sha256": lineage[
            "parent_scientific_payload_sha256"
        ],
        "scientific_payload_sha256": lineage["scientific_payload_sha256"],
        "parent_screening_trace_sha256": lineage[
            "parent_screening_trace_sha256"
        ],
        "screening_trace_sha256": lineage["screening_trace_sha256"],
        "audit_trace_changed": lineage["audit_trace_changed"],
        "parent_revision_payload_sha256": lineage[
            "parent_revision_payload_sha256"
        ],
        "revision_payload_sha256": lineage["revision_payload_sha256"],
    }
    assert len(lineage["parent_scientific_payload_sha256"]) == 64
    assert len(lineage["scientific_payload_sha256"]) == 64
    assert len(lineage["parent_screening_trace_sha256"]) == 64
    assert len(lineage["screening_trace_sha256"]) == 64
    assert isinstance(lineage["audit_trace_changed"], bool)
    assert len(lineage["parent_revision_payload_sha256"]) == 64
    assert len(lineage["revision_payload_sha256"]) == 64
    assert any(hypothesis.revision >= 1 for hypothesis in revised_packet.hypotheses)

    packet_artifacts = [
        item
        for item in await store.artifact_manifest(created.run_id)
        if item["artifact_type"] == "HypothesisPacket"
    ]
    assert {item["artifact_id"] for item in packet_artifacts} == {
        original_packet_id,
        revised_packet.packet_id,
    }
    g0_checkpoint = await store.latest_stage_checkpoint(created.run_id)
    assert g0_checkpoint is not None
    assert g0_checkpoint.completed_node == "g0"
    assert g0_checkpoint.hypothesis_packet is not None
    assert g0_checkpoint.hypothesis_packet.packet_id == revised_packet.packet_id
    assert g0_checkpoint.g0_verdict is not None
    assert g0_checkpoint.g0_verdict.verdict is GateVerdictCode.PASS
    assert {original_packet_id, revised_packet.packet_id} <= set(
        g0_checkpoint.artifact_sha256
    )
    assert await store.verify_ledger(created.run_id)
    first.close()

    restarted_store = SQLiteRunStore(settings.database_path)
    restarted = DrugPilotOrchestrator(settings=settings, store=restarted_store)
    forbidden_ma1 = _ForbiddenSyncAgent("MA1")
    restarted.ma1 = forbidden_ma1
    restarted.ma2 = _InterruptingSyncAgent("ma2-after-resumed-planner")
    _simulate_process_loss(restarted)
    status = await restarted.resume_status(created.run_id)
    assert status["eligible"] is True
    assert status["resume_from_node"] == "planner"

    await restarted.resume(created.run_id)
    await _wait_for_owned_task(restarted, created.run_id)
    resumed = await restarted_store.get_snapshot(created.run_id)
    assert resumed is not None
    assert resumed.stage is RunStage.MA2
    assert resumed.hypothesis_packet is not None
    assert resumed.hypothesis_packet.packet_id == revised_packet.packet_id
    assert resumed.plan_spec is not None
    assert resumed.plan_spec.hypothesis_packet_id == revised_packet.packet_id
    assert forbidden_ma1.calls == 0
    planner_checkpoint = await restarted_store.latest_stage_checkpoint(
        created.run_id
    )
    assert planner_checkpoint is not None
    assert planner_checkpoint.completed_node == "planner"
    assert planner_checkpoint.hypothesis_packet is not None
    assert planner_checkpoint.hypothesis_packet.packet_id == revised_packet.packet_id
    assert planner_checkpoint.plan_spec is not None
    assert (
        planner_checkpoint.plan_spec.hypothesis_packet_id
        == revised_packet.packet_id
    )
    assert await restarted_store.verify_ledger(created.run_id)
    restarted.close()


@pytest.mark.asyncio
async def test_completed_ma2_tool_call_ids_are_reused_after_restart(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "resume-after-ma2")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    first.ma3 = _InterruptingReviewer(first.ma3)
    _simulate_process_loss(first)

    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    interrupted = await store.get_snapshot(created.run_id)
    assert interrupted is not None
    assert interrupted.stage is RunStage.G1
    assert interrupted.candidate_packet is not None
    candidate_packet_id = interrupted.candidate_packet.packet_id
    tool_call_ids = [call.call_id for call in interrupted.candidate_packet.tool_calls]
    assert tool_call_ids
    first.close()

    restarted_store = SQLiteRunStore(settings.database_path)
    restarted = DrugPilotOrchestrator(settings=settings, store=restarted_store)
    forbidden_ma1 = _ForbiddenSyncAgent("MA1")
    forbidden_planner = _ForbiddenSyncAgent("Planner")
    forbidden_ma2 = _ForbiddenSyncAgent("MA2")
    restarted.ma1 = forbidden_ma1
    restarted.planner = forbidden_planner
    restarted.ma2 = forbidden_ma2

    # Two callers racing to resume are coalesced into one owned task.
    await asyncio.gather(
        restarted.resume(created.run_id),
        restarted.resume(created.run_id),
    )
    await _wait_for_owned_task(restarted, created.run_id)
    completed = await restarted_store.get_snapshot(created.run_id)
    assert completed is not None
    assert completed.stage is RunStage.HUMAN_GATE
    assert completed.candidate_packet is not None
    assert completed.candidate_packet.packet_id == candidate_packet_id
    assert [call.call_id for call in completed.candidate_packet.tool_calls] == tool_call_ids
    assert forbidden_ma1.calls == forbidden_planner.calls == forbidden_ma2.calls == 0
    checkpoint_count = len(
        await restarted_store.checkpoint_manifest(created.run_id)
    )
    waiting = await restarted.resume(created.run_id)
    assert waiting.stage is RunStage.HUMAN_GATE
    waiting_status = await restarted.resume_status(created.run_id)
    assert waiting_status["eligible"] is False
    assert waiting_status["reason_code"] == "AWAITING_HUMAN_DECISION"
    assert len(await restarted_store.checkpoint_manifest(created.run_id)) == checkpoint_count
    assert await restarted_store.verify_ledger(created.run_id)
    # ``close`` would target the intentionally forbidden MA2 double.


@pytest.mark.asyncio
async def test_resume_rebases_only_single_input_cap_after_matching_failure(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "resume-single-input-cap")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    first.ma3 = _InterruptingReviewer(first.ma3)
    _simulate_process_loss(first)

    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    interrupted = await store.get_snapshot(created.run_id)
    assert interrupted is not None
    assert interrupted.stage is RunStage.G1
    original_budget = interrupted.brief.budget
    interrupted.stage = RunStage.ABSTAINED
    interrupted.error = (
        "single input cap would be exceeded "
        f"({original_budget.max_single_input_tokens + 1:,}/"
        f"{original_budget.max_single_input_tokens:,})"
    )
    await store.save_snapshot(interrupted)
    first.close()

    increased_limit = original_budget.max_single_input_tokens + 50_000
    restarted_settings = settings.model_copy(
        update={"max_single_input_tokens": increased_limit}
    )
    restarted_store = SQLiteRunStore(settings.database_path)
    restarted = DrugPilotOrchestrator(
        settings=restarted_settings,
        store=restarted_store,
    )
    captured: dict[str, object] = {}

    async def _capture_execute(run_id, resumed_brief, resume_state):
        captured["run_id"] = run_id
        captured["brief"] = resumed_brief
        captured["state"] = resume_state

    restarted._execute = _capture_execute  # type: ignore[method-assign]
    await restarted.resume(created.run_id)
    await _wait_for_owned_task(restarted, created.run_id)

    resumed = await restarted_store.get_snapshot(created.run_id)
    assert resumed is not None
    assert resumed.brief.budget.max_single_input_tokens == increased_limit
    assert resumed.brief.budget.model_copy(
        update={"max_single_input_tokens": original_budget.max_single_input_tokens}
    ) == original_budget
    assert captured["run_id"] == created.run_id
    captured_brief = captured["brief"]
    assert isinstance(captured_brief, ResearchBrief)
    assert captured_brief.budget.max_single_input_tokens == increased_limit
    captured_state = captured["state"]
    assert isinstance(captured_state, dict)
    assert captured_state["brief"] == captured_brief

    with sqlite3.connect(settings.database_path) as connection:
        event_row = connection.execute(
            """
            SELECT event_json FROM audit_ledger
            WHERE run_id = ? AND event_json LIKE ?
            ORDER BY sequence DESC LIMIT 1
            """,
            (created.run_id, '%"event_type":"workflow_resume_queued"%'),
        ).fetchone()
    assert event_row is not None
    resume_event = json.loads(event_row[0])
    assert resume_event["metadata"]["budget_resume_override"] == {
        "previous_max_single_input_tokens": (
            original_budget.max_single_input_tokens
        ),
        "effective_max_single_input_tokens": increased_limit,
    }
    assert await restarted_store.verify_ledger(created.run_id)
    restarted.close()


@pytest.mark.asyncio
async def test_resume_rebases_only_cost_cap_after_matching_failure(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "resume-cost-cap")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    first.ma3 = _InterruptingReviewer(first.ma3)
    _simulate_process_loss(first)

    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    interrupted = await store.get_snapshot(created.run_id)
    assert interrupted is not None
    assert interrupted.stage is RunStage.G1
    original_budget = interrupted.brief.budget
    carried_input_limit = original_budget.max_single_input_tokens + 50_000
    carried_budget = original_budget.model_copy(
        update={"max_single_input_tokens": carried_input_limit}
    )
    interrupted.brief = interrupted.brief.model_copy(
        update={"budget": carried_budget}
    )
    interrupted.stage = RunStage.ABSTAINED
    interrupted.error = (
        "run cost cap would be exceeded "
        f"(${original_budget.max_run_cost_usd + 0.16:.2f}/"
        f"${original_budget.max_run_cost_usd:.2f})"
    )
    await store.save_snapshot(interrupted)
    first.close()

    increased_limit = original_budget.max_run_cost_usd + 5.0
    restarted_settings = settings.model_copy(
        update={"max_run_cost_usd": increased_limit}
    )
    restarted_store = SQLiteRunStore(settings.database_path)
    restarted = DrugPilotOrchestrator(
        settings=restarted_settings,
        store=restarted_store,
    )
    captured: dict[str, object] = {}

    async def _capture_execute(run_id, resumed_brief, resume_state):
        captured["run_id"] = run_id
        captured["brief"] = resumed_brief
        captured["state"] = resume_state

    restarted._execute = _capture_execute  # type: ignore[method-assign]
    await restarted.resume(created.run_id)
    await _wait_for_owned_task(restarted, created.run_id)

    resumed = await restarted_store.get_snapshot(created.run_id)
    assert resumed is not None
    assert resumed.brief.budget.max_run_cost_usd == increased_limit
    assert resumed.brief.budget.max_single_input_tokens == carried_input_limit
    assert resumed.brief.budget.model_copy(
        update={"max_run_cost_usd": original_budget.max_run_cost_usd}
    ) == carried_budget
    assert captured["run_id"] == created.run_id
    captured_brief = captured["brief"]
    assert isinstance(captured_brief, ResearchBrief)
    assert captured_brief.budget.max_run_cost_usd == increased_limit
    captured_state = captured["state"]
    assert isinstance(captured_state, dict)
    assert captured_state["brief"] == captured_brief

    with sqlite3.connect(settings.database_path) as connection:
        event_row = connection.execute(
            """
            SELECT event_json FROM audit_ledger
            WHERE run_id = ? AND event_json LIKE ?
            ORDER BY sequence DESC LIMIT 1
            """,
            (created.run_id, '%"event_type":"workflow_resume_queued"%'),
        ).fetchone()
    assert event_row is not None
    resume_event = json.loads(event_row[0])
    assert resume_event["metadata"]["budget_resume_override"] == {
        "previous_max_run_cost_usd": original_budget.max_run_cost_usd,
        "effective_max_run_cost_usd": increased_limit,
    }
    assert await restarted_store.verify_ledger(created.run_id)
    restarted.close()


@pytest.mark.asyncio
async def test_resume_rejects_scientific_brief_change_despite_larger_caps(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "resume-scientific-brief-mismatch")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    first.ma3 = _InterruptingReviewer(first.ma3)
    _simulate_process_loss(first)

    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    interrupted = await store.get_snapshot(created.run_id)
    assert interrupted is not None
    interrupted.stage = RunStage.ABSTAINED
    interrupted.error = "single input cap would be exceeded (1,000,001/1,000,000)"
    interrupted.brief = interrupted.brief.model_copy(
        update={
            "research_goal": interrupted.brief.research_goal + " changed",
            "budget": interrupted.brief.budget.model_copy(
                update={
                    "max_single_input_tokens": (
                        interrupted.brief.budget.max_single_input_tokens + 50_000
                    )
                }
            ),
        }
    )
    await store.save_snapshot(interrupted)
    first.close()

    restarted = DrugPilotOrchestrator(
        settings=settings.model_copy(
            update={
                "max_single_input_tokens": (
                    interrupted.brief.budget.max_single_input_tokens + 50_000
                )
            }
        ),
        store=SQLiteRunStore(settings.database_path),
    )
    status = await restarted.resume_status(created.run_id)
    assert status["eligible"] is False
    assert status["reason_code"] == "CHECKPOINT_BRIEF_MISMATCH"
    with pytest.raises(ValueError, match="Research Brief does not match"):
        await restarted.resume(created.run_id)
    assert created.run_id not in restarted._tasks
    restarted.close()


@pytest.mark.asyncio
async def test_incompatible_checkpoint_fails_closed(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "incompatible-checkpoint")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    first.planner = _InterruptingSyncAgent("planner")
    _simulate_process_loss(first)
    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    first.close()

    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            """
            UPDATE stage_checkpoints SET workflow_version = ?
            WHERE sequence = (
                SELECT MAX(sequence) FROM stage_checkpoints WHERE run_id = ?
            )
            """,
            ("unsupported-workflow-v99", created.run_id),
        )

    restarted = DrugPilotOrchestrator(
        settings=settings,
        store=SQLiteRunStore(settings.database_path),
    )
    with pytest.raises(ValueError, match="checkpoint verification failed"):
        await restarted.resume(created.run_id)
    assert created.run_id not in restarted._tasks
    restarted.close()


@pytest.mark.asyncio
async def test_resume_status_api_exposes_verified_orphan_without_private_state(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "resume-status-api")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    first.planner = _InterruptingSyncAgent("planner")
    _simulate_process_loss(first)
    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    first.close()

    app = create_app(
        settings=settings,
        store=SQLiteRunStore(settings.database_path),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(f"/api/runs/{created.run_id}/resume-status")
        manifest_response = await client.get(
            f"/api/runs/{created.run_id}/artifacts"
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "run_id": created.run_id,
        "eligible": True,
        "reason_code": "CHECKPOINT_READY",
        "reason": "A verified checkpoint can resume from the first incomplete node.",
        "active_in_process": False,
        "current_stage": "planner",
        "resume_from_node": "planner",
        "checkpoint_completed_node": "g0",
        "checkpoint_integrity": "verified",
    }
    serialized = str(payload).lower()
    assert "prompt" not in serialized
    assert "reasoning" not in serialized
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["ledger_valid"] is True
    assert manifest["stage_checkpoints"][-1]["completed_node"] == "g0"
    assert manifest["stage_checkpoints"][-1]["schema_version"] == "1.0"
    assert len(manifest["stage_checkpoints"][-1]["sha256"]) == 64


@pytest.mark.asyncio
async def test_persisted_human_hold_cannot_be_reopened_by_resume(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    settings = _settings(tmp_path, "human-hold-no-reopen")
    store = SQLiteRunStore(settings.database_path)
    first = DrugPilotOrchestrator(settings=settings, store=store)
    created = await first.create_run(brief)
    await _wait_for_owned_task(first, created.run_id)
    waiting = await store.get_snapshot(created.run_id)
    assert waiting is not None
    assert waiting.stage is RunStage.HUMAN_GATE
    held = await first.apply_human_decision(
        HumanGateDecision(
            run_id=created.run_id,
            decision="HOLD",
            researcher="Checkpoint safety test",
            rationale="Persist HOLD and verify that resume cannot reopen the gate.",
        )
    )
    assert held.stage is RunStage.ABSTAINED
    assert held.human_decision is not None
    first.close()

    restarted = DrugPilotOrchestrator(
        settings=settings,
        store=SQLiteRunStore(settings.database_path),
    )
    status = await restarted.resume_status(created.run_id)
    assert status["eligible"] is False
    assert status["reason_code"] == "HUMAN_DECISION_ALREADY_RECORDED"
    assert status["resume_from_node"] is None
    with pytest.raises(ValueError, match="human decision cannot be reopened"):
        await restarted.resume(created.run_id)
    assert created.run_id not in restarted._tasks
    restored = await restarted.store.get_snapshot(created.run_id)
    assert restored is not None
    assert restored.stage is RunStage.ABSTAINED
    assert restored.human_decision is not None
    assert restored.human_decision.decision.value == "HOLD"
    restarted.close()


@pytest.mark.asyncio
async def test_old_done_callback_cannot_remove_newer_registered_task(
    tmp_path,
) -> None:
    settings = _settings(tmp_path, "task-callback-identity")
    orchestrator = DrugPilotOrchestrator(
        settings=settings,
        store=SQLiteRunStore(settings.database_path),
    )
    release_new = asyncio.Event()
    old_task = asyncio.create_task(asyncio.sleep(0))
    new_task = asyncio.create_task(release_new.wait())
    orchestrator._track_task("run_callback_race", old_task)
    orchestrator._track_task("run_callback_race", new_task)
    await old_task
    await asyncio.sleep(0)
    assert orchestrator._tasks.get("run_callback_race") is new_task

    new_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await new_task
    await asyncio.sleep(0)
    assert "run_callback_race" not in orchestrator._tasks
    orchestrator.close()
