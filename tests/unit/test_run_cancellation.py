"""Local cancellation regressions. No scientific API or paid call is made."""
import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.config import Settings
from drugpilot.api.app import create_app
from drugpilot.domain.models import ExecutionEvent, G1ReauditRecord, RunMode, RunSnapshot, RunStage, TokenUsage, utc_now
from drugpilot.orchestration.budget import BudgetExceeded
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.orchestration.human_review import HumanReviewService
from drugpilot.storage.sqlite import RunCancellationRequested, SQLiteRunStore


def setup(tmp_path, brief):
    store = SQLiteRunStore(tmp_path / "cancel.sqlite3")
    settings = Settings(_env_file=None, mode=RunMode.MOCK, auto_activate_openai=False,
        public_evidence_enabled=False, demo_fixture_fallback_enabled=True,
        database_path=store.path, mock_step_delay_ms=0)
    orchestrator = DrugPilotOrchestrator(settings=settings, store=store)
    snapshot = RunSnapshot(run_id="run_cancel_test", mode=RunMode.MOCK,
        stage=RunStage.MA1, status_message="Active test", brief=brief)
    return orchestrator, store, snapshot


@pytest.mark.asyncio
async def test_cancel_drains_call_preserves_usage_and_releases_lease(tmp_path, brief):
    o, store, snapshot = setup(tmp_path, brief)
    await store.save_snapshot(snapshot)
    await store.commit_artifact_event(snapshot, "existing", "Test", {"original": True},
        ExecutionEvent(run_id=snapshot.run_id, stage=RunStage.MA1, event_type="original",
                       message="Original artifact", artifact_ids=["existing"]))
    await store.claim_live_execution_lease(snapshot.run_id, lease_token="owned", now=utc_now(),
        lease_ttl_seconds=300, max_active_live_runs=1)
    started, finish = asyncio.Event(), asyncio.Event()
    dispatched = []

    async def fake_graph(state, config):
        dispatched.append("first")
        started.set()
        await finish.wait()  # This represents a call already sent to a provider.
        await o._record_llm_usage(snapshot.run_id, TokenUsage(component="test", model="gpt-5.6-luna",
            input_tokens=100, output_tokens=20))
        await o._cancellation_checkpoint(snapshot.run_id)
        dispatched.append("second")

    o.graph = SimpleNamespace(ainvoke=fake_graph)
    task = asyncio.create_task(o._execute_with_lease(snapshot.run_id, brief, lease_token="owned"))
    o._track_task(snapshot.run_id, task)
    await started.wait()
    pending = await o.cancel(snapshot.run_id)
    assert pending.stage is RunStage.CANCEL_REQUESTED
    assert (await o.cancel(snapshot.run_id)).cancellation_requested_at == pending.cancellation_requested_at
    assert not task.done()
    assert (await store.live_execution_lease_status(snapshot.run_id, now=utc_now()))["state"] == "active"
    finish.set()
    await asyncio.wait_for(task, 5)
    final = await store.get_snapshot(snapshot.run_id)
    assert final.stage is RunStage.CANCELLED
    assert final.cancelled_at is not None
    assert dispatched == ["first"]
    assert sum(u.total_tokens for u in final.token_usage) == 120
    assert sum(u.estimated_cost_usd for u in final.token_usage) > 0
    assert await store.read_artifact(snapshot.run_id, "existing", "Test") == {"original": True}
    assert await store.verify_ledger(snapshot.run_id)
    assert (await store.live_execution_lease_status(snapshot.run_id, now=utc_now()))["state"] == "released"
    assert (await o.resume_status(snapshot.run_id))["reason_code"] == "RUN_CANCELLED"
    with pytest.raises(RunCancellationRequested):
        await o.resume(snapshot.run_id)
    with pytest.raises(RunCancellationRequested):
        await HumanReviewService(o).recover(snapshot.run_id)
    assert (await o.cancel(snapshot.run_id)).stage is RunStage.CANCELLED
    o.close()


@pytest.mark.asyncio
async def test_durable_fence_rejects_stale_writes_even_from_another_store(tmp_path, brief):
    o, store, snapshot = setup(tmp_path, brief)
    await store.save_snapshot(snapshot)
    stale = snapshot.model_copy(deep=True)
    await store.set_cancellation(snapshot.run_id)
    other = SQLiteRunStore(store.path)
    with pytest.raises(RunCancellationRequested):
        await other.save_snapshot(stale)
    with pytest.raises(RunCancellationRequested):
        await other.save_artifact(snapshot.run_id, "late", "Test", {"late": True})
    with pytest.raises(RunCancellationRequested):
        await other.append_event(ExecutionEvent(run_id=snapshot.run_id, stage=RunStage.MA1,
            event_type="late", message="late"))
    with pytest.raises(RunCancellationRequested):
        await other.claim_live_execution_lease(snapshot.run_id, lease_token="new", now=utc_now(),
            lease_ttl_seconds=300, max_active_live_runs=1)
    await store.set_cancellation(snapshot.run_id, finished=True)
    with pytest.raises(RunCancellationRequested):
        await other.append_token_usage(snapshot.run_id, TokenUsage(component="late", model="gpt-4.1"))
    assert await store.verify_ledger(snapshot.run_id)
    o.close()


@pytest.mark.asyncio
async def test_cancel_does_not_touch_unowned_or_terminal_run(tmp_path, brief):
    o, store, snapshot = setup(tmp_path, brief)
    await store.save_snapshot(snapshot)
    with pytest.raises(ValueError, match="RUN_NOT_OWNED"):
        await o.cancel(snapshot.run_id)
    assert (await store.get_snapshot(snapshot.run_id)).stage is RunStage.MA1
    snapshot.stage = RunStage.HUMAN_GATE
    await store.save_snapshot(snapshot)
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    o._track_task(snapshot.run_id, task)
    with pytest.raises(ValueError, match="RUN_NOT_ACTIVE"):
        await o.cancel(snapshot.run_id)
    blocker.set()
    await task
    o.close()


@pytest.mark.asyncio
async def test_http_cancel_status_and_idempotent_request(tmp_path, brief):
    store = SQLiteRunStore(tmp_path / "api-cancel.sqlite3")
    app = create_app(Settings(_env_file=None, mode=RunMode.MOCK, auto_activate_openai=False,
        public_evidence_enabled=False, demo_fixture_fallback_enabled=True,
        beta_access_auth_required=False, database_path=store.path, mock_step_delay_ms=0), store)
    started, finish = asyncio.Event(), asyncio.Event()
    async with app.router.lifespan_context(app):
        o = app.state.orchestrator
        async def wait_graph(state, config):
            started.set()
            await finish.wait()
        o.graph = SimpleNamespace(ainvoke=wait_graph)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            created = await client.post("/api/runs", json={"research_goal": brief.research_goal, "disease": brief.disease})
            assert created.status_code == 202
            run_id = created.json()["run_id"]
            await started.wait()
            assert (await client.get(f"/api/runs/{run_id}/cancel-status")).json()["eligible"]
            task = o._tasks[run_id]
            cancelled = await client.post(f"/api/runs/{run_id}/cancel")
            assert cancelled.status_code == 202
            assert cancelled.json()["stage"] == "cancel_requested"
            assert not (await client.get(f"/api/runs/{run_id}/cancel-status")).json()["eligible"]
            finish.set()
            await asyncio.wait_for(task, 5)
            final = await client.post(f"/api/runs/{run_id}/cancel")
            assert final.status_code == 202
            assert final.json()["stage"] == "cancelled"
            assert (await client.post(f"/api/runs/{run_id}/resume")).status_code == 409
            assert (await client.post("/api/runs/unknown/cancel")).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("provider failed"), BudgetExceeded("limit reached")])
async def test_cancel_wins_race_inside_terminal_error_handler(tmp_path, brief, error, monkeypatch):
    o, store, snapshot = setup(tmp_path, brief)
    await store.save_snapshot(snapshot)
    async def fail_graph(state, config):
        raise error
    async def racing_materialization(*args, **kwargs):
        await o.cancel(snapshot.run_id)
        await store.save_artifact(snapshot.run_id, "late_error", "Test", {})
    o.graph = SimpleNamespace(ainvoke=fail_graph)
    monkeypatch.setattr(o, "_materialize_next_action_packet", racing_materialization)
    task = asyncio.create_task(o._execute(snapshot.run_id, brief))
    o._track_task(snapshot.run_id, task)
    await asyncio.wait_for(task, 5)
    assert (await store.get_snapshot(snapshot.run_id)).stage is RunStage.CANCELLED
    assert await store.verify_ledger(snapshot.run_id)
    o.close()


@pytest.mark.asyncio
async def test_restart_reconciles_expired_cancellation_without_replaying(tmp_path, brief):
    o, store, snapshot = setup(tmp_path, brief)
    record = G1ReauditRecord(request_id="cancel-recovery-test-01", request_sha256="a" * 64,
        input_sha256="b" * 64, source_sha256="c" * 64, researcher="test", rationale="test",
        mode="rules", previous_verdict_id="verdict_original")
    snapshot.g1_reaudits = [record]
    await store.save_snapshot(snapshot)
    await store.commit_artifact_event(snapshot, record.record_id, "G1ReauditRecord", record,
        ExecutionEvent(run_id=snapshot.run_id, stage=RunStage.MA1, event_type="review_started",
                       message="original", artifact_ids=[record.record_id]))
    await store.claim_live_execution_lease(snapshot.run_id, lease_token="old-owner", now=utc_now(),
        lease_ttl_seconds=300, max_active_live_runs=1)
    await store.set_cancellation(snapshot.run_id, lease_token="old-owner")
    await o.recover_pending_cancellations()
    assert (await store.get_snapshot(snapshot.run_id)).stage is RunStage.CANCEL_REQUESTED
    await store.expire_live_execution_lease_for_recovery(snapshot.run_id, lease_token="old-owner",
        reason="test-process-stopped", now=utc_now() - timedelta(seconds=1))
    await o.recover_pending_cancellations()
    final = await store.get_snapshot(snapshot.run_id)
    assert final.stage is RunStage.CANCELLED
    assert final.g1_reaudits[-1].status == "failed"
    assert final.g1_reaudits[-1].error_code == "INTERRUPTED_CANCELLATION_ACCOUNTING_UNCERTAIN"
    assert final.g1_reaudits[-1].record_id != record.record_id
    assert await store.read_artifact(snapshot.run_id, record.record_id, "G1ReauditRecord")
    assert final.events[-1].metadata["usage_accounting_complete"] is False
    assert (await store.live_execution_lease_status(snapshot.run_id, now=utc_now()))["state"] == "released"
    assert await store.verify_ledger(snapshot.run_id)
    o.close()


def test_ma2_cancel_checkpoint_runs_before_next_tool():
    from drugpilot.agents.ma2 import _TraceRecorder
    calls = []
    def stop():
        raise RunCancellationRequested("stop")
    recorder = _TraceRecorder("unit", before_operation=stop)
    with pytest.raises(RunCancellationRequested):
        recorder.invoke(task_id=None, tool_name="test", operation="test", inputs={},
                        function=lambda: calls.append("dispatched"))
    assert calls == []
    assert recorder.calls == []
