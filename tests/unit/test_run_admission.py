from __future__ import annotations

import asyncio
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import (
    HumanGateDecision,
    RunMode,
    RunStage,
    utc_now,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.storage import SQLiteRunStore


REQUEST = {
    "research_goal": "Identify an auditable target hypothesis for admission testing.",
    "disease": "Acute myeloid leukemia",
}
# Deterministic example UUIDs for idempotency; these are not API credentials.
KEY_A = "00000000-0000-4000-8000-000000000001"
KEY_B = "00000000-0000-4000-8000-000000000002"


def _settings(
    path: Path,
    *,
    mode: RunMode,
    max_active: int = 1,
    max_hourly: int = 20,
) -> Settings:
    return Settings(
        _env_file=None,
        mode=mode,
        openai_api_key=("sk-test-live-admission" if mode is RunMode.LIVE else None),
        auto_activate_openai=False,
        database_path=path,
        beta_access_auth_required=False,
        beta_max_active_live_runs=max_active,
        beta_max_new_runs_per_hour=max_hourly,
        public_evidence_enabled=False,
        demo_fixture_fallback_enabled=True,
        mock_step_delay_ms=0,
    )


def _app(
    path: Path,
    *,
    mode: RunMode = RunMode.LIVE,
    max_active: int = 1,
    max_hourly: int = 20,
):
    settings = _settings(
        path,
        mode=mode,
        max_active=max_active,
        max_hourly=max_hourly,
    )
    store = SQLiteRunStore(path)
    return create_app(settings=settings, store=store), store


@pytest.mark.asyncio
async def test_live_run_requires_a_bounded_idempotency_key(
    tmp_path, monkeypatch
) -> None:
    async def no_execute(self, *args, **kwargs):
        return None

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", no_execute)
    app, _ = _app(tmp_path / "required.sqlite3")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/api/runs", json=REQUEST)
        too_short = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": "short"},
            json=REQUEST,
        )

    assert missing.status_code == 400
    assert "required for live" in missing.text
    assert too_short.status_code == 400
    assert "16-128" in too_short.text


@pytest.mark.asyncio
async def test_same_key_same_request_replays_one_atomic_run_under_race(
    tmp_path, monkeypatch
) -> None:
    release = asyncio.Event()

    async def hold_execute(self, *args, **kwargs):
        await release.wait()

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", hold_execute)
    app, store = _app(tmp_path / "race.sqlite3", max_active=1)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first, replay = await asyncio.gather(
            client.post(
                "/api/runs",
                headers={"Idempotency-Key": KEY_A},
                json=REQUEST,
            ),
            client.post(
                "/api/runs",
                headers={"Idempotency-Key": KEY_A},
                json=REQUEST,
            ),
        )
        assert first.status_code == replay.status_code == 202
        run_id = first.json()["run_id"]
        assert replay.json()["run_id"] == run_id

        health = (await client.get("/api/health")).json()
        assert health["run_admission"] == {
            "idempotency_required": True,
            "active_live_runs": 1,
            "expired_live_leases": 0,
            "recovery_required": False,
            "max_active_live_runs": 1,
            "hourly_admitted_count": 1,
            "hourly_admitted_cap": 20,
        }

        different = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_A},
            json={**REQUEST, "disease": "Melanoma"},
        )
        capacity = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_B},
            json=REQUEST,
        )
        release.set()
        await asyncio.sleep(0)

    assert different.status_code == 409
    assert capacity.status_code == 429
    assert capacity.headers["retry-after"] == "30"
    admission = await store.get_run_admission(KEY_A)
    assert admission is not None
    assert admission["run_id"] == run_id
    assert len(admission["request_sha256"]) == 64
    assert len(await store.list_snapshots(10)) == 1
    assert len(await store.artifact_manifest(run_id)) == 2
    assert await store.verify_ledger(run_id) is True


@pytest.mark.asyncio
async def test_idempotency_mapping_survives_app_restart(
    tmp_path, monkeypatch
) -> None:
    async def no_execute(self, *args, **kwargs):
        return None

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", no_execute)
    path = tmp_path / "restart.sqlite3"
    first_app, _ = _app(path)
    first_transport = ASGITransport(app=first_app)
    async with AsyncClient(
        transport=first_transport, base_url="http://test"
    ) as client:
        first = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_A},
            json=REQUEST,
        )
    assert first.status_code == 202

    restarted_app, restarted_store = _app(path)
    restarted_transport = ASGITransport(app=restarted_app)
    async with AsyncClient(
        transport=restarted_transport, base_url="http://test"
    ) as client:
        replay = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_A},
            json=REQUEST,
        )

    assert replay.status_code == 202
    assert replay.json()["run_id"] == first.json()["run_id"]
    assert len(await restarted_store.list_snapshots(10)) == 1


@pytest.mark.asyncio
async def test_orphaned_mapping_fails_closed_without_creating_a_second_run(
    tmp_path, monkeypatch
) -> None:
    async def no_execute(self, *args, **kwargs):
        return None

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", no_execute)
    path = tmp_path / "orphan.sqlite3"
    app, store = _app(path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_A},
            json=REQUEST,
        )
        assert created.status_code == 202

        # Simulate out-of-band database damage that bypassed foreign keys.  A
        # retry must never invent a replacement run for the durable mapping.
        with sqlite3.connect(path) as connection:
            connection.execute(
                "DELETE FROM runs WHERE run_id = ?", (created.json()["run_id"],)
            )

        replay = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_A},
            json=REQUEST,
        )

    assert replay.status_code == 409
    assert "IDEMPOTENCY_MAPPING_ORPHANED" in replay.text
    assert await store.get_run_admission(KEY_A) is not None
    assert await store.get_snapshot(created.json()["run_id"]) is None


@pytest.mark.asyncio
async def test_hourly_limit_is_auth_independent_and_counts_only_successes(
    tmp_path, monkeypatch
) -> None:
    async def no_execute(self, *args, **kwargs):
        return None

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", no_execute)
    app, store = _app(
        tmp_path / "hourly.sqlite3",
        max_active=2,
        max_hourly=1,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        invalid = await client.post("/api/runs", json=REQUEST)
        first = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_A},
            json=REQUEST,
        )
        second = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_B},
            json=REQUEST,
        )
        replay = await client.post(
            "/api/runs",
            headers={"Idempotency-Key": KEY_A},
            json=REQUEST,
        )

    assert invalid.status_code == 400
    assert first.status_code == replay.status_code == 202
    assert replay.json()["run_id"] == first.json()["run_id"]
    assert second.status_code == 429
    assert second.headers["retry-after"] == "3600"
    assert len(await store.list_snapshots(10)) == 1


@pytest.mark.asyncio
async def test_distinct_apps_cannot_race_past_global_live_cap(
    tmp_path, monkeypatch
) -> None:
    release = asyncio.Event()

    async def hold_execute(self, *args, **kwargs):
        await release.wait()

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", hold_execute)
    path = tmp_path / "multi-app.sqlite3"
    app_a, store = _app(path, max_active=1)
    app_b, _ = _app(path, max_active=1)
    async with (
        AsyncClient(transport=ASGITransport(app=app_a), base_url="http://a") as a,
        AsyncClient(transport=ASGITransport(app=app_b), base_url="http://b") as b,
    ):
        responses = await asyncio.gather(
            a.post("/api/runs", headers={"Idempotency-Key": KEY_A}, json=REQUEST),
            b.post("/api/runs", headers={"Idempotency-Key": KEY_B}, json=REQUEST),
        )
        health = (await b.get("/api/health")).json()["run_admission"]
        release.set()
        await asyncio.sleep(0)

    assert sorted(item.status_code for item in responses) == [202, 429]
    assert len(await store.list_snapshots(10)) == 1
    assert health["active_live_runs"] == 1
    assert health["recovery_required"] is False


@pytest.mark.asyncio
async def test_restart_reports_and_explicitly_recovers_expired_created_lease(
    tmp_path, monkeypatch
) -> None:
    release = asyncio.Event()

    async def hold_execute(self, *args, **kwargs):
        await release.wait()

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", hold_execute)
    path = tmp_path / "expired-recovery.sqlite3"
    app, _ = _app(path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://first"
    ) as client:
        created = await client.post(
            "/api/runs", headers={"Idempotency-Key": KEY_A}, json=REQUEST
        )
    run_id = created.json()["run_id"]
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE live_execution_leases SET expires_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run_id),
        )

    restarted, _ = _app(path)
    async with AsyncClient(
        transport=ASGITransport(app=restarted), base_url="http://restart"
    ) as client:
        health = (await client.get("/api/health")).json()["run_admission"]
        blocked_new = await client.post(
            "/api/runs", headers={"Idempotency-Key": KEY_B}, json=REQUEST
        )
        recovered = await client.post(f"/api/runs/{run_id}/resume")
        duplicate_resume = await client.post(f"/api/runs/{run_id}/resume")
        release.set()
        await asyncio.sleep(0)

    assert health["active_live_runs"] == 0
    assert health["expired_live_leases"] == 1
    assert health["recovery_required"] is True
    assert blocked_new.status_code == 409
    assert recovered.status_code == 202
    assert duplicate_resume.status_code == 409


@pytest.mark.asyncio
async def test_lease_status_contract_and_expired_owner_cannot_renew(
    tmp_path, monkeypatch
) -> None:
    release = asyncio.Event()

    async def hold_execute(self, *args, **kwargs):
        await release.wait()

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", hold_execute)
    path = tmp_path / "lease-states.sqlite3"
    app, store = _app(path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/runs", headers={"Idempotency-Key": KEY_A}, json=REQUEST
        )
    run_id = created.json()["run_id"]
    now = utc_now()
    active = await store.live_execution_lease_status(run_id, now=now)
    assert active is not None and active["state"] == "active"
    with sqlite3.connect(path) as connection:
        token = connection.execute(
            "SELECT lease_token FROM live_execution_leases WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE live_execution_leases SET expires_at = ? WHERE run_id = ?",
            ((now - timedelta(seconds=1)).isoformat(), run_id),
        )
    expired = await store.live_execution_lease_status(run_id, now=now)
    assert expired is not None and expired["state"] == "expired_recovery_required"
    assert await store.renew_live_execution_lease(
        run_id, lease_token=token, now=now, lease_ttl_seconds=900
    ) is False
    assert not await store.retire_expired_live_execution_lease(
        run_id, expected_stage="human_gate", now=now
    )
    assert await store.retire_expired_live_execution_lease(
        run_id, expected_stage="created", now=now
    )
    released = await store.live_execution_lease_status(run_id, now=now)
    assert released is not None and released["state"] == "released"
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_stage", "with_human_decision"),
    [
        (RunStage.HUMAN_GATE, False),
        (RunStage.COMPLETED, False),
        (RunStage.BLOCKED, False),
        (RunStage.ABSTAINED, True),
    ],
)
async def test_expired_terminal_lease_cleanup_never_restarts_workflow(
    tmp_path, monkeypatch, terminal_stage, with_human_decision
) -> None:
    release = asyncio.Event()
    execute_calls = 0

    async def hold_execute(self, *args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        await release.wait()

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", hold_execute)
    path = tmp_path / f"terminal-reconcile-{terminal_stage.value}.sqlite3"
    app, store = _app(path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://first"
    ) as client:
        created = await client.post(
            "/api/runs", headers={"Idempotency-Key": KEY_A}, json=REQUEST
        )
    run_id = created.json()["run_id"]
    await asyncio.sleep(0)
    snapshot = await store.get_snapshot(run_id)
    assert snapshot is not None
    snapshot.stage = terminal_stage
    if with_human_decision:
        snapshot.human_decision = HumanGateDecision(
            run_id=run_id,
            decision="HOLD",
            researcher="lease cleanup test",
            rationale="Verify cleanup-only recovery for a persisted decision.",
        )
    await store.save_snapshot(snapshot)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE live_execution_leases SET expires_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run_id),
        )

    restarted, _ = _app(path)
    async with AsyncClient(
        transport=ASGITransport(app=restarted), base_url="http://restart"
    ) as client:
        status = (await client.get(f"/api/runs/{run_id}/resume-status")).json()
        reconciled = await client.post(f"/api/runs/{run_id}/resume")

    assert status["reason_code"] == "EXPIRED_TERMINAL_LEASE_RECONCILIATION_REQUIRED"
    assert status["execution_lease_state"] == "expired_recovery_required"
    assert reconciled.status_code == 202
    assert reconciled.json()["stage"] == terminal_stage.value
    lease = await store.live_execution_lease_status(run_id, now=utc_now())
    assert lease is not None and lease["state"] == "released"
    assert lease["release_reason"] == "expired_terminal_reconciled"
    assert execute_calls == 1
    async with AsyncClient(
        transport=ASGITransport(app=restarted), base_url="http://restart"
    ) as client:
        admitted = await client.post(
            "/api/runs", headers={"Idempotency-Key": KEY_B}, json=REQUEST
        )
    assert admitted.status_code == 202
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["return", "raise", "cancel"])
async def test_live_wrapper_finally_releases_lease(
    tmp_path, monkeypatch, outcome
) -> None:
    blocker = asyncio.Event()

    async def controlled_execute(self, *args, **kwargs):
        if outcome == "raise":
            raise RuntimeError("bounded test failure")
        if outcome == "cancel":
            await blocker.wait()

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", controlled_execute)
    path = tmp_path / f"finally-{outcome}.sqlite3"
    app, store = _app(path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/runs", headers={"Idempotency-Key": KEY_A}, json=REQUEST
        )
    run_id = created.json()["run_id"]
    await asyncio.sleep(0)
    if outcome == "cancel":
        matching = [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == f"drugpilot:{run_id}"
        ]
        assert len(matching) == 1
        matching[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await matching[0]
    else:
        for _ in range(20):
            lease = await store.live_execution_lease_status(run_id, now=utc_now())
            if lease is not None and lease["state"] == "released":
                break
            await asyncio.sleep(0.01)
    lease = await store.live_execution_lease_status(run_id, now=utc_now())
    assert lease is not None and lease["state"] == "released"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [RunMode.MOCK, RunMode.HYBRID])
async def test_mock_and_hybrid_api_clients_remain_compatible_without_a_key(
    tmp_path, monkeypatch, mode
) -> None:
    async def no_execute(self, *args, **kwargs):
        return None

    monkeypatch.setattr(DrugPilotOrchestrator, "_execute", no_execute)
    app, _ = _app(tmp_path / f"{mode.value}.sqlite3", mode=mode)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/runs", json=REQUEST)
        health = (await client.get("/api/health")).json()

    assert created.status_code == 202
    assert health["run_admission"]["idempotency_required"] is False


def test_browser_reuses_one_web_crypto_uuid_for_an_ambiguous_submission() -> None:
    script = (
        Path(__file__).parents[2]
        / "src"
        / "drugpilot"
        / "web"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "globalThis.crypto" in script
    assert "cryptoApi.randomUUID()" in script
    assert "cryptoApi.getRandomValues(new Uint8Array(16))" in script
    assert "createSubmissionForBody(requestBody)" in script
    assert "state.createSubmission?.body === body" in script
    assert "sessionStorage.getItem(pendingSubmissionStorageKey)" in script
    assert "sessionStorage.setItem(" in script
    assert "pendingSubmissionLimit = 8" in script
    assert "forgetCreateSubmission(submission)" in script
    assert 'headers: { "Idempotency-Key": submission.idempotencyKey }' in script
    assert "![408, 409, 429].includes(error.status)" in script
