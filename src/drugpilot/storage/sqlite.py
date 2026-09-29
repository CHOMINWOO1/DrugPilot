from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from drugpilot.domain.models import (
    ExecutionEvent,
    RunSnapshot,
    RunStage,
    TokenUsage,
    WorkflowCheckpoint,
    new_id,
    utc_now,
)


class CheckpointError(RuntimeError):
    """Base class for a checkpoint that cannot be trusted or resumed."""


class CheckpointCompatibilityError(CheckpointError):
    """The persisted checkpoint uses an unsupported state/workflow schema."""


class CheckpointIntegrityError(CheckpointError):
    """The persisted checkpoint, artifact bindings, or audit binding is invalid."""


class RunAdmissionExists(RuntimeError):
    """An idempotency key was already committed to a durable run."""

    def __init__(self, *, run_id: str, request_sha256: str) -> None:
        super().__init__("run admission already exists")
        self.run_id = run_id
        self.request_sha256 = request_sha256


class RunAdmissionLimitExceeded(RuntimeError):
    """A durable hourly, active-execution, or recovery guard rejected a run."""

    def __init__(
        self,
        reason_code: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.retry_after_seconds = retry_after_seconds


class LiveExecutionLeaseConflict(ValueError):
    """A resume could not atomically acquire the durable paid-run lease."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class RunCancellationRequested(ValueError):
    """Durable cancellation fences every later scientific write/dispatch."""


_CANCEL_STAGES = {RunStage.CANCEL_REQUESTED.value, RunStage.CANCELLED.value}
_ACTIVE_STAGES = {stage.value for stage in (
    RunStage.CREATED, RunStage.BRIEF_VALIDATED, RunStage.MA1, RunStage.G0,
    RunStage.PLANNER, RunStage.MA2, RunStage.G1,
)}


class SQLiteRunStore:
    """Small local checkpoint, artifact and append-only audit store.

    SQLite is deliberately sufficient for the local web demonstration.  The
    interface keeps persistence separate from orchestration so a managed store
    can replace it later without changing any agent.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS audit_ledger (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    previous_hash TEXT,
                    entry_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_audit_run_sequence
                    ON audit_ledger(run_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_artifacts_run_type
                    ON artifacts(run_id, artifact_type);

                CREATE TABLE IF NOT EXISTS stage_checkpoints (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    checkpoint_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    completed_node TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    workflow_version TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_sequence
                    ON stage_checkpoints(run_id, sequence);

                CREATE TABLE IF NOT EXISTS run_admissions (
                    idempotency_key TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL
                        CHECK(length(request_sha256) = 64),
                    run_id TEXT NOT NULL UNIQUE,
                    admitted_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_run_admissions_admitted_at
                    ON run_admissions(admitted_at);

                CREATE TABLE IF NOT EXISTS live_execution_leases (
                    run_id TEXT PRIMARY KEY,
                    lease_token TEXT NOT NULL UNIQUE,
                    lease_state TEXT NOT NULL
                        CHECK(lease_state IN ('active', 'released')),
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    released_at TEXT,
                    release_reason TEXT,
                    generation INTEGER NOT NULL CHECK(generation >= 1),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_live_leases_state_expiry
                    ON live_execution_leases(lease_state, expires_at);
                """
            )

    async def save_snapshot(self, snapshot: RunSnapshot) -> None:
        snapshot.updated_at = utc_now()
        payload = snapshot.model_dump_json()
        async with self._write_lock:
            await asyncio.to_thread(self._save_snapshot_sync, snapshot, payload)

    def _save_snapshot_sync(self, snapshot: RunSnapshot, payload: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_not_cancelled(connection, snapshot.run_id)
            connection.execute(
                """
                INSERT INTO runs(run_id, stage, snapshot_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    stage=excluded.stage,
                    snapshot_json=excluded.snapshot_json,
                    updated_at=excluded.updated_at
                """,
                (
                    snapshot.run_id,
                    snapshot.stage.value,
                    payload,
                    snapshot.created_at.isoformat(),
                    snapshot.updated_at.isoformat(),
                ),
            )

    @staticmethod
    def _ensure_not_cancelled(connection, run_id: str, *, allow_pending=False):
        row = connection.execute("SELECT stage FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row and row["stage"] in _CANCEL_STAGES:
            if allow_pending and row["stage"] == RunStage.CANCEL_REQUESTED.value:
                return
            raise RunCancellationRequested("RUN_CANCELLED: this run cannot be modified or resumed")

    async def append_token_usage(self, run_id: str, usage: TokenUsage) -> RunSnapshot:
        """Meter a completed in-flight call without overwriting cancellation."""
        def commit():
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._ensure_not_cancelled(connection, run_id, allow_pending=True)
                row = connection.execute("SELECT snapshot_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                snapshot = RunSnapshot.model_validate_json(row["snapshot_json"])
                snapshot.token_usage.append(usage)
                snapshot.updated_at = utc_now()
                connection.execute("UPDATE runs SET snapshot_json=?,updated_at=? WHERE run_id=?",
                    (snapshot.model_dump_json(), snapshot.updated_at.isoformat(), run_id))
                return snapshot
        async with self._write_lock:
            return await asyncio.to_thread(commit)

    async def pending_cancellation_run_ids(self) -> list[str]:
        def read():
            with self._connect() as connection:
                return [row["run_id"] for row in connection.execute(
                    "SELECT run_id FROM runs WHERE stage=?", (RunStage.CANCEL_REQUESTED.value,))]
        return await asyncio.to_thread(read)

    async def set_cancellation(self, run_id: str, *, finished=False,
                               lease_token: str | None = None,
                               interrupted=False) -> tuple[RunSnapshot, ExecutionEvent | None]:
        """Commit cancellation and its audit event atomically; never erase costs."""
        def commit():
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute("SELECT snapshot_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                snapshot = RunSnapshot.model_validate_json(row["snapshot_json"])
                if snapshot.stage is RunStage.CANCELLED or (not finished and snapshot.stage is RunStage.CANCEL_REQUESTED):
                    return snapshot, None
                lease = connection.execute("SELECT lease_token,lease_state,expires_at FROM live_execution_leases WHERE run_id=?", (run_id,)).fetchone()
                if lease and lease["lease_state"] == "active":
                    if interrupted:
                        if lease["expires_at"] > utc_now().isoformat():
                            raise ValueError("RUN_ACTIVE: cancellation owner has not expired")
                    elif lease["lease_token"] != lease_token:
                        raise ValueError("RUN_NOT_OWNED_BY_THIS_SERVER")
                if finished:
                    if snapshot.stage is not RunStage.CANCEL_REQUESTED:
                        raise ValueError("CANCELLATION_NOT_REQUESTED")
                else:
                    if snapshot.stage.value not in _ACTIVE_STAGES and not any(r.status == "running" for r in snapshot.g1_reaudits):
                        raise ValueError("RUN_NOT_ACTIVE")
                    if lease and lease["lease_state"] == "active" and (
                            lease["lease_token"] != lease_token or lease["expires_at"] <= utc_now().isoformat()):
                        raise ValueError("RUN_NOT_OWNED_BY_THIS_SERVER")
                now = utc_now()
                snapshot.stage = RunStage.CANCELLED if finished else RunStage.CANCEL_REQUESTED
                snapshot.cancellation_requested_at = snapshot.cancellation_requested_at or now
                if finished:
                    snapshot.cancelled_at = now
                snapshot.status_message = (
                    "사용자 요청으로 실행이 중단되었습니다. 이미 발생한 호출 비용과 결과는 보존됩니다."
                    if finished else "중단 요청 접수 · 진행 중인 호출이 종료되면 중단됩니다. 해당 호출 비용은 발생할 수 있습니다."
                )
                snapshot.updated_at = now
                artifacts = {}
                if finished:
                    for index, record in enumerate(snapshot.g1_reaudits):
                        if record.status != "running":
                            continue
                        closed = record.model_copy(update={"record_id": new_id("review_record"),
                            "status": "failed", "error_code": "INTERRUPTED_CANCELLATION_ACCOUNTING_UNCERTAIN" if interrupted else "USER_CANCELLED",
                            "finished_at": now, "usage_end_index": len(snapshot.token_usage)})
                        encoded_record = closed.model_dump_json()
                        record_hash = hashlib.sha256(encoded_record.encode()).hexdigest()
                        connection.execute("INSERT INTO artifacts(artifact_id,run_id,artifact_type,schema_version,sha256,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
                            (closed.record_id, run_id, "G1ReauditRecord", "1.0", record_hash, encoded_record, now.isoformat()))
                        artifacts[closed.record_id] = record_hash
                        snapshot.g1_reaudits[index] = closed
                if interrupted:
                    snapshot.status_message += " 서버 중단 시점의 외부 호출은 사용량 확인이 필요합니다."
                    if lease and lease["lease_state"] == "active":
                        connection.execute("UPDATE live_execution_leases SET lease_state='released',released_at=?,heartbeat_at=?,release_reason='interrupted_user_cancellation' WHERE run_id=?",
                            (now.isoformat(), now.isoformat(), run_id))
                event = ExecutionEvent(run_id=run_id, stage=snapshot.stage,
                    event_type="workflow_cancelled" if finished else "cancellation_requested",
                    message=snapshot.status_message, node="orchestrator.cancel",
                    artifact_ids=list(artifacts),
                    metadata={"cooperative": True, "in_flight_call_may_be_billed": True,
                              "artifacts_preserved": True, "usage_preserved": True,
                              "artifact_sha256": artifacts, "interrupted_recovery": interrupted,
                              "usage_accounting_complete": bool(finished and not interrupted
                                  and not any(u.estimated for u in snapshot.token_usage))})
                # Include any event committed just before the cancellation fence.
                rows = connection.execute("SELECT event_json,entry_hash FROM audit_ledger WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()
                snapshot.events = [ExecutionEvent.model_validate_json(r["event_json"]) for r in rows]
                previous = rows[-1]["entry_hash"] if rows else "GENESIS"
                encoded = event.model_dump_json()
                entry_hash = hashlib.sha256(f"{previous}|{encoded}".encode()).hexdigest()
                connection.execute("INSERT INTO audit_ledger(event_id,run_id,event_json,previous_hash,entry_hash,created_at) VALUES(?,?,?,?,?,?)",
                    (event.event_id, run_id, encoded, previous, entry_hash, event.created_at.isoformat()))
                snapshot.events.append(event)
                connection.execute("UPDATE runs SET stage=?,snapshot_json=?,updated_at=? WHERE run_id=?",
                    (snapshot.stage.value, snapshot.model_dump_json(), now.isoformat(), run_id))
                return snapshot, event
        async with self._write_lock:
            return await asyncio.to_thread(commit)

    async def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        row = await asyncio.to_thread(self._get_run_row, run_id)
        return RunSnapshot.model_validate_json(row["snapshot_json"]) if row else None

    def _get_run_row(self, run_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT snapshot_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()

    async def list_snapshots(self, limit: int = 50) -> list[RunSnapshot]:
        rows = await asyncio.to_thread(self._list_run_rows, limit)
        return [RunSnapshot.model_validate_json(row["snapshot_json"]) for row in rows]

    async def list_snapshot_summaries(
        self, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return list-page fields without loading multi-megabyte snapshots.

        The complete immutable snapshot remains available from ``get_snapshot``.
        SQLite JSON extraction keeps the sidebar query bounded to the small
        fields that it renders, even when candidate evidence and event ledgers
        make an individual run many megabytes large.
        """

        rows = await asyncio.to_thread(self._list_run_summary_rows, limit)
        summaries: list[dict[str, Any]] = []
        for row in rows:
            brief = self._decode_summary_json(row["brief_json"], {})
            intake = self._decode_summary_json(
                row["intake_validation_json"], {}
            )
            resolved_disease = self._decode_summary_json(
                row["resolved_disease_json"], None
            )
            source_statuses = self._decode_summary_json(
                row["target_source_statuses_json"], []
            )
            ma1_metadata = {
                "intake_validation": intake,
                "resolved_disease": resolved_disease,
                "target_discovery_enabled": self._summary_bool(
                    row["target_discovery_enabled"]
                ),
                "external_target_seed_count": row["external_target_seed_count"],
                "target_fallback_used": self._summary_bool(
                    row["target_fallback_used"]
                ),
                "target_discovery_error": row["target_discovery_error"],
                "target_source_statuses": source_statuses,
            }
            summaries.append({
                "run_id": row["run_id"],
                "mode": row["mode"],
                "stage": row["stage"],
                "status_message": row["status_message"],
                "brief": brief,
                "hypothesis_packet": {"metadata": ma1_metadata},
                "g0_verdict": (
                    {"verdict": row["g0_verdict"]}
                    if row["g0_verdict"] is not None else None
                ),
                "g1_verdict": (
                    {"verdict": row["g1_verdict"]}
                    if row["g1_verdict"] is not None else None
                ),
                "human_decision": (
                    {"decision": row["human_decision"]}
                    if row["human_decision"] is not None else None
                ),
                "rework_loops": int(row["rework_loops"] or 0),
                "token_usage_count": int(row["token_usage_count"] or 0),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "error": row["error"],
                "list_summary": True,
            })
        return summaries

    @staticmethod
    def _decode_summary_json(value: Any, default: Any) -> Any:
        if not isinstance(value, str):
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _summary_bool(value: Any) -> bool | None:
        return None if value is None else bool(value)

    def _list_run_rows(self, limit: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT snapshot_json FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()

    def _list_run_summary_rows(self, limit: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    run_id,
                    stage,
                    created_at,
                    updated_at,
                    json_extract(snapshot_json, '$.mode') AS mode,
                    json_extract(snapshot_json, '$.status_message')
                        AS status_message,
                    json_extract(snapshot_json, '$.brief') AS brief_json,
                    json_extract(
                        snapshot_json,
                        '$.hypothesis_packet.metadata.intake_validation'
                    ) AS intake_validation_json,
                    json_extract(
                        snapshot_json,
                        '$.hypothesis_packet.metadata.resolved_disease'
                    ) AS resolved_disease_json,
                    json_extract(
                        snapshot_json,
                        '$.hypothesis_packet.metadata.target_discovery_enabled'
                    ) AS target_discovery_enabled,
                    json_extract(
                        snapshot_json,
                        '$.hypothesis_packet.metadata.external_target_seed_count'
                    ) AS external_target_seed_count,
                    json_extract(
                        snapshot_json,
                        '$.hypothesis_packet.metadata.target_fallback_used'
                    ) AS target_fallback_used,
                    json_extract(
                        snapshot_json,
                        '$.hypothesis_packet.metadata.target_discovery_error'
                    ) AS target_discovery_error,
                    json_extract(
                        snapshot_json,
                        '$.hypothesis_packet.metadata.target_source_statuses'
                    ) AS target_source_statuses_json,
                    json_extract(snapshot_json, '$.g0_verdict.verdict')
                        AS g0_verdict,
                    json_extract(snapshot_json, '$.g1_verdict.verdict')
                        AS g1_verdict,
                    json_extract(snapshot_json, '$.human_decision.decision')
                        AS human_decision,
                    json_extract(snapshot_json, '$.rework_loops')
                        AS rework_loops,
                    json_array_length(snapshot_json, '$.token_usage')
                        AS token_usage_count,
                    json_extract(snapshot_json, '$.error') AS error
                FROM runs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    async def get_run_admission(
        self, idempotency_key: str
    ) -> dict[str, str] | None:
        """Return one durable request-to-run mapping without mutating it."""

        row = await asyncio.to_thread(
            self._get_run_admission_row, idempotency_key
        )
        return dict(row) if row is not None else None

    def _get_run_admission_row(
        self, idempotency_key: str
    ) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT idempotency_key, request_sha256, run_id, admitted_at
                FROM run_admissions WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

    async def count_runs_created_since(self, since: datetime) -> int:
        """Count successful run admissions, including legacy unkeyed runs."""

        return await asyncio.to_thread(
            self._count_runs_created_since_sync, since.isoformat()
        )

    def _count_runs_created_since_sync(self, since_iso: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE created_at >= ?",
                (since_iso,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    async def commit_admitted_run(
        self,
        snapshot: RunSnapshot,
        *,
        artifacts: list[tuple[str, str, Any, str]],
        event: ExecutionEvent,
        idempotency_key: str | None,
        request_sha256: str | None,
        admitted_at: datetime,
        live_execution: bool,
        lease_token: str | None,
        lease_ttl_seconds: int,
        max_active_live_runs: int,
        max_new_runs_per_hour: int,
    ) -> dict[str, str]:
        """Atomically persist a live run and its idempotency mapping.

        The workflow task must be started only after this transaction returns.
        Therefore an ambiguous HTTP response can always be resolved by reading
        the key mapping, while a process crash before task creation has made no
        paid model or scientific call and remains explicitly resumable.
        """

        if (idempotency_key is None) != (request_sha256 is None):
            raise ValueError(
                "idempotency key and request digest must be supplied together"
            )
        if live_execution != (lease_token is not None):
            raise ValueError(
                "live admission requires exactly one execution lease token"
            )

        prepared: list[tuple[str, str, str, str, str]] = []
        artifact_digests: dict[str, str] = {}
        for artifact_id, artifact_type, payload, schema_version in artifacts:
            if hasattr(payload, "model_dump_json"):
                encoded = payload.model_dump_json()
            else:
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if artifact_id in artifact_digests:
                raise CheckpointIntegrityError(
                    "initial run contains a duplicate artifact identifier"
                )
            artifact_digests[artifact_id] = digest
            prepared.append(
                (artifact_id, artifact_type, schema_version, digest, encoded)
            )

        if event.run_id != snapshot.run_id:
            raise CheckpointIntegrityError(
                "run admission event does not match the snapshot"
            )
        if set(event.artifact_ids) != set(artifact_digests):
            raise CheckpointIntegrityError(
                "run admission event artifact IDs do not match"
            )
        if event.metadata.get("artifact_sha256") != artifact_digests:
            raise CheckpointIntegrityError(
                "run admission event artifact digests do not match"
            )
        if not any(item.event_id == event.event_id for item in snapshot.events):
            raise CheckpointIntegrityError(
                "run admission snapshot does not contain its audit event"
            )

        snapshot.updated_at = utc_now()
        snapshot_json = snapshot.model_dump_json()
        event_json = event.model_dump_json()
        async with self._write_lock:
            await asyncio.to_thread(
                self._commit_admitted_run_sync,
                snapshot,
                snapshot_json,
                prepared,
                event,
                event_json,
                idempotency_key,
                request_sha256,
                admitted_at.isoformat(),
                live_execution,
                lease_token,
                (admitted_at + timedelta(seconds=lease_ttl_seconds)).isoformat(),
                (admitted_at - timedelta(hours=1)).isoformat(),
                max_active_live_runs,
                max_new_runs_per_hour,
            )
        return artifact_digests

    def _commit_admitted_run_sync(
        self,
        snapshot: RunSnapshot,
        snapshot_json: str,
        artifacts: list[tuple[str, str, str, str, str]],
        event: ExecutionEvent,
        event_json: str,
        idempotency_key: str | None,
        request_sha256: str | None,
        admitted_at_iso: str,
        live_execution: bool,
        lease_token: str | None,
        lease_expires_at_iso: str,
        hourly_since_iso: str,
        max_active_live_runs: int,
        max_new_runs_per_hour: int,
    ) -> None:
        with self._connect() as connection:
            # Serialize admission across processes as well as relying on the
            # API's lighter in-process lock.  The UNIQUE key remains the final
            # authority if a second worker reaches this transaction.
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key is not None:
                existing = connection.execute(
                    """
                    SELECT request_sha256, run_id FROM run_admissions
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    raise RunAdmissionExists(
                        run_id=str(existing["run_id"]),
                        request_sha256=str(existing["request_sha256"]),
                    )

            if live_execution:
                lease_counts = connection.execute(
                    """
                    SELECT
                        SUM(CASE WHEN expires_at > ? THEN 1 ELSE 0 END) AS active,
                        SUM(CASE WHEN expires_at <= ? THEN 1 ELSE 0 END) AS expired
                    FROM live_execution_leases
                    WHERE lease_state = 'active'
                    """,
                    (admitted_at_iso, admitted_at_iso),
                ).fetchone()
                active_count = int(lease_counts["active"] or 0)
                expired_count = int(lease_counts["expired"] or 0)
                if expired_count:
                    raise RunAdmissionLimitExceeded(
                        "LIVE_LEASE_RECOVERY_REQUIRED"
                    )
                if active_count >= max_active_live_runs:
                    raise RunAdmissionLimitExceeded(
                        "ACTIVE_LIVE_RUN_CAP",
                        retry_after_seconds=30,
                    )

            hourly = connection.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE created_at >= ?",
                (hourly_since_iso,),
            ).fetchone()
            if int(hourly["count"] if hourly is not None else 0) >= max_new_runs_per_hour:
                raise RunAdmissionLimitExceeded(
                    "HOURLY_RUN_CAP",
                    retry_after_seconds=3_600,
                )
            if connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (snapshot.run_id,)
            ).fetchone() is not None:
                raise CheckpointIntegrityError(
                    "new run identifier already exists"
                )

            connection.execute(
                """
                INSERT INTO runs(run_id, stage, snapshot_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot.run_id,
                    snapshot.stage.value,
                    snapshot_json,
                    snapshot.created_at.isoformat(),
                    snapshot.updated_at.isoformat(),
                ),
            )
            for artifact_id, artifact_type, schema_version, digest, encoded in artifacts:
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        artifact_id, run_id, artifact_type, schema_version,
                        sha256, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        snapshot.run_id,
                        artifact_type,
                        schema_version,
                        digest,
                        encoded,
                        admitted_at_iso,
                    ),
                )

            entry_hash = hashlib.sha256(
                f"GENESIS|{event_json}".encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO audit_ledger(
                    event_id, run_id, event_json, previous_hash,
                    entry_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    snapshot.run_id,
                    event_json,
                    "GENESIS",
                    entry_hash,
                    event.created_at.isoformat(),
                ),
            )
            if idempotency_key is not None and request_sha256 is not None:
                connection.execute(
                    """
                    INSERT INTO run_admissions(
                        idempotency_key, request_sha256, run_id, admitted_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        idempotency_key,
                        request_sha256,
                        snapshot.run_id,
                        admitted_at_iso,
                    ),
                )
            if live_execution and lease_token is not None:
                connection.execute(
                    """
                    INSERT INTO live_execution_leases(
                        run_id, lease_token, lease_state, acquired_at,
                        heartbeat_at, expires_at, released_at,
                        release_reason, generation
                    ) VALUES (?, ?, 'active', ?, ?, ?, NULL, NULL, 1)
                    """,
                    (
                        snapshot.run_id,
                        lease_token,
                        admitted_at_iso,
                        admitted_at_iso,
                        lease_expires_at_iso,
                    ),
                )

    async def admission_health(
        self,
        *,
        now: datetime,
        hourly_since: datetime,
    ) -> dict[str, int | bool]:
        """Return one DB-backed admission snapshot without exposing tokens."""

        return await asyncio.to_thread(
            self._admission_health_sync,
            now.isoformat(),
            hourly_since.isoformat(),
        )

    def _admission_health_sync(
        self,
        now_iso: str,
        hourly_since_iso: str,
    ) -> dict[str, int | bool]:
        with self._connect() as connection:
            connection.execute("BEGIN")
            hourly = connection.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE created_at >= ?",
                (hourly_since_iso,),
            ).fetchone()
            leases = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN expires_at > ? THEN 1 ELSE 0 END) AS active,
                    SUM(CASE WHEN expires_at <= ? THEN 1 ELSE 0 END) AS expired
                FROM live_execution_leases
                WHERE lease_state = 'active'
                """,
                (now_iso, now_iso),
            ).fetchone()
        active = int(leases["active"] or 0)
        expired = int(leases["expired"] or 0)
        return {
            "hourly_admitted_count": int(hourly["count"] or 0),
            "active_live_runs": active,
            "expired_live_leases": expired,
            "recovery_required": bool(expired),
        }

    async def live_execution_lease_status(
        self,
        run_id: str,
        *,
        now: datetime,
    ) -> dict[str, Any] | None:
        row = await asyncio.to_thread(self._live_lease_row, run_id)
        if row is None:
            return None
        payload = dict(row)
        if payload["lease_state"] == "released":
            derived_state = "released"
        elif str(payload["expires_at"]) <= now.isoformat():
            derived_state = "expired_recovery_required"
        else:
            derived_state = "active"
        return {
            "run_id": str(payload["run_id"]),
            "state": derived_state,
            "acquired_at": str(payload["acquired_at"]),
            "heartbeat_at": str(payload["heartbeat_at"]),
            "expires_at": str(payload["expires_at"]),
            "released_at": payload["released_at"],
            "release_reason": payload["release_reason"],
            "generation": int(payload["generation"]),
        }

    def _live_lease_row(self, run_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT run_id, lease_state, acquired_at, heartbeat_at,
                       expires_at, released_at, release_reason, generation
                FROM live_execution_leases WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

    async def claim_live_execution_lease(
        self,
        run_id: str,
        *,
        lease_token: str,
        now: datetime,
        lease_ttl_seconds: int,
        max_active_live_runs: int,
    ) -> dict[str, Any]:
        """Atomically claim one resume lease without exceeding global capacity."""

        async with self._write_lock:
            return await asyncio.to_thread(
                self._claim_live_execution_lease_sync,
                run_id,
                lease_token,
                now.isoformat(),
                (now + timedelta(seconds=lease_ttl_seconds)).isoformat(),
                max_active_live_runs,
            )

    def _claim_live_execution_lease_sync(
        self,
        run_id: str,
        lease_token: str,
        now_iso: str,
        expires_at_iso: str,
        max_active_live_runs: int,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_not_cancelled(connection, run_id)
            if connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone() is None:
                raise KeyError(run_id)
            existing = connection.execute(
                """
                SELECT lease_state, expires_at, generation
                FROM live_execution_leases WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if (
                existing is not None
                and existing["lease_state"] == "active"
                and str(existing["expires_at"]) > now_iso
            ):
                raise LiveExecutionLeaseConflict("RUN_ACTIVE")

            other_expired = connection.execute(
                """
                SELECT COUNT(*) AS count FROM live_execution_leases
                WHERE lease_state = 'active' AND expires_at <= ? AND run_id != ?
                """,
                (now_iso, run_id),
            ).fetchone()
            if int(other_expired["count"] or 0):
                raise LiveExecutionLeaseConflict(
                    "OTHER_RUN_LEASE_RECOVERY_REQUIRED"
                )
            other_active = connection.execute(
                """
                SELECT COUNT(*) AS count FROM live_execution_leases
                WHERE lease_state = 'active' AND expires_at > ? AND run_id != ?
                """,
                (now_iso, run_id),
            ).fetchone()
            if int(other_active["count"] or 0) >= max_active_live_runs:
                raise LiveExecutionLeaseConflict("ACTIVE_LIVE_RUN_CAP")

            recovered = bool(
                existing is not None
                and existing["lease_state"] == "active"
                and str(existing["expires_at"]) <= now_iso
            )
            if existing is None:
                generation = 1
                connection.execute(
                    """
                    INSERT INTO live_execution_leases(
                        run_id, lease_token, lease_state, acquired_at,
                        heartbeat_at, expires_at, released_at,
                        release_reason, generation
                    ) VALUES (?, ?, 'active', ?, ?, ?, NULL, NULL, ?)
                    """,
                    (
                        run_id,
                        lease_token,
                        now_iso,
                        now_iso,
                        expires_at_iso,
                        generation,
                    ),
                )
            else:
                generation = int(existing["generation"]) + 1
                connection.execute(
                    """
                    UPDATE live_execution_leases
                    SET lease_token = ?, lease_state = 'active', acquired_at = ?,
                        heartbeat_at = ?, expires_at = ?, released_at = NULL,
                        release_reason = NULL, generation = ?
                    WHERE run_id = ?
                    """,
                    (
                        lease_token,
                        now_iso,
                        now_iso,
                        expires_at_iso,
                        generation,
                        run_id,
                    ),
                )
        return {
            "run_id": run_id,
            "generation": generation,
            "recovered_expired_lease": recovered,
        }

    async def renew_live_execution_lease(
        self,
        run_id: str,
        *,
        lease_token: str,
        now: datetime,
        lease_ttl_seconds: int,
    ) -> bool:
        async with self._write_lock:
            return await asyncio.to_thread(
                self._renew_live_execution_lease_sync,
                run_id,
                lease_token,
                now.isoformat(),
                (now + timedelta(seconds=lease_ttl_seconds)).isoformat(),
            )

    def _renew_live_execution_lease_sync(
        self,
        run_id: str,
        lease_token: str,
        now_iso: str,
        expires_at_iso: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE live_execution_leases
                SET heartbeat_at = ?, expires_at = ?
                WHERE run_id = ? AND lease_token = ? AND lease_state = 'active'
                  AND expires_at > ?
                """,
                (now_iso, expires_at_iso, run_id, lease_token, now_iso),
            )
        return cursor.rowcount == 1

    async def expire_live_execution_lease_for_recovery(
        self,
        run_id: str,
        *,
        lease_token: str,
        reason: str,
        now: datetime,
    ) -> bool:
        """Fail closed after admission committed but workflow was not registered."""

        async with self._write_lock:
            return await asyncio.to_thread(
                self._expire_live_execution_lease_for_recovery_sync,
                run_id,
                lease_token,
                reason,
                now.isoformat(),
            )

    def _expire_live_execution_lease_for_recovery_sync(
        self, run_id: str, lease_token: str, reason: str, now_iso: str
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE live_execution_leases
                SET expires_at = ?, heartbeat_at = ?, release_reason = ?
                WHERE run_id = ? AND lease_token = ? AND lease_state = 'active'
                """,
                (now_iso, now_iso, reason[:120], run_id, lease_token),
            )
        return cursor.rowcount == 1

    async def retire_expired_live_execution_lease(
        self, run_id: str, *, expected_stage: str, now: datetime
    ) -> bool:
        """Reconcile a verified terminal run without starting paid work."""

        async with self._write_lock:
            return await asyncio.to_thread(
                self._retire_expired_live_execution_lease_sync,
                run_id,
                expected_stage,
                now.isoformat(),
            )

    def _retire_expired_live_execution_lease_sync(
        self, run_id: str, expected_stage: str, now_iso: str
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            persisted = connection.execute(
                "SELECT stage FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if persisted is None or str(persisted["stage"]) != expected_stage:
                return False
            cursor = connection.execute(
                """
                UPDATE live_execution_leases
                SET lease_state = 'released', released_at = ?, heartbeat_at = ?,
                    release_reason = 'expired_terminal_reconciled'
                WHERE run_id = ? AND lease_state = 'active' AND expires_at <= ?
                """,
                (now_iso, now_iso, run_id, now_iso),
            )
        return cursor.rowcount == 1

    async def release_live_execution_lease(
        self,
        run_id: str,
        *,
        lease_token: str,
        reason: str,
        now: datetime,
    ) -> bool:
        async with self._write_lock:
            return await asyncio.to_thread(
                self._release_live_execution_lease_sync,
                run_id,
                lease_token,
                reason,
                now.isoformat(),
            )

    def _release_live_execution_lease_sync(
        self,
        run_id: str,
        lease_token: str,
        reason: str,
        now_iso: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE live_execution_leases
                SET lease_state = 'released', released_at = ?,
                    release_reason = ?, heartbeat_at = ?
                WHERE run_id = ? AND lease_token = ? AND lease_state = 'active'
                """,
                (now_iso, reason[:120], now_iso, run_id, lease_token),
            )
        return cursor.rowcount == 1

    async def commit_review_update(
        self, snapshot: RunSnapshot, *, artifacts: list[tuple[str, str, Any]],
        events: list[ExecutionEvent], lease_token: str,
        checkpoint: WorkflowCheckpoint | None = None,
    ) -> None:
        """Commit a review request/result and its complete provenance atomically.

        No paid call may start before the running request is durably visible.
        No partial result may orphan artifacts or replace the previous verdict.
        """
        encoded_artifacts = []
        for artifact_id, artifact_type, payload in artifacts:
            encoded = payload.model_dump_json()
            encoded_artifacts.append((artifact_id, artifact_type, encoded,
                hashlib.sha256(encoded.encode("utf-8")).hexdigest()))
        snapshot.updated_at = utc_now()
        async with self._write_lock:
            await asyncio.to_thread(self._commit_review_update_sync, snapshot,
                snapshot.model_dump_json(), encoded_artifacts, events, lease_token, checkpoint)

    def _commit_review_update_sync(self, snapshot, snapshot_json, artifacts, events,
                                   lease_token, checkpoint):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_not_cancelled(connection, snapshot.run_id)
            lease = connection.execute(
                "SELECT lease_token, lease_state, expires_at FROM live_execution_leases WHERE run_id = ?",
                (snapshot.run_id,),
            ).fetchone()
            if (lease is None or lease["lease_token"] != lease_token
                    or lease["lease_state"] != "active" or lease["expires_at"] <= utc_now().isoformat()):
                raise CheckpointIntegrityError("review execution lease is not owned or expired")
            for artifact_id, artifact_type, encoded, digest in artifacts:
                existing = connection.execute(
                    "SELECT run_id, artifact_type, schema_version, sha256 FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
                if existing:
                    if tuple(existing) != (snapshot.run_id, artifact_type, "1.0", digest):
                        raise CheckpointIntegrityError("review artifact identifier is immutable")
                else:
                    connection.execute(
                        "INSERT INTO artifacts(artifact_id,run_id,artifact_type,schema_version,sha256,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
                        (artifact_id, snapshot.run_id, artifact_type, "1.0", digest, encoded, utc_now().isoformat()),
                    )
            manifest = {r["artifact_id"]: r["sha256"] for r in connection.execute(
                "SELECT artifact_id,sha256 FROM artifacts WHERE run_id = ?", (snapshot.run_id,))}
            for event in events:
                if event.run_id != snapshot.run_id or any(
                    manifest.get(i) != event.metadata.get("artifact_sha256", {}).get(i)
                    or i not in manifest for i in event.artifact_ids
                ):
                    raise CheckpointIntegrityError("review event artifact binding mismatch")
                encoded = event.model_dump_json()
                existing = connection.execute("SELECT event_json FROM audit_ledger WHERE event_id = ?",
                                              (event.event_id,)).fetchone()
                if existing:
                    if existing["event_json"] != encoded:
                        raise CheckpointIntegrityError("review event identifier is immutable")
                    continue
                previous = connection.execute(
                    "SELECT entry_hash FROM audit_ledger WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                    (snapshot.run_id,),
                ).fetchone()
                previous_hash = previous["entry_hash"] if previous else "GENESIS"
                entry_hash = hashlib.sha256(f"{previous_hash}|{encoded}".encode("utf-8")).hexdigest()
                connection.execute(
                    "INSERT INTO audit_ledger(event_id,run_id,event_json,previous_hash,entry_hash,created_at) VALUES(?,?,?,?,?,?)",
                    (event.event_id, snapshot.run_id, encoded, previous_hash, entry_hash, event.created_at.isoformat()),
                )
            if checkpoint:
                digest = self.checkpoint_digest(checkpoint)
                bound = next((e for e in events if e.event_type == "stage_checkpoint_committed"
                              and e.metadata.get("checkpoint_id") == checkpoint.checkpoint_id), None)
                if (checkpoint.run_id != snapshot.run_id or checkpoint.artifact_sha256 != manifest
                        or bound is None or bound.metadata.get("checkpoint_sha256") != digest
                        or bound.metadata.get("artifact_sha256") != manifest
                        or set(bound.artifact_ids) != set(manifest)
                        or bound.metadata.get("completed_node") != checkpoint.completed_node
                        or bound.metadata.get("checkpoint_schema_version") != checkpoint.schema_version
                        or bound.metadata.get("workflow_version") != checkpoint.workflow_version):
                    raise CheckpointIntegrityError("review checkpoint binding mismatch")
                connection.execute(
                    "INSERT INTO stage_checkpoints(checkpoint_id,run_id,completed_node,schema_version,workflow_version,sha256,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (checkpoint.checkpoint_id, snapshot.run_id, checkpoint.completed_node,
                     checkpoint.schema_version, checkpoint.workflow_version, digest,
                     checkpoint.model_dump_json(), checkpoint.created_at.isoformat()),
                )
            # Every newly inserted artifact must be bound, including request-only commits.
            bound_ids = {i for e in events for i in e.artifact_ids}
            if any(i not in bound_ids for i, _, _, _ in artifacts):
                raise CheckpointIntegrityError("unbound review artifact")
            cursor = connection.execute(
                "UPDATE runs SET stage = ?, snapshot_json = ?, updated_at = ? WHERE run_id = ?",
                (snapshot.stage.value, snapshot_json, snapshot.updated_at.isoformat(), snapshot.run_id),
            )
            if cursor.rowcount != 1:
                raise CheckpointIntegrityError("review run does not exist")

    async def read_artifact(self, run_id: str, artifact_id: str, artifact_type: str) -> dict | None:
        """Read one scoped artifact and verify its canonical stored payload."""
        def read():
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload_json, sha256 FROM artifacts WHERE run_id = ? AND artifact_id = ? AND artifact_type = ?",
                    (run_id, artifact_id, artifact_type),
                ).fetchone()
            if row is None:
                return None
            if hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest() != row["sha256"]:
                raise CheckpointIntegrityError("artifact payload digest mismatch")
            return json.loads(row["payload_json"])
        return await asyncio.to_thread(read)

    async def save_artifact(
        self,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        payload: Any,
        schema_version: str = "1.0",
    ) -> str:
        if hasattr(payload, "model_dump_json"):
            encoded = payload.model_dump_json()
        else:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        async with self._write_lock:
            return await asyncio.to_thread(
                self._save_artifact_sync,
                run_id,
                artifact_id,
                artifact_type,
                schema_version,
                digest,
                encoded,
            )

    async def commit_artifact_event(
        self, snapshot: RunSnapshot, artifact_id: str, artifact_type: str,
        payload: Any, event: ExecutionEvent,
    ) -> str:
        """Atomically bind an intermediate artifact, event and run snapshot.

        Used before paid dispatch and after each source interpretation, so a
        later preflight failure cannot leave unbound intermediate artifacts.
        It never repairs or rewrites previously persisted run history.
        """
        encoded = (payload.model_dump_json() if hasattr(payload, "model_dump_json")
                   else json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if event.run_id != snapshot.run_id or event.artifact_ids != [artifact_id]:
            raise CheckpointIntegrityError("intermediate event identity mismatch")
        event = event.model_copy(update={"metadata": {**event.metadata,
            "artifact_sha256": {artifact_id: digest}}}, deep=True)
        committed = snapshot.model_copy(deep=True)
        committed.events.append(event)
        committed.updated_at = utc_now()
        async with self._write_lock:
            await asyncio.to_thread(self._commit_artifact_event_sync, committed,
                artifact_id, artifact_type, encoded, digest, event)
        return digest

    def _commit_artifact_event_sync(self, snapshot, artifact_id, artifact_type,
                                    encoded, digest, event):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_not_cancelled(connection, snapshot.run_id)
            if connection.execute("SELECT 1 FROM runs WHERE run_id=?", (snapshot.run_id,)).fetchone() is None:
                raise KeyError(snapshot.run_id)
            existing = connection.execute(
                "SELECT run_id,artifact_type,schema_version,sha256 FROM artifacts WHERE artifact_id=?",
                (artifact_id,)).fetchone()
            if existing is not None and tuple(existing) != (snapshot.run_id, artifact_type, "1.0", digest):
                raise CheckpointIntegrityError("intermediate artifact identifier is immutable")
            if existing is None:
                connection.execute(
                    "INSERT INTO artifacts(artifact_id,run_id,artifact_type,schema_version,sha256,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
                    (artifact_id,snapshot.run_id,artifact_type,"1.0",digest,encoded,utc_now().isoformat()))
            if connection.execute("SELECT 1 FROM audit_ledger WHERE event_id=?", (event.event_id,)).fetchone():
                raise CheckpointIntegrityError("intermediate event identifier already committed")
            prior = connection.execute(
                "SELECT entry_hash FROM audit_ledger WHERE run_id=? ORDER BY sequence DESC LIMIT 1",
                (snapshot.run_id,)).fetchone()
            previous = prior["entry_hash"] if prior else "GENESIS"
            event_json = event.model_dump_json()
            entry_hash = hashlib.sha256(f"{previous}|{event_json}".encode()).hexdigest()
            connection.execute(
                "INSERT INTO audit_ledger(event_id,run_id,event_json,previous_hash,entry_hash,created_at) VALUES(?,?,?,?,?,?)",
                (event.event_id,snapshot.run_id,event_json,previous,entry_hash,event.created_at.isoformat()))
            connection.execute("UPDATE runs SET stage=?,snapshot_json=?,updated_at=? WHERE run_id=?",
                (snapshot.stage.value,snapshot.model_dump_json(),snapshot.updated_at.isoformat(),snapshot.run_id))

    def _save_artifact_sync(
        self,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        schema_version: str,
        digest: str,
        encoded: str,
    ) -> str:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_not_cancelled(connection, run_id)
            existing = connection.execute(
                "SELECT run_id, artifact_type, schema_version, sha256 FROM artifacts "
                "WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if existing is not None:
                matches = (
                    existing["run_id"] == run_id
                    and existing["artifact_type"] == artifact_type
                    and existing["schema_version"] == schema_version
                    and existing["sha256"] == digest
                )
                if not matches:
                    raise ValueError(
                        f"artifact {artifact_id} is immutable and cannot be replaced"
                    )
                return digest
            connection.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, run_id, artifact_type, schema_version,
                    sha256, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    run_id,
                    artifact_type,
                    schema_version,
                    digest,
                    encoded,
                    utc_now().isoformat(),
                ),
            )
        return digest

    @staticmethod
    def checkpoint_digest(checkpoint: WorkflowCheckpoint) -> str:
        encoded = checkpoint.model_dump_json()
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def commit_stage_checkpoint(
        self,
        checkpoint: WorkflowCheckpoint,
        event: ExecutionEvent,
        snapshot: RunSnapshot,
    ) -> str:
        """Atomically bind resumable state to artifacts, ledger, and snapshot.

        A completed node is resumable only after all four records commit in the
        same SQLite transaction.  Replaying the exact checkpoint/event pair is
        idempotent; any conflicting reuse of an identifier fails closed.
        """

        if not (
            checkpoint.run_id == event.run_id == snapshot.run_id
        ):
            raise CheckpointIntegrityError("checkpoint run binding does not match")
        encoded = checkpoint.model_dump_json()
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        metadata = event.metadata
        if (
            metadata.get("checkpoint_id") != checkpoint.checkpoint_id
            or metadata.get("checkpoint_sha256") != digest
            or metadata.get("checkpoint_schema_version")
            != checkpoint.schema_version
            or metadata.get("workflow_version") != checkpoint.workflow_version
            or metadata.get("completed_node") != checkpoint.completed_node
        ):
            raise CheckpointIntegrityError("checkpoint audit binding is incomplete")
        if set(event.artifact_ids) != set(checkpoint.artifact_sha256):
            raise CheckpointIntegrityError("checkpoint artifact IDs do not match")
        if metadata.get("artifact_sha256") != checkpoint.artifact_sha256:
            raise CheckpointIntegrityError("checkpoint artifact digests do not match")

        snapshot.updated_at = utc_now()
        snapshot_json = snapshot.model_dump_json()
        event_json = event.model_dump_json()
        async with self._write_lock:
            return await asyncio.to_thread(
                self._commit_stage_checkpoint_sync,
                checkpoint,
                encoded,
                digest,
                event,
                event_json,
                snapshot,
                snapshot_json,
            )

    def _commit_stage_checkpoint_sync(
        self,
        checkpoint: WorkflowCheckpoint,
        encoded: str,
        digest: str,
        event: ExecutionEvent,
        event_json: str,
        snapshot: RunSnapshot,
        snapshot_json: str,
    ) -> str:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_not_cancelled(connection, snapshot.run_id)
            run = connection.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (checkpoint.run_id,)
            ).fetchone()
            if run is None:
                raise CheckpointIntegrityError("checkpoint run does not exist")

            artifacts = {
                row["artifact_id"]: row["sha256"]
                for row in connection.execute(
                    "SELECT artifact_id, sha256 FROM artifacts WHERE run_id = ?",
                    (checkpoint.run_id,),
                ).fetchall()
            }
            for artifact_id, expected in checkpoint.artifact_sha256.items():
                if artifacts.get(artifact_id) != expected:
                    raise CheckpointIntegrityError(
                        f"checkpoint artifact binding failed for {artifact_id}"
                    )

            existing_checkpoint = connection.execute(
                """
                SELECT run_id, completed_node, schema_version, workflow_version,
                       sha256, payload_json
                FROM stage_checkpoints WHERE checkpoint_id = ?
                """,
                (checkpoint.checkpoint_id,),
            ).fetchone()
            if existing_checkpoint is None:
                connection.execute(
                    """
                    INSERT INTO stage_checkpoints(
                        checkpoint_id, run_id, completed_node, schema_version,
                        workflow_version, sha256, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.run_id,
                        checkpoint.completed_node,
                        checkpoint.schema_version,
                        checkpoint.workflow_version,
                        digest,
                        encoded,
                        checkpoint.created_at.isoformat(),
                    ),
                )
            else:
                matches = (
                    existing_checkpoint["run_id"] == checkpoint.run_id
                    and existing_checkpoint["completed_node"]
                    == checkpoint.completed_node
                    and existing_checkpoint["schema_version"]
                    == checkpoint.schema_version
                    and existing_checkpoint["workflow_version"]
                    == checkpoint.workflow_version
                    and existing_checkpoint["sha256"] == digest
                    and existing_checkpoint["payload_json"] == encoded
                )
                if not matches:
                    raise CheckpointIntegrityError(
                        "checkpoint identifier is immutable and cannot be replaced"
                    )

            existing_event = connection.execute(
                "SELECT event_json, entry_hash FROM audit_ledger WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if existing_event is None:
                previous = connection.execute(
                    """
                    SELECT entry_hash FROM audit_ledger
                    WHERE run_id = ? ORDER BY sequence DESC LIMIT 1
                    """,
                    (event.run_id,),
                ).fetchone()
                previous_hash = previous["entry_hash"] if previous else "GENESIS"
                entry_hash = hashlib.sha256(
                    f"{previous_hash}|{event_json}".encode("utf-8")
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO audit_ledger(
                        event_id, run_id, event_json, previous_hash,
                        entry_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.run_id,
                        event_json,
                        previous_hash,
                        entry_hash,
                        event.created_at.isoformat(),
                    ),
                )
            elif existing_event["event_json"] != event_json:
                raise CheckpointIntegrityError(
                    "checkpoint audit event is immutable and cannot be replaced"
                )

            connection.execute(
                """
                UPDATE runs
                SET stage = ?, snapshot_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    snapshot.stage.value,
                    snapshot_json,
                    snapshot.updated_at.isoformat(),
                    snapshot.run_id,
                ),
            )
        return digest

    async def latest_stage_checkpoint(
        self,
        run_id: str,
        *,
        schema_version: str = "1.0",
        workflow_version: str = "drugpilot-stage-workflow-v1",
    ) -> WorkflowCheckpoint | None:
        row = await asyncio.to_thread(self._latest_checkpoint_row, run_id)
        if row is None:
            return None
        actual = hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
        if actual != row["sha256"]:
            raise CheckpointIntegrityError("checkpoint payload digest mismatch")
        if (
            row["schema_version"] != schema_version
            or row["workflow_version"] != workflow_version
        ):
            raise CheckpointCompatibilityError(
                "checkpoint schema or workflow version is incompatible"
            )
        try:
            checkpoint = WorkflowCheckpoint.model_validate_json(row["payload_json"])
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "checkpoint payload does not match the supported schema"
            ) from exc
        if (
            checkpoint.checkpoint_id != row["checkpoint_id"]
            or checkpoint.run_id != run_id
            or checkpoint.completed_node != row["completed_node"]
            or checkpoint.schema_version != row["schema_version"]
            or checkpoint.workflow_version != row["workflow_version"]
        ):
            raise CheckpointIntegrityError("checkpoint row metadata mismatch")
        if not await self.verify_ledger(run_id):
            raise CheckpointIntegrityError(
                "checkpoint cannot resume because the audit ledger is invalid"
            )
        return checkpoint

    def _latest_checkpoint_row(self, run_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT checkpoint_id, run_id, completed_node, schema_version,
                       workflow_version, sha256, payload_json, created_at
                FROM stage_checkpoints
                WHERE run_id = ? ORDER BY sequence DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()

    async def checkpoint_manifest(self, run_id: str) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(self._checkpoint_rows, run_id)
        return [dict(row) for row in rows]

    def _checkpoint_rows(self, run_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT checkpoint_id, completed_node, schema_version,
                       workflow_version, sha256, created_at
                FROM stage_checkpoints
                WHERE run_id = ? ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()

    async def append_event(self, event: ExecutionEvent) -> str:
        async with self._write_lock:
            return await asyncio.to_thread(self._append_event_sync, event)

    def _append_event_sync(self, event: ExecutionEvent) -> str:
        encoded = event.model_dump_json()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_not_cancelled(connection, event.run_id)
            previous = connection.execute(
                "SELECT entry_hash FROM audit_ledger WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (event.run_id,),
            ).fetchone()
            previous_hash = previous["entry_hash"] if previous else "GENESIS"
            entry_hash = hashlib.sha256(
                f"{previous_hash}|{encoded}".encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO audit_ledger(
                    event_id, run_id, event_json, previous_hash, entry_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    encoded,
                    previous_hash,
                    entry_hash,
                    event.created_at.isoformat(),
                ),
            )
        return entry_hash

    async def verify_ledger(self, run_id: str) -> bool:
        rows = await asyncio.to_thread(self._ledger_rows, run_id)
        previous_hash = "GENESIS"
        for row in rows:
            expected = hashlib.sha256(
                f"{previous_hash}|{row['event_json']}".encode("utf-8")
            ).hexdigest()
            if row["previous_hash"] != previous_hash or row["entry_hash"] != expected:
                return False
            previous_hash = row["entry_hash"]
        artifact_rows = await asyncio.to_thread(self._artifact_payload_rows, run_id)
        artifact_digests: dict[str, str] = {}
        for row in artifact_rows:
            actual = hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
            if actual != row["sha256"]:
                return False
            artifact_digests[row["artifact_id"]] = actual
        # Every event that advertises an artifact also binds its digest into
        # the append-only event payload.  Older databases without bindings are
        # accepted only for events that do not reference artifacts.
        for row in rows:
            event = json.loads(row["event_json"])
            ids = event.get("artifact_ids", [])
            bound = event.get("metadata", {}).get("artifact_sha256", {})
            for artifact_id in ids:
                if artifact_id not in artifact_digests:
                    return False
                if bound.get(artifact_id) != artifact_digests[artifact_id]:
                    return False
        bound_ids = {
            artifact_id
            for row in rows
            for artifact_id in json.loads(row["event_json"]).get(
                "metadata", {}
            ).get("artifact_sha256", {})
        }
        if bound_ids != set(artifact_digests):
            return False
        checkpoint_rows = await asyncio.to_thread(self._checkpoint_payload_rows, run_id)
        checkpoint_events: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            event = json.loads(row["event_json"])
            checkpoint_id = event.get("metadata", {}).get("checkpoint_id")
            if checkpoint_id and event.get("event_type") == "stage_checkpoint_committed":
                checkpoint_events.setdefault(checkpoint_id, []).append(event)
        for row in checkpoint_rows:
            actual = hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
            if actual != row["sha256"]:
                return False
            try:
                checkpoint = WorkflowCheckpoint.model_validate_json(
                    row["payload_json"]
                )
            except Exception:
                return False
            if (
                checkpoint.checkpoint_id != row["checkpoint_id"]
                or checkpoint.run_id != run_id
                or checkpoint.completed_node != row["completed_node"]
                or checkpoint.schema_version != row["schema_version"]
                or checkpoint.workflow_version != row["workflow_version"]
            ):
                return False
            for artifact_id, expected in checkpoint.artifact_sha256.items():
                if artifact_digests.get(artifact_id) != expected:
                    return False
            bound_events = checkpoint_events.get(checkpoint.checkpoint_id, [])
            if len(bound_events) != 1:
                return False
            metadata = bound_events[0].get("metadata", {})
            if (
                metadata.get("checkpoint_sha256") != actual
                or metadata.get("checkpoint_schema_version")
                != checkpoint.schema_version
                or metadata.get("workflow_version") != checkpoint.workflow_version
                or metadata.get("completed_node") != checkpoint.completed_node
                or metadata.get("artifact_sha256") != checkpoint.artifact_sha256
            ):
                return False
        return True

    def _ledger_rows(self, run_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT event_json, previous_hash, entry_hash
                FROM audit_ledger WHERE run_id = ? ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()

    def _artifact_payload_rows(self, run_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT artifact_id, sha256, payload_json FROM artifacts WHERE run_id = ?",
                (run_id,),
            ).fetchall()

    def _checkpoint_payload_rows(self, run_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT checkpoint_id, run_id, completed_node, schema_version,
                       workflow_version, sha256, payload_json
                FROM stage_checkpoints WHERE run_id = ? ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()

    async def artifact_manifest(self, run_id: str) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(self._artifact_rows, run_id)
        return [dict(row) for row in rows]

    def _artifact_rows(self, run_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT artifact_id, artifact_type, schema_version, sha256, created_at
                FROM artifacts WHERE run_id = ? ORDER BY created_at
                """,
                (run_id,),
            ).fetchall()
