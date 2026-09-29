from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import ResearchBrief, RunMode, RunSnapshot, RunStage
from drugpilot.storage import SQLiteRunStore


@pytest.mark.asyncio
async def test_call_trace_endpoint_is_read_only_and_returns_named_sections(
    tmp_path, brief: ResearchBrief
) -> None:
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "call-trace-api.sqlite3",
        mock_step_delay_ms=0,
    )
    store = SQLiteRunStore(settings.database_path)
    snapshot = RunSnapshot(
        run_id="run_empty_trace",
        mode=RunMode.MOCK,
        stage=RunStage.CREATED,
        status_message="Created",
        brief=brief,
    )
    await store.save_snapshot(snapshot)
    app = create_app(settings=settings, store=store)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/runs/run_empty_trace/calls")
        missing = await client.get("/api/runs/missing/calls")

    response.raise_for_status()
    payload = response.json()
    assert payload["run_id"] == "run_empty_trace"
    assert payload["summary"] == {
        "total_calls": 0,
        "model_calls": 0,
        "public_data_calls": 0,
        "scientific_tool_calls": 0,
        "successful": 0,
        "failed": 0,
        "cached": 0,
        "fixture": 0,
    }
    assert payload["model_calls"] == []
    assert payload["public_data_calls"] == []
    assert payload["scientific_tool_calls"] == []
    assert missing.status_code == 404
