from __future__ import annotations

import asyncio
import base64
from hashlib import sha256

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import AuditFinding, GateName, Severity


@pytest.fixture
async def api_review(tmp_path):
    app = create_app(settings=Settings(_env_file=None, database_path=tmp_path / "review-api.sqlite3",
        auto_activate_openai=False, OPENAI_API_KEY=None, ANTHROPIC_API_KEY=None,
        public_evidence_enabled=False, demo_fixture_fallback_enabled=True, mock_step_delay_ms=0))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/runs", json={
                "research_goal": "Identify source-linked FLT3 inhibitory candidates for review.",
                "disease": "Acute myeloid leukemia", "target_cell_type": "MOLM-13",
                "candidates_per_route": 3, "research_depth": "standard"})
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            o = app.state.orchestrator
            task = o._tasks.get(run_id)
            if task:
                await task
            s = await o.store.get_snapshot(run_id)
            finding = AuditFinding(gate=GateName.G1, category="evidence_provenance", severity=Severity.HIGH,
                code="TEST_ONLY_MISSING_DOCUMENT", message="Source document requires human review.",
                artifact_id=s.candidate_packet.candidates[0].candidate_id)
            verdict = o.ma3._decide(GateName.G1, [finding], [], reviewer_invoked=False)
            digest = await o.store.save_artifact(run_id, verdict.verdict_id, "GateVerdict", verdict)
            s.g1_verdict = verdict
            await o._emit(s, s.stage, "test_only_review", "Isolated API-test document request",
                artifact_ids=[verdict.verdict_id], metadata={"artifact_sha256": {verdict.verdict_id:digest}})
            await o.store.save_snapshot(s)
            await o.materialize_next_actions(run_id)
            yield client, o, await o.store.get_snapshot(run_id)


@pytest.mark.asyncio
async def test_document_upload_download_resolution_and_note_remain_non_approval(api_review):
    client, o, s = api_review
    root = f"/api/runs/{s.run_id}"
    action = s.next_action_packet.actions[0]
    content = b"Original source document for an isolated software test."
    payload = {"evidence_type":"source_document", "submitted_by":"Test researcher",
        "notes":"Original document received; scientific applicability is not established.",
        "source_filename":"source.txt", "media_type":"text/plain",
        "source_content_base64":base64.b64encode(content).decode()}
    response = await client.post(f"{root}/actions/{action.action_id}/evidence", json=payload)
    assert response.status_code == 200, response.text
    submission = response.json()["action_evidence_submissions"][0]
    assert submission["source_artifact_sha256"] == sha256(content).hexdigest()
    repeated = await client.post(f"{root}/actions/{action.action_id}/evidence", json=payload)
    assert len(repeated.json()["action_evidence_submissions"]) == 1
    changed = await client.post(f"{root}/actions/{action.action_id}/evidence",
                                json={**payload, "notes":"Different review metadata"})
    assert changed.status_code == 409
    download = await client.get(f"{root}/attachments/{submission['attachment_id']}")
    assert download.content == content
    assert download.headers["x-content-type-options"] == "nosniff"
    assert "attachment" in download.headers["content-disposition"]
    assert (await client.get(f"/api/runs/run_other/attachments/{submission['attachment_id']}")).status_code == 404
    resolution = await client.post(f"{root}/actions/{action.action_id}/resolve", json={
        "supporting_submission_ids":[submission["submission_id"]], "resolved_by":"Test reviewer",
        "rationale":"Review source receipt, without scientific approval.", "completion_attested":True})
    assert resolution.status_code == 200
    assert resolution.json()["next_action_progress"]["approval_allowed"] is False
    note = await client.post(f"{root}/review-notes", json={"researcher":"Test reviewer", "note":"Wait for more data."})
    assert note.status_code == 200 and note.json()["stage"] == "human_gate"
    forbidden = await client.post(f"{root}/human-decision", json={"researcher":"Test reviewer",
        "rationale":"Cannot bypass the audit", "decision":"APPROVE", "selected_candidate_ids":[]})
    assert forbidden.status_code == 409
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_reaudit_api_requires_idempotency_and_records_one_new_audit(api_review):
    client, o, s = api_review
    root = f"/api/runs/{s.run_id}"
    payload = {"base_verdict_id":s.g1_verdict.verdict_id, "researcher":"Test reviewer",
        "rationale":"Recheck corrected rules", "mode":"rules"}
    assert (await client.post(f"{root}/reaudits", json=payload)).status_code == 400
    assert (await client.post(f"{root}/reaudits", json=payload,
        headers={"Idempotency-Key":"too-short"})).status_code == 400
    headers = {"Idempotency-Key":"review-api-idempotent-0001"}
    first = await client.post(f"{root}/reaudits", json=payload, headers=headers)
    assert first.status_code == 202, first.text
    task = o._tasks.get(s.run_id)
    if task:
        await task
    replay = await client.post(f"{root}/reaudits", json=payload, headers=headers)
    assert replay.status_code == 202
    assert len(replay.json()["g1_reaudits"]) == 1
    assert replay.json()["g1_reaudits"][0]["status"] == "completed"
    assert await o.store.verify_ledger(s.run_id)
