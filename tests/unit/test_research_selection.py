from __future__ import annotations

import asyncio
from hashlib import sha256

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import (
    AuditFinding, GateName, GateVerdictCode, HumanDecisionCode, HumanGateDecision,
    NamespacedRegistryIdentifier, ReservedControlCompound, ReworkOrder, RunSnapshot, RunStage,
    Severity, ValidationResult, new_id,
)
from drugpilot.orchestration.research_selection import (
    POLICY_VERSION, ResearchSelectionService, eligible_candidate_ids, project_readiness,
)


@pytest.fixture
async def selection_run(tmp_path, brief):
    app = create_app(settings=Settings(_env_file=None,
        database_path=tmp_path / "selection.sqlite3", auto_activate_openai=False,
        OPENAI_API_KEY=None, ANTHROPIC_API_KEY=None, mock_step_delay_ms=0,
        public_evidence_enabled=False, demo_fixture_fallback_enabled=True))
    async with app.router.lifespan_context(app):
        o = app.state.orchestrator
        created = await o.create_run(brief)
        await o._tasks[created.run_id]
        s = await o.store.get_snapshot(created.run_id)
        assert s.stage is RunStage.HUMAN_GATE and s.g1_verdict.verdict is GateVerdictCode.PASS
        yield app, o, ResearchSelectionService(o), s


def request_for(s):
    return {"researcher": "Test researcher", "rationale": "Prioritize for a future experiment only.",
        "selected_candidate_ids": [sorted(eligible_candidate_ids(s))[0]],
        "expected_g1_verdict_id": s.g1_verdict.verdict_id,
        "expected_candidate_packet_sha256": sha256(s.candidate_packet.model_dump_json().encode()).hexdigest(),
        "limitations_acknowledged": True, "policy_version": POLICY_VERSION}


async def save_verdict(o, s, findings):
    verdict = s.g1_verdict.model_copy(deep=True, update={"verdict_id": new_id("verdict"), "findings": findings})
    digest = await o.store.save_artifact(s.run_id, verdict.verdict_id, "GateVerdict", verdict)
    s.g1_verdict = verdict
    await o._emit(s, s.stage, "test_only_audit", "Isolated test audit",
        artifact_ids=[verdict.verdict_id], metadata={"artifact_sha256": {verdict.verdict_id: digest}})
    await o.store.save_snapshot(s)
    return s


@pytest.mark.asyncio
async def test_current_limitations_selection_is_immutable_nonfinal_and_hash_bound(selection_run):
    _, o, service, s = selection_run
    candidate_id = request_for(s)["selected_candidate_ids"][0]
    findings = [AuditFinding(gate=GateName.G1, category="model_appropriateness",
        severity=Severity.LOW, code="ASSAY_VARIANT_CLASS_ONLY", message="Exact variant not verified.", artifact_id=candidate_id),
        AuditFinding(gate=GateName.G1, category="safety", severity=Severity.MEDIUM,
        code="ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION", message="Predicted ADMET is not measured safety.", artifact_id=candidate_id),
        AuditFinding(gate=GateName.G1, category="scientific", severity=Severity.LOW,
        code="REVIEWER_UNCONFIRMED", message="Unconfirmed advisory remains visible.", artifact_id=candidate_id,
        deterministic=False, confirmation_status="unconfirmed")]
    await save_verdict(o, s, findings)
    await o.materialize_next_actions(s.run_id)
    s = await o.store.get_snapshot(s.run_id)
    before = await o.store.artifact_manifest(s.run_id)
    ready = await service.readiness(s.run_id)
    assert ready["analysis_complete"] and ready["research_selection"]["allowed"]
    assert len(ready["research_selection"]["limitations"]) == 2
    assert len(ready["research_selection"]["advisories"]) == 1
    result = await service.select(s.run_id, **request_for(s))
    record = result.research_candidate_selection
    assert result.stage is RunStage.HUMAN_GATE and result.human_decision is None
    assert result.g1_verdict == s.g1_verdict and result.candidate_packet == s.candidate_packet
    assert result.next_action_packet == s.next_action_packet and result.next_action_progress == s.next_action_progress
    assert result.token_usage == s.token_usage and result.action_resolutions == s.action_resolutions
    assert record.experimental_validation == "not_verified"
    with pytest.raises(ValidationError):
        record.rationale = "changed"
    assert await o.store.read_artifact(s.run_id, record.selection_id, "ResearchCandidateSelection") == record.model_dump(mode="json")
    event = result.events[-1]
    assert event.artifact_ids == [record.selection_id]
    assert event.metadata["artifact_sha256"][record.selection_id] == sha256(record.model_dump_json().encode()).hexdigest()
    after = {a["artifact_id"]: a["sha256"] for a in await o.store.artifact_manifest(s.run_id)}
    assert all(after[a["artifact_id"]] == a["sha256"] for a in before)
    assert await o.store.verify_ledger(s.run_id)
    refreshed = await service.readiness(s.run_id)
    assert refreshed["research_selection"]["state"] == "selected"
    assert not refreshed["research_selection"]["allowed"]
    assert refreshed["experimental_validation"]["state"] == "not_verified"
    with pytest.raises(ValueError, match="APPROVE is locked"):
        await o.apply_human_decision(HumanGateDecision(run_id=s.run_id, decision=HumanDecisionCode.APPROVE,
            researcher="Test", rationale="Existing final approval remains locked.", selected_candidate_ids=[candidate_id]))


@pytest.mark.asyncio
@pytest.mark.parametrize("code,severity,expected", [
    ("UNKNOWN_LOW", Severity.LOW, "UNSUPPORTED_FINDING"),
    ("ASSAY_VARIANT_EXPLICIT_MISMATCH", Severity.MEDIUM, "UNSUPPORTED_FINDING"),
    ("ASSAY_VARIANT_CONTEXT_UNKNOWN", Severity.HIGH, "BLOCKING_FINDING"),
    ("ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION", Severity.CRITICAL, "BLOCKING_FINDING"),
])
async def test_non_allowlisted_and_severe_findings_fail_closed(selection_run, code, severity, expected):
    _, o, service, s = selection_run
    await save_verdict(o, s, [AuditFinding(gate=GateName.G1, category="scientific", severity=severity,
        code=code, message="Blocked isolated test finding", artifact_id=s.candidate_packet.candidates[0].candidate_id)])
    ready = await service.readiness(s.run_id)
    assert not ready["research_selection"]["allowed"]
    with pytest.raises(ValueError, match=expected):
        await service.select(s.run_id, **request_for(s))


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["excluded", "unranked", "policy", "forbidden", "reserved", "restricted", "unresolved", "route"])
async def test_shared_candidate_screening_rejects_invalid_candidates(selection_run, change):
    _, _, _, s = selection_run
    c = next(c for c in s.candidate_packet.candidates if c.candidate_id in eligible_candidate_ids(s))
    if change == "excluded": c.status = "excluded"
    elif change == "unranked":
        c.rank = None
        c.portfolio_rank = None
    elif change == "policy": c.metadata["policy_eligible"] = False
    elif change == "forbidden": s.brief.constraints.forbidden_drugs.append(c.name)
    elif change == "reserved":
        s.brief.constraints.reserved_control_compounds.append(ReservedControlCompound(name=c.name,
            registry_ids=[NamespacedRegistryIdentifier(namespace="test", identifier="control")]))
    elif change in {"restricted", "unresolved"}:
        c.metadata["restricted_safety_screen"] = {"disposition": "excluded" if change == "restricted" else "unresolved"}
    else:
        if s.candidate_packet.portfolio_enabled: c.metadata["portfolio_target_key"] = "foreign_target"
        else: s.candidate_packet.metadata["route_decision"] = {"selected_route": "foreign_route"}
    assert c.candidate_id not in eligible_candidate_ids(s)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed,expected", [
    ({"selected_candidate_ids": ["foreign_candidate"]}, "INELIGIBLE_OR_UNKNOWN"),
    ({"expected_g1_verdict_id": "old_verdict"}, "STALE_G1"),
    ({"expected_candidate_packet_sha256": "0" * 64}, "STALE_CANDIDATE"),
    ({"limitations_acknowledged": False}, "ACKNOWLEDGEMENT"),
    ({"researcher": " "}, "INPUT_REQUIRED"),
    ({"policy_version": "old-policy"}, "ACKNOWLEDGEMENT"),
])
async def test_submission_freshness_scope_and_explicit_ack(selection_run, changed, expected):
    _, _, service, s = selection_run
    with pytest.raises(ValueError, match=expected):
        await service.select(s.run_id, **{**request_for(s), **changed})


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["revise", "g0", "cancelled", "decision", "validation", "busy", "ledger", "binding"])
async def test_projection_terminal_audit_and_integrity_locks(selection_run, condition):
    _, _, _, s = selection_run
    kwargs = {"ledger_valid": True, "binding_valid": True}
    if condition == "revise": s.g1_verdict = s.g1_verdict.model_copy(update={"verdict": GateVerdictCode.REVISE,
        "rework_order": ReworkOrder(gate=GateName.G1, target_stage="ma2", failed_node="test", required_actions=["recheck"])})
    elif condition == "g0": s.g0_verdict = s.g0_verdict.model_copy(update={"verdict": GateVerdictCode.BLOCK})
    elif condition == "cancelled": s.stage = RunStage.CANCELLED
    elif condition == "decision": s.human_decision = HumanGateDecision(run_id=s.run_id, decision=HumanDecisionCode.APPROVE, researcher="Test", rationale="Historical approval is not wetlab proof")
    elif condition == "validation": s.g1_verdict.validations.append(ValidationResult(name="failed", passed=False, control_type="negative"))
    elif condition == "busy": kwargs["busy"] = True
    elif condition == "ledger": kwargs["ledger_valid"] = False
    else: kwargs["binding_valid"] = False
    ready = project_readiness(s, **kwargs)
    assert not ready["research_selection"]["allowed"]
    assert ready["experimental_validation"]["state"] == "not_verified"


@pytest.mark.asyncio
async def test_stale_selection_and_concurrent_duplicate_cannot_change_history(selection_run):
    _, o, service, s = selection_run
    results = await asyncio.gather(service.select(s.run_id, **request_for(s)),
        service.select(s.run_id, **request_for(s)), return_exceptions=True)
    assert sum(isinstance(result, RunSnapshot) for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    selected = await o.store.get_snapshot(s.run_id)
    for name in ("g1_verdict", "g0_verdict", "candidate_packet", "brief", "plan_spec"):
        changed = selected.model_copy(deep=True)
        artifact = getattr(changed, name)
        field = "verdict_id" if "verdict" in name else "packet_id" if name == "candidate_packet" else "brief_id" if name == "brief" else "plan_id"
        setattr(artifact, field, "changed_id")
        assert project_readiness(changed, ledger_valid=True, binding_valid=True)["research_selection"]["state"] == "stale"
    manifest = await o.store.artifact_manifest(s.run_id)
    assert sum(a["artifact_type"] == "ResearchCandidateSelection" for a in manifest) == 1
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_api_read_only_historical_shape_and_request_contract(selection_run):
    app, o, _, s = selection_run
    before = (await o.store.get_snapshot(s.run_id)).model_dump_json()
    assert "research_candidate_selection" not in s.model_dump()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        root = f"/api/runs/{s.run_id}"
        get = await client.get(root + "/result-readiness")
        assert get.status_code == 200 and get.json()["research_selection"]["allowed"]
        assert (await o.store.get_snapshot(s.run_id)).model_dump_json() == before
        missing_ack = request_for(s)
        missing_ack.pop("limitations_acknowledged")
        assert (await client.post(root + "/research-selection", json=missing_ack)).status_code == 422
        stale = {**request_for(s), "expected_g1_verdict_id": "old"}
        assert (await client.post(root + "/research-selection", json=stale)).status_code == 409
        response = await client.post(root + "/research-selection", json=request_for(s))
        assert response.status_code == 200, response.text
        assert response.json()["stage"] == "human_gate"
        assert response.json()["human_decision"] is None
        assert response.json()["research_candidate_selection"]["experimental_validation"] == "not_verified"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [None, "missing", "error", "fixture", "query_hash", "http_error", "http_fixture", "response_hash", "unavailable", "foreign_hypothesis", "raw_status", "unknown_source"])
async def test_counterevidence_absence_requires_completed_source_provenance(selection_run, change):
    _, _, _, s = selection_run
    hypothesis = s.hypothesis_packet.hypotheses[0]
    queries = ["example target counter evidence search", "independent bounded query"]
    summary = {"status": "completed_empty", "separate_query": True, "fixture_used": False,
        "sources": ["PubMed", "Europe PMC"],
        "queries": queries, "query_sha256s": sorted(sha256(q.encode()).hexdigest() for q in queries),
        "query": None, "query_sha256": None,
        "http_call_provenance": {"status": "completed", "status_code": 200, "fixture": False,
            "method": "GET", "endpoint": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", "response_sha256": "a" * 64,
            "parameter_sha256": "b" * 64}}
    hypothesis.target.metadata["counter_evidence_search"] = summary
    finding = AuditFinding(gate=GateName.G0, category="evidence_provenance", severity=Severity.MEDIUM,
        code="COUNTER_EVIDENCE_NOT_FOUND", message="Completed bounded search found no direct refutation; not proof of no counterevidence.",
        artifact_id=hypothesis.hypothesis_id)
    s.g0_verdict.findings.append(finding)
    if change == "missing": hypothesis.target.metadata.pop("counter_evidence_search")
    elif change == "error": summary["status"] = "error"
    elif change == "fixture": summary["fixture_used"] = True
    elif change == "query_hash": summary["query_sha256s"] = ["c" * 64]
    elif change == "http_error": summary["http_call_provenance"]["status_code"] = 500
    elif change == "http_fixture": summary["http_call_provenance"]["fixture"] = True
    elif change == "response_hash": summary["http_call_provenance"].pop("response_sha256")
    elif change == "unavailable": finding.code = "COUNTER_EVIDENCE_SEARCH_UNAVAILABLE"
    elif change == "foreign_hypothesis": finding.artifact_id = "foreign_hypothesis"
    elif change == "raw_status": summary["raw_search_status"] = "error"
    elif change == "unknown_source": summary["http_call_provenance"]["endpoint"] = "https://example.test/search"
    result = project_readiness(s, ledger_valid=True, binding_valid=True)["research_selection"]
    assert result["allowed"] is (change is None)
    if change is None:
        limitation = next(item for item in result["limitations"] if item["code"] == finding.code)
        assert limitation["message"] == finding.message
        assert limitation["hypothesis_id"] == hypothesis.hypothesis_id and limitation["candidate_id"] is None


@pytest.mark.asyncio
async def test_cancellation_fence_prevents_atomic_selection_commit(selection_run, monkeypatch):
    _, o, service, s = selection_run
    before = await o.store.artifact_manifest(s.run_id)
    original_commit = o.store.commit_review_update
    async def cancel_before_commit(snapshot, **kwargs):
        # Isolated DB injection simulates a cancellation committed by another owner.
        cancelled = await o.store.get_snapshot(s.run_id)
        cancelled.stage = RunStage.CANCELLED
        await o.store.save_snapshot(cancelled)
        return await original_commit(snapshot, **kwargs)
    monkeypatch.setattr(o.store, "commit_review_update", cancel_before_commit)
    with pytest.raises(ValueError, match="CANCEL"):
        await service.select(s.run_id, **request_for(s))
    stored = await o.store.get_snapshot(s.run_id)
    assert stored.research_candidate_selection is None
    assert stored.events == s.events
    assert await o.store.artifact_manifest(s.run_id) == before


@pytest.mark.asyncio
async def test_snapshot_tamper_is_not_mistaken_for_valid_ledger(selection_run):
    _, o, service, s = selection_run
    s.candidate_packet.candidates[0].name = "Tampered snapshot only"
    await o.store.save_snapshot(s)
    assert await o.store.verify_ledger(s.run_id)
    result = await service.readiness(s.run_id)
    assert "CURRENT_ARTIFACT_BINDING_INVALID" in result["research_selection"]["blocking_reasons"]
    with pytest.raises(ValueError, match="CURRENT_ARTIFACT_BINDING_INVALID"):
        await service.select(s.run_id, **request_for(s))
