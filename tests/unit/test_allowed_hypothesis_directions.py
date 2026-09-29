"""Direction-selection contracts using isolated synthetic evidence, never live calls."""
from hashlib import sha256
import importlib
import json
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from drugpilot.agents.ma1 import Critic, Proposer, TargetSeed, _expand_directional_hypotheses
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import (
    MA3AssuranceAgent, _required_candidate_direction, _retained_primary_role_is_linked,
)
from drugpilot.agents.planner import PlannerAgent
from drugpilot.api.app import CreateRunRequest, _canonical_create_request_sha256, create_app
from drugpilot.config import Settings
from drugpilot.domain.models import (
    GateName, GateVerdictCode, HypothesisPacket, HypothesisTrack, ProjectConstraints, ResearchBrief,
    RunMode, RunSnapshot, RunStage,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.storage import SQLiteRunStore
from test_qa_direction_resolution import brief, build, record
from tests.unit.test_scientific_tool_health import _ready_report
from test_reviewer_provider import _fixture_ma1


def request_payload(**updates):
    return {"research_goal": "Assess source-supported target biology.",
            "disease": "Acute myeloid leukemia", "intervention_direction": "unspecified", **updates}


@pytest.mark.parametrize("allowed", [[], ["inhibit", "inhibit"], ["unspecified"], ["unknown"],
                                    ["inhibit", "activate", "modulate", "inhibit"]])
@pytest.mark.parametrize("model", [CreateRunRequest, ProjectConstraints])
def test_invalid_direction_filters_fail_closed(model, allowed):
    data = {"intervention_direction": "unspecified", "allowed_hypothesis_directions": allowed}
    if model is CreateRunRequest:
        data = request_payload(**data)
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize("requested", ["inhibit", "activate"])
@pytest.mark.parametrize("model", [CreateRunRequest, ProjectConstraints])
def test_explicit_filter_requires_broad_intake(model, requested):
    data = {"intervention_direction": requested, "allowed_hypothesis_directions": [requested]}
    if model is CreateRunRequest:
        data = request_payload(**data)
    with pytest.raises(ValidationError, match="requires modulate or unspecified"):
        model.model_validate(data)


def test_legacy_json_shape_and_request_hash_are_unchanged_without_filter():
    request = CreateRunRequest.model_validate(request_payload())
    old_shape = request.model_dump(mode="json")
    assert "allowed_hypothesis_directions" not in old_shape
    expected_hash = sha256(json.dumps(old_shape, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
    assert _canonical_create_request_sha256(request) == expected_hash
    explicit_none = CreateRunRequest.model_validate(request_payload(allowed_hypothesis_directions=None))
    assert explicit_none.model_dump(mode="json") == old_shape
    assert _canonical_create_request_sha256(explicit_none) == expected_hash
    constrained = CreateRunRequest.model_validate(request_payload(allowed_hypothesis_directions=["activate"]))
    assert _canonical_create_request_sha256(constrained) != expected_hash
    legacy = ProjectConstraints(intervention_direction="modulate").model_dump(mode="json")
    assert "allowed_hypothesis_directions" not in legacy
    assert ProjectConstraints.model_validate(legacy).model_dump(mode="json") == legacy


@pytest.mark.parametrize("module_name", ["scripts.jak2_et_paper_native_v5_free_preflight",
                                        "scripts.jak2_et_r18_validation_v6_free_preflight"])
def test_isolated_ast_preflight_models_keep_direction_validation_dependencies(module_name):
    preflight = importlib.import_module(module_name)
    baseline = preflight._extract_create_run_request(request_payload())
    selected = preflight._extract_create_run_request(request_payload(allowed_hypothesis_directions=["activate"]))
    assert selected == baseline and len(selected) == 64
    for allowed in ([], ["inhibit", "inhibit"], ["unspecified"]):
        with pytest.raises(preflight.PreflightContractError, match="CREATE_RUN_REQUEST_VALIDATION_FAILED"):
            preflight._extract_create_run_request(request_payload(allowed_hypothesis_directions=allowed))


@pytest.mark.parametrize("intake", ["modulate", "unspecified"])
@pytest.mark.parametrize("allowed", [None, ["inhibit", "activate"], ["inhibit"], ["activate"], ["modulate"]])
def test_only_allowed_source_supported_branches_are_produced(intake, allowed):
    b = brief(intake)
    b.constraints.allowed_hypothesis_directions = allowed
    evidence = [record("FLT3 inhibition reduced viability in acute myeloid leukemia cells.", intake, "inh"),
                record("Activation of FLT3 increased apoptosis in acute myeloid leukemia cells.", intake, "act"),
                record("Modulation of FLT3 reduced viability in acute myeloid leukemia cells.", intake, "mod")]
    original_hashes = {e.evidence_id: e.excerpt_hash for e in evidence}
    h = build(b, evidence)
    before = h.model_dump_json()
    branches = _expand_directional_hypotheses(b, h)
    assert {branch.intervention_direction for branch in branches} == set(allowed or ["inhibit", "activate", "modulate"])
    assert b.constraints.intervention_direction == intake
    assert h.model_dump_json() == before
    for branch in branches:
        assert branch.target.metadata["direction_resolution"]["status"] == "resolved_direct_primary"
        assert branch.target.metadata["direction_resolution"]["requested_direction"] == intake
        assert {e.evidence_id: e.excerpt_hash for e in branch.evidence} == original_hashes
        assert all(e.metadata["query_intervention_direction"] == intake for e in branch.evidence)


@pytest.mark.parametrize("intake", ["modulate", "unspecified"])
def test_missing_allowed_direction_does_not_fallback_to_other_evidence(intake):
    b = brief(intake)
    b.constraints.allowed_hypothesis_directions = ["activate"]
    branches = _expand_directional_hypotheses(b, build(b, [record(
        "FLT3 inhibition reduced viability in acute myeloid leukemia cells.", intake)]))
    assert len(branches) == 1
    assert branches[0].target.metadata["direction_resolution"]["status"] == "direction_unresolved"
    critique = Critic().review(b, branches[0])
    assert not critique.accepted
    assert "INTERVENTION_DIRECTION_UNRESOLVED" in critique.reason_codes
    assert "HYPOTHESIS_DIRECTION_NOT_ALLOWED" in critique.reason_codes


def test_critic_g0_planner_and_g1_reject_a_disallowed_but_source_linked_hypothesis():
    b = brief("unspecified")
    h = _expand_directional_hypotheses(b, build(b, [record(
        "FLT3 inhibition reduced viability in acute myeloid leukemia cells.", "unspecified")]))[0]
    assert _retained_primary_role_is_linked(h, "direct_target_direction_primary", b)
    packet = HypothesisPacket(brief_id=b.brief_id, hypotheses=[h])
    before = packet.model_dump_json()
    b.constraints.allowed_hypothesis_directions = ["activate"]
    critique = Critic().review(b, h)
    assert "HYPOTHESIS_DIRECTION_NOT_ALLOWED" in critique.reason_codes
    assert not critique.accepted
    assert not _retained_primary_role_is_linked(h, "direct_target_direction_primary", b)
    agent = MA3AssuranceAgent()
    findings = agent._audit_g0(b, packet)
    assert "RETAINED_INTERVENTION_DIRECTION_MISMATCH" in {f.code for f in findings}
    assert agent._decide(GateName.G0, findings, []).verdict is not GateVerdictCode.PASS
    with pytest.raises(ValueError, match="outside allowed_hypothesis_directions"):
        PlannerAgent().run(packet, b)
    candidate = SimpleNamespace(target=h.target, metadata={"portfolio_hypothesis_id": h.hypothesis_id})
    assert _required_candidate_direction(b, candidate, packet) is None
    assert packet.model_dump_json() == before


def test_allowed_hypothesis_keeps_g1_source_linked_direction_not_candidate_assertion():
    b = brief("modulate")
    b.constraints.allowed_hypothesis_directions = ["inhibit"]
    h = _expand_directional_hypotheses(b, build(b, [record(
        "FLT3 inhibition reduced viability in acute myeloid leukemia cells.", "modulate")]))[0]
    packet = HypothesisPacket(brief_id=b.brief_id, hypotheses=[h])
    candidate = SimpleNamespace(target=h.target, metadata={"portfolio_hypothesis_id": h.hypothesis_id,
                                                         "direction_requested": "activate"})
    assert _required_candidate_direction(b, candidate, packet) == "inhibit"
    candidate.metadata["portfolio_hypothesis_id"] = "not_a_retained_hypothesis"
    assert _required_candidate_direction(b, candidate, packet) is None


@pytest.mark.parametrize("target,disease", [("EGFR", "lung cancer"), ("JAK2", "myelofibrosis")])
def test_selection_is_target_and_disease_independent(target, disease):
    b = ResearchBrief(disease=disease, research_goal=f"Assess source-supported {target} biology.",
                     constraints=ProjectConstraints(intervention_direction="unspecified",
                                                    allowed_hypothesis_directions=["activate"]))
    evidence = [record(f"{target} inhibition reduced viability in {disease} cells.", "unspecified", "inh"),
                record(f"Activation of {target} increased apoptosis in {disease} cells.", "unspecified", "act")]
    class Retriever:
        def retrieve(self, *args):
            return evidence
    h = Proposer(Retriever())._build_hypothesis(
        b, TargetSeed(target, target, HypothesisTrack.LITERATURE, 1.0), 1)
    branches = _expand_directional_hypotheses(b, h)
    assert [branch.intervention_direction for branch in branches] == ["activate"]
    assert branches[0].target.metadata["direction_resolution"]["status"] == "resolved_direct_primary"


def test_g1_audit_blocks_disallowed_direction_even_with_existing_plan_and_candidates(brief):
    hypotheses = _fixture_ma1(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    candidates = MA2Agent().run(plan, hypotheses)  # Explicitly offline synthetic component gateway.
    brief.constraints.intervention_direction = "unspecified"
    brief.constraints.allowed_hypothesis_directions = ["activate"]
    agent = MA3AssuranceAgent()
    findings = agent._audit_g1(brief, hypotheses, plan, candidates)
    codes = {finding.code for finding in findings}
    assert "RETAINED_INTERVENTION_DIRECTION_MISMATCH" in codes
    assert "DIRECTION_MISMATCH" in codes
    assert agent._decide(GateName.G1, findings, []).verdict is not GateVerdictCode.PASS


@pytest.mark.asyncio
async def test_api_persists_filter_and_health_advertises_it_without_executing_research(tmp_path, monkeypatch):
    report, _ = _ready_report(tmp_path)
    monkeypatch.setattr("drugpilot.api.app.probe_scientific_tools", lambda **kwargs: report)
    received = []

    async def capture_create_run(self, new_brief, **kwargs):
        received.append(new_brief)
        return RunSnapshot(run_id="run_direction_contract", mode=RunMode.MOCK,
                           stage=RunStage.CREATED, status_message="Synthetic API contract only", brief=new_brief)

    monkeypatch.setattr(DrugPilotOrchestrator, "create_run", capture_create_run)
    settings = Settings(_env_file=None, mode=RunMode.MOCK, database_path=tmp_path / "direction.sqlite3",
                        openai_api_key=None, anthropic_api_key=None, public_evidence_enabled=False,
                        ma2_backend_mode="fixture", mock_step_delay_ms=0)
    app = create_app(settings=settings, store=SQLiteRunStore(settings.database_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["direction_policy"] == {
            "version": "source-bound-direction-selection-v1",
            "allowed_hypothesis_directions": True, "search_remains_broad": True,
        }
        response = await client.post("/api/runs", json=request_payload(allowed_hypothesis_directions=["activate"]))
        assert response.status_code == 202, response.text
        assert response.json()["brief"]["constraints"]["allowed_hypothesis_directions"] == ["activate"]
        assert response.json()["brief"]["constraints"]["intervention_direction"] == "unspecified"
        old_response = await client.post("/api/runs", json=request_payload())
        assert old_response.status_code == 202
        assert "allowed_hypothesis_directions" not in old_response.json()["brief"]["constraints"]
        bad = await client.post("/api/runs", json=request_payload(allowed_hypothesis_directions=[]))
        assert bad.status_code == 422
    assert len(received) == 2
    assert received[0].constraints.allowed_hypothesis_directions == ["activate"]
    assert received[1].constraints.allowed_hypothesis_directions is None
