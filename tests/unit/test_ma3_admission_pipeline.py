"""Synthetic, isolated MA1-submission → MA3 → Planner/MA2 boundary tests.

These tests never open a database or dispatch scientific/network calls. The
source text is synthetic unit-test input, not a research fixture or live result.
"""
from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.agents.hypothesis_admission import POLICY, admitted_support_ids, assess_provisional
from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import GateVerdictCode, ProjectConstraints
from drugpilot.tools.route_policy import derive_route_evidence, select_route_from_evidence

# Reuse only explicit synthetic test builders, never application fixture data.
from test_hypothesis_admission import MODEL, Provider, ReviewCache, assessment, packet, proposal


def settings():
    return Settings(_env_file=None, ma1_provisional_admission_enabled=False,
                    dynamic_target_discovery_enabled=False, dynamic_target_fallback_enabled=True,
                    demo_fixture_fallback_enabled=True)


def routed_proposal(brief, identifier="approved-hypothesis"):
    item = proposal(brief, identifier)
    item.target.metadata.update(association_score=.5, evidence_summary={"known_drug_score": .6})
    # The raw classification must not be relabelled to manufacture support.
    for evidence in item.evidence:
        evidence.supports = "context"
    return item


def capture_ma2_dispatch(monkeypatch):
    agent = MA2Agent(ma2_portfolio_enabled=True)
    captured = []
    sentinel = object()

    def dispatch(plan, hypotheses, **kwargs):
        captured.append((plan, hypotheses, kwargs))
        return sentinel

    monkeypatch.setattr(agent, "_run_portfolio", dispatch)
    return agent, captured, sentinel


async def test_mixed_review_g0_planner_and_ma2_receive_approved_subset_only(brief, monkeypatch):
    accepted, rejected = routed_proposal(brief), routed_proposal(brief, "rejected-hypothesis")
    submitted = packet(brief, accepted, rejected)
    original = submitted.model_dump_json()
    provider = Provider([assessment(accepted), assessment(rejected, status="rejected", reason="Source does not support this proposed scope.")])
    approved, report = await assess_provisional(brief, submitted, provider, MODEL)
    assert submitted.model_dump_json() == original
    assert report["submitted_count"] == 2 and report["approved_count"] == 1
    assert [entry["status"] for entry in report["entries"]] == ["approved", "rejected"]
    assert len(approved.hypotheses) == 1 and not approved.provisional_hypotheses
    h = approved.hypotheses[0]
    assert all(e.supports == "context" for e in h.evidence)
    assert h.evidence == accepted.evidence
    assert admitted_support_ids(h) == {"ev_dir", "ev_assoc"}
    assert derive_route_evidence(h, h.target)["support_evidence_count"] == 2
    decision = select_route_from_evidence(h, h.target)
    assert decision["selected_route"] == "repurposing"
    assert decision["drug_evidence_tier"] == "strong"
    verdict = await MA3AssuranceAgent(settings=settings()).audit_g0(brief, approved)
    assert verdict.verdict is GateVerdictCode.PASS, [(f.code, f.message) for f in verdict.findings]
    plan = PlannerAgent(settings=settings()).run(approved, brief)
    assert plan.hypothesis_packet_id == approved.packet_id
    agent, captured, sentinel = capture_ma2_dispatch(monkeypatch)
    assert agent.run(plan, approved) is sentinel
    assert len(captured) == 1
    assert [h.hypothesis_id for h in captured[0][1].hypotheses] == [accepted.hypothesis_id]
    assert rejected.hypothesis_id not in [h.hypothesis_id for h in captured[0][1].hypotheses]


async def test_pending_submissions_cannot_pass_g0_or_reach_planner_ma2(brief, monkeypatch):
    item = routed_proposal(brief)
    submitted = packet(brief, item)
    verdict = await MA3AssuranceAgent(settings=settings()).audit_g0(brief, submitted)
    assert verdict.verdict is not GateVerdictCode.PASS
    assert "PROVISIONAL_HYPOTHESES_NOT_REVIEWED" in {f.code for f in verdict.findings}
    with pytest.raises(ValueError, match="provisional"):
        PlannerAgent(settings=settings()).run(submitted, brief)
    agent, captured, _ = capture_ma2_dispatch(monkeypatch)
    with pytest.raises(ValueError, match="provisional"):
        agent.run(object(), submitted)
    assert not captured


async def test_valid_same_target_tracks_are_breadth_warning_not_scientific_failure(brief):
    first, second = routed_proposal(brief, "literature"), routed_proposal(brief, "target-biology")
    from drugpilot.domain.models import HypothesisTrack
    second.track = HypothesisTrack.TBIO
    approved, _ = await assess_provisional(brief, packet(brief, first, second),
        Provider([assessment(first), assessment(second)]), MODEL)
    assert len(approved.hypotheses) == 2
    verdict = await MA3AssuranceAgent(settings=settings()).audit_g0(brief, approved)
    assert verdict.verdict is GateVerdictCode.PASS
    assert "LIMITED_TARGET_DIVERSITY" in {f.code for f in verdict.findings}
    control = next(v for v in verdict.validations if v.name == "target_diversity_control")
    assert control.passed and control.details["unique_targets"] == 1
    assert control.details["policy"] == "nonempty_approved_pool_breadth_advisory"
    # This change cannot hide rank errors or alter the historical policy.
    approved.hypotheses[0].rank = None
    verdict = await MA3AssuranceAgent(settings=settings()).audit_g0(brief, approved)
    assert verdict.verdict is not GateVerdictCode.PASS
    approved.metadata.pop("admission_policy")
    assert not next(v for v in MA3AssuranceAgent()._validate_g0(approved)
                    if v.name == "target_diversity_control").passed


async def test_new_policy_empty_approved_pool_still_fails_diversity(brief):
    empty = packet(brief)
    verdict = await MA3AssuranceAgent(settings=settings()).audit_g0(brief, empty)
    assert verdict.verdict is not GateVerdictCode.PASS
    assert not next(v for v in verdict.validations if v.name == "target_diversity_control").passed


async def test_legacy_g0_valid_packet_cannot_claim_new_policy_without_certificate(brief, monkeypatch):
    historical = MA1Agent(settings=settings()).run(brief)
    assurance = MA3AssuranceAgent(settings=settings())
    assert (await assurance.audit_g0(brief, historical)).verdict is GateVerdictCode.PASS
    historical.metadata["admission_policy"] = POLICY
    verdict = await assurance.audit_g0(brief, historical)
    assert verdict.verdict is not GateVerdictCode.PASS
    assert "MA3_ADMISSION_CERTIFICATE_INVALID" in {f.code for f in verdict.findings}
    with pytest.raises(ValueError, match="source-bound MA3 approval"):
        PlannerAgent(settings=settings()).run(historical, brief)
    agent, captured, _ = capture_ma2_dispatch(monkeypatch)
    with pytest.raises(ValueError, match="source-bound MA3 approval"):
        agent.run(object(), historical)
    assert not captured


async def test_allowed_direction_filter_is_authoritative_across_admission_and_pipeline(brief, monkeypatch):
    values = brief.constraints.model_dump(mode="json")
    values.update(intervention_direction="unspecified", allowed_hypothesis_directions=["activate"])
    brief.constraints = ProjectConstraints.model_validate(values)
    item = routed_proposal(brief)
    approved, report = await assess_provisional(brief, packet(brief, item), Provider([assessment(item)]), MODEL)
    assert not approved.hypotheses
    assert report["entries"][0]["status"] != "approved"
    assert (await MA3AssuranceAgent(settings=settings()).audit_g0(brief, approved)).verdict is not GateVerdictCode.PASS
    with pytest.raises(ValueError):
        PlannerAgent(settings=settings()).run(approved, brief)
    agent, captured, _ = capture_ma2_dispatch(monkeypatch)
    with pytest.raises(ValueError):
        agent.run(object(), approved)
    assert not captured


async def test_changed_source_invalidates_route_support_g0_planner_and_ma2(brief, monkeypatch):
    item = routed_proposal(brief)
    approved, _ = await assess_provisional(brief, packet(brief, item), Provider([assessment(item)]), MODEL)
    approved.hypotheses[0].evidence[0].external_id = "changed-source-not-reviewed"
    h = approved.hypotheses[0]
    assert not admitted_support_ids(h)
    assert derive_route_evidence(h, h.target)["support_evidence_count"] == 0
    assert select_route_from_evidence(h, h.target)["drug_evidence_tier"] != "strong"
    verdict = await MA3AssuranceAgent(settings=settings()).audit_g0(brief, approved)
    assert verdict.verdict is not GateVerdictCode.PASS
    assert "MA3_ADMISSION_CERTIFICATE_INVALID" in {f.code for f in verdict.findings}
    with pytest.raises(ValueError, match="source-bound MA3 approval"):
        PlannerAgent(settings=settings()).run(approved, brief)
    agent, captured, _ = capture_ma2_dispatch(monkeypatch)
    with pytest.raises(ValueError, match="source-bound MA3 approval"):
        agent.run(object(), approved)
    assert not captured


async def test_resumed_partial_review_reuses_first_response_then_routes_only_approved(brief, monkeypatch):
    first, second = routed_proposal(brief), routed_proposal(brief, "second-held")
    submitted = packet(brief, first, second)
    provider = Provider([assessment(first), assessment(second, status="hold", reason="No further source available.")])
    cache = ReviewCache()
    billed, checked = [], []

    async def usage(item):
        billed.append(deepcopy(item))

    async def preflight(*_):
        checked.append(True)
        if len(checked) == 2:
            raise RuntimeError("bounded synthetic budget stop")

    with pytest.raises(RuntimeError, match="budget stop"):
        await assess_provisional(brief, submitted, provider, MODEL, usage, preflight, **cache.callbacks())
    assert len(provider.calls) == len(billed) == len(cache.saves) == 1
    approved, report = await assess_provisional(brief, submitted, provider, MODEL, usage, **cache.callbacks())
    assert len(provider.calls) == len(billed) == len(cache.saves) == 2
    assert [entry["review_cache"] for entry in report["entries"]] == ["reused", "fresh"]
    assert [entry["status"] for entry in report["entries"]] == ["approved", "hold"]
    assert (await MA3AssuranceAgent(settings=settings()).audit_g0(brief, approved)).verdict is GateVerdictCode.PASS
    plan = PlannerAgent(settings=settings()).run(approved, brief)
    agent, captured, sentinel = capture_ma2_dispatch(monkeypatch)
    assert agent.run(plan, approved) is sentinel
    assert len(captured[0][1].hypotheses) == 1
    assert sum(item.total_tokens for item in billed) == 300


@pytest.mark.parametrize("enabled,semantic_config", [(True, True), (True, False), (False, True), (False, False)])
async def test_health_discloses_admission_policy_and_semantic_delegation_without_io(enabled, semantic_config, monkeypatch):
    api = importlib.import_module("drugpilot.api.app")
    # Exercise the actual ASGI health endpoint, replacing infrastructure only.
    # No SQLiteRunStore, LLM provider, scientific probe, or HTTP transport opens.
    store = SimpleNamespace(admission_health=AsyncMock(return_value={
        "active_live_runs": 0, "expired_live_leases": 0, "recovery_required": False,
        "hourly_admitted_count": 0,
    }))
    orchestrator = SimpleNamespace(store=store, providers=SimpleNamespace(
        component_backend="synthetic-unit-test", reviewer_backend="synthetic-unit-test"))
    monkeypatch.setattr(api, "DrugPilotOrchestrator", lambda **_: orchestrator)
    monkeypatch.setattr(api, "_ma2_health_snapshot", lambda *_: {"capabilities": []})
    monkeypatch.setattr(api, "probe_scientific_tools", lambda **_: SimpleNamespace(health_payload=lambda **_: {}))
    monkeypatch.setattr(api, "receptor_profile_health", lambda *_, **__: {"accepted_profile_count": 0})
    monkeypatch.setattr(api, "get_scientific_http_client", lambda: SimpleNamespace(
        trust_backend="unit-test", proxy_mode="none", trust_env=False, max_attempts=0))
    config = Settings(_env_file=None, mode="mock", openai_api_key=None, anthropic_api_key=None,
        ma2_backend_mode="fixture", public_evidence_enabled=False,
        ma1_provisional_admission_enabled=enabled, ma1_semantic_interpretation_enabled=semantic_config)
    app = api.create_app(settings=config, store=store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://isolated-unit-test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    result = response.json()
    assert result["hypothesis_admission"]["policy"] == (POLICY if enabled else "ma1-legacy-primary-gates")
    assert result["hypothesis_admission"]["ma1_provisional_enabled"] is enabled
    limits = result["hypothesis_admission"]["review_limits"]
    if enabled:
        from drugpilot.agents.hypothesis_admission import MAX_HYPOTHESES, MAX_RECORDS, MAX_SOURCE_CHARS
        assert limits["max_hypotheses_per_packet"] == MAX_HYPOTHESES
        assert limits["max_source_records_per_hypothesis"] == MAX_RECORDS
        assert limits["max_source_characters_per_hypothesis"] == MAX_SOURCE_CHARS
        assert limits["run_budget_preflight_still_required"] is True
        assert limits["capacity_exhaustion_is_scientific_rejection"] is False
    else:
        assert limits is None
    assert result["dynamic_target_discovery"]["semantic_interpretation_enabled"] is (semantic_config and not enabled)
    assert result["dynamic_target_discovery"]["semantic_interpretation_policy"] == (
        "delegated_to_ma3_source_bound_admission" if enabled else "source_bound_semantics_v1")
    assert result["public_http_transport"]["reachability_probed"] is False
    store.admission_health.assert_awaited_once()
