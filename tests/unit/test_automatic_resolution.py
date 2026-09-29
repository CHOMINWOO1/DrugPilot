"""Isolated temporary-store tests; never call production providers or scientific APIs."""
import asyncio
from hashlib import sha256
import pytest

from drugpilot.domain.models import AuditFinding, AutomaticResolutionReport, GateName, GateVerdictCode, G1ReauditRecord, HumanGateDecision, RunMode, Severity, utc_now
from drugpilot.orchestration.human_review import HumanReviewService
from drugpilot.orchestration.automatic_resolution import COUNT_CODE, LIMITS, approval_locked, current_report, summarize
from test_human_review import review_run, start_and_wait, IsolatedSourceCollector


@pytest.fixture(autouse=True)
def isolated_rules(review_run,monkeypatch):
    # Science-rule coverage lives in MA3 tests; this module isolates coordination.
    from drugpilot.agents.ma3_reviewer_projection import ReviewerProjectionMA3AssuranceAgent
    o,_,_=review_run
    async def rules(self,*args,**kwargs):
        return o.ma3._decide(GateName.G1,[],[])
    monkeypatch.setattr(ReviewerProjectionMA3AssuranceAgent,"audit_g1",rules)


async def prepare(review_run, *, concern=False):
    o, _, s = review_run
    o.settings.public_evidence_enabled = True
    o.settings.demo_fixture_fallback_enabled = False
    s.mode = RunMode.LIVE
    if concern:
        finding = AuditFinding(gate=GateName.G1, category="model_appropriateness",
            severity=Severity.LOW, code=COUNT_CODE, message="Isolated test count concern", deterministic=False)
        s.g1_verdict = o.ma3._decide(GateName.G1, [finding], [], reviewer_invoked=True)
        digest = await o.store.save_artifact(s.run_id, s.g1_verdict.verdict_id, "GateVerdict", s.g1_verdict)
        await o._emit(s,s.stage,"test_only_concern","Isolated test",artifact_ids=[s.g1_verdict.verdict_id],
            metadata={"artifact_sha256":{s.g1_verdict.verdict_id:digest}})
    await o.store.save_snapshot(s)
    collector = IsolatedSourceCollector()
    return o, HumanReviewService(o, source_enricher=collector), s, collector


@pytest.mark.asyncio
async def test_automatic_completion_idempotency_and_original_hashes(review_run):
    o,service,s,collector = await prepare(review_run, concern=True)
    before = await o.store.artifact_manifest(s.run_id)
    result = await start_and_wait(service,o,s,mode="automatic")
    report = result.automatic_resolution_reports[-1]
    assert report.outcome == "READY_FOR_REVIEW" and report.code_checks
    assert [x["step"] for x in report.steps] == ["sources","rules","external"]
    assert report.steps[-1]["status"] == "skipped"
    assert report.usage_start_index == report.usage_end_index and result.token_usage == s.token_usage
    assert result.candidate_packet == s.candidate_packet and result.human_decision is None
    assert current_report(result) == report and not approval_locked(result)
    assert not (await service.status(s.run_id))["modes"]["automatic"]["eligible"]
    await start_and_wait(service,o,s,mode="automatic")
    assert collector.calls == 1
    after = {a["artifact_id"]:a["sha256"] for a in await o.store.artifact_manifest(s.run_id)}
    assert all(after[a["artifact_id"]] == a["sha256"] for a in before)
    assert await o.store.verify_ledger(s.run_id)
    assert result.human_decision is None  # Readiness never invents human approval.
    assert result.next_action_packet is None
    assert any(f.code == COUNT_CODE and f.confirmation_status == "unconfirmed"
               for f in result.g1_verdict.findings)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["clean", "external_error", "provider_disabled"])
async def test_bounded_external_review_and_failed_request_not_replayed(review_run, monkeypatch, mode):
    o,service,s,_ = await prepare(review_run, concern=True)
    calls = []
    o.ma3.reviewer_provider = object()
    if mode == "provider_disabled": o.settings.automatic_resolution_external_review_enabled = False
    async def audit(*args, **kwargs):
        calls.append(kwargs["supplemental_evidence"])
        assert calls[-1].automatic_code_checks
        if mode == "external_error": raise RuntimeError("test-only ambiguous response")
        return o.ma3._decide(GateName.G1, [], [], reviewer_invoked=True)
    monkeypatch.setattr(o.ma3, "audit_g1", audit)
    result = await start_and_wait(service,o,s,mode="automatic",allow_paid=True)
    report = result.automatic_resolution_reports[-1]
    assert len(calls) == (0 if mode == "provider_disabled" else 1)
    assert report.outcome == ("HOLD" if mode == "external_error" else "READY_FOR_REVIEW")
    assert result.human_decision is None and result.candidate_packet == s.candidate_packet
    await start_and_wait(service,o,s,mode="automatic",allow_paid=True)
    assert len(calls) <= 1 and await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["source_error", "timeout"])
async def test_source_failure_and_timeout_end_hold_without_model_call(review_run, monkeypatch, failure):
    o,service,s,_ = await prepare(review_run)
    class Broken:
        async def collect(self,snapshot):
            if failure == "timeout": await asyncio.Event().wait()
            raise RuntimeError("Test-only unavailable source")
    service.source_enricher = Broken()
    if failure == "timeout": monkeypatch.setitem(LIMITS,"elapsed_seconds",0.1)
    result = await start_and_wait(service,o,s,mode="automatic",allow_paid=True)
    report = result.automatic_resolution_reports[-1]
    assert report.outcome == "HOLD" and approval_locked(result)
    assert "AUTOMATED_STEP_FAILED_NO_AUTOMATIC_REPLAY" in report.reason_codes
    assert result.token_usage == s.token_usage and result.candidate_packet == s.candidate_packet
    assert not any(r.status == "running" for r in result.g1_reaudits)
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_native_workflow_auto_runs_under_existing_lease(review_run,monkeypatch):
    o,_,s,_ = await prepare(review_run, concern=True)
    from drugpilot.tools.review_source_enrichment import PublicSourceEnricher
    collector=IsolatedSourceCollector()
    monkeypatch.setattr(PublicSourceEnricher,"collect",lambda self,snapshot:collector.collect(snapshot))
    token="test-native-automatic-lease"
    await o.store.claim_live_execution_lease(s.run_id,lease_token=token,now=utc_now(),
        lease_ttl_seconds=120,max_active_live_runs=1)
    o._workflow_lease_tokens[s.run_id]=token
    try:
        await o._await_human({"run_id":s.run_id})
    finally:
        await o.store.release_live_execution_lease(s.run_id,lease_token=token,reason="test-finished",now=utc_now())
    result=await o.store.get_snapshot(s.run_id)
    assert collector.calls == 1 and result.automatic_resolution_reports[-1].outcome == "READY_FOR_REVIEW"
    assert result.human_decision is None and result.next_action_packet is None
    assert "자동 분석 완료" in result.status_message and await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_failed_parent_after_partial_pass_never_unlocks_approval(review_run, monkeypatch):
    o,service,s,_=await prepare(review_run)
    original=o.store.commit_review_update
    async def fail_final(*args,**kwargs):
        if any(kind=="AutomaticResolutionReport" for _,kind,_ in kwargs.get("artifacts",[])):
            raise RuntimeError("Test-only interrupted final transaction")
        return await original(*args,**kwargs)
    monkeypatch.setattr(o.store,"commit_review_update",fail_final)
    result=await start_and_wait(service,o,s,mode="automatic")
    assert result.g1_verdict.verdict is GateVerdictCode.PASS and not result.automatic_resolution_reports
    assert next(r for r in result.g1_reaudits if r.mode=="automatic").status == "failed"
    assert approval_locked(result)
    with pytest.raises(ValueError,match="AUTOMATIC_COMPLETION_NOT_VERIFIED"):
        await service.run_automatic_inline(s.run_id,"not-used")
    with pytest.raises(ValueError,match="APPROVE"):
        await o.apply_human_decision(HumanGateDecision(run_id=s.run_id,decision="APPROVE",researcher="Test",
            rationale="No partial pass",selected_candidate_ids=[s.candidate_packet.candidates[0].candidate_id]))
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_resolution_outcomes_distinguish_missing_evidence_from_block(review_run):
    _,_,s=review_run
    assert summarize(s,[])[0] == "READY_FOR_REVIEW"
    s.g1_verdict.findings=[AuditFinding(gate=GateName.G1,category="safety",severity=Severity.MEDIUM,
        code="ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION",message="Isolated test prediction")]
    outcome,reasons,_=summarize(s,[])
    assert outcome == "HOLD" and "EXPERIMENTAL_VALIDATION_UNAVAILABLE" in reasons
    s.g1_verdict.verdict=GateVerdictCode.BLOCK
    assert summarize(s,[])[0]=="BLOCKED"


@pytest.mark.asyncio
async def test_child_start_publish_failure_leaves_no_running_record(review_run,monkeypatch):
    o,service,s,_=await prepare(review_run)
    original=o.broker.publish
    async def publish(event):
        if event.event_type=="g1_reaudit_running" and event.artifact_ids:
            current=await o.store.get_snapshot(s.run_id)
            if any(r.mode=="sources" and r.status=="running" for r in current.g1_reaudits):
                raise RuntimeError("Test-only UI delivery failure")
        await original(event)
    monkeypatch.setattr(o.broker,"publish",publish)
    result=await start_and_wait(service,o,s,mode="automatic")
    assert result.automatic_resolution_reports[-1].outcome=="HOLD"
    assert not any(r.status=="running" for r in result.g1_reaudits)
    assert await o.store.verify_ledger(s.run_id)


@pytest.mark.asyncio
async def test_budget_preflight_stops_external_without_science_substitution(review_run,monkeypatch):
    from drugpilot.orchestration.budget import BudgetExceeded
    o,service,s,_=await prepare(review_run,concern=True)
    calls=[]
    o.ma3.reviewer_provider=object()
    async def preflight(*args):
        raise BudgetExceeded("test-only original run budget")
    async def audit(*args,**kwargs):
        await kwargs["before_review"]("gpt-5.6-sol","isolated",100)
        calls.append("must never call")
    monkeypatch.setattr(o,"_preflight_llm",preflight)
    monkeypatch.setattr(o.ma3,"audit_g1",audit)
    result=await start_and_wait(service,o,s,mode="automatic",allow_paid=True)
    assert not calls and result.automatic_resolution_reports[-1].outcome=="HOLD"
    assert result.brief.budget==s.brief.budget and result.token_usage==s.token_usage
    assert result.candidate_packet==s.candidate_packet


@pytest.mark.asyncio
async def test_cancellation_has_no_paid_replay_or_unfinished_child(review_run,monkeypatch):
    o,service,s,_=await prepare(review_run)
    entered=asyncio.Event()
    class Pending:
        async def collect(self,snapshot):
            entered.set()
            await asyncio.Event().wait()
    service.source_enricher=Pending()
    await service.start(s.run_id,request_id="test-auto-cancel",base_verdict_id=s.g1_verdict.verdict_id,
        researcher="Test",rationale="Isolated cancellation",mode="automatic")
    await entered.wait()
    task=o._tasks[s.run_id]
    task.cancel()
    await task
    result=await o.store.get_snapshot(s.run_id)
    assert not any(r.status=="running" for r in result.g1_reaudits)
    assert approval_locked(result) and not result.automatic_resolution_reports
    assert not (await service.status(s.run_id))["modes"]["automatic"]["eligible"]
    assert result.token_usage==s.token_usage and await o.store.verify_ledger(s.run_id)
