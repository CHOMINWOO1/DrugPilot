"""Isolated failure-injection tests; no paid requests or live DB writes."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import pytest

from drugpilot.agents.hypothesis_projection import (
    build_hypothesis_projection, expand_hypothesis_projection, compact_hypothesis_input, _body_hash,
)
from drugpilot.agents.ma3_reviewer_projection import ReviewerProjectionIntegrityError, ReviewerProjectionMA3AssuranceAgent
from drugpilot.agents.ma1 import MA1Agent, Router, Critic, _direct_target_intervention_pattern
from drugpilot.config import Settings
from drugpilot.domain.models import ExcludedHypothesis, HypothesisPacket, RunSnapshot, RunMode, RunStage, ExecutionEvent
from drugpilot.storage.sqlite import SQLiteRunStore, CheckpointIntegrityError
from drugpilot.orchestration.semantic_interpretation import interpret_target
from drugpilot.agents.evidence_semantics import EvidenceMeaningBatch
from drugpilot.providers.llm import MockLLMProvider
from test_evidence_semantics import evidence, meaning
from test_molecular_hypothesis_branching import _proposal, _scoped
from drugpilot.agents.ma1 import _expand_molecular_hypotheses


def _large_packet(brief):
    block={"source":"synthetic-test-only", "quote":"preserve this exact negative source passage; "*500,
        "role":"context", "external_id":"synthetic-source", "$drugpilot_canonical_ref_v1":"literal not a reference"}
    return HypothesisPacket(brief_id=brief.brief_id,hypotheses=[],metadata={"history":[deepcopy(block) for _ in range(100)]})


def test_lossless_hypothesis_transport_preserves_reserved_keys_and_source(brief):
    p=_large_packet(brief)
    before=p.model_dump(mode="json")
    projection=build_hypothesis_projection(p)
    assert expand_hypothesis_projection(projection)==before
    assert p.model_dump(mode="json")==before
    assert len(projection.model_dump_json()) < len(p.model_dump_json())/10
    assert compact_hypothesis_input(p).schema_version=="lossless_hypothesis_input_v1"


@pytest.mark.parametrize("mutation",["hash","source","size","duplicate","dangling","cycle"])
def test_hypothesis_transport_rejects_corruption_even_after_envelope_rehash(brief,mutation):
    raw=build_hypothesis_projection(_large_packet(brief)).model_dump(mode="json")
    if mutation=="source": raw["source_packet_sha256"]="0"*64
    elif mutation=="size": raw["shared_canonical_blocks"][0]["canonical_size_bytes"]+=1
    elif mutation=="duplicate": raw["shared_canonical_blocks"].append(deepcopy(raw["shared_canonical_blocks"][0]))
    elif mutation=="dangling": raw["encoded_hypothesis_packet"]={"$drugpilot_canonical_ref_v1":"0"*64}
    elif mutation=="cycle":
        b=raw["shared_canonical_blocks"][0]
        b["value"]={"$drugpilot_canonical_ref_v1":b["sha256"]}
    raw["projection_sha256"]=_body_hash(raw) if mutation!="hash" else "0"*64
    with pytest.raises(ReviewerProjectionIntegrityError): expand_hypothesis_projection(raw)


@pytest.mark.asyncio
async def test_g0_compacts_only_reviewer_input_not_rules(brief):
    p=_large_packet(brief)
    a=ReviewerProjectionMA3AssuranceAgent(reviewer_provider=MockLLMProvider())
    captured=[]
    async def preflight(model,prompt,limit): captured.append(json.loads(prompt))
    verdict=await a.audit_g0(brief,p,before_review=preflight)
    sent=captured[0]
    assert expand_hypothesis_projection(sent["artifacts"]["hypothesis_packet"])==p.model_dump(mode="json")
    assert sent["deterministic_findings"]==[f.model_dump(mode="json") for f in verdict.findings]
    assert any(f.code=="NO_HYPOTHESIS" for f in verdict.findings)


def test_exclusion_scope_is_explicit_and_legacy_json_is_unchanged(brief):
    raw={"target_symbol":"OLD", "reason_codes":["UNSUPPORTED_CLAIM"],"source_iteration":1}
    assert ExcludedHypothesis.model_validate(raw).model_dump(mode="json")==raw
    base,_=_proposal(brief,[_scoped(brief)])
    branches=_expand_molecular_hypotheses(brief,base)
    accepted,excluded=Router().route([Critic().review(brief,h) for h in branches],1)
    assert not accepted and len(excluded)==2
    assert [e.hypothesis_id for e in excluded]==[h.hypothesis_id for h in branches]
    assert excluded[1].molecular_scope==branches[1].target.metadata["molecular_hypothesis_scope"]


def test_new_packet_timestamp_is_not_brief_timestamp(brief):
    brief.created_at=datetime(2020,1,1,tzinfo=timezone.utc)
    packet=MA1Agent(settings=Settings(_env_file=None,mode=RunMode.MOCK,mock_step_delay_ms=0)).run(brief)
    assert packet.generated_at > brief.created_at


@pytest.mark.parametrize("text,expected",[
    ("blocking XYZ2 signaling reduced cell viability",False),
    ("inhibition of XYZ2 signaling reduced cell viability",False),
    ("suppressed XYZ2-mediated growth",False),
    ("XYZ2 inhibition reduced cell viability",True),
    ("A selective XYZ2 inhibitor reduced cell viability",True),
    ("The compound directly bound to XYZ2",True),
])
def test_pathway_readout_is_not_direct_target_intervention(text,expected):
    assert bool(_direct_target_intervention_pattern("XYZ2","inhibit").search(text)) is expected


async def _store(tmp_path,brief):
    store=SQLiteRunStore(tmp_path/"isolated.sqlite3")
    s=RunSnapshot(run_id="unit-only",mode=RunMode.MOCK,stage=RunStage.MA1,status_message="unit",brief=brief)
    await store.save_snapshot(s)
    return store,s


@pytest.mark.asyncio
async def test_intermediate_transaction_rolls_back_artifact_on_event_collision(tmp_path,brief):
    store,s=await _store(tmp_path,brief)
    e=ExecutionEvent(run_id=s.run_id,stage=s.stage,event_type="test",message="unit",artifact_ids=["a"])
    await store.commit_artifact_event(s,"a","Unit",{"data":1},e)
    s=await store.get_snapshot(s.run_id)
    duplicate=e.model_copy(update={"artifact_ids":["b"]},deep=True)
    with pytest.raises(CheckpointIntegrityError):
        await store.commit_artifact_event(s,"b","Unit",{"data":2},duplicate)
    assert await store.read_artifact(s.run_id,"b","Unit") is None
    assert await store.verify_ledger(s.run_id)
    with pytest.raises(CheckpointIntegrityError):
        await store.commit_artifact_event(s,"a","Unit",{"data":"changed"},e)
    assert await store.verify_ledger(s.run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure",[False,True])
async def test_semantic_interruption_keeps_ledger_complete_and_never_replays(tmp_path,brief,failure):
    store,s=await _store(tmp_path,brief)
    count=0
    e=evidence()
    async def enhance(*args,**kwargs):
        nonlocal count
        count+=1
        if failure: raise RuntimeError("synthetic interruption before later stage")
        return EvidenceMeaningBatch(records=[meaning(e)])
    async def trace(*args,**kwargs): pass
    o=SimpleNamespace(store=store,settings=Settings(_env_file=None),_llm_enhance=enhance,_trace_event=trace)
    seed=SimpleNamespace(symbol="FLT3")
    if failure:
        with pytest.raises(RuntimeError): await interpret_target(o,s.run_id,brief,seed,1,[e])
        with pytest.raises(ValueError,match="INCOMPLETE_NO_AUTOMATIC_REPLAY"):
            await interpret_target(o,s.run_id,brief,seed,1,[e])
        assert count==1
    else:
        await interpret_target(o,s.run_id,brief,seed,1,[e])
        await interpret_target(o,s.run_id,brief,seed,2,[e])
        assert count==2
    assert await store.verify_ledger(s.run_id)
    restored=await store.get_snapshot(s.run_id)
    assert restored.events and all(e.metadata.get("artifact_sha256") for e in restored.events)
