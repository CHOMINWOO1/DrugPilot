"""Isolated synthetic source bundles; never production evidence or paid I/O."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from drugpilot.agents.evidence_semantics import KEY, POLICY, EvidenceMeaningBatch, context_digest, digest, permits
from drugpilot.agents.molecular_scope import mentions, source_scope
from drugpilot.config import Settings
from drugpilot.orchestration.evidence_carryover import load_prior_bundles, merge_sources, preserved_meanings
from drugpilot.orchestration.semantic_interpretation import interpret_target
from drugpilot.orchestration.g0_followup import FollowupRetriever
from test_evidence_semantics import annotate, evidence, meaning


@pytest.mark.parametrize("text", ["FLT3/ITD", "FLT3-ITD", "FLT3 internal tandem duplication", "KIT/ITD"])
def test_explicit_itd_punctuation_is_label_not_construct_identity(text):
    target = "KIT" if text.startswith("KIT") else "FLT3"
    rows = mentions(text, target)
    assert len(rows) == 1 and rows[0]["label"] == target + ":ITD"
    assert not rows[0]["identity_verified"]
    assert text[rows[0]["start"]:rows[0]["end"]] == rows[0]["quote"]


@pytest.mark.parametrize("text", ["FLT3/ITDlike", "FLT3/D835", "BCL2/ITD", "FLT3 and ITD", "ITD alone"])
def test_partial_or_unbound_mutation_is_not_itd(text):
    assert not mentions(text, "FLT3")


def test_literal_alias_quotes_can_agree_without_changing_models_or_primary_gate(brief):
    e=evidence("FLT3 internal tandem duplication (FLT3/ITD) inhibition reduced viability in acute myeloid leukemia cells.")
    a=meaning(e, target_quote="FLT3 internal tandem duplication", intervention_quote="FLT3/ITD", molecular_scope_label="FLT3:ITD")
    b=a.model_copy(update={"target_quote":"FLT3/ITD"})
    old=e.model_dump_json()
    result=annotate(brief,e,a,b)
    assert permits(result,brief,"FLT3","direction","FLT3:ITD")
    assert not permits(result,brief,"FLT3","direction")
    assert not source_scope(result,"FLT3")["construct_identity_verified"]
    assert e.model_dump_json()==old


def fixture_store(brief, es=None, **updates):
    bundle={"run_id":"unit", "target":"FLT3", "iteration":1, "policy":POLICY,
        "context_sha256":context_digest(brief,"FLT3"), "interpretation_contract_sha256":"contract",
        "semantic_models":["a","b"],"evidence":[e.model_dump(mode="json") for e in es or [evidence()]]}
    bundle.update(updates)
    bid="ma1ev_"+digest(bundle)[:32]
    meta={"target":"FLT3","evidence_bundle_id":bid,
        "evidence_bundle_sha256":sha256(json.dumps(bundle,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()}
    snapshot=SimpleNamespace(run_id="unit",events=[SimpleNamespace(event_type="ma1_semantic_interpretation_completed",metadata=meta)])
    async def get_snapshot(rid): return snapshot
    async def read(rid, aid, kind):
        assert (rid,aid,kind)==("unit",bid,"MA1EvidenceBundle")
        return deepcopy(bundle)
    return SimpleNamespace(get_snapshot=get_snapshot,read_artifact=read),bundle,meta


@pytest.mark.asyncio
async def test_same_run_verified_bundle_preserved_with_lineage(brief):
    store,bundle,_=fixture_store(brief)
    old=deepcopy(bundle)
    records,lineage=await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])
    assert records["flt3"][0].external_id=="synthetic-paper-1"
    assert lineage[0]["new_retrieval"] is False and bundle==old


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"interpretation_contract_sha256":"legacy"},{"semantic_models":["x","b"]},
    {"context_sha256":"other"}])
async def test_legacy_model_or_context_has_no_reuse_authority(brief,changes):
    store,_,_=fixture_store(brief,**changes)
    assert await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])==({},[])


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["event_hash","payload","run","target"])
async def test_event_and_bundle_tampering_fails_closed(brief,field):
    store,bundle,meta=fixture_store(brief)
    if field=="event_hash": meta["evidence_bundle_sha256"]="0"*64
    elif field=="payload": bundle["evidence"][0]["title"]="tampered"
    elif field=="run": bundle["run_id"]="another-run"
    else: meta["target"]="BCL2"
    with pytest.raises(ValueError,match="BINDING_MISMATCH"):
        await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["fixture","mock","bad_hash","provenance"])
async def test_rehashed_bundle_does_not_make_fixture_or_bad_excerpt_valid(brief,change):
    e=evidence()
    if change=="bad_hash":e.excerpt_hash="0"*64
    elif change=="provenance":e.metadata["provenance_class"]="synthetic"
    else:e.metadata[change]=True
    store,_,_=fixture_store(brief,[e])
    with pytest.raises(ValueError,match="SOURCE_UNVERIFIABLE"):
        await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])


def metadata_only(brief):
    from drugpilot.agents.evidence_semantics import attach_interpretations
    e=evidence(external_id="synthetic-metadata-only")
    e.excerpt_hash=None
    e.metadata.update(excerpt=None,content_availability="metadata_only",direct_support_eligible=False)
    return attach_interpretations(brief,"FLT3",[e],[],EvidenceMeaningBatch(records=[]),EvidenceMeaningBatch(records=[]))[0]


def source_record_only(brief, *, carrier):
    from drugpilot.agents.evidence_semantics import attach_interpretations
    e=evidence(external_id="synthetic-source-record-only")
    e.supports="counter" if carrier=="detail" else "context"
    e.metadata.pop("excerpt",None)
    e.metadata.pop("content_availability",None)
    e.metadata.pop("direct_support_eligible",None)
    value=e.title if carrier=="title" else "bounded source detail without selected excerpt"
    if carrier=="detail":e.metadata["detail"]=value
    e.excerpt_hash=sha256(value.encode()).hexdigest()
    return attach_interpretations(brief,"FLT3",[e],[],EvidenceMeaningBatch(records=[]),EvidenceMeaningBatch(records=[]))[0]


@pytest.mark.asyncio
async def test_metadata_only_hit_is_audited_but_not_reused_as_evidence(brief):
    old=metadata_only(brief)
    store,bundle,_=fixture_store(brief,[evidence(),old]);before=deepcopy(bundle)
    records,lineage=await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])
    assert len(records["flt3"])==1 and bundle==before
    assert lineage[0]["omitted_non_evidence"]==[{"evidence_id":old.evidence_id,
        "reason":"source_record_without_bound_excerpt","original_record_preserved":True}]


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["title","detail"])
async def test_source_record_hash_without_selected_excerpt_is_preserved_not_reused(brief,carrier):
    old=source_record_only(brief,carrier=carrier)
    store,bundle,_=fixture_store(brief,[evidence(),old]);before=deepcopy(bundle)
    records,lineage=await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])
    assert len(records["flt3"])==1 and bundle==before
    assert lineage[0]["omitted_non_evidence"]==[{"evidence_id":old.evidence_id,
        "reason":"source_record_without_bound_excerpt","original_record_preserved":True}]


@pytest.mark.asyncio
async def test_unexplained_hash_without_selected_excerpt_still_fails_closed(brief):
    old=source_record_only(brief,carrier="title")
    old.excerpt_hash="0"*64
    store,_,_=fixture_store(brief,[old])
    with pytest.raises(ValueError,match="SOURCE_UNVERIFIABLE"):
        await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["fixture","mock","text_without_hash","hash_without_text",
    "synthetic","stale_binding","positive_meaning","claimed_support"])
async def test_metadata_omission_never_hides_corruption_or_false_evidence(brief,change):
    e=metadata_only(brief)
    if change in {"fixture","mock"}: e.metadata[change]=True
    elif change=="text_without_hash": e.metadata["excerpt"]="unhashed source"
    elif change=="hash_without_text": e.excerpt_hash="0"*64
    elif change=="synthetic": e.metadata["provenance_class"]="synthetic"
    elif change=="stale_binding": e.metadata[KEY]["source_sha256"]="0"*64
    elif change=="positive_meaning": e.metadata[KEY]["extractor"]=meaning(evidence()).model_dump()
    else:e.metadata["direct_support_eligible"]=True
    store,_,_=fixture_store(brief,[e])
    with pytest.raises(ValueError,match="SOURCE_UNVERIFIABLE"):
        await load_prior_bundles(store,"unit",brief,contract="contract",models=["a","b"])


@pytest.mark.asyncio
async def test_loader_only_compatibility_preserves_disagreement_not_model_resampling(brief):
    from drugpilot.orchestration.evidence_carryover import metadata_omission_compatible_contract
    from drugpilot.agents.evidence_semantics import source_records
    e=annotate(brief,evidence(),b=meaning(evidence(),outcome="resistance"))
    store,_,_=fixture_store(brief,[e,metadata_only(brief)],
        interpretation_contract_sha256=metadata_omission_compatible_contract())
    records,lineage=await load_prior_bundles(store,"unit",brief,contract="new",models=["a","b"])
    assert lineage[0]["loader_only_compatibility_used"]
    assert records["flt3"][0].metadata[KEY]["status"]=="disagreement"
    assert len(preserved_meanings(brief,"FLT3",source_records(records["flt3"]),records["flt3"]))==1


def test_merge_preserves_current_and_prior_sources_without_double_count_or_mutation():
    a,b=evidence(),evidence(external_id="synthetic-paper-2")
    before=[e.model_dump_json() for e in [a,b]]
    assert [e.evidence_id for e in merge_sources([a],[a,b])]==[a.evidence_id,b.evidence_id]
    assert [e.model_dump_json() for e in [a,b]]==before
    bad=a.model_copy(deep=True);bad.metadata["excerpt"]+="unbound"
    with pytest.raises(ValueError,match="SOURCE_ID_CONFLICT"):merge_sources([a],[bad])
    with pytest.raises(ValueError,match="DUPLICATE_CURRENT_ID"):merge_sources([a,a],[])


def test_followup_empty_response_does_not_erase_prior_sources():
    e=evidence();seed=SimpleNamespace(symbol="FLT3")
    delegate=SimpleNamespace(public=SimpleNamespace(retrieve_g0_followup=lambda *a,**k:((),[])),retrieve=lambda *a:())
    plan={"plan_id":"unit","source_elapsed_seconds":120,"tasks":[{"target":"FLT3","status":"scheduled","role":"direction","source_paths":[]}]}
    wrapper=FollowupRetriever(delegate,plan,{"flt3":[e]})
    assert wrapper.retrieve(None,seed,2)[0]==e
    wrapper.plan["tasks"]=[]
    assert wrapper.retrieve(None,seed,3)[0]==e
    assert wrapper.history_entry()["attempted_source_task_keys"]==[]


@pytest.mark.asyncio
@pytest.mark.parametrize("result_kind", ["agreement","disagreement","missing"])
async def test_prior_decision_is_not_resampled_in_later_revision(brief,result_kind):
    e=evidence()
    if result_kind=="missing":
        from drugpilot.agents.evidence_semantics import attach_interpretations,source_records
        old=attach_interpretations(brief,"FLT3",[e],source_records([e]),EvidenceMeaningBatch(records=[]),EvidenceMeaningBatch(records=[]))[0]
    else:old=annotate(brief,e,meaning(e),meaning(e,**({"outcome":"resistance"} if result_kind=="disagreement" else {})))
    async def enhance(*a,**k):raise AssertionError("prior answer must not be resampled")
    events=[]
    async def event(*a,**k):events.append(k["metadata"])
    runtime=SimpleNamespace(settings=Settings(_env_file=None),_llm_enhance=enhance,_trace_event=event)
    result=await interpret_target(runtime,"unit",brief,SimpleNamespace(symbol="FLT3"),2,[old],preserved_evidence=[old])
    assert result[0].metadata[KEY]==old.metadata[KEY]
    assert events[0]["new_model_calls"]==0 and events[0]["preserved_interpretation_count"]==1
    if result_kind!="agreement":assert not permits(result[0],brief,"FLT3","direction")


@pytest.mark.asyncio
async def test_only_new_source_is_blindly_interpreted(brief):
    old=annotate(brief,evidence(),b=meaning(evidence(),outcome="resistance"))
    new=evidence(external_id="synthetic-new")
    calls=[]
    async def enhance(*a,**k):
        calls.append(k["prompt"])
        assert "synthetic-new" in k["prompt"] and "synthetic-paper-1" not in k["prompt"]
        return EvidenceMeaningBatch(records=[meaning(new)])
    async def event(*a,**k):pass
    runtime=SimpleNamespace(settings=Settings(_env_file=None),_llm_enhance=enhance,_trace_event=event)
    result=await interpret_target(runtime,"unit",brief,SimpleNamespace(symbol="FLT3"),2,[new,old],preserved_evidence=[old])
    assert len(calls)==2 and result[1].metadata[KEY]["status"]=="disagreement"
    assert permits(result[0],brief,"FLT3","direction")


def test_conflicting_or_tampered_cached_answers_fail_closed(brief):
    from drugpilot.agents.evidence_semantics import source_records
    a=annotate(brief,evidence());b=annotate(brief,evidence(),b=meaning(evidence(),outcome="resistance"))
    with pytest.raises(ValueError,match="MEANING_CONFLICT"):
        preserved_meanings(brief,"FLT3",source_records([a]),[a,b])
    a.metadata[KEY]["context_sha256"]="tampered"
    with pytest.raises(ValueError,match="MEANING_BINDING_MISMATCH"):
        preserved_meanings(brief,"FLT3",source_records([a]),[a])


def disputed_packet(brief):
    from test_g0_followup import packet,verdict
    from drugpilot.orchestration.g0_followup import build_plan
    e=evidence(external_id="12345")
    e=annotate(brief,e,b=meaning(e,disease_context="unknown"))
    p=packet(brief,("FLT3",))
    plan=build_plan(brief,p,verdict(),rework_loops=0,target_limit=1)
    keys=[path["source_task_key"] for t in plan["tasks"] for path in t["source_paths"]]
    path=next(path for t in plan["tasks"] if t["role"]=="direction" for path in t["source_paths"] if path["source"]=="PubMed")
    p.metadata["g0_followup_history"]=[{"plan_id":"prior","attempted_source_task_keys":keys,"observations":[
        {"target":"FLT3","source":"PubMed","role":"direction","attempted":True,"status":"completed_with_results",
         "plan_id":"prior","source_task_keys":[path["source_task_key"]],"trace":{"screening_entries":[
             {"status":"selected","rejection_codes":[],"external_id":e.external_id,
              "extraction_observation":{"excerpt_selected":True,"excerpt_sha256":e.excerpt_hash}}]}}]}]
    p.metadata["rejected_screening_trace"]["entries"][0]["unique_external_records"]=[
        {"source":"PubMed","external_id":e.external_id,"provenance_classes":["external_live"],"evidence_units":[
            {"evidence_id":e.evidence_id,"bounded_excerpt":e.metadata["excerpt"],"excerpt_hash":e.excerpt_hash,KEY:e.metadata[KEY]}]}]
    return p,e,plan


def test_disputed_selected_source_advances_only_pubmed_not_epmc(brief):
    from test_g0_followup import verdict
    from drugpilot.orchestration.g0_followup import build_plan
    from drugpilot.orchestration.evidence_carryover import ALTERNATIVE_KEY,alternative_selections,verify_alternatives
    p,e,old=disputed_packet(brief)
    before=p.model_dump_json()
    proofs=alternative_selections(brief,p,"FLT3")
    assert verify_alternatives(brief,"FLT3",proofs,[e])==["12345"]
    plan=build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)
    task=next(t for t in plan["tasks"] if t["role"]=="direction")
    assert task[ALTERNATIVE_KEY]==proofs and task["status"]=="scheduled"
    assert task["task_key"]==next(t["task_key"] for t in old["tasks"] if t["role"]=="direction")
    assert {x["source"]:x["status"] for x in task["source_paths"]}=={"Europe PMC":"already_attempted","PubMed":"available"}
    assert p.model_dump_json()==before
    assert not build_plan(brief,p,verdict(),rework_loops=2,target_limit=1)["eligible"]


@pytest.mark.parametrize("mutation", ["not_selected","other_role","other_context","bad_hash","fixture","agreement"])
def test_unbound_or_nondisputed_trace_cannot_authorize_alternative(brief,mutation):
    from drugpilot.orchestration.evidence_carryover import alternative_selections
    p,e,_=disputed_packet(brief)
    o=p.metadata["g0_followup_history"][0]["observations"][0]
    r=p.metadata["rejected_screening_trace"]["entries"][0]["unique_external_records"][0]
    u=r["evidence_units"][0]
    if mutation=="not_selected":o["trace"]["screening_entries"][0]["status"]="rejected"
    elif mutation=="other_role":o["role"]="association"
    elif mutation=="other_context":u[KEY]["context_sha256"]="other"
    elif mutation=="bad_hash":u["bounded_excerpt"]+="tampered"
    elif mutation=="fixture":r["provenance_classes"]=["fixture"]
    else:u[KEY]["status"]="grounded_agreement"
    assert alternative_selections(brief,p,"FLT3")==[]


def test_alternative_proof_is_reverified_before_any_http(brief):
    from drugpilot.orchestration.evidence_carryover import alternative_selections,verify_alternatives
    p,e,_=disputed_packet(brief)
    proofs=alternative_selections(brief,p,"FLT3")
    with pytest.raises(ValueError,match="NOT_BOUND_TO_PRIOR_SOURCE"):
        verify_alternatives(brief,"FLT3",proofs,[])
    proofs[0]["pmid"]="67890"
    with pytest.raises(ValueError,match="NOT_BOUND_TO_PRIOR_SOURCE"):
        verify_alternatives(brief,"FLT3",proofs,[e])


def test_public_lane_receives_prior_context_and_skips_only_bound_disputed_id(brief,monkeypatch):
    from test_g0_followup import verdict
    from drugpilot.agents.ma1 import TargetSeed
    from drugpilot.domain.models import HypothesisTrack
    from drugpilot.orchestration.g0_followup import build_plan
    from drugpilot.tools.public_biology import PublicBiologyRetriever
    p,e,_=disputed_packet(brief)
    plan=build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)
    plan["tasks"]=[t for t in plan["tasks"] if t["role"]=="direction"]
    public=PublicBiologyRetriever(enabled_sources=("PubMed",))
    received=[]
    def lookup(*args,**kw):
        received.append(kw["excluded_ids"])
        return SimpleNamespace(records=(),status="empty",stop_reason="no_alternative",trace={})
    monkeypatch.setattr(public,"_pubmed_primary_direction_supplement",lookup)
    wrapper=FollowupRetriever(SimpleNamespace(public=public),plan,{"flt3":[e]})
    records=wrapper.retrieve(brief,TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1),2)
    assert received==[{"12345"}] and records==tuple([e])
    assert wrapper.observations[0]["prior_source_evidence_preserved"] is True


@pytest.mark.asyncio
async def test_real_sqlite_bundle_roundtrip_retains_disagreement_without_paid_replay(tmp_path,brief):
    from drugpilot.storage.sqlite import SQLiteRunStore
    from drugpilot.domain.models import RunSnapshot,RunMode,RunStage,ExecutionEvent
    from drugpilot.orchestration.semantic_interpretation import interpretation_contract
    store=SQLiteRunStore(tmp_path/"carryover-unit.sqlite3")
    await store.save_snapshot(RunSnapshot(run_id="roundtrip",mode=RunMode.MOCK,stage=RunStage.MA1,status_message="unit",brief=brief))
    e=evidence();calls=[]
    async def enhance(*a,**kw):
        calls.append(kw["component"])
        return EvidenceMeaningBatch(records=[meaning(e,**({"outcome":"resistance"} if "Verifier" in kw["component"] else {}))])
    async def event(run_id,kind,message,**kw):
        s=await store.get_snapshot(run_id)
        s.events.append(ExecutionEvent(run_id=run_id,stage=s.stage,event_type=kind,message=message,**kw))
        await store.save_snapshot(s)
    settings=Settings(_env_file=None)
    runtime=SimpleNamespace(store=store,settings=settings,_llm_enhance=enhance,_trace_event=event)
    first=await interpret_target(runtime,"roundtrip",brief,SimpleNamespace(symbol="FLT3"),1,[e])
    records,lineage=await load_prior_bundles(store,"roundtrip",brief,contract=interpretation_contract(),models=[settings.terra_model,settings.sol_model])
    assert len(lineage)==1 and records["flt3"]==tuple(first)
    second=await interpret_target(runtime,"roundtrip",brief,SimpleNamespace(symbol="FLT3"),2,records["flt3"],preserved_evidence=records["flt3"])
    assert len(calls)==2 and second[0].metadata[KEY]["status"]=="disagreement"
    # Both original and later bundle events remain bound and readable.
    again,history=await load_prior_bundles(store,"roundtrip",brief,contract=interpretation_contract(),models=[settings.terra_model,settings.sol_model])
    assert len(history)==2 and len(again["flt3"])==1
