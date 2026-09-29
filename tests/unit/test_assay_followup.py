"""Isolated synthetic transport tests. Never live candidates or evidence fixtures."""
from copy import deepcopy
from hashlib import sha256
from html import escape
from types import SimpleNamespace

import httpx
import pytest

from drugpilot.agents.ma1 import TargetSeed, _assign_claim_evidence
from drugpilot.agents.evidence_semantics import source_records
from drugpilot.domain.models import HypothesisTrack
from drugpilot.orchestration.g0_followup import build_plan, FollowupRetriever
from drugpilot.tools.assay_followup import (ROLE, assay_spans, query_for, required_assays, retrieve_assay_context)
from drugpilot.tools.evidence_gap import transient_failure
from drugpilot.tools import public_biology as pb
from test_g0_followup import packet, verdict


def seed(target="FLT3"):
    return TargetSeed(target,target,HypothesisTrack.LITERATURE,1)


def requested(brief):
    b = brief.model_copy(deep=True)
    b.human_priorities = ["Preserve exact source and required assay provenance."]
    return b


class Client:
    def __init__(self, texts=None, *, count=1, failure=None, pubtype="Journal Article"):
        self.ids = [str(1000+i) for i in range(count)]
        self.texts = texts or ["FLT3 inhibition had an IC50 of 12 nM in acute myeloid leukemia cells."]*count
        self.calls, self.failure, self.pubtype = [], failure, pubtype

    def get(self, url, **kwargs):
        params = kwargs["params"]
        self.calls.append((url,params))
        if self.failure == "timeout" or (self.failure == "partial_fetch" and len(self.calls) == 4):
            raise httpx.ReadTimeout("secret-key-must-not-be-logged")
        if self.failure == "503":
            return httpx.Response(503,request=httpx.Request("GET",url))
        if url == pb.PUBMED_SEARCH_URL:
            result = {"esearchresult":{"count":str(len(self.ids)),"idlist":self.ids}}
            return httpx.Response(200,json=result,request=httpx.Request("GET",url))
        ids = params["id"].split(",")
        if url == pb.PUBMED_SUMMARY_URL:
            result = {"uids":ids,**{i:{"uid":i,"title":"Primary assay measurement","pubtype":[self.pubtype]} for i in ids}}
            if self.failure == "summary_id":
                result[ids[0]]["uid"] = "99999"
            return httpx.Response(200,json={"result":result},request=httpx.Request("GET",url))
        assert url == pb.PUBMED_FETCH_URL
        if self.failure == "fetch_order":
            ids = list(reversed(ids))
        articles = [f'<PubmedArticle><MedlineCitation><PMID>{i}</PMID><Article><Abstract><AbstractText Label="RESULTS">{escape(self.texts[self.ids.index(i)])}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>' for i in ids]
        xml = "<PubmedArticleSet>"+"".join(articles)+"</PubmedArticleSet>"
        return httpx.Response(200,text=xml,request=httpx.Request("GET",url))


def public(client, **kwargs):
    return pb.PublicBiologyRetriever(client=client,enabled_sources=("PubMed",),
        ncbi_rate_limiter=SimpleNamespace(acquire=lambda **k:None),**kwargs)


@pytest.mark.parametrize("target,disease,assay,unit",[
    ("FLT3","Acute myeloid leukemia","IC50","12 nM"),
    ("ABL1","Chronic myeloid leukemia","Ki","0.8 µM"),
    ("ALK","Non-small cell lung cancer","Kd","25 pM"),
    ("RET","Thyroid cancer","EC50","2.3 μM"),
])
def test_source_spans_are_exact_context_not_measurements(brief,target,disease,assay,unit):
    b = brief.model_copy(deep=True)
    b.disease = disease
    b.constraints.required_assays = [assay]
    text = f"Other background. {target} had an {assay} of {unit} in cells."
    rows = assay_spans(text,target,required_assays(b))
    assert len(rows)==1
    r = rows[0]
    assert text[r["start"]:r["end"]]==r["excerpt"]
    assert sha256(r["excerpt"].encode()).hexdigest()==r["excerpt_sha256"]
    assert not r["measurement_assignment_verified"] and not r["disease_model_match_verified"]
    for m in r["assay_mentions"]+r["quantity_mentions"]:
        assert text[m["start"]:m["end"]]==m["quote"]
    assert assay in query_for(b,seed(target)) and disease.split()[0] in query_for(b,seed(target))


@pytest.mark.parametrize("text",[
    "BCL2 had an IC50 of 12 nM.", "FLT3-ITD was studied. IC50 was 12 nM.",
    "FLT3 kinase was measured at 12 nM.", "FLT3 had an IC50 of 12 percent.",
    "FLT3 had an IC50 without a numerical value.", "FLT30 had an IC50 of 12 nM.",
    "FLT3 had an IC50 of 12 nM. "+"word "*300,
])
def test_no_cross_sentence_or_partial_token_quantity_mapping(text):
    rows = assay_spans(text,"FLT3",("IC50","Ki","Kd"))
    if text.startswith("FLT3 had an IC50 of 12 nM."):
        assert len(rows)==1 and len(rows[0]["excerpt"])<100
    else:
        assert rows==[]


def test_oversize_sentence_and_unsupported_endpoint_do_not_create_excerpt(brief):
    assert assay_spans("FLT3 "+"word "*300+" IC50 12 nM.","FLT3",("IC50",)) == []
    b=brief.model_copy(deep=True)
    b.constraints.required_assays=['IC50" OR anything','unknown']
    assert not required_assays(b)
    with pytest.raises(ValueError,match="NO_SUPPORTED"):
        query_for(b,seed())


@pytest.mark.parametrize("text",[
    "The ABL1 KD was assayed with drug at 100 nM.",
    "ABL1 was assayed with KI at 100 nM.",
])
def test_ambiguous_uppercase_domain_or_reagent_is_not_an_assay_token(text):
    assert assay_spans(text,"ABL1",("IC50","Ki","Kd"))==[]


def test_ic50_context_does_not_mislabel_kd_domain_as_dissociation_assay():
    rows=assay_spans("The ABL1 KD was measured with IC50 of 12 nM.","ABL1",("IC50","Ki","Kd"))
    assert len(rows)==1 and [m["quote"] for m in rows[0]["assay_mentions"]]==["IC50"]


def test_assay_diagnostic_reports_context_not_primary_support(brief):
    records,rows=public(Client()).retrieve_g0_followup(brief,seed(),roles=["assay_context"])
    gap=rows[0]["evidence_gap"]
    assert len(records)==1 and gap["exclusive_record_buckets"]=={"assay_context_unassigned_measurement":1}
    assert "preserve_assay_context_without_assigning_potency" in gap["next_actions"]
    assert gap["outcome"]=="records_returned_not_gate_pass"
    assert gap["matched_context_documents"]==gap["retained_context_documents"]==1
    assert gap["omitted_matching_context_documents"]==0


def test_real_adapter_retains_context_with_no_primary_or_semantic_model_authority(brief):
    client=Client()
    result=retrieve_assay_context(public(client),brief,seed())
    assert result.status=="completed_with_context" and len(result.records)==1
    e=result.records[0]
    assert e.supports=="context" and e.metadata["role_specific_record"]==ROLE
    assert not e.metadata["verified_direct_primary"] and not e.metadata["direct_support_eligible"]
    assert e.metadata["provenance_class"]=="external_live" and not e.metadata["mock"]
    assert not source_records(result.records)
    claims=_assign_claim_evidence(brief,target_symbol="FLT3",evidence=result.records)
    assert not claims.direction_primary_ids and not claims.association_primary_ids
    assert len(client.calls)==result.trace["http_requests_used"]==3
    assert result.trace["bioactivity_records_generated"]==0
    assert all(c["fixture"] is False for c in result.trace["http_call_provenance"])
    # A context label is not the only guard: a forged supports flag is insufficient.
    forged=e.model_copy(update={"supports":"support"},deep=True)
    assert not _assign_claim_evidence(brief,target_symbol="FLT3",evidence=[forged]).direction_primary_ids


def test_all_documents_are_screened_and_retention_omissions_are_counted(brief):
    client=Client(count=41)
    result=retrieve_assay_context(public(client,max_records_per_source=2),brief,seed())
    assert len(result.records)==2
    assert result.trace["records_screened"]==41 and result.trace["unsearched_result_count"]==0
    assert result.trace["matched_document_count"]==41 and result.trace["omitted_matching_document_count"]==39
    assert len(client.calls)==5


@pytest.mark.parametrize("failure",["timeout","503","summary_id","fetch_order","partial_fetch"])
def test_errors_never_return_partial_context_or_expose_exception_text(brief,failure):
    result=retrieve_assay_context(public(Client(count=21,failure=failure)),brief,seed())
    assert result.status=="error" and result.records==()
    assert "secret-key" not in str(result.trace)
    row={"source":"PubMed","attempted":True,"record_count":0,"status":result.status,"trace":result.trace}
    assert transient_failure(row)==(failure in {"timeout","503","partial_fetch"})


@pytest.mark.parametrize("pubtype",["Review","Editorial","Meta-Analysis"])
def test_nonprimary_publication_never_returned(brief,pubtype):
    result=retrieve_assay_context(public(Client(pubtype=pubtype)),brief,seed())
    assert not result.records and result.trace["records_screened"]==1


def test_no_hits_and_local_budget_exhaustion_are_not_retried(brief):
    empty=retrieve_assay_context(public(Client(count=0)),brief,seed())
    assert empty.status=="completed_empty" and empty.trace["search_result_count"]==0
    ticks=iter([0,121])
    client=Client()
    timed=retrieve_assay_context(public(client,clock=lambda:next(ticks)),brief,seed())
    assert not client.calls and timed.records==() and timed.status=="error"
    assert not transient_failure({"source":"PubMed","attempted":True,"record_count":0,"status":"error","trace":timed.trace})


def test_assay_task_added_last_without_changing_primary_task_keys_or_priorities(brief):
    b=requested(brief)
    p=packet(b,("FLT3",))
    before=p.model_dump_json()
    plan=build_plan(b,p,verdict(),rework_loops=0,target_limit=1)
    assert [t["role"] for t in plan["tasks"]]==["association","direction","assay_context","assay_identity","assay_fulltext"]
    context=next(t for t in plan["tasks"] if t["role"]=="assay_context")
    assert [s["source"] for s in context["source_paths"]]==["PubMed"]
    assert context["role_authority"]=="required_assay_search_context_not_primary_gate_evidence"
    assert [s["source"] for s in next(t for t in plan["tasks"] if t["role"]=="assay_identity")["source_paths"]]==["ChEMBL"]
    assert [s["source"] for s in plan["tasks"][-1]["source_paths"]]==["Europe PMC"]
    assert {t["priority_value"] for t in plan["tasks"]}=={2}
    assert p.model_dump_json()==before
    p.metadata["g0_followup_history"]=[{"attempted_task_keys":[t["task_key"] for t in plan["tasks"]]}]
    later=build_plan(b,p,verdict(),rework_loops=1,target_limit=1)
    assert all(t["status"]=="already_attempted" for t in later["tasks"])


def test_assay_task_respects_existing_budget_lineage_and_allowlist(brief):
    b=requested(brief)
    p=packet(b,("FLT3",))
    assert not build_plan(b,p,verdict(),rework_loops=2,target_limit=1)["eligible"]
    p.metadata["revision_lineage"]={"scientific_changed":False}
    assert not build_plan(b,p,verdict(),rework_loops=0,target_limit=1)["eligible"]
    b.data_policy.allowed_sources=["Europe PMC"]
    assert not any(t["role"]=="assay_context" for t in build_plan(b,p,verdict(),rework_loops=0,target_limit=1)["tasks"])


def test_wrapper_calls_only_assay_lane_when_other_paths_consumed(brief):
    b=requested(brief)
    p=packet(b,("FLT3",))
    plan=build_plan(b,p,verdict(),rework_loops=0,target_limit=1)
    p.metadata["g0_followup_history"]=[{"attempted_task_keys":[t["task_key"] for t in plan["tasks"] if t["role"]!="assay_context"]}]
    plan=build_plan(b,p,verdict(),rework_loops=1,target_limit=1)
    client=Client()
    wrapper=FollowupRetriever(SimpleNamespace(public=public(client)),plan)
    assert len(wrapper.retrieve(b,seed(),1))==1
    assert len(wrapper.observations)==1 and wrapper.observations[0]["role"]=="assay_context"
    assert len(wrapper.history_entry()["attempted_source_task_keys"])==1


def test_adapter_source_and_outer_budget_guards_prevent_assay_calls(brief):
    client=Client()
    b=brief.model_copy(deep=True)
    b.data_policy.allowed_sources=[]
    records,rows=public(client).retrieve_g0_followup(b,seed(),roles=["assay_context"])
    assert records==() and not client.calls and rows[0]["reason_code"]=="SOURCE_NOT_ALLOWED"
    with pytest.raises(ValueError,match="only by PubMed"):
        public(client).retrieve_g0_followup(brief,seed(),roles=["assay_context"],source_paths=[{"source":"Europe PMC","role":"assay_context"}])
