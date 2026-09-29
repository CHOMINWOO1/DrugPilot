"""Synthetic isolated adapter tests, never scientific fixtures for a live run."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from drugpilot.tools.evidence_gap import content_observation, diagnose_observation, transient_failure
from drugpilot.tools.public_biology import _publication_content_from_sections, PublicBiologyRetriever
from drugpilot.orchestration.g0_followup import FollowupRetriever, build_plan, public_path_retry
from drugpilot.agents.ma1 import TargetSeed
from drugpilot.domain.models import HypothesisTrack
from test_g0_followup import packet, verdict


def row(entries=(), **updates):
    return {"source":"PubMed", "role":"direction", "attempted":True, "record_count":0,
        "status":"completed_bounded_without_verified_result", "reason_code":"returned_candidates_exhausted_without_verified_direct_primary",
        "trace":{"status":"completed_bounded_without_verified_result", "records_screened":len(entries),
            "search_result_count":len(entries), "unsearched_result_count":0, "screening_entries":list(entries)},
        **updates}


def entry(codes, obs=None):
    return {"external_id":"12345", "status":"rejected", "rejection_codes":codes,
            **({"extraction_observation":obs} if obs is not None else {})}


def content(text):
    return _publication_content_from_sections([("RESULTS", text)],
        target_symbol="FLT3", disease="acute myeloid leukemia", intervention_direction="inhibit")


def test_received_abstract_without_relevant_excerpt_is_not_missing_source():
    c = content("We evaluated a different disease and a different target.")
    assert c.content_hash and not c.excerpt
    obs = content_observation(c)
    assert obs["abstract_received"] and not obs["excerpt_selected"]
    assert "text" not in obs and "excerpt" not in obs
    result = diagnose_observation(row([entry(["abstract_excerpt_unavailable"], obs)]))
    assert result["exclusive_record_buckets"] == {"abstract_received_excerpt_not_selected":1}
    assert "try_unattempted_licensed_fulltext_path" in result["next_actions"]
    assert not result["biological_absence_established"]


def test_no_abstract_and_legacy_ambiguous_excerpt_are_distinct():
    obs = content_observation(content(""))
    result = diagnose_observation(row([
        entry(["abstract_excerpt_unavailable"], obs), entry(["abstract_excerpt_unavailable"])]))
    assert result["exclusive_record_buckets"] == {
        "abstract_not_available":1, "historical_extraction_state_unknown":1}


@pytest.mark.parametrize("codes,bucket", [
    (["review_publication_type"], "publication_ineligible"),
    (["primary_publication_type_not_verified"], "publication_metadata_unverified"),
    (["measured_endpoint_missing", "favorable_endpoint_polarity_missing"], "selected_excerpt_criteria_unmet"),
    (["counter_signal", "unfavorable_endpoint_polarity"], "counter_or_mixed_context_not_direct_refutation"),
    (["invented_code"], "unclassified_observation"),
])
def test_record_buckets_do_not_double_count_or_create_scientific_verdict(codes, bucket):
    r = row([entry(codes)])
    before = deepcopy(r)
    result = diagnose_observation(r)
    assert result["exclusive_record_buckets"] == {bucket:1}
    assert result["classified_entry_count"] == 1
    assert not result["scientific_verdict_changed"]
    assert r == before


def test_pending_unit_is_not_counted_as_verified_support():
    e = {**entry(["primary_result_context_missing"]), "document_unit_source_preserved":True}
    r = diagnose_observation(row([e], record_count=1))
    assert r["outcome"] == "document_unit_validation_pending"


def test_zero_hits_is_not_literature_absence_and_unknown_does_not_become_zero():
    r = diagnose_observation(row())
    assert r["outcome"] == "no_hits_in_this_query"
    assert not r["biological_absence_established"]
    unknown = diagnose_observation({"source":"PubMed", "attempted":True,"status":"empty"})
    assert unknown["outcome"] == "insufficient_diagnostic_telemetry"
    assert unknown["search_result_count"] is None


def test_production_zero_hit_status_is_diagnosed_and_never_retried(monkeypatch, brief):
    import httpx
    public = PublicBiologyRetriever(enabled_sources=("PubMed",))
    def get(url, **kwargs):
        return httpx.Response(200, request=httpx.Request("GET", url),
            json={"esearchresult":{"count":"0", "idlist":[]}})
    monkeypatch.setattr(public, "_ncbi_get", get)
    records, rows = public.retrieve_g0_followup(brief,
        TargetSeed("FLT3", "FLT3", HypothesisTrack.LITERATURE, 1), roles=["direction"])
    assert records == ()
    assert rows[0]["status"] == "completed_empty"
    assert rows[0]["evidence_gap"]["outcome"] == "no_hits_in_this_query"
    assert rows[0]["evidence_gap"]["search_result_count"] == 0
    assert not transient_failure(rows[0])


@pytest.mark.parametrize("failure", [
    {"status":"failed","reason_code":"TimeoutError"},
    {"status":"error","trace":{"error_type":"ReadTimeout"}},
    {"status":"error","trace":{"http_call_provenance":[{"status":"error","status_code":503}]}},
])
def test_typed_transient_public_failures_have_bounded_retry_hint(failure):
    r = row(**failure)
    assert transient_failure(r)
    assert diagnose_observation(r)["outcome"] == "transient_retrieval_failure"


@pytest.mark.parametrize("overrides", [
    {"attempted":False}, {"record_count":1}, {"record_count":False}, {"record_count":None},
    {"source":"OpenAI"}, {"status":"empty"}, {"reason_code":"HTTP 503 please retry"},
    {"reason_code":"SchemaValidationError"},
    {"reason_code":"HTTPStatusError", "trace":{"http_call_provenance":[{"status":"error","status_code":401}]}},
    {"reason_code":"HTTPStatusError", "trace":{"http_call_provenance":[{"status":"error","status_code":"503"}]}},
])
def test_ambiguous_nontransient_paid_or_nonzero_results_do_not_grant_retry(overrides):
    r = row(status="failed", reason_code="TimeoutError")
    r.update(overrides)
    assert not transient_failure(r)


def failed_history(brief, p, *, code="TimeoutError", status="failed"):
    plan = build_plan(brief, p, verdict(), rework_loops=0, target_limit=1)
    def retrieve(*a, **kwargs):
        return (), [row(role=role, source="PubMed", status=status, reason_code=code)
                    for role in kwargs["roles"]]
    wrapper = FollowupRetriever(SimpleNamespace(public=SimpleNamespace(retrieve_g0_followup=retrieve)), plan)
    wrapper.retrieve(brief, TargetSeed("FLT3", "FLT3", HypothesisTrack.LITERATURE, 1), 1)
    return wrapper.history_entry()


def test_bound_transient_retries_once_without_resetting_task_keys(brief):
    p = packet(brief, ("FLT3",))
    hist = failed_history(brief, p)
    p.metadata["g0_followup_history"] = [hist]
    plan = build_plan(brief, p, verdict(), rework_loops=1, target_limit=1)
    pubmed = [path for t in plan["tasks"] for path in t["source_paths"] if path["source"]=="PubMed"]
    assert all(path["status"] == "available" and path["next_attempt_number"]==2 for path in pubmed)
    assert {path["source_task_key"] for path in pubmed} == set(hist["attempted_source_task_keys"])
    second = deepcopy(hist)
    second["plan_id"] = plan["plan_id"]
    for r in second["observations"]: r["plan_id"] = second["plan_id"]
    p.metadata["g0_followup_history"].append(second)
    final = build_plan(brief, p, verdict(), rework_loops=1, target_limit=1)
    assert all(path["status"]=="already_attempted" and path["retry_decision"]["reason"]=="PUBLIC_PATH_RETRY_CAP_REACHED"
               for t in final["tasks"] for path in t["source_paths"] if path["source"]=="PubMed")


@pytest.mark.parametrize("code,status", [("no_verified_records","empty"),("SchemaError","failed"),("HTTPStatusError","failed")])
def test_normal_empty_or_nontransient_result_stays_consumed(brief, code, status):
    p = packet(brief, ("FLT3",))
    p.metadata["g0_followup_history"] = [failed_history(brief,p,code=code,status=status)]
    plan = build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)
    assert all(path["status"]=="already_attempted" for t in plan["tasks"]
               for path in t["source_paths"] if path["source"]=="PubMed")


def test_budget_lineage_and_forbidden_guards_still_apply_to_transient_retry(brief):
    p = packet(brief, ("FLT3",))
    p.metadata["g0_followup_history"] = [failed_history(brief,p)]
    assert not build_plan(brief,p,verdict(),rework_loops=brief.constraints.max_rework_loops,target_limit=1)["eligible"]
    p.metadata["revision_lineage"] = {"scientific_changed":False}
    assert not build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)["eligible"]
    p.metadata.pop("revision_lineage")
    p.excluded[0].reason_codes.append("FORBIDDEN_TARGET")
    assert build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)["tasks"] == []


@pytest.mark.parametrize("mutation", ["target","source","plan_id","role","keys_type"])
def test_unbound_history_cannot_enable_retry(brief, mutation):
    p = packet(brief, ("FLT3",))
    h = failed_history(brief,p)
    for r in h["observations"]:
        if mutation=="keys_type": r["source_task_keys"] = str(r["source_task_keys"])
        else:r[mutation]="unbound"
    p.metadata["g0_followup_history"]=[h]
    plan=build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)
    assert all(path["status"]=="already_attempted" for t in plan["tasks"]
               for path in t["source_paths"] if path["source"]=="PubMed")


def test_duplicate_history_cannot_reset_or_inflate_attempt_counter(brief):
    p=packet(brief,("FLT3",))
    h=failed_history(brief,p)
    p.metadata["g0_followup_history"]=[h,deepcopy(h)]
    plan=build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)
    assert all(path["retry_decision"]["attempts"]==1 for t in plan["tasks"]
               for path in t["source_paths"] if path["source"]=="PubMed")


def test_forged_diagnostic_does_not_grant_retry(brief):
    p=packet(brief,("FLT3",))
    h=failed_history(brief,p,code="no_verified_records",status="empty")
    for r in h["observations"]:r["evidence_gap"]={"outcome":"transient_retrieval_failure"}
    p.metadata["g0_followup_history"]=[h]
    plan=build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)
    assert all(path["status"]=="already_attempted" for t in plan["tasks"]
               for path in t["source_paths"] if path["source"]=="PubMed")


def test_actual_adapter_records_diagnostic_without_fabricating_evidence(monkeypatch,brief):
    public=PublicBiologyRetriever(enabled_sources=("PubMed",))
    def failure(*args,**kwargs):raise TimeoutError("secret message must not be exposed")
    monkeypatch.setattr(public,"_pubmed_primary_direction_supplement",failure)
    records, rows=public.retrieve_g0_followup(brief,TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1),roles=["direction"])
    assert records == ()
    assert rows[0]["evidence_gap"]["outcome"]=="transient_retrieval_failure"
    assert "secret" not in str(rows)


def test_partial_oa_errors_are_visible_even_when_abstract_records_returned():
    r=row(source="Europe PMC",role="oa_context",status="completed",record_count=5,
        oa_trace={"status":"completed_with_errors_without_evidence",
            "error_reason_codes":["parent_references_HTTPStatusError"],
            "observed":{"parent_fulltexts_parsed":1,"child_fulltexts_parsed":0}})
    d=diagnose_observation(r)
    assert d["outcome"]=="records_returned_with_source_errors"
    assert d["oa_parent_fulltexts_parsed"]==1
    assert not transient_failure(r)


def test_conflicting_same_attempt_is_not_retried(brief):
    p=packet(brief,("FLT3",))
    h=failed_history(brief,p)
    duplicate=deepcopy(h["observations"][0])
    duplicate["status"]="empty"
    h["observations"].append(duplicate)
    p.metadata["g0_followup_history"]=[h]
    plan=build_plan(brief,p,verdict(),rework_loops=1,target_limit=1)
    paths=[path for t in plan["tasks"] for path in t["source_paths"] if path["source"]=="PubMed"]
    assert any(path["retry_decision"]["reason"]=="CONFLICTING_SOURCE_ATTEMPT_HISTORY" for path in paths)
