"""Synthetic search-contract tests only; no live candidate/source injection."""
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma1 import TargetSeed
from drugpilot.agents.molecular_scope import (
    SEARCH_KEY, hypothesis_scope, fusion_search_bindings, validated_fusion_search_terms,
)
from drugpilot.domain.models import HypothesisTrack
from drugpilot.orchestration.g0_followup import build_plan, FollowupRetriever
from drugpilot.tools.public_biology import PublicBiologyRetriever
from test_evidence_semantics import evidence
from test_g0_followup import packet, verdict


def scoped_packet(brief, text="ETV6::FLT3 fusion was observed in acute myeloid leukemia cells."):
    e = evidence(text)
    scope = hypothesis_scope(brief, "FLT3", [e], "ETV6::FLT3")
    p = packet(brief, ("FLT3",))
    entry = p.metadata["rejected_screening_trace"]["entries"][0]
    entry["molecular_hypothesis_scope"] = scope
    entry["unique_external_records"] = [{"sources":[e.source], "external_ids":[e.external_id],
        "provenance_classes":["external_live"], "evidence_units":[{
            "evidence_id":e.evidence_id, "bounded_excerpt":text, "excerpt_hash":e.excerpt_hash}]}]
    return p, entry


def test_plan_uses_exact_observed_fusion_and_keeps_canonical_target(brief):
    p, entry = scoped_packet(brief)
    before = p.model_dump_json()
    plan = build_plan(brief, p, verdict(), rework_loops=0, target_limit=1)
    assert p.model_dump_json() == before
    for task in plan["tasks"]:
        assert task["target"] == "FLT3"
        assert validated_fusion_search_terms(task[SEARCH_KEY], "FLT3") == ("ETV6::FLT3",)
    plain = build_plan(brief, packet(brief,("FLT3",)), verdict(), rework_loops=0, target_limit=1)
    assert {t["task_key"] for t in plan["tasks"]}.isdisjoint(t["task_key"] for t in plain["tasks"])


@pytest.mark.parametrize("mutation", [
    lambda e: e["molecular_hypothesis_scope"].update(label="BCR::ABL1"),
    lambda e: e["molecular_hypothesis_scope"]["anchors"][0].update(start=0, end=4),
    lambda e: e["molecular_hypothesis_scope"]["anchors"][0].update(quote="BCR::ABL1"),
    lambda e: e["molecular_hypothesis_scope"]["anchors"][0].update(source="Invented"),
    lambda e: e["molecular_hypothesis_scope"]["anchors"][0].update(external_id="other-paper"),
    lambda e: e["unique_external_records"][0].update(provenance_classes=["fixture"]),
    lambda e: e["unique_external_records"][0]["evidence_units"][0].update(bounded_excerpt="ETV6::FLT3"),
    lambda e: e["unique_external_records"][0]["evidence_units"][0].update(excerpt_hash="0"*64),
])
def test_tampered_truncated_or_wrong_scope_never_supplies_query_terms(brief, mutation):
    p, entry = scoped_packet(brief)
    mutation(entry)
    assert fusion_search_bindings(brief, entry, "FLT3") == []
    plan = build_plan(brief, p, verdict(), rework_loops=0, target_limit=1)
    assert all(SEARCH_KEY not in task for task in plan["tasks"])


def test_negative_fusion_mention_can_guide_search_but_has_no_support_authority(brief):
    _, entry = scoped_packet(brief, "ETV6-FLT3 fusion was not detected in acute myeloid leukemia cells.")
    bound = fusion_search_bindings(brief, entry, "FLT3")
    assert validated_fusion_search_terms(bound, "FLT3") == ("ETV6::FLT3", "ETV6-FLT3")
    assert bound[0]["authority"] == "search_scope_only_not_support_or_identity"


def test_source_queries_and_execution_trace_retain_observed_label(monkeypatch, brief):
    p, entry = scoped_packet(brief)
    bindings = fusion_search_bindings(brief, entry, "FLT3")
    seed = TargetSeed("FLT3", "FLT3", HypothesisTrack.LITERATURE, 1,
        discovery_metadata={SEARCH_KEY:bindings})
    public = PublicBiologyRetriever(enabled_sources=("PubMed",))
    queries = [public._pubmed_primary_direction_query(brief,seed),
        public._pubmed_primary_association_query(brief,seed),public._europe_pmc_query(brief,seed)]
    assert all("ETV6::FLT3" in q for q in queries)
    assert all('"FLT3"[Title/Abstract]' not in q for q in queries)
    received = []
    def direction(*args, **kwargs):
        received.append(kwargs["query"])
        return SimpleNamespace(records=(), status="empty", stop_reason="no_records", trace={})
    monkeypatch.setattr(public, "_pubmed_primary_direction_supplement", direction)
    records, rows = public.retrieve_g0_followup(brief,seed,roles=["direction"])
    assert not records and "ETV6::FLT3" in received[0]
    assert rows[0]["fusion_search_bindings"] == bindings
    assert rows[0]["search_scope_authority"] == "search_only_not_support_or_identity"


def test_wrapper_passes_bound_source_not_reviewer_prose(brief):
    p, _ = scoped_packet(brief)
    plan = build_plan(brief,p,verdict(),rework_loops=0,target_limit=1)
    seeds = []
    def retrieve(b,s,**kwargs):
        seeds.append(s)
        return (), []
    wrapper = FollowupRetriever(SimpleNamespace(public=SimpleNamespace(retrieve_g0_followup=retrieve)),plan)
    seed = TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1)
    wrapper.retrieve(brief,seed,1)
    assert SEARCH_KEY not in seed.discovery_metadata
    assert validated_fusion_search_terms(seeds[0].discovery_metadata[SEARCH_KEY],"FLT3") == ("ETV6::FLT3",)


def test_present_invalid_binding_is_rejected_before_any_http(brief):
    seed = TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1,
        discovery_metadata={SEARCH_KEY:[{"label":"INVENTED::FLT3"}]})
    with pytest.raises(ValueError,match="FUSION_SEARCH_BINDING_INVALID"):
        PublicBiologyRetriever().retrieve_g0_followup(brief,seed,roles=["direction"])


def test_incompatible_user_scope_cannot_be_overridden(brief):
    _, entry = scoped_packet(brief)
    other = brief.model_copy(update={"research_goal":"Study FLT3-ITD only"})
    assert fusion_search_bindings(other,entry,"FLT3") == []
