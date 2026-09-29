"""Synthetic isolated variant-query tests, never a live run fixture."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma1 import TargetSeed
from drugpilot.agents.molecular_scope import hypothesis_scope
from drugpilot.agents.variant_search_scope import (
    KEY,
    from_screening_entry,
    query_terms,
    validate,
)
from drugpilot.domain.models import HypothesisTrack
from drugpilot.orchestration.g0_followup import FollowupRetriever, build_plan
from drugpilot.tools.public_biology import PublicBiologyRetriever, PUBMED_SEARCH_URL, PUBMED_SUMMARY_URL, PUBMED_FETCH_URL
from test_evidence_semantics import evidence
from test_g0_followup import packet, verdict
from test_public_biology import FakeResponse


def scoped(brief,target="FLT3",text="FLT3-ITD was observed in acute myeloid leukemia cells.",label="FLT3:ITD"):
    e=evidence(text)
    p=packet(brief,(target,))
    entry=p.metadata["rejected_screening_trace"]["entries"][0]
    entry["molecular_hypothesis_scope"]=hypothesis_scope(brief,target,[e],label)
    entry["unique_external_records"]=[{"sources":[e.source],"external_ids":[e.external_id],
        "provenance_classes":["external_live"],"evidence_units":[{"evidence_id":e.evidence_id,
            "excerpt_hash":e.excerpt_hash,"bounded_excerpt":text}]}]
    return p,entry


@pytest.mark.parametrize("target,text,label,term",[
    ("FLT3","FLT3-ITD was observed.","FLT3:ITD","FLT3-ITD"),
    ("EGFR","EGFR L858R was observed.","EGFR:L858R","EGFR L858R"),
    ("MET","MET exon 14 skipping was observed.","MET:EXON 14 SKIPPING","MET exon 14 skipping"),
])
def test_source_observed_variant_guides_query_not_identity_or_claim(brief,target,text,label,term):
    p,entry=scoped(brief,target,text,label)
    before=p.model_dump_json()
    bindings=from_screening_entry(brief,entry,target)
    assert validate(bindings,target)==(term,)
    assert bindings[0]["authority"]=="search_scope_only_not_support_or_identity"
    plan=build_plan(brief,p,verdict(),rework_loops=0,target_limit=1)
    assert all(t[KEY]==bindings for t in plan["tasks"])
    assert p.model_dump_json()==before
    seed=TargetSeed(target,target,HypothesisTrack.LITERATURE,1,discovery_metadata={KEY:bindings})
    assert term in PublicBiologyRetriever._pubmed_primary_direction_query(brief,seed)


def test_validated_itd_quote_expands_only_search_vocabulary(brief):
    _, entry = scoped(
        brief,
        text=(
            "FLT3 internal tandem duplication was observed in acute myeloid "
            "leukemia cells."
        ),
    )
    bindings = from_screening_entry(brief, entry, "FLT3")

    assert validate(bindings, "FLT3") == ("FLT3 internal tandem duplication",)
    assert query_terms(bindings, "FLT3") == (
        "FLT3 internal tandem duplication",
        "FLT3-ITD",
        "FLT3/ITD",
        "FLT3 ITD",
        "internal tandem duplication of FLT3",
    )
    assert bindings[0]["authority"] == "search_scope_only_not_support_or_identity"


def test_g0_direction_query_includes_bounded_itd_aliases(brief, monkeypatch):
    _, entry = scoped(
        brief,
        text=(
            "FLT3 internal tandem duplication was observed in acute myeloid "
            "leukemia cells."
        ),
    )
    bindings = from_screening_entry(brief, entry, "FLT3")
    public = PublicBiologyRetriever(enabled_sources=("PubMed",))
    calls = []

    def direction(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(records=(), status="empty", stop_reason="no_records", trace={})

    monkeypatch.setattr(public, "_pubmed_primary_direction_supplement", direction)
    plan = build_plan(brief, packet(brief, ("FLT3",)), verdict(), rework_loops=0, target_limit=1)
    plan["tasks"] = [task for task in plan["tasks"] if task["role"] == "direction"]
    for task in plan["tasks"]:
        task[KEY] = bindings
    FollowupRetriever(SimpleNamespace(public=public), plan).retrieve(
        brief,
        TargetSeed("FLT3", "FLT3", HypothesisTrack.LITERATURE, 1),
        1,
    )

    query = calls[0]["query"]
    assert '"FLT3 internal tandem duplication"[Title/Abstract]' in query
    assert '"FLT3-ITD"[Title/Abstract]' in query
    assert len(query) < 2000


@pytest.mark.parametrize("mutate",[
    lambda e:e["molecular_hypothesis_scope"].update(label="FLT3:D835Y"),
    lambda e:e["molecular_hypothesis_scope"]["anchors"][0].update(quote="EGFR L858R"),
    lambda e:e["molecular_hypothesis_scope"]["anchors"][0].update(start=99),
    lambda e:e["unique_external_records"][0].update(provenance_classes=["fixture"]),
    lambda e:e["unique_external_records"][0]["evidence_units"][0].update(excerpt_hash="0"*64),
    lambda e:e["unique_external_records"][0]["evidence_units"][0].update(bounded_excerpt="Truncated"),
])
def test_invalid_anchors_cannot_change_query(brief,mutate):
    p,e=scoped(brief)
    mutate(e)
    assert from_screening_entry(brief,e,"FLT3")==[]
    assert all(KEY not in t for t in build_plan(brief,p,verdict(),rework_loops=0,target_limit=1)["tasks"])


def test_wrapper_and_lane_require_source_bound_variant(brief,monkeypatch):
    p,e=scoped(brief)
    bindings=from_screening_entry(brief,e,"FLT3")
    public=PublicBiologyRetriever(enabled_sources=("PubMed",))
    calls=[]
    def direction(*args,**kw):
        calls.append(kw)
        return SimpleNamespace(records=(),status="empty",stop_reason="no_records",trace={})
    monkeypatch.setattr(public,"_pubmed_primary_direction_supplement",direction)
    wrapper=FollowupRetriever(SimpleNamespace(public=public),build_plan(brief,p,verdict(),rework_loops=0,target_limit=1))
    # Limit test to direction; no real transport for other role.
    wrapper.plan["tasks"]=[t for t in wrapper.plan["tasks"] if t["role"]=="direction"]
    seed=TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1)
    wrapper.retrieve(brief,seed,1)
    assert calls[0]["required_molecular_scopes"]==("FLT3:ITD",)
    assert "FLT3-ITD" in calls[0]["query"] and KEY not in seed.discovery_metadata
    assert len(calls[0]["query"])<1800
    assert '"inhibition"[Title/Abstract]' in calls[0]["query"]
    assert 'NOT (Review[Publication Type]' in calls[0]["query"]
    assert wrapper.observations[0]["variant_search_bindings"]==bindings


class Client:
    def __init__(self): self.calls=[]
    def get(self,url,**kw):
        self.calls.append(url)
        if url==PUBMED_SEARCH_URL:
            return FakeResponse({"esearchresult":{"count":"2","idlist":["1","2"]}})
        if url==PUBMED_SUMMARY_URL:
            return FakeResponse({"result":{"uids":["1","2"],**{i:{"uid":i,"title":"Synthetic original research","pubtype":["Journal Article"]} for i in ["1","2"]}}})
        if url==PUBMED_FETCH_URL:
            texts=["The inhibitor bound FLT3, blocked cell-cycle progression and induced apoptosis, reducing proliferation in FLT3-ITD acute myeloid leukemia cells.",
                "FLT3-ITD inhibition significantly reduced cell proliferation in acute myeloid leukemia cells."]
            return FakeResponse(text="<PubmedArticleSet>"+"".join(
                f'<PubmedArticle><MedlineCitation><PMID>{i}</PMID><Article><Abstract><AbstractText Label="RESULTS">{t}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>'
                for i,t in enumerate(texts,1))+"</PubmedArticleSet>")
        raise AssertionError(url)


def test_scope_lane_does_not_stop_at_canonical_action_in_mutant_cells(brief):
    _,entry=scoped(brief)
    bindings=from_screening_entry(brief,entry,"FLT3")
    seed=TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1,discovery_metadata={KEY:bindings})
    client=Client()
    public=PublicBiologyRetriever(client=client,enabled_sources=("PubMed",))
    result=public._pubmed_primary_direction_supplement(brief,seed,query="synthetic unit test",common_params={},
        excluded_ids=set(),direction_model_context=None,required_molecular_scopes=("FLT3:ITD",))
    assert len(client.calls)==3
    assert [r.external_id for r in result.records]==["2"]
    assert "requested_molecular_scope_direct_intervention_missing" in result.trace["screening_entries"][0]["rejection_codes"]
    assert result.trace["required_molecular_scopes"]==["FLT3:ITD"]
    assert result.trace["search_scope_is_not_construct_identity"] is True


def test_unbound_requirement_rejected_before_http(brief):
    client=Client()
    public=PublicBiologyRetriever(client=client)
    with pytest.raises(ValueError,match="NOT_SOURCE_BOUND"):
        public._pubmed_primary_direction_supplement(brief,TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1),
            query="not sent",common_params={},excluded_ids=set(),direction_model_context=None,
            required_molecular_scopes=("FLT3:ITD",))
    assert client.calls==[]


@pytest.mark.parametrize("target,text,expected",[
    ("EGFR","EGFR-L858R inhibition reduced proliferation.",False),
    ("EGFR","EGFR L858R inhibition reduced proliferation.",False),
    ("FLT3","FLT3-ITD inhibition reduced proliferation.",False),
    ("EGFR","KRAS-G12D inhibition reduced proliferation.",True),
    ("EGFR","L858R inhibition reduced proliferation.",True),
    ("FLT3","FLT3-ITD/BCL2 inhibition reduced proliferation.",True),
    ("FLT3","FLT3-ITD inhibition and BCL2 inhibition reduced proliferation.",True),
])
def test_variant_subject_ownership_not_canonical_or_foreign_inference(target,text,expected):
    from drugpilot.tools.public_biology import _direct_support_rejection_codes
    assert ("intervention_subject_not_target" in _direct_support_rejection_codes(text,target)) is expected
