"""Offline end-to-end PubMed lane contracts; no scientific run/database writes."""
from dataclasses import replace
from hashlib import sha256
from xml.sax.saxutils import escape

import pytest

from drugpilot.agents.evidence_semantics import source_review_text
from drugpilot.agents.ma1 import TargetSeed
from drugpilot.config import Settings
from drugpilot.domain.models import HypothesisTrack, RunMode
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.tools.public_biology import (
    PUBMED_SEARCH_URL, PUBMED_SUMMARY_URL, PUBMED_FETCH_URL,
    _publication_content_from_sections,
)
from test_public_biology import FakeResponse, _retriever
from test_qa_direction_resolution import brief


CELL_SURVIVAL = "FLT3 inhibition reduced cell survival in acute myeloid leukemia cells."
NEGATIVE = "FLT3 inhibition did not reduce cell survival in acute myeloid leukemia cells."
ADJACENT = "We established acute myeloid leukemia cell cultures. FLT3 knockdown reduced their viability."


class OriginalSourceClient:
    def __init__(self, texts=(CELL_SURVIVAL,), mutation=None):
        self.texts, self.mutation, self.calls = texts, mutation, []
        self.ids = [str(990001 + i) for i in range(len(texts))]
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url == PUBMED_SEARCH_URL:
            return FakeResponse({"esearchresult": {"count": str(len(self.ids)), "idlist": self.ids}})
        if url == PUBMED_SUMMARY_URL:
            ids = self.ids if self.mutation != "summary_id" else ["123456"]
            result = {"uids": ids}
            result.update({identifier: {"uid": identifier, "title": "Synthetic primary experiment",
                "pubtype": ["Journal Article"]} for identifier in ids})
            return FakeResponse({"result": result})
        if url == PUBMED_FETCH_URL:
            articles = "".join(
                "<PubmedArticle><MedlineCitation><PMID>" + ("123456" if self.mutation == "fetch_id" else identifier)
                + "</PMID><Article><Abstract><AbstractText Label='RESULTS'>" + escape(text)
                + "</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>"
                for identifier, text in zip(self.ids, self.texts))
            return FakeResponse(text="<invalid" if self.mutation == "xml_schema" else
                "<PubmedArticleSet>" + articles + "</PubmedArticleSet>")
        raise AssertionError(url)


def run_lane(client, *, enabled, cap=4):
    retriever = _retriever(client, enabled_sources=["PubMed"], max_records_per_source=cap,
        ma3_source_review_enabled=enabled)
    outcome = retriever._pubmed_primary_direction_supplement(
        brief("inhibit"), TargetSeed("FLT3", "FLT3", HypothesisTrack.LITERATURE, 1.0),
        query='"synthetic offline source review"', common_params={"tool": "DrugPilotUnit"},
        excluded_ids=set(), direction_model_context=None)
    return outcome


@pytest.mark.parametrize("text", [CELL_SURVIVAL, NEGATIVE, ADJACENT])
def test_original_source_survives_semantic_rejection_only_under_new_policy(text):
    legacy = run_lane(OriginalSourceClient([text]), enabled=False)
    assert legacy.records == ()
    client = OriginalSourceClient([text])
    outcome = run_lane(client, enabled=True)
    assert outcome.status == "completed_with_source_review_context"
    assert len(outcome.records) == 1
    source = outcome.records[0]
    assert source.supports == "context"
    assert source.metadata["direct_support_eligible"] is False
    assert source.metadata["verified_direct_primary"] is False
    assert source.metadata["scientific_approval"] is False
    assert source_review_text(source) == text
    assert source.metadata["content_hash"] == sha256(text.encode()).hexdigest()
    if text == ADJACENT:
        assert source.excerpt_hash is None and source.metadata["excerpt"] is None
    else:
        assert source.excerpt_hash == sha256(text.encode()).hexdigest()
        assert "non_direction_or_negative_context" in source.metadata["direct_support_rejection_codes"]
    assert outcome.trace["screening_entries"][0]["source_review_disposition"] == "retained_pending_ma3"
    assert outcome.trace["screening_entries"][0]["legacy_screening_status"] == "rejected"
    assert len(client.calls) == outcome.trace["http_requests_used"] == 3


def test_legacy_verified_source_is_not_labeled_approved_by_new_collection_policy():
    text = "FLT3 inhibition reduced viability in acute myeloid leukemia cells."
    old = run_lane(OriginalSourceClient([text]), enabled=False)
    assert old.records[0].metadata["verified_direct_primary"] is True
    new = run_lane(OriginalSourceClient([text]), enabled=True)
    assert new.records[0].metadata["verified_direct_primary"] is False
    assert new.records[0].metadata["direct_support_rejection_codes"] == []
    assert new.records[0].metadata["ma3_source_review_required"] is True


def test_source_cap_is_explicit_and_does_not_trigger_new_searches():
    client = OriginalSourceClient([CELL_SURVIVAL, NEGATIVE, ADJACENT])
    outcome = run_lane(client, enabled=True, cap=1)
    assert len(outcome.records) == 1
    assert outcome.trace["source_review_eligible_count"] == 3
    assert outcome.trace["source_review_retained_count"] == 1
    assert outcome.trace["source_review_omitted_count"] == 2
    assert outcome.trace["source_review_record_limit"] == 1
    assert outcome.trace["records_screened"] == 3
    assert len(client.calls) == 3
    assert [x["source_review_disposition"] for x in outcome.trace["screening_entries"]] == [
        "retained_pending_ma3", "omitted_record_limit", "omitted_record_limit"]


@pytest.mark.parametrize("mutation", ["summary_id", "fetch_id", "xml_schema"])
def test_source_identity_or_schema_failure_remains_fail_closed(mutation):
    outcome = run_lane(OriginalSourceClient(mutation=mutation), enabled=True)
    assert outcome.records == () and outcome.status == "error"


def test_bad_hash_in_parsed_content_does_not_become_source_only_evidence():
    content = _publication_content_from_sections([("RESULTS", CELL_SURVIVAL)], target_symbol="FLT3",
        disease="acute myeloid leukemia", intervention_direction="inhibit")
    retriever = _retriever(OriginalSourceClient(), ma3_source_review_enabled=True)
    invalid = replace(content, content_hash="0" * 64)
    assert retriever._source_only_direction_record(invalid, pmid="990001", title="Synthetic source",
        publication_types=["Journal Article"], query="unit", direction="inhibit",
        rejection_codes=["non_direction_or_negative_context"]) is None


def test_main_pubmed_path_preserves_unselected_abstract_only_for_new_policy():
    b = brief("inhibit")
    seed = TargetSeed("FLT3", "FLT3", HypothesisTrack.LITERATURE, 1.0)
    for enabled in (False, True):
        client = OriginalSourceClient([ADJACENT])
        retriever = _retriever(client, enabled_sources=["PubMed"], ma3_source_review_enabled=enabled)
        records = retriever._pubmed(b, seed)
        originals = [e for e in records if e.external_id == "990001"]
        assert originals
        assert any(source_review_text(e) == ADJACENT for e in originals) is enabled
        assert all(e.excerpt_hash is None for e in originals)
        assert all(e.metadata.get("verified_direct_primary") is not True for e in originals)


@pytest.mark.parametrize("enabled", [False, True])
def test_default_and_run_local_public_retrievers_receive_exact_feature_flag(tmp_path, enabled):
    orchestrator = DrugPilotOrchestrator(settings=Settings(_env_file=None, mode=RunMode.MOCK,
        database_path=tmp_path / "unused.sqlite3", public_evidence_enabled=True,
        ma1_provisional_admission_enabled=enabled))
    assert orchestrator.ma1.proposer.retriever.public.ma3_source_review_enabled is enabled
    assert orchestrator._ma1_for_brief(brief("inhibit")).proposer.retriever.public.ma3_source_review_enabled is enabled
