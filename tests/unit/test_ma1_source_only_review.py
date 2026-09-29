"""Source-only admission contracts; synthetic in-memory records, no live calls."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from drugpilot.agents.evidence_semantics import (
    ABSTRACT_CONTEXT_KEY, abstract_context, interpretation_text, source_review_text,
)
from drugpilot.agents.ma1 import (
    Critic, MA1Agent, Router, _expand_directional_hypotheses,
    _expand_molecular_hypotheses, _prepare_ma3_submission,
)
from drugpilot.config import Settings
from drugpilot.domain.models import EvidenceRef, RunMode
from drugpilot.orchestration.ma1_source_bundles import load_source_bundles, persist_source_bundle
from drugpilot.tools.public_biology import _publication_content_from_sections
from test_ma1_source_bundles import Harness
from test_qa_direction_resolution import brief, build, record


ABSTRACT = (
    "We established acute myeloid leukemia cell cultures. "
    "FLT3 knockdown reduced their viability by 42% compared with non-targeting siRNA controls."
)


def abstract_source(text=ABSTRACT):
    return EvidenceRef(source="PubMed", external_id="synthetic-source-only-unit",
        title="Synthetic source-only contract", supports="context",
        metadata={"provenance_class": "external_live", "excerpt": None,
            "content_hash_scope": "normalized_full_abstract",
            "content_hash": sha256(text.encode()).hexdigest(),
            ABSTRACT_CONTEXT_KEY: abstract_context(text)})


def test_complete_hash_bound_abstract_survives_legacy_sentence_selector_failure():
    content = _publication_content_from_sections([("RESULTS", ABSTRACT)], target_symbol="FLT3",
        disease="acute myeloid leukemia", intervention_direction="unspecified")
    assert content.excerpt is None and content.excerpt_selection_status == "related_excerpt_not_found"
    e = abstract_source()
    e.metadata[ABSTRACT_CONTEXT_KEY] = content.interpretation_context
    before = e.model_dump_json()
    assert interpretation_text(e) is None  # Historical primary selector unchanged.
    assert source_review_text(e) == ABSTRACT
    assert e.model_dump_json() == before and e.excerpt_hash is None and e.supports == "context"


@pytest.mark.parametrize("change", [
    "content_hash", "abstract_hash", "text", "count", "count_bool", "scope", "policy",
    "authority", "provenance", "fixture", "mock", "source", "external_id", "dangling_excerpt_hash",
    "empty_excerpt", "corrupt_excerpt", "oversized", "no_context",
])
def test_no_abstract_fallback_for_invalid_or_unavailable_source(change):
    e = abstract_source()
    context = e.metadata[ABSTRACT_CONTEXT_KEY]
    if change == "content_hash": e.metadata["content_hash"] = "0" * 64
    elif change == "abstract_hash": context["sha256"] = "0" * 64
    elif change == "text": context["text"] += " Changed."
    elif change == "count": context["char_count"] += 1
    elif change == "count_bool": context["char_count"] = True
    elif change == "scope": e.metadata["content_hash_scope"] = "title"
    elif change == "policy": context["policy"] = "unknown"
    elif change == "authority": context["authority"] = "approved"
    elif change == "provenance": e.metadata["provenance_class"] = "fixture"
    elif change == "fixture": e.metadata["fixture"] = True
    elif change == "mock": e.metadata["mock"] = True
    elif change == "source": e.source = "not-allowed"
    elif change == "external_id": e.external_id = " "
    elif change == "dangling_excerpt_hash": e.excerpt_hash = "0" * 64
    elif change == "empty_excerpt": e.metadata["excerpt"] = ""
    elif change == "corrupt_excerpt":
        e.metadata["excerpt"] = ABSTRACT
        e.excerpt_hash = "0" * 64
    elif change == "oversized": e = abstract_source("x" * 6001)
    elif change == "no_context": e.metadata.pop(ABSTRACT_CONTEXT_KEY)
    assert source_review_text(e) is None


def test_legacy_good_excerpt_contract_is_unchanged():
    e = record("FLT3 inhibition reduced viability in acute myeloid leukemia cells.", "inhibit")
    assert source_review_text(e) == interpretation_text(e) == e.metadata["excerpt"]


def test_source_only_hypothesis_reaches_ma3_without_fake_primary_links():
    b = brief("inhibit")
    h = build(b, [abstract_source()])
    before = deepcopy(h.evidence)
    original = Critic().review(b, h)
    assert not original.accepted and not any(c.evidence_ids for c in h.claims)
    p = _prepare_ma3_submission(b, original)
    assert p.hypothesis.target.metadata["ma1_admission"]["status"] == "provisional_pending_ma3"
    assert p.hypothesis.evidence == before
    assert not any(c.evidence_ids for c in p.hypothesis.claims)
    assert p.hypothesis.target.metadata["ma1_admission"]["complete_abstract_without_selected_excerpt_ids"] == [before[0].evidence_id]


def test_oversized_unavailable_context_does_not_poison_an_independent_valid_source():
    b = brief("inhibit")
    h = build(b, [abstract_source(), abstract_source("x" * 6001)])
    p = _prepare_ma3_submission(b, Critic().review(b, h))
    assert p.hypothesis.target.metadata["ma1_admission"]["status"] == "provisional_pending_ma3"
    assert p.hypothesis.target.metadata["ma1_admission"]["source_bound_literature_count"] == 1


@pytest.mark.asyncio
async def test_source_only_bundle_roundtrip_remains_bound_and_unchanged():
    b, e, harness = brief("inhibit"), abstract_source(), Harness()
    before = e.model_dump_json()
    await persist_source_bundle(harness, "unit", b, SimpleNamespace(symbol="FLT3"), 1, [e])
    sources, lineage = await load_source_bundles(harness, "unit", b)
    assert sources["flt3"][0].model_dump_json() == before
    assert lineage[0]["source_record_count"] == 1 and lineage[0]["new_model_calls"] == 0


@pytest.mark.parametrize("intake", ["modulate", "unspecified"])
@pytest.mark.parametrize("allowed", [None, ["inhibit"], ["activate"], ["modulate"], ["inhibit", "activate"]])
def test_new_policy_proposes_every_allowed_direction_without_claiming_resolution(intake, allowed):
    b = brief(intake)
    b.constraints.allowed_hypothesis_directions = allowed
    sources = [record("FLT3 inhibition reduced viability in acute myeloid leukemia cells.", intake, "inh"),
        record("Activation of FLT3 decreased cell survival in acute myeloid leukemia cells.", intake, "act")]
    h = build(b, sources)
    before = h.model_dump_json()
    branches = _expand_directional_hypotheses(b, h, provisional_admission=True)
    assert {x.intervention_direction for x in branches} == set(allowed or ["inhibit", "activate", "modulate"])
    assert len({x.hypothesis_id for x in branches}) == len(branches) <= 3
    for branch in branches:
        resolution = branch.target.metadata["direction_resolution"]
        assert resolution["status"] == "direction_proposed_for_ma3" and resolution["resolved_direction"] is None
        assert resolution["scientific_approval"] is False and resolution["evidence_ids"] == []
        assert resolution["potential_ma3_review_calls"] == len(branches)
        assert branch.evidence == h.evidence  # Including opposite-direction raw context.
        prepared = _prepare_ma3_submission(b, Critic().review(b, branch))
        assert prepared.hypothesis.target.metadata["ma1_admission"]["status"] == "provisional_pending_ma3"
    assert h.model_dump_json() == before


def test_explicit_request_and_legacy_broad_behavior_are_unchanged():
    b = brief("inhibit")
    h = build(b, [abstract_source()])
    assert _expand_directional_hypotheses(b, h, provisional_admission=True) == [h]
    b = brief("unspecified")
    h = build(b, [record("FLT3 inhibition reduced viability in acute myeloid leukemia cells.", "unspecified", "inh"),
        record("Activation of FLT3 decreased cell survival in acute myeloid leukemia cells.", "unspecified", "act")])
    legacy = _expand_directional_hypotheses(b, h)
    assert [x.intervention_direction for x in legacy] == ["inhibit"]
    assert legacy[0].target.metadata["direction_resolution"]["status"] == "resolved_direct_primary"


@pytest.mark.parametrize("hard_failure", ["empty_sources", "forbidden"])
def test_question_enumeration_does_not_relax_hard_constraints(hard_failure):
    b = brief("unspecified")
    if hard_failure == "forbidden": b.constraints.forbidden_targets = ["FLT3"]
    h = build(b, [] if hard_failure == "empty_sources" else [abstract_source()])
    branches = _expand_directional_hypotheses(b, h, provisional_admission=True)
    prepared = [_prepare_ma3_submission(b, Critic().review(b, x)) for x in branches]
    selected, rejected = Router().route(prepared, 1, provisional_admission=True)
    assert not selected and len(rejected) == 3


def test_abstract_only_variant_remains_raw_without_inventing_scope_anchor():
    b = brief("inhibit")
    e = abstract_source("We established acute myeloid leukemia cultures. FLT3-ITD was present in these cells.")
    h = build(b, [e])
    branches = _expand_molecular_hypotheses(b, h)
    assert len(branches) == 1
    assert "molecular_hypothesis_scope" not in branches[0].target.metadata
    p = _prepare_ma3_submission(b, Critic().review(b, branches[0]))
    assert p.hypothesis.evidence[0].metadata[ABSTRACT_CONTEXT_KEY]["text"] == e.metadata[ABSTRACT_CONTEXT_KEY]["text"]
    assert "not a claim that every" in p.hypothesis.target.metadata["ma1_admission"]["molecular_scope_enumeration"]


def test_agent_wires_new_question_policy_and_retrieves_each_target_only_once():
    class Retriever:
        def __init__(self): self.calls = []
        def retrieve(self, b, seed, iteration):
            self.calls.append(seed.symbol)
            return [abstract_source(ABSTRACT.replace("FLT3", seed.symbol))]
    retriever = Retriever()
    packet = MA1Agent(settings=Settings(_env_file=None, mode=RunMode.MOCK,
        ma1_provisional_admission_enabled=True), evidence_retriever=retriever).run(brief("unspecified"))
    assert not packet.hypotheses and packet.provisional_hypotheses
    for target in {x.target.symbol for x in packet.provisional_hypotheses}:
        assert {x.intervention_direction for x in packet.provisional_hypotheses if x.target.symbol == target} == {"inhibit", "activate", "modulate"}
    assert len(retriever.calls) == len(set(retriever.calls))
    assert packet.metadata["broad_direction_max_questions_per_target_scope"] == 3
