"""Synthetic component contracts only; these records are never live evidence."""
from hashlib import sha256

import pytest

from drugpilot.agents.evidence_semantics import (
    EvidenceMeaning, EvidenceMeaningBatch, attach_interpretations, record_key,
    source_binding, source_records,
)
from drugpilot.agents.ma1 import (
    Critic, MA1Agent, Proposer, TargetSeed, _evidence_linkage_entries,
    _expand_directional_hypotheses, _primary_direction_excerpt,
)
from drugpilot.domain.models import EvidenceRef, HypothesisPacket, HypothesisTrack, ProjectConstraints, ResearchBrief
from drugpilot.config import Settings
from drugpilot.tools.public_biology import (
    _DIRECTION_EXCERPT_PATTERNS, _direction_aware_query_terms,
    _direct_target_intervention_pattern, _target_intervention_precision_query_phrases,
)


def brief(direction):
    return ResearchBrief(
        disease="Acute myeloid leukemia", research_goal="Assess source-supported FLT3 biology.",
        constraints=ProjectConstraints(intervention_direction=direction),
    )


def record(text, direction, suffix="1"):
    return EvidenceRef(
        evidence_id="synthetic-" + suffix, source="PubMed", external_id="synthetic-paper-" + suffix,
        title="Synthetic component record", supports="support", excerpt_hash=sha256(text.encode()).hexdigest(),
        metadata={"excerpt": text, "content_hash": sha256(text.encode()).hexdigest(),
            "content_availability": "abstract_excerpt", "full_text_stored": False,
            "source_location": "abstract:RESULTS:chars:0-300",
            "excerpt_selection_policy": "same-sentence-target-disease-direction-aware-mechanistic-ranking-v2",
            "query_strategy": "target-disease-direction-mechanistic-v2",
            "query_intervention_direction": direction, "provenance_class": "external_live",
            "publication_types": ["Journal Article"]},
    )


def build(b, records):
    class Retriever:
        def retrieve(self, *args):
            return records
    return Proposer(Retriever())._build_hypothesis(
        b, TargetSeed("FLT3", "FLT3", HypothesisTrack.LITERATURE, 1.0), 1)


@pytest.mark.parametrize("intake", ["modulate", "unspecified"])
def test_broad_intake_branches_only_source_supported_directions(intake):
    b = brief(intake)
    evidence = [record("FLT3 inhibition reduced viability in acute myeloid leukemia cells.", intake, "inh"),
                record("Activation of FLT3 increased apoptosis in acute myeloid leukemia cells.", intake, "act")]
    before = {e.evidence_id: source_binding(e) for e in evidence}
    hypotheses = _expand_directional_hypotheses(b, build(b, evidence))
    assert {h.intervention_direction for h in hypotheses} == {"inhibit", "activate"}
    assert len({h.hypothesis_id for h in hypotheses}) == 2
    direction_claims = []
    for h in hypotheses:
        expected = "synthetic-inh" if h.intervention_direction == "inhibit" else "synthetic-act"
        units = h.target.metadata["claim_evidence_assignment"]["claim_units"]
        direct = [u for u in units if u["role"] == "direct_target_direction_primary"]
        assert [u["evidence_id"] for u in direct] == [expected]
        direction_claims.extend(u["claim_id"] for u in direct)
        assert h.target.metadata["direction_resolution"]["additional_model_calls"] == 0
        assert {e.evidence_id: source_binding(e) for e in h.evidence} == before
        assert all(e.metadata["query_intervention_direction"] == intake for e in h.evidence)
    assert len(set(direction_claims)) == 2


@pytest.mark.parametrize("direction,text", [
    ("inhibit", "Knockdown of FLT3 reduced viability in acute myeloid leukemia cells."),
    ("inhibit", "FLT3 silencing reduced proliferation in acute myeloid leukemia cells."),
    ("activate", "Overexpression of FLT3 increased apoptosis in acute myeloid leukemia cells."),
    ("activate", "A FLT3 agonist increased apoptosis in acute myeloid leukemia cells."),
    ("modulate", "Modulation of FLT3 reduced viability in acute myeloid leukemia cells."),
])
def test_direct_experimental_phrases_remain_admissible(direction, text):
    e = record(text, direction)
    assert _primary_direction_excerpt(brief(direction), "FLT3", e, intervention_direction=direction) == text
    assert _direct_target_intervention_pattern("FLT3", direction).search(text)


@pytest.mark.parametrize("text", [
    "A FLT3 inhibitor reduced STAT3 phosphorylation in acute myeloid leukemia cells.",
    "Suppression of FLT3 signaling reduced viability in acute myeloid leukemia cells.",
    "FLT3 was associated with proliferation in acute myeloid leukemia patients.",
    "FLT3 inhibition increased proliferation in acute myeloid leukemia cells.",
    "FLT3 inhibition reduced viability in lung cancer cells.",
])
def test_broad_input_does_not_promote_indirect_or_wrong_context(text):
    b = brief("unspecified")
    hypotheses = _expand_directional_hypotheses(b, build(b, [record(text, "unspecified")]))
    assert len(hypotheses) == 1
    assert hypotheses[0].target.metadata["direction_resolution"]["status"] == "direction_unresolved"
    critique = Critic().review(b, hypotheses[0])
    assert "INTERVENTION_DIRECTION_UNRESOLVED" in critique.reason_codes
    assert not critique.accepted


def test_semantic_agreement_stays_bound_to_original_broad_request():
    b = brief("unspecified")
    text = "FLT3 inhibition reduced viability in acute myeloid leukemia cells."
    e = record(text, "unspecified")
    m = EvidenceMeaning(record_key=record_key(e), study_context="measured_primary", target_relation="direct",
        disease_context="actual_model", intervention="direct_requested", outcome="favorable",
        target_quote="FLT3", model_quote="acute myeloid leukemia cells", intervention_quote="FLT3 inhibition",
        result_quote="reduced viability", molecular_scope_label="")
    annotated = attach_interpretations(b, "FLT3", [e], source_records([e]),
        EvidenceMeaningBatch(records=[m]), EvidenceMeaningBatch(records=[m]))[0]
    assert _primary_direction_excerpt(b, "FLT3", annotated, intervention_direction="inhibit") == text
    assert _primary_direction_excerpt(b, "FLT3", annotated, intervention_direction="activate") is None
    changed = b.model_copy(update={"research_goal": "A different research question."})
    assert _primary_direction_excerpt(changed, "FLT3", annotated, intervention_direction="inhibit") is None


@pytest.mark.parametrize("intake", ["modulate", "unspecified"])
def test_broad_queries_include_both_actions_and_genetic_perturbation(intake):
    terms = _direction_aware_query_terms(intake)
    assert all(t in terms for t in ("inhibition", "activation", "knockdown", "silencing", "overexpression"))
    phrases = _target_intervention_precision_query_phrases("FLT3", intake)
    assert "FLT3 inhibition" in phrases and "FLT3 activation" in phrases
    assert "silencing of FLT3" in phrases and "overexpression of FLT3" in phrases
    assert _DIRECTION_EXCERPT_PATTERNS[intake].search("knockdown")


def test_linkage_trace_preserves_semantic_and_claim_admission_disagreement():
    b = brief("inhibit")
    e = record("A FLT3 inhibitor reduced STAT3 phosphorylation in acute myeloid leukemia cells.", "inhibit")
    h = build(b, [e])
    critique = Critic().review(b, h)
    trace = _evidence_linkage_entries(b, critique)
    assert len(trace) == 1
    assert trace[0]["source"] == "PubMed"
    assert trace[0]["hypothesis_id"] == h.hypothesis_id
    assert trace[0]["external_id"] == e.external_id
    assert "DIRECT_OR_PHENOTYPIC_ENDPOINT_MISSING" in trace[0]["direction_rejection_codes"]
    assert trace[0]["diagnostic_only"] is True
    assert "direct_target_direction_primary" not in trace[0]["linked_roles"]


def test_concrete_request_cannot_be_overridden_by_an_opposite_direction():
    e = record("Activation of FLT3 increased apoptosis in acute myeloid leukemia cells.", "inhibit")
    assert _primary_direction_excerpt(brief("inhibit"), "FLT3", e, intervention_direction="activate") is None


def test_directional_branch_preserves_detailed_original_falsification_protocol():
    b = brief("unspecified")
    original = (
        "Use donor-derived AML cultures with vehicle and matched non-targeting siRNA controls. "
        "Collect ATP viability at 24 and 72 hours with four technical replicates. "
        "Reject if the pre-registered 12% viability difference is not met and report the "
        "confidence interval; confirm rescue with an siRNA-resistant construct."
    )
    h = build(b, [record("FLT3 silencing reduced viability in acute myeloid leukemia cells.", "unspecified")])
    h.falsification_test = original
    branches = _expand_directional_hypotheses(b, h)
    assert len(branches) == 1
    assert branches[0].falsification_test.startswith(original + "\n")
    assert "Resolved intervention direction for this hypothesis: inhibit (FLT3)" in branches[0].falsification_test
    assert "30%" not in branches[0].falsification_test
    assert "two disease-relevant models" not in branches[0].falsification_test
    assert h.falsification_test == original


def test_packet_linkage_trace_is_bounded_and_counts_omitted_rows(tmp_path):
    class Retriever:
        def retrieve(self, request, seed, iteration):
            return [record(f"{seed.symbol} was associated with proliferation in acute myeloid leukemia patients.",
                           request.constraints.intervention_direction, seed.symbol + str(i)) for i in range(60)]
    packet = MA1Agent(
        settings=Settings(mode="mock", database_path=tmp_path / "test.sqlite3", mock_step_delay_ms=0),
        evidence_retriever=Retriever(),
    ).run(brief("unspecified"))
    trace = packet.metadata["evidence_linkage_trace"]
    assert trace["entry_count_total"] > 240
    assert trace["entry_count_recorded"] == len(trace["entries"]) == 240
    assert trace["entry_count_total"] == trace["entry_count_recorded"] + trace["entry_count_omitted"]
    assert trace["direction_rejection_counts"]["DIRECT_TARGET_INTERVENTION_MISSING"] > 240
    assert not packet.hypotheses


def test_planner_broad_input_requires_each_resolved_hypothesis_direction():
    from drugpilot.agents.planner import PlannerAgent
    b = brief("unspecified")
    branches = _expand_directional_hypotheses(b, build(b, [
        record("FLT3 inhibition reduced viability in acute myeloid leukemia cells.", "unspecified")]))
    packet = HypothesisPacket(brief_id=b.brief_id, hypotheses=branches)
    tasks = PlannerAgent()._tasks(packet=packet, target_symbols=["FLT3"], candidates_per_route=2,
        total_budget=10000, brief=b, context_note="")
    criteria = [text for task in tasks for text in task.acceptance_criteria]
    assert any("linked approved hypothesis direction" in text and "FLT3: inhibit" in text for text in criteria)
    assert not any("equals 'unspecified'" in text for text in criteria)
