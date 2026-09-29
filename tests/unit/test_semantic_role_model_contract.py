"""Synthetic unit-only role/model separation; never injected into live evidence."""
import json

import pytest

from drugpilot.agents.evidence_semantics import (
    INTERPRETATION_CONTEXT_POLICY, KEY, EvidenceMeaning, EvidenceMeaningBatch,
    attach_interpretations, interpretation_prompt, permits, source_records,
)
from test_evidence_semantics import evidence, meaning


@pytest.mark.parametrize("disease,target,planned", [
    ("Breast cancer", "ERBB2", "Future isogenic mammary cell pair"),
    ("Non-small cell lung cancer", "EGFR", "Prospective mutant and corrected pair"),
    ("Chronic myeloid leukemia", "ABL1", "Future patient-derived validation model"),
])
def test_source_role_prompt_separates_future_model_without_case_specific_overrides(
    brief, disease, target, planned,
):
    b = brief.model_copy(update={"disease":disease,
        "constraints":brief.constraints.model_copy(update={"target_cell_type":planned})})
    prompt = interpretation_prompt(b, target, [])
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["request"]["target"] == target
    assert payload["request"]["disease"] == disease
    assert payload["request"]["planned_validation_model"] == planned
    assert "model_scope" not in payload["request"]
    assert payload["request"]["interpretation_context_policy"] == INTERPRETATION_CONTEXT_POLICY
    assert "not the proposed future validation experiment" in prompt
    assert "Association is not therapeutic direction or causal proof" in prompt
    assert "Exact-model applicability and experimental confirmation remain separate checks" in prompt


def annotate(brief, e, m):
    batch = EvidenceMeaningBatch(records=[m])
    return attach_interpretations(brief,"FLT3",[e],source_records([e]),batch,batch)[0]


def test_measured_patient_cohort_association_is_not_therapeutic_intervention(brief):
    e = evidence("FLT3 expression was associated with shorter survival in acute myeloid leukemia patients.")
    m = meaning(e,study_context="association_primary",intervention="none",outcome="unfavorable",
        model_quote="acute myeloid leukemia patients",intervention_quote="",
        result_quote="associated with shorter survival")
    r = annotate(brief,e,m)
    assert permits(r,brief,"FLT3","association")
    assert not permits(r,brief,"FLT3","direction")
    assert r.metadata[KEY]["planned_model_equivalence_verified"] is False
    assert r.excerpt_hash == e.excerpt_hash and r.metadata["excerpt"] == e.metadata["excerpt"]


@pytest.mark.parametrize("disease_context", ["different_model","background_only","unknown"])
def test_other_disease_or_ambiguous_model_still_cannot_support_claim(brief,disease_context):
    e = evidence()
    m = meaning(e,disease_context=disease_context)
    r = annotate(brief,e,m)
    assert not permits(r,brief,"FLT3","direction")
    assert not permits(r,brief,"FLT3","association")


@pytest.mark.parametrize("target_relation", ["indirect","absent","unknown"])
def test_direct_intervention_requirement_not_removed_for_direction(brief,target_relation):
    e = evidence()
    r = annotate(brief,e,meaning(e,target_relation=target_relation))
    assert not permits(r,brief,"FLT3","direction")


def test_schema_descriptions_distinguish_association_and_direction():
    properties = EvidenceMeaning.model_json_schema()["properties"]
    assert "association_primary" in properties["target_relation"]["description"]
    assert "NOT exact equivalence" in properties["disease_context"]["description"]
    assert properties["target_relation"]["enum"] == ["direct","indirect","absent","unknown"]
    assert properties["disease_context"]["enum"] == ["actual_model","background_only","different_model","unknown"]
