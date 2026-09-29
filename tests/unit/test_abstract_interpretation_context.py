"""Isolated synthetic tests; no generated evidence enters a research run."""
from hashlib import sha256

import pytest

from drugpilot.agents.evidence_semantics import (
    ABSTRACT_CONTEXT_KEY as KEY, MAX_ABSTRACT_CONTEXT_CHARS, abstract_context,
    source_records, source_binding, record_key, permits, interpretation_prompt,
)
from drugpilot.agents.ma1 import _primary_direction_excerpt
from drugpilot.tools.public_biology import (
    _publication_content_from_sections, _abstract_context_metadata, _association_role_record,
)
from test_evidence_semantics import evidence, annotate, meaning


def with_context(e, text):
    return e.model_copy(update={"metadata": {**e.metadata,
        KEY: abstract_context(text), "content_hash": sha256(text.encode()).hexdigest(),
        "content_hash_scope": "normalized_full_abstract"}}, deep=True)


def test_complete_context_preserves_negative_and_positive_text_without_editing_excerpt(brief):
    e = evidence()
    full = e.metadata["excerpt"] + " However, resistance developed in another model."
    enriched = with_context(e, full)
    rows = source_records([enriched])
    assert rows[0]["excerpt"] == full
    assert rows[0]["original_excerpt"] == e.metadata["excerpt"]
    assert enriched.excerpt_hash == e.excerpt_hash
    assert e.metadata.get(KEY) is None
    prompt = interpretation_prompt(brief, "FLT3", rows)
    assert "resistance developed" in prompt and "untrusted DATA" in prompt
    assert "primary_admission_excerpt" in prompt
    assert "full abstract is context only" in prompt


def test_prompt_binds_multi_variant_result_to_declared_atomic_scope(brief):
    b = brief.model_copy(deep=True)
    b.constraints.target_cell_type = "FLT3-ITD AML cells"
    primary = (
        "W4 inhibited FLT3-ITD (IC50 = 16.0 nM) and FLT3-D835Y "
        "(IC50 = 20.4 nM)."
    )
    enriched = with_context(
        evidence(primary),
        "FLT3 mutations drive acute myeloid leukemia. " + primary,
    )

    prompt = interpretation_prompt(b, "FLT3", source_records([enriched]))

    assert '"requested_molecular_scope_labels": ["FLT3:ITD"]' in prompt
    assert f'"primary_admission_excerpt": "{primary}"' in prompt
    assert "do not merge the other variant" in prompt


def test_context_may_ground_interpretation_but_cannot_replace_atomic_primary_gate(brief):
    e = evidence("FLT3 inhibition was studied in acute myeloid leukemia cells.")
    full = e.metadata["excerpt"] + " Treatment reduced viability in those cells."
    enriched = with_context(e, full)
    m = meaning(enriched)
    result = annotate(brief, enriched, m, m)
    assert permits(result, brief, "FLT3", "direction")
    assert result.metadata["excerpt"] == e.metadata["excerpt"]
    assert _primary_direction_excerpt(brief, "FLT3", result) is None


@pytest.mark.parametrize("mutation", [
    lambda m: m[KEY].update(text=m[KEY]["text"] + " invented"),
    lambda m: m[KEY].update(sha256="0" * 64),
    lambda m: m[KEY].update(char_count=1),
    lambda m: m[KEY].update(char_count=True),
    lambda m: m[KEY].update(policy="unverified"),
    lambda m: m[KEY].update(status="partial"),
    lambda m: m[KEY].update(source_scope="publisher_pdf"),
    lambda m: m[KEY].update(authority="primary_evidence"),
    lambda m: m.update(content_hash="f" * 64),
    lambda m: m.update(content_hash_scope="europe_pmc_oa_jats_xml"),
    lambda m: m.update(abstract_interpretation_context=None),
    lambda m: m.update(fixture=True),
])
def test_invalid_context_fails_closed_without_legacy_positive_fallback(brief, mutation):
    e = with_context(evidence(), evidence().metadata["excerpt"] + " Further context.")
    good = annotate(brief, e)
    assert permits(good, brief, "FLT3", "direction")
    altered = good.model_copy(deep=True)
    mutation(altered.metadata)
    assert source_records([altered]) == []
    assert not permits(altered, brief, "FLT3", "direction")


def test_context_identity_changes_cache_and_source_bindings_but_legacy_keys_unchanged():
    e = evidence()
    from drugpilot.agents.evidence_semantics import digest, publication_key
    assert record_key(e) == digest({"publication":publication_key(e), "excerpt_hash":e.excerpt_hash})
    a = with_context(e, e.metadata["excerpt"] + " Alpha.")
    b = with_context(e, e.metadata["excerpt"] + " Beta.")
    assert len({record_key(e), record_key(a), record_key(b)}) == 3
    assert len({source_binding(e), source_binding(a), source_binding(b)}) == 3


def test_context_quotes_must_be_exact_and_from_same_document(brief):
    e = with_context(evidence(), evidence().metadata["excerpt"] + " Further context.")
    m = meaning(e, result_quote="Fabricated improved survival")
    assert not permits(annotate(brief, e, m, m), brief, "FLT3", "direction")
    unrelated = with_context(evidence(), "A different paper without the original excerpt.")
    assert source_records([unrelated]) == []


def test_oversized_abstract_is_explicitly_omitted_not_silently_truncated():
    e = evidence()
    full = e.metadata["excerpt"] + "x" * MAX_ABSTRACT_CONTEXT_CHARS
    large = with_context(e, full)
    assert large.metadata[KEY]["text"] is None
    assert large.metadata[KEY]["char_count"] == len(full)
    assert large.metadata[KEY]["status"] == "omitted_size_limit"
    assert source_records([large])[0]["excerpt"] == e.metadata["excerpt"]
    boundary = abstract_context("x" * MAX_ABSTRACT_CONTEXT_CHARS)
    assert boundary["status"] == "complete"


def test_parser_retains_complete_normalized_abstract_and_child_shares_document_hash():
    sections = [("RESULTS", evidence().metadata["excerpt"]),
                ("CONCLUSIONS", "Resistance remained a limitation.")]
    content = _publication_content_from_sections(sections, target_symbol="FLT3",
        disease="acute myeloid leukemia", intervention_direction="inhibit")
    full = " ".join(text for _, text in sections)
    assert content.interpretation_context == abstract_context(full)
    assert content.content_hash == sha256(full.encode()).hexdigest()
    metadata = _abstract_context_metadata(content)
    assert metadata["abstract_context_stored"] is True
    metadata[KEY]["text"] = "mutated copy"
    assert content.interpretation_context["text"] == full


def test_association_child_receives_same_complete_context_with_its_own_excerpt():
    association = "FLT3 mutations were detected in 30% of acute myeloid leukemia patients."
    full = evidence().metadata["excerpt"] + " " + association
    content = _publication_content_from_sections([("RESULTS",full)], target_symbol="FLT3",
        disease="acute myeloid leukemia", intervention_direction="inhibit")
    assert content.association_excerpt == association
    parent = with_context(evidence(), full)
    child = _association_role_record(parent,content)
    assert child is not None
    assert child.metadata["excerpt"] == association
    assert child.excerpt_hash != parent.excerpt_hash
    assert child.metadata[KEY] == parent.metadata[KEY]
    assert source_records([child])[0]["excerpt"] == full
    assert child.metadata["direct_support_eligible"] is False


def test_no_related_primary_or_association_excerpt_does_not_retain_unrelated_abstract():
    content = _publication_content_from_sections([("RESULTS","EGFR signaling was evaluated in lung carcinoma cells.")],
        target_symbol="FLT3",disease="acute myeloid leukemia")
    assert content.excerpt is None and content.association_excerpt is None
    assert _abstract_context_metadata(content) == {}


def test_noncontiguous_pair_keeps_legacy_excerpt_input_and_missing_abstract_keeps_legacy():
    e = evidence("First source span. Last source span.")
    e.metadata["source_location"] = "abstract-paired:test-contract"
    e = with_context(e, "First source span. Intervening text. Last source span.")
    assert source_records([e])[0]["excerpt"] == e.metadata["excerpt"]
    absent = _publication_content_from_sections([], target_symbol="FLT3", disease="AML")
    assert _abstract_context_metadata(absent) == {}
