"""Synthetic span-contract tests only; never live biological evidence."""
from hashlib import sha256
import pytest

from drugpilot.agents.molecular_scope import mentions, source_scope, scope_check, annotate_scopes
from drugpilot.agents.evidence_semantics import EvidenceMeaningBatch, attach_interpretations, permits, source_records, source_binding
from test_evidence_semantics import evidence, meaning


@pytest.mark.parametrize("target,text,label", [
    ("ALK","EML4-ALK fusion","EML4::ALK"),
    ("EML4","EML4–ALK fusion","EML4::ALK"),
    ("RET","CCDC6::RET","CCDC6::RET"),
    ("NTRK1","NOTCH2-NTRK1 fusion","NOTCH2::NTRK1"),
    ("FLT3","FLT3-ITD","FLT3:ITD"),
    ("JAK2","JAK2 V617F","JAK2:V617F"),
    ("EGFR","EGFR-L858R","EGFR:L858R"),
    ("MET","MET exon 14 skipping","MET:EXON 14 SKIPPING"),
    ("ERBB2","ERBB2 exon 20 insertions","ERBB2:EXON 20 INSERTIONS"),
    ("EGFR","wild-type EGFR","EGFR:WT"),
])
def test_exact_owner_linked_label_spans(target,text,label):
    result = mentions(text,target)
    assert [m["label"] for m in result] == [label]
    assert all(text[m["start"]:m["end"]] == m["quote"] for m in result)
    assert all(m["identity_verified"] is False for m in result)


@pytest.mark.parametrize("target,text", [
    ("FLT3","NRAS Q61R cells"), ("ALK","MAPK1-MAPK3 pathway"),
    ("STAT3","JAK2-STAT3 signaling"), ("FLT3","G12D"),
    ("EGFR","EGFR inhibition"), ("ALK","ALK-positive cells"),
    ("RET","CCDC6-RET complex. Another unrelated fusion was detected."),
])
def test_unknown_or_unlinked_tokens_are_not_specific_constructs(target,text):
    assert mentions(text,target) == []


def test_target_variant_is_not_misread_as_gene_fusion():
    rows = mentions("EGFR-L858R mutation and EML4-ALK fusion", "EGFR")
    assert [m["kind"] for m in rows] == ["amino_acid_variant"]


def test_negated_mention_is_not_claimed_as_observed_positive():
    row = mentions("No EML4-ALK fusion was detected.", "ALK")[0]
    assert row["assertion_status"] == "mention_not_presence_or_effect"


def test_hash_and_provenance_guard_and_no_implicit_wild_type():
    e = evidence("EGFR inhibition was measured.")
    assert source_scope(e,"EGFR")["mentions"] == []
    assert not source_scope(e,"EGFR")["wild_type_inferred"]
    e.metadata["excerpt"] = "EGFR L858R inhibition was measured."
    assert not source_scope(e,"EGFR")["source_verified"]
    e.excerpt_hash = sha256(e.metadata["excerpt"].encode()).hexdigest()
    e.metadata["fixture"] = True
    assert not source_scope(e,"EGFR")["source_verified"]


def test_observed_scope_does_not_rewrite_brief_or_source(brief):
    e = evidence("EML4-ALK fusion was measured in lung cancer cells.")
    before = brief.model_dump_json(), e.model_dump_json()
    annotated = annotate_scopes(brief,"ALK",[e])[0]
    result = annotated.metadata["molecular_scope"]
    assert result["declared_labels"] == []
    assert result["observed_labels"] == ["EML4::ALK"]
    assert result["status"] == "scoped_source_not_general_target"
    assert source_binding(e) == source_binding(annotated)
    assert (brief.model_dump_json(),e.model_dump_json()) == before


def test_declared_variant_does_not_match_a_different_variant_or_wild_type(brief):
    b = brief.model_copy(deep=True)
    b.constraints.target_cell_type = "EGFR L858R cells"
    for text in ("EGFR T790M was studied.", "wild-type EGFR was studied."):
        assert scope_check(b,"EGFR",evidence(text))["status"] == "declared_observed_scope_mismatch"
    assert scope_check(b,"EGFR",evidence("EGFR L858R was studied."))["status"] == "no_detected_scope_conflict"
    assert scope_check(b,"EGFR",evidence("EGFR L858R was studied."))["source_scope"]["construct_identity_verified"] is False


def test_declared_variant_is_not_invalidated_by_separately_reported_variant(brief):
    b = brief.model_copy(deep=True)
    b.constraints.target_cell_type = "FLT3-ITD AML cells"
    result = scope_check(
        b,
        "FLT3",
        evidence(
            "Compound W4 inhibited FLT3-ITD (IC50 = 16.0 nM) and "
            "FLT3-D835Y (IC50 = 20.4 nM)."
        ),
    )
    assert result["status"] == "declared_scope_present_with_additional_scopes"
    assert result["declared_labels"] == ["FLT3:ITD"]
    assert result["observed_labels"] == ["FLT3:D835Y", "FLT3:ITD"]


def test_identical_model_votes_cannot_generalize_a_fusion_partner(brief):
    e = evidence("EML4-ALK fusion inhibition reduced viability in acute myeloid leukemia cells.")
    m = meaning(e, target_quote="ALK",intervention_quote="EML4-ALK fusion inhibition")
    batch = EvidenceMeaningBatch(records=[m])
    result = attach_interpretations(brief,"ALK",[e],source_records([e]),batch,batch)[0]
    assert result.metadata["semantic_interpretation"]["status"] == "grounded_agreement"
    assert not permits(result,brief,"ALK","direction")
    result.metadata["molecular_scope"]["status"] = "no_detected_scope_conflict"
    assert not permits(result,brief,"ALK","direction")
    del result.metadata["molecular_scope"]
    assert not permits(result,brief,"ALK","direction")


def test_scope_overflow_is_visible_and_cannot_silently_pass(brief):
    e = evidence(" ".join("EGFR L858R." for _ in range(14)))
    s = scope_check(brief,"EGFR",e)
    assert s["status"] == "scope_unverifiable"
    assert s["source_scope"]["mention_count"] == 14
    assert s["source_scope"]["mentions_omitted"] == 2
