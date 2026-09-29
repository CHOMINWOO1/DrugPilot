from __future__ import annotations

import pytest
from drugpilot.tools.assay_variant_alignment import assay_variant_alignment as _align, extract_target_variants


def assay_variant_alignment(**kwargs):
    kwargs.setdefault("expected_target_chembl_id", "CHEMBL1974")
    return _align(**kwargs)


def _context(description: str | None, variant: str | None) -> dict:
    return {
        "schema_version": "chembl_assay_context_v1",
        "retrieval_status": "completed",
        "missing_values_imputed": False,
        "inference_used": False,
        "assay_chembl_id": "CHEMBL6123853",
        "binding_status": "exact",
        "record": {
            "assay_chembl_id": "CHEMBL6123853",
            "target_chembl_id": "CHEMBL1974",
            "description": {"excerpt": description} if description else None,
            "variant_sequence": {"excerpt": variant} if variant else None,
        },
        "condition_coverage": {
            "mutation_state": (
                "explicit_variant_sequence"
                if variant
                else "unknown_not_assumed_wild_type"
            )
        },
    }


def test_generic_itd_is_only_class_alignment_and_non_scoring() -> None:
    result = assay_variant_alignment(
        target_symbol="FLT3",
        hypothesis_texts=("Evaluate FLT3-ITD AML",),
        chembl_assay_context=_context(
            "Inhibition of human FLT3 ITD mutant",
            "{'mutation': 'UNDEFINED MUTATION'}",
        ),
    )

    assert result["status"] == "variant_class_match_only"
    assert result["exact_variant_match"] is False
    assert result["construct_identity_verified"] is False
    assert result["assay_variant_sequence_unresolved"] is True
    assert result["declared_hypothesis_variants"] == ["FLT3-ITD"]
    assert "FLT3-ITD" in result["observed_assay_variants"]
    assert result["score_effect_applied"] is False
    assert result["eligibility_effect_applied"] is False


@pytest.mark.parametrize("text", ["NRAS Q61R", "NRAS-Q61R", "NRAS:Q61R",
                                  "FLT3 inhibition in NRAS Q61R cells", "Q61R"])
def test_free_prose_does_not_reassign_another_gene_variant(text):
    assert extract_target_variants("FLT3", (text,)) == ()


@pytest.mark.parametrize("variant", ["NRAS Q61R", "{'gene': 'NRAS', 'mutation': 'Q61R'}"])
def test_embedded_variant_owner_is_not_discarded(variant):
    result = assay_variant_alignment(target_symbol="FLT3", hypothesis_texts=("FLT3 Q61R",),
        chembl_assay_context=_context("FLT3 inhibition", variant))
    assert result["observed_assay_variants"] == []
    assert result["status"] == "unknown_not_assumed_match"


@pytest.mark.parametrize("field", ["binding_status", "assay_chembl_id", "target_chembl_id", "expected_target",
                                  "schema_version", "retrieval_status", "missing_values_imputed", "inference_used"])
def test_bare_variant_requires_exact_target_and_assay_binding(field):
    context = _context(None, "D835Y")
    expected = "CHEMBL1974"
    if field == "target_chembl_id":
        context["record"][field] = "OTHER"
    elif field == "expected_target":
        expected = None
    else:
        context[field] = "OTHER"
    result = assay_variant_alignment(target_symbol="FLT3", hypothesis_texts=("FLT3 D835Y",),
        chembl_assay_context=context, expected_target_chembl_id=expected)
    assert result["status"] == "unknown_not_assumed_match"
    assert result["observed_assay_variants"] == []


def test_exact_named_residue_is_annotation_not_construct_identity():
    result = assay_variant_alignment(target_symbol="FLT3", hypothesis_texts=("FLT3 D835Y",),
        chembl_assay_context=_context(None, "D835Y"))
    assert result["status"] == "variant_annotation_match"
    assert result["exact_variant_match"] is False


def test_partial_and_compound_labels_are_not_exact():
    for description in ("FLT3 ITD", "FLT3 ITD and FLT3 D835Y"):
        result = assay_variant_alignment(target_symbol="FLT3",
            hypothesis_texts=("FLT3 ITD and FLT3 D835Y",),
            chembl_assay_context=_context(description, None))
        assert result["status"] == "partial_variant_overlap"


def test_explicitly_undefined_sequence_does_not_support_named_residue():
    result = assay_variant_alignment(target_symbol="FLT3", hypothesis_texts=("FLT3 D835Y",),
        chembl_assay_context=_context("FLT3 D835Y", "UNDEFINED MUTATION"))
    assert result["status"] == "unknown_not_assumed_match"


def test_explicit_residue_mismatch_is_retained_as_limitation() -> None:
    result = assay_variant_alignment(
        target_symbol="FLT3",
        hypothesis_texts=("Test FLT3-ITD inhibition",),
        chembl_assay_context=_context(
            "Inhibition of FLT3 D835Y mutant",
            "{'mutation': 'D835Y'}",
        ),
    )

    assert result["status"] == "explicit_mismatch"
    assert result["observed_assay_variants"] == ["FLT3-D835Y"]
    assert result["reason_codes"] == ["ASSAY_VARIANT_EXPLICIT_MISMATCH"]


def test_absent_assay_variant_is_never_assumed_wild_type_or_match() -> None:
    result = assay_variant_alignment(
        target_symbol="JAK2",
        hypothesis_texts=("Test JAK2-V617F cells",),
        chembl_assay_context=_context("Inhibition of human JAK2", None),
    )

    assert result["status"] == "unknown_not_assumed_match"
    assert result["observed_assay_variants"] == []
    assert result["unknown_assumed_wild_type"] is False
    assert result["unknown_assumed_match"] is False


def test_no_declared_variant_does_not_create_a_requirement() -> None:
    result = assay_variant_alignment(
        target_symbol="FLT3",
        hypothesis_texts=("Evaluate FLT3 inhibition in AML",),
        chembl_assay_context=_context(
            "Inhibition of FLT3 D835Y mutant",
            "{'mutation': 'D835Y'}",
        ),
    )

    assert result["status"] == "not_declared"
    assert result["declared_hypothesis_variants"] == []
    assert result["eligibility_effect_applied"] is False
