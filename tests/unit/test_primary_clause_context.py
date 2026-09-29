"""Scoped experimental-context semantics; synthetic tests, no live evidence."""
import pytest

from drugpilot.agents.ma3 import (
    _primary_clause_semantically_unsafe as ma3_unsafe,
    _paper_v6_clause_is_unambiguous,
)
from drugpilot.tools.live_ma2_gateway import _primary_clause_semantically_unsafe as gateway_unsafe
from drugpilot.tools.primary_abstract_paper_native_search_v6 import _clause_semantically_unsafe as v6_unsafe

BASE = "Compound A inhibited FLT3 with an IC50 of 10 nM"


def unambiguous(sentence, *, competitors=()):
    return _paper_v6_clause_is_unambiguous(sentence, aliases=["Compound A"],
        primary={"competitor_alias_provenance": {
            "policy": "all_bounded_pool_candidate_aliases_except_current_v1",
            "truncated": False, "competitor_exact_aliases": list(competitors),
        }}, target_symbol="FLT3", endpoint="IC50", relation="=", value=10, unit="nM")


@pytest.mark.parametrize("suffix", [
    ".", " in an AML cell model.", " in cellular models.", " in a murine model.",
    " in an animal model.", " in a xenograft model.", " in an organoid model.",
    " in an in vitro model.", " without cytotoxicity.", " without significant cytotoxicity.",
    " without detectable toxicity.", " without measurable cytotoxicity.",
])
def test_verified_potency_not_negated_by_experimental_context(suffix):
    sentence = BASE + suffix
    assert not ma3_unsafe(sentence)
    assert not gateway_unsafe(sentence)
    assert not v6_unsafe(sentence)
    assert unambiguous(sentence)


@pytest.mark.parametrize("sentence", [
    "A model predicted that " + BASE + ".",
    "Computational cell models predicted that " + BASE + ".",
    "Cell modeling indicated that " + BASE + ".",
    "It was estimated that " + BASE + " in an AML cell model.",
    "In silico " + BASE + " in an AML cell model.",
    "Compound A did not inhibit FLT3 with an IC50 of 10 nM in an AML cell model.",
    "Compound A inhibited FLT3 but its IC50 was not 10 nM.",
    "Compound A was without inhibition of FLT3 at an IC50 of 10 nM.",
    BASE + " without experimental validation.",
    BASE + " without cytotoxicity data.",
    "Compound A might inhibit FLT3 with an IC50 of 10 nM without cytotoxicity.",
    BASE + " according to a previous study.",
    BASE.replace("10 nM", "approximately 10 nM") + " in an AML cell model.",
])
def test_prediction_negation_uncertainty_and_missing_validation_still_rejected(sentence):
    assert ma3_unsafe(sentence)
    assert gateway_unsafe(sentence)
    assert v6_unsafe(sentence)
    assert not unambiguous(sentence)


def test_context_exception_does_not_allow_combination_attribution_or_mixed_owner():
    combined = "Compound A in combination with Compound B inhibited FLT3 with an IC50 of 10 nM in an AML cell model."
    assert not unambiguous(combined, competitors=["Compound B"])
    split_owner = "Compound A inhibited FLT3; Compound B had an IC50 of 10 nM in an AML cell model."
    assert not unambiguous(split_owner, competitors=["Compound B"])


def test_context_exception_does_not_waive_exact_numeric_match():
    assert not unambiguous(BASE.replace("10 nM", "20 nM") + " in an AML cell model.")
