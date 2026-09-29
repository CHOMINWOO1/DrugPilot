from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
import hashlib
from typing import Any

import pytest

from drugpilot.tools.primary_abstract_paper_native_v5 import (
    PAPER_NATIVE_V5_GRAMMAR,
    PAPER_NATIVE_V5_PARSER_SCHEMA_VERSION,
    PrimaryAbstractPair,
    parse_paper_native_primary_abstract_v5,
)


BRACKETED = (
    "Alpha was the most potent GENE inhibitor "
    "(half-maximal inhibitory concentration [IC50] at physiological [1mM] ATP, 2.9nM), "
    "followed by Beta (17nM), Gamma (29nM), and Delta (39nM), "
    "and the most selective."
)
UNBRACKETED = BRACKETED.replace("[IC50]", "IC50")
ALIASES = {
    "candidate-alpha": {"Alpha", "Alpha code"},
    "candidate-beta": {"Beta"},
    "candidate-gamma": {"Gamma"},
    "candidate-delta": {"Delta"},
    "candidate-absent": {"Epsilon"},
}


def _parse(sentence: str = BRACKETED, **overrides: Any) -> tuple[PrimaryAbstractPair, ...]:
    arguments: dict[str, Any] = {
        "candidate_aliases": ALIASES,
        "target_aliases": {"GENE", "Gene protein"},
        "endpoint": "IC50",
        "section": "Abstract",
        "publication_types": ["Journal Article", "Comparative Study"],
    }
    arguments.update(overrides)
    return parse_paper_native_primary_abstract_v5(sentence, **arguments)


def test_bracketed_endpoint_is_accepted_without_repair_and_all_offsets_are_exact() -> None:
    pairs = _parse()

    assert len(pairs) == 4
    assert PAPER_NATIVE_V5_PARSER_SCHEMA_VERSION == "primary_abstract_paper_native/v5"
    assert PAPER_NATIVE_V5_GRAMMAR == "head_measurement_followed_by_candidate_value_list_v1"
    assert [pair.candidate_id for pair in pairs] == [
        "candidate-alpha",
        "candidate-beta",
        "candidate-gamma",
        "candidate-delta",
    ]
    assert [pair.value_raw for pair in pairs] == ["2.9", "17", "29", "39"]
    assert [pair.value_decimal for pair in pairs] == [
        Decimal("2.9"),
        Decimal("17"),
        Decimal("29"),
        Decimal("39"),
    ]
    assert all(pair.endpoint == "IC50" for pair in pairs)
    assert all(pair.endpoint_span.text == "IC50" for pair in pairs)
    assert all(pair.endpoint_wrapper_raw == "[IC50]" for pair in pairs)
    assert all(pair.endpoint_wrapper_span is not None for pair in pairs)
    assert all(pair.grammar == PAPER_NATIVE_V5_GRAMMAR for pair in pairs)
    assert all(pair.relation_raw is None for pair in pairs)
    assert all(pair.relation_normalized is None for pair in pairs)
    assert all(pair.relation_basis == "reported_point_estimate" for pair in pairs)
    assert all(pair.literal_equality is False for pair in pairs)
    assert pairs[0].target_scope_basis == "explicit_common_head"
    assert all(
        pair.target_scope_basis == "inherited_from_common_head"
        and pair.endpoint_scope_basis == "inherited_from_common_head"
        and pair.assay_condition_scope_basis == "inherited_from_common_head"
        for pair in pairs[1:]
    )
    assert pairs[0].assay_condition.value_decimal == Decimal("1")
    assert pairs[0].assay_condition.span.text == "at physiological [1mM] ATP"

    for pair in pairs:
        spans = [
            pair.sentence_span,
            pair.candidate_span,
            pair.target_span,
            pair.endpoint_span,
            pair.endpoint_wrapper_span,
            pair.assay_condition.span,
            pair.assay_condition.value_span,
            pair.assay_condition.unit_span,
            pair.assay_condition.subject_span,
            pair.measurement_span,
            pair.value_span,
            pair.unit_span,
        ]
        assert all(
            span is not None and BRACKETED[span.start : span.end] == span.text
            for span in spans
        )
        assert pair.sentence_sha256 == hashlib.sha256(BRACKETED.encode()).hexdigest()
        assert len(pair.evidence_sha256) == 64

    with pytest.raises(FrozenInstanceError):
        pairs[0].endpoint_wrapper_raw = None  # type: ignore[misc]


def test_bare_endpoint_is_accepted_and_has_no_synthetic_wrapper() -> None:
    pairs = _parse(UNBRACKETED)

    assert len(pairs) == 4
    assert all(pair.endpoint_span.text == "IC50" for pair in pairs)
    assert all(pair.endpoint_wrapper_raw is None for pair in pairs)
    assert all(pair.endpoint_wrapper_span is None for pair in pairs)


def test_literal_equality_is_only_recorded_when_source_contains_equals() -> None:
    sentence = BRACKETED.replace("Beta (17nM)", "Beta (= 17nM)")
    pairs = _parse(sentence)

    assert len(pairs) == 4
    assert pairs[0].relation_raw is None
    assert pairs[0].relation_normalized is None
    assert pairs[0].relation_basis == "reported_point_estimate"
    assert pairs[1].relation_raw == "="
    assert pairs[1].relation_normalized == "="
    assert pairs[1].relation_basis == "literal_equality"
    assert pairs[1].literal_equality is True
    assert pairs[1].relation_span is not None
    assert pairs[1].relation_span.text == "="


def test_mapping_order_cannot_change_pairs_or_canonical_evidence_hashes() -> None:
    assert _parse(candidate_aliases=dict(reversed(tuple(ALIASES.items())))) == _parse()


def test_contextual_alphanumeric_numerals_are_allowed_only_inside_bound_identifiers() -> None:
    sentence = BRACKETED.replace("Alpha", "Alpha3").replace("GENE", "GENE2")
    aliases = {
        **{key: value for key, value in ALIASES.items() if key != "candidate-alpha"},
        "candidate-alpha": {"Alpha3"},
    }

    pairs = _parse(
        sentence,
        candidate_aliases=aliases,
        target_aliases={"GENE2"},
    )

    assert len(pairs) == 4
    assert pairs[0].candidate_span.text == "Alpha3"
    assert all(pair.target_span.text == "GENE2" for pair in pairs)
    assert all(pair.endpoint_span.text == "IC50" for pair in pairs)


def test_legitimate_experimental_wording_remains_accepted() -> None:
    sentence = BRACKETED.replace(
        "Alpha was the most potent",
        "Alpha was experimentally the most potent",
    )
    assert len(_parse(sentence)) == 4


def test_retrospective_expected_and_observed_wording_remains_accepted() -> None:
    retrospective = BRACKETED.replace(
        "Alpha was the most potent",
        "Alpha was experimentally, as expected, the most potent",
    )
    observed = BRACKETED.replace(
        "Alpha was the most potent",
        "Alpha was experimentally observed to be the most potent",
    )

    assert len(_parse(retrospective)) == 4
    assert len(_parse(observed)) == 4


@pytest.mark.parametrize(
    "head",
    [
        "Alpha was measured as the most potent GENE inhibitor",
        "Alpha was determined to be the most potent GENE inhibitor",
        "Alpha was determined to inhibit GENE",
        "Alpha inhibited GENE",
    ],
    ids=[
        "measured-potency",
        "determined-potency",
        "determined-inhibition",
        "observed-active-inhibition",
    ],
)
def test_closed_observed_or_measured_head_forms_remain_accepted(head: str) -> None:
    sentence = BRACKETED.replace("Alpha was the most potent GENE inhibitor", head)

    assert len(_parse(sentence)) == 4


def test_unbound_alphanumeric_contextual_numeral_rejects_the_complete_group() -> None:
    sentence = BRACKETED.replace("most potent GENE", "most potent phase3 GENE")
    assert _parse(sentence) == ()


@pytest.mark.parametrize(
    "phrase",
    [
        "predicted in silico",
        "prediction based",
        "computational",
        "computationally assessed",
        "modeled",
        "modelled",
        "simulated",
        "docking-derived",
    ],
)
def test_nonexperimental_prediction_or_computation_language_rejects_whole_group(
    phrase: str,
) -> None:
    sentence = BRACKETED.replace(
        "Alpha was the most potent",
        f"Alpha was {phrase} the most potent",
    )
    assert _parse(sentence) == ()


@pytest.mark.parametrize(
    "phrase",
    [
        "projected",
        "hypothetical",
        "putative",
        "planned to be",
        "expected to be",
        "would be",
        "will be",
        "proposed to be",
        "prospective",
        "prospectively",
    ],
)
def test_forward_or_nonobservational_wording_rejects_whole_group(phrase: str) -> None:
    sentence = BRACKETED.replace(
        "Alpha was the most potent",
        f"Alpha was {phrase} the most potent",
    )
    assert _parse(sentence) == ()


@pytest.mark.parametrize(
    "head",
    [
        "Alpha was in-silico the most potent",
        "Alpha was modeling the most potent",
        "Alpha was modelling the most potent",
        "Alpha was simulation ranked the most potent",
        "Alpha was projecting the most potent",
        "Alpha was estimated to be the most potent",
        "Alpha was forecast as the most potent",
        "Alpha was anticipated to be the most potent",
        "Alpha was aimed to be the most potent",
        "Alpha was plans to be the most potent",
        "Alpha was proposes to be the most potent",
        "Alpha was will likely be the most potent",
        "Alpha was would have been the most potent",
    ],
    ids=[
        "hyphenated-in-silico",
        "modeling",
        "modelling",
        "simulation",
        "projecting",
        "estimated",
        "forecast",
        "anticipated",
        "aimed",
        "plans",
        "proposes",
        "will-likely",
        "would-have-been",
    ],
)
def test_adversarial_nonobservational_morphology_rejects_whole_group(head: str) -> None:
    sentence = BRACKETED.replace("Alpha was the most potent", head)

    assert _parse(sentence) == ()


def test_unrecognized_head_modifier_fails_closed_without_blacklist_dependency() -> None:
    sentence = BRACKETED.replace(
        "Alpha was the most potent",
        "Alpha was algorithmically ranked the most potent",
    )

    assert _parse(sentence) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        BRACKETED.replace("[IC50]", "[IC50"),
        BRACKETED.replace("[IC50]", "IC50]"),
        BRACKETED.replace("[IC50]", "[[IC50]]"),
        BRACKETED.replace("[IC50]", "[IC50)]"),
        BRACKETED.replace("[1mM]", "[1mM"),
        BRACKETED.replace("[IC50]", "[[[[[IC50]]]]]"),
    ],
    ids=[
        "endpoint-unclosed",
        "endpoint-unopened",
        "endpoint-double-wrapper",
        "crossed-delimiter",
        "condition-unclosed",
        "delimiter-depth",
    ],
)
def test_delimiter_stack_and_exact_single_endpoint_wrapper_fail_closed(sentence: str) -> None:
    assert _parse(sentence) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        BRACKETED.replace(
            "Beta (17nM), Gamma (29nM), and Delta (39nM)",
            "Beta and Gamma, respectively (17nM and 29nM), and Delta (39nM)",
        ),
        BRACKETED.replace("17nM", "17-19nM"),
        BRACKETED.replace("17nM", "<17nM"),
        BRACKETED.replace("17nM", "17/19nM"),
        BRACKETED.replace("Beta (17nM)", "Beta compared with Gamma (17nM)"),
        BRACKETED.replace("[IC50] at", "[IC50] and EC50 at"),
        BRACKETED.replace("Beta (17nM)", "Beta for OTHER (17nM)"),
        BRACKETED.replace("Beta (17nM)", "Beta at [2mM] ATP (17nM)"),
        BRACKETED[:-1] + " in 3 assays.",
    ],
    ids=[
        "respectively",
        "range",
        "inequality",
        "slash-list",
        "comparator",
        "endpoint-reset",
        "target-reset",
        "condition-reset",
        "unconsumed-number",
    ],
)
def test_ambiguous_or_reset_scope_and_unconsumed_values_reject_whole_group(
    sentence: str,
) -> None:
    assert _parse(sentence) == ()


def test_at_least_three_distinct_candidates_are_required_and_partial_salvage_is_forbidden() -> None:
    two = (
        "Alpha was the most potent GENE inhibitor "
        "(IC50 at physiological [1mM] ATP, 2.9nM), followed by Beta (17nM)."
    )
    assert _parse(two) == ()

    incomplete = {key: value for key, value in ALIASES.items() if key != "candidate-delta"}
    assert _parse(candidate_aliases=incomplete) == ()


def test_endpoint_input_is_a_base_token_not_a_wrapped_or_expanded_phrase() -> None:
    assert _parse(endpoint="[IC50]") == ()
    assert _parse(endpoint="half-maximal inhibitory concentration [IC50]") == ()
