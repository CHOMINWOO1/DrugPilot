from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import re
from typing import Any

import pytest

from drugpilot.tools.primary_abstract_pairs import (
    PrimaryAbstractPair,
    parse_primary_abstract_pairs,
)


ENDPOINT = "half-maximal inhibitory concentration [IC50]"
BASE_SENTENCE = (
    "Alphatinib was the most potent KIN2 inhibitor "
    "(half-maximal inhibitory concentration [IC50] at physiological [1mM] ATP, 2.9nM), "
    "followed by Betanib (17nM), Gammanib (29nM), and Deltanib (39nM), "
    "and the most selective."
)
CANDIDATE_ALIASES = {
    "candidate-alpha": {"Alphatinib"},
    "candidate-beta": {"Betanib"},
    "candidate-gamma": {"Gammanib"},
    "candidate-delta": {"Deltanib"},
    # Discovered aliases need not all occur in a particular source sentence.
    "candidate-absent": {"Epsilonib"},
}


def _parse(
    sentence: str = BASE_SENTENCE,
    **overrides: Any,
) -> tuple[PrimaryAbstractPair, ...]:
    arguments: dict[str, Any] = {
        "candidate_aliases": CANDIDATE_ALIASES,
        "target_aliases": {"KIN2", "Kinase Two"},
        "endpoint": ENDPOINT,
        "section": "Abstract",
        "publication_types": ["Journal Article", "Research Support, Non-U.S. Gov't"],
    }
    arguments.update(overrides)
    return parse_primary_abstract_pairs(sentence, **arguments)


def _replace(old: str, new: str) -> str:
    assert BASE_SENTENCE.count(old) == 1
    return BASE_SENTENCE.replace(old, new)


def test_jak2_shaped_generic_sentence_returns_source_order_and_exact_provenance() -> None:
    pairs = _parse()

    assert isinstance(pairs, tuple)
    assert [pair.candidate_id for pair in pairs] == [
        "candidate-alpha",
        "candidate-beta",
        "candidate-gamma",
        "candidate-delta",
    ]
    assert [pair.candidate_alias for pair in pairs] == [
        "Alphatinib",
        "Betanib",
        "Gammanib",
        "Deltanib",
    ]
    assert [pair.value_raw for pair in pairs] == ["2.9", "17", "29", "39"]
    assert [pair.unit_raw for pair in pairs] == ["nM", "nM", "nM", "nM"]
    assert [pair.pair_role for pair in pairs] == [
        "common_head",
        "direct_follower",
        "direct_follower",
        "direct_follower",
    ]
    assert [pair.pair_index for pair in pairs] == [0, 1, 2, 3]

    sentence_hash = hashlib.sha256(BASE_SENTENCE.encode("utf-8")).hexdigest()
    assert all(pair.sentence_sha256 == sentence_hash for pair in pairs)
    assert len({pair.evidence_sha256 for pair in pairs}) == 4
    assert all(re.fullmatch(r"[0-9a-f]{64}", pair.evidence_sha256) for pair in pairs)
    assert all(pair.relation_raw is None for pair in pairs)
    assert all(pair.relation_span is None for pair in pairs)
    assert all(pair.relation_normalized == "=" for pair in pairs)
    assert all(
        pair.relation_basis == "direct_assertive_measurement" for pair in pairs
    )

    condition = pairs[0].assay_condition
    assert condition.span.text == "at physiological [1mM] ATP"
    assert condition.value_raw == "1"
    assert condition.unit_raw == "mM"
    assert condition.subject == "ATP"
    assert all(pair.assay_condition == condition for pair in pairs)
    assert all(pair.target_alias == "KIN2" for pair in pairs)
    assert all(pair.endpoint == ENDPOINT for pair in pairs)
    assert all(pair.section == "Abstract" for pair in pairs)
    assert all(
        pair.publication_types
        == ("journal article", "research support, non-u.s. gov't")
        for pair in pairs
    )

    for pair in pairs:
        spans = (
            pair.sentence_span,
            pair.candidate_span,
            pair.target_span,
            pair.endpoint_span,
            pair.assay_condition.span,
            pair.assay_condition.value_span,
            pair.assay_condition.unit_span,
            pair.assay_condition.subject_span,
            pair.measurement_span,
            pair.value_span,
            pair.unit_span,
        )
        assert all(BASE_SENTENCE[span.start : span.end] == span.text for span in spans)

    with pytest.raises(FrozenInstanceError):
        pairs[0].value_raw = "999"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pairs[0].assay_condition.subject = "OTHER"  # type: ignore[misc]


def test_literal_equals_is_raw_only_when_present() -> None:
    sentence = _replace("Betanib (17nM)", "Betanib (= 17nM)")
    pairs = _parse(sentence)

    assert len(pairs) == 4
    assert pairs[0].relation_raw is None
    assert pairs[1].relation_raw == "="
    assert pairs[1].relation_span is not None
    assert pairs[1].relation_span.text == "="
    assert pairs[1].measurement_span.text == "= 17nM"
    assert all(pair.relation_normalized == "=" for pair in pairs)


def test_results_section_and_literal_head_equals_are_accepted() -> None:
    sentence = _replace(", 2.9nM), followed", ", =2.9nM), followed")
    pairs = _parse(sentence, section="results")

    assert len(pairs) == 4
    assert pairs[0].relation_raw == "="
    assert pairs[0].relation_span is not None
    assert pairs[0].relation_span.text == "="
    assert all(pair.section == "Results" for pair in pairs)


def test_input_mapping_order_cannot_change_output_or_hashes() -> None:
    reversed_aliases = dict(reversed(tuple(CANDIDATE_ALIASES.items())))

    expected = _parse()
    actual = _parse(candidate_aliases=reversed_aliases)

    assert actual == expected


def test_sentence_reordering_maps_each_direct_value_to_its_exact_alias() -> None:
    sentence = _replace(
        "Betanib (17nM), Gammanib (29nM), and Deltanib (39nM)",
        "Deltanib (39nM), Betanib (17nM), and Gammanib (29nM)",
    )

    pairs = _parse(sentence)

    assert [(pair.candidate_id, pair.value_raw) for pair in pairs] == [
        ("candidate-alpha", "2.9"),
        ("candidate-delta", "39"),
        ("candidate-beta", "17"),
        ("candidate-gamma", "29"),
    ]


@pytest.mark.parametrize(
    "candidate_aliases,sentence",
    [
        (
            {
                "one": {"Alphatinib"},
                "two": {"alphatinib"},
                "beta": {"Betanib"},
                "gamma": {"Gammanib"},
                "delta": {"Deltanib"},
            },
            BASE_SENTENCE,
        ),
        (
            {
                **CANDIDATE_ALIASES,
                "candidate-beta-xr": {"Betanib XR"},
            },
            _replace("Betanib (17nM)", "Betanib XR (17nM)"),
        ),
        (
            {
                **CANDIDATE_ALIASES,
                "candidate-beta": {"Betanib", "BetaDrug"},
            },
            _replace("Betanib (17nM)", "Betanib BetaDrug (17nM)"),
        ),
    ],
    ids=["cross-id-casefold-collision", "overlap", "two-aliases-one-id"],
)
def test_alias_collisions_fail_the_whole_sentence(
    candidate_aliases: dict[str, set[str]], sentence: str
) -> None:
    assert _parse(sentence, candidate_aliases=candidate_aliases) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        BASE_SENTENCE + " " + ("x" * 140),
        _replace(", followed by", ". Followed by"),
        _replace(
            "Betanib (17nM), Gammanib (29nM), and Deltanib (39nM)",
            "Betanib, Gammanib, and Deltanib, respectively (17nM, 29nM, and 39nM)",
        ),
        BASE_SENTENCE[:-1] + " [12].",
        _replace("2.9nM", "previously reported 2.9nM"),
        _replace("17nM", "about 17nM"),
        _replace("17nM", "~17nM"),
        _replace("17nM", "<17nM"),
        _replace("17nM", "17±2nM"),
        _replace("17nM", "17-19nM"),
        _replace("17nM", "0nM"),
        _replace("[1mM] ATP", "[1mm] ATP"),
        _replace("Betanib (17nM)", "Betanib"),
        _replace("Betanib (17nM)", "Betanib showed (17nM)"),
        _replace("Betanib (17nM)", "Betanib (17nM and 18nM)"),
        _replace("Betanib (17nM)", "Betanib (17)"),
        _replace(
            "Betanib (17nM), Gammanib (29nM), and Deltanib (39nM)",
            "Betanib (17), Gammanib (29), and Deltanib (39nM)",
        ),
        BASE_SENTENCE[:-1] + " for KIN2.",
        _replace("Betanib (17nM)", "Betanib for KIN3 (17nM)"),
        _replace("[IC50] at", "[IC50] and IC50 at"),
        _replace("[IC50] at", "[IC50] and Kd at"),
        _replace("[IC50] at", "[IC50] and pSTAT5 at"),
        _replace(
            "Betanib (17nM)",
            "Betanib at physiological [1mM] ATP (17nM)",
        ),
        _replace(", 2.9nM), followed", "), followed"),
        BASE_SENTENCE[:-1] + ", in 3 assays.",
        _replace(
            "Betanib (17nM), Gammanib",
            "Betanib (17nM), Betanib (18nM), Gammanib",
        ),
        _replace("followed by", "and then"),
        _replace("followed by", "followed\nby"),
        " " + BASE_SENTENCE,
        BASE_SENTENCE + " ",
        _replace("was the most", "was not the most"),
        _replace("was the most", "may be the most"),
        _replace("[1mM] ATP, 2.9nM", "[1mM] ATP and 2.9nM"),
        _replace("Betanib (17nM)", "Betanib (17nM, 95% confidence)"),
        _replace("17nM", "= 17nM ="),
    ],
    ids=[
        "over-360",
        "cross-sentence",
        "respectively-only",
        "citation",
        "previously-reported",
        "about",
        "tilde",
        "less-than",
        "plus-minus",
        "range",
        "zero",
        "condition-unit-case-is-exact",
        "candidate-without-value",
        "value-not-direct",
        "competing-values",
        "missing-unit",
        "shared-tail-unit",
        "target-twice",
        "different-tail-target",
        "endpoint-twice",
        "additional-kd-endpoint",
        "additional-pstat-endpoint",
        "second-condition",
        "condition-cannot-be-potency",
        "unconsumed-number",
        "duplicate-candidate",
        "missing-followed-by",
        "cross-line",
        "leading-space",
        "trailing-space",
        "negative-relation",
        "uncertain-relation",
        "missing-head-comma",
        "confidence-number",
        "competing-equals",
    ],
)
def test_adversarial_sentence_matrix_is_fail_closed(sentence: str) -> None:
    assert _parse(sentence) == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"section": "Introduction"},
        {"section": "Methods"},
        {"section": ""},
        {"publication_types": ["Review"]},
        {"publication_types": ["Journal Article", "Systematic Review"]},
        {"publication_types": ["Editorial"]},
        {"publication_types": ["Dataset"]},
        {"publication_types": []},
        {"candidate_aliases": {}},
        {"target_aliases": []},
        {"target_aliases": "KIN2"},
        {"endpoint": "IC50"},
    ],
    ids=[
        "introduction",
        "methods",
        "empty-section",
        "review",
        "mixed-review",
        "editorial",
        "no-primary-type",
        "empty-publication-types",
        "empty-candidates",
        "empty-targets",
        "target-string-not-collection",
        "endpoint-not-exact-full-string",
    ],
)
def test_metadata_and_discovery_inputs_fail_closed(overrides: dict[str, Any]) -> None:
    assert _parse(**overrides) == ()


def test_unknown_candidate_alias_prevents_partial_salvage() -> None:
    incomplete_aliases = {
        candidate_id: aliases
        for candidate_id, aliases in CANDIDATE_ALIASES.items()
        if candidate_id != "candidate-delta"
    }

    assert _parse(candidate_aliases=incomplete_aliases) == ()


def test_target_alias_ambiguity_and_overlap_are_rejected() -> None:
    assert _parse(target_aliases={"KIN2", "kin2"}) == ()
    sentence = BASE_SENTENCE.replace("KIN2 inhibitor", "KIN2 kinase inhibitor")
    assert _parse(sentence, target_aliases={"KIN2", "KIN2 kinase"}) == ()


def test_non_string_or_custom_shape_inputs_never_escape_the_boundary() -> None:
    assert parse_primary_abstract_pairs(  # type: ignore[arg-type]
        None,
        candidate_aliases=CANDIDATE_ALIASES,
        target_aliases={"KIN2"},
        endpoint=ENDPOINT,
        section="Abstract",
        publication_types="Journal Article",
    ) == ()
    assert _parse(candidate_aliases={"candidate-alpha": "Alphatinib"}) == ()
