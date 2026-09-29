from copy import deepcopy
import hashlib

import pytest

from drugpilot.agents.ma3 import (
    _paper_v6_clause_is_unambiguous,
    _paper_v6_exact_value_unit_present,
    _primary_clause_binding as _ma3_primary_clause_binding,
    _primary_exact_activity_value_spans,
)
from drugpilot.domain.models import ActivityMeasurement, TargetCandidate
from drugpilot.tools.live_ma2_gateway import (
    _exact_activity_value_spans,
    _primary_clause_binding as _live_primary_clause_binding,
)
from drugpilot.tools.primary_abstract_paper_native_search_v6 import (
    PAPER_NATIVE_V6_POLICY_TOKEN,
    PaperNativeV6ContractError,
    _exact_value_unit_present,
    aggregate_candidate_primary_support_v6,
)


IDS = [f"CHEMBL{i}0" for i in range(1, 13)]


def _unit(candidate_id: str, index: int):
    alias = f"Drug{index}"
    value = 10.0 + index
    sentence = (
        f"{alias} inhibits FLT3 with an experimentally measured "
        f"IC50 of {value:g} nM."
    )
    return {
        "candidate_id": candidate_id,
        "candidate_aliases": [alias],
        "sentence": sentence,
        "sentence_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
        "document_id_sha256": hashlib.sha256(f"paper-{index}".encode()).hexdigest(),
        "source_location": f"abstract.results.sentence[{index}]",
        "target_symbol": "FLT3",
        "predicate": "inhibits",
        "direction": "inhibit",
        "endpoint": "IC50", "relation": "=", "value": value, "unit": "nM",
        "primary_source_attestation_sha256": hashlib.sha256(f"attest-{index}".encode()).hexdigest(),
        "publication_type_screen_passed": True, "publication_types": ["Journal Article"],
        "verification_status": "verified",
        "is_primary_support": True,
        "is_review": False,
        "fixture": False,
    }


def _support():
    return {candidate_id: _unit(candidate_id, i) for i, candidate_id in enumerate(IDS)}


def _aggregate(support=None, *, requested=3):
    return aggregate_candidate_primary_support_v6(
        fixed_candidate_ids=IDS, support_by_candidate=support or _support(),
        expected_target_symbol="FLT3", expected_endpoint="IC50",
        expected_direction="inhibit", requested_eligible_count=requested,
    )


def test_three_candidates_may_use_three_independent_primary_records():
    result = _aggregate()
    assert result["status"] == "verified"
    assert result["policy"] == PAPER_NATIVE_V6_POLICY_TOKEN
    assert result["candidate_count"] == 3
    assert result["supported_candidate_count"] == 12
    assert result["candidate_ids"] == IDS[:3]
    assert result["not_promoted_candidate_count"] == 9
    assert result["no_reordering"] is result["no_backfill"] is True


def test_html_subscript_endpoint_whitespace_is_equivalent_not_inferred():
    support = _support()
    for index, candidate_id in enumerate(IDS, start=1):
        value = 10.0 + index - 1
        sentence = (
            f"Drug{index - 1} inhibits FLT3 with an experimentally measured "
            f"IC 50 of {value:g} nM."
        )
        support[candidate_id]["sentence"] = sentence
        support[candidate_id]["sentence_sha256"] = hashlib.sha256(
            sentence.encode()
        ).hexdigest()

    result = _aggregate(support)

    assert result["candidate_ids"] == IDS[:3]
    assert all(
        unit["endpoint"] == "IC50"
        for unit in result["candidate_support_units"]
    )


@pytest.mark.parametrize("mutation", [
    "reorder", "outside", "wrong_candidate", "wrong_sentence_hash", "alias_absent",
    "secondary", "review", "fixture", "wrong_direction", "missing_location",
    "endpoint", "value", "unit", "publication_type_screen",
])
def test_atomic_or_aggregation_mutations_fail_closed(mutation):
    support = _support()
    fixed = list(IDS)
    if mutation == "reorder":
        support = {IDS[1]: support[IDS[1]], IDS[0]: support[IDS[0]], IDS[2]: support[IDS[2]]}
    elif mutation == "outside": support = {**support, "CHEMBL999": _unit("CHEMBL999", 99)}
    elif mutation == "wrong_candidate": support[IDS[0]]["candidate_id"] = IDS[1]
    elif mutation == "wrong_sentence_hash": support[IDS[0]]["sentence_sha256"] = "0" * 64
    elif mutation == "alias_absent": support[IDS[0]]["candidate_aliases"] = ["AbsentDrug"]
    elif mutation == "secondary": support[IDS[0]]["is_primary_support"] = False
    elif mutation == "review": support[IDS[0]]["is_review"] = True
    elif mutation == "fixture": support[IDS[0]]["fixture"] = True
    elif mutation == "wrong_direction": support[IDS[0]]["direction"] = "activate"
    elif mutation == "missing_location": support[IDS[0]]["source_location"] = ""
    elif mutation == "endpoint": support[IDS[0]]["endpoint"] = "Ki"
    elif mutation == "value": support[IDS[0]]["value"] = "10"
    elif mutation == "unit": support[IDS[0]]["unit"] = "mg/mL"
    elif mutation == "publication_type_screen": support[IDS[0]]["publication_type_screen_passed"] = False
    with pytest.raises(PaperNativeV6ContractError):
        _aggregate(support)


def test_less_than_three_never_promotes():
    with pytest.raises(PaperNativeV6ContractError, match="INSUFFICIENT_VERIFIED_SUPPORT"):
        _aggregate({key: _support()[key] for key in IDS[:2]})


def test_sentence_without_exact_value_and_unit_never_aggregates():
    support = _support()
    sentence = "Drug0 inhibits FLT3 with an experimentally measured IC50."
    support[IDS[0]]["sentence"] = sentence
    support[IDS[0]]["sentence_sha256"] = hashlib.sha256(
        sentence.encode()
    ).hexdigest()
    with pytest.raises(PaperNativeV6ContractError, match="ATOMIC_SUPPORT_INVALID"):
        _aggregate(support)


@pytest.mark.parametrize(
    ("mutation", "replacement"),
    [
        ("target", ("FLT3", "KIT")),
        ("endpoint", ("IC50", "Ki")),
        ("value", ("10 nM", "11 nM")),
        ("unit", ("10 nM", "10 uM")),
        ("relation", ("10 nM", ">10 nM")),
    ],
)
def test_atomic_sentence_binding_single_field_mutations_fail(mutation, replacement):
    support = _support()
    sentence = support[IDS[0]]["sentence"].replace(*replacement)
    assert sentence != support[IDS[0]]["sentence"], mutation
    support[IDS[0]]["sentence"] = sentence
    support[IDS[0]]["sentence_sha256"] = hashlib.sha256(
        sentence.encode()
    ).hexdigest()
    with pytest.raises(PaperNativeV6ContractError, match="ATOMIC_SUPPORT_INVALID"):
        _aggregate(support)


@pytest.mark.parametrize(
    ("sentence", "value", "unit", "relation", "expected"),
    [
        ("IC50 of 10 nM", 10.0, "nM", "=", True),
        ("IC50 of >10 nM", 10.0, "nM", ">", True),
        ("IC50 of >10 nM", 10.0, "nM", "=", False),
        ("IC50 of 1e-3 nM", 0.001, "nM", "=", True),
        ("IC50 of 10 µM", 10.0, "µM", "=", True),
        ("IC50 of 10 uM", 10.0, "µM", "=", False),
    ],
)
def test_v6_producer_and_ma3_exact_value_unit_semantics_match(
    sentence, value, unit, relation, expected
):
    assert _exact_value_unit_present(
        sentence, value=value, unit=unit, relation=relation
    ) is expected
    assert _paper_v6_exact_value_unit_present(
        sentence, value, unit, relation
    ) is expected


@pytest.mark.parametrize(
    "unsafe_sentence",
    [
        "approximately 10 nM",
        "about 10 nM",
        "~10 nM",
        "≈10 nM",
        "≤10 nM",
        "≥10 nM",
        "-10 nM",
        "5-10 nM",
        "5 to 10 nM",
        "ranged from 5 to 10 nM",
        "between 5 and 10 nM",
        "up to 10 nM",
        "at most 10 nM",
        "at least 10 nM",
        "more than 10 nM",
        "above 10 nM",
        "below 10 nM",
        "under 10 nM",
        "over 10 nM",
    ],
)
def test_four_exact_matchers_reject_unsafe_or_non_ascii_numeric_semantics(
    unsafe_sentence,
):
    clause = f"Drug0 inhibits FLT3 with an IC50 of {unsafe_sentence}."
    activity = ActivityMeasurement(
        endpoint="IC50",
        relation="=",
        value=10.0,
        unit="nM",
        assay_type="B",
        source_evidence_id="ev-unsafe",
    )
    assert not _exact_value_unit_present(
        clause, value=10.0, unit="nM", relation="="
    )
    assert not _paper_v6_exact_value_unit_present(
        clause, 10.0, "nM", "="
    )
    assert _exact_activity_value_spans(clause, activity) == []
    assert _primary_exact_activity_value_spans(clause, activity) == []
    assert not _paper_v6_clause_is_unambiguous(
        clause,
        aliases=["Drug0"],
        primary={
            "competitor_alias_provenance": {
                "policy": (
                    "all_bounded_pool_candidate_aliases_except_current_v1"
                ),
                "competitor_exact_aliases": [],
                "truncated": False,
            }
        },
        target_symbol="FLT3",
        endpoint="IC50",
        relation="=",
        value=10.0,
        unit="nM",
    )


@pytest.mark.parametrize(
    "unsafe_clause",
    [
        "Drug0 did not inhibit FLT3 with an IC50 of 10 nM.",
        "Drug0 failed to inhibit FLT3 with an IC50 of 10 nM.",
        "Drug0 is predicted to inhibit FLT3 with an IC50 of 10 nM.",
        "Drug0 was simulated to inhibit FLT3 with an IC50 of 10 nM.",
        "Drug0 may inhibit FLT3 with an IC50 of 10 nM.",
        "Drug0 could inhibit FLT3 with an IC50 of 10 nM.",
        "Drug0 was previously reported to inhibit FLT3 with an IC50 of 10 nM.",
        "As reported, Drug0 inhibits FLT3 with an IC50 of 10 nM.",
    ],
)
def test_live_and_independent_ma3_binders_reject_nonexperimental_clauses(
    unsafe_clause,
):
    target = TargetCandidate(symbol="FLT3", uniprot_id="P36888")
    activity = ActivityMeasurement(
        endpoint="IC50", relation="=", value=10.0, unit="nM",
        assay_type="B", source_evidence_id="ev-unsafe",
    )
    kwargs = {
        "candidate_aliases": ["Drug0"],
        "competitor_aliases": [],
        "target": target,
    }
    assert _live_primary_clause_binding(
        unsafe_clause, measurement=activity, **kwargs
    ) is None
    assert _ma3_primary_clause_binding(
        unsafe_clause, activity=activity, **kwargs
    ) is None
    assert not _exact_value_unit_present(
        unsafe_clause, value=10.0, unit="nM", relation="="
    )
    assert not _paper_v6_exact_value_unit_present(
        unsafe_clause, 10.0, "nM", "="
    )
    assert not _paper_v6_clause_is_unambiguous(
        unsafe_clause,
        aliases=["Drug0"],
        primary={
            "competitor_alias_provenance": {
                "policy": (
                    "all_bounded_pool_candidate_aliases_except_current_v1"
                ),
                "competitor_exact_aliases": [],
                "truncated": False,
            }
        },
        target_symbol="FLT3",
        endpoint="IC50",
        relation="=",
        value=10.0,
        unit="nM",
    )


def test_live_and_independent_ma3_binders_accept_measured_exact_clause():
    clause = "Drug0 inhibits FLT3 with an IC50 of 10 nM."
    target = TargetCandidate(symbol="FLT3", uniprot_id="P36888")
    activity = ActivityMeasurement(
        endpoint="IC50", relation="=", value=10.0, unit="nM",
        assay_type="B", source_evidence_id="ev-exact",
    )
    kwargs = {
        "candidate_aliases": ["Drug0"],
        "competitor_aliases": [],
        "target": target,
    }
    assert _live_primary_clause_binding(
        clause, measurement=activity, **kwargs
    ) is not None
    assert _ma3_primary_clause_binding(
        clause, activity=activity, **kwargs
    ) is not None


@pytest.mark.parametrize("excluded_type", [
    "Expression of Concern",
    "Retraction Notice",
    "Retraction of Publication",
    "Protocol",
    "Clinical Trial Protocol",
])
def test_excluded_pubmed_type_never_enters_v6_aggregation(excluded_type):
    support = _support()
    support[IDS[0]]["publication_types"] = ["Journal Article", excluded_type]
    with pytest.raises(PaperNativeV6ContractError, match="ATOMIC_SUPPORT_INVALID"):
        _aggregate(support)


def test_missing_early_pool_members_is_allowed_but_order_is_fixed():
    support = _support()
    chosen = [IDS[2], IDS[5], IDS[7], IDS[9]]
    result = _aggregate({key: support[key] for key in chosen})
    assert result["supported_candidate_count"] == 4
    assert result["candidate_ids"] == chosen[:3]


def test_candidate_duplicates_and_forged_unit_hash_fail():
    with pytest.raises(PaperNativeV6ContractError, match="DUPLICATE"):
        aggregate_candidate_primary_support_v6(
            fixed_candidate_ids=[IDS[0], IDS[0], *IDS[2:]], support_by_candidate=_support(),
            expected_target_symbol="FLT3", expected_endpoint="IC50",
            expected_direction="inhibit", requested_eligible_count=3)
    support = _support()
    support[IDS[0]]["support_unit_sha256"] = "0" * 64
    with pytest.raises(PaperNativeV6ContractError, match="HASH_MISMATCH"):
        _aggregate(support)


def test_hashes_are_exact_and_deterministic():
    first = _aggregate()
    second = _aggregate(deepcopy(_support()))
    assert first == second
    changed = _support()
    sentence = "Drug0 directly inhibits FLT3 with an IC50 of 10 nM."
    changed[IDS[0]]["sentence"] = sentence
    changed[IDS[0]]["sentence_sha256"] = hashlib.sha256(sentence.encode()).hexdigest()
    third = _aggregate(changed)
    assert third["group_sha256"] != first["group_sha256"]
