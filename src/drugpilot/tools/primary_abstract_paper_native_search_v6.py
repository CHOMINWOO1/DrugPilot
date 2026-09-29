"""Deterministic candidate-per-primary-record aggregation for paper-native v6.

Unlike v5, a promotion group does not require all candidates to occur in one
sentence or publication.  Upstream retrieval/parser code supplies independently
verified, bounded primary-record sentences; this module validates their atomic
binding and aggregates them without reordering or backfilling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any
import unicodedata

PAPER_NATIVE_V6_SEARCH_SCHEMA_VERSION = "primary_abstract_paper_native_search/v6"
PAPER_NATIVE_V6_POLICY_TOKEN = (
    "dynamic_pubmed_primary_abstract_paper_native_v6_required"
)
PAPER_NATIVE_V6_AGGREGATION_POLICY = (
    "candidate_per_primary_record_atomic_support_then_fixed_order_aggregation_v1"
)
MINIMUM_VERIFIED_CANDIDATES = 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_SENTENCE_CHARS = 360
_ALLOWED_RELATIONS = frozenset({"=", "<", ">", "<=", ">="})
_ALLOWED_UNITS = frozenset({"pM", "nM", "uM", "µM", "μM", "mM", "M"})
_REQUIRED_PUBLICATION_TYPE = "journal article"
PAPER_NATIVE_V6_EXCLUDED_PUBLICATION_TYPES = frozenset({
    "review", "systematic review", "meta analysis", "guideline",
    "practice guideline", "editorial", "letter", "comment", "news",
    "interview", "biography", "historical article",
    "patient education handout", "consensus development conference",
    "consensus development conference nih", "retracted publication",
    "published erratum", "expression of concern", "retraction notice",
    "retraction of publication", "protocol", "clinical trial protocol",
})
_UNSAFE_PRIMARY_CLAUSE = re.compile(
    r"\b(?:previously|formerly|earlier|prior)\s+(?:was\s+|were\s+)?reported\b|"
    r"\b(?:was|were)\s+(?:previously|formerly|earlier)\s+reported\b|"
    r"\b(?:according\s+to|as\s+reported|previous\s+stud(?:y|ies)|prior\s+work)\b|"
    r"\b(?:may|might|could|possibly|potentially|apparently|appears?|suggests?|suggested)\b|"
    # Experimental model nouns and a terminal non-cytotoxicity qualifier do
    # not negate an otherwise independently verified quantitative result.
    r"\b(?:no|not|neither|without(?!\s+(?:(?:detectable|significant|measurable|overt)\s+)?"
    r"(?:cytotoxicity|toxicity)\s*(?:[.,;:!?)]|$))|failed\s+to|fails\s+to|did\s+not)\b|"
    r"\b(?:predict(?:ed|ing|ion|ions|ive|ively)?|"
    r"(?<!cell )(?<!cellular )(?<!animal )(?<!murine )(?<!xenograft )(?<!organoid )(?<!vitro )(?<!vivo )models?|"
    r"model(?:ed|ing|led|ling)|"
    r"simulat(?:e|es|ed|ing|ion|ions|ive|ively)|"
    r"calculat(?:e|es|ed|ing|ion|ions)|computational(?:ly)?|"
    r"in[\s-]+silico|prospective(?:ly)?|hypothetical(?:ly)?|putative(?:ly)?|"
    r"project(?:ed|ing|ion|ions)|estimat(?:e|es|ed|ing|ion|ions)|"
    r"forecast(?:s|ed|ing)?|anticipated)\b|"
    r"\b(?:planned|expected|proposed|anticipated|aimed)\s+to\s+be\b|"
    r"\b(?:plans?|proposes?|aims?)\s+to\s+be\b|"
    r"\b(?:will|would)(?:\s+\w+){0,2}\s+(?:be|been)\b|"
    r"\b(?:about|approximate(?:ly)?|approx\.?|roughly|around|circa)\s*(?=\d|\.\d)|"
    r"(?:±|\+/-|~|≈|≤|≥)|(?<![\w.])[+-]\s*(?=\d)|"
    r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
    r"(?:[-–]|\bto\b)\s*\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])|"
    r"\b(?:range[sd]?|ranging)\s+from\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b|"
    r"\bbetween\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?\b|"
    r"\b(?:versus|vs\.?|compared\s+(?:with|to)|relative\s+to|"
    r"greater\s+than|less\s+than|higher\s+than|lower\s+than)\b",
    flags=re.IGNORECASE,
)
_UNSAFE_NUMERIC_WORD_PREFIX = re.compile(
    r"(?<!\w)(?:about|approximate(?:ly)?|approx\.?|roughly|around|circa|"
    r"up\s+to|at\s+(?:most|least)|more\s+than|less\s+than|"
    r"greater\s+than|higher\s+than|lower\s+than|above|below|under|over)\s*$",
    flags=re.IGNORECASE,
)
_UNSAFE_NUMERIC_SYMBOL_PREFIX = re.compile(
    r"(?:±|\+/-|~|≈|≤|≥|(?<![\w.])[+-])\s*$"
)


class PaperNativeV6ContractError(ValueError):
    """Raised when independently supplied evidence is not atomically verifiable."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_strings(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PaperNativeV6ContractError(f"{label}_NOT_SEQUENCE")
    rows = list(value)
    if not rows or any(not isinstance(row, str) or not row.strip() for row in rows):
        raise PaperNativeV6ContractError(f"{label}_INVALID")
    if len(rows) != len(set(rows)):
        raise PaperNativeV6ContractError(f"{label}_DUPLICATE")
    return rows


def _phrase_present(sentence: str, alias: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", alias.casefold())
    if not words:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\s\-_]+".join(map(re.escape, words)) + r"(?![A-Za-z0-9])"
    return re.search(pattern, sentence.casefold()) is not None


def _endpoint_present(sentence: str, endpoint: str) -> bool:
    """Match endpoint labels across harmless HTML-subscript whitespace.

    ChEMBL abstracts can encode ``IC<sub>50</sub>``. The bounded text
    normalizer preserves that source as ``IC 50`` while ordinary abstracts
    use ``IC50``. These are the same endpoint label; accepting the optional
    separator does not alter candidate, target, relation, value, or unit
    binding.
    """

    chunks = re.findall(r"[A-Za-z]+|[0-9]+", endpoint.casefold())
    if not chunks:
        return False
    pattern = (
        r"(?<![A-Za-z0-9])"
        + r"[\s\-_]*".join(map(re.escape, chunks))
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, sentence.casefold()) is not None


def _publication_type_key(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def publication_types_valid_v6(value: Any) -> bool:
    """Validate the exact normalized PubMed type list used by V6 producers."""

    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) for item in value
    ):
        return False
    normalized_items = [" ".join(item.split()) for item in value]
    normalized_keys = [_publication_type_key(item) for item in normalized_items]
    if (
        normalized_items != value
        or any(not key for key in normalized_keys)
        or len(normalized_keys) != len(set(normalized_keys))
    ):
        return False
    keys = set(normalized_keys)
    return (
        _REQUIRED_PUBLICATION_TYPE in keys
        and not keys.intersection(PAPER_NATIVE_V6_EXCLUDED_PUBLICATION_TYPES)
    )


def _clause_semantically_unsafe(text: str) -> bool:
    return _UNSAFE_PRIMARY_CLAUSE.search(
        unicodedata.normalize("NFKC", text)
    ) is not None


def _numeric_match_has_unsafe_prefix(
    text: str, match: re.Match[str], *, number_group: str,
) -> bool:
    before_match = unicodedata.normalize("NFKC", text[:match.start()])
    before_number = unicodedata.normalize(
        "NFKC", text[:match.start(number_group)]
    )
    return bool(
        _UNSAFE_PRIMARY_CLAUSE.search(unicodedata.normalize("NFKC", text))
        or _UNSAFE_NUMERIC_WORD_PREFIX.search(before_match)
        or _UNSAFE_NUMERIC_SYMBOL_PREFIX.search(before_number)
    )


def _exact_value_unit_present(
    sentence: str, *, value: Any, unit: Any, relation: Any,
) -> bool:
    if isinstance(value, bool) or not isinstance(unit, str):
        return False
    try:
        expected = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    for match in re.finditer(
        r"(?<![A-Za-z0-9.])(?P<relation><=|>=|<|>|=)?\s*"
        r"(?P<value>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
        r"(?P<unit>pM|nM|uM|µM|μM|mM|M)(?![A-Za-z0-9])",
        sentence,
    ):
        if _numeric_match_has_unsafe_prefix(
            sentence, match, number_group="value"
        ):
            continue
        try:
            same_value = Decimal(match.group("value")) == expected
        except InvalidOperation:
            same_value = False
        observed_unit = match.group("unit")
        same_unit = observed_unit == unit
        if (
            same_value
            and same_unit
            and (match.group("relation") or "=") == relation
        ):
            return True
    return False


def _atomic_unit(candidate_id: str, raw: Any, *, expected_target_symbol: str,
                 expected_endpoint: str, expected_direction: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise PaperNativeV6ContractError("SUPPORT_UNIT_NOT_MAPPING")
    sentence = raw.get("sentence")
    aliases = _exact_strings(raw.get("candidate_aliases"), label="CANDIDATE_ALIASES")
    if (
        raw.get("candidate_id") != candidate_id
        or not isinstance(sentence, str)
        or sentence != sentence.strip()
        or not sentence
        or len(sentence) > _MAX_SENTENCE_CHARS
        or _clause_semantically_unsafe(sentence)
        or raw.get("sentence_sha256") != _text_sha256(sentence)
        or not any(_phrase_present(sentence, alias) for alias in aliases)
        or raw.get("verification_status") != "verified"
        or raw.get("is_primary_support") is not True
        or raw.get("is_review") is not False
        or raw.get("fixture") is not False
        or raw.get("predicate") not in {"inhibits", "inhibition"}
        or raw.get("direction") != expected_direction
        or raw.get("target_symbol") != expected_target_symbol
        or raw.get("endpoint") != expected_endpoint
        or raw.get("relation") not in _ALLOWED_RELATIONS
        or type(raw.get("value")) not in {int, float}
        or not (0 < raw["value"] < 1_000_000_000)
        or raw.get("unit") not in _ALLOWED_UNITS
        or not _phrase_present(sentence, expected_target_symbol)
        or not _endpoint_present(sentence, expected_endpoint)
        or not _exact_value_unit_present(
            sentence,
            value=raw.get("value"),
            unit=raw.get("unit"),
            relation=raw.get("relation"),
        )
        or not isinstance(raw.get("primary_source_attestation_sha256"), str)
        or not _SHA256_RE.fullmatch(raw["primary_source_attestation_sha256"])
        or raw.get("publication_type_screen_passed") is not True
        or not publication_types_valid_v6(raw.get("publication_types"))
        or not isinstance(raw.get("document_id_sha256"), str)
        or not _SHA256_RE.fullmatch(raw["document_id_sha256"])
        or not isinstance(raw.get("source_location"), str)
        or not raw.get("source_location")
    ):
        raise PaperNativeV6ContractError("ATOMIC_SUPPORT_INVALID")
    projection = {
        "candidate_id": candidate_id,
        "candidate_aliases": aliases,
        "direction": expected_direction,
        "document_id_sha256": raw["document_id_sha256"],
        "fixture": False,
        "is_primary_support": True,
        "is_review": False,
        "predicate": raw["predicate"],
        "endpoint": expected_endpoint,
        "relation": raw["relation"],
        "value": raw["value"],
        "unit": raw["unit"],
        "primary_source_attestation_sha256": raw[
            "primary_source_attestation_sha256"
        ],
        "publication_types": list(raw["publication_types"]),
        "publication_type_screen_passed": True,
        "sentence": sentence,
        "sentence_sha256": raw["sentence_sha256"],
        "source_location": raw["source_location"],
        "target_symbol": raw["target_symbol"],
        "verification_status": "verified",
    }
    projection["support_unit_sha256"] = _canonical_sha256(projection)
    supplied = raw.get("support_unit_sha256")
    if supplied is not None and supplied != projection["support_unit_sha256"]:
        raise PaperNativeV6ContractError("SUPPORT_UNIT_HASH_MISMATCH")
    return projection


def aggregate_candidate_primary_support_v6(
    *, fixed_candidate_ids: Sequence[str], support_by_candidate: Mapping[str, Any],
    expected_target_symbol: str, expected_endpoint: str,
    expected_direction: str, requested_eligible_count: int,
) -> dict[str, Any]:
    """Validate and aggregate independent candidate support in fixed-pool order.

    Missing candidates are recorded as non-promoted. Submitted units never get
    skipped: every supplied unit must validate and remain an ordered subsequence.
    """

    fixed = _exact_strings(fixed_candidate_ids, label="FIXED_CANDIDATE_IDS")
    if len(fixed) < MINIMUM_VERIFIED_CANDIDATES:
        raise PaperNativeV6ContractError("INSUFFICIENT_FIXED_POOL")
    if (not isinstance(support_by_candidate, Mapping)
            or type(requested_eligible_count) is not int
            or requested_eligible_count < MINIMUM_VERIFIED_CANDIDATES
            or requested_eligible_count > len(fixed)
            or not all(isinstance(x, str) and x for x in (
                expected_target_symbol, expected_endpoint, expected_direction))):
        raise PaperNativeV6ContractError("SUPPORT_MAP_NOT_MAPPING")
    supported_ids = list(support_by_candidate)
    if len(supported_ids) < MINIMUM_VERIFIED_CANDIDATES:
        raise PaperNativeV6ContractError("INSUFFICIENT_VERIFIED_SUPPORT")
    expected_subsequence = [candidate_id for candidate_id in fixed if candidate_id in support_by_candidate]
    if supported_ids != expected_subsequence:
        raise PaperNativeV6ContractError("SUPPORT_MAP_ORDER_OR_MEMBERSHIP_MISMATCH")
    all_units = [_atomic_unit(
        candidate_id, support_by_candidate[candidate_id],
        expected_target_symbol=expected_target_symbol,
        expected_endpoint=expected_endpoint,
        expected_direction=expected_direction,
    ) for candidate_id in supported_ids]
    selected_ids = supported_ids[:requested_eligible_count]
    units = all_units[:requested_eligible_count]
    rejected_ids = [candidate_id for candidate_id in fixed if candidate_id not in selected_ids]
    all_unit_hashes = [unit["support_unit_sha256"] for unit in all_units]
    unit_hashes = all_unit_hashes[:requested_eligible_count]
    if len(all_unit_hashes) != len(set(all_unit_hashes)):
        raise PaperNativeV6ContractError("DUPLICATE_SUPPORT_UNIT")
    group_projection = {
        "candidate_ids": selected_ids,
        "member_support_unit_sha256": unit_hashes,
        "minimum_verified_candidates": MINIMUM_VERIFIED_CANDIDATES,
        "no_backfill": True,
        "no_reordering": True,
        "policy": PAPER_NATIVE_V6_POLICY_TOKEN,
        "aggregation_policy": PAPER_NATIVE_V6_AGGREGATION_POLICY,
    }
    return {
        "schema_version": PAPER_NATIVE_V6_SEARCH_SCHEMA_VERSION,
        "policy": PAPER_NATIVE_V6_POLICY_TOKEN,
        "aggregation_policy": PAPER_NATIVE_V6_AGGREGATION_POLICY,
        "status": "verified",
        "candidate_support_units": units,
        "candidate_ids": selected_ids,
        "candidate_count": len(selected_ids),
        "supported_candidate_count": len(supported_ids),
        "supported_candidate_ids_order_sha256": _canonical_sha256(supported_ids),
        "not_promoted_candidate_count": len(rejected_ids),
        "not_promoted_candidate_ids_order_sha256": _canonical_sha256(rejected_ids),
        "candidate_ids_order_sha256": _canonical_sha256(selected_ids),
        "candidate_ids_set_sha256": _canonical_sha256(sorted(selected_ids)),
        "group_sha256": _canonical_sha256(group_projection),
        "no_reordering": True,
        "no_backfill": True,
    }


__all__ = [
    "MINIMUM_VERIFIED_CANDIDATES",
    "PAPER_NATIVE_V6_AGGREGATION_POLICY",
    "PAPER_NATIVE_V6_EXCLUDED_PUBLICATION_TYPES",
    "PAPER_NATIVE_V6_POLICY_TOKEN",
    "PAPER_NATIVE_V6_SEARCH_SCHEMA_VERSION",
    "PaperNativeV6ContractError",
    "aggregate_candidate_primary_support_v6",
    "publication_types_valid_v6",
]
