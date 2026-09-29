"""Versioned fail-closed parser for paper-native comparative measurements.

The module intentionally has no network, database, or agent dependencies.  It
recognises one narrow evidence grammar: a common assay head for the first
candidate followed by direct ``candidate (value unit)`` pairs.  Anything that
cannot be accounted for exactly is rejected as an empty tuple.

Version 5 accepts either a bare endpoint token or one endpoint token wrapped
exactly once in balanced square brackets (for example ``[IC50]``).  Both the
inner endpoint and the optional wrapper retain their exact source spans.  No
text repair, endpoint expansion, value imputation, or numeric normalization is
performed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Literal
import unicodedata

__all__ = [
    "AssayCondition",
    "PAPER_NATIVE_V5_GRAMMAR",
    "PAPER_NATIVE_V5_PARSER_SCHEMA_VERSION",
    "PrimaryAbstractPair",
    "SourceSpan",
    "parse_paper_native_primary_abstract_v5",
]


PAPER_NATIVE_V5_PARSER_SCHEMA_VERSION = "primary_abstract_paper_native/v5"
PAPER_NATIVE_V5_GRAMMAR = "head_measurement_followed_by_candidate_value_list_v1"
_MAX_DELIMITER_DEPTH = 4
_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = r"(?:pM|nM|uM|µM|μM|mM|M)"

_CONNECTOR_RE = re.compile(r",\s+followed\s+by\s+", re.IGNORECASE)
_FOLLOWED_BY_RE = re.compile(r"\bfollowed\s+by\b", re.IGNORECASE)
_HEAD_MEASUREMENT_RE = re.compile(
    rf",\s*(?P<relation>=)?\s*(?P<value>{_NUMBER})\s*"
    rf"(?P<unit>{_UNIT})\s*\)\s*$"
)
_TAIL_MEASUREMENT_RE = re.compile(
    rf"\s*\(\s*(?P<relation>=)?\s*(?P<value>{_NUMBER})\s*"
    rf"(?P<unit>{_UNIT})\s*\)"
)
_BRACKETED_CONDITION_RE = re.compile(
    rf"\bat\s+(?:(?:[A-Za-z][A-Za-z-]*)\s+){{0,4}}"
    rf"\[\s*(?P<value>{_NUMBER})\s*(?P<unit>(?-i:{_UNIT}))\s*\]"
    r"\s+(?P<subject>[A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)
_UNBRACKETED_CONDITION_RE = re.compile(
    rf"\bat\s+(?:(?:[A-Za-z][A-Za-z-]*)\s+){{0,4}}"
    rf"(?P<value>{_NUMBER})\s*(?P<unit>(?-i:{_UNIT}))"
    r"\s+(?P<subject>[A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(rf"(?P<value>{_NUMBER})")

_ENDPOINT_MARKER_RE = re.compile(
    r"(?<![\w-])(?:p?IC50|p?EC50|GI50|LC50|CC50|AC50|ED50|"
    r"p?K[di]|pSTAT\d+)(?![\w-])",
    re.IGNORECASE,
)
_PRIMARY_PUBLICATION_TYPES = frozenset(
    {
        "journal article",
        "research article",
        "original article",
        "clinical study",
        "clinical trial",
        "comparative study",
        "evaluation study",
        "observational study",
        "multicenter study",
        "randomized controlled trial",
    }
)
_NON_PRIMARY_PUBLICATION_RE = re.compile(
    r"(?:review|meta-analysis|editorial|letter|comment|guideline|news|"
    r"retracted publication|published erratum)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"(?:\[\s*\d+(?:\s*[-,–]\s*\d+)*\s*\]|"
    r"\b(?:PMID|PMCID|doi)\s*:?\s*\S+|"
    r"\([^()]*\bet\s+al\.[^()]*\b(?:19|20)\d{2}[^()]*\))",
    re.IGNORECASE,
)
_PRIOR_REPORT_RE = re.compile(
    r"\b(?:previously|formerly|earlier|prior)\s+(?:was\s+|were\s+)?reported\b|"
    r"\breported\s+(?:previously|earlier)\b|"
    r"\b(?:according\s+to|as\s+reported|previous\s+stud(?:y|ies)|prior\s+work)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"\b(?:may|might|could|possibly|potentially|apparently|appears?|"
    r"suggests?|suggested)\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:no|not|neither|without|failed\s+to|fails\s+to|did\s+not)\b",
    re.IGNORECASE,
)
_APPROXIMATE_RE = re.compile(
    r"\b(?:about|approximately|approx\.?|roughly|around|circa)\s*(?=\d|\.\d)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    rf"(?:{_NUMBER})\s*(?:-|–|—|to)\s*(?:{_NUMBER})|\b(?:range|ranged)\b",
    re.IGNORECASE,
)
_RELATIONAL_SYMBOL_RE = re.compile(r"(?:±|\+/-|~|≈|≤|≥|<|>)")
_VALUE_COMPARATOR_RE = re.compile(
    r"\b(?:versus|vs\.?|compared\s+(?:with|to)|relative\s+to|"
    r"greater\s+than|less\s+than|higher\s+than|lower\s+than)\b",
    re.IGNORECASE,
)
_NONEXPERIMENTAL_LANGUAGE_RE = re.compile(
    r"\b(?:predicted|predictions?|in(?:\s+|-)silico|computational(?:ly)?|"
    r"model(?:ed|led|ing|ling)|simulat(?:ed|ing|ions?)|"
    r"project(?:ed|ing|ions?)|estimat(?:ed|ing|ions?)|"
    r"forecast(?:s|ed|ing)?|anticipat(?:ed|ing|ions?)|aimed|plans?|"
    r"proposes?|docking[-\s]derived|hypothetical|putative|"
    r"prospective(?:ly)?)\b|"
    r"\b(?:planned|expected|proposed)\s+to\s+be\b|"
    r"\bwill\s+(?:likely\s+)?be\b|\bwill\s+likely\b|"
    r"\bwould\s+(?:likely\s+)?(?:be|have\s+been)\b",
    re.IGNORECASE,
)
_OBSERVED_POTENCY_PREFIX_RE = re.compile(
    r"^\s+(?:was|is)\s+(?:"
    r"(?:(?:experimentally|directly)\s+)?|"
    r"(?:experimentally|directly),\s+as\s+expected,\s+|"
    r"(?:(?:experimentally|directly)\s+)?"
    r"(?:observed|measured|determined)\s+(?:to\s+be|as)\s+"
    r")(?:(?:the|a|an)\s+)?(?:most\s+)?potent\s+$",
    re.IGNORECASE,
)
_OBSERVED_TO_INHIBIT_PREFIX_RE = re.compile(
    r"^\s+(?:was|is)\s+(?:(?:experimentally|directly)\s+)?"
    r"(?:observed|measured|determined)\s+to\s+inhibit\s+$",
    re.IGNORECASE,
)
_OBSERVED_ACTIVE_INHIBITION_PREFIX_RE = re.compile(
    r"^\s+(?:(?:experimentally|directly)\s+)?(?:inhibited|inhibits)\s+$",
    re.IGNORECASE,
)
_INHIBITOR_SUFFIX_RE = re.compile(r"^\s+inhibitor\s*$", re.IGNORECASE)
_EMPTY_TARGET_SUFFIX_RE = re.compile(r"^\s*$")
_TAIL_SUFFIX_RE = re.compile(
    r"(?:\.|,\s+and\s+the\s+most\s+selective\.)?", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """An immutable half-open span into the unmodified input sentence."""

    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class AssayCondition:
    """The one common assay condition shared by every returned pair."""

    span: SourceSpan
    value_raw: str
    value_decimal: Decimal
    value_span: SourceSpan
    unit_raw: str
    unit_span: SourceSpan
    subject: str
    subject_span: SourceSpan


@dataclass(frozen=True, slots=True)
class PrimaryAbstractPair:
    """One directly attested candidate measurement from the accepted sentence.

    ``relation_raw`` is ``None`` unless the source literally contains ``=``.
    A bare scalar is retained as a reported point estimate; it is never
    relabelled as literal equality.
    """

    sentence_span: SourceSpan
    sentence_sha256: str
    evidence_sha256: str
    grammar: Literal["head_measurement_followed_by_candidate_value_list_v1"]
    pair_index: int
    pair_role: Literal["common_head", "direct_follower"]
    candidate_id: str
    candidate_alias: str
    candidate_span: SourceSpan
    target_alias: str
    target_span: SourceSpan
    endpoint: str
    endpoint_span: SourceSpan
    endpoint_wrapper_raw: str | None
    endpoint_wrapper_span: SourceSpan | None
    assay_condition: AssayCondition
    measurement_span: SourceSpan
    value_raw: str
    value_decimal: Decimal
    value_span: SourceSpan
    unit_raw: str
    unit_span: SourceSpan
    relation_raw: str | None
    relation_span: SourceSpan | None
    relation_normalized: Literal["="] | None
    relation_basis: Literal["literal_equality", "reported_point_estimate"]
    literal_equality: bool
    target_scope_basis: Literal[
        "explicit_common_head", "inherited_from_common_head"
    ]
    endpoint_scope_basis: Literal[
        "explicit_common_head", "inherited_from_common_head"
    ]
    assay_condition_scope_basis: Literal[
        "explicit_common_head", "inherited_from_common_head"
    ]
    section: Literal["Abstract", "Results"]
    publication_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Occurrence:
    owner: str
    configured_alias: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Measurement:
    full_end: int
    measurement_span: SourceSpan
    value_raw: str
    value_decimal: Decimal
    value_span: SourceSpan
    unit_raw: str
    unit_span: SourceSpan
    relation_raw: str | None
    relation_span: SourceSpan | None


class _Reject(ValueError):
    """Internal sentinel for an ordinary fail-closed rejection."""


def parse_paper_native_primary_abstract_v5(
    sentence: str,
    *,
    candidate_aliases: Mapping[str, Collection[str]],
    target_aliases: Collection[str],
    endpoint: str,
    section: str,
    publication_types: Collection[str] | str,
) -> tuple[PrimaryAbstractPair, ...]:
    """Parse one narrow paper-native comparative measurement sentence.

    Inputs are never repaired or expanded.  The returned tuple is in source
    order and all nested records are frozen.  Any ambiguity, unsupported
    grammar, invalid metadata, or unconsumed number rejects the whole sentence
    and returns ``()``; partial results are never returned.
    """

    try:
        return _parse(
            sentence,
            candidate_aliases=candidate_aliases,
            target_aliases=target_aliases,
            endpoint=endpoint,
            section=section,
            publication_types=publication_types,
        )
    except (Exception,):
        # This is a trust-boundary parser.  Malformed/custom Mapping and
        # Collection implementations must fail closed just like bad text.
        return ()


def _parse(
    sentence: str,
    *,
    candidate_aliases: Mapping[str, Collection[str]],
    target_aliases: Collection[str],
    endpoint: str,
    section: str,
    publication_types: Collection[str] | str,
) -> tuple[PrimaryAbstractPair, ...]:
    _validate_sentence(sentence)
    _validate_delimiter_stack(sentence)
    canonical_section = _validate_section(section)
    canonical_publication_types = _validate_publication_types(publication_types)
    _reject_unsafe_language(sentence)

    candidates = _candidate_occurrences(sentence, candidate_aliases)
    if len(candidates) < 3:
        raise _Reject("at least three distinct candidates are required")

    targets = _alias_occurrences(sentence, target_aliases, owner="target")
    if len(targets) != 1:
        raise _Reject("the target must occur exactly once")
    target = targets[0]

    if not isinstance(endpoint, str) or not endpoint or endpoint != endpoint.strip():
        raise _Reject("invalid endpoint")
    if any(character in endpoint for character in "\r\n[]"):
        raise _Reject("invalid endpoint")
    if _ENDPOINT_MARKER_RE.fullmatch(endpoint) is None:
        raise _Reject("unsupported endpoint")
    endpoint_matches = _exact_matches(sentence, endpoint)
    if len(endpoint_matches) != 1:
        raise _Reject("the endpoint must occur exactly once")
    endpoint_start, endpoint_end = endpoint_matches[0]
    endpoint_span = _span(sentence, endpoint_start, endpoint_end)
    endpoint_wrapper_span = _endpoint_wrapper(
        sentence, endpoint_start=endpoint_start, endpoint_end=endpoint_end
    )
    endpoint_wrapper_raw = (
        endpoint_wrapper_span.text if endpoint_wrapper_span is not None else None
    )
    endpoint_lexical_start = (
        endpoint_wrapper_span.start
        if endpoint_wrapper_span is not None
        else endpoint_start
    )
    endpoint_lexical_end = (
        endpoint_wrapper_span.end if endpoint_wrapper_span is not None else endpoint_end
    )

    candidate_ranges = [(item.start, item.end) for item in candidates]
    if _overlaps_any(target.start, target.end, candidate_ranges):
        raise _Reject("target/candidate alias overlap")
    if _ranges_overlap(
        target.start, target.end, endpoint_lexical_start, endpoint_lexical_end
    ):
        raise _Reject("target/endpoint overlap")

    connectors = list(_CONNECTOR_RE.finditer(sentence))
    if len(connectors) != 1 or len(_FOLLOWED_BY_RE.findall(sentence)) != 1:
        raise _Reject("exactly one followed-by connector is required")
    connector = connectors[0]
    head_end = connector.start()
    head = sentence[:head_end]
    if not head.endswith(")"):
        raise _Reject("the common head must end in a parenthesis")

    head_candidates = [item for item in candidates if item.end <= head_end]
    followers = [item for item in candidates if item.start >= connector.end()]
    if len(head_candidates) != 1 or len(followers) != len(candidates) - 1:
        raise _Reject("every candidate must belong to the head or follower list")
    first = head_candidates[0]
    if first.start != 0:
        raise _Reject("the head candidate must begin the sentence")
    if any(item.start < connector.end() for item in followers):
        raise _Reject("candidate in connector")

    per_owner = Counter(item.owner for item in candidates)
    if any(count != 1 for count in per_owner.values()):
        raise _Reject("a candidate identifier is mentioned more than once")

    if head.count("(") != 1 or head.count(")") != 1:
        raise _Reject("the common head must contain one parenthesis")
    open_parenthesis = head.index("(")
    if not (first.end < target.start < target.end < open_parenthesis):
        raise _Reject("candidate/target/common-head order mismatch")
    _validate_observed_head_clause(
        sentence,
        candidate_end=first.end,
        target_start=target.start,
        target_end=target.end,
        open_parenthesis=open_parenthesis,
    )
    if not (
        open_parenthesis
        < endpoint_lexical_start
        < endpoint_lexical_end
        < head_end
    ):
        raise _Reject("the endpoint is outside the common head")

    _reject_additional_endpoints(sentence, endpoint_span)

    condition_match = _single_condition(sentence)
    if not (
        endpoint_lexical_end
        <= condition_match.start()
        < condition_match.end()
        <= head_end
    ):
        raise _Reject("the assay condition is outside the common head")
    if sentence[endpoint_lexical_end : condition_match.start()].strip():
        raise _Reject("unparsed text between endpoint and assay condition")
    condition = _condition_from_match(sentence, condition_match)

    head_match = _HEAD_MEASUREMENT_RE.search(head)
    if head_match is None:
        raise _Reject("the head requires its own exact measurement")
    if condition_match.end() > head_match.start():
        raise _Reject("condition/potency overlap")
    if sentence[condition_match.end() : head_match.start()].strip():
        raise _Reject("unparsed text between assay condition and potency")
    head_measurement = _measurement_from_match(sentence, head_match, full_end=head_end)
    _require_positive(head_measurement.value_raw)

    measurements: list[_Measurement] = [head_measurement]
    previous_end = connector.end()
    for index, candidate in enumerate(followers):
        gap = sentence[previous_end : candidate.start]
        if index == 0:
            if gap:
                raise _Reject("text before the first follower")
        elif index == len(followers) - 1:
            if re.fullmatch(r",\s+and\s+", gap, re.IGNORECASE) is None:
                raise _Reject("the final follower must use explicit and")
        elif re.fullmatch(r",\s+", gap) is None:
            raise _Reject("invalid follower separator")

        match = _TAIL_MEASUREMENT_RE.match(sentence, candidate.end)
        if match is None:
            raise _Reject("each follower needs one direct parenthesized measurement")
        measurement = _measurement_from_match(sentence, match, full_end=match.end())
        _require_positive(measurement.value_raw)
        measurements.append(measurement)
        previous_end = match.end()

    if _TAIL_SUFFIX_RE.fullmatch(sentence[previous_end:]) is None:
        raise _Reject("unparsed tail text")

    expected_numbers = {
        (condition.value_span.start, condition.value_span.end),
        *((item.value_span.start, item.value_span.end) for item in measurements),
    }
    ignored_lexical_ranges = [
        *((item.start, item.end) for item in candidates),
        (target.start, target.end),
        (endpoint_lexical_start, endpoint_lexical_end),
    ]
    observed_numbers: set[tuple[int, int]] = set()
    for match in _NUMERIC_TOKEN_RE.finditer(sentence):
        numeric_span = (match.start("value"), match.end("value"))
        if numeric_span in expected_numbers:
            observed_numbers.add(numeric_span)
            continue
        if _contained_by_any(*numeric_span, ignored_lexical_ranges):
            continue
        raise _Reject("unconsumed contextual numeral")
    if observed_numbers != expected_numbers:
        raise _Reject("unconsumed or missing numeric measurement")
    if (condition.value_span.start, condition.value_span.end) in {
        (item.value_span.start, item.value_span.end) for item in measurements
    }:
        raise _Reject("the assay condition cannot be reused as potency")

    sentence_sha256 = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
    sentence_span = _span(sentence, 0, len(sentence))
    target_span = _span(sentence, target.start, target.end)

    result: list[PrimaryAbstractPair] = []
    for pair_index, (candidate, measurement) in enumerate(
        zip((first, *followers), measurements, strict=True)
    ):
        candidate_span = _span(sentence, candidate.start, candidate.end)
        evidence_sha256 = _evidence_hash(
            sentence_sha256=sentence_sha256,
            pair_index=pair_index,
            candidate_id=candidate.owner,
            candidate_span=candidate_span,
            target_span=target_span,
            endpoint_span=endpoint_span,
            endpoint_wrapper_span=endpoint_wrapper_span,
            condition=condition,
            measurement=measurement,
            section=canonical_section,
            publication_types=canonical_publication_types,
        )
        result.append(
            PrimaryAbstractPair(
                sentence_span=sentence_span,
                sentence_sha256=sentence_sha256,
                evidence_sha256=evidence_sha256,
                grammar=PAPER_NATIVE_V5_GRAMMAR,
                pair_index=pair_index,
                pair_role="common_head" if pair_index == 0 else "direct_follower",
                candidate_id=candidate.owner,
                candidate_alias=candidate_span.text,
                candidate_span=candidate_span,
                target_alias=target_span.text,
                target_span=target_span,
                endpoint=endpoint_span.text,
                endpoint_span=endpoint_span,
                endpoint_wrapper_raw=endpoint_wrapper_raw,
                endpoint_wrapper_span=endpoint_wrapper_span,
                assay_condition=condition,
                measurement_span=measurement.measurement_span,
                value_raw=measurement.value_raw,
                value_decimal=measurement.value_decimal,
                value_span=measurement.value_span,
                unit_raw=measurement.unit_raw,
                unit_span=measurement.unit_span,
                relation_raw=measurement.relation_raw,
                relation_span=measurement.relation_span,
                relation_normalized=("=" if measurement.relation_raw == "=" else None),
                relation_basis=(
                    "literal_equality"
                    if measurement.relation_raw == "="
                    else "reported_point_estimate"
                ),
                literal_equality=measurement.relation_raw == "=",
                target_scope_basis=(
                    "explicit_common_head"
                    if pair_index == 0
                    else "inherited_from_common_head"
                ),
                endpoint_scope_basis=(
                    "explicit_common_head"
                    if pair_index == 0
                    else "inherited_from_common_head"
                ),
                assay_condition_scope_basis=(
                    "explicit_common_head"
                    if pair_index == 0
                    else "inherited_from_common_head"
                ),
                section=canonical_section,
                publication_types=canonical_publication_types,
            )
        )
    return tuple(result)


def _validate_sentence(sentence: str) -> None:
    if not isinstance(sentence, str) or not sentence or len(sentence) > 360:
        raise _Reject("invalid sentence length")
    if sentence != sentence.strip() or "\n" in sentence or "\r" in sentence:
        raise _Reject("the raw sentence must be preserved as one line")
    if "?" in sentence or "!" in sentence or ";" in sentence:
        raise _Reject("unsupported sentence punctuation")
    core = sentence[:-1] if sentence.endswith(".") else sentence
    if re.search(r"(?<!\d)\.(?!\d)", core):
        raise _Reject("multiple sentences or abbreviations are not accepted")


def _validate_observed_head_clause(
    sentence: str,
    *,
    candidate_end: int,
    target_start: int,
    target_end: int,
    open_parenthesis: int,
) -> None:
    """Require an observed/measured inhibition assertion, not a forecast.

    The target span itself is supplied by the exact alias binder.  This gate
    accounts for every character around that span up to the assay parenthesis,
    so adding an unrecognised modifier cannot inherit assertive semantics from
    a later ``inhibitor`` token.
    """

    prefix = sentence[candidate_end:target_start]
    suffix = sentence[target_end:open_parenthesis]
    potency_observation = (
        _OBSERVED_POTENCY_PREFIX_RE.fullmatch(prefix) is not None
        and _INHIBITOR_SUFFIX_RE.fullmatch(suffix) is not None
    )
    direct_observation = (
        (
            _OBSERVED_TO_INHIBIT_PREFIX_RE.fullmatch(prefix) is not None
            or _OBSERVED_ACTIVE_INHIBITION_PREFIX_RE.fullmatch(prefix)
            is not None
        )
        and _EMPTY_TARGET_SUFFIX_RE.fullmatch(suffix) is not None
    )
    if not (potency_observation or direct_observation):
        raise _Reject("the head is not a closed observed inhibition assertion")


def _validate_delimiter_stack(sentence: str) -> None:
    stack: list[tuple[str, int]] = []
    opening = {"(": ")", "[": "]"}
    for offset, character in enumerate(sentence):
        if character in opening:
            stack.append((character, offset))
            if len(stack) > _MAX_DELIMITER_DEPTH:
                raise _Reject("delimiter depth exceeded")
        elif character in ") ]".replace(" ", ""):
            if not stack:
                raise _Reject("unmatched closing delimiter")
            opener, _ = stack.pop()
            if opening[opener] != character:
                raise _Reject("crossed delimiters")
    if stack:
        raise _Reject("unclosed delimiter")


def _validate_section(section: str) -> Literal["Abstract", "Results"]:
    if not isinstance(section, str):
        raise _Reject("invalid section")
    normalized = unicodedata.normalize("NFKC", section).strip().casefold()
    if normalized == "abstract":
        return "Abstract"
    if normalized == "results":
        return "Results"
    raise _Reject("only Abstract or Results is accepted")


def _validate_publication_types(
    publication_types: Collection[str] | str,
) -> tuple[str, ...]:
    if isinstance(publication_types, str):
        values: tuple[object, ...] = (publication_types,)
    elif isinstance(publication_types, Collection):
        values = tuple(publication_types)
    else:
        raise _Reject("invalid publication types")
    if not values or any(not isinstance(value, str) for value in values):
        raise _Reject("invalid publication types")
    normalized = tuple(
        sorted(
            {
                unicodedata.normalize("NFKC", value).strip().casefold()
                for value in values
                if value.strip()
            }
        )
    )
    if not normalized or any(_NON_PRIMARY_PUBLICATION_RE.search(item) for item in normalized):
        raise _Reject("review/non-primary publication type")
    if not any(item in _PRIMARY_PUBLICATION_TYPES for item in normalized):
        raise _Reject("no primary publication type")
    return normalized


def _reject_unsafe_language(sentence: str) -> None:
    if "respectively" in sentence.casefold():
        raise _Reject("respectively-only mapping is not accepted")
    if _CITATION_RE.search(sentence):
        raise _Reject("citation in evidence sentence")
    if _PRIOR_REPORT_RE.search(sentence):
        raise _Reject("previously reported evidence")
    if _UNCERTAIN_RE.search(sentence) or _NEGATIVE_RE.search(sentence):
        raise _Reject("non-assertive relation")
    if _APPROXIMATE_RE.search(sentence):
        raise _Reject("approximate value")
    if _RANGE_RE.search(sentence) or _RELATIONAL_SYMBOL_RE.search(sentence):
        raise _Reject("non-scalar or bounded value")
    if "/" in sentence or _VALUE_COMPARATOR_RE.search(sentence):
        raise _Reject("unsupported list or comparator")
    if _NONEXPERIMENTAL_LANGUAGE_RE.search(sentence):
        raise _Reject("nonexperimental evidence language")


def _candidate_occurrences(
    sentence: str, candidate_aliases: Mapping[str, Collection[str]]
) -> list[_Occurrence]:
    if not isinstance(candidate_aliases, Mapping) or not candidate_aliases:
        raise _Reject("invalid candidate alias map")
    occurrences: list[_Occurrence] = []
    global_alias_owners: dict[str, str] = {}
    for candidate_id, aliases in candidate_aliases.items():
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id != candidate_id.strip():
            raise _Reject("invalid candidate identifier")
        clean_aliases = _validate_alias_collection(aliases)
        for alias in clean_aliases:
            key = _alias_key(alias)
            previous_owner = global_alias_owners.get(key)
            if previous_owner is not None:
                raise _Reject("candidate alias collision")
            global_alias_owners[key] = candidate_id
            for start, end in _exact_matches(sentence, alias):
                occurrences.append(
                    _Occurrence(
                        owner=candidate_id,
                        configured_alias=alias,
                        start=start,
                        end=end,
                    )
                )
    occurrences.sort(key=lambda item: (item.start, item.end, item.owner))
    _reject_occurrence_overlaps(occurrences)
    return occurrences


def _alias_occurrences(
    sentence: str, aliases: Collection[str], *, owner: str
) -> list[_Occurrence]:
    clean_aliases = _validate_alias_collection(aliases)
    keys: set[str] = set()
    occurrences: list[_Occurrence] = []
    for alias in clean_aliases:
        key = _alias_key(alias)
        if key in keys:
            raise _Reject("alias collision")
        keys.add(key)
        for start, end in _exact_matches(sentence, alias):
            occurrences.append(
                _Occurrence(owner=owner, configured_alias=alias, start=start, end=end)
            )
    occurrences.sort(key=lambda item: (item.start, item.end, item.configured_alias))
    _reject_occurrence_overlaps(occurrences)
    return occurrences


def _validate_alias_collection(aliases: Collection[str]) -> tuple[str, ...]:
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Collection):
        raise _Reject("aliases must be a collection of exact strings")
    values = tuple(aliases)
    if not values:
        raise _Reject("empty alias collection")
    if any(
        not isinstance(alias, str)
        or not alias
        or alias != alias.strip()
        or "\n" in alias
        or "\r" in alias
        for alias in values
    ):
        raise _Reject("invalid alias")
    return tuple(sorted(values, key=lambda value: (_alias_key(value), value)))


def _alias_key(alias: str) -> str:
    return unicodedata.normalize("NFKC", alias).casefold()


def _exact_matches(sentence: str, value: str) -> list[tuple[int, int]]:
    pattern = re.compile(
        rf"(?<![\w-]){re.escape(value)}(?![\w-])",
        re.IGNORECASE,
    )
    return [(match.start(), match.end()) for match in pattern.finditer(sentence)]


def _endpoint_wrapper(
    sentence: str, *, endpoint_start: int, endpoint_end: int
) -> SourceSpan | None:
    """Return one exact balanced ``[...]`` wrapper or reject partial/nested use."""

    has_left = endpoint_start > 0 and sentence[endpoint_start - 1] == "["
    has_right = endpoint_end < len(sentence) and sentence[endpoint_end] == "]"
    if has_left != has_right:
        raise _Reject("endpoint wrapper is not balanced")
    if not has_left:
        return None
    wrapper_start = endpoint_start - 1
    wrapper_end = endpoint_end + 1
    if (
        (wrapper_start > 0 and sentence[wrapper_start - 1] == "[")
        or (wrapper_end < len(sentence) and sentence[wrapper_end] == "]")
    ):
        raise _Reject("endpoint wrapper must occur exactly once")
    return _span(sentence, wrapper_start, wrapper_end)


def _reject_occurrence_overlaps(occurrences: list[_Occurrence]) -> None:
    for previous, current in zip(occurrences, occurrences[1:]):
        if current.start < previous.end:
            raise _Reject("overlapping aliases")


def _single_condition(sentence: str) -> re.Match[str]:
    matches = [
        *_BRACKETED_CONDITION_RE.finditer(sentence),
        *_UNBRACKETED_CONDITION_RE.finditer(sentence),
    ]
    matches.sort(key=lambda match: (match.start(), match.end()))
    if len(matches) != 1:
        raise _Reject("exactly one common assay condition is required")
    _require_positive(matches[0].group("value"))
    return matches[0]


def _condition_from_match(sentence: str, match: re.Match[str]) -> AssayCondition:
    value_raw = match.group("value")
    return AssayCondition(
        span=_span(sentence, match.start(), match.end()),
        value_raw=value_raw,
        value_decimal=_positive_decimal(value_raw),
        value_span=_span(sentence, match.start("value"), match.end("value")),
        unit_raw=match.group("unit"),
        unit_span=_span(sentence, match.start("unit"), match.end("unit")),
        subject=match.group("subject"),
        subject_span=_span(sentence, match.start("subject"), match.end("subject")),
    )


def _measurement_from_match(
    sentence: str, match: re.Match[str], *, full_end: int
) -> _Measurement:
    value_span = _span(sentence, match.start("value"), match.end("value"))
    unit_span = _span(sentence, match.start("unit"), match.end("unit"))
    relation_raw = match.group("relation")
    relation_span = (
        _span(sentence, match.start("relation"), match.end("relation"))
        if relation_raw is not None
        else None
    )
    measurement_start = (
        relation_span.start if relation_span is not None else value_span.start
    )
    return _Measurement(
        full_end=full_end,
        measurement_span=_span(sentence, measurement_start, unit_span.end),
        value_raw=value_span.text,
        value_decimal=_positive_decimal(value_span.text),
        value_span=value_span,
        unit_raw=unit_span.text,
        unit_span=unit_span,
        relation_raw=relation_raw,
        relation_span=relation_span,
    )


def _require_positive(value: str) -> None:
    _positive_decimal(value)


def _positive_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise _Reject("invalid decimal") from error
    if not parsed.is_finite() or parsed <= 0:
        raise _Reject("measurement must be a positive finite scalar")
    return parsed


def _reject_additional_endpoints(sentence: str, endpoint_span: SourceSpan) -> None:
    for match in _ENDPOINT_MARKER_RE.finditer(sentence):
        if not (
            endpoint_span.start <= match.start()
            and match.end() <= endpoint_span.end
        ):
            raise _Reject("additional endpoint marker")


def _evidence_hash(
    *,
    sentence_sha256: str,
    pair_index: int,
    candidate_id: str,
    candidate_span: SourceSpan,
    target_span: SourceSpan,
    endpoint_span: SourceSpan,
    endpoint_wrapper_span: SourceSpan | None,
    condition: AssayCondition,
    measurement: _Measurement,
    section: str,
    publication_types: tuple[str, ...],
) -> str:
    payload = {
        "candidate_id": candidate_id,
        "candidate_span": [candidate_span.start, candidate_span.end, candidate_span.text],
        "condition_span": [
            condition.span.start,
            condition.span.end,
            condition.span.text,
        ],
        "condition_subject": condition.subject,
        "condition_unit_raw": condition.unit_raw,
        "condition_value_raw": condition.value_raw,
        "endpoint_span": [endpoint_span.start, endpoint_span.end, endpoint_span.text],
        "endpoint_wrapper_raw": (
            endpoint_wrapper_span.text if endpoint_wrapper_span is not None else None
        ),
        "endpoint_wrapper_span": (
            [
                endpoint_wrapper_span.start,
                endpoint_wrapper_span.end,
                endpoint_wrapper_span.text,
            ]
            if endpoint_wrapper_span is not None
            else None
        ),
        "grammar": PAPER_NATIVE_V5_GRAMMAR,
        "measurement_span": [
            measurement.measurement_span.start,
            measurement.measurement_span.end,
            measurement.measurement_span.text,
        ],
        "pair_index": pair_index,
        "scope_basis": (
            "explicit_common_head"
            if pair_index == 0
            else "inherited_from_common_head"
        ),
        "publication_types": publication_types,
        "relation_basis": (
            "literal_equality"
            if measurement.relation_raw == "="
            else "reported_point_estimate"
        ),
        "relation_normalized": "=" if measurement.relation_raw == "=" else None,
        "relation_raw": measurement.relation_raw,
        "literal_equality": measurement.relation_raw == "=",
        "schema": PAPER_NATIVE_V5_PARSER_SCHEMA_VERSION,
        "section": section,
        "sentence_sha256": sentence_sha256,
        "target_span": [target_span.start, target_span.end, target_span.text],
        "unit_raw": measurement.unit_raw,
        "value_decimal": str(measurement.value_decimal),
        "value_raw": measurement.value_raw,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _span(sentence: str, start: int, end: int) -> SourceSpan:
    return SourceSpan(start=start, end=end, text=sentence[start:end])


def _ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _overlaps_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(_ranges_overlap(start, end, other_start, other_end) for other_start, other_end in ranges)


def _contained_by_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(other_start <= start and end <= other_end for other_start, other_end in ranges)
