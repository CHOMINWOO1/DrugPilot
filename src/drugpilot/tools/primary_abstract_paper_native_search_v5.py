"""Bounded PubMed adapter for paper-native comparative abstract evidence v5.

The adapter performs at most one ESearch, one ESummary, and one EFetch call.
It stores only hashes, bounded evidence sentences, and structured screening
records; full abstracts and raw response bodies are never returned.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Literal, Protocol
import unicodedata
from urllib.parse import urlencode
from xml.etree import ElementTree

from drugpilot.tools.primary_abstract_paper_native_v5 import (
    PAPER_NATIVE_V5_GRAMMAR,
    PrimaryAbstractPair,
    parse_paper_native_primary_abstract_v5,
)

__all__ = [
    "CandidateSearchAliases",
    "HttpCallProvenance",
    "PAPER_NATIVE_V5_POLICY_TOKEN",
    "PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION",
    "PrimaryAbstractGroup",
    "PrimaryAbstractSearchOutcome",
    "PrimaryStudyAttestation",
    "RecordScreening",
    "search_paper_native_primary_abstract_v5",
]


PAPER_NATIVE_V5_POLICY_TOKEN = (
    "dynamic_pubmed_primary_abstract_paper_native_v5_required"
)
PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION = "primary_abstract_paper_native_search/v5"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_ABSOLUTE_MAX_CANDIDATES = 12
_ABSOLUTE_MAX_DOCUMENTS = 8
_MAX_TARGET_ALIASES = 8
_MAX_EXACT_ALIASES_PER_CANDIDATE = 16
_MAX_QUERY_TERM_CHARS = 160
_MAX_SENTENCE_CHARS = 360
_MAX_ARTICLE_TITLE_CHARS = 500
_MAX_SENTENCES_PER_RECORD = 64
_MAX_SECTION_CHARS = 20_000
_MAX_FETCH_BYTES = 2_000_000
_MAX_GET_QUERY_BYTES = 7_000
_MIN_VERIFIED_CANDIDATES = 3

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
_EXCLUDED_PUBLICATION_RE = re.compile(
    r"(?:review|meta-analysis|editorial|letter|comment|guideline|news|"
    r"retracted publication|retraction (?:notice|of publication)|published erratum|"
    r"expression of concern|protocol)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SUPPORTED_ENDPOINT_RE = re.compile(
    r"(?:p?IC50|p?EC50|GI50|LC50|CC50|AC50|ED50|p?K[di]|pSTAT\d+)",
    re.IGNORECASE,
)
_METHOD_PASSIVE_RE = re.compile(
    r"^(?:inhibitory\s+)?(?:potenc(?:y|ies)|selectivit(?:y|ies)|effects?|"
    r"activit(?:y|ies)|inhibition|signaling|cell\s+growth|growth|viability|"
    r"responses?|profiles?).{0,220}\b(?:was|were)\s+"
    r"(?:(?:directly|experimentally)\s+)?"
    r"(?P<verb>measured|determined|assessed|evaluated|tested|quantified)\s+"
    r"(?P<method>by|using|with|in)\b",
    re.IGNORECASE,
)
_METHOD_ACTIVE_RE = re.compile(
    r"^(?:we|this\s+study)\s+"
    r"(?:(?:directly|experimentally)\s+)?"
    r"(?P<verb>measured|determined|assessed|evaluated|tested|quantified)\b"
    r".{0,220}\b(?P<method>by|using|with|in)\b",
    re.IGNORECASE,
)
_METHOD_UNSAFE_RE = re.compile(
    r"\b(?:review|meta-analysis|previously|formerly|earlier|prior\s+(?:work|study)|"
    r"as\s+reported|according\s+to|predicted|in(?:\s+|-)silico|computational|"
    r"modeled|modelled|simulation|hypothetical|putative)\b",
    re.IGNORECASE,
)
_LINKED_STATUS_NOTICE_TYPES = frozenset(
    {
        "correctedandrepublishedfrom",
        "correctedandrepublishedin",
        "erratumfor",
        "erratumin",
        "expressionofconcernfor",
        "expressionofconcernin",
        "republishedfrom",
        "republishedin",
        "retractionof",
        "retractionin",
        "updateof",
        "updatein",
    }
)
_MAX_LINKED_STATUS_NOTICES = 32


class ScientificHttpClientLike(Protocol):
    """The read-only subset of :class:`ScientificHttpClient` used here."""

    max_attempts: int

    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class CandidateSearchAliases:
    """One bounded query alias and all exact parser aliases for a candidate."""

    preferred_search_alias: str
    exact_aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HttpCallProvenance:
    """Secret-free provenance for one attempted public HTTP call."""

    stage: Literal["esearch", "esummary", "efetch"]
    method: Literal["GET"]
    endpoint: str
    parameter_keys: tuple[str, ...]
    parameter_sha256: str
    status: Literal["completed", "failed"]
    status_code: int | None
    response_sha256: str | None
    error_type: str | None
    call_sha256: str


@dataclass(frozen=True, slots=True)
class PrimaryStudyAttestation:
    """Bounded positive evidence that the selected PubMed record is original research."""

    schema_version: Literal["pubmed_primary_study_attestation/v2"]
    policy: Literal[
        "same_abstract_experimental_method_and_linked_notice_check_v2"
    ]
    status: Literal["verified"]
    document_id: str
    article_title: str
    article_title_sha256: str
    method_sentence_index: int
    method_sentence: str
    method_sentence_sha256: str
    method_indicator: Literal[
        "passive_experimental_measurement_method",
        "active_experimental_measurement_method",
    ]
    summary_publication_types: tuple[str, ...]
    fetch_publication_types: tuple[str, ...]
    retraction_safety_policy: Literal[
        "pubmed_types_and_comments_corrections_fail_closed_v1"
    ]
    retraction_safe: Literal[True]
    correction_safe: Literal[True]
    linked_status_notice: Literal["none_detected"]
    linked_status_notice_types: tuple[()]
    linked_status_notice_pmids: tuple[()]
    fixture: Literal[False]
    attestation_sha256: str


@dataclass(frozen=True, slots=True)
class PrimaryAbstractGroup:
    """One independently revalidatable, bounded evidence sentence."""

    document_id: str
    document_rank: int
    section: Literal["Abstract", "Results"]
    sentence_index: int
    sentence: str
    sentence_sha256: str
    group_sha256: str
    primary_study_attestation: PrimaryStudyAttestation
    summary_publication_types: tuple[str, ...]
    fetch_publication_types: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    pairs: tuple[PrimaryAbstractPair, ...]


@dataclass(frozen=True, slots=True)
class RecordScreening:
    """Bounded disposition for one unique ESearch document identifier."""

    document_id: str
    publication_types: tuple[str, ...]
    fetch_publication_types: tuple[str, ...]
    status: Literal["verified", "rejected"]
    rejection_codes: tuple[str, ...]
    eligible_section_count: int
    sentence_count: int
    bounded_sentence_count: int
    verified_group_count: int
    record_sha256: str


@dataclass(frozen=True, slots=True)
class PrimaryAbstractSearchOutcome:
    """Complete immutable result of one bounded three-stage PubMed search."""

    schema_version: Literal["primary_abstract_paper_native_search/v5"]
    policy_token: Literal[
        "dynamic_pubmed_primary_abstract_paper_native_v5_required"
    ]
    grammar: Literal["head_measurement_followed_by_candidate_value_list_v1"]
    status: Literal["verified", "not_verified", "failed"]
    reason_code: str | None
    query: str | None
    query_sha256: str | None
    query_translation: str | None
    query_translation_sha256: str | None
    search_sort: Literal["pub date"]
    max_candidates: int
    max_documents: int
    requested_candidate_count: int
    queried_candidate_count: int
    requested_document_count: int
    returned_document_count: int
    unique_document_count: int
    total_result_count: int
    unsearched_result_count: int
    returned_document_ids: tuple[str, ...]
    returned_document_ids_sha256: str | None
    screened_document_count: int
    screened_sentence_count: int
    verified_candidate_ids: tuple[str, ...]
    calls: tuple[HttpCallProvenance, ...]
    records: tuple[RecordScreening, ...]
    groups: tuple[PrimaryAbstractGroup, ...]
    selected_group_sha256: str | None
    outcome_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedInput:
    target_aliases: tuple[str, ...]
    endpoint: str
    candidates: tuple[tuple[str, CandidateSearchAliases], ...]
    parser_aliases: Mapping[str, tuple[str, ...]]
    requested_candidate_count: int
    max_candidates: int
    max_documents: int


@dataclass(slots=True)
class _RecordDraft:
    document_id: str
    publication_types: tuple[str, ...] = ()
    fetch_publication_types: tuple[str, ...] = ()
    rejection_codes: list[str] = field(default_factory=list)
    eligible_section_count: int = 0
    sentence_count: int = 0
    bounded_sentence_count: int = 0
    groups: list[PrimaryAbstractGroup] = field(default_factory=list)


class _Reject(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StageFailure(RuntimeError):
    def __init__(self, code: str, provenance: HttpCallProvenance) -> None:
        super().__init__(code)
        self.code = code
        self.provenance = provenance


@dataclass(frozen=True, slots=True)
class _SearchBatch:
    document_ids: tuple[str, ...]
    total_result_count: int
    query_translation: str


@dataclass(frozen=True, slots=True)
class _FetchedRecord:
    publication_types: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]
    article_title: str
    method_sentences: tuple[tuple[int, str, str], ...]
    linked_status_notice_types: tuple[str, ...]
    linked_status_notice_pmids: tuple[str, ...]


def search_paper_native_primary_abstract_v5(
    client: ScientificHttpClientLike,
    *,
    target_aliases: Collection[str],
    endpoint: str,
    candidates: Mapping[str, CandidateSearchAliases],
    max_candidates: int = _ABSOLUTE_MAX_CANDIDATES,
    max_documents: int = _ABSOLUTE_MAX_DOCUMENTS,
) -> PrimaryAbstractSearchOutcome:
    """Search PubMed once and verify direct paired measurements fail-closed.

    Mapping order is not trusted: candidates and aliases are canonicalized
    before the query is built.  Invalid discovery input performs no HTTP call.
    A failure at any attempted HTTP stage returns no records or groups.
    """

    try:
        _validate_single_attempt_client(client)
        validated = _validate_input(
            target_aliases=target_aliases,
            endpoint=endpoint,
            candidates=candidates,
            max_candidates=max_candidates,
            max_documents=max_documents,
        )
        query = _build_query(validated)
    except Exception as error:
        return _empty_outcome(
            status="failed",
            reason_code=(error.code if isinstance(error, _Reject) else "invalid_input"),
            query=None,
            max_candidates=_safe_limit(max_candidates, _ABSOLUTE_MAX_CANDIDATES),
            max_documents=_safe_limit(max_documents, _ABSOLUTE_MAX_DOCUMENTS),
            requested_candidate_count=(len(candidates) if isinstance(candidates, Mapping) else 0),
        )

    query_sha256 = _sha256_text(query)
    calls: list[HttpCallProvenance] = []
    returned_document_count = 0
    unique_document_count = 0
    total_result_count = 0
    query_translation: str | None = None
    query_translation_sha256: str | None = None
    returned_document_ids: tuple[str, ...] = ()
    returned_document_ids_sha256: str | None = None

    search_params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": validated.max_documents,
        "sort": "pub date",
    }
    try:
        search_payload, provenance = _get_json(
            client, stage="esearch", endpoint=PUBMED_SEARCH_URL, params=search_params
        )
        calls.append(provenance)
        search_batch = _search_batch(
            search_payload, max_documents=validated.max_documents
        )
        raw_ids = list(search_batch.document_ids)
        total_result_count = search_batch.total_result_count
        query_translation = search_batch.query_translation
        query_translation_sha256 = _sha256_text(query_translation)
    except _StageFailure as error:
        calls.append(error.provenance)
        return _failed_after_query(
            validated, query, query_sha256, calls, error.code
        )
    except _Reject as error:
        return _failed_after_query(
            validated, query, query_sha256, calls, error.code
        )

    returned_document_count = len(raw_ids)
    returned_document_ids = tuple(raw_ids)
    returned_document_ids_sha256 = _hash_sequence(returned_document_ids)
    document_ids = tuple(dict.fromkeys(raw_ids))
    unique_document_count = len(document_ids)
    if not document_ids:
        return _completed_outcome(
            validated=validated,
            status="not_verified",
            reason_code="NOT_FOUND_WITHIN_BOUND",
            query=query,
            query_sha256=query_sha256,
            returned_document_count=0,
            unique_document_count=0,
            total_result_count=total_result_count,
            query_translation=query_translation,
            query_translation_sha256=query_translation_sha256,
            returned_document_ids=returned_document_ids,
            returned_document_ids_sha256=returned_document_ids_sha256,
            calls=calls,
            records=(),
            groups=(),
            screened_sentence_count=0,
        )

    summary_params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(document_ids),
        "retmode": "json",
    }
    try:
        summary_payload, provenance = _get_json(
            client,
            stage="esummary",
            endpoint=PUBMED_SUMMARY_URL,
            params=summary_params,
        )
        calls.append(provenance)
        summary_items = _summary_items(summary_payload, document_ids)
    except _StageFailure as error:
        calls.append(error.provenance)
        return _failed_after_query(
            validated,
            query,
            query_sha256,
            calls,
            error.code,
            returned_document_count=returned_document_count,
            unique_document_count=unique_document_count,
            total_result_count=total_result_count,
            query_translation=query_translation,
            query_translation_sha256=query_translation_sha256,
            returned_document_ids=returned_document_ids,
            returned_document_ids_sha256=returned_document_ids_sha256,
        )
    except _Reject as error:
        return _failed_after_query(
            validated,
            query,
            query_sha256,
            calls,
            error.code,
            returned_document_count=returned_document_count,
            unique_document_count=unique_document_count,
            total_result_count=total_result_count,
            query_translation=query_translation,
            query_translation_sha256=query_translation_sha256,
            returned_document_ids=returned_document_ids,
            returned_document_ids_sha256=returned_document_ids_sha256,
        )

    drafts = {document_id: _RecordDraft(document_id) for document_id in document_ids}
    eligible_ids: list[str] = []
    for document_id in document_ids:
        draft = drafts[document_id]
        item = summary_items.get(document_id)
        if item is None:
            draft.rejection_codes.append("summary_missing")
            continue
        publication_types = _publication_types(item.get("pubtype"))
        draft.publication_types = publication_types
        if not _is_primary_non_review(publication_types):
            draft.rejection_codes.append("non_primary_or_review")
            continue
        eligible_ids.append(document_id)

    fetch_records: Mapping[str, _FetchedRecord] = {}
    duplicate_fetch_ids: frozenset[str] = frozenset()
    if eligible_ids:
        fetch_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(eligible_ids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        try:
            fetch_text, provenance = _get_text(
                client,
                stage="efetch",
                endpoint=PUBMED_FETCH_URL,
                params=fetch_params,
            )
            calls.append(provenance)
            fetch_records, duplicate_fetch_ids = _parse_fetch_xml(
                fetch_text, expected_ids=frozenset(eligible_ids)
            )
        except _StageFailure as error:
            calls.append(error.provenance)
            return _failed_after_query(
                validated,
                query,
                query_sha256,
                calls,
                error.code,
                returned_document_count=returned_document_count,
                unique_document_count=unique_document_count,
                total_result_count=total_result_count,
                query_translation=query_translation,
                query_translation_sha256=query_translation_sha256,
                returned_document_ids=returned_document_ids,
                returned_document_ids_sha256=returned_document_ids_sha256,
            )
        except _Reject as error:
            return _failed_after_query(
                validated,
                query,
                query_sha256,
                calls,
                error.code,
                returned_document_count=returned_document_count,
                unique_document_count=unique_document_count,
                total_result_count=total_result_count,
                query_translation=query_translation,
                query_translation_sha256=query_translation_sha256,
                returned_document_ids=returned_document_ids,
                returned_document_ids_sha256=returned_document_ids_sha256,
            )

    all_groups: list[PrimaryAbstractGroup] = []
    screened_sentence_count = 0
    document_rank_by_id = {
        document_id: rank for rank, document_id in enumerate(document_ids)
    }
    for document_id in eligible_ids:
        draft = drafts[document_id]
        if document_id in duplicate_fetch_ids:
            draft.rejection_codes.append("duplicate_fetch_record")
            continue
        fetched = fetch_records.get(document_id)
        if fetched is None:
            draft.rejection_codes.append("abstract_missing")
            continue
        draft.fetch_publication_types = fetched.publication_types
        if not _is_primary_non_review(fetched.publication_types):
            draft.rejection_codes.append("fetch_non_primary_or_review")
            continue
        primary_study = _primary_study_attestation(
            document_id=document_id,
            article_title=fetched.article_title,
            method_sentences=fetched.method_sentences,
            summary_publication_types=draft.publication_types,
            fetch_publication_types=fetched.publication_types,
            linked_status_notice_types=fetched.linked_status_notice_types,
            linked_status_notice_pmids=fetched.linked_status_notice_pmids,
        )
        if primary_study is None:
            draft.rejection_codes.append(
                "linked_status_notice_not_safe"
                if fetched.linked_status_notice_types
                else "primary_study_attestation_missing"
            )
            continue
        sections = fetched.sections
        if any(len(text) > _MAX_SECTION_CHARS for _, text in sections):
            draft.rejection_codes.append("abstract_bounds_exceeded")
            continue

        sentences: list[tuple[str, str]] = []
        for section, text in sections:
            draft.eligible_section_count += 1
            sentences.extend((section, value) for value in _sentences(text))
        draft.sentence_count = len(sentences)
        if len(sentences) > _MAX_SENTENCES_PER_RECORD:
            draft.rejection_codes.append("abstract_bounds_exceeded")
            continue

        saw_under_three = False
        seen_group_hashes: set[str] = set()
        for sentence_index, (section, sentence) in enumerate(sentences):
            if not sentence or len(sentence) > _MAX_SENTENCE_CHARS:
                continue
            draft.bounded_sentence_count += 1
            screened_sentence_count += 1
            pairs = parse_paper_native_primary_abstract_v5(
                sentence,
                candidate_aliases=validated.parser_aliases,
                target_aliases=validated.target_aliases,
                endpoint=validated.endpoint,
                section=section,
                publication_types=tuple(
                    sorted(
                        set(draft.publication_types)
                        | set(draft.fetch_publication_types)
                    )
                ),
            )
            if not pairs:
                continue
            candidate_ids = tuple(dict.fromkeys(pair.candidate_id for pair in pairs))
            if len(candidate_ids) < _MIN_VERIFIED_CANDIDATES:
                saw_under_three = True
                continue
            group = _group(
                document_id=document_id,
                document_rank=document_rank_by_id[document_id],
                section=section,
                sentence_index=sentence_index,
                sentence=sentence,
                primary_study_attestation=primary_study,
                summary_publication_types=draft.publication_types,
                fetch_publication_types=draft.fetch_publication_types,
                candidate_ids=candidate_ids,
                pairs=pairs,
                query_sha256=query_sha256,
            )
            if group.group_sha256 in seen_group_hashes:
                continue
            seen_group_hashes.add(group.group_sha256)
            draft.groups.append(group)
            all_groups.append(group)

        if draft.groups:
            draft.rejection_codes.clear()
        elif draft.eligible_section_count == 0:
            draft.rejection_codes.append("no_eligible_abstract_section")
        elif draft.bounded_sentence_count == 0:
            draft.rejection_codes.append("no_bounded_sentence")
        elif saw_under_three:
            draft.rejection_codes.append("insufficient_distinct_candidates")
        else:
            draft.rejection_codes.append("no_primary_pair_group")

    records = tuple(_freeze_record(drafts[document_id]) for document_id in document_ids)
    status: Literal["verified", "not_verified"] = (
        "verified" if all_groups else "not_verified"
    )
    return _completed_outcome(
        validated=validated,
        status=status,
        reason_code=None if all_groups else "NOT_FOUND_WITHIN_BOUND",
        query=query,
        query_sha256=query_sha256,
        returned_document_count=returned_document_count,
        unique_document_count=unique_document_count,
        total_result_count=total_result_count,
        query_translation=query_translation,
        query_translation_sha256=query_translation_sha256,
        returned_document_ids=returned_document_ids,
        returned_document_ids_sha256=returned_document_ids_sha256,
        calls=calls,
        records=records,
        groups=tuple(all_groups),
        screened_sentence_count=screened_sentence_count,
    )


def _validate_input(
    *,
    target_aliases: Collection[str],
    endpoint: str,
    candidates: Mapping[str, CandidateSearchAliases],
    max_candidates: int,
    max_documents: int,
) -> _ValidatedInput:
    if type(max_candidates) is not int or not 1 <= max_candidates <= _ABSOLUTE_MAX_CANDIDATES:
        raise _Reject("invalid_candidate_limit")
    if type(max_documents) is not int or not 1 <= max_documents <= _ABSOLUTE_MAX_DOCUMENTS:
        raise _Reject("invalid_document_limit")
    targets = _exact_strings(
        target_aliases, maximum_count=_MAX_TARGET_ALIASES, maximum_chars=_MAX_QUERY_TERM_CHARS
    )
    for target in targets:
        _validate_query_term(target)
    endpoint_value = _bounded_string(endpoint, maximum_chars=_MAX_QUERY_TERM_CHARS)
    _validate_query_term(endpoint_value)
    if "[" in endpoint_value or "]" in endpoint_value:
        raise _Reject("endpoint_must_be_base_token")
    if _SUPPORTED_ENDPOINT_RE.fullmatch(endpoint_value) is None:
        raise _Reject("unsupported_endpoint")
    if not isinstance(candidates, Mapping) or not candidates:
        raise _Reject("invalid_candidates")

    normalized_ids: set[str] = set()
    alias_owners: dict[str, str] = {}
    validated_candidates: list[tuple[str, CandidateSearchAliases]] = []
    target_keys = {_key(value) for value in targets}
    for candidate_id, spec in candidates.items():
        candidate_id = _bounded_string(candidate_id, maximum_chars=160)
        candidate_key = _key(candidate_id)
        if candidate_key in normalized_ids:
            raise _Reject("candidate_id_conflict")
        normalized_ids.add(candidate_key)
        if not isinstance(spec, CandidateSearchAliases):
            raise _Reject("invalid_candidate_alias_spec")
        preferred = _bounded_string(
            spec.preferred_search_alias, maximum_chars=_MAX_QUERY_TERM_CHARS
        )
        _validate_query_term(preferred)
        aliases = _exact_strings(
            spec.exact_aliases,
            maximum_count=_MAX_EXACT_ALIASES_PER_CANDIDATE,
            maximum_chars=_MAX_QUERY_TERM_CHARS,
            require_tuple=True,
        )
        alias_keys = {_key(value) for value in aliases}
        if _key(preferred) not in alias_keys:
            raise _Reject("preferred_alias_not_exact")
        if alias_keys & target_keys:
            raise _Reject("target_candidate_alias_conflict")
        for alias in aliases:
            alias_key = _key(alias)
            if alias_key in alias_owners:
                raise _Reject("alias_conflict")
            alias_owners[alias_key] = candidate_id
        validated_candidates.append(
            (
                candidate_id,
                CandidateSearchAliases(
                    preferred_search_alias=preferred,
                    exact_aliases=aliases,
                ),
            )
        )

    if len(validated_candidates) < _MIN_VERIFIED_CANDIDATES:
        raise _Reject("insufficient_candidate_pool")
    if len(validated_candidates) > max_candidates:
        raise _Reject("candidate_pool_bound_exceeded")
    validated_candidates.sort(key=lambda item: (_key(item[0]), item[0]))
    selected = tuple(validated_candidates)
    return _ValidatedInput(
        target_aliases=targets,
        endpoint=endpoint_value,
        candidates=selected,
        parser_aliases={item[0]: item[1].exact_aliases for item in selected},
        requested_candidate_count=len(validated_candidates),
        max_candidates=max_candidates,
        max_documents=max_documents,
    )


def _build_query(validated: _ValidatedInput) -> str:
    target_clause = " OR ".join(_pubmed_term(value) for value in validated.target_aliases)
    candidate_clause = " OR ".join(
        _pubmed_term(spec.preferred_search_alias)
        for _, spec in validated.candidates
    )
    return (
        f"({target_clause}) AND {_pubmed_term(validated.endpoint)} AND "
        f"({candidate_clause})"
    )


def _pubmed_term(value: str) -> str:
    return f'"{value}"[Title/Abstract]'


def _get_json(
    client: ScientificHttpClientLike,
    *,
    stage: Literal["esearch", "esummary"],
    endpoint: str,
    params: Mapping[str, Any],
) -> tuple[Mapping[str, Any], HttpCallProvenance]:
    response, provenance = _get_response(
        client, stage=stage, endpoint=endpoint, params=params
    )
    try:
        payload = response.json()
    except Exception as error:
        raise _StageFailure(f"{stage}_json_error", provenance) from error
    if not isinstance(payload, Mapping):
        raise _StageFailure(f"{stage}_payload_invalid", provenance)
    return payload, provenance


def _get_text(
    client: ScientificHttpClientLike,
    *,
    stage: Literal["efetch"],
    endpoint: str,
    params: Mapping[str, Any],
) -> tuple[str, HttpCallProvenance]:
    response, provenance = _get_response(
        client, stage=stage, endpoint=endpoint, params=params
    )
    try:
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            content = getattr(response, "content", None)
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="strict")
    except Exception as error:
        raise _StageFailure("efetch_text_invalid", provenance) from error
    if not isinstance(text, str):
        raise _StageFailure("efetch_text_missing", provenance)
    if len(text.encode("utf-8")) > _MAX_FETCH_BYTES:
        raise _StageFailure("efetch_body_bound_exceeded", provenance)
    return text, provenance


def _get_response(
    client: ScientificHttpClientLike,
    *,
    stage: Literal["esearch", "esummary", "efetch"],
    endpoint: str,
    params: Mapping[str, Any],
) -> tuple[Any, HttpCallProvenance]:
    if len(urlencode(dict(params)).encode("utf-8")) > _MAX_GET_QUERY_BYTES:
        raise _Reject("get_query_byte_bound_exceeded")
    parameter_sha256 = _parameter_hash(params)
    parameter_keys = tuple(sorted(f"params.{key}" for key in params))
    try:
        response = client.get(endpoint, params=dict(params))
    except Exception as error:
        provenance = _call_provenance(
            stage=stage,
            endpoint=endpoint,
            parameter_keys=parameter_keys,
            parameter_sha256=parameter_sha256,
            status="failed",
            status_code=None,
            response_sha256=None,
            error_type=type(error).__name__,
        )
        raise _StageFailure(f"{stage}_transport_failure", provenance) from error

    status_code_raw = getattr(response, "status_code", None)
    status_code = int(status_code_raw) if isinstance(status_code_raw, int) else None
    response_sha256 = hashlib.sha256(_response_bytes(response)).hexdigest()
    completed = status_code == 200
    provenance = _call_provenance(
        stage=stage,
        endpoint=endpoint,
        parameter_keys=parameter_keys,
        parameter_sha256=parameter_sha256,
        status="completed" if completed else "failed",
        status_code=status_code,
        response_sha256=response_sha256,
        error_type=None if completed else "HttpStatusError",
    )
    if not completed:
        raise _StageFailure(f"{stage}_http_failure", provenance)
    return response, provenance


def _call_provenance(
    *,
    stage: Literal["esearch", "esummary", "efetch"],
    endpoint: str,
    parameter_keys: tuple[str, ...],
    parameter_sha256: str,
    status: Literal["completed", "failed"],
    status_code: int | None,
    response_sha256: str | None,
    error_type: str | None,
) -> HttpCallProvenance:
    payload = {
        "endpoint": endpoint,
        "error_type": error_type,
        "method": "GET",
        "parameter_keys": parameter_keys,
        "parameter_sha256": parameter_sha256,
        "response_sha256": response_sha256,
        "schema": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "status_code": status_code,
    }
    return HttpCallProvenance(
        stage=stage,
        method="GET",
        endpoint=endpoint,
        parameter_keys=parameter_keys,
        parameter_sha256=parameter_sha256,
        status=status,
        status_code=status_code,
        response_sha256=response_sha256,
        error_type=error_type,
        call_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


def _search_batch(payload: Mapping[str, Any], *, max_documents: int) -> _SearchBatch:
    result = payload.get("esearchresult")
    if not isinstance(result, Mapping):
        raise _Reject("esearch_payload_invalid")
    if _has_payload_message(result.get("warninglist")):
        raise _Reject("esearch_warning")
    if _has_payload_message(result.get("errorlist")) or result.get("ERROR"):
        raise _Reject("esearch_error")
    count_raw = result.get("count")
    if isinstance(count_raw, bool) or not str(count_raw).strip().isdigit():
        raise _Reject("esearch_count_invalid")
    total_result_count = int(str(count_raw).strip())
    query_translation = result.get("querytranslation")
    if (
        not isinstance(query_translation, str)
        or not query_translation
        or query_translation != query_translation.strip()
        or len(query_translation) > 20_000
    ):
        raise _Reject("esearch_querytranslation_invalid")
    raw = result.get("idlist")
    if not isinstance(raw, list) or len(raw) > max_documents or len(raw) > _ABSOLUTE_MAX_DOCUMENTS:
        raise _Reject("esearch_document_bound_invalid")
    ids: list[str] = []
    for value in raw:
        document_id = str(value).strip()
        if not document_id.isdigit():
            raise _Reject("esearch_id_invalid")
        ids.append(document_id)
    if total_result_count < len(ids):
        raise _Reject("esearch_count_invalid")
    return _SearchBatch(
        document_ids=tuple(ids),
        total_result_count=total_result_count,
        query_translation=query_translation,
    )


def _has_payload_message(value: Any) -> bool:
    if value is None or value is False or value == "":
        return False
    if isinstance(value, Mapping):
        return any(_has_payload_message(item) for item in value.values())
    if isinstance(value, Collection) and not isinstance(value, (str, bytes)):
        return any(_has_payload_message(item) for item in value)
    return True


def _summary_items(
    payload: Mapping[str, Any], expected_ids: tuple[str, ...]
) -> dict[str, Mapping[str, Any]]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise _Reject("esummary_payload_invalid")
    raw_uids = result.get("uids")
    if not isinstance(raw_uids, list):
        raise _Reject("esummary_uids_invalid")
    uids = tuple(str(value).strip() for value in raw_uids)
    if any(not value.isdigit() for value in uids):
        raise _Reject("esummary_uids_invalid")
    if len(set(uids)) != len(uids):
        raise _Reject("esummary_uid_duplicate")
    if not set(uids).issubset(set(expected_ids)):
        raise _Reject("esummary_uid_unexpected")
    items: dict[str, Mapping[str, Any]] = {}
    for document_id in uids:
        item = result.get(document_id)
        if not isinstance(item, Mapping):
            continue
        item_uid = item.get("uid")
        if item_uid is not None and str(item_uid).strip() != document_id:
            raise _Reject("esummary_item_uid_mismatch")
        items[document_id] = item
    return items


def _publication_types(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Collection) and not isinstance(value, (str, bytes, Mapping)):
        raw = tuple(value)
    else:
        return ()
    if any(not isinstance(item, str) for item in raw):
        return ()
    return tuple(
        sorted(
            {
                unicodedata.normalize("NFKC", item).strip().casefold()
                for item in raw
                if item.strip()
            }
        )
    )


def _is_primary_non_review(publication_types: tuple[str, ...]) -> bool:
    return bool(publication_types) and not any(
        _EXCLUDED_PUBLICATION_RE.search(value) for value in publication_types
    ) and any(value in _PRIMARY_PUBLICATION_TYPES for value in publication_types)


def _parse_fetch_xml(
    text: str, *, expected_ids: frozenset[str]
) -> tuple[dict[str, _FetchedRecord], frozenset[str]]:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise _Reject("efetch_xml_invalid") from error
    records: dict[str, _FetchedRecord] = {}
    duplicates: set[str] = set()
    articles = [
        *root.findall("./PubmedArticle"),
        *root.findall("./PubmedBookArticle"),
    ]
    for article in articles:
        pmid_node = article.find("./MedlineCitation/PMID")
        if pmid_node is None:
            pmid_node = article.find(".//PMID")
        document_id = _element_text(pmid_node)
        if not document_id or not document_id.isdigit():
            raise _Reject("efetch_id_invalid")
        if document_id not in expected_ids:
            raise _Reject("efetch_id_unexpected")
        if document_id in records:
            duplicates.add(document_id)
            continue
        fetch_publication_types = _publication_types(
            tuple(
                value
                for element in article.findall(
                    ".//PublicationTypeList/PublicationType"
                )
                if (value := _element_text(element))
            )
        )
        article_title = _element_text(article.find(".//Article/ArticleTitle"))
        linked_status_notice_types, linked_status_notice_pmids = (
            _linked_status_notices(article)
        )
        method_sentences: list[tuple[int, str, str]] = []
        abstract_sentence_index = 0
        sections: list[tuple[str, str]] = []
        for element in article.findall(".//Abstract/AbstractText"):
            text_value = _element_text(element)
            if not text_value:
                continue
            for sentence in _sentences(text_value):
                indicator = _experimental_method_indicator(sentence)
                if indicator is not None:
                    method_sentences.append(
                        (abstract_sentence_index, sentence, indicator)
                    )
                abstract_sentence_index += 1
            category = str(element.attrib.get("NlmCategory") or "").strip().casefold()
            label = str(element.attrib.get("Label") or "").strip().casefold()
            if category in {"results", "result"}:
                section: Literal["Abstract", "Results"] = "Results"
            elif category in {"abstract", "unassigned"}:
                section = "Abstract"
            elif category:
                continue
            elif not label:
                section: Literal["Abstract", "Results"] = "Abstract"
            elif label in {"abstract", "results", "result", "findings"}:
                section = (
                    "Results" if label in {"results", "result", "findings"} else "Abstract"
                )
            else:
                continue
            sections.append((section, text_value))
        records[document_id] = _FetchedRecord(
            publication_types=fetch_publication_types,
            sections=tuple(sections),
            article_title=article_title,
            method_sentences=tuple(method_sentences),
            linked_status_notice_types=linked_status_notice_types,
            linked_status_notice_pmids=linked_status_notice_pmids,
        )
    return records, frozenset(duplicates)


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split()).strip()


def _linked_status_notices(
    article: ElementTree.Element,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    types: list[str] = []
    pmids: list[str] = []
    for element in article.findall(
        ".//CommentsCorrectionsList/CommentsCorrections"
    ):
        raw_type = element.attrib.get("RefType")
        if not isinstance(raw_type, str):
            raise _Reject("efetch_linked_notice_type_invalid")
        notice_type = unicodedata.normalize("NFKC", raw_type).strip().casefold()
        notice_key = re.sub(r"[^a-z]", "", notice_type)
        if notice_key not in _LINKED_STATUS_NOTICE_TYPES:
            continue
        if len(types) >= _MAX_LINKED_STATUS_NOTICES:
            raise _Reject("efetch_linked_notice_bound_exceeded")
        linked_pmid = _element_text(element.find("./PMID"))
        if linked_pmid and not linked_pmid.isdigit():
            raise _Reject("efetch_linked_notice_pmid_invalid")
        types.append(notice_key)
        pmids.append(linked_pmid)
    return tuple(types), tuple(pmids)


def _sentences(text: str) -> tuple[str, ...]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return ()
    return tuple(value.strip() for value in _SENTENCE_SPLIT_RE.split(normalized) if value.strip())


def _experimental_method_indicator(sentence: str) -> str | None:
    if (
        not sentence
        or sentence != sentence.strip()
        or len(sentence) > _MAX_SENTENCE_CHARS
        or "\r" in sentence
        or "\n" in sentence
        or _METHOD_UNSAFE_RE.search(sentence)
    ):
        return None
    if _METHOD_PASSIVE_RE.search(sentence):
        return "passive_experimental_measurement_method"
    if _METHOD_ACTIVE_RE.search(sentence):
        return "active_experimental_measurement_method"
    return None


def _primary_study_attestation(
    *,
    document_id: str,
    article_title: str,
    method_sentences: tuple[tuple[int, str, str], ...],
    summary_publication_types: tuple[str, ...],
    fetch_publication_types: tuple[str, ...],
    linked_status_notice_types: tuple[str, ...],
    linked_status_notice_pmids: tuple[str, ...],
) -> PrimaryStudyAttestation | None:
    if (
        not document_id.isdigit()
        or not article_title
        or article_title != article_title.strip()
        or len(article_title) > _MAX_ARTICLE_TITLE_CHARS
        or "\r" in article_title
        or "\n" in article_title
        or not _is_primary_non_review(summary_publication_types)
        or not _is_primary_non_review(fetch_publication_types)
        or not method_sentences
        or linked_status_notice_types
        or linked_status_notice_pmids
    ):
        return None
    sentence_index, sentence, indicator = method_sentences[0]
    if (
        type(sentence_index) is not int
        or sentence_index < 0
        or _experimental_method_indicator(sentence) != indicator
    ):
        return None
    payload = {
        "article_title": article_title,
        "article_title_sha256": _sha256_text(article_title),
        "document_id": document_id,
        "fetch_publication_types": fetch_publication_types,
        "fixture": False,
        "method_indicator": indicator,
        "method_sentence": sentence,
        "method_sentence_index": sentence_index,
        "method_sentence_sha256": _sha256_text(sentence),
        "policy": (
            "same_abstract_experimental_method_and_linked_notice_check_v2"
        ),
        "schema_version": "pubmed_primary_study_attestation/v2",
        "status": "verified",
        "summary_publication_types": summary_publication_types,
        "retraction_safety_policy": (
            "pubmed_types_and_comments_corrections_fail_closed_v1"
        ),
        "retraction_safe": True,
        "correction_safe": True,
        "linked_status_notice": "none_detected",
        "linked_status_notice_types": (),
        "linked_status_notice_pmids": (),
    }
    return PrimaryStudyAttestation(
        **payload,
        attestation_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


def _group(
    *,
    document_id: str,
    document_rank: int,
    section: str,
    sentence_index: int,
    sentence: str,
    primary_study_attestation: PrimaryStudyAttestation,
    summary_publication_types: tuple[str, ...],
    fetch_publication_types: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    pairs: tuple[PrimaryAbstractPair, ...],
    query_sha256: str,
) -> PrimaryAbstractGroup:
    sentence_sha256 = _sha256_text(sentence)
    payload = {
        "candidate_ids": candidate_ids,
        "document_id": document_id,
        "document_rank": document_rank,
        "grammar": PAPER_NATIVE_V5_GRAMMAR,
        "fetch_publication_types": fetch_publication_types,
        "pair_evidence_sha256": [pair.evidence_sha256 for pair in pairs],
        "primary_study_attestation_sha256": (
            primary_study_attestation.attestation_sha256
        ),
        "query_sha256": query_sha256,
        "schema": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "section": section,
        "sentence_index": sentence_index,
        "sentence_sha256": sentence_sha256,
        "summary_publication_types": summary_publication_types,
    }
    group_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return PrimaryAbstractGroup(
        document_id=document_id,
        document_rank=document_rank,
        section="Results" if section == "Results" else "Abstract",
        sentence_index=sentence_index,
        sentence=sentence,
        sentence_sha256=sentence_sha256,
        group_sha256=group_sha256,
        primary_study_attestation=primary_study_attestation,
        summary_publication_types=summary_publication_types,
        fetch_publication_types=fetch_publication_types,
        candidate_ids=candidate_ids,
        pairs=pairs,
    )


def _freeze_record(draft: _RecordDraft) -> RecordScreening:
    fields = {
        "bounded_sentence_count": draft.bounded_sentence_count,
        "document_id": draft.document_id,
        "eligible_section_count": draft.eligible_section_count,
        "fetch_publication_types": draft.fetch_publication_types,
        "publication_types": draft.publication_types,
        "rejection_codes": tuple(dict.fromkeys(draft.rejection_codes)),
        "schema": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "sentence_count": draft.sentence_count,
        "status": "verified" if draft.groups else "rejected",
        "verified_group_count": len(draft.groups),
    }
    return RecordScreening(
        document_id=draft.document_id,
        publication_types=draft.publication_types,
        fetch_publication_types=draft.fetch_publication_types,
        status="verified" if draft.groups else "rejected",
        rejection_codes=fields["rejection_codes"],
        eligible_section_count=draft.eligible_section_count,
        sentence_count=draft.sentence_count,
        bounded_sentence_count=draft.bounded_sentence_count,
        verified_group_count=len(draft.groups),
        record_sha256=hashlib.sha256(_canonical_json(fields)).hexdigest(),
    )


def _completed_outcome(
    *,
    validated: _ValidatedInput,
    status: Literal["verified", "not_verified"],
    reason_code: str | None,
    query: str,
    query_sha256: str,
    returned_document_count: int,
    unique_document_count: int,
    total_result_count: int,
    query_translation: str | None,
    query_translation_sha256: str | None,
    returned_document_ids: tuple[str, ...],
    returned_document_ids_sha256: str | None,
    calls: list[HttpCallProvenance],
    records: tuple[RecordScreening, ...],
    groups: tuple[PrimaryAbstractGroup, ...],
    screened_sentence_count: int,
) -> PrimaryAbstractSearchOutcome:
    verified_candidate_ids = tuple(
        dict.fromkeys(
            candidate_id for group in groups for candidate_id in group.candidate_ids
        )
    )
    return _finalize_outcome(
        schema_version=PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        policy_token=PAPER_NATIVE_V5_POLICY_TOKEN,
        grammar=PAPER_NATIVE_V5_GRAMMAR,
        status=status,
        reason_code=reason_code,
        query=query,
        query_sha256=query_sha256,
        query_translation=query_translation,
        query_translation_sha256=query_translation_sha256,
        search_sort="pub date",
        max_candidates=validated.max_candidates,
        max_documents=validated.max_documents,
        requested_candidate_count=validated.requested_candidate_count,
        queried_candidate_count=len(validated.candidates),
        requested_document_count=validated.max_documents,
        returned_document_count=returned_document_count,
        unique_document_count=unique_document_count,
        total_result_count=total_result_count,
        unsearched_result_count=max(total_result_count - returned_document_count, 0),
        returned_document_ids=returned_document_ids,
        returned_document_ids_sha256=returned_document_ids_sha256,
        screened_document_count=len(records),
        screened_sentence_count=screened_sentence_count,
        verified_candidate_ids=verified_candidate_ids,
        calls=tuple(calls),
        records=records,
        groups=groups,
        selected_group_sha256=(groups[0].group_sha256 if groups else None),
    )


def _failed_after_query(
    validated: _ValidatedInput,
    query: str,
    query_sha256: str,
    calls: list[HttpCallProvenance],
    reason_code: str,
    *,
    returned_document_count: int = 0,
    unique_document_count: int = 0,
    total_result_count: int = 0,
    query_translation: str | None = None,
    query_translation_sha256: str | None = None,
    returned_document_ids: tuple[str, ...] = (),
    returned_document_ids_sha256: str | None = None,
) -> PrimaryAbstractSearchOutcome:
    return _finalize_outcome(
        schema_version=PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        policy_token=PAPER_NATIVE_V5_POLICY_TOKEN,
        grammar=PAPER_NATIVE_V5_GRAMMAR,
        status="failed",
        reason_code=reason_code,
        query=query,
        query_sha256=query_sha256,
        query_translation=query_translation,
        query_translation_sha256=query_translation_sha256,
        search_sort="pub date",
        max_candidates=validated.max_candidates,
        max_documents=validated.max_documents,
        requested_candidate_count=validated.requested_candidate_count,
        queried_candidate_count=len(validated.candidates),
        requested_document_count=validated.max_documents,
        returned_document_count=returned_document_count,
        unique_document_count=unique_document_count,
        total_result_count=total_result_count,
        unsearched_result_count=max(total_result_count - returned_document_count, 0),
        returned_document_ids=returned_document_ids,
        returned_document_ids_sha256=returned_document_ids_sha256,
        screened_document_count=0,
        screened_sentence_count=0,
        verified_candidate_ids=(),
        calls=tuple(calls),
        records=(),
        groups=(),
        selected_group_sha256=None,
    )


def _empty_outcome(
    *,
    status: Literal["failed"],
    reason_code: str,
    query: None,
    max_candidates: int,
    max_documents: int,
    requested_candidate_count: int,
) -> PrimaryAbstractSearchOutcome:
    return _finalize_outcome(
        schema_version=PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        policy_token=PAPER_NATIVE_V5_POLICY_TOKEN,
        grammar=PAPER_NATIVE_V5_GRAMMAR,
        status=status,
        reason_code=reason_code,
        query=query,
        query_sha256=None,
        query_translation=None,
        query_translation_sha256=None,
        search_sort="pub date",
        max_candidates=max_candidates,
        max_documents=max_documents,
        requested_candidate_count=requested_candidate_count,
        queried_candidate_count=0,
        requested_document_count=max_documents,
        returned_document_count=0,
        unique_document_count=0,
        total_result_count=0,
        unsearched_result_count=0,
        returned_document_ids=(),
        returned_document_ids_sha256=None,
        screened_document_count=0,
        screened_sentence_count=0,
        verified_candidate_ids=(),
        calls=(),
        records=(),
        groups=(),
        selected_group_sha256=None,
    )


def _finalize_outcome(**fields: Any) -> PrimaryAbstractSearchOutcome:
    """Hash the complete immutable outcome without recursively hashing itself."""

    provisional = PrimaryAbstractSearchOutcome(outcome_sha256="", **fields)
    payload = asdict(provisional)
    payload.pop("outcome_sha256", None)
    return PrimaryAbstractSearchOutcome(
        outcome_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        **fields,
    )


def _exact_strings(
    values: Collection[str],
    *,
    maximum_count: int,
    maximum_chars: int,
    require_tuple: bool = False,
) -> tuple[str, ...]:
    if require_tuple and not isinstance(values, tuple):
        raise _Reject("exact_aliases_not_tuple")
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise _Reject("invalid_exact_strings")
    raw = tuple(values)
    if not raw or len(raw) > maximum_count:
        raise _Reject("exact_string_bound_invalid")
    cleaned = tuple(_bounded_string(value, maximum_chars=maximum_chars) for value in raw)
    keys = [_key(value) for value in cleaned]
    if len(set(keys)) != len(keys):
        raise _Reject("alias_conflict")
    return tuple(sorted(cleaned, key=lambda value: (_key(value), value)))


def _validate_single_attempt_client(client: ScientificHttpClientLike) -> None:
    """Make the three logical call bound an actual wire-attempt bound.

    ``ScientificHttpClient`` retries inside ``get``.  The adapter therefore
    accepts only an explicitly single-attempt instance; otherwise three stages
    could silently become six wire requests and the call provenance would be
    incomplete.
    """

    if not callable(getattr(client, "get", None)):
        raise _Reject("invalid_http_client")
    if type(getattr(client, "max_attempts", None)) is not int:
        raise _Reject("client_retry_policy_unattested")
    if client.max_attempts != 1:
        raise _Reject("client_retry_policy_not_single_attempt")


def _bounded_string(value: Any, *, maximum_chars: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _Reject("invalid_string")
    if len(value) > maximum_chars or "\r" in value or "\n" in value:
        raise _Reject("string_bound_invalid")
    return value


def _validate_query_term(value: str) -> None:
    if '"' in value or any(ord(character) < 32 for character in value):
        raise _Reject("unsafe_query_term")


def _key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _parameter_hash(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json({"params": dict(params)})).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    try:
        return _canonical_json(response.json())
    except Exception:
        return b""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_sequence(values: tuple[str, ...]) -> str:
    return hashlib.sha256(
        _canonical_json({"ordered_values": values})
    ).hexdigest()


def _safe_limit(value: Any, maximum: int) -> int:
    return value if type(value) is int and 1 <= value <= maximum else 0
