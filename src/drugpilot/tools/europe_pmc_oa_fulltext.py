"""Pure, bounded Europe PMC OA full-text and reference parsing utilities.

This module has no network boundary.  Callers fetch official Europe PMC
responses, attest article eligibility from independently retrieved metadata,
and pass the bounded response body here.  Only short sentence excerpts and
cryptographic hashes are returned; raw JATS/full-text content is never retained
by any result model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Collection, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
from xml.etree import ElementTree


MAX_OA_JATS_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_EXCERPT_CHARS = 320
DEFAULT_MAX_SENTENCE_SPANS = 64
DEFAULT_MAX_REFERENCES = 64
MAX_JATS_ELEMENTS = 100_000
MAX_JATS_DEPTH = 128

RETRIEVAL_POLICY = "epmc_oa_fulltext_one_hop_v1"
EXCERPT_SELECTION_POLICY = "europe_pmc_oa_jats_results_v1"
REFERENCE_POLICY = "europe_pmc_reference_one_hop_v1"
XML_PARSER_POLICY = "external_jats_doctype_stripped_no_dtd_no_entities_v1"

_PMCID_PATTERN = re.compile(r"PMC[0-9]+", re.I)
_PMID_PATTERN = re.compile(r"[0-9]+")
_DOI_PATTERN = re.compile(r"10\.[0-9]{4,9}/\S+", re.I)
_SAFE_EXTERNAL_JATS_DOCTYPE_PATTERN = re.compile(
    r"<!DOCTYPE\s+article\s+(?:"
    r"SYSTEM\s+(?:\"[^\"<>\[\]]+\"|'[^'<>\[\]]+')|"
    r"PUBLIC\s+(?:\"[^\"<>\[\]]+\"|'[^'<>\[\]]+')\s+"
    r"(?:\"[^\"<>\[\]]+\"|'[^'<>\[\]]+')"
    r")\s*>",
    re.I | re.S,
)
_XML_DECLARATION_ONLY_PATTERN = re.compile(
    r"\s*(?:<\?xml\s+[^?]*\?>\s*)?\Z",
    re.I | re.S,
)
_SENTENCE_PATTERN = re.compile(
    r"(?:\d+\.\d+|[pP]\.[A-Za-z][0-9]{1,5}[A-Za-z*](?![A-Za-z0-9])|"
    r"[^.!?])+(?:[.!?]+|$)"
)
_ALLOWED_SECTION_PATTERN = re.compile(
    r"\b(?:results?|findings?|case(?:[- ](?:presentation|report|description|"
    r"study|history|findings?))?)\b",
    re.I,
)
_EXCLUDED_SECTION_PATTERN = re.compile(
    r"\b(?:intro(?:duction)?|background|discussion|methods?|materials?|"
    r"conclusions?|references?|supplement(?:ary)?|figures?|tables?|captions?|"
    r"acknowledg(?:e)?ments?)\b",
    re.I,
)
_RESEARCH_PUBLICATION_TYPE_PATTERN = re.compile(
    r"\b(?:journal article|research[- ]article|clinical trial|randomized controlled|"
    r"comparative study|evaluation study|validation study|multicenter study|"
    r"observational study|preclinical study|case reports?)\b",
    re.I,
)
_DISALLOWED_PUBLICATION_TYPE_PATTERN = re.compile(
    r"\b(?:review|meta[- ]?analysis|guideline|consensus|editorial|commentary|"
    r"letter|correction|retract(?:ed|ion)|expression of concern)\b",
    re.I,
)
_BLOCKED_CONTENT_TAGS = frozenset(
    {
        "ack",
        "app",
        "app-group",
        "boxed-text",
        "caption",
        "fig",
        "fig-group",
        "fn",
        "fn-group",
        "notes",
        "ref-list",
        "supplementary-material",
        "supplementary-materials",
        "table",
        "table-wrap",
        "table-wrap-group",
    }
)
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_LICENSE_LABEL_PATHS: Mapping[str, str] = {
    "cc by": "/licenses/by/",
    "cc by-sa": "/licenses/by-sa/",
    "cc by-nd": "/licenses/by-nd/",
    "cc by-nc": "/licenses/by-nc/",
    "cc by-nc-sa": "/licenses/by-nc-sa/",
    "cc by-nc-nd": "/licenses/by-nc-nd/",
    "cc0": "/publicdomain/zero/",
}


class EuropePMCOAValidationError(ValueError):
    """Fail-closed rejection carrying one stable, audit-safe reason code."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True, slots=True)
class OAArticleEligibility:
    """Caller-attested publication metadata required before JATS admission."""

    pmcid: str
    pmid: str | None
    publication_types: tuple[str, ...]
    is_open_access: bool
    license_uri: str
    is_retracted: bool = False
    doi: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pmcid", _canonical_pmcid(self.pmcid))
        object.__setattr__(
            self,
            "pmid",
            _canonical_pmid(self.pmid) if self.pmid is not None else None,
        )
        object.__setattr__(
            self,
            "publication_types",
            tuple(
                dict.fromkeys(
                    text
                    for value in self.publication_types
                    if (text := _normalize_text(str(value)))
                )
            ),
        )
        object.__setattr__(self, "license_uri", str(self.license_uri).strip())
        object.__setattr__(
            self,
            "doi",
            _canonical_doi(self.doi) if self.doi is not None else None,
        )


@dataclass(frozen=True, slots=True)
class OneHopReference:
    """Identifier-only reference edge; it never inherits an evidence role."""

    parent_pmcid: str
    parent_pmid: str | None
    reference_position: int
    reference_id: str
    pmcid: str | None
    pmid: str | None
    doi: str | None
    canonical_id: str
    identity_keys: tuple[str, ...]
    citation_lineage_sha256: str
    citation_hop: int = 1
    citation_relation: str = "reference"

    def to_lineage_metadata(self) -> dict[str, Any]:
        parent_external_id = self.parent_pmid or self.parent_pmcid
        return {
            "source_database": "Europe PMC",
            "discovery_origin": "reference_one_hop",
            "citation_hop": 1,
            "citation_depth": 1,
            "citation_relation": "reference",
            "citation_parent_external_id": parent_external_id,
            "citation_parent_pmcid": self.parent_pmcid,
            "citation_parent_pmid": self.parent_pmid,
            "citation_reference_position": self.reference_position,
            "citation_child_canonical_id": self.canonical_id,
            "citation_child_identity_keys": list(self.identity_keys),
            "citation_child_pmcid": self.pmcid,
            "citation_child_pmid": self.pmid,
            "citation_child_doi": self.doi,
            "citation_root_external_id": parent_external_id,
            "citation_lineage_sha256": self.citation_lineage_sha256,
            "lineage_verified": True,
            "evidence_role_inherited": False,
            "independent_primary_verification_required": True,
        }


@dataclass(frozen=True, slots=True)
class OAFullTextSpan:
    """One bounded sentence window from an admitted JATS body section."""

    text: str
    section_kind: str
    section_title: str
    section_index: int
    paragraph_index: int
    sentence_index: int
    char_start: int
    char_end: int
    source_location: str
    excerpt_sha256: str
    excerpt_truncated: bool
    original_sentence_char_count: int


@dataclass(frozen=True, slots=True)
class OAFullTextAudit:
    """Serializable document attestation that deliberately omits raw content."""

    policy_version: str
    pmcid: str
    pmid: str | None
    doi: str | None
    publication_types: tuple[str, ...]
    license_uri: str
    license_attestation_source: str
    jats_license_present: bool
    input_bytes: int
    document_sha256: str
    eligible_content_sha256: str
    eligible_section_count: int
    eligible_paragraph_count: int
    emitted_span_count: int
    spans_truncated_by_limit: bool
    external_doctype_removed: bool
    full_text_stored: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_database": "Europe PMC",
            "content_retrieval_policy": RETRIEVAL_POLICY,
            "excerpt_selection_policy": self.policy_version,
            "pmcid": self.pmcid,
            "pmid": self.pmid,
            "doi": self.doi,
            "publication_types": list(self.publication_types),
            "is_open_access": True,
            "full_text_license": self.license_uri,
            "full_text_license_attestation_source": self.license_attestation_source,
            "jats_license_present": self.jats_license_present,
            "publication_status": "primary_research_article",
            "input_byte_count": self.input_bytes,
            "content_hash": self.document_sha256,
            "full_text_content_hash": self.document_sha256,
            "content_hash_algorithm": "sha256",
            "content_hash_scope": "europe_pmc_oa_jats_xml",
            "xml_parser_policy": XML_PARSER_POLICY,
            "external_doctype_removed": self.external_doctype_removed,
            "eligible_content_hash": self.eligible_content_sha256,
            "eligible_section_count": self.eligible_section_count,
            "eligible_paragraph_count": self.eligible_paragraph_count,
            "emitted_span_count": self.emitted_span_count,
            "spans_truncated_by_limit": self.spans_truncated_by_limit,
            "full_text_stored": False,
            "content_storage_policy": "bounded_oa_excerpt_only_no_full_text",
        }


@dataclass(frozen=True, slots=True)
class OAFullTextResult:
    """Bounded parse result; only short spans and reference identifiers remain."""

    audit: OAFullTextAudit
    spans: tuple[OAFullTextSpan, ...]
    references: tuple[OneHopReference, ...]

    def to_audit_metadata(self) -> dict[str, Any]:
        return {
            **self.audit.to_metadata(),
            "reference_policy": REFERENCE_POLICY,
            "reference_count": len(self.references),
            "raw_full_text_retained": False,
        }

    def evidence_metadata(
        self,
        span: OAFullTextSpan,
        *,
        lineage: OneHopReference | None = None,
    ) -> dict[str, Any]:
        if span not in self.spans:
            raise EuropePMCOAValidationError(
                "unknown_excerpt_span",
                "the supplied excerpt span does not belong to this parse result",
            )
        article_keys = document_identity_keys(
            pmcid=self.audit.pmcid,
            pmid=self.audit.pmid,
            doi=self.audit.doi,
        )
        if lineage is not None:
            identifiers_match = (
                (lineage.pmcid is None or lineage.pmcid == self.audit.pmcid)
                and (lineage.pmid is None or lineage.pmid == self.audit.pmid)
                and (lineage.doi is None or lineage.doi == self.audit.doi)
            )
            if article_keys.isdisjoint(lineage.identity_keys) or not identifiers_match:
                raise EuropePMCOAValidationError(
                    "citation_child_identity_mismatch",
                    "reference identifiers do not resolve to the parsed child article",
                )
            lineage_metadata = lineage.to_lineage_metadata()
            evidence_origin = "oa_results_fulltext_reference_one_hop"
        else:
            canonical_id = _preferred_canonical_id(
                pmcid=self.audit.pmcid,
                pmid=self.audit.pmid,
                doi=self.audit.doi,
            )
            lineage_payload = _canonical_json_bytes(
                {
                    "policy": RETRIEVAL_POLICY,
                    "hop": 0,
                    "relation": "direct_search",
                    "child": canonical_id,
                }
            )
            lineage_metadata = {
                "source_database": "Europe PMC",
                "discovery_origin": "direct_search",
                "citation_hop": 0,
                "citation_depth": 0,
                "citation_relation": "direct_search",
                "citation_parent_external_id": None,
                "citation_parent_pmcid": None,
                "citation_parent_pmid": None,
                "citation_reference_position": None,
                "citation_child_canonical_id": canonical_id,
                "citation_child_identity_keys": sorted(article_keys),
                "citation_child_pmcid": self.audit.pmcid,
                "citation_child_pmid": self.audit.pmid,
                "citation_child_doi": self.audit.doi,
                "citation_root_external_id": canonical_id,
                "citation_lineage_sha256": sha256(lineage_payload).hexdigest(),
                "lineage_verified": True,
                "evidence_role_inherited": False,
                "independent_primary_verification_required": True,
            }
            evidence_origin = "oa_results_fulltext_direct"
        return {
            **self.audit.to_metadata(),
            **lineage_metadata,
            "evidence_origin": evidence_origin,
            "content_availability": "oa_fulltext_excerpt",
            "source_location": span.source_location,
            "source_location_scope": "normalized_admitted_paragraph_text",
            "source_location_char_end_exclusive": True,
            "excerpt": span.text,
            "excerpt_hash": span.excerpt_sha256,
            "excerpt_char_count": len(span.text),
            "excerpt_max_chars": MAX_EXCERPT_CHARS,
            "excerpt_truncated": span.excerpt_truncated,
            "original_sentence_char_count": span.original_sentence_char_count,
            "section_kind": span.section_kind,
            "section_title": span.section_title,
            "independent_primary_verification": True,
        }


@dataclass(frozen=True, slots=True)
class ReferenceParseAudit:
    policy_version: str
    parent_pmcid: str
    parent_pmid: str | None
    input_bytes: int
    response_sha256: str
    raw_reference_count: int
    emitted_reference_count: int
    references_truncated_by_limit: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_database": "Europe PMC",
            "discovery_origin": "reference_one_hop",
            "reference_policy": self.policy_version,
            "citation_hop_limit": 1,
            "citation_parent_pmcid": self.parent_pmcid,
            "citation_parent_pmid": self.parent_pmid,
            "reference_input_byte_count": self.input_bytes,
            "reference_response_sha256": self.response_sha256,
            "raw_reference_count": self.raw_reference_count,
            "emitted_reference_count": self.emitted_reference_count,
            "references_truncated_by_limit": self.references_truncated_by_limit,
            "raw_reference_payload_retained": False,
        }


@dataclass(frozen=True, slots=True)
class OneHopReferenceResult:
    audit: ReferenceParseAudit
    references: tuple[OneHopReference, ...]

    def to_audit_metadata(self) -> dict[str, Any]:
        return self.audit.to_metadata()


@dataclass(frozen=True, slots=True)
class _EligibleParagraph:
    section_kind: str
    section_title: str
    section_index: int
    paragraph_index: int
    text: str


def parse_oa_jats(
    xml: bytes | str,
    *,
    eligibility: OAArticleEligibility,
    allowed_license_uris: Collection[str],
    max_spans: int = DEFAULT_MAX_SENTENCE_SPANS,
) -> OAFullTextResult:
    """Validate and parse bounded official Europe PMC OA JATS content."""

    if not 1 <= max_spans <= 256:
        raise EuropePMCOAValidationError(
            "invalid_span_limit",
            "max_spans must be between 1 and 256",
        )
    _validate_eligibility(eligibility, allowed_license_uris)
    raw = _bounded_utf8_bytes(xml, maximum=MAX_OA_JATS_BYTES, payload_name="JATS XML")
    parseable, external_doctype_removed = _prepare_xml(raw)
    try:
        root = ElementTree.fromstring(parseable)
    except ElementTree.ParseError as exc:
        raise EuropePMCOAValidationError(
            "invalid_jats_xml",
            "JATS response is not well-formed XML",
        ) from exc
    _guard_tree_shape(root)

    article_pmcid, article_pmid, article_doi = _article_identifiers(root)
    if article_pmcid != eligibility.pmcid:
        raise EuropePMCOAValidationError(
            "pmcid_mismatch",
            "JATS PMCID does not exactly match caller-attested metadata",
        )
    if eligibility.pmid is not None and article_pmid != eligibility.pmid:
        raise EuropePMCOAValidationError(
            "pmid_mismatch",
            "JATS PMID does not exactly match caller-attested metadata",
        )
    if eligibility.doi is not None and article_doi != eligibility.doi:
        raise EuropePMCOAValidationError(
            "doi_mismatch",
            "JATS DOI does not exactly match caller-attested metadata",
        )
    resolved_license_uri, license_attestation_source = _validate_jats_license(
        root,
        eligibility.license_uri,
    )

    paragraphs, eligible_section_count = _eligible_paragraphs(root)
    spans, truncated_by_limit = _sentence_spans(paragraphs, max_spans=max_spans)
    eligible_payload = "\n".join(item.text for item in paragraphs).encode("utf-8")
    audit = OAFullTextAudit(
        policy_version=EXCERPT_SELECTION_POLICY,
        pmcid=article_pmcid,
        pmid=article_pmid,
        doi=article_doi,
        publication_types=eligibility.publication_types,
        license_uri=resolved_license_uri,
        license_attestation_source=license_attestation_source,
        jats_license_present=(license_attestation_source != "europe_pmc_metadata"),
        input_bytes=len(raw),
        document_sha256=sha256(raw).hexdigest(),
        eligible_content_sha256=sha256(eligible_payload).hexdigest(),
        eligible_section_count=eligible_section_count,
        eligible_paragraph_count=len(paragraphs),
        emitted_span_count=len(spans),
        spans_truncated_by_limit=truncated_by_limit,
        external_doctype_removed=external_doctype_removed,
    )
    references = _jats_references(
        root,
        parent_pmcid=article_pmcid,
        parent_pmid=article_pmid,
        max_references=DEFAULT_MAX_REFERENCES,
        visited_keys=document_identity_keys(
            pmcid=article_pmcid,
            pmid=article_pmid,
            doi=article_doi,
        ),
    )
    return OAFullTextResult(
        audit=audit,
        spans=spans,
        references=references,
    )


def parse_europe_pmc_references(
    payload: bytes | str | Mapping[str, Any],
    *,
    parent_pmcid: str,
    parent_pmid: str | None = None,
    max_references: int = DEFAULT_MAX_REFERENCES,
    visited_keys: Collection[str] = (),
) -> OneHopReferenceResult:
    """Parse a bounded Europe PMC ``/references`` JSON response."""

    if not 1 <= max_references <= 256:
        raise EuropePMCOAValidationError(
            "invalid_reference_limit",
            "max_references must be between 1 and 256",
        )
    canonical_pmcid = _canonical_pmcid(parent_pmcid)
    canonical_pmid = (
        _canonical_pmid(parent_pmid) if parent_pmid is not None else None
    )
    if isinstance(payload, Mapping):
        raw = _canonical_json_bytes(payload)
        parsed: Any = payload
    else:
        raw = _bounded_utf8_bytes(
            payload,
            maximum=MAX_REFERENCE_PAYLOAD_BYTES,
            payload_name="reference payload",
        )
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EuropePMCOAValidationError(
                "invalid_reference_payload",
                "Europe PMC reference response is not valid UTF-8 JSON",
            ) from exc
    if len(raw) > MAX_REFERENCE_PAYLOAD_BYTES:
        raise EuropePMCOAValidationError(
            "reference_payload_too_large",
            f"reference payload exceeds {MAX_REFERENCE_PAYLOAD_BYTES} bytes",
        )
    if not isinstance(parsed, Mapping):
        raise EuropePMCOAValidationError(
            "invalid_reference_payload",
            "Europe PMC reference response must be a JSON object",
        )
    container = parsed.get("referenceList", {})
    values = container.get("reference", []) if isinstance(container, Mapping) else []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EuropePMCOAValidationError(
            "invalid_reference_payload",
            "referenceList.reference must be an array",
        )
    raw_references: list[OneHopReference] = []
    for index, item in enumerate(values, 1):
        if not isinstance(item, Mapping):
            continue
        position = _positive_int(item.get("citedOrder"), default=index)
        source = _normalize_text(str(item.get("source") or "")).upper()
        external_id = _normalize_text(str(item.get("id") or ""))
        pmid = _optional_pmid(item.get("pmid"))
        pmcid = _optional_pmcid(item.get("pmcid"))
        if source == "MED" and external_id:
            pmid = _optional_pmid(external_id) or pmid
        elif source == "PMC" and external_id:
            pmcid = _optional_pmcid(external_id) or pmcid
        doi = _optional_doi(item.get("doi"))
        reference = _make_reference(
            parent_pmcid=canonical_pmcid,
            parent_pmid=canonical_pmid,
            reference_position=position,
            reference_id=str(item.get("refId") or item.get("id") or position),
            pmcid=pmcid,
            pmid=pmid,
            doi=doi,
        )
        if reference is not None:
            raw_references.append(reference)
    deduplicated = deduplicate_one_hop_references(raw_references)
    blocked = {
        str(value).strip().casefold()
        for value in (
            *visited_keys,
            *document_identity_keys(pmcid=canonical_pmcid, pmid=canonical_pmid),
        )
        if str(value).strip()
    }
    acyclic = exclude_visited_references(deduplicated, blocked)
    selected = acyclic[:max_references]
    audit = ReferenceParseAudit(
        policy_version=REFERENCE_POLICY,
        parent_pmcid=canonical_pmcid,
        parent_pmid=canonical_pmid,
        input_bytes=len(raw),
        response_sha256=sha256(raw).hexdigest(),
        raw_reference_count=len(raw_references),
        emitted_reference_count=len(selected),
        references_truncated_by_limit=len(acyclic) > len(selected),
    )
    return OneHopReferenceResult(audit=audit, references=selected)


def document_identity_keys(
    *,
    pmcid: str | None = None,
    pmid: str | None = None,
    doi: str | None = None,
) -> frozenset[str]:
    """Return normalized cross-database keys for deduplication/cycle checks."""

    keys: set[str] = set()
    if pmid is not None:
        keys.add(f"MED:{_canonical_pmid(pmid)}")
    if pmcid is not None:
        keys.add(f"PMC:{_canonical_pmcid(pmcid)}")
    if doi is not None:
        normalized_doi = _canonical_doi(doi)
        keys.add(f"DOI:{normalized_doi}")
    return frozenset(keys)


def deduplicate_one_hop_references(
    references: Iterable[OneHopReference],
) -> tuple[OneHopReference, ...]:
    """Keep the first cited occurrence and merge later identifier aliases."""

    groups: list[list[OneHopReference]] = []
    for candidate in references:
        candidate_keys = set(candidate.identity_keys)
        matching = [
            index
            for index, group in enumerate(groups)
            if candidate_keys.intersection(
                key for item in group for key in item.identity_keys
            )
        ]
        if not matching:
            groups.append([candidate])
            continue
        combined = [candidate]
        for index in reversed(matching):
            combined.extend(groups.pop(index))
        groups.append(combined)

    merged = [_merge_reference_group(group) for group in groups]
    return tuple(
        sorted(
            merged,
            key=lambda item: (item.reference_position, item.canonical_id),
        )
    )


def _merge_reference_group(
    group: Sequence[OneHopReference],
) -> OneHopReference:
    ordered = sorted(
        group,
        key=lambda item: (item.reference_position, item.reference_id),
    )
    first = ordered[0]
    parents = {(item.parent_pmcid, item.parent_pmid) for item in ordered}
    if len(parents) != 1:
        raise EuropePMCOAValidationError(
            "reference_parent_mismatch",
            "deduplicated reference aliases must share one parent article",
        )

    def unique_identifier(name: str) -> str | None:
        values = {
            value
            for item in ordered
            if (value := getattr(item, name)) is not None
        }
        if len(values) > 1:
            raise EuropePMCOAValidationError(
                "reference_identifier_conflict",
                f"deduplicated reference aliases contain conflicting {name} values",
            )
        return next(iter(values), None)

    replacement = _make_reference(
        parent_pmcid=first.parent_pmcid,
        parent_pmid=first.parent_pmid,
        reference_position=first.reference_position,
        reference_id=first.reference_id,
        pmcid=unique_identifier("pmcid"),
        pmid=unique_identifier("pmid"),
        doi=unique_identifier("doi"),
    )
    assert replacement is not None
    return replacement


def exclude_visited_references(
    references: Iterable[OneHopReference],
    visited_keys: Collection[str],
) -> tuple[OneHopReference, ...]:
    """Remove parent/self/already-seen documents; references cannot recurse."""

    normalized_visited = {
        str(value).strip().casefold()
        for value in visited_keys
        if str(value).strip()
    }
    return tuple(
        item
        for item in references
        if normalized_visited.isdisjoint(
            str(value).casefold() for value in item.identity_keys
        )
        and item.citation_hop == 1
    )


def _validate_eligibility(
    eligibility: OAArticleEligibility,
    allowed_license_uris: Collection[str],
) -> None:
    if not eligibility.is_open_access:
        raise EuropePMCOAValidationError(
            "not_open_access",
            "caller metadata does not attest Europe PMC open access",
        )
    if eligibility.is_retracted:
        raise EuropePMCOAValidationError(
            "retracted_publication",
            "retracted publications cannot supply primary evidence",
        )
    if not eligibility.publication_types:
        raise EuropePMCOAValidationError(
            "publication_type_missing",
            "caller must supply publication types",
        )
    if any(
        _DISALLOWED_PUBLICATION_TYPE_PATTERN.search(value)
        for value in eligibility.publication_types
    ):
        raise EuropePMCOAValidationError(
            "non_primary_publication_type",
            "review, synthesis, correction, or retraction types are not admitted",
        )
    if not any(
        _RESEARCH_PUBLICATION_TYPE_PATTERN.search(value)
        for value in eligibility.publication_types
    ):
        raise EuropePMCOAValidationError(
            "research_publication_type_not_verified",
            "a primary research-article publication type is required",
        )
    allowed = {str(value).strip() for value in allowed_license_uris if str(value).strip()}
    if not eligibility.license_uri or eligibility.license_uri not in allowed:
        raise EuropePMCOAValidationError(
            "license_not_allowed",
            "caller-attested OA license is absent from the exact allow-list",
        )


def _bounded_utf8_bytes(
    payload: bytes | str,
    *,
    maximum: int,
    payload_name: str,
) -> bytes:
    try:
        raw = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EuropePMCOAValidationError(
            "unsupported_payload_encoding",
            f"{payload_name} must be UTF-8",
        ) from exc
    if len(raw) > maximum:
        reason = (
            "jats_payload_too_large"
            if payload_name == "JATS XML"
            else "reference_payload_too_large"
        )
        raise EuropePMCOAValidationError(
            reason,
            f"{payload_name} exceeds {maximum} bytes",
        )
    try:
        raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EuropePMCOAValidationError(
            "unsupported_payload_encoding",
            f"{payload_name} must be UTF-8",
        ) from exc
    return raw


def _prepare_xml(raw: bytes) -> tuple[bytes, bool]:
    """Return parser-safe XML without loading an external JATS DTD.

    Europe PMC legitimately emits some OA JATS documents with a declaration
    such as ``<!DOCTYPE article PUBLIC ... \"JATS-...dtd\">``.  The stdlib
    parser does not need that DTD for the bounded fields consumed here.  We
    therefore admit exactly one external ``article`` declaration in the XML
    prolog, remove it before parsing, and continue to reject internal subsets,
    entity declarations, misplaced declarations, and every other DOCTYPE
    shape.  The audit hash and byte count remain bound to the original response.
    """

    decoded = raw.decode("utf-8-sig")
    folded = decoded.casefold()
    if "<!entity" in folded:
        raise EuropePMCOAValidationError(
            "jats_entity_forbidden",
            "ENTITY declarations are forbidden",
        )
    doctype_offsets = [
        match.start() for match in re.finditer(r"<!doctype", decoded, re.I)
    ]
    if not doctype_offsets:
        return raw, False
    matches = list(_SAFE_EXTERNAL_JATS_DOCTYPE_PATTERN.finditer(decoded))
    if len(doctype_offsets) != 1 or len(matches) != 1:
        raise EuropePMCOAValidationError(
            "jats_doctype_forbidden",
            "only one external article DOCTYPE is permitted",
        )
    declaration = matches[0]
    if declaration.start() != doctype_offsets[0]:
        raise EuropePMCOAValidationError(
            "jats_doctype_forbidden",
            "DOCTYPE declaration is not an admitted external JATS declaration",
        )
    prefix = decoded[: declaration.start()].lstrip("\ufeff")
    suffix = decoded[declaration.end() :]
    if (
        _XML_DECLARATION_ONLY_PATTERN.fullmatch(prefix) is None
        or re.match(r"\s*<article(?:\s|>)", suffix, re.I) is None
    ):
        raise EuropePMCOAValidationError(
            "jats_doctype_forbidden",
            "external JATS DOCTYPE must appear once in the XML prolog",
        )
    parseable = decoded[: declaration.start()] + suffix
    return parseable.encode("utf-8"), True


def _guard_tree_shape(root: ElementTree.Element) -> None:
    element_count = 0
    stack: list[tuple[ElementTree.Element, int]] = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        element_count += 1
        if element_count > MAX_JATS_ELEMENTS:
            raise EuropePMCOAValidationError(
                "jats_element_limit_exceeded",
                f"JATS contains more than {MAX_JATS_ELEMENTS} elements",
            )
        if depth > MAX_JATS_DEPTH:
            raise EuropePMCOAValidationError(
                "jats_depth_limit_exceeded",
                f"JATS nesting exceeds {MAX_JATS_DEPTH} elements",
            )
        stack.extend((child, depth + 1) for child in element)


def _find_article_meta(root: ElementTree.Element) -> ElementTree.Element:
    for element in root.iter():
        if _local_name(element.tag) == "article-meta":
            return element
    raise EuropePMCOAValidationError(
        "article_metadata_missing",
        "JATS article-meta was not found",
    )


def _article_identifiers(
    root: ElementTree.Element,
) -> tuple[str, str | None, str | None]:
    article_meta = _find_article_meta(root)
    identifiers: dict[str, set[str]] = {
        "pmc": set(),
        "pmid": set(),
        "doi": set(),
    }
    for element in article_meta.iter():
        if _local_name(element.tag) != "article-id":
            continue
        identifier_type = str(element.attrib.get("pub-id-type") or "").casefold()
        text = _normalize_text("".join(element.itertext()))
        if not text:
            continue
        if identifier_type in {"pmc", "pmcid"}:
            identifiers["pmc"].add(_canonical_pmcid(text))
        elif identifier_type in {"pmid", "pubmed"}:
            identifiers["pmid"].add(_canonical_pmid(text))
        elif identifier_type == "doi":
            identifiers["doi"].add(_canonical_doi(text))
    if len(identifiers["pmc"]) != 1:
        raise EuropePMCOAValidationError(
            "jats_pmcid_missing_or_ambiguous",
            "JATS must contain exactly one valid article-level PMCID",
        )
    if len(identifiers["pmid"]) > 1:
        raise EuropePMCOAValidationError(
            "jats_pmid_ambiguous",
            "JATS contains conflicting article-level PMIDs",
        )
    if len(identifiers["doi"]) > 1:
        raise EuropePMCOAValidationError(
            "jats_doi_ambiguous",
            "JATS contains conflicting article-level DOIs",
        )
    pmcid = next(iter(identifiers["pmc"]))
    pmid = next(iter(identifiers["pmid"]), None)
    doi = next(iter(identifiers["doi"]), None)
    return pmcid, pmid, doi


def _validate_jats_license(
    root: ElementTree.Element,
    expected: str,
) -> tuple[str, str]:
    uris: set[str] = set()
    article_meta = _find_article_meta(root)
    for element in article_meta.iter():
        if _local_name(element.tag) != "license":
            continue
        direct = element.attrib.get(_XLINK_HREF) or element.attrib.get("href")
        if direct and str(direct).strip():
            uris.add(str(direct).strip())
        for child in element.iter():
            if _local_name(child.tag) not in {"ext-link", "uri"}:
                continue
            href = child.attrib.get(_XLINK_HREF) or child.attrib.get("href")
            if href and str(href).strip():
                uris.add(str(href).strip())
    if not uris:
        # The caller has already independently admitted the exact article from
        # Europe PMC metadata as OA under an allow-listed licence.  Some valid
        # Europe PMC JATS documents omit the redundant article-level licence
        # element.  Preserve the official metadata attestation explicitly;
        # identity and retraction checks still fail closed, and any conflicting
        # JATS licence below remains a hard mismatch.
        return expected, "europe_pmc_metadata"
    if expected in uris:
        return expected, "europe_pmc_metadata_and_jats"
    expected_label = _normalize_license_label(expected)
    expected_path = _LICENSE_LABEL_PATHS.get(expected_label)
    if expected_path:
        matches = sorted(
            uri
            for uri in uris
            if _is_creative_commons_license_uri(uri, expected_path)
        )
        if matches:
            return matches[0], "europe_pmc_metadata_and_jats"
    raise EuropePMCOAValidationError(
        "jats_license_mismatch",
        "JATS license URI does not match caller-attested license metadata",
    )


def _eligible_paragraphs(
    root: ElementTree.Element,
) -> tuple[tuple[_EligibleParagraph, ...], int]:
    body = next(
        (element for element in root.iter() if _local_name(element.tag) == "body"),
        None,
    )
    if body is None:
        return (), 0
    paragraphs: list[_EligibleParagraph] = []
    section_counter = 0
    eligible_section_ids: set[int] = set()

    def visit(
        element: ElementTree.Element,
        *,
        inherited_eligible: bool,
        inherited_kind: str | None,
        inherited_title: str,
        inherited_section_index: int,
    ) -> None:
        nonlocal section_counter
        tag = _local_name(element.tag)
        if tag in _BLOCKED_CONTENT_TAGS:
            return
        eligible = inherited_eligible
        kind = inherited_kind
        title = inherited_title
        section_index = inherited_section_index
        if tag == "sec":
            section_counter += 1
            section_index = section_counter
            title_element = next(
                (
                    child
                    for child in element
                    if _local_name(child.tag) == "title"
                ),
                None,
            )
            title = _normalize_text(
                "".join(title_element.itertext())
                if title_element is not None
                else ""
            )[:160]
            section_type = _normalize_text(
                str(element.attrib.get("sec-type") or "")
            )
            descriptor = f"{section_type} {title}".strip()
            if _EXCLUDED_SECTION_PATTERN.search(descriptor):
                eligible = False
                kind = None
            elif _ALLOWED_SECTION_PATTERN.search(descriptor):
                eligible = True
                kind = _section_kind(descriptor)
                eligible_section_ids.add(section_index)
            elif inherited_eligible:
                eligible = True
                kind = inherited_kind
                eligible_section_ids.add(section_index)
            else:
                eligible = False
                kind = None
        if tag == "p" and eligible and kind is not None:
            text = _visible_text(element)
            if text:
                paragraphs.append(
                    _EligibleParagraph(
                        section_kind=kind,
                        section_title=title,
                        section_index=section_index,
                        paragraph_index=len(paragraphs) + 1,
                        text=text,
                    )
                )
            return
        for child in element:
            if tag == "sec" and _local_name(child.tag) == "title":
                continue
            visit(
                child,
                inherited_eligible=eligible,
                inherited_kind=kind,
                inherited_title=title,
                inherited_section_index=section_index,
            )

    for child in body:
        visit(
            child,
            inherited_eligible=False,
            inherited_kind=None,
            inherited_title="",
            inherited_section_index=0,
        )
    return tuple(paragraphs), len(eligible_section_ids)


def _sentence_spans(
    paragraphs: Sequence[_EligibleParagraph],
    *,
    max_spans: int,
) -> tuple[tuple[OAFullTextSpan, ...], bool]:
    spans: list[OAFullTextSpan] = []
    truncated_by_limit = False
    for paragraph in paragraphs:
        matches = list(_SENTENCE_PATTERN.finditer(paragraph.text))
        if not matches and paragraph.text:
            fallback = re.match(r"[\s\S]+", paragraph.text)
            matches = [fallback] if fallback is not None else []
        for sentence_index, match in enumerate(matches, 1):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            if len(spans) >= max_spans:
                truncated_by_limit = True
                return tuple(spans), truncated_by_limit
            excerpt, local_end, truncated = _bounded_prefix(sentence)
            leading = len(match.group(0)) - len(match.group(0).lstrip())
            char_start = match.start() + leading
            char_end = char_start + local_end
            location = (
                f"full_text:{paragraph.section_kind}:sec:{paragraph.section_index}:"
                f"p:{paragraph.paragraph_index}:sentence:{sentence_index}:"
                f"chars:{char_start}-{char_end}"
            )
            spans.append(
                OAFullTextSpan(
                    text=excerpt,
                    section_kind=paragraph.section_kind,
                    section_title=paragraph.section_title,
                    section_index=paragraph.section_index,
                    paragraph_index=paragraph.paragraph_index,
                    sentence_index=sentence_index,
                    char_start=char_start,
                    char_end=char_end,
                    source_location=location,
                    excerpt_sha256=sha256(excerpt.encode("utf-8")).hexdigest(),
                    excerpt_truncated=truncated,
                    original_sentence_char_count=len(sentence),
                )
            )
    return tuple(spans), truncated_by_limit


def _jats_references(
    root: ElementTree.Element,
    *,
    parent_pmcid: str,
    parent_pmid: str | None,
    max_references: int,
    visited_keys: Collection[str],
) -> tuple[OneHopReference, ...]:
    raw: list[OneHopReference] = []
    position = 0
    for ref_list in root.iter():
        if _local_name(ref_list.tag) != "ref-list":
            continue
        for ref in ref_list:
            if _local_name(ref.tag) != "ref":
                continue
            position += 1
            pmcid: str | None = None
            pmid: str | None = None
            doi: str | None = None
            for identifier in ref.iter():
                if _local_name(identifier.tag) not in {"pub-id", "object-id"}:
                    continue
                id_type = str(
                    identifier.attrib.get("pub-id-type")
                    or identifier.attrib.get("object-id-type")
                    or ""
                ).casefold()
                value = _normalize_text("".join(identifier.itertext()))
                if not value:
                    continue
                if id_type in {"pmid", "pubmed"}:
                    pmid = _optional_pmid(value) or pmid
                elif id_type in {"pmc", "pmcid"}:
                    pmcid = _optional_pmcid(value) or pmcid
                elif id_type == "doi":
                    doi = _optional_doi(value) or doi
            candidate = _make_reference(
                parent_pmcid=parent_pmcid,
                parent_pmid=parent_pmid,
                reference_position=position,
                reference_id=str(ref.attrib.get("id") or position),
                pmcid=pmcid,
                pmid=pmid,
                doi=doi,
            )
            if candidate is not None:
                raw.append(candidate)
    return exclude_visited_references(
        deduplicate_one_hop_references(raw),
        visited_keys,
    )[:max_references]


def _make_reference(
    *,
    parent_pmcid: str,
    parent_pmid: str | None,
    reference_position: int,
    reference_id: str,
    pmcid: str | None,
    pmid: str | None,
    doi: str | None,
) -> OneHopReference | None:
    keys = document_identity_keys(pmcid=pmcid, pmid=pmid, doi=doi)
    if not keys:
        return None
    canonical_id = _preferred_canonical_id(pmcid=pmcid, pmid=pmid, doi=doi)
    payload = _canonical_json_bytes(
        {
            "policy": REFERENCE_POLICY,
            "hop": 1,
            "relation": "reference",
            "parent_pmcid": parent_pmcid,
            "parent_pmid": parent_pmid,
            "position": reference_position,
            "child": canonical_id,
            "identity_keys": sorted(keys),
        }
    )
    return OneHopReference(
        parent_pmcid=parent_pmcid,
        parent_pmid=parent_pmid,
        reference_position=max(1, int(reference_position)),
        reference_id=_normalize_text(reference_id) or str(reference_position),
        pmcid=pmcid,
        pmid=pmid,
        doi=doi,
        canonical_id=canonical_id,
        identity_keys=tuple(sorted(keys)),
        citation_lineage_sha256=sha256(payload).hexdigest(),
    )


def _visible_text(element: ElementTree.Element) -> str:
    parts: list[str] = []

    def append(node: ElementTree.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if _local_name(child.tag) not in _BLOCKED_CONTENT_TAGS:
                append(child)
            if child.tail:
                parts.append(child.tail)

    append(element)
    return _normalize_text(" ".join(parts))


def _section_kind(value: str) -> str:
    folded = value.casefold()
    if re.search(r"\bcase(?:[- ]|\b)", folded):
        return "case"
    if re.search(r"\bfindings?\b", folded):
        return "findings"
    return "results"


def _bounded_prefix(sentence: str) -> tuple[str, int, bool]:
    if len(sentence) <= MAX_EXCERPT_CHARS:
        return sentence, len(sentence), False
    split = sentence.rfind(" ", 0, MAX_EXCERPT_CHARS + 1)
    end = split if split >= MAX_EXCERPT_CHARS // 2 else MAX_EXCERPT_CHARS
    excerpt = sentence[:end].rstrip()
    return excerpt, len(excerpt), True


def _preferred_canonical_id(
    *,
    pmcid: str | None,
    pmid: str | None,
    doi: str | None,
) -> str:
    if pmid is not None:
        return f"MED:{_canonical_pmid(pmid)}"
    if pmcid is not None:
        return f"PMC:{_canonical_pmcid(pmcid)}"
    if doi is not None:
        return f"DOI:{_canonical_doi(doi)}"
    raise EuropePMCOAValidationError(
        "reference_identifier_missing",
        "at least one PMID, PMCID, or DOI is required",
    )


def _canonical_pmcid(value: object) -> str:
    normalized = _normalize_text(str(value)).upper()
    if not _PMCID_PATTERN.fullmatch(normalized):
        raise EuropePMCOAValidationError(
            "invalid_pmcid",
            "PMCID must match PMC followed by digits",
        )
    return normalized


def _canonical_pmid(value: object) -> str:
    normalized = _normalize_text(str(value))
    if not _PMID_PATTERN.fullmatch(normalized):
        raise EuropePMCOAValidationError(
            "invalid_pmid",
            "PMID must contain digits only",
        )
    return normalized


def _canonical_doi(value: object) -> str:
    normalized = _normalize_text(str(value)).casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    if not _DOI_PATTERN.fullmatch(normalized):
        raise EuropePMCOAValidationError(
            "invalid_doi",
            "DOI must use the 10.<registrant>/<suffix> form",
        )
    return normalized


def _optional_pmcid(value: object) -> str | None:
    if value is None or not _normalize_text(str(value)):
        return None
    try:
        return _canonical_pmcid(value)
    except EuropePMCOAValidationError:
        return None


def _optional_pmid(value: object) -> str | None:
    if value is None or not _normalize_text(str(value)):
        return None
    try:
        return _canonical_pmid(value)
    except EuropePMCOAValidationError:
        return None


def _optional_doi(value: object) -> str | None:
    if value is None or not _normalize_text(str(value)):
        return None
    try:
        return _canonical_doi(value)
    except EuropePMCOAValidationError:
        return None


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(1, default)
    return max(1, parsed)


def _local_name(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].casefold()


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalize_license_label(value: str) -> str:
    normalized = _normalize_text(value).casefold().replace("_", "-")
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _is_creative_commons_license_uri(uri: str, expected_path: str) -> bool:
    parsed = urlsplit(uri)
    hostname = (parsed.hostname or "").casefold()
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and hostname in {"creativecommons.org", "www.creativecommons.org"}
        and parsed.path.casefold().startswith(expected_path)
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


__all__ = [
    "DEFAULT_MAX_REFERENCES",
    "DEFAULT_MAX_SENTENCE_SPANS",
    "EXCERPT_SELECTION_POLICY",
    "EuropePMCOAValidationError",
    "MAX_EXCERPT_CHARS",
    "MAX_JATS_DEPTH",
    "MAX_JATS_ELEMENTS",
    "MAX_OA_JATS_BYTES",
    "MAX_REFERENCE_PAYLOAD_BYTES",
    "OAArticleEligibility",
    "OAFullTextAudit",
    "OAFullTextResult",
    "OAFullTextSpan",
    "OneHopReference",
    "OneHopReferenceResult",
    "REFERENCE_POLICY",
    "RETRIEVAL_POLICY",
    "XML_PARSER_POLICY",
    "ReferenceParseAudit",
    "deduplicate_one_hop_references",
    "document_identity_keys",
    "exclude_visited_references",
    "parse_europe_pmc_references",
    "parse_oa_jats",
]
