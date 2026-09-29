"""Keyless, read-only public biology retrieval for MA1.

The adapter deliberately keeps retrieval separate from scientific inference.  A
keyword match from PubMed or Europe PMC is recorded as supporting *retrieval*
evidence, while target registry records from Open Targets and UniProt are
recorded as context.  None of these records are relabelled as causal proof or
counter-evidence.

All six services can be queried without an LLM key:

* PubMed through NCBI E-utilities (ESearch, ESummary, then bounded EFetch
  abstract evidence)
* Europe PMC through its REST search endpoint
* Open Targets through its GraphQL endpoint
* UniProtKB through the UniProt REST API
* DGIdb through its GraphQL endpoint
* Pharos through its GraphQL endpoint

The class is synchronous because MA1's current ``EvidenceRetriever`` boundary
is synchronous.  Its HTTP client is injectable, so tests and offline runs never
need to touch the network.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256
from html import unescape
import json
import re
from threading import Lock
from time import monotonic, sleep
from typing import Any, Callable, Mapping, Protocol, Sequence
from xml.etree import ElementTree

from drugpilot.agents.ma1 import EvidenceRetriever, TargetSeed
from drugpilot.agents.evidence_semantics import ABSTRACT_CONTEXT_KEY, abstract_context, source_review_text
from drugpilot.agents.molecular_scope import SEARCH_KEY, validated_fusion_search_terms
from drugpilot.agents.variant_search_scope import (
    KEY as VARIANT_SEARCH_KEY,
    query_terms as variant_query_terms,
    validate as validate_variant_search,
)
from drugpilot.domain.models import EvidenceRef, ResearchBrief
from drugpilot.tools.http import ScientificHttpClient, get_scientific_http_client
from drugpilot.tools.ncbi_rate_limit import NCBIRateLimiter
from drugpilot.tools.validation_trace import (
    EPMCNotExecutedReason,
    canonical_epmc_not_executed_trace,
)


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_REST_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
DGIDB_GRAPHQL_URL = "https://dgidb.org/api/graphql"
PHAROS_GRAPHQL_URL = "https://pharos-api.ncats.io/graphql"

_SUPPORTED_SOURCES = (
    "PubMed",
    "Europe PMC",
    "Open Targets",
    "UniProt",
    "DGIdb",
    "Pharos",
)

_MAX_PUBLICATION_EXCERPT_CHARS = 320
_EPMC_OA_FULLTEXT_POLICY = "epmc_oa_fulltext_one_hop_v1"
_EPMC_OA_EXCERPT_POLICY = "europe_pmc_oa_jats_results_v1"
# Inspect every already-selected core publication record when needed.  The
# independent HTTP/byte/time ledgers below remain the authoritative safety
# bounds; an arbitrary two-parent cutoff otherwise hides eligible OA papers.
_EPMC_OA_PARENT_LIMIT = 8
_EPMC_OA_REFERENCE_LIMIT = 6
_EPMC_OA_CHILD_FULLTEXT_LIMIT = 2
_EPMC_OA_HTTP_CALL_LIMIT = 12
# Preserve enough request-local capacity for one reference child's metadata
# lookup and full text. Reference pagination must not starve the one-hop lane.
_EPMC_OA_CHILD_HTTP_CALL_RESERVE = 2
_EPMC_OA_DOCUMENT_BYTE_LIMIT = 4 * 1024 * 1024
_EPMC_OA_AGGREGATE_BYTE_LIMIT = 12 * 1024 * 1024
_EPMC_OA_ELAPSED_SECONDS = 20.0
_EPMC_OA_ALLOWED_LICENSE_URIS = (
    "CC BY",
    "CC BY-SA",
    "CC BY-ND",
    "CC BY-NC",
    "CC BY-NC-SA",
    "CC BY-NC-ND",
    "CC0",
    "https://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by-nc/4.0/",
    "http://creativecommons.org/licenses/by-nc/4.0/",
    "https://creativecommons.org/licenses/by-nd/4.0/",
    "http://creativecommons.org/licenses/by-nd/4.0/",
    "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "http://creativecommons.org/licenses/by-nc-nd/4.0/",
    "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "http://creativecommons.org/licenses/by-nc-sa/4.0/",
    "https://creativecommons.org/licenses/by-sa/4.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
)
_PUBMED_COUNTER_RAW_DOCUMENT_BUDGET = 20
# PubMed ESearch exposes at most 10,000 PubMed identifiers in one result
# window.  The direct-direction lane is already a precision query and screens
# records deterministically, so use the complete source window instead of an
# arbitrary top-N cutoff.  If PubMed reports more than this source limit, the
# trace remains explicitly non-exhaustive and the selector still fails closed.
_PRIMARY_DIRECTION_RAW_DOCUMENT_BUDGET = 10_000
_PRIMARY_DIRECTION_HTTP_REQUEST_BUDGET = 3
# NCBI recommends switching away from one URL request when an E-utility call
# contains more than roughly 200 UIDs.  Keep ESummary GET requests at that
# published boundary so the injectable read-only client, request hashing, and
# per-call rate limiter retain their existing semantics without constructing a
# single unbounded query string.
_PRIMARY_DIRECTION_SUMMARY_BATCH_SIZE = 200
_PRIMARY_DIRECTION_FETCH_BATCH_SIZE = 20
_MAX_PAIRED_DIRECTION_EXCERPT_CHARS = 640
_COMPOUND_LINKED_DIRECTION_EXCERPT_POLICY = (
    "same-compound-target-selective-intervention-primary-outcome-pair-v1"
)
_PRIMARY_DIRECTION_QUERY_POLICY = (
    "brief-derived-target-action-experimental-direction-primary-v3"
)
_PRIMARY_DIRECTION_QUERY_CONSTRUCTION_POLICY = (
    "brief-derived-target-action-experimental-direction-primary-v3"
)
_PRIMARY_DIRECTION_CONTEXT_MATCH_POLICY = (
    "literal-disease-or-explicit-model-plus-target-variant-same-sentence-v1"
)
_VERIFIED_DIRECT_PRIMARY_POLICY = (
    "primary-nonreview-explicit-target-action-favorable-endpoint-v2"
)
_PRIMARY_ASSOCIATION_PREVALENCE_RAW_DOCUMENT_BUDGET = 20
_PRIMARY_ASSOCIATION_OTHER_RAW_DOCUMENT_BUDGET = 16
_PRIMARY_ASSOCIATION_RAW_DOCUMENT_BUDGET = (
    _PRIMARY_ASSOCIATION_PREVALENCE_RAW_DOCUMENT_BUDGET
    + _PRIMARY_ASSOCIATION_OTHER_RAW_DOCUMENT_BUDGET
)
_PRIMARY_ASSOCIATION_LANE_HTTP_REQUEST_BUDGET = 3
_PRIMARY_ASSOCIATION_HTTP_REQUEST_BUDGET = 6
_ASSOCIATION_EXCERPT_POLICY = (
    "same-sentence-target-disease-primary-association-ranking-v2"
)

# Retrieval stays target/disease constrained, but a direction-aware scientific
# clause keeps prevalence/prognosis-only publications from dominating the
# small result window returned by PubMed and Europe PMC.  These are retrieval
# terms, not evidence labels: every returned abstract must still pass the
# same-sentence target+disease excerpt contract below.
_DIRECTION_QUERY_TERMS: Mapping[str, tuple[str, ...]] = {
    "inhibit": (
        "inhibit",
        "inhibited",
        "inhibition",
        "inhibitor",
        "blockade",
        "suppression",
        "knockdown",
        "silencing",
    ),
    "activate": (
        "activate",
        "activated",
        "activation",
        "activator",
        "agonist",
        "stimulation",
        "overexpression",
    ),
    "modulate": (
        "modulate",
        "modulated",
        "modulation",
        "modulator",
        "regulation",
        "perturbation",
    ),
}

_MECHANISTIC_QUERY_TERMS: tuple[str, ...] = (
    "phosphorylation",
    "viability",
    "proliferation",
    "apoptosis",
    "sensitivity",
    "response",
    "IC50",
    "EC50",
    "Ki",
    "Kd",
    "potency",
    "dose response",
)

_DIRECTION_EXCERPT_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "inhibit": re.compile(
        r"\b(?:inhibit(?:ed|ing|ion|or|ors|ory|s)?|block(?:ade|ed|ing|s)?|"
        r"suppress(?:ed|es|ing|ion)?|knock[- ]?down|silenc(?:e|ed|ing))\b",
        re.I,
    ),
    "activate": re.compile(
        r"\b(?:activat(?:e|ed|es|ing|ion|or|ors)|agonis(?:t|ts|m)|"
        r"stimulat(?:e|ed|es|ing|ion)|overexpress(?:ed|es|ing|ion)?)\b",
        re.I,
    ),
    "modulate": re.compile(
        r"\b(?:modulat(?:e|ed|es|ing|ion|or|ors)|regulat(?:e|ed|es|ing|ion)|"
        r"perturb(?:ed|ing|ation|ations)?)\b",
        re.I,
    ),
}
# Retrieval uses broad recall for these intake choices. MA1 resolves each
# hypothesis direction independently before any primary claim can be retained.
_BROAD_DIRECTION_EXCERPT_PATTERN = re.compile(
    "(?:" + "|".join(pattern.pattern for pattern in _DIRECTION_EXCERPT_PATTERNS.values()) + ")",
    re.I,
)
_DIRECTION_EXCERPT_PATTERNS = {
    **_DIRECTION_EXCERPT_PATTERNS,
    "modulate": _BROAD_DIRECTION_EXCERPT_PATTERN,
    "unspecified": _BROAD_DIRECTION_EXCERPT_PATTERN,
}

_MECHANISTIC_ENDPOINT_PATTERN = re.compile(
    r"\b(?:phosphorylat\w*|viabilit\w*|proliferat\w*|apopto\w*|"
    r"cytotoxi\w*|sensitiv\w*|respons(?:e|es|ive)|IC50|EC50|Ki|Kd|"
    r"potenc\w*|tumou?r growth|tumou?r burden|cell survival|"
    r"signaling)\b",
    re.I,
)

_EXPERIMENTAL_RESULT_PATTERN = re.compile(
    r"\b(?:reduc(?:e|ed|es|ing|tion)|decreas(?:e|ed|es|ing)|"
    r"increas(?:e|ed|es|ing)|induc(?:e|ed|es|ing|tion)|"
    r"suppress(?:ed|es|ing|ion)|impair(?:ed|s|ing|ment)|"
    r"enhanc(?:e|ed|es|ing|ement)|show(?:ed|s|ing)|"
    r"achiev(?:e|ed|es|ing)|demonstrat(?:e|ed|es|ing|ion)|"
    r"dose[- ]dependent|significant(?:ly)?)\b",
    re.I,
)

_NON_DIRECTION_PRIMARY_PATTERN = re.compile(
    r"\b(?:prevalen\w*|frequen\w*|common(?:ly)?|prognos\w*|survival|"
    r"associated with|association|resistan\w*|refractor\w*|relaps\w*|"
    r"remains? unclear|unknown role|failed to|did not|no response|"
    r"limited efficacy)\b",
    re.I,
)

_ASSOCIATION_PREDICATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "mutation_prevalence",
        re.compile(
            r"(?:\b(?:frequen\w*|prevalen\w*|harbou?r\w*|carried|carry|"
            r"approximately\s+\d+)\b|\d+(?:\.\d+)?\s*%)",
            re.I,
        ),
    ),
    (
        "disease_target_association",
        re.compile(
            r"\b(?:associat(?:e|ed|es|ion)|correlat(?:e|ed|es|ion)|"
            r"enrich(?:ed|ment)|dependency|driver|characteri[sz](?:e|ed|es))\b",
            re.I,
        ),
    ),
    (
        "disease_target_expression",
        re.compile(
            r"\b(?:overexpress(?:ed|es|ion)?|upregulat(?:e|ed|es|ion)|"
            r"constitutive(?:ly)?\s+activat\w*)\b",
            re.I,
        ),
    ),
)

_MUTATION_OR_VARIANT_PATTERN = re.compile(
    r"\b(?:mutation|mutations|mutant|variant|variants|ITD|"
    r"internal tandem duplication|rearrangement|rearrangements|"
    r"rearranged|fusion|fusions|fusion-positive)\b",
    re.I,
)
_PRIMARY_ASSOCIATION_RESULT_PATTERN = re.compile(
    r"(?:\b(?:we (?:found|observed|detected|identified|showed|report)|"
    r"our (?:cohort|series|analysis|study)|patients?|cases?|cohort|samples?|"
    r"cell lines?|significant(?:ly)?|compared with|versus|odds ratio|"
    r"hazard ratio|p\s*[<=>])\b|\d+(?:\.\d+)?\s*%)",
    re.I,
)
_PRIMARY_ASSOCIATION_EXPLICIT_STUDY_PATTERN = re.compile(
    r"\b(?:we (?:found|observed|detected|identified|showed|report)|"
    r"(?:our|this) (?:cohort|series|analysis|study)|"
    r"(?:among|in|of)\s+\d+\s+(?:patients?|cases?|samples?)|"
    r"(?:cohort|series)\s+of\s+\d+|n\s*=\s*\d+|"
    r"(?:detected|identified|observed|found|present|occurred)\s+in\s+\d+)\b",
    re.I,
)
_PRIMARY_ASSOCIATION_NON_HUMAN_COHORT_PATTERN = re.compile(
    r"\b(?:cell lines?|cell cultures?|in vitro|assays?|organoids?|"
    r"xenografts?|murine|mice|mouse|animal models?|preclinical models?)\b",
    re.I,
)
_VARIANT_SEPARATOR_PATTERN = r"[-/\u2010-\u2015\u2212]"
_VARIANT_NEGATIVE_SUFFIX_GUARD = (
    r"(?!\s*(?:(?:[-\u2212\u2013\u2014]\s*)?negative\b|"
    r"\(\s*[-\u2212\u2013\u2014]\s*\)|"
    r"[-\u2212\u2013\u2014](?=\s*(?:[,;.)]|$))))"
)
_PRIMARY_ASSOCIATION_NEGATED_OR_ZERO_PATTERN = re.compile(
    r"(?:\b(?:no|none)\b[^.!?]{0,48}\b(?:mutation|mutations|variant|variants|"
    r"ITD|internal tandem duplication)\b|"
    r"\b(?:mutation|mutations|variant|variants|ITD|internal tandem duplication)"
    r"\b[^.!?]{0,48}\b(?:absent|not detected|not found|none)\b|"
    r"\b(?:ITD\b|internal tandem duplication\b|mutations?\b|variants?\b|"
    r"[A-Za-z][0-9]{1,5}[A-Za-z*](?![A-Za-z0-9]))"
    r"(?:\s*[-\u2212–—]\s*|\s+(?:(?:is|are|was|were)\s+)?)negative\b|"
    r"\b(?:ITD|internal tandem duplication|mutations?|variants?|"
    r"[A-Za-z][0-9]{1,5}[A-Za-z*](?![A-Za-z0-9]))"
    r"\s*\(\s*[-\u2212–—]\s*\)|"
    r"\bnon(?:[-\u2010-\u2015\u2212 ]?)ITD\b|"
    r"\b0\s+(?:of\s+\d+|patients?|cases?|samples?)\b|"
    r"(?<![\d.])0(?:\.0+)?\s*%)",
    re.I,
)
_PRIMARY_ASSOCIATION_NULL_CUE_PATTERN = (
    r"(?:no|not|without|insufficient|insufficiency|"
    r"non[- ]?significant|insignificant|absence(?:\s+of)?|absent|"
    r"cannot|can\s+not|lack(?:s|ed|ing)?(?:\s+of)?|"
    r"fail(?:s|ed)?\s+to|(?:do|does|did)\s+not)"
)
_PRIMARY_ASSOCIATION_NULL_CONCEPT_PATTERN = (
    r"(?:evidence|associat\w*|correlat\w*|significan\w*|"
    r"expression|levels?|higher|lower|overexpress\w*|upregulat\w*|"
    r"different)"
)
_PRIMARY_ASSOCIATION_NEGATIVE_RELATION_PATTERN = re.compile(
    rf"(?:\b{_PRIMARY_ASSOCIATION_NULL_CUE_PATTERN}\b"
    rf"[^.!?]{{0,96}}\b{_PRIMARY_ASSOCIATION_NULL_CONCEPT_PATTERN}\b|"
    rf"\b{_PRIMARY_ASSOCIATION_NULL_CONCEPT_PATTERN}\b"
    rf"[^.!?]{{0,96}}\b{_PRIMARY_ASSOCIATION_NULL_CUE_PATTERN}\b|"
    r"\b(?:unassociated|uncorrelated)\b)",
    re.I,
)
_PRIMARY_ASSOCIATION_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"\s*(?:;|\b(?:whereas|while|but)\b)\s*",
    re.I,
)
_PRIMARY_ASSOCIATION_RELATION_CONTEXT_PATTERN = re.compile(
    r"\b(?:evidence|associat\w*|correlat\w*|significan\w*|"
    r"expression|levels?|overexpress\w*|upregulat\w*)\b",
    re.I,
)
_NEW_GENE_OR_VARIANT_SUBJECT_PATTERN = re.compile(
    rf"(?:\b[A-Z][A-Z0-9]{{1,11}}(?:\s*{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)"
    r"(?:mutants?|mutations?|variants?|ITD|[A-Z][0-9]{1,5}[A-Z*])\b|"
    r"\b[A-Z]*[0-9][A-Z0-9]*\s+(?:subgroup|cohort|cases?|patients?)\b|"
    r"\b(?:ITD|[A-Z][0-9]{1,5}[A-Z*])\s+"
    r"(?:subgroup|cohort|cases?|patients?)\b)"
)
_ABSTRACT_SENTENCE_PATTERN = re.compile(
    r"(?:\d+\.\d+|[pP]\.[A-Za-z][0-9]{1,5}[A-Za-z*]"
    r"(?![A-Za-z0-9])|[^.!?])+(?:[.!?]+|$)"
)
_COMPARATIVE_EXPRESSION_PATTERN = re.compile(
    r"\b(?:higher|lower|increas(?:e|ed)|decreas(?:e|ed)|elevated|reduced|"
    r"compared with|versus|fold|significant(?:ly)?)\b",
    re.I,
)
_ASSOCIATION_CONTEXT_EXCLUSION_PATTERN = re.compile(
    r"\b(?:acquired|resistan\w*|refractor\w*|relaps\w*|"
    r"(?:after|during|following) (?:therapy|treatment)|response to|"
    r"non[- ]?response|treatment outcome)\b",
    re.I,
)
_PRIMARY_PUBLICATION_TYPE_PATTERN = re.compile(
    r"\b(?:journal article|research[- ]article|clinical trial|"
    r"randomized controlled|comparative study|evaluation study|"
    r"validation study|multicenter study|observational study|"
    r"preclinical study)\b",
    re.I,
)
_REVIEW_TITLE_PATTERN = re.compile(
    r"\b(?:systematic review|review article|meta-analysis|consensus|guideline)\b",
    re.I,
)
_NON_PRIMARY_PUBLICATION_TYPE_PATTERN = re.compile(
    r"\b(?:meta[- ]?analysis|guideline|practice guideline|consensus)\b",
    re.I,
)

_INDIRECT_INTERVENTION_CONTEXT_PATTERNS: tuple[
    tuple[str, re.Pattern[str]], ...
] = (
    (
        "combination_or_co_treatment",
        re.compile(
            r"\b(?:combination|combined (?:inhibition|treatment|therapy|with)|"
            r"combined\s+[^.!?]{1,48}\band\b[^.!?]{0,48}\b"
            r"(?:inhibition|treatment|therapy|blockade)|"
            r"co[- ](?:inhibition|treat(?:ed|ment)|administration|therapy)|"
            r"dual (?:inhibition|targeting|blockade))\b",
            re.I,
        ),
    ),
    (
        "response_modifier",
        re.compile(
            r"\b(?:potentiat(?:e|ed|es|ing|ion)|sensiti[sz](?:e|ed|es|ing|ation)|"
            r"resensiti[sz](?:e|ed|es|ing|ation)|synerg(?:y|istic|istically)|"
            r"(?:increas\w*|enhanc\w*|elevat\w*|improv\w*)\s+"
            r"(?:the\s+)?(?:drug\s+)?sensitiv\w*|"
            r"enhanc(?:e|ed|es|ing|ement)\s+(?:the\s+)?(?:drug\s+)?response)\b",
            re.I,
        ),
    ),
    (
        "sensitization_to_other_intervention",
        re.compile(
            r"\b(?:increas\w*|enhanc\w*|elevat\w*|improv\w*|"
            r"sensiti[sz](?:e|ed|es|ing|ation)|"
            r"resensiti[sz](?:e|ed|es|ing|ation))\b"
            r"[^.!?]{0,72}\bsensitiv\w*\b[^.!?]{0,96}\bto\b",
            re.I,
        ),
    ),
    (
        "other_drug_response",
        re.compile(
            r"\b(?:response|sensitivity)\s+to\s+(?:treatment\s+with\s+)?"
            r"[A-Za-z][A-Za-z0-9-]{2,}|"
            r"\b[A-Za-z][A-Za-z0-9-]*(?:tinib|ciclib|mab|platin|taxel|"
            r"mycin|navir|oxetine|azole|parib|oclax)\s+"
            r"(?:response|sensitivity)\b",
            re.I,
        ),
    ),
)

_INTERVENTION_NOUN = (
    r"(?:inhibition|activation|blockade|suppression|silencing|knock[- ]?down|"
    r"overexpression|depletion|targeting)"
)
_GENE_LIKE_TOKEN = r"[A-Za-z][A-Za-z0-9-]{1,20}"

_COUNTER_SIGNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("resistance", re.compile(r"\b(?:resistan(?:ce|t)|drug[- ]resistant)\b", re.I)),
    ("non_response", re.compile(r"\b(?:non[- ]?response|non[- ]?responder|refractory to)\b", re.I)),
    ("lack_of_benefit", re.compile(r"\b(?:lack(?:ed|s|ing)? of (?:response|benefit|efficacy)|no (?:significant )?(?:response|benefit|efficacy)|limited (?:response|benefit|efficacy))\b", re.I)),
    ("opposite_or_null_result", re.compile(r"\b(?:failed to (?:improve|inhibit|reduce|respond)|not associated with|did not (?:improve|inhibit|reduce|respond))\b", re.I)),
    ("escape_or_relapse", re.compile(r"\b(?:target (?:loss|escape)|escape mechanism|relapse (?:during|after|despite))\b", re.I)),
    ("dose_limiting_toxicity", re.compile(r"\b(?:dose[- ]limiting toxicit|treatment[- ]limiting toxicit|unacceptable toxicit)\w*\b", re.I)),
)


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class _CacheEntry:
    stored_at: float
    records: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class _PublicationContent:
    """Transient abstract parsing result.

    The primary-admission excerpt stays bounded. A complete normalized abstract
    may separately be retained as hash-bound interpretation context; oversized
    abstracts are explicitly omitted, not silently truncated. No full paper is stored.
    """

    excerpt: str | None = None
    source_location: str | None = None
    content_hash: str | None = None
    interpretation_context: dict | None = None
    doi: str | None = None
    pmcid: str | None = None
    excerpt_truncated: bool = False
    excerpt_selection_status: str = "not_available"
    counter_signal_category: str | None = None
    counter_signal: str | None = None
    association_excerpt: str | None = None
    association_source_location: str | None = None
    association_excerpt_truncated: bool = False
    association_predicate: str | None = None
    association_scope: str | None = None
    association_score: int = 0
    direct_support_rejection_codes: tuple[str, ...] = ()
    target_lexical_match: bool | None = None


@dataclass(frozen=True)
class _TargetVariantClause:
    """One target-adjacent subtype clause derived only from the Brief."""

    kind: str
    canonical: str
    query_phrases: tuple[str, ...]


@dataclass(frozen=True)
class _PrimaryAssociationRequirement:
    target_symbol: str
    requested: bool
    required_scope: str | None
    variant_clauses: tuple[_TargetVariantClause, ...] = ()
    trigger_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PrimaryAssociationQueryLane:
    """One deterministic precision-first PubMed association query."""

    lane_id: str
    query: str
    raw_document_budget: int


@dataclass(frozen=True)
class _PrevalenceQuantitation:
    start: int
    numerator: int | None
    denominator: int | None
    percent: float
    percent_text: str


@dataclass(frozen=True)
class _PrimaryAssociationSupplementOutcome:
    records: tuple[EvidenceRef, ...]
    status: str
    stop_reason: str
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class _PrimaryDirectionSupplementOutcome:
    records: tuple[EvidenceRef, ...]
    status: str
    stop_reason: str
    trace: Mapping[str, Any]


class _PubMedBatchValidationError(ValueError):
    """Fail-closed signal for an incomplete or reordered PubMed batch."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class _PubMedSearchValidationError(ValueError):
    """Fail-closed signal for a malformed PubMed ESearch result window."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class _PrimaryDirectionModelContext:
    """Exact, Brief-declared molecular model context for direction evidence."""

    model_aliases: tuple[str, ...]
    variant_clauses: tuple[_TargetVariantClause, ...]


@dataclass(frozen=True)
class _PubMedCounterOutcome:
    records: tuple[EvidenceRef, ...]
    status: str
    query: str
    query_sha256: str
    raw_hit_count: int
    http_call_provenance: Mapping[str, Any]


@dataclass
class _EuropePMCOAEscalationState:
    """Mutable, request-local counters for the optional Europe PMC lane."""

    started_at: float
    http_calls: list[dict[str, Any]] = field(default_factory=list)
    response_bytes_total: int = 0
    parent_fulltexts_attempted: int = 0
    parent_fulltexts_parsed: int = 0
    external_jats_doctypes_removed: int = 0
    references_retained: int = 0
    child_metadata_checked: int = 0
    child_fulltexts_attempted: int = 0
    child_fulltexts_parsed: int = 0
    cross_parent_duplicate_references_dropped: int = 0
    stop_reason: str | None = None
    error_reason_codes: list[str] = field(default_factory=list)
    eligibility_rejections: dict[str, int] = field(default_factory=dict)
    reference_audits: list[dict[str, Any]] = field(default_factory=list)
    discovery_parents: list[dict[str, Any]] = field(default_factory=list)


class _EuropePMCOABudgetStop(RuntimeError):
    """Internal bounded-stop signal; never escapes the optional lane."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class _EuropePMCOAWireResult:
    """One retry-free bounded response from the Europe PMC supplemental lane."""

    response: Any
    body: bytes
    response_body_complete: bool
    overflow_reason: str | None
    transport_mode: str
    transport_body_cap_enforced: bool
    declared_content_length: int | None


class _EuropePMCOAWireReadError(RuntimeError):
    """Preserve bounded partial-read accounting when a stream fails."""

    def __init__(
        self,
        cause: Exception,
        *,
        body_prefix: bytes,
        status_code: int | None,
        declared_content_length: int | None,
        transport_mode: str,
        transport_body_cap_enforced: bool,
    ) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause
        self.body_prefix = body_prefix
        self.status_code = status_code
        self.declared_content_length = declared_content_length
        self.transport_mode = transport_mode
        self.transport_body_cap_enforced = transport_body_cap_enforced


class PublicBiologyRetriever(EvidenceRetriever):
    """Retrieve compact, traceable MA1 records from public databases.

    Caching is maintained per source rather than for the whole request.  If one
    public service is temporarily unavailable, successful sources are reused
    from cache while only the failed source is retried on the next invocation.

    ``enabled_sources`` is an optional adapter-level allow-list applied before
    the per-brief data policy.  ``None`` retains the original standalone
    behaviour (all six adapters enabled), while workflow callers can select a
    bounded default subset and enable slower adapters such as Pharos only for
    an explicit on-demand request.

    ``max_source_calls`` and ``max_elapsed_seconds`` are enforced between
    source adapters. An already admitted synchronous source is allowed to
    finish under its per-request timeout; later sources receive explicit
    ``skipped_*_budget`` audit statuses instead of being silently omitted.

    ``fallback`` is optional.  When supplied, it is used only if every enabled
    and brief-allowed public source fails or returns no records.  Fallback
    records are explicitly labelled as fixtures; they can never be mistaken
    for a live citation.
    """

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 8.0,
        max_records_per_source: int = 4,
        max_source_calls: int | None = None,
        max_elapsed_seconds: float | None = None,
        cache_ttl_seconds: float = 3_600.0,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.15,
        fallback: EvidenceRetriever | None = None,
        enabled_sources: Sequence[str] | None = None,
        ncbi_api_key: str | None = None,
        ncbi_tool: str = "DrugPilot",
        ncbi_email: str | None = None,
        counter_evidence_enabled: bool = False,
        epmc_oa_fulltext_one_hop_v1: bool = False,
        ma3_source_review_enabled: bool = False,
        primary_direction_raw_document_budget: int = (
            _PRIMARY_DIRECTION_RAW_DOCUMENT_BUDGET
        ),
        ncbi_rate_limiter: NCBIRateLimiter | None = None,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= max_records_per_source <= 20:
            raise ValueError("max_records_per_source must be between 1 and 20")
        if max_source_calls is not None and not 1 <= max_source_calls <= len(
            _SUPPORTED_SOURCES
        ):
            raise ValueError(
                f"max_source_calls must be between 1 and {len(_SUPPORTED_SOURCES)}"
            )
        if max_elapsed_seconds is not None and max_elapsed_seconds <= 0:
            raise ValueError("max_elapsed_seconds must be positive")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must not be negative")
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        if not 1 <= primary_direction_raw_document_budget <= 10_000:
            raise ValueError(
                "primary_direction_raw_document_budget must be between 1 and 10000"
            )

        requested_sources = (
            set(_SUPPORTED_SOURCES)
            if enabled_sources is None
            else {str(source).strip() for source in enabled_sources}
        )
        unknown_sources = requested_sources.difference(_SUPPORTED_SOURCES)
        if unknown_sources:
            unknown = ", ".join(sorted(unknown_sources))
            raise ValueError(f"unsupported public biology source(s): {unknown}")

        self.client = client or get_scientific_http_client()
        self.timeout_seconds = timeout_seconds
        self.max_records_per_source = max_records_per_source
        self.max_source_calls = max_source_calls
        self.max_elapsed_seconds = max_elapsed_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.fallback = fallback
        # Preserve the canonical execution order regardless of caller order or
        # duplicates.  This keeps call traces and tests deterministic.
        self.enabled_sources = tuple(
            source for source in _SUPPORTED_SOURCES if source in requested_sources
        )
        self.ncbi_api_key = ncbi_api_key.strip() if ncbi_api_key else None
        self.ncbi_tool = ncbi_tool.strip() or "DrugPilot"
        self.ncbi_email = ncbi_email.strip() if ncbi_email else None
        self.counter_evidence_enabled = bool(counter_evidence_enabled)
        self.ma3_source_review_enabled = bool(ma3_source_review_enabled)
        self.primary_direction_raw_document_budget = int(
            primary_direction_raw_document_budget
        )
        self.epmc_oa_fulltext_one_hop_v1 = bool(
            epmc_oa_fulltext_one_hop_v1
        )
        self._clock = clock
        self._sleeper = sleeper
        self._cache: dict[tuple[str, ...], _CacheEntry] = {}
        self._cache_lock = Lock()
        self._last_failures: dict[str, str] = {}
        self._last_epmc_oa_fulltext_trace: dict[str, Any] = (
            canonical_epmc_not_executed_trace(
                EPMCNotExecutedReason.UPSTREAM_EVIDENCE_UNAVAILABLE
            )
            if self.epmc_oa_fulltext_one_hop_v1
            else {
                "schema_version": "epmc_oa_fulltext_one_hop_trace/v1",
                "policy": _EPMC_OA_FULLTEXT_POLICY,
                "enabled": False,
                "status": "disabled",
            }
        )
        self._ncbi_rate_limiter = ncbi_rate_limiter or NCBIRateLimiter(
            clock=clock,
            sleeper=sleeper,
        )

    @property
    def last_failures(self) -> Mapping[str, str]:
        """Return the most recent per-source failure summary without secrets."""

        with self._cache_lock:
            return dict(self._last_failures)

    @property
    def last_epmc_oa_fulltext_trace(self) -> Mapping[str, Any]:
        """Return the bounded supplemental lane status without raw content."""

        with self._cache_lock:
            return deepcopy(self._last_epmc_oa_fulltext_trace)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        allowed = [
            source
            for source in self.enabled_sources
            if source in brief.data_policy.allowed_sources
        ]
        if not allowed:
            return self._fallback_or_raise(
                brief,
                seed,
                iteration,
                {"configuration": "no supported public biology source is allowed"},
            )

        records: list[EvidenceRef] = []
        failures: dict[str, str] = {}
        started_at = self._clock()
        admitted_source_calls = 0
        evidence_retry = (
            seed.discovery_source == "prior_revision_evidence_retry"
        )
        for source in allowed:
            if (
                self.max_source_calls is not None
                and admitted_source_calls >= self.max_source_calls
            ):
                failures[source] = "skipped_call_budget"
                continue
            if (
                self.max_elapsed_seconds is not None
                and self._clock() - started_at >= self.max_elapsed_seconds
            ):
                failures[source] = "skipped_time_budget"
                continue
            # The budget is counted at the source-adapter boundary, not per
            # underlying HTTP exchange. PubMed, for example, uses bounded
            # ESearch, ESummary and EFetch requests as one provenance-bearing
            # source lookup.
            admitted_source_calls += 1
            cached = (
                None
                if evidence_retry
                else self._cached_records(source, brief, seed)
            )
            if cached is not None:
                records.extend(cached)
                continue
            try:
                source_records = self._with_retry(source, brief, seed)
            except Exception as exc:  # isolate a single public-service failure
                failures[source] = type(exc).__name__
                continue
            if not source_records:
                failures[source] = "empty_result"
                continue
            self._store_records(source, brief, seed, source_records)
            records.extend(item.model_copy(deep=True) for item in source_records)

        with self._cache_lock:
            self._last_failures = dict(failures)

        if not records:
            return self._fallback_or_raise(brief, seed, iteration, failures)

        # Attach a compact coverage warning to the records returned in this
        # invocation.  This makes partial retrieval visible to G0 without
        # mutating the cached source payloads.
        if failures:
            records = [
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "partial_source_failures": dict(failures),
                        }
                    },
                    deep=True,
                )
                for item in records
            ]
        if evidence_retry:
            records = [
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "evidence_retry_cache_bypassed": True,
                            "evidence_retry_cache_policy": (
                                "fresh_public_lookup_required"
                            ),
                        }
                    },
                    deep=True,
                )
                for item in records
            ]
        return tuple(records)

    def retrieve_g0_followup(self, brief, seed, *, roles, source_paths=None, elapsed_seconds=None, prior_records=()):
        """Run cause-specific precision lanes, never fabricate fallback records.

        Each admitted lane keeps its existing HTTP/document/timeout guards.
        The outer elapsed/source budget is checked before admitting another
        lane. A failed or empty lane is observable even when no evidence exists.
        """
        roles = tuple(dict.fromkeys(roles))
        if not roles or not set(roles).issubset({"direction", "association", "assay_context", "assay_identity", "assay_fulltext"}):
            raise ValueError("Unsupported G0 follow-up role")
        if {"assay_context", "assay_identity", "assay_fulltext"} & set(roles):
            from drugpilot.tools.assay_followup import required_assays
            if not required_assays(brief):
                raise ValueError("No supported required assay endpoint")
        if source_paths is not None and any(p.get("role") == "assay_context" and p.get("source") != "PubMed" for p in source_paths):
            raise ValueError("Assay context follow-up is supported only by PubMed")
        if source_paths is not None and any(p.get("role") == "assay_identity" and p.get("source") != "ChEMBL" for p in source_paths):
            raise ValueError("Assay identity follow-up is supported only by ChEMBL")
        if source_paths is not None and any(p.get("role") == "assay_fulltext" and p.get("source") != "Europe PMC" for p in source_paths):
            raise ValueError("Assay fulltext follow-up is supported only by Europe PMC")
        limit = self.max_elapsed_seconds
        if elapsed_seconds is not None:
            if not 0 < elapsed_seconds <= 120:
                raise ValueError("G0 follow-up admission limit must be at most 120 seconds")
            limit = elapsed_seconds
        fusion_terms = _fusion_query_terms(seed)
        variant_bindings=seed.discovery_metadata.get(VARIANT_SEARCH_KEY,[])
        variant_terms=validate_variant_search(variant_bindings,seed.symbol)
        variant_labels=tuple(sorted({b["label"] for b in variant_bindings}))
        oa_roles = tuple(role for role in roles if role not in {"assay_context", "assay_identity", "assay_fulltext"})
        lanes = [*((role, "PubMed") for role in roles if role not in {"assay_identity", "assay_fulltext"}), *([("oa_context", "Europe PMC")] if oa_roles else []),
            *([("assay_identity", "ChEMBL")] if "assay_identity" in roles else []),
            *([("assay_fulltext", "Europe PMC")] if "assay_fulltext" in roles else [])]
        if source_paths is not None:
            if any(p.get("source") not in ({"ChEMBL"} if p.get("role") == "assay_identity" else {"PubMed", "Europe PMC"}) or p.get("role") not in roles for p in source_paths):
                raise ValueError("Unsupported G0 follow-up source path")
            requested = {(p["role"], p["source"]) for p in source_paths}
            oa_roles = tuple(role for role in roles if role != "assay_fulltext" and (role, "Europe PMC") in requested)
            # Reserve first admission for the previously starved OA path.
            # All internal OA/HTTP/license/identity guards remain unchanged.
            lanes = ([("oa_context", "Europe PMC")] if oa_roles else []) + [
                (role, "PubMed") for role in roles if (role, "PubMed") in requested] + [
                ("assay_identity", "ChEMBL")] * (("assay_identity", "ChEMBL") in requested) + [
                ("assay_fulltext", "Europe PMC")] * (("assay_fulltext", "Europe PMC") in requested)
        records, observations = list(prior_records), []
        started = self._clock()
        calls = 0
        allowed = set(self.enabled_sources) & set(brief.data_policy.allowed_sources)
        # Dedicated planned chemistry join; not a new default biology adapter.
        # It needs verified PubMed contexts (current or hash-checked same-run
        # carryover) and the explicit brief allow-list.
        if "assay_identity" in roles and "PubMed" in allowed and "ChEMBL" in brief.data_policy.allowed_sources:
            allowed.add("ChEMBL")
        common = {"tool": self.ncbi_tool}
        if self.ncbi_email:
            common["email"] = self.ncbi_email
        if self.ncbi_api_key:
            common["api_key"] = self.ncbi_api_key
        seed = replace(seed, discovery_metadata={**seed.discovery_metadata,
            "g0_followup_roles": list(oa_roles)})
        for role, source in lanes:
            row = {"role": role, "source": source, "attempted": False, "record_count": 0,
                "admission_limit_seconds": limit, "elapsed_limit_semantics":"lane_admission_not_exact_wall_clock_deadline"}
            if SEARCH_KEY in seed.discovery_metadata and fusion_terms:
                row.update(fusion_search_terms=list(fusion_terms),
                    fusion_search_bindings=deepcopy(seed.discovery_metadata[SEARCH_KEY]),
                    search_scope_authority="search_only_not_support_or_identity")
            if variant_terms:
                row.update(variant_search_terms=list(variant_terms),
                    variant_search_bindings=deepcopy(variant_bindings),
                    search_scope_authority="search_only_not_support_or_identity")
            if source not in allowed:
                row.update(status="not_attempted", reason_code="SOURCE_NOT_ALLOWED")
            elif ((self.max_source_calls is not None and calls >= self.max_source_calls)
                  or (limit is not None and self._clock() - started >= limit)):
                row.update(status="not_attempted", reason_code="SOURCE_OR_TIME_BUDGET")
            else:
                calls += 1
                row["attempted"] = True
                try:
                    if role == "assay_fulltext":
                        from drugpilot.tools.assay_fulltext import retrieve_assay_fulltext
                        result = retrieve_assay_fulltext(self, brief, seed, records,
                            elapsed_seconds=max(0.001, limit - (self._clock() - started)) if limit is not None else 120)
                        records = list(result.records)
                        new = ()
                        row.update(status=result.status, reason_code=result.reason, trace=dict(result.trace),
                            attempted=bool(result.trace["http_requests_used"]),
                            annotated_context_count=result.trace["annotations_created"])
                    elif role == "assay_identity":
                        from drugpilot.tools.assay_identity import resolve_assay_identities
                        result = resolve_assay_identities(self, brief, seed, records,
                            elapsed_seconds=max(0.001, limit - (self._clock() - started)) if limit is not None else 120)
                        records = list(result.records)
                        new = ()  # Annotations preserve the original evidence IDs, not duplicate records.
                        row.update(status=result.status, reason_code=result.reason, trace=dict(result.trace),
                            attempted=bool(result.trace["http_requests_used"]),
                            annotated_context_count=result.trace["annotations_created"])
                    elif role == "direction":
                        from drugpilot.orchestration.evidence_carryover import ALTERNATIVE_KEY
                        alternative_exclusions = seed.discovery_metadata.get(ALTERNATIVE_KEY, [])
                        result = self._pubmed_primary_direction_supplement(brief, seed,
                            query=_g0_direction_query(self,brief,seed),
                            common_params=common, excluded_ids=set(alternative_exclusions),
                            direction_model_context=_primary_direction_model_context(brief, seed.symbol),
                            **({"required_molecular_scopes":variant_labels} if variant_labels else {}))
                        new = result.records
                        row.update(status=result.status, reason_code=result.stop_reason, trace=dict(result.trace))
                        if alternative_exclusions:
                            row.update(prior_disputed_selection_pmids=alternative_exclusions,
                                selection_policy="continue_to_other_source_not_resample_disagreement",
                                prior_source_evidence_preserved=True)
                    elif role == "assay_context":
                        from drugpilot.tools.assay_followup import retrieve_assay_context
                        result = retrieve_assay_context(self, brief, seed,
                            elapsed_seconds=max(0.001, limit - (self._clock() - started)) if limit is not None else 120)
                        new = result.records
                        row.update(status=result.status, reason_code=result.stop_reason, trace=dict(result.trace))
                    elif role == "association":
                        requirement = _primary_association_requirement(brief, seed.symbol)
                        result = self._pubmed_primary_association_supplement(brief, seed,
                            query_lanes=self._pubmed_primary_association_query_lanes(brief, seed, requirement=requirement),
                            excluded_ids=set(), requirement=requirement)
                        new = result.records
                        row.update(status=result.status, reason_code=result.stop_reason, trace=dict(result.trace))
                    else:
                        new = self._europe_pmc(brief, seed)
                        row.update(status="completed" if new else "empty", reason_code="EPMC_SOURCE_PATH_COMPLETED",
                            oa_trace=self.last_epmc_oa_fulltext_trace)
                    if prior_records:
                        from drugpilot.orchestration.evidence_carryover import merge_sources
                        records = list(merge_sources(new, records))
                    else:
                        records.extend(new)
                    row["record_count"] = len(new)
                except Exception as exc:
                    if isinstance(exc, ValueError) and str(exc).startswith("CARRYOVER_"):
                        raise  # A conflicting source ID must not fall back to the old record.
                    row.update(status="failed", reason_code=type(exc).__name__)
            from drugpilot.tools.evidence_gap import diagnose_observation
            row["evidence_gap"] = diagnose_observation(row)
            observations.append(row)
        return tuple(records), observations

    def _with_retry(
        self,
        source: str,
        brief: ResearchBrief,
        seed: TargetSeed,
    ) -> list[EvidenceRef]:
        adapters: Mapping[str, Callable[[ResearchBrief, TargetSeed], list[EvidenceRef]]] = {
            "PubMed": self._pubmed,
            "Europe PMC": self._europe_pmc,
            "Open Targets": self._open_targets,
            "UniProt": self._uniprot,
            "DGIdb": self._dgidb,
            "Pharos": self._pharos,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return adapters[source](brief, seed)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts and self.retry_backoff_seconds:
                    self._sleeper(self.retry_backoff_seconds * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    def _pubmed(self, brief: ResearchBrief, seed: TargetSeed) -> list[EvidenceRef]:
        query = self._pubmed_query(brief, seed)
        requirement = _primary_association_requirement(brief, seed.symbol)
        direction_model_context = _primary_direction_model_context(
            brief,
            seed.symbol,
        )
        common_params: dict[str, str] = {"tool": self.ncbi_tool}
        if self.ncbi_email:
            common_params["email"] = self.ncbi_email
        if self.ncbi_api_key:
            common_params["api_key"] = self.ncbi_api_key

        search_response = self._ncbi_get(
            PUBMED_SEARCH_URL,
            params={
                **common_params,
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": self.max_records_per_source,
                "sort": "relevance",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        search_response.raise_for_status()
        search_payload = search_response.json()
        ids = [str(value) for value in search_payload.get("esearchresult", {}).get("idlist", [])]
        result: Mapping[str, Any] = {}
        if ids:
            summary_response = self._ncbi_get(
                PUBMED_SUMMARY_URL,
                params={
                    **common_params,
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "json",
                },
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            summary_response.raise_for_status()
            result = summary_response.json().get("result", {})

        # ESummary intentionally does not carry article abstracts.  Fetch the
        # XML abstract records separately and treat this enrichment as
        # best-effort: a missing abstract must not turn traceable publication
        # metadata into a fabricated or failed evidence record.
        content_by_pmid: dict[str, _PublicationContent] = {}
        abstract_error_type: str | None = None
        if ids:
            try:
                fetch_response = self._ncbi_get(
                    PUBMED_FETCH_URL,
                    params={
                        **common_params,
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "xml",
                        "rettype": "abstract",
                    },
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                )
                fetch_response.raise_for_status()
                content_by_pmid = _parse_pubmed_content(
                    _response_text(fetch_response),
                    target_symbol=seed.symbol,
                    disease=brief.disease,
                    intervention_direction=brief.constraints.intervention_direction,
                    direction_model_context=direction_model_context,
                )
            except Exception as exc:
                # The title/identifier metadata remains usable, but the record is
                # explicitly marked metadata-only so downstream LLMs cannot claim
                # that the abstract was read.
                abstract_error_type = type(exc).__name__

        evidence: list[EvidenceRef] = []
        association_candidates: list[tuple[int, EvidenceRef]] = []
        main_verified_direct_primary = False
        ordered_ids = [str(value) for value in result.get("uids", ids)]
        for pmid in ordered_ids[: self.max_records_per_source]:
            item = result.get(pmid)
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or f"PubMed record {pmid}").strip()
            publication_types = [
                str(value) for value in item.get("pubtype", [])
            ]
            authors = [
                str(author.get("name"))
                for author in item.get("authors", [])
                if isinstance(author, Mapping) and author.get("name")
            ]
            content = content_by_pmid.get(pmid, _PublicationContent())
            title_target_match = bool(
                re.search(_target_token_pattern(seed.symbol), title)
            )
            if (
                content.content_hash
                and content.target_lexical_match is False
                and not title_target_match
            ):
                continue
            direction_context_basis = (
                _primary_direction_context_match_basis(
                    content.excerpt,
                    target_symbol=seed.symbol,
                    disease=brief.disease,
                    context=direction_model_context,
                )
                if content.excerpt
                else None
            )
            doi = content.doi or _summary_article_id(item, "doi")
            pmcid = content.pmcid or _summary_article_id(item, "pmc")
            record_abstract_status = (
                "unavailable"
                if abstract_error_type
                else "available"
                if content.content_hash
                else "not_available"
            )
            direction_rejection_codes = (
                _primary_direction_content_rejection_codes(
                    content,
                    title=title,
                    publication_types=publication_types,
                    target_symbol=seed.symbol,
                    disease=brief.disease,
                    intervention_direction=(
                        brief.constraints.intervention_direction
                    ),
                    direction_model_context=direction_model_context,
                    require_primary_result_context=True,
                )
            )
            verified_main_direction = not direction_rejection_codes
            content_metadata: dict[str, Any] = {
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "content_availability": (
                    "abstract_excerpt" if content.excerpt else "metadata_only"
                ),
                "abstract_retrieval_status": record_abstract_status,
                "source_location": content.source_location,
                "excerpt": content.excerpt,
                "content_hash": content.content_hash,
                **_abstract_context_metadata(content, source_review=self.ma3_source_review_enabled),
                "content_hash_algorithm": "sha256" if content.content_hash else None,
                "content_hash_scope": (
                    "normalized_full_abstract" if content.content_hash else None
                ),
                "excerpt_char_count": len(content.excerpt) if content.excerpt else 0,
                "excerpt_max_chars": _MAX_PUBLICATION_EXCERPT_CHARS,
                "excerpt_truncated": content.excerpt_truncated,
                "excerpt_selection_status": content.excerpt_selection_status,
                "excerpt_selection_policy": (
                    "same-sentence-target-disease-direction-aware-mechanistic-ranking-v2"
                ),
                "excerpt_relevance_basis": [
                    "target_symbol",
                    "disease",
                    "intervention_direction",
                    "mechanistic_endpoint",
                    "experimental_result_language",
                ],
                "query_strategy": "target-disease-direction-mechanistic-v2",
                "query_intervention_direction": (
                    brief.constraints.intervention_direction
                ),
                "claim_linkage_status": "pending_ma1_claim_reconciliation",
                "direction_disease_context_match_basis": (
                    direction_context_basis
                ),
                "direction_disease_context_match_policy": (
                    _PRIMARY_DIRECTION_CONTEXT_MATCH_POLICY
                ),
                "predeclared_model_aliases": list(
                    direction_model_context.model_aliases
                    if direction_model_context is not None
                    else ()
                ),
                "predeclared_target_variants": [
                    clause.canonical
                    for clause in (
                        direction_model_context.variant_clauses
                        if direction_model_context is not None
                        else ()
                    )
                ],
                "counter_category": content.counter_signal_category,
                "counter_signal": content.counter_signal,
                "classification_policy": (
                    "same-clause-target-counter-signal-proximity80-v1"
                    if content.counter_signal_category
                    else "conservative-direct-intervention-subject-v1"
                    if content.direct_support_rejection_codes
                    else "association-role-only-v1"
                    if _publication_has_association_candidate(content)
                    else _VERIFIED_DIRECT_PRIMARY_POLICY
                    if verified_main_direction
                    else "primary-direction-validation-failed-v1"
                ),
                "direct_support_eligible": verified_main_direction,
                "direct_support_rejection_codes": list(direction_rejection_codes),
                "review_status": (
                    "researcher_review_required"
                    if direction_rejection_codes
                    else None
                ),
                "content_storage_policy": "bounded_excerpt_only_no_full_text",
                "full_text_stored": False,
                "content_retrieval_endpoint": PUBMED_FETCH_URL,
                "ncbi_tool": self.ncbi_tool,
                "ncbi_email_configured": bool(self.ncbi_email),
                "ncbi_api_key_configured": bool(self.ncbi_api_key),
            }
            if abstract_error_type:
                content_metadata["abstract_retrieval_error_type"] = abstract_error_type
            parent_record = self._record(
                    source="PubMed",
                    external_id=pmid,
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    supports=_publication_evidence_role(
                        content,
                        counter_evidence_enabled=self.counter_evidence_enabled,
                        direct_support_eligible=verified_main_direction,
                    ),
                    query=query,
                    endpoint=PUBMED_SUMMARY_URL,
                    excerpt_text=content.excerpt,
                    hash_title_when_no_excerpt=False,
                    extra={
                        "publication_date": item.get("pubdate"),
                        "journal": item.get("fulljournalname") or item.get("source"),
                        "authors": authors,
                        "publication_types": publication_types,
                        "evidence_scope": _publication_evidence_scope(
                            content,
                            direct_support_eligible=verified_main_direction,
                        ),
                        **content_metadata,
                    },
                )
            association_record = _association_role_record(
                parent_record,
                content,
            )
            if association_record is not None:
                candidate_rejections = (
                    _primary_association_candidate_rejection_codes(
                        association_record,
                        requirement=requirement,
                        title=title,
                        publication_types=publication_types,
                    )
                )
                parent_record = parent_record.model_copy(
                    update={
                        "metadata": {
                            **parent_record.metadata,
                            "primary_association_candidate_status": (
                                "rejected"
                                if candidate_rejections
                                else "accepted"
                            ),
                            "primary_association_candidate_rejection_codes": list(
                                candidate_rejections
                            ),
                            "primary_association_candidate_retained": not bool(
                                candidate_rejections
                            ),
                        }
                    },
                    deep=True,
                )
                if not candidate_rejections:
                    association_record = _association_role_record(
                        parent_record,
                        content,
                    )
                    assert association_record is not None
                    association_candidates.append(
                        (content.association_score, association_record)
                    )
            parent_record = parent_record.model_copy(
                update={
                    "metadata": {
                        **parent_record.metadata,
                        "verified_direct_primary": verified_main_direction,
                        "verified_direct_primary_policy": (
                            _VERIFIED_DIRECT_PRIMARY_POLICY
                        ),
                        "verified_direct_primary_rejection_codes": list(
                            direction_rejection_codes
                        ),
                    }
                },
                deep=True,
            )
            if verified_main_direction:
                main_verified_direct_primary = True
            evidence.append(parent_record)

        direction_query: str | None = None
        direction_status = "not_requested"
        direction_stop_reason = (
            "main_query_verified_direct_primary_available"
            if main_verified_direct_primary
            else "brief_does_not_require_primary_sources"
        )
        direction_trace: Mapping[str, Any] = {
            "schema_version": "1.0",
            "status": direction_status,
            "stop_reason": direction_stop_reason,
            "query_policy": _PRIMARY_DIRECTION_QUERY_POLICY,
            "query_construction_policy": (
                _PRIMARY_DIRECTION_QUERY_CONSTRUCTION_POLICY
            ),
            "trigger": "main_query_no_verified_direct_primary",
            "main_query_verified_direct_primary": (
                main_verified_direct_primary
            ),
            "http_request_budget": (
                _PRIMARY_DIRECTION_HTTP_REQUEST_BUDGET
            ),
            "http_requests_used": 0,
            "raw_document_budget": (
                self.primary_direction_raw_document_budget
            ),
            "raw_hit_count": 0,
            "raw_documents_fetched": 0,
            "records_screened": 0,
            "screening_entries": [],
            "rejection_counts": {},
            "deduplication_policy": (
                "exclude_main_query_ids_then_stable_external_id_v1"
            ),
            "disease_context_match_policy": (
                _PRIMARY_DIRECTION_CONTEXT_MATCH_POLICY
            ),
            "predeclared_model_aliases": list(
                direction_model_context.model_aliases
                if direction_model_context is not None
                else ()
            ),
            "predeclared_target_variants": [
                clause.canonical
                for clause in (
                    direction_model_context.variant_clauses
                    if direction_model_context is not None
                    else ()
                )
            ],
            "full_text_stored": False,
        }
        if (
            brief.data_policy.require_primary_sources
            and not main_verified_direct_primary
        ):
            direction_query = self._pubmed_primary_direction_query(
                brief,
                seed,
            )
            try:
                direction_outcome = (
                    self._pubmed_primary_direction_supplement(
                        brief,
                        seed,
                        query=direction_query,
                        common_params=common_params,
                        excluded_ids={item.external_id for item in evidence},
                        direction_model_context=direction_model_context,
                    )
                )
            except Exception as exc:
                direction_status = "error"
                direction_stop_reason = (
                    f"unexpected_{type(exc).__name__}:fail_closed"
                )
                direction_trace = {
                    **direction_trace,
                    "status": direction_status,
                    "stop_reason": direction_stop_reason,
                    "query": direction_query,
                    "query_sha256": sha256(
                        direction_query.encode("utf-8")
                    ).hexdigest(),
                    "error_type": type(exc).__name__,
                }
            else:
                evidence.extend(direction_outcome.records)
                direction_status = direction_outcome.status
                direction_stop_reason = direction_outcome.stop_reason
                direction_trace = direction_outcome.trace

        required_association_available = any(
            _association_record_satisfies_requirement(record, requirement)
            for _, record in association_candidates
        )
        supplement_query: str | None = None
        supplement_status = "not_requested"
        supplement_stop_reason = (
            "main_query_required_association_available"
            if required_association_available
            else "brief_does_not_request_primary_association"
        )
        supplement_trace: Mapping[str, Any] = {
            "schema_version": "2.0",
            "status": supplement_status,
            "stop_reason": supplement_stop_reason,
            "required_scope": requirement.required_scope,
            "variant_clauses": [
                clause.canonical for clause in requirement.variant_clauses
            ],
            "http_request_budget": _PRIMARY_ASSOCIATION_HTTP_REQUEST_BUDGET,
            "http_requests_used": 0,
            "raw_document_budget": _PRIMARY_ASSOCIATION_RAW_DOCUMENT_BUDGET,
            "raw_documents_fetched": 0,
            "records_screened": 0,
            "screening_entries": [],
            "rejection_counts": {},
            "query_budget": 2,
            "query_count_executed": 0,
            "lanes": [],
            "full_text_stored": False,
        }
        if requirement.requested and not required_association_available:
            supplement_query_lanes = self._pubmed_primary_association_query_lanes(
                brief,
                seed,
                requirement=requirement,
            )
            supplement_query = supplement_query_lanes[0].query
            try:
                supplement_outcome = self._pubmed_primary_association_supplement(
                    brief,
                    seed,
                    query_lanes=supplement_query_lanes,
                    excluded_ids={item.external_id for item in evidence},
                    requirement=requirement,
                )
            except Exception as exc:
                supplement_records = []
                supplement_status = "error"
                supplement_stop_reason = (
                    f"unexpected_{type(exc).__name__}:fail_closed"
                )
                supplement_trace = {
                    **supplement_trace,
                    "status": supplement_status,
                    "stop_reason": supplement_stop_reason,
                    "query_sha256": sha256(
                        supplement_query.encode("utf-8")
                    ).hexdigest(),
                    "error_type": type(exc).__name__,
                }
            else:
                supplement_records = list(supplement_outcome.records)
                supplement_status = supplement_outcome.status
                supplement_stop_reason = supplement_outcome.stop_reason
                supplement_trace = supplement_outcome.trace
            evidence.extend(supplement_records)
        evidence.extend(record for _, record in association_candidates)

        counter_metadata: dict[str, Any] = {}
        if self.counter_evidence_enabled:
            counter_query = self._pubmed_counter_query(brief, seed)
            counter_query_sha256 = sha256(
                counter_query.encode("utf-8")
            ).hexdigest()
            try:
                counter_outcome = self._pubmed_counter_records(
                    brief,
                    seed,
                    counter_query,
                    common_params=common_params,
                    excluded_ids={item.external_id for item in evidence},
                )
            except Exception as exc:
                counter_metadata = {
                    "counter_search_status": "error",
                    "counter_search_source": "PubMed",
                    "counter_search_query": counter_query,
                    "counter_search_query_sha256": counter_query_sha256,
                    "counter_search_error_type": type(exc).__name__,
                    "counter_search_separate_query": True,
                    "counter_search_raw_document_budget": (
                        _PUBMED_COUNTER_RAW_DOCUMENT_BUDGET
                    ),
                    "counter_search_raw_hit_count": 0,
                    "counter_search_http_call_provenance": {},
                }
            else:
                evidence.extend(counter_outcome.records)
                counter_metadata = {
                    "counter_search_status": counter_outcome.status,
                    "counter_search_source": "PubMed",
                    "counter_search_query": counter_outcome.query,
                    "counter_search_query_sha256": (
                        counter_outcome.query_sha256
                    ),
                    "counter_search_separate_query": True,
                    "counter_search_raw_document_budget": (
                        _PUBMED_COUNTER_RAW_DOCUMENT_BUDGET
                    ),
                    "counter_search_raw_hit_count": (
                        counter_outcome.raw_hit_count
                    ),
                    "counter_search_http_call_provenance": dict(
                        counter_outcome.http_call_provenance
                    ),
                }
            evidence = [
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            **counter_metadata,
                        }
                    },
                    deep=True,
                )
                for item in evidence
            ]
        epmc_oa_trace: dict[str, Any] | None = None
        if self.epmc_oa_fulltext_one_hop_v1:
            try:
                supplemental, epmc_oa_trace = (
                    self._pubmed_epmc_oa_fulltext_escalation(
                        brief,
                        seed,
                        pubmed_records=evidence,
                        requirement=requirement,
                    )
                )
            except Exception as exc:
                supplemental = []
                epmc_oa_trace = self._epmc_oa_contained_error_trace(
                    brief,
                    abstract_records=evidence,
                    requirement=requirement,
                    error_type=type(exc).__name__,
                )
                epmc_oa_trace["trigger"]["trigger_scope"] = (
                    "pubmed_pmcid_cross_source"
                )
                epmc_oa_trace["cross_source_escalation"] = True
            evidence.extend(supplemental)
        if epmc_oa_trace is not None:
            evidence = [
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "epmc_oa_fulltext_one_hop_trace": deepcopy(
                                epmc_oa_trace
                            ),
                        }
                    },
                    deep=True,
                )
                for item in evidence
            ]
            with self._cache_lock:
                self._last_epmc_oa_fulltext_trace = deepcopy(epmc_oa_trace)
        selected = _select_publication_records_with_role_budget(
            evidence,
            max_records=self.max_records_per_source,
        )
        return [
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "primary_association_supplement_status": supplement_status,
                        "primary_association_supplement_query": supplement_query,
                        "primary_association_supplement_stop_reason": (
                            supplement_stop_reason
                        ),
                        "primary_association_supplement_query_budget": 2,
                        "primary_association_supplement_trace": dict(
                            supplement_trace
                        ),
                        "primary_direction_supplement_status": (
                            direction_status
                        ),
                        "primary_direction_supplement_query": (
                            direction_query
                        ),
                        "primary_direction_supplement_stop_reason": (
                            direction_stop_reason
                        ),
                        "primary_direction_supplement_query_policy": (
                            _PRIMARY_DIRECTION_QUERY_POLICY
                        ),
                        "primary_direction_supplement_trace": dict(
                            direction_trace
                        ),
                    }
                },
                deep=True,
            )
            for item in selected
        ]

    def _pubmed_epmc_oa_fulltext_escalation(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        *,
        pubmed_records: Sequence[EvidenceRef],
        requirement: _PrimaryAssociationRequirement,
    ) -> tuple[list[EvidenceRef], dict[str, Any]]:
        """Resolve PubMed PMCIDs through Europe PMC and inspect eligible OA JATS.

        PubMed remains the discovery source.  Europe PMC is queried only for
        the exact PMCIDs already present in the retrieved PubMed records; it
        cannot introduce an unrelated paper or promote metadata to evidence.
        """

        needs_direct, needs_association, needs_assay_trace = (
            self._epmc_oa_evidence_needs(
            pubmed_records,
            brief=brief,
            requirement=requirement,
            )
        )
        source_local_needs = (needs_direct, needs_association)
        requested_roles = set(seed.discovery_metadata.get("g0_followup_roles", ())) & {"direction", "association"}
        needs_direct = needs_direct or "direction" in requested_roles
        needs_association = needs_association or "association" in requested_roles
        trigger = {
            "g0_requested_roles": sorted(requested_roles),
            "trigger_scope": "pubmed_pmcid_cross_source",
            "global_cross_source_sufficiency_not_evaluated": False,
            "research_depth": brief.research_depth,
            "abstract_verified_direct_primary_available": not source_local_needs[0],
            "required_association_requested": requirement.requested,
            "abstract_required_association_available": (
                not source_local_needs[1] if requirement.requested else None
            ),
            "required_assay_trace_requested": (
                self._epmc_oa_assay_trace_required(brief)
            ),
            "abstract_required_assay_trace_available": (
                not needs_assay_trace
                if self._epmc_oa_assay_trace_required(brief)
                else None
            ),
            "abstract_evidence_insufficient": bool(
                any(source_local_needs) or needs_assay_trace
            ),
            "needed_roles": [
                role
                for needed, role in (
                    (needs_direct, "verified_direct_primary"),
                    (needs_association, "required_primary_association"),
                    (needs_assay_trace, "required_assay_trace"),
                )
                if needed
            ],
        }
        if not (needs_direct or needs_association or needs_assay_trace):
            return [], self._epmc_oa_trace(
                None,
                trigger=trigger,
                status="not_needed",
                stop_reason=(
                    "pubmed_abstract_required_evidence_sufficient"
                ),
                result_records=(),
            )

        pmcids = list(
            dict.fromkeys(
                normalized
                for record in pubmed_records
                if (
                    normalized := _normalize_pmcid(
                        record.metadata.get("pmcid")
                    )
                )
            )
        )[:_EPMC_OA_PARENT_LIMIT]
        if not pmcids:
            return [], self._epmc_oa_trace(
                None,
                trigger=trigger,
                status="skipped",
                stop_reason="skipped_no_pubmed_pmcid",
                result_records=(),
            )

        query = " OR ".join(f"PMCID:{pmcid}" for pmcid in pmcids)
        response = self.client.get(
            EUROPE_PMC_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "pageSize": len(pmcids),
                "resultType": "core",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        raw_results = response.json().get("resultList", {}).get("result", [])
        allowed = set(pmcids)
        exact_results = [
            item
            for item in raw_results
            if isinstance(item, Mapping)
            and _normalize_pmcid(item.get("pmcid")) in allowed
        ]
        supplemental, trace = self._europe_pmc_oa_fulltext_one_hop(
            brief,
            seed,
            query=query,
            search_results=exact_results,
            abstract_records=pubmed_records,
            requirement=requirement,
            preferred_pmcid_order=pmcids,
        )
        trace["trigger"] = trigger
        trace["cross_source_escalation"] = True
        trace["discovery_source"] = "PubMed"
        trace["resolution_source"] = "Europe PMC"
        trace["requested_pmcids"] = pmcids
        trace["resolved_exact_pmcids"] = [
            _normalize_pmcid(item.get("pmcid"))
            for item in exact_results
        ]
        trace["metadata_query_sha256"] = sha256(
            query.encode("utf-8")
        ).hexdigest()
        trace.pop("trace_sha256", None)
        trace["trace_sha256"] = sha256(
            json.dumps(
                trace,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return supplemental, trace

    def _pubmed_primary_direction_supplement(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        *,
        query: str,
        common_params: Mapping[str, str],
        excluded_ids: set[str],
        direction_model_context: _PrimaryDirectionModelContext | None,
        required_molecular_scopes: tuple[str, ...] = (),
    ) -> _PrimaryDirectionSupplementOutcome:
        """Run one precision direction lane over PubMed's complete result window."""
        if required_molecular_scopes:
            bindings=seed.discovery_metadata.get(VARIANT_SEARCH_KEY,[])
            validate_variant_search(bindings,seed.symbol)
            if tuple(sorted({b["label"] for b in bindings}))!=required_molecular_scopes:
                raise ValueError("VARIANT_DIRECTION_REQUIREMENT_NOT_SOURCE_BOUND")
        http_requests_used = 0
        http_request_budget = _PRIMARY_DIRECTION_HTTP_REQUEST_BUDGET
        raw_hit_count = 0
        search_result_count = 0
        returned_id_count = 0
        duplicate_id_count = 0
        excluded_main_query_id_count = 0
        raw_documents_fetched = 0
        records_screened = 0
        summary_batch_count = 0
        summary_batches_completed = 0
        abstract_batch_count = 0
        abstract_batches_completed = 0
        call_provenance: list[dict[str, Any]] = []
        summary_batch_audits: list[dict[str, Any]] = []
        abstract_batch_audits: list[dict[str, Any]] = []
        screening_entries: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        source_review_records: list[EvidenceRef] = []
        source_review_eligible_count = 0
        query_sha256 = sha256(query.encode("utf-8")).hexdigest()

        def identifier_sequence_sha256(values: Sequence[str]) -> str:
            return sha256(
                json.dumps(
                    list(values),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        def reject(entry: dict[str, Any], *codes: str) -> None:
            normalized = tuple(dict.fromkeys(code for code in codes if code))
            entry["status"] = "rejected"
            entry["rejection_codes"] = list(normalized)
            for code in normalized:
                rejection_counts[code] = rejection_counts.get(code, 0) + 1

        def finish(
            *,
            status: str,
            stop_reason: str,
            record: EvidenceRef | None = None,
            error_stage: str | None = None,
            error_type: str | None = None,
        ) -> _PrimaryDirectionSupplementOutcome:
            if http_requests_used != len(call_provenance):
                raise RuntimeError(
                    "primary_direction_http_provenance_invariant_failed"
                )
            trace: dict[str, Any] = {
                "schema_version": "1.0",
                **({"required_molecular_scopes":list(required_molecular_scopes),
                    "molecular_scope_selection_policy":"existing_scoped_intervention_predicate_v1",
                    "variant_search_query_policy":"compact_bound_variant_disease_action_endpoint_v1",
                    "search_scope_is_not_construct_identity":True} if required_molecular_scopes else {}),
                "status": status,
                "stop_reason": stop_reason,
                "coverage_scope": (
                    "single_relevance_ranked_pubmed_result_window_screen"
                ),
                "query": query,
                "query_sha256": query_sha256,
                "query_policy": _PRIMARY_DIRECTION_QUERY_POLICY,
                "query_construction_policy": (
                    _PRIMARY_DIRECTION_QUERY_CONSTRUCTION_POLICY
                ),
                "trigger": "main_query_no_verified_direct_primary",
                "main_query_verified_direct_primary": False,
                "http_request_budget": (
                    http_request_budget
                ),
                "http_requests_used": http_requests_used,
                "http_request_accounting_unit": (
                    "logical_adapter_request_before_transport_retries"
                ),
                "transport_attempt_semantics": (
                    "retry_attempts_owned_by_injected_http_client;"
                    "one_adapter_request_counted_per_ncbi_get"
                ),
                "raw_document_budget": (
                    self.primary_direction_raw_document_budget
                ),
                "raw_hit_count": raw_hit_count,
                "search_result_count": search_result_count,
                "returned_id_count": returned_id_count,
                "duplicate_id_count": duplicate_id_count,
                "excluded_main_query_id_count": (
                    excluded_main_query_id_count
                ),
                "raw_documents_fetched": raw_documents_fetched,
                "records_screened": records_screened,
                "summary_batch_size": (
                    _PRIMARY_DIRECTION_SUMMARY_BATCH_SIZE
                ),
                "summary_batch_count": summary_batch_count,
                "summary_batches_completed": summary_batches_completed,
                "summary_batch_audits": list(summary_batch_audits),
                "abstract_batch_size": (
                    _PRIMARY_DIRECTION_FETCH_BATCH_SIZE
                ),
                "abstract_batch_count": abstract_batch_count,
                "abstract_batches_completed": abstract_batches_completed,
                "abstract_batch_audits": list(abstract_batch_audits),
                "unsearched_result_count": max(
                    0,
                    search_result_count - returned_id_count,
                ),
                "screening_entries": list(screening_entries),
                "rejection_counts": {
                    key: rejection_counts[key]
                    for key in sorted(rejection_counts)
                },
                "http_call_provenance": list(call_provenance),
                "http_call_provenance_complete": True,
                "deduplication_policy": (
                    "exclude_main_query_ids_then_stable_external_id_v1"
                ),
                "disease_match_policy": (
                    _PRIMARY_DIRECTION_CONTEXT_MATCH_POLICY
                ),
                "predeclared_model_aliases": list(
                    direction_model_context.model_aliases
                    if direction_model_context is not None
                    else ()
                ),
                "predeclared_target_variants": [
                    clause.canonical
                    for clause in (
                        direction_model_context.variant_clauses
                        if direction_model_context is not None
                        else ()
                    )
                ],
                "direction_match_policy": (
                    "direct_target_intervention_plus_measured_positive_result_v1"
                ),
                "content_storage_policy": (
                    "bounded_excerpt_only_no_full_text"
                ),
                "full_text_stored": False,
            }
            if self.ma3_source_review_enabled:
                trace["source_review_policy"] = "bounded_original_sources_pending_ma3_v1"
                trace["source_review_retained_count"] = len(source_review_records)
                trace["source_review_eligible_count"] = source_review_eligible_count
                trace["source_review_omitted_count"] = max(0, source_review_eligible_count - len(source_review_records))
                trace["source_review_record_limit"] = self.max_records_per_source
                trace["source_review_selection"] = "stable PubMed relevance order; no semantic pass required"
                trace["scientific_approval"] = False
            if error_stage:
                trace["error_stage"] = error_stage
            if error_type:
                trace["error_type"] = error_type
            prepared_records: list[EvidenceRef] = []
            originals = ([record] if record is not None else list(source_review_records)
                if self.ma3_source_review_enabled and status != "error" else [])
            for original in originals:
                prepared = original.model_copy(
                    update={
                        "metadata": {
                            **original.metadata,
                            "primary_direction_supplement": True,
                            "primary_direction_supplement_status": status,
                            "primary_direction_supplement_stop_reason": (
                                stop_reason
                            ),
                            "primary_direction_supplement_query": query,
                            "primary_direction_supplement_query_policy": (
                                _PRIMARY_DIRECTION_QUERY_POLICY
                            ),
                            "primary_direction_supplement_query_construction_policy": (
                                _PRIMARY_DIRECTION_QUERY_CONSTRUCTION_POLICY
                            ),
                            "primary_direction_supplement_trace": trace,
                        }
                    },
                    deep=True,
                )
                prepared_records.append(prepared)
            return _PrimaryDirectionSupplementOutcome(
                records=tuple(prepared_records),
                status=status,
                stop_reason=stop_reason,
                trace=trace,
            )

        search_params: dict[str, Any] = {
            **dict(common_params),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": self.primary_direction_raw_document_budget,
            "sort": "relevance",
        }
        search: Any | None = None
        try:
            http_requests_used += 1
            search = self._ncbi_get(
                PUBMED_SEARCH_URL,
                params=search_params,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            search.raise_for_status()
        except Exception as exc:
            call_provenance.append(
                _bounded_http_failure_provenance(
                    endpoint=PUBMED_SEARCH_URL,
                    params=search_params,
                    error=exc,
                    response=search,
                )
            )
            return finish(
                status="error",
                stop_reason=(
                    f"search_{type(exc).__name__}:fail_closed"
                ),
                error_stage="search",
                error_type=type(exc).__name__,
            )
        call_provenance.append(
            _bounded_http_call_provenance(
                search,
                endpoint=PUBMED_SEARCH_URL,
                params=search_params,
            )
        )
        try:
            search_result_count, raw_ids = _validate_pubmed_search_payload(
                search.json(),
                raw_document_budget=(
                    self.primary_direction_raw_document_budget
                ),
            )
        except Exception as exc:
            reason_code = (
                exc.reason_code
                if isinstance(exc, _PubMedSearchValidationError)
                else type(exc).__name__
            )
            return finish(
                status="error",
                stop_reason=f"search_schema_{reason_code}:fail_closed",
                error_stage="search_schema",
                error_type=type(exc).__name__,
            )

        raw_hit_count = len(raw_ids)
        deduplicated_ids = list(dict.fromkeys(raw_ids))
        returned_id_count = len(deduplicated_ids)
        duplicate_id_count = raw_hit_count - returned_id_count
        excluded_main_query_id_count = sum(
            identifier in excluded_ids for identifier in deduplicated_ids
        )
        ids = [
            identifier
            for identifier in deduplicated_ids
            if identifier not in excluded_ids
        ]
        summary_batch_count = (
            len(ids) + _PRIMARY_DIRECTION_SUMMARY_BATCH_SIZE - 1
        ) // _PRIMARY_DIRECTION_SUMMARY_BATCH_SIZE
        abstract_batch_count = (
            len(ids) + _PRIMARY_DIRECTION_FETCH_BATCH_SIZE - 1
        ) // _PRIMARY_DIRECTION_FETCH_BATCH_SIZE
        http_request_budget = (
            1 + summary_batch_count + abstract_batch_count
        )
        if not raw_ids:
            return finish(
                status="completed_empty",
                stop_reason="search_returned_zero_hits",
            )
        if not ids:
            return finish(
                status="completed_bounded_without_verified_result",
                stop_reason="search_returned_no_unseen_candidates",
            )

        summary_result: dict[str, Mapping[str, Any]] = {}
        for batch_index, batch_start in enumerate(
            range(0, len(ids), _PRIMARY_DIRECTION_SUMMARY_BATCH_SIZE),
            1,
        ):
            batch_ids = ids[
                batch_start : batch_start + _PRIMARY_DIRECTION_SUMMARY_BATCH_SIZE
            ]
            summary_params = {
                **dict(common_params),
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "json",
            }
            audit: dict[str, Any] = {
                "batch_index": batch_index,
                "logical_request_ordinal": http_requests_used + 1,
                "requested_count": len(batch_ids),
                "requested_id_sequence_sha256": (
                    identifier_sequence_sha256(batch_ids)
                ),
                "response_body_stored": False,
            }
            response_status_code: int | None = None
            provenance_recorded = False
            summary: Any | None = None
            try:
                http_requests_used += 1
                summary = self._ncbi_get(
                    PUBMED_SUMMARY_URL,
                    params=summary_params,
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                )
                response_status_code = int(
                    getattr(summary, "status_code", 0) or 0
                )
                summary.raise_for_status()
                call_provenance.append(
                    _bounded_http_call_provenance(
                        summary,
                        endpoint=PUBMED_SUMMARY_URL,
                        params=summary_params,
                    )
                )
                provenance_recorded = True
                response_payload = summary.json()
                result_payload = (
                    response_payload.get("result")
                    if isinstance(response_payload, Mapping)
                    else None
                )
                raw_returned_ids = (
                    result_payload.get("uids")
                    if isinstance(result_payload, Mapping)
                    else None
                )
                if (
                    isinstance(raw_returned_ids, Sequence)
                    and not isinstance(
                        raw_returned_ids,
                        (str, bytes, bytearray),
                    )
                ):
                    normalized_returned_ids = [
                        str(value).strip() for value in raw_returned_ids
                    ]
                    audit["returned_count"] = len(normalized_returned_ids)
                    audit["returned_id_sequence_sha256"] = (
                        identifier_sequence_sha256(normalized_returned_ids)
                    )
                validated = _validate_pubmed_summary_batch(
                    result_payload,
                    batch_ids,
                )
                summary_result.update(validated)
                summary_batches_completed += 1
                audit.update(
                    {
                        "status": "completed",
                        "status_code": response_status_code,
                        "identity_validation": "exact_ordered_match",
                    }
                )
                summary_batch_audits.append(audit)
            except Exception as exc:
                if not provenance_recorded:
                    call_provenance.append(
                        _bounded_http_failure_provenance(
                            endpoint=PUBMED_SUMMARY_URL,
                            params=summary_params,
                            error=exc,
                            response=summary,
                        )
                    )
                reason_code = (
                    exc.reason_code
                    if isinstance(exc, _PubMedBatchValidationError)
                    else type(exc).__name__
                )
                audit.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_reason_code": reason_code,
                    }
                )
                if response_status_code is not None:
                    audit["status_code"] = response_status_code
                summary_batch_audits.append(audit)
                return finish(
                    status="error",
                    stop_reason=(
                        f"summary_batch_{reason_code}:fail_closed"
                    ),
                    error_stage="summary_batch",
                    error_type=type(exc).__name__,
                )

        content_by_pmid: dict[str, _PublicationContent] = {}
        for batch_index, batch_start in enumerate(
            range(0, len(ids), _PRIMARY_DIRECTION_FETCH_BATCH_SIZE),
            1,
        ):
            batch_ids = ids[
                batch_start : batch_start + _PRIMARY_DIRECTION_FETCH_BATCH_SIZE
            ]
            fetch_params = {
                **dict(common_params),
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "xml",
                "rettype": "abstract",
            }
            audit = {
                "batch_index": batch_index,
                "logical_request_ordinal": http_requests_used + 1,
                "requested_count": len(batch_ids),
                "requested_id_sequence_sha256": (
                    identifier_sequence_sha256(batch_ids)
                ),
                "response_body_stored": False,
            }
            response_status_code = None
            provenance_recorded = False
            fetch: Any | None = None
            try:
                http_requests_used += 1
                fetch = self._ncbi_get(
                    PUBMED_FETCH_URL,
                    params=fetch_params,
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                )
                response_status_code = int(
                    getattr(fetch, "status_code", 0) or 0
                )
                fetch.raise_for_status()
                call_provenance.append(
                    _bounded_http_call_provenance(
                        fetch,
                        endpoint=PUBMED_FETCH_URL,
                        params=fetch_params,
                    )
                )
                provenance_recorded = True
                xml_text = _response_text(fetch)
                returned_batch_ids = _pubmed_xml_record_ids(xml_text)
                audit["returned_count"] = len(returned_batch_ids)
                audit["returned_id_sequence_sha256"] = (
                    identifier_sequence_sha256(returned_batch_ids)
                )
                _validate_pubmed_batch_ids(batch_ids, returned_batch_ids)
                content_by_pmid.update(
                    _parse_pubmed_content(
                        xml_text,
                        target_symbol=seed.symbol,
                        disease=brief.disease,
                        intervention_direction=(
                            brief.constraints.intervention_direction
                        ),
                        direction_model_context=direction_model_context,
                    )
                )
                raw_documents_fetched += len(batch_ids)
                abstract_batches_completed += 1
                audit.update(
                    {
                        "status": "completed",
                        "status_code": response_status_code,
                        "identity_validation": "exact_ordered_match",
                    }
                )
                abstract_batch_audits.append(audit)
            except Exception as exc:
                if not provenance_recorded:
                    call_provenance.append(
                        _bounded_http_failure_provenance(
                            endpoint=PUBMED_FETCH_URL,
                            params=fetch_params,
                            error=exc,
                            response=fetch,
                        )
                    )
                reason_code = (
                    exc.reason_code
                    if isinstance(exc, _PubMedBatchValidationError)
                    else type(exc).__name__
                )
                audit.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_reason_code": reason_code,
                    }
                )
                if response_status_code is not None:
                    audit["status_code"] = response_status_code
                abstract_batch_audits.append(audit)
                return finish(
                    status="error",
                    stop_reason=(
                        f"abstract_batch_{reason_code}:fail_closed"
                    ),
                    error_stage="abstract_batch",
                    error_type=type(exc).__name__,
                )

        # Keep a source requiring the new document-unit check separate from
        # verified single-sentence results. Never stop the search early for it.
        deferred_document_source: EvidenceRef | None = None
        ordered_ids = list(ids)
        for rank, pmid in enumerate(ordered_ids, 1):
            if pmid not in ids:
                continue
            records_screened += 1
            entry: dict[str, Any] = {
                "external_id": pmid,
                "rank": rank,
            }
            screening_entries.append(entry)
            item = summary_result.get(pmid)
            if not isinstance(item, Mapping):
                reject(entry, "summary_metadata_missing")
                continue
            title = str(
                item.get("title") or f"PubMed record {pmid}"
            ).strip()
            publication_types = [
                str(value) for value in item.get("pubtype", [])
            ]
            content = content_by_pmid.get(pmid)
            if content is None:
                reject(entry, "abstract_unavailable")
                continue
            from drugpilot.tools.evidence_gap import content_observation
            entry["extraction_observation"] = content_observation(content)
            rejection_codes = _primary_direction_content_rejection_codes(
                content,
                title=title,
                publication_types=publication_types,
                target_symbol=seed.symbol,
                disease=brief.disease,
                intervention_direction=(
                    brief.constraints.intervention_direction
                ),
                direction_model_context=direction_model_context,
                require_primary_result_context=True,
            )
            if required_molecular_scopes:
                # Search termination must satisfy the requested source-scoped
                # role, not only a canonical-protein action elsewhere in a
                # mutant-cell sentence. The downstream MA1 gate is unchanged.
                from drugpilot.agents.ma1 import _scope_direct_intervention_present
                if not any(_scope_direct_intervention_present(content.excerpt or "",seed.symbol,
                        brief.constraints.intervention_direction,label) for label in required_molecular_scopes):
                    rejection_codes=(*rejection_codes,"requested_molecular_scope_direct_intervention_missing")
                entry["required_molecular_scopes"]=list(required_molecular_scopes)
            if self.ma3_source_review_enabled:
                # These conditions describe scientific linkage, not whether an
                # original document may be read by MA3. Keep their diagnostics,
                # but never convert collection into direct/primary approval.
                if rejection_codes:
                    reject(entry, *rejection_codes)
                else:
                    entry["status"] = "legacy_primary_conditions_met"
                    entry["rejection_codes"] = []
                entry["legacy_screening_status"] = entry["status"]
                review_record = self._source_only_direction_record(
                    content, pmid=pmid, title=title, publication_types=publication_types,
                    query=query, direction=brief.constraints.intervention_direction,
                    rejection_codes=rejection_codes,
                )
                if review_record is None:
                    entry["source_review_disposition"] = "source_text_unavailable_or_invalid"
                else:
                    source_review_eligible_count += 1
                    if len(source_review_records) < self.max_records_per_source:
                        source_review_records.append(review_record)
                        entry["source_review_disposition"] = "retained_pending_ma3"
                        entry["status"] = "source_review_pending"
                    else:
                        entry["source_review_disposition"] = "omitted_record_limit"
                continue
            if rejection_codes:
                reject(entry, *rejection_codes)
                context = content.interpretation_context or {}
                if (deferred_document_source is None and content.excerpt
                        and context.get("status") == "complete"
                        and not _primary_publication_metadata_rejection_codes(
                            title=title, publication_types=publication_types)):
                    from drugpilot.agents.document_result_units import candidate_units
                    units = candidate_units(brief, seed.symbol, context.get("text"))
                    if units:
                        # Preserve the short source excerpt and its rejection
                        # codes, without calling it verified direction support.
                        deferred_document_source = self._record(
                            source="PubMed", external_id=pmid, title=title,
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            supports="context", query=query, endpoint=PUBMED_FETCH_URL,
                            excerpt_text=content.excerpt, hash_title_when_no_excerpt=False,
                            extra={"pmid":pmid, "pmcid":content.pmcid, "doi":content.doi,
                                "publication_types":publication_types, "excerpt":content.excerpt,
                                "source_location":content.source_location, "content_hash":content.content_hash,
                                "content_hash_algorithm":"sha256", "content_hash_scope":"normalized_full_abstract",
                                **_abstract_context_metadata(content), "content_availability":"abstract_excerpt",
                                "full_text_stored":False,"excerpt_selection_status":content.excerpt_selection_status,
                                "excerpt_selection_policy":"same-sentence-target-disease-direction-aware-mechanistic-ranking-v2",
                                "query_strategy":_PRIMARY_DIRECTION_QUERY_POLICY,
                                "query_intervention_direction":brief.constraints.intervention_direction,
                                "direct_support_eligible":False,"verified_direct_primary":False,
                                "direct_support_rejection_codes":list(rejection_codes),
                                "verified_direct_primary_rejection_codes":list(rejection_codes),
                                "document_unit_validation_pending":True,"document_unit_candidate_count":len(units),
                                "content_storage_policy":"bounded_primary_excerpt_with_complete_abstract_context",
                                "evidence_scope":"source context awaiting document-unit and independent semantic checks",
                                "additional_network_call":True})
                        entry["document_unit_source_preserved"] = True
                continue
            excerpt = content.excerpt
            assert excerpt is not None
            assert content.source_location is not None
            assert content.content_hash is not None
            context_match_basis = _primary_direction_context_match_basis(
                excerpt,
                target_symbol=seed.symbol,
                disease=brief.disease,
                context=direction_model_context,
            )
            assert context_match_basis is not None
            entry["status"] = "selected"
            entry["rejection_codes"] = []
            entry["disease_context_match_basis"] = context_match_basis
            record = self._record(
                source="PubMed",
                external_id=pmid,
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                supports="support",
                query=query,
                endpoint=PUBMED_FETCH_URL,
                excerpt_text=excerpt,
                hash_title_when_no_excerpt=False,
                extra={
                    "pmid": pmid,
                    "pmcid": content.pmcid,
                    "doi": (
                        content.doi or _summary_article_id(item, "doi")
                    ),
                    "publication_date": item.get("pubdate"),
                    "journal": (
                        item.get("fulljournalname") or item.get("source")
                    ),
                    "publication_types": publication_types,
                    "content_availability": "abstract_excerpt",
                    "abstract_retrieval_status": "available",
                    "source_location": content.source_location,
                    "excerpt": excerpt,
                    "content_hash": content.content_hash,
                    **_abstract_context_metadata(content),
                    "content_hash_algorithm": "sha256",
                    "content_hash_scope": "normalized_full_abstract",
                    "excerpt_char_count": len(excerpt),
                    "excerpt_max_chars": _MAX_PUBLICATION_EXCERPT_CHARS,
                    "excerpt_truncated": content.excerpt_truncated,
                    "excerpt_selection_status": (
                        content.excerpt_selection_status
                    ),
                    "excerpt_selection_policy": (
                        _COMPOUND_LINKED_DIRECTION_EXCERPT_POLICY
                        if content.excerpt_selection_status
                        == "selected_compound_linked_document_pair"
                        else "same-sentence-target-disease-direction-aware-"
                        "mechanistic-ranking-v2"
                    ),
                    "excerpt_relevance_basis": [
                        "target_symbol",
                        "exact_base_disease_alias",
                        "direct_target_intervention",
                        "measured_endpoint",
                        "positive_result_language",
                    ],
                    "query_strategy": _PRIMARY_DIRECTION_QUERY_POLICY,
                    "query_construction_policy": (
                        _PRIMARY_DIRECTION_QUERY_CONSTRUCTION_POLICY
                    ),
                    "query_intervention_direction": (
                        brief.constraints.intervention_direction
                    ),
                    "claim_linkage_status": (
                        "pending_ma1_claim_reconciliation"
                    ),
                    "direction_disease_context_match_basis": (
                        context_match_basis
                    ),
                    "direction_disease_context_match_policy": (
                        _PRIMARY_DIRECTION_CONTEXT_MATCH_POLICY
                    ),
                    "predeclared_model_aliases": list(
                        direction_model_context.model_aliases
                        if direction_model_context is not None
                        else ()
                    ),
                    "predeclared_target_variants": [
                        clause.canonical
                        for clause in (
                            direction_model_context.variant_clauses
                            if direction_model_context is not None
                            else ()
                        )
                    ],
                    "counter_category": None,
                    "counter_signal": None,
                    "classification_policy": (
                        "direct-target-intervention-measured-positive-result-v1"
                    ),
                    "direct_support_eligible": True,
                    "verified_direct_primary": True,
                    "verified_direct_primary_policy": (
                        _VERIFIED_DIRECT_PRIMARY_POLICY
                    ),
                    "verified_direct_primary_rejection_codes": [],
                    "direct_support_rejection_codes": [],
                    "review_status": None,
                    "evidence_scope": (
                        "primary-publication direct target-direction measured "
                        "positive response excerpt; not causal proof"
                    ),
                    "additional_network_call": True,
                    "content_storage_policy": (
                        "bounded_excerpt_only_no_full_text"
                    ),
                    "full_text_stored": False,
                    "content_retrieval_endpoint": PUBMED_FETCH_URL,
                },
            )
            return finish(
                status="completed_with_results",
                stop_reason="first_verified_direct_primary_selected",
                record=record,
            )

        if source_review_records:
            return finish(status="completed_with_source_review_context",
                stop_reason="bounded_original_sources_retained_for_ma3")
        if deferred_document_source is not None:
            return finish(status="completed_with_context_pending_document_unit_validation",
                stop_reason="no_verified_single_excerpt_document_unit_check_pending",
                record=deferred_document_source)
        return finish(
            status="completed_bounded_without_verified_result",
            stop_reason=(
                "raw_document_budget_exhausted_without_verified_direct_primary"
                if search_result_count > returned_id_count
                else "returned_candidates_exhausted_without_verified_direct_primary"
            ),
        )

    def _source_only_direction_record(
        self, content: _PublicationContent, *, pmid: str, title: str,
        publication_types: Sequence[str], query: str, direction: str,
        rejection_codes: Sequence[str],
    ) -> EvidenceRef | None:
        """Bind a retrieved original for MA3, with zero primary-claim authority."""
        record = self._record(
            source="PubMed", external_id=pmid, title=title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", supports="context",
            query=query, endpoint=PUBMED_FETCH_URL, excerpt_text=content.excerpt,
            hash_title_when_no_excerpt=False,
            extra={"pmid": pmid, "pmcid": content.pmcid, "doi": content.doi,
                "publication_types": list(publication_types), "excerpt": content.excerpt,
                "source_location": content.source_location, "content_hash": content.content_hash,
                "content_hash_algorithm": "sha256", "content_hash_scope": "normalized_full_abstract",
                **_abstract_context_metadata(content, source_review=self.ma3_source_review_enabled),
                "content_availability": "abstract_excerpt" if content.excerpt else "complete_abstract_context",
                "excerpt_selection_status": content.excerpt_selection_status,
                "query_strategy": _PRIMARY_DIRECTION_QUERY_POLICY,
                "query_intervention_direction": direction, "full_text_stored": False,
                "direct_support_eligible": False, "verified_direct_primary": False,
                "direct_support_rejection_codes": list(rejection_codes),
                "verified_direct_primary_rejection_codes": list(rejection_codes),
                "counter_category": content.counter_signal_category,
                "counter_signal": content.counter_signal,
                "ma3_source_review_required": True, "scientific_approval": False,
                "claim_linkage_status": "pending_ma3_source_bound_review",
                "evidence_scope": "Original publication context only; semantic diagnostics are not collection filters",
                "content_storage_policy": "original_excerpt_and_complete_hash_bound_abstract_when_available",
                "additional_network_call": True},
        )
        return record if source_review_text(record) is not None else None

    def _pubmed_counter_records(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        query: str,
        *,
        common_params: Mapping[str, str],
        excluded_ids: set[str],
    ) -> _PubMedCounterOutcome:
        """Run one distinct, bounded PubMed counter-evidence lane."""

        search_params: dict[str, Any] = {
            **dict(common_params),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": _PUBMED_COUNTER_RAW_DOCUMENT_BUDGET,
            "sort": "relevance",
        }
        search_response = self._ncbi_get(
            PUBMED_SEARCH_URL,
            params=search_params,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        search_response.raise_for_status()
        search_provenance = _bounded_http_call_provenance(
            search_response,
            endpoint=PUBMED_SEARCH_URL,
            params=search_params,
        )
        payload = search_response.json()
        raw_ids = [
            str(value)
            for value in payload.get("esearchresult", {}).get("idlist", [])
        ][:_PUBMED_COUNTER_RAW_DOCUMENT_BUDGET]
        ids = list(
            dict.fromkeys(
                value
                for value in raw_ids
                if value and value not in excluded_ids
            )
        )
        query_sha256 = sha256(query.encode("utf-8")).hexdigest()
        if not ids:
            return _PubMedCounterOutcome(
                records=(),
                status="completed_empty",
                query=query,
                query_sha256=query_sha256,
                raw_hit_count=len(raw_ids),
                http_call_provenance=search_provenance,
            )

        summary_response = self._ncbi_get(
            PUBMED_SUMMARY_URL,
            params={
                **dict(common_params),
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        summary_response.raise_for_status()
        summary_result = summary_response.json().get("result", {})

        fetch_response = self._ncbi_get(
            PUBMED_FETCH_URL,
            params={
                **dict(common_params),
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        fetch_response.raise_for_status()
        content_by_pmid = _parse_pubmed_content(
            _response_text(fetch_response),
            target_symbol=seed.symbol,
            disease=brief.disease,
            intervention_direction=brief.constraints.intervention_direction,
        )

        records: list[EvidenceRef] = []
        ordered_ids = [str(value) for value in summary_result.get("uids", ids)]
        for pmid in ordered_ids[:_PUBMED_COUNTER_RAW_DOCUMENT_BUDGET]:
            item = summary_result.get(pmid)
            if not isinstance(item, Mapping):
                continue
            content = content_by_pmid.get(pmid)
            if content is None or not content.counter_signal_category:
                continue
            title = str(
                item.get("title") or f"PubMed record {pmid}"
            ).strip()
            publication_types = [
                str(value) for value in item.get("pubtype", [])
            ]
            if _primary_publication_metadata_rejection_codes(
                title,
                publication_types,
            ):
                continue
            excerpt = content.excerpt
            if not excerpt or not content.source_location or not content.content_hash:
                continue
            records.append(
                self._record(
                    source="PubMed",
                    external_id=pmid,
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    supports="counter",
                    query=query,
                    endpoint=PUBMED_SUMMARY_URL,
                    excerpt_text=excerpt,
                    hash_title_when_no_excerpt=False,
                    extra={
                        "pmid": pmid,
                        "pmcid": content.pmcid,
                        "doi": content.doi,
                        "publication_types": publication_types,
                        "content_availability": "abstract_excerpt",
                        "abstract_retrieval_status": "available",
                        "source_location": content.source_location,
                        "excerpt": excerpt,
                        "content_hash": content.content_hash,
                        **_abstract_context_metadata(content),
                        "content_hash_algorithm": "sha256",
                        "content_hash_scope": "normalized_full_abstract",
                        "excerpt_char_count": len(excerpt),
                        "excerpt_max_chars": _MAX_PUBLICATION_EXCERPT_CHARS,
                        "excerpt_truncated": content.excerpt_truncated,
                        "excerpt_selection_status": "selected_counter_signal",
                        "counter_search_status": "completed_with_results",
                        "counter_search_source": "PubMed",
                        "counter_search_query": query,
                        "counter_search_query_sha256": query_sha256,
                        "counter_search_separate_query": True,
                        "counter_search_raw_document_budget": (
                            _PUBMED_COUNTER_RAW_DOCUMENT_BUDGET
                        ),
                        "counter_search_raw_hit_count": len(raw_ids),
                        "counter_search_http_call_provenance": dict(
                            search_provenance
                        ),
                        "counter_category": content.counter_signal_category,
                        "counter_signal": content.counter_signal,
                        "classification_policy": (
                            "same-clause-direct-target-intervention-counter-"
                            "signal-proximity80-v2"
                        ),
                        "review_status": "researcher_review_required",
                        "evidence_scope": (
                            "primary-publication limitation/counter context; "
                            "not a direct target-direction refutation"
                        ),
                        "content_storage_policy": (
                            "bounded_excerpt_only_no_full_text"
                        ),
                        "full_text_stored": False,
                        "content_retrieval_endpoint": PUBMED_FETCH_URL,
                        "direct_support_eligible": False,
                        "direct_support_rejection_codes": [],
                    },
                )
            )
            if len(records) >= self.max_records_per_source:
                break
        return _PubMedCounterOutcome(
            records=tuple(records),
            status=(
                "completed_with_results" if records else "completed_empty"
            ),
            query=query,
            query_sha256=query_sha256,
            raw_hit_count=len(raw_ids),
            http_call_provenance=search_provenance,
        )

    def _pubmed_primary_association_supplement(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        *,
        query_lanes: Sequence[_PrimaryAssociationQueryLane],
        excluded_ids: set[str],
        requirement: _PrimaryAssociationRequirement,
    ) -> _PrimaryAssociationSupplementOutcome:
        """Screen two disjoint precision-first lanes under one strict budget."""

        common_params: dict[str, str] = {"tool": self.ncbi_tool}
        if self.ncbi_email:
            common_params["email"] = self.ncbi_email
        if self.ncbi_api_key:
            common_params["api_key"] = self.ncbi_api_key
        lanes = tuple(query_lanes)
        if (
            len(lanes) > 2
            or sum(lane.raw_document_budget for lane in lanes)
            > _PRIMARY_ASSOCIATION_RAW_DOCUMENT_BUDGET
            or any(
                lane.raw_document_budget <= 0
                for lane in lanes
            )
        ):
            raise ValueError("primary association query lanes exceed bounds")

        http_requests_used = 0
        raw_documents_fetched = 0
        records_screened = 0
        search_result_count = 0
        returned_id_count = 0
        excluded_main_query_id_count = 0
        excluded_prior_lane_id_count = 0
        screening_entries: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        lane_traces: list[dict[str, Any]] = []
        error_summaries: list[dict[str, str]] = []
        prior_lane_ids: set[str] = set()

        def reject(entry: dict[str, Any], *codes: str) -> None:
            normalized = tuple(dict.fromkeys(code for code in codes if code))
            entry["status"] = "rejected"
            entry["rejection_codes"] = list(normalized)
            for code in normalized:
                rejection_counts[code] = rejection_counts.get(code, 0) + 1

        def finish(
            *,
            status: str,
            stop_reason: str,
            record: EvidenceRef | None = None,
            parent_record: EvidenceRef | None = None,
        ) -> _PrimaryAssociationSupplementOutcome:
            trace: dict[str, Any] = {
                "schema_version": "2.0",
                "status": status,
                "stop_reason": stop_reason,
                "coverage_scope": (
                    "two_lane_relevance_ranked_bounded_abstract_screen_not_exhaustive"
                ),
                "query_sha256": sha256(
                    "\n".join(lane.query for lane in lanes).encode("utf-8")
                ).hexdigest(),
                "query_strategy": (
                    "brief-derived-precision-first-primary-association-v4"
                ),
                "query_budget": 2,
                "query_count_executed": len(lane_traces),
                "lane_http_request_budget": (
                    _PRIMARY_ASSOCIATION_LANE_HTTP_REQUEST_BUDGET
                ),
                "lanes": list(lane_traces),
                "required_scope": requirement.required_scope,
                "variant_clauses": [
                    clause.canonical for clause in requirement.variant_clauses
                ],
                "variant_kinds": [
                    clause.kind for clause in requirement.variant_clauses
                ],
                "trigger_fields": list(requirement.trigger_fields),
                "http_request_budget": (
                    _PRIMARY_ASSOCIATION_HTTP_REQUEST_BUDGET
                ),
                "http_requests_used": http_requests_used,
                "raw_document_budget": (
                    _PRIMARY_ASSOCIATION_RAW_DOCUMENT_BUDGET
                ),
                "raw_documents_fetched": raw_documents_fetched,
                "records_screened": records_screened,
                "search_result_count": search_result_count,
                "returned_id_count": returned_id_count,
                "excluded_main_query_id_count": (
                    excluded_main_query_id_count
                ),
                "excluded_prior_lane_id_count": excluded_prior_lane_id_count,
                "unsearched_result_count": sum(
                    int(lane.get("unsearched_result_count") or 0)
                    for lane in lane_traces
                ),
                "screening_entries": list(screening_entries),
                "rejection_counts": {
                    key: rejection_counts[key]
                    for key in sorted(rejection_counts)
                },
                "content_storage_policy": (
                    "bounded_excerpt_only_no_full_text"
                ),
                "full_text_stored": False,
            }
            if error_summaries:
                trace["lane_errors"] = list(error_summaries)
            records: tuple[EvidenceRef, ...] = ()
            if record is not None:
                if parent_record is None:
                    raise ValueError(
                        "selected association supplement requires its parent record"
                    )
                record = record.model_copy(
                    update={
                        "metadata": {
                            **record.metadata,
                            "primary_association_supplement_status": status,
                            "primary_association_supplement_stop_reason": (
                                stop_reason
                            ),
                            "primary_association_supplement_trace": trace,
                            "primary_association_supplement_http_request_budget": (
                                _PRIMARY_ASSOCIATION_HTTP_REQUEST_BUDGET
                            ),
                            "primary_association_supplement_http_requests_used": (
                                http_requests_used
                            ),
                        }
                    },
                    deep=True,
                )
                parent_record = parent_record.model_copy(
                    update={
                        "metadata": {
                            **parent_record.metadata,
                            "primary_association_supplement_status": status,
                            "primary_association_supplement_stop_reason": (
                                stop_reason
                            ),
                            "primary_association_supplement_trace": trace,
                            "primary_association_supplement": True,
                            "primary_association_supplement_parent": True,
                        }
                    },
                    deep=True,
                )
                records = (parent_record, record)
            return _PrimaryAssociationSupplementOutcome(
                records=records,
                status=status,
                stop_reason=stop_reason,
                trace=trace,
            )

        selected: EvidenceRef | None = None
        selected_parent: EvidenceRef | None = None
        for lane_index, lane in enumerate(lanes, 1):
            lane_http_requests = 0
            lane_raw_documents = 0
            lane_records_screened = 0
            lane_search_count = 0
            lane_returned_count = 0
            lane_main_excluded = 0
            lane_prior_excluded = 0
            lane_trace: dict[str, Any] = {
                "lane_index": lane_index,
                "lane_id": lane.lane_id,
                "query": lane.query,
                "query_sha256": sha256(lane.query.encode("utf-8")).hexdigest(),
                "sort": "relevance",
                "status": "pending",
                "stop_reason": "not_started",
                "http_request_budget": (
                    _PRIMARY_ASSOCIATION_LANE_HTTP_REQUEST_BUDGET
                ),
                "http_requests_used": 0,
                "raw_document_budget": lane.raw_document_budget,
                "raw_documents_fetched": 0,
                "records_screened": 0,
                "search_result_count": 0,
                "returned_id_count": 0,
                "excluded_main_query_id_count": 0,
                "excluded_prior_lane_id_count": 0,
                "unsearched_result_count": 0,
            }
            lane_traces.append(lane_trace)

            try:
                http_requests_used += 1
                lane_http_requests += 1
                search = self._ncbi_get(
                    PUBMED_SEARCH_URL,
                    params={
                        **common_params,
                        "db": "pubmed",
                        "term": lane.query,
                        "retmode": "json",
                        "retmax": lane.raw_document_budget,
                        "sort": "relevance",
                    },
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                )
                search.raise_for_status()
                search_result = search.json().get("esearchresult", {})
                raw_ids = [
                    str(value).strip()
                    for value in search_result.get("idlist", [])
                    if str(value).strip()
                ][: lane.raw_document_budget]
                lane_search_count = _safe_nonnegative_int(
                    search_result.get("count"),
                    default=len(raw_ids),
                )
            except Exception as exc:
                error_type = type(exc).__name__
                lane_trace.update(
                    {
                        "status": "error",
                        "stop_reason": f"search_{error_type}:fail_closed",
                        "http_requests_used": lane_http_requests,
                    }
                )
                error_summaries.append(
                    {
                        "lane_id": lane.lane_id,
                        "error_stage": "search",
                        "error_type": error_type,
                    }
                )
                continue

            deduplicated_ids = list(dict.fromkeys(raw_ids))
            lane_returned_count = len(deduplicated_ids)
            lane_main_excluded = sum(
                identifier in excluded_ids
                for identifier in deduplicated_ids
            )
            lane_prior_excluded = sum(
                identifier in prior_lane_ids
                for identifier in deduplicated_ids
            )
            ids = [
                identifier
                for identifier in deduplicated_ids
                if identifier not in excluded_ids
                and identifier not in prior_lane_ids
            ]
            prior_lane_ids.update(deduplicated_ids)
            search_result_count += lane_search_count
            returned_id_count += lane_returned_count
            excluded_main_query_id_count += lane_main_excluded
            excluded_prior_lane_id_count += lane_prior_excluded
            lane_trace.update(
                {
                    "search_result_count": lane_search_count,
                    "returned_id_count": lane_returned_count,
                    "excluded_main_query_id_count": lane_main_excluded,
                    "excluded_prior_lane_id_count": lane_prior_excluded,
                    "unsearched_result_count": max(
                        0,
                        lane_search_count - lane_returned_count,
                    ),
                }
            )
            if not ids:
                lane_trace.update(
                    {
                        "status": "completed_bounded_without_verified_result",
                        "stop_reason": "search_returned_no_unseen_candidates",
                        "http_requests_used": lane_http_requests,
                    }
                )
                continue

            try:
                http_requests_used += 1
                lane_http_requests += 1
                summary = self._ncbi_get(
                    PUBMED_SUMMARY_URL,
                    params={
                        **common_params,
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "json",
                    },
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                )
                summary.raise_for_status()
                result = summary.json().get("result", {})
                http_requests_used += 1
                lane_http_requests += 1
                fetch = self._ncbi_get(
                    PUBMED_FETCH_URL,
                    params={
                        **common_params,
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "xml",
                        "rettype": "abstract",
                    },
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                )
                fetch.raise_for_status()
                content_by_pmid = _parse_pubmed_content(
                    _response_text(fetch),
                    target_symbol=seed.symbol,
                    disease=brief.disease,
                    intervention_direction=(
                        brief.constraints.intervention_direction
                    ),
                )
            except Exception as exc:
                error_type = type(exc).__name__
                lane_trace.update(
                    {
                        "status": "error",
                        "stop_reason": (
                            f"summary_or_abstract_{error_type}:fail_closed"
                        ),
                        "http_requests_used": lane_http_requests,
                    }
                )
                error_summaries.append(
                    {
                        "lane_id": lane.lane_id,
                        "error_stage": "summary_or_abstract",
                        "error_type": error_type,
                    }
                )
                continue

            lane_raw_documents = len(ids)
            raw_documents_fetched += lane_raw_documents
            for lane_rank, pmid in enumerate(ids, start=1):
                records_screened += 1
                lane_records_screened += 1
                entry: dict[str, Any] = {
                    "external_id": pmid,
                    "lane_id": lane.lane_id,
                    "lane_rank": lane_rank,
                    "screening_rank": records_screened,
                }
                screening_entries.append(entry)
                item = result.get(pmid) if isinstance(result, Mapping) else None
                if not isinstance(item, Mapping):
                    reject(entry, "summary_metadata_missing")
                    continue
                publication_types = [
                    str(value) for value in item.get("pubtype", [])
                ]
                title = str(
                    item.get("title") or f"PubMed record {pmid}"
                ).strip()
                metadata_rejections = (
                    _primary_publication_metadata_rejection_codes(
                        title,
                        publication_types,
                    )
                )
                if metadata_rejections:
                    reject(entry, *metadata_rejections)
                    continue
                content = content_by_pmid.get(pmid)
                if content is not None:
                    from drugpilot.tools.evidence_gap import content_observation
                    entry["extraction_observation"] = content_observation(content)
                if content is None or not content.content_hash:
                    reject(entry, "abstract_unavailable")
                    continue
                parent = self._record(
                    source="PubMed",
                    external_id=pmid,
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    supports="context",
                    query=lane.query,
                    endpoint=PUBMED_FETCH_URL,
                    excerpt_text=content.excerpt,
                    hash_title_when_no_excerpt=False,
                    extra={
                        "pmid": pmid,
                        "pmcid": content.pmcid,
                        "doi": content.doi or _summary_article_id(item, "doi"),
                        "publication_date": item.get("pubdate"),
                        "journal": (
                            item.get("fulljournalname") or item.get("source")
                        ),
                        "publication_types": publication_types,
                        "content_availability": (
                            "abstract_excerpt"
                            if content.excerpt
                            else "metadata_only"
                        ),
                        "source_location": content.source_location,
                        "excerpt": content.excerpt,
                        "content_hash": content.content_hash,
                        **_abstract_context_metadata(content),
                        "content_hash_algorithm": "sha256",
                        "content_hash_scope": "normalized_full_abstract",
                        "full_text_stored": False,
                        "query_strategy": (
                            "brief-derived-precision-first-primary-association-v4"
                        ),
                        "primary_association_query_lane": lane.lane_id,
                        "claim_linkage_status": (
                            "pending_ma1_claim_reconciliation"
                        ),
                        "content_storage_policy": (
                            "bounded_excerpt_only_no_full_text"
                        ),
                    },
                )
                association = _association_role_record(parent, content)
                if association is None:
                    reject(entry, "primary_association_excerpt_not_found")
                    continue
                if not _association_record_satisfies_requirement(
                    association,
                    requirement,
                ):
                    reject(entry, "required_subtype_or_variant_not_matched")
                    continue
                entry["status"] = "selected"
                entry["rejection_codes"] = []
                selected = association.model_copy(
                    update={
                        "metadata": {
                            **association.metadata,
                            "additional_network_call": True,
                            "query_strategy": (
                                "brief-derived-precision-first-primary-association-v4"
                            ),
                            "primary_association_query_lane": lane.lane_id,
                            "primary_association_supplement": True,
                            "parent_provenance_pointer_type": (
                                "selected_publication_record_id_not_claim_link"
                            ),
                            "parent_record_available_for_atomic_selection": True,
                            "parent_record_in_packet": False,
                        }
                    },
                    deep=True,
                )
                selected_parent = parent
                break

            lane_trace.update(
                {
                    "http_requests_used": lane_http_requests,
                    "raw_documents_fetched": lane_raw_documents,
                    "records_screened": lane_records_screened,
                }
            )
            if selected is not None:
                lane_trace.update(
                    {
                        "status": "completed_with_results",
                        "stop_reason": (
                            "first_verified_required_association_selected"
                        ),
                    }
                )
                break

            lane_trace.update(
                {
                    "status": "completed_bounded_without_verified_result",
                    "stop_reason": (
                        "raw_document_budget_exhausted_without_required_association"
                        if lane_search_count > lane_returned_count
                        else "returned_candidates_exhausted_without_required_association"
                    ),
                }
            )

        if selected is not None:
            return finish(
                status="completed_with_results",
                stop_reason="first_verified_required_association_selected",
                record=selected,
                parent_record=selected_parent,
            )
        if error_summaries:
            return finish(
                status="error",
                stop_reason="lane_error_without_verified_result:fail_closed",
            )
        coverage_exhausted = any(
            int(lane.get("unsearched_result_count") or 0) > 0
            for lane in lane_traces
        )
        return finish(
            status="completed_bounded_without_verified_result",
            stop_reason=(
                "raw_document_budget_exhausted_without_required_association"
                if coverage_exhausted
                else "returned_candidates_exhausted_without_required_association"
            ),
        )

    def _europe_pmc(self, brief: ResearchBrief, seed: TargetSeed) -> list[EvidenceRef]:
        query = self._europe_pmc_query(brief, seed)
        requirement = _primary_association_requirement(brief, seed.symbol)
        response = self.client.get(
            EUROPE_PMC_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "pageSize": self.max_records_per_source,
                "resultType": "core",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        evidence: list[EvidenceRef] = []
        association_candidates: list[tuple[int, EvidenceRef]] = []
        for item in results[: self.max_records_per_source]:
            if not isinstance(item, Mapping):
                continue
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            external_id = pmid or pmcid or item.get("id")
            if not external_id:
                continue
            title = str(item.get("title") or f"Europe PMC record {external_id}").strip()
            publication_types = _europe_pmc_publication_types(item)
            abstract = _plain_text(str(item.get("abstractText") or ""))
            content = _publication_content_from_sections(
                (("abstract", abstract),) if abstract else (),
                target_symbol=seed.symbol,
                disease=brief.disease,
                intervention_direction=brief.constraints.intervention_direction,
                doi=_clean_optional_text(item.get("doi")),
                pmcid=_clean_optional_text(pmcid),
            )
            title_target_match = bool(
                re.search(_target_token_pattern(seed.symbol), title)
            )
            if (
                content.content_hash
                and content.target_lexical_match is False
                and not title_target_match
            ):
                continue
            direction_rejection_codes = (
                _primary_direction_content_rejection_codes(
                    content,
                    title=title,
                    publication_types=publication_types,
                    target_symbol=seed.symbol,
                    disease=brief.disease,
                    intervention_direction=(
                        brief.constraints.intervention_direction
                    ),
                )
            )
            verified_main_direction = not direction_rejection_codes
            article_path = (
                f"MED/{pmid}" if pmid else f"PMC/{pmcid}" if pmcid else str(external_id)
            )
            parent_record = self._record(
                    source="Europe PMC",
                    external_id=str(external_id),
                    title=title,
                    url=f"https://europepmc.org/article/{article_path}",
                    supports=_publication_evidence_role(
                        content,
                        counter_evidence_enabled=self.counter_evidence_enabled,
                        direct_support_eligible=verified_main_direction,
                    ),
                    query=query,
                    endpoint=EUROPE_PMC_SEARCH_URL,
                    excerpt_text=content.excerpt,
                    hash_title_when_no_excerpt=False,
                    extra={
                        "pmid": _clean_optional_text(pmid),
                        "pmcid": _clean_optional_text(pmcid),
                        "doi": content.doi,
                        "publication_year": item.get("pubYear"),
                        "journal": item.get("journalTitle"),
                        "publication_types": publication_types,
                        "cited_by_count": item.get("citedByCount"),
                        "is_open_access": item.get("isOpenAccess"),
                        "content_availability": (
                            "abstract_excerpt" if content.excerpt else "metadata_only"
                        ),
                        "abstract_retrieval_status": (
                            "available" if content.content_hash else "not_available"
                        ),
                        "source_location": content.source_location,
                        "excerpt": content.excerpt,
                        "content_hash": content.content_hash,
                        **_abstract_context_metadata(content, source_review=self.ma3_source_review_enabled),
                        "content_hash_algorithm": (
                            "sha256" if content.content_hash else None
                        ),
                        "content_hash_scope": (
                            "normalized_full_abstract" if content.content_hash else None
                        ),
                        "excerpt_char_count": (
                            len(content.excerpt) if content.excerpt else 0
                        ),
                        "excerpt_max_chars": _MAX_PUBLICATION_EXCERPT_CHARS,
                        "excerpt_truncated": content.excerpt_truncated,
                        "excerpt_selection_status": content.excerpt_selection_status,
                        "excerpt_selection_policy": (
                            "same-sentence-target-disease-direction-aware-mechanistic-ranking-v2"
                        ),
                        "excerpt_relevance_basis": [
                            "target_symbol",
                            "disease",
                            "intervention_direction",
                            "mechanistic_endpoint",
                            "experimental_result_language",
                        ],
                        "query_strategy": "target-disease-direction-mechanistic-v2",
                        "query_intervention_direction": (
                            brief.constraints.intervention_direction
                        ),
                        "claim_linkage_status": "pending_ma1_claim_reconciliation",
                        "counter_category": content.counter_signal_category,
                        "counter_signal": content.counter_signal,
                        "classification_policy": (
                            "same-clause-target-counter-signal-proximity80-v1"
                            if content.counter_signal_category
                            else "conservative-direct-intervention-subject-v1"
                            if content.direct_support_rejection_codes
                            else "association-role-only-v1"
                            if _publication_has_association_candidate(content)
                            else _VERIFIED_DIRECT_PRIMARY_POLICY
                            if verified_main_direction
                            else "primary-direction-validation-failed-v1"
                        ),
                        "direct_support_eligible": verified_main_direction,
                        "direct_support_rejection_codes": list(
                            direction_rejection_codes
                        ),
                        "review_status": (
                            "researcher_review_required"
                            if direction_rejection_codes
                            else None
                        ),
                        "content_storage_policy": "bounded_excerpt_only_no_full_text",
                        "full_text_stored": False,
                        "content_retrieval_endpoint": EUROPE_PMC_SEARCH_URL,
                        "evidence_scope": _publication_evidence_scope(
                            content,
                            direct_support_eligible=verified_main_direction,
                        ),
                    },
                )
            association_record = _association_role_record(
                parent_record,
                content,
            )
            if association_record is not None:
                candidate_rejections = (
                    _primary_association_candidate_rejection_codes(
                        association_record,
                        requirement=requirement,
                        title=title,
                        publication_types=publication_types,
                    )
                )
                parent_record = parent_record.model_copy(
                    update={
                        "metadata": {
                            **parent_record.metadata,
                            "primary_association_candidate_status": (
                                "rejected"
                                if candidate_rejections
                                else "accepted"
                            ),
                            "primary_association_candidate_rejection_codes": list(
                                candidate_rejections
                            ),
                            "primary_association_candidate_retained": not bool(
                                candidate_rejections
                            ),
                        }
                    },
                    deep=True,
                )
                if not candidate_rejections:
                    association_record = _association_role_record(
                        parent_record,
                        content,
                    )
                    assert association_record is not None
                    association_candidates.append(
                        (content.association_score, association_record)
                    )
            parent_record = parent_record.model_copy(
                update={
                    "metadata": {
                        **parent_record.metadata,
                        "verified_direct_primary": verified_main_direction,
                        "verified_direct_primary_policy": (
                            _VERIFIED_DIRECT_PRIMARY_POLICY
                        ),
                        "verified_direct_primary_rejection_codes": list(
                            direction_rejection_codes
                        ),
                    }
                },
                deep=True,
            )
            evidence.append(parent_record)
        evidence.extend(record for _, record in association_candidates)
        epmc_oa_trace: dict[str, Any] | None = None
        if self.epmc_oa_fulltext_one_hop_v1:
            try:
                supplemental, epmc_oa_trace = (
                    self._europe_pmc_oa_fulltext_one_hop(
                        brief,
                        seed,
                        query=query,
                        search_results=results,
                        abstract_records=evidence,
                        requirement=requirement,
                    )
                )
            except Exception as exc:
                # The optional lane must never make an already successful
                # abstract retrieval fail.  Record only the exception type;
                # messages may contain response fragments or local details.
                supplemental = []
                epmc_oa_trace = self._epmc_oa_contained_error_trace(
                    brief,
                    abstract_records=evidence,
                    requirement=requirement,
                    error_type=type(exc).__name__,
                )
            evidence.extend(supplemental)
        if self.counter_evidence_enabled:
            counter_query = self._europe_pmc_counter_query(brief, seed)
            counter_query_sha256 = sha256(
                counter_query.encode("utf-8")
            ).hexdigest()
            try:
                counter_records = self._europe_pmc_counter_records(
                    brief,
                    seed,
                    counter_query,
                    excluded_ids={item.external_id for item in evidence},
                )
                counter_status = (
                    "completed_with_results"
                    if counter_records
                    or any(item.supports == "counter" for item in evidence)
                    else "completed_empty"
                )
                evidence = [
                    item.model_copy(
                        update={
                            "metadata": {
                                **item.metadata,
                                "counter_search_status": counter_status,
                                "counter_search_query": counter_query,
                                "counter_search_query_sha256": (
                                    counter_query_sha256
                                ),
                                "counter_search_source": "Europe PMC",
                                "counter_search_separate_query": True,
                                "counter_search_raw_hit_count": len(
                                    counter_records
                                ),
                            }
                        },
                        deep=True,
                    )
                    for item in evidence
                ]
                evidence.extend(counter_records)
            except Exception as exc:
                evidence = [
                    item.model_copy(
                        update={
                            "metadata": {
                                **item.metadata,
                                "counter_search_status": "error",
                                "counter_search_query": counter_query,
                                "counter_search_query_sha256": (
                                    counter_query_sha256
                                ),
                                "counter_search_source": "Europe PMC",
                                "counter_search_separate_query": True,
                                "counter_search_error_type": type(exc).__name__,
                            }
                        },
                        deep=True,
                    )
                    for item in evidence
                ]
        if epmc_oa_trace is not None:
            evidence = [
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "epmc_oa_fulltext_one_hop_trace": deepcopy(
                                epmc_oa_trace
                            ),
                        }
                    },
                    deep=True,
                )
                for item in evidence
            ]
            with self._cache_lock:
                self._last_epmc_oa_fulltext_trace = deepcopy(epmc_oa_trace)
        return _select_publication_records_with_role_budget(
            evidence,
            max_records=self.max_records_per_source,
        )

    def _epmc_oa_contained_error_trace(
        self,
        brief: ResearchBrief,
        *,
        abstract_records: Sequence[EvidenceRef],
        requirement: _PrimaryAssociationRequirement,
        error_type: str,
    ) -> dict[str, Any]:
        needs_direct, needs_association, needs_assay_trace = (
            self._epmc_oa_evidence_needs(
            abstract_records,
            brief=brief,
            requirement=requirement,
            )
        )
        trigger = {
            "trigger_scope": "europe_pmc_source_local_abstract_pass",
            "global_cross_source_sufficiency_not_evaluated": True,
            "research_depth": brief.research_depth,
            "abstract_verified_direct_primary_available": not needs_direct,
            "required_association_requested": requirement.requested,
            "abstract_required_association_available": (
                not needs_association if requirement.requested else None
            ),
            "required_assay_trace_requested": (
                self._epmc_oa_assay_trace_required(brief)
            ),
            "abstract_required_assay_trace_available": (
                not needs_assay_trace
                if self._epmc_oa_assay_trace_required(brief)
                else None
            ),
            "abstract_evidence_insufficient": bool(
                needs_direct or needs_association or needs_assay_trace
            ),
            "needed_roles": [
                role
                for needed, role in (
                    (needs_direct, "verified_direct_primary"),
                    (needs_association, "required_primary_association"),
                    (needs_assay_trace, "required_assay_trace"),
                )
                if needed
            ],
        }
        trace = self._epmc_oa_trace(
            None,
            trigger=trigger,
            status="completed_with_errors_without_evidence",
            stop_reason="unexpected_optional_lane_error_contained",
            result_records=(),
        )
        trace["error_reason_codes"] = [f"outer_{error_type}"]
        trace["outer_error_contained"] = True
        trace.pop("trace_sha256", None)
        trace["trace_sha256"] = sha256(
            json.dumps(
                trace,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return trace

    def _europe_pmc_oa_fulltext_one_hop(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        *,
        query: str,
        search_results: Sequence[Any],
        abstract_records: Sequence[EvidenceRef],
        requirement: _PrimaryAssociationRequirement,
        preferred_pmcid_order: Sequence[str] = (),
    ) -> tuple[list[EvidenceRef], dict[str, Any]]:
        """Escalate to bounded OA JATS and backward references only.

        The normal Europe PMC search result remains authoritative until it is
        shown to lack either direct target-direction evidence or a
        Brief-required association excerpt.  This optional lane never follows
        forward citations, publisher links, PDF links, or more than one
        reference edge.  Every error is contained so already retrieved
        abstract evidence is preserved without a fixture substitution.
        """

        from drugpilot.tools.europe_pmc_oa_fulltext import (
            EuropePMCOAValidationError,
            document_identity_keys,
            exclude_visited_references,
            parse_europe_pmc_references,
            parse_oa_jats,
        )

        depth = brief.research_depth
        needs_direct, needs_association, needs_assay_trace = (
            self._epmc_oa_evidence_needs(
            abstract_records,
            brief=brief,
            requirement=requirement,
            )
        )
        source_local_needs = (needs_direct, needs_association)
        requested_roles = set(seed.discovery_metadata.get("g0_followup_roles", ())) & {"direction", "association"}
        needs_direct = needs_direct or "direction" in requested_roles
        needs_association = needs_association or "association" in requested_roles
        trigger = {
            "g0_requested_roles": sorted(requested_roles),
            "trigger_scope": "europe_pmc_source_local_abstract_pass",
            "global_cross_source_sufficiency_not_evaluated": True,
            "abstract_verified_direct_primary_available": not source_local_needs[0],
            "required_association_requested": requirement.requested,
            "abstract_required_association_available": (
                not source_local_needs[1] if requirement.requested else None
            ),
            "required_assay_trace_requested": (
                self._epmc_oa_assay_trace_required(brief)
            ),
            "abstract_required_assay_trace_available": (
                not needs_assay_trace
                if self._epmc_oa_assay_trace_required(brief)
                else None
            ),
            "abstract_evidence_insufficient": bool(
                any(source_local_needs) or needs_assay_trace
            ),
            "needed_roles": [
                role
                for needed, role in (
                    (needs_direct, "verified_direct_primary"),
                    (needs_association, "required_primary_association"),
                    (needs_assay_trace, "required_assay_trace"),
                )
                if needed
            ],
        }
        if not (needs_direct or needs_association or needs_assay_trace):
            return [], self._epmc_oa_trace(
                None,
                trigger=trigger,
                status="not_needed",
                stop_reason="abstract_required_evidence_sufficient",
                result_records=(),
            )

        state = _EuropePMCOAEscalationState(started_at=self._clock())
        supplemental: list[EvidenceRef] = []
        reference_parents: list[tuple[str, str | None, str]] = []
        # Only successfully identity/license-validated parent JATS may supply
        # embedded reference edges. These are discovery IDs, never support.
        parent_reference_documents: dict[str, Any] = {}
        visited_keys: set[str] = set()
        ordered_search_results = _ordered_europe_pmc_parent_results(
            search_results,
            preferred_pmcid_order=preferred_pmcid_order,
        )
        for raw_item in ordered_search_results:
            if not isinstance(raw_item, Mapping):
                continue
            visited_keys.update(
                document_identity_keys(
                    pmcid=_clean_optional_text(raw_item.get("pmcid")),
                    pmid=_clean_optional_text(raw_item.get("pmid")),
                    doi=_clean_optional_text(raw_item.get("doi")),
                )
            )

        parent_candidates: list[tuple[str, Mapping[str, Any], Any]] = []
        for raw_item in ordered_search_results:
            if not isinstance(raw_item, Mapping):
                continue
            eligibility, rejection = _europe_pmc_oa_eligibility(raw_item)
            if rejection is None:
                assert eligibility is not None
                parent_candidates.append(("primary", raw_item, eligibility))
            else:
                review_parent, review_rejection = (
                    _europe_pmc_review_reference_parent(raw_item)
                )
                if review_parent is not None:
                    parent_candidates.append(
                        ("review", raw_item, review_parent)
                    )
                    state.discovery_parents.append(
                        {
                            **review_parent,
                            "discovery_parent_publication_type": "review",
                            "discovery_only": True,
                            "inherited_role": False,
                            "evidence_role_inherited": False,
                            "support_emitted": False,
                            "fulltext_requested": False,
                        }
                    )
                else:
                    reason = (
                        review_rejection
                        if review_rejection
                        != "not_review_discovery_parent"
                        else rejection
                    )
                    _increment_reason(
                        state.eligibility_rejections,
                        str(reason or rejection),
                    )
            if len(parent_candidates) >= _EPMC_OA_PARENT_LIMIT:
                break

        for parent_kind, item, parent_identity in parent_candidates:
            if not self._epmc_oa_roles_still_needed(
                brief,
                abstract_records,
                supplemental,
                requirement=requirement,
                requested_roles=requested_roles,
            ):
                state.stop_reason = "parent_fulltext_satisfied_missing_roles"
                break
            if parent_kind == "review":
                reference_parents.append(
                    (
                        str(parent_identity["pmcid"]),
                        str(parent_identity["pmid"]),
                        "review",
                    )
                )
                continue
            eligibility = parent_identity
            endpoint = _europe_pmc_fulltext_url(eligibility.pmcid)
            state.parent_fulltexts_attempted += 1
            try:
                _response, raw_xml = self._epmc_oa_get(
                    state,
                    endpoint,
                    params={},
                    role="parent_fulltext",
                    document_body=True,
                )
                parsed = parse_oa_jats(
                    raw_xml,
                    eligibility=eligibility,
                    allowed_license_uris=_EPMC_OA_ALLOWED_LICENSE_URIS,
                    max_spans=64,
                )
            except _EuropePMCOABudgetStop as exc:
                state.stop_reason = exc.reason_code
                break
            except EuropePMCOAValidationError as exc:
                state.error_reason_codes.append(exc.reason_code)
                continue
            except Exception as exc:
                state.error_reason_codes.append(
                    f"parent_fulltext_{type(exc).__name__}"
                )
                continue
            state.parent_fulltexts_parsed += 1
            state.external_jats_doctypes_removed += int(
                parsed.audit.external_doctype_removed
            )
            reference_parents.append(
                (eligibility.pmcid, eligibility.pmid, "primary")
            )
            parent_reference_documents[eligibility.pmcid] = parsed
            state.discovery_parents.append(
                {
                    "pmcid": eligibility.pmcid,
                    "pmid": eligibility.pmid,
                    "discovery_parent_publication_type": "primary",
                    "discovery_only": False,
                    "inherited_role": False,
                    "evidence_role_inherited": False,
                    "fulltext_requested": True,
                }
            )
            supplemental.extend(
                self._epmc_oa_article_records(
                    brief,
                    seed,
                    item=item,
                    parsed=parsed,
                    query=query,
                    fulltext_endpoint=endpoint,
                    lineage=None,
                    abstract_records=abstract_records,
                )
            )

        if self._epmc_oa_roles_still_needed(
            brief,
            abstract_records,
            supplemental,
            requirement=requirement,
            requested_roles=requested_roles,
        ) and state.stop_reason is None:
            references: list[Any] = []
            for parent_pmcid, parent_pmid, parent_kind in reference_parents:
                remaining = _EPMC_OA_REFERENCE_LIMIT - len(references)
                if remaining <= 0:
                    state.stop_reason = "reference_limit_reached"
                    break
                parent_document = parent_reference_documents.get(parent_pmcid)
                if parent_document is not None and parent_document.references:
                    # The already downloaded XML contains the actual cited
                    # identifiers. Avoid an unnecessary /references dependency;
                    # child identity, OA license and primary evidence checks
                    # below remain exactly the same as for API-discovered IDs.
                    eligible_references = exclude_visited_references(
                        parent_document.references, visited_keys
                    )
                    selected_references = eligible_references[:remaining]
                    state.reference_audits.append({
                        "reference_source": "validated_parent_jats",
                        "reference_policy": "parent_jats_identifiers_one_hop_v1",
                        "parent_pmcid": parent_pmcid,
                        "parent_pmid": parent_pmid,
                        "parent_document_sha256": parent_document.audit.document_sha256,
                        "parent_license_uri": parent_document.audit.license_uri,
                        "parsed_reference_count": len(parent_document.references),
                        "emitted_reference_count": len(selected_references),
                        "references_truncated_by_limit": len(eligible_references) > remaining,
                        "citation_lineage_sha256s": [r.citation_lineage_sha256 for r in selected_references],
                        "discovery_parent_publication_type": parent_kind,
                        "inherited_role": False,
                        "evidence_role_inherited": False,
                        "raw_full_text_retained": False,
                    })
                    references.extend(selected_references)
                    continue
                endpoint = _europe_pmc_references_url(
                    pmcid=parent_pmcid,
                    pmid=parent_pmid,
                )
                params = {
                    "format": "json",
                    "page": 1,
                    "pageSize": remaining,
                }
                try:
                    _response, raw_references = self._epmc_oa_get(
                        state,
                        endpoint,
                        params=params,
                        role="parent_references",
                        document_body=False,
                    )
                    parsed_references = parse_europe_pmc_references(
                        raw_references,
                        parent_pmcid=parent_pmcid,
                        parent_pmid=parent_pmid,
                        max_references=remaining,
                        visited_keys=visited_keys,
                    )
                except _EuropePMCOABudgetStop as exc:
                    state.stop_reason = exc.reason_code
                    break
                except EuropePMCOAValidationError as exc:
                    state.error_reason_codes.append(exc.reason_code)
                    continue
                except Exception as exc:
                    state.error_reason_codes.append(
                        f"parent_references_{type(exc).__name__}"
                    )
                    continue
                state.reference_audits.append(
                    {
                        **parsed_references.to_audit_metadata(),
                        "discovery_parent_publication_type": parent_kind,
                        "inherited_role": False,
                        "evidence_role_inherited": False,
                    }
                )
                references.extend(parsed_references.references)

            references, cross_parent_drops = (
                _deduplicate_references_preserving_first_lineage(references)
            )
            state.cross_parent_duplicate_references_dropped += (
                cross_parent_drops
            )
            references = list(
                exclude_visited_references(references, visited_keys)
            )[:_EPMC_OA_REFERENCE_LIMIT]
            state.references_retained = len(references)

            for reference in references:
                if not self._epmc_oa_roles_still_needed(
                    brief,
                    abstract_records,
                    supplemental,
                    requirement=requirement,
                    requested_roles=requested_roles,
                ):
                    state.stop_reason = "child_fulltext_satisfied_missing_roles"
                    break
                if (
                    state.child_fulltexts_attempted
                    >= _EPMC_OA_CHILD_FULLTEXT_LIMIT
                ):
                    state.stop_reason = "child_fulltext_limit_reached"
                    break
                child_query = _europe_pmc_reference_metadata_query(reference)
                if child_query is None:
                    _increment_reason(
                        state.eligibility_rejections,
                        "reference_identifier_not_searchable",
                    )
                    continue
                metadata_params = {
                    "query": child_query,
                    "format": "json",
                    "pageSize": 2,
                    "resultType": "core",
                }
                state.child_metadata_checked += 1
                try:
                    _response, raw_metadata = self._epmc_oa_get(
                        state,
                        EUROPE_PMC_SEARCH_URL,
                        params=metadata_params,
                        role="child_metadata",
                        document_body=False,
                    )
                    child_item = _exact_europe_pmc_child_metadata(
                        raw_metadata,
                        reference,
                    )
                    child_eligibility, rejection = (
                        _europe_pmc_oa_eligibility(child_item)
                    )
                    if rejection is not None:
                        _increment_reason(
                            state.eligibility_rejections,
                            f"child_{rejection}",
                        )
                        continue
                    assert child_eligibility is not None
                except _EuropePMCOABudgetStop as exc:
                    state.stop_reason = exc.reason_code
                    break
                except EuropePMCOAValidationError as exc:
                    state.error_reason_codes.append(exc.reason_code)
                    continue
                except Exception as exc:
                    state.error_reason_codes.append(
                        f"child_metadata_{type(exc).__name__}"
                    )
                    continue

                child_endpoint = _europe_pmc_fulltext_url(
                    child_eligibility.pmcid
                )
                state.child_fulltexts_attempted += 1
                try:
                    _response, raw_xml = self._epmc_oa_get(
                        state,
                        child_endpoint,
                        params={},
                        role="child_fulltext",
                        document_body=True,
                    )
                    parsed_child = parse_oa_jats(
                        raw_xml,
                        eligibility=child_eligibility,
                        allowed_license_uris=_EPMC_OA_ALLOWED_LICENSE_URIS,
                        max_spans=64,
                    )
                except _EuropePMCOABudgetStop as exc:
                    state.stop_reason = exc.reason_code
                    break
                except EuropePMCOAValidationError as exc:
                    state.error_reason_codes.append(exc.reason_code)
                    continue
                except Exception as exc:
                    state.error_reason_codes.append(
                        f"child_fulltext_{type(exc).__name__}"
                    )
                    continue
                state.child_fulltexts_parsed += 1
                state.external_jats_doctypes_removed += int(
                    parsed_child.audit.external_doctype_removed
                )
                supplemental.extend(
                    self._epmc_oa_article_records(
                        brief,
                        seed,
                        item=child_item,
                        parsed=parsed_child,
                        query=child_query,
                        fulltext_endpoint=child_endpoint,
                        lineage=reference,
                        abstract_records=abstract_records,
                    )
                )

        roles_still_needed = self._epmc_oa_roles_still_needed(
            brief,
            abstract_records,
            supplemental,
            requirement=requirement,
            requested_roles=requested_roles,
        )
        if state.stop_reason is None:
            state.stop_reason = (
                "bounded_sources_exhausted_without_all_missing_roles"
                if roles_still_needed
                else "fulltext_or_reference_evidence_satisfied_missing_roles"
            )
        status = (
            "completed_with_evidence"
            if supplemental and not roles_still_needed
            else "completed_partial_evidence"
            if supplemental
            else "completed_bounded_without_evidence"
        )
        if state.stop_reason.startswith("skipped_"):
            status = "skipped"
        elif state.error_reason_codes and not supplemental:
            status = "completed_with_errors_without_evidence"
        return supplemental, self._epmc_oa_trace(
            state,
            trigger=trigger,
            status=status,
            stop_reason=state.stop_reason,
            result_records=supplemental,
        )

    @staticmethod
    def _epmc_oa_evidence_needs(
        records: Sequence[EvidenceRef],
        *,
        brief: ResearchBrief,
        requirement: _PrimaryAssociationRequirement,
    ) -> tuple[bool, bool, bool]:
        direct_available = any(
            item.metadata.get("verified_direct_primary") is True
            for item in records
        )
        association_available = any(
            _association_record_satisfies_requirement(item, requirement)
            for item in records
        )
        assay_trace_required = (
            PublicBiologyRetriever._epmc_oa_assay_trace_required(brief)
        )
        assay_trace_available = any(
            PublicBiologyRetriever._epmc_oa_record_has_assay_trace(
                item,
                required_assays=brief.constraints.required_assays,
            )
            for item in records
        )
        return (
            not direct_available,
            bool(requirement.requested and not association_available),
            bool(assay_trace_required and not assay_trace_available),
        )

    @staticmethod
    def _epmc_oa_assay_trace_required(brief: ResearchBrief) -> bool:
        text = " ".join(
            (
                brief.research_goal,
                brief.rationale or "",
                *brief.human_priorities,
            )
        ).casefold()
        assay_requested = bool(
            re.search(r"\b(?:assay|ic50|ki|kd|ec50)\b|실험|측정", text)
        )
        trace_requested = bool(
            re.search(
                r"\b(?:provenance|traceability|traceable|exact source)\b|"
                r"출처|근거|추적|원문",
                text,
            )
        )
        return bool(
            brief.constraints.required_assays
            and assay_requested
            and trace_requested
        )

    @staticmethod
    def _epmc_oa_record_has_assay_trace(
        record: EvidenceRef,
        *,
        required_assays: Sequence[str],
    ) -> bool:
        if record.metadata.get("verified_direct_primary") is not True:
            return False
        text = " ".join(
            (
                record.title,
                str(record.metadata.get("excerpt") or ""),
            )
        )
        endpoints = [
            str(value).strip()
            for value in required_assays
            if str(value).strip()
        ]
        has_endpoint = any(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(endpoint)}(?![A-Za-z0-9])",
                text,
                re.I,
            )
            for endpoint in endpoints
        )
        has_value_unit = bool(
            re.search(
                r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*"
                r"(?:pM|nM|uM|µM|μM|mM)(?![A-Za-z0-9])",
                text,
                re.I,
            )
        )
        has_experimental_context = bool(
            re.search(
                r"\b(?:assay|measured|measurement|incubat(?:ed|ion)|"
                r"binding|inhibition|protein|cells?|substrate|control)\b",
                text,
                re.I,
            )
        )
        return has_endpoint and has_value_unit and has_experimental_context

    def _epmc_oa_roles_still_needed(
        self,
        brief: ResearchBrief,
        abstract_records: Sequence[EvidenceRef],
        supplemental_records: Sequence[EvidenceRef],
        *,
        requirement: _PrimaryAssociationRequirement,
        requested_roles: Sequence[str] = (),
    ) -> bool:
        direct, association, assay_trace = self._epmc_oa_evidence_needs(
            (*abstract_records, *supplemental_records),
            brief=brief,
            requirement=requirement,
        )
        if requested_roles:
            # A cause-bound G0 request cannot be satisfied by the same abstract
            # that prompted re-examination. Inspect newly retrieved OA records
            # without mutating the original abstract's classifications.
            new_direct, new_association, _ = self._epmc_oa_evidence_needs(
                supplemental_records, brief=brief, requirement=requirement)
            direct = direct or ("direction" in requested_roles and new_direct)
            association = association or ("association" in requested_roles and new_association)
        return direct or association or assay_trace

    def _epmc_oa_get(
        self,
        state: _EuropePMCOAEscalationState,
        endpoint: str,
        *,
        params: Mapping[str, Any],
        role: str,
        document_body: bool,
    ) -> tuple[Any, bytes]:
        """Perform one retry-free official Europe PMC GET under hard caps."""

        if not (
            endpoint == EUROPE_PMC_SEARCH_URL
            or endpoint.startswith(f"{EUROPE_PMC_REST_URL}/")
        ):
            raise _EuropePMCOABudgetStop("skipped_non_official_endpoint")
        if len(state.http_calls) >= _EPMC_OA_HTTP_CALL_LIMIT:
            raise _EuropePMCOABudgetStop("skipped_http_call_budget")
        if (
            role == "parent_references"
            and len(state.http_calls)
            >= _EPMC_OA_HTTP_CALL_LIMIT - _EPMC_OA_CHILD_HTTP_CALL_RESERVE
        ):
            raise _EuropePMCOABudgetStop(
                "reference_phase_budget_reserved_for_child"
            )
        elapsed = self._clock() - state.started_at
        remaining = _EPMC_OA_ELAPSED_SECONDS - elapsed
        if remaining <= 0:
            raise _EuropePMCOABudgetStop("skipped_elapsed_budget")
        aggregate_remaining = (
            _EPMC_OA_AGGREGATE_BYTE_LIMIT - state.response_bytes_total
        )
        if aggregate_remaining <= 0:
            raise _EuropePMCOABudgetStop("skipped_aggregate_byte_budget")
        accepted_body_limit = aggregate_remaining
        overflow_reason = "skipped_aggregate_byte_budget"
        if document_body and _EPMC_OA_DOCUMENT_BYTE_LIMIT <= accepted_body_limit:
            accepted_body_limit = _EPMC_OA_DOCUMENT_BYTE_LIMIT
            overflow_reason = "skipped_document_byte_budget"
        clean_params = {str(key): value for key, value in params.items()}
        parameter_payload = json.dumps(
            {"params": clean_params},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        ledger: dict[str, Any] = {
            "role": role,
            "method": "GET",
            "endpoint": endpoint,
            "parameter_keys": sorted(
                f"params.{key}" for key in clean_params
            ),
            "parameter_sha256": sha256(parameter_payload).hexdigest(),
            "attempt_count": 1,
            "wire_attempt_count": 1,
            "retry_count": 0,
            "retries_allowed": False,
            "fixture": False,
            "raw_response_stored": False,
            "full_text_stored": False,
            "accepted_body_limit_bytes": accepted_body_limit,
            "document_acceptance_limit_bytes": (
                _EPMC_OA_DOCUMENT_BYTE_LIMIT if document_body else None
            ),
            "aggregate_acceptance_limit_bytes": (
                _EPMC_OA_AGGREGATE_BYTE_LIMIT
            ),
            # A chunked response needs one sentinel byte to prove that the
            # acceptance limit was exceeded.  That byte is hashed for the
            # ledger and discarded with the bounded prefix; it is never parsed
            # or stored as evidence.
            "stream_overflow_sentinel_bytes": 1,
        }
        if role == "parent_references":
            ledger.update(
                {
                    "endpoint_contract": (
                        "official_europe_pmc_rest_source_id_references"
                    ),
                    "endpoint_contract_template": "/{source}/{id}/references",
                    "endpoint_source": (
                        "PMC" if "/PMC/" in endpoint else "MED"
                    ),
                    "source_and_accession_prefixes_are_distinct": True,
                }
            )
        advertised_attempts = getattr(self.client, "max_attempts", 1)
        if (
            not isinstance(self.client, ScientificHttpClient)
            and isinstance(advertised_attempts, int)
            and not isinstance(advertised_attempts, bool)
            and advertised_attempts != 1
        ):
            ledger.update(
                {
                    "status": "skipped",
                    "attempt_count": 0,
                    "wire_attempt_count": 0,
                    "error_type": "unsafe_retrying_injected_transport",
                    "transport_mode": "injected_transport_rejected",
                    "transport_body_cap_enforced": False,
                }
            )
            state.http_calls.append(ledger)
            raise _EuropePMCOABudgetStop(
                "skipped_retrying_injected_transport"
            )
        try:
            wire = self._epmc_oa_wire_get_once(
                endpoint,
                params=clean_params,
                timeout=min(self.timeout_seconds, remaining),
                accepted_body_limit=accepted_body_limit,
                overflow_reason=overflow_reason,
                deadline_at=state.started_at + _EPMC_OA_ELAPSED_SECONDS,
            )
        except _EuropePMCOAWireReadError as exc:
            raw = exc.body_prefix
            state.response_bytes_total += len(raw)
            ledger.update(
                {
                    "status": "error",
                    "error_type": type(exc.cause).__name__,
                    "status_code": exc.status_code,
                    "response_bytes": len(raw),
                    "response_bytes_read": len(raw),
                    "response_body_complete": False,
                    "response_prefix_sha256": sha256(raw).hexdigest(),
                    "declared_content_length": exc.declared_content_length,
                    "transport_mode": exc.transport_mode,
                    "transport_body_cap_enforced": (
                        exc.transport_body_cap_enforced
                    ),
                }
            )
            state.http_calls.append(ledger)
            raise exc.cause
        except Exception as exc:
            ledger.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "response_bytes": 0,
                    "response_bytes_read": 0,
                    "response_body_complete": False,
                }
            )
            state.http_calls.append(ledger)
            raise
        response = wire.response
        raw = wire.body
        status_code = int(getattr(response, "status_code", 0) or 0)
        ledger.update(
            {
                "status": (
                    "skipped_elapsed_budget"
                    if wire.overflow_reason == "skipped_elapsed_budget"
                    else "skipped_body_cap"
                    if not wire.response_body_complete
                    else "completed"
                    if 200 <= status_code < 300
                    else "http_error"
                ),
                "status_code": status_code,
                "response_bytes": len(raw),
                "response_bytes_read": len(raw),
                "response_body_complete": wire.response_body_complete,
                (
                    "response_sha256"
                    if wire.response_body_complete
                    else "response_prefix_sha256"
                ): sha256(raw).hexdigest(),
                "declared_content_length": wire.declared_content_length,
                "transport_mode": wire.transport_mode,
                "transport_body_cap_enforced": (
                    wire.transport_body_cap_enforced
                ),
            }
        )
        state.http_calls.append(ledger)
        state.response_bytes_total += len(raw)
        if not wire.response_body_complete:
            assert wire.overflow_reason is not None
            raise _EuropePMCOABudgetStop(wire.overflow_reason)
        if self._clock() - state.started_at > _EPMC_OA_ELAPSED_SECONDS:
            raise _EuropePMCOABudgetStop("skipped_elapsed_budget")
        if not 200 <= status_code < 300:
            response.raise_for_status()
            raise RuntimeError("Europe PMC returned a non-success status")
        return response, raw

    def _epmc_oa_wire_get_once(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any],
        timeout: float,
        accepted_body_limit: int,
        overflow_reason: str,
        deadline_at: float,
    ) -> _EuropePMCOAWireResult:
        """Issue exactly one wire request and bound the live response stream.

        The process-wide :class:`ScientificHttpClient` normally retries safe
        GETs.  The OA supplemental lane deliberately bypasses that wrapper and
        uses its already configured, pooled ``httpx.Client`` once.  This keeps
        the same TLS/proxy/session configuration while making the twelve-call
        ledger a wire-attempt ledger rather than a logical-call approximation.

        ``Content-Length`` is rejected before reading a body when possible.
        Unknown-length/chunked bodies are read only through the configured
        acceptance limit plus one overflow sentinel byte, then the response is
        closed.  The exact upstream transfer size is not claimed; the trace
        records the bounded application-read semantics explicitly.
        """

        if accepted_body_limit < 0:
            raise ValueError("accepted_body_limit must not be negative")
        if isinstance(self.client, ScientificHttpClient):
            raw_client = getattr(self.client, "_client", None)
            if raw_client is None or not hasattr(raw_client, "stream"):
                raise RuntimeError(
                    "ScientificHttpClient does not expose its pooled stream"
                )
            body = bytearray()
            response: Any | None = None
            declared_content_length: int | None = None
            try:
                with raw_client.stream(
                    "GET",
                    endpoint,
                    params=dict(params),
                    timeout=timeout,
                    follow_redirects=False,
                    headers={"Accept-Encoding": "identity"},
                ) as response:
                    if self._clock() >= deadline_at:
                        return _EuropePMCOAWireResult(
                            response=response,
                            body=b"",
                            response_body_complete=False,
                            overflow_reason="skipped_elapsed_budget",
                            transport_mode=(
                                "scientific_httpx_one_shot_stream"
                            ),
                            transport_body_cap_enforced=True,
                            declared_content_length=None,
                        )
                    declared_content_length = _response_content_length(response)
                    if (
                        declared_content_length is not None
                        and declared_content_length > accepted_body_limit
                    ):
                        return _EuropePMCOAWireResult(
                            response=response,
                            body=b"",
                            response_body_complete=False,
                            overflow_reason=overflow_reason,
                            transport_mode=(
                                "scientific_httpx_one_shot_stream"
                            ),
                            transport_body_cap_enforced=True,
                            declared_content_length=declared_content_length,
                        )
                    read_limit = accepted_body_limit + 1
                    # Preserve the transport's natural decoded chunk cadence.
                    # Fixed-size rechunking can buffer a slow drip for longer
                    # than the acceptance deadline before yielding control.
                    for chunk in response.iter_bytes():
                        if self._clock() >= deadline_at:
                            return _EuropePMCOAWireResult(
                                response=response,
                                body=bytes(body),
                                response_body_complete=False,
                                overflow_reason="skipped_elapsed_budget",
                                transport_mode=(
                                    "scientific_httpx_one_shot_stream"
                                ),
                                transport_body_cap_enforced=True,
                                declared_content_length=(
                                    declared_content_length
                                ),
                            )
                        remaining = read_limit - len(body)
                        if remaining <= 0:
                            break
                        body.extend(chunk[:remaining])
                        if self._clock() >= deadline_at:
                            return _EuropePMCOAWireResult(
                                response=response,
                                body=bytes(body),
                                response_body_complete=False,
                                overflow_reason="skipped_elapsed_budget",
                                transport_mode=(
                                    "scientific_httpx_one_shot_stream"
                                ),
                                transport_body_cap_enforced=True,
                                declared_content_length=(
                                    declared_content_length
                                ),
                            )
                        if len(chunk) > remaining or len(body) > accepted_body_limit:
                            return _EuropePMCOAWireResult(
                                response=response,
                                body=bytes(body),
                                response_body_complete=False,
                                overflow_reason=overflow_reason,
                                transport_mode=(
                                    "scientific_httpx_one_shot_stream"
                                ),
                                transport_body_cap_enforced=True,
                                declared_content_length=(
                                    declared_content_length
                                ),
                            )
                    return _EuropePMCOAWireResult(
                        response=response,
                        body=bytes(body),
                        response_body_complete=True,
                        overflow_reason=None,
                        transport_mode="scientific_httpx_one_shot_stream",
                        transport_body_cap_enforced=True,
                        declared_content_length=declared_content_length,
                    )
            except Exception as exc:
                raise _EuropePMCOAWireReadError(
                    exc,
                    body_prefix=bytes(body),
                    status_code=(
                        int(getattr(response, "status_code", 0) or 0)
                        if response is not None
                        else None
                    ),
                    declared_content_length=declared_content_length,
                    transport_mode="scientific_httpx_one_shot_stream",
                    transport_body_cap_enforced=True,
                ) from exc

        # Injectable clients keep the existing deterministic unit-test seam.
        # Their response may already be buffered, so the ledger says plainly
        # that only the production ScientificHttpClient path enforces the body
        # cap at stream-read time.
        response = self.client.get(
            endpoint,
            params=dict(params),
            timeout=timeout,
            follow_redirects=False,
        )
        raw = _response_payload_bytes(response)
        complete = len(raw) <= accepted_body_limit
        return _EuropePMCOAWireResult(
            response=response,
            body=(raw if complete else raw[: accepted_body_limit + 1]),
            response_body_complete=complete,
            overflow_reason=None if complete else overflow_reason,
            transport_mode="injected_client_single_get",
            transport_body_cap_enforced=False,
            declared_content_length=_response_content_length(response),
        )

    def _epmc_oa_trace(
        self,
        state: _EuropePMCOAEscalationState | None,
        *,
        trigger: Mapping[str, Any],
        status: str,
        stop_reason: str,
        result_records: Sequence[EvidenceRef],
    ) -> dict[str, Any]:
        calls = deepcopy(state.http_calls) if state is not None else []
        trace: dict[str, Any] = {
            "schema_version": "epmc_oa_fulltext_one_hop_trace/v1",
            "policy": _EPMC_OA_FULLTEXT_POLICY,
            "excerpt_selection_policy": _EPMC_OA_EXCERPT_POLICY,
            "enabled": True,
            "status": status,
            "stop_reason": stop_reason,
            "trigger": dict(trigger),
            "bounds": {
                "parent_fulltexts": _EPMC_OA_PARENT_LIMIT,
                "references": _EPMC_OA_REFERENCE_LIMIT,
                "child_fulltexts": _EPMC_OA_CHILD_FULLTEXT_LIMIT,
                "added_http_calls": _EPMC_OA_HTTP_CALL_LIMIT,
                "added_wire_attempts": _EPMC_OA_HTTP_CALL_LIMIT,
                "bytes_per_document": _EPMC_OA_DOCUMENT_BYTE_LIMIT,
                "aggregate_response_bytes": _EPMC_OA_AGGREGATE_BYTE_LIMIT,
                "elapsed_seconds": _EPMC_OA_ELAPSED_SECONDS,
                "elapsed_acceptance_deadline_seconds": (
                    _EPMC_OA_ELAPSED_SECONDS
                ),
                "citation_hops": 1,
            },
            "observed": {
                "added_http_call_count": len(calls),
                "added_wire_attempt_count": sum(
                    int(call.get("wire_attempt_count") or 0) for call in calls
                ),
                "response_bytes_total": (
                    state.response_bytes_total if state is not None else 0
                ),
                "parent_fulltexts_attempted": (
                    state.parent_fulltexts_attempted if state is not None else 0
                ),
                "parent_fulltexts_parsed": (
                    state.parent_fulltexts_parsed if state is not None else 0
                ),
                "external_jats_doctypes_removed": (
                    state.external_jats_doctypes_removed
                    if state is not None
                    else 0
                ),
                "references_retained": (
                    state.references_retained if state is not None else 0
                ),
                "cross_parent_duplicate_references_dropped": (
                    state.cross_parent_duplicate_references_dropped
                    if state is not None
                    else 0
                ),
                "child_metadata_checked": (
                    state.child_metadata_checked if state is not None else 0
                ),
                "child_fulltexts_attempted": (
                    state.child_fulltexts_attempted if state is not None else 0
                ),
                "child_fulltexts_parsed": (
                    state.child_fulltexts_parsed if state is not None else 0
                ),
                "result_record_count": len(result_records),
                "elapsed_seconds": round(
                    self._clock() - state.started_at,
                    3,
                )
                if state is not None
                else 0.0,
            },
            "eligibility_rejections": (
                dict(sorted(state.eligibility_rejections.items()))
                if state is not None
                else {}
            ),
            "error_reason_codes": (
                list(dict.fromkeys(state.error_reason_codes))
                if state is not None
                else []
            ),
            "reference_audits": (
                deepcopy(state.reference_audits) if state is not None else []
            ),
            "discovery_parents": (
                deepcopy(state.discovery_parents) if state is not None else []
            ),
            "http_calls": calls,
            "http_call_ledger_sha256": sha256(
                json.dumps(
                    calls,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "official_europe_pmc_only": True,
            "forward_citations_followed": False,
            "publisher_or_pdf_calls": 0,
            "fixture_calls": 0,
            "retry_policy": "one_wire_attempt_per_ledger_entry_no_retries",
            "elapsed_limit_semantics": (
                "request_admission_and_response_acceptance_deadline"
            ),
            "elapsed_budget_semantics": "admission_deadline",
            "exact_wall_clock_cap_claimed": False,
            "exact_transport_wall_clock_deadline_claimed": False,
            "response_body_limit_semantics": {
                "document_bytes": "accepted_body_bytes",
                "aggregate_bytes": "accepted_or_bounded_prefix_bytes_read",
                "live_transport": (
                    "content_length_precheck_or_stream_read_limit_plus_one_sentinel"
                ),
                "unknown_length_stream_overflow_sentinel_bytes": 1,
                "exact_upstream_network_transfer_bytes_claimed": False,
                "injected_buffered_client_stream_cap_claimed": False,
            },
            "raw_xml_stored": False,
            "raw_reference_payload_stored": False,
            "full_text_stored": False,
        }
        trace["trace_sha256"] = sha256(
            json.dumps(
                trace,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return trace

    def _epmc_oa_article_records(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        *,
        item: Mapping[str, Any],
        parsed: Any,
        query: str,
        fulltext_endpoint: str,
        lineage: Any | None,
        abstract_records: Sequence[EvidenceRef],
    ) -> list[EvidenceRef]:
        """Project at most one direct and one association role per OA paper."""

        pmid = parsed.audit.pmid or _clean_optional_text(item.get("pmid"))
        pmcid = parsed.audit.pmcid
        canonical_doi = parsed.audit.doi
        external_id = pmid or pmcid
        if external_id is None:
            return []
        title = _plain_text(
            str(item.get("title") or f"Europe PMC record {external_id}")
        )
        publication_types = _europe_pmc_publication_types(item)
        article_path = f"MED/{pmid}" if pmid else f"PMC/{pmcid}"
        abstract_parent = next(
            (
                record
                for record in abstract_records
                if record.source == "Europe PMC"
                and (
                    record.external_id == external_id
                    or record.metadata.get("pmid") == pmid
                    or record.metadata.get("pmcid") == pmcid
                )
                and not record.metadata.get("role_specific_record")
            ),
            None,
        )
        direct_candidates: list[tuple[tuple[int, int, str], EvidenceRef]] = []
        association_candidates: list[tuple[int, EvidenceRef]] = []
        association_parents: list[tuple[int, EvidenceRef]] = []
        for span in parsed.spans:
            content = _publication_content_from_sections(
                (("full_text", span.text),),
                target_symbol=seed.symbol,
                disease=brief.disease,
                intervention_direction=(
                    brief.constraints.intervention_direction
                ),
                doi=canonical_doi,
                pmcid=pmcid,
            )
            if content.excerpt is None and content.association_excerpt is None:
                continue
            direction_rejections = _primary_direction_content_rejection_codes(
                content,
                title=title,
                publication_types=publication_types,
                target_symbol=seed.symbol,
                disease=brief.disease,
                intervention_direction=(
                    brief.constraints.intervention_direction
                ),
            )
            direct_eligible = not direction_rejections
            selected_excerpt = content.excerpt or content.association_excerpt
            if selected_excerpt is None:
                continue
            parser_metadata = parsed.evidence_metadata(
                span,
                lineage=lineage,
            )
            parser_span_hash = parser_metadata.get("excerpt_hash")
            selected_excerpt_hash = sha256(
                selected_excerpt.encode("utf-8")
            ).hexdigest()
            record = self._record(
                source="Europe PMC",
                external_id=external_id,
                title=title,
                url=f"https://europepmc.org/article/{article_path}",
                supports="support" if direct_eligible else "context",
                query=query,
                endpoint=fulltext_endpoint,
                excerpt_text=selected_excerpt,
                hash_title_when_no_excerpt=False,
                extra={
                    **parser_metadata,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    # The parser binds and canonicalizes metadata/JATS identity.
                    # Never overwrite that audited lowercase DOI with a
                    # differently cased or missing search-result field.
                    "doi": canonical_doi,
                    "publication_year": item.get("pubYear"),
                    "journal": item.get("journalTitle"),
                    "publication_types": publication_types,
                    "is_open_access": True,
                    "source_location": parser_metadata.get(
                        "source_location"
                    ),
                    "excerpt": selected_excerpt,
                    "excerpt_hash": selected_excerpt_hash,
                    "excerpt_char_count": len(selected_excerpt),
                    "excerpt_max_chars": _MAX_PUBLICATION_EXCERPT_CHARS,
                    "excerpt_truncated": bool(
                        parser_metadata.get("excerpt_truncated")
                        or (
                            content.excerpt_truncated
                            if content.excerpt is not None
                            else content.association_excerpt_truncated
                        )
                    ),
                    "excerpt_selection_status": (
                        "selected" if content.excerpt is not None else "context"
                    ),
                    "excerpt_selection_policy": _EPMC_OA_EXCERPT_POLICY,
                    "scientific_excerpt_classification_policy": (
                        "same-sentence-target-disease-direction-aware-"
                        "mechanistic-ranking-v2"
                    ),
                    "full_text_span_excerpt_hash": parser_span_hash,
                    "content_retrieval_policy": _EPMC_OA_FULLTEXT_POLICY,
                    "content_retrieval_endpoint": fulltext_endpoint,
                    "metadata_retrieval_endpoint": EUROPE_PMC_SEARCH_URL,
                    "metadata_query_sha256": sha256(
                        query.encode("utf-8")
                    ).hexdigest(),
                    "abstract_parent_evidence_id": (
                        abstract_parent.evidence_id
                        if abstract_parent is not None
                        else None
                    ),
                    "abstract_parent_content_hash": (
                        abstract_parent.metadata.get("content_hash")
                        if abstract_parent is not None
                        else None
                    ),
                    "direct_support_eligible": direct_eligible,
                    "direct_support_rejection_codes": list(
                        direction_rejections
                    ),
                    "verified_direct_primary": direct_eligible,
                    "verified_direct_primary_policy": (
                        _VERIFIED_DIRECT_PRIMARY_POLICY
                    ),
                    "verified_direct_primary_rejection_codes": list(
                        direction_rejections
                    ),
                    "claim_linkage_status": (
                        "pending_ma1_claim_reconciliation"
                    ),
                    "counter_category": None,
                    "counter_signal": None,
                    "review_status": (
                        None
                        if direct_eligible
                        else "researcher_review_required"
                    ),
                    "full_text_stored": False,
                    "raw_xml_stored": False,
                    "content_storage_policy": (
                        "bounded_oa_excerpt_only_no_full_text"
                    ),
                    "evidence_scope": (
                        "primary non-review OA full-text direct target-"
                        "direction measured evidence"
                        if direct_eligible
                        else "primary non-review OA full-text context; not "
                        "direct target-direction proof"
                    ),
                },
            )
            origin_digest = sha256(
                (
                    f"Europe PMC|{external_id}|oa_fulltext_excerpt|"
                    f"{parser_metadata.get('citation_hop')}|"
                    f"{parser_metadata.get('citation_relation')}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            record = record.model_copy(
                update={
                    "evidence_id": f"ev_{origin_digest}",
                    "metadata": {
                        **record.metadata,
                        "evidence_id_origin_policy": (
                            "source-external-content-origin-hop-relation-v1"
                        ),
                    },
                },
                deep=True,
            )
            if direct_eligible:
                direct_score = (
                    len(_MECHANISTIC_ENDPOINT_PATTERN.findall(selected_excerpt)),
                    len(_EXPERIMENTAL_RESULT_PATTERN.findall(selected_excerpt)),
                    str(parser_metadata.get("source_location") or ""),
                )
                direct_candidates.append((direct_score, record))
            association = _fulltext_association_role_record(record, content)
            if association is None:
                continue
            association_rejections = (
                _primary_association_candidate_rejection_codes(
                    association,
                    requirement=_primary_association_requirement(
                        brief,
                        seed.symbol,
                    ),
                    title=title,
                    publication_types=publication_types,
                )
            )
            if association_rejections:
                continue
            association_candidates.append(
                (content.association_score, association)
            )
            association_parents.append((content.association_score, record))

        selected: list[EvidenceRef] = []
        selected_parent: EvidenceRef | None = None
        if direct_candidates:
            direct_candidates.sort(key=lambda value: value[0], reverse=True)
            selected_parent = direct_candidates[0][1]
            selected.append(selected_parent)
        if association_candidates:
            association_candidates.sort(
                key=lambda value: (value[0], value[1].evidence_id),
                reverse=True,
            )
            if not selected and association_parents:
                association_parents.sort(
                    key=lambda value: (value[0], value[1].evidence_id),
                    reverse=True,
                )
                selected_parent = association_parents[0][1]
                selected.append(selected_parent)
            association = association_candidates[0][1]
            if selected_parent is not None:
                association = association.model_copy(
                    update={
                        "metadata": {
                            **association.metadata,
                            "association_source_excerpt_hash": (
                                association.excerpt_hash
                            ),
                            "parent_evidence_id": selected_parent.evidence_id,
                            "parent_excerpt_hash": selected_parent.excerpt_hash,
                            "parent_provenance_pointer_type": (
                                "selected_oa_document_record_id_not_claim_link"
                            ),
                        }
                    },
                    deep=True,
                )
            selected.append(association)
        return selected

    def _europe_pmc_counter_records(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        query: str,
        *,
        excluded_ids: set[str],
    ) -> list[EvidenceRef]:
        response = self.client.get(
            EUROPE_PMC_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "pageSize": min(20, max(6, self.max_records_per_source * 4)),
                "resultType": "core",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        query_sha256 = sha256(query.encode("utf-8")).hexdigest()
        raw_hit_count = len(results) if isinstance(results, list) else 0
        records: list[EvidenceRef] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            external_id = str(pmid or pmcid or item.get("id") or "").strip()
            if not external_id or external_id in excluded_ids:
                continue
            title = _plain_text(str(item.get("title") or ""))
            abstract = _plain_text(str(item.get("abstractText") or ""))
            publication_types = _europe_pmc_publication_types(item)
            if _primary_publication_metadata_rejection_codes(
                title,
                publication_types,
            ):
                continue
            classified = _classify_counter_signal(
                title,
                abstract,
                seed.symbol,
                intervention_direction=brief.constraints.intervention_direction,
            )
            if classified is None:
                continue
            category, signal, excerpt, location = classified
            article_path = (
                f"MED/{pmid}" if pmid else f"PMC/{pmcid}" if pmcid else external_id
            )
            records.append(
                self._record(
                    source="Europe PMC",
                    external_id=external_id,
                    title=title or f"Europe PMC record {external_id}",
                    url=f"https://europepmc.org/article/{article_path}",
                    supports="counter",
                    query=query,
                    endpoint=EUROPE_PMC_SEARCH_URL,
                    excerpt_text=excerpt,
                    extra={
                        "publication_year": item.get("pubYear"),
                        "journal": item.get("journalTitle"),
                        "cited_by_count": item.get("citedByCount"),
                        "is_open_access": item.get("isOpenAccess"),
                        "counter_search_status": "completed_with_results",
                        "counter_search_source": "Europe PMC",
                        "counter_search_query": query,
                        "counter_search_query_sha256": query_sha256,
                        "counter_search_separate_query": True,
                        "counter_search_raw_hit_count": raw_hit_count,
                        "counter_category": category,
                        "counter_signal": signal,
                        "publication_types": publication_types,
                        "classification_policy": (
                            "same-clause-direct-target-intervention-counter-"
                            "signal-proximity80-v2"
                        ),
                        "review_status": "researcher_review_required",
                        "source_location": location,
                        "detail": excerpt,
                        "evidence_scope": (
                            "real literature limitation/counter-signal; not a definitive "
                            "refutation until researcher review"
                        ),
                    },
                )
            )
            if len(records) >= self.max_records_per_source:
                break
        return records

    def _open_targets(self, brief: ResearchBrief, seed: TargetSeed) -> list[EvidenceRef]:
        query = seed.symbol
        graphql = {
            "query": (
                "query SearchTarget($q:String!,$size:Int!){"
                "search(queryString:$q,page:{index:0,size:$size}){"
                "hits{id name entity description}}}"
            ),
            "variables": {"q": query, "size": self.max_records_per_source},
        }
        response = self.client.post(
            OPEN_TARGETS_GRAPHQL_URL,
            json=graphql,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors") and not payload.get("data"):
            raise ValueError("Open Targets returned GraphQL errors")
        hits = payload.get("data", {}).get("search", {}).get("hits", [])
        evidence: list[EvidenceRef] = []
        for hit in hits[: self.max_records_per_source]:
            if not isinstance(hit, Mapping) or hit.get("entity") != "target":
                continue
            external_id = hit.get("id")
            if not external_id:
                continue
            name = str(hit.get("name") or seed.symbol)
            expected_ensembl = str(seed.ensembl_id or "").strip().casefold()
            identity_matches = (
                str(external_id).strip().casefold() == expected_ensembl
                if expected_ensembl
                else name.strip().casefold() == seed.symbol.strip().casefold()
            )
            if not identity_matches:
                continue
            evidence.append(
                self._record(
                    source="Open Targets",
                    external_id=str(external_id),
                    title=f"Open Targets target record: {name}",
                    url=f"https://platform.opentargets.org/target/{external_id}",
                    supports="context",
                    query=query,
                    endpoint=OPEN_TARGETS_GRAPHQL_URL,
                    extra={
                        "entity": hit.get("entity"),
                        "description": hit.get("description"),
                        "disease": brief.disease,
                        "disease_id": brief.disease_id,
                        "evidence_scope": "target identity context",
                        "canonical_target_symbol": seed.symbol,
                        "canonical_target_ensembl_id": seed.ensembl_id,
                        "canonical_identity_match": True,
                    },
                )
            )
            # A target identity lookup must contribute at most one exact
            # canonical record; fuzzy search neighbours are not evidence for
            # the requested target.
            break
        return evidence

    def _uniprot(self, brief: ResearchBrief, seed: TargetSeed) -> list[EvidenceRef]:
        del brief
        canonical_accession = str(seed.uniprot_id or "").strip().upper()
        query = (
            f"accession:{canonical_accession}"
            if canonical_accession
            else f"gene_exact:{seed.symbol} AND organism_id:9606 AND reviewed:true"
        )
        response = self.client.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "size": 1 if canonical_accession else self.max_records_per_source,
                "fields": (
                    "accession,id,protein_name,gene_names,organism_name,reviewed,cc_function"
                ),
            },
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        evidence: list[EvidenceRef] = []
        for item in results[: self.max_records_per_source]:
            if not isinstance(item, Mapping):
                continue
            accession = item.get("primaryAccession")
            if not accession:
                continue
            normalized_accession = str(accession).strip().upper()
            if canonical_accession and normalized_accession != canonical_accession:
                continue
            protein_name = _uniprot_protein_name(item) or seed.target_name
            gene_names = _uniprot_gene_names(item)
            evidence.append(
                self._record(
                    source="UniProt",
                    external_id=str(accession),
                    title=f"UniProtKB protein record: {protein_name}",
                    url=f"https://www.uniprot.org/uniprotkb/{accession}/entry",
                    supports="context",
                    query=query,
                    endpoint=UNIPROT_SEARCH_URL,
                    extra={
                        "uniprot_id": item.get("uniProtkbId"),
                        "entry_type": item.get("entryType"),
                        "gene_names": gene_names,
                        "organism": _mapping_value(item.get("organism"), "scientificName"),
                        "evidence_scope": "curated protein identity and function context",
                        "canonical_target_accession": canonical_accession or None,
                        "canonical_only_scope": bool(canonical_accession),
                    },
                )
            )
        return evidence

    def _dgidb(self, brief: ResearchBrief, seed: TargetSeed) -> list[EvidenceRef]:
        del brief
        query = seed.symbol
        graphql = {
            "query": (
                "query GeneContext($names:[String!]!){genes(names:$names){nodes{"
                "name longName conceptId geneCategoriesWithSources{name sourceNames}"
                "interactions{drug{name conceptId} interactionScore "
                "interactionTypes{type directionality} publications{pmid} "
                "sources{sourceDbName}}}}}"
            ),
            "variables": {"names": [seed.symbol]},
        }
        response = self.client.post(
            DGIDB_GRAPHQL_URL,
            json=graphql,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors") and not payload.get("data"):
            raise ValueError("DGIdb returned GraphQL errors")
        nodes = payload.get("data", {}).get("genes", {}).get("nodes", [])
        evidence: list[EvidenceRef] = []
        for node in nodes[: self.max_records_per_source]:
            if not isinstance(node, Mapping):
                continue
            external_id = node.get("conceptId") or node.get("name")
            if not external_id:
                continue
            gene_name = str(node.get("name") or seed.symbol)
            interactions = node.get("interactions", [])
            compact_interactions = [
                _compact_dgidb_interaction(item)
                for item in interactions[: self.max_records_per_source]
                if isinstance(item, Mapping)
            ]
            evidence.append(
                self._record(
                    source="DGIdb",
                    external_id=str(external_id),
                    title=f"DGIdb drug-gene context: {gene_name}",
                    url=f"https://dgidb.org/genes/{external_id}",
                    supports="context",
                    query=query,
                    endpoint=DGIDB_GRAPHQL_URL,
                    extra={
                        "gene_name": node.get("longName"),
                        "categories": node.get("geneCategoriesWithSources", []),
                        "interaction_count": len(interactions),
                        "interactions": compact_interactions,
                        "evidence_scope": (
                            "aggregated drug-gene interaction context; not efficacy proof"
                        ),
                    },
                )
            )
        return evidence

    def _pharos(self, brief: ResearchBrief, seed: TargetSeed) -> list[EvidenceRef]:
        del brief
        query = seed.symbol
        graphql = {
            "query": (
                "query TargetSearch($term:String!,$top:Int!){"
                "targets(filter:{term:$term},top:$top){count targets{"
                "name sym description tdl fam novelty}}}"
            ),
            "variables": {"term": seed.symbol, "top": self.max_records_per_source},
        }
        response = self.client.post(
            PHAROS_GRAPHQL_URL,
            json=graphql,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors") and not payload.get("data"):
            raise ValueError("Pharos returned GraphQL errors")
        target_result = payload.get("data", {}).get("targets", {})
        targets = target_result.get("targets", [])
        evidence: list[EvidenceRef] = []
        for target in targets[: self.max_records_per_source]:
            if not isinstance(target, Mapping):
                continue
            symbol = target.get("sym")
            if not symbol:
                continue
            evidence.append(
                self._record(
                    source="Pharos",
                    external_id=str(symbol),
                    title=f"Pharos target development record: {target.get('name') or symbol}",
                    url=f"https://pharos.nih.gov/targets/{symbol}",
                    supports="context",
                    query=query,
                    endpoint=PHAROS_GRAPHQL_URL,
                    extra={
                        "description": target.get("description"),
                        "tdl": target.get("tdl"),
                        "target_family": target.get("fam"),
                        "novelty": target.get("novelty"),
                        "result_count": target_result.get("count"),
                        "evidence_scope": (
                            "target development level and druggability context"
                        ),
                    },
                )
            )
        return evidence

    def _record(
        self,
        *,
        source: str,
        external_id: str,
        title: str,
        url: str,
        supports: str,
        query: str,
        endpoint: str,
        extra: Mapping[str, Any],
        excerpt_text: str | None = None,
        hash_title_when_no_excerpt: bool = True,
    ) -> EvidenceRef:
        excerpt_hash = (
            sha256(excerpt_text.encode("utf-8")).hexdigest()
            if excerpt_text
            else sha256(title.encode("utf-8")).hexdigest()
            if hash_title_when_no_excerpt
            else None
        )
        digest = sha256(
            (
                f"{source}|{external_id}|{query}|"
                f"{extra.get('content_availability') or ''}|"
                f"{extra.get('source_location') or ''}|{excerpt_hash or ''}"
            ).encode("utf-8")
        ).hexdigest()[:12]
        return EvidenceRef(
            evidence_id=f"ev_{digest}",
            source=source,
            external_id=external_id,
            title=title,
            url=url,
            supports=supports,
            excerpt_hash=excerpt_hash,
            metadata={
                **dict(extra),
                "provenance_class": "external_live",
                "origin_provenance_class": "external_live",
                "origin_content_hash": (
                    extra.get("content_hash") or excerpt_hash
                ),
                "query": query,
                "live": True,
                "mock": False,
                "cache_hit": False,
                "retrieval_endpoint": endpoint,
                "evidence_id_origin_policy": (
                    "source-external-query-location-excerpt-v2"
                ),
            },
        )

    def _ncbi_get(self, url: str, **kwargs: Any) -> Any:
        """Issue a rate-bounded NCBI request using configured identity params.

        NCBI permits a higher request rate with an API key.  Serializing these
        calls at the adapter boundary keeps multiple MA1 target lookups within
        the published etiquette limit without changing the injectable HTTP
        client used by tests.
        """

        self._ncbi_rate_limiter.acquire(
            api_key_configured=bool(self.ncbi_api_key)
        )
        return self.client.get(url, **kwargs)

    def _cache_key(
        self,
        source: str,
        brief: ResearchBrief,
        seed: TargetSeed,
    ) -> tuple[str, ...]:
        association_requirement = (
            _primary_association_requirement(brief, seed.symbol)
            if source in {"PubMed", "Europe PMC"}
            else None
        )
        association_requirement_key = (
            "|".join(
                (
                    str(association_requirement.required_scope or "none"),
                    *(
                        f"{clause.kind}:{clause.canonical.casefold()}"
                        for clause in association_requirement.variant_clauses
                    ),
                )
            )
            if association_requirement is not None
            and association_requirement.requested
            else "not_requested"
        )
        return (
            source,
            brief.disease.casefold().strip(),
            (brief.disease_id or "").casefold().strip(),
            seed.symbol.casefold().strip(),
            (
                brief.constraints.intervention_direction
                if source in {"PubMed", "Europe PMC"}
                else ""
            ),
            association_requirement_key,
            (
                str(brief.research_depth or "legacy")
                if source == "Europe PMC"
                else ""
            ),
            (
                _EPMC_OA_FULLTEXT_POLICY
                if source == "Europe PMC"
                and self.epmc_oa_fulltext_one_hop_v1
                else "abstract_only"
                if source == "Europe PMC"
                else ""
            ),
        )

    def _cached_records(
        self,
        source: str,
        brief: ResearchBrief,
        seed: TargetSeed,
    ) -> tuple[EvidenceRef, ...] | None:
        key = self._cache_key(source, brief, seed)
        now = self._clock()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now - entry.stored_at > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            records = tuple(item.model_copy(deep=True) for item in entry.records)
            for item in records:
                current_hash = (
                    item.metadata.get("content_hash") or item.excerpt_hash
                )
                if (
                    item.metadata.get("provenance_class") != "external_live"
                    or item.metadata.get("origin_provenance_class")
                    != "external_live"
                    or item.metadata.get("mock") is True
                    or not current_hash
                    or item.metadata.get("origin_content_hash") != current_hash
                ):
                    self._cache.pop(key, None)
                    return None
        return tuple(
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "provenance_class": "external_cache",
                        "origin_provenance_class": "external_live",
                        "origin_content_hash": (
                            item.metadata.get("content_hash")
                            or item.excerpt_hash
                        ),
                        "live": False,
                        "cache_hit": True,
                        "retrieved_live": True,
                    }
                },
                deep=True,
            )
            for item in records
        )

    def _store_records(
        self,
        source: str,
        brief: ResearchBrief,
        seed: TargetSeed,
        records: Sequence[EvidenceRef],
    ) -> None:
        entry = _CacheEntry(
            stored_at=self._clock(),
            records=tuple(item.model_copy(deep=True) for item in records),
        )
        with self._cache_lock:
            self._cache[self._cache_key(source, brief, seed)] = entry

    def _fallback_or_raise(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
        failures: Mapping[str, str],
    ) -> Sequence[EvidenceRef]:
        with self._cache_lock:
            self._last_failures = dict(failures)
        if self.fallback is None:
            detail = "; ".join(f"{source}: {reason}" for source, reason in failures.items())
            raise RuntimeError(
                f"No public biology evidence returned for {seed.symbol}"
                + (f" ({detail})" if detail else "")
            )

        query = self._plain_query(brief, seed)
        fixtures = self.fallback.retrieve(brief, seed, iteration)
        return tuple(
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "provenance_class": "fixture_fallback",
                        "query": query,
                        "live": False,
                        "mock": True,
                        "cache_hit": False,
                        "fallback_reason": dict(failures),
                    }
                },
                deep=True,
            )
            for item in fixtures
        )

    @staticmethod
    def _plain_query(brief: ResearchBrief, seed: TargetSeed) -> str:
        return f'"{_quote_term(seed.symbol)}" AND "{_quote_term(brief.disease)}"'

    @classmethod
    def _pubmed_query(cls, brief: ResearchBrief, seed: TargetSeed) -> str:
        disease_clause = _pubmed_disease_query_clause(
            brief.disease,
            target_symbol=seed.symbol,
            terms=_disease_query_terms(
                brief.disease,
                target_symbol=seed.symbol,
            ),
            field="Title/Abstract",
        )
        scientific_clause = " OR ".join(
            f'"{_quote_term(term)}"[Title/Abstract]'
            for term in _direction_aware_query_terms(
                brief.constraints.intervention_direction
            )
        )
        return (
            f'"{_quote_term(seed.symbol)}"[Title/Abstract] AND '
            f"({disease_clause}) AND "
            f"({scientific_clause})"
        )

    @classmethod
    def _pubmed_primary_association_query(
        cls,
        brief: ResearchBrief,
        seed: TargetSeed,
        *,
        requirement: _PrimaryAssociationRequirement | None = None,
    ) -> str:
        """Return the first, highest-precision association query.

        Kept as a narrow compatibility surface for callers that inspect the
        generated query.  Live supplement retrieval uses both lanes returned
        by :meth:`_pubmed_primary_association_query_lanes`.
        """

        return cls._pubmed_primary_association_query_lanes(
            brief,
            seed,
            requirement=requirement,
        )[0].query

    @classmethod
    def _pubmed_primary_direction_query(
        cls,
        brief: ResearchBrief,
        seed: TargetSeed,
    ) -> str:
        """Return one bounded direct-action direction precision query.

        A title-only query is precise but misses primary studies whose title
        names the target action while the disease and measured result appear
        only in the abstract.  Keep that high-precision branch, and add one
        exact target-action Title/Abstract branch bound to a disease alias and
        an experimental endpoint.  This is still a single query and a bounded,
        deterministically batched operation.  Retrieval never grants support:
        every result must pass the unchanged fail-closed abstract selector.
        """

        disease_clause = _pubmed_disease_query_clause(
            brief.disease,
            target_symbol=seed.symbol,
            terms=_primary_direction_disease_query_terms(
                brief.disease,
                target_symbol=seed.symbol,
            ),
            field="Title",
        )
        intervention_clause = " OR ".join(
            f'"{_quote_term(term)}"[Title]'
            for query_target in (_fusion_query_terms(seed) or (seed.symbol,))
            for term in _target_intervention_title_query_phrases(
                query_target,
                brief.constraints.intervention_direction,
            )
        )
        positive_response_clause = " OR ".join(
            f'"{term}"[Title]'
            for term in ("response", "responses")
        )
        literal_disease_branch = (
            f"({disease_clause}) AND ({intervention_clause}) AND "
            f"({positive_response_clause})"
        )
        abstract_disease_clause = _pubmed_disease_query_clause(
            brief.disease,
            target_symbol=seed.symbol,
            terms=_disease_query_terms(
                brief.disease,
                target_symbol=seed.symbol,
            ),
            field="Title/Abstract",
        )
        abstract_intervention_clause = " OR ".join(
            f'"{_quote_term(term)}"[Title/Abstract]'
            for query_target in (_fusion_query_terms(seed) or (seed.symbol,))
            for term in _target_intervention_precision_query_phrases(
                query_target,
                brief.constraints.intervention_direction,
            )
        )
        experimental_endpoint_clause = " OR ".join(
            f'"{term}"[Title/Abstract]'
            for term in (
                "phosphorylation",
                "viability",
                "proliferation",
                "growth arrest",
                "tumor growth",
                "tumour growth",
                "apoptosis",
                "cytotoxicity",
                "IC50",
                "response",
            )
        )
        abstract_experimental_branch = (
            f"({abstract_disease_clause}) AND "
            f"({abstract_intervention_clause}) AND "
            f"({experimental_endpoint_clause})"
        )
        context = _primary_direction_model_context(brief, seed.symbol)
        positive_branch = literal_disease_branch
        if context is not None:
            model_clause = " OR ".join(
                f'"{_quote_term(alias)}"[Title]'
                for alias in context.model_aliases
            )
            variant_phrases = tuple(
                dict.fromkeys(
                    phrase
                    for clause in context.variant_clauses
                    for phrase in clause.query_phrases
                )
            )[:12]
            variant_clause = " OR ".join(
                f'"{_quote_term(phrase)}"[Title]'
                for phrase in variant_phrases
            )
            molecular_model_branch = (
                f"({model_clause}) AND ({variant_clause}) AND "
                f"({intervention_clause})"
            )
            positive_branch = (
                f"({literal_disease_branch}) OR "
                f"({molecular_model_branch})"
            )
        return (
            f"(({positive_branch}) OR ({abstract_experimental_branch})) "
            "NOT (Editorial[Publication Type] OR Guideline[Publication Type]) "
            "NOT review[Publication Type]"
        )

    @classmethod
    def _pubmed_primary_association_query_lanes(
        cls,
        brief: ResearchBrief,
        seed: TargetSeed,
        *,
        requirement: _PrimaryAssociationRequirement | None = None,
    ) -> tuple[_PrimaryAssociationQueryLane, ...]:
        requirement = requirement or _primary_association_requirement(
            brief,
            seed.symbol,
        )
        disease_clause = _pubmed_disease_query_clause(
            brief.disease,
            target_symbol=seed.symbol,
            terms=_disease_query_terms(
                brief.disease,
                target_symbol=seed.symbol,
            ),
            field="Title/Abstract",
        )
        fusion_terms = _fusion_query_terms(seed)
        if fusion_terms:
            target_clause = "(" + " OR ".join(
                f'"{_quote_term(term)}"[Title/Abstract]' for term in fusion_terms) + ")"
        elif requirement.variant_clauses:
            variant_phrases = tuple(
                dict.fromkeys(
                    phrase
                    for clause in requirement.variant_clauses
                    for phrase in clause.query_phrases
                )
            )
            target_clause = " OR ".join(
                f'"{_quote_term(phrase)}"[Title/Abstract]'
                for phrase in variant_phrases
            )
            target_clause = f"({target_clause})"
        else:
            target_clause = (
                f'"{_quote_term(seed.symbol)}"[Title/Abstract]'
            )
        association_signal_clause = " OR ".join(
            f'"{term}"[Title/Abstract]'
            for term in (
                "frequency",
                "incidence",
                "association",
                "correlation",
                "expression",
                "rearrangement",
                "fusion",
            )
        )
        sampling_clause = " OR ".join(
            f'"{term}"[Title/Abstract]'
            for term in ("patients", "cases", "cohort", "samples")
        )
        prevalence_positive = (
            f"{target_clause} AND "
            f'({disease_clause}) AND "prevalence"[Title]'
        )
        rearrangement_requirement = any(
            clause.kind == "gene_rearrangement_or_fusion"
            for clause in requirement.variant_clauses
        )
        other_lane_id = "abstract_primary_association"
        if rearrangement_requirement:
            exact_disease_title_clause = _pubmed_disease_query_clause(
                brief.disease,
                target_symbol=seed.symbol,
                terms=_primary_direction_disease_query_terms(
                    brief.disease,
                    target_symbol=seed.symbol,
                ),
                field="Title",
            )
            # Coordinated titles such as ``ALK and ROS1 gene fusions`` do not
            # contain the contiguous phrase ``ALK gene fusions``.  Bind the
            # exact target and the exact gene-event vocabulary independently
            # at retrieval time, then leave target ownership, disease scope,
            # cohort quantitation, and sentence atomicity to the unchanged
            # fail-closed abstract selector.
            target_title_clause = (
                f'"{_quote_term(seed.symbol)}"[Title]'
            )
            gene_event_title_clause = " OR ".join(
                f'"{term}"[Title]'
                for term in (
                    "gene fusion",
                    "gene fusions",
                    "gene rearrangement",
                    "gene rearrangements",
                )
            )
            other_primary_positive = (
                f"{target_title_clause} AND "
                f"({exact_disease_title_clause}) AND "
                f"({gene_event_title_clause})"
            )
            other_lane_id = "title_target_variant_cohort_result_p2"
        else:
            other_primary_positive = (
                f"{target_clause} AND "
                f"({disease_clause}) AND ({association_signal_clause}) AND "
                f"({sampling_clause})"
            )
        prevalence_query = (
            f"({prevalence_positive}) NOT (Review[Publication Type])"
        )
        other_primary_query = (
            f"({other_primary_positive}) NOT \"prevalence\"[Title] "
            "NOT (Review[Publication Type])"
        )
        return (
            _PrimaryAssociationQueryLane(
                lane_id="title_prevalence",
                query=prevalence_query,
                raw_document_budget=(
                    _PRIMARY_ASSOCIATION_PREVALENCE_RAW_DOCUMENT_BUDGET
                ),
            ),
            _PrimaryAssociationQueryLane(
                lane_id=other_lane_id,
                query=other_primary_query,
                raw_document_budget=(
                    _PRIMARY_ASSOCIATION_OTHER_RAW_DOCUMENT_BUDGET
                ),
            ),
        )

    @classmethod
    def _pubmed_counter_query(
        cls,
        brief: ResearchBrief,
        seed: TargetSeed,
    ) -> str:
        disease_clause = _pubmed_disease_query_clause(
            brief.disease,
            target_symbol=seed.symbol,
            terms=_disease_query_terms(
                brief.disease,
                target_symbol=seed.symbol,
            ),
            field="Title/Abstract",
        )
        counter_clause = " OR ".join(
            f'"{term}"[Title/Abstract]'
            for term in (
                "resistance",
                "resistant",
                "refractory",
                "non-response",
                "lack of response",
                "no benefit",
                "limited efficacy",
                "failed to improve",
                "target escape",
                "dose-limiting toxicity",
            )
        )
        return (
            f'"{_quote_term(seed.symbol)}"[Title/Abstract] AND '
            f"({disease_clause}) AND ({counter_clause}) NOT "
            "(Review[Publication Type])"
        )

    @classmethod
    def _europe_pmc_query(cls, brief: ResearchBrief, seed: TargetSeed) -> str:
        disease_clause = _europe_pmc_disease_query_clause(
            brief.disease,
            target_symbol=seed.symbol,
            terms=_disease_query_terms(
                brief.disease,
                target_symbol=seed.symbol,
            ),
        )
        scientific_clause = " OR ".join(
            f'TITLE_ABS:"{_quote_term(term)}"'
            for term in _direction_aware_query_terms(
                brief.constraints.intervention_direction
            )
        )
        fusion_terms = _fusion_query_terms(seed)
        target_clause = ("(" + " OR ".join(f'TITLE_ABS:"{_quote_term(t)}"' for t in fusion_terms) + ")"
            if fusion_terms else f'TITLE_ABS:"{_quote_term(seed.symbol)}"')
        return (
            f'{target_clause} AND '
            f"({disease_clause}) AND "
            f"({scientific_clause})"
        )

    @classmethod
    def _europe_pmc_counter_query(cls, brief: ResearchBrief, seed: TargetSeed) -> str:
        disease_clause = _europe_pmc_disease_query_clause(
            brief.disease,
            target_symbol=seed.symbol,
            terms=_disease_query_terms(
                brief.disease,
                target_symbol=seed.symbol,
            ),
        )
        return (
            f'TITLE_ABS:"{_quote_term(seed.symbol)}" AND '
            f"({disease_clause}) AND "
            '(TITLE_ABS:resistance OR TITLE_ABS:resistant OR '
            'TITLE_ABS:refractory OR TITLE_ABS:"non-response" OR '
            'TITLE_ABS:"lack of response" OR TITLE_ABS:"no benefit" OR '
            'TITLE_ABS:"limited efficacy" OR TITLE_ABS:"failed to improve" OR '
            'TITLE_ABS:"target escape" OR TITLE_ABS:"dose-limiting toxicity")'
        )


def _direction_aware_query_terms(intervention_direction: str) -> tuple[str, ...]:
    if intervention_direction in {"modulate", "unspecified"}:
        # Modulation is intentionally direction-agnostic at intake. Search
        # both inhibitory and activating vocabularies so antibody, blockade,
        # agonist and related literature is not excluded before the unchanged
        # direction/evidence validators inspect it.
        direction_terms = (
            *_DIRECTION_QUERY_TERMS["modulate"],
            *_DIRECTION_QUERY_TERMS["inhibit"],
            *_DIRECTION_QUERY_TERMS["activate"],
        )
    else:
        direction_terms = _DIRECTION_QUERY_TERMS.get(
            intervention_direction,
            _DIRECTION_QUERY_TERMS["modulate"],
        )
    # dict preserves insertion order while guarding against accidental term
    # duplication if the curated vocabularies evolve.
    return tuple(dict.fromkeys((*direction_terms, *_MECHANISTIC_QUERY_TERMS)))


def _target_intervention_query_phrases(
    target_symbol: str,
    intervention_direction: str,
) -> tuple[str, ...]:
    """Derive exact target/action phrases without target-specific fixtures."""

    target = " ".join(target_symbol.split()).strip()
    if intervention_direction == "inhibit":
        actions = (
            "inhibitor",
            "inhibitors",
            "selective inhibitor",
            "specific inhibitor",
            "tyrosine kinase inhibitor",
            "tyrosine kinase inhibitors",
            "inhibition",
        )
    elif intervention_direction == "activate":
        actions = (
            "activator",
            "activators",
            "agonist",
            "agonists",
            "activation",
        )
    else:
        actions = (
            "modulator",
            "modulators",
            "modulation",
            "regulation",
            "inhibition",
            "inhibitor",
            "knockdown",
            "silencing",
            "activation",
            "agonist",
            "overexpression",
        )
    return tuple(f"{target} {action}" for action in actions)


def _target_intervention_precision_query_phrases(
    target_symbol: str,
    intervention_direction: str,
) -> tuple[str, ...]:
    """Return bounded exact phrases covering both target/action word orders."""

    target = " ".join(target_symbol.split()).strip()
    if intervention_direction in {"modulate", "unspecified"}:
        # Balance directions before applying the bounded query size; a prefix
        # slice of inhibit phrases alone would silently lose activation recall.
        return tuple(dict.fromkeys(
            (*_target_intervention_query_phrases(target, intervention_direction),
             f"inhibition of {target}", f"silencing of {target}",
             f"activation of {target}", f"overexpression of {target}",
             f"modulation of {target}")
        ))[:16]
    target_first = _target_intervention_query_phrases(
        target,
        intervention_direction,
    )
    if intervention_direction == "inhibit":
        action_first = (
            f"inhibit {target}",
            f"inhibits {target}",
            f"inhibited {target}",
            f"inhibition of {target}",
            f"suppression of {target}",
            f"blockade of {target}",
        )
    elif intervention_direction == "activate":
        action_first = (
            f"activate {target}",
            f"activates {target}",
            f"activated {target}",
            f"activation of {target}",
            f"stimulation of {target}",
        )
    else:
        action_first = (
            f"modulate {target}",
            f"modulates {target}",
            f"modulated {target}",
            f"modulation of {target}",
            f"regulation of {target}",
        )
    return tuple(dict.fromkeys((*target_first, *action_first)))[:16]


def _target_intervention_title_query_phrases(
    target_symbol: str,
    intervention_direction: str,
) -> tuple[str, ...]:
    """Return the two generic target-bound action nouns for title search."""

    target = " ".join(target_symbol.split()).strip()
    if intervention_direction in {"modulate", "unspecified"}:
        return tuple(f"{target} {action}" for action in (
            "inhibitor", "activation", "agonist", "modulator", "knockdown", "overexpression",
        ))
    actions = {
        "inhibit": ("inhibitor", "inhibitors"),
        "activate": ("activator", "activators"),
        "modulate": ("modulator", "modulators"),
    }.get(
        intervention_direction,
        ("modulator", "modulators"),
    )
    return tuple(f"{target} {action}" for action in actions)


def _primary_direction_disease_query_terms(
    disease: str,
    *,
    target_symbol: str,
) -> tuple[str, ...]:
    """Keep precision retrieval on exact full disease/base-name aliases."""

    molecular = _target_molecular_disease_base(
        disease,
        target_symbol=target_symbol,
    )
    if molecular is not None:
        base, _ = molecular
        return tuple(
            dict.fromkeys((disease, *_terminal_plural_aliases(base)))
        )
    return tuple(dict.fromkeys((disease, *_disease_initialisms(disease))))


def _primary_association_supplement_requested(
    brief: ResearchBrief,
    target_symbol: str,
) -> bool:
    return _primary_association_requirement(brief, target_symbol).requested


def _primary_association_requirement(
    brief: ResearchBrief,
    target_symbol: str,
) -> _PrimaryAssociationRequirement:
    fields: tuple[tuple[str, str], ...] = (
        ("research_goal", brief.research_goal),
        ("rationale", brief.rationale or ""),
        *(
            (f"human_priorities[{index}]", value)
            for index, value in enumerate(brief.human_priorities)
        ),
    )
    clauses: list[_TargetVariantClause] = []
    trigger_fields: list[str] = []
    explicit_association = False
    for field_name, text in fields:
        if not text:
            continue
        field_clauses = _target_variant_clauses(text, target_symbol)
        if field_clauses:
            clauses.extend(field_clauses)
            trigger_fields.append(field_name)
        if re.search(
            r"(?:\b(?:primary association|primary prevalence|cohort prevalence|"
            r"non-review primary cohort|primary patient cohort)\b|"
            r"1차[^.!?]{0,32}(?:환자\s*)?(?:코호트|연관|유병)|"
            r"질환[- ]?표적\s*연관)",
            text,
            re.I,
        ):
            explicit_association = True
            trigger_fields.append(field_name)
    unique_clauses = tuple(
        {
            (clause.kind, clause.canonical.casefold()): clause
            for clause in clauses
        }.values()
    )
    # MA1 requires a primary target-disease association whenever the Brief
    # requires primary sources. Retrieval must honor the same contract even
    # when the user did not spell out "primary association" in prose.
    requested = bool(brief.data_policy.require_primary_sources)
    if requested and not trigger_fields:
        trigger_fields.append("data_policy.require_primary_sources")
    clause_kinds = {clause.kind for clause in unique_clauses}
    if clause_kinds == {"gene_rearrangement_or_fusion"}:
        required_scope = "rearranged_or_fusion_target_subtype"
    elif unique_clauses:
        required_scope = "activating_or_itd_target_subtype"
    elif requested:
        required_scope = "target_general"
    else:
        required_scope = None
    return _PrimaryAssociationRequirement(
        target_symbol=target_symbol,
        requested=requested,
        required_scope=required_scope,
        variant_clauses=unique_clauses,
        trigger_fields=tuple(dict.fromkeys(trigger_fields)),
    )


def _target_variant_clauses(
    text: str,
    target_symbol: str,
) -> tuple[_TargetVariantClause, ...]:
    """Extract only exact target-adjacent molecular clauses from Brief text."""

    target = rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
    clauses: list[_TargetVariantClause] = []
    itd = re.compile(
        rf"(?:{target}\s*(?:{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)"
        rf"(?:ITD(?![A-Za-z0-9])|internal tandem duplication\b)|"
        rf"internal tandem duplication(?:\s+(?:of|in))?\s+{target})"
        rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}",
        re.I,
    )
    if itd.search(text):
        clauses.append(
            _TargetVariantClause(
                kind="internal_tandem_duplication",
                canonical=f"{target_symbol}-ITD",
                query_phrases=(
                    f"{target_symbol}-ITD",
                    f"{target_symbol} ITD",
                    f"{target_symbol} internal tandem duplication",
                ),
            )
        )

    amino_substitution = re.compile(
        rf"{target}\s*(?:(?:{_VARIANT_SEPARATOR_PATTERN}|:)\s*|\s+)"
        r"(?:p\.)?([A-Za-z][0-9]{1,5}[A-Za-z*])"
        rf"(?![A-Za-z0-9]){_VARIANT_NEGATIVE_SUFFIX_GUARD}",
        re.I,
    )
    for match in amino_substitution.finditer(text):
        variant = match.group(1).upper()
        clauses.append(
            _TargetVariantClause(
                kind="amino_acid_substitution",
                canonical=f"{target_symbol} {variant}",
                query_phrases=(
                    f"{target_symbol} {variant}",
                    f"{target_symbol}-{variant}",
                    f"{target_symbol} p.{variant}",
                ),
            )
        )

    activating = re.compile(
        rf"(?:{target}[^.!?]{{0,40}}\bactivating\s+(?:mutation|variant)s?\b|"
        rf"\bactivating\s+{target}\s+(?:mutation|variant)s?\b|"
        rf"\bactivating\s+(?:mutation|variant)s?[^.!?]{{0,24}}{target})",
        re.I,
    )
    if activating.search(text):
        clauses.append(
            _TargetVariantClause(
                kind="activating_variant",
                canonical=f"{target_symbol} activating mutation",
                query_phrases=(
                    f"{target_symbol} activating mutation",
                    f"{target_symbol} activating variant",
                    f"activating {target_symbol} mutation",
                ),
            )
        )
    rearrangement_or_fusion = re.compile(
        rf"(?:{_target_rearrangement_or_fusion_pattern(target_symbol)}|"
        rf"{target}\s*(?:재배열|융합))",
        re.I,
    )
    if rearrangement_or_fusion.search(text):
        clauses.append(
            _TargetVariantClause(
                kind="gene_rearrangement_or_fusion",
                canonical=f"{target_symbol} rearrangement/fusion",
                query_phrases=(
                    f"{target_symbol} rearrangement",
                    f"{target_symbol} gene rearrangement",
                    f"{target_symbol} gene rearrangements",
                    f"{target_symbol}-rearranged",
                    f"{target_symbol} fusion",
                    f"{target_symbol} gene fusion",
                    f"{target_symbol} gene fusions",
                    f"{target_symbol}-fusion-positive",
                ),
            )
        )
    return tuple(clauses)


def _primary_direction_model_context(
    brief: ResearchBrief,
    target_symbol: str,
) -> _PrimaryDirectionModelContext | None:
    """Derive one exact molecular-model context only from the Brief.

    A free-text disease alias is never replaced by a generic cell-type token.
    This alternative is enabled only when ``target_cell_type`` was explicitly
    configured and the Brief also declares an exact target-linked molecular
    variant.  Model identifiers are deterministic, compact alphanumeric cell
    line names derived from that field, not inferred biology.
    """

    configured_model = _clean_optional_text(
        brief.constraints.target_cell_type
    )
    if configured_model is None:
        return None

    variant_fields = (
        brief.disease,
        brief.research_goal,
        brief.rationale or "",
        configured_model,
        *brief.human_priorities,
    )
    allowed_kinds = {
        "internal_tandem_duplication",
        "amino_acid_substitution",
        "gene_rearrangement_or_fusion",
    }
    variants = tuple(
        {
            (clause.kind, clause.canonical.casefold()): clause
            for value in variant_fields
            for clause in _target_variant_clauses(value, target_symbol)
            if clause.kind in allowed_kinds
        }.values()
    )
    if not variants:
        return None

    # Comparator text belongs to a different experimental arm and must not
    # become a retrieval alias for the requested disease model.
    primary_model_text = re.split(
        r"\b(?:with|versus|vs\.?)\b",
        configured_model,
        maxsplit=1,
        flags=re.I,
    )[0]
    blocked = {
        re.sub(r"[^a-z0-9]+", "", value.casefold())
        for value in (
            target_symbol,
            *(
                phrase
                for clause in variants
                for phrase in (clause.canonical, *clause.query_phrases)
            ),
            *(
                token
                for clause in variants
                for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9]*",
                    clause.canonical,
                )
            ),
        )
    }
    aliases: list[str] = []
    for match in re.finditer(
        r"(?<![A-Za-z0-9])"
        r"([A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)"
        r"(?![A-Za-z0-9])",
        primary_model_text,
    ):
        alias = match.group(1)
        if not re.search(r"\d", alias):
            continue
        normalized = re.sub(r"[^a-z0-9]+", "", alias.casefold())
        if (
            not normalized
            or normalized in blocked
            or normalized.startswith(
                re.sub(r"[^a-z0-9]+", "", target_symbol.casefold())
            )
            or re.match(r"^(?:p?stat|ic|ec)\d", normalized)
        ):
            continue
        aliases.append(alias)
        if len(aliases) == 4:
            break
    model_aliases = tuple(dict.fromkeys(aliases))
    if not model_aliases:
        return None
    return _PrimaryDirectionModelContext(
        model_aliases=model_aliases,
        variant_clauses=variants[:4],
    )


def _association_record_satisfies_requirement(
    record: EvidenceRef,
    requirement: _PrimaryAssociationRequirement,
) -> bool:
    if (
        record.metadata.get("role_specific_record")
        != "primary_target_disease_association_excerpt"
    ):
        return False
    if not requirement.variant_clauses:
        return True
    if record.metadata.get("association_scope") != requirement.required_scope:
        return False
    excerpt = str(record.metadata.get("excerpt") or "")
    exact_clauses = tuple(
        clause
        for clause in requirement.variant_clauses
        if clause.kind
        in {"internal_tandem_duplication", "amino_acid_substitution"}
    )
    required_clauses = exact_clauses or requirement.variant_clauses
    return any(
        _variant_clause_is_association_subject(
            excerpt,
            target_symbol=requirement.target_symbol,
            clause=clause,
            predicate=str(record.metadata.get("association_predicate") or ""),
        )
        for clause in required_clauses
    )


def _primary_association_candidate_rejection_codes(
    record: EvidenceRef,
    *,
    requirement: _PrimaryAssociationRequirement,
    title: str,
    publication_types: Sequence[str],
) -> tuple[str, ...]:
    """Apply the same primary-source and Brief scope gate to every child."""

    codes = list(
        _primary_publication_metadata_rejection_codes(
            title,
            publication_types,
        )
    )
    if requirement.requested and not _association_record_satisfies_requirement(
        record,
        requirement,
    ):
        codes.append("required_subtype_or_variant_not_matched")
    return tuple(dict.fromkeys(codes))


def _excerpt_matches_variant_clause(
    excerpt: str,
    *,
    target_symbol: str,
    clause: _TargetVariantClause,
) -> bool:
    target = rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
    if clause.kind == "internal_tandem_duplication":
        return bool(
            re.search(
                rf"(?:{target}\s*(?:{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)"
                rf"(?:ITD(?![A-Za-z0-9])|internal tandem duplication\b)|"
                rf"internal tandem duplication(?:\s+(?:of|in))?\s+{target})"
                rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}",
                excerpt,
                re.I,
            )
        )
    if clause.kind == "amino_acid_substitution":
        variant = clause.canonical.split()[-1]
        return bool(
            re.search(
                rf"{target}\s*(?:(?:{_VARIANT_SEPARATOR_PATTERN}|:)\s*|\s+)"
                rf"(?:p\.)?"
                rf"{re.escape(variant)}(?![A-Za-z0-9])"
                rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}",
                excerpt,
                re.I,
            )
        )
    if clause.kind == "gene_rearrangement_or_fusion":
        return bool(
            re.search(
                _target_rearrangement_or_fusion_pattern(target_symbol),
                excerpt,
                re.I,
            )
        )
    return bool(
        re.search(
            rf"(?:{target}[^.!?]{{0,40}}\bactivating\s+(?:mutation|variant)s?\b|"
            rf"\bactivating\s+{target}\s+(?:mutation|variant)s?\b)"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}",
            excerpt,
            re.I,
        )
    )


def _has_predeclared_target_model_variant_match(
    text: str,
    *,
    target_symbol: str,
    context: _PrimaryDirectionModelContext | None,
) -> bool:
    """Require target, exact model, and target-linked variant in one sentence."""

    if context is None:
        return False
    if re.search(_target_token_pattern(target_symbol), text, re.I) is None:
        return False
    model_match = any(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            text,
            re.I,
        )
        for alias in context.model_aliases
    )
    if not model_match:
        return False
    return any(
        _excerpt_matches_variant_clause(
            text,
            target_symbol=target_symbol,
            clause=clause,
        )
        for clause in context.variant_clauses
    )


def _primary_direction_context_match_basis(
    text: str,
    *,
    target_symbol: str,
    disease: str,
    context: _PrimaryDirectionModelContext | None,
) -> str | None:
    """Name the exact context contract satisfied by a stored sentence."""

    if _has_direct_target_disease_match(
        text,
        target_symbol=target_symbol,
        disease=disease,
    ):
        return "literal_disease"
    if _has_predeclared_target_model_variant_match(
        text,
        target_symbol=target_symbol,
        context=context,
    ):
        return "predeclared_molecular_model"
    return None


def _variant_clause_subject_pattern(
    *,
    target_symbol: str,
    clause: _TargetVariantClause,
) -> str:
    target = _target_token_pattern(target_symbol)
    if clause.kind == "internal_tandem_duplication":
        return (
            rf"(?:{target}\s*(?:{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)"
            rf"(?:ITD(?![A-Za-z0-9])|internal tandem duplication\b)|"
            rf"internal tandem duplication(?:\s+(?:of|in))?\s+{target})"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}"
        )
    if clause.kind == "amino_acid_substitution":
        variant = clause.canonical.split()[-1]
        return (
            rf"{target}\s*(?:(?:{_VARIANT_SEPARATOR_PATTERN}|:)\s*|\s+)"
            rf"(?:p\.)?"
            rf"{re.escape(variant)}(?![A-Za-z0-9])"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}"
        )
    if clause.kind == "gene_rearrangement_or_fusion":
        return _target_rearrangement_or_fusion_pattern(target_symbol)
    return (
        rf"(?:{target}\s*(?:[- ]\s*)?activating\s+"
        rf"(?:mutation|variant)s?|"
        rf"activating\s+{target}\s+(?:mutation|variant)s?)"
        rf"(?![A-Za-z0-9]){_VARIANT_NEGATIVE_SUFFIX_GUARD}"
    )


def _variant_clause_is_association_subject(
    excerpt: str,
    *,
    target_symbol: str,
    clause: _TargetVariantClause,
    predicate: str,
) -> bool:
    """Bind a required Brief variant to the positive association relation."""

    variant = _variant_clause_subject_pattern(
        target_symbol=target_symbol,
        clause=clause,
    )
    if predicate == "mutation_prevalence":
        return bool(
            _target_prevalence_quantitations(
                excerpt,
                target_symbol=target_symbol,
                variant_pattern=variant,
            )
        )
    if predicate == "disease_target_association":
        return bool(
            re.search(
                rf"(?:\bassociation\s+between\s+{variant}\s+and\b|"
                rf"{variant}(?:\s+(?:mutation|mutations|variant|variants))?"
                rf"\s+(?:is|are|was|were)?\s*(?:associat|correlat)\w*)",
                excerpt,
                re.I,
            )
        )
    if predicate == "disease_target_expression":
        return _variant_is_expression_subject(
            excerpt,
            variant_pattern=variant,
        )
    return False


def _variant_is_expression_subject(
    text: str,
    *,
    variant_pattern: str,
) -> bool:
    """Require the subtype itself to be the grammatical expression subject."""

    comparative = (
        r"(?:higher|lower|elevated|reduced|increased|decreased|"
        r"overexpressed|upregulated)"
    )
    patterns = (
        re.compile(
            rf"(?:{variant_pattern})\s+(?:expression|levels?)\b"
            rf"[^,;.!?]{{0,32}}\b{comparative}\b",
            re.I,
        ),
        re.compile(
            rf"(?:{variant_pattern})\s+(?:is|are|was|were)\s+"
            rf"(?:significantly\s+)?{comparative}\b",
            re.I,
        ),
        re.compile(
            rf"(?:expression|levels?)\s+of\s+(?:{variant_pattern})"
            rf"[^,;.!?]{{0,32}}\b{comparative}\b",
            re.I,
        ),
        re.compile(
            rf"(?:overexpression|upregulation)\s+of\s+"
            rf"(?:{variant_pattern})",
            re.I,
        ),
    )
    return any(
        pattern.search(clause)
        for clause in _association_clauses(text)
        for pattern in patterns
    )


def _verified_primary_publication_metadata(
    title: str,
    publication_types: Sequence[str],
) -> bool:
    return not _primary_publication_metadata_rejection_codes(
        title,
        publication_types,
    )


def _primary_publication_metadata_rejection_codes(
    title: str,
    publication_types: Sequence[str],
) -> tuple[str, ...]:
    codes: list[str] = []
    if not publication_types:
        codes.append("publication_type_missing")
    elif any("review" in value.casefold() for value in publication_types):
        codes.append("review_publication_type")
    elif any(
        _NON_PRIMARY_PUBLICATION_TYPE_PATTERN.search(value)
        for value in publication_types
    ):
        codes.append("non_primary_synthesis_publication_type")
    elif not any(
        _PRIMARY_PUBLICATION_TYPE_PATTERN.search(value)
        for value in publication_types
    ):
        codes.append("primary_publication_type_not_verified")
    if _REVIEW_TITLE_PATTERN.search(title):
        codes.append("review_title")
    return tuple(codes)


def _safe_nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(0, default)
    return max(0, parsed)


def _quote_term(value: str) -> str:
    return " ".join(value.replace('"', " ").split())


def _disease_acronym_anchor_terms(
    disease: str,
    *,
    target_symbol: str | None,
) -> tuple[str, ...]:
    """Return bounded disease-name anchors for ambiguous acronym queries."""

    molecular = _target_molecular_disease_base(
        disease,
        target_symbol=target_symbol,
    )
    source = molecular[0] if molecular is not None else disease
    blocked = {
        "acute",
        "chronic",
        "disease",
        "disorder",
        "syndrome",
        "condition",
        "tumor",
        "tumour",
        "cancer",
        "positive",
        "negative",
        "rearranged",
        "rearrangement",
    }
    target_key = (target_symbol or "").casefold()
    anchors = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", source)
        if token.casefold() not in blocked
        and token.casefold() != target_key
        and token.upper() != token
    ]
    return tuple(dict.fromkeys(anchors))[:3]


def _is_query_acronym(value: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Z][A-Z0-9-]{2,7}", value)
        and any(char.isalpha() for char in value)
    )


def _pubmed_disease_query_clause(
    disease: str,
    *,
    target_symbol: str | None,
    terms: Sequence[str],
    field: str,
) -> str:
    anchors = _disease_acronym_anchor_terms(
        disease,
        target_symbol=target_symbol,
    )
    clauses: list[str] = []
    for term in terms:
        literal = f'"{_quote_term(term)}"[{field}]'
        if _is_query_acronym(term) and anchors:
            anchor_clause = " OR ".join(
                f'"{_quote_term(anchor)}"[{field}]' for anchor in anchors
            )
            clauses.append(f"({literal} AND ({anchor_clause}))")
        else:
            clauses.append(literal)
    return " OR ".join(clauses)


def _europe_pmc_disease_query_clause(
    disease: str,
    *,
    target_symbol: str | None,
    terms: Sequence[str],
) -> str:
    anchors = _disease_acronym_anchor_terms(
        disease,
        target_symbol=target_symbol,
    )
    clauses: list[str] = []
    for term in terms:
        literal = f'TITLE_ABS:"{_quote_term(term)}"'
        if _is_query_acronym(term) and anchors:
            anchor_clause = " OR ".join(
                f'TITLE_ABS:"{_quote_term(anchor)}"' for anchor in anchors
            )
            clauses.append(f"({literal} AND ({anchor_clause}))")
        else:
            clauses.append(literal)
    return " OR ".join(clauses)


def _disease_initialisms(disease: str) -> tuple[str, ...]:
    """Derive a bounded exact initialism such as AML, TNBC, or NSCLC."""

    explicit = tuple(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(
                r"\(([A-Z][A-Z0-9-]{2,7})\)",
                disease,
            )
        )
    )
    if explicit:
        return explicit
    tokens = re.findall(r"[A-Za-z0-9]+", disease)
    if len(tokens) < 2:
        return ()
    initialism = "".join(token[0] for token in tokens if token).upper()
    return (initialism,) if 3 <= len(initialism) <= 8 else ()


def _target_molecular_disease_base(
    disease: str,
    *,
    target_symbol: str | None,
) -> tuple[str, str] | None:
    """Split only an exact target-prefixed molecular disease qualifier."""

    if not target_symbol:
        return None
    target = re.escape(target_symbol.strip())
    match = re.match(
        rf"^\s*{target}\s*(?:[-\u2010-\u2015]\s*|\s+)"
        r"(?P<qualifier>rearranged|rearrangement[- ]positive|"
        r"fusion[- ]positive|fusion|mutant|mutated)\s+"
        r"(?P<base>.+?)\s*$",
        disease,
        re.I,
    )
    if match is None:
        return None
    base = " ".join(match.group("base").split()).strip()
    qualifier = " ".join(match.group("qualifier").split()).casefold()
    return (base, qualifier) if base else None


def _terminal_plural_aliases(value: str) -> tuple[str, ...]:
    normalized = " ".join(value.split()).strip()
    if re.search(r"\btumor$", normalized, re.I):
        return (normalized, normalized + "s")
    if re.search(r"\btumour$", normalized, re.I):
        return (normalized, normalized + "s")
    return (normalized,)


def _disease_query_terms(
    disease: str,
    *,
    target_symbol: str | None = None,
) -> tuple[str, ...]:
    """Return bounded exact disease aliases without dropping molecular scope."""

    terms: list[str] = [disease, *_disease_initialisms(disease)]
    molecular = _target_molecular_disease_base(
        disease,
        target_symbol=target_symbol,
    )
    if molecular is not None:
        base, _ = molecular
        for alias in _terminal_plural_aliases(base):
            terms.append(alias)
        terms.extend(
            initialism
            for initialism in _disease_initialisms(base)
            if initialism.casefold() != (target_symbol or "").casefold()
        )
    return tuple(dict.fromkeys(term for term in terms if term))


def _europe_pmc_publication_types(item: Mapping[str, Any]) -> list[str]:
    """Normalize Europe PMC's nested publication-type shape for audit use."""

    raw: Any = item.get("pubTypeList")
    if isinstance(raw, Mapping):
        raw = raw.get("pubType", [])
    if raw is None:
        raw = item.get("pubType", [])
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [str(value) for value in raw]
    else:
        values = []
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _normalize_pmcid(value: Any) -> str | None:
    """Return one canonical PMCID without guessing or fuzzy correction."""

    normalized = _clean_optional_text(value)
    if normalized is None:
        return None
    normalized = normalized.upper()
    if re.fullmatch(r"PMC[0-9]+", normalized) is None:
        return None
    return normalized


def _ordered_europe_pmc_parent_results(
    results: Sequence[Any],
    *,
    preferred_pmcid_order: Sequence[str] = (),
) -> list[Any]:
    """Return a response-order-independent parent schedule.

    Cross-source PubMed escalation preserves the already ranked PMCID request
    order. Source-local Europe PMC results use canonical identifiers as stable
    tie-breakers, so upstream response permutation cannot change which bounded
    parents receive the full-text budget.
    """

    preferred_rank = {
        normalized: index
        for index, value in enumerate(preferred_pmcid_order)
        if (normalized := _normalize_pmcid(value)) is not None
    }

    def key(item: Any) -> tuple[int, int, str, str, str]:
        if not isinstance(item, Mapping):
            return (2, len(preferred_rank), "", "", repr(type(item)))
        pmcid = _normalize_pmcid(item.get("pmcid")) or ""
        pmid = _clean_optional_text(item.get("pmid")) or ""
        doi = (_clean_optional_text(item.get("doi")) or "").casefold()
        if pmcid in preferred_rank:
            return (0, preferred_rank[pmcid], pmcid, pmid, doi)
        return (1, len(preferred_rank), pmcid, pmid, doi)

    return sorted(results, key=key)


def _europe_pmc_oa_eligibility(
    item: Mapping[str, Any],
) -> tuple[Any | None, str | None]:
    """Build independently attested parser eligibility or one reason code."""

    from drugpilot.tools.europe_pmc_oa_fulltext import OAArticleEligibility

    pmcid = _clean_optional_text(item.get("pmcid"))
    if pmcid is None:
        return None, "pmcid_missing"
    if not _europe_pmc_true(item.get("isOpenAccess")):
        return None, "not_open_access"
    title = _plain_text(str(item.get("title") or ""))
    publication_types = _europe_pmc_publication_types(item)
    metadata_rejections = _primary_publication_metadata_rejection_codes(
        title,
        publication_types,
    )
    if metadata_rejections:
        return None, f"metadata_{metadata_rejections[0]}"
    if any(
        _europe_pmc_true(item.get(key))
        for key in ("isRetracted", "retracted", "is_retracted")
    ):
        return None, "retracted_publication"
    raw_license = _clean_optional_text(
        item.get("license")
        or item.get("licenseUri")
        or item.get("licenseURI")
        or item.get("licenseUrl")
        or item.get("licenseURL")
    )
    if raw_license is None:
        return None, "license_missing"
    canonical_labels = {
        value.casefold(): value
        for value in _EPMC_OA_ALLOWED_LICENSE_URIS
        if value.startswith("CC")
    }
    license_identifier = canonical_labels.get(
        raw_license.casefold(),
        raw_license,
    )
    if license_identifier not in _EPMC_OA_ALLOWED_LICENSE_URIS:
        return None, "license_not_allowed"
    try:
        return (
            OAArticleEligibility(
                pmcid=pmcid,
                pmid=_clean_optional_text(item.get("pmid")),
                publication_types=tuple(publication_types),
                is_open_access=True,
                license_uri=license_identifier,
                is_retracted=False,
                doi=_clean_optional_text(item.get("doi")),
            ),
            None,
        )
    except ValueError:
        return None, "article_identity_invalid"


def _europe_pmc_review_reference_parent(
    item: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Admit an exact, non-retracted review only as a discovery parent.

    Review and meta-analysis rows can lead to independently verified primary
    children through the official Europe PMC references endpoint.  They never
    enter the JATS evidence parser and never inherit or emit an evidence role.
    Both PMCID and PMID are required so the reference edge is bound to the
    exact search-result identity rather than a title or journal heuristic.
    """

    publication_types = _europe_pmc_publication_types(item)
    if not any(
        re.search(r"\b(?:review|meta[- ]?analysis)\b", value, re.I)
        for value in publication_types
    ):
        return None, "not_review_discovery_parent"
    pmcid = _clean_optional_text(item.get("pmcid"))
    pmid = _clean_optional_text(item.get("pmid"))
    if pmcid is None or re.fullmatch(r"PMC[0-9]+", pmcid, re.I) is None:
        return None, "review_parent_pmcid_invalid"
    if pmid is None or re.fullmatch(r"[0-9]+", pmid) is None:
        return None, "review_parent_pmid_invalid"
    status_text = " ".join(
        value
        for value in (
            _plain_text(str(item.get("title") or "")),
            *publication_types,
            _plain_text(str(item.get("publicationStatus") or "")),
        )
        if value
    )
    if any(
        _europe_pmc_true(item.get(key))
        for key in (
            "isRetracted",
            "retracted",
            "is_retracted",
            "isExpressionOfConcern",
            "expressionOfConcern",
            "is_expression_of_concern",
        )
    ) or re.search(
        r"\b(?:expression of concern|retract(?:ed|ion))\b",
        status_text,
        re.I,
    ):
        return None, "review_parent_retracted_or_expression_of_concern"
    return (
        {
            "pmcid": pmcid.upper(),
            "pmid": pmid,
            "publication_types": publication_types,
        },
        None,
    )


def _europe_pmc_true(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().casefold() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _europe_pmc_fulltext_url(pmcid: str) -> str:
    return f"{EUROPE_PMC_REST_URL}/{pmcid}/fullTextXML"


def _europe_pmc_references_url(
    *,
    pmcid: str | None,
    pmid: str | None,
) -> str:
    if pmcid:
        # Europe PMC's official contract is /{source}/{id}/references.
        # Here source=PMC while the accession itself is also PMC-prefixed;
        # /PMC/PMC123... is intentional, not a duplicated path component.
        return f"{EUROPE_PMC_REST_URL}/PMC/{pmcid}/references"
    if pmid:
        return f"{EUROPE_PMC_REST_URL}/MED/{pmid}/references"
    raise ValueError("a parent PMCID or PMID is required")


def _europe_pmc_reference_metadata_query(reference: Any) -> str | None:
    if reference.pmcid:
        return f"PMCID:{reference.pmcid}"
    if reference.pmid:
        return f"EXT_ID:{reference.pmid} AND SRC:MED"
    if reference.doi:
        safe_doi = str(reference.doi).replace('"', "")
        return f'DOI:"{safe_doi}"'
    return None


def _deduplicate_references_preserving_first_lineage(
    references: Sequence[Any],
) -> tuple[tuple[Any, ...], int]:
    """Deduplicate children globally without merging different parents.

    The parser safely merges aliases only within one parent reference list.
    Across parents, merging would invalidate the immutable parent lineage and
    deliberately raises.  The retrieval layer instead keeps the first parent
    in deterministic search/position order and audits every later overlapping
    child as a drop.
    """

    groups: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        identity_keys = set(reference.identity_keys)
        matching = [
            group_index
            for group_index, group in enumerate(groups)
            if identity_keys.intersection(group["identity_keys"])
        ]
        if not matching:
            groups.append(
                {
                    "first_index": index,
                    "reference": reference,
                    "identity_keys": identity_keys,
                }
            )
            continue
        first_index = index
        first_reference = reference
        combined_keys = set(identity_keys)
        for group_index in reversed(matching):
            group = groups.pop(group_index)
            combined_keys.update(group["identity_keys"])
            if group["first_index"] < first_index:
                first_index = group["first_index"]
                first_reference = group["reference"]
        groups.append(
            {
                "first_index": first_index,
                "reference": first_reference,
                "identity_keys": combined_keys,
            }
        )
    groups.sort(key=lambda group: int(group["first_index"]))
    retained = tuple(group["reference"] for group in groups)
    return retained, len(references) - len(retained)


def _exact_europe_pmc_child_metadata(
    raw_payload: bytes,
    reference: Any,
) -> Mapping[str, Any]:
    """Require one independently fetched child row bound to the reference."""

    from drugpilot.tools.europe_pmc_oa_fulltext import (
        EuropePMCOAValidationError,
        document_identity_keys,
    )

    try:
        payload = json.loads(raw_payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EuropePMCOAValidationError(
            "child_metadata_invalid_json",
            "child metadata response is not valid UTF-8 JSON",
        ) from exc
    values = payload.get("resultList", {}).get("result", [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EuropePMCOAValidationError(
            "child_metadata_invalid_shape",
            "resultList.result must be an array",
        )
    matches: list[Mapping[str, Any]] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        try:
            keys = document_identity_keys(
                pmcid=_clean_optional_text(item.get("pmcid")),
                pmid=_clean_optional_text(item.get("pmid")),
                doi=_clean_optional_text(item.get("doi")),
            )
        except ValueError:
            continue
        # A reference may carry several independently useful identifiers.  An
        # exact metadata re-search must agree with every identifier supplied by
        # that edge; accepting a row merely because one key overlaps could bind
        # a PMID to a conflicting PMCID/DOI and trigger the wrong full-text GET.
        reference_keys = set(reference.identity_keys)
        if reference_keys and reference_keys.issubset(keys):
            matches.append(item)
    if len(matches) != 1:
        reason = (
            "child_metadata_not_found"
            if not matches
            else "child_metadata_ambiguous"
        )
        raise EuropePMCOAValidationError(
            reason,
            "exact reference identity must resolve to one metadata row",
        )
    return matches[0]


def _response_payload_bytes(response: Any) -> bytes:
    raw = getattr(response, "content", None)
    if isinstance(raw, bytes):
        return raw
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text.encode("utf-8")
    payload = response.json()
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _response_content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw_value = headers.get("Content-Length")
    except AttributeError:
        return None
    if raw_value is None:
        return None
    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _increment_reason(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _contains_disease_initialism(
    text: str,
    disease: str,
    *,
    excluded_initialisms: Sequence[str] = (),
) -> bool:
    excluded = {value.casefold() for value in excluded_initialisms}
    return any(
        re.search(
            rf"(?<![A-Z0-9]){re.escape(value)}(?![A-Z0-9])",
            text,
        )
        for value in _disease_initialisms(disease)
        if value.casefold() not in excluded
    )


def _plain_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", unescape(value))
    return " ".join(without_tags.split())


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    raise ValueError("PubMed EFetch returned no XML response body")


def _bounded_http_call_provenance(
    response: Any,
    *,
    endpoint: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a secret-free hash attestation for one completed public call."""

    request = _bounded_http_request_provenance(
        endpoint=endpoint,
        params=params,
    )
    raw_content = getattr(response, "content", None)
    if not isinstance(raw_content, bytes):
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            raw_content = text.encode("utf-8")
        else:
            raw_content = json.dumps(
                response.json(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
    return {
        **request,
        "status": "completed",
        "status_code": int(getattr(response, "status_code", 0) or 0),
        "response_sha256": sha256(raw_content).hexdigest(),
        "fixture": False,
    }


def _bounded_http_request_provenance(
    *,
    endpoint: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash request parameters while redacting values of secret-like keys."""

    sensitive_markers = (
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "password",
        "secret",
        "token",
    )
    clean = {
        str(key): (
            "[REDACTED]"
            if any(marker in str(key).casefold() for marker in sensitive_markers)
            else value
        )
        for key, value in params.items()
    }
    parameter_payload = json.dumps(
        {"params": clean},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "method": "GET",
        "endpoint": endpoint,
        "parameter_keys": sorted(f"params.{key}" for key in clean),
        "parameter_sha256": sha256(parameter_payload).hexdigest(),
    }


def _bounded_http_failure_provenance(
    *,
    endpoint: str,
    params: Mapping[str, Any],
    error: Exception,
    response: Any | None,
) -> dict[str, Any]:
    """Return one body-free, message-free provenance row for a failed call."""

    request = _bounded_http_request_provenance(
        endpoint=endpoint,
        params=params,
    )
    provenance = {
        "method": request["method"],
        "endpoint": request["endpoint"],
        "parameter_sha256": request["parameter_sha256"],
        "status": "error",
        "error_type": type(error).__name__,
    }
    status_code = int(getattr(response, "status_code", 0) or 0)
    if status_code:
        provenance["status_code"] = status_code
    return provenance


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized or None


def _summary_article_id(item: Mapping[str, Any], identifier_type: str) -> str | None:
    expected = identifier_type.casefold()
    for identifier in item.get("articleids", []):
        if not isinstance(identifier, Mapping):
            continue
        id_type = str(
            identifier.get("idtype")
            or identifier.get("IdType")
            or identifier.get("type")
            or ""
        ).casefold()
        if id_type != expected:
            continue
        return _clean_optional_text(
            identifier.get("value") or identifier.get("id")
        )
    return None


def _validate_pubmed_search_payload(
    payload: Any,
    *,
    raw_document_budget: int,
) -> tuple[int, list[str]]:
    """Validate one exact PubMed ESearch JSON window without coercive defaults."""

    if not isinstance(payload, Mapping):
        raise _PubMedSearchValidationError("response_not_mapping")
    result = payload.get("esearchresult")
    if not isinstance(result, Mapping):
        raise _PubMedSearchValidationError("esearchresult_not_mapping")
    raw_ids = result.get("idlist")
    if (
        not isinstance(raw_ids, Sequence)
        or isinstance(raw_ids, (str, bytes, bytearray))
    ):
        raise _PubMedSearchValidationError("idlist_not_sequence")
    ids: list[str] = []
    for value in raw_ids:
        if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
            raise _PubMedSearchValidationError("pmid_not_numeric_nonempty")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise _PubMedSearchValidationError("duplicate_pmid")
    if len(ids) > raw_document_budget:
        raise _PubMedSearchValidationError("idlist_exceeds_raw_document_budget")
    raw_count = result.get("count")
    if isinstance(raw_count, bool):
        raise _PubMedSearchValidationError("count_not_strict_nonnegative_int")
    if isinstance(raw_count, int):
        count = raw_count
    elif isinstance(raw_count, str) and re.fullmatch(r"0|[1-9][0-9]*", raw_count):
        count = int(raw_count)
    else:
        raise _PubMedSearchValidationError("count_not_strict_nonnegative_int")
    if count < 0:
        raise _PubMedSearchValidationError("count_not_strict_nonnegative_int")
    if (count == 0) != (len(ids) == 0):
        raise _PubMedSearchValidationError("count_zero_idlist_consistency")
    if count < len(ids):
        raise _PubMedSearchValidationError("count_less_than_idlist")
    if len(ids) != min(count, raw_document_budget):
        raise _PubMedSearchValidationError("returned_id_window_incomplete")
    return count, ids


def _validate_pubmed_batch_ids(
    requested_ids: Sequence[str],
    returned_ids: Sequence[Any],
) -> tuple[str, ...]:
    """Require one complete, duplicate-free PubMed batch in request order."""

    requested = tuple(str(value).strip() for value in requested_ids)
    returned = tuple(str(value).strip() for value in returned_ids)
    if len(returned) != len(requested):
        raise _PubMedBatchValidationError("returned_id_count_mismatch")
    if len(set(returned)) != len(returned):
        raise _PubMedBatchValidationError("duplicate_returned_id")
    if set(returned) != set(requested):
        raise _PubMedBatchValidationError("returned_id_membership_mismatch")
    if returned != requested:
        raise _PubMedBatchValidationError("returned_id_order_mismatch")
    return returned


def _validate_pubmed_summary_batch(
    payload: Any,
    requested_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """Validate ESummary identity before any metadata is admitted downstream."""

    if not isinstance(payload, Mapping):
        raise _PubMedBatchValidationError("result_not_mapping")
    raw_uids = payload.get("uids")
    if (
        not isinstance(raw_uids, Sequence)
        or isinstance(raw_uids, (str, bytes, bytearray))
    ):
        raise _PubMedBatchValidationError("uids_not_sequence")
    returned = _validate_pubmed_batch_ids(requested_ids, raw_uids)
    validated: dict[str, Mapping[str, Any]] = {}
    for identifier in returned:
        item = payload.get(identifier)
        if not isinstance(item, Mapping):
            raise _PubMedBatchValidationError("summary_metadata_missing")
        item_uid = _clean_optional_text(item.get("uid"))
        if item_uid != identifier:
            raise _PubMedBatchValidationError("summary_metadata_uid_mismatch")
        validated[identifier] = item
    return validated


def _xml_local_name(element: ElementTree.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _pubmed_document_nodes(
    root: ElementTree.Element,
) -> tuple[ElementTree.Element, ...]:
    """Return direct Article/Book children in source document order."""

    allowed = {"PubmedArticle", "PubmedBookArticle"}
    return tuple(child for child in root if _xml_local_name(child) in allowed)


def _first_descendant_by_local_name(
    root: ElementTree.Element,
    local_name: str,
) -> ElementTree.Element | None:
    return next(
        (element for element in root.iter() if _xml_local_name(element) == local_name),
        None,
    )


def _pubmed_publication_article_ids(
    article: ElementTree.Element,
) -> dict[str, str]:
    """Return identifiers from the publication's own ArticleIdList only.

    ``PubmedData`` can also contain a later ``ReferenceList`` whose cited
    papers have nested ``ArticleId`` elements.  Walking every descendant lets
    the last cited DOI or PMCID overwrite the current publication identity.
    Restricting traversal to the direct PubmedData/ArticleIdList path prevents
    that cross-attachment.  Conflicting duplicate identifiers fail closed.
    """

    values: dict[str, set[str]] = {}
    for data in article:
        if _xml_local_name(data) not in {"PubmedData", "PubmedBookData"}:
            continue
        for identifier_list in data:
            if _xml_local_name(identifier_list) != "ArticleIdList":
                continue
            for identifier in identifier_list:
                if _xml_local_name(identifier) != "ArticleId":
                    continue
                id_type = str(identifier.attrib.get("IdType") or "").casefold()
                value = _clean_optional_text("".join(identifier.itertext()))
                if id_type and value:
                    values.setdefault(id_type, set()).add(value)
    return {
        id_type: next(iter(observed))
        for id_type, observed in values.items()
        if len(observed) == 1
    }


def _pubmed_xml_record_ids(xml_text: str) -> tuple[str, ...]:
    """Return EFetch record identifiers without retaining the response body."""

    if not xml_text.strip():
        return ()
    root = ElementTree.fromstring(xml_text)
    returned: list[str] = []
    for article in _pubmed_document_nodes(root):
        pmid_node = _first_descendant_by_local_name(article, "PMID")
        pmid = _clean_optional_text(
            "".join(pmid_node.itertext()) if pmid_node is not None else None
        )
        if not pmid:
            raise _PubMedBatchValidationError("fetch_record_id_missing")
        returned.append(pmid)
    return tuple(returned)


def _parse_pubmed_content(
    xml_text: str,
    *,
    target_symbol: str,
    disease: str,
    intervention_direction: str | None = None,
    direction_model_context: _PrimaryDirectionModelContext | None = None,
) -> dict[str, _PublicationContent]:
    if not xml_text.strip():
        return {}
    root = ElementTree.fromstring(xml_text)
    records: dict[str, _PublicationContent] = {}
    for article in _pubmed_document_nodes(root):
        pmid_node = _first_descendant_by_local_name(article, "PMID")
        pmid = _clean_optional_text(
            "".join(pmid_node.itertext()) if pmid_node is not None else None
        )
        if not pmid:
            continue

        sections: list[tuple[str, str]] = []
        abstract_elements = (
            element
            for element in article.iter()
            if _xml_local_name(element) == "AbstractText"
        )
        for index, element in enumerate(abstract_elements, 1):
            text = _plain_text("".join(element.itertext()))
            if not text:
                continue
            label = _clean_optional_text(
                element.attrib.get("Label") or element.attrib.get("NlmCategory")
            ) or f"section-{index}"
            sections.append((label, text))

        article_ids = _pubmed_publication_article_ids(article)
        doi = article_ids.get("doi")
        pmcid = article_ids.get("pmc") or article_ids.get("pmcid")

        records[pmid] = _publication_content_from_sections(
            sections,
            target_symbol=target_symbol,
            disease=disease,
            intervention_direction=intervention_direction,
            direction_model_context=direction_model_context,
            doi=doi,
            pmcid=pmcid,
        )
    return records


def _publication_content_from_sections(
    sections: Sequence[tuple[str, str]],
    *,
    target_symbol: str,
    disease: str,
    intervention_direction: str | None = None,
    direction_model_context: _PrimaryDirectionModelContext | None = None,
    doi: str | None = None,
    pmcid: str | None = None,
) -> _PublicationContent:
    normalized_sections = [
        (_plain_text(label), _plain_text(text))
        for label, text in sections
        if _plain_text(text)
    ]
    if not normalized_sections:
        return _PublicationContent(
            doi=doi,
            pmcid=pmcid,
            excerpt_selection_status="abstract_not_available",
        )

    full_abstract = " ".join(text for _, text in normalized_sections)
    context = abstract_context(full_abstract)
    target_lexical_match = bool(
        re.search(_target_token_pattern(target_symbol), full_abstract)
    )
    selected = _bounded_relevant_excerpt(
        normalized_sections,
        target_symbol=target_symbol,
        disease=disease,
        intervention_direction=intervention_direction,
        direction_model_context=direction_model_context,
    )
    paired_direction_selected = False
    if selected is None and intervention_direction == "inhibit":
        selected = _bounded_compound_linked_direction_excerpt(
            normalized_sections,
            target_symbol=target_symbol,
            disease=disease,
        )
        paired_direction_selected = selected is not None
    association = _bounded_primary_association_excerpt(
        normalized_sections,
        target_symbol=target_symbol,
        disease=disease,
    )
    association_updates: dict[str, Any] = {}
    if association is not None:
        (
            association_excerpt,
            association_location,
            association_truncated,
            association_predicate,
            association_scope,
            association_score,
        ) = association
        association_updates = {
            "association_excerpt": association_excerpt,
            "association_source_location": association_location,
            "association_excerpt_truncated": association_truncated,
            "association_predicate": association_predicate,
            "association_scope": association_scope,
            "association_score": association_score,
        }
    if selected is None:
        return _PublicationContent(
            content_hash=sha256(full_abstract.encode("utf-8")).hexdigest(),
            interpretation_context=context,
            doi=doi,
            pmcid=pmcid,
            excerpt_selection_status="related_excerpt_not_found",
            target_lexical_match=target_lexical_match,
            **association_updates,
        )
    excerpt, location, truncated = selected
    counter_signal = _classify_counter_signal(
        "",
        excerpt,
        target_symbol,
        intervention_direction=intervention_direction,
    )
    counter_category = counter_signal[0] if counter_signal else None
    counter_text = counter_signal[1] if counter_signal else None
    direct_support_rejections = _direct_support_rejection_codes(
        excerpt,
        target_symbol,
    )
    return _PublicationContent(
        excerpt=excerpt,
        source_location=location,
        content_hash=sha256(full_abstract.encode("utf-8")).hexdigest(),
        interpretation_context=context,
        doi=doi,
        pmcid=pmcid,
        excerpt_truncated=truncated,
        excerpt_selection_status=(
            "selected_counter_signal"
            if counter_signal
            else "selected_compound_linked_document_pair"
            if paired_direction_selected
            else "selected"
        ),
        counter_signal_category=counter_category,
        counter_signal=counter_text,
        direct_support_rejection_codes=direct_support_rejections,
        target_lexical_match=target_lexical_match,
        **association_updates,
    )


def _fusion_query_terms(seed: TargetSeed) -> tuple[str, ...]:
    legacy=validated_fusion_search_terms(seed.discovery_metadata[SEARCH_KEY],seed.symbol) if SEARCH_KEY in seed.discovery_metadata else ()
    variants=variant_query_terms(seed.discovery_metadata[VARIANT_SEARCH_KEY],seed.symbol) if VARIANT_SEARCH_KEY in seed.discovery_metadata else ()
    return tuple(dict.fromkeys((*legacy,*variants)))


def _g0_direction_query(public,brief,seed):
    precise=public._pubmed_primary_direction_query(brief,seed)
    bindings=seed.discovery_metadata.get(VARIANT_SEARCH_KEY,[])
    terms=variant_query_terms(bindings,seed.symbol)
    if not terms:
        return precise
    scope=' OR '.join('"'+_quote_term(term)+'"[Title/Abstract]' for term in terms)
    disease=_pubmed_disease_query_clause(brief.disease,target_symbol=seed.symbol,
        terms=_disease_query_terms(brief.disease,target_symbol=seed.symbol),field="Title/Abstract")
    context=_primary_direction_model_context(brief,seed.symbol)
    if context and context.model_aliases:
        models=' OR '.join('"'+_quote_term(term)+'"[Title/Abstract]' for term in context.model_aliases)
        disease='('+disease+') OR ('+models+')'
    action=' OR '.join('"'+term+'"[Title/Abstract]' for term in _direction_aware_query_terms(brief.constraints.intervention_direction))
    endpoint=' OR '.join('"'+term+'"[Title/Abstract]' for term in _MECHANISTIC_QUERY_TERMS)
    # Query vocabulary may be separated in a paper. Retrieval still requires
    # the unchanged source-scoped intervention predicate in the selected
    # result; this broader branch never grants evidence/construct identity.
    # Factor common clauses instead of concatenating two long GET queries
    # (which can cause HTTP 414). This is broad retrieval, strict selection.
    return f'({scope}) AND ({disease}) AND ({action}) AND ({endpoint}) NOT (Review[Publication Type] OR Editorial[Publication Type] OR Guideline[Publication Type])'


def _abstract_context_metadata(content: _PublicationContent, *, source_review: bool = False) -> dict:
    if content.interpretation_context is None or not (
        source_review or content.excerpt or content.association_excerpt
    ):
        return {}
    return {ABSTRACT_CONTEXT_KEY: deepcopy(content.interpretation_context),
        "abstract_context_stored": content.interpretation_context["status"] == "complete",
        "abstract_context_storage_policy": "complete_within_size_budget_otherwise_omitted_no_full_paper"}


def _publication_has_association_candidate(
    content: _PublicationContent,
) -> bool:
    """True only when the parent excerpt itself is the association sentence.

    A single primary paper may contain both a mechanistic result sentence and
    a separate target-disease association sentence.  The latter is emitted as
    its own role-specific record and must not demote the independently selected
    mechanistic parent excerpt to ``association_role_only``.
    """

    if not content.excerpt or not content.association_excerpt:
        return False
    return " ".join(content.excerpt.split()).casefold() == " ".join(
        content.association_excerpt.split()
    ).casefold()


def _direct_target_intervention_pattern(
    target_symbol: str,
    direction: str,
) -> re.Pattern[str]:
    target = _target_token_pattern(target_symbol) + r"(?![- ](?:signall?ing|pathways?|cascades?|mediated)\b)"
    if direction == "inhibit":
        descriptor = r"(?:(?:(?:oncogenic|mutant|mutated|wild[- ]type)\s+)?(?:(?:tyrosine\s+)?kinase|protein|receptor|fusion(?:\s+protein)?)\s+)?"
        target_inhibitor = (
            rf"{target}(?:[- ](?:selective|specific))?[- ]"
            rf"(?:(?:tyrosine[- ]kinase|kinase)[- ])?inhibitors?"
        )
        relationship = (
            rf"(?:\b(?:inhibit(?:ed|s|ing)|suppress(?:ed|es|ing)|"
            rf"block(?:ed|s|ing))\s+(?:the\s+)?{descriptor}{target}"
            rf"(?![- ]mediated)"
            rf"(?:[- ]kinase|\s+(?:kinase(?:\s+activity)?|activity|"
            rf"phosphorylation))?|"
            rf"\b(?:inhibition|suppression|blockade)\s+of\s+(?:the\s+)?{descriptor}{target}|"
            rf"\b(?:knock[- ]?down|silencing)\s+of\s+(?:the\s+)?{target}|"
            rf"{target}[- ](?:knock[- ]?down|silencing)\b|"
            rf"\b(?:silenc(?:e|ed|ing)|knocked\s+down)\s+{target}|"
            rf"{target}\s+(?:was\s+)?(?:silenced|knocked\s+down)\b|"
            rf"{target}\s+(?:kinase\s+)?(?:inhibition|suppression|blockade)\b|"
            rf"{target}\s+(?:kinase(?:\s+activity)?|activity|phosphorylation)\s+"
            rf"(?:was\s+)?(?:inhibit(?:ed)?|suppress(?:ed)?|block(?:ed)?)|"
            rf"\b(?:directly\s+)?engag(?:e|ed|es|ing)\s+(?:with\s+)?{target}|"
            rf"{target}\s+(?:was\s+)?(?:directly\s+)?engag(?:e|ed)|"
            rf"\bbound(?:\s+directly)?\s+(?:to\s+)?[^.!?]{{0,48}}{target}|"
            rf"{target_inhibitor}\b)"
        )
    elif direction == "activate":
        relationship = (
            rf"(?:\b(?:activat(?:e|ed|es|ing)|stimulat(?:e|ed|es|ing))\s+"
            rf"(?:the\s+)?{target}\b|"
            rf"\b(?:activation|stimulation|overexpression)\s+of\s+{target}\b|"
            rf"\boverexpress(?:ed|ing)?\s+{target}\b|"
            rf"{target}\s+(?:was\s+)?overexpressed\b|"
            rf"{target}\s+(?:activation|stimulation|overexpression)\b|"
            rf"{target}(?:[- ](?:selective|specific))?[- ]agonists?\b)"
        )
    else:
        relationship = (
            rf"(?:\b(?:modulat(?:e|ed|es|ing)|regulat(?:e|ed|es|ing))\s+"
            rf"(?:the\s+)?{target}\b|"
            rf"\b(?:modulation|regulation)\s+of\s+{target}\b|"
            rf"{target}\s+(?:modulation|regulation)\b)"
        )
    if direction in {"modulate", "unspecified"}:
        relationship = "(?:" + "|".join((relationship,
            _direct_target_intervention_pattern(target_symbol, "inhibit").pattern,
            _direct_target_intervention_pattern(target_symbol, "activate").pattern)) + ")"
    return re.compile(relationship, re.I)


def _observed_direct_intervention_present(excerpt: str, target: str, direction: str) -> bool:
    """Literal action on the source-named entity, not on an isolated fusion partner.

    Scope annotation/admission still prevents promoting the scoped entity to a
    general gene claim. This only corrects the missing-action reason code.
    """
    from drugpilot.agents.molecular_scope import mentions
    entities = [target, *(m["quote"] for m in mentions(excerpt, target))]
    return any(_direct_target_intervention_pattern(entity, direction).search(excerpt)
               for entity in entities)


def _has_admissible_direction_endpoint(
    excerpt: str,
    target_symbol: str,
) -> bool:
    """Reject downstream-only signaling readouts as direct target support."""

    if re.search(
        r"\b(?:viabilit\w*|proliferat\w*|apopto\w*|cytotoxi\w*|"
        r"sensitiv\w*|respons(?:e|es|ive)|IC50|EC50|Ki|Kd|potenc\w*|"
        r"tumou?r (?:growth|burden)|cell survival|differentiation|"
        r"oncometabolite(?: levels?)?|metabolite levels?|"
        r"2[- ]?(?:HG|hydroxyglutarate)|target[- ]engagement|bound to)\b",
        excerpt,
        re.I,
    ):
        return True

    target = _target_token_pattern(target_symbol)
    return re.search(
        rf"(?:{target}\s+(?:(?:kinase|enzymatic)\s+)?"
        rf"(?:activity|signaling|phosphorylat\w*)\b|"
        rf"{target}\s+(?:was\s+)?phosphorylat\w*\b|"
        rf"\b(?:phosphorylat\w*|(?:kinase|enzymatic) activity|signaling)"
        rf"\s+of\s+(?:the\s+)?{target})",
        excerpt,
        re.I,
    ) is not None


def _direction_endpoint_polarity_status(
    excerpt: str,
    intervention_direction: str,
) -> str:
    """Classify only explicit, favorable endpoint changes as positive results."""

    decrease = (
        r"(?:reduc\w*|decreas\w*|suppress\w*|impair\w*|"
        r"inhibit(?:ed|s|ing)?|"
        r"attenuat\w*|abolish\w*|block\w*|lower\w*)"
    )
    increase = (
        r"(?:increas\w*|enhanc\w*|elevat\w*|induc\w*|promot\w*|"
        r"stimulat\w*)"
    )
    adverse_growth = (
        r"(?:viabilit\w*|proliferat\w*|tumou?r\s+(?:growth|burden)|"
        r"cell\s+survival)"
    )
    favorable_gain = r"(?:apopto\w*|cytotoxi\w*|sensitiv\w*)"
    signal = (
        r"(?:phosphorylat\w*|(?:kinase|enzymatic)\s+activity|"
        r"target[- ]engagement|signaling)"
    )

    def related(effect: str, endpoint: str) -> bool:
        # Keep an effect bound to its local coordinated clause.  Without this
        # boundary, ``suppressed signaling and induced apoptosis`` incorrectly
        # binds ``suppressed`` to apoptosis and is classified as harmful.
        clauses = re.split(
            r"[;,.]|\b(?:and|but|whereas|while)\b",
            excerpt,
            flags=re.I,
        )
        return any(
            re.search(
                rf"(?:\b{effect}\b[^.!?]{{0,48}}\b{endpoint}\b|"
                rf"\b{endpoint}\b[^.!?]{{0,48}}\b{effect}\b)",
                clause,
                re.I,
            )
            is not None
            for clause in clauses
        )

    def affirmed_compound_effect(pattern: str) -> bool:
        """Accept explicit endpoint-effect compounds unless locally negated."""

        for match in re.finditer(pattern, excerpt, re.I):
            prefix = excerpt[max(0, match.start() - 56) : match.start()]
            if re.search(
                r"\b(?:no|not|without|failed\s+to|did\s+not|"
                r"lack(?:ed|s|ing)?(?:\s+of)?)\b[^.!?]{0,40}$",
                prefix,
                re.I,
            ):
                continue
            return True
        return False

    unfavorable = related(increase, adverse_growth) or related(
        decrease,
        favorable_gain,
    )
    if intervention_direction == "inhibit":
        unfavorable = unfavorable or related(increase, signal)
    elif intervention_direction == "activate":
        unfavorable = unfavorable or related(decrease, signal)
    if unfavorable:
        return "unfavorable"

    favorable = related(decrease, adverse_growth) or related(
        increase,
        favorable_gain,
    )
    if intervention_direction == "inhibit":
        favorable = favorable or related(decrease, signal)
    elif intervention_direction == "activate":
        favorable = favorable or related(increase, signal)
    favorable = favorable or bool(
        re.search(
            r"\b(?:show(?:ed|s|ing)|achiev(?:e|ed|es|ing)|"
            r"demonstrat(?:e|ed|es|ing)|had)\b[^.!?]{0,96}"
            r"\b(?:objective|complete|partial|durable|drastic|clinical\s+)?"
            r"respons(?:e|es)\b|"
            r"\b(?:tumou?r\s+regression|dose[- ]dependent\s+"
            r"(?:response|cytotoxicity|growth\s+inhibition))\b|"
            r"\b(?:IC50|EC50|Ki|Kd)\s*(?:of|=|was|were|:)?\s*"
            r"[<>\u2264\u2265]?\s*\d",
            excerpt,
            re.I,
        )
    )
    favorable = favorable or affirmed_compound_effect(
        r"\b(?:result(?:ed|ing)\s+in|caus(?:e|ed|es|ing))\s+"
        r"(?:apopto\w*|cytotoxi\w*|differentiation)\b"
    )
    if intervention_direction == "inhibit":
        favorable = favorable or affirmed_compound_effect(
            r"\b(?:anti[- ]?proliferative(?:\s+(?:effect|effects|activity))?|"
            r"growth[- ]inhibitory(?:\s+(?:effect|effects|activity))?|"
            r"pro[- ]?apoptotic(?:\s+(?:effect|effects|activity))?|"
            r"(?:potent\s+)?inducers?\s+of\s+apoptosis)\b"
        )
    return "favorable" if favorable else "ambiguous"


def _primary_direction_content_rejection_codes(
    content: _PublicationContent,
    *,
    title: str,
    publication_types: Sequence[str],
    target_symbol: str,
    disease: str,
    intervention_direction: str,
    direction_model_context: _PrimaryDirectionModelContext | None = None,
    require_primary_result_context: bool = False,
) -> tuple[str, ...]:
    """Mirror MA1's direct-primary contract before admitting lane evidence."""

    codes = list(
        _primary_publication_metadata_rejection_codes(
            title,
            publication_types,
        )
    )
    excerpt = content.excerpt
    if not excerpt or not content.source_location or not content.content_hash:
        codes.append("abstract_excerpt_unavailable")
        return tuple(dict.fromkeys(codes))
    if content.counter_signal_category:
        codes.append("counter_signal")
    codes.extend(content.direct_support_rejection_codes)
    if _publication_has_association_candidate(content):
        codes.append("association_role_only")
    if require_primary_result_context and not _primary_direction_result_context_verified(
        content,
        title=title,
        target_symbol=target_symbol,
        intervention_direction=intervention_direction,
    ):
        codes.append("primary_result_context_missing")
    if _primary_direction_context_match_basis(
        excerpt,
        target_symbol=target_symbol,
        disease=disease,
        context=direction_model_context,
    ) is None:
        codes.append("exact_target_base_disease_match_missing")
    direction_pattern = _DIRECTION_EXCERPT_PATTERNS.get(
        intervention_direction,
        _DIRECTION_EXCERPT_PATTERNS["modulate"],
    )
    if direction_pattern.search(excerpt) is None:
        codes.append("requested_direction_missing")
    if not _observed_direct_intervention_present(excerpt, target_symbol, intervention_direction):
        codes.append("direct_target_intervention_missing")
    if not _has_admissible_direction_endpoint(excerpt, target_symbol):
        codes.append("downstream_only_endpoint")
    if _MECHANISTIC_ENDPOINT_PATTERN.search(excerpt) is None:
        codes.append("measured_endpoint_missing")
    if _EXPERIMENTAL_RESULT_PATTERN.search(excerpt) is None:
        codes.append("positive_result_language_missing")
    polarity = _direction_endpoint_polarity_status(
        excerpt,
        intervention_direction,
    )
    if polarity == "unfavorable":
        codes.append("unfavorable_endpoint_polarity")
    elif polarity != "favorable":
        codes.append("favorable_endpoint_polarity_missing")
    if _NON_DIRECTION_PRIMARY_PATTERN.search(excerpt):
        codes.append("non_direction_or_negative_context")
    return tuple(dict.fromkeys(codes))


def _primary_direction_result_context_verified(
    content: _PublicationContent,
    *,
    title: str,
    target_symbol: str,
    intervention_direction: str,
) -> bool:
    """Reject background-only statements that are not this paper's result."""

    excerpt = content.excerpt or ""
    location = content.source_location or ""
    if not excerpt:
        return False
    if content.excerpt_selection_status == "selected_compound_linked_document_pair":
        return True
    if re.search(
        r"abstract:[^|]*(?:RESULT|CONCLUSION|FINDING)",
        location,
        re.I,
    ):
        return True
    if re.search(
        r"\b(?:we\s+(?:show|showed|found|demonstrate|demonstrated|observed|"
        r"report|reported)|this\s+study\s+(?:shows?|showed|demonstrates?|"
        r"demonstrated|found)|our\s+(?:study|results?|findings?))\b",
        excerpt,
        re.I,
    ):
        return True
    return bool(
        _direct_target_intervention_pattern(
            target_symbol,
            intervention_direction,
        ).search(title)
        and _MECHANISTIC_ENDPOINT_PATTERN.search(excerpt)
        and _EXPERIMENTAL_RESULT_PATTERN.search(excerpt)
    )


def _verified_primary_direction_content(
    content: _PublicationContent,
    *,
    title: str,
    publication_types: Sequence[str],
    target_symbol: str,
    disease: str,
    intervention_direction: str,
    direction_model_context: _PrimaryDirectionModelContext | None = None,
) -> bool:
    return not _primary_direction_content_rejection_codes(
        content,
        title=title,
        publication_types=publication_types,
        target_symbol=target_symbol,
        disease=disease,
        intervention_direction=intervention_direction,
        direction_model_context=direction_model_context,
    )


def _publication_evidence_role(
    content: _PublicationContent,
    *,
    counter_evidence_enabled: bool,
    direct_support_eligible: bool,
) -> str:
    if not content.excerpt:
        return "context"
    if content.counter_signal_category:
        return "counter" if counter_evidence_enabled else "context"
    return "support" if direct_support_eligible else "context"


def _publication_evidence_scope(
    content: _PublicationContent,
    *,
    direct_support_eligible: bool,
) -> str:
    if not content.excerpt:
        return "publication metadata only; not support evidence"
    if content.counter_signal_category:
        return (
            "target-disease abstract limitation/counter signal; researcher review "
            "required and not support evidence"
        )
    if content.direct_support_rejection_codes:
        return (
            "target-disease combination, response-modifier, multi-target, or "
            "other-intervention context; not direct target-direction support"
        )
    if _publication_has_association_candidate(content):
        return (
            "primary target-disease association/prevalence excerpt; "
            "association role only and not intervention-direction proof"
        )
    if not direct_support_eligible:
        return (
            "target-disease or molecular-model context; explicit direct target "
            "action was not verified in the bounded excerpt"
        )
    return "target-disease abstract excerpt; not causal proof"


def _direct_support_rejection_codes(
    excerpt: str,
    target_symbol: str,
) -> tuple[str, ...]:
    """Fail closed when the stored sentence does not isolate the target action."""

    codes: list[str] = []
    target = rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
    other_target = (
        rf"(?!{re.escape(target_symbol)}\b)(?!ITD\b)"
        r"[A-Za-z][A-Za-z0-9-]{1,20}"
    )
    if re.search(
        rf"(?:{target}\s*/\s*{other_target}|"
        rf"{other_target}\s*/\s*{target})",
        excerpt,
        re.I,
    ):
        codes.append("multi_target_slash_intervention")
    if re.search(
        rf"(?i:{_INTERVENTION_NOUN})(?:\s+of)?\s+{target}\s+"
        r"(?:and|plus)\s+[A-Z][A-Z0-9-]{1,20}",
        excerpt,
    ):
        codes.append("multi_target_co_intervention")
    target_inhibitor = (
        rf"{target}(?:[- ](?:selective|specific))?[- ]"
        rf"(?:(?:tyrosine[- ]kinase|kinase)[- ])?inhibitors?"
    )
    target_inhibitor_response_spans = [
        match.span()
        for match in re.finditer(
            rf"\b(?:response|sensitivity)\s+to\s+"
            rf"(?:(?:a|an|the|one|two|three|both|multiple)\s+)*"
            rf"{target_inhibitor}\b",
            excerpt,
            re.I,
        )
    ]
    if re.search(
        rf"\b(?:enhanc(?:e|ed|es|ing|ement)|"
        rf"potentiat(?:e|ed|es|ing|ion)|"
        rf"sensiti[sz](?:e|ed|es|ing|ation)|"
        rf"resensiti[sz](?:e|ed|es|ing|ation)|"
        rf"synerg(?:y|istic|istically))\b[^.!?]{{0,96}}"
        rf"\b{target_inhibitor}\b|"
        rf"\b{target_inhibitor}\b[^.!?]{{0,96}}"
        rf"\b(?:enhanc(?:e|ed|es|ing|ement)|"
        rf"potentiat(?:e|ed|es|ing|ion)|"
        rf"sensiti[sz](?:e|ed|es|ing|ation)|"
        rf"resensiti[sz](?:e|ed|es|ing|ation)|"
        rf"synerg(?:y|istic|istically))\b",
        excerpt,
        re.I,
    ):
        codes.append("target_inhibitor_response_modifier")
    if re.search(
        rf"\b(?!{target_symbol}\b)(?!inhibition\b)(?!treatment\b)"
        rf"[A-Za-z][A-Za-z0-9-]{{2,30}}\s+"
        rf"(?:enhanc(?:e|ed|es|ing)|potentiat(?:e|ed|es|ing)|"
        rf"sensiti[sz](?:e|ed|es|ing))\b[^.!?]{{0,160}}"
        rf"\b{target}\s+(?:inhibition|blockade|suppression)\b",
        excerpt,
        re.I,
    ):
        codes.append("other_intervention_response_modifier")
    for code, pattern in _INDIRECT_INTERVENTION_CONTEXT_PATTERNS:
        matches = list(pattern.finditer(excerpt))
        if code == "other_drug_response" and matches:
            matches = [
                match
                for match in matches
                if not any(
                    start <= match.start() < end
                    for start, end in target_inhibitor_response_spans
                )
            ]
        if matches:
            codes.append(code)

    gene_token = r"[A-Z][A-Z0-9-]{1,20}"
    subject_matches = [
        match
        for pattern in (
            re.compile(
                rf"(?i:{_INTERVENTION_NOUN})\s+of\s+"
                rf"(?P<subjects>{gene_token}(?:\s*/\s*{gene_token})*)"
            ),
            re.compile(
                rf"(?P<subjects>{gene_token}(?:\s*/\s*{gene_token})*)\s+"
                rf"(?i:{_INTERVENTION_NOUN})"
            ),
        )
        for match in pattern.finditer(excerpt)
    ]
    subject_groups=[match.group("subjects") for match in subject_matches]
    intervention_subjects = {
        value.casefold()
        for group in subject_groups
        for value in re.findall(gene_token, group)
    }
    # A token such as EGFR-L858R is the explicitly named scoped entity,
    # not another gene. Require every noncanonical subject span to end at
    # its exact source-owned variant mention; a free mutation token or a
    # different co-intervention is not enough. Later scope/semantic gates
    # still prohibit promoting this to a general/wild-type target claim.
    from drugpilot.agents.molecular_scope import mentions
    variant_mentions=[m for m in mentions(excerpt,target_symbol)
        if m["kind"] in {"amino_acid_variant","itd_class","exon_event"}]
    scoped_subjects_bound=bool(subject_matches) and all(
        match.group("subjects").casefold()==target_symbol.casefold()
        or any(m["start"]<=match.start("subjects") and m["end"]==match.end("subjects")
            for m in variant_mentions)
        for match in subject_matches)
    if (
        intervention_subjects
        and target_symbol.casefold() not in intervention_subjects
        and not scoped_subjects_bound
    ):
        codes.append("intervention_subject_not_target")
    return tuple(dict.fromkeys(codes))


def _fulltext_association_role_record(
    parent: EvidenceRef,
    content: _PublicationContent,
) -> EvidenceRef | None:
    """Create an association role without weakening OA JATS provenance."""

    excerpt = content.association_excerpt
    if not excerpt or not content.association_source_location:
        return None
    document_hash = parent.metadata.get("content_hash")
    source_location = parent.metadata.get("source_location")
    if not document_hash or not source_location:
        return None
    digest = sha256(
        (
            f"{parent.evidence_id}|"
            "primary_target_disease_association_excerpt"
        ).encode("utf-8")
    ).hexdigest()[:12]
    return parent.model_copy(
        update={
            "evidence_id": f"ev_{digest}",
            "supports": "support",
            "excerpt_hash": sha256(excerpt.encode("utf-8")).hexdigest(),
            "metadata": {
                **parent.metadata,
                "content_availability": "oa_fulltext_excerpt",
                "source_location": source_location,
                "excerpt": excerpt,
                "excerpt_hash": sha256(excerpt.encode("utf-8")).hexdigest(),
                "excerpt_char_count": len(excerpt),
                "excerpt_max_chars": _MAX_PUBLICATION_EXCERPT_CHARS,
                "excerpt_truncated": content.association_excerpt_truncated,
                "excerpt_selection_status": "selected_primary_association",
                # MA1's full-text consumer authenticates this parser token.
                "excerpt_selection_policy": _EPMC_OA_EXCERPT_POLICY,
                "association_excerpt_selection_policy": (
                    _ASSOCIATION_EXCERPT_POLICY
                ),
                "excerpt_relevance_basis": [
                    "target_symbol",
                    "disease",
                    "primary_association_result",
                    "association_predicate",
                ],
                "role_specific_record": (
                    "primary_target_disease_association_excerpt"
                ),
                "association_predicate": content.association_predicate,
                "association_scope": content.association_scope,
                "association_excerpt_relevance_score": content.association_score,
                "parent_evidence_id": parent.evidence_id,
                "parent_excerpt_hash": parent.excerpt_hash,
                "parent_provenance_pointer_type": (
                    "stable_document_record_id_not_claim_link"
                ),
                "same_document_external_id": parent.external_id,
                "duplicate_document_role": "association_excerpt",
                "additional_network_call": False,
                "direct_support_eligible": False,
                "direct_support_rejection_codes": ["association_role_only"],
                "verified_direct_primary": False,
                "verified_direct_primary_policy": (
                    _VERIFIED_DIRECT_PRIMARY_POLICY
                ),
                "verified_direct_primary_rejection_codes": [
                    "association_role_only"
                ],
                "evidence_scope": (
                    "primary non-review OA full-text target-disease "
                    "association/prevalence excerpt; not intervention-"
                    "direction proof"
                ),
                "claim_linkage_status": "pending_ma1_claim_reconciliation",
                "content_retrieval_policy": _EPMC_OA_FULLTEXT_POLICY,
                "content_hash": document_hash,
                "content_hash_algorithm": "sha256",
                "content_hash_scope": "europe_pmc_oa_jats_xml",
                "content_storage_policy": (
                    "bounded_oa_excerpt_only_no_full_text"
                ),
                "full_text_stored": False,
                "raw_xml_stored": False,
            },
        },
        deep=True,
    )


def _association_role_record(
    parent: EvidenceRef,
    content: _PublicationContent,
) -> EvidenceRef | None:
    """Create one immutable, role-specific excerpt from an existing abstract.

    The parent publication record remains unchanged.  The child record shares
    only document-level provenance (external identifier and full-abstract
    content hash), while carrying its own bounded excerpt, location and hash.
    No network request or full-text retention is introduced by this split.
    """

    excerpt = content.association_excerpt
    location = content.association_source_location
    content_hash = content.content_hash
    if not excerpt or not location or not content_hash:
        return None
    digest = sha256(
        (
            f"{parent.evidence_id}|primary_target_disease_association_excerpt|"
            f"{sha256(excerpt.encode('utf-8')).hexdigest()}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    excerpt_hash = sha256(excerpt.encode("utf-8")).hexdigest()
    return parent.model_copy(
        update={
            "evidence_id": f"ev_{digest}",
            "supports": "support",
            "excerpt_hash": excerpt_hash,
            "metadata": {
                **parent.metadata,
                "content_availability": "abstract_excerpt",
                "source_location": location,
                "excerpt": excerpt,
                "excerpt_char_count": len(excerpt),
                "excerpt_max_chars": _MAX_PUBLICATION_EXCERPT_CHARS,
                "excerpt_truncated": content.association_excerpt_truncated,
                "excerpt_selection_status": "selected_primary_association",
                "excerpt_selection_policy": _ASSOCIATION_EXCERPT_POLICY,
                "excerpt_relevance_basis": [
                    "target_symbol",
                    "disease",
                    "primary_association_result",
                    "association_predicate",
                ],
                "role_specific_record": (
                    "primary_target_disease_association_excerpt"
                ),
                "association_predicate": content.association_predicate,
                "association_scope": content.association_scope,
                "association_excerpt_relevance_score": content.association_score,
                "parent_evidence_id": parent.evidence_id,
                "parent_excerpt_hash": parent.excerpt_hash,
                "parent_provenance_pointer_type": (
                    "stable_document_record_id_not_claim_link"
                ),
                "same_document_external_id": parent.external_id,
                "duplicate_document_role": "association_excerpt",
                "additional_network_call": False,
                "counter_category": None,
                "counter_signal": None,
                "classification_policy": None,
                "review_status": None,
                "direct_support_eligible": False,
                "direct_support_rejection_codes": ["association_role_only"],
                "verified_direct_primary": False,
                "verified_direct_primary_policy": (
                    _VERIFIED_DIRECT_PRIMARY_POLICY
                ),
                "verified_direct_primary_rejection_codes": [
                    "association_role_only"
                ],
                "evidence_scope": (
                    "primary-publication target-disease association/prevalence "
                    "excerpt; not intervention-direction proof"
                ),
                "claim_linkage_status": "pending_ma1_claim_reconciliation",
                "content_storage_policy": (
                    "bounded_excerpt_only_no_full_text"
                ),
                "full_text_stored": False,
                # Retained from the parent and asserted explicitly here so a
                # downstream audit can prove that both bounded excerpts came
                # from the same already-fetched abstract.
                "content_hash": content_hash,
                **_abstract_context_metadata(content),
                "content_hash_algorithm": "sha256",
                "content_hash_scope": "normalized_full_abstract",
            },
        },
        deep=True,
    )


def _select_publication_records_with_role_budget(
    records: Sequence[EvidenceRef],
    *,
    max_records: int,
) -> list[EvidenceRef]:
    """Select parent/derived/context records under one physical-record cap.

    Every association excerpt points to a distinct publication record rather
    than merely to an identifier.  The pair is therefore admitted atomically.
    If the remaining physical-record budget cannot hold both records, the
    association is omitted; a dangling ``parent_evidence_id`` is never emitted.
    """

    if max_records <= 0:
        return []
    unique = list({item.evidence_id: item for item in records}.values())
    direct_parents = [
        item
        for item in unique
        if not item.metadata.get("role_specific_record")
        and item.supports == "support"
        and item.metadata.get("excerpt_selection_status") == "selected"
    ]
    association_records = [
        item
        for item in unique
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    ]
    direct_parent_ids = {item.evidence_id for item in direct_parents}
    association_record_ids = {item.evidence_id for item in association_records}
    records_by_id = {item.evidence_id: item for item in unique}

    atomic_associations = [
        item
        for item in association_records
        if item.metadata.get("parent_provenance_pointer_type")
        in {
            "stable_document_record_id_not_claim_link",
            "selected_publication_record_id_not_claim_link",
            "selected_oa_document_record_id_not_claim_link",
        }
    ]
    atomic_association_ids = {
        item.evidence_id for item in atomic_associations
    }
    atomic_pairs: list[tuple[EvidenceRef, EvidenceRef]] = []
    invalid_atomic_association_ids: set[str] = set()
    for association in atomic_associations:
        parent_id = str(association.metadata.get("parent_evidence_id") or "")
        parent = records_by_id.get(parent_id)
        content_availability = association.metadata.get("content_availability")
        if (
            parent is None
            or parent.evidence_id == association.evidence_id
            or bool(parent.metadata.get("role_specific_record"))
            or content_availability
            not in {"abstract_excerpt", "oa_fulltext_excerpt"}
            or parent.metadata.get("content_availability")
            != content_availability
            or parent.source != association.source
            or parent.external_id != association.external_id
            or parent.metadata.get("content_hash")
            != association.metadata.get("content_hash")
            or association.metadata.get("parent_excerpt_hash")
            != parent.excerpt_hash
        ):
            invalid_atomic_association_ids.add(association.evidence_id)
            continue
        atomic_pairs.append((parent, association))

    def priority(item: EvidenceRef) -> tuple[int, int, str]:
        # ``list.sort`` temporarily exposes an empty list while computing keys;
        # membership against the list being sorted therefore cannot define its
        # own key.  Stable identity sets keep the verified-primary rank intact.
        if item.evidence_id in direct_parent_ids:
            excerpt = str(item.metadata.get("excerpt") or "")
            score = len(_MECHANISTIC_ENDPOINT_PATTERN.findall(excerpt))
            return (
                6 if item.metadata.get("verified_direct_primary") is True else 5,
                score,
                item.evidence_id,
            )
        if item.evidence_id in association_record_ids:
            return (
                4,
                int(
                    item.metadata.get(
                        "association_excerpt_relevance_score"
                    )
                    or 0
                ),
                item.evidence_id,
            )
        if (
            item.metadata.get("content_availability")
            == "oa_fulltext_excerpt"
            and not item.metadata.get("role_specific_record")
        ):
            # Keep the origin-bound OA document record beside a role-specific
            # association excerpt so its parent ID/hash pointer remains
            # resolvable under a crowded source-level record budget.
            return 5, 0, item.evidence_id
        if item.supports == "counter":
            return 3, 0, item.evidence_id
        if item.metadata.get("content_availability") == "abstract_excerpt":
            return 2, 0, item.evidence_id
        return 1, 0, item.evidence_id

    direct_parents.sort(key=lambda item: priority(item), reverse=True)
    association_records.sort(key=lambda item: priority(item), reverse=True)
    atomic_pairs.sort(
        key=lambda pair: priority(pair[1]),
        reverse=True,
    )
    oa_atomic_pairs = [
        pair
        for pair in atomic_pairs
        if pair[1].metadata.get("parent_provenance_pointer_type")
        == "selected_oa_document_record_id_not_claim_link"
    ]
    abstract_atomic_pairs = [
        pair
        for pair in atomic_pairs
        if pair not in oa_atomic_pairs
    ]
    selected: list[EvidenceRef] = []
    selected_ids: set[str] = set()
    selected_atomic_association_ids: set[str] = set()

    def append_selected(item: EvidenceRef) -> bool:
        if item.evidence_id in selected_ids:
            return True
        if len(selected) >= max_records:
            return False
        selected.append(item)
        selected_ids.add(item.evidence_id)
        return True

    def reserve_best_pair(
        pairs: Sequence[tuple[EvidenceRef, EvidenceRef]],
    ) -> None:
        for parent, association in pairs:
            required_slots = sum(
                item.evidence_id not in selected_ids
                for item in (parent, association)
            )
            if required_slots > max_records - len(selected):
                continue
            append_selected(parent)
            append_selected(association)
            selected_atomic_association_ids.add(association.evidence_id)
            break

    # OA role excerpts require their selected origin record and receive the
    # long-standing atomic priority under tiny record caps.
    reserve_best_pair(oa_atomic_pairs)

    # Preserve the strongest verified direction record before allocating an
    # abstract-derived association pair.  The pair remains indivisible, but it
    # must not evict the only verified mechanistic publication.
    for parent in direct_parents:
        if parent.metadata.get("verified_direct_primary") is True:
            append_selected(parent)
            break

    # Reserve the best abstract parent+association pair before standalone
    # records.  With a one-record cap no pair is eligible.
    reserve_best_pair(abstract_atomic_pairs)

    # Preserve the prior direct-parent-first behavior whenever the atomic
    # pair has not consumed the entire physical-record budget.
    for parent in direct_parents:
        if append_selected(parent):
            break
    for association in association_records:
        if association.evidence_id in atomic_association_ids:
            continue
        if append_selected(association):
            break

    remaining = sorted(
        (item for item in unique if item.evidence_id not in selected_ids),
        key=priority,
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= max_records:
            break
        if item.evidence_id in invalid_atomic_association_ids:
            continue
        if item.evidence_id in atomic_association_ids:
            parent_id = str(item.metadata.get("parent_evidence_id") or "")
            parent = records_by_id.get(parent_id)
            if parent is None:
                continue
            required_slots = sum(
                record.evidence_id not in selected_ids
                for record in (parent, item)
            )
            if required_slots > max_records - len(selected):
                continue
            append_selected(parent)
            append_selected(item)
            selected_atomic_association_ids.add(item.evidence_id)
            continue
        append_selected(item)

    strict_drop_reasons: dict[str, int] = {}
    invalid_count = len(invalid_atomic_association_ids)
    if invalid_count:
        strict_drop_reasons["invalid_or_missing_publication_parent"] = invalid_count
    budget_drop_count = len(atomic_association_ids) - len(
        selected_atomic_association_ids
    ) - invalid_count
    if budget_drop_count:
        strict_drop_reasons["insufficient_physical_record_budget"] = (
            budget_drop_count
        )
    # This invariant is intentionally checked before returning records.  It
    # protects later changes to ranking from silently reintroducing a dangling
    # selected-OA-document pointer.
    assert all(
        str(item.metadata.get("parent_evidence_id") or "") in selected_ids
        for item in selected
        if item.evidence_id in atomic_association_ids
    )
    candidate_role_counts = {
        "parent_publication": sum(
            not bool(item.metadata.get("role_specific_record"))
            for item in unique
        ),
        "association_excerpt": len(association_records),
        "counter_signal": sum(item.supports == "counter" for item in unique),
    }
    dropped_primary_association_rejection_counts: dict[str, int] = {}
    for candidate in unique:
        if candidate.metadata.get("primary_association_candidate_status") != "rejected":
            continue
        raw_codes = candidate.metadata.get(
            "primary_association_candidate_rejection_codes",
            (),
        )
        if not isinstance(raw_codes, Sequence) or isinstance(raw_codes, (str, bytes)):
            continue
        for raw_code in raw_codes:
            code = str(raw_code).strip()
            if code:
                dropped_primary_association_rejection_counts[code] = (
                    dropped_primary_association_rejection_counts.get(code, 0) + 1
                )
    return [
        item.model_copy(
            update={
                "metadata": {
                    **item.metadata,
                    "source_record_budget_policy": (
                        "atomic-publication-parent-association-then-role-rank-v3"
                    ),
                    "source_record_budget_semantics": (
                        "physical_evidence_record_hard_cap"
                    ),
                    "source_record_atomic_dependency_policy": (
                        "association_requires_parent_record_in_packet_v2"
                    ),
                    "source_record_budget": max_records,
                    "source_record_candidate_count": len(unique),
                    "source_record_dropped_count": max(
                        0,
                        len(unique) - len(selected),
                    ),
                    "source_record_candidate_role_counts": (
                        candidate_role_counts
                    ),
                    "source_record_dropped_primary_association_rejection_counts": (
                        dict(sorted(dropped_primary_association_rejection_counts.items()))
                    ),
                    "source_record_atomic_association_candidate_count": len(
                        atomic_association_ids
                    ),
                    "source_record_atomic_association_selected_count": len(
                        selected_atomic_association_ids
                    ),
                    "source_record_atomic_association_dropped_count": sum(
                        strict_drop_reasons.values()
                    ),
                    "source_record_atomic_association_drop_reasons": dict(
                        sorted(strict_drop_reasons.items())
                    ),
                    "source_record_dangling_parent_pointer_count": 0,
                    **(
                        {
                            "parent_record_in_packet": True,
                            "parent_provenance_pointer_resolved": True,
                        }
                        if item.evidence_id
                        in selected_atomic_association_ids
                        else {}
                    ),
                    "source_record_selected_rank": rank,
                    "source_record_selected_role": (
                        "association_excerpt"
                        if item.evidence_id in association_record_ids
                        else "counter_signal"
                        if item.supports == "counter"
                        else "parent_publication"
                    ),
                }
            },
            deep=True,
        )
        for rank, item in enumerate(selected, 1)
    ]


def _target_token_pattern(target_symbol: str) -> str:
    exact = re.escape(target_symbol)
    lower = re.escape(target_symbol.casefold())
    token = exact if exact == lower else rf"(?:{exact}|{lower})"
    # Human gene symbols are case-sensitive. Accept the canonical uppercase
    # spelling and an all-lowercase indexing variant, but reject mixed-case
    # collisions such as eGFR (estimated glomerular filtration rate) for EGFR.
    return rf"(?<![A-Za-z0-9])(?-i:{token})(?![A-Za-z0-9])"


def _target_rearrangement_or_fusion_pattern(target_symbol: str) -> str:
    """One shared TARGET [gene] rearrangement/fusion subject grammar."""

    target = _target_token_pattern(target_symbol)
    return (
        rf"(?:{target}\s*(?:[-\u2010-\u2015]\s*|\s+)"
        r"(?:gene\s+)?(?:rearrang(?:ed|ement|ements)|"
        r"fusion(?:[- ]positive|s)?)\b|"
        r"\b(?:rearrang(?:ement|ements)|fusion|fusions)\s+"
        rf"(?:of|in|involving)\s+{target}\b)"
        rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}"
    )


def _association_clauses(text: str) -> tuple[str, ...]:
    """Split only on explicit contrast/clause boundaries used by abstracts."""

    return tuple(
        clause.strip(" ,")
        for clause in _PRIMARY_ASSOCIATION_CLAUSE_BOUNDARY_PATTERN.split(text)
        if clause.strip(" ,")
    )


def _has_target_bound_association_rejection(
    sentence: str,
    *,
    target_symbol: str,
) -> bool:
    """Bind null/negative signals to a clause that actually names the target."""

    target = re.compile(_target_token_pattern(target_symbol), re.I)
    negative_target_itd_prefix = re.compile(
        rf"\bnon(?:\s*{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)"
        rf"{_target_token_pattern(target_symbol)}"
        rf"(?:\s*{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)ITD\b",
        re.I,
    )
    for clause in _association_clauses(sentence):
        if not target.search(clause):
            continue
        if negative_target_itd_prefix.search(clause):
            return True
        relation_context = bool(
            _PRIMARY_ASSOCIATION_RELATION_CONTEXT_PATTERN.search(clause)
        )
        if (
            _PRIMARY_ASSOCIATION_NEGATED_OR_ZERO_PATTERN.search(clause)
            and (relation_context or _MUTATION_OR_VARIANT_PATTERN.search(clause))
        ):
            return True
        if (
            relation_context
            and _PRIMARY_ASSOCIATION_NEGATIVE_RELATION_PATTERN.search(clause)
        ):
            return True
    return False


def _target_variant_subject_pattern(target_symbol: str) -> str:
    target = _target_token_pattern(target_symbol)
    rearrangement_or_fusion = _target_rearrangement_or_fusion_pattern(
        target_symbol
    )
    return (
        rf"(?:{target}(?:(?:\s*{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)"
        rf"ITD(?![A-Za-z0-9])|"
        rf"\s+(?:mutation|mutations|mutant|variant|variants|"
        rf"internal tandem duplication)\b|"
        rf"\s*(?:(?:{_VARIANT_SEPARATOR_PATTERN}|:)\s*|\s+)"
        rf"(?:p\.)?[A-Za-z][0-9]{{1,5}}[A-Za-z*](?![A-Za-z0-9]))|"
        rf"\b(?:mutation|mutations|variant|variants)\s+"
        rf"(?:of|in|involving)\s+{target}|{rearrangement_or_fusion})"
        rf"(?![A-Za-z0-9]){_VARIANT_NEGATIVE_SUFFIX_GUARD}"
    )


def _association_predicate_for_target(
    sentence: str,
    target_symbol: str,
) -> str | None:
    """Return a predicate only when its grammatical subject is the target."""

    target = _target_token_pattern(target_symbol)
    target_variant = re.compile(
        _target_variant_subject_pattern(target_symbol),
        re.I,
    )
    if _target_prevalence_quantitations(
        sentence,
        target_symbol=target_symbol,
    ):
        return "mutation_prevalence"
    quantified_cases = (
        r"(?:\d+\s+(?:(?:of\s+\d+\s+)?(?:patients?|cases?|samples?))|"
        r"\d+\s+of\s+\d+)\s*\(\s*\d+(?:\.\d+)?\s*%\s*\)"
    )
    parenthetical_itd_alias = (
        rf"(?:\s*\(\s*{target}\s*{_VARIANT_SEPARATOR_PATTERN}\s*ITD\s*\))?"
    )
    target_prevalence = re.compile(
        rf"(?:{target_variant.pattern}{parenthetical_itd_alias}"
        rf"(?:\s+mutations?)?\s+"
        rf"(?:is|are|was|were)?\s*(?:frequent|prevalent|"
        rf"(?:detected|identified|observed|found|present|occur(?:red|s))\s+in)"
        rf"\s+(?:approximately\s+)?(?:\d+(?:\.\d+)?\s*%|"
        rf"{quantified_cases})|"
        rf"\b(?:frequency|prevalence)\s+of\s+{target_variant.pattern}"
        rf"[^.!?]{{0,20}}\d+(?:\.\d+)?\s*%|"
        rf"\d+(?:\.\d+)?\s*%\s+of\s+(?:the\s+)?(?:patients?|cases?|samples?)"
        rf"[^.!?]{{0,24}}\b(?:harbou?red|carried|had)\s+{target_variant.pattern})",
        re.I,
    )
    if target_prevalence.search(sentence):
        return "mutation_prevalence"

    # A single-patient/case observation is not prevalence, but in a rare
    # disease it can be a bounded primary target-disease association record.
    # Keep that role separate from any intervention-response claim.
    patient_variant_observation = re.compile(
        rf"(?:{target_variant.pattern}(?:\s+gene)?\s+"
        rf"(?:was|were)?\s*(?:identified|detected|found|observed)|"
        rf"(?:identified|detected|found|observed)\s+"
        rf"(?:an?\s+)?{target_variant.pattern})",
        re.I,
    )
    if (
        re.search(r"\b(?:patient|case)\b", sentence, re.I)
        and patient_variant_observation.search(sentence)
    ):
        return "disease_target_association"

    association_relation = re.compile(
        rf"(?:{target}\s+(?:dependency|driver)\b|"
        rf"\b(?:dependency|dependent)\s+(?:on\s+)?{target}|"
        rf"\bassociation\s+between\s+{target_variant.pattern}\s+and\b|"
        rf"\bassociation\s+between\s+{target}\s+and\b|"
        rf"{target}(?:\s+(?:mutation|mutations|variant|variants))?\s+"
        rf"(?:is|are|was|were)?\s*(?:associat|correlat)\w*|"
        rf"{target_variant.pattern}\s+(?:is|are|was|were)\s+"
        rf"(?:associat|correlat)\w*)",
        re.I,
    )
    if association_relation.search(sentence):
        return "disease_target_association"

    expression_subject = (
        rf"(?:{target_variant.pattern}|{target}(?!\s*{_VARIANT_SEPARATOR_PATTERN}))"
    )
    expression_relation = re.compile(
        rf"(?:{expression_subject}\s+(?:is|was|are|were)?\s*"
        rf"(?:overexpress(?:ed)?|"
        rf"upregulat(?:ed)?|expression)\b|"
        rf"\b(?:overexpression|upregulation|expression|levels?)\s+of\s+"
        rf"{expression_subject})",
        re.I,
    )
    if expression_relation.search(sentence):
        return "disease_target_expression"
    return None


def _target_prevalence_quantitations(
    sentence: str,
    *,
    target_symbol: str,
    variant_pattern: str | None = None,
) -> tuple[_PrevalenceQuantitation, ...]:
    """Extract only numeric tuples grammatically linked to the target variant."""

    target = _target_token_pattern(target_symbol)
    variant = variant_pattern or _target_variant_subject_pattern(target_symbol)
    alias = (
        rf"(?:\s*\(\s*{target}\s*{_VARIANT_SEPARATOR_PATTERN}\s*ITD\s*\))?"
    )
    payload = (
        r"(?:"
        r"(?P<ratio_n>\d+)\s+of\s+(?P<ratio_m>\d+)"
        r"(?:\s+(?:patients?|cases?|samples?))?\s*"
        r"\(\s*(?P<ratio_p>\d+(?:\.\d+)?)\s*%\s*\)|"
        r"(?P<count_n>\d+)\s+(?:patients?|cases?|samples?)\s*"
        r"\(\s*(?P<count_p>\d+(?:\.\d+)?)\s*%\s*\)|"
        r"(?P<percent_only>\d+(?:\.\d+)?)\s*%"
        r")"
    )
    patterns = (
        re.compile(
            rf"\b(?:of|among|in)\s+(?:the\s+)?"
            rf"(?P<cohort_m>\d+)\s+[^,;.!?]{{1,160}},\s*"
            rf"(?P<cohort_n>\d+)\s*\(\s*"
            rf"(?P<cohort_p>\d+(?:\.\d+)?)\s*%\s*\)\s+"
            rf"(?:showed|had|harbou?red|carried|demonstrated|displayed)\s+"
            rf"{variant}",
            re.I,
        ),
        re.compile(
            rf"{variant}{alias}(?:\s+mutations?)?\s+"
            rf"(?:is|are|was|were)?\s*(?:frequent|prevalent|"
            rf"(?:detected|identified|observed|found|present|occur(?:red|s))"
            rf"\s+in)\s+(?:approximately\s+)?{payload}",
            re.I,
        ),
        re.compile(
            rf"\b(?:frequency|prevalence)\s+of\s+{variant}"
            rf"(?:(?!\b(?:and|or|whereas|while|but)\b)[^.!?;]){{0,20}}"
            rf"(?P<relation_p>\d+(?:\.\d+)?)\s*%",
            re.I,
        ),
        re.compile(
            rf"(?P<reverse_p>\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?"
            rf"(?:patients?|cases?|samples?)[^.!?]{{0,24}}"
            rf"\b(?:harbou?red|carried|had)\s+{variant}",
            re.I,
        ),
        # Multi-gene cohort summaries commonly put the percentage before
        # each gene, for example ``(73% with JAK2 mutations, 23% with CALR
        # mutations)``.  Bind each percentage to the immediately following
        # target variant so a neighbouring gene's percentage cannot leak.
        re.compile(
            rf"(?P<reverse_with_p>\d+(?:\.\d+)?)\s*%\s+"
            rf"(?:with|had|harbou?red|carried)\s+{variant}",
            re.I,
        ),
    )
    continuation_pattern = re.compile(
        rf"\s*(?:,\s*(?:(?:and|or|versus|vs\.?)\s+)?|"
        rf"(?:and|or|versus|vs\.?)\s+)(?:in\s+)?"
        rf"(?:approximately\s+)?{payload}",
        re.I,
    )
    extracted: list[_PrevalenceQuantitation] = []

    def append_quantitation(match: re.Match[str]) -> None:
        groups = match.groupdict()
        if groups.get("cohort_p") is not None:
            percent_text = str(groups["cohort_p"])
            numerator = int(str(groups["cohort_n"]))
            denominator = int(str(groups["cohort_m"]))
        elif groups.get("ratio_p") is not None:
            percent_text = str(groups["ratio_p"])
            numerator = int(str(groups["ratio_n"]))
            denominator = int(str(groups["ratio_m"]))
        elif groups.get("count_p") is not None:
            percent_text = str(groups["count_p"])
            numerator = int(str(groups["count_n"]))
            denominator = None
        else:
            percent_text = str(
                groups.get("percent_only")
                or groups.get("relation_p")
                or groups.get("reverse_p")
                or groups.get("reverse_with_p")
            )
            numerator = None
            denominator = None
        extracted.append(
            _PrevalenceQuantitation(
                start=match.start(),
                numerator=numerator,
                denominator=denominator,
                percent=float(percent_text),
                percent_text=percent_text,
            )
        )

    def continuation_has_new_subject(start: int) -> bool:
        tail = sentence[start:]
        boundary = re.search(
            r"[,;.!?]|\b(?:and|or|whereas|while|but)\b",
            tail,
            re.I,
        )
        bounded_tail = tail[: boundary.start()] if boundary else tail
        return bool(_NEW_GENE_OR_VARIANT_SUBJECT_PATTERN.search(bounded_tail))

    for pattern in patterns:
        for match in pattern.finditer(sentence):
            append_quantitation(match)
            cursor = match.end()
            while continuation := continuation_pattern.match(sentence, cursor):
                if continuation_has_new_subject(continuation.end()):
                    break
                append_quantitation(continuation)
                cursor = continuation.end()
    return tuple(
        {
            (
                item.start,
                item.numerator,
                item.denominator,
                item.percent_text,
            ): item
            for item in sorted(extracted, key=lambda item: item.start)
        }.values()
    )


def _explicit_primary_cohort_totals(sentence: str) -> tuple[tuple[int, int], ...]:
    patterns = (
        re.compile(
            r"\bamong\s+(?:the\s+)?(\d+)\s+"
            r"(?:[A-Za-z][A-Za-z-]*\s+){0,8}(?:patients?|cases?|samples?)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:in\s+(?:our|the|a)\s+)?cohort\s+of\s+(\d+)\s+"
            r"(?:[A-Za-z][A-Za-z-]*\s+){0,8}(?:patients?|cases?|samples?)\b",
            re.I,
        ),
        re.compile(
            r"\b(\d+)\s+(?:[A-Za-z][A-Za-z-]*\s+){0,8}"
            r"(?:patients?|cases?|samples?)\s+were\s+"
            r"(?:enrolled|included|analysed|analyzed|evaluated)\b",
            re.I,
        ),
        re.compile(
            r"(?:^|[.;:]\s*)\b(?:in|of)\s+(\d+)\s+"
            r"(?:[A-Za-z][A-Za-z-]*\s+){0,8}"
            r"(?:patients?|cases?|samples?)\b",
            re.I,
        ),
        re.compile(r"\bn\s*=\s*(\d+)\b", re.I),
    )
    totals = [
        (match.start(), int(match.group(1)))
        for pattern in patterns
        for match in pattern.finditer(sentence)
    ]
    return tuple(sorted(set(totals)))


def _has_explicit_primary_disease_cohort(
    sentence: str,
    *,
    disease: str,
    target_symbol: str,
) -> bool:
    """Require ``Of N <exact disease alias>`` in the selected sentence."""

    molecular = _target_molecular_disease_base(
        disease,
        target_symbol=target_symbol,
    )
    base = molecular[0] if molecular is not None else disease
    full_aliases = _terminal_plural_aliases(base)
    prefix = r"\b(?i:of|among|in)\s+(?:(?i:the)\s+)?\d+\s+"
    for alias in full_aliases:
        alias_pattern = r"\s+".join(
            re.escape(token) for token in alias.split()
        )
        if re.search(
            rf"{prefix}{alias_pattern}(?=\s|[,;(])",
            sentence,
            re.I,
        ):
            return True
        if re.search(
            rf"\b(?:from|of|among|in)\s+(?:the\s+)?\d+\s+"
            rf"(?:patients?|cases?|samples?)\s+(?:with|having)\s+"
            rf"{alias_pattern}(?=\s|[,;(])",
            sentence,
            re.I,
        ):
            return True
    return any(
        re.search(
            rf"{prefix}(?<![A-Z0-9]){re.escape(initialism)}s?"
            rf"(?![A-Za-z0-9])",
            sentence,
        )
        for initialism in _disease_initialisms(base)
        if initialism.casefold() != target_symbol.casefold()
    )


def _quantitation_matches_denominator(
    quantitation: _PrevalenceQuantitation,
    denominator: int,
) -> bool:
    numerator = quantitation.numerator
    if (
        numerator is None
        or numerator <= 0
        or denominator <= 0
        or numerator > denominator
        or not 0 < quantitation.percent <= 100
    ):
        return False
    decimals = (
        len(quantitation.percent_text.rsplit(".", 1)[1])
        if "." in quantitation.percent_text
        else 0
    )
    rounding_tolerance = 0.5 * (10 ** (-decimals)) + 1e-9
    expected_percent = 100 * numerator / denominator
    return abs(quantitation.percent - expected_percent) <= rounding_tolerance


def _has_positive_prevalence_quantitation(
    sentence: str,
    *,
    target_symbol: str,
) -> bool:
    """Validate every numeric tuple attached to the target prevalence relation."""

    quantitations = _target_prevalence_quantitations(
        sentence,
        target_symbol=target_symbol,
    )
    if not quantitations:
        return False
    cohort_totals = _explicit_primary_cohort_totals(sentence)
    for quantitation in quantitations:
        if not 0 < quantitation.percent <= 100:
            return False
        if quantitation.numerator is None:
            continue
        explicit_total = next(
            (
                total
                for position, total in reversed(cohort_totals)
                if position <= quantitation.start
            ),
            cohort_totals[0][1] if cohort_totals else None,
        )
        denominator = (
            quantitation.denominator
            if quantitation.denominator is not None
            else explicit_total
        )
        if denominator is None:
            if quantitation.numerator <= 0:
                return False
            continue
        if (
            quantitation.denominator is not None
            and explicit_total is not None
            and quantitation.denominator != explicit_total
        ):
            return False
        if not _quantitation_matches_denominator(
            quantitation,
            denominator,
        ):
            return False
    return True


def _has_target_subtype_or_variant(text: str, target_symbol: str) -> bool:
    target = rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
    return bool(
        re.search(
            rf"(?:{target}\s*(?:{_VARIANT_SEPARATOR_PATTERN}\s*|\s+)"
            rf"(?:ITD(?![A-Za-z0-9])|internal tandem duplication\b)"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}|"
            rf"internal tandem duplication(?:\s+(?:of|in))?\s+{target}"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}|"
            rf"{target}\s*(?:(?:{_VARIANT_SEPARATOR_PATTERN}|:)\s*|\s+)"
            rf"(?:p\.)?"
            rf"[A-Za-z][0-9]{{1,5}}[A-Za-z*](?![A-Za-z0-9])"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}|"
            rf"{target}[^.!?]{{0,48}}\b(?:activating (?:mutation|variant)|"
            rf"constitutive(?:ly)? activat\w*)\b"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}|"
            rf"\bactivating\s+{target}\s+(?:mutation|variant)s?\b"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}|"
            rf"\bactivating\s+(?:mutation|variant)s?\s+"
            rf"(?:in|of)\s+(?:the\s+)?{target}(?:\s+gene)?\b"
            rf"{_VARIANT_NEGATIVE_SUFFIX_GUARD}|"
            rf"{_target_rearrangement_or_fusion_pattern(target_symbol)})",
            text,
            re.I,
        )
    )


def _has_target_rearrangement_or_fusion(
    text: str,
    target_symbol: str,
) -> bool:
    return bool(
        re.search(
            _target_rearrangement_or_fusion_pattern(target_symbol),
            text,
            re.I,
        )
    )


def _bounded_primary_association_excerpt(
    sections: Sequence[tuple[str, str]],
    *,
    target_symbol: str,
    disease: str,
) -> tuple[str, str, bool, str, str, int] | None:
    """Select one bounded primary association/prevalence sentence.

    This is intentionally independent from the mechanistic selector.  It may
    retain a second short sentence from the already-fetched abstract, but it
    never retains the full abstract and never uses title-only text.
    """

    candidates: list[
        tuple[int, int, int, str, str, int, str, str]
    ] = []
    for section_index, (label, text) in enumerate(sections):
        matches = list(_ABSTRACT_SENTENCE_PATTERN.finditer(text))
        if not matches:
            matches = [re.match(r"[\s\S]+", text)]  # type: ignore[list-item]
        for sentence_index, match in enumerate(matches, 1):
            if match is None:
                continue
            sentence = match.group(0).strip()
            if not sentence or not _has_strict_association_target_disease_match(
                sentence,
                target_symbol=target_symbol,
                disease=disease,
            ):
                continue
            if re.search(r"\b(?:background|introduction|objective)s?\b", label, re.I):
                continue
            if _ASSOCIATION_CONTEXT_EXCLUSION_PATTERN.search(sentence):
                continue
            from drugpilot.agents.molecular_scope import functional_association_sentence
            functional_relation = functional_association_sentence(sentence,target_symbol)
            if _PRIMARY_ASSOCIATION_NON_HUMAN_COHORT_PATTERN.search(sentence) and not functional_relation:
                continue
            if _has_target_bound_association_rejection(
                sentence,
                target_symbol=target_symbol,
            ):
                continue
            predicate = "functional_target_disease_association" if functional_relation else _association_predicate_for_target(
                sentence,
                target_symbol,
            )
            if predicate is None:
                continue
            result_section = bool(
                re.search(r"\b(?:result|finding|conclusion)s?\b", label, re.I)
            )
            if functional_relation:
                # Functional experiments are not a human prevalence cohort.
                # Background assertions or a bare mechanistic sentence do not qualify.
                result_section = bool(re.search(r"\b(?:result|finding)s?\b",label,re.I)
                    or re.match(r"\s*(?:Results?|Findings?)\b",sentence,re.I))
                if not result_section and not _PRIMARY_ASSOCIATION_EXPLICIT_STUDY_PATTERN.search(sentence):
                    continue
            if not (
                result_section
                or _PRIMARY_ASSOCIATION_EXPLICIT_STUDY_PATTERN.search(sentence)
                or _has_explicit_primary_disease_cohort(
                    sentence,
                    disease=disease,
                    target_symbol=target_symbol,
                )
            ):
                continue
            if not _PRIMARY_ASSOCIATION_RESULT_PATTERN.search(sentence):
                continue
            if (
                predicate == "mutation_prevalence"
                and not _has_positive_prevalence_quantitation(
                    sentence,
                    target_symbol=target_symbol,
                )
            ):
                continue
            if (
                predicate == "mutation_prevalence"
                and not _MUTATION_OR_VARIANT_PATTERN.search(sentence)
                and not _has_target_subtype_or_variant(
                    sentence,
                    target_symbol,
                )
            ):
                continue
            if (
                predicate == "disease_target_expression"
                and not _COMPARATIVE_EXPRESSION_PATTERN.search(sentence)
            ):
                continue
            subtype_specific = (
                _variant_is_expression_subject(
                    sentence,
                    variant_pattern=_target_variant_subject_pattern(
                        target_symbol
                    ),
                )
                if predicate == "disease_target_expression"
                else _has_target_subtype_or_variant(
                    sentence,
                    target_symbol,
                )
            )
            scope = (
                "rearranged_or_fusion_target_subtype"
                if _has_target_rearrangement_or_fusion(
                    sentence,
                    target_symbol,
                )
                else "activating_or_itd_target_subtype"
                if subtype_specific
                else "target_general"
            )
            predicate_score = {
                "mutation_prevalence": 14,
                "disease_target_association": 10,
                "disease_target_expression": 8,
                "functional_target_disease_association": 10,
            }[predicate]
            score = predicate_score + (12 if subtype_specific else 0)
            if re.search(r"\b(?:result|finding)s?\b", label, re.I):
                score += 2
            candidates.append(
                (
                    score,
                    -section_index,
                    -sentence_index,
                    label,
                    sentence,
                    match.start(),
                    predicate,
                    scope,
                )
            )

    for (
        score,
        _,
        _,
        label,
        sentence,
        sentence_start,
        predicate,
        scope,
    ) in sorted(candidates, key=lambda item: item[:3], reverse=True):
        excerpt, local_start, local_end, truncated = _bounded_window(
            sentence,
            target_symbol=target_symbol,
            disease=disease,
            max_chars=_MAX_PUBLICATION_EXCERPT_CHARS,
        )
        if not _has_strict_association_target_disease_match(
            excerpt,
            target_symbol=target_symbol,
            disease=disease,
        ):
            continue
        absolute_start = sentence_start + local_start
        absolute_end = sentence_start + local_end
        return (
            excerpt,
            f"abstract:{label}:chars:{absolute_start}-{absolute_end}",
            truncated,
            predicate,
            scope,
            score,
        )
    return None


def _bounded_relevant_excerpt(
    sections: Sequence[tuple[str, str]],
    *,
    target_symbol: str,
    disease: str,
    intervention_direction: str | None = None,
    direction_model_context: _PrimaryDirectionModelContext | None = None,
) -> tuple[str, str, bool] | None:
    target_pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])",
        re.I,
    )
    disease_phrase = " ".join(disease.casefold().split())
    disease_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", disease_phrase)
        if len(token) >= 3
        and token not in {"disease", "cancer", "syndrome", "disorder"}
    }

    ranked: list[tuple[int, int, int, str, str, int, int]] = []
    for section_index, (label, text) in enumerate(sections):
        matches = list(_ABSTRACT_SENTENCE_PATTERN.finditer(text))
        if not matches:
            matches = [re.match(r"[\s\S]+", text)]  # type: ignore[list-item]
        for sentence_index, match in enumerate(matches, 1):
            if match is None:
                continue
            sentence = match.group(0).strip()
            if not sentence:
                continue
            normalized = sentence.casefold()
            if target_pattern.search(sentence) is None:
                continue
            context_match_basis = _primary_direction_context_match_basis(
                sentence,
                target_symbol=target_symbol,
                disease=disease,
                context=direction_model_context,
            )
            if context_match_basis is None:
                continue
            normalized_match_text = " ".join(
                re.findall(r"[a-z0-9]+", normalized)
            )
            exact_disease = bool(
                disease_phrase
                and " ".join(re.findall(r"[a-z0-9]+", disease_phrase))
                in normalized_match_text
            )
            matched_disease_tokens = sum(
                1
                for token in disease_tokens
                if re.search(rf"\b{re.escape(token)}\b", normalized_match_text)
            )
            initialism_disease = _contains_disease_initialism(sentence, disease)
            score = 10 + (
                8
                if exact_disease
                else 6
                if initialism_disease
                else 6
                if context_match_basis == "predeclared_molecular_model"
                else min(4, matched_disease_tokens)
            )
            score += _mechanistic_excerpt_score(
                sentence,
                section_label=label,
                intervention_direction=intervention_direction,
            )
            # Keep rejected context available for audit when no cleaner
            # sentence exists, but always prefer an independently eligible
            # direct result over a combination/response-modifier sentence in
            # the same abstract.
            if (
                intervention_direction is not None
                and not _direct_support_rejection_codes(sentence, target_symbol)
                and _direct_target_intervention_pattern(
                    target_symbol,
                    intervention_direction,
                ).search(sentence)
                is not None
                and _has_admissible_direction_endpoint(
                    sentence,
                    target_symbol,
                )
                and _MECHANISTIC_ENDPOINT_PATTERN.search(sentence) is not None
                and _EXPERIMENTAL_RESULT_PATTERN.search(sentence) is not None
                and _direction_endpoint_polarity_status(
                    sentence,
                    intervention_direction,
                )
                == "favorable"
            ):
                score += 100
            # Earlier sentences win a tie, but relevance always dominates.
            ranked.append(
                (
                    score,
                    -section_index,
                    -sentence_index,
                    label,
                    sentence,
                    match.start(),
                    match.end(),
                )
            )

    if not ranked:
        return None

    for _, _, _, label, sentence, sentence_start, _ in sorted(
        ranked,
        key=lambda item: item[:3],
        reverse=True,
    ):
        excerpt, local_start, local_end, truncated = _bounded_window(
            sentence,
            target_symbol=target_symbol,
            disease=disease,
            max_chars=_MAX_PUBLICATION_EXCERPT_CHARS,
        )
        # A long sentence can be clipped.  Re-check the stored slice so that
        # the retained quotation itself, rather than an unseen part of the
        # abstract, contains both the target and disease context.
        if _primary_direction_context_match_basis(
            excerpt,
            target_symbol=target_symbol,
            disease=disease,
            context=direction_model_context,
        ) is None:
            continue
        absolute_start = sentence_start + local_start
        absolute_end = sentence_start + local_end
        return (
            excerpt,
            f"abstract:{label}:chars:{absolute_start}-{absolute_end}",
            truncated,
        )
    return None


def _bounded_compound_linked_direction_excerpt(
    sections: Sequence[tuple[str, str]],
    *,
    target_symbol: str,
    disease: str,
) -> tuple[str, str, bool] | None:
    """Pair target identity and outcome sentences only through the same drug.

    Structured clinical abstracts commonly define a target-selective compound
    in BACKGROUND and report its measured outcome in RESULTS or CONCLUSIONS.
    This helper admits that document-local structure only when the exact same
    compound token occurs in both sentences and the abstract explicitly
    declares the disease abbreviation used by those sentences.
    """

    sentence_rows: list[tuple[str, str, int, int]] = []
    for label, text in sections:
        matches = list(_ABSTRACT_SENTENCE_PATTERN.finditer(text))
        if not matches:
            fallback = re.match(r"[\s\S]+", text)
            matches = [fallback] if fallback is not None else []
        for match in matches:
            sentence = match.group(0).strip()
            if sentence:
                sentence_rows.append((label, sentence, match.start(), match.end()))

    normalized_disease = " ".join(disease.split()).strip()
    if not normalized_disease:
        return None
    disease_pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(normalized_disease)}(?![A-Za-z0-9])",
        re.I,
    )
    declaration_rows: list[tuple[str, str, int, int]] = []
    declared_aliases: set[str] = set()
    for row in sentence_rows:
        if disease_pattern.search(row[1]) is None:
            continue
        declaration_rows.append(row)
        declared_aliases.update(
            match.group(1).upper()
            for match in re.finditer(
                rf"{re.escape(normalized_disease)}\s*\(\s*([A-Z][A-Z0-9-]{{1,8}})\s*\)",
                row[1],
                re.I,
            )
        )

    def has_disease_context(sentence: str) -> bool:
        if disease_pattern.search(sentence):
            return True
        return any(
            re.search(
                rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])",
                sentence,
            )
            for alias in declared_aliases
        )

    target = _target_token_pattern(target_symbol)
    compound_before = re.compile(
        rf"\b(?P<compound>[A-Za-z][A-Za-z0-9-]{{2,30}})\s*,?\s+"
        rf"(?:(?:is|was)\s+)?(?:an?\s+)?(?:oral\s+)?{target}"
        rf"(?:[- ](?:selective|specific))?[- ]"
        rf"(?:(?:tyrosine[- ]kinase|kinase)[- ])?inhibitor\b",
        re.I,
    )
    compound_after = re.compile(
        rf"\b{target}(?:[- ](?:selective|specific))?[- ]"
        rf"(?:(?:tyrosine[- ]kinase|kinase)[- ])?inhibitor\s+"
        rf"(?P<compound>[A-Za-z][A-Za-z0-9-]{{2,30}})\b",
        re.I,
    )
    anchors: list[tuple[str, tuple[str, str, int, int]]] = []
    for row in sentence_rows:
        if _direct_target_intervention_pattern(
            target_symbol,
            "inhibit",
        ).search(row[1]) is None:
            continue
        match = compound_before.search(row[1]) or compound_after.search(row[1])
        if match is None:
            continue
        compound = match.group("compound").strip()
        if compound.casefold() in {"inhibitor", "inhibitors", "therapy"}:
            continue
        anchors.append((compound, row))

    for compound, anchor in anchors:
        compound_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(compound)}(?![A-Za-z0-9])",
            re.I,
        )
        for result in sentence_rows:
            label, sentence, _, _ = result
            if result == anchor or compound_pattern.search(sentence) is None:
                continue
            if not has_disease_context(sentence):
                continue
            if _MECHANISTIC_ENDPOINT_PATTERN.search(sentence) is None:
                continue
            if _EXPERIMENTAL_RESULT_PATTERN.search(sentence) is None:
                continue
            if _direction_endpoint_polarity_status(sentence, "inhibit") != "favorable":
                continue
            if not (
                re.search(r"\b(?:result|conclusion|finding)s?\b", label, re.I)
                or re.search(
                    r"\b(?:significant(?:ly)?|P\s*[<=>]|\d+(?:\.\d+)?%)\b",
                    sentence,
                    re.I,
                )
            ):
                continue

            selected_rows: list[tuple[str, str, int, int]] = []
            if not disease_pattern.search(anchor[1]) and not disease_pattern.search(sentence):
                declaration = next(
                    (
                        row
                        for row in declaration_rows
                        if any(
                            re.search(
                                rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])",
                                anchor[1] + " " + sentence,
                            )
                            for alias in declared_aliases
                        )
                    ),
                    None,
                )
                if declaration is None:
                    continue
                declaration_match = re.search(
                    rf"{re.escape(normalized_disease)}\s*\(\s*"
                    rf"(?:{'|'.join(re.escape(alias) for alias in sorted(declared_aliases))})"
                    rf"\s*\)",
                    declaration[1],
                    re.I,
                )
                if declaration_match is None:
                    continue
                selected_rows.append(
                    (
                        declaration[0],
                        declaration_match.group(0),
                        declaration[2] + declaration_match.start(),
                        declaration[2] + declaration_match.end(),
                    )
                )
            for row in (anchor, result):
                if row not in selected_rows:
                    selected_rows.append(row)
            excerpt = " ".join(row[1] for row in selected_rows)
            if len(excerpt) > _MAX_PAIRED_DIRECTION_EXCERPT_CHARS:
                continue
            location = "|".join(
                f"abstract:{row[0]}:chars:{row[2]}-{row[3]}"
                for row in selected_rows
            )
            return excerpt, f"abstract-paired:{location}", False
    return None


def _mechanistic_excerpt_score(
    sentence: str,
    *,
    section_label: str,
    intervention_direction: str | None,
) -> int:
    """Rank direct experimental language without relaxing excerpt eligibility."""

    score = 0
    direction_pattern = _DIRECTION_EXCERPT_PATTERNS.get(
        intervention_direction or ""
    )
    if direction_pattern is not None and direction_pattern.search(sentence):
        score += 12

    endpoint_hits = {
        match.group(0).casefold()
        for match in _MECHANISTIC_ENDPOINT_PATTERN.finditer(sentence)
    }
    score += min(8, len(endpoint_hits) * 2)

    if _EXPERIMENTAL_RESULT_PATTERN.search(sentence):
        score += 4
    if re.search(r"\b(?:result|conclusion|finding)s?\b", section_label, re.I):
        score += 2
    return score


def _bounded_window(
    text: str,
    *,
    target_symbol: str,
    disease: str,
    max_chars: int,
) -> tuple[str, int, int, bool]:
    if len(text) <= max_chars:
        return text, 0, len(text), False

    folded = text.casefold()
    anchors = [
        position
        for position in (
            folded.find(target_symbol.casefold()),
            folded.find(" ".join(disease.casefold().split())),
        )
        if position >= 0
    ]
    anchors.extend(
        match.start()
        for initialism in _disease_initialisms(disease)
        for match in re.finditer(
            rf"(?<![A-Z0-9]){re.escape(initialism)}(?![A-Z0-9])",
            text,
        )
    )
    anchor = min(anchors) if anchors else 0
    start = max(0, anchor - max_chars // 3)
    end = min(len(text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    if start:
        next_space = text.find(" ", start)
        if 0 <= next_space < end:
            start = next_space + 1
    if end < len(text):
        previous_space = text.rfind(" ", start, end)
        if previous_space > start:
            end = previous_space
    return text[start:end].strip(), start, end, True


def _has_direct_target_disease_match(
    text: str,
    *,
    target_symbol: str,
    disease: str,
) -> bool:
    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])",
        text,
        re.I,
    ) is None:
        return False
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    normalized_disease = " ".join(
        re.findall(r"[a-z0-9]+", disease.casefold())
    )
    molecular = _target_molecular_disease_base(
        disease,
        target_symbol=target_symbol,
    )
    if molecular is not None:
        base, qualifier = molecular
        normalized_aliases = {
            " ".join(re.findall(r"[a-z0-9]+", alias.casefold()))
            for alias in _terminal_plural_aliases(base)
        }
        base_match = any(
            alias and f" {alias} " in f" {normalized_text} "
            for alias in normalized_aliases
        ) or _contains_disease_initialism(
            text,
            base,
            excluded_initialisms=(target_symbol,),
        )
        if not base_match:
            return False
        if "rearrang" in qualifier or "fusion" in qualifier:
            return bool(
                re.search(
                    _target_rearrangement_or_fusion_pattern(target_symbol),
                    text,
                    re.I,
                )
            )
        elif "mut" in qualifier:
            target = _target_token_pattern(target_symbol)
            return bool(
                re.search(
                    rf"(?:{target}\s*(?:[-\u2010-\u2015]\s*|\s+)"
                    r"(?:mutant|mutated|mutation(?:s)?)\b|"
                    r"\b(?:mutation|mutations)\s+(?:of|in)\s+"
                    rf"{target}\b){_VARIANT_NEGATIVE_SUFFIX_GUARD}",
                    text,
                    re.I,
                )
            )
        else:
            return False
    if normalized_disease and normalized_disease in normalized_text:
        return True
    if _contains_disease_initialism(text, disease):
        return True
    disease_tokens = {
        token
        for token in normalized_disease.split()
        if len(token) >= 3
        and token not in {"disease", "cancer", "syndrome", "disorder"}
    }
    required = 1 if len(disease_tokens) <= 1 else 2
    matched = sum(
        1
        for token in disease_tokens
        if re.search(rf"\b{re.escape(token)}\b", normalized_text)
    )
    return bool(disease_tokens) and matched >= required


def _has_strict_association_target_disease_match(
    text: str,
    *,
    target_symbol: str,
    disease: str,
) -> bool:
    """Association excerpts require the exact disease or uppercase initialism."""

    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])",
        text,
        re.I,
    ) is None:
        return False
    normalized_text = " ".join(
        re.findall(r"[a-z0-9]+", text.casefold())
    )
    normalized_disease = " ".join(
        re.findall(r"[a-z0-9]+", disease.casefold())
    )
    if (
        normalized_disease
        and f" {normalized_disease} " in f" {normalized_text} "
    ):
        return True
    if _contains_disease_initialism(text, disease):
        return True
    molecular = _target_molecular_disease_base(
        disease,
        target_symbol=target_symbol,
    )
    if molecular is None:
        return False
    base, qualifier = molecular
    normalized_aliases = {
        " ".join(re.findall(r"[a-z0-9]+", alias.casefold()))
        for alias in _terminal_plural_aliases(base)
    }
    base_match = any(
        alias and f" {alias} " in f" {normalized_text} "
        for alias in normalized_aliases
    ) or _contains_disease_initialism(
        text,
        base,
        excluded_initialisms=(target_symbol,),
    )
    if not base_match:
        return False
    target = _target_token_pattern(target_symbol)
    if "rearrang" in qualifier or "fusion" in qualifier:
        return bool(
            re.search(
                _target_rearrangement_or_fusion_pattern(target_symbol),
                text,
                re.I,
            )
        )
    elif "mut" in qualifier:
        return bool(
            re.search(
                rf"(?:{target}\s*(?:[-\u2010-\u2015]\s*|\s+)"
                r"(?:mutant|mutated|mutation(?:s)?)\b|"
                r"\b(?:mutation|mutations)\s+(?:of|in)\s+"
                rf"{target}\b){_VARIANT_NEGATIVE_SUFFIX_GUARD}",
                text,
                re.I,
            )
        )
    else:
        return False


def _classify_counter_signal(
    title: str,
    abstract: str,
    target_symbol: str,
    *,
    intervention_direction: str | None = None,
) -> tuple[str, str, str, str] | None:
    target_pattern = re.compile(_target_token_pattern(target_symbol))
    for location, text in (("title", title), ("abstract", abstract)):
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            clauses = re.split(
                r"\s+(?:whereas|but|although|however)\s+|\s*;\s*",
                sentence,
                flags=re.I,
            )
            for clause in clauses:
                target_match = target_pattern.search(clause)
                if target_match is None:
                    continue
                if (
                    intervention_direction is not None
                    and _direct_target_intervention_pattern(
                        target_symbol,
                        intervention_direction,
                    ).search(clause)
                    is None
                ):
                    continue
                for category, pattern in _COUNTER_SIGNAL_PATTERNS:
                    match = pattern.search(clause)
                    if match is None:
                        continue
                    if abs(target_match.start() - match.start()) > 80:
                        continue
                    if (
                        category == "opposite_or_null_result"
                        and _counter_signal_negates_an_adverse_outcome(
                            clause,
                            signal_match=match,
                        )
                    ):
                        continue
                    if category in {"resistance", "non_response"} and not (
                        _counter_signal_is_bound_to_target_intervention(
                            clause,
                            target_symbol=target_symbol,
                            intervention_direction=intervention_direction,
                            signal_match=match,
                        )
                    ):
                        continue
                    excerpt = clause.strip()[:600]
                    return category, match.group(0), excerpt, location
    return None


def _counter_signal_negates_an_adverse_outcome(
    clause: str,
    *,
    signal_match: re.Match[str],
) -> bool:
    """Do not invert a negated adverse association into counter evidence.

    ``not associated with`` is directionally incomplete by itself.  A paper
    saying treatment was *not* associated with an inferior outcome is neutral
    or favorable, whereas "not associated with improved survival" remains a
    genuine lack-of-benefit signal.  Only suppress the former, explicitly
    adverse object; ambiguous null results continue to require review.
    """

    if signal_match.group(0).strip().casefold() != "not associated with":
        return False
    object_window = clause[signal_match.end() : signal_match.end() + 120]
    return bool(
        re.search(
            r"^\s*(?:an?\s+|the\s+)?(?:"
            r"inferior|worse|poor(?:er)?|adverse|unfavou?rable|"
            r"increased\s+(?:risk|mortality|toxicity)|higher\s+(?:risk|mortality)|"
            r"treatment\s+failure|disease\s+progression|progression|relapse|toxicity"
            r")\b",
            object_window,
            re.I,
        )
    )


def _counter_signal_is_bound_to_target_intervention(
    clause: str,
    *,
    target_symbol: str,
    intervention_direction: str | None,
    signal_match: re.Match[str],
) -> bool:
    """Reject baseline/refractory-disease language unrelated to the intervention."""

    if intervention_direction is None:
        return False
    intervention = _direct_target_intervention_pattern(
        target_symbol,
        intervention_direction,
    )
    intervention_matches = list(intervention.finditer(clause))
    if not intervention_matches:
        return False
    signal_start, signal_end = signal_match.span()
    relation_after_signal = re.compile(
        r"^.{0,24}\b(?:to|against|during|after|following|despite|on)\b",
        re.I | re.S,
    )
    relation_after_intervention = re.compile(
        r"^.{0,32}\b(?:resistan(?:ce|t)|refractory|non[- ]?response|"
        r"non[- ]?responder)\b",
        re.I | re.S,
    )
    for intervention_match in intervention_matches:
        if intervention_match.start() >= signal_end:
            bridge = clause[signal_end : intervention_match.start()]
            if relation_after_signal.search(bridge):
                return True
        elif signal_start >= intervention_match.end():
            bridge_and_signal = clause[
                intervention_match.end() : signal_match.end()
            ]
            if relation_after_intervention.search(bridge_and_signal):
                return True
    return False


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _uniprot_protein_name(item: Mapping[str, Any]) -> str | None:
    description = item.get("proteinDescription")
    if not isinstance(description, Mapping):
        return None
    recommended = description.get("recommendedName")
    if isinstance(recommended, Mapping):
        full_name = recommended.get("fullName")
        if isinstance(full_name, Mapping) and full_name.get("value"):
            return str(full_name["value"])
    submission = description.get("submissionNames")
    if isinstance(submission, Sequence) and submission:
        first = submission[0]
        if isinstance(first, Mapping):
            full_name = first.get("fullName")
            if isinstance(full_name, Mapping) and full_name.get("value"):
                return str(full_name["value"])
    return None


def _uniprot_gene_names(item: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    genes = item.get("genes")
    if not isinstance(genes, Sequence):
        return values
    for gene in genes:
        if not isinstance(gene, Mapping):
            continue
        name = gene.get("geneName")
        if isinstance(name, Mapping) and name.get("value"):
            values.append(str(name["value"]))
        for synonym in gene.get("synonyms", []):
            if isinstance(synonym, Mapping) and synonym.get("value"):
                values.append(str(synonym["value"]))
    return list(dict.fromkeys(values))


def _compact_dgidb_interaction(item: Mapping[str, Any]) -> dict[str, Any]:
    drug = item.get("drug") if isinstance(item.get("drug"), Mapping) else {}
    return {
        "drug_name": drug.get("name"),
        "drug_id": drug.get("conceptId"),
        "interaction_score": item.get("interactionScore"),
        "interaction_types": item.get("interactionTypes", []),
        "pmids": [
            publication.get("pmid")
            for publication in item.get("publications", [])
            if isinstance(publication, Mapping) and publication.get("pmid")
        ],
        "source_databases": [
            source.get("sourceDbName")
            for source in item.get("sources", [])
            if isinstance(source, Mapping) and source.get("sourceDbName")
        ],
    }


__all__ = [
    "DGIDB_GRAPHQL_URL",
    "EUROPE_PMC_SEARCH_URL",
    "OPEN_TARGETS_GRAPHQL_URL",
    "PHAROS_GRAPHQL_URL",
    "PUBMED_SEARCH_URL",
    "PUBMED_SUMMARY_URL",
    "PublicBiologyRetriever",
    "UNIPROT_SEARCH_URL",
]
