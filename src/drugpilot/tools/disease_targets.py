"""Disease-agnostic, keyless target discovery for MA1.

This module is intentionally independent from the current MA1/graph wiring.  Its
single integration boundary is :meth:`DiseaseTargetDiscovery.discover`, which
accepts a disease label plus an optional ontology identifier and returns a
strict, provenance-rich :class:`DiseaseTargetDiscoveryResult`.

The implementation uses two read-only public APIs and no LLM:

* Open Targets GraphQL resolves a disease and returns its ranked
  ``associatedTargets`` rows.  Open Targets association scores are retained as
  ranking heuristics; they are *not* represented as probabilities or proof of
  causality.
* PubMed E-utilities returns publication metadata.  Explicit gene-like tokens
  in titles may add literature context to an Open Targets candidate.  A
  PubMed-only candidate is admitted only after an exact Open Targets target
  lookup supplies its approved symbol and Ensembl target identifier.  Such a
  candidate remains unscored and is labelled as a co-mention, never as a
  causal association.

HTTP is injectable, caching is per remote operation, and failures are isolated
per source.  The module never emits fixture/fallback candidates; MA1 may apply
its own explicit fallback policy after inspecting the result.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from re import compile as compile_regex
from threading import Lock
from time import monotonic, sleep
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence
from unicodedata import normalize as unicode_normalize

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drugpilot.tools.http import get_scientific_http_client
from drugpilot.tools.ncbi_rate_limit import NCBIRateLimiter


OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

SourceName = Literal["Open Targets", "PubMed"]
RetrievalMode = Literal["live", "cache", "mixed"]
SourceState = Literal["live", "cache", "partial", "empty", "failed", "skipped"]
TrackHint = Literal["literature", "tbio"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DiseaseMatchProvenance(_StrictModel):
    matched_label: str = Field(min_length=1)
    matched_label_kind: Literal["preferred_name", "exact_synonym"]
    normalization_method: Literal["nfkc_casefold_alphanumeric_tokens_v1"] = (
        "nfkc_casefold_alphanumeric_tokens_v1"
    )
    normalized_query: str = Field(min_length=1)
    normalized_matched_label: str = Field(min_length=1)
    similarity_method: Literal[
        "normalized_exact",
        "sequence_matcher_token_guard_v1",
    ]
    similarity_score: float = Field(ge=0.0, le=1.0)
    runner_up_similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    acceptance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_margin: float | None = Field(default=None, ge=0.0, le=1.0)


class DiseaseResolution(_StrictModel):
    disease_id: str = Field(min_length=1)
    disease_name: str = Field(min_length=1)
    description: str | None = None
    source: Literal["Open Targets"] = "Open Targets"
    match_method: Literal[
        "provided_id",
        "exact_name",
        "exact_synonym",
        "unique_high_similarity",
        "top_search_hit",
    ]
    retrieval_mode: RetrievalMode
    match_provenance: DiseaseMatchProvenance | None = None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_high_similarity_authority(self) -> "DiseaseResolution":
        if self.match_method != "unique_high_similarity":
            return self
        provenance = self.match_provenance
        if (
            provenance is None
            or provenance.similarity_method != "sequence_matcher_token_guard_v1"
            or provenance.acceptance_threshold is None
            or provenance.minimum_margin is None
            or provenance.similarity_score < provenance.acceptance_threshold
            or not self.warnings
        ):
            raise ValueError(
                "high-similarity disease resolution requires passing provenance and a warning"
            )
        if (
            provenance.runner_up_similarity_score is not None
            and provenance.similarity_score - provenance.runner_up_similarity_score
            < provenance.minimum_margin
        ):
            raise ValueError(
                "high-similarity disease resolution must exceed the runner-up margin"
            )
        return self


class DataTypeScore(_StrictModel):
    evidence_type: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class TractabilityAssessment(_StrictModel):
    label: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    value: bool


class TargetSourceRecord(_StrictModel):
    source: SourceName
    record_id: str = Field(min_length=1)
    record_type: Literal[
        "target_disease_association",
        "publication_metadata",
        "target_identity_resolution",
    ]
    title: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    relationship: Literal[
        "ranked_association",
        "co_mention_context_only",
        "identity_mapping_only",
    ]
    query: str = Field(min_length=1)
    retrieval_endpoint: str = Field(pattern=r"^https://")
    retrieval_mode: RetrievalMode
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssociationEvidenceSummary(_StrictModel):
    overall_association_score: float | None = Field(default=None, ge=0.0, le=1.0)
    datatype_scores: tuple[DataTypeScore, ...] = ()
    # Open Targets 26.03 exposes intervention-stage evidence as ``clinical``.
    # Keep it separate from the legacy ``known_drug`` field so provenance is
    # explicit and schema evolution cannot silently change scientific meaning.
    clinical_evidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    known_drug_score: float | None = Field(default=None, ge=0.0, le=1.0)
    tractability: tuple[TractabilityAssessment, ...] = ()
    pubmed_record_count: int = Field(default=0, ge=0)
    score_interpretation: str = Field(min_length=1)


class TargetProvenance(_StrictModel):
    discovery_method: Literal[
        "opentargets_ranked_association",
        "pubmed_explicit_symbol_co_mention",
    ]
    primary_source: SourceName
    contributing_sources: tuple[SourceName, ...]
    causality_interpretation: Literal[
        "association_score_is_not_causal_proof",
        "co_mention_only_no_causality_inferred",
    ]
    fallback_used: Literal[False] = False


class DiseaseTargetCandidate(_StrictModel):
    rank: int = Field(ge=1)
    gene_symbol: str = Field(min_length=1)
    ensembl_target_id: str = Field(pattern=r"^ENSG[0-9]+$")
    target_name: str | None = None
    biotype: str | None = None
    association_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_summary: AssociationEvidenceSummary
    source_records: tuple[TargetSourceRecord, ...]
    provenance: TargetProvenance
    track_hint: TrackHint | None = None
    track_hint_basis: str | None = None


class SourceOperationMetric(_StrictModel):
    operation: str = Field(min_length=1)
    success: bool
    cache_hit: bool
    operation_records_observed: int = Field(ge=0)
    unique_records_observed: int = Field(ge=0)
    records_screened: int = Field(ge=0)
    failure_code: str | None = None


class SourceStatus(_StrictModel):
    source: SourceName
    status: SourceState
    operations: tuple[str, ...]
    # Retained for backward compatibility. This is the sum of observations
    # made by individual HTTP operations and can double-count the same record
    # (for example PubMed ESearch followed by ESummary).
    records_observed: int = Field(ge=0)
    operation_records_observed: int | None = Field(default=None, ge=0)
    unique_records_observed: int | None = Field(default=None, ge=0)
    records_screened: int | None = Field(default=None, ge=0)
    operation_metrics: tuple[SourceOperationMetric, ...] = ()
    cache_hits: int = Field(ge=0)
    failure_codes: tuple[str, ...] = ()


class DiseaseTargetDiscoveryResult(_StrictModel):
    query_disease: str = Field(min_length=1)
    requested_disease_id: str | None = None
    resolved_disease: DiseaseResolution | None = None
    candidates: tuple[DiseaseTargetCandidate, ...]
    source_statuses: tuple[SourceStatus, ...]
    association_total: int | None = Field(default=None, ge=0)
    max_targets: int = Field(ge=1, le=50)
    fallback_used: Literal[False] = False
    interpretation_notes: tuple[str, ...] = (
        "Open Targets association scores are ranking heuristics, not probabilities or causal proof.",
        "PubMed title co-mentions provide literature context only and do not establish causality.",
    )


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class _CacheEntry:
    stored_at: float
    payload: Any


@dataclass(frozen=True)
class _FetchResult:
    payload: Any | None
    mode: Literal["live", "cache"] | None


@dataclass(frozen=True)
class _OperationOutcome:
    source: SourceName
    operation: str
    success: bool
    cache_hit: bool
    records_observed: int
    failure_code: str | None = None
    record_identities: tuple[str, ...] = ()
    records_screened: int = 0


@dataclass(frozen=True)
class _DiseaseLabelCandidate:
    node: Mapping[str, Any]
    label: str
    label_kind: Literal["preferred_name", "exact_synonym"]
    similarity_score: float


@dataclass
class _CandidateDraft:
    gene_symbol: str
    ensembl_target_id: str
    target_name: str | None
    biotype: str | None
    association_score: float | None
    datatype_scores: tuple[DataTypeScore, ...]
    tractability: tuple[TractabilityAssessment, ...]
    source_records: list[TargetSourceRecord]
    discovery_method: Literal[
        "opentargets_ranked_association",
        "pubmed_explicit_symbol_co_mention",
    ]


_GENE_TOKEN = compile_regex(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,9})(?![A-Za-z0-9])")
_GENE_TOKEN_STOPWORDS = frozenset(
    {
        "ABSTRACT",
        "ANALYSIS",
        "AND",
        "CANCER",
        "CELL",
        "CELLS",
        "DNA",
        "DISEASE",
        "GENE",
        "GENES",
        "HUMAN",
        "MRNA",
        "NOVEL",
        "OR",
        "PATHWAY",
        "PROTEIN",
        "PROTEINS",
        "RNA",
        "ROLE",
        "SIGNAL",
        "SIGNALING",
        "STUDY",
        "TARGET",
        "TARGETS",
        "THE",
        "TUMOR",
        "TUMOUR",
    }
)
_TBIO_EVIDENCE_TYPES = frozenset(
    {
        "affected_pathway",
        "animal_model",
        "genetic_association",
        "rna_expression",
        "somatic_mutation",
    }
)
_LITERATURE_EVIDENCE_TYPES = frozenset({"literature", "text_mining"})
_FUZZY_DISEASE_MATCH_THRESHOLD = 0.88
_FUZZY_DISEASE_MATCH_MINIMUM_MARGIN = 0.08


class DiseaseTargetDiscovery:
    """Resolve a disease and return deterministic, evidence-traceable targets.

    The class does not implement an MA1 protocol on purpose.  A caller can map
    each ``DiseaseTargetCandidate`` into its own seed type after reviewing
    ``source_statuses`` and ``fallback_used``.
    """

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 8.0,
        max_targets: int = 10,
        pubmed_max_records: int = 20,
        max_pubmed_supplements: int = 3,
        target_lookup_limit: int = 8,
        cache_ttl_seconds: float = 3_600.0,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.15,
        ncbi_api_key: str | None = None,
        ncbi_tool: str = "DrugPilot",
        ncbi_email: str | None = None,
        ncbi_rate_limiter: NCBIRateLimiter | None = None,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= max_targets <= 50:
            raise ValueError("max_targets must be between 1 and 50")
        if not 1 <= pubmed_max_records <= 100:
            raise ValueError("pubmed_max_records must be between 1 and 100")
        if not 0 <= max_pubmed_supplements <= 20:
            raise ValueError("max_pubmed_supplements must be between 0 and 20")
        if not 0 <= target_lookup_limit <= 30:
            raise ValueError("target_lookup_limit must be between 0 and 30")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must not be negative")
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")

        self.client = client or get_scientific_http_client()
        self.timeout_seconds = timeout_seconds
        self.max_targets = max_targets
        self.pubmed_max_records = pubmed_max_records
        self.max_pubmed_supplements = max_pubmed_supplements
        self.target_lookup_limit = target_lookup_limit
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.ncbi_api_key = ncbi_api_key.strip() if ncbi_api_key else None
        self.ncbi_tool = ncbi_tool.strip() or "DrugPilot"
        self.ncbi_email = ncbi_email.strip() if ncbi_email else None
        self._clock = clock
        self._sleeper = sleeper
        self._ncbi_rate_limiter = ncbi_rate_limiter or NCBIRateLimiter(
            clock=clock,
            sleeper=sleeper,
        )
        self._cache: dict[tuple[str, str, str], _CacheEntry] = {}
        self._cache_lock = Lock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def discover(
        self,
        disease: str,
        disease_id: str | None = None,
        *,
        max_targets: int | None = None,
        allowed_sources: Sequence[str] | None = None,
    ) -> DiseaseTargetDiscoveryResult:
        """Discover targets without fixtures or LLM inference.

        Remote failures are converted to ``SourceStatus`` entries.  Invalid
        local inputs still raise ``ValueError`` because they are caller errors.
        """

        query_disease = " ".join(disease.split())
        if not query_disease:
            raise ValueError("disease must not be blank")
        requested_id = disease_id.strip() if disease_id and disease_id.strip() else None
        limit = self.max_targets if max_targets is None else max_targets
        if not 1 <= limit <= 50:
            raise ValueError("max_targets must be between 1 and 50")

        allowed = (
            frozenset(("Open Targets", "PubMed"))
            if allowed_sources is None
            else frozenset(allowed_sources)
        )
        outcomes: list[_OperationOutcome] = []
        resolution = (
            self._resolve_disease(query_disease, requested_id, outcomes)
            if "Open Targets" in allowed
            else None
        )
        association_total: int | None = None
        drafts: dict[str, _CandidateDraft] = {}

        if resolution is not None:
            association_payload = self._safe_cached_fetch(
                source="Open Targets",
                operation="associated_targets",
                cache_identity=f"{resolution.disease_id.casefold()}|{limit}",
                outcomes=outcomes,
                fetch=lambda: self._fetch_associated_targets(resolution.disease_id, limit),
                record_counter=_association_row_count,
            )
            if association_payload.payload is not None:
                association_total, drafts = self._parse_associations(
                    association_payload.payload,
                    resolution,
                    association_payload.mode or "live",
                )

        publications = (
            self._retrieve_pubmed(query_disease, outcomes)
            if "PubMed" in allowed
            else []
        )
        # PubMed-only target admission still requires an exact Open Targets
        # identity.  Do not bypass the caller's policy by resolving a title
        # token through Open Targets when that source is forbidden.
        if publications and "Open Targets" in allowed:
            self._attach_pubmed_context(
                query_disease=query_disease,
                publications=publications,
                drafts=drafts,
                outcomes=outcomes,
                candidate_limit=limit,
            )

        candidates = self._finalize_candidates(drafts, limit)
        return DiseaseTargetDiscoveryResult(
            query_disease=query_disease,
            requested_disease_id=requested_id,
            resolved_disease=resolution,
            candidates=candidates,
            source_statuses=self._summarize_statuses(
                outcomes,
                allowed_sources=allowed,
            ),
            association_total=association_total,
            max_targets=limit,
        )

    def _resolve_disease(
        self,
        disease: str,
        disease_id: str | None,
        outcomes: list[_OperationOutcome],
    ) -> DiseaseResolution | None:
        if disease_id:
            fetched = self._safe_cached_fetch(
                source="Open Targets",
                operation="resolve_disease_id",
                cache_identity=disease_id.casefold(),
                outcomes=outcomes,
                fetch=lambda: self._fetch_disease_by_id(disease_id),
                record_counter=lambda payload: int(
                    isinstance(payload.get("data", {}).get("disease"), Mapping)
                ),
            )
            disease_node = _graphql_data(fetched.payload, "disease")
            if disease_node is None:
                return None
            return _disease_resolution(
                disease_node,
                match_method="provided_id",
                mode=fetched.mode or "live",
            )

        fetched = self._safe_cached_fetch(
            source="Open Targets",
            operation="search_disease",
            cache_identity=_normalized_disease_label(disease),
            outcomes=outcomes,
            fetch=lambda: self._search_disease(disease),
            record_counter=_search_hit_count,
        )
        hits = _search_hits(fetched.payload, entity="disease")
        search_mode = fetched.mode or "live"
        relaxed_search_used = False
        if not hits:
            relaxed_search_used = True
            relaxed_hits: dict[str, Mapping[str, Any]] = {}
            relaxed_modes: list[Literal["live", "cache"]] = []
            for relaxed_query in _relaxed_disease_search_queries(disease):
                relaxed_fetch = self._safe_cached_fetch(
                    source="Open Targets",
                    operation="search_disease_relaxed",
                    cache_identity=_normalized_disease_label(relaxed_query),
                    outcomes=outcomes,
                    fetch=lambda query=relaxed_query: self._search_disease(
                        query,
                        size=20,
                    ),
                    record_counter=_search_hit_count,
                )
                if relaxed_fetch.mode is not None:
                    relaxed_modes.append(relaxed_fetch.mode)
                for hit in _search_hits(relaxed_fetch.payload, entity="disease"):
                    identifier = str(hit.get("id") or "").strip()
                    if identifier and identifier not in relaxed_hits:
                        relaxed_hits[identifier] = hit
                    if len(relaxed_hits) >= 25:
                        break
                if len(relaxed_hits) >= 25:
                    break
            hits = list(relaxed_hits.values())
            if not hits:
                return None
            search_mode = (
                "cache"
                if fetched.mode == "cache"
                and relaxed_modes
                and all(mode == "cache" for mode in relaxed_modes)
                else "live"
            )
        query_key = _normalized_disease_label(disease)
        exact_by_id = {
            identifier: hit
            for hit in hits
            if (identifier := str(hit.get("id") or "").strip())
            and _normalized_disease_label(str(hit.get("name") or "")) == query_key
        }
        if len(exact_by_id) == 1:
            exact = next(iter(exact_by_id.values()))
            matched_label = str(exact.get("name") or "").strip()
            return _disease_resolution(
                exact,
                match_method="exact_name",
                mode=search_mode,
                match_provenance=_disease_match_provenance(
                    disease,
                    matched_label,
                    label_kind="preferred_name",
                    similarity_method="normalized_exact",
                    similarity_score=1.0,
                ),
            )
        if len(exact_by_id) > 1:
            return None

        # Search rank alone is not disease identity evidence.  Verify only the
        # bounded set of returned ontology IDs against Open Targets' own
        # ``hasExactSynonym`` annotations.  Related/narrow synonyms and fuzzy
        # top hits remain unresolved, and an ambiguous exact synonym fails
        # closed instead of silently selecting the first result.
        candidate_ids = tuple(
            dict.fromkeys(
                identifier
                for hit in hits[:25 if relaxed_search_used else 5]
                if (identifier := str(hit.get("id") or "").strip())
            )
        )
        if not candidate_ids:
            return None
        synonym_fetch = self._safe_cached_fetch(
            source="Open Targets",
            operation="verify_exact_disease_synonyms",
            cache_identity=(
                f"{query_key}|"
                + ",".join(identifier.casefold() for identifier in candidate_ids)
            ),
            outcomes=outcomes,
            fetch=lambda: self._fetch_disease_synonyms(candidate_ids),
            record_counter=_disease_list_count,
        )
        exact_synonym_nodes = _exact_synonym_matches(
            synonym_fetch.payload,
            disease,
            allowed_ids=frozenset(candidate_ids),
        )
        if len(exact_synonym_nodes) != 1:
            fuzzy_match, runner_up_score = _unique_high_similarity_match(
                synonym_fetch.payload,
                disease,
                allowed_ids=frozenset(candidate_ids),
            )
            if fuzzy_match is None:
                return None
            return _disease_resolution(
                fuzzy_match.node,
                match_method="unique_high_similarity",
                mode=synonym_fetch.mode or "live",
                match_provenance=_disease_match_provenance(
                    disease,
                    fuzzy_match.label,
                    label_kind=fuzzy_match.label_kind,
                    similarity_method="sequence_matcher_token_guard_v1",
                    similarity_score=fuzzy_match.similarity_score,
                    runner_up_similarity_score=runner_up_score,
                    acceptance_threshold=_FUZZY_DISEASE_MATCH_THRESHOLD,
                    minimum_margin=_FUZZY_DISEASE_MATCH_MINIMUM_MARGIN,
                ),
                warnings=(
                    "The input disease label was conservatively typo-corrected to a unique "
                    "high-similarity Open Targets preferred name or exact synonym.",
                ),
            )
        exact_synonym_node = exact_synonym_nodes[0]
        matched_synonym = _matching_exact_synonym_label(
            exact_synonym_node,
            disease,
        )
        return _disease_resolution(
            exact_synonym_node,
            match_method="exact_synonym",
            mode=synonym_fetch.mode or "live",
            match_provenance=_disease_match_provenance(
                disease,
                matched_synonym or disease,
                label_kind="exact_synonym",
                similarity_method="normalized_exact",
                similarity_score=1.0,
            ),
        )

    def _fetch_disease_by_id(self, disease_id: str) -> Mapping[str, Any]:
        query = """
        query ResolveDiseaseId($diseaseId: String!) {
          disease(efoId: $diseaseId) { id name description }
        }
        """
        return self._post_graphql(query, {"diseaseId": disease_id})

    def _fetch_disease_synonyms(
        self,
        disease_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        query = """
        query ResolveDiseaseSynonyms($diseaseIds: [String!]!) {
          diseases(efoIds: $diseaseIds) {
            id
            name
            description
            synonyms { relation terms }
          }
        }
        """
        return self._post_graphql(query, {"diseaseIds": list(disease_ids)})

    def _search_disease(self, disease: str, *, size: int = 5) -> Mapping[str, Any]:
        query = """
        query SearchDisease($query: String!, $size: Int!) {
          search(
            queryString: $query
            entityNames: ["disease"]
            page: {index: 0, size: $size}
          ) { hits { id name entity description } }
        }
        """
        return self._post_graphql(query, {"query": disease, "size": size})

    def _fetch_associated_targets(self, disease_id: str, limit: int) -> Mapping[str, Any]:
        query = """
        query DiseaseAssociatedTargets($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(page: {index: 0, size: $size}) {
              count
              rows {
                score
                datatypeScores { id score }
                target {
                  id
                  approvedSymbol
                  approvedName
                  biotype
                  tractability { label modality value }
                }
              }
            }
          }
        }
        """
        return self._post_graphql(query, {"diseaseId": disease_id, "size": limit})

    def _post_graphql(self, query: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self.client.post(
            OPEN_TARGETS_GRAPHQL_URL,
            json={"query": query, "variables": dict(variables)},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("Open Targets returned a non-object payload")
        if payload.get("errors"):
            raise ValueError("Open Targets returned GraphQL errors")
        return payload

    def _parse_associations(
        self,
        payload: Mapping[str, Any],
        resolution: DiseaseResolution,
        mode: Literal["live", "cache"],
    ) -> tuple[int | None, dict[str, _CandidateDraft]]:
        disease_node = _graphql_data(payload, "disease")
        if disease_node is None:
            return None, {}
        associations = disease_node.get("associatedTargets")
        if not isinstance(associations, Mapping):
            return None, {}
        raw_count = associations.get("count")
        total = int(raw_count) if isinstance(raw_count, int) and raw_count >= 0 else None
        rows = associations.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return total, {}

        drafts: dict[str, _CandidateDraft] = {}
        for row in rows:
            draft = self._association_draft(row, resolution, mode)
            if draft is None:
                continue
            key = draft.ensembl_target_id
            current = drafts.get(key)
            if current is None or (draft.association_score or 0.0) > (
                current.association_score or 0.0
            ):
                drafts[key] = draft
        return total, drafts

    def _association_draft(
        self,
        row: Any,
        resolution: DiseaseResolution,
        mode: Literal["live", "cache"],
    ) -> _CandidateDraft | None:
        if not isinstance(row, Mapping):
            return None
        target = row.get("target")
        if not isinstance(target, Mapping):
            return None
        ensembl_id = str(target.get("id") or "").strip()
        symbol = str(target.get("approvedSymbol") or "").strip()
        score = _unit_score(row.get("score"))
        if not ensembl_id.startswith("ENSG") or not ensembl_id[4:].isdigit() or not symbol:
            return None
        if score is None:
            return None

        datatype_scores = _parse_datatype_scores(row.get("datatypeScores"))
        tractability = _parse_tractability(target.get("tractability"))
        query = f"disease(efoId: {resolution.disease_id}) -> associatedTargets"
        source_record = TargetSourceRecord(
            source="Open Targets",
            record_id=f"{resolution.disease_id}:{ensembl_id}",
            record_type="target_disease_association",
            title=f"{symbol} association with {resolution.disease_name}",
            url=(
                "https://platform.opentargets.org/evidence/"
                f"{ensembl_id}/{resolution.disease_id}"
            ),
            relationship="ranked_association",
            query=query,
            retrieval_endpoint=OPEN_TARGETS_GRAPHQL_URL,
            retrieval_mode=mode,
            score=score,
            metadata={
                "disease_id": resolution.disease_id,
                "disease_name": resolution.disease_name,
                "score_scope": "Open Targets overall association ranking heuristic",
            },
        )
        return _CandidateDraft(
            gene_symbol=symbol,
            ensembl_target_id=ensembl_id,
            target_name=_optional_text(target.get("approvedName")),
            biotype=_optional_text(target.get("biotype")),
            association_score=score,
            datatype_scores=datatype_scores,
            tractability=tractability,
            source_records=[source_record],
            discovery_method="opentargets_ranked_association",
        )

    def _retrieve_pubmed(
        self,
        disease: str,
        outcomes: list[_OperationOutcome],
    ) -> list[tuple[Mapping[str, Any], RetrievalMode]]:
        query = self._pubmed_query(disease)
        common = self._ncbi_common_params()
        search = self._safe_cached_fetch(
            source="PubMed",
            operation="search_publications",
            cache_identity=query,
            outcomes=outcomes,
            fetch=lambda: self._pubmed_search(query, common),
            record_counter=_pubmed_search_count,
        )
        ids = _pubmed_ids(search.payload)
        if not ids:
            return []
        summary = self._safe_cached_fetch(
            source="PubMed",
            operation="publication_metadata",
            cache_identity=",".join(ids),
            outcomes=outcomes,
            fetch=lambda: self._pubmed_summary(ids, common),
            record_counter=_pubmed_summary_count,
        )
        if summary.payload is None:
            return []
        retrieval_mode: RetrievalMode
        if search.mode == summary.mode:
            retrieval_mode = search.mode or "live"
        else:
            retrieval_mode = "mixed"
        return [(item, retrieval_mode) for item in _pubmed_items(summary.payload, ids)]

    def _pubmed_search(
        self,
        query: str,
        common_params: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._ncbi_rate_limiter.acquire(
            api_key_configured=bool(self.ncbi_api_key)
        )
        response = self.client.get(
            PUBMED_SEARCH_URL,
            params={
                **common_params,
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": self.pubmed_max_records,
                "sort": "relevance",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("PubMed ESearch returned a non-object payload")
        return payload

    def _pubmed_summary(
        self,
        ids: Sequence[str],
        common_params: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._ncbi_rate_limiter.acquire(
            api_key_configured=bool(self.ncbi_api_key)
        )
        response = self.client.get(
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
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("PubMed ESummary returned a non-object payload")
        return payload

    def _attach_pubmed_context(
        self,
        *,
        query_disease: str,
        publications: Sequence[tuple[Mapping[str, Any], RetrievalMode]],
        drafts: dict[str, _CandidateDraft],
        outcomes: list[_OperationOutcome],
        candidate_limit: int,
    ) -> None:
        by_symbol = {draft.gene_symbol.casefold(): draft for draft in drafts.values()}
        tokens_to_publications: dict[str, list[tuple[Mapping[str, Any], RetrievalMode]]] = {}
        for publication, mode in publications:
            title = str(publication.get("title") or "")
            for token in _extract_gene_tokens(title):
                tokens_to_publications.setdefault(token, []).append((publication, mode))

        for symbol_key, draft in sorted(by_symbol.items()):
            matches = tokens_to_publications.get(draft.gene_symbol.upper(), [])
            for publication, mode in matches:
                draft.source_records.append(
                    self._pubmed_record(query_disease, draft.gene_symbol, publication, mode)
                )

        if self.max_pubmed_supplements == 0 or len(drafts) >= candidate_limit:
            return
        remaining = min(self.max_pubmed_supplements, candidate_limit - len(drafts))
        attempted = 0
        added = 0
        existing_symbols = set(by_symbol)
        ordered_tokens = sorted(
            tokens_to_publications,
            key=lambda token: (-len(tokens_to_publications[token]), token),
        )
        for token in ordered_tokens:
            if added >= remaining or attempted >= self.target_lookup_limit:
                break
            if token.casefold() in existing_symbols:
                continue
            attempted += 1
            resolved = self._resolve_target_symbol(token, outcomes)
            if resolved is None:
                continue
            ensembl_id, approved_symbol, target_name, mode = resolved
            if ensembl_id in drafts or approved_symbol.casefold() in existing_symbols:
                continue
            identity_record = TargetSourceRecord(
                source="Open Targets",
                record_id=ensembl_id,
                record_type="target_identity_resolution",
                title=f"Open Targets target identity: {approved_symbol}",
                url=f"https://platform.opentargets.org/target/{ensembl_id}",
                relationship="identity_mapping_only",
                query=token,
                retrieval_endpoint=OPEN_TARGETS_GRAPHQL_URL,
                retrieval_mode=mode,
                metadata={
                    "identity_match": "exact approved symbol",
                    "association_not_asserted": True,
                },
            )
            pubmed_records = [
                self._pubmed_record(query_disease, approved_symbol, publication, pubmed_mode)
                for publication, pubmed_mode in tokens_to_publications[token]
            ]
            drafts[ensembl_id] = _CandidateDraft(
                gene_symbol=approved_symbol,
                ensembl_target_id=ensembl_id,
                target_name=target_name,
                biotype=None,
                association_score=None,
                datatype_scores=(),
                tractability=(),
                source_records=[identity_record, *pubmed_records],
                discovery_method="pubmed_explicit_symbol_co_mention",
            )
            existing_symbols.add(approved_symbol.casefold())
            added += 1

    def _resolve_target_symbol(
        self,
        token: str,
        outcomes: list[_OperationOutcome],
    ) -> tuple[str, str, str | None, Literal["live", "cache"]] | None:
        fetched = self._safe_cached_fetch(
            source="Open Targets",
            operation="resolve_pubmed_symbol",
            cache_identity=token.casefold(),
            outcomes=outcomes,
            fetch=lambda: self._search_target(token),
            record_counter=_search_hit_count,
        )
        hits = _search_hits(fetched.payload, entity="target")
        exact = next(
            (hit for hit in hits if str(hit.get("name", "")).casefold() == token.casefold()),
            None,
        )
        if exact is None:
            return None
        ensembl_id = str(exact.get("id") or "").strip()
        approved_symbol = str(exact.get("name") or "").strip()
        if not ensembl_id.startswith("ENSG") or not ensembl_id[4:].isdigit():
            return None
        return (
            ensembl_id,
            approved_symbol,
            _optional_text(exact.get("description")),
            fetched.mode or "live",
        )

    def _search_target(self, symbol: str) -> Mapping[str, Any]:
        query = """
        query SearchTarget($query: String!, $size: Int!) {
          search(
            queryString: $query
            entityNames: ["target"]
            page: {index: 0, size: $size}
          ) { hits { id name entity description } }
        }
        """
        return self._post_graphql(query, {"query": symbol, "size": 5})

    def _pubmed_record(
        self,
        disease: str,
        symbol: str,
        item: Mapping[str, Any],
        mode: RetrievalMode,
    ) -> TargetSourceRecord:
        pmid = str(item.get("uid") or item.get("pmid") or "").strip()
        title = str(item.get("title") or f"PubMed record {pmid}").strip()
        authors = [
            str(author.get("name"))
            for author in item.get("authors", [])
            if isinstance(author, Mapping) and author.get("name")
        ]
        return TargetSourceRecord(
            source="PubMed",
            record_id=pmid,
            record_type="publication_metadata",
            title=title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            relationship="co_mention_context_only",
            query=self._pubmed_query(disease),
            retrieval_endpoint=PUBMED_SUMMARY_URL,
            retrieval_mode=mode,
            metadata={
                "explicit_symbol": symbol,
                "publication_date": item.get("pubdate"),
                "journal": item.get("fulljournalname") or item.get("source"),
                "authors": authors,
                "publication_types": item.get("pubtype", []),
                "evidence_scope": "title co-mention only; no causality inferred",
            },
        )

    def _finalize_candidates(
        self,
        drafts: Mapping[str, _CandidateDraft],
        limit: int,
    ) -> tuple[DiseaseTargetCandidate, ...]:
        ordered = sorted(
            drafts.values(),
            key=lambda draft: (
                draft.association_score is None,
                -(draft.association_score or 0.0),
                -_pubmed_record_count(draft.source_records),
                draft.gene_symbol.casefold(),
                draft.ensembl_target_id,
            ),
        )[:limit]
        result: list[DiseaseTargetCandidate] = []
        for rank, draft in enumerate(ordered, start=1):
            pubmed_count = _pubmed_record_count(draft.source_records)
            known_drug_score = next(
                (
                    item.score
                    for item in draft.datatype_scores
                    if item.evidence_type == "known_drug"
                ),
                None,
            )
            clinical_evidence_score = next(
                (
                    item.score
                    for item in draft.datatype_scores
                    if item.evidence_type == "clinical"
                ),
                None,
            )
            track_hint, track_basis = _track_hint(draft)
            is_pubmed_only = draft.discovery_method == "pubmed_explicit_symbol_co_mention"
            result.append(
                DiseaseTargetCandidate(
                    rank=rank,
                    gene_symbol=draft.gene_symbol,
                    ensembl_target_id=draft.ensembl_target_id,
                    target_name=draft.target_name,
                    biotype=draft.biotype,
                    association_score=draft.association_score,
                    evidence_summary=AssociationEvidenceSummary(
                        overall_association_score=draft.association_score,
                        datatype_scores=draft.datatype_scores,
                        clinical_evidence_score=clinical_evidence_score,
                        known_drug_score=known_drug_score,
                        tractability=draft.tractability,
                        pubmed_record_count=pubmed_count,
                        score_interpretation=(
                            "No association score is available; the candidate is retained only "
                            "as a traceable PubMed title co-mention."
                            if is_pubmed_only
                            else "Open Targets score is a ranking heuristic, not a probability "
                            "or causal conclusion."
                        ),
                    ),
                    source_records=tuple(draft.source_records),
                    provenance=TargetProvenance(
                        discovery_method=draft.discovery_method,
                        primary_source="PubMed" if is_pubmed_only else "Open Targets",
                        contributing_sources=_contributing_sources(draft.source_records),
                        causality_interpretation=(
                            "co_mention_only_no_causality_inferred"
                            if is_pubmed_only
                            else "association_score_is_not_causal_proof"
                        ),
                    ),
                    track_hint=track_hint,
                    track_hint_basis=track_basis,
                )
            )
        return tuple(result)

    def _safe_cached_fetch(
        self,
        *,
        source: SourceName,
        operation: str,
        cache_identity: str,
        outcomes: list[_OperationOutcome],
        fetch: Callable[[], Mapping[str, Any]],
        record_counter: Callable[[Mapping[str, Any]], int],
    ) -> _FetchResult:
        key = (source, operation, cache_identity)
        cached = self._cache_get(key)
        if cached is not None:
            count = _safe_record_count(record_counter, cached)
            outcomes.append(
                _OperationOutcome(
                    source=source,
                    operation=operation,
                    success=True,
                    cache_hit=True,
                    records_observed=count,
                    record_identities=_operation_record_identities(
                        source,
                        operation,
                        cached,
                    ),
                    records_screened=_operation_records_screened(
                        source,
                        operation,
                        count,
                    ),
                )
            )
            return _FetchResult(cached, "cache")

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = fetch()
                self._cache_put(key, payload)
                count = _safe_record_count(record_counter, payload)
                outcomes.append(
                    _OperationOutcome(
                        source=source,
                        operation=operation,
                        success=True,
                        cache_hit=False,
                        records_observed=count,
                        record_identities=_operation_record_identities(
                            source,
                            operation,
                            payload,
                        ),
                        records_screened=_operation_records_screened(
                            source,
                            operation,
                            count,
                        ),
                    )
                )
                return _FetchResult(payload, "live")
            except Exception as exc:  # source/operation failure isolation
                last_error = exc
                if attempt < self.max_attempts and self.retry_backoff_seconds:
                    self._sleeper(self.retry_backoff_seconds * (2 ** (attempt - 1)))
        outcomes.append(
            _OperationOutcome(
                source,
                operation,
                False,
                False,
                0,
                type(last_error).__name__ if last_error is not None else "UnknownError",
            )
        )
        return _FetchResult(None, None)

    def _cache_get(self, key: tuple[str, str, str]) -> Mapping[str, Any] | None:
        now = self._clock()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now - entry.stored_at > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            return deepcopy(entry.payload)

    def _cache_put(self, key: tuple[str, str, str], payload: Mapping[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = _CacheEntry(self._clock(), deepcopy(payload))

    def _summarize_statuses(
        self,
        outcomes: Sequence[_OperationOutcome],
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> tuple[SourceStatus, ...]:
        allowed = (
            frozenset(("Open Targets", "PubMed"))
            if allowed_sources is None
            else frozenset(allowed_sources)
        )
        statuses: list[SourceStatus] = []
        for source in ("Open Targets", "PubMed"):
            relevant = [item for item in outcomes if item.source == source]
            if not relevant:
                statuses.append(
                    SourceStatus(
                        source=source,
                        status="skipped",
                        operations=(),
                        records_observed=0,
                        operation_records_observed=0,
                        unique_records_observed=0,
                        records_screened=0,
                        operation_metrics=(),
                        cache_hits=0,
                        failure_codes=(
                            ("policy_rejected",) if source not in allowed else ()
                        ),
                    )
                )
                continue
            successes = [item for item in relevant if item.success]
            failures = [item for item in relevant if not item.success]
            records = sum(item.records_observed for item in relevant)
            identities = {
                identity
                for item in relevant
                for identity in item.record_identities
            }
            anonymous_unique_records = sum(
                item.records_observed
                for item in relevant
                if item.success
                and item.records_observed
                and not item.record_identities
            )
            unique_records = len(identities) + anonymous_unique_records
            screened_records = sum(item.records_screened for item in relevant)
            operation_metrics = tuple(
                SourceOperationMetric(
                    operation=item.operation,
                    success=item.success,
                    cache_hit=item.cache_hit,
                    operation_records_observed=item.records_observed,
                    unique_records_observed=(
                        len(set(item.record_identities))
                        if item.record_identities
                        else item.records_observed
                        if item.success
                        else 0
                    ),
                    records_screened=item.records_screened,
                    failure_code=item.failure_code,
                )
                for item in relevant
            )
            if failures and successes:
                status: SourceState = "partial"
            elif failures:
                status = "failed"
            elif all(item.cache_hit for item in relevant):
                status = "cache"
            elif records == 0:
                status = "empty"
            else:
                status = "live"
            statuses.append(
                SourceStatus(
                    source=source,
                    status=status,
                    operations=tuple(item.operation for item in relevant),
                    records_observed=records,
                    operation_records_observed=records,
                    unique_records_observed=unique_records,
                    records_screened=screened_records,
                    operation_metrics=operation_metrics,
                    cache_hits=sum(item.cache_hit for item in relevant),
                    failure_codes=tuple(
                        item.failure_code for item in failures if item.failure_code
                    ),
                )
            )
        return tuple(statuses)

    def _ncbi_common_params(self) -> dict[str, str]:
        params = {"tool": self.ncbi_tool}
        if self.ncbi_email:
            params["email"] = self.ncbi_email
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        return params

    @staticmethod
    def _pubmed_query(disease: str) -> str:
        escaped = " ".join(disease.replace('"', " ").split())
        return (
            f'"{escaped}"[Title/Abstract] AND '
            "(gene[Title/Abstract] OR protein[Title/Abstract] OR target[Title/Abstract])"
        )


def _graphql_data(payload: Any, field: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None
    value = data.get(field)
    return value if isinstance(value, Mapping) else None


def _search_hits(payload: Any, *, entity: str) -> list[Mapping[str, Any]]:
    search = _graphql_data(payload, "search")
    if search is None:
        return []
    raw_hits = search.get("hits")
    if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, (str, bytes)):
        return []
    return [
        hit
        for hit in raw_hits
        if isinstance(hit, Mapping) and str(hit.get("entity", "")).casefold() == entity
    ]


def _search_hit_count(payload: Mapping[str, Any]) -> int:
    search = _graphql_data(payload, "search")
    hits = search.get("hits", []) if search else []
    return len(hits) if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)) else 0


def _disease_nodes(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return []
    diseases = data.get("diseases")
    if not isinstance(diseases, Sequence) or isinstance(diseases, (str, bytes)):
        return []
    return [node for node in diseases if isinstance(node, Mapping)]


def _disease_list_count(payload: Mapping[str, Any]) -> int:
    return len(_disease_nodes(payload))


def _exact_synonym_matches(
    payload: Any,
    query_disease: str,
    *,
    allowed_ids: frozenset[str],
) -> list[Mapping[str, Any]]:
    query_key = _normalized_disease_label(query_disease)
    matches: dict[str, Mapping[str, Any]] = {}
    for node in _disease_nodes(payload):
        identifier = str(node.get("id") or "").strip()
        if not identifier or identifier not in allowed_ids:
            continue
        synonyms = node.get("synonyms")
        if not isinstance(synonyms, Sequence) or isinstance(synonyms, (str, bytes)):
            continue
        for synonym_group in synonyms:
            if not isinstance(synonym_group, Mapping):
                continue
            relation = str(synonym_group.get("relation") or "").strip().casefold()
            if relation != "hasexactsynonym":
                continue
            terms = synonym_group.get("terms")
            if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
                continue
            if any(
                isinstance(term, str)
                and _normalized_disease_label(term) == query_key
                for term in terms
            ):
                matches[identifier] = node
                break
    return list(matches.values())


def _normalized_disease_label(value: str) -> str:
    normalized = unicode_normalize("NFKC", value).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized)
        .split()
    )


def _relaxed_disease_search_queries(value: str) -> tuple[str, ...]:
    """Return a small, deterministic set of unchanged adjacent-token anchors.

    This is deliberately not a local spelling dictionary.  It only asks Open
    Targets for a broader official candidate set when the full label returns
    no hits.  Identity is still decided later from the complete original input
    using the high similarity threshold and runner-up margin.
    """

    tokens = tuple(_normalized_disease_label(value).split())
    if not 3 <= len(tokens) <= 8:
        return ()
    queries = [
        " ".join(tokens[index : index + 2])
        for index in range(len(tokens) - 1)
        if sum(len(token) for token in tokens[index : index + 2]) >= 8
    ]
    return tuple(dict.fromkeys(queries[:6]))


def _exact_synonym_labels(node: Mapping[str, Any]) -> tuple[str, ...]:
    synonyms = node.get("synonyms")
    if not isinstance(synonyms, Sequence) or isinstance(synonyms, (str, bytes)):
        return ()
    labels: list[str] = []
    for synonym_group in synonyms:
        if not isinstance(synonym_group, Mapping):
            continue
        relation = str(synonym_group.get("relation") or "").strip().casefold()
        if relation != "hasexactsynonym":
            continue
        terms = synonym_group.get("terms")
        if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
            continue
        labels.extend(
            term.strip()
            for term in terms
            if isinstance(term, str) and term.strip()
        )
    return tuple(dict.fromkeys(labels))


def _matching_exact_synonym_label(
    node: Mapping[str, Any],
    query_disease: str,
) -> str | None:
    query_key = _normalized_disease_label(query_disease)
    return next(
        (
            label
            for label in _exact_synonym_labels(node)
            if _normalized_disease_label(label) == query_key
        ),
        None,
    )


def _token_guarded_label_similarity(query: str, candidate: str) -> float:
    query_key = _normalized_disease_label(query)
    candidate_key = _normalized_disease_label(candidate)
    query_tokens = tuple(query_key.split())
    candidate_tokens = tuple(candidate_key.split())
    if (
        len(query_tokens) < 2
        or len(candidate_tokens) < 2
        or min(len(query_key.replace(" ", "")), len(candidate_key.replace(" ", ""))) < 10
        or abs(len(query_tokens) - len(candidate_tokens)) > 1
    ):
        return 0.0

    character_score = SequenceMatcher(None, query_key, candidate_key).ratio()

    def directional_token_score(
        left: tuple[str, ...],
        right: tuple[str, ...],
    ) -> float:
        return sum(
            max(SequenceMatcher(None, token, other).ratio() for other in right)
            for token in left
        ) / len(left)

    token_score = min(
        directional_token_score(query_tokens, candidate_tokens),
        directional_token_score(candidate_tokens, query_tokens),
    )
    return min(character_score, token_score)


def _unique_high_similarity_match(
    payload: Any,
    query_disease: str,
    *,
    allowed_ids: frozenset[str],
) -> tuple[_DiseaseLabelCandidate | None, float | None]:
    best_by_id: dict[str, _DiseaseLabelCandidate] = {}
    for node in _disease_nodes(payload):
        identifier = str(node.get("id") or "").strip()
        if not identifier or identifier not in allowed_ids:
            continue
        preferred_name = str(node.get("name") or "").strip()
        labels: tuple[tuple[str, Literal["preferred_name", "exact_synonym"]], ...] = (
            ((preferred_name, "preferred_name"),) if preferred_name else ()
        ) + tuple((label, "exact_synonym") for label in _exact_synonym_labels(node))
        for label, label_kind in labels:
            score = _token_guarded_label_similarity(query_disease, label)
            current = best_by_id.get(identifier)
            if current is None or score > current.similarity_score:
                best_by_id[identifier] = _DiseaseLabelCandidate(
                    node=node,
                    label=label,
                    label_kind=label_kind,
                    similarity_score=score,
                )

    ranked = sorted(
        best_by_id.values(),
        key=lambda item: (-item.similarity_score, str(item.node.get("id") or "")),
    )
    if not ranked:
        return None, None
    best = ranked[0]
    runner_up_score = ranked[1].similarity_score if len(ranked) > 1 else None
    if best.similarity_score < _FUZZY_DISEASE_MATCH_THRESHOLD:
        return None, runner_up_score
    if (
        runner_up_score is not None
        and best.similarity_score - runner_up_score
        < _FUZZY_DISEASE_MATCH_MINIMUM_MARGIN
    ):
        return None, runner_up_score
    return best, runner_up_score


def _disease_match_provenance(
    query: str,
    matched_label: str,
    *,
    label_kind: Literal["preferred_name", "exact_synonym"],
    similarity_method: Literal[
        "normalized_exact",
        "sequence_matcher_token_guard_v1",
    ],
    similarity_score: float,
    runner_up_similarity_score: float | None = None,
    acceptance_threshold: float | None = None,
    minimum_margin: float | None = None,
) -> DiseaseMatchProvenance:
    return DiseaseMatchProvenance(
        matched_label=matched_label,
        matched_label_kind=label_kind,
        normalized_query=_normalized_disease_label(query),
        normalized_matched_label=_normalized_disease_label(matched_label),
        similarity_method=similarity_method,
        similarity_score=float(similarity_score),
        runner_up_similarity_score=(
            float(runner_up_similarity_score)
            if runner_up_similarity_score is not None
            else None
        ),
        acceptance_threshold=acceptance_threshold,
        minimum_margin=minimum_margin,
    )


def _association_row_count(payload: Mapping[str, Any]) -> int:
    disease = _graphql_data(payload, "disease")
    association = disease.get("associatedTargets") if disease else None
    rows = association.get("rows", []) if isinstance(association, Mapping) else []
    return len(rows) if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else 0


def _disease_resolution(
    node: Mapping[str, Any],
    *,
    match_method: Literal[
        "provided_id",
        "exact_name",
        "exact_synonym",
        "unique_high_similarity",
        "top_search_hit",
    ],
    mode: Literal["live", "cache"],
    match_provenance: DiseaseMatchProvenance | None = None,
    warnings: tuple[str, ...] = (),
) -> DiseaseResolution | None:
    identifier = str(node.get("id") or "").strip()
    name = str(node.get("name") or "").strip()
    if not identifier or not name:
        return None
    return DiseaseResolution(
        disease_id=identifier,
        disease_name=name,
        description=_optional_text(node.get("description")),
        match_method=match_method,
        retrieval_mode=mode,
        match_provenance=match_provenance,
        warnings=warnings,
    )


def _unit_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if 0.0 <= score <= 1.0 else None


def _parse_datatype_scores(value: Any) -> tuple[DataTypeScore, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    scores: dict[str, float] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        evidence_type = str(item.get("id") or "").strip()
        score = _unit_score(item.get("score"))
        if not evidence_type or score is None:
            continue
        scores[evidence_type] = max(score, scores.get(evidence_type, 0.0))
    return tuple(
        DataTypeScore(evidence_type=evidence_type, score=score)
        for evidence_type, score in sorted(scores.items())
    )


def _parse_tractability(value: Any) -> tuple[TractabilityAssessment, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    parsed: list[TractabilityAssessment] = []
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("value"), bool):
            continue
        label = str(item.get("label") or "").strip()
        modality = str(item.get("modality") or "").strip()
        if not label or not modality:
            continue
        parsed.append(
            TractabilityAssessment(label=label, modality=modality, value=item["value"])
        )
    return tuple(sorted(parsed, key=lambda item: (item.modality, item.label)))


def _pubmed_ids(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    result = payload.get("esearchresult")
    raw_ids = result.get("idlist", []) if isinstance(result, Mapping) else []
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        return []
    return [str(value) for value in raw_ids if str(value).strip()]


def _pubmed_search_count(payload: Mapping[str, Any]) -> int:
    return len(_pubmed_ids(payload))


def _pubmed_items(payload: Any, ids: Sequence[str]) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return []
    ordered_ids = result.get("uids")
    if not isinstance(ordered_ids, Sequence) or isinstance(ordered_ids, (str, bytes)):
        ordered_ids = ids
    return [
        item
        for identifier in ordered_ids
        if isinstance((item := result.get(str(identifier))), Mapping)
    ]


def _pubmed_summary_count(payload: Mapping[str, Any]) -> int:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return 0
    uids = result.get("uids", [])
    return len(uids) if isinstance(uids, Sequence) and not isinstance(uids, (str, bytes)) else 0


def _operation_record_identities(
    source: SourceName,
    operation: str,
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return stable source-record identities for cross-operation de-duplication."""

    identifiers: list[str] = []
    if source == "PubMed":
        if operation == "search_publications":
            identifiers = _pubmed_ids(payload)
        else:
            result = payload.get("result")
            uids = result.get("uids", []) if isinstance(result, Mapping) else []
            if isinstance(uids, Sequence) and not isinstance(uids, (str, bytes)):
                identifiers = [str(value) for value in uids if str(value).strip()]
        return tuple(
            f"pubmed:{identifier}"
            for identifier in dict.fromkeys(identifiers)
        )

    data = payload.get("data")
    if not isinstance(data, Mapping):
        return ()
    search = data.get("search")
    if isinstance(search, Mapping):
        hits = search.get("hits")
        if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)):
            identifiers.extend(
                str(hit.get("id") or "").strip()
                for hit in hits
                if isinstance(hit, Mapping) and str(hit.get("id") or "").strip()
            )
    disease = data.get("disease")
    if isinstance(disease, Mapping):
        associated = disease.get("associatedTargets")
        rows = associated.get("rows") if isinstance(associated, Mapping) else None
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            identifiers.extend(
                str(target.get("id") or "").strip()
                for row in rows
                if isinstance(row, Mapping)
                and isinstance((target := row.get("target")), Mapping)
                and str(target.get("id") or "").strip()
            )
        elif operation != "associated_targets":
            # In an associated-targets response the embedded disease is query
            # context, not an additional association record.
            disease_id = str(disease.get("id") or "").strip()
            if disease_id:
                identifiers.append(disease_id)
    diseases = data.get("diseases")
    if isinstance(diseases, Sequence) and not isinstance(diseases, (str, bytes)):
        identifiers.extend(
            str(node.get("id") or "").strip()
            for node in diseases
            if isinstance(node, Mapping) and str(node.get("id") or "").strip()
        )
    return tuple(
        f"opentargets:{identifier}"
        for identifier in dict.fromkeys(identifiers)
    )


def _operation_records_screened(
    source: SourceName,
    operation: str,
    records_observed: int,
) -> int:
    # ESearch discovers identifiers. ESummary is the stage at which PubMed
    # records are actually inspected for exact target title co-mentions.
    if source == "PubMed" and operation == "search_publications":
        return 0
    return records_observed


def _extract_gene_tokens(title: str) -> tuple[str, ...]:
    tokens = {
        match.group(1)
        for match in _GENE_TOKEN.finditer(title)
        if match.group(1) not in _GENE_TOKEN_STOPWORDS
    }
    return tuple(sorted(tokens))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pubmed_record_count(records: Sequence[TargetSourceRecord]) -> int:
    return sum(record.source == "PubMed" for record in records)


def _contributing_sources(
    records: Sequence[TargetSourceRecord],
) -> tuple[SourceName, ...]:
    present = {record.source for record in records}
    return tuple(source for source in ("Open Targets", "PubMed") if source in present)


def _track_hint(draft: _CandidateDraft) -> tuple[TrackHint | None, str | None]:
    if draft.discovery_method == "pubmed_explicit_symbol_co_mention":
        return (
            "literature",
            "Candidate is an exact-symbol PubMed title co-mention with Open Targets identity mapping; no association score is available.",
        )
    positive_types = {
        item.evidence_type for item in draft.datatype_scores if item.score > 0.0
    }
    tbio_types = sorted(positive_types & _TBIO_EVIDENCE_TYPES)
    if tbio_types:
        return (
            "tbio",
            "Open Targets reports non-zero disease-specific biological evidence types: "
            + ", ".join(tbio_types),
        )
    if positive_types and positive_types <= _LITERATURE_EVIDENCE_TYPES:
        return (
            "literature",
            "Open Targets association evidence is limited to literature/text-mining types.",
        )
    return None, None


def _safe_record_count(
    counter: Callable[[Mapping[str, Any]], int],
    payload: Mapping[str, Any],
) -> int:
    try:
        return max(0, int(counter(payload)))
    except Exception:
        return 0


__all__ = [
    "AssociationEvidenceSummary",
    "DataTypeScore",
    "DiseaseResolution",
    "DiseaseTargetCandidate",
    "DiseaseTargetDiscovery",
    "DiseaseTargetDiscoveryResult",
    "OPEN_TARGETS_GRAPHQL_URL",
    "PUBMED_SEARCH_URL",
    "PUBMED_SUMMARY_URL",
    "SourceOperationMetric",
    "SourceStatus",
    "TargetProvenance",
    "TargetSourceRecord",
    "TractabilityAssessment",
]
