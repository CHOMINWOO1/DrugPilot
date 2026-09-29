"""Read-only public adapters for ARCHS4 expression and BioGRID interactions.

The adapters in this module are deliberately standalone.  They can be wired
into orchestration later without claiming that a catalog entry is live before
an actual lookup runs.

ARCHS4 is queried through its documented public ``/sigpy/data/correlation``
endpoint.  Although that analytical query uses HTTP POST, it only reads the
public expression collection and does not create or mutate remote resources.
This is the practical interactive path on Windows: the official alternative is
an ARCHS4 HDF5 snapshot plus ``archs4py``, and current human gene-level files
are tens of gigabytes.

BioGRID is queried through its official ``/interactions`` REST endpoint.  The
service requires a free, per-application access key.  Without a valid key the
adapter returns ``not_configured`` and performs no network I/O.  Access-key
values are never copied into evidence, cache keys, errors, or provenance.

Both sources are normalized as context evidence.  Co-expression and curated
interaction records can inform a hypothesis, but this module never upgrades
them to disease causality or counter-evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
import re
from threading import Lock
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence
from urllib.parse import quote

import httpx
from pydantic import Field, model_validator

from drugpilot.domain.models import EvidenceRef, StrictModel
from drugpilot.tools.http import classify_transport_error, get_scientific_http_client


ARCHS4_CORRELATION_URL = "https://maayanlab.cloud/sigpy/data/correlation"
ARCHS4_GENE_PAGE_URL = "https://maayanlab.cloud/archs4/gene"
ARCHS4_DOCUMENTATION_URL = "https://archs4.org/help"
ARCHS4_DOWNLOAD_URL = "https://archs4.org/download"

BIOGRID_INTERACTIONS_URL = "https://webservice.thebiogrid.org/interactions"
BIOGRID_DOCUMENTATION_URL = (
    "https://wiki.thebiogrid.org/doku.php/biogridrest"
)
BIOGRID_ACCESS_KEY_ENV = "BIOGRID_ACCESS_KEY"


ExpressionInteractionSource = Literal["ARCHS4", "BioGRID"]
ExpressionInteractionStatus = Literal[
    "ok",
    "empty",
    "skipped",
    "not_configured",
    "error",
]
Species = Literal["human", "mouse"]
ContextKind = Literal["disease", "cell_type", "tissue"]

SUPPORTED_EXPRESSION_INTERACTION_SOURCES: tuple[
    ExpressionInteractionSource, ...
] = ("ARCHS4", "BioGRID")

_TAXONOMY_BY_SPECIES: Mapping[Species, int] = {
    "human": 9606,
    "mouse": 10090,
}
_BIOGRID_KEY_PATTERN = re.compile(r"^[A-Za-z0-9]{32}$")


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


class ExpressionInteractionLookup(StrictModel):
    """Gene plus optional biological contexts for both public adapters.

    ARCHS4 runs one documented gene-correlation query for each supplied
    disease, cell type, or tissue term.  BioGRID only supports gene/organism
    filtering; the three context fields remain attached to BioGRID provenance
    but are explicitly not represented as server-side filters.
    """

    gene_symbol: str = Field(min_length=1, max_length=80)
    disease: str | None = Field(default=None, max_length=300)
    cell_type: str | None = Field(default=None, max_length=200)
    tissue: str | None = Field(default=None, max_length=200)
    species: Species = "human"
    limit_per_source: int = Field(default=5, ge=1, le=100)
    sources: list[ExpressionInteractionSource] = Field(
        default_factory=lambda: list(SUPPORTED_EXPRESSION_INTERACTION_SOURCES)
    )

    @model_validator(mode="after")
    def normalize_inputs(self) -> "ExpressionInteractionLookup":
        gene_symbol = self.gene_symbol.strip()
        if not gene_symbol:
            raise ValueError("gene_symbol must not be blank")
        object.__setattr__(self, "gene_symbol", gene_symbol)

        for field_name in ("disease", "cell_type", "tissue"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, value.strip() or None)
        object.__setattr__(self, "sources", list(dict.fromkeys(self.sources)))
        return self

    @property
    def contexts(self) -> tuple[tuple[ContextKind, str], ...]:
        contexts: list[tuple[ContextKind, str]] = []
        seen: set[str] = set()
        for kind in ("disease", "cell_type", "tissue"):
            value = getattr(self, kind)
            if not value:
                continue
            normalized = value.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            contexts.append((kind, value))
        return tuple(contexts)

    @property
    def tax_id(self) -> int:
        return _TAXONOMY_BY_SPECIES[self.species]


class ExpressionInteractionSourceResult(StrictModel):
    source: ExpressionInteractionSource
    status: ExpressionInteractionStatus
    queried: bool
    cache_hit: bool = False
    record_count: int = Field(default=0, ge=0)
    query_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    detail: str | None = None


class ExpressionInteractionRegistryEntry(StrictModel):
    """Configuration truth, kept separate from live reachability."""

    source: ExpressionInteractionSource
    access_mode: Literal["public_api", "credentialed_api"]
    runtime_state: Literal["adapter_ready", "not_configured"]
    connectivity_status: Literal["not_tested"] = "not_tested"
    requires_api_key: bool
    endpoint: str
    documentation_url: str
    bulk_download_url: str | None = None
    credential_environment_variable: str | None = None
    required_inputs: list[str]
    note: str


class ExpressionInteractionResult(StrictModel):
    lookup: ExpressionInteractionLookup
    evidence: list[EvidenceRef] = Field(default_factory=list)
    sources: list[ExpressionInteractionSourceResult] = Field(default_factory=list)


@dataclass(frozen=True)
class _CacheEntry:
    stored_at: float
    records: tuple[EvidenceRef, ...]
    query_count: int


def _evidence_id(source: ExpressionInteractionSource, external_id: str) -> str:
    digest = sha256(
        f"expression-interaction|{source}|{external_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"ev_exprint_{digest}"


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _source_error_type(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return f"upstream_http_{exc.response.status_code}"
    if isinstance(exc, httpx.TransportError):
        return classify_transport_error(exc)
    return type(exc).__name__


class PublicExpressionInteractionRetriever:
    """Retrieve ARCHS4 expression context and BioGRID gene interactions.

    Remote failures are isolated by source.  Successful and empty responses are
    cached with a TTL; cached evidence records preserve that they were first
    obtained live while changing their provenance class to ``external_cache``.
    """

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        biogrid_access_key: str | None = None,
        timeout_seconds: float = 8.0,
        cache_ttl_seconds: float = 3_600.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must not be negative")

        self.client: HTTPClient = client or get_scientific_http_client()
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock
        raw_key = (
            biogrid_access_key
            if biogrid_access_key is not None
            else os.getenv(BIOGRID_ACCESS_KEY_ENV)
        )
        self._biogrid_access_key = raw_key.strip() if raw_key else None
        self._cache: dict[tuple[Any, ...], _CacheEntry] = {}
        self._cache_lock = Lock()

    @property
    def biogrid_configuration_error(self) -> str | None:
        if not self._biogrid_access_key:
            return f"{BIOGRID_ACCESS_KEY_ENV} is not set"
        if not _BIOGRID_KEY_PATTERN.fullmatch(self._biogrid_access_key):
            return (
                f"{BIOGRID_ACCESS_KEY_ENV} must be a 32-character "
                "alphanumeric key"
            )
        return None

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def registry(self) -> list[ExpressionInteractionRegistryEntry]:
        """Return adapter/configuration truth without claiming connectivity."""

        return [
            ExpressionInteractionRegistryEntry(
                source="ARCHS4",
                access_mode="public_api",
                runtime_state="adapter_ready",
                requires_api_key=False,
                endpoint=ARCHS4_CORRELATION_URL,
                documentation_url=ARCHS4_DOCUMENTATION_URL,
                bulk_download_url=ARCHS4_DOWNLOAD_URL,
                required_inputs=[
                    "gene_symbol",
                    "one of disease, cell_type, or tissue",
                ],
                note=(
                    "Uses the documented read-only correlation analysis API. "
                    "The official HDF5/archs4py path is preferable for large "
                    "batch work but is not downloaded by this adapter."
                ),
            ),
            ExpressionInteractionRegistryEntry(
                source="BioGRID",
                access_mode="credentialed_api",
                runtime_state=(
                    "adapter_ready"
                    if self.biogrid_configuration_error is None
                    else "not_configured"
                ),
                requires_api_key=True,
                endpoint=BIOGRID_INTERACTIONS_URL,
                documentation_url=BIOGRID_DOCUMENTATION_URL,
                credential_environment_variable=BIOGRID_ACCESS_KEY_ENV,
                required_inputs=["gene_symbol", BIOGRID_ACCESS_KEY_ENV],
                note=(
                    "BioGRID requires a free 32-character application access "
                    "key. Disease, cell type, and tissue are provenance context; "
                    "the REST request itself is filtered by gene and organism."
                ),
            ),
        ]

    def lookup(
        self,
        request: ExpressionInteractionLookup,
    ) -> ExpressionInteractionResult:
        evidence: list[EvidenceRef] = []
        summaries: list[ExpressionInteractionSourceResult] = []
        requested_sources = set(request.sources)

        for source in SUPPORTED_EXPRESSION_INTERACTION_SOURCES:
            if source not in requested_sources:
                summaries.append(
                    ExpressionInteractionSourceResult(
                        source=source,
                        status="skipped",
                        queried=False,
                        detail="not requested",
                    )
                )
                continue

            if source == "ARCHS4" and not request.contexts:
                summaries.append(
                    ExpressionInteractionSourceResult(
                        source=source,
                        status="skipped",
                        queried=False,
                        detail=(
                            "ARCHS4 correlation requires at least one disease, "
                            "cell_type, or tissue metadata term"
                        ),
                    )
                )
                continue

            if source == "BioGRID" and self.biogrid_configuration_error:
                summaries.append(
                    ExpressionInteractionSourceResult(
                        source=source,
                        status="not_configured",
                        queried=False,
                        error_type=(
                            "missing_access_key"
                            if not self._biogrid_access_key
                            else "invalid_access_key_configuration"
                        ),
                        detail=self.biogrid_configuration_error,
                    )
                )
                continue

            cached = self._cached(source, request)
            if cached is not None:
                evidence.extend(
                    record.model_copy(deep=True) for record in cached.records
                )
                summaries.append(
                    ExpressionInteractionSourceResult(
                        source=source,
                        status="ok" if cached.records else "empty",
                        queried=True,
                        cache_hit=True,
                        record_count=len(cached.records),
                        query_count=cached.query_count,
                    )
                )
                continue

            try:
                if source == "ARCHS4":
                    records, query_count = self._archs4(request)
                else:
                    records, query_count = self._biogrid(request)
            except Exception as exc:  # isolate each independent public source
                summaries.append(
                    ExpressionInteractionSourceResult(
                        source=source,
                        status="error",
                        queried=True,
                        query_count=(len(request.contexts) if source == "ARCHS4" else 1),
                        error_type=_source_error_type(exc),
                    )
                )
                continue

            stored = self._store(source, request, records, query_count=query_count)
            evidence.extend(record.model_copy(deep=True) for record in stored)
            summaries.append(
                ExpressionInteractionSourceResult(
                    source=source,
                    status="ok" if stored else "empty",
                    queried=True,
                    record_count=len(stored),
                    query_count=query_count,
                )
            )

        return ExpressionInteractionResult(
            lookup=request.model_copy(deep=True),
            evidence=evidence,
            sources=summaries,
        )

    def _archs4(
        self,
        request: ExpressionInteractionLookup,
    ) -> tuple[list[EvidenceRef], int]:
        records: list[EvidenceRef] = []
        for context_kind, context_term in request.contexts:
            request_payload = {
                "gene": request.gene_symbol,
                "meta": context_term,
                "species": request.species,
                "k": request.limit_per_source,
            }
            response = self.client.post(
                ARCHS4_CORRELATION_URL,
                json=request_payload,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("ARCHS4 response is not a JSON object")
            if payload.get("error"):
                raise ValueError("ARCHS4 returned an error payload")

            positive = self._correlated_genes(
                payload.get("positive_correlated_genes"),
                request.limit_per_source,
            )
            negative = self._correlated_genes(
                payload.get("negative_correlated_genes"),
                request.limit_per_source,
            )
            samples_value = payload.get("samples")
            samples = (
                [str(value) for value in samples_value]
                if isinstance(samples_value, Sequence)
                and not isinstance(samples_value, (str, bytes))
                else []
            )
            mean_expression = payload.get("mean_log_expression")
            if not positive and not negative and not samples and mean_expression is None:
                continue

            external_id = (
                f"{request.species}:{request.gene_symbol}:"
                f"{context_kind}:{context_term}"
            )
            query = (
                f"gene={request.gene_symbol}; {context_kind}={context_term}; "
                f"species={request.species}"
            )
            title = (
                f"ARCHS4 expression correlation for {request.gene_symbol} "
                f"in {context_term}"
            )
            records.append(
                self._record(
                    source="ARCHS4",
                    external_id=external_id,
                    title=title,
                    url=(
                        f"{ARCHS4_GENE_PAGE_URL}/"
                        f"{quote(request.gene_symbol, safe='')}"
                    ),
                    query=query,
                    endpoint=ARCHS4_CORRELATION_URL,
                    evidence_scope=(
                        "ARCHS4 observational RNA-seq co-expression within a "
                        "metadata-matched sample set; not disease causality, "
                        "mechanistic direction, or a physical interaction claim"
                    ),
                    extra={
                        "access_path": "public_api",
                        "http_method": "POST",
                        "read_only": True,
                        "species": request.species,
                        "gene_symbol": request.gene_symbol,
                        "context_kind": context_kind,
                        "context_term": context_term,
                        "api_response_gene": payload.get("gene"),
                        "api_search_term": payload.get("searchterm"),
                        "mean_log_expression": mean_expression,
                        "positive_correlated_genes": positive,
                        "negative_correlated_genes": negative,
                        "sample_count": len(samples),
                        "sample_ids": samples[: request.limit_per_source],
                        "requested_k": request.limit_per_source,
                    },
                )
            )
        return records, len(request.contexts)

    def _biogrid(
        self,
        request: ExpressionInteractionLookup,
    ) -> tuple[list[EvidenceRef], int]:
        # lookup() validates configuration before reaching this branch.
        assert self._biogrid_access_key is not None
        params = {
            "accesskey": self._biogrid_access_key,
            "format": "json",
            "geneList": request.gene_symbol,
            "searchNames": "true",
            "includeInteractors": "true",
            "includeInteractorInteractions": "false",
            "interSpeciesExcluded": "true",
            "taxId": request.tax_id,
            "max": request.limit_per_source,
        }
        response = self.client.get(
            BIOGRID_INTERACTIONS_URL,
            params=params,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        interactions = self._biogrid_interactions(payload)
        records: list[EvidenceRef] = []

        for payload_key, item in interactions[: request.limit_per_source]:
            interaction_id = item.get("BIOGRID_INTERACTION_ID") or payload_key
            symbol_a = item.get("OFFICIAL_SYMBOL_A")
            symbol_b = item.get("OFFICIAL_SYMBOL_B")
            if not interaction_id or not symbol_a or not symbol_b:
                continue
            experimental_system = item.get("EXPERIMENTAL_SYSTEM")
            title = f"BioGRID interaction: {symbol_a} — {symbol_b}"
            if experimental_system:
                title += f" ({experimental_system})"
            external_id = str(interaction_id)
            records.append(
                self._record(
                    source="BioGRID",
                    external_id=external_id,
                    title=title,
                    url=(
                        "https://thebiogrid.org/interaction/"
                        f"{quote(external_id, safe='')}"
                    ),
                    query=(
                        f"gene={request.gene_symbol}; tax_id={request.tax_id}; "
                        "include_first_order_interactors=true"
                    ),
                    endpoint=BIOGRID_INTERACTIONS_URL,
                    evidence_scope=(
                        "BioGRID curated physical or genetic interaction context; "
                        "not proof that the interaction is active, causal, or "
                        "therapeutically relevant in the supplied disease, cell, "
                        "or tissue context"
                    ),
                    extra={
                        "access_path": "credentialed_api",
                        "http_method": "GET",
                        "read_only": True,
                        "credential_used": True,
                        "credential_environment_variable": BIOGRID_ACCESS_KEY_ENV,
                        "credential_value_recorded": False,
                        "gene_symbol_query": request.gene_symbol,
                        "species": request.species,
                        "tax_id_filter": request.tax_id,
                        "disease_context": request.disease,
                        "cell_type_context": request.cell_type,
                        "tissue_context": request.tissue,
                        "server_filter_scope": "gene_symbol_and_tax_id",
                        "biogrid_id_a": item.get("BIOGRID_ID_A"),
                        "biogrid_id_b": item.get("BIOGRID_ID_B"),
                        "entrez_gene_a": item.get("ENTREZ_GENE_A"),
                        "entrez_gene_b": item.get("ENTREZ_GENE_B"),
                        "official_symbol_a": symbol_a,
                        "official_symbol_b": symbol_b,
                        "experimental_system": experimental_system,
                        "experimental_system_type": item.get(
                            "EXPERIMENTAL_SYSTEM_TYPE"
                        ),
                        "pubmed_author": item.get("PUBMED_AUTHOR"),
                        "pubmed_id": item.get("PUBMED_ID"),
                        "organism_a": item.get("ORGANISM_A"),
                        "organism_b": item.get("ORGANISM_B"),
                        "throughput": item.get("THROUGHPUT"),
                        "quantitation": item.get("QUANTITATION"),
                        "modification": item.get("MODIFICATION"),
                        "ontology_terms": item.get("ONTOLOGY_TERMS"),
                        "qualifications": item.get("QUALIFICATIONS"),
                        "tags": item.get("TAGS"),
                        "source_database": item.get("SOURCEDB"),
                    },
                )
            )
        return records, 1

    @staticmethod
    def _correlated_genes(value: Any, limit: int) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in _mapping_sequence(value)[:limit]:
            gene = item.get("gene")
            correlation = item.get("correlation")
            if gene is None or correlation is None:
                continue
            normalized.append(
                _compact(
                    {
                        "gene": str(gene),
                        "correlation": correlation,
                        "mean_log_expression": item.get("mean_log_expression"),
                    }
                )
            )
        return normalized

    @staticmethod
    def _biogrid_interactions(
        payload: Any,
    ) -> list[tuple[str | None, Mapping[str, Any]]]:
        if isinstance(payload, Mapping):
            status = str(payload.get("STATUS") or "").casefold()
            if status in {"error", "warning"}:
                raise ValueError("BioGRID returned an error payload")
            if "BIOGRID_INTERACTION_ID" in payload:
                return [(None, payload)]
            return [
                (str(key), value)
                for key, value in payload.items()
                if isinstance(value, Mapping)
            ]
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            return [
                (None, value)
                for value in payload
                if isinstance(value, Mapping)
            ]
        raise ValueError("BioGRID response is not a JSON object or array")

    def _record(
        self,
        *,
        source: ExpressionInteractionSource,
        external_id: str,
        title: str,
        url: str,
        query: str,
        endpoint: str,
        evidence_scope: str,
        extra: Mapping[str, Any],
    ) -> EvidenceRef:
        return EvidenceRef(
            evidence_id=_evidence_id(source, external_id),
            source=source,
            external_id=external_id,
            title=title,
            url=url,
            supports="context",
            excerpt_hash=sha256(title.encode("utf-8")).hexdigest(),
            metadata={
                "provenance_class": "external_live",
                "query": query,
                "live": True,
                "mock": False,
                "cache_hit": False,
                "retrieval_adapter": type(self).__name__,
                "retrieval_endpoint": endpoint,
                "context_only": True,
                "causal_inference_allowed": False,
                "counter_evidence_allowed": False,
                "evidence_scope": evidence_scope,
                **_compact(extra),
            },
        )

    def _cache_key(
        self,
        source: ExpressionInteractionSource,
        request: ExpressionInteractionLookup,
    ) -> tuple[Any, ...]:
        return (
            source,
            request.gene_symbol.casefold(),
            (request.disease or "").casefold(),
            (request.cell_type or "").casefold(),
            (request.tissue or "").casefold(),
            request.species,
            request.limit_per_source,
        )

    def _cached(
        self,
        source: ExpressionInteractionSource,
        request: ExpressionInteractionLookup,
    ) -> _CacheEntry | None:
        key = self._cache_key(source, request)
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if self._clock() - entry.stored_at > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            records = tuple(record.model_copy(deep=True) for record in entry.records)
        cached_records = tuple(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "provenance_class": "external_cache",
                        "live": False,
                        "cache_hit": True,
                        "retrieved_live": True,
                        "origin_provenance_class": str(
                            record.metadata.get("provenance_class") or ""
                        ),
                        "origin_content_hash": str(
                            record.metadata.get("content_hash")
                            or record.excerpt_hash
                            or ""
                        ),
                    }
                },
                deep=True,
            )
            for record in records
        )
        return _CacheEntry(
            stored_at=entry.stored_at,
            records=cached_records,
            query_count=entry.query_count,
        )

    def _store(
        self,
        source: ExpressionInteractionSource,
        request: ExpressionInteractionLookup,
        records: Sequence[EvidenceRef],
        *,
        query_count: int,
    ) -> tuple[EvidenceRef, ...]:
        frozen = tuple(record.model_copy(deep=True) for record in records)
        with self._cache_lock:
            self._cache[self._cache_key(source, request)] = _CacheEntry(
                stored_at=self._clock(),
                records=frozen,
                query_count=query_count,
            )
        return tuple(record.model_copy(deep=True) for record in frozen)


__all__ = [
    "ARCHS4_CORRELATION_URL",
    "ARCHS4_DOCUMENTATION_URL",
    "ARCHS4_DOWNLOAD_URL",
    "BIOGRID_ACCESS_KEY_ENV",
    "BIOGRID_DOCUMENTATION_URL",
    "BIOGRID_INTERACTIONS_URL",
    "ExpressionInteractionLookup",
    "ExpressionInteractionRegistryEntry",
    "ExpressionInteractionResult",
    "ExpressionInteractionSource",
    "ExpressionInteractionSourceResult",
    "ExpressionInteractionStatus",
    "PublicExpressionInteractionRetriever",
    "SUPPORTED_EXPRESSION_INTERACTION_SOURCES",
]
