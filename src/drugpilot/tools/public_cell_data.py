"""Read-only public single-cell metadata adapters.

Two access paths are intentionally kept separate:

* Human Cell Atlas (HCA) project metadata is queried directly through the
  official, keyless Azul Data Browser REST API.
* CZ CELLxGENE Discover Census is queried through an *optional, operator
  supplied* HTTP sidecar.  The official Census client is based on TileDB-SOMA
  and its installation documentation does not support Windows.  The sidecar
  is therefore a small deployment boundary for a supported Linux/macOS host;
  it is not represented as an official CELLxGENE REST API.

The HCA API exposes exact-match disease and selected-cell-type project
filters, but no gene or expression filter.  A gene-only HCA lookup is skipped,
and a gene supplied with another HCA filter is recorded as an ignored input.
The adapter never turns a project/metadata match, gene presence, or a cell
count into causal, differential-expression, or therapeutic evidence.

CELLxGENE sidecar contract (read-only HTTP GET)
------------------------------------------------

``GET {base_url}/v1/cellxgene-census/lookup`` accepts these query parameters:

``organism`` (always ``homo_sapiens``), ``census_version``, ``limit``, and any
of ``disease``, ``gene_symbol``, and ``cell_type`` that were provided.  A
successful response has the following minimum shape::

    {
      "schema_version": "1.0",
      "census_version": "2025-11-08",
      "records": [
        {
          "dataset_id": "...",
          "dataset_title": "...",
          "matched_cell_count": 123,
          "gene_symbol": "TP53",
          "feature_id": "ENSG...",
          "disease": ["glioblastoma"],
          "cell_type": ["malignant cell"]
        }
      ]
    }

Extra fields are tolerated but only a conservative allow-list is retained in
provenance.  ``matched_cell_count`` and gene presence are context, not an
expression effect.  The adapter sends no credentials and performs no write
verb.  Operators exposing an unauthenticated sidecar beyond localhost must
apply their own network controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from threading import Lock
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence
from urllib.parse import quote, urlsplit

from pydantic import Field, model_validator

from drugpilot.domain.models import EvidenceRef, StrictModel
from drugpilot.tools.http import get_scientific_http_client


HCA_PROJECTS_URL = "https://service.azul.data.humancellatlas.org/index/projects"
HCA_PROJECT_PAGE_URL = "https://explore.data.humancellatlas.org/projects"
HCA_DOCUMENTATION_URL = (
    "https://data.humancellatlas.org/apis/api-documentation/data-browser-api"
)
HCA_OPENAPI_URL = "https://service.azul.data.humancellatlas.org/openapi.json"

CELLXGENE_CENSUS_DOCUMENTATION_URL = (
    "https://chanzuckerberg.github.io/cellxgene-census/"
)
CELLXGENE_CENSUS_INSTALLATION_URL = (
    "https://chanzuckerberg.github.io/cellxgene-census/"
    "cellxgene_census_docsite_installation.html"
)
CELLXGENE_SIDECAR_PATH = "/v1/cellxgene-census/lookup"
CELLXGENE_SIDECAR_SCHEMA_VERSION = "1.0"


CellDataSource = Literal["Human Cell Atlas", "CELLxGENE Census"]
CellDataSourceStatus = Literal[
    "ok",
    "empty",
    "skipped",
    "not_configured",
    "error",
]
CellDataProvenance = Literal[
    "external_live",
    "external_cache",
    "not_applicable",
]

SUPPORTED_CELL_DATA_SOURCES: tuple[CellDataSource, ...] = (
    "Human Cell Atlas",
    "CELLxGENE Census",
)


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...


class CellDataLookup(StrictModel):
    """A bounded lookup across disease, gene, and cell-type dimensions."""

    disease: str | None = Field(default=None, max_length=300)
    gene_symbol: str | None = Field(default=None, max_length=80)
    cell_type: str | None = Field(default=None, max_length=200)
    limit_per_source: int = Field(default=5, ge=1, le=20)
    census_version: str = Field(default="stable", min_length=1, max_length=80)
    sources: list[CellDataSource] = Field(
        default_factory=lambda: list(SUPPORTED_CELL_DATA_SOURCES)
    )

    @model_validator(mode="after")
    def normalize_inputs(self) -> "CellDataLookup":
        for field_name in ("disease", "gene_symbol", "cell_type"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, value.strip() or None)
        census_version = self.census_version.strip()
        if not census_version:
            raise ValueError("census_version must not be blank")
        object.__setattr__(self, "census_version", census_version)
        object.__setattr__(self, "sources", list(dict.fromkeys(self.sources)))
        if not any((self.disease, self.gene_symbol, self.cell_type)):
            raise ValueError(
                "at least one of disease, gene_symbol, or cell_type is required"
            )
        return self


class CellDataSourceResult(StrictModel):
    source: CellDataSource
    status: CellDataSourceStatus
    queried: bool
    cache_hit: bool = False
    record_count: int = Field(default=0, ge=0)
    provenance_class: CellDataProvenance = "not_applicable"
    error_type: str | None = None
    detail: str | None = None
    ignored_inputs: list[str] = Field(default_factory=list)


class CellDataRegistryEntry(StrictModel):
    """Configuration truth, deliberately separate from reachability."""

    source: CellDataSource
    access_mode: Literal["live_api", "sidecar_http"]
    runtime_state: Literal["adapter_ready", "configuration_required"]
    configuration_state: Literal[
        "keyless_public",
        "configured",
        "not_configured",
    ]
    connectivity_status: Literal["not_tested"] = "not_tested"
    requires_credentials: bool = False
    endpoint: str | None = None
    documentation_url: str
    supported_inputs: list[str]
    unsupported_inputs: list[str] = Field(default_factory=list)
    note: str


class CellDataResult(StrictModel):
    lookup: CellDataLookup
    evidence: list[EvidenceRef] = Field(default_factory=list)
    sources: list[CellDataSourceResult] = Field(default_factory=list)


@dataclass(frozen=True)
class _CacheEntry:
    stored_at: float
    records: tuple[EvidenceRef, ...]


class PublicCellDataRetriever:
    """Aggregate independent HCA and optional CELLxGENE Census lookups.

    A transport or schema failure in either source is converted to that
    source's status and does not prevent the other source from returning data.
    Cache entries are source-local and include empty results.
    """

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        cellxgene_sidecar_url: str | None = None,
        timeout_seconds: float = 8.0,
        cache_ttl_seconds: float = 3_600.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must not be negative")
        self.client: HTTPClient = client or get_scientific_http_client()
        self.cellxgene_sidecar_url = _normalize_sidecar_url(cellxgene_sidecar_url)
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock
        self._cache: dict[tuple[Any, ...], _CacheEntry] = {}
        self._cache_lock = Lock()

    @property
    def cellxgene_sidecar_endpoint(self) -> str | None:
        if self.cellxgene_sidecar_url is None:
            return None
        return f"{self.cellxgene_sidecar_url}{CELLXGENE_SIDECAR_PATH}"

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def registry(self) -> list[CellDataRegistryEntry]:
        """Describe executable paths without claiming they are reachable."""

        endpoint = self.cellxgene_sidecar_endpoint
        return [
            CellDataRegistryEntry(
                source="Human Cell Atlas",
                access_mode="live_api",
                runtime_state="adapter_ready",
                configuration_state="keyless_public",
                endpoint=HCA_PROJECTS_URL,
                documentation_url=HCA_DOCUMENTATION_URL,
                supported_inputs=["disease", "cell_type"],
                unsupported_inputs=["gene_symbol"],
                note=(
                    "Uses the official public Azul project index with exact-match "
                    "sampleDisease and selectedCellType filters, restricted to "
                    "accessible Homo sapiens projects. The API has no gene or "
                    "expression-search field."
                ),
            ),
            CellDataRegistryEntry(
                source="CELLxGENE Census",
                access_mode="sidecar_http",
                runtime_state=(
                    "adapter_ready" if endpoint else "configuration_required"
                ),
                configuration_state="configured" if endpoint else "not_configured",
                endpoint=endpoint,
                documentation_url=CELLXGENE_CENSUS_DOCUMENTATION_URL,
                supported_inputs=["disease", "gene_symbol", "cell_type"],
                note=(
                    "Calls an operator-supplied, credential-free read-only HTTP "
                    "sidecar implementing DrugPilot's v1 contract. This is not an "
                    "official CELLxGENE REST endpoint; the sidecar must run the "
                    "official Census/TileDB-SOMA client on a supported host."
                ),
            ),
        ]

    def lookup(self, request: CellDataLookup) -> CellDataResult:
        evidence: list[EvidenceRef] = []
        summaries: list[CellDataSourceResult] = []
        requested = set(request.sources)

        for source in SUPPORTED_CELL_DATA_SOURCES:
            if source not in requested:
                summaries.append(
                    CellDataSourceResult(
                        source=source,
                        status="skipped",
                        queried=False,
                        detail="not requested",
                    )
                )
                continue

            ignored_inputs: list[str] = []
            if source == "Human Cell Atlas":
                if request.gene_symbol:
                    ignored_inputs.append("gene_symbol")
                if not request.disease and not request.cell_type:
                    summaries.append(
                        CellDataSourceResult(
                            source=source,
                            status="skipped",
                            queried=False,
                            detail=(
                                "official HCA project API has no gene/expression "
                                "lookup; disease or cell_type is required"
                            ),
                            ignored_inputs=ignored_inputs,
                        )
                    )
                    continue
            elif self.cellxgene_sidecar_endpoint is None:
                summaries.append(
                    CellDataSourceResult(
                        source=source,
                        status="not_configured",
                        queried=False,
                        detail="cellxgene_sidecar_url is not configured",
                    )
                )
                continue

            cached = self._cached(source, request)
            if cached is not None:
                evidence.extend(record.model_copy(deep=True) for record in cached)
                summaries.append(
                    CellDataSourceResult(
                        source=source,
                        status="ok" if cached else "empty",
                        queried=True,
                        cache_hit=True,
                        record_count=len(cached),
                        provenance_class="external_cache",
                        ignored_inputs=ignored_inputs,
                    )
                )
                continue

            try:
                if source == "Human Cell Atlas":
                    records = self._human_cell_atlas(request)
                else:
                    records = self._cellxgene_census(request)
            except Exception as exc:  # isolate each remote source
                summaries.append(
                    CellDataSourceResult(
                        source=source,
                        status="error",
                        queried=True,
                        error_type=type(exc).__name__,
                        ignored_inputs=ignored_inputs,
                    )
                )
                continue

            stored = self._store(source, request, records)
            evidence.extend(record.model_copy(deep=True) for record in stored)
            summaries.append(
                CellDataSourceResult(
                    source=source,
                    status="ok" if stored else "empty",
                    queried=True,
                    record_count=len(stored),
                    provenance_class="external_live",
                    ignored_inputs=ignored_inputs,
                )
            )

        return CellDataResult(
            lookup=request.model_copy(deep=True),
            evidence=evidence,
            sources=summaries,
        )

    def _human_cell_atlas(self, request: CellDataLookup) -> list[EvidenceRef]:
        filters: dict[str, dict[str, list[Any]]] = {
            "genusSpecies": {"is": ["Homo sapiens"]},
            "accessible": {"is": [True]},
        }
        if request.disease:
            filters["sampleDisease"] = {"is": [request.disease]}
        if request.cell_type:
            filters["selectedCellType"] = {"is": [request.cell_type]}
        query = json.dumps(filters, ensure_ascii=False, separators=(",", ":"))
        response = self.client.get(
            HCA_PROJECTS_URL,
            params={
                "catalog": "dcp60",
                "filters": query,
                "size": request.limit_per_source,
                "sort": "projectTitle",
                "order": "asc",
            },
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("HCA response is not an object")
        hits = _mapping_sequence(payload.get("hits"))
        records: list[EvidenceRef] = []
        for hit in hits[: request.limit_per_source]:
            projects = _mapping_sequence(hit.get("projects"))
            if not projects:
                continue
            project = projects[0]
            project_id = project.get("projectId") or hit.get("entryId")
            if not project_id:
                continue
            title = str(
                project.get("projectTitle") or f"HCA project {project_id}"
            ).strip()
            records.append(
                self._record(
                    source="Human Cell Atlas",
                    external_id=str(project_id),
                    title=f"HCA project metadata: {title}",
                    url=f"{HCA_PROJECT_PAGE_URL}/{quote(str(project_id), safe='')}",
                    query=query,
                    endpoint=HCA_PROJECTS_URL,
                    evidence_scope=(
                        "exact-filter project metadata match; not gene-expression, "
                        "differential-expression, disease-causality, or therapeutic "
                        "evidence"
                    ),
                    access_path="official_keyless_rest_api",
                    extra={
                        "api_catalog": "dcp60",
                        "project_id": str(project_id),
                        "project_title": title,
                        "project_shortname": project.get("projectShortname"),
                        "project_description": project.get("projectDescription"),
                        "estimated_cell_count": project.get("estimatedCellCount"),
                        "data_use_restriction": project.get("dataUseRestriction"),
                        "accessible": project.get("accessible"),
                        "species": _nested_values(hit, "donorOrganisms", "genusSpecies"),
                        "sample_diseases": _nested_values(hit, "samples", "disease"),
                        "donor_diseases": _nested_values(
                            hit, "donorOrganisms", "disease"
                        ),
                        "specimen_diseases": _nested_values(
                            hit, "specimens", "disease"
                        ),
                        "selected_cell_types": _nested_values(
                            hit, "cellSuspensions", "selectedCellType"
                        ),
                        "organs": _nested_values(hit, "samples", "organ"),
                        "publications": _hca_publications(project),
                        "gene_lookup_supported": False,
                        "ignored_inputs": (
                            ["gene_symbol"] if request.gene_symbol else []
                        ),
                    },
                )
            )
        return records

    def _cellxgene_census(self, request: CellDataLookup) -> list[EvidenceRef]:
        endpoint = self.cellxgene_sidecar_endpoint
        assert endpoint is not None
        params: dict[str, Any] = {
            "organism": "homo_sapiens",
            "census_version": request.census_version,
            "limit": request.limit_per_source,
        }
        if request.disease:
            params["disease"] = request.disease
        if request.gene_symbol:
            params["gene_symbol"] = request.gene_symbol
        if request.cell_type:
            params["cell_type"] = request.cell_type

        response = self.client.get(
            endpoint,
            params=params,
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("CELLxGENE sidecar response is not an object")
        if payload.get("schema_version") != CELLXGENE_SIDECAR_SCHEMA_VERSION:
            raise ValueError("unsupported CELLxGENE sidecar schema_version")
        records_payload = payload.get("records")
        if not isinstance(records_payload, Sequence) or isinstance(
            records_payload, (str, bytes)
        ):
            raise ValueError("CELLxGENE sidecar records is not an array")
        census_version = payload.get("census_version")
        if not census_version:
            raise ValueError("CELLxGENE sidecar omitted census_version")
        provenance = _mapping(payload.get("provenance"))

        query = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
        records: list[EvidenceRef] = []
        for item in _mapping_sequence(records_payload)[: request.limit_per_source]:
            dataset_id = item.get("dataset_id")
            if not dataset_id:
                continue
            dataset_title = str(
                item.get("dataset_title") or f"CELLxGENE dataset {dataset_id}"
            ).strip()
            collection_id = item.get("collection_id")
            record_url = item.get("url")
            if not record_url and collection_id:
                record_url = (
                    "https://cellxgene.cziscience.com/collections/"
                    f"{quote(str(collection_id), safe='')}"
                )
            if not record_url:
                record_url = CELLXGENE_CENSUS_DOCUMENTATION_URL
            records.append(
                self._record(
                    source="CELLxGENE Census",
                    external_id=str(dataset_id),
                    title=f"CELLxGENE Census slice metadata: {dataset_title}",
                    url=str(record_url),
                    query=query,
                    endpoint=endpoint,
                    evidence_scope=(
                        "sidecar-reported Census metadata/gene-presence match; "
                        "matched cell counts are not differential-expression, "
                        "causal, or therapeutic evidence"
                    ),
                    access_path="configured_sidecar_http",
                    extra={
                        "sidecar_contract": (
                            "drugpilot.cellxgene-census.lookup/"
                            f"{CELLXGENE_SIDECAR_SCHEMA_VERSION}"
                        ),
                        "direct_upstream_call": False,
                        "upstream_source": "CZ CELLxGENE Discover Census",
                        "upstream_access_api": "cellxgene-census / TileDB-SOMA",
                        "census_version": str(census_version),
                        "dataset_id": str(dataset_id),
                        "dataset_title": dataset_title,
                        "collection_id": collection_id,
                        "matched_cell_count": item.get("matched_cell_count"),
                        "diseases": _as_scalar_list(item.get("disease")),
                        "disease_ontology_term_ids": _as_scalar_list(
                            item.get("disease_ontology_term_id")
                        ),
                        "cell_types": _as_scalar_list(item.get("cell_type")),
                        "cell_type_ontology_term_ids": _as_scalar_list(
                            item.get("cell_type_ontology_term_id")
                        ),
                        "gene_symbol": item.get("gene_symbol"),
                        "feature_id": item.get("feature_id"),
                        "gene_present": item.get("gene_present"),
                        "assays": _as_scalar_list(item.get("assay")),
                        "tissues": _as_scalar_list(item.get("tissue")),
                        "citation": item.get("citation"),
                        "sidecar_provenance": _compact(
                            {
                                "census_uri": provenance.get("census_uri"),
                                "census_build_date": provenance.get(
                                    "census_build_date"
                                ),
                                "cellxgene_census_version": provenance.get(
                                    "cellxgene_census_version"
                                ),
                                "queried_at": provenance.get("queried_at"),
                            }
                        ),
                    },
                )
            )
        return records

    def _record(
        self,
        *,
        source: CellDataSource,
        external_id: str,
        title: str,
        url: str,
        query: str,
        endpoint: str,
        evidence_scope: str,
        access_path: str,
        extra: Mapping[str, Any],
    ) -> EvidenceRef:
        digest = sha256(
            f"public-cell-data|{source}|{external_id}".encode("utf-8")
        ).hexdigest()[:12]
        return EvidenceRef(
            evidence_id=f"ev_cell_{digest}",
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
                "access_path": access_path,
                "read_only": True,
                "credentials_sent": False,
                "context_only": True,
                "causal_inference_allowed": False,
                "counter_evidence_allowed": False,
                "differential_expression_claim_allowed": False,
                "evidence_scope": evidence_scope,
                **_compact(extra),
            },
        )

    def _cache_key(
        self,
        source: CellDataSource,
        request: CellDataLookup,
    ) -> tuple[Any, ...]:
        return (
            source,
            (request.disease or "").casefold(),
            (request.gene_symbol or "").casefold(),
            (request.cell_type or "").casefold(),
            request.census_version.casefold(),
            request.limit_per_source,
        )

    def _cached(
        self,
        source: CellDataSource,
        request: CellDataLookup,
    ) -> tuple[EvidenceRef, ...] | None:
        key = self._cache_key(source, request)
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if self._clock() - entry.stored_at > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            records = tuple(record.model_copy(deep=True) for record in entry.records)
        return tuple(
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

    def _store(
        self,
        source: CellDataSource,
        request: CellDataLookup,
        records: Sequence[EvidenceRef],
    ) -> tuple[EvidenceRef, ...]:
        frozen = tuple(record.model_copy(deep=True) for record in records)
        with self._cache_lock:
            self._cache[self._cache_key(source, request)] = _CacheEntry(
                stored_at=self._clock(),
                records=frozen,
            )
        return tuple(record.model_copy(deep=True) for record in frozen)


def _normalize_sidecar_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("cellxgene_sidecar_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("cellxgene_sidecar_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("cellxgene_sidecar_url must not contain query or fragment")
    return normalized


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _as_scalar_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            item
            for item in value
            if isinstance(item, (str, int, float, bool))
        ]
    return []


def _nested_values(
    parent: Mapping[str, Any],
    collection_key: str,
    value_key: str,
) -> list[Any]:
    values: list[Any] = []
    for item in _mapping_sequence(parent.get(collection_key)):
        for value in _as_scalar_list(item.get(value_key)):
            if value not in values:
                values.append(value)
    return values


def _hca_publications(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _compact(
            {
                "title": item.get("publicationTitle"),
                "doi": item.get("doi"),
                "url": item.get("publicationUrl"),
                "official_hca_publication": item.get("officialHcaPublication"),
            }
        )
        for item in _mapping_sequence(project.get("publications"))
    ]


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != [] and value != {}
    }


__all__ = [
    "CELLXGENE_CENSUS_DOCUMENTATION_URL",
    "CELLXGENE_CENSUS_INSTALLATION_URL",
    "CELLXGENE_SIDECAR_PATH",
    "CELLXGENE_SIDECAR_SCHEMA_VERSION",
    "HCA_DOCUMENTATION_URL",
    "HCA_OPENAPI_URL",
    "HCA_PROJECTS_URL",
    "CellDataLookup",
    "CellDataRegistryEntry",
    "CellDataResult",
    "CellDataSourceResult",
    "PublicCellDataRetriever",
    "SUPPORTED_CELL_DATA_SOURCES",
]
