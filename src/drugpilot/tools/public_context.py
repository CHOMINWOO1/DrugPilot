"""On-demand, context-only public data adapters for MA1.

This module covers the lightweight public sources in the DrugPilot proposal
that are not part of :mod:`drugpilot.tools.public_biology`.  Retrieval is kept
outside the orchestration graph on purpose: callers opt in when they need
preprint, expression-reference, network, pathway, ontology, or trial context.

Every returned :class:`~drugpilot.domain.models.EvidenceRef` has
``supports="context"`` and an explicit ``context_only`` provenance marker.
Keyword matches and database annotations must not be promoted to causal
support or counter-evidence by this adapter.

The eight lightweight access paths are keyless and read-only:

* bioRxiv and medRxiv metadata, federated through Europe PMC
* GTEx reference-gene metadata
* STRING protein-interaction context
* Reactome search context
* KEGG human-gene annotation context
* QuickGO gene-product annotations
* ClinicalTrials.gov study metadata

Four additional read-only adapters are available through the same lookup
contract when the deterministic context router explicitly requests them:
Human Cell Atlas project metadata, an operator-supplied CELLxGENE Census
sidecar, ARCHS4 expression correlation, and credentialed BioGRID
interactions.  They are not part of the default lookup because cell, expression,
and network sources are admitted only when the research brief makes them
relevant.  Missing sidecar or access-key configuration is returned as
``not_configured`` without network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence
from urllib.parse import quote, urlencode

from pydantic import Field, model_validator

from drugpilot.domain.models import EvidenceRef, StrictModel
from drugpilot.tools.http import get_scientific_http_client
from drugpilot.tools.public_cell_data import (
    CellDataLookup,
    PublicCellDataRetriever,
)
from drugpilot.tools.public_expression_interaction import (
    ExpressionInteractionLookup,
    PublicExpressionInteractionRetriever,
)


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
GTEX_GENE_SEARCH_URL = "https://gtexportal.org/api/v2/reference/geneSearch"
STRING_NETWORK_URL = "https://version-12-0.string-db.org/api/json/network"
REACTOME_SEARCH_URL = "https://reactome.org/ContentService/search/query"
KEGG_HUMAN_GENE_FIND_URL = "https://rest.kegg.jp/find/hsa"
QUICKGO_ANNOTATION_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
CLINICAL_TRIALS_SEARCH_URL = "https://clinicaltrials.gov/api/v2/studies"


ContextSource = Literal[
    "bioRxiv",
    "medRxiv",
    "GTEx",
    "STRING",
    "Reactome",
    "KEGG",
    "QuickGO",
    "ClinicalTrials.gov",
    "Human Cell Atlas",
    "CELLxGENE Census",
    "ARCHS4",
    "BioGRID",
]
ContextSourceStatus = Literal[
    "ok", "empty", "skipped", "not_configured", "error"
]

SUPPORTED_CONTEXT_SOURCES: tuple[ContextSource, ...] = (
    "bioRxiv",
    "medRxiv",
    "GTEx",
    "STRING",
    "Reactome",
    "KEGG",
    "QuickGO",
    "ClinicalTrials.gov",
)

ROUTABLE_CONTEXT_SOURCES: tuple[ContextSource, ...] = (
    *SUPPORTED_CONTEXT_SOURCES,
    "Human Cell Atlas",
    "CELLxGENE Census",
    "ARCHS4",
    "BioGRID",
)


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


class MA1ContextLookup(StrictModel):
    """Inputs shared by the context adapters.

    ``uniprot_id`` is optional because only QuickGO requires it.  ``gencode_id``
    is retained for future GTEx expression expansion; the current lightweight
    adapter uses the official gene-search endpoint and therefore accepts a gene
    symbol without resolving an expression matrix.
    """

    disease: str = Field(min_length=2, max_length=300)
    gene_symbol: str = Field(min_length=1, max_length=80)
    uniprot_id: str | None = Field(default=None, max_length=30)
    gencode_id: str | None = Field(default=None, max_length=50)
    cell_type: str | None = Field(default=None, max_length=200)
    limit_per_source: int = Field(default=3, ge=1, le=10)
    sources: list[ContextSource] = Field(
        default_factory=lambda: list(SUPPORTED_CONTEXT_SOURCES)
    )

    @model_validator(mode="after")
    def normalize_inputs(self) -> "MA1ContextLookup":
        object.__setattr__(self, "disease", self.disease.strip())
        object.__setattr__(self, "gene_symbol", self.gene_symbol.strip())
        for field_name in ("uniprot_id", "gencode_id", "cell_type"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, value.strip() or None)
        object.__setattr__(self, "sources", list(dict.fromkeys(self.sources)))
        return self


class ContextSourceResult(StrictModel):
    source: ContextSource
    status: ContextSourceStatus
    queried: bool
    cache_hit: bool = False
    record_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    detail: str | None = None


class ContextRegistryEntry(StrictModel):
    """Adapter availability, explicitly separate from live reachability."""

    source: ContextSource
    access_mode: Literal["live_api", "federated_api"]
    runtime_state: Literal["adapter_ready"] = "adapter_ready"
    connectivity_status: Literal["not_tested"] = "not_tested"
    requires_api_key: bool = False
    endpoint: str
    documentation_url: str
    required_inputs: list[str]
    note: str


class MA1ContextResult(StrictModel):
    lookup: MA1ContextLookup
    evidence: list[EvidenceRef] = Field(default_factory=list)
    sources: list[ContextSourceResult] = Field(default_factory=list)


@dataclass(frozen=True)
class _CacheEntry:
    stored_at: float
    records: tuple[EvidenceRef, ...]


class MA1PublicContextRetriever:
    """Aggregate independent, keyless MA1 context lookups.

    Failures are isolated per source.  The returned source summaries distinguish
    success, empty results, missing optional identifiers, cache hits, and
    transport/schema errors without treating missing data as negative evidence.
    """

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 8.0,
        cache_ttl_seconds: float = 3_600.0,
        clock: Callable[[], float] = monotonic,
        string_caller_identity: str = "DrugPilot",
        cell_data_retriever: PublicCellDataRetriever | None = None,
        expression_interaction_retriever: (
            PublicExpressionInteractionRetriever | None
        ) = None,
        cellxgene_sidecar_url: str | None = None,
        biogrid_access_key: str | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must not be negative")
        if not string_caller_identity.strip():
            raise ValueError("string_caller_identity must not be blank")
        self.client: HTTPClient = client or get_scientific_http_client()
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock
        self.string_caller_identity = string_caller_identity.strip()
        self.cell_data_retriever = cell_data_retriever or PublicCellDataRetriever(
            client=self.client,
            cellxgene_sidecar_url=cellxgene_sidecar_url,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
            clock=clock,
        )
        self.expression_interaction_retriever = (
            expression_interaction_retriever
            or PublicExpressionInteractionRetriever(
                client=self.client,
                biogrid_access_key=biogrid_access_key,
                timeout_seconds=timeout_seconds,
                cache_ttl_seconds=cache_ttl_seconds,
                clock=clock,
            )
        )
        self._cache: dict[tuple[Any, ...], _CacheEntry] = {}
        self._cache_lock = Lock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def registry(self) -> list[ContextRegistryEntry]:
        """Return configuration truth; ``adapter_ready`` is not a health claim."""

        return [
            ContextRegistryEntry(
                source="bioRxiv",
                access_mode="federated_api",
                endpoint=EUROPE_PMC_SEARCH_URL,
                documentation_url="https://europepmc.org/RestfulWebService",
                required_inputs=["gene_symbol", "disease"],
                note=(
                    "Preprint metadata is queried through Europe PMC with an "
                    "explicit bioRxiv server filter; reachability is tested only "
                    "when lookup() runs."
                ),
            ),
            ContextRegistryEntry(
                source="medRxiv",
                access_mode="federated_api",
                endpoint=EUROPE_PMC_SEARCH_URL,
                documentation_url="https://europepmc.org/RestfulWebService",
                required_inputs=["gene_symbol", "disease"],
                note=(
                    "Preprint metadata is queried through Europe PMC with an "
                    "explicit medRxiv server filter; reachability is tested only "
                    "when lookup() runs."
                ),
            ),
            ContextRegistryEntry(
                source="GTEx",
                access_mode="live_api",
                endpoint=GTEX_GENE_SEARCH_URL,
                documentation_url="https://gtexportal.org/api/v2/docs",
                required_inputs=["gene_symbol"],
                note=(
                    "Uses lightweight reference-gene metadata, not bulk expression "
                    "matrices or a disease-specific expression claim."
                ),
            ),
            ContextRegistryEntry(
                source="STRING",
                access_mode="live_api",
                endpoint=STRING_NETWORK_URL,
                documentation_url="https://string-db.org/help/api/",
                required_inputs=["gene_symbol"],
                note=(
                    "Returns interaction-network context from the versioned STRING "
                    "12 API; an association score is not causal proof."
                ),
            ),
            ContextRegistryEntry(
                source="Reactome",
                access_mode="live_api",
                endpoint=REACTOME_SEARCH_URL,
                documentation_url="https://reactome.org/dev/content-service",
                required_inputs=["gene_symbol"],
                note="Returns human pathway search matches as context only.",
            ),
            ContextRegistryEntry(
                source="KEGG",
                access_mode="live_api",
                endpoint=KEGG_HUMAN_GENE_FIND_URL,
                documentation_url="https://www.kegg.jp/kegg/rest/keggapi.html",
                required_inputs=["gene_symbol"],
                note="Returns KEGG human-gene annotations as context only.",
            ),
            ContextRegistryEntry(
                source="QuickGO",
                access_mode="live_api",
                endpoint=QUICKGO_ANNOTATION_URL,
                documentation_url="https://www.ebi.ac.uk/QuickGO/api/",
                required_inputs=["uniprot_id"],
                note=(
                    "Requires a resolved UniProt accession; GO annotations are "
                    "functional context, not disease causality."
                ),
            ),
            ContextRegistryEntry(
                source="ClinicalTrials.gov",
                access_mode="live_api",
                endpoint=CLINICAL_TRIALS_SEARCH_URL,
                documentation_url=(
                    "https://clinicaltrials.gov/data-api/about-api/api-migration"
                ),
                required_inputs=["gene_symbol", "disease"],
                note=(
                    "Returns registered study metadata; registration, phase, and "
                    "status do not establish efficacy."
                ),
            ),
        ]

    def lookup(self, request: MA1ContextLookup) -> MA1ContextResult:
        adapters: Mapping[
            ContextSource, Callable[[MA1ContextLookup], list[EvidenceRef]]
        ] = {
            "bioRxiv": lambda value: self._preprints(value, "bioRxiv"),
            "medRxiv": lambda value: self._preprints(value, "medRxiv"),
            "GTEx": self._gtex,
            "STRING": self._string,
            "Reactome": self._reactome,
            "KEGG": self._kegg,
            "QuickGO": self._quickgo,
            "ClinicalTrials.gov": self._clinical_trials,
        }
        evidence: list[EvidenceRef] = []
        summaries: list[ContextSourceResult] = []
        requested = set(request.sources)

        sources_to_report = [
            *SUPPORTED_CONTEXT_SOURCES,
            *(
                source
                for source in ROUTABLE_CONTEXT_SOURCES[len(SUPPORTED_CONTEXT_SOURCES) :]
                if source in requested
            ),
        ]
        for source in sources_to_report:
            if source not in requested:
                summaries.append(
                    ContextSourceResult(
                        source=source,
                        status="skipped",
                        queried=False,
                        detail="not requested",
                    )
                )
                continue
            if source in {"Human Cell Atlas", "CELLxGENE Census"}:
                result = self.cell_data_retriever.lookup(
                    CellDataLookup(
                        disease=request.disease,
                        gene_symbol=request.gene_symbol,
                        cell_type=request.cell_type,
                        limit_per_source=request.limit_per_source,
                        sources=[source],
                    )
                )
                summary = next(item for item in result.sources if item.source == source)
                evidence.extend(item.model_copy(deep=True) for item in result.evidence)
                summaries.append(
                    ContextSourceResult(
                        source=source,
                        status=summary.status,
                        queried=summary.queried,
                        cache_hit=summary.cache_hit,
                        record_count=summary.record_count,
                        error_type=summary.error_type,
                        detail=summary.detail,
                    )
                )
                continue
            if source in {"ARCHS4", "BioGRID"}:
                result = self.expression_interaction_retriever.lookup(
                    ExpressionInteractionLookup(
                        gene_symbol=request.gene_symbol,
                        disease=request.disease,
                        cell_type=request.cell_type,
                        limit_per_source=request.limit_per_source,
                        sources=[source],
                    )
                )
                summary = next(item for item in result.sources if item.source == source)
                evidence.extend(item.model_copy(deep=True) for item in result.evidence)
                summaries.append(
                    ContextSourceResult(
                        source=source,
                        status=summary.status,
                        queried=summary.queried,
                        cache_hit=summary.cache_hit,
                        record_count=summary.record_count,
                        error_type=summary.error_type,
                        detail=summary.detail,
                    )
                )
                continue
            if source == "QuickGO" and not request.uniprot_id:
                summaries.append(
                    ContextSourceResult(
                        source=source,
                        status="skipped",
                        queried=False,
                        detail="uniprot_id is required",
                    )
                )
                continue

            cached = self._cached(source, request)
            if cached is not None:
                evidence.extend(record.model_copy(deep=True) for record in cached)
                summaries.append(
                    ContextSourceResult(
                        source=source,
                        status="ok" if cached else "empty",
                        queried=True,
                        cache_hit=True,
                        record_count=len(cached),
                    )
                )
                continue

            try:
                records = adapters[source](request)
            except Exception as exc:  # each remote source fails independently
                summaries.append(
                    ContextSourceResult(
                        source=source,
                        status="error",
                        queried=True,
                        error_type=type(exc).__name__,
                    )
                )
                continue

            stored = self._store(source, request, records)
            evidence.extend(record.model_copy(deep=True) for record in stored)
            summaries.append(
                ContextSourceResult(
                    source=source,
                    status="ok" if stored else "empty",
                    queried=True,
                    record_count=len(stored),
                )
            )

        return MA1ContextResult(
            lookup=request.model_copy(deep=True),
            evidence=evidence,
            sources=summaries,
        )

    def _preprints(
        self,
        request: MA1ContextLookup,
        server: Literal["bioRxiv", "medRxiv"],
    ) -> list[EvidenceRef]:
        query = (
            f'TITLE_ABS:"{_quoted(request.gene_symbol)}" AND '
            f'TITLE_ABS:"{_quoted(request.disease)}" AND SRC:PPR AND '
            f'PUBLISHER:"{server}"'
        )
        response = self.client.get(
            EUROPE_PMC_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "pageSize": request.limit_per_source,
                "resultType": "core",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        records: list[EvidenceRef] = []
        for item in _mapping_sequence(results)[: request.limit_per_source]:
            external_id = item.get("doi") or item.get("id") or item.get("pmcid")
            if not external_id:
                continue
            title = str(item.get("title") or f"{server} preprint {external_id}").strip()
            doi = item.get("doi")
            url = (
                f"https://doi.org/{quote(str(doi), safe='/')}"
                if doi
                else f"https://europepmc.org/article/PPR/{quote(str(external_id))}"
            )
            records.append(
                self._record(
                    source=server,
                    external_id=str(external_id),
                    title=title,
                    url=url,
                    query=query,
                    endpoint=EUROPE_PMC_SEARCH_URL,
                    evidence_scope=(
                        "preprint keyword-match metadata; not peer-reviewed and not "
                        "causal or counter evidence"
                    ),
                    extra={
                        "access_path": "Europe PMC federation",
                        "preprint_server": server,
                        "doi": doi,
                        "publication_date": item.get("firstPublicationDate")
                        or item.get("firstIndexDate"),
                        "authors": item.get("authorString"),
                        "is_preprint": True,
                    },
                )
            )
        return records

    def _gtex(self, request: MA1ContextLookup) -> list[EvidenceRef]:
        query = request.gencode_id or request.gene_symbol
        response = self.client.get(
            GTEX_GENE_SEARCH_URL,
            params={
                "geneId": query,
                "gencodeVersion": "v26",
                "genomeBuild": "GRCh38/hg38",
                "itemsPerPage": request.limit_per_source,
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        items = _payload_records(payload)
        records: list[EvidenceRef] = []
        for item in items[: request.limit_per_source]:
            gencode_id = item.get("gencodeId") or item.get("geneId")
            symbol = item.get("geneSymbol") or item.get("symbol") or request.gene_symbol
            external_id = gencode_id or symbol
            if not external_id:
                continue
            title = f"GTEx reference gene: {symbol}"
            records.append(
                self._record(
                    source="GTEx",
                    external_id=str(external_id),
                    title=title,
                    url=f"https://gtexportal.org/home/gene/{quote(str(external_id))}",
                    query=query,
                    endpoint=GTEX_GENE_SEARCH_URL,
                    evidence_scope=(
                        "GTEx reference-gene metadata; not a disease-specific "
                        "expression or causal claim"
                    ),
                    extra={
                        "gene_symbol": symbol,
                        "gencode_id": gencode_id,
                        "gene_name": item.get("geneName") or item.get("description"),
                        "chromosome": item.get("chromosome"),
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "strand": item.get("strand"),
                        "dataset_scope": "reference metadata",
                    },
                )
            )
        return records

    def _string(self, request: MA1ContextLookup) -> list[EvidenceRef]:
        query = request.gene_symbol
        response = self.client.get(
            STRING_NETWORK_URL,
            params={
                "identifiers": query,
                "species": 9606,
                "required_score": 700,
                "add_nodes": request.limit_per_source,
                "caller_identity": self.string_caller_identity,
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise ValueError("STRING response is not a list")
        network_url = "https://string-db.org/cgi/network?" + urlencode(
            {"identifiers": query, "species": 9606}
        )
        records: list[EvidenceRef] = []
        for item in _mapping_sequence(payload)[: request.limit_per_source]:
            protein_a = item.get("preferredName_A") or item.get("stringId_A")
            protein_b = item.get("preferredName_B") or item.get("stringId_B")
            if not protein_a or not protein_b:
                continue
            external_id = f"{item.get('stringId_A') or protein_a}|{item.get('stringId_B') or protein_b}"
            title = f"STRING interaction context: {protein_a} — {protein_b}"
            records.append(
                self._record(
                    source="STRING",
                    external_id=external_id,
                    title=title,
                    url=network_url,
                    query=query,
                    endpoint=STRING_NETWORK_URL,
                    evidence_scope=(
                        "functional-association network context; association score "
                        "is not mechanistic or causal proof"
                    ),
                    extra={
                        "protein_a": protein_a,
                        "protein_b": protein_b,
                        "score": item.get("score"),
                        "experimental_score": item.get("experimental"),
                        "database_score": item.get("database"),
                        "text_mining_score": item.get("textmining"),
                    },
                )
            )
        return records

    def _reactome(self, request: MA1ContextLookup) -> list[EvidenceRef]:
        query = request.gene_symbol
        response = self.client.get(
            REACTOME_SEARCH_URL,
            params={
                "query": query,
                "species": "Homo sapiens",
                "types": "Pathway",
                "cluster": "true",
            },
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        entries = _reactome_entries(response.json())
        records: list[EvidenceRef] = []
        for item in entries[: request.limit_per_source]:
            stable_id = item.get("stId") or item.get("stableId") or item.get("dbId")
            if not stable_id:
                continue
            name = item.get("name") or item.get("displayName") or stable_id
            if isinstance(name, list):
                name = name[0] if name else stable_id
            title = f"Reactome pathway context: {name}"
            records.append(
                self._record(
                    source="Reactome",
                    external_id=str(stable_id),
                    title=title,
                    url=f"https://reactome.org/content/detail/{quote(str(stable_id))}",
                    query=query,
                    endpoint=REACTOME_SEARCH_URL,
                    evidence_scope=(
                        "curated pathway search context; pathway membership does not "
                        "establish disease causality"
                    ),
                    extra={
                        "schema_class": item.get("type") or item.get("schemaClass"),
                        "species": item.get("species"),
                        "exact_type": item.get("exactType"),
                    },
                )
            )
        return records

    def _kegg(self, request: MA1ContextLookup) -> list[EvidenceRef]:
        query = request.gene_symbol
        url = f"{KEGG_HUMAN_GENE_FIND_URL}/{quote(query, safe='')}"
        response = self.client.get(
            url,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        records: list[EvidenceRef] = []
        for line in str(response.text).splitlines()[: request.limit_per_source]:
            if "\t" not in line:
                continue
            external_id, description = line.split("\t", 1)
            external_id = external_id.strip()
            description = description.strip()
            if not external_id:
                continue
            title = f"KEGG human-gene context: {description or external_id}"
            records.append(
                self._record(
                    source="KEGG",
                    external_id=external_id,
                    title=title,
                    url=f"https://www.kegg.jp/entry/{quote(external_id)}",
                    query=query,
                    endpoint=KEGG_HUMAN_GENE_FIND_URL,
                    evidence_scope=(
                        "KEGG human-gene annotation context; not a pathway effect or "
                        "disease-causality conclusion"
                    ),
                    extra={"description": description, "organism": "Homo sapiens"},
                )
            )
        return records

    def _quickgo(self, request: MA1ContextLookup) -> list[EvidenceRef]:
        assert request.uniprot_id is not None
        gene_product_id = (
            request.uniprot_id
            if ":" in request.uniprot_id
            else f"UniProtKB:{request.uniprot_id}"
        )
        query = gene_product_id
        response = self.client.get(
            QUICKGO_ANNOTATION_URL,
            params={
                "geneProductId": gene_product_id,
                "taxonId": 9606,
                "limit": request.limit_per_source,
            },
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        items = _mapping_sequence(response.json().get("results", []))
        records: list[EvidenceRef] = []
        for item in items[: request.limit_per_source]:
            go_id = item.get("goId")
            if not go_id:
                continue
            reference = item.get("reference") or item.get("referenceId")
            external_id = f"{gene_product_id}|{go_id}|{reference or 'annotation'}"
            symbol = item.get("symbol") or request.gene_symbol
            title = f"QuickGO annotation: {symbol} — {go_id}"
            records.append(
                self._record(
                    source="QuickGO",
                    external_id=external_id,
                    title=title,
                    url=f"https://www.ebi.ac.uk/QuickGO/term/{quote(str(go_id), safe=':')}",
                    query=query,
                    endpoint=QUICKGO_ANNOTATION_URL,
                    evidence_scope=(
                        "GO functional annotation context; not disease causality or "
                        "therapeutic validation"
                    ),
                    extra={
                        "gene_product_id": item.get("geneProductId") or gene_product_id,
                        "gene_symbol": symbol,
                        "go_id": go_id,
                        "go_aspect": item.get("goAspect") or item.get("aspect"),
                        "evidence_code": item.get("evidenceCode"),
                        "qualifier": item.get("qualifier"),
                        "reference": reference,
                        "assigned_by": item.get("assignedBy"),
                    },
                )
            )
        return records

    def _clinical_trials(self, request: MA1ContextLookup) -> list[EvidenceRef]:
        query = f'"{request.gene_symbol}" AND "{request.disease}"'
        response = self.client.get(
            CLINICAL_TRIALS_SEARCH_URL,
            params={"query.term": query, "pageSize": request.limit_per_source},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        studies = _mapping_sequence(response.json().get("studies", []))
        records: list[EvidenceRef] = []
        for study in studies[: request.limit_per_source]:
            protocol = study.get("protocolSection")
            if not isinstance(protocol, Mapping):
                continue
            identification = _mapping(protocol.get("identificationModule"))
            status = _mapping(protocol.get("statusModule"))
            design = _mapping(protocol.get("designModule"))
            conditions = _mapping(protocol.get("conditionsModule"))
            arms = _mapping(protocol.get("armsInterventionsModule"))
            nct_id = identification.get("nctId")
            if not nct_id:
                continue
            title = str(
                identification.get("briefTitle")
                or identification.get("officialTitle")
                or f"Clinical trial {nct_id}"
            )
            interventions = [
                {
                    "name": item.get("name"),
                    "type": item.get("type"),
                }
                for item in _mapping_sequence(arms.get("interventions", []))
            ]
            records.append(
                self._record(
                    source="ClinicalTrials.gov",
                    external_id=str(nct_id),
                    title=title,
                    url=f"https://clinicaltrials.gov/study/{quote(str(nct_id))}",
                    query=query,
                    endpoint=CLINICAL_TRIALS_SEARCH_URL,
                    evidence_scope=(
                        "registered study metadata and keyword match; registration, "
                        "phase, and recruitment status are not efficacy evidence"
                    ),
                    extra={
                        "overall_status": status.get("overallStatus"),
                        "study_type": design.get("studyType"),
                        "phases": design.get("phases", []),
                        "conditions": conditions.get("conditions", []),
                        "interventions": interventions,
                    },
                )
            )
        return records

    def _record(
        self,
        *,
        source: ContextSource,
        external_id: str,
        title: str,
        url: str,
        query: str,
        endpoint: str,
        evidence_scope: str,
        extra: Mapping[str, Any],
    ) -> EvidenceRef:
        digest = sha256(f"context|{source}|{external_id}".encode("utf-8")).hexdigest()[:12]
        return EvidenceRef(
            evidence_id=f"ev_ctx_{digest}",
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
        source: ContextSource,
        request: MA1ContextLookup,
    ) -> tuple[Any, ...]:
        return (
            source,
            request.disease.casefold(),
            request.gene_symbol.casefold(),
            (request.uniprot_id or "").casefold(),
            (request.gencode_id or "").casefold(),
            (request.cell_type or "").casefold(),
            request.limit_per_source,
        )

    def _cached(
        self,
        source: ContextSource,
        request: MA1ContextLookup,
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
        source: ContextSource,
        request: MA1ContextLookup,
        records: Sequence[EvidenceRef],
    ) -> tuple[EvidenceRef, ...]:
        frozen = tuple(record.model_copy(deep=True) for record in records)
        with self._cache_lock:
            self._cache[self._cache_key(source, request)] = _CacheEntry(
                stored_at=self._clock(),
                records=frozen,
            )
        return tuple(record.model_copy(deep=True) for record in frozen)


def _quoted(value: str) -> str:
    return " ".join(value.replace('"', " ").split())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _payload_records(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    for key in ("data", "results", "genes"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return _mapping_sequence(value)
        if isinstance(value, Mapping):
            for nested_key in ("data", "results", "genes"):
                nested = value.get(nested_key)
                if isinstance(nested, Sequence) and not isinstance(
                    nested, (str, bytes)
                ):
                    return _mapping_sequence(nested)
    return []


def _reactome_entries(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    entries: list[Mapping[str, Any]] = []
    for result in _mapping_sequence(payload.get("results", [])):
        nested = result.get("entries")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            entries.extend(_mapping_sequence(nested))
        elif any(key in result for key in ("stId", "stableId", "dbId")):
            entries.append(result)
    if not entries:
        entries.extend(_mapping_sequence(payload.get("entries", [])))
    return entries


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


__all__ = [
    "CLINICAL_TRIALS_SEARCH_URL",
    "ContextRegistryEntry",
    "ContextSource",
    "ContextSourceResult",
    "EUROPE_PMC_SEARCH_URL",
    "GTEX_GENE_SEARCH_URL",
    "KEGG_HUMAN_GENE_FIND_URL",
    "MA1ContextLookup",
    "MA1ContextResult",
    "MA1PublicContextRetriever",
    "QUICKGO_ANNOTATION_URL",
    "REACTOME_SEARCH_URL",
    "ROUTABLE_CONTEXT_SOURCES",
    "STRING_NETWORK_URL",
    "SUPPORTED_CONTEXT_SOURCES",
]
