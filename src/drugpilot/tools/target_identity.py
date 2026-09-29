"""Canonical human target identity resolution for MA1/MA2 hand-off.

The resolver is deliberately narrower than a general gene search service.  It
accepts a gene symbol and an optional Ensembl gene identifier, then asks Open
Targets and UniProt for an exact human-target match.  It never invents an
identifier when either service is unavailable or returns an ambiguous result.

Network calls are read-only, independently isolated, bounded by the configured
timeout, and cached with an explicit provenance transition from
``external_live`` to ``external_cache``.  A deterministic fixture can be used
for offline demonstrations only when the caller explicitly supplies a mapping;
fixture provenance is preserved in the returned audit record.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from pydantic import Field, field_validator, model_validator

from drugpilot.domain.models import StrictModel
from drugpilot.tools.http import get_scientific_http_client


OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

ResolutionStatus = Literal["resolved", "partial", "unresolved", "fixture"]
SourceStatus = Literal["ok", "empty", "error", "skipped"]
IdentitySource = Literal["Open Targets", "UniProt", "fixture"]
ProvenanceClass = Literal[
    "external_live", "external_cache", "fixture_fallback", "policy_control"
]


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


class TargetIdentityQuery(StrictModel):
    """A target name received from discovery, with an optional known ID."""

    gene_symbol: str = Field(min_length=1, max_length=80)
    ensembl_id: str | None = Field(default=None, pattern=r"^ENSG\d{11}$")

    @field_validator("gene_symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("gene_symbol must be a string")
        normalized = value.strip()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError("gene_symbol must be a non-empty symbol without spaces")
        return normalized.upper()

    @field_validator("ensembl_id", mode="before")
    @classmethod
    def normalize_ensembl(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("ensembl_id must be a string")
        # Ensembl release suffixes are not part of the stable target ID.
        return value.strip().upper().split(".", maxsplit=1)[0]


class CanonicalTargetIdentity(StrictModel):
    """Verified identifiers that downstream chemistry/structure tools can use."""

    canonical_symbol: str | None = Field(default=None, max_length=80)
    ensembl_id: str | None = Field(default=None, pattern=r"^ENSG\d{11}$")
    uniprot_accession: str | None = Field(default=None, max_length=30)
    uniprot_entry_name: str | None = Field(default=None, max_length=100)
    protein_name: str | None = Field(default=None, max_length=1_000)
    organism: Literal["Homo sapiens"] = "Homo sapiens"
    taxon_id: Literal[9606] = 9606

    @model_validator(mode="after")
    def require_one_resolved_identifier(self) -> "CanonicalTargetIdentity":
        if not any((self.canonical_symbol, self.ensembl_id, self.uniprot_accession)):
            raise ValueError("at least one resolved target identifier is required")
        return self


class FixtureTargetIdentity(StrictModel):
    """An explicitly labelled deterministic identity for offline demonstrations."""

    identity: CanonicalTargetIdentity
    fixture_label: str = Field(min_length=1, max_length=200)
    provenance_class: Literal["fixture_fallback"] = "fixture_fallback"


class IdentitySourceResult(StrictModel):
    """Per-service audit record; an error is not treated as negative evidence."""

    source: IdentitySource
    status: SourceStatus
    queried: bool
    cache_hit: bool = False
    provenance_class: ProvenanceClass
    query: str
    matched_by: Literal["gene_symbol", "ensembl_id", "fixture"] | None = None
    canonical_symbol: str | None = None
    ensembl_id: str | None = Field(default=None, pattern=r"^ENSG\d{11}$")
    uniprot_accession: str | None = None
    error_type: str | None = None
    endpoint: str | None = None


class TargetIdentityResolution(StrictModel):
    """Result designed to be attached to MA1 evidence or passed directly to MA2."""

    query: TargetIdentityQuery
    status: ResolutionStatus
    identity: CanonicalTargetIdentity | None = None
    sources: list[IdentitySourceResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_identity_pair(self) -> "TargetIdentityResolution":
        if self.status == "unresolved" and self.identity is not None:
            raise ValueError("unresolved results cannot contain an identity")
        if self.status != "unresolved" and self.identity is None:
            raise ValueError("resolved, partial, and fixture results require an identity")
        if self.status == "fixture" and not any(
            item.provenance_class == "fixture_fallback" for item in self.sources
        ):
            raise ValueError("fixture status requires explicit fixture provenance")
        return self

    @property
    def ready_for_structure_lookup(self) -> bool:
        return bool(self.identity and self.identity.uniprot_accession)


@dataclass(frozen=True)
class _CacheEntry:
    stored_at: float
    resolution: TargetIdentityResolution


@dataclass(frozen=True)
class _OpenTargetsMatch:
    canonical_symbol: str
    ensembl_id: str
    protein_name: str | None
    matched_by: Literal["gene_symbol", "ensembl_id"]


@dataclass(frozen=True)
class _UniProtMatch:
    canonical_symbol: str
    accession: str
    entry_name: str | None
    protein_name: str | None
    ensembl_ids: tuple[str, ...]


class TargetIdentityResolver:
    """Resolve exact human target identities with traceable partial failure.

    ``fixture_mapping`` is opt-in.  Supplying it is the only way a fixture can
    be returned, and those results use ``status='fixture'`` plus
    ``provenance_class='fixture_fallback'``.  Live and cached identifiers are
    never synthesized from the input string.
    """

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: float = 3_600.0,
        fixture_mapping: Mapping[str, FixtureTargetIdentity] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must not be negative")
        self.client: HTTPClient = client or get_scientific_http_client()
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.fixture_mapping = {
            symbol.strip().upper(): fixture.model_copy(deep=True)
            for symbol, fixture in (fixture_mapping or {}).items()
        }
        self._clock = clock
        self._cache: dict[tuple[str, str, tuple[str, ...]], _CacheEntry] = {}
        self._cache_lock = Lock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def resolve(
        self,
        gene_symbol: str,
        ensembl_id: str | None = None,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> TargetIdentityResolution:
        return self.resolve_query(
            TargetIdentityQuery(gene_symbol=gene_symbol, ensembl_id=ensembl_id),
            allowed_sources=allowed_sources,
        )

    def resolve_many(
        self,
        queries: Sequence[TargetIdentityQuery],
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> list[TargetIdentityResolution]:
        return [
            self.resolve_query(query, allowed_sources=allowed_sources)
            for query in queries
        ]

    def resolve_query(
        self,
        query: TargetIdentityQuery,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> TargetIdentityResolution:
        allowed = (
            frozenset(("Open Targets", "UniProt"))
            if allowed_sources is None
            else frozenset(allowed_sources)
        )
        cached = self._cached(query, allowed)
        if cached is not None:
            return cached

        source_results: list[IdentitySourceResult] = []
        warnings: list[str] = []

        open_targets: _OpenTargetsMatch | None = None
        if "Open Targets" not in allowed:
            source_results.append(
                IdentitySourceResult(
                    source="Open Targets",
                    status="skipped",
                    queried=False,
                    provenance_class="policy_control",
                    query=query.ensembl_id or query.gene_symbol,
                    error_type="policy_rejected",
                    endpoint=OPEN_TARGETS_GRAPHQL_URL,
                )
            )
            warnings.append("Open Targets skipped: data policy rejected the source")
        else:
            try:
                open_targets = self._open_targets(query)
                source_results.append(
                    self._open_targets_summary(query, open_targets)
                )
            except Exception as exc:  # one public service must not disable the other
                source_results.append(
                    IdentitySourceResult(
                        source="Open Targets",
                        status="error",
                        queried=True,
                        provenance_class="external_live",
                        query=query.ensembl_id or query.gene_symbol,
                        error_type=type(exc).__name__,
                        endpoint=OPEN_TARGETS_GRAPHQL_URL,
                    )
                )
                warnings.append(f"Open Targets unavailable: {type(exc).__name__}")

        uniprot_symbol = open_targets.canonical_symbol if open_targets else query.gene_symbol
        uniprot: _UniProtMatch | None = None
        if "UniProt" not in allowed:
            source_results.append(
                IdentitySourceResult(
                    source="UniProt",
                    status="skipped",
                    queried=False,
                    provenance_class="policy_control",
                    query=self._uniprot_query(uniprot_symbol),
                    error_type="policy_rejected",
                    endpoint=UNIPROT_SEARCH_URL,
                )
            )
            warnings.append("UniProt skipped: data policy rejected the source")
        else:
            try:
                uniprot = self._uniprot(uniprot_symbol)
                source_results.append(self._uniprot_summary(uniprot_symbol, uniprot))
            except Exception as exc:  # preserve a valid Open Targets partial result
                source_results.append(
                    IdentitySourceResult(
                        source="UniProt",
                        status="error",
                        queried=True,
                        provenance_class="external_live",
                        query=self._uniprot_query(uniprot_symbol),
                        error_type=type(exc).__name__,
                        endpoint=UNIPROT_SEARCH_URL,
                    )
                )
                warnings.append(f"UniProt unavailable: {type(exc).__name__}")

        resolution = self._combine(query, open_targets, uniprot, source_results, warnings)
        if resolution.status == "unresolved":
            resolution = self._fixture_or_unresolved(resolution)

        # Do not cache fixtures or complete failures: a later live request must
        # be able to recover as soon as the external service is available.
        if resolution.status in {"resolved", "partial"}:
            self._store(query, resolution, allowed)
        return resolution

    def _open_targets(self, query: TargetIdentityQuery) -> _OpenTargetsMatch | None:
        if query.ensembl_id:
            payload = {
                "query": (
                    "query TargetIdentity($ensemblId:String!){"
                    "target(ensemblId:$ensemblId){"
                    "id approvedSymbol approvedName biotype}}"
                ),
                "variables": {"ensemblId": query.ensembl_id},
            }
        else:
            payload = {
                "query": (
                    "query TargetSearch($q:String!){"
                    "search(queryString:$q,page:{index:0,size:10}){"
                    "hits{id name entity description}}}"
                ),
                "variables": {"q": query.gene_symbol},
            }
        response = self.client.post(
            OPEN_TARGETS_GRAPHQL_URL,
            json=payload,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("errors") and not body.get("data"):
            raise ValueError("Open Targets returned GraphQL errors")

        if query.ensembl_id:
            target = body.get("data", {}).get("target")
            if not isinstance(target, Mapping):
                return None
            ensembl_id = _valid_ensembl(target.get("id"))
            symbol = _clean_string(target.get("approvedSymbol"))
            if ensembl_id != query.ensembl_id or not symbol:
                return None
            return _OpenTargetsMatch(
                canonical_symbol=symbol,
                ensembl_id=ensembl_id,
                protein_name=_clean_string(target.get("approvedName")),
                matched_by="ensembl_id",
            )

        hits = body.get("data", {}).get("search", {}).get("hits", [])
        exact_matches = [
            hit
            for hit in hits
            if isinstance(hit, Mapping)
            and hit.get("entity") == "target"
            and _clean_string(hit.get("name"))
            and str(hit.get("name")).casefold() == query.gene_symbol.casefold()
            and _valid_ensembl(hit.get("id"))
        ]
        if len(exact_matches) != 1:
            return None
        hit = exact_matches[0]
        return _OpenTargetsMatch(
            canonical_symbol=str(hit["name"]),
            ensembl_id=str(hit["id"]),
            protein_name=_clean_string(hit.get("description")),
            matched_by="gene_symbol",
        )

    def _uniprot(self, symbol: str) -> _UniProtMatch | None:
        query = self._uniprot_query(symbol)
        response = self.client.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "size": 10,
                "fields": (
                    "accession,id,protein_name,gene_names,organism_name,"
                    "organism_id,reviewed,xref_ensembl"
                ),
            },
            headers={"Accept": "application/json"},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        exact: list[Mapping[str, Any]] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            if _organism_taxon(item) != 9606:
                continue
            primary_gene = _uniprot_primary_gene(item)
            if primary_gene and primary_gene.casefold() == symbol.casefold():
                exact.append(item)
        if not exact:
            return None

        # Prefer the manually reviewed Swiss-Prot entry.  If multiple reviewed
        # entries remain, accession ordering makes the selection deterministic.
        exact.sort(
            key=lambda item: (
                0 if _is_reviewed_uniprot_entry(item) else 1,
                str(item.get("primaryAccession", "")),
            )
        )
        selected = exact[0]
        accession = _clean_string(selected.get("primaryAccession"))
        primary_gene = _uniprot_primary_gene(selected)
        if not accession or not primary_gene:
            return None
        return _UniProtMatch(
            canonical_symbol=primary_gene,
            accession=accession,
            entry_name=_clean_string(selected.get("uniProtkbId")),
            protein_name=_uniprot_protein_name(selected),
            ensembl_ids=tuple(_uniprot_ensembl_ids(selected)),
        )

    @staticmethod
    def _uniprot_query(symbol: str) -> str:
        return f"gene_exact:{symbol} AND organism_id:9606"

    @staticmethod
    def _open_targets_summary(
        query: TargetIdentityQuery,
        match: _OpenTargetsMatch | None,
    ) -> IdentitySourceResult:
        return IdentitySourceResult(
            source="Open Targets",
            status="ok" if match else "empty",
            queried=True,
            provenance_class="external_live",
            query=query.ensembl_id or query.gene_symbol,
            matched_by=match.matched_by if match else None,
            canonical_symbol=match.canonical_symbol if match else None,
            ensembl_id=match.ensembl_id if match else None,
            endpoint=OPEN_TARGETS_GRAPHQL_URL,
        )

    def _uniprot_summary(
        self,
        symbol: str,
        match: _UniProtMatch | None,
    ) -> IdentitySourceResult:
        return IdentitySourceResult(
            source="UniProt",
            status="ok" if match else "empty",
            queried=True,
            provenance_class="external_live",
            query=self._uniprot_query(symbol),
            matched_by="gene_symbol" if match else None,
            canonical_symbol=match.canonical_symbol if match else None,
            ensembl_id=match.ensembl_ids[0] if match and len(match.ensembl_ids) == 1 else None,
            uniprot_accession=match.accession if match else None,
            endpoint=UNIPROT_SEARCH_URL,
        )

    def _combine(
        self,
        query: TargetIdentityQuery,
        open_targets: _OpenTargetsMatch | None,
        uniprot: _UniProtMatch | None,
        source_results: list[IdentitySourceResult],
        warnings: list[str],
    ) -> TargetIdentityResolution:
        if open_targets and uniprot:
            if open_targets.canonical_symbol.casefold() != uniprot.canonical_symbol.casefold():
                warnings.append("Open Targets and UniProt canonical symbols disagree")
                uniprot = None
            elif uniprot.ensembl_ids and open_targets.ensembl_id not in uniprot.ensembl_ids:
                warnings.append("Open Targets and UniProt Ensembl identifiers disagree")
                # Do not combine a protein accession with a conflicting gene
                # identity.  Keep the Open Targets gene as a partial result;
                # downstream structure lookup remains disabled until a human
                # or a later source resolves the conflict.
                uniprot = None

        canonical_symbol = (
            open_targets.canonical_symbol
            if open_targets
            else uniprot.canonical_symbol if uniprot else None
        )
        ensembl_id = open_targets.ensembl_id if open_targets else None
        if not ensembl_id and uniprot and len(uniprot.ensembl_ids) == 1:
            ensembl_id = uniprot.ensembl_ids[0]
        identity = None
        if canonical_symbol or ensembl_id or uniprot:
            identity = CanonicalTargetIdentity(
                canonical_symbol=canonical_symbol,
                ensembl_id=ensembl_id,
                uniprot_accession=uniprot.accession if uniprot else None,
                uniprot_entry_name=uniprot.entry_name if uniprot else None,
                protein_name=(
                    uniprot.protein_name
                    if uniprot and uniprot.protein_name
                    else open_targets.protein_name if open_targets else None
                ),
            )
        if identity is None:
            status: ResolutionStatus = "unresolved"
        elif identity.ensembl_id and identity.uniprot_accession and identity.canonical_symbol:
            status = "resolved"
        else:
            status = "partial"
        return TargetIdentityResolution(
            query=query,
            status=status,
            identity=identity,
            sources=source_results,
            warnings=warnings,
        )

    def _fixture_or_unresolved(
        self,
        unresolved: TargetIdentityResolution,
    ) -> TargetIdentityResolution:
        fixture = self.fixture_mapping.get(unresolved.query.gene_symbol)
        if fixture is None:
            return unresolved
        return TargetIdentityResolution(
            query=unresolved.query,
            status="fixture",
            identity=fixture.identity.model_copy(deep=True),
            sources=[
                *unresolved.sources,
                IdentitySourceResult(
                    source="fixture",
                    status="ok",
                    queried=False,
                    provenance_class="fixture_fallback",
                    query=unresolved.query.gene_symbol,
                    matched_by="fixture",
                    canonical_symbol=fixture.identity.canonical_symbol,
                    ensembl_id=fixture.identity.ensembl_id,
                    uniprot_accession=fixture.identity.uniprot_accession,
                ),
            ],
            warnings=[
                *unresolved.warnings,
                f"Offline fixture used: {fixture.fixture_label}",
            ],
        )

    def _cache_key(
        self,
        query: TargetIdentityQuery,
        allowed_sources: Sequence[str],
    ) -> tuple[str, str, tuple[str, ...]]:
        return (
            query.gene_symbol,
            query.ensembl_id or "",
            tuple(sorted(set(allowed_sources))),
        )

    def _cached(
        self,
        query: TargetIdentityQuery,
        allowed_sources: Sequence[str],
    ) -> TargetIdentityResolution | None:
        key = self._cache_key(query, allowed_sources)
        now = self._clock()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now - entry.stored_at > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            cached = entry.resolution.model_copy(deep=True)
        cached.sources = [
            source.model_copy(
                update={
                    "queried": False,
                    "cache_hit": True,
                    "provenance_class": (
                        "external_cache"
                        if source.provenance_class == "external_live"
                        else source.provenance_class
                    ),
                }
            )
            for source in cached.sources
        ]
        return cached

    def _store(
        self,
        query: TargetIdentityQuery,
        resolution: TargetIdentityResolution,
        allowed_sources: Sequence[str],
    ) -> None:
        entry = _CacheEntry(
            stored_at=self._clock(),
            resolution=resolution.model_copy(deep=True),
        )
        with self._cache_lock:
            self._cache[self._cache_key(query, allowed_sources)] = entry


def _clean_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _valid_ensembl(value: Any) -> str | None:
    cleaned = _clean_string(value)
    if cleaned and len(cleaned) == 15 and cleaned.startswith("ENSG") and cleaned[4:].isdigit():
        return cleaned
    return None


def _organism_taxon(item: Mapping[str, Any]) -> int | None:
    organism = item.get("organism")
    if not isinstance(organism, Mapping):
        return None
    taxon = organism.get("taxonId")
    try:
        return int(taxon)
    except (TypeError, ValueError):
        return None


def _uniprot_primary_gene(item: Mapping[str, Any]) -> str | None:
    genes = item.get("genes")
    if not isinstance(genes, Sequence) or not genes:
        return None
    first = genes[0]
    if not isinstance(first, Mapping):
        return None
    gene_name = first.get("geneName")
    if not isinstance(gene_name, Mapping):
        return None
    return _clean_string(gene_name.get("value"))


def _is_reviewed_uniprot_entry(item: Mapping[str, Any]) -> bool:
    """Return true only for a Swiss-Prot reviewed entry.

    A substring test is unsafe because UniProt's literal label
    ``UniProtKB unreviewed (TrEMBL)`` also contains the word ``reviewed``.
    The stable API label begins with ``UniProtKB reviewed`` for curated
    Swiss-Prot records.
    """

    entry_type = str(item.get("entryType") or "").strip().casefold()
    return entry_type.startswith("uniprotkb reviewed")


def _uniprot_protein_name(item: Mapping[str, Any]) -> str | None:
    description = item.get("proteinDescription")
    if not isinstance(description, Mapping):
        return None
    recommended = description.get("recommendedName")
    if isinstance(recommended, Mapping):
        full_name = recommended.get("fullName")
        if isinstance(full_name, Mapping):
            name = _clean_string(full_name.get("value"))
            if name:
                return name
    return None


def _uniprot_ensembl_ids(item: Mapping[str, Any]) -> list[str]:
    references = item.get("uniProtKBCrossReferences")
    if not isinstance(references, Sequence):
        return []
    identifiers: list[str] = []
    for reference in references:
        if not isinstance(reference, Mapping) or reference.get("database") != "Ensembl":
            continue
        candidates: list[Any] = [reference.get("id")]
        properties = reference.get("properties")
        if isinstance(properties, Sequence):
            candidates.extend(
                property_item.get("value")
                for property_item in properties
                if isinstance(property_item, Mapping)
                and property_item.get("key") in {"GeneId", "Gene ID"}
            )
        for candidate in candidates:
            ensembl = _valid_ensembl(str(candidate).split(".", maxsplit=1)[0] if candidate else None)
            if ensembl:
                identifiers.append(ensembl)
    return list(dict.fromkeys(identifiers))


__all__ = [
    "CanonicalTargetIdentity",
    "FixtureTargetIdentity",
    "IdentitySourceResult",
    "OPEN_TARGETS_GRAPHQL_URL",
    "TargetIdentityQuery",
    "TargetIdentityResolution",
    "TargetIdentityResolver",
    "UNIPROT_SEARCH_URL",
]
