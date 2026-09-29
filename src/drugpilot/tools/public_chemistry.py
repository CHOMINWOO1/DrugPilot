from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path
import re
from threading import Lock
from typing import Any, Literal, Protocol, Sequence
from urllib.parse import quote

import httpx
from pydantic import Field, model_validator

from drugpilot.domain.models import EvidenceRef, StrictModel
from drugpilot.tools.http import classify_transport_error, get_scientific_http_client


PublicSource = Literal[
    "ChEMBL",
    "PubChem",
    "BindingDB",
    "DrugCentral",
    "RCSB PDB",
    "AlphaFold DB",
]
SourceStatus = Literal["ok", "empty", "skipped", "not_configured", "error"]


class ChemistryLookup(StrictModel):
    """Identifiers available when MA2 enriches a candidate and its target.

    Chemical identifiers are used by ChEMBL and PubChem.  A UniProt accession
    is used independently by RCSB PDB and AlphaFold DB.  At least one side must
    be present, so the same model also supports target-only structure lookup.
    """

    candidate_name: str | None = Field(default=None, max_length=500)
    smiles: str | None = Field(default=None, max_length=8_000)
    inchikey: str | None = Field(default=None, max_length=100)
    uniprot_id: str | None = Field(default=None, max_length=30)
    limit_per_source: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def require_identifier(self) -> "ChemistryLookup":
        values = (self.candidate_name, self.smiles, self.inchikey, self.uniprot_id)
        if not any(value and value.strip() for value in values):
            raise ValueError("at least one chemical or UniProt identifier is required")
        for field_name in ("candidate_name", "smiles", "inchikey", "uniprot_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, value.strip() or None)
        return self

    @property
    def chemical_query(self) -> tuple[str, str] | None:
        if self.inchikey:
            return "inchikey", self.inchikey
        if self.smiles:
            return "smiles", self.smiles
        if self.candidate_name:
            return "name", self.candidate_name
        return None


class SourceEnrichment(StrictModel):
    source: PublicSource
    status: SourceStatus
    queried: bool
    cache_hit: bool = False
    record_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    detail: str | None = None


class SourceRegistryEntry(StrictModel):
    """Configuration truth for one source; it is not a reachability claim."""

    source: PublicSource
    access_mode: Literal["live_api", "local_dataset"]
    runtime_state: Literal["ready", "not_configured", "missing_dataset"]
    requires_api_key: bool = False
    endpoint: str
    documentation_url: str
    dataset_path: str | None = None
    note: str


class PublicChemistryEnrichment(StrictModel):
    """Audit-ready enrichment result that can be attached to a CandidateCard.

    ``evidence`` can be appended directly to ``CandidateCard.evidence``.  The
    source summaries make empty, skipped and failed sources explicit instead
    of silently treating missing data as negative evidence.
    """

    lookup: ChemistryLookup
    evidence: list[EvidenceRef] = Field(default_factory=list)
    sources: list[SourceEnrichment] = Field(default_factory=list)


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


def _evidence_id(prefix: str, external_id: str) -> str:
    digest = sha256(f"{prefix}|{external_id}".encode("utf-8")).hexdigest()[:12]
    return f"ev_{prefix}_{digest}"


def _excerpt_hash(title: str) -> str:
    return sha256(title.encode("utf-8")).hexdigest()


def _compact_metadata(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _source_error_type(exc: Exception) -> str:
    """Preserve an upstream HTTP status instead of calling it connectivity."""

    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return f"upstream_http_{exc.response.status_code}"
    if isinstance(exc, httpx.TransportError):
        return classify_transport_error(exc)
    return type(exc).__name__


class _CachedSourceAdapter:
    source: PublicSource
    endpoint: str

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.client: HTTPClient = client or get_scientific_http_client()
        self.timeout_seconds = timeout_seconds
        self._cache: dict[tuple[Any, ...], tuple[EvidenceRef, ...]] = {}
        self._cache_lock = Lock()

    def _cached(self, key: tuple[Any, ...]) -> tuple[EvidenceRef, ...] | None:
        with self._cache_lock:
            if key not in self._cache:
                return None
            records = self._cache[key]
        return tuple(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "cache_hit": True,
                        "provenance_class": "external_cache",
                    }
                },
                deep=True,
            )
            for record in records
        )

    def _store(
        self,
        key: tuple[Any, ...],
        records: list[EvidenceRef],
    ) -> tuple[EvidenceRef, ...]:
        frozen = tuple(record.model_copy(deep=True) for record in records)
        with self._cache_lock:
            self._cache[key] = frozen
        return tuple(record.model_copy(deep=True) for record in frozen)

    def _summary(
        self,
        records: tuple[EvidenceRef, ...],
        *,
        cache_hit: bool,
    ) -> tuple[tuple[EvidenceRef, ...], SourceEnrichment]:
        return records, SourceEnrichment(
            source=self.source,
            status="ok" if records else "empty",
            queried=True,
            cache_hit=cache_hit,
            record_count=len(records),
        )


class ChEMBLAdapter(_CachedSourceAdapter):
    """Read-only compact molecule lookup through the official ChEMBL API."""

    source: PublicSource = "ChEMBL"
    endpoint = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
    molecule_endpoint = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

    def lookup(
        self,
        request: ChemistryLookup,
    ) -> tuple[tuple[EvidenceRef, ...], SourceEnrichment]:
        chemical_query = request.chemical_query
        if chemical_query is None:
            return (), SourceEnrichment(source=self.source, status="skipped", queried=False)
        query_type, query_value = chemical_query
        key = (query_type, query_value.casefold(), request.limit_per_source)
        cached = self._cached(key)
        if cached is not None:
            return self._summary(cached, cache_hit=True)

        if query_type == "name":
            endpoint = self.endpoint
            params = {"q": query_value, "limit": request.limit_per_source}
            query_semantics = "full_text_name_search"
        elif query_type == "inchikey":
            endpoint = self.molecule_endpoint
            params = {
                "molecule_structures__standard_inchi_key": query_value,
                "limit": request.limit_per_source,
            }
            query_semantics = "exact_standard_inchi_key"
        else:
            # ChEMBL documents this connectivity operator specifically for
            # SMILES.  It intentionally ignores stereochemical differences;
            # retain that scope in provenance instead of mislabelling a
            # full-text hit as exact molecular identity.
            endpoint = self.molecule_endpoint
            params = {
                "molecule_structures__canonical_smiles__connectivity": query_value,
                "limit": request.limit_per_source,
            }
            query_semantics = "smiles_connectivity"

        response = self.client.get(
            endpoint,
            params=params,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        molecules = payload.get("molecules", [])
        if not isinstance(molecules, list):
            raise ValueError("ChEMBL response has no molecule list")

        records: list[EvidenceRef] = []
        for molecule in molecules[: request.limit_per_source]:
            if not isinstance(molecule, dict):
                continue
            chembl_id = molecule.get("molecule_chembl_id")
            if not chembl_id:
                continue
            structures = molecule.get("molecule_structures") or {}
            if not isinstance(structures, dict):
                structures = {}
            title = molecule.get("pref_name") or f"ChEMBL molecule {chembl_id}"
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id("chembl", str(chembl_id)),
                    source=self.source,
                    external_id=str(chembl_id),
                    title=str(title),
                    url=f"https://www.ebi.ac.uk/chembl/explore/compound/{quote(str(chembl_id))}",
                    supports="context",
                    excerpt_hash=_excerpt_hash(str(title)),
                    metadata=_compact_metadata(
                        live=True,
                        mock=False,
                        cache_hit=False,
                        provenance_class="external_live",
                        retrieval_adapter="ChEMBLAdapter",
                        api_endpoint=endpoint,
                        query_type=query_type,
                        query_semantics=query_semantics,
                        molecule_type=molecule.get("molecule_type"),
                        max_phase=molecule.get("max_phase"),
                        first_approval=molecule.get("first_approval"),
                        canonical_smiles=structures.get("canonical_smiles"),
                        standard_inchi_key=structures.get("standard_inchi_key"),
                        evidence_scope="compound identity and development metadata",
                    ),
                )
            )
        stored = self._store(key, records)
        return self._summary(stored, cache_hit=False)


class PubChemAdapter(_CachedSourceAdapter):
    """Read-only compound identity and physicochemical metadata from PUG REST."""

    source: PublicSource = "PubChem"
    endpoint = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    _properties = (
        "IUPACName,CanonicalSMILES,IsomericSMILES,InChIKey,"
        "MolecularFormula,MolecularWeight"
    )

    def lookup(
        self,
        request: ChemistryLookup,
    ) -> tuple[tuple[EvidenceRef, ...], SourceEnrichment]:
        chemical_query = request.chemical_query
        if chemical_query is None:
            return (), SourceEnrichment(source=self.source, status="skipped", queried=False)
        query_type, query_value = chemical_query
        namespace = {"inchikey": "inchikey", "name": "name", "smiles": "smiles"}[query_type]
        key = (namespace, query_value.casefold(), request.limit_per_source)
        cached = self._cached(key)
        if cached is not None:
            return self._summary(cached, cache_hit=True)

        encoded = quote(query_value, safe="")
        url = f"{self.endpoint}/compound/{namespace}/{encoded}/property/{self._properties}/JSON"
        response = self.client.get(
            url,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        properties = payload.get("PropertyTable", {}).get("Properties", [])
        if not isinstance(properties, list):
            raise ValueError("PubChem response has no property list")

        records: list[EvidenceRef] = []
        for item in properties[: request.limit_per_source]:
            if not isinstance(item, dict):
                continue
            cid = item.get("CID")
            if cid is None:
                continue
            title = item.get("IUPACName") or f"PubChem compound CID {cid}"
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id("pubchem", str(cid)),
                    source=self.source,
                    external_id=str(cid),
                    title=str(title),
                    url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                    supports="context",
                    excerpt_hash=_excerpt_hash(str(title)),
                    metadata=_compact_metadata(
                        live=True,
                        mock=False,
                        cache_hit=False,
                        provenance_class="external_live",
                        retrieval_adapter="PubChemAdapter",
                        api_endpoint=self.endpoint,
                        query_type=query_type,
                        iupac_name=item.get("IUPACName"),
                        canonical_smiles=item.get("CanonicalSMILES")
                        or item.get("ConnectivitySMILES"),
                        isomeric_smiles=item.get("IsomericSMILES") or item.get("SMILES"),
                        canonical_smiles_source_field=(
                            "CanonicalSMILES" if item.get("CanonicalSMILES") else
                            "ConnectivitySMILES" if item.get("ConnectivitySMILES") else None
                        ),
                        isomeric_smiles_source_field=(
                            "IsomericSMILES" if item.get("IsomericSMILES") else
                            "SMILES" if item.get("SMILES") else None
                        ),
                        inchikey=item.get("InChIKey"),
                        molecular_formula=item.get("MolecularFormula"),
                        molecular_weight=item.get("MolecularWeight"),
                        evidence_scope="compound identity and physicochemical metadata",
                    ),
                )
            )
        stored = self._store(key, records)
        return self._summary(stored, cache_hit=False)


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(_walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_dicts(child))
    return found


def _normalized_mapping(row: dict[str, Any]) -> dict[str, Any]:
    return {
        re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
        for key, value in row.items()
    }


def _first(mapping: dict[str, Any], *aliases: str) -> Any:
    for alias in aliases:
        value = mapping.get(re.sub(r"[^a-z0-9]", "", alias.casefold()))
        if value not in (None, ""):
            return value
    return None


class BindingDBAdapter(_CachedSourceAdapter):
    """Read-only target-affinity records from BindingDB's official REST service."""

    source: PublicSource = "BindingDB"
    endpoint = "https://bindingdb.org/rest/getLigandsByUniprots"

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 8.0,
        affinity_cutoff_nm: int = 10_000,
    ) -> None:
        super().__init__(client=client, timeout_seconds=timeout_seconds)
        if affinity_cutoff_nm <= 0:
            raise ValueError("affinity_cutoff_nm must be positive")
        self.affinity_cutoff_nm = affinity_cutoff_nm

    def lookup(
        self,
        request: ChemistryLookup,
    ) -> tuple[tuple[EvidenceRef, ...], SourceEnrichment]:
        if not request.uniprot_id:
            return (), SourceEnrichment(source=self.source, status="skipped", queried=False)
        accession = request.uniprot_id.upper()
        key = (accession, self.affinity_cutoff_nm, request.limit_per_source)
        cached = self._cached(key)
        if cached is not None:
            return self._summary(cached, cache_hit=True)

        response = self.client.get(
            self.endpoint,
            params={
                "uniprot": accession,
                "cutoff": self.affinity_cutoff_nm,
                "response": "application/json",
            },
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()

        records: list[EvidenceRef] = []
        seen: set[str] = set()
        for raw in _walk_dicts(payload):
            item = _normalized_mapping(raw)
            monomer_id = _first(
                item,
                "monomerid",
                "bindingdbmonomerid",
                "bdbm",
                "ligandid",
            )
            # BindingDB's current REST response uses the singular ``smile``;
            # older snapshots and wrappers have also used the plural aliases.
            smiles = _first(item, "smile", "smiles", "ligandsmiles")
            affinity_type = _first(item, "affinitytype", "type")
            affinity_value = _first(item, "affinity", "affinityvalue", "value")
            ic50 = _first(item, "ic50")
            ki = _first(item, "ki")
            kd = _first(item, "kd")
            if monomer_id is None or not any(
                value is not None for value in (smiles, affinity_value, ic50, ki, kd)
            ):
                continue
            external_id = str(monomer_id)
            if external_id in seen:
                continue
            seen.add(external_id)
            label = (
                f"{affinity_type} {affinity_value}"
                if affinity_type and affinity_value
                else f"IC50 {ic50}" if ic50
                else f"Ki {ki}" if ki
                else f"Kd {kd}" if kd
                else "binding record"
            )
            title = f"BindingDB {external_id} against {accession}: {label}"
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id("bindingdb", external_id),
                    source=self.source,
                    external_id=external_id,
                    title=title,
                    url=f"https://bindingdb.org/rwd/bind/chemsearch/marvin/MolStructure.jsp?monomerid={quote(external_id)}",
                    supports="context",
                    excerpt_hash=_excerpt_hash(title),
                    metadata=_compact_metadata(
                        live=True,
                        mock=False,
                        cache_hit=False,
                        provenance_class="external_live",
                        retrieval_adapter="BindingDBAdapter",
                        api_endpoint=self.endpoint,
                        query_type="uniprot_affinity_cutoff",
                        uniprot_accession=accession,
                        affinity_cutoff_nm=self.affinity_cutoff_nm,
                        affinity_type=affinity_type,
                        affinity_value=affinity_value,
                        ic50=ic50,
                        ki=ki,
                        kd=kd,
                        smiles=smiles,
                        evidence_scope="measured target-ligand affinity metadata",
                    ),
                )
            )
            if len(records) >= request.limit_per_source:
                break
        stored = self._store(key, records)
        return self._summary(stored, cache_hit=False)


class DrugCentralLocalAdapter(_CachedSourceAdapter):
    """Read an official DrugCentral TSV snapshot from local storage.

    DrugCentral's documented, complete distribution path is a PostgreSQL dump
    and downloadable TSV files.  This adapter therefore never implies that an
    unstable web endpoint is active.  Without ``dataset_path`` it reports
    ``not_configured`` and performs no I/O.
    """

    source: PublicSource = "DrugCentral"
    endpoint = "https://drugcentral.org/download"

    def __init__(
        self,
        *,
        dataset_path: str | Path | None = None,
        dataset_version: str | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        super().__init__(client=None, timeout_seconds=timeout_seconds)
        self.dataset_path = Path(dataset_path).expanduser() if dataset_path else None
        self.dataset_version = dataset_version

    @property
    def runtime_state(self) -> Literal["ready", "not_configured", "missing_dataset"]:
        if self.dataset_path is None:
            return "not_configured"
        if not self.dataset_path.is_file():
            return "missing_dataset"
        return "ready"

    def lookup(
        self,
        request: ChemistryLookup,
    ) -> tuple[tuple[EvidenceRef, ...], SourceEnrichment]:
        if self.runtime_state == "not_configured":
            return (), SourceEnrichment(
                source=self.source,
                status="not_configured",
                queried=False,
                error_type="local_dataset_not_configured",
            )
        if self.runtime_state == "missing_dataset":
            raise FileNotFoundError(str(self.dataset_path))
        assert self.dataset_path is not None

        query_parts = (
            request.candidate_name,
            request.smiles,
            request.inchikey,
            request.uniprot_id,
        )
        key = (
            str(self.dataset_path.resolve()),
            self.dataset_path.stat().st_mtime_ns,
            *(value.casefold() if value else None for value in query_parts),
            request.limit_per_source,
        )
        cached = self._cached(key)
        if cached is not None:
            return self._summary(cached, cache_hit=True)

        records: list[EvidenceRef] = []
        with self.dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError("DrugCentral TSV has no header")
            for row_number, raw in enumerate(reader, start=2):
                row = _normalized_mapping(raw)
                if not self._matches(request, row):
                    continue
                drug_id = _first(row, "struct_id", "drugcentral_id", "drug_id", "id")
                name = _first(row, "name", "drug_name", "inn", "preferred_name")
                accession = _first(row, "accession", "uniprot", "uniprot_id")
                external_id = str(drug_id or f"row-{row_number}")
                title = str(name or f"DrugCentral record {external_id}")
                records.append(
                    EvidenceRef(
                        evidence_id=_evidence_id(
                            "drugcentral",
                            f"{self.dataset_path.name}|{external_id}|{row_number}",
                        ),
                        source=self.source,
                        external_id=external_id,
                        title=title,
                        url="https://drugcentral.org/",
                        supports="context",
                        excerpt_hash=_excerpt_hash(title),
                        metadata=_compact_metadata(
                            live=False,
                            mock=False,
                            cache_hit=False,
                            provenance_class="external_local_dataset",
                            retrieval_adapter="DrugCentralLocalAdapter",
                            official_download_url=self.endpoint,
                            dataset_filename=self.dataset_path.name,
                            dataset_version=self.dataset_version,
                            dataset_row_number=row_number,
                            drug_name=name,
                            uniprot_accession=accession,
                            gene=_first(row, "gene", "gene_symbol"),
                            action_type=_first(row, "action_type", "action"),
                            moa=_first(row, "moa", "mechanism_of_action"),
                            activity_type=_first(row, "act_type", "activity_type"),
                            activity_value=_first(row, "act_value", "activity_value"),
                            activity_unit=_first(row, "act_unit", "activity_unit"),
                            smiles=_first(row, "smiles"),
                            inchikey=_first(row, "inchikey", "inchi_key"),
                            evidence_scope="downloaded drug-target and approval metadata",
                        ),
                    )
                )
                if len(records) >= request.limit_per_source:
                    break
        stored = self._store(key, records)
        return self._summary(stored, cache_hit=False)

    @staticmethod
    def _matches(request: ChemistryLookup, row: dict[str, Any]) -> bool:
        comparisons: list[bool] = []
        name = _first(row, "name", "drug_name", "inn", "preferred_name")
        smiles = _first(row, "smiles")
        inchikey = _first(row, "inchikey", "inchi_key")
        accession = _first(row, "accession", "uniprot", "uniprot_id")
        if request.candidate_name and name is not None:
            comparisons.append(str(name).casefold() == request.candidate_name.casefold())
        if request.smiles and smiles is not None:
            comparisons.append(str(smiles).strip() == request.smiles)
        if request.inchikey and inchikey is not None:
            comparisons.append(str(inchikey).casefold() == request.inchikey.casefold())
        if request.uniprot_id and accession is not None:
            comparisons.append(str(accession).casefold() == request.uniprot_id.casefold())
        return bool(comparisons) and all(comparisons)


class RCSBPDBAdapter(_CachedSourceAdapter):
    """Find experimental PDB entries mapped to a UniProt accession."""

    source: PublicSource = "RCSB PDB"
    endpoint = "https://search.rcsb.org/rcsbsearch/v2/query"

    def lookup(
        self,
        request: ChemistryLookup,
    ) -> tuple[tuple[EvidenceRef, ...], SourceEnrichment]:
        if not request.uniprot_id:
            return (), SourceEnrichment(source=self.source, status="skipped", queried=False)
        accession = request.uniprot_id.upper()
        key = (accession, request.limit_per_source)
        cached = self._cached(key)
        if cached is not None:
            return self._summary(cached, cache_hit=True)

        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": (
                        "rcsb_polymer_entity_container_identifiers."
                        "reference_sequence_identifiers.database_accession"
                    ),
                    "operator": "exact_match",
                    "value": accession,
                },
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": request.limit_per_source},
                "results_content_type": ["experimental"],
            },
        }
        response = self.client.post(
            self.endpoint,
            json=query,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        result_set = payload.get("result_set", [])
        if not isinstance(result_set, list):
            raise ValueError("RCSB response has no result set")

        records: list[EvidenceRef] = []
        seen: set[str] = set()
        for result in result_set:
            if not isinstance(result, dict) or not result.get("identifier"):
                continue
            pdb_id = str(result["identifier"]).split("_")[0].upper()
            if pdb_id in seen:
                continue
            seen.add(pdb_id)
            title = f"Experimental structure mapped to {accession}: PDB {pdb_id}"
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id("rcsb", pdb_id),
                    source=self.source,
                    external_id=pdb_id,
                    title=title,
                    url=f"https://www.rcsb.org/structure/{quote(pdb_id)}",
                    supports="context",
                    excerpt_hash=_excerpt_hash(title),
                    metadata=_compact_metadata(
                        live=True,
                        mock=False,
                        cache_hit=False,
                        provenance_class="external_live",
                        retrieval_adapter="RCSBPDBAdapter",
                        api_endpoint=self.endpoint,
                        query_type="uniprot_exact_match",
                        uniprot_accession=accession,
                        search_score=result.get("score"),
                        evidence_scope="experimental target structure availability",
                    ),
                )
            )
            if len(records) >= request.limit_per_source:
                break
        stored = self._store(key, records)
        return self._summary(stored, cache_hit=False)


class AlphaFoldDBAdapter(_CachedSourceAdapter):
    """Retrieve predicted-structure metadata for a UniProt accession."""

    source: PublicSource = "AlphaFold DB"
    endpoint = "https://alphafold.ebi.ac.uk/api/prediction"

    def lookup(
        self,
        request: ChemistryLookup,
    ) -> tuple[tuple[EvidenceRef, ...], SourceEnrichment]:
        if not request.uniprot_id:
            return (), SourceEnrichment(source=self.source, status="skipped", queried=False)
        accession = request.uniprot_id.upper()
        key = (accession, request.limit_per_source)
        cached = self._cached(key)
        if cached is not None:
            return self._summary(cached, cache_hit=True)

        response = self.client.get(
            f"{self.endpoint}/{quote(accession)}",
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        entries = payload if isinstance(payload, list) else payload.get("results", [])
        if not isinstance(entries, list):
            raise ValueError("AlphaFold response has no prediction list")

        records: list[EvidenceRef] = []
        for entry in entries[: request.limit_per_source]:
            if not isinstance(entry, dict):
                continue
            external_id = entry.get("entryId") or entry.get("uniprotAccession")
            if not external_id:
                continue
            entry_accession = entry.get("uniprotAccession") or accession
            description = entry.get("uniprotDescription")
            title = description or f"AlphaFold predicted structure for {entry_accession}"
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id("alphafold", str(external_id)),
                    source=self.source,
                    external_id=str(external_id),
                    title=str(title),
                    url=f"https://alphafold.ebi.ac.uk/entry/{quote(str(entry_accession))}",
                    supports="context",
                    excerpt_hash=_excerpt_hash(str(title)),
                    metadata=_compact_metadata(
                        live=True,
                        mock=False,
                        cache_hit=False,
                        provenance_class="external_live",
                        retrieval_adapter="AlphaFoldDBAdapter",
                        api_endpoint=self.endpoint,
                        query_type="uniprot_accession",
                        uniprot_accession=entry_accession,
                        gene=entry.get("gene"),
                        organism=entry.get("organismScientificName"),
                        tax_id=entry.get("taxId"),
                        model_confidence=entry.get("globalMetricValue"),
                        latest_version=entry.get("latestVersion"),
                        cif_url=entry.get("cifUrl"),
                        pdb_url=entry.get("pdbUrl"),
                        pae_url=entry.get("paeDocUrl"),
                        evidence_scope="predicted target structure availability",
                    ),
                )
            )
        stored = self._store(key, records)
        return self._summary(stored, cache_hit=False)


class PublicChemistryEnricher:
    """Aggregate keyless public sources without coupling them to MA2 control flow."""

    def __init__(
        self,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 8.0,
        chembl: ChEMBLAdapter | None = None,
        pubchem: PubChemAdapter | None = None,
        bindingdb: BindingDBAdapter | None = None,
        drugcentral: DrugCentralLocalAdapter | None = None,
        drugcentral_dataset_path: str | Path | None = None,
        drugcentral_dataset_version: str | None = None,
        rcsb: RCSBPDBAdapter | None = None,
        alphafold: AlphaFoldDBAdapter | None = None,
    ) -> None:
        self.drugcentral = drugcentral or DrugCentralLocalAdapter(
            dataset_path=drugcentral_dataset_path,
            dataset_version=drugcentral_dataset_version,
        )
        self.adapters = (
            chembl or ChEMBLAdapter(client=client, timeout_seconds=timeout_seconds),
            pubchem or PubChemAdapter(client=client, timeout_seconds=timeout_seconds),
            bindingdb or BindingDBAdapter(client=client, timeout_seconds=timeout_seconds),
            self.drugcentral,
            rcsb or RCSBPDBAdapter(client=client, timeout_seconds=timeout_seconds),
            alphafold or AlphaFoldDBAdapter(client=client, timeout_seconds=timeout_seconds),
        )

    def registry(self) -> list[SourceRegistryEntry]:
        """Describe configured access paths without claiming live connectivity."""

        api_specs: list[tuple[PublicSource, str, str, str]] = [
            (
                "ChEMBL",
                ChEMBLAdapter.endpoint,
                "https://www.ebi.ac.uk/chembl/api/data/docs",
                "Keyless REST adapter; reachability is recorded by each query result.",
            ),
            (
                "PubChem",
                PubChemAdapter.endpoint,
                "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
                "Keyless PUG REST adapter; reachability is recorded by each query result.",
            ),
            (
                "BindingDB",
                BindingDBAdapter.endpoint,
                "https://bindingdb.org/rwd/bind/BindingDBRESTfulAPI.jsp",
                "Keyless REST adapter using UniProt and an affinity cutoff.",
            ),
            (
                "RCSB PDB",
                RCSBPDBAdapter.endpoint,
                "https://www.rcsb.org/docs/programmatic-access/web-apis-overview",
                "Keyless Search API adapter for experimental structures.",
            ),
            (
                "AlphaFold DB",
                AlphaFoldDBAdapter.endpoint,
                "https://alphafold.ebi.ac.uk/api/",
                "Keyless prediction metadata adapter keyed by UniProt accession.",
            ),
        ]
        entries = [
            SourceRegistryEntry(
                source=source,
                access_mode="live_api",
                runtime_state="ready",
                endpoint=endpoint,
                documentation_url=documentation,
                note=note,
            )
            for source, endpoint, documentation, note in api_specs
        ]
        entries.insert(
            3,
            SourceRegistryEntry(
                source="DrugCentral",
                access_mode="local_dataset",
                runtime_state=self.drugcentral.runtime_state,
                endpoint=DrugCentralLocalAdapter.endpoint,
                documentation_url="https://drugcentral.org/download",
                dataset_path=(
                    str(self.drugcentral.dataset_path)
                    if self.drugcentral.dataset_path is not None
                    else None
                ),
                note=(
                    "Uses the official downloaded TSV/PostgreSQL distribution; "
                    "no live REST availability is implied."
                ),
            ),
        )
        return entries

    def enrich(
        self,
        request: ChemistryLookup,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> PublicChemistryEnrichment:
        evidence: list[EvidenceRef] = []
        summaries: list[SourceEnrichment] = []
        allowed = None if allowed_sources is None else frozenset(allowed_sources)
        for adapter in self.adapters:
            if allowed is not None and adapter.source not in allowed:
                summaries.append(
                    SourceEnrichment(
                        source=adapter.source,
                        status="skipped",
                        queried=False,
                        error_type="policy_rejected",
                        detail=(
                            "ResearchBrief.data_policy.allowed_sources excludes "
                            "this source"
                        ),
                    )
                )
                continue
            try:
                records, summary = adapter.lookup(request)
            except Exception as exc:  # isolate public-service and schema failures by source
                records = ()
                summary = SourceEnrichment(
                    source=adapter.source,
                    status="error",
                    queried=True,
                    error_type=_source_error_type(exc),
                )
            evidence.extend(record.model_copy(deep=True) for record in records)
            summaries.append(summary)
        return PublicChemistryEnrichment(
            lookup=request.model_copy(deep=True),
            evidence=evidence,
            sources=summaries,
        )


__all__ = [
    "AlphaFoldDBAdapter",
    "BindingDBAdapter",
    "ChEMBLAdapter",
    "ChemistryLookup",
    "DrugCentralLocalAdapter",
    "PubChemAdapter",
    "PublicChemistryEnricher",
    "PublicChemistryEnrichment",
    "RCSBPDBAdapter",
    "SourceEnrichment",
    "SourceRegistryEntry",
]
