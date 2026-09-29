"""Conditional, read-only adapters for restricted or metered data sources.

These sources cannot truthfully be presented as generally live:

* DrugCentral publishes downloadable TSV/PostgreSQL data.  The adapter below
  reuses :class:`DrugCentralLocalAdapter` and never scrapes the website.
* DrugBank requires a licensed API key sent in the ``Authorization`` header.
* Google Patents Public Datasets require a Google Cloud project, BigQuery job
  credentials, and may incur query charges.

Every public ``query`` method converts missing configuration into an explicit
``not_configured`` result.  No fixture record is manufactured and no empty
result is interpreted as proof that a drug is safe or that an invention is
novel.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from threading import Lock
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence
import unicodedata
from urllib.parse import quote

import httpx
from pydantic import Field, field_validator, model_validator

from drugpilot.domain.models import EvidenceRef, StrictModel
from drugpilot.tools.http import classify_transport_error, get_scientific_http_client
from drugpilot.tools.public_chemistry import ChemistryLookup, DrugCentralLocalAdapter


RestrictedSource = Literal["DrugCentral", "DrugBank", "Google Patents"]
RestrictedOperation = Literal[
    "repurposing",
    "identity_regulatory",
    "contraindication",
    "patent_novelty",
]
RestrictedStatus = Literal["ok", "empty", "skipped", "not_configured", "error"]


DRUGCENTRAL_DOWNLOAD_DOCUMENTATION = "https://drugcentral.org/download"
DRUGBANK_DISCOVERY_DOCUMENTATION = "https://docs.drugbank.com/discovery/v1/"
DRUGBANK_API_DOCUMENTATION = "https://docs.drugbank.com/v1/"
GOOGLE_PATENTS_DATASET_DOCUMENTATION = (
    "https://console.cloud.google.com/marketplace/product/"
    "google_patents_public_datasets/google-patents-public-data"
)
BIGQUERY_PUBLIC_DATA_DOCUMENTATION = "https://cloud.google.com/bigquery/public-data"
GOOGLE_PATENTS_TABLE = "patents-public-data.google_patents_research.publications"


class RestrictedSourceQuery(StrictModel):
    """A bounded query shared by the three conditional adapters."""

    operation: RestrictedOperation
    candidate_name: str | None = Field(default=None, max_length=500)
    drug_id: str | None = Field(default=None, max_length=100)
    inchikey: str | None = Field(default=None, max_length=100)
    target_uniprot: str | None = Field(default=None, max_length=30)
    target_name: str | None = Field(default=None, max_length=300)
    condition: str | None = Field(default=None, max_length=500)
    novelty_terms: tuple[str, ...] = Field(default=(), max_length=12)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator(
        "candidate_name",
        "drug_id",
        "inchikey",
        "target_uniprot",
        "target_name",
        "condition",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("novelty_terms", mode="before")
    @classmethod
    def normalize_terms(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values: Iterable[object] = (value,)
        else:
            raw_values = value  # type: ignore[assignment]
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in raw_values:
            term = str(raw).strip()
            if not term:
                continue
            if len(term) > 100:
                raise ValueError("each novelty term must be at most 100 characters")
            folded = term.casefold()
            if folded not in seen:
                seen.add(folded)
                normalized.append(term)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_operation_inputs(self) -> "RestrictedSourceQuery":
        if self.operation == "repurposing" and not (
            self.target_uniprot or self.target_name
        ):
            raise ValueError("repurposing requires target_uniprot or target_name")
        if self.operation == "identity_regulatory" and not (
            self.candidate_name or self.drug_id or self.inchikey
        ):
            raise ValueError(
                "identity_regulatory requires candidate_name, drug_id, or inchikey"
            )
        if self.operation == "contraindication" and not (
            self.candidate_name or self.drug_id
        ):
            raise ValueError("contraindication requires candidate_name or drug_id")
        if self.operation == "patent_novelty" and not self.novelty_terms:
            raise ValueError("patent_novelty requires at least one novelty term")
        return self


class RestrictedSourceSummary(StrictModel):
    source: RestrictedSource
    operation: RestrictedOperation
    status: RestrictedStatus
    queried: bool
    cache_hit: bool = False
    record_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    documentation_url: str
    required_configuration: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class RestrictedSourceResult(StrictModel):
    query: RestrictedSourceQuery
    evidence: list[EvidenceRef] = Field(default_factory=list)
    summary: RestrictedSourceSummary


@dataclass(frozen=True, slots=True)
class _QueryOutcome:
    records: tuple[EvidenceRef, ...]
    provider_cache_hit: bool = False


class _NotConfigured(RuntimeError):
    pass


def _evidence_id(prefix: str, external_id: str) -> str:
    digest = sha256(f"{prefix}|{external_id}".encode("utf-8")).hexdigest()[:12]
    return f"ev_{prefix}_{digest}"


def _text_hash(value: object) -> str | None:
    text = str(value or "").strip()
    return sha256(text.encode("utf-8")).hexdigest() if text else None


def _normalized_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
        for key, value in row.items()
    }


def _first(mapping: Mapping[str, Any], *aliases: str) -> Any:
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.casefold())
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalized_exact_name(value: Any) -> str:
    """Normalize presentation differences without broadening exact identity."""

    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _normalized_identity_key(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


def _error_type(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return f"upstream_http_{exc.response.status_code}"
    if isinstance(exc, httpx.TransportError):
        return classify_transport_error(exc)
    if isinstance(exc, ModuleNotFoundError):
        return "client_dependency_missing"
    name = type(exc).__name__
    if name == "DefaultCredentialsError":
        return "application_default_credentials_missing"
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).casefold()


class _ConditionalReadOnlyAdapter:
    source: RestrictedSource
    documentation_url: str
    required_configuration: tuple[str, ...]
    limitations: tuple[str, ...]
    supported_operations: frozenset[RestrictedOperation]

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], _QueryOutcome] = {}
        self._cache_lock = Lock()

    def query(self, request: RestrictedSourceQuery) -> RestrictedSourceResult:
        """Execute a configured read-only query and retain truthful provenance."""

        if request.operation not in self.supported_operations:
            return self._result(
                request,
                status="skipped",
                queried=False,
                error_type="operation_not_supported_by_source",
            )

        configuration_error = self._configuration_error(request)
        if configuration_error:
            return self._result(
                request,
                status="not_configured",
                queried=False,
                error_type=configuration_error,
            )

        cache_key = self._cache_key(request)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            records = tuple(self._as_cache_record(item) for item in cached.records)
            return self._result(
                request,
                status="ok" if records else "empty",
                queried=True,
                cache_hit=True,
                records=records,
            )

        try:
            outcome = self._perform(request)
        except _NotConfigured as exc:
            return self._result(
                request,
                status="not_configured",
                queried=False,
                error_type=str(exc),
            )
        except Exception as exc:  # isolate source, credential, and schema failures
            configuration_error = self._runtime_configuration_error(exc)
            if configuration_error:
                return self._result(
                    request,
                    status="not_configured",
                    queried=False,
                    error_type=configuration_error,
                )
            return self._result(
                request,
                status="error",
                queried=True,
                error_type=_error_type(exc),
            )

        records = outcome.records
        if outcome.provider_cache_hit:
            records = tuple(self._as_cache_record(item) for item in records)
        frozen = _QueryOutcome(
            records=tuple(item.model_copy(deep=True) for item in records),
            provider_cache_hit=outcome.provider_cache_hit,
        )
        with self._cache_lock:
            self._cache[cache_key] = frozen
        return self._result(
            request,
            status="ok" if records else "empty",
            queried=True,
            cache_hit=outcome.provider_cache_hit,
            records=records,
        )

    def _result(
        self,
        request: RestrictedSourceQuery,
        *,
        status: RestrictedStatus,
        queried: bool,
        cache_hit: bool = False,
        records: Sequence[EvidenceRef] = (),
        error_type: str | None = None,
    ) -> RestrictedSourceResult:
        copied = [record.model_copy(deep=True) for record in records]
        return RestrictedSourceResult(
            query=request.model_copy(deep=True),
            evidence=copied,
            summary=RestrictedSourceSummary(
                source=self.source,
                operation=request.operation,
                status=status,
                queried=queried,
                cache_hit=cache_hit,
                record_count=len(copied),
                error_type=error_type,
                documentation_url=self.documentation_url,
                required_configuration=self.required_configuration,
                limitations=self.limitations,
            ),
        )

    @staticmethod
    def _as_cache_record(record: EvidenceRef) -> EvidenceRef:
        origin = str(record.metadata.get("provenance_class") or "unknown")
        if origin == "external_cache":
            origin = str(record.metadata.get("origin_provenance_class") or origin)
        return record.model_copy(
            update={
                "metadata": {
                    **record.metadata,
                    "live": False,
                    "cache_hit": True,
                    "provenance_class": "external_cache",
                    "origin_provenance_class": origin,
                }
            },
            deep=True,
        )

    def _cache_key(self, request: RestrictedSourceQuery) -> tuple[Any, ...]:
        payload = request.model_dump(mode="json")
        return (
            self.source,
            *(f"{key}={payload[key]!r}" for key in sorted(payload)),
            *self._configuration_fingerprint(),
        )

    def _configuration_fingerprint(self) -> tuple[Any, ...]:
        return ()

    def _configuration_error(self, request: RestrictedSourceQuery) -> str | None:
        raise NotImplementedError

    def _runtime_configuration_error(self, exc: Exception) -> str | None:
        return None

    def _perform(self, request: RestrictedSourceQuery) -> _QueryOutcome:
        raise NotImplementedError


class DrugCentralRestrictedAdapter(_ConditionalReadOnlyAdapter):
    """Extend the existing official-download DrugCentral adapter.

    ``repurposing`` always delegates to ``DrugCentralLocalAdapter``.  Identity,
    regulatory, withdrawal, and contraindication columns are additionally read
    from the configured TSV when present.  The adapter does not infer a clean
    safety status when those optional columns are absent.
    """

    source: RestrictedSource = "DrugCentral"
    documentation_url = DRUGCENTRAL_DOWNLOAD_DOCUMENTATION
    required_configuration = (
        "official DrugCentral drug-target TSV or compatible licensed export",
    )
    limitations = (
        "The public interaction TSV is not a complete contraindication source.",
        "Missing withdrawal/contraindication fields are unknown, not negative.",
        "The local snapshot can be older than the DrugCentral website.",
    )
    supported_operations = frozenset(
        {"repurposing", "identity_regulatory", "contraindication"}
    )

    _contraindication_aliases = (
        "contraindication",
        "contraindications",
        "boxed_warning",
        "warning",
    )
    _regulatory_aliases = (
        "regulatory_status",
        "approval_status",
        "approved",
        "fda_approved",
        "ema_approved",
        "pmda_approved",
        "withdrawn",
        "withdrawal_status",
        "market_status",
    )

    def __init__(
        self,
        *,
        dataset_path: str | Path | None = None,
        dataset_version: str | None = None,
        local_adapter: DrugCentralLocalAdapter | None = None,
        supplemental_dataset_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.local_adapter = local_adapter or DrugCentralLocalAdapter(
            dataset_path=dataset_path,
            dataset_version=dataset_version,
        )
        inherited_path = getattr(self.local_adapter, "dataset_path", None)
        selected_path = supplemental_dataset_path or inherited_path or dataset_path
        self.dataset_path = Path(selected_path).expanduser() if selected_path else None
        self.dataset_version = dataset_version or getattr(
            self.local_adapter, "dataset_version", None
        )

    @property
    def runtime_state(self) -> Literal["ready", "not_configured", "missing_dataset"]:
        state = getattr(self.local_adapter, "runtime_state", None)
        if state in {"ready", "not_configured", "missing_dataset"}:
            return state
        if self.dataset_path is None:
            return "not_configured"
        return "ready" if self.dataset_path.is_file() else "missing_dataset"

    def _configuration_error(self, request: RestrictedSourceQuery) -> str | None:
        if request.operation == "repurposing" and self.runtime_state == "not_configured":
            return "local_dataset_not_configured"
        if request.operation != "repurposing" and self.dataset_path is None:
            return "local_dataset_not_configured"
        return None

    def _configuration_fingerprint(self) -> tuple[Any, ...]:
        if self.dataset_path is None:
            return (None, self.dataset_version)
        try:
            return (
                str(self.dataset_path.resolve()),
                self.dataset_path.stat().st_mtime_ns,
                self.dataset_version,
            )
        except OSError:
            return (str(self.dataset_path), "missing", self.dataset_version)

    def _perform(self, request: RestrictedSourceQuery) -> _QueryOutcome:
        if request.operation == "repurposing":
            if self.runtime_state == "missing_dataset":
                base_path = getattr(self.local_adapter, "dataset_path", None)
                raise FileNotFoundError(str(base_path))
            return self._repurposing(request)
        return self._local_row_query(request)

    def _repurposing(self, request: RestrictedSourceQuery) -> _QueryOutcome:
        if not request.target_uniprot:
            raise _NotConfigured("drugcentral_interaction_tsv_requires_uniprot")
        lookup = ChemistryLookup(
            candidate_name=request.candidate_name,
            inchikey=request.inchikey,
            uniprot_id=request.target_uniprot,
            limit_per_source=min(request.limit, 10),
        )
        records, summary = self.local_adapter.lookup(lookup)
        if summary.status == "not_configured":
            raise _NotConfigured(summary.error_type or "local_dataset_not_configured")
        if summary.status == "error":
            raise RuntimeError(summary.error_type or "drugcentral_local_adapter_error")
        enriched = tuple(
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "restricted_operation": request.operation,
                        "local_adapter_reused": True,
                        "regulatory_status_checked": False,
                    }
                },
                deep=True,
            )
            for item in records
        )
        return _QueryOutcome(enriched, provider_cache_hit=summary.cache_hit)

    def _local_row_query(self, request: RestrictedSourceQuery) -> _QueryOutcome:
        if self.dataset_path is None:
            raise _NotConfigured("local_dataset_not_configured")
        if not self.dataset_path.is_file():
            raise FileNotFoundError(str(self.dataset_path))

        with self.dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError("DrugCentral TSV has no header")
            normalized_headers = {
                re.sub(r"[^a-z0-9]", "", name.casefold()) for name in reader.fieldnames
            }
            capability_aliases = (
                self._contraindication_aliases
                if request.operation == "contraindication"
                else self._contraindication_aliases + self._regulatory_aliases
            )
            capability_present = any(
                re.sub(r"[^a-z0-9]", "", name.casefold()) in normalized_headers
                for name in capability_aliases
            )
            if request.operation == "contraindication" and not capability_present:
                raise _NotConfigured(
                    "drugcentral_contraindication_fields_not_present_in_dataset"
                )

            records: list[EvidenceRef] = []
            for row_number, raw in enumerate(reader, start=2):
                row = _normalized_mapping(raw)
                if not self._matches_local_identity(request, row):
                    continue
                contraindication = _first(row, *self._contraindication_aliases)
                if request.operation == "contraindication" and not contraindication:
                    continue
                if (
                    request.operation == "contraindication"
                    and request.condition
                    and request.condition.casefold()
                    not in str(contraindication).casefold()
                ):
                    continue
                records.append(
                    self._row_evidence(
                        request,
                        row,
                        row_number=row_number,
                        regulatory_capability=capability_present,
                    )
                )
                if len(records) >= request.limit:
                    break
        return _QueryOutcome(tuple(records))

    @staticmethod
    def _matches_local_identity(
        request: RestrictedSourceQuery, row: Mapping[str, Any]
    ) -> bool:
        """Apply stable-field precedence and reject any comparable conflict.

        A TSV export need not contain every possible identity column.  Only
        fields supplied by the caller *and* present in the row are compared.
        DrugCentral ID has first precedence, followed by InChIKey.  If either
        comparable stable field conflicts, an exact name cannot rescue the row.
        Name is used only when no stable field can be compared.
        """

        stable_match = False
        if request.drug_id:
            value = _first(row, "struct_id", "drugcentral_id", "drug_id", "id")
            if value is not None:
                if _normalized_identity_key(value) != _normalized_identity_key(
                    request.drug_id
                ):
                    return False
                stable_match = True
        if request.inchikey:
            value = _first(row, "inchikey", "inchi_key")
            if value is not None:
                if _normalized_identity_key(value) != _normalized_identity_key(
                    request.inchikey
                ):
                    return False
                stable_match = True
        if stable_match:
            return True
        if not request.candidate_name:
            return False
        value = _first(row, "name", "drug_name", "inn", "preferred_name")
        if value is None:
            return False
        return _normalized_exact_name(value).casefold() == _normalized_exact_name(
            request.candidate_name
        ).casefold()

    def _row_evidence(
        self,
        request: RestrictedSourceQuery,
        row: Mapping[str, Any],
        *,
        row_number: int,
        regulatory_capability: bool,
    ) -> EvidenceRef:
        drug_id = _first(row, "struct_id", "drugcentral_id", "drug_id", "id")
        name = _first(row, "name", "drug_name", "inn", "preferred_name")
        external_id = str(drug_id or f"row-{row_number}")
        contraindication = _first(row, *self._contraindication_aliases)
        status = _first(
            row,
            "regulatory_status",
            "approval_status",
            "market_status",
            "status",
        )
        withdrawn_raw = _first(row, "withdrawn", "withdrawal_status")
        withdrawn = self._explicit_boolean(withdrawn_raw)
        title_kind = (
            "contraindication"
            if request.operation == "contraindication"
            else "identity/regulatory record"
        )
        title = f"DrugCentral {title_kind}: {name or external_id}"
        return EvidenceRef(
            evidence_id=_evidence_id(
                "drugcentral_restricted",
                f"{self.dataset_path.name if self.dataset_path else ''}|"
                f"{external_id}|{row_number}|{request.operation}",
            ),
            source=self.source,
            external_id=external_id,
            title=title,
            url="https://drugcentral.org/",
            supports=(
                "counter"
                if request.operation == "contraindication" or withdrawn is True
                else "context"
            ),
            excerpt_hash=_text_hash(contraindication or title),
            metadata={
                "live": False,
                "mock": False,
                "cache_hit": False,
                "provenance_class": "external_local_dataset",
                "retrieval_adapter": type(self).__name__,
                "official_download_url": self.documentation_url,
                "dataset_filename": self.dataset_path.name if self.dataset_path else None,
                "dataset_version": self.dataset_version,
                "dataset_row_number": row_number,
                "restricted_operation": request.operation,
                "drug_name": name,
                "drugcentral_id": str(drug_id) if drug_id is not None else None,
                "inchikey": _first(row, "inchikey", "inchi_key"),
                "regulatory_status": status,
                "withdrawn": withdrawn,
                "contraindication_present": bool(contraindication),
                "contraindication_hash": _text_hash(contraindication),
                "regulatory_fields_present": regulatory_capability,
                "absence_is_not_negative_evidence": True,
            },
        )

    @staticmethod
    def _explicit_boolean(value: Any) -> bool | None:
        if value is None:
            return None
        normalized = str(value).strip().casefold()
        if normalized in {"true", "yes", "y", "1", "withdrawn"}:
            return True
        if normalized in {"false", "no", "n", "0", "active", "marketed"}:
            return False
        return None


class _HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...


class DrugBankLicensedAdapter(_ConditionalReadOnlyAdapter):
    """Read-only DrugBank API adapter enabled only by a licensed API key."""

    source: RestrictedSource = "DrugBank"
    documentation_url = DRUGBANK_API_DOCUMENTATION
    required_configuration = (
        "DrugBank licence/subscription",
        "DRUGBANK_API_KEY (development or production key)",
    )
    limitations = (
        "API availability, fields, region coverage, quota, and reuse follow the licence.",
        "A missing contraindication record is not proof that a drug is safe.",
        "Repurposing bonds show database relationships, not clinical efficacy.",
    )
    supported_operations = frozenset(
        {"repurposing", "identity_regulatory", "contraindication"}
    )
    discovery_base_url = "https://api.drugbank.com/discovery/v1"
    regional_base_url = "https://api.drugbank.com/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: _HTTPClient | None = None,
        timeout_seconds: float = 8.0,
        dataset_version: str | None = None,
    ) -> None:
        super().__init__()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.api_key = str(api_key).strip() if api_key else None
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.dataset_version = dataset_version

    @property
    def runtime_state(self) -> Literal["ready", "not_configured"]:
        return "ready" if self.api_key else "not_configured"

    def _configuration_error(self, request: RestrictedSourceQuery) -> str | None:
        if not self.api_key:
            return "licensed_api_key_not_configured"
        if request.operation == "identity_regulatory" and not (
            request.drug_id or request.candidate_name
        ):
            return "documented_drugbank_id_or_name_required"
        return None

    def _configuration_fingerprint(self) -> tuple[Any, ...]:
        # A one-way prefix prevents results from crossing key/licence contexts
        # without retaining the credential itself.
        key_scope = sha256((self.api_key or "").encode("utf-8")).hexdigest()[:12]
        return (key_scope, self.dataset_version)

    def _perform(self, request: RestrictedSourceQuery) -> _QueryOutcome:
        if request.operation == "repurposing":
            return _QueryOutcome(tuple(self._query_bonds(request)))
        if request.operation == "identity_regulatory":
            return _QueryOutcome(tuple(self._query_identity(request)))
        return _QueryOutcome(tuple(self._query_contraindications(request)))

    def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        if not self.api_key:
            raise _NotConfigured("licensed_api_key_not_configured")
        client = self.client or get_scientific_http_client()
        response = client.get(
            url,
            params=dict(params or {}),
            headers={"Authorization": self.api_key, "Accept": "application/json"},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.json()

    def _find_drugs(self, request: RestrictedSourceQuery) -> list[Mapping[str, Any]]:
        if request.drug_id:
            if not re.fullmatch(r"DB\d{5,}", request.drug_id, flags=re.IGNORECASE):
                raise _NotConfigured("valid_drugbank_id_required")
            payload = self._get_json(
                f"{self.discovery_base_url}/drugs/{quote(request.drug_id.upper(), safe='')}"
            )
            return [payload] if isinstance(payload, Mapping) else []
        if not request.candidate_name:
            raise _NotConfigured("documented_drugbank_id_or_name_required")
        exact_name = _normalized_exact_name(request.candidate_name)
        escaped = exact_name.replace("\\", "\\\\").replace('"', '\\"')
        payload = self._get_json(
            f"{self.discovery_base_url}/drugs",
            params={"q": f'name:"{escaped}"', "per_page": request.limit},
        )
        rows = payload if isinstance(payload, list) else []
        exact: list[Mapping[str, Any]] = []
        wanted = exact_name.casefold()
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            names = [raw.get("name"), *(raw.get("synonyms") or [])]
            if any(
                _normalized_exact_name(name).casefold() == wanted
                for name in names
                if name
            ):
                exact.append(raw)
        return exact[: request.limit]

    def _query_identity(self, request: RestrictedSourceQuery) -> list[EvidenceRef]:
        records: list[EvidenceRef] = []
        for item in self._find_drugs(request)[: request.limit]:
            drug_id = item.get("drugbank_id")
            if not drug_id:
                continue
            name = item.get("name") or str(drug_id)
            groups = tuple(str(value) for value in (item.get("groups") or []))
            availability = item.get("availability_by_region") or []
            if not isinstance(availability, list):
                availability = []
            cancelled_regions = tuple(
                str(region.get("region"))
                for region in availability
                if isinstance(region, Mapping)
                and (
                    region.get("pre_market_cancelled") is True
                    or region.get("post_market_cancelled") is True
                )
                and region.get("region")
            )
            identifiers = item.get("identifiers") or {}
            pharmacology = item.get("pharmacology") or {}
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id("drugbank_identity", str(drug_id)),
                    source=self.source,
                    external_id=str(drug_id),
                    title=f"DrugBank identity/regulatory record: {name}",
                    url=f"https://go.drugbank.com/drugs/{quote(str(drug_id), safe='')}",
                    supports=(
                        "counter"
                        if any(group.casefold() == "withdrawn" for group in groups)
                        else "context"
                    ),
                    excerpt_hash=_text_hash(
                        pharmacology.get("indication_description")
                        if isinstance(pharmacology, Mapping)
                        else name
                    ),
                    metadata={
                        "live": True,
                        "mock": False,
                        "cache_hit": False,
                        "provenance_class": "external_live",
                        "retrieval_adapter": type(self).__name__,
                        "api_endpoint": f"{self.discovery_base_url}/drugs/<ID>",
                        "documentation_url": DRUGBANK_DISCOVERY_DOCUMENTATION,
                        "restricted_operation": request.operation,
                        "dataset_version": self.dataset_version,
                        "drug_name": name,
                        "drugbank_id": str(drug_id),
                        "inchikey": (
                            identifiers.get("inchikey")
                            if isinstance(identifiers, Mapping)
                            else None
                        ),
                        "regulatory_groups": groups,
                        "withdrawn": any(
                            group.casefold() == "withdrawn" for group in groups
                        ),
                        "cancelled_regions": cancelled_regions,
                        "availability_by_region": [
                            {
                                key: region.get(key)
                                for key in (
                                    "region",
                                    "max_phase",
                                    "marketed_prescription",
                                    "pre_market_cancelled",
                                    "post_market_cancelled",
                                )
                            }
                            for region in availability
                            if isinstance(region, Mapping)
                        ],
                        "licensed_content_text_retained": False,
                    },
                )
            )
        return records

    def _query_bonds(self, request: RestrictedSourceQuery) -> list[EvidenceRef]:
        if request.target_uniprot:
            target = request.target_uniprot.upper()
            url = (
                f"{self.discovery_base_url}/polypeptides/"
                f"{quote(target, safe='')}/bonds"
            )
            params: Mapping[str, Any] = {"per_page": request.limit}
            query_semantics = "exact_uniprot_bonds"
        else:
            assert request.target_name is not None
            target = request.target_name
            escaped = target.replace("\\", "\\\\").replace('"', '\\"')
            url = f"{self.discovery_base_url}/bonds/targets"
            params = {"q": f'name:"{escaped}"', "per_page": request.limit}
            query_semantics = "exact_target_name_search"
        payload = self._get_json(url, params=params)
        rows = payload if isinstance(payload, list) else []
        records: list[EvidenceRef] = []
        seen: set[str] = set()
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            bond_type = str(item.get("type") or "")
            if bond_type and bond_type.casefold() != "target":
                continue
            drug = item.get("drug") or {}
            bio_entity = item.get("bio_entity") or {}
            if not isinstance(drug, Mapping) or not drug.get("drugbank_id"):
                continue
            drug_id = str(drug["drugbank_id"])
            bio_id = (
                str(bio_entity.get("bio_entity_id") or "")
                if isinstance(bio_entity, Mapping)
                else ""
            )
            unique = f"{drug_id}|{bio_id}|{target}"
            if unique in seen:
                continue
            seen.add(unique)
            name = drug.get("name") or drug_id
            title = f"DrugBank target bond: {name} → {target}"
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id("drugbank_bond", unique),
                    source=self.source,
                    external_id=drug_id,
                    title=title,
                    url=f"https://go.drugbank.com/drugs/{quote(drug_id, safe='')}",
                    supports="context",
                    excerpt_hash=_text_hash(title),
                    metadata={
                        "live": True,
                        "mock": False,
                        "cache_hit": False,
                        "provenance_class": "external_live",
                        "retrieval_adapter": type(self).__name__,
                        "api_endpoint": url,
                        "documentation_url": DRUGBANK_DISCOVERY_DOCUMENTATION,
                        "restricted_operation": request.operation,
                        "query_semantics": query_semantics,
                        "target_query": target,
                        "drug_name": name,
                        "drugbank_id": drug_id,
                        "bio_entity_id": bio_id or None,
                        "bio_entity_name": (
                            bio_entity.get("name")
                            if isinstance(bio_entity, Mapping)
                            else None
                        ),
                        "known_action": item.get("known_action"),
                        "actions": item.get("actions") or [],
                        "regulatory_status_checked": False,
                        "clinical_efficacy_inferred": False,
                    },
                )
            )
            if len(records) >= request.limit:
                break
        return records

    def _query_contraindications(
        self, request: RestrictedSourceQuery
    ) -> list[EvidenceRef]:
        drugs = self._find_drugs(request)
        records: list[EvidenceRef] = []
        for drug in drugs:
            drug_id = drug.get("drugbank_id")
            if not drug_id:
                continue
            name = str(drug.get("name") or drug_id)
            endpoint = (
                f"{self.regional_base_url}/drugs/"
                f"{quote(str(drug_id), safe='')}/contraindications"
            )
            payload = self._get_json(
                endpoint,
                params={"per_page": min(request.limit, 50)},
            )
            rows = payload if isinstance(payload, list) else []
            for index, item in enumerate(rows):
                if not isinstance(item, Mapping):
                    continue
                conditions = self._contraindication_condition_names(item)
                if request.condition and not any(
                    request.condition.casefold() in value.casefold()
                    for value in conditions
                ):
                    continue
                regions = item.get("regions")
                hypersensitivity = bool(item.get("hypersensitivity"))
                label = ", ".join(conditions) if conditions else "label restriction"
                unique = f"{drug_id}|{index}|{label}|{regions}"
                records.append(
                    EvidenceRef(
                        evidence_id=_evidence_id("drugbank_contra", unique),
                        source=self.source,
                        external_id=str(drug_id),
                        title=f"DrugBank contraindication for {name}: {label}",
                        url=f"https://go.drugbank.com/drugs/{quote(str(drug_id), safe='')}",
                        supports="counter",
                        excerpt_hash=_text_hash(item),
                        metadata={
                            "live": True,
                            "mock": False,
                            "cache_hit": False,
                            "provenance_class": "external_live",
                            "retrieval_adapter": type(self).__name__,
                            "api_endpoint": endpoint,
                            "documentation_url": self.documentation_url,
                            "restricted_operation": request.operation,
                            "drug_name": name,
                            "drugbank_id": str(drug_id),
                            "patient_conditions": conditions,
                            "regions": regions,
                            "hypersensitivity": hypersensitivity,
                            "route": item.get("route") or [],
                            "dose_form": item.get("dose_form") or [],
                            "licensed_content_text_retained": False,
                            "absence_is_not_negative_evidence": True,
                        },
                    )
                )
                if len(records) >= request.limit:
                    return records
        return records

    @staticmethod
    def _contraindication_condition_names(item: Mapping[str, Any]) -> tuple[str, ...]:
        names: list[str] = []
        for raw in item.get("patient_conditions") or []:
            if isinstance(raw, Mapping) and raw.get("name"):
                names.append(str(raw["name"]))
        for raw in item.get("lab_values") or []:
            if isinstance(raw, Mapping) and raw.get("name"):
                names.append(str(raw["name"]))
        return tuple(names)


class BigQueryExecution(StrictModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    job_id: str | None = None
    total_bytes_processed: int | None = Field(default=None, ge=0)
    provider_cache_hit: bool = False


class BigQueryExecutor(Protocol):
    def execute(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        maximum_bytes_billed: int,
        location: str,
    ) -> BigQueryExecution: ...


class GoogleCloudBigQueryExecutor:
    """Small optional-dependency seam around ``google-cloud-bigquery``."""

    def __init__(self, *, project: str, client: Any | None = None) -> None:
        self.project = project.strip()
        if not self.project:
            raise ValueError("project is required")
        self._client = client

    def execute(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        maximum_bytes_billed: int,
        location: str,
    ) -> BigQueryExecution:
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # optional dependency remains explicit
            raise ModuleNotFoundError("google-cloud-bigquery is not installed") from exc

        client = self._client or bigquery.Client(project=self.project)
        job_config = bigquery.QueryJobConfig(
            use_legacy_sql=False,
            use_query_cache=True,
            maximum_bytes_billed=maximum_bytes_billed,
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "terms", "STRING", list(parameters["terms"])
                ),
                bigquery.ScalarQueryParameter(
                    "limit", "INT64", int(parameters["limit"])
                ),
            ],
        )
        job = client.query(sql, job_config=job_config, location=location)
        rows = [dict(row) for row in job.result()]
        return BigQueryExecution(
            rows=rows,
            job_id=getattr(job, "job_id", None),
            total_bytes_processed=getattr(job, "total_bytes_processed", None),
            provider_cache_hit=bool(getattr(job, "cache_hit", False)),
        )


class GooglePatentsBigQueryAdapter(_ConditionalReadOnlyAdapter):
    """Metered keyword screen over the Google Patents research table.

    The query checks title and extracted ``top_terms`` fields.  It returns
    potentially relevant prior-art context only; it does not inspect claims,
    establish priority, determine legal status, or provide a novelty opinion.
    """

    source: RestrictedSource = "Google Patents"
    documentation_url = GOOGLE_PATENTS_DATASET_DOCUMENTATION
    required_configuration = (
        "Google Cloud project used to run/bill BigQuery jobs",
        "BigQuery API and Application Default Credentials",
        "google-cloud-bigquery Python package (unless an executor is injected)",
    )
    limitations = (
        "Keyword matches in title/top_terms are not a legal novelty determination.",
        "The screen does not establish patent family priority or claim scope.",
        "Public datasets have no SLA and BigQuery query charges/quotas can apply.",
    )
    supported_operations = frozenset({"patent_novelty"})

    _sql = f"""
SELECT
  publication_number,
  title,
  url,
  country,
  publication_description,
  top_terms
FROM `{GOOGLE_PATENTS_TABLE}` AS publication
WHERE (
  SELECT LOGICAL_AND(
    STRPOS(LOWER(IFNULL(publication.title, '')), LOWER(term)) > 0
    OR EXISTS (
      SELECT 1
      FROM UNNEST(IFNULL(publication.top_terms, [])) AS top_term
      WHERE STRPOS(LOWER(top_term), LOWER(term)) > 0
    )
  )
  FROM UNNEST(@terms) AS term
)
ORDER BY publication_number DESC
LIMIT @limit
""".strip()

    def __init__(
        self,
        *,
        project: str | None = None,
        executor: BigQueryExecutor | None = None,
        maximum_bytes_billed: int = 20_000_000_000,
        location: str = "US",
        dataset_version: str | None = None,
    ) -> None:
        super().__init__()
        if maximum_bytes_billed <= 0:
            raise ValueError("maximum_bytes_billed must be positive")
        self.project = str(project).strip() if project else None
        self.executor = executor
        self.maximum_bytes_billed = maximum_bytes_billed
        self.location = location.strip() or "US"
        self.dataset_version = dataset_version

    @property
    def runtime_state(self) -> Literal["ready", "not_configured"]:
        return "ready" if self.project else "not_configured"

    def _configuration_error(self, request: RestrictedSourceQuery) -> str | None:
        return None if self.project else "google_cloud_project_not_configured"

    def _runtime_configuration_error(self, exc: Exception) -> str | None:
        if isinstance(exc, ModuleNotFoundError):
            return "google_cloud_bigquery_dependency_not_configured"
        if type(exc).__name__ == "DefaultCredentialsError":
            return "google_application_default_credentials_not_configured"
        return None

    def _configuration_fingerprint(self) -> tuple[Any, ...]:
        return (
            self.project,
            self.location,
            self.maximum_bytes_billed,
            self.dataset_version,
        )

    def _perform(self, request: RestrictedSourceQuery) -> _QueryOutcome:
        assert self.project is not None
        executor = self.executor or GoogleCloudBigQueryExecutor(project=self.project)
        execution = executor.execute(
            self._sql,
            parameters={"terms": request.novelty_terms, "limit": request.limit},
            maximum_bytes_billed=self.maximum_bytes_billed,
            location=self.location,
        )
        records: list[EvidenceRef] = []
        for row in execution.rows[: request.limit]:
            publication_number = row.get("publication_number")
            if not publication_number:
                continue
            title = str(row.get("title") or f"Patent {publication_number}")
            url = row.get("url") or (
                "https://patents.google.com/patent/"
                f"{quote(str(publication_number).replace('-', ''), safe='')}"
            )
            records.append(
                EvidenceRef(
                    evidence_id=_evidence_id(
                        "google_patents", str(publication_number)
                    ),
                    source=self.source,
                    external_id=str(publication_number),
                    title=title,
                    url=str(url),
                    supports="context",
                    excerpt_hash=_text_hash(title),
                    metadata={
                        "live": not execution.provider_cache_hit,
                        "mock": False,
                        "cache_hit": execution.provider_cache_hit,
                        "provenance_class": (
                            "external_cache"
                            if execution.provider_cache_hit
                            else "external_live"
                        ),
                        "origin_provenance_class": "external_live",
                        "retrieval_adapter": type(self).__name__,
                        "api_endpoint": "BigQuery Jobs API",
                        "documentation_url": self.documentation_url,
                        "public_dataset_table": GOOGLE_PATENTS_TABLE,
                        "billing_project": self.project,
                        "dataset_version": self.dataset_version,
                        "restricted_operation": request.operation,
                        "query_semantics": "all_terms_in_title_or_extracted_top_terms",
                        "query_terms": request.novelty_terms,
                        "country": row.get("country"),
                        "publication_description": row.get(
                            "publication_description"
                        ),
                        "top_terms": row.get("top_terms") or [],
                        "bigquery_job_id": execution.job_id,
                        "total_bytes_processed": execution.total_bytes_processed,
                        "maximum_bytes_billed": self.maximum_bytes_billed,
                        "legal_novelty_determination": False,
                        "potential_prior_art_context_only": True,
                    },
                )
            )
        return _QueryOutcome(
            tuple(records), provider_cache_hit=execution.provider_cache_hit
        )


__all__ = [
    "BIGQUERY_PUBLIC_DATA_DOCUMENTATION",
    "BigQueryExecution",
    "BigQueryExecutor",
    "DRUGBANK_API_DOCUMENTATION",
    "DRUGBANK_DISCOVERY_DOCUMENTATION",
    "DRUGCENTRAL_DOWNLOAD_DOCUMENTATION",
    "DrugBankLicensedAdapter",
    "DrugCentralRestrictedAdapter",
    "GOOGLE_PATENTS_DATASET_DOCUMENTATION",
    "GOOGLE_PATENTS_TABLE",
    "GoogleCloudBigQueryExecutor",
    "GooglePatentsBigQueryAdapter",
    "RestrictedSourceQuery",
    "RestrictedSourceResult",
    "RestrictedSourceSummary",
]
