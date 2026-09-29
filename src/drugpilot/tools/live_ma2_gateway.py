from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import Any, Callable, Iterable, Iterator, Mapping, NoReturn, Protocol, Sequence
import unicodedata
from urllib.parse import quote
import sys

import httpx

from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateRoute,
    EvidenceRef,
    TargetCandidate,
)
from drugpilot.tools.executable import (
    ExecutableRunner,
    ScientificDataUnavailable,
    ScientificToolError,
    ScientificToolUnavailable,
)
from drugpilot.tools.http import ScientificHttpClient, classify_transport_error
from drugpilot.tools.primary_abstract_pairs import PrimaryAbstractPair
from drugpilot.tools.primary_abstract_search import (
    CandidateSearchAliases,
    PrimaryAbstractGroup,
    PrimaryAbstractSearchOutcome,
    search_primary_abstract_pairs,
)
from drugpilot.tools.primary_abstract_paper_native_v5 import (
    PAPER_NATIVE_V5_GRAMMAR,
    PrimaryAbstractPair as PaperNativeV5Pair,
)
from drugpilot.tools.primary_abstract_paper_native_search_v5 import (
    CandidateSearchAliases as PaperNativeV5CandidateSearchAliases,
    PAPER_NATIVE_V5_POLICY_TOKEN,
    PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
    PrimaryAbstractGroup as PaperNativeV5Group,
    PrimaryAbstractSearchOutcome as PaperNativeV5SearchOutcome,
    PrimaryStudyAttestation as PaperNativeV5PrimaryStudyAttestation,
    search_paper_native_primary_abstract_v5,
)
from drugpilot.tools.primary_abstract_paper_native_search_v6 import (
    PAPER_NATIVE_V6_POLICY_TOKEN,
)
from drugpilot.tools.protocols import CandidateSeed
from drugpilot.tools.activity_comparability import CHEMBL_ASSAY_CONTEXT_SCHEMA
from drugpilot.tools.worker_runtime import (
    ISOLATED_PYTHON_FLAGS,
    isolated_worker_environment,
    validate_isolated_worker_environment,
    worker_environment_snapshot,
)


_CHEMBL_STABLE_ID = re.compile(r"^CHEMBL[1-9][0-9]*$")
_BRICS_WORKER_PATH = Path(__file__).resolve().with_name("brics_worker.py")


class HTTPClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...


class _UnconfiguredPrimarySearchClient:
    """A distinct, zero-wire fail-closed seam for injected gateway tests."""

    max_attempts = 1

    def get(self, url: str, **kwargs: Any) -> Any:
        raise ScientificDataUnavailable(
            "A dedicated primary_search_client was not injected"
        )


class DockingBackend(Protocol):
    source_version: str
    last_result: Any

    def dock(self, *, smiles: str, target: TargetCandidate) -> float: ...


class ADMETBackend(Protocol):
    source_version: str
    last_result: Any

    def predict(self, *, smiles: str) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class HTTPCallProvenance:
    source: str
    endpoint: str
    parameter_keys: tuple[str, ...]
    parameter_hash: str
    timeout_seconds: float
    status_code: int
    response_sha256: str
    started_at: str
    finished_at: str
    runtime_seconds: float
    attempt_count: int = 1
    max_attempts: int = 1
    exhausted: bool = False
    cache_hit: bool = False
    operation: str | None = None
    operation_budget_seconds: float | None = None
    attempts: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "endpoint": self.endpoint,
            "parameter_keys": list(self.parameter_keys),
            "parameter_hash": self.parameter_hash,
            "timeout_seconds": self.timeout_seconds,
            "status_code": self.status_code,
            "response_sha256": self.response_sha256,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_seconds": self.runtime_seconds,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "exhausted": self.exhausted,
            "cache_hit": self.cache_hit,
            "operation": self.operation,
            "operation_budget_seconds": self.operation_budget_seconds,
            "attempts": [dict(item) for item in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class _MeasuredLigand:
    seed: CandidateSeed
    chembl_id: str
    pchembl_value: float | None
    max_phase: int


class NoMeasuredLigandCandidates(ScientificDataUnavailable):
    """A successful authoritative query conclusively found no usable pool.

    Network, timeout, schema and local-tool errors deliberately do not use
    this type.  MA2 can therefore distinguish a completed negative search
    from an unresolved scientific-tool failure without guessing candidates.
    """

    conclusive = True

    def __init__(
        self,
        *,
        target_symbol: str,
        target_uniprot_id: str | None,
        direction: str,
        eligible_count: int,
        minimum_required: int,
        reason_code: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.target_symbol = target_symbol
        self.target_uniprot_id = target_uniprot_id
        self.direction = direction
        self.eligible_count = eligible_count
        self.minimum_required = minimum_required
        self.reason_code = reason_code
        self.provenance = dict(provenance or {})
        super().__init__(
            f"{reason_code}: {target_symbol} ({target_uniprot_id or 'no UniProt'}), "
            f"direction={direction}, eligible={eligible_count}, "
            f"required={minimum_required}; no candidate was substituted"
        )


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_environment(
    frozen: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the complete allowlisted environment for local workers."""

    return (
        isolated_worker_environment()
        if frozen is None
        else validate_isolated_worker_environment(frozen)
    )


def _stable_id(prefix: str, value: str) -> str:
    digest = sha256(f"{prefix}|{value}".encode("utf-8")).hexdigest()[:16]
    return f"ev_{prefix}_{digest}"


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _normalize_activity(row: Mapping[str, Any]) -> ActivityMeasurement | None:
    """Admit only fully deposited ChEMBL standard fields.

    The typed measurement is an exact projection of the deposited record.  A
    missing relation is not equality, and alternate unit spellings/case are
    not silently canonicalised because doing so would sever the measurement
    from the raw evidence that MA3 re-checks.
    """

    required_fields = (
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
    )
    if any(field not in row or row.get(field) is None for field in required_fields):
        return None

    endpoint = row.get("standard_type")
    if not isinstance(endpoint, str) or endpoint not in {"IC50", "Ki", "Kd", "EC50"}:
        return None
    relation = row.get("standard_relation")
    if not isinstance(relation, str):
        return None
    if relation not in {"=", "<", ">", "<=", ">="}:
        return None
    raw_value = row.get("standard_value")
    if isinstance(raw_value, bool):
        return None
    try:
        deposited_value = Decimal(str(raw_value))
    except (InvalidOperation, ValueError):
        return None
    if not deposited_value.is_finite() or deposited_value <= 0:
        return None
    unit = row.get("standard_units")
    if not isinstance(unit, str) or unit not in {"pM", "nM", "uM", "mM"}:
        return None
    activity_id = str(row.get("activity_id") or "unknown")
    return ActivityMeasurement(
        endpoint=endpoint,  # type: ignore[arg-type]
        relation=relation,  # type: ignore[arg-type]
        value=float(deposited_value),
        unit=unit,  # type: ignore[arg-type]
        assay_type=str(row.get("bao_label") or row.get("assay_type") or "ChEMBL assay"),
        source_evidence_id=_stable_id("chembl_activity", activity_id),
    )


def _assay_context(assay_type: str) -> str:
    """Classify assay format without inferring a stronger biological claim."""

    normalized = " ".join(assay_type.casefold().replace("_", " ").split())
    if any(term in normalized for term in ("cell", "tissue", "organism", "phenotyp")):
        return "cellular_or_system_context"
    if any(
        term in normalized
        for term in (
            "single protein",
            "biochemical",
            "protein binding",
            "purified protein",
            "enzyme assay",
        )
    ):
        return "direct_single_protein_context"
    return "assay_context_unspecified"


def _assay_supports_direct_target_engagement(assay_type: str) -> bool:
    return _assay_context(assay_type) == "direct_single_protein_context"


def _assay_engagement_assessment(
    *,
    assay_type: str,
    assay_context: Mapping[str, Any] | None,
    assay_chembl_id: str,
    target_chembl_id: str,
    target_symbol: str,
) -> dict[str, Any]:
    """Classify direct engagement from exact, self-consistent assay evidence.

    ChEMBL occasionally exposes a generic BAO label for an otherwise explicit
    recombinant biochemical assay.  The generic label must not erase that
    evidence, but free-text alone is too weak to promote a direct-engagement
    claim.  The fallback therefore requires an exact bound assay record, a
    direct high-confidence human target assignment, no cellular fields, and an
    untruncated/hash-valid description that explicitly binds the target to a
    recombinant/purified inhibition assay with ATP and substrate.
    """

    if _assay_supports_direct_target_engagement(assay_type):
        return {
            "schema_version": "assay_engagement_assessment_v1",
            "assay_context": "direct_single_protein_context",
            "direct_single_protein_engagement": True,
            "basis": "typed_assay_label",
            "reason_codes": [],
            "description_similarity_used": False,
            "missing_values_imputed": False,
        }
    if _assay_context(assay_type) == "cellular_or_system_context":
        return {
            "schema_version": "assay_engagement_assessment_v1",
            "assay_context": "cellular_or_system_context",
            "direct_single_protein_engagement": False,
            "basis": "typed_cellular_assay_label",
            "reason_codes": ["CELLULAR_ASSAY_NOT_DIRECT_SINGLE_PROTEIN"],
            "description_similarity_used": False,
            "missing_values_imputed": False,
        }

    reasons: list[str] = []
    record = (
        assay_context.get("record")
        if isinstance(assay_context, Mapping)
        else None
    )
    if (
        not isinstance(assay_context, Mapping)
        or assay_context.get("schema_version") != CHEMBL_ASSAY_CONTEXT_SCHEMA
        or assay_context.get("retrieval_status") != "completed"
        or assay_context.get("binding_status") != "exact"
        or assay_context.get("assay_chembl_id") != assay_chembl_id
        or assay_context.get("missing_values_imputed") is not False
        or not isinstance(record, Mapping)
    ):
        reasons.append("EXACT_ASSAY_CONTEXT_UNAVAILABLE")
        record = {}
    if (
        record.get("assay_chembl_id") != assay_chembl_id
        or record.get("target_chembl_id") != target_chembl_id
    ):
        reasons.append("EXACT_ASSAY_TARGET_BINDING_MISMATCH")
    if (
        record.get("assay_type") != "B"
        or record.get("assay_organism") != "Homo sapiens"
        or record.get("relationship_type") != "D"
        or record.get("confidence_score") != 9
    ):
        reasons.append("DIRECT_HUMAN_TARGET_ASSIGNMENT_UNATTESTED")
    if any(
        record.get(key) not in (None, "")
        for key in (
            "assay_cell_type",
            "assay_tissue",
            "assay_strain",
            "cell_chembl_id",
            "tissue_chembl_id",
        )
    ):
        reasons.append("CELLULAR_OR_TISSUE_CONTEXT_PRESENT")

    description = record.get("description")
    excerpt = (
        description.get("excerpt")
        if isinstance(description, Mapping)
        else None
    )
    normalized = _normalize_document_text(excerpt)
    description_valid = bool(
        isinstance(description, Mapping)
        and isinstance(excerpt, str)
        and excerpt == normalized
        and description.get("excerpt_truncated") is False
        and description.get("content_length") == len(normalized)
        and description.get("content_sha256")
        == sha256(normalized.encode("utf-8")).hexdigest()
    )
    if not description_valid:
        reasons.append("EXACT_ASSAY_DESCRIPTION_UNATTESTED")
    else:
        target_present = bool(
            target_symbol.strip()
            and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(target_symbol.strip())}(?![A-Za-z0-9])",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        normalized_lower = normalized.casefold()
        explicit_direct_terms = bool(
            target_present
            and re.search(r"(?<!\w)(?:recombinant|purified)(?!\w)", normalized_lower)
            and re.search(r"(?<!\w)(?:inhibit\w*|kinase|enzyme)(?!\w)", normalized_lower)
            and re.search(r"(?<!\w)atp(?!\w)", normalized_lower)
            and re.search(r"(?<!\w)substrate(?!\w)", normalized_lower)
        )
        if not explicit_direct_terms:
            reasons.append("EXPLICIT_RECOMBINANT_ASSAY_BINDING_UNATTESTED")

    direct = not reasons
    return {
        "schema_version": "assay_engagement_assessment_v1",
        "assay_context": (
            "direct_single_protein_context"
            if direct
            else "assay_context_unspecified"
        ),
        "direct_single_protein_engagement": direct,
        "basis": (
            "exact_bound_recombinant_biochemical_assay_record"
            if direct
            else "insufficient_exact_direct_assay_evidence"
        ),
        "reason_codes": sorted(set(reasons)),
        "description_similarity_used": False,
        "missing_values_imputed": False,
    }


def _candidate_mechanism_claim(
    *,
    target_symbol: str,
    measurement: ActivityMeasurement,
    direction_rule: str,
    mechanism_identity: Mapping[str, Any] | None,
    assay_context_label: str | None = None,
) -> str:
    """Derive the exact promoted claim without strengthening assay context."""

    context = assay_context_label or _assay_context(measurement.assay_type)
    if mechanism_identity is not None:
        if context == "direct_single_protein_context":
            return (
                "ChEMBL-curated inhibitor mechanism and measured direct "
                f"IC50 activity against {target_symbol}"
            )
        if context == "cellular_or_system_context":
            return (
                "ChEMBL-curated inhibitor mechanism with measured cellular "
                f"IC50 activity for {target_symbol}; direct single-protein "
                "engagement not established by the selected activity row"
            )
        return (
            "ChEMBL-curated inhibitor mechanism with measured IC50 activity "
            f"for {target_symbol}; assay context unspecified; direct "
            "engagement not established"
        )
    if direction_rule == "measured_ic50_endpoint":
        if context == "direct_single_protein_context":
            return f"Measured inhibitory potency against {target_symbol}"
        if context == "cellular_or_system_context":
            return (
                f"Measured cellular inhibitory potency in a {target_symbol}-linked "
                "assay; direct single-protein engagement not established"
            )
        return (
            f"Measured inhibitory potency for {target_symbol}; assay context "
            "unspecified; direct engagement not established"
        )
    return (
        f"Measured {measurement.endpoint} activity for {target_symbol}; "
        "agonist/antagonist direction unresolved"
    )


def _measurement_direction_rule(
    measurement: ActivityMeasurement,
    direction: str,
) -> str | None:
    if direction == "modulate":
        return "measured_exact_target_activity"
    if direction == "inhibit" and measurement.endpoint == "IC50":
        return "measured_ic50_endpoint"
    # Ki/Kd establish binding, not intervention direction. EC50 establishes
    # effect potency but not agonism versus antagonism without an explicit
    # action/mechanism record, which this ChEMBL activity query does not have.
    return None


_PRIMARY_EVIDENCE_SCHEMA_VERSION = "candidate_primary_evidence_v2"
_PRIMARY_EXCERPT_MAX_CHARS = 360
_PRIMARY_ACTIVITY_DOCUMENT_TOP_K = 3
_MAX_ACTIVITY_ROWS_PER_OPERATION = 80
_MAX_MECHANISM_ROWS_PER_OPERATION = 80
_MAX_REPURPOSING_CANDIDATE_IDENTITIES = 12
_MAX_GENERIC_REPURPOSING_POOL_IDENTITIES = 100
_FIXED_REPURPOSING_POOL_POLICIES = frozenset({
    "dynamic_pubmed_primary_abstract_paired_v4_required",
    PAPER_NATIVE_V5_POLICY_TOKEN, PAPER_NATIVE_V6_POLICY_TOKEN,
})
_MAX_ACTIVITY_ROWS_PER_CANDIDATE = 20
_MAX_EXACT_CANDIDATE_ALIASES = 32
_MAX_PRIMARY_ALIAS_POOL_CANDIDATES = 100
_MAX_PRIMARY_COMPETITOR_ALIASES = (
    _MAX_PRIMARY_ALIAS_POOL_CANDIDATES * _MAX_EXACT_CANDIDATE_ALIASES
)
_PRIMARY_DOCUMENT_BATCH_SIZE = 80
_PRIMARY_PUBMED_BATCH_MAX_IDS = (
    _MAX_PRIMARY_ALIAS_POOL_CANDIDATES * _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
)
_ASSAY_CONTEXT_TEXT_MAX_CHARS = 1_200
_ASSAY_CONTEXT_BATCH_SIZE = 80
_PRIMARY_RELEVANCE_POLICY = (
    "exact_candidate_target_endpoint_value_same_clause_v3"
)
_PRIMARY_PAIRED_V4_SCHEMA = "candidate_primary_evidence_paired_v4"
_PRIMARY_PAIRED_V4_POLICY = (
    "same_sentence_primary_pairs_exact_chembl_activity_and_assay_v4"
)
_PRIMARY_PAIRED_V4_MAX_CANDIDATES = 12
_PRIMARY_PAIRED_V4_MAX_DOCUMENTS = 8
_PRIMARY_PAIRED_V4_MAX_ALIASES = 16
_PRIMARY_PAPER_NATIVE_V5_SCHEMA = "candidate_primary_evidence_paper_native_v5"
_PRIMARY_PAPER_NATIVE_V5_SOURCE_CONTRACT = "primary_abstract_paper_native_v5"
_PRIMARY_PAPER_NATIVE_V5_MAX_CANDIDATES = 12
_PRIMARY_PAPER_NATIVE_V5_MAX_DOCUMENTS = 8
_PRIMARY_PAPER_NATIVE_V5_MAX_ALIASES = 16
_PRIMARY_PAPER_NATIVE_V5_TRUSTED_SYNONYM_TYPES = frozenset(
    {
        "BAN",
        "DCF",
        "INN",
        "JAN",
        "RESEARCH_CODE",
        "TRADE_NAME",
        "USAN",
    }
)
_COMPLETE_ACTIVITY_PAGINATION_POLICIES = frozenset(
    {PAPER_NATIVE_V5_POLICY_TOKEN, PAPER_NATIVE_V6_POLICY_TOKEN}
)
_CANDIDATE_ALIAS_SOURCE = "chembl_pref_name_and_molecule_synonyms_exact_v1"
_PRIMARY_ALIAS_INVENTORY_SCHEMA = "candidate_pool_alias_inventory_v1"
_PUBMED_TYPE_SCHEMA = "pubmed_esummary_publication_type_v1"
_PUBMED_TYPE_POLICY = "journal_article_without_excluded_publication_type_v1"
_PUBMED_REQUIRED_TYPE = "journal article"
_PUBMED_EXCLUDED_TYPES = frozenset(
    {
        "review",
        "systematic review",
        "meta analysis",
        "guideline",
        "practice guideline",
        "editorial",
        "letter",
        "comment",
        "news",
        "interview",
        "biography",
        "historical article",
        "patient education handout",
        "consensus development conference",
        "consensus development conference nih",
        "retracted publication",
        "published erratum",
        "expression of concern",
        "retraction notice",
        "retraction of publication",
        "protocol",
        "clinical trial protocol",
    }
)
_ENDPOINT_SPECIFIC_TERMS: dict[str, frozenset[str]] = {
    "ic50": frozenset(
        {
            "ic50",
            "ic 50",
            "half maximal inhibitory concentration",
            "50 percent inhibitory concentration",
        }
    ),
    "ki": frozenset({"ki", "inhibition constant", "inhibitory constant"}),
    "kd": frozenset({"kd", "dissociation constant"}),
    "ec50": frozenset(
        {
            "ec50",
            "ec 50",
            "half maximal effective concentration",
            "50 percent effective concentration",
        }
    ),
}


def _normalized_pubmed_types(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value or "").split())
        key = _normalize_identity_phrase(item)
        if not item or not key or key in seen:
            continue
        normalized.append(item)
        seen.add(key)
    return normalized


def _pubmed_type_decision(publication_types: Sequence[str]) -> tuple[bool, list[str]]:
    normalized = {
        _normalize_identity_phrase(value)
        for value in publication_types
        if _normalize_identity_phrase(value)
    }
    excluded = sorted(normalized.intersection(_PUBMED_EXCLUDED_TYPES))
    return _PUBMED_REQUIRED_TYPE in normalized and not excluded, excluded


def _normalize_document_text(value: Any) -> str:
    """Normalize ChEMBL document text without retaining markup or full text."""

    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return " ".join(text.split())


def _bounded_assay_text(value: Any) -> dict[str, Any] | None:
    """Retain a bounded exact-source assay passage with a full-content hash."""

    normalized = _normalize_document_text(value)
    if not normalized:
        return None
    excerpt = normalized[:_ASSAY_CONTEXT_TEXT_MAX_CHARS]
    return {
        "excerpt": excerpt,
        "excerpt_truncated": len(excerpt) < len(normalized),
        "content_length": len(normalized),
        "content_sha256": sha256(normalized.encode("utf-8")).hexdigest(),
    }


def _exact_assay_parameters(value: Any) -> list[dict[str, Any]]:
    """Copy deposited parameter fields without normalization or inference."""

    if not isinstance(value, list):
        return []
    allowed = (
        "type",
        "standard_type",
        "relation",
        "standard_relation",
        "value",
        "standard_value",
        "units",
        "standard_units",
        "text_value",
        "standard_text_value",
        "comments",
    )
    return [
        {key: item.get(key) for key in allowed if key in item}
        for item in value[:100]
        if isinstance(item, Mapping)
    ]


def _assay_condition_coverage(
    *,
    variant_sequence: Any,
    parameters: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Report only explicit deposited condition records; absence stays unknown."""

    exact_types = {
        re.sub(
            r"[\s_-]+",
            "_",
            str(
                item.get("standard_type") or item.get("type") or ""
            ).strip().casefold(),
        )
        for item in parameters
    }
    return {
        "construct": "unknown_no_dedicated_field",
        "mutation_state": (
            "explicit_variant_sequence"
            if str(variant_sequence or "").strip()
            else "unknown_not_assumed_wild_type"
        ),
        "atp": "explicit_parameter" if "atp" in exact_types else "unknown",
        "substrate": (
            "explicit_parameter" if "substrate" in exact_types else "unknown"
        ),
        "detection_technology": (
            "explicit_parameter"
            if "detection_technology" in exact_types
            else "unknown"
        ),
    }


def _normalize_identity_phrase(value: Any) -> str:
    return " ".join(
        re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE)
    )


def _matched_exact_phrases(text: str, phrases: Sequence[str]) -> list[str]:
    """Match complete normalized identity phrases, never component tokens."""

    matched: list[str] = []
    normalized_text = _normalize_identity_phrase(text)
    normalized_phrases = {
        _normalize_identity_phrase(phrase)
        for phrase in phrases
        if _normalize_identity_phrase(phrase)
    }
    for phrase in sorted(normalized_phrases):
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_text):
            matched.append(phrase)
    return matched


def _chembl_exact_aliases(molecule: Mapping[str, Any]) -> list[str]:
    """Return a bounded, non-fuzzy alias list from exact ChEMBL fields only."""

    values: list[Any] = [molecule.get("pref_name")]
    raw_synonyms = molecule.get("molecule_synonyms")
    if isinstance(raw_synonyms, list):
        for item in raw_synonyms[: _MAX_EXACT_CANDIDATE_ALIASES * 2]:
            if isinstance(item, Mapping):
                values.append(item.get("molecule_synonym"))
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = _normalize_document_text(value)
        normalized = _normalize_identity_phrase(alias)
        if not normalized or normalized in seen:
            continue
        aliases.append(alias)
        seen.add(normalized)
        if len(aliases) >= _MAX_EXACT_CANDIDATE_ALIASES:
            break
    return aliases


def _candidate_pool_alias_inventory(
    aliases_by_id: Mapping[str, Sequence[str]],
    *,
    expected_molecule_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Persist the complete bounded pool used to detect value attribution.

    The activity query is already capped at one hundred candidate molecules
    and each ChEMBL alias list is capped above.  If either bound is ever
    exceeded, the inventory is marked truncated and primary verification
    fails closed instead of examining an incomplete competitor set.
    """

    all_ids = sorted(
        {
            str(value)
            for value in (
                expected_molecule_ids
                if expected_molecule_ids is not None
                else aliases_by_id
            )
            if str(value)
        }
    )
    retained_ids = all_ids[:_MAX_PRIMARY_ALIAS_POOL_CANDIDATES]
    truncated = len(all_ids) > len(retained_ids)
    missing_molecule_ids = sorted(
        molecule_id
        for molecule_id in retained_ids
        if molecule_id not in aliases_by_id
    )
    aliasless_molecule_ids = sorted(
        molecule_id
        for molecule_id in retained_ids
        if molecule_id in aliases_by_id
        and not aliases_by_id.get(molecule_id)
    )
    if missing_molecule_ids or aliasless_molecule_ids:
        truncated = True
    records: list[dict[str, Any]] = []
    total_alias_count = 0
    for molecule_id in retained_ids:
        raw_aliases = list(aliases_by_id.get(molecule_id, ()))
        if len(raw_aliases) > _MAX_EXACT_CANDIDATE_ALIASES:
            truncated = True
        aliases = raw_aliases[:_MAX_EXACT_CANDIDATE_ALIASES]
        total_alias_count += len(aliases)
        records.append(
            {
                "molecule_chembl_id": molecule_id,
                "aliases": aliases,
            }
        )
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": _PRIMARY_ALIAS_INVENTORY_SCHEMA,
        "pool_candidate_cap": _MAX_PRIMARY_ALIAS_POOL_CANDIDATES,
        "per_candidate_alias_cap": _MAX_EXACT_CANDIDATE_ALIASES,
        "competitor_alias_cap": _MAX_PRIMARY_COMPETITOR_ALIASES,
        "candidate_count": len(all_ids),
        "retained_candidate_count": len(records),
        "alias_count": total_alias_count,
        "source_candidate_ids": all_ids,
        "missing_molecule_ids": missing_molecule_ids,
        "aliasless_molecule_ids": aliasless_molecule_ids,
        "complete": not truncated,
        "truncated": truncated,
        "records": records,
        "inventory_sha256": sha256(canonical).hexdigest(),
        "fixture": False,
    }


def _competitor_alias_provenance(
    inventory: Mapping[str, Any],
    *,
    current_molecule_chembl_id: str,
) -> dict[str, Any]:
    records = inventory.get("records")
    alias_owners: dict[str, dict[str, Any]] = {}
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            molecule_id = str(record.get("molecule_chembl_id") or "")
            if not molecule_id or molecule_id == current_molecule_chembl_id:
                continue
            aliases = record.get("aliases")
            if not isinstance(aliases, list):
                continue
            for alias in aliases:
                raw_alias = str(alias)
                normalized = _normalize_identity_phrase(raw_alias)
                if not normalized:
                    continue
                owner = alias_owners.setdefault(
                    normalized,
                    {"alias": raw_alias, "molecule_chembl_ids": []},
                )
                owner["molecule_chembl_ids"].append(molecule_id)
    owner_rows = [
        {
            "alias": item["alias"],
            "molecule_chembl_ids": sorted(set(item["molecule_chembl_ids"])),
        }
        for _normalized, item in sorted(alias_owners.items())
    ]
    competitor_aliases = [item["alias"] for item in owner_rows]
    truncated = bool(inventory.get("truncated")) or (
        len(competitor_aliases) > _MAX_PRIMARY_COMPETITOR_ALIASES
    )
    return {
        "policy": "all_bounded_pool_candidate_aliases_except_current_v1",
        "current_molecule_chembl_id": current_molecule_chembl_id,
        "competitor_exact_aliases": competitor_aliases[
            :_MAX_PRIMARY_COMPETITOR_ALIASES
        ],
        "alias_owners": owner_rows[:_MAX_PRIMARY_COMPETITOR_ALIASES],
        "inventory": deepcopy(dict(inventory)),
        "inventory_sha256": inventory.get("inventory_sha256"),
        "truncated": truncated,
        "fixture": False,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _paper_native_v6_delegate_seed_source_projection(
    *,
    fixed_candidate_ids: Sequence[str],
    seeds: Sequence[CandidateSeed],
) -> dict[str, Any]:
    """Bind every fixed-pool identity to its exact delegate seed source row."""

    fixed_ids = list(fixed_candidate_ids)
    seed_by_id = {
        str(seed.metadata.get("chembl_id") or ""): seed
        for seed in seeds
        if str(seed.metadata.get("chembl_id") or "")
    }
    rows: list[dict[str, Any]] = []
    returned_ids: list[str] = []
    for pool_index, candidate_id in enumerate(fixed_ids):
        seed = seed_by_id.get(candidate_id)
        if seed is None:
            row = {
                "pool_index": pool_index,
                "fixed_pool_rank": pool_index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": False,
                "activity_count": 0,
                "evidence_count": 0,
                "activity_id": None,
                "document_chembl_id": None,
                "pmid": None,
                "primary_verification_status": None,
                "primary_reason_code": None,
                "primary_attestation_sha256": None,
                "fixture": False,
            }
        else:
            returned_ids.append(candidate_id)
            evidence = seed.evidence[0] if len(seed.evidence) == 1 else None
            primary = (
                evidence.metadata.get("primary_source")
                if evidence is not None
                and isinstance(evidence.metadata, Mapping)
                else None
            )
            row = {
                "pool_index": pool_index,
                "fixed_pool_rank": pool_index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": True,
                "activity_count": len(seed.activities),
                "evidence_count": len(seed.evidence),
                "activity_id": (
                    str(evidence.external_id)
                    if evidence is not None
                    else None
                ),
                "document_chembl_id": (
                    primary.get("document_chembl_id")
                    if isinstance(primary, Mapping)
                    else None
                ),
                "pmid": (
                    primary.get("pmid")
                    if isinstance(primary, Mapping)
                    else None
                ),
                "primary_verification_status": (
                    primary.get("verification_status")
                    if isinstance(primary, Mapping)
                    else None
                ),
                "primary_reason_code": (
                    primary.get("reason_code")
                    if isinstance(primary, Mapping)
                    else None
                ),
                "primary_attestation_sha256": (
                    _canonical_sha256(dict(primary))
                    if isinstance(primary, Mapping)
                    else None
                ),
                "fixture": False,
            }
        row["row_sha256"] = _canonical_sha256(row)
        rows.append(row)
    projection = {
        "schema_version": (
            "paper_native_v6_delegate_seed_source_binding_projection/v1"
        ),
        "fixed_pool_candidate_ids": fixed_ids,
        "fixed_pool_candidate_ids_sha256": _canonical_sha256(fixed_ids),
        "delegate_return_candidate_ids": returned_ids,
        "delegate_return_candidate_ids_sha256": _canonical_sha256(
            returned_ids
        ),
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
        "complete": True,
        "fixture": False,
    }
    projection["attestation_sha256"] = _canonical_sha256(projection)
    return projection


def _source_parameter_sha256(params: Mapping[str, Any]) -> str:
    """Mirror ``_get_json``'s deposited request-parameter digest."""

    encoded = json.dumps(
        dict(params),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _fixed_pool_molecule_projection(
    *,
    fixed_pool_ids: Sequence[str],
    molecules_by_id: Mapping[str, Mapping[str, Any]],
    aliases_by_id: Mapping[str, Sequence[str]],
    molecule_query: HTTPCallProvenance | Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the bounded V6 pool to the exact resolved ChEMBL molecule rows.

    The full HTTP response remains body-free and is identified by the call's
    response digest.  For every fixed-pool identity we additionally retain a
    bounded projection of the source fields that are promoted into candidate
    cards.  This lets MA3 recompute the SMILES/name/phase/alias copies instead
    of trusting a mutable candidate metadata row.
    """

    query = (
        molecule_query.as_dict()
        if isinstance(molecule_query, HTTPCallProvenance)
        else deepcopy(dict(molecule_query))
    )
    synonym_row_limit = _MAX_EXACT_CANDIDATE_ALIASES * 2
    rows: list[dict[str, Any]] = []
    for pool_index, molecule_id in enumerate(fixed_pool_ids):
        molecule = molecules_by_id.get(molecule_id)
        molecule = molecule if isinstance(molecule, Mapping) else {}
        structures = molecule.get("molecule_structures")
        structures = structures if isinstance(structures, Mapping) else {}
        raw_synonyms = molecule.get("molecule_synonyms")
        source_key_present = "molecule_synonyms" in molecule
        synonym_rows: list[dict[str, Any]] = []
        synonym_rows_valid = isinstance(raw_synonyms, list)
        if isinstance(raw_synonyms, list):
            for source_index, raw_synonym in enumerate(
                raw_synonyms[:synonym_row_limit]
            ):
                is_mapping = isinstance(raw_synonym, Mapping)
                raw_value = (
                    raw_synonym.get("molecule_synonym")
                    if is_mapping
                    else None
                )
                raw_type = raw_synonym.get("syn_type") if is_mapping else None
                fields_valid = bool(
                    is_mapping
                    and (raw_value is None or isinstance(raw_value, str))
                    and (raw_type is None or isinstance(raw_type, str))
                    and (
                        not isinstance(raw_value, str)
                        or (
                            raw_value == raw_value.strip()
                            and "\r" not in raw_value
                            and "\n" not in raw_value
                            and len(raw_value) <= 300
                        )
                    )
                    and (
                        not isinstance(raw_type, str)
                        or (
                            raw_type == raw_type.strip()
                            and "\r" not in raw_type
                            and "\n" not in raw_type
                            and len(raw_type) <= 80
                        )
                    )
                )
                synonym_rows_valid = synonym_rows_valid and fields_valid
                synonym_rows.append(
                    {
                        "source_index": source_index,
                        "row_type": "mapping" if is_mapping else type(raw_synonym).__name__,
                        "molecule_synonym": (
                            raw_value if isinstance(raw_value, str) else None
                        ),
                        "syn_type": raw_type if isinstance(raw_type, str) else None,
                        "fields_valid": fields_valid,
                        "source_row_sha256": _canonical_sha256(raw_synonym),
                    }
                )
        elif raw_synonyms is None:
            # Missing/null synonym metadata is represented exactly.  It is not
            # repaired or treated as evidence, but a valid preferred name can
            # still define the bounded exact-alias set.
            synonym_rows_valid = True

        pref_name = molecule.get("pref_name")
        pref_name_valid = bool(
            pref_name is None
            or (
                isinstance(pref_name, str)
                and pref_name == pref_name.strip()
                and "\r" not in pref_name
                and "\n" not in pref_name
                and len(pref_name) <= 300
            )
        )
        raw_phase = molecule.get("max_phase")
        phase_value = _safe_float(raw_phase)
        max_phase = (
            int(phase_value)
            if (
                not isinstance(raw_phase, bool)
                and phase_value is not None
                and 1 <= phase_value <= 4
                and float(phase_value).is_integer()
            )
            else None
        )
        canonical_smiles = structures.get("canonical_smiles")
        smiles_valid = bool(
            isinstance(canonical_smiles, str)
            and canonical_smiles
            and canonical_smiles == canonical_smiles.strip()
            and "\r" not in canonical_smiles
            and "\n" not in canonical_smiles
            and len(canonical_smiles) <= 10_000
        )
        exact_aliases = list(aliases_by_id.get(molecule_id, ()))
        source_row = {
            "molecule_chembl_id": molecule.get("molecule_chembl_id"),
            "canonical_parent_molecule_chembl_id": molecule_id,
            "pref_name": pref_name if isinstance(pref_name, str) else None,
            "max_phase": max_phase,
            "canonical_smiles": (
                canonical_smiles if isinstance(canonical_smiles, str) else None
            ),
            "molecule_synonyms_source_key_present": source_key_present,
            "molecule_synonyms_source_type": (
                "missing" if not source_key_present else type(raw_synonyms).__name__
            ),
            "molecule_synonyms_source_row_count": (
                len(raw_synonyms) if isinstance(raw_synonyms, list) else None
            ),
            "molecule_synonyms_source_sha256": (
                _canonical_sha256(raw_synonyms) if source_key_present else None
            ),
            "molecule_synonym_row_limit": synonym_row_limit,
            "molecule_synonym_retained_count": len(synonym_rows),
            "molecule_synonyms_truncated": bool(
                isinstance(raw_synonyms, list)
                and len(raw_synonyms) > synonym_row_limit
            ),
            "molecule_synonyms": synonym_rows,
            "exact_aliases": exact_aliases,
            "molecule_query_parameter_sha256": query.get("parameter_hash"),
            "molecule_response_sha256": query.get("response_sha256"),
            "source_molecule_row_sha256": (
                _canonical_sha256(molecule) if molecule else None
            ),
            "fixture": False,
        }
        row_complete = bool(
            _CHEMBL_STABLE_ID.fullmatch(molecule_id)
            and molecule.get("molecule_chembl_id") == molecule_id
            and pref_name_valid
            and max_phase is not None
            and smiles_valid
            and synonym_rows_valid
            and exact_aliases
            and len(exact_aliases) <= _MAX_EXACT_CANDIDATE_ALIASES
            and query.get("parameter_hash")
            and query.get("response_sha256")
        )
        row = {
            "schema_version": "chembl_fixed_pool_molecule_projection_row_v1",
            "pool_index": pool_index,
            **source_row,
            "projection_complete": row_complete,
        }
        row["source_field_projection_sha256"] = _canonical_sha256(source_row)
        row["row_sha256"] = _canonical_sha256(row)
        rows.append(row)

    projection: dict[str, Any] = {
        "schema_version": "chembl_fixed_pool_molecule_projection_v1",
        "fixed_pool_candidate_ids": list(fixed_pool_ids),
        "fixed_pool_candidate_ids_sha256": _canonical_sha256(
            list(fixed_pool_ids)
        ),
        "molecule_query": query,
        "molecule_query_sha256": _canonical_sha256(query),
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
        "complete": bool(
            rows
            and len(rows) == len(fixed_pool_ids)
            and all(row["projection_complete"] for row in rows)
        ),
        "fixture": False,
    }
    projection["attestation_sha256"] = _canonical_sha256(projection)
    return projection


def _deduplicated_call_provenance(
    calls: Iterable[HTTPCallProvenance | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve first-observed exact calls while collapsing copied batches."""

    output: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str, str]] = set()
    for raw in calls:
        item = raw.as_dict() if isinstance(raw, HTTPCallProvenance) else dict(raw)
        key = (
            str(item.get("source") or ""),
            str(item.get("endpoint") or ""),
            str(item.get("parameter_hash") or ""),
            str(item.get("response_sha256") or ""),
        )
        if key in observed:
            continue
        observed.add(key)
        output.append(deepcopy(item))
    return output


_PAPER_NATIVE_V5_NAMESPACE_ALIAS_RE = re.compile(
    r"^(?:CHEMBL|PUBCHEM(?:\s+CID)?|CID|DRUGBANK|DB|UNII|CAS)"
    r"(?:[\s:_-]*[A-Z0-9][A-Z0-9._-]*)$",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_CAS_ALIAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


def _paper_native_v5_alias_key(value: str) -> str:
    """Use the parser/search module's exact NFKC and casefold identity."""

    return unicodedata.normalize("NFKC", value).casefold()


def _paper_native_v5_namespace_shaped_alias(value: str) -> bool:
    return bool(
        _PAPER_NATIVE_V5_NAMESPACE_ALIAS_RE.fullmatch(value)
        or _PAPER_NATIVE_V5_CAS_ALIAS_RE.fullmatch(value)
    )


def _paper_native_v5_identity_tokens(value: str) -> tuple[str, ...]:
    """Return complete alphanumeric tokens after exact Unicode folding."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _paper_native_v5_target_containing_alias(
    value: str,
    *,
    target_aliases: Sequence[str],
) -> bool:
    alias_identity = _normalize_identity_phrase(value)
    if not alias_identity:
        return True
    alias_tokens = _paper_native_v5_identity_tokens(value)
    for target_alias in target_aliases:
        target_identity = _normalize_identity_phrase(target_alias)
        if not target_identity:
            continue
        if (
            re.search(
                rf"(?<!\w){re.escape(target_identity)}(?!\w)",
                alias_identity,
            )
            is not None
        ):
            return True

        # ChEMBL aliases sometimes insert punctuation inside a target symbol.
        # Match only a complete contiguous token sequence whose collapsed
        # identity is exactly the target identity.  This catches punctuation
        # equivalence without accepting incidental substrings in longer tokens.
        target_tokens = _paper_native_v5_identity_tokens(target_alias)
        target_collapsed = "".join(target_tokens)
        if not target_collapsed:
            continue
        for start in range(len(alias_tokens)):
            collapsed = ""
            for token in alias_tokens[start:]:
                collapsed += token
                if len(collapsed) > len(target_collapsed):
                    break
                if collapsed == target_collapsed:
                    return True
    return False


def _paper_native_v5_alias_inventory(
    *,
    target: TargetCandidate,
    fixed_pool_ids: Sequence[str],
    direct_support_candidate_ids: Sequence[str],
    molecules_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, PaperNativeV5CandidateSearchAliases],
]:
    """Build an exact, source-attested alias inventory before PubMed.

    Preferred names are accepted from their named ChEMBL field. Synonyms are
    accepted only when their deposited ``syn_type`` is explicitly allowlisted.
    Identifier-shaped, target-containing, and cross-owner aliases are removed
    with reason-coded attestations rather than silently repaired.
    """

    target_aliases = _paired_v4_target_aliases(target)
    direct_id_set = set(direct_support_candidate_ids)
    provisional_by_id: dict[str, list[dict[str, Any]]] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    for fixed_rank, chembl_id in enumerate(fixed_pool_ids, start=1):
        if chembl_id not in direct_id_set:
            continue
        molecule = molecules_by_id.get(chembl_id)
        malformed_source_codes: list[str] = []
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        if not isinstance(molecule, Mapping):
            malformed_source_codes.append("MOLECULE_RECORD_MISSING")
            molecule = {}

        preferred = molecule.get("pref_name")
        source_values: list[tuple[str, str | None, Any, int | None]] = [
            ("pref_name", None, preferred, None)
        ]
        synonyms_present = "molecule_synonyms" in molecule
        raw_synonyms = molecule.get("molecule_synonyms")
        synonym_source_reason_codes: list[str] = []
        synonym_source_rows: list[dict[str, Any]] = []
        synonym_row_limit = _MAX_EXACT_CANDIDATE_ALIASES * 2
        if not synonyms_present:
            synonym_source_reason_codes.append(
                "MOLECULE_SYNONYMS_KEY_MISSING"
            )
        elif raw_synonyms is None:
            synonym_source_reason_codes.append("MOLECULE_SYNONYMS_NULL")
        elif not isinstance(raw_synonyms, list):
            synonym_source_reason_codes.append("MOLECULE_SYNONYMS_NOT_LIST")
        else:
            if len(raw_synonyms) > synonym_row_limit:
                synonym_source_reason_codes.append(
                    "MOLECULE_SYNONYMS_BOUND_EXCEEDED"
                )
            for index, item in enumerate(raw_synonyms[:synonym_row_limit]):
                row_reason_codes: list[str] = []
                if not isinstance(item, Mapping):
                    row_reason_codes.append(
                        "MOLECULE_SYNONYM_ROW_NOT_OBJECT"
                    )
                    synonym_source_rows.append(
                        {
                            "source_index": index,
                            "row_type": type(item).__name__,
                            "row_keys": [],
                            "syn_type": None,
                            "molecule_synonym": None,
                            "reason_codes": row_reason_codes,
                        }
                    )
                    synonym_source_reason_codes.extend(row_reason_codes)
                    continue
                syn_type = item.get("syn_type")
                synonym = item.get("molecule_synonym")
                if (
                    not isinstance(syn_type, str)
                    or not syn_type
                    or syn_type != syn_type.strip()
                    or not isinstance(synonym, str)
                    or not synonym
                    or synonym != synonym.strip()
                ):
                    row_reason_codes.append(
                        "MOLECULE_SYNONYM_FIELDS_MALFORMED"
                    )
                synonym_source_rows.append(
                    {
                        "source_index": index,
                        "row_type": "mapping",
                        "row_keys": sorted(str(key) for key in item),
                        "syn_type": syn_type if isinstance(syn_type, str) else None,
                        "molecule_synonym": (
                            synonym if isinstance(synonym, str) else None
                        ),
                        "reason_codes": row_reason_codes,
                    }
                )
                if row_reason_codes:
                    synonym_source_reason_codes.extend(row_reason_codes)
                    continue
                source_values.append(
                    ("molecule_synonyms", syn_type, synonym, index)
                )

        malformed_source_codes.extend(synonym_source_reason_codes)
        synonym_source_inventory: dict[str, Any] = {
            "schema_version": "paper_native_v5_molecule_synonyms_source_v1",
            "source": "ChEMBL.molecule.molecule_synonyms",
            "field": "molecule_synonyms",
            "source_key_present": synonyms_present,
            "list_typed": isinstance(raw_synonyms, list),
            "source_value_type": (
                "missing" if not synonyms_present else type(raw_synonyms).__name__
            ),
            "source_row_count": (
                len(raw_synonyms) if isinstance(raw_synonyms, list) else None
            ),
            "mapping_row_count": (
                sum(isinstance(item, Mapping) for item in raw_synonyms)
                if isinstance(raw_synonyms, list)
                else None
            ),
            "non_mapping_row_indices": (
                [
                    index
                    for index, item in enumerate(raw_synonyms)
                    if not isinstance(item, Mapping)
                ]
                if isinstance(raw_synonyms, list)
                else []
            ),
            "rows_sha256": (
                _canonical_sha256(raw_synonyms)
                if isinstance(raw_synonyms, list)
                else None
            ),
            "observed_value_sha256": (
                _canonical_sha256(raw_synonyms) if synonyms_present else None
            ),
            "bounded_row_limit": synonym_row_limit,
            "inspected_row_count": len(synonym_source_rows),
            "truncated": bool(
                isinstance(raw_synonyms, list)
                and len(raw_synonyms) > synonym_row_limit
            ),
            "rows": synonym_source_rows,
            "reason_codes": sorted(set(synonym_source_reason_codes)),
            "complete": not bool(synonym_source_reason_codes),
        }
        synonym_source_inventory["inventory_sha256"] = _canonical_sha256(
            synonym_source_inventory
        )

        for source, syn_type, raw_alias, source_index in source_values:
            if not isinstance(raw_alias, str) or not raw_alias:
                if source == "pref_name":
                    malformed_source_codes.append("PREFERRED_NAME_MALFORMED")
                continue
            if (
                raw_alias != raw_alias.strip()
                or "\r" in raw_alias
                or "\n" in raw_alias
                or len(raw_alias) > 160
            ):
                malformed_source_codes.append("ALIAS_STRING_MALFORMED")
                continue
            source_record = {
                "source": source,
                "source_index": source_index,
                "syn_type": syn_type,
                "alias": raw_alias,
            }
            if (
                source == "molecule_synonyms"
                and syn_type not in _PRIMARY_PAPER_NATIVE_V5_TRUSTED_SYNONYM_TYPES
            ):
                rejected.append(
                    {**source_record, "reason_code": "SYN_TYPE_NOT_TRUSTED"}
                )
                continue
            alias_key = _paper_native_v5_alias_key(raw_alias)
            if not alias_key or alias_key in seen_keys:
                rejected.append(
                    {**source_record, "reason_code": "EMPTY_OR_DUPLICATE_ALIAS"}
                )
                continue
            if _paper_native_v5_namespace_shaped_alias(raw_alias):
                rejected.append(
                    {**source_record, "reason_code": "NAMESPACE_SHAPED_ALIAS"}
                )
                continue
            if _paper_native_v5_target_containing_alias(
                raw_alias,
                target_aliases=target_aliases,
            ):
                rejected.append(
                    {**source_record, "reason_code": "TARGET_CONTAINING_ALIAS"}
                )
                continue
            seen_keys.add(alias_key)
            accepted.append({**source_record, "alias_key": alias_key})

        provisional_by_id[chembl_id] = accepted
        records_by_id[chembl_id] = {
            "fixed_pool_rank": fixed_rank,
            "molecule_chembl_id": chembl_id,
            "preferred_name": preferred if isinstance(preferred, str) else None,
            "synonym_source_present": synonyms_present,
            "synonym_source_row_count": (
                len(raw_synonyms) if isinstance(raw_synonyms, list) else None
            ),
            "molecule_synonyms_source_inventory": synonym_source_inventory,
            "molecule_synonyms_source_row_count": synonym_source_inventory[
                "source_row_count"
            ],
            "molecule_synonyms_source_sha256": synonym_source_inventory[
                "inventory_sha256"
            ],
            "molecule_synonyms_source_reason_codes": list(
                synonym_source_inventory["reason_codes"]
            ),
            "molecule_synonyms_source_complete": synonym_source_inventory[
                "complete"
            ],
            "trusted_synonym_types": sorted(
                _PRIMARY_PAPER_NATIVE_V5_TRUSTED_SYNONYM_TYPES
            ),
            "malformed_source_codes": sorted(set(malformed_source_codes)),
            "accepted_before_owner_check": deepcopy(accepted),
            "rejected_aliases": rejected,
        }

    owners: dict[str, set[str]] = {}
    for chembl_id, aliases in provisional_by_id.items():
        for item in aliases:
            owners.setdefault(str(item["alias_key"]), set()).add(chembl_id)
    conflicting_keys = {
        key: sorted(owner_ids)
        for key, owner_ids in owners.items()
        if len(owner_ids) != 1
    }

    search_candidates: dict[str, PaperNativeV5CandidateSearchAliases] = {}
    records: list[dict[str, Any]] = []
    for chembl_id in direct_support_candidate_ids:
        record = records_by_id.get(
            chembl_id,
            {
                "fixed_pool_rank": None,
                "molecule_chembl_id": chembl_id,
                "malformed_source_codes": ["ALIAS_RECORD_MISSING"],
                "accepted_before_owner_check": [],
                "rejected_aliases": [],
            },
        )
        retained: list[dict[str, Any]] = []
        owner_rejected: list[dict[str, Any]] = []
        for item in provisional_by_id.get(chembl_id, []):
            alias_key = str(item["alias_key"])
            if alias_key in conflicting_keys:
                owner_rejected.append(
                    {
                        **item,
                        "reason_code": "ALIAS_OWNER_CONFLICT",
                        "owner_molecule_chembl_ids": conflicting_keys[alias_key],
                    }
                )
            else:
                retained.append(item)
        bounded = retained[:_PRIMARY_PAPER_NATIVE_V5_MAX_ALIASES]
        if len(retained) > len(bounded):
            record["malformed_source_codes"] = sorted(
                {
                    *record.get("malformed_source_codes", []),
                    "TRUSTED_ALIAS_BOUND_EXCEEDED",
                }
            )
        exact_aliases = tuple(str(item["alias"]) for item in bounded)
        preferred_alias = exact_aliases[0] if exact_aliases else None
        if exact_aliases and not record.get("malformed_source_codes"):
            search_candidates[chembl_id] = PaperNativeV5CandidateSearchAliases(
                preferred_search_alias=str(preferred_alias),
                exact_aliases=exact_aliases,
            )
        finalized_record = {
            **record,
            "owner_conflict_rejections": owner_rejected,
            "accepted_aliases": list(exact_aliases),
            "accepted_alias_count": len(exact_aliases),
            "preferred_search_alias": preferred_alias,
            "source_attestation_complete": not bool(
                record.get("malformed_source_codes")
            ),
            "query_eligible": chembl_id in search_candidates,
        }
        finalized_record["record_sha256"] = _canonical_sha256(finalized_record)
        records.append(finalized_record)

    complete = bool(
        direct_support_candidate_ids
        and len(search_candidates) == len(direct_support_candidate_ids)
    )
    body: dict[str, Any] = {
        "schema_version": "paper_native_v5_alias_inventory_v1",
        "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
        "source_policy": (
            "chembl_pref_name_plus_explicit_trusted_syn_type_v1"
        ),
        "target_aliases": list(target_aliases),
        "endpoint": "IC50",
        "fixed_pool_candidate_ids": list(fixed_pool_ids),
        "direct_support_candidate_ids": list(direct_support_candidate_ids),
        "queried_candidate_ids": list(search_candidates),
        "trusted_synonym_types": sorted(
            _PRIMARY_PAPER_NATIVE_V5_TRUSTED_SYNONYM_TYPES
        ),
        "per_candidate_alias_limit": _PRIMARY_PAPER_NATIVE_V5_MAX_ALIASES,
        "records": records,
        "owner_conflicts": [
            {"alias_key": key, "molecule_chembl_ids": owner_ids}
            for key, owner_ids in sorted(conflicting_keys.items())
        ],
        "complete": complete,
        "normalization_applied_to_source_aliases": False,
        "pubmed_or_result_used_for_alias_selection": False,
        "fixture": False,
    }
    body["inventory_sha256"] = _canonical_sha256(body)
    return body, search_candidates if complete else {}


def _paper_native_v5_support_row_audit(
    *,
    target: TargetCandidate,
    target_chembl_id: str,
    chembl_id: str,
    fixed_pool_rank: int,
    retained_index: int,
    raw: Mapping[str, Any],
    measurement: ActivityMeasurement,
    direction_rule: str,
    direction: str,
    assay_context: Mapping[str, Any] | None,
    activity_call: HTTPCallProvenance | None,
    activity_query_parameters: Mapping[str, Any],
    activity_endpoint: str,
    candidate_context_reason_codes: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Audit one retained row and freeze it only if every direct gate holds."""

    reasons: list[str] = list(candidate_context_reason_codes)
    activity_id = str(raw.get("activity_id") or "").strip()
    assay_id = str(raw.get("assay_chembl_id") or "").strip()
    raw_document_id = raw.get("document_chembl_id")
    document_id = (
        raw_document_id
        if isinstance(raw_document_id, str)
        and _CHEMBL_STABLE_ID.fullmatch(raw_document_id) is not None
        else None
    )
    activity_query = (
        activity_call.as_dict() if activity_call is not None else None
    )
    activity_query_parameters_snapshot = deepcopy(
        dict(activity_query_parameters)
    )
    activity_parameter_sha256 = _source_parameter_sha256(
        activity_query_parameters_snapshot
    )
    activity_query_binding: dict[str, Any] = {
        "molecule_chembl_id": chembl_id,
        "target_chembl_id": target_chembl_id,
        "endpoint": activity_endpoint,
        "parameters": activity_query_parameters_snapshot,
        "parameter_sha256": activity_parameter_sha256,
    }
    if (
        target.metadata.get("chembl_target_id") != target_chembl_id
        or target.metadata.get("chembl_target_type") != "SINGLE PROTEIN"
        or target.metadata.get("chembl_organism") != "Homo sapiens"
        or not isinstance(target.uniprot_id, str)
        or not target.uniprot_id.strip()
    ):
        reasons.append("TARGET_CHEMBL_UNIPROT_HUMAN_BINDING_INCOMPLETE")
    if (
        raw.get("molecule_chembl_id") != chembl_id
        or raw.get("parent_molecule_chembl_id") != chembl_id
    ):
        reasons.append("EXACT_PARENT_MOLECULE_BINDING_FAILED")
    if raw.get("target_chembl_id") != target_chembl_id:
        reasons.append("EXACT_TARGET_CHEMBL_BINDING_FAILED")
    if direction != "inhibit" or direction_rule != "measured_ic50_endpoint":
        reasons.append("INHIBIT_DIRECTION_BINDING_FAILED")
    if measurement.endpoint != "IC50" or raw.get("standard_type") != "IC50":
        reasons.append("IC50_ENDPOINT_BINDING_FAILED")
    if measurement.relation != "=" or raw.get("standard_relation") != "=":
        reasons.append("EXACT_RELATION_BINDING_FAILED")
    if (
        raw.get("standard_units") != measurement.unit
        or not _decimal_equal(raw.get("standard_value"), measurement.value)
    ):
        reasons.append("RAW_TYPED_MEASUREMENT_TUPLE_MISMATCH")
    try:
        raw_decimal = Decimal(str(raw.get("standard_value")))
        typed_decimal = Decimal(str(measurement.value))
    except (InvalidOperation, TypeError, ValueError):
        raw_decimal = Decimal("NaN")
        typed_decimal = Decimal("NaN")
    if (
        not raw_decimal.is_finite()
        or raw_decimal <= 0
        or not typed_decimal.is_finite()
        or typed_decimal <= 0
    ):
        reasons.append("MEASUREMENT_NOT_POSITIVE_FINITE")
    if (
        "data_validity_comment" not in raw
        or raw.get("data_validity_comment") not in (None, "", "Manually validated")
    ):
        reasons.append("DATA_VALIDITY_NOT_ALLOWED")
    if not activity_id or not assay_id or activity_call is None:
        reasons.append("ACTIVITY_PROVENANCE_INCOMPLETE")
    elif (
        activity_call.source != "ChEMBL"
        or activity_call.endpoint != activity_endpoint
        or list(activity_call.parameter_keys)
        != sorted(activity_query_parameters_snapshot)
        or activity_call.parameter_hash != activity_parameter_sha256
        or activity_query_parameters_snapshot.get("target_chembl_id")
        != target_chembl_id
        or activity_query_parameters_snapshot.get(
            "parent_molecule_chembl_id"
        )
        != chembl_id
        or activity_query_parameters_snapshot.get("target_organism")
        != "Homo sapiens"
        or activity_query_parameters_snapshot.get("assay_type") != "B"
        or activity_query_parameters_snapshot.get("standard_type") != "IC50"
        or activity_query_parameters_snapshot.get("order_by") != "activity_id"
        or activity_query_parameters_snapshot.get("limit")
        != _MAX_ACTIVITY_ROWS_PER_CANDIDATE
    ):
        reasons.append("ACTIVITY_QUERY_CANDIDATE_BINDING_INCOMPLETE")
    if document_id is None:
        reasons.append("DOCUMENT_PROVENANCE_INCOMPLETE")
    if not _assay_supports_direct_target_engagement(measurement.assay_type):
        reasons.append("TYPED_ASSAY_NOT_DIRECT_SINGLE_PROTEIN")

    assay_record = (
        assay_context.get("record")
        if isinstance(assay_context, Mapping)
        else None
    )
    if (
        not isinstance(assay_context, Mapping)
        or assay_context.get("schema_version") != CHEMBL_ASSAY_CONTEXT_SCHEMA
        or assay_context.get("retrieval_status") != "completed"
        or assay_context.get("binding_status") != "exact"
        or assay_context.get("assay_chembl_id") != assay_id
        or assay_context.get("missing_values_imputed") is not False
        or assay_context.get("inference_used") is not False
        or not isinstance(assay_record, Mapping)
    ):
        reasons.append("ASSAY_CONTEXT_BINDING_INCOMPLETE")
        assay_record = {}
    if (
        assay_record.get("assay_chembl_id") != assay_id
        or assay_record.get("target_chembl_id") != target_chembl_id
        or assay_record.get("assay_type") != "B"
        or assay_record.get("assay_organism") != "Homo sapiens"
        or assay_record.get("bao_format") != "BAO_0000357"
        or not _assay_supports_direct_target_engagement(
            str(assay_record.get("bao_label") or "")
        )
    ):
        reasons.append("BAO_SINGLE_PROTEIN_ASSAY_BINDING_INCOMPLETE")

    assay_context_snapshot = (
        deepcopy(dict(assay_context))
        if isinstance(assay_context, Mapping)
        else None
    )

    raw_tuple = {
        "activity_id": activity_id or None,
        "molecule_chembl_id": raw.get("molecule_chembl_id"),
        "parent_molecule_chembl_id": raw.get("parent_molecule_chembl_id"),
        "target_chembl_id": raw.get("target_chembl_id"),
        "assay_chembl_id": assay_id or None,
        "document_chembl_id": document_id or None,
        "assay_type": raw.get("assay_type"),
        "bao_label": raw.get("bao_label"),
        "standard_type": raw.get("standard_type"),
        "standard_relation": raw.get("standard_relation"),
        "standard_value": raw.get("standard_value"),
        "standard_units": raw.get("standard_units"),
        "data_validity_comment": raw.get("data_validity_comment"),
        "pchembl_value": raw.get("pchembl_value"),
    }
    typed_tuple = {
        "endpoint": measurement.endpoint,
        "relation": measurement.relation,
        "value": measurement.value,
        "unit": measurement.unit,
        "assay_type": measurement.assay_type,
        "source_evidence_id": measurement.source_evidence_id,
    }
    audit: dict[str, Any] = {
        "schema_version": "paper_native_v5_direct_support_row_audit_v1",
        "molecule_chembl_id": chembl_id,
        "fixed_pool_rank": fixed_pool_rank,
        "retained_rank": retained_index + 1,
        "retained_index": retained_index,
        "activity_id": activity_id or None,
        "assay_chembl_id": assay_id or None,
        "document_chembl_id": document_id or None,
        "status": "supported" if not reasons else "not_supported",
        "reason_codes": sorted(set(reasons)),
        "raw_measurement_tuple": raw_tuple,
        "raw_measurement_tuple_sha256": _canonical_sha256(raw_tuple),
        "typed_measurement_tuple": typed_tuple,
        "typed_measurement_tuple_sha256": _canonical_sha256(typed_tuple),
        "assay_context": assay_context_snapshot,
        "assay_context_sha256": (
            _canonical_sha256(assay_context_snapshot)
            if assay_context_snapshot is not None
            else None
        ),
        "chembl_activity_query": deepcopy(activity_query),
        "chembl_activity_query_sha256": (
            _canonical_sha256(activity_query)
            if activity_query is not None
            else None
        ),
        "chembl_activity_query_parameters": (
            activity_query_parameters_snapshot
        ),
        "chembl_activity_query_parameters_sha256": (
            activity_parameter_sha256
        ),
        "chembl_activity_query_candidate_binding": activity_query_binding,
        "chembl_activity_query_candidate_binding_sha256": (
            _canonical_sha256(activity_query_binding)
        ),
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
        "fixture": False,
    }
    audit["row_audit_sha256"] = _canonical_sha256(audit)
    if reasons:
        return audit, None

    support: dict[str, Any] = {
        "schema_version": "paper_native_v5_pre_pubmed_chembl_support_v1",
        "verification_status": "verified",
        "reason_code": None,
        "molecule_chembl_id": chembl_id,
        "fixed_pool_rank": fixed_pool_rank,
        "retained_rank": retained_index + 1,
        "retained_index": retained_index,
        "activity_id": activity_id,
        "source_evidence_id": measurement.source_evidence_id,
        "target_chembl_id": target_chembl_id,
        "target_uniprot_id": target.uniprot_id,
        "target_organism": "Homo sapiens",
        "target_type": "SINGLE PROTEIN",
        "assay_chembl_id": assay_id,
        "document_chembl_id": document_id or None,
        "raw_measurement_tuple": raw_tuple,
        "raw_measurement_tuple_sha256": audit["raw_measurement_tuple_sha256"],
        "typed_measurement_tuple": typed_tuple,
        "typed_measurement_tuple_sha256": audit[
            "typed_measurement_tuple_sha256"
        ],
        "assay_context": deepcopy(assay_context_snapshot),
        "assay_context_sha256": audit["assay_context_sha256"],
        "chembl_activity_query": deepcopy(activity_query),
        "chembl_activity_query_sha256": audit[
            "chembl_activity_query_sha256"
        ],
        "chembl_activity_query_parameters": deepcopy(
            activity_query_parameters_snapshot
        ),
        "chembl_activity_query_parameters_sha256": activity_parameter_sha256,
        "chembl_activity_query_candidate_binding": deepcopy(
            activity_query_binding
        ),
        "chembl_activity_query_candidate_binding_sha256": audit[
            "chembl_activity_query_candidate_binding_sha256"
        ],
        "selected_before_pubmed_search": True,
        "selection_policy": "first_strict_direct_support_in_retained_order_v1",
        "pubmed_value_used_for_selection": False,
        "cross_source_numeric_equality_required": False,
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
        "fixture": False,
    }
    support["support_attestation_sha256"] = _canonical_sha256(support)
    return audit, support


def _paper_native_v5_direct_support_inventory(
    *,
    target: TargetCandidate,
    target_chembl_id: str,
    fixed_pool_ids: Sequence[str],
    activity_rows_by_molecule: Mapping[
        str,
        Sequence[tuple[dict[str, Any], ActivityMeasurement, str]],
    ],
    assay_contexts: Mapping[str, Mapping[str, Any]],
    activity_calls_by_id: Mapping[str, HTTPCallProvenance],
    activity_row_queries_by_id: Mapping[
        str,
        Mapping[str, tuple[HTTPCallProvenance, dict[str, Any]]],
    ],
    identity_resolution: Mapping[str, Mapping[str, Any]],
    mechanism_provenance_by_id: Mapping[str, Mapping[str, Any]],
    molecules_by_id: Mapping[str, Mapping[str, Any]],
    activity_endpoint: str,
    min_pchembl_value: float,
    direction: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Select each candidate's first strict support before PubMed is called."""

    records: list[dict[str, Any]] = []
    support_by_id: dict[str, dict[str, Any]] = {}
    fixed_pool_identity_bindings: list[dict[str, Any]] = []
    for fixed_rank, chembl_id in enumerate(fixed_pool_ids, start=1):
        identity_snapshot = (
            deepcopy(dict(identity_resolution[chembl_id]))
            if isinstance(identity_resolution.get(chembl_id), Mapping)
            else None
        )
        mechanism_snapshot = (
            deepcopy(dict(mechanism_provenance_by_id[chembl_id]))
            if isinstance(mechanism_provenance_by_id.get(chembl_id), Mapping)
            else None
        )
        molecule = molecules_by_id.get(chembl_id)
        identity_phase = (
            identity_snapshot.get("max_phase")
            if isinstance(identity_snapshot, Mapping)
            else None
        )
        molecule_snapshot = (
            {
                "molecule_chembl_id": molecule.get("molecule_chembl_id"),
                # Discovery has already parsed the deposited molecule phase,
                # required an integral clinical phase, and matched it exactly
                # to the mechanism phase.  Bind the v5 attestation to that
                # canonical identity value rather than to ChEMBL's equivalent
                # wire representation (for example ``"4.0"``).
                "max_phase": identity_phase,
                "molecule_type": molecule.get("molecule_type"),
            }
            if isinstance(molecule, Mapping)
            else None
        )
        context_reasons: list[str] = []
        molecule_phase = (
            molecule_snapshot.get("max_phase")
            if isinstance(molecule_snapshot, Mapping)
            else None
        )
        if (
            not isinstance(identity_snapshot, Mapping)
            or identity_snapshot.get("molecule_chembl_id") != chembl_id
            or identity_snapshot.get("parent_molecule_chembl_id") != chembl_id
            or identity_snapshot.get("status") != "selected"
            or isinstance(identity_phase, bool)
            or not isinstance(identity_phase, int)
            or identity_phase < 1
        ):
            context_reasons.append("FIXED_POOL_IDENTITY_RESOLUTION_INCOMPLETE")
        if (
            not isinstance(molecule_snapshot, Mapping)
            or molecule_snapshot.get("molecule_chembl_id") != chembl_id
            or molecule_snapshot.get("molecule_type") != "Small molecule"
            or isinstance(molecule_phase, bool)
            or not isinstance(molecule_phase, int)
            or molecule_phase != identity_phase
        ):
            context_reasons.append("FIXED_POOL_MOLECULE_PHASE_BINDING_INCOMPLETE")
        mechanism_rows = (
            mechanism_snapshot.get("rows")
            if isinstance(mechanism_snapshot, Mapping)
            else None
        )
        if (
            not isinstance(mechanism_snapshot, Mapping)
            or mechanism_snapshot.get("schema_version")
            != "chembl_mechanism_identity_v1"
            or mechanism_snapshot.get("parent_molecule_chembl_id") != chembl_id
            or not isinstance(mechanism_rows, list)
            or not mechanism_rows
            or any(
                not isinstance(row, Mapping)
                or row.get("canonical_parent_molecule_chembl_id") != chembl_id
                or row.get("target_chembl_id") != target_chembl_id
                or row.get("action_type") != "INHIBITOR"
                or row.get("direct_interaction") not in (1, "1", True)
                or row.get("molecular_mechanism") not in (1, "1", True)
                or _safe_float(row.get("max_phase")) != identity_phase
                for row in (mechanism_rows or [])
            )
        ):
            context_reasons.append("FIXED_POOL_MECHANISM_BINDING_INCOMPLETE")
        candidate_identity_binding: dict[str, Any] = {
            "schema_version": "paper_native_v5_fixed_pool_identity_binding_v1",
            "molecule_chembl_id": chembl_id,
            "fixed_pool_rank": fixed_rank,
            "target_chembl_id": target_chembl_id,
            "target_uniprot_id": target.uniprot_id,
            "target_type": target.metadata.get("chembl_target_type"),
            "target_organism": target.metadata.get("chembl_organism"),
            "identity_resolution": identity_snapshot,
            "mechanism_provenance": mechanism_snapshot,
            "molecule_metadata": molecule_snapshot,
            "binding_status": (
                "exact" if not context_reasons else "incomplete"
            ),
            "reason_codes": sorted(set(context_reasons)),
        }
        candidate_identity_binding["binding_sha256"] = _canonical_sha256(
            candidate_identity_binding
        )
        fixed_pool_identity_bindings.append(candidate_identity_binding)

        activity_query_parameters = {
            "target_chembl_id": target_chembl_id,
            "parent_molecule_chembl_id": chembl_id,
            "target_organism": "Homo sapiens",
            "assay_type": "B",
            "standard_type": "IC50",
            "pchembl_value__gte": min_pchembl_value,
            "order_by": "activity_id",
            "limit": _MAX_ACTIVITY_ROWS_PER_CANDIDATE,
        }
        row_audits: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for retained_index, (raw, measurement, direction_rule) in enumerate(
            activity_rows_by_molecule.get(chembl_id, ())
        ):
            assay_id = str(raw.get("assay_chembl_id") or "").strip()
            activity_id = str(raw.get("activity_id") or "").strip()
            row_query = activity_row_queries_by_id.get(chembl_id, {}).get(
                activity_id
            )
            row_activity_call = (
                row_query[0]
                if row_query is not None
                else activity_calls_by_id.get(chembl_id)
            )
            row_activity_parameters = (
                deepcopy(row_query[1])
                if row_query is not None
                else activity_query_parameters
            )
            audit, support = _paper_native_v5_support_row_audit(
                target=target,
                target_chembl_id=target_chembl_id,
                chembl_id=chembl_id,
                fixed_pool_rank=fixed_rank,
                retained_index=retained_index,
                raw=raw,
                measurement=measurement,
                direction_rule=direction_rule,
                direction=direction,
                assay_context=assay_contexts.get(assay_id),
                activity_call=row_activity_call,
                activity_query_parameters=row_activity_parameters,
                activity_endpoint=activity_endpoint,
                candidate_context_reason_codes=context_reasons,
            )
            row_audits.append(audit)
            if selected is None and support is not None:
                selected = support
        if selected is not None:
            support_by_id[chembl_id] = selected
        candidate_reason_codes = sorted(
            {
                code
                for audit in row_audits
                for code in audit.get("reason_codes", [])
            }
        )
        record: dict[str, Any] = {
            "molecule_chembl_id": chembl_id,
            "fixed_pool_rank": fixed_rank,
            "candidate_identity_binding": deepcopy(
                candidate_identity_binding
            ),
            "candidate_identity_binding_sha256": candidate_identity_binding[
                "binding_sha256"
            ],
            "retained_row_count": len(row_audits),
            "row_audits": row_audits,
            "selection_status": (
                "selected" if selected is not None else "not_selected"
            ),
            "reason_code": (
                None
                if selected is not None
                else "NO_STRICT_DIRECT_CHEMBL_SUPPORT_IN_RETAINED_ROWS"
            ),
            "observed_row_reason_codes": candidate_reason_codes,
            "selected_support": deepcopy(selected),
        }
        record["record_sha256"] = _canonical_sha256(record)
        records.append(record)
    selected_ids = [
        chembl_id for chembl_id in fixed_pool_ids if chembl_id in support_by_id
    ]
    snapshots = [deepcopy(support_by_id[item]) for item in selected_ids]
    selection_order = [
        {
            "molecule_chembl_id": item["molecule_chembl_id"],
            "fixed_pool_rank": item["fixed_pool_rank"],
            "retained_rank": item["retained_rank"],
            "activity_id": item["activity_id"],
        }
        for item in snapshots
    ]
    body: dict[str, Any] = {
        "schema_version": "paper_native_v5_direct_support_inventory_v1",
        "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
        "fixed_pool_candidate_ids": list(fixed_pool_ids),
        "fixed_pool_candidate_ids_sha256": _canonical_sha256(
            list(fixed_pool_ids)
        ),
        "fixed_pool_identity_bindings": fixed_pool_identity_bindings,
        "fixed_pool_identity_bindings_sha256": _canonical_sha256(
            fixed_pool_identity_bindings
        ),
        "records": records,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_count": len(selected_ids),
        "selected_support_snapshots": snapshots,
        "selected_support_snapshots_sha256": _canonical_sha256(snapshots),
        "selection_order": selection_order,
        "selection_order_sha256": _canonical_sha256(selection_order),
        "selection_completed_before_pubmed_search": True,
        "pubmed_or_result_used_for_selection": False,
        "no_reordering": True,
        "no_backfill": True,
        "fixture": False,
    }
    body["inventory_sha256"] = _canonical_sha256(body)
    return body, support_by_id


def _paper_native_v5_outcome_payload(
    outcome: PaperNativeV5SearchOutcome,
) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            asdict(outcome),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _paper_native_v5_not_attempted_payload(
    *,
    reason_code: str,
    requested_candidate_count: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "policy_token": PAPER_NATIVE_V5_POLICY_TOKEN,
        "grammar": PAPER_NATIVE_V5_GRAMMAR,
        "status": "not_verified",
        "reason_code": reason_code,
        "query": None,
        "query_sha256": None,
        "query_translation": None,
        "query_translation_sha256": None,
        "search_sort": "pub date",
        "max_candidates": _PRIMARY_PAPER_NATIVE_V5_MAX_CANDIDATES,
        "max_documents": _PRIMARY_PAPER_NATIVE_V5_MAX_DOCUMENTS,
        "requested_candidate_count": requested_candidate_count,
        "queried_candidate_count": 0,
        "requested_document_count": _PRIMARY_PAPER_NATIVE_V5_MAX_DOCUMENTS,
        "returned_document_count": 0,
        "unique_document_count": 0,
        "total_result_count": 0,
        "unsearched_result_count": 0,
        "returned_document_ids": [],
        "returned_document_ids_sha256": None,
        "screened_document_count": 0,
        "screened_sentence_count": 0,
        "verified_candidate_ids": [],
        "calls": [],
        "records": [],
        "groups": [],
        "selected_group_sha256": None,
    }
    body["outcome_sha256"] = _canonical_sha256(body)
    return body


def _paper_native_v5_span_matches(sentence: str, span: Any) -> bool:
    start = getattr(span, "start", None)
    end = getattr(span, "end", None)
    text = getattr(span, "text", None)
    return bool(
        type(start) is int
        and type(end) is int
        and 0 <= start <= end <= len(sentence)
        and isinstance(text, str)
        and sentence[start:end] == text
    )


def _paper_native_v5_group_valid(
    group: PaperNativeV5Group,
    *,
    query_sha256: str,
    fixed_pool_ids: Sequence[str],
    support_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    fixed_id_set = set(fixed_pool_ids)
    primary_study = group.primary_study_attestation
    primary_study_body = (
        asdict(primary_study)
        if isinstance(primary_study, PaperNativeV5PrimaryStudyAttestation)
        else {}
    )
    observed_primary_study_hash = primary_study_body.pop(
        "attestation_sha256", None
    )
    if (
        not primary_study_body
        or observed_primary_study_hash != _canonical_sha256(primary_study_body)
        or primary_study.schema_version
        != "pubmed_primary_study_attestation/v2"
        or primary_study.policy
        != "same_abstract_experimental_method_and_linked_notice_check_v2"
        or primary_study.status != "verified"
        or primary_study.document_id != group.document_id
        or primary_study.summary_publication_types
        != group.summary_publication_types
        or primary_study.fetch_publication_types != group.fetch_publication_types
        or primary_study.retraction_safety_policy
        != "pubmed_types_and_comments_corrections_fail_closed_v1"
        or primary_study.retraction_safe is not True
        or primary_study.correction_safe is not True
        or primary_study.linked_status_notice != "none_detected"
        or primary_study.linked_status_notice_types != ()
        or primary_study.linked_status_notice_pmids != ()
        or primary_study.fixture is not False
        or not group.document_id.isdigit()
        or type(group.document_rank) is not int
        or group.document_rank < 0
        or type(group.sentence_index) is not int
        or group.sentence_index < 0
        or len(group.candidate_ids) < 3
        or len(group.candidate_ids) != len(group.pairs)
        or tuple(pair.candidate_id for pair in group.pairs)
        != group.candidate_ids
        or len(set(group.candidate_ids)) != len(group.candidate_ids)
        or not set(group.candidate_ids).issubset(fixed_id_set)
        or not set(group.candidate_ids).issubset(support_by_id)
        or group.sentence_sha256
        != sha256(group.sentence.encode("utf-8")).hexdigest()
    ):
        return False
    expected_group_hash = _canonical_sha256(
        {
            "candidate_ids": group.candidate_ids,
            "document_id": group.document_id,
            "document_rank": group.document_rank,
            "grammar": PAPER_NATIVE_V5_GRAMMAR,
            "fetch_publication_types": group.fetch_publication_types,
            "pair_evidence_sha256": [
                pair.evidence_sha256 for pair in group.pairs
            ],
            "primary_study_attestation_sha256": (
                primary_study.attestation_sha256
            ),
            "query_sha256": query_sha256,
            "schema": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
            "section": group.section,
            "sentence_index": group.sentence_index,
            "sentence_sha256": group.sentence_sha256,
            "summary_publication_types": group.summary_publication_types,
        }
    )
    if expected_group_hash != group.group_sha256:
        return False
    for pair in group.pairs:
        spans = [
            pair.sentence_span,
            pair.candidate_span,
            pair.target_span,
            pair.endpoint_span,
            pair.assay_condition.span,
            pair.assay_condition.value_span,
            pair.assay_condition.unit_span,
            pair.assay_condition.subject_span,
            pair.measurement_span,
            pair.value_span,
            pair.unit_span,
        ]
        if pair.endpoint_wrapper_span is not None:
            spans.append(pair.endpoint_wrapper_span)
        if pair.relation_span is not None:
            spans.append(pair.relation_span)
        if (
            pair.grammar != PAPER_NATIVE_V5_GRAMMAR
            or pair.section != group.section
            or set(pair.publication_types)
            != set(group.summary_publication_types).union(
                group.fetch_publication_types
            )
            or pair.sentence_sha256 != group.sentence_sha256
            or pair.sentence_span.start != 0
            or pair.sentence_span.end != len(group.sentence)
            or pair.sentence_span.text != group.sentence
            or pair.relation_basis
            not in {"literal_equality", "reported_point_estimate"}
            or pair.literal_equality != (pair.relation_raw == "=")
            or pair.relation_normalized
            != ("=" if pair.relation_raw == "=" else None)
            or not all(
                _paper_native_v5_span_matches(group.sentence, span)
                for span in spans
            )
        ):
            return False
    return True


def _paper_native_v5_promotions(
    *,
    outcome: PaperNativeV5SearchOutcome | None,
    outcome_payload: Mapping[str, Any],
    fixed_pool_ids: Sequence[str],
    support_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    summary: dict[str, Any] = {
        "schema_version": "paper_native_v5_promotion_v1",
        "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
        "status": "not_verified",
        "reason_code": None,
        "selected_group_sha256": None,
        "selected_group_candidate_ids": [],
        "verified_member_candidate_ids": [],
        "verified_member_count": 0,
        "minimum_group_candidate_count": 3,
        "group_member_failure_rejects_entire_group": True,
        "cross_source_numeric_equality_required": False,
        "pubmed_value_used_for_activity_selection": False,
        "pubmed_value_used_for_candidate_rank": False,
        "pubmed_value_used_for_candidate_ordering": False,
        "no_partial_group_promotion": True,
        "no_reordering": True,
        "no_backfill": True,
        "fixture": False,
    }
    if outcome is None:
        summary["reason_code"] = str(
            outcome_payload.get("reason_code")
            or "PAPER_NATIVE_V5_SEARCH_NOT_ATTEMPTED"
        )
        summary["promotion_sha256"] = _canonical_sha256(summary)
        return {}, summary
    outcome_body = asdict(outcome)
    observed_outcome_hash = outcome_body.pop("outcome_sha256", None)
    if (
        outcome.schema_version != PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION
        or outcome.policy_token != PAPER_NATIVE_V5_POLICY_TOKEN
        or outcome.grammar != PAPER_NATIVE_V5_GRAMMAR
        or outcome.status != "verified"
        or not outcome.query_sha256
        or observed_outcome_hash != _canonical_sha256(outcome_body)
        or outcome_payload.get("outcome_sha256") != observed_outcome_hash
    ):
        summary["reason_code"] = (
            outcome.reason_code or "PAPER_NATIVE_V5_SEARCH_NOT_VERIFIED"
        )
        summary["promotion_sha256"] = _canonical_sha256(summary)
        return {}, summary
    ordered_groups = sorted(
        outcome.groups,
        key=lambda group: (
            group.document_rank,
            group.sentence_index,
            group.group_sha256,
        ),
    )
    if (
        not ordered_groups
        or tuple(ordered_groups) != outcome.groups
        or outcome.selected_group_sha256 != ordered_groups[0].group_sha256
        or not _paper_native_v5_group_valid(
            ordered_groups[0],
            query_sha256=outcome.query_sha256,
            fixed_pool_ids=fixed_pool_ids,
            support_by_id=support_by_id,
        )
    ):
        summary["reason_code"] = "PAPER_NATIVE_V5_GROUP_INTEGRITY_MISMATCH"
        summary["promotion_sha256"] = _canonical_sha256(summary)
        return {}, summary
    selected_group = ordered_groups[0]
    pair_by_id = {pair.candidate_id: pair for pair in selected_group.pairs}
    if len(pair_by_id) != len(selected_group.candidate_ids):
        summary["reason_code"] = "PAPER_NATIVE_V5_GROUP_MEMBER_BINDING_FAILED"
        summary["promotion_sha256"] = _canonical_sha256(summary)
        return {}, summary
    member_ids = [
        chembl_id
        for chembl_id in fixed_pool_ids
        if chembl_id in pair_by_id
    ]
    if (
        len(member_ids) != len(selected_group.candidate_ids)
        or any(chembl_id not in support_by_id for chembl_id in member_ids)
    ):
        summary["reason_code"] = "PAPER_NATIVE_V5_GROUP_MEMBER_SUPPORT_FAILED"
        summary["promotion_sha256"] = _canonical_sha256(summary)
        return {}, summary
    promotions = {
        chembl_id: {
            "group": selected_group,
            "pair": pair_by_id[chembl_id],
            "pre_pubmed_chembl_support": deepcopy(support_by_id[chembl_id]),
        }
        for chembl_id in member_ids
    }
    summary.update(
        {
            "status": "verified",
            "reason_code": None,
            "selected_group_sha256": selected_group.group_sha256,
            "selected_group_candidate_ids": list(
                selected_group.candidate_ids
            ),
            "verified_member_candidate_ids": member_ids,
            "verified_member_count": len(member_ids),
        }
    )
    summary["promotion_sha256"] = _canonical_sha256(summary)
    return promotions, summary


def _paper_native_v5_candidate_envelope(
    *,
    chembl_id: str,
    target: TargetCandidate,
    target_chembl_id: str,
    fixed_pool_ids: Sequence[str],
    support_by_id: Mapping[str, Mapping[str, Any]],
    option: Mapping[str, Any] | None,
    alias_inventory: Mapping[str, Any],
    outcome_payload: Mapping[str, Any],
    promotion_summary: Mapping[str, Any],
) -> dict[str, Any]:
    support = support_by_id.get(chembl_id)
    verified = option is not None
    group = option.get("group") if isinstance(option, Mapping) else None
    pair = option.get("pair") if isinstance(option, Mapping) else None
    if verified and (
        not isinstance(group, PaperNativeV5Group)
        or not isinstance(pair, PaperNativeV5Pair)
        or not isinstance(support, Mapping)
    ):
        raise ValueError("invalid paper-native-v5 promotion option")
    if verified:
        reason_code = None
    elif support is None:
        reason_code = "PRE_PUBMED_CHEMBL_DIRECT_SUPPORT_NOT_ESTABLISHED"
    elif promotion_summary.get("status") == "verified":
        reason_code = "NOT_IN_SELECTED_PAPER_NATIVE_V5_GROUP"
    else:
        reason_code = str(
            promotion_summary.get("reason_code")
            or "PAPER_NATIVE_V5_NOT_VERIFIED"
        )
    group_payload = (
        json.loads(
            json.dumps(
                asdict(group),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
        if isinstance(group, PaperNativeV5Group)
        else None
    )
    fixed_pool_rank = (
        fixed_pool_ids.index(chembl_id) + 1
        if chembl_id in fixed_pool_ids
        else None
    )
    envelope: dict[str, Any] = {
        "schema_version": _PRIMARY_PAPER_NATIVE_V5_SCHEMA,
        "source_contract": _PRIMARY_PAPER_NATIVE_V5_SOURCE_CONTRACT,
        "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
        "verification_status": "verified" if verified else "not_verified",
        "is_primary_support": verified,
        "reason_code": reason_code,
        "source": "PubMed",
        "pmid": group.document_id if isinstance(group, PaperNativeV5Group) else None,
        "source_url": (
            f"https://pubmed.ncbi.nlm.nih.gov/{quote(group.document_id)}/"
            if isinstance(group, PaperNativeV5Group)
            else None
        ),
        "molecule_chembl_id": chembl_id,
        "fixed_pool_rank": fixed_pool_rank,
        "fixed_pool_candidate_ids": list(fixed_pool_ids),
        "fixed_pool_candidate_ids_sha256": _canonical_sha256(
            list(fixed_pool_ids)
        ),
        "target_binding": {
            "target_symbol": target.symbol,
            "target_accession": target.uniprot_id,
            "target_chembl_id": target_chembl_id,
            "target_type": target.metadata.get("chembl_target_type"),
            "target_organism": target.metadata.get("chembl_organism"),
            "target_alias": pair.target_alias if isinstance(pair, PaperNativeV5Pair) else None,
            "target_span": (
                asdict(pair.target_span)
                if isinstance(pair, PaperNativeV5Pair)
                else None
            ),
        },
        "alias_inventory": deepcopy(dict(alias_inventory)),
        "search_outcome": deepcopy(dict(outcome_payload)),
        "group": group_payload,
        "group_member_candidate_ids": (
            list(group.candidate_ids)
            if isinstance(group, PaperNativeV5Group)
            else []
        ),
        "selected_group_sha256": (
            group.group_sha256
            if isinstance(group, PaperNativeV5Group)
            else None
        ),
        "selected_pair_index": (
            pair.pair_index if isinstance(pair, PaperNativeV5Pair) else None
        ),
        "selected_pair": (
            json.loads(
                json.dumps(
                    asdict(pair),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
            if isinstance(pair, PaperNativeV5Pair)
            else None
        ),
        "selected_pair_evidence_sha256": (
            pair.evidence_sha256
            if isinstance(pair, PaperNativeV5Pair)
            else None
        ),
        "pre_pubmed_chembl_support": deepcopy(support),
        "promotion_summary": deepcopy(dict(promotion_summary)),
        "cross_source_numeric_equality_required": False,
        "cross_source_numeric_comparison_performed": False,
        "pubmed_value_used_for_activity_selection": False,
        "pubmed_value_used_for_candidate_rank": False,
        "pubmed_value_used_for_candidate_ordering": False,
        "candidate_seed_activity_is_pre_pubmed_chembl_support": bool(support),
        "no_partial_group_promotion": True,
        "no_reordering": True,
        "no_backfill": True,
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
        "fixture": False,
    }
    envelope["attestation_sha256"] = _canonical_sha256(envelope)
    return envelope


def _paired_v4_alias_key(value: str) -> str:
    """Match the paired parser's exact NFKC/casefold alias contract."""

    return unicodedata.normalize("NFKC", value).casefold()


def _decimal_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return False


def _paired_v4_direct_exact_ic50(
    raw: Mapping[str, Any],
    measurement: ActivityMeasurement,
) -> bool:
    """Apply the pre-registered direct/exact gate before any PubMed query."""

    return bool(
        measurement.endpoint == "IC50"
        and measurement.relation == "="
        and _assay_supports_direct_target_engagement(measurement.assay_type)
        and raw.get("standard_type") == measurement.endpoint
        and raw.get("standard_relation") == measurement.relation
        and raw.get("standard_units") == measurement.unit
        and _decimal_equal(raw.get("standard_value"), measurement.value)
        and str(raw.get("assay_chembl_id") or "").strip()
    )


def _paired_v4_target_aliases(target: TargetCandidate) -> tuple[str, ...]:
    values = (target.symbol, target.target_name)
    by_key: dict[str, str] = {}
    for value in values:
        item = str(value or "").strip()
        key = item.casefold()
        if item and key not in by_key:
            by_key[key] = item
    return tuple(
        sorted(
            by_key.values(),
            key=lambda item: (item.casefold(), item),
        )[:8]
    )


def _paired_v4_query_inventory(
    *,
    target: TargetCandidate,
    selected_ids: Sequence[str],
    activity_rows_by_molecule: Mapping[
        str,
        Sequence[tuple[dict[str, Any], ActivityMeasurement, str]],
    ],
    molecules_by_id: Mapping[str, Mapping[str, Any]],
    aliases_by_id: Mapping[str, Sequence[str]],
    alias_conflicts_by_id: Mapping[str, Sequence[str]],
    frozen_cohort: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, CandidateSearchAliases]]:
    """Freeze the exact bounded input used by the paired-abstract adapter."""

    target_aliases = _paired_v4_target_aliases(target)
    selected_rows = frozen_cohort.get("selected_candidate_rows")
    selected_rows = selected_rows if isinstance(selected_rows, list) else []
    frozen_ids = [
        str(item.get("molecule_chembl_id") or "")
        for item in selected_rows
        if isinstance(item, Mapping)
        and str(item.get("molecule_chembl_id") or "")
    ]
    frozen_id_set = set(frozen_ids)
    records: list[dict[str, Any]] = []
    fixed_pool_alias_records: list[dict[str, Any]] = []
    search_candidates: dict[str, CandidateSearchAliases] = {}
    alias_conflict_ids: list[str] = []
    projected_alias_ids: list[str] = []
    direct_ids: list[str] = []
    for fixed_rank, chembl_id in enumerate(selected_ids, start=1):
        rows = activity_rows_by_molecule.get(chembl_id, ())
        direct_rows = [
            index
            for index, (raw, measurement, _direction_rule) in enumerate(rows)
            if _paired_v4_direct_exact_ic50(raw, measurement)
        ]
        if not direct_rows:
            continue
        direct_ids.append(chembl_id)
        source_aliases = tuple(aliases_by_id.get(chembl_id, ()))
        exact_aliases = source_aliases[:_PRIMARY_PAIRED_V4_MAX_ALIASES]
        if len(source_aliases) > len(exact_aliases):
            projected_alias_ids.append(chembl_id)
        molecule = molecules_by_id.get(chembl_id, {})
        preferred = str(molecule.get("pref_name") or "").strip()
        if preferred not in exact_aliases and exact_aliases:
            preferred = exact_aliases[0]
        conflicts = list(alias_conflicts_by_id.get(chembl_id, ()))
        if conflicts:
            alias_conflict_ids.append(chembl_id)
        record = {
            "fixed_pool_rank": fixed_rank,
            "molecule_chembl_id": chembl_id,
            "preferred_search_alias": preferred or None,
            "exact_aliases": list(exact_aliases),
            "source_alias_count": len(source_aliases),
            "direct_exact_ic50_retained_ranks": [
                index + 1 for index in direct_rows
            ],
            "candidate_alias_conflicts": conflicts,
        }
        fixed_pool_alias_records.append(record)
        if chembl_id not in frozen_id_set:
            continue
        records.append(record)
        if preferred and exact_aliases:
            search_candidates[chembl_id] = CandidateSearchAliases(
                preferred_search_alias=preferred,
                exact_aliases=tuple(exact_aliases),
            )

    queried_ids = sorted(
        frozen_ids,
        key=lambda item: (item.casefold(), item),
    )[:_PRIMARY_PAIRED_V4_MAX_CANDIDATES]
    fixed_pool_alias_complete = bool(
        len(fixed_pool_alias_records) == len(direct_ids)
        and all(
            record.get("preferred_search_alias")
            and record.get("exact_aliases")
            for record in fixed_pool_alias_records
        )
        and not alias_conflict_ids
    )
    queried_records_complete = bool(
        frozen_ids
        and len(records) == len(frozen_ids)
        and set(search_candidates) == frozen_id_set
    )
    complete = fixed_pool_alias_complete and queried_records_complete
    fixed_pool_alias_records_sha256 = _canonical_sha256(
        fixed_pool_alias_records
    )
    body: dict[str, Any] = {
        "schema_version": "primary_abstract_paired_v4_query_inventory_v1",
        "target_aliases": list(target_aliases),
        "endpoint": "IC50",
        "max_candidates": _PRIMARY_PAIRED_V4_MAX_CANDIDATES,
        "max_documents": _PRIMARY_PAIRED_V4_MAX_DOCUMENTS,
        "prewarning_pool_count": len(selected_ids),
        "prewarning_pool_ids": list(selected_ids),
        "paired_v4_direct_exact_pool_count": len(direct_ids),
        "paired_v4_direct_exact_pool_ids": direct_ids,
        "fixed_pool_alias_record_count": len(fixed_pool_alias_records),
        "fixed_pool_alias_records": fixed_pool_alias_records,
        "fixed_pool_alias_records_sha256": fixed_pool_alias_records_sha256,
        "fixed_pool_alias_inventory_complete": fixed_pool_alias_complete,
        "pre_pubmed_frozen_cohort": deepcopy(dict(frozen_cohort)),
        "queried_candidate_count": len(queried_ids),
        "queried_candidate_ids": queried_ids,
        "records_scope": "pre_pubmed_frozen_cohort_query_subset",
        "records": records,
        "candidate_alias_conflict_ids": alias_conflict_ids,
        "bounded_alias_projection_ids": projected_alias_ids,
        "alias_projection_policy": (
            "first_16_exact_chembl_aliases_in_source_order_v1"
        ),
        "complete": complete,
        "truncated": False,
        "fixture": False,
    }
    body["inventory_sha256"] = _canonical_sha256(body)
    selected_candidates = (
        {
            candidate_id: search_candidates[candidate_id]
            for candidate_id in queried_ids
        }
        if complete
        else {}
    )
    return body, selected_candidates


def _paired_v4_search_outcome_payload(
    outcome: PrimaryAbstractSearchOutcome,
) -> dict[str, Any]:
    # Canonical JSON projection makes tuple/list representation explicit for
    # downstream immutable metadata and avoids consumer-specific coercion.
    payload = json.loads(
        json.dumps(
            asdict(outcome),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    payload["outcome_sha256"] = _canonical_sha256(payload)
    return payload


def _paired_v4_not_attempted_outcome(
    *,
    reason_code: str,
    requested_candidate_count: int,
) -> PrimaryAbstractSearchOutcome:
    """Represent a pre-search fail-closed decision with a zero-call ledger."""

    return PrimaryAbstractSearchOutcome(
        schema_version="primary_abstract_search/v1",
        status="not_verified",
        reason_code=reason_code,
        query=None,
        query_sha256=None,
        max_candidates=_PRIMARY_PAIRED_V4_MAX_CANDIDATES,
        max_documents=_PRIMARY_PAIRED_V4_MAX_DOCUMENTS,
        requested_candidate_count=requested_candidate_count,
        queried_candidate_count=0,
        requested_document_count=_PRIMARY_PAIRED_V4_MAX_DOCUMENTS,
        returned_document_count=0,
        unique_document_count=0,
        screened_document_count=0,
        screened_sentence_count=0,
        verified_candidate_ids=(),
        calls=(),
        records=(),
        groups=(),
    )


def _paired_v4_span_matches(sentence: str, span: Any) -> bool:
    start = getattr(span, "start", None)
    end = getattr(span, "end", None)
    text = getattr(span, "text", None)
    return bool(
        type(start) is int
        and type(end) is int
        and 0 <= start <= end <= len(sentence)
        and isinstance(text, str)
        and sentence[start:end] == text
    )


def _paired_v4_valid_group(
    group: PrimaryAbstractGroup,
    *,
    query_sha256: str,
    inventory: Mapping[str, Any],
    publication_types_by_document: Mapping[str, tuple[str, ...]],
) -> bool:
    records = inventory.get("records")
    if not isinstance(records, list):
        return False
    aliases_by_id = {
        str(record.get("molecule_chembl_id") or ""): tuple(
            _paired_v4_alias_key(str(value))
            for value in record.get("exact_aliases", [])
        )
        for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("exact_aliases"), list)
    }
    target_aliases = tuple(
        _paired_v4_alias_key(str(value))
        for value in inventory.get("target_aliases", [])
    )
    alias_owners: dict[str, set[str]] = {}
    for candidate_id, aliases in aliases_by_id.items():
        if not candidate_id or not aliases or len(set(aliases)) != len(aliases):
            return False
        for alias in aliases:
            if not alias:
                return False
            alias_owners.setdefault(alias, set()).add(candidate_id)
    target_alias_keys = set(target_aliases)
    if (
        not target_aliases
        or len(target_alias_keys) != len(target_aliases)
        or any(len(owners) != 1 for owners in alias_owners.values())
        or target_alias_keys.intersection(alias_owners)
    ):
        return False
    endpoint_key = _paired_v4_alias_key(str(inventory.get("endpoint") or ""))
    if (
        not group.document_id.isdigit()
        or len(group.candidate_ids) < 3
        or len(group.candidate_ids) != len(group.pairs)
        or tuple(pair.candidate_id for pair in group.pairs)
        != group.candidate_ids
        or len(set(group.candidate_ids)) != len(group.candidate_ids)
        or group.sentence_sha256
        != sha256(group.sentence.encode("utf-8")).hexdigest()
    ):
        return False
    expected_group_hash = _canonical_sha256(
        {
            "candidate_ids": group.candidate_ids,
            "document_id": group.document_id,
            "pair_evidence_sha256": [
                pair.evidence_sha256 for pair in group.pairs
            ],
            "query_sha256": query_sha256,
            "schema": "primary_abstract_search/v1",
            "section": group.section,
            "sentence_sha256": group.sentence_sha256,
        }
    )
    if expected_group_hash != group.group_sha256:
        return False
    publication_types = publication_types_by_document.get(group.document_id)
    if publication_types is None:
        return False
    first_condition: Any = None
    for pair in group.pairs:
        aliases = aliases_by_id.get(pair.candidate_id, ())
        if (
            _paired_v4_alias_key(pair.candidate_alias) not in aliases
            or _paired_v4_alias_key(pair.target_alias) not in target_alias_keys
            or _paired_v4_alias_key(pair.endpoint) != endpoint_key
            or pair.relation_normalized != "="
            or pair.relation_basis != "direct_assertive_measurement"
            or pair.section != group.section
            or pair.publication_types != publication_types
            or pair.sentence_sha256 != group.sentence_sha256
            or pair.sentence_span.text != group.sentence
            or pair.sentence_span.start != 0
            or pair.sentence_span.end != len(group.sentence)
        ):
            return False
        spans = (
            pair.sentence_span,
            pair.candidate_span,
            pair.target_span,
            pair.endpoint_span,
            pair.measurement_span,
            pair.value_span,
            pair.unit_span,
            pair.assay_condition.span,
            pair.assay_condition.value_span,
            pair.assay_condition.unit_span,
            pair.assay_condition.subject_span,
        )
        if pair.relation_span is not None:
            spans = (*spans, pair.relation_span)
        if not all(_paired_v4_span_matches(group.sentence, span) for span in spans):
            return False
        condition = asdict(pair.assay_condition)
        if first_condition is None:
            first_condition = condition
        elif condition != first_condition:
            return False
    return True


def _paired_v4_condition_field(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    string_only: bool,
) -> tuple[str | None, list[str], str | None]:
    observed: list[tuple[str, str]] = []
    for key in keys:
        if key not in record or record.get(key) is None:
            continue
        raw = record.get(key)
        if isinstance(raw, bool):
            return None, [], "BOOLEAN_CONDITION_FIELD"
        if string_only and not isinstance(raw, str):
            return None, [], "NON_STRING_CONDITION_FIELD"
        value = str(raw)
        if (
            not value
            or value != value.strip()
            or "\r" in value
            or "\n" in value
        ):
            return None, [], "CONDITION_FIELD_REQUIRES_REPAIR"
        observed.append((key, value))
    if not observed:
        return None, [], "CONDITION_FIELD_MISSING"
    values = {value for _key, value in observed}
    if len(values) != 1:
        return None, [key for key, _value in observed], "CONDITION_FIELD_CONFLICT"
    if not string_only:
        try:
            numeric = Decimal(observed[0][1])
        except InvalidOperation:
            return None, [key for key, _value in observed], "CONDITION_VALUE_INVALID"
        if not numeric.is_finite() or numeric <= 0:
            return None, [key for key, _value in observed], "CONDITION_VALUE_INVALID"
    return observed[0][1], [key for key, _value in observed], None


def _paired_v4_list_source_inventory(
    *,
    source: str,
    container: Mapping[str, Any] | None,
    field: str,
) -> tuple[dict[str, Any], list[Any]]:
    """Attest the source list shape before any bounded row projection."""

    field_present = isinstance(container, Mapping) and field in container
    observed = container.get(field) if field_present else None
    list_typed = isinstance(observed, list)
    rows = observed if list_typed else []
    non_mapping_indices = [
        index
        for index, item in enumerate(rows)
        if not isinstance(item, Mapping)
    ]
    body = {
        "schema_version": "exact_assay_condition_list_source_v1",
        "source": source,
        "field": field,
        "field_present": field_present,
        "list_typed": list_typed,
        "observed_type": type(observed).__name__ if field_present else None,
        "row_count": len(rows) if list_typed else None,
        "mapping_row_count": (
            sum(isinstance(item, Mapping) for item in rows)
            if list_typed
            else None
        ),
        "non_mapping_row_indices": non_mapping_indices,
        "rows_sha256": _canonical_sha256(rows) if list_typed else None,
        "observed_value_sha256": (
            _canonical_sha256(observed) if field_present else None
        ),
        "complete": bool(
            field_present and list_typed and not non_mapping_indices
        ),
    }
    return {**body, "inventory_sha256": _canonical_sha256(body)}, rows


def _paired_v4_valid_list_source_inventory(
    value: Any,
    *,
    source: str,
    field: str,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    body = {
        key: deepcopy(item)
        for key, item in value.items()
        if key != "inventory_sha256"
    }
    return bool(
        value.get("schema_version")
        == "exact_assay_condition_list_source_v1"
        and value.get("source") == source
        and value.get("field") == field
        and value.get("inventory_sha256") == _canonical_sha256(body)
    )


def _paired_v4_exact_condition_binding(
    *,
    raw: Mapping[str, Any],
    assay_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Freeze one exact common assay condition before PubMed is queried."""

    context_record = (
        assay_context.get("record")
        if isinstance(assay_context, Mapping)
        else None
    )
    invalid_records: list[dict[str, Any]] = []
    activity_inventory, property_rows = _paired_v4_list_source_inventory(
        source="activity_properties",
        container=raw,
        field="activity_properties",
    )
    parameter_source = "chembl_assay_context.record.assay_parameters"
    parameter_attestation = (
        context_record.get("assay_parameters_source_inventory")
        if isinstance(context_record, Mapping)
        else None
    )
    parameter_attestation_valid = _paired_v4_valid_list_source_inventory(
        parameter_attestation,
        source=parameter_source,
        field="assay_parameters",
    )
    if parameter_attestation_valid:
        parameter_inventory = deepcopy(dict(parameter_attestation))
    else:
        parameter_inventory, _unused_rows = (
            _paired_v4_list_source_inventory(
                source=parameter_source,
                container=None,
                field="assay_parameters",
            )
        )
        invalid_records.append(
            {
                "source": parameter_source,
                "source_index": None,
                "reason_codes": [
                    "CONDITION_SOURCE_INVENTORY_MISSING_OR_INVALID"
                ],
                "present_keys": [],
            }
        )
    parameter_rows = (
        context_record.get("assay_parameters")
        if isinstance(context_record, Mapping)
        and isinstance(context_record.get("assay_parameters"), list)
        else []
    )
    source_inventory = [activity_inventory, parameter_inventory]
    sources: list[tuple[str, list[Any]]] = [
        ("activity_properties", property_rows),
        (parameter_source, parameter_rows),
    ]
    for inventory_record in source_inventory:
        source = str(inventory_record.get("source") or "")
        field = str(inventory_record.get("field") or "")
        if inventory_record.get("field_present") is not True:
            invalid_records.append(
                {
                    "source": source,
                    "source_index": None,
                    "reason_codes": ["CONDITION_SOURCE_FIELD_MISSING"],
                    "present_keys": [],
                }
            )
        elif inventory_record.get("list_typed") is not True:
            invalid_records.append(
                {
                    "source": source,
                    "source_index": None,
                    "reason_codes": ["CONDITION_SOURCE_FIELD_NOT_LIST"],
                    "present_keys": [field],
                }
            )
        else:
            invalid_records.extend(
                {
                    "source": source,
                    "source_index": index,
                    "reason_codes": ["CONDITION_SOURCE_ROW_NOT_OBJECT"],
                    "present_keys": [],
                }
                for index in inventory_record.get(
                    "non_mapping_row_indices", []
                )
            )
    if (
        parameter_attestation_valid
        and parameter_inventory.get("list_typed") is True
        and parameter_inventory.get("row_count") != len(parameter_rows)
    ):
        invalid_records.append(
            {
                "source": parameter_source,
                "source_index": None,
                "reason_codes": ["CONDITION_SOURCE_PROJECTION_COUNT_MISMATCH"],
                "present_keys": ["assay_parameters"],
            }
        )
    relevant_keys = {
        "type",
        "standard_type",
        "value",
        "standard_value",
        "units",
        "standard_units",
    }
    exact_records: list[dict[str, Any]] = []
    if (
        isinstance(context_record, Mapping)
        and context_record.get("assay_parameters_truncated") is not False
    ):
        invalid_records.append(
            {
                "source": "chembl_assay_context.record.assay_parameters",
                "source_index": None,
                "reason_codes": ["ASSAY_PARAMETERS_COMPLETENESS_UNATTESTED"],
                "present_keys": [],
            }
        )
    for source, rows in sources:
        for index, item in enumerate(rows):
            if not isinstance(item, Mapping):
                continue
            subject, subject_fields, subject_error = _paired_v4_condition_field(
                item,
                ("standard_type", "type"),
                string_only=True,
            )
            value, value_fields, value_error = _paired_v4_condition_field(
                item,
                ("standard_value", "value"),
                string_only=False,
            )
            unit, unit_fields, unit_error = _paired_v4_condition_field(
                item,
                ("standard_units", "units"),
                string_only=True,
            )
            errors = [
                error
                for error in (subject_error, value_error, unit_error)
                if error is not None
            ]
            if errors:
                invalid_records.append(
                    {
                        "source": source,
                        "source_index": index,
                        "reason_codes": errors,
                        "present_keys": sorted(
                            relevant_keys.intersection(item)
                        ),
                    }
                )
                continue
            exact_records.append(
                {
                    "source": source,
                    "source_index": index,
                    "subject": subject,
                    "value_raw": value,
                    "unit_raw": unit,
                    "subject_fields": subject_fields,
                    "value_fields": value_fields,
                    "unit_fields": unit_fields,
                }
            )

    unique_conditions = sorted(
        {
            (
                str(record["subject"]),
                str(record["value_raw"]),
                str(record["unit_raw"]),
            )
            for record in exact_records
        }
    )
    if invalid_records:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_RECORD_INCOMPLETE_OR_CONFLICTING"
    elif not unique_conditions:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_MISSING"
    elif len(unique_conditions) != 1:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_AMBIGUOUS_OR_CONFLICTING"
    else:
        status = "exact"
        reason_code = None
    selected = unique_conditions[0] if status == "exact" else None
    body: dict[str, Any] = {
        "schema_version": "exact_assay_condition_binding_v1",
        "status": status,
        "reason_code": reason_code,
        "source_inventory_schema_version": (
            "exact_assay_condition_source_inventory_v1"
        ),
        "source_inventory_count": len(source_inventory),
        "source_inventory": source_inventory,
        "source_inventory_complete": all(
            item["complete"] is True for item in source_inventory
        ),
        "source_inventory_sha256": _canonical_sha256(source_inventory),
        "subject": selected[0] if selected is not None else None,
        "value_raw": selected[1] if selected is not None else None,
        "unit_raw": selected[2] if selected is not None else None,
        "unique_condition_count": len(unique_conditions),
        "unique_conditions": [
            {"subject": subject, "value_raw": value, "unit_raw": unit}
            for subject, value, unit in unique_conditions
        ],
        "source_record_count": len(exact_records),
        "source_records": exact_records,
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records,
        "normalization_applied": False,
        "case_repair_applied": False,
        "whitespace_repair_applied": False,
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
    }
    body["binding_sha256"] = _canonical_sha256(body)
    return body


def _paired_v4_assay_signature(
    *,
    target: TargetCandidate,
    target_chembl_id: str,
    raw: Mapping[str, Any],
    measurement: ActivityMeasurement,
    assay_context: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not _paired_v4_direct_exact_ic50(raw, measurement):
        return None
    if not isinstance(assay_context, Mapping):
        return None
    record = assay_context.get("record")
    assay_id = str(raw.get("assay_chembl_id") or "").strip()
    if (
        assay_context.get("schema_version") != CHEMBL_ASSAY_CONTEXT_SCHEMA
        or assay_context.get("retrieval_status") != "completed"
        or assay_context.get("binding_status") != "exact"
        or assay_context.get("assay_chembl_id") != assay_id
        or assay_context.get("missing_values_imputed") is not False
        or assay_context.get("inference_used") is not False
        or not isinstance(record, Mapping)
        or record.get("assay_chembl_id") != assay_id
        or record.get("target_chembl_id") != target_chembl_id
    ):
        return None
    condition_binding = _paired_v4_exact_condition_binding(
        raw=raw,
        assay_context=assay_context,
    )
    if condition_binding.get("status") != "exact":
        return None
    properties = (
        raw.get("activity_properties")
        if isinstance(raw.get("activity_properties"), list)
        else []
    )
    body = {
        "assay_chembl_id": assay_id,
        "assay_type": measurement.assay_type,
        "assay_record_type": record.get("assay_type"),
        "target_chembl_id": target_chembl_id,
        "target_accession": target.uniprot_id,
        "endpoint": measurement.endpoint,
        "relation": measurement.relation,
        "unit": measurement.unit,
        "activity_properties_sha256": _canonical_sha256(properties),
        "assay_condition_binding": condition_binding,
    }
    return {**body, "signature_sha256": _canonical_sha256(body)}


def _paired_v4_pair_condition_matches_signature(
    pair: PrimaryAbstractPair,
    signature: Mapping[str, Any],
) -> bool:
    binding = signature.get("assay_condition_binding")
    if not isinstance(binding, Mapping):
        return False
    body = {
        key: deepcopy(value)
        for key, value in binding.items()
        if key != "binding_sha256"
    }
    source_inventory = binding.get("source_inventory")
    expected_sources = (
        ("activity_properties", "activity_properties"),
        (
            "chembl_assay_context.record.assay_parameters",
            "assay_parameters",
        ),
    )
    source_inventory_valid = bool(
        isinstance(source_inventory, list)
        and len(source_inventory) == len(expected_sources)
        and binding.get("source_inventory_count") == len(expected_sources)
        and binding.get("source_inventory_complete") is True
        and binding.get("source_inventory_sha256")
        == _canonical_sha256(source_inventory)
        and all(
            item.get("complete") is True
            and _paired_v4_valid_list_source_inventory(
                item,
                source=source,
                field=field,
            )
            for item, (source, field) in zip(
                source_inventory, expected_sources, strict=True
            )
            if isinstance(item, Mapping)
        )
        and all(isinstance(item, Mapping) for item in source_inventory)
    )
    condition = pair.assay_condition
    return bool(
        binding.get("schema_version") == "exact_assay_condition_binding_v1"
        and binding.get("status") == "exact"
        and binding.get("reason_code") is None
        and source_inventory_valid
        and binding.get("unique_condition_count") == 1
        and binding.get("normalization_applied") is False
        and binding.get("case_repair_applied") is False
        and binding.get("whitespace_repair_applied") is False
        and binding.get("unit_conversion_applied") is False
        and binding.get("missing_values_imputed") is False
        and binding.get("binding_sha256") == _canonical_sha256(body)
        and binding.get("subject") == condition.subject
        and binding.get("value_raw") == condition.value_raw
        and binding.get("unit_raw") == condition.unit_raw
    )


def _paired_v4_freeze_assay_cohort(
    *,
    target: TargetCandidate,
    target_chembl_id: str,
    selected_ids: Sequence[str],
    activity_rows_by_molecule: Mapping[
        str,
        Sequence[tuple[dict[str, Any], ActivityMeasurement, str]],
    ],
    assay_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose one assay cohort before PubMed by a fixed coverage rule."""

    by_signature: dict[str, dict[str, Any]] = {}
    bounded_rows: list[dict[str, Any]] = []
    fixed_rank_by_id = {
        chembl_id: rank for rank, chembl_id in enumerate(selected_ids, start=1)
    }
    for chembl_id in selected_ids:
        rows = activity_rows_by_molecule.get(chembl_id, ())
        for retained_index, (raw, measurement, _direction_rule) in enumerate(rows):
            activity_id = str(raw.get("activity_id") or "").strip()
            assay_id = str(raw.get("assay_chembl_id") or "").strip()
            assay_context = assay_contexts.get(assay_id)
            condition_binding = _paired_v4_exact_condition_binding(
                raw=raw,
                assay_context=assay_context,
            )
            signature = _paired_v4_assay_signature(
                target=target,
                target_chembl_id=target_chembl_id,
                raw=raw,
                measurement=measurement,
                assay_context=assay_context,
            )
            bounded_rows.append(
                {
                    "molecule_chembl_id": chembl_id,
                    "fixed_pool_rank": fixed_rank_by_id[chembl_id],
                    "retained_rank": retained_index + 1,
                    "activity_id": activity_id or None,
                    "source_evidence_id": measurement.source_evidence_id,
                    "standard_type": raw.get("standard_type"),
                    "standard_relation": raw.get("standard_relation"),
                    "standard_value": raw.get("standard_value"),
                    "standard_units": raw.get("standard_units"),
                    "typed_endpoint": measurement.endpoint,
                    "typed_relation": measurement.relation,
                    "typed_value": measurement.value,
                    "typed_unit": measurement.unit,
                    "assay_chembl_id": assay_id or None,
                    "direct_exact_gate": _paired_v4_direct_exact_ic50(
                        raw, measurement
                    ),
                    "assay_condition_binding": condition_binding,
                    "exact_assay_signature": deepcopy(signature),
                }
            )
            if signature is None:
                continue
            signature_hash = str(signature["signature_sha256"])
            cohort = by_signature.setdefault(
                signature_hash,
                {
                    "selected_assay_signature": deepcopy(signature),
                    "candidate_rows": {},
                },
            )
            candidate_rows = cohort["candidate_rows"]
            existing = candidate_rows.get(chembl_id)
            selection_key = (retained_index + 1, activity_id)
            existing_key = (
                (existing["retained_rank"], existing["activity_id"] or "")
                if isinstance(existing, Mapping)
                else None
            )
            if existing_key is None or selection_key < existing_key:
                candidate_rows[chembl_id] = {
                    "molecule_chembl_id": chembl_id,
                    "fixed_pool_rank": fixed_rank_by_id[chembl_id],
                    "retained_rank": retained_index + 1,
                    "retained_index": retained_index,
                    "activity_id": activity_id or None,
                    "source_evidence_id": measurement.source_evidence_id,
                }

    considered: list[dict[str, Any]] = []
    for signature_hash, item in by_signature.items():
        candidate_rows = item["candidate_rows"]
        ordered_rows = [
            deepcopy(candidate_rows[candidate_id])
            for candidate_id in selected_ids
            if candidate_id in candidate_rows
        ]
        signature = deepcopy(item["selected_assay_signature"])
        considered.append(
            {
                "signature_sha256": signature_hash,
                "assay_chembl_id": signature.get("assay_chembl_id"),
                "selected_assay_signature": signature,
                "distinct_candidate_coverage": len(ordered_rows),
                "candidate_ids": [
                    row["molecule_chembl_id"] for row in ordered_rows
                ],
                "candidate_rows": ordered_rows,
            }
        )
    considered.sort(
        key=lambda item: (
            -int(item["distinct_candidate_coverage"]),
            str(item["signature_sha256"]),
            str(item.get("assay_chembl_id") or ""),
        )
    )
    selected = considered[0] if considered and considered[0][
        "distinct_candidate_coverage"
    ] >= 3 else None
    body: dict[str, Any] = {
        "schema_version": "pre_pubmed_exact_assay_cohort_v1",
        "selection_status": "selected" if selected is not None else "not_selected",
        "reason_code": (
            None
            if selected is not None
            else "NO_PRE_PUBMED_THREE_CANDIDATE_EXACT_ASSAY_COHORT"
        ),
        "selection_policy": (
            "max_distinct_fixed_pool_candidate_coverage_then_"
            "signature_sha256_then_assay_id_v1"
        ),
        "within_candidate_row_policy": (
            "retained_rank_then_activity_id_ascending_v1"
        ),
        "minimum_distinct_candidate_coverage": 3,
        "considered_signature_count": len(considered),
        "considered_signatures": considered,
        "retained_top_k_rows": bounded_rows,
        "selected_signature": (
            deepcopy(selected["selected_assay_signature"])
            if selected is not None
            else None
        ),
        "selected_signature_sha256": (
            selected["signature_sha256"] if selected is not None else None
        ),
        "selected_candidate_count": (
            selected["distinct_candidate_coverage"] if selected is not None else 0
        ),
        "selected_candidate_ids": (
            list(selected["candidate_ids"]) if selected is not None else []
        ),
        "selected_candidate_rows": (
            deepcopy(selected["candidate_rows"]) if selected is not None else []
        ),
        "pubmed_or_result_used_for_cohort_selection": False,
        "no_backfill": True,
        "missing_values_imputed": False,
        "fixture": False,
    }
    body["cohort_attestation_sha256"] = _canonical_sha256(body)
    return body


def _paired_v4_pair_matches_activity(
    pair: PrimaryAbstractPair,
    *,
    chembl_id: str,
    target_chembl_id: str,
    raw: Mapping[str, Any],
    measurement: ActivityMeasurement,
) -> bool:
    return bool(
        pair.candidate_id == chembl_id
        and _paired_v4_alias_key(pair.endpoint)
        == _paired_v4_alias_key(measurement.endpoint)
        and pair.relation_normalized == measurement.relation == "="
        and pair.unit_raw == measurement.unit
        and _decimal_equal(pair.value_raw, measurement.value)
        and raw.get("molecule_chembl_id") == chembl_id
        and raw.get("parent_molecule_chembl_id") == chembl_id
        and raw.get("target_chembl_id") == target_chembl_id
        and raw.get("standard_type") == measurement.endpoint
        and raw.get("standard_relation") == measurement.relation
        and raw.get("standard_units") == measurement.unit
        and _decimal_equal(raw.get("standard_value"), pair.value_raw)
    )


def _paired_v4_promotions(
    *,
    outcome: PrimaryAbstractSearchOutcome,
    inventory: Mapping[str, Any],
    target: TargetCandidate,
    target_chembl_id: str,
    activity_rows_by_molecule: Mapping[
        str,
        Sequence[tuple[dict[str, Any], ActivityMeasurement, str]],
    ],
    assay_contexts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Match one verified pair to one retained ChEMBL row without backfill."""

    summary: dict[str, Any] = {
        "policy": _PRIMARY_PAIRED_V4_POLICY,
        "status": "not_verified",
        "reason_code": None,
        "verified_group_count": len(outcome.groups),
        "eligible_same_assay_cohort_count": 0,
        "promoted_candidate_ids": [],
        "cross_group_conflict_candidate_ids": [],
        "measurement_mismatch_candidate_ids": [],
        "assay_condition_mismatch_candidate_ids": [],
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
        "no_backfill": True,
        "fixture": False,
    }
    if (
        outcome.schema_version != "primary_abstract_search/v1"
        or outcome.status != "verified"
        or not outcome.query_sha256
        or inventory.get("complete") is not True
        or inventory.get("truncated") is not False
        or inventory.get("candidate_alias_conflict_ids")
    ):
        summary["reason_code"] = (
            "CANDIDATE_ALIAS_CONFLICT"
            if inventory.get("candidate_alias_conflict_ids")
            else outcome.reason_code or "PRIMARY_ABSTRACT_SEARCH_NOT_VERIFIED"
        )
        return {}, summary

    publication_types_by_document = {
        record.document_id: record.publication_types
        for record in outcome.records
        if record.status == "verified"
    }
    valid_groups = [
        group
        for group in outcome.groups
        if _paired_v4_valid_group(
            group,
            query_sha256=outcome.query_sha256,
            inventory=inventory,
            publication_types_by_document=publication_types_by_document,
        )
    ]
    if len(valid_groups) != len(outcome.groups) or not valid_groups:
        summary["reason_code"] = "PRIMARY_ABSTRACT_GROUP_INTEGRITY_MISMATCH"
        return {}, summary

    pair_signatures_by_id: dict[str, set[tuple[str, Decimal, str, str]]] = {}
    for group in valid_groups:
        for pair in group.pairs:
            try:
                value = Decimal(pair.value_raw)
            except InvalidOperation:
                summary["reason_code"] = "PRIMARY_ABSTRACT_VALUE_INVALID"
                return {}, summary
            pair_signatures_by_id.setdefault(pair.candidate_id, set()).add(
                (
                    pair.endpoint,
                    value,
                    pair.unit_raw,
                    pair.relation_normalized,
                )
            )
    conflicts = sorted(
        candidate_id
        for candidate_id, signatures in pair_signatures_by_id.items()
        if len(signatures) != 1
    )
    if conflicts:
        summary["reason_code"] = "CROSS_GROUP_VALUE_OR_UNIT_MISMATCH"
        summary["cross_group_conflict_candidate_ids"] = conflicts
        return {}, summary

    frozen = inventory.get("pre_pubmed_frozen_cohort")
    if not isinstance(frozen, Mapping):
        summary["reason_code"] = "PRE_PUBMED_COHORT_PROVENANCE_MISSING"
        return {}, summary
    frozen_body = {
        key: deepcopy(value)
        for key, value in frozen.items()
        if key != "cohort_attestation_sha256"
    }
    if (
        frozen.get("selection_status") != "selected"
        or frozen.get("pubmed_or_result_used_for_cohort_selection") is not False
        or frozen.get("cohort_attestation_sha256")
        != _canonical_sha256(frozen_body)
        or not isinstance(frozen.get("selected_signature"), Mapping)
        or not isinstance(frozen.get("selected_candidate_rows"), list)
    ):
        summary["reason_code"] = "PRE_PUBMED_COHORT_INTEGRITY_MISMATCH"
        return {}, summary
    frozen_signature = dict(frozen["selected_signature"])
    frozen_rows = {
        str(item.get("molecule_chembl_id") or ""): item
        for item in frozen["selected_candidate_rows"]
        if isinstance(item, Mapping)
        and str(item.get("molecule_chembl_id") or "")
    }

    options_by_id: dict[str, list[dict[str, Any]]] = {}
    cohorts: list[dict[str, Any]] = []
    measurement_mismatch_ids: set[str] = set()
    condition_mismatch_ids: set[str] = set()
    for group in valid_groups:
        candidate_matches: dict[str, dict[str, Any]] = {}
        for pair in group.pairs:
            selected_row = frozen_rows.get(pair.candidate_id)
            if not isinstance(selected_row, Mapping):
                continue
            retained_index = selected_row.get("retained_index")
            rows = activity_rows_by_molecule.get(pair.candidate_id, ())
            if type(retained_index) is not int or not 0 <= retained_index < len(rows):
                continue
            raw, measurement, _direction_rule = rows[retained_index]
            if str(raw.get("activity_id") or "").strip() != str(
                selected_row.get("activity_id") or ""
            ).strip():
                continue
            if not _paired_v4_pair_matches_activity(
                pair,
                chembl_id=pair.candidate_id,
                target_chembl_id=target_chembl_id,
                raw=raw,
                measurement=measurement,
            ):
                measurement_mismatch_ids.add(pair.candidate_id)
                continue
            assay_id = str(raw.get("assay_chembl_id") or "").strip()
            signature = _paired_v4_assay_signature(
                target=target,
                target_chembl_id=target_chembl_id,
                raw=raw,
                measurement=measurement,
                assay_context=assay_contexts.get(assay_id),
            )
            if signature != frozen_signature:
                measurement_mismatch_ids.add(pair.candidate_id)
                continue
            if not _paired_v4_pair_condition_matches_signature(
                pair, signature
            ):
                condition_mismatch_ids.add(pair.candidate_id)
                continue
            candidate_matches[pair.candidate_id] = {
                "group": group,
                "pair": pair,
                "retained_index": retained_index,
                "assay_signature": signature,
            }
        if len(candidate_matches) < 3:
            continue
        cohort_ids = tuple(sorted(candidate_matches))
        cohorts.append(
            {
                "group_sha256": group.group_sha256,
                "signature_sha256": frozen_signature["signature_sha256"],
                "candidate_ids": list(cohort_ids),
                "pre_pubmed_frozen_signature": True,
            }
        )
        for candidate_id, option in candidate_matches.items():
            options_by_id.setdefault(candidate_id, []).append(
                {**option, "cohort_candidate_ids": cohort_ids}
            )

    if not options_by_id:
        summary["measurement_mismatch_candidate_ids"] = sorted(
            measurement_mismatch_ids
        )
        summary["assay_condition_mismatch_candidate_ids"] = sorted(
            condition_mismatch_ids
        )
        summary["reason_code"] = (
            "PRIMARY_ABSTRACT_ASSAY_CONDITION_MISMATCH"
            if condition_mismatch_ids
            else "NO_THREE_CANDIDATE_EXACT_ASSAY_MATCH"
        )
        return {}, summary

    promotions: dict[str, dict[str, Any]] = {}
    for candidate_id, options in options_by_id.items():
        options.sort(
            key=lambda item: (
                int(item["group"].document_id),
                item["group"].group_sha256,
                item["assay_signature"]["signature_sha256"],
                item["retained_index"],
            )
        )
        promotions[candidate_id] = options[0]
    summary.update(
        {
            "status": "verified",
            "reason_code": None,
            "eligible_same_assay_cohort_count": len(cohorts),
            "eligible_same_assay_cohorts": cohorts,
            "promoted_candidate_ids": sorted(promotions),
            "measurement_mismatch_candidate_ids": sorted(
                measurement_mismatch_ids
            ),
            "assay_condition_mismatch_candidate_ids": sorted(
                condition_mismatch_ids
            ),
        }
    )
    summary["promotion_sha256"] = _canonical_sha256(summary)
    return promotions, summary


def _paired_v4_evidence_envelope(
    *,
    chembl_id: str,
    target: TargetCandidate,
    target_chembl_id: str,
    option: Mapping[str, Any],
    inventory: Mapping[str, Any],
    outcome_payload: Mapping[str, Any],
    promotion_summary: Mapping[str, Any],
    raw: Mapping[str, Any],
    measurement: ActivityMeasurement,
    activity_id: str,
    activity_call: HTTPCallProvenance,
    assay_context: Mapping[str, Any],
) -> dict[str, Any]:
    group = option.get("group")
    pair = option.get("pair")
    signature = option.get("assay_signature")
    retained_index = option.get("retained_index")
    if (
        not isinstance(group, PrimaryAbstractGroup)
        or not isinstance(pair, PrimaryAbstractPair)
        or not isinstance(signature, Mapping)
        or type(retained_index) is not int
    ):
        raise ValueError("invalid paired-v4 promotion option")
    selected_measurement = {
        "activity_id": activity_id,
        "source_evidence_id": measurement.source_evidence_id,
        "retained_rank": retained_index + 1,
        "molecule_chembl_id": chembl_id,
        "target_chembl_id": target_chembl_id,
        "endpoint": measurement.endpoint,
        "relation": measurement.relation,
        "value": measurement.value,
        "unit": measurement.unit,
        "assay_type": measurement.assay_type,
        "deposited_standard_type": raw.get("standard_type"),
        "deposited_standard_relation": raw.get("standard_relation"),
        "deposited_standard_value": raw.get("standard_value"),
        "deposited_standard_units": raw.get("standard_units"),
        "pair_value_raw": pair.value_raw,
        "pair_unit_raw": pair.unit_raw,
        "pair_relation_normalized": pair.relation_normalized,
        "pair_assay_condition": asdict(pair.assay_condition),
        "exact_decimal_value_match": _decimal_equal(
            raw.get("standard_value"), pair.value_raw
        ),
        "exact_unit_match": raw.get("standard_units") == pair.unit_raw,
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
    }
    assay_query = assay_context.get("assay_query")
    tool_call_link = {
        "operation": "retrieve_repurposing_primary_abstract_paired_v4",
        "tool_call_record_linkage_status": (
            "pending_orchestrator_projection_gateway_has_no_packet_call_id"
        ),
        "primary_search_input_sha256": inventory.get("inventory_sha256"),
        "primary_search_output_sha256": outcome_payload.get(
            "outcome_sha256"
        ),
        "primary_search_http_call_count": len(
            outcome_payload.get("calls", [])
            if isinstance(outcome_payload.get("calls"), (list, tuple))
            else []
        ),
        "primary_search_http_calls": deepcopy(
            outcome_payload.get("calls", [])
        ),
        "chembl_activity_query": activity_call.as_dict(),
        "chembl_assay_query": deepcopy(assay_query),
        "invented_tool_call_id": False,
    }
    group_payload = json.loads(
        json.dumps(
            asdict(group),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    envelope: dict[str, Any] = {
        "schema_version": _PRIMARY_PAIRED_V4_SCHEMA,
        "verification_status": "verified",
        "is_primary_support": True,
        "reason_code": None,
        "policy": _PRIMARY_PAIRED_V4_POLICY,
        "source": "PubMed",
        "pmid": group.document_id,
        "source_url": (
            f"https://pubmed.ncbi.nlm.nih.gov/{quote(group.document_id)}/"
        ),
        "publication_types": list(pair.publication_types),
        "section": group.section,
        "full_text_stored": False,
        "hash_algorithm": "sha256",
        "molecule_chembl_id": chembl_id,
        "target_binding": {
            "target_symbol": target.symbol,
            "target_accession": target.uniprot_id,
            "target_chembl_id": target_chembl_id,
            "target_alias": pair.target_alias,
            "target_span": asdict(pair.target_span),
        },
        "alias_inventory": deepcopy(dict(inventory)),
        "search_outcome": deepcopy(dict(outcome_payload)),
        "group": group_payload,
        "selected_pair_index": pair.pair_index,
        "selected_pair_evidence_sha256": pair.evidence_sha256,
        "selected_measurement": selected_measurement,
        "selected_assay_signature": deepcopy(dict(signature)),
        "selected_assay_condition_binding": deepcopy(
            signature.get("assay_condition_binding")
        ),
        "same_assay_cohort_candidate_ids": list(
            option.get("cohort_candidate_ids", ())
        ),
        "promotion_summary": deepcopy(dict(promotion_summary)),
        "tool_call_link": tool_call_link,
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
        "pubmed_or_result_used_for_cohort_selection": False,
        "fixture": False,
    }
    envelope["attestation_sha256"] = _canonical_sha256(envelope)
    return envelope


def _target_exact_phrases(target: TargetCandidate) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for value in (
        target.symbol,
        target.uniprot_id,
        target.target_name,
        target.metadata.get("chembl_target_id"),
    ):
        normalized = _normalize_identity_phrase(value)
        if normalized and normalized not in seen:
            phrases.append(str(value))
            seen.add(normalized)
    return phrases


def _endpoint_specific_phrases(endpoint: str) -> list[str]:
    normalized = _normalize_identity_phrase(endpoint)
    if not normalized:
        return []
    return sorted(_ENDPOINT_SPECIFIC_TERMS.get(normalized, frozenset({normalized})))


def _matched_exact_activity_values(
    text: str,
    measurement: ActivityMeasurement,
) -> list[str]:
    """Match the bound numeric value and unit without conversion or inference."""

    return [
        str(item["text"])
        for item in _exact_activity_value_spans(text, measurement)
    ]


def _exact_phrase_spans(
    text: str,
    phrases: Sequence[str],
) -> list[dict[str, Any]]:
    """Return exact normalized phrase spans while retaining source offsets."""

    spans: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for phrase in phrases:
        normalized = _normalize_identity_phrase(phrase)
        tokens = re.findall(r"[\w]+", str(phrase), flags=re.UNICODE)
        if not normalized or not tokens:
            continue
        pattern = re.compile(
            r"(?<!\w)" + r"[^\w]+".join(re.escape(token) for token in tokens) + r"(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        )
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), normalized)
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                {
                    "normalized_phrase": normalized,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    spans.sort(
        key=lambda item: (
            int(item["start"]),
            int(item["end"]),
            str(item["normalized_phrase"]),
        )
    )
    return spans


_UNSAFE_PRIMARY_CLAUSE = re.compile(
    r"\b(?:previously|formerly|earlier|prior)\s+(?:was\s+|were\s+)?reported\b|"
    r"\b(?:was|were)\s+(?:previously|formerly|earlier)\s+reported\b|"
    r"\b(?:according\s+to|as\s+reported|previous\s+stud(?:y|ies)|prior\s+work)\b|"
    r"\b(?:may|might|could|possibly|potentially|apparently|appears?|suggests?|suggested)\b|"
    # Keep this scoped context distinction aligned with the independent MA3
    # and paper-native v6 validators; neither exception grants potency proof.
    r"\b(?:no|not|neither|without(?!\s+(?:(?:detectable|significant|measurable|overt)\s+)?"
    r"(?:cytotoxicity|toxicity)\s*(?:[.,;:!?)]|$))|failed\s+to|fails\s+to|did\s+not)\b|"
    r"\b(?:predict(?:ed|ing|ion|ions|ive|ively)?|"
    r"(?<!cell )(?<!cellular )(?<!animal )(?<!murine )(?<!xenograft )(?<!organoid )(?<!vitro )(?<!vivo )models?|"
    r"model(?:ed|ing|led|ling)|"
    r"simulat(?:e|es|ed|ing|ion|ions|ive|ively)|"
    r"calculat(?:e|es|ed|ing|ion|ions)|computational(?:ly)?|"
    r"in[\s-]+silico|prospective(?:ly)?|hypothetical(?:ly)?|putative(?:ly)?|"
    r"project(?:ed|ing|ion|ions)|estimat(?:e|es|ed|ing|ion|ions)|"
    r"forecast(?:s|ed|ing)?|anticipated)\b|"
    r"\b(?:planned|expected|proposed|anticipated|aimed)\s+to\s+be\b|"
    r"\b(?:plans?|proposes?|aims?)\s+to\s+be\b|"
    r"\b(?:will|would)(?:\s+\w+){0,2}\s+(?:be|been)\b|"
    r"\b(?:about|approximate(?:ly)?|approx\.?|roughly|around|circa)\s*(?=\d|\.\d)|"
    r"(?:±|\+/-|~|≈|≤|≥)|(?<![\w.])[+-]\s*(?=\d)|"
    r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
    r"(?:[-–]|\bto\b)\s*\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])|"
    r"\b(?:range[sd]?|ranging)\s+from\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b|"
    r"\bbetween\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?\b|"
    r"\b(?:versus|vs\.?|compared\s+(?:with|to)|relative\s+to|"
    r"greater\s+than|less\s+than|higher\s+than|lower\s+than)\b",
    flags=re.IGNORECASE,
)
_UNSAFE_NUMERIC_WORD_PREFIX = re.compile(
    r"(?<!\w)(?:about|approximate(?:ly)?|approx\.?|roughly|around|circa|"
    r"up\s+to|at\s+(?:most|least)|more\s+than|less\s+than|"
    r"greater\s+than|higher\s+than|lower\s+than|above|below|under|over)\s*$",
    flags=re.IGNORECASE,
)
_UNSAFE_NUMERIC_SYMBOL_PREFIX = re.compile(
    r"(?:±|\+/-|~|≈|≤|≥|(?<![\w.])[+-])\s*$"
)


def _primary_clause_semantically_unsafe(text: str) -> bool:
    return _UNSAFE_PRIMARY_CLAUSE.search(
        unicodedata.normalize("NFKC", text)
    ) is not None


def _numeric_match_has_unsafe_prefix(
    text: str, match: re.Match[str], *, number_group: str,
) -> bool:
    before_match = unicodedata.normalize("NFKC", text[:match.start()])
    before_number = unicodedata.normalize(
        "NFKC", text[:match.start(number_group)]
    )
    return bool(
        _UNSAFE_PRIMARY_CLAUSE.search(unicodedata.normalize("NFKC", text))
        or _UNSAFE_NUMERIC_WORD_PREFIX.search(before_match)
        or _UNSAFE_NUMERIC_SYMBOL_PREFIX.search(before_number)
    )


def _exact_activity_value_spans(
    text: str,
    measurement: ActivityMeasurement,
) -> list[dict[str, Any]]:
    """Return exact relation/value/unit spans with no unit canonicalisation."""

    try:
        expected = Decimal(str(measurement.value))
    except InvalidOperation:
        return []
    relation_compatibility = {"=": {"", "="}, "<": {"<"}, "<=": {"<="}, ">": {">"}, ">=": {">="}}
    allowed_relations = relation_compatibility.get(measurement.relation, set())
    if not allowed_relations:
        return []
    unit = re.escape(measurement.unit)
    pattern = re.compile(
        rf"(?<![\w.])(?P<relation><=|>=|<|>|=)?\s*"
        rf"(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
        rf"(?P<unit>{unit})(?!\w)",
    )
    matches: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        if _numeric_match_has_unsafe_prefix(
            text, match, number_group="number"
        ):
            continue
        relation = match.group("relation") or ""
        if relation not in allowed_relations:
            continue
        try:
            reported = Decimal(match.group("number"))
        except InvalidOperation:
            continue
        if reported == expected:
            leading = len(match.group(0)) - len(match.group(0).lstrip())
            trailing = len(match.group(0)) - len(match.group(0).rstrip())
            matches.append(
                {
                    "text": match.group(0).strip(),
                    "start": match.start() + leading,
                    "end": match.end() - trailing,
                    "relation": relation,
                    "number": match.group("number"),
                    "unit": match.group("unit"),
                }
            )
    return matches


def _top_level_parenthetical_spans(text: str) -> list[tuple[int, int]]:
    """Return balanced top-level parenthetical spans, including delimiters."""

    spans: list[tuple[int, int]] = []
    depth = 0
    start: int | None = None
    for index, character in enumerate(text):
        if character == "(":
            if depth == 0:
                start = index
            depth += 1
        elif character == ")":
            if depth == 0:
                return []
            depth -= 1
            if depth == 0:
                assert start is not None
                spans.append((start, index + 1))
                start = None
    return spans if depth == 0 else []


def _primary_parenthetical_binding(
    sentence: str,
    *,
    candidate_aliases: Sequence[str],
    competitor_aliases: Sequence[str],
    target: TargetCandidate,
    measurement: ActivityMeasurement,
) -> dict[str, Any] | None:
    """Bind one exact target/activity parenthetical without list inference.

    Primary abstracts frequently place an individually labelled comparator or
    target measurement inside parentheses while the surrounding sentence also
    contains an unrelated ``respectively`` list.  Rejecting the whole sentence
    discards an otherwise atomic source assertion.  This grammar admits only a
    balanced parenthetical containing one endpoint and one numeric value/unit,
    with explicit candidate/target ownership.  No list position is inferred.
    """

    body = sentence.rstrip(".!?").rstrip()
    if (
        not body
        or len(body) > _PRIMARY_EXCERPT_MAX_CHARS
        or _primary_clause_semantically_unsafe(body)
        or re.search(r"[!?]|(?<!\d)\.(?!\d)", body)
    ):
        return None
    candidate_spans = _exact_phrase_spans(body, candidate_aliases)
    competitor_spans = _exact_phrase_spans(body, competitor_aliases)
    target_spans = _exact_phrase_spans(body, _target_exact_phrases(target))
    endpoint_spans = _exact_phrase_spans(
        body,
        _endpoint_specific_phrases(measurement.endpoint),
    )
    value_spans = _exact_activity_value_spans(body, measurement)
    if (
        not candidate_spans
        or competitor_spans
        or not target_spans
        or not endpoint_spans
        or not value_spans
    ):
        return None

    candidate_coordinates = {
        (int(item["start"]), int(item["end"])) for item in candidate_spans
    }
    target_coordinates = {
        (int(item["start"]), int(item["end"])) for item in target_spans
    }
    if len(candidate_coordinates) != 1:
        return None
    candidate_span = candidate_spans[0]

    all_numeric_unit_spans = list(
        re.finditer(
            r"(?<![\w.])(?:<=|>=|<|>|=)?\s*"
            r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
            r"(?:pM|nM|uM|µM|μM|mM|M)(?!\w)",
            body,
        )
    )
    for group_start, group_end in _top_level_parenthetical_spans(body):
        group_text = body[group_start + 1:group_end - 1]
        if re.search(
            r"(?<!\w)(?:and|or|respectively)(?!\w)",
            group_text,
            flags=re.IGNORECASE,
        ):
            continue
        group_endpoint_spans = [
            item
            for item in endpoint_spans
            if group_start < int(item["start"])
            and int(item["end"]) < group_end
        ]
        group_value_spans = [
            item
            for item in value_spans
            if group_start < int(item["start"])
            and int(item["end"]) < group_end
        ]
        group_numeric_spans = [
            item
            for item in all_numeric_unit_spans
            if group_start < item.start() and item.end() < group_end
        ]
        if not (
            len(
                {
                    (int(item["start"]), int(item["end"]))
                    for item in group_endpoint_spans
                }
            )
            == 1
            and len(
                {
                    (int(item["start"]), int(item["end"]))
                    for item in group_value_spans
                }
            )
            == 1
            and len(group_numeric_spans) == 1
        ):
            continue
        group_target_spans = [
            item
            for item in target_spans
            if group_start < int(item["start"])
            and int(item["end"]) < group_end
        ]
        candidate_immediately_before = (
            int(candidate_span["end"]) <= group_start
            and not body[int(candidate_span["end"]):group_start].strip()
        )
        immediately_preceding_targets = [
            item
            for item in target_spans
            if int(item["end"]) <= group_start
            and not body[int(item["end"]):group_start].strip()
        ]
        target_span: dict[str, Any] | None = None
        pattern: str | None = None
        if (
            candidate_immediately_before
            and len(
                {
                    (int(item["start"]), int(item["end"]))
                    for item in group_target_spans
                }
            )
            == 1
        ):
            pattern = "candidate_parenthetical_target_endpoint_value"
            target_span = group_target_spans[0]
        elif (
            len(
                {
                    (int(item["start"]), int(item["end"]))
                    for item in group_target_spans
                }
            )
            == 1
            and int(candidate_span["end"]) <= group_start
            and re.search(
                r"(?<!\w)(?:inhibit\w*|potent|activity)(?!\w)",
                body[
                    int(candidate_span["end"]):int(
                        group_target_spans[0]["start"]
                    )
                ],
                flags=re.IGNORECASE,
            )
        ):
            target_span = group_target_spans[0]
            pattern = "candidate_subject_parenthetical_target_endpoint_value"
        elif (
            len(
                {
                    (int(item["start"]), int(item["end"]))
                    for item in immediately_preceding_targets
                }
            )
            == 1
            and int(candidate_span["end"])
            <= int(immediately_preceding_targets[0]["start"])
            and re.search(
                r"(?<!\w)(?:inhibit\w*|potent|activity)(?!\w)",
                body[
                    int(candidate_span["end"]):int(
                        immediately_preceding_targets[0]["start"]
                    )
                ],
                flags=re.IGNORECASE,
            )
        ):
            target_span = immediately_preceding_targets[0]
            pattern = "candidate_subject_target_parenthetical_endpoint_value"
        elif (
            candidate_immediately_before
            and len(target_coordinates) == 1
            and (target_span := target_spans[0]) is not None
            and int(target_span["end"]) <= int(candidate_span["start"])
            and re.search(
                r"(?<!\w)(?:positive\s+control|reference|control|"
                r"comparable\s+to(?:\s+that)?\s+of)(?!\w)",
                body[int(target_span["end"]):int(candidate_span["start"])],
                flags=re.IGNORECASE,
            )
        ):
            pattern = "single_target_context_candidate_parenthetical_endpoint_value"
        if pattern is None or target_span is None:
            continue
        endpoint_span = group_endpoint_spans[0]
        value_span = group_value_spans[0]
        return {
            "binding_pattern": pattern,
            "matched_groups": ["candidate", "target", "activity"],
            "matched_terms": {
                "candidate": sorted(
                    {
                        str(item["normalized_phrase"])
                        for item in candidate_spans
                    }
                ),
                "target": sorted(
                    {str(item["normalized_phrase"]) for item in target_spans}
                ),
                "activity": sorted(
                    {
                        str(item["normalized_phrase"])
                        for item in endpoint_spans
                    }
                ),
            },
            "matched_exact_value_unit": [str(value_span["text"])],
            "competitor_alias_matches": [],
            "bound_spans": {
                "candidate": candidate_span,
                "target": target_span,
                "endpoint": endpoint_span,
                "value": value_span,
                "atomic_parenthetical": {
                    "start": group_start,
                    "end": group_end,
                    "text": body[group_start:group_end],
                },
            },
        }
    return None


def _primary_clause_binding(
    clause: str,
    *,
    candidate_aliases: Sequence[str],
    competitor_aliases: Sequence[str],
    target: TargetCandidate,
    measurement: ActivityMeasurement,
) -> dict[str, Any] | None:
    """Require deterministic grammatical ownership of one exact value."""

    clause_body = clause.rstrip(".!?").rstrip()
    if (
        _primary_clause_semantically_unsafe(clause)
        or
        re.search(
            r";|:|\u2014|\u2013|(?<!\w)(?:whereas|while|but)(?!\w)",
            clause_body,
            flags=re.IGNORECASE,
        )
        or re.search(r"[!?]|(?<!\d)\.(?!\d)", clause_body)
        or re.search(
            r"(?<!\w)respectively(?!\w)",
            clause,
            flags=re.IGNORECASE,
        )
    ):
        return None
    candidate_spans = _exact_phrase_spans(clause, candidate_aliases)
    competitor_spans = _exact_phrase_spans(clause, competitor_aliases)
    target_spans = _exact_phrase_spans(clause, _target_exact_phrases(target))
    endpoint_spans = _exact_phrase_spans(
        clause,
        _endpoint_specific_phrases(measurement.endpoint),
    )
    value_spans = _exact_activity_value_spans(clause, measurement)
    numeric_unit_span_count = len(
        re.findall(
            r"(?<![\w.])(?:<=|>=|<|>|=)?\s*"
            r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
            r"(?:pM|nM|uM|µM|μM|mM|M)(?!\w)",
            clause,
        )
    )
    if (
        not candidate_spans
        or competitor_spans
        or not target_spans
        or not endpoint_spans
        or not value_spans
        or numeric_unit_span_count != 1
    ):
        return None
    if (
        len(
            {(int(item["start"]), int(item["end"])) for item in target_spans}
        )
        != 1
        or len(
            {(int(item["start"]), int(item["end"])) for item in endpoint_spans}
        )
        != 1
        or len(
            {(int(item["start"]), int(item["end"])) for item in value_spans}
        )
        != 1
    ):
        return None

    binding_pattern: str | None = None
    bound_spans: dict[str, dict[str, Any]] | None = None
    for candidate_span in candidate_spans:
        for target_span in target_spans:
            for endpoint_span in endpoint_spans:
                for value_span in value_spans:
                    if (
                        int(candidate_span["end"])
                        <= int(target_span["start"])
                        and int(target_span["end"])
                        <= int(endpoint_span["start"])
                        and int(endpoint_span["end"])
                        <= int(value_span["start"])
                    ):
                        binding_pattern = "candidate_target_endpoint_value"
                    elif (
                        int(candidate_span["end"])
                        <= int(endpoint_span["start"])
                        and int(endpoint_span["end"])
                        <= int(value_span["start"])
                        and int(value_span["end"])
                        <= int(target_span["start"])
                        and re.search(
                            r"(?<!\w)(?:against|for)(?!\w)",
                            clause[
                                int(value_span["end"]):int(target_span["start"])
                            ],
                            flags=re.IGNORECASE,
                        )
                    ):
                        binding_pattern = (
                            "candidate_endpoint_value_against_target"
                        )
                    elif (
                        int(endpoint_span["end"])
                        <= int(candidate_span["start"])
                        and int(candidate_span["end"])
                        <= int(target_span["start"])
                        and int(target_span["end"])
                        <= int(value_span["start"])
                        and re.search(
                            r"(?<!\w)(?:of|for)(?!\w)",
                            clause[
                                int(endpoint_span["end"]):int(candidate_span["start"])
                            ],
                            flags=re.IGNORECASE,
                        )
                        and re.search(
                            r"(?<!\w)(?:against|for)(?!\w)",
                            clause[
                                int(candidate_span["end"]):int(target_span["start"])
                            ],
                            flags=re.IGNORECASE,
                        )
                    ):
                        binding_pattern = (
                            "endpoint_of_candidate_against_target_value"
                        )
                    else:
                        continue
                    earliest = min(
                        int(candidate_span["start"]),
                        int(target_span["start"]),
                        int(endpoint_span["start"]),
                        int(value_span["start"]),
                    )
                    latest = max(
                        int(candidate_span["end"]),
                        int(target_span["end"]),
                        int(endpoint_span["end"]),
                        int(value_span["end"]),
                    )
                    if re.search(
                        r"(?<!\w)(?:and|or)(?!\w)",
                        clause[earliest:latest],
                        flags=re.IGNORECASE,
                    ):
                        binding_pattern = None
                        continue
                    bound_spans = {
                        "candidate": candidate_span,
                        "target": target_span,
                        "endpoint": endpoint_span,
                        "value": value_span,
                    }
                    break
                if binding_pattern is not None:
                    break
            if binding_pattern is not None:
                break
        if binding_pattern is not None:
            break
    if binding_pattern is None or bound_spans is None:
        return None
    return {
        "binding_pattern": binding_pattern,
        "matched_groups": ["candidate", "target", "activity"],
        "matched_terms": {
            "candidate": sorted(
                {str(item["normalized_phrase"]) for item in candidate_spans}
            ),
            "target": sorted(
                {str(item["normalized_phrase"]) for item in target_spans}
            ),
            "activity": sorted(
                {str(item["normalized_phrase"]) for item in endpoint_spans}
            ),
        },
        "matched_exact_value_unit": [
            str(item["text"]) for item in value_spans
        ],
        "competitor_alias_matches": [],
        "bound_spans": bound_spans,
    }


def _select_primary_excerpt(
    abstract: Any,
    *,
    candidate_aliases: Sequence[str],
    competitor_aliases: Sequence[str],
    target: TargetCandidate,
    measurement: ActivityMeasurement,
) -> dict[str, Any] | None:
    """Select one exact, bounded abstract sentence with explicit relevance.

    ChEMBL's exact activity-to-document foreign key establishes document
    lineage.  The excerpt adds a human-auditable passage, but it is admitted
    only when one deterministic clause binds the current candidate, target,
    endpoint, and exact relation/value/unit.  A competing pool alias in that
    clause, an ambiguous ``respectively`` construction, unit conversion, or
    cross-clause attribution fails closed.
    """

    normalized = _normalize_document_text(abstract)
    if not normalized:
        return None
    candidate_terms = list(candidate_aliases)
    target_terms = _target_exact_phrases(target)
    if not candidate_terms or not target_terms:
        return None

    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for sentence_index, match in enumerate(
        re.finditer(r".+?(?:[!?]+|(?<!\d)\.(?!\d)|$)", normalized)
    ):
        raw_sentence = match.group(0)
        leading = len(raw_sentence) - len(raw_sentence.lstrip())
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        parenthetical_binding = _primary_parenthetical_binding(
            sentence,
            candidate_aliases=candidate_terms,
            competitor_aliases=competitor_aliases,
            target=target,
            measurement=measurement,
        )
        parenthetical_excerpt = sentence.rstrip(".!?").rstrip()
        if parenthetical_binding is not None:
            ranked.append(
                (
                    sentence_index,
                    -1,
                    {
                        "excerpt": parenthetical_excerpt,
                        "excerpt_location": {
                            "field": "abstract",
                            "sentence_index": sentence_index,
                            "clause_index": None,
                            "normalized_char_start": match.start() + leading,
                            "normalized_char_end": (
                                match.start()
                                + leading
                                + len(parenthetical_excerpt)
                            ),
                            "selection_scope": (
                                "atomic_parenthetical_with_sentence_context"
                            ),
                        },
                        "excerpt_truncated": False,
                        **parenthetical_binding,
                        "normalized_content_hash": sha256(
                            normalized.encode("utf-8")
                        ).hexdigest(),
                        "normalized_content_length": len(normalized),
                    },
                )
            )
            continue
        if re.search(
            r"(?<!\w)respectively(?!\w)",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue
        clause_start = 0
        separators = list(
            re.finditer(
                r"\s*(?:;|:|\u2014|\u2013|(?<!\w)(?:whereas|while|but)(?!\w))\s*",
                sentence,
                flags=re.IGNORECASE,
            )
        )
        boundaries = [*separators, None]
        for clause_index, separator in enumerate(boundaries):
            clause_end = separator.start() if separator is not None else len(sentence)
            raw_clause = sentence[clause_start:clause_end]
            clause_leading = len(raw_clause) - len(raw_clause.lstrip())
            clause = raw_clause.strip()
            absolute_start = (
                match.start() + leading + clause_start + clause_leading
            )
            if clause and len(clause) <= _PRIMARY_EXCERPT_MAX_CHARS:
                binding = _primary_clause_binding(
                    clause,
                    candidate_aliases=candidate_terms,
                    competitor_aliases=competitor_aliases,
                    target=target,
                    measurement=measurement,
                )
                if binding is not None:
                    ranked.append(
                        (
                            sentence_index,
                            clause_index,
                            {
                                "excerpt": clause,
                                "excerpt_location": {
                                    "field": "abstract",
                                    "sentence_index": sentence_index,
                                    "clause_index": clause_index,
                                    "normalized_char_start": absolute_start,
                                    "normalized_char_end": absolute_start + len(clause),
                                },
                                "excerpt_truncated": False,
                                **binding,
                                "normalized_content_hash": sha256(
                                    normalized.encode("utf-8")
                                ).hexdigest(),
                                "normalized_content_length": len(normalized),
                            },
                        )
                    )
            if separator is None:
                break
            clause_start = separator.end()
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _bounded_chembl_operation(operation: str):
    """Share one hard deadline across every ChEMBL call in an operation."""

    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with self._chembl_operation_budget(operation):
                return function(self, *args, **kwargs)

        return wrapped

    return decorate


class LiveScientificGateway:
    """Actual MA2 candidate gateway backed by ChEMBL, RDKit, Vina and ADMET-AI.

    No fixture or hash-derived scientific value exists in this class. Missing
    measured ligands, RDKit generation capacity, receptor/box profiles,
    docking executables, or ADMET model endpoints are hard failures.
    """

    status_endpoint = "https://www.ebi.ac.uk/chembl/api/data/status.json"
    target_endpoint = "https://www.ebi.ac.uk/chembl/api/data/target.json"
    mechanism_endpoint = "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
    activity_endpoint = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    assay_endpoint = "https://www.ebi.ac.uk/chembl/api/data/assay.json"
    molecule_endpoint = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    document_endpoint = "https://www.ebi.ac.uk/chembl/api/data/document.json"
    warning_endpoint = "https://www.ebi.ac.uk/chembl/api/data/drug_warning.json"
    pubmed_summary_endpoint = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    )
    uniprot_record_endpoint = "https://rest.uniprot.org/uniprotkb"
    _retryable_transport_categories = frozenset(
        {
            "read_timeout",
            "connect_timeout",
            "connection_pool_timeout",
            "connection",
            "remote_protocol",
        }
    )
    _retryable_status_codes = frozenset({408, 425, 429, 500, 502, 503, 504})
    backend_kind = "live"
    fixture = False

    def __init__(
        self,
        *,
        docking: DockingBackend,
        admet: ADMETBackend,
        client: HTTPClient | None = None,
        primary_search_client: HTTPClient | None = None,
        timeout_seconds: float = 12.0,
        max_http_attempts: int = 2,
        retry_backoff_seconds: float = 0.25,
        max_retry_backoff_seconds: float = 2.0,
        operation_timeout_seconds: float = 60.0,
        min_pchembl_value: float = 5.0,
        repurposing_evidence_policy: str | None = None,
        python_executable: str | Path | None = None,
        worker_environment: Mapping[str, str] | None = None,
        brics_worker_source_sha256: str | None = None,
        brics_timeout_seconds: float = 120.0,
        brics_max_builds: int = 20_000,
        sleeper: Callable[[float], None] = sleep,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if (
            timeout_seconds <= 0
            or operation_timeout_seconds <= 0
            or brics_timeout_seconds <= 0
        ):
            raise ValueError("Scientific source/tool timeouts must be positive")
        if not 1 <= max_http_attempts <= 3:
            raise ValueError("max_http_attempts must be between 1 and 3")
        if retry_backoff_seconds < 0 or max_retry_backoff_seconds < 0:
            raise ValueError("ChEMBL retry backoff values must not be negative")
        if not 0 <= min_pchembl_value <= 14:
            raise ValueError("min_pchembl_value must be between 0 and 14")
        if brics_max_builds < 1:
            raise ValueError("brics_max_builds must be positive")
        self.docking = docking
        self.admet = admet
        self._owns_client = client is None
        self.client = (
            client
            if client is not None
            else ScientificHttpClient(
                timeout_seconds=timeout_seconds,
                max_attempts=max_http_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_delay_seconds=max_retry_backoff_seconds,
            )
        )
        # PubMed paired-abstract search has a hard three-stage wire bound.
        # Its adapter therefore requires a separate one-attempt transport;
        # the ChEMBL client's retry policy is deliberately left unchanged.
        # When a custom ChEMBL client is injected without a matching search
        # client (the normal unit-test seam), a distinct zero-wire client
        # fails closed. Normal live construction always owns two real pools.
        self._owns_primary_search_client = (
            primary_search_client is None and client is None
        )
        self.primary_search_client = (
            primary_search_client
            if primary_search_client is not None
            else ScientificHttpClient(
                timeout_seconds=timeout_seconds,
                max_attempts=1,
                retry_backoff_seconds=0.0,
                max_retry_delay_seconds=0.0,
            )
            if client is None
            else _UnconfiguredPrimarySearchClient()
        )
        self.timeout_seconds = timeout_seconds
        self.max_http_attempts = max_http_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_retry_backoff_seconds = max_retry_backoff_seconds
        self.operation_timeout_seconds = operation_timeout_seconds
        self._sleeper = sleeper
        self._clock = clock
        self.min_pchembl_value = min_pchembl_value
        self.repurposing_evidence_policy = repurposing_evidence_policy
        self.brics_timeout_seconds = brics_timeout_seconds
        self.brics_max_builds = brics_max_builds
        self._worker_environment = _worker_environment(worker_environment)
        observed_brics_worker_sha256 = sha256(
            _BRICS_WORKER_PATH.read_bytes()
        ).hexdigest()
        self._brics_worker_source_sha256 = (
            observed_brics_worker_sha256
            if brics_worker_source_sha256 is None
            else brics_worker_source_sha256
        )
        if (
            not isinstance(self._brics_worker_source_sha256, str)
            or len(self._brics_worker_source_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in self._brics_worker_source_sha256)
            or self._brics_worker_source_sha256 != observed_brics_worker_sha256
        ):
            raise ScientificToolUnavailable("RDKit BRICS worker source is not frozen")
        self.brics = ExecutableRunner(
            "RDKit BRICS generator",
            executable=python_executable or sys.executable,
            # The worker's isolated --check payload is the sole version source.
            # Leaving the runner's generic lazy probe enabled would spawn an
            # unisolated, uncounted ``python --version`` child during provenance.
            version_args=(),
        )
        self._chembl_version: str | None = None
        self._chembl_version_probe_complete = False
        self._chembl_version_resolution: dict[str, Any] = {
            "status": "not_probed",
            "reason_code": None,
            "detail": None,
            "provenance": {},
        }
        self._http_calls: list[HTTPCallProvenance] = []
        # Successful authoritative responses are reused only inside this
        # gateway instance (normally one run). There is no persistent or
        # cross-run cache, so a cached response cannot silently outlive its
        # recorded ChEMBL release or query parameters.
        self._json_cache: dict[
            tuple[str, str], tuple[dict[str, Any], HTTPCallProvenance]
        ] = {}
        self._operation_context: ContextVar[dict[str, Any] | None] = ContextVar(
            f"drugpilot_chembl_operation_{id(self)}",
            default=None,
        )
        self._normalized_targets: dict[str, TargetCandidate] = {}
        self._seed_cache: dict[tuple[str, str], CandidateSeed] = {}
        self._activity_cache: dict[tuple[str, str], tuple[ActivityMeasurement, ...]] = {}
        self.last_operation_provenance: dict[str, Any] = {}
        self._brics_readiness_cache: dict[str, Any] | None = None

    @property
    def source_version(self) -> str:
        return (
            f"chembl={self._chembl_version or 'unresolved'};"
            f"{self.docking.source_version};{self.admet.source_version}"
        )

    def close(self) -> None:
        """Close only HTTP pools constructed by this gateway."""

        if self._owns_client:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
        if (
            self._owns_primary_search_client
            and self.primary_search_client is not self.client
        ):
            close = getattr(self.primary_search_client, "close", None)
            if callable(close):
                close()

    @_bounded_chembl_operation("normalize_target")
    def normalize_target(self, target: TargetCandidate) -> TargetCandidate:
        if not target.uniprot_id:
            raise ScientificDataUnavailable(
                f"Live MA2 requires a resolved UniProt accession for {target.symbol}"
            )
        accession = target.uniprot_id.upper()
        self._require_chembl_version()
        target_resolution_fallback: dict[str, Any] | None = None
        try:
            payload, call = self._get_json(
                self.target_endpoint,
                params={
                    "target_components__accession": accession,
                    "limit": 30,
                },
            )
        except ScientificDataUnavailable as exc:
            failure = deepcopy(getattr(exc, "provenance", {}))
            if failure.get("retryable") is not True:
                raise
            (
                selected,
                call,
                target_resolution_fallback,
            ) = self._resolve_target_via_uniprot_xref(
                accession,
                trigger_failure=failure,
            )
        else:
            targets = payload.get("targets", [])
            if not isinstance(targets, list):
                raise ScientificDataUnavailable(
                    "ChEMBL target response has no target list"
                )
            exact: list[dict[str, Any]] = []
            for item in targets:
                if not isinstance(item, dict):
                    continue
                components = item.get("target_components") or []
                accessions = {
                    str(component.get("accession") or "").upper()
                    for component in components
                    if isinstance(component, dict)
                }
                if accession in accessions:
                    exact.append(item)
            if not exact:
                raise NoMeasuredLigandCandidates(
                    target_symbol=target.symbol,
                    target_uniprot_id=target.uniprot_id,
                    direction="target_resolution",
                    eligible_count=0,
                    minimum_required=1,
                    reason_code="NO_EXACT_CHEMBL_TARGET",
                    provenance={
                        "target_query": call.as_dict(),
                        "matching_policy": (
                            "exact target-component UniProt accession"
                        ),
                    },
                )
            exact.sort(
                key=lambda item: (
                    str(item.get("target_type")) != "SINGLE PROTEIN",
                    str(item.get("organism")) != "Homo sapiens",
                    str(item.get("target_chembl_id")),
                )
            )
            selected = exact[0]
        target_chembl_id = str(selected.get("target_chembl_id") or "").strip()
        if not target_chembl_id:
            raise ScientificDataUnavailable("ChEMBL exact target is missing its stable ID")
        normalized = target.model_copy(
            update={
                "target_name": selected.get("pref_name") or target.target_name,
                "identity_status": "resolved",
                "identity_sources": list(
                    dict.fromkeys([*target.identity_sources, "ChEMBL", "UniProt"])
                ),
                "metadata": {
                    **target.metadata,
                    "chembl_target_id": target_chembl_id,
                    "chembl_target_type": selected.get("target_type"),
                    "chembl_organism": selected.get("organism"),
                    "chembl_version": self._chembl_version,
                    "chembl_version_resolution": deepcopy(
                        self._chembl_version_resolution
                    ),
                    "chembl_target_resolution": call.as_dict(),
                    **(
                        {
                            "chembl_target_resolution_fallback": deepcopy(
                                target_resolution_fallback
                            )
                        }
                        if target_resolution_fallback is not None
                        else {}
                    ),
                    "fixture": False,
                },
            }
        )
        self._normalized_targets[target.symbol.casefold()] = normalized
        self.last_operation_provenance = {
            "operation": "normalize_target",
            "http": call.as_dict(),
            "target_resolution_fallback": deepcopy(
                target_resolution_fallback
            ),
            "chembl_version": self._chembl_version,
            "chembl_version_resolution": deepcopy(
                self._chembl_version_resolution
            ),
            "fixture": False,
        }
        return normalized

    def _resolve_target_via_uniprot_xref(
        self,
        accession: str,
        *,
        trigger_failure: Mapping[str, Any],
    ) -> tuple[dict[str, Any], HTTPCallProvenance, dict[str, Any]]:
        """Resolve a ChEMBL target through one exact official UniProt xref."""

        uniprot_call: HTTPCallProvenance | None = None
        chembl_call: HTTPCallProvenance | None = None

        def fail(
            reason_code: str,
            message: str,
            *,
            source: str,
            detail: Mapping[str, Any] | None = None,
        ) -> None:
            unavailable = ScientificDataUnavailable(
                f"{reason_code}: {message}"
            )
            unavailable.provenance = {
                "source": source,
                "operation": "normalize_target_via_uniprot_chembl_xref",
                "reason_code": reason_code,
                "target_uniprot_accession": accession,
                "target_accession_query_failure": deepcopy(
                    dict(trigger_failure)
                ),
                "uniprot_record_query": (
                    uniprot_call.as_dict()
                    if uniprot_call is not None
                    else None
                ),
                "chembl_target_query": (
                    chembl_call.as_dict()
                    if chembl_call is not None
                    else None
                ),
                "detail": deepcopy(dict(detail or {})),
                "fixture": False,
            }
            raise unavailable

        uniprot_url = (
            f"{self.uniprot_record_endpoint}/{quote(accession)}.json"
        )
        try:
            uniprot_payload, uniprot_call = self._get_json(
                uniprot_url,
                params={},
                source="UniProt",
            )
        except ScientificDataUnavailable as exc:
            provenance = deepcopy(getattr(exc, "provenance", {}))
            provenance.update(
                {
                    "reason_code": "UNIPROT_RECORD_UNAVAILABLE",
                    "target_uniprot_accession": accession,
                    "target_accession_query_failure": deepcopy(
                        dict(trigger_failure)
                    ),
                    "fixture": False,
                }
            )
            exc.provenance = provenance
            raise
        returned_accession = str(
            uniprot_payload.get("primaryAccession") or ""
        ).upper()
        if returned_accession != accession:
            fail(
                "UNIPROT_ACCESSION_CONFLICT",
                "official UniProt record did not match the requested accession",
                source="UniProt",
                detail={"returned_primary_accession": returned_accession},
            )
        raw_xrefs = uniprot_payload.get("uniProtKBCrossReferences")
        if not isinstance(raw_xrefs, list):
            fail(
                "UNIPROT_CHEMBL_XREF_MISSING",
                "official UniProt record has no cross-reference list",
                source="UniProt",
            )
        chembl_xrefs = [
            str(item.get("id") or "").strip()
            for item in raw_xrefs
            if isinstance(item, dict)
            and str(item.get("database") or "").casefold() == "chembl"
            and str(item.get("id") or "").strip()
        ]
        if not chembl_xrefs:
            fail(
                "UNIPROT_CHEMBL_XREF_MISSING",
                "official UniProt record has no exact ChEMBL cross-reference",
                source="UniProt",
            )
        if len(chembl_xrefs) != 1:
            fail(
                "UNIPROT_CHEMBL_XREF_AMBIGUOUS",
                "official UniProt record has multiple ChEMBL cross-references",
                source="UniProt",
                detail={"chembl_xref_count": len(chembl_xrefs)},
            )
        chembl_target_id = chembl_xrefs[0]
        try:
            chembl_payload, chembl_call = self._get_json(
                self.target_endpoint,
                params={
                    "target_chembl_id": chembl_target_id,
                    "limit": 1,
                },
                source="ChEMBL",
            )
        except ScientificDataUnavailable as exc:
            provenance = deepcopy(getattr(exc, "provenance", {}))
            provenance.update(
                {
                    "reason_code": "CHEMBL_XREF_TARGET_QUERY_UNAVAILABLE",
                    "target_uniprot_accession": accession,
                    "uniprot_chembl_xref": chembl_target_id,
                    "target_accession_query_failure": deepcopy(
                        dict(trigger_failure)
                    ),
                    "uniprot_record_query": uniprot_call.as_dict(),
                    "fixture": False,
                }
            )
            exc.provenance = provenance
            raise
        targets = chembl_payload.get("targets")
        if not isinstance(targets, list) or not targets:
            fail(
                "CHEMBL_XREF_TARGET_NOT_FOUND",
                "ChEMBL returned no target for the exact UniProt xref",
                source="ChEMBL",
                detail={"uniprot_chembl_xref": chembl_target_id},
            )
        exact_targets = [
            item
            for item in targets
            if isinstance(item, dict)
            and str(item.get("target_chembl_id") or "").strip()
            == chembl_target_id
        ]
        if len(exact_targets) != 1:
            fail(
                "CHEMBL_XREF_TARGET_AMBIGUOUS",
                "ChEMBL target-ID response was not exactly one target",
                source="ChEMBL",
                detail={
                    "uniprot_chembl_xref": chembl_target_id,
                    "exact_target_count": len(exact_targets),
                },
            )
        selected = exact_targets[0]
        components = selected.get("target_components")
        component_accessions = {
            str(component.get("accession") or "").upper()
            for component in components or []
            if isinstance(component, dict)
            and str(component.get("accession") or "").strip()
        }
        if component_accessions != {accession}:
            fail(
                "CHEMBL_XREF_TARGET_ACCESSION_CONFLICT",
                "ChEMBL target components conflict with the requested UniProt accession",
                source="ChEMBL",
                detail={
                    "uniprot_chembl_xref": chembl_target_id,
                    "returned_component_accessions": sorted(
                        component_accessions
                    ),
                },
            )
        if (
            str(selected.get("target_type") or "") != "SINGLE PROTEIN"
            or str(selected.get("organism") or "") != "Homo sapiens"
        ):
            fail(
                "CHEMBL_XREF_TARGET_CLASS_CONFLICT",
                "xref target is not a human single-protein ChEMBL target",
                source="ChEMBL",
                detail={
                    "uniprot_chembl_xref": chembl_target_id,
                    "target_type": selected.get("target_type"),
                    "organism": selected.get("organism"),
                },
            )
        fallback = {
            "used": True,
            "reason_code": "CHEMBL_ACCESSION_QUERY_UNAVAILABLE_UNIPROT_XREF",
            "target_accession_query_failure": deepcopy(
                dict(trigger_failure)
            ),
            "uniprot_record_query": uniprot_call.as_dict(),
            "uniprot_primary_accession": returned_accession,
            "uniprot_chembl_xref": chembl_target_id,
            "chembl_target_query": chembl_call.as_dict(),
            "independent_validation": {
                "target_chembl_id_exact": True,
                "target_component_accessions": sorted(
                    component_accessions
                ),
                "target_type": "SINGLE PROTEIN",
                "organism": "Homo sapiens",
            },
            "matching_policy": (
                "one exact official UniProt ChEMBL xref followed by exact "
                "ChEMBL target ID and target-component accession validation"
            ),
            "fixture": False,
        }
        return selected, chembl_call, fallback

    def readiness(self, target: TargetCandidate | None = None) -> dict[str, Any]:
        target_reason = None
        if target is not None and not target.uniprot_id:
            target_reason = "A resolved UniProt accession is required for exact ChEMBL routing"
        chembl = {
            "tool": "ChEMBL",
            "operation": "retrieve_measured_target_ligands",
            "status": "completed" if target_reason is None else "unavailable",
            "backend": "official_live_rest_api",
            "version": self._chembl_version,
            "fixture": False,
            "required": True,
            "reason": target_reason,
            "provenance": {
                "status_endpoint": self.status_endpoint,
                "target_endpoint": self.target_endpoint,
                "mechanism_endpoint": self.mechanism_endpoint,
                "activity_endpoint": self.activity_endpoint,
                "version_resolution": deepcopy(
                    self._chembl_version_resolution
                ),
                "reachability_is_verified_on_execution": True,
            },
        }
        brics = self._brics_readiness()
        docking_readiness = (
            self.docking.readiness(target)  # type: ignore[attr-defined]
            if hasattr(self.docking, "readiness")
            else {
                "tool": "docking",
                "operation": "score_binding_pose",
                "status": "completed",
                "backend": self.docking.source_version,
                "fixture": False,
                "required": True,
                "reason": None,
                "provenance": {},
            }
        )
        admet_readiness = (
            self.admet.readiness()  # type: ignore[attr-defined]
            if hasattr(self.admet, "readiness")
            else {
                "tool": "ADMET",
                "operation": "predict_six_endpoint_panel",
                "status": "completed",
                "backend": self.admet.source_version,
                "fixture": False,
                "required": True,
                "reason": None,
                "provenance": {},
            }
        )
        tools = [chembl, brics, docking_readiness, admet_readiness]
        unavailable = [
            item
            for item in tools
            if item.get("required") is True
            and item.get("status") != "completed"
        ]
        return {
            "backend_kind": self.backend_kind,
            "fixture": self.fixture,
            "status": "unavailable" if unavailable else "completed",
            "source_version": self.source_version,
            "tools": tools,
            "reason": (
                "; ".join(str(item.get("reason")) for item in unavailable)
                if unavailable
                else None
            ),
        }

    def _brics_readiness(self) -> dict[str, Any]:
        self._require_brics_worker_source()
        if self._brics_readiness_cache is not None:
            return deepcopy(self._brics_readiness_cache)
        try:
            result = self.brics.run(
                (
                    *ISOLATED_PYTHON_FLAGS,
                    str(_BRICS_WORKER_PATH),
                    "--check",
                ),
                timeout_seconds=min(15.0, self.brics_timeout_seconds),
                env=self._worker_environment,
            )
            payload = json.loads(result.stdout)
            ready = bool(
                payload.get("status") == "ready"
                and payload.get("rdkit_version")
                and payload.get("python_isolated") is True
                and payload.get("no_user_site") is True
                and payload.get("worker_source_sha256")
                == self._brics_worker_source_sha256
            )
            readiness = {
                "tool": "RDKit BRICS",
                "operation": "generate_candidates_from_measured_ligands",
                "status": "completed" if ready else "unavailable",
                "backend": "isolated_local_process",
                "version": payload.get("rdkit_version"),
                "fixture": False,
                "required": True,
                "reason": None if ready else "RDKit BRICS readiness probe failed",
                "provenance": {
                    "measured_ligand_seed_required": True,
                    "timeout_seconds": self.brics_timeout_seconds,
                    "max_builds": self.brics_max_builds,
                    "worker_runtime_attestation": {
                        "schema_version": "drugpilot_local_readiness_child/v1",
                        "worker": "RDKit BRICS",
                        "command_mode": "--check",
                        "python_isolation_flags": list(ISOLATED_PYTHON_FLAGS),
                        "worker_filename": _BRICS_WORKER_PATH.name,
                        "worker_source_sha256": payload.get(
                            "worker_source_sha256"
                        ),
                        "worker_environment": worker_environment_snapshot(
                            self._worker_environment
                        ),
                        "model_loaded": False,
                        "inference_executed": False,
                        "network_access_required": False,
                        "monitor_direct_child_instrumentation": False,
                        "attestation_basis": (
                            "frozen_worker_source_and_sanitized_process_contract"
                        ),
                    },
                    "readiness_execution": result.provenance.as_dict(),
                },
            }
        except (ScientificToolError, ValueError, json.JSONDecodeError) as exc:
            readiness = {
                "tool": "RDKit BRICS",
                "operation": "generate_candidates_from_measured_ligands",
                "status": "unavailable",
                "backend": "isolated_local_process",
                "version": None,
                "fixture": False,
                "required": True,
                "reason": f"BRICS readiness probe failed: {type(exc).__name__}",
                "provenance": {
                    "measured_ligand_seed_required": True,
                    "timeout_seconds": self.brics_timeout_seconds,
                    "max_builds": self.brics_max_builds,
                },
            }
        self._brics_readiness_cache = deepcopy(readiness)
        return deepcopy(readiness)

    def _require_brics_worker_source(self) -> None:
        if sha256(_BRICS_WORKER_PATH.read_bytes()).hexdigest() != (
            self._brics_worker_source_sha256
        ):
            raise ScientificToolUnavailable(
                "RDKit BRICS worker source changed after runtime freeze"
            )

    @_bounded_chembl_operation("retrieve_repurposing")
    def retrieve_repurposing(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]:
        if limit < 1:
            raise ValueError("repurposing candidate limit must be positive")
        fixed_policy = self.repurposing_evidence_policy in _FIXED_REPURPOSING_POOL_POLICIES
        maximum = (_MAX_REPURPOSING_CANDIDATE_IDENTITIES if fixed_policy
                   else _MAX_GENERIC_REPURPOSING_POOL_IDENTITIES)
        if limit > maximum:
            raise ValueError(
                "repurposing candidate limit must not exceed "
                f"{maximum}"
            )
        pool_limit = _MAX_REPURPOSING_CANDIDATE_IDENTITIES if fixed_policy else max(
            _MAX_REPURPOSING_CANDIDATE_IDENTITIES, limit,
        )
        http_start = len(self._http_calls)
        audited_calls = getattr(self.client, "calls", None)
        audited_start = len(audited_calls) if isinstance(audited_calls, list) else 0
        try:
            ligands = self._fetch_measured_ligands(
                target,
                direction=direction,
                fetch_limit=_MAX_ACTIVITY_ROWS_PER_OPERATION,
                minimum_required=1,
                require_clinical=True,
                # Freeze the declared mechanism-first pool before any
                # warning, activity, document, assay, or PubMed evidence is
                # observed.  ``limit`` is MA2's requested eligible count; it
                # never caps the returned surviving pool. Exact paper policies
                # retain their preregistered top-12; generic discovery accepts
                # the public adapter's bounded reserve-pool contract (<=100).
                candidate_identity_limit=pool_limit,
                discovery_lane="mechanism_first",
            )
        except Exception as exc:
            failure = getattr(exc, "provenance", {})
            completed_empty = (
                isinstance(exc, NoMeasuredLigandCandidates)
                and exc.conclusive is True
            )
            current_audited = getattr(self.client, "calls", None)
            self.last_operation_provenance = {
                "operation": "retrieve_repurposing",
                "status": "completed_empty" if completed_empty else "error",
                "source": "ChEMBL live API",
                "source_search_complete": completed_empty,
                "search_completed": completed_empty,
                "partial_source_result": not completed_empty,
                "chembl_version": self._chembl_version,
                "chembl_version_resolution": deepcopy(
                    self._chembl_version_resolution
                ),
                "requested_limit": limit,
                "requested_eligible_count": limit,
                "candidate_count": 0 if completed_empty else None,
                "result_returned": False,
                "candidate_ordering": (
                    "fixed_prewarning_clinical_phase_desc_then_stable_"
                    "parent_molecule_chembl_id_v1"
                ),
                "pchembl_used_for_prewarning_selection": False,
                "primary_document_used_for_prewarning_selection": False,
                "primary_document_used_for_candidate_ordering": False,
                "gateway_return_cap_applied": False,
                "eligible_quota_evaluation_owner": "MA2",
                "failure": (
                    deepcopy(dict(failure))
                    if isinstance(failure, Mapping)
                    else {"exception_type": type(exc).__name__}
                ),
                "http_calls": [
                    call.as_dict() for call in self._http_calls[http_start:]
                ],
                "http_call_scope": (
                    "successful_gateway_calls_since_operation_start; failed "
                    "calls remain in logical_client_calls and failure provenance"
                ),
                "logical_client_calls": (
                    deepcopy(current_audited[audited_start:])
                    if isinstance(current_audited, list)
                    else []
                ),
                "fixture": False,
            }
            raise
        # _fetch_measured_ligands preserves the fixed pre-evidence phase/ID
        # order after reason-coded gateway exclusions.  Return that complete
        # bounded pool so MA2 can apply reserved/scientific/policy exclusions
        # in order until its requested eligible count is reached.
        selected = list(ligands)
        seeds = [item.seed for item in selected]
        returned_ids = [item.chembl_id for item in selected]
        fixed_ids = list(
            seeds[0].metadata.get("candidate_discovery_bounds", {}).get(
                "prewarning_pool_ids", []
            )
        )
        fixed_id_set = set(fixed_ids)
        returned_id_set = set(returned_ids)
        expected_return_order = [
            chembl_id
            for chembl_id in fixed_ids
            if chembl_id in returned_id_set
        ]
        delegate_seed_projection = (
            _paper_native_v6_delegate_seed_source_projection(
                fixed_candidate_ids=fixed_ids,
                seeds=seeds,
            )
            if self.repurposing_evidence_policy
            == PAPER_NATIVE_V6_POLICY_TOKEN
            else None
        )
        bounds_delta = {
            "requested_eligible_count": limit,
            "fixed_prewarning_pool_limit": pool_limit,
            "gateway_return_candidate_count": len(seeds),
            "gateway_return_candidate_ids": returned_ids,
            "gateway_return_cap_applied": False,
            "eligible_quota_evaluation_owner": "MA2",
            "gateway_returned_from_fixed_prewarning_pool": bool(
                returned_ids
                and set(returned_ids).issubset(fixed_id_set)
            ),
            "gateway_return_order_matches_fixed_prewarning_subsequence": (
                returned_ids == expected_return_order
            ),
            "no_backfill_beyond_prewarning_pool": True,
            **(
                {
                    "delegate_seed_source_binding_projection": deepcopy(
                        delegate_seed_projection
                    )
                }
                if delegate_seed_projection is not None
                else {}
            ),
        }
        for seed in seeds:
            metadata_holders = [
                seed.metadata,
                *(evidence.metadata for evidence in seed.evidence),
            ]
            for metadata in metadata_holders:
                bounds = metadata.get("candidate_discovery_bounds")
                if isinstance(bounds, dict):
                    bounds.update(deepcopy(bounds_delta))
        for seed in seeds:
            self._remember_seed(target, seed)
        current_audited = getattr(self.client, "calls", None)
        discovery_bounds = seeds[0].metadata.get("candidate_discovery_bounds", {})
        activity_counts = discovery_bounds.get("activity_counts_by_id", {})
        truncated_activity = any(
            isinstance(counts, Mapping) and isinstance(counts.get("total_count"), int)
            and isinstance(counts.get("inspected_count"), int)
            and counts["inspected_count"] < counts["total_count"]
            for counts in activity_counts.values()
        ) if isinstance(activity_counts, Mapping) else False
        source_search_complete = not (
            bool(discovery_bounds.get("activity_failure_count"))
            or bool(discovery_bounds.get("molecule_resolution_unresolved_ids"))
            or truncated_activity
        )
        self.last_operation_provenance = {
            "operation": "retrieve_repurposing",
            "status": "completed",
            "source": "ChEMBL live API",
            "source_search_complete": source_search_complete,
            "search_completed": source_search_complete,
            "partial_source_result": not source_search_complete,
            "exhaustive_discovery": False,
            "completeness_scope": "declared_bounded_candidate_and_activity_queries_only",
            "chembl_version": self._chembl_version,
            "chembl_version_resolution": deepcopy(
                self._chembl_version_resolution
            ),
            "candidate_count": len(seeds),
            "requested_limit": limit,
            "requested_eligible_count": limit,
            "fixed_prewarning_pool_limit": pool_limit,
            "gateway_return_candidate_ids": returned_ids,
            "partial_result": len(seeds) < limit,
            "candidate_ordering": (
                "fixed_prewarning_clinical_phase_desc_then_stable_"
                "parent_molecule_chembl_id_v1"
            ),
            "prewarning_candidate_ordering": (
                "clinical_max_phase_desc_then_stable_molecule_chembl_id_v1"
            ),
            "pchembl_threshold_only": True,
            "pchembl_used_for_candidate_ordering": False,
            "primary_document_used_for_prewarning_selection": False,
            "primary_document_used_for_candidate_ordering": False,
            "gateway_return_cap_applied": False,
            "gateway_return_order_matches_fixed_prewarning_subsequence": (
                returned_ids == expected_return_order
            ),
            "eligible_quota_evaluation_owner": "MA2",
            "candidate_discovery_bounds": deepcopy(
                seeds[0].metadata.get("candidate_discovery_bounds", {})
            ),
            **(
                {
                    "delegate_seed_source_binding_projection": deepcopy(
                        delegate_seed_projection
                    )
                }
                if delegate_seed_projection is not None
                else {}
            ),
            "http_calls": [
                call.as_dict() for call in self._http_calls[http_start:]
            ],
            "http_call_scope": "all_successful_gateway_calls_since_operation_start",
            "logical_client_calls": (
                deepcopy(current_audited[audited_start:])
                if isinstance(current_audited, list)
                else []
            ),
            "fixture": False,
        }
        return seeds

    @_bounded_chembl_operation("generate_de_novo")
    def generate_de_novo(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]:
        parent_count = max(12, min(60, limit * 4))
        parents = self._fetch_measured_ligands(
            target,
            direction=direction,
            fetch_limit=max(100, parent_count * 5),
            minimum_required=max(2, parent_count // 3),
            require_clinical=False,
        )[:parent_count]
        if len(parents) < 2:
            raise ScientificDataUnavailable(
                "De novo BRICS generation requires at least two measured target ligands"
            )
        with TemporaryDirectory(prefix="drugpilot-brics-") as temp_value:
            temp = Path(temp_value)
            input_path = temp / "parents.json"
            output_path = temp / "generated.json"
            input_path.write_text(
                json.dumps(
                    {"parent_smiles": [item.seed.smiles for item in parents]},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
                newline="\n",
            )
            self._require_brics_worker_source()
            execution = self.brics.run(
                (
                    *ISOLATED_PYTHON_FLAGS,
                    str(_BRICS_WORKER_PATH),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--limit",
                    str(limit),
                    "--max-builds",
                    str(self.brics_max_builds),
                ),
                timeout_seconds=self.brics_timeout_seconds,
                env=self._worker_environment,
            )
            if not output_path.is_file():
                raise ScientificDataUnavailable("RDKit BRICS produced no candidate artifact")
            generated = json.loads(output_path.read_text(encoding="utf-8"))
            if not (
                isinstance(generated, dict)
                and generated.get("python_isolated") is True
                and generated.get("no_user_site") is True
                and generated.get("worker_source_sha256")
                == self._brics_worker_source_sha256
            ):
                raise ScientificDataUnavailable(
                    "RDKit BRICS worker isolation/source attestation is invalid"
                )
        molecules = generated.get("molecules", [])
        if not isinstance(molecules, list) or not 1 <= len(molecules) <= limit:
            raise ScientificDataUnavailable(
                "RDKit BRICS returned an invalid candidate count: "
                f"{len(molecules) if isinstance(molecules, list) else 0}; "
                f"expected 1..{limit}"
            )
        parent_ids = [item.chembl_id for item in parents]
        parent_evidence = tuple(item.seed.evidence[0] for item in parents[:3] if item.seed.evidence)
        seeds: list[CandidateSeed] = []
        for index, molecule in enumerate(molecules, start=1):
            smiles = str(molecule.get("smiles") or "").strip()
            if not smiles:
                raise ScientificDataUnavailable("RDKit BRICS candidate is missing SMILES")
            stable = str(molecule.get("inchikey") or sha256(smiles.encode()).hexdigest()[:14])
            nearest_index = int(molecule.get("nearest_measured_seed_index") or 0)
            nearest_parent_id = (
                parent_ids[nearest_index]
                if 0 <= nearest_index < len(parent_ids)
                else None
            )
            seed = CandidateSeed(
                name=f"DN-{target.symbol}-{stable[:14]}",
                smiles=smiles,
                mechanism=(
                    f"RDKit BRICS analogue for {target.symbol}; intended direction="
                    f"{direction}, activity unmeasured"
                ),
                evidence=parent_evidence,
                metadata={
                    "fixture": False,
                    "generator": "RDKit BRICS",
                    "generator_version": generated.get("rdkit_version"),
                    "generation_rank": index,
                    "measured_seed_chembl_ids": parent_ids,
                    "measured_seed_count": len(parents),
                    "novelty_scope": generated.get("novelty_scope"),
                    "selection_policy": generated.get("selection_policy"),
                    "nearest_measured_seed_chembl_id": nearest_parent_id,
                    "nearest_measured_seed_tanimoto": molecule.get(
                        "nearest_measured_seed_tanimoto"
                    ),
                    "generation_descriptors": {
                        key: molecule.get(key)
                        for key in (
                            "molecular_weight",
                            "hbd",
                            "hba",
                            "ring_count",
                            "qed",
                            "structural_alerts",
                        )
                    },
                    "generation_execution": execution.provenance.as_dict(),
                    "activity_status": "unmeasured_generated_candidate",
                    "direction_requested": direction,
                    "direction_evidence_rules": [
                        "de_novo_generation_intended_direction"
                    ],
                    "policy_eligible": True,
                },
            )
            seeds.append(seed)
            self._remember_seed(target, seed)
        self.last_operation_provenance = {
            "operation": "generate_de_novo",
            "generator": "RDKit BRICS",
            "execution": execution.provenance.as_dict(),
            "parent_chembl_ids": parent_ids,
            "candidate_count": len(seeds),
            "requested_limit": limit,
            "partial_result": len(seeds) < limit,
            "fixture": False,
        }
        return seeds

    def dock(self, *, smiles: str, target: TargetCandidate) -> float:
        score = self.docking.dock(smiles=smiles, target=target)
        result = self.docking.last_result
        self.last_operation_provenance = {
            "operation": "dock",
            "fixture": False,
            "result": result.provenance() if result is not None else None,
        }
        return score

    def predict_admet(self, *, smiles: str) -> Mapping[str, float]:
        predictions = self.admet.predict(smiles=smiles)
        result = self.admet.last_result
        self.last_operation_provenance = {
            "operation": "predict_admet",
            "fixture": False,
            "result": result.provenance() if result is not None else None,
        }
        return predictions

    def refine(
        self,
        seed: CandidateSeed,
        *,
        reasons: Sequence[str],
        iteration: int,
    ) -> CandidateSeed:
        raise ScientificToolUnavailable(
            "No validated property-conditioned molecular optimizer is configured; "
            "live MA2 must set max_refinement_iterations=0"
        )

    def restore_activity(
        self,
        *,
        candidate_name: str,
        target: TargetCandidate,
    ) -> Sequence[ActivityMeasurement]:
        key = (target.symbol.casefold(), candidate_name.casefold())
        activities = self._activity_cache.get(key)
        if not activities:
            raise ScientificDataUnavailable(
                f"No measured live activity is cached for {candidate_name}"
            )
        return tuple(item.model_copy(deep=True) for item in activities)

    def restore_structure(
        self,
        *,
        candidate_name: str,
        route: CandidateRoute,
        target: TargetCandidate,
        direction: str,
    ) -> CandidateSeed:
        key = (target.symbol.casefold(), candidate_name.casefold())
        seed = self._seed_cache.get(key)
        if seed is None:
            raise ScientificDataUnavailable(
                f"No live-source structure is cached for {candidate_name}"
            )
        return seed

    def _discover_activity_first_ligands(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        fetch_limit: int,
        minimum_required: int,
        require_clinical: bool,
        candidate_identity_limit: int | None,
    ) -> dict[str, Any]:
        """Retain the measured-ligand discovery used only by de-novo seeding."""

        normalized = self._ensure_normalized(target)
        target_chembl_id = str(
            normalized.metadata.get("chembl_target_id") or ""
        )
        if not target_chembl_id:
            raise ScientificDataUnavailable(
                "Normalized target lacks a ChEMBL target ID"
            )
        raw_activity_row_limit = min(
            _MAX_ACTIVITY_ROWS_PER_OPERATION,
            max(1, int(fetch_limit)),
        )
        activity_params = {
            "target_chembl_id": target_chembl_id,
            "target_organism": "Homo sapiens",
            "assay_type": "B",
            "standard_type__in": "IC50,Ki,Kd,EC50",
            "pchembl_value__gte": self.min_pchembl_value,
            "order_by": "activity_id",
            "limit": raw_activity_row_limit,
        }
        activity_query_fallback: dict[str, Any] | None = None
        try:
            payload, activity_call = self._get_json(
                self.activity_endpoint,
                params=activity_params,
            )
        except ScientificDataUnavailable as exc:
            ordered_failure = deepcopy(getattr(exc, "provenance", {}))
            error_type = str(ordered_failure.get("error_type") or "")
            if ordered_failure.get("retryable") is not True:
                raise
            fallback_params = {
                key: value
                for key, value in activity_params.items()
                if key not in {"order_by", "limit", "offset"}
            }
            (
                payload,
                activity_call,
                bounded_pagination,
            ) = self._fetch_bounded_activity_pages(
                fallback_params,
                requested_limit=raw_activity_row_limit,
                trigger_failure=ordered_failure,
            )
            activity_query_fallback = {
                "used": True,
                "reason_code": (
                    "CHEMBL_SERVER_ORDERING_UNAVAILABLE"
                    if error_type.startswith("upstream_http_5")
                    else "CHEMBL_ACTIVITY_QUERY_UNAVAILABLE_BOUNDED_PAGINATION"
                ),
                "ordered_request_failure": ordered_failure,
                "fallback_policy": (
                    "identical scientific filters; remote ordering omitted; "
                    "page size <=20; <=4 pages and <=80 raw rows; deterministic "
                    "clinical-phase and stable-identifier ordering"
                ),
                **bounded_pagination,
            }
        raw_activities = payload.get("activities", [])
        if not isinstance(raw_activities, list):
            raise ScientificDataUnavailable(
                "ChEMBL activity response has no activity list"
            )
        raw_activity_response_count = len(raw_activities)
        activities = raw_activities[:raw_activity_row_limit]
        raw_activity_rows_truncated = (
            len(activities) < raw_activity_response_count
        )

        activity_rows_by_molecule: dict[
            str,
            list[tuple[dict[str, Any], ActivityMeasurement, str]],
        ] = {}
        for raw in activities:
            if not isinstance(raw, dict):
                continue
            if "data_validity_comment" not in raw:
                continue
            if raw["data_validity_comment"] not in (
                None,
                "",
                "Manually validated",
            ):
                continue
            chembl_id = str(raw.get("molecule_chembl_id") or "").strip()
            measurement = _normalize_activity(raw)
            if not chembl_id or measurement is None:
                continue
            direction_rule = _measurement_direction_rule(
                measurement, direction
            )
            if direction_rule is None:
                continue
            activity_rows_by_molecule.setdefault(chembl_id, []).append(
                (raw, measurement, direction_rule)
            )
        eligible_activity_count_by_molecule = {
            chembl_id: len(rows)
            for chembl_id, rows in activity_rows_by_molecule.items()
        }
        for rows in activity_rows_by_molecule.values():
            rows.sort(
                key=lambda item: (
                    -(_safe_float(item[0].get("pchembl_value")) or 0.0),
                    str(item[0].get("activity_id") or ""),
                )
            )
            del rows[_PRIMARY_ACTIVITY_DOCUMENT_TOP_K:]
        if len(activity_rows_by_molecule) < minimum_required:
            raise NoMeasuredLigandCandidates(
                target_symbol=normalized.symbol,
                target_uniprot_id=normalized.uniprot_id,
                direction=direction,
                eligible_count=len(activity_rows_by_molecule),
                minimum_required=minimum_required,
                reason_code=(
                    "INSUFFICIENT_DIRECTION_COMPATIBLE_MEASURED_LIGANDS"
                ),
                provenance={
                    "target_chembl_id": target_chembl_id,
                    "activity_query": activity_call.as_dict(),
                    "activity_query_fallback": deepcopy(
                        activity_query_fallback
                    ),
                    "raw_activity_row_limit": raw_activity_row_limit,
                    "raw_activity_response_count": (
                        raw_activity_response_count
                    ),
                    "raw_activity_count": len(activities),
                    "raw_activity_rows_truncated": (
                        raw_activity_rows_truncated
                    ),
                },
            )

        discovered_ids = sorted(activity_rows_by_molecule)
        molecule_payload, molecule_call = self._get_json(
            self.molecule_endpoint,
            params={
                "molecule_chembl_id__in": ",".join(discovered_ids),
                "limit": len(discovered_ids),
            },
        )
        molecules = molecule_payload.get("molecules", [])
        if not isinstance(molecules, list):
            raise ScientificDataUnavailable(
                "ChEMBL molecule response has no molecule list"
            )
        requested_molecule_ids = set(discovered_ids)
        resolved_molecule_by_id = {
            str(item.get("molecule_chembl_id")): item
            for item in molecules
            if isinstance(item, dict)
            and str(item.get("molecule_chembl_id") or "")
            in requested_molecule_ids
        }

        identity_resolution: dict[str, dict[str, Any]] = {}
        ranked_identities: list[tuple[int, str]] = []
        for chembl_id in discovered_ids:
            molecule = resolved_molecule_by_id.get(chembl_id)
            if molecule is None:
                identity_resolution[chembl_id] = {
                    "molecule_chembl_id": chembl_id,
                    "status": "excluded",
                    "max_phase": None,
                    "reason_codes": ["MOLECULE_METADATA_NOT_RESOLVED"],
                }
                continue
            raw_max_phase = _safe_float(molecule.get("max_phase"))
            if raw_max_phase is None:
                identity_resolution[chembl_id] = {
                    "molecule_chembl_id": chembl_id,
                    "status": "excluded",
                    "max_phase": None,
                    "reason_codes": ["CLINICAL_PHASE_UNRESOLVED"],
                }
                continue
            max_phase = int(raw_max_phase)
            if require_clinical and max_phase < 1:
                identity_resolution[chembl_id] = {
                    "molecule_chembl_id": chembl_id,
                    "status": "excluded",
                    "max_phase": max_phase,
                    "reason_codes": ["CLINICAL_PHASE_BELOW_REQUIRED"],
                }
                continue
            ranked_identities.append((max_phase, chembl_id))

        ranked_identities.sort(key=lambda item: (-item[0], item[1]))
        prewarning_limit = (
            min(candidate_identity_limit, len(ranked_identities))
            if candidate_identity_limit is not None
            else len(ranked_identities)
        )
        selected_ranked_identities = ranked_identities[:prewarning_limit]
        selected_ids = [
            chembl_id
            for _max_phase, chembl_id in selected_ranked_identities
        ]
        selected_id_set = set(selected_ids)
        for max_phase, chembl_id in ranked_identities:
            selected = chembl_id in selected_id_set
            identity_resolution[chembl_id] = {
                "molecule_chembl_id": chembl_id,
                "status": "selected" if selected else "excluded",
                "max_phase": max_phase,
                "reason_codes": [
                    "PREWARNING_POOL_SELECTED"
                    if selected
                    else "PREWARNING_POOL_LIMIT_REACHED"
                ],
            }

        candidate_discovery_bounds = {
            "schema_version": "bounded_candidate_identity_pool_v1",
            "discovery_lane": "activity_first_de_novo_only",
            "raw_activity_row_limit": raw_activity_row_limit,
            "raw_activity_response_count": raw_activity_response_count,
            "raw_activity_count": len(activities),
            "raw_activity_rows_truncated": raw_activity_rows_truncated,
            "discovered_identity_count": len(discovered_ids),
            "discovered_identity_ids": list(discovered_ids),
            "molecule_resolution_batch_count": 1,
            "molecule_resolution_limit": _MAX_ACTIVITY_ROWS_PER_OPERATION,
            "molecule_resolution_requested_count": len(discovered_ids),
            "molecule_resolution_requested_ids": list(discovered_ids),
            "molecule_resolution_count": len(resolved_molecule_by_id),
            "molecule_resolution_unresolved_ids": sorted(
                requested_molecule_ids.difference(resolved_molecule_by_id)
            ),
            "prewarning_pool_limit": candidate_identity_limit,
            "prewarning_pool_count": len(selected_ids),
            "prewarning_pool_ids": list(selected_ids),
            "prewarning_selection_policy": (
                "clinical_max_phase_desc_then_stable_molecule_chembl_id_v1"
            ),
            "returned_candidate_ordering_policy": (
                "clinical_phase_desc_then_verified_primary_evidence_then_"
                "stable_molecule_chembl_id_v1"
            ),
            "pchembl_used_for_prewarning_selection": False,
            "primary_document_used_for_prewarning_selection": False,
            "identity_resolution": [
                deepcopy(identity_resolution[chembl_id])
                for chembl_id in discovered_ids
            ],
            "no_backfill_beyond_prewarning_pool": True,
            "fixture": False,
            "missing_values_imputed": False,
        }
        if not selected_ids:
            raise NoMeasuredLigandCandidates(
                target_symbol=normalized.symbol,
                target_uniprot_id=normalized.uniprot_id,
                direction=direction,
                eligible_count=0,
                minimum_required=minimum_required,
                reason_code="INSUFFICIENT_CLINICAL_MEASURED_LIGANDS",
                provenance={
                    "target_chembl_id": target_chembl_id,
                    "activity_query": activity_call.as_dict(),
                    "activity_query_fallback": deepcopy(
                        activity_query_fallback
                    ),
                    "molecule_query": molecule_call.as_dict(),
                    "candidate_discovery_bounds": deepcopy(
                        candidate_discovery_bounds
                    ),
                    "fixture": False,
                    "missing_values_imputed": False,
                },
            )

        return {
            "normalized": normalized,
            "target_chembl_id": target_chembl_id,
            "discovered_ids": discovered_ids,
            "selected_ids": selected_ids,
            "resolved_molecule_by_id": resolved_molecule_by_id,
            "identity_resolution": identity_resolution,
            "candidate_discovery_bounds": candidate_discovery_bounds,
            "activity_rows_by_molecule": {
                chembl_id: activity_rows_by_molecule[chembl_id]
                for chembl_id in selected_ids
            },
            "eligible_activity_count_by_molecule": {
                chembl_id: eligible_activity_count_by_molecule[chembl_id]
                for chembl_id in selected_ids
            },
            "activity_calls_by_id": {
                chembl_id: activity_call for chembl_id in selected_ids
            },
            "activity_query_fallbacks_by_id": {
                chembl_id: deepcopy(activity_query_fallback)
                for chembl_id in selected_ids
            },
            "activity_failures_by_id": {},
            "molecule_call": molecule_call,
            "molecule_query_scope_ids": list(discovered_ids),
            "mechanism_provenance_by_id": {},
        }

    def _discover_mechanism_first_ligands(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        minimum_required: int,
        candidate_identity_limit: int,
    ) -> dict[str, Any]:
        """Discover one fixed clinical identity pool before any activity lookup."""

        normalized = self._ensure_normalized(target)
        target_chembl_id = str(
            normalized.metadata.get("chembl_target_id") or ""
        ).strip()
        if not target_chembl_id:
            raise ScientificDataUnavailable(
                "Normalized target lacks a ChEMBL target ID"
            )
        target_type = str(
            normalized.metadata.get("chembl_target_type") or ""
        )
        target_organism = str(
            normalized.metadata.get("chembl_organism") or ""
        )
        if (
            target_type != "SINGLE PROTEIN"
            or target_organism != "Homo sapiens"
        ):
            unavailable = ScientificDataUnavailable(
                "Mechanism-first repurposing requires an exact human "
                "single-protein ChEMBL target"
            )
            unavailable.provenance = {
                "source": "ChEMBL",
                "operation": "mechanism_first_candidate_discovery",
                "reason_code": "TARGET_NOT_EXACT_HUMAN_SINGLE_PROTEIN",
                "chembl_version": self._chembl_version,
                "chembl_version_resolution": deepcopy(
                    self._chembl_version_resolution
                ),
                "target_chembl_id": target_chembl_id,
                "target_type": target_type or None,
                "target_organism": target_organism or None,
                "fixture": False,
                "missing_values_imputed": False,
            }
            raise unavailable
        if direction not in {"inhibit", "modulate"}:
            # An adapter capability gap is not a completed negative search.
            # In particular, no activator query has been performed here.
            unavailable = ScientificToolUnavailable(
                "UNSUPPORTED_MECHANISM_ACTION_DIRECTION: mechanism-first "
                f"candidate discovery does not support {direction}"
            )
            unavailable.reason_code = "UNSUPPORTED_MECHANISM_ACTION_DIRECTION"
            unavailable.conclusive = False
            unavailable.provenance = {
                    "source": "ChEMBL",
                    "operation": "mechanism_first_candidate_discovery",
                    "reason_code": unavailable.reason_code,
                    "requested_direction": direction,
                    "search_completed": False,
                    "query_completed": False,
                    "outcome": "unsupported_capability_not_searched",
                    "chembl_version": self._chembl_version,
                    "chembl_version_resolution": deepcopy(
                        self._chembl_version_resolution
                    ),
                    "target_chembl_id": target_chembl_id,
                    "supported_directions": ["inhibit", "modulate"],
                    "required_action_type": "INHIBITOR",
                    "no_activity_first_fallback": True,
                    "fixture": False,
                }
            raise unavailable

        mechanism_params = {
            "target_chembl_id": target_chembl_id,
            "action_type": "INHIBITOR",
            "max_phase__gte": 1,
            "direct_interaction": 1,
            "molecular_mechanism": 1,
            "order_by": "molecule_chembl_id",
            "limit": _MAX_MECHANISM_ROWS_PER_OPERATION,
        }
        mechanism_payload, mechanism_call = self._get_json(
            self.mechanism_endpoint,
            params=mechanism_params,
        )
        raw_mechanisms = mechanism_payload.get("mechanisms")
        page_meta = mechanism_payload.get("page_meta")
        if not isinstance(raw_mechanisms, list) or not isinstance(
            page_meta, Mapping
        ):
            unavailable = ScientificDataUnavailable(
                "ChEMBL mechanism response lacks a bounded list or page metadata"
            )
            unavailable.provenance = {
                "source": "ChEMBL",
                "operation": "mechanism_first_candidate_discovery",
                "reason_code": "MECHANISM_RESPONSE_METADATA_MISSING",
                "chembl_version": self._chembl_version,
                "chembl_version_resolution": deepcopy(
                    self._chembl_version_resolution
                ),
                "target_chembl_id": target_chembl_id,
                "mechanism_query": mechanism_call.as_dict(),
                "fixture": False,
            }
            raise unavailable
        # Generic reserve pools may exceed the old 12-identity paper contract.
        # Keep each wire page <=80 and the complete mechanism scan bounded by
        # the declared pool (at most100 rows). Never follow an untrusted next URL
        # or describe an interrupted/over-bound scan as a negative result.
        mechanism_row_limit = max(_MAX_MECHANISM_ROWS_PER_OPERATION, candidate_identity_limit)
        mechanism_calls = [mechanism_call]
        mechanism_row_queries = {_canonical_sha256(row): mechanism_call for row in raw_mechanisms}
        if self.repurposing_evidence_policy not in _FIXED_REPURPOSING_POOL_POLICIES:
            raw_total = page_meta.get("total_count")
            try:
                initial_total = int(raw_total) if not isinstance(raw_total, bool) else -1
            except (TypeError, ValueError):
                initial_total = -1
            if 0 <= initial_total <= mechanism_row_limit:
                raw_mechanisms = list(raw_mechanisms)
                while 0 < len(raw_mechanisms) < initial_total and page_meta.get("next") not in (None, ""):
                    offset = len(raw_mechanisms)
                    page_limit = min(_MAX_MECHANISM_ROWS_PER_OPERATION, mechanism_row_limit - offset)
                    page_payload, page_call = self._get_json(self.mechanism_endpoint,
                        params={**mechanism_params, "offset": offset, "limit": page_limit})
                    rows = page_payload.get("mechanisms")
                    metadata = page_payload.get("page_meta")
                    valid_page = isinstance(rows, list) and isinstance(metadata, Mapping)
                    if valid_page:
                        hashes = [_canonical_sha256(row) for row in rows]
                        valid_page = (
                            metadata.get("total_count") == raw_total and 0 < len(rows) <= page_limit
                            and metadata.get("offset", offset) == offset
                            and offset + len(rows) <= initial_total
                            and not set(hashes).intersection(mechanism_row_queries)
                            and len(set(hashes)) == len(hashes)
                        )
                    if not valid_page:
                        unavailable = ScientificDataUnavailable("ChEMBL mechanism pagination is incomplete or inconsistent")
                        unavailable.provenance = {
                            "source": "ChEMBL", "operation": "mechanism_first_candidate_discovery",
                            "reason_code": "MECHANISM_IDENTITY_SET_INCOMPLETE_OR_OVER_CAP",
                            "source_search_complete": False, "search_completed": False,
                            "mechanism_queries": [call.as_dict() for call in [*mechanism_calls, page_call]],
                            "raw_mechanism_row_limit": mechanism_row_limit,
                            "raw_mechanism_response_count": offset, "fixture": False,
                        }
                        raise unavailable
                    mechanism_calls.append(page_call)
                    mechanism_row_queries.update({value: page_call for value in hashes})
                    raw_mechanisms.extend(rows)
                    page_meta = metadata
        raw_mechanism_rows_sha256 = sha256(
            json.dumps(
                raw_mechanisms,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        total_count_value = page_meta.get("total_count")
        if isinstance(total_count_value, bool):
            total_count_value = None
        try:
            mechanism_total_count = int(total_count_value)
        except (TypeError, ValueError):
            mechanism_total_count = -1
        if (
            mechanism_total_count < 0
            or mechanism_total_count > mechanism_row_limit
            or len(raw_mechanisms) != mechanism_total_count
            or len(raw_mechanisms) > mechanism_row_limit
            or page_meta.get("next") not in (None, "")
        ):
            unavailable = ScientificDataUnavailable(
                "ChEMBL mechanism identity set is incomplete or exceeds its cap"
            )
            unavailable.provenance = {
                "source": "ChEMBL",
                "operation": "mechanism_first_candidate_discovery",
                "reason_code": "MECHANISM_IDENTITY_SET_INCOMPLETE_OR_OVER_CAP",
                "chembl_version": self._chembl_version,
                "chembl_version_resolution": deepcopy(
                    self._chembl_version_resolution
                ),
                "target_chembl_id": target_chembl_id,
                "mechanism_query": mechanism_call.as_dict(),
                "raw_mechanism_row_limit": mechanism_row_limit,
                "source_search_complete": False,
                "search_completed": False,
                "mechanism_queries": [call.as_dict() for call in mechanism_calls],
                "raw_mechanism_response_count": len(raw_mechanisms),
                "mechanism_total_count": mechanism_total_count,
                "raw_mechanism_rows_sha256": raw_mechanism_rows_sha256,
                "no_activity_first_fallback": True,
                "fixture": False,
                "missing_values_imputed": False,
            }
            raise unavailable

        required_fields = {
            "action_type",
            "direct_interaction",
            "max_phase",
            "mechanism_refs",
            "molecular_mechanism",
            "molecule_chembl_id",
            "parent_molecule_chembl_id",
            "target_chembl_id",
        }
        malformed_rows: list[dict[str, Any]] = []
        parent_rows: dict[str, list[dict[str, Any]]] = {}
        invalid_parent_reasons: dict[str, set[str]] = {}
        parents_by_form: dict[str, set[str]] = {}
        for index, raw in enumerate(raw_mechanisms):
            if not isinstance(raw, dict):
                malformed_rows.append(
                    {
                        "row_index": index,
                        "reason_codes": ["MECHANISM_ROW_NOT_OBJECT"],
                    }
                )
                continue
            missing_fields = sorted(required_fields.difference(raw))
            form_id = str(
                raw.get("molecule_chembl_id") or ""
            ).strip()
            parent_field = raw.get("parent_molecule_chembl_id")
            parent_id = str(parent_field or form_id).strip()
            reasons: list[str] = []
            if missing_fields:
                reasons.append("MECHANISM_REQUIRED_FIELDS_MISSING")
            if not form_id or not parent_id:
                reasons.append("MECHANISM_STABLE_ID_MISSING")
            if str(raw.get("target_chembl_id") or "") != target_chembl_id:
                reasons.append("MECHANISM_TARGET_ID_MISMATCH")
            if str(raw.get("action_type") or "") != "INHIBITOR":
                reasons.append("MECHANISM_ACTION_TYPE_MISMATCH")
            if raw.get("direct_interaction") not in (1, "1", True):
                reasons.append("MECHANISM_DIRECT_INTERACTION_NOT_ESTABLISHED")
            if raw.get("molecular_mechanism") not in (1, "1", True):
                reasons.append("MOLECULAR_MECHANISM_NOT_ESTABLISHED")
            raw_phase_value = raw.get("max_phase")
            raw_phase = _safe_float(raw_phase_value)
            if (
                isinstance(raw_phase_value, bool)
                or raw_phase is None
                or raw_phase < 1
                or raw_phase > 4
                or not float(raw_phase).is_integer()
            ):
                reasons.append("MECHANISM_CLINICAL_PHASE_INVALID")
            refs = raw.get("mechanism_refs")
            if not isinstance(refs, list) or not refs:
                reasons.append("MECHANISM_REFERENCE_MISSING")
            elif not any(
                isinstance(ref, Mapping)
                and str(ref.get("ref_type") or "").strip()
                and (
                    str(ref.get("ref_id") or "").strip()
                    or str(ref.get("ref_url") or "").strip()
                )
                for ref in refs
            ):
                reasons.append("MECHANISM_REFERENCE_MALFORMED")
            if form_id and parent_id:
                parents_by_form.setdefault(form_id, set()).add(parent_id)
                parent_rows.setdefault(parent_id, []).append(raw)
            if reasons:
                malformed_rows.append(
                    {
                        "row_index": index,
                        "molecule_chembl_id": form_id or None,
                        "parent_molecule_chembl_id": parent_id or None,
                        "missing_fields": missing_fields,
                        "reason_codes": reasons,
                    }
                )
                if parent_id:
                    invalid_parent_reasons.setdefault(parent_id, set()).update(
                        reasons
                    )

        form_parent_conflicts = {
            form_id: sorted(parent_ids)
            for form_id, parent_ids in parents_by_form.items()
            if len(parent_ids) != 1
        }
        for form_id, parent_ids in form_parent_conflicts.items():
            for parent_id in parent_ids:
                invalid_parent_reasons.setdefault(parent_id, set()).add(
                    "FORM_PARENT_ID_CONFLICT"
                )

        discovered_ids = sorted(parent_rows)
        identity_resolution: dict[str, dict[str, Any]] = {
            parent_id: {
                "molecule_chembl_id": parent_id,
                "parent_molecule_chembl_id": parent_id,
                "status": "excluded",
                "max_phase": None,
                "reason_codes": sorted(
                    invalid_parent_reasons.get(parent_id, set())
                ),
            }
            for parent_id in discovered_ids
        }
        valid_parent_ids: list[str] = []
        mechanism_phase_by_id: dict[str, int] = {}
        mechanism_provenance_by_id: dict[str, dict[str, Any]] = {}
        for parent_id in discovered_ids:
            if invalid_parent_reasons.get(parent_id):
                continue
            rows = parent_rows[parent_id]
            phase_values = {
                int(float(str(row.get("max_phase")))) for row in rows
            }
            if len(phase_values) != 1:
                identity_resolution[parent_id]["reason_codes"].append(
                    "MECHANISM_MAX_PHASE_CONFLICT"
                )
                continue
            phase = next(iter(phase_values))
            mechanism_phase_by_id[parent_id] = phase
            retained_rows = [
                {
                    "mec_id": row.get("mec_id"),
                    "record_id": row.get("record_id"),
                    "molecule_chembl_id": row.get("molecule_chembl_id"),
                    "parent_molecule_chembl_id": row.get(
                        "parent_molecule_chembl_id"
                    ),
                    "canonical_parent_molecule_chembl_id": parent_id,
                    "target_chembl_id": row.get("target_chembl_id"),
                    "action_type": row.get("action_type"),
                    "direct_interaction": row.get("direct_interaction"),
                    "molecular_mechanism": row.get("molecular_mechanism"),
                    "max_phase": row.get("max_phase"),
                    "mechanism_of_action": row.get("mechanism_of_action"),
                    "mechanism_refs": deepcopy(row.get("mechanism_refs")),
                    "variant_sequence": row.get("variant_sequence"),
                }
                for row in rows
            ]
            retained_hash = sha256(
                json.dumps(
                    retained_rows,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            mechanism_provenance_by_id[parent_id] = {
                "schema_version": "chembl_mechanism_identity_v1",
                "parent_molecule_chembl_id": parent_id,
                "row_count": len(retained_rows),
                "rows_sha256": retained_hash,
                "rows": retained_rows,
                "query": mechanism_row_queries[_canonical_sha256(rows[0])].as_dict(),
                **({"queries": [call.as_dict() for call in mechanism_calls
                    if any(mechanism_row_queries[_canonical_sha256(row)] is call for row in rows)]}
                   if len(mechanism_calls) > 1 else {}),
                "fixture": False,
                "missing_values_imputed": False,
            }
            valid_parent_ids.append(parent_id)

        if not valid_parent_ids:
            raise NoMeasuredLigandCandidates(
                target_symbol=normalized.symbol,
                target_uniprot_id=normalized.uniprot_id,
                direction=direction,
                eligible_count=0,
                minimum_required=minimum_required,
                reason_code="NO_VALID_BOUNDED_MECHANISM_IDENTITIES",
                provenance={
                    "chembl_version": self._chembl_version,
                    "chembl_version_resolution": deepcopy(
                        self._chembl_version_resolution
                    ),
                    "target_chembl_id": target_chembl_id,
                    "mechanism_query": mechanism_call.as_dict(),
                    "raw_mechanism_count": len(raw_mechanisms),
                    "mechanism_total_count": mechanism_total_count,
                    "raw_mechanism_rows_sha256": (
                        raw_mechanism_rows_sha256
                    ),
                    "discovered_identity_ids": list(discovered_ids),
                    "invalid_identity_reason_codes_by_id": {
                        parent_id: list(
                            identity_resolution[parent_id]["reason_codes"]
                        )
                        for parent_id in discovered_ids
                    },
                    "malformed_mechanism_rows": malformed_rows,
                    "form_parent_conflicts": form_parent_conflicts,
                    "no_activity_first_fallback": True,
                    "fixture": False,
                },
            )

        molecule_payload, molecule_call = self._get_json(
            self.molecule_endpoint,
            params={
                "molecule_chembl_id__in": ",".join(valid_parent_ids),
                "limit": len(valid_parent_ids),
                "order_by": "molecule_chembl_id",
            },
        )
        molecules = molecule_payload.get("molecules")
        if not isinstance(molecules, list):
            unavailable = ScientificDataUnavailable(
                "ChEMBL molecule response has no molecule list"
            )
            unavailable.provenance = {
                "source": "ChEMBL",
                "operation": "mechanism_first_candidate_discovery",
                "reason_code": "MOLECULE_RESPONSE_LIST_MISSING",
                "chembl_version": self._chembl_version,
                "chembl_version_resolution": deepcopy(
                    self._chembl_version_resolution
                ),
                "target_chembl_id": target_chembl_id,
                "mechanism_query": mechanism_call.as_dict(),
                "raw_mechanism_row_limit": (
                    _MAX_MECHANISM_ROWS_PER_OPERATION
                ),
                "raw_mechanism_response_count": len(raw_mechanisms),
                "mechanism_total_count": mechanism_total_count,
                "raw_mechanism_rows_sha256": raw_mechanism_rows_sha256,
                "discovered_identity_ids": list(discovered_ids),
                "valid_mechanism_identity_ids": list(valid_parent_ids),
                "molecule_query": molecule_call.as_dict(),
                "no_activity_first_fallback": True,
                "fixture": False,
                "missing_values_imputed": False,
            }
            raise unavailable
        molecule_rows_sha256 = sha256(
            json.dumps(
                molecules,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        requested_molecule_ids = set(valid_parent_ids)
        molecule_rows_by_id: dict[str, list[dict[str, Any]]] = {}
        unexpected_molecule_ids: list[str] = []
        for item in molecules:
            if not isinstance(item, dict):
                continue
            chembl_id = str(item.get("molecule_chembl_id") or "").strip()
            if chembl_id not in requested_molecule_ids:
                if chembl_id:
                    unexpected_molecule_ids.append(chembl_id)
                continue
            molecule_rows_by_id.setdefault(chembl_id, []).append(item)
        resolved_molecule_by_id: dict[str, dict[str, Any]] = {}
        ranked_identities: list[tuple[int, str]] = []
        for parent_id in valid_parent_ids:
            rows = molecule_rows_by_id.get(parent_id, [])
            resolution = identity_resolution[parent_id]
            if len(rows) != 1:
                resolution["reason_codes"].append(
                    "MOLECULE_METADATA_NOT_EXACTLY_ONE"
                )
                continue
            molecule = rows[0]
            resolved_molecule_by_id[parent_id] = molecule
            raw_phase_value = molecule.get("max_phase")
            raw_phase = _safe_float(raw_phase_value)
            if (
                isinstance(raw_phase_value, bool)
                or raw_phase is None
                or raw_phase < 1
                or raw_phase > 4
                or not float(raw_phase).is_integer()
            ):
                resolution["reason_codes"].append(
                    "CLINICAL_PHASE_UNRESOLVED"
                )
                continue
            phase = int(raw_phase)
            resolution["max_phase"] = phase
            if phase != mechanism_phase_by_id[parent_id]:
                resolution["reason_codes"].append(
                    "MECHANISM_MOLECULE_PHASE_CONFLICT"
                )
                continue
            if str(molecule.get("molecule_type") or "") != "Small molecule":
                resolution["reason_codes"].append(
                    "MOLECULE_TYPE_NOT_SMALL_MOLECULE"
                )
                continue
            structures = molecule.get("molecule_structures")
            if not isinstance(structures, dict) or not str(
                structures.get("canonical_smiles") or ""
            ).strip():
                resolution["reason_codes"].append(
                    "CANONICAL_SMILES_UNRESOLVED"
                )
                continue
            ranked_identities.append((phase, parent_id))

        ranked_identities.sort(key=lambda item: (-item[0], item[1]))
        selected_ranked_identities = ranked_identities[:candidate_identity_limit]
        selected_ids = [
            parent_id
            for _phase, parent_id in selected_ranked_identities
        ]
        selected_id_set = set(selected_ids)
        for phase, parent_id in ranked_identities:
            selected = parent_id in selected_id_set
            identity_resolution[parent_id].update(
                {
                    "status": "selected" if selected else "excluded",
                    "max_phase": phase,
                    "reason_codes": [
                        "PREACTIVITY_POOL_SELECTED"
                        if selected
                        else "PREACTIVITY_POOL_LIMIT_REACHED"
                    ],
                }
            )

        activity_rows_by_molecule: dict[
            str,
            list[tuple[dict[str, Any], ActivityMeasurement, str]],
        ] = {parent_id: [] for parent_id in selected_ids}
        eligible_activity_count_by_molecule = {
            parent_id: 0 for parent_id in selected_ids
        }
        activity_calls_by_id: dict[str, HTTPCallProvenance] = {}
        activity_page_calls_by_id: dict[
            str, list[HTTPCallProvenance]
        ] = {}
        activity_page_audits_by_id: dict[str, list[dict[str, Any]]] = {}
        activity_row_queries_by_id: dict[
            str,
            dict[str, tuple[HTTPCallProvenance, dict[str, Any]]],
        ] = {}
        activity_failures_by_id: dict[str, dict[str, Any]] = {}
        activity_counts_by_id: dict[str, dict[str, Any]] = {}
        for parent_id in selected_ids:
            activity_params = {
                "target_chembl_id": target_chembl_id,
                "parent_molecule_chembl_id": parent_id,
                "target_organism": "Homo sapiens",
                "assay_type": "B",
                "standard_type": "IC50",
                "pchembl_value__gte": self.min_pchembl_value,
                "order_by": "activity_id",
                "limit": _MAX_ACTIVITY_ROWS_PER_CANDIDATE,
            }
            try:
                if (
                    self.repurposing_evidence_policy
                    in _COMPLETE_ACTIVITY_PAGINATION_POLICIES
                ):
                    (
                        activities,
                        page_calls,
                        page_audits,
                        row_queries,
                        activity_total_count,
                    ) = self._fetch_complete_exact_parent_activity_pages(
                        activity_params
                    )
                    activity_call = page_calls[0]
                    raw_activities = activities
                else:
                    activity_payload, activity_call = self._get_json(
                        self.activity_endpoint,
                        params=activity_params,
                    )
                    raw_activities = activity_payload.get("activities")
                    activity_page_meta = activity_payload.get("page_meta")
                    if not isinstance(raw_activities, list) or not isinstance(
                        activity_page_meta, Mapping
                    ):
                        unavailable = ScientificDataUnavailable(
                            "ChEMBL activity response metadata is missing"
                        )
                        unavailable.provenance = {
                            "reason_code": "ACTIVITY_RESPONSE_METADATA_MISSING",
                            "query": activity_call.as_dict(),
                            "fixture": False,
                        }
                        raise unavailable
                    raw_total = activity_page_meta.get("total_count")
                    if isinstance(raw_total, bool):
                        raw_total = None
                    try:
                        activity_total_count = int(raw_total)
                    except (TypeError, ValueError):
                        activity_total_count = -1
                    if activity_total_count < 0:
                        unavailable = ScientificDataUnavailable(
                            "ChEMBL returned an invalid activity total_count"
                        )
                        unavailable.provenance = {
                            "reason_code": "ACTIVITY_TOTAL_COUNT_INVALID",
                            "query": activity_call.as_dict(),
                            "fixture": False,
                        }
                        raise unavailable
                    activities = raw_activities[
                        :_MAX_ACTIVITY_ROWS_PER_CANDIDATE
                    ]
                    page_calls = [activity_call]
                    row_queries = {
                        str(row.get("activity_id") or "").strip(): (
                            activity_call,
                            deepcopy(activity_params),
                        )
                        for row in activities
                        if isinstance(row, dict)
                        and str(row.get("activity_id") or "").strip()
                    }
                    page_audit = {
                        "page_index": 0,
                        "offset": 0,
                        "limit": _MAX_ACTIVITY_ROWS_PER_CANDIDATE,
                        "response_count": len(activities),
                        "reported_total_count": activity_total_count,
                        "activity_ids": list(row_queries),
                        "activity_ids_sha256": _canonical_sha256(
                            list(row_queries)
                        ),
                        "response_rows_sha256": _canonical_sha256(
                            activities
                        ),
                        "query": activity_call.as_dict(),
                        "query_parameters": deepcopy(activity_params),
                    }
                    page_audit["page_audit_sha256"] = _canonical_sha256(
                        page_audit
                    )
                    page_audits = [page_audit]
            except ScientificDataUnavailable as exc:
                failure = getattr(exc, "provenance", {})
                activity_failures_by_id[parent_id] = (
                    deepcopy(dict(failure))
                    if isinstance(failure, Mapping)
                    else {"exception_type": type(exc).__name__}
                )
                identity_resolution[parent_id]["status"] = "excluded"
                identity_resolution[parent_id]["reason_codes"].append(
                    str(
                        failure.get("reason_code")
                        if isinstance(failure, Mapping)
                        else ""
                    )
                    or "EXACT_PARENT_ACTIVITY_QUERY_UNAVAILABLE"
                )
                continue
            activity_calls_by_id[parent_id] = activity_call
            activity_page_calls_by_id[parent_id] = page_calls
            activity_page_audits_by_id[parent_id] = page_audits
            activity_row_queries_by_id[parent_id] = row_queries
            identity_mismatch_count = 0
            activity_identity_mismatch_count = 0
            target_identity_mismatch_count = 0
            parent_identity_mismatch_count = 0
            endpoint_mismatch_count = 0
            assay_type_mismatch_count = 0
            pchembl_threshold_mismatch_count = 0
            valid_rows: list[
                tuple[dict[str, Any], ActivityMeasurement, str]
            ] = []
            direct_row_count = 0
            for raw in activities:
                if not isinstance(raw, dict):
                    continue
                raw_activity_id = str(raw.get("activity_id") or "").strip()
                try:
                    activity_id_number = int(raw_activity_id)
                except (TypeError, ValueError):
                    activity_id_number = -1
                if (
                    activity_id_number < 1
                    or raw_activity_id != str(activity_id_number)
                ):
                    identity_mismatch_count += 1
                    activity_identity_mismatch_count += 1
                    continue
                row_target_id = str(
                    raw.get("target_chembl_id") or ""
                ).strip()
                if row_target_id != target_chembl_id:
                    identity_mismatch_count += 1
                    target_identity_mismatch_count += 1
                    continue
                row_parent_id = str(
                    raw.get("parent_molecule_chembl_id")
                    or raw.get("molecule_chembl_id")
                    or ""
                ).strip()
                if row_parent_id != parent_id:
                    identity_mismatch_count += 1
                    parent_identity_mismatch_count += 1
                    continue
                if raw.get("standard_type") != "IC50":
                    endpoint_mismatch_count += 1
                    continue
                if raw.get("assay_type") != "B":
                    assay_type_mismatch_count += 1
                    continue
                raw_pchembl = raw.get("pchembl_value")
                if isinstance(raw_pchembl, bool):
                    pchembl_threshold_mismatch_count += 1
                    continue
                try:
                    deposited_pchembl = Decimal(str(raw_pchembl))
                except (InvalidOperation, ValueError):
                    deposited_pchembl = Decimal("NaN")
                if (
                    not deposited_pchembl.is_finite()
                    or deposited_pchembl < Decimal(
                        str(self.min_pchembl_value)
                    )
                ):
                    pchembl_threshold_mismatch_count += 1
                    continue
                if "data_validity_comment" not in raw or raw[
                    "data_validity_comment"
                ] not in (None, "", "Manually validated"):
                    continue
                measurement = _normalize_activity(raw)
                if measurement is None:
                    continue
                direction_rule = _measurement_direction_rule(
                    measurement, direction
                )
                if direction_rule is None:
                    continue
                valid_rows.append((raw, measurement, direction_rule))
                if _assay_supports_direct_target_engagement(
                    measurement.assay_type
                ):
                    direct_row_count += 1
            eligible_activity_count_by_molecule[parent_id] = len(valid_rows)
            valid_rows.sort(
                key=lambda item: (
                    -(_safe_float(item[0].get("pchembl_value")) or 0.0),
                    int(str(item[0].get("activity_id") or "0")),
                )
            )
            complete_pagination = (
                self.repurposing_evidence_policy
                in _COMPLETE_ACTIVITY_PAGINATION_POLICIES
            )
            paper_native_v6_full_primary_audit = (
                self.repurposing_evidence_policy
                == PAPER_NATIVE_V6_POLICY_TOKEN
            )
            activity_rows_by_molecule[parent_id] = (
                valid_rows
                if paper_native_v6_full_primary_audit
                else valid_rows[:_PRIMARY_ACTIVITY_DOCUMENT_TOP_K]
            )
            eligible_directional_activity_ids = [
                str(item[0].get("activity_id")) for item in valid_rows
            ]
            retained_document_activity_ids = [
                str(item[0].get("activity_id"))
                for item in activity_rows_by_molecule[parent_id]
            ]
            activity_counts_by_id[parent_id] = {
                "total_count": activity_total_count,
                "response_count": (
                    len(activities)
                    if complete_pagination
                    else len(raw_activities)
                ),
                "response_sha256": activity_call.response_sha256,
                "response_rows_sha256": sha256(
                    json.dumps(
                        (
                            activities
                            if complete_pagination
                            else raw_activities
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "inspected_count": len(activities),
                "inspected_cap": (
                    None
                    if complete_pagination
                    else _MAX_ACTIVITY_ROWS_PER_CANDIDATE
                ),
                "inspected_rows_sha256": sha256(
                    json.dumps(
                        activities,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    ).hexdigest(),
                "truncated": (
                    False
                    if complete_pagination
                    else activity_total_count > len(activities)
                ),
                "complete_result_set": (
                    complete_pagination
                    and len(activities) == activity_total_count
                ),
                "page_size": _MAX_ACTIVITY_ROWS_PER_CANDIDATE,
                "page_count": len(page_calls),
                "page_audits": deepcopy(page_audits),
                "page_audits_sha256": _canonical_sha256(page_audits),
                "identity_mismatch_count": identity_mismatch_count,
                "activity_identity_mismatch_count": (
                    activity_identity_mismatch_count
                ),
                "target_identity_mismatch_count": (
                    target_identity_mismatch_count
                ),
                "parent_identity_mismatch_count": (
                    parent_identity_mismatch_count
                ),
                "endpoint_mismatch_count": endpoint_mismatch_count,
                "assay_type_mismatch_count": assay_type_mismatch_count,
                "pchembl_threshold_mismatch_count": (
                    pchembl_threshold_mismatch_count
                ),
                "eligible_directional_count": len(valid_rows),
                "eligible_directional_activity_ids": (
                    eligible_directional_activity_ids
                ),
                "eligible_directional_activity_ids_sha256": (
                    _canonical_sha256(eligible_directional_activity_ids)
                ),
                "direct_single_protein_count": direct_row_count,
                "retained_document_rows": len(
                    activity_rows_by_molecule[parent_id]
                ),
                "retained_document_activity_ids": (
                    retained_document_activity_ids
                ),
                "retained_document_activity_ids_sha256": (
                    _canonical_sha256(retained_document_activity_ids)
                ),
                "primary_document_audit_result_cap": (
                    None
                    if paper_native_v6_full_primary_audit
                    else _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
                ),
                "all_eligible_rows_retained_for_primary_audit": (
                    paper_native_v6_full_primary_audit
                    and len(activity_rows_by_molecule[parent_id])
                    == len(valid_rows)
                ),
                "query": activity_call.as_dict(),
            }
            if not valid_rows:
                identity_resolution[parent_id]["status"] = "excluded"
                identity_resolution[parent_id]["reason_codes"].append(
                    "NO_EXACT_PARENT_DIRECTION_COMPATIBLE_ACTIVITY"
                )

        fixed_pool_with_activity = [
            parent_id
            for parent_id in selected_ids
            if activity_rows_by_molecule[parent_id]
        ]
        mechanism_identity_hash = sha256(
            json.dumps(
                {
                    parent_id: mechanism_provenance_by_id[parent_id][
                        "rows_sha256"
                    ]
                    for parent_id in sorted(mechanism_provenance_by_id)
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        eligible_identity_ids = [
            parent_id for _phase, parent_id in ranked_identities
        ]
        valid_parent_id_set = set(valid_parent_ids)
        selected_id_set = set(selected_ids)
        candidate_discovery_bounds = {
            "schema_version": (
                "bounded_mechanism_first_candidate_identity_pool_v1"
            ),
            "discovery_lane": "chembl_mechanism_first_single_default_v1",
            "chembl_version": self._chembl_version,
            "chembl_version_resolution": deepcopy(
                self._chembl_version_resolution
            ),
            "target_chembl_id": target_chembl_id,
            "target_type": target_type,
            "target_organism": target_organism,
            "mechanism_action_type": "INHIBITOR",
            "mechanism_query": mechanism_call.as_dict(),
            "molecule_query": molecule_call.as_dict(),
            "raw_mechanism_row_limit": mechanism_row_limit,
            **({"mechanism_queries": [call.as_dict() for call in mechanism_calls],
                "mechanism_page_count": len(mechanism_calls)} if len(mechanism_calls) > 1 else {}),
            "raw_mechanism_response_count": len(raw_mechanisms),
            "mechanism_total_count": mechanism_total_count,
            "mechanism_complete_within_bound": True,
            "raw_mechanism_rows_sha256": raw_mechanism_rows_sha256,
            "mechanism_identity_sha256": mechanism_identity_hash,
            "mechanism_parent_dedup_policy": (
                "parent_molecule_chembl_id_else_form_id; one canonical parent "
                "identity; conflicting form-parent mappings excluded"
            ),
            "malformed_mechanism_rows": malformed_rows,
            "form_parent_conflicts": form_parent_conflicts,
            "discovered_identity_count": len(discovered_ids),
            "discovered_identity_ids": list(discovered_ids),
            "valid_mechanism_identity_count": len(valid_parent_ids),
            "valid_mechanism_identity_ids": list(valid_parent_ids),
            "invalid_mechanism_identity_ids": sorted(
                set(discovered_ids).difference(valid_parent_id_set)
            ),
            "molecule_resolution_batch_count": 1,
            "molecule_response_rows_sha256": molecule_rows_sha256,
            "molecule_resolution_requested_count": len(valid_parent_ids),
            "molecule_resolution_requested_ids": list(valid_parent_ids),
            "molecule_resolution_count": len(resolved_molecule_by_id),
            "molecule_resolution_unresolved_ids": sorted(
                requested_molecule_ids.difference(resolved_molecule_by_id)
            ),
            "molecule_resolution_unexpected_ids": sorted(
                set(unexpected_molecule_ids)
            ),
            "eligible_identity_count": len(ranked_identities),
            "eligible_identity_ids": eligible_identity_ids,
            "prewarning_pool_limit": candidate_identity_limit,
            "prewarning_pool_count": len(selected_ids),
            "prewarning_pool_ids": list(selected_ids),
            "preactivity_pool_count": len(selected_ids),
            "preactivity_pool_ids": list(selected_ids),
            "preactivity_pool": [
                {
                    "rank": rank,
                    "parent_molecule_chembl_id": parent_id,
                    "max_phase": phase,
                }
                for rank, (phase, parent_id) in enumerate(
                    selected_ranked_identities,
                    start=1,
                )
            ],
            "preactivity_excluded_identity_ids": sorted(
                set(discovered_ids).difference(selected_id_set)
            ),
            "preactivity_exclusion_reason_codes_by_id": {
                parent_id: list(
                    identity_resolution[parent_id]["reason_codes"]
                )
                for parent_id in discovered_ids
                if parent_id not in selected_id_set
            },
            "prewarning_selection_policy": (
                "clinical_max_phase_desc_then_parent_molecule_chembl_id_v1"
            ),
            "selection_inputs": ["max_phase", "parent_molecule_chembl_id"],
            "pchembl_used_for_prewarning_selection": False,
            "primary_document_used_for_prewarning_selection": False,
            "warning_used_for_prewarning_selection": False,
            "activity_used_for_prewarning_selection": False,
            "activity_query_policy": (
                (
                    "complete_exact_parent_activity_pagination_v2; 20-row "
                    "wire pages; all reported rows inspected; IC50 direction "
                    "filter; no identity replacement; partial pages discarded"
                )
                if self.repurposing_evidence_policy
                in _COMPLETE_ACTIVITY_PAGINATION_POLICIES
                else (
                    "one_exact_parent_query_per_fixed_identity; <=20 rows "
                    "each; IC50 direction filter; no identity replacement"
                )
            ),
            "activity_complete_pagination_required": (
                self.repurposing_evidence_policy
                in _COMPLETE_ACTIVITY_PAGINATION_POLICIES
            ),
            "activity_rows_per_candidate_limit": (
                _MAX_ACTIVITY_ROWS_PER_CANDIDATE
            ),
            "activity_rows_per_candidate_limit_semantics": (
                "wire_page_size_not_result_cap"
                if self.repurposing_evidence_policy
                in _COMPLETE_ACTIVITY_PAGINATION_POLICIES
                else "result_cap"
            ),
            "activity_result_row_limit": (
                None
                if self.repurposing_evidence_policy
                in _COMPLETE_ACTIVITY_PAGINATION_POLICIES
                else _MAX_ACTIVITY_ROWS_PER_CANDIDATE
            ),
            "activity_candidate_query_limit": candidate_identity_limit,
            "activity_logical_call_count": sum(
                len(calls) for calls in activity_page_calls_by_id.values()
            ),
            "activity_query_attempted_count": len(selected_ids),
            "activity_query_attempted_ids": list(selected_ids),
            "activity_query_success_count": len(activity_calls_by_id),
            "activity_failure_count": len(activity_failures_by_id),
            "activity_failures_by_id": deepcopy(activity_failures_by_id),
            "activity_counts_by_id": deepcopy(activity_counts_by_id),
            "activity_page_audits_by_id": deepcopy(
                activity_page_audits_by_id
            ),
            "activity_page_audits_by_id_sha256": _canonical_sha256(
                activity_page_audits_by_id
            ),
            "fixed_pool_with_activity_count": len(fixed_pool_with_activity),
            "fixed_pool_with_activity_ids": fixed_pool_with_activity,
            "mechanism_provenance_by_id": deepcopy(
                mechanism_provenance_by_id
            ),
            "identity_resolution": [
                deepcopy(identity_resolution[parent_id])
                for parent_id in discovered_ids
            ],
            "no_activity_first_fallback": True,
            "no_backfill_beyond_prewarning_pool": True,
            "returned_candidate_ordering_policy": (
                "surviving_fixed_prewarning_clinical_phase_desc_then_"
                "stable_parent_molecule_chembl_id_subsequence_v1"
            ),
            "fixture": False,
            "missing_values_imputed": False,
        }
        if len(fixed_pool_with_activity) < minimum_required:
            raise NoMeasuredLigandCandidates(
                target_symbol=normalized.symbol,
                target_uniprot_id=normalized.uniprot_id,
                direction=direction,
                eligible_count=len(fixed_pool_with_activity),
                minimum_required=minimum_required,
                reason_code=(
                    "INSUFFICIENT_FIXED_POOL_EXACT_PARENT_ACTIVITY"
                ),
                provenance={
                    "target_chembl_id": target_chembl_id,
                    "mechanism_query": mechanism_call.as_dict(),
                    "molecule_query": molecule_call.as_dict(),
                    "candidate_discovery_bounds": deepcopy(
                        candidate_discovery_bounds
                    ),
                    "no_activity_first_fallback": True,
                    "fixture": False,
                    "missing_values_imputed": False,
                },
            )

        return {
            "normalized": normalized,
            "target_chembl_id": target_chembl_id,
            "discovered_ids": discovered_ids,
            "selected_ids": selected_ids,
            "resolved_molecule_by_id": resolved_molecule_by_id,
            "identity_resolution": identity_resolution,
            "candidate_discovery_bounds": candidate_discovery_bounds,
            "activity_rows_by_molecule": activity_rows_by_molecule,
            "eligible_activity_count_by_molecule": (
                eligible_activity_count_by_molecule
            ),
            "activity_calls_by_id": activity_calls_by_id,
            "activity_page_calls_by_id": activity_page_calls_by_id,
            "activity_row_queries_by_id": activity_row_queries_by_id,
            "activity_query_fallbacks_by_id": {
                parent_id: None for parent_id in selected_ids
            },
            "activity_failures_by_id": activity_failures_by_id,
            "molecule_call": molecule_call,
            "molecule_query_scope_ids": list(valid_parent_ids),
            "mechanism_provenance_by_id": mechanism_provenance_by_id,
        }

    def _fetch_measured_ligands(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        fetch_limit: int,
        minimum_required: int,
        require_clinical: bool,
        candidate_identity_limit: int | None = None,
        discovery_lane: str = "activity_first",
    ) -> list[_MeasuredLigand]:
        if candidate_identity_limit is not None and candidate_identity_limit < 1:
            raise ValueError("candidate identity limit must be positive")
        if discovery_lane == "mechanism_first":
            if candidate_identity_limit is None:
                raise ValueError(
                    "mechanism-first discovery requires a fixed identity limit"
                )
            discovery = self._discover_mechanism_first_ligands(
                target,
                direction=direction,
                minimum_required=minimum_required,
                candidate_identity_limit=candidate_identity_limit,
            )
        elif discovery_lane == "activity_first":
            discovery = self._discover_activity_first_ligands(
                target,
                direction=direction,
                fetch_limit=fetch_limit,
                minimum_required=minimum_required,
                require_clinical=require_clinical,
                candidate_identity_limit=candidate_identity_limit,
            )
        else:
            raise ValueError(f"unsupported ligand discovery lane: {discovery_lane}")

        normalized = discovery["normalized"]
        target_chembl_id = discovery["target_chembl_id"]
        discovered_ids = discovery["discovered_ids"]
        selected_ids = discovery["selected_ids"]
        resolved_molecule_by_id = discovery["resolved_molecule_by_id"]
        identity_resolution = discovery["identity_resolution"]
        candidate_discovery_bounds = discovery["candidate_discovery_bounds"]
        activity_rows_by_molecule = discovery["activity_rows_by_molecule"]
        eligible_activity_count_by_molecule = discovery[
            "eligible_activity_count_by_molecule"
        ]
        activity_calls_by_id = discovery["activity_calls_by_id"]
        activity_page_calls_by_id = discovery.get(
            "activity_page_calls_by_id", {}
        )
        activity_row_queries_by_id = discovery.get(
            "activity_row_queries_by_id", {}
        )
        activity_query_fallbacks_by_id = discovery[
            "activity_query_fallbacks_by_id"
        ]
        activity_failures_by_id = discovery.get(
            "activity_failures_by_id", {}
        )
        molecule_call = discovery["molecule_call"]
        molecule_query_scope_ids = discovery.get(
            "molecule_query_scope_ids", discovered_ids
        )
        mechanism_provenance_by_id = discovery[
            "mechanism_provenance_by_id"
        ]
        molecule_by_id = {
            chembl_id: resolved_molecule_by_id[chembl_id]
            for chembl_id in selected_ids
        }
        aliases_by_id = {
            chembl_id: _chembl_exact_aliases(molecule)
            for chembl_id, molecule in molecule_by_id.items()
        }
        alias_owners: dict[str, set[str]] = {}
        for chembl_id, aliases in aliases_by_id.items():
            for alias in aliases:
                normalized_alias = _normalize_identity_phrase(alias)
                if normalized_alias:
                    alias_owners.setdefault(normalized_alias, set()).add(
                        chembl_id
                    )
        alias_conflicts_by_id: dict[str, list[str]] = {
            chembl_id: sorted(
                alias
                for alias in aliases
                if len(
                    alias_owners.get(_normalize_identity_phrase(alias), set())
                )
                > 1
            )
            for chembl_id, aliases in aliases_by_id.items()
        }
        alias_inventory = _candidate_pool_alias_inventory(
            aliases_by_id,
            expected_molecule_ids=selected_ids,
        )
        fixed_pool_molecule_projection = _fixed_pool_molecule_projection(
            fixed_pool_ids=selected_ids,
            molecules_by_id=molecule_by_id,
            aliases_by_id=aliases_by_id,
            molecule_query=molecule_call,
        )
        try:
            (
                warnings_by_id,
                warning_calls_by_id,
                warning_failures_by_id,
                warning_batch_fallback_used,
                warning_batch_failure,
            ) = self._fetch_drug_warnings(selected_ids)
        except ScientificDataUnavailable as exc:
            failure = getattr(exc, "provenance", {})
            failure = (
                deepcopy(dict(failure))
                if isinstance(failure, Mapping)
                else {"exception_type": type(exc).__name__}
            )
            failure["candidate_discovery_bounds"] = deepcopy(
                candidate_discovery_bounds
            )
            failure["warning_screen_identity_count"] = len(selected_ids)
            failure["warning_screen_identity_ids"] = list(selected_ids)
            failure["warning_screen_identity_limit"] = candidate_identity_limit
            failure["missing_values_imputed"] = False
            exc.provenance = failure
            raise
        downstream_candidate_ids: list[str] = []
        for chembl_id in selected_ids:
            resolution = identity_resolution[chembl_id]
            if chembl_id in warning_failures_by_id:
                resolution["warning_screen_status"] = "unresolved"
                resolution["downstream_status"] = "excluded"
                resolution["reason_codes"].append(
                    "REGULATORY_WARNING_STATUS_UNRESOLVED"
                )
                continue
            resolution["warning_screen_status"] = "resolved"
            if not activity_rows_by_molecule.get(chembl_id):
                resolution["downstream_status"] = "excluded"
                continue
            molecule = molecule_by_id[chembl_id]
            structures = molecule.get("molecule_structures") or {}
            if not isinstance(structures, dict):
                resolution["downstream_status"] = "excluded"
                resolution["reason_codes"].append(
                    "MOLECULE_STRUCTURE_METADATA_UNRESOLVED"
                )
                continue
            if not str(structures.get("canonical_smiles") or "").strip():
                resolution["downstream_status"] = "excluded"
                resolution["reason_codes"].append(
                    "CANONICAL_SMILES_UNRESOLVED"
                )
                continue
            resolution["downstream_status"] = "evidence_resolution_eligible"
            resolution["reason_codes"].append(
                "WARNING_AND_STRUCTURE_STATUS_RESOLVED"
            )
            downstream_candidate_ids.append(chembl_id)

        downstream_id_set = set(downstream_candidate_ids)
        activity_rows_by_molecule = {
            chembl_id: rows
            for chembl_id, rows in activity_rows_by_molecule.items()
            if chembl_id in downstream_id_set
        }
        eligible_activity_count_by_molecule = {
            chembl_id: count
            for chembl_id, count in eligible_activity_count_by_molecule.items()
            if chembl_id in downstream_id_set
        }
        paper_native_v5_active = bool(
            discovery_lane == "mechanism_first"
            and self.repurposing_evidence_policy
            == PAPER_NATIVE_V5_POLICY_TOKEN
        )
        paper_native_v6_active = bool(
            discovery_lane == "mechanism_first"
            and self.repurposing_evidence_policy
            == PAPER_NATIVE_V6_POLICY_TOKEN
        )
        paper_native_v5_legacy_lookup_attestation: dict[str, Any] | None = None
        if paper_native_v5_active:
            # The v5 contract owns one dedicated, bounded PubMed search.  It
            # must never execute the legacy ChEMBL-document or PubMed-type
            # enrichment path, even when retained activities carry real
            # document_chembl_id values.
            document_ids: list[str] = []
            document_candidate_ids: list[str] = []
            documents_by_id: dict[str, dict[str, Any]] = {}
            document_calls_by_id: dict[str, HTTPCallProvenance] = {}
            document_failures_by_id: dict[str, dict[str, Any]] = {}
            document_query_batches_by_id: dict[str, dict[str, Any]] = {}
            pubmed_types_by_pmid: dict[str, dict[str, Any]] = {}
            pubmed_candidate_ids: list[str] = []
            paper_native_v5_legacy_lookup_attestation = {
                "legacy_primary_document_lookup_executed": False,
                "legacy_primary_document_lookup_candidate_count": 0,
                "legacy_primary_document_lookup_candidate_ids": [],
                "legacy_primary_document_lookup_document_count": 0,
                "legacy_primary_document_lookup_document_ids": [],
                "legacy_pubmed_type_lookup_executed": False,
                "legacy_pubmed_type_lookup_candidate_count": 0,
                "legacy_pubmed_type_lookup_candidate_ids": [],
                "legacy_pubmed_type_lookup_pmid_count": 0,
                "legacy_pubmed_type_lookup_pmid_ids": [],
            }
        else:
            document_ids = sorted(
                {
                    str(raw.get("document_chembl_id") or "").strip()
                    for rows in activity_rows_by_molecule.values()
                    for raw, _measurement, _direction_rule in rows
                    if _CHEMBL_STABLE_ID.fullmatch(
                        str(raw.get("document_chembl_id") or "").strip()
                    )
                }
            )
            document_candidate_ids = sorted(
                chembl_id
                for chembl_id, rows in activity_rows_by_molecule.items()
                if any(
                    _CHEMBL_STABLE_ID.fullmatch(
                        str(raw.get("document_chembl_id") or "").strip()
                    )
                    for raw, _measurement, _direction_rule in rows
                )
            )
            (
                documents_by_id,
                document_calls_by_id,
                document_failures_by_id,
                document_query_batches_by_id,
            ) = self._fetch_primary_documents(document_ids)
            pubmed_types_by_pmid = self._fetch_pubmed_publication_types(
                documents_by_id
            )
            pubmed_document_ids = {
                document_id
                for document_id, document in documents_by_id.items()
                if str(document.get("pubmed_id") or "").strip().isdigit()
            }
            pubmed_candidate_ids = sorted(
                chembl_id
                for chembl_id, rows in activity_rows_by_molecule.items()
                if any(
                    str(raw.get("document_chembl_id") or "").strip()
                    in pubmed_document_ids
                    for raw, _measurement, _direction_rule in rows
                )
            )
        assay_ids = sorted(
            {
                str(raw.get("assay_chembl_id") or "").strip()
                for rows in activity_rows_by_molecule.values()
                for raw, _measurement, _direction_rule in rows
                if str(raw.get("assay_chembl_id") or "").strip()
            }
        )
        assay_candidate_ids = sorted(
            {
                chembl_id
                for chembl_id, rows in activity_rows_by_molecule.items()
                if any(
                    str(raw.get("assay_chembl_id") or "").strip()
                    for raw, _measurement, _direction_rule in rows
                )
            }
        )
        assay_contexts = self._fetch_assay_contexts(assay_ids)
        paired_v4_promotions: dict[str, dict[str, Any]] = {}
        paired_v4_inventory: dict[str, Any] | None = None
        paired_v4_outcome_payload: dict[str, Any] | None = None
        paired_v4_promotion_summary: dict[str, Any] | None = None
        paper_native_v5_support_inventory: dict[str, Any] | None = None
        paper_native_v5_support_by_id: dict[str, dict[str, Any]] = {}
        paper_native_v5_alias_inventory: dict[str, Any] | None = None
        paper_native_v5_outcome_payload: dict[str, Any] | None = None
        paper_native_v5_promotion_summary: dict[str, Any] | None = None
        paper_native_v5_promotions: dict[str, dict[str, Any]] = {}
        if paper_native_v5_active:
            (
                paper_native_v5_support_inventory,
                paper_native_v5_support_by_id,
            ) = _paper_native_v5_direct_support_inventory(
                target=normalized,
                target_chembl_id=target_chembl_id,
                fixed_pool_ids=selected_ids,
                activity_rows_by_molecule=activity_rows_by_molecule,
                assay_contexts=assay_contexts,
                activity_calls_by_id=activity_calls_by_id,
                activity_row_queries_by_id=activity_row_queries_by_id,
                identity_resolution=identity_resolution,
                mechanism_provenance_by_id=mechanism_provenance_by_id,
                molecules_by_id=molecule_by_id,
                activity_endpoint=self.activity_endpoint,
                min_pchembl_value=self.min_pchembl_value,
                direction=direction,
            )
            direct_support_ids = [
                chembl_id
                for chembl_id in selected_ids
                if chembl_id in paper_native_v5_support_by_id
            ]
            (
                paper_native_v5_alias_inventory,
                paper_native_v5_search_candidates,
            ) = _paper_native_v5_alias_inventory(
                target=normalized,
                fixed_pool_ids=selected_ids,
                direct_support_candidate_ids=direct_support_ids,
                molecules_by_id=molecule_by_id,
            )
            paper_native_v5_outcome: PaperNativeV5SearchOutcome | None
            if len(direct_support_ids) < 3:
                paper_native_v5_outcome = None
                paper_native_v5_outcome_payload = (
                    _paper_native_v5_not_attempted_payload(
                        reason_code=(
                            "FEWER_THAN_THREE_PRE_PUBMED_DIRECT_SUPPORT_CANDIDATES"
                        ),
                        requested_candidate_count=len(direct_support_ids),
                    )
                )
            elif paper_native_v5_alias_inventory.get("complete") is not True:
                paper_native_v5_outcome = None
                paper_native_v5_outcome_payload = (
                    _paper_native_v5_not_attempted_payload(
                        reason_code="PRE_PUBMED_ALIAS_INVENTORY_INCOMPLETE",
                        requested_candidate_count=len(direct_support_ids),
                    )
                )
            else:
                paper_native_v5_outcome = (
                    search_paper_native_primary_abstract_v5(
                        self.primary_search_client,  # type: ignore[arg-type]
                        target_aliases=tuple(
                            paper_native_v5_alias_inventory["target_aliases"]
                        ),
                        endpoint=str(
                            paper_native_v5_alias_inventory["endpoint"]
                        ),
                        candidates=paper_native_v5_search_candidates,
                        max_candidates=_PRIMARY_PAPER_NATIVE_V5_MAX_CANDIDATES,
                        max_documents=_PRIMARY_PAPER_NATIVE_V5_MAX_DOCUMENTS,
                    )
                )
                paper_native_v5_outcome_payload = (
                    _paper_native_v5_outcome_payload(paper_native_v5_outcome)
                )
            (
                paper_native_v5_promotions,
                paper_native_v5_promotion_summary,
            ) = _paper_native_v5_promotions(
                outcome=paper_native_v5_outcome,
                outcome_payload=paper_native_v5_outcome_payload,
                fixed_pool_ids=selected_ids,
                support_by_id=paper_native_v5_support_by_id,
            )
            support_snapshots = [
                deepcopy(paper_native_v5_support_by_id[chembl_id])
                for chembl_id in direct_support_ids
            ]
            primary_search_provenance = {
                "schema_version": "primary_abstract_paper_native_gateway_v5",
                "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
                "transport": {
                    "dedicated_from_chembl_client": (
                        self.primary_search_client is not self.client
                    ),
                    "max_attempts": getattr(
                        self.primary_search_client, "max_attempts", None
                    ),
                    "required_max_attempts": 1,
                    "logical_call_limit": 3,
                },
                "fixed_pool_candidate_ids": list(selected_ids),
                "fixed_pool_candidate_ids_sha256": _canonical_sha256(
                    list(selected_ids)
                ),
                "direct_support_inventory": deepcopy(
                    paper_native_v5_support_inventory
                ),
                "alias_inventory": deepcopy(paper_native_v5_alias_inventory),
                "search_outcome": deepcopy(paper_native_v5_outcome_payload),
                "promotion": deepcopy(paper_native_v5_promotion_summary),
                "pubmed_or_result_used_for_support_selection": False,
                "pubmed_value_used_for_activity_selection": False,
                "cross_source_numeric_equality_required": False,
                "no_paired_v4_search_or_fallback": True,
                "no_reordering": True,
                "no_backfill": True,
                **deepcopy(paper_native_v5_legacy_lookup_attestation or {}),
                "fixture": False,
            }
            primary_search_provenance["provenance_sha256"] = (
                _canonical_sha256(primary_search_provenance)
            )
            candidate_discovery_bounds[
                "paper_native_v5_direct_support_inventory"
            ] = deepcopy(paper_native_v5_support_inventory)
            candidate_discovery_bounds[
                "paper_native_v5_direct_support_snapshots"
            ] = support_snapshots
            candidate_discovery_bounds[
                "paper_native_v5_direct_support_candidate_ids"
            ] = direct_support_ids
            candidate_discovery_bounds[
                "paper_native_v5_direct_support_snapshots_sha256"
            ] = _canonical_sha256(support_snapshots)
            candidate_discovery_bounds[
                "paper_native_v5_fixed_pool_candidate_ids_sha256"
            ] = _canonical_sha256(list(selected_ids))
            candidate_discovery_bounds[
                "paper_native_v5_pubmed_used_for_support_selection"
            ] = False
            candidate_discovery_bounds[
                "paper_native_v5_pubmed_value_used_for_ordering"
            ] = False
            candidate_discovery_bounds[
                "paper_native_v5_no_reordering"
            ] = True
            candidate_discovery_bounds[
                "paper_native_v5_no_backfill"
            ] = True
            candidate_discovery_bounds.update(
                deepcopy(paper_native_v5_legacy_lookup_attestation or {})
            )
            candidate_discovery_bounds["primary_abstract_search"] = (
                primary_search_provenance
            )
        elif paper_native_v6_active:
            # V6 binds one independently verified primary record per
            # candidate.  It must never run or inherit paired-v4's
            # same-publication cohort search, promotion, or retained-index
            # selection.
            candidate_discovery_bounds[
                "paper_native_v6_delegate_primary_selection"
            ] = {
                "schema_version": "paper_native_v6_delegate_primary_selection_v1",
                "policy": PAPER_NATIVE_V6_POLICY_TOKEN,
                "selection_scope": "candidate_specific_verified_primary_record",
                "paired_v4_search_executed": False,
                "paired_v4_promotion_used": False,
                "paired_v4_retained_index_used": False,
                "candidate_ordering_changed": False,
                "fixture": False,
            }
        elif discovery_lane == "mechanism_first":
            frozen_cohort = _paired_v4_freeze_assay_cohort(
                target=normalized,
                target_chembl_id=target_chembl_id,
                selected_ids=selected_ids,
                activity_rows_by_molecule=activity_rows_by_molecule,
                assay_contexts=assay_contexts,
            )
            (
                paired_v4_inventory,
                paired_v4_search_candidates,
            ) = _paired_v4_query_inventory(
                target=normalized,
                selected_ids=selected_ids,
                activity_rows_by_molecule=activity_rows_by_molecule,
                molecules_by_id=molecule_by_id,
                aliases_by_id=aliases_by_id,
                alias_conflicts_by_id=alias_conflicts_by_id,
                frozen_cohort=frozen_cohort,
            )
            if frozen_cohort.get("selection_status") != "selected":
                paired_v4_outcome = _paired_v4_not_attempted_outcome(
                    reason_code=str(
                        frozen_cohort.get("reason_code")
                        or "PRE_PUBMED_ASSAY_COHORT_NOT_SELECTED"
                    ),
                    requested_candidate_count=0,
                )
            elif paired_v4_inventory.get("complete") is not True:
                paired_v4_outcome = _paired_v4_not_attempted_outcome(
                    reason_code="PRE_PUBMED_ALIAS_INVENTORY_INCOMPLETE",
                    requested_candidate_count=int(
                        frozen_cohort.get("selected_candidate_count") or 0
                    ),
                )
            else:
                paired_v4_outcome = search_primary_abstract_pairs(
                    self.primary_search_client,  # type: ignore[arg-type]
                    target_aliases=tuple(
                        paired_v4_inventory["target_aliases"]
                    ),
                    endpoint=str(paired_v4_inventory["endpoint"]),
                    candidates=paired_v4_search_candidates,
                    max_candidates=_PRIMARY_PAIRED_V4_MAX_CANDIDATES,
                    max_documents=_PRIMARY_PAIRED_V4_MAX_DOCUMENTS,
                )
            paired_v4_outcome_payload = (
                _paired_v4_search_outcome_payload(paired_v4_outcome)
            )
            (
                paired_v4_promotions,
                paired_v4_promotion_summary,
            ) = _paired_v4_promotions(
                outcome=paired_v4_outcome,
                inventory=paired_v4_inventory,
                target=normalized,
                target_chembl_id=target_chembl_id,
                activity_rows_by_molecule=activity_rows_by_molecule,
                assay_contexts=assay_contexts,
            )
            primary_search_provenance: dict[str, Any] = {
                "schema_version": "primary_abstract_search_gateway_v1",
                "policy": _PRIMARY_PAIRED_V4_POLICY,
                "transport": {
                    "dedicated_from_chembl_client": (
                        self.primary_search_client is not self.client
                    ),
                    "max_attempts": getattr(
                        self.primary_search_client, "max_attempts", None
                    ),
                    "required_max_attempts": 1,
                    "logical_call_limit": 3,
                },
                "alias_inventory": deepcopy(paired_v4_inventory),
                "search_outcome": deepcopy(paired_v4_outcome_payload),
                "promotion": deepcopy(paired_v4_promotion_summary),
                "pre_pubmed_cohort_selection": deepcopy(frozen_cohort),
                "pubmed_or_result_used_for_cohort_selection": False,
                "no_backfill": True,
                "fixture": False,
            }
            primary_search_provenance["provenance_sha256"] = (
                _canonical_sha256(primary_search_provenance)
            )
            candidate_discovery_bounds[
                "paired_v4_direct_exact_pool_count"
            ] = paired_v4_inventory["paired_v4_direct_exact_pool_count"]
            candidate_discovery_bounds[
                "paired_v4_direct_exact_pool_ids"
            ] = list(paired_v4_inventory["paired_v4_direct_exact_pool_ids"])
            candidate_discovery_bounds["primary_abstract_search"] = (
                primary_search_provenance
            )
        warning_logical_call_count = (
            1
            if not warning_batch_fallback_used
            else 1
            + len(warning_calls_by_id)
            + sum(
                1
                for failure in warning_failures_by_id.values()
                if failure.get("endpoint")
            )
        )
        warning_screen_summary = {
            "batch_fallback_used": warning_batch_fallback_used,
            "batch_failure": deepcopy(warning_batch_failure),
            "requested_candidate_count": len(selected_ids),
            "requested_candidate_ids": list(selected_ids),
            "screened_candidate_count": len(warning_calls_by_id),
            "unresolved_candidate_count": len(warning_failures_by_id),
            "unresolved_candidate_ids": sorted(warning_failures_by_id),
            "logical_call_count": warning_logical_call_count,
            "logical_call_limit": 1 + len(selected_ids),
            "logical_call_limit_satisfied": (
                warning_logical_call_count <= 1 + len(selected_ids)
            ),
            "fixture": False,
            "missing_values_imputed": False,
        }

        def source_stage(
            *,
            scope_ids: Sequence[str],
            calls: Iterable[HTTPCallProvenance | Mapping[str, Any]],
            failures: Sequence[Mapping[str, Any]],
            recovery_policy: str,
        ) -> dict[str, Any]:
            unique_calls = _deduplicated_call_provenance(calls)
            stage = {
                "scope_ids": list(scope_ids),
                "scope_count": len(scope_ids),
                "successful_calls": unique_calls,
                "successful_call_count": len(unique_calls),
                "failures": [deepcopy(dict(item)) for item in failures],
                "failure_count": len(failures),
                "recovery_policy": recovery_policy,
            }
            stage["stage_sha256"] = _canonical_sha256(stage)
            return stage

        activity_failure_records = [
            {
                "scope_id": chembl_id,
                "failure": deepcopy(activity_failures_by_id[chembl_id]),
            }
            for chembl_id in sorted(activity_failures_by_id)
        ]
        document_failure_records = [
            {
                "scope_id": document_id,
                "failure": deepcopy(document_failures_by_id[document_id]),
            }
            for document_id in sorted(document_failures_by_id)
        ]
        assay_success_calls: list[Mapping[str, Any]] = []
        assay_failure_records: list[dict[str, Any]] = []
        for assay_id in assay_ids:
            context = assay_contexts.get(assay_id, {})
            query = context.get("assay_query") if isinstance(context, Mapping) else None
            if isinstance(context, Mapping) and context.get(
                "retrieval_status"
            ) == "partial":
                assay_failure_records.append(
                    {
                        "scope_id": assay_id,
                        "failure": {
                            "retrieval_status": "partial",
                            "reason_code": context.get("reason_code"),
                            "query": deepcopy(dict(query or {})),
                        },
                    }
                )
            if (
                isinstance(query, Mapping)
                and query.get("source") == "ChEMBL"
                and query.get("status_code") == 200
            ):
                assay_success_calls.append(query)
        warning_failure_records: list[dict[str, Any]] = []
        if isinstance(warning_batch_failure, Mapping):
            warning_failure_records.append(
                {
                    "scope_id": "__batch__",
                    "failure": deepcopy(dict(warning_batch_failure)),
                }
            )
        warning_failure_records.extend(
            {
                "scope_id": chembl_id,
                "failure": deepcopy(warning_failures_by_id[chembl_id]),
            }
            for chembl_id in sorted(warning_failures_by_id)
        )
        pubmed_success_calls: list[Mapping[str, Any]] = []
        pubmed_failure_records: list[dict[str, Any]] = []
        for pmid in sorted(
            pubmed_types_by_pmid,
            key=lambda value: (int(value), value),
        ):
            item = pubmed_types_by_pmid[pmid]
            query = item.get("query") if isinstance(item, Mapping) else None
            if isinstance(item, Mapping) and item.get(
                "retrieval_status"
            ) == "partial":
                pubmed_failure_records.append(
                    {
                        "scope_id": pmid,
                        "failure": {
                            "retrieval_status": "partial",
                            "reason_code": item.get("reason_code"),
                            "query": deepcopy(dict(query or {})),
                        },
                    }
                )
            if (
                isinstance(query, Mapping)
                and query.get("source") == "PubMed"
                and query.get("status_code") == 200
            ):
                pubmed_success_calls.append(query)

        source_stages = {
            "mechanism": source_stage(
                scope_ids=(
                    [target_chembl_id]
                    if isinstance(
                        candidate_discovery_bounds.get("mechanism_query"),
                        Mapping,
                    )
                    else []
                ),
                calls=(
                    [candidate_discovery_bounds["mechanism_query"]]
                    if isinstance(
                        candidate_discovery_bounds.get("mechanism_query"),
                        Mapping,
                    )
                    else []
                ),
                failures=[],
                recovery_policy=(
                    "fail_operation_no_recovery_v1"
                    if discovery_lane == "mechanism_first"
                    else "not_applicable_activity_first_v1"
                ),
            ),
            "molecule": source_stage(
                scope_ids=list(molecule_query_scope_ids),
                calls=[molecule_call],
                failures=[],
                recovery_policy="fail_operation_no_recovery_v1",
            ),
            "activity": source_stage(
                scope_ids=list(selected_ids),
                calls=(
                    [
                        call
                        for chembl_id in selected_ids
                        for call in activity_page_calls_by_id.get(
                            chembl_id, []
                        )
                    ]
                    if activity_page_calls_by_id
                    else list(activity_calls_by_id.values())
                ),
                failures=activity_failure_records,
                recovery_policy=(
                    "candidate_isolation_without_backfill_v1"
                ),
            ),
            "assay": source_stage(
                scope_ids=list(assay_ids),
                calls=assay_success_calls,
                failures=assay_failure_records,
                recovery_policy="explicit_unknown_context_no_imputation_v1",
            ),
            "document": source_stage(
                scope_ids=list(document_ids),
                calls=[
                    document_calls_by_id[document_id]
                    for document_id in document_ids
                    if document_id in document_calls_by_id
                ],
                failures=document_failure_records,
                recovery_policy="unverified_primary_no_substitution_v1",
            ),
            "warning": source_stage(
                scope_ids=list(selected_ids),
                calls=[
                    warning_calls_by_id[chembl_id]
                    for chembl_id in selected_ids
                    if chembl_id in warning_calls_by_id
                ],
                failures=warning_failure_records,
                recovery_policy=(
                    "retryable_batch_then_candidate_isolation_v1"
                    if warning_batch_fallback_used
                    else "candidate_exclusion_on_unresolved_warning_v1"
                ),
            ),
            "pubmed_type": source_stage(
                scope_ids=list(
                    sorted(
                        pubmed_types_by_pmid,
                        key=lambda value: (int(value), value),
                    )
                ),
                calls=pubmed_success_calls,
                failures=pubmed_failure_records,
                recovery_policy="unverified_type_no_substitution_v1",
            ),
        }
        source_call_inventory = {
            "schema_version": "scientific_source_call_outcome_inventory_v1",
            "required_stages": [
                "mechanism",
                "molecule",
                "activity",
                "assay",
                "document",
                "warning",
                "pubmed_type",
            ],
            "stages": source_stages,
            "successful_call_count": sum(
                int(stage["successful_call_count"])
                for stage in source_stages.values()
            ),
            "failure_count": sum(
                int(stage["failure_count"])
                for stage in source_stages.values()
            ),
            "all_failure_records_represented": True,
            "fixture": False,
        }
        source_call_inventory["inventory_sha256"] = _canonical_sha256(
            source_call_inventory
        )
        recovery_events = [
            {
                "stage": stage_name,
                "scope_id": str(failure.get("scope_id") or ""),
                "policy": str(stage["recovery_policy"]),
            }
            for stage_name, stage in source_stages.items()
            for failure in stage["failures"]
        ]
        source_recovery_attestation = {
            "schema_version": "scientific_source_recovery_attestation_v1",
            "allowlisted_policies": sorted(
                {
                    str(stage["recovery_policy"])
                    for stage in source_stages.values()
                }
            ),
            "events": recovery_events,
            "event_count": len(recovery_events),
            "failure_count": source_call_inventory["failure_count"],
            "no_unlisted_recovery": True,
            "fixture": False,
        }
        source_recovery_attestation["attestation_sha256"] = (
            _canonical_sha256(source_recovery_attestation)
        )
        candidate_discovery_bounds["source_call_inventory"] = (
            source_call_inventory
        )
        candidate_discovery_bounds["source_recovery_attestation"] = (
            source_recovery_attestation
        )
        candidate_discovery_bounds["fixed_pool_molecule_projection"] = (
            fixed_pool_molecule_projection
        )
        candidate_discovery_bounds["warning_screen_identity_count"] = len(
            selected_ids
        )
        candidate_discovery_bounds["warning_screen_identity_ids"] = list(
            selected_ids
        )
        candidate_discovery_bounds["warning_screen_within_prewarning_limit"] = (
            candidate_identity_limit is None
            or len(selected_ids) <= candidate_identity_limit
        )
        candidate_discovery_bounds["warning_screen"] = deepcopy(
            warning_screen_summary
        )
        candidate_discovery_bounds["downstream_candidate_identity_count"] = len(
            downstream_candidate_ids
        )
        candidate_discovery_bounds["downstream_candidate_identity_ids"] = list(
            downstream_candidate_ids
        )
        candidate_discovery_bounds["identity_resolution"] = [
            deepcopy(identity_resolution[chembl_id])
            for chembl_id in discovered_ids
        ]
        candidate_discovery_bounds["document_lookup_candidate_identity_count"] = len(
            document_candidate_ids
        )
        candidate_discovery_bounds["document_lookup_candidate_identity_ids"] = list(
            document_candidate_ids
        )
        candidate_discovery_bounds["document_lookup_id_count"] = len(document_ids)
        candidate_discovery_bounds[
            "pubmed_type_lookup_candidate_identity_count"
        ] = len(pubmed_candidate_ids)
        candidate_discovery_bounds[
            "pubmed_type_lookup_candidate_identity_ids"
        ] = list(pubmed_candidate_ids)
        candidate_discovery_bounds["pubmed_type_lookup_pmid_count"] = len(
            pubmed_types_by_pmid
        )
        candidate_discovery_bounds["assay_lookup_candidate_identity_count"] = len(
            assay_candidate_ids
        )
        candidate_discovery_bounds["assay_lookup_candidate_identity_ids"] = list(
            assay_candidate_ids
        )
        candidate_discovery_bounds["assay_lookup_id_count"] = len(assay_ids)
        candidate_discovery_bounds["downstream_candidate_identity_limit_satisfied"] = (
            candidate_identity_limit is None
            or (
                len(document_candidate_ids) <= candidate_identity_limit
                and len(pubmed_candidate_ids) <= candidate_identity_limit
                and len(assay_candidate_ids) <= candidate_identity_limit
            )
        )
        measured: list[_MeasuredLigand] = []
        for chembl_id, activity_rows in activity_rows_by_molecule.items():
            if not activity_rows:
                continue
            activity_call = activity_calls_by_id.get(chembl_id)
            if activity_call is None:
                continue
            activity_query_fallback = activity_query_fallbacks_by_id.get(
                chembl_id
            )
            mechanism_identity = mechanism_provenance_by_id.get(chembl_id)
            # A missing regulatory-warning response is unknown status, not
            # evidence of no warning. Isolate that candidate and continue with
            # independently resolved candidates from the same target.
            warning_call = warning_calls_by_id.get(chembl_id)
            if warning_call is None:
                continue
            molecule = molecule_by_id.get(chembl_id)
            if molecule is None:
                continue
            structures = molecule.get("molecule_structures") or {}
            if not isinstance(structures, dict):
                continue
            smiles = str(structures.get("canonical_smiles") or "").strip()
            if not smiles:
                continue
            max_phase = int(_safe_float(molecule.get("max_phase")) or 0)
            if require_clinical and max_phase < 1:
                continue
            title = str(molecule.get("pref_name") or chembl_id)
            exact_aliases = aliases_by_id.get(chembl_id, [])
            alias_conflicts = alias_conflicts_by_id.get(chembl_id, [])
            competitor_aliases = _competitor_alias_provenance(
                alias_inventory,
                current_molecule_chembl_id=chembl_id,
            )
            attested_rows: list[
                tuple[
                    dict[str, Any],
                    ActivityMeasurement,
                    str,
                    str,
                    str,
                    dict[str, Any],
                ]
            ] = []
            for raw_row, row_measurement, row_direction_rule in activity_rows:
                row_activity_id = str(
                    raw_row.get("activity_id") or f"{chembl_id}-activity"
                )
                row_document_id = str(
                    raw_row.get("document_chembl_id") or ""
                ).strip()
                row_primary = self._primary_source_attestation(
                    activity_id=row_activity_id,
                    chembl_id=chembl_id,
                    target_chembl_id=target_chembl_id,
                    candidate_aliases=exact_aliases,
                    candidate_alias_conflicts=alias_conflicts,
                    competitor_alias_provenance=competitor_aliases,
                    target=normalized,
                    measurement=row_measurement,
                    direction_rule=row_direction_rule,
                    mechanism_identity=mechanism_identity,
                    raw_activity=raw_row,
                    assay_context=assay_contexts.get(
                        str(raw_row.get("assay_chembl_id") or "").strip()
                    ),
                    document_chembl_id=row_document_id,
                    document=documents_by_id.get(row_document_id),
                    document_call=document_calls_by_id.get(row_document_id),
                    document_failure=document_failures_by_id.get(
                        row_document_id
                    ),
                    document_query_batch=document_query_batches_by_id.get(
                        row_document_id
                    ),
                    pubmed_types_by_pmid=pubmed_types_by_pmid,
                )
                attested_rows.append(
                    (
                        raw_row,
                        row_measurement,
                        row_direction_rule,
                        row_activity_id,
                        row_document_id,
                        row_primary,
                    )
                )
            paired_v4_option = paired_v4_promotions.get(chembl_id)
            paper_native_v5_support = paper_native_v5_support_by_id.get(
                chembl_id
            )
            paper_native_v5_option = paper_native_v5_promotions.get(chembl_id)
            if paper_native_v5_active and paper_native_v5_support is not None:
                selected_index = int(
                    paper_native_v5_support["retained_index"]
                )
            elif not paper_native_v6_active and paired_v4_option is not None:
                selected_index = int(paired_v4_option["retained_index"])
            else:
                selected_index = next(
                    (
                        index
                        for index, item in enumerate(attested_rows)
                        if item[5].get("verification_status") == "verified"
                        and item[5].get("is_primary_support") is True
                    ),
                    0,
                )
            (
                raw,
                measurement,
                direction_rule,
                activity_id,
                document_chembl_id,
                primary_source,
            ) = attested_rows[selected_index]
            selected_row_query = activity_row_queries_by_id.get(
                chembl_id, {}
            ).get(activity_id)
            if selected_row_query is not None:
                activity_call = selected_row_query[0]
            primary_source["activity_document_selection"] = {
                "policy": (
                    "complete_exact_parent_activity_document_rows_v1"
                    if paper_native_v6_active
                    else "bounded_top_k_activity_document_rows_v1"
                ),
                "top_k": (
                    None
                    if paper_native_v6_active
                    else _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
                ),
                "result_row_cap": (
                    None
                    if paper_native_v6_active
                    else _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
                ),
                "eligible_row_count": eligible_activity_count_by_molecule.get(
                    chembl_id, len(attested_rows)
                ),
                "inspected_row_count": len(attested_rows),
                "truncated": (
                    False
                    if paper_native_v6_active
                    else eligible_activity_count_by_molecule.get(
                        chembl_id, len(attested_rows)
                    )
                    > len(attested_rows)
                ),
                "all_eligible_rows_inspected": (
                    paper_native_v6_active
                    and len(attested_rows)
                    == eligible_activity_count_by_molecule.get(
                        chembl_id, len(attested_rows)
                    )
                ),
                "selected_rank": selected_index + 1,
                "selected_activity_id": activity_id,
                "selected_by_pre_pubmed_paired_v4_cohort": (
                    not paper_native_v6_active and paired_v4_option is not None
                ),
                "paired_v4_selection_precedes_pubmed_search": (
                    not paper_native_v6_active and paired_v4_option is not None
                ),
                "selected_by_candidate_primary_record_v6": (
                    paper_native_v6_active
                ),
                "paper_native_v6_paired_search_or_selection_used": False,
                "selected_by_pre_pubmed_paper_native_v5_direct_support": (
                    paper_native_v5_active
                    and paper_native_v5_support is not None
                ),
                "paper_native_v5_selection_precedes_pubmed_search": (
                    paper_native_v5_active
                    and paper_native_v5_support is not None
                ),
                "paper_native_v5_pubmed_value_used_for_selection": False,
                "pchembl_scope": (
                    "bounded_within_molecule_document_inspection_only"
                ),
                "pchembl_used_for_candidate_ordering": False,
                "rows": [
                    {
                        "rank": index + 1,
                        "activity_id": item[3],
                        "document_chembl_id": item[4] or None,
                        "endpoint": item[1].endpoint,
                        "pchembl_value": _safe_float(
                            item[0].get("pchembl_value")
                        ),
                        "verification_status": item[5].get(
                            "verification_status"
                        ),
                        "reason_code": item[5].get("reason_code"),
                    }
                    for index, item in enumerate(attested_rows)
                ],
                "fixture": False,
                "missing_values_imputed": False,
            }
            assay_id = str(raw.get("assay_chembl_id") or "").strip()
            assay_context = assay_contexts.get(
                assay_id,
                {
                    "schema_version": CHEMBL_ASSAY_CONTEXT_SCHEMA,
                    "retrieval_status": "missing",
                    "binding_status": "unresolved",
                    "reason_code": "ACTIVITY_ASSAY_ID_MISSING",
                    "assay_chembl_id": assay_id or None,
                    "record": None,
                    "condition_coverage": {
                        "construct": "unknown",
                        "mutation_state": "unknown_not_assumed_wild_type",
                        "atp": "unknown",
                        "substrate": "unknown",
                        "detection_technology": "unknown",
                    },
                    "assay_query": None,
                    "missing_values_imputed": False,
                    "inference_used": False,
                    "unit_conversion_applied": False,
                    "description_similarity_used": False,
                },
            )
            engagement_assessment = _assay_engagement_assessment(
                assay_type=measurement.assay_type,
                assay_context=assay_context,
                assay_chembl_id=assay_id,
                target_chembl_id=target_chembl_id,
                target_symbol=normalized.symbol,
            )
            resolved_assay_context = str(
                engagement_assessment["assay_context"]
            )
            direct_engagement = bool(
                engagement_assessment["direct_single_protein_engagement"]
            )
            paired_v4_envelope: dict[str, Any] | None = None
            if (
                not paper_native_v6_active
                and paired_v4_option is not None
                and paired_v4_inventory is not None
                and paired_v4_outcome_payload is not None
                and paired_v4_promotion_summary is not None
            ):
                paired_v4_envelope = _paired_v4_evidence_envelope(
                    chembl_id=chembl_id,
                    target=normalized,
                    target_chembl_id=target_chembl_id,
                    option=paired_v4_option,
                    inventory=paired_v4_inventory,
                    outcome_payload=paired_v4_outcome_payload,
                    promotion_summary=paired_v4_promotion_summary,
                    raw=raw,
                    measurement=measurement,
                    activity_id=activity_id,
                    activity_call=activity_call,
                    assay_context=assay_context,
                )
            paper_native_v5_envelope: dict[str, Any] | None = None
            if (
                paper_native_v5_active
                and paper_native_v5_alias_inventory is not None
                and paper_native_v5_outcome_payload is not None
                and paper_native_v5_promotion_summary is not None
            ):
                paper_native_v5_envelope = (
                    _paper_native_v5_candidate_envelope(
                        chembl_id=chembl_id,
                        target=normalized,
                        target_chembl_id=target_chembl_id,
                        fixed_pool_ids=selected_ids,
                        support_by_id=paper_native_v5_support_by_id,
                        option=paper_native_v5_option,
                        alias_inventory=paper_native_v5_alias_inventory,
                        outcome_payload=paper_native_v5_outcome_payload,
                        promotion_summary=paper_native_v5_promotion_summary,
                    )
                )
            paper_native_v5_source_separation: dict[str, Any] | None = None
            if paper_native_v5_envelope is not None:
                paper_native_v5_source_separation = {
                    "schema_version": (
                        "paper_native_v5_independent_source_separation_v1"
                    ),
                    "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
                    "legacy_candidate_primary_evidence_v2_applicable": False,
                    "legacy_chembl_document_metadata_lookup_executed": False,
                    "chembl_document_chembl_id": document_chembl_id or None,
                    "pubmed_document_id": paper_native_v5_envelope.get("pmid"),
                    "same_document_binding_required": False,
                    "independent_measurements_not_equated": True,
                    "fixture": False,
                }
                paper_native_v5_source_separation["attestation_sha256"] = (
                    _canonical_sha256(paper_native_v5_source_separation)
                )
            evidence = EvidenceRef(
                evidence_id=measurement.source_evidence_id
                or _stable_id("chembl_activity", activity_id),
                source="ChEMBL",
                external_id=activity_id,
                title=(
                    f"Measured {measurement.endpoint} for {chembl_id} against "
                    f"{target_chembl_id}"
                ),
                url=(
                    "https://www.ebi.ac.uk/chembl/explore/activity/"
                    f"{quote(activity_id)}"
                ),
                supports="support",
                excerpt_hash=(
                    None
                    if paper_native_v5_active
                    else primary_source.get("excerpt_hash")
                ),
                metadata={
                    "fixture": False,
                    "live": True,
                    "provenance_class": "external_live",
                    "chembl_version": self._chembl_version,
                    "target_chembl_id": target_chembl_id,
                    "target_accession": normalized.uniprot_id,
                    "target_uniprot_id": normalized.uniprot_id,
                    "target_gene": normalized.symbol,
                    "target_resolution_method": "exact_uniprot_component",
                    "molecule_chembl_id": chembl_id,
                    "activity_id": activity_id,
                    "assay_chembl_id": assay_id or None,
                    "document_chembl_id": document_chembl_id or None,
                    "data_validity_comment": raw["data_validity_comment"],
                    "pchembl_value": _safe_float(raw.get("pchembl_value")),
                    # Preserve the deposited fields separately from the typed
                    # ActivityMeasurement.  Comparability requires exact
                    # equality between them; normalized/missing source fields
                    # are never rewritten into an apparently explicit bind.
                    "standard_type": str(
                        raw.get("standard_type") or ""
                    ).strip(),
                    "standard_relation": str(
                        raw.get("standard_relation") or ""
                    ).strip(),
                    "standard_value": raw.get("standard_value"),
                    "standard_units": str(
                        raw.get("standard_units") or ""
                    ).strip(),
                    "activity_properties": deepcopy(
                        raw.get("activity_properties")
                        if isinstance(raw.get("activity_properties"), list)
                        else []
                    ),
                    "assay_type": measurement.assay_type,
                    "assay_context": resolved_assay_context,
                    "chembl_assay_context": deepcopy(assay_context),
                    "assay_engagement_assessment": deepcopy(
                        engagement_assessment
                    ),
                    "direct_single_protein_engagement": direct_engagement,
                    "activity_query": activity_call.as_dict(),
                    "activity_query_fallback": deepcopy(
                        activity_query_fallback
                    ),
                    "candidate_discovery_lane": discovery_lane,
                    "mechanism_identity": deepcopy(mechanism_identity),
                    "molecule_query": molecule_call.as_dict(),
                    "drug_warning_query": warning_call.as_dict(),
                    "candidate_pool_warning_screen": warning_screen_summary,
                    "candidate_discovery_bounds": deepcopy(
                        candidate_discovery_bounds
                    ),
                    "intervention_direction_requested": direction,
                    "direction_evidence_rule": direction_rule,
                    **(
                        {
                            "primary_source_contract": deepcopy(
                                paper_native_v5_source_separation
                            )
                        }
                        if paper_native_v5_source_separation is not None
                        else {"primary_source": primary_source}
                    ),
                    **(
                        {
                            "primary_abstract_paired_v4": deepcopy(
                                paired_v4_envelope
                            )
                        }
                        if paired_v4_envelope is not None
                        else {}
                    ),
                    **(
                        {
                            "primary_abstract_paper_native_v5": deepcopy(
                                paper_native_v5_envelope
                            )
                        }
                        if paper_native_v5_envelope is not None
                        else {}
                    ),
                },
            )
            warnings = warnings_by_id.get(chembl_id, [])
            withdrawn = any(
                "withdraw" in json.dumps(item, ensure_ascii=False).casefold()
                for item in warnings
            )
            metadata = {
                "fixture": False,
                "live": True,
                "provenance_class": "external_live",
                "drug_id": chembl_id,
                "chembl_id": chembl_id,
                "aliases": list(exact_aliases),
                "chembl_exact_aliases": list(exact_aliases),
                "chembl_alias_source": _CANDIDATE_ALIAS_SOURCE,
                "chembl_alias_conflicts": list(alias_conflicts),
                "max_phase": max_phase,
                "development_stage": (
                    "chembl_max_phase_4"
                    if max_phase == 4
                    else "chembl_clinical_phase_metadata"
                    if max_phase >= 1
                    else "chembl_preclinical_phase_metadata"
                ),
                "development_stage_basis": (
                    "ChEMBL max_phase metadata only; not an independent regulatory "
                    "approval attestation"
                ),
                "regulatory_approval_claimed": False,
                "regulatory_status": (
                    "withdrawn"
                    if withdrawn
                    else "unknown_no_matching_chembl_warning_record"
                ),
                "drug_warnings": warnings,
                "drug_warning_query": warning_call.as_dict(),
                "candidate_pool_warning_screen": warning_screen_summary,
                "candidate_discovery_bounds": deepcopy(
                    candidate_discovery_bounds
                ),
                "candidate_discovery_lane": discovery_lane,
                "mechanism_identity": deepcopy(mechanism_identity),
                "canonical_smiles": smiles,
                "standard_inchi_key": structures.get("standard_inchi_key"),
                "measured_target": target_chembl_id,
                "measured_target_uniprot_id": normalized.uniprot_id,
                "measured_target_gene": normalized.symbol,
                "target_resolution_method": "exact_uniprot_component",
                "direction_requested": direction,
                "direction_evidence_rules": [direction_rule],
                "direction_evidence_scope": (
                    "IC50 supports potency in its recorded assay context; a cell-based "
                    "IC50 is not direct single-protein engagement. Ki/Kd support binding "
                    "only; EC50 does not establish agonism versus antagonism without "
                    "action evidence"
                ),
                "activity_assay_type": measurement.assay_type,
                "activity_assay_context": resolved_assay_context,
                "activity_assay_chembl_id": assay_id or None,
                "chembl_assay_context": deepcopy(assay_context),
                "assay_engagement_assessment": deepcopy(
                    engagement_assessment
                ),
                "direct_single_protein_engagement": direct_engagement,
                **(
                    {
                        "primary_abstract_paired_v4": deepcopy(
                            paired_v4_envelope
                        )
                    }
                    if paired_v4_envelope is not None
                    else {}
                ),
                **(
                    {
                        "primary_abstract_paper_native_v5": deepcopy(
                            paper_native_v5_envelope
                        )
                    }
                    if paper_native_v5_envelope is not None
                    else {}
                ),
            }
            mechanism = _candidate_mechanism_claim(
                target_symbol=target.symbol,
                measurement=measurement,
                direction_rule=direction_rule,
                mechanism_identity=(
                    mechanism_identity
                    if isinstance(mechanism_identity, Mapping)
                    else None
                ),
                assay_context_label=resolved_assay_context,
            )
            seed = CandidateSeed(
                name=title,
                smiles=smiles,
                mechanism=mechanism,
                activities=(measurement,),
                evidence=(evidence,),
                metadata=metadata,
            )
            measured.append(
                _MeasuredLigand(
                    seed=seed,
                    chembl_id=chembl_id,
                    pchembl_value=_safe_float(raw.get("pchembl_value")),
                    max_phase=max_phase,
                )
            )
        if len(measured) < minimum_required:
            if warning_failures_by_id:
                unavailable = ScientificDataUnavailable(
                    "ChEMBL regulatory-warning screening was unavailable for too "
                    "many measured candidates"
                )
                unavailable.provenance = {
                    "source": "ChEMBL",
                    "target_chembl_id": target_chembl_id,
                    "operation": "candidate_regulatory_warning_screen",
                    "reason_code": "REGULATORY_WARNING_SCREEN_UNAVAILABLE",
                    "eligible_after_resolved_warning_screen": len(measured),
                    "minimum_required": minimum_required,
                    "warning_screen": warning_screen_summary,
                    "candidate_failures": warning_failures_by_id,
                    "candidate_discovery_bounds": deepcopy(
                        candidate_discovery_bounds
                    ),
                    "fixture": False,
                    "missing_values_imputed": False,
                }
                raise unavailable
            raise NoMeasuredLigandCandidates(
                target_symbol=normalized.symbol,
                target_uniprot_id=normalized.uniprot_id,
                direction=direction,
                eligible_count=len(measured),
                minimum_required=minimum_required,
                reason_code=(
                    "INSUFFICIENT_CLINICAL_MEASURED_LIGANDS"
                    if require_clinical
                    else "INSUFFICIENT_STRUCTURE_RESOLVED_MEASURED_LIGANDS"
                ),
                provenance={
                    "target_chembl_id": target_chembl_id,
                    "activity_queries": [
                        activity_calls_by_id[chembl_id].as_dict()
                        for chembl_id in selected_ids
                        if chembl_id in activity_calls_by_id
                    ],
                    "molecule_query": molecule_call.as_dict(),
                    "drug_warning_queries": [
                        call.as_dict()
                        for call in {
                            (call.parameter_hash, call.response_sha256): call
                            for call in warning_calls_by_id.values()
                        }.values()
                    ],
                    "warning_screen": warning_screen_summary,
                    "candidate_discovery_bounds": deepcopy(
                        candidate_discovery_bounds
                    ),
                    "direction_compatible_activity_count": len(
                        activity_rows_by_molecule
                    ),
                    "clinical_phase_required": require_clinical,
                },
            )
        measured.sort(key=lambda item: (-item.max_phase, item.chembl_id))
        return measured

    def _fetch_bounded_activity_pages(
        self,
        scientific_filters: Mapping[str, Any],
        *,
        requested_limit: int,
        trigger_failure: Mapping[str, Any],
    ) -> tuple[
        dict[str, Any],
        HTTPCallProvenance,
        dict[str, Any],
    ]:
        """Recover from a retryable large ChEMBL request with bounded pages."""

        raw_row_cap = min(80, max(1, int(requested_limit)))
        page_size = min(20, raw_row_cap)
        max_pages = min(4, (raw_row_cap + page_size - 1) // page_size)
        activities: list[dict[str, Any]] = []
        calls: list[HTTPCallProvenance] = []
        failed_page: dict[str, Any] | None = None
        stop_reason = "bounded_page_limit_reached"
        for page_index in range(max_pages):
            offset = page_index * page_size
            remaining = raw_row_cap - len(activities)
            if remaining <= 0:
                stop_reason = "raw_row_cap_reached"
                break
            page_limit = min(page_size, remaining)
            params = {
                **dict(scientific_filters),
                "limit": page_limit,
                "offset": offset,
            }
            try:
                payload, call = self._get_json(
                    self.activity_endpoint,
                    params=params,
                )
            except ScientificDataUnavailable as exc:
                failed_page = {
                    "page_index": page_index,
                    "offset": offset,
                    "limit": page_limit,
                    "failure": deepcopy(getattr(exc, "provenance", {})),
                }
                stop_reason = "page_request_unavailable"
                if calls:
                    break
                provenance = deepcopy(getattr(exc, "provenance", {}))
                provenance["bounded_pagination"] = {
                    "page_size": page_size,
                    "max_pages": max_pages,
                    "raw_row_cap": raw_row_cap,
                    "successful_page_count": 0,
                    "failed_page": failed_page,
                    "trigger_failure": deepcopy(dict(trigger_failure)),
                    "fixture": False,
                }
                exc.provenance = provenance
                raise
            page_activities = payload.get("activities")
            if not isinstance(page_activities, list):
                unavailable = ScientificDataUnavailable(
                    "ChEMBL bounded activity page has no activity list"
                )
                unavailable.provenance = {
                    "source": "ChEMBL",
                    "error_type": "invalid_json_schema",
                    "retryable": False,
                    "page_index": page_index,
                    "offset": offset,
                    "limit": page_limit,
                    "page_call": call.as_dict(),
                    "trigger_failure": deepcopy(dict(trigger_failure)),
                    "fixture": False,
                }
                if not calls:
                    raise unavailable
                failed_page = {
                    "page_index": page_index,
                    "offset": offset,
                    "limit": page_limit,
                    "failure": deepcopy(unavailable.provenance),
                }
                stop_reason = "invalid_page_schema"
                break
            calls.append(call)
            activities.extend(
                dict(item)
                for item in page_activities
                if isinstance(item, dict)
            )
            page_meta = payload.get("page_meta")
            total_count = (
                _safe_float(page_meta.get("total_count"))
                if isinstance(page_meta, dict)
                else None
            )
            if total_count is not None and offset + len(page_activities) >= int(
                total_count
            ):
                stop_reason = "reported_result_end"
                break
            if len(page_activities) < page_limit:
                stop_reason = "short_page"
                break
        if not calls:
            unavailable = ScientificDataUnavailable(
                "ChEMBL bounded activity pagination returned no authoritative page"
            )
            unavailable.provenance = {
                "source": "ChEMBL",
                "error_type": "bounded_pagination_empty",
                "retryable": False,
                "trigger_failure": deepcopy(dict(trigger_failure)),
                "fixture": False,
            }
            raise unavailable
        activities = activities[:raw_row_cap]
        return (
            {"activities": activities},
            calls[0],
            {
                "bounded_pagination": True,
                "page_size": page_size,
                "max_pages": max_pages,
                "raw_row_cap": raw_row_cap,
                "successful_page_count": len(calls),
                "raw_activity_count": len(activities),
                "partial_result": failed_page is not None,
                "stop_reason": stop_reason,
                "page_calls": [call.as_dict() for call in calls],
                "failed_page": failed_page,
                "scientific_filter_keys": sorted(scientific_filters),
                "missing_values_imputed": False,
                "fixture": False,
            },
        )

    def _fetch_complete_exact_parent_activity_pages(
        self,
        activity_parameters: Mapping[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        list[HTTPCallProvenance],
        list[dict[str, Any]],
        dict[str, tuple[HTTPCallProvenance, dict[str, Any]]],
        int,
    ]:
        """Read every row reported by one exact-parent ChEMBL query.

        Twenty rows is the wire page size, not a scientific eligibility cap.
        The authoritative ``total_count`` fixes the finite page set.  A
        missing/changed count, short page, duplicate/non-monotonic activity
        identity, failed page, or operation-deadline exhaustion makes the
        candidate incomplete; partial rows are never promoted.
        """

        base_parameters = dict(activity_parameters)
        if base_parameters.get("limit") != _MAX_ACTIVITY_ROWS_PER_CANDIDATE:
            raise ValueError("exact-parent activity page size must be 20")
        if "offset" in base_parameters:
            raise ValueError("the first exact-parent page must not set offset")

        rows: list[dict[str, Any]] = []
        calls: list[HTTPCallProvenance] = []
        page_audits: list[dict[str, Any]] = []
        row_queries: dict[
            str, tuple[HTTPCallProvenance, dict[str, Any]]
        ] = {}
        expected_total: int | None = None
        previous_activity_id: int | None = None
        offset = 0

        def incomplete(
            reason_code: str,
            *,
            detail: str,
            page_parameters: Mapping[str, Any] | None = None,
            cause: ScientificDataUnavailable | None = None,
        ) -> NoReturn:
            unavailable = ScientificDataUnavailable(detail)
            failure = (
                deepcopy(getattr(cause, "provenance", {}))
                if cause is not None
                else None
            )
            unavailable.provenance = {
                "source": "ChEMBL",
                "operation": "complete_exact_parent_activity_pagination",
                "reason_code": reason_code,
                "page_size": _MAX_ACTIVITY_ROWS_PER_CANDIDATE,
                "reported_total_count": expected_total,
                "completed_row_count": len(rows),
                "completed_page_count": len(calls),
                "failed_page_parameters": (
                    deepcopy(dict(page_parameters))
                    if page_parameters is not None
                    else None
                ),
                "completed_pages": deepcopy(page_audits),
                "failure": failure,
                "partial_rows_discarded": True,
                "operation_deadline_enforced": True,
                "fixture": False,
                "missing_values_imputed": False,
            }
            if cause is None:
                raise unavailable
            raise unavailable from cause

        while expected_total is None or offset < expected_total:
            page_index = len(calls)
            params = dict(base_parameters)
            if page_index:
                params["offset"] = offset
            try:
                payload, call = self._get_json(
                    self.activity_endpoint,
                    params=params,
                )
            except ScientificDataUnavailable as exc:
                incomplete(
                    "ACTIVITY_COMPLETE_PAGINATION_UNAVAILABLE",
                    detail=(
                        "A ChEMBL exact-parent activity page was unavailable; "
                        "partial activity rows were discarded"
                    ),
                    page_parameters=params,
                    cause=exc,
                )

            raw_page = payload.get("activities")
            page_meta = payload.get("page_meta")
            if not isinstance(raw_page, list) or not isinstance(
                page_meta, Mapping
            ):
                incomplete(
                    "ACTIVITY_RESPONSE_METADATA_MISSING",
                    detail=(
                        "A ChEMBL exact-parent activity page lacked its row "
                        "list or page metadata"
                    ),
                    page_parameters=params,
                )
            raw_total = page_meta.get("total_count")
            if isinstance(raw_total, bool):
                raw_total = None
            try:
                page_total = int(raw_total)
            except (TypeError, ValueError):
                page_total = -1
            if page_total < 0:
                incomplete(
                    "ACTIVITY_TOTAL_COUNT_INVALID",
                    detail="ChEMBL returned an invalid activity total_count",
                    page_parameters=params,
                )
            if expected_total is None:
                expected_total = page_total
            elif page_total != expected_total:
                incomplete(
                    "ACTIVITY_TOTAL_COUNT_CHANGED_DURING_PAGINATION",
                    detail=(
                        "ChEMBL activity total_count changed while reading "
                        "the exact-parent result set"
                    ),
                    page_parameters=params,
                )

            remaining = expected_total - offset
            expected_page_count = min(
                _MAX_ACTIVITY_ROWS_PER_CANDIDATE,
                max(0, remaining),
            )
            if len(raw_page) != expected_page_count:
                incomplete(
                    "ACTIVITY_COMPLETE_PAGINATION_SHORT_PAGE",
                    detail=(
                        "ChEMBL returned a short or oversized activity page "
                        "before the reported result set was complete"
                    ),
                    page_parameters=params,
                )

            normalized_page: list[dict[str, Any]] = []
            page_activity_ids: list[str] = []
            for raw_row in raw_page:
                if not isinstance(raw_row, dict):
                    incomplete(
                        "ACTIVITY_PAGINATION_ROW_INVALID",
                        detail=(
                            "ChEMBL returned a non-object activity row during "
                            "complete pagination"
                        ),
                        page_parameters=params,
                    )
                activity_id_text = str(
                    raw_row.get("activity_id") or ""
                ).strip()
                try:
                    activity_id_number = int(activity_id_text)
                except (TypeError, ValueError):
                    activity_id_number = -1
                if (
                    activity_id_number < 1
                    or activity_id_text in row_queries
                    or (
                        previous_activity_id is not None
                        and activity_id_number <= previous_activity_id
                    )
                ):
                    incomplete(
                        "ACTIVITY_PAGINATION_IDENTITY_INCOMPLETE",
                        detail=(
                            "ChEMBL activity IDs were missing, duplicated, or "
                            "not strictly increasing across ordered pages"
                        ),
                        page_parameters=params,
                    )
                previous_activity_id = activity_id_number
                copied_row = dict(raw_row)
                normalized_page.append(copied_row)
                page_activity_ids.append(activity_id_text)
                row_queries[activity_id_text] = (call, deepcopy(params))

            calls.append(call)
            rows.extend(normalized_page)
            page_audit: dict[str, Any] = {
                "page_index": page_index,
                "offset": offset,
                "limit": _MAX_ACTIVITY_ROWS_PER_CANDIDATE,
                "response_count": len(normalized_page),
                "reported_total_count": expected_total,
                "activity_ids": page_activity_ids,
                "activity_ids_sha256": _canonical_sha256(page_activity_ids),
                "response_rows_sha256": _canonical_sha256(normalized_page),
                "query": call.as_dict(),
                "query_parameters": deepcopy(params),
            }
            page_audit["page_audit_sha256"] = _canonical_sha256(page_audit)
            page_audits.append(page_audit)
            offset += len(normalized_page)
            if expected_total == 0:
                break

        if expected_total is None or len(rows) != expected_total:
            incomplete(
                "ACTIVITY_COMPLETE_PAGINATION_INCOMPLETE",
                detail=(
                    "ChEMBL exact-parent activity pagination did not collect "
                    "the complete reported result set"
                ),
            )
        return rows, calls, page_audits, row_queries, expected_total

    def _fetch_assay_contexts(
        self,
        assay_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch exact ChEMBL assay records in finite batches.

        Assay context is interpretive metadata, not a prerequisite for
        retaining an otherwise exact activity measurement.  A timeout or a
        missing record therefore becomes an explicit partial/unknown context;
        it never fills construct, mutation, ATP, substrate, or detection data.
        """

        exact_ids = sorted(
            {
                str(item).strip()
                for item in assay_ids
                if str(item).strip()
            }
        )
        # Every exact assay identity attached to a completely paginated V6
        # activity pool is queried.  The finite per-request batch size and the
        # enclosing operation deadline bound I/O; an arbitrary total-ID cap
        # must not turn a later verified primary row into a false negative.
        requested_ids = exact_ids
        contexts: dict[str, dict[str, Any]] = {}

        for start in range(0, len(requested_ids), _ASSAY_CONTEXT_BATCH_SIZE):
            batch = requested_ids[start : start + _ASSAY_CONTEXT_BATCH_SIZE]
            try:
                payload, call = self._get_json(
                    self.assay_endpoint,
                    params={
                        "assay_chembl_id__in": ",".join(batch),
                        "limit": len(batch),
                    },
                )
            except ScientificDataUnavailable as exc:
                failure = deepcopy(getattr(exc, "provenance", {}))
                for assay_id in batch:
                    contexts[assay_id] = {
                        "schema_version": CHEMBL_ASSAY_CONTEXT_SCHEMA,
                        "retrieval_status": "partial",
                        "binding_status": "unresolved",
                        "reason_code": "ASSAY_LOOKUP_UNAVAILABLE",
                        "assay_chembl_id": assay_id,
                        "record": None,
                        "condition_coverage": {
                            "construct": "unknown",
                            "mutation_state": "unknown_not_assumed_wild_type",
                            "atp": "unknown",
                            "substrate": "unknown",
                            "detection_technology": "unknown",
                        },
                        "assay_query": failure,
                        "missing_values_imputed": False,
                        "inference_used": False,
                        "unit_conversion_applied": False,
                        "description_similarity_used": False,
                    }
                continue

            raw_assays = payload.get("assays")
            if not isinstance(raw_assays, list):
                for assay_id in batch:
                    contexts[assay_id] = {
                        "schema_version": CHEMBL_ASSAY_CONTEXT_SCHEMA,
                        "retrieval_status": "partial",
                        "binding_status": "unresolved",
                        "reason_code": "ASSAY_RESPONSE_INVALID",
                        "assay_chembl_id": assay_id,
                        "record": None,
                        "condition_coverage": {
                            "construct": "unknown",
                            "mutation_state": "unknown_not_assumed_wild_type",
                            "atp": "unknown",
                            "substrate": "unknown",
                            "detection_technology": "unknown",
                        },
                        "assay_query": call.as_dict(),
                        "missing_values_imputed": False,
                        "inference_used": False,
                        "unit_conversion_applied": False,
                        "description_similarity_used": False,
                    }
                continue
            by_id: dict[str, Mapping[str, Any]] = {}
            assay_rows_invalid = False
            for item in raw_assays:
                if not isinstance(item, Mapping):
                    assay_rows_invalid = True
                    continue
                assay_id = str(item.get("assay_chembl_id") or "").strip()
                if not assay_id or assay_id not in batch or assay_id in by_id:
                    assay_rows_invalid = True
                    continue
                by_id[assay_id] = item
            if assay_rows_invalid:
                for assay_id in batch:
                    contexts[assay_id] = {
                        "schema_version": CHEMBL_ASSAY_CONTEXT_SCHEMA,
                        "retrieval_status": "partial",
                        "binding_status": "unresolved",
                        "reason_code": "ASSAY_RESPONSE_ROWS_INVALID",
                        "assay_chembl_id": assay_id,
                        "record": None,
                        "condition_coverage": {
                            "construct": "unknown",
                            "mutation_state": "unknown_not_assumed_wild_type",
                            "atp": "unknown",
                            "substrate": "unknown",
                            "detection_technology": "unknown",
                        },
                        "assay_query": call.as_dict(),
                        "missing_values_imputed": False,
                        "inference_used": False,
                        "unit_conversion_applied": False,
                        "description_similarity_used": False,
                    }
                continue
            for assay_id in batch:
                raw = by_id.get(assay_id)
                if raw is None:
                    contexts[assay_id] = {
                        "schema_version": CHEMBL_ASSAY_CONTEXT_SCHEMA,
                        "retrieval_status": "missing",
                        "binding_status": "unresolved",
                        "reason_code": "ASSAY_RECORD_NOT_FOUND",
                        "assay_chembl_id": assay_id,
                        "record": None,
                        "condition_coverage": {
                            "construct": "unknown",
                            "mutation_state": "unknown_not_assumed_wild_type",
                            "atp": "unknown",
                            "substrate": "unknown",
                            "detection_technology": "unknown",
                        },
                        "assay_query": call.as_dict(),
                        "missing_values_imputed": False,
                        "inference_used": False,
                        "unit_conversion_applied": False,
                        "description_similarity_used": False,
                    }
                    continue

                parameter_source_inventory, _raw_parameter_rows = (
                    _paired_v4_list_source_inventory(
                        source=(
                            "chembl_assay_context.record.assay_parameters"
                        ),
                        container=raw,
                        field="assay_parameters",
                    )
                )
                parameters = _exact_assay_parameters(
                    raw.get("assay_parameters")
                )
                variant_sequence = _bounded_assay_text(
                    raw.get("variant_sequence")
                )
                contexts[assay_id] = {
                    "schema_version": CHEMBL_ASSAY_CONTEXT_SCHEMA,
                    "retrieval_status": "completed",
                    "binding_status": "exact",
                    "reason_code": None,
                    "assay_chembl_id": assay_id,
                    "record": {
                        "assay_chembl_id": assay_id,
                        "target_chembl_id": raw.get("target_chembl_id"),
                        "document_chembl_id": raw.get("document_chembl_id"),
                        "assay_type": raw.get("assay_type"),
                        "assay_category": raw.get("assay_category"),
                        "assay_organism": raw.get("assay_organism"),
                        "assay_cell_type": raw.get("assay_cell_type"),
                        "assay_tissue": raw.get("assay_tissue"),
                        "assay_strain": raw.get("assay_strain"),
                        "assay_subcellular_fraction": raw.get(
                            "assay_subcellular_fraction"
                        ),
                        "cell_chembl_id": raw.get("cell_chembl_id"),
                        "tissue_chembl_id": raw.get("tissue_chembl_id"),
                        "relationship_type": raw.get("relationship_type"),
                        "relationship_description": raw.get(
                            "relationship_description"
                        ),
                        "bao_format": raw.get("bao_format"),
                        "bao_label": raw.get("bao_label"),
                        "confidence_score": raw.get("confidence_score"),
                        "confidence_description": raw.get(
                            "confidence_description"
                        ),
                        "src_id": raw.get("src_id"),
                        "src_assay_id": raw.get("src_assay_id"),
                        "description": _bounded_assay_text(
                            raw.get("description")
                        ),
                        "variant_sequence": variant_sequence,
                        "assay_parameters": parameters,
                        "assay_parameters_source_inventory": (
                            parameter_source_inventory
                        ),
                        "assay_parameter_count": len(parameters),
                        "assay_parameters_truncated": bool(
                            isinstance(raw.get("assay_parameters"), list)
                            and len(raw["assay_parameters"]) > len(parameters)
                        ),
                    },
                    "condition_coverage": _assay_condition_coverage(
                        variant_sequence=raw.get("variant_sequence"),
                        parameters=parameters,
                    ),
                    "assay_query": call.as_dict(),
                    "missing_values_imputed": False,
                    "inference_used": False,
                    "unit_conversion_applied": False,
                    "description_similarity_used": False,
                }
        return contexts

    def _fetch_primary_documents(
        self,
        document_ids: Sequence[str],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, HTTPCallProvenance],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        """Resolve exact activity documents without making MA2 availability depend on it.

        Candidate/activity retrieval remains usable when the supplementary
        document request fails.  The unresolved state is persisted on every
        affected evidence record and MA3 applies the ResearchBrief policy.
        """

        exact_ids = sorted(
            {
                str(item).strip()
                for item in document_ids
                if _CHEMBL_STABLE_ID.fullmatch(str(item).strip())
            }
        )
        if not exact_ids:
            return {}, {}, {}, {}
        resolved: dict[str, dict[str, Any]] = {}
        calls_by_id: dict[str, HTTPCallProvenance] = {}
        failures_by_id: dict[str, dict[str, Any]] = {}
        batch_attestations_by_id: dict[str, dict[str, Any]] = {}
        for start in range(0, len(exact_ids), _PRIMARY_DOCUMENT_BATCH_SIZE):
            batch = exact_ids[start : start + _PRIMARY_DOCUMENT_BATCH_SIZE]
            query_parameters = {
                "document_chembl_id__in": ",".join(batch),
                "limit": len(batch),
            }
            try:
                payload, call = self._get_json(
                    self.document_endpoint,
                    params=query_parameters,
                    cache_response=False,
                )
            except ScientificDataUnavailable as exc:
                failure = {
                    "retrieval_status": "partial",
                    "reason_code": "PRIMARY_DOCUMENT_LOOKUP_UNAVAILABLE",
                    "detail": str(exc)[:500],
                    "provenance": deepcopy(getattr(exc, "provenance", {})),
                    "fixture": False,
                }
                for document_id in batch:
                    failures_by_id[document_id] = deepcopy(failure)
                continue
            documents = payload.get("documents")
            if not isinstance(documents, list):
                failure = {
                    "retrieval_status": "partial",
                    "reason_code": "PRIMARY_DOCUMENT_RESPONSE_INVALID",
                    "detail": "ChEMBL document response has no document list",
                    "provenance": call.as_dict(),
                    "fixture": False,
                }
                for document_id in batch:
                    calls_by_id[document_id] = call
                    failures_by_id[document_id] = deepcopy(failure)
                continue
            allowed = set(batch)
            seen_document_ids: set[str] = set()
            document_rows_invalid = False
            for item in documents:
                if not isinstance(item, Mapping):
                    document_rows_invalid = True
                    continue
                document_id = str(
                    item.get("document_chembl_id") or ""
                ).strip()
                if (
                    not document_id
                    or _CHEMBL_STABLE_ID.fullmatch(document_id) is None
                    or document_id not in allowed
                    or document_id in seen_document_ids
                ):
                    document_rows_invalid = True
                    continue
                seen_document_ids.add(document_id)
            if document_rows_invalid:
                failure = {
                    "retrieval_status": "partial",
                    "reason_code": "PRIMARY_DOCUMENT_RESPONSE_ROWS_INVALID",
                    "detail": (
                        "ChEMBL document response contains malformed, "
                        "duplicate, or unexpected rows"
                    ),
                    "provenance": call.as_dict(),
                    "fixture": False,
                }
                for document_id in batch:
                    calls_by_id[document_id] = call
                    failures_by_id[document_id] = deepcopy(failure)
                continue
            for document_id in batch:
                calls_by_id[document_id] = call
                attestation = {
                    "schema_version": "chembl_document_query_batch_v1",
                    "document_chembl_ids": list(batch),
                    "document_count": len(batch),
                    "batch_cap": _PRIMARY_DOCUMENT_BATCH_SIZE,
                    "selected_document_chembl_id": document_id,
                    "selected_document_in_batch": True,
                    "query_parameters": deepcopy(query_parameters),
                    "query_parameter_hash": call.parameter_hash,
                    "call_parameter_hash": call.parameter_hash,
                    "truncated": False,
                    "fixture": False,
                }
                attestation["attestation_sha256"] = _canonical_sha256(
                    attestation
                )
                batch_attestations_by_id[document_id] = attestation
            for item in documents:
                if not isinstance(item, dict):
                    continue
                document_id = str(
                    item.get("document_chembl_id") or ""
                ).strip()
                if document_id in allowed:
                    resolved[document_id] = dict(item)
        return (
            resolved,
            calls_by_id,
            failures_by_id,
            batch_attestations_by_id,
        )

    def _fetch_pubmed_publication_types(
        self,
        documents_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Verify every selected document PMID in bounded official batches.

        This request retrieves ESummary metadata only.  No PubMed abstract or
        full text is requested or stored.  A missing/failed exact PMID result
        remains unverified and never falls back to a ChEMBL title or DOI.
        Production requests use the dedicated one-attempt PubMed transport,
        never the ChEMBL pool.  The injected-client compatibility seam is
        retained only for tests that intentionally omit a primary client.
        """

        exact_pmids = sorted(
            {
                str(document.get("pubmed_id") or "").strip()
                for document in documents_by_id.values()
                if str(document.get("pubmed_id") or "").strip().isdigit()
            },
            key=lambda value: (int(value), value),
        )
        if not exact_pmids:
            return {}
        output: dict[str, dict[str, Any]] = {}
        use_dedicated_transport = not isinstance(
            self.primary_search_client,
            _UnconfiguredPrimarySearchClient,
        )
        pubmed_transport = (
            self.primary_search_client
            if use_dedicated_transport
            else self.client
        )
        batch_count = (
            len(exact_pmids) + _PRIMARY_PUBMED_BATCH_MAX_IDS - 1
        ) // _PRIMARY_PUBMED_BATCH_MAX_IDS
        for batch_index, start in enumerate(
            range(0, len(exact_pmids), _PRIMARY_PUBMED_BATCH_MAX_IDS),
            start=1,
        ):
            requested_pmids = exact_pmids[
                start : start + _PRIMARY_PUBMED_BATCH_MAX_IDS
            ]
            batch = {
                "requested_pmids": requested_pmids,
                "requested_count": len(requested_pmids),
                "eligible_pmid_count": len(requested_pmids),
                "batch_cap": _PRIMARY_PUBMED_BATCH_MAX_IDS,
                "truncated": False,
                "single_batch": batch_count == 1,
                "batch_index": batch_index,
                "batch_count": batch_count,
                "global_eligible_pmid_count": len(exact_pmids),
            }
            query_params = {
                "db": "pubmed",
                "id": ",".join(requested_pmids),
                "retmode": "json",
                "version": "2.0",
            }
            try:
                payload, call = self._get_json(
                    self.pubmed_summary_endpoint,
                    params=query_params,
                    cache_response=False,
                    source="PubMed",
                    transport_client=pubmed_transport,
                    gateway_attempt_limit_override=(
                        1 if use_dedicated_transport else None
                    ),
                )
            except ScientificDataUnavailable as exc:
                failure = deepcopy(getattr(exc, "provenance", {}))
                for pmid in requested_pmids:
                    output[pmid] = {
                        "schema_version": _PUBMED_TYPE_SCHEMA,
                        "verification_status": "unverified",
                        "retrieval_status": "partial",
                        "reason_code": "PUBMED_TYPE_LOOKUP_UNAVAILABLE",
                        "pmid": pmid,
                        "publication_types": [],
                        "required_publication_type": "Journal Article",
                        "excluded_publication_types": [],
                        "type_policy": _PUBMED_TYPE_POLICY,
                        "batch": deepcopy(batch),
                        "query": failure,
                        "fixture": False,
                    }
                continue

            raw_result = payload.get("result")
            result_mapping_valid = isinstance(raw_result, Mapping)
            result = raw_result if result_mapping_valid else {}
            raw_uids = result.get("uids")
            observed_uids = (
                [str(item).strip() for item in raw_uids]
                if isinstance(raw_uids, list)
                else []
            )
            observed_uid_set_valid = (
                isinstance(raw_uids, list)
                and all(
                    isinstance(item, str) and item.strip().isdigit()
                    for item in raw_uids
                )
                and len(observed_uids) == len(set(observed_uids))
                and set(observed_uids).issubset(set(requested_pmids))
            )
            for pmid in requested_pmids:
                reason_code: str | None = None
                publication_types: list[str] = []
                excluded_types: list[str] = []
                verified = False
                record_schema_invalid = False
                record = result.get(pmid)
                if not observed_uid_set_valid:
                    reason_code = "PUBMED_RESULT_UID_SET_INVALID"
                    record_schema_invalid = True
                elif pmid not in observed_uids:
                    reason_code = "PUBMED_EXACT_PMID_RESULT_MISSING"
                elif not isinstance(record, Mapping):
                    reason_code = "PUBMED_RESULT_RECORD_SCHEMA_INVALID"
                    record_schema_invalid = True
                else:
                    record_uid = str(record.get("uid") or "").strip()
                    if record_uid != pmid:
                        reason_code = "PUBMED_RESULT_PMID_MISMATCH"
                        record_schema_invalid = True
                    elif (
                        not isinstance(record.get("pubtype"), list)
                        or any(
                            not isinstance(item, str)
                            or not item
                            or item != " ".join(item.split())
                            or "\r" in item
                            or "\n" in item
                            or len(item) > 200
                            for item in (
                                record.get("pubtype")
                                if isinstance(record.get("pubtype"), list)
                                else []
                            )
                        )
                    ):
                        reason_code = "PUBMED_PUBLICATION_TYPE_SCHEMA_INVALID"
                        record_schema_invalid = True
                    else:
                        publication_types = _normalized_pubmed_types(
                            record.get("pubtype")
                        )
                        if len(publication_types) != len(record["pubtype"]):
                            reason_code = (
                                "PUBMED_PUBLICATION_TYPE_SCHEMA_INVALID"
                            )
                            record_schema_invalid = True
                            publication_types = []
                        else:
                            verified, excluded_types = _pubmed_type_decision(
                                publication_types
                            )
                            if excluded_types:
                                reason_code = (
                                    "PUBMED_EXCLUDED_PUBLICATION_TYPE"
                                )
                            elif not verified:
                                reason_code = (
                                    "PUBMED_JOURNAL_ARTICLE_TYPE_MISSING"
                                )
                output[pmid] = {
                    "schema_version": _PUBMED_TYPE_SCHEMA,
                    "verification_status": (
                        "verified" if verified else "unverified"
                    ),
                    "retrieval_status": (
                        "completed"
                        if (
                            result_mapping_valid
                            and observed_uid_set_valid
                            and not record_schema_invalid
                        )
                        else "partial"
                    ),
                    "reason_code": reason_code,
                    "pmid": pmid,
                    "publication_types": publication_types,
                    "required_publication_type": "Journal Article",
                    "excluded_publication_types": excluded_types,
                    "type_policy": _PUBMED_TYPE_POLICY,
                    "batch": {
                        **deepcopy(batch),
                        "observed_result_uids": observed_uids,
                    },
                    "query": call.as_dict(),
                    "fixture": False,
                }
        return output

    @staticmethod
    def _primary_source_attestation(
        *,
        activity_id: str,
        chembl_id: str,
        target_chembl_id: str,
        candidate_aliases: Sequence[str],
        candidate_alias_conflicts: Sequence[str],
        competitor_alias_provenance: Mapping[str, Any],
        target: TargetCandidate,
        measurement: ActivityMeasurement,
        direction_rule: str,
        mechanism_identity: Mapping[str, Any] | None,
        raw_activity: Mapping[str, Any],
        assay_context: Mapping[str, Any] | None,
        document_chembl_id: str,
        document: Mapping[str, Any] | None,
        document_call: HTTPCallProvenance | None,
        document_failure: Mapping[str, Any] | None,
        document_query_batch: Mapping[str, Any] | None,
        pubmed_types_by_pmid: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        raw_standard_type = raw_activity.get("standard_type")
        raw_standard_relation = raw_activity.get("standard_relation")
        raw_standard_value = raw_activity.get("standard_value")
        raw_standard_units = raw_activity.get("standard_units")
        raw_value_matches = False
        if not isinstance(raw_standard_value, bool):
            try:
                raw_value_matches = Decimal(str(raw_standard_value)) == Decimal(
                    str(measurement.value)
                )
            except (InvalidOperation, ValueError):
                raw_value_matches = False
        deposited_activity = {
            "schema_version": "chembl_deposited_activity_exact_v1",
            "standard_type": raw_standard_type,
            "standard_relation": raw_standard_relation,
            "standard_value": raw_standard_value,
            "standard_units": raw_standard_units,
            "typed_exact_match": (
                raw_standard_type == measurement.endpoint
                and raw_standard_relation == measurement.relation
                and raw_value_matches
                and raw_standard_units == measurement.unit
            ),
            "missing_values_imputed": False,
            "normalization_applied": False,
            "unit_conversion_applied": False,
        }
        assay_id = str(raw_activity.get("assay_chembl_id") or "").strip()
        engagement_assessment = _assay_engagement_assessment(
            assay_type=measurement.assay_type,
            assay_context=assay_context,
            assay_chembl_id=assay_id,
            target_chembl_id=target_chembl_id,
            target_symbol=target.symbol,
        )
        resolved_assay_context = str(
            engagement_assessment["assay_context"]
        )
        direct_engagement = bool(
            engagement_assessment["direct_single_protein_engagement"]
        )
        mechanism_claim = _candidate_mechanism_claim(
            target_symbol=target.symbol,
            measurement=measurement,
            direction_rule=direction_rule,
            mechanism_identity=mechanism_identity,
            assay_context_label=resolved_assay_context,
        )
        binding = {
            "activity_id": activity_id,
            "molecule_chembl_id": chembl_id,
            "target_chembl_id": target_chembl_id,
            "assay_chembl_id": raw_activity.get("assay_chembl_id"),
            "document_chembl_id": document_chembl_id or None,
            "endpoint": measurement.endpoint,
            "relation": measurement.relation,
            "value": measurement.value,
            "unit": measurement.unit,
            "assay_type": measurement.assay_type,
            "assay_context": resolved_assay_context,
            "direct_single_protein_engagement": direct_engagement,
            "direction_evidence_rule": direction_rule,
            "mechanism_identity_sha256": (
                _canonical_sha256(dict(mechanism_identity))
                if isinstance(mechanism_identity, Mapping)
                else None
            ),
            "mechanism_claim": mechanism_claim,
            "mechanism_claim_sha256": _canonical_sha256(mechanism_claim),
            "deposited_activity": deposited_activity,
        }
        base: dict[str, Any] = {
            "schema_version": _PRIMARY_EVIDENCE_SCHEMA_VERSION,
            "verification_status": "unverified",
            "retrieval_status": "completed",
            "is_primary_support": False,
            "reason_code": None,
            "document_chembl_id": document_chembl_id or None,
            "doc_type": None,
            "pmid": None,
            "doi": None,
            "title": None,
            "publication_types": [],
            "chembl_reported_publication_types": [],
            "pubmed_type_verification": None,
            "source_url": None,
            "retrieved_field": None,
            "excerpt": None,
            "excerpt_location": None,
            "excerpt_hash": None,
            "content_hash": None,
            "content_length": None,
            "full_text_stored": False,
            "hash_algorithm": "sha256",
            "activity_binding": binding,
            "candidate_exact_aliases": list(candidate_aliases),
            "candidate_alias_source": _CANDIDATE_ALIAS_SOURCE,
            "candidate_alias_conflicts": list(candidate_alias_conflicts),
            "competitor_alias_provenance": deepcopy(
                dict(competitor_alias_provenance)
            ),
            "relevance": None,
            "document_query": (
                document_call.as_dict()
                if document_call is not None
                else deepcopy(dict(document_failure or {}))
            ),
            "document_query_batch": (
                deepcopy(dict(document_query_batch))
                if document_query_batch is not None
                else None
            ),
        }
        if not candidate_aliases:
            base["reason_code"] = "CANDIDATE_EXACT_ALIAS_MISSING"
            return base
        if candidate_alias_conflicts:
            base["reason_code"] = "CANDIDATE_ALIAS_CONFLICT"
            return base
        if (
            competitor_alias_provenance.get("truncated") is not False
            or competitor_alias_provenance.get("policy")
            != "all_bounded_pool_candidate_aliases_except_current_v1"
        ):
            base["reason_code"] = "CANDIDATE_POOL_ALIAS_INVENTORY_UNVERIFIED"
            return base
        if deposited_activity["typed_exact_match"] is not True:
            base["reason_code"] = "DEPOSITED_ACTIVITY_TYPED_BINDING_MISMATCH"
            return base
        if not document_chembl_id:
            base["reason_code"] = "ACTIVITY_DOCUMENT_ID_MISSING"
            return base
        if document_failure is not None:
            base["retrieval_status"] = "partial"
            base["reason_code"] = str(
                document_failure.get("reason_code")
                or "PRIMARY_DOCUMENT_LOOKUP_UNAVAILABLE"
            )
            return base
        if document is None:
            base["reason_code"] = "CHEMBL_DOCUMENT_NOT_FOUND"
            return base

        exact_document_id = str(
            document.get("document_chembl_id") or ""
        ).strip()
        if exact_document_id != document_chembl_id:
            base["reason_code"] = "CHEMBL_DOCUMENT_ID_MISMATCH"
            return base
        doc_type = str(document.get("doc_type") or "").strip()
        pmid = str(document.get("pubmed_id") or "").strip()
        if pmid and not pmid.isdigit():
            pmid = ""
        doi = str(document.get("doi") or "").strip()
        title = _normalize_document_text(document.get("title")) or None
        chembl_publication_types: list[str] = []
        for key in (
            "publication_types",
            "publication_type",
            "pubmed_types",
            "pubmed_type",
            "article_type",
        ):
            raw_types = document.get(key)
            values = raw_types if isinstance(raw_types, list) else [raw_types]
            for value in values:
                normalized_type = _normalize_document_text(value)
                if (
                    normalized_type
                    and normalized_type not in chembl_publication_types
                ):
                    chembl_publication_types.append(normalized_type)
        pubmed_type = (
            deepcopy(dict(pubmed_types_by_pmid.get(pmid, {})))
            if pmid
            else None
        )
        publication_types = (
            list(pubmed_type.get("publication_types", []))
            if isinstance(pubmed_type, Mapping)
            and isinstance(pubmed_type.get("publication_types"), list)
            else []
        )
        base.update(
            {
                "doc_type": doc_type or None,
                "pmid": pmid or None,
                "doi": doi or None,
                "title": title,
                "publication_types": publication_types,
                "chembl_reported_publication_types": (
                    chembl_publication_types
                ),
                "pubmed_type_verification": pubmed_type,
                "source_url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{quote(pmid)}/"
                    if pmid
                    else f"https://doi.org/{quote(doi, safe='/')}"
                    if doi
                    else (
                        "https://www.ebi.ac.uk/chembl/explore/document/"
                        f"{quote(document_chembl_id)}"
                    )
                ),
            }
        )
        if doc_type.casefold() != "publication":
            base["reason_code"] = "DOCUMENT_NOT_PRIMARY_PUBLICATION"
            return base
        review_markers = [
            doc_type,
            title or "",
            *chembl_publication_types,
            *publication_types,
        ]
        if any(
            re.search(
                r"(?<!\w)(?:review|systematic review|meta analysis|guideline|editorial|letter|comment)(?!\w)",
                _normalize_identity_phrase(value),
            )
            for value in review_markers
        ):
            base["reason_code"] = "DOCUMENT_REVIEW_NOT_PRIMARY_ACTIVITY_REPORT"
            return base
        if not pmid:
            base["reason_code"] = (
                "PUBMED_ID_REQUIRED_FOR_PRIMARY_TYPE"
                if doi
                else "PRIMARY_IDENTIFIER_MISSING"
            )
            return base
        if (
            not isinstance(pubmed_type, Mapping)
            or pubmed_type.get("schema_version") != _PUBMED_TYPE_SCHEMA
            or pubmed_type.get("pmid") != pmid
            or pubmed_type.get("verification_status") != "verified"
            or pubmed_type.get("retrieval_status") != "completed"
            or pubmed_type.get("type_policy") != _PUBMED_TYPE_POLICY
        ):
            base["reason_code"] = str(
                pubmed_type.get("reason_code")
                or "PUBMED_TYPE_VERIFICATION_MISSING"
                if isinstance(pubmed_type, Mapping)
                else "PUBMED_TYPE_VERIFICATION_MISSING"
            )
            return base

        abstract = _normalize_document_text(document.get("abstract"))
        if not abstract:
            base.update(
                {
                    "verification_status": "metadata_only",
                    "reason_code": "PRIMARY_ABSTRACT_NOT_AVAILABLE",
                }
            )
            return base
        selected = _select_primary_excerpt(
            abstract,
            candidate_aliases=candidate_aliases,
            competitor_aliases=competitor_alias_provenance.get(
                "competitor_exact_aliases", []
            ),
            target=target,
            measurement=measurement,
        )
        content_hash = sha256(abstract.encode("utf-8")).hexdigest()
        base.update(
            {
                "retrieved_field": "abstract",
                "content_hash": content_hash,
                "content_length": len(abstract),
            }
        )
        if selected is None:
            base["reason_code"] = "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED"
            return base
        excerpt = str(selected["excerpt"])
        excerpt_hash = sha256(excerpt.encode("utf-8")).hexdigest()
        base.update(
            {
                "verification_status": "verified",
                "is_primary_support": True,
                "reason_code": None,
                "excerpt": excerpt,
                "excerpt_location": selected["excerpt_location"],
                "excerpt_hash": excerpt_hash,
                "content_hash": selected["normalized_content_hash"],
                "content_length": selected["normalized_content_length"],
                "relevance": {
                    "matched_groups": selected["matched_groups"],
                    "matched_terms": selected["matched_terms"],
                    "matched_exact_value_unit": selected[
                        "matched_exact_value_unit"
                    ],
                    "binding_pattern": selected["binding_pattern"],
                    "bound_spans": selected["bound_spans"],
                    "competitor_alias_matches": selected[
                        "competitor_alias_matches"
                    ],
                    "excerpt_truncated": selected["excerpt_truncated"],
                    "policy": _PRIMARY_RELEVANCE_POLICY,
                },
            }
        )
        return base

    def _fetch_drug_warnings(
        self,
        chembl_ids: list[str],
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, HTTPCallProvenance],
        dict[str, dict[str, Any]],
        bool,
        dict[str, Any] | None,
    ]:
        """Resolve warning status with transient batch-failure isolation.

        The normal path remains one bounded batch request. If and only if that
        request exhausts a retryable transport/status policy, retry each
        candidate independently. Candidates whose individual warning query is
        still unresolved are omitted by the caller; they are never interpreted
        as having no warning.
        """

        batch_fallback_used = False
        batch_failure: dict[str, Any] | None = None
        try:
            payload, call = self._get_json(
                self.warning_endpoint,
                params={
                    "molecule_chembl_id__in": ",".join(chembl_ids),
                    "limit": max(100, len(chembl_ids) * 10),
                },
            )
        except ScientificDataUnavailable as exc:
            provenance = getattr(exc, "provenance", {})
            if not isinstance(provenance, Mapping) or not provenance.get(
                "retryable", False
            ):
                raise
            batch_fallback_used = True
            batch_failure = deepcopy(dict(provenance))
            grouped: dict[str, list[dict[str, Any]]] = {}
            calls_by_id: dict[str, HTTPCallProvenance] = {}
            failures_by_id: dict[str, dict[str, Any]] = {}
            for index, chembl_id in enumerate(chembl_ids):
                remaining = self._remaining_operation_seconds()
                if remaining is not None and remaining <= 0:
                    for unresolved_id in chembl_ids[index:]:
                        failures_by_id[unresolved_id] = {
                            "source": "ChEMBL",
                            "error_type": "operation_timeout",
                            "operation": "candidate_regulatory_warning_screen",
                            "exhausted": True,
                            "cache_hit": False,
                            "fixture": False,
                        }
                    break
                try:
                    individual_payload, individual_call = self._get_json(
                        self.warning_endpoint,
                        params={
                            "molecule_chembl_id__in": chembl_id,
                            "limit": 100,
                        },
                    )
                    individual_grouped, individual_issues = (
                        self._parse_drug_warnings(
                            individual_payload,
                            requested_ids=[chembl_id],
                        )
                    )
                except ScientificDataUnavailable as individual_exc:
                    individual_provenance = getattr(
                        individual_exc, "provenance", {}
                    )
                    failures_by_id[chembl_id] = (
                        dict(individual_provenance)
                        if isinstance(individual_provenance, Mapping)
                        else {
                            "source": "ChEMBL",
                            "error_type": type(individual_exc).__name__,
                            "fixture": False,
                        }
                    )
                    continue
                if individual_issues:
                    failures_by_id[chembl_id] = {
                        "source": "ChEMBL",
                        "retrieval_status": "partial",
                        "reason_code": "DRUG_WARNING_RESPONSE_ROWS_INVALID",
                        "query": individual_call.as_dict(),
                        "issue_count": len(individual_issues),
                        "issues_sha256": _canonical_sha256(individual_issues),
                        "fixture": False,
                    }
                    continue
                calls_by_id[chembl_id] = individual_call
                grouped[chembl_id] = individual_grouped.get(chembl_id, [])
            return (
                grouped,
                calls_by_id,
                failures_by_id,
                batch_fallback_used,
                batch_failure,
            )

        grouped, issues = self._parse_drug_warnings(
            payload,
            requested_ids=chembl_ids,
        )
        if issues:
            batch_failure = {
                "source": "ChEMBL",
                "retrieval_status": "partial",
                "reason_code": "DRUG_WARNING_RESPONSE_ROWS_INVALID",
                "query": call.as_dict(),
                "issue_count": len(issues),
                "issues_sha256": _canonical_sha256(issues),
                "fixture": False,
            }
        return (
            grouped,
            {chembl_id: call for chembl_id in chembl_ids},
            {},
            batch_fallback_used,
            batch_failure,
        )

    @staticmethod
    def _parse_drug_warnings(
        payload: Mapping[str, Any],
        *,
        requested_ids: Sequence[str],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        raw_warnings = payload.get("drug_warnings", [])
        if not isinstance(raw_warnings, list):
            raise ScientificDataUnavailable("ChEMBL drug-warning response has no warning list")
        grouped: dict[str, list[dict[str, Any]]] = {}
        requested = set(requested_ids)
        issues: list[dict[str, Any]] = []
        observed_rows: set[str] = set()
        retained_keys = (
            "warning_type",
            "warning_class",
            "warning_description",
            "warning_country",
            "warning_year",
            "efo_term",
            "efo_id",
        )
        for index, item in enumerate(raw_warnings):
            if not isinstance(item, dict):
                issues.append({"row_index": index, "reason": "ROW_NOT_OBJECT"})
                continue
            chembl_id = str(item.get("molecule_chembl_id") or "").strip()
            if not chembl_id:
                issues.append({"row_index": index, "reason": "MOLECULE_ID_MISSING"})
                continue
            if chembl_id not in requested:
                issues.append(
                    {
                        "row_index": index,
                        "reason": "UNEXPECTED_MOLECULE_ID",
                        "molecule_chembl_id": chembl_id,
                    }
                )
                continue
            row_hash = _canonical_sha256(item)
            if row_hash in observed_rows:
                issues.append(
                    {
                        "row_index": index,
                        "reason": "DUPLICATE_WARNING_ROW",
                        "molecule_chembl_id": chembl_id,
                    }
                )
                continue
            observed_rows.add(row_hash)
            grouped.setdefault(chembl_id, []).append(
                {key: item.get(key) for key in retained_keys if item.get(key) is not None}
            )
        return grouped, issues

    def _ensure_normalized(self, target: TargetCandidate) -> TargetCandidate:
        cached = self._normalized_targets.get(target.symbol.casefold())
        if cached is not None and cached.uniprot_id == target.uniprot_id:
            return cached
        return self.normalize_target(target)

    def _remember_seed(self, target: TargetCandidate, seed: CandidateSeed) -> None:
        key = (target.symbol.casefold(), seed.name.casefold())
        self._seed_cache[key] = seed
        self._activity_cache[key] = tuple(seed.activities)

    @contextmanager
    def _chembl_operation_budget(self, operation: str) -> Iterator[None]:
        """Apply one deadline to nested calls without resetting it per request."""

        current = self._operation_context.get()
        if current is not None:
            yield
            return
        started = self._clock()
        token = self._operation_context.set(
            {
                "operation": operation,
                "started": started,
                "deadline": started + self.operation_timeout_seconds,
            }
        )
        try:
            yield
        finally:
            self._operation_context.reset(token)

    def _remaining_operation_seconds(self) -> float | None:
        context = self._operation_context.get()
        if context is None:
            return None
        return max(0.0, float(context["deadline"]) - self._clock())

    def _operation_name(self) -> str | None:
        context = self._operation_context.get()
        return str(context["operation"]) if context is not None else None

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(
                    max(0.0, float(retry_after)),
                    self.max_retry_backoff_seconds,
                )
            except ValueError:
                pass
        return min(
            self.retry_backoff_seconds * (2 ** (attempt - 1)),
            self.max_retry_backoff_seconds,
        )

    def _sleep_within_operation_budget(self, delay: float) -> None:
        if delay <= 0:
            return
        remaining = self._remaining_operation_seconds()
        if remaining is not None:
            delay = min(delay, max(0.0, remaining))
        if delay > 0:
            self._sleeper(delay)

    def _require_chembl_version(self) -> str:
        if self._chembl_version is not None:
            return self._chembl_version
        if self._chembl_version_probe_complete:
            return "unresolved"
        try:
            payload, call = self._get_json(self.status_endpoint, params={})
        except ScientificDataUnavailable as exc:
            provenance = getattr(exc, "provenance", {})
            error_type = (
                str(provenance.get("error_type") or "")
                if isinstance(provenance, Mapping)
                else ""
            )
            remaining_budget = (
                _safe_float(
                    provenance.get("remaining_operation_budget_seconds")
                )
                if isinstance(provenance, Mapping)
                else None
            )
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("retryable") is not True
                or error_type == "operation_timeout"
                or (
                    remaining_budget is not None
                    and remaining_budget <= 0
                )
            ):
                # TLS verification failures and malformed/non-authoritative
                # responses remain fail-closed.  Only a bounded retryable
                # outage of the release-metadata endpoint is isolated here.
                raise
            # The ChEMBL status endpoint is release metadata, not the
            # authoritative target/activity dataset.  A metadata-only outage
            # must remain visible in provenance but must not suppress exact
            # UniProt target resolution and measured-activity requests that
            # are independently hashed and validated below.  If those data
            # endpoints share the transport outage they still fail normally.
            self._chembl_version_probe_complete = True
            self._chembl_version_resolution = {
                "status": "unavailable",
                "reason_code": "CHEMBL_VERSION_METADATA_UNAVAILABLE",
                "detail": str(exc),
                "provenance": deepcopy(dict(provenance)),
            }
            return "unresolved"
        candidate: object | None = None
        for key in ("chembl_db_version", "chembl_release", "version"):
            if key in payload and payload.get(key) not in (None, ""):
                candidate = payload.get(key)
                break
        if candidate is None:
            self._chembl_version_probe_complete = True
            self._chembl_version_resolution = {
                "status": "unavailable",
                "reason_code": "CHEMBL_VERSION_METADATA_MISSING",
                "detail": "ChEMBL status did not expose a database version",
                "provenance": call.as_dict(),
            }
            return "unresolved"
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, (str, int, float))
            or (
                isinstance(candidate, float)
                and not Decimal(str(candidate)).is_finite()
            )
            or not str(candidate).strip()
            or str(candidate) != str(candidate).strip()
            or len(str(candidate)) > 100
            or "\r" in str(candidate)
            or "\n" in str(candidate)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._ -]{0,99}", str(candidate)
            )
            is None
        ):
            self._chembl_version_probe_complete = True
            self._chembl_version_resolution = {
                "status": "unavailable",
                "reason_code": "CHEMBL_VERSION_METADATA_INVALID",
                "detail": "ChEMBL status exposed a non-scalar or malformed version",
                "provenance": call.as_dict(),
            }
            return "unresolved"
        self._chembl_version = str(candidate)
        self._chembl_version_probe_complete = True
        self._chembl_version_resolution = {
            "status": "completed",
            "reason_code": None,
            "detail": None,
            "provenance": call.as_dict(),
        }
        return self._chembl_version

    def _get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any],
        cache_response: bool = True,
        source: str = "ChEMBL",
        transport_client: HTTPClient | None = None,
        gateway_attempt_limit_override: int | None = None,
    ) -> tuple[dict[str, Any], HTTPCallProvenance]:
        request_client = (
            transport_client if transport_client is not None else self.client
        )
        if (
            gateway_attempt_limit_override is not None
            and gateway_attempt_limit_override not in {1, 2, 3}
        ):
            raise ValueError("gateway attempt override must be between 1 and 3")
        started_at = _utc_iso()
        started = self._clock()
        parameter_payload = json.dumps(
            dict(params),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        parameter_hash = sha256(parameter_payload).hexdigest()
        parameter_keys = tuple(sorted(str(key) for key in params))
        cache_key = (endpoint, parameter_hash)
        operation = self._operation_name()
        cached = self._json_cache.get(cache_key) if cache_response else None
        if cached is not None:
            cached_payload, origin = cached
            call = HTTPCallProvenance(
                source=source,
                endpoint=endpoint,
                parameter_keys=parameter_keys,
                parameter_hash=parameter_hash,
                timeout_seconds=0.0,
                status_code=origin.status_code,
                response_sha256=origin.response_sha256,
                started_at=started_at,
                finished_at=_utc_iso(),
                runtime_seconds=round(self._clock() - started, 6),
                attempt_count=0,
                max_attempts=(
                    gateway_attempt_limit_override
                    or self.max_http_attempts
                ),
                exhausted=False,
                cache_hit=True,
                operation=operation,
                operation_budget_seconds=self.operation_timeout_seconds,
                attempts=(),
            )
            self._http_calls.append(call)
            return deepcopy(cached_payload), call

        # The MA2 ScientificHttpClient already owns bounded retries. Do not
        # wrap it in a second retry loop: that would multiply attempts
        # and could turn one slow public API into a retry storm. Injected/simple
        # clients receive the same total-attempt policy here.
        client_attempt_limit = getattr(request_client, "max_attempts", 1)
        if not isinstance(client_attempt_limit, int) or client_attempt_limit < 1:
            client_attempt_limit = 1
        gateway_attempt_limit = (
            gateway_attempt_limit_override
            if gateway_attempt_limit_override is not None
            else 1
            if client_attempt_limit > 1
            else self.max_http_attempts
        )
        total_attempt_limit = max(client_attempt_limit, gateway_attempt_limit)
        attempts: list[dict[str, Any]] = []

        def client_failure_count() -> int | None:
            recent_failures = getattr(request_client, "recent_failures", None)
            if not callable(recent_failures):
                return None
            return len(recent_failures())

        def observed_transport_attempts(
            before: int | None,
            *,
            completed: bool,
        ) -> int:
            after = client_failure_count()
            failures = (
                max(0, after - before)
                if before is not None and after is not None
                else 0
            )
            return max(1, failures + (1 if completed else 0))

        def raise_unavailable(
            message: str,
            *,
            error_type: str,
            cause: BaseException | None,
            retryable: bool,
            status_code: int | None = None,
            exhausted: bool = False,
        ) -> None:
            unavailable = ScientificDataUnavailable(message)
            unavailable.provenance = {
                "source": source,
                "endpoint": endpoint,
                "parameter_keys": list(parameter_keys),
                "parameter_hash": parameter_hash,
                "timeout_seconds": self.timeout_seconds,
                "operation": operation,
                "operation_budget_seconds": self.operation_timeout_seconds,
                "remaining_operation_budget_seconds": (
                    round(remaining, 6)
                    if (remaining := self._remaining_operation_seconds()) is not None
                    else None
                ),
                "error_type": error_type,
                "exception_type": type(cause).__name__ if cause is not None else None,
                "status_code": status_code,
                "retryable": retryable,
                "attempt_count": sum(
                    int(item.get("transport_attempt_count", 1))
                    for item in attempts
                ),
                "max_attempts": total_attempt_limit,
                "exhausted": exhausted,
                "cache_hit": False,
                "attempts": [dict(item) for item in attempts],
                "started_at": started_at,
                "finished_at": _utc_iso(),
                "runtime_seconds": round(self._clock() - started, 6),
                "fixture": False,
            }
            if cause is None:
                raise unavailable
            raise unavailable from cause

        response: Any = None
        payload: Any = None
        for attempt in range(1, gateway_attempt_limit + 1):
            remaining = self._remaining_operation_seconds()
            if remaining is not None and remaining <= 0:
                raise_unavailable(
                    f"{source} operation budget exhausted before request completion",
                    error_type="operation_timeout",
                    cause=None,
                    retryable=True,
                    exhausted=True,
                )
            # Reserve a fair slice for retries already managed inside the
            # pooled client. A single socket read can never consume the whole
            # multi-call target/candidate operation budget.
            request_timeout = self.timeout_seconds
            if remaining is not None:
                request_timeout = min(
                    request_timeout,
                    max(0.05, remaining / client_attempt_limit),
                )
            attempt_started = self._clock()
            failures_before = client_failure_count()
            try:
                response = request_client.get(
                    endpoint,
                    params=dict(params),
                    timeout=request_timeout,
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.TransportError as exc:
                category = classify_transport_error(exc)
                retryable = category in self._retryable_transport_categories
                outer_retry = retryable and attempt < gateway_attempt_limit
                attempts.append(
                    {
                        "attempt": attempt,
                        "transport_attempt_count": observed_transport_attempts(
                            failures_before,
                            completed=False,
                        ),
                        "timeout_seconds": round(request_timeout, 6),
                        "status": "error",
                        "error_type": category,
                        "exception_type": type(exc).__name__,
                        "runtime_seconds": round(self._clock() - attempt_started, 6),
                        "retrying": outer_retry,
                    }
                )
                if outer_retry:
                    self._sleep_within_operation_budget(
                        self._retry_delay(attempt, None)
                    )
                    continue
                raise_unavailable(
                    f"{source} request unavailable ({category})",
                    error_type=category,
                    cause=exc,
                    retryable=retryable,
                    exhausted=retryable,
                )
            except httpx.HTTPStatusError as exc:
                status_code = int(exc.response.status_code)
                retryable = status_code in self._retryable_status_codes
                outer_retry = retryable and attempt < gateway_attempt_limit
                attempts.append(
                    {
                        "attempt": attempt,
                        "transport_attempt_count": observed_transport_attempts(
                            failures_before,
                            completed=False,
                        ),
                        "timeout_seconds": round(request_timeout, 6),
                        "status": "error",
                        "error_type": f"upstream_http_{status_code}",
                        "status_code": status_code,
                        "runtime_seconds": round(self._clock() - attempt_started, 6),
                        "retrying": outer_retry,
                    }
                )
                if outer_retry:
                    retry_after = exc.response.headers.get("Retry-After")
                    self._sleep_within_operation_budget(
                        self._retry_delay(attempt, retry_after)
                    )
                    continue
                raise_unavailable(
                    f"{source} request unavailable (HTTP {status_code})",
                    error_type=f"upstream_http_{status_code}",
                    cause=exc,
                    retryable=retryable,
                    status_code=status_code,
                    exhausted=retryable,
                )
            except (TypeError, ValueError) as exc:
                attempts.append(
                    {
                        "attempt": attempt,
                        "transport_attempt_count": observed_transport_attempts(
                            failures_before,
                            completed=False,
                        ),
                        "timeout_seconds": round(request_timeout, 6),
                        "status": "error",
                        "error_type": "invalid_json_response",
                        "exception_type": type(exc).__name__,
                        "runtime_seconds": round(self._clock() - attempt_started, 6),
                        "retrying": False,
                    }
                )
                raise_unavailable(
                    f"{source} response could not be decoded as JSON",
                    error_type="invalid_json_response",
                    cause=exc,
                    retryable=False,
                )
            else:
                attempts.append(
                    {
                        "attempt": attempt,
                        "transport_attempt_count": observed_transport_attempts(
                            failures_before,
                            completed=True,
                        ),
                        "timeout_seconds": round(request_timeout, 6),
                        "status": "completed",
                        "status_code": int(response.status_code),
                        "runtime_seconds": round(self._clock() - attempt_started, 6),
                        "retrying": False,
                    }
                )
                break

        if not isinstance(payload, dict):
            raise_unavailable(
                f"{source} returned a non-object JSON response",
                error_type="invalid_json_schema",
                cause=None,
                retryable=False,
            )
        raw_content = getattr(response, "content", None)
        if not isinstance(raw_content, bytes):
            raw_content = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        call = HTTPCallProvenance(
            source=source,
            endpoint=endpoint,
            parameter_keys=parameter_keys,
            parameter_hash=parameter_hash,
            timeout_seconds=request_timeout,
            status_code=int(response.status_code),
            response_sha256=sha256(raw_content).hexdigest(),
            started_at=started_at,
            finished_at=_utc_iso(),
            runtime_seconds=round(self._clock() - started, 6),
            attempt_count=sum(
                int(item.get("transport_attempt_count", 1)) for item in attempts
            ),
            max_attempts=total_attempt_limit,
            exhausted=False,
            cache_hit=False,
            operation=operation,
            operation_budget_seconds=self.operation_timeout_seconds,
            attempts=tuple(attempts),
        )
        if cache_response:
            self._json_cache[cache_key] = (deepcopy(payload), call)
        self._http_calls.append(call)
        return deepcopy(payload), call


__all__ = [
    "HTTPCallProvenance",
    "LiveScientificGateway",
    "NoMeasuredLigandCandidates",
]
