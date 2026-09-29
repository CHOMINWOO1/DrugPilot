"""Public-data-backed retrieval and merging of real repurposing candidates.

Exact DrugCentral target-interaction rows are combined with exact-target,
typed measured ligands from a live scientific delegate (normally ChEMBL).
Chemical records merge only on exact InChIKey or exact SMILES identity. The
module never invents candidates, guesses a missing activity unit, promotes a
fuzzy name match, or copies the requested intervention direction onto binding
evidence that does not establish it.

The gateway also delegates docking, ADMET and de-novo operations. Unsupported
calculations and zero-qualified searches fail closed without fixture fallback.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateRoute,
    EvidenceRef,
    GatewayReadiness,
    ReadinessStatus,
    ScientificBackendKind,
    TargetCandidate,
    ToolReadiness,
)
from drugpilot.tools.protocols import CandidateSeed, CandidateToolGateway
from drugpilot.tools.public_chemistry import (
    BindingDBAdapter,
    ChEMBLAdapter,
    ChemistryLookup,
    PubChemAdapter,
)
from drugpilot.tools.primary_abstract_paper_native_search_v6 import (
    MINIMUM_VERIFIED_CANDIDATES,
    PAPER_NATIVE_V6_AGGREGATION_POLICY,
    PAPER_NATIVE_V6_POLICY_TOKEN,
    PaperNativeV6ContractError,
    aggregate_candidate_primary_support_v6,
    publication_types_valid_v6,
)


PAIRED_V4_REQUIRED_POLICY = (
    "dynamic_pubmed_primary_abstract_paired_v4_required"
)
PAPER_NATIVE_V5_REQUIRED_POLICY = (
    "dynamic_pubmed_primary_abstract_paper_native_v5_required"
)
PAPER_NATIVE_V6_REQUIRED_POLICY = PAPER_NATIVE_V6_POLICY_TOKEN
_PAIRED_V4_ENVELOPE_KEY = "primary_abstract_paired_v4"
_PAIRED_V4_ENVELOPE_SCHEMA = "candidate_primary_evidence_paired_v4"
_PAPER_NATIVE_V5_ENVELOPE_KEY = "primary_abstract_paper_native_v5"
_PAPER_NATIVE_V5_ENVELOPE_SCHEMA = (
    "candidate_primary_evidence_paper_native_v5"
)
_PAPER_NATIVE_V5_SOURCE_CONTRACT = "primary_abstract_paper_native_v5"
_PAPER_NATIVE_V5_SEARCH_SCHEMA = "primary_abstract_paper_native_gateway_v5"
_PAPER_NATIVE_V6_ENVELOPE_KEY = "primary_abstract_paper_native_v6"
_PAPER_NATIVE_V6_ENVELOPE_SCHEMA = (
    "candidate_primary_evidence_paper_native_v6"
)
_PAPER_NATIVE_V6_SOURCE_CONTRACT = "primary_abstract_paper_native_v6"
_PAPER_NATIVE_V6_SEARCH_SCHEMA = "primary_abstract_paper_native_gateway_v6"
_PAPER_NATIVE_V6_WRAPPER_SCREEN_SCHEMA = (
    "paper_native_v6_wrapper_atomic_screen_projection/v1"
)
_PAPER_NATIVE_V6_WRAPPER_SCREEN_STATUSES = frozenset(
    {"verified", "rejected", "not_returned"}
)
_PAPER_NATIVE_V6_WRAPPER_SCREEN_REASONS = frozenset(
    {
        "SEED_CARDINALITY_NOT_EXACT",
        "PRIMARY_SOURCE_NOT_MAPPING",
        "PAPER_NATIVE_V6_EXACT_BINDING_INVALID",
        "DELEGATE_NOT_RETURNED",
    }
)
_PAIRED_V4_DISCOVERY_SCHEMA = "bounded_mechanism_first_candidate_identity_pool_v1"
_PAIRED_V4_DISCOVERY_LANE = "chembl_mechanism_first_single_default_v1"
_PAIRED_V4_PRIMARY_POLICY = (
    "same_sentence_primary_pairs_exact_chembl_activity_and_assay_v4"
)
_PAIRED_V4_MAX_MECHANISM_ROWS = 80
_PAIRED_V4_MAX_CANDIDATES = 12
_PAIRED_V4_RETURN_ORDERING = (
    "surviving_fixed_prewarning_clinical_phase_desc_then_"
    "stable_parent_molecule_chembl_id_subsequence_v1"
)


class LiveRepurposingError(RuntimeError):
    """Base class for fail-closed live-repurposing errors."""


class RepurposingSourceUnavailable(LiveRepurposingError):
    """The configured authoritative local source cannot be read."""


class PairedV4RepurposingContractUnavailable(RepurposingSourceUnavailable):
    """The exact paired-v4 delegate contract is absent or contradictory."""

    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.detail = detail
        self.provenance = {
            "operation": "retrieve_candidates",
            "policy": PAIRED_V4_REQUIRED_POLICY,
            "reason_code": reason_code,
            "detail": detail,
            "fixture": False,
            "fixture_fallback_used": False,
            "missing_values_imputed": False,
            **dict(provenance or {}),
        }
        super().__init__(f"{reason_code}: {detail}")


class PaperNativeV5RepurposingContractUnavailable(RepurposingSourceUnavailable):
    """The exact paper-native-v5 delegate contract is absent or contradictory."""

    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.detail = detail
        self.provenance = {
            "operation": "retrieve_candidates",
            "policy": PAPER_NATIVE_V5_REQUIRED_POLICY,
            "reason_code": reason_code,
            "detail": detail,
            "fixture": False,
            "fixture_fallback_used": False,
            "missing_values_imputed": False,
            **dict(provenance or {}),
        }
        super().__init__(f"{reason_code}: {detail}")


class PaperNativeV6RepurposingContractUnavailable(RepurposingSourceUnavailable):
    """The candidate-per-primary-record v6 contract is unavailable."""

    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.detail = detail
        self.provenance = {
            "operation": "retrieve_candidates",
            "policy": PAPER_NATIVE_V6_REQUIRED_POLICY,
            "reason_code": reason_code,
            "detail": detail,
            "fixture": False,
            "fixture_fallback_used": False,
            "missing_values_imputed": False,
            **dict(provenance or {}),
        }
        super().__init__(f"{reason_code}: {detail}")


class NoRealRepurposingCandidates(LiveRepurposingError):
    """No direction-compatible, structure-resolved candidate was found."""

    def __init__(
        self,
        *,
        target: str,
        direction: str,
        matching_rows: int,
        direction_compatible_rows: int,
        unresolved_structures: int,
        source_failures: Sequence[str] = (),
        criteria_rejections: Sequence[str] = (),
        source_search_complete: bool = True,
    ) -> None:
        self.target = target
        self.direction = direction
        self.matching_rows = matching_rows
        self.direction_compatible_rows = direction_compatible_rows
        self.unresolved_structures = unresolved_structures
        self.source_failures = tuple(source_failures)
        self.criteria_rejections = tuple(criteria_rejections)
        # A zero-result search is scientifically different from an unavailable
        # source.  It is conclusive only when the authoritative target rows
        # were read successfully and none are compatible with the requested
        # intervention direction.  Rows that require an unresolved structure
        # remain an incomplete execution and must fail closed.
        self.conclusive_empty_search = (
            source_search_complete
            and not self.source_failures
            and self.direction_compatible_rows == 0
            and self.unresolved_structures == 0
        )
        self.provenance = {
            "operation": "retrieve_candidates",
            "search_completed": self.conclusive_empty_search,
            "source_search_complete": source_search_complete,
            "partial_source_result": not source_search_complete,
            "target": self.target,
            "direction": self.direction,
            "matching_rows": self.matching_rows,
            "direction_compatible_rows": self.direction_compatible_rows,
            "unresolved_structures": self.unresolved_structures,
            "source_failures": list(self.source_failures),
            "criteria_rejections": list(self.criteria_rejections),
            "fixture": False,
            "fixture_fallback_used": False,
            "missing_values_imputed": False,
        }
        failure_detail = (
            f" source_failures={list(self.source_failures)}."
            if self.source_failures
            else ""
        )
        criteria_detail = (
            f" criteria_rejections={list(self.criteria_rejections)}."
            if self.criteria_rejections
            else ""
        )
        super().__init__(
            "No real repurposing candidates were available for "
            f"{target} ({direction}); matching_rows={matching_rows}, "
            f"direction_compatible_rows={direction_compatible_rows}, "
            f"unresolved_structures={unresolved_structures}. "
            "No fixture candidates were substituted."
            f"{failure_detail}{criteria_detail}"
        )


class LiveScientificToolNotConfigured(LiveRepurposingError):
    """A non-retrieval MA2 calculation has no configured live delegate."""


class _LookupAdapter(Protocol):
    def lookup(self, request: ChemistryLookup) -> tuple[tuple[EvidenceRef, ...], Any]: ...


@dataclass(frozen=True, slots=True)
class _DrugCentralRow:
    row_number: int
    drug_name: str
    drug_id: str
    target_accession: str | None
    target_gene: str | None
    target_name: str | None
    organism: str | None
    action_type: str | None
    moa: str | None
    activity_type: str | None
    activity_value: str | None
    activity_unit: str | None
    relation: str | None
    activity_source: str | None
    activity_source_url: str | None
    smiles: str | None
    inchikey: str | None


_INHIBIT_ACTION_TERMS = frozenset(
    {
        "antagonist",
        "blocker",
        "inhibitor",
        "negative allosteric modulator",
        "negative modulator",
        "suppressor",
    }
)
_ACTIVATE_ACTION_TERMS = frozenset(
    {
        "activator",
        "agonist",
        "positive allosteric modulator",
        "positive modulator",
        "stimulator",
    }
)
_ENDPOINTS = {
    "ic50": "IC50",
    "ki": "Ki",
    "kd": "Kd",
    "ec50": "EC50",
}
_UNITS = {
    "pm": "pM",
    "nm": "nM",
    "um": "uM",
    "µm": "uM",
    "μm": "uM",
    "mm": "mM",
}
_RELATIONS = {"=", "<", ">", "<=", ">="}
_AFFINITY_WITH_UNIT = re.compile(
    r"^\s*(<=|>=|=|<|>)?\s*"
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*"
    r"(pM|nM|uM|µM|μM|mM)\s*$",
    re.IGNORECASE,
)


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


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_evidence_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"ev_{prefix}_{digest}"


def _excerpt_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _paper_native_v6_assay_context(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = " ".join(value.casefold().replace("_", " ").split())
    if any(
        term in normalized
        for term in ("cell", "tissue", "organism", "phenotyp")
    ):
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


def _paper_native_v6_assay_engagement(
    *,
    assay_type: str,
    chembl_assay_context: object,
    assay_chembl_id: str,
    target_chembl_id: str,
    target_symbol: str,
) -> tuple[str, bool]:
    """Independently reproduce direct engagement from deposited assay fields."""

    label_context = _paper_native_v6_assay_context(assay_type)
    if label_context == "direct_single_protein_context":
        return label_context, True
    if label_context == "cellular_or_system_context":
        return label_context, False
    context = (
        chembl_assay_context
        if isinstance(chembl_assay_context, Mapping)
        else {}
    )
    record = context.get("record")
    if not isinstance(record, Mapping):
        return "assay_context_unspecified", False
    description = record.get("description")
    excerpt = (
        description.get("excerpt")
        if isinstance(description, Mapping)
        else None
    )
    normalized = " ".join(excerpt.split()) if isinstance(excerpt, str) else ""
    exact_description = bool(
        isinstance(description, Mapping)
        and excerpt == normalized
        and "<" not in normalized
        and ">" not in normalized
        and description.get("excerpt_truncated") is False
        and description.get("content_length") == len(normalized)
        and description.get("content_sha256")
        == sha256(normalized.encode("utf-8")).hexdigest()
    )
    explicit_description = bool(
        exact_description
        and target_symbol.strip()
        and re.search(
            rf"(?<![A-Za-z0-9]){re.escape(target_symbol.strip())}(?![A-Za-z0-9])",
            normalized,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"(?<!\w)(?:recombinant|purified)(?!\w)",
            normalized,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"(?<!\w)(?:inhibit\w*|kinase|enzyme)(?!\w)",
            normalized,
            flags=re.IGNORECASE,
        )
        and re.search(r"(?<!\w)atp(?!\w)", normalized, flags=re.IGNORECASE)
        and re.search(
            r"(?<!\w)substrate(?!\w)",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    direct = bool(
        context.get("schema_version") == "chembl_assay_context_v1"
        and context.get("retrieval_status") == "completed"
        and context.get("binding_status") == "exact"
        and context.get("assay_chembl_id") == assay_chembl_id
        and context.get("missing_values_imputed") is False
        and record.get("assay_chembl_id") == assay_chembl_id
        and record.get("target_chembl_id") == target_chembl_id
        and record.get("assay_type") == "B"
        and record.get("assay_organism") == "Homo sapiens"
        and record.get("relationship_type") == "D"
        and record.get("confidence_score") == 9
        and all(
            record.get(key) in (None, "")
            for key in (
                "assay_cell_type",
                "assay_tissue",
                "assay_strain",
                "cell_chembl_id",
                "tissue_chembl_id",
            )
        )
        and explicit_description
    )
    return (
        "direct_single_protein_context"
        if direct
        else "assay_context_unspecified",
        direct,
    )


def _paper_native_v6_mechanism_claim(
    *,
    target_symbol: str,
    assay_type: str,
    assay_context: str | None = None,
) -> str:
    context = assay_context or _paper_native_v6_assay_context(assay_type)
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
        f"for {target_symbol}; assay context unspecified; direct engagement "
        "not established"
    )


def _paper_native_v6_delegate_seed_source_projection(
    *,
    fixed_candidate_ids: Sequence[str],
    seeds: Sequence[CandidateSeed],
) -> dict[str, Any]:
    """Independently rebuild the producer's full fixed-pool seed binding."""

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


def _paper_native_v6_complete_activity_pagination_valid(
    bounds: Mapping[str, Any],
    fixed_candidate_ids: Sequence[str],
) -> bool:
    """Independently recompute the delegate's complete-page attestation."""

    fixed_ids = list(fixed_candidate_ids)
    counts = bounds.get("activity_counts_by_id")
    audits_by_id = bounds.get("activity_page_audits_by_id")
    if (
        not fixed_ids
        or not isinstance(counts, Mapping)
        or not isinstance(audits_by_id, Mapping)
        or list(counts) != fixed_ids
        or list(audits_by_id) != fixed_ids
        or bounds.get("activity_query_attempted_count") != len(fixed_ids)
        or bounds.get("activity_query_attempted_ids") != fixed_ids
        or bounds.get("activity_query_success_count") != len(fixed_ids)
        or bounds.get("activity_failure_count") != 0
        or bounds.get("activity_failures_by_id") != {}
        or bounds.get("activity_complete_pagination_required") is not True
        or bounds.get("activity_rows_per_candidate_limit_semantics")
        != "wire_page_size_not_result_cap"
        or bounds.get("activity_result_row_limit") is not None
        or bounds.get("activity_page_audits_by_id_sha256")
        != _canonical_sha256(dict(audits_by_id))
    ):
        return False

    expected_logical_calls = 0
    target_id = bounds.get("target_chembl_id")
    previous_global_ids: set[str] = set()
    for candidate_id in fixed_ids:
        count = counts.get(candidate_id)
        page_audits = audits_by_id.get(candidate_id)
        if not isinstance(count, Mapping) or not isinstance(page_audits, list):
            return False
        total = count.get("total_count")
        if type(total) is not int or total < 0:
            return False
        if (
            count.get("response_count") != total
            or count.get("inspected_count") != total
            or count.get("truncated") is not False
            or count.get("complete_result_set") is not True
            or count.get("page_size") != 20
            or count.get("page_count") != len(page_audits)
            or not page_audits
            or count.get("page_audits") != page_audits
            or count.get("page_audits_sha256")
            != _canonical_sha256(page_audits)
            or re.fullmatch(
                r"[0-9a-f]{64}", str(count.get("response_rows_sha256") or "")
            )
            is None
        ):
            return False
        expected_logical_calls += len(page_audits)
        observed_ids: list[str] = []
        offset = 0
        previous_numeric_id: int | None = None
        for page_index, raw_page in enumerate(page_audits):
            if not isinstance(raw_page, Mapping):
                return False
            page = dict(raw_page)
            supplied_hash = page.pop("page_audit_sha256", None)
            activity_ids = page.get("activity_ids")
            params = page.get("query_parameters")
            expected_count = min(20, max(0, total - offset))
            if (
                supplied_hash != _canonical_sha256(page)
                or page.get("page_index") != page_index
                or page.get("offset") != offset
                or page.get("limit") != 20
                or page.get("response_count") != expected_count
                or page.get("reported_total_count") != total
                or not isinstance(activity_ids, list)
                or len(activity_ids) != expected_count
                or len(activity_ids) != len(set(activity_ids))
                or page.get("activity_ids_sha256")
                != _canonical_sha256(activity_ids)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(page.get("response_rows_sha256") or ""),
                )
                is None
                or not isinstance(params, Mapping)
                or params.get("target_chembl_id") != target_id
                or params.get("parent_molecule_chembl_id") != candidate_id
                or params.get("target_organism") != "Homo sapiens"
                or params.get("assay_type") != "B"
                or params.get("standard_type") != "IC50"
                or params.get("pchembl_value__gte") != 5.0
                or params.get("order_by") != "activity_id"
                or params.get("limit") != 20
                or params.get("offset", 0) != offset
            ):
                return False
            for activity_id in activity_ids:
                if not isinstance(activity_id, str) or not activity_id.isdigit():
                    return False
                numeric_id = int(activity_id)
                if (
                    numeric_id < 1
                    or activity_id in previous_global_ids
                    or previous_numeric_id is not None
                    and numeric_id <= previous_numeric_id
                ):
                    return False
                previous_numeric_id = numeric_id
                previous_global_ids.add(activity_id)
                observed_ids.append(activity_id)
            offset += expected_count
        eligible_ids = count.get("eligible_directional_activity_ids")
        retained_ids = count.get("retained_document_activity_ids")
        if (
            offset != total
            or len(observed_ids) != total
            or not isinstance(eligible_ids, list)
            or not isinstance(retained_ids, list)
            or eligible_ids != retained_ids
            or len(eligible_ids) != len(set(eligible_ids))
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"[1-9][0-9]*", item) is None
                for item in eligible_ids
            )
            or not set(eligible_ids).issubset(observed_ids)
            or count.get("eligible_directional_count") != len(eligible_ids)
            or count.get("retained_document_rows") != len(retained_ids)
            or count.get("eligible_directional_activity_ids_sha256")
            != _canonical_sha256(eligible_ids)
            or count.get("retained_document_activity_ids_sha256")
            != _canonical_sha256(retained_ids)
            or count.get("primary_document_audit_result_cap") is not None
            or count.get("all_eligible_rows_retained_for_primary_audit")
            is not True
        ):
            return False
    return bounds.get("activity_logical_call_count") == expected_logical_calls


def _normalized_action(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    ) or None


def _direction_compatible(row: _DrugCentralRow, direction: str) -> tuple[bool, str]:
    """Return compatibility and the explicit evidence rule that justified it."""

    if direction == "modulate":
        return True, "modulation_accepts_exact_target_interaction"

    action = _normalized_action(row.action_type)
    if direction == "inhibit":
        if action in _INHIBIT_ACTION_TERMS:
            return True, "drugcentral_action_type"
        # IC50 is explicitly an inhibitory endpoint.  This does not infer a
        # numeric unit and does not generalise Ki/Kd binding to inhibition.
        if (row.activity_type or "").strip().casefold() == "ic50":
            return True, "drugcentral_ic50_endpoint"
        return False, "no_explicit_inhibitory_direction"

    if direction == "activate":
        if action in _ACTIVATE_ACTION_TERMS:
            return True, "drugcentral_action_type"
        return False, "no_explicit_activating_direction"

    return False, "unsupported_intervention_direction"


def _activity_from_drugcentral(
    row: _DrugCentralRow,
    evidence_id: str,
) -> ActivityMeasurement | None:
    endpoint = _ENDPOINTS.get((row.activity_type or "").strip().casefold())
    unit = _UNITS.get((row.activity_unit or "").strip().casefold())
    relation = (row.relation or "=").strip()
    if endpoint is None or unit is None or relation not in _RELATIONS:
        return None
    try:
        value = float(row.activity_value or "")
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return ActivityMeasurement(
        endpoint=endpoint,
        relation=relation,
        value=value,
        unit=unit,
        assay_type=(
            f"DrugCentral target interaction; source={row.activity_source or 'unspecified'}"
        ),
        source_evidence_id=evidence_id,
    )


def _activity_from_bindingdb(evidence: EvidenceRef) -> ActivityMeasurement | None:
    metadata = evidence.metadata
    endpoint_raw = _clean(metadata.get("affinity_type"))
    endpoint = _ENDPOINTS.get((endpoint_raw or "").casefold())
    candidates = [
        metadata.get((endpoint_raw or "").casefold()),
        metadata.get("affinity_value"),
    ]
    if endpoint is None:
        for key, normalized in (("ic50", "IC50"), ("ki", "Ki"), ("kd", "Kd")):
            if metadata.get(key) not in (None, ""):
                endpoint = normalized
                candidates.insert(0, metadata.get(key))
                break
    if endpoint is None:
        return None
    unit_hint = _clean(metadata.get("affinity_unit"))
    for raw in candidates:
        if raw in (None, ""):
            continue
        text = f"{raw} {unit_hint}" if unit_hint else str(raw)
        match = _AFFINITY_WITH_UNIT.match(text)
        if match is None:
            continue
        relation = match.group(1) or "="
        value = float(match.group(2))
        unit = _UNITS[match.group(3).casefold()]
        if value <= 0:
            continue
        return ActivityMeasurement(
            endpoint=endpoint,
            relation=relation,
            value=value,
            unit=unit,
            assay_type="BindingDB measured target-ligand affinity",
            source_evidence_id=evidence.evidence_id,
        )
    return None


class PublicRepurposingGateway:
    """Retrieve auditable repurposing candidates without fixture substitution.

    DrugCentral contributes declared drug-target/action records. The live
    delegate contributes exact-target, typed measured clinical ligands. PubChem
    and ChEMBL name lookup resolve structures only for declared DrugCentral
    drugs, and BindingDB evidence attaches only on exact resolved SMILES.
    """

    source_version = "drugcentral-chembl-public-repurposing-v2"
    backend_kind = ScientificBackendKind.LIVE.value
    fixture = False

    def __init__(
        self,
        *,
        drugcentral_dataset_path: str | Path,
        drugcentral_dataset_version: str | None = None,
        pubchem: _LookupAdapter | None = None,
        chembl: _LookupAdapter | None = None,
        bindingdb: _LookupAdapter | None = None,
        timeout_seconds: float = 8.0,
        delegate: CandidateToolGateway | None = None,
        repurposing_evidence_policy: str | None = None,
    ) -> None:
        self.dataset_path = Path(drugcentral_dataset_path).expanduser()
        self.dataset_version = drugcentral_dataset_version
        self.pubchem = pubchem or PubChemAdapter(timeout_seconds=timeout_seconds)
        self.chembl = chembl or ChEMBLAdapter(timeout_seconds=timeout_seconds)
        self.bindingdb = bindingdb or BindingDBAdapter(timeout_seconds=timeout_seconds)
        self.delegate = delegate
        self.repurposing_evidence_policy = repurposing_evidence_policy
        self._retrieval_cache: dict[tuple[str, str], tuple[CandidateSeed, ...]] = {}
        self.last_operation_provenance: dict[str, Any] = {}
        self._paper_native_v6_delegate_seed_audit_snapshot: (
            tuple[CandidateSeed, ...] | None
        ) = None

    def consume_paper_native_v6_delegate_seed_audit_snapshot(
        self,
    ) -> tuple[CandidateSeed, ...]:
        """Return the bounded pre-filter V6 seeds once, then erase the copy.

        This audit-only handoff lets an in-process independent validator
        recompute the full fixed-pool source projection from the actual
        delegate objects.  It persists neither HTTP bodies nor full papers.
        """

        snapshot = self._paper_native_v6_delegate_seed_audit_snapshot
        self._paper_native_v6_delegate_seed_audit_snapshot = None
        return deepcopy(snapshot) if snapshot is not None else ()

    def readiness(self, target: TargetCandidate | None = None) -> GatewayReadiness:
        dataset_ready = self.dataset_path.is_file()
        delegate_report = None
        delegate_readiness = getattr(self.delegate, "readiness", None)
        if callable(delegate_readiness):
            try:
                delegate_report = delegate_readiness(target)
            except Exception:
                delegate_report = None
        delegate_capabilities = {
            (item.tool_name, item.operation): item
            for item in getattr(delegate_report, "capabilities", ())
        }
        legacy_tools = (
            list(delegate_report.get("tools", []))
            if isinstance(delegate_report, Mapping)
            else []
        )

        def legacy_tool(*needles: str) -> Mapping[str, Any] | None:
            for item in legacy_tools:
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("tool") or "").casefold()
                operation_name = str(item.get("operation") or "").casefold()
                haystack = f"{name} {operation_name}"
                if any(needle.casefold() in haystack for needle in needles):
                    return item
            return None

        def capability(
            tool_name: str,
            operation: str,
            *,
            required: bool,
            available: bool,
            reason_code: str,
            requirements: Sequence[str] = (),
            legacy: Mapping[str, Any] | None = None,
        ) -> ToolReadiness:
            delegated = delegate_capabilities.get((tool_name, operation))
            if delegated is not None:
                return delegated
            if legacy is not None:
                available = available or str(legacy.get("status")) == "completed"
            return ToolReadiness(
                tool_name=tool_name,
                operation=operation,
                status=(
                    ReadinessStatus.READY
                    if available
                    else ReadinessStatus.NOT_CONFIGURED
                ),
                required=required,
                backend_kind=ScientificBackendKind.LIVE,
                backend_name=(
                    type(self.delegate).__name__
                    if self.delegate is not None
                    else type(self).__name__
                ),
                backend_version=(
                    str(legacy.get("version"))
                    if legacy is not None and legacy.get("version")
                    else self.source_version
                ),
                fixture=False,
                reason_code=None if available else reason_code,
                detail=(
                    None if available else str(
                        (legacy or {}).get("reason")
                        or "No real backend is configured; fixture fallback is disabled."
                    )
                ),
                input_requirements=list(requirements),
                provenance={
                    "dataset_filename": self.dataset_path.name,
                    "dataset_version": self.dataset_version,
                    "dataset_available": dataset_ready,
                    "delegate_configured": self.delegate is not None,
                    "delegate_readiness": dict(legacy or {}),
                },
            )

        delegated = self.delegate is not None
        chembl_readiness = legacy_tool("chembl")
        brics_readiness = legacy_tool("brics")
        docking_readiness = legacy_tool("vina", "docking")
        admet_readiness = legacy_tool("admet")
        de_novo_ready = delegated and all(
            item is not None and str(item.get("status")) == "completed"
            for item in (chembl_readiness, brics_readiness)
        )
        docking_ready = bool(
            delegated
            and docking_readiness is not None
            and str(docking_readiness.get("status")) == "completed"
        )
        admet_ready = bool(
            delegated
            and admet_readiness is not None
            and str(admet_readiness.get("status")) == "completed"
        )
        return GatewayReadiness(
            gateway_name=type(self).__name__,
            source_version=self.source_version,
            backend_kind=ScientificBackendKind.LIVE,
            fixture=False,
            capabilities=[
                capability(
                    "target_normalizer",
                    "normalize_target",
                    required=True,
                    available=True,
                    reason_code="TARGET_IDENTITY_UNAVAILABLE",
                    requirements=("MA1-resolved target identity",),
                    legacy=None,
                ),
                capability(
                    "repurposing_search",
                    "retrieve_candidates",
                    required=True,
                    available=dataset_ready,
                    reason_code="PUBLIC_REPURPOSING_SOURCE_NOT_CONFIGURED",
                    requirements=(
                        "DrugCentral interaction TSV or exact-target ChEMBL live API",
                    ),
                    legacy=chembl_readiness,
                ),
                capability(
                    "de_novo_generator",
                    "generate_candidates",
                    required=True,
                    available=de_novo_ready,
                    reason_code="LIVE_DE_NOVO_NOT_CONFIGURED",
                    requirements=("measured ligand seeds", "RDKit BRICS worker"),
                    legacy=(
                        brics_readiness
                        if brics_readiness is not None
                        and str(brics_readiness.get("status")) != "completed"
                        else chembl_readiness
                    ),
                ),
                capability(
                    "docking",
                    "score_binding_pose",
                    required=False,
                    available=docking_ready,
                    reason_code="LIVE_DOCKING_NOT_CONFIGURED",
                    requirements=("validated receptor PDBQT and docking box",),
                    legacy=docking_readiness,
                ),
                capability(
                    "admet_predictor",
                    "predict_six_endpoint_panel",
                    required=True,
                    available=admet_ready,
                    reason_code="LIVE_ADMET_NOT_CONFIGURED",
                    requirements=("versioned ADMET model",),
                    legacy=admet_readiness,
                ),
            ],
        )

    def normalize_target(self, target: TargetCandidate) -> TargetCandidate:
        # MA1 owns target identity. DrugCentral exact-accession/gene retrieval
        # must not depend on an unrelated ChEMBL target-resolution request.
        # The scientific delegate resolves ChEMBL lazily only when its own
        # measured-ligand/de-novo operations actually need that identifier.
        normalized = target.model_copy(deep=True)
        self.last_operation_provenance = {
            "operation": "normalize_target",
            "method": "MA1-resolved target identity pass-through",
            "target_symbol": normalized.symbol,
            "target_uniprot_resolved": bool(normalized.uniprot_id),
            "fixture": False,
        }
        return normalized

    def retrieve_repurposing(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]:
        if limit < 1:
            raise ValueError("limit must be positive")

        if self.repurposing_evidence_policy == PAPER_NATIVE_V6_REQUIRED_POLICY:
            return self._retrieve_paper_native_v6_delegate_candidates(
                target,
                direction=direction,
                limit=limit,
            )

        if self.repurposing_evidence_policy == PAPER_NATIVE_V5_REQUIRED_POLICY:
            return self._retrieve_paper_native_v5_delegate_candidates(
                target,
                direction=direction,
                limit=limit,
            )

        if self.repurposing_evidence_policy == PAIRED_V4_REQUIRED_POLICY:
            return self._retrieve_paired_v4_delegate_candidates(
                target,
                direction=direction,
                limit=limit,
            )

        # Search beyond the requested eligible quota so that an unmeasured
        # DrugCentral row (or a withdrawn ChEMBL drug) cannot crowd a better
        # evidenced candidate out before the two real sources are combined.
        # MA2 treats ``limit`` as its eligible-candidate quota.  Return the
        # already bounded reserve pool so an explicit exclusion can be
        # backfilled without a second or unbounded discovery request.
        pool_limit = min(100, max(limit * 4, 10))
        local_seeds: list[CandidateSeed] = []
        delegate_seeds: list[CandidateSeed] = []
        source_failures: list[str] = []
        criteria_rejections: list[str] = []
        local_error: RepurposingSourceUnavailable | None = None
        local_stats: dict[str, int] = {
            "matching_rows": 0,
            "direction_compatible_rows": 0,
            "unresolved_structures": 0,
        }

        try:
            local_seeds, local_stats = self._retrieve_drugcentral_candidates(
                target,
                direction=direction,
                limit=pool_limit,
            )
        except RepurposingSourceUnavailable as exc:
            local_error = exc
            source_failures.append(f"DrugCentral:{type(exc).__name__}:{exc}")

        delegate_direction_rejected = 0
        delegate_provenance: dict[str, Any] = {}
        if self.delegate is not None:
            if bool(getattr(self.delegate, "fixture", False)):
                source_failures.append(
                    "delegate:fixture_backend_rejected:live retrieval never accepts fixtures"
                )
            else:
                try:
                    raw_delegate_seeds = self.delegate.retrieve_repurposing(
                        target,
                        direction=direction,
                        limit=pool_limit,
                    )
                    payload = getattr(self.delegate, "last_operation_provenance", {})
                    if isinstance(payload, Mapping):
                        delegate_provenance = dict(payload)
                    for seed in raw_delegate_seeds:
                        prepared = self._prepare_delegate_seed(
                            seed,
                            target=target,
                            direction=direction,
                        )
                        if prepared is None:
                            delegate_direction_rejected += 1
                            continue
                        delegate_seeds.append(prepared)
                    if delegate_direction_rejected and not delegate_seeds:
                        criteria_rejections.append(
                            "delegate:all_candidates_rejected:missing exact-target "
                            "typed activity provenance or direction evidence"
                        )
                except Exception as exc:
                    payload = getattr(self.delegate, "last_operation_provenance", {})
                    if isinstance(payload, Mapping):
                        delegate_provenance = dict(payload)
                    failure = getattr(exc, "provenance", {})
                    if isinstance(failure, Mapping) and failure:
                        delegate_provenance["failure"] = dict(failure)
                    destination = (
                        criteria_rejections
                        if bool(getattr(exc, "conclusive", False))
                        else source_failures
                    )
                    destination.append(
                        f"delegate:{type(exc).__name__}:{str(exc)[:500]}"
                    )

        combined = self._merge_and_rank_candidates(
            [*local_seeds, *delegate_seeds],
            target=target,
            direction=direction,
        )[:pool_limit]

        source_search_complete = not (
            source_failures or local_stats.get("unresolved_structures", 0)
            or delegate_provenance.get("source_search_complete") is False
            or delegate_provenance.get("search_completed") is False
            or delegate_provenance.get("partial_source_result") is True
        )

        if not combined:
            # Preserve the old, useful configuration error when DrugCentral
            # is the only configured source.  With a delegate configured, a
            # combined typed error records both source outcomes instead of
            # silently falling back to a fixture.
            if local_error is not None and self.delegate is None:
                raise local_error
            empty_result = NoRealRepurposingCandidates(
                target=target.uniprot_id or target.symbol,
                direction=direction,
                matching_rows=local_stats["matching_rows"],
                direction_compatible_rows=local_stats[
                    "direction_compatible_rows"
                ],
                unresolved_structures=local_stats["unresolved_structures"],
                source_failures=source_failures,
                criteria_rejections=criteria_rejections,
                source_search_complete=source_search_complete,
            )
            empty_result.provenance["delegate_provenance"] = delegate_provenance
            self.last_operation_provenance = deepcopy(empty_result.provenance)
            raise empty_result

        cache_key = (target.uniprot_id or target.symbol, direction)
        self._retrieval_cache[cache_key] = tuple(combined)
        self.last_operation_provenance = {
            "operation": "retrieve_candidates",
            "method": "exact-target DrugCentral + measured ChEMBL evidence merge",
            "source_version": self.source_version,
            "dataset_filename": self.dataset_path.name,
            "dataset_version": self.dataset_version,
            **local_stats,
            "drugcentral_candidates": len(local_seeds),
            "delegate_candidates_accepted": len(delegate_seeds),
            "delegate_candidates_direction_or_provenance_rejected": (
                delegate_direction_rejected
            ),
            "returned_candidates": len(combined),
            "requested_limit": limit,
            "search_pool_limit": pool_limit,
            "requested_eligible_count": limit,
            "discovery_pool_limit": pool_limit,
            "discovery_pool_count": len(combined),
            "bounded_discovery": True,
            "candidate_ordering": (
                "policy_status_target_linkage_clinical_phase_primary_evidence_stable_id"
            ),
            "pchembl_used_for_candidate_ordering": False,
            "source_failures": source_failures,
            "criteria_rejections": criteria_rejections,
            "partial_source_result": not source_search_complete,
            "source_search_complete": source_search_complete,
            "search_completed": source_search_complete,
            "exhaustive_discovery": False,
            "completeness_scope": "declared_bounded_source_queries_only",
            "delegate_provenance": delegate_provenance,
            "fixture": False,
            "missing_values_imputed": False,
        }
        return tuple(combined)

    def _retrieve_paired_v4_delegate_candidates(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> tuple[CandidateSeed, ...]:
        """Preserve one attested mechanism-first delegate pool byte-for-byte.

        The exact paired-v4 policy freezes candidate identity and ordering in
        the scientific delegate before activity, warning and primary-literature
        evidence is observed.  Mixing a second discovery source here would
        silently replace that preregistered pool before MA2 owns quota
        evaluation.  Therefore this branch accepts only the delegate's complete
        surviving subsequence and independently checks its cross-copied bounds
        and envelope membership before returning it unchanged.
        """

        delegate = self.delegate
        if delegate is None:
            raise PairedV4RepurposingContractUnavailable(
                "PAIRED_V4_LIVE_DELEGATE_REQUIRED",
                "The exact paired-v4 policy requires a live scientific delegate.",
            )
        if bool(getattr(delegate, "fixture", False)):
            raise PairedV4RepurposingContractUnavailable(
                "PAIRED_V4_FIXTURE_DELEGATE_REJECTED",
                "A fixture delegate cannot supply the exact paired-v4 contract.",
            )

        try:
            raw_seeds = delegate.retrieve_repurposing(
                target,
                direction=direction,
                limit=limit,
            )
        except Exception as exc:
            payload = getattr(delegate, "last_operation_provenance", {})
            provenance = (
                {"delegate_provenance": dict(payload)}
                if isinstance(payload, Mapping)
                else {}
            )
            raise PairedV4RepurposingContractUnavailable(
                "PAIRED_V4_DELEGATE_RETRIEVAL_UNAVAILABLE",
                (
                    "The live paired-v4 delegate did not return its frozen "
                    f"candidate pool ({type(exc).__name__}: {str(exc)[:300]})."
                ),
                provenance=provenance,
            ) from exc

        seeds = tuple(raw_seeds)
        delegate_provenance = getattr(
            delegate, "last_operation_provenance", {}
        )
        if not isinstance(delegate_provenance, Mapping):
            raise PairedV4RepurposingContractUnavailable(
                "PAIRED_V4_DELEGATE_PROVENANCE_MISSING",
                "The paired-v4 delegate returned no structured provenance.",
            )
        self._validate_paired_v4_delegate_contract(
            seeds,
            delegate_provenance=delegate_provenance,
            requested_limit=limit,
        )

        candidate_ids = [self._paired_v4_seed_id(seed) for seed in seeds]
        envelope_ids = [
            candidate_id
            for candidate_id, seed in zip(candidate_ids, seeds, strict=True)
            if self._paired_v4_seed_envelope(seed, candidate_id) is not None
        ]
        cache_key = (target.uniprot_id or target.symbol, direction)
        self._retrieval_cache[cache_key] = seeds
        self.last_operation_provenance = {
            "operation": "retrieve_candidates",
            "method": "paired-v4 live delegate frozen-pool pass-through",
            "source_version": self.source_version,
            "policy": PAIRED_V4_REQUIRED_POLICY,
            "requested_limit": limit,
            "requested_eligible_count": limit,
            "returned_candidates": len(seeds),
            "gateway_return_candidate_ids": candidate_ids,
            "paired_v4_envelope_candidate_ids": envelope_ids,
            "delegate_pool_preserved_exactly": True,
            "local_candidate_discovery_executed": False,
            "candidate_merge_used": False,
            "candidate_reranking_used": False,
            "candidate_replacement_used": False,
            "fixture": False,
            "missing_values_imputed": False,
            "delegate_provenance": dict(delegate_provenance),
        }
        return seeds

    def _validate_paired_v4_delegate_contract(
        self,
        seeds: Sequence[CandidateSeed],
        *,
        delegate_provenance: Mapping[str, Any],
        requested_limit: int,
    ) -> None:
        def fail(reason_code: str, detail: str) -> None:
            raise PairedV4RepurposingContractUnavailable(
                reason_code,
                detail,
                provenance={
                    "delegate_provenance": dict(delegate_provenance),
                },
            )

        if (
            delegate_provenance.get("operation") != "retrieve_repurposing"
            or delegate_provenance.get("status") != "completed"
            or delegate_provenance.get("fixture") is not False
            or delegate_provenance.get("requested_limit") != requested_limit
            or delegate_provenance.get("requested_eligible_count")
            != requested_limit
            or delegate_provenance.get("fixed_prewarning_pool_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or delegate_provenance.get("eligible_quota_evaluation_owner")
            != "MA2"
            or delegate_provenance.get(
                "gateway_return_order_matches_fixed_prewarning_subsequence"
            )
            is not True
            or delegate_provenance.get("gateway_return_cap_applied") is not False
        ):
            fail(
                "PAIRED_V4_DELEGATE_PROVENANCE_INVALID",
                "The delegate operation attestation does not preserve MA2-owned "
                "quota evaluation over one completed live fixed-pool subsequence.",
            )
        if not seeds:
            fail(
                "PAIRED_V4_DELEGATE_POOL_EMPTY",
                "The delegate returned no candidate identities for paired-v4 evaluation.",
            )

        candidate_ids = [self._paired_v4_seed_id(seed) for seed in seeds]
        if any(candidate_id is None for candidate_id in candidate_ids):
            fail(
                "PAIRED_V4_CANDIDATE_IDENTITY_MISSING",
                "Every paired-v4 delegate seed requires one authoritative ChEMBL ID.",
            )
        stable_ids = [str(candidate_id) for candidate_id in candidate_ids]
        if len(set(stable_ids)) != len(stable_ids):
            fail(
                "PAIRED_V4_CANDIDATE_IDENTITY_DUPLICATE",
                "The paired-v4 delegate pool contains duplicate ChEMBL identities.",
            )
        if (
            delegate_provenance.get("candidate_count") != len(stable_ids)
            or delegate_provenance.get("gateway_return_candidate_ids")
            != stable_ids
        ):
            fail(
                "PAIRED_V4_DELEGATE_POOL_MISMATCH",
                "Delegate provenance candidate IDs do not exactly match returned seeds.",
            )

        bounds = delegate_provenance.get("candidate_discovery_bounds")
        if not isinstance(bounds, Mapping):
            fail(
                "PAIRED_V4_DISCOVERY_BOUNDS_MISSING",
                "The delegate did not retain structured mechanism-first discovery bounds.",
            )
        if (
            bounds.get("schema_version") != _PAIRED_V4_DISCOVERY_SCHEMA
            or bounds.get("discovery_lane") != _PAIRED_V4_DISCOVERY_LANE
            or bounds.get("target_type") != "SINGLE PROTEIN"
            or bounds.get("target_organism") != "Homo sapiens"
            or bounds.get("mechanism_action_type") != "INHIBITOR"
            or bounds.get("raw_mechanism_row_limit")
            != _PAIRED_V4_MAX_MECHANISM_ROWS
            or bounds.get("mechanism_complete_within_bound") is not True
            or bounds.get("prewarning_pool_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("prewarning_selection_policy")
            != "clinical_max_phase_desc_then_parent_molecule_chembl_id_v1"
            or bounds.get("selection_inputs")
            != ["max_phase", "parent_molecule_chembl_id"]
            or bounds.get("pchembl_used_for_prewarning_selection") is not False
            or bounds.get("primary_document_used_for_prewarning_selection")
            is not False
            or bounds.get("warning_used_for_prewarning_selection") is not False
            or bounds.get("activity_used_for_prewarning_selection") is not False
            or bounds.get("activity_candidate_query_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("no_activity_first_fallback") is not True
            or bounds.get("returned_candidate_ordering_policy")
            != _PAIRED_V4_RETURN_ORDERING
            or bounds.get("fixture") is not False
            or bounds.get("missing_values_imputed") is not False
            or bounds.get("requested_eligible_count") != requested_limit
            or bounds.get("fixed_prewarning_pool_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("no_backfill_beyond_prewarning_pool") is not True
            or bounds.get("gateway_returned_from_fixed_prewarning_pool") is not True
            or bounds.get(
                "gateway_return_order_matches_fixed_prewarning_subsequence"
            )
            is not True
            or bounds.get("gateway_return_cap_applied") is not False
            or bounds.get("eligible_quota_evaluation_owner") != "MA2"
            or bounds.get("gateway_return_candidate_count") != len(stable_ids)
            or bounds.get("downstream_candidate_identity_count")
            != len(stable_ids)
            or bounds.get("gateway_return_candidate_ids") != stable_ids
            or bounds.get("downstream_candidate_identity_ids") != stable_ids
        ):
            fail(
                "PAIRED_V4_DISCOVERY_BOUNDS_INVALID",
                "Mechanism-first bounds do not attest one unmodified live surviving pool.",
            )
        fixed_ids = bounds.get("prewarning_pool_ids")
        if (
            not isinstance(fixed_ids, list)
            or not fixed_ids
            or len(set(fixed_ids)) != len(fixed_ids)
            or len(fixed_ids) > _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("prewarning_pool_count") != len(fixed_ids)
            or bounds.get("preactivity_pool_ids") != fixed_ids
            or bounds.get("activity_query_attempted_ids") != fixed_ids
            or stable_ids
            != [
                candidate_id
                for candidate_id in fixed_ids
                if candidate_id in set(stable_ids)
            ]
        ):
            fail(
                "PAIRED_V4_FIXED_POOL_ORDER_CONTRADICTORY",
                "Returned candidate IDs are not the declared fixed-pool subsequence.",
            )

        for seed in seeds:
            seed_bounds = seed.metadata.get("candidate_discovery_bounds")
            if seed_bounds != bounds:
                fail(
                    "PAIRED_V4_SEED_BOUNDS_COPY_MISMATCH",
                    "A delegate seed does not retain the identical discovery-bounds copy.",
                )
            for evidence in seed.evidence:
                evidence_bounds = evidence.metadata.get(
                    "candidate_discovery_bounds"
                )
                if evidence_bounds != bounds:
                    fail(
                        "PAIRED_V4_EVIDENCE_BOUNDS_COPY_MISMATCH",
                        "Candidate evidence omits or contradicts the delegate discovery bounds.",
                    )

        primary_search = bounds.get("primary_abstract_search")
        promotion = (
            primary_search.get("promotion")
            if isinstance(primary_search, Mapping)
            else None
        )
        promoted_ids = (
            promotion.get("promoted_candidate_ids")
            if isinstance(promotion, Mapping)
            else None
        )
        if (
            not isinstance(primary_search, Mapping)
            or primary_search.get("policy") != _PAIRED_V4_PRIMARY_POLICY
            or primary_search.get("fixture") is not False
            or primary_search.get("no_backfill") is not True
            or not isinstance(promotion, Mapping)
            or promotion.get("status") != "verified"
            or not isinstance(promoted_ids, list)
            or not promoted_ids
            or len(set(promoted_ids)) != len(promoted_ids)
        ):
            fail(
                "PAIRED_V4_PRIMARY_ENVELOPE_CONTRACT_MISSING",
                "The delegate bounds do not declare a verified paired-v4 promotion set.",
            )

        envelope_ids: list[str] = []
        for seed, candidate_id in zip(seeds, stable_ids, strict=True):
            envelope = self._paired_v4_seed_envelope(seed, candidate_id)
            if envelope is not None:
                envelope_ids.append(candidate_id)
        if set(envelope_ids) != set(str(value) for value in promoted_ids):
            fail(
                "PAIRED_V4_ENVELOPE_MEMBERSHIP_MISMATCH",
                "Verified promotion IDs and returned envelope-bearing seeds differ.",
            )

    @staticmethod
    def _paired_v4_seed_id(seed: CandidateSeed) -> str | None:
        values = {
            str(seed.metadata.get(key) or "").strip().upper()
            for key in ("chembl_id", "drug_id")
            if str(seed.metadata.get(key) or "").strip()
        }
        if len(values) != 1:
            return None
        candidate_id = next(iter(values))
        return candidate_id if re.fullmatch(r"CHEMBL\d+", candidate_id) else None

    @staticmethod
    def _paired_v4_seed_envelope(
        seed: CandidateSeed,
        candidate_id: str,
    ) -> Mapping[str, Any] | None:
        copies = [
            holder.get(_PAIRED_V4_ENVELOPE_KEY)
            for holder in (
                seed.metadata,
                *(evidence.metadata for evidence in seed.evidence),
            )
            if holder.get(_PAIRED_V4_ENVELOPE_KEY) is not None
        ]
        if not copies:
            return None
        first = copies[0]
        if not isinstance(first, Mapping) or any(copy != first for copy in copies[1:]):
            raise PairedV4RepurposingContractUnavailable(
                "PAIRED_V4_ENVELOPE_COPY_MISMATCH",
                "Candidate and evidence paired-v4 envelope copies differ.",
            )
        if (
            first.get("schema_version") != _PAIRED_V4_ENVELOPE_SCHEMA
            or first.get("verification_status") != "verified"
            or first.get("policy") != _PAIRED_V4_PRIMARY_POLICY
            or first.get("molecule_chembl_id") != candidate_id
            or first.get("fixture") is not False
            or first.get("missing_values_imputed") is not False
        ):
            raise PairedV4RepurposingContractUnavailable(
                "PAIRED_V4_ENVELOPE_INVALID",
                "A paired-v4 envelope is malformed or bound to another candidate.",
            )
        return first

    def _retrieve_paper_native_v6_delegate_candidates(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> tuple[CandidateSeed, ...]:
        """Aggregate independently attested ChEMBL-linked primary records.

        The scientific delegate owns the fixed mechanism-first pool, complete
        activity pagination and candidate-specific ChEMBL-document/PubMed-type
        attestation.  This wrapper admits only exact candidate/activity/source
        triples and then freezes every verified member in the original pool
        order.  It never searches for replacements or copies evidence between
        candidates.
        """

        self._paper_native_v6_delegate_seed_audit_snapshot = None
        delegate = self.delegate
        if delegate is None or bool(getattr(delegate, "fixture", False)):
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_LIVE_DELEGATE_REQUIRED",
                "The v6 contract requires one non-fixture live scientific delegate.",
            )
        try:
            raw_seeds = tuple(
                delegate.retrieve_repurposing(
                    target,
                    direction=direction,
                    limit=limit,
                )
            )
        except Exception as exc:
            payload = getattr(delegate, "last_operation_provenance", {})
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_DELEGATE_RETRIEVAL_UNAVAILABLE",
                "The live v6 delegate did not return a complete fixed pool.",
                provenance={
                    "delegate_provenance": (
                        dict(payload) if isinstance(payload, Mapping) else {}
                    ),
                    "error_type_sha256": sha256(
                        type(exc).__name__.encode("utf-8")
                    ).hexdigest(),
                },
            ) from exc

        delegate_provenance = getattr(delegate, "last_operation_provenance", {})
        if not isinstance(delegate_provenance, Mapping):
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_DELEGATE_PROVENANCE_MISSING",
                "The v6 delegate returned no structured provenance.",
            )
        bounds = delegate_provenance.get("candidate_discovery_bounds")
        fixed_ids = (
            bounds.get("prewarning_pool_ids")
            if isinstance(bounds, Mapping)
            else None
        )
        pagination_valid = bool(
            isinstance(fixed_ids, list)
            and _paper_native_v6_complete_activity_pagination_valid(
                bounds,
                fixed_ids,
            )
        )
        source_inventory = (
            bounds.get("source_call_inventory")
            if isinstance(bounds, Mapping)
            else None
        )
        source_inventory_body = (
            dict(source_inventory)
            if isinstance(source_inventory, Mapping)
            else {}
        )
        source_inventory_hash = source_inventory_body.pop(
            "inventory_sha256", None
        )
        source_inventory_valid = bool(
            isinstance(source_inventory, Mapping)
            and source_inventory.get("schema_version")
            == "scientific_source_call_outcome_inventory_v1"
            and source_inventory.get("failure_count") == 0
            and source_inventory.get("all_failure_records_represented") is True
            and source_inventory.get("fixture") is False
            and source_inventory_hash
            == _canonical_sha256(source_inventory_body)
        )
        stable_ids = [self._paired_v4_seed_id(seed) for seed in raw_seeds]
        if (
            not raw_seeds
            or len(raw_seeds) > _PAIRED_V4_MAX_CANDIDATES
            or any(candidate_id is None for candidate_id in stable_ids)
            or len(set(stable_ids)) != len(stable_ids)
            or delegate_provenance.get("operation") != "retrieve_repurposing"
            or delegate_provenance.get("status") != "completed"
            or delegate_provenance.get("fixture") is not False
            or delegate_provenance.get("requested_limit") != limit
            or delegate_provenance.get("requested_eligible_count") != limit
            or delegate_provenance.get("gateway_return_cap_applied") is not False
            or delegate_provenance.get("eligible_quota_evaluation_owner") != "MA2"
            or not isinstance(bounds, Mapping)
            or bounds.get("schema_version") != _PAIRED_V4_DISCOVERY_SCHEMA
            or bounds.get("discovery_lane") != _PAIRED_V4_DISCOVERY_LANE
            or bounds.get("activity_complete_pagination_required") is not True
            or bounds.get("activity_rows_per_candidate_limit_semantics")
            != "wire_page_size_not_result_cap"
            or bounds.get("activity_result_row_limit") is not None
            or bounds.get("no_backfill_beyond_prewarning_pool") is not True
            or bounds.get("fixture") is not False
            or not pagination_valid
            or not source_inventory_valid
        ):
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_DELEGATE_PROVENANCE_INVALID",
                "The delegate did not preserve a complete live fixed-pool activity contract.",
                provenance={"delegate_provenance": dict(delegate_provenance)},
            )
        stable_id_values = [str(value) for value in stable_ids]
        if (
            not isinstance(fixed_ids, list)
            or not fixed_ids
            or len(fixed_ids) > _PAIRED_V4_MAX_CANDIDATES
            or len(fixed_ids) != len(set(fixed_ids))
            or stable_id_values
            != [value for value in fixed_ids if value in set(stable_id_values)]
        ):
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_FIXED_POOL_ORDER_INVALID",
                "Returned identities are not an exact fixed-pool subsequence.",
                provenance={"delegate_provenance": dict(delegate_provenance)},
            )
        declared_seed_projection = bounds.get(
            "delegate_seed_source_binding_projection"
        )
        recomputed_seed_projection = (
            _paper_native_v6_delegate_seed_source_projection(
                fixed_candidate_ids=fixed_ids,
                seeds=raw_seeds,
            )
        )
        if (
            declared_seed_projection != recomputed_seed_projection
            or delegate_provenance.get(
                "delegate_seed_source_binding_projection"
            )
            != declared_seed_projection
        ):
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_DELEGATE_SEED_SOURCE_BINDING_INVALID",
                (
                    "The full fixed-pool delegate seed/source projection did "
                    "not exactly bind the returned activities and primary records."
                ),
                provenance={"delegate_provenance": dict(delegate_provenance)},
            )
        self._paper_native_v6_delegate_seed_audit_snapshot = deepcopy(
            raw_seeds
        )
        seed_projection_rows = recomputed_seed_projection["rows"]

        support_by_id: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        seed_by_id = dict(zip(stable_id_values, raw_seeds, strict=True))
        wrapper_screen_by_id: dict[str, dict[str, Any]] = {}

        def record_wrapper_screen(
            *,
            candidate_id: str,
            delegate_returned: bool,
            status: str,
            reason_code: str | None,
            cardinality_exact: bool,
            primary_mapping: bool,
            exact_binding: bool,
        ) -> None:
            if (
                status not in _PAPER_NATIVE_V6_WRAPPER_SCREEN_STATUSES
                or (
                    reason_code is not None
                    and reason_code
                    not in _PAPER_NATIVE_V6_WRAPPER_SCREEN_REASONS
                )
            ):
                raise AssertionError("unbounded paper-native-v6 screen outcome")
            checks = {
                "cardinality_exact": cardinality_exact,
                "primary_mapping": primary_mapping,
                "exact_binding": exact_binding,
            }
            pool_index = fixed_ids.index(candidate_id)
            row = {
                "pool_index": pool_index,
                "fixed_pool_rank": pool_index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": delegate_returned,
                "wrapper_screen_status": status,
                "wrapper_screen_reason_code": reason_code,
                "wrapper_screen_checks": checks,
                "wrapper_screen_checks_sha256": _canonical_sha256(checks),
                "fixture": False,
            }
            row["row_sha256"] = _canonical_sha256(row)
            wrapper_screen_by_id[candidate_id] = row

        for candidate_id, seed in zip(stable_id_values, raw_seeds, strict=True):
            reason_code = "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
            cardinality_exact = (
                len(seed.activities) == 1 and len(seed.evidence) == 1
            )
            evidence = seed.evidence[0] if len(seed.evidence) == 1 else None
            primary = (
                evidence.metadata.get("primary_source")
                if evidence is not None
                and isinstance(evidence.metadata, Mapping)
                else None
            )
            primary_mapping = isinstance(primary, Mapping)
            if not cardinality_exact:
                record_wrapper_screen(
                    candidate_id=candidate_id,
                    delegate_returned=True,
                    status="rejected",
                    reason_code="SEED_CARDINALITY_NOT_EXACT",
                    cardinality_exact=False,
                    primary_mapping=primary_mapping,
                    exact_binding=False,
                )
                rejected.append(
                    {"molecule_chembl_id": candidate_id, "reason_code": reason_code}
                )
                continue
            measurement = seed.activities[0]
            assert evidence is not None
            if not primary_mapping:
                record_wrapper_screen(
                    candidate_id=candidate_id,
                    delegate_returned=True,
                    status="rejected",
                    reason_code="PRIMARY_SOURCE_NOT_MAPPING",
                    cardinality_exact=True,
                    primary_mapping=False,
                    exact_binding=False,
                )
                rejected.append(
                    {"molecule_chembl_id": candidate_id, "reason_code": reason_code}
                )
                continue
            assert isinstance(primary, Mapping)
            binding = primary.get("activity_binding")
            deposited = (
                binding.get("deposited_activity")
                if isinstance(binding, Mapping)
                else None
            )
            pubmed_type = primary.get("pubmed_type_verification")
            publication_types = primary.get("publication_types")
            excerpt = primary.get("excerpt")
            aliases = seed.metadata.get("chembl_exact_aliases")
            source_location = primary.get("excerpt_location")
            pmid = primary.get("pmid")
            assay_type = measurement.assay_type
            assay_id = str(
                binding.get("assay_chembl_id")
                if isinstance(binding, Mapping)
                else ""
            ).strip()
            assay_context, direct_engagement = (
                _paper_native_v6_assay_engagement(
                    assay_type=assay_type,
                    chembl_assay_context=evidence.metadata.get(
                        "chembl_assay_context"
                    ),
                    assay_chembl_id=assay_id,
                    target_chembl_id=str(
                        bounds.get("target_chembl_id") or ""
                    ),
                    target_symbol=target.symbol,
                )
            )
            mechanism_identity = seed.metadata.get("mechanism_identity")
            mechanism_provenance = bounds.get(
                "mechanism_provenance_by_id"
            )
            mechanism_claim = _paper_native_v6_mechanism_claim(
                target_symbol=target.symbol,
                assay_type=assay_type,
                assay_context=assay_context,
            )
            activity_counts = bounds.get("activity_counts_by_id")
            activity_count = (
                activity_counts.get(candidate_id)
                if isinstance(activity_counts, Mapping)
                else None
            )
            selection = primary.get("activity_document_selection")
            selection_rows = (
                selection.get("rows")
                if isinstance(selection, Mapping)
                else None
            )
            selection_activity_ids = (
                [
                    str(row.get("activity_id") or "")
                    for row in selection_rows
                    if isinstance(row, Mapping)
                ]
                if isinstance(selection_rows, list)
                else None
            )
            exact_binding = bool(
                primary.get("schema_version") == "candidate_primary_evidence_v2"
                and primary.get("verification_status") == "verified"
                and primary.get("is_primary_support") is True
                and primary.get("reason_code") is None
                and isinstance(binding, Mapping)
                and str(binding.get("activity_id") or "")
                == str(evidence.external_id)
                and binding.get("molecule_chembl_id") == candidate_id
                and binding.get("target_chembl_id")
                == bounds.get("target_chembl_id")
                and binding.get("endpoint") == measurement.endpoint
                and binding.get("relation") == measurement.relation
                and binding.get("value") == measurement.value
                and binding.get("unit") == measurement.unit
                and binding.get("assay_type") == assay_type
                and binding.get("assay_context") == assay_context
                and binding.get("direct_single_protein_engagement")
                is direct_engagement
                and binding.get("direction_evidence_rule")
                == "measured_ic50_endpoint"
                and evidence.metadata.get("assay_type") == assay_type
                and evidence.metadata.get("assay_context") == assay_context
                and evidence.metadata.get(
                    "direct_single_protein_engagement"
                )
                is direct_engagement
                and seed.metadata.get("activity_assay_type") == assay_type
                and seed.metadata.get("activity_assay_context")
                == assay_context
                and seed.metadata.get("direct_single_protein_engagement")
                is direct_engagement
                and seed.metadata.get("direction_requested") == direction
                and seed.metadata.get("direction_evidence_rules")
                == ["measured_ic50_endpoint"]
                and evidence.metadata.get(
                    "intervention_direction_requested"
                )
                == direction
                and evidence.metadata.get("direction_evidence_rule")
                == "measured_ic50_endpoint"
                and isinstance(mechanism_identity, Mapping)
                and evidence.metadata.get("mechanism_identity")
                == mechanism_identity
                and isinstance(mechanism_provenance, Mapping)
                and mechanism_provenance.get(candidate_id)
                == mechanism_identity
                and binding.get("mechanism_identity_sha256")
                == _canonical_sha256(dict(mechanism_identity))
                and binding.get("mechanism_claim") == mechanism_claim
                and binding.get("mechanism_claim_sha256")
                == _canonical_sha256(mechanism_claim)
                and seed.mechanism == mechanism_claim
                and isinstance(activity_count, Mapping)
                and isinstance(selection, Mapping)
                and isinstance(selection_rows, list)
                and all(
                    isinstance(row, Mapping) for row in selection_rows
                )
                and selection_activity_ids
                == activity_count.get(
                    "retained_document_activity_ids"
                )
                and selection.get("eligible_row_count")
                == activity_count.get("eligible_directional_count")
                and selection.get("inspected_row_count")
                == activity_count.get("retained_document_rows")
                and selection.get("inspected_row_count")
                == len(selection_rows)
                and selection.get("selected_activity_id")
                == str(evidence.external_id)
                and type(selection.get("selected_rank")) is int
                and 1
                <= selection.get("selected_rank")
                <= len(selection_rows)
                and selection_activity_ids[
                    selection.get("selected_rank") - 1
                ]
                == str(evidence.external_id)
                and isinstance(deposited, Mapping)
                and deposited.get("typed_exact_match") is True
                and measurement.source_evidence_id == evidence.evidence_id
                and isinstance(excerpt, str)
                and primary.get("excerpt_hash") == _excerpt_hash(excerpt)
                and isinstance(aliases, list)
                and bool(aliases)
                and isinstance(source_location, Mapping)
                and isinstance(pmid, str)
                and pmid.isdigit()
                and isinstance(pubmed_type, Mapping)
                and pubmed_type.get("verification_status") == "verified"
                and pubmed_type.get("retrieval_status") == "completed"
                and pubmed_type.get("pmid") == pmid
                and publication_types_valid_v6(publication_types)
            )
            if not exact_binding:
                record_wrapper_screen(
                    candidate_id=candidate_id,
                    delegate_returned=True,
                    status="rejected",
                    reason_code="PAPER_NATIVE_V6_EXACT_BINDING_INVALID",
                    cardinality_exact=True,
                    primary_mapping=True,
                    exact_binding=False,
                )
                rejected.append(
                    {"molecule_chembl_id": candidate_id, "reason_code": reason_code}
                )
                continue
            record_wrapper_screen(
                candidate_id=candidate_id,
                delegate_returned=True,
                status="verified",
                reason_code=None,
                cardinality_exact=True,
                primary_mapping=True,
                exact_binding=True,
            )
            primary_sha256 = _canonical_sha256(dict(primary))
            support_by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_aliases": list(aliases),
                "sentence": excerpt,
                "sentence_sha256": _excerpt_hash(excerpt),
                "document_id_sha256": _excerpt_hash(pmid),
                "source_location": json.dumps(
                    source_location,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "target_symbol": target.symbol,
                "predicate": "inhibits",
                "direction": direction,
                "endpoint": measurement.endpoint,
                "relation": measurement.relation,
                "value": measurement.value,
                "unit": measurement.unit,
                "primary_source_attestation_sha256": primary_sha256,
                "publication_type_screen_passed": True,
                "publication_types": list(publication_types),
                "verification_status": "verified",
                "is_primary_support": True,
                "is_review": False,
                "fixture": False,
            }

        for candidate_id in fixed_ids:
            if candidate_id not in wrapper_screen_by_id:
                record_wrapper_screen(
                    candidate_id=candidate_id,
                    delegate_returned=False,
                    status="not_returned",
                    reason_code="DELEGATE_NOT_RETURNED",
                    cardinality_exact=False,
                    primary_mapping=False,
                    exact_binding=False,
                )
        wrapper_screen_rows = [
            wrapper_screen_by_id[candidate_id] for candidate_id in fixed_ids
        ]
        wrapper_screen_projection = {
            "schema_version": _PAPER_NATIVE_V6_WRAPPER_SCREEN_SCHEMA,
            "fixed_pool_candidate_ids": list(fixed_ids),
            "fixed_pool_candidate_ids_sha256": _canonical_sha256(fixed_ids),
            "row_count": len(wrapper_screen_rows),
            "rows": wrapper_screen_rows,
            "rows_sha256": _canonical_sha256(wrapper_screen_rows),
            "complete": True,
            "fixture": False,
        }
        wrapper_screen_projection["attestation_sha256"] = _canonical_sha256(
            wrapper_screen_projection
        )

        ordered_support = {
            candidate_id: support_by_id[candidate_id]
            for candidate_id in fixed_ids
            if candidate_id in support_by_id
        }
        if len(ordered_support) < MINIMUM_VERIFIED_CANDIDATES:
            support_grade = "B" if ordered_support else "C"
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT",
                (
                    f"Only {len(ordered_support)} independently bound primary "
                    f"record(s) passed v6; {MINIMUM_VERIFIED_CANDIDATES} are "
                    "required."
                ),
                provenance={
                    "delegate_provenance": dict(delegate_provenance),
                    "support_candidate_count": len(ordered_support),
                    "minimum_verified_candidate_count": (
                        MINIMUM_VERIFIED_CANDIDATES
                    ),
                    "evidence_grade": support_grade,
                    "evidence_grade_semantics": (
                        "partial_exact_primary_support_not_promotion_eligible"
                        if support_grade == "B"
                        else "activity_only_or_no_exact_primary_support_not_"
                        "promotion_eligible"
                    ),
                    "promotion_eligible": False,
                    "rejection_count": len(rejected),
                    "rejections_sha256": _canonical_sha256(rejected),
                    "wrapper_atomic_screen_projection": deepcopy(
                        wrapper_screen_projection
                    ),
                },
            )
        try:
            aggregation = aggregate_candidate_primary_support_v6(
                fixed_candidate_ids=fixed_ids,
                support_by_candidate=ordered_support,
                expected_target_symbol=target.symbol,
                expected_endpoint="IC50",
                expected_direction=direction,
                requested_eligible_count=len(ordered_support),
            )
        except PaperNativeV6ContractError as exc:
            raise PaperNativeV6RepurposingContractUnavailable(
                "PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID",
                (
                    "Three or more primary support units were present, but "
                    "their aggregation contract was contradictory."
                ),
                provenance={
                    "delegate_provenance": dict(delegate_provenance),
                    "support_candidate_count": len(ordered_support),
                    "minimum_verified_candidate_count": (
                        MINIMUM_VERIFIED_CANDIDATES
                    ),
                    "evidence_grade": None,
                    "promotion_eligible": False,
                    "rejection_count": len(rejected),
                    "rejections_sha256": _canonical_sha256(rejected),
                    "wrapper_atomic_screen_projection": deepcopy(
                        wrapper_screen_projection
                    ),
                    "contract_error_type_sha256": sha256(
                        type(exc).__name__.encode("utf-8")
                    ).hexdigest(),
                    "contract_error_code_sha256": sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                },
            ) from exc

        selected_ids = list(aggregation["candidate_ids"])
        aggregated_units_by_id = {
            str(item.get("candidate_id")): item
            for item in aggregation["candidate_support_units"]
            if isinstance(item, Mapping)
        }
        rejection_by_id = {
            str(item.get("molecule_chembl_id")): str(
                item.get("reason_code") or "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
            )
            for item in rejected
            if isinstance(item, Mapping)
        }
        resolution_by_id = {
            str(item.get("molecule_chembl_id")): item
            for item in bounds.get("identity_resolution", [])
            if isinstance(item, Mapping)
            and str(item.get("molecule_chembl_id") or "")
        }
        dispositions: list[dict[str, Any]] = []
        for pool_index, candidate_id in enumerate(fixed_ids):
            seed = seed_by_id.get(candidate_id)
            source_row = seed_projection_rows[pool_index]
            screen_row = wrapper_screen_rows[pool_index]
            primary = (
                seed.evidence[0].metadata.get("primary_source")
                if seed is not None and len(seed.evidence) == 1
                else None
            )
            verified = candidate_id in aggregated_units_by_id
            resolution = resolution_by_id.get(candidate_id, {})
            delegate_reason_codes = resolution.get("reason_codes", [])
            if not isinstance(delegate_reason_codes, list) or any(
                not isinstance(item, str) for item in delegate_reason_codes
            ):
                delegate_reason_codes = []
            reason_code = None
            if not verified:
                reason_code = rejection_by_id.get(candidate_id)
                if reason_code is None:
                    reason_code = (
                        str(delegate_reason_codes[-1])
                        if delegate_reason_codes
                        else "DELEGATE_FIXED_POOL_IDENTITY_NOT_RETURNED"
                    )
            disposition = {
                "pool_index": pool_index,
                "fixed_pool_rank": pool_index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": seed is not None,
                "verification_status": (
                    "verified" if verified else "unverified"
                ),
                "reason_code": reason_code,
                "delegate_reason_codes": list(delegate_reason_codes),
                "activity_id": (
                    str(seed.evidence[0].external_id)
                    if seed is not None and len(seed.evidence) == 1
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
                "source_primary_verification_status": source_row[
                    "primary_verification_status"
                ],
                "source_primary_verification_reason_code": source_row[
                    "primary_reason_code"
                ],
                "source_primary_attestation_sha256": source_row[
                    "primary_attestation_sha256"
                ],
                "wrapper_screen_status": screen_row[
                    "wrapper_screen_status"
                ],
                "wrapper_screen_reason_code": screen_row[
                    "wrapper_screen_reason_code"
                ],
                "wrapper_screen_checks": deepcopy(
                    screen_row["wrapper_screen_checks"]
                ),
                "wrapper_screen_checks_sha256": screen_row[
                    "wrapper_screen_checks_sha256"
                ],
                "wrapper_screen_row_sha256": screen_row["row_sha256"],
                "support_unit_sha256": (
                    aggregated_units_by_id[candidate_id].get(
                        "support_unit_sha256"
                    )
                    if verified
                    else None
                ),
                "fixture": False,
            }
            if (
                disposition["delegate_returned"]
                is not source_row["delegate_returned"]
                or disposition["activity_id"] != source_row["activity_id"]
                or disposition["document_chembl_id"]
                != source_row["document_chembl_id"]
                or disposition["pmid"] != source_row["pmid"]
                or disposition["delegate_returned"]
                is not screen_row["delegate_returned"]
                or verified
                is not (screen_row["wrapper_screen_status"] == "verified")
            ):
                raise PaperNativeV6RepurposingContractUnavailable(
                    "PAPER_NATIVE_V6_DISPOSITION_SOURCE_BINDING_INVALID",
                    (
                        "A fixed-pool disposition did not match its exact "
                        "delegate seed/source projection."
                    ),
                    provenance={
                        "candidate_id_sha256": _canonical_sha256(
                            candidate_id
                        ),
                        "pool_index": pool_index,
                    },
                )
            disposition["disposition_sha256"] = _canonical_sha256(
                disposition
            )
            dispositions.append(disposition)
        search = {
            "schema_version": _PAPER_NATIVE_V6_SEARCH_SCHEMA,
            "policy": PAPER_NATIVE_V6_REQUIRED_POLICY,
            "aggregation_policy": PAPER_NATIVE_V6_AGGREGATION_POLICY,
            "status": "verified",
            "fixed_pool_candidate_ids": list(fixed_ids),
            "fixed_pool_candidate_ids_sha256": _canonical_sha256(fixed_ids),
            "verified_candidate_ids": selected_ids,
            "verified_candidate_count": len(selected_ids),
            "candidate_support_aggregation": aggregation,
            "candidate_primary_dispositions": dispositions,
            "candidate_primary_disposition_count": len(dispositions),
            "candidate_primary_dispositions_sha256": _canonical_sha256(
                dispositions
            ),
            "rejected_candidate_count": sum(
                item["verification_status"] == "unverified"
                for item in dispositions
            ),
            "rejected_candidate_projection_sha256": _canonical_sha256(
                [
                    item
                    for item in dispositions
                    if item["verification_status"] == "unverified"
                ]
            ),
            "no_reordering": True,
            "no_backfill": True,
            "fixture": False,
        }
        search["provenance_sha256"] = _canonical_sha256(search)
        common_bounds = deepcopy(dict(bounds))
        delegate_return_ids = list(
            common_bounds.get("gateway_return_candidate_ids", stable_id_values)
        )
        common_bounds["delegate_gateway_return_candidate_ids"] = delegate_return_ids
        common_bounds["delegate_gateway_return_candidate_ids_sha256"] = (
            _canonical_sha256(delegate_return_ids)
        )
        common_bounds["gateway_return_candidate_count"] = len(selected_ids)
        common_bounds["gateway_return_candidate_ids"] = selected_ids
        common_bounds["gateway_return_candidate_ids_sha256"] = (
            _canonical_sha256(selected_ids)
        )
        common_bounds["gateway_return_cap_applied"] = False
        common_bounds["eligible_quota_evaluation_owner"] = "MA2"
        common_bounds["gateway_returned_from_fixed_prewarning_pool"] = True
        common_bounds[
            "gateway_return_order_matches_fixed_prewarning_subsequence"
        ] = selected_ids == [
            candidate_id
            for candidate_id in fixed_ids
            if candidate_id in set(selected_ids)
        ]
        common_bounds["primary_abstract_search"] = deepcopy(search)
        common_bounds["paper_native_v6_verified_candidate_ids"] = selected_ids
        common_bounds["paper_native_v6_no_reordering"] = True
        common_bounds["paper_native_v6_no_backfill"] = True
        common_bounds["paper_native_v6_evidence_grade"] = "A"
        common_bounds["paper_native_v6_evidence_grade_semantics"] = (
            "minimum_three_independently_bound_exact_primary_records_"
            "promotion_eligible"
        )
        common_bounds["paper_native_v6_promotion_eligible"] = True
        common_bounds["wrapper_atomic_screen_projection"] = deepcopy(
            wrapper_screen_projection
        )

        promoted: list[CandidateSeed] = []
        unit_by_id = {
            str(item["candidate_id"]): item
            for item in aggregation["candidate_support_units"]
        }
        for fixed_rank, candidate_id in enumerate(fixed_ids, start=1):
            if candidate_id not in unit_by_id:
                continue
            seed = seed_by_id.get(candidate_id)
            if seed is None:
                raise PaperNativeV6RepurposingContractUnavailable(
                    "PAPER_NATIVE_V6_PROMOTED_SEED_MISSING",
                    "A verified v6 unit has no corresponding delegate seed.",
                )
            envelope = {
                "schema_version": _PAPER_NATIVE_V6_ENVELOPE_SCHEMA,
                "source_contract": _PAPER_NATIVE_V6_SOURCE_CONTRACT,
                "policy": PAPER_NATIVE_V6_REQUIRED_POLICY,
                "aggregation_policy": PAPER_NATIVE_V6_AGGREGATION_POLICY,
                "verification_status": "verified",
                "molecule_chembl_id": candidate_id,
                "fixed_pool_rank": fixed_rank,
                "fixed_pool_candidate_ids": list(fixed_ids),
                "group_sha256": aggregation["group_sha256"],
                "group_candidate_ids": selected_ids,
                "candidate_support_unit": deepcopy(unit_by_id[candidate_id]),
                "candidate_support_unit_sha256": unit_by_id[candidate_id][
                    "support_unit_sha256"
                ],
                "primary_source_sha256": unit_by_id[candidate_id][
                    "primary_source_attestation_sha256"
                ],
                "no_reordering": True,
                "no_backfill": True,
                "fixture": False,
                "missing_values_imputed": False,
            }
            envelope["attestation_sha256"] = _canonical_sha256(envelope)
            evidence = tuple(
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "candidate_discovery_bounds": deepcopy(common_bounds),
                            _PAPER_NATIVE_V6_ENVELOPE_KEY: deepcopy(envelope),
                        }
                    },
                    deep=True,
                )
                for item in seed.evidence
            )
            promoted.append(
                replace(
                    seed,
                    evidence=evidence,
                    metadata={
                        **seed.metadata,
                        "candidate_discovery_bounds": deepcopy(common_bounds),
                        _PAPER_NATIVE_V6_ENVELOPE_KEY: deepcopy(envelope),
                    },
                )
            )

        cache_key = (target.uniprot_id or target.symbol, direction)
        self._retrieval_cache[cache_key] = tuple(promoted)
        self.last_operation_provenance = {
            "operation": "retrieve_candidates",
            "method": "paper-native-v6 candidate-per-primary-record fixed-pool pass-through",
            "source_version": self.source_version,
            "policy": PAPER_NATIVE_V6_REQUIRED_POLICY,
            "requested_limit": limit,
            "requested_eligible_count": limit,
            "returned_candidates": len(promoted),
            "gateway_return_candidate_ids": selected_ids,
            "gateway_return_cap_applied": False,
            "eligible_quota_evaluation_owner": "MA2",
            "gateway_return_order_matches_fixed_prewarning_subsequence": True,
            "paper_native_v6_verified_candidate_ids": selected_ids,
            "support_candidate_count": len(promoted),
            "minimum_verified_candidate_count": (
                MINIMUM_VERIFIED_CANDIDATES
            ),
            "evidence_grade": "A",
            "evidence_grade_semantics": (
                "minimum_three_independently_bound_exact_primary_records_"
                "promotion_eligible"
            ),
            "promotion_eligible": True,
            "rejection_count": len(rejected),
            "rejections_sha256": _canonical_sha256(rejected),
            "wrapper_atomic_screen_projection": deepcopy(
                wrapper_screen_projection
            ),
            "delegate_pool_preserved_as_verified_subsequence": True,
            "local_candidate_discovery_executed": False,
            "candidate_merge_used": False,
            "candidate_reranking_used": False,
            "candidate_replacement_used": False,
            "candidate_discovery_bounds": deepcopy(common_bounds),
            "fixture": False,
            "fixture_fallback_used": False,
            "missing_values_imputed": False,
            "delegate_provenance": dict(delegate_provenance),
        }
        return tuple(promoted)

    def _retrieve_paper_native_v5_delegate_candidates(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> tuple[CandidateSeed, ...]:
        """Pass through one attested paper-native-v5 fixed-pool result.

        Candidate discovery, evidence attachment and fixed-pool ordering all
        belong to the live scientific delegate for this exact policy.  Local
        DrugCentral/PubChem/BindingDB discovery, merging, replacement and
        reranking are therefore forbidden in this branch.
        """

        delegate = self.delegate
        if delegate is None:
            raise PaperNativeV5RepurposingContractUnavailable(
                "PAPER_NATIVE_V5_LIVE_DELEGATE_REQUIRED",
                "The exact paper-native-v5 policy requires a live scientific delegate.",
            )
        if bool(getattr(delegate, "fixture", False)):
            raise PaperNativeV5RepurposingContractUnavailable(
                "PAPER_NATIVE_V5_FIXTURE_DELEGATE_REJECTED",
                "A fixture delegate cannot supply the paper-native-v5 contract.",
            )

        try:
            raw_seeds = delegate.retrieve_repurposing(
                target,
                direction=direction,
                limit=limit,
            )
        except Exception as exc:
            payload = getattr(delegate, "last_operation_provenance", {})
            provenance = (
                {"delegate_provenance": dict(payload)}
                if isinstance(payload, Mapping)
                else {}
            )
            raise PaperNativeV5RepurposingContractUnavailable(
                "PAPER_NATIVE_V5_DELEGATE_RETRIEVAL_UNAVAILABLE",
                (
                    "The live paper-native-v5 delegate did not return its fixed "
                    f"candidate pool ({type(exc).__name__}: {str(exc)[:300]})."
                ),
                provenance=provenance,
            ) from exc

        seeds = tuple(raw_seeds)
        delegate_provenance = getattr(delegate, "last_operation_provenance", {})
        if not isinstance(delegate_provenance, Mapping):
            raise PaperNativeV5RepurposingContractUnavailable(
                "PAPER_NATIVE_V5_DELEGATE_PROVENANCE_MISSING",
                "The paper-native-v5 delegate returned no structured provenance.",
            )
        self._validate_paper_native_v5_delegate_contract(
            seeds,
            delegate_provenance=delegate_provenance,
            requested_limit=limit,
        )

        candidate_ids = [self._paired_v4_seed_id(seed) for seed in seeds]
        verified_ids = [
            str(candidate_id)
            for candidate_id, seed in zip(candidate_ids, seeds, strict=True)
            if self._paper_native_v5_seed_envelope(
                seed,
                str(candidate_id),
            ).get("verification_status")
            == "verified"
        ]
        cache_key = (target.uniprot_id or target.symbol, direction)
        self._retrieval_cache[cache_key] = seeds
        self.last_operation_provenance = {
            "operation": "retrieve_candidates",
            "method": "paper-native-v5 live delegate fixed-pool pass-through",
            "source_version": self.source_version,
            "policy": PAPER_NATIVE_V5_REQUIRED_POLICY,
            "requested_limit": limit,
            "requested_eligible_count": limit,
            "returned_candidates": len(seeds),
            "gateway_return_candidate_ids": candidate_ids,
            "paper_native_v5_verified_candidate_ids": verified_ids,
            "delegate_pool_preserved_exactly": True,
            "local_candidate_discovery_executed": False,
            "candidate_merge_used": False,
            "candidate_reranking_used": False,
            "candidate_replacement_used": False,
            "fixture": False,
            "fixture_fallback_used": False,
            "missing_values_imputed": False,
            "delegate_provenance": dict(delegate_provenance),
        }
        return seeds

    def _validate_paper_native_v5_delegate_contract(
        self,
        seeds: Sequence[CandidateSeed],
        *,
        delegate_provenance: Mapping[str, Any],
        requested_limit: int,
    ) -> None:
        def fail(reason_code: str, detail: str) -> None:
            raise PaperNativeV5RepurposingContractUnavailable(
                reason_code,
                detail,
                provenance={"delegate_provenance": dict(delegate_provenance)},
            )

        if (
            delegate_provenance.get("operation") != "retrieve_repurposing"
            or delegate_provenance.get("status") != "completed"
            or delegate_provenance.get("fixture") is not False
            or delegate_provenance.get("requested_limit") != requested_limit
            or delegate_provenance.get("requested_eligible_count")
            != requested_limit
            or delegate_provenance.get("fixed_prewarning_pool_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or delegate_provenance.get("eligible_quota_evaluation_owner")
            != "MA2"
            or delegate_provenance.get(
                "gateway_return_order_matches_fixed_prewarning_subsequence"
            )
            is not True
            or delegate_provenance.get("gateway_return_cap_applied") is not False
        ):
            fail(
                "PAPER_NATIVE_V5_DELEGATE_PROVENANCE_INVALID",
                "The delegate attestation does not preserve one completed live "
                "fixed-pool subsequence with MA2-owned quota evaluation.",
            )
        if not seeds or len(seeds) > _PAIRED_V4_MAX_CANDIDATES:
            fail(
                "PAPER_NATIVE_V5_DELEGATE_POOL_SIZE_INVALID",
                "The delegate must return between one and twelve fixed-pool candidates.",
            )

        candidate_ids = [self._paired_v4_seed_id(seed) for seed in seeds]
        if any(candidate_id is None for candidate_id in candidate_ids):
            fail(
                "PAPER_NATIVE_V5_CANDIDATE_IDENTITY_MISSING",
                "Every paper-native-v5 seed requires one authoritative ChEMBL ID.",
            )
        stable_ids = [str(candidate_id) for candidate_id in candidate_ids]
        if len(set(stable_ids)) != len(stable_ids):
            fail(
                "PAPER_NATIVE_V5_CANDIDATE_IDENTITY_DUPLICATE",
                "The paper-native-v5 delegate pool contains duplicate identities.",
            )
        if (
            delegate_provenance.get("candidate_count") != len(stable_ids)
            or delegate_provenance.get("gateway_return_candidate_ids")
            != stable_ids
        ):
            fail(
                "PAPER_NATIVE_V5_DELEGATE_POOL_MISMATCH",
                "Delegate candidate IDs do not exactly match the returned seed tuple.",
            )

        bounds = delegate_provenance.get("candidate_discovery_bounds")
        if not isinstance(bounds, Mapping):
            fail(
                "PAPER_NATIVE_V5_DISCOVERY_BOUNDS_MISSING",
                "The delegate did not retain structured mechanism-first bounds.",
            )
        if (
            bounds.get("schema_version") != _PAIRED_V4_DISCOVERY_SCHEMA
            or bounds.get("discovery_lane") != _PAIRED_V4_DISCOVERY_LANE
            or bounds.get("target_type") != "SINGLE PROTEIN"
            or bounds.get("target_organism") != "Homo sapiens"
            or bounds.get("mechanism_action_type") != "INHIBITOR"
            or bounds.get("raw_mechanism_row_limit")
            != _PAIRED_V4_MAX_MECHANISM_ROWS
            or bounds.get("mechanism_complete_within_bound") is not True
            or bounds.get("prewarning_pool_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("fixed_prewarning_pool_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("prewarning_selection_policy")
            != "clinical_max_phase_desc_then_parent_molecule_chembl_id_v1"
            or bounds.get("selection_inputs")
            != ["max_phase", "parent_molecule_chembl_id"]
            or bounds.get("pchembl_used_for_prewarning_selection") is not False
            or bounds.get("primary_document_used_for_prewarning_selection")
            is not False
            or bounds.get("warning_used_for_prewarning_selection") is not False
            or bounds.get("activity_used_for_prewarning_selection") is not False
            or bounds.get("activity_candidate_query_limit")
            != _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("no_activity_first_fallback") is not True
            or bounds.get("returned_candidate_ordering_policy")
            != _PAIRED_V4_RETURN_ORDERING
            or bounds.get("fixture") is not False
            or bounds.get("missing_values_imputed") is not False
            or bounds.get("requested_eligible_count") != requested_limit
            or bounds.get("no_backfill_beyond_prewarning_pool") is not True
            or bounds.get("gateway_returned_from_fixed_prewarning_pool") is not True
            or bounds.get(
                "gateway_return_order_matches_fixed_prewarning_subsequence"
            )
            is not True
            or bounds.get("gateway_return_cap_applied") is not False
            or bounds.get("eligible_quota_evaluation_owner") != "MA2"
            or bounds.get("gateway_return_candidate_count") != len(stable_ids)
            or bounds.get("downstream_candidate_identity_count")
            != len(stable_ids)
            or bounds.get("gateway_return_candidate_ids") != stable_ids
            or bounds.get("downstream_candidate_identity_ids") != stable_ids
        ):
            fail(
                "PAPER_NATIVE_V5_DISCOVERY_BOUNDS_INVALID",
                "Mechanism-first bounds do not attest one unmodified live pool.",
            )
        fixed_ids = bounds.get("prewarning_pool_ids")
        if (
            not isinstance(fixed_ids, list)
            or not fixed_ids
            or len(set(fixed_ids)) != len(fixed_ids)
            or len(fixed_ids) > _PAIRED_V4_MAX_CANDIDATES
            or bounds.get("prewarning_pool_count") != len(fixed_ids)
            or bounds.get("preactivity_pool_ids") != fixed_ids
            or bounds.get("activity_query_attempted_ids") != fixed_ids
            or stable_ids
            != [candidate_id for candidate_id in fixed_ids if candidate_id in set(stable_ids)]
        ):
            fail(
                "PAPER_NATIVE_V5_FIXED_POOL_ORDER_CONTRADICTORY",
                "Returned candidates are not the declared fixed-pool subsequence.",
            )

        for seed in seeds:
            if seed.metadata.get("candidate_discovery_bounds") != bounds:
                fail(
                    "PAPER_NATIVE_V5_SEED_BOUNDS_COPY_MISMATCH",
                    "A seed does not retain the identical discovery-bounds copy.",
                )
            for evidence in seed.evidence:
                if evidence.metadata.get("candidate_discovery_bounds") != bounds:
                    fail(
                        "PAPER_NATIVE_V5_EVIDENCE_BOUNDS_COPY_MISMATCH",
                        "Candidate evidence omits or contradicts discovery bounds.",
                    )

        primary_search = bounds.get("primary_abstract_search")
        promotion = (
            primary_search.get("promotion")
            if isinstance(primary_search, Mapping)
            else None
        )
        verified_ids = (
            promotion.get("verified_member_candidate_ids")
            if isinstance(promotion, Mapping)
            else None
        )
        group_sha256 = (
            promotion.get("selected_group_sha256")
            if isinstance(promotion, Mapping)
            else None
        )
        if (
            not isinstance(primary_search, Mapping)
            or primary_search.get("schema_version")
            != _PAPER_NATIVE_V5_SEARCH_SCHEMA
            or primary_search.get("policy") != PAPER_NATIVE_V5_REQUIRED_POLICY
            or primary_search.get("no_backfill") is not True
            or primary_search.get("fixture") is not False
            or not isinstance(promotion, Mapping)
            or promotion.get("status") != "verified"
            or not isinstance(verified_ids, list)
            or len(verified_ids) < 3
            or len(set(verified_ids)) != len(verified_ids)
            or verified_ids
            != [candidate_id for candidate_id in stable_ids if candidate_id in set(verified_ids)]
            or not isinstance(group_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", group_sha256) is None
        ):
            fail(
                "PAPER_NATIVE_V5_PRIMARY_SEARCH_CONTRACT_INVALID",
                "The bounds do not declare one verified paper-native-v5 group.",
            )

        support_ids = bounds.get(
            "paper_native_v5_direct_support_candidate_ids"
        )
        support_snapshots = bounds.get("paper_native_v5_direct_support_snapshots")
        if (
            not isinstance(support_ids, list)
            or len(set(support_ids)) != len(support_ids)
            or not set(verified_ids).issubset(set(support_ids))
            or not isinstance(support_snapshots, list)
            or [
                snapshot.get("molecule_chembl_id")
                for snapshot in support_snapshots
                if isinstance(snapshot, Mapping)
            ]
            != support_ids
            or len(support_snapshots) != len(support_ids)
        ):
            fail(
                "PAPER_NATIVE_V5_DIRECT_SUPPORT_BOUNDS_INVALID",
                "Verified members lack independent ChEMBL direct-support bounds.",
            )

        envelope_verified_ids: list[str] = []
        group_contracts: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for seed, candidate_id in zip(seeds, stable_ids, strict=True):
            envelope = self._paper_native_v5_seed_envelope(seed, candidate_id)
            if envelope.get("verification_status") != "verified":
                continue
            envelope_verified_ids.append(candidate_id)
            group = envelope.get("group")
            sentence_ids = group.get("candidate_ids") if isinstance(group, Mapping) else None
            member_ids = envelope.get("group_member_candidate_ids")
            support = envelope.get("pre_pubmed_chembl_support")
            pairs = group.get("pairs") if isinstance(group, Mapping) else None
            pair_index = envelope.get("selected_pair_index")
            selected_pair = envelope.get("selected_pair")
            pair_sha256 = envelope.get("selected_pair_evidence_sha256")
            support_snapshot = next(
                (
                    snapshot
                    for snapshot in support_snapshots
                    if isinstance(snapshot, Mapping)
                    and snapshot.get("molecule_chembl_id") == candidate_id
                ),
                None,
            )
            if (
                not isinstance(group, Mapping)
                or group.get("group_sha256") != group_sha256
                or envelope.get("selected_group_sha256") != group_sha256
                or not isinstance(sentence_ids, list)
                or len(sentence_ids) < 3
                or len(set(sentence_ids)) != len(sentence_ids)
                or set(sentence_ids) != set(verified_ids)
                or not isinstance(member_ids, list)
                or member_ids != sentence_ids
                or set(member_ids) != set(verified_ids)
                or candidate_id not in sentence_ids
                or not isinstance(support, Mapping)
                or support != support_snapshot
                or not self._paper_native_v5_activity_matches_support(seed, support)
                or isinstance(pair_index, bool)
                or not isinstance(pair_index, int)
                or not isinstance(pairs, list)
                or not 0 <= pair_index < len(pairs)
                or not isinstance(selected_pair, Mapping)
                or pairs[pair_index] != selected_pair
                or not isinstance(pair_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", pair_sha256) is None
                or selected_pair.get("evidence_sha256") != pair_sha256
            ):
                fail(
                    "PAPER_NATIVE_V5_VERIFIED_ENVELOPE_INVALID",
                    "A verified envelope does not bind the complete group and support.",
                )
            group_contracts.append(
                (group_sha256, tuple(sentence_ids), tuple(member_ids))
            )
        if envelope_verified_ids != verified_ids:
            fail(
                "PAPER_NATIVE_V5_ENVELOPE_MEMBERSHIP_MISMATCH",
                "Promotion IDs and verified envelope-bearing seeds differ.",
            )
        if len(set(group_contracts)) != 1:
            fail(
                "PAPER_NATIVE_V5_GROUP_COPY_MISMATCH",
                "Verified candidates do not carry one identical group contract.",
            )

    @staticmethod
    def _paper_native_v5_activity_matches_support(
        seed: CandidateSeed,
        support: Mapping[str, Any],
    ) -> bool:
        """Bind ranking activity to the ChEMBL snapshot frozen before PubMed."""

        if len(seed.activities) != 1:
            return False
        measurement = seed.activities[0]
        typed = support.get("typed_measurement_tuple")
        if not isinstance(typed, Mapping):
            return False
        projected = {
            "endpoint": measurement.endpoint,
            "relation": measurement.relation,
            "value": measurement.value,
            "unit": measurement.unit,
            "assay_type": measurement.assay_type,
            "source_evidence_id": measurement.source_evidence_id,
        }
        if dict(typed) != projected:
            return False
        evidence_ids = {item.evidence_id for item in seed.evidence}
        return bool(
            measurement.source_evidence_id
            and measurement.source_evidence_id in evidence_ids
            and support.get("source_evidence_id")
            == measurement.source_evidence_id
        )

    @staticmethod
    def _paper_native_v5_seed_envelope(
        seed: CandidateSeed,
        candidate_id: str,
    ) -> Mapping[str, Any]:
        holders = (seed.metadata, *(evidence.metadata for evidence in seed.evidence))
        copies = [holder.get(_PAPER_NATIVE_V5_ENVELOPE_KEY) for holder in holders]
        if not copies or any(copy is None for copy in copies):
            raise PaperNativeV5RepurposingContractUnavailable(
                "PAPER_NATIVE_V5_ENVELOPE_COPY_MISSING",
                "Every seed and evidence record must carry the v5 envelope.",
            )
        first = copies[0]
        if not isinstance(first, Mapping) or any(copy != first for copy in copies[1:]):
            raise PaperNativeV5RepurposingContractUnavailable(
                "PAPER_NATIVE_V5_ENVELOPE_COPY_MISMATCH",
                "Candidate and evidence paper-native-v5 envelope copies differ.",
            )
        if (
            first.get("schema_version") != _PAPER_NATIVE_V5_ENVELOPE_SCHEMA
            or first.get("source_contract") != _PAPER_NATIVE_V5_SOURCE_CONTRACT
            or first.get("molecule_chembl_id") != candidate_id
            or first.get("verification_status") not in {"verified", "not_verified"}
            or first.get("fixture") is not False
            or first.get("missing_values_imputed") is not False
            or first.get("cross_source_numeric_equality_required") is not False
            or first.get("pubmed_value_used_for_activity_selection") is not False
            or first.get("pubmed_value_used_for_candidate_rank") is not False
            or first.get("pubmed_value_used_for_candidate_ordering") is not False
        ):
            raise PaperNativeV5RepurposingContractUnavailable(
                "PAPER_NATIVE_V5_ENVELOPE_INVALID",
                "A v5 envelope is malformed or bound to another candidate.",
            )
        return first

    def _retrieve_drugcentral_candidates(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> tuple[list[CandidateSeed], dict[str, int]]:
        rows = self._read_exact_target_rows(target)
        compatible: list[tuple[_DrugCentralRow, str]] = []
        for row in rows:
            allowed, rule = _direction_compatible(row, direction)
            if allowed:
                compatible.append((row, rule))

        binding_records = self._bindingdb_records(target, max(limit * 4, 10))
        binding_by_smiles: dict[str, list[EvidenceRef]] = {}
        for record in binding_records:
            smiles = _clean(record.metadata.get("smiles"))
            if smiles:
                binding_by_smiles.setdefault(smiles, []).append(record)

        grouped: dict[str, list[tuple[_DrugCentralRow, str]]] = {}
        for row, rule in compatible:
            grouped.setdefault(row.drug_id.casefold(), []).append((row, rule))

        seeds: list[CandidateSeed] = []
        unresolved_structures = 0
        seen_identity: set[str] = set()
        for group in grouped.values():
            row = group[0][0]
            structure = self._resolve_structure(row)
            if structure is None:
                unresolved_structures += 1
                continue
            smiles, inchikey, identity_evidence, structure_source = structure
            identity_key = (inchikey or smiles).casefold()
            if identity_key in seen_identity:
                continue
            seen_identity.add(identity_key)

            interaction_evidence: list[EvidenceRef] = []
            activities: list[ActivityMeasurement] = []
            direction_rules: set[str] = set()
            for interaction_row, rule in group:
                evidence = self._drugcentral_evidence(interaction_row, rule)
                interaction_evidence.append(evidence)
                direction_rules.add(rule)
                activity = _activity_from_drugcentral(interaction_row, evidence.evidence_id)
                if activity is not None:
                    activities.append(activity)

            matched_binding = binding_by_smiles.get(smiles, [])
            for evidence in matched_binding:
                interaction_evidence.append(
                    evidence.model_copy(update={"supports": "support"}, deep=True)
                )
                activity = _activity_from_bindingdb(evidence)
                if activity is not None:
                    activities.append(activity)

            activities = self._dedupe_activities(activities)
            action = row.action_type or row.activity_type or "target interaction"
            mechanism = f"DrugCentral-reported {action} of {target.symbol}"
            evidence = self._dedupe_evidence(
                [*interaction_evidence, *identity_evidence]
            )
            seeds.append(
                self._annotate_source_seed(
                    CandidateSeed(
                    name=row.drug_name,
                    smiles=smiles,
                    mechanism=mechanism,
                    activities=tuple(activities),
                    evidence=tuple(evidence),
                    metadata={
                        "fixture": False,
                        "retrieval_backend": "DrugCentralLocalTSV",
                        "provenance_class": "external_local_dataset",
                        "drug_id": row.drug_id,
                        "identifiers": [row.drug_id],
                        "aliases": [row.drug_name],
                        "drug_classes": [],
                        "target_accession": row.target_accession,
                        "target_gene": row.target_gene,
                        "direction_requested": direction,
                        "direction_evidence_rules": sorted(direction_rules),
                        "structure_source": structure_source,
                        "structure_selection": self._structure_selection_trace(
                            smiles, inchikey, identity_evidence, structure_source,
                        ),
                        "resolved_inchikey": inchikey,
                        "dataset_filename": self.dataset_path.name,
                        "dataset_version": self.dataset_version,
                        "raw_activity_records": [
                            self._raw_activity_payload(item[0]) for item in group
                        ],
                        "activity_units_inferred": False,
                        "bindingdb_exact_smiles_match_count": len(matched_binding),
                    },
                    ),
                    target=target,
                    direction=direction,
                    backend="DrugCentralLocalTSV",
                )
            )
            if len(seeds) >= limit:
                break

        return seeds, {
            "matching_rows": len(rows),
            "direction_compatible_rows": len(compatible),
            "unresolved_structures": unresolved_structures,
        }

    def _prepare_delegate_seed(
        self,
        seed: CandidateSeed,
        *,
        target: TargetCandidate,
        direction: str,
    ) -> CandidateSeed | None:
        """Accept only real, exact-target, direction-supported measurements."""

        if bool(seed.metadata.get("fixture")) or not seed.smiles.strip():
            return None
        linked_count = self._target_linked_activity_count(seed, target)
        direction_rules = self._direction_rules(seed, target, direction)
        if linked_count < 1 or not direction_rules:
            return None
        return self._annotate_source_seed(
            seed,
            target=target,
            direction=direction,
            backend=self._backend_label(seed),
            direction_rules=direction_rules,
            target_linked_activity_count=linked_count,
        )

    def _annotate_source_seed(
        self,
        seed: CandidateSeed,
        *,
        target: TargetCandidate,
        direction: str,
        backend: str,
        direction_rules: Sequence[str] | None = None,
        target_linked_activity_count: int | None = None,
    ) -> CandidateSeed:
        metadata = dict(seed.metadata)
        existing_rules = metadata.get("direction_evidence_rules") or []
        if isinstance(existing_rules, str):
            existing_rules = [existing_rules]
        rules = set(str(item) for item in existing_rules)
        if direction_rules is not None:
            rules.update(direction_rules)
        linked_count = (
            self._target_linked_activity_count(seed, target)
            if target_linked_activity_count is None
            else target_linked_activity_count
        )
        metadata.update(
            {
                "fixture": False,
                "direction_requested": direction,
                "direction_evidence_rules": sorted(rules),
                "target_linked_activity_count": linked_count,
                "retrieval_backends": sorted(
                    {
                        backend,
                        *(
                            str(item)
                            for item in (metadata.get("retrieval_backends") or [])
                        ),
                    }
                ),
            }
        )
        metadata.setdefault("retrieval_backend", backend)
        return CandidateSeed(
            name=seed.name,
            smiles=seed.smiles,
            mechanism=seed.mechanism,
            activities=tuple(item.model_copy(deep=True) for item in seed.activities),
            evidence=tuple(item.model_copy(deep=True) for item in seed.evidence),
            metadata=metadata,
        )

    def _merge_and_rank_candidates(
        self,
        seeds: Sequence[CandidateSeed],
        *,
        target: TargetCandidate,
        direction: str,
    ) -> list[CandidateSeed]:
        grouped: dict[str, list[CandidateSeed]] = {}
        for seed in seeds:
            grouped.setdefault(self._chemical_identity(seed), []).append(seed)

        merged = [
            self._merge_identity_group(
                group,
                identity=identity,
                target=target,
                direction=direction,
            )
            for identity, group in grouped.items()
        ]
        return sorted(merged, key=self._candidate_priority)

    def _merge_identity_group(
        self,
        seeds: Sequence[CandidateSeed],
        *,
        identity: str,
        target: TargetCandidate,
        direction: str,
    ) -> CandidateSeed:
        ordered = sorted(seeds, key=self._candidate_priority)
        primary = ordered[0]
        if len(ordered) == 1:
            return primary

        metadata = dict(primary.metadata)
        evidence = self._dedupe_evidence(
            [item for seed in ordered for item in seed.evidence]
        )
        activities = self._dedupe_activities(
            [item for seed in ordered for item in seed.activities]
        )
        backends = sorted({self._backend_label(seed) for seed in ordered})
        aliases = sorted(
            {
                seed.name
                for seed in ordered
            }
            | {
                str(alias)
                for seed in ordered
                for alias in (seed.metadata.get("aliases") or [])
            },
            key=str.casefold,
        )
        identifiers = sorted(
            {
                str(identifier)
                for seed in ordered
                for identifier in (
                    *(seed.metadata.get("identifiers") or []),
                    seed.metadata.get("drug_id"),
                    seed.metadata.get("chembl_id"),
                )
                if identifier not in (None, "")
            },
            key=str.casefold,
        )
        warnings: list[dict[str, Any]] = []
        seen_warnings: set[str] = set()
        for seed in ordered:
            for warning in (seed.metadata.get("drug_warnings") or []):
                if not isinstance(warning, Mapping):
                    continue
                copied = dict(warning)
                signature = json.dumps(copied, ensure_ascii=False, sort_keys=True, default=str)
                if signature in seen_warnings:
                    continue
                seen_warnings.add(signature)
                warnings.append(copied)

        statuses = [
            str(seed.metadata.get("regulatory_status") or "").strip()
            for seed in ordered
        ]
        if any(
            self._is_prohibited_status(status)
            for status in statuses
        ):
            metadata["regulatory_status"] = "withdrawn"

        metadata.update(
            {
                "fixture": False,
                "retrieval_backend": "MergedPublicLiveEvidence",
                "retrieval_backends": backends,
                "merged_source_count": len(backends),
                "deduplication_identity": identity,
                "aliases": aliases,
                "identifiers": identifiers,
                "drug_warnings": warnings,
                "direction_requested": direction,
                "direction_evidence_rules": sorted(
                    {
                        str(rule)
                        for seed in ordered
                        for rule in (seed.metadata.get("direction_evidence_rules") or [])
                    }
                ),
                "target_linked_activity_count": self._target_linked_activity_count(
                    CandidateSeed(
                        name=primary.name,
                        smiles=primary.smiles,
                        mechanism=primary.mechanism,
                        activities=tuple(activities),
                        evidence=tuple(evidence),
                        metadata=metadata,
                    ),
                    target,
                ),
                "source_metadata": [
                    {
                        "backend": self._backend_label(seed),
                        "candidate_name": seed.name,
                        "metadata": dict(seed.metadata),
                    }
                    for seed in ordered
                ],
            }
        )
        raw_records = [
            dict(record)
            for seed in ordered
            for record in (seed.metadata.get("raw_activity_records") or [])
            if isinstance(record, Mapping)
        ]
        if raw_records:
            metadata["raw_activity_records"] = raw_records

        return CandidateSeed(
            name=primary.name,
            smiles=primary.smiles,
            mechanism=primary.mechanism,
            activities=tuple(activities),
            evidence=tuple(evidence),
            metadata=metadata,
        )

    @staticmethod
    def _backend_label(seed: CandidateSeed) -> str:
        explicit = seed.metadata.get("retrieval_backend")
        if explicit:
            return str(explicit)
        if seed.metadata.get("chembl_id") or seed.metadata.get("measured_target"):
            return "ChEMBLLiveAPI"
        return "LiveScientificDelegate"

    @staticmethod
    def _chemical_identity(seed: CandidateSeed) -> str:
        for key in (
            "standard_inchi_key",
            "resolved_inchikey",
            "inchikey",
            "inchi_key",
        ):
            value = str(seed.metadata.get(key) or "").strip()
            if value:
                return f"inchikey:{value.casefold()}"
        return f"smiles:{seed.smiles.strip()}"

    def _target_linked_activity_count(
        self,
        seed: CandidateSeed,
        target: TargetCandidate,
    ) -> int:
        return len(self._target_linked_activities(seed, target))

    @staticmethod
    def _target_linked_activities(
        seed: CandidateSeed,
        target: TargetCandidate,
    ) -> list[ActivityMeasurement]:
        evidence_by_id = {item.evidence_id: item for item in seed.evidence}
        measured_target = str(seed.metadata.get("measured_target") or "").strip()
        linked_activities: list[ActivityMeasurement] = []
        for activity in seed.activities:
            if not activity.source_evidence_id:
                continue
            evidence = evidence_by_id.get(activity.source_evidence_id)
            if evidence is None:
                continue
            metadata = evidence.metadata
            accession = str(metadata.get("target_accession") or "").strip()
            gene = str(metadata.get("target_gene") or "").strip()
            chembl_target = str(metadata.get("target_chembl_id") or "").strip()
            if accession:
                linked = bool(
                    target.uniprot_id
                    and accession.casefold() == target.uniprot_id.casefold()
                )
            elif gene:
                linked = gene.casefold() == target.symbol.casefold()
            else:
                # LiveScientificGateway resolves ChEMBL by the supplied exact
                # UniProt accession and records the same normalized target on
                # both the seed and its activity evidence.
                linked = bool(
                    chembl_target
                    and measured_target
                    and chembl_target == measured_target
                )
            if linked:
                linked_activities.append(activity)
        return linked_activities

    def _direction_rules(
        self,
        seed: CandidateSeed,
        target: TargetCandidate,
        direction: str,
    ) -> tuple[str, ...]:
        linked_count = self._target_linked_activity_count(seed, target)
        if linked_count < 1:
            return ()
        if direction == "modulate":
            return ("measured_exact_target_activity",)

        explicit_rules: set[str] = set()
        for evidence in seed.evidence:
            metadata = evidence.metadata
            for raw in (
                metadata.get("action_type"),
                metadata.get("mechanism_of_action"),
                metadata.get("direction_evidence_rule"),
            ):
                action = _normalized_action(_clean(raw))
                if direction == "inhibit" and action in _INHIBIT_ACTION_TERMS:
                    explicit_rules.add("explicit_inhibitory_action")
                if direction == "activate" and action in _ACTIVATE_ACTION_TERMS:
                    explicit_rules.add("explicit_activating_action")

        linked_activities = self._target_linked_activities(seed, target)
        if direction == "inhibit" and any(
            activity.endpoint == "IC50" for activity in linked_activities
        ):
            explicit_rules.add("measured_ic50_endpoint")
        # EC50 alone reports potency of an effect, not whether that effect is
        # agonism. Activation therefore requires an explicit action record.
        return tuple(sorted(explicit_rules))

    def _candidate_priority(self, seed: CandidateSeed) -> tuple[Any, ...]:
        status = str(seed.metadata.get("regulatory_status") or "").casefold()
        prohibited = self._is_prohibited_status(status)
        linked_count = int(seed.metadata.get("target_linked_activity_count") or 0)
        max_phase = int(seed.metadata.get("max_phase") or 0)
        primary_verified = any(
            isinstance(item.metadata.get("primary_source"), Mapping)
            and item.metadata["primary_source"].get("verification_status")
            == "verified"
            and item.metadata["primary_source"].get("is_primary_support") is True
            for item in seed.evidence
        )
        stable_id = str(
            seed.metadata.get("chembl_id")
            or seed.metadata.get("drug_id")
            or seed.metadata.get("standard_inchi_key")
            or ""
        ).casefold()
        return (
            prohibited,
            -linked_count,
            -len(seed.activities),
            -max_phase,
            not primary_verified,
            stable_id,
            seed.name.casefold(),
        )

    @staticmethod
    def _is_prohibited_status(value: str) -> bool:
        normalized = " ".join(
            re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
        )
        if not normalized or normalized.startswith("not withdraw"):
            return False
        return normalized in {
            "banned",
            "prohibited",
            "recalled",
            "withdrawn",
            "withdrawn from market",
        }

    def generate_de_novo(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]:
        delegate = self._require_delegate("de novo generation")
        result = delegate.generate_de_novo(
            target, direction=direction, limit=limit
        )
        self._capture_delegate_provenance(delegate, "generate_candidates")
        return result

    def dock(self, *, smiles: str, target: TargetCandidate) -> float:
        delegate = self._require_delegate("docking")
        result = delegate.dock(smiles=smiles, target=target)
        self._capture_delegate_provenance(delegate, "score_binding_pose")
        return result

    def predict_admet(self, *, smiles: str) -> Mapping[str, float]:
        delegate = self._require_delegate("ADMET prediction")
        result = delegate.predict_admet(smiles=smiles)
        self._capture_delegate_provenance(delegate, "predict_six_endpoint_panel")
        return result

    def _capture_delegate_provenance(
        self,
        delegate: CandidateToolGateway,
        operation: str,
    ) -> None:
        payload = getattr(delegate, "last_operation_provenance", {})
        self.last_operation_provenance = {
            **(dict(payload) if isinstance(payload, Mapping) else {}),
            "operation": operation,
            "wrapper": type(self).__name__,
            "wrapper_source_version": self.source_version,
            "fixture": False,
            "missing_values_imputed": False,
        }

    def refine(
        self,
        seed: CandidateSeed,
        *,
        reasons: Sequence[str],
        iteration: int,
    ) -> CandidateSeed:
        return self._require_delegate("candidate refinement").refine(
            seed, reasons=reasons, iteration=iteration
        )

    def restore_activity(
        self,
        *,
        candidate_name: str,
        target: TargetCandidate,
    ) -> Sequence[ActivityMeasurement]:
        for (target_key, _direction), seeds in self._retrieval_cache.items():
            if target_key != (target.uniprot_id or target.symbol):
                continue
            for seed in seeds:
                if seed.name.casefold() == candidate_name.casefold():
                    return seed.activities
        return self._require_delegate("activity restoration").restore_activity(
            candidate_name=candidate_name,
            target=target,
        )

    def restore_structure(
        self,
        *,
        candidate_name: str,
        route: CandidateRoute,
        target: TargetCandidate,
        direction: str,
    ) -> CandidateSeed:
        if route is CandidateRoute.REPURPOSING:
            cache_key = (target.uniprot_id or target.symbol, direction)
            for seed in self._retrieval_cache.get(cache_key, ()):
                if seed.name.casefold() == candidate_name.casefold():
                    return seed
        return self._require_delegate("structure restoration").restore_structure(
            candidate_name=candidate_name,
            route=route,
            target=target,
            direction=direction,
        )

    def _require_delegate(self, operation: str) -> CandidateToolGateway:
        if self.delegate is None:
            raise LiveScientificToolNotConfigured(
                f"No live delegate is configured for {operation}; fixture fallback is disabled"
            )
        return self.delegate

    def _read_exact_target_rows(self, target: TargetCandidate) -> list[_DrugCentralRow]:
        if not self.dataset_path.is_file():
            raise RepurposingSourceUnavailable(
                f"DrugCentral dataset is unavailable: {self.dataset_path}"
            )
        accession_query = _clean(target.uniprot_id)
        gene_query = _clean(target.symbol)
        if accession_query is None and gene_query is None:
            raise NoRealRepurposingCandidates(
                target="unidentified target",
                direction="unknown",
                matching_rows=0,
                direction_compatible_rows=0,
                unresolved_structures=0,
            )

        rows: list[_DrugCentralRow] = []
        try:
            handle = self.dataset_path.open("r", encoding="utf-8-sig", newline="")
        except OSError as exc:
            raise RepurposingSourceUnavailable(
                f"DrugCentral dataset cannot be opened: {self.dataset_path}: {exc}"
            ) from exc
        with handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise RepurposingSourceUnavailable("DrugCentral TSV has no header")
            for row_number, raw in enumerate(reader, start=2):
                row = _normalized_mapping(raw)
                accession = _clean(_first(row, "accession", "uniprot", "uniprot_id"))
                gene = _clean(_first(row, "gene", "gene_symbol"))
                if accession_query is not None:
                    matched = accession is not None and accession.casefold() == accession_query.casefold()
                else:
                    matched = gene is not None and gene.casefold() == gene_query.casefold()
                if not matched:
                    continue
                organism = _clean(_first(row, "organism"))
                if organism and organism.casefold() not in {"homo sapiens", "human"}:
                    continue
                drug_name = _clean(_first(row, "drug_name", "name", "inn", "preferred_name"))
                drug_id = _clean(_first(row, "struct_id", "drugcentral_id", "drug_id", "id"))
                if drug_name is None or drug_id is None:
                    continue
                rows.append(
                    _DrugCentralRow(
                        row_number=row_number,
                        drug_name=drug_name,
                        drug_id=drug_id,
                        target_accession=accession,
                        target_gene=gene,
                        target_name=_clean(_first(row, "target_name")),
                        organism=organism,
                        action_type=_clean(_first(row, "action_type", "action")),
                        moa=_clean(_first(row, "moa", "mechanism_of_action")),
                        activity_type=_clean(_first(row, "act_type", "activity_type")),
                        activity_value=_clean(_first(row, "act_value", "activity_value")),
                        activity_unit=_clean(_first(row, "act_unit", "activity_unit")),
                        relation=_clean(_first(row, "relation")),
                        activity_source=_clean(_first(row, "act_source", "activity_source")),
                        activity_source_url=_clean(_first(row, "act_source_url")),
                        smiles=_clean(_first(row, "smiles", "canonical_smiles")),
                        inchikey=_clean(_first(row, "inchikey", "inchi_key")),
                    )
                )
        return rows

    def _resolve_structure(
        self,
        row: _DrugCentralRow,
    ) -> tuple[str, str | None, list[EvidenceRef], str] | None:
        if row.smiles:
            return row.smiles, row.inchikey, [], "DrugCentralLocalTSV"
        request = ChemistryLookup(candidate_name=row.drug_name, limit_per_source=3)
        for source, adapter in (("PubChem", self.pubchem), ("ChEMBL", self.chembl)):
            try:
                records, _summary = adapter.lookup(request)
            except Exception:
                continue
            for record in records:
                # PubChem's PUG name namespace resolves a supplied name or
                # synonym.  ChEMBL's endpoint is full-text, so require an exact
                # preferred-name match before accepting its structure.
                if source == "ChEMBL" and record.title.casefold() != row.drug_name.casefold():
                    continue
                # PubChem canonical/connectivity SMILES can omit known stereo.
                # Select the reported isomeric form, never enumerate or infer it.
                isomeric = (_clean(record.metadata.get("isomeric_smiles"))
                            if source == "PubChem" else None)
                smiles = isomeric or _clean(record.metadata.get("canonical_smiles"))
                if smiles is None:
                    continue
                inchikey = _clean(
                    record.metadata.get("inchikey")
                    or record.metadata.get("standard_inchi_key")
                )
                if isomeric is not None or (inchikey and re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", inchikey.upper())):
                    # No downgrade to a connectivity-only structure after a bad
                    # isomeric record. Require the actual computation parent to
                    # retain the full source identity when the source supplies it.
                    from drugpilot.tools.chemistry import ChemistryValidator
                    # Canonical-only records may also report a stereo-specific
                    # key. A real full-key mismatch is not an unknown identity.
                    report = ChemistryValidator().inspect(smiles)
                    if (not report.valid or report.backend != "rdkit"
                            or not report.inchikey):
                        if inchikey and re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", inchikey.upper()):
                            # An unreadable reported structure does not erase
                            # its declared full identity. Do not replace this
                            # unresolved binding with a later keyless graph.
                            return None
                        continue
                    if inchikey and report.inchikey != inchikey.upper():
                        # An observed full-identity conflict is not a missing
                        # structure. A later keyless/connectivity-only record
                        # must not erase it, even from another named source.
                        return None
                return smiles, inchikey, [record.model_copy(deep=True)], source
        return None

    @staticmethod
    def _structure_selection_trace(
        smiles: str, inchikey: str | None, evidence: Sequence[EvidenceRef], source: str,
    ) -> dict[str, Any]:
        """Link the computation input to an unchanged reported structure field."""
        selected = next((item for item in evidence if item.source == source and
            smiles in {_clean(item.metadata.get("isomeric_smiles")),
                       _clean(item.metadata.get("canonical_smiles"))}), None)
        field = ("isomeric_smiles" if selected is not None and
                 smiles == _clean(selected.metadata.get("isomeric_smiles")) and source == "PubChem"
                 else "canonical_smiles" if selected is not None else "local_dataset_smiles")
        return {
            "policy": "source_reported_isomeric_preferred_v1", "source": source,
            "evidence_id": selected.evidence_id if selected is not None else None,
            "external_id": selected.external_id if selected is not None else None,
            "metadata_field": field,
            "source_property_field": (selected.metadata.get(field + "_source_field")
                                      if selected is not None else None),
            "selected_smiles_sha256": sha256(smiles.encode("utf-8")).hexdigest(),
            "reported_inchikey": inchikey,
            "isomeric_input_available": bool(selected is not None and _clean(selected.metadata.get("isomeric_smiles"))),
            "stereochemistry_invented": False,
            "identity_validation": ("rdkit_computation_parent_full_inchikey_match" if field == "isomeric_smiles" and inchikey
                                    else "no_new_identity_claim; downstream_chemistry_and_source_checks_required"),
        }

    def _bindingdb_records(
        self,
        target: TargetCandidate,
        limit: int,
    ) -> tuple[EvidenceRef, ...]:
        if not target.uniprot_id:
            return ()
        try:
            records, _summary = self.bindingdb.lookup(
                ChemistryLookup(
                    uniprot_id=target.uniprot_id,
                    limit_per_source=min(10, max(1, limit)),
                )
            )
        except Exception:
            return ()
        return tuple(record.model_copy(deep=True) for record in records)

    def _drugcentral_evidence(
        self,
        row: _DrugCentralRow,
        direction_rule: str,
    ) -> EvidenceRef:
        title = (
            f"DrugCentral target interaction: {row.drug_name} - "
            f"{row.target_gene or row.target_accession or row.target_name}"
        )
        return EvidenceRef(
            evidence_id=_stable_evidence_id(
                "drugcentral_candidate",
                self.dataset_path.name,
                self.dataset_version or "unversioned",
                row.row_number,
                row.drug_id,
            ),
            source="DrugCentral",
            external_id=row.drug_id,
            title=title,
            url="https://drugcentral.org/",
            supports="support",
            excerpt_hash=_excerpt_hash(title),
            metadata={
                "live": False,
                "mock": False,
                "fixture": False,
                "cache_hit": False,
                "provenance_class": "external_local_dataset",
                "retrieval_adapter": type(self).__name__,
                "official_download_url": "https://drugcentral.org/download",
                "dataset_filename": self.dataset_path.name,
                "dataset_version": self.dataset_version,
                "dataset_row_number": row.row_number,
                "drug_name": row.drug_name,
                "drug_id": row.drug_id,
                "target_accession": row.target_accession,
                "target_gene": row.target_gene,
                "target_name": row.target_name,
                "organism": row.organism,
                "action_type": row.action_type,
                "moa": row.moa,
                "direction_evidence_rule": direction_rule,
                "raw_activity": self._raw_activity_payload(row),
                "activity_units_inferred": False,
                "activity_source_url": row.activity_source_url,
                "evidence_scope": "exact target-drug interaction and declared activity metadata",
            },
        )

    @staticmethod
    def _raw_activity_payload(row: _DrugCentralRow) -> dict[str, str | None]:
        return {
            "ACT_TYPE": row.activity_type,
            "ACT_VALUE": row.activity_value,
            "ACT_UNIT": row.activity_unit,
            "RELATION": row.relation,
            "ACT_SOURCE": row.activity_source,
        }

    @staticmethod
    def _dedupe_evidence(items: Sequence[EvidenceRef]) -> list[EvidenceRef]:
        by_id: dict[str, EvidenceRef] = {}
        for item in items:
            by_id[item.evidence_id] = item.model_copy(deep=True)
        return list(by_id.values())

    @staticmethod
    def _dedupe_activities(
        items: Sequence[ActivityMeasurement],
    ) -> list[ActivityMeasurement]:
        by_value: dict[tuple[Any, ...], ActivityMeasurement] = {}
        for item in items:
            key = (
                item.endpoint,
                item.relation,
                item.value,
                item.unit,
                item.source_evidence_id,
            )
            by_value[key] = item.model_copy(deep=True)
        return list(by_value.values())


__all__ = [
    "LiveRepurposingError",
    "LiveScientificToolNotConfigured",
    "NoRealRepurposingCandidates",
    "PublicRepurposingGateway",
    "RepurposingSourceUnavailable",
]
