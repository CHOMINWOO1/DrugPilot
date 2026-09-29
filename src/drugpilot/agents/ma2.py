from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from drugpilot.domain.models import (
    ADMETExecutionResult,
    ActivityMeasurement,
    CandidateCard,
    CandidateMetrics,
    CandidatePacket,
    CandidateRoute,
    ChemistryExecutionResult,
    DockingExecutionResult,
    EvidenceRef,
    FaultType,
    GatewayReadiness,
    Hypothesis,
    HypothesisPacket,
    PlanSpec,
    ReadinessStatus,
    ReworkOrder,
    ScientificBackendKind,
    ScientificExecutionStatus,
    TargetCandidate,
    TargetExecutionSummary,
    ToolReadiness,
    ToolCallRecord,
    new_id,
    utc_now,
)
from drugpilot.tools.chemistry import ChemistryValidator
from drugpilot.tools.substructure_policy import screen_forbidden_substructures
from drugpilot.tools.activity_comparability import (
    comparability_summary,
    derive_activity_rankings,
    measurement_evidence_quality,
)
from drugpilot.tools.assay_variant_alignment import assay_variant_alignment, bound_semantic_hypothesis
from drugpilot.tools.candidate_evidence_identity import (
    attest_candidate_evidence,
    candidate_linked_evidence,
    evidence_scoring_cardinality,
)
from drugpilot.tools.candidate_policy import (
    candidate_identity,
    screen_forbidden_drug,
    screen_reserved_control,
)
from drugpilot.tools.fixture_gateway import DeterministicFixtureGateway
from drugpilot.tools.executable import (
    ScientificDataUnavailable,
    ScientificToolError,
    ScientificToolUnavailable,
)
from drugpilot.tools.live_repurposing_gateway import (
    LiveRepurposingError,
    NoRealRepurposingCandidates,
    PAPER_NATIVE_V5_REQUIRED_POLICY,
    PAPER_NATIVE_V6_REQUIRED_POLICY,
)
from drugpilot.tools.live_ma2_gateway import NoMeasuredLigandCandidates
from drugpilot.tools.protocols import CandidateSeed, CandidateToolGateway
from drugpilot.tools.portfolio_ranking import rank_portfolio, ranking_attestation
from drugpilot.tools.route_policy import (
    KNOWN_DRUG_REPURPOSING_THRESHOLD,
    MIN_DE_NOVO_ASSOCIATION_SCORE,
    ROUTE_POLICY_VERSION,
    UNDEREXPLORED_KNOWN_DRUG_MAX,
    select_hypothesis_for_priority,
    select_route_from_evidence,
)


T = TypeVar("T")


class PaperNativeV5GroupUnavailable(ScientificDataUnavailable):
    """The indivisible paper-native-v5 evidence group is contradictory."""


class PaperNativeV6GroupUnavailable(ScientificDataUnavailable):
    """The candidate-per-primary-record v6 group is contradictory."""


_PAPER_NATIVE_V5_GROUP_PROJECTION_STOP_RULE = (
    "eligible_quota_and_declared_group_projection_complete_v1"
)
_PAPER_NATIVE_V5_GROUP_PROJECTION_STOP_REASON = (
    "ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"
)
_PAPER_NATIVE_V6_GROUP_PROJECTION_STOP_RULE = (
    "eligible_quota_reached_in_exact_v6_verified_pool_prefix_v1"
)
_PAPER_NATIVE_V6_GROUP_PROJECTION_STOP_REASON = (
    "ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX"
)
_PROSPECTIVE_FALSIFICATION_KO = re.compile(
    r"최소\s*(?P<replicates>\d+)\s*개의?\s*독립\s*생물학적\s*반복.*?"
    r"급성\s*생존율\s*감소가\s*(?P<acute_viability>\d+(?:\.\d+)?)%\s*이하인\s*노출에서\s*"
    r"(?P<early_time>\d+(?:\.\d+)?)시간\s*"
    r"(?P<early_readout>[A-Za-z][A-Za-z0-9_-]*)가\s*vehicle\s*대비\s*"
    r"(?P<early_effect>\d+(?:\.\d+)?)%\s*이상\s*감소하고.*?"
    r"(?P<late_time>\d+(?:\.\d+)?)시간\s*"
    r"(?P<late_readout>[A-Za-z][A-Za-z0-9_-]*(?:\s+count)?)가\s*vehicle\s*대비\s*"
    r"(?P<late_effect>\d+(?:\.\d+)?)%\s*이상\s*감소하며\s*"
    r"각\s*방향성\s*효과의\s*(?P<ci_level>\d+(?:\.\d+)?)%\s*신뢰구간이\s*0을\s*제외",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class _CandidateDiscoveryOutcome:
    """Internal typed boundary for a successful search, including zero hits."""

    seeds: tuple[CandidateSeed, ...]
    outcome: str
    reason_code: str | None
    provenance: Mapping[str, Any]


class CandidateEvidenceEnricher(Protocol):
    """Optional read-only public-data enrichment boundary for ranked candidates."""

    def enrich(
        self,
        request: Any,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> Any: ...


class RestrictedCandidateEvidenceEnricher(Protocol):
    """Route-aware boundary for local, licensed, or metered candidate sources."""

    def enrich_candidate(
        self,
        candidate: CandidateCard,
        hypothesis: Hypothesis,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> Any: ...


_PROHIBITED_EXTERNAL_STATUSES = frozenset(
    {
        "banned",
        "prohibited",
        "recalled",
        "suspended",
        "withdrawn",
        "withdrawn from market",
    }
)

_REQUIRED_ADMET_ENDPOINTS = (
    "hepatotoxicity",
    "herg_risk",
    "ames_risk",
    "cyp3a4_inhibition",
    "clearance_risk",
    "oral_bioavailability",
)

_PREDICTIVE_ADMET_WARNING_FLAGS = frozenset(
    {
        "hepatotoxicity_high",
        "herg_risk_high",
        "ames_risk_high",
        "cyp3a4_inhibition_high",
        "clearance_risk_high",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return {str(key): _jsonable(item) for key, item in vars(value).items()}
    return value


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def _as_provenance(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        return dict(payload) if isinstance(payload, Mapping) else {}
    provenance = getattr(value, "provenance", None)
    if callable(provenance):
        payload = provenance()
        return dict(payload) if isinstance(payload, Mapping) else {}
    if isinstance(provenance, Mapping):
        return dict(provenance)
    return {}


def _result_provenance(value: Any) -> dict[str, Any]:
    provenance = _as_provenance(value)
    backend_version = getattr(value, "backend_version", None)
    if backend_version:
        provenance.setdefault("backend_version", str(backend_version))
    return provenance


def _exception_provenance(exc: Exception) -> dict[str, Any]:
    provenance = _as_provenance(getattr(exc, "provenance", None))
    provenance.setdefault("exception_type", type(exc).__name__)
    return provenance


_PAPER_NATIVE_V6_FAILURE_REASON_CODES = frozenset(
    {
        "PAPER_NATIVE_V6_REPURPOSING_CONTRACT_UNAVAILABLE",
        "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT",
        "PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID",
    }
)


def _bounded_paper_native_v6_failure_summary(
    reason_code: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project hash-verified, bounded V6 failure fields into packet metadata.

    The failed tool call keeps the complete raw provenance, but the regular
    candidate artifact also needs enough structured information to explain
    why each member of the fixed pool did not qualify.  Only projections whose
    canonical hashes and row contracts recompute exactly are copied here.
    """

    if reason_code not in _PAPER_NATIVE_V6_FAILURE_REASON_CODES:
        return None

    def bounded_count(value: object, *, minimum: int = 0) -> int | None:
        return (
            value
            if type(value) is int and minimum <= value <= 12
            else None
        )

    sha256_pattern = re.compile(r"[0-9a-f]{64}")
    reason_pattern = re.compile(r"[A-Z][A-Z0-9_]{0,119}")

    def valid_sha256(value: object) -> bool:
        return isinstance(value, str) and sha256_pattern.fullmatch(value) is not None

    def projection_attestation_valid(value: Mapping[str, Any]) -> bool:
        supplied = value.get("attestation_sha256")
        if not valid_sha256(supplied):
            return False
        without_attestation = dict(value)
        without_attestation.pop("attestation_sha256", None)
        return supplied == _fingerprint(without_attestation)

    projection = provenance.get("wrapper_atomic_screen_projection")
    delegate_provenance = provenance.get("delegate_provenance")
    source_projection = (
        delegate_provenance.get("delegate_seed_source_binding_projection")
        if isinstance(delegate_provenance, Mapping)
        else None
    )
    projection_summary: dict[str, Any] | None = None
    if isinstance(projection, Mapping):
        fixed_ids = projection.get("fixed_pool_candidate_ids")
        rows = projection.get("rows")
        hashes = {
            key: projection.get(key)
            for key in (
                "fixed_pool_candidate_ids_sha256",
                "rows_sha256",
                "attestation_sha256",
            )
        }
        source_reason_by_id: dict[str, str | None] = {}
        source_projection_valid = False
        if isinstance(source_projection, Mapping):
            source_rows = source_projection.get("rows")
            source_projection_valid = bool(
                source_projection.get("schema_version")
                == "paper_native_v6_delegate_seed_source_binding_projection/v1"
                and source_projection.get("fixed_pool_candidate_ids") == fixed_ids
                and source_projection.get("fixed_pool_candidate_ids_sha256")
                == _fingerprint(fixed_ids)
                and isinstance(source_rows, list)
                and source_projection.get("row_count") == len(source_rows)
                and source_projection.get("rows_sha256") == _fingerprint(source_rows)
                and source_projection.get("complete") is True
                and source_projection.get("fixture") is False
                and projection_attestation_valid(source_projection)
            )
            if source_projection_valid:
                for index, source_row in enumerate(source_rows):
                    if not isinstance(source_row, Mapping):
                        source_projection_valid = False
                        break
                    candidate_id = source_row.get("molecule_chembl_id")
                    supplied_row_hash = source_row.get("row_sha256")
                    row_without_hash = dict(source_row)
                    row_without_hash.pop("row_sha256", None)
                    source_reason = source_row.get("primary_reason_code")
                    if not (
                        source_row.get("pool_index") == index
                        and source_row.get("fixed_pool_rank") == index + 1
                        and isinstance(fixed_ids, list)
                        and index < len(fixed_ids)
                        and candidate_id == fixed_ids[index]
                        and source_row.get("fixture") is False
                        and supplied_row_hash == _fingerprint(row_without_hash)
                        and (
                            source_reason is None
                            or (
                                isinstance(source_reason, str)
                                and reason_pattern.fullmatch(source_reason) is not None
                            )
                        )
                    ):
                        source_projection_valid = False
                        break
                    source_reason_by_id[str(candidate_id)] = source_reason
            if not source_projection_valid:
                source_reason_by_id = {}

        projected_rows: list[dict[str, Any]] = []
        projection_valid = bool(
            projection.get("schema_version")
            == "paper_native_v6_wrapper_atomic_screen_projection/v1"
            and isinstance(fixed_ids, list)
            and 1 <= len(fixed_ids) <= 12
            and len(fixed_ids) == len(set(fixed_ids))
            and all(
                isinstance(candidate_id, str)
                and re.fullmatch(r"CHEMBL[1-9][0-9]*", candidate_id) is not None
                for candidate_id in fixed_ids
            )
            and projection.get("fixed_pool_candidate_ids_sha256")
            == _fingerprint(fixed_ids)
            and isinstance(rows, list)
            and projection.get("row_count") == len(rows) == len(fixed_ids)
            and projection.get("rows_sha256") == _fingerprint(rows)
            and projection.get("complete") is True
            and projection.get("fixture") is False
            and projection_attestation_valid(projection)
        )
        if projection_valid:
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    projection_valid = False
                    break
                status = row.get("wrapper_screen_status")
                wrapper_reason = row.get("wrapper_screen_reason_code")
                checks = row.get("wrapper_screen_checks")
                supplied_row_hash = row.get("row_sha256")
                row_without_hash = dict(row)
                row_without_hash.pop("row_sha256", None)
                checks_valid = bool(
                    isinstance(checks, Mapping)
                    and set(checks)
                    == {"cardinality_exact", "primary_mapping", "exact_binding"}
                    and all(type(value) is bool for value in checks.values())
                    and row.get("wrapper_screen_checks_sha256")
                    == _fingerprint(dict(checks))
                )
                status_valid = bool(
                    checks_valid
                    and (
                        status == "verified"
                        and wrapper_reason is None
                        and row.get("delegate_returned") is True
                        and all(checks.values())
                        or status == "not_returned"
                        and wrapper_reason == "DELEGATE_NOT_RETURNED"
                        and row.get("delegate_returned") is False
                        and not any(checks.values())
                        or status == "rejected"
                        and wrapper_reason == "SEED_CARDINALITY_NOT_EXACT"
                        and row.get("delegate_returned") is True
                        and checks.get("cardinality_exact") is False
                        and checks.get("exact_binding") is False
                        or status == "rejected"
                        and wrapper_reason == "PRIMARY_SOURCE_NOT_MAPPING"
                        and row.get("delegate_returned") is True
                        and checks.get("cardinality_exact") is True
                        and checks.get("primary_mapping") is False
                        and checks.get("exact_binding") is False
                        or status == "rejected"
                        and wrapper_reason == "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
                        and row.get("delegate_returned") is True
                        and checks.get("cardinality_exact") is True
                        and checks.get("primary_mapping") is True
                        and checks.get("exact_binding") is False
                    )
                )
                candidate_id = row.get("molecule_chembl_id")
                if not (
                    row.get("pool_index") == index
                    and row.get("fixed_pool_rank") == index + 1
                    and candidate_id == fixed_ids[index]
                    and row.get("fixture") is False
                    and supplied_row_hash == _fingerprint(row_without_hash)
                    and status_valid
                ):
                    projection_valid = False
                    break
                projected_rows.append(
                    {
                        "fixed_pool_rank": index + 1,
                        "molecule_chembl_id": candidate_id,
                        "delegate_returned": row.get("delegate_returned"),
                        "wrapper_screen_status": status,
                        "wrapper_screen_reason_code": wrapper_reason,
                        "source_primary_reason_code": source_reason_by_id.get(
                            str(candidate_id)
                        ),
                        "wrapper_screen_checks": dict(checks),
                        "row_sha256": supplied_row_hash,
                    }
                )
        if projection_valid:
            rejection_reason_counts: dict[str, int] = {}
            for row in projected_rows:
                if row["wrapper_screen_status"] == "verified":
                    continue
                row_reason = (
                    row.get("source_primary_reason_code")
                    or row.get("wrapper_screen_reason_code")
                )
                if isinstance(row_reason, str):
                    rejection_reason_counts[row_reason] = (
                        rejection_reason_counts.get(row_reason, 0) + 1
                    )
            projection_summary = {
                "schema_version": projection.get("schema_version"),
                "fixed_pool_candidate_ids": list(fixed_ids),
                "row_count": len(projected_rows),
                "rows": projected_rows,
                "complete": True,
                "fixture": False,
                "source_reason_projection_verified": source_projection_valid,
                "rejection_reason_counts": rejection_reason_counts,
                **hashes,
            }
    grade = provenance.get("evidence_grade")
    if grade not in {"B", "C"}:
        grade = None
    summary = {
        "reason_code": reason_code,
        "support_candidate_count": bounded_count(
            provenance.get("support_candidate_count")
        ),
        "minimum_verified_candidate_count": bounded_count(
            provenance.get("minimum_verified_candidate_count"), minimum=1
        ),
        "evidence_grade": grade,
        "promotion_eligible": (
            False if provenance.get("promotion_eligible") is False else None
        ),
        "rejection_count": bounded_count(provenance.get("rejection_count")),
        "rejections_sha256": (
            provenance.get("rejections_sha256")
            if isinstance(provenance.get("rejections_sha256"), str)
            and re.fullmatch(
                r"[0-9a-f]{64}", provenance["rejections_sha256"]
            )
            is not None
            else None
        ),
        "wrapper_atomic_screen_projection": projection_summary,
        "raw_provenance_retained_in_tool_call": True,
        "fixture_fallback_used": False,
        "missing_values_imputed": False,
    }
    return summary


class _TraceRecorder:
    def __init__(
        self,
        source_version: str,
        existing: Iterable[ToolCallRecord] = (),
        *,
        backend_kind: ScientificBackendKind = ScientificBackendKind.FIXTURE,
        fixture: bool = True,
        before_operation: Callable[[], None] | None = None,
    ) -> None:
        self.source_version = source_version
        self.calls = list(existing)
        self.backend_kind = backend_kind
        self.fixture = fixture
        self.before_operation = before_operation

    def invoke(
        self,
        *,
        task_id: str | None,
        tool_name: str,
        operation: str,
        inputs: Any,
        function: Callable[[], T],
        cached: bool = False,
        chemical_policy_screen_id: str | None = None,
    ) -> tuple[T, str]:
        if self.before_operation is not None:
            self.before_operation()
        started = utc_now()
        input_hash = _fingerprint(inputs)
        molecular_input_binding = None
        if chemical_policy_screen_id is not None:
            if tool_name not in {"chemistry_validator", "docking", "admet_predictor"} or not isinstance(inputs, Mapping):
                raise ValueError("Chemical policy binding requires molecular computation inputs")
            # Snapshot actual call arguments, not result-provider assertions.
            # The target is already preserved on the CandidateCard; its digest
            # avoids duplicating potentially large evidence metadata per call.
            arguments = _jsonable(dict(inputs))
            target_input = arguments.pop("target", None)
            molecular_input_binding = {
                "schema_version": "chemical-compute-input-v1",
                "screen_call_id": chemical_policy_screen_id,
                "arguments": arguments,
                "target_input_sha256": _fingerprint(target_input) if "target" in inputs else None,
            }
        try:
            output = function()
        except Exception as exc:
            finished = utc_now()
            status = (
                ScientificExecutionStatus.UNAVAILABLE
                if isinstance(
                    exc,
                    (
                        LiveRepurposingError,
                        ScientificDataUnavailable,
                        ScientificToolUnavailable,
                    ),
                )
                else ScientificExecutionStatus.FAILED
            )
            provenance = _exception_provenance(exc)
            provenance.setdefault("input_hash", input_hash)
            if molecular_input_binding is not None:
                provenance["molecular_input_binding"] = molecular_input_binding
            self.calls.append(
                ToolCallRecord(
                    task_id=task_id,
                    tool_name=tool_name,
                    operation=operation,
                    input_hash=input_hash,
                    output_hash=_fingerprint({"error": type(exc).__name__, "message": str(exc)}),
                    source_version=self.source_version,
                    started_at=started,
                    finished_at=finished,
                    cached=cached,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    backend_kind=self.backend_kind,
                    fixture=self.fixture,
                    execution_status=status,
                    provenance=provenance,
                )
            )
            raise
        finished = utc_now()
        status = getattr(output, "status", ScientificExecutionStatus.COMPLETED)
        if not isinstance(status, ScientificExecutionStatus):
            status = ScientificExecutionStatus(str(status))
        provenance = _result_provenance(output)
        provenance.setdefault("input_hash", input_hash)
        if molecular_input_binding is not None:
            provenance["molecular_input_binding"] = molecular_input_binding
        record = ToolCallRecord(
            task_id=task_id,
            tool_name=tool_name,
            operation=operation,
            input_hash=input_hash,
            output_hash=_fingerprint(output),
            source_version=self.source_version,
            started_at=started,
            finished_at=finished,
            cached=cached,
            success=status in {
                ScientificExecutionStatus.COMPLETED,
                ScientificExecutionStatus.SKIPPED,
            },
            backend_kind=self.backend_kind,
            fixture=self.fixture,
            execution_status=status,
            provenance=provenance,
        )
        self.calls.append(record)
        return output, record.call_id


class MA2Agent:
    """Candidate discovery agent with evidence-selected routing and tool lineage.

    The default gateway is fully deterministic and offline.  Swapping the
    gateway is the only change needed to connect REST, MCP, docking, or ADMET
    services later; orchestration and artifact schemas remain unchanged.
    """

    def __init__(
        self,
        gateway: CandidateToolGateway | None = None,
        chemistry: ChemistryValidator | None = None,
        *,
        max_refinement_iterations: int = 1,
        evidence_enricher: CandidateEvidenceEnricher | None = None,
        restricted_evidence_enricher: RestrictedCandidateEvidenceEnricher | None = None,
        enrichment_per_route: int = 0,
        ma2_portfolio_enabled: bool = False,
    ) -> None:
        if max_refinement_iterations < 0:
            raise ValueError("max_refinement_iterations must be non-negative")
        if enrichment_per_route < 0:
            raise ValueError("enrichment_per_route must be non-negative")
        self.gateway = gateway or DeterministicFixtureGateway()
        raw_backend_kind = getattr(
            self.gateway,
            "backend_kind",
            ScientificBackendKind.FIXTURE.value,
        )
        self.backend_kind = (
            raw_backend_kind
            if isinstance(raw_backend_kind, ScientificBackendKind)
            else ScientificBackendKind(str(raw_backend_kind))
        )
        self.fixture_mode = bool(
            getattr(
                self.gateway,
                "fixture",
                self.backend_kind is ScientificBackendKind.FIXTURE,
            )
        )
        if self.fixture_mode != (self.backend_kind is ScientificBackendKind.FIXTURE):
            raise ValueError("gateway fixture declaration conflicts with backend_kind")
        # Fixture behavior must stay repeatable even when RDKit happens to be
        # installed on the host. Live mode, conversely, must never use the
        # syntax approximation.
        self.chemistry = chemistry or ChemistryValidator(
            prefer_rdkit=not self.fixture_mode
        )
        self.max_refinement_iterations = max_refinement_iterations
        self.evidence_enricher = evidence_enricher
        self.restricted_evidence_enricher = restricted_evidence_enricher
        self.enrichment_per_route = enrichment_per_route
        self.ma2_portfolio_enabled = bool(ma2_portfolio_enabled)
        self.before_operation: Callable[[], None] | None = None

    def close(self) -> None:
        """Release live scientific workers owned by this MA2 instance.

        Fixture gateways do not own processes, while the live delegate may
        keep an ADMET inference worker resident between predictions.  Traverse
        only the explicitly composed gateway/delegate components and call each
        unique lifecycle hook once.
        """

        delegate = getattr(self.gateway, "delegate", None)
        components = (
            getattr(delegate, "admet", None),
            getattr(delegate, "docking", None),
            delegate,
            self.gateway,
        )
        closed: set[int] = set()
        for component in components:
            if component is None or id(component) in closed:
                continue
            closed.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                close()

    def _gateway_readiness(
        self,
        target: TargetCandidate | None,
    ) -> GatewayReadiness:
        declared = getattr(self.gateway, "readiness", None)
        if callable(declared):
            try:
                report = declared(target)
                return (
                    report
                    if isinstance(report, GatewayReadiness)
                    else GatewayReadiness.model_validate(report)
                )
            except Exception as exc:
                if self.fixture_mode:
                    raise
                return self._fallback_readiness(
                    target,
                    declaration_error=f"{type(exc).__name__}: {exc}",
                )
        return self._fallback_readiness(target)

    def _fallback_readiness(
        self,
        target: TargetCandidate | None,
        *,
        declaration_error: str | None = None,
    ) -> GatewayReadiness:
        """Conservative capability declaration for structural legacy gateways."""

        delegate = getattr(self.gateway, "delegate", None)
        dataset_path = getattr(self.gateway, "dataset_path", None)
        dataset_ready = bool(dataset_path and Path(dataset_path).is_file())
        capabilities: list[ToolReadiness] = []
        definitions = (
            ("target_normalizer", "normalize_target", True, True),
            (
                "repurposing_search",
                "retrieve_candidates",
                True,
                self.fixture_mode or dataset_ready or delegate is not None,
            ),
            (
                "de_novo_generator",
                "generate_candidates",
                True,
                self.fixture_mode or delegate is not None,
            ),
            (
                "docking",
                "score_binding_pose",
                False,
                self.fixture_mode or delegate is not None,
            ),
            (
                "admet_predictor",
                "predict_six_endpoint_panel",
                True,
                self.fixture_mode or delegate is not None,
            ),
        )
        for tool_name, operation, required, nominally_available in definitions:
            ready = nominally_available and declaration_error is None
            capabilities.append(
                ToolReadiness(
                    tool_name=tool_name,
                    operation=operation,
                    status=(
                        ReadinessStatus.READY
                        if ready
                        else ReadinessStatus.NOT_CONFIGURED
                    ),
                    required=required,
                    backend_kind=self.backend_kind,
                    backend_name=type(self.gateway).__name__,
                    backend_version=str(self.gateway.source_version),
                    fixture=self.fixture_mode,
                    reason_code=(
                        None
                        if ready
                        else "READINESS_NOT_DECLARED_OR_TOOL_NOT_CONFIGURED"
                    ),
                    detail=(
                        declaration_error
                        or (
                            None
                            if ready
                            else "The gateway did not declare this executable capability."
                        )
                    ),
                    input_requirements=(
                        [f"resolved target: {target.symbol}"] if target else []
                    ),
                    provenance={
                        "declaration": "legacy_gateway_conservative_inference",
                        "dataset_available": dataset_ready,
                        "delegate_configured": delegate is not None,
                    },
                )
            )
        return GatewayReadiness(
            gateway_name=type(self.gateway).__name__,
            source_version=str(self.gateway.source_version),
            backend_kind=self.backend_kind,
            fixture=self.fixture_mode,
            capabilities=capabilities,
        )

    def _unavailable_packet(
        self,
        *,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        readiness: GatewayReadiness,
        reason_code: str,
        detail: str,
        route_decision: Mapping[str, Any] | None = None,
        hypothesis_selection: Mapping[str, Any] | None = None,
        candidates: Sequence[CandidateCard] = (),
    ) -> CandidatePacket:
        failure_provenance = next(
            (
                call.provenance
                for call in reversed(recorder.calls)
                if not call.success
                and isinstance(call.provenance, Mapping)
                and call.provenance.get("reason_code") == reason_code
            ),
            {},
        )
        failure_summary = _bounded_paper_native_v6_failure_summary(
            reason_code,
            failure_provenance,
        )
        return CandidatePacket(
            plan_id=plan.plan_id,
            candidates=self._attest_final_policy_eligibility(candidates),
            tool_calls=recorder.calls,
            backend_kind=self.backend_kind,
            execution_status=ScientificExecutionStatus.UNAVAILABLE,
            gateway_readiness=readiness,
            abstain_reason_codes=[reason_code],
            metadata={
                "route_decision": dict(route_decision or {}),
                "primary_hypothesis_selection": dict(hypothesis_selection or {}),
                "scientific_execution": {
                    "status": "unavailable",
                    "reason_codes": [reason_code],
                    "detail": detail,
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                    **(
                        {"failure_summary": failure_summary}
                        if failure_summary is not None
                        else {}
                    ),
                },
                **(
                    {"scientific_failure_summary": failure_summary}
                    if failure_summary is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _unavailable_reason_code(exc: Exception) -> str:
        # Live scientific adapters expose bounded machine-readable reason
        # codes for distinctions such as an unavailable upstream service,
        # insufficient independently bound primary records, or a contradictory
        # source contract.  Preserve that exact distinction in the candidate
        # packet instead of collapsing every case to the exception class name.
        # Fail closed on free-form or malformed values so exception text can
        # never become an unbounded status code.
        provenance = _exception_provenance(exc)
        for value in (
            getattr(exc, "reason_code", None),
            provenance.get("reason_code"),
        ):
            if (
                isinstance(value, str)
                and re.fullmatch(r"[A-Z][A-Z0-9_]{0,119}", value) is not None
            ):
                return value
        name = type(exc).__name__
        normalized = "".join(
            ("_" + char if char.isupper() and index else char).upper()
            for index, char in enumerate(name)
        )
        return normalized or "SCIENTIFIC_TOOL_UNAVAILABLE"

    @staticmethod
    def _candidate_unavailable_codes(
        candidates: Sequence[CandidateCard],
    ) -> list[str]:
        codes: list[str] = []
        for candidate in candidates:
            for result in (
                candidate.chemistry_execution,
                candidate.docking_execution,
                candidate.admet_execution,
            ):
                if result is None or result.status not in {
                    ScientificExecutionStatus.UNAVAILABLE,
                    ScientificExecutionStatus.FAILED,
                }:
                    continue
                codes.append(result.reason_code or "SCIENTIFIC_TOOL_UNAVAILABLE")
        return list(dict.fromkeys(codes))

    @staticmethod
    def _attest_final_policy_eligibility(
        candidates: Sequence[CandidateCard],
    ) -> list[CandidateCard]:
        """Synchronize final policy metadata with status and ranking.

        Seed and route metadata describe pre-compute eligibility.  A later
        scientific exclusion must override that provisional value in every
        emitted candidate artifact.  Conversely, only a non-excluded card
        that actually received a within-target rank is finally eligible.
        This is a metadata attestation only; it does not alter status, rank,
        score, or gate behavior.
        """

        attested: list[CandidateCard] = []
        for candidate in candidates:
            ranked = candidate.rank is not None
            policy_eligible = candidate.status != "excluded" and ranked
            attested.append(
                candidate.model_copy(
                    update={
                        "metadata": {
                            **candidate.metadata,
                            "policy_eligible": policy_eligible,
                            "policy_status_attestation": {
                                "rule": (
                                    "policy_eligible_iff_candidate_is_not_"
                                    "excluded_and_ranked"
                                ),
                                "candidate_status": candidate.status,
                                "ranked": ranked,
                                "policy_eligible": policy_eligible,
                            },
                        }
                    },
                    deep=True,
                )
            )
        return attested

    @staticmethod
    def _attest_prospective_falsification_status(
        candidates: Sequence[CandidateCard],
        hypothesis: Hypothesis,
    ) -> list[CandidateCard]:
        """Keep prospective wet-lab criteria distinct from observed support.

        MA2 may rank source-eligible candidates for follow-up, but it has not
        performed the ResearchBrief's prospective experiment.  The immutable
        packet therefore records an explicit per-candidate unevaluated state
        instead of leaving reviewers to infer that distinction from prose.
        """

        falsification_test = hypothesis.falsification_test.strip()
        test_sha256 = sha256(falsification_test.encode("utf-8")).hexdigest()
        contract = MA2Agent._prospective_falsification_contract(
            falsification_test
        )
        from drugpilot.agents.hypothesis_admission import admitted_tested_model_context
        tested_model_context = admitted_tested_model_context(hypothesis)
        attested: list[CandidateCard] = []
        for candidate in candidates:
            payload = {
                "schema_version": "candidate_prospective_falsification_v3",
                "candidate_id": candidate.candidate_id,
                "hypothesis_id": hypothesis.hypothesis_id,
                "status": "not_evaluated",
                "prospective_evidence_disposition": (
                    "unobserved_no_outcome"
                ),
                "prospective_follow_up_support": False,
                "default_falsification_applied": False,
                "biological_refutation_established": False,
                "ranking_interpretation": (
                    "source_supported_pre_experiment_prioritization_only"
                ),
                "falsification_test": falsification_test,
                "falsification_test_sha256": test_sha256,
                "experimentally_observed": False,
                "used_for_source_eligibility": False,
                "used_for_candidate_ranking": False,
                "machine_readable_contract": deepcopy(contract),
                "fixture": False,
            }
            payload["attestation_sha256"] = sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            attested.append(
                candidate.model_copy(
                    update={
                        "metadata": {
                            **candidate.metadata,
                            "prospective_falsification_status": payload,
                            **({"ma3_tested_model_context": deepcopy(tested_model_context)}
                               if tested_model_context is not None else {}),
                        }
                    },
                    deep=True,
                )
            )
        return attested

    @staticmethod
    def _prospective_falsification_contract(
        falsification_test: str,
    ) -> dict[str, Any]:
        """Parse one closed researcher-authored joint wet-lab rule.

        The parser is deliberately narrow and fail-closed.  It never invents
        a threshold: every numeric field is copied from one exact matched
        clause and the clause/hash are retained.  An unmatched rule remains
        explicitly unstructured and therefore cannot be mistaken for a
        machine-evaluated positive result.
        """

        match = _PROSPECTIVE_FALSIFICATION_KO.search(falsification_test)
        if match is None:
            return {
                "schema_version": "prospective_joint_criteria_contract/v2",
                "policy": "closed_researcher_requirement_parse_no_unobserved_outcome_v2",
                "status": "unstructured",
                "evaluation_status": "not_evaluated",
                "criteria_conjunction": "all",
                "candidate_specific": True,
                "when_unobserved": "not_evaluated",
                "when_observed_but_qc_invalid": "invalid_or_inconclusive",
                "when_observed_criteria_not_met": "requires_validated_outcome_review",
                "source_clause": None,
                "source_clause_sha256": None,
                "criteria": None,
                "observations": None,
            }

        def number(name: str) -> int | float:
            value = float(match.group(name))
            return int(value) if value.is_integer() else value

        source_clause = match.group(0)
        replicates = int(match.group("replicates"))
        ci_level = number("ci_level")
        criteria = {
            "minimum_independent_biological_replicates_per_directional_endpoint": (
                replicates
            ),
            "qualifying_exposure": {
                "same_exposure_required_for": [
                    "early_target_modulation",
                    "acute_viability_loss",
                    "late_viable_cell_change",
                ],
                "acute_viability_loss_relation": "<=",
                "acute_viability_loss_threshold_percent": number(
                    "acute_viability"
                ),
                "minimum_independent_biological_replicates": replicates,
                "confidence_interval_level_percent": ci_level,
                "confidence_interval_must_exclude_zero": True,
                "selection_rule": (
                    "lowest_pre_specified_nonzero_exposure_satisfying_"
                    "acute_viability_ceiling_before_directional_outcome_review"
                ),
                "selection_rule_basis": (
                    "generic_anti_post_observation_selection_guard"
                ),
                "selection_rule_researcher_authored": False,
                "exposure_grid_must_be_pre_specified_before_outcome_review": (
                    True
                ),
                "exposure_grid": None,
                "exposure_grid_status": (
                    "required_not_supplied_fail_closed"
                ),
            },
            "directional_endpoints": [
                {
                    "key": "early_target_modulation",
                    "readout": match.group("early_readout"),
                    "timepoint_hours": number("early_time"),
                    "effect": "decrease_vs_vehicle",
                    "effect_relation": ">=",
                    "effect_threshold_percent": number("early_effect"),
                    "minimum_independent_biological_replicates": replicates,
                    "confidence_interval_level_percent": ci_level,
                    "confidence_interval_must_exclude_zero": True,
                },
                {
                    "key": "late_viable_cell_change",
                    "readout": match.group("late_readout"),
                    "timepoint_hours": number("late_time"),
                    "effect": "decrease_vs_vehicle",
                    "effect_relation": ">=",
                    "effect_threshold_percent": number("late_effect"),
                    "minimum_independent_biological_replicates": replicates,
                    "confidence_interval_level_percent": ci_level,
                    "confidence_interval_must_exclude_zero": True,
                },
            ],
        }
        observations = {
            "qualifying_exposure": {
                "value": None,
                "unit": None,
                "acute_viability_loss_percent": None,
                "acute_viability_biological_replicate_count": None,
                "acute_viability_confidence_interval_95": {
                    "lower": None,
                    "upper": None,
                    "excludes_zero": None,
                },
                "early_target_and_acute_viability_same_exposure": None,
                "late_endpoint_same_exposure": None,
                "all_endpoints_same_qualifying_exposure": None,
                "selection_performed_before_directional_outcome_review": None,
                "tested_exposure_grid": None,
            },
            "early_target_modulation": {
                "effect_percent_vs_vehicle": None,
                "biological_replicate_count": None,
                "confidence_interval_95": {
                    "lower": None,
                    "upper": None,
                    "excludes_zero": None,
                },
            },
            "late_viable_cell_change": {
                "effect_percent_vs_vehicle": None,
                "biological_replicate_count": None,
                "confidence_interval_95": {
                    "lower": None,
                    "upper": None,
                    "excludes_zero": None,
                },
            },
        }
        return {
            "schema_version": "prospective_joint_criteria_contract/v2",
            "policy": "closed_researcher_requirement_parse_no_unobserved_outcome_v2",
            "status": "machine_readable",
            "evaluation_status": "not_evaluated",
            "criteria_conjunction": "all",
            "candidate_specific": True,
            "when_unobserved": "not_evaluated",
            "when_observed_but_qc_invalid": "invalid_or_inconclusive",
            "when_observed_criteria_not_met": "requires_validated_outcome_review",
            "source_clause": source_clause,
            "source_clause_sha256": sha256(
                source_clause.encode("utf-8")
            ).hexdigest(),
            "criteria": criteria,
            "criteria_sha256": sha256(
                json.dumps(
                    criteria,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "observations": observations,
        }

    def _retrieve_repurposing_outcome(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> _CandidateDiscoveryOutcome:
        """Return a completed zero-hit search without converting source failures.

        ``NoRealRepurposingCandidates`` also covers structure-resolution
        failures.  Only its explicit ``conclusive_empty_search`` state is a
        completed scientific result; every other instance is re-raised and
        retains the normal fail-closed path.
        """

        try:
            seeds = tuple(
                self.gateway.retrieve_repurposing(
                    target,
                    direction=direction,
                    limit=limit,
                )
            )
        except NoMeasuredLigandCandidates as exc:
            if not exc.conclusive:
                raise
            return _CandidateDiscoveryOutcome(
                seeds=(),
                outcome="no_candidates",
                reason_code=exc.reason_code,
                provenance={
                    **dict(exc.provenance),
                    "search_completed": True,
                    "outcome": "completed_no_candidates",
                    "reason_code": exc.reason_code,
                    "eligible_count": exc.eligible_count,
                    "minimum_required": exc.minimum_required,
                    "fixture": False,
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                },
            )
        except NoRealRepurposingCandidates as exc:
            if not exc.conclusive_empty_search:
                raise
            reason_code = (
                "NO_TARGET_LINKED_REPURPOSING_CANDIDATES"
                if exc.matching_rows == 0
                else "NO_DIRECTION_COMPATIBLE_REPURPOSING_CANDIDATES"
            )
            return _CandidateDiscoveryOutcome(
                seeds=(),
                outcome="no_candidates",
                reason_code=reason_code,
                provenance={
                    **dict(exc.provenance),
                    "outcome": "completed_no_candidates",
                    "reason_code": reason_code,
                },
            )
        return _CandidateDiscoveryOutcome(
            seeds=seeds,
            outcome="candidates",
            reason_code=None,
            provenance={
                **self._last_gateway_provenance(),
                "outcome": "candidates_returned",
                "returned_candidates": len(seeds),
            },
        )

    def _generate_de_novo_outcome(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> _CandidateDiscoveryOutcome:
        """Distinguish a conclusive ChEMBL-negative pool from tool failure."""

        try:
            seeds = tuple(
                self.gateway.generate_de_novo(
                    target,
                    direction=direction,
                    limit=limit,
                )
            )
        except NoMeasuredLigandCandidates as exc:
            if not exc.conclusive:
                raise
            return _CandidateDiscoveryOutcome(
                seeds=(),
                outcome="no_candidates",
                reason_code=exc.reason_code,
                provenance={
                    **dict(exc.provenance),
                    "search_completed": True,
                    "outcome": "completed_no_candidates",
                    "reason_code": exc.reason_code,
                    "eligible_count": exc.eligible_count,
                    "minimum_required": exc.minimum_required,
                    "fixture": False,
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                },
            )
        return _CandidateDiscoveryOutcome(
            seeds=seeds,
            outcome="candidates",
            reason_code=None,
            provenance={
                **self._last_gateway_provenance(),
                "outcome": "candidates_returned",
                "returned_candidates": len(seeds),
            },
        )

    def _completed_no_candidate_packet(
        self,
        *,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        readiness: GatewayReadiness,
        route_decision: Mapping[str, Any],
        hypothesis_selection: Mapping[str, Any],
        outcome: _CandidateDiscoveryOutcome,
    ) -> CandidatePacket:
        reason_code = outcome.reason_code or "NO_QUALIFIED_CANDIDATES"
        candidate_backfill = {
            "requested_eligible_count": plan.candidates_per_route,
            "discovery_pool_count": 0,
            "discovery_pool_limit": (
                plan.candidates_per_route
                if self.fixture_mode
                else min(100, max(4 * plan.candidates_per_route, 10))
            ),
            "evaluated_count": 0,
            "eligible_count": 0,
            "excluded_count": 0,
            "quota_met": False,
            "pool_exhausted": True,
            "evaluated_identities": [],
            "unevaluated_identities": [],
            "stop_reason": "DISCOVERY_POOL_EMPTY",
            "unbounded_search_used": False,
        }
        return CandidatePacket(
            plan_id=plan.plan_id,
            candidates=[],
            tool_calls=recorder.calls,
            backend_kind=self.backend_kind,
            execution_status=ScientificExecutionStatus.COMPLETED,
            gateway_readiness=readiness,
            metadata={
                "route_decision": dict(route_decision),
                "primary_hypothesis_selection": dict(hypothesis_selection),
                "candidate_yield": {
                    "requested_limit": plan.candidates_per_route,
                    "returned_count": 0,
                    "partial": True,
                    "fixture_fill_used": False,
                    "backend_kind": self.backend_kind.value,
                    "requested_eligible_count": plan.candidates_per_route,
                    "discovery_pool_count": 0,
                    "evaluated_count": 0,
                    "eligible_count": 0,
                    "quota_met": False,
                    "pool_exhausted": True,
                },
                "candidate_backfill": candidate_backfill,
                "candidate_search_outcome": {
                    "status": "no_candidates",
                    "reason_code": reason_code,
                    "search_completed": True,
                    "provenance": dict(outcome.provenance),
                },
                "scientific_execution": {
                    "status": "completed",
                    "outcome": "no_candidates",
                    "reason_codes": [reason_code],
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                },
            },
        )

    def run(
        self,
        plan: PlanSpec,
        hypotheses: HypothesisPacket,
        *,
        fault: FaultType = FaultType.NONE,
        allowed_sources: Sequence[str] | None = None,
    ) -> CandidatePacket:
        if hypotheses.provisional_hypotheses:
            raise ValueError("MA2 cannot execute provisional hypotheses awaiting MA3")
        if hypotheses.metadata.get("admission_policy") == "ma3-source-bound-v1":
            from drugpilot.agents.hypothesis_admission import admitted_support_ids
            if not hypotheses.hypotheses or any(
                not admitted_support_ids(h) for h in hypotheses.hypotheses
            ):
                raise ValueError("MA2 requires source-bound MA3 approval for every hypothesis")
        if self.ma2_portfolio_enabled:
            return self._run_portfolio(
                plan,
                hypotheses,
                fault=fault,
                allowed_sources=allowed_sources,
            )
        return self._run_single(
            plan,
            hypotheses,
            fault=fault,
            allowed_sources=allowed_sources,
        )

    @staticmethod
    def _portfolio_target_key(target: TargetCandidate) -> str:
        if target.uniprot_id:
            return f"uniprot:{target.uniprot_id.strip().upper()}"
        if target.ensembl_id:
            return f"ensembl:{target.ensembl_id.strip().upper()}"
        return f"symbol:{target.symbol.strip().upper()}"

    def _portfolio_identity_resolved(self, target: TargetCandidate) -> bool:
        if target.identity_status == "resolved":
            return bool(target.uniprot_id or target.ensembl_id or target.symbol.strip())
        return self.fixture_mode and target.identity_status == "fixture"

    @staticmethod
    def _portfolio_paper_v5_group_metadata(
        candidates: Sequence[CandidateCard],
        target_packet_metadata: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Preserve an exact single-target paper-native group at packet scope.

        The v5 and v6 verifiers audit the finite candidate prefix as one
        indivisible group.  Portfolio assembly must therefore carry the two
        already-produced target-packet summaries without synthesizing or
        merging them.  A multi-target or internally inconsistent projection is
        deliberately omitted so the downstream verifier fails closed.
        """

        if not candidates:
            return {}
        target_keys = {
            key
            for candidate in candidates
            if isinstance(
                key := candidate.metadata.get("portfolio_target_key"), str
            )
            and key
        }
        if len(target_keys) != 1:
            return {}
        source_metadata = target_packet_metadata.get(next(iter(target_keys)))
        if not isinstance(source_metadata, Mapping):
            return {}
        backfill = source_metadata.get("candidate_backfill")
        candidate_yield = source_metadata.get("candidate_yield")
        if not isinstance(backfill, Mapping) or not isinstance(
            candidate_yield, Mapping
        ):
            return {}
        for projection_key in (
            "paper_native_v6_group_projection",
            "paper_native_v5_group_projection",
        ):
            projection = backfill.get(projection_key)
            if (
                isinstance(projection, Mapping)
                and candidate_yield.get(projection_key) == projection
                and all(
                    candidate.metadata.get(projection_key) == projection
                    for candidate in candidates
                )
            ):
                return {
                    "candidate_backfill": deepcopy(dict(backfill)),
                    "candidate_yield": deepcopy(dict(candidate_yield)),
                }
        return {}

    @staticmethod
    def _minimum_eligible_aggregation_required(
        candidate_backfill: Mapping[str, Any],
    ) -> bool:
        """Return whether a verified paper-native group declares a hard quota."""

        return any(
            isinstance(candidate_backfill.get(key), Mapping)
            for key in (
                "paper_native_v6_group_projection",
                "paper_native_v5_group_projection",
            )
        )

    def _run_portfolio(
        self,
        plan: PlanSpec,
        hypotheses: HypothesisPacket,
        *,
        fault: FaultType,
        allowed_sources: Sequence[str] | None,
    ) -> CandidatePacket:
        """Execute every unique G0-approved, identity-resolved target independently."""

        if plan.hypothesis_packet_id != hypotheses.packet_id:
            raise ValueError("PlanSpec does not reference the supplied HypothesisPacket")
        if not hypotheses.hypotheses:
            raise ValueError("MA2 requires at least one hypothesis")

        primary, primary_selection = self._select_primary_hypothesis(plan, hypotheses)
        ordered_hypotheses = [primary] + [
            item
            for item in hypotheses.hypotheses
            if item.hypothesis_id != primary.hypothesis_id
        ]
        seen_targets: set[str] = set()
        summaries: list[TargetExecutionSummary] = []
        candidates: list[CandidateCard] = []
        calls: list[ToolCallRecord] = []
        public_queries: list[Any] = []
        max_iterations = 1
        first_readiness: GatewayReadiness | None = None
        primary_route_decision: dict[str, Any] = {}
        route_decisions: dict[str, dict[str, Any]] = {}
        hypothesis_scores: dict[str, float] = {}
        target_packet_metadata: dict[str, Mapping[str, Any]] = {}
        incomplete_quota_target_keys: set[str] = set()
        fault_pending = fault

        for hypothesis in ordered_hypotheses:
            key = self._portfolio_target_key(hypothesis.target)
            hypothesis_scores.setdefault(key, hypothesis.score)
            identity_resolved = self._portfolio_identity_resolved(hypothesis.target)
            if key in seen_targets:
                summaries.append(
                    TargetExecutionSummary(
                        hypothesis_id=hypothesis.hypothesis_id,
                        track=hypothesis.track,
                        target_key=key,
                        target=hypothesis.target,
                        hypothesis_score=hypothesis.score,
                        identity_resolved=identity_resolved,
                        execution_eligible=False,
                        attempted=False,
                        status="duplicate",
                        requested_candidate_count=plan.candidates_per_route,
                        reason_codes=["DUPLICATE_TARGET_IDENTITY"],
                        detail="A higher-priority hypothesis already represents this stable target identity.",
                    )
                )
                continue
            seen_targets.add(key)
            if not identity_resolved:
                readiness = self._gateway_readiness(hypothesis.target)
                if first_readiness is None:
                    first_readiness = readiness
                summaries.append(
                    TargetExecutionSummary(
                        hypothesis_id=hypothesis.hypothesis_id,
                        track=hypothesis.track,
                        target_key=key,
                        target=hypothesis.target,
                        hypothesis_score=hypothesis.score,
                        identity_resolved=False,
                        execution_eligible=False,
                        attempted=False,
                        readiness=readiness,
                        status="ineligible",
                        requested_candidate_count=plan.candidates_per_route,
                        reason_codes=["TARGET_IDENTITY_UNRESOLVED"],
                        detail="No resolved stable target identity was available for scientific tools.",
                    )
                )
                continue

            intended_route_decision = self._select_route(
                hypothesis, hypothesis.target
            )
            route_decisions[key] = dict(intended_route_decision)
            if hypothesis.hypothesis_id == primary.hypothesis_id:
                primary_route_decision = dict(intended_route_decision)
            route_fallback = intended_route_decision.get("fallback", {})
            if (
                isinstance(route_fallback, Mapping)
                and route_fallback.get("abstain_recommended") is True
            ):
                route_value = intended_route_decision.get("selected_route")
                summaries.append(
                    TargetExecutionSummary(
                        hypothesis_id=hypothesis.hypothesis_id,
                        track=hypothesis.track,
                        target_key=key,
                        target=hypothesis.target,
                        hypothesis_score=hypothesis.score,
                        identity_resolved=True,
                        execution_eligible=False,
                        attempted=False,
                        route=(CandidateRoute(route_value) if route_value else None),
                        route_decision=dict(intended_route_decision),
                        status="ineligible",
                        requested_candidate_count=plan.candidates_per_route,
                        reason_codes=["ROUTE_EVIDENCE_INSUFFICIENT"],
                        detail=(
                            "The deterministic route policy requested additional "
                            "evidence; no low-confidence fallback route was executed."
                        ),
                    )
                )
                continue

            single_hypotheses = hypotheses.model_copy(
                update={"hypotheses": [hypothesis]},
                deep=True,
            )
            single = self._run_single(
                plan,
                single_hypotheses,
                fault=fault_pending,
                allowed_sources=allowed_sources,
            )
            target_packet_metadata[key] = single.metadata
            fault_pending = FaultType.NONE
            if first_readiness is None and single.gateway_readiness is not None:
                first_readiness = single.gateway_readiness
            max_iterations = max(max_iterations, single.generation_iterations)
            calls.extend(single.tool_calls)
            public_queries.extend(single.metadata.get("public_enrichment_queries", []))
            decision = single.metadata.get("route_decision", {})
            decision = dict(decision) if isinstance(decision, Mapping) else {}
            if not decision:
                # The route is an MA1-evidence policy decision, so it remains
                # auditable even when target normalization or a downstream
                # executable fails before the route tool call can run.
                decision = dict(intended_route_decision)
            route_decisions[key] = decision
            if hypothesis.hypothesis_id == primary.hypothesis_id:
                primary_route_decision = decision

            candidate_backfill = single.metadata.get("candidate_backfill", {})
            candidate_backfill = (
                dict(candidate_backfill)
                if isinstance(candidate_backfill, Mapping)
                else {}
            )
            if (
                candidate_backfill.get("quota_met") is False
                and self._minimum_eligible_aggregation_required(
                    candidate_backfill
                )
            ):
                incomplete_quota_target_keys.add(key)

            target_cards = [
                card.model_copy(
                    update={
                        "target_rank": card.rank,
                        "metadata": {
                            **card.metadata,
                            "portfolio_target_key": key,
                            "portfolio_hypothesis_id": hypothesis.hypothesis_id,
                            "portfolio_hypothesis_score": hypothesis.score,
                            "portfolio_route": decision.get("selected_route"),
                        },
                    },
                    deep=True,
                )
                for card in single.candidates
            ]
            candidates.extend(target_cards)
            eligible_cards = [item for item in target_cards if item.status != "excluded"]
            reasons = list(single.abstain_reason_codes)
            if single.execution_status is ScientificExecutionStatus.COMPLETED:
                status = "completed" if eligible_cards else "no_candidates"
                if eligible_cards and key in incomplete_quota_target_keys:
                    status = "partial"
                    reasons = self._dedupe(
                        [
                            *reasons,
                            "ELIGIBLE_CANDIDATE_QUOTA_NOT_MET_POOL_EXHAUSTED",
                        ]
                    )
                if status == "no_candidates":
                    search_outcome = single.metadata.get(
                        "candidate_search_outcome", {}
                    )
                    search_reason = (
                        search_outcome.get("reason_code")
                        if isinstance(search_outcome, Mapping)
                        else None
                    )
                    reasons = self._dedupe(
                        [
                            str(search_reason or "NO_QUALIFIED_CANDIDATES"),
                            *(
                                reason
                                for item in target_cards
                                for reason in item.exclusion_reasons
                            ),
                        ]
                    )
            elif single.execution_status is ScientificExecutionStatus.FAILED:
                status = "failed"
            elif target_cards and eligible_cards:
                status = "partial"
            else:
                status = "unavailable"
            if status != "completed" and not reasons:
                reasons = ["TARGET_SCIENTIFIC_EXECUTION_INCOMPLETE"]
            route_value = decision.get("selected_route")
            summaries.append(
                TargetExecutionSummary(
                    hypothesis_id=hypothesis.hypothesis_id,
                    track=hypothesis.track,
                    target_key=key,
                    target=hypothesis.target,
                    hypothesis_score=hypothesis.score,
                    identity_resolved=True,
                    execution_eligible=True,
                    attempted=True,
                    route=(CandidateRoute(route_value) if route_value else None),
                    route_decision=decision,
                    readiness=single.gateway_readiness,
                    status=status,
                    requested_candidate_count=plan.candidates_per_route,
                    discovered_candidate_count=len(target_cards),
                    evaluated_candidate_count=int(
                        candidate_backfill.get("evaluated_count", len(target_cards))
                    ),
                    eligible_candidate_count=len(eligible_cards),
                    excluded_candidate_count=len(target_cards) - len(eligible_cards),
                    candidate_backfill=candidate_backfill,
                    candidate_ids=[item.candidate_id for item in target_cards],
                    tool_call_ids=[item.call_id for item in single.tool_calls],
                    reason_codes=reasons,
                    detail=single.metadata.get("scientific_execution", {}).get("detail"),
                )
            )

        rankable = [
            candidate
            for candidate in candidates
            if candidate.metadata.get("portfolio_target_key")
            not in incomplete_quota_target_keys
        ]
        held = [
            candidate.model_copy(
                update={
                    "portfolio_score": None,
                    "portfolio_rank": None,
                    "metadata": {
                        **candidate.metadata,
                        "portfolio_ranking": {
                            "version": ranking_attestation()["version"],
                            "status": "withheld",
                            "reason_code": (
                                "ELIGIBLE_CANDIDATE_QUOTA_NOT_MET_POOL_EXHAUSTED"
                            ),
                            "minimum_eligible_aggregation_required": True,
                        },
                    },
                },
                deep=True,
            )
            for candidate in candidates
            if candidate.metadata.get("portfolio_target_key")
            in incomplete_quota_target_keys
        ]
        ranked = self._attest_final_policy_eligibility(
            [
                *rank_portfolio(
                    rankable,
                    hypothesis_scores=hypothesis_scores,
                ),
                *held,
            ]
        )
        unique_calls = list({item.call_id: item for item in calls}.values())
        eligible_count = sum(item.execution_eligible for item in summaries)
        attempted_count = sum(item.attempted for item in summaries)
        completed_count = sum(item.status == "completed" for item in summaries)
        no_candidate_count = sum(
            item.status == "no_candidates" for item in summaries
        )
        concluded_count = completed_count + no_candidate_count
        unavailable_count = sum(
            item.status in {"partial", "unavailable", "failed", "ineligible", "duplicate"}
            for item in summaries
        )
        incomplete_eligible = [
            item
            for item in summaries
            if item.execution_eligible
            if item.status not in {"completed", "no_candidates", "duplicate"}
        ]
        reason_codes = self._dedupe(
            [code for item in incomplete_eligible for code in item.reason_codes]
        )
        if eligible_count == 0:
            execution_status = ScientificExecutionStatus.UNAVAILABLE
            non_executable_reasons = self._dedupe(
                [
                    code
                    for item in summaries
                    if item.status != "duplicate"
                    for code in item.reason_codes
                ]
            )
            reason_codes = self._dedupe(
                non_executable_reasons
                or ["NO_IDENTITY_RESOLVED_TARGETS"]
            )
        elif all(
            item.status in {"completed", "no_candidates", "duplicate"}
            for item in summaries
            if item.execution_eligible
        ):
            execution_status = ScientificExecutionStatus.COMPLETED
            reason_codes = []
        elif concluded_count > 0 or any(
            item.status == "partial" and item.eligible_candidate_count > 0
            for item in summaries
        ):
            execution_status = ScientificExecutionStatus.PARTIAL
            reason_codes = self._dedupe(["PORTFOLIO_PARTIAL_TARGET_COVERAGE", *reason_codes])
        elif any(item.status == "failed" for item in incomplete_eligible):
            execution_status = ScientificExecutionStatus.FAILED
            reason_codes = self._dedupe(["PORTFOLIO_TARGET_EXECUTION_FAILED", *reason_codes])
        else:
            execution_status = ScientificExecutionStatus.UNAVAILABLE
            reason_codes = self._dedupe(["PORTFOLIO_TARGETS_UNAVAILABLE", *reason_codes])

        paper_v5_group_metadata = self._portfolio_paper_v5_group_metadata(
            ranked, target_packet_metadata
        )

        return CandidatePacket(
            plan_id=plan.plan_id,
            candidates=ranked,
            tool_calls=unique_calls,
            generation_iterations=max_iterations,
            backend_kind=self.backend_kind,
            execution_status=execution_status,
            gateway_readiness=first_readiness,
            abstain_reason_codes=reason_codes,
            portfolio_enabled=True,
            g0_approved_target_count=len(summaries),
            eligible_target_count=eligible_count,
            attempted_target_count=attempted_count,
            completed_target_count=completed_count,
            unavailable_target_count=unavailable_count,
            target_executions=summaries,
            metadata={
                **paper_v5_group_metadata,
                "portfolio_enabled": True,
                "g0_approved_target_count": len(summaries),
                "eligible_target_count": eligible_count,
                "attempted_target_count": attempted_count,
                "completed_target_count": completed_count,
                "concluded_target_count": concluded_count,
                "no_candidate_target_count": no_candidate_count,
                "unavailable_target_count": unavailable_count,
                "portfolio_complete": execution_status is ScientificExecutionStatus.COMPLETED,
                "portfolio_has_eligible_candidate": any(
                    item.status != "excluded"
                    and item.portfolio_rank is not None
                    for item in ranked
                ),
                "portfolio_ranking": ranking_attestation(),
                "activity_comparability": self._activity_comparability_summary(
                    ranked
                ),
                "portfolio_route_decisions": route_decisions,
                "route_decision": primary_route_decision,
                "primary_hypothesis_selection": primary_selection,
                "public_enrichment_queries": public_queries,
                "exclusion_summary": self._exclusion_summary(ranked),
                "scientific_execution": {
                    "status": execution_status.value,
                    "reason_codes": reason_codes,
                    "outcome": (
                        "candidate_portfolio"
                        if any(
                            item.status != "excluded"
                            and item.portfolio_rank is not None
                            for item in ranked
                        )
                        else "candidate_portfolio_pending_quota"
                        if any(item.status != "excluded" for item in ranked)
                        else "no_qualified_candidates"
                    ),
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                },
            },
        )

    def _run_single(
        self,
        plan: PlanSpec,
        hypotheses: HypothesisPacket,
        *,
        fault: FaultType = FaultType.NONE,
        allowed_sources: Sequence[str] | None = None,
    ) -> CandidatePacket:
        """Execute the backward-compatible single-target component contract."""

        hypothesis, hypothesis_selection = self._select_primary_hypothesis(
            plan, hypotheses
        )
        readiness = self._gateway_readiness(hypothesis.target)
        recorder = _TraceRecorder(
            self.gateway.source_version,
            backend_kind=self.backend_kind,
            fixture=self.fixture_mode,
            before_operation=self.before_operation,
        )
        normalize_task = self._task_id(plan, None, "normalize", "target")
        try:
            target, normalization_trace = recorder.invoke(
                task_id=normalize_task,
                tool_name="target_normalizer",
                operation="normalize_target",
                inputs=hypothesis.target,
                function=lambda: self.gateway.normalize_target(hypothesis.target),
            )
        except (LiveRepurposingError, ScientificToolError) as exc:
            return self._unavailable_packet(
                plan=plan,
                recorder=recorder,
                readiness=readiness,
                reason_code=self._unavailable_reason_code(exc),
                detail=str(exc),
            )
        direction = hypothesis.intervention_direction
        count = plan.candidates_per_route
        discovery_pool_limit = (
            count
            if self.fixture_mode
            else 12
            if self._paper_native_v5_required()
            else min(100, max(4 * count, 10))
        )
        route_decision = self._select_route(hypothesis, target)
        selected_route = CandidateRoute(route_decision["selected_route"])
        route_task = self._task_id(plan, None, "Select one evidence", "route")
        _, route_decision_trace = recorder.invoke(
            task_id=route_task,
            tool_name="candidate_judge",
            operation="assign_route_from_ma1_evidence",
            inputs={
                "hypothesis_id": hypothesis.hypothesis_id,
                "target": target,
                "hypothesis_evidence": hypothesis.evidence,
                "policy_version": ROUTE_POLICY_VERSION,
            },
            function=lambda: route_decision,
        )

        try:
            if selected_route is CandidateRoute.REPURPOSING:
                discovery_task = self._task_id(
                    plan, CandidateRoute.REPURPOSING, "repurpos", "search"
                )
                discovery_outcome, discovery_trace = recorder.invoke(
                    task_id=discovery_task,
                    tool_name="repurposing_search",
                    operation="retrieve_candidates",
                    inputs={
                        "target": target,
                        "direction": direction,
                        "limit": count,
                        "requested_eligible_count": count,
                        "discovery_pool_limit": discovery_pool_limit,
                    },
                    function=lambda: self._retrieve_repurposing_outcome(
                        target,
                        direction=direction,
                        limit=count,
                    ),
                )
                seeds = discovery_outcome.seeds
            else:
                discovery_task = self._task_id(
                    plan, CandidateRoute.DE_NOVO, "de_novo", "generate"
                )
                discovery_outcome, discovery_trace = recorder.invoke(
                    task_id=discovery_task,
                    tool_name="de_novo_generator",
                    operation="generate_candidates",
                    inputs={
                        "target": target,
                        "direction": direction,
                        "limit": count,
                        "requested_eligible_count": count,
                        "discovery_pool_limit": discovery_pool_limit,
                    },
                    function=lambda: self._generate_de_novo_outcome(
                        target,
                        direction=direction,
                        limit=count,
                    ),
                )
                seeds = discovery_outcome.seeds
        except (LiveRepurposingError, ScientificToolError) as exc:
            return self._unavailable_packet(
                plan=plan,
                recorder=recorder,
                readiness=readiness,
                reason_code=self._unavailable_reason_code(exc),
                detail=str(exc),
                route_decision=route_decision,
                hypothesis_selection=hypothesis_selection,
            )

        if discovery_outcome.outcome == "no_candidates":
            return self._completed_no_candidate_packet(
                plan=plan,
                recorder=recorder,
                readiness=readiness,
                route_decision=route_decision,
                hypothesis_selection=hypothesis_selection,
                outcome=discovery_outcome,
            )

        paper_native_v5_active = (
            selected_route is CandidateRoute.REPURPOSING
            and self._paper_native_v5_required()
        )
        paper_native_v6_active = (
            selected_route is CandidateRoute.REPURPOSING
            and self._paper_native_v6_required()
        )
        paper_native_group_active = bool(
            paper_native_v5_active or paper_native_v6_active
        )
        paper_native_declared_group_ids: tuple[str, ...] = ()
        if paper_native_v5_active:
            try:
                paper_native_declared_group_ids = (
                    self._validate_paper_native_v5_group(seeds)
                )
            except PaperNativeV5GroupUnavailable as exc:
                return self._unavailable_packet(
                    plan=plan,
                    recorder=recorder,
                    readiness=readiness,
                    reason_code=self._unavailable_reason_code(exc),
                    detail=str(exc),
                    route_decision=route_decision,
                    hypothesis_selection=hypothesis_selection,
                )
        elif paper_native_v6_active:
            try:
                paper_native_declared_group_ids = (
                    self._validate_paper_native_v6_group(seeds)
                )

            except PaperNativeV6GroupUnavailable as exc:
                return self._unavailable_packet(
                    plan=plan,
                    recorder=recorder,
                    readiness=readiness,
                    reason_code=self._unavailable_reason_code(exc),
                    detail=str(exc),
                    route_decision=route_decision,
                    hypothesis_selection=hypothesis_selection,
                )

        paper_native_v6_control_rows: list[dict[str, Any]] = []
        if paper_native_v6_active:
            for pool_index, seed in enumerate(seeds):
                seed_metadata = self._seed_policy_metadata(seed)
                reserved = screen_reserved_control(
                    candidate_name=seed.name,
                    metadata=seed_metadata,
                    reserved_controls=plan.reserved_control_compounds,
                )
                paper_native_v6_control_rows.append({
                    "pool_index": pool_index,
                    "candidate_name": seed.name,
                    "candidate_identity": candidate_identity(
                        candidate_name=seed.name, metadata=seed_metadata
                    ),
                    "candidate_registry_metadata": {
                        "chembl_id": self._paper_native_v5_candidate_id(seed)
                    },
                    "reserved_control_screen": reserved.as_metadata(),
                    "reserved_excluded": reserved.excluded,
                    "control_only": reserved.matched,
                })

        candidates: list[CandidateCard] = []
        max_generation_iteration = 1
        if self.fixture_mode and len(seeds) != count:
            raise ValueError(
                f"{selected_route.value} fixture returned {len(seeds)} candidates; expected {count}"
            )
        if not self.fixture_mode and not 1 <= len(seeds) <= discovery_pool_limit:
            return self._unavailable_packet(
                plan=plan,
                recorder=recorder,
                readiness=readiness,
                reason_code="LIVE_CANDIDATE_COUNT_UNAVAILABLE",
                detail=(
                    f"Live {selected_route.value} retrieval returned {len(seeds)} "
                    "candidates for a bounded discovery pool range of "
                    f"1..{discovery_pool_limit}."
                ),
                route_decision=route_decision,
                hypothesis_selection=hypothesis_selection,
            )
        evaluated_identities: list[dict[str, Any]] = []
        projected_group_candidate_ids: list[str] = []
        declared_group_id_set = set(paper_native_declared_group_ids)
        eligible_count = 0
        for seed_index, seed in enumerate(seeds):
            complete_declared_group_projected = declared_group_id_set.issubset(
                set(projected_group_candidate_ids)
            )
            projected_is_exact_declared_prefix = (
                projected_group_candidate_ids
                == list(
                    paper_native_declared_group_ids[
                        : len(projected_group_candidate_ids)
                    ]
                )
            )
            if eligible_count >= count and (
                not paper_native_group_active
                or paper_native_v6_active
                and projected_is_exact_declared_prefix
                or paper_native_v5_active
                and complete_declared_group_projected
            ):
                break
            try:
                candidate, iterations = self._evaluate_seed(
                    plan=plan,
                    recorder=recorder,
                    hypothesis=hypothesis,
                    target=target,
                    route=selected_route,
                    seed=seed,
                    base_trace_ids=[
                        normalization_trace,
                        route_decision_trace,
                        discovery_trace,
                    ],
                    allow_refinement=(
                        selected_route is CandidateRoute.DE_NOVO
                        and self.fixture_mode
                    ),
                )
            except (LiveRepurposingError, ScientificToolError) as exc:
                return self._unavailable_packet(
                    plan=plan,
                    recorder=recorder,
                    readiness=readiness,
                    reason_code=self._unavailable_reason_code(exc),
                    detail=str(exc),
                    route_decision=route_decision,
                    hypothesis_selection=hypothesis_selection,
                    candidates=candidates,
                )
            candidates.append(candidate)
            seed_candidate_id = self._paper_native_v5_candidate_id(seed)
            if (
                paper_native_group_active
                and seed_candidate_id in declared_group_id_set
            ):
                projected_group_candidate_ids.append(str(seed_candidate_id))
            if candidate.status != "excluded":
                eligible_count += 1
            evaluated_identities.append(
                {
                    **candidate_identity(
                        candidate_name=seed.name,
                        metadata=self._seed_policy_metadata(seed),
                    ),
                    "pool_index": seed_index,
                    "candidate_id": candidate.candidate_id,
                    "outcome": (
                        "eligible"
                        if candidate.status != "excluded"
                        else "excluded"
                    ),
                    "reason_codes": list(candidate.exclusion_reasons),
                }
            )
            max_generation_iteration = max(max_generation_iteration, iterations)

        evaluated_count = len(candidates)
        quota_met = eligible_count >= count
        returned_pool_evaluated = evaluated_count >= len(seeds)
        discovery_provenance = discovery_outcome.provenance
        source_search_complete = not (
            discovery_provenance.get("partial_source_result") is True
            or bool(discovery_provenance.get("source_failures"))
            or discovery_provenance.get("source_search_complete") is False
            or discovery_provenance.get("search_completed") is False
        )
        pool_exhausted = returned_pool_evaluated and source_search_complete
        complete_declared_group_projected = bool(
            paper_native_declared_group_ids
            and set(projected_group_candidate_ids) == declared_group_id_set
        )
        projected_is_exact_declared_prefix = bool(
            paper_native_declared_group_ids
            and projected_group_candidate_ids
            == list(
                paper_native_declared_group_ids[
                    : len(projected_group_candidate_ids)
                ]
            )
        )
        projection_complete = bool(
            projected_is_exact_declared_prefix
            if paper_native_v6_active
            else complete_declared_group_projected
        )
        group_stop_condition_satisfied = bool(
            paper_native_group_active and quota_met and projection_complete
        )
        projection_attestation: dict[str, Any] | None = None
        if paper_native_group_active:
            projection_version = "v6" if paper_native_v6_active else "v5"
            projection_policy = (
                PAPER_NATIVE_V6_REQUIRED_POLICY
                if paper_native_v6_active
                else PAPER_NATIVE_V5_REQUIRED_POLICY
            )
            stop_rule = (
                _PAPER_NATIVE_V6_GROUP_PROJECTION_STOP_RULE
                if paper_native_v6_active
                else _PAPER_NATIVE_V5_GROUP_PROJECTION_STOP_RULE
            )
            projection_attestation = {
                "schema_version": (
                    f"paper_native_{projection_version}_group_projection_v1"
                ),
                "policy": projection_policy,
                "declared_group_candidate_ids": list(
                    paper_native_declared_group_ids
                ),
                "declared_group_candidate_ids_sha256": _fingerprint(
                    list(paper_native_declared_group_ids)
                ),
                "projected_group_candidate_ids": list(
                    projected_group_candidate_ids
                ),
                "projected_group_candidate_ids_sha256": _fingerprint(
                    projected_group_candidate_ids
                ),
                "projection_complete": projection_complete,
                "projection_scope": (
                    "exact_evaluated_fixed_pool_prefix"
                    if paper_native_v6_active
                    else "complete_declared_group"
                ),
                "projected_is_exact_declared_prefix": (
                    projected_is_exact_declared_prefix
                ),
                "complete_declared_group_projected": (
                    complete_declared_group_projected
                ),
                "unevaluated_declared_group_candidate_ids": list(
                    paper_native_declared_group_ids[
                        len(projected_group_candidate_ids) :
                    ]
                ),
                "unevaluated_declared_group_candidate_ids_sha256": _fingerprint(
                    list(
                        paper_native_declared_group_ids[
                            len(projected_group_candidate_ids) :
                        ]
                    )
                ),
                "eligible_quota_reached": quota_met,
                "requested_eligible_count": count,
                "eligible_returned_count": eligible_count,
                "evaluated_fixed_pool_prefix_count": evaluated_count,
                "stop_rule": stop_rule,
                "stop_rule_sha256": sha256(
                    stop_rule.encode("utf-8")
                ).hexdigest(),
                "stop_condition_satisfied": group_stop_condition_satisfied,
                "no_second_discovery": True,
                "no_backfill_beyond_fixed_pool": True,
                "downstream_evaluation_cap_enforced": (
                    paper_native_v6_active
                ),
                "fixture": False,
            }
            projection_attestation["projection_sha256"] = _fingerprint(
                projection_attestation
            )
            candidates = [
                candidate.model_copy(
                    update={
                        "metadata": {
                            **candidate.metadata,
                            f"paper_native_{projection_version}_group_projection": deepcopy(
                                projection_attestation
                            ),
                        }
                    },
                    deep=True,
                )
                for candidate in candidates
            ]
        unevaluated_reason = (
            (
                _PAPER_NATIVE_V6_GROUP_PROJECTION_STOP_REASON
                if paper_native_v6_active
                else _PAPER_NATIVE_V5_GROUP_PROJECTION_STOP_REASON
            )
            if group_stop_condition_satisfied
            else "ELIGIBLE_CANDIDATE_QUOTA_MET"
        )
        unevaluated_identities = [
            {
                **candidate_identity(
                    candidate_name=seed.name,
                    metadata=self._seed_policy_metadata(seed),
                ),
                "pool_index": index,
                "reason_codes": [unevaluated_reason],
            }
            for index, seed in enumerate(seeds[evaluated_count:], start=evaluated_count)
        ]
        paper_native_v6_control_attestation: dict[str, Any] | None = None
        if paper_native_v6_active:
            rows = [
                {**row, "downstream_evaluated": index < evaluated_count}
                for index, row in enumerate(paper_native_v6_control_rows)
            ]
            paper_native_v6_control_attestation = {
                "schema_version": "paper_native_v6_full_pool_control_screening/v1",
                "policy": PAPER_NATIVE_V6_REQUIRED_POLICY,
                "row_count": len(rows),
                "rows": rows,
                "fixture": False,
            }
            paper_native_v6_control_attestation["attestation_sha256"] = (
                _fingerprint(paper_native_v6_control_attestation)
            )
        candidate_backfill = {
            "requested_eligible_count": count,
            "discovery_pool_count": len(seeds),
            "discovery_pool_limit": discovery_pool_limit,
            "evaluated_count": evaluated_count,
            "eligible_count": eligible_count,
            "excluded_count": evaluated_count - eligible_count,
            "quota_met": quota_met,
            "pool_exhausted": pool_exhausted,
            "returned_pool_evaluated": returned_pool_evaluated,
            "source_search_complete": source_search_complete,
            "exhaustive_discovery": False,
            "evaluated_identities": evaluated_identities,
            "unevaluated_identities": unevaluated_identities,
            "stop_reason": (
                (
                    _PAPER_NATIVE_V6_GROUP_PROJECTION_STOP_REASON
                    if paper_native_v6_active
                    else _PAPER_NATIVE_V5_GROUP_PROJECTION_STOP_REASON
                )
                if group_stop_condition_satisfied
                else "ELIGIBLE_CANDIDATE_QUOTA_MET"
                if quota_met
                else "SOURCE_DISCOVERY_INCOMPLETE"
                if not source_search_complete
                else "FINITE_DISCOVERY_POOL_EXHAUSTED"
            ),
            "unbounded_search_used": False,
            **(
                {"paper_native_v6_full_pool_control_screening": deepcopy(
                    paper_native_v6_control_attestation
                )}
                if paper_native_v6_control_attestation is not None else {}
            ),
            "discovery_provenance": dict(discovery_outcome.provenance),
            **(
                {
                    (
                        "paper_native_v6_group_projection"
                        if paper_native_v6_active
                        else "paper_native_v5_group_projection"
                    ): deepcopy(
                        projection_attestation
                    )
                }
                if projection_attestation is not None
                else {}
            ),
        }
        candidate_yield = {
            "requested_limit": count,
            "returned_count": len(seeds),
            "partial": not quota_met,
            "fixture_fill_used": False,
            "backend_kind": self.backend_kind.value,
            "requested_eligible_count": count,
            "discovery_pool_count": len(seeds),
            "evaluated_count": evaluated_count,
            "eligible_count": eligible_count,
            "quota_met": quota_met,
            "pool_exhausted": pool_exhausted,
            "returned_pool_evaluated": returned_pool_evaluated,
            "source_search_complete": source_search_complete,
            "exhaustive_discovery": False,
            **(
                {"paper_native_v6_full_pool_control_screening": deepcopy(
                    paper_native_v6_control_attestation
                )}
                if paper_native_v6_control_attestation is not None else {}
            ),
            **(
                {
                    (
                        "paper_native_v6_group_projection"
                        if paper_native_v6_active
                        else "paper_native_v5_group_projection"
                    ): deepcopy(
                        projection_attestation
                    )
                }
                if projection_attestation is not None
                else {}
            ),
        }

        unavailable_codes = self._candidate_unavailable_codes(candidates)
        if not source_search_complete and not quota_met:
            unavailable_codes = self._dedupe([*unavailable_codes, "CANDIDATE_DISCOVERY_SOURCE_INCOMPLETE"])
        if unavailable_codes:
            return CandidatePacket(
                plan_id=plan.plan_id,
                candidates=self._attest_final_policy_eligibility(candidates),
                tool_calls=recorder.calls,
                generation_iterations=max_generation_iteration,
                backend_kind=self.backend_kind,
                execution_status=ScientificExecutionStatus.UNAVAILABLE,
                gateway_readiness=readiness,
                abstain_reason_codes=unavailable_codes,
                metadata={
                    "route_decision": route_decision,
                    "primary_hypothesis_selection": hypothesis_selection,
                    "scientific_execution": {
                        "status": "unavailable",
                        "reason_codes": unavailable_codes,
                        "missing_values_imputed": False,
                    },
                    "candidate_yield": candidate_yield,
                    "candidate_backfill": candidate_backfill,
                    "candidate_search_outcome": {
                        "status": "incomplete_search", "search_completed": False,
                        "reason_code": "CANDIDATE_DISCOVERY_SOURCE_INCOMPLETE",
                    } if not source_search_complete else {
                        "status": "candidate_evaluation_unavailable", "search_completed": True,
                        "reason_code": unavailable_codes[0],
                    },
                },
            )

        ranked = self._rank(plan, recorder, candidates)
        ranked = self._attest_prospective_falsification_status(
            ranked,
            hypothesis,
        )
        precompute_queries = self._restricted_precompute_queries(ranked)
        ranked, enrichment_queries = self._enrich_ranked_candidates(
            ranked,
            hypothesis=hypothesis,
            allowed_sources=allowed_sources,
        )
        ranked, calls = self._inject_fault(fault, ranked, recorder.calls)
        eligible_ranked = [item for item in ranked if item.status != "excluded"]
        search_outcome = (
            {
                "status": (
                    "candidates" if quota_met else "partial_candidates"
                ),
                "reason_code": (
                    None
                    if quota_met
                    else "ELIGIBLE_CANDIDATE_QUOTA_NOT_MET_POOL_EXHAUSTED"
                ),
                "search_completed": source_search_complete,
            }
            if eligible_ranked
            else {
                "status": "no_candidates",
                "reason_code": "NO_QUALIFIED_CANDIDATES",
                "search_completed": True,
            }
        )
        return CandidatePacket(
            plan_id=plan.plan_id,
            candidates=ranked,
            tool_calls=calls,
            generation_iterations=max_generation_iteration,
            backend_kind=self.backend_kind,
            execution_status=ScientificExecutionStatus.COMPLETED,
            gateway_readiness=readiness,
            metadata={
                "public_enrichment_queries": [
                    *precompute_queries,
                    *enrichment_queries,
                ],
                "route_decision": route_decision,
                "primary_hypothesis_selection": hypothesis_selection,
                "exclusion_summary": self._exclusion_summary(ranked),
                "candidate_yield": candidate_yield,
                "candidate_backfill": candidate_backfill,
                "candidate_search_outcome": search_outcome,
                "activity_comparability": self._activity_comparability_summary(
                    ranked
                ),
                "scientific_execution": {
                    "status": "completed",
                    "outcome": search_outcome["status"],
                    "reason_codes": (
                        []
                        if eligible_ranked
                        else ["NO_QUALIFIED_CANDIDATES"]
                    ),
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                },
            },
        )

    def rework(
        self,
        plan: PlanSpec,
        hypotheses: HypothesisPacket,
        previous: CandidatePacket,
        order: ReworkOrder,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> CandidatePacket:
        if self._repair_intent(order) == frozenset({"semantic_attestation"}):
            return self._rework_semantic_contracts(plan, hypotheses, previous, order)
        if previous.portfolio_enabled:
            return self._rework_portfolio(
                plan,
                hypotheses,
                previous,
                order,
                allowed_sources=allowed_sources,
            )
        return self._rework_single(
            plan,
            hypotheses,
            previous,
            order,
            allowed_sources=allowed_sources,
        )

    def _rework_semantic_contracts(self, plan, hypotheses, previous, order):
        """Copy-only repair: no rescreen, ranking, gateway, enrichment or model."""
        if order.target_stage != "ma2" or previous.plan_id != plan.plan_id:
            raise ValueError("Semantic rework stage or plan binding mismatch")
        affected = set(order.affected_candidate_ids) or {c.candidate_id for c in previous.candidates}
        if affected - {c.candidate_id for c in previous.candidates}:
            raise ValueError("Semantic rework references unknown candidates")
        cards = []
        for candidate in previous.candidates:
            if candidate.candidate_id not in affected:
                cards.append(candidate.model_copy(deep=True))
                continue
            hypothesis = bound_semantic_hypothesis(hypotheses, previous, candidate)
            if hypothesis is None:
                raise ScientificDataUnavailable("Semantic rework requires unambiguous persisted hypothesis binding")
            updated = self._attest_prospective_falsification_status([candidate], hypothesis)[0]
            updated.metadata["assay_variant_alignment"] = assay_variant_alignment(
                target_symbol=candidate.target.symbol,
                molecular_scope=hypothesis.target.metadata.get("molecular_hypothesis_scope"),
                hypothesis_texts=(hypothesis.mechanism, hypothesis.falsification_test),
                chembl_assay_context=candidate.metadata.get("chembl_assay_context"),
                expected_target_chembl_id=candidate.metadata.get("measured_target"))
            cards.append(updated)
        provenance = {"schema_version": "semantic_contract_copy_rework_v1",
            "order_id": order.order_id, "source_packet_id": previous.packet_id,
            "source_packet_sha256": sha256(json.dumps(previous.model_dump(mode="json"),
                ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "affected_candidate_ids": sorted(affected), "network_calls": 0,
            "model_calls": 0, "ranking_recomputed": False, "observations_created": False}
        return previous.model_copy(update={"packet_id": new_id("cp"), "generated_at": utc_now(),
            "candidates": cards,
            "metadata": {**previous.metadata, "semantic_contract_rework": provenance},
            "reused_artifact_ids": self._dedupe([*previous.reused_artifact_ids, previous.packet_id,
                *(c.candidate_id for c in previous.candidates), *(c.call_id for c in previous.tool_calls),
                *order.preserve_artifact_ids])}, deep=True)

    def _rework_portfolio(
        self,
        plan: PlanSpec,
        hypotheses: HypothesisPacket,
        previous: CandidatePacket,
        order: ReworkOrder,
        *,
        allowed_sources: Sequence[str] | None,
    ) -> CandidatePacket:
        """Repair only affected target/candidate scopes and reuse all others."""

        if order.target_stage != "ma2":
            raise ValueError(f"MA2 cannot execute rework targeted at {order.target_stage}")
        affected = set(order.affected_candidate_ids)
        cards_by_id = {item.candidate_id: item for item in previous.candidates}
        unknown = affected - set(cards_by_id)
        if unknown:
            raise ValueError(f"ReworkOrder references unknown candidates: {sorted(unknown)}")
        repair_intent = self._repair_intent(order)
        trace_only = "trace" in repair_intent
        if not affected and not trace_only:
            return self._run_portfolio(
                plan,
                hypotheses,
                fault=FaultType.NONE,
                allowed_sources=allowed_sources,
            )
        if trace_only and not affected:
            recorder = _TraceRecorder(
                self.gateway.source_version,
                self._repair_trace_task_ids(plan, previous),
                backend_kind=self.backend_kind,
                fixture=self.fixture_mode,
                before_operation=self.before_operation,
            )
            trace_task = self._task_id(plan, None, "Rank policy", "rank")
            _, repair_trace = recorder.invoke(
                task_id=trace_task,
                tool_name="candidate_judge",
                operation="repair_plan_trace",
                inputs={"order": order, "previous_packet_id": previous.packet_id},
                function=lambda: {
                    "valid_task_ids": sorted(task.task_id for task in plan.tasks),
                    "repaired": True,
                    "portfolio_scoped": True,
                },
            )
            return previous.model_copy(
                update={
                    "candidates": self._attest_final_policy_eligibility(
                        previous.candidates
                    ),
                    "tool_calls": recorder.calls,
                    "reused_artifact_ids": self._dedupe(
                        [
                            *previous.reused_artifact_ids,
                            *(item.candidate_id for item in previous.candidates),
                            *(item.call_id for item in previous.tool_calls),
                            *order.preserve_artifact_ids,
                        ]
                    ),
                    "metadata": {
                        **previous.metadata,
                        "last_portfolio_rework": {
                            "scope": "plan_trace_only",
                            "repair_trace_id": repair_trace,
                        },
                    },
                },
                deep=True,
            )

        affected_keys = {
            str(cards_by_id[candidate_id].metadata.get("portfolio_target_key"))
            for candidate_id in affected
        }
        summary_by_key = {item.target_key: item for item in previous.target_executions}
        hypothesis_by_id = {
            item.hypothesis_id: item for item in hypotheses.hypotheses
        }
        merged_cards = list(previous.candidates)
        merged_calls = list(previous.tool_calls)
        updated_summaries = list(previous.target_executions)
        generation_iterations = previous.generation_iterations
        target_packet_metadata: dict[str, Mapping[str, Any]] = {}
        previous_target_keys = {
            key
            for candidate in previous.candidates
            if isinstance(
                key := candidate.metadata.get("portfolio_target_key"), str
            )
            and key
        }
        if len(previous_target_keys) == 1:
            target_packet_metadata[next(iter(previous_target_keys))] = (
                previous.metadata
            )

        for key in sorted(affected_keys):
            summary = summary_by_key.get(key)
            if summary is None or summary.hypothesis_id not in hypothesis_by_id:
                raise ValueError(f"Portfolio rework has no hypothesis summary for {key}")
            hypothesis = hypothesis_by_id[summary.hypothesis_id]
            target_cards = [
                item
                for item in previous.candidates
                if item.metadata.get("portfolio_target_key") == key
            ]
            target_call_ids = set(summary.tool_call_ids)
            target_calls = [
                item for item in previous.tool_calls if item.call_id in target_call_ids
            ]
            sub_status = {
                "completed": ScientificExecutionStatus.COMPLETED,
                "partial": ScientificExecutionStatus.PARTIAL,
                "failed": ScientificExecutionStatus.FAILED,
            }.get(summary.status, ScientificExecutionStatus.UNAVAILABLE)
            sub_previous = CandidatePacket(
                plan_id=plan.plan_id,
                candidates=target_cards,
                tool_calls=target_calls,
                generation_iterations=previous.generation_iterations,
                metadata={
                    "route_decision": summary.route_decision,
                    "candidate_backfill": summary.candidate_backfill,
                    **(
                        {
                            "candidate_yield": deepcopy(candidate_yield)
                        }
                        if isinstance(
                            candidate_yield := target_packet_metadata.get(
                                key, {}
                            ).get("candidate_yield"),
                            Mapping,
                        )
                        else {}
                    ),
                    "primary_hypothesis_selection": {
                        "selected_hypothesis_id": hypothesis.hypothesis_id,
                        "selection_policy": "portfolio_target_scope",
                    },
                },
                backend_kind=previous.backend_kind,
                execution_status=sub_status,
                gateway_readiness=summary.readiness,
                abstain_reason_codes=(
                    list(summary.reason_codes)
                    if sub_status is not ScientificExecutionStatus.COMPLETED
                    else []
                ),
            )
            sub_order = order.model_copy(
                update={
                    "affected_candidate_ids": sorted(
                        affected.intersection(item.candidate_id for item in target_cards)
                    )
                },
                deep=True,
            )
            sub_hypotheses = hypotheses.model_copy(
                update={"hypotheses": [hypothesis]},
                deep=True,
            )
            repaired = self._rework_single(
                plan,
                sub_hypotheses,
                sub_previous,
                sub_order,
                allowed_sources=allowed_sources,
            )
            target_packet_metadata[key] = repaired.metadata
            generation_iterations = max(
                generation_iterations, repaired.generation_iterations
            )
            repaired_cards = [
                item.model_copy(
                    update={
                        "target_rank": item.rank,
                        "metadata": {
                            **item.metadata,
                            "portfolio_target_key": key,
                            "portfolio_hypothesis_id": hypothesis.hypothesis_id,
                            "portfolio_hypothesis_score": hypothesis.score,
                            "portfolio_route": summary.route.value if summary.route else None,
                        },
                    },
                    deep=True,
                )
                for item in repaired.candidates
            ]
            merged_cards = [
                item
                for item in merged_cards
                if item.metadata.get("portfolio_target_key") != key
            ] + repaired_cards
            merged_calls = [
                item for item in merged_calls if item.call_id not in target_call_ids
            ] + repaired.tool_calls
            eligible_cards = [item for item in repaired_cards if item.status != "excluded"]
            if repaired.execution_status is ScientificExecutionStatus.COMPLETED:
                replacement_status = (
                    "completed" if eligible_cards else "no_candidates"
                )
            elif repaired.execution_status is ScientificExecutionStatus.FAILED:
                replacement_status = "failed"
            elif (
                repaired.execution_status is ScientificExecutionStatus.PARTIAL
                and eligible_cards
            ):
                replacement_status = "partial"
            else:
                replacement_status = "unavailable"
            search_outcome = repaired.metadata.get("candidate_search_outcome", {})
            search_reason = (
                search_outcome.get("reason_code")
                if isinstance(search_outcome, Mapping)
                else None
            )
            if replacement_status == "completed":
                replacement_reasons: list[str] = []
                replacement_detail = None
            elif replacement_status == "no_candidates":
                replacement_reasons = self._dedupe(
                    [
                        str(search_reason or "NO_QUALIFIED_CANDIDATES"),
                        *(
                            reason
                            for item in repaired_cards
                            for reason in item.exclusion_reasons
                        ),
                    ]
                )
                replacement_detail = (
                    "Target search/evaluation completed without an eligible candidate."
                )
            else:
                replacement_reasons = self._dedupe(
                    [
                        *repaired.abstain_reason_codes,
                        *(
                            reason
                            for item in repaired_cards
                            for reason in item.exclusion_reasons
                            if reason
                        ),
                    ]
                ) or ["TARGET_SCIENTIFIC_EXECUTION_INCOMPLETE"]
                scientific_execution = repaired.metadata.get(
                    "scientific_execution", {}
                )
                replacement_detail = (
                    scientific_execution.get("detail")
                    if isinstance(scientific_execution, Mapping)
                    else None
                ) or "Target scientific execution did not complete."
            replacement = summary.model_copy(
                update={
                    "status": replacement_status,
                    "readiness": repaired.gateway_readiness,
                    "discovered_candidate_count": len(repaired_cards),
                    "evaluated_candidate_count": sum(
                        item.chemistry_execution is not None for item in repaired_cards
                    ),
                    "eligible_candidate_count": len(eligible_cards),
                    "excluded_candidate_count": len(repaired_cards) - len(eligible_cards),
                    "candidate_backfill": dict(
                        repaired.metadata.get(
                            "candidate_backfill", summary.candidate_backfill
                        )
                    ),
                    "candidate_ids": [item.candidate_id for item in repaired_cards],
                    "tool_call_ids": [item.call_id for item in repaired.tool_calls],
                    "reason_codes": replacement_reasons,
                    "detail": replacement_detail,
                },
                deep=True,
            )
            updated_summaries = [
                replacement if item.target_key == key else item
                for item in updated_summaries
            ]

        hypothesis_scores = {
            item.target_key: item.hypothesis_score for item in updated_summaries
        }
        ranked = self._attest_final_policy_eligibility(
            rank_portfolio(merged_cards, hypothesis_scores=hypothesis_scores)
        )
        unique_calls = list({item.call_id: item for item in merged_calls}.values())
        eligible_count = sum(item.execution_eligible for item in updated_summaries)
        completed_count = sum(item.status == "completed" for item in updated_summaries)
        no_candidate_count = sum(
            item.status == "no_candidates" for item in updated_summaries
        )
        concluded_count = completed_count + no_candidate_count
        incomplete = [
            item
            for item in updated_summaries
            if item.execution_eligible
            if item.status not in {"completed", "no_candidates", "duplicate"}
        ]
        if eligible_count and all(
            item.status in {"completed", "no_candidates", "duplicate"}
            for item in updated_summaries
            if item.execution_eligible
        ):
            status = ScientificExecutionStatus.COMPLETED
            reasons: list[str] = []
        elif concluded_count:
            status = ScientificExecutionStatus.PARTIAL
            reasons = self._dedupe(
                [
                    "PORTFOLIO_PARTIAL_TARGET_COVERAGE",
                    *(code for item in incomplete for code in item.reason_codes),
                ]
            )
        else:
            status = ScientificExecutionStatus.UNAVAILABLE
            reasons = self._dedupe(
                [
                    "PORTFOLIO_TARGETS_UNAVAILABLE",
                    *(code for item in incomplete for code in item.reason_codes),
                ]
            )
        untouched = [
            item.candidate_id
            for item in previous.candidates
            if item.metadata.get("portfolio_target_key") not in affected_keys
        ]
        untouched_calls = [
            item.call_id
            for item in previous.tool_calls
            if all(item.call_id not in set(summary_by_key[key].tool_call_ids) for key in affected_keys)
        ]
        unavailable_count = sum(
            item.status in {"partial", "unavailable", "failed", "ineligible", "duplicate"}
            for item in updated_summaries
        )
        paper_v5_group_metadata = self._portfolio_paper_v5_group_metadata(
            ranked, target_packet_metadata
        )
        prior_metadata_without_v5_group = {
            key: value
            for key, value in previous.metadata.items()
            if key not in {"candidate_backfill", "candidate_yield"}
        }
        return CandidatePacket(
            plan_id=plan.plan_id,
            candidates=ranked,
            tool_calls=unique_calls,
            reused_artifact_ids=self._dedupe(
                [
                    *previous.reused_artifact_ids,
                    *untouched,
                    *untouched_calls,
                    *order.preserve_artifact_ids,
                ]
            ),
            generation_iterations=generation_iterations,
            backend_kind=previous.backend_kind,
            execution_status=status,
            gateway_readiness=previous.gateway_readiness,
            abstain_reason_codes=reasons,
            portfolio_enabled=True,
            g0_approved_target_count=len(updated_summaries),
            eligible_target_count=eligible_count,
            attempted_target_count=sum(item.attempted for item in updated_summaries),
            completed_target_count=completed_count,
            unavailable_target_count=unavailable_count,
            target_executions=updated_summaries,
            metadata={
                **prior_metadata_without_v5_group,
                **paper_v5_group_metadata,
                "portfolio_complete": status is ScientificExecutionStatus.COMPLETED,
                "completed_target_count": completed_count,
                "concluded_target_count": concluded_count,
                "no_candidate_target_count": no_candidate_count,
                "unavailable_target_count": unavailable_count,
                "portfolio_has_eligible_candidate": any(
                    item.status != "excluded" for item in ranked
                ),
                "portfolio_ranking": ranking_attestation(),
                "activity_comparability": self._activity_comparability_summary(
                    ranked
                ),
                "exclusion_summary": self._exclusion_summary(ranked),
                "scientific_execution": {
                    "status": status.value,
                    "reason_codes": reasons,
                    "outcome": (
                        "candidate_portfolio"
                        if any(item.status != "excluded" for item in ranked)
                        else "no_qualified_candidates"
                    ),
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                },
                "last_portfolio_rework": {
                    "affected_target_keys": sorted(affected_keys),
                    "affected_candidate_ids": sorted(affected),
                    "unaffected_artifacts_reused": True,
                },
            },
        )

    def _rework_single(
        self,
        plan: PlanSpec,
        hypotheses: HypothesisPacket,
        previous: CandidatePacket,
        order: ReworkOrder,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> CandidatePacket:
        """Apply a G1 ReworkOrder while retaining unaffected computations."""

        if order.target_stage != "ma2":
            raise ValueError(f"MA2 cannot execute rework targeted at {order.target_stage}")
        if previous.plan_id != plan.plan_id:
            raise ValueError("previous CandidatePacket belongs to a different PlanSpec")

        hypothesis, _ = self._select_primary_hypothesis(plan, hypotheses)
        target = self.gateway.normalize_target(hypothesis.target)
        direction = hypothesis.intervention_direction
        affected = set(order.affected_candidate_ids)
        existing_ids = {candidate.candidate_id for candidate in previous.candidates}
        unknown = affected - existing_ids
        if unknown:
            raise ValueError(f"ReworkOrder references unknown candidates: {sorted(unknown)}")

        repair_intent = self._repair_intent(order)
        trace_only = "trace" in repair_intent
        if not affected and not trace_only:
            # Packet/route-level validation failures may not identify a single
            # candidate.  Treat them as an explicit full candidate-set rerun,
            # rather than turning a valid REVISE verdict into an internal
            # workflow failure.
            return self.run(
                plan,
                hypotheses,
                fault=FaultType.NONE,
                allowed_sources=allowed_sources,
            )

        repaired_calls = self._repair_trace_task_ids(plan, previous)
        recorder = _TraceRecorder(
            self.gateway.source_version,
            repaired_calls,
            backend_kind=self.backend_kind,
            fixture=self.fixture_mode,
            before_operation=self.before_operation,
        )
        repaired: list[CandidateCard] = []
        generation_iterations = previous.generation_iterations

        for candidate in previous.candidates:
            screened = self._rescreen_candidate(plan, recorder, candidate)
            if screened.status == "excluded" or candidate.candidate_id not in affected:
                repaired.append(screened)
                continue
            updated, iterations = self._repair_candidate(
                plan=plan,
                recorder=recorder,
                hypothesis=hypothesis,
                target=target,
                direction=direction,
                previous=screened,
                repair_intent=repair_intent,
            )
            repaired.append(updated)
            generation_iterations = max(generation_iterations, iterations)

        if trace_only:
            trace_task = self._task_id(plan, None, "Rank policy", "rank")
            _, repair_trace = recorder.invoke(
                task_id=trace_task,
                tool_name="candidate_judge",
                operation="repair_plan_trace",
                inputs={"order": order, "previous_packet_id": previous.packet_id},
                function=lambda: {
                    "valid_task_ids": sorted(task.task_id for task in plan.tasks),
                    "repaired": True,
                },
            )
            if affected:
                repaired = [
                    card.model_copy(
                        update={"tool_trace_ids": self._dedupe([*card.tool_trace_ids, repair_trace])}
                    )
                    if card.candidate_id in affected
                    else card
                    for card in repaired
                ]

        ranked = self._rank(plan, recorder, repaired)
        # Rework creates a new packet; never carry legacy semantic assertions
        # forward or rewrite the checkpointed source packet in place.
        ranked = [candidate.model_copy(update={"metadata": {
            **candidate.metadata,
            "assay_variant_alignment_hypothesis_id": hypothesis.hypothesis_id,
            "assay_variant_alignment": assay_variant_alignment(
                target_symbol=candidate.target.symbol,
                molecular_scope=hypothesis.target.metadata.get("molecular_hypothesis_scope"),
                hypothesis_texts=(hypothesis.mechanism, hypothesis.falsification_test),
                chembl_assay_context=candidate.metadata.get("chembl_assay_context"),
                expected_target_chembl_id=candidate.metadata.get("measured_target"),
            ),
        }}, deep=True) for candidate in ranked]
        ranked = self._attest_prospective_falsification_status(ranked, hypothesis)
        precompute_queries = self._restricted_precompute_queries(ranked)
        ranked, enrichment_queries = self._enrich_ranked_candidates(
            ranked,
            hypothesis=hypothesis,
            allowed_sources=allowed_sources,
        )
        unavailable_codes = self._candidate_unavailable_codes(ranked)
        eligible_ranked = [item for item in ranked if item.status != "excluded"]
        if unavailable_codes:
            execution_status = ScientificExecutionStatus.UNAVAILABLE
            abstain_reason_codes = unavailable_codes
            search_outcome = {
                "status": "unavailable",
                "reason_code": unavailable_codes[0],
                "search_completed": False,
            }
        else:
            execution_status = ScientificExecutionStatus.COMPLETED
            abstain_reason_codes = []
            search_outcome = (
                {
                    "status": "candidates",
                    "reason_code": None,
                    "search_completed": True,
                }
                if eligible_ranked
                else {
                    "status": "no_candidates",
                    "reason_code": "NO_QUALIFIED_CANDIDATES",
                    "search_completed": True,
                }
            )
        untouched = existing_ids - affected
        reused = self._dedupe(
            [
                *previous.reused_artifact_ids,
                *sorted(untouched),
                *(call.call_id for call in previous.tool_calls),
                *order.preserve_artifact_ids,
            ]
        )
        return CandidatePacket(
            plan_id=plan.plan_id,
            candidates=ranked,
            tool_calls=recorder.calls,
            reused_artifact_ids=reused,
            generation_iterations=generation_iterations,
            backend_kind=self.backend_kind,
            execution_status=execution_status,
            gateway_readiness=self._gateway_readiness(target),
            abstain_reason_codes=abstain_reason_codes,
            metadata={
                **{
                    key: _jsonable(value)
                    for key, value in previous.metadata.items()
                    if key
                    not in {
                        "public_enrichment_queries",
                        "exclusion_summary",
                        "candidate_search_outcome",
                        "scientific_execution",
                    }
                },
                "public_enrichment_queries": [
                    *precompute_queries,
                    *enrichment_queries,
                ],
                "exclusion_summary": self._exclusion_summary(ranked),
                "candidate_search_outcome": search_outcome,
                "activity_comparability": self._activity_comparability_summary(
                    ranked
                ),
                "scientific_execution": {
                    "status": execution_status.value,
                    "outcome": search_outcome["status"],
                    "reason_codes": abstain_reason_codes,
                    "fixture_fallback_used": False,
                    "missing_values_imputed": False,
                },
            },
        )

    def _screen_substructure(
        self,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        smiles: str,
        metadata: Mapping[str, Any],
        *,
        repair: bool = False,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Record the exact immutable policy check before molecular computation."""
        result, trace_id = recorder.invoke(
            task_id=self._task_id(plan, None, "Rank policy", "rank"),
            tool_name="candidate_judge",
            operation="rescreen_forbidden_substructure" if repair else "screen_forbidden_substructure",
            inputs={"smiles": smiles, "forbidden_substructures": plan.forbidden_substructures},
            function=lambda: screen_forbidden_substructures(smiles, plan.forbidden_substructures).as_metadata(),
        )
        updated = {
            **dict(metadata),
            "forbidden_substructure_screen": result,
            "forbidden_substructure_screens": [
                *list(metadata.get("forbidden_substructure_screens", [])),
                {"call_id": trace_id, "result": result},
            ],
        }
        if result["blocked"]:
            updated["policy_eligible"] = False
            updated["missing_values_imputed"] = False
        return updated, trace_id, result

    def _rescreen_candidate(
        self,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        candidate: CandidateCard,
    ) -> CandidateCard:
        """Re-apply immutable control/drug policy before repair computation."""

        reserved = screen_reserved_control(
            candidate_name=candidate.name,
            metadata=candidate.metadata,
            reserved_controls=plan.reserved_control_compounds,
        )
        screen = screen_forbidden_drug(
            candidate_name=candidate.name,
            metadata=candidate.metadata,
            forbidden_names=plan.forbidden_drugs,
            forbidden_ids=plan.forbidden_drug_ids,
            forbidden_classes=plan.forbidden_drug_classes,
        )
        screen_task = self._task_id(plan, None, "Rank policy", "rank")
        _, reserved_trace_id = recorder.invoke(
            task_id=screen_task,
            tool_name="candidate_judge",
            operation="rescreen_reserved_assay_control",
            inputs={
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.name,
                "candidate_metadata": candidate.metadata,
                "reserved_control_compounds": plan.reserved_control_compounds,
            },
            function=reserved.as_metadata,
        )
        _, forbidden_trace_id = recorder.invoke(
            task_id=screen_task,
            tool_name="candidate_judge",
            operation="rescreen_forbidden_drug",
            inputs={
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.name,
                "candidate_metadata": candidate.metadata,
            },
            function=screen.as_metadata,
        )
        metadata = {
            **candidate.metadata,
            "reserved_control_screen": reserved.as_metadata(),
            "control_only": reserved.matched,
            "forbidden_drug_screen": screen.as_metadata(),
            "policy_eligible": not reserved.excluded and not screen.excluded,
        }
        trace_ids = self._dedupe(
            [
                *candidate.tool_trace_ids,
                reserved_trace_id,
                forbidden_trace_id,
            ]
        )
        chemical_screen = None
        if plan.forbidden_substructures:
            metadata, chemical_trace, chemical_screen = self._screen_substructure(
                plan, recorder, candidate.smiles, metadata, repair=True,
            )
            trace_ids = self._dedupe([*trace_ids, chemical_trace])
        chemical_blocked = chemical_screen is not None and chemical_screen["blocked"]
        if not reserved.excluded and not screen.excluded and not chemical_blocked:
            return candidate.model_copy(
                update={
                    "metadata": metadata,
                    "tool_trace_ids": trace_ids,
                },
                deep=True,
            )
        flags = set(candidate.safety_flags)
        if reserved.matched:
            flags.add("reserved_assay_control")
        if reserved.identity_conflict:
            flags.add("reserved_control_identity_conflict")
        if screen.excluded:
            flags.add("forbidden_drug")
        if screen.prohibited_statuses:
            flags.add("prohibited_regulatory_status")
        if chemical_blocked:
            flags.add("forbidden_substructure" if chemical_screen["matched"] else
                      "invalid_structure" if chemical_screen["status"] == "invalid_structure" else "chemical_policy_check_unavailable")
        return candidate.model_copy(
            update={
                "metadata": metadata,
                "tool_trace_ids": trace_ids,
                "safety_flags": sorted(flags),
                "status": "excluded",
                "score": 0.0,
                "rank": None,
                "target_rank": None,
                "portfolio_rank": None,
                "portfolio_score": None,
                "exclusion_reasons": self._dedupe(
                    [*candidate.exclusion_reasons, *reserved.reason_codes, *screen.reason_codes,
                     *(chemical_screen["reason_codes"] if chemical_blocked else [])]
                ),
            },
            deep=True,
        )

    def _enrich_ranked_candidates(
        self,
        candidates: Sequence[CandidateCard],
        *,
        hypothesis: Hypothesis,
        allowed_sources: Sequence[str] | None = None,
    ) -> tuple[list[CandidateCard], list[dict[str, Any]]]:
        """Attach public DB records to a bounded number of ranked candidates.

        Retrieval does not alter scores, ranks, docking, ADMET, or activity
        values. Source failures are non-fatal and successful records retain
        their own query/cache provenance.
        """

        if (
            self.evidence_enricher is None
            and self.restricted_evidence_enricher is None
        ) or self.enrichment_per_route == 0:
            return [candidate.model_copy(deep=True) for candidate in candidates], []

        from drugpilot.tools.public_chemistry import ChemistryLookup

        enriched: list[CandidateCard] = []
        source_queries: list[dict[str, Any]] = []
        recorded_target_context: set[tuple[str, str]] = set()
        route_counts = {
            CandidateRoute.REPURPOSING: 0,
            CandidateRoute.DE_NOVO: 0,
        }
        for candidate in candidates:
            if (
                candidate.status == "excluded"
                or route_counts[candidate.route] >= self.enrichment_per_route
            ):
                enriched.append(candidate.model_copy(deep=True))
                continue

            route_counts[candidate.route] += 1
            # The dependency-free chemistry validator uses a MOCK-prefixed
            # placeholder when RDKit cannot calculate a real InChIKey. Never
            # send that synthetic identifier to an external registry; fall
            # back to the canonical SMILES query instead.
            external_inchikey = candidate.inchikey
            if external_inchikey and external_inchikey.upper().startswith("MOCK"):
                external_inchikey = None
            request = ChemistryLookup(
                candidate_name=candidate.name,
                smiles=candidate.smiles,
                inchikey=external_inchikey,
                uniprot_id=candidate.target.uniprot_id,
                limit_per_source=1,
            )
            results: list[Any] = []
            if self.evidence_enricher is not None:
                try:
                    results.append(
                        self.evidence_enricher.enrich(
                            request,
                            allowed_sources=allowed_sources,
                        )
                    )
                except Exception as exc:
                    source_queries.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "candidate_name": candidate.name,
                            "route": candidate.route.value,
                            "source": "public_chemistry_enricher",
                            "status": "error",
                            "queried": True,
                            "record_count": 0,
                            "error_type": type(exc).__name__,
                        }
                    )
            if self.restricted_evidence_enricher is not None:
                try:
                    restricted = self.restricted_evidence_enricher.enrich_candidate(
                        candidate,
                        hypothesis,
                        allowed_sources=allowed_sources,
                    )
                    results.append(restricted)
                    for decision in getattr(restricted, "routing", []):
                        source_queries.append(
                            {
                                "candidate_id": candidate.candidate_id,
                                "candidate_name": candidate.name,
                                "route": candidate.route.value,
                                "record_type": "source_routing_decision",
                                **_jsonable(decision),
                            }
                        )
                except Exception as exc:
                    source_queries.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "candidate_name": candidate.name,
                            "route": candidate.route.value,
                            "source": "restricted_candidate_enricher",
                            "status": "error",
                            "queried": False,
                            "record_count": 0,
                            "error_type": type(exc).__name__,
                        }
                    )

            evidence_by_id = {
                item.evidence_id: item.model_copy(deep=True)
                for item in candidate.evidence
            }
            for result in results:
                for summary in getattr(result, "sources", []):
                    payload = (
                        summary.model_dump(mode="json")
                        if hasattr(summary, "model_dump")
                        else _jsonable(summary)
                    )
                    source_queries.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "candidate_name": candidate.name,
                            "route": candidate.route.value,
                            **payload,
                        }
                    )
                candidate_records = list(getattr(result, "evidence", []))
                target_records = list(getattr(result, "target_context", []))
                for item in candidate_records:
                    linked = candidate_linked_evidence(candidate, item)
                    if linked is not None:
                        evidence_by_id[linked.evidence_id] = linked
                        continue
                    target_records.append(item)
                for item in target_records:
                    decision = attest_candidate_evidence(candidate, item)
                    context_key = (
                        self._portfolio_target_key(candidate.target),
                        item.evidence_id,
                    )
                    if context_key in recorded_target_context:
                        continue
                    recorded_target_context.add(context_key)
                    source_queries.append(
                        {
                            "record_type": "target_context_evidence",
                            "target_key": context_key[0],
                            "target_symbol": candidate.target.symbol,
                            "source": item.source,
                            "external_id": item.external_id,
                            "evidence_id": item.evidence_id,
                            "title": item.title,
                            "supports": item.supports,
                            "identity_attestation": decision.as_metadata(),
                            "metadata": _jsonable(dict(item.metadata)),
                        }
                    )
            enriched.append(
                candidate.model_copy(
                    update={"evidence": list(evidence_by_id.values())},
                    deep=True,
                )
            )
        return enriched, source_queries

    @staticmethod
    def _restricted_precompute_queries(
        candidates: Sequence[CandidateCard],
    ) -> list[dict[str, Any]]:
        """Flatten pre-compute safety provenance into the packet-level trace."""

        queries: list[dict[str, Any]] = []
        for candidate in candidates:
            screen = candidate.metadata.get("restricted_safety_screen")
            if not isinstance(screen, Mapping):
                continue
            for routing in screen.get("routing", []):
                if isinstance(routing, Mapping):
                    queries.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "candidate_name": candidate.name,
                            "route": candidate.route.value,
                            "phase": "precompute_safety_screen",
                            "record_type": "source_routing_decision",
                            **_jsonable(dict(routing)),
                        }
                    )
            for summary in screen.get("sources", []):
                if isinstance(summary, Mapping):
                    queries.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "candidate_name": candidate.name,
                            "route": candidate.route.value,
                            "phase": "precompute_safety_screen",
                            **_jsonable(dict(summary)),
                        }
                    )
        return queries

    @staticmethod
    def _restricted_safety_disposition(result: Any) -> dict[str, Any]:
        """Interpret only pre-compute safety operations from typed adapters.

        Empty contraindication results are deliberately not labelled ``safe``.
        They mean only that the bounded query returned no matching record.
        """

        summaries = [
            summary
            for summary in getattr(result, "sources", [])
            if getattr(summary, "operation", None)
            in {"identity_regulatory", "contraindication"}
        ]
        evidence = [
            item
            for item in getattr(result, "evidence", [])
            if item.metadata.get("restricted_operation")
            in {"identity_regulatory", "contraindication"}
        ]
        routing = [
            item
            for item in getattr(result, "routing", [])
            if isinstance(item, Mapping)
            and item.get("operation")
            in {"identity_regulatory", "contraindication"}
        ]

        unresolved = [
            summary
            for summary in summaries
            if getattr(summary, "status", None) == "error"
        ]
        statuses: set[str] = set()
        withdrawn_sources: set[str] = set()
        for item in evidence:
            metadata = item.metadata
            if metadata.get("withdrawn") is True:
                withdrawn_sources.add(item.source)
            for key in (
                "regulatory_status",
                "approval_status",
                "marketing_status",
                "status",
                "regulatory_groups",
            ):
                raw = metadata.get(key)
                values = raw if isinstance(raw, (list, tuple, set)) else [raw]
                statuses.update(
                    str(value).strip()
                    for value in values
                    if value is not None and str(value).strip()
                )
        prohibited_statuses = sorted(
            {
                status
                for status in statuses
                if status.casefold() in _PROHIBITED_EXTERNAL_STATUSES
            },
            key=str.casefold,
        )
        contraindications = [
            item
            for item in evidence
            if item.metadata.get("restricted_operation") == "contraindication"
            and (
                item.metadata.get("contraindication_present") is not False
                or item.supports == "counter"
            )
        ]

        reasons: list[str] = []
        if unresolved:
            reasons.extend(
                "restricted_safety_lookup_unresolved:"
                f"{summary.source}:{summary.operation}:{summary.error_type or 'unknown'}"
                for summary in unresolved
            )
            disposition = "unresolved"
        else:
            reasons.extend(
                f"prohibited_regulatory_status:withdrawn:{source}"
                for source in sorted(withdrawn_sources)
            )
            reasons.extend(
                f"prohibited_regulatory_status:{status}"
                for status in prohibited_statuses
            )
            reasons.extend(
                f"contraindication_for_research_disease:{item.source}:{item.external_id}"
                for item in contraindications
            )
            if reasons:
                disposition = "excluded"
            elif summaries and all(
                getattr(summary, "status", None) == "not_configured"
                for summary in summaries
            ):
                disposition = "not_configured"
            else:
                disposition = "eligible"

        configured = [
            summary
            for summary in summaries
            if getattr(summary, "status", None) != "not_configured"
        ]
        coverage = (
            "unconfigured"
            if disposition == "not_configured"
            else "failed"
            if disposition == "unresolved"
            else "full"
            if summaries and len(configured) == len(summaries)
            else "partial"
        )
        limitations = sorted(
            {
                str(note)
                for summary in summaries
                for note in getattr(summary, "limitations", ())
            }
        )
        limitations.append(
            "No matching contraindication record is not proof of clinical safety."
        )
        return {
            "screened": True,
            "disposition": disposition,
            "coverage": coverage,
            "reason_codes": list(dict.fromkeys(reasons)),
            "sources": [summary.model_dump(mode="json") for summary in summaries],
            "routing": [_jsonable(dict(item)) for item in routing],
            "evidence_ids": [item.evidence_id for item in evidence],
            "prohibited_statuses": prohibited_statuses,
            "withdrawn_sources": sorted(withdrawn_sources),
            "contraindication_evidence_ids": [
                item.evidence_id for item in contraindications
            ],
            "absence_is_not_negative_evidence": True,
            "limitations": list(dict.fromkeys(limitations)),
        }

    @staticmethod
    def _legacy_select_route(
        hypothesis: Hypothesis,
        target: TargetCandidate,
    ) -> dict[str, Any]:
        """Choose one route from persisted MA1 evidence, never LLM intuition.

        Open Targets scores are used only as operational routing signals.  The
        returned confidence describes completeness/consistency of the routing
        evidence; it is not a probability that the biological hypothesis is
        correct or that a medicine will work.
        """

        metadata = target.metadata if isinstance(target.metadata, Mapping) else {}
        raw_summary = metadata.get("evidence_summary")
        summary = raw_summary if isinstance(raw_summary, Mapping) else {}
        known_drug_score = MA2Agent._optional_unit_score(
            summary.get("known_drug_score")
        )
        association_score = MA2Agent._optional_unit_score(
            metadata.get("association_score")
        )
        if association_score is None:
            association_score = MA2Agent._optional_unit_score(
                summary.get("overall_association_score")
            )
        known_drug_count = metadata.get("known_drug_count")
        if not isinstance(known_drug_count, int) or known_drug_count < 0:
            known_drug_count = None

        # MA1 currently uses ``target.tdl`` as an operational supply-category
        # label for some records.  It is authoritative TDL evidence only when
        # its own provenance explicitly names a target-development database.
        operational_category = (target.tdl or "unknown").strip()
        tdl_source = str(metadata.get("tdl_source") or "").strip()
        verified_tdl = tdl_source.casefold() in {
            "pharos",
            "pharos api",
            "target central resource database",
        }
        tdl_key = operational_category.casefold()
        well_characterized_tdl = verified_tdl and tdl_key in {"tclin", "tchem"}
        underexplored_tdl = verified_tdl and tdl_key in {"tbio", "tdark"}
        support_evidence = [
            item for item in hypothesis.evidence if item.supports == "support"
        ]
        evidence_sources = sorted({item.source for item in hypothesis.evidence})
        explicit_known_drug_signal = bool(
            well_characterized_tdl
            or (known_drug_count is not None and known_drug_count > 0)
            or (
                known_drug_score is not None
                and known_drug_score >= KNOWN_DRUG_REPURPOSING_THRESHOLD
            )
        )
        evidence_supported = bool(support_evidence) and (
            association_score is None
            or association_score >= MIN_DE_NOVO_ASSOCIATION_SCORE
        )
        known_drug_source_complete = bool(summary)
        explicit_underexplored_signal = bool(
            underexplored_tdl
            or (
                known_drug_source_complete
                and (
                    known_drug_count == 0
                    or known_drug_score is None
                    or known_drug_score <= UNDEREXPLORED_KNOWN_DRUG_MAX
                )
            )
        )

        fallback_used = False
        fallback_reason: str | None = None
        if explicit_known_drug_signal:
            route = CandidateRoute.REPURPOSING
            status = "well_characterized_or_drug_evidenced"
            rationale = (
                "Disease-linked known-drug evidence or a source-verified Tclin/Tchem classification "
                "meets the fixed repurposing threshold; de novo generation is suppressed."
            )
            confidence = _clamp(
                0.74 + 0.20 * (known_drug_score or 0.0) + (0.05 if well_characterized_tdl else 0.0),
                maximum=0.98,
            )
        elif explicit_underexplored_signal and evidence_supported:
            route = CandidateRoute.DE_NOVO
            status = "underexplored_but_evidence_supported"
            rationale = (
                "The target has biological support but no material disease-linked "
                "known-drug signal; repurposing retrieval is suppressed and de novo "
                "design is selected."
            )
            confidence = _clamp(
                0.67
                + 0.18 * (association_score or 0.0)
                + (0.08 if underexplored_tdl else 0.0),
                maximum=0.95,
            )
        else:
            # Missing/ambiguous target metadata cannot justify running both
            # portfolios.  The persisted MA1 supply track is the conservative,
            # reproducible tie-break: literature -> retrieval, Tbio -> design.
            fallback_used = True
            route = (
                CandidateRoute.DE_NOVO
                if hypothesis.track.value == "tbio"
                else CandidateRoute.REPURPOSING
            )
            status = "indeterminate_target_knownness"
            fallback_reason = (
                "Known-drug metadata was absent, incomplete, or fell in the policy grey zone; "
                f"the persisted MA1 {hypothesis.track.value} track selected {route.value}."
            )
            rationale = fallback_reason
            confidence = 0.45

        factors = [
            {
                "name": "known_drug_score",
                "value": known_drug_score,
                "source": "MA1.target.metadata.evidence_summary",
                "interpretation": "Open Targets routing heuristic; not an efficacy probability",
            },
            {
                "name": "known_drug_count",
                "value": known_drug_count,
                "source": "MA1.target.metadata",
                "interpretation": "A positive disease-linked count favors repurposing",
            },
            {
                "name": "ma1_operational_target_category",
                "value": operational_category,
                "source": "MA1.target.tdl",
                "interpretation": (
                    "Not treated as authoritative TDL unless tdl_source names Pharos/TCRD; "
                    f"verified={verified_tdl}"
                ),
            },
            {
                "name": "association_score",
                "value": association_score,
                "source": "MA1.target.metadata",
                "interpretation": "Minimum support check for an underexplored de novo target",
            },
            {
                "name": "support_evidence_count",
                "value": len(support_evidence),
                "source": "MA1.Hypothesis.evidence",
                "interpretation": "At least one linked support record is required",
            },
            {
                "name": "ma1_track",
                "value": hypothesis.track.value,
                "source": "MA1.Hypothesis.track",
                "interpretation": "Used only as a deterministic fallback tie-break",
            },
        ]
        return {
            "policy_version": ROUTE_POLICY_VERSION,
            "hypothesis_id": hypothesis.hypothesis_id,
            "target_symbol": target.symbol,
            "selected_route": route.value,
            "selected_routes": [route.value],
            "decision": route.value.upper(),
            "rationale": rationale,
            "factors": factors,
            "known_drug_count": known_drug_count,
            "known_drug_score": known_drug_score,
            "target_novelty": status,
            "target_status": status,
            "evidence_sources": evidence_sources,
            "thresholds": {
                "known_drug_repurposing_min": KNOWN_DRUG_REPURPOSING_THRESHOLD,
                "underexplored_known_drug_max": UNDEREXPLORED_KNOWN_DRUG_MAX,
                "de_novo_association_min": MIN_DE_NOVO_ASSOCIATION_SCORE,
            },
            "confidence": round(confidence, 3),
            "confidence_interpretation": (
                "routing evidence completeness; not biological or clinical success probability"
            ),
            "fallback": {
                "used": fallback_used,
                "reason": fallback_reason,
                "requires_evidence_request": fallback_used,
            },
            "policy_eligible": True,
            "attestation": "exactly_one_route_selected_from_persisted_ma1_evidence",
        }

    @staticmethod
    def _optional_unit_score(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if 0.0 <= parsed <= 1.0 else None

    @staticmethod
    def _select_route(
        hypothesis: Hypothesis,
        target: TargetCandidate,
    ) -> dict[str, Any]:
        """Delegate route selection to the shared pure evidence policy."""

        return select_route_from_evidence(hypothesis, target)

    @staticmethod
    def _exclusion_summary(candidates: Sequence[CandidateCard]) -> dict[str, Any]:
        excluded = [card for card in candidates if card.status == "excluded"]
        forbidden = [
            card
            for card in excluded
            if card.metadata.get("forbidden_drug_screen", {}).get("matched")
        ]
        reserved_controls = [
            card
            for card in excluded
            if card.metadata.get("reserved_control_screen", {}).get("matched")
        ]
        reserved_identity_conflicts = [
            card
            for card in excluded
            if card.metadata.get("reserved_control_screen", {}).get(
                "identity_conflict"
            )
        ]
        reason_counts: dict[str, int] = {}
        for card in excluded:
            for reason in card.exclusion_reasons:
                category = reason.split(":", 1)[0]
                reason_counts[category] = reason_counts.get(category, 0) + 1
        return {
            "screened_candidate_count": len(candidates),
            "eligible_candidate_count": len(candidates) - len(excluded),
            "excluded_candidate_count": len(excluded),
            "forbidden_or_withdrawn_count": len(forbidden),
            "reserved_assay_control_count": len(reserved_controls),
            "reserved_control_identity_conflict_count": len(
                reserved_identity_conflicts
            ),
            "excluded_candidate_ids": [card.candidate_id for card in excluded],
            "reason_counts": dict(sorted(reason_counts.items())),
            "attestation": "excluded_candidates_are_unranked_and_not_policy_eligible",
        }

    def _paper_native_v5_required(self) -> bool:
        return (
            getattr(self.gateway, "repurposing_evidence_policy", None)
            == PAPER_NATIVE_V5_REQUIRED_POLICY
        )

    def _paper_native_v6_required(self) -> bool:
        return (
            getattr(self.gateway, "repurposing_evidence_policy", None)
            == PAPER_NATIVE_V6_REQUIRED_POLICY
        )

    @staticmethod
    def _paper_native_v5_candidate_id(seed: CandidateSeed) -> str | None:
        values = {
            str(seed.metadata.get(key) or "").strip().upper()
            for key in ("chembl_id", "drug_id")
            if str(seed.metadata.get(key) or "").strip()
        }
        if len(values) != 1:
            return None
        candidate_id = next(iter(values))
        return candidate_id if re.fullmatch(r"CHEMBL\d+", candidate_id) else None

    @classmethod
    def _paper_native_v5_envelope(
        cls,
        seed: CandidateSeed,
    ) -> Mapping[str, Any]:
        candidate_id = cls._paper_native_v5_candidate_id(seed)
        holders = (seed.metadata, *(item.metadata for item in seed.evidence))
        copies = [holder.get("primary_abstract_paper_native_v5") for holder in holders]
        if candidate_id is None or not copies or any(copy is None for copy in copies):
            raise PaperNativeV5GroupUnavailable(
                "A v5 fixed-pool seed or evidence record is missing its bound envelope."
            )
        first = copies[0]
        if not isinstance(first, Mapping) or any(copy != first for copy in copies[1:]):
            raise PaperNativeV5GroupUnavailable(
                "Paper-native-v5 envelope copies are not byte-equivalent."
            )
        if (
            first.get("schema_version")
            != "candidate_primary_evidence_paper_native_v5"
            or first.get("source_contract") != "primary_abstract_paper_native_v5"
            or first.get("molecule_chembl_id") != candidate_id
            or first.get("verification_status") not in {"verified", "not_verified"}
            or first.get("fixture") is not False
            or first.get("missing_values_imputed") is not False
            or first.get("cross_source_numeric_equality_required") is not False
            or first.get("pubmed_value_used_for_activity_selection") is not False
            or first.get("pubmed_value_used_for_candidate_rank") is not False
            or first.get("pubmed_value_used_for_candidate_ordering") is not False
        ):
            raise PaperNativeV5GroupUnavailable(
                "A paper-native-v5 envelope is malformed or identity-contradictory."
            )
        return first

    @classmethod
    def _validate_paper_native_v5_group(
        cls,
        seeds: Sequence[CandidateSeed],
    ) -> tuple[str, ...]:
        """Require the complete verified group before any candidate computation."""

        if not 1 <= len(seeds) <= 12:
            raise PaperNativeV5GroupUnavailable(
                "Paper-native-v5 requires one finite fixed pool of at most twelve seeds."
            )
        candidate_ids = [cls._paper_native_v5_candidate_id(seed) for seed in seeds]
        if any(candidate_id is None for candidate_id in candidate_ids):
            raise PaperNativeV5GroupUnavailable(
                "Every paper-native-v5 seed requires one authoritative ChEMBL ID."
            )
        stable_ids = [str(candidate_id) for candidate_id in candidate_ids]
        if len(set(stable_ids)) != len(stable_ids):
            raise PaperNativeV5GroupUnavailable(
                "The paper-native-v5 fixed pool contains duplicate identities."
            )

        verified: list[tuple[str, Mapping[str, Any]]] = []
        for candidate_id, seed in zip(stable_ids, seeds, strict=True):
            envelope = cls._paper_native_v5_envelope(seed)
            if envelope.get("verification_status") == "verified":
                verified.append((candidate_id, envelope))
        if len(verified) < 3:
            raise PaperNativeV5GroupUnavailable(
                "No complete three-or-more-member paper-native-v5 group was returned."
            )

        group_contract: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
        verified_ids = [candidate_id for candidate_id, _ in verified]
        seed_by_id = dict(zip(stable_ids, seeds, strict=True))
        for candidate_id, envelope in verified:
            group = envelope.get("group")
            sentence_ids = group.get("candidate_ids") if isinstance(group, Mapping) else None
            member_ids = envelope.get("group_member_candidate_ids")
            group_sha256 = group.get("group_sha256") if isinstance(group, Mapping) else None
            support = envelope.get("pre_pubmed_chembl_support")
            pairs = group.get("pairs") if isinstance(group, Mapping) else None
            pair_index = envelope.get("selected_pair_index")
            selected_pair = envelope.get("selected_pair")
            pair_sha256 = envelope.get("selected_pair_evidence_sha256")
            if (
                not isinstance(group_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", group_sha256) is None
                or envelope.get("selected_group_sha256") != group_sha256
                or not isinstance(sentence_ids, list)
                or len(sentence_ids) < 3
                or len(set(sentence_ids)) != len(sentence_ids)
                or not isinstance(member_ids, list)
                or len(member_ids) < 3
                or len(set(member_ids)) != len(member_ids)
                or set(sentence_ids) != set(member_ids)
                or candidate_id not in member_ids
                or not isinstance(support, Mapping)
                or support.get("molecule_chembl_id") != candidate_id
                or not isinstance(support.get("activity_id"), (str, int))
                or not isinstance(support.get("support_attestation_sha256"), str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(support.get("support_attestation_sha256")),
                )
                is None
                or isinstance(pair_index, bool)
                or not isinstance(pair_index, int)
                or not isinstance(pairs, list)
                or not 0 <= pair_index < len(pairs)
                or not isinstance(selected_pair, Mapping)
                or pairs[pair_index] != selected_pair
                or not isinstance(pair_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", pair_sha256) is None
                or selected_pair.get("evidence_sha256") != pair_sha256
                or not cls._paper_native_v5_activity_matches_support(
                    seed_by_id[candidate_id],
                    support,
                )
            ):
                raise PaperNativeV5GroupUnavailable(
                    "A verified v5 member lacks its complete group or ChEMBL support."
                )
            current = (group_sha256, tuple(sentence_ids), tuple(member_ids))
            if group_contract is None:
                group_contract = current
            elif current != group_contract:
                raise PaperNativeV5GroupUnavailable(
                    "Verified v5 members carry contradictory group copies."
                )

        assert group_contract is not None
        declared_member_ids = list(group_contract[2])
        if (
            set(declared_member_ids) != set(verified_ids)
            or any(candidate_id not in stable_ids for candidate_id in declared_member_ids)
        ):
            raise PaperNativeV5GroupUnavailable(
                "A declared paper-native-v5 group member is missing from the returned pool."
            )
        return tuple(declared_member_ids)

    @staticmethod
    def _paper_native_v5_activity_matches_support(
        seed: CandidateSeed,
        support: Mapping[str, Any],
    ) -> bool:
        if len(seed.activities) != 1:
            return False
        measurement = seed.activities[0]
        typed = support.get("typed_measurement_tuple")
        if not isinstance(typed, Mapping):
            return False
        if dict(typed) != {
            "endpoint": measurement.endpoint,
            "relation": measurement.relation,
            "value": measurement.value,
            "unit": measurement.unit,
            "assay_type": measurement.assay_type,
            "source_evidence_id": measurement.source_evidence_id,
        }:
            return False
        return bool(
            measurement.source_evidence_id
            and support.get("source_evidence_id")
            == measurement.source_evidence_id
            and measurement.source_evidence_id
            in {item.evidence_id for item in seed.evidence}
        )

    @classmethod
    def _paper_native_v6_envelope(
        cls,
        seed: CandidateSeed,
    ) -> Mapping[str, Any]:
        candidate_id = cls._paper_native_v5_candidate_id(seed)
        holders = (seed.metadata, *(item.metadata for item in seed.evidence))
        copies = [holder.get("primary_abstract_paper_native_v6") for holder in holders]
        if candidate_id is None or not copies or any(copy is None for copy in copies):
            raise PaperNativeV6GroupUnavailable(
                "A v6 fixed-pool seed or evidence record is missing its bound envelope."
            )
        first = copies[0]
        if not isinstance(first, Mapping) or any(copy != first for copy in copies[1:]):
            raise PaperNativeV6GroupUnavailable(
                "Paper-native-v6 envelope copies are not byte-equivalent."
            )
        body = dict(first)
        attestation = body.pop("attestation_sha256", None)
        if (
            first.get("schema_version")
            != "candidate_primary_evidence_paper_native_v6"
            or first.get("source_contract") != "primary_abstract_paper_native_v6"
            or first.get("policy") != PAPER_NATIVE_V6_REQUIRED_POLICY
            or first.get("verification_status") != "verified"
            or first.get("molecule_chembl_id") != candidate_id
            or first.get("fixture") is not False
            or first.get("missing_values_imputed") is not False
            or not isinstance(attestation, str)
            or re.fullmatch(r"[0-9a-f]{64}", attestation) is None
            or _fingerprint(body) != attestation
        ):
            raise PaperNativeV6GroupUnavailable(
                "A paper-native-v6 envelope is malformed or identity-contradictory."
            )
        return first

    @classmethod
    def _validate_paper_native_v6_group(
        cls,
        seeds: Sequence[CandidateSeed],
    ) -> tuple[str, ...]:
        if not 3 <= len(seeds) <= 12:
            raise PaperNativeV6GroupUnavailable(
                "Paper-native-v6 requires at least three and at most twelve verified seeds."
            )
        candidate_ids = [cls._paper_native_v5_candidate_id(seed) for seed in seeds]
        if any(candidate_id is None for candidate_id in candidate_ids):
            raise PaperNativeV6GroupUnavailable(
                "Every paper-native-v6 seed requires one authoritative ChEMBL ID."
            )
        stable_ids = [str(candidate_id) for candidate_id in candidate_ids]
        if len(stable_ids) != len(set(stable_ids)):
            raise PaperNativeV6GroupUnavailable(
                "The paper-native-v6 group contains duplicate identities."
            )
        expected_group: tuple[str, tuple[str, ...]] | None = None
        previous_rank = 0
        for candidate_id, seed in zip(stable_ids, seeds, strict=True):
            envelope = cls._paper_native_v6_envelope(seed)
            group_ids = envelope.get("group_candidate_ids")
            group_sha256 = envelope.get("group_sha256")
            unit = envelope.get("candidate_support_unit")
            unit_hash = envelope.get("candidate_support_unit_sha256")
            fixed_rank = envelope.get("fixed_pool_rank")
            if (
                not isinstance(group_ids, list)
                or group_ids != stable_ids
                or len(group_ids) < 3
                or not isinstance(group_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", group_sha256) is None
                or not isinstance(unit, Mapping)
                or unit.get("candidate_id") != candidate_id
                or unit.get("support_unit_sha256") != unit_hash
                or not isinstance(unit_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", unit_hash) is None
                or isinstance(fixed_rank, bool)
                or not isinstance(fixed_rank, int)
                or fixed_rank <= previous_rank
                or not cls._paper_native_v6_activity_matches_unit(seed, unit)
            ):
                raise PaperNativeV6GroupUnavailable(
                    "A verified v6 member lacks its atomic support or fixed order."
                )
            previous_rank = fixed_rank
            current = (group_sha256, tuple(group_ids))
            if expected_group is None:
                expected_group = current
            elif current != expected_group:
                raise PaperNativeV6GroupUnavailable(
                    "Verified v6 members carry contradictory group copies."
                )
        return tuple(stable_ids)

    @staticmethod
    def _paper_native_v6_activity_matches_unit(
        seed: CandidateSeed,
        unit: Mapping[str, Any],
    ) -> bool:
        if len(seed.activities) != 1 or len(seed.evidence) != 1:
            return False
        measurement = seed.activities[0]
        evidence = seed.evidence[0]
        return bool(
            measurement.source_evidence_id == evidence.evidence_id
            and unit.get("endpoint") == measurement.endpoint
            and unit.get("relation") == measurement.relation
            and unit.get("value") == measurement.value
            and unit.get("unit") == measurement.unit
            and unit.get("verification_status") == "verified"
            and unit.get("is_primary_support") is True
            and unit.get("fixture") is False
        )

    @classmethod
    def _paper_native_v6_primary_screen(
        cls,
        seed: CandidateSeed,
    ) -> dict[str, Any]:
        candidate_id = cls._paper_native_v5_candidate_id(seed)
        envelope = cls._paper_native_v6_envelope(seed)
        unit = envelope.get("candidate_support_unit")
        valid = bool(
            isinstance(unit, Mapping)
            and cls._paper_native_v6_activity_matches_unit(seed, unit)
        )
        return {
            "policy": PAPER_NATIVE_V6_REQUIRED_POLICY,
            "candidate_id": candidate_id,
            "verification_status": envelope.get("verification_status"),
            "group_sha256": envelope.get("group_sha256"),
            "independent_candidate_primary_support_verified": valid,
            "excluded": not valid,
            "reason_codes": [] if valid else ["CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"],
            "fixture": False,
        }

    @classmethod
    def _paper_native_v5_primary_screen(
        cls,
        seed: CandidateSeed,
    ) -> dict[str, Any]:
        candidate_id = cls._paper_native_v5_candidate_id(seed)
        envelope = cls._paper_native_v5_envelope(seed)
        reasons: list[str] = []
        if envelope.get("verification_status") != "verified":
            reasons.append("CANDIDATE_PRIMARY_SOURCE_UNVERIFIED")
        support = envelope.get("pre_pubmed_chembl_support")
        if envelope.get("verification_status") == "verified" and (
            not isinstance(support, Mapping)
            or support.get("molecule_chembl_id") != candidate_id
            or not isinstance(support.get("activity_id"), (str, int))
            or not isinstance(support.get("support_attestation_sha256"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(support.get("support_attestation_sha256")),
            )
            is None
            or not cls._paper_native_v5_activity_matches_support(seed, support)
        ):
            reasons.append("CANDIDATE_DIRECT_TARGET_SUPPORT_UNVERIFIED")
        return {
            "policy": PAPER_NATIVE_V5_REQUIRED_POLICY,
            "candidate_id": candidate_id,
            "verification_status": envelope.get("verification_status"),
            "group_sha256": (
                envelope.get("group", {}).get("group_sha256")
                if isinstance(envelope.get("group"), Mapping)
                else None
            ),
            "independent_chembl_direct_support_verified": not reasons,
            "pubmed_values_used_for_candidate_activity_or_ranking": False,
            "excluded": bool(reasons),
            "reason_codes": reasons,
            "fixture": False,
        }

    def _evaluate_seed(
        self,
        *,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        hypothesis: Hypothesis,
        target: TargetCandidate,
        route: CandidateRoute,
        seed: CandidateSeed,
        base_trace_ids: Sequence[str],
        allow_refinement: bool,
    ) -> tuple[CandidateCard, int]:
        policy_metadata = self._seed_policy_metadata(seed)
        policy_metadata = {
            **policy_metadata,
            "assay_variant_alignment_hypothesis_id": (
                hypothesis.hypothesis_id
            ),
            "assay_variant_alignment": assay_variant_alignment(
                molecular_scope=hypothesis.target.metadata.get("molecular_hypothesis_scope"),
                target_symbol=target.symbol,
                hypothesis_texts=(
                    hypothesis.mechanism,
                    hypothesis.falsification_test,
                ),
                chembl_assay_context=policy_metadata.get(
                    "chembl_assay_context"
                ),
                expected_target_chembl_id=policy_metadata.get("measured_target"),
            ),
        }
        reserved = screen_reserved_control(
            candidate_name=seed.name,
            metadata=policy_metadata,
            reserved_controls=plan.reserved_control_compounds,
        )
        screen_task = self._task_id(plan, None, "Rank policy", "rank")
        _, reserved_trace = recorder.invoke(
            task_id=screen_task,
            tool_name="candidate_judge",
            operation="screen_reserved_assay_control",
            inputs={
                "candidate_name": seed.name,
                "candidate_metadata": policy_metadata,
                "reserved_control_compounds": plan.reserved_control_compounds,
            },
            function=reserved.as_metadata,
        )
        reserved_metadata = {
            **policy_metadata,
            "reserved_control_screen": reserved.as_metadata(),
            "control_only": reserved.matched,
            "policy_eligible": not reserved.excluded,
        }
        reserved_trace_ids = self._dedupe(
            [*base_trace_ids, reserved_trace]
        )
        if reserved.excluded:
            flags = (
                ["reserved_assay_control"]
                if reserved.matched
                else ["reserved_control_identity_conflict"]
            )
            return (
                CandidateCard(
                    route=route,
                    name=seed.name,
                    target=target,
                    mechanism=seed.mechanism,
                    smiles=seed.smiles,
                    activities=list(seed.activities),
                    safety_flags=flags,
                    evidence=list(seed.evidence),
                    tool_trace_ids=reserved_trace_ids,
                    metadata={
                        **reserved_metadata,
                        "policy_eligible": False,
                    },
                    score=0.0,
                    rank=None,
                    status="excluded",
                    exclusion_reasons=list(reserved.reason_codes),
                ),
                1,
            )

        primary_trace: str | None = None
        policy_metadata = reserved_metadata
        if (
            route is CandidateRoute.REPURPOSING
            and (
                self._paper_native_v5_required()
                or self._paper_native_v6_required()
            )
        ):
            v6_policy = self._paper_native_v6_required()
            primary_screen = (
                self._paper_native_v6_primary_screen(seed)
                if v6_policy
                else self._paper_native_v5_primary_screen(seed)
            )
            _, primary_trace = recorder.invoke(
                task_id=screen_task,
                tool_name="candidate_judge",
                operation=(
                    "screen_paper_native_v6_primary_source"
                    if v6_policy
                    else "screen_paper_native_v5_primary_source"
                ),
                inputs={
                    "candidate_name": seed.name,
                    "candidate_metadata": policy_metadata,
                    "policy": (
                        PAPER_NATIVE_V6_REQUIRED_POLICY
                        if v6_policy
                        else PAPER_NATIVE_V5_REQUIRED_POLICY
                    ),
                },
                function=lambda: primary_screen,
            )
            screen_metadata_key = (
                "paper_native_v6_primary_screen"
                if v6_policy
                else "paper_native_v5_primary_screen"
            )
            policy_metadata = {
                **policy_metadata,
                screen_metadata_key: primary_screen,
                "policy_eligible": not primary_screen["excluded"],
            }
            if primary_screen["excluded"]:
                reasons = list(primary_screen["reason_codes"])
                flags = [
                    (
                        "primary_source_unverified"
                        if reason == "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
                        else "direct_target_support_unverified"
                    )
                    for reason in reasons
                ]
                return (
                    CandidateCard(
                        route=route,
                        name=seed.name,
                        target=target,
                        mechanism=seed.mechanism,
                        smiles=seed.smiles,
                        activities=list(seed.activities),
                        safety_flags=flags,
                        evidence=list(seed.evidence),
                        tool_trace_ids=self._dedupe(
                            [*reserved_trace_ids, primary_trace]
                        ),
                        metadata={**policy_metadata, "policy_eligible": False},
                        score=0.0,
                        rank=None,
                        status="excluded",
                        exclusion_reasons=reasons,
                    ),
                    1,
                )

        restricted_evidence: list[Any] = []
        restricted_trace: str | None = None
        if (
            route is CandidateRoute.REPURPOSING
            and self.restricted_evidence_enricher is not None
        ):
            restricted_task = self._task_id(
                plan, CandidateRoute.REPURPOSING, "repurpos", "search"
            )
            probe = CandidateCard(
                route=route,
                name=seed.name,
                target=target,
                mechanism=seed.mechanism,
                smiles=seed.smiles,
                activities=list(seed.activities),
                evidence=list(seed.evidence),
                metadata=policy_metadata,
            )
            try:
                restricted, restricted_trace = recorder.invoke(
                    task_id=restricted_task,
                    tool_name="restricted_regulatory_registry",
                    operation="precompute_identity_regulatory_contraindication",
                    inputs={
                        "candidate_name": seed.name,
                        "candidate_metadata": policy_metadata,
                        "target": target,
                        "disease": hypothesis.disease,
                    },
                    function=lambda: self.restricted_evidence_enricher.enrich_candidate(
                        probe, hypothesis
                    ),
                )
                restricted_evidence = list(getattr(restricted, "evidence", []))
                restricted_screen = self._restricted_safety_disposition(restricted)
            except Exception as exc:
                # ``_TraceRecorder`` has already retained the failed call.  An
                # unexpected adapter exception is uncertainty, never a clean
                # registry result, and therefore blocks expensive computation.
                if recorder.calls:
                    restricted_trace = recorder.calls[-1].call_id
                restricted_screen = {
                    "screened": True,
                    "disposition": "unresolved",
                    "coverage": "failed",
                    "reason_codes": [
                        f"restricted_safety_lookup_unresolved:{type(exc).__name__}"
                    ],
                    "sources": [],
                    "routing": [],
                    "limitations": [
                        "An adapter exception is unknown safety status, not negative evidence."
                    ],
                }
            policy_metadata = {
                **policy_metadata,
                "restricted_safety_screen": restricted_screen,
                "restricted_source_records": [
                    {
                        "source": item.source,
                        "drug_id": item.external_id,
                        **dict(item.metadata),
                    }
                    for item in restricted_evidence
                ],
            }
        screen = screen_forbidden_drug(
            candidate_name=seed.name,
            metadata=policy_metadata,
            forbidden_names=plan.forbidden_drugs,
            forbidden_ids=plan.forbidden_drug_ids,
            forbidden_classes=plan.forbidden_drug_classes,
        )
        _, screen_trace = recorder.invoke(
            task_id=screen_task,
            tool_name="candidate_judge",
            operation="screen_forbidden_drug",
            inputs={
                "candidate_name": seed.name,
                "candidate_metadata": policy_metadata,
                "forbidden_drugs": plan.forbidden_drugs,
                "forbidden_drug_ids": plan.forbidden_drug_ids,
                "forbidden_drug_classes": plan.forbidden_drug_classes,
            },
            function=screen.as_metadata,
        )
        trace_ids = self._dedupe(
            [
                *base_trace_ids,
                reserved_trace,
                *([primary_trace] if primary_trace else []),
                *([restricted_trace] if restricted_trace else []),
                screen_trace,
            ]
        )
        restricted_disposition = str(
            policy_metadata.get("restricted_safety_screen", {}).get(
                "disposition", "not_run"
            )
        )
        restricted_reasons = list(
            policy_metadata.get("restricted_safety_screen", {}).get(
                "reason_codes", []
            )
        )
        seed_metadata = {
            **policy_metadata,
            "reserved_control_screen": reserved.as_metadata(),
            "control_only": False,
            "forbidden_drug_screen": screen.as_metadata(),
            "policy_eligible": not reserved.excluded
            and not screen.excluded
            and restricted_disposition not in {"excluded", "unresolved"},
        }
        if screen.excluded or restricted_disposition in {"excluded", "unresolved"}:
            flags = ["forbidden_drug"]
            if screen.prohibited_statuses:
                flags.append("prohibited_regulatory_status")
            if restricted_disposition == "excluded":
                flags.append("restricted_safety_exclusion")
            if restricted_disposition == "unresolved":
                flags.append("restricted_safety_unresolved")
            return (
                CandidateCard(
                    route=route,
                    name=seed.name,
                    target=target,
                    mechanism=seed.mechanism,
                    smiles=seed.smiles,
                    activities=list(seed.activities),
                    safety_flags=flags,
                    evidence=self._dedupe_evidence(
                        [*seed.evidence, *restricted_evidence]
                    ),
                    tool_trace_ids=trace_ids,
                    metadata=seed_metadata,
                    score=0.0,
                    rank=None,
                    status="excluded",
                    exclusion_reasons=self._dedupe(
                        [*screen.reason_codes, *restricted_reasons]
                    ),
                ),
                1,
            )

        seed = replace(
            seed,
            evidence=tuple(
                self._dedupe_evidence([*seed.evidence, *restricted_evidence])
            ),
        )

        current = seed
        iteration = 0
        while True:
            card, evaluation_trace = self._evaluate_once(
                plan=plan,
                recorder=recorder,
                target=target,
                route=route,
                seed=current,
                trace_ids=trace_ids,
                candidate_metadata=seed_metadata,
            )

            trace_ids = evaluation_trace
            if plan.forbidden_substructures:
                seed_metadata["forbidden_substructure_screens"] = list(
                    card.metadata.get("forbidden_substructure_screens", [])
                )
            refinable = {
                "hepatotoxicity_high",
                "herg_risk_high",
                "invalid_structure",
                "molecular_weight_out_of_range",
                "clogp_out_of_range",
            }.intersection(card.safety_flags)
            if (
                not allow_refinement
                or not refinable
                or iteration >= self.max_refinement_iterations
            ):
                return card, 1 + iteration
            iteration += 1
            refine_task = self._task_id(plan, CandidateRoute.DE_NOVO, "de_novo", "generate")
            current, refinement_trace = recorder.invoke(
                task_id=refine_task,
                tool_name="de_novo_generator",
                operation="refine_candidate",
                inputs={
                    "seed": current,
                    "reasons": sorted(refinable),
                    "iteration": iteration,
                },
                function=lambda current=current, refinable=refinable, iteration=iteration: self.gateway.refine(
                    current,
                    reasons=sorted(refinable),
                    iteration=iteration,
                ),
            )
            trace_ids = self._dedupe([*trace_ids, refinement_trace])

    @staticmethod
    def _seed_policy_metadata(seed: CandidateSeed) -> dict[str, Any]:
        """Expose source IDs/statuses to the pre-compute policy screen."""

        return {
            **dict(seed.metadata),
            "source_evidence": [
                {
                    "source": item.source,
                    "drug_id": item.external_id,
                    "evidence_id": item.evidence_id,
                    "metadata": dict(item.metadata),
                }
                for item in seed.evidence
            ],
            "identifiers": [item.external_id for item in seed.evidence],
        }

    def _evaluate_once(
        self,
        *,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        target: TargetCandidate,
        route: CandidateRoute,
        seed: CandidateSeed,
        trace_ids: Sequence[str],
        candidate_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[CandidateCard, list[str]]:
        route_hint = route
        policy_trace = None
        if plan.forbidden_substructures:
            candidate_metadata, policy_trace, policy = self._screen_substructure(
                plan, recorder, seed.smiles, candidate_metadata or seed.metadata,
            )
            trace_ids = self._dedupe([*trace_ids, policy_trace])
            if policy["blocked"]:
                return CandidateCard(
                    route=route, name=seed.name, target=target,
                    mechanism=seed.mechanism, smiles=seed.smiles,
                    activities=list(seed.activities), evidence=list(seed.evidence),
                    tool_trace_ids=list(trace_ids), metadata=dict(candidate_metadata),
                    safety_flags=["forbidden_substructure" if policy["matched"] else
                                  "invalid_structure" if policy["status"] == "invalid_structure" else "chemical_policy_check_unavailable"],
                    status="excluded", exclusion_reasons=list(policy["reason_codes"]),
                ), list(trace_ids)
        chemistry_task = self._task_id(plan, route_hint, "chemistry", "standard")
        chemistry_inputs = {"name": seed.name, "smiles": seed.smiles}
        chemistry, chemistry_trace = recorder.invoke(
            task_id=chemistry_task,
            tool_name="chemistry_validator",
            operation="validate_and_describe",
            inputs=chemistry_inputs,
            chemical_policy_screen_id=policy_trace,
            function=lambda: self._chemistry_result(
                seed.smiles,
                input_hash=_fingerprint(chemistry_inputs),
            ),
        )
        accumulated = self._dedupe([*trace_ids, chemistry_trace])
        if chemistry.status is not ScientificExecutionStatus.COMPLETED:
            card = CandidateCard(
                route=route,
                name=seed.name,
                target=target,
                mechanism=seed.mechanism,
                smiles=seed.smiles,
                activities=list(seed.activities),
                evidence=list(seed.evidence),
                tool_trace_ids=accumulated,
                metadata={
                    **dict(candidate_metadata or seed.metadata),
                    "missing_values_imputed": False,
                },
                safety_flags=["scientific_tool_unavailable"],
                status="excluded",
                exclusion_reasons=[
                    chemistry.reason_code or "CHEMISTRY_TOOL_UNAVAILABLE"
                ],
                chemistry_execution=chemistry,
            )
            return card, accumulated
        if not chemistry.valid or chemistry.canonical_smiles is None:
            card = CandidateCard(
                route=route,
                name=seed.name,
                target=target,
                mechanism=seed.mechanism,
                smiles=seed.smiles,
                activities=list(seed.activities),
                evidence=list(seed.evidence),
                tool_trace_ids=accumulated,
                metadata=dict(candidate_metadata or seed.metadata),
                safety_flags=["invalid_structure"],
                status="excluded",
                exclusion_reasons=list(chemistry.errors),
                chemistry_execution=chemistry,
            )
            return card, accumulated

        docking_task = self._task_id(plan, route_hint, "dock")
        docking_inputs = {"smiles": chemistry.canonical_smiles, "target": target}
        target_linked_activity = self._has_target_linked_measured_activity(
            seed,
            target,
        )
        docking_optional = not self._plan_tool_required(plan, "docking")
        if not self.fixture_mode and docking_optional and target_linked_activity:
            docking, docking_trace = recorder.invoke(
                task_id=docking_task,
                tool_name="docking",
                operation="score_binding_pose",
                inputs=docking_inputs,
                chemical_policy_screen_id=policy_trace,
                function=lambda: DockingExecutionResult(
                    status=ScientificExecutionStatus.SKIPPED,
                    score_kcal_mol=None,
                    method="not-run",
                    backend_version=str(self.gateway.source_version),
                    backend_kind=self.backend_kind,
                    fixture=False,
                    input_hash=_fingerprint(docking_inputs),
                    reason_code="DIRECT_MEASURED_ACTIVITY_AVAILABLE",
                    provenance={
                        "plan_tool_required": False,
                        "measured_activity_count": len(seed.activities),
                        "target_linked_activity_verified": True,
                        "missing_values_imputed": False,
                    },
                ),
            )
        elif (
            not self.fixture_mode
            and docking_optional
            and (
                docking_capability := self._docking_capability(target)
            ) is not None
            and docking_capability.status is not ReadinessStatus.READY
        ):
            # The target-specific profile is absent or not production-ready.
            # Docking is optional in the approved plan, so retain a truthful
            # not-assessed result and continue to the independently required
            # ADMET execution.  Candidate eligibility is decided below; no
            # neutral docking score is created.
            docking, docking_trace = recorder.invoke(
                task_id=docking_task,
                tool_name="docking",
                operation="score_binding_pose",
                inputs=docking_inputs,
                chemical_policy_screen_id=policy_trace,
                function=lambda: DockingExecutionResult(
                    status=ScientificExecutionStatus.SKIPPED,
                    score_kcal_mol=None,
                    method="not-run",
                    backend_version=docking_capability.backend_version,
                    backend_kind=self.backend_kind,
                    fixture=False,
                    input_hash=_fingerprint(docking_inputs),
                    reason_code="OPTIONAL_DOCKING_NOT_ASSESSED",
                    provenance={
                        "plan_tool_required": False,
                        "target_linked_activity_verified": False,
                        "capability_status": docking_capability.status.value,
                        "capability_reason_code": docking_capability.reason_code,
                        "capability_detail": docking_capability.detail,
                        "capability_backend": docking_capability.backend_name,
                        "capability_provenance": dict(
                            docking_capability.provenance
                        ),
                        "candidate_eligible_without_target_engagement": False,
                        "missing_values_imputed": False,
                    },
                ),
            )
        else:
            docking, docking_trace = recorder.invoke(
                task_id=docking_task,
                tool_name="docking",
                operation="score_binding_pose",
                inputs=docking_inputs,
                chemical_policy_screen_id=policy_trace,
                function=lambda: self._docking_result(
                    smiles=chemistry.canonical_smiles,
                    target=target,
                    input_hash=_fingerprint(docking_inputs),
                ),
            )
        admet_task = self._task_id(plan, route_hint, "admet", "safety")
        admet_inputs = {"smiles": chemistry.canonical_smiles}
        admet, admet_trace = recorder.invoke(
            task_id=admet_task,
            tool_name="admet_predictor",
            operation="predict_six_endpoint_panel",
            inputs=admet_inputs,
            chemical_policy_screen_id=policy_trace,
            function=lambda: self._admet_result(
                smiles=chemistry.canonical_smiles,
                input_hash=_fingerprint(admet_inputs),
            ),
        )
        accumulated = self._dedupe([*accumulated, docking_trace, admet_trace])
        unavailable = [
            result
            for result in (docking, admet)
            if result.status in {
                ScientificExecutionStatus.UNAVAILABLE,
                ScientificExecutionStatus.FAILED,
            }
        ]
        if unavailable:
            reasons = [
                result.reason_code or "SCIENTIFIC_TOOL_UNAVAILABLE"
                for result in unavailable
            ]
            return (
                CandidateCard(
                    route=route,
                    name=seed.name,
                    target=target,
                    mechanism=seed.mechanism,
                    smiles=chemistry.canonical_smiles,
                    inchikey=chemistry.inchikey,
                    activities=list(seed.activities),
                    metrics=chemistry.metrics,
                    safety_flags=["scientific_tool_unavailable"],
                    evidence=list(seed.evidence),
                    tool_trace_ids=accumulated,
                    metadata={
                        **dict(candidate_metadata or seed.metadata),
                        "missing_values_imputed": False,
                    },
                    status="excluded",
                    exclusion_reasons=self._dedupe(reasons),
                    chemistry_execution=chemistry,
                    docking_execution=docking,
                    admet_execution=admet,
                ),
                accumulated,
            )
        metrics = self._merge_metrics(chemistry, docking, admet)
        flags, exclusions = self._risk_flags(metrics)
        if (
            docking.status is ScientificExecutionStatus.SKIPPED
            and docking.reason_code == "OPTIONAL_DOCKING_NOT_ASSESSED"
            and not target_linked_activity
        ):
            flags = self._dedupe([*flags, "target_engagement_not_assessed"])
            exclusions = self._dedupe(
                [*exclusions, "TARGET_ENGAGEMENT_EVIDENCE_UNAVAILABLE"]
            )
        ood_score = self._ood_score(metrics)
        metrics = metrics.model_copy(update={"ood_score": ood_score})
        applicability_assessment = self._applicability_assessment(metrics)
        cell_context_activity = self._has_cell_context_activity(seed)
        card = CandidateCard(
            route=route,
            name=seed.name,
            target=target,
            mechanism=seed.mechanism,
            smiles=chemistry.canonical_smiles,
            inchikey=chemistry.inchikey,
            activities=list(seed.activities),
            metrics=metrics,
            safety_flags=flags,
            evidence=list(seed.evidence),
            tool_trace_ids=accumulated,
            metadata={
                **dict(candidate_metadata or seed.metadata),
                "missing_values_imputed": False,
                "activity_strength_status": (
                    "cellular_activity_available_direct_target_engagement_unresolved"
                    if cell_context_activity and not target_linked_activity
                    else "typed_measured_activity_available"
                    if seed.activities
                    else "typed_measured_activity_unavailable"
                ),
                "target_engagement_assessment": (
                    "direct_measured_activity"
                    if target_linked_activity
                    else (
                        "cellular_activity_only_direct_target_engagement_not_established"
                        if cell_context_activity
                        and docking.reason_code == "OPTIONAL_DOCKING_NOT_ASSESSED"
                        else "not_assessed_no_production_docking_profile"
                        if docking.reason_code == "OPTIONAL_DOCKING_NOT_ASSESSED"
                        else "docking_assessed"
                    )
                ),
                "admet_decision_policy": (
                    "prediction_as_warning_not_confirmatory_exclusion_v1"
                ),
                "admet_prediction_requires_experimental_confirmation": True,
                "applicability_assessment": applicability_assessment,
            },
            status="excluded" if exclusions else "conditional",
            exclusion_reasons=exclusions,
            chemistry_execution=chemistry,
            docking_execution=docking,
            admet_execution=admet,
        )
        return card, accumulated

    @staticmethod
    def _plan_tool_required(plan: PlanSpec, tool_name: str) -> bool:
        return any(
            assignment.tool_name == tool_name and assignment.required
            for task in plan.tasks
            for assignment in task.tools
        )

    def _docking_capability(
        self,
        target: TargetCandidate,
    ) -> ToolReadiness | None:
        readiness = self._gateway_readiness(target)
        return next(
            (
                item
                for item in readiness.capabilities
                if item.tool_name == "docking"
                and item.operation == "score_binding_pose"
            ),
            None,
        )

    @staticmethod
    def _has_target_linked_measured_activity(
        seed: CandidateSeed,
        target: TargetCandidate,
    ) -> bool:
        """Require each docking waiver to cite activity tied to this target.

        Stable identifiers take precedence over gene symbols.  In particular,
        a matching symbol must never rescue an evidence record that declares a
        conflicting UniProt accession; otherwise wrong-target activity could
        waive the independently planned docking check.
        """

        evidence_by_id = {
            evidence.evidence_id: evidence
            for evidence in seed.evidence
        }
        target_uniprot = (target.uniprot_id or "").casefold()
        target_symbol = target.symbol.casefold()
        target_chembl = str(
            target.metadata.get("chembl_target_id") or ""
        ).casefold()
        for activity in seed.activities:
            if not activity.source_evidence_id:
                continue
            evidence = evidence_by_id.get(activity.source_evidence_id)
            if evidence is None:
                continue
            metadata = evidence.metadata
            direct_attestation = metadata.get(
                "direct_single_protein_engagement"
            )
            if direct_attestation is not True and not (
                direct_attestation is None
                and MA2Agent._assay_type_supports_direct_engagement(
                    activity.assay_type
                )
            ):
                continue
            accessions = {
                str(metadata.get(key) or "").casefold()
                for key in (
                    "target_accession",
                    "target_uniprot_id",
                    "uniprot_accession",
                    "uniprot_id",
                )
            }
            genes = {
                str(metadata.get(key) or "").casefold()
                for key in ("target_gene", "gene", "gene_symbol")
            }
            chembl_targets = {
                str(metadata.get(key) or "").casefold()
                for key in ("target_chembl_id", "chembl_target_id")
            }
            accessions.discard("")
            genes.discard("")
            chembl_targets.discard("")
            if target_uniprot:
                linked = (
                    accessions == {target_uniprot}
                    if accessions
                    else bool(
                        target_chembl
                        and chembl_targets == {target_chembl}
                    )
                )
            elif target_chembl:
                linked = chembl_targets == {target_chembl}
            else:
                linked = bool(
                    target_symbol
                    and not accessions
                    and not chembl_targets
                    and genes == {target_symbol}
                )
            if linked:
                return True
        return False

    @staticmethod
    def _assay_type_supports_direct_engagement(assay_type: str) -> bool:
        normalized = " ".join(
            assay_type.casefold().replace("_", " ").split()
        )
        if any(
            term in normalized
            for term in ("cell", "tissue", "organism", "phenotyp")
        ):
            return False
        return any(
            term in normalized
            for term in (
                "single protein",
                "biochemical",
                "protein binding",
                "purified protein",
                "enzyme assay",
            )
        )

    @staticmethod
    def _has_cell_context_activity(seed: CandidateSeed) -> bool:
        return any(
            any(
                term in " ".join(
                    activity.assay_type.casefold().replace("_", " ").split()
                )
                for term in ("cell", "tissue", "organism", "phenotyp")
            )
            for activity in seed.activities
        )

    def _chemistry_result(
        self,
        smiles: str,
        *,
        input_hash: str,
    ) -> ChemistryExecutionResult:
        backend = self.chemistry.backend_name
        if not self.fixture_mode and backend != "rdkit":
            return ChemistryExecutionResult(
                status=ScientificExecutionStatus.UNAVAILABLE,
                backend_name=backend,
                backend_version=None,
                backend_kind=self.backend_kind,
                fixture=False,
                input_hash=input_hash,
                reason_code="RDKIT_NOT_AVAILABLE",
                errors=[
                    "Live chemistry validation requires RDKit; syntax fallback is demo-only."
                ],
                provenance={"missing_values_imputed": False},
            )
        report = self.chemistry.inspect(smiles)
        return ChemistryExecutionResult(
            status=ScientificExecutionStatus.COMPLETED,
            valid=report.valid,
            canonical_smiles=report.canonical_smiles,
            inchikey=report.inchikey,
            metrics=report.metrics,
            backend_name=report.backend,
            backend_version=(
                str(self.gateway.source_version)
                if self.fixture_mode
                else getattr(self.chemistry, "backend_version", None)
            ),
            backend_kind=self.backend_kind,
            fixture=self.fixture_mode,
            input_hash=input_hash,
            errors=list(report.errors),
            provenance={
                "validator": type(self.chemistry).__name__,
                "standardization_version": report.standardization_version,
                "original_canonical_smiles": report.original_canonical_smiles,
                "identity_parent_smiles": report.identity_parent_smiles,
                "tautomer_parent_smiles": report.tautomer_parent_smiles,
                "stereo_status": report.stereo_status,
                "transformations": list(report.transformations),
                "standardization_warnings": list(report.warnings),
                **dict(report.provenance),
                "missing_values_imputed": False,
            },
        )

    def _docking_result(
        self,
        *,
        smiles: str,
        target: TargetCandidate,
        input_hash: str,
    ) -> DockingExecutionResult:
        raw = self.gateway.dock(smiles=smiles, target=target)
        if isinstance(raw, DockingExecutionResult):
            if raw.input_hash != input_hash:
                return DockingExecutionResult(
                    status=ScientificExecutionStatus.FAILED,
                    method=raw.method,
                    backend_version=raw.backend_version,
                    backend_kind=self.backend_kind,
                    fixture=self.fixture_mode,
                    input_hash=input_hash,
                    reason_code="DOCKING_INPUT_HASH_MISMATCH",
                    provenance={
                        **raw.provenance,
                        "reported_input_hash": raw.input_hash,
                    },
                )
            return raw
        provenance = self._last_gateway_provenance()
        if not self.fixture_mode and not provenance:
            return DockingExecutionResult(
                status=ScientificExecutionStatus.UNAVAILABLE,
                method="unverified-live-docking",
                backend_version=None,
                backend_kind=self.backend_kind,
                fixture=False,
                input_hash=input_hash,
                reason_code="LIVE_DOCKING_PROVENANCE_MISSING",
                provenance={"missing_values_imputed": False},
            )
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return DockingExecutionResult(
                status=ScientificExecutionStatus.FAILED,
                method=str(provenance.get("method") or "docking"),
                backend_version=self._provenance_version(provenance),
                backend_kind=self.backend_kind,
                fixture=self.fixture_mode,
                input_hash=input_hash,
                reason_code="DOCKING_RESULT_NOT_NUMERIC",
                provenance=provenance,
            )
        return DockingExecutionResult(
            status=ScientificExecutionStatus.COMPLETED,
            score_kcal_mol=score,
            method=str(
                provenance.get("method")
                or ("deterministic-fixture" if self.fixture_mode else "AutoDock Vina")
            ),
            backend_version=(
                self._provenance_version(provenance)
                or str(self.gateway.source_version)
            ),
            backend_kind=self.backend_kind,
            fixture=self.fixture_mode,
            input_hash=input_hash,
            provenance={**provenance, "missing_values_imputed": False},
        )

    def _admet_result(
        self,
        *,
        smiles: str,
        input_hash: str,
    ) -> ADMETExecutionResult:
        raw = self.gateway.predict_admet(smiles=smiles)
        if isinstance(raw, ADMETExecutionResult):
            if raw.input_hash == input_hash:
                return raw
            return ADMETExecutionResult(
                status=ScientificExecutionStatus.FAILED,
                required_endpoints=list(_REQUIRED_ADMET_ENDPOINTS),
                missing_endpoints=list(_REQUIRED_ADMET_ENDPOINTS),
                method=raw.method,
                backend_version=raw.backend_version,
                backend_kind=self.backend_kind,
                fixture=self.fixture_mode,
                input_hash=input_hash,
                reason_code="ADMET_INPUT_HASH_MISMATCH",
                provenance={
                    **raw.provenance,
                    "reported_input_hash": raw.input_hash,
                },
            )
        provenance = self._last_gateway_provenance()
        if not self.fixture_mode and not provenance:
            return ADMETExecutionResult(
                status=ScientificExecutionStatus.UNAVAILABLE,
                required_endpoints=list(_REQUIRED_ADMET_ENDPOINTS),
                missing_endpoints=list(_REQUIRED_ADMET_ENDPOINTS),
                method="unverified-live-admet",
                backend_kind=self.backend_kind,
                fixture=False,
                input_hash=input_hash,
                reason_code="LIVE_ADMET_PROVENANCE_MISSING",
                provenance={"missing_values_imputed": False},
            )
        scores: dict[str, float] = {}
        invalid: list[str] = []
        if not isinstance(raw, Mapping):
            invalid.append("result_not_mapping")
        else:
            for key, value in raw.items():
                if key not in _REQUIRED_ADMET_ENDPOINTS:
                    continue
                if isinstance(value, bool):
                    invalid.append(str(key))
                    continue
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    invalid.append(str(key))
                    continue
                if not 0.0 <= parsed <= 1.0:
                    invalid.append(str(key))
                    continue
                scores[str(key)] = parsed
        missing = sorted(set(_REQUIRED_ADMET_ENDPOINTS) - set(scores))
        if missing or invalid:
            return ADMETExecutionResult(
                status=ScientificExecutionStatus.UNAVAILABLE,
                required_endpoints=list(_REQUIRED_ADMET_ENDPOINTS),
                missing_endpoints=list(_REQUIRED_ADMET_ENDPOINTS),
                method=str(provenance.get("method") or "ADMET predictor"),
                backend_version=self._provenance_version(provenance),
                backend_kind=self.backend_kind,
                fixture=self.fixture_mode,
                input_hash=input_hash,
                reason_code="ADMET_ENDPOINTS_INCOMPLETE_OR_INVALID",
                provenance={
                    **provenance,
                    "observed_endpoints": sorted(scores),
                    "missing_endpoints": missing,
                    "invalid_endpoints": sorted(set(invalid)),
                    "partial_values_discarded": True,
                    "missing_values_imputed": False,
                },
            )
        return ADMETExecutionResult(
            status=ScientificExecutionStatus.COMPLETED,
            scores=scores,
            required_endpoints=list(_REQUIRED_ADMET_ENDPOINTS),
            missing_endpoints=[],
            method=str(
                provenance.get("method")
                or ("deterministic-fixture" if self.fixture_mode else "ADMET-AI")
            ),
            backend_version=(
                self._provenance_version(provenance)
                or str(self.gateway.source_version)
            ),
            backend_kind=self.backend_kind,
            fixture=self.fixture_mode,
            input_hash=input_hash,
            provenance={**provenance, "missing_values_imputed": False},
        )

    def _last_gateway_provenance(self) -> dict[str, Any]:
        for gateway in (self.gateway, getattr(self.gateway, "delegate", None)):
            if gateway is None:
                continue
            payload = getattr(gateway, "last_operation_provenance", None)
            if isinstance(payload, Mapping) and payload:
                return {str(key): _jsonable(value) for key, value in payload.items()}
        return {}

    @staticmethod
    def _provenance_version(provenance: Mapping[str, Any]) -> str | None:
        for key in (
            "backend_version",
            "source_version",
            "model_version",
            "tool_version",
            "version",
        ):
            value = provenance.get(key)
            if value not in (None, ""):
                return str(value)
        execution = provenance.get("execution")
        if isinstance(execution, Mapping):
            value = execution.get("executable_version")
            if value not in (None, ""):
                return str(value)
        return None

    @staticmethod
    def _repair_intent(order: ReworkOrder) -> frozenset[str]:
        """Resolve the rework scope from the failed node, not unrelated notes.

        A G1 order contains every unresolved finding in ``required_actions``.
        Only the first finding selects ``failed_node``.  Scanning the combined
        prose made an ADMET rework look like a structure rework whenever a
        later reviewer note happened to mention a compound structure.  Keep
        the durable failed-node contract authoritative and use only the first
        action as a compatibility fallback for older/generic node names.
        """

        node = order.failed_node.strip().casefold()
        if node in {"ma2.tool_dispatch", "ma2.plan_trace"}:
            return frozenset({"trace"})
        if node == "ma2.standardize":
            return frozenset({"structure"})
        if node == "ma2.activity_normalization":
            return frozenset({"activity"})
        if node == "ma2.admet":
            return frozenset({"admet"})
        if node in {"ma2.dock", "ma2.docking"}:
            return frozenset({"docking"})
        if node == "ma2.ranking":
            # Ranking-policy findings concern the derived score/attestation,
            # not the persisted source measurement.  In particular, a G1
            # cross-assay recalculation must remain restart-safe: rework below
            # always reruns ``_rank`` from the checkpointed CandidateCard, so
            # it must not try to recover the same activity from a process-local
            # gateway cache merely because the finding message says
            # "activity ranking".
            return frozenset({"ranking"})
        if node in {"ma2.assay_variant_alignment", "ma2.prospective_falsification"}:
            return frozenset({"semantic_attestation"})

        primary_action = (
            order.required_actions[0].casefold()
            if order.required_actions
            else ""
        )
        text = f"{node} {primary_action}"
        intents: set[str] = set()
        if any(
            word in text
            for word in (
                "trace",
                "plan task",
                "plan_trace",
                "tool_dispatch",
                "unknown task",
                "task mismatch",
            )
        ):
            intents.add("trace")
        if any(
            word in text
            for word in ("smiles", "structure", "standardize", "invalid_structure")
        ):
            intents.add("structure")
        if any(word in text for word in ("ic50", "activity", "unit", "scale")):
            intents.add("activity")
        if any(word in text for word in ("direction", "mechanism")):
            intents.add("direction")
        if any(word in text for word in ("admet", "safety", "toxicity")):
            intents.add("admet")
        if "dock" in text:
            intents.add("docking")
        return frozenset(intents)

    def _restore_persisted_candidate_seed(
        self,
        candidate: CandidateCard,
        *,
        target: TargetCandidate,
        direction: str,
        allow_structure_repair: bool,
    ) -> CandidateSeed:
        """Restore a candidate from its checkpointed, tool-attested artifact.

        Rework can run in a new process with a fresh scientific gateway, so an
        in-memory name cache is not a durable source of truth.  The candidate
        artifact already retains the standardized structure, InChIKey, source
        identifiers, target-linked evidence and chemistry execution
        attestation.  Validate those immutable fields exactly and fail closed
        on any conflict; never query a fixture or invent a replacement.
        """

        def unavailable(message: str) -> ScientificDataUnavailable:
            error = ScientificDataUnavailable(
                f"Persisted candidate identity is not reusable for "
                f"{candidate.candidate_id}: {message}"
            )
            error.provenance = {
                "operation": "restore_persisted_candidate_identity",
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.name,
                "target_key": self._portfolio_target_key(target),
                "fixture": self.fixture_mode,
                "missing_values_imputed": False,
            }
            return error

        expected_target_key = self._portfolio_target_key(target)
        candidate_target_key = self._portfolio_target_key(candidate.target)
        if candidate_target_key != expected_target_key:
            raise unavailable(
                f"candidate target {candidate_target_key} conflicts with "
                f"requested target {expected_target_key}"
            )
        declared_target_key = str(
            candidate.metadata.get("portfolio_target_key") or ""
        ).strip()
        if (
            not self.fixture_mode
            and declared_target_key
            and declared_target_key.casefold() != expected_target_key.casefold()
        ):
            raise unavailable(
                f"portfolio target {declared_target_key} conflicts with "
                f"requested target {expected_target_key}"
            )

        target_uniprot = (target.uniprot_id or "").strip().casefold()
        candidate_uniprot = (candidate.target.uniprot_id or "").strip().casefold()
        if target_uniprot and candidate_uniprot != target_uniprot:
            raise unavailable("candidate UniProt identity is absent or conflicting")
        for key in ("measured_target_uniprot_id", "target_accession", "target_uniprot_id"):
            declared = str(candidate.metadata.get(key) or "").strip().casefold()
            if declared and target_uniprot and declared != target_uniprot:
                raise unavailable(f"declared {key}={declared} conflicts with target")
        target_symbol = target.symbol.strip().casefold()
        for key in ("measured_target_gene", "target_gene"):
            declared = str(candidate.metadata.get(key) or "").strip().casefold()
            if declared and declared != target_symbol:
                raise unavailable(f"declared {key}={declared} conflicts with target")
        declared_direction = str(
            candidate.metadata.get("direction_requested") or ""
        ).strip().casefold()
        if declared_direction and declared_direction != direction.strip().casefold():
            raise unavailable(
                f"declared direction {declared_direction} conflicts with {direction}"
            )

        chemistry = candidate.chemistry_execution
        if (
            chemistry is None
            or chemistry.status is not ScientificExecutionStatus.COMPLETED
            or chemistry.valid is not True
            or not chemistry.canonical_smiles
            or not chemistry.inchikey
        ):
            raise unavailable("completed valid chemistry attestation is missing")
        if chemistry.backend_kind is not self.backend_kind:
            raise unavailable("chemistry backend kind conflicts with the active run")
        if chemistry.fixture != self.fixture_mode:
            raise unavailable("chemistry fixture boundary conflicts with the active run")

        canonical_smiles = chemistry.canonical_smiles.strip()
        canonical_inchikey = chemistry.inchikey.strip().upper()
        if not allow_structure_repair and candidate.smiles.strip() != canonical_smiles:
            raise unavailable("candidate SMILES conflicts with chemistry attestation")
        if candidate.inchikey and candidate.inchikey.strip().upper() != canonical_inchikey:
            raise unavailable("candidate InChIKey conflicts with chemistry attestation")
        declared_smiles = str(
            candidate.metadata.get("canonical_smiles") or ""
        ).strip()
        if declared_smiles and declared_smiles != canonical_smiles:
            raise unavailable("source canonical SMILES conflicts with chemistry attestation")
        declared_inchikey = str(
            candidate.metadata.get("standard_inchi_key")
            or candidate.metadata.get("resolved_inchikey")
            or ""
        ).strip().upper()
        if declared_inchikey and declared_inchikey != canonical_inchikey:
            raise unavailable("source InChIKey conflicts with chemistry attestation")

        source_ids = {
            str(candidate.metadata.get(key) or "").strip().casefold()
            for key in ("chembl_id", "drug_id", "compound_id")
        }
        identifiers = candidate.metadata.get("identifiers", [])
        if isinstance(identifiers, (list, tuple, set, frozenset)):
            source_ids.update(
                str(value).strip().casefold()
                for value in identifiers
                if str(value).strip()
            )
        source_ids.update(
            evidence.external_id.strip().casefold()
            for evidence in candidate.evidence
            if evidence.external_id.strip()
        )
        source_ids.discard("")
        source_ids.add(canonical_inchikey.casefold())

        if not self.fixture_mode:
            if candidate.metadata.get("fixture") is True or chemistry.fixture:
                raise unavailable("fixture provenance cannot be restored in live mode")
            if any(evidence.metadata.get("fixture") is True for evidence in candidate.evidence):
                raise unavailable("fixture evidence cannot be restored in live mode")
            if not source_ids or not candidate.evidence:
                raise unavailable("live source identifiers or evidence are missing")

        restored = CandidateSeed(
            name=candidate.name,
            smiles=canonical_smiles,
            mechanism=candidate.mechanism,
            activities=tuple(candidate.activities),
            evidence=tuple(candidate.evidence),
            metadata={
                **candidate.metadata,
                "rework_identity_attestation": {
                    "candidate_id": candidate.candidate_id,
                    "source": "checkpointed_candidate_artifact",
                    "target_key": expected_target_key,
                    "inchikey": canonical_inchikey,
                    "source_ids": sorted(source_ids),
                    "chemistry_input_hash": chemistry.input_hash,
                    "fixture": self.fixture_mode,
                    "validated": True,
                    "missing_values_imputed": False,
                },
            },
        )
        if (
            not self.fixture_mode
            and candidate.route is CandidateRoute.REPURPOSING
            and not self._has_exact_persisted_target_provenance(restored, target)
        ):
            raise unavailable(
                "repurposing activity is not exactly linked to the requested target"
            )
        return restored

    @staticmethod
    def _has_exact_persisted_target_provenance(
        seed: CandidateSeed,
        target: TargetCandidate,
    ) -> bool:
        """Check exact source-target identity without claiming target engagement.

        A cell-based activity can be attributed to a ChEMBL target record while
        still being insufficient to waive docking or prove direct engagement.
        Rework persistence needs the former identity assertion only; the
        stricter assay-appropriateness decision remains in
        ``_has_target_linked_measured_activity``.
        """

        evidence_by_id = {item.evidence_id: item for item in seed.evidence}
        target_uniprot = (target.uniprot_id or "").strip().casefold()
        target_symbol = target.symbol.strip().casefold()
        target_chembl = str(
            target.metadata.get("chembl_target_id") or ""
        ).strip().casefold()
        for activity in seed.activities:
            evidence = evidence_by_id.get(activity.source_evidence_id or "")
            if evidence is None:
                continue
            metadata = evidence.metadata
            accessions = {
                str(metadata.get(key) or "").strip().casefold()
                for key in (
                    "target_accession",
                    "target_uniprot_id",
                    "uniprot_accession",
                    "uniprot_id",
                )
            }
            genes = {
                str(metadata.get(key) or "").strip().casefold()
                for key in ("target_gene", "gene", "gene_symbol")
            }
            chembl_targets = {
                str(metadata.get(key) or "").strip().casefold()
                for key in ("target_chembl_id", "chembl_target_id")
            }
            accessions.discard("")
            genes.discard("")
            chembl_targets.discard("")
            if target_uniprot:
                if accessions:
                    linked = accessions == {target_uniprot}
                else:
                    linked = bool(
                        target_chembl
                        and chembl_targets == {target_chembl}
                    )
            elif target_chembl:
                linked = chembl_targets == {target_chembl}
            else:
                linked = (
                    bool(target_symbol)
                    and not accessions
                    and not chembl_targets
                    and genes == {target_symbol}
                )
            if linked:
                return True
        return False

    def _repair_candidate(
        self,
        *,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        hypothesis: Hypothesis,
        target: TargetCandidate,
        direction: str,
        previous: CandidateCard,
        repair_intent: frozenset[str],
    ) -> tuple[CandidateCard, int]:
        trace_ids = list(previous.tool_trace_ids)
        card = previous.model_copy(deep=True)
        structure_error = "structure" in repair_intent
        activity_error = "activity" in repair_intent
        direction_error = "direction" in repair_intent
        admet_error = "admet" in repair_intent
        docking_error = "docking" in repair_intent

        persisted_seed: CandidateSeed | None = None
        if structure_error or docking_error or admet_error:
            identity_task = self._task_id(plan, None, "Rank policy", "rank")
            persisted_seed, identity_trace = recorder.invoke(
                task_id=identity_task,
                tool_name="candidate_judge",
                operation="validate_persisted_candidate_identity",
                inputs={
                    "candidate_id": previous.candidate_id,
                    "candidate_name": previous.name,
                    "route": previous.route,
                    "target_key": self._portfolio_target_key(target),
                    "inchikey": previous.inchikey,
                    "chemistry_input_hash": (
                        previous.chemistry_execution.input_hash
                        if previous.chemistry_execution is not None
                        else None
                    ),
                    "allow_structure_repair": structure_error,
                },
                function=lambda: self._restore_persisted_candidate_seed(
                    previous,
                    target=target,
                    direction=direction,
                    allow_structure_repair=structure_error,
                ),
            )
            trace_ids = self._dedupe([*trace_ids, identity_trace])
            card = card.model_copy(
                update={
                    "metadata": dict(persisted_seed.metadata),
                    "tool_trace_ids": trace_ids,
                },
                deep=True,
            )

        if structure_error:
            if persisted_seed is None:  # pragma: no cover - guarded above
                raise RuntimeError("persisted structure was not validated")
            rebuilt, iterations = self._evaluate_seed(
                plan=plan,
                recorder=recorder,
                hypothesis=hypothesis,
                target=target,
                route=previous.route,
                seed=persisted_seed,
                base_trace_ids=trace_ids,
                allow_refinement=previous.route is CandidateRoute.DE_NOVO,
            )
            return rebuilt.model_copy(update={"candidate_id": previous.candidate_id}), iterations

        # Restoring a persisted seed may change the SMILES representation. Check
        # the actual restored input, not only the previous card, before repair.
        repair_policy_trace = None
        if plan.forbidden_substructures and persisted_seed is not None and (docking_error or admet_error):
            metadata, policy_trace, policy = self._screen_substructure(
                plan, recorder, persisted_seed.smiles, card.metadata, repair=True,
            )
            trace_ids = self._dedupe([*trace_ids, policy_trace])
            repair_policy_trace = policy_trace
            card = card.model_copy(update={"metadata": metadata, "tool_trace_ids": trace_ids}, deep=True)
            if policy["blocked"]:
                return card.model_copy(update={
                    "status": "excluded", "score": 0.0, "rank": None,
                    "target_rank": None, "portfolio_rank": None, "portfolio_score": None,
                    "safety_flags": self._dedupe([*card.safety_flags,
                        "forbidden_substructure" if policy["matched"] else
                        "invalid_structure" if policy["status"] == "invalid_structure" else "chemical_policy_check_unavailable"]),
                    "exclusion_reasons": self._dedupe([*card.exclusion_reasons, *policy["reason_codes"]]),
                }), 0

        if activity_error:
            activity_task = self._task_id(plan, CandidateRoute.REPURPOSING, "repurpos", "search")
            activities, activity_trace = recorder.invoke(
                task_id=activity_task,
                tool_name="repurposing_search",
                operation="restore_activity_units",
                inputs={"candidate_id": previous.candidate_id, "name": previous.name},
                function=lambda: self.gateway.restore_activity(
                    candidate_name=previous.name,
                    target=target,
                ),
            )
            card = card.model_copy(
                update={
                    "activities": list(activities),
                    "tool_trace_ids": self._dedupe([*trace_ids, activity_trace]),
                }
            )

        if direction_error:
            card = card.model_copy(update={"mechanism": f"{direction} {target.symbol}"})

        if docking_error:
            if persisted_seed is None:  # pragma: no cover - guarded above
                raise RuntimeError("persisted structure was not validated")
            docking_task = self._task_id(plan, previous.route, "dock")
            docking, docking_trace = recorder.invoke(
                task_id=docking_task,
                tool_name="docking",
                operation="rescore_binding_pose",
                chemical_policy_screen_id=repair_policy_trace,
                inputs={
                    "candidate_id": previous.candidate_id,
                    "smiles": persisted_seed.smiles,
                    "target": target,
                },
                function=lambda: self.gateway.dock(
                    smiles=persisted_seed.smiles,
                    target=target,
                ),
            )
            card = card.model_copy(
                update={
                    "metrics": card.metrics.model_copy(
                        update={"docking_score_kcal_mol": docking}
                    ),
                    "tool_trace_ids": self._dedupe([*card.tool_trace_ids, docking_trace]),
                }
            )

        if admet_error:
            if persisted_seed is None:  # pragma: no cover - guarded above
                raise RuntimeError("persisted structure was not validated")
            admet_task = self._task_id(plan, previous.route, "admet", "safety")
            admet_inputs = {"smiles": persisted_seed.smiles}
            admet, admet_trace = recorder.invoke(
                task_id=admet_task,
                tool_name="admet_predictor",
                operation="repeat_six_endpoint_panel",
                inputs=admet_inputs,
                chemical_policy_screen_id=repair_policy_trace,
                function=lambda: self._admet_result(
                    smiles=persisted_seed.smiles,
                    input_hash=_fingerprint(admet_inputs),
                ),
            )
            if admet.status is ScientificExecutionStatus.COMPLETED:
                metrics = card.metrics.model_copy(
                    update={"admet_scores": dict(admet.scores)}
                )
                flags, exclusions = self._risk_flags(metrics)
                card = card.model_copy(
                    update={
                        "metrics": metrics,
                        "admet_execution": admet,
                        "safety_flags": flags,
                        "status": "excluded" if exclusions else "conditional",
                        "exclusion_reasons": exclusions,
                        "tool_trace_ids": self._dedupe(
                            [*card.tool_trace_ids, admet_trace]
                        ),
                    }
                )
            else:
                reason = admet.reason_code or "SCIENTIFIC_TOOL_UNAVAILABLE"
                card = card.model_copy(
                    update={
                        "metrics": card.metrics.model_copy(
                            update={"admet_scores": {}}
                        ),
                        "admet_execution": admet,
                        "safety_flags": self._dedupe(
                            [*card.safety_flags, "scientific_tool_unavailable"]
                        ),
                        "status": "excluded",
                        "exclusion_reasons": self._dedupe(
                            [*card.exclusion_reasons, reason]
                        ),
                        "tool_trace_ids": self._dedupe(
                            [*card.tool_trace_ids, admet_trace]
                        ),
                    }
                )

        return card, 1

    def _rank(
        self,
        plan: PlanSpec,
        recorder: _TraceRecorder,
        candidates: Sequence[CandidateCard],
    ) -> list[CandidateCard]:
        activity_rankings = (
            {}
            if self.fixture_mode
            else derive_activity_rankings(candidates)
        )
        scored: list[CandidateCard] = []
        for card in candidates:
            activity_ranking = activity_rankings.get(card.candidate_id)
            metadata = dict(card.metadata)
            if activity_ranking is not None:
                metadata["activity_ranking"] = dict(activity_ranking)
                metadata["activity_score_breakdown"] = (
                    self._live_score_breakdown(card, activity_ranking)
                )
            scored.append(
                card.model_copy(
                    update={
                        "score": self._score(
                            card,
                            activity_ranking=activity_ranking,
                        ),
                        "metadata": metadata,
                    },
                    deep=True,
                )
            )
        paper_native_v5_order = self._paper_native_v5_required()
        paper_native_v6_order = self._paper_native_v6_required()
        paper_native_fixed_order = bool(
            paper_native_v5_order or paper_native_v6_order
        )
        fixed_pool_rank = (
            self._paper_native_v6_fixed_pool_rank
            if paper_native_v6_order
            else self._paper_native_v5_fixed_pool_rank
        )
        ordering_key = (
            lambda card: (
                card.status == "excluded",
                fixed_pool_rank(card),
                card.name,
                card.candidate_id,
            )
        ) if paper_native_fixed_order else (
            lambda card: (
                card.status == "excluded",
                -card.score,
                card.route.value,
                card.name,
                card.candidate_id,
            )
        )
        ordered = sorted(
            scored,
            key=ordering_key,
        )
        summary = [
            {
                "candidate_id": card.candidate_id,
                "route": card.route.value,
                "score": card.score,
                "excluded": card.status == "excluded",
                "activity_ranking": card.metadata.get("activity_ranking"),
                "fixed_pool_rank": (
                    fixed_pool_rank(card)
                    if paper_native_fixed_order
                    else None
                ),
                "score_used_for_ordering": not paper_native_fixed_order,
            }
            for card in ordered
        ]
        judge_task = self._task_id(plan, None, "Rank policy", "rank")
        _, judge_trace = recorder.invoke(
            task_id=judge_task,
            tool_name="candidate_judge",
            operation="deterministic_multi_objective_ranking",
            inputs=summary,
            function=lambda: {
                "eligible_ordered_candidate_ids": [
                    item["candidate_id"] for item in summary if not item["excluded"]
                ],
                "excluded_candidate_ids": [
                    item["candidate_id"] for item in summary if item["excluded"]
                ],
            },
        )
        route_priority_count: dict[CandidateRoute, int] = {
            CandidateRoute.REPURPOSING: 0,
            CandidateRoute.DE_NOVO: 0,
        }
        ranked: list[CandidateCard] = []
        eligible_rank = 0
        for card in ordered:
            status = card.status
            rank: int | None = None
            if status != "excluded":
                eligible_rank += 1
                rank = eligible_rank
            has_predictive_safety_warning = bool(
                _PREDICTIVE_ADMET_WARNING_FLAGS.intersection(card.safety_flags)
            )
            if (
                status != "excluded"
                and not has_predictive_safety_warning
                and route_priority_count[card.route] < 5
            ):
                status = "priority"
                route_priority_count[card.route] += 1
            elif status != "excluded":
                status = "conditional"
            ranked.append(
                card.model_copy(
                    update={
                        "rank": rank,
                        "status": status,
                        "tool_trace_ids": self._dedupe([*card.tool_trace_ids, judge_trace]),
                        "metadata": {
                            **card.metadata,
                            "ranking_missing_value_policy": (
                                "fixture_legacy_defaults"
                                if self.fixture_mode
                            else "available_dimensions_only_no_neutral_imputation"
                            ),
                            "candidate_ordering_policy": (
                                "fixed_prewarning_pool_order_v1"
                                if paper_native_fixed_order
                                else "deterministic_multi_objective_score_v1"
                            ),
                            "score_used_for_candidate_ordering": (
                                not paper_native_fixed_order
                            ),
                        },
                    }
                )
            )
        return self._attest_final_policy_eligibility(ranked)

    @staticmethod
    def _paper_native_v5_fixed_pool_rank(card: CandidateCard) -> int:
        candidate_id = str(
            card.metadata.get("chembl_id")
            or card.metadata.get("drug_id")
            or ""
        ).strip()
        bounds = card.metadata.get("candidate_discovery_bounds")
        pool = (
            bounds.get("preactivity_pool")
            if isinstance(bounds, Mapping)
            else None
        )
        if isinstance(pool, list):
            for row in pool:
                if (
                    isinstance(row, Mapping)
                    and row.get("parent_molecule_chembl_id") == candidate_id
                    and type(row.get("rank")) is int
                    and int(row["rank"]) > 0
                ):
                    return int(row["rank"])
        return 10**9

    @staticmethod
    def _paper_native_v6_fixed_pool_rank(card: CandidateCard) -> int:
        envelope = card.metadata.get("primary_abstract_paper_native_v6")
        if isinstance(envelope, Mapping):
            rank = envelope.get("fixed_pool_rank")
            if type(rank) is int and rank > 0:
                return rank
        return 10**9

    @staticmethod
    def _activity_comparability_summary(
        candidates: Sequence[CandidateCard],
    ) -> dict[str, Any]:
        rankings = {
            candidate.candidate_id: dict(candidate.metadata["activity_ranking"])
            for candidate in candidates
            if isinstance(candidate.metadata.get("activity_ranking"), Mapping)
        }
        return comparability_summary(rankings)

    @staticmethod
    def _merge_metrics(
        chemistry: ChemistryExecutionResult,
        docking: DockingExecutionResult,
        admet: ADMETExecutionResult,
    ) -> CandidateMetrics:
        return chemistry.metrics.model_copy(
            update={
                "docking_score_kcal_mol": docking.score_kcal_mol,
                "admet_scores": dict(admet.scores),
            }
        )

    @staticmethod
    def _risk_flags(metrics: CandidateMetrics) -> tuple[list[str], list[str]]:
        """Separate model-predicted liabilities from deterministic exclusions.

        ADMET-AI outputs are screening predictions from molecular structure, not
        confirmatory clinical or experimental safety findings.  High predicted
        DILI/hERG/AMES/CYP/clearance values therefore remain visible risk flags
        and keep a candidate conditional, but cannot by themselves erase a
        measured target-active candidate.  Hard exclusion is reserved here for
        deterministic structural bounds; forbidden/withdrawn drug policy is
        enforced independently before computation.
        """

        flags: list[str] = []
        exclusions: list[str] = []
        if metrics.molecular_weight is not None and not 120 <= metrics.molecular_weight <= 550:
            flags.append("molecular_weight_out_of_range")
        if metrics.clogp is not None and not -1.5 <= metrics.clogp <= 5.0:
            flags.append("clogp_out_of_range")
        if metrics.tpsa is not None and metrics.tpsa > 160:
            flags.append("tpsa_high")
        thresholds = {
            "hepatotoxicity": (0.55, "hepatotoxicity_high"),
            "herg_risk": (0.55, "herg_risk_high"),
            "ames_risk": (0.55, "ames_risk_high"),
            "cyp3a4_inhibition": (0.65, "cyp3a4_inhibition_high"),
            "clearance_risk": (0.65, "clearance_risk_high"),
        }
        for endpoint, (threshold, label) in thresholds.items():
            if metrics.admet_scores.get(endpoint, 0.0) >= threshold:
                flags.append(label)
        if metrics.molecular_weight is not None and metrics.molecular_weight > 750:
            exclusions.append("molecular_weight_hard_limit")
        return sorted(set(flags)), sorted(set(exclusions))

    @staticmethod
    def _ood_score(metrics: CandidateMetrics) -> float:
        """Return a transparent descriptor-distance proxy, not calibrated OOD.

        This legacy scalar remains in ``CandidateMetrics`` for deterministic
        ranking compatibility.  The accompanying candidate metadata declares
        that it is not a learned applicability-domain or confidence score.
        """

        penalties = []
        if metrics.molecular_weight is not None:
            penalties.append(_clamp(abs(metrics.molecular_weight - 335.0) / 500.0))
        if metrics.clogp is not None:
            penalties.append(_clamp(abs(metrics.clogp - 2.2) / 7.0))
        if metrics.tpsa is not None:
            penalties.append(_clamp(abs(metrics.tpsa - 65.0) / 180.0))
        return round(sum(penalties) / len(penalties), 3) if penalties else 1.0

    @staticmethod
    def _applicability_assessment(metrics: CandidateMetrics) -> dict[str, Any]:
        score = metrics.ood_score
        if score is None:
            status = "unavailable"
        elif score >= 0.85:
            status = "high_descriptor_distance"
        elif score >= 0.60:
            status = "boundary_descriptor_distance"
        else:
            status = "low_descriptor_distance"
        observed = [
            name
            for name, value in (
                ("molecular_weight", metrics.molecular_weight),
                ("clogp", metrics.clogp),
                ("tpsa", metrics.tpsa),
            )
            if value is not None
        ]
        return {
            "schema_version": "1.0",
            "method": "mw-clogp-tpsa-descriptor-distance-proxy-v1",
            "score": score,
            "status": status,
            "calibrated": False,
            "model_domain_claim": False,
            "observed_descriptor_count": len(observed),
            "observed_descriptors": observed,
            "nearest_training_compound": None,
            "prediction_interval": None,
            "reason_code": "UNCERTAINTY_NOT_CALIBRATED",
            "limitations": (
                "A simple descriptor-distance ranking signal; not a learned "
                "applicability-domain or confidence estimate."
            ),
            "missing_values_imputed": False,
        }

    def _score(
        self,
        card: CandidateCard,
        *,
        activity_ranking: Mapping[str, Any] | None = None,
    ) -> float:
        if card.status == "excluded":
            return 0.0
        if not self.fixture_mode:
            return self._score_live_without_imputation(
                card,
                activity_ranking=activity_ranking,
            )
        docking = card.metrics.docking_score_kcal_mol
        docking_score = _clamp((-(docking or -5.0) - 5.0) / 5.0)
        risk_keys = (
            "hepatotoxicity",
            "herg_risk",
            "ames_risk",
            "cyp3a4_inhibition",
            "clearance_risk",
        )
        risks = [card.metrics.admet_scores.get(key, 0.5) for key in risk_keys]
        safety_score = 1.0 - sum(risks) / len(risks)
        bioavailability = card.metrics.admet_scores.get("oral_bioavailability", 0.5)
        sa = card.metrics.synthetic_accessibility
        developability = 0.5 * bioavailability + 0.5 * (
            1.0 - _clamp(((sa if sa is not None else 5.5) - 1.0) / 9.0)
        )
        if card.activities:
            unit_factor = {"pM": 0.001, "nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0}
            best_nm = min(activity.value * unit_factor[activity.unit] for activity in card.activities)
            activity_score = 1.0 - _clamp(math.log10(max(1.0, best_nm)) / 5.0)
        else:
            activity_score = docking_score * 0.8
        evidence_count = int(
            evidence_scoring_cardinality(card.evidence)[
                "unique_scientific_record_count"
            ]
        )
        evidence_score = min(1.0, evidence_count / 2.0)
        warning_penalty = min(0.24, 0.04 * len(card.safety_flags))
        score = (
            0.28 * activity_score
            + 0.25 * docking_score
            + 0.25 * safety_score
            + 0.12 * developability
            + 0.10 * evidence_score
            - warning_penalty
        )
        return round(_clamp(score), 6)

    @staticmethod
    def _live_score_breakdown(
        card: CandidateCard,
        activity_ranking: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the live score inputs with activity semantics made explicit."""

        if card.status == "excluded":
            return {
                "schema_version": "live_score_breakdown_v3",
                "components": [],
                "evidence_cardinality": evidence_scoring_cardinality(
                    card.evidence
                ),
                "numerator": 0.0,
                "denominator": 0.0,
                "warning_penalty": 0.0,
                "final_score": 0.0,
                "excluded": True,
            }

        resolved_activity = (
            dict(activity_ranking)
            if activity_ranking is not None
            else derive_activity_rankings([card]).get(card.candidate_id, {})
        )
        weighted: list[tuple[str, float, float]] = []
        if card.activities:
            activity_value = float(
                resolved_activity.get(
                    "activity_component_value",
                    measurement_evidence_quality(card),
                )
            )
            weighted.append(("activity", 0.28, _clamp(activity_value)))
        docking = card.metrics.docking_score_kcal_mol
        if docking is not None:
            weighted.append(
                ("docking", 0.25, _clamp((-docking - 5.0) / 5.0))
            )

        risk_keys = (
            "hepatotoxicity",
            "herg_risk",
            "ames_risk",
            "cyp3a4_inhibition",
            "clearance_risk",
        )
        if all(key in card.metrics.admet_scores for key in risk_keys):
            risks = [card.metrics.admet_scores[key] for key in risk_keys]
            weighted.append(
                ("predicted_safety", 0.25, 1.0 - sum(risks) / len(risks))
            )

        bioavailability = card.metrics.admet_scores.get("oral_bioavailability")
        sa = card.metrics.synthetic_accessibility
        if bioavailability is not None and sa is not None:
            developability = 0.5 * bioavailability + 0.5 * (
                1.0 - _clamp((sa - 1.0) / 9.0)
            )
            weighted.append(("developability", 0.12, developability))
        evidence_cardinality = evidence_scoring_cardinality(card.evidence)
        unique_evidence_count = int(
            evidence_cardinality["unique_scientific_record_count"]
        )
        if unique_evidence_count:
            weighted.append(
                (
                    "evidence_count",
                    0.10,
                    min(1.0, unique_evidence_count / 2.0),
                )
            )
        # The disclosed component values are the auditable score inputs.  Use
        # those exact six-decimal values for the numerator so an independent
        # recomputation cannot differ from a hidden higher-precision value at
        # a rounding boundary.
        disclosed_weighted = [
            (name, weight, round(value, 6))
            for name, weight, value in weighted
        ]
        numerator = sum(
            weight * value for _name, weight, value in disclosed_weighted
        )
        denominator = sum(
            weight for _name, weight, _value in disclosed_weighted
        )
        warning_penalty = min(0.24, 0.04 * len(card.safety_flags))
        final_score = (
            round(_clamp(numerator / denominator - warning_penalty), 6)
            if denominator
            else 0.0
        )
        return {
            "schema_version": "live_score_breakdown_v3",
            "components": [
                {"name": name, "weight": weight, "value": value}
                for name, weight, value in disclosed_weighted
            ],
            "activity_ranking": resolved_activity,
            "evidence_cardinality": evidence_cardinality,
            "numerator": round(numerator, 9),
            "denominator": round(denominator, 9),
            "warning_penalty": round(warning_penalty, 9),
            "final_score": final_score,
            "excluded": False,
        }

    @staticmethod
    def _score_live_without_imputation(
        card: CandidateCard,
        activity_ranking: Mapping[str, Any] | None = None,
    ) -> float:
        """Rank observed values without cross-assay potency or pseudo-values."""

        return float(
            MA2Agent._live_score_breakdown(
                card,
                activity_ranking,
            )["final_score"]
        )

    @staticmethod
    def _primary_hypothesis(plan: PlanSpec, packet: HypothesisPacket) -> Hypothesis:
        hypothesis, _ = MA2Agent._select_primary_hypothesis(plan, packet)
        return hypothesis

    @staticmethod
    def _select_primary_hypothesis(
        plan: PlanSpec,
        packet: HypothesisPacket,
    ) -> tuple[Hypothesis, dict[str, Any]]:
        if plan.hypothesis_packet_id != packet.packet_id:
            raise ValueError("PlanSpec does not reference the supplied HypothesisPacket")
        if not packet.hypotheses:
            raise ValueError("MA2 requires at least one hypothesis")
        return select_hypothesis_for_priority(
            packet.hypotheses,
            plan.discovery_priority,
        )

    @staticmethod
    def _task_id(
        plan: PlanSpec,
        route: CandidateRoute | None,
        *name_hints: str,
    ) -> str:
        if route is not None:
            routed = [task for task in plan.tasks if task.route is route]
            for hint in name_hints:
                match = next(
                    (
                        task
                        for task in routed
                        if hint.lower() in task.name.lower()
                        or any(hint.lower() in tool.tool_name.lower() for tool in task.tools)
                    ),
                    None,
                )
                if match:
                    return match.task_id
        for hint in name_hints:
            match = next(
                (
                    task
                    for task in plan.tasks
                    if hint.lower() in task.name.lower()
                    or any(hint.lower() in tool.tool_name.lower() for tool in task.tools)
                ),
                None,
            )
            if match:
                return match.task_id
        if route is not None and routed:
            return routed[0].task_id
        if not plan.tasks:
            raise ValueError("PlanSpec has no executable tasks")
        return plan.tasks[0].task_id

    @staticmethod
    def _repair_trace_task_ids(
        plan: PlanSpec,
        previous: CandidatePacket,
    ) -> list[ToolCallRecord]:
        valid = {task.task_id for task in plan.tasks}
        fallback = plan.tasks[0].task_id
        operation_to_task: dict[tuple[str, str], str] = {}
        tool_to_task: dict[str, str] = {}
        for task in plan.tasks:
            for assignment in task.tools:
                tool_to_task.setdefault(assignment.tool_name, task.task_id)
                for operation in assignment.allowed_operations:
                    operation_to_task[(assignment.tool_name, operation)] = task.task_id
        return [
            call.model_copy(
                update={
                    "task_id": operation_to_task.get(
                        (call.tool_name, call.operation),
                        tool_to_task.get(call.tool_name, fallback),
                    )
                }
            )
            if call.task_id is not None and call.task_id not in valid
            else call.model_copy(deep=True)
            for call in previous.tool_calls
        ]

    @staticmethod
    def _inject_fault(
        fault: FaultType,
        candidates: list[CandidateCard],
        calls: list[ToolCallRecord],
    ) -> tuple[list[CandidateCard], list[ToolCallRecord]]:
        if fault is FaultType.NONE:
            return candidates, calls
        mutated = [candidate.model_copy(deep=True) for candidate in candidates]
        mutated_calls = [call.model_copy(deep=True) for call in calls]
        rep_index = next(
            (i for i, card in enumerate(mutated) if card.route is CandidateRoute.REPURPOSING),
            0,
        )
        target = mutated[rep_index]
        if fault is FaultType.G1_IC50_UNIT:
            activities = list(target.activities)
            if activities:
                original = activities[0]
                activities[0] = original.model_copy(
                    update={"endpoint": "IC50", "value": 2_500_000.0, "unit": "nM"}
                )
            else:
                activities = [
                    ActivityMeasurement(
                        endpoint="IC50",
                        value=2_500_000.0,
                        unit="nM",
                        assay_type="fault-injected activity",
                    )
                ]
            mutated[rep_index] = target.model_copy(update={"activities": activities})
        elif fault is FaultType.G1_INVALID_SMILES:
            mutated[rep_index] = target.model_copy(update={"smiles": "NOT_A_SMILES?"})
        elif fault is FaultType.G1_DIRECTION_MISMATCH:
            current = target.mechanism.lower()
            wrong = "activate" if "inhibit" in current else "inhibit"
            mutated[rep_index] = target.model_copy(
                update={"mechanism": f"{wrong} {target.target.symbol}"}
            )
        elif fault is FaultType.G1_PLAN_TRACE_MISMATCH:
            referenced = set(target.tool_trace_ids)
            index = next(
                (i for i, call in enumerate(mutated_calls) if call.call_id in referenced),
                0,
            )
            mutated_calls[index] = mutated_calls[index].model_copy(
                update={"task_id": "task_fault_not_in_plan"}
            )
        return mutated, mutated_calls

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _dedupe_evidence(values: Iterable[EvidenceRef]) -> list[EvidenceRef]:
        return list({item.evidence_id: item for item in values}.values())


__all__ = ["MA2Agent"]
