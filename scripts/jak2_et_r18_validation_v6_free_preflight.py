"""Fixture-free, no-model free preflight for the r18 JAK2/ET packet.

The live entry point owns its read-only public-source adapters.  It never
imports the API application, opens the run database, invokes a model, submits
a run, or accepts candidate/PMID/fixture/runtime injection from the CLI.

The protocol intentionally ships with a pending production-source register.
Until the root audit freezes the final r18 bytes and pins the resulting
protocol hash below, the live path fails before constructing any HTTP client.
Synthetic runtimes exist only behind :func:`run_contract_test`; their reports
are permanently labelled ``CONTRACT_TEST_ONLY`` and never authorize spending.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
from time import monotonic
from typing import Any, Literal, NoReturn, TextIO
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pydantic import BaseModel, Field, model_validator  # noqa: E402

from drugpilot.config import Settings  # noqa: E402
from drugpilot.domain.models import (  # noqa: E402
    FaultType,
    HypothesisDirection,
    ResearchDepth,
    ReservedControlCompound,
    validate_hypothesis_direction_selection,
)
from drugpilot.tools.http import ScientificHttpClient  # noqa: E402


SCHEMA_VERSION = "jak2_et_r18_validation_v6_free_preflight/v1"
FREEZE_SCHEMA_VERSION = "jak2_et_r18_validation_v6_freeze/v1"
DEFAULT_REQUEST = (
    ROOT
    / "docs"
    / "validation"
    / "JAK2_ET_PAPER_NATIVE_V5_LIVE_E2E_REQUEST_R3_2026-08-28.json"
)
DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "validation"
    / "JAK2_ET_R18_VALIDATION_V6_PROTOCOL_KO_R3_2026-09-01.md"
)
EXPECTED_REQUEST_SHA256 = (
    "97139e34d6f7829dce99ee0d68af23dc0253d4f3ffdff97dd5dd80f458d82210"
)
# Pinned only after the root audit froze the final source/runtime manifest and
# the isolated full regression passed. The normalized harness hash replaces
# this assignment value with a placeholder to avoid a protocol/script cycle.
EXPECTED_PROTOCOL_SHA256: str | None = "61c863b81d14be5db599c03b657a35f302a158aefda97b626e5f3c9aae48b262"
PAPER_NATIVE_V5_POLICY_TOKEN = (
    "dynamic_pubmed_primary_abstract_paper_native_v5_required"
)

FREEZE_BEGIN = "<!-- R18_FREEZE_MANIFEST_BEGIN -->"
FREEZE_END = "<!-- R18_FREEZE_MANIFEST_END -->"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
RESULT_NAME_RE = re.compile(
    r"^JAK2_ET_R18_VALIDATION_V6_FREE_PREFLIGHT_R3_"
    r"\d{8}T\d{6}Z\.json$"
)
TARGET_SYMBOL = "JAK2"
TARGET_UNIPROT = "O60674"
MINIMUM_NONCONTROL_CANDIDATES = 3
ACTIVITY_PAGE_SIZE = 20
MECHANISM_ROW_LIMIT = 80
RETRY_LEDGER_LIMIT = 64
FIXED_POOL_LIMIT = 12
CHEMBL_ID_RE = re.compile(r"^CHEMBL[1-9][0-9]*$")

CRITICAL_SOURCE_PATHS = (
    "src/drugpilot/config.py",
    "src/drugpilot/api/app.py",
    "src/drugpilot/api/call_trace.py",
    "src/drugpilot/domain/models.py",
    "src/drugpilot/storage/sqlite.py",
    "src/drugpilot/orchestration/graph.py",
    "src/drugpilot/orchestration/budget.py",
    "src/drugpilot/orchestration/goal_focus.py",
    "src/drugpilot/orchestration/tool_coverage.py",
    "src/drugpilot/agents/ma1.py",
    "src/drugpilot/agents/planner.py",
    "src/drugpilot/agents/ma2.py",
    "src/drugpilot/agents/ma3.py",
    "src/drugpilot/agents/ma3_reviewer_projection.py",
    "src/drugpilot/tools/public_biology.py",
    "src/drugpilot/tools/europe_pmc_oa_fulltext.py",
    "src/drugpilot/tools/disease_targets.py",
    "src/drugpilot/tools/target_identity.py",
    "src/drugpilot/tools/http.py",
    "src/drugpilot/tools/source_catalog.py",
    "src/drugpilot/tools/live_ma2_gateway.py",
    "src/drugpilot/tools/live_repurposing_gateway.py",
    "src/drugpilot/tools/primary_abstract_paper_native_v5.py",
    "src/drugpilot/tools/primary_abstract_paper_native_search_v5.py",
    "src/drugpilot/tools/candidate_policy.py",
    "src/drugpilot/tools/activity_comparability.py",
    "src/drugpilot/tools/assay_variant_alignment.py",
    "src/drugpilot/tools/portfolio_ranking.py",
    "src/drugpilot/tools/admet_ai_adapter.py",
    "src/drugpilot/tools/chemistry.py",
    "src/drugpilot/providers/llm.py",
    "src/drugpilot/web/static/app.js",
    "src/drugpilot/web/static/app.css",
    "src/drugpilot/web/templates/index.html",
)
HARNESS_PATHS = (
    "scripts/jak2_et_r18_validation_v6_free_preflight.py",
    "tests/unit/test_jak2_et_r18_validation_v6_free_preflight.py",
)
RESULTS_DIR = (ROOT / "docs" / "validation" / "results").resolve()
ALLOWED_HTTP_HOSTS = frozenset(
    {
        "www.ebi.ac.uk",
        "eutils.ncbi.nlm.nih.gov",
        "www.uniprot.org",
        "rest.uniprot.org",
        "api.platform.opentargets.org",
        "dgidb.org",
    }
)
SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "access_token",
)
FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "candidate_id",
        "candidate_ids",
        "candidate_name",
        "candidate_names",
        "pmid",
        "pmids",
        "pmcid",
        "pmcids",
        "external_id",
        "external_ids",
        "abstract",
        "full_abstract",
        "excerpt",
        "response_body",
        "raw_xml",
        "full_text",
    }
)
SENSITIVE_STRING_RE = re.compile(
    r"(?i)(?:\bbearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}|"
    r"(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*\S+)"
)
SENSITIZATION_RE = re.compile(
    r"(?is)\b(?:increase\w*|enhance\w*)\b.{0,120}"
    r"\bsensitiv\w*\b.{0,100}\bto\b"
)
CANONICAL_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9-]{0,31}$")


class PreflightContractError(ValueError):
    """Bounded fail-closed error with a non-secret reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RuntimeEnvironmentMismatch(PreflightContractError):
    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        super().__init__("RUNTIME_ENVIRONMENT_MISMATCH")
        self.snapshot = deepcopy(dict(snapshot))


@dataclass(frozen=True, slots=True)
class ValidationContext:
    request_path: Path
    request_sha256: str
    protocol_path: Path
    protocol_sha256: str
    request_model_ast_sha256: str
    raw_request: dict[str, Any]
    registry: dict[str, Any]
    production_freeze: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _safe_endpoint(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    path = re.sub(
        r"(?i)(?<=/)(?:PMC[0-9]+|CHEMBL[0-9]+|[0-9]{5,})(?=/|$)",
        "{record}",
        parsed.path,
    )
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    try:
        return _canonical_json(response.json())
    except Exception:
        return b""


def _failure_diagnostics(values: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        source = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        if not isinstance(source, Mapping):
            continue
        row = {
            key: source.get(key)
            for key in (
                "method",
                "endpoint",
                "category",
                "attempt",
                "max_attempts",
                "status_code",
                "retrying",
            )
            if source.get(key) is not None
        }
        if row:
            result.append(row)
    return result


class AuditedScientificHttpClient(ScientificHttpClient):
    """Production transport with a bounded body-free request ledger.

    Subclassing, rather than wrapping, is material: Europe PMC OA checks for
    ``ScientificHttpClient`` and then uses its pooled one-shot stream so the
    production byte cap remains enforced while its own OA ledger is retained.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        super().__init__(**kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        method = method.upper()
        started = monotonic()
        failures_before = len(self.recent_failures())
        parameters: dict[str, Any] = {}
        parameter_keys: list[str] = []
        for container in ("params", "json", "data"):
            value = kwargs.get(container)
            if not isinstance(value, Mapping):
                continue
            clean: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).casefold()
                clean[str(key)] = (
                    "[REDACTED]"
                    if any(marker in normalized for marker in SENSITIVE_KEY_MARKERS)
                    else item
                )
                parameter_keys.append(f"{container}.{key}")
            parameters[container] = clean
        base = {
            "method": method,
            "endpoint": _safe_endpoint(url),
            "parameter_keys": sorted(parameter_keys),
            "parameter_sha256": _canonical_sha256(parameters),
            "fixture": False,
        }
        try:
            response = super().request(method, url, **kwargs)
        except Exception as exc:
            failures = self.recent_failures()
            self.calls.append(
                {
                    **base,
                    "status": "failed",
                    "status_code": None,
                    "response_sha256": None,
                    "attempt_count_observed": max(
                        1, len(failures) - failures_before
                    ),
                    "retry_diagnostics": _failure_diagnostics(
                        failures[failures_before:]
                    ),
                    "exception_type": type(exc).__name__,
                    "runtime_seconds": round(monotonic() - started, 6),
                }
            )
            raise
        failures = self.recent_failures()
        status = getattr(response, "status_code", None)
        self.calls.append(
            {
                **base,
                "status": "completed",
                "status_code": status if isinstance(status, int) else None,
                "response_sha256": sha256(_response_bytes(response)).hexdigest(),
                "attempt_count_observed": max(
                    1, len(failures) - failures_before + 1
                ),
                "retry_diagnostics": _failure_diagnostics(
                    failures[failures_before:]
                ),
                "runtime_seconds": round(monotonic() - started, 6),
            }
        )
        return response


class _NoDockingPreflightBackend:
    source_version = "docking=not_invoked_free_preflight"
    last_result = None

    def readiness(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "tool": "docking",
            "status": "not_configured",
            "required": False,
            "fixture": False,
        }

    def dock(self, *_: Any, **__: Any) -> float:
        raise RuntimeError("docking is outside the free-preflight boundary")


def _extract_create_run_request(raw: Mapping[str, Any]) -> str:
    """Validate via an AST-extracted class without importing the API app."""

    app_path = ROOT / "src" / "drugpilot" / "api" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CreateRunRequest"
    ]
    if len(nodes) != 1:
        raise PreflightContractError("CREATE_RUN_REQUEST_AST_NODE_MISMATCH")
    module = ast.Module(body=[nodes[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "__name__": "_r18_v6_preflight_request_schema",
        "BaseModel": BaseModel,
        "Field": Field,
        "model_validator": model_validator,
        "Literal": Literal,
        "HypothesisDirection": HypothesisDirection,
        "validate_hypothesis_direction_selection": validate_hypothesis_direction_selection,
        "ResearchDepth": ResearchDepth,
        "ReservedControlCompound": ReservedControlCompound,
        "FaultType": FaultType,
    }
    exec(compile(module, str(app_path), "exec"), namespace)
    model = namespace.get("CreateRunRequest")
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise PreflightContractError("CREATE_RUN_REQUEST_AST_EXECUTION_FAILED")
    unknown = sorted(set(raw).difference(model.model_fields))
    if unknown:
        raise PreflightContractError("CREATE_RUN_REQUEST_UNKNOWN_KEYS")
    try:
        model.model_rebuild(_types_namespace=namespace)
        model.model_validate(dict(raw))
    except Exception as exc:
        raise PreflightContractError("CREATE_RUN_REQUEST_VALIDATION_FAILED") from exc
    return sha256(
        ast.dump(nodes[0], include_attributes=False).encode("utf-8")
    ).hexdigest()


def _parse_freeze_registry(protocol_text: str) -> dict[str, Any]:
    if protocol_text.count(FREEZE_BEGIN) != 1 or protocol_text.count(FREEZE_END) != 1:
        raise PreflightContractError("PROTOCOL_FREEZE_REGISTER_MARKER_MISMATCH")
    start = protocol_text.index(FREEZE_BEGIN) + len(FREEZE_BEGIN)
    end = protocol_text.index(FREEZE_END, start)
    body = protocol_text[start:end].strip()
    if not body.startswith("```json") or not body.endswith("```"):
        raise PreflightContractError("PROTOCOL_FREEZE_REGISTER_FENCE_MISMATCH")
    payload = body[len("```json") : -len("```")].strip()
    try:
        registry = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PreflightContractError("PROTOCOL_FREEZE_REGISTER_INVALID_JSON") from exc
    if not isinstance(registry, dict):
        raise PreflightContractError("PROTOCOL_FREEZE_REGISTER_INVALID")
    return registry


def _production_tree_manifest() -> dict[str, Any]:
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (ROOT / "src" / "drugpilot").rglob("*.py"),
            key=lambda item: item.relative_to(ROOT).as_posix(),
        )
    ]
    return {
        "file_count": len(records),
        "manifest_sha256": _canonical_sha256(records),
    }


def _source_freeze_audit(registry: Mapping[str, Any]) -> dict[str, Any]:
    if registry.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise PreflightContractError("SOURCE_FREEZE_SCHEMA_MISMATCH")
    if registry.get("status") != "FROZEN_FINAL_R18_SOURCE":
        raise PreflightContractError("SOURCE_FREEZE_PENDING")
    source = _mapping(registry.get("production_source_freeze"))
    rows = _sequence(source.get("critical_files"))
    if len(rows) != len(CRITICAL_SOURCE_PATHS):
        raise PreflightContractError("SOURCE_FREEZE_CRITICAL_COUNT_MISMATCH")
    observed: dict[str, str] = {}
    for row in rows:
        item = _mapping(row)
        path = str(item.get("path") or "")
        digest = str(item.get("sha256") or "").casefold()
        if path in observed or not HASH_RE.fullmatch(digest):
            raise PreflightContractError("SOURCE_FREEZE_CRITICAL_ROW_INVALID")
        observed[path] = digest
    if tuple(observed) != CRITICAL_SOURCE_PATHS:
        raise PreflightContractError("SOURCE_FREEZE_CRITICAL_PATH_SET_MISMATCH")
    records: list[dict[str, str]] = []
    for relative in CRITICAL_SOURCE_PATHS:
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PreflightContractError("SOURCE_FREEZE_PATH_ESCAPE") from exc
        if not path.is_file() or _sha256_file(path) != observed[relative]:
            raise PreflightContractError("SOURCE_FREEZE_HASH_MISMATCH")
        records.append({"path": relative, "sha256": observed[relative]})
    tree = _production_tree_manifest()
    expected_count = source.get("python_tree_file_count")
    expected_hash = str(source.get("python_tree_manifest_sha256") or "").casefold()
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
        or not HASH_RE.fullmatch(expected_hash)
    ):
        raise PreflightContractError("SOURCE_FREEZE_TREE_PENDING")
    if tree["file_count"] != expected_count or tree["manifest_sha256"] != expected_hash:
        raise PreflightContractError("SOURCE_FREEZE_TREE_MISMATCH")
    return {
        "verified": True,
        "critical_file_count": len(records),
        "critical_manifest_sha256": _canonical_sha256(records),
        "python_tree_file_count": tree["file_count"],
        "python_tree_manifest_sha256": tree["manifest_sha256"],
    }


def _runtime_snapshot(settings: Settings) -> dict[str, Any]:
    return {
        "build_variant": settings.build_variant,
        "configured_mode": settings.mode.value,
        "effective_mode": settings.effective_mode.value,
        "mode_auto_activated": settings.openai_auto_activated,
        "ma2_backend_mode": settings.ma2_backend_mode,
        "ma2_live_scientific_tools_enabled": settings.ma2_live_scientific_tools_enabled,
        "ma2_live_allow_experimental_docking_profiles": (
            settings.ma2_live_allow_experimental_docking_profiles
        ),
        "ma2_portfolio_enabled": settings.ma2_portfolio_enabled,
        "ma2_chembl_timeout_seconds": settings.ma2_chembl_timeout_seconds,
        "ma2_chembl_operation_timeout_seconds": (
            settings.ma2_chembl_operation_timeout_seconds
        ),
        "ma2_chembl_max_attempts": settings.ma2_chembl_max_attempts,
        "public_evidence_enabled": settings.public_evidence_enabled,
        "evidence_research_depth": settings.evidence_research_depth,
        "public_evidence_max_records_per_source": (
            settings.public_evidence_max_records_per_source
        ),
        "public_evidence_max_source_calls_per_target": (
            settings.public_evidence_max_source_calls_per_target
        ),
        "public_evidence_max_elapsed_seconds_per_target": (
            settings.public_evidence_max_elapsed_seconds_per_target
        ),
        "public_evidence_max_attempts": settings.public_evidence_max_attempts,
        "epmc_oa_fulltext_one_hop_v1": settings.epmc_oa_fulltext_one_hop_v1,
        "dynamic_target_discovery_enabled": settings.dynamic_target_discovery_enabled,
        "dynamic_target_max_candidates": settings.dynamic_target_max_candidates,
        "ma1_supplementary_evidence_retry_budget": (
            settings.ma1_supplementary_evidence_retry_budget
        ),
        "dynamic_target_fallback_enabled": settings.dynamic_target_fallback_enabled,
        "demo_fixture_fallback_enabled": settings.demo_fixture_fallback_enabled,
        "max_llm_calls": settings.max_llm_calls,
        "max_run_tokens": settings.max_run_tokens,
        "hard_max_run_tokens": settings.hard_max_run_tokens,
        "max_single_input_tokens": settings.max_single_input_tokens,
        "max_run_cost_usd": settings.max_run_cost_usd,
        "candidates_per_route": settings.candidates_per_route,
        "max_ma1_iterations": settings.max_ma1_iterations,
        "max_rework_loops": settings.max_rework_loops,
        "beta_max_active_live_runs": settings.beta_max_active_live_runs,
        "beta_max_new_runs_per_hour": settings.beta_max_new_runs_per_hour,
        "beta_live_lease_ttl_seconds": settings.beta_live_lease_ttl_seconds,
        "beta_live_lease_heartbeat_seconds": (
            settings.beta_live_lease_heartbeat_seconds
        ),
        "idempotency_required": settings.effective_mode.value == "live",
    }


def _runtime_environment_gate(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    expected_copy = deepcopy(dict(expected))
    observed_copy = deepcopy(dict(observed))
    checks = {
        key: key in observed_copy and observed_copy.get(key) == value
        for key, value in expected_copy.items()
    }
    checks["exact_key_set"] = set(expected_copy) == set(observed_copy)
    checks["fixture_fallbacks_disabled"] = (
        observed_copy.get("dynamic_target_fallback_enabled") is False
        and observed_copy.get("demo_fixture_fallback_enabled") is False
    )
    checks["epmc_oa_enabled"] = (
        observed_copy.get("epmc_oa_fulltext_one_hop_v1") is True
    )
    checks["live_backends"] = (
        observed_copy.get("configured_mode") == "live"
        and observed_copy.get("effective_mode") == "live"
        and observed_copy.get("ma2_backend_mode") == "live"
        and observed_copy.get("ma2_live_scientific_tools_enabled") is True
    )
    checks["live_admission_is_bounded"] = bool(
        type(observed_copy.get("beta_max_active_live_runs")) is int
        and observed_copy["beta_max_active_live_runs"] >= 1
        and type(observed_copy.get("beta_max_new_runs_per_hour")) is int
        and observed_copy["beta_max_new_runs_per_hour"] >= 1
        and observed_copy.get("idempotency_required") is True
    )
    checks["lease_heartbeat_precedes_half_ttl"] = bool(
        type(observed_copy.get("beta_live_lease_ttl_seconds")) is int
        and type(observed_copy.get("beta_live_lease_heartbeat_seconds")) is int
        and observed_copy["beta_live_lease_heartbeat_seconds"] * 2
        < observed_copy["beta_live_lease_ttl_seconds"]
    )
    verified = all(checks.values())
    body = {
        "schema_version": "jak2_et_r18_validation_v6_runtime/v1",
        "expected": expected_copy,
        "observed": observed_copy,
        "checks": checks,
        "verified": verified,
        "secret_values_included": False,
    }
    body["snapshot_sha256"] = _canonical_sha256(body)
    return body


def _require_runtime_environment(context: ValidationContext) -> dict[str, Any]:
    expected = _mapping(context.registry.get("runtime"))
    try:
        settings = Settings(_env_file=ROOT / ".env")
    except Exception as exc:
        snapshot = {
            "schema_version": "jak2_et_r18_validation_v6_runtime/v1",
            "expected": deepcopy(dict(expected)),
            "observed": {},
            "checks": {"settings_constructed": False},
            "verified": False,
            "secret_values_included": False,
            "settings_exception_type": type(exc).__name__,
        }
        snapshot["snapshot_sha256"] = _canonical_sha256(snapshot)
        raise RuntimeEnvironmentMismatch(snapshot) from exc
    gate = _runtime_environment_gate(expected, _runtime_snapshot(settings))
    if not gate["verified"]:
        raise RuntimeEnvironmentMismatch(gate)
    return gate


def _build_brief(raw: Mapping[str, Any], request_sha256: str) -> Any:
    from drugpilot.domain.models import ProjectConstraints, ResearchBrief

    controls = [
        ReservedControlCompound.model_validate(item)
        for item in _sequence(raw.get("reserved_control_compounds"))
    ]
    return ResearchBrief(
        brief_id=f"brief_r18_v6_preflight_{request_sha256[:12]}",
        research_goal=str(raw.get("research_goal") or ""),
        disease=str(raw.get("disease") or ""),
        disease_id=raw.get("disease_id"),
        rationale=raw.get("rationale"),
        research_depth=raw.get("research_depth"),
        constraints=ProjectConstraints(
            target_cell_type=raw.get("target_cell_type"),
            intervention_direction=raw.get("intervention_direction", "inhibit"),
            chronic_safety_required=raw.get("chronic_safety_required", True),
            druggability_required=raw.get("druggability_required", True),
            discovery_priority=raw.get("discovery_priority", "auto"),
            forbidden_targets=list(raw.get("forbidden_targets") or []),
            forbidden_substructures=list(raw.get("forbidden_substructures") or []),
            forbidden_drugs=list(raw.get("forbidden_drugs") or []),
            forbidden_drug_ids=list(raw.get("forbidden_drug_ids") or []),
            forbidden_drug_classes=list(raw.get("forbidden_drug_classes") or []),
            reserved_control_compounds=controls,
            candidates_per_route=int(raw.get("candidates_per_route") or 3),
            max_ma1_iterations=int(raw.get("max_ma1_iterations") or 3),
            max_rework_loops=int(raw.get("max_rework_loops") or 0),
        ),
        human_priorities=list(raw.get("human_priorities") or []),
        fault_injection=raw.get("fault_injection", "none"),
    )


def load_validation_context(
    request_path: Path = DEFAULT_REQUEST,
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    expected_protocol_sha256: str | None = EXPECTED_PROTOCOL_SHA256,
) -> ValidationContext:
    request_path = request_path.resolve()
    protocol_path = protocol_path.resolve()
    if expected_protocol_sha256 is None:
        raise PreflightContractError("PROTOCOL_HASH_PENDING")
    if not HASH_RE.fullmatch(expected_protocol_sha256.casefold()):
        raise PreflightContractError("PROTOCOL_HASH_INVALID")
    request_bytes = request_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    request_sha = sha256(request_bytes).hexdigest()
    protocol_sha = sha256(protocol_bytes).hexdigest()
    if request_sha != EXPECTED_REQUEST_SHA256:
        raise PreflightContractError("REQUEST_HASH_MISMATCH")
    if protocol_sha != expected_protocol_sha256.casefold():
        raise PreflightContractError("PROTOCOL_HASH_MISMATCH")
    try:
        raw = json.loads(request_bytes)
    except json.JSONDecodeError as exc:
        raise PreflightContractError("REQUEST_JSON_INVALID") from exc
    if not isinstance(raw, dict):
        raise PreflightContractError("REQUEST_JSON_OBJECT_REQUIRED")
    model_hash = _extract_create_run_request(raw)
    protocol_text = protocol_bytes.decode("utf-8")
    registry = _parse_freeze_registry(protocol_text)
    registered_request = _mapping(registry.get("request"))
    if (
        registered_request.get("path")
        != request_path.relative_to(ROOT).as_posix()
        or str(registered_request.get("sha256") or "").casefold() != request_sha
    ):
        raise PreflightContractError("PROTOCOL_REQUEST_BINDING_MISMATCH")
    if registry.get("create_run_request_ast_sha256") != model_hash:
        raise PreflightContractError("REQUEST_MODEL_AST_HASH_MISMATCH")
    production_freeze = _source_freeze_audit(registry)
    _build_brief(raw, request_sha)
    return ValidationContext(
        request_path=request_path,
        request_sha256=request_sha,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha,
        request_model_ast_sha256=model_hash,
        raw_request=deepcopy(raw),
        registry=deepcopy(registry),
        production_freeze=production_freeze,
    )


class _LiveRuntime:
    """Own all actual public adapters; constructed only after local gates."""

    def __init__(self, context: ValidationContext) -> None:
        from drugpilot.tools.admet_ai_adapter import ADMETAIAdapter
        from drugpilot.tools.live_ma2_gateway import LiveScientificGateway

        runtime = _mapping(context.registry.get("runtime"))
        timeout = float(runtime["ma2_chembl_timeout_seconds"])
        attempts = int(runtime["ma2_chembl_max_attempts"])
        self.context = context
        self.ma1_client = AuditedScientificHttpClient(
            timeout_seconds=float(
                runtime["public_evidence_max_elapsed_seconds_per_target"]
            ),
            max_attempts=int(runtime["public_evidence_max_attempts"]),
            retry_backoff_seconds=0.25,
            max_retry_delay_seconds=2.0,
        )
        self.chembl_client = AuditedScientificHttpClient(
            timeout_seconds=timeout,
            max_attempts=attempts,
            retry_backoff_seconds=0.25,
            max_retry_delay_seconds=2.0,
        )
        self.pubmed_client = AuditedScientificHttpClient(
            timeout_seconds=timeout,
            max_attempts=1,
            retry_backoff_seconds=0.0,
            max_retry_delay_seconds=0.0,
        )
        self.admet = ADMETAIAdapter()
        self.gateway = LiveScientificGateway(
            docking=_NoDockingPreflightBackend(),
            admet=self.admet,
            client=self.chembl_client,
            primary_search_client=self.pubmed_client,
            timeout_seconds=timeout,
            max_http_attempts=attempts,
            retry_backoff_seconds=0.25,
            max_retry_backoff_seconds=2.0,
            operation_timeout_seconds=float(
                runtime["ma2_chembl_operation_timeout_seconds"]
            ),
            repurposing_evidence_policy=PAPER_NATIVE_V5_POLICY_TOKEN,
        )

    def probe_candidates(self, context: ValidationContext) -> dict[str, Any]:
        from drugpilot.domain.models import TargetCandidate
        from drugpilot.tools.candidate_policy import screen_reserved_control
        from drugpilot.tools.chemistry import ChemistryValidator

        target = TargetCandidate(
            symbol=TARGET_SYMBOL,
            uniprot_id=TARGET_UNIPROT,
            target_name="Janus kinase 2",
            identity_status="resolved",
            identity_sources=["preregistered_request"],
        )
        normalized: Any = None
        seeds: tuple[Any, ...] = ()
        error_type: str | None = None
        error_reason_code: str | None = None
        exception_provenance: Mapping[str, Any] = {}
        readiness: Mapping[str, Any] = {}
        local_chemistry: dict[str, Any] = {
            "tool": "local chemistry",
            "status": "unavailable",
            "required": True,
            "fixture": False,
        }
        chembl_start = len(self.chembl_client.calls)
        pubmed_start = len(self.pubmed_client.calls)
        try:
            normalized = self.gateway.normalize_target(target)
            readiness = _mapping(self.gateway.readiness(normalized))
            chemistry = ChemistryValidator(prefer_rdkit=True)
            chemistry_probe = chemistry.inspect("C")
            local_chemistry = {
                "tool": "local chemistry",
                "status": (
                    "completed"
                    if chemistry.backend_name == "rdkit"
                    and chemistry_probe.valid
                    and chemistry_probe.backend == "rdkit"
                    else "unavailable"
                ),
                "backend": chemistry.backend_name,
                "version": chemistry.backend_version,
                "required": True,
                "fixture": False,
                "probe_input_stored": False,
            }
            seeds = tuple(
                self.gateway.retrieve_repurposing(
                    normalized,
                    direction="inhibit",
                    limit=FIXED_POOL_LIMIT,
                )
            )
        except Exception as exc:
            error_type = type(exc).__name__
            exception_provenance = _mapping(getattr(exc, "provenance", {}))
            error_reason_code = str(
                exception_provenance.get("reason_code")
                or "SCIENTIFIC_RETRIEVAL_UNAVAILABLE"
            )
        provenance = _mapping(self.gateway.last_operation_provenance)
        if not provenance and exception_provenance:
            provenance = exception_provenance
        brief = _build_brief(context.raw_request, context.request_sha256)
        noncontrols: list[Any] = []
        reserved_count = 0
        control_identity_conflict_count = 0
        for seed in seeds:
            screen = screen_reserved_control(
                candidate_name=str(getattr(seed, "name", "") or ""),
                metadata=_mapping(getattr(seed, "metadata", {})),
                reserved_controls=brief.constraints.reserved_control_compounds,
            )
            if screen.identity_conflict:
                control_identity_conflict_count += 1
            elif screen.matched:
                reserved_count += 1
            else:
                noncontrols.append(seed)
        identifiers = [
            str(_mapping(getattr(seed, "metadata", {})).get("chembl_id") or "")
            for seed in noncontrols
        ]
        return {
            "target": (
                normalized.model_dump(mode="json")
                if hasattr(normalized, "model_dump")
                else None
            ),
            "returned_seeds": [
                seed.model_dump(mode="json")
                if hasattr(seed, "model_dump")
                else seed
                for seed in seeds
            ],
            "candidate_count": len(seeds),
            "noncontrol_candidate_count": len(noncontrols),
            "reserved_control_count": reserved_count,
            "control_identity_conflict_count": control_identity_conflict_count,
            "candidate_identity_set_sha256": _canonical_sha256(identifiers),
            "provenance": deepcopy(dict(provenance)),
            "readiness": deepcopy(dict(readiness)),
            "local_chemistry_readiness": local_chemistry,
            "chembl_calls": deepcopy(self.chembl_client.calls[chembl_start:]),
            "pubmed_calls": deepcopy(self.pubmed_client.calls[pubmed_start:]),
            "error_type": error_type,
            "error_reason_code": error_reason_code,
        }

    def probe_ma1(self, context: ValidationContext) -> dict[str, Any]:
        from drugpilot.agents.ma1 import MA1Agent
        from drugpilot.orchestration.goal_focus import derive_goal_focus
        from drugpilot.tools.disease_targets import DiseaseTargetDiscovery
        from drugpilot.tools.evidence import ResilientEvidenceRetriever
        from drugpilot.tools.ncbi_rate_limit import get_process_ncbi_rate_limiter
        from drugpilot.tools.public_biology import PublicBiologyRetriever
        from drugpilot.tools.target_identity import TargetIdentityResolver

        runtime = _mapping(context.registry.get("runtime"))
        settings = Settings(_env_file=ROOT / ".env")
        brief = _build_brief(context.raw_request, context.request_sha256)
        focus = derive_goal_focus(brief)
        if (
            focus.target_scope_mode != "explicit_only"
            or tuple(focus.executable_target_mentions) != (TARGET_SYMBOL,)
        ):
            raise PreflightContractError("MA1_EXPLICIT_SCOPE_MISMATCH")
        budget = settings.evidence_budget_for(brief.research_depth)
        rate_limiter = get_process_ncbi_rate_limiter()
        retriever = PublicBiologyRetriever(
            client=self.ma1_client,
            timeout_seconds=budget.core_timeout_seconds,
            max_records_per_source=budget.core_records_per_source,
            max_source_calls=budget.core_source_calls_per_target,
            max_elapsed_seconds=budget.core_elapsed_seconds_per_target,
            max_attempts=budget.core_max_attempts,
            retry_backoff_seconds=budget.retry_backoff_seconds,
            fallback=None,
            enabled_sources=[
                "PubMed",
                "Europe PMC",
                "Open Targets",
                "UniProt",
                "DGIdb",
            ],
            ncbi_api_key=settings.ncbi_api_key,
            ncbi_tool=settings.ncbi_tool,
            ncbi_email=settings.ncbi_email,
            ncbi_rate_limiter=rate_limiter,
            counter_evidence_enabled=True,
            epmc_oa_fulltext_one_hop_v1=bool(
                runtime["epmc_oa_fulltext_one_hop_v1"]
            ),
        )
        evidence_retriever = ResilientEvidenceRetriever(
            public=retriever,
            fixture_fallback_enabled=False,
        )
        discovery = DiseaseTargetDiscovery(
            client=self.ma1_client,
            timeout_seconds=budget.core_timeout_seconds,
            max_targets=budget.target_max_candidates,
            pubmed_max_records=budget.target_pubmed_records,
            max_pubmed_supplements=budget.target_pubmed_supplements,
            target_lookup_limit=budget.target_lookup_limit,
            max_attempts=budget.core_max_attempts,
            retry_backoff_seconds=budget.retry_backoff_seconds,
            ncbi_api_key=settings.ncbi_api_key,
            ncbi_tool=settings.ncbi_tool,
            ncbi_email=settings.ncbi_email,
            ncbi_rate_limiter=rate_limiter,
        )
        identity = TargetIdentityResolver(
            client=self.ma1_client,
            timeout_seconds=budget.core_timeout_seconds,
        )
        run_settings = settings.model_copy(
            update={"dynamic_target_max_candidates": budget.target_max_candidates},
            deep=True,
        )
        http_start = len(self.ma1_client.calls)
        packet = MA1Agent(
            run_settings,
            evidence_retriever=evidence_retriever,
            target_discovery=discovery,
            identity_resolver=identity,
        ).run(brief)
        return {
            "packet": packet.model_dump(mode="json"),
            "scope": {
                "target_scope_mode": focus.target_scope_mode,
                "executable_target_mentions": list(
                    focus.executable_target_mentions
                ),
            },
            "epmc_oa_trace": deepcopy(dict(retriever.last_epmc_oa_fulltext_trace)),
            "http_calls": deepcopy(self.ma1_client.calls[http_start:]),
        }

    def close(self) -> None:
        self.ma1_client.close()
        self.chembl_client.close()
        self.pubmed_client.close()
        method = getattr(self.admet, "close", None)
        if callable(method):
            method()


def _fixture_findings(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            current = f"{path}.{key}"
            normalized = str(key).casefold()
            fixture_key = bool(
                normalized in {"fixture", "mock", "fallback"}
                or normalized.endswith("_fixture")
                or normalized.endswith("_fixture_count")
                or normalized.endswith("_fixture_calls")
                or normalized.endswith("_fallback_used")
                or normalized.endswith("_fallback_authorized")
                or normalized in {"fallback_reason", "fallback_role"}
            )
            if fixture_key and item not in (
                False,
                None,
                "",
                0,
                [],
                {},
            ):
                findings.append(current)
            if normalized in {"provenance_class", "origin_provenance_class"} and str(
                item
            ).casefold() in {"fixture", "fixture_fallback", "mock"}:
                findings.append(current)
            findings.extend(_fixture_findings(item, current))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            findings.extend(_fixture_findings(item, f"{path}[{index}]"))
    return findings


def _audit_retry_ledger(value: Any, *, expected_budget: int) -> dict[str, Any]:
    ledger = _mapping(value)
    records = _sequence(ledger.get("entries"))
    schema_valid = bool(
        ledger.get("schema_version") == "ma1_evidence_retry_attempt_ledger/v1"
        and ledger.get("authoritative") is True
    )
    valid = bool(
        schema_valid
        and ledger.get("entry_limit") == RETRY_LEDGER_LIMIT
        and len(records) <= RETRY_LEDGER_LIMIT
        and ledger.get("entry_count_total") == len(records)
        and ledger.get("entry_count_recorded") == len(records)
        and ledger.get("entry_count_omitted") == 0
    )
    canonical: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in records:
        item = _mapping(row)
        symbol = str(item.get("target_symbol") or "")
        attempts = item.get("attempt_count")
        row_valid = bool(
            CANONICAL_SYMBOL_RE.fullmatch(symbol)
            and symbol not in canonical
            and not isinstance(attempts, bool)
            and isinstance(attempts, int)
            and attempts >= 0
        )
        valid = valid and row_valid
        if row_valid:
            canonical.add(symbol)
            normalized.append(
                {"canonical_symbol": symbol, "actual_attempt_count": attempts}
            )
    valid = valid and expected_budget == 1
    return {
        "verified": bool(valid),
        "authoritative": ledger.get("authoritative") is True,
        "record_count": len(records),
        "records_sha256": _canonical_sha256(normalized),
        "entry_limit": RETRY_LEDGER_LIMIT,
        "supplementary_retry_budget": expected_budget,
        "preflight_rework_executed": False,
    }


def _audit_epmc_trace(trace_value: Any, evidence: Sequence[Any]) -> dict[str, Any]:
    trace = _mapping(trace_value)
    evidence_traces = [
        _mapping(_mapping(item).get("metadata")).get(
            "epmc_oa_fulltext_one_hop_trace"
        )
        for item in evidence
        if isinstance(
            _mapping(_mapping(item).get("metadata")).get(
                "epmc_oa_fulltext_one_hop_trace"
            ),
            Mapping,
        )
    ]
    all_traces = [trace, *[_mapping(item) for item in evidence_traces]]
    recognized_statuses = {
        "not_needed",
        "completed_with_evidence",
        "completed_partial_evidence",
        "completed_bounded_without_evidence",
        "completed_with_errors_without_evidence",
        "skipped",
    }
    base_valid = bool(
        trace.get("schema_version") == "epmc_oa_fulltext_one_hop_trace/v1"
        and trace.get("policy") == "epmc_oa_fulltext_one_hop_v1"
        and trace.get("enabled") is True
        and trace.get("status") in recognized_statuses
        and trace.get("publisher_or_pdf_calls", 0) in (0, None)
        and trace.get("fixture_calls", 0) in (0, None)
        and trace.get("raw_xml_stored", False) is False
        and trace.get("full_text_stored", False) is False
    )
    trace_valid = base_valid
    for item in all_traces:
        if not item:
            continue
        if (
            item.get("policy") != "epmc_oa_fulltext_one_hop_v1"
            or item.get("enabled") is not True
            or item.get("publisher_or_pdf_calls", 0) not in (0, None)
        ):
            trace_valid = False
    storage_valid = all(
        _mapping(item).get("metadata", {}).get("full_text_stored") is not True
        and _mapping(item).get("metadata", {}).get("raw_xml_stored") is not True
        for item in evidence
    )
    raw_calls = list(_sequence(trace.get("http_calls")))
    call_projection: list[dict[str, Any]] = []
    calls_valid = True
    for raw in raw_calls:
        call = _mapping(raw)
        endpoint = _safe_endpoint(call.get("endpoint"))
        host = (urlsplit(endpoint).hostname or "").casefold()
        response_hash = call.get("response_sha256") or call.get(
            "response_prefix_sha256"
        )
        valid = bool(
            call.get("method") == "GET"
            and host == "www.ebi.ac.uk"
            and call.get("fixture") is False
            and call.get("raw_response_stored") is False
            and call.get("full_text_stored") is False
            and HASH_RE.fullmatch(str(call.get("parameter_sha256") or ""))
            and HASH_RE.fullmatch(str(response_hash or ""))
            and call.get("wire_attempt_count") == 1
            and call.get("retries_allowed") is False
        )
        calls_valid = calls_valid and valid
        call_projection.append(
            {
                "method": call.get("method"),
                "endpoint": endpoint,
                "status": call.get("status"),
                "status_code": call.get("status_code"),
                "parameter_sha256": call.get("parameter_sha256"),
                "response_sha256": response_hash,
                "wire_attempt_count": call.get("wire_attempt_count"),
                "transport_body_cap_enforced": call.get(
                    "transport_body_cap_enforced"
                ),
            }
        )
    if raw_calls and not all(
        _mapping(item).get("transport_body_cap_enforced") is True
        for item in raw_calls
    ):
        calls_valid = False
    return {
        "verified": bool(trace_valid and storage_valid and calls_valid),
        "enabled": trace.get("enabled") is True,
        "status": trace.get("status"),
        "evidence_trace_count": len(evidence_traces),
        "trace_projection_sha256": _canonical_sha256(all_traces),
        "http_call_count": len(call_projection),
        "http_calls": call_projection,
        "http_calls_sha256": _canonical_sha256(call_projection),
        "production_stream_body_cap_verified": bool(
            not raw_calls
            or all(
                _mapping(item).get("transport_body_cap_enforced") is True
                for item in raw_calls
            )
        ),
        "raw_fulltext_or_xml_stored": not storage_valid,
    }


def _audit_ma1_probe(
    probe_value: Any,
    *,
    expected_retry_budget: int,
) -> dict[str, Any]:
    probe = _mapping(probe_value)
    packet = _mapping(probe.get("packet"))
    metadata = _mapping(packet.get("metadata"))
    hypotheses = list(_sequence(packet.get("hypotheses")))
    exclusions = list(_sequence(packet.get("excluded")))
    scope = _mapping(probe.get("scope"))
    pool = _mapping(metadata.get("ma1_pool_contract"))
    fixture_findings = _fixture_findings(probe)
    direct_count = 0
    association_count = 0
    sensitization_promotions = 0
    source_counts: Counter[str] = Counter()
    provenance_valid = True
    evidence_hash_rows: list[dict[str, Any]] = []
    identity_valid = bool(hypotheses)
    all_evidence: list[Any] = []
    for raw_hypothesis in hypotheses:
        hypothesis = _mapping(raw_hypothesis)
        target = _mapping(hypothesis.get("target"))
        identity_valid = bool(
            identity_valid
            and target.get("symbol") == TARGET_SYMBOL
            and target.get("uniprot_id") == TARGET_UNIPROT
            and target.get("identity_status") == "resolved"
        )
        evidence = list(_sequence(hypothesis.get("evidence")))
        all_evidence.extend(evidence)
        for raw in evidence:
            item = _mapping(raw)
            item_metadata = _mapping(item.get("metadata"))
            source = str(item.get("source") or "")
            source_counts[source] += 1
            provenance = str(item_metadata.get("provenance_class") or "")
            origin = str(
                item_metadata.get("origin_provenance_class") or provenance
            )
            if (
                provenance not in {"external_live", "external_cache"}
                or origin != "external_live"
            ):
                provenance_valid = False
            direct = bool(
                item_metadata.get("verified_direct_primary") is True
                and item_metadata.get("direct_support_eligible") is not False
            )
            if direct:
                direct_count += 1
            if (
                item_metadata.get("role_specific_record")
                == "primary_target_disease_association_excerpt"
            ):
                association_count += 1
            text = " ".join(
                str(value or "")
                for value in (
                    item.get("title"),
                    item_metadata.get("excerpt"),
                    item_metadata.get("primary_direction_excerpt"),
                )
            )
            if direct and SENSITIZATION_RE.search(text):
                sensitization_promotions += 1
            evidence_hash_rows.append(
                {
                    "source": source,
                    "supports": item.get("supports"),
                    "verified_direct_primary": item_metadata.get(
                        "verified_direct_primary"
                    ),
                    "direct_support_eligible": item_metadata.get(
                        "direct_support_eligible"
                    ),
                    "role": item_metadata.get("role_specific_record"),
                    "content_hash": item_metadata.get("content_hash")
                    or item.get("excerpt_hash"),
                }
            )
    epmc = _audit_epmc_trace(probe.get("epmc_oa_trace"), all_evidence)
    retry = _audit_retry_ledger(
        metadata.get("ma1_evidence_retry_attempt_ledger"),
        expected_budget=expected_retry_budget,
    )
    search_count = metadata.get("ma1_search_candidate_count")
    screened_count = metadata.get("ma1_screened_candidate_count")
    executable_count = metadata.get("ma1_executable_hypothesis_count")
    pool_valid = bool(
        pool.get("schema_version") == "search-screen-executable-pools-v1"
        and pool.get("search_pool_role") == "search_only_not_executable"
        and pool.get("search_stage_direction_semantics")
        == "not_assessed_search_stage"
        and pool.get("executable_pool_role") == "retained_hypotheses_only"
        and type(search_count) is int
        and type(screened_count) is int
        and type(executable_count) is int
        and search_count == pool.get("search_candidate_count")
        and screened_count == pool.get("screened_candidate_count")
        and executable_count == pool.get("executable_hypothesis_count")
        and executable_count == len(hypotheses)
        and search_count >= screened_count >= executable_count >= 1
    )
    scope_valid = bool(
        scope.get("target_scope_mode") == "explicit_only"
        and list(_sequence(scope.get("executable_target_mentions")))
        == [TARGET_SYMBOL]
        and metadata.get("ma1_executable_target_scope_applied") is True
        and list(
            _sequence(metadata.get("ma1_executable_target_scope_symbols"))
        )
        == [TARGET_SYMBOL]
        and metadata.get("ma1_scope_evidence_missing_fail_closed") is False
    )
    screening_trace = _mapping(metadata.get("rejected_screening_trace"))
    screening_entries = list(_sequence(screening_trace.get("entries")))
    screening_trace_valid = bool(
        screening_trace.get("schema_version") == "1.0"
        and screening_trace.get("scope") == "router_rejected_only"
        and screening_trace.get("immutable_projection") is True
        and screening_trace.get("entry_limit") == 24
        and screening_trace.get("entry_count_recorded")
        == len(screening_entries)
        and type(screening_trace.get("entry_count_total")) is int
        and screening_trace["entry_count_total"] >= len(screening_entries)
        and screening_trace.get("entry_count_omitted")
        == screening_trace["entry_count_total"] - len(screening_entries)
        and metadata.get("screening_trace_sha256")
        == _canonical_sha256(screening_trace)
    )
    http = _http_safety(
        list(_sequence(probe.get("http_calls"))),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    verified = bool(
        identity_valid
        and all_evidence
        and provenance_valid
        and not fixture_findings
        and direct_count >= 1
        and association_count >= 1
        and sensitization_promotions == 0
        and scope_valid
        and pool_valid
        and screening_trace_valid
        and epmc["verified"]
        and retry["verified"]
        and http["verified"]
        and metadata.get("target_fallback_used") is False
        and metadata.get("demo_fixture_fallback_authorized") is False
    )
    return {
        "verified": verified,
        "production_ma1_run_used": True,
        "explicit_scope_verified": scope_valid,
        "identity_verified": identity_valid,
        "search_pool_count": search_count,
        "screened_pool_count": screened_count,
        "executable_pool_count": executable_count,
        "pool_contract_verified": pool_valid,
        "excluded_count": len(exclusions),
        "screening_trace_verified": screening_trace_valid,
        "screening_trace_sha256": _canonical_sha256(screening_trace),
        "evidence_count": len(all_evidence),
        "evidence_projection_sha256": _canonical_sha256(evidence_hash_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "external_origin_verified": provenance_valid,
        "fixture_finding_count": len(fixture_findings),
        "direct_direction_primary_count": direct_count,
        "primary_target_disease_association_count": association_count,
        "sensitization_direct_promotion_count": sensitization_promotions,
        "epmc_oa": epmc,
        "retry_ledger": retry,
        "public_http": http,
    }


def _activity_page_set_valid(
    count: Mapping[str, Any], *, expected_parent: str, target_id: str
) -> bool:
    total = count.get("total_count")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        return False
    if (
        count.get("response_count") != total
        or count.get("inspected_count") != total
        or count.get("complete_result_set") is not True
        or count.get("truncated") is not False
        or count.get("inspected_cap") is not None
        or count.get("page_size") != ACTIVITY_PAGE_SIZE
    ):
        return False
    pages = list(_sequence(count.get("page_audits")))
    expected_page_count = max(1, math.ceil(total / ACTIVITY_PAGE_SIZE))
    if count.get("page_count") != expected_page_count or len(pages) != expected_page_count:
        return False
    prior_activity_id: int | None = None
    observed = 0
    for index, raw_page in enumerate(pages):
        page = _mapping(raw_page)
        offset = index * ACTIVITY_PAGE_SIZE
        expected_count = min(ACTIVITY_PAGE_SIZE, max(0, total - offset))
        identifiers = list(_sequence(page.get("activity_ids")))
        parameters = _mapping(page.get("query_parameters"))
        expected_parameters = {
            "target_chembl_id": target_id,
            "parent_molecule_chembl_id": expected_parent,
            "target_organism": "Homo sapiens",
            "assay_type": "B",
            "standard_type": "IC50",
            "pchembl_value__gte": 5.0,
            "order_by": "activity_id",
            "limit": ACTIVITY_PAGE_SIZE,
        }
        if index:
            expected_parameters["offset"] = offset
        if (
            page.get("page_index") != index
            or page.get("offset") != offset
            or page.get("limit") != ACTIVITY_PAGE_SIZE
            or page.get("response_count") != expected_count
            or page.get("reported_total_count") != total
            or len(identifiers) != expected_count
            or dict(parameters) != expected_parameters
            or page.get("activity_ids_sha256")
            != _canonical_sha256(identifiers)
            or page.get("page_audit_sha256")
            != _canonical_sha256(
                {
                    key: value
                    for key, value in page.items()
                    if key != "page_audit_sha256"
                }
            )
        ):
            return False
        for identifier in identifiers:
            try:
                numeric = int(str(identifier))
            except (TypeError, ValueError):
                return False
            if numeric < 1 or (
                prior_activity_id is not None and numeric <= prior_activity_id
            ):
                return False
            prior_activity_id = numeric
        observed += len(identifiers)
    return bool(
        observed == total
        and count.get("page_audits_sha256") == _canonical_sha256(pages)
    )


def _http_safety(
    calls: Sequence[Any], *, allowed_methods: frozenset[str] = frozenset({"GET"})
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    verified = True
    for raw in calls:
        call = _mapping(raw)
        endpoint = _safe_endpoint(call.get("endpoint"))
        host = (urlsplit(endpoint).hostname or "").casefold()
        method = str(call.get("method") or "").upper()
        safe = bool(
            method in allowed_methods
            and host in ALLOWED_HTTP_HOSTS
            and urlsplit(endpoint).scheme == "https"
            and call.get("fixture") is False
            and call.get("status") in {"completed", "failed"}
            and HASH_RE.fullmatch(str(call.get("parameter_sha256") or ""))
            and (
                (
                    call.get("status") == "completed"
                    and type(call.get("status_code")) is int
                    and HASH_RE.fullmatch(
                        str(call.get("response_sha256") or "")
                    )
                )
                or (
                    call.get("status") == "failed"
                    and call.get("status_code") is None
                    and call.get("response_sha256") is None
                )
            )
        )
        verified = verified and safe
        normalized.append(
            {
                "method": method,
                "endpoint": endpoint,
                "status": call.get("status"),
                "status_code": call.get("status_code"),
                "response_sha256": call.get("response_sha256"),
                "parameter_keys": list(_sequence(call.get("parameter_keys"))),
                "parameter_sha256": call.get("parameter_sha256"),
                "attempt_count_observed": call.get("attempt_count_observed"),
                "fixture": call.get("fixture"),
            }
        )
    return {
        "verified": bool(verified and normalized),
        "call_count": len(normalized),
        "calls": normalized,
        "calls_sha256": _canonical_sha256(normalized),
    }


def _hash_field_valid(value: Mapping[str, Any], field: str) -> bool:
    digest = str(value.get(field) or "")
    if not HASH_RE.fullmatch(digest):
        return False
    body = {key: item for key, item in value.items() if key != field}
    return digest == _canonical_sha256(body)


def _candidate_readiness_audit(
    value: Any, local_chemistry_value: Any
) -> dict[str, Any]:
    readiness = _mapping(value)
    tools = [_mapping(item) for item in _sequence(readiness.get("tools"))]
    by_name = {str(item.get("tool") or ""): item for item in tools}
    required_names = ("ChEMBL", "RDKit BRICS", "ADMET-AI")
    required_valid = all(
        name in by_name
        and by_name[name].get("status") == "completed"
        and by_name[name].get("required") is True
        and by_name[name].get("fixture") is False
        for name in required_names
    )
    docking = by_name.get("docking", {})
    docking_valid = bool(
        docking
        and docking.get("required") is False
        and docking.get("fixture") is False
        and docking.get("status") in {"not_configured", "completed"}
    )
    local = _mapping(local_chemistry_value)
    local_valid = bool(
        local.get("tool") == "local chemistry"
        and local.get("status") == "completed"
        and local.get("backend") == "rdkit"
        and local.get("required") is True
        and local.get("fixture") is False
        and local.get("probe_input_stored") is False
    )
    projection = [
        {
            "tool": str(item.get("tool") or ""),
            "status": item.get("status"),
            "required": item.get("required"),
            "fixture": item.get("fixture"),
        }
        for item in tools
    ]
    projection.append(
        {
            "tool": "local chemistry",
            "status": local.get("status"),
            "required": local.get("required"),
            "fixture": local.get("fixture"),
        }
    )
    return {
        "verified": bool(required_valid and docking_valid and local_valid),
        "required_tools_verified": required_valid,
        "local_chemistry_verified": local_valid,
        "optional_docking_nonblocking_verified": docking_valid,
        "tool_count": len(projection),
        "tools": projection,
        "projection_sha256": _canonical_sha256(projection),
    }


def _paper_native_v5_audit(
    search_value: Any,
    *,
    fixed_ids: Sequence[str],
) -> dict[str, Any]:
    search = _mapping(search_value)
    support = _mapping(search.get("direct_support_inventory"))
    support_records = [_mapping(item) for item in _sequence(support.get("records"))]
    selected_ids = list(_sequence(support.get("selected_candidate_ids")))
    support_valid = bool(
        search.get("schema_version") == "primary_abstract_paper_native_gateway_v5"
        and search.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN
        and search.get("fixture") is False
        and _hash_field_valid(search, "provenance_sha256")
        and support.get("schema_version")
        == "paper_native_v5_direct_support_inventory_v1"
        and support.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN
        and support.get("fixture") is False
        and _hash_field_valid(support, "inventory_sha256")
        and list(_sequence(support.get("fixed_pool_candidate_ids")))
        == list(fixed_ids)
        and support.get("fixed_pool_candidate_ids_sha256")
        == _canonical_sha256(list(fixed_ids))
        and len(support_records) == len(fixed_ids)
        and len(selected_ids) >= MINIMUM_NONCONTROL_CANDIDATES
        and selected_ids == [item for item in fixed_ids if item in set(selected_ids)]
        and support.get("selected_candidate_count") == len(selected_ids)
        and support.get("selection_completed_before_pubmed_search") is True
        and support.get("pubmed_or_result_used_for_selection") is False
        and support.get("no_reordering") is True
        and support.get("no_backfill") is True
    )
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for record in support_records:
        candidate_id = str(record.get("molecule_chembl_id") or "")
        selected = _mapping(record.get("selected_support"))
        params = _mapping(selected.get("chembl_activity_query_parameters"))
        record_base_valid = bool(
            candidate_id in fixed_ids
            and _hash_field_valid(record, "record_sha256")
        )
        if candidate_id not in selected_ids:
            record_valid = bool(record_base_valid and not selected)
        else:
            record_valid = bool(
            record_base_valid
            and selected.get("verification_status") == "verified"
            and selected.get("molecule_chembl_id") == candidate_id
            and selected.get("target_uniprot_id") == TARGET_UNIPROT
            and selected.get("target_organism") == "Homo sapiens"
            and selected.get("target_type") == "SINGLE PROTEIN"
            and selected.get("selected_before_pubmed_search") is True
            and selected.get("selection_policy")
            == "first_strict_direct_support_in_retained_order_v1"
            and selected.get("pubmed_value_used_for_selection") is False
            and params.get("parent_molecule_chembl_id") == candidate_id
            and params.get("target_organism") == "Homo sapiens"
            and params.get("assay_type") == "B"
            and params.get("standard_type") == "IC50"
            and params.get("pchembl_value__gte") == 5.0
            and params.get("order_by") == "activity_id"
            and params.get("limit") == ACTIVITY_PAGE_SIZE
            and _hash_field_valid(selected, "support_attestation_sha256")
            )
        support_valid = support_valid and record_valid
        if record_valid and candidate_id in selected_ids:
            selected_by_id[candidate_id] = selected

    aliases = _mapping(search.get("alias_inventory"))
    alias_records = [_mapping(item) for item in _sequence(aliases.get("records"))]
    alias_valid = bool(
        aliases.get("schema_version") == "paper_native_v5_alias_inventory_v1"
        and aliases.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN
        and aliases.get("fixture") is False
        and aliases.get("complete") is True
        and _hash_field_valid(aliases, "inventory_sha256")
        and list(_sequence(aliases.get("fixed_pool_candidate_ids")))
        == list(fixed_ids)
        and list(_sequence(aliases.get("direct_support_candidate_ids")))
        == selected_ids
        and list(_sequence(aliases.get("queried_candidate_ids"))) == selected_ids
        and aliases.get("per_candidate_alias_limit") == 16
        and len(alias_records) == len(selected_ids)
        and aliases.get("owner_conflicts") == []
        and aliases.get("pubmed_or_result_used_for_alias_selection") is False
    )
    for record in alias_records:
        accepted = list(_sequence(record.get("accepted_aliases")))
        source = _mapping(record.get("molecule_synonyms_source_inventory"))
        alias_valid = bool(
            alias_valid
            and record.get("molecule_chembl_id") in selected_ids
            and _hash_field_valid(record, "record_sha256")
            and _hash_field_valid(source, "inventory_sha256")
            and source.get("complete") is True
            and record.get("source_attestation_complete") is True
            and record.get("query_eligible") is True
            and 1 <= len(accepted) <= 16
            and record.get("accepted_alias_count") == len(accepted)
            and record.get("preferred_search_alias") == accepted[0]
        )

    outcome = _mapping(search.get("search_outcome"))
    promotion = _mapping(search.get("promotion"))
    groups = [_mapping(item) for item in _sequence(outcome.get("groups"))]
    selected_group_hash = str(promotion.get("selected_group_sha256") or "")
    matching_groups = [
        item for item in groups if item.get("group_sha256") == selected_group_hash
    ]
    joint_valid = bool(
        outcome.get("policy_token") == PAPER_NATIVE_V5_POLICY_TOKEN
        and outcome.get("status") == "verified"
        and _hash_field_valid(outcome, "outcome_sha256")
        and len(matching_groups) == 1
        and promotion.get("schema_version") == "paper_native_v5_promotion_v1"
        and promotion.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN
        and promotion.get("status") == "verified"
        and promotion.get("fixture") is False
        and _hash_field_valid(promotion, "promotion_sha256")
        and promotion.get("verified_member_count")
        == len(_sequence(promotion.get("verified_member_candidate_ids")))
        and promotion.get("verified_member_count", 0)
        >= MINIMUM_NONCONTROL_CANDIDATES
        and list(_sequence(promotion.get("verified_member_candidate_ids")))
        == list(_sequence(promotion.get("selected_group_candidate_ids")))
        and set(_sequence(promotion.get("verified_member_candidate_ids"))).issubset(
            set(selected_ids)
        )
        and promotion.get("group_member_failure_rejects_entire_group") is True
        and promotion.get("no_partial_group_promotion") is True
        and promotion.get("no_reordering") is True
        and promotion.get("no_backfill") is True
    )
    return {
        "verified": bool(support_valid and alias_valid and joint_valid),
        "strict_support_verified": support_valid,
        "complete_alias_inventory_verified": alias_valid,
        "joint_pubmed_group_verified": joint_valid,
        "strict_support_count": len(selected_by_id),
        "alias_record_count": len(alias_records),
        "joint_group_count": len(matching_groups),
        "projection_sha256": _canonical_sha256(
            {
                "search": search.get("provenance_sha256"),
                "support": support.get("inventory_sha256"),
                "aliases": aliases.get("inventory_sha256"),
                "outcome": outcome.get("outcome_sha256"),
                "promotion": promotion.get("promotion_sha256"),
            }
        ),
    }


def _audit_candidate_probe(
    probe_value: Any, *, reserved_controls: Sequence[Any]
) -> dict[str, Any]:
    from drugpilot.tools.candidate_policy import screen_reserved_control

    probe = _mapping(probe_value)
    target = _mapping(probe.get("target"))
    target_metadata = _mapping(target.get("metadata"))
    provenance = _mapping(probe.get("provenance"))
    bounds = _mapping(provenance.get("candidate_discovery_bounds"))
    if not bounds and provenance.get("candidate_discovery_bounds") is None:
        bounds = _mapping(_mapping(provenance.get("provenance")).get("candidate_discovery_bounds"))
    fixed_ids = [str(item) for item in _sequence(bounds.get("prewarning_pool_ids"))]
    preactivity = [_mapping(item) for item in _sequence(bounds.get("preactivity_pool"))]
    expected_fixed_order = [
        str(item.get("parent_molecule_chembl_id") or "")
        for item in sorted(
            preactivity,
            key=lambda item: (
                -int(item.get("max_phase"))
                if type(item.get("max_phase")) is int
                else 1,
                str(item.get("parent_molecule_chembl_id") or ""),
            ),
        )
    ]
    counts_by_identity = _mapping(bounds.get("activity_counts_by_id"))
    pagination_rows = list(counts_by_identity.values())
    target_id = str(bounds.get("target_chembl_id") or "")
    pagination_verified = bool(
        fixed_ids
        and list(counts_by_identity) == fixed_ids
        and all(
            _activity_page_set_valid(
                _mapping(counts_by_identity[parent_id]),
                expected_parent=parent_id,
                target_id=target_id,
            )
            for parent_id in fixed_ids
        )
    )
    fixture_findings = _fixture_findings(probe)
    seeds = [_mapping(item) for item in _sequence(probe.get("returned_seeds"))]
    returned_ids: list[str] = []
    noncontrol_ids: list[str] = []
    computed_reserved = 0
    computed_conflicts = 0
    envelopes_valid = bool(seeds)
    search = _mapping(bounds.get("primary_abstract_search"))
    promotion = _mapping(search.get("promotion"))
    verified_members = set(
        str(item)
        for item in _sequence(promotion.get("verified_member_candidate_ids"))
    )
    for seed in seeds:
        metadata = _mapping(seed.get("metadata"))
        candidate_id = str(metadata.get("chembl_id") or "")
        returned_ids.append(candidate_id)
        control = screen_reserved_control(
            candidate_name=str(seed.get("name") or ""),
            metadata=metadata,
            reserved_controls=reserved_controls,
        )
        if control.identity_conflict:
            computed_conflicts += 1
        elif control.matched:
            computed_reserved += 1
        else:
            noncontrol_ids.append(candidate_id)
        envelope = _mapping(metadata.get("primary_abstract_paper_native_v5"))
        envelopes_valid = bool(
            envelopes_valid
            and CHEMBL_ID_RE.fullmatch(candidate_id)
            and candidate_id in fixed_ids
            and envelope.get("schema_version")
            == "candidate_primary_evidence_paper_native_v5"
            and envelope.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN
            and envelope.get("verification_status") == "verified"
            and envelope.get("is_primary_support") is True
            and envelope.get("molecule_chembl_id") == candidate_id
            and list(_sequence(envelope.get("fixed_pool_candidate_ids")))
            == fixed_ids
            and candidate_id in verified_members
            and envelope.get("fixture") is False
            and envelope.get("no_reordering") is True
            and envelope.get("no_backfill") is True
            and _hash_field_valid(envelope, "attestation_sha256")
        )
    candidate_count = probe.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        candidate_count = 0
    fixed_with_activity = bounds.get("fixed_pool_with_activity_count")
    if isinstance(fixed_with_activity, bool) or not isinstance(
        fixed_with_activity, int
    ):
        fixed_with_activity = 0
    fixed_with_activity_ids = [
        str(item) for item in _sequence(bounds.get("fixed_pool_with_activity_ids"))
    ]
    chembl_http = _http_safety(list(_sequence(probe.get("chembl_calls"))))
    pubmed_calls = list(_sequence(probe.get("pubmed_calls")))
    pubmed_http = _http_safety(pubmed_calls)
    audited_pubmed_signatures = [
        (
            _mapping(item).get("parameter_sha256"),
            _mapping(item).get("response_sha256"),
        )
        for item in pubmed_calls
    ]
    provenance_pubmed_signatures = [
        (
            _mapping(item).get("parameter_sha256"),
            _mapping(item).get("response_sha256"),
        )
        for item in _sequence(
            _mapping(search.get("search_outcome")).get("calls")
        )
    ]
    pubmed_ledger_aligned = bool(
        audited_pubmed_signatures
        and audited_pubmed_signatures == provenance_pubmed_signatures
    )
    raw_mechanism_count = bounds.get("raw_mechanism_response_count")
    mechanism_total = bounds.get("mechanism_total_count")
    mechanism_verified = bool(
        bounds.get("schema_version")
        == "bounded_mechanism_first_candidate_identity_pool_v1"
        and bounds.get("target_organism") == "Homo sapiens"
        and bounds.get("target_type") == "SINGLE PROTEIN"
        and bounds.get("mechanism_action_type") == "INHIBITOR"
        and bounds.get("raw_mechanism_row_limit") == MECHANISM_ROW_LIMIT
        and type(raw_mechanism_count) is int
        and type(mechanism_total) is int
        and 0 <= raw_mechanism_count == mechanism_total <= MECHANISM_ROW_LIMIT
        and bounds.get("mechanism_complete_within_bound") is True
        and bounds.get("prewarning_pool_limit") == FIXED_POOL_LIMIT
        and bounds.get("prewarning_pool_count") == len(fixed_ids)
        and 0 < len(fixed_ids) <= FIXED_POOL_LIMIT
        and len(set(fixed_ids)) == len(fixed_ids)
        and all(CHEMBL_ID_RE.fullmatch(item) for item in fixed_ids)
        and [str(item.get("parent_molecule_chembl_id") or "") for item in preactivity]
        == fixed_ids
        and expected_fixed_order == fixed_ids
        and [item.get("rank") for item in preactivity]
        == list(range(1, len(preactivity) + 1))
        and bounds.get("selection_inputs")
        == ["max_phase", "parent_molecule_chembl_id"]
        and bounds.get("pchembl_used_for_prewarning_selection") is False
        and bounds.get("primary_document_used_for_prewarning_selection") is False
        and bounds.get("warning_used_for_prewarning_selection") is False
        and bounds.get("activity_used_for_prewarning_selection") is False
        and bounds.get("no_backfill_beyond_prewarning_pool") is True
        and bounds.get("fixture") is False
    )
    target_verified = bool(
        target.get("symbol") == TARGET_SYMBOL
        and target.get("uniprot_id") == TARGET_UNIPROT
        and target.get("identity_status") == "resolved"
        and target_metadata.get("chembl_target_id") == target_id
        and target_metadata.get("chembl_target_type") == "SINGLE PROTEIN"
        and target_metadata.get("chembl_organism") == "Homo sapiens"
        and target_metadata.get("fixture") is False
    )
    activity_contract_verified = bool(
        bounds.get("activity_complete_pagination_required") is True
        and bounds.get("activity_rows_per_candidate_limit") == ACTIVITY_PAGE_SIZE
        and bounds.get("activity_rows_per_candidate_limit_semantics")
        == "wire_page_size_not_result_cap"
        and bounds.get("activity_result_row_limit") is None
        and bounds.get("no_activity_first_fallback") is True
        and bounds.get("no_backfill_beyond_prewarning_pool") is True
        and bounds.get("activity_failure_count") == 0
        and fixed_with_activity == len(fixed_with_activity_ids)
        and fixed_with_activity_ids
        == [item for item in fixed_ids if item in set(fixed_with_activity_ids)]
    )
    paper_native = _paper_native_v5_audit(search, fixed_ids=fixed_ids)
    readiness = _candidate_readiness_audit(
        probe.get("readiness"), probe.get("local_chemistry_readiness")
    )
    returned_order_verified = bool(
        returned_ids
        and len(returned_ids) == len(set(returned_ids))
        and returned_ids == [item for item in fixed_ids if item in set(returned_ids)]
        and bounds.get("gateway_returned_from_fixed_prewarning_pool") is True
        and bounds.get("gateway_return_order_matches_fixed_prewarning_subsequence")
        is True
    )
    controls_verified = bool(
        probe.get("reserved_control_count") == computed_reserved
        and probe.get("control_identity_conflict_count") == computed_conflicts == 0
        and probe.get("noncontrol_candidate_count") == len(noncontrol_ids)
        and len(noncontrol_ids) >= MINIMUM_NONCONTROL_CANDIDATES
        and probe.get("candidate_identity_set_sha256")
        == _canonical_sha256(noncontrol_ids)
    )
    complete = bool(
        target_verified
        and mechanism_verified
        and activity_contract_verified
        and pagination_verified
        and paper_native["verified"]
        and readiness["verified"]
        and returned_order_verified
        and controls_verified
        and envelopes_valid
        and candidate_count == len(seeds)
        and fixed_with_activity >= MINIMUM_NONCONTROL_CANDIDATES
        and not fixture_findings
        and chembl_http["verified"]
        and pubmed_http["verified"]
        and pubmed_ledger_aligned
        and probe.get("error_type") is None
    )
    totals = [
        int(_mapping(item).get("total_count") or 0)
        for item in pagination_rows
        if isinstance(_mapping(item).get("total_count"), int)
        and not isinstance(_mapping(item).get("total_count"), bool)
    ]
    return {
        "verified": complete,
        "target_contract_verified": target_verified,
        "mechanism_pool_verified": mechanism_verified,
        "activity_contract_verified": activity_contract_verified,
        "returned_order_verified": returned_order_verified,
        "reserved_control_exclusion_verified": controls_verified,
        "returned_envelopes_verified": envelopes_valid,
        "candidate_count": candidate_count,
        "noncontrol_candidate_count": len(noncontrol_ids),
        "reserved_control_count": computed_reserved,
        "candidate_identity_set_sha256": probe.get("candidate_identity_set_sha256")
        or _canonical_sha256([]),
        "fixed_pool_with_activity_count": fixed_with_activity,
        "activity_identity_count": len(pagination_rows),
        "activity_total_row_count": sum(totals),
        "activity_max_rows_for_one_identity": max(totals, default=0),
        "twenty_is_wire_page_size_not_result_cap": (
            bounds.get("activity_rows_per_candidate_limit_semantics")
            == "wire_page_size_not_result_cap"
        ),
        "complete_pagination_verified": pagination_verified,
        "paper_native_v5": paper_native,
        "readiness": readiness,
        "pubmed_ledger_aligned": pubmed_ledger_aligned,
        "activity_failure_count": bounds.get("activity_failure_count"),
        "fixture_finding_count": len(fixture_findings),
        "scientific_operation_error_type": probe.get("error_type"),
        "scientific_operation_reason_code": probe.get("error_reason_code"),
        "chembl_http": chembl_http,
        "pubmed_http": pubmed_http,
    }


_PROTOCOL_HASH_ASSIGNMENT_RE = re.compile(
    r'(?m)^EXPECTED_PROTOCOL_SHA256: str \| None = '
    r'(?:None|"[0-9a-f]{64}")$'
)


def _normalized_harness_script_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized, count = _PROTOCOL_HASH_ASSIGNMENT_RE.subn(
        'EXPECTED_PROTOCOL_SHA256: str | None = "<PROTOCOL_SHA256_PLACEHOLDER>"',
        text,
    )
    if count != 1:
        raise PreflightContractError("HARNESS_PROTOCOL_HASH_SLOT_MISMATCH")
    return sha256(normalized.encode("utf-8")).hexdigest()


def _harness_manifest(registry: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _mapping(registry.get("harness_freeze"))
    rows: list[dict[str, Any]] = []
    for relative in HARNESS_PATHS:
        path = ROOT / relative
        expected = str(
            frozen.get(
                "normalized_script_sha256"
                if relative.startswith("scripts/")
                else "test_sha256"
            )
            or ""
        ).casefold()
        observed = (
            _normalized_harness_script_sha256(path)
            if path.is_file() and relative.startswith("scripts/")
            else _sha256_file(path)
            if path.is_file()
            else None
        )
        rows.append(
            {
                "path": relative,
                "present": path.is_file(),
                "hash_mode": (
                    "expected_protocol_hash_slot_normalized_v1"
                    if relative.startswith("scripts/")
                    else "exact_bytes"
                ),
                "sha256": observed,
                "expected_sha256": expected or None,
                "verified": bool(
                    path.is_file()
                    and HASH_RE.fullmatch(expected)
                    and observed == expected
                ),
            }
        )
    return {
        "schema_version": "jak2_et_r18_validation_v6_harness_manifest/v1",
        "verified": bool(
            frozen.get("normalization")
            == "expected_protocol_sha256_assignment_placeholder_v1"
            and all(row["verified"] for row in rows)
        ),
        "excluded_from_production_source_freeze": True,
        "records": rows,
        "manifest_sha256": _canonical_sha256(rows),
    }


def _assert_safe_report(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_REPORT_KEYS:
                raise PreflightContractError("UNSAFE_REPORT_KEY")
            negative_secret_attestation = bool(
                item is False
                and normalized in {
                    "secret_values_included",
                    "secret_values_stored",
                }
            )
            authorization_policy_attestation = bool(
                normalized == "explicit_user_authorization_required"
                and item is True
            )
            if (
                any(marker in normalized for marker in SENSITIVE_KEY_MARKERS)
                and not negative_secret_attestation
                and not authorization_policy_attestation
            ):
                raise PreflightContractError("SECRET_BEARING_REPORT_KEY")
            _assert_safe_report(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_safe_report(item, f"{path}[{index}]")
    elif isinstance(value, str) and SENSITIVE_STRING_RE.search(value):
        raise PreflightContractError("SECRET_BEARING_REPORT_VALUE")


def _execute(
    context: ValidationContext,
    runtime: Any,
    *,
    execution_mode: Literal["live", "contract_test"],
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        ma1_probe = runtime.probe_ma1(context)
    except Exception as exc:
        ma1_probe = {
            "packet": {},
            "scope": {},
            "epmc_oa_trace": {},
            "http_calls": [],
            "operation_error_type": type(exc).__name__,
        }
    try:
        candidate_probe = runtime.probe_candidates(context)
    except Exception as exc:
        candidate_probe = {
            "target": None,
            "candidate_count": 0,
            "provenance": {},
            "chembl_calls": [],
            "pubmed_calls": [],
            "error_type": type(exc).__name__,
            "error_reason_code": "SCIENTIFIC_RETRIEVAL_UNAVAILABLE",
        }
    runtime_expected = _mapping(context.registry.get("runtime"))
    ma1 = _audit_ma1_probe(
        ma1_probe,
        expected_retry_budget=int(
            runtime_expected.get("ma1_supplementary_evidence_retry_budget") or 0
        ),
    )
    candidate = _audit_candidate_probe(
        candidate_probe,
        reserved_controls=list(
            _sequence(context.raw_request.get("reserved_control_compounds"))
        ),
    )
    fixture_finding_count = int(ma1.get("fixture_finding_count") or 0) + int(
        candidate.get("fixture_finding_count") or 0
    )
    harness = _harness_manifest(context.registry)
    optional_docking = {
        "status": "not_configured_or_intentionally_not_invoked",
        "required": False,
        "blocking": False,
        "experimental_profiles_allowed": runtime_expected.get(
            "ma2_live_allow_experimental_docking_profiles"
        ),
        "verified_nonblocking": runtime_expected.get(
            "ma2_live_allow_experimental_docking_profiles"
        )
        is False,
    }
    blockers: list[str] = []
    for passed, code in (
        (bool(runtime_environment.get("verified")), "RUNTIME_ENVIRONMENT_NOT_VERIFIED"),
        (context.production_freeze.get("verified") is True, "SOURCE_FREEZE_NOT_VERIFIED"),
        (harness["verified"], "HARNESS_MANIFEST_NOT_VERIFIED"),
        (ma1["verified"], "MA1_PUBLIC_EVIDENCE_NOT_READY"),
        (candidate["verified"], "CANDIDATE_SOURCE_NOT_READY"),
        (optional_docking["verified_nonblocking"], "DOCKING_POLICY_MISMATCH"),
    ):
        if not passed:
            blockers.append(code)
    technical_ready = not blockers
    if execution_mode == "contract_test":
        decision = "CONTRACT_TEST_ONLY"
    elif technical_ready:
        decision = "PREFLIGHT_READY_FOR_EXPLICIT_USER_AUTHORIZATION"
    else:
        decision = "PRECHECK_NOT_READY"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "execution_mode": execution_mode,
        "decision": decision,
        "contract_test_would_be_decision": (
            "PREFLIGHT_READY_FOR_EXPLICIT_USER_AUTHORIZATION"
            if execution_mode == "contract_test" and technical_ready
            else "PRECHECK_NOT_READY"
            if execution_mode == "contract_test"
            else None
        ),
        "technical_ready": technical_ready,
        "paid_run_authorized": False,
        "automatic_submission_allowed": False,
        "explicit_user_authorization_required": True,
        "immutable_input_gate": {
            "request_file_name": context.request_path.name,
            "request_sha256": context.request_sha256,
            "protocol_file_name": context.protocol_path.name,
            "protocol_sha256": context.protocol_sha256,
            "create_run_request_ast_sha256": context.request_model_ast_sha256,
            "unknown_top_level_key_count": 0,
            "policy_marker": PAPER_NATIVE_V5_POLICY_TOKEN,
            "validation_v6_is_packet_revision_not_source_contract": True,
        },
        "production_source_freeze": deepcopy(context.production_freeze),
        "runtime_environment_gate": deepcopy(dict(runtime_environment)),
        "harness_manifest": harness,
        "ma1_public_evidence_gate": ma1,
        "candidate_source_gate": candidate,
        "optional_docking_gate": optional_docking,
        "execution_boundary": {
            "read_only_public_sources_only": True,
            "model_calls": 0,
            "llm_calls": 0,
            "reviewer_calls": 0,
            "model_tokens": 0,
            "model_cost_usd": 0,
            "api_server_calls": 0,
            "run_db_reads": 0,
            "run_db_writes": 0,
            "candidate_injection_count": 0,
            "pmid_injection_count": 0,
            "fixture_calls": fixture_finding_count,
            "fixture_finding_count": fixture_finding_count,
            "raw_response_bodies_stored": False,
            "full_abstracts_or_papers_stored": False,
            "secret_values_stored": False,
            "injected_runtime_count": 0 if execution_mode == "live" else 1,
        },
        "blockers": blockers,
        "interpretation": {
            "technical_ready_does_not_permit_paid_run": True,
            "g0_or_g1_pass_guaranteed": False,
            "candidate_success_claimed": False,
            "clinical_efficacy_or_safety_claimed": False,
            "human_gate_auto_submit": False,
        },
    }
    _assert_safe_report(report)
    report["report_sha256"] = _canonical_sha256(report)
    return report


def run_preflight(
    request_path: Path = DEFAULT_REQUEST,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Only live path; no runtime or transport injection seam."""

    context = load_validation_context(request_path, protocol_path)
    environment = _require_runtime_environment(context)
    runtime = _LiveRuntime(context)
    try:
        return _execute(
            context,
            runtime,
            execution_mode="live",
            runtime_environment=environment,
        )
    finally:
        runtime.close()


def run_contract_test(
    request_path: Path,
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    runtime: Any,
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Synthetic test seam that can never authorize or submit a paid run."""

    context = load_validation_context(
        request_path,
        protocol_path,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    expected = _mapping(context.registry.get("runtime"))
    gate = _runtime_environment_gate(expected, runtime_environment)
    try:
        return _execute(
            context,
            runtime,
            execution_mode="contract_test",
            runtime_environment=gate,
        )
    finally:
        method = getattr(runtime, "close", None)
        if callable(method):
            method()


def _validated_output_path(raw: str) -> Path:
    path = Path(raw).resolve()
    try:
        path.relative_to(RESULTS_DIR)
    except ValueError as exc:
        raise PreflightContractError("OUTPUT_PATH_OUTSIDE_RESULTS_DIR") from exc
    if path.parent != RESULTS_DIR or not RESULT_NAME_RE.fullmatch(path.name):
        raise PreflightContractError("OUTPUT_PATH_PATTERN_MISMATCH")
    return path


def _reserve_output(raw: str) -> tuple[Path, TextIO]:
    path = _validated_output_path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.open("x", encoding="utf-8", newline="\n")


def _attest_output_reservation(report: Mapping[str, Any], path: Path) -> dict[str, Any]:
    body = deepcopy(dict(report))
    body["output_record"] = {
        "file_name": path.name,
        "exclusive_create_reserved_before_context_runtime_network": True,
        "held_handle_used_for_terminal_record": True,
        "existing_record_overwritten": False,
    }
    body.pop("report_sha256", None)
    _assert_safe_report(body)
    body["report_sha256"] = _canonical_sha256(body)
    return body


def emit_report(report: Mapping[str, Any], handle: TextIO) -> None:
    json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _fatal_report(exc: Exception) -> dict[str, Any]:
    reason = exc.code if isinstance(exc, PreflightContractError) else "PREFLIGHT_FATAL"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "execution_mode": "live",
        "decision": "PRECHECK_NOT_READY",
        "technical_ready": False,
        "paid_run_authorized": False,
        "automatic_submission_allowed": False,
        "explicit_user_authorization_required": True,
        "fatal_error": {
            "reason_code": reason,
            "exception_type": type(exc).__name__,
        },
        "execution_boundary": {
            "model_calls": 0,
            "api_server_calls": 0,
            "run_db_reads": 0,
            "run_db_writes": 0,
            "paid_posts": 0,
        },
    }
    if isinstance(exc, RuntimeEnvironmentMismatch):
        report["runtime_environment_gate"] = deepcopy(exc.snapshot)
    _assert_safe_report(report)
    report["report_sha256"] = _canonical_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_REQUEST))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", required=True)
    return parser


def _terminal_exit_code(report: Mapping[str, Any]) -> int:
    if report.get("fatal_error"):
        return 1
    if (
        report.get("execution_mode") == "contract_test"
        and report.get("decision") == "CONTRACT_TEST_ONLY"
        and report.get("paid_run_authorized") is False
    ):
        return 0
    if (
        report.get("execution_mode") == "live"
        and
        report.get("decision")
        == "PREFLIGHT_READY_FOR_EXPLICIT_USER_AUTHORIZATION"
        and report.get("technical_ready") is True
        and report.get("paid_run_authorized") is False
        and report.get("automatic_submission_allowed") is False
    ):
        return 0
    if (
        report.get("decision") == "PRECHECK_NOT_READY"
        and report.get("technical_ready") is False
        and report.get("paid_run_authorized") is False
    ):
        return 2
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path, handle = _reserve_output(args.output)
    except Exception:
        return 1
    with handle:
        try:
            report = run_preflight(
                Path(args.input), protocol_path=Path(args.protocol)
            )
        except Exception as exc:
            report = _fatal_report(exc)
        terminal = _attest_output_reservation(report, path)
        emit_report(terminal, handle)
    return _terminal_exit_code(terminal)


if __name__ == "__main__":
    raise SystemExit(main())
