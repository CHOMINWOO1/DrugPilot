"""One-shot, fixture-free free preflight for the JAK2/ET paper-native-v5 run.

The live CLI owns its read-only ChEMBL and PubMed transports and exposes no
client, gateway, fixture, offline, model, database, or API-server injection
option.  Unit tests use :func:`run_contract_test`, which is permanently marked
``CONTRACT_TEST_ONLY`` and can never authorize a paid run.

Only bounded structured provenance, hashes, one accepted evidence sentence,
and secret-free logical call ledgers are retained.  Raw response bodies and
full abstracts are never returned by this harness.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from time import monotonic
from typing import Any, Literal, TextIO
import unicodedata
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pydantic import BaseModel, Field, model_validator  # noqa: E402

from drugpilot.agents.ma3 import MA3AssuranceAgent  # noqa: E402
from drugpilot.config import Settings  # noqa: E402
from drugpilot.domain.models import (  # noqa: E402
    CandidateCard,
    CandidatePacket,
    CandidateRoute,
    DataSourcePolicy,
    FaultType,
    HypothesisDirection,
    ProjectConstraints,
    ResearchBrief,
    ResearchDepth,
    ReservedControlCompound,
    ScientificBackendKind,
    ScientificExecutionStatus,
    TargetCandidate,
    validate_hypothesis_direction_selection,
)
from drugpilot.tools.candidate_policy import (  # noqa: E402
    candidate_identity,
    screen_forbidden_drug,
    screen_reserved_control,
)
from drugpilot.tools.admet_ai_adapter import ADMETAIAdapter  # noqa: E402
from drugpilot.tools.http import ScientificHttpClient  # noqa: E402
from drugpilot.tools.live_ma2_gateway import LiveScientificGateway  # noqa: E402
from drugpilot.tools.primary_abstract_paper_native_search_v5 import (  # noqa: E402
    PAPER_NATIVE_V5_POLICY_TOKEN,
    PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
    PUBMED_FETCH_URL,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
)
from drugpilot.tools.primary_abstract_paper_native_v5 import (  # noqa: E402
    PAPER_NATIVE_V5_GRAMMAR,
    parse_paper_native_primary_abstract_v5,
)


SCHEMA_VERSION = "jak2_et_paper_native_v5_free_preflight/v2"
DEFAULT_REQUEST = (
    ROOT
    / "docs"
    / "validation"
    / "JAK2_ET_PAPER_NATIVE_V5_LIVE_E2E_REQUEST_2026-08-27.json"
)
DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "validation"
    / "JAK2_ET_PAPER_NATIVE_V5_LIVE_E2E_PROTOCOL_KO_R2_2026-08-27.md"
)
R1_PROTOCOL = (
    ROOT
    / "docs"
    / "validation"
    / "JAK2_ET_PAPER_NATIVE_V5_LIVE_E2E_PROTOCOL_KO_2026-08-27.md"
)
PRIOR_NOT_READY_RESULT = (
    ROOT
    / "docs"
    / "validation"
    / "results"
    / "JAK2_ET_PAPER_NATIVE_V5_FREE_PREFLIGHT_20260827T060944Z.json"
)
# These constants are the external immutable R2/R1/result document register.
# Harness/test hashes are recorded only in each result artifact, so binding
# these document bytes cannot create a manifest cycle.
EXPECTED_REQUEST_SHA256 = (
    "8132358076355bfa8e0db2688d656b9d400d027bf5de0a1f2b7a363efe9743c3"
)
EXPECTED_PROTOCOL_SHA256 = (
    "26a9dfa535c18708a2aea27dfec783eefa5bbabaae6040e5c52b152f3d739199"
)
EXPECTED_R1_PROTOCOL_SHA256 = (
    "f5967c02bc9f301f5a9cc0b708cf03646ea0db3fe0b394af93ce3b354688deeb"
)
EXPECTED_PRIOR_RESULT_FILE_SHA256 = (
    "8f4c7388b3155ba798d2a63163c44ace2bc83ef24e6ba3b3aa1f767e72564820"
)
EXPECTED_PRIOR_RESULT_REPORT_SHA256 = (
    "52fbc8ddde85c4285217f26f10cc2ee33fb45dad39b8566493fc1b5c4b065e7f"
)

SOURCE_ROLE_PATHS: Mapping[str, str] = {
    "v5 parser": "src/drugpilot/tools/primary_abstract_paper_native_v5.py",
    "v5 PubMed search": (
        "src/drugpilot/tools/primary_abstract_paper_native_search_v5.py"
    ),
    "live MA2 gateway": "src/drugpilot/tools/live_ma2_gateway.py",
    "repurposing wrapper": "src/drugpilot/tools/live_repurposing_gateway.py",
    "orchestration graph": "src/drugpilot/orchestration/graph.py",
    "call accounting trace": "src/drugpilot/api/call_trace.py",
    "budget guard": "src/drugpilot/orchestration/budget.py",
    "MA2": "src/drugpilot/agents/ma2.py",
    "MA3": "src/drugpilot/agents/ma3.py",
    "MA3 reviewer projection": "src/drugpilot/agents/ma3_reviewer_projection.py",
    "MA1": "src/drugpilot/agents/ma1.py",
    "public biology": "src/drugpilot/tools/public_biology.py",
    "LLM provider usage": "src/drugpilot/providers/llm.py",
    "result UI accounting": "src/drugpilot/web/static/app.js",
    "result UI cache binding": "src/drugpilot/web/templates/index.html",
}
HARNESS_PATHS: Mapping[str, str] = {
    "free_preflight_harness": (
        "scripts/jak2_et_paper_native_v5_free_preflight.py"
    ),
    "free_preflight_tests": (
        "tests/unit/test_jak2_et_paper_native_v5_free_preflight.py"
    ),
}
RESULTS_DIR = (ROOT / "docs" / "validation" / "results").resolve()
RESULT_NAME = re.compile(
    r"^JAK2_ET_PAPER_NATIVE_V5_FREE_PREFLIGHT_\d{8}T\d{6}Z\.json$"
)
SOURCE_ROW_RE = re.compile(
    r"^\|\s*(?P<role>[^|]+?)\s*\|\s*`(?P<path>[^`]+)`\s*\|\s*"
    r"`(?P<sha>[0-9A-Fa-f]{64})`\s*\|\s*$",
    re.MULTILINE,
)
REQUEST_HASH_RE = re.compile(
    r"요청 파일 SHA-256:\s*`(?P<sha>[0-9A-Fa-f]{64})`"
)
PRODUCTION_TREE_COUNT_RE = re.compile(
    r"Production Python tree file count:\s*`(?P<count>[1-9][0-9]*)`"
)
PRODUCTION_TREE_HASH_RE = re.compile(
    r"Production Python tree canonical SHA-256:\s*"
    r"`(?P<sha>[0-9A-Fa-f]{64})`"
)
R1_PROTOCOL_HASH_RE = re.compile(
    r"R1 protocol SHA-256:\s*`(?P<sha>[0-9A-Fa-f]{64})`"
)
PRIOR_RESULT_FILE_HASH_RE = re.compile(
    r"선행 NOT_READY result file SHA-256:\s*"
    r"`(?P<sha>[0-9A-Fa-f]{64})`"
)
PRIOR_RESULT_REPORT_HASH_RE = re.compile(
    r"선행 NOT_READY report SHA-256:\s*"
    r"`(?P<sha>[0-9A-Fa-f]{64})`"
)
CHEMBL_ID_RE = re.compile(r"^CHEMBL[0-9]+$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
REASON_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_.:-]{0,127}$")
SENSITIVE_PARAMETER_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)

TARGET_SYMBOL = "JAK2"
TARGET_UNIPROT = "O60674"
RESERVED_NAME = "RUXOLITINIB"
RESERVED_CHEMBL_ID = "CHEMBL1789941"
MECHANISM_ROW_LIMIT = 80
FIXED_POOL_LIMIT = 12
ACTIVITY_ROW_LIMIT = 20
ALIAS_LIMIT = 16
PUBMED_DOCUMENT_LIMIT = 8
PUBMED_LOGICAL_CALL_LIMIT = 3
CHEMBL_OPERATION_TIMEOUT_SECONDS = 300.0
SOURCE_HTTP_TIMEOUT_SECONDS = 30.0
CHEMBL_MAX_ATTEMPTS = 2
PUBMED_MAX_ATTEMPTS = 1
PROMOTED_NONCONTROL_LIMIT = 3
DIRECT_SUPPORT_RETAINED_ROW_LIMIT = ACTIVITY_ROW_LIMIT
REASON_CODE_LIST_LIMIT = 32
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_PACKAGE_VERSIONS: Mapping[str, str] = {
    "rdkit": "2026.03.5",
    "pydantic": "2.13.4",
    "httpx": "0.28.1",
}
EXPECTED_WORKSPACE_INTERPRETER_RELATIVE_PATH = ".venv/Scripts/python.exe"
EXPECTED_PAID_CAPS: Mapping[str, int | float] = {
    "max_llm_calls": 20,
    "max_run_tokens": 2_000_000,
    "hard_max_run_tokens": 2_500_000,
    "max_run_cost_usd": 5.0,
    "max_single_input_tokens": 250_000,
}
EXPECTED_RUNTIME_SETTINGS: Mapping[str, Any] = {
    "configured_mode": "live",
    "effective_mode": "live",
    "mode_auto_activated": False,
    "ma2_backend_mode": "live",
    "ma2_fixture_backend_configured": False,
    "ma2_live_scientific_tools_enabled": True,
    "demo_fixture_fallback_enabled": False,
    "public_evidence_enabled": True,
    "ma1_public_nonfixture_evidence_configured": True,
    "ma2_chembl_timeout_seconds": SOURCE_HTTP_TIMEOUT_SECONDS,
    "ma2_chembl_operation_timeout_seconds": (
        CHEMBL_OPERATION_TIMEOUT_SECONDS
    ),
    "ma2_chembl_max_attempts": CHEMBL_MAX_ATTEMPTS,
    **EXPECTED_PAID_CAPS,
}


class PreflightContractError(ValueError):
    """Fail-closed immutable-input or provenance-contract violation."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RuntimeEnvironmentMismatch(PreflightContractError):
    """A secret-free runtime/config mismatch detected before live clients."""

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        super().__init__("RUNTIME_ENVIRONMENT_MISMATCH")
        self.runtime_environment_gate = deepcopy(dict(snapshot))


@dataclass(frozen=True, slots=True)
class ValidationContext:
    request_path: Path
    request_sha256: str
    protocol_path: Path
    protocol_sha256: str
    r1_protocol_path: Path
    r1_protocol_sha256: str
    prior_result_path: Path
    prior_result_file_sha256: str
    prior_result_report_sha256: str
    source_manifest: tuple[dict[str, str], ...]
    production_tree_file_count: int
    production_tree_manifest_sha256: str
    create_run_request_model_sha256: str
    raw: dict[str, Any]
    brief: ResearchBrief
    reserved_controls: tuple[ReservedControlCompound, ...]


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


def _jsonable(value: Any) -> Any:
    return json.loads(_canonical_json(value).decode("utf-8"))


def _without_hash(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: deepcopy(item) for name, item in value.items() if name != key}


def _hash_field_valid(value: Any, key: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and isinstance(value.get(key), str)
        and HASH_RE.fullmatch(str(value.get(key)))
        and value.get(key) == _canonical_sha256(_without_hash(value, key))
    )


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _safe_endpoint(url: str) -> str:
    parsed = urlsplit(str(url))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _safe_parameter_audit(
    kwargs: Mapping[str, Any],
) -> tuple[list[str], str, str | None]:
    retained: dict[str, Any] = {}
    for container in ("params", "json", "data"):
        value = kwargs.get(container)
        if not isinstance(value, Mapping):
            continue
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            clean[str(key)] = (
                "[REDACTED]"
                if any(marker in normalized for marker in SENSITIVE_PARAMETER_MARKERS)
                else item
            )
        retained[container] = clean
    keys = sorted(
        f"{container}.{key}"
        for container, value in retained.items()
        if isinstance(value, Mapping)
        for key in value
    )
    params = retained.get("params")
    source_parameter_sha256 = (
        sha256(
            json.dumps(
                dict(params),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if isinstance(params, Mapping)
        else None
    )
    return keys, _canonical_sha256(retained), source_parameter_sha256


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


class AuditedHttpClient:
    """Record secret-free logical HTTP calls without retaining response bodies."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.max_attempts = int(getattr(client, "max_attempts", 1) or 1)
        self.calls: list[dict[str, Any]] = []

    def recent_failures(self) -> tuple[Any, ...]:
        method = getattr(self._client, "recent_failures", None)
        return tuple(method()) if callable(method) else ()

    def get(self, url: str, **kwargs: Any) -> Any:
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        return self._request("POST", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        started = monotonic()
        (
            parameter_keys,
            parameter_sha256,
            source_parameter_sha256,
        ) = _safe_parameter_audit(kwargs)
        failures_before = len(self.recent_failures())
        base = {
            "method": method,
            "endpoint": _safe_endpoint(url),
            "parameter_keys": parameter_keys,
            "parameter_sha256": parameter_sha256,
            "source_parameter_sha256": source_parameter_sha256,
            "fixture": False,
        }
        try:
            response = getattr(self._client, method.casefold())(url, **kwargs)
        except Exception as exc:
            failures = self.recent_failures()
            retry_failures = max(0, len(failures) - failures_before)
            self.calls.append(
                {
                    **base,
                    "status": "error",
                    "status_code": None,
                    "response_sha256": None,
                    "exception_type": type(exc).__name__,
                    "attempt_count_observed": max(1, retry_failures),
                    "retry_diagnostics": _failure_diagnostics(
                        failures[failures_before:]
                    ),
                    "runtime_seconds": round(monotonic() - started, 6),
                }
            )
            raise
        failures = self.recent_failures()
        retry_failures = max(0, len(failures) - failures_before)
        status_code = getattr(response, "status_code", None)
        self.calls.append(
            {
                **base,
                "status": "completed",
                "status_code": status_code if isinstance(status_code, int) else None,
                "response_sha256": sha256(_response_bytes(response)).hexdigest(),
                "attempt_count_observed": max(
                    1,
                    retry_failures
                    + (1 if isinstance(status_code, int) and status_code < 400 else 0),
                ),
                "retry_diagnostics": _failure_diagnostics(
                    failures[failures_before:]
                ),
                "runtime_seconds": round(monotonic() - started, 6),
            }
        )
        return response

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def _failure_diagnostics(values: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        source = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        if not isinstance(source, Mapping):
            continue
        result.append(
            {
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
        )
    return result


class _NoDockingPreflightBackend:
    source_version = "docking=not_invoked_free_preflight"
    last_result = None

    def readiness(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "tool": "docking",
            "status": "unavailable",
            "required": False,
            "fixture": False,
            "reason": "optional and intentionally not invoked in free preflight",
        }

    def dock(self, *_: Any, **__: Any) -> float:
        raise RuntimeError("docking is outside the free-preflight boundary")


class _LiveRuntime:
    """Internally owned production gateway; constructed only by the live path."""

    def __init__(self) -> None:
        self.chembl_client = AuditedHttpClient(
            ScientificHttpClient(
                timeout_seconds=SOURCE_HTTP_TIMEOUT_SECONDS,
                max_attempts=CHEMBL_MAX_ATTEMPTS,
                retry_backoff_seconds=0.25,
                max_retry_delay_seconds=2.0,
            )
        )
        self.pubmed_client = AuditedHttpClient(
            ScientificHttpClient(
                timeout_seconds=SOURCE_HTTP_TIMEOUT_SECONDS,
                max_attempts=PUBMED_MAX_ATTEMPTS,
                retry_backoff_seconds=0.0,
                max_retry_delay_seconds=0.0,
            )
        )
        self.admet = ADMETAIAdapter()
        self.gateway = LiveScientificGateway(
            docking=_NoDockingPreflightBackend(),
            admet=self.admet,
            client=self.chembl_client,
            primary_search_client=self.pubmed_client,
            timeout_seconds=SOURCE_HTTP_TIMEOUT_SECONDS,
            max_http_attempts=CHEMBL_MAX_ATTEMPTS,
            retry_backoff_seconds=0.25,
            max_retry_backoff_seconds=2.0,
            operation_timeout_seconds=CHEMBL_OPERATION_TIMEOUT_SECONDS,
            repurposing_evidence_policy=PAPER_NATIVE_V5_POLICY_TOKEN,
        )

    def chembl_calls(self) -> list[dict[str, Any]]:
        return deepcopy(self.chembl_client.calls)

    def pubmed_calls(self) -> list[dict[str, Any]]:
        return deepcopy(self.pubmed_client.calls)

    def admet_readiness(self, seeds: Sequence[Any]) -> dict[str, Any]:
        package_readiness = self.admet.readiness()
        checks: list[dict[str, Any]] = []
        try:
            from rdkit import Chem
            from rdkit.Chem import Crippen, Descriptors, Lipinski

            for seed in seeds:
                metadata = getattr(seed, "metadata", {})
                chembl_id = str(metadata.get("chembl_id") or "")
                smiles = str(getattr(seed, "smiles", "") or "")
                expected_inchikey = str(metadata.get("standard_inchi_key") or "")
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
                observed_inchikey = (
                    Chem.MolToInchiKey(molecule) if molecule is not None else ""
                )
                descriptors: tuple[float, ...] = ()
                if molecule is not None:
                    descriptors = (
                        float(Descriptors.MolWt(molecule)),
                        float(Crippen.MolLogP(molecule)),
                        float(Lipinski.NumHDonors(molecule)),
                        float(Lipinski.NumHAcceptors(molecule)),
                    )
                checks.append(
                    {
                        "candidate_id": chembl_id,
                        "input_sha256": sha256(smiles.encode("utf-8")).hexdigest(),
                        "source_inchikey_present": bool(expected_inchikey),
                        "rdkit_parse_completed": molecule is not None,
                        "exact_inchikey_match": bool(
                            expected_inchikey
                            and observed_inchikey == expected_inchikey
                        ),
                        "basic_descriptor_probe_completed": bool(
                            descriptors
                            and all(math.isfinite(value) for value in descriptors)
                        ),
                    }
                )
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason_code": "ADMET_LOCAL_READINESS_EXCEPTION",
                "exception_type": type(exc).__name__,
                "package_readiness": _bounded_readiness(package_readiness),
                "candidate_structure_checks": checks,
                "admet_inference_executed": False,
                "fixture": False,
            }
        ready = bool(
            package_readiness.get("status") == "completed"
            and checks
            and all(
                item["source_inchikey_present"]
                and item["rdkit_parse_completed"]
                and item["exact_inchikey_match"]
                and item["basic_descriptor_probe_completed"]
                for item in checks
            )
        )
        return {
            "status": "completed" if ready else "unavailable",
            "reason_code": None if ready else "ADMET_LOCAL_NOT_READY",
            "package_readiness": _bounded_readiness(package_readiness),
            "candidate_structure_checks": checks,
            "admet_inference_executed": False,
            "fixture": False,
        }

    def close(self) -> None:
        self.chembl_client.close()
        self.pubmed_client.close()
        close = getattr(self.admet, "close", None)
        if callable(close):
            close()


def _bounded_readiness(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "unavailable", "fixture": False}
    provenance = value.get("provenance")
    return {
        "tool": value.get("tool"),
        "operation": value.get("operation"),
        "status": value.get("status"),
        "backend": value.get("backend"),
        "version": value.get("version"),
        "fixture": value.get("fixture"),
        "reason": value.get("reason"),
        "provenance": {
            "packages": deepcopy(provenance.get("packages", {})),
            "checkpoint_count": provenance.get("checkpoint_count"),
            "worker_started": provenance.get("worker_started"),
        }
        if isinstance(provenance, Mapping)
        else {},
    }


def _extract_create_run_request(raw: Mapping[str, Any]) -> str:
    """Validate with only the AST-extracted model; never import the API app."""

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
        "__name__": "_paper_native_v5_preflight_schema",
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
    return sha256(ast.dump(nodes[0], include_attributes=False).encode("utf-8")).hexdigest()


def _source_manifest(protocol_text: str) -> tuple[dict[str, str], ...]:
    observed: dict[str, tuple[str, str]] = {}
    for match in SOURCE_ROW_RE.finditer(protocol_text):
        role = match.group("role").strip()
        path = match.group("path")
        digest = match.group("sha").lower()
        if role in SOURCE_ROLE_PATHS:
            if role in observed:
                raise PreflightContractError("SOURCE_MANIFEST_DUPLICATE_ROLE")
            observed[role] = (path, digest)
    if set(observed) != set(SOURCE_ROLE_PATHS):
        raise PreflightContractError("SOURCE_MANIFEST_ROLE_SET_MISMATCH")
    records: list[dict[str, str]] = []
    for role, expected_path in SOURCE_ROLE_PATHS.items():
        path, expected_sha = observed[role]
        if path != expected_path:
            raise PreflightContractError("SOURCE_MANIFEST_PATH_MISMATCH")
        absolute = (ROOT / path).resolve()
        try:
            absolute.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PreflightContractError("SOURCE_MANIFEST_PATH_ESCAPE") from exc
        actual_sha = _sha256_file(absolute)
        if actual_sha != expected_sha:
            raise PreflightContractError("SOURCE_MANIFEST_HASH_MISMATCH")
        records.append(
            {
                "role": role,
                "path": path,
                "sha256": actual_sha,
            }
        )
    return tuple(records)


def _production_tree_manifest() -> dict[str, Any]:
    root = ROOT / "src" / "drugpilot"
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            root.rglob("*.py"),
            key=lambda item: item.relative_to(ROOT).as_posix(),
        )
        if path.is_file()
    ]
    if not records:
        raise PreflightContractError("PRODUCTION_TREE_MANIFEST_EMPTY")
    return {
        "canonicalization": (
            "UTF-8 canonical JSON array sorted by POSIX relative path; "
            "each row is {path,sha256}"
        ),
        "file_count": len(records),
        "manifest_sha256": _canonical_sha256(records),
    }


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    value = getattr(module, "__version__", None)
    return value if isinstance(value, str) and value else None


def _module_origin_is_workspace_venv(module_name: str) -> bool:
    """Attest package origin without retaining an absolute local path."""

    try:
        module = importlib.import_module(module_name)
        origin = Path(module.__file__).resolve()
        site_packages = (ROOT / ".venv" / "Lib" / "site-packages").resolve()
        return origin.is_relative_to(site_packages)
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _expected_runtime_environment() -> dict[str, Any]:
    return {
        "python_version": EXPECTED_PYTHON_VERSION,
        "workspace_interpreter_relative_path": (
            EXPECTED_WORKSPACE_INTERPRETER_RELATIVE_PATH
        ),
        "package_versions": dict(EXPECTED_PACKAGE_VERSIONS),
        "package_module_origins_within_workspace_venv": {
            name: True for name in EXPECTED_PACKAGE_VERSIONS
        },
        "settings": dict(EXPECTED_RUNTIME_SETTINGS),
    }


def _runtime_environment_check_results(
    observed: Any,
) -> dict[str, bool]:
    body = observed if isinstance(observed, Mapping) else {}
    packages = body.get("package_versions")
    package_origins = body.get(
        "package_module_origins_within_workspace_venv"
    )
    settings = body.get("settings")
    packages = packages if isinstance(packages, Mapping) else {}
    package_origins = (
        package_origins if isinstance(package_origins, Mapping) else {}
    )
    settings = settings if isinstance(settings, Mapping) else {}
    return {
        "python_version_exact": body.get("python_version")
        == EXPECTED_PYTHON_VERSION,
        "workspace_venv_python_exact": (
            body.get("workspace_venv_python") is True
        ),
        **{
            f"{name}_version_exact": packages.get(name) == expected
            for name, expected in EXPECTED_PACKAGE_VERSIONS.items()
        },
        **{
            f"{name}_workspace_venv_origin": (
                package_origins.get(name) is True
            )
            for name in EXPECTED_PACKAGE_VERSIONS
        },
        "settings_loaded": (
            body.get("settings_validation_exception_type") is None
        ),
        **{
            f"settings_{name}_exact": settings.get(name) == expected
            and type(settings.get(name)) is type(expected)
            for name, expected in EXPECTED_RUNTIME_SETTINGS.items()
        },
    }


def _runtime_environment_audit() -> dict[str, Any]:
    """Collect only the allow-listed, non-secret effective runtime fields."""

    expected_interpreter = (
        ROOT / EXPECTED_WORKSPACE_INTERPRETER_RELATIVE_PATH
    ).resolve()
    try:
        workspace_venv_python = Path(sys.executable).resolve() == (
            expected_interpreter
        )
    except (OSError, RuntimeError):
        workspace_venv_python = False
    observed_packages = {
        name: _module_version(name)
        for name in EXPECTED_PACKAGE_VERSIONS
    }
    observed_package_origins = {
        name: _module_origin_is_workspace_venv(name)
        for name in EXPECTED_PACKAGE_VERSIONS
    }
    settings_error_type: str | None = None
    observed_settings: dict[str, Any] = {
        "configured_mode": None,
        "effective_mode": None,
        "mode_auto_activated": None,
        "ma2_backend_mode": None,
        "ma2_fixture_backend_configured": None,
        "ma2_live_scientific_tools_enabled": None,
        "demo_fixture_fallback_enabled": None,
        "public_evidence_enabled": None,
        "ma1_public_nonfixture_evidence_configured": None,
        "ma2_chembl_timeout_seconds": None,
        "ma2_chembl_operation_timeout_seconds": None,
        "ma2_chembl_max_attempts": None,
        **{name: None for name in EXPECTED_PAID_CAPS},
    }
    try:
        settings = Settings(_env_file=ROOT / ".env")
        configured_mode = _enum_value(settings.mode)
        effective_mode = _enum_value(settings.effective_mode)
        observed_settings = {
            "configured_mode": configured_mode,
            "effective_mode": effective_mode,
            "mode_auto_activated": configured_mode != effective_mode,
            "ma2_backend_mode": settings.ma2_backend_mode,
            "ma2_fixture_backend_configured": (
                settings.ma2_backend_mode == "fixture"
            ),
            "ma2_live_scientific_tools_enabled": (
                settings.ma2_live_scientific_tools_enabled
            ),
            "demo_fixture_fallback_enabled": (
                settings.demo_fixture_fallback_enabled
            ),
            "public_evidence_enabled": settings.public_evidence_enabled,
            "ma1_public_nonfixture_evidence_configured": (
                settings.public_evidence_enabled
                and not settings.demo_fixture_fallback_enabled
            ),
            "ma2_chembl_timeout_seconds": (
                settings.ma2_chembl_timeout_seconds
            ),
            "ma2_chembl_operation_timeout_seconds": (
                settings.ma2_chembl_operation_timeout_seconds
            ),
            "ma2_chembl_max_attempts": settings.ma2_chembl_max_attempts,
            **{
                name: getattr(settings, name)
                for name in EXPECTED_PAID_CAPS
            },
        }
    except Exception as exc:
        # Exception messages and Settings dumps may contain secret inputs.
        # Only the exception class is safe and useful for fail-closed audit.
        settings_error_type = type(exc).__name__

    observed = {
        "python_version": platform.python_version(),
        "workspace_venv_python": workspace_venv_python,
        "package_versions": observed_packages,
        "package_module_origins_within_workspace_venv": (
            observed_package_origins
        ),
        "settings": observed_settings,
        "settings_validation_exception_type": settings_error_type,
    }
    checks = _runtime_environment_check_results(observed)
    snapshot: dict[str, Any] = {
        "schema_version": "jak2_et_paper_native_v5_runtime_environment/v1",
        "expected": _expected_runtime_environment(),
        "observed": observed,
        "checks": checks,
        "verified": all(checks.values()),
        "settings_source": "drugpilot.config.Settings",
        "settings_resolution": (
            "workspace .env with normal Settings/process-environment precedence"
        ),
        "secret_values_included": False,
        "secret_field_names_included": [],
        "http_clients_or_live_runtime_constructed_by_audit": False,
    }
    snapshot["snapshot_sha256"] = _canonical_sha256(snapshot)
    return snapshot


def _runtime_environment_gate_valid(snapshot: Any) -> bool:
    """Recompute the complete snapshot contract before any live client."""

    if not isinstance(snapshot, Mapping):
        return False
    expected_top_keys = {
        "schema_version",
        "expected",
        "observed",
        "checks",
        "verified",
        "settings_source",
        "settings_resolution",
        "secret_values_included",
        "secret_field_names_included",
        "http_clients_or_live_runtime_constructed_by_audit",
        "snapshot_sha256",
    }
    observed = snapshot.get("observed")
    checks = snapshot.get("checks")
    if (
        set(snapshot) != expected_top_keys
        or not isinstance(observed, Mapping)
        or set(observed)
        != {
            "python_version",
            "workspace_venv_python",
            "package_versions",
            "package_module_origins_within_workspace_venv",
            "settings",
            "settings_validation_exception_type",
        }
        or not isinstance(observed.get("package_versions"), Mapping)
        or set(observed["package_versions"]) != set(EXPECTED_PACKAGE_VERSIONS)
        or not isinstance(
            observed.get("package_module_origins_within_workspace_venv"),
            Mapping,
        )
        or set(observed["package_module_origins_within_workspace_venv"])
        != set(EXPECTED_PACKAGE_VERSIONS)
        or not isinstance(observed.get("settings"), Mapping)
        or set(observed["settings"]) != set(EXPECTED_RUNTIME_SETTINGS)
        or not isinstance(checks, Mapping)
    ):
        return False
    recomputed_checks = _runtime_environment_check_results(observed)
    return bool(
        snapshot.get("schema_version")
        == "jak2_et_paper_native_v5_runtime_environment/v1"
        and snapshot.get("expected") == _expected_runtime_environment()
        and dict(checks) == recomputed_checks
        and checks
        and all(type(value) is bool and value is True for value in checks.values())
        and snapshot.get("verified") is True
        and snapshot.get("settings_source") == "drugpilot.config.Settings"
        and snapshot.get("settings_resolution")
        == "workspace .env with normal Settings/process-environment precedence"
        and snapshot.get("secret_values_included") is False
        and snapshot.get("secret_field_names_included") == []
        and snapshot.get(
            "http_clients_or_live_runtime_constructed_by_audit"
        )
        is False
        and _hash_field_valid(snapshot, "snapshot_sha256")
    )


def _require_runtime_environment() -> dict[str, Any]:
    snapshot = _runtime_environment_audit()
    if not _runtime_environment_gate_valid(snapshot):
        raise RuntimeEnvironmentMismatch(snapshot)
    return snapshot


def _validate_production_tree_freeze(protocol_text: str) -> dict[str, Any]:
    count_matches = list(PRODUCTION_TREE_COUNT_RE.finditer(protocol_text))
    hash_matches = list(PRODUCTION_TREE_HASH_RE.finditer(protocol_text))
    if len(count_matches) != 1 or len(hash_matches) != 1:
        raise PreflightContractError("PRODUCTION_TREE_REGISTER_MISMATCH")
    expected_count = int(count_matches[0].group("count"))
    expected_sha = hash_matches[0].group("sha").lower()
    actual = _production_tree_manifest()
    if actual["file_count"] != expected_count:
        raise PreflightContractError("PRODUCTION_TREE_FILE_COUNT_MISMATCH")
    if actual["manifest_sha256"] != expected_sha:
        raise PreflightContractError("PRODUCTION_TREE_HASH_MISMATCH")
    return actual


def _validate_r2_lineage(protocol_text: str) -> dict[str, Any]:
    """Verify the immutable R1 protocol/result chain before live runtime."""

    r1_relative = R1_PROTOCOL.relative_to(ROOT).as_posix()
    prior_relative = PRIOR_NOT_READY_RESULT.relative_to(ROOT).as_posix()
    if protocol_text.count(r1_relative) != 1:
        raise PreflightContractError("R2_R1_PROTOCOL_PATH_LINK_MISMATCH")
    if protocol_text.count(prior_relative) != 1:
        raise PreflightContractError("R2_PRIOR_RESULT_PATH_LINK_MISMATCH")
    embedded_contracts = (
        (
            R1_PROTOCOL_HASH_RE,
            EXPECTED_R1_PROTOCOL_SHA256,
            "R2_R1_PROTOCOL_HASH_LINK_MISMATCH",
        ),
        (
            PRIOR_RESULT_FILE_HASH_RE,
            EXPECTED_PRIOR_RESULT_FILE_SHA256,
            "R2_PRIOR_RESULT_FILE_HASH_LINK_MISMATCH",
        ),
        (
            PRIOR_RESULT_REPORT_HASH_RE,
            EXPECTED_PRIOR_RESULT_REPORT_SHA256,
            "R2_PRIOR_RESULT_REPORT_HASH_LINK_MISMATCH",
        ),
    )
    for pattern, expected, code in embedded_contracts:
        matches = list(pattern.finditer(protocol_text))
        if len(matches) != 1 or matches[0].group("sha").lower() != expected:
            raise PreflightContractError(code)

    if _sha256_file(R1_PROTOCOL) != EXPECTED_R1_PROTOCOL_SHA256:
        raise PreflightContractError("IMMUTABLE_R1_PROTOCOL_HASH_MISMATCH")
    prior_file_sha256 = _sha256_file(PRIOR_NOT_READY_RESULT)
    if prior_file_sha256 != EXPECTED_PRIOR_RESULT_FILE_SHA256:
        raise PreflightContractError("IMMUTABLE_PRIOR_RESULT_FILE_HASH_MISMATCH")
    try:
        prior = json.loads(PRIOR_NOT_READY_RESULT.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightContractError("IMMUTABLE_PRIOR_RESULT_JSON_INVALID") from exc
    output_record = _mapping(prior.get("output_record")) if isinstance(prior, Mapping) else {}
    if (
        not isinstance(prior, Mapping)
        or prior.get("schema_version")
        != "jak2_et_paper_native_v5_free_preflight/v1"
        or prior.get("report_sha256")
        != EXPECTED_PRIOR_RESULT_REPORT_SHA256
        or not _hash_field_valid(prior, "report_sha256")
        or prior.get("decision") != "PRECHECK_NOT_READY"
        or prior.get("paid_run_authorized") is not False
        or prior.get("paid_run_stop_required") is not True
        or output_record.get("file_name") != PRIOR_NOT_READY_RESULT.name
    ):
        raise PreflightContractError("IMMUTABLE_PRIOR_RESULT_CONTRACT_MISMATCH")
    return {
        "r1_protocol_path": r1_relative,
        "r1_protocol_sha256": EXPECTED_R1_PROTOCOL_SHA256,
        "prior_result_path": prior_relative,
        "prior_result_file_sha256": prior_file_sha256,
        "prior_result_report_sha256": EXPECTED_PRIOR_RESULT_REPORT_SHA256,
        "prior_result_decision": "PRECHECK_NOT_READY",
        "prior_paid_run_authorized": False,
        "lineage_verified_before_http": True,
    }


def _harness_manifest() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for role, relative in HARNESS_PATHS.items():
        absolute = (ROOT / relative).resolve()
        try:
            absolute.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PreflightContractError("HARNESS_MANIFEST_PATH_ESCAPE") from exc
        records.append(
            {
                "role": role,
                "path": relative,
                "exists": absolute.is_file(),
                "size_bytes": absolute.stat().st_size if absolute.is_file() else None,
                "sha256": _sha256_file(absolute) if absolute.is_file() else None,
            }
        )
    body: dict[str, Any] = {
        "schema_version": "paper_native_v5_free_preflight_harness_manifest/v1",
        "records": records,
        "excluded_from_protocol_production_source_manifest": True,
        "binding_stage": "preflight_result_then_paid_run_audit_appendix",
        "subsequent_paid_audit_must_match_exact_bytes": True,
        "verified": bool(
            records
            and all(
                item["exists"] is True
                and type(item["size_bytes"]) is int
                and item["size_bytes"] > 0
                and isinstance(item["sha256"], str)
                and HASH_RE.fullmatch(item["sha256"])
                for item in records
            )
        ),
    }
    body["manifest_sha256"] = _canonical_sha256(body)
    return body


def load_validation_context(
    request_path: Path = DEFAULT_REQUEST,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> ValidationContext:
    request_path = request_path.resolve()
    protocol_path = protocol_path.resolve()
    request_bytes = request_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    request_sha = sha256(request_bytes).hexdigest()
    protocol_sha = sha256(protocol_bytes).hexdigest()
    if request_sha != EXPECTED_REQUEST_SHA256:
        raise PreflightContractError("IMMUTABLE_REQUEST_HASH_MISMATCH")
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise PreflightContractError("IMMUTABLE_PROTOCOL_HASH_MISMATCH")
    protocol_text = protocol_bytes.decode("utf-8")
    embedded = list(REQUEST_HASH_RE.finditer(protocol_text))
    if len(embedded) != 1 or embedded[0].group("sha").lower() != request_sha:
        raise PreflightContractError("PROTOCOL_REQUEST_HASH_LINK_MISMATCH")
    lineage = _validate_r2_lineage(protocol_text)
    source_manifest = _source_manifest(protocol_text)
    production_tree = _validate_production_tree_freeze(protocol_text)
    try:
        raw = json.loads(request_bytes.decode("utf-8"))
    except Exception as exc:
        raise PreflightContractError("REQUEST_JSON_INVALID") from exc
    if not isinstance(raw, dict):
        raise PreflightContractError("REQUEST_JSON_NOT_OBJECT")
    model_sha = _extract_create_run_request(raw)
    _validate_exact_request(raw)
    reserved = tuple(
        ReservedControlCompound.model_validate(item)
        for item in raw["reserved_control_compounds"]
    )
    constraints = ProjectConstraints(
        target_cell_type=raw["target_cell_type"],
        intervention_direction=raw["intervention_direction"],
        chronic_safety_required=raw["chronic_safety_required"],
        druggability_required=raw["druggability_required"],
        discovery_priority=raw["discovery_priority"],
        forbidden_targets=raw["forbidden_targets"],
        forbidden_substructures=raw["forbidden_substructures"],
        forbidden_drugs=raw["forbidden_drugs"],
        forbidden_drug_ids=raw["forbidden_drug_ids"],
        forbidden_drug_classes=raw["forbidden_drug_classes"],
        reserved_control_compounds=list(reserved),
        candidates_per_route=raw["candidates_per_route"],
        max_ma1_iterations=raw["max_ma1_iterations"],
        max_rework_loops=raw["max_rework_loops"],
    )
    brief = ResearchBrief(
        brief_id=f"brief_jak2_et_v5_preflight_{request_sha[:12]}",
        research_goal=raw["research_goal"],
        disease=raw["disease"],
        rationale=raw["rationale"],
        constraints=constraints,
        data_policy=DataSourcePolicy(
            allowed_sources=["ChEMBL", "PubMed"],
            require_primary_sources=True,
        ),
        research_depth=raw["research_depth"],
        human_priorities=raw["human_priorities"],
        fault_injection=raw["fault_injection"],
    )
    return ValidationContext(
        request_path=request_path,
        request_sha256=request_sha,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha,
        r1_protocol_path=R1_PROTOCOL.resolve(),
        r1_protocol_sha256=str(lineage["r1_protocol_sha256"]),
        prior_result_path=PRIOR_NOT_READY_RESULT.resolve(),
        prior_result_file_sha256=str(lineage["prior_result_file_sha256"]),
        prior_result_report_sha256=str(
            lineage["prior_result_report_sha256"]
        ),
        source_manifest=source_manifest,
        production_tree_file_count=int(production_tree["file_count"]),
        production_tree_manifest_sha256=str(
            production_tree["manifest_sha256"]
        ),
        create_run_request_model_sha256=model_sha,
        raw=raw,
        brief=brief,
        reserved_controls=reserved,
    )


def _validate_exact_request(raw: Mapping[str, Any]) -> None:
    exact = {
        "disease": "essential thrombocythemia",
        "research_depth": "comprehensive",
        "intervention_direction": "inhibit",
        "chronic_safety_required": False,
        "druggability_required": True,
        "discovery_priority": "repurposing_target",
        "candidates_per_route": 3,
        "max_ma1_iterations": 3,
        "max_rework_loops": 2,
        "fault_injection": "none",
    }
    if any(type(raw.get(key)) is not type(value) or raw.get(key) != value for key, value in exact.items()):
        raise PreflightContractError("IMMUTABLE_REQUEST_FIELD_MISMATCH")
    goal = str(raw.get("research_goal") or "")
    cell = str(raw.get("target_cell_type") or "")
    if not all(term in goal for term in ("JAK2", TARGET_UNIPROT, "JAK2-V617F")):
        raise PreflightContractError("EXACT_TARGET_SCOPE_MISSING")
    if not all(term in cell for term in ("SET-2", "JAK2-V617F", "isogenic JAK2-WT")):
        raise PreflightContractError("EXACT_MODEL_SCOPE_MISSING")
    for key in (
        "forbidden_targets",
        "forbidden_substructures",
        "forbidden_drugs",
        "forbidden_drug_ids",
        "forbidden_drug_classes",
    ):
        if raw.get(key) != []:
            raise PreflightContractError("REQUEST_ALLOWLIST_OR_FORBIDDEN_SEED_PRESENT")
    priorities = raw.get("human_priorities")
    if (
        not isinstance(priorities, list)
        or priorities.count(PAPER_NATIVE_V5_POLICY_TOKEN) != 1
    ):
        raise PreflightContractError("PAPER_NATIVE_V5_TOKEN_COUNT_MISMATCH")
    reserved = raw.get("reserved_control_compounds")
    expected_reserved = [
        {
            "name": RESERVED_NAME,
            "registry_ids": [
                {"namespace": "ChEMBL", "identifier": RESERVED_CHEMBL_ID}
            ],
        }
    ]
    if reserved != expected_reserved:
        raise PreflightContractError("RESERVED_CONTROL_MISMATCH")
    serialized = json.dumps(raw, ensure_ascii=False)
    chembl_ids = {
        value.upper()
        for value in re.findall(r"CHEMBL[0-9]+", serialized, re.I)
    }
    if chembl_ids != {RESERVED_CHEMBL_ID}:
        raise PreflightContractError("PREDECLARED_CANDIDATE_ID_PRESENT")
    if re.search(r"\b(?:PMID|PMCID)\s*[:=]?\s*\d+", serialized, re.I):
        raise PreflightContractError("PREDECLARED_PUBLICATION_ID_PRESENT")
    if re.search(r"\b(?:fixture|fallback|injected[_ -]?client)\b", serialized, re.I):
        raise PreflightContractError("EXECUTABLE_FIXTURE_OR_FALLBACK_MARKER_PRESENT")


def _target_source(context: ValidationContext) -> TargetCandidate:
    return TargetCandidate(
        symbol=TARGET_SYMBOL,
        uniprot_id=TARGET_UNIPROT,
        target_name="Janus kinase 2",
        target_cell_type=context.raw["target_cell_type"],
        identity_status="resolved",
        identity_sources=["immutable_request"],
        metadata={"request_sha256": context.request_sha256, "fixture": False},
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bounded_reason_code(value: Any) -> tuple[str | None, bool]:
    """Return one secret-safe, bounded machine reason without free text."""

    if value is None:
        return None, True
    if isinstance(value, str) and REASON_CODE_RE.fullmatch(value):
        return value, True
    return "UNSAFE_OR_UNBOUNDED_REASON_CODE", False


def _bounded_reason_codes(value: Any) -> tuple[list[str], bool]:
    """Project a bounded reason-code list without leaking malformed values."""

    if not isinstance(value, list) or len(value) > REASON_CODE_LIST_LIMIT:
        return ["UNSAFE_OR_UNBOUNDED_REASON_CODES"], False
    if any(
        not isinstance(item, str) or REASON_CODE_RE.fullmatch(item) is None
        for item in value
    ):
        return ["UNSAFE_OR_UNBOUNDED_REASON_CODES"], False
    return list(value), True


def _bounded_sha256(value: Any) -> tuple[str | None, bool]:
    """Project only a canonical SHA-256 string, with optional absence."""

    if value is None:
        return None, True
    if isinstance(value, str) and HASH_RE.fullmatch(value):
        return value, True
    return None, False


def _seed_id(seed: Any) -> str:
    return str(_mapping(getattr(seed, "metadata", {})).get("chembl_id") or "")


def _seed_envelope(seed: Any) -> Mapping[str, Any]:
    metadata = _mapping(getattr(seed, "metadata", {}))
    envelope = metadata.get("primary_abstract_paper_native_v5")
    return _mapping(envelope)


def _exact_target_check(target: Any) -> dict[str, Any]:
    metadata = _mapping(getattr(target, "metadata", {}))
    checks = {
        "symbol": getattr(target, "symbol", None) == TARGET_SYMBOL,
        "uniprot": getattr(target, "uniprot_id", None) == TARGET_UNIPROT,
        "resolved": getattr(target, "identity_status", None) == "resolved",
        "target_id_present": bool(metadata.get("chembl_target_id")),
        "single_protein": metadata.get("chembl_target_type") == "SINGLE PROTEIN",
        "human": metadata.get("chembl_organism") == "Homo sapiens",
        "fixture_false": metadata.get("fixture") is False,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "symbol": getattr(target, "symbol", None),
        "uniprot_id": getattr(target, "uniprot_id", None),
        "chembl_target_id": metadata.get("chembl_target_id"),
        "target_type": metadata.get("chembl_target_type"),
        "organism": metadata.get("chembl_organism"),
    }


def _mechanism_pool_audit(
    provenance: Mapping[str, Any], seeds: Sequence[Any]
) -> dict[str, Any]:
    bounds = _mapping(provenance.get("candidate_discovery_bounds"))
    fixed_ids = list(bounds.get("prewarning_pool_ids") or [])
    preactivity = bounds.get("preactivity_pool")
    preactivity = preactivity if isinstance(preactivity, list) else []
    returned_ids = [_seed_id(seed) for seed in seeds]
    expected_order = [
        str(item.get("parent_molecule_chembl_id") or "")
        for item in sorted(
            (item for item in preactivity if isinstance(item, Mapping)),
            key=lambda item: (
                -int(item.get("max_phase"))
                if type(item.get("max_phase")) is int
                else 1,
                str(item.get("parent_molecule_chembl_id") or ""),
            ),
        )
    ]
    raw_count = bounds.get("raw_mechanism_response_count")
    total_count = bounds.get("mechanism_total_count")
    checks = {
        "schema": bounds.get("schema_version")
        == "bounded_mechanism_first_candidate_identity_pool_v1",
        "fixture_false": bounds.get("fixture") is False,
        "exact_target_type": bounds.get("target_type") == "SINGLE PROTEIN",
        "exact_target_organism": bounds.get("target_organism") == "Homo sapiens",
        "inhibitor_mechanism": bounds.get("mechanism_action_type") == "INHIBITOR",
        "mechanism_limit": bounds.get("raw_mechanism_row_limit")
        == MECHANISM_ROW_LIMIT,
        "mechanism_complete": bounds.get("mechanism_complete_within_bound") is True,
        "mechanism_count_bounded_complete": (
            type(raw_count) is int
            and type(total_count) is int
            and 0 <= raw_count == total_count <= MECHANISM_ROW_LIMIT
        ),
        "fixed_pool_bound": (
            0 < len(fixed_ids) <= FIXED_POOL_LIMIT
            and len(set(fixed_ids)) == len(fixed_ids)
            and all(isinstance(item, str) and CHEMBL_ID_RE.fullmatch(item) for item in fixed_ids)
        ),
        "fixed_pool_count": bounds.get("prewarning_pool_count") == len(fixed_ids),
        "preactivity_exact_copy": [
            str(item.get("parent_molecule_chembl_id") or "")
            for item in preactivity
            if isinstance(item, Mapping)
        ]
        == fixed_ids,
        "phase_then_stable_id_order": expected_order == fixed_ids,
        "selection_inputs_only": bounds.get("selection_inputs")
        == ["max_phase", "parent_molecule_chembl_id"],
        "activity_not_used_for_pool": bounds.get("activity_used_for_prewarning_selection")
        is False,
        "pubmed_not_used_for_pool": bounds.get(
            "primary_document_used_for_prewarning_selection"
        )
        is False,
        "pchembl_not_used_for_pool": bounds.get("pchembl_used_for_prewarning_selection")
        is False,
        "no_backfill": bounds.get("no_backfill_beyond_prewarning_pool") is True,
        "returned_exact_fixed_pool": returned_ids == fixed_ids,
        "returned_ids_unique": len(returned_ids) == len(set(returned_ids)),
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "fixed_pool_candidate_ids": fixed_ids,
        "fixed_pool_count": len(fixed_ids),
        "returned_candidate_ids": returned_ids,
        "returned_candidate_count": len(returned_ids),
        "bounds_sha256": _canonical_sha256(bounds) if bounds else None,
    }


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _direct_support_audit(
    search_provenance: Mapping[str, Any], fixed_ids: Sequence[str]
) -> dict[str, Any]:
    inventory = _mapping(search_provenance.get("direct_support_inventory"))
    records = inventory.get("records")
    records = records if isinstance(records, list) else []
    selected_ids = list(inventory.get("selected_candidate_ids") or [])
    record_checks: list[dict[str, Any]] = []
    for record in records[:FIXED_POOL_LIMIT]:
        record = _mapping(record)
        audits = record.get("row_audits")
        audits = audits if isinstance(audits, list) else []
        selection_status = record.get("selection_status")
        selection_status_valid = selection_status in {
            "selected",
            "not_selected",
        }
        record_sha256, record_sha256_bounded = _bounded_sha256(
            record.get("record_sha256")
        )
        observed_reason_codes, observed_reason_codes_valid = (
            _bounded_reason_codes(record.get("observed_row_reason_codes"))
        )
        record_reason_code, record_reason_code_valid = _bounded_reason_code(
            record.get("reason_code")
        )
        identity_binding = _mapping(record.get("candidate_identity_binding"))
        identity_reason_codes, identity_reason_codes_valid = (
            _bounded_reason_codes(identity_binding.get("reason_codes"))
            if identity_binding
            else ([], True)
        )
        identity_binding_status = identity_binding.get("binding_status")
        identity_binding_status_valid = bool(
            not identity_binding
            or identity_binding_status in {"exact", "incomplete"}
        )
        identity_binding_sha256, identity_binding_sha256_bounded = (
            _bounded_sha256(identity_binding.get("binding_sha256"))
            if identity_binding
            else (None, True)
        )
        row_projections: list[dict[str, Any]] = []
        row_reason_codes_valid = True
        row_statuses_valid = True
        projected_hashes_bounded = bool(
            record_sha256_bounded and identity_binding_sha256_bounded
        )
        for raw_audit in audits[:DIRECT_SUPPORT_RETAINED_ROW_LIMIT]:
            audit = _mapping(raw_audit)
            row_status = audit.get("status")
            row_status_valid = row_status in {"supported", "not_supported"}
            row_reason_codes, row_codes_valid = _bounded_reason_codes(
                audit.get("reason_codes")
            )
            row_reason_codes_valid = row_reason_codes_valid and row_codes_valid
            row_statuses_valid = row_statuses_valid and row_status_valid
            projected_hashes: dict[str, str | None] = {}
            for hash_field in (
                "row_audit_sha256",
                "raw_measurement_tuple_sha256",
                "typed_measurement_tuple_sha256",
                "assay_context_sha256",
                "chembl_activity_query_sha256",
                "chembl_activity_query_parameters_sha256",
                "chembl_activity_query_candidate_binding_sha256",
            ):
                bounded_hash, bounded_hash_valid = _bounded_sha256(
                    audit.get(hash_field)
                )
                projected_hashes[hash_field] = bounded_hash
                projected_hashes_bounded = bool(
                    projected_hashes_bounded and bounded_hash_valid
                )
            row_projections.append(
                {
                    "retained_rank": audit.get("retained_rank"),
                    "retained_index": audit.get("retained_index"),
                    "activity_id": audit.get("activity_id"),
                    "assay_chembl_id": audit.get("assay_chembl_id"),
                    "document_chembl_id": audit.get("document_chembl_id"),
                    "status": row_status if row_status_valid else "invalid",
                    "reason_codes": row_reason_codes,
                    **projected_hashes,
                    "fixture": audit.get("fixture"),
                }
            )
        ordered = True
        order_values: list[tuple[Decimal, str]] = []
        for audit in audits:
            raw_tuple = _mapping(_mapping(audit).get("raw_measurement_tuple"))
            pchembl = _decimal(raw_tuple.get("pchembl_value"))
            activity_id = str(_mapping(audit).get("activity_id") or "")
            if pchembl is None or not activity_id:
                ordered = False
                continue
            order_values.append((pchembl, activity_id))
        if order_values != sorted(order_values, key=lambda item: (-item[0], item[1])):
            ordered = False
        supported = [
            _mapping(item)
            for item in audits
            if _mapping(item).get("status") == "supported"
        ]
        selected = _mapping(record.get("selected_support"))
        selected_reason_code, selected_reason_code_valid = _bounded_reason_code(
            selected.get("reason_code")
        )
        selected_verification_status = selected.get("verification_status")
        selected_verification_status_valid = bool(
            not selected or selected_verification_status == "verified"
        )
        selected_support_sha256, selected_support_sha256_bounded = (
            _bounded_sha256(selected.get("support_attestation_sha256"))
            if selected
            else (None, True)
        )
        first_supported = supported[0] if supported else {}
        selected_consistent = bool(
            (not selected and not supported)
            or (
                selected
                and first_supported
                and selected.get("activity_id") == first_supported.get("activity_id")
                and selected.get("retained_index") == first_supported.get("retained_index")
            )
        )
        params = _mapping(selected.get("chembl_activity_query_parameters"))
        strict_selected = bool(
            not selected
            or (
                _hash_field_valid(selected, "support_attestation_sha256")
                and selected.get("verification_status") == "verified"
                and selected.get("selected_before_pubmed_search") is True
                and selected.get("selection_policy")
                == "first_strict_direct_support_in_retained_order_v1"
                and selected.get("target_organism") == "Homo sapiens"
                and selected.get("target_type") == "SINGLE PROTEIN"
                and params.get("parent_molecule_chembl_id")
                == record.get("molecule_chembl_id")
                and params.get("target_organism") == "Homo sapiens"
                and params.get("assay_type") == "B"
                and params.get("standard_type") == "IC50"
                and params.get("order_by") == "activity_id"
                and params.get("limit") == ACTIVITY_ROW_LIMIT
                and selected.get("pubmed_value_used_for_selection") is False
            )
        )
        bounded_projection = bool(
            len(audits) <= DIRECT_SUPPORT_RETAINED_ROW_LIMIT
            and record.get("retained_row_count") == len(audits)
            and observed_reason_codes_valid
            and record_reason_code_valid
            and identity_reason_codes_valid
            and row_reason_codes_valid
            and selected_reason_code_valid
            and selection_status_valid
            and identity_binding_status_valid
            and row_statuses_valid
            and selected_verification_status_valid
            and projected_hashes_bounded
            and selected_support_sha256_bounded
        )
        selected_projection = (
            {
                "activity_id": selected.get("activity_id"),
                "assay_chembl_id": selected.get("assay_chembl_id"),
                "document_chembl_id": selected.get("document_chembl_id"),
                "retained_rank": selected.get("retained_rank"),
                "retained_index": selected.get("retained_index"),
                "verification_status": (
                    selected_verification_status
                    if selected_verification_status_valid
                    else "invalid"
                ),
                "reason_code": selected_reason_code,
                "support_attestation_sha256": selected_support_sha256,
                "fixture": selected.get("fixture"),
            }
            if selected
            else None
        )
        record_checks.append(
            {
                "candidate_id": record.get("molecule_chembl_id"),
                "verified": bool(
                    _hash_field_valid(record, "record_sha256")
                    and all(
                        _hash_field_valid(_mapping(item), "row_audit_sha256")
                        for item in audits
                    )
                    and ordered
                    and selected_consistent
                    and strict_selected
                    and bounded_projection
                ),
                "retained_row_count": len(audits),
                "selected_activity_id": selected.get("activity_id"),
                "selected": bool(selected),
                "selection_status": (
                    selection_status if selection_status_valid else "invalid"
                ),
                "reason_code": record_reason_code,
                "observed_row_reason_codes": observed_reason_codes,
                "record_sha256": record_sha256,
                "bounded_projection_verified": bounded_projection,
                "candidate_identity_binding": {
                    "binding_status": (
                        identity_binding_status
                        if identity_binding_status_valid
                        else "invalid"
                    ),
                    "reason_codes": identity_reason_codes,
                    "binding_sha256": identity_binding_sha256,
                }
                if identity_binding
                else None,
                "row_audits": row_projections,
                "row_audit_projection_complete": bool(
                    len(row_projections) == len(audits)
                ),
                "selected_support": selected_projection,
            }
        )
    expected_selected_order = [item for item in fixed_ids if item in set(selected_ids)]
    checks = {
        "inventory_hash": _hash_field_valid(inventory, "inventory_sha256"),
        "schema": inventory.get("schema_version")
        == "paper_native_v5_direct_support_inventory_v1",
        "policy": inventory.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN,
        "fixture_false": inventory.get("fixture") is False,
        "fixed_pool_exact": inventory.get("fixed_pool_candidate_ids") == list(fixed_ids),
        "fixed_pool_hash": inventory.get("fixed_pool_candidate_ids_sha256")
        == _canonical_sha256(list(fixed_ids)),
        "record_coverage": len(records) == len(fixed_ids),
        "bounded_candidate_projection": bool(
            len(records) <= FIXED_POOL_LIMIT
            and len(record_checks) == len(records)
        ),
        "records_verified": bool(record_checks) and all(
            item["verified"] for item in record_checks
        ),
        "selected_order": selected_ids == expected_selected_order,
        "selected_count": inventory.get("selected_candidate_count")
        == len(selected_ids),
        "at_least_three_supports": len(selected_ids) >= 3,
        "selection_before_pubmed": inventory.get(
            "selection_completed_before_pubmed_search"
        )
        is True,
        "pubmed_not_used": inventory.get("pubmed_or_result_used_for_selection")
        is False,
        "no_reordering": inventory.get("no_reordering") is True,
        "no_backfill": inventory.get("no_backfill") is True,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_count": len(selected_ids),
        "candidate_records": record_checks,
        "candidate_record_source_count": len(records),
        "candidate_record_projection_complete": bool(
            len(record_checks) == len(records)
        ),
        "candidate_record_limit": FIXED_POOL_LIMIT,
        "retained_rows_per_candidate_limit": (
            DIRECT_SUPPORT_RETAINED_ROW_LIMIT
        ),
        "reason_code_list_limit": REASON_CODE_LIST_LIMIT,
        "raw_measurement_rows_stored": False,
        "inventory_sha256": inventory.get("inventory_sha256"),
    }


def _alias_audit(
    search_provenance: Mapping[str, Any],
    fixed_ids: Sequence[str],
    supported_ids: Sequence[str],
) -> dict[str, Any]:
    inventory = _mapping(search_provenance.get("alias_inventory"))
    records = inventory.get("records")
    records = records if isinstance(records, list) else []
    row_checks: list[dict[str, Any]] = []
    for raw in records:
        row = _mapping(raw)
        aliases = row.get("accepted_aliases")
        aliases = aliases if isinstance(aliases, list) else []
        source_inventory = _mapping(row.get("molecule_synonyms_source_inventory"))
        row_checks.append(
            {
                "candidate_id": row.get("molecule_chembl_id"),
                "verified": bool(
                    _hash_field_valid(row, "record_sha256")
                    and _hash_field_valid(source_inventory, "inventory_sha256")
                    and row.get("source_attestation_complete") is True
                    and row.get("query_eligible") is True
                    and source_inventory.get("complete") is True
                    and isinstance(aliases, list)
                    and 1 <= len(aliases) <= ALIAS_LIMIT
                    and row.get("accepted_alias_count") == len(aliases)
                    and row.get("preferred_search_alias") == aliases[0]
                ),
                "preferred_search_alias": row.get("preferred_search_alias"),
                "accepted_alias_count": len(aliases),
            }
        )
    target_aliases = inventory.get("target_aliases")
    target_aliases = target_aliases if isinstance(target_aliases, list) else []
    queried_ids = list(inventory.get("queried_candidate_ids") or [])
    checks = {
        "inventory_hash": _hash_field_valid(inventory, "inventory_sha256"),
        "schema": inventory.get("schema_version")
        == "paper_native_v5_alias_inventory_v1",
        "policy": inventory.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN,
        "fixture_false": inventory.get("fixture") is False,
        "complete": inventory.get("complete") is True,
        "endpoint": inventory.get("endpoint") == "IC50",
        "fixed_pool_exact": inventory.get("fixed_pool_candidate_ids") == list(fixed_ids),
        "supported_exact": inventory.get("direct_support_candidate_ids")
        == list(supported_ids),
        "queried_exact": queried_ids == list(supported_ids),
        "target_alias_bound": (
            TARGET_SYMBOL in target_aliases and 1 <= len(target_aliases) <= 8
        ),
        "candidate_alias_bound": inventory.get("per_candidate_alias_limit")
        == ALIAS_LIMIT,
        "record_coverage": len(records) == len(supported_ids),
        "records_verified": bool(row_checks) and all(item["verified"] for item in row_checks),
        "no_owner_conflicts": inventory.get("owner_conflicts") == [],
        "no_result_adaptation": inventory.get(
            "pubmed_or_result_used_for_alias_selection"
        )
        is False,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "queried_candidate_ids": queried_ids,
        "target_aliases": target_aliases,
        "records": row_checks,
        "inventory_sha256": inventory.get("inventory_sha256"),
    }


def _expected_query(alias_inventory: Mapping[str, Any]) -> str | None:
    target_aliases = alias_inventory.get("target_aliases")
    records = alias_inventory.get("records")
    if not isinstance(target_aliases, list) or not isinstance(records, list):
        return None
    target_values = sorted(
        (str(item) for item in target_aliases),
        key=lambda item: (unicodedata.normalize("NFKC", item).casefold(), item),
    )
    preferred_by_id = {
        str(_mapping(item).get("molecule_chembl_id") or ""): str(
            _mapping(item).get("preferred_search_alias") or ""
        )
        for item in records
    }
    candidate_values = [
        preferred_by_id[candidate_id]
        for candidate_id in sorted(
            preferred_by_id,
            key=lambda item: (unicodedata.normalize("NFKC", item).casefold(), item),
        )
        if preferred_by_id[candidate_id]
    ]
    if not target_values or not candidate_values:
        return None
    term = lambda value: f'"{value}"[Title/Abstract]'
    return (
        f"({' OR '.join(term(item) for item in target_values)}) AND "
        f"{term(str(alias_inventory.get('endpoint') or ''))} AND "
        f"({' OR '.join(term(item) for item in candidate_values)}) "
        "NOT Review[Publication Type]"
    )


def _call_hash_valid(call: Mapping[str, Any]) -> bool:
    body = {
        "endpoint": call.get("endpoint"),
        "error_type": call.get("error_type"),
        "method": call.get("method"),
        "parameter_keys": call.get("parameter_keys"),
        "parameter_sha256": call.get("parameter_sha256"),
        "response_sha256": call.get("response_sha256"),
        "schema": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "stage": call.get("stage"),
        "status": call.get("status"),
        "status_code": call.get("status_code"),
    }
    return call.get("call_sha256") == _canonical_sha256(body)


def _record_hash_valid(record: Mapping[str, Any]) -> bool:
    body = {
        "bounded_sentence_count": record.get("bounded_sentence_count"),
        "document_id": record.get("document_id"),
        "eligible_section_count": record.get("eligible_section_count"),
        "fetch_publication_types": record.get("fetch_publication_types"),
        "publication_types": record.get("publication_types"),
        "rejection_codes": record.get("rejection_codes"),
        "schema": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "sentence_count": record.get("sentence_count"),
        "status": record.get("status"),
        "verified_group_count": record.get("verified_group_count"),
    }
    return record.get("record_sha256") == _canonical_sha256(body)


def _parser_replay(
    group: Mapping[str, Any], alias_inventory: Mapping[str, Any]
) -> bool:
    sentence = group.get("sentence")
    records = alias_inventory.get("records")
    if not isinstance(sentence, str) or not isinstance(records, list):
        return False
    candidate_aliases = {
        str(_mapping(item).get("molecule_chembl_id") or ""): tuple(
            _mapping(item).get("accepted_aliases") or []
        )
        for item in records
    }
    publication_types = tuple(
        dict.fromkeys(
            [
                *list(group.get("summary_publication_types") or []),
                *list(group.get("fetch_publication_types") or []),
            ]
        )
    )
    pairs = parse_paper_native_primary_abstract_v5(
        sentence,
        candidate_aliases=candidate_aliases,
        target_aliases=tuple(alias_inventory.get("target_aliases") or []),
        endpoint=str(alias_inventory.get("endpoint") or ""),
        section=str(group.get("section") or ""),
        publication_types=publication_types,
    )
    return bool(
        len(pairs) >= 3
        and _jsonable([asdict(pair) for pair in pairs]) == group.get("pairs")
    )


def _group_hash_valid(group: Mapping[str, Any], query_sha256: str) -> bool:
    pairs = group.get("pairs")
    pairs = pairs if isinstance(pairs, list) else []
    body = {
        "candidate_ids": group.get("candidate_ids"),
        "document_id": group.get("document_id"),
        "document_rank": group.get("document_rank"),
        "grammar": PAPER_NATIVE_V5_GRAMMAR,
        "fetch_publication_types": group.get("fetch_publication_types"),
        "pair_evidence_sha256": [
            _mapping(item).get("evidence_sha256") for item in pairs
        ],
        "query_sha256": query_sha256,
        "schema": PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "section": group.get("section"),
        "sentence_index": group.get("sentence_index"),
        "sentence_sha256": group.get("sentence_sha256"),
        "summary_publication_types": group.get("summary_publication_types"),
    }
    sentence = group.get("sentence")
    return bool(
        isinstance(sentence, str)
        and len(sentence) <= 360
        and group.get("sentence_sha256") == sha256(sentence.encode("utf-8")).hexdigest()
        and group.get("group_sha256") == _canonical_sha256(body)
    )


def _pubmed_audit(
    search_provenance: Mapping[str, Any], pubmed_calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    alias_inventory = _mapping(search_provenance.get("alias_inventory"))
    outcome = _mapping(search_provenance.get("search_outcome"))
    promotion = _mapping(search_provenance.get("promotion"))
    calls = outcome.get("calls")
    calls = calls if isinstance(calls, list) else []
    records = outcome.get("records")
    records = records if isinstance(records, list) else []
    groups = outcome.get("groups")
    groups = groups if isinstance(groups, list) else []
    expected_query = _expected_query(alias_inventory)
    ordered_groups = sorted(
        (_mapping(item) for item in groups),
        key=lambda item: (
            int(item.get("document_rank") or 0),
            int(item.get("sentence_index") or 0),
            str(item.get("group_sha256") or ""),
        ),
    )
    selected_group = ordered_groups[0] if ordered_groups else {}
    stage_order = [str(_mapping(item).get("stage") or "") for item in calls]
    expected_stage_order = ["esearch", "esummary", "efetch"]
    expected_endpoints = [PUBMED_SEARCH_URL, PUBMED_SUMMARY_URL, PUBMED_FETCH_URL]
    if not calls and not pubmed_calls:
        execution_state = "not_executed"
    elif len(calls) == len(pubmed_calls) == PUBMED_LOGICAL_CALL_LIMIT:
        execution_state = "completed_required_sequence"
    else:
        execution_state = "partial_or_mismatched"
    outcome_reason_code, outcome_reason_code_valid = _bounded_reason_code(
        outcome.get("reason_code")
    )
    outcome_status = outcome.get("status")
    outcome_status_valid = outcome_status in {
        "verified",
        "not_verified",
        "failed",
    }
    wire_matches = bool(
        len(calls) == len(pubmed_calls) == PUBMED_LOGICAL_CALL_LIMIT
        and all(
            _audited_call_fingerprint(_mapping(wire)) is not None
            and _mapping(call).get("endpoint") == _mapping(wire).get("endpoint")
            and _mapping(call).get("method") == _mapping(wire).get("method")
            and _mapping(call).get("parameter_keys")
            == _mapping(wire).get("parameter_keys")
            and _mapping(call).get("parameter_sha256")
            == _mapping(wire).get("parameter_sha256")
            and _mapping(call).get("status") == _mapping(wire).get("status")
            and _mapping(call).get("status_code")
            == _mapping(wire).get("status_code")
            and _mapping(call).get("response_sha256")
            == _mapping(wire).get("response_sha256")
            and _mapping(wire).get("fixture") is False
            for call, wire in zip(calls, pubmed_calls, strict=True)
        )
    )
    returned_ids = outcome.get("returned_document_ids")
    returned_ids = returned_ids if isinstance(returned_ids, list) else []
    group_ids = list(selected_group.get("candidate_ids") or [])
    checks = {
        "gateway_provenance_hash": _hash_field_valid(
            search_provenance, "provenance_sha256"
        ),
        "gateway_schema": search_provenance.get("schema_version")
        == "primary_abstract_paper_native_gateway_v5",
        "gateway_policy": search_provenance.get("policy")
        == PAPER_NATIVE_V5_POLICY_TOKEN,
        "fixture_false": search_provenance.get("fixture") is False,
        "dedicated_single_attempt": (
            _mapping(search_provenance.get("transport")).get(
                "dedicated_from_chembl_client"
            )
            is True
            and _mapping(search_provenance.get("transport")).get("max_attempts")
            == PUBMED_MAX_ATTEMPTS
            and _mapping(search_provenance.get("transport")).get(
                "logical_call_limit"
            )
            == PUBMED_LOGICAL_CALL_LIMIT
        ),
        "legacy_document_lookup_zero": search_provenance.get(
            "legacy_primary_document_lookup_executed"
        )
        is False
        and search_provenance.get("legacy_primary_document_lookup_candidate_ids")
        == []
        and search_provenance.get("legacy_primary_document_lookup_document_ids")
        == [],
        "legacy_pubmed_type_lookup_zero": search_provenance.get(
            "legacy_pubmed_type_lookup_executed"
        )
        is False
        and search_provenance.get("legacy_pubmed_type_lookup_candidate_ids") == []
        and search_provenance.get("legacy_pubmed_type_lookup_pmid_ids") == [],
        "outcome_hash": _hash_field_valid(outcome, "outcome_sha256"),
        "outcome_schema": outcome.get("schema_version")
        == PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
        "bounded_stage_status_and_reason": bool(
            outcome_status_valid and outcome_reason_code_valid
        ),
        "outcome_verified": outcome.get("status") == "verified"
        and outcome.get("reason_code") is None,
        "sort_and_bounds": outcome.get("search_sort") == "pub date"
        and outcome.get("max_documents") == PUBMED_DOCUMENT_LIMIT
        and outcome.get("requested_document_count") == PUBMED_DOCUMENT_LIMIT
        and 0 <= len(returned_ids) <= PUBMED_DOCUMENT_LIMIT,
        "query_exact": bool(
            expected_query
            and outcome.get("query") == expected_query
            and outcome.get("query_sha256")
            == sha256(expected_query.encode("utf-8")).hexdigest()
        ),
        "query_translation_hash": isinstance(outcome.get("query_translation"), str)
        and outcome.get("query_translation_sha256")
        == sha256(str(outcome.get("query_translation")).encode("utf-8")).hexdigest(),
        "returned_order_hash": outcome.get("returned_document_ids_sha256")
        == _canonical_sha256({"ordered_values": returned_ids}),
        "unsearched_count": type(outcome.get("unsearched_result_count")) is int
        and outcome.get("unsearched_result_count")
        == max(int(outcome.get("total_result_count") or 0) - len(returned_ids), 0),
        "logical_call_order": len(calls) == PUBMED_LOGICAL_CALL_LIMIT
        and stage_order == expected_stage_order
        and [str(_mapping(item).get("endpoint") or "") for item in calls]
        == expected_endpoints,
        "logical_call_hashes": bool(calls)
        and all(_call_hash_valid(_mapping(item)) for item in calls),
        "logical_calls_completed": bool(calls)
        and all(
            _mapping(item).get("method") == "GET"
            and _mapping(item).get("status") == "completed"
            and _mapping(item).get("status_code") == 200
            and isinstance(_mapping(item).get("response_sha256"), str)
            and HASH_RE.fullmatch(str(_mapping(item).get("response_sha256")))
            for item in calls
        ),
        "wire_ledger_exact": wire_matches,
        "wire_attempts_one": bool(pubmed_calls)
        and all(
            _mapping(item).get("attempt_count_observed") == 1
            for item in pubmed_calls
        ),
        "record_hashes": bool(records)
        and all(_record_hash_valid(_mapping(item)) for item in records),
        "selected_group_present": bool(selected_group)
        and len(group_ids) >= 3
        and len(group_ids) == len(set(group_ids)),
        "selected_group_order": bool(ordered_groups)
        and ordered_groups == [
            _mapping(item) for item in groups
        ]
        and outcome.get("selected_group_sha256")
        == selected_group.get("group_sha256"),
        "selected_group_hash": bool(selected_group)
        and _group_hash_valid(selected_group, str(outcome.get("query_sha256") or "")),
        "parser_replay": bool(selected_group)
        and _parser_replay(selected_group, alias_inventory),
        "promotion_hash": _hash_field_valid(promotion, "promotion_sha256"),
        "promotion_indivisible": promotion.get("status") == "verified"
        and promotion.get("selected_group_sha256")
        == selected_group.get("group_sha256")
        and set(promotion.get("verified_member_candidate_ids") or [])
        == set(group_ids)
        and promotion.get("verified_member_count") == len(group_ids)
        and promotion.get("no_partial_group_promotion") is True,
        "cross_source_separation": search_provenance.get(
            "pubmed_value_used_for_activity_selection"
        )
        is False
        and search_provenance.get("cross_source_numeric_equality_required")
        is False
        and search_provenance.get("no_reordering") is True
        and search_provenance.get("no_backfill") is True,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "stage_status": outcome_status if outcome_status_valid else "invalid",
        "reason_code": outcome_reason_code,
        "execution_state": execution_state,
        "call_checks_applicable": execution_state != "not_executed",
        "query": outcome.get("query"),
        "query_sha256": outcome.get("query_sha256"),
        "query_translation": outcome.get("query_translation"),
        "query_translation_sha256": outcome.get("query_translation_sha256"),
        "total_result_count": outcome.get("total_result_count"),
        "returned_document_ids": returned_ids,
        "returned_document_ids_sha256": outcome.get(
            "returned_document_ids_sha256"
        ),
        "unsearched_result_count": outcome.get("unsearched_result_count"),
        "selected_group": {
            "document_id": selected_group.get("document_id"),
            "document_rank": selected_group.get("document_rank"),
            "section": selected_group.get("section"),
            "sentence_index": selected_group.get("sentence_index"),
            "sentence": selected_group.get("sentence"),
            "sentence_sha256": selected_group.get("sentence_sha256"),
            "group_sha256": selected_group.get("group_sha256"),
            "summary_publication_types": selected_group.get(
                "summary_publication_types"
            ),
            "fetch_publication_types": selected_group.get(
                "fetch_publication_types"
            ),
            "candidate_ids": group_ids,
            "pairs": selected_group.get("pairs"),
        }
        if selected_group
        else None,
        "logical_calls": deepcopy(calls),
        "outcome_sha256": outcome.get("outcome_sha256"),
        "promotion_sha256": promotion.get("promotion_sha256"),
    }


def _seed_policy_metadata(seed: Any) -> dict[str, Any]:
    evidence = list(getattr(seed, "evidence", ()) or ())
    return {
        **deepcopy(dict(_mapping(getattr(seed, "metadata", {})))),
        "source_evidence": [
            {
                "source": getattr(item, "source", None),
                "drug_id": getattr(item, "external_id", None),
                "evidence_id": getattr(item, "evidence_id", None),
                "metadata": deepcopy(
                    dict(_mapping(getattr(item, "metadata", {})))
                ),
            }
            for item in evidence
        ],
        "identifiers": [
            getattr(item, "external_id", None) for item in evidence
        ],
    }


def _seed_primary_screen(seed: Any, candidate_id: str) -> dict[str, Any]:
    envelope = _seed_envelope(seed)
    support = _mapping(envelope.get("pre_pubmed_chembl_support"))
    activities = list(getattr(seed, "activities", ()) or ())
    evidence = list(getattr(seed, "evidence", ()) or ())
    activity = activities[0] if len(activities) == 1 else None
    typed = _mapping(support.get("typed_measurement_tuple"))
    exact_support = bool(
        envelope.get("verification_status") == "verified"
        and support.get("molecule_chembl_id") == candidate_id
        and activity is not None
        and getattr(activity, "endpoint", None) == typed.get("endpoint")
        and getattr(activity, "relation", None) == typed.get("relation")
        and _decimal(getattr(activity, "value", None))
        == _decimal(typed.get("value"))
        and getattr(activity, "unit", None) == typed.get("unit")
        and getattr(activity, "assay_type", None) == typed.get("assay_type")
        and getattr(activity, "source_evidence_id", None)
        == typed.get("source_evidence_id")
        and getattr(activity, "source_evidence_id", None)
        in {getattr(item, "evidence_id", None) for item in evidence}
    )
    reasons: list[str] = []
    if envelope.get("verification_status") != "verified":
        reasons.append("CANDIDATE_PRIMARY_SOURCE_UNVERIFIED")
    elif not exact_support:
        reasons.append("CANDIDATE_DIRECT_TARGET_SUPPORT_UNVERIFIED")
    return {
        "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
        "candidate_id": candidate_id,
        "verification_status": envelope.get("verification_status"),
        "group_sha256": _mapping(envelope.get("group")).get("group_sha256"),
        "independent_chembl_direct_support_verified": not reasons,
        "pubmed_values_used_for_candidate_activity_or_ranking": False,
        "excluded": bool(reasons),
        "reason_codes": reasons,
        "fixture": False,
    }


def _build_g1_candidate_packet(
    *,
    context: ValidationContext,
    seeds: Sequence[Any],
    fixed_ids: Sequence[str],
    group_ids: Sequence[str],
    selected_group_sha256: str,
    target: TargetCandidate,
    discovery_provenance: Mapping[str, Any],
) -> tuple[CandidatePacket, list[dict[str, Any]]]:
    """Project the exact MA2 finite prefix without running models or reviewers."""

    seeds_by_id = {_seed_id(seed): seed for seed in seeds}
    seed_ids = [_seed_id(seed) for seed in seeds]
    if (
        not fixed_ids
        or len(set(fixed_ids)) != len(fixed_ids)
        or seed_ids != list(fixed_ids)
        or len(seeds_by_id) != len(seed_ids)
    ):
        raise PreflightContractError("G1_PACKET_FIXED_POOL_SEEDS_INCOMPLETE")
    if (
        len(group_ids) < 3
        or len(set(group_ids)) != len(group_ids)
        or not set(group_ids).issubset(set(fixed_ids))
        or not set(group_ids).issubset(seeds_by_id)
    ):
        raise PreflightContractError("G1_PACKET_SOURCE_GROUP_INCOMPLETE")
    last_group_index = max(fixed_ids.index(item) for item in group_ids)
    evaluated_ids = list(fixed_ids[: last_group_index + 1])
    if not set(evaluated_ids).issubset(seeds_by_id):
        raise PreflightContractError("G1_PACKET_FIXED_PREFIX_INCOMPLETE")

    reserved_ids = {
        item.identifier.casefold()
        for control in context.reserved_controls
        for item in control.registry_ids
        if item.namespace.casefold() == "chembl"
    }
    eligible_ids = [
        item
        for item in evaluated_ids
        if item in set(group_ids) and item.casefold() not in reserved_ids
    ][:PROMOTED_NONCONTROL_LIMIT]
    eligible_count = len(eligible_ids)
    projected_ids = [item for item in evaluated_ids if item in set(group_ids)]
    quota_met = eligible_count >= PROMOTED_NONCONTROL_LIMIT
    projection_complete = set(projected_ids) == set(group_ids)
    stop_satisfied = quota_met and projection_complete
    stop_rule = "eligible_quota_and_declared_group_projection_complete_v1"
    projection: dict[str, Any] = {
        "schema_version": "paper_native_v5_group_projection_v1",
        "policy": PAPER_NATIVE_V5_POLICY_TOKEN,
        "declared_group_candidate_ids": list(group_ids),
        "declared_group_candidate_ids_sha256": _canonical_sha256(
            list(group_ids)
        ),
        "projected_group_candidate_ids": projected_ids,
        "projected_group_candidate_ids_sha256": _canonical_sha256(
            projected_ids
        ),
        "projection_complete": projection_complete,
        "eligible_quota_reached": quota_met,
        "requested_eligible_count": PROMOTED_NONCONTROL_LIMIT,
        "eligible_returned_count": eligible_count,
        "evaluated_fixed_pool_prefix_count": len(evaluated_ids),
        "stop_rule": stop_rule,
        "stop_rule_sha256": sha256(stop_rule.encode("utf-8")).hexdigest(),
        "stop_condition_satisfied": stop_satisfied,
        "no_second_discovery": True,
        "no_backfill_beyond_fixed_pool": True,
        "fixture": False,
    }
    projection["projection_sha256"] = _canonical_sha256(projection)

    cards: list[CandidateCard] = []
    rows: list[dict[str, Any]] = []
    evaluated_identities: list[dict[str, Any]] = []
    seed_preservation_rows: list[dict[str, Any]] = []
    for pool_index, candidate_id in enumerate(evaluated_ids):
        seed = seeds_by_id[candidate_id]
        metadata = _seed_policy_metadata(seed)
        reserved = screen_reserved_control(
            candidate_name=str(getattr(seed, "name", candidate_id)),
            metadata=metadata,
            reserved_controls=context.reserved_controls,
        )
        forbidden = screen_forbidden_drug(
            candidate_name=str(getattr(seed, "name", candidate_id)),
            metadata=metadata,
            forbidden_names=context.brief.constraints.forbidden_drugs,
            forbidden_ids=context.brief.constraints.forbidden_drug_ids,
            forbidden_classes=context.brief.constraints.forbidden_drug_classes,
        )
        primary_screen = _seed_primary_screen(seed, candidate_id)
        eligible = candidate_id in eligible_ids
        if reserved.excluded:
            reasons = list(reserved.reason_codes)
        elif candidate_id not in set(group_ids):
            reasons = list(primary_screen["reason_codes"] or [
                "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
            ])
        elif not eligible:
            reasons = ["NONCONTROL_QUOTA_CONTEXT_ONLY"]
        else:
            reasons = []
        rank = eligible_ids.index(candidate_id) + 1 if eligible else None
        status = "priority" if eligible else "excluded"
        final_metadata = {
            **metadata,
            "reserved_control_screen": reserved.as_metadata(),
            "control_only": reserved.matched,
            "forbidden_drug_screen": forbidden.as_metadata(),
            "paper_native_v5_primary_screen": primary_screen,
            "paper_native_v5_group_projection": deepcopy(projection),
            "policy_eligible": eligible,
            "policy_status_attestation": {
                "rule": (
                    "policy_eligible_iff_candidate_is_not_excluded_and_ranked"
                ),
                "candidate_status": status,
                "ranked": rank is not None,
                "policy_eligible": eligible,
            },
        }
        safety_flags = (
            ["reserved_assay_control"] if reserved.matched else []
        )
        card = CandidateCard(
            candidate_id=(
                f"cand_preflight_{pool_index + 1:02d}_{candidate_id.casefold()}"
            ),
            route=CandidateRoute.REPURPOSING,
            name=str(getattr(seed, "name", candidate_id)),
            target=target,
            mechanism=str(getattr(seed, "mechanism", "")),
            smiles=str(getattr(seed, "smiles", "")),
            inchikey=(
                str(_mapping(getattr(seed, "metadata", {})).get(
                    "standard_inchi_key"
                ) or "")
                or None
            ),
            activities=deepcopy(list(getattr(seed, "activities", ()) or ())),
            safety_flags=safety_flags,
            evidence=deepcopy(list(getattr(seed, "evidence", ()) or ())),
            metadata=final_metadata,
            score=0.0,
            rank=rank,
            target_rank=rank,
            status=status,
            exclusion_reasons=reasons,
        )
        cards.append(card)
        source_metadata = deepcopy(
            dict(_mapping(getattr(seed, "metadata", {})))
        )
        source_activities = list(getattr(seed, "activities", ()) or ())
        source_evidence = list(getattr(seed, "evidence", ()) or ())
        metadata_preserved = all(
            card.metadata.get(key) == value
            for key, value in source_metadata.items()
        )
        activities_preserved = card.activities == source_activities
        evidence_preserved = card.evidence == source_evidence
        seed_preservation_rows.append(
            {
                "molecule_chembl_id": candidate_id,
                "metadata_preserved": metadata_preserved,
                "activities_preserved": activities_preserved,
                "evidence_preserved": evidence_preserved,
                "source_metadata_sha256": _canonical_sha256(source_metadata),
                "projected_source_metadata_sha256": _canonical_sha256(
                    {
                        key: card.metadata.get(key)
                        for key in source_metadata
                    }
                ),
                "fixture": False,
            }
        )
        identity = candidate_identity(
            candidate_name=card.name,
            metadata=metadata,
        )
        evaluated_identities.append(
            {
                **identity,
                "pool_index": pool_index,
                "candidate_id": card.candidate_id,
                "outcome": "eligible" if eligible else "excluded",
                "reason_codes": reasons,
            }
        )
        envelope = _seed_envelope(seed)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": card.name,
                "fixed_pool_index": pool_index,
                "source_group_member": candidate_id in set(group_ids),
                "projected": True,
                "envelope_verified": (
                    envelope.get("verification_status") == "verified"
                    if candidate_id in set(group_ids)
                    else envelope.get("verification_status") != "verified"
                ),
                "control_only": reserved.matched,
                "policy_eligible": eligible,
                "rank": rank,
                "status": status,
                "exclusion_reasons": reasons,
                "selected_group_sha256": envelope.get(
                    "selected_group_sha256"
                ),
                "fixture": False,
            }
        )

    unevaluated_identities = [
        {
            **candidate_identity(
                candidate_name=str(
                    getattr(seeds_by_id[candidate_id], "name", candidate_id)
                ),
                metadata=_seed_policy_metadata(seeds_by_id[candidate_id]),
            ),
            "pool_index": pool_index,
            "reason_codes": [
                "ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"
            ],
        }
        for pool_index, candidate_id in enumerate(
            fixed_ids[len(evaluated_ids) :], start=len(evaluated_ids)
        )
        if candidate_id in seeds_by_id
    ]
    pool_exhausted = len(evaluated_ids) == len(fixed_ids)
    candidate_backfill = {
        "requested_eligible_count": PROMOTED_NONCONTROL_LIMIT,
        "discovery_pool_count": len(fixed_ids),
        "discovery_pool_limit": FIXED_POOL_LIMIT,
        "evaluated_count": len(evaluated_ids),
        "eligible_count": eligible_count,
        "excluded_count": len(evaluated_ids) - eligible_count,
        "quota_met": quota_met,
        "pool_exhausted": pool_exhausted,
        "evaluated_identities": evaluated_identities,
        "unevaluated_identities": unevaluated_identities,
        "stop_reason": (
            "ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"
            if stop_satisfied
            else "FINITE_DISCOVERY_POOL_EXHAUSTED"
        ),
        "unbounded_search_used": False,
        "discovery_provenance": {
            **deepcopy(dict(discovery_provenance)),
            "outcome": "candidates_returned",
            "returned_candidates": len(seeds),
        },
        "paper_native_v5_group_projection": deepcopy(projection),
    }
    candidate_yield = {
        "requested_limit": PROMOTED_NONCONTROL_LIMIT,
        "returned_count": len(seeds),
        "partial": not quota_met,
        "fixture_fill_used": False,
        "backend_kind": ScientificBackendKind.LIVE.value,
        "requested_eligible_count": PROMOTED_NONCONTROL_LIMIT,
        "discovery_pool_count": len(fixed_ids),
        "evaluated_count": len(evaluated_ids),
        "eligible_count": eligible_count,
        "quota_met": quota_met,
        "pool_exhausted": pool_exhausted,
        "paper_native_v5_group_projection": deepcopy(projection),
    }
    seed_preservation = {
        "schema_version": "paper_native_v5_seed_preservation_v1",
        "evaluated_candidate_count": len(evaluated_ids),
        "rows": seed_preservation_rows,
        "verified": bool(seed_preservation_rows)
        and all(
            item["metadata_preserved"]
            and item["activities_preserved"]
            and item["evidence_preserved"]
            for item in seed_preservation_rows
        ),
        "fixture": False,
    }
    seed_preservation["attestation_sha256"] = _canonical_sha256(
        seed_preservation
    )
    packet = CandidatePacket(
        packet_id=f"cp_preflight_{selected_group_sha256[:16]}",
        plan_id="plan_paper_native_v5_free_preflight",
        candidates=cards,
        generation_iterations=1,
        metadata={
            "candidate_yield": candidate_yield,
            "candidate_backfill": candidate_backfill,
            "preflight_seed_preservation": seed_preservation,
            "preflight_packet_scope": (
                "reviewer_free_deterministic_paper_native_v5_audit_only"
            ),
            "fixture": False,
        },
        backend_kind=ScientificBackendKind.LIVE,
        execution_status=ScientificExecutionStatus.COMPLETED,
    )
    return packet, rows


def _ma3_v5_audit(
    context: ValidationContext,
    packet: CandidatePacket,
) -> dict[str, Any]:
    assurance = MA3AssuranceAgent(reviewer_provider=None)
    findings = assurance._audit_candidate_primary_sources_for_policy(
        context.brief,
        packet,
    )
    payload = [
        {
            "code": item.code,
            "severity": item.severity.value,
            "artifact_id": item.artifact_id,
            "failed_node": item.failed_node,
            "deterministic": item.deterministic,
        }
        for item in findings
    ]
    packet_dump = packet.model_dump(mode="json")
    preservation = _mapping(
        _mapping(getattr(packet, "metadata", {})).get(
            "preflight_seed_preservation"
        )
    )
    return {
        "verified": findings == [],
        "deterministic_audit_invoked": True,
        "audit_owner": "drugpilot.agents.ma3.MA3AssuranceAgent",
        "audit_method": "_audit_candidate_primary_sources_for_policy",
        "private_hash_pinned_v5_only_boundary": True,
        "producer_parser_or_search_called_by_audit": False,
        "external_reviewer_invoked": False,
        "model_or_llm_calls": 0,
        "finding_count": len(payload),
        "findings": payload,
        "candidate_packet_sha256": _canonical_sha256(packet_dump),
        "candidate_packet_candidate_count": len(packet.candidates),
        "candidate_packet_full_seed_fields_preserved": (
            preservation.get("verified") is True
            and _hash_field_valid(preservation, "attestation_sha256")
        ),
    }


def _projection_audit(
    context: ValidationContext,
    seeds: Sequence[Any],
    fixed_ids: Sequence[str],
    pubmed: Mapping[str, Any],
    target: Any,
    discovery_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], CandidatePacket | None]:
    selected_group = _mapping(pubmed.get("selected_group"))
    group_ids = list(selected_group.get("candidate_ids") or [])
    seeds_by_id = {_seed_id(seed): seed for seed in seeds}
    ranked_noncontrols = [
        item for item in fixed_ids if item in set(group_ids) and item != RESERVED_CHEMBL_ID
    ][:PROMOTED_NONCONTROL_LIMIT]
    rows: list[dict[str, Any]] = []
    envelope_checks: list[bool] = []
    for candidate_id in group_ids:
        seed = seeds_by_id.get(candidate_id)
        envelope = _seed_envelope(seed) if seed is not None else {}
        pair = _mapping(envelope.get("selected_pair"))
        support = _mapping(envelope.get("pre_pubmed_chembl_support"))
        envelope_ok = bool(
            seed is not None
            and _hash_field_valid(envelope, "attestation_sha256")
            and envelope.get("source_contract")
            == "primary_abstract_paper_native_v5"
            and envelope.get("policy") == PAPER_NATIVE_V5_POLICY_TOKEN
            and envelope.get("verification_status") == "verified"
            and envelope.get("is_primary_support") is True
            and envelope.get("molecule_chembl_id") == candidate_id
            and envelope.get("selected_group_sha256")
            == selected_group.get("group_sha256")
            and set(envelope.get("group_member_candidate_ids") or [])
            == set(group_ids)
            and pair.get("candidate_id") == candidate_id
            and support.get("molecule_chembl_id") == candidate_id
            and _hash_field_valid(support, "support_attestation_sha256")
            and envelope.get("cross_source_numeric_comparison_performed") is False
            and envelope.get("pubmed_value_used_for_candidate_rank") is False
            and envelope.get("no_partial_group_promotion") is True
        )
        envelope_checks.append(envelope_ok)
        is_control = candidate_id == RESERVED_CHEMBL_ID
        rank = (
            ranked_noncontrols.index(candidate_id) + 1
            if candidate_id in ranked_noncontrols
            else None
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": getattr(seed, "name", None) if seed is not None else None,
                "source_group_member": True,
                "projected": seed is not None,
                "envelope_verified": envelope_ok,
                "control_only": is_control,
                "policy_eligible": bool(not is_control and rank is not None),
                "rank": rank,
                "exclusion_reason": (
                    "RESERVED_ASSAY_CONTROL"
                    if is_control
                    else "NONCONTROL_QUOTA_CONTEXT_ONLY"
                    if rank is None
                    else None
                ),
                "selected_group_sha256": envelope.get("selected_group_sha256"),
                "selected_pair_evidence_sha256": envelope.get(
                    "selected_pair_evidence_sha256"
                ),
                "pre_pubmed_support_sha256": support.get(
                    "support_attestation_sha256"
                ),
                "fixture": False,
            }
        )
    noncontrol_count = sum(item != RESERVED_CHEMBL_ID for item in group_ids)
    checks = {
        "all_declared_members_projected": bool(group_ids)
        and set(group_ids).issubset(seeds_by_id),
        "all_member_envelopes_verified": bool(envelope_checks)
        and all(envelope_checks),
        "at_least_three_noncontrols_after_reserved": noncontrol_count >= 3,
        "eligible_rank_limit": len(ranked_noncontrols) <= PROMOTED_NONCONTROL_LIMIT,
        "fixed_pool_rank_order": ranked_noncontrols
        == [item for item in fixed_ids if item in set(ranked_noncontrols)],
        "reserved_not_ranked": all(
            item["rank"] is None and item["policy_eligible"] is False
            for item in rows
            if item["control_only"]
        ),
        "source_group_not_truncated": len(rows) == len(group_ids),
    }
    packet_constructed = False
    packet_seed_preservation = False
    packet: CandidatePacket | None = None
    packet_projection_sha256: str | None = None
    if isinstance(target, TargetCandidate) and rows:
        try:
            packet, packet_rows = _build_g1_candidate_packet(
                context=context,
                seeds=seeds,
                fixed_ids=fixed_ids,
                group_ids=group_ids,
                selected_group_sha256=str(
                    selected_group.get("group_sha256") or ""
                ),
                target=target,
                discovery_provenance=discovery_provenance,
            )
            projection = [
                {
                    "candidate_id": item.metadata.get("molecule_chembl_id"),
                    "rank": item.rank,
                    "status": item.status,
                    "exclusion_reasons": item.exclusion_reasons,
                }
                for item in packet.candidates
            ]
            packet_constructed = bool(packet_rows) and all(
                item.get("projected") is True for item in packet_rows
            )
            preservation = _mapping(
                packet.metadata.get("preflight_seed_preservation")
            )
            packet_seed_preservation = bool(
                preservation.get("verified") is True
                and _hash_field_valid(
                    preservation, "attestation_sha256"
                )
            )
            rows = packet_rows
            packet_projection_sha256 = _canonical_sha256(projection)
        except Exception:
            packet_constructed = False
            packet = None
    checks["candidate_packet_schema_projection"] = packet_constructed
    checks["candidate_packet_full_seed_preservation"] = (
        packet_seed_preservation
    )
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "source_group_candidate_ids": group_ids,
        "source_group_member_count": len(group_ids),
        "verified_noncontrol_candidate_ids": [
            item for item in group_ids if item != RESERVED_CHEMBL_ID
        ],
        "verified_noncontrol_count": noncontrol_count,
        "eligible_ranked_noncontrol_candidate_ids": ranked_noncontrols,
        "eligible_ranked_noncontrol_count": len(ranked_noncontrols),
        "candidate_packet_projection_preview": rows,
        "candidate_packet_constructed": packet_constructed,
        "candidate_packet_schema_version": "1.0" if packet_constructed else None,
        "candidate_packet_projection_sha256": packet_projection_sha256,
        "g1_faithful_full_seed_packet": packet_constructed,
        "deterministic_public_parser_replay_used": True,
    }, packet


_CHEMBL_ENDPOINT_ROLES: Mapping[str, str] = {
    LiveScientificGateway.status_endpoint: "status",
    LiveScientificGateway.target_endpoint: "target",
    LiveScientificGateway.mechanism_endpoint: "mechanism",
    LiveScientificGateway.molecule_endpoint: "molecule",
    LiveScientificGateway.activity_endpoint: "activity",
    LiveScientificGateway.assay_endpoint: "assay",
    LiveScientificGateway.warning_endpoint: "warning",
}


def _deposited_chembl_fingerprint(call: Mapping[str, Any]) -> str | None:
    parameter_keys = call.get("parameter_keys")
    if not isinstance(parameter_keys, list):
        return None
    body = {
        "method": "GET",
        "endpoint": call.get("endpoint"),
        "parameter_keys": parameter_keys,
        "source_parameter_sha256": call.get("parameter_hash"),
        "status": "completed",
        "status_code": call.get("status_code"),
        "response_sha256": call.get("response_sha256"),
        "attempt_count_observed": call.get("attempt_count"),
        "fixture": False,
    }
    if (
        call.get("source") != "ChEMBL"
        or call.get("cache_hit") is not False
        or body["endpoint"] not in _CHEMBL_ENDPOINT_ROLES
        or not isinstance(body["source_parameter_sha256"], str)
        or HASH_RE.fullmatch(str(body["source_parameter_sha256"])) is None
        or not isinstance(body["response_sha256"], str)
        or HASH_RE.fullmatch(str(body["response_sha256"])) is None
        or body["status_code"] != 200
        or call.get("max_attempts") != CHEMBL_MAX_ATTEMPTS
        or type(body["attempt_count_observed"]) is not int
        or not 1 <= int(body["attempt_count_observed"]) <= CHEMBL_MAX_ATTEMPTS
    ):
        return None
    return _canonical_sha256(body)


def _wire_chembl_fingerprint(call: Mapping[str, Any]) -> str | None:
    raw_keys = call.get("parameter_keys")
    if not isinstance(raw_keys, list):
        return None
    parameter_keys: list[str] = []
    for item in raw_keys:
        text = str(item)
        if not text.startswith("params."):
            return None
        parameter_keys.append(text.removeprefix("params."))
    body = {
        "method": call.get("method"),
        "endpoint": call.get("endpoint"),
        "parameter_keys": parameter_keys,
        "source_parameter_sha256": call.get("source_parameter_sha256"),
        "status": call.get("status"),
        "status_code": call.get("status_code"),
        "response_sha256": call.get("response_sha256"),
        "attempt_count_observed": call.get("attempt_count_observed"),
        "fixture": call.get("fixture"),
    }
    if (
        body["method"] != "GET"
        or body["endpoint"] not in _CHEMBL_ENDPOINT_ROLES
        or not isinstance(body["source_parameter_sha256"], str)
        or HASH_RE.fullmatch(str(body["source_parameter_sha256"])) is None
        or body["status"] != "completed"
        or body["status_code"] != 200
        or not isinstance(body["response_sha256"], str)
        or HASH_RE.fullmatch(str(body["response_sha256"])) is None
        or type(body["attempt_count_observed"]) is not int
        or not 1 <= int(body["attempt_count_observed"]) <= CHEMBL_MAX_ATTEMPTS
        or body["fixture"] is not False
    ):
        return None
    return _canonical_sha256(body)


def _audited_call_fingerprint(call: Mapping[str, Any]) -> str | None:
    keys = (
        "method",
        "endpoint",
        "parameter_keys",
        "parameter_sha256",
        "source_parameter_sha256",
        "status",
        "status_code",
        "response_sha256",
        "attempt_count_observed",
        "retry_diagnostics",
        "runtime_seconds",
        "fixture",
    )
    if set(call) != set(keys):
        return None
    body = {key: deepcopy(call.get(key)) for key in keys}
    if (
        body["method"] != "GET"
        or not isinstance(body["endpoint"], str)
        or not isinstance(body["parameter_keys"], list)
        or not isinstance(body["parameter_sha256"], str)
        or HASH_RE.fullmatch(body["parameter_sha256"]) is None
        or not isinstance(body["source_parameter_sha256"], str)
        or HASH_RE.fullmatch(body["source_parameter_sha256"]) is None
        or body["status"] != "completed"
        or body["status_code"] != 200
        or not isinstance(body["response_sha256"], str)
        or HASH_RE.fullmatch(body["response_sha256"]) is None
        or type(body["attempt_count_observed"]) is not int
        or not isinstance(body["retry_diagnostics"], list)
        or len(body["retry_diagnostics"])
        != body["attempt_count_observed"] - 1
        or any(
            not isinstance(item, Mapping)
            for item in body["retry_diagnostics"]
        )
        or not isinstance(body["runtime_seconds"], (int, float))
        or isinstance(body["runtime_seconds"], bool)
        or float(body["runtime_seconds"]) < 0
        or not math.isfinite(float(body["runtime_seconds"]))
        or body["fixture"] is not False
    ):
        return None
    return _canonical_sha256(body)


def _deposited_chembl_calls(
    target: Any,
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = _mapping(getattr(target, "metadata", {}))
    version = _mapping(metadata.get("chembl_version_resolution"))
    normalization = [
        deepcopy(dict(_mapping(version.get("provenance")))),
        deepcopy(dict(_mapping(metadata.get("chembl_target_resolution")))),
    ]
    retrieval_raw = provenance.get("http_calls")
    retrieval = (
        [deepcopy(dict(item)) for item in retrieval_raw if isinstance(item, Mapping)]
        if isinstance(retrieval_raw, list)
        else []
    )
    return normalization, retrieval


def _http_ledger_audit(
    target: Any,
    provenance: Mapping[str, Any],
    chembl_calls_before: Sequence[Mapping[str, Any]],
    chembl_calls_after_normalize: Sequence[Mapping[str, Any]],
    chembl_calls: Sequence[Mapping[str, Any]],
    pubmed_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allowed_pubmed = {PUBMED_SEARCH_URL, PUBMED_SUMMARY_URL, PUBMED_FETCH_URL}
    chembl_before_safe = [deepcopy(dict(item)) for item in chembl_calls_before]
    chembl_normalized_safe = [
        deepcopy(dict(item)) for item in chembl_calls_after_normalize
    ]
    chembl_safe = [deepcopy(dict(item)) for item in chembl_calls]
    pubmed_safe = [deepcopy(dict(item)) for item in pubmed_calls]
    observed_calls = [*chembl_safe, *pubmed_safe]
    allowed_observed_endpoints = {
        *_CHEMBL_ENDPOINT_ROLES,
        *allowed_pubmed,
    }
    observed_http_safety_checks = {
        "observed_http_nonempty": bool(observed_calls),
        "observed_methods_get_only": bool(observed_calls)
        and all(item.get("method") == "GET" for item in observed_calls),
        "observed_endpoints_public_allowlisted": bool(observed_calls)
        and all(
            item.get("endpoint") in allowed_observed_endpoints
            for item in observed_calls
        ),
        "observed_fixture_false": bool(observed_calls)
        and all(item.get("fixture") is False for item in observed_calls),
    }
    observed_http_safety_verified = all(
        observed_http_safety_checks.values()
    )
    if not pubmed_safe:
        pubmed_execution_state = "not_executed"
        pubmed_observed_public_read_only: bool | None = None
    elif len(pubmed_safe) == PUBMED_LOGICAL_CALL_LIMIT:
        pubmed_execution_state = "completed_required_sequence"
        pubmed_observed_public_read_only = all(
            item.get("method") == "GET"
            and item.get("endpoint") in allowed_pubmed
            and item.get("fixture") is False
            for item in pubmed_safe
        )
    else:
        pubmed_execution_state = "partial_or_mismatched"
        pubmed_observed_public_read_only = all(
            item.get("method") == "GET"
            and item.get("endpoint") in allowed_pubmed
            and item.get("fixture") is False
            for item in pubmed_safe
        )
    normalization_wire = chembl_normalized_safe[len(chembl_before_safe) :]
    retrieval_wire = chembl_safe[len(chembl_normalized_safe) :]
    _normalization_deposited, retrieval_deposited = _deposited_chembl_calls(
        target, provenance
    )
    deposited = retrieval_deposited
    deposited_fingerprints = [
        _deposited_chembl_fingerprint(item) for item in deposited
    ]
    wire_fingerprints = [
        _wire_chembl_fingerprint(item) for item in retrieval_wire
    ]
    deposited_counter = Counter(
        item for item in deposited_fingerprints if item is not None
    )
    wire_counter = Counter(item for item in wire_fingerprints if item is not None)
    missing = list((deposited_counter - wire_counter).elements())
    extras = list((wire_counter - deposited_counter).elements())
    endpoint_role_counts = Counter(
        _CHEMBL_ENDPOINT_ROLES.get(str(item.get("endpoint") or ""), "unknown")
        for item in chembl_safe
    )
    logical_raw = provenance.get("logical_client_calls")
    logical_calls = (
        [deepcopy(dict(item)) for item in logical_raw if isinstance(item, Mapping)]
        if isinstance(logical_raw, list)
        else []
    )
    logical_fingerprints = [
        _audited_call_fingerprint(item) for item in logical_calls
    ]
    retrieval_wire_fingerprints = [
        _audited_call_fingerprint(item) for item in retrieval_wire
    ]
    checks = {
        "chembl_nonempty": bool(chembl_safe) and bool(deposited),
        "chembl_phase_snapshots_monotonic": not chembl_before_safe
        and len(chembl_normalized_safe) >= 2
        and len(chembl_safe) > len(chembl_normalized_safe)
        and chembl_safe[: len(chembl_normalized_safe)]
        == chembl_normalized_safe,
        "chembl_normalize_harness_owned_phase": [
            _CHEMBL_ENDPOINT_ROLES.get(str(item.get("endpoint") or ""))
            for item in normalization_wire
        ]
        == ["status", "target"]
        and all(_wire_chembl_fingerprint(item) is not None for item in normalization_wire),
        "chembl_public_read_only": bool(chembl_safe) and all(
            item.get("method") == "GET"
            and str(item.get("endpoint") or "").startswith(
                "https://www.ebi.ac.uk/chembl/api/data/"
            )
            and item.get("fixture") is False
            for item in chembl_safe
        ),
        "chembl_attempt_bound": bool(chembl_safe) and all(
            type(item.get("attempt_count_observed")) is int
            and 1 <= item.get("attempt_count_observed") <= CHEMBL_MAX_ATTEMPTS
            for item in chembl_safe
        ),
        "chembl_deposited_rows_valid": bool(deposited_fingerprints)
        and None not in deposited_fingerprints,
        "chembl_wire_rows_valid": bool(wire_fingerprints)
        and None not in wire_fingerprints,
        "chembl_retrieval_exact_multiset_binding": bool(deposited_counter)
        and deposited_counter == wire_counter
        and not missing
        and not extras,
        "chembl_required_operation_roles": all(
            endpoint_role_counts.get(role, 0) > 0
            for role in (
                "status",
                "target",
                "mechanism",
                "molecule",
                "activity",
                "assay",
                "warning",
            )
        )
        and endpoint_role_counts.get("unknown", 0) == 0,
        "chembl_retrieval_logical_ledger_exact": bool(logical_calls)
        and None not in logical_fingerprints
        and None not in retrieval_wire_fingerprints
        and Counter(logical_fingerprints)
        == Counter(retrieval_wire_fingerprints),
        "pubmed_public_read_only": len(pubmed_safe)
        == PUBMED_LOGICAL_CALL_LIMIT
        and all(
            item.get("method") == "GET"
            and item.get("endpoint") in allowed_pubmed
            and item.get("fixture") is False
            for item in pubmed_safe
        ),
        "pubmed_wire_rows_valid": len(pubmed_safe)
        == PUBMED_LOGICAL_CALL_LIMIT
        and all(
            _audited_call_fingerprint(item) is not None
            for item in pubmed_safe
        ),
        "pubmed_logical_bound": len(pubmed_safe) == PUBMED_LOGICAL_CALL_LIMIT,
        "pubmed_stage_endpoint_order": [
            item.get("endpoint") for item in pubmed_safe
        ]
        == [PUBMED_SEARCH_URL, PUBMED_SUMMARY_URL, PUBMED_FETCH_URL],
        "pubmed_attempt_bound": bool(pubmed_safe) and all(
            item.get("attempt_count_observed") == PUBMED_MAX_ATTEMPTS
            for item in pubmed_safe
        ),
    }
    required_stage_ledger_complete = all(checks.values())
    return {
        "verified": required_stage_ledger_complete,
        "required_stage_ledger_complete": required_stage_ledger_complete,
        "observed_http_safety_verified": observed_http_safety_verified,
        "observed_http_safety_checks": observed_http_safety_checks,
        "observed_http_call_count": len(observed_calls),
        "checks": checks,
        "chembl_operation_timeout_seconds": CHEMBL_OPERATION_TIMEOUT_SECONDS,
        "source_http_timeout_seconds": SOURCE_HTTP_TIMEOUT_SECONDS,
        "chembl_max_attempts": CHEMBL_MAX_ATTEMPTS,
        "chembl_logical_call_count": len(chembl_safe),
        "chembl_logical_calls": chembl_safe,
        "chembl_deposited_call_count": len(deposited),
        "chembl_harness_owned_normalization_call_count": len(
            normalization_wire
        ),
        "chembl_deposited_retrieval_call_count": len(retrieval_deposited),
        "chembl_endpoint_role_counts": dict(sorted(endpoint_role_counts.items())),
        "chembl_deposited_multiset_sha256": _canonical_sha256(
            sorted(deposited_counter.elements())
        ),
        "chembl_wire_multiset_sha256": _canonical_sha256(
            sorted(wire_counter.elements())
        ),
        "chembl_missing_binding_count": len(missing),
        "chembl_missing_binding_fingerprints": sorted(missing),
        "chembl_extra_wire_count": len(extras),
        "chembl_extra_wire_fingerprints": sorted(extras),
        "normalization_provenance_boundary": {
            "harness_owned_phase_snapshot": True,
            "independent_deposited_ledger_re_attestation_claimed": False,
            "exact_target_output_checked_separately": True,
            "phase_start_call_count": len(chembl_before_safe),
            "phase_start_ordered_calls_sha256": _canonical_sha256(
                chembl_before_safe
            ),
            "phase_end_call_count": len(chembl_normalized_safe),
            "phase_call_count": len(normalization_wire),
            "phase_ordered_calls_sha256": _canonical_sha256(
                normalization_wire
            ),
        },
        "retrieval_wire_ordered_calls_sha256": _canonical_sha256(
            retrieval_wire
        ),
        "retrieval_deposited_logical_calls_sha256": _canonical_sha256(
            logical_calls
        ),
        "pubmed_max_attempts": PUBMED_MAX_ATTEMPTS,
        "pubmed_logical_call_limit": PUBMED_LOGICAL_CALL_LIMIT,
        "pubmed_logical_call_count": len(pubmed_safe),
        "pubmed_logical_calls": pubmed_safe,
        "pubmed_execution_state": pubmed_execution_state,
        "pubmed_call_checks_applicable": bool(pubmed_safe),
        "pubmed_observed_public_read_only": (
            pubmed_observed_public_read_only
        ),
        "raw_response_bodies_stored": False,
        "full_abstracts_stored": False,
        "secret_headers_or_values_stored": False,
    }


def _runtime_calls(runtime: Any, name: str) -> list[dict[str, Any]]:
    method = getattr(runtime, name, None)
    raw = method() if callable(method) else []
    return [deepcopy(dict(item)) for item in raw if isinstance(item, Mapping)]


def _runtime_admet(runtime: Any, seeds: Sequence[Any]) -> dict[str, Any]:
    method = getattr(runtime, "admet_readiness", None)
    if not callable(method):
        return {
            "status": "unavailable",
            "reason_code": "ADMET_LOCAL_READINESS_CONTRACT_MISSING",
            "admet_inference_executed": False,
            "fixture": False,
        }
    raw = method(seeds)
    return deepcopy(dict(raw)) if isinstance(raw, Mapping) else {
        "status": "unavailable",
        "reason_code": "ADMET_LOCAL_READINESS_INVALID",
        "admet_inference_executed": False,
        "fixture": False,
    }


def _gateway_contract_audit(
    gateway: Any,
    *,
    execution_mode: Literal["live", "contract_test"],
) -> dict[str, Any]:
    primary_client = getattr(gateway, "primary_search_client", None)
    checks = {
        "production_class_on_live_path": (
            execution_mode == "contract_test"
            or isinstance(gateway, LiveScientificGateway)
        ),
        "fixture_false": getattr(gateway, "fixture", None) is False,
        "exact_policy_dispatch": getattr(
            gateway, "repurposing_evidence_policy", None
        )
        == PAPER_NATIVE_V5_POLICY_TOKEN,
        "chembl_operation_timeout": getattr(
            gateway, "operation_timeout_seconds", None
        )
        == CHEMBL_OPERATION_TIMEOUT_SECONDS,
        "source_http_timeout": getattr(gateway, "timeout_seconds", None)
        == SOURCE_HTTP_TIMEOUT_SECONDS,
        "chembl_retry_bound": getattr(gateway, "max_http_attempts", None)
        == CHEMBL_MAX_ATTEMPTS,
        "dedicated_pubmed_transport": primary_client
        is not getattr(gateway, "client", None),
        "pubmed_single_attempt": getattr(primary_client, "max_attempts", None)
        == PUBMED_MAX_ATTEMPTS,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "gateway_class": type(gateway).__name__,
        "policy_marker": getattr(gateway, "repurposing_evidence_policy", None),
        "chembl_operation_timeout_seconds": getattr(
            gateway, "operation_timeout_seconds", None
        ),
        "source_http_timeout_seconds": getattr(
            gateway, "timeout_seconds", None
        ),
        "chembl_max_attempts": getattr(gateway, "max_http_attempts", None),
        "pubmed_max_attempts": getattr(primary_client, "max_attempts", None),
    }


def _execute(
    context: ValidationContext,
    runtime: Any,
    *,
    execution_mode: Literal["live", "contract_test"],
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    gateway = getattr(runtime, "gateway", None)
    if gateway is None:
        raise PreflightContractError("RUNTIME_GATEWAY_MISSING")
    gateway_contract = _gateway_contract_audit(
        gateway, execution_mode=execution_mode
    )
    runtime_environment_gate = deepcopy(dict(runtime_environment))
    runtime_environment_verified = _runtime_environment_gate_valid(
        runtime_environment_gate
    )
    harness_manifest = _harness_manifest()
    source_target = _target_source(context)
    normalized: Any = None
    seeds: tuple[Any, ...] = ()
    operation_error: dict[str, Any] | None = None
    normalize_calls = 0
    discovery_calls = 0
    chembl_calls_before = _runtime_calls(runtime, "chembl_calls")
    chembl_calls_after_normalize: list[dict[str, Any]] | None = None
    try:
        normalize_calls += 1
        normalized = gateway.normalize_target(source_target)
        chembl_calls_after_normalize = _runtime_calls(runtime, "chembl_calls")
        discovery_calls += 1
        seeds = tuple(
            gateway.retrieve_repurposing(
                normalized,
                direction="inhibit",
                limit=PROMOTED_NONCONTROL_LIMIT,
            )
        )
    except Exception as exc:
        operation_error = {
            "exception_type": type(exc).__name__,
            "reason_code": str(
                _mapping(getattr(exc, "provenance", {})).get("reason_code")
                or "SCIENTIFIC_RETRIEVAL_UNAVAILABLE"
            ),
            "provenance_sha256": _canonical_sha256(
                _mapping(getattr(exc, "provenance", {}))
            ),
        }
    provenance = _mapping(getattr(gateway, "last_operation_provenance", {}))
    chembl_calls = _runtime_calls(runtime, "chembl_calls")
    if chembl_calls_after_normalize is None:
        chembl_calls_after_normalize = deepcopy(chembl_calls)
    pubmed_calls = _runtime_calls(runtime, "pubmed_calls")
    target_gate = _exact_target_check(normalized)
    mechanism = _mechanism_pool_audit(provenance, seeds)
    bounds = _mapping(provenance.get("candidate_discovery_bounds"))
    search_provenance = _mapping(bounds.get("primary_abstract_search"))
    direct = _direct_support_audit(
        search_provenance, mechanism["fixed_pool_candidate_ids"]
    )
    aliases = _alias_audit(
        search_provenance,
        mechanism["fixed_pool_candidate_ids"],
        direct["selected_candidate_ids"],
    )
    pubmed = _pubmed_audit(search_provenance, pubmed_calls)
    projection, g1_packet = _projection_audit(
        context,
        seeds,
        mechanism["fixed_pool_candidate_ids"],
        pubmed,
        normalized,
        provenance,
    )
    ma3 = (
        _ma3_v5_audit(context, g1_packet)
        if g1_packet is not None
        else {
            "verified": False,
            "deterministic_audit_invoked": False,
            "audit_owner": "drugpilot.agents.ma3.MA3AssuranceAgent",
            "audit_method": "_audit_candidate_primary_sources_for_policy",
            "private_hash_pinned_v5_only_boundary": True,
            "producer_parser_or_search_called_by_audit": False,
            "external_reviewer_invoked": False,
            "model_or_llm_calls": 0,
            "finding_count": None,
            "findings": [],
            "reason_code": "G1_FAITHFUL_PACKET_NOT_CONSTRUCTED",
        }
    )
    http_ledger = _http_ledger_audit(
        normalized,
        provenance,
        chembl_calls_before,
        chembl_calls_after_normalize,
        chembl_calls,
        pubmed_calls,
    )
    eligible_seed_ids = set(projection["eligible_ranked_noncontrol_candidate_ids"])
    admet_seeds = [seed for seed in seeds if _seed_id(seed) in eligible_seed_ids]
    admet = _runtime_admet(runtime, admet_seeds)
    eligible_candidate_count = len(eligible_seed_ids)
    admet_upstream_dependency_satisfied = bool(
        eligible_candidate_count >= PROMOTED_NONCONTROL_LIMIT
        and len(admet_seeds) == eligible_candidate_count
    )
    if not admet_seeds:
        admet.update(
            {
                "status": "upstream_not_evaluated",
                "reason_code": "UPSTREAM_NO_ELIGIBLE_CANDIDATES",
            }
        )
    admet.update(
        {
            "evaluation_applicable": bool(admet_seeds),
            "eligible_candidate_count": eligible_candidate_count,
            "structure_input_candidate_count": len(admet_seeds),
            "minimum_required_candidate_count": PROMOTED_NONCONTROL_LIMIT,
            "upstream_dependency_satisfied": (
                admet_upstream_dependency_satisfied
            ),
            "blocking_applicable": (
                eligible_candidate_count >= PROMOTED_NONCONTROL_LIMIT
            ),
        }
    )
    admet_verified = bool(
        admet_upstream_dependency_satisfied
        and admet.get("status") == "completed"
        and admet.get("fixture") is False
        and admet.get("admet_inference_executed") is False
        and len(admet_seeds) == len(eligible_seed_ids) >= 3
    )
    blockers: list[str] = []
    gates: list[tuple[bool, str]] = [
        (
            runtime_environment_verified,
            "RUNTIME_ENVIRONMENT_NOT_VERIFIED",
        ),
        (gateway_contract["verified"], "PRODUCTION_GATEWAY_CONTRACT_MISMATCH"),
        (harness_manifest["verified"], "HARNESS_MANIFEST_NOT_VERIFIED"),
        (operation_error is None, "SCIENTIFIC_RETRIEVAL_UNAVAILABLE"),
        (normalize_calls == 1, "TARGET_NORMALIZATION_CALL_COUNT_MISMATCH"),
        (discovery_calls == 1, "MECHANISM_DISCOVERY_CALL_COUNT_MISMATCH"),
        (target_gate["verified"], "EXACT_JAK2_O60674_NOT_VERIFIED"),
        (mechanism["verified"], "FIXED_TOP12_MECHANISM_POOL_NOT_VERIFIED"),
        (direct["verified"], "STRICT_DIRECT_SUPPORT_NOT_VERIFIED"),
        (aliases["verified"], "ALIAS_INVENTORY_NOT_VERIFIED"),
        (pubmed["verified"], "JOINT_PUBMED_GROUP_NOT_VERIFIED"),
        (projection["verified"], "GROUP_MEMBER_PROJECTION_NOT_VERIFIED"),
        (ma3["verified"], "MA3_DETERMINISTIC_V5_AUDIT_NOT_VERIFIED"),
    ]
    if eligible_candidate_count >= PROMOTED_NONCONTROL_LIMIT:
        gates.append((admet_verified, "ADMET_LOCAL_NOT_READY"))
    gates.append((http_ledger["verified"], "HTTP_LEDGER_NOT_VERIFIED"))
    for passed, code in gates:
        if not passed:
            blockers.append(code)
    blockers = list(dict.fromkeys(blockers))
    would_ready = not blockers
    decision = (
        "PREFLIGHT_READY_FOR_BOUNDED_PAID_VALIDATION"
        if execution_mode == "live" and would_ready
        else "PRECHECK_NOT_READY"
        if execution_mode == "live"
        else "CONTRACT_TEST_ONLY"
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "execution_mode": execution_mode,
        "decision": decision,
        "contract_test_would_be_decision": (
            "PREFLIGHT_READY_FOR_BOUNDED_PAID_VALIDATION"
            if execution_mode == "contract_test" and would_ready
            else "PRECHECK_NOT_READY"
            if execution_mode == "contract_test"
            else None
        ),
        "paid_run_authorized": bool(execution_mode == "live" and would_ready),
        "paid_run_stop_required": not bool(execution_mode == "live" and would_ready),
        "immutable_request_gate": {
            "request_file_name": context.request_path.name,
            "request_sha256": context.request_sha256,
            "protocol_file_name": context.protocol_path.name,
            "protocol_sha256": context.protocol_sha256,
            "r1_protocol_file_name": context.r1_protocol_path.name,
            "r1_protocol_sha256": context.r1_protocol_sha256,
            "prior_result_file_name": context.prior_result_path.name,
            "prior_result_file_sha256": context.prior_result_file_sha256,
            "prior_result_report_sha256": (
                context.prior_result_report_sha256
            ),
            "prior_result_decision": "PRECHECK_NOT_READY",
            "prior_result_immutable_and_unmodified": True,
            "r2_lineage_verified_before_http": True,
            "create_run_request_ast_model_sha256": (
                context.create_run_request_model_sha256
            ),
            "create_run_request_ast_validated": True,
            "unknown_top_level_key_count": 0,
            "policy_marker": PAPER_NATIVE_V5_POLICY_TOKEN,
            "policy_marker_occurrence_count": 1,
            "source_manifest": [deepcopy(item) for item in context.source_manifest],
            "source_manifest_verified_before_http": True,
            "production_tree_file_count": context.production_tree_file_count,
            "production_tree_manifest_sha256": (
                context.production_tree_manifest_sha256
            ),
            "production_tree_verified_before_http": True,
        },
        "one_live_attempt_contract": {
            "attempt_ordinal": 1,
            "attempt_scope": "r2_free_preflight",
            "lineage_preflight_attempt_ordinal": 2,
            "fresh_attempt_reason": (
                "validated_integral_max_phase_canonical_binding_build_fix"
            ),
            "gate_threshold_candidate_pmid_change": False,
            "prior_not_ready_result_preserved": True,
            "automatic_paid_run_start": False,
            "automatic_rerun": False,
            "rerun_requires_new_recorded_reason": True,
            "output_mode": "exclusive_create_append_only_record",
        },
        "harness_manifest": harness_manifest,
        "runtime_environment_gate": runtime_environment_gate,
        "production_gateway_contract": gateway_contract,
        "target_gate": target_gate,
        "mechanism_first_gate": mechanism,
        "direct_support_gate": direct,
        "alias_inventory_gate": aliases,
        "joint_pubmed_gate": pubmed,
        "candidate_projection_gate": projection,
        "ma3_deterministic_v5_gate": ma3,
        "admet_local_readiness_gate": {
            **admet,
            "verified": admet_verified,
        },
        "http_ledger": http_ledger,
        "execution_boundary": {
            "read_only_public_http_only": http_ledger[
                "observed_http_safety_verified"
            ],
            "observed_http_safety_verified": http_ledger[
                "observed_http_safety_verified"
            ],
            "required_http_stage_ledger_complete": http_ledger[
                "required_stage_ledger_complete"
            ],
            "model_calls": 0,
            "llm_calls": 0,
            "model_tokens": 0,
            "model_cost_usd": 0,
            "admet_inference_calls": 0,
            "run_db_reads": 0,
            "run_db_writes": 0,
            "api_server_calls": 0,
            "fixture_calls": 0,
            "fixture_records_detected": 0,
            "runtime_environment_verified": runtime_environment_verified,
            "runtime_environment_verified_before_live_runtime_construction": (
                True if execution_mode == "live" else None
            ),
            "injected_runtime_count": 0 if execution_mode == "live" else 1,
            "candidate_injection_count": 0,
            "candidate_allowlist_count": 0,
            "raw_response_bodies_stored": False,
            "full_abstracts_stored": False,
            "secret_values_stored": False,
            "ma3_deterministic_v5_audit_invoked": ma3[
                "deterministic_audit_invoked"
            ],
            "ma3_external_reviewer_invoked": False,
        },
        "blockers": blockers,
        "scientific_operation_error": operation_error,
        "interpretation": {
            "ready_is_only_paid_validation_permission": True,
            "g1_pass_guaranteed": False,
            "clinical_efficacy_claimed": False,
            "clinical_safety_claimed": False,
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
    """Execute the only live path; no runtime or transport can be injected."""

    context = load_validation_context(request_path, protocol_path)
    runtime_environment = _require_runtime_environment()
    runtime = _LiveRuntime()
    try:
        return _execute(
            context,
            runtime,
            execution_mode="live",
            runtime_environment=runtime_environment,
        )
    finally:
        runtime.close()


def run_contract_test(
    request_path: Path,
    *,
    protocol_path: Path,
    runtime: Any,
) -> dict[str, Any]:
    """Synthetic-only test seam; its result can never authorize a paid run."""

    context = load_validation_context(request_path, protocol_path)
    try:
        runtime_environment = _require_runtime_environment()
        return _execute(
            context,
            runtime,
            execution_mode="contract_test",
            runtime_environment=runtime_environment,
        )
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()


def _assert_safe_report(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in {
                "abstract",
                "full_abstract",
                "full_text",
                "raw_response",
                "response_body",
                "api_key",
                "authorization",
                "password",
                "secret",
            }:
                raise RuntimeError(f"forbidden report field: {path}.{key}")
            _assert_safe_report(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_safe_report(item, f"{path}[{index}]")
        return
    if isinstance(value, str) and re.search(r"\bsk-[A-Za-z0-9_-]{12,}", value):
        raise RuntimeError(f"possible secret value: {path}")
    if path.endswith(".sentence") and isinstance(value, str) and len(value) > 360:
        raise RuntimeError(f"evidence sentence exceeds bound: {path}")


def _validated_output_path(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if resolved.parent != RESULTS_DIR or RESULT_NAME.fullmatch(resolved.name) is None:
        raise PreflightContractError("OUTPUT_PATH_NOT_APPEND_ONLY_REGISTER")
    return resolved


def _reserve_output(raw: str) -> tuple[Path, TextIO]:
    output = _validated_output_path(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, output.open("x", encoding="utf-8", newline="\n")


def _attest_output_reservation(
    report: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    final = deepcopy(dict(report))
    final.pop("report_sha256", None)
    final["output_record"] = {
        "file_name": output.name,
        "exclusive_create_reserved_before_context_and_runtime": True,
        "held_handle_used_for_terminal_record": True,
        "stdout_authorization_forbidden": True,
    }
    final["report_sha256"] = _canonical_sha256(final)
    return final


def emit_report(report: Mapping[str, Any], handle: TextIO) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


def _fatal_report(exc: Exception) -> dict[str, Any]:
    fatal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "execution_mode": "live",
        "decision": "PRECHECK_NOT_READY",
        "paid_run_authorized": False,
        "paid_run_stop_required": True,
        "fatal_error": {
            "exception_type": type(exc).__name__,
            "reason_code": getattr(exc, "code", "UNCLASSIFIED_PREFLIGHT_FATAL"),
        },
        "execution_boundary": {
            "model_calls": 0,
            "llm_calls": 0,
            "model_tokens": 0,
            "model_cost_usd": 0,
            "admet_inference_calls": 0,
            "run_db_reads": 0,
            "run_db_writes": 0,
            "api_server_calls": 0,
            "fixture_calls": 0,
            "fixture_records_detected": 0,
            "runtime_environment_verified": False,
            "runtime_environment_checked_before_live_runtime_construction": (
                isinstance(exc, RuntimeEnvironmentMismatch)
            ),
            "injected_runtime_count": 0,
            "candidate_injection_count": 0,
            "candidate_allowlist_count": 0,
            "raw_response_bodies_stored": False,
            "full_abstracts_stored": False,
            "secret_values_stored": False,
        },
        "human_gate_auto_submit": False,
    }
    runtime_environment = getattr(
        exc, "runtime_environment_gate", None
    )
    if isinstance(runtime_environment, Mapping):
        fatal["runtime_environment_gate"] = deepcopy(
            dict(runtime_environment)
        )
    _assert_safe_report(fatal)
    fatal["report_sha256"] = _canonical_sha256(fatal)
    return fatal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot, read-only, no-model JAK2/ET paper-native-v5 free preflight"
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "required docs/validation/results/"
            "JAK2_ET_PAPER_NATIVE_V5_FREE_PREFLIGHT_<timestamp>.json"
        ),
    )
    return parser


def _terminal_exit_code(report: Mapping[str, Any]) -> int:
    if (
        report.get("decision")
        == "PREFLIGHT_READY_FOR_BOUNDED_PAID_VALIDATION"
        and report.get("paid_run_authorized") is True
        and report.get("paid_run_stop_required") is False
    ):
        return 0
    if (
        report.get("decision") == "PRECHECK_NOT_READY"
        and report.get("paid_run_authorized") is False
        and report.get("paid_run_stop_required") is True
    ):
        return 2
    raise PreflightContractError("PREFLIGHT_TERMINAL_STATE_INVALID")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output, handle = _reserve_output(args.output)
    except Exception as exc:
        sys.stderr.write(
            json.dumps(_fatal_report(exc), ensure_ascii=False, default=str)
            + "\n"
        )
        return 1
    with handle:
        try:
            report = run_preflight(args.input, protocol_path=args.protocol)
            exit_code = _terminal_exit_code(report)
            persisted = _attest_output_reservation(report, output)
            emit_report(persisted, handle)
        except Exception as exc:
            fatal = _attest_output_reservation(_fatal_report(exc), output)
            try:
                handle.seek(0)
                handle.truncate(0)
                emit_report(fatal, handle)
            except Exception:
                pass
            return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
