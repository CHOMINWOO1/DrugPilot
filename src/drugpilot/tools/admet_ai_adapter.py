from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread
from time import monotonic
from typing import Any, Mapping
from uuid import uuid4

from drugpilot.tools.executable import (
    ExecutionProvenance,
    ScientificDataUnavailable,
    ScientificToolExecutionError,
    ScientificToolTimeout,
    ScientificToolUnavailable,
    bounded_excerpt,
)
from drugpilot.tools.worker_runtime import (
    ISOLATED_PYTHON_FLAGS,
    isolated_worker_environment,
    validate_isolated_worker_environment,
    worker_environment_snapshot,
)


_CLASSIFICATION_ENDPOINTS = {
    "hepatotoxicity": "DILI",
    "herg_risk": "hERG",
    "ames_risk": "AMES",
    "cyp3a4_inhibition": "CYP3A4_Veith",
    "oral_bioavailability": "Bioavailability_Ma",
}
_CLEARANCE_RAW_ENDPOINT = "Clearance_Hepatocyte_AZ"
_CLEARANCE_PERCENTILE_ENDPOINT = (
    "Clearance_Hepatocyte_AZ_drugbank_approved_percentile"
)
_ADMET_WORKER_PATH = Path(__file__).resolve().with_name("admet_worker.py")
_REQUIRED_WORKER_PACKAGES = ("admet-ai", "chemprop", "torch", "rdkit")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ADMETExecutionProvenance:
    backend: str
    fixture: bool
    model_metadata: Mapping[str, Any]
    endpoint_mapping: Mapping[str, Any]
    timeout_seconds: float
    started_at: str
    finished_at: str
    runtime_seconds: float
    input_sha256: str
    output_sha256: str
    cache_hit: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ADMETPredictionResult:
    scores: Mapping[str, float]
    raw_selected_predictions: Mapping[str, float]
    execution: ADMETExecutionProvenance

    def provenance(self) -> dict[str, Any]:
        return {
            "fixture": False,
            "method": "ADMET-AI pretrained Chemprop ensemble",
            "scores": dict(self.scores),
            "raw_selected_predictions": dict(self.raw_selected_predictions),
            "execution": self.execution.as_dict(),
        }


class ADMETAIAdapter:
    """Persistent, timeout-controlled ADMET-AI v2 inference adapter.

    Five panel values are direct classification probabilities. The legacy
    ``clearance_risk`` slot is *not* a fabricated model probability: it is the
    percentile (0-1) of the raw ADMET-AI hepatocyte-clearance regression value
    relative to ADMET-AI's packaged approved-DrugBank reference predictions.
    Both the raw value and this semantic are retained in provenance.
    """

    backend_kind = "live"
    fixture = False

    def __init__(
        self,
        *,
        python_executable: str | Path | None = None,
        startup_timeout_seconds: float = 180.0,
        prediction_timeout_seconds: float = 90.0,
        readiness_timeout_seconds: float = 30.0,
        worker_environment: Mapping[str, str] | None = None,
        worker_source_sha256: str | None = None,
    ) -> None:
        if (
            startup_timeout_seconds <= 0
            or prediction_timeout_seconds <= 0
            or readiness_timeout_seconds <= 0
        ):
            raise ValueError("ADMET-AI timeouts must be positive")
        try:
            self.python_executable = Path(
                python_executable or sys.executable
            ).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ScientificToolUnavailable(
                "ADMET-AI Python executable is missing"
            ) from exc
        if not self.python_executable.is_file():
            raise ScientificToolUnavailable("ADMET-AI Python executable is missing")
        self.startup_timeout_seconds = startup_timeout_seconds
        self.prediction_timeout_seconds = prediction_timeout_seconds
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self._worker_environment = (
            isolated_worker_environment()
            if worker_environment is None
            else validate_isolated_worker_environment(worker_environment)
        )
        observed_worker_sha256 = sha256(_ADMET_WORKER_PATH.read_bytes()).hexdigest()
        self._worker_source_sha256 = (
            observed_worker_sha256
            if worker_source_sha256 is None
            else worker_source_sha256
        )
        if (
            not isinstance(self._worker_source_sha256, str)
            or len(self._worker_source_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in self._worker_source_sha256)
            or self._worker_source_sha256 != observed_worker_sha256
        ):
            raise ScientificToolUnavailable("ADMET-AI worker source is not frozen")
        self._process: subprocess.Popen[str] | None = None
        self._messages: Queue[dict[str, Any]] = Queue()
        self._stderr_lines: list[str] = []
        self._lock = Lock()
        self._model_metadata: dict[str, Any] = {}
        self._readiness_cache: dict[str, Any] | None = None
        self._readiness_worker_payload: dict[str, Any] = {}
        self._cache: dict[str, ADMETPredictionResult] = {}
        self.last_result: ADMETPredictionResult | None = None

    @property
    def source_version(self) -> str:
        packages = self._readiness_worker_payload.get("packages")
        if not isinstance(packages, Mapping):
            packages = {}
        return (
            f"admet-ai={packages.get('admet-ai') or 'worker-unverified'};"
            f"chemprop={packages.get('chemprop') or 'worker-unverified'};"
            f"torch={packages.get('torch') or 'worker-unverified'}"
        )

    def readiness(self) -> dict[str, Any]:
        self._require_worker_source()
        if self._readiness_cache is not None:
            cached = deepcopy(self._readiness_cache)
            provenance = dict(cached.get("provenance") or {})
            provenance["worker_started"] = bool(
                self._process is not None and self._process.poll() is None
            )
            cached["provenance"] = provenance
            return cached

        command = self._worker_command("--check")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        timed_out = False
        return_code: int | None = None
        stdout = ""
        stderr = ""
        payload: dict[str, Any] = {}
        error_type: str | None = None
        try:
            completed = subprocess.run(
                command,
                shell=False,
                env=dict(self._worker_environment),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.readiness_timeout_seconds,
                check=False,
                creationflags=creationflags,
            )
            return_code = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            lines = [line for line in stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError("ADMET readiness emitted an invalid line count")
            parsed = json.loads(lines[0])
            if not isinstance(parsed, dict):
                raise ValueError("ADMET readiness emitted a non-object")
            payload = parsed
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            error_type = type(exc).__name__
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error_type = type(exc).__name__

        packages = payload.get("packages")
        importable = payload.get("package_modules_importable")
        checkpoint_count = payload.get("checkpoint_count")
        ready = bool(
            return_code == 0
            and payload.get("type") == "readiness"
            and payload.get("status") == "ready"
            and isinstance(packages, dict)
            and set(packages) == set(_REQUIRED_WORKER_PACKAGES)
            and all(isinstance(packages[name], str) and packages[name]
                    for name in _REQUIRED_WORKER_PACKAGES)
            and isinstance(importable, dict)
            and importable == {name: True for name in _REQUIRED_WORKER_PACKAGES}
            and type(checkpoint_count) is int
            and checkpoint_count > 0
            and payload.get("model_loaded") is False
            and payload.get("inference_executed") is False
            and payload.get("network_access_required") is False
            and payload.get("python_isolated") is True
            and payload.get("no_user_site") is True
            and payload.get("worker_source_sha256")
            == self._worker_source_sha256
        )
        if ready:
            self._readiness_worker_payload = dict(payload)
        missing = (
            list(payload.get("missing_packages"))
            if isinstance(payload.get("missing_packages"), list)
            else list(_REQUIRED_WORKER_PACKAGES)
        )
        result = {
            "tool": "ADMET-AI",
            "operation": "predict_six_endpoint_panel",
            "status": "completed" if ready else "unavailable",
            "backend": "local_pretrained_model",
            "version": packages.get("admet-ai") if isinstance(packages, dict) else None,
            "fixture": False,
            "required": True,
            "reason": None if ready else (
                f"configured interpreter missing packages: {missing}"
                if missing
                else "configured interpreter readiness probe failed"
            ),
            "provenance": {
                "packages": dict(packages) if isinstance(packages, dict) else {},
                "checkpoint_count": (
                    checkpoint_count if type(checkpoint_count) is int else 0
                ),
                "configured_interpreter_probe": True,
                "model_loaded_during_readiness": False,
                "inference_executed_during_readiness": False,
                "network_required_during_readiness": False,
                "timeout_seconds": self.readiness_timeout_seconds,
                "timed_out": timed_out,
                "exit_code": return_code,
                "error_type": error_type,
                "stdout_sha256": sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": sha256(stderr.encode("utf-8")).hexdigest(),
                "stderr_excerpt": bounded_excerpt(stderr, limit=240),
                "worker_source_sha256": sha256(
                    _ADMET_WORKER_PATH.read_bytes()
                ).hexdigest(),
                "worker_environment": worker_environment_snapshot(
                    self._worker_environment
                ),
                "worker_runtime_attestation": {
                    "schema_version": "drugpilot_local_readiness_child/v1",
                    "worker": "ADMET-AI",
                    "command_mode": "--check",
                    "python_isolation_flags": list(ISOLATED_PYTHON_FLAGS),
                    "worker_filename": _ADMET_WORKER_PATH.name,
                    "worker_source_sha256": payload.get(
                        "worker_source_sha256"
                    ),
                    "worker_environment": worker_environment_snapshot(
                        self._worker_environment
                    ),
                    "model_loaded": payload.get("model_loaded"),
                    "inference_executed": payload.get("inference_executed"),
                    "network_access_required": payload.get(
                        "network_access_required"
                    ),
                    "monitor_direct_child_instrumentation": False,
                    "attestation_basis": (
                        "frozen_worker_source_and_sanitized_process_contract"
                    ),
                },
                "readiness_execution": {
                    "sanitized_argv": [
                        self.python_executable.name,
                        *ISOLATED_PYTHON_FLAGS,
                        "<path:admet_worker.py>",
                        "--check",
                    ],
                    "timeout_seconds": self.readiness_timeout_seconds,
                    "exit_code": return_code,
                    "timed_out": timed_out,
                    "stdout_sha256": sha256(
                        stdout.encode("utf-8")
                    ).hexdigest(),
                    "stderr_sha256": sha256(
                        stderr.encode("utf-8")
                    ).hexdigest(),
                },
                "worker_started": False,
            },
        }
        self._readiness_cache = deepcopy(result)
        return result

    def _worker_command(self, mode: str) -> list[str]:
        if mode not in {"--check", "--serve"}:
            raise ValueError("unsupported ADMET worker mode")
        return [
            str(self.python_executable),
            *ISOLATED_PYTHON_FLAGS,
            str(_ADMET_WORKER_PATH),
            mode,
        ]

    def _require_worker_source(self) -> None:
        if sha256(_ADMET_WORKER_PATH.read_bytes()).hexdigest() != (
            self._worker_source_sha256
        ):
            raise ScientificToolUnavailable(
                "ADMET-AI worker source changed after runtime freeze"
            )

    def predict(self, *, smiles: str) -> Mapping[str, float]:
        return self.predict_with_provenance(smiles=smiles).scores

    def predict_with_provenance(self, *, smiles: str) -> ADMETPredictionResult:
        normalized = smiles.strip()
        if not normalized:
            raise ValueError("SMILES must not be blank")
        with self._lock:
            cached = self._cache.get(normalized)
            if cached is not None:
                result = replace(
                    cached,
                    execution=replace(cached.execution, cache_hit=True, runtime_seconds=0.0),
                )
                self.last_result = result
                return result
            self._ensure_worker()
            assert self._process is not None and self._process.stdin is not None
            request_id = uuid4().hex
            request = {"id": request_id, "smiles": normalized}
            started_at = _utc_iso()
            started = monotonic()
            try:
                self._process.stdin.write(
                    json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n"
                )
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate_worker()
                raise ScientificToolUnavailable("ADMET-AI worker pipe is unavailable") from exc

            response = self._await_message(
                expected_id=request_id,
                timeout_seconds=self.prediction_timeout_seconds,
            )
            runtime = monotonic() - started
            if response is None:
                provenance = self._failure_provenance(
                    started_at=started_at,
                    runtime_seconds=runtime,
                    timed_out=True,
                )
                self._terminate_worker()
                raise ScientificToolTimeout(
                    f"ADMET-AI exceeded {self.prediction_timeout_seconds:g} seconds",
                    provenance=provenance,
                )
            if response.get("type") != "prediction":
                provenance = self._failure_provenance(
                    started_at=started_at,
                    runtime_seconds=runtime,
                    timed_out=False,
                    stderr=str(response.get("detail") or "prediction failed"),
                )
                raise ScientificToolExecutionError(
                    f"ADMET-AI prediction failed: {response.get('error_type') or 'unknown'}",
                    provenance=provenance,
                )
            raw = response.get("predictions")
            if not isinstance(raw, dict):
                raise ScientificDataUnavailable("ADMET-AI returned no prediction mapping")
            raw_predictions = {str(key): float(value) for key, value in raw.items()}
            scores, selected, endpoint_mapping = self._map_panel(raw_predictions)
            execution = ADMETExecutionProvenance(
                backend="ADMET-AI 2 / Chemprop ensemble",
                fixture=False,
                model_metadata=dict(self._model_metadata),
                endpoint_mapping=endpoint_mapping,
                timeout_seconds=self.prediction_timeout_seconds,
                started_at=started_at,
                finished_at=_utc_iso(),
                runtime_seconds=round(runtime, 6),
                input_sha256=sha256(normalized.encode("utf-8")).hexdigest(),
                output_sha256=_hash_json(selected),
                cache_hit=False,
            )
            result = ADMETPredictionResult(
                scores=scores,
                raw_selected_predictions=selected,
                execution=execution,
            )
            self._cache[normalized] = result
            self.last_result = result
            return result

    def close(self) -> None:
        with self._lock:
            self._terminate_worker()

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._require_worker_source()
        readiness = self.readiness()
        if readiness["status"] != "completed":
            raise ScientificToolUnavailable(str(readiness.get("reason") or "ADMET-AI unavailable"))
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            self._worker_command("--serve"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            env=dict(self._worker_environment),
            creationflags=creationflags,
        )
        assert self._process.stdout is not None and self._process.stderr is not None
        Thread(target=self._read_stdout, daemon=True).start()
        Thread(target=self._read_stderr, daemon=True).start()
        try:
            message = self._messages.get(timeout=self.startup_timeout_seconds)
        except Empty as exc:
            self._terminate_worker()
            raise ScientificToolUnavailable(
                f"ADMET-AI model initialization exceeded {self.startup_timeout_seconds:g} seconds"
            ) from exc
        if message.get("type") != "ready":
            self._terminate_worker()
            raise ScientificToolUnavailable(
                f"ADMET-AI model initialization failed: {message.get('detail') or message.get('error_type')}"
            )
        metadata = message.get("metadata")
        if not (
            isinstance(metadata, dict)
            and metadata.get("model_bundle_sha256")
            and metadata.get("python_isolated") is True
            and metadata.get("no_user_site") is True
            and metadata.get("worker_source_sha256")
            == self._worker_source_sha256
        ):
            self._terminate_worker()
            raise ScientificToolUnavailable(
                "ADMET-AI worker omitted isolation/model provenance"
            )
        self._model_metadata = dict(metadata)

    def _await_message(
        self,
        *,
        expected_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any] | None:
        deadline = monotonic() + timeout_seconds
        while monotonic() < deadline:
            remaining = max(0.001, deadline - monotonic())
            try:
                message = self._messages.get(timeout=remaining)
            except Empty:
                return None
            if str(message.get("id")) == expected_id:
                return message
        return None

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self._messages.put(message)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_lines.append(line.rstrip())
            del self._stderr_lines[:-100]

    def _terminate_worker(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _failure_provenance(
        self,
        *,
        started_at: str,
        runtime_seconds: float,
        timed_out: bool,
        stderr: str | None = None,
    ) -> ExecutionProvenance:
        stderr_text = stderr or "\n".join(self._stderr_lines[-20:])
        return ExecutionProvenance(
            tool_name="ADMET-AI",
            executable_name=self.python_executable.name,
            executable_version=f"Python {sys.version.split()[0]}",
            sanitized_argv=(
                self.python_executable.name,
                *ISOLATED_PYTHON_FLAGS,
                _ADMET_WORKER_PATH.name,
                "--serve",
            ),
            timeout_seconds=self.prediction_timeout_seconds,
            started_at=started_at,
            finished_at=_utc_iso(),
            runtime_seconds=round(runtime_seconds, 6),
            exit_code=None if timed_out else (self._process.poll() if self._process else None),
            timed_out=timed_out,
            stdout_sha256=sha256(b"").hexdigest(),
            stderr_sha256=sha256(stderr_text.encode("utf-8")).hexdigest(),
            stdout_excerpt="",
            stderr_excerpt=bounded_excerpt(stderr_text),
        )

    @staticmethod
    def _map_panel(
        raw: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        required = [*_CLASSIFICATION_ENDPOINTS.values(), _CLEARANCE_RAW_ENDPOINT, _CLEARANCE_PERCENTILE_ENDPOINT]
        missing = [endpoint for endpoint in required if endpoint not in raw]
        if missing:
            raise ScientificDataUnavailable(
                f"ADMET-AI is missing required endpoints: {', '.join(missing)}"
            )
        scores: dict[str, float] = {}
        endpoint_mapping: dict[str, Any] = {}
        selected: dict[str, float] = {}
        for canonical, endpoint in _CLASSIFICATION_ENDPOINTS.items():
            value = float(raw[endpoint])
            if not 0.0 <= value <= 1.0:
                raise ScientificDataUnavailable(
                    f"ADMET-AI classification endpoint {endpoint} is outside [0, 1]"
                )
            scores[canonical] = value
            selected[endpoint] = value
            endpoint_mapping[canonical] = {
                "source_endpoint": endpoint,
                "task_type": "classification",
                "units": "probability",
                "transformation": "none",
            }

        clearance_raw = float(raw[_CLEARANCE_RAW_ENDPOINT])
        clearance_percentile = float(raw[_CLEARANCE_PERCENTILE_ENDPOINT])
        if not 0.0 <= clearance_percentile <= 100.0:
            raise ScientificDataUnavailable(
                "ADMET-AI DrugBank clearance percentile is outside [0, 100]"
            )
        scores["clearance_risk"] = clearance_percentile / 100.0
        selected[_CLEARANCE_RAW_ENDPOINT] = clearance_raw
        selected[_CLEARANCE_PERCENTILE_ENDPOINT] = clearance_percentile
        endpoint_mapping["clearance_risk"] = {
            "source_endpoint": _CLEARANCE_RAW_ENDPOINT,
            "source_task_type": "regression",
            "source_units": "uL/min/10^6 cells",
            "reference_endpoint": _CLEARANCE_PERCENTILE_ENDPOINT,
            "transformation": "approved-DrugBank percentile divided by 100",
            "semantic": "high predicted hepatocyte clearance relative to approved-drug reference; not a toxicity probability",
        }
        return scores, selected, endpoint_mapping

    def __del__(self) -> None:  # pragma: no cover - best-effort interpreter cleanup
        try:
            self._terminate_worker()
        except Exception:
            pass


__all__ = [
    "ADMETAIAdapter",
    "ADMETExecutionProvenance",
    "ADMETPredictionResult",
]
