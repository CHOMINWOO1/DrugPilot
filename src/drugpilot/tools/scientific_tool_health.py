from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from drugpilot.tools.receptor_qualification import (
    PROFILE_SCHEMA,
    ReceptorToolchain,
    ToolCapability,
)


_PROBE_MARKER = "__DRUGPILOT_SCIENTIFIC_PROBE__"
_VERSION = re.compile(r"(?<![A-Za-z0-9])v?(\d+(?:\.\d+)+(?:[-+._A-Za-z0-9]*))")
_WINDOWS_SHELL_SUFFIXES = {".bat", ".cmd", ".ps1"}
_MAX_PROFILE_REGISTRY_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProbeProcessResult:
    """Minimal process result used by the injectable, shell-free probe runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


ProbeRunner = Callable[
    [Sequence[str], float, Mapping[str, str] | None],
    ProbeProcessResult,
]


@dataclass(frozen=True, slots=True)
class LocalReceptorToolConfiguration:
    """Resolved private inputs consumable by ``LocalReceptorPreparationConfig``.

    This object intentionally retains exact local paths for an offline CLI.  It
    must not be serialized into an API response; ``ScientificToolProbeReport``
    exposes a separate, path-free health projection for that purpose.
    """

    python_executable: Path
    rdkit_root: Path | None
    openmm_plugin_dir: Path | None
    meeko_receptor_executable: Path | None
    meeko_ligand_executable: Path | None
    p2rank_executable: Path | None
    p2rank_prefix_args: tuple[str, ...]
    vina_executable: Path | None


@dataclass(frozen=True, slots=True)
class ScientificToolEntry:
    tool_id: str
    label: str
    operation: str
    group_id: str
    capability: ToolCapability
    configured: bool


@dataclass(frozen=True, slots=True)
class ScientificToolProbeReport:
    entries: tuple[ScientificToolEntry, ...]
    receptor_toolchain: ReceptorToolchain
    local_receptor_configuration: LocalReceptorToolConfiguration
    probed_at: str

    def by_id(self) -> dict[str, ScientificToolEntry]:
        return {item.tool_id: item for item in self.entries}

    def health_payload(
        self,
        *,
        receptor_builder_enabled: bool,
        ma2_scientific_tools_enabled: bool,
        ma2_capabilities: Sequence[Mapping[str, Any]] = (),
        accepted_profile_count: int = 0,
    ) -> dict[str, Any]:
        """Return a bounded public projection with no command output or paths."""
        indexed = self.by_id()
        ma2_index = {
            str(item.get("tool_name") or ""): item
            for item in ma2_capabilities
            if isinstance(item, Mapping)
        }

        def probed_item(
            tool_id: str,
            *,
            mode: str,
            label: str | None = None,
            operation: str | None = None,
        ) -> dict[str, Any]:
            entry = indexed[tool_id]
            available = entry.capability.available
            return {
                "tool_id": tool_id,
                "label": label or entry.label,
                "operation": operation or entry.operation,
                "status": "AVAILABLE" if available else "UNAVAILABLE",
                "available": available,
                "configured": entry.configured,
                "version": entry.capability.version,
                "reason_code": (
                    None
                    if available
                    else entry.capability.reason or "PROBE_FAILED"
                ),
                "mode": mode,
            }

        rdkit = indexed["rdkit_standardization"]
        run_items = [
            probed_item("rdkit_standardization", mode="RUN"),
        ]
        de_novo = ma2_index.get("de_novo_generator")
        de_novo_ready = bool(
            ma2_scientific_tools_enabled
            and de_novo is not None
            and str(de_novo.get("status") or "").casefold() == "ready"
            and rdkit.capability.available
        )
        run_items.append(
            {
                "tool_id": "rdkit_brics",
                "label": "RDKit BRICS",
                "operation": "de novo candidate generation from measured ligand seeds",
                "status": (
                    "AVAILABLE"
                    if de_novo_ready
                    else "NOT_CONFIGURED"
                    if not ma2_scientific_tools_enabled
                    else "UNAVAILABLE"
                ),
                "available": de_novo_ready,
                "configured": ma2_scientific_tools_enabled,
                "version": rdkit.capability.version,
                "reason_code": (
                    None
                    if de_novo_ready
                    else "MA2_SCIENTIFIC_TOOLS_DISABLED"
                    if not ma2_scientific_tools_enabled
                    else str(
                        (de_novo or {}).get("reason_code")
                        or rdkit.capability.reason
                        or "LIVE_DE_NOVO_UNAVAILABLE"
                    )
                ),
                "mode": "RUN",
            }
        )
        run_items.append(
            probed_item(
                "rdkit_standardization",
                mode="ON_DEMAND",
                label="RDKit 2D depiction",
                operation="on-demand 2D molecular structure rendering",
            )
            | {"tool_id": "rdkit_2d_depiction"}
        )
        admet = ma2_index.get("admet_predictor")
        admet_ready = bool(
            ma2_scientific_tools_enabled
            and admet is not None
            and str(admet.get("status") or "").casefold() == "ready"
        )
        run_items.append(
            {
                "tool_id": "admet_ai",
                "label": "ADMET-AI",
                "operation": "versioned six-endpoint ADMET panel",
                "status": (
                    "AVAILABLE"
                    if admet_ready
                    else "NOT_CONFIGURED"
                    if not ma2_scientific_tools_enabled
                    else "UNAVAILABLE"
                ),
                "available": admet_ready,
                "configured": ma2_scientific_tools_enabled,
                "version": (
                    _safe_version((admet or {}).get("backend_version"))
                    if admet is not None
                    else None
                ),
                "reason_code": (
                    None
                    if admet_ready
                    else "MA2_SCIENTIFIC_TOOLS_DISABLED"
                    if not ma2_scientific_tools_enabled
                    else str(
                        (admet or {}).get("reason_code")
                        or "LIVE_ADMET_UNAVAILABLE"
                    )
                ),
                "mode": "RUN",
            }
        )
        vina = probed_item("vina", mode="TARGET_CHECK_AT_RUN")
        if not ma2_scientific_tools_enabled:
            vina.update(
                status="NOT_CONFIGURED",
                available=False,
                reason_code="MA2_SCIENTIFIC_TOOLS_DISABLED",
            )
        elif not indexed["vina"].capability.available:
            vina["status"] = "UNAVAILABLE"
        elif accepted_profile_count < 1:
            vina.update(
                status="PROFILE_REQUIRED",
                available=False,
                reason_code="NO_ACCEPTED_PRODUCTION_PROFILE",
            )
        else:
            vina.update(
                status="TARGET_CHECK_AT_RUN",
                available=True,
                reason_code=None,
            )
        run_items.append(vina)

        builder_items = [
            probed_item("pdbfixer_openmm", mode="OFFLINE"),
            probed_item("meeko_receptor", mode="OFFLINE"),
            probed_item("meeko_ligand", mode="OFFLINE"),
            probed_item("p2rank_java", mode="OFFLINE"),
            probed_item("posebusters", mode="OFFLINE"),
        ]
        public_groups = [
            {
                "group_id": "run_tools",
                "label": "RUN TOOLS",
                "execution_enabled": ma2_scientific_tools_enabled,
                "execution_reason_code": (
                    None
                    if ma2_scientific_tools_enabled
                    else "MA2_SCIENTIFIC_TOOLS_DISABLED"
                ),
                "items": run_items,
            },
            {
                "group_id": "profile_builder",
                "label": "PROFILE BUILDER",
                "execution_enabled": receptor_builder_enabled,
                "execution_reason_code": (
                    None
                    if receptor_builder_enabled
                    else "RECEPTOR_BUILDER_DISABLED"
                ),
                "items": builder_items,
            },
        ]
        status_counts: dict[str, int] = {}
        for group in public_groups:
            for item in group["items"]:
                status = str(item["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "schema_version": "1.0",
            "probe_mode": "external_process_exact_executable",
            "probed_at": self.probed_at,
            "raw_output_exposed": False,
            "local_paths_exposed": False,
            "summary": status_counts,
            "groups": public_groups,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_runner(
    command: Sequence[str],
    timeout_seconds: float,
    environment: Mapping[str, str] | None,
) -> ProbeProcessResult:
    process_environment = os.environ.copy()
    if environment:
        process_environment.update(environment)
    completed = subprocess.run(
        [str(value) for value in command],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
        env=process_environment,
    )
    return ProbeProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _resolved_executable(
    explicit: str | Path | None,
    command_names: Sequence[str],
) -> tuple[Path | None, bool]:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        return (candidate.resolve(), True) if candidate.is_file() else (None, True)
    for command in command_names:
        found = shutil.which(command)
        if found:
            return Path(found).resolve(), False
    return None, False


def _resolved_directory(explicit: str | Path | None) -> tuple[Path | None, bool]:
    if explicit is None:
        return None, False
    candidate = Path(explicit).expanduser()
    return (candidate.resolve(), True) if candidate.is_dir() else (None, True)


def _safe_version(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or len(text) > 120:
        return None
    # Version fields are deliberately much stricter than generic bounded
    # output: path separators, quotes and control characters are rejected.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+;:=()\-]{0,119}", text):
        return None
    return text


def _version_from_output(value: str) -> str | None:
    matched = _VERSION.search(value[:2_000])
    return _safe_version(matched.group(1)) if matched else None


def _unavailable(
    tool_id: str,
    reason_code: str,
    *,
    executable: Path | None = None,
) -> ToolCapability:
    return ToolCapability(
        tool_id=tool_id,
        available=False,
        version=None,
        executable_path=str(executable) if executable else None,
        reason=reason_code,
    )


def _run(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    runner: ProbeRunner,
    environment: Mapping[str, str] | None = None,
) -> tuple[ProbeProcessResult | None, str | None]:
    try:
        return (
            runner(
                tuple(str(value) for value in command),
                timeout_seconds,
                environment,
            ),
            None,
        )
    except subprocess.TimeoutExpired:
        return None, "PROBE_TIMEOUT"
    except (OSError, subprocess.SubprocessError):
        return None, "PROBE_EXECUTION_FAILED"
    except Exception:
        # A custom runner is supported for offline verification/tests, but its
        # exception text is never propagated into public health.
        return None, "PROBE_EXECUTION_FAILED"


def _python_json_probe(
    tool_id: str,
    python_executable: Path | None,
    script: str,
    *,
    timeout_seconds: float,
    runner: ProbeRunner,
    environment: Mapping[str, str] | None = None,
) -> tuple[ToolCapability, Mapping[str, Any]]:
    if python_executable is None:
        return _unavailable(tool_id, "PYTHON_INTERPRETER_NOT_FOUND"), {}
    result, error = _run(
        (str(python_executable), "-c", script),
        timeout_seconds=timeout_seconds,
        runner=runner,
        environment=environment,
    )
    if error:
        return _unavailable(tool_id, error, executable=python_executable), {}
    assert result is not None
    if result.returncode != 0:
        return (
            _unavailable(
                tool_id,
                "PROBE_NONZERO_EXIT",
                executable=python_executable,
            ),
            {},
        )
    payload: Mapping[str, Any] = {}
    for line in result.stdout.splitlines():
        if not line.startswith(_PROBE_MARKER):
            continue
        try:
            parsed = json.loads(line[len(_PROBE_MARKER) :])
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
            break
    if payload.get("ok") is not True:
        return (
            _unavailable(
                tool_id,
                "PROBE_CONTRACT_FAILED",
                executable=python_executable,
            ),
            {},
        )
    version = _safe_version(payload.get("version"))
    return (
        ToolCapability(
            tool_id=tool_id,
            available=True,
            version=version,
            executable_path=str(python_executable),
        ),
        payload,
    )


def _cli_probe(
    tool_id: str,
    executable: Path | None,
    args: Sequence[str],
    *,
    timeout_seconds: float,
    runner: ProbeRunner,
    known_version: str | None = None,
    require_version: bool = False,
    environment: Mapping[str, str] | None = None,
) -> ToolCapability:
    if executable is None:
        return _unavailable(tool_id, "EXECUTABLE_NOT_FOUND")
    if executable.suffix.casefold() in _WINDOWS_SHELL_SUFFIXES:
        return _unavailable(
            tool_id,
            "WINDOWS_SHELL_WRAPPER_UNSUPPORTED",
            executable=executable,
        )
    result, error = _run(
        (str(executable), *args),
        timeout_seconds=timeout_seconds,
        runner=runner,
        environment=environment,
    )
    if error:
        return _unavailable(tool_id, error, executable=executable)
    assert result is not None
    if result.returncode != 0:
        return _unavailable(tool_id, "PROBE_NONZERO_EXIT", executable=executable)
    version = _safe_version(known_version) or _version_from_output(
        f"{result.stdout}\n{result.stderr}"
    )
    if require_version and version is None:
        return _unavailable(
            tool_id,
            "VERSION_OUTPUT_UNRECOGNIZED",
            executable=executable,
        )
    return ToolCapability(
        tool_id=tool_id,
        available=True,
        version=version,
        executable_path=str(executable),
    )


def _combined_capability(
    tool_id: str,
    capabilities: Sequence[ToolCapability],
    *,
    executable: Path | None,
) -> ToolCapability:
    failed = next((item for item in capabilities if not item.available), None)
    if failed is not None:
        return _unavailable(
            tool_id,
            failed.reason or "PROBE_FAILED",
            executable=executable,
        )
    versions = [item.version for item in capabilities if item.version]
    return ToolCapability(
        tool_id=tool_id,
        available=True,
        version=_safe_version("; ".join(dict.fromkeys(versions))) if versions else None,
        executable_path=str(executable) if executable else None,
    )


_RDKIT_PROBE = f"""
import json
from importlib.metadata import version
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
mol = Chem.MolFromSmiles('[Na+].[O-]C(=O)c1ccccc1')
assert mol is not None
clean = rdMolStandardize.Cleanup(mol)
parent = rdMolStandardize.FragmentParent(clean)
parent = rdMolStandardize.Uncharger().uncharge(parent)
tautomer = rdMolStandardize.TautomerParent(parent)
assert parent.GetNumHeavyAtoms() > 0 and tautomer.GetNumHeavyAtoms() > 0
print('{_PROBE_MARKER}' + json.dumps({{'ok': True, 'version': version('rdkit')}}))
"""


_PDBFIXER_OPENMM_PROBE = f"""
import json
from importlib.metadata import version
from pdbfixer import PDBFixer
import openmm
assert callable(PDBFixer)
assert openmm.System().getNumParticles() == 0
openmm_version = getattr(getattr(openmm, 'version', None), 'full_version', None) or version('openmm')
print('{_PROBE_MARKER}' + json.dumps({{
    'ok': True,
    'version': 'pdbfixer=' + version('pdbfixer') + '; openmm=' + str(openmm_version),
}}))
"""


_MEEKO_PROBE = f"""
import json
from importlib.metadata import version
import meeko
assert meeko is not None
print('{_PROBE_MARKER}' + json.dumps({{'ok': True, 'version': version('meeko')}}))
"""


_POSEBUSTERS_PROBE = f"""
import json
from importlib.metadata import version
from posebusters import PoseBusters
assert PoseBusters is not None
print('{_PROBE_MARKER}' + json.dumps({{'ok': True, 'version': version('posebusters')}}))
"""


def probe_scientific_tools(
    *,
    python_executable: str | Path | None = None,
    receptor_python_executable: str | Path | None = None,
    rdkit_root: str | Path | None = None,
    openmm_plugin_dir: str | Path | None = None,
    meeko_receptor_executable: str | Path | None = None,
    meeko_ligand_executable: str | Path | None = None,
    p2rank_executable: str | Path | None = None,
    p2rank_jar_path: str | Path | None = None,
    java_executable: str | Path | None = None,
    vina_executable: str | Path | None = None,
    timeout_seconds: float = 10.0,
    runner: ProbeRunner | None = None,
) -> ScientificToolProbeReport:
    """Probe the exact configured interpreters/executables without a shell.

    A P2Rank JAR plus its sibling ``lib`` directory and Java is the preferred
    Windows configuration. Native executable discovery is retained for
    Unix-like deployments, while shell wrappers are rejected rather than
    silently invoking a command processor.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    run = runner or _default_runner
    service_python, service_python_explicit = _resolved_executable(
        python_executable or sys.executable,
        (),
    )
    receptor_python, receptor_python_explicit = _resolved_executable(
        receptor_python_executable or service_python,
        (),
    )
    resolved_rdkit_root, rdkit_root_explicit = _resolved_directory(rdkit_root)
    resolved_openmm_plugins, openmm_plugins_explicit = _resolved_directory(
        openmm_plugin_dir
    )
    worker_environment: dict[str, str] = {}
    if resolved_rdkit_root is not None:
        worker_environment["RDBASE"] = os.path.relpath(
            resolved_rdkit_root,
            start=Path.cwd().resolve(),
        )
    if resolved_openmm_plugins is not None:
        worker_environment["OPENMM_PLUGIN_DIR"] = os.path.relpath(
            resolved_openmm_plugins,
            start=Path.cwd().resolve(),
        )
    meeko_receptor, meeko_receptor_explicit = _resolved_executable(
        meeko_receptor_executable,
        ("mk_prepare_receptor", "mk_prepare_receptor.exe"),
    )
    meeko_ligand, meeko_ligand_explicit = _resolved_executable(
        meeko_ligand_executable,
        ("mk_prepare_ligand", "mk_prepare_ligand.exe"),
    )
    vina, vina_explicit = _resolved_executable(
        vina_executable,
        ("vina", "vina.exe"),
    )
    java, java_explicit = _resolved_executable(java_executable, ("java", "java.exe"))

    rdkit, _ = _python_json_probe(
        "rdkit_standardization",
        service_python,
        _RDKIT_PROBE,
        timeout_seconds=timeout_seconds,
        runner=run,
        environment=worker_environment,
    )
    pdbfixer_openmm, _ = _python_json_probe(
        "pdbfixer",
        receptor_python,
        _PDBFIXER_OPENMM_PROBE,
        timeout_seconds=timeout_seconds,
        runner=run,
        environment=worker_environment,
    )
    meeko_module, meeko_payload = _python_json_probe(
        "meeko_module",
        receptor_python,
        _MEEKO_PROBE,
        timeout_seconds=timeout_seconds,
        runner=run,
        environment=worker_environment,
    )
    meeko_version = _safe_version(meeko_payload.get("version"))
    meeko_receptor_probe = _cli_probe(
        "meeko_receptor",
        meeko_receptor,
        ("--help",),
        timeout_seconds=timeout_seconds,
        runner=run,
        known_version=meeko_version,
        environment=worker_environment,
    )
    meeko_ligand_probe = _cli_probe(
        "meeko_ligand",
        meeko_ligand,
        ("--help",),
        timeout_seconds=timeout_seconds,
        runner=run,
        known_version=meeko_version,
        environment=worker_environment,
    )
    posebusters, _ = _python_json_probe(
        "posebusters",
        receptor_python,
        _POSEBUSTERS_PROBE,
        timeout_seconds=timeout_seconds,
        runner=run,
        environment=worker_environment,
    )
    java_probe = _cli_probe(
        "java",
        java,
        ("-version",),
        timeout_seconds=timeout_seconds,
        runner=run,
        require_version=True,
    )

    p2rank_command: Path | None = None
    p2rank_prefix_args: tuple[str, ...] = ()
    jar_explicit = p2rank_jar_path is not None
    jar = Path(p2rank_jar_path).expanduser() if p2rank_jar_path is not None else None
    jar = jar.resolve() if jar is not None and jar.is_file() else None
    if jar_explicit and jar is None:
        p2rank_probe = _unavailable("p2rank", "P2RANK_JAR_NOT_FOUND")
    elif jar is not None:
        p2rank_command = java
        # The official P2Rank distribution JAR is not a fat/executable JAR.
        # Reproduce the vendor launcher's classpath without invoking its shell
        # wrapper. Java expands the trailing classpath wildcard itself.
        classpath = os.pathsep.join((str(jar), str(jar.parent / "lib" / "*")))
        p2rank_prefix_args = (
            "-cp",
            classpath,
            "cz.siret.prank.program.Main",
        )
        if not java_probe.available:
            p2rank_probe = _unavailable(
                "p2rank",
                java_probe.reason or "JAVA_UNAVAILABLE",
                executable=java,
            )
        else:
            p2rank_probe = _cli_probe(
                "p2rank",
                java,
                (*p2rank_prefix_args, "-v"),
                timeout_seconds=timeout_seconds,
                runner=run,
                require_version=True,
            )
    else:
        native_p2rank, _ = _resolved_executable(
            p2rank_executable,
            ("prank", "p2rank"),
        )
        p2rank_command = native_p2rank
        p2rank_probe = _cli_probe(
            "p2rank",
            native_p2rank,
            ("-v",),
            timeout_seconds=timeout_seconds,
            runner=run,
            require_version=True,
        )
    p2rank_java = _combined_capability(
        "p2rank_java",
        (java_probe, p2rank_probe),
        executable=p2rank_command,
    )
    meeko = _combined_capability(
        "meeko",
        (meeko_module, meeko_receptor_probe, meeko_ligand_probe),
        executable=meeko_receptor,
    )
    vina_probe = _cli_probe(
        "vina",
        vina,
        ("--version",),
        timeout_seconds=timeout_seconds,
        runner=run,
        require_version=True,
        environment=worker_environment,
    )

    if rdkit_root_explicit and resolved_rdkit_root is None:
        invalid_root = "RDKIT_ROOT_DIRECTORY_UNAVAILABLE"
        rdkit = _unavailable("rdkit_standardization", invalid_root)
        meeko_module = _unavailable("meeko_module", invalid_root)
        meeko_receptor_probe = _unavailable("meeko_receptor", invalid_root)
        meeko_ligand_probe = _unavailable("meeko_ligand", invalid_root)
        posebusters = _unavailable("posebusters", invalid_root)
    if openmm_plugins_explicit and resolved_openmm_plugins is None:
        pdbfixer_openmm = _unavailable(
            "pdbfixer",
            "OPENMM_PLUGIN_DIRECTORY_UNAVAILABLE",
        )

    # Rebuild aggregates after enforcing explicit runtime-directory failures.
    meeko = _combined_capability(
        "meeko",
        (meeko_module, meeko_receptor_probe, meeko_ligand_probe),
        executable=meeko_receptor,
    )

    receptor_toolchain = ReceptorToolchain(
        capabilities=(
            ToolCapability(
                tool_id="pdbfixer",
                available=pdbfixer_openmm.available,
                version=pdbfixer_openmm.version,
                executable_path=str(receptor_python) if receptor_python else None,
                reason=pdbfixer_openmm.reason,
            ),
            meeko,
            ToolCapability(
                tool_id="p2rank",
                available=p2rank_java.available,
                version=p2rank_java.version,
                executable_path=(
                    str(p2rank_command) if p2rank_command is not None else None
                ),
                reason=p2rank_java.reason,
            ),
            ToolCapability(
                tool_id="posebusters",
                available=posebusters.available,
                version=posebusters.version,
                executable_path=str(receptor_python) if receptor_python else None,
                reason=posebusters.reason,
            ),
            vina_probe,
        ),
        probed_at=_now_iso(),
    )
    entries = (
        ScientificToolEntry(
            "rdkit_standardization",
            "RDKit standardization",
            "cleanup, parent selection, charge normalization and tautomer parent",
            "chemical_standardization",
            rdkit,
            service_python is not None,
        ),
        ScientificToolEntry(
            "pdbfixer_openmm",
            "PDBFixer / OpenMM",
            "receptor repair and hydrogen placement",
            "receptor_preparation",
            pdbfixer_openmm,
            receptor_python is not None,
        ),
        ScientificToolEntry(
            "meeko_receptor",
            "Meeko receptor",
            "receptor PDBQT preparation",
            "receptor_preparation",
            _combined_capability(
                "meeko_receptor",
                (meeko_module, meeko_receptor_probe),
                executable=meeko_receptor,
            ),
            meeko_receptor is not None,
        ),
        ScientificToolEntry(
            "meeko_ligand",
            "Meeko ligand",
            "validation and candidate ligand PDBQT preparation",
            "receptor_preparation",
            _combined_capability(
                "meeko_ligand",
                (meeko_module, meeko_ligand_probe),
                executable=meeko_ligand,
            ),
            meeko_ligand is not None,
        ),
        ScientificToolEntry(
            "p2rank_java",
            "P2Rank / Java",
            "predicted pocket detection",
            "receptor_preparation",
            p2rank_java,
            bool(
                p2rank_command
                and (jar is not None or p2rank_executable is not None)
                and java is not None
            ),
        ),
        ScientificToolEntry(
            "posebusters",
            "PoseBusters",
            "docked-pose plausibility validation",
            "receptor_preparation",
            posebusters,
            receptor_python is not None,
        ),
        ScientificToolEntry(
            "vina",
            "AutoDock Vina",
            "redocking and optional candidate docking",
            "molecular_docking",
            vina_probe,
            vina is not None,
        ),
    )
    return ScientificToolProbeReport(
        entries=entries,
        receptor_toolchain=receptor_toolchain,
        local_receptor_configuration=LocalReceptorToolConfiguration(
            python_executable=receptor_python or Path(sys.executable).resolve(),
            rdkit_root=resolved_rdkit_root,
            openmm_plugin_dir=resolved_openmm_plugins,
            meeko_receptor_executable=meeko_receptor,
            meeko_ligand_executable=meeko_ligand,
            p2rank_executable=p2rank_command,
            p2rank_prefix_args=p2rank_prefix_args,
            vina_executable=vina,
        ),
        probed_at=receptor_toolchain.probed_at,
    )


def probe_receptor_toolchain(**kwargs: Any) -> ReceptorToolchain:
    """Reusable offline-builder entry point for the exact same live probes."""

    return probe_scientific_tools(**kwargs).receptor_toolchain


def _complete_acceptance(profile: Mapping[str, Any]) -> bool:
    acceptance = profile.get("production_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("accepted") is not True:
        return False
    return all(
        isinstance(acceptance.get(field), str) and acceptance[field].strip()
        for field in ("reviewer", "accepted_at", "protocol_version", "benchmark_scope")
    )


def _accepted_profile(profile: Mapping[str, Any]) -> bool:
    if profile.get("production_ready") is not True:
        return False
    if str(profile.get("profile_status") or "") != "production_ready":
        return False
    if profile.get("fixture") is True or not _complete_acceptance(profile):
        return False
    # Candidate profiles emitted by the new builder carry the v2 schema.  A
    # registry wrapper may additionally carry schema_version=1.0, but a legacy
    # manually toggled production_ready flag is never sufficient.
    schema = profile.get("schema")
    if schema == PROFILE_SCHEMA:
        validations = profile.get("validations")
        pose = validations.get("pose") if isinstance(validations, Mapping) else None
        return bool(
            isinstance(pose, Mapping)
            and pose.get("passed") is True
            and not profile.get("production_readiness_reasons")
        )
    return schema is None and profile.get("validated") is True


def receptor_profile_health(
    registry_path: str | Path | None,
    report: ScientificToolProbeReport,
    *,
    receptor_builder_enabled: bool,
    ma2_scientific_tools_enabled: bool,
) -> dict[str, Any]:
    """Summarize profile acceptance without returning IDs, files or paths."""

    reason_codes: list[str] = []
    profiles: list[Mapping[str, Any]] = []
    configured = registry_path is not None
    if registry_path is None:
        registry_status = "not_configured"
        reason_codes.append("RECEPTOR_PROFILE_REGISTRY_NOT_CONFIGURED")
    else:
        path = Path(registry_path).expanduser()
        if not path.is_file():
            registry_status = "missing"
            reason_codes.append("RECEPTOR_PROFILE_REGISTRY_MISSING")
        elif path.stat().st_size > _MAX_PROFILE_REGISTRY_BYTES:
            registry_status = "invalid"
            reason_codes.append("RECEPTOR_PROFILE_REGISTRY_TOO_LARGE")
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                raw_profiles = (
                    payload.get("profiles")
                    if isinstance(payload, Mapping)
                    else None
                )
                if isinstance(raw_profiles, list):
                    profiles = [item for item in raw_profiles if isinstance(item, Mapping)]
                    valid_shape = len(profiles) == len(raw_profiles) and bool(profiles)
                elif isinstance(payload, Mapping) and payload.get("schema") == PROFILE_SCHEMA:
                    profiles = [payload]
                    valid_shape = True
                else:
                    valid_shape = False
                if not valid_shape:
                    raise ValueError("unsupported profile registry shape")
                registry_status = "loaded"
            except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
                registry_status = "invalid"
                profiles = []
                reason_codes.append("RECEPTOR_PROFILE_REGISTRY_INVALID")

    accepted_count = sum(1 for item in profiles if _accepted_profile(item))
    experimental_count = len(profiles) - accepted_count
    tools = report.receptor_toolchain.by_id()
    full_builder_tool_ids = (*ReceptorToolchain.REQUIRED_TOOLS, "vina")
    builder_tools_ready = all(
        tools.get(tool_id) is not None and tools[tool_id].available
        for tool_id in full_builder_tool_ids
    )
    execution_tools_ready = all(
        tools.get(tool_id) is not None and tools[tool_id].available
        for tool_id in ("meeko", "vina")
    )
    builder_ready = receptor_builder_enabled and builder_tools_ready
    production_ready = bool(
        ma2_scientific_tools_enabled
        and accepted_count > 0
        and execution_tools_ready
    )
    if accepted_count == 0:
        reason_codes.append("NO_ACCEPTED_PRODUCTION_PROFILE")
    if not receptor_builder_enabled:
        reason_codes.append("RECEPTOR_BUILDER_DISABLED")
    elif not builder_tools_ready:
        reason_codes.append("RECEPTOR_BUILDER_TOOLCHAIN_UNAVAILABLE")
    if not ma2_scientific_tools_enabled:
        reason_codes.append("MA2_SCIENTIFIC_TOOLS_DISABLED")
    elif not execution_tools_ready:
        reason_codes.append("DOCKING_EXECUTION_TOOLCHAIN_UNAVAILABLE")
    return {
        "configured": configured,
        "registry_status": registry_status,
        "profile_count": len(profiles),
        "accepted_profile_count": accepted_count,
        "experimental_profile_count": experimental_count,
        "builder_enabled": receptor_builder_enabled,
        "builder_ready": builder_ready,
        "production_ready": production_ready,
        "acceptance_required": True,
        "raw_profile_content_exposed": False,
        "local_paths_exposed": False,
        "reason_codes": sorted(set(reason_codes)),
    }


__all__ = [
    "LocalReceptorToolConfiguration",
    "ProbeProcessResult",
    "ScientificToolEntry",
    "ScientificToolProbeReport",
    "probe_receptor_toolchain",
    "probe_scientific_tools",
    "receptor_profile_health",
]
