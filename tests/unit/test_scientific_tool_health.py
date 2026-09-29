from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from drugpilot.tools.scientific_tool_health import (
    ProbeProcessResult,
    ScientificToolProbeReport,
    probe_scientific_tools,
    receptor_profile_health,
)


MARKER = "__DRUGPILOT_SCIENTIFIC_PROBE__"


class FakeProbeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Mapping[str, str]]] = []

    def __call__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
        environment: Mapping[str, str] | None,
    ) -> ProbeProcessResult:
        values = tuple(str(value) for value in command)
        self.calls.append((values, dict(environment or {})))
        assert timeout_seconds == 3.0
        if "-c" in values:
            script = values[-1]
            if "MolStandardize" in script:
                version = "2026.3.5"
            elif "PDBFixer" in script:
                version = "pdbfixer=1.12.0; openmm=8.5.2"
            elif "import meeko" in script:
                version = "0.7.1"
            elif "PoseBusters" in script:
                version = "0.6.5"
            else:  # pragma: no cover - proves every Python probe is recognized
                raise AssertionError(script)
            return ProbeProcessResult(
                0,
                f'{MARKER}{json.dumps({"ok": True, "version": version})}\n',
            )
        if values[-1] == "-version":
            return ProbeProcessResult(0, stderr='openjdk version "21.0.12"')
        if "cz.siret.prank.program.Main" in values:
            assert values[-1] == "-v"
            return ProbeProcessResult(0, stdout="P2Rank 2.5.1")
        if values[-1] == "--version":
            return ProbeProcessResult(0, stdout="AutoDock Vina v1.2.7")
        if values[-1] == "--help":
            return ProbeProcessResult(0, stdout="usage")
        raise AssertionError(values)


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"probe")
    return path


def _ready_report(tmp_path: Path) -> tuple[ScientificToolProbeReport, FakeProbeRunner]:
    python = _file(tmp_path / "python.exe")
    receptor = _file(tmp_path / "mk_prepare_receptor.exe")
    ligand = _file(tmp_path / "mk_prepare_ligand.exe")
    java = _file(tmp_path / "java.exe")
    jar = _file(tmp_path / "p2rank" / "bin" / "p2rank.jar")
    (jar.parent / "lib").mkdir()
    vina = _file(tmp_path / "vina.exe")
    rdkit_root = tmp_path / "runtime" / "rdkit"
    openmm_plugins = tmp_path / "runtime" / "openmm_plugins"
    rdkit_root.mkdir(parents=True)
    openmm_plugins.mkdir(parents=True)
    runner = FakeProbeRunner()
    report = probe_scientific_tools(
        python_executable=python,
        receptor_python_executable=python,
        rdkit_root=rdkit_root,
        openmm_plugin_dir=openmm_plugins,
        meeko_receptor_executable=receptor,
        meeko_ligand_executable=ligand,
        p2rank_jar_path=jar,
        java_executable=java,
        vina_executable=vina,
        timeout_seconds=3.0,
        runner=runner,
    )
    return report, runner


def test_exact_toolchain_probe_is_reusable_and_public_projection_has_no_paths(
    tmp_path: Path,
) -> None:
    report, runner = _ready_report(tmp_path)

    report.receptor_toolchain.require_all()
    assert report.receptor_toolchain.by_id()["vina"].available is True
    private = report.local_receptor_configuration
    assert private.p2rank_executable == (tmp_path / "java.exe").resolve()
    assert private.p2rank_prefix_args[0] == "-cp"
    assert "cz.siret.prank.program.Main" in private.p2rank_prefix_args
    assert private.rdkit_root == (tmp_path / "runtime" / "rdkit").resolve()
    assert private.openmm_plugin_dir == (
        tmp_path / "runtime" / "openmm_plugins"
    ).resolve()

    java_calls = [call for call, _ in runner.calls if call[0].endswith("java.exe")]
    assert any(call[-1] == "-version" for call in java_calls)
    assert any("cz.siret.prank.program.Main" in call for call in java_calls)
    python_environments = [env for call, env in runner.calls if "-c" in call]
    assert python_environments
    assert all("RDBASE" in env and "OPENMM_PLUGIN_DIR" in env for env in python_environments)
    assert all(not Path(env["RDBASE"]).is_absolute() for env in python_environments)

    public = report.health_payload(
        receptor_builder_enabled=True,
        ma2_scientific_tools_enabled=False,
    )
    serialized = json.dumps(public)
    assert str(tmp_path) not in serialized
    assert public["raw_output_exposed"] is False
    assert public["local_paths_exposed"] is False
    assert [group["label"] for group in public["groups"]] == [
        "RUN TOOLS",
        "PROFILE BUILDER",
    ]
    builder = public["groups"][1]
    assert all(item["status"] == "AVAILABLE" for item in builder["items"])


def test_public_run_tools_include_runtime_utility_and_profile_required_vina(
    tmp_path: Path,
) -> None:
    report, _ = _ready_report(tmp_path)
    payload = report.health_payload(
        receptor_builder_enabled=False,
        ma2_scientific_tools_enabled=True,
        ma2_capabilities=(
            {
                "tool_name": "de_novo_generator",
                "status": "ready",
                "backend_version": "rdkit-2026.3.5",
            },
            {
                "tool_name": "admet_predictor",
                "status": "ready",
                "backend_version": "2.0.0",
            },
        ),
        accepted_profile_count=0,
    )
    items = {item["tool_id"]: item for item in payload["groups"][0]["items"]}

    assert set(items) == {
        "rdkit_standardization",
        "rdkit_brics",
        "rdkit_2d_depiction",
        "admet_ai",
        "vina",
    }
    assert items["rdkit_brics"]["status"] == "AVAILABLE"
    assert items["rdkit_2d_depiction"]["mode"] == "ON_DEMAND"
    assert items["admet_ai"]["version"] == "2.0.0"
    assert items["vina"]["status"] == "PROFILE_REQUIRED"
    assert items["vina"]["available"] is False


def test_p2rank_windows_shell_wrapper_is_rejected_without_execution(
    tmp_path: Path,
) -> None:
    python = _file(tmp_path / "python.exe")
    java = _file(tmp_path / "java.exe")
    batch = _file(tmp_path / "prank.bat")
    runner = FakeProbeRunner()
    report = probe_scientific_tools(
        python_executable=python,
        receptor_python_executable=python,
        p2rank_executable=batch,
        java_executable=java,
        timeout_seconds=3.0,
        runner=runner,
    )

    p2rank = report.receptor_toolchain.by_id()["p2rank"]
    assert p2rank.available is False
    assert p2rank.reason == "WINDOWS_SHELL_WRAPPER_UNSUPPORTED"
    assert not any(call[0].endswith("prank.bat") for call, _ in runner.calls)


def test_invalid_explicit_runtime_directory_fails_closed(tmp_path: Path) -> None:
    report, _ = _ready_report(tmp_path)
    private = report.local_receptor_configuration
    rerun = probe_scientific_tools(
        python_executable=private.python_executable,
        receptor_python_executable=private.python_executable,
        rdkit_root=tmp_path / "missing-rdkit",
        openmm_plugin_dir=private.openmm_plugin_dir,
        meeko_receptor_executable=private.meeko_receptor_executable,
        meeko_ligand_executable=private.meeko_ligand_executable,
        p2rank_executable=private.p2rank_executable,
        vina_executable=private.vina_executable,
        timeout_seconds=3.0,
        runner=FakeProbeRunner(),
    )

    items = rerun.by_id()
    assert items["rdkit_standardization"].capability.reason == (
        "RDKIT_ROOT_DIRECTORY_UNAVAILABLE"
    )
    assert items["posebusters"].capability.reason == (
        "RDKIT_ROOT_DIRECTORY_UNAVAILABLE"
    )


def test_profile_summary_requires_complete_acceptance_and_ready_execution_tools(
    tmp_path: Path,
) -> None:
    report, _ = _ready_report(tmp_path)
    registry = tmp_path / "profiles.json"
    base_profile = {
        "profile_id": "IDH1-test",
        "validated": True,
        "profile_status": "production_ready",
        "production_ready": True,
        "fixture": False,
    }
    registry.write_text(
        json.dumps({"schema_version": "1.0", "profiles": [base_profile]}),
        encoding="utf-8",
    )
    missing_acceptance = receptor_profile_health(
        registry,
        report,
        receptor_builder_enabled=True,
        ma2_scientific_tools_enabled=True,
    )
    assert missing_acceptance["accepted_profile_count"] == 0
    assert missing_acceptance["production_ready"] is False

    base_profile["production_acceptance"] = {
        "accepted": True,
        "reviewer": "scientist-01",
        "accepted_at": "2026-08-21T00:00:00Z",
        "protocol_version": "1.0",
        "benchmark_scope": "IDH1 R132H external redocking benchmark",
    }
    registry.write_text(
        json.dumps({"schema_version": "1.0", "profiles": [base_profile]}),
        encoding="utf-8",
    )
    accepted = receptor_profile_health(
        registry,
        report,
        receptor_builder_enabled=True,
        ma2_scientific_tools_enabled=True,
    )
    assert accepted["accepted_profile_count"] == 1
    assert accepted["production_ready"] is True
    assert accepted["local_paths_exposed"] is False

