from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

import pytest

from drugpilot.tools.executable import (
    ExecutableResult,
    ExecutionProvenance,
    ScientificToolExecutionError,
)
from drugpilot.tools.local_receptor_preparation import (
    LocalPreparationRunners,
    LocalReceptorPreparationBackend,
    LocalReceptorPreparationConfig,
)
from drugpilot.tools.receptor_qualification import (
    CandidateDockingProfileBuilder,
    ComponentRole,
    FallbackReason,
    PocketOrigin,
    ReceptorPipelineError,
    ReceptorQualifier,
    ReceptorToolchain,
    StructureSource,
    TargetStructureRequest,
    ToolCapability,
    structure_candidate_from_dict,
)


ROOT = Path(__file__).resolve().parents[2]
FLT3_FIXTURE = ROOT / "tests" / "fixtures" / "receptor_pipeline" / "flt3_structure_contract.json"


def _provenance(tool: str, args: Sequence[str]) -> ExecutionProvenance:
    now = datetime.now(timezone.utc).isoformat()
    return ExecutionProvenance(
        tool_name=tool,
        executable_name=f"{tool}.fake",
        executable_version="contract-test",
        sanitized_argv=tuple(args),
        timeout_seconds=10.0,
        started_at=now,
        finished_at=now,
        runtime_seconds=0.001,
        exit_code=0,
        timed_out=False,
        stdout_sha256="0" * 64,
        stderr_sha256="0" * 64,
        stdout_excerpt="",
        stderr_excerpt="",
    )


def _arg(args: Sequence[str], name: str, cwd: str | Path | None = None) -> Path:
    path = Path(args[args.index(name) + 1])
    if not path.is_absolute() and cwd is not None:
        path = Path(cwd) / path
    return path


@dataclass
class FakeRunner:
    tool: str
    calls: list[tuple[str, ...]]
    environments: list[dict[str, str] | None] | None = None

    def run(self, args, *, timeout_seconds, cwd=None, env=None):
        values = tuple(str(item) for item in args)
        self.calls.append(values)
        if self.environments is not None:
            self.environments.append(dict(env) if env is not None else None)
        if self.tool == "python" and "drugpilot.tools.pdbfixer_receptor_worker" in values:
            source = _arg(values, "--input", cwd)
            _arg(values, "--output", cwd).write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N\nEND\n",
                encoding="utf-8",
            )
            _arg(values, "--report", cwd).write_text(
                json.dumps({"source": source.name, "missing_atoms_added": True}),
                encoding="utf-8",
            )
        elif self.tool == "python" and "drugpilot.tools.posebusters_validation_worker" in values:
            _arg(values, "--output-sdf", cwd).write_text("top pose\n$$$$\n", encoding="utf-8")
            _arg(values, "--posebusters-report", cwd).write_text(
                json.dumps(
                    {
                        "passed": True,
                        "checked_pose_count": 1,
                        "failure_codes": [],
                    }
                ),
                encoding="utf-8",
            )
            if "--redocking-report" in values:
                _arg(values, "--posebusters-reference-output", cwd).write_text(
                    "crystal reference\n$$$$\n",
                    encoding="utf-8",
                )
                _arg(values, "--redocking-report", cwd).write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "threshold_angstrom": 2.0,
                            "top_ranked_pose_rmsd_angstrom": 1.1,
                            "mapping_method": "contract RMSD",
                        }
                    ),
                    encoding="utf-8",
                )
        elif self.tool == "meeko_receptor":
            _arg(values, "--write_pdbqt", cwd).write_text("ATOM receptor\n", encoding="utf-8")
            _arg(values, "--write_json", cwd).write_text("{}\n", encoding="utf-8")
            _arg(values, "--write_vina_box", cwd).write_text("box\n", encoding="utf-8")
        elif self.tool == "meeko_ligand":
            _arg(values, "-o", cwd).write_text("ATOM ligand\n", encoding="utf-8")
        elif self.tool == "vina":
            _arg(values, "--out", cwd).write_text(
                "MODEL 1\nREMARK VINA RESULT: -8.0 0 0\nATOM ligand\nENDMDL\n",
                encoding="utf-8",
            )
        elif self.tool == "p2rank":
            output = _arg(values, "-o", cwd)
            output.mkdir(parents=True, exist_ok=True)
            (output / "receptor_fixed.pdb_predictions.csv").write_text(
                "rank,score,probability,center_x,center_y,center_z\n"
                "1,12.5,0.91,11.0,12.0,13.0\n",
                encoding="utf-8",
            )
        return ExecutableResult(
            stdout="{}",
            stderr="",
            provenance=_provenance(self.tool, values),
        )


@dataclass
class FailingRunner:
    tool: str
    calls: list[tuple[str, ...]]

    def run(self, args, *, timeout_seconds, cwd=None, env=None):
        values = tuple(str(item) for item in args)
        self.calls.append(values)
        provenance = _provenance(self.tool, values)
        raise ScientificToolExecutionError(
            f"{self.tool} contract failure",
            provenance=provenance,
        )


def _runners() -> tuple[LocalPreparationRunners, list[tuple[str, ...]]]:
    calls: list[tuple[str, ...]] = []
    return (
        LocalPreparationRunners(
            python=FakeRunner("python", calls),
            meeko_receptor=FakeRunner("meeko_receptor", calls),
            meeko_ligand=FakeRunner("meeko_ligand", calls),
            p2rank=FakeRunner("p2rank", calls),
            vina=FakeRunner("vina", calls),
        ),
        calls,
    )


def _toolchain(*, posebusters: bool = True) -> ReceptorToolchain:
    return ReceptorToolchain(
        capabilities=tuple(
            ToolCapability(
                tool_id=name,
                available=(posebusters if name == "posebusters" else True),
                version="contract-test" if name != "p2rank" else "2.5.1",
                reason=None if name != "posebusters" or posebusters else "not installed",
            )
            for name in ReceptorToolchain.REQUIRED_TOOLS
        ),
        probed_at="2026-08-21T00:00:00+00:00",
    )


def _manifest():
    payload = json.loads(FLT3_FIXTURE.read_text(encoding="utf-8"))
    request = TargetStructureRequest(**payload["request"])
    rcsb = [structure_candidate_from_dict(item) for item in payload["rcsb_candidates"]]
    alpha = structure_candidate_from_dict(payload["alphafold_candidate"])
    return request, rcsb, alpha


def _source(path: Path, *, with_ligand: bool) -> Path:
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N"
    ]
    if with_ligand:
        lines.extend(
            (
                "HETATM    2  C1  TST A 900       1.000   2.000   3.000  1.00 20.00           C",
                "HETATM    3  C2  TST A 900       5.000   8.000   9.000  1.00 20.00           C",
            )
        )
    path.write_text("\n".join([*lines, "END", ""]), encoding="utf-8")
    return path


def _config(tmp_path: Path, *, source: Path, output: str, reference: bool):
    validation = tmp_path / f"{output}-ligand.sdf"
    validation.write_text("contract ligand\n$$$$\n", encoding="utf-8")
    return LocalReceptorPreparationConfig(
        source_structure=source,
        output_dir=tmp_path / output,
        python_executable=Path("python"),
        meeko_receptor_executable=Path("mk_prepare_receptor"),
        meeko_ligand_executable=Path("mk_prepare_ligand"),
        p2rank_executable=Path("prank"),
        vina_executable=Path("vina"),
        validation_ligand_sdf=validation,
        reference_component_sdf=validation if reference else None,
        box_ligand_instance_id="TST:A:900" if reference else None,
        p2rank_min_probability=0.80,
    )


def test_real_backend_contract_executes_cocrystal_protocol_with_injected_runners(
    tmp_path: Path,
) -> None:
    request, candidates, _ = _manifest()
    selection = ReceptorQualifier().select(request, candidates)
    source = _source(tmp_path / "flt3-contract.pdb", with_ligand=True)
    runners, calls = _runners()
    backend = LocalReceptorPreparationBackend(
        _config(tmp_path, source=source, output="experimental", reference=True),
        runners=runners,
    )

    prepared = backend.prepare(selection, _toolchain())

    assert prepared.pocket.origin is PocketOrigin.COCRYSTAL_LIGAND
    assert prepared.pocket.ligand_instance_id == "TST:A:900"
    assert prepared.pocket.center == (3.0, 5.0, 6.0)
    assert prepared.pocket.size == (16.0, 18.0, 18.0)
    assert prepared.redocking.passed is True
    assert prepared.pose_validation.passed is True
    roles = {item.role for item in prepared.artifacts}
    assert {
        "pdbfixer_receptor",
        "meeko_receptor_pdbqt",
        "docked_pose_pdbqt",
        "posebusters_report",
        "redocking_report",
        "crystal_ligand_pdb",
    } <= roles
    assert not any(call and call[0] == "predict" for call in calls)

    profile = CandidateDockingProfileBuilder().build(
        profile_id="FLT3-local-contract-v1",
        profile_version="1.0",
        selection=selection,
        toolchain=_toolchain(),
        prepared=prepared,
    )
    assert profile["production_ready"] is False
    assert profile["fixture"] is True
    assert all(len(item["sha256"]) == 64 for item in profile["artifacts"])


def test_real_backend_contract_runs_p2rank_for_alphafold_and_labels_prediction(
    tmp_path: Path,
) -> None:
    request, candidates, alpha = _manifest()
    selection = ReceptorQualifier().select(
        request,
        (candidates[2],),
        alphafold_candidate=alpha,
    )
    assert selection.fallback_reason is FallbackReason.NO_EXACT_RCSB_HOLO
    source = _source(tmp_path / "af-flt3-contract.pdb", with_ligand=False)
    runners, calls = _runners()
    backend = LocalReceptorPreparationBackend(
        _config(tmp_path, source=source, output="alphafold", reference=False),
        runners=runners,
    )

    prepared = backend.prepare(selection, _toolchain())

    assert selection.selected.source is StructureSource.ALPHAFOLD_PREDICTED
    assert prepared.pocket.origin is PocketOrigin.P2RANK_PREDICTED
    assert prepared.pocket.prediction_label == "predicted"
    assert prepared.pocket.center == (11.0, 12.0, 13.0)
    assert prepared.pocket.prediction_score == 12.5
    assert prepared.redocking.status.value == "not_applicable"
    assert any(call and call[0] == "predict" and "alphafold" in call for call in calls)
    assert "p2rank_output" in {item.role for item in prepared.artifacts}


def test_backend_fails_before_any_runner_when_posebusters_is_missing(tmp_path: Path) -> None:
    request, candidates, _ = _manifest()
    selection = ReceptorQualifier().select(request, candidates)
    source = _source(tmp_path / "flt3-contract.pdb", with_ligand=True)
    runners, calls = _runners()
    backend = LocalReceptorPreparationBackend(
        _config(tmp_path, source=source, output="missing-tool", reference=True),
        runners=runners,
    )
    with pytest.raises(ReceptorPipelineError) as captured:
        backend.prepare(selection, _toolchain(posebusters=False))
    assert captured.value.code == "RECEPTOR_PREPARATION_TOOLCHAIN_UNAVAILABLE"
    assert captured.value.details["missing_tools"] == ["posebusters"]
    assert calls == []


def test_backend_passes_relative_rdkit_and_openmm_runtime_paths_to_workers(
    tmp_path: Path,
) -> None:
    request, candidates, _ = _manifest()
    selection = ReceptorQualifier().select(request, candidates)
    source = _source(tmp_path / "flt3-contract.pdb", with_ligand=True)
    config = _config(tmp_path, source=source, output="relative-runtime", reference=True)
    rdkit_root = tmp_path / "runtime" / "rdkit"
    openmm_plugins = tmp_path / "runtime" / "openmm-plugins"
    rdkit_root.mkdir(parents=True)
    openmm_plugins.mkdir(parents=True)
    config = LocalReceptorPreparationConfig(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field not in {"rdkit_root", "openmm_plugin_dir"}
        },
        rdkit_root=rdkit_root,
        openmm_plugin_dir=openmm_plugins,
    )
    calls: list[tuple[str, ...]] = []
    environments: list[dict[str, str] | None] = []
    runners = LocalPreparationRunners(
        python=FakeRunner("python", calls, environments),
        meeko_receptor=FakeRunner("meeko_receptor", calls, environments),
        meeko_ligand=FakeRunner("meeko_ligand", calls, environments),
        p2rank=FakeRunner("p2rank", calls, environments),
        vina=FakeRunner("vina", calls, environments),
    )

    LocalReceptorPreparationBackend(config, runners=runners).prepare(
        selection,
        _toolchain(),
    )

    configured = [env for env in environments if env and "RDBASE" in env]
    assert len(configured) == 4
    for environment in configured:
        assert environment["RDBASE"] == os.path.relpath(
            rdkit_root.resolve(),
            start=config.output_dir.resolve(),
        )
        assert environment["OPENMM_PLUGIN_DIR"] == os.path.relpath(
            openmm_plugins.resolve(),
            start=config.output_dir.resolve(),
        )
        assert str(ROOT / "src") in environment["PYTHONPATH"]


def test_backend_rejects_retaining_the_selected_cocrystal_ligand(tmp_path: Path) -> None:
    request, candidates, _ = _manifest()
    selection = ReceptorQualifier().select(request, candidates)
    source = _source(tmp_path / "flt3-contract.pdb", with_ligand=True)
    config = _config(tmp_path, source=source, output="retained-ligand", reference=True)
    config = LocalReceptorPreparationConfig(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field != "retain_component_instances"
        },
        retain_component_instances=("TST:A:900",),
    )
    runners, calls = _runners()
    backend = LocalReceptorPreparationBackend(config, runners=runners)
    with pytest.raises(ReceptorPipelineError) as captured:
        backend.prepare(selection, _toolchain())
    assert captured.value.code == "BOUND_LIGAND_RETAINED_IN_RECEPTOR"
    assert calls == []


def test_backend_wraps_subprocess_failure_with_execution_provenance(tmp_path: Path) -> None:
    request, candidates, _ = _manifest()
    selection = ReceptorQualifier().select(request, candidates)
    source = _source(tmp_path / "flt3-contract.pdb", with_ligand=True)
    calls: list[tuple[str, ...]] = []
    runners = LocalPreparationRunners(
        python=FailingRunner("python", calls),
        meeko_receptor=FakeRunner("meeko_receptor", calls),
        meeko_ligand=FakeRunner("meeko_ligand", calls),
        p2rank=FakeRunner("p2rank", calls),
        vina=FakeRunner("vina", calls),
    )
    backend = LocalReceptorPreparationBackend(
        _config(tmp_path, source=source, output="tool-failure", reference=True),
        runners=runners,
    )

    with pytest.raises(ReceptorPipelineError) as captured:
        backend.prepare(selection, _toolchain())

    assert captured.value.code == "LOCAL_RECEPTOR_TOOL_EXECUTION_FAILED"
    assert captured.value.details["execution"]["tool_name"] == "python"
    assert len(calls) == 1
    assert "drugpilot.tools.pdbfixer_receptor_worker" in calls[0]
