from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Protocol, Sequence

from drugpilot.tools.executable import (
    ExecutableResult,
    ExecutableRunner,
    ScientificToolExecutionError,
    ScientificToolTimeout,
    ScientificToolUnavailable,
)
from drugpilot.tools.receptor_qualification import (
    DockingPocket,
    FileArtifact,
    PocketOrigin,
    PoseValidation,
    PreparedReceptorRun,
    ReceptorPipelineError,
    ReceptorToolchain,
    RedockingValidation,
    StructureSelection,
    StructureSource,
    ValidationStatus,
)


class Runner(Protocol):
    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecutableResult: ...


def _required_file(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ReceptorPipelineError(
            "LOCAL_RECEPTOR_OUTPUT_MISSING",
            f"Local receptor preparation omitted {role}",
            details={"role": role, "filename": resolved.name},
        )
    return resolved


def _execution(result: ExecutableResult) -> dict[str, Any]:
    return result.provenance.as_dict()


@dataclass(frozen=True, slots=True)
class LocalReceptorPreparationConfig:
    source_structure: Path
    output_dir: Path
    python_executable: Path
    meeko_receptor_executable: Path
    meeko_ligand_executable: Path
    p2rank_executable: Path
    vina_executable: Path
    validation_ligand_sdf: Path
    rdkit_root: Path | None = None
    openmm_plugin_dir: Path | None = None
    p2rank_prefix_args: tuple[str, ...] = ()
    reference_component_sdf: Path | None = None
    box_ligand_instance_id: str | None = None
    retain_component_instances: tuple[str, ...] = ()
    retain_water_instances: tuple[str, ...] = ()
    replace_nonstandard_residues: bool = False
    add_missing_residues: bool = False
    add_hydrogens_with_pdbfixer: bool = False
    ph: float = 7.4
    cocrystal_box_padding_angstrom: float = 6.0
    predicted_box_size_angstrom: tuple[float, float, float] = (22.0, 22.0, 22.0)
    p2rank_min_probability: float = 0.0
    ligand_preparation_timeout_seconds: float = 120.0
    receptor_preparation_timeout_seconds: float = 300.0
    p2rank_timeout_seconds: float = 300.0
    docking_timeout_seconds: float = 300.0
    validation_timeout_seconds: float = 300.0
    redocking_rmsd_threshold_angstrom: float = 2.0
    pdbfixer_seed: int = 20260819
    vina_seed: int = 20260819
    vina_exhaustiveness: int = 8
    vina_num_modes: int = 9
    vina_cpu: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.ph < 14.0:
            raise ValueError("ph must be between 0 and 14")
        if self.cocrystal_box_padding_angstrom <= 0:
            raise ValueError("cocrystal_box_padding_angstrom must be positive")
        if len(self.predicted_box_size_angstrom) != 3 or any(
            value <= 0 for value in self.predicted_box_size_angstrom
        ):
            raise ValueError("predicted_box_size_angstrom must be a positive 3-vector")
        if not 0.0 <= self.p2rank_min_probability <= 1.0:
            raise ValueError("p2rank_min_probability must be in [0, 1]")
        for value in (
            self.ligand_preparation_timeout_seconds,
            self.receptor_preparation_timeout_seconds,
            self.p2rank_timeout_seconds,
            self.docking_timeout_seconds,
            self.validation_timeout_seconds,
            self.redocking_rmsd_threshold_angstrom,
        ):
            if value <= 0:
                raise ValueError("Scientific execution timeouts/thresholds must be positive")
        if self.vina_exhaustiveness < 1 or self.vina_num_modes < 1 or self.vina_cpu < 1:
            raise ValueError("Vina execution parameters must be positive")
        if self.pdbfixer_seed < 1:
            raise ValueError("pdbfixer_seed must be positive for reproducible atom placement")


@dataclass(frozen=True, slots=True)
class LocalPreparationRunners:
    python: Runner
    meeko_receptor: Runner
    meeko_ligand: Runner
    p2rank: Runner
    vina: Runner

    @classmethod
    def from_config(cls, config: LocalReceptorPreparationConfig) -> "LocalPreparationRunners":
        return cls(
            python=ExecutableRunner(
                "PDBFixer/PoseBusters Python worker",
                executable=config.python_executable,
                version_args=("--version",),
            ),
            meeko_receptor=ExecutableRunner(
                "Meeko receptor preparation",
                executable=config.meeko_receptor_executable,
                version_args=("--help",),
            ),
            meeko_ligand=ExecutableRunner(
                "Meeko validation-ligand preparation",
                executable=config.meeko_ligand_executable,
                version_args=("--help",),
            ),
            p2rank=ExecutableRunner(
                "P2Rank pocket prediction",
                executable=config.p2rank_executable,
                version_args=(*config.p2rank_prefix_args, "-v"),
            ),
            vina=ExecutableRunner(
                "AutoDock Vina redocking",
                executable=config.vina_executable,
                version_args=("--version",),
            ),
        )


class LocalReceptorPreparationBackend:
    """Execute a receptor build/validation protocol without graph-side fallback.

    The backend intentionally receives already-qualified structure metadata. It
    does not query RCSB or AlphaFold and cannot promote a profile; it only emits
    auditable preparation artifacts and validation records for the independent
    candidate-profile builder.
    """

    def __init__(
        self,
        config: LocalReceptorPreparationConfig,
        *,
        runners: LocalPreparationRunners | None = None,
    ) -> None:
        self.config = config
        try:
            self.runners = runners or LocalPreparationRunners.from_config(config)
        except ScientificToolUnavailable as exc:
            raise ReceptorPipelineError(
                "LOCAL_RECEPTOR_EXECUTABLE_UNAVAILABLE",
                str(exc),
            ) from exc

    def prepare(
        self,
        selection: StructureSelection,
        toolchain: ReceptorToolchain,
    ) -> PreparedReceptorRun:
        toolchain.require_all()
        try:
            return self._prepare(selection)
        except ReceptorPipelineError:
            raise
        except (ScientificToolExecutionError, ScientificToolTimeout) as exc:
            raise ReceptorPipelineError(
                "LOCAL_RECEPTOR_TOOL_EXECUTION_FAILED",
                str(exc),
                details={"execution": exc.provenance.as_dict()},
            ) from exc
        except ScientificToolUnavailable as exc:
            raise ReceptorPipelineError(
                "LOCAL_RECEPTOR_EXECUTABLE_UNAVAILABLE",
                str(exc),
            ) from exc
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ReceptorPipelineError(
                "LOCAL_RECEPTOR_PREPARATION_FAILED",
                f"Local receptor preparation failed closed: {type(exc).__name__}: {exc}",
            ) from exc

    def _prepare(self, selection: StructureSelection) -> PreparedReceptorRun:
        config = self.config
        selected = selection.selected
        source = _required_file(config.source_structure, "source_structure")
        validation_sdf = _required_file(config.validation_ligand_sdf, "validation_ligand_sdf")
        if selected.source is StructureSource.RCSB_EXPERIMENTAL:
            if config.reference_component_sdf is None:
                raise ReceptorPipelineError(
                    "REDOCKING_REFERENCE_MISSING",
                    "An experimental holo receptor requires an exact reference-component SDF",
                )
            reference_sdf = _required_file(
                config.reference_component_sdf,
                "reference_component_sdf",
            )
        else:
            reference_sdf = None

        output = config.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ReceptorPipelineError(
                "RECEPTOR_OUTPUT_DIRECTORY_NOT_EMPTY",
                "Receptor output directory must be empty to preserve immutable artifacts",
                details={"directory_name": output.name},
            )
        worker_environment = self._worker_environment(output)

        fixed_pdb = output / "receptor_fixed.pdb"
        fixer_report = output / "pdbfixer_report.json"
        receptor_pdbqt = output / "receptor.pdbqt"
        receptor_json = output / "receptor.json"
        box_file = output / "vina_box.txt"
        prepared_ligand = output / "validation_ligand.pdbqt"
        docked_pose = output / "docked.pdbqt"
        top_pose_sdf = output / "docked_top_pose.sdf"
        posebusters_report = output / "posebusters_report.json"
        posebusters_reference = output / "posebusters_crystal_reference.sdf"
        redocking_report = output / "redocking_validation.json"
        validation_input = output / "validation_input.sdf"
        shutil.copyfile(validation_sdf, validation_input)
        _required_file(validation_input, "validation_ligand_execution_input")
        if reference_sdf is not None:
            reference_input = output / "reference_component.sdf"
            shutil.copyfile(reference_sdf, reference_input)
            _required_file(reference_input, "reference_component_execution_input")
        else:
            reference_input = None

        retain_components = self._validate_retention(selection)
        fixer_args = [
            "-m",
            "drugpilot.tools.pdbfixer_receptor_worker",
            "--input",
            str(source),
            "--output",
            fixed_pdb.name,
            "--report",
            fixer_report.name,
            "--chains",
            ",".join(selected.chains),
            "--ph",
            str(config.ph),
            "--seed",
            str(config.pdbfixer_seed),
        ]
        for instance in retain_components:
            fixer_args.extend(("--retain-component", instance))
        for instance in sorted(set(config.retain_water_instances)):
            fixer_args.extend(("--retain-water", instance))
        if config.replace_nonstandard_residues:
            fixer_args.append("--replace-nonstandard-residues")
        if config.add_missing_residues:
            fixer_args.append("--add-missing-residues")
        if config.add_hydrogens_with_pdbfixer:
            fixer_args.append("--add-hydrogens")
        fixer_execution = self.runners.python.run(
            fixer_args,
            timeout_seconds=config.receptor_preparation_timeout_seconds,
            cwd=output,
            env=worker_environment,
        )
        _required_file(fixed_pdb, "pdbfixer_receptor")
        _required_file(fixer_report, "pdbfixer_report")

        p2rank_artifact: Path | None = None
        p2rank_execution: ExecutableResult | None = None
        crystal_ligand_pdb: Path | None = None
        if selected.source is StructureSource.RCSB_EXPERIMENTAL:
            ligand = self._select_box_ligand(selection)
            crystal_ligand_pdb = output / "crystal_ligand.pdb"
            coordinates = self._extract_bound_ligand(
                source,
                ligand.instance_id,
                crystal_ligand_pdb,
            )
            pocket = self._cocrystal_pocket(ligand.instance_id, coordinates)
        else:
            p2rank_dir = output / "p2rank"
            p2rank_dir.mkdir()
            p2rank_execution = self.runners.p2rank.run(
                (
                    *config.p2rank_prefix_args,
                    "predict",
                    "-f",
                    fixed_pdb.name,
                    "-c",
                    "alphafold",
                    "-visualizations",
                    "0",
                    "-threads",
                    "1",
                    "-o",
                    p2rank_dir.name,
                ),
                timeout_seconds=config.p2rank_timeout_seconds,
                cwd=output,
            )
            predictions = sorted(p2rank_dir.glob("*_predictions.csv"))
            if len(predictions) != 1:
                raise ReceptorPipelineError(
                    "P2RANK_PREDICTION_ARTIFACT_AMBIGUOUS",
                    "P2Rank must emit exactly one predictions CSV",
                    details={"artifact_count": len(predictions)},
                )
            p2rank_artifact = _required_file(predictions[0], "p2rank_output")
            pocket = self._predicted_pocket(p2rank_artifact)

        center = pocket.center
        size = pocket.size
        meeko_execution = self.runners.meeko_receptor.run(
            (
                "--read_pdb",
                fixed_pdb.name,
                "--write_pdbqt",
                receptor_pdbqt.name,
                "--write_json",
                receptor_json.name,
                "--write_vina_box",
                box_file.name,
                "--box_center",
                *(str(value) for value in center),
                "--box_size",
                *(str(value) for value in size),
                "--charge_model",
                "gasteiger",
            ),
            timeout_seconds=config.receptor_preparation_timeout_seconds,
            cwd=output,
            env=worker_environment,
        )
        for path, role in (
            (receptor_pdbqt, "meeko_receptor_pdbqt"),
            (receptor_json, "meeko_receptor_json"),
            (box_file, "meeko_vina_box"),
        ):
            _required_file(path, role)

        ligand_execution = self.runners.meeko_ligand.run(
            ("-i", validation_input.name, "-o", prepared_ligand.name),
            timeout_seconds=config.ligand_preparation_timeout_seconds,
            cwd=output,
            env=worker_environment,
        )
        _required_file(prepared_ligand, "prepared_validation_ligand")

        vina_execution = self.runners.vina.run(
            (
                "--receptor",
                receptor_pdbqt.name,
                "--ligand",
                prepared_ligand.name,
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
                "--exhaustiveness",
                str(config.vina_exhaustiveness),
                "--num_modes",
                str(config.vina_num_modes),
                "--cpu",
                str(config.vina_cpu),
                "--seed",
                str(config.vina_seed),
                "--out",
                docked_pose.name,
            ),
            timeout_seconds=config.docking_timeout_seconds,
            cwd=output,
        )
        _required_file(docked_pose, "docked_pose_pdbqt")

        mode = "redock" if selected.source is StructureSource.RCSB_EXPERIMENTAL else "dock"
        validation_args = [
            "-m",
            "drugpilot.tools.posebusters_validation_worker",
            "--docked",
            docked_pose.name,
            "--receptor",
            fixed_pdb.name,
            "--output-sdf",
            top_pose_sdf.name,
            "--posebusters-report",
            posebusters_report.name,
            "--mode",
            mode,
        ]
        if mode == "redock":
            assert reference_input is not None
            assert crystal_ligand_pdb is not None
            validation_args.extend(
                (
                    "--reference-sdf",
                    reference_input.name,
                    "--crystal-ligand-pdb",
                    crystal_ligand_pdb.name,
                    "--posebusters-reference-output",
                    posebusters_reference.name,
                    "--redocking-report",
                    redocking_report.name,
                    "--rmsd-threshold",
                    str(config.redocking_rmsd_threshold_angstrom),
                )
            )
        validation_execution = self.runners.python.run(
            validation_args,
            timeout_seconds=config.validation_timeout_seconds,
            cwd=output,
            env=worker_environment,
        )
        _required_file(top_pose_sdf, "docked_top_pose_sdf")
        _required_file(posebusters_report, "posebusters_report")
        pose_payload = json.loads(posebusters_report.read_text(encoding="utf-8"))
        failures = tuple(str(item) for item in pose_payload.get("failure_codes", []))
        pose_validation = PoseValidation(
            status=(
                ValidationStatus.PASSED
                if pose_payload.get("passed") is True and not failures
                else ValidationStatus.FAILED
            ),
            validator="PoseBusters",
            checked_pose_count=int(pose_payload.get("checked_pose_count", 0)),
            failure_codes=failures,
            validation_artifact_role="posebusters_report",
        )
        if mode == "redock":
            _required_file(redocking_report, "redocking_report")
            _required_file(posebusters_reference, "posebusters_crystal_reference")
            redocking_payload = json.loads(redocking_report.read_text(encoding="utf-8"))
            redocking_validation = RedockingValidation(
                status=(
                    ValidationStatus.PASSED
                    if redocking_payload.get("passed") is True
                    else ValidationStatus.FAILED
                ),
                threshold_angstrom=float(redocking_payload["threshold_angstrom"]),
                top_pose_rmsd_angstrom=float(
                    redocking_payload["top_ranked_pose_rmsd_angstrom"]
                ),
                validation_artifact_role="redocking_report",
                method=str(redocking_payload.get("mapping_method") or "redocking RMSD"),
            )
        else:
            redocking_validation = RedockingValidation(
                status=ValidationStatus.NOT_APPLICABLE,
                threshold_angstrom=None,
                top_pose_rmsd_angstrom=None,
                validation_artifact_role=None,
                method="Predicted pocket has no co-crystal redocking reference",
            )

        artifacts = [
            FileArtifact("source_structure", source, selected.source.value),
            FileArtifact(
                "validation_ligand_sdf",
                validation_sdf,
                "operator-supplied exact validation ligand",
            ),
            FileArtifact(
                "validation_ligand_execution_input_sdf",
                validation_input,
                "byte-identical ASCII-name staging for RDKit/Meeko",
            ),
            FileArtifact(
                "pdbfixer_receptor",
                fixed_pdb,
                "PDBFixer/OpenMM",
                _execution(fixer_execution),
            ),
            FileArtifact(
                "pdbfixer_report",
                fixer_report,
                "PDBFixer/OpenMM",
                _execution(fixer_execution),
            ),
            FileArtifact(
                "meeko_receptor_pdbqt",
                receptor_pdbqt,
                "Meeko",
                _execution(meeko_execution),
            ),
            FileArtifact(
                "meeko_receptor_json",
                receptor_json,
                "Meeko",
                _execution(meeko_execution),
            ),
            FileArtifact(
                "meeko_vina_box",
                box_file,
                "Meeko",
                _execution(meeko_execution),
            ),
            FileArtifact(
                "prepared_validation_ligand_pdbqt",
                prepared_ligand,
                "Meeko",
                _execution(ligand_execution),
            ),
            FileArtifact(
                "docked_pose_pdbqt",
                docked_pose,
                "AutoDock Vina",
                _execution(vina_execution),
            ),
            FileArtifact(
                "docked_top_pose_sdf",
                top_pose_sdf,
                "Meeko/RDKit pose reconstruction",
                _execution(validation_execution),
            ),
            FileArtifact(
                "posebusters_report",
                posebusters_report,
                "PoseBusters",
                _execution(validation_execution),
            ),
        ]
        if reference_sdf is not None:
            artifacts.append(
                FileArtifact(
                    "reference_component_sdf",
                    reference_sdf,
                    "operator-supplied exact chemical component reference",
                )
            )
        if reference_input is not None:
            artifacts.append(
                FileArtifact(
                    "reference_component_execution_input_sdf",
                    reference_input,
                    "byte-identical ASCII-name staging for RDKit/PoseBusters",
                )
            )
        if crystal_ligand_pdb is not None:
            artifacts.append(
                FileArtifact(
                    "crystal_ligand_pdb",
                    crystal_ligand_pdb,
                    "exact source-structure HETATM extraction",
                )
            )
        if mode == "redock":
            artifacts.append(
                FileArtifact(
                    "redocking_report",
                    redocking_report,
                    "symmetry-aware heavy-atom RMSD",
                    _execution(validation_execution),
                )
            )
            artifacts.append(
                FileArtifact(
                    "posebusters_crystal_reference_sdf",
                    posebusters_reference,
                    "RDKit exact graph with selected co-crystal coordinates",
                    _execution(validation_execution),
                )
            )
        if p2rank_artifact is not None and p2rank_execution is not None:
            artifacts.append(
                FileArtifact(
                    "p2rank_output",
                    p2rank_artifact,
                    "P2Rank alphafold profile",
                    _execution(p2rank_execution),
                )
            )
        return PreparedReceptorRun(
            pocket=pocket,
            artifacts=tuple(artifacts),
            redocking=redocking_validation,
            pose_validation=pose_validation,
        )

    def _worker_environment(self, output: Path) -> dict[str, str]:
        """Build a merged worker environment without persisting local paths.

        Some Windows RDKit/OpenMM builds cannot open package data or plugin
        directories when their absolute path contains non-ASCII characters.
        Passing paths relative to the subprocess working directory avoids that
        platform limitation while keeping every worker in the clone-local
        environment.  ``ExecutableRunner`` deliberately omits environment
        values from provenance, so these machine-local paths are never stored.
        """

        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        python_path = [
            item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item
        ]
        if source_root not in python_path:
            python_path.insert(0, source_root)
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        environment.setdefault("PYTHONUTF8", "1")

        for variable, configured in (
            ("RDBASE", self.config.rdkit_root),
            ("OPENMM_PLUGIN_DIR", self.config.openmm_plugin_dir),
        ):
            if configured is None:
                continue
            resolved = configured.expanduser().resolve()
            if not resolved.is_dir():
                raise ReceptorPipelineError(
                    "LOCAL_RECEPTOR_RUNTIME_DATA_UNAVAILABLE",
                    f"Configured {variable} runtime directory is unavailable",
                    details={"runtime_variable": variable},
                )
            environment[variable] = os.path.relpath(resolved, start=output)
        return environment

    def _validate_retention(self, selection: StructureSelection) -> tuple[str, ...]:
        available = {item.instance_id: item for item in selection.selected.components}
        requested = tuple(sorted(set(self.config.retain_component_instances)))
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ReceptorPipelineError(
                "RECEPTOR_RETENTION_COMPONENT_UNKNOWN",
                "A retained receptor component was absent from the qualified inventory",
                details={"unknown_component_instances": unknown},
            )
        box_ligands = {item.instance_id for item in selection.selected.box_ligands}
        retained_box_ligands = sorted(set(requested) & box_ligands)
        if retained_box_ligands:
            raise ReceptorPipelineError(
                "BOUND_LIGAND_RETAINED_IN_RECEPTOR",
                "The validation co-crystal ligand must be removed from the docking receptor",
                details={"ligand_instances": retained_box_ligands},
            )
        return requested

    def _select_box_ligand(self, selection: StructureSelection):
        ligands = selection.selected.box_ligands
        requested = self.config.box_ligand_instance_id
        if requested:
            matches = [item for item in ligands if item.instance_id == requested]
            if len(matches) != 1:
                raise ReceptorPipelineError(
                    "COCRYSTAL_LIGAND_SELECTION_MISMATCH",
                    "Requested co-crystal ligand instance was not qualified",
                    details={
                        "requested": requested,
                        "available": [item.instance_id for item in ligands],
                    },
                )
            return matches[0]
        if len(ligands) != 1:
            raise ReceptorPipelineError(
                "COCRYSTAL_LIGAND_SELECTION_REQUIRED",
                "An explicit ligand instance is required when a structure has multiple box ligands",
                details={"available": [item.instance_id for item in ligands]},
            )
        return ligands[0]

    @staticmethod
    def _extract_bound_ligand(
        source: Path,
        instance_id: str,
        output: Path,
    ) -> list[tuple[float, float, float]]:
        component, chain, residue_text = instance_id.split(":", 2)
        coordinates: list[tuple[float, float, float]] = []
        records: list[str] = []
        for line in source.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line.startswith("HETATM") or len(line) < 54:
                continue
            if line[16] not in {" ", "A"}:
                continue
            if (
                line[17:20].strip().upper() != component.upper()
                or line[21].strip() != chain
                or line[22:26].strip() != residue_text
            ):
                continue
            records.append(line)
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
        if not records:
            raise ReceptorPipelineError(
                "COCRYSTAL_LIGAND_COORDINATES_MISSING",
                "The selected ligand instance was absent from the source coordinate file",
            )
        output.write_text(
            "\n".join([*records, "END", ""]),
            encoding="utf-8",
            newline="\n",
        )
        return coordinates

    def _cocrystal_pocket(
        self,
        ligand_instance_id: str,
        coordinates: list[tuple[float, float, float]],
    ) -> DockingPocket:
        center: list[float] = []
        size: list[float] = []
        padding = self.config.cocrystal_box_padding_angstrom
        for axis in range(3):
            values = [point[axis] for point in coordinates]
            center.append(round((min(values) + max(values)) / 2.0, 3))
            size.append(round(max(values) - min(values) + 2.0 * padding, 3))
        return DockingPocket(
            origin=PocketOrigin.COCRYSTAL_LIGAND,
            center=tuple(center),  # type: ignore[arg-type]
            size=tuple(size),  # type: ignore[arg-type]
            ligand_instance_id=ligand_instance_id,
        )

    def _predicted_pocket(self, predictions: Path) -> DockingPocket:
        with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            rows = [
                {str(key).strip(): str(value).strip() for key, value in row.items()}
                for row in reader
            ]
        if not rows:
            raise ReceptorPipelineError(
                "P2RANK_NO_POCKETS",
                "P2Rank emitted no predicted pockets",
            )
        rows.sort(key=lambda row: int(float(row.get("rank") or 999999)))
        top = rows[0]
        probability = float(top.get("probability") or 0.0)
        if probability < self.config.p2rank_min_probability:
            raise ReceptorPipelineError(
                "P2RANK_PROBABILITY_BELOW_THRESHOLD",
                "Top predicted pocket did not pass the configured probability threshold",
                details={
                    "probability": probability,
                    "threshold": self.config.p2rank_min_probability,
                },
            )
        try:
            center = tuple(float(top[key]) for key in ("center_x", "center_y", "center_z"))
        except (KeyError, ValueError) as exc:
            raise ReceptorPipelineError(
                "P2RANK_CENTER_UNAVAILABLE",
                "P2Rank output omitted a parseable pocket center",
            ) from exc
        return DockingPocket(
            origin=PocketOrigin.P2RANK_PREDICTED,
            center=center,  # type: ignore[arg-type]
            size=self.config.predicted_box_size_angstrom,
            prediction_label="predicted",
            prediction_score=float(top.get("score") or 0.0),
            source_artifact_role="p2rank_output",
        )


__all__ = [
    "LocalPreparationRunners",
    "LocalReceptorPreparationBackend",
    "LocalReceptorPreparationConfig",
    "Runner",
]
