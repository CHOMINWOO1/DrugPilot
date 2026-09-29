from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping

from drugpilot.config import Settings
from drugpilot.tools.local_receptor_preparation import (
    LocalReceptorPreparationBackend,
    LocalReceptorPreparationConfig,
)
from drugpilot.tools.receptor_qualification import (
    CandidateDockingProfileBuilder,
    ComponentRole,
    FileArtifact,
    PreparedReceptorRun,
    ProductionAcceptance,
    ReceptorPipelineError,
    ReceptorQualifier,
    ReceptorToolchain,
    StructureSelection,
    StructureSource,
    TargetStructureRequest,
    parse_legacy_rcsb_pdb,
)
from drugpilot.tools.scientific_tool_health import probe_scientific_tools


MANIFEST_SCHEMA = "drugpilot.qualified-receptor-build-manifest/1.0"
PROFILE_FILENAME = "candidate_docking_profile.json"
MANIFEST_COPY_FILENAME = "qualified_build_manifest.json"


@dataclass(frozen=True, slots=True)
class QualifiedBuildInput:
    manifest_path: Path
    profile_id: str
    profile_version: str
    selection: StructureSelection
    source_structure: Path
    validation_ligand_sdf: Path
    reference_component_sdf: Path
    box_ligand_instance_id: str
    retained_components: tuple[str, ...]
    retained_waters: tuple[str, ...]
    preparation: Mapping[str, Any]


def _error(code: str, message: str, **details: Any) -> ReceptorPipelineError:
    return ReceptorPipelineError(code, message, details=details)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_file(manifest: Path, value: Any, expected: Any, role: str) -> Path:
    if not isinstance(value, str) or not isinstance(expected, str):
        raise _error("RECEPTOR_BUILD_MANIFEST_INVALID", f"{role} path/hash is invalid")
    relative = Path(value)
    root = manifest.parent.resolve()
    resolved = (root / relative).resolve()
    if relative.is_absolute():
        raise _error("RECEPTOR_BUILD_MANIFEST_PATH_INVALID", f"{role} must be relative")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise _error(
            "RECEPTOR_BUILD_MANIFEST_PATH_INVALID", f"{role} escapes its manifest"
        ) from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise _error("RECEPTOR_BUILD_INPUT_MISSING", f"{role} is missing")
    if len(expected) != 64 or _digest(resolved) != expected.casefold():
        raise _error("RECEPTOR_BUILD_INPUT_HASH_MISMATCH", f"{role} hash changed")
    return resolved


def load_qualified_build_input(manifest_path: str | Path) -> QualifiedBuildInput:
    """Re-qualify a pinned, non-fixture exact selection using existing science code."""

    manifest = Path(manifest_path).expanduser().resolve()
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
            raise _error(
                "RECEPTOR_BUILD_MANIFEST_SCHEMA_UNSUPPORTED", "Unsupported manifest schema"
            )
        if raw.get("fixture") is not False:
            raise _error(
                "FIXTURE_RECEPTOR_BUILD_FORBIDDEN", "Fixture input cannot build a profile"
            )
        qualification = raw["qualification"]
        if qualification.get("status") != "qualified_exact":
            raise _error(
                "RECEPTOR_SELECTION_NOT_QUALIFIED", "Exact qualification is required"
            )

        request_raw = raw["request"]
        identity_fields = {
            "target_symbol",
            "uniprot_accession",
            "isoform_accession",
            "required_mutations",
            "organism_tax_id",
        }
        missing = sorted(identity_fields - set(request_raw))
        if missing:
            raise _error(
                "RECEPTOR_EXACT_IDENTITY_MISSING",
                "Target, UniProt, isoform, mutation set and organism must be explicit",
                missing_fields=missing,
            )
        if not isinstance(request_raw["required_mutations"], list):
            raise ValueError("required_mutations must be a list")
        request = TargetStructureRequest(
            target_symbol=request_raw["target_symbol"],
            uniprot_accession=request_raw["uniprot_accession"],
            isoform_accession=request_raw["isoform_accession"],
            required_mutations=tuple(request_raw["required_mutations"]),
            organism_tax_id=request_raw["organism_tax_id"],
        )

        structure = raw["structure"]
        if structure.get("source") != StructureSource.RCSB_EXPERIMENTAL.value:
            raise _error(
                "RECEPTOR_BUILD_SOURCE_UNSUPPORTED",
                "Only an exact experimental RCSB holo structure is supported",
            )
        source = _pinned_file(
            manifest, structure["path"], structure["sha256"], "source structure"
        )
        roles = {
            key.upper(): ComponentRole(value)
            for key, value in structure["component_roles"].items()
        }
        candidate = parse_legacy_rcsb_pdb(
            source,
            structure_id=structure["structure_id"],
            component_roles=roles,
            box_ligand_components=tuple(structure["box_ligand_components"]),
            isoform_accession=request.isoform_accession,
        )
        selection = ReceptorQualifier().select(request, (candidate,))
        attested_id = str(qualification["selected_structure_id"]).upper()
        if (
            selection.selected.fixture
            or not selection.production_eligible_source
            or not selection.selected.mutation_annotation_complete
            or selection.selected.structure_id != attested_id
        ):
            raise _error(
                "RECEPTOR_SELECTION_NOT_QUALIFIED",
                "Parsed source does not match the exact qualified selection",
            )

        prep = raw["preparation"]
        validation_sdf = _pinned_file(
            manifest,
            prep["validation_ligand_sdf"],
            prep["validation_ligand_sha256"],
            "validation ligand",
        )
        reference_sdf = _pinned_file(
            manifest,
            prep["reference_component_sdf"],
            prep["reference_component_sha256"],
            "reference component",
        )
        box_instance = prep["box_ligand_instance_id"]
        if box_instance not in {item.instance_id for item in candidate.box_ligands}:
            raise _error(
                "COCRYSTAL_LIGAND_SELECTION_MISMATCH",
                "Box ligand is not an exact qualified ligand instance",
            )
        retained_components = tuple(prep["retain_component_instances"])
        retained_waters = tuple(prep["retain_water_instances"])
        for field, values in (
            ("retain_component_instances", retained_components),
            ("retain_water_instances", retained_waters),
        ):
            if len(values) != len(set(values)) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise ValueError(f"{field} must contain unique non-empty strings")
        for field in (
            "replace_nonstandard_residues",
            "add_missing_residues",
            "add_hydrogens_with_pdbfixer",
        ):
            if not isinstance(prep[field], bool):
                raise ValueError(f"{field} must be boolean")
        profile = raw["profile"]
        if not profile["profile_id"].strip() or not profile["profile_version"].strip():
            raise ValueError("profile ID/version is required")
    except ReceptorPipelineError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _error(
            "RECEPTOR_BUILD_MANIFEST_INVALID", "Invalid qualified receptor build manifest"
        ) from exc

    return QualifiedBuildInput(
        manifest,
        profile["profile_id"],
        profile["profile_version"],
        selection,
        source,
        validation_sdf,
        reference_sdf,
        box_instance,
        retained_components,
        retained_waters,
        prep,
    )


def _configured_report(settings: Settings, probe: Callable[..., Any]):
    if not settings.receptor_builder_enabled:
        raise _error("RECEPTOR_BUILDER_DISABLED", "Offline receptor builder is disabled")
    return probe(
        python_executable=settings.receptor_python_executable,
        receptor_python_executable=settings.receptor_python_executable,
        rdkit_root=settings.receptor_rdkit_root,
        openmm_plugin_dir=settings.receptor_openmm_plugin_dir,
        meeko_receptor_executable=settings.receptor_meeko_receptor_executable,
        meeko_ligand_executable=settings.receptor_meeko_ligand_executable,
        p2rank_executable=settings.receptor_p2rank_executable,
        p2rank_jar_path=settings.receptor_p2rank_jar_path,
        java_executable=settings.receptor_java_executable,
        vina_executable=settings.vina_binary,
        timeout_seconds=settings.receptor_probe_timeout_seconds,
    )


def _local_config(build: QualifiedBuildInput, output: Path, report: Any):
    local = report.local_receptor_configuration
    tools = report.receptor_toolchain.by_id()
    unavailable = [
        name
        for name in (*ReceptorToolchain.REQUIRED_TOOLS, "vina")
        if name not in tools or not tools[name].available
    ]
    if unavailable or any(
        path is None
        for path in (
            local.meeko_receptor_executable,
            local.meeko_ligand_executable,
            local.p2rank_executable,
            local.vina_executable,
        )
    ):
        raise _error(
            "RECEPTOR_PREPARATION_TOOLCHAIN_UNAVAILABLE",
            "Configured receptor tools did not pass exact executable probes",
            unavailable_tools=sorted(unavailable),
        )
    prep = build.preparation
    try:
        return LocalReceptorPreparationConfig(
            source_structure=build.source_structure,
            output_dir=output,
            python_executable=local.python_executable,
            rdkit_root=local.rdkit_root,
            openmm_plugin_dir=local.openmm_plugin_dir,
            meeko_receptor_executable=local.meeko_receptor_executable,
            meeko_ligand_executable=local.meeko_ligand_executable,
            p2rank_executable=local.p2rank_executable,
            p2rank_prefix_args=local.p2rank_prefix_args,
            vina_executable=local.vina_executable,
            validation_ligand_sdf=build.validation_ligand_sdf,
            reference_component_sdf=build.reference_component_sdf,
            box_ligand_instance_id=build.box_ligand_instance_id,
            retain_component_instances=build.retained_components,
            retain_water_instances=build.retained_waters,
            replace_nonstandard_residues=prep["replace_nonstandard_residues"],
            add_missing_residues=prep["add_missing_residues"],
            add_hydrogens_with_pdbfixer=prep["add_hydrogens_with_pdbfixer"],
            ph=prep.get("ph", 7.4),
            cocrystal_box_padding_angstrom=prep.get(
                "cocrystal_box_padding_angstrom", 6.0
            ),
            redocking_rmsd_threshold_angstrom=prep.get(
                "redocking_rmsd_threshold_angstrom", 2.0
            ),
            pdbfixer_seed=prep.get("pdbfixer_seed", 20260819),
            vina_seed=prep.get("vina_seed", 20260819),
            vina_exhaustiveness=prep.get("vina_exhaustiveness", 8),
            vina_num_modes=prep.get("vina_num_modes", 9),
            vina_cpu=prep.get("vina_cpu", 1),
        )
    except (TypeError, ValueError) as exc:
        raise _error(
            "RECEPTOR_BUILD_MANIFEST_INVALID", "Invalid preparation protocol"
        ) from exc


def _require_validation(prepared: PreparedReceptorRun) -> None:
    if not prepared.pose_validation.passed:
        raise _error(
            "POSEBUSTERS_VALIDATION_FAILED",
            "PoseBusters failed; no profile was emitted",
            failure_codes=list(prepared.pose_validation.failure_codes),
        )
    if not prepared.redocking.passed:
        raise _error(
            "REDOCKING_VALIDATION_FAILED",
            "Redocking RMSD failed; no profile was emitted",
            threshold_angstrom=prepared.redocking.threshold_angstrom,
            top_pose_rmsd_angstrom=prepared.redocking.top_pose_rmsd_angstrom,
        )


def build_candidate_profile(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    probe: Callable[..., Any] = probe_scientific_tools,
    backend_factory: Callable[[LocalReceptorPreparationConfig], Any] = (
        LocalReceptorPreparationBackend
    ),
) -> Path:
    build = load_qualified_build_input(manifest_path)
    report = _configured_report(settings, probe)
    report.receptor_toolchain.require_all()
    output = Path(output_dir).expanduser().resolve()
    config = _local_config(build, output, report)
    if output.exists():
        raise _error(
            "RECEPTOR_OUTPUT_DIRECTORY_COLLISION",
            "Immutable output already exists and will not be overwritten",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise _error(
            "RECEPTOR_OUTPUT_DIRECTORY_COLLISION", "Another process reserved the output"
        ) from exc

    prepared = backend_factory(config).prepare(build.selection, report.receptor_toolchain)
    _require_validation(prepared)
    manifest_copy = output / MANIFEST_COPY_FILENAME
    if manifest_copy.exists():
        raise _error("RECEPTOR_ARTIFACT_OUTPUT_COLLISION", "Manifest copy collision")
    shutil.copyfile(build.manifest_path, manifest_copy)
    prepared = replace(
        prepared,
        artifacts=prepared.artifacts
        + (
            FileArtifact(
                "qualified_build_manifest",
                manifest_copy,
                "operator-supplied exact-selection manifest",
            ),
        ),
    )
    profile = CandidateDockingProfileBuilder().build(
        profile_id=build.profile_id,
        profile_version=build.profile_version,
        selection=build.selection,
        toolchain=report.receptor_toolchain,
        prepared=prepared,
        acceptance=ProductionAcceptance(accepted=False),
    )
    if profile["production_ready"] or profile["profile_status"] != "candidate_not_accepted":
        raise _error(
            "CANDIDATE_PROFILE_ACCEPTANCE_BOUNDARY_BROKEN",
            "Offline output must remain an unaccepted candidate",
        )
    destination = output / PROFILE_FILENAME
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(profile, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise _error("RECEPTOR_PROFILE_OUTPUT_COLLISION", "Profile output collision") from exc
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable, unaccepted schema-2 receptor candidate. "
            "This never promotes or updates the active registry."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = build_candidate_profile(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            settings=Settings(_env_file=args.env_file),
        )
        profile = json.loads(destination.read_text(encoding="utf-8"))
    except ReceptorPipelineError as exc:
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "code": "RECEPTOR_BUILD_CONFIGURATION_INVALID",
                    "message": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "candidate_not_accepted",
                "schema": profile["schema"],
                "profile_sha256": profile["profile_sha256"],
                "output": str(destination),
                "active_registry_updated": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
