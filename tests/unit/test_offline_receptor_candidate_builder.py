from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_receptor_candidate_profile import (
    MANIFEST_COPY_FILENAME,
    PROFILE_FILENAME,
    _parser,
    build_candidate_profile,
    load_qualified_build_input,
)
from drugpilot.tools.receptor_qualification import (
    FileArtifact,
    DockingPocket,
    PocketOrigin,
    PoseValidation,
    PreparedReceptorRun,
    ReceptorPipelineError,
    ReceptorToolchain,
    RedockingValidation,
    ToolCapability,
    ValidationStatus,
)
from drugpilot.tools.scientific_tool_health import LocalReceptorToolConfiguration


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "data"
    / "docking"
    / "IDH1"
    / "qualified-6U4J-PWV-A503-build-manifest.json"
)
ACTIVE_REGISTRY = ROOT / "data" / "docking" / "IDH1" / "6U4J-PWV-A503" / "profile.json"


def _toolchain() -> ReceptorToolchain:
    return ReceptorToolchain(
        capabilities=tuple(
            ToolCapability(
                tool_id=tool_id,
                available=True,
                version="offline-builder-test",
            )
            for tool_id in (*ReceptorToolchain.REQUIRED_TOOLS, "vina")
        ),
        probed_at="2026-08-21T00:00:00+00:00",
    )


def _report():
    return SimpleNamespace(
        receptor_toolchain=_toolchain(),
        local_receptor_configuration=LocalReceptorToolConfiguration(
            python_executable=Path("python"),
            rdkit_root=None,
            openmm_plugin_dir=None,
            meeko_receptor_executable=Path("mk_prepare_receptor"),
            meeko_ligand_executable=Path("mk_prepare_ligand"),
            p2rank_executable=Path("prank"),
            p2rank_prefix_args=(),
            vina_executable=Path("vina"),
        ),
    )


def _settings():
    return SimpleNamespace(
        receptor_builder_enabled=True,
        receptor_python_executable=Path("python"),
        receptor_meeko_receptor_executable=Path("mk_prepare_receptor"),
        receptor_meeko_ligand_executable=Path("mk_prepare_ligand"),
        receptor_p2rank_executable=Path("prank"),
        receptor_p2rank_jar_path=None,
        receptor_java_executable=None,
        receptor_rdkit_root=None,
        receptor_openmm_plugin_dir=None,
        vina_binary=Path("vina"),
        receptor_probe_timeout_seconds=1.0,
    )


@dataclass
class FakeBackend:
    config: object
    pose_passed: bool = True
    redocking_passed: bool = True

    def prepare(self, selection, toolchain):
        output = self.config.output_dir
        artifacts = [
            FileArtifact(
                role="source_structure",
                path=self.config.source_structure,
                produced_by="real pinned input",
            )
        ]
        for role in (
            "pdbfixer_receptor",
            "meeko_receptor_pdbqt",
            "posebusters_report",
            "redocking_report",
        ):
            path = output / f"{role}.txt"
            path.write_text(f"auditable {role}\n", encoding="utf-8")
            artifacts.append(FileArtifact(role=role, path=path, produced_by="test backend"))
        return PreparedReceptorRun(
            pocket=DockingPocket(
                origin=PocketOrigin.COCRYSTAL_LIGAND,
                center=(10.0, -4.0, 23.0),
                size=(21.0, 20.0, 19.0),
                ligand_instance_id="PWV:A:503",
            ),
            artifacts=tuple(artifacts),
            redocking=RedockingValidation(
                status=(
                    ValidationStatus.PASSED
                    if self.redocking_passed
                    else ValidationStatus.FAILED
                ),
                threshold_angstrom=2.0,
                top_pose_rmsd_angstrom=0.25 if self.redocking_passed else 3.5,
                validation_artifact_role="redocking_report",
                method="symmetry-aware test RMSD",
            ),
            pose_validation=PoseValidation(
                status=(
                    ValidationStatus.PASSED
                    if self.pose_passed
                    else ValidationStatus.FAILED
                ),
                validator="PoseBusters",
                checked_pose_count=1,
                failure_codes=() if self.pose_passed else ("distance_geometry",),
                validation_artifact_role="posebusters_report",
            ),
        )


def test_checked_in_idh1_manifest_reparses_and_exactly_requalifies_real_6u4j() -> None:
    build_input = load_qualified_build_input(MANIFEST)

    assert build_input.selection.selected.fixture is False
    assert build_input.selection.production_eligible_source is True
    assert build_input.selection.selected.structure_id == "6U4J"
    assert build_input.selection.request.target_symbol == "IDH1"
    assert build_input.selection.request.uniprot_accession == "O75874"
    assert build_input.selection.request.required_mutations == ("R132H",)
    assert build_input.box_ligand_instance_id == "PWV:A:503"
    assert build_input.selection.selected.source_hashes["source_pdb_sha256"] == (
        "2c46a9bdcd4455bbe2328ffb72a091b72fd5a2352a0b4b9daa1193efca7c93a7"
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("fixture", "FIXTURE_RECEPTOR_BUILD_FORBIDDEN"),
        ("missing_identity", "RECEPTOR_EXACT_IDENTITY_MISSING"),
        ("unqualified", "RECEPTOR_SELECTION_NOT_QUALIFIED"),
    ),
)
def test_manifest_fails_closed_before_execution_for_fixture_or_unqualified_input(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if mutation == "fixture":
        raw["fixture"] = True
    elif mutation == "missing_identity":
        del raw["request"]["required_mutations"]
    else:
        raw["qualification"]["status"] = "pending"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ReceptorPipelineError) as captured:
        load_qualified_build_input(path)

    assert captured.value.code == expected_code


def test_builder_emits_only_unaccepted_schema2_candidate_and_never_touches_registry(
    tmp_path: Path,
) -> None:
    registry_before = ACTIVE_REGISTRY.read_bytes()
    output = tmp_path / "candidate-run-001"

    profile_path = build_candidate_profile(
        manifest_path=MANIFEST,
        output_dir=output,
        settings=_settings(),
        probe=lambda **_: _report(),
        backend_factory=lambda config: FakeBackend(config),
    )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile_path.name == PROFILE_FILENAME
    assert profile["schema"] == "drugpilot.candidate-docking-profile/2.0"
    assert profile["production_ready"] is False
    assert profile["profile_status"] == "candidate_not_accepted"
    assert profile["production_acceptance"]["accepted"] is False
    assert "PRODUCTION_ACCEPTANCE_MISSING" in profile["production_readiness_reasons"]
    assert MANIFEST_COPY_FILENAME in {item["filename"] for item in profile["artifacts"]}
    assert ACTIVE_REGISTRY.read_bytes() == registry_before

    with pytest.raises(ReceptorPipelineError) as captured:
        build_candidate_profile(
            manifest_path=MANIFEST,
            output_dir=output,
            settings=_settings(),
            probe=lambda **_: _report(),
            backend_factory=lambda config: FakeBackend(config),
        )
    assert captured.value.code == "RECEPTOR_OUTPUT_DIRECTORY_COLLISION"


@pytest.mark.parametrize(
    ("pose_passed", "redocking_passed", "expected_code"),
    (
        (False, True, "POSEBUSTERS_VALIDATION_FAILED"),
        (True, False, "REDOCKING_VALIDATION_FAILED"),
    ),
)
def test_builder_emits_no_profile_when_pose_or_rmsd_validation_fails(
    tmp_path: Path,
    pose_passed: bool,
    redocking_passed: bool,
    expected_code: str,
) -> None:
    output = tmp_path / expected_code.casefold()

    with pytest.raises(ReceptorPipelineError) as captured:
        build_candidate_profile(
            manifest_path=MANIFEST,
            output_dir=output,
            settings=_settings(),
            probe=lambda **_: _report(),
            backend_factory=lambda config: FakeBackend(
                config,
                pose_passed=pose_passed,
                redocking_passed=redocking_passed,
            ),
        )

    assert captured.value.code == expected_code
    assert not (output / PROFILE_FILENAME).exists()
    assert not (output / MANIFEST_COPY_FILENAME).exists()


def test_cli_has_no_acceptance_or_registry_promotion_option() -> None:
    help_text = _parser().format_help().casefold()
    assert "--accept" not in help_text
    assert "--promote" not in help_text
    assert "--registry" not in help_text
