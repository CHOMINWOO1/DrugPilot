from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from drugpilot.tools.receptor_qualification import (
    CandidateDockingProfileBuilder,
    ComponentRole,
    DockingPocket,
    FallbackReason,
    FileArtifact,
    PocketOrigin,
    PoseValidation,
    PreparedReceptorRun,
    ProductionAcceptance,
    ReceptorPipelineError,
    ReceptorPreparationPipeline,
    ReceptorQualifier,
    ReceptorToolchain,
    RedockingValidation,
    StructureSource,
    TargetStructureRequest,
    ToolCapability,
    ValidationStatus,
    parse_legacy_rcsb_pdb,
    structure_candidate_from_dict,
)


ROOT = Path(__file__).resolve().parents[2]
FLT3_FIXTURE = ROOT / "tests" / "fixtures" / "receptor_pipeline" / "flt3_structure_contract.json"
IDH1_PDB = ROOT / "data" / "docking" / "IDH1" / "source" / "6U4J.pdb"
IDH1_PROFILE = ROOT / "data" / "docking" / "IDH1" / "6U4J-PWV-A503" / "profile.json"


def _flt3_fixture():
    payload = json.loads(FLT3_FIXTURE.read_text(encoding="utf-8"))
    request = TargetStructureRequest(**payload["request"])
    candidates = [structure_candidate_from_dict(item) for item in payload["rcsb_candidates"]]
    alphafold = structure_candidate_from_dict(payload["alphafold_candidate"])
    return payload, request, candidates, alphafold


def _ready_toolchain() -> ReceptorToolchain:
    return ReceptorToolchain(
        capabilities=tuple(
            ToolCapability(tool_id=name, available=True, version="test-contract")
            for name in ReceptorToolchain.REQUIRED_TOOLS
        ),
        probed_at="2026-08-21T00:00:00+00:00",
    )


def _write_artifacts(tmp_path: Path) -> tuple[FileArtifact, ...]:
    artifacts = []
    for role in (
        "source_structure",
        "pdbfixer_receptor",
        "meeko_receptor_pdbqt",
        "posebusters_report",
        "redocking_report",
    ):
        path = tmp_path / f"{role}.txt"
        path.write_text(f"auditable {role}\n", encoding="utf-8")
        artifacts.append(FileArtifact(role=role, path=path, produced_by="contract-test"))
    return tuple(artifacts)


def test_existing_idh1_6u4j_is_exact_mutant_holo_but_not_production_ready() -> None:
    candidate = parse_legacy_rcsb_pdb(
        IDH1_PDB,
        structure_id="6U4J",
        component_roles={
            "PWV": ComponentRole.BINDING_LIGAND,
            "NAP": ComponentRole.COFACTOR,
            "CL": ComponentRole.ION,
            "FLC": ComponentRole.ADDITIVE,
        },
        box_ligand_components=("PWV",),
    )
    selection = ReceptorQualifier().select(
        TargetStructureRequest(
            target_symbol="IDH1",
            uniprot_accession="O75874",
            required_mutations=("R132H",),
        ),
        (candidate,),
    )

    assert selection.selected.structure_id == "6U4J"
    assert selection.selected.observed_mutations == ("R132H",)
    assert selection.selected.quality.resolution_angstrom == pytest.approx(2.11)
    assert selection.selected.quality.r_free == pytest.approx(0.22)
    assert selection.selected.waters == 860
    assert len(selection.selected.missing_residues) == 23
    assert {item.component_id for item in selection.selected.box_ligands} == {"PWV"}
    assert {item.role for item in selection.selected.components} >= {
        ComponentRole.BINDING_LIGAND,
        ComponentRole.COFACTOR,
        ComponentRole.ION,
        ComponentRole.ADDITIVE,
    }

    existing = json.loads(IDH1_PROFILE.read_text(encoding="utf-8"))["profiles"][0]
    assert existing["profile_id"] == "IDH1-R132H-6U4J-PWV-A503-v1"
    assert existing["redocking_validation"]["passed"] is True
    assert existing["production_ready"] is False
    assert existing["profile_status"] == "experimental_smoke_only"


def test_idh1_wild_type_request_rejects_6u4j_r132h() -> None:
    candidate = parse_legacy_rcsb_pdb(
        IDH1_PDB,
        structure_id="6U4J",
        component_roles={"PWV": ComponentRole.BINDING_LIGAND},
        box_ligand_components=("PWV",),
    )
    ranked, rejected = ReceptorQualifier().qualify_rcsb(
        TargetStructureRequest("IDH1", "O75874"),
        (candidate,),
    )
    assert ranked == ()
    assert rejected[0].reason_codes == ("MUTATION_SET_MISMATCH",)


def test_flt3_contract_ranks_explicit_quality_and_rejects_mutation_mismatch() -> None:
    payload, request, candidates, _ = _flt3_fixture()
    assert payload["fixture"] is True
    assert payload["purpose"].endswith("not a live scientific result")

    ranked, rejected = ReceptorQualifier().qualify_rcsb(request, candidates)
    assert [item.candidate.structure_id for item in ranked] == [
        "FIX-FLT3-HIGH",
        "FIX-FLT3-LOW",
    ]
    assert ranked[0].rank_vector["resolution_angstrom"] == 2.0
    assert ranked[0].rank_vector["r_free"] == 0.22
    mismatch = next(item for item in rejected if item.structure_id.endswith("MUTATION-MISMATCH"))
    assert "MUTATION_SET_MISMATCH" in mismatch.reason_codes
    selection = ReceptorQualifier().select(request, candidates)
    assert selection.production_eligible_source is False


def test_explicit_isoform_mismatch_is_not_accepted() -> None:
    _, request, candidates, _ = _flt3_fixture()
    wrong_isoform = replace(candidates[0], isoform_accession="P36888-2")
    ranked, rejected = ReceptorQualifier().qualify_rcsb(request, (wrong_isoform,))
    assert ranked == ()
    assert rejected[0].reason_codes == ("UNIPROT_ISOFORM_MISMATCH",)


def test_alphafold_is_typed_fallback_and_never_production_eligible() -> None:
    _, request, candidates, alphafold = _flt3_fixture()
    selection = ReceptorQualifier().select(
        request,
        (candidates[2],),
        alphafold_candidate=alphafold,
    )
    assert selection.selected.source is StructureSource.ALPHAFOLD_PREDICTED
    assert selection.fallback_reason is FallbackReason.NO_EXACT_RCSB_HOLO
    assert selection.production_eligible_source is False


def test_alphafold_does_not_substitute_for_requested_mutant() -> None:
    _, _, _, alphafold = _flt3_fixture()
    mutant_request = TargetStructureRequest(
        "FLT3",
        "P36888",
        isoform_accession="P36888-1",
        required_mutations=("D835Y",),
    )
    with pytest.raises(ReceptorPipelineError) as captured:
        ReceptorQualifier().select(
            mutant_request,
            (),
            alphafold_candidate=alphafold,
        )
    assert captured.value.code == "ALPHAFOLD_TARGET_MISMATCH"
    assert captured.value.details["reason_codes"] == ["MUTATION_SET_MISMATCH"]


def test_missing_preparation_tool_fails_before_backend_call() -> None:
    _, request, candidates, _ = _flt3_fixture()
    selection = ReceptorQualifier().select(request, candidates)
    toolchain = ReceptorToolchain(
        capabilities=(
            ToolCapability(
                tool_id="pdbfixer",
                available=False,
                version=None,
                reason="not installed",
            ),
        ),
        probed_at="2026-08-21T00:00:00+00:00",
    )

    class Backend:
        called = False

        def prepare(self, selection, toolchain):  # pragma: no cover - must not run
            self.called = True
            raise AssertionError("backend must not be invoked")

    backend = Backend()
    with pytest.raises(ReceptorPipelineError) as captured:
        ReceptorPreparationPipeline().prepare_profile(
            profile_id="FLT3-contract-v1",
            profile_version="1.0",
            selection=selection,
            toolchain=toolchain,
            backend=backend,
        )
    assert captured.value.code == "RECEPTOR_PREPARATION_TOOLCHAIN_UNAVAILABLE"
    assert set(captured.value.details["missing_tools"]) == set(
        ReceptorToolchain.REQUIRED_TOOLS
    )
    assert backend.called is False


def test_candidate_profile_hashes_every_artifact_and_requires_acceptance(tmp_path: Path) -> None:
    _, request, candidates, _ = _flt3_fixture()
    selection = ReceptorQualifier().select(request, candidates)
    prepared = PreparedReceptorRun(
        pocket=DockingPocket(
            origin=PocketOrigin.COCRYSTAL_LIGAND,
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            ligand_instance_id="TST:A:900",
        ),
        artifacts=_write_artifacts(tmp_path),
        redocking=RedockingValidation(
            status=ValidationStatus.PASSED,
            threshold_angstrom=2.0,
            top_pose_rmsd_angstrom=1.1,
            validation_artifact_role="redocking_report",
            method="symmetry-aware heavy-atom RMSD",
        ),
        pose_validation=PoseValidation(
            status=ValidationStatus.PASSED,
            validator="PoseBusters",
            checked_pose_count=9,
            failure_codes=(),
            validation_artifact_role="posebusters_report",
        ),
    )
    profile = CandidateDockingProfileBuilder().build(
        profile_id="FLT3-contract-v1",
        profile_version="1.0",
        selection=selection,
        toolchain=_ready_toolchain(),
        prepared=prepared,
    )

    assert len(profile["profile_sha256"]) == 64
    assert all(len(item["sha256"]) == 64 for item in profile["artifacts"])
    assert profile["validations"]["redocking"]["passed"] is True
    assert profile["validations"]["pose"]["passed"] is True
    assert profile["production_ready"] is False
    assert "PRODUCTION_ACCEPTANCE_MISSING" in profile["production_readiness_reasons"]
    assert "FIXTURE_STRUCTURE_NOT_PRODUCTION_ELIGIBLE" in profile["production_readiness_reasons"]


def test_alphafold_profile_requires_predicted_label_and_stays_not_ready(tmp_path: Path) -> None:
    _, request, candidates, alphafold = _flt3_fixture()
    selection = ReceptorQualifier().select(
        request,
        (candidates[2],),
        alphafold_candidate=alphafold,
    )
    p2rank = tmp_path / "p2rank_predictions.csv"
    p2rank.write_text("rank,score\n1,7.2\n", encoding="utf-8")
    artifacts = (*_write_artifacts(tmp_path), FileArtifact("p2rank_output", p2rank, "P2Rank"))
    prepared = PreparedReceptorRun(
        pocket=DockingPocket(
            origin=PocketOrigin.P2RANK_PREDICTED,
            center=(1.0, 2.0, 3.0),
            size=(20.0, 20.0, 20.0),
            prediction_label="predicted",
            prediction_score=7.2,
            source_artifact_role="p2rank_output",
        ),
        artifacts=artifacts,
        redocking=RedockingValidation(
            status=ValidationStatus.NOT_APPLICABLE,
            threshold_angstrom=None,
            top_pose_rmsd_angstrom=None,
            validation_artifact_role=None,
            method="No co-crystal reference is available for a predicted pocket",
        ),
        pose_validation=PoseValidation(
            status=ValidationStatus.PASSED,
            validator="PoseBusters",
            checked_pose_count=1,
            failure_codes=(),
            validation_artifact_role="posebusters_report",
        ),
    )
    profile = CandidateDockingProfileBuilder().build(
        profile_id="AF-FLT3-contract-v1",
        profile_version="1.0",
        selection=selection,
        toolchain=_ready_toolchain(),
        prepared=prepared,
        acceptance=ProductionAcceptance(
            accepted=True,
            reviewer="contract-reviewer",
            accepted_at="2026-08-21T00:00:00+00:00",
            protocol_version="contract-v1",
            benchmark_scope="contract only",
        ),
    )
    assert profile["production_ready"] is False
    assert "ALPHAFOLD_FALLBACK_NOT_PRODUCTION_ELIGIBLE" in profile[
        "production_readiness_reasons"
    ]
    assert "REDOCKING_NOT_AVAILABLE" in profile["production_readiness_reasons"]
