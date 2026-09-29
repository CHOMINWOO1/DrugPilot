from __future__ import annotations

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.domain.models import ScientificExecutionStatus
from drugpilot.tools.chemistry import ChemistryValidator


class _LiveGateway:
    source_version = "chemical-standardization-test-v1"
    backend_kind = "live"
    fixture = False


def test_rdkit_standardization_removes_counterion_and_records_lineage() -> None:
    validator = ChemistryValidator(prefer_rdkit=True)
    assert validator.backend_name == "rdkit"

    report = validator.inspect("[Na+].O=C([O-])c1ccccc1")

    assert report.valid is True
    assert report.canonical_smiles == "O=C(O)c1ccccc1"
    assert report.identity_parent_smiles == report.canonical_smiles
    assert report.original_canonical_smiles != report.canonical_smiles
    assert report.tautomer_parent_smiles
    assert "salt_or_solvent_fragment_removed" in report.transformations
    assert "charge_normalized" in report.transformations
    assert report.standardization_version == "rdkit-molstandardize-parent-v1"
    assert report.provenance["fragment_count_input"] == 2
    assert report.provenance["fragment_count_parent"] == 1
    assert report.provenance["microstate_enumeration"] == "not_performed"


def test_rdkit_standardization_preserves_explicit_stereo_and_flags_unknown() -> None:
    validator = ChemistryValidator(prefer_rdkit=True)

    specified = validator.inspect("C[C@H](O)C(=O)O")
    unspecified = validator.inspect("CC(O)C(=O)O")

    assert specified.valid is True
    assert "@" in str(specified.canonical_smiles)
    assert specified.stereo_status == "specified_stereocenters"
    assert specified.warnings == ()
    assert unspecified.valid is True
    assert unspecified.stereo_status == "unspecified_stereocenters"
    assert "UNSPECIFIED_STEREOCENTER" in unspecified.warnings


def test_ma2_live_chemistry_result_exposes_standardization_provenance() -> None:
    agent = MA2Agent(gateway=_LiveGateway())

    result = agent._chemistry_result(  # noqa: SLF001 - focused contract test
        "[Na+].O=C([O-])c1ccccc1",
        input_hash="a" * 64,
    )

    assert result.status is ScientificExecutionStatus.COMPLETED
    assert result.valid is True
    assert result.canonical_smiles == "O=C(O)c1ccccc1"
    assert result.provenance["standardization_version"] == (
        "rdkit-molstandardize-parent-v1"
    )
    assert "salt_or_solvent_fragment_removed" in result.provenance["transformations"]
    assert result.provenance["original_canonical_smiles"]
    assert result.provenance["identity_parent_smiles"] == result.canonical_smiles
    assert result.provenance["tautomer_parent_smiles"]
    assert result.provenance["missing_values_imputed"] is False


def test_invalid_smiles_remains_fail_closed_without_rdkit_log_flood(capfd) -> None:
    report = ChemistryValidator(prefer_rdkit=True).inspect("c1cccc1")
    captured = capfd.readouterr()

    assert report.valid is False
    assert report.canonical_smiles is None
    assert report.errors == ("RDKit could not parse SMILES",)
    assert "kekul" not in captured.err.casefold()
    assert captured.err == ""
