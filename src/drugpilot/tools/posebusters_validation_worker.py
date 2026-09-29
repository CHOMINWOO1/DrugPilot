from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from typing import Any

from drugpilot.tools.pose_rmsd import (
    calculate_redocking_rmsd,
    write_crystal_reference_sdf,
)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _flat_label(value: Any) -> str:
    if isinstance(value, tuple):
        return "::".join(str(item) for item in value if str(item))
    return str(value)


def _top_pose_sdf(docked_pdbqt: Path, output_sdf: Path) -> int:
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit import Chem
    except (ImportError, OSError) as exc:
        raise RuntimeError("Meeko and RDKit are required to reconstruct a Vina pose") from exc
    pdbqt = PDBQTMolecule.from_file(str(docked_pdbqt), skip_typing=True)
    molecules = [item for item in RDKitMolCreate.from_pdbqt_mol(pdbqt) if item is not None]
    if len(molecules) != 1 or molecules[0].GetNumConformers() < 1:
        raise ValueError("Meeko did not reconstruct exactly one docked ligand graph")
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    try:
        writer.write(molecules[0], confId=0)
    finally:
        writer.close()
    if not output_sdf.is_file() or output_sdf.stat().st_size == 0:
        raise RuntimeError("RDKit did not write the top-ranked docked pose SDF")
    return molecules[0].GetNumConformers()


def validate(
    *,
    docked_pdbqt: Path,
    receptor_pdb: Path,
    output_sdf: Path,
    posebusters_report_path: Path,
    mode: str,
    reference_sdf: Path | None,
    crystal_ligand_pdb: Path | None,
    posebusters_reference_output: Path | None,
    redocking_report_path: Path | None,
    rmsd_threshold: float,
) -> dict[str, Any]:
    try:
        from posebusters import PoseBusters
        from pandas.api.types import is_bool_dtype
    except (ImportError, OSError) as exc:
        raise RuntimeError("PoseBusters and pandas are required for docked-pose validation") from exc
    if mode not in {"redock", "dock"}:
        raise ValueError("mode must be redock or dock")
    if mode == "redock" and (reference_sdf is None or crystal_ligand_pdb is None):
        raise ValueError("Redocking validation requires reference SDF and crystal ligand PDB")

    mode_count = _top_pose_sdf(docked_pdbqt, output_sdf)
    posebusters_reference = reference_sdf
    reference_provenance = None
    if mode == "redock":
        assert reference_sdf is not None
        assert crystal_ligand_pdb is not None
        if posebusters_reference_output is None:
            raise ValueError("posebusters_reference_output is required in redock mode")
        reference_provenance = write_crystal_reference_sdf(
            reference_component_sdf=reference_sdf,
            crystal_ligand_pdb=crystal_ligand_pdb,
            output_sdf=posebusters_reference_output,
        )
        posebusters_reference = posebusters_reference_output
    buster = PoseBusters(config=mode, top_n=1, max_workers=0)
    table = buster.bust(
        [output_sdf],
        posebusters_reference if mode == "redock" else None,
        receptor_pdb,
        full_report=False,
    )
    if len(table.index) != 1:
        raise ValueError("PoseBusters did not return exactly one top-pose row")
    checks: dict[str, bool] = {}
    for column in table.columns:
        value = table.iloc[0][column]
        # NumPy 2.x may expose scalar booleans as ``numpy.bool`` rather than
        # Python ``bool``/legacy ``bool_``.  Column dtype is the stable pandas
        # contract and avoids silently dropping every validation check.
        if is_bool_dtype(table[column].dtype):
            checks[_flat_label(column)] = bool(value)
    if not checks:
        raise ValueError("PoseBusters report contained no boolean validation checks")
    failures = sorted(name for name, passed in checks.items() if not passed)
    pose_payload = {
        "passed": not failures,
        "mode": mode,
        "checked_pose_count": 1,
        "available_vina_mode_count": mode_count,
        "checks": checks,
        "failure_codes": failures,
        "posebusters_version": _version("posebusters"),
        "rdkit_version": _version("rdkit"),
        "meeko_version": _version("meeko"),
        "reference_provenance": reference_provenance,
    }
    posebusters_report_path.parent.mkdir(parents=True, exist_ok=True)
    posebusters_report_path.write_text(
        json.dumps(pose_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    redocking_payload = None
    if mode == "redock":
        assert reference_sdf is not None
        assert crystal_ligand_pdb is not None
        if redocking_report_path is None:
            raise ValueError("redocking_report_path is required in redock mode")
        redocking = calculate_redocking_rmsd(
            docked_pdbqt=docked_pdbqt,
            reference_component_sdf=reference_sdf,
            crystal_ligand_pdb=crystal_ligand_pdb,
            threshold_angstrom=rmsd_threshold,
        )
        redocking_payload = redocking.as_dict()
        redocking_report_path.parent.mkdir(parents=True, exist_ok=True)
        redocking_report_path.write_text(
            json.dumps(redocking_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
    return {"posebusters": pose_payload, "redocking": redocking_payload}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PoseBusters and optional redocking RMSD")
    parser.add_argument("--docked", required=True, type=Path)
    parser.add_argument("--receptor", required=True, type=Path)
    parser.add_argument("--output-sdf", required=True, type=Path)
    parser.add_argument("--posebusters-report", required=True, type=Path)
    parser.add_argument("--mode", choices=("redock", "dock"), required=True)
    parser.add_argument("--reference-sdf", type=Path)
    parser.add_argument("--crystal-ligand-pdb", type=Path)
    parser.add_argument("--posebusters-reference-output", type=Path)
    parser.add_argument("--redocking-report", type=Path)
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            docked_pdbqt=args.docked,
            receptor_pdb=args.receptor,
            output_sdf=args.output_sdf,
            posebusters_report_path=args.posebusters_report,
            mode=args.mode,
            reference_sdf=args.reference_sdf if args.reference_sdf else None,
            crystal_ligand_pdb=(
                args.crystal_ligand_pdb if args.crystal_ligand_pdb else None
            ),
            posebusters_reference_output=(
                args.posebusters_reference_output
                if args.posebusters_reference_output
                else None
            ),
            redocking_report_path=(
                args.redocking_report if args.redocking_report else None
            ),
            rmsd_threshold=args.rmsd_threshold,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"error_type": type(exc).__name__, "detail": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
