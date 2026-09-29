from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def prepare_ligand(smiles: str, output_path: Path, *, seed: int) -> dict[str, object]:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
    except (ImportError, OSError) as exc:
        raise RuntimeError("RDKit is required for ligand preparation") from exc
    # Invalid input remains a structured ValueError below; the isolated worker
    # does not also emit the same parser/kekulization diagnostic to stderr.
    RDLogger.DisableLog("rdApp.error")
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except (ImportError, OSError) as exc:
        raise RuntimeError("Meeko is required for ligand preparation") from exc

    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("RDKit could not parse the ligand SMILES")
    canonical_smiles = Chem.MolToSmiles(base, canonical=True, isomericSmiles=True)
    try:
        inchikey = Chem.MolToInchiKey(base) or None
    except Exception:
        inchikey = None

    molecule = Chem.AddHs(base)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.useRandomCoords = False
    status = AllChem.EmbedMolecule(molecule, parameters)
    if status != 0:
        raise RuntimeError(f"RDKit ETKDG embedding failed with status {status}")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        force_field = "MMFF94"
        optimization_status = AllChem.MMFFOptimizeMolecule(
            molecule,
            maxIters=500,
        )
    else:
        force_field = "UFF"
        optimization_status = AllChem.UFFOptimizeMolecule(
            molecule,
            maxIters=500,
        )
    if optimization_status < 0:
        raise RuntimeError("RDKit force-field optimization failed")

    preparator = MoleculePreparation()
    setups = preparator.prepare(molecule)
    if not setups:
        raise RuntimeError("Meeko produced no ligand setup")
    pdbqt_string, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"Meeko PDBQT export failed: {error}")
    if not pdbqt_string.strip():
        raise RuntimeError("Meeko produced an empty PDBQT document")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pdbqt_string, encoding="utf-8", newline="\n")
    return {
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "rdkit_version": _package_version("rdkit"),
        "meeko_version": _package_version("meeko"),
        "embedding": "ETKDGv3",
        "random_seed": seed,
        "force_field": force_field,
        "optimization_status": optimization_status,
        "atom_count_with_hydrogens": molecule.GetNumAtoms(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one ligand PDBQT for Vina")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_ligand(args.smiles, args.output, seed=args.seed)
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


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
