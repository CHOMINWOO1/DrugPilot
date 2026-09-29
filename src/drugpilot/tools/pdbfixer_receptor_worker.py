from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from typing import Any


_STANDARD_POLYMER_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "A",
    "C",
    "G",
    "I",
    "U",
    "DA",
    "DC",
    "DG",
    "DI",
    "DT",
    "UNK",
}


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _instance(residue_name: str, chain_id: str, residue_id: str) -> str:
    return f"{residue_name.upper()}:{chain_id}:{int(residue_id)}"


def _input_polymer_positions(path: Path) -> set[tuple[str, int]]:
    positions: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith("ATOM  ") and len(line) >= 27:
            positions.add((line[21].strip(), int(line[22:26].strip())))
            continue
        # Include DBREF positions so explicitly enabled missing-residue builds
        # are not deleted as if they were heterogens.
        if line.startswith("DBREF "):
            fields = line.split()
            if len(fields) >= 5:
                chain, start, end = fields[2], int(fields[3]), int(fields[4])
                positions.update((chain, value) for value in range(start, end + 1))
    return positions


def prepare_receptor(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    chains: set[str],
    retain_components: set[str],
    retain_waters: set[str],
    replace_nonstandard_residues: bool,
    add_missing_residues: bool,
    add_hydrogens: bool,
    ph: float,
    seed: int,
) -> dict[str, Any]:
    try:
        from pdbfixer import PDBFixer
        from openmm import version as openmm_version
        from openmm.app import Modeller, PDBFile
    except (ImportError, OSError) as exc:
        raise RuntimeError("PDBFixer and OpenMM are required for receptor repair") from exc

    if not input_path.is_file():
        raise FileNotFoundError(input_path.name)
    if not chains:
        raise ValueError("At least one exact protein chain is required")
    if not 0.0 < ph < 14.0:
        raise ValueError("pH must be between 0 and 14")
    if seed < 1:
        raise ValueError("A positive PDBFixer seed is required")

    polymer_positions = _input_polymer_positions(input_path)
    fixer = PDBFixer(filename=str(input_path))
    input_chain_ids = [chain.id for chain in fixer.topology.chains()]
    missing_chains = sorted(chains - set(input_chain_ids))
    if missing_chains:
        raise ValueError(f"Requested chains are absent from the input: {missing_chains}")
    remove_chain_ids = sorted(set(input_chain_ids) - chains)
    if remove_chain_ids:
        fixer.removeChains(chainIds=remove_chain_ids)

    fixer.findMissingResidues()
    chain_by_index = {chain.index: chain.id for chain in fixer.topology.chains()}
    missing_residues = {
        f"{chain_by_index.get(chain_index, chain_index)}:{residue_index}": [
            str(name) for name in residue_names
        ]
        for (chain_index, residue_index), residue_names in fixer.missingResidues.items()
    }
    if not add_missing_residues:
        fixer.missingResidues = {}

    fixer.findNonstandardResidues()
    nonstandard = [
        {
            "residue": _instance(residue.name, residue.chain.id, residue.id),
            "replacement": str(replacement),
        }
        for residue, replacement in fixer.nonstandardResidues
    ]
    if nonstandard and not replace_nonstandard_residues:
        raise ValueError(
            "Non-standard polymer residues require an explicit replacement policy"
        )
    if nonstandard:
        fixer.replaceNonstandardResidues()

    fixer.findMissingAtoms()
    missing_atoms = {
        _instance(residue.name, residue.chain.id, residue.id): [
            str(atom.name) for atom in atoms
        ]
        for residue, atoms in fixer.missingAtoms.items()
    }
    missing_terminal_atoms = {
        _instance(residue.name, residue.chain.id, residue.id): [str(name) for name in names]
        for residue, names in fixer.missingTerminals.items()
    }
    fixer.addMissingAtoms(seed=seed)

    modeller = Modeller(fixer.topology, fixer.positions)
    deleted: list[str] = []
    retained: list[str] = []
    to_delete = []
    for residue in modeller.topology.residues():
        instance = _instance(residue.name, residue.chain.id, residue.id)
        if residue.chain.id not in chains:
            to_delete.append(residue)
            deleted.append(instance)
            continue
        if (
            (residue.chain.id, int(residue.id)) in polymer_positions
            or residue.name.upper() in _STANDARD_POLYMER_RESIDUES
        ):
            retained.append(instance)
            continue
        if residue.name.upper() in {"HOH", "WAT"}:
            if instance in retain_waters:
                retained.append(instance)
            else:
                to_delete.append(residue)
                deleted.append(instance)
            continue
        if instance in retain_components:
            retained.append(instance)
        else:
            to_delete.append(residue)
            deleted.append(instance)
    if to_delete:
        modeller.delete(to_delete)
    if add_hydrogens:
        # This intentionally fails when retained non-polymers do not have a
        # usable template; no partial or neutral receptor is written.
        modeller.addHydrogens(pH=ph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("PDBFixer/OpenMM did not write a repaired receptor")

    payload = {
        "pdbfixer_version": _version("pdbfixer"),
        "openmm_version": getattr(openmm_version, "full_version", None),
        "input_chain_ids": input_chain_ids,
        "chains": sorted(chains),
        "retained_component_instances": sorted(retain_components),
        "retained_water_instances": sorted(retain_waters),
        "retained_residue_instances": sorted(set(retained)),
        "deleted_residue_instances": sorted(set(deleted)),
        "detected_missing_residues": missing_residues,
        "missing_residues_added": add_missing_residues,
        "detected_nonstandard_residues": nonstandard,
        "nonstandard_residues_replaced": replace_nonstandard_residues,
        "detected_missing_atoms": missing_atoms,
        "detected_missing_terminal_atoms": missing_terminal_atoms,
        "missing_atoms_added": True,
        "hydrogens_added": add_hydrogens,
        "ph": ph,
        "seed": seed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair a receptor with PDBFixer/OpenMM")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--retain-component", action="append", default=[])
    parser.add_argument("--retain-water", action="append", default=[])
    parser.add_argument("--replace-nonstandard-residues", action="store_true")
    parser.add_argument("--add-missing-residues", action="store_true")
    parser.add_argument("--add-hydrogens", action="store_true")
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_receptor(
            input_path=args.input.resolve(),
            output_path=args.output.resolve(),
            report_path=args.report.resolve(),
            chains={item.strip() for item in args.chains.split(",") if item.strip()},
            retain_components={item.strip() for item in args.retain_component if item.strip()},
            retain_waters={item.strip() for item in args.retain_water if item.strip()},
            replace_nonstandard_residues=args.replace_nonstandard_residues,
            add_missing_residues=args.add_missing_residues,
            add_hydrogens=args.add_hydrogens,
            ph=args.ph,
            seed=args.seed,
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
