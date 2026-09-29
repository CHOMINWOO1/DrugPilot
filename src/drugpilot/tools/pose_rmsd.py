from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from math import sqrt
from pathlib import Path
import re
from typing import Any


_VINA_RESULT = re.compile(
    r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class PoseModeRMSD:
    mode: int
    affinity_kcal_mol: float
    heavy_atom_rmsd_angstrom: float


@dataclass(frozen=True, slots=True)
class RedockingValidationResult:
    passed: bool
    threshold_angstrom: float
    top_ranked_pose_rmsd_angstrom: float
    best_pose_rmsd_angstrom: float
    best_pose_mode: int
    heavy_atom_count: int
    symmetry_mapping_count: int
    coordinate_frame: str
    mapping_method: str
    alignment_applied: bool
    modes: tuple[PoseModeRMSD, ...]
    source_hashes: dict[str, str]
    software_versions: dict[str, str | None]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modes"] = [asdict(mode) for mode in self.modes]
        return payload


def _crystal_coordinates(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    elements: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 78:
            continue
        if line[16] not in {" ", "A"}:
            continue
        element = line[76:78].strip().title()
        if not element or element == "H":
            continue
        elements.append(element)
        coordinates.append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        )
    if not coordinates:
        raise ValueError("The crystal-ligand PDB contains no heavy-atom coordinates")
    return elements, coordinates


def write_crystal_reference_sdf(
    *,
    reference_component_sdf: str | Path,
    crystal_ligand_pdb: str | Path,
    output_sdf: str | Path,
) -> dict[str, Any]:
    """Write an exact reference graph in the experimental coordinate frame.

    RCSB chemical-component SDF files usually contain idealized coordinates.
    Pose-validation software must instead compare a redocked pose with the
    bound ligand coordinates extracted from the selected structure.  This
    function preserves the component graph/stereochemistry, verifies the full
    heavy-atom element sequence, replaces its coordinates with the crystal
    coordinates, and writes an auditable SDF for downstream validators.
    """

    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except (ImportError, OSError) as exc:
        raise RuntimeError("RDKit is required to build a crystal reference SDF") from exc

    source = Path(reference_component_sdf)
    crystal = Path(crystal_ligand_pdb)
    destination = Path(output_sdf)
    for path in (source, crystal):
        if not path.is_file():
            raise FileNotFoundError(path.name)
    mol_block = source.read_text(encoding="utf-8", errors="strict").split("$$$$", 1)[0]
    template = Chem.MolFromMolBlock(
        mol_block,
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    if template is None:
        raise ValueError("RDKit could not parse the reference component SDF")
    reference = Chem.RemoveHs(template)
    expected_elements = [atom.GetSymbol() for atom in reference.GetAtoms()]
    crystal_elements, coordinates = _crystal_coordinates(crystal)
    if crystal_elements != expected_elements:
        raise ValueError(
            "Reference SDF and crystal PDB heavy-atom sequences do not match exactly"
        )
    conformer = reference.GetConformer()
    for index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(index, Point3D(*coordinate))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(destination))
    try:
        writer.write(reference, confId=0)
    finally:
        writer.close()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("RDKit did not write the crystal reference SDF")
    return {
        "heavy_atom_count": reference.GetNumAtoms(),
        "reference_component_sdf_sha256": _digest(source),
        "crystal_ligand_pdb_sha256": _digest(crystal),
        "crystal_reference_sdf_sha256": _digest(destination),
        "coordinate_frame": "selected source co-crystal coordinates",
    }


def calculate_redocking_rmsd(
    *,
    docked_pdbqt: str | Path,
    reference_component_sdf: str | Path,
    crystal_ligand_pdb: str | Path,
    threshold_angstrom: float = 2.0,
) -> RedockingValidationResult:
    """Calculate symmetry-aware heavy-atom RMSD in the receptor coordinate frame.

    The RCSB chemical-component SDF supplies bond orders and stereochemistry.
    Its heavy-atom sequence is accepted only when the complete element sequence
    exactly matches the extracted co-crystal PDB instance.  RDKit then finds all
    chirality-aware graph isomorphisms between that reference and the Meeko-
    reconstructed Vina poses.  RMSD is calculated without superposition because
    both pose sets already share the prepared receptor coordinate frame.
    """

    if threshold_angstrom <= 0:
        raise ValueError("The RMSD threshold must be positive")
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except (ImportError, OSError) as exc:
        raise RuntimeError("RDKit and Meeko are required for pose RMSD validation") from exc

    docked_path = Path(docked_pdbqt).resolve()
    sdf_path = Path(reference_component_sdf).resolve()
    crystal_path = Path(crystal_ligand_pdb).resolve()
    for path in (docked_path, sdf_path, crystal_path):
        if not path.is_file():
            raise FileNotFoundError(path.name)

    # Parse the mol block in Python so RDKit's native file layer never has to
    # open a non-ASCII Windows workspace path.
    sdf_text = sdf_path.read_text(encoding="utf-8", errors="strict")
    mol_block = sdf_text.split("$$$$", 1)[0]
    template_with_hydrogens = Chem.MolFromMolBlock(
        mol_block,
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    if template_with_hydrogens is None:
        raise ValueError("RDKit could not parse the reference component SDF")
    reference = Chem.RemoveHs(template_with_hydrogens)
    reference_elements = [atom.GetSymbol() for atom in reference.GetAtoms()]
    crystal_elements, crystal_coordinates = _crystal_coordinates(crystal_path)
    if crystal_elements != reference_elements:
        raise ValueError(
            "Reference SDF and crystal PDB heavy-atom sequences do not match exactly"
        )
    reference_conformer = reference.GetConformer()
    for index, coordinate in enumerate(crystal_coordinates):
        reference_conformer.SetAtomPosition(index, Point3D(*coordinate))

    pdbqt = PDBQTMolecule.from_file(str(docked_path), skip_typing=True)
    reconstructed = RDKitMolCreate.from_pdbqt_mol(pdbqt)
    molecules = [molecule for molecule in reconstructed if molecule is not None]
    if len(molecules) != 1:
        raise ValueError("Meeko did not reconstruct exactly one docked ligand graph")
    docked = Chem.RemoveHs(molecules[0])
    if docked.GetNumAtoms() != reference.GetNumAtoms():
        raise ValueError("Docked and reference heavy-atom counts differ")
    mappings = docked.GetSubstructMatches(
        reference,
        uniquify=False,
        useChirality=True,
        maxMatches=100_000,
    )
    if not mappings or any(len(mapping) != reference.GetNumAtoms() for mapping in mappings):
        raise ValueError("No complete chirality-aware heavy-atom graph mapping was found")

    affinities = [float(value) for value in _VINA_RESULT.findall(docked_path.read_text(
        encoding="utf-8", errors="strict"
    ))]
    if len(affinities) != docked.GetNumConformers():
        raise ValueError("Vina mode count and reconstructed conformer count differ")
    modes: list[PoseModeRMSD] = []
    atom_count = reference.GetNumAtoms()
    for conformer_index, affinity in enumerate(affinities):
        docked_conformer = docked.GetConformer(conformer_index)
        rmsd = min(
            sqrt(
                sum(
                    (
                        docked_conformer.GetAtomPosition(mapping[reference_index])
                        - reference_conformer.GetAtomPosition(reference_index)
                    ).LengthSq()
                    for reference_index in range(atom_count)
                )
                / atom_count
            )
            for mapping in mappings
        )
        modes.append(
            PoseModeRMSD(
                mode=conformer_index + 1,
                affinity_kcal_mol=affinity,
                heavy_atom_rmsd_angstrom=rmsd,
            )
        )
    best = min(modes, key=lambda mode: mode.heavy_atom_rmsd_angstrom)
    top_rmsd = modes[0].heavy_atom_rmsd_angstrom
    return RedockingValidationResult(
        passed=top_rmsd <= threshold_angstrom,
        threshold_angstrom=threshold_angstrom,
        top_ranked_pose_rmsd_angstrom=top_rmsd,
        best_pose_rmsd_angstrom=best.heavy_atom_rmsd_angstrom,
        best_pose_mode=best.mode,
        heavy_atom_count=atom_count,
        symmetry_mapping_count=len(mappings),
        coordinate_frame="prepared receptor / source co-crystal frame",
        mapping_method=(
            "exact RCSB component-SDF/PDB heavy-atom element-sequence check; "
            "complete chirality-aware RDKit graph isomorphisms; minimum raw "
            "coordinate RMSD across symmetry mappings"
        ),
        alignment_applied=False,
        modes=tuple(modes),
        source_hashes={
            "docked_pdbqt_sha256": _digest(docked_path),
            "reference_component_sdf_sha256": _digest(sdf_path),
            "crystal_ligand_pdb_sha256": _digest(crystal_path),
        },
        software_versions={
            "rdkit": _version("rdkit"),
            "meeko": _version("meeko"),
        },
    )


__all__ = [
    "PoseModeRMSD",
    "RedockingValidationResult",
    "calculate_redocking_rmsd",
    "write_crystal_reference_sdf",
]
