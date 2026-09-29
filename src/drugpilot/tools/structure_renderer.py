from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import re
from typing import Any, Final


MAX_SMILES_LENGTH: Final = 512
MAX_ATOM_COUNT: Final = 256
SVG_WIDTH: Final = 360
SVG_HEIGHT: Final = 240
_MAX_SVG_BYTES: Final = 1_000_000
_ALLOWED_SMILES = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/.:%]+$")
_UNSAFE_SVG = re.compile(
    r"<\s*(?:script|foreignObject)\b|javascript\s*:|\bon[a-z]+\s*=",
    re.IGNORECASE,
)


class InvalidStructureInput(ValueError):
    """The supplied value cannot be treated as a bounded molecule input."""


class StructureToolkitUnavailable(RuntimeError):
    """The required local 2D chemistry toolkit is not available."""


class StructureRenderingError(RuntimeError):
    """A validated structure could not be rendered safely."""


@dataclass(frozen=True, slots=True)
class StructureDrawing:
    svg: str
    canonical_smiles: str
    etag: str


def _rdkit_modules() -> tuple[Any, Any]:
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
        from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise StructureToolkitUnavailable(
            "Local 2D structure rendering is unavailable."
        ) from exc
    return Chem, rdMolDraw2D


def _canonicalize_smiles(smiles: str) -> str:
    if not isinstance(smiles, str):
        raise InvalidStructureInput("Invalid SMILES input.")
    compact = smiles.strip()
    if (
        not compact
        or len(compact) > MAX_SMILES_LENGTH
        or any(character.isspace() for character in compact)
        or _ALLOWED_SMILES.fullmatch(compact) is None
    ):
        raise InvalidStructureInput("Invalid SMILES input.")

    Chem, _ = _rdkit_modules()
    try:
        molecule = Chem.MolFromSmiles(compact)
    except Exception as exc:
        raise InvalidStructureInput("Invalid SMILES input.") from exc
    if (
        molecule is None
        or molecule.GetNumAtoms() < 1
        or molecule.GetNumAtoms() > MAX_ATOM_COUNT
    ):
        raise InvalidStructureInput("Invalid SMILES input.")
    try:
        canonical = str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    except Exception as exc:
        raise InvalidStructureInput("Invalid SMILES input.") from exc
    if not canonical or len(canonical) > MAX_SMILES_LENGTH:
        raise InvalidStructureInput("Invalid SMILES input.")
    return canonical


@lru_cache(maxsize=256)
def _render_canonical_smiles_svg(canonical_smiles: str) -> StructureDrawing:
    Chem, rdMolDraw2D = _rdkit_modules()
    try:
        molecule = Chem.MolFromSmiles(canonical_smiles)
        if molecule is None:
            raise ValueError("canonical SMILES could not be parsed")
        drawer = rdMolDraw2D.MolDraw2DSVG(SVG_WIDTH, SVG_HEIGHT)
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        svg = str(drawer.GetDrawingText())
    except Exception as exc:
        raise StructureRenderingError(
            "The 2D structure could not be rendered."
        ) from exc

    encoded = svg.encode("utf-8")
    if (
        len(encoded) > _MAX_SVG_BYTES
        or "<svg" not in svg.lower()
        or _UNSAFE_SVG.search(svg) is not None
    ):
        raise StructureRenderingError("The 2D structure could not be rendered safely.")
    return StructureDrawing(
        svg=svg,
        canonical_smiles=canonical_smiles,
        etag=sha256(encoded).hexdigest(),
    )


def render_structure_svg(smiles: str) -> StructureDrawing:
    """Validate a bounded SMILES value and render a cached local RDKit SVG.

    No syntax-only or remote-image fallback is permitted: if RDKit is missing
    or rendering fails, callers receive a typed fail-closed error.
    """

    return _render_canonical_smiles_svg(_canonicalize_smiles(smiles))

