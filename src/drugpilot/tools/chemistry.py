from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Mapping

from drugpilot.domain.models import CandidateMetrics


_ATOM_PATTERN = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|[BCNOFPSIK]|[bcnops]")
_ALLOWED_SMILES = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/.:%]+$")
_ATOMIC_WEIGHTS = {
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90,
    "Si": 28.085,
    "Na": 22.990,
    "Li": 6.94,
    "Mg": 24.305,
    "Ca": 40.078,
    "Al": 26.982,
}

_STANDARDIZATION_VERSION = "rdkit-molstandardize-parent-v1"


@dataclass(frozen=True, slots=True)
class ChemistryReport:
    valid: bool
    canonical_smiles: str | None
    inchikey: str | None
    metrics: CandidateMetrics
    backend: str
    errors: tuple[str, ...] = ()
    original_canonical_smiles: str | None = None
    identity_parent_smiles: str | None = None
    tautomer_parent_smiles: str | None = None
    stereo_status: str = "not_assessed"
    transformations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    standardization_version: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


class ChemistryValidator:
    """SMILES validation with RDKit when present and a deterministic fallback.

    The fallback is intentionally a syntax/descriptor approximation for mock
    and UI tests.  It is never represented as a replacement for a chemistry
    toolkit in live mode.
    """

    def __init__(self, *, prefer_rdkit: bool = True) -> None:
        self._chem: Any | None = None
        self._descriptors: Any | None = None
        self._crr: Any | None = None
        self._lipinski: Any | None = None
        self._standardizer: Any | None = None
        self._rd_base: Any | None = None
        if prefer_rdkit:
            try:
                from rdkit import Chem, rdBase  # type: ignore[import-not-found]
                from rdkit.Chem import Crippen, Descriptors, Lipinski  # type: ignore[import-not-found]
                from rdkit.Chem.MolStandardize import (  # type: ignore[import-not-found]
                    rdMolStandardize,
                )

                self._chem = Chem
                self._descriptors = Descriptors
                self._crr = Crippen
                self._lipinski = Lipinski
                self._standardizer = rdMolStandardize
                self._rd_base = rdBase
            except (ImportError, OSError):
                pass

    @property
    def backend_name(self) -> str:
        if self._chem is None:
            return "syntax-fallback"
        if self._standardizer is None:
            return "rdkit-without-molstandardize"
        return "rdkit"

    @property
    def backend_version(self) -> str | None:
        if self._chem is None:
            return "drugpilot-syntax-fallback-v1"
        try:
            from rdkit import rdBase  # type: ignore[import-not-found]

            return str(rdBase.rdkitVersion)
        except (ImportError, AttributeError):  # pragma: no cover - optional build detail
            return None

    def inspect(self, smiles: str) -> ChemistryReport:
        if self._chem is not None:
            return self._inspect_rdkit(smiles)
        return self._inspect_fallback(smiles)

    def _inspect_rdkit(self, smiles: str) -> ChemistryReport:
        assert self._chem is not None
        compact = "".join(smiles.split())
        if self._rd_base is not None:
            with self._rd_base.BlockLogs():
                mol = self._chem.MolFromSmiles(compact)
        else:  # pragma: no cover - optional legacy RDKit build
            mol = self._chem.MolFromSmiles(compact)
        if mol is None:
            return ChemistryReport(
                valid=False,
                canonical_smiles=None,
                inchikey=None,
                metrics=CandidateMetrics(),
                backend="rdkit",
                errors=("RDKit could not parse SMILES",),
            )
        if self._standardizer is None:
            return ChemistryReport(
                valid=False,
                canonical_smiles=None,
                inchikey=None,
                metrics=CandidateMetrics(),
                backend="rdkit-without-molstandardize",
                errors=("RDKit MolStandardize is unavailable",),
                original_canonical_smiles=self._chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True,
                ),
            )

        original_canonical = self._chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
        transformations: list[str] = []
        warnings: list[str] = []
        try:
            blocker = (
                self._rd_base.BlockLogs()
                if self._rd_base is not None
                else None
            )
            if blocker is not None:
                blocker.__enter__()
            try:
                cleaned = self._standardizer.Cleanup(mol)
                cleaned_smiles = self._chem.MolToSmiles(
                    cleaned,
                    canonical=True,
                    isomericSmiles=True,
                )
                if cleaned_smiles != original_canonical:
                    transformations.append("functional_group_normalization")

                parent = self._standardizer.FragmentParent(cleaned)
                parent_before_charge = self._chem.MolToSmiles(
                    parent,
                    canonical=True,
                    isomericSmiles=True,
                )
                if len(self._chem.GetMolFrags(parent)) < len(self._chem.GetMolFrags(cleaned)):
                    transformations.append("salt_or_solvent_fragment_removed")

                parent = self._standardizer.Uncharger().uncharge(parent)
                self._chem.SanitizeMol(parent)
                identity_parent = self._chem.MolToSmiles(
                    parent,
                    canonical=True,
                    isomericSmiles=True,
                )
                if identity_parent != parent_before_charge:
                    transformations.append("charge_normalized")

                tautomer = self._standardizer.TautomerEnumerator().Canonicalize(parent)
                tautomer_parent = self._chem.MolToSmiles(
                    tautomer,
                    canonical=True,
                    isomericSmiles=True,
                )
                if tautomer_parent != identity_parent:
                    transformations.append("tautomer_identity_normalized")
            finally:
                if blocker is not None:
                    blocker.__exit__(None, None, None)
        except (RuntimeError, ValueError, TypeError) as exc:
            return ChemistryReport(
                valid=False,
                canonical_smiles=None,
                inchikey=None,
                metrics=CandidateMetrics(),
                backend="rdkit",
                errors=(f"RDKit standardization failed: {type(exc).__name__}",),
                original_canonical_smiles=original_canonical,
                standardization_version=_STANDARDIZATION_VERSION,
                provenance={
                    "input_smiles_sha256": sha256(compact.encode("utf-8")).hexdigest(),
                    "missing_values_imputed": False,
                },
            )

        chiral_centers = self._chem.FindMolChiralCenters(
            parent,
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
        unspecified_stereo = sum(label == "?" for _, label in chiral_centers)
        if unspecified_stereo:
            stereo_status = "unspecified_stereocenters"
            warnings.append("UNSPECIFIED_STEREOCENTER")
        elif chiral_centers:
            stereo_status = "specified_stereocenters"
        else:
            stereo_status = "no_tetrahedral_stereocenter_detected"

        canonical = identity_parent
        try:
            inchikey = self._chem.MolToInchiKey(parent) or None
        except Exception:  # pragma: no cover - depends on optional RDKit build
            inchikey = None
        rings = int(self._lipinski.RingCount(parent))
        metrics = CandidateMetrics(
            molecular_weight=round(float(self._descriptors.MolWt(parent)), 3),
            clogp=round(float(self._crr.MolLogP(parent)), 3),
            tpsa=round(float(self._descriptors.TPSA(parent)), 3),
            hbd=int(self._lipinski.NumHDonors(parent)),
            hba=int(self._lipinski.NumHAcceptors(parent)),
            synthetic_accessibility=round(
                min(10.0, 1.5 + rings * 0.45 + parent.GetNumAtoms() * 0.035),
                3,
            ),
        )
        return ChemistryReport(
            valid=True,
            canonical_smiles=canonical,
            inchikey=inchikey,
            metrics=metrics,
            backend="rdkit",
            original_canonical_smiles=original_canonical,
            identity_parent_smiles=identity_parent,
            tautomer_parent_smiles=tautomer_parent,
            stereo_status=stereo_status,
            transformations=tuple(dict.fromkeys(transformations)),
            warnings=tuple(warnings),
            standardization_version=_STANDARDIZATION_VERSION,
            provenance={
                "input_smiles_sha256": sha256(compact.encode("utf-8")).hexdigest(),
                "standardized_smiles_sha256": sha256(
                    identity_parent.encode("utf-8")
                ).hexdigest(),
                "tautomer_parent_sha256": sha256(
                    tautomer_parent.encode("utf-8")
                ).hexdigest(),
                "fragment_count_input": len(self._chem.GetMolFrags(mol)),
                "fragment_count_parent": len(self._chem.GetMolFrags(parent)),
                "stereocenter_count": len(chiral_centers),
                "unspecified_stereocenter_count": unspecified_stereo,
                "microstate_enumeration": "not_performed",
                "evaluation_structure_policy": "uncharged_fragment_parent_v1",
                "identity_tautomer_policy": "canonical_tautomer_recorded_not_scored_v1",
                "missing_values_imputed": False,
            },
        )

    def _inspect_fallback(self, smiles: str) -> ChemistryReport:
        compact = "".join(smiles.split())
        errors: list[str] = []
        if not compact:
            errors.append("SMILES is empty")
        if compact and not _ALLOWED_SMILES.fullmatch(compact):
            errors.append("SMILES contains unsupported characters")
        if compact.count("(") != compact.count(")"):
            errors.append("unbalanced parentheses")
        if compact.count("[") != compact.count("]"):
            errors.append("unbalanced brackets")
        atoms = _ATOM_PATTERN.findall(compact)
        if not atoms:
            errors.append("no recognized atom tokens")
        residue = _ATOM_PATTERN.sub("", compact)
        residue = re.sub(r"[0-9@+\-\[\]\(\)=#$\\/.:%]", "", residue)
        if residue:
            errors.append(f"unrecognized token sequence: {residue[:12]}")
        if errors:
            return ChemistryReport(
                valid=False,
                canonical_smiles=None,
                inchikey=None,
                metrics=CandidateMetrics(),
                backend="syntax-fallback",
                errors=tuple(errors),
            )

        normalized_atoms = [atom.upper() if len(atom) == 1 else atom for atom in atoms]
        carbon = sum(atom in {"C", "c"} for atom in atoms)
        nitrogen = sum(atom in {"N", "n"} for atom in atoms)
        oxygen = sum(atom in {"O", "o"} for atom in atoms)
        sulfur = sum(atom in {"S", "s"} for atom in atoms)
        halogen = sum(atom in {"F", "Cl", "Br", "I"} for atom in atoms)
        heavy_weight = sum(_ATOMIC_WEIGHTS.get(atom, _ATOMIC_WEIGHTS.get(atom.upper(), 12.0)) for atom in normalized_atoms)
        estimated_hydrogen = max(1, carbon * 2 + 2 - nitrogen - halogen)
        ring_count = len(set(re.findall(r"\d", compact)))
        metrics = CandidateMetrics(
            molecular_weight=round(heavy_weight + estimated_hydrogen * 1.008, 3),
            clogp=round(max(-1.5, 0.42 * carbon + 0.35 * halogen - 0.65 * (nitrogen + oxygen + sulfur)), 3),
            tpsa=round(25.0 * nitrogen + 17.0 * oxygen + 25.0 * sulfur, 3),
            hbd=min(nitrogen + oxygen, 6),
            hba=min(nitrogen + oxygen + sulfur, 12),
            synthetic_accessibility=round(min(10.0, 1.5 + 0.08 * len(atoms) + 0.45 * ring_count), 3),
        )
        digest = sha256(compact.encode("utf-8")).hexdigest().upper()
        surrogate_key = f"MOCK{digest[:10]}-{digest[10:20]}-{digest[20]}"
        return ChemistryReport(True, compact, surrogate_key, metrics, "syntax-fallback")
