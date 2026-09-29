"""Exact user-requested forbidden SMARTS screening, never string containment.

SMILES fragments which are valid SMARTS retain their established interpretation.
Parsing/resource errors are unresolved policy checks, not absence of a motif.
No structures, experimental measurements, or safety conclusions are generated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Sequence

POLICY = "rdkit-forbidden-substructure-v1"
# Parser/input resource bounds, not medicinal-chemistry acceptance thresholds.
MAX_PATTERNS = 128
MAX_PATTERN_CHARS = 4096
MAX_QUERY_ATOMS = 256
MAX_SMILES_CHARS = 100_000


@dataclass(frozen=True)
class SubstructureScreen:
    status: str
    patterns: tuple[str, ...] = ()
    matched_patterns: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    input_sha256: str | None = None
    rdkit_version: str | None = None
    source_smiles: str | None = None
    standardized_parent_smiles: str | None = None

    @property
    def matched(self) -> bool:
        return bool(self.matched_patterns)

    @property
    def blocked(self) -> bool:
        return self.status not in {"not_requested", "clear"}

    def as_metadata(self) -> dict:
        return {"policy": POLICY, "method": "RDKit SMARTS molecular-subgraph matching",
                **asdict(self), "matched": self.matched, "blocked": self.blocked,
                "screening_completed": self.status in {"clear", "matched"},
                "prediction_used": False, "string_matching_fallback_used": False}


def _load_rdkit():
    from rdkit import Chem, rdBase
    return Chem, rdBase


def substructure_input_hash(smiles: str, patterns: Sequence[str]) -> str:
    payload = {"smiles": smiles, "forbidden_substructures": list(patterns)}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _compile(patterns: Sequence[str]):
    if not isinstance(patterns, (list, tuple)):
        return SubstructureScreen("invalid_policy", reason_codes=("FORBIDDEN_SUBSTRUCTURE_POLICY_INVALID",)), None, []
    if len(patterns) > MAX_PATTERNS or any(not isinstance(p, str) or not p.strip()
                                         or len(p) > MAX_PATTERN_CHARS for p in patterns):
        return SubstructureScreen("invalid_policy", reason_codes=("FORBIDDEN_SUBSTRUCTURE_PATTERN_INPUT_INVALID",)), None, []
    values = tuple(patterns)
    if not values:
        return SubstructureScreen("not_requested"), None, []
    try:
        Chem, rdBase = _load_rdkit()
    except Exception:
        return SubstructureScreen("unavailable", values,
                                  reason_codes=("FORBIDDEN_SUBSTRUCTURE_BACKEND_UNAVAILABLE",)), None, []
    queries = []
    try:
        for pattern in values:
            # Do not silently accept a valid prefix followed by a molecule name.
            parameters = Chem.SmartsParserParams()
            parameters.parseName = False
            parameters.allowCXSMILES = False
            query = Chem.MolFromSmarts(pattern, parameters)
            if query is None or query.GetNumAtoms() == 0:
                return SubstructureScreen("invalid_policy", values,
                    reason_codes=("FORBIDDEN_SUBSTRUCTURE_PATTERN_INVALID",), rdkit_version=rdBase.rdkitVersion), None, []
            if query.GetNumAtoms() > MAX_QUERY_ATOMS:
                return SubstructureScreen("invalid_policy", values,
                    reason_codes=("FORBIDDEN_SUBSTRUCTURE_QUERY_RESOURCE_LIMIT",), rdkit_version=rdBase.rdkitVersion), None, []
            queries.append(query)
    except Exception:
        return SubstructureScreen("invalid_policy", values,
            reason_codes=("FORBIDDEN_SUBSTRUCTURE_PATTERN_INVALID",), rdkit_version=rdBase.rdkitVersion), None, []
    return SubstructureScreen("valid_policy", values, rdkit_version=rdBase.rdkitVersion), Chem, queries


def validate_forbidden_substructures(patterns: Sequence[str]) -> SubstructureScreen:
    """Validate a new execution policy without changing persisted input text."""
    return _compile(patterns)[0]


def screen_forbidden_substructures(smiles: str, patterns: Sequence[str]) -> SubstructureScreen:
    policy, Chem, queries = _compile(patterns)
    if policy.status == "not_requested":
        return policy
    fingerprint = substructure_input_hash(smiles, patterns) if isinstance(patterns, (list, tuple)) else None
    if policy.status != "valid_policy":
        return SubstructureScreen(policy.status, policy.patterns, reason_codes=policy.reason_codes,
                                  input_sha256=fingerprint, rdkit_version=policy.rdkit_version,
                                  source_smiles=smiles if isinstance(smiles, str) and len(smiles) <= MAX_SMILES_CHARS else None)
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > MAX_SMILES_CHARS:
        return SubstructureScreen("invalid_structure", policy.patterns,
            reason_codes=("FORBIDDEN_SUBSTRUCTURE_STRUCTURE_UNRESOLVED",), input_sha256=fingerprint,
            rdkit_version=policy.rdkit_version,
            source_smiles=smiles if isinstance(smiles, str) and len(smiles) <= MAX_SMILES_CHARS else None)
    try:
        parameters = Chem.SmilesParserParams()
        parameters.parseName = False
        parameters.allowCXSMILES = False
        molecule = Chem.MolFromSmiles(smiles, parameters)
        if molecule is None or molecule.GetNumAtoms() == 0:
            return SubstructureScreen("invalid_structure", policy.patterns,
                reason_codes=("FORBIDDEN_SUBSTRUCTURE_STRUCTURE_UNRESOLVED",), input_sha256=fingerprint,
                rdkit_version=policy.rdkit_version, source_smiles=smiles)
        match = Chem.SubstructMatchParameters()
        match.useChirality = True
        # Screen both the submitted graph and the same parent representation
        # used by ChemistryValidator before its descriptor calculation. No
        # descriptors are calculated here; salts/neutralisation cannot hide a
        # forbidden motif introduced or removed by parent standardisation.
        from rdkit.Chem.MolStandardize import rdMolStandardize
        cleaned = rdMolStandardize.Cleanup(molecule)
        parent = rdMolStandardize.FragmentParent(cleaned)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
        parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        matched = tuple(pattern for pattern, query in zip(policy.patterns, queries)
                        if any(graph.HasSubstructMatch(query, match) for graph in (molecule, parent)))
    except Exception:
        return SubstructureScreen("unavailable", policy.patterns,
            reason_codes=("FORBIDDEN_SUBSTRUCTURE_MATCH_UNAVAILABLE",), input_sha256=fingerprint,
            rdkit_version=policy.rdkit_version, source_smiles=smiles)
    return SubstructureScreen("matched" if matched else "clear", policy.patterns, matched,
        ("FORBIDDEN_SUBSTRUCTURE_MATCH",) if matched else (), fingerprint, policy.rdkit_version, smiles, parent_smiles)
