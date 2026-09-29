from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Protocol, Sequence


PROFILE_SCHEMA = "drugpilot.candidate-docking-profile/2.0"
PIPELINE_VERSION = "1.0.0"


class ReceptorPipelineError(RuntimeError):
    """Typed, fail-closed receptor qualification/preparation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class StructureSource(str, Enum):
    RCSB_EXPERIMENTAL = "rcsb_experimental"
    ALPHAFOLD_PREDICTED = "alphafold_predicted"


class ComponentRole(str, Enum):
    BINDING_LIGAND = "binding_ligand"
    COFACTOR = "cofactor"
    METAL = "metal"
    ION = "ion"
    WATER = "water"
    ADDITIVE = "additive"
    UNKNOWN = "unknown"


class FallbackReason(str, Enum):
    NO_EXACT_RCSB_HOLO = "no_exact_rcsb_holo"
    RCSB_QUALITY_BELOW_THRESHOLD = "rcsb_quality_below_threshold"
    RCSB_RETRIEVAL_UNAVAILABLE = "rcsb_retrieval_unavailable"


class PocketOrigin(str, Enum):
    COCRYSTAL_LIGAND = "co_crystal_ligand"
    P2RANK_PREDICTED = "p2rank_predicted"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


_MUTATION = re.compile(r"^(?:P\.)?([A-Z])(\d+)([A-Z])$", re.IGNORECASE)
_UNIPROT = re.compile(r"^[A-Z0-9]+(?:-\d+)?$")
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _normalise_accession(value: str) -> str:
    result = value.strip().upper()
    if not result or not _UNIPROT.fullmatch(result):
        raise ValueError(f"Invalid UniProt accession: {value!r}")
    return result


def _base_accession(value: str) -> str:
    return _normalise_accession(value).split("-", 1)[0]


def _normalise_mutation(value: str) -> str:
    result = value.strip().upper()
    match = _MUTATION.fullmatch(result)
    if match is None:
        raise ValueError(
            f"Mutation {value!r} must use one-letter protein notation such as R132H"
        )
    return f"{match.group(1)}{int(match.group(2))}{match.group(3)}"


@dataclass(frozen=True, slots=True)
class TargetStructureRequest:
    target_symbol: str
    uniprot_accession: str
    isoform_accession: str | None = None
    required_mutations: tuple[str, ...] = ()
    organism_tax_id: int = 9606

    def __post_init__(self) -> None:
        symbol = self.target_symbol.strip().upper()
        accession = _normalise_accession(self.uniprot_accession)
        isoform = (
            _normalise_accession(self.isoform_accession)
            if self.isoform_accession
            else None
        )
        if isoform and _base_accession(isoform) != _base_accession(accession):
            raise ValueError("The requested isoform must belong to the requested UniProt entry")
        mutations = tuple(sorted({_normalise_mutation(item) for item in self.required_mutations}))
        if not symbol:
            raise ValueError("target_symbol is required")
        if self.organism_tax_id <= 0:
            raise ValueError("organism_tax_id must be positive")
        object.__setattr__(self, "target_symbol", symbol)
        object.__setattr__(self, "uniprot_accession", _base_accession(accession))
        object.__setattr__(self, "isoform_accession", isoform)
        object.__setattr__(self, "required_mutations", mutations)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MissingResidue:
    chain_id: str
    residue_name: str
    residue_number: int
    insertion_code: str = ""


@dataclass(frozen=True, slots=True)
class BoundComponent:
    component_id: str
    role: ComponentRole
    chain_id: str
    residue_number: int
    atom_count: int
    suitable_for_box: bool = False
    occupancy: float | None = None
    mean_b_factor: float | None = None

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id is required")
        if self.atom_count < 1:
            raise ValueError("atom_count must be positive")
        if self.suitable_for_box and self.role is not ComponentRole.BINDING_LIGAND:
            raise ValueError("Only a binding ligand may define a co-crystal box")

    @property
    def instance_id(self) -> str:
        return f"{self.component_id.upper()}:{self.chain_id}:{self.residue_number}"


@dataclass(frozen=True, slots=True)
class StructureQuality:
    experimental_method: str
    resolution_angstrom: float | None
    r_free: float | None
    r_work: float | None
    sequence_coverage: float
    completeness_percent: float | None = None
    mean_b_factor: float | None = None
    release_date: str | None = None
    revision_date: str | None = None
    global_confidence: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.sequence_coverage <= 1.0:
            raise ValueError("sequence_coverage must be in [0, 1]")
        if self.resolution_angstrom is not None and self.resolution_angstrom <= 0:
            raise ValueError("resolution_angstrom must be positive")
        for label, value in (("r_free", self.r_free), ("r_work", self.r_work)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class StructureCandidate:
    structure_id: str
    source: StructureSource
    uniprot_accession: str
    isoform_accession: str | None
    organism_tax_id: int
    chains: tuple[str, ...]
    observed_mutations: tuple[str, ...]
    mutation_annotation_complete: bool
    quality: StructureQuality
    components: tuple[BoundComponent, ...] = ()
    waters: int = 0
    missing_residues: tuple[MissingResidue, ...] = ()
    missing_binding_site_residues: tuple[MissingResidue, ...] = ()
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    fixture: bool = False

    def __post_init__(self) -> None:
        structure_id = self.structure_id.strip().upper()
        accession = _normalise_accession(self.uniprot_accession)
        isoform = (
            _normalise_accession(self.isoform_accession)
            if self.isoform_accession
            else None
        )
        mutations = tuple(sorted({_normalise_mutation(item) for item in self.observed_mutations}))
        if not structure_id:
            raise ValueError("structure_id is required")
        if self.organism_tax_id <= 0:
            raise ValueError("organism_tax_id must be positive")
        if self.waters < 0:
            raise ValueError("waters cannot be negative")
        object.__setattr__(self, "structure_id", structure_id)
        object.__setattr__(self, "uniprot_accession", _base_accession(accession))
        object.__setattr__(self, "isoform_accession", isoform)
        object.__setattr__(self, "chains", tuple(sorted(set(self.chains))))
        object.__setattr__(self, "observed_mutations", mutations)
        object.__setattr__(self, "source_hashes", dict(self.source_hashes))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def box_ligands(self) -> tuple[BoundComponent, ...]:
        return tuple(
            component
            for component in self.components
            if component.role is ComponentRole.BINDING_LIGAND
            and component.suitable_for_box
        )

    @property
    def is_holo(self) -> bool:
        return bool(self.box_ligands)

    def inventory(self) -> dict[str, Any]:
        grouped = {role.value: [] for role in ComponentRole}
        for component in self.components:
            grouped[component.role.value].append(
                {**asdict(component), "role": component.role.value}
            )
        return {
            "components": grouped,
            "water_count": self.waters,
            "missing_residues": [asdict(item) for item in self.missing_residues],
            "missing_binding_site_residues": [
                asdict(item) for item in self.missing_binding_site_residues
            ],
        }

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["components"] = [
            {**asdict(item), "role": item.role.value} for item in self.components
        ]
        return payload


@dataclass(frozen=True, slots=True)
class QualificationThresholds:
    max_resolution_angstrom: float = 4.0
    max_r_free: float = 0.35
    min_sequence_coverage: float = 0.70
    max_missing_binding_site_residues: int = 0


@dataclass(frozen=True, slots=True)
class RankedStructure:
    candidate: StructureCandidate
    rank: int
    rank_vector: Mapping[str, Any]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "structure_id": self.candidate.structure_id,
            "source": self.candidate.source.value,
            "rank_vector": dict(self.rank_vector),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class RejectedStructure:
    structure_id: str
    source: StructureSource
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "source": self.source.value,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class StructureSelection:
    request: TargetStructureRequest
    selected: StructureCandidate
    ranked_rcsb: tuple[RankedStructure, ...]
    rejected: tuple[RejectedStructure, ...]
    fallback_reason: FallbackReason | None
    production_eligible_source: bool
    selected_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "selected_structure": self.selected.as_dict(),
            "ranked_rcsb": [item.as_dict() for item in self.ranked_rcsb],
            "rejected": [item.as_dict() for item in self.rejected],
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
            "production_eligible_source": self.production_eligible_source,
            "selected_at": self.selected_at,
            "pipeline_version": PIPELINE_VERSION,
        }


class ReceptorQualifier:
    """Select an exact target-matched RCSB holo structure or a typed AF fallback."""

    _METHOD_TIER = {
        "X-RAY DIFFRACTION": 0,
        "ELECTRON MICROSCOPY": 1,
        "NEUTRON DIFFRACTION": 2,
        "SOLUTION NMR": 3,
        "SOLID-STATE NMR": 3,
    }

    def __init__(self, thresholds: QualificationThresholds | None = None) -> None:
        self.thresholds = thresholds or QualificationThresholds()

    def _identity_reasons(
        self,
        request: TargetStructureRequest,
        candidate: StructureCandidate,
    ) -> list[str]:
        reasons: list[str] = []
        if candidate.uniprot_accession != request.uniprot_accession:
            reasons.append("UNIPROT_ACCESSION_MISMATCH")
        if request.isoform_accession is not None:
            if candidate.isoform_accession != request.isoform_accession:
                reasons.append("UNIPROT_ISOFORM_MISMATCH")
        if candidate.organism_tax_id != request.organism_tax_id:
            reasons.append("ORGANISM_MISMATCH")
        if not candidate.mutation_annotation_complete:
            reasons.append("MUTATION_ANNOTATION_INCOMPLETE")
        elif candidate.observed_mutations != request.required_mutations:
            reasons.append("MUTATION_SET_MISMATCH")
        return reasons

    def _quality_reasons(self, candidate: StructureCandidate) -> list[str]:
        thresholds = self.thresholds
        quality = candidate.quality
        reasons: list[str] = []
        if candidate.source is not StructureSource.RCSB_EXPERIMENTAL:
            reasons.append("NOT_RCSB_EXPERIMENTAL")
            return reasons
        if not candidate.is_holo:
            reasons.append("NO_QUALIFYING_COCRYSTAL_LIGAND")
        method = quality.experimental_method.strip().upper()
        if method not in self._METHOD_TIER:
            reasons.append("UNSUPPORTED_EXPERIMENTAL_METHOD")
        if method in {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "NEUTRON DIFFRACTION"}:
            if quality.resolution_angstrom is None:
                reasons.append("RESOLUTION_MISSING")
            elif quality.resolution_angstrom > thresholds.max_resolution_angstrom:
                reasons.append("RESOLUTION_BELOW_QUALITY_GATE")
        if method == "X-RAY DIFFRACTION":
            if quality.r_free is None:
                reasons.append("R_FREE_MISSING")
            elif quality.r_free > thresholds.max_r_free:
                reasons.append("R_FREE_BELOW_QUALITY_GATE")
        if quality.sequence_coverage < thresholds.min_sequence_coverage:
            reasons.append("SEQUENCE_COVERAGE_BELOW_QUALITY_GATE")
        if (
            len(candidate.missing_binding_site_residues)
            > thresholds.max_missing_binding_site_residues
        ):
            reasons.append("BINDING_SITE_RESIDUES_MISSING")
        return reasons

    def _sort_key(self, candidate: StructureCandidate) -> tuple[Any, ...]:
        quality = candidate.quality
        ligand = self._selected_ligand(candidate)
        method = quality.experimental_method.strip().upper()
        return (
            self._METHOD_TIER.get(method, 99),
            quality.resolution_angstrom
            if quality.resolution_angstrom is not None
            else float("inf"),
            quality.r_free if quality.r_free is not None else float("inf"),
            -quality.sequence_coverage,
            len(candidate.missing_binding_site_residues),
            len(candidate.missing_residues),
            -(ligand.occupancy if ligand.occupancy is not None else 0.0),
            ligand.mean_b_factor if ligand.mean_b_factor is not None else float("inf"),
            candidate.structure_id,
        )

    @staticmethod
    def _selected_ligand(candidate: StructureCandidate) -> BoundComponent:
        return sorted(
            candidate.box_ligands,
            key=lambda item: (
                -(item.occupancy if item.occupancy is not None else 0.0),
                item.mean_b_factor if item.mean_b_factor is not None else float("inf"),
                item.instance_id,
            ),
        )[0]

    def _rank_vector(self, candidate: StructureCandidate) -> dict[str, Any]:
        quality = candidate.quality
        ligand = self._selected_ligand(candidate)
        return {
            "method_tier": self._METHOD_TIER[
                quality.experimental_method.strip().upper()
            ],
            "resolution_angstrom": quality.resolution_angstrom,
            "r_free": quality.r_free,
            "sequence_coverage": quality.sequence_coverage,
            "missing_binding_site_residue_count": len(
                candidate.missing_binding_site_residues
            ),
            "missing_residue_count": len(candidate.missing_residues),
            "box_ligand_instance": ligand.instance_id,
            "box_ligand_occupancy": ligand.occupancy,
            "box_ligand_mean_b_factor": ligand.mean_b_factor,
        }

    def qualify_rcsb(
        self,
        request: TargetStructureRequest,
        candidates: Sequence[StructureCandidate],
    ) -> tuple[tuple[RankedStructure, ...], tuple[RejectedStructure, ...]]:
        eligible: list[StructureCandidate] = []
        rejected: list[RejectedStructure] = []
        for candidate in candidates:
            reasons = [
                *self._identity_reasons(request, candidate),
                *self._quality_reasons(candidate),
            ]
            unique_reasons = tuple(dict.fromkeys(reasons))
            if unique_reasons:
                rejected.append(
                    RejectedStructure(
                        structure_id=candidate.structure_id,
                        source=candidate.source,
                        reason_codes=unique_reasons,
                    )
                )
            else:
                eligible.append(candidate)
        ordered = sorted(eligible, key=self._sort_key)
        ranked = tuple(
            RankedStructure(
                candidate=candidate,
                rank=index,
                rank_vector=self._rank_vector(candidate),
                reasons=(
                    "exact UniProt/organism/mutation identity constraints",
                    "qualifying co-crystal ligand",
                    "explicit experimental quality metadata passed",
                ),
            )
            for index, candidate in enumerate(ordered, start=1)
        )
        return ranked, tuple(rejected)

    def select(
        self,
        request: TargetStructureRequest,
        rcsb_candidates: Sequence[StructureCandidate],
        *,
        alphafold_candidate: StructureCandidate | None = None,
        rcsb_retrieval_unavailable: bool = False,
    ) -> StructureSelection:
        ranked, rejected = self.qualify_rcsb(request, rcsb_candidates)
        if ranked:
            return StructureSelection(
                request=request,
                selected=ranked[0].candidate,
                ranked_rcsb=ranked,
                rejected=rejected,
                fallback_reason=None,
                production_eligible_source=not ranked[0].candidate.fixture,
                selected_at=_now_iso(),
            )

        if alphafold_candidate is None:
            raise ReceptorPipelineError(
                "NO_QUALIFIED_RECEPTOR_STRUCTURE",
                "No exact target-matched, quality-qualified RCSB holo structure was available",
                details={"rejected": [item.as_dict() for item in rejected]},
            )
        if alphafold_candidate.source is not StructureSource.ALPHAFOLD_PREDICTED:
            raise ReceptorPipelineError(
                "INVALID_FALLBACK_SOURCE",
                "The fallback candidate must be an AlphaFold predicted structure",
            )
        identity_reasons = self._identity_reasons(request, alphafold_candidate)
        if identity_reasons:
            raise ReceptorPipelineError(
                "ALPHAFOLD_TARGET_MISMATCH",
                "AlphaFold fallback did not exactly match the requested target identity",
                details={"reason_codes": identity_reasons},
            )
        if rcsb_retrieval_unavailable:
            fallback = FallbackReason.RCSB_RETRIEVAL_UNAVAILABLE
        elif any(
            reason in {
                "RESOLUTION_MISSING",
                "RESOLUTION_BELOW_QUALITY_GATE",
                "R_FREE_MISSING",
                "R_FREE_BELOW_QUALITY_GATE",
                "SEQUENCE_COVERAGE_BELOW_QUALITY_GATE",
                "BINDING_SITE_RESIDUES_MISSING",
            }
            for item in rejected
            for reason in item.reason_codes
        ):
            fallback = FallbackReason.RCSB_QUALITY_BELOW_THRESHOLD
        else:
            fallback = FallbackReason.NO_EXACT_RCSB_HOLO
        return StructureSelection(
            request=request,
            selected=alphafold_candidate,
            ranked_rcsb=(),
            rejected=rejected,
            fallback_reason=fallback,
            production_eligible_source=False,
            selected_at=_now_iso(),
        )


@dataclass(frozen=True, slots=True)
class ToolCapability:
    tool_id: str
    available: bool
    version: str | None
    executable_path: str | None = None
    executable_sha256: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReceptorToolchain:
    capabilities: tuple[ToolCapability, ...]
    probed_at: str

    REQUIRED_TOOLS = ("pdbfixer", "meeko", "p2rank", "posebusters")

    def by_id(self) -> dict[str, ToolCapability]:
        return {item.tool_id: item for item in self.capabilities}

    def require_all(self) -> None:
        indexed = self.by_id()
        missing = [
            tool_id
            for tool_id in self.REQUIRED_TOOLS
            if tool_id not in indexed or not indexed[tool_id].available
        ]
        if missing:
            raise ReceptorPipelineError(
                "RECEPTOR_PREPARATION_TOOLCHAIN_UNAVAILABLE",
                "Receptor preparation is fail-closed because required scientific tools are unavailable",
                details={
                    "missing_tools": missing,
                    "capabilities": [item.as_dict() for item in self.capabilities],
                },
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "probed_at": self.probed_at,
            "capabilities": [item.as_dict() for item in self.capabilities],
        }


class ReceptorToolchainProbe:
    """Probe only; it never installs or silently substitutes a dependency."""

    def __init__(
        self,
        *,
        p2rank_executable: str | Path | None = None,
        meeko_executable: str | Path | None = None,
    ) -> None:
        self.p2rank_executable = Path(p2rank_executable) if p2rank_executable else None
        self.meeko_executable = Path(meeko_executable) if meeko_executable else None

    @staticmethod
    def _python_capability(tool_id: str, module_name: str) -> ToolCapability:
        available = find_spec(module_name) is not None
        return ToolCapability(
            tool_id=tool_id,
            available=available,
            version=_package_version(tool_id) if available else None,
            reason=None if available else f"Python module {module_name!r} is not installed",
        )

    @staticmethod
    def _executable_capability(
        tool_id: str,
        explicit: Path | None,
        command_names: tuple[str, ...],
        *,
        package_name: str | None = None,
    ) -> ToolCapability:
        candidate: Path | None = None
        if explicit is not None:
            resolved = explicit.expanduser().resolve()
            if resolved.is_file():
                candidate = resolved
        else:
            for command in command_names:
                found = shutil.which(command)
                if found:
                    candidate = Path(found).resolve()
                    break
        return ToolCapability(
            tool_id=tool_id,
            available=candidate is not None,
            version=_package_version(package_name) if package_name else None,
            executable_path=str(candidate) if candidate else None,
            executable_sha256=_file_digest(candidate) if candidate else None,
            reason=None if candidate else f"Executable not found: {', '.join(command_names)}",
        )

    def probe(self) -> ReceptorToolchain:
        pdbfixer = self._python_capability("pdbfixer", "pdbfixer")
        posebusters = self._python_capability("posebusters", "posebusters")
        meeko_python = self._python_capability("meeko", "meeko")
        meeko_cli = self._executable_capability(
            "meeko",
            self.meeko_executable,
            ("mk_prepare_receptor", "mk_prepare_receptor.exe"),
            package_name="meeko",
        )
        meeko = ToolCapability(
            tool_id="meeko",
            available=meeko_python.available and meeko_cli.available,
            version=meeko_python.version or meeko_cli.version,
            executable_path=meeko_cli.executable_path,
            executable_sha256=meeko_cli.executable_sha256,
            reason=(
                None
                if meeko_python.available and meeko_cli.available
                else "; ".join(
                    item
                    for item in (meeko_python.reason, meeko_cli.reason)
                    if item
                )
            ),
        )
        p2rank = self._executable_capability(
            "p2rank",
            self.p2rank_executable,
            ("prank", "prank.bat", "p2rank"),
        )
        return ReceptorToolchain(
            capabilities=(pdbfixer, meeko, p2rank, posebusters),
            probed_at=_now_iso(),
        )


@dataclass(frozen=True, slots=True)
class DockingPocket:
    origin: PocketOrigin
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    ligand_instance_id: str | None = None
    prediction_label: str | None = None
    prediction_score: float | None = None
    source_artifact_role: str | None = None

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("Docking pocket center and size must be 3-vectors")
        if any(float(value) <= 0 for value in self.size):
            raise ValueError("Docking pocket dimensions must be positive")
        if self.origin is PocketOrigin.COCRYSTAL_LIGAND and not self.ligand_instance_id:
            raise ValueError("A co-crystal pocket requires ligand_instance_id")
        if self.origin is PocketOrigin.P2RANK_PREDICTED:
            if self.prediction_label != "predicted":
                raise ValueError("A P2Rank pocket must be explicitly labelled predicted")
            if not self.source_artifact_role:
                raise ValueError("A P2Rank pocket requires a prediction artifact")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["origin"] = self.origin.value
        return payload


@dataclass(frozen=True, slots=True)
class FileArtifact:
    role: str
    path: Path
    produced_by: str
    execution: Mapping[str, Any] = field(default_factory=dict)

    def audited(self) -> dict[str, Any]:
        resolved = self.path.expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise ReceptorPipelineError(
                "RECEPTOR_ARTIFACT_MISSING",
                f"Required receptor artifact is missing or empty: {self.role}",
                details={"role": self.role, "filename": resolved.name},
            )
        return {
            "role": self.role,
            "filename": resolved.name,
            "byte_count": resolved.stat().st_size,
            "sha256": _file_digest(resolved),
            "produced_by": self.produced_by,
            "execution": dict(self.execution),
        }


@dataclass(frozen=True, slots=True)
class RedockingValidation:
    status: ValidationStatus
    threshold_angstrom: float | None
    top_pose_rmsd_angstrom: float | None
    validation_artifact_role: str | None
    method: str

    @property
    def passed(self) -> bool:
        return (
            self.status is ValidationStatus.PASSED
            and self.threshold_angstrom is not None
            and self.top_pose_rmsd_angstrom is not None
            and self.top_pose_rmsd_angstrom <= self.threshold_angstrom
            and self.validation_artifact_role is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status.value, "passed": self.passed}


@dataclass(frozen=True, slots=True)
class PoseValidation:
    status: ValidationStatus
    validator: str
    checked_pose_count: int
    failure_codes: tuple[str, ...]
    validation_artifact_role: str | None

    @property
    def passed(self) -> bool:
        return (
            self.status is ValidationStatus.PASSED
            and self.validator.casefold() == "posebusters"
            and self.checked_pose_count > 0
            and not self.failure_codes
            and self.validation_artifact_role is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status.value, "passed": self.passed}


@dataclass(frozen=True, slots=True)
class ProductionAcceptance:
    accepted: bool
    reviewer: str | None = None
    accepted_at: str | None = None
    protocol_version: str | None = None
    benchmark_scope: str | None = None

    def is_complete(self) -> bool:
        if not self.accepted:
            return False
        return all(
            value and str(value).strip()
            for value in (
                self.reviewer,
                self.accepted_at,
                self.protocol_version,
                self.benchmark_scope,
            )
        )


@dataclass(frozen=True, slots=True)
class PreparedReceptorRun:
    pocket: DockingPocket
    artifacts: tuple[FileArtifact, ...]
    redocking: RedockingValidation
    pose_validation: PoseValidation


class ReceptorPreparationBackend(Protocol):
    """Integration seam for a future local PDBFixer/Meeko/P2Rank runner."""

    def prepare(
        self,
        selection: StructureSelection,
        toolchain: ReceptorToolchain,
    ) -> PreparedReceptorRun: ...


class CandidateDockingProfileBuilder:
    REQUIRED_ARTIFACT_ROLES = {
        "source_structure",
        "pdbfixer_receptor",
        "meeko_receptor_pdbqt",
        "posebusters_report",
    }

    def build(
        self,
        *,
        profile_id: str,
        profile_version: str,
        selection: StructureSelection,
        toolchain: ReceptorToolchain,
        prepared: PreparedReceptorRun,
        acceptance: ProductionAcceptance | None = None,
    ) -> dict[str, Any]:
        toolchain.require_all()
        if not profile_id.strip() or not profile_version.strip():
            raise ValueError("profile_id and profile_version are required")
        artifacts = [item.audited() for item in prepared.artifacts]
        artifact_roles = [str(item["role"]) for item in artifacts]
        if len(artifact_roles) != len(set(artifact_roles)):
            raise ReceptorPipelineError(
                "RECEPTOR_ARTIFACT_ROLE_DUPLICATED",
                "Each receptor artifact role must be unique",
            )
        by_role = {str(item["role"]): item for item in artifacts}
        missing_roles = sorted(self.REQUIRED_ARTIFACT_ROLES - set(by_role))
        if missing_roles:
            raise ReceptorPipelineError(
                "RECEPTOR_ARTIFACT_SET_INCOMPLETE",
                "Prepared receptor output omitted required auditable artifacts",
                details={"missing_artifact_roles": missing_roles},
            )
        validation_roles = {
            prepared.redocking.validation_artifact_role,
            prepared.pose_validation.validation_artifact_role,
        } - {None}
        missing_validation_roles = sorted(validation_roles - set(by_role))
        if missing_validation_roles:
            raise ReceptorPipelineError(
                "RECEPTOR_VALIDATION_ARTIFACT_MISSING",
                "A declared receptor validation artifact is absent from the audited artifact set",
                details={"missing_artifact_roles": missing_validation_roles},
            )

        selected = selection.selected
        if not selected.fixture:
            declared_hashes = {
                value.casefold()
                for key, value in selected.source_hashes.items()
                if key.casefold().endswith("sha256")
                and re.fullmatch(r"[0-9a-fA-F]{64}", str(value))
            }
            source_hash = str(by_role["source_structure"]["sha256"]).casefold()
            if not declared_hashes or source_hash not in declared_hashes:
                raise ReceptorPipelineError(
                    "RECEPTOR_SOURCE_HASH_MISMATCH",
                    "The prepared source structure does not match qualified source provenance",
                    details={
                        "qualified_source_hashes": sorted(declared_hashes),
                        "artifact_source_sha256": source_hash,
                    },
                )
        pocket = prepared.pocket
        readiness_reasons: list[str] = []
        if selected.source is StructureSource.RCSB_EXPERIMENTAL:
            if pocket.origin is not PocketOrigin.COCRYSTAL_LIGAND:
                raise ReceptorPipelineError(
                    "EXPERIMENTAL_POCKET_ORIGIN_INVALID",
                    "An RCSB holo structure must derive its box from the selected co-crystal ligand",
                )
            ligand_ids = {item.instance_id for item in selected.box_ligands}
            if pocket.ligand_instance_id not in ligand_ids:
                raise ReceptorPipelineError(
                    "COCRYSTAL_LIGAND_MISMATCH",
                    "Docking box ligand did not match a qualified bound ligand instance",
                )
            if not prepared.redocking.passed:
                readiness_reasons.append("REDOCKING_NOT_PASSED")
        else:
            if pocket.origin is not PocketOrigin.P2RANK_PREDICTED:
                raise ReceptorPipelineError(
                    "ALPHAFOLD_POCKET_ORIGIN_INVALID",
                    "An AlphaFold fallback requires an explicitly labelled P2Rank predicted pocket",
                )
            if pocket.source_artifact_role not in by_role:
                raise ReceptorPipelineError(
                    "PREDICTED_POCKET_ARTIFACT_MISSING",
                    "The predicted pocket source artifact is not present",
                )
            readiness_reasons.extend(
                ("ALPHAFOLD_FALLBACK_NOT_PRODUCTION_ELIGIBLE", "REDOCKING_NOT_AVAILABLE")
            )
        if not prepared.pose_validation.passed:
            readiness_reasons.append("POSEBUSTERS_VALIDATION_NOT_PASSED")
        acceptance = acceptance or ProductionAcceptance(accepted=False)
        if not acceptance.is_complete():
            readiness_reasons.append("PRODUCTION_ACCEPTANCE_MISSING")
        if selected.fixture:
            readiness_reasons.append("FIXTURE_STRUCTURE_NOT_PRODUCTION_ELIGIBLE")
        production_ready = not readiness_reasons

        payload: dict[str, Any] = {
            "schema": PROFILE_SCHEMA,
            "profile_id": profile_id.strip(),
            "profile_version": profile_version.strip(),
            "pipeline_version": PIPELINE_VERSION,
            "created_at": _now_iso(),
            "target": selection.request.as_dict(),
            "selection": selection.as_dict(),
            "receptor_inventory": selected.inventory(),
            "pocket": pocket.as_dict(),
            "toolchain": toolchain.as_dict(),
            "artifacts": artifacts,
            "validations": {
                "redocking": prepared.redocking.as_dict(),
                "pose": prepared.pose_validation.as_dict(),
            },
            "production_acceptance": asdict(acceptance),
            "profile_status": (
                "production_ready" if production_ready else "candidate_not_accepted"
            ),
            "production_ready": production_ready,
            "production_readiness_reasons": sorted(set(readiness_reasons)),
            "fixture": selected.fixture,
        }
        payload["profile_sha256"] = _json_digest(payload)
        return payload

    def write(self, payload: Mapping[str, Any], path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        return destination


class ReceptorPreparationPipeline:
    """Require the complete toolchain before invoking any preparation backend."""

    def __init__(self, builder: CandidateDockingProfileBuilder | None = None) -> None:
        self.builder = builder or CandidateDockingProfileBuilder()

    def prepare_profile(
        self,
        *,
        profile_id: str,
        profile_version: str,
        selection: StructureSelection,
        toolchain: ReceptorToolchain,
        backend: ReceptorPreparationBackend,
        acceptance: ProductionAcceptance | None = None,
    ) -> dict[str, Any]:
        toolchain.require_all()
        prepared = backend.prepare(selection, toolchain)
        return self.builder.build(
            profile_id=profile_id,
            profile_version=profile_version,
            selection=selection,
            toolchain=toolchain,
            prepared=prepared,
            acceptance=acceptance,
        )


def parse_legacy_rcsb_pdb(
    path: str | Path,
    *,
    structure_id: str,
    component_roles: Mapping[str, ComponentRole | str],
    box_ligand_components: Sequence[str],
    isoform_accession: str | None = None,
) -> StructureCandidate:
    """Parse auditable identity/quality/inventory fields from a legacy PDB file.

    Component roles are deliberately supplied by the caller.  The parser never
    guesses that a heterogen is a drug-like ligand, cofactor, metal or buffer.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source.name)
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    text = "\n".join(lines)

    dbrefs: list[tuple[str, int, int, str]] = []
    for line in lines:
        match = re.match(
            r"^DBREF\s+\S+\s+(\S)\s+(-?\d+)\s+(-?\d+)\s+UNP\s+(\S+)",
            line,
        )
        if match:
            dbrefs.append(
                (match.group(1), int(match.group(2)), int(match.group(3)), match.group(4))
            )
    if not dbrefs:
        raise ReceptorPipelineError(
            "PDB_UNIPROT_MAPPING_MISSING",
            "The PDB file contains no UniProt DBREF mapping",
        )
    accessions = {_base_accession(item[3]) for item in dbrefs}
    if len(accessions) != 1:
        raise ReceptorPipelineError(
            "PDB_TARGET_MAPPING_AMBIGUOUS",
            "The PDB file maps protein chains to more than one UniProt accession",
            details={"accessions": sorted(accessions)},
        )
    accession = next(iter(accessions))
    chains = tuple(sorted(item[0] for item in dbrefs if _base_accession(item[3]) == accession))

    mutations: set[str] = set()
    for line in lines:
        match = re.match(
            r"^SEQADV\s+\S+\s+([A-Z]{3})\s+\S\s+(-?\d+)\s+UNP\s+\S+\s+([A-Z]{3})\s+\d+\s+ENGINEERED MUTATION",
            line,
        )
        if not match:
            continue
        observed = _THREE_TO_ONE.get(match.group(1))
        reference = _THREE_TO_ONE.get(match.group(3))
        if not observed or not reference:
            raise ReceptorPipelineError(
                "PDB_MUTATION_NOT_NORMALISABLE",
                "An engineered mutation could not be converted to one-letter notation",
            )
        mutations.add(f"{reference}{int(match.group(2))}{observed}")

    missing: list[MissingResidue] = []
    for line in lines:
        match = re.match(
            r"^REMARK 465\s+(?:\d+\s+)?([A-Z]{3})\s+(\S)\s+(-?\d+)(\S?)\s*$",
            line,
        )
        if match and match.group(2) in chains:
            missing.append(
                MissingResidue(
                    chain_id=match.group(2),
                    residue_name=match.group(1),
                    residue_number=int(match.group(3)),
                    insertion_code=match.group(4),
                )
            )

    observed_residues: dict[str, set[int]] = {chain: set() for chain in chains}
    component_atom_counts: dict[tuple[str, str, int], int] = {}
    component_occupancies: dict[tuple[str, str, int], list[float]] = {}
    component_b_factors: dict[tuple[str, str, int], list[float]] = {}
    waters: set[tuple[str, int]] = set()
    for line in lines:
        if len(line) < 54:
            continue
        record = line[:6].strip()
        chain = line[21].strip()
        residue_name = line[17:20].strip().upper()
        try:
            residue_number = int(line[22:26].strip())
        except ValueError:
            continue
        if record == "ATOM" and chain in observed_residues:
            observed_residues[chain].add(residue_number)
        elif record == "HETATM":
            if residue_name == "HOH":
                waters.add((chain, residue_number))
                continue
            key = (residue_name, chain, residue_number)
            component_atom_counts[key] = component_atom_counts.get(key, 0) + 1
            if len(line) >= 66:
                try:
                    component_occupancies.setdefault(key, []).append(float(line[54:60]))
                    component_b_factors.setdefault(key, []).append(float(line[60:66]))
                except ValueError:
                    pass

    coverage_values: list[float] = []
    for chain, start, end, mapped_accession in dbrefs:
        if _base_accession(mapped_accession) != accession:
            continue
        total = end - start + 1
        observed = sum(start <= residue <= end for residue in observed_residues[chain])
        coverage_values.append(observed / total)
    coverage = min(coverage_values) if coverage_values else 0.0

    roles = {
        component.upper(): (
            role if isinstance(role, ComponentRole) else ComponentRole(str(role))
        )
        for component, role in component_roles.items()
    }
    box_ids = {item.upper() for item in box_ligand_components}
    components: list[BoundComponent] = []
    for (component_id, chain, residue_number), atom_count in sorted(
        component_atom_counts.items()
    ):
        role = roles.get(component_id, ComponentRole.UNKNOWN)
        occupancies = component_occupancies.get((component_id, chain, residue_number), [])
        b_factors = component_b_factors.get((component_id, chain, residue_number), [])
        components.append(
            BoundComponent(
                component_id=component_id,
                role=role,
                chain_id=chain,
                residue_number=residue_number,
                atom_count=atom_count,
                suitable_for_box=component_id in box_ids,
                occupancy=(sum(occupancies) / len(occupancies) if occupancies else None),
                mean_b_factor=(sum(b_factors) / len(b_factors) if b_factors else None),
            )
        )

    def _float(pattern: str) -> float | None:
        match = re.search(pattern, text, re.MULTILINE)
        return float(match.group(1)) if match else None

    method_match = re.search(r"^EXPDTA\s+(.+?)\s*$", text, re.MULTILINE)
    tax_match = re.search(r"ORGANISM_TAXID:\s*(\d+)", text)
    if not method_match or not tax_match:
        raise ReceptorPipelineError(
            "PDB_QUALITY_METADATA_INCOMPLETE",
            "The PDB header omitted method or organism metadata",
        )
    release_candidates: list[tuple[int, str]] = []
    for line in lines:
        match = re.match(r"^REVDAT\s+(\d+)\s+(\d{2}-[A-Z]{3}-\d{2})", line)
        if match:
            release_candidates.append((int(match.group(1)), match.group(2)))
    initial_release = None
    latest_revision = None
    if release_candidates:
        initial_release = min(release_candidates, key=lambda item: item[0])[1]
        latest_revision = max(release_candidates, key=lambda item: item[0])[1]

    return StructureCandidate(
        structure_id=structure_id,
        source=StructureSource.RCSB_EXPERIMENTAL,
        uniprot_accession=accession,
        isoform_accession=isoform_accession,
        organism_tax_id=int(tax_match.group(1)),
        chains=chains,
        observed_mutations=tuple(sorted(mutations)),
        mutation_annotation_complete=True,
        quality=StructureQuality(
            experimental_method=method_match.group(1).strip(),
            resolution_angstrom=_float(r"^REMARK\s+2\s+RESOLUTION\.\s+([0-9.]+)"),
            r_free=_float(r"^REMARK\s+3\s+FREE R VALUE\s+:\s*([0-9.]+)"),
            r_work=_float(r"^REMARK\s+3\s+R VALUE\s+\(WORKING SET\)\s+:\s*([0-9.]+)"),
            sequence_coverage=coverage,
            completeness_percent=_float(
                r"^REMARK\s+3\s+COMPLETENESS FOR RANGE\s+\(%\)\s+:\s*([0-9.]+)"
            ),
            mean_b_factor=_float(
                r"^REMARK\s+3\s+MEAN B VALUE\s+\(OVERALL, A\*\*2\)\s+:\s*([0-9.]+)"
            ),
            release_date=initial_release,
            revision_date=latest_revision,
        ),
        components=tuple(components),
        waters=len(waters),
        missing_residues=tuple(missing),
        missing_binding_site_residues=(),
        source_hashes={"source_pdb_sha256": _file_digest(source)},
        provenance={
            "parser": "DrugPilot legacy PDB fixed-column/header parser",
            "parser_version": PIPELINE_VERSION,
            "source_filename": source.name,
            "component_roles_caller_supplied": True,
        },
        fixture=False,
    )


def structure_candidate_from_dict(raw: Mapping[str, Any]) -> StructureCandidate:
    quality_raw = raw.get("quality")
    if not isinstance(quality_raw, Mapping):
        raise ValueError("Structure candidate quality must be an object")
    components_raw = raw.get("components", [])
    missing_raw = raw.get("missing_residues", [])
    missing_site_raw = raw.get("missing_binding_site_residues", [])
    return StructureCandidate(
        structure_id=str(raw["structure_id"]),
        source=StructureSource(str(raw["source"])),
        uniprot_accession=str(raw["uniprot_accession"]),
        isoform_accession=(
            str(raw["isoform_accession"]) if raw.get("isoform_accession") else None
        ),
        organism_tax_id=int(raw.get("organism_tax_id", 9606)),
        chains=tuple(str(item) for item in raw.get("chains", [])),
        observed_mutations=tuple(str(item) for item in raw.get("observed_mutations", [])),
        mutation_annotation_complete=raw.get("mutation_annotation_complete") is True,
        quality=StructureQuality(**dict(quality_raw)),
        components=tuple(
            BoundComponent(
                component_id=str(item["component_id"]),
                role=ComponentRole(str(item["role"])),
                chain_id=str(item["chain_id"]),
                residue_number=int(item["residue_number"]),
                atom_count=int(item["atom_count"]),
                suitable_for_box=item.get("suitable_for_box") is True,
                occupancy=(float(item["occupancy"]) if item.get("occupancy") is not None else None),
                mean_b_factor=(
                    float(item["mean_b_factor"])
                    if item.get("mean_b_factor") is not None
                    else None
                ),
            )
            for item in components_raw
        ),
        waters=int(raw.get("waters", 0)),
        missing_residues=tuple(MissingResidue(**dict(item)) for item in missing_raw),
        missing_binding_site_residues=tuple(
            MissingResidue(**dict(item)) for item in missing_site_raw
        ),
        source_hashes=dict(raw.get("source_hashes", {})),
        provenance=dict(raw.get("provenance", {})),
        fixture=raw.get("fixture") is True,
    )


__all__ = [
    "BoundComponent",
    "CandidateDockingProfileBuilder",
    "ComponentRole",
    "DockingPocket",
    "FallbackReason",
    "FileArtifact",
    "MissingResidue",
    "PIPELINE_VERSION",
    "PROFILE_SCHEMA",
    "PocketOrigin",
    "PoseValidation",
    "PreparedReceptorRun",
    "ProductionAcceptance",
    "QualificationThresholds",
    "RankedStructure",
    "ReceptorPipelineError",
    "ReceptorPreparationBackend",
    "ReceptorPreparationPipeline",
    "ReceptorQualifier",
    "ReceptorToolchain",
    "ReceptorToolchainProbe",
    "RedockingValidation",
    "RejectedStructure",
    "StructureCandidate",
    "StructureQuality",
    "StructureSelection",
    "StructureSource",
    "TargetStructureRequest",
    "ToolCapability",
    "ValidationStatus",
    "parse_legacy_rcsb_pdb",
    "structure_candidate_from_dict",
]
