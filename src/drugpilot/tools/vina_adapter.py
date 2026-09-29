from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from drugpilot.domain.models import TargetCandidate
from drugpilot.tools.executable import (
    ExecutableRunner,
    ExecutionProvenance,
    ScientificToolExecutionError,
    ScientificToolUnavailable,
)


_VINA_TABLE_SCORE = re.compile(
    r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+(?:\d+(?:\.\d+)?|0)\s+(?:\d+(?:\.\d+)?|0)\s*$",
    re.MULTILINE,
)
_VINA_REMARK_SCORE = re.compile(
    r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _triplet(value: Any, *, field_name: str, positive: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers")
    result = tuple(float(item) for item in value)
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{field_name} values must be positive")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DockingBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ValidatedDockingProfile:
    profile_id: str
    target_symbol: str
    uniprot_id: str | None
    receptor_pdbqt: Path
    box: DockingBox
    validated: bool
    validation_method: str
    receptor_source: Mapping[str, Any]
    profile_version: str
    explicit_selection_required: bool = True
    profile_status: str = "experimental_smoke_only"
    production_ready: bool = False
    validation_scope: str | None = None
    redocking_validation: Mapping[str, Any] | None = None

    def provenance(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "target_symbol": self.target_symbol,
            "uniprot_id": self.uniprot_id,
            "validated": self.validated,
            "validation_method": self.validation_method,
            "receptor_source": dict(self.receptor_source),
            "explicit_selection_required": self.explicit_selection_required,
            "profile_status": self.profile_status,
            "production_ready": self.production_ready,
            "validation_scope": self.validation_scope,
            "redocking_validation": (
                dict(self.redocking_validation) if self.redocking_validation else None
            ),
            "receptor_filename": self.receptor_pdbqt.name,
            "receptor_sha256": _file_digest(self.receptor_pdbqt),
            "box": asdict(self.box),
        }


class DockingProfileRegistry:
    """Load target-specific receptor and experimentally validated box profiles."""

    def __init__(self, profiles: list[ValidatedDockingProfile]) -> None:
        self._profiles = tuple(profiles)
        self._by_symbol = {profile.target_symbol.casefold(): profile for profile in profiles}
        self._by_uniprot = {
            profile.uniprot_id.casefold(): profile
            for profile in profiles
            if profile.uniprot_id
        }

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    @classmethod
    def from_json(cls, path: str | Path) -> "DockingProfileRegistry":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ScientificToolUnavailable(
                f"Docking profile registry is missing: {config_path.name}"
            )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("Unsupported docking profile schema_version")
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError("Docking registry must contain at least one profile")
        profiles = [cls._parse_profile(item, config_path.parent) for item in raw_profiles]
        return cls(profiles)

    @staticmethod
    def _parse_profile(raw: Any, base_dir: Path) -> ValidatedDockingProfile:
        if not isinstance(raw, dict):
            raise ValueError("Each docking profile must be an object")
        required_text = (
            "profile_id",
            "profile_version",
            "target_symbol",
            "receptor_pdbqt",
            "validation_method",
        )
        values: dict[str, str] = {}
        for key in required_text:
            value = str(raw.get(key) or "").strip()
            if not value:
                raise ValueError(f"Docking profile is missing {key}")
            values[key] = value
        if raw.get("validated") is not True:
            raise ValueError(
                f"Docking profile {values['profile_id']} is not explicitly validated"
            )
        receptor = Path(values["receptor_pdbqt"])
        if not receptor.is_absolute():
            receptor = (base_dir / receptor).resolve()
        if not receptor.is_file():
            raise ScientificToolUnavailable(
                f"Prepared receptor is missing for profile {values['profile_id']}"
            )
        if receptor.suffix.casefold() != ".pdbqt":
            raise ValueError("Prepared receptor must use the .pdbqt format")
        header = receptor.read_text(encoding="utf-8", errors="replace")[:40_000]
        if "ATOM" not in header and "HETATM" not in header:
            raise ValueError("Prepared receptor PDBQT contains no atom records")
        box_raw = raw.get("box")
        if not isinstance(box_raw, dict):
            raise ValueError("Docking profile requires a box object")
        box = DockingBox(
            center=_triplet(box_raw.get("center"), field_name="box.center"),
            size=_triplet(box_raw.get("size"), field_name="box.size", positive=True),
        )
        if any(item > 80 for item in box.size):
            raise ValueError("Docking box dimensions above 80 A require manual review")
        receptor_source = raw.get("receptor_source")
        if not isinstance(receptor_source, dict) or not receptor_source:
            raise ValueError("Docking profile requires receptor_source provenance")
        return ValidatedDockingProfile(
            profile_id=values["profile_id"],
            target_symbol=values["target_symbol"].upper(),
            uniprot_id=(str(raw["uniprot_id"]).upper() if raw.get("uniprot_id") else None),
            receptor_pdbqt=receptor,
            box=box,
            validated=True,
            validation_method=values["validation_method"],
            receptor_source=dict(receptor_source),
            profile_version=values["profile_version"],
            explicit_selection_required=raw.get("explicit_selection_required") is not False,
            profile_status=str(raw.get("profile_status") or "experimental_smoke_only"),
            production_ready=raw.get("production_ready") is True,
            validation_scope=(
                str(raw["validation_scope"]) if raw.get("validation_scope") else None
            ),
            redocking_validation=(
                dict(raw["redocking_validation"])
                if isinstance(raw.get("redocking_validation"), dict)
                else None
            ),
        )

    def require(self, target: TargetCandidate) -> ValidatedDockingProfile:
        profile = None
        if target.uniprot_id:
            profile = self._by_uniprot.get(target.uniprot_id.casefold())
        if profile is None:
            profile = self._by_symbol.get(target.symbol.casefold())
        if profile is None:
            raise ScientificToolUnavailable(
                f"No validated receptor/box profile is configured for {target.symbol}"
            )
        requested_profile = str(target.metadata.get("docking_profile_id") or "")
        if profile.explicit_selection_required and requested_profile != profile.profile_id:
            raise ScientificToolUnavailable(
                f"Docking profile {profile.profile_id} is explicit-only; "
                "the hypothesis target did not request that exact profile"
            )
        if target.uniprot_id and profile.uniprot_id:
            if target.uniprot_id.casefold() != profile.uniprot_id.casefold():
                raise ValueError(
                    f"Docking profile target mismatch for {target.symbol}: UniProt differs"
                )
        return profile


@dataclass(frozen=True, slots=True)
class VinaDockingResult:
    score_kcal_mol: float
    profile_provenance: Mapping[str, Any]
    ligand_preparation: Mapping[str, Any]
    ligand_preparation_execution: ExecutionProvenance
    vina_execution: ExecutionProvenance
    prepared_ligand_sha256: str
    docked_pose_sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            "fixture": False,
            "method": "AutoDock Vina",
            "score_unit": "kcal/mol",
            "profile": dict(self.profile_provenance),
            "ligand_preparation": dict(self.ligand_preparation),
            "ligand_preparation_execution": self.ligand_preparation_execution.as_dict(),
            "vina_execution": self.vina_execution.as_dict(),
            "prepared_ligand_sha256": self.prepared_ligand_sha256,
            "docked_pose_sha256": self.docked_pose_sha256,
        }


class VinaDockingAdapter:
    """Execute Meeko ligand preparation and AutoDock Vina fail-closed."""

    backend_kind = "live"
    fixture = False

    def __init__(
        self,
        *,
        profiles: DockingProfileRegistry,
        vina_executable: str | Path | None = None,
        python_executable: str | Path | None = None,
        ligand_prep_timeout_seconds: float = 60.0,
        docking_timeout_seconds: float = 180.0,
        exhaustiveness: int = 8,
        num_modes: int = 9,
        cpu: int = 1,
        seed: int = 20260819,
        allow_experimental_profiles: bool = False,
        runtime_variant: str = "production",
    ) -> None:
        if ligand_prep_timeout_seconds <= 0 or docking_timeout_seconds <= 0:
            raise ValueError("Scientific tool timeouts must be positive")
        if exhaustiveness < 1 or num_modes < 1 or cpu < 1:
            raise ValueError("Vina exhaustiveness, num_modes and cpu must be positive")
        if allow_experimental_profiles and not any(
            marker in runtime_variant.casefold()
            for marker in ("test", "demo", "development", "dev")
        ):
            raise ValueError(
                "Experimental docking profiles are restricted to explicit "
                "test/demo/development variants"
            )
        self.profiles = profiles
        self.vina = ExecutableRunner(
            "AutoDock Vina",
            executable=vina_executable,
            default_command="vina",
            version_args=("--version",),
        )
        self.ligand_preparer = ExecutableRunner(
            "RDKit/Meeko ligand preparation",
            executable=python_executable or sys.executable,
            version_args=("--version",),
        )
        self.ligand_prep_timeout_seconds = ligand_prep_timeout_seconds
        self.docking_timeout_seconds = docking_timeout_seconds
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.cpu = cpu
        self.seed = seed
        self.allow_experimental_profiles = allow_experimental_profiles
        self.runtime_variant = runtime_variant
        self.last_result: VinaDockingResult | None = None

    @property
    def source_version(self) -> str:
        vina_version = (self.vina.version or "unknown").replace("\n", " ")
        return (
            f"vina={vina_version};rdkit={_package_version('rdkit') or 'missing'};"
            f"meeko={_package_version('meeko') or 'missing'}"
        )

    def dock(self, *, smiles: str, target: TargetCandidate) -> float:
        return self.dock_with_provenance(smiles=smiles, target=target).score_kcal_mol

    def readiness(self, target: TargetCandidate | None = None) -> dict[str, Any]:
        packages = {
            "rdkit": _package_version("rdkit"),
            "meeko": _package_version("meeko"),
        }
        reason = None
        profile_id = None
        profile_status = None
        production_ready = None
        if any(value is None for value in packages.values()):
            reason = "RDKit or Meeko is not installed"
        elif target is None:
            reason = "Target-specific docking profile readiness has not been evaluated"
        else:
            try:
                profile = self._require_profile(target)
                profile_id = profile.profile_id
                profile_status = profile.profile_status
                production_ready = profile.production_ready
            except (ScientificToolUnavailable, ValueError) as exc:
                reason = str(exc)
        ready = reason is None
        return {
            "tool": "AutoDock Vina",
            "operation": "score_binding_pose",
            "status": "completed" if ready else "unavailable",
            "backend": "local_executable",
            "version": self.vina.version,
            "fixture": False,
            "required": True,
            "reason": reason,
            "provenance": {
                "binary_filename": self.vina.executable_name,
                "packages": packages,
                "profile_count": self.profiles.profile_count,
                "selected_profile_id": profile_id,
                "selected_profile_status": profile_status,
                "selected_profile_production_ready": production_ready,
                "experimental_profile_override": self.allow_experimental_profiles,
                "runtime_variant": self.runtime_variant,
                "target_specific_check": target is not None,
            },
        }

    def dock_with_provenance(
        self,
        *,
        smiles: str,
        target: TargetCandidate,
    ) -> VinaDockingResult:
        profile = self._require_profile(target)
        with TemporaryDirectory(prefix="drugpilot-vina-") as temp_value:
            temp_dir = Path(temp_value)
            ligand_path = temp_dir / "ligand.pdbqt"
            output_path = temp_dir / "docked.pdbqt"
            prep = self.ligand_preparer.run(
                (
                    "-m",
                    "drugpilot.tools.ligand_prep_worker",
                    "--smiles",
                    smiles,
                    "--output",
                    str(ligand_path),
                    "--seed",
                    str(self.seed),
                ),
                timeout_seconds=self.ligand_prep_timeout_seconds,
            )
            try:
                prep_metadata = json.loads(prep.stdout)
            except json.JSONDecodeError as exc:
                raise ScientificToolExecutionError(
                    "Ligand preparation returned invalid provenance JSON",
                    provenance=prep.provenance,
                ) from exc
            if not ligand_path.is_file() or ligand_path.stat().st_size == 0:
                raise ScientificToolExecutionError(
                    "Ligand preparation did not create a PDBQT file",
                    provenance=prep.provenance,
                )

            center = profile.box.center
            size = profile.box.size
            vina = self.vina.run(
                (
                    "--receptor",
                    str(profile.receptor_pdbqt),
                    "--ligand",
                    str(ligand_path),
                    "--center_x",
                    str(center[0]),
                    "--center_y",
                    str(center[1]),
                    "--center_z",
                    str(center[2]),
                    "--size_x",
                    str(size[0]),
                    "--size_y",
                    str(size[1]),
                    "--size_z",
                    str(size[2]),
                    "--exhaustiveness",
                    str(self.exhaustiveness),
                    "--num_modes",
                    str(self.num_modes),
                    "--cpu",
                    str(self.cpu),
                    "--seed",
                    str(self.seed),
                    "--out",
                    str(output_path),
                ),
                timeout_seconds=self.docking_timeout_seconds,
            )
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise ScientificToolExecutionError(
                    "Vina completed without a docked-pose file",
                    provenance=vina.provenance,
                )
            pose_text = output_path.read_text(encoding="utf-8", errors="replace")
            match = _VINA_TABLE_SCORE.search(vina.stdout)
            if match is None:
                match = _VINA_REMARK_SCORE.search(pose_text)
            if match is None:
                raise ScientificToolExecutionError(
                    "Vina output did not contain a parseable mode-1 affinity",
                    provenance=vina.provenance,
                )
            score = float(match.group(1))
            result = VinaDockingResult(
                score_kcal_mol=score,
                profile_provenance=profile.provenance(),
                ligand_preparation=prep_metadata,
                ligand_preparation_execution=prep.provenance,
                vina_execution=vina.provenance,
                prepared_ligand_sha256=_file_digest(ligand_path),
                docked_pose_sha256=_file_digest(output_path),
            )
            self.last_result = result
            return result

    def _require_profile(self, target: TargetCandidate) -> ValidatedDockingProfile:
        profile = self.profiles.require(target)
        if profile.production_ready:
            return profile
        if self.allow_experimental_profiles:
            return profile
        raise ScientificToolUnavailable(
            f"Docking profile {profile.profile_id} is {profile.profile_status} "
            "and is not production_ready"
        )


__all__ = [
    "DockingBox",
    "DockingProfileRegistry",
    "ValidatedDockingProfile",
    "VinaDockingAdapter",
    "VinaDockingResult",
]
