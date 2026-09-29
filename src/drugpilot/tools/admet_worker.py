from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import io
import json
from pathlib import Path
import sys
from typing import Any


_REQUIRED_PACKAGES = {
    "admet-ai": "admet_ai",
    "chemprop": "chemprop",
    "torch": "torch",
    "rdkit": "rdkit",
}


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _model_bundle_hash(models_dir: Path) -> str:
    combined = sha256()
    paths = sorted(models_dir.glob("**/*.pt"))
    if not paths:
        raise FileNotFoundError("ADMET-AI contains no packaged model checkpoints")
    for path in paths:
        file_digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        combined.update(path.relative_to(models_dir).as_posix().encode("utf-8"))
        combined.update(file_digest.digest())
    return combined.hexdigest()


def _worker_identity() -> dict[str, Any]:
    return {
        "python_isolated": bool(sys.flags.isolated),
        "no_user_site": bool(sys.flags.no_user_site),
        "worker_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _emit(payload: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    sys.__stdout__.flush()


def check() -> int:
    """Verify this interpreter's packages/checkpoints without loading a model."""

    packages = {name: _version(name) for name in _REQUIRED_PACKAGES}
    importable: dict[str, bool] = {}
    package_root: Path | None = None
    for package_name, module_name in _REQUIRED_PACKAGES.items():
        try:
            spec = find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        importable[package_name] = spec is not None
        if module_name == "admet_ai" and spec is not None:
            locations = list(spec.submodule_search_locations or ())
            if len(locations) == 1:
                package_root = Path(locations[0]).resolve()

    models_dir = (
        package_root / "resources" / "models"
        if package_root is not None
        else None
    )
    checkpoint_count = (
        len([path for path in models_dir.rglob("*.pt") if path.is_file()])
        if models_dir is not None and models_dir.is_dir()
        else 0
    )
    missing = [
        name
        for name in _REQUIRED_PACKAGES
        if packages[name] is None or not importable[name]
    ]
    identity = _worker_identity()
    ready = bool(
        not missing
        and checkpoint_count > 0
        and identity["python_isolated"] is True
        and identity["no_user_site"] is True
    )
    _emit(
        {
            "type": "readiness",
            "status": "ready" if ready else "unavailable",
            "packages": packages,
            "package_modules_importable": importable,
            "checkpoint_count": checkpoint_count,
            "missing_packages": missing,
            "model_loaded": False,
            "inference_executed": False,
            "network_access_required": False,
            **identity,
        }
    )
    return 0 if ready else 2


def serve() -> int:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from admet_ai import ADMETModel
            from admet_ai.constants import DEFAULT_MODELS_DIR

            model = ADMETModel(num_workers=0)
        metadata = {
            "admet_ai_version": _version("admet-ai"),
            "chemprop_version": _version("chemprop"),
            "torch_version": _version("torch"),
            "rdkit_version": _version("rdkit"),
            "model_bundle_sha256": _model_bundle_hash(Path(DEFAULT_MODELS_DIR)),
            "ensemble_count": model.num_ensembles,
            "device": model.device,
            "drugbank_reference": "ADMET-AI packaged approved-drug reference",
            **_worker_identity(),
        }
    except Exception as exc:
        _emit(
            {
                "type": "startup_error",
                "error_type": type(exc).__name__,
                "detail": str(exc),
            }
        )
        return 2
    _emit({"type": "ready", "metadata": metadata})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = str(request["id"])
            smiles = str(request["smiles"])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                raw = model.predict(smiles=smiles)
            if not isinstance(raw, dict):
                raise TypeError("ADMET-AI returned a non-dictionary prediction")
            predictions = {
                str(key): float(value)
                for key, value in raw.items()
                if value is not None
            }
            _emit(
                {
                    "type": "prediction",
                    "id": request_id,
                    "predictions": predictions,
                }
            )
        except Exception as exc:
            _emit(
                {
                    "type": "prediction_error",
                    "id": str(request.get("id", "unknown")) if "request" in locals() else "unknown",
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent ADMET-AI inference worker")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--serve", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return check() if args.check else serve()


if __name__ == "__main__":  # pragma: no cover - exercised as a child process
    raise SystemExit(main())
