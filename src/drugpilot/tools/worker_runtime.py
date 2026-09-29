from __future__ import annotations

from hashlib import sha256
import json
import os
from typing import Mapping


WORKER_ENVIRONMENT_SCHEMA = "drugpilot_scientific_worker_environment/v1"
ISOLATED_PYTHON_FLAGS = ("-I", "-B")

# Only operating-system values needed to start Python or load native scientific
# libraries may cross the process boundary.  In particular, Python import-path
# controls, user-site controls, proxy settings, credentials, and arbitrary
# application variables are never inherited.
_INHERITED_ENVIRONMENT_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_FIXED_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def isolated_worker_environment(
    parent_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the complete, minimal environment for scientific workers.

    The returned mapping is intended to be passed directly as ``env=``.  It is
    not an overlay.  Parent ``PYTHONPATH``/``PYTHONHOME`` and possible secret or
    proxy variables therefore cannot enter the child process.  Windows keys
    are matched case-insensitively and emitted with one canonical spelling so
    the frozen manifest is stable.
    """

    source = os.environ if parent_environment is None else parent_environment
    casefolded: dict[str, str] = {}
    for raw_key, raw_value in source.items():
        key = str(raw_key)
        value = str(raw_value)
        if "\x00" in key or "\x00" in value:
            continue
        casefolded[key.casefold()] = value

    environment = {
        key: casefolded[key.casefold()]
        for key in _INHERITED_ENVIRONMENT_KEYS
        if casefolded.get(key.casefold())
    }
    environment.update(_FIXED_ENVIRONMENT)
    return dict(sorted(environment.items()))


def validate_isolated_worker_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Validate a previously frozen worker environment without parent reads."""

    supplied = {str(key): str(value) for key, value in environment.items()}
    if any("\x00" in key or "\x00" in value for key, value in supplied.items()):
        raise ValueError("scientific worker environment contains NUL")
    allowed = set(_INHERITED_ENVIRONMENT_KEYS) | set(_FIXED_ENVIRONMENT)
    if set(supplied) - allowed:
        raise ValueError("scientific worker environment contains non-allowlisted keys")
    if any(supplied.get(key) != value for key, value in _FIXED_ENVIRONMENT.items()):
        raise ValueError("scientific worker environment constants do not match")
    return dict(sorted(supplied.items()))


def worker_environment_snapshot(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Create a value-redacted but exact freeze manifest for one environment."""

    selected = (
        isolated_worker_environment()
        if environment is None
        else validate_isolated_worker_environment(environment)
    )
    keys = sorted(selected)
    return {
        "schema_version": WORKER_ENVIRONMENT_SCHEMA,
        "environment_keys": keys,
        "environment_keys_sha256": _canonical_hash(keys),
        "environment_sha256": _canonical_hash(selected),
        "python_isolation_flags": list(ISOLATED_PYTHON_FLAGS),
        "python_isolation_flags_sha256": _canonical_hash(
            list(ISOLATED_PYTHON_FLAGS)
        ),
        "parent_pythonpath_inherited": False,
        "parent_pythonhome_inherited": False,
        "parent_user_site_inherited": False,
        "arbitrary_parent_variables_inherited": False,
        "environment_values_included": False,
    }


__all__ = [
    "ISOLATED_PYTHON_FLAGS",
    "WORKER_ENVIRONMENT_SCHEMA",
    "isolated_worker_environment",
    "validate_isolated_worker_environment",
    "worker_environment_snapshot",
]
