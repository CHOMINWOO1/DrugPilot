from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import shutil
import subprocess
from time import monotonic
from typing import Mapping, Sequence


_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_PATH_IN_TEXT = re.compile(r"[A-Za-z]:[\\/][^\r\n]*")
_POSIX_PATH_IN_TEXT = re.compile(r"(?<!\w)/(?:[^\s/]+/)+[^\s]*")


class ScientificToolError(RuntimeError):
    """Base class for fail-closed scientific-tool failures."""


class ScientificToolUnavailable(ScientificToolError):
    """Raised when an explicitly required executable/model is unavailable."""


class ScientificDataUnavailable(ScientificToolError):
    """Raised when a live source cannot supply the required scientific input."""


class ScientificToolExecutionError(ScientificToolError):
    """Raised when an executable returns a non-zero exit status."""

    def __init__(self, message: str, *, provenance: "ExecutionProvenance") -> None:
        super().__init__(message)
        self.provenance = provenance


class ScientificToolTimeout(ScientificToolError):
    """Raised when a scientific process exceeds its declared wall-time limit."""

    def __init__(self, message: str, *, provenance: "ExecutionProvenance") -> None:
        super().__init__(message)
        self.provenance = provenance


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    tool_name: str
    executable_name: str
    executable_version: str | None
    sanitized_argv: tuple[str, ...]
    timeout_seconds: float
    started_at: str
    finished_at: str
    runtime_seconds: float
    exit_code: int | None
    timed_out: bool
    stdout_sha256: str
    stderr_sha256: str
    stdout_excerpt: str
    stderr_excerpt: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutableResult:
    stdout: str
    stderr: str
    provenance: ExecutionProvenance


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _digest(value: bytes | str | None) -> str:
    if value is None:
        payload = b""
    elif isinstance(value, bytes):
        payload = value
    else:
        payload = value.encode("utf-8", errors="replace")
    return sha256(payload).hexdigest()


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith(("/", "./", "../", "~"))
        or bool(_WINDOWS_ABSOLUTE_PATH.match(value))
        or "\\" in value
    )


def sanitize_argument(value: str) -> str:
    """Redact secrets and local paths before persisting command provenance."""

    lowered = value.casefold()
    marker_text = re.sub(r"[^a-z0-9]", "_", lowered)
    if any(marker in marker_text for marker in _SECRET_MARKERS):
        if "=" in value:
            key, _ = value.split("=", 1)
            return f"{key}=<redacted>"
        return "<redacted>"

    if "=" in value:
        key, candidate = value.split("=", 1)
        if _looks_like_path(candidate):
            return f"{key}=<path:{Path(candidate).name or 'value'}>"
    if _looks_like_path(value):
        return f"<path:{Path(value).name or 'value'}>"
    return value


def sanitize_arguments(values: Sequence[str]) -> tuple[str, ...]:
    """Sanitize argv while also redacting values after secret-bearing flags."""

    sanitized: list[str] = []
    redact_next = False
    for value in values:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        clean = sanitize_argument(value)
        sanitized.append(clean)
        marker_text = re.sub(r"[^a-z0-9]", "_", value.casefold())
        redact_next = (
            value.startswith("-")
            and "=" not in value
            and any(marker in marker_text for marker in _SECRET_MARKERS)
        )
    return tuple(sanitized)


def bounded_excerpt(value: bytes | str | None, *, limit: int = 1_000) -> str:
    text = _decode(value).replace("\x00", "")
    # Avoid accidentally persisting a credential echoed by an upstream tool.
    lines: list[str] = []
    for line in text.splitlines():
        marker_text = re.sub(r"[^a-z0-9]", "_", line.casefold())
        if any(marker in marker_text for marker in _SECRET_MARKERS):
            lines.append("<redacted line>")
        else:
            redacted = _WINDOWS_PATH_IN_TEXT.sub("<redacted local path>", line)
            redacted = _POSIX_PATH_IN_TEXT.sub("<redacted local path>", redacted)
            lines.append(redacted)
    compact = "\n".join(lines)
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}…<truncated>"


class ExecutableRunner:
    """Run one configured executable without a shell and retain bounded provenance.

    The runner never records the absolute executable path, working directory,
    environment values, or unbounded process output. Scientific adapters can
    therefore attach ``ExecutionProvenance.as_dict()`` to an artifact without
    leaking local paths or credentials.
    """

    def __init__(
        self,
        tool_name: str,
        *,
        executable: str | Path | None = None,
        default_command: str | None = None,
        version_args: Sequence[str] = ("--version",),
        version_timeout_seconds: float = 10.0,
    ) -> None:
        if not tool_name.strip():
            raise ValueError("tool_name must not be blank")
        if version_timeout_seconds <= 0:
            raise ValueError("version_timeout_seconds must be positive")
        requested = str(executable) if executable is not None else default_command
        if not requested:
            raise ValueError("executable or default_command is required")
        resolved = shutil.which(requested)
        if resolved is None:
            candidate = Path(requested).expanduser()
            if candidate.is_file():
                resolved = str(candidate.resolve())
        if resolved is None:
            raise ScientificToolUnavailable(
                f"Required executable for {tool_name!r} was not found"
            )
        self.tool_name = tool_name
        self.executable = Path(resolved)
        self.version_args = tuple(version_args)
        self.version_timeout_seconds = version_timeout_seconds
        self._version: str | None = None

    @property
    def executable_name(self) -> str:
        return self.executable.name

    @property
    def version(self) -> str | None:
        if self._version is not None:
            return self._version
        if not self.version_args:
            return None
        try:
            completed = subprocess.run(
                [str(self.executable), *self.version_args],
                shell=False,
                capture_output=True,
                timeout=self.version_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = bounded_excerpt(completed.stdout or completed.stderr, limit=240)
        self._version = output.strip() or None
        return self._version

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecutableResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        clean_args = tuple(str(value) for value in args)
        if any("\x00" in value for value in clean_args):
            raise ValueError("command arguments must not contain NUL bytes")
        command = [str(self.executable), *clean_args]
        sanitized = (self.executable_name, *sanitize_arguments(clean_args))
        started_at = _utc_iso()
        started = monotonic()
        stdout: bytes | str | None = b""
        stderr: bytes | str | None = b""
        try:
            completed = subprocess.run(
                command,
                shell=False,
                cwd=str(cwd) if cwd is not None else None,
                env=(dict(env) if env is not None else None),
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            provenance = self._provenance(
                sanitized=sanitized,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                started=started,
                exit_code=completed.returncode,
                timed_out=False,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout
            stderr = exc.stderr
            provenance = self._provenance(
                sanitized=sanitized,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                started=started,
                exit_code=None,
                timed_out=True,
                stdout=stdout,
                stderr=stderr,
            )
            raise ScientificToolTimeout(
                f"{self.tool_name} exceeded {timeout_seconds:g} seconds",
                provenance=provenance,
            ) from exc
        except OSError as exc:
            raise ScientificToolUnavailable(
                f"Unable to start {self.tool_name}: {type(exc).__name__}"
            ) from exc

        if provenance.exit_code != 0:
            raise ScientificToolExecutionError(
                f"{self.tool_name} exited with status {provenance.exit_code}",
                provenance=provenance,
            )
        return ExecutableResult(
            stdout=_decode(stdout),
            stderr=_decode(stderr),
            provenance=provenance,
        )

    def _provenance(
        self,
        *,
        sanitized: tuple[str, ...],
        timeout_seconds: float,
        started_at: str,
        started: float,
        exit_code: int | None,
        timed_out: bool,
        stdout: bytes | str | None,
        stderr: bytes | str | None,
    ) -> ExecutionProvenance:
        return ExecutionProvenance(
            tool_name=self.tool_name,
            executable_name=self.executable_name,
            executable_version=self.version,
            sanitized_argv=sanitized,
            timeout_seconds=timeout_seconds,
            started_at=started_at,
            finished_at=_utc_iso(),
            runtime_seconds=round(monotonic() - started, 6),
            exit_code=exit_code,
            timed_out=timed_out,
            stdout_sha256=_digest(stdout),
            stderr_sha256=_digest(stderr),
            stdout_excerpt=bounded_excerpt(stdout),
            stderr_excerpt=bounded_excerpt(stderr),
        )


__all__ = [
    "ExecutableResult",
    "ExecutableRunner",
    "ExecutionProvenance",
    "ScientificToolError",
    "ScientificDataUnavailable",
    "ScientificToolExecutionError",
    "ScientificToolTimeout",
    "ScientificToolUnavailable",
    "bounded_excerpt",
    "sanitize_argument",
    "sanitize_arguments",
]
