from __future__ import annotations

from hashlib import sha256
import json
import sys
from types import SimpleNamespace

import pytest

from drugpilot.tools import admet_ai_adapter as adapter_module
from drugpilot.tools.admet_ai_adapter import ADMETAIAdapter
from drugpilot.tools import executable as executable_module
from drugpilot.tools.live_ma2_gateway import (
    LiveScientificGateway,
    _BRICS_WORKER_PATH,
)
from drugpilot.tools.executable import ScientificToolUnavailable
from drugpilot.tools.worker_runtime import (
    isolated_worker_environment,
    validate_isolated_worker_environment,
    worker_environment_snapshot,
)


def test_worker_environment_drops_parent_python_controls_secrets_and_proxies():
    parent = {
        "Path": "C:\\Windows\\System32",
        "SystemRoot": "C:\\Windows",
        "TEMP": "C:\\Temp",
        "PYTHONPATH": "C:\\attacker\\sitecustomize",
        "PYTHONHOME": "C:\\attacker",
        "OPENAI_API_KEY": "secret",
        "HTTPS_PROXY": "http://credential@example.invalid",
        "ARBITRARY_FLAG": "poison",
    }

    environment = isolated_worker_environment(parent)

    assert environment["PATH"] == parent["Path"]
    assert environment["SYSTEMROOT"] == parent["SystemRoot"]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert set(environment).isdisjoint(
        {"PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY", "HTTPS_PROXY", "ARBITRARY_FLAG"}
    )


def test_worker_environment_snapshot_is_value_redacted_and_exact():
    first_environment = isolated_worker_environment({"PATH": "one"})
    second_environment = isolated_worker_environment({"PATH": "two"})

    first = worker_environment_snapshot(first_environment)
    second = worker_environment_snapshot(second_environment)

    assert first["environment_values_included"] is False
    assert "one" not in json.dumps(first)
    assert first["environment_sha256"] != second["environment_sha256"]
    assert first["python_isolation_flags"] == ["-I", "-B"]
    assert validate_isolated_worker_environment(first_environment) == first_environment
    with pytest.raises(ValueError, match="non-allowlisted"):
        validate_isolated_worker_environment({**first_environment, "SECRET": "x"})


def test_worker_adapters_reject_source_hash_mismatch_before_child_execution(
    tmp_path,
):
    executable = tmp_path / "configured-python.exe"
    executable.write_bytes(b"configured interpreter")
    with pytest.raises(ScientificToolUnavailable, match="source is not frozen"):
        ADMETAIAdapter(
            python_executable=executable,
            worker_source_sha256="0" * 64,
        )
    with pytest.raises(ScientificToolUnavailable, match="source is not frozen"):
        LiveScientificGateway(
            docking=SimpleNamespace(source_version="none", last_result=None),
            admet=SimpleNamespace(source_version="none", last_result=None),
            client=SimpleNamespace(),
            python_executable=sys.executable,
            brics_worker_source_sha256="0" * 64,
        )


def _ready_payload(*, isolated: bool = True) -> dict[str, object]:
    return {
        "type": "readiness",
        "status": "ready",
        "packages": {
            "admet-ai": "2.0",
            "chemprop": "2.0",
            "torch": "2.0",
            "rdkit": "2026.03",
        },
        "package_modules_importable": {
            "admet-ai": True,
            "chemprop": True,
            "torch": True,
            "rdkit": True,
        },
        "checkpoint_count": 5,
        "missing_packages": [],
        "model_loaded": False,
        "inference_executed": False,
        "network_access_required": False,
        "python_isolated": isolated,
        "no_user_site": True,
        "worker_source_sha256": sha256(
            adapter_module._ADMET_WORKER_PATH.read_bytes()
        ).hexdigest(),
    }


def test_admet_readiness_uses_configured_interpreter_and_isolated_check(
    tmp_path, monkeypatch,
):
    executable = tmp_path / "configured-python.exe"
    executable.write_bytes(b"configured interpreter")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_ready_payload()) + "\n",
            stderr="",
        )

    monkeypatch.setattr(adapter_module.subprocess, "run", fake_run)
    adapter = ADMETAIAdapter(
        python_executable=executable,
        worker_environment=isolated_worker_environment({
            "PATH": "C:\\runtime",
            "PYTHONPATH": "C:\\poison",
            "OPENAI_API_KEY": "secret",
        }),
    )

    readiness = adapter.readiness()

    assert readiness["status"] == "completed"
    assert captured["command"] == [
        str(executable.resolve()),
        "-I",
        "-B",
        str(adapter_module._ADMET_WORKER_PATH),
        "--check",
    ]
    assert captured["kwargs"]["shell"] is False
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert "OPENAI_API_KEY" not in captured["kwargs"]["env"]
    attestation = readiness["provenance"]["worker_runtime_attestation"]
    assert attestation["model_loaded"] is False
    assert attestation["inference_executed"] is False
    assert attestation["network_access_required"] is False


def test_admet_readiness_rejects_nonisolated_worker(tmp_path, monkeypatch):
    executable = tmp_path / "configured-python.exe"
    executable.write_bytes(b"configured interpreter")
    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_ready_payload(isolated=False)) + "\n",
            stderr="",
        ),
    )

    readiness = ADMETAIAdapter(python_executable=executable).readiness()

    assert readiness["status"] == "unavailable"
    assert readiness["provenance"]["configured_interpreter_probe"] is True


def test_admet_serve_uses_same_isolated_command_and_frozen_environment(
    tmp_path, monkeypatch,
):
    executable = tmp_path / "configured-python.exe"
    executable.write_bytes(b"configured interpreter")
    environment = isolated_worker_environment({
        "PATH": "C:\\runtime",
        "PYTHONPATH": "C:\\poison",
        "OPENAI_API_KEY": "secret",
    })
    adapter = ADMETAIAdapter(
        python_executable=executable,
        worker_environment=environment,
    )
    adapter._readiness_cache = {
        "status": "completed",
        "reason": None,
        "provenance": {},
    }
    adapter._messages.put({
        "type": "ready",
        "metadata": {
            "model_bundle_sha256": "a" * 64,
            "python_isolated": True,
            "no_user_site": True,
            "worker_source_sha256": sha256(
                adapter_module._ADMET_WORKER_PATH.read_bytes()
            ).hexdigest(),
        },
    })
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.stdin = SimpleNamespace()
            self.stdout = []
            self.stderr = []
            self._returncode = None

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = 0

        def wait(self, timeout=None):
            return self._returncode

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(adapter_module.subprocess, "Popen", fake_popen)
    adapter._ensure_worker()
    try:
        assert captured["command"] == [
            str(executable.resolve()),
            "-I",
            "-B",
            str(adapter_module._ADMET_WORKER_PATH),
            "--serve",
        ]
        assert captured["kwargs"]["shell"] is False
        assert captured["kwargs"]["env"] == environment
        assert "PYTHONPATH" not in captured["kwargs"]["env"]
        assert "OPENAI_API_KEY" not in captured["kwargs"]["env"]
    finally:
        adapter.close()


def test_brics_readiness_spawns_exactly_one_isolated_check_child(monkeypatch):
    calls = []
    payload = {
        "status": "ready",
        "generator": "RDKit BRICS",
        "rdkit_version": "2026.03",
        "python_isolated": True,
        "no_user_site": True,
        "worker_source_sha256": sha256(_BRICS_WORKER_PATH.read_bytes()).hexdigest(),
    }

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(json.dumps(payload) + "\n").encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(executable_module.subprocess, "run", fake_run)
    environment = isolated_worker_environment({
        "PATH": "C:\\runtime",
        "PYTHONPATH": "C:\\poison",
        "OPENAI_API_KEY": "secret",
    })
    gateway = LiveScientificGateway(
        docking=SimpleNamespace(source_version="none", last_result=None),
        admet=SimpleNamespace(source_version="none", last_result=None),
        client=SimpleNamespace(),
        python_executable=sys.executable,
        worker_environment=environment,
    )

    readiness = gateway._brics_readiness()

    assert readiness["status"] == "completed"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        str(gateway.brics.executable),
        "-I",
        "-B",
        str(_BRICS_WORKER_PATH),
        "--check",
    ]
    assert kwargs["shell"] is False
    assert kwargs["env"] == environment
    assert "PYTHONPATH" not in kwargs["env"]
    assert readiness["provenance"]["readiness_execution"][
        "executable_version"
    ] is None
