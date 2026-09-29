from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_start_script_keeps_localhost_default_and_requires_explicit_lan_share() -> None:
    script = (ROOT / "start_drugpilot.ps1").read_text(encoding="utf-8")

    assert '[switch]$ShareOnLan' in script
    assert 'if ($ShareOnLan) { "0.0.0.0" } else { "127.0.0.1" }' in script
    assert "LAN sharing is enabled without application login" in script
    assert "LAN URL:" in script


def test_start_script_recovers_from_non_ascii_venv_launcher_failure() -> None:
    script = (ROOT / "start_drugpilot.ps1").read_text(encoding="utf-8")

    assert "pyvenv.cfg" in script
    assert "Lib\\site-packages" in script
    assert '$env:PYTHONPATH = "$SourcePath;$SitePackages"' in script
    assert '$env:DRUGPILOT_ADMET_PYTHON_EXECUTABLE = $Python' in script
