from __future__ import annotations

from fastapi.testclient import TestClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.storage import SQLiteRunStore


def _client(tmp_path, *, max_runs: int = 3) -> TestClient:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "beta-auth.sqlite3",
        beta_access_auth_required=True,
        beta_access_username="beta-reviewer",
        beta_access_password="review-password-1234",
        beta_max_new_runs_per_hour=max_runs,
    )
    return TestClient(
        create_app(settings=settings, store=SQLiteRunStore(settings.database_path))
    )


def test_beta_guard_requires_valid_basic_auth_and_exposes_no_secret(tmp_path) -> None:
    with _client(tmp_path) as client:
        missing = client.get("/api/health")
        wrong = client.get("/api/health", auth=("beta-reviewer", "wrong-password"))
        accepted = client.get(
            "/api/health", auth=("beta-reviewer", "review-password-1234")
        )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"].startswith("Basic")
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert accepted.headers["x-drugpilot-access-control"] == "basic-auth"
    payload = accepted.json()
    assert payload["access_control"] == {
        "authentication_required": True,
        "scheme": "basic",
        "max_new_runs_per_hour": 3,
    }
    assert "review-password" not in accepted.text


def test_beta_guard_limits_new_run_creation_per_process_hour(tmp_path) -> None:
    request = {
        "research_goal": "Identify a traceable target hypothesis for review.",
        "disease": "Acute myeloid leukemia",
    }
    with _client(tmp_path, max_runs=1) as client:
        first = client.post(
            "/api/runs",
            auth=("beta-reviewer", "review-password-1234"),
            json=request,
        )
        second = client.post(
            "/api/runs",
            auth=("beta-reviewer", "review-password-1234"),
            json=request,
        )

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.headers["retry-after"] == "3600"
