from __future__ import annotations

from importlib.util import find_spec

import pytest
from httpx import ASGITransport, AsyncClient

from drugpilot.api.app import create_app
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode
from drugpilot.storage import SQLiteRunStore
from drugpilot.tools.structure_renderer import (
    MAX_SMILES_LENGTH,
    StructureRenderingError,
    _render_canonical_smiles_svg,
    render_structure_svg,
)


def _app(tmp_path):
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "structure-api.sqlite3",
        public_evidence_enabled=False,
        demo_fixture_fallback_enabled=True,
        ma2_backend_mode="fixture",
        mock_step_delay_ms=0,
    )
    return create_app(settings=settings, store=SQLiteRunStore(settings.database_path))


@pytest.mark.skipif(find_spec("rdkit") is None, reason="RDKit optional dependency is absent")
@pytest.mark.asyncio
async def test_structure_svg_endpoint_renders_and_supports_private_etag_cache(tmp_path) -> None:
    app = _app(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/chemistry/structure.svg",
            params={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        )
        cached = await client.get(
            "/api/chemistry/structure.svg",
            params={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            headers={"If-None-Match": response.headers["etag"]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    assert response.headers["cache-control"] == "private, max-age=86400, immutable"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.text.lstrip().startswith("<?xml")
    assert "<svg" in response.text
    assert cached.status_code == 304
    assert cached.content == b""


@pytest.mark.asyncio
async def test_structure_svg_endpoint_rejects_invalid_and_overlong_input_without_echo(
    tmp_path,
) -> None:
    app = _app(tmp_path)
    secret = "not a SMILES <private-value>"
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        invalid = await client.get(
            "/api/chemistry/structure.svg",
            params={"smiles": secret},
        )
        overlong = await client.get(
            "/api/chemistry/structure.svg",
            params={"smiles": "C" * (MAX_SMILES_LENGTH + 1)},
        )

    assert invalid.status_code == 400
    assert invalid.json() == {"detail": "invalid or unsupported SMILES"}
    assert secret not in invalid.text
    assert overlong.status_code == 400
    assert len(overlong.text) < 100


@pytest.mark.asyncio
async def test_structure_svg_endpoint_sanitizes_renderer_failure(
    tmp_path,
    monkeypatch,
) -> None:
    import drugpilot.api.app as app_module

    def fail_renderer(smiles: str):
        raise StructureRenderingError(f"sensitive renderer detail for {smiles}")

    monkeypatch.setattr(app_module, "render_structure_svg", fail_renderer)
    app = _app(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/chemistry/structure.svg",
            params={"smiles": "CCO"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "2D structure rendering is unavailable"}
    assert "sensitive" not in response.text


@pytest.mark.skipif(find_spec("rdkit") is None, reason="RDKit optional dependency is absent")
def test_structure_svg_renderer_caches_canonical_drawing() -> None:
    _render_canonical_smiles_svg.cache_clear()

    first = render_structure_svg("CCO")
    second = render_structure_svg("OCC")

    assert first is second
    assert first.canonical_smiles == "CCO"
    assert _render_canonical_smiles_svg.cache_info().hits == 1

