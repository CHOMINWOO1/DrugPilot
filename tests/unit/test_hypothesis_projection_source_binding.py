"""Source-change detection only; no live database or model operations."""
from pathlib import Path

from drugpilot.orchestration.human_review import source_digest


def test_hypothesis_projection_module_is_bound_by_source_digest(monkeypatch):
    before = source_digest()
    original = Path.read_bytes
    touched = []

    def read(path):
        raw = original(path)
        if path.name == "hypothesis_projection.py":
            touched.append(path)
            return raw + b"\n# synthetic test-only source change\n"
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    assert source_digest() != before
    assert len(touched) == 1


def test_source_digest_is_stable_without_changes():
    assert source_digest() == source_digest()
