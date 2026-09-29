import sqlite3

import pytest

from scripts.snapshot_database import snapshot


def test_snapshot_includes_committed_wal_and_development_writes_are_isolated(tmp_path):
    source, destination = tmp_path / "shared.sqlite3", tmp_path / "dev.sqlite3"
    with sqlite3.connect(source) as shared:
        shared.execute("PRAGMA journal_mode=WAL")
        shared.execute("CREATE TABLE runs(run_id TEXT)")
        shared.execute("INSERT INTO runs VALUES ('historical')")
        shared.commit()
        report = snapshot(source, destination)
        assert report["quick_check"] == "ok"
        assert report["table_counts"]["runs"] == 1
        with sqlite3.connect(destination) as dev:
            dev.execute("INSERT INTO runs VALUES ('development-only')")
            dev.commit()
            assert dev.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert shared.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    with pytest.raises(ValueError):
        snapshot(source, destination)
    with pytest.raises(ValueError):
        snapshot(source, source)
