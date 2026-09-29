"""Consistent, non-destructive SQLite snapshot for an isolated development service."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def snapshot(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(strict=True), destination.resolve()
    if source == destination or destination.exists():
        raise ValueError("Snapshot destination must be a new, separate file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(destination) as copied:
            original.backup(copied, pages=2048)
            check = copied.execute("PRAGMA quick_check").fetchone()[0]
            if check != "ok":
                raise RuntimeError(f"Snapshot integrity check failed: {check}")
            tables = [row[0] for row in copied.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )]
            counts = {name: copied.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                      for name in tables}
    return {"source": str(source), "destination": str(destination), "quick_check": check,
            "table_counts": counts, "bytes": destination.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.source, args.destination), ensure_ascii=False))
