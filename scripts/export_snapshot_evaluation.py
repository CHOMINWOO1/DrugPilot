from __future__ import annotations

import argparse
from pathlib import Path

from drugpilot.evaluation import (
    EvaluationMode,
    build_snapshot_evaluation_run,
    load_benchmark,
    load_run_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert persisted DrugPilot RunSnapshot JSON exports into one "
            "offline EvaluationRun. No database or external source is queried."
        )
    )
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--checksum", required=True, type=Path)
    parser.add_argument(
        "--snapshot",
        action="append",
        required=True,
        metavar="CASE_ID=PATH",
        help="Repeat once for every case in the benchmark.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[item.value for item in EvaluationMode],
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--code-revision")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    mappings: dict[str, Path] = {}
    for value in args.snapshot:
        if "=" not in value:
            parser.error("--snapshot must use CASE_ID=PATH")
        case_id, raw_path = value.split("=", 1)
        case_id = case_id.strip()
        if not case_id or not raw_path.strip():
            parser.error("--snapshot must use non-empty CASE_ID=PATH")
        if case_id in mappings:
            parser.error(f"duplicate snapshot case: {case_id}")
        mappings[case_id] = Path(raw_path.strip())

    benchmark = load_benchmark(
        args.benchmark,
        checksum_path=args.checksum,
    )
    snapshots = {
        case_id: load_run_snapshot(path) for case_id, path in mappings.items()
    }
    run = build_snapshot_evaluation_run(
        benchmark,
        snapshots,
        mode=EvaluationMode(args.mode),
        run_id=args.run_id,
        code_revision=args.code_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"wrote {args.output} ({len(run.predictions)} cases, mode={run.mode.value})"
    )


if __name__ == "__main__":
    main()
