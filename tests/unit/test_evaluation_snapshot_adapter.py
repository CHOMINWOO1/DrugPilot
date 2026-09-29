from __future__ import annotations

import os
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode, RunSnapshot, RunStage
from drugpilot.evaluation import (
    BenchmarkRunner,
    EvaluationMode,
    EvaluationQuery,
    EvaluationRun,
    RunSnapshotAdapter,
    SnapshotConversionError,
    benchmark_digest,
    build_snapshot_evaluation_run,
    load_run_snapshot,
)
from drugpilot.evaluation.io import BenchmarkArtifact
from drugpilot.evaluation.models import (
    BenchmarkCase,
    BenchmarkClassification,
    BenchmarkDataset,
    BenchmarkLabels,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def persisted_snapshot(brief) -> RunSnapshot:
    hypotheses = MA1Agent(
        Settings(
            mode=RunMode.MOCK,
            public_evidence_enabled=False,
            dynamic_target_discovery_enabled=False,
            dynamic_target_fallback_enabled=True,
            demo_fixture_fallback_enabled=True,
        )
    ).run(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    candidates = MA2Agent(ma2_portfolio_enabled=True).run(plan, hypotheses)
    started = datetime(2026, 8, 21, 1, 0, tzinfo=timezone.utc)
    return RunSnapshot(
        run_id="run_snapshot_adapter_test",
        mode=RunMode.MOCK,
        stage=RunStage.G1,
        status_message="Persisted evaluation export",
        brief=brief,
        hypothesis_packet=hypotheses,
        plan_spec=plan,
        candidate_packet=candidates,
        created_at=started,
        updated_at=started + timedelta(seconds=12.5),
    )


def _query(snapshot: RunSnapshot, *, disease: str | None = None) -> EvaluationQuery:
    return EvaluationQuery(
        case_id="snapshot-case-001",
        disease=disease or snapshot.brief.disease,
        research_goal=snapshot.brief.research_goal,
    )


def _benchmark(snapshot: RunSnapshot) -> BenchmarkArtifact:
    dataset = BenchmarkDataset(
        dataset_id="snapshot-adapter-test",
        version="1.0.0-test",
        title="Snapshot adapter unit benchmark",
        classification=BenchmarkClassification.DEMO_NON_PRODUCTION,
        description="Illustrative unit-test labels only.",
        created_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
        cases=(
            BenchmarkCase(
                case_id="snapshot-case-001",
                disease=snapshot.brief.disease,
                research_goal=snapshot.brief.research_goal,
                labels=BenchmarkLabels(
                    relevant_target_ids=(
                        snapshot.hypothesis_packet.hypotheses[0].target.symbol,
                    ),
                    expected_target_branches=tuple(
                        item.target.symbol
                        for item in snapshot.hypothesis_packet.hypotheses
                    ),
                    annotation_note="Illustrative unit-test labels.",
                ),
            ),
        ),
        labels_are_illustrative=True,
        production_approved=False,
    )
    return BenchmarkArtifact(
        dataset=dataset,
        digest=benchmark_digest(dataset),
        source_path=Path("unit-test-benchmark.json"),
    )


def test_snapshot_adapter_preserves_targets_candidates_calls_and_provenance(
    persisted_snapshot,
) -> None:
    prediction = RunSnapshotAdapter(code_revision="test-rev").convert(
        persisted_snapshot,
        _query(persisted_snapshot),
    )

    packet = persisted_snapshot.candidate_packet
    assert packet is not None
    assert len(prediction.targets) == len(packet.target_executions)
    assert len(prediction.candidates) == len(packet.candidates)
    assert prediction.runtime_ms == 12_500.0
    assert prediction.provenance.source_run_id == persisted_snapshot.run_id
    assert prediction.provenance.source_run_stage == "g1"
    assert prediction.provenance.source_run_mode == "mock"
    assert prediction.provenance.source_run_status_message == "Persisted evaluation export"
    assert prediction.provenance.code_revision == "test-rev"
    assert len(prediction.provenance.source_snapshot_digest or "") == 64
    assert prediction.provenance.label_fields_received == ()
    assert prediction.provenance.benchmark_payload_accessed is False

    by_id = {item.candidate_id: item for item in prediction.candidates}
    original = packet.candidates[0]
    converted = by_id[original.candidate_id]
    assert original.name in converted.identifiers
    assert original.smiles in converted.identifiers
    if original.inchikey:
        assert original.inchikey in converted.identifiers
    tool_attempts = [item for item in prediction.execution_attempts if item.kind == "tool"]
    assert {item.call_id for item in tool_attempts} == {
        item.call_id for item in packet.tool_calls
    }
    assert all(item.raw_status for item in tool_attempts)
    assert all(item.fixture for item in tool_attempts)


def test_adapter_rejects_wrong_public_case_and_never_needs_labels(
    persisted_snapshot,
) -> None:
    adapter = RunSnapshotAdapter()
    query = _query(persisted_snapshot)
    first = adapter.convert(persisted_snapshot, query)
    second = adapter.convert(persisted_snapshot, query.model_copy())

    assert first == second
    assert not hasattr(query, "labels")
    with pytest.raises(SnapshotConversionError, match="disease"):
        adapter.convert(
            persisted_snapshot,
            _query(persisted_snapshot, disease="unrelated disease"),
        )


def test_load_snapshot_accepts_direct_and_export_wrapper_json(
    tmp_path,
    persisted_snapshot,
) -> None:
    direct = tmp_path / "direct.json"
    direct.write_text(persisted_snapshot.model_dump_json(), encoding="utf-8")
    wrapper = tmp_path / "wrapper.json"

    assert load_run_snapshot(direct).run_id == persisted_snapshot.run_id
    # Use a normal JSON wrapper rather than relying on SQLite or opening a DB.
    wrapper.write_text(
        json.dumps({"snapshot_json": persisted_snapshot.model_dump_json()}),
        encoding="utf-8",
    )
    assert load_run_snapshot(wrapper).run_id == persisted_snapshot.run_id


def test_complete_snapshot_run_is_immediately_scoreable(persisted_snapshot) -> None:
    benchmark = _benchmark(persisted_snapshot)
    run = build_snapshot_evaluation_run(
        benchmark,
        {"snapshot-case-001": persisted_snapshot},
        mode=EvaluationMode.DETERMINISTIC,
        code_revision="test-rev",
    )

    report = BenchmarkRunner(k_values=(1, 3)).evaluate(benchmark, run)

    assert run.mode is EvaluationMode.DETERMINISTIC
    assert report.retrieval_isolation_verified is True
    assert report.metric("target_branch_coverage").value == 1.0


def test_cli_exports_evaluation_run_without_database_or_network(
    tmp_path,
    persisted_snapshot,
) -> None:
    benchmark = _benchmark(persisted_snapshot)
    benchmark_path = tmp_path / "benchmark.json"
    checksum_path = tmp_path / "benchmark.sha256"
    snapshot_path = tmp_path / "snapshot.json"
    output_path = tmp_path / "evaluation-run.json"
    benchmark_path.write_text(benchmark.dataset.model_dump_json(indent=2), encoding="utf-8")
    checksum_path.write_text(
        f"{benchmark.digest}  benchmark.json\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(persisted_snapshot.model_dump_json(), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_snapshot_evaluation.py"),
        "--benchmark",
        str(benchmark_path),
        "--checksum",
        str(checksum_path),
        "--snapshot",
        f"snapshot-case-001={snapshot_path}",
        "--mode",
        "deterministic",
        "--output",
        str(output_path),
        "--run-id",
        "cli-snapshot-eval",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    run = EvaluationRun.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert run.run_id == "cli-snapshot-eval"
    assert run.predictions[0].provenance.source_run_id == persisted_snapshot.run_id
