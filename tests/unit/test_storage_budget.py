from __future__ import annotations

import pytest

from drugpilot.domain.models import (
    BudgetPolicy,
    ExecutionEvent,
    RunMode,
    RunSnapshot,
    RunStage,
    TokenUsage,
)
from drugpilot.orchestration.budget import BudgetExceeded, BudgetGuard
from drugpilot.storage.sqlite import SQLiteRunStore


@pytest.mark.asyncio
async def test_snapshot_artifact_and_hash_chain(tmp_path, brief):
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    snapshot = RunSnapshot(
        run_id="run_test",
        mode=RunMode.MOCK,
        stage=RunStage.CREATED,
        status_message="created",
        brief=brief,
    )
    await store.save_snapshot(snapshot)
    manifest_digest = await store.save_artifact(
        snapshot.run_id, brief.brief_id, "ResearchBrief", brief
    )
    event = ExecutionEvent(
        run_id=snapshot.run_id,
        stage=RunStage.CREATED,
        event_type="created",
        message="Run created",
        artifact_ids=[brief.brief_id],
        metadata={"artifact_sha256": {brief.brief_id: manifest_digest}},
    )
    await store.append_event(event)

    restored = await store.get_snapshot(snapshot.run_id)
    assert restored is not None
    assert restored.brief.brief_id == brief.brief_id
    assert await store.verify_ledger(snapshot.run_id)
    manifest = await store.artifact_manifest(snapshot.run_id)
    assert manifest[0]["artifact_type"] == "ResearchBrief"


def test_budget_guard_enforces_call_cap_at_exact_run_token_boundary():
    policy = BudgetPolicy(
        max_run_tokens=100_000,
        hard_max_run_tokens=120_000,
        max_llm_calls=2,
        max_single_input_tokens=50_000,
        max_run_cost_usd=10,
    )
    guard = BudgetGuard(policy)
    usages = [
        TokenUsage(
            component="MA1",
            model="gpt-5.6-terra",
            input_tokens=40_000,
            output_tokens=10_000,
        )
    ]
    guard.price_usage(usages[0])
    guard.check(usages)
    usages.append(
        TokenUsage(
            component="Planner",
            model="gpt-5.6-terra",
            input_tokens=40_000,
            output_tokens=10_000,
        )
    )
    guard.check(usages)
    usages.append(
        TokenUsage(
            component="MA2",
            model="gpt-5.6-terra",
        )
    )
    with pytest.raises(BudgetExceeded, match="call cap"):
        guard.check(usages)


def test_budget_guard_preflight_rejects_single_input_and_projected_overshoot():
    policy = BudgetPolicy(
        max_run_tokens=100_000,
        hard_max_run_tokens=120_000,
        max_llm_calls=2,
        max_single_input_tokens=50_000,
        max_run_cost_usd=10,
    )
    guard = BudgetGuard(policy)
    with pytest.raises(BudgetExceeded, match="single input cap"):
        guard.preflight(
            [], model="gpt-5.6-terra", input_tokens=50_001, max_output_tokens=1
        )
    with pytest.raises(BudgetExceeded, match="hard token cap"):
        guard.preflight(
            [], model="gpt-5.6-terra", input_tokens=50_000, max_output_tokens=80_001
        )
    with pytest.raises(BudgetExceeded, match="run token cap"):
        guard.preflight(
            [], model="gpt-5.6-terra", input_tokens=50_000, max_output_tokens=50_001
        )
    guard.preflight(
        [], model="gpt-5.6-terra", input_tokens=50_000, max_output_tokens=50_000
    )
    with pytest.raises(BudgetExceeded, match="no approved price"):
        guard.preflight(
            [], model="unpriced-live-reviewer", input_tokens=1_000, max_output_tokens=100
        )
    guard.preflight(
        [], model="claude-opus-4-1", input_tokens=10_000, max_output_tokens=1_000
    )


def test_budget_guard_postflight_enforces_run_and_hard_token_boundaries():
    policy = BudgetPolicy(
        max_run_tokens=100_000,
        hard_max_run_tokens=120_000,
        max_llm_calls=20,
        max_single_input_tokens=50_000,
        max_run_cost_usd=10,
    )
    guard = BudgetGuard(policy)

    guard.check(
        [
            TokenUsage(
                component="MA1",
                model="gpt-5.6-terra",
                input_tokens=50_000,
                output_tokens=50_000,
            )
        ]
    )
    with pytest.raises(BudgetExceeded, match="run token cap"):
        guard.check(
            [
                TokenUsage(
                    component="MA1",
                    model="gpt-5.6-terra",
                    input_tokens=50_000,
                    output_tokens=50_001,
                )
            ]
        )
    with pytest.raises(BudgetExceeded, match="hard token cap"):
        guard.check(
            [
                TokenUsage(
                    component="MA1",
                    model="gpt-5.6-terra",
                    input_tokens=50_000,
                    output_tokens=70_001,
                )
            ]
        )


def test_budget_guard_estimated_live_usage_stops_but_mock_estimate_does_not():
    policy = BudgetPolicy(
        max_run_tokens=100_000,
        hard_max_run_tokens=120_000,
        max_llm_calls=20,
        max_single_input_tokens=50_000,
        max_run_cost_usd=10,
    )
    guard = BudgetGuard(policy)
    live_estimate = TokenUsage(
        component="MA1",
        model="gpt-5.6-luna",
        input_tokens=100,
        output_tokens=20,
        estimated=True,
    )
    mock_estimate = live_estimate.model_copy(
        update={"model": "mock:gpt-5.6-luna"},
        deep=True,
    )

    with pytest.raises(BudgetExceeded, match="live LLM usage is estimated"):
        guard.check([live_estimate])
    with pytest.raises(BudgetExceeded, match="prior live LLM usage is estimated"):
        guard.preflight(
            [live_estimate],
            model="gpt-5.6-luna",
            input_tokens=100,
            max_output_tokens=20,
        )

    guard.check([mock_estimate])
    guard.preflight(
        [mock_estimate],
        model="gpt-5.6-luna",
        input_tokens=100,
        max_output_tokens=20,
    )
