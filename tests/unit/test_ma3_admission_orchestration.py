"""Isolated admission persistence and no-bypass contracts; no paid calls."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import RunMode, GateVerdictCode
from drugpilot.orchestration.graph import _hypothesis_packet_scientific_sha256, _ma1_completion_message
from drugpilot.orchestration.hypothesis_admission import review_submission


def submission(brief):
    p = MA1Agent(settings=Settings(_env_file=None, mode=RunMode.MOCK,
        auto_activate_openai=False, dynamic_target_discovery_enabled=False)).run(brief)
    p.provisional_hypotheses = p.hypotheses
    p.hypotheses = []
    p.metadata["admission_policy"] = "ma3-source-bound-v1"
    return p


class MemoryStore:
    def __init__(self):
        self.items = {}

    async def read_artifact(self, run_id, artifact_id, artifact_type):
        return deepcopy(self.items.get((run_id, artifact_id, artifact_type)))


class Harness:
    def __init__(self):
        self.settings = SimpleNamespace(ma1_provisional_admission_enabled=True)
        self.ma3 = SimpleNamespace(reviewer_model="gpt-5.6-sol", reviewer_provider=object())
        self.store = MemoryStore()
        self.events = []

    async def _artifact_transition(self, run, stage, message, event, node, aid, atype, payload, metadata=None):
        raw = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else deepcopy(payload)
        key = (run, aid, atype)
        assert key not in self.store.items or self.store.items[key] == raw
        self.store.items[key] = raw
        self.events.append((event, metadata))

    async def _record_llm_usage(self, *args):
        raise AssertionError("No provider dispatch permitted in these tests")


@pytest.mark.asyncio
async def test_admission_result_cached_and_original_preserved(brief, monkeypatch):
    import drugpilot.agents.hypothesis_admission as admission
    real = admission.assess_provisional
    calls = []

    async def fake(*args, **kwargs):
        calls.append(kwargs["provider"])
        kwargs["provider"] = None
        return await real(*args, **kwargs)

    monkeypatch.setattr(admission, "assess_provisional", fake)
    o, p = Harness(), submission(brief)
    original = p.model_dump(mode="json")
    first = await review_submission(o, brief, p, "unit")
    second = await review_submission(o, brief, p, "unit")
    assert first == second
    assert len(calls) == 1
    assert p.model_dump(mode="json") == original
    assert not first.provisional_hypotheses and not first.hypotheses
    assert first.metadata["ma3_admission_report"]["submitted_count"] == len(p.provisional_hypotheses)
    assert o.events[0][0] == "ma3_admission_started"


@pytest.mark.asyncio
async def test_partial_batch_reuses_per_hypothesis_cache_without_changing_report_hash(brief, monkeypatch):
    import drugpilot.agents.hypothesis_admission as admission
    real = admission.assess_provisional
    o, p = Harness(), submission(brief)

    async def interrupted(*args, **kwargs):
        await kwargs["before_dispatch"]("test-dispatch")
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(admission, "assess_provisional", interrupted)
    with pytest.raises(RuntimeError):
        await review_submission(o, brief, p, "unit")

    async def resumed(*args, **kwargs):
        assert kwargs["provider"] is o.ma3.reviewer_provider
        assert await kwargs["load_review"]("test-dispatch") == {"ambiguous": True}
        kwargs["provider"] = None
        return await real(*args, **kwargs)

    monkeypatch.setattr(admission, "assess_provisional", resumed)
    output = await review_submission(o, brief, p, "unit")
    report = deepcopy(output.metadata["ma3_admission_report"])
    expected = report.pop("report_sha256")
    assert admission._digest(report) == expected
    assert not output.hypotheses


@pytest.mark.asyncio
async def test_identical_review_keys_are_namespaced_by_run(brief, monkeypatch):
    import drugpilot.agents.hypothesis_admission as admission
    o, p = Harness(), submission(brief)

    async def interrupted(*args, **kwargs):
        assert await kwargs["load_review"]("same-review") is None
        await kwargs["before_dispatch"]("same-review")
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(admission, "assess_provisional", interrupted)
    for run in ("run_a", "run_b"):
        with pytest.raises(RuntimeError):
            await review_submission(o, brief, p, run)
    ids = [aid for run, aid, kind in o.store.items if kind == "MA3HypothesisReviewAttempt"]
    assert len(ids) == 2 and len(set(ids)) == 2


@pytest.mark.asyncio
async def test_pending_submission_cannot_pass_g0_or_planner(brief):
    p = submission(brief)
    verdict = await MA3AssuranceAgent().audit_g0(brief, p)
    assert verdict.verdict != GateVerdictCode.PASS
    assert any(f.code == "PROVISIONAL_HYPOTHESES_NOT_REVIEWED" for f in verdict.findings)
    with pytest.raises(ValueError, match="provisional"):
        PlannerAgent().run(p, brief)


@pytest.mark.asyncio
async def test_new_policy_cannot_fall_back_to_legacy_g0_approval(brief):
    p = submission(brief)
    p.hypotheses = p.provisional_hypotheses
    p.provisional_hypotheses = []
    verdict = await MA3AssuranceAgent().audit_g0(brief, p)
    assert verdict.verdict != GateVerdictCode.PASS
    assert any(f.code == "MA3_ADMISSION_CERTIFICATE_INVALID" for f in verdict.findings)


def test_pending_science_affects_digest_and_completion_is_not_rejection(brief):
    p = submission(brief)
    original_hash = _hypothesis_packet_scientific_sha256(p)
    assert "provisional" in _ma1_completion_message(p, intake_abstain_recommended=False)
    p.provisional_hypotheses[0].mechanism += " changed"
    assert _hypothesis_packet_scientific_sha256(p) != original_hash
