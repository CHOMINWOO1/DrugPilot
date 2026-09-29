"""Offline contract tests: review submissions never imply scientific approval."""
from __future__ import annotations

from hashlib import sha256

import pytest

from drugpilot.agents.ma1 import (
    Critic, Critique, MA1Agent, Router, _prepare_ma3_submission,
)
from drugpilot.config import Settings
from drugpilot.domain.models import (
    Claim, EvidenceRef, Hypothesis, HypothesisPacket, HypothesisTrack,
    ProjectConstraints, ResearchBrief, RunMode, TargetCandidate,
)


def _brief() -> ResearchBrief:
    return ResearchBrief(
        research_goal="Investigate source-bound hypotheses in acute myeloid leukemia.",
        disease="Acute myeloid leukemia",
        constraints=ProjectConstraints(max_ma1_iterations=2),
    )


def _evidence(symbol: str = "FLT3") -> EvidenceRef:
    # Deliberate unit-test text; no network, database, or scientific run write.
    excerpt = f"A laboratory study measured {symbol} in AML cells."
    return EvidenceRef(
        source="PubMed", external_id="unit-test-record", title="Unit test source",
        supports="context", excerpt_hash=sha256(excerpt.encode()).hexdigest(),
        metadata={"excerpt": excerpt, "provenance_class": "external_live"},
    )


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        track=HypothesisTrack.LITERATURE, disease=_brief().disease,
        target=TargetCandidate(symbol="FLT3"), mechanism="Test a prospective mechanism.",
        intervention_direction="inhibit", falsification_test="Measure a controlled dose response.",
        claims=[Claim(text="A prospective hypothesis without established source linkage.")],
        evidence=[_evidence()],
    )


def test_empty_new_field_preserves_historical_serialization_and_hash() -> None:
    packet = HypothesisPacket(brief_id="legacy", hypotheses=[_hypothesis()])
    legacy_payload = packet.model_dump(mode="json")
    assert "provisional_hypotheses" not in legacy_payload
    restored = HypothesisPacket.model_validate(legacy_payload)
    assert restored.provisional_hypotheses == []
    assert restored.model_dump_json() == packet.model_dump_json()
    assert sha256(restored.model_dump_json().encode()).hexdigest() == sha256(
        packet.model_dump_json().encode()).hexdigest()
    assert Settings(_env_file=None).ma1_provisional_admission_enabled is False


def test_semantic_failure_submitted_without_changing_claims_or_evidence() -> None:
    brief, hypothesis = _brief(), _hypothesis()
    before = hypothesis.model_dump(mode="json")
    original = Critic().review(brief, hypothesis)
    assert "UNSUPPORTED_CLAIM" in original.reason_codes
    prepared = _prepare_ma3_submission(brief, original)
    assert prepared.accepted is False
    assert prepared.hypothesis.target.metadata["ma1_admission"]["status"] == "provisional_pending_ma3"
    submitted, excluded = Router().route([prepared], 1, provisional_admission=True)
    assert len(submitted) == 1 and excluded == []
    assert submitted[0].claims[0].evidence_ids == []
    assert submitted[0].evidence == hypothesis.evidence
    assert submitted[0].intervention_direction == hypothesis.intervention_direction
    assert hypothesis.model_dump(mode="json") == before
    assert Router().route([original], 1)[0] == []


def test_legacy_accepted_hypothesis_also_requires_ma3_approval() -> None:
    hypothesis = _hypothesis()
    original = Critique(hypothesis=hypothesis, accepted=True)
    prepared = _prepare_ma3_submission(_brief(), original)
    admission = prepared.hypothesis.target.metadata["ma1_admission"]
    assert admission["legacy_critic_accepted"] is True
    assert admission["scientific_approval"] is False
    assert admission["status"] == "provisional_pending_ma3"


def test_broad_unresolved_placeholder_does_not_pretend_to_violate_selected_actual_direction() -> None:
    brief = _brief()
    brief.constraints.intervention_direction = "unspecified"
    brief.constraints.allowed_hypothesis_directions = ["inhibit"]
    hypothesis = _hypothesis()
    hypothesis.intervention_direction = "modulate"
    hypothesis.target.metadata["direction_resolution"] = {
        "status": "direction_unresolved", "resolved_direction": None,
        "requested_direction": "unspecified", "evidence_ids": [],
    }
    critique = Critic().review(brief, hypothesis)
    assert "HYPOTHESIS_DIRECTION_NOT_ALLOWED" in critique.reason_codes
    prepared = _prepare_ma3_submission(brief, critique)
    admission = prepared.hypothesis.target.metadata["ma1_admission"]
    assert admission["status"] == "provisional_pending_ma3"
    assert admission["direction_filter_deferred_until_source_resolution"] is True
    assert prepared.hypothesis.intervention_direction == "modulate"
    assert prepared.hypothesis.target.metadata["direction_resolution"]["resolved_direction"] is None
    assert prepared.accepted is False


def test_resolved_disallowed_direction_remains_hard_rejected() -> None:
    brief = _brief()
    brief.constraints.intervention_direction = "unspecified"
    brief.constraints.allowed_hypothesis_directions = ["inhibit"]
    hypothesis = _hypothesis()
    hypothesis.intervention_direction = "activate"
    hypothesis.target.metadata["direction_resolution"] = {
        "status": "resolved_from_source", "resolved_direction": "activate"}
    prepared = _prepare_ma3_submission(brief, Critic().review(brief, hypothesis))
    assert prepared.hypothesis.target.metadata["ma1_admission"]["status"] == "excluded_hard_constraint"
    assert "HYPOTHESIS_DIRECTION_NOT_ALLOWED" in prepared.hypothesis.target.metadata["ma1_admission"]["hard_reason_codes"]


@pytest.mark.parametrize("reason", [
    "UNSUPPORTED_CLAIM", "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY",
    "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION", "INTERVENTION_DIRECTION_UNRESOLVED",
    "VARIANT_SCOPE_UNRESOLVED", "DIRECT_TARGET_DIRECTION_DISCORDANT",
])
def test_only_named_semantic_failures_delegated(reason: str) -> None:
    prepared = _prepare_ma3_submission(
        _brief(), Critique(hypothesis=_hypothesis(), accepted=False, reason_codes=(reason,)))
    submitted, excluded = Router().route([prepared], 1, provisional_admission=True)
    assert len(submitted) == 1 and not excluded
    assert reason in submitted[0].target.metadata["ma1_admission"]["legacy_reason_codes"]


@pytest.mark.parametrize("reason", [
    "FORBIDDEN_TARGET", "HYPOTHESIS_DIRECTION_NOT_ALLOWED",
    "MOLECULAR_HYPOTHESIS_SCOPE_INVALID", "MISSING_FALSIFICATION_TEST",
    "BROKEN_EVIDENCE_LINK", "BROKEN_COUNTER_EVIDENCE_LINK",
    "UNAPPROVED_EVIDENCE_SOURCE", "UNRECOGNIZED_FUTURE_FAILURE",
])
def test_hard_and_unknown_failures_stay_excluded(reason: str) -> None:
    prepared = _prepare_ma3_submission(
        _brief(), Critique(hypothesis=_hypothesis(), accepted=False, reason_codes=(reason,)))
    submitted, excluded = Router().route([prepared], 1, provisional_admission=True)
    assert submitted == [] and len(excluded) == 1
    assert reason in excluded[0].reason_codes


@pytest.mark.parametrize("corruption,expected", [
    ("hash", "EVIDENCE_EXCERPT_HASH_INVALID"),
    ("fixture", "EVIDENCE_PROVENANCE_UNVERIFIED"),
    ("no_excerpt", "MISSING_SOURCE_BOUND_LITERATURE"),
    ("no_evidence", "MISSING_EVIDENCE"),
    ("duplicate_id", "EVIDENCE_IDENTITY_INVALID"),
    ("no_claims", "MISSING_CLAIMS"),
])
def test_source_integrity_is_not_relaxed(corruption: str, expected: str) -> None:
    hypothesis = _hypothesis()
    if corruption == "hash":
        hypothesis.evidence[0].excerpt_hash = "0" * 64
    elif corruption == "fixture":
        hypothesis.evidence[0].metadata["fixture"] = True
    elif corruption == "no_excerpt":
        hypothesis.evidence[0].metadata.pop("excerpt")
    elif corruption == "no_evidence":
        hypothesis.evidence = []
    elif corruption == "duplicate_id":
        hypothesis.evidence.append(hypothesis.evidence[0].model_copy(deep=True))
    else:
        hypothesis.claims = []
    prepared = _prepare_ma3_submission(_brief(), Critic().review(_brief(), hypothesis))
    submitted, excluded = Router().route([prepared], 1, provisional_admission=True)
    assert submitted == []
    assert expected in excluded[0].reason_codes


class _ContextRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, brief, seed, iteration):
        self.calls.append(seed.symbol)
        return [_evidence(seed.symbol)]


def test_flag_on_agent_emits_only_bounded_provisional_pool_and_honest_trace() -> None:
    retriever = _ContextRetriever()
    settings = Settings(
        _env_file=None, mode=RunMode.MOCK, max_ma1_iterations=2,
        ma1_provisional_admission_enabled=True,
    )
    packet = MA1Agent(settings=settings, evidence_retriever=retriever).run(_brief())
    assert packet.hypotheses == []
    assert len(packet.provisional_hypotheses) > 0
    assert packet.iterations <= 2
    assert len(retriever.calls) == len(set(retriever.calls))
    assert packet.metadata["admission_policy"] == "ma3-source-bound-v1"
    assert packet.metadata["ma1_executable_hypothesis_count"] == 0
    assert packet.metadata["ma1_provisional_hypothesis_count"] == len(packet.provisional_hypotheses)
    assert packet.metadata["ma1_pool_contract"]["executable_hypothesis_count"] == 0
    for hypothesis in packet.provisional_hypotheses:
        assert hypothesis.target.metadata["ma1_admission"]["scientific_approval"] is False
        assert all(not claim.evidence_ids for claim in hypothesis.claims)
    screened = packet.metadata["screened_target_sources"]
    assert all(item["disposition"] == "provisional" for item in screened)
    trace = packet.metadata["rejected_screening_trace"]
    assert trace["scope"] == "legacy_critic_diagnostics_not_final_disposition"
    assert all(item["diagnostic_only"] is True for item in trace["entries"])


def test_flag_off_agent_keeps_existing_exclusion_behavior() -> None:
    packet = MA1Agent(settings=Settings(_env_file=None, mode=RunMode.MOCK),
                      evidence_retriever=_ContextRetriever()).run(_brief())
    assert not packet.hypotheses and not packet.provisional_hypotheses
    assert packet.excluded
    assert "admission_policy" not in packet.metadata
    assert "provisional_hypotheses" not in packet.model_dump()
