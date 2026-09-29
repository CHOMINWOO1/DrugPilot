from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import inspect
import json

import pytest

import drugpilot.agents.ma3_reviewer_projection as projection_module
from drugpilot.agents.ma3 import MA3AssuranceAgent, MA3_REVIEWER_SYSTEM_PROMPT
from drugpilot.agents.ma3_reviewer_projection import (
    GENERIC_G1_REVIEW_PROJECTION_SCHEMA,
    PAPER_NATIVE_V5_G1_REVIEW_PROJECTION_SCHEMA,
    ReviewerProjectionIntegrityError,
    ReviewerProjectionMA3AssuranceAgent,
    build_generic_g1_reviewer_projection,
    build_paper_native_v5_g1_reviewer_projection,
    expand_paper_native_v5_g1_reviewer_projection,
)
from drugpilot.domain.models import HypothesisPacket, PlanSpec
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.providers.llm import (
    MockLLMProvider,
    conservative_structured_input_tokens,
    rough_tokens,
)
from drugpilot.providers.structured_outputs import ReviewerRoutingAssessmentOutput
from drugpilot.tools.live_repurposing_gateway import (
    PAIRED_V4_REQUIRED_POLICY,
)

from test_ma3_primary_abstract_paper_native_v5 import (
    REQUIRED_TOKEN,
    _brief,
    _packet,
    _with_group_projection,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _rehash_projection(raw: dict[str, object]) -> None:
    body = dict(raw)
    body.pop("projection_sha256", None)
    raw["projection_sha256"] = _canonical_hash(body)


def _context(packet):
    brief = _brief()
    hypotheses = HypothesisPacket(
        brief_id=brief.brief_id,
        hypotheses=[],
    )
    plan = PlanSpec(
        plan_id=packet.plan_id,
        hypothesis_packet_id=hypotheses.packet_id,
        tasks=[],
        approved_tools=[],
        stop_conditions=[],
        candidates_per_route=3,
        total_token_budget=10_000,
    )
    return brief, hypotheses, plan


async def _captured_g1_prompt(agent, packet):
    brief, hypotheses, plan = _context(packet)
    prompts: list[str] = []

    async def capture(_model: str, prompt: str, max_output: int) -> None:
        assert max_output == 4_000
        prompts.append(prompt)

    verdict = await agent.audit_g1(
        brief,
        hypotheses,
        plan,
        packet,
        before_review=capture,
    )
    assert len(prompts) == 1
    return prompts[0], verdict


def _top_twelve(packet):
    packet = packet.model_copy(deep=True)
    originals = list(packet.candidates)
    packet.candidates = [
        originals[index % len(originals)].model_copy(
            update={
                "candidate_id": f"cand_worst_case_{index + 1:02d}",
                "name": (
                    f"{originals[index % len(originals)].name} "
                    f"worst-case-{index + 1:02d}"
                ),
            },
            deep=True,
        )
        for index in range(12)
    ]
    return packet


def _first_reference(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        if set(value) == {projection_module._REFERENCE_KEY}:
            return value
        for child in value.values():
            found = _first_reference(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _first_reference(child)
            if found is not None:
                return found
    return None


def _reference_count(value: object, digest: str) -> int:
    if isinstance(value, dict):
        own = int(
            set(value) == {projection_module._REFERENCE_KEY}
            and value[projection_module._REFERENCE_KEY] == digest
        )
        return own + sum(_reference_count(child, digest) for child in value.values())
    if isinstance(value, list):
        return sum(_reference_count(child, digest) for child in value)
    return 0


def test_group4_projection_is_lossless_hashed_and_deduplicates_large_contracts() -> None:
    packet = _with_group_projection(_packet())
    source = packet.model_dump(mode="json")
    source_json = _compact_json(source)

    # This is the production-shaped group-of-four footprint that previously
    # made the exact preflight estimator reject G1 before a paid review call.
    # Contract fields may grow or shrink as long as the source remains a
    # production-sized over-cap packet and the lossless projection stays well
    # below the reviewer boundary.  Pinning an exact fixture byte count made
    # unrelated provenance hardening look like a functional regression.
    assert len(source_json.encode("utf-8")) > 3_000_000
    assert rough_tokens(source_json) > 250_000

    projection = build_paper_native_v5_g1_reviewer_projection(packet)
    restored = expand_paper_native_v5_g1_reviewer_projection(projection)

    assert restored == source
    assert _canonical_bytes(restored) == _canonical_bytes(source)
    assert projection.source_packet_sha256 == _canonical_hash(source)
    assert projection.reviewer_index_sha256 == _canonical_hash(
        projection.reviewer_index.model_dump(mode="json")
    )
    assert len(projection.model_dump_json().encode("utf-8")) < 200_000

    evidence_metadata = source["candidates"][0]["evidence"][0]["metadata"]
    bounds = evidence_metadata["candidate_discovery_bounds"]
    envelope = evidence_metadata["primary_abstract_paper_native_v5"]
    block_hashes = [item.sha256 for item in projection.shared_canonical_blocks]
    raw_projection = projection.model_dump(mode="json")
    for repeated in (bounds, envelope):
        digest = _canonical_hash(repeated)
        assert block_hashes.count(digest) == 1
        assert _reference_count(raw_projection, digest) >= 2


def test_reviewer_index_preserves_every_candidate_conclusion_and_count() -> None:
    packet = _with_group_projection(_packet())
    projection = build_paper_native_v5_g1_reviewer_projection(packet)

    assert projection.reviewer_index.candidate_count == len(packet.candidates)
    assert projection.reviewer_index.tool_call_count == len(packet.tool_calls)
    assert projection.reviewer_index.target_execution_count == len(
        packet.target_executions
    )
    for source, indexed in zip(
        packet.candidates,
        projection.reviewer_index.candidates,
        strict=True,
    ):
        expected_sources = dict(
            sorted(Counter(item.source for item in source.evidence).items())
        )
        assert indexed.candidate_id == source.candidate_id
        assert indexed.name == source.name
        assert indexed.status == source.status
        assert indexed.score == source.score
        assert indexed.rank == source.rank
        assert indexed.target_rank == source.target_rank
        assert indexed.portfolio_rank == source.portfolio_rank
        assert indexed.portfolio_score == source.portfolio_score
        assert indexed.exclusion_reasons == source.exclusion_reasons
        assert indexed.safety_flags == source.safety_flags
        assert indexed.evidence_count == len(source.evidence)
        assert indexed.activity_count == len(source.activities)
        assert indexed.tool_count == len(source.tool_trace_ids)
        assert indexed.tool_trace_count == len(source.tool_trace_ids)
        assert indexed.source_count == len(source.evidence)
        assert indexed.distinct_source_count == len(expected_sources)
        assert indexed.source_counts == expected_sources


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("self_hash", "self hash"),
        ("block_hash", "size mismatch"),
        ("dangling", "dangling"),
        ("cycle", "cyclic"),
        ("duplicate_collision", "duplicate shared block hash collision"),
        ("source_hash", "CandidatePacket hash"),
    ],
)
def test_projection_integrity_tamper_fails_closed(
    mutation: str,
    message: str,
) -> None:
    projection = build_paper_native_v5_g1_reviewer_projection(
        _with_group_projection(_packet())
    )
    raw = projection.model_dump(mode="json")
    blocks = raw["shared_canonical_blocks"]
    assert isinstance(blocks, list) and blocks
    first = blocks[0]
    assert isinstance(first, dict)

    if mutation == "self_hash":
        raw["reference_semantics"] = "tampered"
    elif mutation == "block_hash":
        first["canonical_size_bytes"] += 1
        _rehash_projection(raw)
    elif mutation == "dangling":
        reference = _first_reference(raw["encoded_candidate_packet"])
        assert reference is not None
        reference[projection_module._REFERENCE_KEY] = "0" * 64
        _rehash_projection(raw)
    elif mutation == "cycle":
        digest = first["sha256"]
        first["value"] = {projection_module._REFERENCE_KEY: digest}
        _rehash_projection(raw)
    elif mutation == "duplicate_collision":
        duplicate = deepcopy(first)
        duplicate["value"] = "different value under the same declared hash"
        blocks.append(duplicate)
        _rehash_projection(raw)
    elif mutation == "source_hash":
        raw["source_packet_sha256"] = "0" * 64
        _rehash_projection(raw)
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    with pytest.raises(ReviewerProjectionIntegrityError, match=message):
        expand_paper_native_v5_g1_reviewer_projection(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "exclusion_reasons",
        "source_counts",
        "evidence_count",
        "candidate_count",
    ],
)
def test_reviewer_index_self_consistent_tamper_fails_closed(
    mutation: str,
) -> None:
    projection = build_paper_native_v5_g1_reviewer_projection(
        _with_group_projection(_packet())
    )
    raw = projection.model_dump(mode="json")
    index = raw["reviewer_index"]
    assert isinstance(index, dict)
    candidates = index["candidates"]
    assert isinstance(candidates, list) and candidates
    first = candidates[0]
    assert isinstance(first, dict)

    if mutation == "status":
        first["status"] = "excluded"
    elif mutation == "exclusion_reasons":
        first["exclusion_reasons"] = ["SELF_CONSISTENT_TAMPER"]
    elif mutation == "source_counts":
        source_counts = first["source_counts"]
        assert isinstance(source_counts, dict) and source_counts
        source = next(iter(source_counts))
        source_counts[source] += 1
    elif mutation == "evidence_count":
        first["evidence_count"] += 1
    elif mutation == "candidate_count":
        index["candidate_count"] += 1
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    raw["reviewer_index_sha256"] = _canonical_hash(index)
    _rehash_projection(raw)

    with pytest.raises(
        ReviewerProjectionIntegrityError,
        match="reviewer index differs from expanded CandidatePacket",
    ):
        expand_paper_native_v5_g1_reviewer_projection(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_field", "CandidatePacket schema is invalid"),
        ("coerced_integer", "canonical typed projection"),
    ],
)
def test_expanded_candidate_packet_must_be_exactly_schema_valid(
    mutation: str,
    message: str,
) -> None:
    packet = _with_group_projection(_packet())
    source = packet.model_dump(mode="json")
    projection = build_paper_native_v5_g1_reviewer_projection(packet)
    raw = projection.model_dump(mode="json")
    encoded = raw["encoded_candidate_packet"]
    assert isinstance(encoded, dict)

    if mutation == "extra_field":
        source["unexpected_reviewer_field"] = True
        encoded["unexpected_reviewer_field"] = True
    elif mutation == "coerced_integer":
        source["generation_iterations"] = "1"
        encoded["generation_iterations"] = "1"
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    raw["source_packet_sha256"] = _canonical_hash(source)
    _rehash_projection(raw)

    with pytest.raises(ReviewerProjectionIntegrityError, match=message):
        expand_paper_native_v5_g1_reviewer_projection(raw)


@pytest.mark.asyncio
async def test_exact_v5_inherited_semantic_review_receives_projection_and_full_audit() -> None:
    packet = _with_group_projection(_packet())
    before = packet.model_dump(mode="json")
    agent = ReviewerProjectionMA3AssuranceAgent(
        reviewer_model="reviewer-test-model",
        reviewer_provider=MockLLMProvider(),
    )

    prompt, verdict = await _captured_g1_prompt(agent, packet)
    payload = json.loads(prompt)
    reviewer_candidate = payload["artifacts"]["candidate_packet"]

    assert reviewer_candidate["schema_version"] == (
        PAPER_NATIVE_V5_G1_REVIEW_PROJECTION_SCHEMA
    )
    assert expand_paper_native_v5_g1_reviewer_projection(
        reviewer_candidate
    ) == before
    assert payload["deterministic_findings"] == [
        finding.model_dump(mode="json") for finding in verdict.findings
    ]
    assert payload["validation_results"] == [
        validation.model_dump(mode="json") for validation in verdict.validations
    ]
    assert rough_tokens(prompt) < 150_000
    assert packet.model_dump(mode="json") == before


@pytest.mark.asyncio
async def test_group4_and_worst_top12_prompts_fit_below_150k_vs_full_over_250k() -> None:
    group4 = _with_group_projection(_packet())
    worst_top12 = _top_twelve(group4)
    for packet, expected_schema_bound in (
        (group4, "under_cap"),
        (worst_top12, "over_cap"),
    ):
        full_agent = MA3AssuranceAgent(
            reviewer_model="reviewer-test-model",
            reviewer_provider=MockLLMProvider(),
        )
        projected_agent = ReviewerProjectionMA3AssuranceAgent(
            reviewer_model="reviewer-test-model",
            reviewer_provider=MockLLMProvider(),
        )
        full_prompt, _full_verdict = await _captured_g1_prompt(
            full_agent,
            packet,
        )
        projected_prompt, _projected_verdict = await _captured_g1_prompt(
            projected_agent,
            packet,
        )

        assert rough_tokens(full_prompt) > 250_000
        assert rough_tokens(projected_prompt) < 150_000
        assert rough_tokens(projected_prompt) < rough_tokens(full_prompt)
        schema_bound = conservative_structured_input_tokens(
            system=MA3_REVIEWER_SYSTEM_PROMPT,
            prompt=projected_prompt,
            response_model=ReviewerRoutingAssessmentOutput,
        )
        if expected_schema_bound == "under_cap":
            assert schema_bound < 250_000
        else:
            # The byte-count upper bound intentionally rejects the artificial
            # 12-candidate stress packet even though rough /4 estimation fit.
            assert schema_bound > 250_000


@pytest.mark.asyncio
async def test_non_v5_delegates_to_base_audit_g1_without_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _with_group_projection(_packet())
    brief, hypotheses, plan = _context(packet)
    brief.human_priorities = [f"{REQUIRED_TOKEN}_near"]
    sentinel = object()
    seen: list[tuple[object, ...]] = []

    async def base_audit(
        self,
        actual_brief,
        actual_hypotheses,
        actual_plan,
        actual_candidates,
        *,
        on_usage=None,
        before_review=None,
    ):
        seen.append(
            (
                self,
                actual_brief,
                actual_hypotheses,
                actual_plan,
                actual_candidates,
                on_usage,
                before_review,
            )
        )
        return sentinel

    monkeypatch.setattr(MA3AssuranceAgent, "audit_g1", base_audit)
    agent = ReviewerProjectionMA3AssuranceAgent()
    result = await agent.audit_g1(brief, hypotheses, plan, packet)

    assert result is sentinel
    assert seen == [(agent, brief, hypotheses, plan, packet, None, None)]


def _stable_findings(verdict) -> list[dict[str, object]]:
    return [
        finding.model_dump(mode="json", exclude={"finding_id"})
        for finding in verdict.findings
    ]


def _stable_validations(verdict) -> list[dict[str, object]]:
    return [
        validation.model_dump(mode="json", exclude={"validation_id"})
        for validation in verdict.validations
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "priorities",
    [
        [REQUIRED_TOKEN, REQUIRED_TOKEN],
        [REQUIRED_TOKEN, PAIRED_V4_REQUIRED_POLICY],
    ],
    ids=["duplicate_v5", "v5_with_paired_v4"],
)
async def test_invalid_v5_policy_is_deterministic_without_paid_review(
    priorities: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _with_group_projection(_packet())
    brief, hypotheses, plan = _context(packet)
    brief.human_priorities = list(priorities)
    base = MA3AssuranceAgent(
        reviewer_model="reviewer-test-model",
        reviewer_provider=None,
    )
    expected = await base.audit_g1(brief, hypotheses, plan, packet)

    projection_calls = 0
    reviewer_callbacks = 0

    def forbidden_projection(_packet):
        nonlocal projection_calls
        projection_calls += 1
        raise AssertionError("invalid policy must not build a reviewer projection")

    def forbidden_callback(*_args, **_kwargs):
        nonlocal reviewer_callbacks
        reviewer_callbacks += 1
        raise AssertionError("invalid policy must not enter semantic review")

    monkeypatch.setattr(
        projection_module,
        "build_paper_native_v5_g1_reviewer_projection",
        forbidden_projection,
    )
    agent = ReviewerProjectionMA3AssuranceAgent(
        reviewer_model="reviewer-test-model",
        reviewer_provider=MockLLMProvider(),
    )
    observed = await agent.audit_g1(
        brief,
        hypotheses,
        plan,
        packet,
        on_usage=forbidden_callback,
        before_review=forbidden_callback,
    )

    assert projection_calls == 0
    assert reviewer_callbacks == 0
    assert observed.reviewer_invoked is False
    assert observed.verdict == expected.verdict
    assert _stable_findings(observed) == _stable_findings(expected)
    assert _stable_validations(observed) == _stable_validations(expected)
    assert "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_TOKEN_INVALID" in {
        finding.code for finding in observed.findings
    }
    assert "External semantic review was not invoked" in observed.rationale


def test_base_semantic_review_decision_and_graph_wiring_are_protected() -> None:
    assert (
        ReviewerProjectionMA3AssuranceAgent._semantic_review
        is MA3AssuranceAgent._semantic_review
    )
    assert (
        ReviewerProjectionMA3AssuranceAgent._semantic_review_r18
        is MA3AssuranceAgent._semantic_review_r18
    )
    assert "self._semantic_review_r18(" in inspect.getsource(
        MA3AssuranceAgent.audit_g0
    )
    assert "self._semantic_review_r18(" in inspect.getsource(
        MA3AssuranceAgent.audit_g1
    )
    assert "self._semantic_review_r18(" in inspect.getsource(
        ReviewerProjectionMA3AssuranceAgent.audit_g1
    )
    assert "compact_hypothesis_input(hypotheses)" in inspect.getsource(
        ReviewerProjectionMA3AssuranceAgent.audit_g1
    )
    assert "system=MA3_REVIEWER_SYSTEM_PROMPT" in inspect.getsource(
        MA3AssuranceAgent._semantic_review_r18
    )
    assert "system=MA3_REVIEWER_SYSTEM_PROMPT" in inspect.getsource(
        DrugPilotOrchestrator._preflight_llm
    )


def test_generic_g1_projection_is_lossless_and_smaller() -> None:
    packet = _with_group_projection(_packet())
    projection = build_generic_g1_reviewer_projection(packet)
    raw_packet = packet.model_dump(mode="json")
    raw_projection = projection.model_dump(mode="json")

    assert projection.schema_version == GENERIC_G1_REVIEW_PROJECTION_SCHEMA
    assert projection.source_packet_sha256 == projection_module._canonical_sha256(
        raw_packet
    )
    assert len(projection_module._canonical_bytes(raw_projection)) < len(
        projection_module._canonical_bytes(raw_packet)
    )
    assert ReviewerProjectionMA3AssuranceAgent._decide is MA3AssuranceAgent._decide
    init_source = inspect.getsource(DrugPilotOrchestrator.__init__)
    assert "ReviewerProjectionMA3AssuranceAgent(" in init_source
