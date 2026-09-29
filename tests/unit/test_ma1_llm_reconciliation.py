from __future__ import annotations

from hashlib import sha256
import inspect
import json

import pytest
from pydantic import ValidationError

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.config import Settings
from drugpilot.orchestration.graph import (
    DrugPilotOrchestrator,
    MA1_KOREAN_NARRATIVE_INSTRUCTION,
    _ma1_narrative_evidence_view,
    _ma1_narrative_prompt_suffix,
    _ma1_narrative_fallback,
    _ma1_operational_priorities,
)
from drugpilot.providers.structured_outputs import (
    ClaimNarrativeOutput,
    HypothesisNarrativeOutput,
    HypothesisNarrativeOverlayOutput,
)


def _packet(brief):
    settings = Settings(
        mode="mock",
        public_evidence_enabled=False,
        dynamic_target_discovery_enabled=False,
        dynamic_target_fallback_enabled=True,
        demo_fixture_fallback_enabled=True,
    )
    return MA1Agent(settings).run(brief)


def test_ma1_narrative_overlay_freezes_every_authority_field(brief) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    proposed.hypotheses[0].mechanism = "LLM-refined bounded mechanism narrative."
    proposed.hypotheses[0].falsification_test = "LLM-refined falsification test."
    proposed.hypotheses[0].claims[0].text = "LLM-refined claim narrative."

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(
        immutable,
        proposed,
    )

    assert reconciled.hypotheses[0].mechanism == proposed.hypotheses[0].mechanism
    assert (
        reconciled.hypotheses[0].falsification_test
        == proposed.hypotheses[0].falsification_test
    )
    assert reconciled.hypotheses[0].claims[0].text == proposed.hypotheses[0].claims[0].text
    assert reconciled.hypotheses[0].target == immutable.hypotheses[0].target
    assert reconciled.hypotheses[0].evidence == immutable.hypotheses[0].evidence
    assert reconciled.hypotheses[0].score == immutable.hypotheses[0].score
    assert reconciled.hypotheses[0].rank == immutable.hypotheses[0].rank
    assert reconciled.metadata == immutable.metadata
    assert (
        reconciled.hypotheses[0].claims[0].evidence_ids
        == immutable.hypotheses[0].claims[0].evidence_ids
    )
    assert (
        reconciled.hypotheses[0].claims[0].counter_evidence_ids
        == immutable.hypotheses[0].claims[0].counter_evidence_ids
    )
    assert (
        reconciled.hypotheses[0].claims[0].confidence
        == immutable.hypotheses[0].claims[0].confidence
    )
    assert "llm_narrative_attestation" not in reconciled.metadata


def test_ma1_live_narrative_attestation_records_only_bounded_changes(brief) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    hypothesis = proposed.hypotheses[0]
    hypothesis.mechanism = "FLT3 억제로 증식 신호를 낮춘다. " + ("기전 " * 300)
    hypothesis.falsification_test = (
        "FLT3 억제 후에도 세포 증식이 유지되는지 검증한다."
    )
    hypothesis.claims[0].text = "FLT3 활성 감소와 AML 세포 증식 억제를 확인한다."

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(
        immutable,
        proposed,
        attestation_model="gpt-5.6-luna",
        attestation_component="MA1",
    )

    attestation = reconciled.metadata["llm_narrative_attestation"]
    assert attestation["schema_version"] == "1.0"
    assert attestation["component"] == "MA1"
    assert attestation["model"] == "gpt-5.6-luna"
    assert attestation["authority_scope"] == "narrative_only"
    assert attestation["changed_field_count"] == 3
    assert {"target", "evidence", "score", "rank", "claim.confidence"} <= set(
        attestation["frozen_authority_fields"]
    )
    assert {item["field"] for item in attestation["changed_fields"]} == {
        "mechanism",
        "falsification_test",
        "claim.text",
    }
    mechanism_change = next(
        item
        for item in attestation["changed_fields"]
        if item["field"] == "mechanism"
    )
    assert mechanism_change["hypothesis_id"] == hypothesis.hypothesis_id
    assert "claim_id" not in mechanism_change
    assert mechanism_change["after"]["truncated"] is True
    assert len(mechanism_change["after"]["text"]) == 600
    assert mechanism_change["after"]["sha256"] == sha256(
        hypothesis.mechanism.encode("utf-8")
    ).hexdigest()
    claim_change = next(
        item
        for item in attestation["changed_fields"]
        if item["field"] == "claim.text"
    )
    assert claim_change["claim_id"] == hypothesis.claims[0].claim_id
    assert claim_change["after"]["text"] == hypothesis.claims[0].text

    # The attestation is orchestrator-authored metadata; every scientific
    # authority field remains an exact copy of the immutable packet.
    assert reconciled.packet_id == immutable.packet_id
    assert reconciled.brief_id == immutable.brief_id
    assert reconciled.hypotheses[0].target == immutable.hypotheses[0].target
    assert reconciled.hypotheses[0].evidence == immutable.hypotheses[0].evidence
    assert reconciled.hypotheses[0].score == immutable.hypotheses[0].score
    assert reconciled.hypotheses[0].rank == immutable.hypotheses[0].rank
    assert (
        reconciled.hypotheses[0].claims[0].confidence
        == immutable.hypotheses[0].claims[0].confidence
    )


def test_ma1_live_attestation_ignores_malicious_authority_payload(brief) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    proposed.hypotheses[0].mechanism = "FLT3 억제 가설을 한국어로 설명한다."
    # Simulate a provider object that somehow bypassed strict DTO validation.
    # Reconciliation must never read these attacker-controlled authority fields.
    proposed.hypotheses[0].__dict__["target"] = {"symbol": "INVENTED"}
    proposed.hypotheses[0].__dict__["score"] = 1.0
    proposed.hypotheses[0].claims[0].__dict__["confidence"] = 1.0

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(
        immutable,
        proposed,
        attestation_model="gpt-5.6-luna",
    )

    assert reconciled.hypotheses[0].target == immutable.hypotheses[0].target
    assert reconciled.hypotheses[0].score == immutable.hypotheses[0].score
    assert (
        reconciled.hypotheses[0].claims[0].confidence
        == immutable.hypotheses[0].claims[0].confidence
    )
    changed = reconciled.metadata["llm_narrative_attestation"]["changed_fields"]
    assert {item["field"] for item in changed} == {"mechanism"}


def test_ma1_live_prompt_requires_korean_and_verbatim_scientific_tokens() -> None:
    prompt_source = inspect.getsource(DrugPilotOrchestrator._run_ma1)

    assert "MA1_KOREAN_NARRATIVE_INSTRUCTION" in prompt_source
    assert "한국어로 작성하세요" in MA1_KOREAN_NARRATIVE_INSTRUCTION
    for literal in (
        "유전자·단백질 기호",
        "약물명",
        "assay명",
        "데이터 소스명",
        "외부 식별자",
        "표준 과학 단위",
        "번역·음역·정규화·수정하지 마세요",
    ):
        assert literal in MA1_KOREAN_NARRATIVE_INSTRUCTION


def test_ma1_live_prompt_separates_evidence_from_prospective_plan() -> None:
    for literal in (
        "근거 경계",
        "동결된 evidence_ids",
        "counter_evidence_ids",
        "연관성, 유병률 또는 예후",
        "인산화 감소",
        "세포 생존율 감소",
        "전향적 가설",
        "앞으로 수행할 검증 계획",
        "새로 만들지 마세요",
    ):
        assert literal in MA1_KOREAN_NARRATIVE_INSTRUCTION


def test_ma1_prompt_surfaces_exact_operational_human_priorities(brief) -> None:
    operational_en = (
        "2 h p-FLT3/total assay; 72 h viability; isogenic FLT3-WT comparator; "
        "n=3; >=50% reduction; 95% CI"
    )
    operational_ko = "대조군에서 30분 인산화 측정 및 생물학적 반복 3회"
    routed_brief = brief.model_copy(
        update={
            "human_priorities": [
                "Use PubMed and ChEMBL broadly",
                f"  {operational_en}  ",
                operational_ko,
                operational_en,
            ]
        },
        deep=True,
    )
    requirements = _ma1_operational_priorities(routed_brief)

    assert requirements == (operational_en, operational_ko)
    suffix = _ma1_narrative_prompt_suffix(
        routed_brief,
        _packet(routed_brief),
        requirements,
    )
    assert "RESEARCHER_OPERATIONAL_REQUIREMENTS:" in suffix
    assert operational_en in suffix
    assert operational_ko in suffix
    assert "Use PubMed and ChEMBL broadly" not in suffix.split(
        "RESEARCHER_OPERATIONAL_REQUIREMENTS:", 1
    )[1].split("RESEARCH_BRIEF:", 1)[0]


def test_ma1_narrative_view_keeps_linked_sources_without_transport_trace(
    brief,
) -> None:
    packet = _packet(brief).model_copy(deep=True)
    hypothesis = packet.hypotheses[0]
    linked_ids = {
        evidence_id
        for claim in hypothesis.claims
        for evidence_id in (*claim.evidence_ids, *claim.counter_evidence_ids)
    }
    assert linked_ids
    linked = next(
        evidence
        for evidence in hypothesis.evidence
        if evidence.evidence_id in linked_ids
    )
    linked.metadata["excerpt"] = "Exact bounded source sentence."
    linked.metadata["primary_direction_supplement_trace"] = {
        "unused_transport_blob": "DO_NOT_SEND_TO_NARRATIVE_MODEL" * 100_000
    }

    view = _ma1_narrative_evidence_view(packet)
    encoded = json.dumps(view, ensure_ascii=False)

    assert view["source_packet_id"] == packet.packet_id
    assert len(view["source_packet_sha256"]) == 64
    assert "DO_NOT_SEND_TO_NARRATIVE_MODEL" not in encoded
    assert "Exact bounded source sentence." in encoded
    assert linked.evidence_id in encoded
    for claim in hypothesis.claims:
        assert claim.claim_id in encoded
        for evidence_id in (*claim.evidence_ids, *claim.counter_evidence_ids):
            assert evidence_id in encoded
    assert len(encoded) < 100_000


def test_ma1_narrative_view_omits_oversized_source_text_without_truncating(
    brief,
) -> None:
    packet = _packet(brief).model_copy(deep=True)
    hypothesis = packet.hypotheses[0]
    linked_ids = {
        evidence_id
        for claim in hypothesis.claims
        for evidence_id in (*claim.evidence_ids, *claim.counter_evidence_ids)
    }
    linked = next(
        evidence
        for evidence in hypothesis.evidence
        if evidence.evidence_id in linked_ids
    )
    oversized = "SENTENCE_BOUNDARY_MARKER " * 400
    linked.metadata["excerpt"] = oversized

    encoded = json.dumps(_ma1_narrative_evidence_view(packet), ensure_ascii=False)

    assert oversized not in encoded
    assert "OVERSIZED_SOURCE_TEXT_NOT_PARTIALLY_QUOTED" in encoded
    assert sha256(oversized.strip().encode("utf-8")).hexdigest() in encoded


def test_ma1_operational_selector_excludes_source_and_missing_coverage_policy(
    brief,
) -> None:
    evidence_boundary = (
        "Keep the hypothesis and every established claim specific to activating "
        "FLT3 or FLT3-ITD biology. A linked excerpt about association, prevalence, "
        "or prognosis must not be rewritten as evidence of inhibition, "
        "phosphorylation change, or cell killing."
    )
    experiment = (
        "Treat the proposed experiment as a prospective falsification plan, not as "
        "an established result. Test a 0.1 nM to 1 uM concentration series; "
        "quantify p-FLT3 Y591 normalized to total FLT3 at 2 hours and viability at "
        "72 hours; use at least 3 independent biological replicates; compare "
        "MOLM-13 with an FLT3-wild-type AML comparator; falsify the hypothesis if "
        "p-FLT3 is not reduced by at least 50 percent or if viability selectivity is "
        "less than 3-fold with a 95 percent confidence interval that includes 1."
    )
    gate_policy = (
        "At G0 assess disease-target direction and claim-linked primary evidence "
        "only. Compound activity, ADMET, ranking, and candidate safety are "
        "downstream MA2 and G1 work."
    )
    source_policy = (
        "Prefer exact UniProt-linked single-protein measured IC50, Ki, or Kd "
        "evidence. Preserve assay context, units, activity ID, ChEMBL document ID, "
        "PMID or DOI, a short relevant abstract excerpt, its location, and hashes."
    )
    missing_coverage_policy = (
        "Do not interpret missing withdrawal, contraindication, primary-source, or "
        "assay records as negative evidence. Preserve partial or unavailable "
        "coverage explicitly."
    )
    routed_brief = brief.model_copy(
        update={
            "human_priorities": [
                evidence_boundary,
                experiment,
                gate_policy,
                source_policy,
                missing_coverage_policy,
            ]
        },
        deep=True,
    )

    assert _ma1_operational_priorities(routed_brief) == (experiment,)


@pytest.mark.parametrize(
    ("priority", "selected"),
    [
        ("Preserve assay context and units in the evidence package", False),
        ("Missing assay records must remain unavailable", False),
        ("Normalize p-FLT3 to total FLT3 with a vehicle control", True),
        ("Use the prespecified viability endpoint and matched control", True),
        ("생존율을 측정하고 대조군과 비교", True),
    ],
)
def test_ma1_operational_selector_requires_an_executable_condition(
    brief,
    priority: str,
    selected: bool,
) -> None:
    routed_brief = brief.model_copy(
        update={"human_priorities": [priority]},
        deep=True,
    )

    assert bool(_ma1_operational_priorities(routed_brief)) is selected


def test_ma1_reconciliation_preserves_missing_operational_conditions_verbatim(
    brief,
) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    proposed.hypotheses[0].falsification_test = (
        "아래 조건에서 표적 억제와 선택적 생존율 감소가 나타나지 않으면 기각한다."
    )
    requirements = (
        "2 h p-FLT3/total assay with vehicle normalization",
        "72 h viability versus isogenic FLT3-WT comparator; n=3; 95% CI",
    )

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(
        immutable,
        proposed,
        attestation_model="gpt-5.6-terra",
        researcher_operational_requirements=requirements,
    )

    result = reconciled.hypotheses[0].falsification_test
    assert all(requirement in result for requirement in requirements)
    assert result.count("연구자 지정 검증 조건(원문)") == 1
    assert "선택적 생존율 감소가 나타나지 않으면" not in result
    assert reconciled.hypotheses[0].target == immutable.hypotheses[0].target
    assert reconciled.hypotheses[0].evidence == immutable.hypotheses[0].evidence
    assert reconciled.hypotheses[0].claims == immutable.hypotheses[0].claims
    falsification_change = next(
        item
        for item in reconciled.metadata["llm_narrative_attestation"]["changed_fields"]
        if item["field"] == "falsification_test"
    )
    assert falsification_change["after"]["sha256"] == sha256(
        result.encode("utf-8")
    ).hexdigest()


def test_ma1_reconciliation_does_not_duplicate_exact_operational_condition(
    brief,
) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    requirement = "2 h p-FLT3/total assay with vehicle normalization"
    proposed.hypotheses[0].falsification_test = (
        f"{requirement}; 반응이 없으면 가설을 기각한다."
    )

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(
        immutable,
        proposed,
        researcher_operational_requirements=(requirement,),
    )

    assert reconciled.hypotheses[0].falsification_test.count(requirement) == 1
    assert "연구자 지정 검증 조건(원문)" in (
        reconciled.hypotheses[0].falsification_test
    )


def test_ma1_reconciliation_discards_weaker_model_quantifier_after_exact_rule(
    brief,
) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    requirement = (
        "각 후보에서 두 방향성 효과의 95% 신뢰구간이 모두 0을 제외해야 하며 "
        "어느 하나라도 충족하지 않으면 refute 또는 partial로 보존한다."
    )
    proposed.hypotheses[0].falsification_test = (
        requirement
        + " 따라서 어느 방향성 효과의 신뢰구간도 0을 제외하지 못할 때만 "
        "가설을 기각한다."
    )

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(
        immutable,
        proposed,
        researcher_operational_requirements=(requirement,),
    )

    result = reconciled.hypotheses[0].falsification_test
    assert result.count(requirement) == 1
    assert "못할 때만" not in result


def test_ma1_reconciliation_fails_closed_when_exact_conditions_exceed_dto(
    brief,
) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    proposed.hypotheses[0].falsification_test = "검증 계획"

    with pytest.raises(ValueError, match="2000-character strict DTO boundary"):
        DrugPilotOrchestrator._reconcile_ma1_narrative(
            immutable,
            proposed,
            researcher_operational_requirements=("2 h assay " + ("나" * 1_980),),
        )


def test_ma1_reconciliation_matches_reordered_claims_by_id(brief) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    first = proposed.hypotheses[0]
    if len(first.claims) < 2:
        pytest.skip("fixture has fewer than two claims")
    expected_text = {
        item.claim_id: f"Narrative for {item.claim_id}" for item in first.claims
    }
    for item in first.claims:
        item.text = expected_text[item.claim_id]
    first.claims.reverse()

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(immutable, proposed)

    assert [item.claim_id for item in reconciled.hypotheses[0].claims] == [
        item.claim_id for item in immutable.hypotheses[0].claims
    ]
    assert {
        item.claim_id: item.text for item in reconciled.hypotheses[0].claims
    } == expected_text


def test_ma1_reconciliation_accepts_variable_atomic_claim_count_by_exact_id_set(
    brief,
) -> None:
    immutable = _packet(brief)
    hypothesis = immutable.hypotheses[0]
    template = hypothesis.claims[0]
    expanded_claims = [
        *hypothesis.claims,
        *[
            template.model_copy(
                update={
                    "claim_id": f"claim_atomic_extra_{index}",
                    "text": f"Deterministic atomic claim {index}",
                },
                deep=True,
            )
            for index in range(1, 4)
        ],
    ]
    immutable = immutable.model_copy(
        update={
            "hypotheses": [
                hypothesis.model_copy(
                    update={"claims": expanded_claims},
                    deep=True,
                ),
                *immutable.hypotheses[1:],
            ]
        },
        deep=True,
    )
    proposed = _ma1_narrative_fallback(immutable)
    expected_text = {
        claim.claim_id: f"한국어 설명 {claim.claim_id}"
        for claim in proposed.hypotheses[0].claims
    }
    for claim in proposed.hypotheses[0].claims:
        claim.text = expected_text[claim.claim_id]
    proposed.hypotheses[0].claims.reverse()

    reconciled = DrugPilotOrchestrator._reconcile_ma1_narrative(
        immutable,
        proposed,
    )

    assert len(reconciled.hypotheses[0].claims) == len(expanded_claims)
    assert [claim.claim_id for claim in reconciled.hypotheses[0].claims] == [
        claim.claim_id for claim in expanded_claims
    ]
    assert {
        claim.claim_id: claim.text for claim in reconciled.hypotheses[0].claims
    } == expected_text
    for base, final in zip(
        expanded_claims,
        reconciled.hypotheses[0].claims,
        strict=True,
    ):
        assert final.evidence_ids == base.evidence_ids
        assert final.counter_evidence_ids == base.counter_evidence_ids
        assert final.confidence == base.confidence
    assert reconciled.hypotheses[0].evidence == hypothesis.evidence


@pytest.mark.parametrize("mutation", ["missing", "unknown", "duplicate"])
def test_ma1_reconciliation_fails_closed_on_hypothesis_id_mismatch(
    brief,
    mutation: str,
) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    if mutation == "missing":
        proposed.hypotheses.pop()
    elif mutation == "unknown":
        proposed.hypotheses[0].hypothesis_id = "hyp_unknown"
    else:
        proposed.hypotheses.append(proposed.hypotheses[0].model_copy(deep=True))

    with pytest.raises(ValueError, match="hypothesis"):
        DrugPilotOrchestrator._reconcile_ma1_narrative(immutable, proposed)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "duplicate"])
def test_ma1_reconciliation_fails_closed_on_claim_id_mismatch(
    brief,
    mutation: str,
) -> None:
    immutable = _packet(brief)
    proposed = _ma1_narrative_fallback(immutable)
    claims = proposed.hypotheses[0].claims
    if mutation == "missing":
        claims.pop()
    elif mutation == "unknown":
        claims[0].claim_id = "claim_unknown"
    else:
        claims.append(claims[0].model_copy(deep=True))

    with pytest.raises(ValueError, match="claim"):
        DrugPilotOrchestrator._reconcile_ma1_narrative(immutable, proposed)


def test_ma1_wire_schema_rejects_malicious_authority_fields() -> None:
    with pytest.raises(ValidationError):
        HypothesisNarrativeOverlayOutput.model_validate(
            {
                "hypotheses": [
                    {
                        "hypothesis_id": "hyp_1",
                        "mechanism": "bounded narrative",
                        "falsification_test": "bounded test",
                        "claims": [{"claim_id": "claim_1", "text": "claim"}],
                        "target": {"symbol": "INVENTED"},
                        "evidence": [],
                        "score": 1.0,
                        "metadata": {"fixture": True},
                    }
                ]
            }
        )


def test_ma1_wire_schema_has_no_authority_fields() -> None:
    hypothesis_fields = set(HypothesisNarrativeOutput.model_fields)
    claim_fields = set(ClaimNarrativeOutput.model_fields)

    assert hypothesis_fields == {
        "hypothesis_id",
        "mechanism",
        "falsification_test",
        "claims",
    }
    assert claim_fields == {"claim_id", "text"}


@pytest.mark.parametrize("field", ["mechanism", "falsification_test", "claim"])
def test_ma1_wire_schema_rejects_unbounded_narrative(field: str) -> None:
    payload = {
        "hypothesis_id": "hyp_1",
        "mechanism": "bounded mechanism",
        "falsification_test": "bounded test",
        "claims": [{"claim_id": "claim_1", "text": "bounded claim"}],
    }
    if field == "claim":
        payload["claims"][0]["text"] = "x" * 2_001
    else:
        payload[field] = "x" * 2_001

    with pytest.raises(ValidationError, match="length"):
        HypothesisNarrativeOutput.model_validate(payload)
