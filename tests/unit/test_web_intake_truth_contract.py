from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _dashboard() -> str:
    return (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")


def _page() -> str:
    return (ROOT / "src/drugpilot/web/templates/index.html").read_text(
        encoding="utf-8"
    )


def _styles() -> str:
    return (ROOT / "src/drugpilot/web/static/app.css").read_text(encoding="utf-8")


def test_dashboard_consumes_structured_intake_validation_and_cross_field_warnings() -> None:
    dashboard = _dashboard()
    page = _page()

    assert 'id="intakeNotice"' in page
    assert 'role="status"' in page
    assert "function getIntakeAssessment" in dashboard
    assert "ma1Meta.intake_validation" in dashboard
    assert "external_verification_status" in dashboard
    assert "field_mismatches" in dashboard
    assert "cross_field_mismatches" in dashboard
    assert "suggested_normalized_name" in dashboard
    assert "연구 입력 불일치 확인" in dashboard
    assert "대상 질환·연구 목표·표적 세포" in dashboard


def test_unresolved_disease_is_an_amber_intake_abstain_not_a_generic_failure() -> None:
    dashboard = _dashboard()
    styles = _styles()

    assert 'label = "질환 확인 필요"' in dashboard
    assert "입력 검증 · ${esc(title)}" in dashboard
    assert "G0 미실행" in dashboard
    assert "근거 없는 후보 생성을 막기 위한 사전 입력 검증 중단 상태" in dashboard
    assert 'stage === "abstained") node.classList.add("is-abstained")' in dashboard
    assert '["blocked", "failed"].includes(stage)' in dashboard
    assert ".stage-node.is-abstained" in styles
    assert "var(--amber-700)" in styles


def test_hypothesis_score_is_labelled_as_packet_completeness_not_evidence_strength() -> None:
    dashboard = _dashboard()

    assert "가설 패킷 완성도" in dashboard
    assert "질환-표적 연관성이나 과학적 신뢰도 점수가 아닙니다" in dashboard
    assert "근거 점수" not in dashboard


def test_fixture_only_hypotheses_are_explicitly_badged_as_non_research_evidence() -> None:
    dashboard = _dashboard()
    styles = _styles()

    assert "function isFixtureEvidence" in dashboard
    assert "function hypothesisProvenanceBadge" in dashboard
    assert 'item?.supports || "support"' in dashboard
    assert "supportEvidence.every(isFixtureEvidence)" in dashboard
    assert "FIXTURE ONLY" in dashboard
    assert "검증용 자료 · 질환 근거로 사용 불가" in dashboard
    assert "FIXTURE 표적 · 검증용" in dashboard
    assert "실제 질환-표적 근거가 아닌 데모·테스트용 fallback 표적" in dashboard
    assert ".hypothesis-card.is-fixture-only" in styles
    assert ".hypothesis-provenance.fixture-only" in styles


def test_abstained_intake_with_no_hypotheses_explains_zero_generation() -> None:
    dashboard = _dashboard()

    assert "근거가 확인되지 않은 질환에 대해 표적이나 가설을 만들지 않았습니다" in dashboard
    assert "Retained 0 · 가설 패킷 점수 미산정" in dashboard
    assert "후보 생성 중단" in dashboard


def test_later_gate_abstain_is_not_relabelled_as_intake_abstain() -> None:
    dashboard = _dashboard()
    abstain_assignment = dashboard.split("const abstained =", 1)[1].split(";", 1)[0]

    assert "raw.abstain_recommended" in abstain_assignment
    assert "raw.status" in abstain_assignment
    assert "snapshot?.stage" not in abstain_assignment
    assert "snapshot?.g0_verdict" not in abstain_assignment


def test_legacy_unresolved_runs_hide_persisted_hypotheses_candidates_and_human_gate() -> None:
    dashboard = _dashboard()

    assert "const legacyUnresolved = Boolean(" in dashboard
    assert "if (intake.diseaseNeedsReview)" in dashboard
    assert "가설 ${compactNumber(hypotheses.length)}건은 과학적 결과로 제시하지 않습니다" in dashboard
    assert "이전 fallback 실행에서 생성된 후보 ${compactNumber(candidates.length)}건은 과학적 결과로 제시하지 않습니다" in dashboard
    assert "|| intake.requiresInputCorrection" in dashboard
