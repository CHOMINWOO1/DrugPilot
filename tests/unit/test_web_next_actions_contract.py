from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_reaudit_buttons_follow_api_eligibility_and_g0_does_not_offer_g1():
    script = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")
    assert "loadReviewEligibility(snapshot)" in script
    assert 'status?.modes?.[kind]?.eligible !== true' in script
    assert '!!s.candidate_packet && !!s.g1_verdict' in script
    assert 'G0 근거 검토에서 종료됐습니다.' in script
    assert 'G0에서 종료된 실행입니다. 아래는 저장된 미해결 조치 기록' in script
    assert '자료 업로드나 후보 승인을 요구하는 단계가 아닙니다.' in script
    assert '${disabled("automatic")}' in script
    assert '${disabled("sources")}' in script
    assert '${disabled("external")}' in script
    assert '전체 재작업 ${s.rework_loops || 0}회' in script


def test_next_action_panel_and_hold_only_human_gate_are_rendered() -> None:
    page = (ROOT / "src/drugpilot/web/templates/index.html").read_text(encoding="utf-8")
    script = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")
    styles = (ROOT / "src/drugpilot/web/static/app.css").read_text(encoding="utf-8")

    assert 'id="nextActionPanel"' in page
    assert 'id="nextActionContent"' in page
    assert "renderNextActions();" in script
    assert 'human_gate_mode === "action_required_hold"' in script
    assert "APPROVE가 잠겨" in script
    assert "synthetic_results_forbidden" not in page
    assert "시스템은 측정값이나 통과 기준을 임의 생성하지 않습니다" in script
    assert "20260929-ma3-admission-v1" in page
    assert 'function g0FollowupHtml()' in script
    assert 'g0_followup_sources_completed' in script
    assert '!openReview && !g0Progress' in script
    assert 'automaticG0Running ? "" : reviewControlsHtml()' in script
    assert 'function semanticFailureHtml()' in script
    assert '!state.snapshot.human_decision && !g0StoppedBeforeMa2(state.snapshot)' in script
    assert '논문에 근거가 없다는 뜻이 아닙니다.' in script
    assert '조회 성공은 근거 적격성이나 통과를 뜻하지 않습니다.' in script
    assert "function g0StoppedBeforeMa2" in script
    assert 'node.classList.add("is-skipped")' in script
    assert "G0 감사 미충족으로 Planner·MA2·G1은 실행되지 않았습니다." in script
    assert ".stage-node.is-skipped" in styles
    assert ".next-action-grid" in styles
    assert "minmax(min(100%, 330px), 1fr)" in styles
    assert "overflow-wrap: anywhere" in styles
    assert 'const openReview = state.snapshot?.stage === "human_gate"' in script
    assert "reviewControlsHtml() + reviewHistoryHtml()" in script
    assert "actionPacket?.actions?.length ?? automaticResult?.remaining_findings?.length ?? 0" in script
    assert "미해결 항목 ${outstandingCount}건" in script
    assert ".review-action-dialog .dialog-header { grid-template-columns: minmax(0, 1fr) 32px; }" in styles
