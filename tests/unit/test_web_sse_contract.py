from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dashboard_subscribes_to_event_broker_named_workflow_event() -> None:
    broker = (ROOT / "src/drugpilot/orchestration/events.py").read_text(encoding="utf-8")
    dashboard = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")

    assert '"event: workflow\\n"' in broker
    assert '["workflow", "execution_event"' in dashboard


def test_dashboard_does_not_present_external_cache_as_fixture() -> None:
    dashboard = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")

    assert 'provenance === "external_cache"' in dashboard
    assert "Fixture fallback · NOT CACHE" in dashboard
    assert 'metadata.live === false || provenance === "fixture_fallback"' not in dashboard


def test_dashboard_explains_evidence_routed_candidate_audit_and_uses_research_wording() -> None:
    dashboard = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")
    page = (ROOT / "src/drugpilot/web/templates/index.html").read_text(encoding="utf-8")

    assert "후보 우선순위 결정 요인" in dashboard
    assert "표적 근거성에 따른 MA2 탐색 경로" in dashboard
    assert "선택 경로" in dashboard
    assert "G0 승인 표적 범위" in dashboard
    assert "표적별 경로 판정" in dashboard
    assert "왜 이 순위가 나왔나?" not in dashboard
    assert "Evidence-governed Virtual Lab" in page


def test_dashboard_treats_human_gate_as_a_live_update_end_state_only() -> None:
    dashboard = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")

    assert 'const LIVE_END_STAGES = new Set([...TERMINAL_STAGES, "human_gate"]);' in dashboard
    assert '!LIVE_END_STAGES.has(state.snapshot.stage)' in dashboard
    assert "if (LIVE_END_STAGES.has(snapshot.stage) && !reviewIsRunning())" in dashboard
    assert "!LIVE_END_STAGES.has(state.snapshot.stage) || reviewIsRunning()" in dashboard
    # Human Gate remains the active workflow stage; it must not be rendered as
    # a failed/completed terminal node merely because background refresh stops.
    assert "const terminal = TERMINAL_STAGES.has(stage);" in dashboard


def test_dashboard_preserves_manual_activity_scroll_position() -> None:
    dashboard = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")

    assert "const previousScrollTop = feed.scrollTop;" in dashboard
    assert "const distanceFromBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight;" in dashboard
    assert "const shouldStickToBottom" in dashboard
    assert "feed.scrollTop = shouldStickToBottom ? feed.scrollHeight : previousScrollTop;" in dashboard


def test_dashboard_restores_keyed_artifact_disclosures_after_rerender() -> None:
    dashboard = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")

    assert "openDisclosures: new Set()" in dashboard
    assert 'details[data-disclosure]' in dashboard
    assert "captureDisclosureState(root)" in dashboard
    assert "restoreDisclosureState(root)" in dashboard
    assert "captureDisclosureState(feed)" in dashboard
    assert "restoreDisclosureState(feed)" in dashboard
    assert "state.openDisclosures.has" in dashboard
    assert ':candidate:${candidate.candidate_id' in dashboard
    assert ':hypothesis:${hypothesis.hypothesis_id' in dashboard


def test_dashboard_exposes_normalized_model_and_tool_call_trace() -> None:
    dashboard = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")
    page = (ROOT / "src/drugpilot/web/templates/index.html").read_text(encoding="utf-8")

    assert "/calls`" in dashboard
    assert "공개 데이터 도구" in dashboard
    assert "MA2 과학 계산 도구" in dashboard
    assert "FIXTURE" in dashboard
    assert "exact는 개별 호출 기록" in dashboard
    assert 'id="openToolTrace"' in page
    assert 'id="toolTraceDialog"' in page
