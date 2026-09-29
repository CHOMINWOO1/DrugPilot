from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def _page() -> str:
    return (ROOT / "src/drugpilot/web/templates/index.html").read_text(
        encoding="utf-8"
    )


def _dashboard() -> str:
    return (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")


def _styles() -> str:
    return (ROOT / "src/drugpilot/web/static/app.css").read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    return text[start_index : text.index(end, start_index)]


def test_overview_is_the_accessible_default_and_report_is_a_single_tab() -> None:
    page = _page()
    dashboard = _dashboard()
    tabs = _between(page, '<div class="artifact-tabs"', '<div id="artifactContent"')
    snapshot_renderer = _between(
        dashboard, "function renderSnapshot()", "function updateRunStatus"
    )

    assert (
        'id="artifactTabOverview" type="button" role="tab" data-tab="overview"'
        in tabs
    )
    assert 'aria-selected="true" tabindex="0">Overview</button>' in tabs
    assert (
        'id="artifactTabReport" type="button" role="tab" data-tab="report"'
        in tabs
    )
    assert 'aria-selected="false" tabindex="-1">RUN REPORT</button>' in tabs
    assert tabs.count('aria-selected="true"') == 1
    assert tabs.count('data-tab="report"') == 1
    assert 'data-tab="technical"' not in tabs
    assert '>TECHNICAL DATA</button>' not in tabs
    assert '>Raw JSON</button>' not in tabs
    assert (
        'id="artifactContent" class="artifact-content" role="tabpanel" '
        'aria-labelledby="artifactTabOverview" tabindex="0"'
    ) in page

    assert 'selectedTab: "overview"' in dashboard
    assert "selectedArtifactRunId: null" in dashboard
    assert 'if (state.selectedArtifactRunId !== snapshot.run_id)' in snapshot_renderer
    assert "state.selectedArtifactRunId = snapshot.run_id" in snapshot_renderer
    assert 'state.selectedTab = "overview";' in snapshot_renderer
    # A live refresh of the same run must preserve the tab the researcher selected.
    assert snapshot_renderer.count('state.selectedTab = "overview";') == 1

    normalizer = _between(
        dashboard, "function normalizeArtifactTab", "function updateSelectedTabs"
    )
    assert '["brief", "overview"].includes(value)' in normalizer
    assert 'if (["raw", "technical"].includes(value)) return "report"' in normalizer
    assert 'data-stage="brief_validated" data-artifact="brief"' in page

    artifact_renderer = _between(
        dashboard, "function renderArtifact()", "function wireTechnicalData"
    )
    assert "overview: renderOverviewArtifact" in artifact_renderer
    assert "report: renderCombinedRunReportArtifact" in artifact_renderer
    assert "technical:" not in artifact_renderer
    assert "raw:" not in artifact_renderer


def test_run_report_reads_like_a_dossier_without_using_current_server_health() -> None:
    dashboard = _dashboard()
    report = _between(
        dashboard, "function renderRunReportArtifact", "function renderOverviewArtifact"
    )
    assert '<article class="run-report" aria-labelledby="runReportTitle">' in report
    assert 'id="runReportTitle"' in report
    for heading in (
        "핵심 결론",
        "연구 범위와 질환 확인",
        "실제 적용 모델",
        "모델 전달·의사결정 기록",
        "독립 감사와 표적 분기 범위",
        "상위 가설",
        "후보 순위와 상태",
        "안전성 신호와 미해결 항목",
        "실제 실행·근거 추적 정보",
        "연구자 결정과 감사 무결성",
    ):
        assert heading in report

    assert "runReportConclusion(snapshot, intake, portfolio)" in report
    assert "runReportHypothesesHtml(snapshot, intake)" in report
    assert "renderExplicitScopeIdentityTrace(hypothesisPacket || {})" in report
    assert "사용자 지정 표적 전체 처리" in report
    assert "MA1 표적 처리와 상위 가설" in report
    assert report.index("${explicitScopeHtml") < report.index(
        "${runReportHypothesesHtml(snapshot, intake)}"
    )
    assert "runReportCandidatesHtml(snapshot, intake, portfolio)" in report
    assert "runReportTargetCoverageHtml(candidatePacket, portfolio)" in report
    assert "runReportIssueRows(snapshot, intake, portfolio)" in report
    assert "runReportEvidenceSummary(snapshot)" in report
    assert "callCountSummary(snapshot, trace)" in report
    assert "snapshot.human_decision" in report
    assert "state.health" not in report
    assert "현재 서버 상태가 아니라 이 실행에 저장된" in report
    assert "admitted는 계획상 조회 허용을 뜻할 뿐" in report
    assert "고유 근거 레코드" in report
    assert "원본 보존" in report
    assert "JSON.stringify(state.snapshot" not in report


def test_report_truth_rules_fail_closed_for_unverified_fixture_or_incomplete_runs() -> None:
    dashboard = _dashboard()
    conclusion = _between(
        dashboard, "function runReportConclusion", "function runReportEvidenceSummary"
    )
    hypothesis_rows = _between(
        dashboard, "function runReportHypothesesHtml", "function runReportCandidatesHtml"
    )
    candidate_rows = _between(
        dashboard, "function runReportCandidatesHtml", "function renderRunReportArtifact"
    )
    issues = _between(
        dashboard, "function runReportIssueRows", "function runReportTargetCoverageHtml"
    )

    assert "intake.requiresInputCorrection" in conclusion
    assert "intake.verificationUnavailable" in conclusion
    assert "intake.demoFixture" in conclusion
    assert 'candidate_packet?.backend_kind || "").toLowerCase() === "fixture"' in conclusion
    assert "portfolio.abstainRequired" in conclusion
    assert conclusion.index("intake.demoFixture") < conclusion.index(
        "portfolio.abstainRequired"
    )
    assert conclusion.index("portfolio.abstainRequired") < conclusion.index(
        'decision === "APPROVE"'
    )
    assert conclusion.index('decision === "APPROVE"') < conclusion.index(
        'g1 === "PASS"'
    )
    assert "G0/G1 표시와 관계없이 실제 연구 후보 승인 근거로 사용할 수 없습니다" in conclusion
    assert "G1 또는 HumanDecision 표시와 관계없이" in conclusion

    assert "!intake.requiresInputCorrection" in hypothesis_rows
    assert "!intake.verificationUnavailable" in hypothesis_rows
    assert "!intake.demoFixture" in hypothesis_rows
    assert "!hypothesisProvenance(item).fixtureOnly" in hypothesis_rows

    assert 'const scienceFixture = String(packet?.backend_kind || "").toLowerCase() === "fixture"' in candidate_rows
    assert "!scienceFixture && g1 === \"PASS\"" in candidate_rows
    assert 'if (decision === "APPROVE")' in candidate_rows
    assert 'selectedIds.has(candidate.candidate_id)' in candidate_rows
    assert "미선택 · 승인 대상 아님" in candidate_rows
    assert "승인 가능한 추천이 아니라 저장된 후보 기록" in candidate_rows

    assert "LLM MOCK · 실제 모델 추론 아님" in issues
    assert "SCIENCE FIXTURE · 실제 과학 실행 결과 아님" in issues
    assert "실제 후보 승인 근거가 아닙니다" in issues


def test_run_report_combines_human_report_and_lazy_lossless_json_appendix() -> None:
    page = _page()
    dashboard = _dashboard()
    artifact_renderer = _between(
        dashboard, "function renderArtifact()", "function wireTechnicalData"
    )
    combined_view = _between(
        dashboard,
        "function renderCombinedRunReportArtifact",
        "function renderTechnicalDataArtifact",
    )
    technical_wiring = _between(
        dashboard, "function wireTechnicalData", "function wireMoleculeStructures"
    )
    technical_view = _between(
        dashboard, "function renderTechnicalDataArtifact", "function configureHumanGate"
    )
    downloader = _between(
        dashboard, "function downloadSnapshot", "async function copyRunId"
    )

    assert "report: renderCombinedRunReportArtifact" in artifact_renderer
    assert "raw:" not in artifact_renderer
    assert "technical:" not in artifact_renderer
    assert "wireTechnicalData(root)" in artifact_renderer
    assert "renderRunReportArtifact()" in combined_view
    assert "renderTechnicalDataArtifact()" in combined_view
    assert combined_view.index("renderRunReportArtifact()") < combined_view.index(
        "renderTechnicalDataArtifact()"
    )
    assert 'data-tab="technical"' not in page
    assert '>TECHNICAL DATA</button>' not in page
    assert page.count('data-tab="report"') == 1
    assert 'data-disclosure="${esc(snapshot.run_id || "run")}:technical-json"' in technical_view
    assert "data-technical-json" in technical_view
    assert "<summary>" in technical_view
    assert "전체 원본 JSON" in technical_view
    assert 'tabindex="0" aria-label="원본 실행 JSON" data-raw-json' in technical_view
    assert "원본 JSON은 일반적인 결과 읽기 화면이 아니며" in technical_view

    assert "if (!disclosure.open" in technical_wiring
    assert 'output.dataset.loaded === "true"' in technical_wiring
    assert "output.textContent = JSON.stringify(state.snapshot, null, 2)" in technical_wiring
    assert 'disclosure.addEventListener("toggle", populate)' in technical_wiring
    assert "output.innerHTML" not in technical_wiring

    assert "JSON.stringify(state.snapshot, null, 2)" in downloader
    assert 'type: "application/json;charset=utf-8"' in downloader
    assert "`drugpilot-${state.snapshot.run_id}.json`" in downloader
    assert "감사·재현용 전체 실행 스냅샷" in page
    assert "원본 JSON 저장" in page


def test_report_tabs_tables_and_scrollable_json_keep_keyboard_semantics() -> None:
    page = _page()
    dashboard = _dashboard()
    styles = _styles()
    tab_keys = _between(dashboard, "function handleTabKeydown", "async function api")
    selected_tabs = _between(
        dashboard, "function updateSelectedTabs", "function renderArtifact"
    )
    report_helpers = _between(
        dashboard, "function runReportTargetCoverageHtml", "function renderRunReportArtifact"
    )
    technical_view = _between(
        dashboard, "function renderTechnicalDataArtifact", "function configureHumanGate"
    )
    artifact_panel = _between(
        page, '<section class="artifact-panel research-artifacts">', '<aside id="contextPanel"'
    )

    assert 'role="tablist" aria-label="산출물 유형"' in artifact_panel
    assert "aria-live" not in artifact_panel
    assert '["ArrowLeft", "ArrowRight", "Home", "End"]' in tab_keys
    assert "event.preventDefault()" in tab_keys
    assert "tabs[nextIndex].focus()" in tab_keys
    assert "tabs[nextIndex].click()" in tab_keys
    assert 'tab.setAttribute("aria-selected", String(selected))' in selected_tabs
    assert "tab.tabIndex = selected ? 0 : -1" in selected_tabs
    assert 'panel.setAttribute("aria-labelledby", selectedTab.id)' in selected_tabs

    assert report_helpers.count("<caption>") >= 2
    assert report_helpers.count('scope="col"') >= 13
    assert 'role="list" aria-label="저장 순위에 따른 상위 가설"' in report_helpers
    assert "<caption>실행 스냅샷에 저장된 산출물 인덱스</caption>" in technical_view
    assert technical_view.count('scope="col"') == 5
    assert ":focus-visible" in styles
    assert ".technical-json-view {" in styles
    assert "overflow: auto;" in _between(
        styles, ".technical-json-view {", "@media (max-width: 980px)"
    )
    assert "@media (max-width: 700px)" in styles


def test_model_execution_and_llm_change_sections_are_record_driven() -> None:
    dashboard = _dashboard()
    report = _between(
        dashboard, "function renderRunReportArtifact", "function renderOverviewArtifact"
    )
    model_view = _between(
        dashboard, "function modelExecutionHtml", "function llmNarrativeAttestation"
    )
    change_view = _between(
        dashboard, "function llmChangesHtml", "function isSkippedTraceStatus"
    )
    handoff_view = _between(
        dashboard, "function modelDecisionTraceHtml", "function llmNarrativeAttestation"
    )
    trace_loader = _between(
        dashboard, "async function loadCallTrace", "function closeToolTrace"
    )

    assert "실제 적용 모델" in report
    assert "모델 전달·의사결정 기록" in report
    assert "모델 서술 변경 전·후" in report
    assert "modelExecutionHtml(snapshot, { includeHeading: false })" in report
    assert "modelDecisionTraceHtml(snapshot)" in report
    assert "llmChangesHtml(snapshot, { includeHeading: false })" in report

    for component in (
        "MA1-INTAKE",
        "MA1",
        "MA3-G0-REVIEWER",
        "PLANNER",
        "MA3-G1-REVIEWER",
    ):
        assert f'"{component}"' in dashboard
    assert "토큰·비용" in model_view
    assert "허용된 권한" in model_view
    assert "실제 반영 결과" in model_view
    assert "estimated_cost_usd" in model_view
    assert "reasoning_tokens" in model_view
    assert "내부 추론 계수" in model_view
    assert "state.health" not in model_view

    assert "모델끼리 자유 대화하는 구조가 아닙니다" in handoff_view
    assert "structured_generation_response" in handoff_view
    assert "artifact_handoff" in handoff_view
    assert "audit_decision" in handoff_view
    assert "selective_rework" in handoff_view
    assert "workflow_terminal" in handoff_view
    assert "interaction.artifact_ids" in handoff_view
    assert "interaction.evidence_ids" in handoff_view
    assert "비공개 내부 사고과정 원문은 표시하지 않습니다" in handoff_view
    assert "revision_lineage" in handoff_view

    assert "llm_narrative_attestation" in dashboard
    assert "changed_fields" in change_view
    assert "frozen_authority_fields" in change_view
    assert "변경 전" in change_view
    assert "변경 후" in change_view
    assert "이전 실행(legacy run): 변경 전·후 diff가 기록되지 않음" in change_view
    assert "추정하지 않습니다" in change_view
    assert 'api(`/api/runs/${encodeURIComponent(runId)}/calls`)' in trace_loader
    assert 'if (state.selectedTab === "report") renderArtifact()' in trace_loader


def test_abstained_runs_show_the_exact_gate_and_rework_reason_in_korean() -> None:
    page = _page()
    dashboard = _dashboard()
    status_view = _between(
        dashboard, "function runStatusPresentation", "function stageLabel"
    )
    pipeline = _between(dashboard, "function renderPipeline", "function inferLastStageIndex")
    report = _between(
        dashboard, "function renderRunReportArtifact", "function renderOverviewArtifact"
    )

    assert 'snapshot?.human_decision?.decision === "HOLD"' in status_view
    assert "연구자 판단 보류" in status_view
    assert "LLM call cap would be exceeded" in status_view
    assert "LLM 호출 한도 도달" in status_view
    assert "단일 모델 입력 한도 초과" in status_view
    assert "실제 과금 토큰이 아닙니다" in status_view
    assert status_view.index('inputCap)') < status_view.index('g0StoppedBeforeMa2(snapshot)')
    assert "G0 최종 재감사를 완료하지 못해 Planner·MA2·G1에는 진입하지 않았습니다" in status_view
    assert "G0 근거 보강 한도 도달" in status_view
    assert "G0 가설 감사 미충족으로 중단" in status_view
    assert "G1 후보·과정 감사 미충족으로 중단" in status_view
    assert "Planner·MA2·G1은 실행하지 않았습니다" in status_view
    assert "서버 오류가 아니라 근거 품질 기준에 따른 종료" in status_view
    assert "brief?.constraints?.max_rework_loops" in status_view
    assert "snapshot?.rework_loops" in status_view
    assert 'stage === "abstained"' in pipeline
    assert "runStatusPresentation(snapshot).detail" in pipeline
    assert "const statusPresentation = runStatusPresentation(snapshot)" in report
    assert "statusPresentation.label" in report
    assert ">RUN REPORT</button>" in page
    assert ">Overview</button>" in page
    assert ">MA1 Results <" in page
    assert ">Plan</button>" in page
    assert ">Candidates <" in page
    assert ">Audit <" in page


def test_new_reviewer_finding_codes_have_korean_titles_and_summaries() -> None:
    dashboard = _dashboard()
    finding_view = _between(
        dashboard, "function findingGroupTitle", "function findingGroupHtml"
    )

    expected = {
        "COUNTER_EVIDENCE_NOT_FOUND": "명시적 반증 근거 미확인",
        "REVIEWER_INHIBITION_DIRECTION_OPPOSES_TARGET_BIOLOGY": "억제 전략이 표적 생물학과 상충",
        "REVIEWER_DNMT3A_DIRECTION_NOT_ESTABLISHED": "DNMT3A 치료 작용 방향 근거 부족",
        "REVIEWER_CHRONIC_SAFETY_REQUIREMENT_UNADDRESSED": "만성 안전성 요구 미검토",
        "REVIEWER_CEBPA_DRUGGABILITY_CONSTRAINT_MISMATCH": "CEBPA 약물화 가능성 제약 불일치",
        "REVIEWER_COUNTER_EVIDENCE_RELEVANCE_MISMATCH": "반증 근거와 치료 주장 관련성 부족",
        "REVIEWER_SUPPORT_LINKS_INCLUDE_NON_SUPPORTIVE_RECORDS": "지지 근거 연결에 비지지 레코드 포함",
        "REVIEWER_NOVELTY_PRIORITY_NOT_REFLECTED": "신규 표적 우선순위가 결과에 반영되지 않음",
        "REVIEWER_CELL_LINE_CONTEXT_MISROUTED_AS_SINGLE_CELL": "세포주 조건을 단일세포 조건으로 잘못 분류",
    }
    for code, label in expected.items():
        assert code in finding_view
        assert label in finding_view

    assert "function genericFindingTitleKo" in finding_view
    assert "function findingReasonLine" in finding_view
    assert "감사 사유:" in finding_view
    assert "원문 감사 메시지는 상세 보기에서 보존" in finding_view
    assert "감사 사유 요약" in dashboard
    assert "해석과 추가 검토 항목" in dashboard
    assert "원문 감사 기록" in dashboard
    assert "function findingCategoryLabel" in dashboard
    assert "근거 출처·연결" in dashboard
    audit_card = _between(
        dashboard, "function auditArtifactCard", "function renderCombinedRunReportArtifact"
    )
    assert "groupedFindings.map" in audit_card
    assert "groupedFindings.slice(0,8)" not in audit_card


def test_actual_sources_and_scientific_tools_report_attempt_outcomes() -> None:
    dashboard = _dashboard()
    resource_view = _between(
        dashboard,
        "function actualExecutionResourcesHtml",
        "function isFailedTraceStatus",
    )
    report = _between(
        dashboard, "function renderRunReportArtifact", "function renderOverviewArtifact"
    )
    overview = _between(
        dashboard, "function renderOverviewArtifact", "function renderHypothesesArtifact"
    )
    assert "데이터 소스와 MA2 실행 도구" in report
    assert "사용된 데이터 소스" in resource_view
    assert "과학 계산" in resource_view
    assert "파이프라인 작업" in resource_view
    for label in ("실행됨", "성공", "실패", "건너뜀·미설정", "레코드", "실행 근거·사유"):
        assert label in resource_view
    assert "recordedExecutionTrace(snapshot)" in resource_view
    assert "execution.publicCalls" in resource_view
    assert "execution.scientificCalls" in resource_view
    assert "state.health" not in resource_view
    assert "actualExecutionResourcesHtml(snapshot, { includeHeading: false })" in report
    assert "modelExecutionHtml(" not in overview
    assert "actualExecutionResourcesHtml(" not in overview


def test_scientific_execution_truth_overrides_projected_success_with_raw_skipped() -> None:
    dashboard = _dashboard()
    truth_helper = _between(
        dashboard, "function scientificExecutionTruth", "function buildCallTraceFallback"
    )
    fallback = _between(
        dashboard, "function buildCallTraceFallback", "function callCountSummary"
    )
    reconciliation = _between(
        dashboard, "function recordedExecutionTrace", "function modelRolePresentation"
    )
    resources = _between(
        dashboard, "function ma2ExecutionToolPresentation", "function isFailedTraceStatus"
    )

    # A Calls API projection may say success even when the persisted scientific
    # execution was skipped. Raw ToolCallRecord, then candidate execution, must
    # win before the projected status is considered.
    status_expression = _between(truth_helper, "const status =", "const reasonCodes")
    assert status_expression.index("rawExecutionStatus") < status_expression.index(
        "candidateStatus"
    )
    assert status_expression.index("candidateStatus") < status_expression.index(
        "projectedStatus"
    )
    assert "status: truth.status" in fallback
    assert "raw_execution_status: truth.rawExecutionStatus" in fallback
    assert "candidate_execution_statuses: truth.candidateStatuses" in fallback
    assert reconciliation.index("...projected") < reconciliation.index("...rawCall")
    assert 'status_authority: truth.rawExecutionStatus' in fallback
    assert '"candidate_packet.tool_calls.execution_status"' in fallback

    # The old FLT3/Vina truth case must remain visible and not be relabelled as
    # a successful computation.
    assert "DIRECT_MEASURED_ACTIVITY_AVAILABLE" in resources
    assert "OPTIONAL_DOCKING_NOT_ASSESSED" in resources
    assert "LIVE_DOCKING_NOT_CONFIGURED" in resources
    assert 'docking: { category: "scientific_computation", engine: "AutoDock Vina"' in resources
    assert 'chemistry_validator: { category: "scientific_computation", engine: "RDKit"' in resources
    assert 'admet_predictor: { category: "scientific_computation", engine: "ADMET-AI"' in resources


def test_run_report_projects_only_allowlisted_v6_failure_reason_from_raw_trace() -> None:
    dashboard = _dashboard()
    truth_helper = _between(
        dashboard, "function scientificExecutionTruth", "function buildCallTraceFallback"
    )
    resources = _between(
        dashboard, "function ma2ExecutionToolPresentation", "function isFailedTraceStatus"
    )

    assert (
        "PAPER_NATIVE_V6_FAILURE_REASON_CODES.has(rawCall?.provenance?.reason_code)"
        in truth_helper
    )
    assert "? rawCall.provenance.reason_code" in truth_helper
    assert ": null" in truth_helper
    assert "reason_codes: truth.reasonCodes" in dashboard
    assert "executionReasonLabel(reason)" in resources
    assert "<small>${esc(reason)}</small>" in resources


def test_hypothesis_views_use_korean_dossier_labels_and_trace_badges() -> None:
    dashboard = _dashboard()
    styles = _styles()
    helpers = _between(
        dashboard, "function hypothesisEvidenceCounts", "function runReportHypothesesHtml"
    )
    hypothesis_view = _between(
        dashboard, "function renderHypothesesArtifact", "function targetSourceTag"
    )
    report_hypotheses = _between(
        dashboard, "function runReportHypothesesHtml", "function runReportCandidatesHtml"
    )

    for label in ("기전 가설", "검증 주장", "반증 기준"):
        assert label in helpers
    for role in ("support", "counter", "context"):
        assert role in helpers
    assert "mechanism_ko" in helpers
    assert "falsification_test_ko" in helpers
    assert "text_ko" in helpers
    assert "DATA ·" in helpers
    assert "TOOL ·" in helpers
    assert "metadata.scientific_tool_name" in helpers
    assert "metadata.scientific_tool" in helpers
    assert "recordedExecutionTrace" not in helpers
    assert "hypothesisNarrativeBody(hypothesis)" in hypothesis_view
    assert "hypothesisSourceToolBadges(hypothesis)" in hypothesis_view
    assert "hypothesisNarrativeBody(item)" in report_hypotheses
    assert "hypothesisEvidenceCounts(item)" in report_hypotheses
    assert ".hypothesis-narrative-body" in styles
    assert ".hypothesis-execution-badge.is-source" in styles
    assert ".hypothesis-execution-badge.is-tool" in styles
    assert "@media (max-width: 700px)" in styles


def test_same_run_sse_and_call_trace_refresh_preserve_the_selected_report() -> None:
    dashboard = _dashboard()
    server_events = _between(
        dashboard, "function handleServerEvent", "function unwrapSnapshot"
    )
    snapshot_renderer = _between(
        dashboard, "function renderSnapshot()", "function updateRunStatus"
    )
    opener = _between(dashboard, "async function openToolTrace", "async function loadCallTrace")
    loader = _between(dashboard, "async function loadCallTrace", "function closeToolTrace")

    assert "state.snapshot = snapshot" in server_events
    assert "renderSnapshot()" in server_events
    assert "selectedTab" not in server_events
    assert 'if (state.selectedArtifactRunId !== snapshot.run_id)' in snapshot_renderer
    assert snapshot_renderer.count('state.selectedTab = "overview";') == 1

    # The dialog uses the same race-safe loader rather than duplicating writes.
    assert 'await loadCallTrace(runId, { quiet: true })' in opener
    assert 'if (state.snapshot?.run_id !== runId) return' in opener
    assert 'if (state.selectedTab === "report") renderArtifact()' in loader
    assert "renderMetrics()" in loader
    assert "renderActivity()" in loader
    assert not re.search(r'\bstate\.selectedTab\s*=(?!=)', opener + loader)
