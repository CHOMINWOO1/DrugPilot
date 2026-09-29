from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _dashboard() -> str:
    return (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")


def _between(body: str, start: str, end: str) -> str:
    return body[body.index(start) : body.index(end, body.index(start))]


def test_portfolio_summary_counts_terminal_zero_hit_branches_as_evaluated() -> None:
    dashboard = _dashboard()

    assert 'count("concluded_target_count"' in dashboard
    assert '["completed", "no_candidates"].includes' in dashboard
    assert "concluded === eligible" in dashboard
    assert "표적 ${portfolio.concluded}/${portfolio.eligible} 평가" in dashboard
    assert "결론 ${coverage.concluded}/${coverage.eligible}개" in dashboard
    assert "검색 완료·적합 후보 없음" in dashboard
    assert "실행 자격 제외 ${coverage.ineligible}개" in dashboard
    assert "실행 미완료 ${coverage.incomplete}개" in dashboard
    assert "미완료·제외" not in dashboard
    assert "portfolio.completed}/${portfolio.eligible} targets complete" not in dashboard


def test_execution_summary_uses_persisted_call_records_before_trace_api_arrives() -> None:
    dashboard = _dashboard()

    assert "function callCountSummary(snapshot = {}, trace = null)" in dashboard
    assert "if (hasNormalizedTrace)" in dashboard
    assert "snapshot?.candidate_packet?.tool_calls" in dashboard
    assert "fallback.scientific_tool_calls.length" in dashboard
    assert "countPublicDataCalls(snapshot)" in dashboard
    assert "renderMetrics();\n        renderActivity();" in dashboard
    assert "callCountSummary(state.snapshot || {}, trace).totalCalls" in dashboard


def test_execution_summary_uses_authoritative_normalized_call_units() -> None:
    dashboard = _dashboard()
    start = dashboard.index("function callCountSummary(snapshot = {}, trace = null)")
    end = dashboard.index("function recordedExecutionTrace", start)
    helper = dashboard[start:end]
    authoritative = helper[helper.index("if (hasNormalizedTrace)") : helper.index(
        "// Before /calls arrives"
    )]
    fallback = helper[helper.index("// Before /calls arrives") :]

    assert "normalizedTraceCount(summary.model_calls, trace.model_calls)" in authoritative
    assert (
        "normalizedTraceCount(summary.public_data_calls, trace.public_data_calls)"
        in authoritative
    )
    assert (
        "normalizedTraceCount(summary.scientific_tool_calls, trace.scientific_tool_calls)"
        in authoritative
    )
    assert "const normalizedTotal = summary.total_calls" in authoritative
    assert "Math.trunc(normalizedTotal)" in authoritative
    assert "countPublicDataCalls(snapshot)" not in authoritative
    assert "Math.max" not in authoritative
    assert "countPublicDataCalls(snapshot)" in fallback
    assert "totalCalls: composedTotal" in fallback
    assert "상단 호출 수는 통합 호출 API의 정규화 기록 수입니다" in dashboard
    assert "aggregate는 개별 HTTP 요청 수가 아닌 소스별 집계 기록 1건" in dashboard


def test_repeated_ma3_findings_are_grouped_with_raw_right_panel_drilldown() -> None:
    dashboard = _dashboard()

    assert "function groupFindings(findings = [])" in dashboard
    assert 'const code = String(finding.code || finding.category || "AUDIT_FINDING")' in dashboard
    assert "예측 ADMET 위험 · 실험 확인 필요" in dashboard
    assert "모델 예측 안전성 신호 · ${candidateCount || records.length}개 후보 확인" in dashboard
    assert "확증적 제외 아님 · 후속 실험 필요 조건부 경고" in dashboard
    assert 'data-context-kind="finding"' in dashboard
    assert 'else if (kind === "finding")' in dashboard
    assert "function findingContextHtml(group, gate, snapshot = {})" in dashboard
    assert "finding.message || \"원문 메시지 미기록\"" in dashboard


def test_candidate_context_uses_progressive_scientific_disclosure_order() -> None:
    dashboard = _dashboard()
    start = dashboard.index("function candidateContextHtml")
    end = dashboard.index("function evidenceGroupHtml", start)
    context = dashboard[start:end]

    result = context.index("<h4>Result</h4>")
    rationale = context.index("<h4>Rationale</h4>")
    evidence = context.index('evidenceGroupHtml("Evidence"')
    provenance = context.index("<h4>Provenance</h4>")
    trace = context.index("<b>Trace</b>")
    assert result < rationale < evidence < provenance < trace
    assert 'data-disclosure="context-trace:' in context


def test_candidate_context_surfaces_recorded_tool_and_library_provenance() -> None:
    dashboard = _dashboard()

    assert "function candidateToolProvenanceHtml(candidate, packet)" in dashboard
    assert "${candidateToolProvenanceHtml(candidate, packet)}" in dashboard
    assert "검증 도구·출처" in dashboard
    assert "RDKit ${chemistryVersion" in dashboard
    assert "ADMET-AI ${admetModel.admet_ai_version" in dashboard
    assert 'docking?.execution_status === "skipped"' in dashboard
    assert "ChEMBL 측정값" in dashboard
    assert "규제·금지약물" in dashboard
    assert "저장된 CandidatePacket 도구 호출과 provenance 기준" in dashboard


def test_candidate_unavailable_state_shows_bounded_reason_and_audit_drilldown() -> None:
    dashboard = _dashboard()

    assert "function executionReasonLabel(value = \"\")" in dashboard
    assert "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT" in dashboard
    assert "후보별 검증 원저 근거 3건 미만 · 승격 불가" in dashboard
    assert "PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID" in dashboard
    assert "원저 근거의 후보·활성·출처 결속 계약 불일치" in dashboard
    assert "portfolio.reasonCodes.map(executionReasonLabel)" in dashboard
    assert "후보별 사유는 아래 감사 상세, 전체 원본은 기술 데이터에 보존" in dashboard
    assert "paperNativeV6FailureHeadline(packet || {})" in dashboard
    assert "${paperV6FailureDetail}" in dashboard


def test_candidate_v6_failure_detail_is_allowlisted_bounded_and_fail_closed() -> None:
    dashboard = _dashboard()
    detail = _between(
        dashboard,
        "const PAPER_NATIVE_V6_FAILURE_REASON_CODES",
        "function executionGroupRows",
    )

    # Only explicit V6 failure contracts can create the detail. The first code
    # preserves pre-split historical packets; arbitrary exception text is never
    # promoted into the UI.
    for reason in (
        "PAPER_NATIVE_V6_INSUFFICIENT_OR_INVALID_PRIMARY_SUPPORT",
        "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT",
        "PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID",
    ):
        assert reason in detail
    assert "PAPER_NATIVE_V6_FAILURE_REASON_CODES.has(reasonCode)" in detail
    assert "provenance.detail" not in detail
    assert "call.error" not in detail
    assert "delegate_provenance" not in detail
    assert "JSON.stringify(provenance)" not in detail

    # Counts and both the fixed pool and failure-card list remain bounded. An
    # over-cap list discloses its total and omitted count rather than silently
    # dropping cards.
    assert "Number.isInteger(value)" in detail
    assert "value >= minimum && value <= maximum" in detail
    assert "fixedIds.length <= 12" in detail
    assert "records: records.slice(0, 12)" in detail
    assert "totalCount: records.length" in detail
    assert "omittedCount: Math.max(0, records.length - 12)" in detail
    assert "전체 ${compactNumber(totalCount)}건" in detail
    assert "생략 ${compactNumber(omittedCount)}건" in detail

    # Projection v1 is accepted only as a complete, non-fixture, ordered,
    # unique ChEMBL pool with syntactically bounded stored attestations.
    assert 'value.schema_version === "paper_native_v6_wrapper_atomic_screen_projection/v1"' in detail
    assert "value.row_count === rows.length" in detail
    assert "value.complete === true" in detail
    assert "value.fixture === false" in detail
    assert "fixedIds.length === new Set(fixedIds).size" in detail
    assert "CHEMBL_MOLECULE_ID_PATTERN.test(candidateId)" in detail
    assert "row.pool_index === index" in detail
    assert "row.fixed_pool_rank === index + 1" in detail
    assert "row.molecule_chembl_id === fixedIds[index]" in detail
    assert "SHA256_HEX_PATTERN.test(row.wrapper_screen_checks_sha256)" in detail
    assert "SHA256_HEX_PATTERN.test(row.row_sha256)" in detail
    assert "row.fixture === false" in detail

    # Status, reason, and atomic checks are cross-bound to the producer
    # contract. This prevents a corrupted row from being labelled as passed.
    assert 'status === "verified"' in detail
    assert "reasonCode == null" in detail
    assert "checks.cardinality_exact === true" in detail
    assert "checks.primary_mapping === true" in detail
    assert "checks.exact_binding === true" in detail
    assert 'reasonCode === "DELEGATE_NOT_RETURNED"' in detail
    assert "row.delegate_returned === false" in detail
    assert 'reasonCode === "SEED_CARDINALITY_NOT_EXACT"' in detail
    assert 'reasonCode === "PRIMARY_SOURCE_NOT_MAPPING"' in detail
    assert 'reasonCode === "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"' in detail
    assert "checks.exact_binding === false" in detail
    assert "supportCandidateCount !== verifiedCount" in detail
    assert "rejectionCount !== rejectedCount" in detail

    # The packet-level projection adds candidate-specific source rejection
    # reasons only when the server verified both source and wrapper projections.
    assert "function paperNativeV6FailureSummaryProjection" in detail
    assert "value.source_reason_projection_verified === true" in detail
    assert "rawProjection.rowsSha256 === value.rows_sha256" in detail
    assert "sourceReasonCode: row.source_primary_reason_code" in detail
    assert "record.projection.reasonCounts" in detail
    assert "주 사유:" in detail
    assert "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED" in detail
    assert "DOCUMENT_REVIEW_NOT_PRIMARY_ACTIVITY_REPORT" in detail

    # Evidence grades explain usability but never relabel B/C as promotable.
    assert "후보별 정확한 원저 3건 이상 · 승격 가능" in detail
    assert "정확한 원저 1–2건 · HOLD" in detail
    assert "정확한 원저 0건 · 활성 DB 참고만 가능" in detail


def test_candidate_v6_failure_detail_escapes_rows_and_handles_old_bad_or_missing_data() -> None:
    dashboard = _dashboard()
    detail = _between(
        dashboard,
        "const PAPER_NATIVE_V6_FAILURE_REASON_CODES",
        "function executionGroupRows",
    )
    candidate_view = _between(
        dashboard,
        "function renderCandidatesArtifact",
        "function ma2ProcessOverview",
    )

    for escaped in (
        "esc(row.candidateId)",
        "esc(value)",
        "esc(reasonLabel)",
        "esc(record.targetSymbol)",
        "esc(record.reasonCode)",
        "esc(executionReasonLabel(record.reasonCode))",
        "esc(record.projection.rowsSha256)",
        "esc(record.projection.attestationSha256)",
        'esc(state.snapshot?.run_id || "run")',
    ):
        assert escaped in detail
    assert 'record.projection.status === "malformed"' in detail
    assert "행 상세를 표시하지 않습니다" in detail
    assert 'record.source === "packet_reason_only"' in detail
    assert "이전 실행 또는 요약 packet에는" in detail
    assert "추정 행은 만들지 않습니다" in detail
    assert "paperNativeV6WrapperScreenProjection(null)" in detail
    assert candidate_view.count("${paperV6FailureDetail}") == 2


def test_hypothesis_context_surfaces_recorded_source_and_audit_provenance() -> None:
    dashboard = _dashboard()

    assert "function hypothesisToolProvenanceHtml(hypothesis, packet)" in dashboard
    assert "${hypothesisToolProvenanceHtml(hypothesis, state.snapshot?.hypothesis_packet || {})}" in dashboard
    assert "검증 소스·경로" in dashboard
    assert "target_discovery_statuses" in dashboard
    assert "targetMetadata.identity?.sources" in dashboard
    assert "context_source_outcomes" in dashboard
    assert "LIVE ${evidenceCounts.live}" in dashboard
    assert "FIXTURE ${evidenceCounts.fixture}" in dashboard
    assert "${renderTargetIdentity(hypothesis.target || {})}" in dashboard
    assert "${renderEvidenceSources(evidence)}" in dashboard
    assert 'class="provenance-details"' in dashboard
    assert "조회·감사 상세" in dashboard
    assert "저장된 HypothesisPacket·source outcome·GateVerdict 기준" in dashboard


def test_hypothesis_portfolio_visually_separates_literature_and_tbio_tracks() -> None:
    dashboard = _dashboard()

    assert 'key: "literature"' in dashboard
    assert 'title: "Literature-supported"' in dashboard
    assert 'key: "tbio"' in dashboard
    assert 'title: "Tbio · underexplored"' in dashboard
    assert 'class="hypothesis-track-groups"' in dashboard
    assert 'class="hypothesis-track-section track-${esc(section.key)}"' in dashboard
    assert "0 · 검색 완료" in dashboard


def test_hypothesis_table_prioritizes_the_narrative_column_width() -> None:
    dashboard = _dashboard()
    styles = (ROOT / "src/drugpilot/web/static/app.css").read_text(encoding="utf-8")

    assert '<col class="hypothesis-col-narrative">' in dashboard
    assert ".hypothesis-table .hypothesis-col-narrative { width: 46%; }" in styles
    assert ".hypothesis-table .hypothesis-col-rank { width: 5%; }" in styles
    assert ".hypothesis-table .hypothesis-col-evidence { width: 8%; }" in styles
    assert ".hypothesis-table .narrative-cell { width: auto; }" in styles


def test_top_candidate_exposes_result_trust_problem_and_compact_chemistry() -> None:
    dashboard = _dashboard()
    start = dashboard.index("function renderDiscoverySummary")
    end = dashboard.index("function handleContextRequest", start)
    summary = dashboard[start:end]

    assert "candidateSummaryChemistryHtml(topCandidate)" in summary
    assert "<dt>Primary result</dt>" in summary
    assert "<dt>Trust</dt>" in summary
    assert "<dt>Open issue</dt>" in summary
    assert "중대한 미해결 감사 항목 없음" in summary
    assert "function candidateSummaryChemistryHtml(candidate)" in dashboard
    assert "candidateIdentityEntries(candidate)" in dashboard


def test_artifact_tab_change_resets_the_result_view_to_its_top() -> None:
    dashboard = _dashboard()

    assert 'const artifactContent = $("#artifactContent")' in dashboard
    assert "artifactContent.scrollTop = 0" in dashboard


def test_discovery_summary_candidate_link_delegates_to_full_candidate_table() -> None:
    dashboard = _dashboard()
    start = dashboard.index("function handleContextRequest")
    end = dashboard.index("function handleContextKeydown", start)
    handler = dashboard[start:end]

    assert 'event.target.closest("[data-summary-tab]")' in handler
    assert "selectArtifactTab(summaryTrigger.dataset.summaryTab)" in handler
    assert handler.index("[data-summary-tab]") < handler.index("[data-context-kind]")
