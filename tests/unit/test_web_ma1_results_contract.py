from __future__ import annotations

from pathlib import Path


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


def test_ma1_results_badge_and_gate_are_scoped_to_retained_packet() -> None:
    page = _page()
    dashboard = _dashboard()
    counts = _between(dashboard, "function renderCounts", "function selectArtifactTab")
    view = _between(
        dashboard, "function renderHypothesesArtifact", "function targetSourceTag"
    )

    assert ">MA1 Results <span id=\"hypothesisCount\">Retained 0</span>" in page
    assert '`Retained ${hypotheses.length}`' in counts
    assert view.count("Packet-level G0 audit") == 1
    assert "Retained hypotheses" in view
    assert "Retained by MA1" in view
    assert "<th>MA1 status</th>" in view
    assert "audit-indicator" not in view
    assert "G0는 개별 행이 아니라 이 packet 전체를 감사합니다" in view


def test_every_evidence_row_discloses_source_identity_origin_role_and_provenance() -> None:
    assert "app.js?v=20260929-ma3-admission-v1" in _page()
    assert "observed_molecular_scopes" in _dashboard()
    assert "원문에 등장한 융합·변이 범위" in _dashboard()
    dashboard = _dashboard()
    evidence_view = _between(
        dashboard, "function evidenceGroupHtml", "function auditContextHtml"
    )

    for stored_field in (
        "item?.source",
        "item?.external_id",
        "metadata.pmid",
        "metadata.pmcid",
        "metadata.doi",
        "metadata.content_availability",
        "metadata.citation_relation",
        "metadata.citation_hop",
        "item?.supports",
        "metadata?.claim_linkage_roles",
        "item.metadata?.provenance_class",
        "item.metadata?.source_location",
    ):
        assert stored_field in evidence_view
    for label in (
        "OA 본문 · reference 1-hop",
        "OA 본문 · direct search",
        "초록 발췌",
        "reference 1-hop · 메타데이터",
        "메타데이터",
        "직접 기전 근거",
        "질환–표적 연관 근거",
        "지지 근거",
        "반증 근거",
        "맥락 근거",
        "외부 실시간 조회",
        "외부 캐시",
        "external_live",
        "external_cache",
        "FIXTURE",
    ):
        assert label in evidence_view
    assert "`${source} · ${contentOrigin ||" in evidence_view
    assert "esc(sourceIdentity)" in evidence_view
    assert "esc(scientificRole)" in evidence_view
    assert "esc(provenanceLabel(provenance))" in evidence_view
    assert "esc(sourceLocation)" in evidence_view
    assert ".slice(0, 240)" in evidence_view
    assert ".slice(0, 120)" in evidence_view
    assert ".slice(0, 80)" in evidence_view


def test_hypothesis_why_evidence_links_disclose_origin_identity_role_and_provenance() -> None:
    dashboard = _dashboard()
    why_view = _between(
        dashboard, "function hypothesisWhyHtml", "function renderEvidenceSources"
    )

    assert "evidenceRoleLabel(item.supports)" in why_view
    assert "evidenceContentOriginLabel(item)" in why_view
    assert "evidenceSourceIdentityLabel(item, contentOrigin)" in why_view
    assert "item?.metadata?.provenance_class" in why_view
    assert "item?.metadata?.cache_hit" in why_view
    assert "item?.metadata?.live" in why_view
    assert "provenanceLabel(provenance)" in why_view
    assert "esc(label)" in why_view
    assert "esc(item.url)" in why_view


def test_evidence_excerpt_label_is_content_specific_not_unconditionally_abstract() -> None:
    dashboard = _dashboard()
    presentation = _between(
        dashboard,
        "function evidenceExcerptPresentation",
        "function evidenceExcerptText",
    )

    oa_branch = presentation.index(
        'metadata.content_availability === "oa_fulltext_excerpt"'
    )
    abstract_branch = presentation.index(
        'metadata.content_availability === "abstract_excerpt"'
    )
    generic_branch = presentation.index(
        'if (metadata.excerpt) return { kind: "저장 원문 발췌"'
    )
    assert oa_branch < abstract_branch < generic_branch
    assert "Europe PMC OA 본문 발췌" in presentation
    assert "초록 원문 발췌" in presentation
    assert "저장 원문 발췌" in presentation


def test_screening_projection_unions_modern_trace_and_exclusion_summaries() -> None:
    dashboard = _dashboard()
    projection = _between(
        dashboard, "function ma1ScreeningProjection", "function screeningTraceDetailHtml"
    )
    screen_view = _between(
        dashboard, "function renderScreenedProposals", "function ma1RevisionReworkHtml"
    )

    for source in (
        "metadata.screened_target_sources",
        "metadata.ma1_screening_history_sources",
        "metadata.rejected_screening_trace",
        "metadata.ma1_rejected_screening_history",
        "packet?.excluded",
    ):
        assert source in projection
    assert "screened.forEach" in projection
    assert "traceEntries.forEach" in projection
    assert "exclusions.forEach" in projection
    assert 'records.push(makeRecord(exclusion, null, "excluded_summary"))' in projection
    assert "if (!screened.length" not in projection
    assert "fallbackSummaryCount" in projection
    assert 'source.disposition || "").toLowerCase() === "retained"' in projection

    assert "Screened proposals — retained가 아닌 항목" in screen_view
    assert "최종 retained hypothesis가 아닙니다" in screen_view
    assert "Screened projection" in screen_view
    assert "Rejected detail trace" in screen_view
    assert "Exclusion summaries" in screen_view
    assert "Bounded projection" in screen_view
    assert "과거 packet 전문을 재구성하지 않습니다" in screen_view
    assert "Prior revision rejected-screening trace" in screen_view
    assert "Prior packet screening preserved" in screen_view
    assert "screeningTraceDetailHtml(record, projection, packet, index)" in screen_view


def test_zero_retained_after_verified_screening_still_renders_ma1_trace() -> None:
    dashboard = _dashboard()
    artifact = _between(
        dashboard, "function renderHypothesesArtifact", "function targetSourceTag"
    )

    assert "!hypotheses.length && intake.abstained" not in artifact
    assert "intake.verificationUnavailable || intake.hasMismatch" in artifact
    assert "renderDiscoveryCandidateTrace(packet)" in artifact
    assert "renderScreenedProposals(packet, screening)" in artifact
    assert "Retained hypotheses 0" in artifact


def test_discovery_summary_reports_screened_vs_discovered_when_g0_stops() -> None:
    page = _page()
    dashboard = _dashboard()
    summary = _between(
        dashboard, "function renderDiscoverySummary", "function handleContextRequest"
    )
    overview = _between(
        dashboard, "function renderOverviewArtifact", "function nonnegativeInteger"
    )

    assert 'id="discoveryTargetCountLabel">Target coverage' in page
    assert "ma1_discovery_candidate_trace_total" in summary
    assert "goal_focus_screened_symbols" in summary
    assert "const discoveredTargets = Math.max(" in summary
    assert "screenedTargets," in summary
    assert '"MA1 targets screened / discovered"' in summary
    assert "retained hypothesis는 0개입니다" in summary
    assert "미심사" in summary
    assert "질환 기반 후보 풀" in overview
    assert "발견 ${compactNumber(discoveredTargetCount)}건" in overview
    assert "질환 후보 심사 ${compactNumber(diseaseScreenedTargetCount)}건" in overview
    assert "현재 질환발굴 seed" in overview
    assert "const explicitScope = ma1ExplicitScopeProjection" in overview
    assert "명시 요청 표적" in overview
    assert "exact ID 해결" in overview
    assert "MA1 Results에서 행을 추정하지 않음" in overview
    assert "const screeningSources =" in overview
    assert 'String(item.target_source || "") !== "explicit_request_scope_identity"' in overview
    assert "const discoveredTargetCount = nonnegativeInteger(" in overview
    assert "const discoveredTargetCount = Math.max(" not in overview


def test_discovery_trace_labels_retry_budget_states_explicitly() -> None:
    dashboard = _dashboard()
    view = _between(
        dashboard,
        "function renderDiscoveryCandidateTrace",
        "function ma1RevisionReworkHtml",
    )

    assert 'screeningAllocation === "scheduled_for_ma1_evidence_retry"' in view
    assert "Allocated to MA1 evidence retry" in view
    assert 'screeningAllocation === "not_scheduled_evidence_retry_budget"' in view
    assert "Not scheduled · evidence-retry budget" in view


def test_explicit_target_scope_ledger_is_fail_closed_and_visible_before_screening() -> None:
    dashboard = _dashboard()
    projection = _between(
        dashboard,
        "function ma1ExplicitScopeProjection",
        "function renderExplicitScopeIdentityTrace",
    )
    view = _between(
        dashboard,
        "function renderExplicitScopeIdentityTrace",
        "function ma1RevisionReworkHtml",
    )
    artifact = _between(
        dashboard, "function renderHypothesesArtifact", "function targetSourceTag"
    )

    for field in (
        "ma1_explicit_scope_conservation_ledger",
        "ma1_request_scope_identity_trace",
        "ma1_request_scope_identity_query_count",
        "ma1_request_scope_identity_seed_count",
        "ma1_request_scope_identity_trace_sha256",
        "request_identity_is_query_scope_not_scientific_support",
        "scientific_support_created",
        "identity_source_outcomes_sha256",
        "unexpected_generated_symbols",
        "conservation_valid",
    ):
        assert field in projection
    for collection in (
        '"requested"',
        '"dynamic_exact_match"',
        '"identity_queried"',
        '"identity_resolved"',
        '"scheduled"',
        '"screened"',
        '"retained"',
        '"excluded"',
    ):
        assert collection in projection
    assert 'ledger.schema_version !== "ma1_explicit_scope_conservation_ledger/v1"' in projection
    assert "value.count !== symbols.length" in projection
    assert "new Set(symbolKeys).size !== symbols.length" in projection
    assert "item.scientific_support_created !== false" in projection
    assert "item.fixture !== false" in projection
    assert "Object.hasOwn(MA1_EXPLICIT_SCOPE_IDENTITY_REASONS" in projection
    assert "return {present: true, valid: false" in projection
    assert 'ledger.target_scope_mode === "explicit_only"' in projection
    assert "rawTrace.length > 0" in projection
    assert 'const allowedSources = new Set(["Open Targets", "UniProt", "fixture"])' in projection
    assert 'outcome.provenanceClass === "fixture_fallback"' in projection
    assert 'reasonCode === "EXPLICIT_TARGET_FIXTURE_IDENTITY_REJECTED"' in projection
    assert "symbolKey(outcome.canonicalSymbol) === symbolKey(requestedSymbol)" in projection
    assert 'String(outcome.ensemblId || "").toLowerCase() === ensembl.toLowerCase()' in projection
    assert 'String(outcome.uniprotAccession || "").toLowerCase() === uniprot.toLowerCase()' in projection
    assert "!sameSet(normalizedCollections.retained.keys, packetRetainedKeys)" in projection
    assert "!sameSet(normalizedCollections.excluded.keys, packetExcludedKeys)" in projection
    assert "identityRejectionKeys" in projection
    assert "const orderedTrace = requested.symbols.map" in projection
    assert "packetExcludedReasonCodes" in projection
    assert ".filter((key) => key && requested.keys.has(key))" in projection
    assert "if (key && !requested.keys.has(key)) continue" in projection
    assert "const existingReasons = packetExcludedReasonCodes.get(key) || []" in projection
    assert "const combinedReasons = [...new Set([...existingReasons, ...reasonCodes])]" in projection

    assert "사용자 지정 표적 보존 장부" in view
    assert "요청한 모든 표적" in view
    assert "exact identity 확인" in view
    assert "질환 연관성, 작용 방향 또는 유효성 근거를 만들지 않으며" in view
    assert "Source library" in view
    assert "Identity outcome / reason" in view
    assert "Final MA1 status / reason" in view
    assert "Retained by MA1" in view
    assert "Identity 단계 종료" in view
    assert "packet exclusion 사유 코드 미기록" in view
    assert "retained/excluded 최종 기록 없음 · 추정하지 않음" in view
    assert "저장 해시의 원문 재계산은 하지 않으며" in view
    assert "현재 결과에서 행이나 탈락 이유를 추정하지 않습니다" in view
    assert "상세 행을 숨겼습니다" in view
    assert "esc(item.requestedSymbol)" in view
    assert "esc(sourceText)" in view
    assert "esc(item.reasonCode)" in view
    assert "outcome.endpoint" not in view
    assert "outcome.error_type" not in view
    assert "${renderExplicitScopeIdentityTrace(packet)}" in artifact
    assert artifact.index("${renderExplicitScopeIdentityTrace(packet)}") < artifact.index(
        "${renderDiscoveryCandidateTrace(packet)}"
    )


def test_explicit_identity_reason_codes_are_bounded_and_human_readable() -> None:
    dashboard = _dashboard()
    reason_map = _between(
        dashboard,
        "const MA1_EXPLICIT_SCOPE_IDENTITY_REASONS",
        "function ma1ExplicitScopeProjection",
    )

    for code in (
        "EXPLICIT_TARGET_FORBIDDEN",
        "EXPLICIT_TARGET_DISEASE_IDENTITY_NOT_RESOLVED",
        "EXPLICIT_TARGET_IDENTITY_RESOLVER_UNAVAILABLE",
        "EXPLICIT_TARGET_FIXTURE_IDENTITY_REJECTED",
        "EXPLICIT_TARGET_IDENTITY_NOT_RESOLVED",
        "EXPLICIT_TARGET_CANONICAL_SYMBOL_MISMATCH",
        "EXPLICIT_TARGET_STABLE_IDENTIFIERS_INCOMPLETE",
        "EXPLICIT_TARGET_STRUCTURE_IDENTITY_NOT_READY",
        "EXPLICIT_TARGET_IDENTITY_SOURCE_CONFLICT",
        "EXPLICIT_TARGET_IDENTITY_PROVENANCE_MISSING",
        "EXPLICIT_TARGET_EXTERNAL_IDENTITY_ATTESTATION_MISSING",
        "EXPLICIT_TARGET_SYMBOL_ATTESTATION_MISSING",
        "EXPLICIT_TARGET_ENSEMBL_ATTESTATION_MISSING",
        "EXPLICIT_TARGET_UNIPROT_ATTESTATION_MISSING",
    ):
        assert code in reason_map
    assert "기록된 허용 사유 코드가 아님" in reason_map


def test_screened_rows_keep_dynamic_values_escaped_and_absence_honest() -> None:
    dashboard = _dashboard()
    detail = _between(
        dashboard, "function screeningTraceDetailHtml", "function renderScreenedProposals"
    )
    screen_view = _between(
        dashboard, "function renderScreenedProposals", "function ma1RevisionReworkHtml"
    )

    for escaped_expression in (
        "esc(record.targetSymbol)",
        "esc(record.hypothesisId)",
        "esc(record.identityStatus",
        "esc(stableIds.join",
        "esc(screeningPhaseLabel(record.screeningPhase))",
        "esc(reason)",
        "esc(sourceLabel)",
    ):
        assert escaped_expression in screen_view
    for escaped_expression in (
        "esc(item.source",
        "esc(item.external_id",
        "esc(item.pmid)",
        "esc(item.doi)",
        "esc(projection.contentPolicy)",
        "esc(disclosureKey)",
    ):
        assert escaped_expression in detail
    assert "Router exclusion 요약만 있으며" in detail
    assert "상세 rejected-screening trace가 기록되지 않았습니다" in detail
    assert "원문 초록 전문과 비공개 데이터는 표시하지 않습니다" in detail


def test_zero_retained_packet_still_renders_screening_and_exclusions() -> None:
    dashboard = _dashboard()
    view = _between(
        dashboard, "function renderHypothesesArtifact", "function targetSourceTag"
    )

    assert 'if (!packet) return \'<div class="artifact-empty">' in view
    assert "const screening = ma1ScreeningProjection(packet)" in view
    assert "const retainedHtml = hypotheses.length" in view
    assert "Retained hypotheses 0" in view
    assert "${renderScreenedProposals(packet, screening)}" in view
    assert "if (!hypotheses.length) return" not in view


def test_ma1_revision_summary_uses_snapshot_events_without_claiming_full_history() -> None:
    dashboard = _dashboard()
    revision = _between(
        dashboard, "function ma1RevisionReworkHtml", "function renderHypothesesArtifact"
    )

    assert "packet?.metadata?.revision_lineage" in revision
    assert 'String(event?.event_type || "") === "rework_order"' in revision
    assert 'String(event?.metadata?.interaction?.kind || "") === "selective_rework"' in revision
    assert 'String(event?.event_type || "") === "ma1_completed"' in revision
    assert "snapshot?.ma1_iterations" in revision
    assert "snapshot?.rework_loops" in revision
    assert "revision?.parent_packet_id" in revision
    assert "revision?.scientific_changed" in revision
    assert "revision?.audit_trace_changed" in revision
    assert "Packet revision / rework summary" in revision
    assert "최신 direct lineage" in revision
    assert "과거 packet 내용 전체를 복원하거나 변경 diff를 추정하지 않습니다" in revision


def test_landing_has_no_fixture_demo_selection_or_prefill_path() -> None:
    page = _page()
    dashboard = _dashboard()
    bindings = _between(dashboard, "function bindControls", "function handleTabKeydown")

    assert 'id="viewDemoRun"' not in page
    assert "VIEW DEMO RUN" not in page
    assert "demoRunBoundary" not in page
    assert "MOCK/FIXTURE workflow demo" not in page
    assert "TNBC 예시" not in page
    assert "VIEW WORKFLOW" in page

    assert "viewDemoRun" not in bindings
    assert "openStoredDemoRunOrPrefill" not in dashboard
    assert "latestStoredFixtureDemoRun" not in dashboard
    assert "prefillTnbcDemoBrief" not in dashboard
    assert '$("#disease").value = "Triple-negative breast cancer"' not in dashboard
    assert '$("#cellType").value = "TNBC cell lines"' not in dashboard


def test_ma1_results_styles_reuse_research_table_and_remain_responsive() -> None:
    styles = _styles()

    for selector in (
        ".ma1-result-section",
        ".ma1-result-heading",
        ".screening-projection-summary",
        ".screening-table",
        ".screening-status.is-retained",
        ".screening-status.is-rejected",
        ".screening-trace-details",
        ".ma1-rework-summary",
        ".hypothesis-table .hypothesis-col-status",
    ):
        assert selector in styles
    assert ".hypothesis-table .hypothesis-col-audit" not in styles
    # Other components may append their own 700px rules; inspect the MA1 block.
    ma1_rule = styles.index(".screening-projection-summary { grid-template-columns: 1fr; }")
    responsive_start = styles.rfind("@media (max-width: 700px)", 0, ma1_rule)
    assert responsive_start >= 0
    responsive = styles[responsive_start : styles.index("\n}", ma1_rule)]
    assert ".screening-projection-summary { grid-template-columns: 1fr; }" in responsive
    assert ".screening-detail-columns," in responsive
