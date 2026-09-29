from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _page() -> str:
    return (ROOT / "src/drugpilot/web/templates/index.html").read_text(
        encoding="utf-8"
    )


def _styles() -> str:
    return (ROOT / "src/drugpilot/web/static/app.css").read_text(encoding="utf-8")


def _dashboard() -> str:
    return (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")


def test_home_uses_stable_english_chrome_while_research_content_can_be_korean() -> None:
    page = _page()

    assert (
        "질환 근거부터<br>후보 평가까지<br>"
        '<span class="welcome-product">DrugPilot</span>'
    ) in page
    assert "EVIDENCE-GOVERNED DISCOVERY" in page
    assert "VIEW DEMO RUN" not in page
    assert "VIEW WORKFLOW" in page
    assert "RESEARCH BRIEF" in page
    assert "RUN HISTORY" in page
    assert "Evidence-linked" in page
    assert "Audit-driven" in page
    assert "Human-controlled" in page
    assert 'class="welcome-visual"' in page
    assert "Evidence<br>Governance" in page
    assert "Human Gate" in page
    assert 'class="orbit-card card-ma1"' in page


def test_result_typography_is_not_smaller_than_the_navigation_copy() -> None:
    styles = _styles()

    assert "Research workbench visual system" in styles
    assert ".activity-title { font-size: 12px; }" in styles
    assert ".activity-message { margin-top: 3px; font-size: 11px;" in styles
    assert ".hypothesis-card h4," in styles
    assert ".candidate-card h4," in styles
    assert ".plan-card h4 { font-size: 14px; }" in styles
    assert ".card-description { min-height: 0; font-size: 11px;" in styles
    assert ".detail-row dd { font-size: 11px; }" in styles


def test_visual_system_uses_flat_research_workbench_surfaces() -> None:
    styles = _styles()

    assert ".orbit-one" in styles
    assert ".orbit-two" in styles
    assert ".lab-core" in styles
    assert ".orbit-card" in styles
    assert "body { background: var(--canvas); }" in styles
    assert ".metric-card-action:hover," in styles
    assert "box-shadow: none;" in styles
    assert "transform: none;" in styles
    assert "Research-record layout" in styles
    assert ".run-workspace > * + * { margin-top: 0; }" in styles
    assert ".card-grid {" in styles
    assert "grid-template-columns: 1fr;" in styles
    assert ".metrics-grid .metric-card {" in styles


def test_result_first_information_architecture_is_present() -> None:
    page = _page()

    discovery = page.index('id="discoverySummary"')
    audit = page.index('class="verdict-panel audit-workbench"')
    execution = page.index('id="executionTracePanel"')
    artifacts = page.index('class="artifact-panel research-artifacts"')
    metadata = page.index('class="metrics-grid run-metadata-line"')

    assert discovery < audit < execution < artifacts
    assert execution < metadata < artifacts
    assert 'id="contextPanel"' in page
    assert 'class="research-table' not in page  # Tables are data-driven, not fake markup.


def test_context_panel_and_tabs_preserve_researcher_interaction_state() -> None:
    page = _page()
    dashboard = _dashboard()

    assert "contextTrigger: null" in dashboard
    assert "trigger?.isConnected" in dashboard
    assert 'event.key !== "Escape"' in dashboard
    assert "handleTabKeydown" in dashboard
    assert 'tab.setAttribute("aria-selected"' in dashboard
    assert "tab.tabIndex = selected ? 0 : -1" in dashboard
    assert "validationStatus" in dashboard
    assert '${gate}:${validationKey(item, index)}' in dashboard
    assert 'aria-controls="verdictContent"' in page
    assert 'aria-controls="artifactContent"' in page


def test_evidence_panel_renders_bounded_publication_excerpts_and_honest_absence_reasons() -> None:
    dashboard = _dashboard()

    assert "function evidenceExcerptPresentation" in dashboard
    assert "if (metadata.excerpt)" in dashboard
    assert "if (metadata.detail)" in dashboard
    assert 'kind: "초록 원문 발췌"' in dashboard
    assert 'kind: "원천 구문"' in dashboard
    assert 'kind: "저장 근거 요약"' in dashboard
    assert 'kind: "저장 근거 구문"' in dashboard
    assert 'kind: "구조화 근거 요약"' in dashboard
    assert 'kind: "문헌 메타데이터"' in dashboard
    assert 'kind: "임상시험 등록 요약"' in dashboard
    assert 'kind: "프리프린트 서지 요약"' in dashboard
    assert 'metadata.excerpt_selection_status === "related_excerpt_not_found"' in dashboard
    assert "표적과 질환을 같은 문장에서 직접 연결한 구문을 찾지 못해" in dashboard
    assert "초록이 제공되지 않아 서지 메타데이터만 기록" in dashboard
    assert "metadata.abstract_retrieval_error_type" in dashboard
    assert "sourceLocation" in dashboard
    assert "근거 구문 미기록" not in dashboard


def test_per_run_research_depth_is_explicit_truthful_and_legacy_safe() -> None:
    page = _page()
    dashboard = _dashboard()
    styles = _styles()

    assert "<legend>RESEARCH DEPTH</legend>" in page
    assert 'name="research_depth" type="radio" value="focused"' in page
    assert 'name="research_depth" type="radio" value="standard" checked' in page
    assert 'name="research_depth" type="radio" value="comprehensive"' in page
    assert "근거·도구 범위, 검토 강도와 소요시간을 조절합니다" in page
    assert "비공개 사고과정이나 결과 정확도를 보장하는 설정은 아닙니다" not in page
    assert 'research_depth: selectedResearchDepth()' in dashboard
    assert "function researchDepthFromSnapshot" in dashboard
    depth_resolver = dashboard[
        dashboard.index("function researchDepthFromSnapshot") : dashboard.index(
            "function researchDepthPresentation"
        )
    ]
    assert "snapshot?.brief?.research_depth" in depth_resolver
    assert "snapshot?.research_depth" not in depth_resolver
    assert "snapshot?.metadata?.research_depth" not in depth_resolver
    assert "plan_spec" not in depth_resolver
    assert "hypothesis_packet" not in depth_resolver
    assert 'label: "미기록 · 이전 실행"' in dashboard
    assert "RESEARCH DEPTH</dt>" in dashboard
    assert "#researchDepthDescription" in dashboard
    assert (
        "가용한 근거와 적용 가능한 도구를 폭넓게 조회 · "
        "다각도 교차 검토 · 소요시간 증가"
    ) in dashboard
    assert "허용된 최대 범위" not in dashboard
    assert "가장 오래 걸림" not in dashboard
    assert ".research-depth-options input:checked + span" in styles
    assert 'app.css?v=20260929-ma3-admission-v1' in page
    assert 'app.js?v=20260929-ma3-admission-v1' in page


def test_saved_run_deep_link_is_loaded_and_kept_shareable() -> None:
    dashboard = _dashboard()

    assert "async function init()" in dashboard
    assert "await loadRuns();" in dashboard
    assert "requestedRunIdFromLocation()" in dashboard
    assert 'parameters.get("run_id")' in dashboard
    assert 'parameters.get("ui")' in dashboard
    assert "/run_[a-f0-9]{12}$/i" in dashboard
    assert "if (requestedRunId) await selectRun(requestedRunId);" in dashboard
    assert 'url.searchParams.set("ui", `g1-live-test-${runId}`)' in dashboard
    assert 'window.history.replaceState({ runId }, "", url)' in dashboard


def test_home_separates_data_sources_from_authoritative_scientific_tool_status() -> None:
    page = _page()
    dashboard = _dashboard()
    styles = _styles()

    tool_panel = page[
        page.index('id="scientificToolsPanel"') : page.index(
            'class="research-depth-field"'
        )
    ]
    tool_definitions = dashboard[
        dashboard.index("const SCIENTIFIC_TOOL_DEFINITIONS") : dashboard.index(
            "const state"
        )
    ]

    assert '<b id="sourceCatalogButtonTitle">DATA SOURCES</b>' in page
    assert '<b id="scientificToolsTitle">SCIENTIFIC TOOLS</b>' in tool_panel
    assert '<span class="source-catalog-icon" aria-hidden="true">' in page
    assert '<span class="scientific-tools-icon" aria-hidden="true">' in tool_panel
    assert '<ellipse cx="12" cy="5.5" rx="7.5" ry="3"></ellipse>' in page
    assert 'M10 3v6l-5.2 8.5' in tool_panel
    assert ">DB</span>" not in page
    assert ">LAB</span>" not in tool_panel
    assert ">FX</span>" not in tool_panel
    assert "READY" not in tool_panel  # Static markup must not claim runtime readiness.
    assert "RDKit" in tool_definitions
    assert "ADMET-AI" in tool_definitions
    assert "AutoDock Vina" in tool_definitions
    assert "ChEMBL" not in tool_definitions  # ChEMBL remains a data source.
    assert 'capability: "de_novo_generator"' in tool_definitions
    assert 'capability: "admet_predictor"' in tool_definitions
    assert 'capability: "docking"' in tool_definitions
    assert 'label: "RUN TOOLS"' in tool_definitions
    assert 'label: "PROFILE BUILDER"' in tool_definitions
    assert "state.health?.scientific_tools?.groups" in dashboard
    assert "state.health?.scientific_tools?.items" in dashboard
    assert "state.health?.ma2?.capabilities" in dashboard
    assert "function scientificToolStatus" in dashboard
    assert "function scientificToolVersion" in dashboard
    assert 'label: "AVAILABLE"' in dashboard
    assert 'label: "NOT CONFIGURED"' in dashboard
    assert 'label: "UNAVAILABLE"' in dashboard
    assert 'label: "PROFILE REQUIRED"' in dashboard
    assert 'label: "OFFLINE"' in dashboard
    assert 'label: "TARGET CHECK AT RUN"' in dashboard
    assert "가용성·설정 상태입니다. 이번 실행의 실제 사용 여부는 TRACE에서 확인합니다." in dashboard
    assert "scientific-tools\",\n        items: groupedItems" in dashboard
    assert "execution_enabled: group.execution_enabled" in dashboard
    assert 'item?.execution_enabled === false ? "EXECUTION DISABLED"' in dashboard
    assert 'source: offline ? "offline" : "legacy"' in dashboard
    assert "MA2 ${readiness}" not in dashboard
    assert 'label: "READY"' not in dashboard
    assert ".scientific-tools-panel > p" not in styles
    assert "max-height: 150px" not in styles
    assert "max-height: min(46vh, 390px)" in styles
    assert ".scientific-tools-panel" in styles
    assert ".source-catalog-button {\n  height: 45px;\n  min-height: 45px;" in styles
    assert ".scientific-tools-toggle {\n  height: 45px;\n  min-height: 45px;" in styles
    assert "grid-template-columns: 30px minmax(0, 1fr) 14px" in styles
    assert ".scientific-tools-icon {\n  width: 30px;\n  height: 30px;" in styles
    assert ".source-catalog-icon svg,\n.scientific-tools-icon svg" in styles
    assert "width: 16px;\n  height: 16px;" in styles
    assert ".scientific-tools-toggle small {\n  overflow: hidden;" in styles
    assert "text-overflow: ellipsis;\n  white-space: nowrap;" in styles
    assert ".scientific-tool-status.is-available" in styles
    assert ".scientific-tool-status.is-profile-required" in styles
    assert ".scientific-tool-status.is-target-check" in styles
    assert '<details id="scientificToolsPanel" class="scientific-tools-panel">' in page
    assert 'class="scientific-tools-chevron"' in page


def test_checkpoint_resume_copy_is_driven_by_verified_resume_status() -> None:
    page = _page()
    dashboard = _dashboard()

    assert '>체크포인트에서 재개</button>' in page
    assert "/resume-status`" in dashboard
    assert "status?.eligible" in dashboard
    assert "status?.active_in_process" in dashboard
    assert 'showToast(\n        "체크포인트에서 실행을 재개했습니다"' in dashboard
    assert 'showToast("전체 워크플로를 다시 시작했습니다"' not in dashboard
    assert "function renderResumeControl" in dashboard
    assert "state.resumeStatusRunId === snapshot?.run_id" in dashboard


def test_candidate_cards_render_local_2d_structures_with_a_safe_fallback() -> None:
    dashboard = _dashboard()

    assert "/api/chemistry/structure.svg?smiles=${encodeURIComponent(smiles)}" in dashboard
    assert 'class="molecule-structure-image"' in dashboard
    assert 'alt="${esc(structureAlt)}"' in dashboard
    assert 'image.addEventListener("error", markUnavailable, { once: true })' in dashboard
    assert "image.complete && image.naturalWidth === 0" in dashboard
    assert "2D 구조를 표시할 수 없습니다" in dashboard


def test_candidate_chemistry_keeps_scientifically_useful_identity_fields_visible() -> None:
    dashboard = _dashboard()
    styles = _styles()

    assert '<div class="smiles"><b>SMILES</b><code>' in dashboard
    assert '["InChIKey", cleanCandidateIdentifier(candidate.inchikey, inchikeyPattern)]' in dashboard
    assert '["ChEMBL", chemblId]' in dashboard
    assert '["PubChem CID", pubchemId]' not in dashboard
    assert '["Candidate ID", cleanCandidateIdentifier(candidate.candidate_id, recordPattern)]' not in dashboard
    assert ".candidate-chemistry { display: grid;" in styles
    assert "grid-template-columns: minmax(132px,172px) minmax(0,1fr);" in styles
    assert ".molecule-structure.is-unavailable .molecule-structure-fallback { display: grid; }" in styles


def test_active_candidate_table_and_context_panel_render_the_structure_ui() -> None:
    dashboard = _dashboard()
    styles = _styles()

    table_renderer = dashboard[
        dashboard.index("function renderCandidatesArtifact") : dashboard.index(
            "function ma2ProcessOverview"
        )
    ]
    context_renderer = dashboard[
        dashboard.index("function candidateContextHtml") : dashboard.index(
            "function evidenceGroupHtml"
        )
    ]
    context_opener = dashboard[
        dashboard.index("function openContextPanel") : dashboard.index(
            "function closeContextPanel"
        )
    ]

    assert "${candidateTableIdentityHtml(candidate)}" in table_renderer
    assert "${candidateChemistryHtml(candidate)}" in context_renderer
    assert "content.innerHTML = html;\n    wireMoleculeStructures(content);" in context_opener
    assert 'class="candidate-table-smiles"' in dashboard
    assert ".molecule-structure-compact {" in styles
    assert ".candidate-table th:nth-child(2) { width: 230px; }" in styles
