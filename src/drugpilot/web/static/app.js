(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const STAGES = [
    "brief_validated",
    "ma1",
    "g0",
    "planner",
    "ma2",
    "g1",
    "human_gate",
  ];

  const TERMINAL_STAGES = new Set(["completed", "blocked", "failed", "abstained", "cancelled"]);
  const LIVE_END_STAGES = new Set([...TERMINAL_STAGES, "human_gate"]);
  const RUNNING_STAGES = new Set(["created", ...STAGES]);
  const KRW_PER_USD = 1500;
  const RESEARCH_DEPTH_META = Object.freeze({
    focused: Object.freeze({
      label: "Focused",
      pill: "FOCUSED DEPTH",
      description: "핵심 근거와 필수 도구 중심 · 간소화된 교차 검토 · 가장 빠른 실행",
    }),
    standard: Object.freeze({
      label: "Standard",
      pill: "STANDARD DEPTH",
      description: "관련 근거와 적용 가능한 도구를 넓게 조회 · 균형 잡힌 교차 검토 · 권장",
    }),
    comprehensive: Object.freeze({
      label: "Comprehensive",
      pill: "COMPREHENSIVE DEPTH",
      description: "가용한 근거와 적용 가능한 도구를 폭넓게 조회 · 다각도 교차 검토 · 소요시간 증가",
    }),
  });
  const SCIENTIFIC_TOOL_DEFINITIONS = Object.freeze([
    Object.freeze({
      key: "rdkit",
      name: "RDKit BRICS",
      group: "run-tools",
      capability: "de_novo_generator",
      description: "BRICS 후보 생성 상태를 표시합니다. 분자 파싱·검증과 2D SVG는 각 요청에서 별도로 fail-closed 검증합니다.",
    }),
    Object.freeze({
      key: "admet",
      name: "ADMET-AI",
      group: "run-tools",
      capability: "admet_predictor",
      description: "버전이 기록된 모델로 6개 안전성·개발가능성 endpoint를 예측합니다.",
    }),
    Object.freeze({
      key: "vina",
      name: "AutoDock Vina",
      group: "run-tools",
      capability: "docking",
      description: "검증된 receptor PDBQT와 docking box가 있는 표적에서만 선택적으로 실행합니다.",
    }),
  ]);
  const SCIENTIFIC_TOOL_GROUPS = Object.freeze([
    Object.freeze({ key: "run-tools", label: "RUN TOOLS" }),
    Object.freeze({ key: "profile-builder", label: "PROFILE BUILDER" }),
  ]);

  const state = {
    health: null,
    runs: [],
    snapshot: null,
    runMode: "mock",
    llmBackend: "mock",
    dataBackend: "fixture",
    selectedTab: "overview",
    selectedArtifactRunId: null,
    selectedGate: "g0",
    selectedGateRunId: null,
    selectedGateUserSet: false,
    eventSource: null,
    pollTimer: null,
    refreshTimer: null,
    eventIds: new Set(),
    openDisclosures: new Set(),
    callTrace: null,
    callTraceLoading: false,
    callTraceRevision: null,
    callTraceRequests: new Set(),
    resumeStatus: null,
    resumeStatusRunId: null,
    resumeStatusLoading: false,
    cancelStatus: null,
    cancelStatusRunId: null,
    cancelStatusLoading: false,
    cancellingRunId: null,
    createSubmission: null,
    contextSelection: null,
    contextTrigger: null,
    loadingRun: false,
    preview: false,
    reviewAction: null,
    reviewStatus: null,
    reviewStatusLoading: false,
    resultReadiness: null,
    resultReadinessKey: null,
    resultReadinessError: null,
    resultReadinessRequests: new Map(),
    researchSelectionContext: null,
    researchSelectionSubmitting: false,
    sourceCatalog: {
      runtime: [],
      evaluation: [],
      loaded: false,
      error: null,
    },
  };

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    bindControls();
    renderArtifact();
    checkHealth();
    loadSourceCatalog();
    await loadRuns();
    const requestedRunId = requestedRunIdFromLocation();
    if (requestedRunId) await selectRun(requestedRunId);
  }

  function requestedRunIdFromLocation() {
    const parameters = new URLSearchParams(window.location.search);
    const direct = String(parameters.get("run_id") || "").trim();
    if (/^run_[a-f0-9]{12}$/i.test(direct)) return direct;
    const ui = String(parameters.get("ui") || "").trim();
    const match = ui.match(/run_[a-f0-9]{12}$/i);
    return match ? match[0] : null;
  }

  function syncRunDeepLink(runId) {
    if (!/^run_[a-f0-9]{12}$/i.test(String(runId || ""))) return;
    const url = new URL(window.location.href);
    url.searchParams.delete("run_id");
    url.searchParams.set("ui", `g1-live-test-${runId}`);
    window.history.replaceState({ runId }, "", url);
  }

  function bindControls() {
    $("#researchForm").addEventListener("submit", createRun);
    $$('#researchForm input[name="research_depth"]').forEach((radio) => {
      radio.addEventListener("change", updateResearchDepthDescription);
    });
    updateResearchDepthDescription();
    $("#direction").addEventListener("change", updateDirectionSettings);
    $("#directionScope").addEventListener("change", updateDirectionSettings);
    updateDirectionSettings();
    $("#refreshRuns").addEventListener("click", loadRuns);
    $("#showArchitecture").addEventListener("click", showArchitecturePreview);
    $("#openSourceCatalog").addEventListener("click", openSourceCatalog);
    $("#closeSourceCatalog").addEventListener("click", closeSourceCatalog);
    $("#sourceCatalogDialog").addEventListener("click", (event) => {
      if (event.target === $("#sourceCatalogDialog")) closeSourceCatalog();
    });
    $("#openToolTrace").addEventListener("click", openToolTrace);
    $("#closeToolTrace").addEventListener("click", closeToolTrace);
    $("#toolTraceDialog").addEventListener("click", (event) => {
      if (event.target === $("#toolTraceDialog")) closeToolTrace();
    });
    $("#newRunMobile").addEventListener("click", openSidebar);
    $("#closeSidebar").addEventListener("click", closeSidebar);
    $("#sidebarScrim").addEventListener("click", closeSidebar);

    $$(".number-control button").forEach((button) => {
      button.addEventListener("click", () => {
        const input = $("#candidateCount");
        const next = Number(input.value || 0) + Number(button.dataset.step || 0);
        input.value = String(Math.min(Number(input.max), Math.max(Number(input.min), next)));
      });
    });

    $$(".stage-node").forEach((node) => {
      node.addEventListener("click", () => selectArtifactTab(node.dataset.artifact || "overview"));
    });

    $$(".artifact-tabs [role=tab]").forEach((tab) => {
      tab.addEventListener("click", () => selectArtifactTab(tab.dataset.tab));
      tab.addEventListener("keydown", (event) => handleTabKeydown(event, $$(".artifact-tabs [role=tab]")));
    });

    $$('[data-summary-tab]').forEach((button) => {
      button.addEventListener("click", () => selectArtifactTab(button.dataset.summaryTab));
    });
    $("#closeContextPanel").addEventListener("click", closeContextPanel);
    [$("#artifactContent"), $("#discoverySummary"), $("#verdictContent"), $("#researchResultPanel")].forEach((root) => {
      root?.addEventListener("click", handleContextRequest);
      root?.addEventListener("keydown", handleContextKeydown);
    });

    $$(".gate-toggle [role=tab]").forEach((tab) => {
      tab.addEventListener("click", () => {
        state.selectedGate = tab.dataset.gate;
        state.selectedGateRunId = state.snapshot?.run_id || null;
        state.selectedGateUserSet = true;
        updateSelectedTabs($$(".gate-toggle [role=tab]"), "gate", state.selectedGate);
        renderVerdict();
      });
      tab.addEventListener("keydown", (event) => handleTabKeydown(event, $$(".gate-toggle [role=tab]")));
    });

    $("#openHumanGate").addEventListener("click", openDecisionDialog);
    $("#openResearchSelection").addEventListener("click", openResearchSelectionDialog);
    $("#researchSelectionForm").addEventListener("submit", submitResearchSelection);
    ["#closeResearchSelection", "#cancelResearchSelection"].forEach((id) => $(id).addEventListener("click", () => {
      if (!state.researchSelectionSubmitting) $("#researchSelectionDialog").close();
    }));
    $("#researchSelectionDialog").addEventListener("cancel", (event) => {
      if (state.researchSelectionSubmitting) event.preventDefault();
    });
    $("#nextActionContent").addEventListener("click", handleReviewAction);
    $("#reviewActionForm").addEventListener("submit", submitReviewAction);
    $("#closeReviewAction").addEventListener("click", () => $("#reviewActionDialog").close());
    $("#downloadObservedTemplate").addEventListener("click", downloadObservedTemplate);
    $$(".dialog-close").forEach((button) => button.addEventListener("click", closeDecisionDialog));
    $("#decisionForm").addEventListener("submit", submitHumanDecision);
    $$('input[name="decision"]').forEach((radio) => radio.addEventListener("change", updateDecisionForm));
    $("#resumeButton").addEventListener("click", resumeRun);
    $("#cancelRunButton").addEventListener("click", cancelRun);
    $("#downloadRun").addEventListener("click", downloadSnapshot);
    $("#copyRunId").addEventListener("click", copyRunId);
    $("#clearActivityFilter").addEventListener("click", () => {
      $("#clearActivityFilter").hidden = true;
      renderActivity();
    });

    window.addEventListener("beforeunload", stopLiveUpdates);
    window.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      if (!$("#contextPanel")?.hidden) {
        closeContextPanel();
        return;
      }
      closeSidebar();
    });
  }

  function handleTabKeydown(event, tabs) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key) || !tabs.length) return;
    event.preventDefault();
    const current = Math.max(0, tabs.indexOf(event.currentTarget));
    const nextIndex = event.key === "Home"
      ? 0
      : event.key === "End"
      ? tabs.length - 1
      : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[nextIndex].focus();
    tabs[nextIndex].click();
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    headers.set("Accept", "application/json");
    const response = await fetch(path, { ...options, headers });
    const text = await response.text();
    let payload = null;
    if (text) {
      try { payload = JSON.parse(text); }
      catch { payload = { detail: text }; }
    }
    if (!response.ok) {
      const detail = payload?.detail || payload?.message || `요청 실패 (${response.status})`;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg || JSON.stringify(item)).join(" · ")
        : String(detail);
      const error = new Error(message);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  async function checkHealth() {
    const badge = $("#healthBadge");
    badge.className = "health-badge is-checking";
    $("#healthText").textContent = "연결 확인 중";
    try {
      const health = await api("/api/health");
      state.health = health || {};
      updateDirectionSettings();
      badge.className = "health-badge is-online";
      $("#healthText").textContent = health?.status === "degraded" ? "제한 모드" : "시스템 정상";
      const mode = health?.effective_mode || health?.mode || health?.run_mode || state.runMode;
      updateBackendBadges(health, mode);
      updateScientificToolsPanel();
      if (state.sourceCatalog.error) renderSourceCatalog();
    } catch (error) {
      state.health = null;
      updateDirectionSettings();
      badge.className = "health-badge is-offline";
      $("#healthText").textContent = "서버 연결 안 됨";
      updateScientificToolsPanel();
      showToast("서버에 연결할 수 없습니다", "웹 서버가 실행 중인지 확인하십시오.", "error");
    }
  }

  function updateModeBadge(mode) {
    updateBackendBadges(state.health || {}, mode);
  }

  function updateBackendBadges(health = {}, mode = state.runMode) {
    if (!Object.keys(health).length) {
      $("#modeBadgeText").textContent = "LLM · 확인 중";
      $("#dataBadgeText").textContent = "DATA · 확인 중";
      $("#backendSummaryTitle").textContent = "LLM · DATA 확인 중";
      return;
    }
    const normalizedMode = String(mode || "mock").toLowerCase();
    if (["mock", "live", "hybrid"].includes(normalizedMode)) state.runMode = normalizedMode;

    const declaredLlm = firstBackendValue(
      health.llm_backend,
      health.backends?.llm,
      health.providers?.llm,
    );
    const openaiConfigured = Boolean(
      health.openai_configured ?? health.providers?.openai_configured ?? health.openai?.configured
    );
    const llmBackend = declaredLlm || (
      openaiConfigured && ["live", "hybrid"].includes(normalizedMode) ? "openai" : "mock"
    );
    state.llmBackend = llmBackend;
    const llmLabel = llmBackend === "openai" ? "OPENAI" : llmBackend.toUpperCase();
    const reviewerBackend = String(health.reviewer_backend || "deterministic-rules");
    const reviewerLabel = reviewerBackend === "openai-read-only"
      ? "OPENAI REVIEW"
      : reviewerBackend === "anthropic"
      ? "ANTHROPIC REVIEW"
      : "RULE AUDIT";
    const badge = $("#modeBadge");
    badge.className = `mode-badge is-${llmBackend === "openai" ? "openai" : "mock"}`;
    badge.title = llmBackend === "openai"
      ? "서버에서 OpenAI LLM 호출이 활성화되었습니다"
      : "유료 LLM 호출 없이 결정적 Mock 출력을 사용합니다";
    $("#modeBadgeText").textContent = `LLM · ${llmLabel}`;

    let dataBackend = firstBackendValue(
      health.data_backend,
      health.backends?.data,
      health.providers?.data,
    );
    const configuredDataBackends = Array.isArray(health.data_backends)
      ? health.data_backends.filter((item) => typeof item === "string" && item.trim())
      : [];
    const dataBackendText = configuredDataBackends.join(" ").toLowerCase();
    const publicEvidenceEnabled = Boolean(health.public_evidence_enabled)
      || /(public|pubmed|europe.?pmc|open.?targets|uniprot|chembl|pubchem|rcsb|dgidb|pharos)/.test(dataBackendText);
    if (!dataBackend && publicEvidenceEnabled) {
      dataBackend = dataBackendText.includes("cache")
        ? "cached public evidence + fixture chemistry"
        : "public evidence + fixture chemistry";
    } else if (!dataBackend && dataBackendText.includes("fallback")) {
      dataBackend = "fixture fallback";
    }
    dataBackend ||= "fixture";
    state.dataBackend = dataBackend;
    const dataLabel = formatBackendLabel(dataBackend);
    const dataBadge = $("#dataBadge");
    if (dataBadge) {
      const publicEvidence = publicEvidenceEnabled || dataBackend.includes("public");
      const fixtureOnly = !publicEvidence && dataBackend.includes("fixture");
      dataBadge.className = `data-badge is-${publicEvidence ? "public-db" : dataBackend.includes("cached") ? "cached" : fixtureOnly ? "fixture" : "connected"}`;
      dataBadge.title = fixtureOnly
        ? "현재 근거·후보·계산값은 검증용 fixture를 포함합니다"
        : `현재 과학 데이터 백엔드: ${configuredDataBackends.length ? configuredDataBackends.join(", ") : dataLabel}`;
      $("#dataBadgeText").textContent = `DATA · ${dataLabel}`;
    }

    const summaryTitle = $("#backendSummaryTitle");
    const summaryText = $("#backendSummaryText");
    const dynamicTargets = Boolean(health.dynamic_target_discovery?.enabled);
    if (summaryTitle) {
      summaryTitle.textContent = `LLM ${llmLabel} · 감사 ${reviewerLabel} · ${dynamicTargets ? "질환별 표적 탐색" : `데이터 ${dataLabel}`}`;
    }
    if (summaryText) {
      summaryText.textContent = dynamicTargets
        ? "질환별 Open Targets·PubMed 표적 발굴 · G0/G1 규칙 감사 상시 실행"
        : publicEvidenceEnabled || dataBackend.includes("public")
        ? "공개 근거 DB는 필요 시 실제 조회 · 후보 화학 계산은 검증용 fixture"
        : dataBackend.includes("fixture")
        ? "OpenAI 연결 여부와 별개로 과학 데이터는 현재 검증용 fixture"
        : "API 키는 브라우저가 아닌 서버의 .env에서만 관리";
    }
  }

  function legacyScientificCapability(tool) {
    const capabilities = Array.isArray(state.health?.ma2?.capabilities)
      ? state.health.ma2.capabilities
      : [];
    const capability = capabilities.find((item) => item?.tool_name === tool.capability);
    return capability && typeof capability === "object" ? capability : null;
  }

  function normalizedScientificToolGroup(item = {}) {
    const declared = String(
      item.group || item.group_id || item.category || item.section || item.kind || ""
    ).trim().toLowerCase().replaceAll("_", "-").replaceAll(" ", "-");
    if (["profile-builder", "profile-builders", "profile-tools", "receptor-profile", "receptor-preparation", "preparation"].includes(declared)) {
      return "profile-builder";
    }
    if (["run-tools", "run-tool", "runtime", "execution", "ma2"].includes(declared)) {
      return "run-tools";
    }
    const identity = String(
      item.display_name || item.label || item.name || item.tool_name || item.tool_id || item.key || ""
    ).toLowerCase();
    return /(pdbfixer|meeko|p2rank|posebusters|receptor|profile builder|profile-builder)/.test(identity)
      ? "profile-builder"
      : "run-tools";
  }

  function scientificToolStatus(item, offline = false) {
    if (offline) return { key: "offline", label: "OFFLINE", detail: "health endpoint offline" };
    const raw = String(item?.status || item?.availability || item?.state || "")
      .trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
    if (
      ["profile_required", "docking_profile_required", "profile_missing"].includes(raw)
      || item?.profile_required === true
    ) {
      return { key: "profile-required", label: "PROFILE REQUIRED", detail: raw || "profile_required" };
    }
    if (
      ["target_check_at_run", "verify_at_run", "verify_on_request", "target_dependent", "unknown"].includes(raw)
      || item?.target_check_at_run === true
      || item?.target_dependent === true
    ) {
      return { key: "target-check", label: "TARGET CHECK AT RUN", detail: raw || "target_dependent" };
    }
    if (["offline", "unreachable", "connection_failed"].includes(raw)) {
      return { key: "offline", label: "OFFLINE", detail: raw };
    }
    if (["not_configured", "configuration_required", "disabled"].includes(raw) || item?.configured === false) {
      return { key: "not-configured", label: "NOT CONFIGURED", detail: raw || "not_configured" };
    }
    if (["unavailable", "failed", "error", "missing_dependency"].includes(raw) || item?.available === false) {
      return { key: "unavailable", label: "UNAVAILABLE", detail: raw || "unavailable" };
    }
    if (["available", "ready", "configured", "completed", "live", "optional", "available_not_default"].includes(raw)) {
      return { key: "available", label: "AVAILABLE", detail: raw };
    }
    if (item?.available === true && item?.configured !== false) {
      return { key: "available", label: "AVAILABLE", detail: raw || "available" };
    }
    return { key: "target-check", label: "TARGET CHECK AT RUN", detail: raw || "status_not_reported" };
  }

  function normalizeScientificTool(item, index, source, offline = false) {
    const name = String(
      item?.display_name || item?.label || item?.name || item?.tool_name || item?.tool_id || item?.key || `Tool ${index + 1}`
    );
    const requirements = Array.isArray(item?.input_requirements)
      ? item.input_requirements.filter(Boolean).slice(0, 3).map(String)
      : [];
    const description = String(item?.description || item?.detail || item?.message || item?.operation || "");
    const version = String(item?.backend_version || item?.version || item?.tool_version || "");
    const versionDetail = scientificToolVersion(version);
    const reason = String(item?.reason_code || item?.reason || "");
    const executionState = item?.execution_enabled === false ? "EXECUTION DISABLED" : "";
    return {
      key: String(item?.key || item?.tool_id || item?.tool_name || `scientific-tool-${index}`),
      name,
      group: normalizedScientificToolGroup(item),
      status: scientificToolStatus(item, offline),
      detail: [description, versionDetail, executionState, ...requirements, reason]
        .filter(Boolean).join(" · "),
      source,
    };
  }

  function scientificToolVersion(value) {
    const normalized = String(value || "").trim().replace(/^v(?=\d)/i, "");
    if (!normalized) return "";
    if (/^\d+(?:\.\d+)+(?:[-+._A-Za-z0-9]*)$/.test(normalized)) return `v${normalized}`;
    return normalized
      .split(";")
      .map((part) => part.trim().replace(/^([A-Za-z][A-Za-z0-9_-]*)=/, "$1 "))
      .filter(Boolean)
      .join(" · ");
  }

  function resolvedScientificTools() {
    const declaredGroups = state.health?.scientific_tools?.groups;
    if (Array.isArray(declaredGroups)) {
      const groupedItems = declaredGroups.flatMap((group) => {
        if (!group || typeof group !== "object" || !Array.isArray(group.items)) return [];
        return group.items
          .filter((item) => item && typeof item === "object")
          .map((item) => ({
            ...item,
            group: group.group_id || group.key || group.label || item.group,
            execution_enabled: group.execution_enabled,
          }));
      });
      return {
        source: "scientific-tools",
        items: groupedItems.map((item, index) => normalizeScientificTool(item, index, "scientific-tools")),
      };
    }
    const declaredItems = state.health?.scientific_tools?.items;
    if (Array.isArray(declaredItems)) {
      return {
        source: "scientific-tools",
        items: declaredItems
          .filter((item) => item && typeof item === "object")
          .map((item, index) => normalizeScientificTool(item, index, "scientific-tools")),
      };
    }
    const offline = !state.health;
    return {
      source: offline ? "offline" : "legacy",
      items: SCIENTIFIC_TOOL_DEFINITIONS.map((tool, index) => {
        const capability = legacyScientificCapability(tool);
        return normalizeScientificTool({
          ...capability,
          key: tool.key,
          display_name: tool.name,
          group: tool.group,
          description: tool.description,
          status: capability?.status,
        }, index, "legacy", offline);
      }),
    };
  }

  function summarizeScientificToolItems(items) {
    if (!items.length) return "0 REPORTED";
    const available = items.filter((item) => item.status.key === "available").length;
    const attention = items.length - available;
    return `${items.length} · ${available} AVAILABLE${attention ? ` · ${attention} CHECK` : ""}`;
  }

  function renderScientificToolGroup(group, items) {
    const groupItems = items.filter((item) => item.group === group.key);
    const rows = groupItems.length
      ? groupItems.map((item) => `<article class="scientific-tool-item is-${esc(item.status.key)}">
          <div class="scientific-tool-item-head">
            <b>${esc(item.name)}</b>
            <span class="scientific-tool-status is-${esc(item.status.key)}">${esc(item.status.label)}</span>
          </div>
          ${item.detail ? `<p>${esc(item.detail)}</p>` : ""}
        </article>`).join("")
      : '<p class="scientific-tool-empty">STATUS NOT REPORTED BY THIS SERVER</p>';
    return `<details class="scientific-tool-group"${group.key === "run-tools" ? " open" : ""}>
      <summary><b>${esc(group.label)}</b><span>${esc(summarizeScientificToolItems(groupItems))}</span></summary>
      <div class="scientific-tool-group-list">${rows}</div>
    </details>`;
  }

  function updateScientificToolsPanel() {
    const summary = $("#scientificToolsSummary");
    const list = $("#scientificToolsList");
    if (!summary || !list) return;
    const resolved = resolvedScientificTools();
    if (resolved.source === "offline") {
      summary.textContent = "SERVER STATUS OFFLINE";
    } else if (resolved.source === "legacy") {
      summary.textContent = `LEGACY STATUS · ${summarizeScientificToolItems(resolved.items)}`;
    } else {
      summary.textContent = summarizeScientificToolItems(resolved.items);
    }
    list.innerHTML = [
      ...SCIENTIFIC_TOOL_GROUPS.map((group) => renderScientificToolGroup(group, resolved.items)),
      '<p class="scientific-tools-boundary">가용성·설정 상태입니다. 이번 실행의 실제 사용 여부는 TRACE에서 확인합니다.</p>',
    ].join("");
  }

  async function loadSourceCatalog() {
    try {
      const payload = await api("/api/sources");
      const { runtime, evaluation } = normalizeSourcePayload(payload);
      if (!runtime.length && !evaluation.length) throw new Error("소스 목록이 비어 있습니다");
      state.sourceCatalog = { runtime, evaluation, loaded: true, error: null };
    } catch (error) {
      state.sourceCatalog = {
        runtime: [],
        evaluation: [],
        loaded: false,
        error: error?.message || "소스 현황을 불러오지 못했습니다",
      };
    }
    updateSourceCatalogButton();
    renderSourceCatalog();
  }

  function normalizeSourcePayload(payload) {
    const root = payload?.sources && typeof payload.sources === "object" ? payload.sources : payload || {};
    const runtimeRaw = firstArray(
      root.runtime,
      root.runtime?.sources,
      root.runtime_sources,
      payload?.runtime_sources,
      Array.isArray(payload?.sources) ? payload.sources : null,
    );
    const evaluationRaw = firstArray(
      root.evaluation,
      root.evaluation?.sources,
      root.evaluation_sources,
      payload?.evaluation_sources,
    );
    return {
      runtime: runtimeRaw.map((source) => normalizeSource(source, false)),
      evaluation: evaluationRaw.map((source) => normalizeSource(source, true)),
    };
  }

  function firstArray(...values) {
    return values.find((value) => Array.isArray(value)) || [];
  }

  function normalizeSource(source, evaluation = false) {
    const access = String(source?.access || source?.access_kind || "public_api").toLowerCase();
    const status = String(source?.status || source?.connection_status || "").toLowerCase();
    const adapterActive = isTruthy(source?.adapter_active ?? source?.adapter?.active ?? source?.connected)
      || ["connected", "connected_adapter", "active", "live"].includes(status);
    const onDemand = isTruthy(source?.on_demand ?? source?.adapter?.on_demand)
      || access === "federated_api"
      || ["available_not_default", "federated"].includes(status);
    const configured = source?.configured === true
      || ["ready", "enabled", "configured", "credential_configured", "sidecar_configured", "cloud_project_configured", "dataset_ready", "on_demand_adapter", "on_demand_context_adapter"].includes(String(source?.configuration_state || "").toLowerCase());
    return {
      sourceId: String(source?.source_id || source?.id || source?.slug || source?.name || "unknown"),
      name: String(source?.name || source?.label || source?.source_id || "Unknown source"),
      stages: Array.isArray(source?.stages)
        ? source.stages.map(String)
        : String(source?.stages || source?.stage || "").split(/[,|]/).map((value) => value.trim()).filter(Boolean),
      access,
      endpoint: String(source?.endpoint || source?.url || ""),
      requirement: String(source?.requirement || source?.requires || ""),
      note: String(source?.note || source?.description || ""),
      status,
      adapterActive,
      onDemand,
      configured,
      evaluation,
      configurationState: String(source?.configuration_state || ""),
    };
  }

  function isTruthy(value) {
    if (value === true) return true;
    return ["true", "1", "yes", "active", "connected", "enabled"].includes(String(value || "").toLowerCase());
  }

  function sourceGroup(source) {
    if (source.evaluation || source.access === "evaluation_dataset") return "evaluation";
    if (source.adapterActive) return "connected";
    if (source.onDemand && source.configured) return "ondemand";
    if (
      source.requirement
      || ["not_configured", "configured_no_adapter", "catalog_only"].includes(source.status)
      || ["optional_sdk", "local_dataset", "credential_required", "cloud_dataset"].includes(source.access)
    ) {
      return "configuration";
    }
    return "ondemand";
  }

  function sourceGroupCounts() {
    return state.sourceCatalog.runtime.reduce((counts, source) => {
      counts[sourceGroup(source)] += 1;
      return counts;
    }, { connected: 0, ondemand: 0, configuration: 0, evaluation: 0 });
  }

  function updateSourceCatalogButton() {
    const title = $("#sourceCatalogButtonTitle");
    const text = $("#sourceCatalogButtonText");
    if (!title || !text) return;
    title.textContent = "DATA SOURCES";
    if (!state.sourceCatalog.loaded) {
      text.textContent = "상세 목록을 열어 서버 연결 상태 확인";
      return;
    }
    const counts = sourceGroupCounts();
    text.textContent = `${state.sourceCatalog.runtime.length} sources · 활성 ${counts.connected} · 필요 시 ${counts.ondemand} · 설정 ${counts.configuration}`;
  }

  function openSourceCatalog() {
    const dialog = $("#sourceCatalogDialog");
    renderSourceCatalog();
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  }

  function closeSourceCatalog() {
    const dialog = $("#sourceCatalogDialog");
    if (!dialog?.open) return;
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
  }

  async function openToolTrace() {
    const dialog = $("#toolTraceDialog");
    const runId = state.snapshot?.run_id;
    const cached = state.callTrace?.run_id === runId ? state.callTrace : null;
    renderToolTrace(cached, { loading: Boolean(!cached && runId && runId !== "preview") });
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    if (!runId || runId === "preview") {
      renderToolTrace(buildCallTraceFallback(state.snapshot));
      return;
    }
    const trace = await loadCallTrace(runId, { quiet: true });
    if (state.snapshot?.run_id !== runId) return;
    if (trace) renderToolTrace(trace);
    else if (!state.callTraceLoading) renderToolTrace(buildCallTraceFallback(state.snapshot), {
      fallbackReason: "호출 이력을 가져오지 못했습니다."
    });
  }

  function callTraceRevision(snapshot) {
    return JSON.stringify([snapshot?.run_id, snapshot?.stage, snapshot?.token_usage?.length,
      snapshot?.hypothesis_packet?.packet_id, snapshot?.candidate_packet?.packet_id,
      snapshot?.g0_verdict?.verdict_id, snapshot?.g1_verdict?.verdict_id,
      snapshot?.automatic_review_reports?.length]);
  }

  async function loadCallTrace(runId, options = {}) {
    state.callTraceRequests ||= new Set();
    if (!runId || runId === "preview" || state.callTraceRequests.has(runId)) return null;
    state.callTraceRequests.add(runId);
    state.callTraceLoading = true;
    const revision = callTraceRevision(state.snapshot);
    let applied = false;
    try {
      const trace = await api(`/api/runs/${encodeURIComponent(runId)}/calls`);
      if (state.snapshot?.run_id === runId) {
        state.callTrace = trace;
        state.callTraceRevision = revision;
        applied = true;
        renderMetrics();
        renderActivity();
        if (state.selectedTab === "report") renderArtifact();
        if ($("#toolTraceDialog")?.open) renderToolTrace(trace);
      }
      return trace;
    } catch (error) {
      if (!options.quiet) showToast("호출 이력 조회 실패", error.message, "error");
      return null;
    } finally {
      state.callTraceRequests.delete(runId);
      state.callTraceLoading = state.callTraceRequests.size > 0;
      // A final snapshot can arrive while the older /calls GET is in flight.
      // Catch up after success only; network errors must not create a loop.
      if (applied && state.snapshot?.run_id === runId && revision !== callTraceRevision(state.snapshot)) {
        void loadCallTrace(runId, { quiet: true });
      }
    }
  }

  function closeToolTrace() {
    const dialog = $("#toolTraceDialog");
    if (!dialog?.open) return;
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
  }

  function renderToolTrace(trace, options = {}) {
    const summary = $("#toolTraceSummary");
    const content = $("#toolTraceContent");
    const intro = $("#toolTraceIntro");
    if (!summary || !content || !intro) return;
    if (options.loading) {
      summary.innerHTML = ["모델 호출", "공개 DB 조회", "과학 도구 실행", "오류·중단"]
        .map((label) => `<span><b>—</b><small>${label}</small></span>`).join("");
      content.innerHTML = '<div class="source-catalog-loading"><span class="status-dot" aria-hidden="true"></span>호출 이력을 불러오는 중</div>';
      return;
    }
    const fallbackTrace = buildCallTraceFallback(state.snapshot);
    const safeTrace = trace || fallbackTrace;
    // A persisted snapshot is the authoritative fallback while the normalized
    // call endpoint is still loading (or when an older run returns empty
    // arrays).  This prevents a recorded 55-call CandidatePacket from being
    // presented as "0 calls" without inventing any activity.
    const recordedOrFallback = (recorded, fallback) => Array.isArray(recorded) && recorded.length
      ? recorded
      : fallback || [];
    const modelCalls = recordedOrFallback(safeTrace?.model_calls, fallbackTrace.model_calls);
    const publicCalls = recordedOrFallback(safeTrace?.public_data_calls, fallbackTrace.public_data_calls);
    const scientificCalls = recordedOrFallback(safeTrace?.scientific_tool_calls, fallbackTrace.scientific_tool_calls);
    const allCalls = [...modelCalls, ...publicCalls, ...scientificCalls];
    const failed = allCalls.filter((item) => isFailedTraceStatus(item.status)).length;
    const summaryData = safeTrace?.summary || {};
    const callCounts = callCountSummary(state.snapshot, safeTrace);
    summary.innerHTML = `
      <span><b>${compactNumber(callCounts.modelCalls)}</b><small>모델 호출</small></span>
      <span><b>${compactNumber(callCounts.publicDataCalls)}</b><small>공개 DB 조회·선택</small></span>
      <span><b>${compactNumber(callCounts.scientificToolCalls)}</b><small>과학 도구 실행</small></span>
      <span><b>${compactNumber(summaryData.failed ?? failed)}</b><small>오류·중단</small></span>`;
    intro.textContent = modelCalls.length
      ? "LLM 구조화 추론, 공개 데이터 조회, 과학 계산 도구 실행을 서로 분리해 기록한 실행 추적입니다."
      : "이 실행의 LLM API 호출은 0건입니다. 표시된 공개 DB와 과학 도구는 Orchestrator의 통제된 실행기에서 호출되었으며, LLM의 임의 function calling으로 과장하지 않습니다.";
    content.innerHTML = `
      ${options.fallbackReason ? `<div class="trace-architecture-note">통합 호출 API를 사용할 수 없어 저장된 실행 스냅샷에서 복원했습니다. ${esc(options.fallbackReason)}</div>` : ""}
      <div class="trace-architecture-note"><b>추적 원칙</b> · 상단 호출 수는 통합 호출 API의 정규화 기록 수입니다. exact는 개별 호출 기록, aggregate는 개별 HTTP 요청 수가 아닌 소스별 집계 기록 1건으로 계산됩니다. FIXTURE는 검증용 계산값이며 공개 DB 실조회와 구분됩니다.</div>
      ${traceSectionHtml("LLM 모델 호출", "구조화 추론·검토 호출과 토큰 사용", modelCalls)}
      ${traceSectionHtml("공개 데이터 도구", "PubMed·Open Targets, 소스 선택 결정 및 후보 공개 DB 어댑터", publicCalls)}
      ${traceSectionHtml("MA2 과학 계산 도구", "PlanSpec allowlist에 따라 실행된 후보 생성·화학·도킹·ADMET·순위화", scientificCalls)}`;
  }

  function traceSectionHtml(title, description, items) {
    const groups = groupTraceItems(items);
    return `<section class="trace-section"><header><div><h3>${esc(title)}</h3><p>${esc(description)}</p></div><span class="trace-section-count">${compactNumber(items.length)} calls · ${compactNumber(groups.length)} groups</span></header>
      ${groups.length ? `<div class="trace-list">${groups.map(traceItemHtml).join("")}</div>` : '<div class="trace-empty">이 실행에서 기록된 호출이 없습니다.</div>'}</section>`;
  }

  function groupTraceItems(items) {
    const groups = new Map();
    items.forEach((item) => {
      // Never collapse different failure causes into a single card.  For
      // example, an upstream HTTP 500 and a TLS transport interruption from
      // the same source require different operator actions even though both
      // have status=error.
      const failureDiagnostic = isFailedTraceStatus(item.status)
        ? [item.detail || "", item?.metadata?.failure_category || "", item?.metadata?.error_type || ""].join("~")
        : "";
      const key = [item.kind, item.agent, item.name, item.operation, item.status, Boolean(item.cached), Boolean(item.fixture), item.granularity, failureDiagnostic].join("|");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    });
    return [...groups.values()].map((records) => ({ ...records[0], _records: records }));
  }

  function traceItemHtml(item) {
    const records = item._records || [item];
    const failed = isFailedTraceStatus(item.status);
    const fixture = Boolean(item.fixture);
    const candidateIds = [...new Set(records.flatMap((record) => record.candidate_ids || []))];
    const taskIds = [...new Set(records.map((record) => record.task_id).filter(Boolean))];
    const recordsObserved = sum(records, (record) => record.records_observed || 0);
    const durationMs = sum(records, (record) => record.duration_ms || 0);
    const detailPayload = records.map((record) => ({
      call_id: record.call_id,
      task_id: record.task_id,
      candidate_ids: record.candidate_ids || [],
      trace_ids: record.trace_ids || [],
      status: record.status,
      started_at: record.started_at,
      finished_at: record.finished_at,
      metadata: record.metadata || {},
    }));
    const meta = [
      item.agent || item.node ? `주체 ${item.agent || item.node}` : "",
      taskIds.length ? `작업 ${taskIds.slice(0, 3).map(shortId).join(", ")}${taskIds.length > 3 ? ` +${taskIds.length - 3}` : ""}` : "",
      candidateIds.length ? `후보 ${candidateIds.slice(0, 5).map(shortId).join(", ")}${candidateIds.length > 5 ? ` +${candidateIds.length - 5}` : ""}` : "",
      recordsObserved ? `레코드 ${recordsObserved}건` : "",
      records.some((record) => record.duration_ms != null) ? `총 소요 ${formatDuration(durationMs)}` : "",
      item.source_version ? `버전 ${item.source_version}` : "",
    ].filter(Boolean);
    const neutral = ["empty", "unknown", "partial", "skipped", "not_configured", "skipped_missing_identifier", "skipped_call_budget", "skipped_time_budget", "policy_rejected"].includes(String(item.status || "").toLowerCase());
    const statusClass = failed ? "failed" : item.cached ? "cached" : neutral ? "neutral" : "ok";
    const statusLabel = item.cached ? "CACHE" : failed ? traceStatusLabel(item.status) : traceStatusLabel(item.status || "success");
    return `<article class="trace-item ${failed ? "is-failed" : ""} ${fixture ? "is-fixture" : ""}">
      <div class="trace-item-head"><div class="trace-item-title"><b>${esc(toolDisplayName(item.name))}</b>${item.operation ? `<code>${esc(item.operation)}</code>` : ""}${records.length > 1 ? `<span class="trace-badge aggregate">${records.length}회</span>` : ""}</div>
      <div><span class="trace-badge ${statusClass}">${esc(statusLabel)}</span>${fixture ? '<span class="trace-badge fixture">FIXTURE</span>' : ""}${item.granularity && item.granularity !== "exact" ? `<span class="trace-badge aggregate">${esc(String(item.granularity).toUpperCase())}</span>` : ""}</div></div>
      ${item.detail ? `<p class="trace-item-operation">${esc(item.detail)}${records.length > 1 ? ` · 동일 유형 ${records.length}회 묶음` : ""}</p>` : ""}
      ${meta.length ? `<div class="trace-item-meta">${meta.map((value) => `<span>${esc(value)}</span>`).join("")}</div>` : ""}
      <details data-disclosure="trace:${esc(item.call_id || `${item.kind}:${item.name}:${item.operation}`)}"><summary>${records.length > 1 ? `개별 호출 ${records.length}건` : "추적 메타데이터"}</summary><pre>${esc(JSON.stringify(records.length === 1 ? detailPayload[0] : detailPayload, null, 2))}</pre></details>
    </article>`;
  }

  const CANDIDATE_EXECUTION_FIELD_BY_TOOL = Object.freeze({
    chemistry_validator: "chemistry_execution",
    docking: "docking_execution",
    admet_predictor: "admet_execution",
  });

  function candidateExecutionRecords(snapshot = {}, toolName = "", candidateIds = []) {
    const field = CANDIDATE_EXECUTION_FIELD_BY_TOOL[toolName];
    if (!field || !candidateIds.length) return [];
    const requested = new Set(candidateIds);
    return (snapshot?.candidate_packet?.candidates || [])
      .filter((candidate) => requested.has(candidate.candidate_id) && candidate?.[field])
      .map((candidate) => ({ candidate_id: candidate.candidate_id, ...candidate[field] }));
  }

  function scientificExecutionTruth(rawCall = {}, candidateExecutions = [], projectedStatus = "") {
    const normalized = (value) => String(value || "").trim().toLowerCase();
    const rawExecutionStatus = normalized(rawCall.execution_status);
    const candidateStatuses = [...new Set(candidateExecutions.map((item) => normalized(item.status)).filter(Boolean))];
    const statusPriority = ["failed", "error", "timeout", "blocked", "unavailable", "skipped", "not_assessed", "not_run", "partial", "completed", "success"];
    const candidateStatus = [...candidateStatuses].sort((a, b) => {
      const left = statusPriority.indexOf(a);
      const right = statusPriority.indexOf(b);
      return (left < 0 ? statusPriority.length : left) - (right < 0 ? statusPriority.length : right);
    })[0] || "";
    const status = rawExecutionStatus
      || candidateStatus
      || (rawCall.success === false ? "failed" : "")
      || normalized(projectedStatus)
      || (rawCall.success === true ? "success" : "unknown");
    const reasonCodes = [...new Set([
      ...candidateExecutions.map((item) => item.reason_code),
      rawCall?.provenance?.capability_reason_code,
      PAPER_NATIVE_V6_FAILURE_REASON_CODES.has(rawCall?.provenance?.reason_code)
        ? rawCall.provenance.reason_code
        : null,
    ].filter(Boolean))];
    const capabilityDetails = [...new Set([
      ...candidateExecutions.map((item) => item?.provenance?.capability_detail),
      rawCall?.provenance?.capability_detail,
    ].filter(Boolean))];
    const backendNames = [...new Set(candidateExecutions.flatMap((item) => [
      item.backend_name,
      item.method && item.method !== "not-run" ? item.method : null,
      item?.provenance?.result?.method,
    ]).filter(Boolean))];
    return {
      status,
      rawExecutionStatus,
      candidateStatuses,
      reasonCodes,
      capabilityDetails,
      backendNames,
    };
  }

  function buildCallTraceFallback(snapshot = {}) {
    const modelCalls = (snapshot?.token_usage || []).map((usage, index) => {
      const accountingUnknown = Boolean(usage.estimated) && !String(usage.model || "").startsWith("mock:");
      return {
        call_id: `usage_${index}`,
        kind: "model",
        agent: usage.component,
        name: usage.model,
        operation: "structured_inference",
        status: accountingUnknown ? "accounting_unknown" : "success",
        cached: false,
        fixture: false,
        granularity: "exact",
        detail: accountingUnknown
          ? "보수적 사용량 상한을 보존했지만 provider의 실제 사용량은 확인하지 못해 실행을 중단했습니다."
          : `입력 ${usage.input_tokens || 0} · 출력 ${usage.output_tokens || 0} · 추론 ${usage.reasoning_tokens || 0} tokens`,
        metadata: {
          input_tokens: usage.input_tokens || 0,
          output_tokens: usage.output_tokens || 0,
          reasoning_tokens: usage.reasoning_tokens || 0,
          total_tokens: Number(usage.input_tokens || 0) + Number(usage.output_tokens || 0) + Number(usage.reasoning_tokens || 0),
          estimated_cost_usd: usage.estimated_cost_usd || 0,
          estimated: Boolean(usage.estimated),
          accounting_status: accountingUnknown ? "unknown" : "exact",
          usage_bound: accountingUnknown ? "conservative_upper_bound" : null,
          actual_usage_known: !accountingUnknown,
        },
      };
    });
    const publicCalls = [];
    (snapshot?.hypothesis_packet?.metadata?.target_source_statuses || []).forEach((source, index) => {
      publicCalls.push({
        call_id: `target_source_${index}`,
        kind: "public_data",
        agent: "MA1 · Target Discovery",
        name: source.source,
        operation: (source.operations || []).join(" · ") || "target_discovery",
        status: source.status || "unknown",
        cached: Number(source.cache_hits || 0) > 0,
        fixture: false,
        granularity: "aggregate",
        records_observed: source.records_observed || 0,
        detail: source.failure_codes?.length ? `오류 ${source.failure_codes.join(", ")}` : "질환-표적 근거 조회",
        metadata: { operations: source.operations || [], failure_codes: source.failure_codes || [] },
      });
    });
    const context = snapshot?.hypothesis_packet?.metadata?.context_routing || {};
    Object.entries(context.source_outcome_counts || {}).forEach(([source, calls], index) => {
      publicCalls.push({
        call_id: `context_source_${index}`,
        kind: "public_data",
        agent: "MA1 · Context Router",
        name: source,
        operation: "context_enrichment",
        status: Number(context.source_record_counts?.[source] || 0) > 0 ? "success" : "completed",
        cached: false,
        fixture: false,
        granularity: "aggregate",
        records_observed: context.source_record_counts?.[source] || 0,
        detail: `${calls}회 조회 · context-only 근거`,
        metadata: {},
      });
    });
    (snapshot?.candidate_packet?.metadata?.public_enrichment_queries || []).forEach((query, index) => {
      publicCalls.push({
        call_id: `candidate_source_${index}`,
        kind: "public_data",
        agent: "MA2 · Public Enrichment",
        name: query.source,
        operation: query.operation || (query.record_type === "source_routing_decision" ? "source_selection" : "candidate_enrichment"),
        status: query.status || (query.queried ? "success" : "skipped"),
        cached: Boolean(query.cache_hit),
        fixture: false,
        granularity: query.record_type === "source_routing_decision" ? "selection_decision" : "exact",
        candidate_ids: query.candidate_id ? [query.candidate_id] : [],
        records_observed: query.record_count || 0,
        detail: query.selection_reason || (query.error_type ? `${query.candidate_name || "후보"} · ${query.error_type}` : query.candidate_name || "후보 공개 DB 조회"),
        metadata: { route: query.route, queried: Boolean(query.queried), selected: query.selected, record_type: query.record_type },
      });
    });
    const candidateByTrace = new Map();
    (snapshot?.candidate_packet?.candidates || []).forEach((candidate) => {
      (candidate.tool_trace_ids || []).forEach((id) => {
        if (!candidateByTrace.has(id)) candidateByTrace.set(id, []);
        candidateByTrace.get(id).push(candidate.candidate_id);
      });
    });
    const taskById = new Map((snapshot?.plan_spec?.tasks || []).map((task) => [task.task_id, task]));
    const scientificCalls = (snapshot?.candidate_packet?.tool_calls || []).map((call) => {
      const task = taskById.get(call.task_id);
      const candidateIds = candidateByTrace.get(call.call_id) || [];
      const candidateExecutions = candidateExecutionRecords(snapshot, call.tool_name, candidateIds);
      const truth = scientificExecutionTruth(call, candidateExecutions);
      const executionDetail = [...truth.reasonCodes, ...truth.capabilityDetails].join(" · ");
      return {
        call_id: call.call_id,
        kind: "scientific_tool",
        agent: task?.route ? `MA2 · ${routeLabel(task.route)}` : "MA2 · Tool Executor",
        node: task?.name,
        name: call.tool_name,
        operation: call.operation,
        status: truth.status,
        cached: Boolean(call.cached),
        fixture: String(call.source_version || "").toLowerCase().includes("fixture"),
        granularity: "exact",
        started_at: call.started_at,
        finished_at: call.finished_at,
        duration_ms: durationBetween(call.started_at, call.finished_at),
        task_id: call.task_id,
        candidate_ids: candidateIds,
        source_version: call.source_version,
        detail: executionDetail || task?.name || call.error || "PlanSpec 승인 도구 실행",
        metadata: {
          input_hash: call.input_hash,
          output_hash: call.output_hash,
          error: call.error,
          raw_execution_status: truth.rawExecutionStatus || null,
          candidate_execution_statuses: truth.candidateStatuses,
          reason_codes: truth.reasonCodes,
          capability_details: truth.capabilityDetails,
          backend_names: truth.backendNames,
          status_authority: truth.rawExecutionStatus
            ? "candidate_packet.tool_calls.execution_status"
            : truth.candidateStatuses.length
              ? "candidate_execution"
              : "tool_call_transport_status",
        },
      };
    });
    const all = [...modelCalls, ...publicCalls, ...scientificCalls];
    return {
      run_id: snapshot?.run_id || "preview",
      summary: {
        total_calls: all.length,
        model_calls: modelCalls.length,
        public_data_calls: publicCalls.length,
        scientific_tool_calls: scientificCalls.length,
        successful: all.filter((item) => isSuccessfulTraceStatus(item.status)).length,
        failed: all.filter((item) => isFailedTraceStatus(item.status)).length,
        cached: all.filter((item) => item.cached).length,
        fixture: all.filter((item) => item.fixture).length,
      },
      model_calls: modelCalls,
      public_data_calls: publicCalls,
      scientific_tool_calls: scientificCalls,
    };
  }

  function callCountSummary(snapshot = {}, trace = null) {
    const fallback = buildCallTraceFallback(snapshot);
    const summary = trace?.summary || {};
    const hasNormalizedTrace = Boolean(
      trace
      && typeof trace === "object"
      && (
        trace.summary
        || Array.isArray(trace.model_calls)
        || Array.isArray(trace.public_data_calls)
        || Array.isArray(trace.scientific_tool_calls)
      ),
    );
    const normalizedTraceCount = (value, records) => {
      const numeric = value == null ? Number.NaN : Number(value);
      if (Number.isFinite(numeric) && numeric >= 0) return Math.trunc(numeric);
      return Array.isArray(records) ? records.length : 0;
    };

    // Once /calls is available, its normalized records and summary define the
    // displayed unit.  Do not compare them with operation-level counts from
    // the snapshot: one aggregate Open Targets record may contain multiple
    // operations, and Math.max would otherwise inflate the visible total.
    if (hasNormalizedTrace) {
      const modelCalls = normalizedTraceCount(summary.model_calls, trace.model_calls);
      const publicDataCalls = normalizedTraceCount(summary.public_data_calls, trace.public_data_calls);
      const scientificToolCalls = normalizedTraceCount(summary.scientific_tool_calls, trace.scientific_tool_calls);
      const composedTotal = modelCalls + publicDataCalls + scientificToolCalls;
      const normalizedTotal = summary.total_calls == null ? Number.NaN : Number(summary.total_calls);
      return {
        modelCalls,
        publicDataCalls,
        scientificToolCalls,
        totalCalls: Number.isFinite(normalizedTotal) && normalizedTotal >= 0
          ? Math.trunc(normalizedTotal)
          : composedTotal,
      };
    }

    // Before /calls arrives, retain the persisted-snapshot fallback so the UI
    // never flashes zero for a run whose recorded artifacts are already loaded.
    const modelCalls = Math.max(
      Array.isArray(snapshot?.token_usage) ? snapshot.token_usage.length : 0,
      fallback.model_calls.length,
    );
    const publicDataCalls = Math.max(
      countPublicDataCalls(snapshot),
      fallback.public_data_calls.length,
    );
    const scientificToolCalls = Math.max(
      Array.isArray(snapshot?.candidate_packet?.tool_calls) ? snapshot.candidate_packet.tool_calls.length : 0,
      fallback.scientific_tool_calls.length,
    );
    const composedTotal = modelCalls + publicDataCalls + scientificToolCalls;
    return {
      modelCalls,
      publicDataCalls,
      scientificToolCalls,
      totalCalls: composedTotal,
    };
  }

  function recordedExecutionTrace(snapshot = state.snapshot || {}) {
    const fallback = buildCallTraceFallback(snapshot);
    const trace = state.callTrace?.run_id === snapshot?.run_id ? state.callTrace : null;
    const recordedOrFallback = (recorded, inferred) => Array.isArray(recorded) && recorded.length
      ? recorded
      : inferred || [];
    const projectedScientific = recordedOrFallback(trace?.scientific_tool_calls, []);
    const rawScientificById = new Map(fallback.scientific_tool_calls.map((call) => [call.call_id, call]));
    const projectedScientificById = new Map(projectedScientific.map((call) => [call.call_id, call]));
    const scientificCalls = fallback.scientific_tool_calls.map((rawCall) => {
      const projected = projectedScientificById.get(rawCall.call_id) || {};
      return {
        ...projected,
        ...rawCall,
        metadata: { ...(projected.metadata || {}), ...(rawCall.metadata || {}) },
      };
    });
    projectedScientific.forEach((projected) => {
      if (rawScientificById.has(projected.call_id)) return;
      const candidateExecutions = candidateExecutionRecords(snapshot, projected.name, projected.candidate_ids || []);
      const truth = scientificExecutionTruth({}, candidateExecutions, projected.status);
      scientificCalls.push({
        ...projected,
        status: truth.status,
        metadata: {
          ...(projected.metadata || {}),
          candidate_execution_statuses: truth.candidateStatuses,
          reason_codes: truth.reasonCodes,
          capability_details: truth.capabilityDetails,
          backend_names: truth.backendNames,
          status_authority: truth.candidateStatuses.length ? "candidate_execution" : "calls_api_projection",
        },
      });
    });
    return {
      source: trace ? "call_api" : "snapshot",
      modelCalls: recordedOrFallback(trace?.model_calls, fallback.model_calls),
      publicCalls: recordedOrFallback(trace?.public_data_calls, fallback.public_data_calls),
      scientificCalls,
    };
  }

  function modelRolePresentation(component = "") {
    const normalized = String(component).toUpperCase().replaceAll("_", "-");
    const roles = {
      "MA1-INTAKE": {
        order: 1,
        stage: "MA1 입력",
        role: "질환 입력 해석",
        authority: "질환명 정규화·설명을 제안하는 범위. 외부 resolver의 식별자·유효성 판정은 변경할 수 없음.",
      },
      "MA1": {
        order: 2,
        stage: "MA1",
        role: "가설 서술 보강",
        authority: "기전, 반증 시험, claim 문장만 보강. 표적·근거·점수·순위·출처는 고정.",
      },
      "MA3-G0-REVIEWER": {
        order: 3,
        stage: "G0 독립 검토",
        role: "가설 독립 검토",
        authority: "저장 산출물에 연결된 추가 우려·불확실성만 제시. G0 verdict는 결정론적 감사표가 결정.",
      },
      "PLANNER": {
        order: 4,
        stage: "실행 계획",
        role: "실행 계획 설명 보강",
        authority: "수락 기준·도구 목적·소스 조회 중점 설명만 제안. 도구 허용·dispatch·예산·제약은 변경 불가.",
      },
      "MA3-G1-REVIEWER": {
        order: 5,
        stage: "G1 독립 검토",
        role: "후보·과정 독립 검토",
        authority: "후보·도구·근거 산출물의 추가 우려·불확실성만 제시. G1 verdict나 후보 순위는 변경 불가.",
      },
    };
    return roles[normalized] || {
      order: 99,
      stage: String(component || "MODEL").toUpperCase(),
      role: "구조화 모델 호출",
      authority: "이 실행에 저장된 구조화 출력 범위. 세부 권한은 원본 trace에서 확인.",
    };
  }

  function reviewerFindingCount(verdict = {}) {
    return (verdict.findings || []).filter((item) => item.deterministic === false || String(item.code || "").startsWith("REVIEWER_")).length;
  }

  function modelCallOutcome(component, snapshot = state.snapshot || {}) {
    const role = modelRolePresentation(component);
    const normalized = String(component).toUpperCase().replaceAll("_", "-");
    const intake = snapshot?.hypothesis_packet?.metadata?.intake_validation || {};
    const attestation = snapshot?.hypothesis_packet?.metadata?.llm_narrative_attestation || null;
    const coverage = snapshot?.plan_spec?.source_coverage || {};
    if (normalized === "MA1-INTAKE") {
      if (!intake.llm_suggestion_used) return "LLM 제안 채택 기록 없음";
      const disease = intake.resolved_disease_name || intake.suggested_normalized_name || intake.query_disease || "질환명";
      return `자문 설명 반영 · ${disease} · 입력 경고 ${compactNumber(intake.warnings?.length || 0)}건`;
    }
    if (normalized === "MA1") {
      if (attestation) return `서술 필드 ${compactNumber(attestation.changed_field_count ?? attestation.changed_fields?.length ?? 0)}건 변경 · 권한 필드 고정 검증`;
      return `가설 ${compactNumber(snapshot?.hypothesis_packet?.hypotheses?.length || 0)}건의 구조화 서술 채택 · 이전 실행은 변경 diff 미기록`;
    }
    if (normalized === "MA3-G0-REVIEWER") {
      const verdict = snapshot?.g0_verdict || {};
      return `${verdict.verdict || "판정 미기록"} · 리뷰어 추가 우려 ${compactNumber(reviewerFindingCount(verdict))}건 · verdict 결정권 없음`;
    }
    if (normalized === "PLANNER") {
      const applied = coverage.llm_overlay_applied === true ? "서술 overlay 반영" : "overlay 반영 미기록";
      return `${applied} · 실행 task ${compactNumber(snapshot?.plan_spec?.tasks?.length || 0)}개 · admitted source ${compactNumber(coverage.admitted_source_count || 0)}개`;
    }
    if (normalized === "MA3-G1-REVIEWER") {
      const verdict = snapshot?.g1_verdict || {};
      return `${verdict.verdict || "판정 미기록"} · 리뷰어 추가 우려 ${compactNumber(reviewerFindingCount(verdict))}건 · 순위·verdict 결정권 없음`;
    }
    return role.role;
  }

  function modelExecutionHtml(snapshot = state.snapshot || {}, options = {}) {
    const execution = recordedExecutionTrace(snapshot);
    const calls = [...execution.modelCalls].sort((a, b) => {
      const roleOrder = modelRolePresentation(a.agent || a.node).order - modelRolePresentation(b.agent || b.node).order;
      if (roleOrder) return roleOrder;
      return String(a.started_at || "").localeCompare(String(b.started_at || ""));
    });
    const totalCost = sum(calls, (call) => Number(call?.metadata?.estimated_cost_usd || 0));
    const totalTokens = sum(calls, (call) => Number(call?.metadata?.total_tokens || 0));
    const heading = options.includeHeading === false ? "" : `<header class="execution-section-heading"><div><span>모델 실행 기록</span><h4>실제 적용된 모델</h4><p>각 모델의 역할과 변경할 수 없는 범위를 함께 표시합니다.</p></div><strong>${compactNumber(calls.length)}회 · ${compactNumber(totalTokens)}토큰 · $${totalCost.toFixed(6)}</strong></header>`;
    if (!calls.length) {
      return `<section class="model-execution ${options.compact ? "is-compact" : ""}">${heading}<div class="report-empty">LLM API 호출 0건 · 이 실행에서 실제 모델이 적용됐다고 표시할 기록이 없습니다.</div></section>`;
    }
    return `<section class="model-execution ${options.compact ? "is-compact" : ""}">${heading}
      <div class="model-execution-table-wrap"><table class="model-execution-table">
        <caption>실제 LLM 호출별 역할, 모델, 상태, 사용량, 권한 범위와 결과</caption>
        <thead><tr><th scope="col">단계·역할</th><th scope="col">모델·상태</th><th scope="col">토큰·비용</th><th scope="col">허용된 권한</th><th scope="col">실제 반영 결과</th></tr></thead>
        <tbody>${calls.map((call) => {
          const role = modelRolePresentation(call.agent || call.node);
          const usage = call.metadata || {};
          const status = traceStatusLabelKo(call.status || "unknown");
          const statusClass = isFailedTraceStatus(call.status) ? "is-failed" : isSuccessfulTraceStatus(call.status) ? "is-success" : "is-neutral";
          const tokenDetail = `입력 ${compactNumber(usage.input_tokens || 0)} · 출력 ${compactNumber(usage.output_tokens || 0)} · 내부 추론 계수 ${compactNumber(usage.reasoning_tokens || 0)}`;
          const cost = Number(usage.estimated_cost_usd || 0);
          const accountingUnknown = String(call.status || "").toLowerCase() === "accounting_unknown" || usage.accounting_status === "unknown";
          const usageQualifier = accountingUnknown
            ? " · 보수적 상한 · 실제 사용량 미확인"
            : usage.estimated ? " · 추정값" : "";
          const outcome = accountingUnknown
            ? "Provider 실제 사용량을 확인하지 못해 보수적 상한을 원장에 보존하고 실행을 중단했습니다."
            : modelCallOutcome(call.agent || call.node, snapshot);
          return `<tr>
            <td><span class="model-stage">${esc(role.stage)}</span><b>${esc(role.role)}</b><small>${esc(call.agent || call.node || "주체 미기록")}</small></td>
            <td><b>${esc(call.name || "모델명 미기록")}</b><span class="execution-status ${statusClass}">${esc(status)}</span>${call.granularity === "aggregate" ? '<small>persisted usage 기반</small>' : call.duration_ms != null ? `<small>${esc(formatDuration(call.duration_ms))}</small>` : ""}</td>
            <td><b>${compactNumber(usage.total_tokens || (Number(usage.input_tokens || 0) + Number(usage.output_tokens || 0) + Number(usage.reasoning_tokens || 0)))}토큰</b><small>${esc(tokenDetail)}</small><small>$${cost.toFixed(6)}${esc(usageQualifier)}</small></td>
            <td><p>${esc(role.authority)}</p></td>
            <td><p>${esc(outcome)}</p></td>
          </tr>`;
        }).join("")}</tbody>
      </table></div>
      <p class="execution-trace-note">${execution.source === "call_api" ? "호출 추적 API의 실제 기록" : "저장 스냅샷의 토큰 사용량에서 복원한 기록"} · ‘내부 추론 계수’는 사용량 수치이며 사고과정 원문이 아닙니다.</p>
    </section>`;
  }

  function modelDecisionTraceHtml(snapshot = state.snapshot || {}) {
    const events = Array.isArray(snapshot?.events) ? snapshot.events : [];
    const acceptedKinds = new Set([
      "structured_generation_response",
      "artifact_handoff",
      "audit_decision",
      "selective_rework",
      "plan_handoff",
      "candidate_handoff",
      "execution_handoff",
    ]);
    const relevantEvents = events.filter((event) => {
      const kind = String(event?.metadata?.interaction?.kind || "");
      return acceptedKinds.has(kind) || String(event?.event_type || "") === "workflow_terminal";
    });
    let terminalRecorded = false;
    const steps = relevantEvents.filter((event) => {
      if (String(event?.event_type || "") !== "workflow_terminal") return true;
      if (terminalRecorded) return false;
      terminalRecorded = true;
      return true;
    });
    const modelCalls = recordedExecutionTrace(snapshot).modelCalls;
    const revision = snapshot?.hypothesis_packet?.metadata?.revision_lineage || null;
    const interactionLabel = (kind) => ({
      structured_generation_response: "구조화 결과 수신",
      artifact_handoff: "산출물 전달",
      audit_decision: "독립 감사 판정",
      selective_rework: "선택 재작업 요청",
      plan_handoff: "실행 계획 전달",
      candidate_handoff: "후보 패킷 전달",
      execution_handoff: "실행 결과 전달",
    })[kind] || "실행 기록";
    const stepDetail = (event) => {
      const interaction = event?.metadata?.interaction || {};
      const kind = String(interaction.kind || "");
      if (kind === "audit_decision") {
        const codes = interaction.details?.finding_codes || [];
        const labels = codes.slice(0, 4).map((code) => findingGroupTitle({ code }));
        const high = Number(interaction.basis?.find((item) => /High 이상/.test(String(item)))?.match(/(\d+)개/)?.[1] || 0);
        return `${verdictLabel(interaction.decision)} · 미해결 감사 항목 ${compactNumber(codes.length)}건${high ? ` · 중요 ${compactNumber(high)}건` : ""}${labels.length ? ` · ${labels.join(" · ")}` : ""}`;
      }
      if (kind === "selective_rework") {
        const details = interaction.details || {};
        return `영향 주장 ${compactNumber(details.affected_claim_ids?.length || 0)}건 · 다시 수행할 단계 ${compactNumber(event?.metadata?.invalidated_nodes?.length || details.invalidated_nodes?.length || 0)}개 · 기존 산출물은 보존`;
      }
      if (kind === "structured_generation_response") {
        const koreanBasis = (interaction.basis || []).filter((item) => /[가-힣]/.test(String(item))).slice(0, 3);
        return koreanBasis.length ? koreanBasis.join(" · ") : "스키마 검증을 통과한 구조화 결과를 저장하고 다음 단계로 전달";
      }
      if (kind === "artifact_handoff") {
        const koreanBasis = (interaction.basis || []).filter((item) => /[가-힣]/.test(String(item))).slice(0, 2);
        return koreanBasis.length ? koreanBasis.join(" · ") : interaction.summary || "구조화 산출물 전달";
      }
      if (String(event?.event_type || "") === "workflow_terminal") return runStatusPresentation(snapshot).detail;
      return interaction.summary || event?.message || "상세 기록 없음";
    };
    const timeline = steps.map((event, index) => {
      const interaction = event?.metadata?.interaction || {};
      const kind = String(interaction.kind || "");
      const terminal = String(event?.event_type || "") === "workflow_terminal";
      const sender = terminal ? "오케스트레이터" : interaction.sender || event.node || "주체 미기록";
      const receiver = terminal ? "실행 종료" : interaction.receiver || "다음 단계";
      const artifacts = [...new Set([...(interaction.artifact_ids || []), ...(event.artifact_ids || [])])];
      const evidenceCount = interaction.evidence_ids?.length || 0;
      const decision = String(interaction.decision || "").toUpperCase();
      const stateClass = terminal
        ? "is-terminal"
        : ["REVISE", "REPLAN", "ABSTAIN", "BLOCK"].includes(decision)
        ? "is-warning"
        : "is-complete";
      return `<li class="model-handoff-step ${stateClass}">
        <span class="model-handoff-index">${String(index + 1).padStart(2, "0")}</span>
        <div class="model-handoff-body">
          <header><b>${esc(terminal ? runStatusPresentation(snapshot).label : interactionLabel(kind))}</b><time>${esc(formatTime(event.created_at))}</time></header>
          <div class="model-handoff-route"><span>${esc(sender)}</span><i aria-hidden="true">→</i><span>${esc(receiver)}</span>${interaction.model ? `<code>${esc(interaction.model)}</code>` : ""}</div>
          <p>${esc(interaction.summary || (terminal ? runStatusPresentation(snapshot).label : "저장된 전달 요약 없음"))}</p>
          <small>${esc(stepDetail(event))}</small>
          ${artifacts.length || evidenceCount ? `<footer>${artifacts.slice(0, 3).map((id) => `<code>${esc(shortId(id))}</code>`).join("")}${evidenceCount ? `<span>연결 근거 ${compactNumber(evidenceCount)}건</span>` : ""}</footer>` : ""}
        </div>
      </li>`;
    }).join("");
    const revisionHtml = revision ? `<aside class="model-revision-lineage">
      <b>재작업 계보 · revision ${compactNumber(revision.revision_number || 0)}</b>
      <span>${esc(shortId(revision.parent_packet_id || "이전 패킷 미기록"))} → ${esc(shortId(snapshot?.hypothesis_packet?.packet_id || "현재 패킷 미기록"))}</span>
      <small>과학 내용 변경 ${revision.scientific_changed ? "있음" : "없음"} · 이전 산출물과 해시를 모두 보존</small>
    </aside>` : "";
    return `<section class="model-handoff-trace">
      <div class="model-handoff-intro">
        <div><b>모델끼리 자유 대화하는 구조가 아닙니다.</b><p>오케스트레이터가 Terra의 구조화 산출물을 Sol 감사에 순서대로 전달하고, Sol의 판정·재작업 요청을 다시 MA1에 돌려보냅니다.</p></div>
        <strong>모델 호출 ${compactNumber(modelCalls.length)}회 · 전달 단계 ${compactNumber(steps.length)}개</strong>
      </div>
      ${timeline ? `<ol class="model-handoff-list">${timeline}</ol>` : '<div class="report-empty">저장된 모델 전달 기록이 없습니다.</div>'}
      ${revisionHtml}
      <p class="model-handoff-boundary"><b>표시 범위</b> 저장된 구조화 출력, 산출물 ID, 근거 연결, Gate 판정, 재작업 지시를 보여줍니다. 모델의 비공개 내부 사고과정 원문은 표시하지 않습니다.</p>
    </section>`;
  }

  function llmNarrativeAttestation(snapshot = state.snapshot || {}) {
    const value = snapshot?.hypothesis_packet?.metadata?.llm_narrative_attestation;
    return value && typeof value === "object" ? value : null;
  }

  function llmChangesHtml(snapshot = state.snapshot || {}, options = {}) {
    const attestation = llmNarrativeAttestation(snapshot);
    const heading = options.includeHeading === false ? "" : `<header class="execution-section-heading"><div><span>모델 서술 변경</span><h4>모델이 바꾸기 전·후</h4><p>실행 패킷에 저장된 가설 문장 차이만 표시합니다.</p></div></header>`;
    if (!attestation) {
      return `<section class="llm-changes ${options.compact ? "is-compact" : ""}">${heading}<div class="legacy-diff-note"><b>이전 실행(legacy run): 변경 전·후 diff가 기록되지 않음</b><span>모델 호출·토큰은 확인할 수 있지만, 어떤 문장이 바뀌었는지는 이 snapshot에 저장되지 않아 추정하지 않습니다.</span></div></section>`;
    }
    const changes = Array.isArray(attestation.changed_fields) ? attestation.changed_fields : [];
    const frozen = Array.isArray(attestation.frozen_authority_fields) ? attestation.frozen_authority_fields : [];
    return `<section class="llm-changes ${options.compact ? "is-compact" : ""}">${heading}
      <div class="llm-change-summary"><b>${esc(attestation.component || "MA1")} · ${esc(attestation.model || "모델 미기록")}</b><span>변경 필드 ${compactNumber(attestation.changed_field_count ?? changes.length)}개 · 권한 ${esc(attestation.authority_scope || "narrative_only")}</span></div>
      ${changes.length ? `<div class="llm-change-list">${changes.map((item) => `<article>
        <header><b>${esc(item.field || "필드")}</b><span>${esc(shortId(item.hypothesis_id || ""))}${item.claim_id ? ` · ${esc(shortId(item.claim_id))}` : ""}</span></header>
        <div><section><span>변경 전</span><p>${esc(item.before?.text || "미기록")}</p><code>${esc(item.before?.sha256 ? shortId(item.before.sha256) : "해시 미기록")}</code></section>
        <section><span>변경 후</span><p>${esc(item.after?.text || "미기록")}</p><code>${esc(item.after?.sha256 ? shortId(item.after.sha256) : "해시 미기록")}</code></section></div>
      </article>`).join("")}</div>` : '<div class="report-empty is-clear">서술 변경 0건으로 기록됨</div>'}
      ${frozen.length ? `<div class="frozen-field-row"><b>변경 금지된 필드</b>${frozen.map((field) => `<span>${esc(field)}</span>`).join("")}</div>` : ""}
    </section>`;
  }

  function isSkippedTraceStatus(status) {
    const value = String(status || "").toLowerCase();
    return value.startsWith("skipped") || ["not_configured", "not_assessed", "not_run", "not_applicable", "optional_not_assessed", "policy_rejected"].includes(value);
  }

  function ma2ExecutionToolPresentation(name = "") {
    const presentations = {
      chemistry_validator: { category: "scientific_computation", engine: "RDKit", purpose: "화학 구조 표준화·검증" },
      admet_predictor: { category: "scientific_computation", engine: "ADMET-AI", purpose: "6종 ADMET·독성 예측" },
      docking: { category: "scientific_computation", engine: "AutoDock Vina", purpose: "결합 포즈 점수화" },
      de_novo_generator: { category: "scientific_computation", engine: "RDKit BRICS", purpose: "신규 분자 후보 생성" },
      target_normalizer: { category: "pipeline_operation", engine: "표적 식별자 정규화", purpose: "식별자·표적 연결" },
      repurposing_search: { category: "pipeline_operation", engine: "약물 재창출 검색", purpose: "공개 데이터 후보 수집·라우팅" },
      restricted_regulatory_registry: { category: "pipeline_operation", engine: "규제·제한 조건 조회", purpose: "제한 레지스트리 확인" },
      candidate_judge: { category: "pipeline_operation", engine: "후보 우선순위 규칙", purpose: "결정론적 제약·순위 평가" },
    };
    return presentations[name] || { category: "pipeline_operation", engine: toolDisplayName(name), purpose: "MA2 pipeline operation" };
  }

  function executionReasonLabel(value = "") {
    const labels = {
      DIRECT_MEASURED_ACTIVITY_AVAILABLE: "표적 연결 실측 활성값 확보 · 선택 도킹 생략",
      OPTIONAL_DOCKING_NOT_ASSESSED: "선택 도킹 미평가",
      LIVE_DOCKING_NOT_CONFIGURED: "검증 receptor/box profile 미설정",
      PAPER_NATIVE_V6_INSUFFICIENT_OR_INVALID_PRIMARY_SUPPORT: "후보별 원저 근거 3건 미만 또는 근거 결속 검증 실패",
      PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT: "후보별 검증 원저 근거 3건 미만 · 승격 불가",
      PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID: "원저 근거의 후보·활성·출처 결속 계약 불일치",
      PAPER_NATIVE_V6_DELEGATE_RETRIEVAL_UNAVAILABLE: "V6 공개 후보 원천 조회 미완료",
      PAPER_NATIVE_V6_DELEGATE_PROVENANCE_INVALID: "V6 후보 원천 provenance 계약 불일치",
    };
    return labels[value] || String(value || "").replaceAll("_", " ");
  }

  const PAPER_NATIVE_V6_FAILURE_REASON_CODES = new Set([
    "PAPER_NATIVE_V6_INSUFFICIENT_OR_INVALID_PRIMARY_SUPPORT",
    "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT",
    "PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID",
  ]);
  const PAPER_NATIVE_V6_WRAPPER_SCREEN_REASON_CODES = new Set([
    "SEED_CARDINALITY_NOT_EXACT",
    "PRIMARY_SOURCE_NOT_MAPPING",
    "PAPER_NATIVE_V6_EXACT_BINDING_INVALID",
    "DELEGATE_NOT_RETURNED",
  ]);
  const SHA256_HEX_PATTERN = /^[a-f0-9]{64}$/i;
  const CHEMBL_MOLECULE_ID_PATTERN = /^CHEMBL[1-9][0-9]*$/;

  function paperNativeV6WrapperReasonLabel(value = "") {
    const labels = {
      SEED_CARDINALITY_NOT_EXACT: "후보별 활성·근거가 정확히 1:1이 아님",
      PRIMARY_SOURCE_NOT_MAPPING: "원저 근거 구조가 없거나 형식이 다름",
      PAPER_NATIVE_V6_EXACT_BINDING_INVALID: "후보·활성·원저의 정확한 결속 검증 실패",
      DELEGATE_NOT_RETURNED: "고정 후보 풀에는 있으나 delegate가 반환하지 않음",
      PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED: "원저 초록에서 후보·표적·활성값을 한 문장으로 명확히 결속하지 못함",
      DOCUMENT_REVIEW_NOT_PRIMARY_ACTIVITY_REPORT: "리뷰 논문으로 판정되어 원저 활성 근거로 사용할 수 없음",
      DOCUMENT_NOT_PRIMARY_PUBLICATION: "원저 논문 유형이 아니어서 후보 근거에서 제외",
      PRIMARY_ABSTRACT_NOT_AVAILABLE: "원저 초록을 확보하지 못함",
      PUBMED_TYPE_VERIFICATION_MISSING: "PubMed에서 원저 논문 유형을 확인하지 못함",
      PUBMED_EXCLUDED_PUBLICATION_TYPE: "리뷰·사설 등 제외 대상 논문 유형",
      PUBMED_JOURNAL_ARTICLE_TYPE_MISSING: "Journal Article 유형 확인 실패",
      ACTIVITY_DOCUMENT_ID_MISSING: "활성 레코드에 연결된 문헌 ID가 없음",
      PRIMARY_DOCUMENT_LOOKUP_UNAVAILABLE: "연결 원문 메타데이터 조회 실패",
    };
    return labels[value] || String(value || "").replaceAll("_", " ");
  }

  function boundedPaperNativeV6Count(value, { minimum = 0, maximum = 12 } = {}) {
    return Number.isInteger(value) && value >= minimum && value <= maximum
      ? value
      : null;
  }

  function paperNativeV6WrapperScreenProjection(value) {
    if (value == null) return { status: "missing", rows: [], rowsSha256: null, attestationSha256: null };
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return { status: "malformed", rows: [], rowsSha256: null, attestationSha256: null };
    }
    const fixedIds = Array.isArray(value.fixed_pool_candidate_ids)
      ? value.fixed_pool_candidate_ids
      : [];
    const rows = Array.isArray(value.rows) ? value.rows : [];
    const hashesValid = [
      value.fixed_pool_candidate_ids_sha256,
      value.rows_sha256,
      value.attestation_sha256,
    ].every((hash) => typeof hash === "string" && SHA256_HEX_PATTERN.test(hash));
    const fixedIdsValid = fixedIds.length >= 1
      && fixedIds.length <= 12
      && fixedIds.length === new Set(fixedIds).size
      && fixedIds.every((candidateId) => (
        typeof candidateId === "string"
        && CHEMBL_MOLECULE_ID_PATTERN.test(candidateId)
      ));
    const rowsValid = rows.length === fixedIds.length && rows.every((row, index) => {
      if (!row || typeof row !== "object" || Array.isArray(row)) return false;
      const status = row.wrapper_screen_status;
      const reasonCode = row.wrapper_screen_reason_code;
      const checks = row.wrapper_screen_checks;
      const checksShapeValid = checks
        && typeof checks === "object"
        && !Array.isArray(checks)
        && Object.keys(checks).length === 3
        && ["cardinality_exact", "primary_mapping", "exact_binding"].every(
          (key) => typeof checks[key] === "boolean",
        );
      const statusContractValid = checksShapeValid && (
        (
          status === "verified"
          && reasonCode == null
          && row.delegate_returned === true
          && checks.cardinality_exact === true
          && checks.primary_mapping === true
          && checks.exact_binding === true
        )
        || (
          status === "not_returned"
          && reasonCode === "DELEGATE_NOT_RETURNED"
          && row.delegate_returned === false
          && checks.cardinality_exact === false
          && checks.primary_mapping === false
          && checks.exact_binding === false
        )
        || (
          status === "rejected"
          && reasonCode === "SEED_CARDINALITY_NOT_EXACT"
          && row.delegate_returned === true
          && checks.cardinality_exact === false
          && checks.exact_binding === false
        )
        || (
          status === "rejected"
          && reasonCode === "PRIMARY_SOURCE_NOT_MAPPING"
          && row.delegate_returned === true
          && checks.cardinality_exact === true
          && checks.primary_mapping === false
          && checks.exact_binding === false
        )
        || (
          status === "rejected"
          && reasonCode === "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
          && row.delegate_returned === true
          && checks.cardinality_exact === true
          && checks.primary_mapping === true
          && checks.exact_binding === false
        )
      );
      return row.pool_index === index
        && row.fixed_pool_rank === index + 1
        && row.molecule_chembl_id === fixedIds[index]
        && typeof row.delegate_returned === "boolean"
        && ["verified", "rejected", "not_returned"].includes(status)
        && (reasonCode == null || PAPER_NATIVE_V6_WRAPPER_SCREEN_REASON_CODES.has(reasonCode))
        && statusContractValid
        && typeof row.wrapper_screen_checks_sha256 === "string"
        && SHA256_HEX_PATTERN.test(row.wrapper_screen_checks_sha256)
        && row.fixture === false
        && typeof row.row_sha256 === "string"
        && SHA256_HEX_PATTERN.test(row.row_sha256);
    });
    const valid = value.schema_version === "paper_native_v6_wrapper_atomic_screen_projection/v1"
      && value.row_count === rows.length
      && value.complete === true
      && value.fixture === false
      && fixedIdsValid
      && rowsValid
      && hashesValid;
    if (!valid) {
      return { status: "malformed", rows: [], rowsSha256: null, attestationSha256: null };
    }
    return {
      status: "valid",
      rows: rows.map((row) => ({
        rank: row.fixed_pool_rank,
        candidateId: row.molecule_chembl_id,
        status: row.wrapper_screen_status,
        reasonCode: row.wrapper_screen_reason_code,
      })),
      rowsSha256: value.rows_sha256,
      attestationSha256: value.attestation_sha256,
    };
  }

  function paperNativeV6FailureSummaryProjection(value, rawProjection = null) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return { status: "missing", rows: [], rowsSha256: null, attestationSha256: null, reasonCounts: {} };
    }
    const fixedIds = Array.isArray(value.fixed_pool_candidate_ids)
      ? value.fixed_pool_candidate_ids
      : [];
    const rows = Array.isArray(value.rows) ? value.rows : [];
    const reasonCounts = value.rejection_reason_counts;
    const hashesValid = [
      value.fixed_pool_candidate_ids_sha256,
      value.rows_sha256,
      value.attestation_sha256,
    ].every((hash) => typeof hash === "string" && SHA256_HEX_PATTERN.test(hash));
    const reasonCountsValid = reasonCounts
      && typeof reasonCounts === "object"
      && !Array.isArray(reasonCounts)
      && Object.entries(reasonCounts).every(([reason, count]) => (
        /^[A-Z][A-Z0-9_]{0,119}$/.test(reason)
        && Number.isInteger(count)
        && count >= 1
        && count <= 12
      ));
    const rowsValid = rows.length === fixedIds.length && rows.every((row, index) => {
      const rawRow = rawProjection?.status === "valid" ? rawProjection.rows[index] : null;
      const sourceReason = row?.source_primary_reason_code;
      return row
        && typeof row === "object"
        && !Array.isArray(row)
        && row.fixed_pool_rank === index + 1
        && row.molecule_chembl_id === fixedIds[index]
        && typeof row.delegate_returned === "boolean"
        && ["verified", "rejected", "not_returned"].includes(row.wrapper_screen_status)
        && (row.wrapper_screen_reason_code == null || PAPER_NATIVE_V6_WRAPPER_SCREEN_REASON_CODES.has(row.wrapper_screen_reason_code))
        && (sourceReason == null || /^[A-Z][A-Z0-9_]{0,119}$/.test(sourceReason))
        && row.wrapper_screen_checks
        && typeof row.wrapper_screen_checks === "object"
        && !Array.isArray(row.wrapper_screen_checks)
        && ["cardinality_exact", "primary_mapping", "exact_binding"].every(
          (key) => typeof row.wrapper_screen_checks[key] === "boolean",
        )
        && typeof row.row_sha256 === "string"
        && SHA256_HEX_PATTERN.test(row.row_sha256)
        && (!rawRow || (
          rawRow.rank === row.fixed_pool_rank
          && rawRow.candidateId === row.molecule_chembl_id
          && rawRow.status === row.wrapper_screen_status
          && rawRow.reasonCode === row.wrapper_screen_reason_code
        ));
    });
    const rawHashesMatch = rawProjection?.status !== "valid" || (
      rawProjection.rowsSha256 === value.rows_sha256
      && rawProjection.attestationSha256 === value.attestation_sha256
    );
    const valid = value.schema_version === "paper_native_v6_wrapper_atomic_screen_projection/v1"
      && fixedIds.length >= 1
      && fixedIds.length <= 12
      && fixedIds.length === new Set(fixedIds).size
      && fixedIds.every((candidateId) => typeof candidateId === "string" && CHEMBL_MOLECULE_ID_PATTERN.test(candidateId))
      && value.row_count === rows.length
      && value.complete === true
      && value.fixture === false
      && value.source_reason_projection_verified === true
      && hashesValid
      && reasonCountsValid
      && rowsValid
      && rawHashesMatch;
    if (!valid) {
      return { status: "malformed", rows: [], rowsSha256: null, attestationSha256: null, reasonCounts: {} };
    }
    return {
      status: "valid",
      rows: rows.map((row) => ({
        rank: row.fixed_pool_rank,
        candidateId: row.molecule_chembl_id,
        status: row.wrapper_screen_status,
        reasonCode: row.wrapper_screen_reason_code,
        sourceReasonCode: row.source_primary_reason_code,
      })),
      rowsSha256: value.rows_sha256,
      attestationSha256: value.attestation_sha256,
      reasonCounts: { ...reasonCounts },
    };
  }

  function paperNativeV6FailureRecords(packet = {}) {
    const calls = Array.isArray(packet?.tool_calls) ? packet.tool_calls : [];
    const executions = Array.isArray(packet?.target_executions) ? packet.target_executions : [];
    const records = [];
    const representedReasonCodes = new Set();
    const failureSummary = packet?.metadata?.scientific_failure_summary;
    const summaryReasonCode = failureSummary?.reason_code;
    let summaryConsumed = false;
    calls.forEach((call, index) => {
      const provenance = call?.provenance;
      if (!provenance || typeof provenance !== "object" || Array.isArray(provenance)) return;
      const reasonCode = provenance.reason_code;
      if (!PAPER_NATIVE_V6_FAILURE_REASON_CODES.has(reasonCode)) return;
      const execution = executions.find((item) => (
        Array.isArray(item?.tool_call_ids) && item.tool_call_ids.includes(call.call_id)
      ));
      const targetSymbol = String(
        execution?.target?.symbol || execution?.target_key || "표적 미기록",
      ).slice(0, 80);
      const supportCandidateCount = boundedPaperNativeV6Count(
        provenance.support_candidate_count,
      );
      const rejectionCount = boundedPaperNativeV6Count(provenance.rejection_count);
      let projection = paperNativeV6WrapperScreenProjection(
        provenance.wrapper_atomic_screen_projection,
      );
      if (projection.status === "valid") {
        const verifiedCount = projection.rows.filter((row) => row.status === "verified").length;
        const rejectedCount = projection.rows.filter((row) => row.status === "rejected").length;
        if (
          (supportCandidateCount != null && supportCandidateCount !== verifiedCount)
          || (rejectionCount != null && rejectionCount !== rejectedCount)
        ) {
          projection = {
            status: "malformed",
            rows: [],
            rowsSha256: null,
            attestationSha256: null,
          };
        }
      }
      if (!summaryConsumed && summaryReasonCode === reasonCode) {
        const summaryProjection = paperNativeV6FailureSummaryProjection(
          failureSummary?.wrapper_atomic_screen_projection,
          projection,
        );
        if (summaryProjection.status === "valid") {
          projection = summaryProjection;
          summaryConsumed = true;
        }
      }
      representedReasonCodes.add(reasonCode);
      records.push({
        key: String(call.call_id || `paper-v6-failure-${index}`),
        targetSymbol,
        reasonCode,
        supportCandidateCount,
        minimumVerifiedCandidateCount: boundedPaperNativeV6Count(
          provenance.minimum_verified_candidate_count,
          { minimum: 1 },
        ),
        rejectionCount,
        evidenceGrade: ["B", "C"].includes(provenance.evidence_grade)
          ? provenance.evidence_grade
          : null,
        promotionEligible: provenance.promotion_eligible === false ? false : null,
        projection,
        source: "failed_tool_call",
      });
    });
    const packetReasons = Array.isArray(packet?.abstain_reason_codes)
      ? packet.abstain_reason_codes
      : [];
    packetReasons.forEach((reasonCode, index) => {
      if (
        !PAPER_NATIVE_V6_FAILURE_REASON_CODES.has(reasonCode)
        || representedReasonCodes.has(reasonCode)
      ) return;
      representedReasonCodes.add(reasonCode);
      const summaryProjection = summaryReasonCode === reasonCode
        ? paperNativeV6FailureSummaryProjection(
          failureSummary?.wrapper_atomic_screen_projection,
        )
        : paperNativeV6WrapperScreenProjection(null);
      records.push({
        key: `paper-v6-packet-reason-${index}`,
        targetSymbol: "표적 미기록",
        reasonCode,
        supportCandidateCount: summaryReasonCode === reasonCode
          ? boundedPaperNativeV6Count(failureSummary?.support_candidate_count)
          : null,
        minimumVerifiedCandidateCount: summaryReasonCode === reasonCode
          ? boundedPaperNativeV6Count(
            failureSummary?.minimum_verified_candidate_count,
            { minimum: 1 },
          )
          : null,
        rejectionCount: summaryReasonCode === reasonCode
          ? boundedPaperNativeV6Count(failureSummary?.rejection_count)
          : null,
        evidenceGrade: summaryReasonCode === reasonCode && ["B", "C"].includes(failureSummary?.evidence_grade)
          ? failureSummary.evidence_grade
          : null,
        promotionEligible: summaryReasonCode === reasonCode && failureSummary?.promotion_eligible === false
          ? false
          : null,
        projection: summaryProjection,
        source: "packet_reason_only",
      });
    });
    return {
      records: records.slice(0, 12),
      totalCount: records.length,
      omittedCount: Math.max(0, records.length - 12),
    };
  }

  function paperNativeV6FailureHeadline(packet = {}) {
    const { records } = paperNativeV6FailureRecords(packet);
    const record = records[0];
    if (!record) return "";
    const parts = [];
    if (record.evidenceGrade) parts.push(`근거 등급 ${record.evidenceGrade}`);
    if (record.supportCandidateCount != null && record.minimumVerifiedCandidateCount != null) {
      parts.push(`검증 ${record.supportCandidateCount}/${record.minimumVerifiedCandidateCount}건`);
    }
    const reasonEntries = record.projection?.status === "valid"
      ? Object.entries(record.projection.reasonCounts || {}).sort((left, right) => right[1] - left[1])
      : [];
    if (reasonEntries.length) {
      const [reason, count] = reasonEntries[0];
      parts.push(`주 사유: ${paperNativeV6WrapperReasonLabel(reason)} ${count}건`);
    } else {
      parts.push(executionReasonLabel(record.reasonCode));
    }
    return parts.join(" · ");
  }

  function paperNativeV6FailureDetailHtml(packet = {}) {
    const { records, totalCount, omittedCount } = paperNativeV6FailureRecords(packet);
    if (!records.length) return "";
    const recordHtml = records.map((record) => {
      const metrics = [
        record.supportCandidateCount != null
          ? `검증 원저 후보 ${record.supportCandidateCount}건`
          : "",
        record.minimumVerifiedCandidateCount != null
          ? `최소 기준 ${record.minimumVerifiedCandidateCount}건`
          : "",
        record.rejectionCount != null ? `탈락 기록 ${record.rejectionCount}건` : "",
        record.evidenceGrade ? `근거 등급 ${record.evidenceGrade}` : "",
        record.promotionEligible === false ? "승격 불가" : "",
      ].filter(Boolean);
      let projectionHtml = "";
      if (record.projection.status === "valid") {
        const rows = record.projection.rows.map((row) => {
          const statusLabel = row.status === "verified"
            ? "검증 통과"
            : row.status === "not_returned"
              ? "미반환"
              : "검증 탈락";
          const reasonLabel = row.reasonCode
            ? paperNativeV6WrapperReasonLabel(row.sourceReasonCode || row.reasonCode)
            : "원저 결속 검증 통과";
          const reasonCodes = [row.sourceReasonCode, row.reasonCode].filter(Boolean);
          return `<tr><td>${row.rank}</td><td><code>${esc(row.candidateId)}</code></td><td>${esc(statusLabel)}</td><td>${esc(reasonLabel)}${reasonCodes.length ? `<small>${reasonCodes.map((value) => `<code>${esc(value)}</code>`).join(" · ")}</small>` : ""}</td></tr>`;
        }).join("");
        projectionHtml = `<div class="report-table-wrap"><table class="report-table"><caption>고정 후보 풀의 후보별 원저 결속 스크린</caption><thead><tr><th scope="col">순위</th><th scope="col">ChEMBL 후보</th><th scope="col">상태</th><th scope="col">사유</th></tr></thead><tbody>${rows}</tbody></table></div>
          <p class="report-footnote">저장 rows SHA-256 <code>${esc(record.projection.rowsSha256)}</code> · attestation SHA-256 <code>${esc(record.projection.attestationSha256)}</code> · 브라우저는 저장된 해시를 표시하며 원문을 재계산하지 않습니다.</p>`;
      } else if (record.projection.status === "malformed") {
        projectionHtml = '<div class="report-empty is-warning">저장된 후보별 스크린 projection의 형식이 불완전하거나 범위를 벗어나 행 상세를 표시하지 않습니다. 전체 원본은 기술 데이터에서 확인하십시오.</div>';
      } else {
        projectionHtml = `<div class="report-empty">${record.source === "packet_reason_only" ? "이전 실행 또는 요약 packet에는" : "이 실패 기록에는"} 구조화된 후보별 스크린 projection이 없습니다. 추정 행은 만들지 않습니다.</div>`;
      }
      return `<article class="artifact-card soft"><header><div><span>V6 PRIMARY-EVIDENCE FAILURE</span><h4>${esc(record.targetSymbol)} · ${esc(executionReasonLabel(record.reasonCode))}</h4></div><code>${esc(record.reasonCode)}</code></header>
        <p>${metrics.length ? esc(metrics.join(" · ")) : "구조화된 개수·등급은 이 기록에 없습니다."}</p>${projectionHtml}</article>`;
    }).join("");
    const omissionNotice = omittedCount
      ? `<div class="report-empty is-warning">전체 ${compactNumber(totalCount)}건 중 앞의 ${compactNumber(records.length)}건만 표시합니다. 생략 ${compactNumber(omittedCount)}건의 원본은 기술 데이터에 보존됩니다.</div>`
      : "";
    return `<section class="evidence-grade-legend"><b>근거 등급</b><span><strong>A</strong> 후보별 정확한 원저 3건 이상 · 승격 가능</span><span><strong>B</strong> 정확한 원저 1–2건 · HOLD</span><span><strong>C</strong> 정확한 원저 0건 · 활성 DB 참고만 가능</span></section><details class="artifact-methodology" data-disclosure="${esc(state.snapshot?.run_id || "run")}:paper-native-v6-failure-detail"><summary>후보 원저 감사 상세 <span>${compactNumber(records.length)}/${compactNumber(totalCount)}건</span></summary>${omissionNotice}<div class="overview-grid">${recordHtml}</div></details>`;
  }

  function executionGroupRows(items = [], options = {}) {
    const groups = new Map();
    items.forEach((item) => {
      const key = String(item.name || "미기록");
      const presentation = options.ma2Tool ? ma2ExecutionToolPresentation(key) : { engine: key, purpose: "" };
      if (!groups.has(key)) groups.set(key, {
        name: key,
        displayName: presentation.engine,
        purpose: presentation.purpose,
        total: 0,
        used: 0,
        success: 0,
        failed: 0,
        skipped: 0,
        neutral: 0,
        noResults: 0,
        records: 0,
        statuses: new Set(),
        operations: new Set(),
        reasonCodes: new Set(),
        capabilityDetails: new Set(),
        backendNames: new Set(),
      });
      const group = groups.get(key);
      const status = String(item.status || "unknown").toLowerCase();
      group.total += 1;
      group.records += Number(item.records_observed || 0);
      group.statuses.add(traceStatusLabelKo(status));
      if (item.operation) group.operations.add(item.operation);
      (item?.metadata?.reason_codes || []).forEach((reason) => group.reasonCodes.add(reason));
      (item?.metadata?.capability_details || []).forEach((detail) => group.capabilityDetails.add(detail));
      (item?.metadata?.backend_names || []).forEach((backend) => group.backendNames.add(backend));
      if (isSkippedTraceStatus(status)) group.skipped += 1;
      else {
        group.used += 1;
        if (isFailedTraceStatus(status)) group.failed += 1;
        else if (["success", "ok", "live", "cache", "cached", "external_cache", "completed", "completed_with_source_issues", "completed_with_quarantine", "empty"].includes(status)) {
          group.success += 1;
          if (status === "empty") group.noResults += 1;
        } else group.neutral += 1;
      }
    });
    return [...groups.values()].sort((a, b) => b.used - a.used || b.total - a.total || a.name.localeCompare(b.name));
  }

  function actualExecutionResourcesHtml(snapshot = state.snapshot || {}, options = {}) {
    const execution = recordedExecutionTrace(snapshot);
    const allSourceRows = executionGroupRows(execution.publicCalls);
    const computationCalls = execution.scientificCalls.filter((call) => ma2ExecutionToolPresentation(call.name).category === "scientific_computation");
    const pipelineCalls = execution.scientificCalls.filter((call) => ma2ExecutionToolPresentation(call.name).category !== "scientific_computation");
    const allComputationRows = executionGroupRows(computationCalls, { ma2Tool: true });
    const allPipelineRows = executionGroupRows(pipelineCalls, { ma2Tool: true });
    const publicRecordCount = sum(execution.publicCalls, (call) => Number(call.records_observed || 0));
    const sourceRows = options.compact ? allSourceRows.slice(0, 8) : allSourceRows;
    const computationRows = options.compact ? allComputationRows.slice(0, 8) : allComputationRows;
    const pipelineRows = options.compact ? allPipelineRows.slice(0, 8) : allPipelineRows;
    const table = (rows, label, { showRecords = true } = {}) => rows.length ? `<div class="execution-resource-table-wrap"><table class="execution-resource-table">
      <caption>${esc(label)} 실행 상태</caption><thead><tr><th scope="col">이름</th><th scope="col">실행됨</th><th scope="col">성공</th><th scope="col">실패</th><th scope="col">건너뜀·미설정</th><th scope="col">기타</th>${showRecords ? '<th scope="col">레코드</th>' : ""}<th scope="col">결과 상태</th><th scope="col">실행 근거·사유</th></tr></thead>
      <tbody>${rows.map((row) => {
        const purpose = [row.purpose, [...row.operations].slice(0, 3).join(" · ")].filter(Boolean).join(" · ") || "operation 미기록";
        const reasons = [...row.reasonCodes].map((reason) => `<span><b>${esc(executionReasonLabel(reason))}</b><small>${esc(reason)}</small></span>`);
        const details = [...row.capabilityDetails].map((detail) => `<span>${esc(detail)}</span>`);
        const reasonHtml = [...reasons, ...details].join("") || '<span class="is-muted">별도 사유 없음</span>';
        return `<tr><td><b>${esc(row.displayName || row.name)}</b><small>${esc(purpose)}</small>${row.backendNames.size ? `<small>recorded backend · ${esc([...row.backendNames].join(" · "))}</small>` : ""}</td><td>${compactNumber(row.used)}</td><td>${compactNumber(row.success)}</td><td class="${row.failed ? "is-failed" : ""}">${compactNumber(row.failed)}</td><td>${compactNumber(row.skipped)}</td><td>${compactNumber(row.neutral)}</td>${showRecords ? `<td>${compactNumber(row.records)}</td>` : ""}<td>${[...row.statuses].map((status) => `<span>${esc(status)}</span>`).join("")}</td><td class="execution-reason-cell">${reasonHtml}</td></tr>`;
      }).join("")}</tbody></table></div>` : '<div class="report-empty">이 실행에 저장된 기록이 없습니다.</div>';
    const block = (className, title, summary, rows, label, tableOptions = {}) => `<section class="execution-resource-block ${className}"><header><b>${title}</b><span>${summary}</span></header>${table(rows, label, tableOptions)}</section>`;
    const headings = options.includeHeading === false ? "" : `<header class="execution-section-heading"><div><span>실제 실행 기록</span><h4>실제 데이터 소스·MA2 실행 도구</h4><p>후보 패킷 원본 실행 상태를 우선하고, 계산 엔진과 파이프라인 작업을 구분합니다.</p></div></header>`;
    const compactOverflow = allSourceRows.length > sourceRows.length || allComputationRows.length > computationRows.length || allPipelineRows.length > pipelineRows.length;
    return `<section class="actual-execution-resources ${options.compact ? "is-compact" : ""}">${headings}<div class="execution-resource-stack">
      ${block("is-data-source", "사용된 데이터 소스", `${compactNumber(allSourceRows.length)}종 · 호출 ${compactNumber(execution.publicCalls.length)}회 · 레코드 ${compactNumber(publicRecordCount)}건`, sourceRows, "데이터 소스")}
      <div class="execution-resource-columns">
        ${block("is-scientific-computation", "과학 계산", `${compactNumber(allComputationRows.length)}개 엔진 · 호출 ${compactNumber(computationCalls.length)}회`, computationRows, "과학 계산", { showRecords: false })}
        ${block("is-pipeline-operation", "파이프라인 작업", `${compactNumber(allPipelineRows.length)}개 작업 · 호출 ${compactNumber(pipelineCalls.length)}회`, pipelineRows, "파이프라인 작업", { showRecords: false })}
      </div>
    </div>${options.compact && compactOverflow ? '<p class="execution-compact-note">사용량 상위 8개씩 표시 · 전체 상태는 RUN REPORT에서 확인</p>' : ""}<p class="execution-trace-note">과학 계산 상태는 CandidatePacket의 ToolCallRecord.execution_status와 후보별 chemistry/docking/admet_execution을 우선합니다. Calls API 투영값은 원본에 없는 필드만 보완합니다. SUCCESS에는 결과 0건(NO RESULTS)으로 정상 종료된 데이터 조회가 포함되며, NOT CONFIGURED·SKIPPED·NOT ASSESSED는 실행 성공으로 계산하지 않습니다.</p></section>`;
  }

  function isFailedTraceStatus(status) {
    return ["accounting_unknown", "failed", "error", "timeout", "blocked", "unavailable"].includes(String(status || "").toLowerCase());
  }

  function isSuccessfulTraceStatus(status) {
    return ["success", "ok", "live", "cache", "partial", "fixture", "completed", "completed_with_source_issues", "completed_with_quarantine"].includes(String(status || "").toLowerCase());
  }

  function traceStatusLabel(status) {
    const value = String(status || "unknown").toLowerCase();
    const labels = { accounting_unknown: "ACCOUNTING UNKNOWN", success: "SUCCESS", ok: "SUCCESS", live: "LIVE", cache: "CACHE", cached: "CACHE", external_cache: "CACHE", completed: "DONE", partial: "PARTIAL", empty: "NO RESULTS", fixture: "FIXTURE", failed: "FAILED", error: "ERROR", timeout: "TIMEOUT", skipped: "SKIPPED", not_configured: "NOT CONFIGURED", policy_rejected: "POLICY REJECTED", fallback: "FALLBACK", running: "RUNNING", unknown: "STATUS UNKNOWN" };
    return labels[value] || value.replaceAll("_", " ").toUpperCase();
  }

  function traceStatusLabelKo(status) {
    const value = String(status || "unknown").toLowerCase();
    const labels = {
      accounting_unknown: "사용량 미확인", success: "성공", ok: "성공", live: "실시간", cache: "캐시", cached: "캐시", external_cache: "외부 캐시",
      completed: "완료", partial: "일부 완료", empty: "결과 없음", fixture: "시험용 자료", failed: "실패",
      error: "오류", timeout: "시간 초과", skipped: "건너뜀", not_configured: "미설정", not_assessed: "미평가",
      policy_rejected: "정책상 제외", fallback: "대체 경로", running: "실행 중", unavailable: "사용 불가", unknown: "상태 미기록",
    };
    return labels[value] || value.replaceAll("_", " ");
  }

  function toolDisplayName(name) {
    const labels = {
      target_normalizer: "표적 식별자 정규화",
      repurposing_search: "약물 재창출 후보 검색",
      de_novo_generator: "신규 분자 후보 생성",
      chemistry_validator: "화학 구조 검증",
      docking: "결합 포즈 점수화",
      admet_predictor: "6종 ADMET·독성 예측",
      candidate_judge: "다기준 후보 우선순위화",
    };
    return labels[name] || name || "알 수 없는 도구";
  }

  function durationBetween(startedAt, finishedAt) {
    const start = Date.parse(startedAt || "");
    const finish = Date.parse(finishedAt || "");
    return Number.isFinite(start) && Number.isFinite(finish) ? Math.max(0, finish - start) : null;
  }

  function formatDuration(milliseconds) {
    const value = Number(milliseconds);
    if (!Number.isFinite(value)) return "—";
    if (value < 1000) return `${Math.round(value)} ms`;
    return `${(value / 1000).toFixed(value < 10000 ? 2 : 1)} s`;
  }

  function renderSourceCatalog() {
    const content = $("#sourceCatalogContent");
    const summary = $("#sourceCatalogSummary");
    if (!content || !summary) return;

    if (!state.sourceCatalog.loaded && !state.sourceCatalog.error) {
      summary.innerHTML = `
        <span><b>—</b><small>활성 어댑터</small></span>
        <span><b>—</b><small>연합·필요 시 조회</small></span>
        <span><b>—</b><small>연결·설정 필요</small></span>
        <span><b>—</b><small>평가 전용</small></span>`;
      content.innerHTML = '<div class="source-catalog-loading"><span class="status-dot" aria-hidden="true"></span>소스 레지스트리를 불러오는 중</div>';
      return;
    }

    if (!state.sourceCatalog.loaded) {
      const publicEnabled = Boolean(state.health?.public_evidence_enabled);
      const backend = firstBackendValue(state.health?.data_backend) || state.dataBackend || "fixture";
      summary.innerHTML = `
        <span><b>${publicEnabled ? "ON" : "—"}</b><small>공개 근거 조회</small></span>
        <span><b>—</b><small>소스 목록</small></span>
        <span><b>${backend.includes("fixture") ? "FIXTURE" : "—"}</b><small>후보 화학 계산</small></span>
        <span><b>—</b><small>평가 전용</small></span>`;
      content.innerHTML = `
        <div class="source-catalog-fallback">
          <b>상세 소스 레지스트리를 불러오지 못했습니다.</b>
          <p>현재 health 응답 기준으로 공개 근거 조회는 <strong>${publicEnabled ? "활성" : "확인 불가"}</strong>, 데이터 백엔드는 <strong>${esc(formatBackendLabel(backend))}</strong>입니다.</p>
          <button id="retrySourceCatalog" class="button secondary compact" type="button">다시 조회</button>
        </div>`;
      $("#retrySourceCatalog", content)?.addEventListener("click", loadSourceCatalog);
      return;
    }

    const counts = sourceGroupCounts();
    summary.innerHTML = `
      <span><b>${counts.connected}</b><small>활성 어댑터</small></span>
      <span><b>${counts.ondemand}</b><small>연합·필요 시 조회</small></span>
      <span><b>${counts.configuration}</b><small>연결·설정 필요</small></span>
      <span><b>${state.sourceCatalog.evaluation.length}</b><small>평가 전용</small></span>`;

    const groups = [
      {
        key: "connected",
        title: "직접 조회 어댑터 활성",
        description: "LLM 키 없이 호출하는 경로가 설치됨 · 실제 응답 여부는 실행 provenance에 기록",
      },
      {
        key: "ondemand",
        title: "연합·필요 시 조회",
        description: "공개 엔드포인트 또는 다른 데이터 허브를 통해 선택적으로 조회",
      },
      {
        key: "configuration",
        title: "추가 연결·설정 필요",
        description: "어댑터·라이선스·API 키·로컬 스냅샷 또는 별도 실행 환경 필요",
      },
    ];

    const runtimeHtml = groups.map((group) => renderSourceGroup(
      group,
      state.sourceCatalog.runtime.filter((source) => sourceGroup(source) === group.key),
    )).join("");
    const evaluationHtml = state.sourceCatalog.evaluation.length
      ? renderSourceGroup(
          {
            key: "evaluation",
            title: `평가 전용 데이터셋 · ${state.sourceCatalog.evaluation.length}종`,
            description: "실행 중 탐색 근거와 분리하여 시스템 성능 평가에 사용",
          },
          state.sourceCatalog.evaluation,
        )
      : "";
    content.innerHTML = runtimeHtml + evaluationHtml;
  }

  function renderSourceGroup(group, sources) {
    if (!sources.length) return "";
    return `
      <section class="source-group source-group-${esc(group.key)}">
        <header>
          <div><b>${esc(group.title)}</b><small>${esc(group.description)}</small></div>
          <span>${sources.length}</span>
        </header>
        <div class="source-list">
          ${sources.map(renderSourceItem).join("")}
        </div>
      </section>`;
  }

  function renderSourceItem(source) {
    const group = sourceGroup(source);
    const isPubMed = source.sourceId.toLowerCase() === "pubmed" || source.name.toLowerCase() === "pubmed";
    const pubmedLabel = source.adapterActive ? "NCBI DIRECT · ACTIVE" : "NCBI DIRECT · ON DEMAND";
    const statusLabel = group === "connected"
      ? "어댑터 활성"
      : source.status === "catalog_only" || source.status === "configured_no_adapter"
      ? "어댑터 필요"
      : group === "configuration"
      ? "설정 필요"
      : group === "evaluation"
      ? "평가용"
      : source.access === "federated_api"
      ? "연합 조회"
      : "필요 시";
    const detail = source.requirement
      || source.note
      || (source.status === "catalog_only" ? "현재 워크플로우 어댑터 미연결" : "")
      || (source.status === "configured_no_adapter" ? "설정은 있으나 실행 어댑터 미연결" : "")
      || accessLabel(source.access);
    const safeEndpoint = /^https?:\/\//i.test(source.endpoint) ? esc(source.endpoint) : "";
    const name = safeEndpoint
      ? `<a href="${safeEndpoint}" target="_blank" rel="noreferrer">${esc(source.name)}</a>`
      : `<b>${esc(source.name)}</b>`;
    return `
      <article class="source-item ${isPubMed ? "is-pubmed" : ""}">
        <div class="source-item-title">
          <span>${name}${isPubMed ? `<em class="ncbi-direct ${source.adapterActive ? "is-active" : ""}">${pubmedLabel}</em>` : ""}</span>
          <i class="source-status status-${esc(group)}">${esc(statusLabel)}</i>
        </div>
        <div class="source-item-meta">
          <span>${source.stages.map((stage) => `<em>${esc(stage)}</em>`).join("") || "<em>공통</em>"}</span>
          <small>${esc(detail)}</small>
        </div>
      </article>`;
  }

  function accessLabel(access) {
    const labels = {
      public_api: "공개 API",
      federated_api: "연합 API",
      optional_sdk: "선택 SDK",
      local_dataset: "로컬 데이터셋",
      credential_required: "인증 필요",
      cloud_dataset: "클라우드 데이터셋",
      evaluation_dataset: "평가 데이터셋",
    };
    return labels[access] || "데이터 소스";
  }

  function firstBackendValue(...values) {
    const value = values.find((item) => typeof item === "string" && item.trim());
    return value ? value.trim().toLowerCase() : "";
  }

  function formatBackendLabel(value) {
    return String(value || "fixture").replaceAll("_", " ").replaceAll("-", " ").toUpperCase();
  }

  function normalizeResearchDepth(value) {
    const normalized = String(value || "").trim().toLowerCase().replaceAll("-", "_");
    return Object.hasOwn(RESEARCH_DEPTH_META, normalized) ? normalized : null;
  }

  function selectedResearchDepth() {
    return normalizeResearchDepth($('#researchForm input[name="research_depth"]:checked')?.value) || "standard";
  }

  function updateResearchDepthDescription() {
    const description = $("#researchDepthDescription");
    if (!description) return;
    description.textContent = RESEARCH_DEPTH_META[selectedResearchDepth()].description;
  }

  function researchDepthFromSnapshot(snapshot = {}) {
    // Only the per-run value persisted on the ResearchBrief is authoritative.
    // Planner/source-coverage metadata can describe an effective server policy,
    // but must not relabel a legacy run whose brief predates this selector.
    return normalizeResearchDepth(snapshot?.brief?.research_depth);
  }

  function researchDepthPresentation(value) {
    const normalized = normalizeResearchDepth(value);
    if (normalized) return { value: normalized, ...RESEARCH_DEPTH_META[normalized] };
    return {
      value: null,
      label: "미기록 · 이전 실행",
      pill: "깊이 미기록",
      description: "이전 실행 스냅샷에는 연구 분석 깊이가 기록되지 않았습니다.",
    };
  }

  function directionSettingsPresentation(direction, scope = "all") {
    const broad = ["modulate", "unspecified"].includes(direction);
    const scopes = { all: null, both: ["inhibit", "activate"], inhibit: ["inhibit"], activate: ["activate"] };
    const allowed = broad ? scopes[scope] : null;
    if (broad && !Object.hasOwn(scopes, scope)) throw new Error("허용 방향을 다시 선택하십시오.");
    const retained = broad ? (allowed || ["inhibit", "activate", "modulate"]) : [direction];
    const labels = { inhibit: "억제", activate: "활성화", modulate: "조절" };
    return {
      broad, allowed,
      description: broad
        ? "억제·활성화·조절 근거를 넓게 찾고, 직접 근거가 있는 방향별로 가설을 나눕니다."
        : `${labels[direction] || "지정"} 방향의 직접 개입 근거를 확인합니다.`,
      search: broad
        ? "억제·활성화·조절 및 knockdown·silencing·overexpression 등 관련 개입 표현을 함께 검색합니다. 허용 방향을 좁혀도 반대·맥락 근거를 검색에서 숨기지 않습니다."
        : direction === "activate"
        ? "활성화·agonist·stimulation·overexpression 등 관련 개입 표현을 함께 검색합니다. 기저 과발현 자체는 활성화 개입 근거가 아닙니다."
        : "억제제·차단·knockdown·silencing 등 관련 개입 표현을 함께 검색합니다.",
      summary: `최종 가설: ${retained.map(d => labels[d] || d).join(" · ")} · 직접 1차 근거 필수`,
    };
  }

  function directionSelectionSupported() {
    return state.health?.direction_policy?.allowed_hypothesis_directions === true;
  }

  function updateDirectionSettings() {
    const direction = $("#direction");
    if (!direction) return;
    const scope = $("#directionScope");
    const info = directionSettingsPresentation(direction.value, scope?.value || "all");
    $("#directionDescription").textContent = info.description;
    $("#directionSearchDescription").textContent = info.search;
    $("#directionSummary").textContent = info.summary;
    $("#directionScopeField").hidden = !info.broad;
    scope.disabled = !info.broad || !directionSelectionSupported();
    $("#directionCapabilityNotice").hidden = !info.broad || directionSelectionSupported();
  }

  function directionSelectionPayload(requireServerSupport = true) {
    const info = directionSettingsPresentation($("#direction").value, $("#directionScope")?.value || "all");
    if (!info.allowed) return {};
    if (requireServerSupport && !directionSelectionSupported()) throw new Error("서버가 상세 방향 설정을 지원하는지 확인할 수 없습니다. 연결을 확인한 뒤 다시 시도하십시오.");
    return { allowed_hypothesis_directions: [...info.allowed] };
  }

  function directionConstraintLabel(constraints = {}) {
    const requested = constraints.intervention_direction;
    if (!requested) return "미기록";
    const allowed = constraints.allowed_hypothesis_directions;
    if (Array.isArray(allowed) && allowed.length) return allowed.map(directionLabel).join(" · ");
    return ["modulate", "unspecified"].includes(requested)
      ? "전체 방향 · 별도 제한 미지정" : directionLabel(requested);
  }

  function directionResolutionHtml(hypothesis = {}) {
    const resolution = hypothesis.target?.metadata?.direction_resolution;
    if (!resolution) return '<p class="direction-binding is-muted">방향 분기 상세 미기록 · 저장된 가설 방향만 표시합니다.</p>';
    const evidence = Array.isArray(resolution.evidence_ids) ? resolution.evidence_ids : [];
    const resolved = resolution.resolved_direction;
    return `<div class="direction-binding"><b>요청 ${esc(directionLabel(resolution.requested_direction))} → ${resolved ? `근거 방향 ${esc(directionLabel(resolved))}` : "방향 미확정"}</b><small>${resolved ? "방향 연결" : "연결 가능한 방향"} 근거 ${evidence.length}건 · G0 심사와 별도</small>${evidence.length ? `<small>근거 ID: ${evidence.map(esc).join(" · ")}</small>` : ""}</div>`;
  }

  function buildCreateRequest({ requireDirectionSupport = true } = {}) {
    const priorities = commaSeparatedValues($("#priorities").value);
    return {
      research_depth: selectedResearchDepth(),
      research_goal: $("#researchGoal").value.trim(),
      disease: $("#disease").value.trim(),
      target_cell_type: $("#cellType").value.trim() || null,
      intervention_direction: $("#direction").value,
      ...directionSelectionPayload(requireDirectionSupport),
      chronic_safety_required: $("#chronicSafety").checked,
      druggability_required: $("#druggability").checked,
      discovery_priority: $("#discoveryPriority").value,
      forbidden_drugs: commaSeparatedValues($("#forbiddenDrugs").value),
      forbidden_drug_ids: commaSeparatedValues($("#forbiddenDrugIds").value),
      forbidden_drug_classes: commaSeparatedValues($("#forbiddenDrugClasses").value),
      candidates_per_route: clamp(Number($("#candidateCount").value), 1, 100),
      max_ma1_iterations: 3,
      max_rework_loops: 2,
      human_priorities: priorities,
      fault_injection: $("#faultInjection").value,
    };
  }

  function webCryptoUuid() {
    const cryptoApi = globalThis.crypto;
    if (typeof cryptoApi?.randomUUID === "function") return cryptoApi.randomUUID();
    if (typeof cryptoApi?.getRandomValues !== "function") {
      throw new Error("안전한 실행 식별자를 생성할 수 없는 브라우저입니다.");
    }
    // getRandomValues remains available for LAN HTTP origins where
    // randomUUID may be withheld because the page is not a secure context.
    const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map((value) => value.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
  }

  const pendingSubmissionStorageKey = "drugpilot.pending-run-submissions.v1";
  const pendingSubmissionLimit = 8;
  const pendingSubmissionBodyLimit = 65536;

  function readPendingSubmissions() {
    try {
      const parsed = JSON.parse(sessionStorage.getItem(pendingSubmissionStorageKey) || "[]");
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((item) => (
        typeof item?.body === "string"
        && item.body.length <= pendingSubmissionBodyLimit
        && typeof item?.idempotencyKey === "string"
        && /^[A-Za-z0-9._~-]{16,128}$/.test(item.idempotencyKey)
        && Number.isFinite(item?.createdAt)
      )).slice(-pendingSubmissionLimit);
    } catch (_error) {
      return [];
    }
  }

  function writePendingSubmissions(items) {
    try {
      sessionStorage.setItem(
        pendingSubmissionStorageKey,
        JSON.stringify(items.slice(-pendingSubmissionLimit)),
      );
    } catch (_error) {
      // Storage can be disabled or full; in-memory retry safety still applies.
    }
  }

  function createSubmissionForBody(body) {
    if (state.createSubmission?.body === body) return state.createSubmission;
    const stored = readPendingSubmissions().find((item) => item.body === body);
    state.createSubmission = Object.freeze({
      body,
      idempotencyKey: stored?.idempotencyKey || webCryptoUuid(),
      createdAt: stored?.createdAt || Date.now(),
    });
    if (!stored) {
      writePendingSubmissions([...readPendingSubmissions(), state.createSubmission]);
    }
    return state.createSubmission;
  }

  function forgetCreateSubmission(submission) {
    if (!submission) return;
    writePendingSubmissions(readPendingSubmissions().filter((item) => !(
      item.body === submission.body
      && item.idempotencyKey === submission.idempotencyKey
    )));
    if (state.createSubmission?.idempotencyKey === submission.idempotencyKey) {
      state.createSubmission = null;
    }
  }

  function commaSeparatedValues(value) {
    return [...new Set(String(value || "").split(",").map((item) => item.trim()).filter(Boolean))];
  }

  function validateResearchForm() {
    const form = $("#researchForm");
    const fields = [$("#disease"), $("#researchGoal"), $("#candidateCount")];
    fields.forEach((field) => field.removeAttribute("aria-invalid"));
    if (!form.checkValidity()) {
      const invalid = fields.find((field) => !field.checkValidity());
      if (invalid) {
        invalid.setAttribute("aria-invalid", "true");
        invalid.focus();
      }
      showFormError("필수 입력값과 후보 수 범위를 확인하십시오.");
      return false;
    }
    if ($("#researchGoal").value.trim().length < 10) {
      $("#researchGoal").setAttribute("aria-invalid", "true");
      showFormError("연구 목표를 10자 이상 입력하십시오.");
      return false;
    }
    showFormError("");
    return true;
  }

  async function createRun(event) {
    event.preventDefault();
    if (!validateResearchForm()) return;
    const button = $("#runButton");
    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="button-icon">●</span><span>실행 생성 중…</span>';
    let submission = null;
    try {
      const requestBody = JSON.stringify(buildCreateRequest());
      stopLiveUpdates();
      state.preview = false;
      submission = createSubmissionForBody(requestBody);
      const payload = await api("/api/runs", {
        method: "POST",
        headers: { "Idempotency-Key": submission.idempotencyKey },
        body: submission.body,
      });
      const snapshot = unwrapSnapshot(payload);
      const runId = snapshot?.run_id || payload?.run_id || payload?.id;
      if (!runId) throw new Error("서버 응답에 실행 ID가 없습니다.");
      syncRunDeepLink(runId);
      state.snapshot = snapshot?.brief ? snapshot : null;
      closeContextPanel();
      state.eventIds.clear();
      state.openDisclosures.clear();
      state.callTrace = null;
      state.resumeStatus = null;
      state.resumeStatusRunId = null;
      showRunWorkspace();
      closeSidebar();
      if (state.snapshot) renderSnapshot();
      else await fetchRun(runId, { quiet: true });
      startLiveUpdates(runId);
      await loadRuns({ quiet: true });
      showToast(
        "연구 실행을 시작했습니다",
        `LLM ${state.llmBackend.toUpperCase()} · DATA ${formatBackendLabel(state.dataBackend)} · ${shortId(runId)}`,
        "success",
      );
      forgetCreateSubmission(submission);
    } catch (error) {
      if (
        Number.isInteger(error.status)
        && error.status >= 400
        && error.status < 500
        && ![408, 409, 429].includes(error.status)
      ) {
        forgetCreateSubmission(submission);
      }
      showFormError(error.message || "연구 실행을 생성하지 못했습니다.");
      showToast("실행 생성 실패", error.message, "error");
    } finally {
      button.disabled = false;
      button.innerHTML = original;
    }
  }

  async function loadRuns(options = {}) {
    const list = $("#runList");
    if (!options.quiet) list.innerHTML = '<div class="list-skeleton"><span></span><span></span><span></span></div>';
    try {
      const payload = await api("/api/run-summaries");
      state.runs = normalizeRunList(payload);
      renderRunList();
    } catch (error) {
      if (!options.quiet) list.innerHTML = '<div class="run-list-empty">실행 목록을 불러오지 못했습니다.</div>';
    }
  }

  function normalizeRunList(payload) {
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload?.runs)) return payload.runs;
    if (Array.isArray(payload?.items)) return payload.items;
    if (Array.isArray(payload?.data)) return payload.data;
    return [];
  }

  function renderRunList() {
    const list = $("#runList");
    if (!state.runs.length) {
      list.innerHTML = '<div class="run-list-empty">아직 저장된 실행이 없습니다.</div>';
      return;
    }
    list.innerHTML = state.runs.slice(0, 12).map((item) => {
      const listedRun = unwrapSnapshot(item) || item;
      const listedId = listedRun.run_id || listedRun.id || "";
      const run = state.snapshot?.run_id === listedId ? state.snapshot : listedRun;
      const id = run.run_id || run.id || "";
      const stage = String(run.stage || run.status || "created").toLowerCase();
      const disease = run.brief?.disease || run.disease || "제목 없는 연구";
      const mode = String(run.mode || "mock").toUpperCase();
      const modeLabel = ["MOCK", "LIVE", "HYBRID"].includes(mode) ? `LLM ${mode}` : mode;
      const depth = researchDepthPresentation(researchDepthFromSnapshot(run));
      const active = state.snapshot?.run_id === id ? " is-active" : "";
      const intake = getIntakeAssessment(run);
      const statusPresentation = runStatusPresentation(run);
      const runStageLabel = intake.diseaseNeedsReview
        ? "질환 확인 필요"
        : intake.verificationUnavailable
        ? "외부 질환 검증 불가"
        : intake.abstained && intake.hasMismatch
        ? "입력 불일치 확인"
        : runListResultLabel(run, statusPresentation.label);
      return `
        <button class="run-list-item${active}" type="button" data-run-id="${esc(id)}">
          <span class="run-list-state ${esc(stage)}"></span>
          <span class="run-list-copy"><b>${esc(disease)}</b><small>${esc(runStageLabel)} · ${esc(shortId(id))}</small></span>
          <span class="run-list-meta">${esc(modeLabel)}<br>${esc(depth.value ? depth.label.toUpperCase() : "DEPTH —")} · ${esc(relativeTime(run.updated_at || run.created_at))}</span>
        </button>`;
    }).join("");
    $$(".run-list-item", list).forEach((button) => {
      button.addEventListener("click", () => selectRun(button.dataset.runId));
    });
  }

  async function selectRun(runId) {
    if (!runId || state.loadingRun) return;
    state.loadingRun = true;
    state.preview = false;
    state.openDisclosures.clear();
    state.callTrace = null;
    state.resumeStatus = null;
    state.resumeStatusRunId = null;
    state.cancelStatus = null;
    state.cancelStatusRunId = null;
    state.resultReadiness = null;
    state.resultReadinessKey = null;
    state.resultReadinessError = null;
    $("#researchSelectionDialog").close();
    closeContextPanel();
    stopLiveUpdates();
    showRunWorkspace();
    setLoadingSnapshot(runId);
    try {
      await fetchRun(runId);
      syncRunDeepLink(runId);
      await loadCallTrace(runId, { quiet: true });
      if (state.snapshot && (!LIVE_END_STAGES.has(state.snapshot.stage) || reviewIsRunning())) startLiveUpdates(runId);
      renderRunList();
      closeSidebar();
    } catch (error) {
      showToast("실행 불러오기 실패", error.message, "error");
    } finally {
      state.loadingRun = false;
    }
  }

  async function fetchRun(runId, options = {}) {
    try {
      const payload = await api(`/api/runs/${encodeURIComponent(runId)}`);
      const snapshot = unwrapSnapshot(payload);
      if (!snapshot?.run_id) throw new Error("실행 정보를 해석할 수 없습니다.");
      state.snapshot = snapshot;
      indexEvents(snapshot.events || []);
      renderSnapshot();
      loadResumeStatus(runId, { quiet: true });
      void loadCancelStatus(runId);
      return snapshot;
    } catch (error) {
      if (!options.quiet) throw error;
      showToast("실행 상태 확인 실패", error.message, "error");
      return null;
    }
  }

  function startLiveUpdates(runId) {
    stopLiveUpdates();
    if (!runId || runId === "preview") return;
    const indicator = $("#liveIndicator");
    indicator.className = "live-indicator";
    indicator.innerHTML = "<i></i> LIVE";

    if ("EventSource" in window) {
      const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
      state.eventSource = source;
      source.onopen = () => {
        indicator.className = "live-indicator";
        indicator.innerHTML = "<i></i> LIVE";
      };
      source.onmessage = (event) => handleServerEvent(event, runId);
      // EventBroker emits named ``workflow`` events.  Named SSE events do not
      // reach ``onmessage``, so subscribe explicitly and retain the aliases
      // for compatibility with older/local backends.
      ["workflow", "execution_event", "event", "snapshot", "run_update"].forEach((name) => {
        source.addEventListener(name, (event) => handleServerEvent(event, runId));
      });
      source.addEventListener("closed", () => fetchRun(runId, { quiet: true }));
      source.onerror = () => {
        indicator.className = "live-indicator is-paused";
        indicator.innerHTML = "<i></i> POLLING";
        if (!state.pollTimer) state.pollTimer = window.setInterval(() => fetchRun(runId, { quiet: true }), 1800);
      };
    } else {
      indicator.className = "live-indicator is-paused";
      indicator.innerHTML = "<i></i> POLLING";
      state.pollTimer = window.setInterval(() => fetchRun(runId, { quiet: true }), 1800);
    }
  }

  function stopLiveUpdates() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    if (state.pollTimer) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    if (state.refreshTimer) {
      window.clearTimeout(state.refreshTimer);
      state.refreshTimer = null;
    }
  }

  function handleServerEvent(message, runId) {
    if (!message?.data) return;
    let payload;
    try { payload = JSON.parse(message.data); }
    catch { payload = { message: message.data }; }
    if (payload?.type === "ping" || payload?.event_type === "ping") return;

    const snapshot = unwrapSnapshot(payload);
    if (snapshot?.brief && snapshot?.run_id) {
      state.snapshot = snapshot;
      indexEvents(snapshot.events || []);
      renderSnapshot();
      return;
    }

    const event = payload?.event || payload;
    if (event?.event_id && !state.eventIds.has(event.event_id) && state.snapshot) {
      state.eventIds.add(event.event_id);
      state.snapshot.events = [...(state.snapshot.events || []), event];
      if (event.stage) state.snapshot.stage = event.stage;
      if (event.message) state.snapshot.status_message = event.message;
      renderActivity();
      renderPipeline();
    }

    if (state.refreshTimer) window.clearTimeout(state.refreshTimer);
    state.refreshTimer = window.setTimeout(() => fetchRun(runId, { quiet: true }), 180);
  }

  function unwrapSnapshot(payload) {
    if (!payload || typeof payload !== "object") return null;
    if (payload.run_id && (payload.brief || payload.stage)) return payload;
    for (const key of ["snapshot", "run", "data", "result"]) {
      if (payload[key]?.run_id) return payload[key];
    }
    return payload.run_id ? payload : null;
  }

  function renderSnapshot() {
    const snapshot = state.snapshot;
    if (!snapshot) return;
    if (state.selectedArtifactRunId !== snapshot.run_id) {
      state.selectedArtifactRunId = snapshot.run_id;
      state.selectedTab = "overview";
    }
    if (state.selectedGateRunId !== snapshot.run_id) {
      state.selectedGateRunId = snapshot.run_id;
      state.selectedGate = snapshot.g1_verdict ? "g1" : "g0";
      state.selectedGateUserSet = false;
    } else if (!state.selectedGateUserSet && snapshot.g1_verdict) {
      state.selectedGate = "g1";
    }
    updateSelectedTabs($$(".gate-toggle [role=tab]"), "gate", state.selectedGate);
    showRunWorkspace();
    updateModeBadge(String(snapshot.mode || state.runMode || "mock").toLowerCase());
    $("#runIdText").textContent = shortId(snapshot.run_id);
    $("#runDisease").textContent = snapshot.brief?.disease || "연구 실행";
    $("#runGoal").textContent = snapshot.brief?.research_goal || snapshot.status_message || "";
    const depth = researchDepthPresentation(researchDepthFromSnapshot(snapshot));
    const depthPill = $("#runDepthPill");
    depthPill.className = `research-depth-pill is-${depth.value || "unrecorded"}`;
    depthPill.textContent = depth.pill;
    depthPill.title = `RESEARCH DEPTH · ${depth.label}. ${depth.description}`;
    updateRunStatus(snapshot);
    renderRunList();
    renderIntakeNotice(snapshot);
    renderPipeline();
    renderDiscoverySummary();
    renderResearchResults();
    void loadResultReadiness(snapshot);
    renderNextActions();
    loadReviewEligibility(snapshot);
    renderMetrics();
    renderActivity();
    renderVerdict();
    renderCounts();
    renderArtifact();
    configureHumanGate();
    if (state.callTrace?.run_id !== snapshot.run_id || state.callTraceRevision !== callTraceRevision(snapshot)) {
      void loadCallTrace(snapshot.run_id, { quiet: true });
    }
    if (state.contextSelection && !$("#contextPanel")?.hidden) {
      openContextPanel(state.contextSelection.kind, state.contextSelection.id, { focus: false });
    }
    if (LIVE_END_STAGES.has(snapshot.stage) && !reviewIsRunning()) {
      stopLiveUpdates();
      const indicator = $("#liveIndicator");
      indicator.className = "live-indicator is-paused";
      indicator.innerHTML = snapshot.stage === "human_gate" ? "<i></i> READY" : "<i></i> ENDED";
    }
  }

  function updateRunStatus(snapshot) {
    const stage = String(snapshot.stage || "created").toLowerCase();
    const intake = getIntakeAssessment(snapshot);
    const pill = $("#runStatusPill");
    let cls = RUNNING_STAGES.has(stage) ? "running" : stage;
    let label = RUNNING_STAGES.has(stage) ? "실행 중" : runStatusPresentation(snapshot).label;
    if (stage === "human_gate") {
      cls = "human_gate";
      label = snapshot.next_action_packet ? "추가 조치 검토" : "연구자 결정 대기";
      if (currentAutomaticResult(snapshot) && !snapshot.human_decision) label = runStatusPresentation(snapshot).label;
    }
    if (snapshot.human_decision?.decision === "HOLD") { cls = "hold"; label = "연구자 판단 보류"; }
    if (intake.abstained && intake.hasMismatch) { cls = "abstained"; label = "입력 불일치 확인"; }
    if (intake.verificationUnavailable) { cls = "abstained"; label = "외부 질환 검증 불가"; }
    if (intake.diseaseNeedsReview) { cls = "abstained"; label = "질환 확인 필요"; }
    if (["cancel_requested", "cancelled"].includes(stage)) { cls = stage; label = runStatusPresentation(snapshot).label; }
    if (["failed", "blocked", "abstained"].includes(stage)) { cls = stage; label = runStatusPresentation(snapshot).label; }
    const result = resultPresentation(snapshot, currentResultReadiness(snapshot));
    if (result.complete) { cls = "completed"; label = result.title; }
    pill.className = `status-pill ${cls}`;
    pill.innerHTML = `<span class="status-dot"></span>${esc(label)}`;
    pill.title = runStatusPresentation(snapshot).detail;
    renderResumeControl(snapshot);
    renderCancelControl(snapshot);
  }

  function renderCancelControl(snapshot = state.snapshot) {
    const button = $("#cancelRunButton");
    const notice = $("#cancelRunNotice");
    if (!button || !notice) return;
    const pending = snapshot?.stage === "cancel_requested" || state.cancellingRunId === snapshot?.run_id;
    const eligible = state.cancelStatusRunId === snapshot?.run_id && state.cancelStatus?.eligible === true;
    button.hidden = state.preview || (!eligible && !pending);
    button.disabled = pending;
    button.textContent = pending ? "중단 처리 중" : "실행 중단";
    notice.hidden = !pending && snapshot?.stage !== "cancelled";
    notice.textContent = pending
      ? "진행 중 호출의 사용량을 기록한 뒤 종료합니다. 추가 단계는 시작하지 않습니다."
      : snapshot?.stage === "cancelled" ? "사용자 요청으로 중단했습니다. 기존 결과와 사용량은 보존됩니다." : "";
  }

  async function loadCancelStatus(runId) {
    if (!runId || runId === "preview" || state.cancelStatusLoading) return;
    state.cancelStatusLoading = true;
    try {
      const status = await api(`/api/runs/${encodeURIComponent(runId)}/cancel-status`);
      if (state.snapshot?.run_id === runId) {
        state.cancelStatus = status;
        state.cancelStatusRunId = runId;
        renderCancelControl();
      }
    } catch {
      if (state.snapshot?.run_id === runId) {
        state.cancelStatus = null;
        state.cancelStatusRunId = runId;
        renderCancelControl();
      }
    } finally { state.cancelStatusLoading = false; }
  }

  async function cancelRun() {
    const runId = state.snapshot?.run_id;
    if (!runId || state.preview || state.cancellingRunId || state.cancelStatus?.eligible !== true || state.cancelStatusRunId !== runId) return;
    state.cancellingRunId = runId;
    renderCancelControl();
    try {
      const snapshot = await api(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
      if (state.snapshot?.run_id === runId) {
        state.snapshot = unwrapSnapshot(snapshot);
        renderSnapshot();
        if (!LIVE_END_STAGES.has(state.snapshot?.stage)) startLiveUpdates(runId);
      }
    } catch (error) {
      showToast("중단 상태 확인 필요", error.message, "error");
      // A failed response can still follow a persisted cancellation. Read it
      // back instead of automatically repeating a mutation.
      if (state.snapshot?.run_id === runId) await fetchRun(runId, { quiet: true });
    } finally {
      state.cancellingRunId = null;
      if (state.snapshot?.run_id === runId) await loadCancelStatus(runId);
      renderCancelControl();
    }
  }

  function renderResumeControl(snapshot = state.snapshot) {
    const button = $("#resumeButton");
    if (!button) return;
    const status = state.resumeStatusRunId === snapshot?.run_id
      ? state.resumeStatus
      : null;
    const eligible = Boolean(status?.eligible) && status?.active_in_process !== true;
    button.hidden = !eligible || state.preview;
    button.textContent = eligible ? "체크포인트에서 재개" : "재개 불가";
    button.title = eligible
      ? `${status.checkpoint_completed_node || "Research Brief"} 완료 · ${status.resume_from_node || "다음 단계"}부터 재개`
      : String(status?.reason || "검증된 중단 체크포인트가 있을 때만 재개할 수 있습니다.");
    button.dataset.resumeFrom = eligible ? String(status.resume_from_node || "") : "";
  }

  async function loadResumeStatus(runId, options = {}) {
    if (!runId || runId === "preview") return null;
    if (state.resumeStatusLoading && state.resumeStatusRunId === runId) {
      return state.resumeStatus;
    }
    state.resumeStatusLoading = true;
    try {
      const status = await api(`/api/runs/${encodeURIComponent(runId)}/resume-status`);
      if (state.snapshot?.run_id === runId) {
        state.resumeStatus = status;
        state.resumeStatusRunId = runId;
        renderResumeControl();
      }
      return status;
    } catch (error) {
      if (state.snapshot?.run_id === runId) {
        state.resumeStatus = null;
        state.resumeStatusRunId = runId;
        renderResumeControl();
      }
      if (!options.quiet) showToast("재개 상태 확인 실패", error.message, "error");
      return null;
    } finally {
      state.resumeStatusLoading = false;
    }
  }

  function getIntakeAssessment(snapshot = {}) {
    const packet = snapshot?.hypothesis_packet || null;
    const ma1Meta = packet?.metadata || {};
    const eventMetadata = [...(snapshot?.events || [])]
      .reverse()
      .map((event) => event?.metadata || {})
      .find((metadata) => metadata.intake_validation || metadata.intake_assessment || metadata.research_brief_assessment);
    const raw = ma1Meta.intake_validation
      || snapshot?.intake_validation
      || snapshot?.metadata?.intake_validation
      || eventMetadata?.intake_validation
      || eventMetadata?.intake_assessment
      || eventMetadata?.research_brief_assessment
      || {};
    const verificationStatus = String(
      raw.external_verification_status || raw.verification_status || ""
    ).toLowerCase();
    const reasonCodes = firstArray(raw.reason_codes, raw.codes, raw.reasons).map(String);
    const fieldMismatches = normalizeIntakeIssues(
      raw.field_mismatches || raw.cross_field_mismatches || raw.mismatches,
      "FIELD_MISMATCH",
    );
    const warnings = normalizeIntakeIssues(raw.warnings, "INTAKE_WARNING");
    const statusText = [
      snapshot?.status_message,
      snapshot?.error,
      raw.status,
      raw.explanation,
      ...reasonCodes,
    ].filter(Boolean).join(" ").toLowerCase();
    const resolvedDisease = ma1Meta.resolved_disease || null;
    const explicitUnresolved = raw.disease_resolved === false
      || verificationStatus === "unresolved"
      || reasonCodes.some((code) => /(?:invalid|unknown|unresolved).*disease|disease.*(?:invalid|unknown|unresolved)/i.test(code));
    const legacyUnresolved = Boolean(
      packet
      && ma1Meta.target_discovery_enabled !== false
      && !resolvedDisease?.disease_id
      && Number(ma1Meta.external_target_seed_count || 0) === 0
      && ma1Meta.target_fallback_used === true
    );
    const diseaseTextSignal = /(?:invalid|unknown|unresolved)[ _-]?disease|disease[^.]{0,40}(?:not resolved|unresolved|invalid)|질환[^.]{0,30}(?:확인|해석|식별)[^.]*(?:실패|불가|필요)/i.test(statusText);
    const targetSourceStatuses = ma1Meta.target_source_statuses || [];
    const allDiscoverySourcesUnavailable = targetSourceStatuses.length > 0
      && targetSourceStatuses.every((source) => /failed|error|timeout|unavailable/.test(String(source?.status || "").toLowerCase()));
    const verificationUnavailable = verificationStatus === "source_unavailable"
      || reasonCodes.some((code) => /source.*unavailable|verification.*unavailable/i.test(code))
      || Boolean(ma1Meta.target_discovery_error)
      || allDiscoverySourcesUnavailable;
    const demoFixture = verificationStatus === "demo_fixture";
    const diseaseNeedsReview = !verificationUnavailable && !demoFixture
      && (explicitUnresolved || legacyUnresolved || diseaseTextSignal);
    // Intake status belongs to the intake assessment only. A later G0/G1
    // abstention must not be relabelled as an input-validation failure after
    // the disease guard already passed and candidates were evaluated.
    const abstained = raw.abstain_recommended === true
      || String(raw.status || "").toLowerCase() === "abstain";
    const hasMismatch = fieldMismatches.length > 0;
    const hasWarning = warnings.length > 0;
    return {
      raw,
      abstained,
      diseaseNeedsReview,
      verificationUnavailable,
      demoFixture,
      hasMismatch,
      requiresInputCorrection: diseaseNeedsReview || hasMismatch,
      hasNotice: diseaseNeedsReview || verificationUnavailable || hasMismatch || hasWarning,
      fieldMismatches,
      warnings,
      reasonCodes,
      queryDisease: raw.query_disease || snapshot?.brief?.disease || snapshot?.disease || "",
      suggestedName: raw.suggested_normalized_name || "",
      explanation: raw.explanation || "",
      verificationStatus,
    };
  }

  function normalizeIntakeIssues(value, fallbackCode) {
    const values = Array.isArray(value) ? value : value ? [value] : [];
    const normalized = values.map((item) => {
      if (typeof item === "string") return { code: fallbackCode, message: item, fields: [] };
      if (!item || typeof item !== "object") return null;
      const fields = firstArray(item.fields, item.field_names, item.related_fields).map(String);
      const message = item.message
        || item.explanation
        || item.detail
        || item.summary
        || item.reason
        || (fields.length ? `${fields.map(intakeFieldLabel).join(" · ")} 값이 일치하지 않습니다.` : fallbackCode);
      return { code: String(item.code || fallbackCode), message: String(message), fields };
    }).filter(Boolean);
    const seen = new Set();
    return normalized.filter((item) => {
      const key = `${item.code}|${item.message}|${item.fields.join(",")}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function intakeFieldLabel(field) {
    return ({
      disease: "대상 질환",
      research_goal: "연구 목표",
      target_cell_type: "표적 세포",
      disease_id: "질환 식별자",
    })[String(field || "").toLowerCase()] || String(field || "입력 항목").replaceAll("_", " ");
  }

  function renderIntakeNotice(snapshot) {
    const notice = $("#intakeNotice");
    if (!notice) return;
    const intake = getIntakeAssessment(snapshot);
    notice.hidden = !intake.hasNotice;
    if (!intake.hasNotice) return;

    const items = [...intake.fieldMismatches, ...intake.warnings];
    const list = $("#intakeNoticeItems");
    notice.className = `intake-notice ${intake.diseaseNeedsReview || intake.verificationUnavailable ? "is-abstain" : "is-warning"}`;
    notice.setAttribute("role", intake.abstained ? "alert" : "status");
    $("#intakeNoticeIcon").textContent = intake.verificationUnavailable ? "↻" : "!";

    if (intake.diseaseNeedsReview) {
      $("#intakeNoticeTitle").textContent = "질환 확인 필요";
      $("#intakeNoticeText").textContent = intake.explanation
        || `입력한 질환 “${intake.queryDisease}”을 공인 외부 데이터에서 확인하지 못해 후보 생성을 진행하지 않았습니다.`;
      $("#intakeNoticeAction").textContent = intake.suggestedName
        ? `권장 명칭 “${intake.suggestedName}”을 확인한 뒤 새 연구로 다시 실행하십시오.`
        : "정식 질환명 또는 MONDO/EFO 식별자로 수정한 뒤 새 연구로 다시 실행하십시오.";
    } else if (intake.verificationUnavailable) {
      $("#intakeNoticeTitle").textContent = "외부 질환 검증 불가";
      $("#intakeNoticeText").textContent = intake.explanation
        || "질환 검증 소스에 연결할 수 없어 후보를 만들지 않고 판단을 보류했습니다.";
      $("#intakeNoticeAction").textContent = "외부 데이터 연결 상태를 확인한 뒤 이 입력으로 다시 실행하십시오.";
    } else if (intake.demoFixture) {
      $("#intakeNoticeTitle").textContent = "DEMO FIXTURE · 검증용 실행";
      $("#intakeNoticeText").textContent = intake.explanation
        || "질환이 외부에서 확인되지 않았으며, 명시적으로 허용된 fixture 자료만 사용합니다.";
      $("#intakeNoticeAction").textContent = "표적·근거·후보와 점수를 과학적 연구 결과로 해석하지 마십시오.";
    } else {
      $("#intakeNoticeTitle").textContent = intake.hasMismatch ? "연구 입력 불일치 확인" : "연구 입력 경고";
      $("#intakeNoticeText").textContent = intake.explanation
        || (intake.hasMismatch
          ? "구조화 입력과 연구 목표에서 서로 다른 맥락이 감지되었습니다."
          : "입력 검증 중 확인이 필요한 항목이 감지되었습니다.");
      $("#intakeNoticeAction").textContent = intake.hasMismatch
        ? "대상 질환·연구 목표·표적 세포가 같은 연구 맥락인지 확인하십시오."
        : "아래 경고를 확인한 뒤 결과를 해석하십시오.";
    }

    list.hidden = items.length === 0;
    list.innerHTML = items.map((item) => {
      const fields = item.fields.length ? `<b>${esc(item.fields.map(intakeFieldLabel).join(" · "))}</b>` : "";
      return `<li>${fields}<span>${esc(item.message)}</span></li>`;
    }).join("");
  }

  function renderPipeline() {
    const snapshot = state.snapshot;
    if (!snapshot) return;
    const intake = getIntakeAssessment(snapshot);
    let stage = String(snapshot.stage || "created").toLowerCase();
    const stoppedAfterG0 = g0StoppedBeforeMa2(snapshot);
    const stageIndex = stage === "created" ? 0 : STAGES.indexOf(stage);
    const terminal = TERMINAL_STAGES.has(stage);
    let inferredIndex = stageIndex;
    if (stage === "completed") inferredIndex = STAGES.length;
    if (terminal && stage !== "completed") inferredIndex = inferLastStageIndex(snapshot);

    $$(".stage-node").forEach((node, index) => {
      node.classList.remove("is-complete", "is-active", "is-failed", "is-abstained", "is-skipped");
      if (stoppedAfterG0) {
        if (index <= STAGES.indexOf("g0")) node.classList.add("is-complete");
        if (index >= STAGES.indexOf("planner") && index <= STAGES.indexOf("g1")) node.classList.add("is-skipped");
        if (index === STAGES.indexOf("human_gate")) node.classList.add("is-active");
        return;
      }
      if (index < inferredIndex || stage === "completed") node.classList.add("is-complete");
      if (index === inferredIndex && !terminal) node.classList.add("is-active");
      if (index === inferredIndex && ["blocked", "failed"].includes(stage)) node.classList.add("is-failed");
      if (index === inferredIndex && stage === "abstained") node.classList.add("is-abstained");
    });
    $$(".pipeline-link").forEach((link, index) => {
      link.classList.remove("is-complete", "is-active", "is-skipped");
      if (stoppedAfterG0) {
        if (index < STAGES.indexOf("g0")) link.classList.add("is-complete");
        else link.classList.add("is-skipped");
        return;
      }
      if (index < inferredIndex || stage === "completed") link.classList.add("is-complete");
      else if (index === inferredIndex) link.classList.add("is-active");
    });
    // A later stage is not evidence that every audit or target branch passed.
    for (const [gate, verdict] of [["g0", snapshot.g0_verdict], ["g1", snapshot.g1_verdict]]) {
      const node = $(`.stage-node[data-stage="${gate}"]`);
      if (node && verdict && verdict.verdict !== "PASS") {
        node.classList.remove("is-complete");
        node.classList.add(verdict.verdict === "BLOCK" ? "is-failed" : "is-abstained");
      }
    }
    const ma2Node = $('.stage-node[data-stage="ma2"]');
    if (ma2Node && snapshot.candidate_packet && portfolioExecutionState(snapshot.candidate_packet).abstainRequired) {
      ma2Node.classList.remove("is-complete");
      ma2Node.classList.add("is-abstained");
    }
    $("#pipelineSummary").textContent = intake.diseaseNeedsReview
      ? "질환을 확인하지 못해 입력 단계에서 중단"
      : intake.verificationUnavailable
      ? "외부 질환 검증 불가로 입력 단계에서 중단"
      : intake.abstained && intake.hasMismatch
      ? "연구 입력 불일치로 입력 단계에서 중단"
      : stoppedAfterG0
      ? runStatusPresentation(snapshot).detail
      : stage === "abstained"
      ? runStatusPresentation(snapshot).detail
      : snapshot.status_message ? `원본 실행 기록: ${snapshot.status_message}` : `${stageLabel(stage)} 단계`;

    const rework = snapshot.g1_verdict?.rework_order || snapshot.g0_verdict?.rework_order;
    const ribbon = $("#reworkRibbon");
    if (rework) {
      ribbon.hidden = false;
      $("#reworkTitle").textContent = `${rework.gate || "감사"} · ${verdictLabel((snapshot.g1_verdict || snapshot.g0_verdict)?.verdict)} 판단`;
      const verdict = snapshot.g1_verdict || snapshot.g0_verdict || {};
      const labels = groupFindings(verdict.findings || []).slice(0, 4).map((group) => findingGroupTitle(group));
      $("#reworkText").textContent = `${String(rework.target_stage || "이전 단계").toUpperCase()} 재작업 · ${labels.join(" · ") || `${rework.required_actions?.length || 0}개 보완 조치`}`;
    } else {
      ribbon.hidden = true;
    }
  }

  function g0StoppedBeforeMa2(snapshot = {}) {
    const verdict = String(snapshot?.g0_verdict?.verdict || "").toUpperCase();
    return String(snapshot?.stage || "").toLowerCase() === "human_gate"
      && ["ABSTAIN", "BLOCK", "REVISE", "REPLAN"].includes(verdict)
      && !snapshot?.plan_spec
      && !snapshot?.candidate_packet
      && !snapshot?.g1_verdict;
  }

  function inferLastStageIndex(snapshot) {
    if (snapshot.g1_verdict) return STAGES.indexOf("g1");
    if (snapshot.candidate_packet) return STAGES.indexOf("ma2");
    if (snapshot.plan_spec) return STAGES.indexOf("planner");
    if (snapshot.g0_verdict) return STAGES.indexOf("g0");
    if (snapshot.hypothesis_packet) return STAGES.indexOf("ma1");
    return 0;
  }

  function portfolioExecutionState(packet = {}) {
    const enabled = packet?.portfolio_enabled === true;
    const executions = Array.isArray(packet?.target_executions) ? packet.target_executions : [];
    const count = (field, predicate) => packet?.[field] != null
      ? Number(packet[field])
      : executions.filter(predicate).length;
    const approved = count("g0_approved_target_count", () => true);
    const eligible = count("eligible_target_count", (item) => item.execution_eligible === true);
    const attempted = count("attempted_target_count", (item) => item.attempted === true);
    const completed = count("completed_target_count", (item) => item.status === "completed");
    const noCandidates = executions.filter((item) => String(item.status || "").toLowerCase() === "no_candidates").length;
    const concluded = count("concluded_target_count", (item) => ["completed", "no_candidates"].includes(String(item.status || "").toLowerCase()));
    const ineligible = executions.filter((item) => ["ineligible", "duplicate"].includes(String(item.status || "").toLowerCase())).length;
    const incomplete = executions.filter((item) => ["partial", "unavailable", "failed"].includes(String(item.status || "").toLowerCase())).length;
    const unavailable = count("unavailable_target_count", (item) => ["partial", "unavailable", "failed", "ineligible", "duplicate"].includes(String(item.status || "").toLowerCase()));
    const executionStatus = String(packet?.execution_status || (packet?.candidates ? "completed" : "pending")).toLowerCase();
    const complete = !enabled || (
      executionStatus === "completed"
      && attempted === eligible
      && concluded === eligible
    );
    const abstainRequired = ["partial", "unavailable", "failed"].includes(executionStatus)
      || (enabled && !complete);
    return {
      enabled,
      executions,
      approved,
      eligible,
      attempted,
      completed,
      noCandidates,
      concluded,
      ineligible,
      incomplete,
      unavailable,
      executionStatus,
      complete,
      abstainRequired,
      reasonCodes: Array.isArray(packet?.abstain_reason_codes) ? packet.abstain_reason_codes : [],
    };
  }

  function portfolioCoverageText(packet = {}, { includeApproved = false } = {}) {
    const coverage = portfolioExecutionState(packet);
    if (!coverage.enabled) return "선택 표적 실행";
    const approved = includeApproved ? `G0 승인 ${coverage.approved}개 · ` : "";
    const unclassifiedUnavailable = Math.max(0, coverage.unavailable - coverage.ineligible - coverage.incomplete);
    return `${approved}실행 자격 ${coverage.eligible}개 · 시도 ${coverage.attempted}개 · 결론 ${coverage.concluded}/${coverage.eligible}개 (후보 평가 ${coverage.completed}개${coverage.noCandidates ? ` · 검색 완료·적합 후보 없음 ${coverage.noCandidates}개` : ""})${coverage.ineligible ? ` · 실행 자격 제외 ${coverage.ineligible}개` : ""}${coverage.incomplete ? ` · 실행 미완료 ${coverage.incomplete}개` : ""}${unclassifiedUnavailable ? ` · 실행 제외·상태 미분류 ${unclassifiedUnavailable}개` : ""}`;
  }

  function portfolioCandidateRank(candidate = {}, packet = {}) {
    const portfolio = portfolioExecutionState(packet);
    if (portfolio.enabled) {
      return candidate.portfolio_rank != null ? `P${candidate.portfolio_rank}` : "P—";
    }
    return candidate.rank != null ? String(candidate.rank) : "—";
  }

  function candidateRankingValue(candidate = {}, packet = {}) {
    const portfolio = portfolioExecutionState(packet);
    return portfolio.enabled ? candidate.portfolio_rank : candidate.rank;
  }

  function candidateRankingScore(candidate = {}, packet = {}) {
    const portfolio = portfolioExecutionState(packet);
    return portfolio.enabled ? candidate.portfolio_score : candidate.score;
  }

  function sortCandidatesForReview(candidates = [], packet = {}) {
    return [...candidates].sort((a, b) => {
      const rankDelta = Number(candidateRankingValue(a, packet) ?? 999)
        - Number(candidateRankingValue(b, packet) ?? 999);
      return rankDelta || Number(candidateRankingScore(b, packet) ?? -1)
        - Number(candidateRankingScore(a, packet) ?? -1);
    });
  }

  function targetExecutionForCandidate(packet = {}, candidate = {}) {
    const executions = Array.isArray(packet?.target_executions) ? packet.target_executions : [];
    const key = candidate?.metadata?.portfolio_target_key;
    const hypothesisId = candidate?.metadata?.portfolio_hypothesis_id;
    return executions.find((item) => key && item.target_key === key)
      || executions.find((item) => hypothesisId && item.hypothesis_id === hypothesisId)
      || executions.find((item) => item.target?.symbol && item.target.symbol === candidate?.target?.symbol)
      || null;
  }

  function g1AuditScopeText(packet = {}, verdict = null) {
    const portfolio = portfolioExecutionState(packet);
    const code = String(verdict?.verdict || "PENDING").toUpperCase();
    if (!portfolio.enabled) {
      return code === "PASS"
        ? "G1 PASS는 선택된 표적의 근거 기반 탐색 경로, 후보·제외 정책과 실행 추적이 구현된 감사 기준을 충족했다는 의미입니다."
        : "G1은 선택된 표적의 탐색 경로, 후보·제외 정책과 실행 추적의 완결성을 판정합니다.";
    }
    const coverage = portfolioCoverageText(packet, { includeApproved: true });
    if (code === "PASS") {
      return `G1 PASS는 ${coverage}로 기록되고, 표적별 경로·후보·제외·도구 추적과 포트폴리오 순위 무결성이 감사 기준을 충족했다는 의미입니다.`;
    }
    if (code === "ABSTAIN" || portfolio.abstainRequired) {
      return `G1 ${code === "PENDING" ? "판정 대기" : code} · ${coverage}. 일부 표적의 실행이 완료되지 않은 포트폴리오는 전체 후보 승인 근거로 사용하지 않습니다.`;
    }
    return `G1 ${code} · ${coverage}. 표적별 실행 범위와 포트폴리오 순위 검사를 함께 해석해야 합니다.`;
  }

  function resultReadinessKey(snapshot = {}) {
    return [snapshot.run_id, snapshot.stage, snapshot.updated_at, snapshot.g1_verdict?.verdict_id, snapshot.candidate_packet?.packet_id,
      snapshot.research_candidate_selection?.selection_id, snapshot.human_decision?.decision_id].join(":");
  }

  function currentResultReadiness(snapshot = state.snapshot) {
    return snapshot && state.resultReadinessKey === resultReadinessKey(snapshot)
      && state.resultReadiness?.run_id === snapshot.run_id ? state.resultReadiness : null;
  }

  function runListResultLabel(snapshot, fallback) {
    // Reuse only the currently selected run's verified projection; no history-wide GETs.
    const readiness = snapshot.run_id === state.snapshot?.run_id ? currentResultReadiness(snapshot) : null;
    if (readiness?.ledger_valid === true) return resultPresentation(snapshot, readiness).title;
    return ["human_gate", "completed"].includes(snapshot.stage) ? `원본 상태 · ${fallback}` : fallback;
  }

  function resultPresentation(snapshot = {}, readiness = null) {
    const stage = String(snapshot.stage || "created").toLowerCase();
    // Runtime/terminal failure takes precedence over old successful artifacts.
    if (["cancel_requested", "cancelled", "failed", "blocked", "abstained"].includes(stage)) {
      return { title: runStatusPresentation(snapshot).label, summary: "보존된 부분 결과입니다. 연구후보 채택과 실험 검증 완료를 의미하지 않습니다.", complete: false };
    }
    if (!["human_gate", "completed"].includes(stage)) {
      return { title: "분석 진행 중", summary: "현재까지 저장된 결과입니다. 최종 분석과 채택 가능 여부는 아직 확정되지 않았습니다.", complete: false };
    }
    if (!readiness) return { title: "결과 상태 확인 중", summary: "저장된 감사 원장과 연구후보 채택 조건을 확인합니다. 확인 전에는 채택할 수 없습니다.", complete: false };
    if (readiness.analysis_complete !== true) return { title: "분석 미완료 · 기록 결과", summary: "완료 조건을 충족하지 못했습니다. 아래 기록은 최종 연구후보 추천이 아닙니다.", complete: false };
    const selection = readiness.research_selection || {};
    const selected = selection.state === "selected" && readiness.ledger_valid === true;
    if (!selected && (selection.allowed !== true || readiness.ledger_valid !== true)) return {
      title: selection.state === "stale" ? "분석 완료 · 기존 채택 재검토 필요" : "분석 완료 · 연구후보 채택 불가",
      summary: readiness.ledger_valid !== true ? "감사 원장 무결성 확인을 통과하지 못했습니다. 기록 후보를 채택 가능한 결과로 해석하지 마세요."
        : "분석은 종료되었지만 현재 채택 조건을 충족하지 못했습니다. 감사 판정과 잠금 사유를 확인하세요.", complete: true,
    };
    return {
      title: selected ? "분석 완료 · 연구후보 채택 기록됨" : "분석 완료 · 조건부 연구후보",
      summary: selected ? "연구자가 선택한 후보가 기록되었습니다. 약효·안전성 및 실험 검증은 별도로 남아 있습니다."
        : "계산·근거 분석 결과가 준비되었습니다. 남은 한계를 확인하고 후속 연구 후보를 직접 선택할 수 있습니다.",
      complete: true,
    };
  }

  function researchSelectionAllowed(snapshot, readiness) {
    const selection = readiness?.research_selection;
    return resultPresentation(snapshot, readiness).complete && readiness?.ledger_valid === true
      && selection?.allowed === true && selection?.policy_version === "research-selection-v1"
      && selection?.expected_g1_verdict_id === snapshot?.g1_verdict?.verdict_id
      && /^[a-f0-9]{64}$/i.test(String(selection?.expected_candidate_packet_sha256 || ""))
      && Boolean(selection?.expected_g1_verdict_id) && Array.isArray(selection?.eligible_candidate_ids)
      && selection.eligible_candidate_ids.length > 0 && !state.preview;
  }

  async function loadResultReadiness(snapshot = state.snapshot, options = {}) {
    if (!snapshot?.run_id || state.preview) return null;
    const key = resultReadinessKey(snapshot);
    if (!options.force && state.resultReadinessKey === key) return currentResultReadiness(snapshot);
    if (state.resultReadinessRequests.has(key)) return state.resultReadinessRequests.get(key);
    const request = (async () => {
      try {
        const readiness = await api(`/api/runs/${encodeURIComponent(snapshot.run_id)}/result-readiness`);
        if (readiness?.run_id !== snapshot.run_id) throw new Error("실행 식별자가 일치하지 않습니다.");
        if (resultReadinessKey(state.snapshot || {}) !== key) return null;
        state.resultReadiness = readiness;
        state.resultReadinessError = null;
        state.resultReadinessKey = key;
        renderResearchResults();
        updateRunStatus(state.snapshot);
        renderRunList();
        return readiness;
      } catch (error) {
        if (resultReadinessKey(state.snapshot || {}) === key) {
          state.resultReadiness = null;
          state.resultReadinessKey = key;
          state.resultReadinessError = "채택 가능 여부를 확인하지 못했습니다. 실행을 다시 불러와 확인하세요.";
          renderResearchResults();
        }
        return null;
      } finally {
        state.resultReadinessRequests.delete(key);
      }
    })();
    state.resultReadinessRequests.set(key, request);
    return request;
  }

  function researchLimitationLabel(item = {}) {
    const code = String(item.code || "");
    if (code === "ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION") return "예측 ADMET 위험 신호 · 실험 확인 필요";
    if (code.startsWith("ASSAY_VARIANT_")) return "정확한 변이·실험 조건 확인 필요";
    if (code === "COUNTER_EVIDENCE_NOT_FOUND") return "반증 검색 완료 · 직접 반증 근거 미확보";
    const text = String(item.message || code || "상세 한계 확인 필요");
    return text.length > 100 ? `${text.slice(0, 100)}…` : text;
  }

  function resultCandidateCardHtml(candidate, packet, readiness) {
    const selection = readiness?.research_selection || {};
    const selected = selection.state === "selected" && (selection.record?.selected_candidate_ids || []).includes(candidate.candidate_id);
    const eligible = researchSelectionAllowed(state.snapshot, readiness) && selection.eligible_candidate_ids.includes(candidate.candidate_id);
    const limitations = (selection.limitations || []).filter((item) => !item.candidate_id || item.candidate_id === candidate.candidate_id);
    const severity = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
    const rankedLimitations = [...limitations].sort((a, b) => Number(String(b.code).startsWith("ADMET_")) - Number(String(a.code).startsWith("ADMET_")) || Number(Boolean(b.candidate_id)) - Number(Boolean(a.candidate_id)) || (severity[b.severity] || 0) - (severity[a.severity] || 0));
    const warnings = [...new Set(rankedLimitations.map(researchLimitationLabel))].slice(0, 2);
    if (!warnings.length) warnings.push(...(candidate.safety_flags || []).slice(0, 2));
    if (!warnings.length) warnings.push("실험 검증 대기 · 예측값은 약효·안전성의 확증이 아닙니다.");
    const score = candidateRankingScore(candidate, packet);
    const structureUrl = candidateStructureUrl(String(candidate.smiles || "").trim());
    const identifier = candidateIdentityEntries(candidate).find(([label]) => label === "ChEMBL");
    const activity = candidate.activities?.[0];
    return `<article class="result-candidate-card">
      <header><span class="result-rank">#${esc(portfolioCandidateRank(candidate, packet))}</span><span class="result-candidate-state">${selected ? "채택 기록" : eligible ? "조건부 · 연구 채택 가능" : "기록 후보 · 채택 잠금"}</span></header>
      <div class="result-candidate-identity"><div><h5>${esc(candidate.name || candidate.candidate_id || "이름 미기록")}</h5><p>${esc(candidate.target?.symbol || "표적 미기록")} · ${esc(routeLabel(candidate.route))}</p>${identifier ? `<small>${esc(identifier[1])}</small>` : ""}</div>
      ${structureUrl ? `<figure class="molecule-structure molecule-structure-compact"><img class="molecule-structure-image" src="${esc(structureUrl)}" alt="${esc(candidate.name || "후보")} 2차원 구조" loading="lazy" decoding="async"><span class="molecule-structure-fallback">2D</span></figure>` : ""}</div>
      <dl><div><dt>상대 우선순위 점수</dt><dd>${score == null ? "미기록" : esc(formatScore(score))}</dd></div><div><dt>근거 / 도구 추적</dt><dd>${candidate.evidence?.length || 0} / ${candidate.tool_trace_ids?.length || 0}</dd></div><div><dt>기록 활성값 · 조건 확인 필요</dt><dd>${activity ? esc(`${activity.endpoint} ${activity.relation || ""} ${activity.value} ${activity.unit || ""}`) : "직접 활성값 미기록"}</dd></div></dl>
      <div class="result-candidate-warning"><b>남은 한계${limitations.length ? ` ${limitations.length}건` : ""}</b><ul>${warnings.map((warning) => `<li>${esc(warning)}</li>`).join("")}</ul></div>
      <button class="record-link" type="button" data-context-kind="candidate" data-context-id="${esc(candidate.candidate_id)}">선정 근거 · 후보 상세 보기 →</button>
    </article>`;
  }

  function renderResearchResults() {
    const snapshot = state.snapshot || {};
    const readiness = currentResultReadiness(snapshot);
    const result = resultPresentation(snapshot, readiness);
    const selection = readiness?.research_selection || {};
    const record = selection.record;
    const packet = snapshot.candidate_packet || {};
    const intake = getIntakeAssessment(snapshot);
    const candidates = intake.requiresInputCorrection ? [] : sortCandidatesForReview((packet.candidates || []).filter((item) => item.status !== "excluded"), packet);
    $("#researchResultTitle").textContent = result.title;
    $("#researchResultSummary").textContent = state.resultReadinessError || result.summary;
    const audit = `G0 ${snapshot.g0_verdict?.verdict || "대기"} · G1 ${snapshot.g1_verdict?.verdict || "대기"}`;
    const selectedLabel = selection.state === "selected" ? `채택 ${record?.selected_candidate_ids?.length || 0}건` : selection.state === "stale" ? "기존 채택 재검토 필요" : researchSelectionAllowed(snapshot, readiness) ? "사용자 선택 대기" : "채택 잠금 / 확인 필요";
    $("#researchResultStates").innerHTML = [
      ["분석", result.complete ? "완료" : result.title], ["산출물·과정 감사", audit], ["연구후보 채택", selectedLabel], ["약효·안전성 실험", "미검증 · 실험 검증 대기"],
    ].map(([label, value]) => `<div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
    const button = $("#openResearchSelection");
    button.disabled = !researchSelectionAllowed(snapshot, readiness) || state.researchSelectionSubmitting;
    button.textContent = selection.state === "selected" ? "연구후보 채택 기록됨" : "연구후보 채택";
    button.title = button.disabled ? "서버가 확인한 현재 채택 조건을 충족해야 합니다. 기존 검토 결정과 별개입니다." : "후보·연구자·선택 근거·한계 확인을 직접 입력합니다.";
    $("#resultCandidateCount").textContent = String(candidates.length);
    $("#resultCandidateTitle").textContent = selection.state === "selected" && readiness?.ledger_valid === true
      ? "연구후보 · 채택 기록" : researchSelectionAllowed(snapshot, readiness) ? "조건부 연구후보" : "현재까지 저장된 후보";
    const cards = $("#resultCandidateCards");
    cards.innerHTML = candidates.length ? candidates.slice(0, 3).map((candidate) => resultCandidateCardHtml(candidate, packet, readiness)).join("")
      : '<p class="summary-placeholder">현재 제시할 연구후보가 없습니다. 가설·감사 탭에서 중단 원인과 보존 기록을 확인하세요.</p>';
    wireMoleculeStructures(cards);
    const limitations = selection.limitations || [];
    const advisories = selection.advisories || [];
    const generalLimitations = limitations.filter((item) => !item.candidate_id);
    $("#researchResultDetails").innerHTML = `${generalLimitations.length ? `<p class="result-general-limitations">가설 수준 한계: ${[...new Set(generalLimitations.map(researchLimitationLabel))].map(esc).join(" · ")}. 근거 미확보가 효능의 증거는 아닙니다.</p>` : ""}<details class="result-status-details"><summary>남은 한계 ${limitations.length}건 · 외부 검토 참고 ${advisories.length}건 · 원본 상태</summary>
      <p>현재 표시용 해석이며 과거 기록은 변경하지 않습니다. 원본 단계: <code>${esc(snapshot.stage || "미기록")}</code> · 원본 상태: ${esc(snapshot.status_message || "미기록")}</p>
      ${record ? `<p>연구자 ${esc(record.researcher)} · 선택 근거: ${esc(record.rationale)} · 기록 상태 ${esc(selection.state)}</p>` : ""}
      ${selection.blocking_reasons?.length ? `<p>채택 잠금 사유: ${selection.blocking_reasons.map(esc).join(" · ")}</p>` : ""}
      ${limitations.length ? `<ul>${limitations.map((item) => `<li>${esc(item.message || item.code)}${item.candidate_id ? ` · ${esc(item.candidate_id)}` : ""}</li>`).join("")}</ul>` : "<p>한계 목록 미기록은 약효·안전성 검증 완료를 뜻하지 않습니다.</p>"}
      ${advisories.length ? `<h5>외부 검토 참고 · 확정 감사 한계와 구분</h5><ul>${advisories.map((item) => `<li>${esc(item.message || item.code)}</li>`).join("")}</ul>` : ""}
      <button class="text-button" type="button" data-summary-tab="audits">G0/G1 원본 감사 보기</button></details>`;
  }

  async function openResearchSelectionDialog() {
    const snapshot = state.snapshot;
    if (!snapshot || $("#openResearchSelection").disabled) return;
    const readiness = await loadResultReadiness(snapshot, { force: true });
    if (state.snapshot?.run_id !== snapshot.run_id || !researchSelectionAllowed(state.snapshot, readiness)) {
      showToast("연구후보 채택 확인 필요", "상태가 변경되었거나 채택 조건을 확인하지 못했습니다.", "error");
      return;
    }
    const selection = readiness.research_selection;
    state.researchSelectionContext = { runId: snapshot.run_id, verdictId: selection.expected_g1_verdict_id, packetSha256: selection.expected_candidate_packet_sha256, policyVersion: selection.policy_version, eligibleIds: [...selection.eligible_candidate_ids] };
    $("#researchSelectionForm").reset();
    $("#submitResearchSelection").disabled = false;
    $("#researchSelectionError").hidden = true;
    const candidates = sortCandidatesForReview((snapshot.candidate_packet?.candidates || []).filter((item) => selection.eligible_candidate_ids.includes(item.candidate_id)), snapshot.candidate_packet || {});
    $("#researchSelectionCandidates").innerHTML = candidates.map((candidate) => `<label class="decision-candidate"><input type="checkbox" name="research_candidate_ids" value="${esc(candidate.candidate_id)}"><span><b>${esc(candidate.name || candidate.candidate_id)}</b><small>${esc(candidate.target?.symbol || "표적 미기록")} · ${esc(routeLabel(candidate.route))}</small></span></label>`).join("");
    $("#selectionLimitations").innerHTML = `<b>확인할 한계</b><ul>${(selection.limitations || []).map((item) => `<li>${esc(item.message || item.code)}</li>`).join("")}<li>실험 검증은 완료되지 않았으며, 약효·안전성은 별도 검증이 필요합니다.</li></ul>`;
    $("#researchSelectionDialog").showModal();
  }

  async function submitResearchSelection(event) {
    event.preventDefault();
    const context = state.researchSelectionContext;
    if (!context || state.preview || state.researchSelectionSubmitting) return;
    const errorNode = $("#researchSelectionError");
    const fail = (message) => { errorNode.textContent = message; errorNode.hidden = false; };
    const selected = $$('input[name="research_candidate_ids"]:checked').map((input) => input.value);
    const researcher = $("#selectionResearcher").value.trim();
    const rationale = $("#selectionRationale").value.trim();
    if (!selected.length || !researcher || !rationale || !$("#selectionAcknowledgement").checked) {
      fail("후보를 직접 선택하고 연구자·선택 근거·한계 확인을 모두 입력하세요.");
      return;
    }
    if (state.snapshot?.run_id !== context.runId || state.snapshot?.g1_verdict?.verdict_id !== context.verdictId || selected.some((id) => !context.eligibleIds.includes(id))) {
      fail("감사 또는 실행 상태가 변경되었습니다. 창을 닫고 현재 조건을 다시 확인하세요.");
      return;
    }
    const payload = { researcher, rationale, selected_candidate_ids: selected, expected_g1_verdict_id: context.verdictId, expected_candidate_packet_sha256: context.packetSha256, limitations_acknowledged: true, policy_version: context.policyVersion };
    state.researchSelectionSubmitting = true;
    $("#submitResearchSelection").disabled = true;
    errorNode.hidden = true;
    try {
      const response = await api(`/api/runs/${encodeURIComponent(context.runId)}/research-selection`, { method: "POST", body: JSON.stringify(payload) });
      if (state.snapshot?.run_id !== context.runId) return;
      const returned = unwrapSnapshot(response);
      if (returned?.brief) state.snapshot = returned;
      else await fetchRun(context.runId, { quiet: true });
      await loadResultReadiness(state.snapshot, { force: true });
      $("#researchSelectionDialog").close();
      renderSnapshot();
      showToast("연구후보 채택을 기록했습니다", "실험 검증과 약효·안전성 승인은 별개입니다.", "success");
    } catch (error) {
      // Ambiguous writes are reconciled by GET only; never repeat POST automatically.
      const fresh = state.snapshot?.run_id === context.runId ? await loadResultReadiness(state.snapshot, { force: true }) : null;
      const saved = fresh?.research_selection?.record;
      const sameSelection = fresh?.research_selection?.state === "selected" && saved?.researcher === researcher && saved?.rationale === rationale
        && saved?.expected_g1_verdict_id === context.verdictId && saved?.candidate_packet_sha256 === context.packetSha256 && saved?.policy_version === context.policyVersion
        && Array.isArray(saved?.selected_candidate_ids) && saved.selected_candidate_ids.length === selected.length && selected.every((id) => saved.selected_candidate_ids.includes(id));
      if (sameSelection) {
        $("#researchSelectionDialog").close();
        showToast("채택 기록을 조회로 확인했습니다", "중복 요청은 보내지 않았습니다.", "success");
      } else {
        fail(error.status === 409 || error.status === 422 ? `채택하지 못했습니다. ${error.message}` : "응답을 확인하지 못했습니다. 중복 제출하지 않았습니다. 창을 닫고 채택 기록을 다시 확인하세요.");
        state.researchSelectionContext = null;
      }
    } finally {
      state.researchSelectionSubmitting = false;
      $("#submitResearchSelection").disabled = !state.researchSelectionContext;
      renderResearchResults();
    }
  }

  function renderDiscoverySummary() {
    const snapshot = state.snapshot || {};
    const hypothesisPacket = snapshot.hypothesis_packet || {};
    const hypotheses = hypothesisPacket.hypotheses || [];
    const admission = ma1AdmissionProjection(hypothesisPacket);
    const ma1Metadata = hypothesisPacket.metadata || {};
    const discoveryTrace = Array.isArray(ma1Metadata.ma1_discovery_candidate_trace)
      ? ma1Metadata.ma1_discovery_candidate_trace
      : [];
    const screenedTargets = new Set(
      (Array.isArray(ma1Metadata.goal_focus_screened_symbols)
        ? ma1Metadata.goal_focus_screened_symbols
        : (ma1Metadata.ma1_screening_history_sources || []).map(screeningTargetSymbol))
        .map((symbol) => String(symbol || "").trim().toLowerCase())
        .filter(Boolean),
    ).size;
    const discoveredTargets = Math.max(
      screenedTargets,
      nonnegativeInteger(
        ma1Metadata.ma1_discovery_candidate_trace_total,
        ma1Metadata.ma1_scope_candidate_pool_count == null
          ? discoveryTrace.length
          : ma1Metadata.ma1_scope_candidate_pool_count,
      ),
    );
    const candidates = snapshot.candidate_packet?.candidates || [];
    const evidence = [...(admission.enabled ? admission.rows.map((row) => row.hypothesis).filter(Boolean) : hypotheses).flatMap((item) => item.evidence || []), ...candidates.flatMap((item) => item.evidence || [])];
    const uniqueEvidence = new Set(evidence.map((item, index) => item.evidence_id || `${item.source || "source"}:${item.external_id || item.title || index}`));
    $("#discoveryEvidenceCount").textContent = compactNumber(uniqueEvidence.size);
    $("#discoveryHypothesisCount").textContent = admission.enabled
      ? `제안 ${compactNumber(admission.submittedCount)} / 승인 ${compactNumber(admission.approvedCount)}`
      : compactNumber(hypotheses.length);
    $("#discoveryCandidateCount").textContent = compactNumber(candidates.length);
    const candidatePacket = snapshot.candidate_packet || {};
    const portfolio = portfolioExecutionState(candidatePacket);
    $("#discoveryTargetCount").textContent = portfolio.enabled
      ? `${compactNumber(portfolio.concluded)}/${compactNumber(portfolio.eligible)}`
      : snapshot.candidate_packet
      ? portfolio.executionStatus === "completed" ? "1/1" : "0/1"
      : `${compactNumber(screenedTargets)}/${compactNumber(discoveredTargets)}`;
    $("#discoveryTargetCountLabel").textContent = snapshot.candidate_packet
      ? "Target branches evaluated"
      : "MA1 targets screened / discovered";

    const latestAudit = snapshot.g1_verdict || (!snapshot.candidate_packet ? snapshot.g0_verdict : null);
    $("#discoverySummaryStatus").textContent = snapshot.g1_verdict
      ? `${snapshot.g1_verdict.gate || "G1"} · ${verdictLabel(snapshot.g1_verdict.verdict)}${portfolio.enabled ? ` · 표적 ${portfolio.concluded}/${portfolio.eligible} 평가` : ""}`
      : snapshot.candidate_packet
      ? `MA2 · ${traceStatusLabelKo(portfolio.executionStatus)}${portfolio.enabled ? ` · 표적 ${portfolio.concluded}/${portfolio.eligible} 평가` : ""}`
      : latestAudit
      ? `${latestAudit.gate || "Audit"} · ${verdictLabel(latestAudit.verdict)}`
      : runStatusPresentation(snapshot).label;
    $("#discoverySummaryStatus").className = `summary-status ${String(snapshot.g1_verdict?.verdict || (snapshot.candidate_packet ? portfolio.executionStatus : latestAudit?.verdict) || snapshot.stage || "pending").toLowerCase()}`;

    const topHypothesis = [...(admission.enabled ? admission.rows.filter((row) => row.status === "approved").map((row) => row.hypothesis) : hypotheses)].sort((a, b) => {
      const rankDelta = Number(a.rank || 999) - Number(b.rank || 999);
      return rankDelta || Number(b.score || 0) - Number(a.score || 0);
    })[0];
    const hypothesisRoot = $("#topHypothesisContent");
    if (!topHypothesis) {
      hypothesisRoot.innerHTML = admission.enabled
        ? `<p class="summary-placeholder">MA1 제안 ${compactNumber(admission.submittedCount)}건 · MA3 승인 ${compactNumber(admission.approvedCount)}건. ${admission.reviewed ? "가설별 보강·보류·기각 이유는 MA1 Results에서 확인할 수 있습니다." : "잠정 가설은 아직 MA3 심사 전이며, MA2 진행 승인을 뜻하지 않습니다."}</p>`
        : hypothesisPacket.packet_id
        ? `<p class="summary-placeholder">MA1이 ${compactNumber(screenedTargets)}개 표적을 심사했지만 retained hypothesis는 0개입니다.${discoveredTargets > screenedTargets ? ` 미심사 ${compactNumber(discoveredTargets - screenedTargets)}개가 남아 있습니다.` : ""}</p>`
        : '<p class="summary-placeholder">MA1 가설을 기다리는 중입니다.</p>';
    } else {
      const support = (topHypothesis.evidence || []).filter((item) => String(item.supports || "support").toLowerCase() === "support").length;
      const counter = (topHypothesis.evidence || []).filter((item) => String(item.supports || "").toLowerCase() === "counter").length;
      const hypothesisId = topHypothesis.hypothesis_id || topHypothesis.target?.symbol || "top";
      hypothesisRoot.innerHTML = `
        <div class="summary-record-title"><strong>${esc(topHypothesis.target?.symbol || "표적 미확정")}</strong><span>${esc(directionLabel(topHypothesis.intervention_direction))}</span></div>
        <p>${esc(topHypothesis.mechanism || topHypothesis.claims?.[0]?.text || "가설 설명 미기록")}</p>
        <dl class="summary-record-meta">
          <div><dt>지지 근거</dt><dd>${compactNumber(support)}</dd></div>
          <div><dt>반증 근거</dt><dd>${compactNumber(counter)}</dd></div>
          <div><dt>G0 감사</dt><dd>${esc(snapshot.g0_verdict ? verdictLabel(snapshot.g0_verdict.verdict) : "대기")}</dd></div>
        </dl>
        <button class="record-link" type="button" data-context-kind="hypothesis" data-context-id="${esc(hypothesisId)}">근거와 반증가능성 검토 <span aria-hidden="true">→</span></button>`;
    }

    const eligibleCandidates = candidates.filter((item) => item.status !== "excluded");
    const topCandidate = sortCandidatesForReview(eligibleCandidates, candidatePacket)[0];
    const candidateRoot = $("#topCandidateContent");
    const candidateTitle = $("#topCandidateTitle");
    const g1Code = String(snapshot.g1_verdict?.verdict || "PENDING").toUpperCase();
    const candidateAudited = g1Code === "PASS" && !portfolio.abstainRequired;
    candidateTitle.textContent = candidateAudited ? "최상위 후보" : "현재 기록상 선두 후보";
    if (!topCandidate) {
      const reasons = portfolio.reasonCodes.length
        ? ` · ${portfolio.reasonCodes.map((value) => String(value).replaceAll("_", " ")).join(" · ")}`
        : "";
      candidateRoot.innerHTML = snapshot.candidate_packet
        ? `<p class="summary-placeholder">MA2 ${esc(portfolio.executionStatus.toUpperCase())} · 감사 가능한 후보가 없습니다${esc(reasons)}.</p>`
        : g0StoppedBeforeMa2(snapshot)
        ? '<p class="summary-placeholder">G0 감사 미충족으로 Planner·MA2·G1은 실행되지 않았습니다.</p>'
        : '<p class="summary-placeholder">MA2 후보 평가를 기다리는 중입니다.</p>';
    } else {
      const activity = topCandidate.activities?.[0];
      const keyResult = activity
        ? `${activity.endpoint} ${activity.relation || ""} ${activity.value} ${activity.unit}`
        : topCandidate.metrics?.docking_score_kcal_mol != null
        ? `Docking ${topCandidate.metrics.docking_score_kcal_mol} kcal/mol`
        : "직접 활성·도킹 결과 미기록";
      const candidateId = topCandidate.candidate_id || topCandidate.name || "top";
      const targetRank = topCandidate.target_rank ?? topCandidate.rank;
      const rankingLabel = portfolio.enabled
        ? `Portfolio #${topCandidate.portfolio_rank || "—"} · Target #${targetRank || "—"}`
        : `Rank #${topCandidate.rank || "—"}`;
      const auditNote = candidateAudited
        ? "G1 감사 적합 포트폴리오의 최상위 후보"
        : portfolio.abstainRequired
        ? `MA2 ${portfolio.executionStatus.toUpperCase()} · 일부 표적 미완료로 승인 불가`
        : `G1 ${g1Code} · 감사 전 기록 후보`;
      const candidateFindings = (snapshot.g1_verdict?.findings || []).filter((finding) => String(finding.artifact_id || "") === String(candidateId));
      const trustNote = candidateAudited
        ? `G1 PASS · 근거 ${topCandidate.evidence?.length || 0}건 · 도구 추적 ${topCandidate.tool_trace_ids?.length || 0}건`
        : auditNote;
      const problemNote = candidateFindings.length
        ? `${findingGroupTitle(groupFindings(candidateFindings)[0])} ${candidateFindings.length}건`
        : topCandidate.safety_flags?.length
        ? `안전성 주의 ${topCandidate.safety_flags.length}건`
        : "중대한 미해결 감사 항목 없음";
      candidateRoot.innerHTML = `
        <div class="summary-record-title"><strong>${esc(topCandidate.name || "Unnamed candidate")}</strong><span>${esc(topCandidate.target?.symbol || "—")}</span></div>
        ${candidateSummaryChemistryHtml(topCandidate)}
        <p>${esc(topCandidate.mechanism || "선정 기전 미기록")}</p>
        <dl class="summary-record-meta">
          <div><dt>Primary result</dt><dd>${esc(keyResult)}</dd></div>
          <div><dt>Ranking</dt><dd>${esc(rankingLabel)}</dd></div>
          <div><dt>Trust</dt><dd>${esc(trustNote)}</dd></div>
          <div><dt>Open issue</dt><dd>${esc(problemNote)}</dd></div>
        </dl>
        <button class="record-link" type="button" data-context-kind="candidate" data-context-id="${esc(candidateId)}">선정 요인과 실행 근거 검토 <span aria-hidden="true">→</span></button>`;
    }
    wireMoleculeStructures(candidateRoot);
  }

  function handleContextRequest(event) {
    const summaryTrigger = event.target.closest("[data-summary-tab]");
    if (summaryTrigger) {
      event.preventDefault();
      selectArtifactTab(summaryTrigger.dataset.summaryTab);
      return;
    }
    const trigger = event.target.closest("[data-context-kind]");
    if (!trigger) return;
    event.preventDefault();
    openContextPanel(trigger.dataset.contextKind, trigger.dataset.contextId, { trigger });
  }

  function handleContextKeydown(event) {
    if (!["Enter", " "].includes(event.key)) return;
    const trigger = event.target.closest("[data-context-kind]");
    if (!trigger) return;
    event.preventDefault();
    openContextPanel(trigger.dataset.contextKind, trigger.dataset.contextId, { trigger });
  }

  function openContextPanel(kind, id, options = {}) {
    const panel = $("#contextPanel");
    const content = $("#contextPanelContent");
    const snapshot = state.snapshot || {};
    state.contextSelection = { kind, id };
    if (options.trigger) state.contextTrigger = options.trigger;
    let title = "Context";
    let kicker = "Evidence detail";
    let html = '<div class="artifact-empty">상세 기록을 찾지 못했습니다.</div>';

    if (kind === "hypothesis") {
      const admission = ma1AdmissionProjection(snapshot.hypothesis_packet || {});
      const admissionRow = admission.rows.find((row) => row.id === String(id));
      const hypothesis = admission.enabled ? admissionRow?.hypothesis : (snapshot.hypothesis_packet?.hypotheses || []).find((item) => String(item.hypothesis_id || item.target?.symbol) === String(id));
      if (hypothesis) {
        title = hypothesis.target?.symbol || "Hypothesis";
        kicker = `Hypothesis · ${hypothesis.track || "MA1"}`;
        html = `${admission.enabled ? ma3AdmissionDispositionHtml(admissionRow, { detail: true }) : ""}${hypothesisContextHtml(hypothesis)}`;
      }
    } else if (kind === "candidate") {
      const candidate = (snapshot.candidate_packet?.candidates || []).find((item) => String(item.candidate_id || item.name) === String(id));
      if (candidate) {
        title = candidate.name || "Candidate";
        kicker = `Candidate · ${routeLabel(candidate.route)}`;
        html = candidateContextHtml(candidate, snapshot.candidate_packet);
      }
    } else if (kind === "audit") {
      const [gatePart, ...validationParts] = String(id || "g0").toLowerCase().split(":");
      const gate = gatePart === "g1" ? "g1" : "g0";
      const validationName = validationParts.join(":") || null;
      const verdict = gate === "g1" ? snapshot.g1_verdict : snapshot.g0_verdict;
      title = validationName ? validationLabel(validationName) : `${gate.toUpperCase()} audit detail`;
      kicker = "Scientific quality assurance";
      html = verdict ? auditContextHtml(verdict, validationName) : '<div class="artifact-empty">감사 기록이 아직 없습니다.</div>';
    } else if (kind === "finding") {
      const [gatePart, ...findingParts] = String(id || "g0").split(":");
      const gate = gatePart.toLowerCase() === "g1" ? "g1" : "g0";
      const findingKey = findingParts.join(":");
      const verdict = gate === "g1" ? snapshot.g1_verdict : snapshot.g0_verdict;
      const group = groupFindings(verdict?.findings || []).find((item) => item.key === findingKey);
      title = group ? findingGroupTitle(group) : "Audit finding";
      kicker = `${gate.toUpperCase()} · Grouped finding`;
      html = group ? findingContextHtml(group, gate, snapshot) : '<div class="artifact-empty">선택한 발견사항 기록을 찾지 못했습니다.</div>';
    }

    $("#contextPanelTitle").textContent = title;
    $("#contextPanelKicker").textContent = kicker;
    content.innerHTML = html;
    wireMoleculeStructures(content);
    panel.hidden = false;
    document.body.classList.add("context-panel-open");
    if (options.focus !== false) $("#closeContextPanel").focus({ preventScroll: true });
  }

  function closeContextPanel() {
    const panel = $("#contextPanel");
    if (!panel || panel.hidden) return;
    panel.hidden = true;
    document.body.classList.remove("context-panel-open");
    state.contextSelection = null;
    const trigger = state.contextTrigger;
    state.contextTrigger = null;
    if (trigger?.isConnected) trigger.focus({ preventScroll: true });
  }

  function hypothesisContextHtml(hypothesis) {
    const support = (hypothesis.evidence || []).filter((item) => String(item.supports || "support").toLowerCase() === "support");
    const counter = (hypothesis.evidence || []).filter((item) => String(item.supports || "").toLowerCase() === "counter");
    const context = (hypothesis.evidence || []).filter((item) => !support.includes(item) && !counter.includes(item));
    const claims = hypothesis.claims || [];
    return `
      <section class="context-section"><h4>Research claim</h4><p class="context-claim">${esc(claims[0]?.text || hypothesis.mechanism || "주장 미기록")}</p>
        <dl class="context-facts"><div><dt>Direction</dt><dd>${esc(directionLabel(hypothesis.intervention_direction))}</dd></div><div><dt>Rank</dt><dd>#${esc(hypothesis.rank || "—")}</dd></div><div><dt>가설 패킷 완성도</dt><dd>${formatScore(hypothesis.score)}</dd></div></dl>
        ${directionResolutionHtml(hypothesis)}
        <p>가설 패킷 완성도는 필수 주장·근거·반증 시험·추적 필드의 충족도이며, 질환-표적 연관성이나 과학적 신뢰도 점수가 아닙니다.</p>
      </section>
      ${hypothesisToolProvenanceHtml(hypothesis, state.snapshot?.hypothesis_packet || {})}
      ${evidenceGroupHtml("Supporting evidence", support)}
      ${evidenceGroupHtml("Contradictory evidence", counter)}
      ${context.length ? evidenceGroupHtml("Context evidence", context) : ""}
      <section class="context-section"><h4>Falsification</h4><p>${esc(hypothesis.falsification_test || "반증 시험 미기록")}</p></section>
      <section class="context-section provenance-chain"><h4>Provenance</h4><code>MA1 → ${esc(hypothesis.hypothesis_id || "Hypothesis")} → G0 → Planner → MA2</code></section>`;
  }

  function hypothesisToolProvenanceHtml(hypothesis, packet) {
    const targetMetadata = hypothesis.target?.metadata || {};
    const discovery = targetMetadata.target_discovery_statuses || [];
    const identity = targetMetadata.identity?.sources || [];
    const evidence = hypothesis.evidence || [];
    const contextOutcomes = targetMetadata.context_source_outcomes || [];
    const statusLabel = (value) => ({
      live: "LIVE",
      ok: "완료",
      empty: "결과 없음",
      error: "오류",
      not_configured: "미설정",
      skipped_call_budget: "호출 예산 생략",
      skipped_not_applicable: "해당 없음",
    })[String(value || "").toLowerCase()] || String(value || "미기록").replaceAll("_", " ");
    const discoverySummary = discovery.length
      ? discovery.map((item) => `${item.source} ${statusLabel(item.status)}${item.records_observed != null ? ` ${item.records_observed}건` : ""}`).join(" · ")
      : "탐색 기록 없음";
    const identitySummary = identity.length
      ? `${identity.map((item) => `${item.source} ${statusLabel(item.status)}`).join(" · ")} · ${hypothesis.target?.ensembl_id || "Ensembl 미기록"} · ${hypothesis.target?.uniprot_id || "UniProt 미기록"}`
      : "ID 정규화 기록 없음";
    const evidenceCounts = evidence.reduce((result, item) => {
      const provenance = String(item.metadata?.provenance_class || (item.metadata?.cache_hit ? "external_cache" : item.metadata?.live ? "external_live" : "recorded"));
      if (provenance.includes("fixture")) result.fixture += 1;
      else if (provenance.includes("cache")) result.cache += 1;
      else if (provenance.includes("live")) result.live += 1;
      else result.recorded += 1;
      return result;
    }, { live: 0, cache: 0, fixture: 0, recorded: 0 });
    const evidenceSources = [...new Set(evidence.map((item) => item.source).filter(Boolean))];
    const evidenceSummary = `${evidenceSources.join(", ") || "소스 미기록"} · LIVE ${evidenceCounts.live} · CACHE ${evidenceCounts.cache} · FIXTURE ${evidenceCounts.fixture} · 기록 ${evidenceCounts.recorded}`;
    const queriedContext = [...new Set(contextOutcomes.filter((item) => item.queried).map((item) => `${item.source} ${statusLabel(item.status)}`))];
    const unavailableContext = [...new Set(contextOutcomes.filter((item) => !item.queried && ["not_configured", "error", "skipped_call_budget"].includes(item.status)).map((item) => `${item.source} ${statusLabel(item.status)}`))];
    const contextSummary = [...queriedContext, ...unavailableContext].join(" · ") || "추가 맥락 조회 없음";
    const interactionBackend = packet?.metadata?.interaction?.backend || (state.snapshot?.mode === "mock" ? "deterministic_local" : "기록된 MA1 backend");
    const g0 = state.snapshot?.g0_verdict;
    const rows = [
      ["질환·표적 탐색", discoverySummary],
      ["표적 ID 정규화", identitySummary],
      ["근거 수집", evidenceSummary],
      ["맥락 소스 라우팅", contextSummary],
      ["가설 구성", `${hypothesis.track || "track 미기록"} track · ${interactionBackend} · LLM ${String(state.snapshot?.mode || "mock").toUpperCase()}`],
      ["독립 감사", g0 ? `${g0.gate || "G0"} ${g0.verdict || "PENDING"} · ${g0.reviewer_model || "deterministic rules"}` : "G0 대기"],
    ];
    const disclosureKey = `${state.snapshot?.run_id || "run"}:hypothesis-provenance:${hypothesis.hypothesis_id || hypothesis.target?.symbol || "unknown"}`;
    return `<section class="context-section tool-provenance-summary hypothesis-source-summary"><h4>검증 소스·경로</h4>
      ${renderTargetIdentity(hypothesis.target || {})}
      ${renderEvidenceSources(evidence)}
      <details class="provenance-details" data-disclosure="${esc(disclosureKey)}"><summary>조회·감사 상세</summary><dl class="context-facts">${rows.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl></details>
      <p class="context-helper">저장된 HypothesisPacket·source outcome·GateVerdict 기준 · 개별 논문과 원문 링크는 Evidence에서 확인</p></section>`;
  }

  function candidateContextHtml(candidate, packet) {
    const activity = candidate.activities?.[0];
    const score = candidateRankingScore(candidate, packet);
    const portfolio = portfolioExecutionState(packet);
    const execution = targetExecutionForCandidate(packet, candidate);
    const targetRank = candidate.target_rank ?? candidate.rank;
    const rankFacts = portfolio.enabled
      ? `<div><dt>Portfolio rank</dt><dd>#${esc(candidate.portfolio_rank || "—")}</dd></div><div><dt>Target rank</dt><dd>#${esc(targetRank || "—")}</dd></div>`
      : `<div><dt>Rank</dt><dd>#${esc(candidate.rank || "—")}</dd></div>`;
    const executionFact = execution
      ? `<div><dt>Target execution</dt><dd>${esc(String(execution.status || "unknown").toUpperCase())}</dd></div>`
      : "";
    return `
      ${candidateChemistryHtml(candidate)}
      <section class="context-section"><h4>Result</h4>
        <dl class="context-facts"><div><dt>Target</dt><dd>${esc(candidate.target?.symbol || "—")}</dd></div><div><dt>Route</dt><dd>${esc(routeLabel(candidate.route))}</dd></div>${rankFacts}<div><dt>${portfolio.enabled ? "Portfolio score" : "Candidate score"}</dt><dd>${formatScore(score)}</dd></div>${portfolio.enabled ? `<div><dt>Target score</dt><dd>${formatScore(candidate.score)}</dd></div>` : ""}${executionFact}<div><dt>Activity</dt><dd>${activity ? esc(`${activity.endpoint} ${activity.relation || ""} ${activity.value} ${activity.unit}`) : "미기록"}</dd></div></dl>
      </section>
      ${candidateToolProvenanceHtml(candidate, packet)}
      <section class="context-section"><h4>Rationale</h4><p class="context-claim">${esc(candidate.mechanism || "기전 미기록")}</p>
        ${portfolio.enabled ? `<p><b>Portfolio coverage</b> · ${esc(portfolioCoverageText(packet, { includeApproved: true }))}. ${portfolio.complete ? "전체 실행 범위가 결론에 도달했습니다." : "일부 표적의 실행이 완료되지 않아 이 후보만으로 전체 포트폴리오를 승인할 수 없습니다."}</p>` : ""}
        ${candidateWhyHtml(candidate, packet)}
      </section>
      ${evidenceGroupHtml("Evidence", candidate.evidence || [])}
      <section class="context-section"><h4>Risk evidence</h4>${admetPanelHtml(candidate)}</section>
      <section class="context-section provenance-chain"><h4>Provenance</h4><code>MA1 target ${esc(candidate.target?.symbol || "—")} → Planner → MA2 ${esc(candidate.candidate_id || candidate.name || "Candidate")}${portfolio.enabled ? ` → portfolio ${esc(portfolioCandidateRank(candidate, packet))}` : ""} → G1</code></section>
      <details class="context-section" data-disclosure="context-trace:${esc(candidate.candidate_id || candidate.name || "candidate")}"><summary><b>Trace</b> · MA2 단계별 도구 실행</summary>${candidateProcessHtml(candidate, packet)}</details>`;
  }

  function candidateToolProvenanceHtml(candidate, packet) {
    const traceIds = new Set(candidate.tool_trace_ids || []);
    const calls = (packet?.tool_calls || []).filter((call) => traceIds.has(call.call_id));
    const byTool = (name) => calls.find((call) => call.tool_name === name);
    const search = byTool(candidate.route === "repurposing" ? "repurposing_search" : "de_novo_generator");
    const chemistry = byTool("chemistry_validator");
    const docking = byTool("docking");
    const admet = byTool("admet_predictor");
    const searchProvenance = search?.provenance || {};
    const delegatedSearch = searchProvenance.delegate_provenance || {};
    const searchSources = [
      searchProvenance.dataset_version ? `DrugCentral ${searchProvenance.dataset_version}` : "",
      delegatedSearch.chembl_version || "",
      candidate.route === "de_novo" && search ? "RDKit BRICS" : "",
    ].filter(Boolean);
    const chemistryVersion = chemistry?.provenance?.backend_version;
    const admetExecution = admet?.provenance?.result?.execution || {};
    const admetModel = admetExecution.model_metadata || {};
    const activity = candidate.activities?.[0];
    const restrictedSources = candidate.metadata?.restricted_safety_screen?.sources || [];
    const queriedSafetySources = [...new Set(restrictedSources.filter((item) => item.queried).map((item) => item.source).filter(Boolean))];
    const unconfiguredSafetySources = [...new Set(restrictedSources.filter((item) => item.status === "not_configured").map((item) => item.source).filter(Boolean))];
    const partialSafetySources = queriedSafetySources.filter((source) => unconfiguredSafetySources.includes(source));
    const completeSafetySources = queriedSafetySources.filter((source) => !partialSafetySources.includes(source));
    const missingSafetySources = unconfiguredSafetySources.filter((source) => !partialSafetySources.includes(source));
    const safetySummary = [
      completeSafetySources.length ? `${completeSafetySources.join(", ")} 조회` : "",
      partialSafetySources.length ? `${partialSafetySources.join(", ")} 부분 조회` : "",
      missingSafetySources.length ? `${missingSafetySources.join(", ")} 미설정` : "",
    ].filter(Boolean).join(" · ") || "조회 기록 없음";
    const g1 = state.snapshot?.g1_verdict;
    const searchStatus = ({ completed: "완료", skipped: "생략", unavailable: "사용 불가", failed: "실패" })[search?.execution_status] || search?.execution_status || "완료";
    const dockingSummary = docking?.execution_status === "skipped"
      ? "생략 · 직접 측정 활성 근거 사용"
      : docking
      ? `${docking.source_version || "도킹 도구"}${candidate.metrics?.docking_score_kcal_mol != null ? ` · ${candidate.metrics.docking_score_kcal_mol} kcal/mol` : ""}`
      : "도구 기록 없음";
    const rows = [
      ["후보 검색", search ? `${searchSources.join(" + ") || search.source_version || "기록된 검색 도구"} · ${searchStatus}` : "도구 기록 없음"],
      ["화학·2D 구조", chemistry ? `RDKit ${chemistryVersion || "버전 미기록"} · 구조 유효성·물성·SVG 렌더링` : "도구 기록 없음"],
      ["활성 근거", activity ? `${activity.endpoint} ${activity.relation || ""} ${activity.value} ${activity.unit} · ${activity.assay_type || "assay 미기록"} · ChEMBL 측정값` : "직접 측정값 없음"],
      ["ADMET", admet ? `ADMET-AI ${admetModel.admet_ai_version || "버전 미기록"} · Chemprop ${admetModel.chemprop_version || "버전 미기록"} · ${admetModel.device ? String(admetModel.device).toUpperCase() : "장치 미기록"}` : "도구 기록 없음"],
      ["Docking", dockingSummary],
      ["규제·금지약물", safetySummary],
      ["독립 감사", g1 ? `${g1.gate || "G1"} ${g1.verdict || "PENDING"} · ${g1.reviewer_model || "deterministic rules"}` : "G1 대기"],
    ];
    return `<section class="context-section tool-provenance-summary"><h4>검증 도구·출처</h4><dl class="context-facts">${rows.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl><p class="context-helper">저장된 CandidatePacket 도구 호출과 provenance 기준 · 원시 해시·호출 ID는 Trace에서 확인</p></section>`;
  }

  function evidenceGroupHtml(title, records) {
    if (!records.length) return `<section class="context-section evidence-group"><h4>${esc(title)}</h4><p class="context-empty">기록 없음</p></section>`;
    return `<section class="context-section evidence-group"><h4>${esc(title)} <span>${records.length}</span></h4><div class="evidence-record-list">${records.map((item) => {
      const safeUrl = /^https?:\/\//i.test(item.url || "") ? esc(item.url) : "";
      const recordTitle = String(item.title || item.external_id || "Evidence record").slice(0, 240);
      const provenance = item.metadata?.provenance_class || (item.metadata?.cache_hit ? "external_cache" : item.metadata?.live ? "external_live" : "recorded");
      const sourceLocation = item.metadata?.source_location ? String(item.metadata.source_location).slice(0, 240) : "";
      const evidenceText = evidenceExcerptPresentation(item);
      const contentOrigin = evidenceContentOriginLabel(item);
      const sourceIdentity = evidenceSourceIdentityLabel(item, contentOrigin);
      const scientificRole = evidenceScientificRoleLabel(item);
      return `<article><div><b>${safeUrl ? `<a href="${safeUrl}" target="_blank" rel="noreferrer">${esc(recordTitle)}</a>` : esc(recordTitle)}</b><span>${esc(sourceIdentity)}</span></div><p class="evidence-record-text is-${esc(evidenceText.tone)}"><span class="evidence-record-kind">${esc(evidenceText.kind)}</span>${esc(evidenceText.text)}</p><small>${esc(scientificRole)} · ${esc(provenanceLabel(provenance))}${sourceLocation ? ` · ${esc(sourceLocation)}` : ""}</small></article>`;
    }).join("")}</div></section>`;
  }

  function evidenceSourceIdentityLabel(item, contentOrigin) {
    const metadata = item?.metadata || {};
    const source = String(metadata.source_database || item?.source || "출처 미기록").slice(0, 120);
    const identifiers = [
      item?.external_id ? `ID ${String(item.external_id).slice(0, 160)}` : "",
      metadata.pmid ? `PMID ${String(metadata.pmid).slice(0, 80)}` : "",
      metadata.pmcid ? `PMCID ${String(metadata.pmcid).slice(0, 80)}` : "",
      metadata.doi ? `DOI ${String(metadata.doi).slice(0, 160)}` : "",
    ].filter(Boolean);
    return `${source} · ${contentOrigin || "콘텐츠 유형 미기록"} · ${identifiers.length ? identifiers.slice(0, 4).join(" · ") : "식별자 미기록"}`;
  }

  function evidenceContentOriginLabel(item) {
    const metadata = item?.metadata || {};
    const availability = String(metadata.content_availability || "").toLowerCase();
    const relation = String(metadata.citation_relation || "").toLowerCase();
    const hop = Number(metadata.citation_hop);
    if (availability === "oa_fulltext_excerpt") {
      return relation === "reference" && hop === 1
        ? "OA 본문 · reference 1-hop"
        : relation === "direct_search" && hop === 0
        ? "OA 본문 · direct search"
        : "OA 본문 발췌";
    }
    if (availability === "abstract_excerpt") return "초록 발췌";
    if (relation === "reference" && hop === 1) return "reference 1-hop · 메타데이터";
    if (metadata.excerpt) return "저장 원문 발췌";
    return "메타데이터";
  }

  function evidenceScientificRoleLabel(item) {
    const roles = Array.isArray(item?.metadata?.claim_linkage_roles)
      ? item.metadata.claim_linkage_roles.map((value) => String(value))
      : [];
    const disposition = evidenceRoleLabel(item?.supports);
    if (roles.includes("direct_target_direction_primary")) return `직접 기전 근거 · ${disposition}`;
    if (roles.includes("primary_target_disease_association")) return `질환–표적 연관 근거 · ${disposition}`;
    return disposition;
  }

  function evidenceRoleLabel(value) {
    const normalized = String(value || "context").slice(0, 80);
    return ({ support: "지지 근거", counter: "반증 근거", context: "맥락 근거" })[normalized.toLowerCase()] || normalized;
  }

  function provenanceLabel(value) {
    const normalized = String(value || "recorded").slice(0, 80);
    return ({ external_live: "external_live · 외부 실시간 조회", external_cache: "external_cache · 외부 캐시", external_mixed: "external_mixed · 외부 실시간·캐시 혼합", fixture_fallback: "fixture · FIXTURE", recorded: "recorded · 저장 기록" })[normalized.toLowerCase()] || normalized.replaceAll("_", " ");
  }

  function evidenceExcerptPresentation(item) {
    const metadata = item?.metadata || {};
    const source = String(item?.source || "");
    if (metadata.content_availability === "oa_fulltext_excerpt" && metadata.excerpt) return { kind: "Europe PMC OA 본문 발췌", tone: "excerpt", text: String(metadata.excerpt) };
    if (metadata.content_availability === "abstract_excerpt" && metadata.excerpt) return { kind: "초록 원문 발췌", tone: "excerpt", text: String(metadata.excerpt) };
    if (metadata.excerpt) return { kind: "저장 원문 발췌", tone: "excerpt", text: String(metadata.excerpt) };
    if (item?.snippet) return { kind: "원천 구문", tone: "excerpt", text: String(item.snippet) };
    if (item?.summary) return { kind: "저장 근거 요약", tone: "structured", text: String(item.summary) };
    if (metadata.detail) return { kind: "저장 근거 구문", tone: "structured", text: String(metadata.detail) };

    const list = (value) => (Array.isArray(value) ? value.filter(Boolean) : value ? [value] : []);
    const joined = (value, limit = 4) => list(value).slice(0, limit).join(", ");
    const score = Number(metadata.quality_score);

    if (source === "Open Targets" || metadata.record_type === "target_disease_association") {
      const disease = metadata.disease_name || metadata.disease_id || "질환 미기록";
      const scoreText = Number.isFinite(score) ? ` · 연관 점수 ${score.toFixed(3)}` : "";
      return { kind: "구조화 근거 요약", tone: "structured", text: `Open Targets에서 ${disease}–표적의 순위 연관 레코드를 확인${scoreText}. 연관성 순위이며 인과관계를 직접 입증하는 값은 아닙니다.` };
    }

    if (source === "UniProt") {
      const aliases = joined(metadata.gene_names);
      const identity = [metadata.uniprot_id, metadata.entry_type, metadata.organism].filter(Boolean).join(" · ");
      return { kind: "구조화 근거 요약", tone: "structured", text: `UniProt 단백질 식별 레코드 확인${identity ? ` · ${identity}` : ""}${aliases ? ` · 유전자명 ${aliases}` : ""}. 표적의 식별자와 기능 맥락을 확인한 기록입니다.` };
    }

    if (source === "DGIdb") {
      const categories = list(metadata.categories).map((entry) => entry?.name).filter(Boolean).slice(0, 4).join(", ");
      const count = Number(metadata.interaction_count || 0);
      return { kind: "구조화 근거 요약", tone: "structured", text: `DGIdb 약물–유전자 상호작용 집계${count ? ` ${compactNumber(count)}건` : ""}${categories ? ` · 분류 ${categories}` : ""}. 약물화 가능성 맥락이며 특정 질환에서의 치료 효능을 입증하는 결과는 아닙니다.` };
    }

    if (source === "Pharos") {
      const facts = [metadata.tdl ? `개발 수준 ${metadata.tdl}` : "", metadata.target_family ? `표적 계열 ${metadata.target_family}` : ""].filter(Boolean).join(" · ");
      return { kind: "구조화 근거 요약", tone: "structured", text: `Pharos 표적 개발·약물화 레코드 확인${facts ? ` · ${facts}` : ""}. 표적 개발 상태에 대한 맥락이며 질환 치료 효과의 직접 근거는 아닙니다.` };
    }

    if (source === "ClinicalTrials.gov") {
      const phases = joined(metadata.phases);
      const interventions = list(metadata.interventions).map((entry) => entry?.name).filter(Boolean).slice(0, 5).join(", ");
      const status = metadata.overall_status || "상태 미기록";
      return { kind: "임상시험 등록 요약", tone: "structured", text: `등록 상태 ${status}${phases ? ` · 단계 ${phases}` : ""}${interventions ? ` · 중재 ${interventions}` : ""}. 시험 등록 정보이며 유효성 결과나 치료 효능을 뜻하지 않습니다.` };
    }

    if (metadata.is_preprint || ["bioRxiv", "medRxiv"].includes(source)) {
      const date = metadata.publication_date ? ` · 게시일 ${metadata.publication_date}` : "";
      return { kind: "프리프린트 서지 요약", tone: "bibliographic", text: `${metadata.preprint_server || source}에서 표적·질환 키워드가 포함된 프리프린트 메타데이터를 확인${date}. 동료심사 전 기록이며 인과 또는 효능 근거로 사용하지 않습니다.` };
    }

    if (["PubMed", "Europe PMC"].includes(source)) {
      const bibliographic = [metadata.journal, metadata.publication_date || metadata.publication_year, joined(metadata.publication_types, 2)].filter(Boolean).join(" · ");
      let absence = "이 실행에는 관련 원문 구문이 저장되지 않아 서지 정보만 표시합니다.";
      if (metadata.excerpt_selection_status === "related_excerpt_not_found") absence = "초록은 조회했지만 표적과 질환을 같은 문장에서 직접 연결한 구문을 찾지 못해 지지 근거로 승격하지 않았습니다.";
      else if (metadata.excerpt_selection_status === "abstract_not_available" || metadata.abstract_retrieval_status === "not_available") absence = "초록이 제공되지 않아 서지 메타데이터만 기록했습니다.";
      else if (metadata.abstract_retrieval_status === "unavailable") absence = `초록 조회에 실패했습니다 · ${metadata.abstract_retrieval_error_type || "원격 소스 오류"}.`;
      return { kind: "문헌 메타데이터", tone: "bibliographic", text: `${bibliographic ? `${bibliographic}. ` : ""}${absence}` };
    }

    if (metadata.description) return { kind: "원천 레코드 설명", tone: "structured", text: String(metadata.description) };
    const scope = String(metadata.evidence_scope || "");
    if (scope) return { kind: "구조화 근거 요약", tone: "structured", text: `원천 레코드에서 ${item?.title || item?.external_id || "해당 항목"}을 확인했습니다. 적용 범위: ${scope}` };
    return { kind: "식별자·제목 기록", tone: "bibliographic", text: `${item?.source || "원천 소스"}에서 “${item?.title || item?.external_id || "제목 미기록"}” 레코드를 확인했습니다. 원문 구문은 저장되지 않아 제목과 원천 식별자만 표시합니다.` };
  }

  function evidenceExcerptText(item) {
    return evidenceExcerptPresentation(item).text;
  }

  function auditContextHtml(verdict, validationName = null) {
    const gate = String(verdict.gate || "g0").toLowerCase().includes("g1") ? "g1" : "g0";
    const scope = gate === "g1"
      ? g1AuditScopeText(state.snapshot?.candidate_packet || {}, verdict)
      : "G0는 가설 패킷의 근거·반증가능성·식별자와 추적성을 판정합니다.";
    if (validationName) {
      const validation = (verdict.validations || []).find((item, index) => validationKey(item, index) === validationName);
      if (!validation) return '<section class="context-section"><h4>Audit control</h4><p class="context-empty">선택한 검사 기록을 찾지 못했습니다.</p></section>';
      const status = validationStatus(validation);
      const details = validation.details || {};
      const evidenceIds = [...new Set([
        ...(Array.isArray(validation.evidence_ids) ? validation.evidence_ids : []),
        ...(Array.isArray(details.evidence_ids) ? details.evidence_ids : []),
        ...(Array.isArray(details.source_evidence_ids) ? details.source_evidence_ids : []),
      ].filter(Boolean))];
      const relatedFindings = (verdict.findings || []).filter((finding) => {
        const linked = finding.validation_name || finding.control_name || finding.metadata?.validation_name || finding.metadata?.control_name;
        return String(linked || "") === String(validation.name || validation.control_type || "");
      });
      return `<section class="context-section"><h4>Audit control</h4><p class="context-claim">${esc(validationLabel(validation.name || validation.control_type))}</p>
        <dl class="context-facts"><div><dt>Status</dt><dd>${esc(status)}</dd></div><div><dt>Control type</dt><dd>${esc(validation.control_type || "recorded validation")}</dd></div><div><dt>Evidence IDs</dt><dd>${evidenceIds.length ? compactNumber(evidenceIds.length) : "미기록"}</dd></div><div><dt>Gate</dt><dd>${gate.toUpperCase()}</dd></div></dl>
      </section>
      <section class="context-section"><h4>Recorded result</h4><p>${esc(validationDetail(validation))}</p>${evidenceIds.length ? `<code class="audit-evidence-ids">${esc(evidenceIds.join(" · "))}</code>` : '<p class="context-empty">이 검사에 직접 연결된 evidence ID가 기록되지 않았습니다.</p>'}</section>
      <section class="context-section"><h4>Recorded details</h4><pre class="context-json">${esc(JSON.stringify(details, null, 2))}</pre></section>
      <section class="context-section"><h4>Related findings</h4><div class="finding-list">${relatedFindings.length ? relatedFindings.map(findingHtml).join("") : '<p class="context-empty">이 검사에 직접 연결된 finding 없음</p>'}</div></section>
      <section class="context-section provenance-chain"><h4>Provenance</h4><code>${gate.toUpperCase()} → ${esc(validation.name || validation.control_type || validationName)} → GateVerdict</code></section>`;
    }
    const groupedFindings = groupFindings(verdict.findings || []);
    return `<section class="context-section"><h4>전체 판정</h4><p class="context-claim">${esc(verdictLabel(verdict.verdict))} · ${esc(gateVerdictSummaryKo(verdict))}</p><p>${esc(scope)}</p></section>${validationTraceHtml(verdict.validations || [], gate)}<section class="context-section"><h4>감사 발견사항</h4><div class="finding-list">${groupedFindings.length ? groupedFindings.map((group) => findingGroupHtml(group, gate)).join("") : '<p class="context-empty">발견사항 없음</p>'}</div></section>`;
  }

  function renderMetrics() {
    const snapshot = state.snapshot || {};
    const intake = getIntakeAssessment(snapshot);
    const usages = snapshot.token_usage || [];
    const inputTokens = sum(usages, (item) => item.input_tokens);
    const outputTokens = sum(usages, (item) => item.output_tokens);
    const reasoningTokens = sum(usages, (item) => item.reasoning_tokens);
    const tokens = inputTokens + outputTokens + reasoningTokens;
    const cost = sum(usages, (item) => item.estimated_cost_usd);
    const trace = state.callTrace?.run_id === snapshot.run_id ? state.callTrace : null;
    const callCounts = callCountSummary(snapshot, trace);
    const modelCalls = callCounts.modelCalls;
    const toolCalls = callCounts.scientificToolCalls;
    const publicDataCalls = callCounts.publicDataCalls;
    const budget = snapshot.brief?.budget?.max_run_tokens || 1_000_000;
    const budgetPercent = budget ? Math.min(999, Math.round(tokens / budget * 100)) : 0;
    $("#metricTokens").textContent = compactNumber(tokens);
    $("#metricTokenBudget").textContent = `예산 ${compactNumber(budget)}의 ${budgetPercent}%`;
    $("#metricCalls").textContent = compactNumber(modelCalls + publicDataCalls + toolCalls);
    $("#metricCallDetail").textContent = `모델 ${modelCalls} · DB ${publicDataCalls} · 과학 도구 ${toolCalls} · 상세 보기`;
    $("#metricCost").textContent = `$${cost.toFixed(cost < 10 ? 2 : 1)}`;
    $("#metricCostKrw").textContent = `약 ${formatKrw(cost * KRW_PER_USD)}`;

    const g0 = snapshot.g0_verdict?.verdict || "—";
    const g1 = snapshot.g1_verdict?.verdict || "—";
    const latest = snapshot.g1_verdict || snapshot.g0_verdict;
    const portfolio = portfolioExecutionState(snapshot.candidate_packet || {});
    $("#metricAudit").textContent = snapshot.candidate_packet && !snapshot.g1_verdict && portfolio.abstainRequired
      ? "G1 판단 필요"
      : latest
      ? verdictLabel(latest.verdict)
      : intake.abstained
      ? "입력 확인 필요"
      : "대기";
    $("#metricAuditDetail").textContent = intake.abstained && !latest
      ? "G0 미실행 · 입력 검증 ABSTAIN"
      : `G0 ${g0} · G1 ${g1}${portfolio.enabled ? ` · MA2 ${portfolio.executionStatus.toUpperCase()} ${portfolio.concluded}/${portfolio.eligible} 결론` : ""}`;
  }

  function renderActivity() {
    const feed = $("#activityFeed");
    const events = state.snapshot?.events || [];
    const trace = state.callTrace?.run_id === state.snapshot?.run_id ? state.callTrace : null;
    const totalCalls = callCountSummary(state.snapshot || {}, trace).totalCalls;
    const usage = state.snapshot?.token_usage || [];
    const tokens = sum(usage, (item) => Number(item.input_tokens || 0) + Number(item.output_tokens || 0) + Number(item.reasoning_tokens || 0));
    const cost = sum(usage, (item) => item.estimated_cost_usd);
    const executionSummary = $("#executionTraceSummary");
    if (executionSummary) {
      executionSummary.textContent = `${compactNumber(totalCalls)} calls · ${compactNumber(tokens)} tokens · $${cost.toFixed(cost < 10 ? 2 : 1)} · ${events.length ? stageLabel(state.snapshot?.stage || "created") : "대기"}`;
    }
    const previousScrollTop = feed.scrollTop;
    const distanceFromBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight;
    const shouldStickToBottom = !feed.querySelector(".activity-item") || distanceFromBottom <= 32;
    captureDisclosureState(feed);
    if (!events.length) {
      feed.innerHTML = `
        <div class="empty-activity"><span aria-hidden="true">⌁</span><b>실행 이벤트를 기다리는 중</b>
        <small>단계별 진행과 감사 판단이 이곳에 표시됩니다.</small></div>`;
      return;
    }
    feed.innerHTML = events.slice(-120).map((event) => {
      const style = eventStyle(event);
      const trace = event.metadata?.interaction;
      const chips = [];
      if (event.node) chips.push(event.node);
      if (event.stage) chips.push(stageLabel(event.stage));
      if (event.artifact_ids?.length) chips.push(`산출물 ${event.artifact_ids.length}`);
      if (trace?.model) chips.push(trace.model);
      if (trace?.backend) chips.push(backendLabel(trace.backend));
      if (trace?.decision) chips.push(trace.decision);
      return `
        <article class="activity-item ${style.className}" data-stage="${esc(event.stage || "")}">
          <span class="activity-marker" aria-hidden="true">${style.icon}</span>
          <div class="activity-copy">
            <div class="activity-title">${esc(eventTitle(event))}</div>
            <div class="activity-message">${esc(event.message || event.event_type || "상태 업데이트")}</div>
            ${chips.length ? `<div class="activity-chips">${chips.map((chip) => `<span class="mini-chip">${esc(chip)}</span>`).join("")}</div>` : ""}
            ${interactionTraceHtml(trace, event.event_id)}
          </div>
          <time class="activity-time" datetime="${esc(event.created_at || "")}">${esc(formatTime(event.created_at))}</time>
        </article>`;
    }).join("");
    restoreDisclosureState(feed);
    feed.scrollTop = shouldStickToBottom ? feed.scrollHeight : previousScrollTop;
  }

  function interactionTraceHtml(trace, eventId) {
    if (!trace?.sender || !trace?.receiver) return "";
    const basis = Array.isArray(trace.basis) ? trace.basis.filter(Boolean).slice(0, 8) : [];
    const evidenceCount = Array.isArray(trace.evidence_ids) ? trace.evidence_ids.length : 0;
    const artifactCount = Array.isArray(trace.artifact_ids) ? trace.artifact_ids.length : 0;
    return `
      <details class="interaction-trace" data-disclosure="interaction:${esc(eventId || `${trace.sender}:${trace.receiver}:${trace.kind || "handoff"}`)}">
        <summary>
          <span class="interaction-route"><b>${esc(trace.sender)}</b><i aria-hidden="true">→</i><b>${esc(trace.receiver)}</b></span>
          <span class="interaction-kind">${esc(interactionKindLabel(trace.kind))}</span>
        </summary>
        <div class="interaction-body">
          <p>${esc(trace.summary || "에이전트 전달 기록")}</p>
          ${basis.length ? `<div class="interaction-basis"><b>판단 근거</b><ul>${basis.map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div>` : ""}
          <div class="interaction-foot">
            ${evidenceCount ? `<span>근거 ID ${evidenceCount}건</span>` : ""}
            ${artifactCount ? `<span>산출물 ${artifactCount}건</span>` : ""}
            ${trace.model ? `<span>모델 ${esc(trace.model)}</span>` : ""}
            <span>근거·판정 요약</span>
          </div>
        </div>
      </details>`;
  }

  function interactionKindLabel(kind) {
    const labels = {
      task_assignment: "업무 지시",
      audit_request: "감사 요청",
      artifact_handoff: "산출물 전달",
      plan_dispatch: "계획 전달",
      audit_decision: "감사 판단",
      selective_rework: "선택 재작업",
      structured_generation_request: "LLM 요청",
      structured_generation_response: "LLM 응답",
      fallback: "대체 실행",
      error: "호출 오류",
    };
    return labels[kind] || String(kind || "상호작용").replaceAll("_", " ");
  }

  function backendLabel(backend) {
    const labels = {
      deterministic_local: "LOCAL RULES",
      deterministic_rules: "RULE AUDIT",
      deterministic_tool_executor: "TOOL EXECUTOR",
      deterministic_rules_with_optional_external_reviewer: "RULES + OPTIONAL REVIEW",
      external_reviewer_plus_rules: "RULES + LLM REVIEW",
      openai_api: "OPENAI API",
      deterministic_fallback: "LOCAL FALLBACK",
    };
    return labels[backend] || backend;
  }

  function renderVerdict() {
    const root = $("#verdictContent");
    const verdict = state.selectedGate === "g1" ? state.snapshot?.g1_verdict : state.snapshot?.g0_verdict;
    if (!verdict) {
      const intake = getIntakeAssessment(state.snapshot || {});
      if (state.selectedGate === "g0" && intake.abstained) {
        const title = intake.diseaseNeedsReview
          ? "질환 확인 필요"
          : intake.verificationUnavailable
          ? "외부 질환 검증 불가"
          : intake.hasMismatch
          ? "연구 입력 불일치 확인"
          : "입력 확인 필요";
        const rationale = intake.explanation || (intake.diseaseNeedsReview
          ? "질환을 공인 외부 데이터에서 확인하지 못해 G0 감사를 시작하지 않았습니다."
          : intake.verificationUnavailable
          ? "외부 검증 소스에 연결할 수 없어 G0 감사를 시작하지 않았습니다."
          : "연구 입력을 명확히 해야 하므로 G0 감사를 시작하지 않았습니다.");
        root.innerHTML = `
          <div class="verdict-hero abstain intake-abstain">
            <span class="verdict-code">ABSTAIN</span>
            <div><b>입력 검증 · ${esc(title)}</b><small>G0 미실행</small></div>
          </div>
          <p class="audit-scope-note">이 표시는 G0 감사 결과가 아니라, 근거 없는 후보 생성을 막기 위한 사전 입력 검증 중단 상태입니다.</p>
          <p class="rationale">${esc(rationale)}</p>`;
        return;
      }
      if (state.selectedGate === "g1" && state.snapshot?.candidate_packet) {
        const packet = state.snapshot.candidate_packet;
        const portfolio = portfolioExecutionState(packet);
        if (portfolio.abstainRequired) {
          root.innerHTML = `
            <div class="verdict-hero abstain">
              <span class="verdict-code">${esc(portfolio.executionStatus.toUpperCase())}</span>
              <div><b>MA2 표적 포트폴리오 미완료</b><small>G1 판정 대기</small></div>
            </div>
            <p class="audit-scope-note">${esc(g1AuditScopeText(packet, null))}</p>
            <p class="rationale">${esc(portfolio.reasonCodes.length ? portfolio.reasonCodes.map((value) => String(value).replaceAll("_", " ")).join(" · ") : "표적별 실행 범위를 확인해야 합니다.")}</p>`;
          return;
        }
      }
      root.innerHTML = `
        <div class="empty-verdict"><span class="shield-empty" aria-hidden="true">◇</span><b>${state.selectedGate.toUpperCase()} 감사 대기</b>
        <p>MA3가 산출물과 실행 추적을 읽기 전용으로 검증합니다.</p></div>`;
      return;
    }
    const code = String(verdict.verdict || "ABSTAIN").toLowerCase();
    const findings = verdict.findings || [];
    const groupedFindings = groupFindings(findings);
    const validations = verdict.validations || [];
    const high = findings.filter((item) => ["CRITICAL", "HIGH"].includes(item.severity)).length;
    const unresolved = findings.filter((item) => !item.resolved).length;
    const scopeNote = code === "pass"
      ? ""
      : state.selectedGate === "g1"
      ? g1AuditScopeText(state.snapshot?.candidate_packet || {}, verdict)
      : "G0 감사는 가설 패킷의 근거·반증가능성·추적성을 검증합니다.";
    root.innerHTML = `
      <div class="verdict-hero ${esc(code)}">
        <span class="verdict-code">${esc(verdict.verdict)}</span>
        <div><b>${esc(verdict.gate)} 감사 · ${esc(verdictLabel(verdict.verdict))}</b>
        <small>${esc(verdict.reviewer_model || "독립 감사 규칙")}${verdict.reviewer_invoked ? " · 외부 모델 감사" : ""}${verdict.meta_critic_invoked ? " · 메타 감사" : ""}</small></div>
      </div>
      ${scopeNote ? `<p class="audit-scope-note">${esc(scopeNote)}</p>` : ""}
      <p class="rationale">${esc(verdict.rationale || "판정 근거가 기록되지 않았습니다.")}</p>
      ${validationTraceHtml(validations, state.selectedGate)}
      <div class="finding-summary">발견사항 ${findings.length}건<span class="severity-count">유형 ${groupedFindings.length}</span><span class="severity-count${high ? " high" : ""}">High+ ${high}</span><span class="severity-count">미해결 ${unresolved}</span></div>
      <div class="finding-list">${groupedFindings.length ? groupedFindings.slice(0, 12).map((group) => findingGroupHtml(group, state.selectedGate)).join("") : '<div class="artifact-empty">등록된 발견사항 없음</div>'}</div>
      ${reworkHtml(verdict.rework_order)}`;
  }

  function validationTraceHtml(validations, gate = state.selectedGate) {
    if (!validations?.length) return '<div class="validation-empty">기록된 감사 검사항목 없음</div>';
    const statuses = validations.map(validationStatus);
    const passed = statuses.filter((status) => status === "PASS").length;
    const warnings = statuses.filter((status) => status === "WARNING").length;
    return `<details class="validation-trace">
      <summary>실제 검사 항목 <b>${passed}/${validations.length} 통과${warnings ? ` · 경고 ${warnings}` : ""}</b></summary>
      <div class="validation-list">${validations.map((item, index) => {
        const status = statuses[index];
        const statusClass = status === "PASS" ? "passed" : status === "WARNING" ? "warning" : "failed";
        return `
        <div class="validation-row ${statusClass}" role="button" tabindex="0" aria-label="${esc(`${validationLabel(item.name)} ${status} 상세 보기`)}" data-context-kind="audit" data-context-id="${esc(`${gate}:${validationKey(item, index)}`)}">
          <span aria-hidden="true">${status === "PASS" ? "✓" : status === "WARNING" ? "△" : "!"}</span>
          <div><b>${esc(validationLabel(item.name))}</b><small>${esc(validationDetail(item))}</small></div>
          <em>${status}</em>
        </div>`;
      }).join("")}</div>
    </details>`;
  }

  function validationKey(validation, index = 0) {
    return String(validation?.name || validation?.control_type || `validation_${index}`).toLowerCase();
  }

  function validationStatus(validation) {
    const recorded = String(validation?.status || validation?.details?.status || "").toUpperCase();
    if (["WARNING", "WARN"].includes(recorded)) return "WARNING";
    if (["FAIL", "FAILED", "ERROR", "BLOCK"].includes(recorded)) return "FAIL";
    return validation?.passed ? "PASS" : "FAIL";
  }

  function validationLabel(name) {
    const labels = {
      rank_integrity_control: "순위 무결성 검사",
      target_diversity_control: "표적 다양성 검사",
      repurposing_positive_control: "약물 재창출 경로 양성 대조",
      de_novo_novelty_control: "신규 분자 경로 신규성 대조",
    };
    return labels[name] || String(name || "감사 통제").replaceAll("_", " ");
  }

  function validationDetail(validation) {
    const details = validation?.details || {};
    const preferred = ["message", "detail", "reason", "checked", "summary"];
    for (const key of preferred) {
      if (details[key] != null && typeof details[key] !== "object") return String(details[key]);
    }
    const entries = Object.entries(details).filter(([, value]) => value != null && typeof value !== "object").slice(0, 3);
    const detailLabels = {
      candidate_count: "후보 수",
      unique_targets: "고유 표적 수",
      checked: "검사 항목",
    };
    return entries.length
      ? entries.map(([key, value]) => `${detailLabels[key] || key.replaceAll("_", " ")}: ${value}`).join(" · ")
      : `${validation.control_type || "audit"} control`;
  }

  function findingHtml(finding) {
    const group = { code: finding.code || finding.category, category: finding.category, records: [finding] };
    return `
      <div class="finding-item">
        <span class="finding-severity ${esc(finding.severity || "INFO")}"></span>
        <div><b>${esc(findingGroupTitle(group))}</b><p>${esc(findingGroupSummary(group))}</p>
        <div class="finding-meta">${esc(finding.code || finding.category || "AUDIT_FINDING")} ${finding.failed_node ? `· ${esc(finding.failed_node)}` : ""}${finding.resolved ? " · 해결됨" : ""}</div></div>
      </div>`;
  }

  function groupFindings(findings = []) {
    const groups = new Map();
    const severityRank = { CRITICAL: 5, HIGH: 4, MEDIUM: 3, LOW: 2, INFO: 1 };
    findings.forEach((finding, index) => {
      const code = String(finding.code || finding.category || "AUDIT_FINDING");
      if (!groups.has(code)) {
        groups.set(code, {
          key: code,
          code,
          category: finding.category || "audit",
          severity: String(finding.severity || "INFO").toUpperCase(),
          records: [],
        });
      }
      const group = groups.get(code);
      group.records.push({ ...finding, _index: index });
      const currentSeverity = String(finding.severity || "INFO").toUpperCase();
      if ((severityRank[currentSeverity] || 0) > (severityRank[group.severity] || 0)) group.severity = currentSeverity;
    });
    return [...groups.values()];
  }

  function findingGroupTitle(group = {}) {
    const labels = {
      HYPOTHESIS_DIRECTION_NOT_ALLOWED: "선택한 허용 방향 밖의 가설",
      INTERVENTION_DIRECTION_UNRESOLVED: "직접 근거로 작용 방향을 확정하지 못함",
      RETAINED_INTERVENTION_DIRECTION_MISMATCH: "보존 가설과 요청·허용 방향 불일치",
      ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION: "예측 ADMET 위험 · 실험 확인 필요",
      SCIENTIFIC_TOOL_UNAVAILABLE: "과학 도구 또는 실데이터 사용 불가",
      PORTFOLIO_TARGETS_UNAVAILABLE: "표적 포트폴리오 실행 미완료",
      NO_REAL_REPURPOSING_CANDIDATES: "재창출 기준 충족 후보 없음",
      UNRESOLVED_DISEASE: "질환 식별 미확정",
      COUNTER_EVIDENCE_NOT_FOUND: "명시적 반증 근거 미확인",
      COUNTER_EVIDENCE_SEARCH_UNAVAILABLE: "반증 근거 검색 사용 불가",
      REVIEWER_PRIMARY_SOURCE_SUPPORT_INSUFFICIENT: "핵심 주장에 대한 1차 문헌 근거 부족",
      REVIEWER_PRIMARY_RESULT_SUPPORT_INSUFFICIENT: "핵심 결과에 대한 1차 근거 부족",
      REVIEWER_TRACK_ATTRIBUTION_MISCHARACTERIZED: "표적 발굴 경로 설명과 저장 기록 불일치",
      REVIEWER_TRACK_PROVENANCE_MISCHARACTERIZED: "표적 경로의 출처 설명이 실제 기록과 불일치",
      REVIEWER_COUNTER_EVIDENCE_SCOPE_MISMATCH: "반증 근거와 주장 범위 불일치",
      REVIEWER_COUNTER_EVIDENCE_LINKAGE_NONDISCRIMINATING: "반증 근거가 주장별로 구분되지 않음",
      REVIEWER_FLT3_VARIANT_AND_MODEL_CONTEXT_UNDERSPECIFIED: "FLT3 변이·세포 모델 조건 불충분",
      REVIEWER_FALSIFICATION_ENDPOINT_NOT_OPERATIONALIZED: "반증 실험 조건이 재현 가능하게 정의되지 않음",
      REVIEWER_SOURCE_ROUTING_RECORD_INCONSISTENCY: "소스 조회 기록과 보존 근거 불일치",
      REVIEWER_AML_SUBTYPE_OVERGENERALIZATION: "AML 아형 범위를 과도하게 일반화",
      REVIEWER_INHIBITION_DIRECTION_OPPOSES_TARGET_BIOLOGY: "억제 전략이 표적 생물학과 상충",
      REVIEWER_DNMT3A_DIRECTION_NOT_ESTABLISHED: "DNMT3A 치료 작용 방향 근거 부족",
      REVIEWER_CHRONIC_SAFETY_REQUIREMENT_UNADDRESSED: "만성 안전성 요구 미검토",
      REVIEWER_CEBPA_DRUGGABILITY_CONSTRAINT_MISMATCH: "CEBPA 약물화 가능성 제약 불일치",
      REVIEWER_COUNTER_EVIDENCE_RELEVANCE_MISMATCH: "반증 근거와 치료 주장 관련성 부족",
      REVIEWER_SUPPORT_LINKS_INCLUDE_NON_SUPPORTIVE_RECORDS: "지지 근거 연결에 비지지 레코드 포함",
      REVIEWER_NOVELTY_PRIORITY_NOT_REFLECTED: "신규 표적 우선순위가 결과에 반영되지 않음",
      REVIEWER_CELL_LINE_CONTEXT_MISROUTED_AS_SINGLE_CELL: "세포주 조건을 단일세포 조건으로 잘못 분류",
      ASSAY_VARIANT_EXPLICIT_MISMATCH: "후보 활성 assay와 가설 변이가 명시적으로 다름",
      ASSAY_VARIANT_CONTEXT_UNKNOWN: "후보 활성 assay의 변이 조건 미확인",
      ASSAY_VARIANT_CLASS_ONLY: "변이 유형만 일치 · 정확한 구성체 미확인",
      ASSAY_VARIANT_PARTIAL_OVERLAP: "변이 일부 겹침 · 복합 변이 조건 미확인",
      ASSAY_VARIANT_ANNOTATION_ONLY: "변이명만 일치 · 정확한 구성체 미확인",
      PROSPECTIVE_FALSIFICATION_ATTESTATION_INVALID: "미실험 상태·사전 실험 기준 기록 불일치",
      ASSAY_VARIANT_ALIGNMENT_ATTESTATION_MISMATCH: "assay–가설 변이 판정 재현 실패",
    };
    return labels[group.code] || genericFindingTitleKo(group.code);
  }

  function genericFindingTitleKo(code) {
    const raw = String(code || "AUDIT_FINDING").replace(/^REVIEWER_/, "");
    const phrases = {
      NOT_FOUND: "미확인",
      NOT_ESTABLISHED: "근거 미확립",
      NOT_OPERATIONALIZED: "구체화되지 않음",
      UNAVAILABLE: "사용 불가",
      INSUFFICIENT: "불충분",
      UNDERSPECIFIED: "조건 부족",
      UNADDRESSED: "미검토",
      MISMATCH: "불일치",
      INCONSISTENCY: "불일치",
      MISCHARACTERIZED: "설명 오류",
    };
    const tokenLabels = {
      PRIMARY: "1차", SOURCE: "소스", RESULT: "결과", SUPPORT: "지지", SUPPORTIVE: "지지성", COUNTER: "반증",
      EVIDENCE: "근거", TRACK: "경로", ATTRIBUTION: "귀속", PROVENANCE: "출처", SCOPE: "범위", LINKAGE: "연결",
      FALSIFICATION: "반증", ENDPOINT: "평가 지표", ROUTING: "라우팅", RECORD: "기록", INHIBITION: "억제",
      DIRECTION: "작용 방향", OPPOSES: "상충", TARGET: "표적", BIOLOGY: "생물학", CHRONIC: "만성", SAFETY: "안전성",
      REQUIREMENT: "요구", DRUGGABILITY: "약물화 가능성", CONSTRAINT: "제약", RELEVANCE: "관련성", LINKS: "연결",
      INCLUDE: "포함", RECORDS: "레코드", NOVELTY: "신규성", PRIORITY: "우선순위", REFLECTED: "반영",
      CELL: "세포", LINE: "세포주", CONTEXT: "맥락", MISROUTED: "잘못 분류", SINGLE: "단일", FOUND: "확인",
      MODEL: "모델", VARIANT: "변이", MECHANISM: "기전", CAVEAT: "주의사항", NONDISCRIMINATING: "구분 부족",
    };
    let normalized = raw;
    Object.entries(phrases).forEach(([key, value]) => { normalized = normalized.replaceAll(key, value); });
    const translated = normalized.split("_").filter(Boolean).map((token) => tokenLabels[token] || (/[가-힣]/.test(token) ? token : "추가 검토")).filter((token, index, all) => index === 0 || token !== all[index - 1]);
    return `외부 감사 · ${translated.join(" · ") || "추가 검토 필요"}`;
  }

  function findingReasonLine(group = {}) {
    const messages = [...new Set((group.records || [])
      .map((item) => String(item?.message || "").replace(/\s+/g, " ").trim())
      .filter(Boolean))];
    if (!messages.length) return "구체적 감사 사유가 원문 기록에 없습니다.";
    const first = messages[0];
    const clipped = first.length > 220 ? `${first.slice(0, 217)}…` : first;
    return messages.length > 1
      ? `${clipped} · 그 밖의 서로 다른 사유 ${compactNumber(messages.length - 1)}건`
      : clipped;
  }

  function findingGroupSummary(group = {}) {
    const records = group.records || [];
    const candidateCount = new Set(records.map((item) => item.artifact_id).filter(Boolean)).size;
    if (group.code === "ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION") {
      return `모델 예측 안전성 신호 · ${candidateCount || records.length}개 후보 확인 · 확증적 제외 아님 · 후속 실험 필요 조건부 경고`;
    }
    const reviewerSummaries = {
      COUNTER_EVIDENCE_NOT_FOUND: "Europe PMC 반증 검색은 완료됐지만 명시적 한계·반대 결과를 담은 기록을 찾지 못함",
      COUNTER_EVIDENCE_SEARCH_UNAVAILABLE: "반증 근거 소스에 접근하지 못해 반대 결과 유무를 확인할 수 없음",
      REVIEWER_PRIMARY_SOURCE_SUPPORT_INSUFFICIENT: "연결된 1차 연구가 빈도·예후 수치를 직접 입증하지 못함 · 원문 근거 보강 필요",
      REVIEWER_PRIMARY_RESULT_SUPPORT_INSUFFICIENT: "핵심 결과를 직접 뒷받침하는 1차 연구가 부족함 · 결과별 근거 재연결 필요",
      REVIEWER_TRACK_ATTRIBUTION_MISCHARACTERIZED: "Open Targets 연관성 발굴과 트랙 재배정 기록을 문헌에서 직접 도출한 것처럼 설명함 · provenance 정정 필요",
      REVIEWER_TRACK_PROVENANCE_MISCHARACTERIZED: "실제 표적 발굴·트랙 배정 기록과 화면에 설명된 출처가 다름 · 저장된 경로 기준으로 정정 필요",
      REVIEWER_COUNTER_EVIDENCE_SCOPE_MISMATCH: "약물별 내성 자료가 FLT3–AML 기본 연관성을 직접 반박하는 근거로 연결됨 · 주장별 재분류 필요",
      REVIEWER_COUNTER_EVIDENCE_LINKAGE_NONDISCRIMINATING: "반증 레코드가 어떤 주장에 적용되는지 구분되지 않음 · 주장별 반증 연결 필요",
      REVIEWER_FLT3_VARIANT_AND_MODEL_CONTEXT_UNDERSPECIFIED: "FLT3-ITD·FLT3 변이 맥락과 MOLM-13 의존성, FLT3 야생형 비교군 정의 부족",
      REVIEWER_FALSIFICATION_ENDPOINT_NOT_OPERATIONALIZED: "30% 개선의 측정 지표·방향·노출 시간·농도·비교군·표적 결합 기준이 구체적이지 않음",
      REVIEWER_SOURCE_ROUTING_RECORD_INCONSISTENCY: "Pharos·맥락 소스의 조회 결과와 가설 패킷에 남은 근거 레코드가 서로 일치하지 않음",
      REVIEWER_AML_SUBTYPE_OVERGENERALIZATION: "특정 AML 분자 아형에서 확인된 결과를 전체 AML에 적용함 · 대상 환자·변이 범위 한정 필요",
      REVIEWER_INHIBITION_DIRECTION_OPPOSES_TARGET_BIOLOGY: "TET2·CEBPA 기능 저하가 백혈병성 표현형을 촉진한다는 근거와 억제 치료 전략이 상충함 · 긍정적 치료 기전 근거 필요",
      REVIEWER_DNMT3A_DIRECTION_NOT_ESTABLISHED: "DNMT3A 변이의 AML 연관성만으로 억제 치료 이득을 결론 낼 수 없음 · 대립유전자·기능 맥락별 작용 방향 근거 필요",
      REVIEWER_CHRONIC_SAFETY_REQUIREMENT_UNADDRESSED: "정상 조혈 전구세포 비교, 장기 노출, 분화 예비력 평가가 없어 만성 안전성 요구를 충족하지 못함",
      REVIEWER_CEBPA_DRUGGABILITY_CONSTRAINT_MISMATCH: "직접 결합체·포켓·리간드 결합 구조·임상 모달리티 근거 없이 CEBPA를 약물화 가능 표적으로 유지함",
      REVIEWER_COUNTER_EVIDENCE_RELEVANCE_MISMATCH: "연결된 내성·대사 자료가 해당 치료 주장을 직접 반박하지 않음 · 주장별 관련성 재검토 필요",
      REVIEWER_SUPPORT_LINKS_INCLUDE_NON_SUPPORTIVE_RECORDS: "일부 연결 문헌이 제시된 치료 기전이나 억제 효과를 직접 지지하지 않음 · 지지 근거 재분류 필요",
      REVIEWER_NOVELTY_PRIORITY_NOT_REFLECTED: "신규 표적 우선 요청과 달리 기존 AML 반복 변이 유전자가 중심으로 선택됨 · 우선순위 반영 근거 필요",
      REVIEWER_CELL_LINE_CONTEXT_MISROUTED_AS_SINGLE_CELL: "AML cell lines를 단일세포 요구로 오인해 HCA·CELLxGENE로 라우팅함 · 세포주 맥락 소스로 정정 필요",
      ASSAY_VARIANT_EXPLICIT_MISMATCH: "가설에 선언된 표적 변이와 ChEMBL 활성 assay의 명시 변이가 다름 · 후보는 근거를 바꾸지 않고 조건부 한계로 유지",
      ASSAY_VARIANT_CONTEXT_UNKNOWN: "가설에는 변이가 있지만 ChEMBL assay에는 변이가 기록되지 않음 · wild-type 또는 일치로 추정하지 않음",
      ASSAY_VARIANT_CLASS_ONLY: "변이 유형 이름은 같지만 정확한 삽입 서열·위치·실험 구성체는 미확인 · 원문 조건 자동 보강 대상",
      ASSAY_VARIANT_PARTIAL_OVERLAP: "일부 변이명만 겹치거나 복합 변이의 동시 존재가 미확인 · 원문 조건 자동 보강 대상",
      ASSAY_VARIANT_ANNOTATION_ONLY: "변이명 표기는 일치하지만 동일 서열·실험 구성체를 보장하지 않음 · 원문 조건 자동 보강 대상",
      PROSPECTIVE_FALSIFICATION_ATTESTATION_INVALID: "미실험을 지지·실패·반증으로 해석할 수 없음 · 원자료 보존 후 상태 계약 재생성 필요",
      ASSAY_VARIANT_ALIGNMENT_ATTESTATION_MISMATCH: "저장된 assay–가설 변이 판정을 원자료에서 재현할 수 없음 · 생산·감사 기록 대조 필요",
    };
    if (reviewerSummaries[group.code]) return reviewerSummaries[group.code];
    if (records.length === 1) return `감사 사유: ${findingReasonLine(group)} · 원문 감사 메시지는 상세 보기에서 보존`;
    return `감사 사유: ${findingReasonLine(group)} · 동일 기준 ${records.length}건 · 후보별 원문·실행 위치 상세 기록`;
  }

  function findingGroupHtml(group, gate = state.selectedGate) {
    const unresolved = (group.records || []).filter((item) => !item.resolved).length;
    const advisory = (group.records || []).some((item) => item.confirmation_status === "unconfirmed");
    return `
      <div class="finding-item" role="button" tabindex="0" aria-label="${esc(`${findingGroupTitle(group)} ${group.records.length}건 상세 보기`)}" data-context-kind="finding" data-context-id="${esc(`${gate}:${group.key}`)}">
        <span class="finding-severity ${esc(group.severity || "INFO")}"></span>
        <div><b>${esc(findingGroupTitle(group))} <span>(${compactNumber(group.records.length)}건)</span></b><p>${esc(findingGroupSummary(group))}</p>
        <div class="finding-meta">${esc(group.code)} · ${esc(findingCategoryLabel(group.category))}${advisory ? " · 검증 전 리뷰 의견 · 단독 재작업 없음" : ""}${unresolved ? ` · 미해결 ${compactNumber(unresolved)}건` : " · 해결됨"} · 상세 보기</div></div>
      </div>`;
  }

  function findingCategoryLabel(value) {
    return ({
      scientific: "과학적 타당성",
      safety: "안전성",
      evidence_provenance: "근거 출처·연결",
      model_appropriateness: "모델·표적 적합성",
      process: "실행 과정",
      audit: "감사",
    })[String(value || "audit").toLowerCase()] || "감사 세부 분류";
  }

  function reviewerAuthorityHtml(finding = {}) {
    if (!finding.confirmation_status) return "";
    const status = finding.confirmation_status === "unconfirmed"
      ? "검증 전 참고 의견 · 이 의견만으로 재작업하지 않습니다."
      : "독립 규칙 검증 결과가 연결된 의견";
    const ids = Array.isArray(finding.confirmation_finding_ids) ? finding.confirmation_finding_ids : [];
    return `<p class="reviewer-authority">원문 등급 ${esc(finding.reported_severity || "미기록")} → 판정 반영 ${esc(finding.effective_severity || finding.severity || "미기록")}<br>${esc(status)}${ids.length ? `<br>확인 기록: ${ids.map(esc).join(" · ")}` : ""}</p>`;
  }

  function findingContextHtml(group, gate, snapshot = {}) {
    const candidates = snapshot?.candidate_packet?.candidates || [];
    const candidateName = (finding) => candidates.find((item) => String(item.candidate_id || "") === String(finding.artifact_id || ""))?.name
      || finding.artifact_id
      || "연결 후보 미기록";
    const failedNodes = [...new Set((group.records || []).map((item) => item.failed_node).filter(Boolean))];
    const rawDetails = (group.records || []).map((finding, index) => `
      <details data-disclosure="finding-raw:${esc(finding.finding_id || `${group.key}:${index}`)}">
        <summary>${esc(candidateName(finding))} · ${esc(finding.severity || "INFO")}</summary>
        <p>${esc(finding.message || "원문 메시지 미기록")}</p>
        ${reviewerAuthorityHtml(finding)}
        <div class="finding-meta">${esc(finding.finding_id || "finding ID 미기록")}${finding.failed_node ? ` · ${esc(finding.failed_node)}` : ""}${finding.resolved ? " · 해결됨" : " · 미해결"}</div>
      </details>`).join("");
    return `
      <section class="context-section"><h4>결과</h4><p class="context-claim">${esc(findingGroupTitle(group))} · ${compactNumber(group.records.length)}건</p></section>
      <section class="context-section"><h4>감사 사유 요약</h4><p class="context-claim">${esc(findingReasonLine(group))}</p></section>
      <section class="context-section"><h4>해석과 추가 검토 항목</h4><p>${esc(findingGroupSummary(group))}</p></section>
      <section class="context-section evidence-group"><h4>원문 감사 기록 <span>${compactNumber(group.records.length)}</span></h4><div class="evidence-record-list">${rawDetails}</div></section>
      <section class="context-section provenance-chain"><h4>출처 연결</h4><code>${esc(gate.toUpperCase())} → ${esc(group.code)} → GateVerdict</code></section>
      <section class="context-section"><h4>실행 추적</h4><p>${failedNodes.length ? esc(failedNodes.join(" · ")) : "실행 노드 미기록"}</p><code class="audit-evidence-ids">${esc((group.records || []).map((item) => item.finding_id || "ID 미기록").join(" · "))}</code></section>`;
  }

  function reworkHtml(order) {
    if (!order) return "";
    const actions = order.required_actions || [];
    return `<div class="rework-box"><b>선택적 재작업 → ${esc(String(order.target_stage || "").toUpperCase())}</b>
      ${actions.length ? `<ul>${actions.map((action) => `<li>${esc(action)}</li>`).join("")}</ul>` : ""}</div>`;
  }

  function renderCounts() {
    const hypotheses = state.snapshot?.hypothesis_packet?.hypotheses || [];
    const candidates = state.snapshot?.candidate_packet?.candidates || [];
    const findings = [
      ...(state.snapshot?.g0_verdict?.findings || []),
      ...(state.snapshot?.g1_verdict?.findings || []),
    ];
    const admission = ma1AdmissionProjection(state.snapshot?.hypothesis_packet || {});
    $("#hypothesisCount").textContent = admission.enabled
      ? `제안 ${admission.submittedCount} · MA3 승인 ${admission.approvedCount}`
      : `Retained ${hypotheses.length}`;
    $("#candidateCountBadge").textContent = candidates.length;
    $("#findingCount").textContent = findings.length;
  }

  function selectArtifactTab(tab) {
    state.selectedTab = normalizeArtifactTab(tab);
    updateSelectedTabs($$(".artifact-tabs [role=tab]"), "tab", state.selectedTab);
    renderArtifact();
    const artifactContent = $("#artifactContent");
    if (artifactContent) artifactContent.scrollTop = 0;
    $(".artifact-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function normalizeArtifactTab(tab) {
    const value = String(tab || "overview").toLowerCase();
    if (["brief", "overview"].includes(value)) return "overview";
    if (["raw", "technical"].includes(value)) return "report";
    return value;
  }

  function updateSelectedTabs(tabs, dataKey, value) {
    let selectedTab = null;
    tabs.forEach((tab) => {
      const selected = tab.dataset[dataKey] === value;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected) selectedTab = tab;
    });
    const panel = dataKey === "tab" ? $("#artifactContent") : dataKey === "gate" ? $("#verdictContent") : null;
    if (panel && selectedTab?.id) panel.setAttribute("aria-labelledby", selectedTab.id);
  }

  function renderArtifact() {
    const root = $("#artifactContent");
    if (!root) return;
    captureDisclosureState(root);
    state.selectedTab = normalizeArtifactTab(state.selectedTab);
    updateSelectedTabs($$(".artifact-tabs [role=tab]"), "tab", state.selectedTab);
    if (!state.snapshot) {
      root.innerHTML = '<div class="artifact-empty">연구를 실행하면 단계별 산출물이 표시됩니다.</div>';
      return;
    }
    const renderers = {
      overview: renderOverviewArtifact,
      report: renderCombinedRunReportArtifact,
      hypotheses: renderHypothesesArtifact,
      plan: renderPlanArtifact,
      candidates: renderCandidatesArtifact,
      audits: renderAuditsArtifact,
    };
    root.innerHTML = (renderers[state.selectedTab] || renderOverviewArtifact)();
    wireMoleculeStructures(root);
    restoreDisclosureState(root);
    wireTechnicalData(root);
  }

  function wireTechnicalData(root) {
    const disclosure = $("[data-technical-json]", root);
    const output = $("[data-raw-json]", disclosure || root);
    if (!disclosure || !output) return;
    const populate = () => {
      if (!disclosure.open || output.dataset.loaded === "true") return;
      output.textContent = JSON.stringify(state.snapshot, null, 2);
      output.dataset.loaded = "true";
    };
    disclosure.addEventListener("toggle", populate);
    populate();
  }

  function wireMoleculeStructures(root) {
    $$(".molecule-structure-image", root).forEach((image) => {
      const figure = image.closest(".molecule-structure");
      const markUnavailable = () => figure?.classList.add("is-unavailable");
      image.addEventListener("error", markUnavailable, { once: true });
      if (image.complete && image.naturalWidth === 0) markUnavailable();
    });
  }

  function captureDisclosureState(root) {
    $$("details[data-disclosure]", root).forEach((details) => {
      const key = details.dataset.disclosure;
      if (!key) return;
      if (details.open) state.openDisclosures.add(key);
      else state.openDisclosures.delete(key);
    });
  }

  function restoreDisclosureState(root) {
    $$("details[data-disclosure]", root).forEach((details) => {
      details.open = state.openDisclosures.has(details.dataset.disclosure);
    });
  }

  function gateVerdictSummaryKo(verdict = {}) {
    const findings = (verdict.findings || []).filter((item) => item.resolved !== true);
    const high = findings.filter((item) => ["HIGH", "CRITICAL"].includes(String(item.severity || "").toUpperCase())).length;
    const groups = groupFindings(findings);
    const labels = groups.slice(0, 4).map((group) => findingGroupTitle(group));
    return `${verdictLabel(verdict.verdict)} · 미해결 ${compactNumber(findings.length)}건${high ? ` · 중요 ${compactNumber(high)}건` : ""}${labels.length ? ` · ${labels.join(" · ")}` : ""}`;
  }

  function runReportConclusion(snapshot, intake, portfolio) {
    const stage = String(snapshot?.stage || "created").toLowerCase();
    const g0 = String(snapshot?.g0_verdict?.verdict || "").toUpperCase();
    const g1 = String(snapshot?.g1_verdict?.verdict || "").toUpperCase();
    const decision = String(snapshot?.human_decision?.decision || "").toUpperCase();
    if (stage === "failed" || snapshot?.error) {
      return {
        className: "critical",
        label: "실행 실패",
        text: "실행이 정상 종료되지 않았습니다. 실패 전까지 저장된 산출물은 참고할 수 있지만 후보 승인 근거로 사용할 수 없습니다.",
      };
    }
    if (intake.requiresInputCorrection) {
      return {
        className: "critical",
        label: "연구 입력 확인 필요",
        text: "질환 식별 또는 입력 필드가 확정되지 않아 후보 생성 중단 상태이며 결과를 과학적 후보 결론으로 제시하지 않습니다. 입력을 수정한 뒤 새 실행이 필요합니다.",
      };
    }
    if (intake.verificationUnavailable) {
      return {
        className: "warning",
        label: "외부 질환 검증 불가",
        text: "질환 확인 소스에 접근하지 못해 fail-closed 후보 생성 중단으로 판단을 보류했습니다. 저장된 중간 기록은 감사용이며 후보 승인 근거가 아닙니다.",
      };
    }
    if (intake.demoFixture) {
      return {
        className: "warning",
        label: "Demo fixture 실행 · 과학 결론 아님",
        text: "질환 확인과 연결 근거가 데모·테스트 자료에 의존한 실행입니다. G0/G1 표시와 관계없이 실제 연구 후보 승인 근거로 사용할 수 없습니다.",
      };
    }
    if (String(snapshot?.candidate_packet?.backend_kind || "").toLowerCase() === "fixture") {
      return {
        className: "warning",
        label: "Science fixture 실행 · 후보 승인 불가",
        text: "후보 생성·계산이 fixture 과학 backend에서 수행되었습니다. G1 또는 HumanDecision 표시와 관계없이 실제 연구 후보 추천이나 승인 근거로 사용할 수 없습니다.",
      };
    }
    if (portfolio.abstainRequired) {
      return {
        className: "warning",
        label: "표적 포트폴리오 미완료",
        text: "실행 자격 표적 일부가 결론에 도달하지 못했습니다. G1 표시와 관계없이 전체 후보군을 승인 가능한 포트폴리오로 해석할 수 없습니다.",
      };
    }
    if (decision === "REJECT") {
      return {
        className: "critical",
        label: "연구자 반려",
        text: "자동 감사 이후 연구자가 후보 진행을 반려했습니다. 기록된 후보는 후속 실험 대상으로 승인되지 않았습니다.",
      };
    }
    if (decision === "HOLD") {
      return {
        className: "warning",
        label: "연구자 보류",
        text: "자동 감사 이후 연구자가 결정을 보류했습니다. 추가 검토가 끝날 때까지 후보 진행 승인이 아닙니다.",
      };
    }
    if (decision === "APPROVE") {
      const selected = snapshot?.human_decision?.selected_candidate_ids?.length || 0;
      return {
        className: "pass",
        label: "연구자 승인 기록됨",
        text: `G1 감사 적합 후보 중 ${selected}건이 연구자에 의해 후속 검토 대상으로 선택되었습니다. 이 결정은 임상 효능·안전성 확정이 아니라 실험 검토 진행 승인입니다.`,
      };
    }
    if (g1 === "PASS") {
      return {
        className: "pass",
        label: "자동 감사 통과 · 연구자 결정 대기",
        text: "G1 감사 기준은 충족했지만 연구자 결정이 아직 없습니다. G1 PASS는 후보 승인이나 임상적 유효성·안전성 확정을 뜻하지 않습니다.",
      };
    }
    if (["ABSTAIN", "BLOCK", "REVISE", "REPLAN"].includes(g1)) {
      return {
        className: g1 === "BLOCK" ? "critical" : "warning",
        label: "G1 후보·과정 감사 미충족으로 중단",
        text: `${gateVerdictSummaryKo(snapshot.g1_verdict)}. 후보·과정 감사의 보완 조건을 충족하지 못해 연구자 결정 단계로 진행하지 않았습니다.`,
      };
    }
    if (g0 === "PASS") {
      return {
        className: "pending",
        label: "후보 평가 진행 중",
        text: "가설 감사 G0는 통과했으며 후보 실행 또는 G1 감사를 기다리고 있습니다. 현재 결과는 중간 산출물입니다.",
      };
    }
    if (["ABSTAIN", "BLOCK", "REVISE", "REPLAN"].includes(g0)) {
      const status = runStatusPresentation(snapshot);
      return {
        className: g0 === "BLOCK" ? "critical" : "warning",
        label: status.label,
        text: status.detail,
      };
    }
    return {
      className: "pending",
      label: stageLabel(stage),
      text: snapshot?.status_message || "실행이 진행 중이며 아직 감사 가능한 최종 결론이 없습니다.",
    };
  }

  function runReportEvidenceSummary(snapshot = {}) {
    const hypotheses = snapshot?.hypothesis_packet?.hypotheses || [];
    const candidates = snapshot?.candidate_packet?.candidates || [];
    const linked = [
      ...hypotheses.flatMap((item) => item.evidence || []),
      ...candidates.flatMap((item) => item.evidence || []),
    ];
    const unique = new Map();
    linked.forEach((item, index) => {
      const key = item.evidence_id
        || `${item.source || "source"}:${item.external_id || item.title || index}`;
      if (!unique.has(key)) unique.set(key, item);
    });
    const summary = { linked: linked.length, unique: unique.size, live: 0, cache: 0, local: 0, fixture: 0, unknown: 0, sources: new Set() };
    unique.forEach((item) => {
      summary.sources.add(item.source || "미기록");
      const metadata = item.metadata || {};
      const provenance = String(metadata.provenance_class || metadata.provenance || "").toLowerCase();
      if (isFixtureEvidence(item)) summary.fixture += 1;
      else if (provenance.includes("cache") || metadata.cache_hit === true) summary.cache += 1;
      else if (provenance.includes("local") || provenance.includes("dataset")) summary.local += 1;
      else if (provenance.includes("live") || metadata.live === true) summary.live += 1;
      else summary.unknown += 1;
    });
    return summary;
  }

  function runReportDate(value) {
    if (!value) return "미기록";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "미기록";
    return new Intl.DateTimeFormat("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(date);
  }

  function runReportAuditCard(label, verdict) {
    if (!verdict) {
      return `<article class="report-audit-card is-pending"><header><b>${esc(label)}</b><span>대기</span></header><p>이 감사 단계의 저장된 판정이 없습니다.</p></article>`;
    }
    const findings = verdict.findings || [];
    const validations = verdict.validations || [];
    const unresolved = findings.filter((item) => item.resolved !== true).length;
    const high = findings.filter((item) => ["HIGH", "CRITICAL"].includes(String(item.severity || "").toUpperCase())).length;
    const passed = validations.filter((item) => validationStatus(item) === "PASS").length;
    const reviewer = verdict.reviewer_invoked === true
      ? `외부 검토 모델 호출 · ${verdict.reviewer_model || "모델명 미기록"}`
      : "리뷰어 모델 미호출";
    const metaCritic = verdict.meta_critic_invoked === true ? " · 메타 비평 호출" : "";
    return `<article class="report-audit-card is-${esc(String(verdict.verdict || "pending").toLowerCase())}">
      <header><b>${esc(label)}</b><span>${esc(verdictLabel(verdict.verdict))}</span></header>
      <p>${esc(gateVerdictSummaryKo(verdict))}</p>
      <dl>
        <div><dt>검사항목</dt><dd>${compactNumber(passed)}/${compactNumber(validations.length)} 통과</dd></div>
        <div><dt>감사 항목</dt><dd>${compactNumber(findings.length)}건 · 미해결 ${compactNumber(unresolved)} · 중요 ${compactNumber(high)}</dd></div>
        <div><dt>판정 경로</dt><dd>${esc(reviewer)}${esc(metaCritic)}</dd></div>
      </dl>
      ${verdict.rework_order ? `<div class="report-audit-action"><b>재작업</b><span>${esc(verdict.rework_order.target_stage || "미기록")} · ${compactNumber(verdict.rework_order.required_actions?.length || 0)}개 조치</span></div>` : ""}
    </article>`;
  }

  function runReportIssueRows(snapshot, intake, portfolio) {
    const rows = [];
    const add = (severity, title, detail, count = 1) => {
      if (!detail && !title) return;
      rows.push({ severity: String(severity || "INFO").toUpperCase(), title, detail, count });
    };
    if (snapshot?.error) add("CRITICAL", "실행 오류", snapshot.error);
    if (String(snapshot?.mode || "").toLowerCase() === "mock") {
      add("MEDIUM", "LLM MOCK · 실제 모델 추론 아님", "저장된 템플릿·결정론적 경로를 사용한 실행입니다. 다음 실제 LLM 연결 실행과 구분해 해석하십시오.");
    }
    if (String(snapshot?.candidate_packet?.backend_kind || "").toLowerCase() === "fixture") {
      add("HIGH", "SCIENCE FIXTURE · 실제 과학 실행 결과 아님", "fixture 후보·도구 결과는 UI·워크플로 검증용이며 실제 후보 승인 근거가 아닙니다.");
    }
    intake.fieldMismatches.forEach((item) => add("HIGH", item.code || "입력 불일치", item.message));
    intake.warnings.forEach((item) => add("MEDIUM", item.code || "입력 경고", item.message));
    if (intake.verificationUnavailable) add("HIGH", "외부 질환 검증 불가", intake.explanation || intake.reasonCodes.join(" · ") || "검증 소스 상태를 확인하십시오.");
    const findings = [
      ...(snapshot?.g0_verdict?.findings || []),
      ...(snapshot?.g1_verdict?.findings || []),
    ];
    groupFindings(findings).forEach((group) => {
      const unresolved = group.records.filter((item) => item.resolved !== true);
      if (!unresolved.length) return;
      add(group.severity, findingGroupTitle(group), findingGroupSummary({ ...group, records: unresolved }), unresolved.length);
    });
    const safetyFlags = new Map();
    (snapshot?.candidate_packet?.candidates || []).forEach((candidate) => {
      (candidate.safety_flags || []).forEach((flag) => {
        const key = String(flag || "미기록 safety flag");
        if (!safetyFlags.has(key)) safetyFlags.set(key, new Set());
        safetyFlags.get(key).add(candidate.candidate_id || candidate.name || "candidate");
      });
    });
    safetyFlags.forEach((candidateIds, flag) => {
      add("MEDIUM", "후보 안전성 신호", `${flag} · ${candidateIds.size}개 후보`, candidateIds.size);
    });
    if (portfolio.reasonCodes.length) {
      add("HIGH", "포트폴리오 판단 보류 사유", portfolio.reasonCodes.join(" · "), portfolio.reasonCodes.length);
    }
    const incompleteTargets = portfolio.executions.filter((item) => ["partial", "unavailable", "failed"].includes(String(item.status || "").toLowerCase()));
    incompleteTargets.forEach((item) => {
      add("HIGH", `${item.target?.symbol || item.target_key || "표적"} 실행 미완료`, (item.reason_codes || []).join(" · ") || item.detail || item.status);
    });
    const failedCalls = (snapshot?.candidate_packet?.tool_calls || []).filter((item) => {
      const status = String(item.execution_status || "").toLowerCase();
      return item.success === false || ["failed", "unavailable"].includes(status);
    });
    if (failedCalls.length) add("HIGH", "과학 도구 실행 실패·불가", `${failedCalls.length}건 · Technical Data의 요청 시점 trace 확인`, failedCalls.length);
    return rows;
  }

  function runReportTargetCoverageHtml(packet, portfolio) {
    const executions = portfolio.executions || [];
    if (!packet) return '<div class="report-empty">MA2 후보 실행 기록이 아직 없습니다.</div>';
    if (!executions.length) {
      return `<div class="report-note">${esc(portfolioCoverageText(packet, { includeApproved: true }))} · 이전 스냅샷에는 표적별 branch 상세가 없을 수 있습니다.</div>`;
    }
    return `<div class="report-table-wrap"><table class="report-table report-target-table">
      <caption>G0 승인 표적별 MA2 실행 범위와 결론</caption>
      <thead><tr><th scope="col">표적</th><th scope="col">실행 자격</th><th scope="col">선택 경로</th><th scope="col">결론 상태</th><th scope="col">후보</th><th scope="col">근거·사유</th></tr></thead>
      <tbody>${executions.map((item) => {
        const status = String(item.status || "미기록");
        const reasons = (item.reason_codes || []).join(" · ") || item.detail || (status === "completed" ? "실행 완료" : status === "no_candidates" ? "검색 완료 · 기준 충족 후보 없음" : "상세 미기록");
        return `<tr>
          <td><b>${esc(item.target?.symbol || item.target_key || "—")}</b><small>${esc(item.track || "")}</small></td>
          <td>${item.execution_eligible ? "예" : "아니오"}${item.attempted ? " · 시도됨" : ""}</td>
          <td>${esc(item.route ? routeLabel(item.route) : "미선택")}</td>
          <td><span class="report-state is-${esc(status)}">${esc(status.replaceAll("_", " "))}</span></td>
          <td>발견 ${compactNumber(item.discovered_candidate_count || 0)} · 적합 ${compactNumber(item.eligible_candidate_count || 0)} · 제외 ${compactNumber(item.excluded_candidate_count || 0)}</td>
          <td>${esc(reasons)}</td>
        </tr>`;
      }).join("")}</tbody>
    </table></div>`;
  }

  function hypothesisEvidenceCounts(hypothesis = {}) {
    const counts = { support: 0, counter: 0, context: 0, other: 0 };
    (hypothesis.evidence || []).forEach((item) => {
      const role = String(item.supports || "context").toLowerCase();
      if (Object.prototype.hasOwnProperty.call(counts, role)) counts[role] += 1;
      else counts.other += 1;
    });
    return counts;
  }

  function hypothesisNarrative(hypothesis = {}) {
    const mechanism = hypothesis.mechanism_ko || hypothesis.mechanism || hypothesis.claims?.[0]?.text_ko || hypothesis.claims?.[0]?.text || "기전 가설 미기록";
    const falsification = hypothesis.falsification_test_ko || hypothesis.falsification_test || "반증 기준 미기록";
    const claims = (hypothesis.claims || []).map((claim) => ({ ...claim, displayText: claim.text_ko || claim.text || "주장 미기록" }));
    const language = /[가-힣]/.test([mechanism, falsification, ...claims.map((claim) => claim.displayText)].join(" ")) ? "한국어 서술" : "영문 저장 기록";
    return { mechanism, falsification, claims, language };
  }

  function hypothesisSourceToolBadges(hypothesis = {}) {
    const evidence = hypothesis.evidence || [];
    const sources = [...new Set(evidence.map((item) => item.source).filter(Boolean))];
    // A global MA1 call is not automatically attributable to every
    // hypothesis.  Tool badges are emitted only when an EvidenceRef itself
    // carries an explicit scientific-tool field.
    const tools = [...new Set(evidence.flatMap((item) => {
      const metadata = item?.metadata || {};
      return [
        metadata.scientific_tool_name,
        metadata.scientific_tool,
        ...(Array.isArray(metadata.scientific_tools) ? metadata.scientific_tools : []),
      ].filter(Boolean);
    }))];
    const sourceBadges = sources.slice(0, 12).map((source) => `<span class="hypothesis-execution-badge is-source">DATA · ${esc(source)}</span>`);
    const toolBadges = tools.slice(0, 6).map((tool) => `<span class="hypothesis-execution-badge is-tool">TOOL · ${esc(toolDisplayName(tool))}</span>`);
    const overflow = Math.max(0, sources.length - 12) + Math.max(0, tools.length - 6);
    if (overflow) toolBadges.push(`<span class="hypothesis-execution-badge">+${compactNumber(overflow)}</span>`);
    return `<div class="hypothesis-execution-badges" aria-label="연결된 데이터 소스와 실행 도구">${[...sourceBadges, ...toolBadges].join("") || '<span class="hypothesis-execution-badge">연결 기록 없음</span>'}</div>`;
  }

  function hypothesisNarrativeBody(hypothesis = {}, options = {}) {
    const narrative = hypothesisNarrative(hypothesis);
    const claims = options.claimLimit == null ? narrative.claims : narrative.claims.slice(0, options.claimLimit);
    const uncalibratedAdmission = hypothesis.target?.metadata?.ma3_admission?.policy === "ma3-source-bound-v1";
    return `<div class="hypothesis-narrative-body">
      <div><b>기전 가설</b><p>${esc(narrative.mechanism)}</p></div>
      <div><b>검증 주장</b>${claims.length ? `<ol>${claims.map((claim) => `<li>${esc(claim.displayText)}${uncalibratedAdmission ? '<small>신뢰도 미보정 · 확률 아님</small>' : claim.confidence != null ? `<small>기록 confidence ${formatScore(claim.confidence)}</small>` : ""}</li>`).join("")}</ol>` : '<p>주장 미기록</p>'}</div>
      <div><b>반증 기준</b><p>${esc(narrative.falsification)}</p></div>
    </div>`;
  }

  function runReportHypothesesHtml(snapshot, intake) {
    const hypotheses = snapshot?.hypothesis_packet?.hypotheses || [];
    const diseaseVerified = !intake.requiresInputCorrection && !intake.verificationUnavailable && !intake.demoFixture;
    const eligible = diseaseVerified
      ? hypotheses.filter((item) => !hypothesisProvenance(item).fixtureOnly)
      : [];
    if (!diseaseVerified) {
      return '<div class="report-empty is-warning">질환 확인이 완료되지 않아 저장된 가설을 주요 결과로 제시하지 않습니다. 전체 기록은 Technical Data에서 보존됩니다.</div>';
    }
    if (snapshot?.hypothesis_packet?.metadata?.admission_policy === "ma3-source-bound-v1" || snapshot?.hypothesis_packet?.metadata?.ma3_admission_report || snapshot?.hypothesis_packet?.provisional_hypotheses?.length) {
      return renderMa3AdmissionHypotheses(snapshot.hypothesis_packet, { report: true });
    }
    if (!eligible.length) return '<div class="report-empty">과학적 결과로 제시할 가설이 없습니다.</div>';
    const ordered = [...eligible].sort((a, b) => Number(a.rank ?? 999) - Number(b.rank ?? 999) || Number(b.score || 0) - Number(a.score || 0));
    return `<div class="report-hypothesis-records" role="list" aria-label="저장 순위에 따른 상위 가설">${ordered.slice(0, 8).map((item, index) => {
      const counts = hypothesisEvidenceCounts(item);
      const narrative = hypothesisNarrative(item);
      return `<article class="report-hypothesis-record" role="listitem">
        <header><span>#${esc(item.rank ?? index + 1)}</span><div><b>${esc(item.target?.symbol || "—")}</b><small>${esc(item.target?.target_name || item.target?.uniprot_id || "표적 정보 미기록")}</small></div><div><span>${esc(item.track || "경로 미기록")}</span><span>${esc(directionLabel(item.intervention_direction))}</span><span>${esc(narrative.language)}</span></div><strong>${formatScore(item.score)}</strong></header>
        ${hypothesisNarrativeBody(item)}
        <footer><div class="hypothesis-evidence-counts"><span><b>${compactNumber(counts.support)}</b>지지</span><span><b>${compactNumber(counts.counter)}</b>반증</span><span><b>${compactNumber(counts.context)}</b>맥락</span></div>${hypothesisSourceToolBadges(item)}</footer>
      </article>`;
    }).join("")}</div>${ordered.length > 8 ? `<p class="report-footnote">상위 8건 표시 · 전체 ${compactNumber(ordered.length)}건은 Hypotheses 탭에서 확인</p>` : ""}`;
  }

  function runReportCandidatesHtml(snapshot, intake, portfolio) {
    const packet = snapshot?.candidate_packet || null;
    const candidates = packet?.candidates || [];
    const excluded = candidates.filter((item) => item.status === "excluded");
    const eligible = sortCandidatesForReview(candidates.filter((item) => item.status !== "excluded"), packet || {});
    const g1 = String(snapshot?.g1_verdict?.verdict || "").toUpperCase();
    const decision = String(snapshot?.human_decision?.decision || "").toUpperCase();
    const selectedIds = new Set(snapshot?.human_decision?.selected_candidate_ids || []);
    const displayAllowed = !intake.requiresInputCorrection && !intake.verificationUnavailable && !intake.demoFixture;
    const scienceFixture = String(packet?.backend_kind || "").toLowerCase() === "fixture";
    const auditReady = displayAllowed && !scienceFixture && g1 === "PASS" && !portfolio.abstainRequired;
    if (!packet) return '<div class="report-empty">후보 패킷이 아직 없습니다.</div>';
    if (!displayAllowed) {
      return '<div class="report-empty is-warning">질환 확인이 완료되지 않아 저장된 후보를 순위 결과로 제시하지 않습니다. 전체 기록은 Technical Data에서 보존됩니다.</div>';
    }
    if (!eligible.length) {
      return `<div class="report-empty">승인 검토 가능한 후보가 없습니다. 제외 후보 ${compactNumber(excluded.length)}건의 사유는 Candidates/Audit 탭에서 확인하십시오.</div>`;
    }
    const statusText = (candidate) => {
      if (decision === "APPROVE") {
        return selectedIds.has(candidate.candidate_id)
          ? "연구자 선택"
          : "미선택 · 승인 대상 아님";
      }
      if (!auditReady) return `${candidate.status === "priority" ? "우선" : "조건부"} · 승인 불가/대기`;
      if (!decision) return `${candidate.status === "priority" ? "우선" : "조건부"} · 연구자 결정 대기`;
      return `${candidate.status === "priority" ? "우선 검토" : "조건부"} · ${decisionLabel(decision)}`;
    };
    return `<div class="report-note ${auditReady ? "" : "is-warning"}">${auditReady
      ? "G1 감사 기준을 충족한 기록 후보입니다. 순위는 후속 검토 우선순위이며 연구자 승인이나 임상적 확정을 뜻하지 않습니다."
      : "감사·실행 범위가 완료되지 않아 아래 항목은 승인 가능한 추천이 아니라 저장된 후보 기록입니다."} 제외 후보 ${compactNumber(excluded.length)}건.</div>
      <div class="report-table-wrap"><table class="report-table report-candidate-table">
        <caption>제외 후보를 분리한 후보 순위와 연구자 결정 상태</caption>
        <thead><tr><th scope="col">순위</th><th scope="col">후보</th><th scope="col">표적 / 경로</th><th scope="col">활성·결합</th><th scope="col">점수</th><th scope="col">상태</th><th scope="col">근거·경고</th></tr></thead>
        <tbody>${eligible.slice(0, 10).map((candidate) => {
          const activity = candidate.activities?.[0];
          const activityText = activity
            ? `${activity.endpoint} ${activity.relation || "="} ${activity.value} ${activity.unit}`
            : candidate.metrics?.docking_score_kcal_mol != null
            ? `Docking ${candidate.metrics.docking_score_kcal_mol} kcal/mol`
            : "측정·도킹값 미기록";
          return `<tr>
            <td class="numeric-cell">${esc(portfolioCandidateRank(candidate, packet || {}))}</td>
            <td><b>${esc(candidate.name || candidate.candidate_id || "—")}</b><small>${esc(candidate.inchikey || candidate.candidate_id || "")}</small></td>
            <td>${esc(candidate.target?.symbol || "—")}<small>${esc(routeLabel(candidate.route))}</small></td>
            <td>${esc(activityText)}</td>
            <td class="numeric-cell">${candidateRankingScore(candidate, packet || {}) == null ? "미기록" : formatScore(candidateRankingScore(candidate, packet || {}))}</td>
            <td><span class="report-state is-${esc(candidate.status || "conditional")}">${esc(statusText(candidate))}</span></td>
            <td>근거 ${compactNumber(candidate.evidence?.length || 0)}건 · 안전성 신호 ${compactNumber(candidate.safety_flags?.length || 0)}건</td>
          </tr>`;
        }).join("")}</tbody>
      </table></div>${eligible.length > 10 ? `<p class="report-footnote">상위 10건 표시 · 전체 ${compactNumber(eligible.length)}건은 Candidates 탭에서 확인</p>` : ""}`;
  }

  function renderRunReportArtifact() {
    const snapshot = state.snapshot || {};
    const brief = snapshot.brief || {};
    const constraints = brief.constraints || {};
    const hypothesisPacket = snapshot.hypothesis_packet || null;
    const candidatePacket = snapshot.candidate_packet || null;
    const intake = getIntakeAssessment(snapshot);
    const portfolio = portfolioExecutionState(candidatePacket || {});
    const conclusion = runReportConclusion(snapshot, intake, portfolio);
    const statusPresentation = runStatusPresentation(snapshot);
    const evidence = runReportEvidenceSummary(snapshot);
    const hypotheses = hypothesisPacket?.hypotheses || [];
    const candidates = candidatePacket?.candidates || [];
    const nonExcludedCandidates = candidates.filter((item) => item.status !== "excluded");
    const trace = state.callTrace?.run_id === snapshot.run_id ? state.callTrace : null;
    const calls = callCountSummary(snapshot, trace);
    const issues = runReportIssueRows(snapshot, intake, portfolio);
    const goalFocus = hypothesisPacket?.metadata?.goal_focus || {};
    const goalCueLabels = (goalFocus.cues || [])
      .map((cue) => goalCueCategoryLabel(cue.category))
      .filter(Boolean);
    const structuredPriority = constraints.discovery_priority || "auto";
    const effectivePriority = snapshot.plan_spec?.discovery_priority || structuredPriority;
    const resolved = hypothesisPacket?.metadata?.resolved_disease || {};
    const explicitScopeHtml = renderExplicitScopeIdentityTrace(hypothesisPacket || {});
    const resolvedName = resolved.disease_name || intake.raw.resolved_disease_name;
    const resolvedId = resolved.disease_id || intake.raw.resolved_disease_id;
    const diseaseResolution = resolvedName
      ? `${resolvedName}${resolvedId ? ` · ${resolvedId}` : ""}`
      : intake.verificationUnavailable
      ? "검증 불가 · 외부 소스 연결 실패"
      : intake.demoFixture
      ? "Demo fixture · 실제 질환 확인 아님"
      : intake.diseaseNeedsReview
      ? "확인 필요"
      : "미기록";
    const sourceCoverage = snapshot.plan_spec?.source_coverage || null;
    const admittedSources = sourceCoverage?.admitted_source_count
      ?? sourceCoverage?.decisions?.filter((item) => item.admitted).length
      ?? 0;
    const toolCalls = candidatePacket?.tool_calls || [];
    const toolCompleted = toolCalls.filter((item) => String(item.execution_status || "").toLowerCase() === "completed").length;
    const toolSkipped = toolCalls.filter((item) => String(item.execution_status || "").toLowerCase() === "skipped").length;
    const toolFailed = toolCalls.filter((item) => item.success === false || ["failed", "unavailable"].includes(String(item.execution_status || "").toLowerCase())).length;
    const toolNames = [...new Set(toolCalls.map((item) => item.tool_name).filter(Boolean))];
    const tokenTotal = sum(snapshot.token_usage || [], (item) => Number(item.input_tokens || 0) + Number(item.output_tokens || 0) + Number(item.reasoning_tokens || 0));
    const costTotal = sum(snapshot.token_usage || [], (item) => item.estimated_cost_usd || 0);
    const hashComplete = toolCalls.filter((item) => item.input_hash && item.output_hash).length;
    const evidenceRecords = [
      ...hypotheses.flatMap((item) => item.evidence || []),
      ...candidates.flatMap((item) => item.evidence || []),
    ];
    const evidenceWithExternalId = evidenceRecords.filter((item) => item.external_id && item.retrieved_at).length;
    const evidenceWithExcerptHash = evidenceRecords.filter((item) => item.excerpt_hash || item.metadata?.content_hash).length;
    const eventArtifactRefs = new Set((snapshot.events || []).flatMap((item) => item.artifact_ids || []));
    const decision = snapshot.human_decision;
    const depth = researchDepthPresentation(researchDepthFromSnapshot(snapshot));
    const issueHtml = issues.length
      ? `<div class="report-issue-list">${issues.slice(0, 16).map((item) => {
        const severityClass = ["CRITICAL", "HIGH"].includes(item.severity) ? "high" : item.severity === "MEDIUM" ? "medium" : "low";
        return `<article class="report-issue is-${severityClass}"><span>${esc(item.severity)}</span><div><b>${esc(item.title)}${item.count > 1 ? ` · ${compactNumber(item.count)}건` : ""}</b><p>${esc(item.detail)}</p></div></article>`;
      }).join("")}</div>${issues.length > 16 ? `<p class="report-footnote">주요 16개 그룹 표시 · 전체 원문은 Audit/Technical Data에서 확인</p>` : ""}`
      : '<div class="report-empty is-clear">저장된 미해결 입력·감사·실행 이슈가 없습니다.</div>';
    const mode = String(snapshot.mode || "미기록").toUpperCase();
    const scienceBackend = String(candidatePacket?.backend_kind || candidatePacket?.gateway_readiness?.backend_kind || "미기록").toUpperCase();
    const trustBoundary = [
      mode === "MOCK"
        ? "LLM MOCK · 실제 모델 추론이 아니라 저장 템플릿·결정론적 경로"
        : `LLM ${mode} · 이 실행 snapshot의 모델 사용 경계`,
      scienceBackend === "FIXTURE"
        ? "SCIENCE FIXTURE · 실제 과학 실행 결과가 아니며 후보 승인 불가"
        : scienceBackend === "LIVE"
        ? "SCIENCE LIVE · 이 실행에 저장된 요청 시점 ToolCallRecord 기준"
        : "SCIENCE backend 미기록 · 성공으로 간주하지 않음",
    ];
    const sourceNames = [...evidence.sources].sort((a, b) => String(a).localeCompare(String(b), "ko"));
    return `<article class="run-report" aria-labelledby="runReportTitle">
      <header class="run-report-cover">
        <div>
          <span class="run-report-kicker">DRUGPILOT · 근거 기반 실행 보고서</span>
          <h4 id="runReportTitle">${esc(brief.disease || "질환 미기록")}</h4>
          <p>${esc(brief.research_goal || snapshot.status_message || "연구 목표 미기록")}</p>
        </div>
        <div class="run-report-cover-meta">
          <span>RUN ${esc(shortId(snapshot.run_id))}</span>
          <span>${esc(runReportDate(snapshot.updated_at || snapshot.created_at))}</span>
        </div>
        <div class="run-report-badges" aria-label="실행 모드 및 상태">
          <span class="is-status" title="${esc(statusPresentation.detail)}">${esc(statusPresentation.label)}</span>
          <span>LLM ${esc(mode)}</span>
          <span>SCIENCE ${esc(scienceBackend)}</span>
          <span>${esc(depth.pill)}</span>
        </div>
      </header>

      <section class="run-report-conclusion is-${esc(conclusion.className)}" aria-labelledby="runReportConclusionTitle">
        <span>핵심 결론</span>
        <h5 id="runReportConclusionTitle">${esc(conclusion.label)}</h5>
        <p>${esc(conclusion.text)}</p>
        <small>현재 서버 상태가 아니라 이 실행에 저장된 스냅샷·감사 판정·연구자 결정 기준</small>
      </section>

      <aside class="run-report-trust-boundary" aria-label="결과 해석 경계">
        <b>해석 경계</b>
        ${trustBoundary.map((item) => `<span>${esc(item)}</span>`).join("")}
      </aside>

      <section class="run-report-metrics" aria-label="핵심 실행 수치">
        <article><strong>${compactNumber(hypotheses.length)}</strong><span>${snapshot.hypothesis_packet?.metadata?.admission_policy === "ma3-source-bound-v1" ? "MA3 승인 가설" : "가설"}</span><small>${snapshot.hypothesis_packet?.metadata?.admission_policy === "ma3-source-bound-v1" ? `MA1 제안 ${compactNumber(ma1AdmissionProjection(snapshot.hypothesis_packet).submittedCount)}건 · ` : ""}G0 ${esc(snapshot.g0_verdict?.verdict || "PENDING")}</small></article>
        <article><strong>${compactNumber(nonExcludedCandidates.length)}</strong><span>검토 후보</span><small>전체 ${compactNumber(candidates.length)} · 제외 ${compactNumber(candidates.length - nonExcludedCandidates.length)}</small></article>
        <article><strong>${portfolio.enabled ? `${compactNumber(portfolio.concluded)}/${compactNumber(portfolio.eligible)}` : compactNumber(portfolio.concluded || 0)}</strong><span>표적 분기 결론</span><small>후보 생성 ${compactNumber(portfolio.completed)} · 적합 후보 없음 ${compactNumber(portfolio.noCandidates)}</small></article>
        <article><strong>${compactNumber(evidence.unique)}</strong><span>고유 근거 레코드</span><small>연결 ${compactNumber(evidence.linked)}건 · ${compactNumber(evidence.sources.size)} sources</small></article>
        <article><strong>${compactNumber(calls.totalCalls)}</strong><span>추적된 호출</span><small>모델 ${compactNumber(calls.modelCalls)} · 데이터 ${compactNumber(calls.publicDataCalls)} · 도구 ${compactNumber(calls.scientificToolCalls)}</small></article>
        <article><strong>${esc(snapshot.g1_verdict?.verdict || "—")}</strong><span>G1 후보·과정 감사</span><small>연구자 ${decision ? decisionLabel(decision.decision) : "결정 미기록"}</small></article>
      </section>

      <section class="run-report-section" aria-labelledby="runReportBriefTitle">
        <header><span>01</span><div><h5 id="runReportBriefTitle">연구 범위와 질환 확인</h5><p>입력한 질문, 구조화 제약과 외부 질환 식별 결과</p></div></header>
        <div class="report-definition-grid">
          <dl>
            <div><dt>입력 질환</dt><dd>${esc(brief.disease || "미기록")}${brief.disease_id ? ` · ${esc(brief.disease_id)}` : ""}</dd></div>
            <div><dt>외부 질환 확인</dt><dd class="${intake.requiresInputCorrection || intake.verificationUnavailable ? "is-warning" : ""}">${esc(diseaseResolution)}</dd></div>
            <div><dt>연구 목표</dt><dd>${esc(brief.research_goal || "미기록")}</dd></div>
            <div><dt>연구 근거</dt><dd>${esc(brief.rationale || "별도 rationale 미기록")}</dd></div>
            <div><dt>RESEARCH DEPTH</dt><dd>${esc(depth.label)} · ${esc(depth.description)}</dd></div>
          </dl>
          <dl>
            <div><dt>표적 세포</dt><dd>${esc(constraints.target_cell_type || "지정 없음")}</dd></div>
            <div><dt>개입 방향</dt><dd>${esc(directionLabel(constraints.intervention_direction))}</dd></div>
            <div><dt>허용 가설 방향</dt><dd>${esc(directionConstraintLabel(constraints))}</dd></div>
            <div><dt>요청 우선순위</dt><dd>${esc(discoveryPriorityLabel(structuredPriority))}</dd></div>
            <div><dt>실제 적용 우선순위</dt><dd>${esc(discoveryPriorityLabel(effectivePriority))}${snapshot.plan_spec?.discovery_priority_derivation ? ` · ${esc(discoveryPriorityDerivationLabel(snapshot.plan_spec.discovery_priority_derivation))}` : ""}</dd></div>
            <div><dt>목표 해석</dt><dd>${goalCueLabels.length ? esc([...new Set(goalCueLabels)].join(" · ")) : "명시적 라우팅 단서 없음"}</dd></div>
            <div><dt>필수 조건</dt><dd>만성 안전성 ${constraints.chronic_safety_required === false ? "선택" : "필수"} · Druggability ${constraints.druggability_required === false ? "선택" : "필수"}</dd></div>
            <div><dt>후보 규모</dt><dd>표적별 선택 경로 최대 ${esc(constraints.candidates_per_route ?? "미기록")}건</dd></div>
            <div><dt>재창출 제외</dt><dd>약물명 ${compactNumber(constraints.forbidden_drugs?.length || 0)} · ID ${compactNumber(constraints.forbidden_drug_ids?.length || 0)} · 계열 ${compactNumber(constraints.forbidden_drug_classes?.length || 0)}</dd></div>
            <div><dt>연구자 우선사항</dt><dd>${brief.human_priorities?.length ? esc(brief.human_priorities.join(" · ")) : "별도 지정 없음"}</dd></div>
            <div><dt>장애 주입</dt><dd>${esc(faultLabel(brief.fault_injection))}</dd></div>
          </dl>
        </div>
        ${intake.hasNotice ? `<div class="report-note is-warning"><b>입력 검증 기록</b> · ${esc(intake.explanation || [...intake.reasonCodes, ...intake.warnings.map((item) => item.message), ...intake.fieldMismatches.map((item) => item.message)].filter(Boolean).join(" · ") || intake.verificationStatus || "확인 필요")}</div>` : ""}
      </section>

      <section class="run-report-section" aria-labelledby="runReportModelsTitle">
        <header><span>02</span><div><h5 id="runReportModelsTitle">실제 적용 모델</h5><p>모델별 역할, 권한 경계, 토큰·비용과 실제 반영 결과</p></div></header>
        ${modelExecutionHtml(snapshot, { includeHeading: false })}
      </section>

      <section class="run-report-section" aria-labelledby="runReportHandoffTitle">
        <header><span>03</span><div><h5 id="runReportHandoffTitle">모델 전달·의사결정 기록</h5><p>Terra 작성 → Sol 독립 감사 → 제한적 재작업의 실제 순서와 산출물 연결</p></div></header>
        ${modelDecisionTraceHtml(snapshot)}
      </section>

      <section class="run-report-section" aria-labelledby="runReportLlmChangesTitle">
        <header><span>04</span><div><h5 id="runReportLlmChangesTitle">모델 서술 변경 전·후</h5><p>가설 서술에 대해 패킷에 저장된 변경 전·후 문장과 변경 금지 필드</p></div></header>
        ${llmChangesHtml(snapshot, { includeHeading: false })}
      </section>

      <section class="run-report-section" aria-labelledby="runReportActualExecutionTitle">
        <header><span>05</span><div><h5 id="runReportActualExecutionTitle">데이터 소스와 MA2 실행 도구</h5><p>실제 계산 엔진과 파이프라인 작업을 분리한 실행·성공·실패·건너뜀 상태</p></div></header>
        ${actualExecutionResourcesHtml(snapshot, { includeHeading: false })}
      </section>

      <section class="run-report-section" aria-labelledby="runReportAuditTitle">
        <header><span>06</span><div><h5 id="runReportAuditTitle">독립 감사와 표적 분기 범위</h5><p>가설 G0, 후보·과정 G1, 그리고 G0 승인 표적별 실제 결론 상태</p></div></header>
        <div class="report-audit-grid">
          ${runReportAuditCard("G0 · 가설 감사", snapshot.g0_verdict)}
          ${runReportAuditCard("G1 · 후보·과정 감사", snapshot.g1_verdict)}
        </div>
        <div class="report-subsection">
          <h6>MA2 표적 분기 실행 범위</h6>
          <p>${esc(portfolioCoverageText(candidatePacket || {}, { includeApproved: true }))}</p>
          ${runReportTargetCoverageHtml(candidatePacket, portfolio)}
        </div>
      </section>

      <section class="run-report-section" aria-labelledby="runReportHypothesesTitle">
        <header><span>07</span><div><h5 id="runReportHypothesesTitle">MA1 표적 처리와 상위 가설</h5><p>명시 요청 표적의 전체 처리 흐름과 retained 가설의 기전·근거를 분리해 표시</p></div></header>
        ${explicitScopeHtml ? `<div class="report-subsection"><h6>사용자 지정 표적 전체 처리</h6>${explicitScopeHtml}</div>` : ""}
        ${runReportHypothesesHtml(snapshot, intake)}
      </section>

      <section class="run-report-section" aria-labelledby="runReportCandidatesTitle">
        <header><span>08</span><div><h5 id="runReportCandidatesTitle">후보 순위와 상태</h5><p>제외 후보를 순위에서 분리하고, 감사·연구자 결정 상태를 명시</p></div></header>
        ${runReportCandidatesHtml(snapshot, intake, portfolio)}
      </section>

      <section class="run-report-section" aria-labelledby="runReportIssuesTitle">
        <header><span>09</span><div><h5 id="runReportIssuesTitle">안전성 신호와 미해결 항목</h5><p>반복 감사 항목과 후보 안전성 신호를 유형별로 묶은 검토 목록</p></div></header>
        ${issueHtml}
      </section>

      <section class="run-report-section" aria-labelledby="runReportProvenanceTitle">
        <header><span>10</span><div><h5 id="runReportProvenanceTitle">실제 실행·근거 추적 정보</h5><p>계획상 허용 범위와 요청 시점 사용 기록을 분리</p></div></header>
        <div class="report-provenance-grid">
          <article>
            <h6>실행 경계</h6>
            <dl>
              <div><dt>LLM 실행 모드</dt><dd>${esc(mode)}${snapshot.token_usage?.length ? ` · usage ${compactNumber(snapshot.token_usage.length)}건` : " · token usage 미기록"}</dd></div>
              <div><dt>과학 실행 백엔드</dt><dd>${esc(scienceBackend)} · packet ${esc(candidatePacket?.execution_status || "미기록")}</dd></div>
              <div><dt>실제 tool-call 기록</dt><dd>${compactNumber(toolCalls.length)}건 · completed ${compactNumber(toolCompleted)} · skipped ${compactNumber(toolSkipped)} · failed/unavailable ${compactNumber(toolFailed)}</dd></div>
              <div><dt>호출 추적 기준</dt><dd>${trace ? "Call Trace API와 저장 snapshot 결합" : "저장 snapshot의 요청·도구 기록 집계"} · 총 ${compactNumber(calls.totalCalls)}건</dd></div>
              <div><dt>도구 이름</dt><dd>${toolNames.length ? esc(toolNames.join(" · ")) : "실제 tool-call 미기록"}</dd></div>
            </dl>
          </article>
          <article>
            <h6>근거 추적 정보</h6>
            <dl>
              <div><dt>실제 저장 근거</dt><dd>고유 ${compactNumber(evidence.unique)}건 · 연결 ${compactNumber(evidence.linked)}건 · ${compactNumber(evidence.sources.size)} sources</dd></div>
              <div><dt>획득 경로</dt><dd>LIVE ${compactNumber(evidence.live)} · CACHE ${compactNumber(evidence.cache)} · LOCAL DATASET ${compactNumber(evidence.local)} · FIXTURE ${compactNumber(evidence.fixture)} · 미분류 ${compactNumber(evidence.unknown)}</dd></div>
              <div><dt>근거 source</dt><dd>${sourceNames.length ? esc(sourceNames.join(" · ")) : "저장 근거 source 없음"}</dd></div>
              <div><dt>계획상 허용·admitted</dt><dd>${sourceCoverage ? `${compactNumber(admittedSources)}/${compactNumber(sourceCoverage.registry_source_count || sourceCoverage.decisions?.length || 0)} sources` : "manifest 미기록"}</dd></div>
              <div><dt>토큰·비용 기록</dt><dd>${snapshot.token_usage?.length ? `${compactNumber(tokenTotal)} tokens · $${costTotal.toFixed(2)} estimated` : "기록 없음 · 0으로 단정하지 않음"}</dd></div>
            </dl>
          </article>
        </div>
        <div class="report-note"><b>해석 원칙</b> · admitted는 계획상 조회 허용을 뜻할 뿐 성공 호출이나 근거 획득을 뜻하지 않습니다. 위 실제 source와 tool 상태는 이 실행에 저장된 EvidenceRef·ToolCallRecord·Call Trace 기준입니다.</div>
      </section>

      <section class="run-report-section" aria-labelledby="runReportIntegrityTitle">
        <header><span>11</span><div><h5 id="runReportIntegrityTitle">연구자 결정과 감사 무결성</h5><p>결정 주체, 스키마·식별자·해시 및 원본 보존 상태</p></div></header>
        <div class="report-definition-grid">
          <dl>
            <div><dt>연구자 결정</dt><dd>${decision ? `${esc(decisionLabel(decision.decision))} · ${esc(decision.researcher || "연구자 미기록")} · 선택 ${compactNumber(decision.selected_candidate_ids?.length || 0)}건` : "미기록 · Human Gate 대기 또는 미도달"}</dd></div>
            <div><dt>결정 근거</dt><dd>${esc(decision?.rationale || "미기록")}</dd></div>
            <div><dt>자동 감사 시각</dt><dd>G0 ${esc(runReportDate(snapshot.g0_verdict?.created_at))} · G1 ${esc(runReportDate(snapshot.g1_verdict?.created_at))}</dd></div>
            <div><dt>실행 기간</dt><dd>${esc(runReportDate(snapshot.created_at))} → ${esc(runReportDate(snapshot.updated_at))} · ${esc(formatDuration(durationBetween(snapshot.created_at, snapshot.updated_at)))}</dd></div>
          </dl>
          <dl>
            <div><dt>스키마 버전</dt><dd>Brief ${esc(brief.schema_version || "미기록")} · Hypothesis ${esc(hypothesisPacket?.schema_version || "—")} · Plan ${esc(snapshot.plan_spec?.schema_version || "—")} · Candidate ${esc(candidatePacket?.schema_version || "—")}</dd></div>
            <div><dt>도구 입·출력 해시</dt><dd>${compactNumber(hashComplete)}/${compactNumber(toolCalls.length)} calls</dd></div>
            <div><dt>근거 외부 ID·조회시각</dt><dd>${compactNumber(evidenceWithExternalId)}/${compactNumber(evidenceRecords.length)} linked records</dd></div>
            <div><dt>근거 content/excerpt hash</dt><dd>${compactNumber(evidenceWithExcerptHash)}/${compactNumber(evidenceRecords.length)} linked records</dd></div>
            <div><dt>Event artifact refs</dt><dd>${compactNumber(eventArtifactRefs.size)}개 · events ${compactNumber(snapshot.events?.length || 0)}건</dd></div>
          </dl>
        </div>
        <div class="report-note"><b>원본 보존</b> · TECHNICAL DATA에서 전체 Raw JSON을 접어서 열 수 있고, 상단의 ‘원본 JSON 저장’으로 동일 snapshot을 다운로드할 수 있습니다. 해시는 기록 변조 탐지에 도움을 주지만 과학적 타당성 자체를 보증하지 않습니다.</div>
      </section>
    </article>`;
  }

  function renderOverviewArtifact() {
    const s = state.snapshot;
    const intake = getIntakeAssessment(s);
    const brief = s.brief || {};
    const constraints = brief.constraints || {};
    const hypothesisCount = s.hypothesis_packet?.hypotheses?.length || 0;
    const admission = ma1AdmissionProjection(s.hypothesis_packet || {});
    const candidateCount = s.candidate_packet?.candidates?.length || 0;
    const candidatePacket = s.candidate_packet || {};
    const portfolio = portfolioExecutionState(candidatePacket);
    const evidenceCount = countEvidence(s);
    const ma1Meta = s.hypothesis_packet?.metadata || {};
    const explicitScope = ma1ExplicitScopeProjection(s.hypothesis_packet || {});
    const screeningSources = Array.isArray(ma1Meta.ma1_screening_history_sources)
      ? ma1Meta.ma1_screening_history_sources.filter((item) => item && typeof item === "object")
      : Array.isArray(ma1Meta.screened_target_sources)
      ? ma1Meta.screened_target_sources.filter((item) => item && typeof item === "object")
      : [];
    const screenedTargetCount = new Set(
      (Array.isArray(ma1Meta.goal_focus_screened_symbols)
        ? ma1Meta.goal_focus_screened_symbols
        : screeningSources.map(screeningTargetSymbol))
        .map((symbol) => String(symbol || "").trim().toLowerCase())
        .filter(Boolean),
    ).size;
    const diseaseScreenedTargetCount = screeningSources.length
      ? new Set(
          screeningSources
            .filter((item) => String(item.target_source || "") !== "explicit_request_scope_identity")
            .map(screeningTargetSymbol)
            .map((symbol) => String(symbol || "").trim().toLowerCase())
            .filter(Boolean),
        ).size
      : explicitScope.present ? 0 : screenedTargetCount;
    const discoveredTargetCount = nonnegativeInteger(
      ma1Meta.ma1_discovery_candidate_trace_total,
      ma1Meta.ma1_scope_candidate_pool_count == null
        ? ma1Meta.external_target_seed_count || 0
        : ma1Meta.ma1_scope_candidate_pool_count,
    );
    const goalFocus = ma1Meta.goal_focus || {};
    const planSpec = s.plan_spec || {};
    const researchDepth = researchDepthPresentation(researchDepthFromSnapshot(s));
    const structuredPriority = constraints.discovery_priority || "auto";
    const effectivePriority = planSpec.discovery_priority || structuredPriority;
    const goalCueLabels = (goalFocus.cues || [])
      .map((cue) => goalCueCategoryLabel(cue.category))
      .filter(Boolean);
    const resolvedDisease = ma1Meta.resolved_disease || {};
    const diseaseResolutionLabel = !s.hypothesis_packet
      ? "대기"
      : resolvedDisease.disease_name || intake.raw.resolved_disease_name
      ? `${resolvedDisease.disease_name || intake.raw.resolved_disease_name}${resolvedDisease.disease_id || intake.raw.resolved_disease_id ? ` · ${resolvedDisease.disease_id || intake.raw.resolved_disease_id}` : ""}`
      : intake.verificationUnavailable
      ? "검증 불가 · 외부 소스 연결 실패"
      : "확인되지 않음";
    const hypothesisEvidence = (s.hypothesis_packet?.hypotheses || []).flatMap((item) => item.evidence || []);
    const fixtureEvidenceCount = hypothesisEvidence.filter(isFixtureEvidence).length;
    const targetStatuses = ma1Meta.target_source_statuses || [];
    const contextRouting = ma1Meta.context_routing || {};
    const stageName = String(s.stage || "created").toLowerCase();
    const completed = TERMINAL_STAGES.has(stageName)
      ? Math.max(0, inferLastStageIndex(s))
      : Math.max(0, STAGES.indexOf(stageName));
    const progress = stageName === "completed" ? 100 : Math.round((completed / (STAGES.length - 1)) * 100);
    return `
      <div class="overview-grid">
        <article class="artifact-card">
          <h4>연구 입력 요약</h4>
          <dl class="detail-list">
            <div class="detail-row"><dt>대상 질환</dt><dd>${esc(brief.disease || "—")}</dd></div>
            <div class="detail-row"><dt>연구 목표</dt><dd>${esc(brief.research_goal || "—")}</dd></div>
            <div class="detail-row"><dt>RESEARCH DEPTH</dt><dd>${esc(researchDepth.label)} · ${esc(researchDepth.description)}</dd></div>
            <div class="detail-row"><dt>세포 유형</dt><dd>${esc(constraints.target_cell_type || "지정 없음")}</dd></div>
            <div class="detail-row"><dt>작용 방향</dt><dd>${esc(directionLabel(constraints.intervention_direction))}</dd></div>
            <div class="detail-row"><dt>허용 가설 방향</dt><dd>${esc(directionConstraintLabel(constraints))}</dd></div>
            <div class="detail-row"><dt>요청 우선순위</dt><dd>${esc(discoveryPriorityLabel(structuredPriority))}</dd></div>
            <div class="detail-row"><dt>실제 적용 우선순위</dt><dd>${esc(discoveryPriorityLabel(effectivePriority))}${planSpec.discovery_priority_derivation ? ` · ${esc(discoveryPriorityDerivationLabel(planSpec.discovery_priority_derivation))}` : ""}</dd></div>
            <div class="detail-row"><dt>목표 해석</dt><dd>${goalCueLabels.length ? esc([...new Set(goalCueLabels)].join(" · ")) : "명시적 라우팅 단서 없음"}</dd></div>
            ${intake.hasMismatch ? `<div class="detail-row intake-detail-warning"><dt>입력 일관성</dt><dd>확인 필요 · ${compactNumber(intake.fieldMismatches.length)}건</dd></div>` : ""}
            <div class="detail-row"><dt>후보 규모</dt><dd>표적별 선택 경로 최대 ${esc(constraints.candidates_per_route ?? "—")}건 · Repurposing 또는 De novo</dd></div>
            <div class="detail-row"><dt>재창출 제외</dt><dd>약물명 ${(constraints.forbidden_drugs || []).length}건 · ID ${(constraints.forbidden_drug_ids || []).length}건 · 계열 ${(constraints.forbidden_drug_classes || []).length}건</dd></div>
            <div class="detail-row"><dt>장애 주입</dt><dd>${esc(faultLabel(brief.fault_injection))}</dd></div>
          </dl>
          ${(brief.human_priorities || []).length ? `<div class="tag-row" style="margin-top:11px">${brief.human_priorities.map((p) => `<span class="tag teal">${esc(p)}</span>`).join("")}</div>` : ""}
        </article>
        <div class="progress-list">
          <article class="artifact-card soft">
            <h4>현재 실행 요약</h4>
            <div class="progress-item"><div class="progress-item-head"><span>전체 진행률</span><b>${clamp(progress,0,100)}%</b></div><div class="progress-track"><div class="progress-fill" style="width:${clamp(progress,0,100)}%"></div></div></div>
            <dl class="detail-list" style="margin-top:12px">
              <div class="detail-row"><dt>연결 레코드</dt><dd>${compactNumber(evidenceCount)}건${fixtureEvidenceCount ? ` · Fixture ${compactNumber(fixtureEvidenceCount)}건` : ""}</dd></div>
              <div class="detail-row"><dt>분석 깊이</dt><dd>${esc(researchDepth.label)}${researchDepth.value ? "" : " · 설정값 없음"}</dd></div>
              <div class="detail-row"><dt>가설 포트폴리오</dt><dd>${admission.enabled ? `MA1 제안 ${compactNumber(admission.submittedCount)}건 · MA3 승인 ${compactNumber(admission.approvedCount)}건` : `${compactNumber(hypothesisCount)}건`}</dd></div>
              <div class="detail-row"><dt>MA2 표적 범위</dt><dd>${s.candidate_packet ? esc(portfolioCoverageText(candidatePacket, { includeApproved: true })) : "대기"}</dd></div>
              <div class="detail-row"><dt>후보 평가</dt><dd>${compactNumber(candidateCount)}건${portfolio.enabled ? ` · ${esc(portfolio.executionStatus.toUpperCase())}` : ""}</dd></div>
              <div class="detail-row"><dt>실행·재작업 횟수</dt><dd>MA1 실행 ${s.ma1_iterations || 0}회 · 전체 재작업 ${s.rework_loops || 0}회</dd></div>
            </dl>
          </article>
          <article class="artifact-card soft dynamic-summary">
            <h4>질환별 동적 표적 발굴</h4>
            <dl class="detail-list">
              <div class="detail-row"><dt>질환 해석</dt><dd class="${intake.diseaseNeedsReview || intake.verificationUnavailable ? "resolution-unverified" : ""}">${esc(diseaseResolutionLabel)}</dd></div>
              <div class="detail-row"><dt>입력 검증</dt><dd>${intake.abstained ? "ABSTAIN · 후보 생성 중단" : intake.hasMismatch ? "경고 · 필드 불일치" : intake.verificationStatus ? esc(intake.verificationStatus.replaceAll("_", " ")) : "진행 중"}</dd></div>
              <div class="detail-row"><dt>질환 기반 후보 풀</dt><dd>발견 ${compactNumber(discoveredTargetCount)}건 · 질환 후보 심사 ${compactNumber(diseaseScreenedTargetCount)}건 · 현재 질환발굴 seed ${compactNumber(ma1Meta.external_target_seed_count || 0)}건</dd></div>
              ${explicitScope.present ? `<div class="detail-row"><dt>명시 요청 표적</dt><dd>${explicitScope.valid ? `요청 ${compactNumber(explicitScope.collections.requested.length)}건 · exact ID 해결 ${compactNumber(explicitScope.collections.identity_resolved.length)}건 · 심사 ${compactNumber(explicitScope.collections.screened.length)}건 · retained ${compactNumber(explicitScope.collections.retained.length)}건 · 제외 ${compactNumber(explicitScope.collections.excluded.length)}건` : "보존 장부 불완전 · MA1 Results에서 행을 추정하지 않음"}</dd></div>` : ""}
              <div class="detail-row"><dt>Fixture 표적</dt><dd>${compactNumber(ma1Meta.fallback_target_seed_count || 0)}건${ma1Meta.target_fallback_used ? " · 검증용, 연구 근거 아님" : ""}</dd></div>
              <div class="detail-row"><dt>맥락 조회</dt><dd>${contextRouting.status || "대기"}${contextRouting.source_calls_used != null ? ` · ${contextRouting.source_calls_used} calls` : ""}</dd></div>
            </dl>
            ${targetStatuses.length ? `<div class="source-status-row">${targetStatuses.map((source) => `<span class="source-status-pill ${esc(source.status || "unknown")}">${esc(source.source)} · ${esc(source.status || "unknown")}</span>`).join("")}</div>` : ""}
          </article>
          <article class="artifact-card soft"><h4>데이터 정책</h4><div class="tag-row">${(brief.data_policy?.allowed_sources || []).slice(0,12).map((source) => `<span class="tag">${esc(source)}</span>`).join("") || '<span class="tag">기본 허용 목록</span>'}</div></article>
        </div>
      </div>`;
  }

  function nonnegativeInteger(value, fallback = 0) {
    if (value == null || value === "") return fallback;
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? Math.floor(number) : fallback;
  }

  function screeningTargetSymbol(record = {}) {
    return String(record.target_symbol || record.target?.symbol || "");
  }

  function screeningRecordKey(record = {}) {
    return [
      String(record.hypothesis_id || ""),
      String(record.source_iteration ?? ""),
      String(record.screening_phase || ""),
    ].join("|");
  }

  function screeningTargetIterationKey(record = {}) {
    return `${screeningTargetSymbol(record).trim().toLowerCase()}|${String(record.source_iteration ?? "")}`;
  }

  function screeningPhaseLabel(phase) {
    const labels = {
      proposal: "Proposal screening",
      prior_reclassification: "Prior revision reclassification",
      router_exclusion_summary: "Router exclusion summary",
    };
    return labels[String(phase || "").toLowerCase()] || String(phase || "단계 미기록").replaceAll("_", " ");
  }

  function ma1ScreeningProjection(packet = {}) {
    const metadata = packet?.metadata && typeof packet.metadata === "object" ? packet.metadata : {};
    const rawTrace = metadata.rejected_screening_trace;
    const trace = rawTrace && typeof rawTrace === "object" && !Array.isArray(rawTrace) ? rawTrace : {};
    const screenedHistory = Array.isArray(metadata.ma1_screening_history_sources)
      ? metadata.ma1_screening_history_sources.filter((item) => item && typeof item === "object")
      : [];
    const screened = screenedHistory.length
      ? screenedHistory
      : Array.isArray(metadata.screened_target_sources)
      ? metadata.screened_target_sources.filter((item) => item && typeof item === "object")
      : [];
    const historyTrace = metadata.ma1_rejected_screening_history && typeof metadata.ma1_rejected_screening_history === "object"
      ? metadata.ma1_rejected_screening_history
      : null;
    const effectiveTrace = historyTrace || trace;
    const traceEntries = Array.isArray(effectiveTrace.entries)
      ? effectiveTrace.entries.filter((item) => item && typeof item === "object")
      : [];
    const exclusions = Array.isArray(packet?.excluded)
      ? packet.excluded.filter((item) => item && typeof item === "object")
      : [];
    const exclusionProvenance = Array.isArray(metadata.ma1_exclusion_provenance)
      ? metadata.ma1_exclusion_provenance.filter((item) => item && typeof item === "object")
      : [];
    const usedTraceIndexes = new Set();
    const records = [];

    const findTrace = (source) => {
      const exactKey = screeningRecordKey(source);
      let match = traceEntries.findIndex((entry, index) => (
        !usedTraceIndexes.has(index)
        && exactKey !== "||"
        && screeningRecordKey(entry) === exactKey
      ));
      if (match < 0) {
        const targetIterationKey = screeningTargetIterationKey(source);
        match = traceEntries.findIndex((entry, index) => (
          !usedTraceIndexes.has(index)
          && targetIterationKey !== "|"
          && screeningTargetIterationKey(entry) === targetIterationKey
          && String(entry.screening_phase || "") === String(source.screening_phase || "")
        ));
      }
      if (match < 0) return null;
      usedTraceIndexes.add(match);
      return traceEntries[match];
    };

    const makeRecord = (source, detail, origin) => {
      const target = detail?.target && typeof detail.target === "object" ? detail.target : {};
      const reasonCodes = [
        ...(Array.isArray(source?.reason_codes) ? source.reason_codes : []),
        ...(Array.isArray(detail?.reason_codes) ? detail.reason_codes : []),
      ].map((item) => String(item)).filter(Boolean);
      const evidenceValue = source?.evidence_count ?? detail?.evidence_count_total;
      return {
        targetSymbol: String(source?.target_symbol || source?.target?.symbol || target.symbol || "표적 미기록"),
        hypothesisId: String(source?.hypothesis_id || detail?.hypothesis_id || ""),
        track: String(source?.track || source?.target?.track || target.track || ""),
        sourceIteration: source?.source_iteration ?? detail?.source_iteration ?? null,
        screeningPhase: String(source?.screening_phase || detail?.screening_phase || (origin === "excluded_summary" ? "router_exclusion_summary" : "")),
        disposition: String(source?.disposition || detail?.disposition || "rejected").toLowerCase(),
        reasonCodes: [...new Set(reasonCodes)],
        targetSource: String(source?.target_source || ""),
        identityStatus: String(source?.identity_status || target.identity_status || ""),
        ensemblId: String(source?.ensembl_id || target.ensembl_id || ""),
        uniprotId: String(source?.uniprot_id || target.uniprot_id || ""),
        evidenceCount: evidenceValue == null ? null : nonnegativeInteger(evidenceValue, 0),
        exclusionRecordOrigin: String(source?.exclusion_record_origin || ""),
        recordOrigin: String(source?.record_origin || detail?.record_origin || ""),
        traceDetail: detail,
        origin,
      };
    };

    screened.forEach((source) => {
      if (String(source.disposition || "").toLowerCase() === "retained") return;
      const detail = findTrace(source);
      records.push(makeRecord(source, detail, detail ? "screened_trace" : "screened"));
    });

    traceEntries.forEach((detail, index) => {
      if (usedTraceIndexes.has(index)) return;
      records.push(makeRecord({}, detail, "trace_only"));
    });

    exclusions.forEach((exclusion) => {
      const provenance = exclusionProvenance.find((item) => (
        screeningTargetSymbol(item).trim().toLowerCase() === screeningTargetSymbol(exclusion).trim().toLowerCase()
        && String(item.source_iteration ?? "") === String(exclusion.source_iteration ?? "")
      ));
      if (provenance) exclusion = {...exclusion, exclusion_record_origin: provenance.record_origin};
      const key = screeningTargetIterationKey(exclusion);
      const existing = records.find((record) => (
        `${record.targetSymbol.trim().toLowerCase()}|${String(record.sourceIteration ?? "")}` === key
      ));
      if (existing) {
        existing.reasonCodes = [...new Set([
          ...existing.reasonCodes,
          ...(Array.isArray(exclusion.reason_codes) ? exclusion.reason_codes.map((item) => String(item)) : []),
        ].filter(Boolean))];
        if (provenance?.record_origin) {
          existing.exclusionRecordOrigin = String(provenance.record_origin);
        }
        return;
      }
      records.push(makeRecord(exclusion, null, "excluded_summary"));
    });

    return {
      records,
      screenedTotal: nonnegativeInteger(metadata.ma1_screening_history_source_count_total, nonnegativeInteger(metadata.screened_target_source_count_total, screened.length)),
      screenedRecorded: nonnegativeInteger(metadata.ma1_screening_history_source_count_recorded, nonnegativeInteger(metadata.screened_target_source_count_recorded, screened.length)),
      screenedOmitted: nonnegativeInteger(metadata.ma1_screening_history_source_count_omitted, nonnegativeInteger(metadata.screened_target_source_count_omitted, 0)),
      rejectedTotal: nonnegativeInteger(effectiveTrace.entry_count_total, traceEntries.length),
      rejectedRecorded: nonnegativeInteger(effectiveTrace.entry_count_recorded, traceEntries.length),
      rejectedOmitted: nonnegativeInteger(effectiveTrace.entry_count_omitted, 0),
      exclusionSummaryCount: exclusions.length,
      fallbackSummaryCount: records.filter((record) => record.origin === "excluded_summary").length,
      contentPolicy: String(effectiveTrace.content_policy || "bounded current-packet projection"),
      immutableProjection: effectiveTrace.immutable_projection === true,
      exclusionProvenanceRecorded: nonnegativeInteger(metadata.ma1_exclusion_provenance_recorded, exclusionProvenance.length),
      exclusionProvenanceTotal: nonnegativeInteger(metadata.ma1_exclusion_provenance_total, exclusionProvenance.length),
    };
  }

  function screeningTraceDetailHtml(record, projection, packet, index) {
    const detail = record.traceDetail;
    if (!detail) {
      const explanation = record.origin === "excluded_summary"
        ? record.exclusionRecordOrigin === "prior_packet_screened_preserved_not_rescreened"
          ? "이 항목은 이전 revision에서 실제 screening된 기록을 보존한 것이며, 현재 revision에서 다시 screening한 결과는 아닙니다."
          : record.exclusionRecordOrigin === "prior_packet_preserved_not_rescreened"
          ? "이 항목은 이전 패킷의 제외 기록을 보존한 것이며, 현재 revision에서 다시 screening한 결과가 아닙니다."
          : "현재 패킷에는 Router exclusion 요약만 있으며 해당 proposal의 상세 rejected-screening trace는 없습니다."
        : "현재 패킷에 이 proposal의 상세 rejected-screening trace가 기록되지 않았습니다.";
      return `<small class="screening-trace-absence">${esc(explanation)}</small>`;
    }
    const externalRecords = Array.isArray(detail.unique_external_records) ? detail.unique_external_records : [];
    const operations = Array.isArray(detail.operation_observations) ? detail.operation_observations : [];
    const assignment = detail.claim_assignment && typeof detail.claim_assignment === "object" ? detail.claim_assignment : {};
    const scopes = detail.direction_triage?.observed_molecular_scopes || [];
    const branch = detail.molecular_hypothesis_scope;
    const branchHtml = branch ? `<p class="screening-boundary"><b>분자 범위별 가설: ${esc(branch.label)}</b><br>기준 유전자 ${esc(branch.target_symbol)} · 원문에서 발굴한 연구 범위 · 입력 조건 변경 없음 · 정확한 실험 construct는 별도 검증</p>` : "";
    const scopeHtml = scopes.length ? `<section><h5>원문에 등장한 융합·변이 범위</h5><p>원문 표기와 입력 조건을 구분합니다. 표기 발견은 변이 존재·단백질 단독 효과·정확한 construct의 검증이 아닙니다.</p><ul class="screening-detail-list">${scopes.map(s=>`<li><b>${esc((s.observed_labels || []).join(", "))}</b><span>${esc(s.source)} · ${esc(s.external_id)}<br>입력 선언: ${esc((s.declared_labels || []).join(", ") || "없음")} · ${esc(s.status)}<br>Claim 연결: ${esc((s.claim_roles || []).join(", ") || "미연결")}</span></li>`).join("")}</ul></section>` : "";
    const disclosureKey = [
      "ma1-screening",
      packet?.packet_id || "packet",
      record.hypothesisId || record.targetSymbol || index,
      record.sourceIteration ?? "iteration",
      index,
    ].join(":");
    const externalHtml = externalRecords.length
      ? `<ul class="screening-detail-list">${externalRecords.map((item) => {
          const unitCount = nonnegativeInteger(item.evidence_unit_count_total, Array.isArray(item.evidence_units) ? item.evidence_units.length : 0);
          const omitted = nonnegativeInteger(item.evidence_unit_count_omitted, 0);
          const rejectionCodes = [...new Set((Array.isArray(item.evidence_units) ? item.evidence_units : []).flatMap((unit) => Array.isArray(unit.direct_support_rejection_codes) ? unit.direct_support_rejection_codes : []))];
          return `<li><b>${esc(item.source || "Source 미기록")} · ${esc(item.external_id || "ID 미기록")}</b><span>evidence units ${compactNumber(unitCount)}${omitted ? ` · omitted ${compactNumber(omitted)}` : ""}${item.pmid ? ` · PMID ${esc(item.pmid)}` : ""}${item.doi ? ` · DOI ${esc(item.doi)}` : ""}${rejectionCodes.length ? ` · direct-support 제외: ${esc(rejectionCodes.join(", "))}` : ""}</span></li>`;
        }).join("")}</ul>`
      : '<p class="screening-detail-empty">기록된 외부 근거 레코드 없음</p>';
    const operationsHtml = operations.length
      ? `<ul class="screening-detail-list">${operations.map((item) => `<li><b>${esc(item.source || item.observation_kind || "Operation")}</b><span>${esc(item.status || item.failure_code || "상태 미기록")}${item.external_id ? ` · ${esc(item.external_id)}` : ""}</span></li>`).join("")}</ul>`
      : '<p class="screening-detail-empty">기록된 source-operation 관찰 없음</p>';
    return `<details class="provenance-details screening-trace-details" data-disclosure="${esc(disclosureKey)}">
      <summary>Bounded rejected-screening detail</summary>
      <dl class="context-facts">
        <div><dt>외부 레코드</dt><dd>${compactNumber(nonnegativeInteger(detail.unique_external_record_count_recorded, externalRecords.length))}/${compactNumber(nonnegativeInteger(detail.unique_external_record_count_total, externalRecords.length))} recorded</dd></div>
        <div><dt>Operation 관찰</dt><dd>${compactNumber(nonnegativeInteger(detail.operation_observation_count_recorded, operations.length))}/${compactNumber(nonnegativeInteger(detail.operation_observation_count_total, operations.length))} recorded</dd></div>
        <div><dt>Claim 연결</dt><dd>${compactNumber(nonnegativeInteger(assignment.claim_count_recorded, Array.isArray(assignment.claims) ? assignment.claims.length : 0))}/${compactNumber(nonnegativeInteger(assignment.claim_count_total, Array.isArray(assignment.claims) ? assignment.claims.length : 0))} recorded</dd></div>
        <div><dt>Atomic linkage</dt><dd>${assignment.atomic_claim_linkage === true ? "기록상 충족" : assignment.atomic_claim_linkage === false ? "기록상 미충족" : "미기록"}</dd></div>
      </dl>
      <div class="screening-detail-columns"><section><h5>External records</h5>${externalHtml}</section><section><h5>Source operations</h5>${operationsHtml}</section></div>
      ${branchHtml}${scopeHtml}
      <p class="screening-boundary"><b>표시 경계</b> ${esc(projection.contentPolicy)} · 원문 초록 전문과 비공개 데이터는 표시하지 않습니다.</p>
    </details>`;
  }

  function renderScreenedProposals(packet, projection) {
    const records = projection.records;
    const admissionMode = packet?.metadata?.admission_policy === "ma3-source-bound-v1";
    const rows = records.map((record, index) => {
      const provisional = record.disposition === "provisional";
      const stableIds = [record.ensemblId, record.uniprotId].filter(Boolean);
      const sourceLabel = record.origin === "excluded_summary"
        ? record.exclusionRecordOrigin === "prior_packet_screened_preserved_not_rescreened"
          ? "Prior packet screening preserved · current revision not re-screened"
          : record.exclusionRecordOrigin === "prior_packet_preserved_not_rescreened"
          ? "Prior packet exclusion · current revision not re-screened"
          : "Router exclusion summary"
        : record.recordOrigin === "prior_revision_screening"
        ? "Prior revision rejected-screening trace"
        : record.traceDetail
        ? "Rejected screening trace"
        : "Screening source record";
      const reasons = record.reasonCodes.length
        ? `<div class="screening-reason-list">${record.reasonCodes.map((reason) => `<code>${esc(reason)}</code>`).join("")}</div>`
        : '<span class="muted-text">Reason code 미기록</span>';
      return `<tr class="screening-row ${provisional ? "is-provisional" : "is-excluded"}">
        <td><strong>${esc(record.traceDetail?.molecular_hypothesis_scope?.label || record.targetSymbol)}</strong><small>${record.traceDetail?.molecular_hypothesis_scope ? `기준 유전자 ${esc(record.targetSymbol)} · ` : ""}${record.hypothesisId ? esc(record.hypothesisId) : "Proposal ID 미기록"}</small></td>
        <td><span class="quiet-status">${esc(record.track || "Track 미기록")}</span><small>${esc(record.identityStatus || "Identity status 미기록")}${stableIds.length ? ` · ${esc(stableIds.join(" · "))}` : ""}</small></td>
        <td><strong>${esc(screeningPhaseLabel(record.screeningPhase))}</strong><small>iteration ${esc(record.sourceIteration ?? "미기록")} · ${esc(sourceLabel)}</small></td>
        <td>${reasons}</td>
        <td class="numeric-cell">${record.evidenceCount == null ? "—" : compactNumber(record.evidenceCount)}<small>${record.evidenceCount == null ? "요약에 미기록" : "records"}</small></td>
        <td>${provisional ? '<span class="screening-status admission-pending">MA1 잠정 제출 기록</span><small>최종 MA3 판정은 상단 가설 카드 참고</small>' : '<span class="screening-status is-rejected">Not retained</span>'}${screeningTraceDetailHtml(record, projection, packet, index)}</td>
      </tr>`;
    }).join("");
    return `<section class="ma1-result-section ma1-screening-results" aria-labelledby="ma1ScreeningTitle">
      <header class="ma1-result-heading"><div><span>SCREENING RECORDS</span><h4 id="ma1ScreeningTitle">${admissionMode ? "MA1 수집·형식 점검 기록" : "Screened proposals — retained가 아닌 항목"}</h4><p>${admissionMode ? "잠정 제출과 형식상 제외를 구분합니다. 기존 규칙의 진단 코드는 참고 기록이며 MA3의 최종 가설 판정과 다를 수 있습니다." : "아래 표적은 최종 retained hypothesis가 아닙니다. Current packet의 screening 기록과 Router exclusion 요약을 구분해 표시합니다."}</p></div><strong>${admissionMode ? "기록" : "Not retained"} ${compactNumber(records.length)}</strong></header>
      <div class="screening-projection-summary" aria-label="MA1 screening bounded projection counts">
        <span><b>Screened projection</b>${compactNumber(projection.screenedRecorded)}/${compactNumber(projection.screenedTotal)} recorded · omitted ${compactNumber(projection.screenedOmitted)}</span>
        <span><b>Rejected detail trace</b>${compactNumber(projection.rejectedRecorded)}/${compactNumber(projection.rejectedTotal)} recorded · omitted ${compactNumber(projection.rejectedOmitted)}</span>
        <span><b>Exclusion summaries</b>${compactNumber(projection.exclusionSummaryCount)} packet records · detail fallback ${compactNumber(projection.fallbackSummaryCount)}</span>
        <span><b>Exclusion provenance</b>${compactNumber(projection.exclusionProvenanceRecorded)}/${compactNumber(projection.exclusionProvenanceTotal)} recorded</span>
      </div>
      <p class="screening-boundary"><b>Bounded projection</b> · current packet에 보존된 bounded revision history만 표시하며 과거 packet 전문을 재구성하지 않습니다.${projection.immutableProjection ? " Rejected trace는 immutable bounded projection으로 기록됐습니다." : ""}</p>
      ${rows ? `<div class="research-table-wrap"><table class="research-table screening-table">
        <caption class="sr-only">최종 retained hypothesis가 아닌 MA1 screening 및 exclusion 기록</caption>
        <thead><tr><th>Target / proposal</th><th>Track / identity</th><th>Screening record</th><th>Reason codes</th><th>Evidence</th><th>Disposition / trace</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>` : '<div class="screening-empty">현재 packet에 retained가 아닌 screening 또는 exclusion 기록이 없습니다.</div>'}
    </section>`;
  }

  function renderDiscoveryCandidateTrace(packet = {}) {
    const metadata = packet?.metadata && typeof packet.metadata === "object" ? packet.metadata : {};
    const records = Array.isArray(metadata.ma1_discovery_candidate_trace)
      ? metadata.ma1_discovery_candidate_trace.filter((item) => item && typeof item === "object")
      : [];
    const total = nonnegativeInteger(
      metadata.ma1_discovery_candidate_trace_total,
      metadata.ma1_scope_candidate_pool_count == null ? records.length : metadata.ma1_scope_candidate_pool_count,
    );
    const recorded = nonnegativeInteger(metadata.ma1_discovery_candidate_trace_recorded, records.length);
    const omitted = nonnegativeInteger(metadata.ma1_discovery_candidate_trace_omitted, Math.max(0, total - recorded));
    const scoped = metadata.ma1_executable_target_scope_applied === true;
    const matches = nonnegativeInteger(metadata.ma1_scope_candidate_match_count, 0);
    const outside = records.filter((item) => String(item.scope_disposition || "") === "outside_explicit_scope").length;
    if (!total && !records.length) return "";
    const rows = records.map((item) => {
      const disposition = String(item.scope_disposition || "portfolio_candidate");
      const screeningAllocation = String(item.screening_allocation || item.screening_disposition || "");
      const reasonCodes = Array.isArray(item.reason_codes) ? item.reason_codes : [];
      const sources = Array.isArray(item.contributing_sources) ? item.contributing_sources : [];
      const status = disposition === "scope_match"
        ? "Explicit scope match"
        : disposition === "outside_explicit_scope"
        ? "Outside requested scope"
        : screeningAllocation === "scheduled_for_ma1"
        ? "Allocated to MA1 screening"
        : screeningAllocation === "scheduled_for_ma1_evidence_retry"
        ? "Allocated to MA1 evidence retry"
        : screeningAllocation === "previous_revision_screened_result_preserved"
        ? "Screened previously · result preserved"
        : screeningAllocation === "not_scheduled_evidence_retry_budget"
        ? "Not scheduled · evidence-retry budget"
        : screeningAllocation === "not_scheduled_iteration_budget"
        ? "Not scheduled · bounded iterations"
        : "Discovery pool";
      return `<tr class="screening-row ${disposition === "outside_explicit_scope" ? "is-excluded" : ""}">
        <td class="rank-cell">${esc(item.rank || "—")}</td>
        <td><strong>${esc(item.target_symbol || "—")}</strong><small>${esc(item.target_name || item.ensembl_id || "이름 미기록")}</small></td>
        <td>${item.association_score == null ? "—" : formatScore(item.association_score)}<small>${esc(item.primary_source || "Source 미기록")}</small></td>
        <td>${sources.length ? esc(sources.join(" · ")) : "—"}</td>
        <td><span class="screening-status ${disposition === "scope_match" ? "is-retained" : disposition === "outside_explicit_scope" ? "is-rejected" : ""}">${esc(status)}</span>${reasonCodes.length ? `<small>${reasonCodes.map((code) => `<code>${esc(code)}</code>`).join(" ")}</small>` : ""}</td>
      </tr>`;
    }).join("");
    const legacyNotice = !records.length && total
      ? '<div class="screening-empty">이 과거 실행은 후보 수만 보존했고 후보별 이름·범위 판정은 저장하지 않았습니다. 새 실행부터 상세 추적이 표시됩니다.</div>'
      : "";
    return `<section class="ma1-result-section ma1-discovery-results" aria-labelledby="ma1DiscoveryTitle">
      <header class="ma1-result-heading"><div><span>DISCOVERY POOL</span><h4 id="ma1DiscoveryTitle">질환 기반 표적 후보 풀</h4><p>외부 표적 발굴 결과와 실행 범위 판정을 표시합니다. 범위 밖 표적은 과학적 탈락이 아니라 이번 요청에서 실행하지 않은 항목입니다.</p></div><strong>Pool ${compactNumber(total)}</strong></header>
      <div class="screening-projection-summary"><span><b>Candidate trace</b>${compactNumber(recorded)}/${compactNumber(total)} recorded · omitted ${compactNumber(omitted)}</span>${scoped ? `<span><b>Explicit scope</b>match ${compactNumber(matches)} · outside ${compactNumber(Math.max(outside, total - matches))}</span>` : ""}<span><b>Rework expansion</b>+${compactNumber(nonnegativeInteger(metadata.ma1_discovery_rework_expansion, 0))} candidates requested</span></div>
      ${rows ? `<div class="research-table-wrap"><table class="research-table screening-table"><thead><tr><th>Rank</th><th>Target</th><th>Association</th><th>Source DB</th><th>Scope disposition</th></tr></thead><tbody>${rows}</tbody></table></div>` : legacyNotice}
    </section>`;
  }

  const MA1_EXPLICIT_SCOPE_IDENTITY_REASONS = Object.freeze({
    EXPLICIT_TARGET_FORBIDDEN: "요청 제약에서 금지된 표적",
    EXPLICIT_TARGET_DISEASE_IDENTITY_NOT_RESOLVED: "질환 식별이 완료되지 않아 표적 ID 조회를 시작하지 않음",
    EXPLICIT_TARGET_IDENTITY_RESOLVER_UNAVAILABLE: "정확 표적 ID 확인기를 사용할 수 없음",
    EXPLICIT_TARGET_FIXTURE_IDENTITY_REJECTED: "fixture 표적 ID이므로 연구 실행 범위에서 제외",
    EXPLICIT_TARGET_IDENTITY_NOT_RESOLVED: "정확한 사람 표적 ID를 확인하지 못함",
    EXPLICIT_TARGET_CANONICAL_SYMBOL_MISMATCH: "요청 표적과 확인된 공식 심볼이 일치하지 않음",
    EXPLICIT_TARGET_STABLE_IDENTIFIERS_INCOMPLETE: "Ensembl 또는 UniProt 안정 식별자가 불완전함",
    EXPLICIT_TARGET_STRUCTURE_IDENTITY_NOT_READY: "구조 조회에 필요한 UniProt 식별자가 준비되지 않음",
    EXPLICIT_TARGET_IDENTITY_SOURCE_CONFLICT: "외부 ID 소스 간 충돌이 확인됨",
    EXPLICIT_TARGET_IDENTITY_PROVENANCE_MISSING: "표적 ID 출처 기록이 없음",
    EXPLICIT_TARGET_EXTERNAL_IDENTITY_ATTESTATION_MISSING: "외부 실시간·캐시 ID 확인 기록이 없음",
    EXPLICIT_TARGET_SYMBOL_ATTESTATION_MISSING: "공식 심볼 확인 기록이 없음",
    EXPLICIT_TARGET_ENSEMBL_ATTESTATION_MISSING: "Ensembl ID 확인 기록이 없음",
    EXPLICIT_TARGET_UNIPROT_ATTESTATION_MISSING: "UniProt ID 확인 기록이 없음",
  });

  function ma1ExplicitScopeIdentityReasonLabel(reasonCode) {
    return MA1_EXPLICIT_SCOPE_IDENTITY_REASONS[String(reasonCode || "")] || "기록된 허용 사유 코드가 아님";
  }

  function ma1ExplicitScopeProjection(packet = {}) {
    const metadata = packet?.metadata && typeof packet.metadata === "object" ? packet.metadata : {};
    const ledger = metadata.ma1_explicit_scope_conservation_ledger;
    const rawTrace = metadata.ma1_request_scope_identity_trace;
    const laneUsed = metadata.ma1_request_scope_identity_lane_used === true;
    const explicitLedger = Boolean(
      ledger
      && typeof ledger === "object"
      && !Array.isArray(ledger)
      && ledger.target_scope_mode === "explicit_only"
    );
    const present = laneUsed || explicitLedger || (Array.isArray(rawTrace) && rawTrace.length > 0);
    if (!present) return {present: false, valid: false, historical: false};
    if (!ledger || typeof ledger !== "object" || Array.isArray(ledger) || !Array.isArray(rawTrace)) {
      return {present: true, valid: false, historical: true};
    }

    const sha256 = (value) => typeof value === "string" && /^[0-9a-f]{64}$/i.test(value);
    const boundedInteger = (value) => Number.isInteger(value) && value >= 0 && value <= 64;
    const symbolValid = (value) => typeof value === "string" && value.length >= 1 && value.length <= 80 && !/\s/.test(value);
    const symbolKey = (value) => String(value || "").trim().toLowerCase();
    const collectionNames = [
      "requested",
      "dynamic_exact_match",
      "identity_queried",
      "identity_resolved",
      "scheduled",
      "screened",
      "retained",
      "excluded",
    ];
    const collections = ledger.collections;
    if (
      ledger.schema_version !== "ma1_explicit_scope_conservation_ledger/v1"
      || ledger.authoritative !== true
      || ledger.target_scope_mode !== "explicit_only"
      || ledger.request_identity_is_query_scope_not_scientific_support !== true
      || ledger.target_fallback_used !== false
      || ledger.conservation_valid !== true
      || !Array.isArray(ledger.unexpected_generated_symbols)
      || ledger.unexpected_generated_symbols.length !== 0
      || !sha256(ledger.ledger_sha256)
      || !collections
      || typeof collections !== "object"
      || Array.isArray(collections)
    ) {
      return {present: true, valid: false, historical: false};
    }

    const normalizedCollections = {};
    for (const name of collectionNames) {
      const value = collections[name];
      if (!value || typeof value !== "object" || Array.isArray(value) || !Array.isArray(value.symbols)) {
        return {present: true, valid: false, historical: false};
      }
      const symbols = value.symbols;
      const symbolKeys = symbols.map(symbolKey);
      if (
        !boundedInteger(value.count)
        || value.count !== symbols.length
        || symbols.length > 64
        || symbols.some((symbol) => !symbolValid(symbol))
        || new Set(symbolKeys).size !== symbols.length
        || !sha256(value.order_sha256)
        || !sha256(value.set_sha256)
      ) {
        return {present: true, valid: false, historical: false};
      }
      normalizedCollections[name] = {symbols: [...symbols], keys: new Set(symbolKeys)};
    }
    const requested = normalizedCollections.requested;
    if (!requested.symbols.length) return {present: true, valid: false, historical: false};
    if (collectionNames.slice(1).some((name) => (
      [...normalizedCollections[name].keys].some((key) => !requested.keys.has(key))
    ))) {
      return {present: true, valid: false, historical: false};
    }
    if ([...normalizedCollections.retained.keys].some((key) => normalizedCollections.excluded.keys.has(key))) {
      return {present: true, valid: false, historical: false};
    }

    const allowedOrigins = new Set(["disease_ranked_exact_match", "explicit_request_scope_identity"]);
    const allowedScheduling = new Set(["scheduled_for_ma1", "unscheduled_capacity_or_duplicate", "not_eligible"]);
    const allowedSources = new Set(["Open Targets", "UniProt", "fixture"]);
    const allowedSourceStatuses = new Set(["ok", "empty", "error", "skipped"]);
    const allowedProvenance = new Set(["external_live", "external_cache", "policy_control", "fixture_fallback"]);
    const allowedMatchedBy = new Set(["gene_symbol", "ensembl_id", "fixture"]);
    const trace = [];
    for (const item of rawTrace) {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return {present: true, valid: false, historical: false};
      }
      const requestedSymbol = item.requested_symbol;
      const reasonCode = item.reason_code;
      const disposition = String(item.disposition || "");
      const scheduling = String(item.scheduling_disposition || "");
      const outcomes = item.identity_source_outcomes;
      if (
        !symbolValid(requestedSymbol)
        || !requested.keys.has(symbolKey(requestedSymbol))
        || !allowedOrigins.has(String(item.origin || ""))
        || typeof item.query_attempted !== "boolean"
        || typeof item.seed_created !== "boolean"
        || item.scientific_support_created !== false
        || item.fixture !== false
        || !allowedScheduling.has(scheduling)
        || !Array.isArray(outcomes)
        || outcomes.length > 4
        || !sha256(item.identity_source_outcomes_sha256)
        || !sha256(item.row_sha256)
      ) {
        return {present: true, valid: false, historical: false};
      }
      const normalizedOutcomes = [];
      for (const outcome of outcomes) {
        if (
          !outcome
          || typeof outcome !== "object"
          || Array.isArray(outcome)
          || !allowedSources.has(String(outcome.source || ""))
          || !allowedSourceStatuses.has(String(outcome.status || ""))
          || typeof outcome.queried !== "boolean"
          || typeof outcome.cache_hit !== "boolean"
          || !allowedProvenance.has(String(outcome.provenance_class || ""))
          || (outcome.matched_by != null && !allowedMatchedBy.has(String(outcome.matched_by)))
          || (outcome.canonical_symbol != null && !symbolValid(outcome.canonical_symbol))
          || (outcome.ensembl_id != null && !/^ENSG\d{11}$/.test(String(outcome.ensembl_id)))
          || (outcome.uniprot_accession != null && (
            typeof outcome.uniprot_accession !== "string"
            || !outcome.uniprot_accession
            || outcome.uniprot_accession.length > 30
            || /\s/.test(outcome.uniprot_accession)
          ))
        ) {
          return {present: true, valid: false, historical: false};
        }
        normalizedOutcomes.push({
          source: String(outcome.source),
          status: String(outcome.status),
          queried: outcome.queried,
          cacheHit: outcome.cache_hit,
          provenanceClass: String(outcome.provenance_class),
          matchedBy: outcome.matched_by == null ? null : String(outcome.matched_by),
          canonicalSymbol: outcome.canonical_symbol == null ? null : String(outcome.canonical_symbol),
          ensemblId: outcome.ensembl_id == null ? null : String(outcome.ensembl_id),
          uniprotAccession: outcome.uniprot_accession == null ? null : String(outcome.uniprot_accession),
        });
      }
      if (disposition === "resolved") {
        const canonical = String(item.canonical_symbol || "");
        const ensembl = String(item.ensembl_id || "");
        const uniprot = String(item.uniprot_id || "");
        const externalOk = normalizedOutcomes.filter((outcome) => (
          outcome.status === "ok" && ["external_live", "external_cache"].includes(outcome.provenanceClass)
        ));
        if (
          reasonCode != null
          || item.seed_created !== true
          || item.identity_status !== "resolved"
          || symbolKey(canonical) !== symbolKey(requestedSymbol)
          || !/^ENSG\d{11}$/.test(ensembl)
          || !uniprot
          || uniprot.length > 30
          || /\s/.test(uniprot)
          || scheduling === "not_eligible"
          || item.query_attempted !== true
          || !externalOk.some((outcome) => symbolKey(outcome.canonicalSymbol) === symbolKey(requestedSymbol))
          || !externalOk.some((outcome) => String(outcome.ensemblId || "").toLowerCase() === ensembl.toLowerCase())
          || !externalOk.some((outcome) => String(outcome.uniprotAccession || "").toLowerCase() === uniprot.toLowerCase())
        ) {
          return {present: true, valid: false, historical: false};
        }
      } else if (
        disposition !== "rejected"
        || item.seed_created !== false
        || scheduling !== "not_eligible"
        || !Object.hasOwn(MA1_EXPLICIT_SCOPE_IDENTITY_REASONS, String(reasonCode || ""))
      ) {
        return {present: true, valid: false, historical: false};
      }
      const hasFixtureOutcome = normalizedOutcomes.some((outcome) => (
        outcome.source === "fixture"
        || outcome.provenanceClass === "fixture_fallback"
        || outcome.matchedBy === "fixture"
      ));
      if (
        hasFixtureOutcome !== (
          disposition === "rejected"
          && reasonCode === "EXPLICIT_TARGET_FIXTURE_IDENTITY_REJECTED"
        )
      ) {
        return {present: true, valid: false, historical: false};
      }
      trace.push({
        requestedSymbol: String(requestedSymbol),
        origin: String(item.origin),
        queryAttempted: item.query_attempted,
        canonicalSymbol: String(item.canonical_symbol || ""),
        ensemblId: String(item.ensembl_id || ""),
        uniprotId: String(item.uniprot_id || ""),
        disposition,
        reasonCode: reasonCode == null ? null : String(reasonCode),
        scheduling,
        outcomes: normalizedOutcomes,
      });
    }
    if (
      trace.length !== requested.symbols.length
      || new Set(trace.map((item) => symbolKey(item.requestedSymbol))).size !== trace.length
      || trace.some((item) => !requested.keys.has(symbolKey(item.requestedSymbol)))
      || !boundedInteger(metadata.ma1_request_scope_identity_query_count)
      || metadata.ma1_request_scope_identity_query_count !== trace.filter((item) => (
        item.origin === "explicit_request_scope_identity" && item.queryAttempted
      )).length
      || !boundedInteger(metadata.ma1_request_scope_identity_seed_count)
      || metadata.ma1_request_scope_identity_seed_count !== trace.filter((item) => (
        item.origin === "explicit_request_scope_identity"
        && item.disposition === "resolved"
        && item.scheduling === "scheduled_for_ma1"
      )).length
      || !sha256(metadata.ma1_request_scope_identity_trace_sha256)
    ) {
      return {present: true, valid: false, historical: false};
    }
    const traceKeys = (predicate) => new Set(trace.filter(predicate).map((item) => symbolKey(item.requestedSymbol)));
    const sameSet = (left, right) => left.size === right.size && [...left].every((key) => right.has(key));
    const packetRetainedKeys = new Set(
      (Array.isArray(packet?.hypotheses) ? packet.hypotheses : [])
        .map((item) => symbolKey(item?.target?.symbol))
        .filter(Boolean),
    );
    const packetExcludedKeys = new Set(
      (Array.isArray(packet?.excluded) ? packet.excluded : [])
        .map((item) => symbolKey(item?.target_symbol))
        .filter((key) => key && requested.keys.has(key)),
    );
    const packetExcludedReasonCodes = new Map();
    for (const item of (Array.isArray(packet?.excluded) ? packet.excluded : [])) {
      const key = symbolKey(item?.target_symbol);
      if (key && !requested.keys.has(key)) continue;
      const reasonCodes = Array.isArray(item?.reason_codes) ? item.reason_codes : [];
      if (
        !key
        || reasonCodes.length > 20
        || reasonCodes.some((code) => typeof code !== "string" || !/^[A-Z0-9_]{1,120}$/.test(code))
      ) {
        return {present: true, valid: false, historical: false};
      }
      const existingReasons = packetExcludedReasonCodes.get(key) || [];
      const combinedReasons = [...new Set([...existingReasons, ...reasonCodes])];
      if (combinedReasons.length > 64) {
        return {present: true, valid: false, historical: false};
      }
      packetExcludedReasonCodes.set(key, combinedReasons);
    }
    const rawIdentityRejections = ledger.identity_rejections;
    if (!Array.isArray(rawIdentityRejections) || rawIdentityRejections.length > 64) {
      return {present: true, valid: false, historical: false};
    }
    const identityRejectionKeys = new Set();
    for (const rejection of rawIdentityRejections) {
      if (
        !rejection
        || typeof rejection !== "object"
        || Array.isArray(rejection)
        || !symbolValid(rejection.requested_symbol)
        || !requested.keys.has(symbolKey(rejection.requested_symbol))
        || !allowedOrigins.has(String(rejection.origin || ""))
        || !Object.hasOwn(MA1_EXPLICIT_SCOPE_IDENTITY_REASONS, String(rejection.reason_code || ""))
      ) {
        return {present: true, valid: false, historical: false};
      }
      identityRejectionKeys.add(`${symbolKey(rejection.requested_symbol)}|${rejection.reason_code}|${rejection.origin}`);
    }
    const traceRejectionKeys = new Set(trace.filter((item) => item.disposition === "rejected").map((item) => (
      `${symbolKey(item.requestedSymbol)}|${item.reasonCode}|${item.origin}`
    )));
    if (
      !sameSet(normalizedCollections.dynamic_exact_match.keys, traceKeys((item) => item.origin === "disease_ranked_exact_match"))
      || !sameSet(normalizedCollections.identity_queried.keys, traceKeys((item) => item.queryAttempted))
      || !sameSet(normalizedCollections.identity_resolved.keys, traceKeys((item) => item.disposition === "resolved"))
      || !sameSet(normalizedCollections.scheduled.keys, traceKeys((item) => item.scheduling === "scheduled_for_ma1"))
      || !sameSet(normalizedCollections.retained.keys, packetRetainedKeys)
      || !sameSet(normalizedCollections.excluded.keys, packetExcludedKeys)
      || !sameSet(identityRejectionKeys, traceRejectionKeys)
    ) {
      return {present: true, valid: false, historical: false};
    }
    const traceByRequestedSymbol = new Map(trace.map((item) => [symbolKey(item.requestedSymbol), item]));
    const orderedTrace = requested.symbols.map((symbol) => traceByRequestedSymbol.get(symbolKey(symbol)));
    if (orderedTrace.some((item) => !item)) {
      return {present: true, valid: false, historical: false};
    }
    return {
      present: true,
      valid: true,
      historical: false,
      trace: orderedTrace,
      collections: Object.fromEntries(collectionNames.map((name) => [name, normalizedCollections[name].symbols])),
      excludedReasonCodes: packetExcludedReasonCodes,
    };
  }

  function renderExplicitScopeIdentityTrace(packet = {}) {
    const projection = ma1ExplicitScopeProjection(packet);
    if (!projection.present) return "";
    if (!projection.valid) {
      const message = projection.historical
        ? "이 과거 실행에는 요청 표적별 정확 ID 확인 장부가 완전하게 저장되지 않았습니다. 현재 결과에서 행이나 탈락 이유를 추정하지 않습니다."
        : "요청 표적 보존 장부와 표적별 ID 추적 기록이 서로 일치하지 않아 상세 행을 숨겼습니다. 원본 packet을 감사해야 합니다.";
      return `<section class="ma1-result-section ma1-explicit-scope-results" aria-labelledby="ma1ExplicitScopeTitle">
        <header class="ma1-result-heading"><div><span>EXPLICIT TARGET SCOPE</span><h4 id="ma1ExplicitScopeTitle">사용자 지정 표적 보존 장부</h4><p>요청한 표적이 ID 확인과 MA1 심사에서 어떻게 처리됐는지 기록합니다.</p></div><strong>검증 불가</strong></header>
        <div class="screening-empty">${esc(message)}</div>
      </section>`;
    }
    const collections = projection.collections;
    const originLabel = (origin) => origin === "disease_ranked_exact_match"
      ? "질환 후보 풀 exact match"
      : "요청 범위 exact identity 조회";
    const schedulingLabel = (value) => value === "scheduled_for_ma1"
      ? "MA1 심사 배정"
      : value === "unscheduled_capacity_or_duplicate"
      ? "용량 또는 중복으로 미배정"
      : "ID 검증 탈락으로 미배정";
    const rows = projection.trace.map((item) => {
      const sourceText = item.outcomes.length
        ? item.outcomes.map((outcome) => `${outcome.source} · ${outcome.status} · ${outcome.provenanceClass}${outcome.cacheHit ? " · cache" : ""}`).join(" / ")
        : item.queryAttempted ? "조회 결과 행 미기록" : "조회하지 않음";
      const stableIds = [
        item.canonicalSymbol ? `Canonical ${item.canonicalSymbol}` : "",
        item.ensemblId ? `Ensembl ${item.ensemblId}` : "",
        item.uniprotId ? `UniProt ${item.uniprotId}` : "",
      ].filter(Boolean);
      const outcome = item.disposition === "resolved"
        ? '<span class="screening-status is-retained">정확 ID 확인</span><small>질환·작용 근거 생성 0건 · 별도 MA1 심사 필요</small>'
        : `<span class="screening-status is-rejected">ID 단계 제외</span><small>${esc(ma1ExplicitScopeIdentityReasonLabel(item.reasonCode))}<br><code>${esc(item.reasonCode)}</code></small>`;
      const requestedKey = String(item.requestedSymbol || "").trim().toLowerCase();
      const retained = collections.retained.some((symbol) => String(symbol).trim().toLowerCase() === requestedKey);
      const excluded = collections.excluded.some((symbol) => String(symbol).trim().toLowerCase() === requestedKey);
      const screened = collections.screened.some((symbol) => String(symbol).trim().toLowerCase() === requestedKey);
      const exclusionReasons = projection.excludedReasonCodes.get(requestedKey) || [];
      const finalOutcome = item.disposition === "rejected"
        ? '<span class="screening-status is-rejected">Identity 단계 종료</span><small>과학적 가설 심사 전 제외</small>'
        : retained
        ? '<span class="screening-status is-retained">Retained by MA1</span><small>G0 packet 감사 대상으로 전달</small>'
        : excluded
        ? `<span class="screening-status is-rejected">Not retained</span><small>${exclusionReasons.length ? exclusionReasons.map((code) => `<code>${esc(code)}</code>`).join(" ") : "packet exclusion 사유 코드 미기록"}</small>`
        : screened
        ? '<span class="screening-status">심사 완료</span><small>retained/excluded 최종 기록 없음 · 추정하지 않음</small>'
        : item.scheduling === "scheduled_for_ma1"
        ? '<span class="screening-status">심사 배정</span><small>screening 결과 미기록</small>'
        : '<span class="screening-status">미심사</span><small>MA1 배정되지 않음</small>';
      return `<tr class="screening-row ${item.disposition === "rejected" ? "is-excluded" : ""}">
        <td><strong>${esc(item.requestedSymbol)}</strong><small>${esc(originLabel(item.origin))}</small></td>
        <td>${stableIds.length ? esc(stableIds.join(" · ")) : "확인된 stable ID 없음"}</td>
        <td>${esc(sourceText)}</td>
        <td><span class="quiet-status">${esc(schedulingLabel(item.scheduling))}</span></td>
        <td>${outcome}</td>
        <td>${finalOutcome}</td>
      </tr>`;
    }).join("");
    return `<section class="ma1-result-section ma1-explicit-scope-results" aria-labelledby="ma1ExplicitScopeTitle">
      <header class="ma1-result-heading"><div><span>EXPLICIT TARGET SCOPE</span><h4 id="ma1ExplicitScopeTitle">사용자 지정 표적 보존 장부</h4><p>요청한 모든 표적의 exact identity 확인, MA1 배정, 심사와 최종 상태를 분리해 표시합니다.</p></div><strong>Requested ${compactNumber(collections.requested.length)}</strong></header>
      <div class="screening-projection-summary" aria-label="MA1 explicit target conservation counts">
        <span><b>Identity</b>queried ${compactNumber(collections.identity_queried.length)} · resolved ${compactNumber(collections.identity_resolved.length)}</span>
        <span><b>MA1</b>scheduled ${compactNumber(collections.scheduled.length)} · screened ${compactNumber(collections.screened.length)}</span>
        <span><b>Outcome</b>retained ${compactNumber(collections.retained.length)} · excluded ${compactNumber(collections.excluded.length)}</span>
      </div>
      <p class="screening-boundary"><b>해석 경계</b> · exact identity 확인은 사용자가 지정한 표적 이름과 stable ID의 조회 범위만 확정합니다. 질환 연관성, 작용 방향 또는 유효성 근거를 만들지 않으며, 기존 MA1 근거 검색과 Critic 기준을 그대로 통과해야 합니다.</p>
      <p class="report-footnote">화면은 저장된 SHA-256 형식과 packet 내부 식별자·상태 결속을 대조합니다. 저장 해시의 원문 재계산은 하지 않으며, 전체 원본은 RUN REPORT의 Technical Data에서 확인합니다.</p>
      <div class="research-table-wrap"><table class="research-table screening-table">
        <caption class="sr-only">사용자가 지정한 모든 표적의 정확 ID 확인 및 MA1 처리 기록</caption>
        <thead><tr><th>Requested target / origin</th><th>Verified identity</th><th>Source library</th><th>MA1 allocation</th><th>Identity outcome / reason</th><th>Final MA1 status / reason</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
    </section>`;
  }

  function ma1RevisionReworkHtml(snapshot = {}, packet = {}) {
    const events = Array.isArray(snapshot?.events) ? snapshot.events : [];
    const revision = packet?.metadata?.revision_lineage && typeof packet.metadata.revision_lineage === "object"
      ? packet.metadata.revision_lineage
      : null;
    const reworkEvents = events.filter((event) => (
      String(event?.event_type || "") === "rework_order"
      || String(event?.metadata?.interaction?.kind || "") === "selective_rework"
    ));
    const completionEvents = events.filter((event) => String(event?.event_type || "") === "ma1_completed");
    const disclosureKey = `ma1-rework:${packet?.packet_id || snapshot?.run_id || "run"}`;
    const booleanRecord = (value) => value === true ? "있음" : value === false ? "없음" : "미기록";
    const eventHtml = reworkEvents.length
      ? `<div class="evidence-record-list ma1-rework-events">${reworkEvents.map((event) => {
          const interaction = event?.metadata?.interaction || {};
          const details = interaction.details || {};
          const affectedClaims = Array.isArray(details.affected_claim_ids) ? details.affected_claim_ids.length : 0;
          const affectedCandidates = Array.isArray(details.affected_candidate_ids) ? details.affected_candidate_ids.length : 0;
          const invalidated = Array.isArray(event?.metadata?.invalidated_nodes) ? event.metadata.invalidated_nodes.length : Array.isArray(details.invalidated_nodes) ? details.invalidated_nodes.length : 0;
          return `<article><b>${esc(event?.metadata?.target_stage || interaction.receiver || "재작업 단계 미기록")}</b><span>${esc(event?.message || interaction.summary || "선택 재작업")}</span><small>${esc(formatTime(event?.created_at))} · affected claims ${compactNumber(affectedClaims)} · candidates ${compactNumber(affectedCandidates)} · invalidated nodes ${compactNumber(invalidated)}</small></article>`;
        }).join("")}</div>`
      : '<p class="screening-detail-empty">저장된 MA1 selective-rework event 없음</p>';
    return `<details class="artifact-methodology ma1-rework-summary" data-disclosure="${esc(disclosureKey)}">
      <summary><b>Packet revision / rework summary</b><span>MA1 iterations ${compactNumber(nonnegativeInteger(snapshot?.ma1_iterations, packet?.iterations || 0))} · rework loops ${compactNumber(nonnegativeInteger(snapshot?.rework_loops, 0))} · packet artifacts ${compactNumber(completionEvents.length || (packet?.packet_id ? 1 : 0))}</span></summary>
      <div class="ma1-rework-body">
        <p class="screening-boundary"><b>표시 범위</b> current packet의 최신 direct lineage와 snapshot event 요약입니다. 과거 packet 내용 전체를 복원하거나 변경 diff를 추정하지 않습니다.</p>
        <dl class="context-facts">
          <div><dt>Current packet</dt><dd>${esc(packet?.packet_id || "미기록")}</dd></div>
          <div><dt>Latest revision</dt><dd>${revision ? compactNumber(nonnegativeInteger(revision.revision_number, 0)) : "lineage 미기록"}</dd></div>
          <div><dt>Parent packet</dt><dd>${esc(revision?.parent_packet_id || "미기록")}</dd></div>
          <div><dt>Rework order</dt><dd>${esc(revision?.rework_order_id || "미기록")}</dd></div>
          <div><dt>Scientific content changed</dt><dd>${booleanRecord(revision?.scientific_changed)}</dd></div>
          <div><dt>Audit trace changed</dt><dd>${booleanRecord(revision?.audit_trace_changed)}</dd></div>
        </dl>
        <section class="ma1-rework-event-section"><h5>Selective rework events</h5>${eventHtml}</section>
      </div>
    </details>`;
  }

  function ma1AdmissionProjection(packet = {}) {
    const approved = Array.isArray(packet.hypotheses) ? packet.hypotheses.filter((item) => item && typeof item === "object") : [];
    const provisional = Array.isArray(packet.provisional_hypotheses) ? packet.provisional_hypotheses.filter((item) => item && typeof item === "object") : [];
    const report = packet.metadata?.ma3_admission_report;
    const enabled = packet.metadata?.admission_policy === "ma3-source-bound-v1" || provisional.length > 0 || Boolean(report);
    if (!enabled) return { enabled: false, rows: [], submittedCount: approved.length, approvedCount: approved.length, reviewed: false, inconsistent: false };
    const entries = Array.isArray(report?.entries) ? report.entries.filter((item) => item && typeof item === "object") : [];
    const reviewed = Array.isArray(report?.entries);
    const approvedById = new Map(approved.filter((item) => item.hypothesis_id).map((item) => [String(item.hypothesis_id), item]));
    const rowsById = new Map();
    const add = (item, entry = null) => {
      const id = String(item?.hypothesis_id || entry?.hypothesis_id || "");
      if (!id) return;
      if (!rowsById.has(id)) rowsById.set(id, { id, hypothesis: item || null, entry: null, duplicate: false });
      const row = rowsById.get(id);
      if (item && !row.hypothesis) row.hypothesis = item;
      if (entry) {
        row.duplicate = Boolean(row.entry);
        row.entry = entry;
      }
    };
    provisional.forEach((item) => add(item));
    entries.forEach((entry) => add(entry.original_proposal && typeof entry.original_proposal === "object" ? entry.original_proposal : null, entry));
    approved.forEach((item) => add(item));
    let inconsistent = false;
    const rows = [...rowsById.values()].map((row) => {
      const rawStatus = String(row.entry?.status || "");
      const hasApprovedArtifact = approvedById.has(row.id);
      const invalid = row.duplicate || (rawStatus === "approved") !== hasApprovedArtifact
        || (row.entry && !["approved", "needs_evidence", "hold", "rejected"].includes(rawStatus))
        || Boolean(row.entry?.hypothesis_id && String(row.entry.hypothesis_id) !== row.id);
      const status = invalid ? "hold" : row.entry ? rawStatus : reviewed ? "hold" : "pending";
      inconsistent ||= invalid;
      return { ...row, hypothesis: status === "approved" ? approvedById.get(row.id) : row.hypothesis,
        target: String(row.entry?.target_symbol || row.hypothesis?.target?.symbol || "표적 미기록"), status,
        reason: invalid ? "MA3 심사 기록과 승인 packet이 일치하지 않아 승인으로 표시하지 않습니다."
          : String(row.entry?.reason || (reviewed ? "가설별 심사 결과 미기록 · 승인하지 않음" : "MA1이 제출한 잠정 가설입니다. MA3의 원문 근거 심사를 기다립니다.")),
        limitations: Array.isArray(row.entry?.limitations) ? row.entry.limitations.map(String) : [],
      };
    });
    const approvedCount = rows.filter((row) => row.status === "approved").length;
    const recordedSubmitted = report?.submitted_count;
    const submittedCount = Number.isInteger(recordedSubmitted) && recordedSubmitted >= rows.length ? recordedSubmitted : rows.length;
    inconsistent ||= reviewed && ((Number.isInteger(recordedSubmitted) && recordedSubmitted !== rows.length)
      || (Number.isInteger(report.approved_count) && report.approved_count !== approvedCount));
    return { enabled, rows, submittedCount, approvedCount, reviewed, inconsistent, report };
  }

  function ma3AdmissionDispositionHtml(row, options = {}) {
    if (!row) return "";
    const labels = { pending: "심사 대기 · 잠정", approved: "MA3 승인 · 제한 범위", needs_evidence: "근거 보강 필요", hold: "보류", rejected: "기각" };
    const status = Object.hasOwn(labels, row.status) ? row.status : "hold";
    const reason = String(row.reason || "판정 이유 미기록");
    const limitations = Array.isArray(row.limitations) ? row.limitations : [];
    const bindings = Array.isArray(row.entry?.source_bindings) ? row.entry.source_bindings.filter((item) => item && typeof item === "object") : [];
    const omissions = row.entry?.omissions;
    const detail = options.detail === true;
    return `<div class="ma3-admission-disposition is-${status}" role="note"><span class="screening-status admission-${status}">${labels[status]}</span>
      <p>${esc(detail ? reason : reason.slice(0, 280))}${!detail && reason.length > 280 ? "…" : ""}</p>
      ${detail && limitations.length ? `<b>적용 범위·남은 한계</b><ul>${limitations.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}
      ${detail && omissions ? `<p>MA3 검토 범위: 확보 ${esc(omissions.available_records ?? "미기록")}건 · 검토 ${esc(omissions.reviewed_records ?? "미기록")}건 · 제외/미포함 ${esc(Array.isArray(omissions.omitted_records) ? omissions.omitted_records.length : "미기록")}건. 전체 문헌을 검토했다는 의미는 아닙니다.</p>` : ""}
      ${detail && bindings.length ? `<details><summary>MA3 심사 출처·해시 ${bindings.length}건</summary><ul>${bindings.map((item) => `<li>${esc(item.source || "출처 미기록")} · ${esc(item.external_id || "ID 미기록")}<small class="admission-report-hash">Evidence ${esc(item.evidence_id || "미기록")}<br>Source binding ${esc(item.source_binding_sha256 || "미기록")}<br>Source text ${esc(item.source_text_sha256 || "미기록")}</small></li>`).join("")}</ul></details>` : ""}
      ${detail ? '<small>MA3 승인은 기록된 가설 범위의 다음 단계 검토 허용입니다. G0 전체 PASS, 후보 검증 또는 임상 효능 확인과는 다릅니다.</small>' : ""}</div>`;
  }

  function renderMa3AdmissionHypotheses(packet, options = {}) {
    const admission = ma1AdmissionProjection(packet);
    const rows = admission.rows;
    return `<section class="ma1-result-section ma3-admission-results" aria-label="MA1 제안과 MA3 가설별 심사">
      <header class="ma1-result-heading"><div><span>PROPOSE → REVIEW</span><h4>MA1 제안 · MA3 심사</h4><p>MA1의 잠정 가설과 MA3의 승인 가설을 구분합니다. 보강·보류·기각된 가설은 승인 목록과 분리되며, MA2 진행에는 G0 전체 판정도 충족해야 합니다.</p></div><strong>제안 ${compactNumber(admission.submittedCount)} · 승인 ${compactNumber(admission.approvedCount)}</strong></header>
      ${admission.inconsistent ? '<p class="admission-integrity-warning" role="alert">가설별 기록·집계에 불일치가 있습니다. 불일치 항목은 승인으로 표시하지 않으며 원본 감사 기록 확인이 필요합니다.</p>' : ""}
      ${!rows.length ? '<div class="retained-empty">기록된 심사 대상 가설이 없습니다.</div>' : `<div class="ma3-admission-cards">${rows.map((row) => {
        const hypothesis = row.hypothesis;
        const evidence = Array.isArray(hypothesis?.evidence) ? hypothesis.evidence : [];
        return `<article class="ma3-admission-card is-${row.status}"><header><div><h5>${esc(row.target)}</h5><small>${esc(row.id)}</small></div><span>${esc(directionLabel(hypothesis?.intervention_direction))}</span></header>
          ${ma3AdmissionDispositionHtml(row)}
          ${hypothesis ? `<p class="admission-proposal-text">${esc(hypothesis.mechanism || hypothesis.claims?.[0]?.text || "가설 서술 미기록")}</p>${hypothesisSourceToolBadges(hypothesis)}<small>제출 근거 ${compactNumber(evidence.length)}건 · 제출 수는 검증 완료 수가 아닙니다.</small>` : '<p>가설 원문은 이 보고서에 포함되지 않았습니다. 저장된 원본 artifact를 확인하십시오.</p>'}
          <details class="admission-review-details" data-disclosure="admission:${esc(row.id)}"><summary>판정 이유·적용 범위 상세</summary>${ma3AdmissionDispositionHtml(row, { detail: true })}
            ${hypothesis?.claims?.length ? `<b>심사 대상 주장</b><ul>${hypothesis.claims.map((claim) => `<li>${esc(claim.text || "주장 미기록")}</li>`).join("")}</ul>` : ""}
            ${evidence.length ? `<b>출처 기록</b><ul>${evidence.slice(0, 20).map((item) => `<li>${evidenceLink(item)}</li>`).join("")}</ul>${evidence.length > 20 ? '<small>상위 20개 출처 표시 · 전체는 근거 상세에서 확인</small>' : ""}` : ""}
            ${admission.report?.report_sha256 ? `<small class="admission-report-hash">MA3 report SHA-256: ${esc(admission.report.report_sha256)}</small>` : ""}
          </details>
          ${hypothesis && !options.report ? `<button class="table-link" type="button" data-context-kind="hypothesis" data-context-id="${esc(row.id)}">주장·근거·출처 상세 보기</button>` : ""}
        </article>`;
      }).join("")}</div>`}
    </section>`;
  }

  function renderMa3AdmissionPacketArtifacts(packet) {
    return `${renderMa3AdmissionHypotheses(packet)}${renderDiscoveryCandidateTrace(packet)}${renderScreenedProposals(packet, ma1ScreeningProjection(packet))}${evidenceLinkageTraceHtml(packet)}${ma1RevisionReworkHtml(state.snapshot || {}, packet)}`;
  }

  function renderHypothesesArtifact() {
    const packet = state.snapshot?.hypothesis_packet;
    const hypotheses = packet?.hypotheses || [];
    const intake = getIntakeAssessment(state.snapshot || {});
    if (intake.diseaseNeedsReview) {
      return `<div class="artifact-blocked-state">
        <span aria-hidden="true">?</span>
        <b>질환 확인 필요</b>
        <p>질환이 확인되지 않은 실행의 가설 ${compactNumber(hypotheses.length)}건은 과학적 결과로 제시하지 않습니다.</p>
        <small>원본 JSON에는 감사 추적을 위해 기존 fixture 기록이 보존됩니다.</small>
      </div>`;
    }
    if (!packet) return '<div class="artifact-empty">MA1 결과 packet이 아직 생성되지 않았습니다.</div>';
    if (ma1AdmissionProjection(packet).enabled) {
      return renderMa3AdmissionPacketArtifacts(packet);
    }
    if (
      !hypotheses.length
      && (intake.verificationUnavailable || intake.hasMismatch)
    ) {
      return `<div class="artifact-blocked-state">
        <span aria-hidden="true">?</span>
        <b>${intake.verificationUnavailable ? "외부 질환 검증 불가" : "연구 입력 확인 필요"}</b>
        <p>${intake.hasMismatch ? "서로 일치하지 않는 연구 입력을 먼저 명확히 해야 하므로 가설을 만들지 않았습니다." : "근거가 확인되지 않은 질환에 대해 표적이나 가설을 만들지 않았습니다."}</p>
        <small>Retained 0 · 가설 패킷 점수 미산정</small>
      </div>`;
    }
    const g0 = state.snapshot?.g0_verdict;
    const screening = ma1ScreeningProjection(packet);
    const trackSections = [
      {
        key: "literature",
        title: "Literature-supported",
        description: "질환–표적 연관성과 공개 문헌을 중심으로 구성된 가설",
      },
      {
        key: "tbio",
        title: "Tbio · underexplored",
        description: "상대적으로 연구·개발 근거가 제한된 표적을 별도 공급 경로로 탐색한 가설",
      },
    ];
    const rowsHtml = (items) => items.map((hypothesis, index) => {
          const counts = hypothesisEvidenceCounts(hypothesis);
          const support = counts.support;
          const counter = counts.counter;
          const id = hypothesis.hypothesis_id || hypothesis.target?.symbol || `hypothesis-${index}`;
          const provenance = hypothesisProvenance(hypothesis);
          const counterSearch = hypothesis.target?.metadata?.counter_evidence_search || {};
          const narrative = hypothesisNarrative(hypothesis);
          const counterLabel = counter
            ? `${compactNumber(counter)} records`
            : counterSearch.status === "completed_empty"
            ? "0 · 검색 완료"
            : "0 · 결과 미확인";
          return `<tr class="research-row ${provenance.fixtureOnly ? "is-fixture" : ""}" data-context-kind="hypothesis" data-context-id="${esc(id)}">
            <td class="rank-cell">${esc(hypothesis.rank || index + 1)}</td>
            <td><strong>${esc(hypothesis.target?.symbol || "—")}</strong><small>${hypothesis.target?.uniprot_id ? esc(hypothesis.target.uniprot_id) : "ID 확인 중"}</small></td>
            <td><span class="quiet-status">${esc(directionLabel(hypothesis.intervention_direction))}</span></td>
            <td class="narrative-cell"><div class="hypothesis-language-line"><span>${esc(narrative.language)}</span><small>가설 패킷 완성도 ${formatScore(hypothesis.score)} · 과학적 신뢰도 아님${provenance.fixtureOnly ? " · FIXTURE ONLY" : ""}</small></div>${hypothesisNarrativeBody(hypothesis)}${hypothesisSourceToolBadges(hypothesis)}</td>
            <td><div class="hypothesis-evidence-counts is-vertical"><button class="table-link" type="button" aria-label="${esc(`${hypothesis.target?.symbol || "가설"} 지지 근거와 상세 보기`)}" data-context-kind="hypothesis" data-context-id="${esc(id)}"><b>${compactNumber(support)}</b>지지</button><span><b>${compactNumber(counter)}</b>반증</span><span><b>${compactNumber(counts.context)}</b>맥락</span></div></td>
            <td><span class="${counter ? "warning-text" : "muted-text"}">${esc(counterLabel)}</span></td>
            <td><span class="screening-status is-retained">Retained by MA1</span></td>
          </tr>`;
        }).join("");
    const retainedHtml = hypotheses.length
      ? `<div class="hypothesis-track-groups">${trackSections.map((section) => {
          const items = hypotheses
            .filter((hypothesis) => String(hypothesis.track || "").toLowerCase() === section.key)
            .sort((a, b) => Number(a.rank || 999) - Number(b.rank || 999) || Number(b.score || 0) - Number(a.score || 0));
          return `<section class="hypothesis-track-section track-${esc(section.key)}">
            <header><div><span class="track-kicker">${section.key === "literature" ? "LITERATURE TRACK" : "TBIO TRACK"}</span><h4>${esc(section.title)}</h4><p>${esc(section.description)}</p></div><strong>Retained ${compactNumber(items.length)}</strong></header>
            ${items.length ? `<div class="research-table-wrap"><table class="research-table hypothesis-table">
              <colgroup>
                <col class="hypothesis-col-rank">
                <col class="hypothesis-col-target">
                <col class="hypothesis-col-direction">
                <col class="hypothesis-col-narrative">
                <col class="hypothesis-col-evidence">
                <col class="hypothesis-col-counter">
                <col class="hypothesis-col-status">
              </colgroup>
              <thead><tr><th>Rank</th><th>Target</th><th>Direction</th><th>Hypothesis</th><th>Evidence</th><th>Counter-evidence</th><th>MA1 status</th></tr></thead>
              <tbody>${rowsHtml(items)}</tbody>
            </table></div>` : '<div class="track-empty">이 공급 경로에서 retained hypothesis가 없습니다.</div>'}
          </section>`;
        }).join("")}</div>`
      : '<div class="retained-empty"><b>Retained hypotheses 0</b><p>MA1이 packet을 생성했지만 현재 retained hypothesis는 없습니다. 아래 screening·exclusion 기록은 가설로 간주하지 않습니다.</p></div>';
    return `<div class="artifact-table-intro"><p>Current MA1 packet의 retained hypotheses와 retained가 아닌 screening 기록을 분리해 표시합니다. Retained 행을 선택하면 Claim → Evidence → Provenance 관계를 확인할 수 있습니다.</p><span>Packet-level G0 audit · ${esc(g0?.verdict || "PENDING")}</span></div>
      <section class="ma1-result-section ma1-retained-results" aria-labelledby="ma1RetainedTitle">
        <header class="ma1-result-heading"><div><span>CURRENT PACKET</span><h4 id="ma1RetainedTitle">Retained hypotheses</h4><p>MA1 Selector가 현재 packet에 보존한 가설만 아래 순위 표에 포함됩니다. G0는 개별 행이 아니라 이 packet 전체를 감사합니다.</p></div><strong>Retained ${compactNumber(hypotheses.length)}</strong></header>
        ${retainedHtml}
      </section>
      ${renderExplicitScopeIdentityTrace(packet)}
      ${renderDiscoveryCandidateTrace(packet)}
      ${renderScreenedProposals(packet, screening)}
      ${evidenceLinkageTraceHtml(packet)}
      ${ma1RevisionReworkHtml(state.snapshot || {}, packet)}`;
  }

  function evidenceLinkageTraceHtml(packet = {}) {
    const trace = packet?.metadata?.evidence_linkage_trace;
    if (!trace || trace.policy !== "hypothesis_scoped_evidence_linkage_trace_v1") return '<div class="evidence-linkage-trace">이 실행의 상세 가설–근거 연결 진단은 미기록입니다. 기존 screening·제외 기록을 확인하십시오.</div>';
    const entries = Array.isArray(trace.entries) ? trace.entries.filter(row => row && typeof row === "object").slice(0, 240) : [];
    const count = value => Number.isInteger(value) && value >= 0 ? String(value) : "미기록";
    const list = value => Array.isArray(value) ? value.map(item => esc(String(item))).join(" · ") : "";
    const reasons = {
      DIRECTION_NOT_REQUESTED: "요청 방향에 해당하지 않음",
      DIRECTION_UNRESOLVED: "직접 개입 방향 미확정",
      NO_PRIMARY_DIRECTION_EVIDENCE: "적격 1차 개입 근거 없음",
      NO_DIRECT_TARGET_INTERVENTION: "직접 표적 개입 연결 없음",
      NO_MEASURED_ENDPOINT: "개입 후 측정 결과 연결 없음",
    };
    return `<details class="evidence-linkage-trace" data-disclosure="evidence-linkage-trace"><summary>가설–근거 연결 진단 · ${count(trace.entry_count_recorded)}/${count(trace.entry_count_total)}개 기록</summary>
      <p>같은 문헌도 가설과 분자 범위에 따라 역할이 달라집니다. 아래는 이번 심사에서 기록한 연결 진단이며, 전체 검색 문헌 수 또는 독립 논문 수와 다릅니다. 상세 미포함 ${count(trace.entry_count_omitted)}개.</p>
      <div class="research-table-wrap"><table><thead><tr><th>가설·범위</th><th>방향</th><th>문헌 출처</th><th>의미 분류</th><th>연결된 주장 역할</th><th>방향 연결 미충족 사유</th><th>가설 판정</th></tr></thead><tbody>${entries.map(row => `<tr>
        <td><b>${esc(row.target_symbol || "미기록")}</b><br>${esc(row.molecular_scope || "일반 표적")}<br><small>${esc(row.hypothesis_id || "")}</small></td>
        <td>${esc(directionLabel(row.requested_direction))} → ${esc(directionLabel(row.resolved_direction))}</td>
        <td>${esc(row.source || "미기록")} · ${esc(row.external_id || row.evidence_id || "미기록")}<details><summary>근거 ID·해시</summary><small>${esc(row.evidence_id || "미기록")}<br>${esc(row.excerpt_hash || "미기록")}</small></details></td>
        <td>${esc(row.semantic_role || "미기록")}<br><small>${esc(row.semantic_status || "미기록")}</small></td>
        <td>${list(row.linked_roles) || "연결 없음"}</td>
        <td>${Array.isArray(row.direction_rejection_codes) && row.direction_rejection_codes.length ? row.direction_rejection_codes.map(code => esc(Object.prototype.hasOwnProperty.call(reasons, code) ? reasons[code] : code)).join(" · ") : "기록된 사유 없음"}</td>
        <td>${row.hypothesis_disposition === "retained" ? "보존" : row.hypothesis_disposition === "provisional" ? "MA1 잠정 제출 · 진단 기록" : row.hypothesis_disposition === "excluded" ? "제외" : "미기록"}<br><small>${list(row.hypothesis_reason_codes)}</small></td>
      </tr>`).join("")}</tbody></table></div></details>`;
  }

  function targetSourceTag(target = {}) {
    const source = target.metadata?.target_discovery_source;
    if (!source) return "";
    const label = source === "external_dynamic" ? "동적 표적" : source === "fixture_fallback" ? "FIXTURE 표적 · 검증용" : source;
    const title = source === "fixture_fallback" ? "실제 질환-표적 근거가 아닌 데모·테스트용 fallback 표적입니다." : "";
    return `<span class="tag target-source ${esc(source)}"${title ? ` title="${esc(title)}"` : ""}>${esc(label)}</span>`;
  }

  function isFixtureEvidence(item = {}) {
    const metadata = item?.metadata || {};
    const provenance = String(metadata.provenance || metadata.provenance_class || "").toLowerCase();
    const source = String(item?.source || "").toLowerCase();
    return provenance === "fixture_fallback" || metadata.mock === true || source.includes("fixture");
  }

  function hypothesisProvenance(hypothesis = {}) {
    const evidence = Array.isArray(hypothesis.evidence) ? hypothesis.evidence : [];
    const fixtureEvidence = evidence.filter(isFixtureEvidence).length;
    const supportEvidence = evidence.filter((item) => String(item?.supports || "support").toLowerCase() === "support");
    const fixtureOnlySupport = supportEvidence.length > 0 && supportEvidence.every(isFixtureEvidence);
    const fixtureTarget = hypothesis.target?.metadata?.target_discovery_source === "fixture_fallback"
      || hypothesis.target?.identity_status === "fixture";
    const fixtureOnly = fixtureOnlySupport
      || (fixtureTarget && (supportEvidence.length === 0 || supportEvidence.every(isFixtureEvidence)));
    return { fixtureEvidence, fixtureTarget, fixtureOnlySupport, fixtureOnly, hasFixture: fixtureTarget || fixtureEvidence > 0 };
  }

  function hypothesisProvenanceBadge(hypothesis = {}) {
    const provenance = hypothesisProvenance(hypothesis);
    if (provenance.fixtureOnly) {
      return `<div class="hypothesis-provenance fixture-only" role="note" title="실제 질환-표적 근거가 아니라 UI·워크플로 검증용 자료만 연결되어 있습니다.">
        <b>FIXTURE ONLY</b><span>검증용 자료 · 질환 근거로 사용 불가</span>
      </div>`;
    }
    if (provenance.hasFixture) {
      return `<div class="hypothesis-provenance fixture-mixed" role="note"><b>FIXTURE 포함</b><span>실제 외부 근거와 분리하여 해석</span></div>`;
    }
    return "";
  }

  function renderTargetIdentity(target = {}) {
    const identities = [
      target.ensembl_id ? `Ensembl ${target.ensembl_id}` : "",
      target.uniprot_id ? `UniProt ${target.uniprot_id}` : "",
      target.identity_status ? target.identity_status : "",
    ].filter(Boolean);
    return identities.length
      ? `<div class="identity-row">${identities.map((item) => `<span>${esc(item)}</span>`).join("")}</div>`
      : "";
  }

  function hypothesisWhyHtml(hypothesis) {
    const claims = Array.isArray(hypothesis.claims) ? hypothesis.claims.slice(0, 5) : [];
    const evidence = Array.isArray(hypothesis.evidence) ? hypothesis.evidence.slice(0, 8) : [];
    const targetMeta = hypothesis.target?.metadata || {};
    const uncalibratedAdmission = targetMeta.ma3_admission?.policy === "ma3-source-bound-v1";
    const discovery = [
      targetMeta.discovery_rank ? `질환-표적 순위 #${targetMeta.discovery_rank}` : "",
      targetMeta.association_score != null ? `Open Targets association ${formatScore(targetMeta.association_score)} (순위 휴리스틱)` : "",
      targetMeta.assignment_basis ? `트랙 배정: ${targetMeta.assignment_basis}` : "",
      targetMeta.discovery_track_hint_basis || "",
      targetMeta.fallback_reason ? `Fallback: ${targetMeta.fallback_reason}` : "",
    ].filter(Boolean);
    const disclosureKey = `${state.snapshot?.run_id || "run"}:hypothesis:${hypothesis.hypothesis_id || hypothesis.target?.symbol || "unknown"}`;
    return `<details class="why-card" data-disclosure="${esc(disclosureKey)}">
      <summary>가설 선정 근거 및 결정 요인</summary>
      <div class="why-content">
        ${discovery.length ? `<b>표적 발굴·트랙 배정</b><ul>${discovery.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}
        ${claims.length ? `<b>검증 가능한 주장</b><ul>${claims.map((claim) => `<li>${esc(claim.text)} <em>${uncalibratedAdmission ? "신뢰도 미보정 · 확률 아님" : `신뢰도 ${formatScore(claim.confidence)}`}</em></li>`).join("")}</ul>` : ""}
        ${evidence.length ? `<b>연결 근거</b><ul>${evidence.map((item) => `<li><span class="evidence-role ${esc(item.supports || "context")}">${esc(evidenceRoleLabel(item.supports))}</span>${evidenceLink(item)}</li>`).join("")}</ul>` : ""}
      </div>
    </details>`;
  }

  function evidenceLink(item) {
    const contentOrigin = evidenceContentOriginLabel(item);
    const sourceIdentity = evidenceSourceIdentityLabel(item, contentOrigin);
    const provenance = item?.metadata?.provenance_class || (item?.metadata?.cache_hit ? "external_cache" : item?.metadata?.live ? "external_live" : "recorded");
    const recordTitle = String(item?.title || item?.external_id || "record").slice(0, 240);
    const label = `${sourceIdentity} · ${recordTitle} · ${provenanceLabel(provenance)}`;
    const safeUrl = /^https?:\/\//i.test(item.url || "") ? esc(item.url) : "";
    return safeUrl
      ? `<a href="${safeUrl}" target="_blank" rel="noreferrer">${esc(label)}</a>`
      : esc(label);
  }

  function renderEvidenceSources(evidence) {
    if (!evidence.length) return '<div class="evidence-sources"><span>Sources</span><em class="source-chip unknown">No evidence</em></div>';
    const sources = new Map();
    evidence.forEach((item) => {
      const metadata = item?.metadata || {};
      const provenance = String(metadata.provenance || metadata.provenance_class || "").toLowerCase();
      const source = String(item?.source || "Unknown source");
      const isCached = provenance === "external_cache" || metadata.cache_hit === true;
      const isFixture = isFixtureEvidence(item);
      const isLive = !isFixture && !isCached
        && (metadata.live === true || provenance === "external_live");
      const label = isFixture
        ? `Fixture fallback · NOT CACHE · ${item?.supports === "counter" ? "COUNTER" : "SUPPORT"}`
        : `${source} · ${isCached ? "CACHE" : isLive ? "LIVE" : "RECORDED"}`;
      const kind = isLive ? "live" : isCached ? "cached" : isFixture ? "fixture" : "unknown";
      sources.set(`${kind}:${label}`, { label, kind });
    });
    return `<div class="evidence-sources"><span>Sources</span>${[...sources.values()].map((source) =>
      `<em class="source-chip ${source.kind}">${esc(source.label)}</em>`
    ).join("")}</div>`;
  }

  function renderPlanArtifact() {
    const plan = state.snapshot?.plan_spec;
    if (!plan) return '<div class="artifact-empty">G0 통과 후 Planner가 실행 명세를 생성합니다.</div>';
    const tasks = plan.tasks || [];
    return `
      <div class="overview-grid" style="grid-template-columns:minmax(0,1.35fr) minmax(250px,.65fr)">
        <div class="plan-list">${tasks.map((task,index) => `
          <article class="plan-card"><span class="plan-index">${String(index+1).padStart(2,"0")}</span><div><h4>${esc(task.name || task.task_id)}</h4>
          <p>${esc((task.acceptance_criteria || []).join(" · ") || "수락 기준 미기록")}</p><div class="plan-tools">${(task.tools || []).map((tool) => `<span class="tag">${esc(tool.tool_name)}</span>`).join("")}</div></div>
          ${task.route ? `<span class="plan-route">${esc(routeLabel(task.route))}</span>` : ""}</article>`).join("")}</div>
        <article class="artifact-card soft"><h4>계획 제어값</h4><dl class="detail-list">
          <div class="detail-row"><dt>계획 개정</dt><dd>Revision ${plan.revision || 0}</dd></div>
          <div class="detail-row"><dt>총 토큰 예산</dt><dd>${compactNumber(plan.total_token_budget)}</dd></div>
          <div class="detail-row"><dt>경로별 후보 상한</dt><dd>표적별 선택 경로당 ${plan.candidates_per_route}건</dd></div>
          <div class="detail-row"><dt>재창출 제외규칙</dt><dd>약물명 ${(plan.forbidden_drugs || []).length}건 · ID ${(plan.forbidden_drug_ids || []).length}건 · 계열 ${(plan.forbidden_drug_classes || []).length}건</dd></div>
          <div class="detail-row"><dt>승인 도구</dt><dd>${(plan.approved_tools || []).length}종</dd></div>
        </dl><h4 style="margin-top:13px">중단 조건</h4><div class="tag-row">${(plan.stop_conditions || []).map((value) => `<span class="tag">${esc(value)}</span>`).join("") || "—"}</div></article>
      </div>`;
  }

  function renderCandidatesArtifact() {
    const packet = state.snapshot?.candidate_packet;
    const candidates = packet?.candidates || [];
    const portfolio = portfolioExecutionState(packet || {});
    const intake = getIntakeAssessment(state.snapshot || {});
    const paperV6FailureDetail = paperNativeV6FailureDetailHtml(packet || {});
    if (intake.diseaseNeedsReview && candidates.length) {
      return `<div class="artifact-blocked-state">
        <span aria-hidden="true">!</span><b>후보 결과 표시 중단</b>
        <p>질환이 확인되지 않은 이전 fallback 실행에서 생성된 후보 ${compactNumber(candidates.length)}건은 과학적 결과로 제시하지 않습니다.</p>
        <small>원본 JSON에는 감사 추적을 위해 기존 기록이 보존됩니다.</small>
      </div>`;
    }
    if (!candidates.length) {
      if (packet && portfolio.abstainRequired) {
        const reason = portfolio.reasonCodes.length
          ? portfolio.reasonCodes.map(executionReasonLabel).join(" · ")
          : "표적별 실행 미완료";
        const auditHeadline = paperNativeV6FailureHeadline(packet || {});
        return `<div class="artifact-blocked-state"><span aria-hidden="true">!</span><b>MA2 ${esc(portfolio.executionStatus.toUpperCase())} · 후보 승인 불가</b><p>${esc(portfolioCoverageText(packet, { includeApproved: true }))}</p><small>${esc(auditHeadline || reason)} · 후보별 사유는 아래 감사 상세, 전체 원본은 기술 데이터에 보존</small></div>${paperV6FailureDetail}`;
      }
      return '<div class="artifact-empty">MA2 후보 평가가 아직 생성되지 않았습니다.</div>';
    }
    const sorted = sortCandidatesForReview(candidates, packet);
    const g1 = state.snapshot?.g1_verdict;
    return `<details class="artifact-methodology"><summary>MA2 route, tool and ranking methodology <span>${compactNumber(packet?.tool_calls?.length || 0)} traced calls</span></summary>${ma2ProcessOverview(packet, candidates)}</details>
      ${paperV6FailureDetail}
      <div class="artifact-table-intro"><p>${portfolio.enabled ? `표적별 후보를 활성·안전성·근거로 평가한 뒤, 표적 내 정규화 순위와 질환-표적 근거를 사용한 portfolio score로 비교합니다. ${esc(portfolioCoverageText(packet, { includeApproved: true }))}.` : "후보를 활성·안전성·근거·통합 점수 기준으로 비교합니다."}${portfolio.abstainRequired ? " 일부 표적 미완료 상태이므로 기록 후보는 승인 가능한 최종 포트폴리오가 아닙니다." : ""}</p><span>G1 ${esc(g1?.verdict || "PENDING")}</span></div>
      <div class="research-table-wrap"><table class="research-table candidate-table">
        <thead><tr><th>${portfolio.enabled ? "Portfolio rank" : "Rank"}</th><th>Candidate</th><th>Target</th><th>Activity</th><th>Safety</th><th>Evidence</th><th>${portfolio.enabled ? "Portfolio score" : "Candidate score"}</th><th>Audit</th></tr></thead>
        <tbody>${sorted.map((candidate, index) => {
          const id = candidate.candidate_id || candidate.name || `candidate-${index}`;
          const activity = candidate.activities?.[0];
          const activityText = activity ? `${activity.endpoint} ${activity.relation || ""} ${activity.value} ${activity.unit}` : candidate.metrics?.docking_score_kcal_mol != null ? `Docking ${candidate.metrics.docking_score_kcal_mol}` : "Not measured";
          const warnings = candidate.safety_flags?.length || 0;
          const rank = candidateRankingValue(candidate, packet);
          const score = candidateRankingScore(candidate, packet);
          return `<tr class="research-row ${candidate.status === "excluded" ? "is-excluded" : ""}" data-context-kind="candidate" data-context-id="${esc(id)}">
            <td class="rank-cell">${rank ? esc(rank) : "—"}</td>
            <td>${candidateTableIdentityHtml(candidate)}</td>
            <td><strong>${esc(candidate.target?.symbol || "—")}</strong><small>${candidate.target_rank || candidate.rank ? `Target rank ${esc(candidate.target_rank ?? candidate.rank)}` : ""}</small></td>
            <td class="numeric-cell">${esc(activityText)}</td>
            <td><span class="safety-indicator ${warnings ? "warning" : "clear"}">${warnings ? `${warnings} flags` : "No flagged risk"}</span></td>
            <td><button class="table-link" type="button" aria-label="${esc(`${candidate.name || "후보"} 근거와 상세 보기`)}" data-context-kind="candidate" data-context-id="${esc(id)}">${compactNumber(candidate.evidence?.length || 0)} records</button></td>
            <td class="score-cell">${score == null ? "—" : formatScore(score)}${portfolio.enabled ? `<small>Target ${formatScore(candidate.score)}</small>` : ""}</td>
            <td><span class="audit-indicator ${String(g1?.verdict || "pending").toLowerCase()}">${esc(g1?.verdict || "PENDING")}</span></td>
          </tr>`;
        }).join("")}</tbody>
      </table></div>`;
  }

  function ma2ProcessOverview(packet, candidates) {
    const portfolio = portfolioExecutionState(packet || {});
    const routeSummary = ["repurposing", "de_novo"].map((route) => {
      const routeCandidates = candidates.filter((candidate) => candidate.route === route);
      const routeTargets = portfolio.enabled
        ? portfolio.executions.filter((item) => item.route === route)
        : [];
      const eligible = routeCandidates.filter((candidate) => candidate.status !== "excluded").length;
      const executed = portfolio.enabled ? routeTargets.length > 0 : routeCandidates.length > 0;
      return `<span class="${executed ? "executed" : "not-executed"}"><b>${esc(routeLabel(route))}</b><small>${executed ? `${portfolio.enabled ? `표적 ${routeTargets.length}개 · ` : ""}후보 ${routeCandidates.length}건 · 감사 가능 ${eligible}건` : "해당 경로로 판정된 표적 없음"}</small></span>`;
    }).join("");
    const routeDecision = normalizeRouteDecision(packet, candidates);
    const calls = packet?.tool_calls || [];
    const routeCalls = calls.filter((call) => ["repurposing_search", "de_novo_generator", "route_selector"].includes(call.tool_name));
    const restrictionCalls = calls.filter((call) => /screen_forbidden_drug|rescreen_forbidden_drug/i.test(String(call.operation || "")));
    const refinementCount = calls.filter((call) => String(call.operation || "").includes("refine")).length;
    const enrichmentCount = (packet?.metadata?.public_enrichment_queries || []).length;
    const excludedCount = candidates.filter((candidate) => candidate.status === "excluded").length;
    const g1 = state.snapshot?.g1_verdict;
    const coverageStatus = portfolio.enabled
      ? portfolio.abstainRequired ? "warning" : portfolio.complete ? "done" : "waiting"
      : "done";
    const steps = [
      [portfolio.enabled ? "G0 승인 표적 범위" : "실행 표적 범위", portfolio.enabled && !portfolio.executions.length ? "unrecorded" : coverageStatus, portfolio.enabled ? portfolioCoverageText(packet, { includeApproved: true }) : "선택 표적 1개"],
      ["표적별 경로 판정", routeDecision.recorded ? "done" : "unrecorded", routeDecision.summary],
      ["검색·생성", routeCalls.length ? "done" : "unrecorded", `${routeCalls.length}건의 도구 호출 기록`],
      ["금지약물 사전 차단", restrictionCalls.length ? "done" : "unrecorded", restrictionCalls.length ? `고비용 계산 전 ${restrictionCalls.length}건 스크리닝` : "스크리닝 trace 미기록"],
      ["초기 계산", calls.some((call) => call.tool_name === "chemistry_validator") ? "done" : "unrecorded", `화학·활성·도킹·6종 ADMET · 제외 ${excludedCount}건`],
      ["필요 시 정제·재계산", refinementCount ? "done" : "neutral", refinementCount ? `${refinementCount}회 반복 정제 후 재계산` : "추가 정제 없음"],
      [portfolio.enabled ? "표적 내·포트폴리오 순위화" : "다기준 순위화", calls.some((call) => call.tool_name === "candidate_judge") || (portfolio.enabled && candidates.some((candidate) => candidate.portfolio_rank != null)) ? "done" : "unrecorded", portfolio.enabled ? `${candidates.filter((candidate) => candidate.portfolio_rank != null).length}/${candidates.length}건 portfolio rank 기록` : `${candidates.length}건 순위 기록`],
      ["순위화 후 근거 보강", enrichmentCount ? "done" : "neutral", enrichmentCount ? `공개 DB ${enrichmentCount}건 조회` : "보강 조회 없음"],
      ["MA3 · G1 인계", g1 ? (g1.verdict === "PASS" ? "done" : "warning") : "waiting", g1 ? `${g1.verdict} · findings ${g1.findings?.length || 0}건` : "감사 대기"],
    ];
    return `<aside class="route-portfolio-note">
      <div><em>${portfolio.enabled ? "MULTI-TARGET EXECUTION PORTFOLIO" : "TARGET-ADAPTIVE ROUTE POLICY"}</em><b>표적 근거성에 따른 MA2 탐색 경로</b><p>${esc(routeDecision.summary)}</p>${routeDecision.scopeSummary ? `<p class="route-confidence-note">${esc(routeDecision.scopeSummary)}</p>` : ""}${routeDecisionDetailHtml(routeDecision)}</div>
      <div class="route-portfolio-stats">${routeSummary}</div>
      ${portfolio.enabled ? `<div class="route-portfolio-stats">${portfolio.executions.map((item) => targetExecutionSummaryHtml(item)).join("") || '<span class="not-executed"><b>표적 실행 기록 없음</b><small>CandidatePacket target_executions 미기록</small></span>'}</div>` : ""}
      <div class="ma2-process-strip">${steps.map(([label,status,detail], index) => `<div class="ma2-process-step ${esc(status)}"><i>${index + 1}</i><span><b>${esc(label)}</b><small>${esc(detail)}</small></span></div>`).join("")}</div>
      ${ma2DiscoveryFunnelHtml(packet, candidates)}
    </aside>`;
  }

  function ma2DiscoveryFunnelHtml(packet, candidates) {
    const candidateBounds = candidates
      .map((candidate) => candidate?.metadata?.candidate_discovery_bounds)
      .find((value) => value && typeof value === "object");
    const backfillBounds = packet?.metadata?.candidate_backfill
      ?.discovery_provenance?.candidate_discovery_bounds;
    const bounds = candidateBounds || backfillBounds;
    if (!bounds || typeof bounds !== "object") {
      return `<details class="ma2-discovery-funnel"><summary>후보 검색 퍼널·탈락 사유 <span>기록 없음</span></summary><p>이 실행에는 후보 고정 풀과 탈락 provenance가 구조화되어 있지 않습니다. 누락을 통과로 해석하지 않습니다.</p></details>`;
    }

    const primarySearch = bounds.primary_abstract_search && typeof bounds.primary_abstract_search === "object"
      ? bounds.primary_abstract_search
      : {};
    const dispositions = Array.isArray(primarySearch.candidate_primary_dispositions)
      ? primarySearch.candidate_primary_dispositions
      : [];
    const preactivityReasons = bounds.preactivity_exclusion_reason_codes_by_id
      && typeof bounds.preactivity_exclusion_reason_codes_by_id === "object"
      ? bounds.preactivity_exclusion_reason_codes_by_id
      : {};
    const fixedPool = Array.isArray(primarySearch.fixed_pool_candidate_ids)
      ? primarySearch.fixed_pool_candidate_ids
      : [];
    const preactivityRows = Object.entries(preactivityReasons).map(([candidateId, reasons]) => ({
      candidateId,
      reasons: Array.isArray(reasons) ? reasons : [],
    }));
    const dispositionRows = dispositions.map((row) => ({
      candidateId: row?.molecule_chembl_id || "ID 미기록",
      status: row?.verification_status || "unverified",
      reasons: row?.reason_code ? [row.reason_code] : [],
    }));
    const verifiedCount = nonnegativeInteger(
      primarySearch.verified_candidate_count,
      dispositionRows.filter((row) => row.status === "verified").length,
    );
    const rejectedCount = nonnegativeInteger(
      primarySearch.rejected_candidate_count,
      Math.max(0, dispositionRows.length - verifiedCount),
    );
    const funnel = [
      ["발견 identity", nonnegativeInteger(bounds.discovered_identity_count, 0)],
      ["1차 적격", nonnegativeInteger(bounds.eligible_identity_count, 0)],
      ["고정 검토 풀", fixedPool.length || nonnegativeInteger(bounds.preactivity_pool_count, 0)],
      ["직접 activity 보유", nonnegativeInteger(bounds.fixed_pool_with_activity_count, 0)],
      ["원저 검증 통과", verifiedCount],
      ["원저·activity 탈락", rejectedCount],
      ["최종 평가", candidates.length],
    ];
    const exclusionRowHtml = (row, verified = false) => {
      const reasons = row.reasons.length
        ? row.reasons.map((reason) => executionReasonLabel(reason)).join(" · ")
        : verified ? "검증 통과" : "사유 코드 미기록";
      return `<li class="${verified ? "verified" : "rejected"}"><b>${esc(row.candidateId)}</b><span>${esc(reasons)}</span></li>`;
    };
    const dispositionHtml = dispositionRows.length
      ? dispositionRows.map((row) => exclusionRowHtml(row, row.status === "verified")).join("")
      : '<li class="unrecorded"><b>검증 단계 내역 없음</b><span>누락을 통과로 해석하지 않음</span></li>';
    const preactivityHtml = preactivityRows.length
      ? preactivityRows.map((row) => exclusionRowHtml(row)).join("")
      : '<li class="unrecorded"><b>사전 제외 내역 없음</b><span>구조화 기록 없음</span></li>';

    return `<details class="ma2-discovery-funnel" data-disclosure="${esc(state.snapshot?.run_id || "run")}:ma2-discovery-funnel">
      <summary>후보 검색 퍼널·탈락 사유 <span>검토 ${compactNumber(fixedPool.length || dispositionRows.length)} · 통과 ${compactNumber(verifiedCount)} · 탈락 ${compactNumber(rejectedCount)}</span></summary>
      <p>조회·식별·고정 풀·원저 검증·최종 평가를 분리합니다. 외부 오류와 근거 부족은 과학적 음성 결과로 바꾸지 않습니다.</p>
      <div class="ma2-funnel-stats">${funnel.map(([label, value]) => `<span><b>${compactNumber(value)}</b><small>${esc(label)}</small></span>`).join("")}</div>
      <div class="ma2-exclusion-ledgers">
        <section><h5>고정 풀 검증 내역</h5><ul>${dispositionHtml}</ul></section>
        <section><h5>고정 풀 이전 사전 제외</h5><ul>${preactivityHtml}</ul></section>
      </div>
    </details>`;
  }

  function targetExecutionSummaryHtml(execution = {}) {
    const status = String(execution.status || "unknown").toLowerCase();
    const route = execution.route ? routeLabel(execution.route) : "경로 미배정";
    const reasons = (execution.reason_codes || []).map((value) => String(value).replaceAll("_", " ")).join(" · ");
    const counts = `후보 ${execution.discovered_candidate_count || 0}건 · 적합 ${execution.eligible_candidate_count || 0}건 · 제외 ${execution.excluded_candidate_count || 0}건`;
    const concluded = ["completed", "no_candidates"].includes(status);
    const statusLabel = status === "completed"
      ? "후보 평가 완료"
      : status === "no_candidates"
      ? "검색 완료 · 적합 후보 없음"
      : status.replaceAll("_", " ").toUpperCase();
    return `<span class="${concluded ? "executed" : "not-executed"}"><b>${esc(execution.target?.symbol || execution.target_key || "Unresolved target")} · ${esc(statusLabel)}</b><small>${esc(route)} · ${esc(counts)}${reasons ? ` · ${esc(reasons)}` : ""}</small></span>`;
  }

  function normalizeRouteDecision(packet, candidates) {
    const metadata = packet?.metadata || {};
    const raw = metadata.route_decision || metadata.route_policy || metadata.route_selection || null;
    const executed = [...new Set(candidates.map((candidate) => routeLabel(candidate.route)))];
    const portfolio = portfolioExecutionState(packet || {});
    if (portfolio.enabled) {
      const singleCandidate = candidates.length === 1 ? candidates[0] : null;
      const focusedExecution = singleCandidate ? targetExecutionForCandidate(packet, singleCandidate) : null;
      const scope = focusedExecution ? [focusedExecution] : portfolio.executions;
      const routeTargets = scope.filter((item) => item.route);
      const routeCounts = ["repurposing", "de_novo"].map((route) => {
        const count = routeTargets.filter((item) => item.route === route).length;
        return count ? `${routeLabel(route)} ${count}개 표적` : "";
      }).filter(Boolean);
      const statusText = focusedExecution
        ? `표적 ${focusedExecution.target?.symbol || focusedExecution.target_key} · ${String(focusedExecution.status || "unknown").toUpperCase()} · ${focusedExecution.route ? routeLabel(focusedExecution.route) : "경로 미배정"}`
        : `${portfolioCoverageText(packet, { includeApproved: true })}${routeCounts.length ? ` · ${routeCounts.join(" · ")}` : ""}`;
      const incomplete = scope.filter((item) => !["completed", "no_candidates", "duplicate"].includes(String(item.status || "").toLowerCase()));
      const reasonCodes = [...new Set([
        ...portfolio.reasonCodes,
        ...scope.flatMap((item) => item.reason_codes || []),
      ])];
      return {
        recorded: portfolio.executions.length > 0,
        portfolio: true,
        summary: statusText,
        scopeSummary: focusedExecution
          ? `전체 MA2 포트폴리오 중 ${focusedExecution.target?.symbol || focusedExecution.target_key} 표적의 후보 실행 기록입니다.`
          : `모든 G0 승인 표적을 target_executions에 보존하며, 식별·실행 불가 표적도 누락하지 않습니다.${incomplete.length ? ` 미완료·비실행 표적 ${incomplete.length}개로 전체 포트폴리오는 ${portfolio.executionStatus.toUpperCase()} 상태입니다.` : ""}`,
        selectionBasis: "표적마다 저장된 route_decision과 실제 실행 경로를 대조합니다.",
        requestedPriority: state.snapshot?.brief?.constraints?.discovery_priority || "auto",
        fallbackToAuto: false,
        skippedHypothesisCount: Math.max(0, portfolio.approved - portfolio.attempted),
        rationale: reasonCodes.map((value) => String(value).replaceAll("_", " ")),
        factors: [],
        policyVersion: metadata.portfolio_ranking?.policy_version || metadata.portfolio_ranking?.version || null,
        thresholds: {},
        fallback: null,
        confidenceInterpretation: null,
        dgidbInteractionCount: null,
        clinicalTrialDrugCount: null,
        knownDrugExamples: [],
        legacyKnownCount: null,
        targetExecutions: scope,
        executionStatus: portfolio.executionStatus,
        abstainReasonCodes: reasonCodes,
      };
    }
    if (!raw) {
      return {
        recorded: false,
        summary: `경로 판정 메타데이터 미기록 · 실제 수행 ${executed.join(", ") || "없음"}`,
        rationale: [],
        factors: [],
      };
    }
    const selectedRaw = raw.selected_routes || raw.selected_route || raw.routes || raw.decision || raw.route;
    const selected = (Array.isArray(selectedRaw) ? selectedRaw : [selectedRaw])
      .filter(Boolean)
      .map((route) => routeLabel(String(route).toLowerCase()));
    const rationaleRaw = raw.rationale || raw.reasons || raw.decision_factors || raw.basis || [];
    const rationale = (Array.isArray(rationaleRaw) ? rationaleRaw : [rationaleRaw])
      .filter(Boolean)
      .map((item) => typeof item === "string" ? item : item.message || item.label || JSON.stringify(item));
    const novelty = raw.target_novelty ?? raw.novelty_status ?? raw.knowledge_status;
    const knownScore = raw.known_drug_score;
    const dgidbInteractionCount = raw.dgidb_interaction_count;
    const clinicalTrialDrugCount = raw.clinical_trial_drug_count;
    const legacyKnownCount = raw.known_drug_count ?? raw.eligible_known_drug_count ?? raw.repurposing_candidate_count;
    const hypotheses = state.snapshot?.hypothesis_packet?.hypotheses || [];
    const primarySelection = metadata.primary_hypothesis_selection || raw.primary_hypothesis_selection || {};
    const consideredHypotheses = Array.isArray(primarySelection.considered_hypotheses)
      ? primarySelection.considered_hypotheses
      : [];
    const portfolioSize = primarySelection.portfolio_size
      ?? primarySelection.hypothesis_count
      ?? primarySelection.total_hypotheses
      ?? (consideredHypotheses.length || hypotheses.length);
    const appliedTargetCount = primarySelection.applied_target_count
      ?? primarySelection.selected_count
      ?? primarySelection.selected_target_count
      ?? 1;
    const skippedHypothesisCount = primarySelection.skipped_hypothesis_count
      ?? Math.max(0, Number(portfolioSize || 0) - Number(appliedTargetCount || 1));
    const requestedPriority = primarySelection.requested_priority
      || state.snapshot?.brief?.constraints?.discovery_priority
      || "auto";
    const selectionReason = primarySelection.selection_reason
      || primarySelection.selection_basis
      || primarySelection.rationale
      || primarySelection.reason
      || null;
    const selectionBasis = selectionReason == null
      ? "저장된 세부 선택 사유 없음"
      : hypothesisSelectionReasonLabel(routeFactorValue(selectionReason, ""));
    const fallbackToAuto = Boolean(primarySelection.fallback_to_auto);
    const selectedTarget = primarySelection.selected_target_symbol || raw.target_symbol || "미기록";
    const scopeSummary = portfolioSize
      ? `기존 실행 방식에서는 MA1 가설 ${portfolioSize}개 중 ${discoveryPriorityLabel(requestedPriority)} 기준으로 선택된 표적 ${selectedTarget}에 경로 정책을 적용했습니다${skippedHypothesisCount ? ` · 미실행 가설 ${skippedHypothesisCount}개` : ""}.`
      : `기존 실행 방식에서는 선택 표적 ${selectedTarget}에 경로 정책을 적용했습니다.`;
    const facts = [
      raw.target_symbol ? `표적 ${raw.target_symbol}` : "",
      selected.length ? `선택 경로 ${selected.join(", ")}` : `실제 수행 ${executed.join(", ") || "없음"}`,
      dgidbInteractionCount != null ? `DGIdb 상호작용 레코드 ${dgidbInteractionCount}건` : "",
      clinicalTrialDrugCount != null ? `임상시험 약물명 ${clinicalTrialDrugCount}종(중복 제거)` : "",
      knownScore != null ? `known-drug score ${Number(knownScore).toFixed(2)}` : "",
      novelty ? `표적 상태 ${routeStatusLabel(novelty)}` : "",
      raw.confidence != null ? `경로 규칙 충족도 ${Number(raw.confidence).toFixed(2)}(성공확률 아님)` : "",
    ].filter(Boolean);
    return {
      recorded: true,
      summary: facts.join(" · "),
      scopeSummary,
      selectionBasis,
      requestedPriority,
      fallbackToAuto,
      skippedHypothesisCount,
      rationale,
      factors: Array.isArray(raw.factors) ? raw.factors : [],
      policyVersion: raw.policy_version || null,
      thresholds: raw.thresholds || {},
      fallback: raw.fallback || null,
      confidenceInterpretation: raw.confidence_interpretation || null,
      dgidbInteractionCount,
      clinicalTrialDrugCount,
      knownDrugExamples: Array.isArray(raw.known_drug_examples) ? raw.known_drug_examples : [],
      legacyKnownCount,
    };
  }

  function routeDecisionDetailHtml(decision) {
    if (!decision.recorded) return "";
    if (decision.portfolio) {
      const executionRows = (decision.targetExecutions || []).map((item) => {
        const routeDecision = item.route_decision || {};
        const selectedRoute = routeDecision.selected_route || routeDecision.route || item.route;
        const knowledge = routeDecision.target_novelty || routeDecision.novelty_status || routeDecision.knowledge_status;
        const reasons = (item.reason_codes || []).map((value) => String(value).replaceAll("_", " "));
        const unreadyCapabilities = (item.readiness?.capabilities || []).filter((capability) => capability.status !== "ready");
        const readiness = unreadyCapabilities.length
          ? ` · 도구 준비 안 됨: ${unreadyCapabilities.map((capability) => `${capability.tool_name}(${capability.reason_code || capability.status})`).join(", ")}`
          : "";
        const detail = [
          `G0 ${item.g0_approved ? "승인" : "미승인"}`,
          item.identity_resolved ? "식별 확정" : "식별 미확정",
          item.execution_eligible ? "실행 자격" : "실행 제외",
          item.attempted ? "시도됨" : "미시도",
          selectedRoute ? routeLabel(selectedRoute) : "경로 미배정",
          knowledge ? routeStatusLabel(knowledge) : "",
          `후보 ${item.discovered_candidate_count || 0}건`,
          `적합 ${item.eligible_candidate_count || 0}건`,
          `제외 ${item.excluded_candidate_count || 0}건`,
        ].filter(Boolean).join(" · ");
        return `<div><dt>${esc(item.target?.symbol || item.target_key || "Unresolved target")} <em>${esc(String(item.status || "unknown").toUpperCase())}</em></dt><dd><b>${esc(detail)}</b><small>${esc(item.detail || reasons.join(" · ") || "표적별 실행 사유 기록 없음")}${esc(readiness)}</small></dd></div>`;
      }).join("");
      const executionKey = (decision.targetExecutions || []).length === 1
        ? decision.targetExecutions[0].target_key
        : "all-targets";
      const disclosureKey = `${state.snapshot?.run_id || "run"}:portfolio-route-decision:${executionKey || "unknown"}`;
      return `<details class="route-decision-record" data-disclosure="${esc(disclosureKey)}"><summary>표적별 경로·실행 범위 보기</summary><div>
        <p><b>적용 범위</b> ${esc(decision.scopeSummary || "표적별 실행 범위 미기록")}</p>
        <p><b>표적 탐색 우선순위</b> ${esc(discoveryPriorityLabel(decision.requestedPriority))} · 경로를 강제하지 않고 각 표적의 구조화 근거로 경로를 판정합니다.</p>
        ${executionRows ? `<dl class="route-factor-list">${executionRows}</dl>` : '<p>표적별 실행 기록 없음</p>'}
        ${decision.abstainReasonCodes?.length ? `<p class="route-confidence-note"><b>ABSTAIN 사유</b> ${esc(decision.abstainReasonCodes.map((value) => String(value).replaceAll("_", " ")).join(" · "))}</p>` : ""}
        <p class="route-confidence-note"><b>순위 해석</b> 표적 간 비교에는 저장된 portfolio score만 사용하며, 서로 다른 표적의 raw docking score를 직접 비교하지 않습니다.</p>
        <p><b>포트폴리오 정책 버전</b> ${esc(decision.policyVersion || "미기록")}</p>
      </div></details>`;
    }
    const factorLabels = {
      known_drug_score: "Open Targets known-drug 점수",
      known_drug_count: "이전 정책 혼합 집계값(참고용)",
      structured_known_drug_evidence: "구조화 약물 연계 레코드 최댓값(참고용)",
      drug_evidence_tier: "재창출 근거 등급",
      disease_trial_target_action_overlap: "질환 임상시험-표적 작용 일치 약물",
      approved_or_clinical_tractability: "승인·임상단계 tractability",
      target_development_level: "표적 개발 수준",
      verified_target_development_level: "검증된 표적 개발 수준(Pharos/TCRD)",
      association_score: "질환-표적 연관 점수",
      association_excerpt_relevance_score: "연관 근거 발췌 관련성 점수",
      support_evidence_count: "지지 근거 수",
      ma1_track: "MA1 공급 트랙",
    };
    const factorInterpretations = {
      known_drug_score: "경로 판정용 지표이며 약효 또는 임상 성공확률이 아닙니다",
      known_drug_count: "서로 다른 출처의 집계 최댓값으로, 고유하거나 적합한 약물 수가 아닙니다",
      structured_known_drug_evidence: "DGIdb 상호작용 레코드와 임상시험 약물명 수를 별도로 해석해야 합니다",
      drug_evidence_tier: "질환·단계·상태·표적 작용을 반영한 결정론적 근거 등급입니다",
      disease_trial_target_action_overlap: "같은 약물이 질환 임상시험과 표적 작용 근거에 함께 나타나는지 확인합니다",
      approved_or_clinical_tractability: "개발 단계 정보는 질환·표적 근거와 함께 해석합니다",
      verified_target_development_level: "Pharos/TCRD 출처가 확인된 경우에만 사용합니다",
      association_score: "De novo 표적의 최소 생물학적 지지 여부를 확인하는 값입니다",
      association_excerpt_relevance_score: "논문 발췌 선정용 휴리스틱이며 유병률·효과크기 같은 생물학적 측정값이 아닙니다",
      support_evidence_count: "연결된 지지 근거가 최소 1건인지 확인합니다",
      ma1_track: "근거가 불충분한 경우에만 제한적으로 사용하는 보조 기준입니다",
    };
    const factorRows = decision.factors.map((factor) => {
      const value = routeFactorValue(factor.value);
      const source = routeFactorValue(factor.source, "출처 미기록");
      const interpretation = factorInterpretations[factor.name] || routeFactorValue(factor.interpretation, "");
      return `<div><dt>${esc(factorLabels[factor.name] || factor.name)}</dt><dd><b>${esc(value)}</b><small>${esc(interpretation)}${source ? ` · ${esc(source)}` : ""}</small></dd></div>`;
    }).join("");
    const thresholdLabels = {
      known_drug_repurposing_min: "재창출 known-drug 최소값",
      weak_known_drug_min: "약한 known-drug 신호 최소값",
      underexplored_known_drug_max: "저개발 표적 known-drug 상한",
      de_novo_association_min: "De novo 질환-표적 연관 최소값",
    };
    const thresholdText = Object.entries(decision.thresholds || {}).map(([key,value]) => `${thresholdLabels[key] || key}=${value}`).join(" · ");
    const observedCounts = [
      decision.dgidbInteractionCount != null ? `DGIdb 상호작용 레코드 ${decision.dgidbInteractionCount}건` : "",
      decision.clinicalTrialDrugCount != null ? `임상시험 약물명 ${decision.clinicalTrialDrugCount}종(중복 제거)` : "",
    ].filter(Boolean).join(" · ");
    const examples = decision.knownDrugExamples.length
      ? `확인 예시(최대 12개): ${decision.knownDrugExamples.join(", ")}`
      : "확인된 약물명 예시 없음";
    const disclosureKey = `${state.snapshot?.run_id || "run"}:route-decision`;
    return `<details class="route-decision-record" data-disclosure="${esc(disclosureKey)}"><summary>경로 판정 규칙·관찰값 보기</summary><div>
      <p><b>적용 범위</b> ${esc(decision.scopeSummary)}</p>
      <p><b>표적 탐색 우선순위</b> ${esc(discoveryPriorityLabel(decision.requestedPriority))} · 경로를 강제하지 않고 자격을 충족한 가설의 선택 순서만 조정합니다.</p>
      ${decision.selectionBasis ? `<p><b>표적 선택 결과</b> ${esc(decision.selectionBasis)}${decision.fallbackToAuto ? " · 요청 조건에 맞는 근거 자격 표적이 없어 자동 우선순위로 복귀했습니다." : ""}</p>` : ""}
      ${decision.rationale.length ? `<p><b>저장된 규칙 근거</b> ${esc(decision.rationale.join(" "))}</p>` : ""}
      ${observedCounts ? `<p><b>약물 연계 관찰값</b> ${esc(observedCounts)}<br><small>${esc(examples)}</small></p>` : ""}
      <p class="route-confidence-note"><b>수치 해석</b> DGIdb 값은 상호작용 레코드 수이고 임상시험 값은 중복 제거된 약물명 수입니다. 어느 값도 고유한 적합 후보 수, 임상 효능 또는 승인 가능성을 뜻하지 않습니다.${decision.legacyKnownCount != null && decision.dgidbInteractionCount == null ? ` 이전 정책 혼합 집계값 ${esc(decision.legacyKnownCount)}건은 참고용으로만 표시합니다.` : ""}</p>
      ${factorRows ? `<dl class="route-factor-list">${factorRows}</dl>` : '<p>판정 요인 미기록</p>'}
      ${thresholdText ? `<p><b>정책 임계값</b> ${esc(thresholdText)}</p>` : ""}
      <p><b>정책 버전</b> ${esc(decision.policyVersion || "미기록")}${decision.fallback?.used ? ` · fallback 사용: ${esc(decision.fallback.reason || "사유 미기록")}` : " · fallback 미사용"}</p>
      ${decision.confidenceInterpretation ? `<p class="route-confidence-note">경로 규칙 충족도는 입력 근거의 완결성을 나타내며 생물학적 타당성, 약효 또는 임상 성공확률이 아닙니다.</p>` : ""}
    </div></details>`;
  }

  function routeFactorValue(value, missing = "미기록") {
    if (value == null || value === "") return missing;
    if (Array.isArray(value)) return value.length ? value.map((item) => routeFactorValue(item, "")).filter(Boolean).join(", ") : "없음";
    if (typeof value === "object") {
      const entries = Object.entries(value).map(([key,item]) => `${key}=${routeFactorValue(item, "미기록")}`);
      return entries.length ? entries.join(" · ") : "없음";
    }
    if (typeof value === "boolean") return value ? "예" : "아니오";
    return String(value);
  }

  function routeStatusLabel(status) {
    return ({
      repurposing_evidence_strong: "질환·표적·작용 기준을 충족한 재창출 근거가 확인됨",
      underexplored_with_weak_drug_context: "약한 약물 연계 맥락은 있으나 강한 재창출 근거는 없음",
      underexplored_without_drug_evidence: "조회된 구조화 소스에서 강한 재창출 근거 미확인",
      ambiguous_drug_evidence: "경로 근거 불충분·판정 보류 영역",
      well_characterized_or_drug_evidenced: "표적-약물 연계 신호가 확인됨",
      underexplored_but_evidence_supported: "생물학적 지지는 있으나 조회된 구조화 소스에서 기존 약물 신호 미확인",
      indeterminate_target_knownness: "표적 근거성 판정 보류 영역",
    })[status] || status;
  }

  function cleanCandidateIdentifier(value, pattern) {
    if (typeof value !== "string" && typeof value !== "number") return "";
    const text = String(value).trim();
    return text && text.length <= 100 && pattern.test(text) ? text : "";
  }

  function candidateIdentityEntries(candidate) {
    const metadata = candidate.metadata || {};
    const evidence = Array.isArray(candidate.evidence) ? candidate.evidence : [];
    const chemblPattern = /^CHEMBL\d+$/i;
    const inchikeyPattern = /^[A-Z]{14}-[A-Z]{10}-[A-Z]$/i;
    const chemblCandidates = [metadata.chembl_id, metadata.drug_id];
    evidence.forEach((item) => {
      if (String(item?.source || "").toLowerCase() !== "chembl") return;
      chemblCandidates.push(item?.metadata?.molecule_chembl_id, item?.external_id);
    });
    const chemblId = chemblCandidates
      .map((value) => cleanCandidateIdentifier(value, chemblPattern))
      .find(Boolean) || "";
    return [
      ["InChIKey", cleanCandidateIdentifier(candidate.inchikey, inchikeyPattern)],
      ["ChEMBL", chemblId],
    ].filter(([, value]) => value);
  }

  function candidateIdentityHtml(candidate) {
    const identifiers = candidateIdentityEntries(candidate);
    return identifiers.length
      ? `<dl class="chemical-identifiers">${identifiers.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl>`
      : '<p class="chemical-identifiers-empty">외부 화합물 식별자 없음</p>';
  }

  function candidateStructureUrl(smiles) {
    return smiles
      ? `/api/chemistry/structure.svg?smiles=${encodeURIComponent(smiles)}`
      : "";
  }

  function candidateTableIdentityHtml(candidate) {
    const smiles = String(candidate.smiles || "").trim();
    const structureUrl = candidateStructureUrl(smiles);
    const structureAlt = `${candidate.name || "후보 화합물"} 2차원 분자 구조`;
    return `<div class="candidate-table-identity">
      <figure class="molecule-structure molecule-structure-compact ${structureUrl ? "" : "is-unavailable"}">
        ${structureUrl ? `<img class="molecule-structure-image" src="${esc(structureUrl)}" alt="${esc(structureAlt)}" loading="lazy" decoding="async">` : ""}
        <span class="molecule-structure-fallback" role="status">2D</span>
      </figure>
      <div class="candidate-table-copy">
        <strong>${esc(candidate.name || "Unnamed")}</strong>
        <small>${esc(routeLabel(candidate.route))} · ${esc(candidate.status || "conditional")}</small>
        <code class="candidate-table-smiles" title="${esc(smiles || "SMILES 없음")}">${esc(smiles || "SMILES 없음")}</code>
      </div>
    </div>`;
  }

  function candidateSummaryChemistryHtml(candidate) {
    const smiles = String(candidate.smiles || "").trim();
    const structureUrl = candidateStructureUrl(smiles);
    const structureAlt = `${candidate.name || "후보 화합물"} 2차원 분자 구조`;
    const identifiers = candidateIdentityEntries(candidate);
    const primaryIdentifier = identifiers.find(([label]) => label === "ChEMBL")
      || identifiers[0]
      || ["Chemical ID", "미기록"];
    return `<div class="candidate-table-identity summary-candidate-identity" aria-label="대표 화합물 구조와 식별자">
      <figure class="molecule-structure molecule-structure-compact ${structureUrl ? "" : "is-unavailable"}">
        ${structureUrl ? `<img class="molecule-structure-image" src="${esc(structureUrl)}" alt="${esc(structureAlt)}" loading="lazy" decoding="async">` : ""}
        <span class="molecule-structure-fallback" role="status">2D</span>
      </figure>
      <div class="candidate-table-copy">
        <small>${esc(primaryIdentifier[0])} · ${esc(primaryIdentifier[1])}</small>
        <code class="candidate-table-smiles" title="${esc(smiles || "SMILES 없음")}">${esc(smiles || "SMILES 없음")}</code>
      </div>
    </div>`;
  }

  function candidateChemistryHtml(candidate) {
    const smiles = String(candidate.smiles || "").trim();
    const structureUrl = candidateStructureUrl(smiles);
    const structureAlt = `${candidate.name || "후보 화합물"} 2차원 분자 구조`;
    return `<section class="candidate-chemistry" aria-label="화합물 구조와 식별자">
      <figure class="molecule-structure ${structureUrl ? "" : "is-unavailable"}">
        ${structureUrl ? `<img class="molecule-structure-image" src="${esc(structureUrl)}" alt="${esc(structureAlt)}" loading="lazy" decoding="async">` : ""}
        <span class="molecule-structure-fallback" role="status">2D 구조를 표시할 수 없습니다</span>
        <figcaption>2D molecular structure</figcaption>
      </figure>
      <div class="chemical-identity">
        <span class="chemical-identity-label">Chemical identity</span>
        <div class="smiles"><b>SMILES</b><code>${esc(smiles || "SMILES 없음")}</code></div>
        ${candidateIdentityHtml(candidate)}
      </div>
    </section>`;
  }

  function candidateHtml(candidate, index, packet) {
    const activity = candidate.activities?.[0];
    const metrics = candidate.metrics || {};
    const exclusions = candidateExclusionReasons(candidate, packet);
    return `
      <article class="candidate-card ${candidate.status === "excluded" ? "is-excluded" : ""}">
        <span class="rank-badge ${candidate.status === "excluded" ? "excluded" : ""}">${candidate.status === "excluded" ? "제외" : candidate.rank != null ? `#${esc(candidate.rank)}` : "—"}</span>
        <span class="card-label">${esc(candidate.target?.symbol || "TARGET")}</span>
        <h4>${esc(candidate.name || candidate.candidate_id)}</h4>
        <p class="card-description">${esc(candidate.mechanism || "기전 설명 없음")}</p>
        <div class="card-meta"><span class="route-badge ${esc(candidate.route)}">${esc(routeLabel(candidate.route))}</span><span class="status-badge ${esc(candidate.status)}">${esc(statusLabel(candidate.status))}</span>${candidate.safety_flags?.length ? `<span class="status-badge excluded">안전성 ${candidate.safety_flags.length}</span>` : ""}</div>
        ${candidateChemistryHtml(candidate)}
        <div class="score-row"><span>종합 점수</span><div class="score-track"><i style="width:${scorePercent(candidate.score)}%"></i></div><b>${formatScore(candidate.score)}</b></div>
        <table class="metric-table"><tbody>
          <tr><th>활성</th><td>${activity ? `${esc(activity.endpoint)} ${esc(activity.relation)} ${esc(activity.value)} ${esc(activity.unit)}` : "—"}</td><th>Docking</th><td>${metrics.docking_score_kcal_mol ?? "—"}</td></tr>
          <tr><th>MW</th><td>${metrics.molecular_weight ? Number(metrics.molecular_weight).toFixed(1) : "—"}</td><th>cLogP</th><td>${metrics.clogp ?? "—"}</td></tr>
        </tbody></table>
        ${exclusions.length ? `<div class="candidate-exclusion"><b>제외·주의 근거</b><ul>${exclusions.map((reason) => `<li>${esc(reason)}</li>`).join("")}</ul></div>` : ""}
        ${candidateProcessHtml(candidate, packet)}
        ${candidateWhyHtml(candidate, packet)}
      </article>`;
  }

  function candidateProcessHtml(candidate, packet) {
    const portfolio = portfolioExecutionState(packet || {});
    const targetExecution = targetExecutionForCandidate(packet, candidate);
    const traceIds = new Set(candidate.tool_trace_ids || []);
    const calls = (packet?.tool_calls || []).filter((call) => traceIds.has(call.call_id));
    const byTool = (name) => calls.filter((call) => call.tool_name === name);
    const normalization = byTool("target_normalizer");
    const routeSelection = calls.filter((call) => call.operation === "assign_route_from_ma1_evidence");
    const routeCalls = byTool(candidate.route === "repurposing" ? "repurposing_search" : "de_novo_generator");
    const refinement = calls.filter((call) => String(call.operation || "").includes("refine"));
    const chemistry = byTool("chemistry_validator");
    const docking = byTool("docking");
    const admet = byTool("admet_predictor");
    const ranking = calls.filter((call) => call.operation === "deterministic_multi_objective_ranking");
    const restrictions = candidateRestrictionRecord(candidate, packet, calls);
    const preScreenExcluded = restrictions.status === "excluded" && Boolean(candidate.metadata?.forbidden_drug_screen?.matched);
    const externalQueries = (packet?.metadata?.public_enrichment_queries || [])
      .filter((query) => query.candidate_id === candidate.candidate_id);
    const activity = candidate.activities?.[0];
    const g1 = state.snapshot?.g1_verdict;
    const routeDecision = normalizeRouteDecision(packet, [candidate]);
    const steps = [
      {
        label: "표적 정규화·탐색 경로 판정",
        status: callStepStatus([...normalization, ...routeSelection]),
        detail: `${candidate.target?.symbol || "TARGET"} · ${routeDecision.summary}${routeSelection.length ? ` · ${traceCallDetail(routeSelection)}` : ""}`,
      },
      {
        label: candidate.route === "repurposing" ? "기존 약물 후보 검색" : "신규 구조 생성",
        status: callStepStatus(routeCalls),
        detail: routeCalls.length ? traceCallDetail(routeCalls) : "해당 경로의 도구 호출 기록 없음",
      },
      {
        label: "입력 금지목록·기록된 규제상태 사전 스크리닝",
        status: restrictions.status,
        detail: `${restrictions.summary} · 도킹·ADMET 전 적용`,
      },
      {
        label: "화학 구조 표준화·유효성",
        status: preScreenExcluded ? "skipped" : callStepStatus(chemistry),
        detail: preScreenExcluded ? "사전 차단으로 고비용 계산 미실행" : chemistry.length ? `${traceCallDetail(chemistry)} · MW ${candidate.metrics?.molecular_weight ? Number(candidate.metrics.molecular_weight).toFixed(1) : "—"} · cLogP ${candidate.metrics?.clogp ?? "—"}` : "화학 검사 기록 없음",
      },
      {
        label: "활성 근거 확인",
        status: preScreenExcluded ? "skipped" : activity ? "done" : "warning",
        detail: preScreenExcluded ? "사전 차단 후 평가 중단" : activity ? `${activity.endpoint} ${activity.relation} ${activity.value} ${activity.unit} · ${activity.assay_type || "assay 미기록"}` : "직접 활성값 미기록 · 도킹 기반 대체점수 사용",
      },
      {
        label: "표적 결합 도킹",
        status: preScreenExcluded ? "skipped" : callStepStatus(docking),
        detail: preScreenExcluded ? "사전 차단으로 도킹 비용 미사용" : docking.length ? `${traceCallDetail(docking)} · ${candidate.metrics?.docking_score_kcal_mol ?? "—"} kcal/mol` : "도킹 기록 없음",
      },
      {
        label: "6종 ADMET·독성 패널",
        status: preScreenExcluded ? "skipped" : callStepStatus(admet),
        detail: preScreenExcluded ? "사전 차단으로 ADMET 비용 미사용" : admet.length ? traceCallDetail(admet) : "ADMET 예측 기록 없음",
      },
      {
        label: "필요 시 구조 정제·재계산",
        status: preScreenExcluded ? "skipped" : refinement.length ? callStepStatus(refinement) : "neutral",
        detail: preScreenExcluded ? "입력 금지목록 또는 이미 기록된 규제상태로 사전 차단되어 미실행" : refinement.length ? `초기 계산 후 ${refinement.length}회 구조 정제, 화학·도킹·ADMET 재계산 · ${traceCallDetail(refinement)}` : (candidate.route === "de_novo" ? "초기 계산의 정제 임계 초과 항목이 없어 반복 종료" : "재창출 경로는 신규 구조 정제 미적용"),
      },
      {
        label: portfolio.enabled ? "표적 내·포트폴리오 순위화" : "다기준 순위화",
        status: ranking.length || (portfolio.enabled && candidate.portfolio_rank != null) ? "done" : "unrecorded",
        detail: candidate.status === "excluded"
          ? `정책 제외 · 순위 미부여${ranking.length ? ` · ${traceCallDetail(ranking)}` : ""}`
          : portfolio.enabled
          ? `표적 내 #${candidate.target_rank ?? candidate.rank ?? "—"} · target score ${formatScore(candidate.score)} · portfolio #${candidate.portfolio_rank || "—"} · portfolio score ${candidate.portfolio_score == null ? "미기록" : formatScore(candidate.portfolio_score)}${ranking.length ? ` · ${traceCallDetail(ranking)}` : ""}`
          : ranking.length ? `저장 순위 #${candidate.rank || "—"} · 점수 ${formatScore(candidate.score)} · ${traceCallDetail(ranking)}` : "순위화 기록 없음",
      },
      {
        label: "순위화 후 공개 DB 근거 보강",
        status: preScreenExcluded ? "skipped" : externalQueries.length ? (externalQueries.some((query) => ["ok", "live", "cache"].includes(query.status)) ? "done" : "warning") : "neutral",
        detail: preScreenExcluded ? "정책 제외 후 공개 DB 보강 미수행" : externalQueries.length ? `순위·점수를 변경하지 않는 근거 추가 · ${externalQueries.length}건 조회 · ${externalQueries.map((query) => `${query.source}:${query.status}`).join(", ")}` : "이 후보의 보강 조회 기록 없음",
      },
      {
        label: "MA3 · G1 후보·과정 감사 인계",
        status: g1 ? (g1.verdict === "PASS" ? "done" : "warning") : "waiting",
        detail: g1
          ? `${g1.verdict} · ${portfolio.enabled ? `${portfolioCoverageText(packet, { includeApproved: true })} · ` : ""}${g1.rationale || "감사 근거 기록됨"}`
          : `후보 패킷 인계 후 감사 대기${targetExecution ? ` · 표적 실행 ${String(targetExecution.status || "unknown").toUpperCase()}` : ""}`,
      },
    ];
    const disclosureKey = `${state.snapshot?.run_id || "run"}:ma2-process:${candidate.candidate_id || candidate.name || "unknown"}`;
    return `<details class="why-card ma2-process-card" data-disclosure="${esc(disclosureKey)}"><summary>MA2 단계별 실행 기록 <em>${calls.length}건 trace</em></summary><div class="ma2-candidate-process">
      <div class="ma2-timeline">${steps.map((step,index) => processStepHtml(step,index)).join("")}</div>
      ${admetPanelHtml(candidate)}
      <p class="trace-integrity-note">표시 내용은 CandidatePacket의 tool trace·후보 지표·감사 산출물에서 구성했습니다. 도구 기록이 없는 검사는 통과로 간주하지 않습니다.</p>
    </div></details>`;
  }

  function processStepHtml(step, index) {
    return `<div class="ma2-timeline-step ${esc(step.status)}"><i>${index + 1}</i><div><b>${esc(step.label)}</b><p>${esc(step.detail)}</p></div><span>${esc(processStatusLabel(step.status))}</span></div>`;
  }

  function callStepStatus(calls) {
    if (!calls.length) return "unrecorded";
    return calls.every((call) => call.success !== false) ? "done" : "failed";
  }

  function traceCallDetail(calls) {
    const operations = [...new Set(calls.map((call) => call.operation || call.tool_name).filter(Boolean))];
    const fixtures = calls.some((call) => String(call.source_version || "").toLowerCase().includes("fixture"));
    return `${operations.join(", ")} · ${calls.length}건${fixtures ? " · FIXTURE" : ""}`;
  }

  function processStatusLabel(status) {
    return ({ done: "기록됨", failed: "실패", warning: "확인 필요", waiting: "대기", neutral: "미수행", skipped: "사전 차단", unrecorded: "미기록", excluded: "제외" })[status] || status;
  }

  function candidateRestrictionRecord(candidate, packet, calls) {
    const metadata = packet?.metadata || {};
    const byCandidate = metadata.exclusion_summary?.by_candidate || metadata.exclusion_summary?.candidates || metadata.candidate_exclusions || {};
    const stored = byCandidate?.[candidate.candidate_id] || byCandidate?.find?.((item) => item.candidate_id === candidate.candidate_id);
    const direct = candidate.metadata?.forbidden_drug_screen || candidate.restriction_screen || candidate.drug_restriction;
    const screeningCalls = calls.filter((call) => /forbidden|restriction|contraindication|banned|withdrawn/i.test(`${call.tool_name} ${call.operation}`));
    const reasons = candidateExclusionReasons(candidate, packet);
    if (direct?.screened) {
      const matched = Boolean(direct.matched);
      const directReasons = [
        ...(direct.matched_names || []).map((item) => `금지약물명 ${item}`),
        ...(direct.matched_ids || []).map((item) => `금지 ID ${item}`),
        ...(direct.matched_classes || []).map((item) => `금지 계열 ${item}`),
        ...(direct.prohibited_statuses || []).map((item) => `회수·금지 상태 ${item}`),
      ];
      return { status: matched ? "excluded" : "done", summary: `${direct.matching_policy || "정확 일치 규칙"} · ${directReasons.join(" · ") || "일치 없음"}` };
    }
    if (stored) {
      const storedReasons = stored.reasons || stored.exclusion_reasons || [];
      const detail = (Array.isArray(storedReasons) ? storedReasons : [storedReasons]).filter(Boolean).map(exclusionReasonLabel);
      const excluded = stored.excluded === true || ["excluded", "blocked", "fail"].includes(String(stored.status || "").toLowerCase());
      return { status: excluded ? "excluded" : "done", summary: detail.join(" · ") || stored.summary || (excluded ? "제외 조건 적용" : "제외 조건과 일치 없음") };
    }
    if (screeningCalls.length) {
      const failed = screeningCalls.some((call) => call.success === false) || candidate.status === "excluded";
      return { status: failed ? "excluded" : "done", summary: `${traceCallDetail(screeningCalls)}${reasons.length ? ` · ${reasons.join(" · ")}` : " · 제외 조건과 일치 없음"}` };
    }
    if (reasons.length) return { status: candidate.status === "excluded" ? "excluded" : "warning", summary: reasons.join(" · ") };
    return { status: "unrecorded", summary: "전용 금지약물·회수약물 스크리닝 trace 미기록" };
  }

  function candidateExclusionReasons(candidate, packet) {
    const reasons = [...(candidate.exclusion_reasons || [])];
    const restriction = candidate.metadata?.forbidden_drug_screen || candidate.restriction_screen || candidate.drug_restriction;
    const extra = restriction?.reasons || restriction?.matches || [];
    (Array.isArray(extra) ? extra : [extra]).filter(Boolean).forEach((item) => reasons.push(typeof item === "string" ? item : item.reason || item.name || JSON.stringify(item)));
    if (candidate.status === "excluded" && !reasons.length) reasons.push("excluded_without_recorded_reason");
    return [...new Set(reasons.map(exclusionReasonLabel))];
  }

  function exclusionReasonLabel(reason) {
    const value = String(reason || "");
    const [code, detail] = value.split(":", 2);
    const labels = {
      forbidden_drug: "연구자 지정 금지약물 일치",
      forbidden_drug_name: "금지약물명 일치",
      forbidden_drug_id: "금지약물 식별자 일치",
      forbidden_drug_class: "금지 약물 계열 일치",
      prohibited_regulatory_status: "회수·판매중단·금지 상태",
      withdrawn_drug: "시장 회수·판매 중단 약물",
      contraindicated_drug: "적응증·안전성 금기 조건 일치",
      molecular_weight_hard_limit: "분자량 hard limit 초과",
      hepatotoxicity_hard_limit: "이전 정책: 예측 DILI 값으로 하드 제외",
      herg_hard_limit: "이전 정책: 예측 hERG 값으로 하드 제외",
      invalid_structure: "구조 표준화·유효성 검사 실패",
      excluded_without_recorded_reason: "제외 상태이나 상세 사유 미기록",
    };
    const label = labels[code] || labels[value];
    return label ? `${label}${detail ? `: ${detail}` : ""}` : value.replaceAll("_", " ");
  }

  function admetPanelHtml(candidate) {
    const scores = candidate.metrics?.admet_scores || {};
    const endpoints = [
      ["hepatotoxicity", "간독성", 0.55, "risk"],
      ["herg_risk", "hERG 심장독성", 0.55, "risk"],
      ["ames_risk", "Ames 변이원성", 0.55, "risk"],
      ["cyp3a4_inhibition", "CYP3A4 저해", 0.65, "risk"],
      ["clearance_risk", "간세포 청소율 백분위", 0.65, "risk"],
      ["oral_bioavailability", "경구 생체이용률", null, "benefit"],
    ];
    return `<section class="admet-panel"><header><b>6종 ADMET 예측 기록</b><small>구조 기반 in silico 선별값 · 고위험 값은 후속 실험이 필요한 경고이며 확정 독성·안전성 판정이 아닙니다</small></header><div class="admet-grid">${endpoints.map(([key,label,threshold,kind]) => {
      const raw = Number(scores[key]);
      const available = Number.isFinite(raw);
      const flagged = available && kind === "risk" && raw >= threshold;
      const status = !available ? "missing" : flagged ? "risk" : kind === "benefit" && raw < 0.35 ? "warning" : "ok";
      return `<div class="admet-cell ${status}"><span>${esc(label)}</span><b>${available ? raw.toFixed(2) : "—"}</b><small>${!available ? "미기록" : threshold != null ? `기준 ${threshold.toFixed(2)} ${flagged ? "이상" : "미만"}` : "개발가능성 지표"}</small></div>`;
    }).join("")}</div></section>`;
  }

  function candidateWhyHtml(candidate, packet) {
    const portfolio = portfolioExecutionState(packet || {});
    const activity = candidate.activities?.[0];
    const metrics = candidate.metrics || {};
    const riskKeys = ["hepatotoxicity", "herg_risk", "ames_risk", "cyp3a4_inhibition", "clearance_risk"];
    const risks = riskKeys.map((key) => Number(metrics.admet_scores?.[key])).filter(Number.isFinite);
    const meanRisk = risks.length ? risks.reduce((total, value) => total + value, 0) / risks.length : null;
    const warnings = candidate.safety_flags?.length || 0;
    const breakdown = scoreBreakdown(candidate, packet);
    const factors = [
      ["약리 활성 근거", "28%", activity ? `${activity.endpoint} ${activity.relation} ${activity.value} ${activity.unit}` : "직접 활성값 부재 · 도킹 기반 대체점수", breakdown.activity],
      ["결합 적합성", "25%", metrics.docking_score_kcal_mol == null ? "도킹값 부재" : `Docking ${metrics.docking_score_kcal_mol} kcal/mol`, breakdown.docking],
      ["예측 ADMET 위험", "25%", meanRisk == null ? "ADMET 예측값 부재" : `5개 선별지표 평균 ${formatScore(meanRisk)}`, breakdown.safety],
      ["개발가능성", "12%", `경구 이용률 ${formatScore(metrics.admet_scores?.oral_bioavailability)} · 합성용이성 ${metrics.synthetic_accessibility ?? "—"}`, breakdown.developability],
      ["근거 충실도", "10%", `연결 근거 ${candidate.evidence?.length || 0}건 · 도구 추적 ${candidate.tool_trace_ids?.length || 0}건`, breakdown.evidence],
      ["예측·안전성 경고 페널티", "감점", warnings ? `${warnings}건 · ${Math.min(24, warnings * 4)}%p 감점` : "감점 없음", -breakdown.penalty],
    ];
    const portfolioRecord = candidate.metadata?.portfolio_ranking || {};
    const portfolioFactorLabels = {
      target_disease_evidence: "질환-표적 근거",
      within_target_rank: "표적 최고점 대비 연속 비율",
      measured_activity_quality: "측정 활성 품질",
      safety: "안전성",
      applicability: "적용 가능성",
      provenance_completeness: "출처 완결성",
    };
    const portfolioFactors = Object.entries(portfolioRecord.factors || {}).map(([key, value]) => {
      const weight = portfolioRecord.weights?.[key];
      return `<div><dt>${esc(portfolioFactorLabels[key] || key)}<em>${weight == null ? "" : `${Math.round(Number(weight) * 100)}%`}</em></dt><dd>${formatScore(value)}</dd></div>`;
    }).join("");
    const variantRecord = candidate.metadata?.assay_variant_alignment || null;
    const variantStatusLabels = {
      exact_match: "이전 판정: 정확 일치 · 재검증 필요",
      variant_class_match_only: "변이 유형만 일치 · 정확한 구성체 미확인",
      partial_variant_overlap: "변이 일부 겹침 · 복합 변이/구성체 미확인",
      variant_annotation_match: "변이명 일치 · 정확한 구성체 미확인",
      explicit_mismatch: "명시적 불일치",
      unknown_not_assumed_match: "미확인 · 일치로 추정 안 함",
      not_declared: "가설 변이 미지정",
    };
    const variantLine = variantRecord
      ? `Assay–가설 변이: ${variantStatusLabels[variantRecord.status] || variantRecord.status || "미기록"} · 가설 ${variantRecord.declared_hypothesis_variants?.join(", ") || "미지정"} · assay ${variantRecord.observed_assay_variants?.join(", ") || "미확인"} · 점수·적격성 자동 변경 없음`
      : "Assay–가설 변이 판정 미기록";
    const disclosureKey = `${state.snapshot?.run_id || "run"}:candidate:${candidate.candidate_id || candidate.name || candidate.smiles || "unknown"}`;
    return `<details class="why-card compact" data-disclosure="${esc(disclosureKey)}"><summary>후보 우선순위 결정 요인</summary><div class="why-content decision-factor-content">
      <p>${breakdown.hardExcluded
        ? `사전 제외 규칙이 적용되어 점수 <b>0.00</b>과 순위 미부여로 처리했습니다. 아래 요소는 순위 결정에 사용되지 않았습니다.`
        : portfolio.enabled
        ? `표적 내 점수 <b>${formatScore(candidate.score)}</b>(#${candidate.target_rank ?? candidate.rank ?? "—"})로 같은 표적의 후보를 정렬하고, 표적 간 비교에는 portfolio score <b>${candidate.portfolio_score == null ? "미기록" : formatScore(candidate.portfolio_score)}</b>(#${candidate.portfolio_rank || "—"})를 사용했습니다. 아래 기여값은 표적 내 점수의 재현값입니다.`
        : `저장된 통합 다기준 점수 <b>${formatScore(candidate.score)}</b>에 따라 정렬했습니다. 각 기여값은 기록된 후보 지표와 MA2 결정식을 사용한 재현값입니다.`}</p>
      <dl class="decision-factor-list">${factors.map(([label, weight, value, contribution]) => `<div><dt>${esc(label)}<em>${esc(weight)}</em></dt><dd>${esc(value)} <strong>${contribution >= 0 ? "+" : ""}${Number(contribution).toFixed(3)}</strong></dd></div>`).join("")}</dl>
      ${portfolio.enabled ? `<p class="score-reconcile-note">표적 간 순위는 raw docking score를 직접 비교하지 않고, 표적 최고 후보 대비 연속 점수 비율과 질환-표적 근거·측정 활성 품질·안전성·적용 가능성·출처 완결성을 결합합니다.</p>${portfolioFactors ? `<dl class="decision-factor-list">${portfolioFactors}</dl>` : '<p class="score-reconcile-note">portfolio ranking factor 기록이 없습니다.</p>'}` : ""}
      <p class="score-reconcile-note">${esc(variantLine)}</p>
      ${breakdown.reconciled ? "" : `<p class="score-reconcile-note">저장 점수와 현재 카드 재현값(${breakdown.total.toFixed(3)})이 다릅니다. 순위화 후 공개 DB 근거가 보강된 경우로, 저장 점수가 원 판정값입니다.</p>`}
    </div></details>`;
  }

  function scoreBreakdown(candidate, packet) {
    if (candidate.status === "excluded") {
      return { activity: 0, docking: 0, safety: 0, developability: 0, evidence: 0, penalty: 0, total: 0, reconciled: Number(candidate.score || 0) === 0, hardExcluded: true };
    }
    const persisted = packet?.metadata?.ranking_breakdown?.[candidate.candidate_id]
      || packet?.metadata?.ranking_breakdowns?.[candidate.candidate_id]
      || candidate.ranking_breakdown;
    if (persisted) {
      const value = (key) => Number(persisted[key]?.contribution ?? persisted[key] ?? 0);
      const result = { activity: value("activity"), docking: value("docking"), safety: value("safety"), developability: value("developability"), evidence: value("evidence"), penalty: Math.abs(value("penalty")) };
      result.total = clamp(result.activity + result.docking + result.safety + result.developability + result.evidence - result.penalty, 0, 1);
      result.reconciled = Math.abs(result.total - Number(candidate.score || 0)) <= 0.002;
      return result;
    }
    const metrics = candidate.metrics || {};
    const docking = Number(metrics.docking_score_kcal_mol ?? -5);
    const dockingScore = clamp((-docking - 5) / 5, 0, 1);
    const riskKeys = ["hepatotoxicity", "herg_risk", "ames_risk", "cyp3a4_inhibition", "clearance_risk"];
    const risks = riskKeys.map((key) => Number(metrics.admet_scores?.[key] ?? 0.5));
    const safetyScore = 1 - risks.reduce((total,value) => total + value, 0) / risks.length;
    const bioavailability = Number(metrics.admet_scores?.oral_bioavailability ?? 0.5);
    const sa = Number(metrics.synthetic_accessibility ?? 5.5);
    const developabilityScore = 0.5 * bioavailability + 0.5 * (1 - clamp((sa - 1) / 9, 0, 1));
    let activityScore = dockingScore * 0.8;
    if (candidate.activities?.length) {
      const factor = { pM: 0.001, nM: 1, uM: 1000, mM: 1000000 };
      const bestNm = Math.min(...candidate.activities.map((activity) => Number(activity.value) * (factor[activity.unit] || 1)));
      activityScore = 1 - clamp(Math.log10(Math.max(1, bestNm)) / 5, 0, 1);
    }
    const result = {
      activity: 0.28 * activityScore,
      docking: 0.25 * dockingScore,
      safety: 0.25 * safetyScore,
      developability: 0.12 * developabilityScore,
      evidence: 0.10 * Math.min(1, (candidate.evidence?.length || 0) / 2),
      penalty: Math.min(0.24, 0.04 * (candidate.safety_flags?.length || 0)),
    };
    result.total = clamp(result.activity + result.docking + result.safety + result.developability + result.evidence - result.penalty, 0, 1);
    result.reconciled = Math.abs(result.total - Number(candidate.score || 0)) <= 0.002;
    return result;
  }

  function renderAuditsArtifact() {
    const g0 = state.snapshot?.g0_verdict;
    const g1 = state.snapshot?.g1_verdict;
    const intake = getIntakeAssessment(state.snapshot || {});
    if (!g0 && !g1 && intake.abstained) {
      return `<div class="artifact-blocked-state">
        <span aria-hidden="true">?</span><b>입력 검증 ABSTAIN · G0 미실행</b>
        <p>질환 확인 또는 외부 검증이 완료되지 않아 감사할 가설 패킷을 만들지 않았습니다.</p>
        <small>감사 실패가 아니라 근거 없는 실행을 막은 사전 중단입니다.</small>
      </div>`;
    }
    if (!g0 && !g1) return '<div class="artifact-empty">MA3 감사 결과가 아직 없습니다.</div>';
    return `<div class="audit-grid">${auditArtifactCard("G0 · 가설 감사", g0)}${auditArtifactCard("G1 · 후보·과정 감사", g1)}</div>`;
  }

  function auditArtifactCard(title, verdict) {
    if (!verdict) {
      const packet = state.snapshot?.candidate_packet;
      const portfolio = portfolioExecutionState(packet || {});
      if (title.startsWith("G1") && packet && portfolio.abstainRequired) {
        return `<article class="artifact-card soft"><h4>${title}</h4><div class="artifact-blocked-state"><span aria-hidden="true">!</span><b>G1 판정 대기 · MA2 ${esc(portfolio.executionStatus.toUpperCase())}</b><p>${esc(portfolioCoverageText(packet, { includeApproved: true }))}</p><small>일부 표적의 실행이 완료되지 않아 전체 포트폴리오를 PASS 또는 연구자 승인 대상으로 표시하지 않습니다.</small></div></article>`;
      }
      return `<article class="artifact-card soft"><h4>${title}</h4><div class="artifact-empty" style="min-height:120px">감사 대기</div></article>`;
    }
    const code = String(verdict.verdict || "ABSTAIN").toLowerCase();
    const scope = title.startsWith("G1")
      ? g1AuditScopeText(state.snapshot?.candidate_packet || {}, verdict)
      : "가설 패킷의 근거·반증가능성·추적성 적합성 판정";
    const gate = title.startsWith("G1") ? "g1" : "g0";
    const groupedFindings = groupFindings(verdict.findings || []);
    const scopeHtml = code === "pass" ? "" : `<p class="audit-scope-note">${esc(scope)}</p>`;
    return `<article class="artifact-card"><h4>${title}</h4><div class="verdict-hero ${esc(code)}"><span class="verdict-code">${esc(verdict.verdict)}</span><div><b>${esc(verdictLabel(verdict.verdict))}</b><small>감사 항목 ${verdict.findings?.length || 0}건 · 검사 ${verdict.validations?.length || 0}건</small></div></div>
      ${scopeHtml}<p class="rationale">${esc(gateVerdictSummaryKo(verdict))}</p>${validationTraceHtml(verdict.validations || [], gate)}<div class="finding-list" style="margin-top:9px">${groupedFindings.map((group) => findingGroupHtml(group, gate)).join("")}</div>${reworkHtml(verdict.rework_order)}</article>`;
  }

  function renderCombinedRunReportArtifact() {
    return `<div class="combined-run-report">${renderRunReportArtifact()}${renderTechnicalDataArtifact()}</div>`;
  }

  function renderTechnicalDataArtifact() {
    const snapshot = state.snapshot || {};
    const hypothesisPacket = snapshot.hypothesis_packet || null;
    const candidatePacket = snapshot.candidate_packet || null;
    const compactJson = JSON.stringify(snapshot);
    const byteSize = new Blob([compactJson], { type: "application/json" }).size;
    const evidenceCount = countEvidence(snapshot);
    const toolCalls = candidatePacket?.tool_calls || [];
    const hashComplete = toolCalls.filter((item) => item.input_hash && item.output_hash).length;
    const artifactRows = [
      ["ResearchBrief", snapshot.brief?.schema_version, snapshot.brief?.brief_id, 1, snapshot.brief?.created_at],
      ["HypothesisPacket", hypothesisPacket?.schema_version, hypothesisPacket?.packet_id, hypothesisPacket?.hypotheses?.length, hypothesisPacket?.generated_at],
      ["G0 GateVerdict", snapshot.g0_verdict?.schema_version, snapshot.g0_verdict?.verdict_id, snapshot.g0_verdict?.findings?.length, snapshot.g0_verdict?.created_at],
      ["PlanSpec", snapshot.plan_spec?.schema_version, snapshot.plan_spec?.plan_id, snapshot.plan_spec?.tasks?.length, snapshot.plan_spec?.created_at],
      ["CandidatePacket", candidatePacket?.schema_version, candidatePacket?.packet_id, candidatePacket?.candidates?.length, candidatePacket?.generated_at],
      ["G1 GateVerdict", snapshot.g1_verdict?.schema_version, snapshot.g1_verdict?.verdict_id, snapshot.g1_verdict?.findings?.length, snapshot.g1_verdict?.created_at],
      ["HumanDecision", "—", snapshot.human_decision?.decision_id, snapshot.human_decision ? 1 : 0, snapshot.human_decision?.created_at],
    ];
    return `<article class="technical-data" aria-labelledby="technicalDataTitle">
      <header class="technical-data-header">
        <div>
          <span>상세 감사 부록</span>
          <h4 id="technicalDataTitle">기술 데이터</h4>
          <p>사람이 읽는 결론은 실행 보고서 위쪽에 있습니다. 이 부록은 재현·감사·장애 분석을 위해 원본 식별자, 스키마, 추적 정보와 전체 스냅샷을 손실 없이 보존합니다.</p>
        </div>
        <span class="technical-size">${esc((byteSize / 1024).toFixed(byteSize >= 1024 * 100 ? 0 : 1))} KB</span>
      </header>

      <section class="technical-summary-grid" aria-label="원본 스냅샷 요약">
        <article><span>실행 ID</span><b>${esc(snapshot.run_id || "미기록")}</b><small>${esc(snapshot.stage || "미기록")} · LLM ${esc(String(snapshot.mode || "미기록").toUpperCase())}</small></article>
        <article><span>레코드</span><b>${compactNumber(evidenceCount)}</b><small>근거 연결 · 이벤트 ${compactNumber(snapshot.events?.length || 0)}건</small></article>
        <article><span>도구 해시</span><b>${compactNumber(hashComplete)}/${compactNumber(toolCalls.length)}</b><small>입력·출력 해시가 모두 있는 호출</small></article>
        <article><span>최종 갱신</span><b>${esc(runReportDate(snapshot.updated_at))}</b><small>생성 ${esc(runReportDate(snapshot.created_at))}</small></article>
      </section>

      <section class="technical-artifact-section" aria-labelledby="technicalArtifactsTitle">
        <div class="technical-section-heading">
          <h5 id="technicalArtifactsTitle">저장 산출물 인덱스</h5>
          <p>‘미기록’은 false/0이 아니라 해당 snapshot에 필드가 없다는 뜻입니다.</p>
        </div>
        <div class="report-table-wrap"><table class="report-table technical-artifact-table">
          <caption>실행 스냅샷에 저장된 산출물 인덱스</caption>
          <thead><tr><th scope="col">산출물</th><th scope="col">스키마</th><th scope="col">식별자</th><th scope="col">주요 레코드</th><th scope="col">기록 시각</th></tr></thead>
          <tbody>${artifactRows.map(([name, schema, id, count, createdAt]) => `<tr>
            <td><b>${esc(name)}</b></td>
            <td>${esc(schema || "미기록")}</td>
            <td><code>${esc(id || "미기록")}</code></td>
            <td>${count == null ? "미기록" : compactNumber(count)}</td>
            <td>${esc(runReportDate(createdAt))}</td>
          </tr>`).join("")}</tbody>
        </table></div>
      </section>

      <section class="technical-purpose" aria-labelledby="technicalPurposeTitle">
        <h5 id="technicalPurposeTitle">왜 원본을 남기나</h5>
        <ul>
          <li>근거 레코드의 외부 ID·URL·조회시각·내용 해시와 주장 연결을 재검토합니다.</li>
          <li>도구 호출 기록의 입력·출력 해시, 백엔드, 캐시, 건너뜀·실패 상태를 재현합니다.</li>
          <li>감사 판정, 검사 결과, 재작업 지시와 실행 이벤트 원장을 축약 없이 확인합니다.</li>
        </ul>
        <p>원본 JSON은 일반적인 결과 읽기 화면이 아니며, 상단의 ‘원본 JSON 저장’ 버튼으로도 동일 snapshot을 받을 수 있습니다.</p>
      </section>

      <details class="technical-json-disclosure" data-disclosure="${esc(snapshot.run_id || "run")}:technical-json" data-technical-json>
        <summary><span><b>전체 원본 JSON</b><small>${esc((byteSize / 1024).toFixed(byteSize >= 1024 * 100 ? 0 : 1))} KB · 필요할 때만 펼침</small></span><i aria-hidden="true">⌄</i></summary>
        <div class="technical-json-note">필드명과 값은 저장 snapshot 원문입니다. 긴 데이터이므로 키보드 초점과 내부 스크롤을 제공합니다.</div>
        <pre class="json-view technical-json-view" tabindex="0" aria-label="원본 실행 JSON" data-raw-json>전체 원본은 이 영역을 펼칠 때 불러옵니다.</pre>
      </details>
    </article>`;
  }

  function renderNextActions() {
    const panel = $("#nextActionPanel");
    if (!panel) return;
    if (["cancel_requested", "cancelled"].includes(state.snapshot?.stage)) {
      panel.hidden = false;
      $("#nextActionSummary").textContent = "사용자 중단 · 추가 자료 제출과 재감사는 닫혀 있습니다. 기존 기록은 결과와 감사 탭에서 확인할 수 있습니다.";
      $("#nextActionDisposition").textContent = "USER STOP";
      $("#nextActionContent").innerHTML = reviewHistoryHtml();
      return;
    }
    const packet = state.snapshot?.next_action_packet;
    $("#nextActionTitle").textContent = packet
      ? `다음 조치 · ${packet.actions?.length || 0}건 / 실험 요청 ${packet.experiment_requests?.length || 0}건`
      : "다음 조치 · 이력 보기";
    const progress = state.snapshot?.next_action_progress;
    const openReview = state.snapshot?.stage === "human_gate" && !state.snapshot?.human_decision && Boolean(state.snapshot?.g1_verdict);
    const g0Progress = g0FollowupHtml();
    const automaticG0Running = Boolean(g0Progress) && ["ma1", "g0"].includes(state.snapshot?.stage);
    panel.hidden = !packet && !(state.snapshot?.g1_reaudits || []).length && !openReview && !g0Progress;
    if (!packet) {
      $("#nextActionSummary").textContent = automaticG0Running ? "원인별 근거 보강·재감사 진행 중 · 사용자 승인 불필요" : "후속 검토와 재감사 이력";
      $("#nextActionDisposition").textContent = automaticG0Running ? "AUTOMATIC G0 RUNNING" : openReview ? "REVIEW AVAILABLE" : "REVIEW HISTORY";
      $("#nextActionContent").innerHTML = g0Progress + semanticFailureHtml() + automaticResolutionHtml() + (automaticG0Running ? "" : reviewControlsHtml() + reviewHistoryHtml());
      return;
    }
    const kindLabel = {
      computational_rework: "코드·계약 보정",
      experiment_required: "추가 실험",
      data_required: "근거 보강",
      tool_configuration: "도구 설정",
      researcher_decision: "연구자 결정",
      resource_required: "자원 승인",
      integrity_block: "무결성 조사",
      operator_review: "운영자 검토",
    };
    const ownerLabel = {
      software_team: "소프트웨어 팀",
      experimental_team: "실험 팀",
      data_curator: "데이터 큐레이터",
      research_lead: "연구 책임자",
      platform_operator: "플랫폼 운영자",
      integrity_officer: "무결성 담당자",
    };
    const actionStatusLabel = {
      pending: "근거 대기",
      evidence_attached: "근거 첨부",
      completed: "제출 검토 기록 · 재감사 필요",
    };
    const experimentStatusLabel = {
      awaiting_results: "실측 결과 대기",
      results_attached: "결과 첨부",
      validated: "제출 검토 기록",
      submission_reviewed: "제출 검토 기록",
    };
    const list = (items = []) => items.length
      ? `<ul>${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>`
      : "";
    $("#nextActionSummary").textContent = g0StoppedBeforeMa2(state.snapshot)
      ? "G0에서 종료된 실행입니다. 아래는 저장된 미해결 조치 기록이며, 후보 생성 전 G1용 보강·재감사는 사용할 수 없습니다."
      : currentAutomaticResult() ? "자동 보완과 판단을 마쳤습니다. 아래 결론과 시도 내역을 확인하십시오. 추가 자료 제출은 선택 사항입니다." : packet.summary || "미해결 감사 항목을 확인하십시오.";
    $("#nextActionDisposition").textContent = String(packet.disposition || "ACTION REQUIRED").replaceAll("_", " ");
    const actions = (packet.actions || []).map((action) => {
      const status = progress?.packet_id === packet.packet_id
        ? progress.action_status?.[action.action_id] || "pending"
        : "pending";
      return `
      <article class="next-action-card">
        <header><h4>${esc(action.title || kindLabel[action.kind] || "후속 조치")}</h4><span class="next-action-meta">${esc(actionStatusLabel[status] || status)} · ${esc(action.priority || "—")} · ${esc(ownerLabel[action.owner] || action.owner || "담당 미정")}</span></header>
        <p>${esc(action.rationale || "상세 사유 미기록")}</p>
        ${action.required_inputs?.length ? `<b class="next-action-meta">필요 입력</b>${list(action.required_inputs)}` : ""}
        ${action.completion_criteria?.length ? `<b class="next-action-meta">완료 조건</b>${list(action.completion_criteria)}` : ""}
        ${reviewSubmissionHtml(action)}
        ${state.snapshot.stage === "human_gate" && !state.snapshot.human_decision && !g0StoppedBeforeMa2(state.snapshot) ? `<div class="review-card-actions"><button class="button secondary" data-review="evidence" data-action-id="${esc(action.action_id)}" ${reviewIsRunning() ? "disabled" : ""}>${action.kind === "experiment_required" ? "실험 결과 제출" : "자료 제출"}</button>${(state.snapshot.action_evidence_submissions || []).some(s => s.action_id === action.action_id) ? `<button class="button secondary" data-review="resolve" data-action-id="${esc(action.action_id)}" ${reviewIsRunning() ? "disabled" : ""}>제출 검토 기록</button>` : ""}</div>` : ""}
      </article>`;
    }).join("");
    const experiments = (packet.experiment_requests || []).map((request) => {
      const status = progress?.packet_id === packet.packet_id
        ? progress.experiment_status?.[request.request_id] || "awaiting_results"
        : "awaiting_results";
      return `
      <article class="next-action-card next-action-experiment">
        <header><h4>${esc(request.title || "추가 실험 요청")}</h4><span class="next-action-meta">${esc(experimentStatusLabel[status] || status)}</span></header>
        <p>${esc(request.objective || "감사 항목 해소")}</p>
        <b class="next-action-meta">필요 assay</b>${list(request.required_assays || [])}
        <details><summary>결과 제출 필드 · ${esc((request.required_result_fields || []).length)}개</summary>${list(request.required_result_fields || [])}</details>
        <p><b>합성 결과 금지:</b> 시스템은 측정값이나 통과 기준을 임의 생성하지 않습니다.</p>
      </article>`;
    }).join("");
    const checkpoint = packet.resume_checkpoint_id
      ? `체크포인트 ${packet.resume_checkpoint_id} · ${packet.resume_from_node || "재진입 단계 검토 필요"}`
      : "검증 가능한 체크포인트 없음 · 무결성 확인 후 재진입";
    const progressText = progress?.packet_id === packet.packet_id
      ? `근거 ${progress.submission_ids?.length || 0}건 · 완료 ${progress.resolution_ids?.length || 0}건${progress.ready_for_reaudit ? " · 재감사 준비 완료" : ""}`
      : "근거 제출 전";
    const gateMessage = state.snapshot?.g1_verdict?.verdict === "PASS"
      ? "G1 PASS는 완료되었습니다. 남은 한계·후속 조치에 따른 기존 APPROVE 잠금은 연구후보 채택과 별도로 유지됩니다."
      : progress?.ready_for_reaudit
      ? "필수 조치는 완료됐지만 자동 승인은 금지됩니다. 새 G1 재감사를 실행해야 합니다."
      : "새 G1 PASS 전에는 APPROVE할 수 없습니다.";
    $("#nextActionContent").innerHTML = `
      ${g0FollowupHtml()}${semanticFailureHtml()}${automaticResolutionHtml()}${reviewControlsHtml()}
      <div class="next-action-alert"><b>승인 잠금</b> · ${esc(gateMessage)} ${esc(checkpoint)} · ${esc(progressText)}</div>
      <div class="next-action-grid">${actions}${experiments}</div>
      <div class="next-action-reentry"><h4>재진입 조건</h4>${list(packet.reentry_conditions || [])}</div>${reviewHistoryHtml()}`;
  }

  function g0FollowupHtml() {
    const events = state.snapshot?.events || [];
    const plans = events.filter(e => e.event_type === "g0_followup_planned").map(e => e.metadata?.followup_plan).filter(Boolean);
    if (!plans.length) return "";
    const latest = plans.at(-1);
    const reports = events.filter(e => e.event_type === "g0_followup_sources_completed").map(e => e.metadata || {});
    const roles = {direction:"작용 방향 1차 근거", association:"질환 연관성 1차 근거", oa_context:"공개 원문·인용 문맥", assay_context:"필수 실험정보 조회 · 약효·표적 활성값 확정 아님"};
    roles.assay_identity = "문헌–화합물–실험 ID 연결 · 효능 판정 아님";
    roles.assay_fulltext = "동일 PMID 공개 원문 · 실험 방법 확인";
    const reasons = {AUTOMATIC_G0_WORK_AVAILABLE:"예산 내 자동 보완 계획", G0_REWORK_BUDGET_EXHAUSTED:"설정된 재작업 횟수 소진", G0_NO_SCIENTIFIC_DELTA:"과학적 내용 변화 없음 · 반복 중단", NO_RETAINED_HYPOTHESIS_NO_NEW_EVIDENCE:"보존 가설 0건 · 새로운 근거 확보 경로 없음 · 반복 중단", G0_NO_SAFE_AUTOMATIC_ACTION:"연결된 자동 경로로 해결할 수 없는 요구"};
    const selections = {EXISTING_PATHS_DONE_NO_SAFE_REPLAY:"기존 경로 완료 · 안전한 재시도 없음", TARGET_BUDGET_DEFERRED_NOT_SCIENTIFIC_REJECTION:"표적 수 예산으로 대기 · 과학적 탈락 아님", BOUND_TRANSIENT_PUBLIC_RETRY_WITHIN_EXISTING_BUDGET:"일시적 공개 API 오류 · 기존 예산 내 1회 재시도", UNATTEMPTED_ALLOWED_SOURCE_PATH:"아직 조회하지 않은 허용 출처", GATE_LINEAGE_OR_BUDGET_DOES_NOT_ALLOW_EXECUTION:"게이트·변경 내역·예산 조건상 실행 불가"};
    return `<section class="automatic-resolution"><h4>G0 원인별 자동 보완</h4><p>${esc(reasons[latest.reason_code] || latest.reason_code)} · 계획 ${plans.length}회 · 원인별 조회 보고 ${reports.length}회</p><p>조회 성공은 근거 적격성이나 통과를 뜻하지 않습니다. 새 자료도 원문 해석·MA1·G0 검사를 다시 거칩니다.</p><details><summary>자동 시도·남은 요구 상세</summary>${reports.map(r=>`<article><b>${esc(r.plan_id)}</b>${(r.observations || []).map(o=>`<p>${esc(o.target)} · ${esc(o.source)} · ${esc(roles[o.role] || o.role)}<br>${o.attempted ? "조회 시도" : "미실행"} · ${esc(o.status)} · 자료 ${Number(o.record_count || 0)}개 · ${esc(o.reason_code || "")}</p>${evidenceGapHtml(o.evidence_gap)}`).join("")}</article>`).join("")}<h5>다음 조회 선택·중단 이유</h5>${(latest.tasks || []).map(t=>`<p>${esc(t.target)} · ${esc(roles[t.role] || t.role)} · ${esc(selections[t.selection_reason] || "이전 실행 · 상세 선택 사유 미기록")}</p>`).join("")}${(latest.manual_requirements || []).map(f=>`<p>남은 요구 · ${esc(f.kind)} · ${esc(f.code)}</p>`).join("")}<p>계획 결속: ${esc(latest.packet_id)} / ${esc(latest.verdict_id)}<br>원본 packet SHA ${esc(latest.packet_sha256)}</p></details></section>`;
  }

  function evidenceGapHtml(gap) {
    if (gap?.policy !== "observed_evidence_gap_v1") return "<p>이전 실행 · 추출 단계별 진단 미기록. 근거가 없다고 단정할 수 없습니다.</p>";
    const outcomes = {transient_retrieval_failure:"일시적 조회 오류", retrieval_or_protocol_failure:"조회·응답 검증 오류", records_returned_with_source_errors:"일부 자료 확보 · 출처 오류 남음", document_unit_validation_pending:"문장 연결 근거 검증 대기", records_returned_not_gate_pass:"자료 확보 · 게이트 판정은 별도", screened_without_qualified_excerpt:"검색 자료 심사 종료 · 적격 발췌 미확보", no_hits_in_this_query:"이번 검색식의 결과 0건", budget_not_attempted:"예산으로 조회 미실행", not_attempted:"조회 미실행", insufficient_diagnostic_telemetry:"진단 기록 부족"};
    const buckets = {abstract_not_available:"초록 미확보", abstract_received_excerpt_not_selected:"초록 수신 · 관련 발췌 미선택(추출 오류 확정 아님)", historical_extraction_state_unknown:"과거 기록만으로 추출 상태 구분 불가", publication_ineligible:"원저 출판 유형 미충족", publication_metadata_unverified:"출판 메타데이터 미검증", selected_excerpt_criteria_unmet:"선택 발췌의 검증 조건 미충족", counter_or_mixed_context_not_direct_refutation:"부정·혼합 문맥(직접 반증 확정 아님)", document_unit_validation_pending:"문장 연결 검증 대기", adapter_selected_pending_downstream_checks:"수집기 선택 · 후속 검증 필요", unclassified_observation:"기타·미분류"};
    const actions = {retry_public_path_once_if_budget_and_lineage_allow:"허용 조건 내 공개 조회 1회 재시도", try_unattempted_licensed_fulltext_path:"미시도 공개 라이선스 원문 경로 확인", do_not_infer_absence_from_legacy_telemetry:"과거 기록의 미확인을 유지", validate_preserved_document_unit_with_existing_gates:"보존된 원문 단위 검증", try_other_unattempted_allowed_path_then_hold:"다른 미시도 허용 경로 확인 후 미해소 시 보류", hold_with_explicit_unresolved_reason:"미해소 사유를 남기고 보류", preserve_partial_records_and_unresolved_source_errors:"부분 자료와 출처 오류를 함께 보존"};
    buckets.assay_context_unassigned_measurement = "실험정보 문맥 · 수치와 약물·표적의 연결 미검증";
    outcomes.assay_identity_inventory_completed = "ChEMBL 문헌·실험 ID 조회 완료 · 효능 판정 아님";
    outcomes.assay_fulltext_context_completed = "동일 문헌 공개 원문 확인 완료 · 실험 동등성 판정 아님";
    actions.preserve_assay_context_without_assigning_potency = "원문 수치를 문맥으로 보존 · 활성값·효능으로 자동 배정하지 않음";
    const count = value => Number.isSafeInteger(value) && value >= 0 ? `${value.toLocaleString()}건` : "미기록";
    const rows = Object.entries(gap.exclusive_record_buckets || {}).filter(([key]) => Object.hasOwn(buckets, key)).map(([key,value])=>`${esc(buckets[key])} ${esc(count(value))}`).join(" · ");
    const next = Array.isArray(gap.next_actions) ? gap.next_actions : [];
    const outcome = Object.hasOwn(outcomes, gap.outcome) ? outcomes[gap.outcome] : "진단 기록 부족";
    const retention = Object.hasOwn(gap, "matched_context_documents") ? `<p>실험정보 문맥 발견 ${esc(count(gap.matched_context_documents))} · 보존 ${esc(count(gap.retained_context_documents))} · 보존 예산으로 미포함 ${esc(count(gap.omitted_matching_context_documents))}(과학적 탈락 아님)</p>` : "";
    const identity = Object.hasOwn(gap, "identity_document_count") ? `<p>ChEMBL 문헌 ${esc(count(gap.identity_document_count))} · 등록 측정행 조회 ${esc(count(gap.identity_activity_rows_screened))} · 표적 ID 연결 ${esc(count(gap.identity_linked_activity_count))}</p><p>논문 속 수치 배정·정확한 변이·실험 조건 확인은 별도입니다. 미기록 조건은 추정하지 않습니다.</p>` : "";
    const annotations = gap.identity_annotation_counts;
    const applicability = annotations && typeof annotations === "object" && !Array.isArray(annotations) ? `<details><summary>실험 적용 범위 · 변이 미기록 ${esc(count(annotations.not_recorded))} / 미확인 ${esc(count(annotations.unresolved_annotation))}</summary><p>변이명 표기 ${esc(count(annotations.named_annotation_only))} · 변이군 표기 ${esc(count(annotations.class_annotation_only))} · 등록 필드 충돌 ${esc(count(annotations.annotation_conflict))}</p><p>UNDEFINED MUTATION은 확인된 변이가 아닙니다. 변이명 일치도 전체 유전자형·정확한 실험체·질환 모델의 일치를 증명하지 않습니다. 원문 및 실험 조건 검증이 필요합니다.</p></details>` : "";
    const ft = gap.assay_fulltext_counts;
    const methods = ft && typeof ft === "object" && !Array.isArray(ft) ? `<details><summary>공개 원문 검증 ${esc(count(ft.parsed))} · 실험 방법 문단 ${esc(count(gap.assay_method_paragraphs))}</summary><p>OA·라이선스 조건 미충족 ${esc(count(ft.oa_not_admitted))} · PMID 미검색 ${esc(count(ft.metadata_not_found))} · 예산으로 미완료 ${esc(count(ft.budget_not_completed))} · 조회·검증 오류 ${esc(count(ft.retrieval_or_validation_error))}</p><p>관련 표 위치 ${esc(count(gap.assay_table_locators))} · 표의 수치 배정은 미검증입니다. 원문 접근 불가는 생물학적 근거 부재가 아닙니다.</p></details>` : "";
    return `<div class="next-action-alert"><b>${esc(outcome)}</b><p>검색 ${esc(count(gap.search_result_count))} · 심사 ${esc(count(gap.records_screened))} · 검색 창 밖 ${esc(count(gap.unsearched_result_count))}</p>${rows ? `<p>${rows}</p>` : ""}${retention}${identity}${applicability}${methods}<p>${next.filter(key=>typeof key === "string" && Object.hasOwn(actions, key)).map(key=>esc(actions[key])).join(" · ")}</p><small>이 검색 범위·선택 발췌에 대한 진단입니다. 전체 문헌의 근거 부재나 과학적 탈락을 증명하지 않습니다.</small></div>`;
  }

  function semanticFailureHtml() {
    const failures = (state.snapshot?.events || []).filter(e => ["llm_fallback", "llm_call_failed"].includes(e.event_type) && String(e.node || "").startsWith("ma1-evidence-"));
    if (!failures.length) return "";
    const reasons = {OUTPUT_LIMIT:"추론·출력 토큰 한도 도달", PROVIDER_INCOMPLETE:"모델 응답 미완료", PROVIDER_REFUSAL:"모델 응답 거부", EMPTY_STRUCTURED_OUTPUT:"구조화 응답 없음", STRUCTURED_OUTPUT_INVALID:"응답 스키마 검증 실패", TRANSPORT_FAILURE:"API 통신 실패"};
    return `<section class="automatic-resolution"><h4>근거 해석 기술 오류 · ${failures.length}건</h4><p>논문에 근거가 없다는 뜻이 아닙니다. 미완료 응답은 미확인으로 남기고, 사용한 토큰·비용은 기록합니다.</p><details><summary>오류 사유 상세</summary>${failures.map(e=>`<p>${esc(e.node)} · ${esc(reasons[e.metadata?.reason_code] || e.metadata?.reason_code || "이전 실행 · 상세 오류 코드 미기록")}</p>`).join("")}</details></section>`;
  }

  function reviewIsRunning() {
    return (state.snapshot?.g1_reaudits || []).some(r => r.status === "running");
  }

  const OBSERVED_FIELDS = ["candidate_id", "assay_name", "replicate_id", "dose", "dose_unit", "timepoint", "endpoint", "measured_value", "value_unit", "control", "uncertainty"];
  const ACTION_EVIDENCE_TYPES = { computational_rework: "code_validation", experiment_required: "experimental_result", data_required: "source_document", tool_configuration: "tool_readiness", researcher_decision: "researcher_policy", resource_required: "resource_approval", integrity_block: "integrity_report", operator_review: "researcher_policy" };

  function currentAutomaticResult(snapshot = state.snapshot) {
    const parent = (snapshot?.g1_reaudits || []).filter(r => r.mode === "automatic").at(-1);
    if (parent?.status !== "completed") return null;
    return (snapshot?.automatic_resolution_reports || []).findLast(r => r.request_id === parent.request_id && r.source_sha256 === parent.source_sha256 && r.final_verdict_id === parent.verdict_id && r.final_verdict_id === snapshot?.g1_verdict?.verdict_id) || null;
  }

  function automaticResolutionHtml() {
    const result = currentAutomaticResult();
    if (!result) return "";
    const labels = {HOLD:"보류", BLOCKED:"진행 불가", READY_FOR_REVIEW:"검토 준비 완료"};
    const reason = {EXPERIMENTAL_VALIDATION_UNAVAILABLE:"실제 실험 검증 자료 없음", EXACT_ASSAY_CONTEXT_NOT_VERIFIED:"정확한 실험·변이 조건 미확인", EXTERNAL_REVIEW_CONCERN_UNRESOLVED:"외부 감사 지적 미해소", CONFIGURED_PATHS_EXHAUSTED_OR_BUDGET_BOUND:"연결된 자동 조회·검토 경로 점검 종료", AUTOMATED_STEP_FAILED_NO_AUTOMATIC_REPLAY:"처리 오류 또는 시간·비용 제한 · 불명확한 호출 반복 안 함", AUDIT_REJECTED_AUTOMATED_PROGRESSION:"감사 기준에 따라 자동 진행 불가", AUTOMATED_CHECKS_COMPLETE_HUMAN_APPROVAL_NOT_GIVEN:"자동 검사 완료 · 연구자 승인은 별도"};
    const usage = (state.snapshot.token_usage || []).slice(result.usage_start_index,result.usage_end_index);
    const cost = usage.reduce((n,u)=>n+Number(u.estimated_cost_usd || 0),0);
    return `<section class="automatic-resolution"><h4>자동 분석 완료 · ${esc(labels[result.outcome] || result.outcome)}</h4><p>${esc(result.reason_codes.map(c=>reason[c] || c).join(" · "))}</p><p>자료를 사람이 찾아 넣을 때까지 실행 중인 상태가 아닙니다. 가용 경로에 대한 이번 자동 분석은 종료됐으며, 새 자료·접근 권한·코드 변경이 생겼을 때 추가 검토할 수 있습니다. 보류는 약효가 없다는 판정이 아닙니다.</p><p>이번 자동 보완 모델 ${usage.length}회 · $${cost.toFixed(6)}</p><details><summary>시도한 경로와 최종 판단 상세</summary>${result.steps.map(s=>`<p>${esc(({sources:"공개 원문 자동 조회",rules:"규칙 재검사",external:"외부 모델 재감사",automatic:"자동 처리",code_recalculation:"집계 재계산"})[s.step] || s.step)} · ${esc(({completed:"완료",failed:"중단",skipped:"추가 호출 없음"})[s.status] || s.status)}${s.reason_code ? " · "+esc(s.reason_code) : ""}</p>`).join("")}${(result.code_checks || []).map(c=>`<p>${esc(c.target)} 약물 집계 재계산: ${Number(c.recomputed_count)}개 · 제외 라벨 ${esc((c.excluded_labels || []).join(", ") || "없음")}<br><small>원본 기록 변경 없이 계산 · ${esc(c.hypothesis_id)}</small></p>`).join("")}${(result.remaining_findings || []).map(f=>`<p>${esc(f.candidate_id || "전체")} · ${esc(f.code)}<br>${esc(f.message)}</p>`).join("")}<small>${esc(result.report_id)}<br>Source SHA ${esc(result.source_sha256)}<br>Candidate SHA ${esc(result.candidate_packet_sha256)}</small></details></section>`;
  }

  async function loadReviewEligibility(snapshot) {
    if (state.preview || snapshot.stage !== "human_gate" || snapshot.human_decision) return;
    const version = `${snapshot.run_id}:${snapshot.updated_at}:${snapshot.events?.length || 0}`;
    if (state.reviewStatus?.snapshotVersion === version || state.reviewStatusLoading === version) return;
    state.reviewStatusLoading = version;
    try {
      const result = await api(`/api/runs/${encodeURIComponent(snapshot.run_id)}/review-status`);
      if (state.snapshot?.run_id !== snapshot.run_id || state.reviewStatusLoading !== version) return;
      state.reviewStatus = {...result, snapshotVersion:version};
      renderNextActions();
    } catch (_) {
      if (state.snapshot?.run_id !== snapshot.run_id || state.reviewStatusLoading !== version) return;
      state.reviewStatus = {run_id:snapshot.run_id, snapshotVersion:version, modes:{}, unavailable:true};
      renderNextActions();
    } finally {
      if (state.reviewStatusLoading === version) state.reviewStatusLoading = false;
    }
  }

  function reviewControlsHtml() {
    const s = state.snapshot;
    const open = s.stage === "human_gate" && !s.human_decision;
    if (!open) return `<p class="review-explanation">이 실행의 최종 결정이 기록되어 추가 제출은 닫혀 있습니다.</p>`;
    const busy = reviewIsRunning();
    const version = `${s.run_id}:${s.updated_at}:${s.events?.length || 0}`;
    const status = state.reviewStatus?.snapshotVersion === version ? state.reviewStatus : null;
    const hasG1 = s.g0_verdict?.verdict === "PASS" && !!s.candidate_packet && !!s.g1_verdict;
    const disabled = kind => busy || !hasG1 || status?.modes?.[kind]?.eligible !== true ? "disabled" : "";
    const eligibilityNotice = !hasG1
      ? "G0 근거 검토에서 종료됐습니다. 후보가 없어 G1용 보강·재감사는 실행할 수 없습니다. MA1 근거 해석·조회 경로를 보완한 뒤 재진입해야 합니다. 과거 조치 분류는 원본 기록이며, 실험 필요 여부는 별도 검토 대상입니다."
      : !status ? "보강 가능 여부를 확인 중입니다."
      : status.unavailable ? "보강 상태를 확인하지 못했습니다. 상태 새로고침 후 다시 확인하십시오."
      : Object.entries(status.modes || {}).filter(([,m])=>!m.eligible).map(([k,m])=>`${k}: ${m.reason_code || "실행 불가"}`).join(" · ");
    return `<div class="review-workspace"><h4>보완과 재감사</h4>
      <p class="review-eligibility-notice">${esc(eligibilityNotice)}</p>
      <p>${hasG1 ? "G1 후보가 있는 라이브 실행은 가용 원문 경로 → 출처 검증 → 규칙·필요 시 예산 내 외부 재감사 → 자동 결론까지 진행합니다. 적격한 기존 실행도 아래 자동 보완으로 이어갈 수 있습니다. 자료 제출은 선택 사항입니다." : "현재 화면에서는 기존 감사 사유를 검토하고 보류 메모를 남길 수 있습니다. 자료 업로드나 후보 승인을 요구하는 단계가 아닙니다."}</p>
      <button class="button human" data-review="automatic" ${disabled("automatic")}>자동 보완·판정 실행</button>
      <div class="review-card-actions"><button class="button secondary" data-review="sources" ${disabled("sources")}>공개 원문 자동 보강 · 무료</button><button class="button secondary" data-review="note" ${busy ? "disabled" : ""}>임시 보류 메모</button><button class="button secondary" data-review="rules" ${disabled("rules")}>규칙 재검사 · 무료</button><button class="button human" data-review="external" ${disabled("external")}>G1 재감사 · 유료 모델</button><button class="button secondary" data-review="refresh">상태 새로고침</button></div>
      <p>${busy ? "재감사 처리 중 · 자료와 결정 입력은 완료 후 가능합니다." : "임시 보류는 제출 창을 유지합니다. 무료 검사는 이전 외부 감사 지적을 유지하며, 유료 재감사는 남은 실행 예산을 사용합니다."}</p>
      <p>자동 보강은 후보의 ChEMBL 실험·문헌 ID로 Europe PMC 공개 원문을 확인하고 관련 구문을 후속 감사 자료에 연결합니다. 논문 확보만으로 실험 검증이나 승인 조건이 해소되지는 않습니다. 직접 첨부한 PDF는 원본 보존용이며 자동 본문 해석은 아직 지원하지 않습니다.</p>
      ${busy ? '<button class="button secondary" data-review="recover">중단 여부 확인·기록 정리</button>' : ""}</div>`;
  }

  function reviewSubmissionHtml(action) {
    const submissions = (state.snapshot.action_evidence_submissions || []).filter(s => s.action_id === action.action_id);
    if (!submissions.length) return "";
    return `<details class="review-submissions"><summary>제출 자료 ${submissions.length}건</summary>${submissions.map(s => `<div><b>${esc(s.source_filename)}</b> · ${esc(s.submitted_by)} · 실측 행 ${(s.observed_results || []).length}<p>${esc(s.notes)}</p><small>SHA-256 ${esc(s.source_artifact_sha256)}</small><p><a href="/api/runs/${encodeURIComponent(state.snapshot.run_id)}/attachments/${encodeURIComponent(s.attachment_id)}" download>원본 자료 받기</a> · 접수 자료 / 과학적 검증 별도</p></div>`).join("")}</details>`;
  }

  function reviewHistoryHtml() {
    const s = state.snapshot || {};
    const notes = s.review_notes || [], audits = s.g1_reaudits || [];
    if (!notes.length && !audits.length) return publicSourceHistoryHtml();
    return `<details class="review-history" open><summary>후속 검토 이력 · 메모 ${notes.length} · 재감사 ${audits.length}</summary>
      ${notes.map(n => `<article><b>임시 보류 메모 · ${esc(n.researcher)}</b><p>${esc(n.note)}</p><small>${esc(n.created_at)}</small></article>`).join("")}
      ${audits.map(r => {
        const usage = (s.token_usage || []).slice(r.usage_start_index, r.usage_end_index ?? undefined);
        const cost = usage.reduce((sum, u) => sum + Number(u.estimated_cost_usd || 0), 0);
        const tokens = usage.reduce((sum, u) => sum + Number(u.input_tokens || 0) + Number(u.output_tokens || 0) + Number(u.reasoning_tokens || 0), 0);
        return `<article><b>${esc(({automatic:"자동 보완·판정", external:"외부 모델 G1 재감사", sources:"공개 원문 자동 보강 · 판정 변경 없음", rules:"규칙 재검사"})[r.mode] || r.mode)} · ${esc(({running:"진행 중", completed:"완료", failed:"중단"})[r.status] || r.status)}</b><p>${esc(r.rationale)}</p><p>이번 처리 ${tokens.toLocaleString()} tokens · $${cost.toFixed(6)}</p><p>해소: ${esc((r.cleared_finding_codes || []).join(", ") || "없음")}<br>유지: ${esc((r.remaining_finding_codes || []).join(", ") || "없음")}</p>${r.error_code ? `<p>중단 사유: ${esc(r.error_code)} · 자동 재실행하지 않음</p>` : ""}<details><summary>감사 연결 기록</summary><p>${esc(r.previous_verdict_id)} → ${esc(r.verdict_id || "미완료")}</p><small>${esc(r.request_id)} · ${esc(r.started_at)}</small></details></article>`;
      }).join("")}</details>${publicSourceHistoryHtml()}`;
  }

  function publicSourceHistoryHtml() {
    const reports = state.snapshot?.public_source_reports || [];
    if (!reports.length) return "";
    const reasons = {REVIEW_CONTEXT_FOUND:"관련 구문 확보 · 과학적 적합성 검토 필요", NO_RELEVANT_CONTEXT_FOUND:"원문 확인 · 관련 구문은 찾지 못함", EXACT_ASSAY_DOCUMENT_LINK_MISSING:"원래 실험과 문헌의 정확한 연결 없음", DOCUMENT_HAS_NO_PUBMED_OR_DOI:"문헌에 PMID·DOI 없음", PUBLICATION_IDENTITY_NOT_UNIQUE:"동일 논문으로 단일 확정하지 못함", PUBLICATION_IDENTITY_CONFLICT:"논문 식별자 불일치", ASSAY_IDENTITY_CHANGED_OR_CONFLICTED:"원래 실험·표적·문헌과 조회 결과 불일치", DOCUMENT_IDENTITY_CONFLICT:"ChEMBL 문헌 식별자 불일치", LIVE_SOURCE_PROVENANCE_REQUIRED:"실제 조회 출처 확인 필요", HTTP_CALL_BUDGET_REACHED:"이번 요청의 HTTP 횟수 한도 도달", ELAPSED_BUDGET_REACHED:"이번 요청의 시간 한도 도달", RESPONSE_BYTE_LIMIT:"응답 크기 안전 한도 초과"};
    Object.assign(reasons, {NCBI_PMC_RECORD_UNAVAILABLE:"NCBI PMC에도 연결된 공개 원문 없음", NCBI_PMCID_MISSING:"NCBI PMC 원문 식별자 없음", NCBI_REUSE_SET_NOT_CONFIRMED:"NCBI 원문 자동 재사용 조건 미확인", NCBI_EMBARGO_OR_RELEASE_CONDITION:"NCBI 공개 유예·배포 조건 확인 필요", NCBI_LICENSE_NOT_ALLOWED_OR_AMBIGUOUS:"NCBI 원문 이용 라이선스 미확인", OA_PMCID_MISSING:"Europe PMC에 연결된 PMCID 없음 · 공개 원문 미확보", OA_NOT_OPEN_ACCESS:"공개 원문 이용 조건 미충족", OA_LICENSE_MISSING:"원문 이용 라이선스 미확인", OA_LICENSE_NOT_ALLOWED:"허용된 원문 이용 라이선스 아님", OA_RETRACTED_PUBLICATION:"철회된 논문 · 근거 사용 제외", ORIGINAL_ACTIVITY_BINDING_CONFLICT:"원래 활성·후보·표적 연결 대조 불일치", ORIGINAL_ACTIVITY_BINDING_NOT_UNIQUE:"원래 활성 근거를 하나로 확정하지 못함", ACTIVITY_IDENTITY_CHANGED_OR_CONFLICTED:"공식 활성 기록과 원래 5개 식별자 불일치", DOCUMENT_IDENTIFIER_INVALID:"문헌 PMID 또는 DOI 형식 오류"});
    return `<section class="public-source-reviews"><h4>자동 보강 원문 · ${reports.length}회</h4><p>아래 구문은 검토용 문헌 맥락입니다. 후보·정확한 실험·변이·측정값의 일치가 자동 검증된 근거는 아닙니다. 기존 후보와 G1 판정은 보존됩니다.</p>${reports.slice().reverse().map(report => `<details class="review-history" open><summary>${esc(report.created_at)} · 문헌 ${(report.documents || []).length}건 · 모델 호출 0</summary>${(report.documents || []).map(doc => {
      const url = /^PMC\d+$/.test(doc.pmcid || "") ? (doc.fulltext_source === "NCBI PMC" ? `https://pmc.ncbi.nlm.nih.gov/articles/${doc.pmcid}/` : `https://europepmc.org/articles/${doc.pmcid}`) : null;
      return `<article><b>${esc(doc.candidate_name)} · ${doc.status === "available_for_review" ? "원문 확인" : doc.status === "not_attempted" ? "조회 전 중단" : "원문 미확보"}</b><p>${esc(reasons[doc.reason_code] || "확인 제한 · " + doc.reason_code)}</p><p>ChEMBL 실험 ${esc(doc.assay_chembl_id || "없음")} → 문헌 ${esc(doc.document_chembl_id || "없음")} → ${esc(doc.fulltext_source || "Europe PMC / NCBI PMC")} ${esc(doc.pmcid || "미확보")}</p><p>PMID ${esc(doc.pmid || "없음")} · DOI ${esc(doc.doi || "없음")}${url ? ` · <a href="${url}" target="_blank" rel="noopener noreferrer">원문 보기</a>` : ""}</p><p>라이선스 ${esc(doc.license || "미확인")} · 식별자 검증 ${doc.identity_verified ? "확인" : "미완료"} · 관련 구문 ${(doc.passages || []).length}개${doc.passage_limit_reached ? " / 표시 상한 도달" : ""}</p>${(doc.passages || []).map(p => `<blockquote><p>${esc(p.text)}${p.truncated ? " [구문 길이 제한으로 일부 생략]" : ""}</p><small>${esc(p.section)} · 후보명 ${p.candidate_alias_present ? "있음" : "없음"} · 표적명 ${p.target_present ? "있음" : "없음"} · ${esc(p.source_location)}</small></blockquote>`).join("")}<details><summary>출처·요청·해시 상세 ${(doc.calls || []).length}회</summary><small>원문 SHA-256 ${esc(doc.document_sha256 || "미확보")}<br>해시 범위 ${esc(doc.document_hash_scope || "raw_response_body")}<br>경로 결과 ${esc((doc.attempt_outcomes || []).join(" → ") || "상세 요청 참조")}<br>관련 지적 ${esc((doc.finding_codes || []).join(", "))}</small>${(doc.calls || []).map(c => `<p>${esc(c.source_database)} · HTTP ${esc(c.status_code ?? "응답 없음")} · ${esc(c.reason_code || (c.complete ? "수신 완료" : "미완료"))}<br><small>${esc(c.endpoint)}<br>${esc(JSON.stringify(c.parameters))}<br>${esc(c.response_sha256 || "본문 해시 없음")} · ${Number(c.response_bytes || 0).toLocaleString()} bytes</small></p>`).join("")}</details></article>`;
    }).join("")}<p>논문으로 자동 처리하지 않는 지적: ${esc((report.non_document_finding_codes || []).join(", ") || "없음")}</p><small>${esc(report.report_id)} · HTTP 최대 ${Number(report.limits?.http_calls)}회 / ${Number(report.limits?.elapsed_seconds)}초 · 재조회 간격 ${Number(report.limits?.refresh_cooldown_seconds) / 60}분</small></details>`).join("")}</section>`;
  }

  async function handleReviewAction(event) {
    const button = event.target.closest("[data-review]");
    if (!button || !state.snapshot?.run_id || state.preview) return;
    const kind = button.dataset.review, runId = state.snapshot.run_id;
    button.disabled = true;
    try {
      const status = await api(`/api/runs/${encodeURIComponent(runId)}/review-status`);
      if (runId !== state.snapshot?.run_id) return;
      if (kind === "refresh") { await fetchRun(runId); return; }
      if (kind === "recover") {
        if (!status.can_recover) throw new Error("현재 서버에서 처리 중이거나 실행 잠금이 유지되고 있습니다. 완료 후 다시 확인하십시오.");
        await api(`/api/runs/${encodeURIComponent(runId)}/reaudits/recover`, {method:"POST"});
        await fetchRun(runId); return;
      }
      if (["rules", "external", "sources", "automatic"].includes(kind) && !status.modes[kind].eligible) {
        const reason = status.modes[kind].reason_code;
        throw new Error(reason === "INPUTS_UNCHANGED" ? "같은 자료와 코드로 이미 검사했습니다. 새 근거 또는 코드 변경 후 다시 검사할 수 있습니다." : reason === "SOURCE_REFRESH_COOLDOWN" ? "같은 후보·지적으로 원문을 이미 조회했습니다. 15분 후 다시 확인할 수 있습니다." : `현재 재감사 불가: ${reason}`);
      }
      if (!["rules", "external", "sources", "automatic"].includes(kind) && !status.can_attach) throw new Error("현재 자료 제출이 닫혀 있거나 재감사 중입니다.");
      const action = (state.snapshot.next_action_packet?.actions || []).find(a => a.action_id === button.dataset.actionId);
      const request = (state.snapshot.next_action_packet?.experiment_requests || []).find(r => action?.finding_ids.some(id => r.source_finding_ids.includes(id)));
      state.reviewAction = {kind, runId, action, request, baseVerdictId:status.base_verdict_id};
      $("#reviewActionForm").reset();
      $("#reviewActionError").hidden = true;
      $("#reviewUploadFields").hidden = kind !== "evidence";
      $("#reviewObservedFields").hidden = action?.kind !== "experiment_required";
      $("#reviewPaidFields").hidden = !["external", "automatic"].includes(kind);
      $("#reviewActionTitle").textContent = ({note:"임시 보류 메모", evidence:"자료 제출", resolve:"제출 검토 기록", rules:"현재 기록 규칙 재검사", automatic:"자동 보완·판정", external:"외부 모델 G1 재감사", sources:"공개 원문 자동 보강 · 무료"})[kind];
      if (kind === "automatic") $("#reviewActionTitle").textContent = "자동 보완·판정";
      $("#reviewActionIntro").textContent = kind === "sources" ? "현재 후보의 정확한 ChEMBL 실험·문헌에서 출발해 Europe PMC 공개 원문과 관련 구문을 확보합니다. 공식 API 최대 24회·180초, 모델 비용 없음. 확보·실패 이유를 모두 기록하며 판정은 변경하지 않습니다. 유료 재감사는 별도 실행입니다." : kind === "note" ? "검토 메모를 저장합니다. 자료 제출과 재감사는 계속 가능합니다." : kind === "resolve" ? "선택 항목에 제출된 자료를 검토했다는 기록입니다. 과학적 합격이나 경고 해제를 의미하지 않습니다." : kind === "evidence" ? action.rationale : "현재 후보·원래 근거·제출 자료를 대상으로 새 감사 이력을 만듭니다. 이전 결과는 보존되며, 미해결 과학 조건은 그대로 검사합니다.";
      if (kind === "automatic") $("#reviewActionIntro").textContent = "Europe PMC와 NCBI PMC의 가용 원문 경로, 집계 재계산, 규칙 재검사를 자동 수행합니다. 새 검토 자료가 있고 비용 사용에 동의한 경우 외부 재감사는 최대 1회입니다. 전체 최대 10분·기존 실행 예산 내에서 보류/진행 불가/검토 준비 결론으로 종료합니다. 자료 업로드나 후보 자동 승인은 하지 않습니다.";
      $("#reviewCandidateHint").textContent = request?.candidate_id ? `이번 결과의 candidate_id: ${request.candidate_id}` : "";
      $("#reviewBudgetText").textContent = `기록 비용 $${Number(status.recorded_cost_usd).toFixed(6)} / 실행 상한 $${Number(status.budget.max_run_cost_usd).toFixed(2)}. 재감사 직전에 예상 입력과 잔여 예산을 서버에서 다시 검사합니다.`;
      $("#submitReviewAction").textContent = kind === "sources" ? "공개 원문 조회" : ["rules", "external"].includes(kind) ? "재감사 실행" : "저장";
      if (kind === "automatic") $("#submitReviewAction").textContent = "자동 보완 시작";
      $("#reviewActionDialog").showModal();
    } catch (error) { showToast("후속 검토", error.message, "error"); }
    finally { button.disabled = false; }
  }

  function parseObservedCsv(text) {
    const rows = []; let row = [], field = "", quoted = false;
    text = text.replace(/^\uFEFF/, "");
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (c === '"') { if (quoted && text[i + 1] === '"') {field += '"'; i++;} else quoted = !quoted; }
      else if (c === ',' && !quoted) {row.push(field); field = "";}
      else if ((c === '\n' || c === '\r') && !quoted) { if (c === '\r' && text[i+1] === '\n') i++; row.push(field); if (row.some(x => x.trim())) rows.push(row); row = []; field = ""; }
      else field += c;
    }
    if (quoted) throw new Error("CSV 따옴표가 닫히지 않았습니다.");
    row.push(field); if (row.some(x => x.trim())) rows.push(row);
    const headers = (rows.shift() || []).map(s => s.trim());
    if (new Set(headers).size !== headers.length || OBSERVED_FIELDS.some(k => !headers.includes(k))) throw new Error("CSV에 필요한 열이 없거나 열 이름이 중복됐습니다. 빈 양식을 확인하십시오.");
    if (!rows.length || rows.length > 5000) throw new Error("실측 행은 1~5,000행이어야 합니다.");
    return rows.map((values, index) => {
      if (values.length !== headers.length) throw new Error(`CSV ${index+2}행의 열 수가 다릅니다.`);
      const result = Object.fromEntries(OBSERVED_FIELDS.map(k => [k, values[headers.indexOf(k)].trim()]));
      for (const key of ["dose", "measured_value", "uncertainty"]) {
        if (!result[key] || !Number.isFinite(Number(result[key]))) throw new Error(`CSV ${index+2}행 ${key}는 유한한 숫자여야 합니다.`);
        result[key] = Number(result[key]);
      }
      return result;
    });
  }

  function downloadObservedTemplate() {
    const url = URL.createObjectURL(new Blob(["\uFEFF" + OBSERVED_FIELDS.join(",") + "\r\n"], {type:"text/csv;charset=utf-8"}));
    const link = document.createElement("a"); link.href = url; link.download = "observed-results-template.csv"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function submitReviewAction(event) {
    event.preventDefault();
    const context = state.reviewAction;
    if (!context || context.runId !== state.snapshot?.run_id) return;
    const button = $("#submitReviewAction"), errorBox = $("#reviewActionError");
    button.disabled = true; errorBox.hidden = true;
    const researcher = $("#reviewResearcher").value.trim(), rationale = $("#reviewRationale").value.trim();
    try {
      if (!researcher || !rationale) throw new Error("연구자와 판단 근거를 입력하십시오.");
      const root = `/api/runs/${encodeURIComponent(context.runId)}`;
      if (context.kind === "note") {
        await api(`${root}/review-notes`, {method:"POST", body:JSON.stringify({researcher, note:rationale})});
      } else if (context.kind === "resolve") {
        const ids = (state.snapshot.action_evidence_submissions || []).filter(s => s.action_id === context.action.action_id).map(s => s.submission_id);
        await api(`${root}/actions/${encodeURIComponent(context.action.action_id)}/resolve`, {method:"POST", body:JSON.stringify({supporting_submission_ids:ids, resolved_by:researcher, rationale, completion_attested:true})});
      } else if (context.kind === "evidence") {
        const file = $("#reviewSourceFile").files[0];
        if (!file || !file.size || file.size > 5000000) throw new Error("0~5 MB 범위의 비어 있지 않은 원본 파일을 선택하십시오.");
        const bytes = new Uint8Array(await file.arrayBuffer());
        let binary = ""; for (let i=0;i<bytes.length;i+=8192) binary += String.fromCharCode(...bytes.subarray(i,i+8192));
        const experimental = context.action.kind === "experiment_required";
        let observed = [];
        if (experimental) {
          if (!$("#reviewObservedAttestation").checked) throw new Error("실측 자료 확인이 필요합니다.");
          const csv = $("#reviewObservedFile").files[0] || file;
          if (!csv.name.toLowerCase().endsWith(".csv") || csv.size > 5000000) throw new Error("구조화된 실험값 CSV를 선택하십시오.");
          if (!window.crypto?.subtle) throw new Error("실험값 해시 계산에는 localhost 또는 HTTPS 접속이 필요합니다.");
          const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(b => b.toString(16).padStart(2,"0")).join("");
          observed = parseObservedCsv(await csv.text()).map(row => ({...row, source_artifact_sha256:hash}));
        }
        await api(`${root}/actions/${encodeURIComponent(context.action.action_id)}/evidence`, {method:"POST", body:JSON.stringify({experiment_request_id:experimental ? context.request?.request_id : null, evidence_type:ACTION_EVIDENCE_TYPES[context.action.kind], submitted_by:researcher, notes:rationale, source_filename:file.name, media_type:file.type || "application/octet-stream", source_content_base64:btoa(binary), observed_results:observed, observed_not_synthetic_attestation:experimental})});
      } else {
        const paid = context.kind === "external" || context.kind === "automatic" && $("#reviewPaidConsent").checked;
        if (context.kind === "external" && !$("#reviewPaidConsent").checked) throw new Error("유료 모델 비용 사용 동의가 필요합니다.");
        const keyName = `drugpilot-reaudit:${context.runId}`;
        const pending = sessionStorage.getItem(keyName);
        if (pending) {
          await fetchRun(context.runId);
          const previous = JSON.parse(pending);
          if (!(state.snapshot.g1_reaudits || []).some(r => r.request_id === previous.requestId)) throw new Error("이전 재감사 요청 응답이 불명확합니다. 요청 기록을 확인하기 전에는 새 요청을 보내지 않습니다.");
          sessionStorage.removeItem(keyName);
          throw new Error("이전 요청의 처리 기록을 확인했습니다. 결과를 검토한 뒤 다시 시도하십시오.");
        }
        const requestId = `review-${crypto.randomUUID()}`;
        sessionStorage.setItem(keyName, JSON.stringify({requestId}));
        try {
          await api(`${root}/reaudits`, {method:"POST", headers:{"Idempotency-Key":requestId}, body:JSON.stringify({base_verdict_id:context.baseVerdictId, researcher, rationale, mode:context.kind, allow_paid:paid})});
        } catch (error) {
          if (error.status >= 400 && error.status < 500) sessionStorage.removeItem(keyName);
          throw error;
        }
        sessionStorage.removeItem(keyName);
      }
      $("#reviewActionDialog").close();
      await fetchRun(context.runId);
      if (reviewIsRunning()) startLiveUpdates(context.runId);
    } catch (error) { errorBox.textContent = error.message; errorBox.hidden = false; }
    finally { button.disabled = false; }
  }

  function configureHumanGate() {
    const snapshot = state.snapshot || {};
    const intake = getIntakeAssessment(snapshot);
    const verdict = snapshot.g1_verdict?.verdict;
    const packet = snapshot.candidate_packet || {};
    const actionPacket = snapshot.next_action_packet;
    const automaticResult = currentAutomaticResult(snapshot);
    const automaticLocked = (snapshot.g1_reaudits || []).some(r => r.mode === "automatic") && automaticResult?.outcome !== "READY_FOR_REVIEW";
    const actionMode = actionPacket?.human_gate_mode === "action_required_hold" || automaticLocked;
    const portfolio = portfolioExecutionState(packet);
    const auditPassed = verdict === "PASS";
    const portfolioReady = !portfolio.enabled || portfolio.complete;
    const available = actionMode
      ? snapshot.stage === "human_gate" && !snapshot.human_decision
      : auditPassed && portfolioReady && (snapshot.stage === "human_gate" || !snapshot.human_decision);
    $("#openHumanGate").disabled = !available || ["cancel_requested", "cancelled"].includes(snapshot.stage) || Boolean(snapshot.human_decision) || intake.requiresInputCorrection || reviewIsRunning();
    $("#openHumanGate").title = actionMode
      ? "미해결 조치를 검토하고 HOLD 또는 REJECT를 기록합니다. APPROVE는 잠겨 있습니다."
      : !auditPassed
      ? `G1 ${verdict || "PENDING"} · PASS 후 연구자 결정을 기록할 수 있습니다.`
      : !portfolioReady
      ? `MA2 ${portfolio.executionStatus.toUpperCase()} · 모든 실행 자격 표적이 완료되어야 합니다.`
      : "G1 감사 적합 후보에 대한 연구자 결정을 기록합니다.";
    const approveInput = $('input[name="decision"][value="APPROVE"]');
    const holdInput = $('input[name="decision"][value="HOLD"]');
    const approveLabel = approveInput?.closest("label");
    if (approveInput) approveInput.disabled = Boolean(actionMode);
    approveLabel?.classList.toggle("is-locked", Boolean(actionMode));
    $("#decisionTitle").textContent = actionMode ? "추가 조치 검토" : "연구자 최종 결정";
    $("#decisionIntro").textContent = actionMode
      ? `이 창의 보류 종료·반려는 검토를 종료하며 이후 자료 제출이 닫힙니다. 계속 보완하려면 다음 조치의 ‘임시 보류 메모’를 사용하십시오. ${auditPassed ? "G1 PASS는 완료됐으나 후속 조치·자동 검토 조건으로 기존 APPROVE는 잠겨 있습니다." : "기존 검토 승인은 G1 PASS와 후속 조건을 충족해야 가능합니다."} 연구후보 채택과 실험 검증은 별개입니다.`
      : "G1은 G0 승인 표적별 탐색 경로, 후보·실행과정·제외 정책과 표적 간 순위 무결성을 감사합니다. 연구자는 전체 표적의 실행 범위와 감사 기록을 확인한 뒤 후속 실험 후보를 선택합니다.";
    if (actionMode && approveInput?.checked && holdInput) holdInput.checked = true;
    if (actionMode) {
      const outstandingCount = actionPacket?.actions?.length ?? automaticResult?.remaining_findings?.length ?? 0;
      $("#decisionCandidates").innerHTML = `<div class="artifact-empty" style="min-height:55px">기존 APPROVE 잠금 · 미해결 항목 ${outstandingCount}건 · ${auditPassed ? "G1 PASS 완료 · 후속 조치·자동 검토 조건 미충족" : "G1 PASS 및 자동 보완 검토 준비 상태 필요"}</div>`;
      updateDecisionForm();
      return;
    }
    if (!auditPassed || !portfolioReady) {
      const reason = portfolio.enabled
        ? `${portfolioCoverageText(packet, { includeApproved: true })} · MA2 ${portfolio.executionStatus.toUpperCase()}`
        : `G1 ${verdict || "PENDING"}`;
      $("#decisionCandidates").innerHTML = `<div class="artifact-empty" style="min-height:55px">후보 선택 비활성 · ${esc(reason)}</div>`;
      return;
    }
    const candidates = sortCandidatesForReview(
      (packet.candidates || []).filter((candidate) => candidate.status !== "excluded"),
      packet,
    ).slice(0, 12);
    const coverageNote = portfolio.enabled
      ? `${portfolioCoverageText(packet, { includeApproved: true })}. G1 PASS가 확인된 전체 포트폴리오에서 후속 실험 후보를 선택하십시오.`
      : "G1 PASS가 확인된 후보입니다. 후속 실험 검토로 보낼 후보만 선택하십시오.";
    $("#decisionCandidates").innerHTML = candidates.length ? `<div class="decision-route-note">${esc(coverageNote)}</div>${candidates.map((candidate,index) => `
      <label class="decision-candidate"><input type="checkbox" name="selected_candidate_ids" value="${esc(candidate.candidate_id)}" ${index === 0 ? "checked" : ""}>
      <span><b>#${esc(portfolioCandidateRank(candidate, packet))} ${esc(candidate.name)}</b><small>${esc(candidate.target?.symbol || "—")} · ${esc(routeLabel(candidate.route))} · ${portfolio.enabled ? "portfolio" : "score"} ${candidateRankingScore(candidate, packet) == null ? "미기록" : formatScore(candidateRankingScore(candidate, packet))}${portfolio.enabled ? ` · target #${esc(candidate.target_rank ?? candidate.rank ?? "—")}` : ""}</small></span></label>`).join("")}` : '<div class="artifact-empty" style="min-height:55px">선택 가능한 후보 없음</div>';
  }

  function openDecisionDialog() {
    if ($("#openHumanGate").disabled) return;
    $("#decisionError").hidden = true;
    updateDecisionForm();
    $("#humanGateDialog").showModal();
  }

  function closeDecisionDialog() {
    if ($("#humanGateDialog").open) $("#humanGateDialog").close();
  }

  function updateDecisionForm() {
    const decision = $('input[name="decision"]:checked')?.value || "APPROVE";
    $("#candidateSelection").hidden = decision !== "APPROVE";
    $("#submitDecision").textContent = decision === "APPROVE" ? "승인 기록" : decision === "HOLD" ? "보류 종료 기록" : "반려 기록";
  }

  async function submitHumanDecision(event) {
    event.preventDefault();
    const snapshot = state.snapshot;
    if (!snapshot?.run_id || state.preview) return;
    const researcher = $("#researcherName").value.trim();
    const rationale = $("#decisionRationale").value.trim();
    const decision = $('input[name="decision"]:checked')?.value || "APPROVE";
    const selected = $$('input[name="selected_candidate_ids"]:checked').map((input) => input.value);
    if (!researcher || !rationale) {
      showDecisionError("연구자와 판단 근거를 모두 입력하십시오.");
      return;
    }
    if (decision === "APPROVE" && snapshot.next_action_packet?.approval_allowed === false) {
      showDecisionError("미해결 후속 조치가 있어 APPROVE가 잠겨 있습니다. HOLD 또는 REJECT를 선택하십시오.");
      return;
    }
    if (decision === "APPROVE" && !selected.length) {
      showDecisionError("승인할 후보를 한 건 이상 선택하십시오.");
      return;
    }
    const button = $("#submitDecision");
    button.disabled = true;
    try {
      const payload = {
        decision,
        researcher,
        rationale,
        selected_candidate_ids: decision === "APPROVE" ? selected : [],
      };
      const response = await api(`/api/runs/${encodeURIComponent(snapshot.run_id)}/human-decision`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const returned = unwrapSnapshot(response);
      if (returned?.brief) state.snapshot = returned;
      else await fetchRun(snapshot.run_id, { quiet: true });
      closeDecisionDialog();
      renderSnapshot();
      loadRuns({ quiet: true });
      showToast("연구자 결정을 기록했습니다", `${decisionLabel(decision)} · ${selected.length}개 후보`, "success");
    } catch (error) {
      showDecisionError(error.message);
    } finally {
      button.disabled = false;
    }
  }

  async function resumeRun() {
    const runId = state.snapshot?.run_id;
    if (!runId || state.preview) return;
    const button = $("#resumeButton");
    button.disabled = true;
    try {
      const status = await loadResumeStatus(runId);
      if (!status?.eligible || status?.active_in_process === true) {
        throw new Error(status?.reason || "재개 가능한 체크포인트가 없습니다.");
      }
      const response = await api(`/api/runs/${encodeURIComponent(runId)}/resume`, { method: "POST", body: "{}" });
      const returned = unwrapSnapshot(response);
      if (returned?.brief) state.snapshot = returned;
      else await fetchRun(runId, { quiet: true });
      state.resumeStatus = null;
      state.resumeStatusRunId = runId;
      renderSnapshot();
      startLiveUpdates(runId);
      showToast(
        "체크포인트에서 실행을 재개했습니다",
        `${status.checkpoint_completed_node || "Research Brief"} 완료 · ${status.resume_from_node || "다음 단계"}부터 실행`,
        "success",
      );
    } catch (error) {
      showToast("다시 실행 실패", error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  function showArchitecturePreview() {
    state.preview = true;
    state.eventIds.clear();
    state.openDisclosures.clear();
    state.callTrace = null;
    state.snapshot = {
      run_id: "preview",
      mode: "mock",
      stage: "created",
      status_message: "구조 미리보기 — Research Brief부터 Human Gate까지의 통제 흐름",
      brief: previewBrief(),
      events: [],
      token_usage: [],
      ma1_iterations: 0,
      rework_loops: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    renderSnapshot();
    $("#runStatusPill").className = "status-pill running";
    $("#runStatusPill").innerHTML = '<span class="status-dot"></span>구조 미리보기';
    $("#openHumanGate").disabled = true;
    $("#downloadRun").disabled = true;
    showToast("시스템 구조 미리보기", "각 단계를 누르면 관련 산출물 탭으로 이동합니다.");
  }

  function previewBrief() {
    const request = buildCreateRequest({ requireDirectionSupport: false });
    return {
      schema_version: "1.0",
      research_depth: request.research_depth,
      research_goal: request.research_goal,
      disease: request.disease,
      constraints: {
        target_cell_type: request.target_cell_type,
        intervention_direction: request.intervention_direction,
        ...(request.allowed_hypothesis_directions ? { allowed_hypothesis_directions: request.allowed_hypothesis_directions } : {}),
        chronic_safety_required: request.chronic_safety_required,
        druggability_required: request.druggability_required,
        forbidden_drugs: request.forbidden_drugs,
        forbidden_drug_ids: request.forbidden_drug_ids,
        forbidden_drug_classes: request.forbidden_drug_classes,
        candidates_per_route: request.candidates_per_route,
        max_ma1_iterations: request.max_ma1_iterations,
        max_rework_loops: request.max_rework_loops,
      },
      human_priorities: request.human_priorities,
      fault_injection: request.fault_injection,
      data_policy: { allowed_sources: [] },
      budget: { max_run_tokens: 1_000_000 },
    };
  }

  function setLoadingSnapshot(runId) {
    $("#researchResultTitle").textContent = "결과 불러오는 중";
    $("#researchResultSummary").textContent = "선택한 실행의 결과를 확인합니다.";
    $("#researchResultStates").innerHTML = "";
    $("#resultCandidateCards").innerHTML = '<p class="summary-placeholder">후보 기록을 불러오는 중입니다.</p>';
    $("#researchResultDetails").innerHTML = "";
    $("#resultCandidateCount").textContent = "0";
    $("#openResearchSelection").disabled = true;
    $("#runIdText").textContent = shortId(runId);
    $("#runDisease").textContent = "실행을 불러오는 중…";
    $("#runGoal").textContent = "저장된 스냅샷과 감사 원장을 조회하고 있습니다.";
    $("#runDepthPill").className = "research-depth-pill is-unrecorded";
    $("#runDepthPill").textContent = "깊이 불러오는 중";
  }

  function showRunWorkspace() {
    $("#welcomeState").hidden = true;
    $("#runWorkspace").hidden = false;
    $("#downloadRun").disabled = false;
  }

  function openSidebar() {
    $(".sidebar").classList.add("is-open");
    $("#sidebarScrim").hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeSidebar() {
    $(".sidebar").classList.remove("is-open");
    $("#sidebarScrim").hidden = true;
    document.body.style.overflow = "";
  }

  function downloadSnapshot() {
    if (!state.snapshot || state.preview) return;
    const content = JSON.stringify(state.snapshot, null, 2);
    const blob = new Blob([content], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `drugpilot-${state.snapshot.run_id}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  async function copyRunId() {
    const id = state.snapshot?.run_id;
    if (!id) return;
    try {
      await navigator.clipboard.writeText(id);
      showToast("실행 ID 복사", id, "success");
    } catch {
      showToast("복사할 실행 ID", id);
    }
  }

  function showFormError(message) {
    const box = $("#formError");
    box.hidden = !message;
    box.textContent = message || "";
  }

  function showDecisionError(message) {
    const box = $("#decisionError");
    box.hidden = !message;
    box.textContent = message || "";
  }

  function showToast(title, message = "", type = "info") {
    const region = $("#toastRegion");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.setAttribute("role", type === "error" ? "alert" : "status");
    toast.innerHTML = `<span class="toast-icon">${type === "error" ? "!" : type === "success" ? "✓" : "i"}</span><div><b>${esc(title)}</b>${message ? `<small>${esc(message)}</small>` : ""}</div><button type="button" aria-label="알림 닫기">×</button>`;
    $("button", toast).addEventListener("click", () => toast.remove());
    region.appendChild(toast);
    window.setTimeout(() => toast.remove(), type === "error" ? 6500 : 4200);
  }

  function indexEvents(events) {
    state.eventIds = new Set(events.map((event) => event.event_id).filter(Boolean));
  }

  function countEvidence(snapshot) {
    const hypothesisEvidence = sum(snapshot.hypothesis_packet?.hypotheses || [], (item) => item.evidence?.length || 0);
    const candidateEvidence = sum(snapshot.candidate_packet?.candidates || [], (item) => item.evidence?.length || 0);
    return hypothesisEvidence + candidateEvidence;
  }

  function countToolEvents(events) {
    return events.filter((event) => /tool/i.test(`${event.event_type || ""} ${event.node || ""}`)).length;
  }

  function countPublicDataCalls(snapshot) {
    const targetCalls = sum(
      snapshot?.hypothesis_packet?.metadata?.target_source_statuses || [],
      (source) => source.operations?.length || 0,
    );
    const contextCalls = Number(snapshot?.hypothesis_packet?.metadata?.context_routing?.source_calls_used || 0);
    const candidateCalls = (snapshot?.candidate_packet?.metadata?.public_enrichment_queries || [])
      .filter((query) => query.queried).length;
    return targetCalls + contextCalls + candidateCalls;
  }

  function eventStyle(event) {
    const text = `${event.event_type || ""} ${event.message || ""}`.toLowerCase();
    if (/abstain|source_unavailable|unresolved_disease|질환 확인 필요|질환 검증 불가/.test(text)) return { className: "warning", icon: "?" };
    if (/fail|error|block|critical/.test(text)) return { className: "error", icon: "!" };
    if (/audit|verdict|g0|g1/.test(text)) return { className: "audit", icon: "A" };
    if (/complete|pass|success|approved/.test(text)) return { className: "success", icon: "✓" };
    if (/warn|revise|replan|abstain|hold/.test(text)) return { className: "warning", icon: "↺" };
    return { className: "", icon: stageAbbreviation(event.stage) };
  }

  function eventTitle(event) {
    const raw = String(event.event_type || "status_update").replaceAll("_", " ");
    return `${stageAbbreviation(event.stage)} · ${raw.replace(/\b\w/g, (char) => char.toUpperCase())}`;
  }

  function stageAbbreviation(stage) {
    const map = { created: "IN", brief_validated: "RB", ma1: "M1", g0: "G0", planner: "PL", ma2: "M2", g1: "G1", human_gate: "HG", completed: "✓", blocked: "!", failed: "!", abstained: "?" };
    return map[String(stage || "").toLowerCase()] || "·";
  }

  function runStatusPresentation(snapshot = {}) {
    const stage = String(snapshot?.stage || "created").toLowerCase();
    if (stage === "cancel_requested") return { label: "중단 처리 중", detail: "진행 중 호출의 결과와 사용량을 기록한 뒤 종료합니다. 추가 단계는 시작하지 않습니다." };
    if (stage === "cancelled") return { label: "사용자 중단", detail: "사용자 요청으로 종료했습니다. 기존 근거·결과·사용량은 보존됩니다." };
    if (["failed", "blocked"].includes(stage)) return { label: stage === "failed" ? "분석 실패 · 부분 기록" : "분석 차단 · 부분 기록", detail: snapshot.status_message || "완료된 연구후보 추천이 아닙니다." };
    const terminalError = `${snapshot?.error || ""} ${snapshot?.status_message || ""}`;
    const inputCap = terminalError.match(/single input cap would be exceeded \(([\d,]+)\/([\d,]+)\)/i);
    if (stage === "abstained" && inputCap) {
      return { label: "단일 모델 입력 한도 초과", detail: `보수적 사전 추산 입력량 ${inputCap[1]} / 한도 ${inputCap[2]} 토큰으로 다음 모델 호출 전에 중단했습니다. 이 수치는 실제 과금 토큰이 아닙니다. 기존 G0 판단과 별개인 입력 크기 제한이며, 입력 구성·예산·무결성을 확인해야 합니다.` };
    }
    if (stage === "abstained" && /(?:run|hard) (?:token|cost) cap|live LLM usage is estimated|no approved price/i.test(terminalError)) {
      return {label: "실행 예산·사용량 검증으로 중단", detail: `${snapshot?.status_message || snapshot?.error}. 기존 감사 판단과 별개로 실행 예산 또는 사용량 검증이 중단 원인입니다.`};
    }
    if (snapshot?.human_decision?.decision === "HOLD") {
      return { label: "연구자 판단 보류", detail: "자동 감사 이후 연구자가 추가 검토를 위해 결정을 보류했습니다." };
    }
    const automatic = currentAutomaticResult(snapshot);
    if (automatic && !snapshot.human_decision && ["human_gate", "completed"].includes(stage)) return {label:`자동 분석 완료 · ${({HOLD:"보류",BLOCKED:"진행 불가",READY_FOR_REVIEW:"검토 준비"})[automatic.outcome] || automatic.outcome}`,detail:snapshot.status_message};
    if (g0StoppedBeforeMa2(snapshot)) {
      const used = Number(snapshot?.rework_loops || 0);
      const max = Number(snapshot?.brief?.constraints?.max_rework_loops ?? 0);
      return {
        label: "G0 근거 보강 후 연구자 검토",
        detail: `G0 ${verdictLabel(snapshot?.g0_verdict?.verdict)} · 재작업 ${compactNumber(used)}/${compactNumber(max)}회 후 Planner·MA2·G1은 실행하지 않았습니다. Human Gate는 보완 조치 검토용이며 후보 승인은 잠겨 있습니다.`,
      };
    }
    if (stage !== "abstained") return { label: stageLabel(stage), detail: snapshot?.status_message || `${stageLabel(stage)} 단계` };
    const intake = getIntakeAssessment(snapshot);
    if (intake.requiresInputCorrection) return { label: "질환·연구 입력 확인 필요", detail: "질환 식별 또는 입력 필드가 확정되지 않아 후보 생성 전에 중단했습니다." };
    if (intake.verificationUnavailable) return { label: "외부 질환 검증 불가로 중단", detail: "질환 검증 소스에 접근하지 못해 근거 없는 실행을 방지하고 중단했습니다." };
    const budgetText = `${snapshot?.error || ""} ${snapshot?.status_message || ""}`;
    const callCap = budgetText.match(/LLM call cap would be exceeded \((\d+)\/(\d+)\)/i);
    if (callCap) {
      const requestedCall = Number(callCap[1]);
      const maxCalls = Number(callCap[2]);
      const recordedCalls = Array.isArray(snapshot?.token_usage)
        ? snapshot.token_usage.length
        : Number.isFinite(Number(snapshot?.token_usage_count))
        ? Number(snapshot.token_usage_count)
        : Math.max(0, requestedCall - 1);
      return {
        label: "LLM 호출 한도 도달",
        detail: `이 실행은 LLM ${compactNumber(recordedCalls)}/${compactNumber(maxCalls)}회를 사용했고 ${compactNumber(requestedCall)}번째 호출 전에 차단됐습니다. G0 최종 재감사를 완료하지 못해 Planner·MA2·G1에는 진입하지 않았습니다. 실행 예산을 조정한 뒤 새 실행이 필요합니다.`,
      };
    }
    const g1 = String(snapshot?.g1_verdict?.verdict || "").toUpperCase();
    if (["ABSTAIN", "BLOCK", "REVISE", "REPLAN"].includes(g1)) {
      return { label: "G1 후보·과정 감사 미충족으로 중단", detail: `${gateVerdictSummaryKo(snapshot.g1_verdict)}. 연구자 결정 단계로 진행하지 않았습니다.` };
    }
    const g0 = String(snapshot?.g0_verdict?.verdict || "").toUpperCase();
    if (["ABSTAIN", "BLOCK", "REVISE", "REPLAN"].includes(g0)) {
      const used = Number(snapshot?.rework_loops || 0);
      const max = Number(snapshot?.brief?.constraints?.max_rework_loops ?? 0);
      const reached = g0 === "REVISE" && used >= max;
      const prefix = reached ? `재작업 ${compactNumber(used)}/${compactNumber(max)}회 한도 소진` : `G0 ${verdictLabel(g0)}`;
      return {
        label: reached ? "G0 근거 보강 한도 도달" : "G0 가설 감사 미충족으로 중단",
        detail: `${gateVerdictSummaryKo(snapshot.g0_verdict)}. ${prefix}으로 Planner·MA2·G1은 실행하지 않았습니다. 이는 서버 오류가 아니라 근거 품질 기준에 따른 종료입니다.`,
      };
    }
    return { label: "근거 부족으로 중단", detail: snapshot?.status_message || "감사 기준을 충족할 근거가 부족해 다음 단계로 진행하지 않았습니다." };
  }

  function stageLabel(stage) {
    const map = {
      cancel_requested: "중단 처리 중",
      cancelled: "사용자 중단",
      created: "생성됨",
      brief_validated: "연구 입력 검증",
      ma1: "MA1 가설 발굴",
      g0: "G0 가설 감사",
      planner: "실행 계획",
      ma2: "MA2 후보 발굴",
      g1: "G1 후보·과정 감사",
      human_gate: "연구자 결정",
      completed: "완료",
      abstained: "근거 부족으로 중단",
      blocked: "차단",
      failed: "실패",
    };
    return map[String(stage || "").toLowerCase()] || String(stage || "—");
  }

  function verdictLabel(verdict) {
    const map = { PASS: "감사 적합", REVISE: "근거 보강 요청", REPLAN: "재계획", ABSTAIN: "근거 부족", BLOCK: "차단" };
    return map[String(verdict || "").toUpperCase()] || String(verdict || "대기");
  }

  function decisionLabel(decision) {
    return ({ APPROVE: "승인", HOLD: "보류", REJECT: "반려" })[decision] || decision;
  }

  function directionLabel(direction) {
    return ({ inhibit: "억제 (Inhibit)", activate: "활성화 (Activate)", modulate: "조절 (Modulate)", unspecified: "미지정 · 근거 기반 탐색" })[direction] || direction || "—";
  }

  function discoveryPriorityLabel(priority) {
    return ({
      auto: "자동 · MA1 최상위 표적",
      novel_target: "신규 표적 우선 · 근거 충족 표적",
      repurposing_target: "재창출 표적 우선 · 강한 약물 근거 표적",
    })[priority] || priority || "자동 · MA1 최상위 표적";
  }

  function discoveryPriorityDerivationLabel(value) {
    return ({
      research_brief_structured_constraint: "구조화 입력 우선",
      explicit_goal_repurposing_cue_when_auto: "연구목표의 재창출 단서 반영",
      explicit_goal_novel_target_cue_when_auto: "연구목표의 신규 표적 단서 반영",
      auto_without_explicit_strategy_cue: "자동 기본값",
      conflicting_goal_strategy_cues_keep_auto: "상충 단서로 자동 유지",
    })[value] || String(value || "");
  }

  function goalCueCategoryLabel(value) {
    return ({
      single_cell: "단일세포",
      expression: "발현",
      network: "상호작용 네트워크",
      pharos: "표적 개발 수준",
      repurposing: "약물 재창출",
      novel_target: "신규·미개척 표적",
      safety: "안전성·독성",
      explicit_target: "명시 표적",
    })[value] || "";
  }

  function hypothesisSelectionReasonLabel(reason) {
    return ({
      MA1_TOP_RANKED_AUTO: "MA1 순위와 점수가 가장 높은 가설을 선택",
      EXPLICIT_EVIDENCE_BACKED_TARGET_FOCUS: "연구목표에 명시된 표적 중 MA1 근거가 확인된 가설을 우선 선택",
      EVIDENCE_QUALIFIED_UNDEREXPLORED_HYPOTHESIS: "생물학적 지지와 근거 완결성을 충족하고 강한 재창출 근거가 없는 가설을 선택",
      EVIDENCE_QUALIFIED_REPURPOSING_HYPOTHESIS: "강한 질환·표적 약물 근거와 근거 완결성을 충족한 가설을 선택",
      NO_EVIDENCE_QUALIFIED_PRIORITY_MATCH_FALLBACK_AUTO: "요청한 우선순위의 근거 자격 가설이 없어 MA1 자동 순위로 복귀",
    })[reason] || reason || "저장된 세부 선택 사유 없음";
  }

  function routeLabel(route) {
    return ({ repurposing: "Drug Repurposing", de_novo: "De novo Design" })[route] || route || "—";
  }

  function statusLabel(status) {
    return ({ priority: "우선 검토", conditional: "추가 검토", excluded: "제외" })[status] || status || "—";
  }

  function faultLabel(fault) {
    const map = {
      none: "없음 · 정상 경로",
      g0_missing_evidence: "G0 근거 누락",
      g1_ic50_unit: "G1 IC50 단위 오류",
      g1_invalid_smiles: "G1 잘못된 SMILES",
      g1_direction_mismatch: "G1 작용 방향 불일치",
      g1_plan_trace_mismatch: "G1 계획-실행 추적 불일치",
    };
    return map[fault] || fault || "없음";
  }

  function relativeTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    const seconds = Math.round((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return "방금 전";
    if (seconds < 3600) return `${Math.floor(seconds / 60)}분 전`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}시간 전`;
    return `${Math.floor(seconds / 86400)}일 전`;
  }

  function formatTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(date);
  }

  function compactNumber(value) {
    const number = Number(value || 0);
    if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(number >= 10_000_000 ? 0 : 2)}M`;
    if (number >= 1_000) return `${(number / 1_000).toFixed(number >= 100_000 ? 0 : 1)}K`;
    return number.toLocaleString("ko-KR");
  }

  function formatKrw(value) {
    return `₩${Math.round(Number(value || 0)).toLocaleString("ko-KR")}`;
  }

  function formatScore(value) {
    const number = Number(value || 0);
    return number.toFixed(2);
  }

  function scorePercent(value) {
    return clamp(Math.round(Number(value || 0) * 100), 0, 100);
  }

  function shortId(value) {
    const text = String(value || "—");
    return text.length > 16 ? `${text.slice(0, 8)}…${text.slice(-4)}` : text;
  }

  function sum(items, getter) {
    return (items || []).reduce((total, item) => total + Number(getter(item) || 0), 0);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, Number(value)));
  }

  function esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

})();
