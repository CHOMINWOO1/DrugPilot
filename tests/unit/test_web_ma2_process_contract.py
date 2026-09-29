from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _dashboard() -> str:
    return (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")


def _page() -> str:
    return (ROOT / "src/drugpilot/web/templates/index.html").read_text(
        encoding="utf-8"
    )


def test_variant_labels_do_not_present_legacy_or_class_match_as_exact():
    dashboard = _dashboard()
    assert 'exact_match: "이전 판정: 정확 일치 · 재검증 필요"' in dashboard
    assert 'variant_class_match_only: "변이 유형만 일치 · 정확한 구성체 미확인"' in dashboard
    assert 'variant_annotation_match: "변이명 일치 · 정확한 구성체 미확인"' in dashboard
    assert "PROSPECTIVE_FALSIFICATION_ATTESTATION_INVALID" in dashboard


def test_dashboard_exposes_recorded_ma2_process_without_claiming_missing_checks_passed() -> None:
    dashboard = _dashboard()

    assert "TARGET-ADAPTIVE ROUTE POLICY" in dashboard
    assert "MA2 단계별 실행 기록" in dashboard
    assert "CandidatePacket의 tool trace·후보 지표·감사 산출물" in dashboard
    assert "도구 기록이 없는 검사는 통과로 간주하지 않습니다" in dashboard
    assert "route_decision" in dashboard
    assert "exclusion_summary" in dashboard


def test_dashboard_exposes_candidate_discovery_funnel_and_each_recorded_rejection() -> None:
    dashboard = _dashboard()

    assert "후보 검색 퍼널·탈락 사유" in dashboard
    assert "discovered_identity_count" in dashboard
    assert "eligible_identity_count" in dashboard
    assert "fixed_pool_with_activity_count" in dashboard
    assert "fixed_pool_candidate_ids" in dashboard
    assert "candidate_primary_dispositions" in dashboard
    assert "preactivity_exclusion_reason_codes_by_id" in dashboard
    assert "고정 풀 검증 내역" in dashboard
    assert "고정 풀 이전 사전 제외" in dashboard
    assert "외부 오류와 근거 부족은 과학적 음성 결과로 바꾸지 않습니다" in dashboard


def test_route_decision_copy_distinguishes_source_records_from_eligible_drugs() -> None:
    dashboard = _dashboard()

    assert "적합 기존 약물" not in dashboard
    assert "DGIdb 상호작용 레코드" in dashboard
    assert "임상시험 약물명" in dashboard
    assert "중복 제거" in dashboard
    assert "고유한 적합 후보 수, 임상 효능 또는 승인 가능성을 뜻하지 않습니다" in dashboard
    assert "경로 규칙 충족도" in dashboard
    assert "성공확률 아님" in dashboard
    assert "모든 G0 승인 표적" in dashboard
    assert "target_executions" in dashboard
    assert "표적별 경로 판정" in dashboard
    assert "1개에만 경로 정책을 적용했습니다" not in dashboard
    assert "표적-약물 연계 신호가 확인됨" in dashboard
    assert "기존 약물·개발 근거가 충분함" not in dashboard


def test_completed_zero_hit_target_is_distinct_from_tool_unavailability() -> None:
    dashboard = _dashboard()

    assert 'item.status || "").toLowerCase() === "no_candidates"' in dashboard
    assert "검색 완료 · 적합 후보 없음" in dashboard
    assert "concluded === eligible" in dashboard
    assert '"partial", "unavailable", "failed"' in dashboard


def test_discovery_priority_is_wired_without_claiming_to_force_a_route() -> None:
    page = _page()
    dashboard = _dashboard()

    assert 'id="discoveryPriority"' in page
    for value in ("auto", "novel_target", "repurposing_target"):
        assert f'value="{value}"' in page
    assert "경로를 강제로 지정하지 않습니다" in page
    assert 'discovery_priority: $("#discoveryPriority").value' in dashboard
    assert "primary_hypothesis_selection" in dashboard
    assert "requested_priority" in dashboard
    assert "selected_target_symbol" in dashboard
    assert "fallback_to_auto" in dashboard
    assert "미실행 가설" in dashboard
    assert "경로를 강제하지 않고 자격을 충족한 가설의 선택 순서만 조정합니다" in dashboard
    assert "요청 조건에 맞는 근거 자격 표적이 없어 자동 우선순위로 복귀했습니다" in dashboard


def test_dashboard_discloses_goal_cues_and_effective_priority() -> None:
    dashboard = _dashboard()

    assert "요청 우선순위" in dashboard
    assert "실제 적용 우선순위" in dashboard
    assert "목표 해석" in dashboard
    assert "discovery_priority_derivation" in dashboard
    assert "연구목표의 재창출 단서 반영" in dashboard
    assert "연구목표의 신규 표적 단서 반영" in dashboard
    assert "상호작용 네트워크" in dashboard
    assert "단일세포" in dashboard


def test_forbidden_drug_screen_is_shown_before_expensive_candidate_computation() -> None:
    dashboard = _dashboard()
    start = dashboard.index("function candidateProcessHtml")
    end = dashboard.index("function processStepHtml", start)
    process_view = dashboard[start:end]

    screen = process_view.index("입력 금지목록·기록된 규제상태 사전 스크리닝")
    chemistry = process_view.index("화학 구조 표준화·유효성")
    docking = process_view.index("표적 결합 도킹")
    admet = process_view.index("6종 ADMET·독성 패널")
    refinement = process_view.index("필요 시 구조 정제·재계산")
    ranking = process_view.index("다기준 순위화")
    enrichment = process_view.index("순위화 후 공개 DB 근거 보강")
    assert screen < chemistry < docking < admet < refinement < ranking < enrichment
    assert "도킹·ADMET 전 적용" in process_view
    assert "사전 차단으로 도킹 비용 미사용" in process_view


def test_dashboard_lists_all_six_persisted_admet_outputs_and_ranking_factors() -> None:
    dashboard = _dashboard()

    for endpoint in (
        "hepatotoxicity",
        "herg_risk",
        "ames_risk",
        "cyp3a4_inhibition",
        "clearance_risk",
        "oral_bioavailability",
    ):
        assert endpoint in dashboard
    for factor in (
        "약리 활성 근거",
        "결합 적합성",
        "예측 ADMET 위험",
        "개발가능성",
        "근거 충실도",
        "예측·안전성 경고 페널티",
    ):
        assert factor in dashboard


def test_research_brief_accepts_explicit_repurposing_exclusion_inputs() -> None:
    page = _page()
    dashboard = _dashboard()

    for element_id, payload_key in (
        ("forbiddenDrugs", "forbidden_drugs"),
        ("forbiddenDrugIds", "forbidden_drug_ids"),
        ("forbiddenDrugClasses", "forbidden_drug_classes"),
    ):
        assert f'id="{element_id}"' in page
        assert f'{payload_key}: commaSeparatedValues($("#{element_id}").value)' in dashboard

    assert "표적별 경로 후보 수" in page
    assert "G0를 통과하고 식별자가 확인된 각 표적" in page
    assert "Evidence-routed single path" not in page
    assert "경로별 후보 수" not in page
    assert "De novo + Repurposing" not in dashboard
