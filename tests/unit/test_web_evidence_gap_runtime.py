"""Execute the shipped renderer, without browser calls or a synthetic live run."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def render(gap):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for renderer runtime check")
    script = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")
    renderer = script.split("  function evidenceGapHtml(gap) {", 1)[1].split("\n  function semanticFailureHtml", 1)[0]
    source = """
const fs = require('fs');
const esc = value => String(value ?? '').replace(/[&<>\"']/g,
  char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '\"':'&quot;', "'":'&#39;'}[char]));
""" + "function evidenceGapHtml(gap) {" + renderer + "\nprocess.stdout.write(evidenceGapHtml(JSON.parse(fs.readFileSync(0, 'utf8'))));"
    result = subprocess.run([node, "-e", source], input=json.dumps(gap, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=10, check=True)
    return result.stdout


@pytest.mark.parametrize("gap", [None, {}, {"policy":"legacy"}])
def test_legacy_diagnostic_is_unknown_not_zero_or_absence(gap):
    html = render(gap)
    assert "진단 미기록" in html and "단정할 수 없습니다" in html
    assert "0건" not in html


def test_real_gap_counts_and_scope_are_rendered():
    html = render({"policy":"observed_evidence_gap_v1", "outcome":"screened_without_qualified_excerpt",
        "search_result_count":111, "records_screened":111, "unsearched_result_count":0,
        "exclusive_record_buckets":{"abstract_not_available":2,"abstract_received_excerpt_not_selected":20},
        "next_actions":["try_other_unattempted_allowed_path_then_hold"]})
    for expected in ("검색 111건", "심사 111건", "검색 창 밖 0건", "초록 미확보 2건",
            "발췌 미선택(추출 오류 확정 아님) 20건", "미해소 시 보류", "과학적 탈락을 증명하지 않습니다"):
        assert expected in html


def test_assay_context_is_not_presented_as_established_potency():
    html=render({"policy":"observed_evidence_gap_v1","outcome":"records_returned_not_gate_pass",
        "exclusive_record_buckets":{"assay_context_unassigned_measurement":5},
        "matched_context_documents":10,"retained_context_documents":5,"omitted_matching_context_documents":5,
        "next_actions":["preserve_assay_context_without_assigning_potency"]})
    assert "수치와 약물·표적의 연결 미검증" in html
    assert "활성값·효능으로 자동 배정하지 않음" in html
    assert "문맥 발견 10건 · 보존 5건 · 보존 예산으로 미포함 5건(과학적 탈락 아님)" in html


def test_identity_inventory_is_not_presented_as_new_candidates_or_efficacy():
    html=render({"policy":"observed_evidence_gap_v1","outcome":"assay_identity_inventory_completed",
        "identity_document_count":3,"identity_activity_rows_screened":408,"identity_linked_activity_count":50})
    assert "ChEMBL 문헌 3건 · 등록 측정행 조회 408건 · 표적 ID 연결 50건" in html
    assert "효능 판정 아님" in html and "미기록 조건은 추정하지 않습니다" in html


def test_applicability_summary_is_not_construct_approval():
    html=render({"policy":"observed_evidence_gap_v1","identity_annotation_counts":{
        "not_recorded":49,"unresolved_annotation":13,"named_annotation_only":12,
        "class_annotation_only":0,"annotation_conflict":0}})
    assert "변이 미기록 49건 / 미확인 13건" in html and "변이명 표기 12건" in html
    assert "UNDEFINED MUTATION은 확인된 변이가 아닙니다" in html
    assert "<details><summary>실험 적용 범위" in html


def test_methods_summary_keeps_access_failures_separate_from_biological_absence():
    html=render({"policy":"observed_evidence_gap_v1","outcome":"assay_fulltext_context_completed",
        "assay_fulltext_counts":{"parsed":4,"oa_not_admitted":9,"metadata_not_found":0,
            "budget_not_completed":1,"retrieval_or_validation_error":2},"assay_method_paragraphs":12,"assay_table_locators":3})
    assert "공개 원문 검증 4건 · 실험 방법 문단 12건" in html
    assert "OA·라이선스 조건 미충족 9건" in html and "예산으로 미완료 1건" in html
    assert "원문 접근 불가는 생물학적 근거 부재가 아닙니다" in html
    assert "표의 수치 배정은 미검증" in html


@pytest.mark.parametrize('value',[None,[],"<script>alert(1)</script>",{"parsed":"<img>"}])
def test_methods_summary_rejects_malformed_injected_fields(value):
    html=render({"policy":"observed_evidence_gap_v1","assay_fulltext_counts":value})
    assert "<script>" not in html and "<img>" not in html


@pytest.mark.parametrize('value',[None,[],"<img onerror=alert(1)>",{"not_recorded":False,"unresolved_annotation":"<img>"}])
def test_applicability_malformed_counts_never_execute_html(value):
    html=render({"policy":"observed_evidence_gap_v1","identity_annotation_counts":value})
    assert "<img" not in html and "false건" not in html


@pytest.mark.parametrize("actions", [{"bad":"type"}, [None, {}, "toString", "<img onerror=alert(1)>"], None])
def test_malformed_or_injected_diagnostic_is_safe(actions):
    html = render({"policy":"observed_evidence_gap_v1", "outcome":"toString", "search_result_count":False,
        "records_screened":-1, "unsearched_result_count":"0", "next_actions":actions,
        "exclusive_record_buckets":{"toString":4,"<img onerror=alert(1)>":6,"abstract_not_available":False}})
    assert "진단 기록 부족" in html
    assert "검색 미기록" in html and "심사 미기록" in html
    assert "초록 미확보 미기록" in html
    assert "[native code]" not in html and "<img" not in html
