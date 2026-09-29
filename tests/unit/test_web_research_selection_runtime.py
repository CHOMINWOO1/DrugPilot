"""Read-only result projection and explicit research selection UI contracts."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")
PAGE = (ROOT / "src/drugpilot/web/templates/index.html").read_text(encoding="utf-8")


def function(name):
    start = re.search(r"^  (?:async )?function " + re.escape(name) + r"\(", SCRIPT, re.M).start()
    following = re.search(r"^  (?:async )?function ", SCRIPT[start + 3:], re.M)
    return SCRIPT[start:start + 3 + following.start()] if following else SCRIPT[start:]


def execute(body):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    completed = subprocess.run([node, "-e", body], capture_output=True, text=True, encoding="utf-8", timeout=15, check=True)
    return json.loads(completed.stdout)


def test_result_and_candidates_precede_collapsed_process_and_next_actions():
    assert PAGE.index('id="researchResultPanel"') < PAGE.index('id="pipelineSection"') < PAGE.index('id="nextActionPanel"')
    assert '<details id="pipelineSection"' in PAGE
    assert '<details id="nextActionPanel"' in PAGE
    assert '<details class="run-goal-details">' in PAGE
    assert 'id="modeBadgeText">LLM · 확인 중' in PAGE
    assert 'id="dataBadgeText">DATA · 확인 중' in PAGE
    assert "20260929-ma3-admission-v1" in PAGE
    assert "실험 검증 완료를 뜻하지 않습니다" in PAGE


@pytest.mark.parametrize("stage", ["ma2", "g1", "failed", "blocked", "cancelled", "cancel_requested", "abstained"])
def test_running_and_failed_states_never_inherit_old_analysis_complete(stage):
    result = execute("const runStatusPresentation=s=>({label:s.stage});\n" + function("resultPresentation")
        + "\nconsole.log(JSON.stringify(resultPresentation(" + json.dumps({"stage": stage})
        + ",{analysis_complete:true,ledger_valid:true,research_selection:{allowed:true,state:'selected'}})));" )
    assert not result["complete"]
    assert "분석 완료" not in result["title"]


@pytest.mark.parametrize("selection,valid,expected", [
    ({"allowed": True, "state": "not_selected"}, True, "조건부 연구후보"),
    ({"allowed": False, "state": "not_selected"}, True, "채택 불가"),
    ({"allowed": True, "state": "not_selected"}, False, "채택 불가"),
    ({"allowed": False, "state": "stale"}, True, "재검토 필요"),
    ({"allowed": False, "state": "selected"}, True, "채택 기록됨"),
])
def test_analysis_completion_is_separate_from_selection_and_experiment(selection, valid, expected):
    readiness = {"analysis_complete": True, "ledger_valid": valid, "research_selection": selection}
    result = execute(function("resultPresentation") + "\nconsole.log(JSON.stringify(resultPresentation({stage:'human_gate'},"
        + json.dumps(readiness) + ")));" )
    assert result["complete"] and expected in result["title"]
    assert "실험 검증 완료" not in result["title"]


def test_selection_requires_backend_authority_and_packet_binding():
    result = execute("const state={preview:false};\n" + function("resultPresentation") + function("researchSelectionAllowed") + """
const snapshot={stage:'human_gate',g1_verdict:{verdict_id:'g1'}};
const ready={analysis_complete:true,ledger_valid:true,research_selection:{allowed:true,policy_version:'research-selection-v1',expected_g1_verdict_id:'g1',expected_candidate_packet_sha256:'a'.repeat(64),eligible_candidate_ids:['candidate_a']}};
const clone=()=>JSON.parse(JSON.stringify(ready));
const cases=[ready];
for(const change of [s=>s.allowed=false,s=>s.expected_g1_verdict_id='old',s=>delete s.expected_candidate_packet_sha256,s=>s.eligible_candidate_ids=[]]) {const r=clone(); change(r.research_selection); cases.push(r);}
console.log(JSON.stringify(cases.map(r=>researchSelectionAllowed(snapshot,r))));
""")
    assert result == [True, False, False, False, False]


def test_open_dialog_never_preselects_candidates_or_acknowledgement():
    code = function("openResearchSelectionDialog")
    assert '.reset()' in code
    assert 'type="checkbox" name="research_candidate_ids"' in code
    assert 'checked' not in code
    assert 'id="selectionAcknowledgement" type="checkbox" required' in PAGE
    assert 'packetSha256: selection.expected_candidate_packet_sha256' in code


@pytest.mark.parametrize("ambiguous", [False, True])
def test_selection_posts_once_with_frozen_guards_and_reconciles_get_only(ambiguous):
    setup = """
const sha='a'.repeat(64); const calls=[]; const nodes={};
const $=id=>nodes[id]||(nodes[id]={value:id==='#selectionResearcher'?'Researcher':'Reason',checked:true,hidden:true,disabled:false,close:()=>{}});
const $$=()=>[{value:'candidate_a'}];
const state={snapshot:{run_id:'run_a',stage:'human_gate',g1_verdict:{verdict_id:'g1'}},preview:false,researchSelectionSubmitting:false,researchSelectionContext:{runId:'run_a',verdictId:'g1',packetSha256:sha,policyVersion:'research-selection-v1',eligibleIds:['candidate_a']}};
const unwrapSnapshot=x=>x,renderSnapshot=()=>{},renderResearchResults=()=>{},showToast=()=>{};
const fetchRun=async()=>{};
const loadResultReadiness=async()=>{calls.push('GET');return {research_selection:{state:'selected',record:{researcher:'Researcher',rationale:'Reason',selected_candidate_ids:['candidate_a'],expected_g1_verdict_id:'g1',candidate_packet_sha256:sha,policy_version:'research-selection-v1'}}};};
const api=async(path,options)=>{calls.push({path,method:options.method,payload:JSON.parse(options.body)});
""" + ("throw new Error('response lost');" if ambiguous else "return {run_id:'run_a',brief:{},g1_verdict:{verdict_id:'g1'}};") + "};\n"
    result = execute(setup + function("submitResearchSelection") + "\n(async()=>{await submitResearchSelection({preventDefault(){}}); console.log(JSON.stringify({calls,busy:state.researchSelectionSubmitting}));})();")
    posts = [call for call in result["calls"] if isinstance(call, dict)]
    assert len(posts) == 1
    assert posts[0]["path"].endswith("/research-selection")
    assert posts[0]["payload"]["expected_g1_verdict_id"] == "g1"
    assert posts[0]["payload"]["expected_candidate_packet_sha256"] == "a" * 64
    assert posts[0]["payload"]["limitations_acknowledged"] is True
    assert result["calls"][-1] == "GET"
    assert not result["busy"]


def test_readiness_get_is_cached_for_unchanged_snapshot():
    result = execute("""
const state={snapshot:{run_id:'run_a',stage:'human_gate',updated_at:'now'},preview:false,resultReadiness:null,resultReadinessKey:null,resultReadinessRequests:new Map()};
let calls=0; const api=async()=>{calls++; return {run_id:'run_a'};};
const renderResearchResults=()=>{},updateRunStatus=()=>{},renderRunList=()=>{};
""" + function("resultReadinessKey") + function("currentResultReadiness") + function("loadResultReadiness") + """
(async()=>{await Promise.all([loadResultReadiness(),loadResultReadiness()]);await loadResultReadiness();const before=calls;state.snapshot.updated_at='later';await loadResultReadiness();console.log(JSON.stringify({before,calls}));})();
""")
    assert result == {"before": 1, "calls": 2}


def test_compact_cards_escape_source_text_and_do_not_dump_smiles():
    result = execute("""
const state={snapshot:{}}; const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const researchSelectionAllowed=()=>false,candidateRankingScore=()=>null,candidateStructureUrl=()=>null,candidateIdentityEntries=()=>[],portfolioCandidateRank=()=>1,routeLabel=x=>x;
""" + function("researchLimitationLabel") + function("resultCandidateCardHtml") + """
console.log(JSON.stringify(resultCandidateCardHtml({candidate_id:'x<script>',name:'<script>',target:{symbol:'<svg>'},smiles:'DO_NOT_RENDER_FULL_SMILES',route:'<img>',safety_flags:['<script>']},{},{})));
""")
    assert "<script>" not in result and "<svg>" not in result
    assert "&lt;script&gt;" in result
    assert "DO_NOT_RENDER_FULL_SMILES" not in result
    assert "남은 한계" in result


def test_original_status_and_unverified_experiment_are_always_explicit():
    code = function("renderResearchResults")
    assert '"약효·안전성 실험", "미검증 · 실험 검증 대기"' in code
    assert 'esc(snapshot.status_message' in code
    assert 'esc(record.rationale)' in code
    assert 'selection.blocking_reasons.map(esc)' in code
    assert 'candidates.slice(0, 3)' in code


def test_known_limitations_are_concise_and_safety_is_prioritized():
    result = execute(function("researchLimitationLabel") + "\nconsole.log(JSON.stringify(['ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION','ASSAY_VARIANT_PARTIAL_OVERLAP','COUNTER_EVIDENCE_NOT_FOUND'].map(code=>researchLimitationLabel({code,message:'long '.repeat(100)}))));")
    assert result == ["예측 ADMET 위험 신호 · 실험 확인 필요", "정확한 변이·실험 조건 확인 필요", "반증 검색 완료 · 직접 반증 근거 미확보"]
    cards = function("resultCandidateCardHtml")
    assert 'startsWith("ADMET_")' in cards
    assert 'rankedLimitations.map(researchLimitationLabel)' in cards
    details = function("renderResearchResults")
    assert '남은 한계 ${limitations.length}건 · 외부 검토 참고 ${advisories.length}건' in details
    assert '근거 미확보가 효능의 증거는 아닙니다' in details


def test_ambiguous_write_receipt_is_bound_to_reviewed_packet_and_verdict():
    code = function("submitResearchSelection")
    assert 'saved?.expected_g1_verdict_id === context.verdictId' in code
    assert 'saved?.candidate_packet_sha256 === context.packetSha256' in code
    assert 'saved?.policy_version === context.policyVersion' in code


def test_sidebar_uses_current_verified_projection_without_fetching_history():
    result = execute("""
const state={snapshot:{run_id:'current'}};
const currentResultReadiness=()=>({ledger_valid:true});
const resultPresentation=()=>({title:'분석 완료 · 조건부 연구후보'});
""" + function("runListResultLabel") + """
console.log(JSON.stringify([runListResultLabel({run_id:'current',stage:'human_gate'},'자동 분석 완료 · 보류'),runListResultLabel({run_id:'other',stage:'human_gate'},'자동 분석 완료 · 보류'),runListResultLabel({run_id:'other',stage:'failed'},'분석 실패')]));
""")
    assert result == ["분석 완료 · 조건부 연구후보", "원본 상태 · 자동 분석 완료 · 보류", "분석 실패"]
    assert "api(" not in function("runListResultLabel")


def test_workflow_raw_status_and_assay_comparison_limit_are_explicit():
    assert '`원본 실행 기록: ${snapshot.status_message}`' in function("renderPipeline")
    assert "서로 다른 실험 조건의 활성값은 직접 효능 비교가 아닙니다." in PAGE
