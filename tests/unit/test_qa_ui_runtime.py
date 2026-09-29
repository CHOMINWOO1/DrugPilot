"""Exercise QA renderers and cancellation interaction, including an ambiguous POST."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")


def execute(body):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    completed = subprocess.run([node, "-e", body], capture_output=True, text=True, encoding="utf-8", timeout=15, check=True)
    return json.loads(completed.stdout)


def function(name, next_name):
    return SCRIPT[SCRIPT.index("  function " + name):SCRIPT.index("  function " + next_name)]


def test_reviewer_raw_severity_is_preserved_and_html_escaped():
    code = function("reviewerAuthorityHtml", "findingContextHtml")
    result = execute("const esc = v => String(v).replaceAll('<', '&lt;').replaceAll('>', '&gt;');\n" + code + "\nconsole.log(JSON.stringify([reviewerAuthorityHtml({}), reviewerAuthorityHtml({confirmation_status:'unconfirmed',reported_severity:'HIGH<script>',effective_severity:'MEDIUM'})]));")
    assert result[0] == ""
    assert "HIGH&lt;script&gt;" in result[1]
    assert "판정 반영 MEDIUM" in result[1] and "재작업하지 않습니다" in result[1]
    assert "<script>" not in result[1]


@pytest.mark.parametrize("ambiguous", [False, True])
def test_cancel_posts_once_and_reads_after_ambiguous_response(ambiguous):
    start = SCRIPT.index("  async function cancelRun()")
    end = SCRIPT.index("  function renderResumeControl", start)
    code = SCRIPT[start:end]
    result = execute("""
const state={snapshot:{run_id:'run_a'},preview:false,cancellingRunId:null,cancelStatus:{eligible:true},cancelStatusRunId:'run_a'};
const calls=[]; const LIVE_END_STAGES=new Set(['cancelled']);
const renderCancelControl=()=>{},renderSnapshot=()=>{},startLiveUpdates=()=>{},showToast=()=>{};
const unwrapSnapshot=x=>x; const loadCancelStatus=async()=>{};
const fetchRun=async()=>{calls.push('GET');};
const api=async(path,options)=>{calls.push(options.method);
""" + ("throw new Error('response lost');" if ambiguous else "return {run_id:'run_a',stage:'cancel_requested'};") + "};\n" + code + "\n(async()=>{await cancelRun();console.log(JSON.stringify({calls,busy:state.cancellingRunId}));})();")
    assert result["calls"] == (["POST", "GET"] if ambiguous else ["POST"])
    assert result["busy"] is None


def test_cancellation_state_overrides_prior_gate_summary():
    code = function("runStatusPresentation", "stageLabel")
    result = execute(code + "\nconsole.log(JSON.stringify(['cancel_requested','cancelled'].map(stage=>runStatusPresentation({stage,human_decision:{decision:'HOLD'}}))));")
    assert [row["label"] for row in result] == ["중단 처리 중", "사용자 중단"]


def test_evidence_linkage_trace_preserves_scoped_roles_and_escapes_sources():
    code = function("evidenceLinkageTraceHtml", "targetSourceTag")
    packet = {"metadata": {"evidence_linkage_trace": {"policy": "hypothesis_scoped_evidence_linkage_trace_v1",
        "entry_count_total": 2, "entry_count_recorded": 2, "entry_count_omitted": 0,
        "entries": [{"target_symbol": "JAK2", "molecular_scope": scope, "external_id": "24858412<script>",
            "source": "PubMed", "linked_roles": roles, "hypothesis_disposition": "excluded"}
            for scope, roles in [("JAK2:V617F", ["primary_target_disease_association"]), ("JAK2", [])]]}}}
    result = execute("const esc=v=>String(v).replaceAll('<','&lt;').replaceAll('>','&gt;'); const directionLabel=v=>v||'미기록';\n"
        + code + "\nconsole.log(JSON.stringify([evidenceLinkageTraceHtml({}),evidenceLinkageTraceHtml(" + json.dumps(packet) + ")]));")
    assert "미기록" in result[0] and "0/0" not in result[0]
    assert "JAK2:V617F" in result[1] and "primary_target_disease_association" in result[1]
    assert "연결 없음" in result[1] and "<script>" not in result[1]
