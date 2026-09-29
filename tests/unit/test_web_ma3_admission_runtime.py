"""MA1 submissions are not MA3 approvals; source-bound report display is read-only."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "src/drugpilot/web/static/app.js").read_text(encoding="utf-8")


def function(name):
    start = re.search(r"^  (?:async )?function " + re.escape(name) + r"\(", SCRIPT, re.M).start()
    following = re.search(r"^  (?:async )?function ", SCRIPT[start + 3:], re.M)
    return SCRIPT[start:start + 3 + following.start()] if following else SCRIPT[start:]


def execute(body):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    result = subprocess.run([node, "-e", body], capture_output=True, text=True,
                            encoding="utf-8", timeout=15, check=True)
    return json.loads(result.stdout)


def projection(packet):
    return execute(function("ma1AdmissionProjection") + "\nconsole.log(JSON.stringify(ma1AdmissionProjection("
                   + json.dumps(packet) + ")));" )


def hypothesis(identifier="h1", symbol="FLT3"):
    return {"hypothesis_id": identifier, "target": {"symbol": symbol}, "mechanism": "Proposed mechanism",
            "intervention_direction": "inhibit", "evidence": [], "claims": []}


def packet(provisional=None, approved=None, entries=None):
    value = {"metadata": {"admission_policy": "ma3-source-bound-v1"},
             "provisional_hypotheses": provisional or [], "hypotheses": approved or []}
    if entries is not None:
        value["metadata"]["ma3_admission_report"] = {"entries": entries, "submitted_count": len(entries),
                                                     "approved_count": len(approved or [])}
    return value


def test_legacy_empty_new_field_preserves_retained_semantics():
    result = projection({"hypotheses": [hypothesis()], "provisional_hypotheses": []})
    assert not result["enabled"] and result["approvedCount"] == 1


@pytest.mark.parametrize("view", ["hypothesisNarrativeBody", "hypothesisWhyHtml"])
def test_admission_placeholder_confidence_is_not_displayed_as_probability(view):
    item = hypothesis()
    item["claims"] = [{"text": "Source-bound research claim", "confidence": 0.5}]
    setup = "const esc=String,formatScore=()=> '50%',state={snapshot:{run_id:'test'}};\n"
    setup += function("hypothesisNarrative") + function(view)
    legacy = execute(setup + "\nconsole.log(JSON.stringify(" + view + "(" + json.dumps(item) + ")));" )
    assert "50%" in legacy
    item["target"]["metadata"] = {"ma3_admission": {"policy": "ma3-source-bound-v1"}}
    current = execute(setup + "\nconsole.log(JSON.stringify(" + view + "(" + json.dumps(item) + ")));" )
    assert "신뢰도 미보정 · 확률 아님" in current
    assert "50%" not in current


def test_provisional_only_is_one_proposal_zero_approvals():
    result = projection(packet(provisional=[hypothesis()]))
    assert result["submittedCount"] == 1 and result["approvedCount"] == 0
    assert result["rows"][0]["status"] == "pending"
    assert not result["reviewed"] and not result["inconsistent"]


def test_report_preserves_nonapproved_proposals_after_provisional_list_is_cleared():
    entries = [{"hypothesis_id": "h1", "target_symbol": "FLT3", "status": "approved",
                "reason": "Narrow source-bound support", "limitations": ["Not clinical efficacy"], "original_proposal": hypothesis()},
               {"hypothesis_id": "h2", "target_symbol": "OTHER", "status": "needs_evidence",
                "reason": "Missing perturbation result", "original_proposal": hypothesis("h2", "OTHER")}]
    result = projection(packet(approved=[hypothesis()], entries=entries))
    assert result["submittedCount"] == 2 and result["approvedCount"] == 1
    assert [row["status"] for row in result["rows"]] == ["approved", "needs_evidence"]
    assert result["rows"][1]["hypothesis"]["hypothesis_id"] == "h2"
    assert not result["inconsistent"]


@pytest.mark.parametrize("status", ["approved", "PASS", "<script>"])
def test_report_cannot_create_approval_without_matching_approved_artifact(status):
    result = projection(packet(provisional=[hypothesis()], entries=[{
        "hypothesis_id": "h1", "status": status, "reason": "claimed", "original_proposal": hypothesis()}]))
    assert result["approvedCount"] == 0
    assert result["rows"][0]["status"] == "hold" and result["inconsistent"]


def test_approved_artifact_without_review_entry_is_not_displayed_as_approved():
    result = projection(packet(approved=[hypothesis()], entries=[]))
    assert result["approvedCount"] == 0 and result["inconsistent"]


def test_duplicate_and_conflicting_report_entries_are_fail_closed():
    entries = [{"hypothesis_id": "h1", "status": status} for status in ["hold", "approved"]]
    result = projection(packet(approved=[hypothesis()], entries=entries))
    assert result["approvedCount"] == 0 and result["rows"][0]["status"] == "hold"


def test_original_proposal_identity_mismatch_is_not_relabelled_approved():
    result = projection(packet(approved=[hypothesis()], entries=[{
        "hypothesis_id": "h1", "status": "approved", "original_proposal": hypothesis("h2")}]))
    assert result["approvedCount"] == 0 and result["inconsistent"]


def test_pending_card_and_full_detail_are_escaped_without_pass_styling():
    value = packet(provisional=[hypothesis(symbol="<script>alert(1)</script>")])
    body = """
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const compactNumber=String,directionLabel=x=>x||'미지정',hypothesisSourceToolBadges=()=>'',evidenceLink=x=>esc(x.title);
""" + function("ma1AdmissionProjection") + function("ma3AdmissionDispositionHtml") + function("renderMa3AdmissionHypotheses")
    html = execute(body + "\nconsole.log(JSON.stringify(renderMa3AdmissionHypotheses(" + json.dumps(value) + ")));" )
    assert "제안 1 · 승인 0" in html and "심사 대기 · 잠정" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "admission-approved" not in html and "is-retained" not in html
    assert "제출 수는 검증 완료 수가 아닙니다" in html


def test_long_reason_remains_available_in_detail_with_limitations():
    reason = "Full reason " * 50 + "<end>"
    body = "const esc=x=>String(x).replaceAll('<','&lt;').replaceAll('>','&gt;');\n" + function("ma3AdmissionDispositionHtml")
    result = execute(body + "\nconst row=" + json.dumps({"status": "hold", "reason": reason, "limitations": ["<unverified>"]})
                     + ";console.log(JSON.stringify({short:ma3AdmissionDispositionHtml(row),full:ma3AdmissionDispositionHtml(row,{detail:true})}));")
    assert "&lt;end&gt;" not in result["short"] and "&lt;end&gt;" in result["full"]
    assert "&lt;unverified&gt;" in result["full"]


def test_summary_badge_counts_proposals_and_approvals_separately():
    body = "const nodes={};const $=id=>nodes[id]||(nodes[id]={});const state={snapshot:{hypothesis_packet:"
    body += json.dumps(packet(provisional=[hypothesis()])) + "}};\n"
    body += function("ma1AdmissionProjection") + function("renderCounts")
    result = execute(body + "\nrenderCounts();console.log(JSON.stringify(nodes));")
    assert result["#hypothesisCount"]["textContent"] == "제안 1 · MA3 승인 0"


def test_review_coverage_and_source_bindings_remain_visible_without_claiming_exhaustiveness():
    row = {"status": "hold", "reason": "More evidence needed", "entry": {
        "omissions": {"available_records": 30, "reviewed_records": 12, "omitted_records": [{}] * 18},
        "source_bindings": [{"source": "<PubMed>", "external_id": "123", "evidence_id": "e1",
                             "source_binding_sha256": "a" * 64, "source_text_sha256": "b" * 64}]}}
    body = "const esc=x=>String(x).replaceAll('<','&lt;').replaceAll('>','&gt;');\n" + function("ma3AdmissionDispositionHtml")
    html = execute(body + "\nconsole.log(JSON.stringify(ma3AdmissionDispositionHtml(" + json.dumps(row) + ",{detail:true})));" )
    assert "확보 30건 · 검토 12건 · 제외/미포함 18건" in html
    assert "전체 문헌을 검토했다는 의미는 아닙니다" in html
    assert "&lt;PubMed&gt;" in html and "a" * 64 in html and "b" * 64 in html


def test_admission_ui_never_posts_or_rewrites_stored_artifacts():
    code = "".join(function(name) for name in ["ma1AdmissionProjection", "ma3AdmissionDispositionHtml", "renderMa3AdmissionHypotheses"])
    assert "fetch(" not in code and "api(" not in code
    assert "packet.hypotheses =" not in code and "packet.metadata =" not in code


def test_provisional_screening_trace_is_not_rendered_as_rejected():
    body = """
const esc=String, compactNumber=String,screeningPhaseLabel=x=>x,screeningTraceDetailHtml=()=>'';
const packet={metadata:{admission_policy:'ma3-source-bound-v1'}};
const projection={records:[{disposition:'provisional',targetSymbol:'FLT3',hypothesisId:'h1',reasonCodes:['LEGACY_DIAGNOSTIC'],track:'literature',evidenceCount:2}]};
""" + function("renderScreenedProposals")
    html = execute(body + "\nconsole.log(JSON.stringify(renderScreenedProposals(packet,projection)));" )
    assert 'screening-row is-provisional' in html and "MA1 잠정 제출 기록" in html
    assert 'screening-status is-rejected' not in html
    assert "진단 코드는 참고 기록" in html
