"""Direction controls exercise the real request builder without a paid run."""
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


def execute(code):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for browser-function regression")
    result = subprocess.run([node, "-e", code], capture_output=True, text=True,
                            encoding="utf-8", timeout=15, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("direction", ["modulate", "unspecified"])
@pytest.mark.parametrize("scope,expected", [("all", None), ("both", ["inhibit", "activate"]),
                                           ("inhibit", ["inhibit"]), ("activate", ["activate"])])
def test_broad_direction_scope_is_real_payload_and_preserves_request(direction, scope, expected):
    code = """
const nodes = { '#direction':{value:DIRECTION}, '#directionScope':{value:SCOPE} };
const $=id=>nodes[id]||{value:'example input',checked:false};
const state={health:{direction_policy:{allowed_hypothesis_directions:true}}};
const selectedResearchDepth=()=> 'standard', commaSeparatedValues=()=>[], clamp=()=>3;
""".replace("DIRECTION", json.dumps(direction)).replace("SCOPE", json.dumps(scope))
    code += "\n".join(function(n) for n in ("directionSettingsPresentation", "directionSelectionSupported",
                                          "directionSelectionPayload", "buildCreateRequest"))
    result = execute(code + "\nconsole.log(JSON.stringify(buildCreateRequest()));")
    assert result["intervention_direction"] == direction
    assert result.get("allowed_hypothesis_directions") == expected
    if expected is None:
        assert "allowed_hypothesis_directions" not in result


@pytest.mark.parametrize("direction", ["inhibit", "activate"])
def test_hidden_scope_cannot_leak_into_concrete_request(direction):
    code = "const state={health:null}; const $=id=>({value:id==='#direction'?" + json.dumps(direction) + ":'both'});\n"
    code += "\n".join(function(n) for n in ("directionSettingsPresentation", "directionSelectionSupported", "directionSelectionPayload"))
    assert execute(code + "\nconsole.log(JSON.stringify(directionSelectionPayload()));") == {}


def test_restricted_option_never_silently_dropped_on_old_or_offline_backend():
    code = "const state={health:{}}; const $=id=>({value:id==='#direction'?'modulate':'inhibit'});\n"
    code += "\n".join(function(n) for n in ("directionSettingsPresentation", "directionSelectionSupported", "directionSelectionPayload"))
    result = execute(code + "\ntry{directionSelectionPayload();console.log(JSON.stringify({blocked:false}));}catch(e){console.log(JSON.stringify({blocked:true,message:e.message}));}")
    assert result["blocked"] and "서버" in result["message"]


def test_controls_update_summary_visibility_and_backend_support():
    code = """
const ids=['direction','directionScope','directionDescription','directionSearchDescription','directionSummary','directionScopeField','directionCapabilityNotice'];
const nodes=Object.fromEntries(ids.map(id=>['#'+id,{value:id==='direction'?'modulate':'inhibit'}]));
const $=id=>nodes[id]; const state={health:{}};
""" + "\n".join(function(n) for n in ("directionSettingsPresentation", "directionSelectionSupported", "updateDirectionSettings"))
    result = execute(code + """
updateDirectionSettings(); const offline=nodes['#directionScope'].disabled;
state.health={direction_policy:{allowed_hypothesis_directions:true}}; updateDirectionSettings();
const broad={enabled:!nodes['#directionScope'].disabled,shown:!nodes['#directionScopeField'].hidden,summary:nodes['#directionSummary'].textContent,search:nodes['#directionSearchDescription'].textContent};
nodes['#direction'].value='activate';updateDirectionSettings();
console.log(JSON.stringify({offline,broad,concreteHidden:nodes['#directionScopeField'].hidden,concreteSummary:nodes['#directionSummary'].textContent}));
""")
    assert result["offline"] and result["broad"]["enabled"] and result["broad"]["shown"]
    assert "최종 가설: 억제 ·" in result["broad"]["summary"]
    assert "반대·맥락 근거" in result["broad"]["search"]
    assert result["concreteHidden"] and "활성화" in result["concreteSummary"]


def test_direction_result_does_not_invent_legacy_evidence_and_escapes_ids():
    code = "const esc=v=>String(v).replaceAll('<','&lt;').replaceAll('>','&gt;');\n"
    code += function("directionLabel") + function("directionResolutionHtml")
    result = execute(code + """
console.log(JSON.stringify([directionResolutionHtml({}),directionResolutionHtml({target:{metadata:{direction_resolution:{requested_direction:'modulate',resolved_direction:'inhibit',evidence_ids:['e<script>']}}}})]));
""")
    assert "미기록" in result[0] and "근거 0건" not in result[0]
    assert "근거 방향" in result[1] and "G0 심사와 별도" in result[1]
    assert "e&lt;script&gt;" in result[1] and "<script>" not in result[1]


def test_selection_explains_filter_not_search_restriction_and_has_cache_bust():
    assert 'id="directionScope"' in PAGE and 'id="directionSummary"' in PAGE
    assert "검색 범위가 아닌 가설 채택 조건" in PAGE
    assert "이전 결과를 변경하지 않습니다" in PAGE
    assert "20260929-ma3-admission-v1" in PAGE
    assert '$("#directionScope").addEventListener("change", updateDirectionSettings)' in SCRIPT
    assert '허용 가설 방향' in SCRIPT and 'directionConstraintLabel(constraints)' in SCRIPT


def test_local_validation_precedes_stream_stop_and_offline_preview_remains_local():
    start = SCRIPT.index("  async function createRun(")
    end = SCRIPT.index("  function ", start)
    body = SCRIPT[start:end]
    assert body.index("JSON.stringify(buildCreateRequest())") < body.index("stopLiveUpdates()")
    assert "requireDirectionSupport: false" in function("previewBrief")
    code = "const state={health:null}; const $=id=>({value:id==='#direction'?'modulate':'activate'});\n"
    code += "\n".join(function(n) for n in ("directionSettingsPresentation", "directionSelectionSupported", "directionSelectionPayload"))
    assert execute(code + "\nconsole.log(JSON.stringify(directionSelectionPayload(false)));") == {"allowed_hypothesis_directions": ["activate"]}
