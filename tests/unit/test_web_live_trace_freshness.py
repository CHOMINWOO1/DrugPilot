"""Execute actual dashboard functions in isolated Node VM; no browser/server writes."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]


def run_js(body):
    node=shutil.which("node")
    if not node:pytest.skip("Node not available")
    prefix=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('src/drugpilot/web/static/app.js','utf8');
function block(start,end){const i=source.indexOf(start);assert(i>=0,start);const j=source.indexOf(end,i+start.length);assert(j>i,end);return source.slice(i,j);}
const pending=[];
const state={snapshot:{run_id:'a',stage:'ma1',token_usage:[]},callTrace:null,callTraceLoading:false,callTraceRevision:null,selectedTab:'overview'};
const context={state,encodeURIComponent,Promise,
  callTraceRevision:s=>JSON.stringify([s?.run_id,s?.stage,s?.token_usage?.length]),
  api:url=>new Promise((resolve,reject)=>pending.push({url,resolve,reject})),
  $:()=>null,renderMetrics(){},renderActivity(){},renderArtifact(){},showToast(){}};
vm.createContext(context);
vm.runInContext(block('  async function loadCallTrace(', '  function closeToolTrace('),context);
'''
    result=subprocess.run([node,"-e",prefix+body],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",timeout=15)
    assert result.returncode==0,result.stdout+result.stderr


def test_inflight_old_trace_refreshes_again_after_terminal_snapshot():
    run_js(r'''
(async()=>{
 const first=context.loadCallTrace('a',{quiet:true});
 state.snapshot={run_id:'a',stage:'human_gate',token_usage:[{}]};
 pending[0].resolve({run_id:'a',summary:{total_calls:0}});
 await first;
 assert.strictEqual(pending.length,2,'terminal snapshot must not retain the in-flight older trace');
 pending[1].resolve({run_id:'a',summary:{total_calls:1}});
 await new Promise(resolve=>setImmediate(resolve));
 assert.strictEqual(state.callTrace.summary.total_calls,1);
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_old_run_response_does_not_overwrite_new_run_or_trigger_replay():
    run_js(r'''
(async()=>{
 const first=context.loadCallTrace('a',{quiet:true});
 state.snapshot={run_id:'b',stage:'ma1',token_usage:[]};
 state.callTrace={run_id:'b',summary:{total_calls:5}};
 pending[0].resolve({run_id:'a',summary:{total_calls:1}});
 await first;
 assert.strictEqual(state.callTrace.run_id,'b');
 assert.strictEqual(pending.length,1);
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_trace_failure_releases_guard_without_recursive_request_storm():
    run_js(r'''
(async()=>{
 const first=context.loadCallTrace('a',{quiet:true});
 pending[0].reject(new Error('network unavailable'));
 assert.strictEqual(await first,null);
 assert.strictEqual(state.callTraceLoading,false);
 assert.strictEqual(pending.length,1);
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_switching_run_can_fetch_while_old_request_remains_inflight():
    run_js(r'''
(async()=>{
 const first=context.loadCallTrace('a',{quiet:true});
 state.snapshot={run_id:'b',stage:'human_gate',token_usage:[{}]};
 const second=context.loadCallTrace('b',{quiet:true});
 assert.strictEqual(pending.length,2);
 pending[0].resolve({run_id:'a',summary:{total_calls:1}});
 await first;
 assert.strictEqual(state.callTraceLoading,true);
 pending[1].resolve({run_id:'b',summary:{total_calls:5}});
 await second;
 assert.strictEqual(state.callTrace.run_id,'b');
 assert.strictEqual(state.callTraceLoading,false);
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_run_history_uses_newly_rendered_snapshot_not_stale_summary():
    run_js(r'''
const list={innerHTML:''};
state.runs=[{run_id:'a',stage:'ma1',brief:{disease:'Test'},mode:'live'}];
state.snapshot={run_id:'a',stage:'human_gate',brief:{disease:'Test'},mode:'live'};
Object.assign(context,{
 $:()=>list,$$:()=>[],unwrapSnapshot:x=>x,esc:x=>String(x),shortId:x=>x,relativeTime:()=>'',
 researchDepthFromSnapshot:()=>'',researchDepthPresentation:()=>({value:null,label:'none'}),
 getIntakeAssessment:()=>({}),runStatusPresentation:r=>({label:r.stage})
});
vm.runInContext(block('  function renderRunList()', '  async function selectRun('),context);
vm.runInContext(block('  function resultReadinessKey(', '  function researchSelectionAllowed('),context);
context.renderRunList();
assert(list.innerHTML.includes('human_gate'),'history must reflect current terminal stage');
assert(list.innerHTML.includes('원본 상태'),'unverified historical status must be explicitly labeled');

state.resultReadiness={run_id:'a',analysis_complete:true,ledger_valid:true,research_selection:{allowed:true,state:'not_selected'}};
state.resultReadinessKey=context.resultReadinessKey(state.snapshot);
context.renderRunList();
assert(list.innerHTML.includes('분석 완료 · 조건부 연구후보'),'current verified projection must replace the raw terminal label');
assert.strictEqual(pending.length,0,'rendering history must not fetch readiness');

state.snapshot={...state.snapshot,updated_at:'newer'};
context.renderRunList();
assert(list.innerHTML.includes('원본 상태 · human_gate'),'outdated projection must not override a newer snapshot');
assert(!list.innerHTML.includes('조건부 연구후보'),'stale projection must fail closed');

state.snapshot={...state.snapshot,stage:'failed'};
state.resultReadinessKey=context.resultReadinessKey(state.snapshot);
context.renderRunList();
assert(list.innerHTML.includes('failed'),'a later failure must override earlier analysis-complete data');
assert(!list.innerHTML.includes('조건부 연구후보'),'failure must not inherit the previous candidate result');
''')
