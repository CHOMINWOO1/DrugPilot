from pathlib import Path


def test_tool_trace_grouping_keeps_distinct_failure_causes_separate() -> None:
    app_js = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "drugpilot"
        / "web"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "const failureDiagnostic = isFailedTraceStatus(item.status)" in app_js
    assert 'item.detail || ""' in app_js
    assert "item?.metadata?.failure_category" in app_js
    assert "item?.metadata?.error_type" in app_js
    assert "item.granularity, failureDiagnostic" in app_js


def test_live_estimated_model_usage_is_never_presented_as_success() -> None:
    app_js = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "drugpilot"
        / "web"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'accountingUnknown ? "accounting_unknown" : "success"' in app_js
    assert '"accounting_unknown", "failed", "error"' in app_js
    assert 'accounting_unknown: "ACCOUNTING UNKNOWN"' in app_js
    assert 'accounting_unknown: "사용량 미확인"' in app_js
    assert 'usage_bound: accountingUnknown ? "conservative_upper_bound" : null' in app_js
    assert "보수적 상한 · 실제 사용량 미확인" in app_js
    assert "원장에 보존하고 실행을 중단" in app_js
