from __future__ import annotations

from datetime import timedelta

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    CandidatePacket,
    ExecutionEvent,
    ResearchBrief,
    RunMode,
    RunSnapshot,
    RunStage,
    ScientificBackendKind,
    ScientificExecutionStatus,
    TokenUsage,
    ToolCallRecord,
    utc_now,
)
from drugpilot.orchestration.explanations import interaction
from drugpilot.api.call_trace import build_run_call_trace


def test_call_trace_projects_bounded_paper_v6_failure_fields(
    brief: ResearchBrief,
) -> None:
    started = utc_now()
    reason = "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT"
    call = ToolCallRecord(
        call_id="tool_paper_v6_failure",
        tool_name="repurposing_search",
        operation="retrieve_candidates",
        input_hash="1" * 64,
        output_hash="2" * 64,
        source_version="live-v6",
        started_at=started,
        finished_at=started + timedelta(milliseconds=10),
        success=False,
        error="bounded failure",
        backend_kind=ScientificBackendKind.LIVE,
        fixture=False,
        execution_status=ScientificExecutionStatus.UNAVAILABLE,
        provenance={
            "reason_code": reason,
            "support_candidate_count": 2,
            "minimum_verified_candidate_count": 3,
            "evidence_grade": "B",
            "promotion_eligible": False,
            "rejection_count": 10,
            "wrapper_atomic_screen_projection": {
                "attestation_sha256": "a" * 64,
            },
            "unbounded_detail": "must not be projected",
        },
    )
    packet = CandidatePacket(
        plan_id="plan_paper_v6_failure",
        candidates=[],
        tool_calls=[call],
        backend_kind=ScientificBackendKind.LIVE,
        execution_status=ScientificExecutionStatus.UNAVAILABLE,
        abstain_reason_codes=[reason],
    )
    snapshot = RunSnapshot(
        run_id="run_paper_v6_failure",
        mode=RunMode.HYBRID,
        stage=RunStage.MA2,
        status_message="MA2 unavailable",
        brief=brief,
        candidate_packet=packet,
    )

    trace = build_run_call_trace(snapshot)

    projected = trace.scientific_tool_calls[0]
    assert projected.metadata["reason_code"] == reason
    assert projected.metadata["support_candidate_count"] == 2
    assert projected.metadata["minimum_verified_candidate_count"] == 3
    assert projected.metadata["evidence_grade"] == "B"
    assert projected.metadata["promotion_eligible"] is False
    assert projected.metadata["rejection_count"] == 10
    assert projected.metadata["wrapper_screen_attestation_sha256"] == (
        "a" * 64
    )
    assert "unbounded_detail" not in projected.metadata


def test_call_trace_marks_estimated_live_usage_as_accounting_unknown(
    brief: ResearchBrief,
) -> None:
    started = ExecutionEvent(
        event_id="evt_estimated_live_start",
        run_id="run_estimated_live",
        stage=RunStage.MA1,
        event_type="llm_call_started",
        message="MA1 structured generation started.",
        node="ma1.llm",
        metadata={
            "interaction": interaction(
                "MA1",
                "LLM · gpt-5.6-luna",
                "structured_generation_request",
                "bounded request",
                model="gpt-5.6-luna",
            )
        },
    )
    snapshot = RunSnapshot(
        run_id="run_estimated_live",
        mode=RunMode.HYBRID,
        stage=RunStage.MA1,
        status_message="Stopped after unknown usage accounting.",
        brief=brief,
        events=[started],
        token_usage=[
            TokenUsage(
                component="MA1",
                model="gpt-5.6-luna",
                input_tokens=1_000,
                output_tokens=100,
                estimated_cost_usd=0.00032,
                estimated=True,
            ),
            TokenUsage(
                component="MA3-G0-Reviewer",
                model="claude-opus-4-1",
                input_tokens=2_000,
                output_tokens=200,
                estimated_cost_usd=0.045,
                estimated=True,
            ),
            TokenUsage(
                component="Mock-Diagnostic",
                model="mock:gpt-5.6-luna",
                input_tokens=10,
                output_tokens=2,
                estimated=True,
            ),
        ],
    )

    trace = build_run_call_trace(snapshot)

    assert trace.summary.model_calls == 3
    assert trace.summary.failed == 2
    assert trace.summary.successful == 1
    live_items = [item for item in trace.model_calls if not item.name.startswith("mock:")]
    assert {item.status for item in live_items} == {"accounting_unknown"}
    assert all(item.metadata["accounting_status"] == "unknown" for item in live_items)
    assert all(item.metadata["usage_bound"] == "conservative_upper_bound" for item in live_items)
    assert all(item.metadata["actual_usage_known"] is False for item in live_items)
    assert all("run was stopped" in (item.detail or "") for item in live_items)
    mock_item = next(item for item in trace.model_calls if item.name.startswith("mock:"))
    assert mock_item.status == "success"


def test_call_trace_preserves_exact_tool_lineage_and_source_granularity(
    tmp_path, brief: ResearchBrief
) -> None:
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "calls.sqlite3",
        candidates_per_route=brief.constraints.candidates_per_route,
    )
    hypotheses = MA1Agent(settings=settings).run(brief)
    metadata = dict(hypotheses.metadata)
    metadata["target_source_statuses"] = [
        {
            "source": "PubMed",
            "status": "cache",
            "operations": ["search", "summary"],
            "records_observed": 4,
            "operation_records_observed": 4,
            "unique_records_observed": 2,
            "records_screened": 2,
            "operation_metrics": [
                {
                    "operation": "search",
                    "success": True,
                    "cache_hit": True,
                    "operation_records_observed": 2,
                    "unique_records_observed": 2,
                    "records_screened": 0,
                    "failure_code": None,
                },
                {
                    "operation": "summary",
                    "success": True,
                    "cache_hit": True,
                    "operation_records_observed": 2,
                    "unique_records_observed": 2,
                    "records_screened": 2,
                    "failure_code": None,
                },
            ],
            "cache_hits": 2,
            "failure_codes": [],
        }
    ]
    hypotheses.metadata = metadata
    first = hypotheses.hypotheses[0]
    target_metadata = dict(first.target.metadata)
    target_metadata["identity"] = {
        "sources": [
            {
                "source": "UniProt",
                "status": "ok",
                "queried": True,
                "cache_hit": False,
                "provenance_class": "external_live",
                "query": first.target.symbol,
                "matched_by": "gene_symbol",
                "endpoint": "https://rest.uniprot.org/uniprotkb/search",
            }
        ]
    }
    first.target.metadata = target_metadata

    plan = PlannerAgent(settings=settings).run(hypotheses, brief)
    candidates = MA2Agent().run(plan, hypotheses)
    started = utc_now()
    completed = started + timedelta(milliseconds=25)
    events = [
        ExecutionEvent(
            event_id="evt_llm_start",
            run_id="run_calls",
            stage=RunStage.MA1,
            event_type="llm_call_started",
            message="MA1 structured generation started.",
            node="ma1.llm",
            metadata={
                "interaction": interaction(
                    "MA1",
                    "LLM · gpt-test",
                    "structured_generation_request",
                    "bounded request",
                    model="gpt-test",
                )
            },
            created_at=started,
        ),
        ExecutionEvent(
            event_id="evt_llm_done",
            run_id="run_calls",
            stage=RunStage.MA1,
            event_type="llm_call_completed",
            message="MA1 structured generation completed.",
            node="ma1.llm",
            metadata={
                "interaction": interaction(
                    "LLM · gpt-test",
                    "MA1",
                    "structured_generation_response",
                    "validated response",
                    model="gpt-test",
                )
            },
            created_at=completed,
        ),
    ]
    snapshot = RunSnapshot(
        run_id="run_calls",
        mode=RunMode.MOCK,
        stage=RunStage.HUMAN_GATE,
        status_message="Ready",
        brief=brief,
        hypothesis_packet=hypotheses,
        plan_spec=plan,
        candidate_packet=candidates,
        events=events,
        token_usage=[
            TokenUsage(
                component="MA1",
                model="gpt-test",
                input_tokens=100,
                output_tokens=20,
            )
        ],
    )

    ledger = build_run_call_trace(snapshot)

    assert ledger.summary.model_calls == 1
    assert ledger.model_calls[0].status == "success"
    assert ledger.model_calls[0].duration_ms == 25
    assert ledger.model_calls[0].metadata["total_tokens"] == 120

    pubmed = next(
        item
        for item in ledger.public_data_calls
        if item.node == "ma1.target_discovery" and item.name == "PubMed"
    )
    assert pubmed.granularity == "aggregate"
    assert pubmed.cached is True
    assert pubmed.records_observed == 2
    assert pubmed.metadata["legacy_records_observed"] == 4
    assert pubmed.metadata["operation_records_observed"] == 4
    assert pubmed.metadata["unique_records_observed"] == 2
    assert pubmed.metadata["records_screened"] == 2
    assert len(pubmed.metadata["operation_metrics"]) == 2
    identity = next(
        item
        for item in ledger.public_data_calls
        if item.node == "ma1.target_identity" and item.name == "UniProt"
    )
    assert identity.granularity == "exact"
    assert identity.metadata["target_symbol"] == first.target.symbol

    assert ledger.scientific_tool_calls
    assert all(item.granularity == "exact" for item in ledger.scientific_tool_calls)
    assert all(item.agent == "MA2" for item in ledger.scientific_tool_calls)
    assert all(item.fixture for item in ledger.scientific_tool_calls)
    assert {item.name for item in ledger.scientific_tool_calls} >= {
        "target_normalizer",
        "repurposing_search",
        "docking",
        "admet_predictor",
        "candidate_judge",
    }
    linked = [item for item in ledger.scientific_tool_calls if item.candidate_ids]
    assert linked
    assert all(item.metadata["input_hash"] for item in linked)
    assert all(item.metadata["output_hash"] for item in linked)


def test_public_source_trace_distinguishes_failures_from_empty_skipped_and_fixture(
    tmp_path, brief: ResearchBrief
) -> None:
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "source-status.sqlite3",
        candidates_per_route=brief.constraints.candidates_per_route,
    )
    hypotheses = MA1Agent(settings=settings).run(brief)
    hypotheses.metadata = {
        **hypotheses.metadata,
        "target_source_statuses": [
            {
                "source": "Open Targets",
                "status": "partial",
                "operations": ["search_disease", "associated_targets"],
                "records_observed": 3,
                "cache_hits": 0,
                "failure_codes": ["ConnectError"],
            },
            {
                "source": "PubMed",
                "status": "empty",
                "operations": ["search_publications"],
                "records_observed": 0,
                "cache_hits": 0,
                "failure_codes": [],
            },
        ],
        "context_routing": {
            "status": "completed_with_source_issues",
            "source_calls_used": 3,
            "planned_source_calls": 3,
            "elapsed_seconds": 0.3,
            "source_outcomes": [
                {
                    "target_symbol": "TP53",
                    "hypothesis_id": hypotheses.hypotheses[0].hypothesis_id,
                    "track": hypotheses.hypotheses[0].track.value,
                    "source": "STRING",
                    "status": "error",
                    "queried": True,
                    "adapter_status": "error",
                    "call_index": 1,
                    "cache_hit": False,
                    "record_count": 0,
                    "rejected_record_count": 0,
                    "elapsed_seconds": 0.1,
                    "error_type": "ReadTimeout",
                    "detail": "upstream did not respond",
                },
                {
                    "target_symbol": "TP53",
                    "hypothesis_id": hypotheses.hypotheses[0].hypothesis_id,
                    "track": hypotheses.hypotheses[0].track.value,
                    "source": "Reactome",
                    "status": "empty",
                    "queried": True,
                    "adapter_status": "empty",
                    "call_index": 2,
                    "cache_hit": False,
                    "record_count": 0,
                    "rejected_record_count": 0,
                    "elapsed_seconds": 0.1,
                    "error_type": None,
                    "detail": None,
                },
                {
                    "target_symbol": "TP53",
                    "hypothesis_id": hypotheses.hypotheses[0].hypothesis_id,
                    "track": hypotheses.hypotheses[0].track.value,
                    "source": "QuickGO",
                    "status": "skipped_missing_identifier",
                    "queried": False,
                    "adapter_status": None,
                    "call_index": None,
                    "cache_hit": False,
                    "record_count": 0,
                    "rejected_record_count": 0,
                    "elapsed_seconds": 0.0,
                    "error_type": None,
                    "detail": "uniprot_id is required",
                },
            ],
        },
    }
    evidence = hypotheses.hypotheses[0].evidence[0]
    evidence.metadata = {
        **evidence.metadata,
        "provenance_class": "fixture_fallback",
        "mock": True,
        "fallback_reason": {
            "PubMed": "empty_result",
            "UniProt": "ConnectError",
            "configuration": "no supported public biology source is allowed",
        },
        "partial_source_failures": {"Europe PMC": "HTTPStatusError"},
    }

    plan = PlannerAgent(settings=settings).run(hypotheses, brief)
    candidates = MA2Agent().run(plan, hypotheses)
    candidates.metadata = {
        **candidates.metadata,
        "public_enrichment_queries": [
            {
                "candidate_id": candidates.candidates[0].candidate_id,
                "candidate_name": candidates.candidates[0].name,
                "route": candidates.candidates[0].route.value,
                "source": "ChEMBL",
                "status": "error",
                "queried": True,
                "record_count": 0,
                "error_type": "upstream_http_500",
            },
            {
                "source": "PubChem",
                "status": "empty",
                "queried": True,
                "record_count": 0,
            },
            {
                "source": "BindingDB",
                "status": "skipped",
                "queried": False,
                "record_count": 0,
            },
            {
                "source": "DrugCentral",
                "status": "not_configured",
                "queried": False,
                "record_count": 0,
                "error_type": "local_dataset_not_configured",
            },
            {
                "source": "Google Patents",
                "status": "skipped",
                "queried": False,
                "record_count": 0,
                "error_type": "policy_rejected",
                "record_type": "source_routing_decision",
                "selection_reason": (
                    "policy_rejected: ResearchBrief.data_policy.allowed_sources "
                    "excludes this source"
                ),
            },
        ],
    }
    snapshot = RunSnapshot(
        run_id="run_source_statuses",
        mode=RunMode.MOCK,
        stage=RunStage.HUMAN_GATE,
        status_message="Ready",
        brief=brief,
        hypothesis_packet=hypotheses,
        plan_spec=plan,
        candidate_packet=candidates,
    )

    trace = build_run_call_trace(snapshot)

    failed = [
        item
        for item in trace.public_data_calls
        if item.status in {"error", "failed", "timeout", "blocked", "unavailable"}
    ]
    assert trace.summary.failed == len(failed) == 5
    assert {item.name for item in failed} == {
        "Open Targets",
        "UniProt",
        "Europe PMC",
        "STRING",
        "ChEMBL",
    }

    evidence_outcomes = [
        item
        for item in trace.public_data_calls
        if item.node == "ma1.evidence_retrieval"
        and item.operation == "retrieve_evidence"
    ]
    assert next(item for item in evidence_outcomes if item.name == "PubMed").status == "empty"
    assert (
        next(item for item in evidence_outcomes if item.name == "PubMed")
        .metadata["failure_category"]
        == "no_results"
    )
    assert (
        next(item for item in evidence_outcomes if item.name == "Public biology").status
        == "not_configured"
    )
    assert next(item for item in evidence_outcomes if item.name == "UniProt").metadata[
        "failure_category"
    ] == "connectivity"
    assert next(item for item in evidence_outcomes if item.name == "Europe PMC").metadata[
        "failure_category"
    ] == "upstream_http"

    context = [
        item for item in trace.public_data_calls if item.node == "ma1.context_router"
    ]
    assert {item.name: item.status for item in context} == {
        "STRING": "error",
        "Reactome": "empty",
        "QuickGO": "skipped_missing_identifier",
    }
    assert next(item for item in context if item.name == "STRING").metadata[
        "failure_category"
    ] == "connectivity"
    assert all(item.granularity == "exact" for item in context)

    candidate_statuses = {
        item.name: item.status
        for item in trace.public_data_calls
        if item.node == "ma2.public_enrichment"
    }
    assert candidate_statuses == {
        "ChEMBL": "error",
        "PubChem": "empty",
        "BindingDB": "skipped",
        "DrugCentral": "not_configured",
        "Google Patents": "skipped",
    }
    policy_rejected = next(
        item
        for item in trace.public_data_calls
        if item.node == "ma2.public_enrichment" and item.name == "Google Patents"
    )
    assert policy_rejected.metadata["failure_category"] is None
    assert policy_rejected.status == "skipped"
    chembl = next(
        item
        for item in trace.public_data_calls
        if item.node == "ma2.public_enrichment" and item.name == "ChEMBL"
    )
    assert chembl.metadata["failure_category"] == "upstream_http"
