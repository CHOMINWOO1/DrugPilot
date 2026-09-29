from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import secrets
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from drugpilot.api.call_trace import RunCallTrace, build_run_call_trace
from drugpilot.config import Settings, get_settings
from drugpilot.domain.models import (
    BudgetPolicy,
    ExperimentalResultRecord,
    FaultType,
    GatewayReadiness,
    HumanDecisionCode,
    HumanGateDecision,
    HypothesisDirection,
    ProjectConstraints,
    ReservedControlCompound,
    ResearchBrief,
    ResearchDepth,
    RunMode,
    RunSnapshot,
    validate_hypothesis_direction_selection,
    utc_now,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.orchestration.human_review import HumanReviewService
from drugpilot.orchestration.research_selection import ResearchSelectionService
from drugpilot.storage import (
    CheckpointIntegrityError,
    RunAdmissionExists,
    RunAdmissionLimitExceeded,
    SQLiteRunStore,
)
from drugpilot.tools.http import get_scientific_http_client
from drugpilot.tools.scientific_tool_health import (
    ScientificToolProbeReport,
    probe_scientific_tools,
    receptor_profile_health,
)
from drugpilot.tools.source_catalog import (
    SourceConfiguration,
    source_status_inventory,
)
from drugpilot.tools.structure_renderer import (
    InvalidStructureInput,
    StructureRenderingError,
    StructureToolkitUnavailable,
    render_structure_svg,
)


class CreateRunRequest(BaseModel):
    research_goal: str = Field(min_length=10, max_length=4_000)
    disease: str = Field(min_length=2, max_length=300)
    disease_id: str | None = None
    rationale: str | None = Field(default=None, max_length=4_000)
    # Optional keeps non-UI API clients backward compatible.  Every new
    # ResearchBrief still persists an explicit effective value below.
    research_depth: ResearchDepth | None = None
    target_cell_type: str | None = None
    intervention_direction: Literal["inhibit", "activate", "modulate", "unspecified"] = "inhibit"
    allowed_hypothesis_directions: list[HypothesisDirection] | None = Field(
        default=None, min_length=1, max_length=3, exclude_if=lambda value: value is None,
    )
    chronic_safety_required: bool = True
    druggability_required: bool = True
    discovery_priority: Literal[
        "auto", "novel_target", "repurposing_target"
    ] = "auto"
    forbidden_targets: list[str] = Field(default_factory=list)
    forbidden_substructures: list[str] = Field(default_factory=list)
    forbidden_drugs: list[str] = Field(default_factory=list)
    forbidden_drug_ids: list[str] = Field(default_factory=list)
    forbidden_drug_classes: list[str] = Field(default_factory=list)
    reserved_control_compounds: list[ReservedControlCompound] = Field(
        default_factory=list,
        max_length=50,
    )
    candidates_per_route: int | None = Field(default=None, ge=1, le=100)
    max_ma1_iterations: int | None = Field(default=None, ge=1, le=3)
    max_rework_loops: int | None = Field(default=None, ge=0, le=5)
    human_priorities: list[str] = Field(default_factory=list)
    fault_injection: FaultType = FaultType.NONE

    @model_validator(mode="after")
    def validate_direction_selection(self) -> "CreateRunRequest":
        validate_hypothesis_direction_selection(
            self.intervention_direction, self.allowed_hypothesis_directions,
        )
        return self


class HumanDecisionRequest(BaseModel):
    decision: HumanDecisionCode
    researcher: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=4_000)
    selected_candidate_ids: list[str] = Field(default_factory=list)


class ResearchSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    researcher: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=4_000)
    selected_candidate_ids: list[str] = Field(min_length=1, max_length=100)
    expected_g1_verdict_id: str = Field(min_length=1, max_length=200)
    expected_candidate_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations_acknowledged: Literal[True]
    policy_version: Literal["research-selection-v1"]


class ActionEvidenceRequest(BaseModel):
    experiment_request_id: str | None = None
    evidence_type: Literal[
        "experimental_result",
        "source_document",
        "code_validation",
        "tool_readiness",
        "researcher_policy",
        "resource_approval",
        "integrity_report",
    ]
    submitted_by: str = Field(min_length=1, max_length=200)
    notes: str = Field(min_length=1, max_length=4_000)
    source_filename: str = Field(min_length=1, max_length=300)
    media_type: str = Field(min_length=1, max_length=200)
    source_content_base64: str = Field(min_length=1, max_length=7_000_000)
    observed_results: list[ExperimentalResultRecord] = Field(
        default_factory=list,
        max_length=5_000,
    )
    observed_not_synthetic_attestation: bool = False


class ActionResolutionRequest(BaseModel):
    supporting_submission_ids: list[str] = Field(min_length=1, max_length=100)
    resolved_by: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=4_000)
    completion_attested: Literal[True]


class ReviewNoteRequest(BaseModel):
    researcher: str = Field(min_length=1, max_length=200)
    note: str = Field(min_length=1, max_length=4_000)


class G1ReauditRequest(BaseModel):
    base_verdict_id: str = Field(min_length=1, max_length=200)
    researcher: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=4_000)
    mode: Literal["rules", "external", "sources", "automatic"] = "rules"
    allow_paid: bool = False


_IDEMPOTENCY_KEY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._~:+/\-]{15,127}$"
)


def _canonical_create_request_sha256(request: CreateRunRequest) -> str:
    encoded = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _bounded_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not _IDEMPOTENCY_KEY_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Idempotency-Key must be 16-128 URL-safe ASCII characters"
        )
    return normalized


def _source_configuration(settings: Settings) -> SourceConfiguration:
    """Project settings reduced to non-secret source availability flags."""

    drugcentral_path = settings.drugcentral_dataset_path
    archs4_path = settings.archs4_dataset_path
    return SourceConfiguration(
        public_evidence_enabled=settings.public_evidence_enabled,
        public_candidate_enrichment_enabled=(
            settings.public_evidence_enabled
            and settings.public_candidate_enrichment_enabled
            and settings.public_candidate_enrichment_per_route > 0
        ),
        drugcentral_dataset_configured=drugcentral_path is not None,
        drugcentral_dataset_available=bool(
            drugcentral_path is not None and drugcentral_path.is_file()
        ),
        archs4_dataset_configured=archs4_path is not None,
        archs4_dataset_available=bool(
            archs4_path is not None and archs4_path.is_file()
        ),
        cellxgene_sidecar_configured=bool(settings.cellxgene_service_url),
        biogrid_credential_configured=bool(settings.biogrid_access_key),
        google_patents_project_configured=bool(settings.google_cloud_project),
        drugbank_credential_configured=bool(settings.drugbank_api_key),
    )


def _ma2_health_snapshot(
    orchestrator: DrugPilotOrchestrator,
    settings: Settings,
) -> dict:
    """Return a non-secret, typed view of the configured MA2 trust boundary."""

    gateway = orchestrator.ma2.gateway
    configured_fixture = settings.ma2_backend_mode == "fixture"
    build_status = getattr(gateway, "build_status", {})
    component_errors = (
        list(build_status.get("component_errors", []))
        if isinstance(build_status, dict)
        else []
    )
    summary = {
        "configured_mode": settings.ma2_backend_mode,
        "gateway_name": type(gateway).__name__,
        "fixture": bool(getattr(gateway, "fixture", configured_fixture)),
        "scientific_tools_requested": settings.ma2_live_scientific_tools_enabled,
        "scientific_delegate_configured": bool(
            isinstance(build_status, dict)
            and build_status.get("scientific_delegate_configured")
        ),
        "fail_closed": not configured_fixture,
        "fixture_fallback_authorized": configured_fixture,
        "max_refinement_iterations": orchestrator.ma2.max_refinement_iterations,
        "lifecycle_managed": True,
        "target_specific_readiness_evaluated": False,
        "component_initialization_failures": [
            str(item.get("component") or "unknown")
            for item in component_errors
            if isinstance(item, dict)
        ],
        "readiness": "unavailable",
        "capabilities": [],
    }
    declared = getattr(gateway, "readiness", None)
    if not callable(declared):
        summary["readiness_reason_code"] = "READINESS_CONTRACT_MISSING"
        return summary
    try:
        report = declared(None)
        typed = (
            report
            if isinstance(report, GatewayReadiness)
            else GatewayReadiness.model_validate(report)
        )
    except Exception as exc:
        # Exception class is enough to distinguish a contract/runtime failure;
        # do not echo paths, credentials, or third-party response bodies.
        summary["readiness_reason_code"] = "READINESS_CHECK_FAILED"
        summary["readiness_error_type"] = type(exc).__name__
        return summary

    capabilities = [
        {
            "tool_name": item.tool_name,
            "operation": item.operation,
            "status": item.status.value,
            "required": item.required,
            "backend_name": item.backend_name,
            "backend_version": item.backend_version,
            "fixture": item.fixture,
            "reason_code": item.reason_code,
            "input_requirements": item.input_requirements,
        }
        for item in typed.capabilities
    ]
    required_unready = [
        item for item in capabilities if item["required"] and item["status"] != "ready"
    ]
    optional_unready = [
        item
        for item in capabilities
        if not item["required"] and item["status"] != "ready"
    ]
    summary.update(
        {
            "gateway_name": typed.gateway_name,
            "source_version": typed.source_version,
            "fixture": typed.fixture,
            "readiness": (
                "unavailable"
                if required_unready
                else "partial"
                if optional_unready
                else "ready"
            ),
            "required_unavailable_count": len(required_unready),
            "optional_unavailable_count": len(optional_unready),
            "capabilities": capabilities,
        }
    )
    return summary


def create_app(
    settings: Settings | None = None,
    store: SQLiteRunStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    orchestrator = DrugPilotOrchestrator(settings=settings, store=store)
    scientific_probe_report: ScientificToolProbeReport | None = None

    def scientific_probe() -> ScientificToolProbeReport:
        """Probe once per server process; dependency changes require restart."""

        nonlocal scientific_probe_report
        if scientific_probe_report is None:
            scientific_probe_report = probe_scientific_tools(
                receptor_python_executable=settings.receptor_python_executable,
                rdkit_root=settings.receptor_rdkit_root,
                openmm_plugin_dir=settings.receptor_openmm_plugin_dir,
                meeko_receptor_executable=(
                    settings.receptor_meeko_receptor_executable
                ),
                meeko_ligand_executable=settings.receptor_meeko_ligand_executable,
                p2rank_executable=settings.receptor_p2rank_executable,
                p2rank_jar_path=settings.receptor_p2rank_jar_path,
                java_executable=settings.receptor_java_executable,
                vina_executable=settings.vina_binary,
                timeout_seconds=settings.receptor_probe_timeout_seconds,
            )
        return scientific_probe_report

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.orchestrator = orchestrator
        await orchestrator.recover_pending_cancellations()
        cancellation_recovery = asyncio.create_task(
            orchestrator.settle_pending_cancellations_after_restart(),
            name="drugpilot-cancellation-recovery",
        )
        try:
            yield
        finally:
            cancellation_recovery.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_recovery
            orchestrator.close()

    app = FastAPI(
        title="Omni-DrugPilot",
        version="0.1.0",
        description=(
            "Evidence-governed hypothesis and drug-candidate discovery with "
            "independent G0/G1 assurance gates."
        ),
        lifespan=lifespan,
    )
    run_admission_lock = asyncio.Lock()

    if settings.beta_access_auth_required:

        @app.middleware("http")
        async def beta_access_guard(request: Request, call_next):
            unauthorized = Response(
                content="Authentication required",
                status_code=401,
                headers={
                    "WWW-Authenticate": 'Basic realm="DrugPilot Beta", charset="UTF-8"',
                    "Cache-Control": "no-store",
                },
            )
            header = request.headers.get("authorization", "")
            if not header.startswith("Basic "):
                return unauthorized
            try:
                decoded = base64.b64decode(
                    header.removeprefix("Basic "), validate=True
                ).decode("utf-8")
                username, password = decoded.split(":", 1)
            except (binascii.Error, UnicodeDecodeError, ValueError):
                return unauthorized
            if not (
                secrets.compare_digest(
                    username, settings.beta_access_username or ""
                )
                and secrets.compare_digest(
                    password, settings.beta_access_password or ""
                )
            ):
                return unauthorized
            response = await call_next(request)
            response.headers["X-DrugPilot-Access-Control"] = "basic-auth"
            return response
    web_root = Path(__file__).resolve().parents[1] / "web"
    static_root = web_root / "static"
    if static_root.exists():
        app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(web_root / "templates" / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        public_http = get_scientific_http_client()
        source_inventory = source_status_inventory(_source_configuration(settings))
        ma2_health = _ma2_health_snapshot(orchestrator, settings)
        tool_report = scientific_probe()
        ma2_scientific_tools_enabled = bool(
            settings.ma2_backend_mode == "live"
            and settings.ma2_live_scientific_tools_enabled
        )
        receptor_profiles = receptor_profile_health(
            settings.docking_profile_registry_path,
            tool_report,
            receptor_builder_enabled=settings.receptor_builder_enabled,
            ma2_scientific_tools_enabled=ma2_scientific_tools_enabled,
        )
        scientific_tools = tool_report.health_payload(
            receptor_builder_enabled=settings.receptor_builder_enabled,
            ma2_scientific_tools_enabled=ma2_scientific_tools_enabled,
            ma2_capabilities=ma2_health.get("capabilities", []),
            accepted_profile_count=int(
                receptor_profiles.get("accepted_profile_count", 0)
            ),
        )
        candidate_public_enrichment = bool(
            settings.public_evidence_enabled
            and settings.public_candidate_enrichment_enabled
            and settings.public_candidate_enrichment_per_route > 0
        )
        data_backends = (
            ["PubMed", "Europe PMC", "Open Targets", "UniProt", "DGIdb"]
            if settings.public_evidence_enabled
            else ["fixture evidence"]
        )
        if candidate_public_enrichment:
            data_backends.extend(
                ["ChEMBL", "PubChem", "BindingDB", "RCSB PDB", "AlphaFold DB"]
            )
        if settings.dynamic_context_enabled and settings.public_evidence_enabled:
            data_backends.extend(
                [
                    "bioRxiv",
                    "medRxiv",
                    "ClinicalTrials.gov",
                    "GTEx",
                    "STRING",
                    "Reactome",
                    "KEGG",
                    "QuickGO",
                    "Human Cell Atlas",
                    "ARCHS4",
                ]
            )
            if settings.cellxgene_service_url:
                data_backends.append("CELLxGENE Census")
            if settings.biogrid_access_key:
                data_backends.append("BioGRID")
        if candidate_public_enrichment:
            if (
                settings.drugcentral_dataset_path is not None
                and settings.drugcentral_dataset_path.is_file()
            ):
                data_backends.append("DrugCentral")
            if settings.drugbank_api_key:
                data_backends.append("DrugBank")
            if settings.google_cloud_project:
                data_backends.append("Google Patents")
        if settings.ma2_backend_mode != "fixture":
            if (
                settings.drugcentral_dataset_path is not None
                and settings.drugcentral_dataset_path.is_file()
                and "DrugCentral" not in data_backends
            ):
                data_backends.append("DrugCentral")
            if settings.ma2_live_scientific_tools_enabled:
                # ChEMBL is a data source. RDKit, ADMET-AI and Vina are
                # computational tools and belong only in ``scientific_tools``.
                data_backends.append("ChEMBL")
        data_backends = list(dict.fromkeys(data_backends))
        research_profiles = {
            depth: settings.evidence_budget_for(depth).as_dict()
            for depth in ("focused", "standard", "comprehensive")
        }
        now = utc_now()
        admission_health = await orchestrator.store.admission_health(
            now=now,
            hourly_since=now - timedelta(hours=1),
        )
        hypothesis_review_limits = None
        if settings.ma1_provisional_admission_enabled:
            from drugpilot.agents.hypothesis_admission import (
                MAX_HYPOTHESES, MAX_RECORDS, MAX_SOURCE_CHARS, MAX_OUTPUT_TOKENS, PROMPT_VERSION,
            )
            from drugpilot.agents.evidence_semantics import MAX_ABSTRACT_CONTEXT_CHARS
            hypothesis_review_limits = {
                "prompt_version": PROMPT_VERSION,
                "max_hypotheses_per_packet": MAX_HYPOTHESES,
                "max_source_records_per_hypothesis": MAX_RECORDS,
                "max_source_characters_per_hypothesis": MAX_SOURCE_CHARS,
                "max_full_abstract_characters": MAX_ABSTRACT_CONTEXT_CHARS,
                "max_output_tokens_per_review": MAX_OUTPUT_TOKENS,
                "run_budget_preflight_still_required": True,
                "capacity_exhaustion_is_scientific_rejection": False,
            }
        return {
            "status": "ok",
            "build_variant": settings.build_variant,
            "hypothesis_admission": {
                "policy": ("ma3-source-bound-v1" if settings.ma1_provisional_admission_enabled
                           else "ma1-legacy-primary-gates"),
                "ma1_provisional_enabled": settings.ma1_provisional_admission_enabled,
                "planner_requires_ma3_approval": True,
                "review_limits": hypothesis_review_limits,
            },
            "access_control": {
                "authentication_required": settings.beta_access_auth_required,
                "scheme": "basic" if settings.beta_access_auth_required else "none",
                "max_new_runs_per_hour": settings.beta_max_new_runs_per_hour,
            },
            "run_admission": {
                "idempotency_required": (
                    settings.effective_mode is RunMode.LIVE
                ),
                "active_live_runs": admission_health["active_live_runs"],
                "expired_live_leases": admission_health[
                    "expired_live_leases"
                ],
                "recovery_required": admission_health["recovery_required"],
                "max_active_live_runs": settings.beta_max_active_live_runs,
                "hourly_admitted_count": admission_health[
                    "hourly_admitted_count"
                ],
                "hourly_admitted_cap": settings.beta_max_new_runs_per_hour,
            },
            "orchestration_backend": "langgraph",
            # ``mode`` remains the backward-compatible runtime value consumed
            # by the dashboard.  The explicit pair makes key-driven activation
            # observable without returning any credential material.
            "mode": settings.effective_mode.value,
            "configured_mode": settings.mode.value,
            "effective_mode": settings.effective_mode.value,
            "mode_auto_activated": settings.openai_auto_activated,
            "llm_backend": orchestrator.providers.component_backend,
            "reviewer_backend": orchestrator.providers.reviewer_backend,
            "openai_configured": bool(settings.openai_api_key),
            "reviewer_configured": (
                bool(settings.anthropic_api_key)
                or (
                    bool(settings.openai_api_key)
                    and settings.openai_ma3_reviewer_enabled
                )
            ),
            "reviewer_provider_independent": bool(settings.anthropic_api_key),
            "orchestrator_llm": False,
            "ma2": ma2_health,
            "scientific_tools": scientific_tools,
            "receptor_profiles": receptor_profiles,
            "public_evidence_enabled": settings.public_evidence_enabled,
            "direction_policy": {
                "version": "source-bound-direction-selection-v1",
                "allowed_hypothesis_directions": True,
                "search_remains_broad": True,
            },
            "automatic_resolution": {
                "enabled": settings.automatic_resolution_enabled,
                "external_review_enabled": settings.automatic_resolution_external_review_enabled,
                "native_live_no_fixture_only": True,
                "max_source_rounds": 1, "max_rules_reviews": 1, "max_external_reviews": 1,
                "elapsed_seconds": 600, "human_approval_generated": False,
            },
            # This is server configuration, not the profile used by every
            # run.  The authoritative per-run value is ResearchBrief.
            "evidence_research": {
                **settings.evidence_budget.as_dict(),
                "server_default": settings.evidence_research_depth,
                "available_profiles": [
                    "focused",
                    "standard",
                    "comprehensive",
                ],
                "resolved_profiles": research_profiles,
                "per_run_selection": True,
            },
            # Configuration only: this is not a claim that a remote database
            # answered.  Per-run call traces remain the reachability record.
            "public_http_transport": {
                "certificate_verification": True,
                "trust_backend": public_http.trust_backend,
                "proxy_mode": public_http.proxy_mode,
                "trust_env": public_http.trust_env,
                "max_attempts": public_http.max_attempts,
                "reachability_probed": False,
            },
            "dynamic_target_discovery": {
                "enabled": settings.dynamic_target_discovery_enabled,
                "semantic_interpretation_enabled": (
                    settings.ma1_semantic_interpretation_enabled
                    and not settings.ma1_provisional_admission_enabled
                ),
                "semantic_interpretation_policy": (
                    "delegated_to_ma3_source_bound_admission"
                    if settings.ma1_provisional_admission_enabled else "source_bound_semantics_v1"
                ),
                "semantic_agreement_bypasses_primary_gates": False,
                "max_candidates": settings.dynamic_target_max_candidates,
                "supplementary_evidence_retry_budget": (
                    settings.ma1_supplementary_evidence_retry_budget
                ),
                # A fallback seed may be prepared for component-level tests,
                # but it cannot authorize the orchestrated workflow unless
                # explicit demo mode is enabled.  The intake guard otherwise
                # fails closed before G0.
                "fallback_enabled": settings.dynamic_target_fallback_enabled,
                "demo_fixture_execution_authorized": (
                    settings.demo_fixture_fallback_enabled
                ),
                "fail_closed_intake_guard": True,
                "sources": ["Open Targets", "PubMed"],
            },
            "target_identity_resolution": {
                "enabled": settings.target_identity_resolution_enabled,
                "sources": ["Open Targets", "UniProt"],
            },
            "dynamic_context_routing": {
                "enabled": (
                    settings.dynamic_context_enabled
                    and settings.public_evidence_enabled
                ),
                "research_depth": settings.evidence_budget.depth,
                "profile_scope": "server_default_only",
                "targets_per_track": (
                    settings.evidence_budget.context_targets_per_track
                ),
                "max_source_calls": settings.evidence_budget.context_source_calls,
                "max_elapsed_seconds": (
                    settings.evidence_budget.context_elapsed_seconds
                ),
                "records_per_source": (
                    settings.evidence_budget.context_records_per_source
                ),
                "context_only": True,
            },
            "evidence_backend": (
                "public-live-fail-closed-no-fixture"
                if settings.public_evidence_enabled
                else "fixture"
            ),
            "candidate_backend": (
                (
                    "fixture computation + public DB enrichment"
                    if candidate_public_enrichment
                    else "fixture"
                )
                if settings.ma2_backend_mode == "fixture"
                else (
                    "live public/local retrieval + configured scientific tools"
                    if ma2_health["scientific_delegate_configured"]
                    else "live public/local retrieval; unavailable tools fail closed"
                )
            ),
            "data_backend": (
                " + ".join(
                    [
                        "public evidence"
                        if settings.public_evidence_enabled
                        else "fixture evidence",
                        (
                            "fixture candidate computation"
                            if settings.ma2_backend_mode == "fixture"
                            else "live candidate execution"
                        ),
                    ]
                )
            ),
            "data_backends": data_backends,
            "on_demand_data_backends": (
                [
                    "Pharos",
                    "bioRxiv",
                    "medRxiv",
                    "ClinicalTrials.gov",
                    "GTEx",
                    "ARCHS4",
                    "STRING",
                    "BioGRID",
                    "Reactome",
                    "KEGG",
                    "QuickGO",
                    "Human Cell Atlas",
                    "CELLxGENE Census",
                    "DrugCentral",
                    "DrugBank",
                    "Google Patents",
                ]
                if settings.public_evidence_enabled
                else []
            ),
            "source_registry": {
                **source_inventory["summary"],
                "endpoint": "/api/sources",
            },
            "checkpoint_store": "sqlite",
            "database_path": str(settings.database_path),
            "policy": {
                "max_run_tokens": settings.max_run_tokens,
                "hard_max_run_tokens": settings.hard_max_run_tokens,
                "max_llm_calls": settings.max_llm_calls,
                "max_single_input_tokens": settings.max_single_input_tokens,
                "max_run_cost_usd": settings.max_run_cost_usd,
            },
        }

    @app.get("/api/sources")
    async def sources() -> dict[str, object]:
        """List report sources and access paths without probing remote systems."""

        return source_status_inventory(_source_configuration(settings))

    @app.get(
        "/api/chemistry/structure.svg",
        response_class=Response,
        responses={
            200: {"content": {"image/svg+xml": {}}},
            400: {"description": "Invalid or unsupported SMILES"},
            503: {"description": "Local structure renderer unavailable"},
        },
    )
    def compound_structure_svg(request: Request, smiles: str) -> Response:
        """Render a bounded SMILES value locally; never substitute a fixture."""

        try:
            drawing = render_structure_svg(smiles)
        except InvalidStructureInput as exc:
            raise HTTPException(400, "invalid or unsupported SMILES") from exc
        except (StructureToolkitUnavailable, StructureRenderingError) as exc:
            # Do not expose import paths, parser diagnostics, or molecule input.
            raise HTTPException(503, "2D structure rendering is unavailable") from exc

        etag = f'"{drawing.etag}"'
        headers = {
            "Cache-Control": "private, max-age=86400, immutable",
            "Content-Disposition": 'inline; filename="compound-structure.svg"',
            "Content-Security-Policy": (
                "default-src 'none'; style-src 'unsafe-inline'; sandbox"
            ),
            "ETag": etag,
            "X-Content-Type-Options": "nosniff",
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(
            content=drawing.svg,
            media_type="image/svg+xml",
            headers=headers,
        )

    @app.post("/api/runs", response_model=RunSnapshot, status_code=202)
    async def create_run(
        request: CreateRunRequest,
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
        ),
    ) -> RunSnapshot:
        try:
            bounded_key = _bounded_idempotency_key(idempotency_key)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if settings.effective_mode is RunMode.LIVE and bounded_key is None:
            raise HTTPException(
                400,
                "Idempotency-Key is required for live run creation",
            )
        request_digest = _canonical_create_request_sha256(request)
        candidates = request.candidates_per_route or settings.candidates_per_route
        brief = ResearchBrief(
            research_goal=request.research_goal,
            disease=request.disease,
            disease_id=request.disease_id,
            rationale=request.rationale,
            research_depth=(
                request.research_depth or settings.evidence_research_depth
            ),
            constraints=ProjectConstraints(
                target_cell_type=request.target_cell_type,
                intervention_direction=request.intervention_direction,
                allowed_hypothesis_directions=request.allowed_hypothesis_directions,
                chronic_safety_required=request.chronic_safety_required,
                druggability_required=request.druggability_required,
                discovery_priority=request.discovery_priority,
                forbidden_targets=request.forbidden_targets,
                forbidden_substructures=request.forbidden_substructures,
                forbidden_drugs=request.forbidden_drugs,
                forbidden_drug_ids=request.forbidden_drug_ids,
                forbidden_drug_classes=request.forbidden_drug_classes,
                reserved_control_compounds=request.reserved_control_compounds,
                candidates_per_route=candidates,
                max_ma1_iterations=request.max_ma1_iterations
                or settings.max_ma1_iterations,
                max_rework_loops=request.max_rework_loops
                if request.max_rework_loops is not None
                else settings.max_rework_loops,
            ),
            budget=BudgetPolicy(
                max_run_tokens=settings.max_run_tokens,
                hard_max_run_tokens=settings.hard_max_run_tokens,
                max_llm_calls=settings.max_llm_calls,
                max_single_input_tokens=settings.max_single_input_tokens,
                max_run_cost_usd=settings.max_run_cost_usd,
            ),
            human_priorities=request.human_priorities,
            fault_injection=request.fault_injection,
        )
        async with run_admission_lock:
            if bounded_key is not None:
                existing = await orchestrator.store.get_run_admission(bounded_key)
                if existing is not None:
                    if existing["request_sha256"] != request_digest:
                        raise HTTPException(
                            409,
                            "Idempotency-Key is already bound to a different request",
                        )
                    snapshot = await orchestrator.store.get_snapshot(
                        existing["run_id"]
                    )
                    if snapshot is None:
                        raise HTTPException(
                            409,
                            "IDEMPOTENCY_MAPPING_ORPHANED: the mapped run is unavailable; operator recovery is required",
                        )
                    return snapshot

            try:
                return await orchestrator.create_run(
                    brief,
                    idempotency_key=bounded_key,
                    request_sha256=(
                        request_digest if bounded_key is not None else None
                    ),
                    admission_controlled=True,
                )
            except RunAdmissionLimitExceeded as exc:
                status_code = (
                    409
                    if exc.reason_code == "LIVE_LEASE_RECOVERY_REQUIRED"
                    else 429
                )
                headers = None
                if status_code == 429:
                    headers = {
                        "Retry-After": str(max(1, exc.retry_after_seconds or 1))
                    }
                raise HTTPException(
                    status_code, exc.reason_code, headers=headers
                ) from exc
            except RunAdmissionExists as exc:
                # A second process may have committed after our read.  The
                # durable UNIQUE key remains authoritative and is replayed only
                # for an identical canonical request.
                if exc.request_sha256 != request_digest:
                    raise HTTPException(
                        409,
                        "Idempotency-Key is already bound to a different request",
                    ) from exc
                snapshot = await orchestrator.store.get_snapshot(exc.run_id)
                if snapshot is None:
                    raise HTTPException(
                        409,
                        "IDEMPOTENCY_MAPPING_ORPHANED: the mapped run is unavailable; operator recovery is required",
                    ) from exc
                return snapshot

    @app.get("/api/runs", response_model=list[RunSnapshot])
    async def list_runs(limit: int = 25) -> list[RunSnapshot]:
        return await orchestrator.store.list_snapshots(max(1, min(limit, 100)))

    @app.get("/api/run-summaries")
    async def list_run_summaries(limit: int = 25) -> list[dict[str, Any]]:
        return await orchestrator.store.list_snapshot_summaries(
            max(1, min(limit, 100))
        )

    @app.get("/api/runs/{run_id}", response_model=RunSnapshot)
    async def get_run(run_id: str) -> RunSnapshot:
        snapshot = await orchestrator.store.get_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(404, "run not found")
        return snapshot

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        if await orchestrator.store.get_snapshot(run_id) is None:
            raise HTTPException(404, "run not found")
        return StreamingResponse(
            orchestrator.broker.stream(run_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/runs/{run_id}/cancel-status")
    async def cancel_status(run_id: str) -> dict[str, Any]:
        try:
            return await orchestrator.cancellation_status(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc

    @app.post("/api/runs/{run_id}/cancel", response_model=RunSnapshot, status_code=202)
    async def cancel_run(run_id: str) -> RunSnapshot:
        """Request cooperative cancellation, preserving in-flight accounting."""
        try:
            return await orchestrator.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/runs/{run_id}/artifacts")
    async def artifact_manifest(run_id: str) -> dict:
        if await orchestrator.store.get_snapshot(run_id) is None:
            raise HTTPException(404, "run not found")
        return {
            "run_id": run_id,
            "ledger_valid": await orchestrator.store.verify_ledger(run_id),
            "artifacts": await orchestrator.store.artifact_manifest(run_id),
            "stage_checkpoints": await orchestrator.store.checkpoint_manifest(run_id),
        }

    @app.get("/api/runs/{run_id}/calls", response_model=RunCallTrace)
    async def call_trace(run_id: str) -> RunCallTrace:
        """Return a normalized, read-only model/public-data/tool call ledger."""

        snapshot = await orchestrator.store.get_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(404, "run not found")
        return build_run_call_trace(snapshot)

    @app.get("/api/runs/{run_id}/result-readiness")
    async def result_readiness(run_id: str) -> dict[str, Any]:
        try:
            return await ResearchSelectionService(orchestrator).readiness(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc

    @app.post("/api/runs/{run_id}/research-selection", response_model=RunSnapshot)
    async def research_selection(run_id: str, request: ResearchSelectionRequest) -> RunSnapshot:
        try:
            return await ResearchSelectionService(orchestrator).select(run_id, **request.model_dump())
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except (ValueError, CheckpointIntegrityError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/runs/{run_id}/human-decision", response_model=RunSnapshot)
    async def human_decision(run_id: str, request: HumanDecisionRequest) -> RunSnapshot:
        try:
            return await orchestrator.apply_human_decision(
                HumanGateDecision(run_id=run_id, **request.model_dump())
            )
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/runs/{run_id}/next-actions", response_model=RunSnapshot)
    async def materialize_next_actions(run_id: str) -> RunSnapshot:
        """Idempotently attach an evidence-preserving handoff to a stopped run."""

        try:
            return await orchestrator.materialize_next_actions(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post(
        "/api/runs/{run_id}/actions/{action_id}/evidence",
        response_model=RunSnapshot,
    )
    async def submit_action_evidence(
        run_id: str,
        action_id: str,
        request: ActionEvidenceRequest,
    ) -> RunSnapshot:
        """Attach a real source artifact and typed evidence rows to one action."""

        try:
            return await orchestrator.submit_action_evidence(
                run_id,
                action_id=action_id,
                **request.model_dump(),
            )
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post(
        "/api/runs/{run_id}/actions/{action_id}/resolve",
        response_model=RunSnapshot,
    )
    async def resolve_next_action(
        run_id: str,
        action_id: str,
        request: ActionResolutionRequest,
    ) -> RunSnapshot:
        """Record attested completion; approval still requires a fresh G1 pass."""

        payload = request.model_dump(exclude={"completion_attested"})
        try:
            return await orchestrator.resolve_next_action(
                run_id,
                action_id=action_id,
                **payload,
            )
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/runs/{run_id}/review-status")
    async def review_status(run_id: str) -> dict:
        try:
            return await HumanReviewService(orchestrator).status(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc

    @app.get("/api/runs/{run_id}/attachments/{attachment_id}")
    async def download_evidence_attachment(run_id: str, attachment_id: str) -> Response:
        snapshot = await orchestrator.store.get_snapshot(run_id)
        if snapshot is None or not any(s.attachment_id == attachment_id for s in snapshot.action_evidence_submissions):
            raise HTTPException(404, "attachment not found")
        try:
            artifact = await orchestrator.store.read_artifact(run_id, attachment_id, "EvidenceAttachmentArtifact")
            if artifact is None:
                raise ValueError("missing artifact")
            content = base64.b64decode(artifact["content_base64"], validate=True)
            if sha256(content).hexdigest() != artifact["sha256"]:
                raise ValueError("hash mismatch")
        except Exception as exc:
            raise HTTPException(409, "Attachment integrity check failed") from exc
        return Response(content, media_type="application/octet-stream", headers={
            "Content-Disposition": "attachment; filename*=UTF-8''" + quote(artifact["filename"], safe=""),
            "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})

    @app.post("/api/runs/{run_id}/review-notes", response_model=RunSnapshot)
    async def review_note(run_id: str, request: ReviewNoteRequest) -> RunSnapshot:
        try:
            return await HumanReviewService(orchestrator).note(run_id, **request.model_dump())
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/runs/{run_id}/reaudits", response_model=RunSnapshot, status_code=202)
    async def start_reaudit(run_id: str, request: G1ReauditRequest,
                           idempotency_key: str | None = Header(default=None)) -> RunSnapshot:
        try:
            key = _bounded_idempotency_key(idempotency_key)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if key is None:
            raise HTTPException(400, "Idempotency-Key is required")
        try:
            return await HumanReviewService(orchestrator).start(
                run_id, request_id=key, **request.model_dump())
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/runs/{run_id}/reaudits/recover", response_model=RunSnapshot)
    async def recover_reaudit(run_id: str) -> RunSnapshot:
        try:
            return await HumanReviewService(orchestrator).recover(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/runs/{run_id}/resume-status")
    async def resume_status(run_id: str) -> dict[str, object]:
        """Expose resume eligibility without leaking prompts or private state."""

        try:
            return await orchestrator.resume_status(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc

    @app.post("/api/runs/{run_id}/resume", response_model=RunSnapshot, status_code=202)
    async def resume_run(run_id: str) -> RunSnapshot:
        try:
            return await orchestrator.resume(run_id)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    return app


app = create_app()
