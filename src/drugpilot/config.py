from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from drugpilot.domain.models import ResearchDepth, RunMode


@dataclass(frozen=True, slots=True)
class EvidenceResearchBudget:
    """Resolved, auditable retrieval limits for one MA1 research run.

    ``focused`` preserves the former interactive caps. ``comprehensive`` uses
    the operator-configured ceilings below.  The profile changes breadth and
    resilience only; it never relaxes source allow-lists, applicability rules,
    provenance requirements, or the no-fixture fail-closed policy.
    """

    depth: ResearchDepth
    core_records_per_source: int
    core_source_calls_per_target: int
    core_elapsed_seconds_per_target: float
    core_timeout_seconds: float
    core_max_attempts: int
    retry_backoff_seconds: float
    target_pubmed_records: int
    target_pubmed_supplements: int
    target_lookup_limit: int
    target_max_candidates: int
    context_targets_per_track: int
    context_source_calls: int
    context_elapsed_seconds: float
    context_records_per_source: int
    context_timeout_seconds: float
    pharos_targets_per_track: int
    candidate_enrichment_per_route: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DRUGPILOT_",
        extra="ignore",
        case_sensitive=False,
    )

    mode: RunMode = RunMode.MOCK
    build_variant: str = "llm-intake-validation-test"
    # Keep the checked-in/demo configuration safe by default, while allowing a
    # server restart after adding only OPENAI_API_KEY to activate real model
    # calls.  The resulting workflow is HYBRID because scientific data/tools
    # can still be deterministic fixtures even when the LLM is live.
    auto_activate_openai: bool = True
    openai_ma3_reviewer_enabled: bool = True
    # Complete configured remediation paths before presenting the final review
    # result. Only paid LIVE/no-fixture workflows run this automatically.
    automatic_resolution_enabled: bool = True
    automatic_resolution_external_review_enabled: bool = True
    database_path: Path = Path("data/drugpilot.sqlite3")
    mock_step_delay_ms: int = Field(default=250, ge=0, le=5_000)
    # MA2 is selected independently of LLM mode. ``fixture`` is the explicit
    # repeatable demo/test backend; ``live`` disables every synthetic fallback
    # and fails closed when a source or executable is not configured.
    ma2_backend_mode: Literal["fixture", "live"] = "fixture"
    ma2_live_scientific_tools_enabled: bool = False
    ma2_live_allow_experimental_docking_profiles: bool = False
    # ChEMBL candidate retrieval is materially slower than MA1 evidence
    # lookups on some institutional networks. Keep an independent, bounded
    # budget so the MA1 interactive timeout cannot truncate MA2 chemistry.
    ma2_chembl_timeout_seconds: float = Field(default=30.0, ge=5.0, le=60.0)
    ma2_chembl_operation_timeout_seconds: float = Field(
        default=120.0, ge=20.0, le=900.0
    )
    ma2_chembl_max_attempts: int = Field(default=2, ge=1, le=3)
    ma2_chembl_retry_backoff_seconds: float = Field(
        default=0.25, ge=0.0, le=2.0
    )
    docking_profile_registry_path: Path | None = None
    admet_python_executable: Path | None = None
    # Receptor-profile construction is an offline, fail-closed workflow.  It
    # is deliberately independent of MA2 execution and disabled until an
    # operator explicitly enables it.  The Python interpreter and every
    # executable are probed out of process so the health endpoint describes
    # the configured environment rather than whichever packages happen to be
    # importable by an unrelated process.
    receptor_builder_enabled: bool = False
    receptor_python_executable: Path | None = None
    receptor_rdkit_root: Path | None = None
    receptor_openmm_plugin_dir: Path | None = None
    receptor_meeko_receptor_executable: Path | None = None
    receptor_meeko_ligand_executable: Path | None = None
    receptor_p2rank_executable: Path | None = None
    receptor_p2rank_jar_path: Path | None = None
    receptor_java_executable: Path | None = None
    receptor_probe_timeout_seconds: float = Field(default=10.0, ge=1.0, le=30.0)
    # Disabled by default to preserve the legacy single-target component
    # contract.  The research service enables this explicitly so every
    # G0-approved, identity-resolved target is attempted independently.
    ma2_portfolio_enabled: bool = False

    # Optional shared access boundary for a LAN/review deployment.  Disabled
    # for the developer server; the stable beta enables it explicitly.
    beta_access_auth_required: bool = False
    beta_access_username: str | None = Field(default=None, max_length=100)
    beta_access_password: str | None = Field(
        default=None,
        min_length=12,
        max_length=300,
        repr=False,
        exclude=True,
    )
    # Paid live runs are admitted independently of the optional LAN access
    # guard.  Keeping the default at one prevents several full-budget runs
    # from being launched concurrently by retries or multiple browser tabs.
    beta_max_active_live_runs: int = Field(default=1, ge=1, le=20)
    beta_max_new_runs_per_hour: int = Field(default=3, ge=1, le=20)
    # A durable SQLite lease is the cross-worker authority for live execution.
    # The TTL exceeds the longest configured blocking ChEMBL operation; the
    # event-loop heartbeat normally renews it far earlier.
    beta_live_lease_ttl_seconds: int = Field(default=900, ge=60, le=3_600)
    beta_live_lease_heartbeat_seconds: int = Field(default=30, ge=5, le=300)

    terra_model: str = "gpt-5.6-terra"
    sol_model: str = "gpt-5.6-sol"
    luna_model: str = "gpt-5.6-luna"
    reviewer_model: str = "claude-opus-4-1"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        repr=False,
        exclude=True,
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        repr=False,
        exclude=True,
    )

    # Optional source-specific configuration.  Credentials are deliberately
    # excluded from repr/model_dump so the health and source-registry APIs can
    # expose only boolean configuration state, never the secret value itself.
    ncbi_api_key: str | None = Field(
        default=None,
        validation_alias="NCBI_API_KEY",
        repr=False,
        exclude=True,
    )
    ncbi_email: str | None = None
    ncbi_tool: str = "DrugPilot"
    biogrid_access_key: str | None = Field(
        default=None,
        validation_alias="BIOGRID_ACCESS_KEY",
        repr=False,
        exclude=True,
    )
    drugbank_api_key: str | None = Field(
        default=None,
        validation_alias="DRUGBANK_API_KEY",
        repr=False,
        exclude=True,
    )
    drugcentral_dataset_path: Path | None = None
    drugcentral_dataset_version: str | None = None
    archs4_dataset_path: Path | None = None
    cellxgene_service_url: str | None = None
    google_cloud_project: str | None = None

    max_run_tokens: int = 1_000_000
    hard_max_run_tokens: int = 5_000_000
    max_llm_calls: int = 72
    max_single_input_tokens: int = 250_000
    max_run_cost_usd: float = 46.67
    candidates_per_route: int = 20
    max_ma1_iterations: int = 3
    max_rework_loops: int = 2

    dgidb_mcp_url: str | None = None
    vina_binary: Path | None = None
    public_evidence_enabled: bool = True
    # Semantic agreement is an additional check; it never bypasses primary gates.
    ma1_semantic_interpretation_enabled: bool = True
    # Opt-in staged policy: MA1 submits source-bound proposals; only MA3 may
    # promote them into the executable hypothesis list.
    ma1_provisional_admission_enabled: bool = False
    public_evidence_timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    # Evidence breadth is deliberately independent of LLM credentials.  A
    # live LLM can reason only over the evidence packet it receives, so the
    # comprehensive profile expands bounded retrieval while retaining the
    # same deterministic data policy and scientific audit boundary.
    evidence_research_depth: ResearchDepth = "comprehensive"
    public_evidence_max_records_per_source: int = Field(default=8, ge=1, le=20)
    public_evidence_max_source_calls_per_target: int = Field(
        default=5, ge=1, le=6
    )
    public_evidence_max_elapsed_seconds_per_target: float = Field(
        default=60.0, ge=5.0, le=300.0
    )
    public_evidence_max_attempts: int = Field(default=2, ge=1, le=3)
    public_evidence_retry_backoff_seconds: float = Field(
        default=0.2, ge=0.0, le=2.0
    )
    # Optional Europe PMC OA JATS escalation. Research depth changes bounded
    # breadth, not whether an otherwise eligible OA primary source may be
    # verified. The retriever still requires the existing abstract pass to be
    # insufficient before making any full-text or one-hop reference request.
    # Keeping the feature off by default preserves abstract-only deployments.
    epmc_oa_fulltext_one_hop_v1: bool = False
    public_candidate_enrichment_enabled: bool = True
    public_candidate_enrichment_per_route: int = Field(default=5, ge=0, le=5)
    dynamic_target_discovery_enabled: bool = True
    dynamic_target_max_candidates: int = Field(default=8, ge=4, le=30)
    # Rework preserves the full fresh-target discovery window and may add this
    # many evidence-only retries on top.  Keeping a separate, small operator
    # bound prevents retry starvation without silently replacing new targets
    # or turning G0 feedback into an unbounded retrieval loop.
    ma1_supplementary_evidence_retry_budget: int = Field(
        default=1, ge=0, le=4
    )
    target_discovery_pubmed_max_records: int = Field(default=40, ge=1, le=100)
    target_discovery_max_pubmed_supplements: int = Field(default=6, ge=0, le=20)
    target_discovery_target_lookup_limit: int = Field(default=8, ge=0, le=30)
    dynamic_target_fallback_enabled: bool = True
    # Fixture targets are useful for isolated component demonstrations, but
    # they must never turn a failed external disease lookup into an apparently
    # evidence-backed research run.  Enabling this flag is an explicit demo
    # override; the normal web workflow remains fail-closed.
    demo_fixture_fallback_enabled: bool = False
    target_identity_resolution_enabled: bool = True
    dynamic_context_enabled: bool = True
    dynamic_context_targets_per_track: int = Field(default=2, ge=0, le=3)
    dynamic_context_max_calls: int = Field(default=36, ge=1, le=100)
    dynamic_context_max_seconds: float = Field(default=90.0, ge=1.0, le=600.0)
    dynamic_context_limit_per_source: int = Field(default=5, ge=1, le=10)
    pharos_on_demand_targets_per_track: int = Field(default=1, ge=0, le=3)

    @property
    def evidence_budget(self) -> EvidenceResearchBudget:
        """Resolve the operator-default profile for backward compatibility."""

        return self.evidence_budget_for(self.evidence_research_depth)

    def evidence_budget_for(
        self, depth: ResearchDepth | None
    ) -> EvidenceResearchBudget:
        """Resolve one run's immutable profile inside operator ceilings.

        This is a pure projection: it never mutates ``Settings`` or a shared
        retriever.  Callers can therefore execute different profiles
        concurrently without one run changing another run's limits.
        """

        effective_depth = depth or self.evidence_research_depth
        if effective_depth == "focused":
            return EvidenceResearchBudget(
                depth="focused",
                core_records_per_source=min(
                    self.public_evidence_max_records_per_source, 2
                ),
                core_source_calls_per_target=min(
                    self.public_evidence_max_source_calls_per_target, 5
                ),
                core_elapsed_seconds_per_target=min(
                    self.public_evidence_max_elapsed_seconds_per_target, 12.0
                ),
                core_timeout_seconds=min(
                    self.public_evidence_timeout_seconds, 2.0
                ),
                core_max_attempts=1,
                retry_backoff_seconds=0.0,
                target_pubmed_records=min(
                    self.target_discovery_pubmed_max_records, 10
                ),
                target_pubmed_supplements=min(
                    self.target_discovery_max_pubmed_supplements, 2
                ),
                target_lookup_limit=min(
                    self.target_discovery_target_lookup_limit, 4
                ),
                target_max_candidates=min(self.dynamic_target_max_candidates, 4),
                context_targets_per_track=min(
                    self.dynamic_context_targets_per_track, 1
                ),
                context_source_calls=min(self.dynamic_context_max_calls, 8),
                context_elapsed_seconds=min(
                    self.dynamic_context_max_seconds, 12.0
                ),
                context_records_per_source=min(
                    self.dynamic_context_limit_per_source, 2
                ),
                context_timeout_seconds=min(
                    self.public_evidence_timeout_seconds, 3.0
                ),
                pharos_targets_per_track=0,
                candidate_enrichment_per_route=(
                    min(self.public_candidate_enrichment_per_route, 1)
                    if self.public_evidence_enabled
                    and self.public_candidate_enrichment_enabled
                    else 0
                ),
            )
        if effective_depth == "standard":
            return EvidenceResearchBudget(
                depth="standard",
                core_records_per_source=min(
                    self.public_evidence_max_records_per_source, 5
                ),
                core_source_calls_per_target=min(
                    self.public_evidence_max_source_calls_per_target, 5
                ),
                core_elapsed_seconds_per_target=min(
                    self.public_evidence_max_elapsed_seconds_per_target, 30.0
                ),
                core_timeout_seconds=min(self.public_evidence_timeout_seconds, 5.0),
                core_max_attempts=min(self.public_evidence_max_attempts, 2),
                retry_backoff_seconds=self.public_evidence_retry_backoff_seconds,
                target_pubmed_records=min(
                    self.target_discovery_pubmed_max_records, 25
                ),
                target_pubmed_supplements=min(
                    self.target_discovery_max_pubmed_supplements, 4
                ),
                target_lookup_limit=min(
                    self.target_discovery_target_lookup_limit, 6
                ),
                target_max_candidates=min(self.dynamic_target_max_candidates, 6),
                context_targets_per_track=min(
                    self.dynamic_context_targets_per_track, 1
                ),
                context_source_calls=min(self.dynamic_context_max_calls, 20),
                context_elapsed_seconds=min(
                    self.dynamic_context_max_seconds, 45.0
                ),
                context_records_per_source=min(
                    self.dynamic_context_limit_per_source, 3
                ),
                context_timeout_seconds=min(
                    self.public_evidence_timeout_seconds, 5.0
                ),
                pharos_targets_per_track=min(
                    self.pharos_on_demand_targets_per_track, 1
                ) if self.public_evidence_enabled else 0,
                candidate_enrichment_per_route=(
                    min(self.public_candidate_enrichment_per_route, 3)
                    if self.public_evidence_enabled
                    and self.public_candidate_enrichment_enabled
                    else 0
                ),
            )
        return EvidenceResearchBudget(
            depth="comprehensive",
            core_records_per_source=self.public_evidence_max_records_per_source,
            core_source_calls_per_target=(
                self.public_evidence_max_source_calls_per_target
            ),
            core_elapsed_seconds_per_target=(
                self.public_evidence_max_elapsed_seconds_per_target
            ),
            core_timeout_seconds=self.public_evidence_timeout_seconds,
            core_max_attempts=self.public_evidence_max_attempts,
            retry_backoff_seconds=self.public_evidence_retry_backoff_seconds,
            target_pubmed_records=self.target_discovery_pubmed_max_records,
            target_pubmed_supplements=(
                self.target_discovery_max_pubmed_supplements
            ),
            target_lookup_limit=self.target_discovery_target_lookup_limit,
            target_max_candidates=self.dynamic_target_max_candidates,
            context_targets_per_track=self.dynamic_context_targets_per_track,
            context_source_calls=self.dynamic_context_max_calls,
            context_elapsed_seconds=self.dynamic_context_max_seconds,
            context_records_per_source=self.dynamic_context_limit_per_source,
            context_timeout_seconds=self.public_evidence_timeout_seconds,
            pharos_targets_per_track=(
                self.pharos_on_demand_targets_per_track
                if self.public_evidence_enabled
                else 0
            ),
            candidate_enrichment_per_route=(
                self.public_candidate_enrichment_per_route
                if self.public_evidence_enabled
                and self.public_candidate_enrichment_enabled
                else 0
            ),
        )

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "ncbi_api_key",
        "biogrid_access_key",
        "drugbank_api_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_api_key(cls, value: object) -> str | None:
        """Treat blank environment values as unconfigured without exposing them."""

        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator(
        "ncbi_email",
        "drugcentral_dataset_version",
        "cellxgene_service_url",
        "google_cloud_project",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator(
        "drugcentral_dataset_path",
        "archs4_dataset_path",
        "vina_binary",
        "docking_profile_registry_path",
        "admet_python_executable",
        "receptor_python_executable",
        "receptor_rdkit_root",
        "receptor_openmm_plugin_dir",
        "receptor_meeko_receptor_executable",
        "receptor_meeko_ligand_executable",
        "receptor_p2rank_executable",
        "receptor_p2rank_jar_path",
        "receptor_java_executable",
        mode="before",
    )
    @classmethod
    def normalize_optional_path(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def effective_mode(self) -> RunMode:
        """Resolve the actual run mode after applying key-based activation."""

        if (
            self.mode is RunMode.MOCK
            and self.auto_activate_openai
            and self.openai_api_key is not None
        ):
            return RunMode.HYBRID
        return self.mode

    @property
    def openai_auto_activated(self) -> bool:
        return self.mode is RunMode.MOCK and self.effective_mode is RunMode.HYBRID

    @model_validator(mode="after")
    def validate_live_keys(self) -> "Settings":
        if self.mode is RunMode.LIVE and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when DRUGPILOT_MODE=live")
        if self.max_run_tokens > self.hard_max_run_tokens:
            raise ValueError("max_run_tokens must not exceed hard_max_run_tokens")
        if self.beta_live_lease_heartbeat_seconds * 2 >= self.beta_live_lease_ttl_seconds:
            raise ValueError(
                "live lease heartbeat must be less than half of the lease TTL"
            )
        if self.beta_access_auth_required and (
            not self.beta_access_username or not self.beta_access_password
        ):
            raise ValueError(
                "beta access authentication requires a username and 12+ character password"
            )
        if self.ma2_live_scientific_tools_enabled and self.ma2_backend_mode != "live":
            raise ValueError(
                "ma2_live_scientific_tools_enabled requires ma2_backend_mode=live"
            )
        if (
            self.ma2_chembl_operation_timeout_seconds
            < self.ma2_chembl_timeout_seconds
        ):
            raise ValueError(
                "ma2_chembl_operation_timeout_seconds must be greater than or "
                "equal to ma2_chembl_timeout_seconds"
            )
        if self.ma2_live_allow_experimental_docking_profiles:
            variant = self.build_variant.casefold()
            if not self.ma2_live_scientific_tools_enabled or not any(
                marker in variant for marker in ("test", "demo", "development", "dev")
            ):
                raise ValueError(
                    "experimental docking profiles require the live scientific-tools "
                    "flag and an explicit test/demo/development build variant"
                )
        return self

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
