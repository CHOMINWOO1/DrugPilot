from __future__ import annotations

from typing import Any

import httpx
import pytest

from drugpilot.agents.ma1 import TargetSeed
from drugpilot.config import Settings
from drugpilot.domain.models import HypothesisTrack, ResearchBrief
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.tools.context_router import DiseaseTargetContextRouter
from drugpilot.tools.disease_targets import DiseaseTargetDiscovery
from drugpilot.tools.evidence import ResilientEvidenceRetriever
from drugpilot.tools.public_biology import PHAROS_GRAPHQL_URL, PublicBiologyRetriever
from drugpilot.tools.public_chemistry import PublicChemistryEnricher
from drugpilot.tools.target_identity import TargetIdentityResolver


class _FailingRecordingClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> None:
        del kwargs
        self.urls.append(url)
        raise httpx.ConnectTimeout("offline", request=httpx.Request("GET", url))

    def post(self, url: str, **kwargs: Any) -> None:
        del kwargs
        self.urls.append(url)
        raise httpx.ConnectTimeout("offline", request=httpx.Request("POST", url))


def test_orchestrator_wires_comprehensive_evidence_limits(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "wiring.sqlite3",
        public_evidence_enabled=True,
        public_evidence_timeout_seconds=9.0,
        evidence_research_depth="comprehensive",
        public_evidence_max_records_per_source=8,
        public_evidence_max_source_calls_per_target=5,
        public_evidence_max_elapsed_seconds_per_target=60,
        public_evidence_max_attempts=2,
        public_evidence_retry_backoff_seconds=0.2,
        target_discovery_pubmed_max_records=40,
        target_discovery_max_pubmed_supplements=6,
        target_discovery_target_lookup_limit=8,
        dynamic_context_targets_per_track=2,
        dynamic_context_max_calls=36,
        dynamic_context_max_seconds=90,
        dynamic_context_limit_per_source=5,
        ncbi_api_key="test-ncbi-key",
        ncbi_email="team@example.test",
        ncbi_tool="DrugPilotTest",
        biogrid_access_key="A" * 32,
        cellxgene_service_url="https://cell-sidecar.example.test",
    )

    orchestrator = DrugPilotOrchestrator(settings=settings)
    retriever = orchestrator.ma1.proposer.retriever

    assert isinstance(retriever, ResilientEvidenceRetriever)
    assert retriever.fixture_fallback_enabled is False
    assert isinstance(retriever.public, PublicBiologyRetriever)
    assert retriever.public.timeout_seconds == 9.0
    assert retriever.public.max_records_per_source == 8
    assert retriever.public.max_source_calls == 5
    assert retriever.public.max_elapsed_seconds == 60
    assert retriever.public.max_attempts == 2
    assert retriever.public.retry_backoff_seconds == 0.2
    assert retriever.public.ncbi_api_key == "test-ncbi-key"
    assert retriever.public.ncbi_email == "team@example.test"
    assert retriever.public.ncbi_tool == "DrugPilotTest"
    assert retriever.public.counter_evidence_enabled is True
    assert retriever.public.enabled_sources == (
        "PubMed",
        "Europe PMC",
        "Open Targets",
        "UniProt",
        "DGIdb",
    )
    assert isinstance(orchestrator.ma2.evidence_enricher, PublicChemistryEnricher)
    assert orchestrator.ma2.enrichment_per_route == 5
    assert isinstance(orchestrator.ma1.target_discovery, DiseaseTargetDiscovery)
    assert orchestrator.ma1.target_discovery.pubmed_max_records == 40
    assert orchestrator.ma1.target_discovery.max_pubmed_supplements == 6
    assert orchestrator.ma1.target_discovery.target_lookup_limit == 8
    assert isinstance(orchestrator.ma1.identity_resolver, TargetIdentityResolver)
    assert isinstance(orchestrator.context_router, DiseaseTargetContextRouter)
    assert isinstance(orchestrator.pharos_router.retriever, PublicBiologyRetriever)
    assert orchestrator.pharos_router.retriever.enabled_sources == ("Pharos",)
    assert orchestrator.pharos_router.max_targets_per_track == 1
    context_retriever = orchestrator.context_router.retriever
    assert (
        context_retriever.cell_data_retriever.cellxgene_sidecar_url
        == "https://cell-sidecar.example.test"
    )
    assert context_retriever.expression_interaction_retriever.biogrid_configuration_error is None
    assert context_retriever.timeout_seconds == 9.0


def test_focused_profile_preserves_legacy_interactive_caps(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "focused.sqlite3",
        public_evidence_enabled=True,
        evidence_research_depth="focused",
        public_evidence_timeout_seconds=20,
        public_evidence_max_records_per_source=20,
        public_evidence_max_attempts=3,
        target_discovery_pubmed_max_records=100,
        target_discovery_max_pubmed_supplements=20,
        target_discovery_target_lookup_limit=30,
        dynamic_context_targets_per_track=3,
        dynamic_context_max_calls=100,
        dynamic_context_max_seconds=600,
        dynamic_context_limit_per_source=10,
    )

    orchestrator = DrugPilotOrchestrator(settings=settings)
    public = orchestrator.ma1.proposer.retriever.public
    discovery = orchestrator.ma1.target_discovery
    budget = settings.evidence_budget

    assert public.timeout_seconds == 2.0
    assert public.max_records_per_source == 2
    assert public.max_source_calls == 5
    assert public.max_elapsed_seconds == 12.0
    assert public.max_attempts == 1
    assert public.retry_backoff_seconds == 0
    assert discovery.pubmed_max_records == 10
    assert discovery.max_pubmed_supplements == 2
    assert discovery.target_lookup_limit == 4
    assert orchestrator.context_router.retriever.timeout_seconds == 3.0
    assert budget.context_targets_per_track == 1
    assert budget.context_source_calls == 8
    assert budget.context_elapsed_seconds == 12.0
    assert budget.context_records_per_source == 2


def test_orchestrator_default_public_retriever_never_calls_pharos(tmp_path) -> None:
    settings = Settings(
        database_path=tmp_path / "no-pharos.sqlite3",
        public_evidence_enabled=True,
    )
    orchestrator = DrugPilotOrchestrator(settings=settings)
    public = orchestrator.ma1.proposer.retriever.public
    client = _FailingRecordingClient()
    public.client = client

    brief = ResearchBrief(
        research_goal="Find traceable evidence for a target hypothesis.",
        disease="Acute myeloid leukemia",
    )
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
    )

    with pytest.raises(RuntimeError):
        public.retrieve(brief, seed, 1)

    assert PHAROS_GRAPHQL_URL not in client.urls
    assert len(set(client.urls)) == 5
    assert len(client.urls) == 5 * settings.evidence_budget.core_max_attempts


def test_orchestrator_uses_fixture_retriever_when_public_evidence_is_disabled(
    tmp_path,
) -> None:
    settings = Settings(
        database_path=tmp_path / "offline.sqlite3",
        public_evidence_enabled=False,
    )

    orchestrator = DrugPilotOrchestrator(settings=settings)

    assert not isinstance(
        orchestrator.ma1.proposer.retriever,
        ResilientEvidenceRetriever,
    )
    assert orchestrator.ma2.evidence_enricher is None
    assert orchestrator.ma2.enrichment_per_route == 0
    assert orchestrator.ma1.target_discovery is None
    assert orchestrator.ma1.identity_resolver is None
    assert orchestrator.context_router is None
    assert orchestrator.pharos_router.retriever is None
