from __future__ import annotations

from typing import Any

from drugpilot.domain.models import (
    CandidateCard,
    CandidateRoute,
    Claim,
    Hypothesis,
    HypothesisTrack,
    TargetCandidate,
)
from drugpilot.tools.context_router import (
    ContextRoutingRequest,
    ContextTarget,
    DiseaseTargetContextRouter,
)
from drugpilot.tools.disease_targets import DiseaseTargetDiscovery
from drugpilot.tools.public_chemistry import ChemistryLookup, PublicChemistryEnricher
from drugpilot.tools.restricted_candidate_enrichment import (
    RoutedRestrictedCandidateEnricher,
)
from drugpilot.tools.target_identity import TargetIdentityResolver


class NeverHTTPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        self.calls.append(("GET", url))
        raise AssertionError("a policy-rejected source issued HTTP I/O")

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append(("POST", url))
        raise AssertionError("a policy-rejected source issued HTTP I/O")


class NeverContextRetriever:
    def __init__(self) -> None:
        self.calls = []

    def lookup(self, request):
        self.calls.append(request)
        raise AssertionError("a policy-rejected context source was queried")


class NeverRestrictedAdapter:
    documentation_url = "https://example.test/docs"
    required_configuration: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __init__(self, source: str) -> None:
        self.source = source
        self.calls = []

    def query(self, request):
        self.calls.append(request)
        raise AssertionError("a policy-rejected restricted source was queried")


def _hypothesis() -> Hypothesis:
    target = TargetCandidate(
        symbol="FLT3",
        ensembl_id="ENSG00000122025",
        uniprot_id="P36888",
        target_name="FLT3",
    )
    return Hypothesis(
        track=HypothesisTrack.TBIO,
        disease="acute myeloid leukemia",
        target=target,
        mechanism="Inhibit FLT3 signalling.",
        intervention_direction="inhibit",
        falsification_test="Reject if two disease models show no response.",
        claims=[Claim(text="FLT3 is a candidate target.")],
        evidence=[],
        score=0.8,
        rank=1,
    )


def test_target_discovery_policy_rejection_issues_no_http() -> None:
    client = NeverHTTPClient()
    result = DiseaseTargetDiscovery(client=client, max_attempts=1).discover(
        "acute myeloid leukemia",
        allowed_sources=[],
    )

    assert client.calls == []
    assert result.candidates == ()
    assert all(item.status == "skipped" for item in result.source_statuses)
    assert all(
        item.failure_codes == ("policy_rejected",)
        for item in result.source_statuses
    )


def test_target_identity_policy_rejection_issues_no_http() -> None:
    client = NeverHTTPClient()
    result = TargetIdentityResolver(client=client).resolve(
        "FLT3",
        "ENSG00000122025",
        allowed_sources=[],
    )

    assert client.calls == []
    assert result.status == "unresolved"
    assert {item.source for item in result.sources} == {"Open Targets", "UniProt"}
    assert all(item.status == "skipped" for item in result.sources)
    assert all(item.error_type == "policy_rejected" for item in result.sources)
    assert all(item.queried is False for item in result.sources)


def test_context_policy_rejection_issues_no_retriever_calls() -> None:
    retriever = NeverContextRetriever()
    result = DiseaseTargetContextRouter(retriever).route(
        ContextRoutingRequest(
            disease="acute myeloid leukemia",
            targets=[
                ContextTarget(
                    symbol="FLT3",
                    track=HypothesisTrack.TBIO,
                    rank=1,
                    score=0.8,
                    uniprot_id="P36888",
                    gencode_id="ENSG00000122025",
                )
            ],
            target_cell_type="myeloid blast",
            allowed_sources=[],
        )
    )

    assert retriever.calls == []
    assert result.source_calls_used == 0
    assert result.planned_source_calls == 0
    assert result.targets
    assert all(item.status == "policy_rejected" for item in result.targets[0].sources)
    assert all(item.queried is False for item in result.targets[0].sources)


def test_public_chemistry_policy_rejection_issues_no_http() -> None:
    client = NeverHTTPClient()
    result = PublicChemistryEnricher(client=client).enrich(
        ChemistryLookup(
            candidate_name="quizartinib",
            smiles="CC",
            uniprot_id="P36888",
        ),
        allowed_sources=[],
    )

    assert client.calls == []
    assert result.evidence == []
    assert len(result.sources) == 6
    assert all(item.status == "skipped" for item in result.sources)
    assert all(item.error_type == "policy_rejected" for item in result.sources)
    assert all(item.queried is False for item in result.sources)


def test_restricted_source_policy_rejection_issues_no_adapter_calls() -> None:
    drugcentral = NeverRestrictedAdapter("DrugCentral")
    drugbank = NeverRestrictedAdapter("DrugBank")
    patents = NeverRestrictedAdapter("Google Patents")
    enricher = RoutedRestrictedCandidateEnricher(
        drugcentral=drugcentral,  # type: ignore[arg-type]
        drugbank=drugbank,  # type: ignore[arg-type]
        google_patents=patents,  # type: ignore[arg-type]
    )
    hypothesis = _hypothesis()
    candidate = CandidateCard(
        route=CandidateRoute.DE_NOVO,
        name="DN-FLT3-001",
        target=hypothesis.target,
        mechanism="inhibit FLT3",
        smiles="CC",
        status="conditional",
    )

    result = enricher.enrich_candidate(
        candidate,
        hypothesis,
        allowed_sources=[],
    )

    assert drugcentral.calls == []
    assert drugbank.calls == []
    assert patents.calls == []
    assert len(result.sources) == 3
    assert all(item.status == "skipped" for item in result.sources)
    assert all(item.error_type == "policy_rejected" for item in result.sources)
    assert all(item.queried is False for item in result.sources)
    assert all(
        item.get("policy_decision") == "policy_rejected"
        for item in result.routing
    )
