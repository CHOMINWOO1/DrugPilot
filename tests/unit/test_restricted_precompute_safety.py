from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    CandidateRoute,
    EvidenceRef,
    GateName,
    GateVerdictCode,
    ReworkOrder,
    ResearchBrief,
    RunMode,
)
from drugpilot.tools.fixture_gateway import DeterministicFixtureGateway
from drugpilot.tools.restricted_candidate_enrichment import (
    RoutedRestrictedCandidateEnricher,
)
from drugpilot.tools.restricted_sources import (
    RestrictedSourceQuery,
    RestrictedSourceResult,
    RestrictedSourceSummary,
)


class CountingGateway(DeterministicFixtureGateway):
    def __init__(self) -> None:
        super().__init__()
        self.docked_smiles: list[str] = []
        self.admet_smiles: list[str] = []

    def dock(self, *, smiles, target):
        self.docked_smiles.append(smiles)
        return super().dock(smiles=smiles, target=target)

    def predict_admet(self, *, smiles):
        self.admet_smiles.append(smiles)
        return super().predict_admet(smiles=smiles)


class ConflictingRegistryIdentityGateway(CountingGateway):
    """Attach two exact-name DrugCentral IDs to every discovery seed."""

    def retrieve_repurposing(self, target, *, direction, limit):
        seeds = super().retrieve_repurposing(
            target,
            direction=direction,
            limit=limit,
        )
        conflicted = []
        for seed in seeds:
            identity_records = tuple(
                EvidenceRef(
                    source="DrugCentral",
                    external_id=external_id,
                    title=f"Conflicting DrugCentral identity {external_id}",
                    supports="context",
                    metadata={
                        "drug_name": seed.name,
                        "drugcentral_id": external_id,
                        "provenance_class": "external_local_dataset",
                    },
                )
                for external_id in ("5306", "9999")
            )
            conflicted.append(
                replace(seed, evidence=(*seed.evidence, *identity_records))
            )
        return conflicted


class FakeRestrictedAdapter:
    def __init__(
        self,
        source: str,
        handler: Callable[[RestrictedSourceQuery], tuple[str, list[EvidenceRef]]],
    ) -> None:
        self.source = source
        self.handler = handler
        self.queries: list[RestrictedSourceQuery] = []
        self.documentation_url = f"https://example.test/{source}"
        self.required_configuration = (f"{source} configuration",)
        self.limitations = (
            "A missing contraindication record is not proof of safety.",
        )

    def query(self, request: RestrictedSourceQuery) -> RestrictedSourceResult:
        self.queries.append(request.model_copy(deep=True))
        status, evidence = self.handler(request)
        return RestrictedSourceResult(
            query=request,
            evidence=evidence,
            summary=RestrictedSourceSummary(
                source=self.source,
                operation=request.operation,
                status=status,
                queried=status not in {"not_configured", "skipped"},
                record_count=len(evidence),
                error_type="ConnectError" if status == "error" else None,
                documentation_url=self.documentation_url,
                required_configuration=self.required_configuration,
                limitations=self.limitations,
            ),
        )


class FakePatentsAdapter:
    source = "Google Patents"

    def query(self, request):  # pragma: no cover - repurposing never calls it
        raise AssertionError(f"unexpected patent query: {request}")


def _pipeline(brief: ResearchBrief):
    hypotheses = MA1Agent(
        Settings(
            mode=RunMode.MOCK,
            public_evidence_enabled=False,
            dynamic_target_discovery_enabled=False,
            dynamic_target_fallback_enabled=True,
            demo_fixture_fallback_enabled=True,
        )
    ).run(brief)
    plan = PlannerAgent().run(hypotheses, brief)
    return hypotheses, plan


def _empty(_request: RestrictedSourceQuery) -> tuple[str, list[EvidenceRef]]:
    return "empty", []


def _withdraw_first(request: RestrictedSourceQuery) -> tuple[str, list[EvidenceRef]]:
    if (
        request.operation == "identity_regulatory"
        and request.candidate_name == "RP-FLT3-001"
    ):
        return "ok", [
            EvidenceRef(
                source="DrugCentral",
                external_id="DC-WITHDRAWN-1",
                title="Configured regulatory identity record",
                supports="counter",
                metadata={
                    "restricted_operation": "identity_regulatory",
                    "drug_name": request.candidate_name,
                    "regulatory_status": "withdrawn",
                    "withdrawn": True,
                    "provenance_class": "external_local_dataset",
                },
            )
        ]
    return "empty", []


def _configured_enricher(drugcentral_handler, drugbank_handler=_empty):
    drugcentral = FakeRestrictedAdapter("DrugCentral", drugcentral_handler)
    drugbank = FakeRestrictedAdapter("DrugBank", drugbank_handler)
    return (
        RoutedRestrictedCandidateEnricher(
            drugcentral=drugcentral,
            drugbank=drugbank,
            google_patents=FakePatentsAdapter(),
        ),
        drugcentral,
        drugbank,
    )


def test_configured_withdrawn_candidate_is_excluded_before_expensive_calls(
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(brief)
    gateway = CountingGateway()
    enricher, _, _ = _configured_enricher(_withdraw_first)

    packet = MA2Agent(
        gateway=gateway,
        restricted_evidence_enricher=enricher,
    ).run(plan, hypotheses)

    withdrawn = next(item for item in packet.candidates if item.name == "RP-FLT3-001")
    assert withdrawn.status == "excluded"
    assert withdrawn.rank is None
    assert withdrawn.metadata["policy_eligible"] is False
    assert withdrawn.metadata["restricted_safety_screen"]["disposition"] == "excluded"
    assert any("withdrawn" in reason for reason in withdrawn.exclusion_reasons)
    assert withdrawn.smiles not in gateway.docked_smiles
    assert withdrawn.smiles not in gateway.admet_smiles
    assert len(gateway.docked_smiles) == 2
    assert len(gateway.admet_smiles) == 2
    assert asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    ).verdict is GateVerdictCode.PASS


def test_external_identity_and_class_feed_the_exact_forbidden_policy(
    brief: ResearchBrief,
) -> None:
    constrained = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={
                    "forbidden_drug_ids": ["DB-EXTERNAL-1"],
                    "forbidden_drug_classes": ["restricted kinase class"],
                }
            )
        },
        deep=True,
    )
    hypotheses, plan = _pipeline(constrained)

    def identity(request: RestrictedSourceQuery):
        if (
            request.operation == "identity_regulatory"
            and request.candidate_name == "RP-FLT3-001"
        ):
            return "ok", [
                EvidenceRef(
                    source="DrugBank",
                    external_id="DB-EXTERNAL-1",
                    title="Licensed identity record",
                    supports="context",
                        metadata={
                            "restricted_operation": "identity_regulatory",
                            "drug_name": request.candidate_name,
                            "drugbank_id": "DB-EXTERNAL-1",
                        "drug_class": "restricted kinase class",
                        "provenance_class": "external_live",
                    },
                )
            ]
        return "empty", []

    enricher, _, _ = _configured_enricher(_empty, identity)
    gateway = CountingGateway()
    packet = MA2Agent(
        gateway=gateway,
        restricted_evidence_enricher=enricher,
    ).run(plan, hypotheses)
    candidate = next(item for item in packet.candidates if item.name == "RP-FLT3-001")

    assert candidate.status == "excluded"
    assert {
        reason.split(":", 1)[0] for reason in candidate.exclusion_reasons
    } == {"forbidden_drug_id", "forbidden_drug_class"}
    assert candidate.smiles not in gateway.docked_smiles
    assert candidate.smiles not in gateway.admet_smiles


def test_configured_transport_failure_is_unresolved_and_g1_abstains(
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(brief)

    def fail_identity(request: RestrictedSourceQuery):
        return ("error", []) if request.operation == "identity_regulatory" else ("empty", [])

    enricher, _, _ = _configured_enricher(fail_identity)
    gateway = CountingGateway()
    packet = MA2Agent(
        gateway=gateway,
        restricted_evidence_enricher=enricher,
    ).run(plan, hypotheses)

    assert gateway.docked_smiles == []
    assert gateway.admet_smiles == []
    assert all(candidate.status == "excluded" for candidate in packet.candidates)
    assert all(
        candidate.metadata["restricted_safety_screen"]["disposition"]
        == "unresolved"
        for candidate in packet.candidates
    )
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.ABSTAIN
    assert "RESTRICTED_SAFETY_LOOKUP_UNRESOLVED" in {
        finding.code for finding in verdict.findings
    }


def test_conflicting_registry_ids_fail_closed_before_expensive_calls(
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(brief)
    gateway = ConflictingRegistryIdentityGateway()
    enricher, drugcentral, _ = _configured_enricher(_empty)

    packet = MA2Agent(
        gateway=gateway,
        restricted_evidence_enricher=enricher,
    ).run(plan, hypotheses)

    assert packet.candidates
    assert gateway.docked_smiles == []
    assert gateway.admet_smiles == []
    assert not any(
        call.tool_name in {"chemistry_validator", "docking", "admet_predictor"}
        for call in packet.tool_calls
    )
    assert all(candidate.status == "excluded" for candidate in packet.candidates)
    assert all(candidate.rank is None for candidate in packet.candidates)
    assert all(candidate.score == 0.0 for candidate in packet.candidates)
    assert all(
        candidate.metadata["policy_eligible"] is False
        for candidate in packet.candidates
    )
    assert all(
        candidate.metadata["restricted_safety_screen"]["disposition"]
        == "unresolved"
        for candidate in packet.candidates
    )
    assert all(
        candidate.metadata["restricted_safety_screen"]["coverage"] == "failed"
        for candidate in packet.candidates
    )
    assert all(
        "candidate_registry_identity_conflict"
        in " ".join(
            candidate.metadata["restricted_safety_screen"]["reason_codes"]
        )
        for candidate in packet.candidates
    ), [
        candidate.metadata["restricted_safety_screen"]["reason_codes"]
        for candidate in packet.candidates
    ]
    compound_queries = [
        query
        for query in drugcentral.queries
        if query.operation in {"identity_regulatory", "contraindication"}
    ]
    assert compound_queries == []

    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.ABSTAIN
    assert "RESTRICTED_SAFETY_LOOKUP_UNRESOLVED" in {
        finding.code for finding in verdict.findings
    }


def test_unconfigured_sources_are_recorded_without_becoming_a_clean_result(
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(brief)

    def missing(_request: RestrictedSourceQuery):
        return "not_configured", []

    enricher, _, _ = _configured_enricher(missing, missing)
    gateway = CountingGateway()
    packet = MA2Agent(
        gateway=gateway,
        restricted_evidence_enricher=enricher,
    ).run(plan, hypotheses)

    assert len(gateway.docked_smiles) == 3
    assert all(
        candidate.metadata["restricted_safety_screen"]["disposition"]
        == "not_configured"
        for candidate in packet.candidates
    )
    summaries = [
        item
        for item in packet.metadata["public_enrichment_queries"]
        if item.get("phase") == "precompute_safety_screen"
        and item.get("record_type") != "source_routing_decision"
    ]
    assert summaries
    assert {item["status"] for item in summaries} == {"not_configured"}
    assert all(item["queried"] is False for item in summaries)


def test_structure_rework_rechecks_restricted_identity_before_recompute(
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(brief)
    enricher, drugcentral, _ = _configured_enricher(_empty)
    ma2 = MA2Agent(
        gateway=CountingGateway(),
        restricted_evidence_enricher=enricher,
    )
    previous = ma2.run(plan, hypotheses)
    candidate = previous.candidates[0]
    before = len(
        [query for query in drugcentral.queries if query.operation == "identity_regulatory"]
    )
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.standardize",
        affected_candidate_ids=[candidate.candidate_id],
        required_actions=["Restore and revalidate the candidate structure."],
    )

    ma2.rework(plan, hypotheses, previous, order)

    after = len(
        [query for query in drugcentral.queries if query.operation == "identity_regulatory"]
    )
    assert after == before + 1
