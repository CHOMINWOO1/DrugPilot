from __future__ import annotations

from hashlib import sha256

import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateMetrics,
    CandidateRoute,
    EvidenceRef,
    GatewayReadiness,
    GateName,
    ReadinessStatus,
    ReworkOrder,
    ScientificBackendKind,
    ScientificExecutionStatus,
    TargetCandidate,
    ToolReadiness,
)
from drugpilot.tools.chemistry import ChemistryReport
from drugpilot.tools.executable import ScientificDataUnavailable
from drugpilot.tools.protocols import CandidateSeed


INITIAL_ADMET = {
    "hepatotoxicity": 0.21,
    "herg_risk": 0.22,
    "ames_risk": 0.23,
    "cyp3a4_inhibition": 0.24,
    "clearance_risk": 0.25,
    "oral_bioavailability": 0.71,
}
REWORKED_ADMET = {
    "hepatotoxicity": 0.31,
    "herg_risk": 0.32,
    "ames_risk": 0.33,
    "cyp3a4_inhibition": 0.34,
    "clearance_risk": 0.35,
    "oral_bioavailability": 0.81,
}
TARGET_UNIPROT = {
    "FLT3": "P36888",
    "BCL2": "P10415",
    "STAT5A": "P42229",
    "KRAS": "P01116",
    "EGFR": "P00533",
}


def _inchikey(smiles: str) -> str:
    return f"{sha256(smiles.encode()).hexdigest()[:14].upper()}-UHFFFAOYSA-N"


class PersistedStructureChemistry:
    backend_name = "rdkit"
    backend_version = "test-rdkit-persisted-structure-v1"

    def inspect(self, smiles: str) -> ChemistryReport:
        return ChemistryReport(
            valid=True,
            canonical_smiles=smiles,
            inchikey=_inchikey(smiles),
            metrics=CandidateMetrics(
                molecular_weight=280.0,
                clogp=2.1,
                tpsa=58.0,
                hbd=1,
                hba=3,
                synthetic_accessibility=3.0,
            ),
            backend="rdkit",
        )


class RunLocalLiveGateway:
    """Live-shaped gateway whose retrieval cache exists only per instance."""

    source_version = "run-local-live-gateway-v1"
    backend_kind = ScientificBackendKind.LIVE.value
    fixture = False

    def __init__(self, admet_scores: dict[str, float]) -> None:
        self.admet_scores = dict(admet_scores)
        self.last_operation_provenance: dict[str, object] = {}
        self.predict_calls = 0
        self.docking_calls = 0
        self.restore_structure_calls = 0

    def readiness(self, target=None) -> GatewayReadiness:
        return GatewayReadiness(
            gateway_name=type(self).__name__,
            source_version=self.source_version,
            backend_kind=ScientificBackendKind.LIVE,
            fixture=False,
            capabilities=[
                ToolReadiness(
                    tool_name=tool,
                    operation=operation,
                    status=ReadinessStatus.READY,
                    required=required,
                    backend_kind=ScientificBackendKind.LIVE,
                    backend_name="run-local-live-test",
                    backend_version=self.source_version,
                    fixture=False,
                )
                for tool, operation, required in (
                    ("target_normalizer", "normalize_target", True),
                    ("repurposing_search", "retrieve_candidates", True),
                    ("de_novo_generator", "generate_candidates", True),
                    ("docking", "score_binding_pose", False),
                    ("admet_predictor", "predict_six_endpoint_panel", True),
                )
            ],
        )

    def normalize_target(self, target: TargetCandidate) -> TargetCandidate:
        return target.model_copy(
            update={
                "uniprot_id": target.uniprot_id or TARGET_UNIPROT[target.symbol],
                "metadata": {
                    **target.metadata,
                    "chembl_target_id": f"CHEMBL-TARGET-{target.symbol}",
                }
            },
            deep=True,
        )

    @staticmethod
    def _seed(
        target: TargetCandidate,
        *,
        direction: str,
        index: int,
        generated: bool,
    ) -> CandidateSeed:
        smiles = f"CC{'C' * index}O"
        inchikey = _inchikey(smiles)
        compound_id = f"LIVE-{target.symbol}-{index}"
        evidence = EvidenceRef(
            source="ChEMBL",
            external_id=compound_id,
            title=f"Exact measured candidate {compound_id}",
            supports="support",
            metadata={
                "fixture": False,
                "provenance_class": "external_live",
                "target_accession": target.uniprot_id,
                "target_gene": target.symbol,
                "molecule_chembl_id": compound_id,
            },
        )
        activities = () if generated else (
            ActivityMeasurement(
                endpoint="IC50",
                value=float(10 + index),
                unit="nM",
                assay_type="cell-based format",
                source_evidence_id=evidence.evidence_id,
            ),
        )
        return CandidateSeed(
            name=("DN" if generated else "RP") + f"-{target.symbol}-{index}",
            smiles=smiles,
            mechanism=f"{direction} {target.symbol}",
            activities=activities,
            evidence=(evidence,),
            metadata={
                "fixture": False,
                "provenance_class": "external_live",
                "drug_id": compound_id,
                "chembl_id": compound_id,
                "identifiers": [compound_id],
                "canonical_smiles": smiles,
                "standard_inchi_key": inchikey,
                "measured_target_uniprot_id": target.uniprot_id,
                "measured_target_gene": target.symbol,
                "direction_requested": direction,
                "direction_evidence_rules": ["measured_ic50_endpoint"],
                "generator": "test-live-generator" if generated else None,
            },
        )

    def retrieve_repurposing(self, target, *, direction, limit):
        return tuple(
            self._seed(
                target,
                direction=direction,
                index=index,
                generated=False,
            )
            for index in range(1, limit + 1)
        )

    def generate_de_novo(self, target, *, direction, limit):
        return tuple(
            self._seed(
                target,
                direction=direction,
                index=index,
                generated=True,
            )
            for index in range(1, limit + 1)
        )

    def dock(self, *, smiles, target):
        self.docking_calls += 1
        self.last_operation_provenance = {
            "method": "test-live-docking",
            "backend_version": "dock-v1",
            "fixture": False,
        }
        return -7.2

    def predict_admet(self, *, smiles):
        self.predict_calls += 1
        self.last_operation_provenance = {
            "method": "test-live-admet",
            "model_version": "admet-rework-v2",
            "fixture": False,
        }
        return dict(self.admet_scores)

    def refine(self, seed, *, reasons, iteration):
        raise AssertionError("live refinement is outside this regression")

    def restore_activity(self, *, candidate_name, target):
        raise AssertionError("activity restoration is outside this regression")

    def restore_structure(self, *, candidate_name, route, target, direction):
        self.restore_structure_calls += 1
        raise ScientificDataUnavailable(
            f"No live-source structure is cached for {candidate_name}"
        )


def _live_portfolio(brief):
    # The test owns a deterministic fixture target catalog.  Production's
    # fail-closed no-source default must not be weakened for this component
    # setup, so fixture authorization is explicit and local.
    hypotheses = MA1Agent(
        settings=Settings(
            _env_file=None,
            dynamic_target_discovery_enabled=False,
            dynamic_target_fallback_enabled=True,
            demo_fixture_fallback_enabled=True,
        )
    ).run(brief)
    live_hypotheses = []
    for hypothesis in hypotheses.hypotheses[:2]:
        target = hypothesis.target.model_copy(
            update={
                "identity_status": "resolved",
                "uniprot_id": (
                    hypothesis.target.uniprot_id
                    or TARGET_UNIPROT[hypothesis.target.symbol]
                ),
                "metadata": {
                    **hypothesis.target.metadata,
                    "target_discovery_source": "external_dynamic",
                    "association_score": max(hypothesis.score, 0.5),
                    "evidence_summary": {
                        "overall_association_score": max(hypothesis.score, 0.5),
                        "known_drug_score": 0.8,
                    },
                },
            },
            deep=True,
        )
        live_evidence = EvidenceRef(
            source="Open Targets",
            external_id=target.ensembl_id or target.uniprot_id or target.symbol,
            title=f"Live target evidence for {target.symbol}",
            supports="support",
            metadata={
                "fixture": False,
                "mock": False,
                "provenance_class": "external_live",
                "query": target.symbol,
            },
        )
        live_hypotheses.append(
            hypothesis.model_copy(
                update={
                    "target": target,
                    "evidence": [*hypothesis.evidence, live_evidence],
                },
                deep=True,
            )
        )
    hypotheses = hypotheses.model_copy(
        update={"hypotheses": live_hypotheses},
        deep=True,
    )
    plan = PlannerAgent().run(hypotheses, brief)
    initial_gateway = RunLocalLiveGateway(INITIAL_ADMET)
    packet = MA2Agent(
        gateway=initial_gateway,
        chemistry=PersistedStructureChemistry(),
        ma2_portfolio_enabled=True,
        max_refinement_iterations=0,
    ).run(plan, hypotheses)
    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len({item.metadata["portfolio_target_key"] for item in packet.candidates}) >= 2
    return hypotheses, plan, packet


def test_admet_rework_uses_persisted_identity_after_gateway_cache_loss_and_preserves_other_targets(
    brief,
) -> None:
    hypotheses, plan, previous = _live_portfolio(brief)
    affected_key = str(previous.candidates[0].metadata["portfolio_target_key"])
    affected_ids = {
        item.candidate_id
        for item in previous.candidates
        if item.metadata["portfolio_target_key"] == affected_key
    }
    untouched_before = {
        item.candidate_id: item
        for item in previous.candidates
        if item.metadata["portfolio_target_key"] != affected_key
    }
    untouched_summaries = {
        item.target_key: item.model_dump(mode="json")
        for item in previous.target_executions
        if item.target_key != affected_key
    }
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.admet",
        affected_candidate_ids=sorted(affected_ids),
        required_actions=[
            "Repeat the candidate ADMET panel with the versioned model.",
            (
                "An unrelated evidence-review note mentions candidate structure "
                "identity and must not widen the ADMET rework scope."
            ),
        ],
    )
    fresh_gateway = RunLocalLiveGateway(REWORKED_ADMET)
    repaired = MA2Agent(
        gateway=fresh_gateway,
        chemistry=PersistedStructureChemistry(),
        ma2_portfolio_enabled=True,
        max_refinement_iterations=0,
    ).rework(plan, hypotheses, previous, order)

    assert fresh_gateway.restore_structure_calls == 0
    assert fresh_gateway.docking_calls == 0
    assert fresh_gateway.predict_calls == len(affected_ids)
    repaired_by_id = {item.candidate_id: item for item in repaired.candidates}
    for candidate_id in affected_ids:
        card = repaired_by_id[candidate_id]
        assert card.metrics.admet_scores == REWORKED_ADMET
        assert card.admet_execution is not None
        assert card.admet_execution.scores == REWORKED_ADMET
        assert card.admet_execution.backend_version == "admet-rework-v2"
        assert card.metadata["rework_identity_attestation"]["validated"] is True
        assert card.metadata["rework_identity_attestation"]["fixture"] is False
    affected_trace_ids = {
        trace_id
        for candidate_id in affected_ids
        for trace_id in repaired_by_id[candidate_id].tool_trace_ids
    }
    affected_calls = [
        call for call in repaired.tool_calls if call.call_id in affected_trace_ids
    ]
    assert any(
        call.operation == "validate_persisted_candidate_identity"
        for call in affected_calls
    )
    assert any(call.operation == "repeat_six_endpoint_panel" for call in affected_calls)
    assert not any(call.operation == "restore_candidate_structure" for call in affected_calls)
    assert all(call.fixture is False for call in affected_calls)

    for candidate_id, before in untouched_before.items():
        after = repaired_by_id[candidate_id]
        assert after.smiles == before.smiles
        assert after.inchikey == before.inchikey
        assert after.activities == before.activities
        assert after.evidence == before.evidence
        assert after.chemistry_execution == before.chemistry_execution
        assert after.docking_execution == before.docking_execution
        assert after.admet_execution == before.admet_execution
        assert after.tool_trace_ids == before.tool_trace_ids
        assert candidate_id in repaired.reused_artifact_ids
    repaired_summaries = {
        item.target_key: item.model_dump(mode="json")
        for item in repaired.target_executions
        if item.target_key != affected_key
    }
    assert repaired_summaries == untouched_summaries


def test_persisted_rework_fails_closed_on_target_provenance_conflict(brief) -> None:
    hypotheses, plan, previous = _live_portfolio(brief)
    affected = previous.candidates[0]
    conflicting = affected.model_copy(
        update={
            "metadata": {
                **affected.metadata,
                "measured_target_uniprot_id": "P99999",
            }
        },
        deep=True,
    )
    tampered = previous.model_copy(
        update={
            "candidates": [
                conflicting if item.candidate_id == affected.candidate_id else item
                for item in previous.candidates
            ]
        },
        deep=True,
    )
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.admet",
        affected_candidate_ids=[affected.candidate_id],
        required_actions=["Repeat the candidate ADMET panel."],
    )
    fresh_gateway = RunLocalLiveGateway(REWORKED_ADMET)
    agent = MA2Agent(
        gateway=fresh_gateway,
        chemistry=PersistedStructureChemistry(),
        ma2_portfolio_enabled=True,
        max_refinement_iterations=0,
    )

    with pytest.raises(
        ScientificDataUnavailable,
        match="measured_target_uniprot_id=.*conflicts with target",
    ):
        agent.rework(plan, hypotheses, tampered, order)

    assert fresh_gateway.predict_calls == 0
    assert fresh_gateway.restore_structure_calls == 0


def test_ranking_rework_after_restart_reuses_persisted_activity_without_gateway_cache(
    brief,
) -> None:
    hypotheses, plan, previous = _live_portfolio(brief)
    affected = next(
        item for item in previous.candidates if item.status != "excluded"
    )
    before_activities = affected.activities
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.ranking",
        affected_candidate_ids=[affected.candidate_id],
        required_actions=[
            (
                "MA3 independently recomputed the activity ranking and found "
                "score_breakdown.recalculation."
            )
        ],
    )
    # This is a new-process gateway: its restore_activity method intentionally
    # raises, proving ranking rework has no hidden dependency on run-local
    # scientific-source caches.
    fresh_gateway = RunLocalLiveGateway(REWORKED_ADMET)
    repaired = MA2Agent(
        gateway=fresh_gateway,
        chemistry=PersistedStructureChemistry(),
        ma2_portfolio_enabled=True,
        max_refinement_iterations=0,
    ).rework(plan, hypotheses, previous, order)

    after = next(
        item
        for item in repaired.candidates
        if item.candidate_id == affected.candidate_id
    )
    assert after.activities == before_activities
    assert fresh_gateway.predict_calls == 0
    assert fresh_gateway.docking_calls == 0
    assert fresh_gateway.restore_structure_calls == 0
    assert repaired.execution_status is ScientificExecutionStatus.COMPLETED
    assert repaired.metadata["last_portfolio_rework"]["unaffected_artifacts_reused"] is True
