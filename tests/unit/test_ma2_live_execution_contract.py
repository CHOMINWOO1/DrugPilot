from __future__ import annotations

import asyncio
from datetime import timedelta
from hashlib import sha256
import json

import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateMetrics,
    EvidenceRef,
    GatewayReadiness,
    GateName,
    GateVerdictCode,
    ReadinessStatus,
    NamespacedRegistryIdentifier,
    ReservedControlCompound,
    ReworkOrder,
    ResearchBrief,
    RunMode,
    ScientificBackendKind,
    ScientificExecutionStatus,
    ToolReadiness,
)
from drugpilot.orchestration.graph import build_ma2_gateway
from drugpilot.tools.chemistry import ChemistryReport
from drugpilot.tools.executable import (
    ScientificDataUnavailable,
    ScientificToolUnavailable,
)
from drugpilot.tools.fixture_gateway import DeterministicFixtureGateway
from drugpilot.tools.live_repurposing_gateway import (
    PaperNativeV6RepurposingContractUnavailable,
    PublicRepurposingGateway,
)
from drugpilot.tools.live_ma2_gateway import NoMeasuredLigandCandidates
from drugpilot.tools.protocols import CandidateSeed


ADMET = {
    "hepatotoxicity": 0.1,
    "herg_risk": 0.2,
    "ames_risk": 0.1,
    "cyp3a4_inhibition": 0.2,
    "clearance_risk": 0.3,
    "oral_bioavailability": 0.7,
}


def _canonical_hash(value) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class StubRDKitChemistry:
    backend_name = "rdkit"
    backend_version = "test-rdkit-2026.08"

    def inspect(self, smiles: str) -> ChemistryReport:
        return ChemistryReport(
            valid=True,
            canonical_smiles=smiles,
            inchikey="LIVE-STRUCTURE-KEY",
            metrics=CandidateMetrics(
                molecular_weight=250.0,
                clogp=2.0,
                tpsa=55.0,
                hbd=1,
                hba=3,
                synthetic_accessibility=3.0,
            ),
            backend="rdkit",
        )


class PartialLiveGateway:
    source_version = "partial-live-test-v1"
    backend_kind = "live"
    fixture = False

    def __init__(
        self,
        *,
        incomplete_admet: bool = False,
        target_linked_activity: bool = True,
        conflicting_target_identifiers: bool = False,
        docking_unavailable: bool = False,
    ) -> None:
        self.incomplete_admet = incomplete_admet
        self.target_linked_activity = target_linked_activity
        self.conflicting_target_identifiers = conflicting_target_identifiers
        self.docking_unavailable = docking_unavailable
        self.last_operation_provenance = {}
        self.docking_calls = 0

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
                    backend_name="live-test-backend",
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

    def normalize_target(self, target):
        return target

    def retrieve_repurposing(self, target, *, direction, limit):
        evidence = EvidenceRef(
            source="DrugCentral",
            external_id="DC-LIVE-1",
            title="Exact live target interaction",
            supports="support",
            metadata={
                "fixture": False,
                "provenance_class": "external_local_dataset",
                "target_accession": (
                    "P99999"
                    if self.conflicting_target_identifiers
                    else target.uniprot_id
                    if self.target_linked_activity
                    else "P99999"
                ),
                "target_gene": (
                    target.symbol
                    if self.target_linked_activity
                    or self.conflicting_target_identifiers
                    else "OTHER"
                ),
            },
        )
        return (
            CandidateSeed(
                name="measured-live-drug",
                smiles="CCO",
                mechanism=f"{direction} {target.symbol}",
                activities=(
                    ActivityMeasurement(
                        endpoint="IC50",
                        value=25.0,
                        unit="nM",
                        assay_type="measured biochemical assay",
                        source_evidence_id=evidence.evidence_id,
                    ),
                ),
                evidence=(evidence,),
                metadata={
                    "fixture": False,
                    "activity_units_inferred": False,
                    "direction_requested": direction,
                    "direction_evidence_rules": ["measured_ic50_endpoint"],
                },
            ),
        )

    def generate_de_novo(self, target, *, direction, limit):
        raise AssertionError("repurposing route expected")

    def dock(self, *, smiles, target):
        self.docking_calls += 1
        if self.docking_unavailable:
            raise ScientificToolUnavailable(
                "No production-ready receptor/box profile is configured"
            )
        raise AssertionError("direct measured activity should permit docking skip")

    def predict_admet(self, *, smiles):
        self.last_operation_provenance = {
            "method": "versioned live ADMET model",
            "model_version": "admet-test-v1",
            "fixture": False,
        }
        if self.incomplete_admet:
            return {key: value for key, value in ADMET.items() if key != "clearance_risk"}
        return dict(ADMET)

    def refine(self, seed, *, reasons, iteration):
        raise AssertionError("live refinement is disabled")

    def restore_activity(self, *, candidate_name, target):
        return ()

    def restore_structure(self, *, candidate_name, route, target, direction):
        raise AssertionError("not required")


class ConclusivePortfolioGateway(PartialLiveGateway):
    """Return one real candidate and conclusive zero-hit outcomes elsewhere."""

    def __init__(self, candidate_symbol: str | None) -> None:
        super().__init__()
        self.candidate_symbol = candidate_symbol
        self.queried_symbols: list[str] = []

    @staticmethod
    def _no_candidates(target, direction: str, minimum_required: int = 1):
        raise NoMeasuredLigandCandidates(
            target_symbol=target.symbol,
            target_uniprot_id=target.uniprot_id,
            direction=direction,
            eligible_count=0,
            minimum_required=minimum_required,
            reason_code="NO_DIRECTION_COMPATIBLE_MEASURED_LIGANDS",
            provenance={
                "query_completed": True,
                "fixture": False,
                "matching_policy": "exact target and declared direction",
            },
        )

    def retrieve_repurposing(self, target, *, direction, limit):
        self.queried_symbols.append(target.symbol)
        if target.symbol == self.candidate_symbol:
            return super().retrieve_repurposing(
                target,
                direction=direction,
                limit=limit,
            )
        return self._no_candidates(target, direction)

    def generate_de_novo(self, target, *, direction, limit):
        self.queried_symbols.append(target.symbol)
        return self._no_candidates(target, direction, minimum_required=2)


class TimeoutIsolatedPortfolioGateway(ConclusivePortfolioGateway):
    """One typed ChEMBL failure must not erase completed target branches."""

    def __init__(self, candidate_symbol: str, timeout_symbol: str) -> None:
        super().__init__(candidate_symbol)
        self.timeout_symbol = timeout_symbol

    def _maybe_timeout(self, target) -> None:
        if target.symbol != self.timeout_symbol:
            return
        unavailable = ScientificDataUnavailable(
            "ChEMBL request unavailable (read_timeout)"
        )
        unavailable.provenance = {
            "source": "ChEMBL",
            "error_type": "read_timeout",
            "attempt_count": 2,
            "max_attempts": 2,
            "exhausted": True,
            "cache_hit": False,
            "fixture": False,
        }
        raise unavailable

    def retrieve_repurposing(self, target, *, direction, limit):
        self._maybe_timeout(target)
        return super().retrieve_repurposing(
            target,
            direction=direction,
            limit=limit,
        )

    def generate_de_novo(self, target, *, direction, limit):
        self._maybe_timeout(target)
        return super().generate_de_novo(
            target,
            direction=direction,
            limit=limit,
        )


class StructuredCandidateContractFailureGateway(PartialLiveGateway):
    """Expose the adapter's bounded source-contract reason to MA2."""

    def __init__(
        self,
        *,
        reason_code: str = "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT",
        support_candidate_count: int = 1,
        evidence_grade: str | None = "B",
    ) -> None:
        super().__init__()
        self.contract_reason_code = reason_code
        self.support_candidate_count = support_candidate_count
        self.evidence_grade = evidence_grade

    def _failure_projections(self):
        fixed_ids = [f"CHEMBL{index + 1}" for index in range(12)]
        source_rows = []
        wrapper_rows = []
        for index, candidate_id in enumerate(fixed_ids):
            verified = index < self.support_candidate_count
            source_row = {
                "pool_index": index,
                "fixed_pool_rank": index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": True,
                "activity_count": 1,
                "evidence_count": 1,
                "activity_id": str(1000 + index),
                "document_chembl_id": f"CHEMBLDOC{index + 1}",
                "pmid": str(2000 + index),
                "primary_verification_status": (
                    "verified" if verified else "unverified"
                ),
                "primary_reason_code": (
                    None if verified else "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED"
                ),
                "primary_attestation_sha256": _canonical_hash(
                    {"candidate_id": candidate_id, "verified": verified}
                ),
                "fixture": False,
            }
            source_row["row_sha256"] = _canonical_hash(source_row)
            source_rows.append(source_row)

            checks = {
                "cardinality_exact": True,
                "primary_mapping": True,
                "exact_binding": verified,
            }
            wrapper_row = {
                "pool_index": index,
                "fixed_pool_rank": index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": True,
                "wrapper_screen_status": "verified" if verified else "rejected",
                "wrapper_screen_reason_code": (
                    None if verified else "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
                ),
                "wrapper_screen_checks": checks,
                "wrapper_screen_checks_sha256": _canonical_hash(checks),
                "fixture": False,
            }
            wrapper_row["row_sha256"] = _canonical_hash(wrapper_row)
            wrapper_rows.append(wrapper_row)

        source_projection = {
            "schema_version": (
                "paper_native_v6_delegate_seed_source_binding_projection/v1"
            ),
            "fixed_pool_candidate_ids": fixed_ids,
            "fixed_pool_candidate_ids_sha256": _canonical_hash(fixed_ids),
            "delegate_return_candidate_ids": fixed_ids,
            "delegate_return_candidate_ids_sha256": _canonical_hash(fixed_ids),
            "row_count": len(source_rows),
            "rows": source_rows,
            "rows_sha256": _canonical_hash(source_rows),
            "complete": True,
            "fixture": False,
        }
        source_projection["attestation_sha256"] = _canonical_hash(
            source_projection
        )
        wrapper_projection = {
            "schema_version": (
                "paper_native_v6_wrapper_atomic_screen_projection/v1"
            ),
            "fixed_pool_candidate_ids": fixed_ids,
            "fixed_pool_candidate_ids_sha256": _canonical_hash(fixed_ids),
            "row_count": len(wrapper_rows),
            "rows": wrapper_rows,
            "rows_sha256": _canonical_hash(wrapper_rows),
            "complete": True,
            "fixture": False,
        }
        wrapper_projection["attestation_sha256"] = _canonical_hash(
            wrapper_projection
        )
        return source_projection, wrapper_projection

    def retrieve_repurposing(self, target, *, direction, limit):
        source_projection, wrapper_projection = self._failure_projections()
        raise PaperNativeV6RepurposingContractUnavailable(
            self.contract_reason_code,
            "The bounded paper-native v6 candidate contract did not pass.",
            provenance={
                "support_candidate_count": self.support_candidate_count,
                "minimum_verified_candidate_count": 3,
                "evidence_grade": self.evidence_grade,
                "promotion_eligible": False,
                "rejection_count": 12 - self.support_candidate_count,
                "rejections_sha256": "d" * 64,
                "delegate_provenance": {
                    "delegate_seed_source_binding_projection": source_projection,
                },
                "wrapper_atomic_screen_projection": wrapper_projection,
                "fixture": False,
            },
        )


class OptionalDockingUnreadyGateway(PartialLiveGateway):
    def __init__(self) -> None:
        super().__init__(target_linked_activity=False)

    def readiness(self, target=None) -> GatewayReadiness:
        report = super().readiness(target)
        return report.model_copy(
            update={
                "capabilities": [
                    capability.model_copy(
                        update={
                            "status": ReadinessStatus.UNAVAILABLE,
                            "reason_code": "NO_PRODUCTION_DOCKING_PROFILE",
                            "detail": "No target-specific validated receptor/box profile.",
                        }
                    )
                    if capability.tool_name == "docking"
                    else capability
                    for capability in report.capabilities
                ]
            },
            deep=True,
        )

    def dock(self, *, smiles, target):
        raise AssertionError("unready optional docking must not be invoked")


class CellBasedOptionalDockingUnreadyGateway(OptionalDockingUnreadyGateway):
    def __init__(self) -> None:
        PartialLiveGateway.__init__(self, target_linked_activity=True)

    def retrieve_repurposing(self, target, *, direction, limit):
        original = PartialLiveGateway.retrieve_repurposing(
            self,
            target,
            direction=direction,
            limit=limit,
        )[0]
        activity = original.activities[0].model_copy(
            update={"assay_type": "cell-based format"}
        )
        evidence = original.evidence[0].model_copy(
            update={
                "metadata": {
                    **original.evidence[0].metadata,
                    "assay_type": "cell-based format",
                    "assay_context": "cellular_or_system_context",
                    "direct_single_protein_engagement": False,
                }
            },
            deep=True,
        )
        return (
            CandidateSeed(
                name=original.name,
                smiles=original.smiles,
                mechanism=(
                    f"Measured cellular inhibitory potency in a {target.symbol}-linked "
                    "assay; direct single-protein engagement not established"
                ),
                activities=(activity,),
                evidence=(evidence,),
                metadata={
                    **dict(original.metadata),
                    "activity_assay_type": "cell-based format",
                    "activity_assay_context": "cellular_or_system_context",
                    "direct_single_protein_engagement": False,
                },
            ),
        )


class HighPredictedRiskGateway(PartialLiveGateway):
    def predict_admet(self, *, smiles):
        self.last_operation_provenance = {
            "method": "versioned live ADMET model",
            "model_version": "admet-test-v1",
            "fixture": False,
        }
        return {
            **ADMET,
            "hepatotoxicity": 0.98,
            "herg_risk": 0.96,
        }


class BoundedBackfillGateway(OptionalDockingUnreadyGateway):
    """Finite, ordered live pool used to verify MA2 replacement semantics."""

    def __init__(self, pool_size: int) -> None:
        PartialLiveGateway.__init__(self, target_linked_activity=True)
        self.pool_size = pool_size
        self.admet_calls = 0
        self.retrieval_calls = 0

    def retrieve_repurposing(self, target, *, direction, limit):
        self.retrieval_calls += 1
        seeds: list[CandidateSeed] = []
        for index in range(self.pool_size):
            direct = index != 0
            assay_type = (
                "measured biochemical single protein assay"
                if direct
                else "cell-based format"
            )
            chembl_id = f"CHEMBL-POOL-{index}"
            evidence = EvidenceRef(
                source="ChEMBL",
                external_id=chembl_id,
                title=f"Finite pool candidate {index}",
                supports="support",
                metadata={
                    "fixture": False,
                    "provenance_class": "external_live",
                    "target_accession": target.uniprot_id,
                    "target_gene": target.symbol,
                    "assay_type": assay_type,
                    "direct_single_protein_engagement": direct,
                },
            )
            seeds.append(
                CandidateSeed(
                    name=f"pool-candidate-{index}",
                    smiles=f"CC{'N' * (index + 1)}",
                    mechanism=f"{direction} {target.symbol}",
                    activities=(
                        ActivityMeasurement(
                            endpoint="IC50",
                            value=float(10 + index),
                            unit="nM",
                            assay_type=assay_type,
                            source_evidence_id=evidence.evidence_id,
                        ),
                    ),
                    evidence=(evidence,),
                    metadata={
                        "fixture": False,
                        "chembl_id": chembl_id,
                        "direction_requested": direction,
                        "direction_evidence_rules": [
                            "measured_ic50_endpoint"
                        ],
                    },
                )
            )
        self.last_operation_provenance = {
            "requested_eligible_count": limit,
            "discovery_pool_limit": min(100, max(4 * limit, 10)),
            "discovery_pool_count": len(seeds),
            "bounded_discovery": True,
            "fixture": False,
        }
        return tuple(seeds)

    def predict_admet(self, *, smiles):
        self.admet_calls += 1
        return super().predict_admet(smiles=smiles)


def _pipeline(tmp_path, brief: ResearchBrief):
    settings = Settings(
        _env_file=None,
        mode=RunMode.MOCK,
        database_path=tmp_path / "contract.sqlite3",
        mock_step_delay_ms=0,
        candidates_per_route=5,
    )
    hypotheses = MA1Agent(settings=settings).run(brief)
    plan = PlannerAgent(settings=settings).run(hypotheses, brief)
    return hypotheses, plan


def _resolved_targets(hypotheses):
    return hypotheses.model_copy(
        update={
            "hypotheses": [
                hypothesis.model_copy(
                    update={
                        "target": hypothesis.target.model_copy(
                            update={"identity_status": "resolved"}
                        )
                    },
                    deep=True,
                )
                for hypothesis in hypotheses.hypotheses
            ]
        },
        deep=True,
    )


def test_live_gateway_accepts_partial_real_result_and_skips_optional_docking(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    gateway = PartialLiveGateway()

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    assert packet.backend_kind is ScientificBackendKind.LIVE
    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len(packet.candidates) == 1
    assert packet.metadata["candidate_yield"] == {
        "requested_limit": plan.candidates_per_route,
        "returned_count": 1,
        "partial": True,
        "fixture_fill_used": False,
        "backend_kind": "live",
        "requested_eligible_count": plan.candidates_per_route,
        "discovery_pool_count": 1,
        "evaluated_count": 1,
        "eligible_count": 1,
        "quota_met": False,
        "pool_exhausted": True,
        "returned_pool_evaluated": True,
        "source_search_complete": True,
        "exhaustive_discovery": False,
    }
    assert gateway.docking_calls == 0
    candidate = packet.candidates[0]
    assert candidate.rank == 1
    assert candidate.metadata["policy_eligible"] is True
    assert candidate.metadata["policy_status_attestation"] == {
        "rule": "policy_eligible_iff_candidate_is_not_excluded_and_ranked",
        "candidate_status": candidate.status,
        "ranked": True,
        "policy_eligible": True,
    }
    assert candidate.docking_execution is not None
    assert candidate.docking_execution.status is ScientificExecutionStatus.SKIPPED
    assert candidate.metrics.docking_score_kcal_mol is None
    assert candidate.admet_execution is not None
    assert candidate.admet_execution.status is ScientificExecutionStatus.COMPLETED
    assert candidate.metadata["ranking_missing_value_policy"] == (
        "available_dimensions_only_no_neutral_imputation"
    )


def test_live_candidate_contract_preserves_bounded_adapter_reason_code(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)

    packet = MA2Agent(
        gateway=StructuredCandidateContractFailureGateway(),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    reason = "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT"
    assert packet.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert packet.candidates == []
    assert packet.abstain_reason_codes == [reason]
    assert packet.metadata["scientific_execution"]["reason_codes"] == [reason]
    failed_call = next(
        item
        for item in packet.tool_calls
        if item.operation == "retrieve_candidates"
    )
    assert failed_call.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert failed_call.provenance["reason_code"] == reason
    assert failed_call.provenance["support_candidate_count"] == 1
    assert failed_call.provenance["rejection_count"] == 11
    wrapper_projection = failed_call.provenance[
        "wrapper_atomic_screen_projection"
    ]
    expected_summary = {
        "reason_code": reason,
        "support_candidate_count": 1,
        "minimum_verified_candidate_count": 3,
        "evidence_grade": "B",
        "promotion_eligible": False,
        "rejection_count": 11,
        "rejections_sha256": "d" * 64,
        "wrapper_atomic_screen_projection": {
            "schema_version": (
                "paper_native_v6_wrapper_atomic_screen_projection/v1"
            ),
            "fixed_pool_candidate_ids": [
                f"CHEMBL{index + 1}" for index in range(12)
            ],
            "row_count": 12,
            "rows": [
                {
                    "fixed_pool_rank": row["fixed_pool_rank"],
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "delegate_returned": row["delegate_returned"],
                    "wrapper_screen_status": row["wrapper_screen_status"],
                    "wrapper_screen_reason_code": row[
                        "wrapper_screen_reason_code"
                    ],
                    "source_primary_reason_code": (
                        None
                        if index == 0
                        else "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED"
                    ),
                    "wrapper_screen_checks": row["wrapper_screen_checks"],
                    "row_sha256": row["row_sha256"],
                }
                for index, row in enumerate(wrapper_projection["rows"])
            ],
            "complete": True,
            "fixture": False,
            "source_reason_projection_verified": True,
            "rejection_reason_counts": {
                "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED": 11
            },
            "fixed_pool_candidate_ids_sha256": wrapper_projection[
                "fixed_pool_candidate_ids_sha256"
            ],
            "rows_sha256": wrapper_projection["rows_sha256"],
            "attestation_sha256": wrapper_projection["attestation_sha256"],
        },
        "raw_provenance_retained_in_tool_call": True,
        "fixture_fallback_used": False,
        "missing_values_imputed": False,
    }
    assert packet.metadata["scientific_failure_summary"] == expected_summary
    assert packet.metadata["scientific_execution"][
        "failure_summary"
    ] == expected_summary


@pytest.mark.parametrize(
    ("reason_code", "support_count", "grade"),
    [
        ("PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT", 0, "C"),
        ("PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID", 4, None),
    ],
)
def test_live_candidate_contract_preserves_c_and_ungraded_failure_summary(
    tmp_path,
    brief: ResearchBrief,
    reason_code,
    support_count,
    grade,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)

    packet = MA2Agent(
        gateway=StructuredCandidateContractFailureGateway(
            reason_code=reason_code,
            support_candidate_count=support_count,
            evidence_grade=grade,
        ),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    assert packet.abstain_reason_codes == [reason_code]
    summary = packet.metadata["scientific_failure_summary"]
    assert summary["reason_code"] == reason_code
    assert summary["support_candidate_count"] == support_count
    assert summary["minimum_verified_candidate_count"] == 3
    assert summary["evidence_grade"] == grade
    assert summary["promotion_eligible"] is False
    assert summary["rejection_count"] == 12 - support_count
    failed_call = next(
        item
        for item in packet.tool_calls
        if item.operation == "retrieve_candidates"
    )
    assert failed_call.provenance["reason_code"] == reason_code


def test_unavailable_reason_code_rejects_unbounded_free_form_value() -> None:
    error = ScientificDataUnavailable("source failed")
    error.reason_code = "not a bounded machine code"

    assert MA2Agent._unavailable_reason_code(error) == (
        "SCIENTIFIC_DATA_UNAVAILABLE"
    )


def test_bounded_pool_backfills_one_exclusion_to_three_eligible_candidates(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    gateway = BoundedBackfillGateway(pool_size=4)

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    eligible = [item for item in packet.candidates if item.status != "excluded"]
    excluded = [item for item in packet.candidates if item.status == "excluded"]
    assert len(packet.candidates) == 4
    assert len(eligible) == 3
    assert [item.rank for item in eligible] == [1, 2, 3]
    assert len(excluded) == 1
    assert "TARGET_ENGAGEMENT_EVIDENCE_UNAVAILABLE" in excluded[0].exclusion_reasons
    audit = packet.metadata["candidate_backfill"]
    assert audit["requested_eligible_count"] == 3
    assert audit["discovery_pool_count"] == 4
    assert audit["evaluated_count"] == 4
    assert audit["eligible_count"] == 3
    assert audit["excluded_count"] == 1
    assert audit["quota_met"] is True
    assert audit["pool_exhausted"] is True
    assert audit["unevaluated_identities"] == []
    assert audit["unbounded_search_used"] is False
    assert gateway.retrieval_calls == 1


def test_bounded_pool_exhaustion_records_honest_partial_candidate_yield(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)

    packet = MA2Agent(
        gateway=BoundedBackfillGateway(pool_size=2),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    audit = packet.metadata["candidate_backfill"]
    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len(packet.candidates) == 2
    assert audit["quota_met"] is False
    assert audit["pool_exhausted"] is True
    assert audit["evaluated_count"] == 2
    assert audit["eligible_count"] == 1
    assert audit["stop_reason"] == "FINITE_DISCOVERY_POOL_EXHAUSTED"
    assert packet.metadata["candidate_yield"]["partial"] is True
    assert packet.metadata["candidate_search_outcome"] == {
        "status": "partial_candidates",
        "reason_code": "ELIGIBLE_CANDIDATE_QUOTA_NOT_MET_POOL_EXHAUSTED",
        "search_completed": True,
    }


def test_only_verified_paper_native_group_enforces_minimum_aggregation_quota() -> None:
    assert MA2Agent._minimum_eligible_aggregation_required(
        {
            "quota_met": False,
            "paper_native_v6_group_projection": {
                "requested_eligible_count": 3,
            },
        }
    )
    assert not MA2Agent._minimum_eligible_aggregation_required(
        {"quota_met": False}
    )


@pytest.mark.parametrize(
    ("registry_id", "expected_reason", "expected_flag", "control_only"),
    [
        (
            "CHEMBL-POOL-0",
            "RESERVED_ASSAY_CONTROL",
            "reserved_assay_control",
            True,
        ),
        (
            "CHEMBL-DIFFERENT",
            "RESERVED_CONTROL_IDENTITY_CONFLICT",
            "reserved_control_identity_conflict",
            False,
        ),
    ],
)
def test_reserved_assay_control_is_never_ranked_and_finite_reserve_fills_quota(
    tmp_path,
    brief: ResearchBrief,
    registry_id: str,
    expected_reason: str,
    expected_flag: str,
    control_only: bool,
) -> None:
    control = ReservedControlCompound(
        name="pool-candidate-0",
        registry_ids=[
            NamespacedRegistryIdentifier(
                namespace="ChEMBL",
                identifier=registry_id,
            )
        ],
    )
    brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={
                    "candidates_per_route": 3,
                    "reserved_control_compounds": [control],
                }
            ),
            "data_policy": brief.data_policy.model_copy(
                update={"require_primary_sources": False}
            ),
        },
        deep=True,
    )
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    gateway = BoundedBackfillGateway(pool_size=5)

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    control_card = next(
        item for item in packet.candidates if item.name == "pool-candidate-0"
    )
    eligible = [item for item in packet.candidates if item.status != "excluded"]
    assert control_card.status == "excluded"
    assert control_card.score == 0.0
    assert control_card.rank is None
    assert control_card.target_rank is None
    assert control_card.portfolio_rank is None
    assert control_card.portfolio_score is None
    assert control_card.exclusion_reasons == [expected_reason]
    assert control_card.metadata["control_only"] is control_only
    assert control_card.metadata["policy_eligible"] is False
    assert control_card.chemistry_execution is None
    assert expected_flag in control_card.safety_flags
    assert "forbidden_drug" not in control_card.safety_flags
    control_calls = [
        call
        for call in packet.tool_calls
        if call.call_id in control_card.tool_trace_ids
    ]
    assert [
        call.operation
        for call in control_calls
        if call.operation == "screen_reserved_assay_control"
    ] == ["screen_reserved_assay_control"]
    assert not any(
        call.tool_name
        in {"chemistry_validator", "docking", "admet_predictor"}
        for call in control_calls
    )
    assert len(eligible) == 3
    assert gateway.admet_calls == 3
    assert gateway.retrieval_calls == 1
    audit = packet.metadata["candidate_backfill"]
    assert audit["quota_met"] is True
    assert audit["pool_exhausted"] is False
    assert audit["evaluated_count"] == 4
    assert len(audit["unevaluated_identities"]) == 1
    assert audit["unevaluated_identities"][0]["name"] == "pool-candidate-4"
    assert audit["unevaluated_identities"][0]["reason_codes"] == [
        "ELIGIBLE_CANDIDATE_QUOTA_MET"
    ]
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.PASS, [
        (item.code, item.message) for item in verdict.findings
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("plan_mismatch", "CANDIDATE_POLICY_MISMATCH"),
        ("metadata_tamper", "RESERVED_CONTROL_SCREEN_MISMATCH"),
        ("promotion", "RESERVED_CONTROL_CANDIDATE_NOT_EXCLUDED"),
        ("exact_call_mismatch", "RESERVED_CONTROL_TRACE_MISSING"),
        ("trace_hash_tamper", "RESERVED_CONTROL_TRACE_MISMATCH"),
        ("compute_after_control", "RESERVED_CONTROL_COMPUTED_AFTER_EXCLUSION"),
        ("order_mismatch", "RESERVED_CONTROL_ORDER_VIOLATION"),
    ],
)
def test_ma3_reserved_control_contract_detects_tamper_call_and_order_mismatch(
    tmp_path,
    brief: ResearchBrief,
    mutation: str,
    expected_code: str,
) -> None:
    control = ReservedControlCompound(
        name="pool-candidate-0",
        registry_ids=[
            NamespacedRegistryIdentifier(
                namespace="ChEMBL",
                identifier="CHEMBL-POOL-0",
            )
        ],
    )
    brief = brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={
                    "candidates_per_route": 3,
                    "reserved_control_compounds": [control],
                }
            ),
            "data_policy": brief.data_policy.model_copy(
                update={"require_primary_sources": False}
            ),
        },
        deep=True,
    )
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    packet = MA2Agent(
        gateway=BoundedBackfillGateway(pool_size=5),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses).model_copy(deep=True)
    control_card = next(
        item for item in packet.candidates if item.name == "pool-candidate-0"
    )
    reserved_call = next(
        call
        for call in packet.tool_calls
        if call.call_id in control_card.tool_trace_ids
        and call.operation == "screen_reserved_assay_control"
    )
    compute_call = next(
        call
        for call in packet.tool_calls
        if call.tool_name
        in {"chemistry_validator", "docking", "admet_predictor"}
    )

    if mutation == "plan_mismatch":
        plan = plan.model_copy(
            update={"reserved_control_compounds": []},
            deep=True,
        )
    elif mutation == "metadata_tamper":
        control_card.metadata["reserved_control_screen"]["matched"] = False
    elif mutation == "promotion":
        control_card.status = "conditional"
        control_card.score = 0.5
        control_card.rank = 1
        control_card.target_rank = 1
        control_card.metadata["policy_eligible"] = True
    elif mutation == "exact_call_mismatch":
        call_index = packet.tool_calls.index(reserved_call)
        packet.tool_calls[call_index] = reserved_call.model_copy(
            update={"operation": "screen_forbidden_drug"},
            deep=True,
        )
    elif mutation == "trace_hash_tamper":
        call_index = packet.tool_calls.index(reserved_call)
        packet.tool_calls[call_index] = reserved_call.model_copy(
            update={"output_hash": "0" * 64},
            deep=True,
        )
    elif mutation in {"compute_after_control", "order_mismatch"}:
        control_card.tool_trace_ids.append(compute_call.call_id)
        if mutation == "order_mismatch":
            call_index = packet.tool_calls.index(reserved_call)
            packet.tool_calls[call_index] = reserved_call.model_copy(
                update={
                    "finished_at": compute_call.started_at
                    + timedelta(seconds=1)
                },
                deep=True,
            )

    findings = MA3AssuranceAgent()._audit_g1(
        brief,
        hypotheses,
        plan,
        packet,
    )
    matched = [item for item in findings if item.code == expected_code]

    assert matched
    assert all(item.severity.value == "CRITICAL" for item in matched)


def test_predicted_admet_liabilities_are_conditional_not_confirmatory_exclusions(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)

    packet = MA2Agent(
        gateway=HighPredictedRiskGateway(),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    candidate = packet.candidates[0]
    assert candidate.status == "conditional"
    assert candidate.rank == 1
    assert candidate.exclusion_reasons == []
    assert {"hepatotoxicity_high", "herg_risk_high"}.issubset(
        candidate.safety_flags
    )
    assert candidate.metadata["admet_decision_policy"] == (
        "prediction_as_warning_not_confirmatory_exclusion_v1"
    )
    assert candidate.metadata[
        "admet_prediction_requires_experimental_confirmation"
    ] is True


def test_mixed_live_portfolio_passes_with_conclusive_zero_hit_branches(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    # This legacy portfolio-contract fixture does not model publication
    # retrieval. Keep its original execution-contract assertion explicit by
    # disabling the independent primary-source policy for this test only.
    brief = brief.model_copy(
        update={
            "data_policy": brief.data_policy.model_copy(
                update={"require_primary_sources": False}
            )
        }
    )
    hypotheses, plan = _pipeline(tmp_path, brief)
    hypotheses = _resolved_targets(hypotheses)
    candidate_symbol = hypotheses.hypotheses[0].target.symbol

    packet = MA2Agent(
        gateway=ConclusivePortfolioGateway(candidate_symbol),
        chemistry=StubRDKitChemistry(),
        ma2_portfolio_enabled=True,
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len([item for item in packet.candidates if item.status != "excluded"]) == 1
    assert sum(item.status == "completed" for item in packet.target_executions) == 1
    assert sum(item.status == "no_candidates" for item in packet.target_executions) == (
        packet.eligible_target_count - 1
    )
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.PASS, [
        (item.code, item.message) for item in verdict.findings
    ]


def test_typed_chembl_timeout_is_isolated_to_one_portfolio_target(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    hypotheses = _resolved_targets(hypotheses)
    executable = [
        item
        for item in hypotheses.hypotheses
        if not MA2Agent._select_route(item, item.target)
        .get("fallback", {})
        .get("abstain_recommended", False)
    ]
    assert len(executable) >= 2
    candidate_hypothesis, timeout_hypothesis = executable[:2]
    gateway = TimeoutIsolatedPortfolioGateway(
        candidate_hypothesis.target.symbol,
        timeout_hypothesis.target.symbol,
    )

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
        ma2_portfolio_enabled=True,
    ).run(plan, hypotheses)

    completed = next(
        item
        for item in packet.target_executions
        if item.hypothesis_id == candidate_hypothesis.hypothesis_id
    )
    timed_out = next(
        item
        for item in packet.target_executions
        if item.hypothesis_id == timeout_hypothesis.hypothesis_id
    )
    assert completed.status == "completed"
    assert timed_out.status == "unavailable"
    assert timed_out.reason_codes == ["SCIENTIFIC_DATA_UNAVAILABLE"]
    assert packet.execution_status is ScientificExecutionStatus.PARTIAL
    assert "PORTFOLIO_PARTIAL_TARGET_COVERAGE" in packet.abstain_reason_codes
    assert any(
        item.status != "excluded"
        and item.target.symbol == candidate_hypothesis.target.symbol
        for item in packet.candidates
    )


def test_all_zero_hit_live_portfolio_abstains_without_tool_failure(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    hypotheses = _resolved_targets(hypotheses)

    packet = MA2Agent(
        gateway=ConclusivePortfolioGateway(None),
        chemistry=StubRDKitChemistry(),
        ma2_portfolio_enabled=True,
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert packet.candidates == []
    assert all(
        item.status == "no_candidates"
        for item in packet.target_executions
        if item.execution_eligible
    )
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.ABSTAIN
    codes = {item.code for item in verdict.findings}
    assert "NO_QUALIFIED_CANDIDATES" in codes
    assert "SCIENTIFIC_TOOL_UNAVAILABLE" not in codes


def test_missing_optional_profile_records_not_assessed_and_still_runs_admet(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    gateway = OptionalDockingUnreadyGateway()

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len(packet.candidates) == 1
    candidate = packet.candidates[0]
    assert candidate.status == "excluded"
    assert candidate.rank is None
    assert candidate.metadata["policy_eligible"] is False
    assert candidate.metadata["policy_status_attestation"] == {
        "rule": "policy_eligible_iff_candidate_is_not_excluded_and_ranked",
        "candidate_status": "excluded",
        "ranked": False,
        "policy_eligible": False,
    }
    assert "TARGET_ENGAGEMENT_EVIDENCE_UNAVAILABLE" in candidate.exclusion_reasons
    assert candidate.docking_execution is not None
    assert candidate.docking_execution.status is ScientificExecutionStatus.SKIPPED
    assert candidate.docking_execution.reason_code == "OPTIONAL_DOCKING_NOT_ASSESSED"
    assert candidate.docking_execution.score_kcal_mol is None
    assert candidate.admet_execution is not None
    assert candidate.admet_execution.status is ScientificExecutionStatus.COMPLETED
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.ABSTAIN
    codes = {item.code for item in verdict.findings}
    assert "NO_QUALIFIED_CANDIDATES" in codes
    assert "SCIENTIFIC_TOOL_UNAVAILABLE" not in codes


def test_cell_based_ic50_cannot_waive_docking_or_claim_direct_engagement(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    gateway = CellBasedOptionalDockingUnreadyGateway()

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    candidate = packet.candidates[0]
    assert candidate.status == "excluded"
    assert candidate.activities[0].assay_type == "cell-based format"
    assert candidate.docking_execution is not None
    assert candidate.docking_execution.reason_code == (
        "OPTIONAL_DOCKING_NOT_ASSESSED"
    )
    assert "TARGET_ENGAGEMENT_EVIDENCE_UNAVAILABLE" in candidate.exclusion_reasons
    assert candidate.metadata["activity_strength_status"] == (
        "cellular_activity_available_direct_target_engagement_unresolved"
    )
    assert candidate.metadata["target_engagement_assessment"] == (
        "cellular_activity_only_direct_target_engagement_not_established"
    )
    assert "direct single-protein engagement not established" in candidate.mechanism
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert "DOCKING_SKIP_BASIS_INVALID" not in {
        finding.code for finding in verdict.findings
    }


def test_direct_activity_skips_docking_only_when_evidence_matches_target(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    gateway = PartialLiveGateway(
        target_linked_activity=False,
        docking_unavailable=True,
    )

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    assert gateway.docking_calls == 1
    assert packet.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert packet.candidates == []
    assert packet.abstain_reason_codes == ["SCIENTIFIC_TOOL_UNAVAILABLE"]


def test_conflicting_stable_target_id_cannot_be_rescued_by_matching_gene_symbol(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    gateway = PartialLiveGateway(
        target_linked_activity=True,
        conflicting_target_identifiers=True,
        docking_unavailable=True,
    )

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    assert gateway.docking_calls == 1
    assert packet.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert packet.abstain_reason_codes == ["SCIENTIFIC_TOOL_UNAVAILABLE"]


def test_ma3_independently_rejects_docking_skip_with_conflicting_stable_target_id(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    packet = MA2Agent(
        gateway=PartialLiveGateway(),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)
    candidate = packet.candidates[0]
    tampered_evidence = [
        item.model_copy(
            update={
                "metadata": {
                    **item.metadata,
                    "target_accession": "P99999",
                    "target_gene": candidate.target.symbol,
                }
            },
            deep=True,
        )
        for item in candidate.evidence
    ]
    packet = packet.model_copy(
        update={
            "candidates": [
                candidate.model_copy(
                    update={"evidence": tampered_evidence},
                    deep=True,
                )
            ]
        },
        deep=True,
    )

    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )

    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "DOCKING_SKIP_BASIS_INVALID" in {
        finding.code for finding in verdict.findings
    }


def test_target_linked_activity_does_not_skip_plan_required_docking(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    required_plan = plan.model_copy(
        update={
            "tasks": [
                task.model_copy(
                    update={
                        "tools": [
                            assignment.model_copy(update={"required": True})
                            if assignment.tool_name == "docking"
                            else assignment
                            for assignment in task.tools
                        ]
                    }
                )
                for task in plan.tasks
            ]
        },
        deep=True,
    )
    gateway = PartialLiveGateway(docking_unavailable=True)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        required_plan,
        hypotheses,
    )

    assert gateway.docking_calls == 1
    assert packet.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert packet.abstain_reason_codes == ["SCIENTIFIC_TOOL_UNAVAILABLE"]


def test_live_portfolio_retains_route_abstention_without_executing_branch(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    hypotheses = _resolved_targets(hypotheses).model_copy(deep=True)
    candidate_hypothesis = hypotheses.hypotheses[0]
    fallback_hypothesis = next(
        item
        for item in reversed(hypotheses.hypotheses)
        if item.target.symbol != candidate_hypothesis.target.symbol
    )
    fallback_hypothesis.target.metadata.update(
        {
            "target_discovery_source": "external_dynamic",
            "association_score": 0.72,
            "evidence_summary": {
                "known_drug_score": None,
                "overall_association_score": 0.72,
                "tractability": [],
            },
        }
    )
    for evidence in fallback_hypothesis.evidence:
        evidence.metadata = {
            **evidence.metadata,
            "mock": False,
            "provenance_class": "external_live",
        }
    gateway = ConclusivePortfolioGateway(candidate_hypothesis.target.symbol)

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
        ma2_portfolio_enabled=True,
    ).run(plan, hypotheses)

    summary = next(
        item
        for item in packet.target_executions
        if item.hypothesis_id == fallback_hypothesis.hypothesis_id
    )
    assert summary.status == "ineligible"
    assert summary.execution_eligible is False
    assert summary.attempted is False
    assert summary.reason_codes == ["ROUTE_EVIDENCE_INSUFFICIENT"]
    assert summary.route_decision["fallback"]["abstain_recommended"] is True
    assert fallback_hypothesis.target.symbol not in gateway.queried_symbols
    assert packet.execution_status is ScientificExecutionStatus.COMPLETED

    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert "PORTFOLIO_ROUTE_POLICY_VIOLATION" not in {
        finding.code for finding in verdict.findings
    }
    assert "PORTFOLIO_COMPLETENESS_MISMATCH" not in {
        finding.code for finding in verdict.findings
    }


def test_portfolio_rework_cannot_launder_unavailable_candidate_to_completed(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    hypotheses = _resolved_targets(hypotheses)
    gateway = ConclusivePortfolioGateway(hypotheses.hypotheses[0].target.symbol)
    agent = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
        ma2_portfolio_enabled=True,
    )
    previous = agent.run(plan, hypotheses)
    candidate = next(item for item in previous.candidates if item.status != "excluded")
    unavailable_docking = candidate.docking_execution.model_copy(
        update={
            "status": ScientificExecutionStatus.UNAVAILABLE,
            "reason_code": "DOCKING_RUNTIME_UNAVAILABLE",
        },
        deep=True,
    )
    candidate = candidate.model_copy(
        update={"docking_execution": unavailable_docking},
        deep=True,
    )
    previous = previous.model_copy(
        update={
            "candidates": [
                candidate if item.candidate_id == candidate.candidate_id else item
                for item in previous.candidates
            ]
        },
        deep=True,
    )
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.judge",
        affected_candidate_ids=[candidate.candidate_id],
        required_actions=["Correct the mechanism direction wording."],
    )

    repaired = agent.rework(plan, hypotheses, previous, order)

    summary = next(
        item
        for item in repaired.target_executions
        if candidate.candidate_id in item.candidate_ids
    )
    assert repaired.execution_status is ScientificExecutionStatus.PARTIAL
    assert "DOCKING_RUNTIME_UNAVAILABLE" in repaired.abstain_reason_codes
    assert summary.status == "unavailable"
    assert summary.reason_codes == ["DOCKING_RUNTIME_UNAVAILABLE"]
    assert repaired.metadata["scientific_execution"]["status"] == "partial"


def test_incomplete_live_admet_fails_closed_and_ma3_abstains(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    packet = MA2Agent(
        gateway=PartialLiveGateway(incomplete_admet=True),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert packet.abstain_reason_codes == ["ADMET_ENDPOINTS_INCOMPLETE_OR_INVALID"]
    assert packet.candidates[0].rank is None
    assert packet.candidates[0].metrics.admet_scores == {}
    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.ABSTAIN
    assert "SCIENTIFIC_TOOL_UNAVAILABLE" in {
        finding.code for finding in verdict.findings
    }


def test_explicit_backend_setting_selects_fixture_or_live_gateway(tmp_path) -> None:
    fixture = build_ma2_gateway(
        Settings(_env_file=None, database_path=tmp_path / "fixture.sqlite3")
    )
    live = build_ma2_gateway(
        Settings(
            _env_file=None,
            database_path=tmp_path / "live.sqlite3",
            ma2_backend_mode="live",
            drugcentral_dataset_path=tmp_path / "missing.tsv",
        )
    )

    assert isinstance(fixture, DeterministicFixtureGateway)
    assert isinstance(live, PublicRepurposingGateway)
    assert fixture.fixture is True
    assert live.fixture is False


def test_ma3_blocks_tampered_tool_input_provenance(
    tmp_path,
    brief: ResearchBrief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    packet = MA2Agent(
        gateway=PartialLiveGateway(),
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)
    first = packet.tool_calls[0]
    tampered = first.model_copy(
        update={"provenance": {**first.provenance, "input_hash": "0" * 64}}
    )
    packet = packet.model_copy(
        update={"tool_calls": [tampered, *packet.tool_calls[1:]]},
        deep=True,
    )

    verdict = asyncio.run(
        MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet)
    )
    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "EXECUTION_INPUT_HASH_MISMATCH" in {
        finding.code for finding in verdict.findings
    }
