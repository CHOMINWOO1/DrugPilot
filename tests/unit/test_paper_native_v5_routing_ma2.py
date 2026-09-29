from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import _paper_v5_group_projection
from drugpilot.domain.models import (
    ActivityMeasurement,
    EvidenceRef,
    GateName,
    NamespacedRegistryIdentifier,
    ReworkOrder,
    ReservedControlCompound,
    ScientificExecutionStatus,
    TargetCandidate,
)
from drugpilot.orchestration.graph import (
    _repurposing_evidence_policy_for_priorities,
    build_ma2_gateway,
)
from drugpilot.tools.live_repurposing_gateway import (
    PAPER_NATIVE_V5_REQUIRED_POLICY,
    PAIRED_V4_REQUIRED_POLICY,
    PaperNativeV5RepurposingContractUnavailable,
    PublicRepurposingGateway,
)
from drugpilot.tools.protocols import CandidateSeed
from test_ma2_live_builder_health import (
    _FakeADMET,
    _FakeDocking,
    _FakeScientificGateway,
    _live_settings,
)
from test_ma2_live_execution_contract import (
    PartialLiveGateway,
    StubRDKitChemistry,
    _pipeline,
)
from test_live_scientific_tools import (
    _PAPER_NATIVE_V5_GROUP_FOUR,
    _PaperNativeV5PubMedClient,
    _run_paper_native_v5,
)


_GROUP_SHA256 = "a" * 64


def _support(candidate_id: str, index: int) -> dict[str, object]:
    return {
        "molecule_chembl_id": candidate_id,
        "fixed_pool_rank": index + 1,
        "retained_rank": index + 1,
        "retained_index": index,
        "activity_id": 7000 + index,
        "source_evidence_id": f"ev_chembl_activity_{candidate_id}",
        "support_attestation_sha256": f"{index + 1:064x}",
        "typed_measurement_tuple": {
            "endpoint": "IC50",
            "relation": "=",
            "value": float(50 - index * 5),
            "unit": "nM",
            "assay_type": "measured biochemical single protein assay",
            "source_evidence_id": f"ev_chembl_activity_{candidate_id}",
        },
    }


def _base_seed(candidate_id: str, index: int) -> CandidateSeed:
    evidence_id = f"ev_chembl_activity_{candidate_id}"
    evidence = EvidenceRef(
        evidence_id=evidence_id,
        source="ChEMBL",
        external_id=str(7000 + index),
        title=f"Exact direct activity for {candidate_id}",
        supports="support",
        metadata={
            "fixture": False,
            "live": True,
            "provenance_class": "external_live",
            "target_gene": "FLT3",
            "target_type": "SINGLE PROTEIN",
            "target_organism": "Homo sapiens",
            "direct_single_protein_engagement": True,
        },
    )
    return CandidateSeed(
        name=f"candidate-{index}",
        smiles=f"CC{'N' * (index + 1)}",
        mechanism="measured FLT3 inhibitory activity",
        activities=(
            ActivityMeasurement(
                endpoint="IC50",
                relation="=",
                value=float(50 - index * 5),
                unit="nM",
                assay_type="measured biochemical single protein assay",
                source_evidence_id=evidence_id,
            ),
        ),
        evidence=(evidence,),
        metadata={
            "fixture": False,
            "live": True,
            "chembl_id": candidate_id,
            "drug_id": candidate_id,
            "direction_requested": "inhibit",
            "direction_evidence_rules": ["measured_ic50_endpoint"],
        },
    )


def _v5_contract(
    *,
    verified_ids: tuple[str, ...] = ("CHEMBL102", "CHEMBL103", "CHEMBL104"),
    declared_member_ids: tuple[str, ...] | None = None,
    requested_limit: int = 3,
    ids: tuple[str, ...] = (
        "CHEMBL101",
        "CHEMBL102",
        "CHEMBL103",
        "CHEMBL104",
        "CHEMBL105",
    ),
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    declared = declared_member_ids or verified_ids
    supports = [_support(candidate_id, index) for index, candidate_id in enumerate(ids)]
    group_pairs = [
        {
            "molecule_chembl_id": candidate_id,
            "evidence_sha256": f"{index + 20:064x}",
        }
        for index, candidate_id in enumerate(declared)
    ]
    bounds: dict[str, object] = {
        "schema_version": "bounded_mechanism_first_candidate_identity_pool_v1",
        "discovery_lane": "chembl_mechanism_first_single_default_v1",
        "target_type": "SINGLE PROTEIN",
        "target_organism": "Homo sapiens",
        "mechanism_action_type": "INHIBITOR",
        "raw_mechanism_row_limit": 80,
        "mechanism_complete_within_bound": True,
        "prewarning_pool_limit": 12,
        "fixed_prewarning_pool_limit": 12,
        "prewarning_pool_count": len(ids),
        "prewarning_pool_ids": list(ids),
        "preactivity_pool_ids": list(ids),
        "preactivity_pool": [
            {
                "rank": index + 1,
                "parent_molecule_chembl_id": candidate_id,
                "max_phase": 4,
            }
            for index, candidate_id in enumerate(ids)
        ],
        "activity_query_attempted_ids": list(ids),
        "prewarning_selection_policy": (
            "clinical_max_phase_desc_then_parent_molecule_chembl_id_v1"
        ),
        "selection_inputs": ["max_phase", "parent_molecule_chembl_id"],
        "pchembl_used_for_prewarning_selection": False,
        "primary_document_used_for_prewarning_selection": False,
        "warning_used_for_prewarning_selection": False,
        "activity_used_for_prewarning_selection": False,
        "activity_candidate_query_limit": 12,
        "no_activity_first_fallback": True,
        "returned_candidate_ordering_policy": (
            "surviving_fixed_prewarning_clinical_phase_desc_then_"
            "stable_parent_molecule_chembl_id_subsequence_v1"
        ),
        "requested_eligible_count": requested_limit,
        "gateway_return_candidate_count": len(ids),
        "gateway_return_candidate_ids": list(ids),
        "downstream_candidate_identity_count": len(ids),
        "downstream_candidate_identity_ids": list(ids),
        "gateway_returned_from_fixed_prewarning_pool": True,
        "gateway_return_order_matches_fixed_prewarning_subsequence": True,
        "gateway_return_cap_applied": False,
        "eligible_quota_evaluation_owner": "MA2",
        "no_backfill_beyond_prewarning_pool": True,
        "fixture": False,
        "missing_values_imputed": False,
        "paper_native_v5_direct_support_candidate_ids": list(ids),
        "paper_native_v5_direct_support_snapshots": deepcopy(supports),
        "paper_native_v5_direct_support_inventory": {
            "records": [
                {
                    "molecule_chembl_id": candidate_id,
                    "status": "selected",
                    "reason_codes": [],
                }
                for candidate_id in ids
            ]
        },
        "primary_abstract_search": {
            "schema_version": "primary_abstract_paper_native_gateway_v5",
            "policy": PAPER_NATIVE_V5_REQUIRED_POLICY,
            "no_backfill": True,
            "fixture": False,
            "promotion": {
                "status": "verified",
                "verified_member_candidate_ids": list(verified_ids),
                "selected_group_sha256": _GROUP_SHA256,
            },
        },
    }

    contracted: list[CandidateSeed] = []
    for index, candidate_id in enumerate(ids):
        verified = candidate_id in verified_ids
        envelope: dict[str, object] = {
            "schema_version": "candidate_primary_evidence_paper_native_v5",
            "source_contract": "primary_abstract_paper_native_v5",
            "molecule_chembl_id": candidate_id,
            "verification_status": "verified" if verified else "not_verified",
            "reason_code": None if verified else "NOT_IN_VERIFIED_GROUP",
            "fixture": False,
            "missing_values_imputed": False,
            "cross_source_numeric_equality_required": False,
            "pubmed_value_used_for_activity_selection": False,
            "pubmed_value_used_for_candidate_rank": False,
            "pubmed_value_used_for_candidate_ordering": False,
        }
        if verified:
            pair_index = (
                list(declared).index(candidate_id) if candidate_id in declared else 0
            )
            selected_pair = deepcopy(group_pairs[pair_index])
            pair_hash = str(selected_pair["evidence_sha256"])
            envelope.update(
                {
                    "selected_group_sha256": _GROUP_SHA256,
                    "group": {
                        "group_sha256": _GROUP_SHA256,
                        "candidate_ids": list(declared),
                        "pairs": deepcopy(group_pairs),
                    },
                    "group_member_candidate_ids": list(declared),
                    "selected_pair_index": pair_index,
                    "selected_pair_evidence_sha256": pair_hash,
                    "selected_pair": selected_pair,
                    "pre_pubmed_chembl_support": deepcopy(supports[index]),
                    # A deliberately unrelated PubMed value must not enter the
                    # typed ChEMBL activity or candidate ranking.
                    "paper_native_measurement": {
                        "value_raw": str(1 + index),
                        "unit_raw": "nM",
                    },
                }
            )
        seed = _base_seed(candidate_id, index)
        seed_metadata = {
            **seed.metadata,
            "candidate_discovery_bounds": deepcopy(bounds),
            "primary_abstract_paper_native_v5": deepcopy(envelope),
        }
        evidence = tuple(
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "candidate_discovery_bounds": deepcopy(bounds),
                        "primary_abstract_paper_native_v5": deepcopy(envelope),
                    }
                },
                deep=True,
            )
            for item in seed.evidence
        )
        contracted.append(replace(seed, metadata=seed_metadata, evidence=evidence))
    provenance: dict[str, object] = {
        "operation": "retrieve_repurposing",
        "status": "completed",
        "source": "ChEMBL live API",
        "candidate_count": len(contracted),
        "requested_limit": requested_limit,
        "requested_eligible_count": requested_limit,
        "fixed_prewarning_pool_limit": 12,
        "gateway_return_candidate_ids": list(ids),
        "gateway_return_cap_applied": False,
        "gateway_return_order_matches_fixed_prewarning_subsequence": True,
        "eligible_quota_evaluation_owner": "MA2",
        "candidate_discovery_bounds": deepcopy(bounds),
        "fixture": False,
    }
    return tuple(contracted), provenance


class _Delegate:
    source_version = "paper-native-v5-test"
    backend_kind = "live"
    fixture = False

    def __init__(self, seeds: tuple[CandidateSeed, ...], provenance: dict[str, object]):
        self.seeds = seeds
        self.last_operation_provenance = provenance
        self.requests: list[tuple[str, str, int]] = []

    def retrieve_repurposing(self, target, *, direction, limit):
        self.requests.append((target.symbol, direction, limit))
        return self.seeds


class _NoLocalLookup:
    def lookup(self, request):  # pragma: no cover - a call is the assertion failure
        raise AssertionError("paper-native-v5 must not execute local lookup")


def _wrapper(tmp_path, seeds, provenance):
    return PublicRepurposingGateway(
        drugcentral_dataset_path=tmp_path / "not-consulted.tsv",
        pubchem=_NoLocalLookup(),
        chembl=_NoLocalLookup(),
        bindingdb=_NoLocalLookup(),
        delegate=_Delegate(seeds, provenance),
        repurposing_evidence_policy=PAPER_NATIVE_V5_REQUIRED_POLICY,
    )


def test_exact_v5_wrapper_returns_delegate_tuple_once_without_local_fallback(tmp_path):
    seeds, provenance = _v5_contract()
    gateway = _wrapper(tmp_path, seeds, provenance)

    result = gateway.retrieve_repurposing(
        TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
        direction="inhibit",
        limit=3,
    )

    assert result == seeds
    assert all(actual is expected for actual, expected in zip(result, seeds, strict=True))
    assert gateway.delegate.requests == [("FLT3", "inhibit", 3)]
    audit = gateway.last_operation_provenance
    assert audit["delegate_pool_preserved_exactly"] is True
    assert audit["local_candidate_discovery_executed"] is False
    assert audit["candidate_merge_used"] is False
    assert audit["candidate_reranking_used"] is False
    assert audit["candidate_replacement_used"] is False
    assert audit["fixture_fallback_used"] is False


def test_exact_v5_wrapper_accepts_actual_live_scientific_gateway_contract(tmp_path):
    produced, _chembl, scientific = _run_paper_native_v5(
        _PaperNativeV5PubMedClient({"91000001": _PAPER_NATIVE_V5_GROUP_FOUR})
    )
    seeds = tuple(produced)
    gateway = _wrapper(
        tmp_path,
        seeds,
        deepcopy(scientific.last_operation_provenance),
    )

    result = gateway.retrieve_repurposing(
        TargetCandidate(symbol="GENE", uniprot_id="P12345"),
        direction="inhibit",
        limit=3,
    )

    assert result == seeds
    assert gateway.last_operation_provenance[
        "paper_native_v5_verified_candidate_ids"
    ] == ["CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104"]


def test_exact_v5_wrapper_rejects_missing_group_member_without_fallback(tmp_path):
    seeds, provenance = _v5_contract(
        declared_member_ids=("CHEMBL102", "CHEMBL103", "CHEMBL999")
    )
    gateway = _wrapper(tmp_path, seeds, provenance)

    with pytest.raises(PaperNativeV5RepurposingContractUnavailable):
        gateway.retrieve_repurposing(
            TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
            direction="inhibit",
            limit=3,
        )

    assert gateway.delegate.requests == [("FLT3", "inhibit", 3)]
    assert gateway.last_operation_provenance == {}


def test_exact_v5_wrapper_rejects_activity_not_bound_to_prepubmed_support(tmp_path):
    seeds, provenance = _v5_contract()
    member = seeds[1]
    tampered_activity = member.activities[0].model_copy(update={"value": 2.0})
    tampered = (
        seeds[0],
        replace(member, activities=(tampered_activity,)),
        *seeds[2:],
    )
    gateway = _wrapper(tmp_path, tampered, provenance)

    with pytest.raises(PaperNativeV5RepurposingContractUnavailable):
        gateway.retrieve_repurposing(
            TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
            direction="inhibit",
            limit=3,
        )

    assert gateway.last_operation_provenance == {}


class _V5MA2Gateway(PartialLiveGateway):
    repurposing_evidence_policy = PAPER_NATIVE_V5_REQUIRED_POLICY

    def __init__(self, seeds: tuple[CandidateSeed, ...]) -> None:
        super().__init__(target_linked_activity=True)
        self.seeds = seeds
        self.retrieval_calls = 0
        self.admet_calls = 0

    def retrieve_repurposing(self, target, *, direction, limit):
        self.retrieval_calls += 1
        self.last_operation_provenance = {
            "requested_eligible_count": limit,
            "discovery_pool_limit": 12,
            "discovery_pool_count": len(self.seeds),
            "bounded_discovery": True,
            "fixture": False,
        }
        return self.seeds

    def predict_admet(self, *, smiles):
        self.admet_calls += 1
        return super().predict_admet(smiles=smiles)


def _with_reserved_candidates(brief, *candidate_indexes: int):
    controls = [
        ReservedControlCompound(
            name=f"candidate-{index}",
            registry_ids=[
                NamespacedRegistryIdentifier(
                    namespace="ChEMBL",
                    identifier=f"CHEMBL{101 + index}",
                )
            ],
        )
        for index in candidate_indexes
    ]
    return brief.model_copy(
        update={
            "constraints": brief.constraints.model_copy(
                update={
                    "candidates_per_route": 3,
                    "reserved_control_compounds": controls,
                }
            )
        },
        deep=True,
    )


def _projection(packet):
    return packet.metadata["candidate_backfill"][
        "paper_native_v5_group_projection"
    ]


def test_ma2_v5_excludes_nonmember_then_continues_within_fixed_pool_quota(
    tmp_path,
    brief,
):
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract()
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert gateway.retrieval_calls == 1
    assert gateway.admet_calls == 3
    assert len(packet.candidates) == 4
    first = next(item for item in packet.candidates if item.name == "candidate-0")
    assert first.status == "excluded"
    assert first.score == 0.0
    assert first.rank is None
    assert first.exclusion_reasons == ["CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"]
    assert first.chemistry_execution is None
    eligible = [item for item in packet.candidates if item.status != "excluded"]
    assert len(eligible) == 3
    assert sorted((item.activities[0].value, item.rank) for item in eligible) == [
        (35.0, 3),
        (40.0, 2),
        (45.0, 1),
    ]
    assert [item.metadata["chembl_id"] for item in eligible] == [
        "CHEMBL102",
        "CHEMBL103",
        "CHEMBL104",
    ]
    assert all(
        item.metadata["candidate_ordering_policy"]
        == "fixed_prewarning_pool_order_v1"
        and item.metadata["score_used_for_candidate_ordering"] is False
        for item in packet.candidates
    )
    assert all(
        item.metadata["paper_native_v5_primary_screen"][
            "pubmed_values_used_for_candidate_activity_or_ranking"
        ]
        is False
        for item in eligible
    )
    audit = packet.metadata["candidate_backfill"]
    assert audit["requested_eligible_count"] == 3
    assert audit["discovery_pool_count"] == 5
    assert audit["evaluated_count"] == 4
    assert audit["quota_met"] is True
    assert audit["unbounded_search_used"] is False
    assert audit["unevaluated_identities"][0]["pool_index"] == 4
    assert audit["unevaluated_identities"][0]["reason_codes"] == [
        "ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"
    ]
    projection = _projection(packet)
    assert projection["declared_group_candidate_ids"] == [
        "CHEMBL102",
        "CHEMBL103",
        "CHEMBL104",
    ]
    assert projection["projected_group_candidate_ids"] == [
        "CHEMBL102",
        "CHEMBL103",
        "CHEMBL104",
    ]
    assert projection["projection_complete"] is True
    assert projection["stop_condition_satisfied"] is True
    assert audit["stop_reason"] == (
        "ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"
    )
    assert packet.metadata["candidate_yield"][
        "paper_native_v5_group_projection"
    ] == projection
    assert all(
        item.metadata["paper_native_v5_group_projection"] == projection
        for item in packet.candidates
    )
    for item in packet.candidates:
        prospective = item.metadata["prospective_falsification_status"]
        assert prospective["schema_version"] == (
            "candidate_prospective_falsification_v3"
        )
        assert prospective["status"] == "not_evaluated"
        assert prospective["prospective_evidence_disposition"] == (
            "unobserved_no_outcome"
        )
        assert prospective["prospective_follow_up_support"] is False
        assert prospective["default_falsification_applied"] is False
        assert prospective["biological_refutation_established"] is False
        assert prospective["experimentally_observed"] is False
        assert prospective["used_for_source_eligibility"] is False
        assert prospective["used_for_candidate_ranking"] is False
        assert prospective["machine_readable_contract"]["status"] == (
            "unstructured"
        )
        assert len(prospective["falsification_test_sha256"]) == 64
        assert len(prospective["attestation_sha256"]) == 64


def test_ma2_v5_group_four_quota_three_projects_fourth_before_stopping(
    tmp_path,
    brief,
):
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract(
        verified_ids=("CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104")
    )
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    projection = _projection(packet)
    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len(packet.candidates) == 4
    assert len([item for item in packet.candidates if item.status != "excluded"]) == 4
    assert gateway.admet_calls == 4
    assert projection["eligible_returned_count"] == 4
    assert projection["requested_eligible_count"] == 3
    assert projection["projected_group_candidate_ids"] == [
        "CHEMBL101",
        "CHEMBL102",
        "CHEMBL103",
        "CHEMBL104",
    ]
    assert projection["projection_complete"] is True
    assert packet.metadata["candidate_backfill"]["evaluated_count"] == 4
    assert packet.metadata["candidate_backfill"]["unevaluated_identities"][0][
        "reason_codes"
    ] == ["ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"]


def test_ma2_v5_single_target_portfolio_preserves_group_attestation(
    tmp_path,
    brief,
):
    hypotheses, plan = _pipeline(tmp_path, brief)
    primary = hypotheses.hypotheses[0]
    hypotheses = hypotheses.model_copy(
        update={
            "hypotheses": [
                primary.model_copy(
                    update={
                        "target": primary.target.model_copy(
                            update={"identity_status": "resolved"}
                        )
                    },
                    deep=True,
                )
            ]
        },
        deep=True,
    )
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    fixed_ids = ["CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104"]
    seeds, _ = _v5_contract(
        ids=tuple(fixed_ids),
        verified_ids=tuple(fixed_ids),
    )
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
        ma2_portfolio_enabled=True,
    ).run(plan, hypotheses)

    backfill = packet.metadata["candidate_backfill"]
    candidate_yield = packet.metadata["candidate_yield"]
    projection = backfill["paper_native_v5_group_projection"]
    assert candidate_yield["paper_native_v5_group_projection"] == projection
    assert _paper_v5_group_projection(
        packet,
        declared_group_ids=fixed_ids,
        fixed_ids=fixed_ids,
    ) is True


def test_ma2_v5_single_target_portfolio_rework_preserves_group_attestation(
    tmp_path,
    brief,
):
    hypotheses, plan = _pipeline(tmp_path, brief)
    primary = hypotheses.hypotheses[0]
    hypotheses = hypotheses.model_copy(
        update={
            "hypotheses": [
                primary.model_copy(
                    update={
                        "target": primary.target.model_copy(
                            update={"identity_status": "resolved"}
                        )
                    },
                    deep=True,
                )
            ]
        },
        deep=True,
    )
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    fixed_ids = ["CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104"]
    seeds, _ = _v5_contract(
        ids=tuple(fixed_ids),
        verified_ids=tuple(fixed_ids),
    )
    gateway = _V5MA2Gateway(seeds)
    agent = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
        ma2_portfolio_enabled=True,
    )
    packet = agent.run(plan, hypotheses)
    original_projection = deepcopy(_projection(packet))
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.admet",
        affected_candidate_ids=[packet.candidates[0].candidate_id],
        required_actions=["Repeat the affected candidate ADMET prediction."],
    )

    repaired = agent.rework(plan, hypotheses, packet, order)

    assert repaired.metadata["candidate_backfill"][
        "paper_native_v5_group_projection"
    ] == original_projection
    assert repaired.metadata["candidate_yield"][
        "paper_native_v5_group_projection"
    ] == original_projection
    assert _paper_v5_group_projection(
        repaired,
        declared_group_ids=fixed_ids,
        fixed_ids=fixed_ids,
    ) is True


def test_ma2_joint_prospective_falsification_contract_is_machine_readable(
    tmp_path,
    brief,
):
    hypotheses, plan = _pipeline(tmp_path, brief)
    rule = (
        "각 후보는 개별적으로 평가한다. 최소 3개의 독립 생물학적 반복에서 "
        "후보별로 급성 생존율 감소가 20% 이하인 노출에서 2시간 pSTAT5가 "
        "vehicle 대비 50% 이상 감소하고 72시간 viable-cell count가 vehicle "
        "대비 30% 이상 감소하며 각 방향성 효과의 95% 신뢰구간이 0을 제외할 "
        "때만 지지한다."
    )
    primary = hypotheses.hypotheses[0]
    hypotheses = hypotheses.model_copy(
        update={
            "hypotheses": [
                primary.model_copy(update={"falsification_test": rule}, deep=True)
            ]
        },
        deep=True,
    )
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract()
    packet = MA2Agent(
        gateway=_V5MA2Gateway(seeds), chemistry=StubRDKitChemistry()
    ).run(plan, hypotheses)

    status = packet.candidates[0].metadata["prospective_falsification_status"]
    contract = status["machine_readable_contract"]
    assert contract["status"] == "machine_readable"
    assert contract["criteria_conjunction"] == "all"
    assert contract["when_unobserved"] == "not_evaluated"
    assert contract["evaluation_status"] == "not_evaluated"
    assert "default_when_unobserved_or_any_criterion_unmet" not in contract
    assert contract["criteria"][
        "minimum_independent_biological_replicates_per_directional_endpoint"
    ] == 3
    qualifying = contract["criteria"]["qualifying_exposure"]
    assert {
        key: qualifying[key]
        for key in (
            "same_exposure_required_for",
            "acute_viability_loss_relation",
            "acute_viability_loss_threshold_percent",
        )
    } == {
        "same_exposure_required_for": [
            "early_target_modulation",
            "acute_viability_loss",
            "late_viable_cell_change",
        ],
        "acute_viability_loss_relation": "<=",
        "acute_viability_loss_threshold_percent": 20,
    }
    assert qualifying["minimum_independent_biological_replicates"] == 3
    assert qualifying["confidence_interval_level_percent"] == 95
    assert qualifying["confidence_interval_must_exclude_zero"] is True
    assert qualifying["selection_rule"] == (
        "lowest_pre_specified_nonzero_exposure_satisfying_"
        "acute_viability_ceiling_before_directional_outcome_review"
    )
    assert qualifying["selection_rule_researcher_authored"] is False
    assert qualifying[
        "exposure_grid_must_be_pre_specified_before_outcome_review"
    ] is True
    assert qualifying["exposure_grid"] is None
    assert qualifying["exposure_grid_status"] == (
        "required_not_supplied_fail_closed"
    )
    early, late = contract["criteria"]["directional_endpoints"]
    assert (early["readout"], early["timepoint_hours"], early["effect_threshold_percent"]) == (
        "pSTAT5",
        2,
        50,
    )
    assert (late["readout"], late["timepoint_hours"], late["effect_threshold_percent"]) == (
        "viable-cell count",
        72,
        30,
    )
    assert early["confidence_interval_level_percent"] == 95
    assert late["confidence_interval_must_exclude_zero"] is True
    assert contract["observations"]["qualifying_exposure"]["value"] is None
    assert contract["observations"]["qualifying_exposure"][
        "acute_viability_biological_replicate_count"
    ] is None
    assert contract["observations"]["qualifying_exposure"][
        "acute_viability_confidence_interval_95"
    ] == {"lower": None, "upper": None, "excludes_zero": None}
    assert contract["observations"]["qualifying_exposure"][
        "late_endpoint_same_exposure"
    ] is None
    assert contract["observations"]["qualifying_exposure"][
        "tested_exposure_grid"
    ] is None
    assert contract["observations"]["early_target_modulation"][
        "confidence_interval_95"
    ]["excludes_zero"] is None
    assert status["prospective_follow_up_support"] is False
    assert status["prospective_evidence_disposition"] == (
        "unobserved_no_outcome"
    )


def test_ma2_v5_reserved_fourth_is_projected_excluded_after_quota(
    tmp_path,
    brief,
):
    brief = _with_reserved_candidates(brief, 3)
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract(
        verified_ids=("CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104")
    )
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    fourth = next(item for item in packet.candidates if item.name == "candidate-3")
    assert fourth.status == "excluded"
    assert fourth.rank is None
    assert fourth.score == 0.0
    assert fourth.chemistry_execution is None
    assert fourth.exclusion_reasons == ["RESERVED_ASSAY_CONTROL"]
    assert gateway.admet_calls == 3
    projection = _projection(packet)
    assert projection["projected_group_candidate_ids"] == [
        "CHEMBL101",
        "CHEMBL102",
        "CHEMBL103",
        "CHEMBL104",
    ]
    assert projection["eligible_returned_count"] == 3
    assert projection["projection_complete"] is True


def test_ma2_v5_reserved_among_first_three_continues_to_fourth_eligible(
    tmp_path,
    brief,
):
    brief = _with_reserved_candidates(brief, 1)
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract(
        verified_ids=("CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104")
    )
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    reserved = next(item for item in packet.candidates if item.name == "candidate-1")
    assert reserved.status == "excluded"
    assert reserved.rank is None
    assert next(
        item for item in packet.candidates if item.name == "candidate-3"
    ).status != "excluded"
    assert [
        (item.metadata["chembl_id"], item.rank)
        for item in packet.candidates
        if item.status != "excluded"
    ] == [
        ("CHEMBL101", 1),
        ("CHEMBL103", 2),
        ("CHEMBL104", 3),
    ]
    assert gateway.admet_calls == 3
    projection = _projection(packet)
    assert projection["eligible_returned_count"] == 3
    assert projection["evaluated_fixed_pool_prefix_count"] == 4
    assert projection["stop_condition_satisfied"] is True


def test_ma2_v5_noncontiguous_group_projects_fixed_pool_order(
    tmp_path,
    brief,
):
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract(
        ids=(
            "CHEMBL101",
            "CHEMBL102",
            "CHEMBL103",
            "CHEMBL104",
            "CHEMBL105",
            "CHEMBL106",
        ),
        verified_ids=("CHEMBL101", "CHEMBL103", "CHEMBL105"),
        declared_member_ids=("CHEMBL105", "CHEMBL101", "CHEMBL103"),
    )
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    projection = _projection(packet)
    assert projection["declared_group_candidate_ids"] == [
        "CHEMBL105",
        "CHEMBL101",
        "CHEMBL103",
    ]
    assert projection["projected_group_candidate_ids"] == [
        "CHEMBL101",
        "CHEMBL103",
        "CHEMBL105",
    ]
    assert projection["evaluated_fixed_pool_prefix_count"] == 5
    assert packet.metadata["candidate_backfill"]["unevaluated_identities"][0][
        "pool_index"
    ] == 5


def test_ma2_v5_quota_not_met_after_group_projection_exhausts_fixed_pool(
    tmp_path,
    brief,
):
    brief = _with_reserved_candidates(brief, 0, 1)
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract(
        verified_ids=("CHEMBL101", "CHEMBL102", "CHEMBL103")
    )
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    projection = _projection(packet)
    assert projection["projection_complete"] is True
    assert projection["eligible_quota_reached"] is False
    assert projection["stop_condition_satisfied"] is False
    assert projection["evaluated_fixed_pool_prefix_count"] == 5
    assert packet.metadata["candidate_backfill"]["pool_exhausted"] is True
    assert packet.metadata["candidate_backfill"]["stop_reason"] == (
        "FINITE_DISCOVERY_POOL_EXHAUSTED"
    )
    assert packet.metadata["candidate_backfill"]["unevaluated_identities"] == []


def test_ma2_v5_missing_declared_member_makes_whole_execution_unavailable(
    tmp_path,
    brief,
):
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    seeds, _ = _v5_contract(
        declared_member_ids=("CHEMBL102", "CHEMBL103", "CHEMBL999")
    )
    gateway = _V5MA2Gateway(seeds)

    packet = MA2Agent(gateway=gateway, chemistry=StubRDKitChemistry()).run(
        plan,
        hypotheses,
    )

    assert packet.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert packet.candidates == []
    assert packet.abstain_reason_codes == ["PAPER_NATIVE_V5_GROUP_UNAVAILABLE"]
    assert gateway.retrieval_calls == 1
    assert gateway.admet_calls == 0


def test_exact_policy_routing_rejects_conflict_and_ignores_near_tokens():
    assert _repurposing_evidence_policy_for_priorities(
        [PAPER_NATIVE_V5_REQUIRED_POLICY]
    ) == PAPER_NATIVE_V5_REQUIRED_POLICY
    assert _repurposing_evidence_policy_for_priorities(
        [f"{PAPER_NATIVE_V5_REQUIRED_POLICY}_extra"]
    ) is None
    with pytest.raises(ValueError, match="Conflicting exact"):
        _repurposing_evidence_policy_for_priorities(
            [PAPER_NATIVE_V5_REQUIRED_POLICY, PAIRED_V4_REQUIRED_POLICY]
        )


def test_builder_passes_v5_policy_only_to_exact_scientific_delegate(tmp_path):
    settings = _live_settings(tmp_path)
    exact = build_ma2_gateway(
        settings,
        repurposing_evidence_policy=PAPER_NATIVE_V5_REQUIRED_POLICY,
        scientific_gateway_factory=_FakeScientificGateway,
        docking_factory=_FakeDocking,
        admet_factory=_FakeADMET,
        profile_registry_factory=lambda path: {"registry": path.name},
    )
    near = build_ma2_gateway(
        settings,
        repurposing_evidence_policy=f"{PAPER_NATIVE_V5_REQUIRED_POLICY}_extra",
        scientific_gateway_factory=_FakeScientificGateway,
        docking_factory=_FakeDocking,
        admet_factory=_FakeADMET,
        profile_registry_factory=lambda path: {"registry": path.name},
    )

    assert exact.delegate.kwargs["repurposing_evidence_policy"] == (
        PAPER_NATIVE_V5_REQUIRED_POLICY
    )
    assert "repurposing_evidence_policy" not in near.delegate.kwargs
