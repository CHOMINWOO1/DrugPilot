from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from drugpilot.domain.models import ActivityMeasurement, EvidenceRef, TargetCandidate
from drugpilot.tools.live_repurposing_gateway import (
    PAPER_NATIVE_V6_REQUIRED_POLICY,
    PaperNativeV6RepurposingContractUnavailable,
    PublicRepurposingGateway,
    _paper_native_v6_delegate_seed_source_projection,
)
from drugpilot.tools.primary_abstract_paper_native_search_v6 import (
    PaperNativeV6ContractError,
)
from drugpilot.tools.protocols import CandidateSeed


IDS = ("CHEMBL10", "CHEMBL20", "CHEMBL30", "CHEMBL40")


def _canonical_sha256(value) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _mechanism_identity(candidate_id: str) -> dict:
    rows = [{
        "canonical_parent_molecule_chembl_id": candidate_id,
        "target_chembl_id": "CHEMBL1974",
        "action_type": "INHIBITOR",
        "direct_interaction": 1,
        "molecular_mechanism": 1,
    }]
    return {
        "schema_version": "chembl_mechanism_identity_v1",
        "parent_molecule_chembl_id": candidate_id,
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
        "fixture": False,
        "missing_values_imputed": False,
    }


def _seed(candidate_id: str, rank: int) -> CandidateSeed:
    evidence_id = f"ev-{rank}"
    activity_id = str(9001 + rank)
    document_id = f"CHEMBL{700001 + rank}"
    excerpt = (
        f"Drug {rank} inhibits FLT3 with an experimentally measured "
        "IC50 of 7 nM."
    )
    assay_type = "single protein"
    assay_context = "direct_single_protein_context"
    mechanism_identity = _mechanism_identity(candidate_id)
    mechanism_claim = (
        "ChEMBL-curated inhibitor mechanism and measured direct "
        "IC50 activity against FLT3"
    )
    primary = {
        "schema_version": "candidate_primary_evidence_v2",
        "verification_status": "verified",
        "is_primary_support": True,
        "reason_code": None,
        "document_chembl_id": document_id,
        "activity_binding": {
            "activity_id": activity_id,
            "document_chembl_id": document_id,
            "molecule_chembl_id": candidate_id,
            "target_chembl_id": "CHEMBL1974",
            "endpoint": "IC50", "relation": "=", "value": 7.0, "unit": "nM",
            "assay_type": assay_type,
            "assay_context": assay_context,
            "direct_single_protein_engagement": True,
            "direction_evidence_rule": "measured_ic50_endpoint",
            "mechanism_identity_sha256": _canonical_sha256(
                mechanism_identity
            ),
            "mechanism_claim": mechanism_claim,
            "mechanism_claim_sha256": _canonical_sha256(mechanism_claim),
            "deposited_activity": {"typed_exact_match": True},
        },
        "activity_document_selection": {
            "policy": "complete_exact_parent_activity_document_rows_v1",
            "top_k": None,
            "result_row_cap": None,
            "eligible_row_count": 1,
            "inspected_row_count": 1,
            "truncated": False,
            "all_eligible_rows_inspected": True,
            "selected_rank": 1,
            "selected_activity_id": activity_id,
            "selected_by_pre_pubmed_paired_v4_cohort": False,
            "paired_v4_selection_precedes_pubmed_search": False,
            "selected_by_candidate_primary_record_v6": True,
            "paper_native_v6_paired_search_or_selection_used": False,
            "selected_by_pre_pubmed_paper_native_v5_direct_support": False,
            "paper_native_v5_selection_precedes_pubmed_search": False,
            "paper_native_v5_pubmed_value_used_for_selection": False,
            "pchembl_scope": (
                "bounded_within_molecule_document_inspection_only"
            ),
            "pchembl_used_for_candidate_ordering": False,
            "rows": [{
                "rank": 1,
                "activity_id": activity_id,
                "document_chembl_id": document_id,
                "endpoint": "IC50",
                "pchembl_value": 7.0,
                "verification_status": "verified",
                "reason_code": None,
            }],
            "fixture": False,
            "missing_values_imputed": False,
        },
        "pubmed_type_verification": {
            "verification_status": "verified", "retrieval_status": "completed",
            "pmid": str(100000 + rank),
        },
        "publication_types": ["Journal Article"],
        "excerpt": excerpt,
        "excerpt_hash": __import__(
            "hashlib"
        ).sha256(excerpt.encode("utf-8")).hexdigest(),
        "excerpt_location": {"section": "abstract", "sentence_index": rank},
        "pmid": str(100000 + rank),
    }
    evidence = EvidenceRef(
        evidence_id=evidence_id, source="ChEMBL", external_id=activity_id,
        title="measured activity", supports="support",
        metadata={
            "fixture": False,
            "primary_source": primary,
            "assay_type": assay_type,
            "assay_context": assay_context,
            "direct_single_protein_engagement": True,
            "intervention_direction_requested": "inhibit",
            "direction_evidence_rule": "measured_ic50_endpoint",
            "mechanism_identity": deepcopy(mechanism_identity),
        },
    )
    activity = ActivityMeasurement(
        endpoint="IC50", relation="=", value=7.0, unit="nM",
        assay_type="single protein", source_evidence_id=evidence_id,
    )
    return CandidateSeed(
        name=f"Drug {rank}", smiles="C" * (rank + 1), mechanism=mechanism_claim,
        activities=(activity,), evidence=(evidence,),
        metadata={
            "chembl_id": candidate_id, "drug_id": candidate_id,
            "chembl_exact_aliases": [f"Drug {rank}"], "fixture": False,
            "activity_assay_type": assay_type,
            "activity_assay_context": assay_context,
            "direct_single_protein_engagement": True,
            "direction_requested": "inhibit",
            "direction_evidence_rules": ["measured_ic50_endpoint"],
            "mechanism_identity": deepcopy(mechanism_identity),
        },
    )


def _bounds(ids=IDS):
    target_id = "CHEMBL1974"
    counts = {}
    audits_by_id = {}
    for index, candidate_id in enumerate(ids, 1):
        activity_ids = [str(9000 + index)]
        params = {
            "target_chembl_id": target_id,
            "parent_molecule_chembl_id": candidate_id,
            "target_organism": "Homo sapiens",
            "assay_type": "B",
            "standard_type": "IC50",
            "pchembl_value__gte": 5.0,
            "order_by": "activity_id",
            "limit": 20,
        }
        page = {
            "page_index": 0,
            "offset": 0,
            "limit": 20,
            "response_count": 1,
            "reported_total_count": 1,
            "activity_ids": activity_ids,
            "activity_ids_sha256": _canonical_sha256(activity_ids),
            "response_rows_sha256": f"{index:064x}",
            "query": {"fixture": False},
            "query_parameters": params,
        }
        page["page_audit_sha256"] = _canonical_sha256(page)
        audits_by_id[candidate_id] = [page]
        counts[candidate_id] = {
            "total_count": 1,
            "response_count": 1,
            "inspected_count": 1,
            "truncated": False,
            "complete_result_set": True,
            "page_size": 20,
            "page_count": 1,
            "page_audits": deepcopy(audits_by_id[candidate_id]),
            "page_audits_sha256": _canonical_sha256(
                audits_by_id[candidate_id]
            ),
            "response_rows_sha256": f"{index + 100:064x}",
            "eligible_directional_count": 1,
            "eligible_directional_activity_ids": activity_ids,
            "eligible_directional_activity_ids_sha256": _canonical_sha256(
                activity_ids
            ),
            "retained_document_rows": 1,
            "retained_document_activity_ids": activity_ids,
            "retained_document_activity_ids_sha256": _canonical_sha256(
                activity_ids
            ),
            "primary_document_audit_result_cap": None,
            "all_eligible_rows_retained_for_primary_audit": True,
        }
    bounds = {
        "schema_version": "bounded_mechanism_first_candidate_identity_pool_v1",
        "discovery_lane": "chembl_mechanism_first_single_default_v1",
        "target_chembl_id": target_id,
        "mechanism_provenance_by_id": {
            candidate_id: _mechanism_identity(candidate_id)
            for candidate_id in ids
        },
        "prewarning_pool_ids": list(ids),
        "activity_complete_pagination_required": True,
        "activity_rows_per_candidate_limit_semantics": "wire_page_size_not_result_cap",
        "activity_result_row_limit": None,
        "activity_query_attempted_count": len(ids),
        "activity_query_attempted_ids": list(ids),
        "activity_query_success_count": len(ids),
        "activity_failure_count": 0,
        "activity_failures_by_id": {},
        "activity_counts_by_id": counts,
        "activity_page_audits_by_id": audits_by_id,
        "activity_page_audits_by_id_sha256": _canonical_sha256(audits_by_id),
        "activity_logical_call_count": len(ids),
        "no_backfill_beyond_prewarning_pool": True,
        "fixture": False,
    }
    source_inventory = {
        "schema_version": "scientific_source_call_outcome_inventory_v1",
        "required_stages": [],
        "stages": {},
        "successful_call_count": 0,
        "failure_count": 0,
        "all_failure_records_represented": True,
        "fixture": False,
    }
    source_inventory["inventory_sha256"] = _canonical_sha256(
        source_inventory
    )
    bounds["source_call_inventory"] = source_inventory
    return bounds


class Delegate:
    fixture = False

    def __init__(self, seeds, provenance):
        self.seeds = tuple(seeds)
        self.provenance = provenance
        self.last_operation_provenance = {}

    def retrieve_repurposing(self, target, *, direction, limit):
        self.last_operation_provenance = deepcopy(self.provenance)
        return self.seeds


def _provenance(ids=IDS):
    return {
        "operation": "retrieve_repurposing", "status": "completed",
        "fixture": False, "requested_limit": 3, "requested_eligible_count": 3,
        "gateway_return_cap_applied": False,
        "eligible_quota_evaluation_owner": "MA2",
        "candidate_discovery_bounds": _bounds(ids),
    }


def _gateway(tmp_path, seeds, provenance=None):
    effective_provenance = deepcopy(provenance or _provenance())
    bounds = effective_provenance["candidate_discovery_bounds"]
    if "delegate_seed_source_binding_projection" not in bounds:
        projection = _paper_native_v6_delegate_seed_source_projection(
            fixed_candidate_ids=bounds["prewarning_pool_ids"],
            seeds=seeds,
        )
        bounds["delegate_seed_source_binding_projection"] = projection
    effective_provenance.setdefault(
        "delegate_seed_source_binding_projection",
        deepcopy(bounds["delegate_seed_source_binding_projection"]),
    )
    delegate = Delegate(seeds, effective_provenance)
    return PublicRepurposingGateway(
        drugcentral_dataset_path=tmp_path / "unused.tsv", delegate=delegate,
        repurposing_evidence_policy=PAPER_NATIVE_V6_REQUIRED_POLICY,
    )


def _retrieve(gateway):
    return gateway.retrieve_repurposing(
        TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
        direction="inhibit", limit=3,
    )


def test_all_verified_are_returned_in_fixed_order_without_cap(tmp_path):
    gateway = _gateway(tmp_path, [_seed(cid, i) for i, cid in enumerate(IDS)])
    rows = _retrieve(gateway)
    assert [row.metadata["chembl_id"] for row in rows] == list(IDS)
    assert gateway.last_operation_provenance["gateway_return_candidate_ids"] == list(IDS)
    assert gateway.last_operation_provenance["gateway_return_cap_applied"] is False
    assert gateway.last_operation_provenance["candidate_reranking_used"] is False
    assert gateway.last_operation_provenance["evidence_grade"] == "A"
    assert gateway.last_operation_provenance["promotion_eligible"] is True
    assert gateway.last_operation_provenance["support_candidate_count"] == len(
        IDS
    )
    assert gateway.last_operation_provenance[
        "minimum_verified_candidate_count"
    ] == 3
    final_bounds = gateway.last_operation_provenance["candidate_discovery_bounds"]
    assert final_bounds["gateway_return_candidate_ids"] == list(IDS)
    assert final_bounds["gateway_return_candidate_count"] == len(IDS)
    assert final_bounds["gateway_return_order_matches_fixed_prewarning_subsequence"] is True
    for rank, row in enumerate(rows, 1):
        envelope = row.metadata["primary_abstract_paper_native_v6"]
        assert envelope["fixed_pool_rank"] == rank
        assert envelope["molecule_chembl_id"] == IDS[rank - 1]
        assert envelope["primary_source_sha256"] == envelope["candidate_support_unit"]["primary_source_attestation_sha256"]


def test_v6_raw_delegate_seed_audit_snapshot_is_bounded_and_one_shot(tmp_path):
    gateway = _gateway(
        tmp_path, [_seed(cid, i) for i, cid in enumerate(IDS)]
    )
    _retrieve(gateway)
    snapshot = gateway.consume_paper_native_v6_delegate_seed_audit_snapshot()
    assert len(snapshot) == len(IDS) <= 12
    assert [seed.metadata["chembl_id"] for seed in snapshot] == list(IDS)
    assert all(len(seed.activities) == 1 for seed in snapshot)
    assert all(len(seed.evidence) == 1 for seed in snapshot)
    assert all(
        isinstance(seed.evidence[0].metadata.get("primary_source"), dict)
        for seed in snapshot
    )
    assert gateway.consume_paper_native_v6_delegate_seed_audit_snapshot() == ()


def test_one_invalid_is_rejected_but_three_verified_survive(tmp_path):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    bad = seeds[1].evidence[0].model_copy(update={"metadata": {
        **seeds[1].evidence[0].metadata,
        "primary_source": {**seeds[1].evidence[0].metadata["primary_source"], "is_primary_support": False},
    }}, deep=True)
    seeds[1] = replace(seeds[1], evidence=(bad,))
    gateway = _gateway(tmp_path, seeds)
    assert [row.metadata["chembl_id"] for row in _retrieve(gateway)] == [IDS[0], IDS[2], IDS[3]]
    final_bounds = gateway.last_operation_provenance["candidate_discovery_bounds"]
    assert final_bounds["delegate_gateway_return_candidate_ids"] == list(IDS)
    assert final_bounds["gateway_return_candidate_ids"] == [IDS[0], IDS[2], IDS[3]]
    screening = final_bounds["wrapper_atomic_screen_projection"]
    assert (
        gateway.last_operation_provenance[
            "wrapper_atomic_screen_projection"
        ]
        == screening
    )
    rejected_screen = screening["rows"][1]
    assert rejected_screen["wrapper_screen_status"] == "rejected"
    assert rejected_screen["wrapper_screen_reason_code"] == (
        "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
    )
    assert rejected_screen["wrapper_screen_checks"] == {
        "cardinality_exact": True,
        "primary_mapping": True,
        "exact_binding": False,
    }
    disposition = final_bounds["primary_abstract_search"][
        "candidate_primary_dispositions"
    ][1]
    assert disposition["source_primary_verification_status"] == "verified"
    assert disposition["source_primary_verification_reason_code"] is None
    assert disposition["source_primary_attestation_sha256"] == (
        final_bounds["delegate_seed_source_binding_projection"]["rows"][1][
            "primary_attestation_sha256"
        ]
    )
    assert disposition["wrapper_screen_row_sha256"] == rejected_screen[
        "row_sha256"
    ]


def test_wrapper_screen_projection_covers_not_returned_fixed_seed(tmp_path):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS[:3])]
    gateway = _gateway(tmp_path, seeds)
    assert [row.metadata["chembl_id"] for row in _retrieve(gateway)] == list(
        IDS[:3]
    )
    screening = gateway.last_operation_provenance[
        "wrapper_atomic_screen_projection"
    ]
    assert screening["row_count"] == len(IDS)
    assert [row["molecule_chembl_id"] for row in screening["rows"]] == list(
        IDS
    )
    missing = screening["rows"][-1]
    assert missing["wrapper_screen_status"] == "not_returned"
    assert missing["wrapper_screen_reason_code"] == "DELEGATE_NOT_RETURNED"
    assert missing["wrapper_screen_checks"] == {
        "cardinality_exact": False,
        "primary_mapping": False,
        "exact_binding": False,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_reason", "expected_primary_mapping"),
    [
        (
            "cardinality",
            "SEED_CARDINALITY_NOT_EXACT",
            True,
        ),
        (
            "primary",
            "PRIMARY_SOURCE_NOT_MAPPING",
            False,
        ),
    ],
)
def test_wrapper_screen_projection_preserves_atomic_rejection_reason(
    tmp_path, mutation, expected_reason, expected_primary_mapping
):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    if mutation == "cardinality":
        seeds[0] = replace(seeds[0], activities=())
    else:
        evidence = seeds[0].evidence[0].model_copy(
            update={
                "metadata": {
                    **seeds[0].evidence[0].metadata,
                    "primary_source": None,
                }
            },
            deep=True,
        )
        seeds[0] = replace(seeds[0], evidence=(evidence,))
    gateway = _gateway(tmp_path, seeds)
    assert [row.metadata["chembl_id"] for row in _retrieve(gateway)] == list(
        IDS[1:]
    )
    screen = gateway.last_operation_provenance[
        "wrapper_atomic_screen_projection"
    ]["rows"][0]
    assert screen["wrapper_screen_status"] == "rejected"
    assert screen["wrapper_screen_reason_code"] == expected_reason
    assert screen["wrapper_screen_checks"]["primary_mapping"] is (
        expected_primary_mapping
    )


@pytest.mark.parametrize("invalid_count", [2, 3])
def test_one_or_two_support_rows_are_grade_b_and_never_promoted(
    tmp_path, invalid_count,
):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    for index in range(invalid_count):
        bad = seeds[index].evidence[0].model_copy(update={"metadata": {
            **seeds[index].evidence[0].metadata, "primary_source": None,
        }}, deep=True)
        seeds[index] = replace(seeds[index], evidence=(bad,))
    with pytest.raises(PaperNativeV6RepurposingContractUnavailable) as caught:
        _retrieve(_gateway(tmp_path, seeds))
    exc = caught.value
    assert exc.reason_code == "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT"
    expected_support_count = len(IDS) - invalid_count
    assert exc.provenance["support_candidate_count"] == expected_support_count
    assert exc.provenance["minimum_verified_candidate_count"] == 3
    assert exc.provenance["evidence_grade"] == "B"
    assert exc.provenance["promotion_eligible"] is False
    assert len(exc.provenance["rejections_sha256"]) == 64
    screening = exc.provenance["wrapper_atomic_screen_projection"]
    assert screening["complete"] is True
    assert screening["row_count"] == len(IDS)
    assert [
        row["wrapper_screen_status"] for row in screening["rows"]
    ] == [
        *(["rejected"] * invalid_count),
        *(["verified"] * expected_support_count),
    ]
    assert len(screening["attestation_sha256"]) == 64


def test_zero_exact_primary_support_is_grade_c_and_never_promoted(tmp_path):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    for index, seed in enumerate(seeds):
        evidence = seed.evidence[0].model_copy(
            update={
                "metadata": {
                    **seed.evidence[0].metadata,
                    "primary_source": None,
                }
            },
            deep=True,
        )
        seeds[index] = replace(seed, evidence=(evidence,))
    with pytest.raises(PaperNativeV6RepurposingContractUnavailable) as caught:
        _retrieve(_gateway(tmp_path, seeds))
    exc = caught.value
    assert exc.reason_code == "PAPER_NATIVE_V6_INSUFFICIENT_PRIMARY_SUPPORT"
    assert exc.provenance["support_candidate_count"] == 0
    assert exc.provenance["evidence_grade"] == "C"
    assert exc.provenance["promotion_eligible"] is False
    assert len(exc.provenance["rejections_sha256"]) == 64
    screening = exc.provenance["wrapper_atomic_screen_projection"]
    assert all(
        row["wrapper_screen_status"] == "rejected"
        for row in screening["rows"]
    )
    assert len(screening["attestation_sha256"]) == 64


def test_three_or_more_support_rows_with_invalid_aggregate_contract_are_not_graded(
    tmp_path, monkeypatch,
):
    def fail_aggregation(**_kwargs):
        raise PaperNativeV6ContractError(
            "SUPPORT_MAP_ORDER_OR_MEMBERSHIP_MISMATCH"
        )

    monkeypatch.setattr(
        "drugpilot.tools.live_repurposing_gateway."
        "aggregate_candidate_primary_support_v6",
        fail_aggregation,
    )
    gateway = _gateway(
        tmp_path, [_seed(cid, i) for i, cid in enumerate(IDS)]
    )
    with pytest.raises(PaperNativeV6RepurposingContractUnavailable) as caught:
        _retrieve(gateway)
    exc = caught.value
    assert (
        exc.reason_code
        == "PAPER_NATIVE_V6_PRIMARY_SUPPORT_CONTRACT_INVALID"
    )
    assert exc.provenance["support_candidate_count"] == len(IDS)
    assert exc.provenance["evidence_grade"] is None
    assert exc.provenance["promotion_eligible"] is False
    assert exc.provenance["contract_error_code_sha256"] == sha256(
        b"SUPPORT_MAP_ORDER_OR_MEMBERSHIP_MISMATCH"
    ).hexdigest()


def test_v6_wrapper_rejects_selection_count_not_bound_to_retained_rows(
    tmp_path,
):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    for index in (0, 1):
        evidence = seeds[index].evidence[0]
        metadata = deepcopy(evidence.metadata)
        metadata["primary_source"]["activity_document_selection"][
            "eligible_row_count"
        ] = 2
        seeds[index] = replace(
            seeds[index],
            evidence=(
                evidence.model_copy(update={"metadata": metadata}, deep=True),
            ),
        )
    with pytest.raises(PaperNativeV6RepurposingContractUnavailable):
        _retrieve(_gateway(tmp_path, seeds))


def test_v6_wrapper_rejects_self_rehashed_retained_id_not_in_page(
    tmp_path,
):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    provenance = _provenance()
    count = provenance["candidate_discovery_bounds"][
        "activity_counts_by_id"
    ][IDS[0]]
    count["eligible_directional_activity_ids"] = ["999"]
    count["eligible_directional_activity_ids_sha256"] = _canonical_sha256(
        ["999"]
    )
    count["retained_document_activity_ids"] = ["999"]
    count["retained_document_activity_ids_sha256"] = _canonical_sha256(
        ["999"]
    )
    with pytest.raises(PaperNativeV6RepurposingContractUnavailable):
        _retrieve(_gateway(tmp_path, seeds, provenance))


@pytest.mark.parametrize(
    "tamper",
    [
        "status",
        "cap",
        "order",
        "activity_failure",
        "page_audit",
        "attestation",
        "binding",
    ],
)
def test_provenance_order_and_attestation_tamper_fail_closed(tmp_path, tamper):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    provenance = _provenance()
    if tamper == "status": provenance["status"] = "partial"
    elif tamper == "cap": provenance["gateway_return_cap_applied"] = True
    elif tamper == "order": provenance["candidate_discovery_bounds"]["prewarning_pool_ids"] = list(reversed(IDS))
    elif tamper == "activity_failure":
        provenance["candidate_discovery_bounds"]["activity_failure_count"] = 1
    elif tamper == "page_audit":
        provenance["candidate_discovery_bounds"]["activity_page_audits_by_id"][IDS[0]][0]["response_count"] = 0
    elif tamper == "attestation":
        primary = deepcopy(seeds[0].evidence[0].metadata["primary_source"])
        primary["pubmed_type_verification"]["verification_status"] = "unverified"
        evidence = seeds[0].evidence[0].model_copy(update={"metadata": {**seeds[0].evidence[0].metadata, "primary_source": primary}}, deep=True)
        seeds[0] = replace(seeds[0], evidence=(evidence,))
        primary = deepcopy(seeds[1].evidence[0].metadata["primary_source"])
        primary["pubmed_type_verification"]["verification_status"] = "unverified"
        evidence = seeds[1].evidence[0].model_copy(update={"metadata": {**seeds[1].evidence[0].metadata, "primary_source": primary}}, deep=True)
        seeds[1] = replace(seeds[1], evidence=(evidence,))
    elif tamper == "binding":
        primary = deepcopy(seeds[0].evidence[0].metadata["primary_source"])
        primary["activity_binding"]["value"] = 999
        evidence = seeds[0].evidence[0].model_copy(update={"metadata": {**seeds[0].evidence[0].metadata, "primary_source": primary}}, deep=True)
        seeds[0] = replace(seeds[0], evidence=(evidence,))
        primary = deepcopy(seeds[1].evidence[0].metadata["primary_source"])
        primary["activity_binding"]["value"] = 999
        evidence = seeds[1].evidence[0].model_copy(update={"metadata": {**seeds[1].evidence[0].metadata, "primary_source": primary}}, deep=True)
        seeds[1] = replace(seeds[1], evidence=(evidence,))
    with pytest.raises(PaperNativeV6RepurposingContractUnavailable):
        _retrieve(_gateway(tmp_path, seeds, provenance))


def _tamper_assay_mechanism_seed(seed: CandidateSeed, mutation: str) -> CandidateSeed:
    if mutation == "activity_assay_type":
        activity = seed.activities[0].model_copy(
            update={"assay_type": "cell-based format"}
        )
        return replace(seed, activities=(activity,))
    if mutation == "candidate_context":
        return replace(seed, metadata={
            **seed.metadata,
            "activity_assay_context": "cellular_or_system_context",
        })
    if mutation == "candidate_mechanism":
        return replace(seed, mechanism="rewritten claim")
    evidence = seed.evidence[0]
    evidence_metadata = deepcopy(evidence.metadata)
    if mutation == "evidence_direct":
        evidence_metadata["direct_single_protein_engagement"] = False
    elif mutation == "evidence_mechanism_identity":
        evidence_metadata["mechanism_identity"] = _mechanism_identity(
            "CHEMBL999"
        )
    else:
        primary = deepcopy(evidence_metadata["primary_source"])
        binding = primary["activity_binding"]
        if mutation == "binding_assay_type":
            binding["assay_type"] = "cell-based format"
        elif mutation == "binding_context":
            binding["assay_context"] = "cellular_or_system_context"
        elif mutation == "binding_direct":
            binding["direct_single_protein_engagement"] = False
        elif mutation == "binding_direction":
            binding["direction_evidence_rule"] = (
                "measured_exact_target_activity"
            )
        elif mutation == "binding_mechanism_hash":
            binding["mechanism_identity_sha256"] = "0" * 64
        elif mutation == "binding_claim":
            binding["mechanism_claim"] = "rewritten claim"
            binding["mechanism_claim_sha256"] = _canonical_sha256(
                "rewritten claim"
            )
        else:
            raise AssertionError(mutation)
        evidence_metadata["primary_source"] = primary
    changed = evidence.model_copy(
        update={"metadata": evidence_metadata}, deep=True
    )
    return replace(seed, evidence=(changed,))


@pytest.mark.parametrize(
    "mutation",
    [
        "binding_assay_type",
        "binding_context",
        "binding_direct",
        "binding_direction",
        "binding_mechanism_hash",
        "binding_claim",
        "activity_assay_type",
        "evidence_direct",
        "evidence_mechanism_identity",
        "candidate_context",
        "candidate_mechanism",
    ],
)
def test_v6_wrapper_rejects_assay_and_mechanism_mutations(tmp_path, mutation):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    seeds[0] = _tamper_assay_mechanism_seed(seeds[0], mutation)
    seeds[1] = _tamper_assay_mechanism_seed(seeds[1], mutation)
    with pytest.raises(PaperNativeV6RepurposingContractUnavailable):
        _retrieve(_gateway(tmp_path, seeds))


@pytest.mark.parametrize(
    "mutation",
    ["activity", "document", "pmid", "primary_hash", "delete_row"],
)
def test_v6_wrapper_rejects_self_rehashed_delegate_seed_projection_tamper(
    tmp_path, mutation,
):
    seeds = [_seed(cid, i) for i, cid in enumerate(IDS)]
    provenance = _provenance()
    projection = _paper_native_v6_delegate_seed_source_projection(
        fixed_candidate_ids=IDS,
        seeds=seeds,
    )
    row = projection["rows"][-1]
    if mutation == "activity":
        row["activity_id"] = "5"
    elif mutation == "document":
        row["document_chembl_id"] = "CHEMBL9999"
    elif mutation == "pmid":
        row["pmid"] = "9999"
    elif mutation == "primary_hash":
        row["primary_attestation_sha256"] = "a" * 64
    else:
        projection["rows"].pop()
        projection["row_count"] -= 1
    if mutation != "delete_row":
        row.pop("row_sha256")
        row["row_sha256"] = _canonical_sha256(row)
    projection["rows_sha256"] = _canonical_sha256(projection["rows"])
    projection.pop("attestation_sha256")
    projection["attestation_sha256"] = _canonical_sha256(projection)
    provenance["candidate_discovery_bounds"][
        "delegate_seed_source_binding_projection"
    ] = deepcopy(projection)
    provenance["delegate_seed_source_binding_projection"] = deepcopy(
        projection
    )
    with pytest.raises(
        PaperNativeV6RepurposingContractUnavailable,
        match="DELEGATE_SEED_SOURCE_BINDING_INVALID",
    ):
        _retrieve(_gateway(tmp_path, seeds, provenance))
