from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma3 import (
    MA3AssuranceAgent,
    _PUBMED_EXCLUDED_TYPES as _MA3_PUBMED_EXCLUDED_TYPES,
    _PAPER_NATIVE_V5_REQUIRED_PRIORITY,
    _PAPER_NATIVE_V6_AGGREGATION_POLICY,
    _PAPER_NATIVE_V6_POLICY,
    _PAPER_NATIVE_V6_PROJECTION_STOP_RULE,
    _PAPER_NATIVE_V6_REQUIRED_PRIORITY,
    _paper_v6_clause_is_unambiguous,
    _paper_v6_candidate_projection,
    _paper_v6_delegate_seed_projection_valid,
    _paper_v6_group_projection_valid,
    _paper_v6_hash,
    _paper_v6_reserved_disposition_valid,
    _paper_v6_strict_unit,
    _primary_parenthetical_binding,
)
from drugpilot.domain.models import (
    ActivityMeasurement,
    NamespacedRegistryIdentifier,
    ReservedControlCompound,
    ScientificBackendKind,
    TargetCandidate,
)
from drugpilot.tools.candidate_policy import candidate_identity, screen_reserved_control
from drugpilot.tools.live_ma2_gateway import (
    _PUBMED_EXCLUDED_TYPES as _LIVE_MA2_PUBMED_EXCLUDED_TYPES,
)
from drugpilot.tools.primary_abstract_paper_native_search_v6 import (
    PAPER_NATIVE_V6_EXCLUDED_PUBLICATION_TYPES,
)


def _fixture_v6_mechanism_query() -> dict:
    params = {
        "target_chembl_id": "CHEMBLTARGET1",
        "action_type": "INHIBITOR",
        "max_phase__gte": 1,
        "direct_interaction": 1,
        "molecular_mechanism": 1,
        "order_by": "molecule_chembl_id",
        "limit": 80,
    }
    return _source_call(
        "ChEMBL",
        "https://www.ebi.ac.uk/chembl/api/data/mechanism.json",
        "mechanism",
        params,
    )


def _fixture_v6_mechanism_identity(candidate_id: str) -> dict:
    rows = [{
        "mec_id": int(candidate_id.removeprefix("CHEMBL")),
        "record_id": int(candidate_id.removeprefix("CHEMBL")),
        "molecule_chembl_id": candidate_id,
        "parent_molecule_chembl_id": candidate_id,
        "canonical_parent_molecule_chembl_id": candidate_id,
        "target_chembl_id": "CHEMBLTARGET1",
        "action_type": "INHIBITOR",
        "direct_interaction": 1,
        "molecular_mechanism": 1,
        "max_phase": 4,
        "mechanism_of_action": "FLT3 inhibitor",
        "mechanism_refs": [{"ref_type": "PMID", "ref_id": "1"}],
        "variant_sequence": None,
    }]
    return {
        "schema_version": "chembl_mechanism_identity_v1",
        "parent_molecule_chembl_id": candidate_id,
        "row_count": len(rows),
        "rows_sha256": _paper_v6_hash(rows),
        "rows": rows,
        "query": _fixture_v6_mechanism_query(),
        "fixture": False,
        "missing_values_imputed": False,
    }


def _candidate(candidate_id: str, rank: int, ids: list[str], unit_hashes: list[str] | None = None, *, target="FLT3", direction="inhibit", status="priority", assay_type="single protein format"):
    sentence = f"{candidate_id} {direction}s {target} with an IC50 of {rank}.0 nM."
    pmid = str(1000 + rank)
    document_id = f"CHEMBL{700000 + rank}"
    assay_context = (
        "direct_single_protein_context"
        if assay_type == "single protein format"
        else "cellular_or_system_context"
        if assay_type == "cell-based format"
        else "assay_context_unspecified"
    )
    direct_engagement = assay_context == "direct_single_protein_context"
    mechanism_identity = _fixture_v6_mechanism_identity(candidate_id)
    mechanism_claim = (
        "ChEMBL-curated inhibitor mechanism and measured direct "
        f"IC50 activity against {target}"
        if direct_engagement
        else (
            "ChEMBL-curated inhibitor mechanism with measured cellular "
            f"IC50 activity for {target}; direct single-protein engagement "
            "not established by the selected activity row"
        )
        if assay_context == "cellular_or_system_context"
        else (
            "ChEMBL-curated inhibitor mechanism with measured IC50 activity "
            f"for {target}; assay context unspecified; direct engagement "
            "not established"
        )
    )
    location = {"section": "abstract", "sentence_index": rank}
    primary = {
        "schema_version": "candidate_primary_evidence_v2",
        "verification_status": "verified",
        "retrieval_status": "completed",
        "is_primary_support": True,
        "reason_code": None,
        "document_chembl_id": document_id,
        "pmid": pmid,
        "excerpt": sentence,
        "excerpt_hash": sha256(sentence.encode()).hexdigest(),
        "excerpt_location": location,
        "publication_types": ["Journal Article"],
        "pubmed_type_verification": {
            "schema_version": "pubmed_esummary_publication_type_v1",
            "type_policy": "journal_article_without_excluded_publication_type_v1",
            "verification_status": "verified", "retrieval_status": "completed",
            "pmid": pmid, "reason_code": None,
            "publication_types": ["Journal Article"],
            "required_publication_type": "Journal Article",
            "excluded_publication_types": [],
            "fixture": False,
        },
        "activity_binding": {
            "activity_id": str(rank),
            "document_chembl_id": document_id,
            "molecule_chembl_id": candidate_id,
            "target_chembl_id": "CHEMBLTARGET1",
            "endpoint": "IC50", "relation": "=", "value": float(rank), "unit": "nM",
            "assay_type": assay_type,
            "assay_context": assay_context,
            "direct_single_protein_engagement": direct_engagement,
            "direction_evidence_rule": "measured_ic50_endpoint",
            "mechanism_identity_sha256": _paper_v6_hash(mechanism_identity),
            "mechanism_claim": mechanism_claim,
            "mechanism_claim_sha256": _paper_v6_hash(mechanism_claim),
            "deposited_activity": {
                "schema_version": "chembl_deposited_activity_exact_v1",
                "standard_type": "IC50", "standard_relation": "=",
                "standard_value": float(rank), "standard_units": "nM",
                "typed_exact_match": True, "missing_values_imputed": False,
                "normalization_applied": False, "unit_conversion_applied": False,
            },
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
            "selected_activity_id": str(rank),
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
                "activity_id": str(rank),
                "document_chembl_id": document_id,
                "endpoint": "IC50",
                "pchembl_value": float(rank),
                "verification_status": "verified",
                "reason_code": None,
            }],
            "fixture": False,
            "missing_values_imputed": False,
        },
        "candidate_exact_aliases": [candidate_id],
        "competitor_alias_provenance": {
            "policy": "all_bounded_pool_candidate_aliases_except_current_v1",
            "competitor_exact_aliases": [],
            "truncated": False,
        },
    }
    primary_hash = _paper_v6_hash(primary)
    unit = {
        "candidate_id": candidate_id,
        "candidate_aliases": [candidate_id],
        "direction": direction,
        "document_id_sha256": sha256(pmid.encode()).hexdigest(),
        "fixture": False,
        "is_primary_support": True,
        "is_review": False,
        "predicate": "inhibits",
        "endpoint": "IC50",
        "relation": "=",
        "value": float(rank),
        "unit": "nM",
        "primary_source_attestation_sha256": primary_hash,
        "publication_types": ["Journal Article"],
        "publication_type_screen_passed": True,
        "sentence": sentence,
        "sentence_sha256": sha256(sentence.encode()).hexdigest(),
        "source_location": __import__("json").dumps(location, sort_keys=True, separators=(",", ":")),
        "target_symbol": target,
        "verification_status": "verified",
    }
    unit["support_unit_sha256"] = _paper_v6_hash(unit)
    group_hashes = unit_hashes or []
    group = {
        "candidate_ids": ids,
        "member_support_unit_sha256": group_hashes,
        "minimum_verified_candidates": 3,
        "no_backfill": True,
        "no_reordering": True,
        "policy": _PAPER_NATIVE_V6_POLICY,
        "aggregation_policy": _PAPER_NATIVE_V6_AGGREGATION_POLICY,
    }
    envelope = {
        "schema_version": "candidate_primary_evidence_paper_native_v6",
        "source_contract": "primary_abstract_paper_native_v6",
        "policy": _PAPER_NATIVE_V6_POLICY,
        "aggregation_policy": _PAPER_NATIVE_V6_AGGREGATION_POLICY,
        "verification_status": "verified",
        "molecule_chembl_id": candidate_id,
        "fixture": False,
        "missing_values_imputed": False,
        "candidate_support_unit": unit,
        "candidate_support_unit_sha256": unit["support_unit_sha256"],
        "primary_source_sha256": primary_hash,
        "group_candidate_ids": ids,
        "fixed_pool_candidate_ids": ids,
        "group_sha256": _paper_v6_hash(group),
        "fixed_pool_rank": rank,
        "no_reordering": True,
        "no_backfill": True,
    }
    envelope["attestation_sha256"] = _paper_v6_hash(envelope)
    evidence = SimpleNamespace(evidence_id=f"ev-{rank}", external_id=str(rank), excerpt_hash=primary["excerpt_hash"], metadata={
        "primary_abstract_paper_native_v6": deepcopy(envelope),
        "primary_source": primary,
        "assay_type": assay_type,
        "assay_context": assay_context,
        "direct_single_protein_engagement": direct_engagement,
        "intervention_direction_requested": direction,
        "direction_evidence_rule": "measured_ic50_endpoint",
        "mechanism_identity": deepcopy(mechanism_identity),
    })
    activity = SimpleNamespace(source_evidence_id=evidence.evidence_id, endpoint="IC50", relation="=", value=float(rank), unit="nM", assay_type=assay_type)
    metadata = {
        "chembl_id": candidate_id,
        "registry_ids": [{"namespace": "chembl", "identifier": candidate_id}],
        "aliases": [candidate_id],
        "chembl_exact_aliases": [candidate_id],
        "canonical_smiles": "C" * rank,
        "max_phase": 4,
        "activity_assay_type": assay_type,
        "activity_assay_context": assay_context,
        "direct_single_protein_engagement": direct_engagement,
        "direction_requested": direction,
        "direction_evidence_rules": ["measured_ic50_endpoint"],
        "mechanism_identity": deepcopy(mechanism_identity),
        "primary_abstract_paper_native_v6": envelope,
    }
    target_obj = SimpleNamespace(symbol=target, metadata={"target_chembl_id": "CHEMBLTARGET1"})
    return SimpleNamespace(candidate_id=f"cand-{rank}", name=candidate_id, smiles="C" * rank, mechanism=mechanism_claim, target=target_obj, status=status, metadata=metadata, evidence=[evidence], activities=[activity])


def _fixture_delegate_seed_projection(ids, candidates) -> dict:
    by_id = {candidate.metadata["chembl_id"]: candidate for candidate in candidates}
    rows = []
    returned_ids = []
    for pool_index, candidate_id in enumerate(ids):
        candidate = by_id.get(candidate_id)
        if candidate is None:
            row = {
                "pool_index": pool_index,
                "fixed_pool_rank": pool_index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": False,
                "activity_count": 0,
                "evidence_count": 0,
                "activity_id": None,
                "document_chembl_id": None,
                "pmid": None,
                "primary_verification_status": None,
                "primary_reason_code": None,
                "primary_attestation_sha256": None,
                "fixture": False,
            }
        else:
            returned_ids.append(candidate_id)
            evidence = candidate.evidence[0]
            primary = evidence.metadata["primary_source"]
            row = {
                "pool_index": pool_index,
                "fixed_pool_rank": pool_index + 1,
                "molecule_chembl_id": candidate_id,
                "delegate_returned": True,
                "activity_count": len(candidate.activities),
                "evidence_count": len(candidate.evidence),
                "activity_id": str(evidence.external_id),
                "document_chembl_id": primary["document_chembl_id"],
                "pmid": primary["pmid"],
                "primary_verification_status": primary[
                    "verification_status"
                ],
                "primary_reason_code": primary["reason_code"],
                "primary_attestation_sha256": _paper_v6_hash(primary),
                "fixture": False,
            }
        row["row_sha256"] = _paper_v6_hash(row)
        rows.append(row)
    projection = {
        "schema_version": (
            "paper_native_v6_delegate_seed_source_binding_projection/v1"
        ),
        "fixed_pool_candidate_ids": list(ids),
        "fixed_pool_candidate_ids_sha256": _paper_v6_hash(ids),
        "delegate_return_candidate_ids": returned_ids,
        "delegate_return_candidate_ids_sha256": _paper_v6_hash(returned_ids),
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": _paper_v6_hash(rows),
        "complete": True,
        "fixture": False,
    }
    projection["attestation_sha256"] = _paper_v6_hash(projection)
    return projection


def _fixture_wrapper_screen_projection(ids) -> dict:
    rows = []
    for pool_index, candidate_id in enumerate(ids):
        checks = {
            "cardinality_exact": True,
            "primary_mapping": True,
            "exact_binding": True,
        }
        row = {
            "pool_index": pool_index,
            "fixed_pool_rank": pool_index + 1,
            "molecule_chembl_id": candidate_id,
            "delegate_returned": True,
            "wrapper_screen_status": "verified",
            "wrapper_screen_reason_code": None,
            "wrapper_screen_checks": checks,
            "wrapper_screen_checks_sha256": _paper_v6_hash(checks),
            "fixture": False,
        }
        row["row_sha256"] = _paper_v6_hash(row)
        rows.append(row)
    projection = {
        "schema_version": (
            "paper_native_v6_wrapper_atomic_screen_projection/v1"
        ),
        "fixed_pool_candidate_ids": list(ids),
        "fixed_pool_candidate_ids_sha256": _paper_v6_hash(ids),
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": _paper_v6_hash(rows),
        "complete": True,
        "fixture": False,
    }
    projection["attestation_sha256"] = _paper_v6_hash(projection)
    return projection


def _source_call(
    source: str,
    endpoint: str,
    label: str,
    params: dict,
) -> dict:
    return {
        "source": source,
        "endpoint": endpoint,
        "parameter_keys": sorted(params),
        "parameter_hash": sha256(
            __import__("json").dumps(
                params, ensure_ascii=False, sort_keys=True, default=str
            ).encode()
        ).hexdigest(),
        "timeout_seconds": 30.0,
        "status_code": 200,
        "response_sha256": sha256(f"response:{label}".encode()).hexdigest(),
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "runtime_seconds": 1.0,
        "attempt_count": 1,
        "max_attempts": 3 if source == "ChEMBL" else 1,
        "exhausted": False,
        "cache_hit": False,
        "operation": "retrieve_repurposing",
        "operation_budget_seconds": 600.0,
        "attempts": [{"attempt": 1, "status": "completed"}],
    }


def _source_stage(scope_ids, calls, policy):
    stage = {
        "scope_ids": list(scope_ids),
        "scope_count": len(scope_ids),
        "successful_calls": deepcopy(calls),
        "successful_call_count": len(calls),
        "failures": [],
        "failure_count": 0,
        "recovery_policy": policy,
    }
    stage["stage_sha256"] = _paper_v6_hash(stage)
    return stage


def _molecule_projection(ids: list[str], molecule_call: dict) -> dict:
    rows = []
    for pool_index, molecule_id in enumerate(ids):
        source_fields = {
            "molecule_chembl_id": molecule_id,
            "canonical_parent_molecule_chembl_id": molecule_id,
            "pref_name": molecule_id,
            "max_phase": 4,
            "canonical_smiles": "C" * (pool_index + 1),
            "molecule_synonyms_source_key_present": True,
            "molecule_synonyms_source_type": "list",
            "molecule_synonyms_source_row_count": 0,
            "molecule_synonyms_source_sha256": _paper_v6_hash([]),
            "molecule_synonym_row_limit": 64,
            "molecule_synonym_retained_count": 0,
            "molecule_synonyms_truncated": False,
            "molecule_synonyms": [],
            "exact_aliases": [molecule_id],
            "molecule_query_parameter_sha256": molecule_call["parameter_hash"],
            "molecule_response_sha256": molecule_call["response_sha256"],
            "source_molecule_row_sha256": _paper_v6_hash(
                {
                    "molecule_chembl_id": molecule_id,
                    "pref_name": molecule_id,
                    "max_phase": 4,
                    "molecule_structures": {
                        "canonical_smiles": "C" * (pool_index + 1)
                    },
                    "molecule_synonyms": [],
                }
            ),
            "fixture": False,
        }
        row = {
            "schema_version": "chembl_fixed_pool_molecule_projection_row_v1",
            "pool_index": pool_index,
            **source_fields,
            "projection_complete": True,
            "source_field_projection_sha256": _paper_v6_hash(source_fields),
        }
        row["row_sha256"] = _paper_v6_hash(row)
        rows.append(row)
    projection = {
        "schema_version": "chembl_fixed_pool_molecule_projection_v1",
        "fixed_pool_candidate_ids": list(ids),
        "fixed_pool_candidate_ids_sha256": _paper_v6_hash(ids),
        "molecule_query": deepcopy(molecule_call),
        "molecule_query_sha256": _paper_v6_hash(molecule_call),
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": _paper_v6_hash(rows),
        "complete": True,
        "fixture": False,
    }
    projection["attestation_sha256"] = _paper_v6_hash(projection)
    return projection


def _source_inventory_bounds(ids: list[str]) -> dict:
    endpoints = {
        "mechanism": "https://www.ebi.ac.uk/chembl/api/data/mechanism.json",
        "molecule": "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
        "activity": "https://www.ebi.ac.uk/chembl/api/data/activity.json",
        "assay": "https://www.ebi.ac.uk/chembl/api/data/assay.json",
        "document": "https://www.ebi.ac.uk/chembl/api/data/document.json",
        "warning": "https://www.ebi.ac.uk/chembl/api/data/drug_warning.json",
        "pubmed_type": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
    }
    mechanism_params = {
        "target_chembl_id": "CHEMBLTARGET1",
        "action_type": "INHIBITOR",
        "max_phase__gte": 1,
        "direct_interaction": 1,
        "molecular_mechanism": 1,
        "order_by": "molecule_chembl_id",
        "limit": 80,
    }
    molecule_params = {
        "molecule_chembl_id__in": ",".join(ids),
        "limit": len(ids),
        "order_by": "molecule_chembl_id",
    }
    mechanism_call = _fixture_v6_mechanism_query()
    molecule_call = _source_call(
        "ChEMBL", endpoints["molecule"], "molecule", molecule_params
    )
    activity_params = {
        candidate_id: {
            "target_chembl_id": "CHEMBLTARGET1",
            "parent_molecule_chembl_id": candidate_id,
            "target_organism": "Homo sapiens",
            "assay_type": "B",
            "standard_type": "IC50",
            "pchembl_value__gte": 5.0,
            "order_by": "activity_id",
            "limit": 20,
        }
        for candidate_id in ids
    }
    activity_calls = {
        candidate_id: _source_call(
            "ChEMBL",
            endpoints["activity"],
            f"activity:{candidate_id}",
            activity_params[candidate_id],
        )
        for candidate_id in ids
    }
    warning_call = _source_call(
        "ChEMBL",
        endpoints["warning"],
        "warning",
        {
            "molecule_chembl_id__in": ",".join(ids),
            "limit": max(100, len(ids) * 10),
        },
    )
    policies = {
        "mechanism": "fail_operation_no_recovery_v1",
        "molecule": "fail_operation_no_recovery_v1",
        "activity": "candidate_isolation_without_backfill_v1",
        "assay": "explicit_unknown_context_no_imputation_v1",
        "document": "unverified_primary_no_substitution_v1",
        "warning": "candidate_exclusion_on_unresolved_warning_v1",
        "pubmed_type": "unverified_type_no_substitution_v1",
    }
    stages = {
        "mechanism": _source_stage(
            ["CHEMBLTARGET1"], [mechanism_call], policies["mechanism"]
        ),
        "molecule": _source_stage(ids, [molecule_call], policies["molecule"]),
        "activity": _source_stage(
            ids, list(activity_calls.values()), policies["activity"]
        ),
        "assay": _source_stage([], [], policies["assay"]),
        "document": _source_stage([], [], policies["document"]),
        "warning": _source_stage(ids, [warning_call], policies["warning"]),
        "pubmed_type": _source_stage([], [], policies["pubmed_type"]),
    }
    inventory = {
        "schema_version": "scientific_source_call_outcome_inventory_v1",
        "required_stages": list(stages),
        "stages": stages,
        "successful_call_count": sum(
            stage["successful_call_count"] for stage in stages.values()
        ),
        "failure_count": 0,
        "all_failure_records_represented": True,
        "fixture": False,
    }
    inventory["inventory_sha256"] = _paper_v6_hash(inventory)
    recovery = {
        "schema_version": "scientific_source_recovery_attestation_v1",
        "allowlisted_policies": sorted(set(policies.values())),
        "events": [],
        "event_count": 0,
        "failure_count": 0,
        "no_unlisted_recovery": True,
        "fixture": False,
    }
    recovery["attestation_sha256"] = _paper_v6_hash(recovery)
    return {
        "target_chembl_id": "CHEMBLTARGET1",
        "mechanism_query": deepcopy(mechanism_call),
        "mechanism_provenance_by_id": {
            candidate_id: _fixture_v6_mechanism_identity(candidate_id)
            for candidate_id in ids
        },
        "valid_mechanism_identity_ids": list(ids),
        "prewarning_pool_ids": list(ids),
        "warning_screen_identity_ids": list(ids),
        "warning_screen": {"batch_fallback_used": False},
        "assay_lookup_id_count": 0,
        "document_lookup_id_count": 0,
        "pubmed_type_lookup_pmid_count": 0,
        "molecule_query": molecule_call,
        "fixed_pool_molecule_projection": _molecule_projection(
            ids, molecule_call
        ),
        "activity_counts_by_id": {
            candidate_id: {
                "page_audits": [
                    {
                        "offset": 0,
                        "activity_ids": [str(rank)],
                        "activity_ids_sha256": _paper_v6_hash(
                            [str(rank)]
                        ),
                        "query_parameters": deepcopy(
                            activity_params[candidate_id]
                        ),
                        "query": deepcopy(activity_calls[candidate_id]),
                    }
                ],
                "eligible_directional_count": 1,
                "eligible_directional_activity_ids": [str(rank)],
                "eligible_directional_activity_ids_sha256": _paper_v6_hash(
                    [str(rank)]
                ),
                "retained_document_rows": 1,
                "retained_document_activity_ids": [str(rank)],
                "retained_document_activity_ids_sha256": _paper_v6_hash(
                    [str(rank)]
                ),
                "primary_document_audit_result_cap": None,
                "all_eligible_rows_retained_for_primary_audit": True,
            }
            for rank, candidate_id in enumerate(ids, 1)
        },
        "source_call_inventory": inventory,
        "source_recovery_attestation": recovery,
    }


def _valid_group(*, target="FLT3", direction="inhibit", statuses=None, ids=None, assay_type="single protein format"):
    ids = ids or ["CHEMBL101", "CHEMBL102", "CHEMBL103"]
    statuses = statuses or ["priority"] * len(ids)
    first = [_candidate(value, rank, ids, target=target, direction=direction, status=statuses[rank - 1], assay_type=assay_type) for rank, value in enumerate(ids, 1)]
    hashes = [item.metadata["primary_abstract_paper_native_v6"]["candidate_support_unit_sha256"] for item in first]
    candidates = [_candidate(value, rank, ids, hashes, target=target, direction=direction, status=statuses[rank - 1], assay_type=assay_type) for rank, value in enumerate(ids, 1)]
    for item in candidates:
        screen = screen_reserved_control(candidate_name=item.name, metadata=item.metadata, reserved_controls=[])
        item.metadata.update({"reserved_control_screen": screen.as_metadata(), "control_only": False})
    projection = {
        "schema_version": "paper_native_v6_group_projection_v1",
        "policy": _PAPER_NATIVE_V6_POLICY,
        "declared_group_candidate_ids": ids,
        "declared_group_candidate_ids_sha256": _paper_v6_hash(ids),
        "projected_group_candidate_ids": ids,
        "projected_group_candidate_ids_sha256": _paper_v6_hash(ids),
        "projection_complete": True,
        "projection_scope": "exact_evaluated_fixed_pool_prefix",
        "projected_is_exact_declared_prefix": True,
        "complete_declared_group_projected": True,
        "unevaluated_declared_group_candidate_ids": [],
        "unevaluated_declared_group_candidate_ids_sha256": _paper_v6_hash([]),
        "eligible_quota_reached": sum(status != "excluded" for status in statuses) >= 3,
        "requested_eligible_count": 3,
        "eligible_returned_count": sum(status != "excluded" for status in statuses),
        "evaluated_fixed_pool_prefix_count": len(ids),
        "stop_rule": _PAPER_NATIVE_V6_PROJECTION_STOP_RULE,
        "stop_rule_sha256": sha256(_PAPER_NATIVE_V6_PROJECTION_STOP_RULE.encode()).hexdigest(),
        "stop_condition_satisfied": sum(status != "excluded" for status in statuses) >= 3,
        "no_second_discovery": True,
        "no_backfill_beyond_fixed_pool": True,
        "downstream_evaluation_cap_enforced": True,
        "fixture": False,
    }
    projection["projection_sha256"] = _paper_v6_hash(projection)
    for item in candidates:
        item.metadata["paper_native_v6_group_projection"] = deepcopy(projection)
    units = [deepcopy(item.metadata["primary_abstract_paper_native_v6"]["candidate_support_unit"]) for item in candidates]
    aggregation = {
        "schema_version": "primary_abstract_paper_native_search/v6",
        "policy": _PAPER_NATIVE_V6_POLICY,
        "aggregation_policy": _PAPER_NATIVE_V6_AGGREGATION_POLICY,
        "status": "verified",
        "candidate_support_units": units,
        "candidate_ids": ids,
        "candidate_count": len(ids),
        "candidate_ids_order_sha256": _paper_v6_hash(ids),
        "candidate_ids_set_sha256": _paper_v6_hash(sorted(ids)),
        "group_sha256": candidates[0].metadata["primary_abstract_paper_native_v6"]["group_sha256"],
        "no_reordering": True, "no_backfill": True,
    }
    wrapper_screen_projection = _fixture_wrapper_screen_projection(ids)
    dispositions = []
    for pool_index, candidate in enumerate(candidates):
        primary = candidate.evidence[0].metadata["primary_source"]
        screen_row = wrapper_screen_projection["rows"][pool_index]
        disposition = {
            "pool_index": pool_index,
            "fixed_pool_rank": pool_index + 1,
            "molecule_chembl_id": ids[pool_index],
            "delegate_returned": True,
            "verification_status": "verified",
            "reason_code": None,
            "delegate_reason_codes": [],
            "activity_id": candidate.evidence[0].external_id,
            "document_chembl_id": primary.get("document_chembl_id"),
            "pmid": primary.get("pmid"),
            "source_primary_verification_status": primary.get(
                "verification_status"
            ),
            "source_primary_verification_reason_code": primary.get(
                "reason_code"
            ),
            "source_primary_attestation_sha256": _paper_v6_hash(primary),
            "wrapper_screen_status": screen_row[
                "wrapper_screen_status"
            ],
            "wrapper_screen_reason_code": screen_row[
                "wrapper_screen_reason_code"
            ],
            "wrapper_screen_checks": deepcopy(
                screen_row["wrapper_screen_checks"]
            ),
            "wrapper_screen_checks_sha256": screen_row[
                "wrapper_screen_checks_sha256"
            ],
            "wrapper_screen_row_sha256": screen_row["row_sha256"],
            "support_unit_sha256": units[pool_index]["support_unit_sha256"],
            "fixture": False,
        }
        disposition["disposition_sha256"] = _paper_v6_hash(disposition)
        dispositions.append(disposition)
    search = {
        "schema_version": "primary_abstract_paper_native_gateway_v6",
        "policy": _PAPER_NATIVE_V6_POLICY,
        "aggregation_policy": _PAPER_NATIVE_V6_AGGREGATION_POLICY,
        "status": "verified",
        "fixed_pool_candidate_ids": ids,
        "fixed_pool_candidate_ids_sha256": _paper_v6_hash(ids),
        "verified_candidate_ids": ids,
        "verified_candidate_count": len(ids),
        "candidate_support_aggregation": aggregation,
        "candidate_primary_dispositions": dispositions,
        "candidate_primary_disposition_count": len(dispositions),
        "candidate_primary_dispositions_sha256": _paper_v6_hash(dispositions),
        "rejected_candidate_count": 0,
        "rejected_candidate_projection_sha256": _paper_v6_hash([]),
        "no_reordering": True, "no_backfill": True, "fixture": False,
    }
    search["provenance_sha256"] = _paper_v6_hash(search)
    bounds = _source_inventory_bounds(ids)
    bounds["delegate_gateway_return_candidate_ids"] = list(ids)
    bounds["delegate_seed_source_binding_projection"] = (
        _fixture_delegate_seed_projection(ids, candidates)
    )
    bounds["wrapper_atomic_screen_projection"] = deepcopy(
        wrapper_screen_projection
    )
    bounds["primary_abstract_search"] = search
    for item in candidates:
        item.metadata["candidate_discovery_bounds"] = deepcopy(bounds)
        item.evidence[0].metadata["candidate_discovery_bounds"] = deepcopy(bounds)
    packet_metadata = {
        "candidate_yield": {
            "requested_eligible_count": 3,
            "requested_limit": 3,
            "returned_count": len(ids),
            "discovery_pool_count": len(ids),
            "evaluated_count": len(ids),
            "eligible_count": sum(status != "excluded" for status in statuses),
            "quota_met": sum(status != "excluded" for status in statuses) >= 3,
            "backend_kind": ScientificBackendKind.LIVE.value,
            "partial": sum(status != "excluded" for status in statuses) < 3,
            "fixture_fill_used": False,
            "pool_exhausted": True,
            "paper_native_v6_group_projection": deepcopy(projection),
        },
        "candidate_backfill": {
            "requested_eligible_count": 3,
            "discovery_pool_count": len(ids),
            "evaluated_count": len(ids),
            "eligible_count": sum(status != "excluded" for status in statuses),
            "excluded_count": sum(status == "excluded" for status in statuses),
            "quota_met": sum(status != "excluded" for status in statuses) >= 3,
            "stop_reason": "ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX",
            "pool_exhausted": True,
            "unbounded_search_used": False,
            "discovery_pool_limit": 12,
            "evaluated_identities": [],
            "unevaluated_identities": [],
            "paper_native_v6_group_projection": deepcopy(projection),
        },
    }
    control_rows = []
    for index, item in enumerate(candidates):
        screen = screen_reserved_control(candidate_name=item.name, metadata=item.metadata, reserved_controls=[])
        control_rows.append({
            "pool_index": index,
            "candidate_name": item.name,
            "candidate_identity": candidate_identity(candidate_name=item.name, metadata=item.metadata),
            "candidate_registry_metadata": {"chembl_id": item.metadata["chembl_id"]},
            "reserved_control_screen": screen.as_metadata(),
            "reserved_excluded": False,
            "control_only": False,
            "downstream_evaluated": True,
        })
    control_attestation = {
        "schema_version": "paper_native_v6_full_pool_control_screening/v1",
        "policy": _PAPER_NATIVE_V6_POLICY,
        "row_count": len(control_rows), "rows": control_rows, "fixture": False,
    }
    control_attestation["attestation_sha256"] = _paper_v6_hash(control_attestation)
    packet_metadata["candidate_backfill"]["evaluated_identities"] = [
        {
            **control_rows[index]["candidate_identity"],
            "pool_index": index,
            "candidate_id": item.candidate_id,
            "outcome": "eligible" if item.status != "excluded" else "excluded",
            "reason_codes": list(getattr(item, "exclusion_reasons", [])),
        }
        for index, item in enumerate(candidates)
    ]
    packet_metadata["candidate_yield"]["paper_native_v6_full_pool_control_screening"] = deepcopy(control_attestation)
    packet_metadata["candidate_backfill"]["paper_native_v6_full_pool_control_screening"] = deepcopy(control_attestation)
    return candidates, packet_metadata


def _valid_evaluated_prefix_group():
    """Return a valid three-candidate prefix of a five-member V6 group."""

    ids = [f"CHEMBL10{index}" for index in range(1, 6)]
    all_candidates, metadata = _valid_group(ids=ids)
    candidates = all_candidates[:3]
    projection = candidates[0].metadata["paper_native_v6_group_projection"]
    projection.update({
        "projected_group_candidate_ids": ids[:3],
        "projected_group_candidate_ids_sha256": _paper_v6_hash(ids[:3]),
        "complete_declared_group_projected": False,
        "unevaluated_declared_group_candidate_ids": ids[3:],
        "unevaluated_declared_group_candidate_ids_sha256": _paper_v6_hash(ids[3:]),
        "eligible_returned_count": 3,
        "evaluated_fixed_pool_prefix_count": 3,
    })
    body = dict(projection)
    body.pop("projection_sha256")
    projection["projection_sha256"] = _paper_v6_hash(body)
    for candidate in candidates[1:]:
        candidate.metadata["paper_native_v6_group_projection"] = deepcopy(projection)

    control = metadata["candidate_yield"][
        "paper_native_v6_full_pool_control_screening"
    ]
    unevaluated = [
        {
            **control["rows"][index]["candidate_identity"],
            "pool_index": index,
            "reason_codes": [
                "ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX"
            ],
        }
        for index in (3, 4)
    ]
    for holder in (metadata["candidate_yield"], metadata["candidate_backfill"]):
        holder["paper_native_v6_group_projection"] = deepcopy(projection)
        holder.update({"evaluated_count": 3, "eligible_count": 3, "quota_met": True})
        ledger = holder["paper_native_v6_full_pool_control_screening"]
        for row in ledger["rows"][3:]:
            row["downstream_evaluated"] = False
        ledger_body = dict(ledger)
        ledger_body.pop("attestation_sha256")
        ledger["attestation_sha256"] = _paper_v6_hash(ledger_body)
    metadata["candidate_yield"]["returned_count"] = len(ids)
    metadata["candidate_backfill"].update({
        "excluded_count": 0,
        "evaluated_identities": deepcopy(
            metadata["candidate_backfill"]["evaluated_identities"][:3]
        ),
        "unevaluated_identities": unevaluated,
    })
    metadata["candidate_yield"]["pool_exhausted"] = False
    metadata["candidate_backfill"]["pool_exhausted"] = False
    return candidates, metadata


def test_v6_independent_candidate_and_group_recomputation_accepts_valid_contract():
    candidates, metadata = _valid_group()
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3) is True


def test_v6_group_projection_binds_score_sorted_candidates_by_stable_id():
    candidates, metadata = _valid_group()
    score_sorted_candidates = [candidates[2], candidates[0], candidates[1]]
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in score_sorted_candidates
    ]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(
        score_sorted_candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    ) is True


def test_v6_candidate_projection_uses_one_enveloped_row_among_supplemental_evidence():
    candidates, metadata = _valid_group()
    candidate = candidates[0]
    supplemental = deepcopy(candidate.evidence[0])
    supplemental.evidence_id = "ev_supplemental_context"
    supplemental.metadata.pop("primary_abstract_paper_native_v6", None)
    candidate.evidence.append(supplemental)

    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )

    duplicate = deepcopy(candidate.evidence[0])
    duplicate.evidence_id = "ev_duplicate_v6_envelope"
    candidate.evidence.append(duplicate)
    assert _paper_v6_candidate_projection(
        candidate, expected_direction="inhibit"
    ) is None


def test_v6_candidate_projection_accepts_exact_decimal_source_string():
    candidates, _ = _valid_group()
    candidate = candidates[0]
    deposited = candidate.evidence[0].metadata["primary_source"][
        "activity_binding"
    ]["deposited_activity"]
    deposited["standard_value"] = str(candidate.activities[0].value)
    _rehash_v6_primary_chain(candidate)

    assert _paper_v6_candidate_projection(
        candidate, expected_direction="inhibit"
    ) is not None


@pytest.mark.parametrize(
    "mutation",
    ["selection_eligible_count", "retained_ids", "pagination_ids"],
)
def test_v6_activity_selection_is_bound_to_paginated_retained_ids(mutation):
    candidates, _ = _valid_group()
    candidate = candidates[0]
    assert _paper_v6_candidate_projection(
        candidate, expected_direction="inhibit"
    ) is not None
    if mutation == "selection_eligible_count":
        primary = candidate.evidence[0].metadata["primary_source"]
        primary["activity_document_selection"]["eligible_row_count"] = 2
        _rehash_v6_primary_chain(candidate)
    else:
        bounds = deepcopy(candidate.metadata["candidate_discovery_bounds"])
        count = bounds["activity_counts_by_id"]["CHEMBL101"]
        if mutation == "retained_ids":
            count["retained_document_activity_ids"] = ["999"]
            count["retained_document_activity_ids_sha256"] = (
                _paper_v6_hash(["999"])
            )
        else:
            count["page_audits"][0]["activity_ids"] = ["999"]
            count["page_audits"][0]["activity_ids_sha256"] = (
                _paper_v6_hash(["999"])
            )
        candidate.metadata["candidate_discovery_bounds"] = deepcopy(bounds)
        candidate.evidence[0].metadata["candidate_discovery_bounds"] = (
            deepcopy(bounds)
        )
    assert _paper_v6_candidate_projection(
        candidate, expected_direction="inhibit"
    ) is None


def test_v6_delegate_projection_accepts_metadata_only_suffix_with_reason():
    candidates, _ = _valid_evaluated_prefix_group()
    bounds = deepcopy(candidates[0].metadata["candidate_discovery_bounds"])
    fixed_ids = bounds["prewarning_pool_ids"]
    projection = bounds["delegate_seed_source_binding_projection"]
    row = projection["rows"][3]
    row["primary_verification_status"] = "metadata_only"
    row["primary_reason_code"] = "PRIMARY_ABSTRACT_NOT_AVAILABLE"
    row.pop("row_sha256")
    row["row_sha256"] = _paper_v6_hash(row)
    projection["rows_sha256"] = _paper_v6_hash(projection["rows"])
    projection.pop("attestation_sha256")
    projection["attestation_sha256"] = _paper_v6_hash(projection)
    dispositions = bounds["primary_abstract_search"][
        "candidate_primary_dispositions"
    ]
    screen_projection = bounds["wrapper_atomic_screen_projection"]
    screen_row = screen_projection["rows"][3]
    checks = {
        "cardinality_exact": True,
        "primary_mapping": True,
        "exact_binding": False,
    }
    screen_row.update({
        "wrapper_screen_status": "rejected",
        "wrapper_screen_reason_code": (
            "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
        ),
        "wrapper_screen_checks": checks,
        "wrapper_screen_checks_sha256": _paper_v6_hash(checks),
    })
    screen_row.pop("row_sha256")
    screen_row["row_sha256"] = _paper_v6_hash(screen_row)
    screen_projection["rows_sha256"] = _paper_v6_hash(
        screen_projection["rows"]
    )
    screen_projection.pop("attestation_sha256")
    screen_projection["attestation_sha256"] = _paper_v6_hash(
        screen_projection
    )
    disposition = dispositions[3]
    disposition.update({
        "verification_status": "unverified",
        "reason_code": "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED",
        "support_unit_sha256": None,
        "source_primary_verification_status": "metadata_only",
        "source_primary_verification_reason_code": (
            "PRIMARY_ABSTRACT_NOT_AVAILABLE"
        ),
        "wrapper_screen_status": "rejected",
        "wrapper_screen_reason_code": (
            "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
        ),
        "wrapper_screen_checks": deepcopy(checks),
        "wrapper_screen_checks_sha256": _paper_v6_hash(checks),
        "wrapper_screen_row_sha256": screen_row["row_sha256"],
    })
    disposition.pop("disposition_sha256")
    disposition["disposition_sha256"] = _paper_v6_hash(disposition)
    assert _paper_v6_delegate_seed_projection_valid(
        candidates,
        bounds=bounds,
        fixed_pool_ids=fixed_ids,
        dispositions=dispositions,
    )

    row["primary_reason_code"] = None
    row.pop("row_sha256")
    row["row_sha256"] = _paper_v6_hash(row)
    projection["rows_sha256"] = _paper_v6_hash(projection["rows"])
    projection.pop("attestation_sha256")
    projection["attestation_sha256"] = _paper_v6_hash(projection)
    assert not _paper_v6_delegate_seed_projection_valid(
        candidates,
        bounds=bounds,
        fixed_pool_ids=fixed_ids,
        dispositions=dispositions,
    )


def _valid_verified_source_rejected_wrapper_suffix():
    candidates, _ = _valid_evaluated_prefix_group()
    bounds = deepcopy(candidates[0].metadata["candidate_discovery_bounds"])
    fixed_ids = bounds["prewarning_pool_ids"]
    index = 3
    source_row = bounds["delegate_seed_source_binding_projection"]["rows"][
        index
    ]
    assert source_row["primary_verification_status"] == "verified"
    assert source_row["primary_reason_code"] is None
    screen_projection = bounds["wrapper_atomic_screen_projection"]
    screen_row = screen_projection["rows"][index]
    checks = {
        "cardinality_exact": True,
        "primary_mapping": True,
        "exact_binding": False,
    }
    screen_row.update({
        "wrapper_screen_status": "rejected",
        "wrapper_screen_reason_code": (
            "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
        ),
        "wrapper_screen_checks": checks,
        "wrapper_screen_checks_sha256": _paper_v6_hash(checks),
    })
    screen_row.pop("row_sha256")
    screen_row["row_sha256"] = _paper_v6_hash(screen_row)
    screen_projection["rows_sha256"] = _paper_v6_hash(
        screen_projection["rows"]
    )
    screen_projection.pop("attestation_sha256")
    screen_projection["attestation_sha256"] = _paper_v6_hash(
        screen_projection
    )
    dispositions = bounds["primary_abstract_search"][
        "candidate_primary_dispositions"
    ]
    disposition = dispositions[index]
    disposition.update({
        "verification_status": "unverified",
        "reason_code": "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED",
        "support_unit_sha256": None,
        "wrapper_screen_status": "rejected",
        "wrapper_screen_reason_code": (
            "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
        ),
        "wrapper_screen_checks": deepcopy(checks),
        "wrapper_screen_checks_sha256": _paper_v6_hash(checks),
        "wrapper_screen_row_sha256": screen_row["row_sha256"],
    })
    disposition.pop("disposition_sha256")
    disposition["disposition_sha256"] = _paper_v6_hash(disposition)
    return candidates, bounds, fixed_ids, dispositions, index


def test_v6_wrapper_accepts_verified_source_rejected_exact_binding_suffix():
    candidates, bounds, fixed_ids, dispositions, index = (
        _valid_verified_source_rejected_wrapper_suffix()
    )
    disposition = dispositions[index]
    assert disposition["source_primary_verification_status"] == "verified"
    assert disposition["source_primary_verification_reason_code"] is None
    assert disposition["wrapper_screen_status"] == "rejected"
    assert _paper_v6_delegate_seed_projection_valid(
        candidates,
        bounds=bounds,
        fixed_pool_ids=fixed_ids,
        dispositions=dispositions,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "source_status",
        "source_reason",
        "source_hash",
        "checks",
        "checks_hash",
        "screen_status",
        "row_hash",
        "membership",
    ],
)
def test_v6_wrapper_screen_cross_copy_mutations_fail_closed(mutation):
    candidates, bounds, fixed_ids, dispositions, index = (
        _valid_verified_source_rejected_wrapper_suffix()
    )
    disposition = dispositions[index]
    screen_projection = bounds["wrapper_atomic_screen_projection"]
    screen_row = screen_projection["rows"][index]
    if mutation == "source_status":
        disposition["source_primary_verification_status"] = "unverified"
    elif mutation == "source_reason":
        disposition["source_primary_verification_reason_code"] = "TAMPERED"
    elif mutation == "source_hash":
        disposition["source_primary_attestation_sha256"] = "0" * 64
    elif mutation == "checks":
        changed = deepcopy(disposition["wrapper_screen_checks"])
        changed["exact_binding"] = True
        disposition["wrapper_screen_checks"] = changed
        disposition["wrapper_screen_checks_sha256"] = _paper_v6_hash(changed)
    elif mutation == "checks_hash":
        screen_row["wrapper_screen_checks_sha256"] = "0" * 64
        disposition["wrapper_screen_checks_sha256"] = "0" * 64
    elif mutation == "screen_status":
        screen_row["wrapper_screen_status"] = "verified"
        disposition["wrapper_screen_status"] = "verified"
    elif mutation == "row_hash":
        screen_row["row_sha256"] = "0" * 64
        disposition["wrapper_screen_row_sha256"] = "0" * 64
    else:
        screen_projection["rows"][index], screen_projection["rows"][
            index + 1
        ] = (
            screen_projection["rows"][index + 1],
            screen_projection["rows"][index],
        )
    if mutation not in {"row_hash", "membership"}:
        screen_row.pop("row_sha256")
        screen_row["row_sha256"] = _paper_v6_hash(screen_row)
        disposition["wrapper_screen_row_sha256"] = screen_row["row_sha256"]
    screen_projection["rows_sha256"] = _paper_v6_hash(
        screen_projection["rows"]
    )
    screen_projection.pop("attestation_sha256")
    screen_projection["attestation_sha256"] = _paper_v6_hash(
        screen_projection
    )
    disposition.pop("disposition_sha256")
    disposition["disposition_sha256"] = _paper_v6_hash(disposition)
    assert not _paper_v6_delegate_seed_projection_valid(
        candidates,
        bounds=bounds,
        fixed_pool_ids=fixed_ids,
        dispositions=dispositions,
    )


def _rehash_v6_primary_chain(candidate) -> None:
    primary = candidate.evidence[0].metadata["primary_source"]
    primary_hash = _paper_v6_hash(primary)
    envelope = candidate.metadata["primary_abstract_paper_native_v6"]
    unit = envelope["candidate_support_unit"]
    unit["primary_source_attestation_sha256"] = primary_hash
    unit.pop("support_unit_sha256", None)
    unit["support_unit_sha256"] = _paper_v6_hash(unit)
    envelope["primary_source_sha256"] = primary_hash
    envelope["candidate_support_unit_sha256"] = unit["support_unit_sha256"]
    envelope.pop("attestation_sha256", None)
    envelope["attestation_sha256"] = _paper_v6_hash(envelope)
    candidate.evidence[0].metadata[
        "primary_abstract_paper_native_v6"
    ] = deepcopy(envelope)


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
        "evidence_context",
        "candidate_direct",
        "candidate_mechanism",
    ],
)
def test_v6_assay_and_mechanism_single_field_mutations_fail(mutation):
    candidates, _ = _valid_group()
    candidate = candidates[0]
    assert _paper_v6_candidate_projection(
        candidate, expected_direction="inhibit"
    ) is not None
    binding = candidate.evidence[0].metadata["primary_source"][
        "activity_binding"
    ]
    if mutation == "binding_assay_type":
        binding["assay_type"] = "cell-based format"
        _rehash_v6_primary_chain(candidate)
    elif mutation == "binding_context":
        binding["assay_context"] = "cellular_or_system_context"
        _rehash_v6_primary_chain(candidate)
    elif mutation == "binding_direct":
        binding["direct_single_protein_engagement"] = False
        _rehash_v6_primary_chain(candidate)
    elif mutation == "binding_direction":
        binding["direction_evidence_rule"] = "measured_exact_target_activity"
        _rehash_v6_primary_chain(candidate)
    elif mutation == "binding_mechanism_hash":
        binding["mechanism_identity_sha256"] = "0" * 64
        _rehash_v6_primary_chain(candidate)
    elif mutation == "binding_claim":
        binding["mechanism_claim"] = "rewritten claim"
        binding["mechanism_claim_sha256"] = _paper_v6_hash(
            "rewritten claim"
        )
        _rehash_v6_primary_chain(candidate)
    elif mutation == "activity_assay_type":
        candidate.activities[0].assay_type = "cell-based format"
    elif mutation == "evidence_context":
        candidate.evidence[0].metadata["assay_context"] = (
            "cellular_or_system_context"
        )
    elif mutation == "candidate_direct":
        candidate.metadata["direct_single_protein_engagement"] = False
    else:
        candidate.mechanism = "rewritten claim"
    assert _paper_v6_candidate_projection(
        candidate, expected_direction="inhibit"
    ) is None


def test_v6_unspecified_assay_context_is_not_promoted_as_cellular():
    candidates, metadata = _valid_group(assay_type="unknown assay format")
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    ) is True
    for candidate in candidates:
        assert "cellular" not in candidate.mechanism.casefold()
        assert (
            "assay context unspecified; direct engagement not established"
            in candidate.mechanism
        )


def test_v6_fixed_pool_molecule_projection_binds_promoted_smiles_after_rehash():
    candidates, metadata = _valid_group()
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )

    # Recompute every local molecule-projection digest after changing the
    # source-side promoted value.  The unchanged candidate/evidence copies
    # must still make the independent group audit fail closed.
    template_bounds = deepcopy(
        candidates[0].metadata["candidate_discovery_bounds"]
    )
    projection = template_bounds["fixed_pool_molecule_projection"]
    projection_row = projection["rows"][0]
    projection_row["canonical_smiles"] = "N"
    projection_row["source_molecule_row_sha256"] = _paper_v6_hash(
        {"canonical_smiles": "N", "molecule_chembl_id": "CHEMBL101"}
    )
    source_keys = {
        "molecule_chembl_id",
        "canonical_parent_molecule_chembl_id",
        "pref_name",
        "max_phase",
        "canonical_smiles",
        "molecule_synonyms_source_key_present",
        "molecule_synonyms_source_type",
        "molecule_synonyms_source_row_count",
        "molecule_synonyms_source_sha256",
        "molecule_synonym_row_limit",
        "molecule_synonym_retained_count",
        "molecule_synonyms_truncated",
        "molecule_synonyms",
        "exact_aliases",
        "molecule_query_parameter_sha256",
        "molecule_response_sha256",
        "source_molecule_row_sha256",
        "fixture",
    }
    projection_row["source_field_projection_sha256"] = _paper_v6_hash(
        {key: projection_row[key] for key in source_keys}
    )
    row_body = dict(projection_row)
    row_body.pop("row_sha256")
    projection_row["row_sha256"] = _paper_v6_hash(row_body)
    projection["rows_sha256"] = _paper_v6_hash(projection["rows"])
    projection_body = dict(projection)
    projection_body.pop("attestation_sha256")
    projection["attestation_sha256"] = _paper_v6_hash(projection_body)
    for candidate in candidates:
        candidate.metadata["candidate_discovery_bounds"] = deepcopy(
            template_bounds
        )
        candidate.evidence[0].metadata["candidate_discovery_bounds"] = (
            deepcopy(template_bounds)
        )

    assert not _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )


def test_v6_full_pool_disposition_cannot_omit_suffix_after_search_rehash():
    candidates, metadata = _valid_evaluated_prefix_group()
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )

    template_bounds = deepcopy(
        candidates[0].metadata["candidate_discovery_bounds"]
    )
    search = template_bounds["primary_abstract_search"]
    del search["candidate_primary_dispositions"][3]
    search["candidate_primary_disposition_count"] = len(
        search["candidate_primary_dispositions"]
    )
    search["candidate_primary_dispositions_sha256"] = _paper_v6_hash(
        search["candidate_primary_dispositions"]
    )
    search_body = dict(search)
    search_body.pop("provenance_sha256")
    search["provenance_sha256"] = _paper_v6_hash(search_body)
    for candidate in candidates:
        candidate.metadata["candidate_discovery_bounds"] = deepcopy(
            template_bounds
        )
        candidate.evidence[0].metadata["candidate_discovery_bounds"] = (
            deepcopy(template_bounds)
        )

    assert not _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )


def test_v6_source_inventory_rejects_semantic_tamper_after_all_self_hashes_are_recomputed():
    candidates, metadata = _valid_group()
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )

    bounds = deepcopy(candidates[0].metadata["candidate_discovery_bounds"])
    inventory = bounds["source_call_inventory"]
    molecule_stage = inventory["stages"]["molecule"]
    molecule_stage["scope_ids"] = molecule_stage["scope_ids"][:-1]
    molecule_stage["scope_count"] = len(molecule_stage["scope_ids"])
    stage_body = dict(molecule_stage)
    stage_body.pop("stage_sha256")
    molecule_stage["stage_sha256"] = _paper_v6_hash(stage_body)
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_sha256")
    inventory["inventory_sha256"] = _paper_v6_hash(inventory_body)
    for candidate in candidates:
        candidate.metadata["candidate_discovery_bounds"] = deepcopy(bounds)
        candidate.evidence[0].metadata["candidate_discovery_bounds"] = deepcopy(
            bounds
        )
    tampered_rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in tampered_rows)
    assert not _paper_v6_group_projection_valid(
        candidates,
        tampered_rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )


def test_v6_source_inventory_rejects_activity_call_copy_tamper_after_rehash():
    candidates, metadata = _valid_group()
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert _paper_v6_group_projection_valid(
        candidates,
        rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )

    bounds = deepcopy(candidates[0].metadata["candidate_discovery_bounds"])
    inventory = bounds["source_call_inventory"]
    activity_stage = inventory["stages"]["activity"]
    activity_stage["successful_calls"][0]["response_sha256"] = "f" * 64
    stage_body = dict(activity_stage)
    stage_body.pop("stage_sha256")
    activity_stage["stage_sha256"] = _paper_v6_hash(stage_body)
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_sha256")
    inventory["inventory_sha256"] = _paper_v6_hash(inventory_body)
    for candidate in candidates:
        candidate.metadata["candidate_discovery_bounds"] = deepcopy(bounds)
        candidate.evidence[0].metadata["candidate_discovery_bounds"] = deepcopy(
            bounds
        )
    tampered_rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in tampered_rows)
    assert not _paper_v6_group_projection_valid(
        candidates,
        tampered_rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )


def test_v6_source_inventory_rejects_rehashed_molecule_parameter_role_tamper():
    candidates, metadata = _valid_group()
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3
    )

    bounds = deepcopy(candidates[0].metadata["candidate_discovery_bounds"])
    inventory = bounds["source_call_inventory"]
    stage = inventory["stages"]["molecule"]
    bounds["molecule_query"]["parameter_hash"] = "f" * 64
    stage["successful_calls"][0]["parameter_hash"] = "f" * 64
    stage_body = dict(stage)
    stage_body.pop("stage_sha256")
    stage["stage_sha256"] = _paper_v6_hash(stage_body)
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_sha256")
    inventory["inventory_sha256"] = _paper_v6_hash(inventory_body)
    for candidate in candidates:
        candidate.metadata["candidate_discovery_bounds"] = deepcopy(bounds)
        candidate.evidence[0].metadata["candidate_discovery_bounds"] = deepcopy(
            bounds
        )
    tampered_rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in tampered_rows)
    assert not _paper_v6_group_projection_valid(
        candidates,
        tampered_rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )


def test_v6_group_fails_closed_on_rehashed_source_failure_inventory():
    candidates, metadata = _valid_group()
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3
    )

    bounds = deepcopy(candidates[0].metadata["candidate_discovery_bounds"])
    inventory = bounds["source_call_inventory"]
    stage = inventory["stages"]["assay"]
    stage["scope_ids"] = ["CHEMBL_ASSAY_BAD"]
    stage["scope_count"] = 1
    stage["failures"] = [
        {
            "scope_id": "CHEMBL_ASSAY_BAD",
            "failure": {
                "retrieval_status": "partial",
                "reason_code": "ASSAY_RESPONSE_INVALID",
            },
        }
    ]
    stage["failure_count"] = 1
    stage_body = dict(stage)
    stage_body.pop("stage_sha256")
    stage["stage_sha256"] = _paper_v6_hash(stage_body)
    bounds["assay_lookup_id_count"] = 1
    inventory["failure_count"] = 1
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_sha256")
    inventory["inventory_sha256"] = _paper_v6_hash(inventory_body)
    recovery = bounds["source_recovery_attestation"]
    recovery["events"] = [
        {
            "stage": "assay",
            "scope_id": "CHEMBL_ASSAY_BAD",
            "policy": "explicit_unknown_context_no_imputation_v1",
        }
    ]
    recovery["event_count"] = 1
    recovery["failure_count"] = 1
    recovery_body = dict(recovery)
    recovery_body.pop("attestation_sha256")
    recovery["attestation_sha256"] = _paper_v6_hash(recovery_body)
    for candidate in candidates:
        candidate.metadata["candidate_discovery_bounds"] = deepcopy(bounds)
        candidate.evidence[0].metadata["candidate_discovery_bounds"] = deepcopy(
            bounds
        )
    tampered_rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in tampered_rows)
    assert not _paper_v6_group_projection_valid(
        candidates,
        tampered_rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )


def test_v6_candidate_mutation_is_rejected_even_if_envelope_is_rehashed():
    candidate = _valid_group()[0][0]
    for holder in (candidate.metadata, candidate.evidence[0].metadata):
        envelope = holder["primary_abstract_paper_native_v6"]
        envelope["candidate_support_unit"]["value"] = 99.0
        body = dict(envelope)
        body.pop("attestation_sha256")
        envelope["attestation_sha256"] = _paper_v6_hash(body)
    assert _paper_v6_candidate_projection(candidate, expected_direction="inhibit") is None


def test_v6_group_projection_wrong_order_is_rejected_even_if_rehashed():
    candidates, metadata = _valid_group()
    baseline_rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert _paper_v6_group_projection_valid(
        candidates,
        baseline_rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )
    projection = deepcopy(
        candidates[0].metadata["paper_native_v6_group_projection"]
    )
    projection["projected_group_candidate_ids"] = list(
        reversed(projection["projected_group_candidate_ids"])
    )
    body = dict(projection)
    body.pop("projection_sha256")
    projection["projection_sha256"] = _paper_v6_hash(body)
    for candidate in candidates:
        candidate.metadata["paper_native_v6_group_projection"] = deepcopy(
            projection
        )
    metadata["candidate_yield"]["paper_native_v6_group_projection"] = deepcopy(
        projection
    )
    metadata["candidate_backfill"]["paper_native_v6_group_projection"] = deepcopy(
        projection
    )
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3) is False


def test_v6_policy_token_collision_is_detectable():
    brief = SimpleNamespace(
        data_policy=SimpleNamespace(require_primary_sources=True),
        human_priorities=[_PAPER_NATIVE_V6_REQUIRED_PRIORITY, _PAPER_NATIVE_V5_REQUIRED_PRIORITY],
    )
    packet = SimpleNamespace(
        backend_kind=ScientificBackendKind.LIVE,
        packet_id="packet-v6",
        candidates=[],
    )
    findings = MA3AssuranceAgent._audit_primary_abstract_paper_native_v6(
        object.__new__(MA3AssuranceAgent), brief, packet
    )
    assert [item.code for item in findings] == [
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_TOKEN_INVALID"
    ]


def test_v6_generic_target_is_not_hard_coded():
    candidates, metadata = _valid_group(target="KIT", direction="inhibit")
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert all(rows)
    assert _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3)


def test_v6_unsupported_modulate_direction_fails_closed_instead_of_using_inhibits():
    candidate = _valid_group(target="KIT", direction="modulate")[0][0]
    assert _paper_v6_candidate_projection(candidate, expected_direction="modulate") is None


def test_v6_reserved_excluded_member_uses_recomputed_eligible_count():
    candidates, metadata = _valid_group(statuses=["excluded", "priority", "priority"])
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert not _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3)


def test_v6_four_member_prefix_with_one_reserved_and_three_eligible_passes():
    ids = ["CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104"]
    candidates, metadata = _valid_group(ids=ids, statuses=["excluded", "priority", "priority", "priority"])
    control = ReservedControlCompound(
        name="CHEMBL101",
        registry_ids=[NamespacedRegistryIdentifier(namespace="chembl", identifier="CHEMBL101")],
    )
    candidate = candidates[0]
    screen = screen_reserved_control(candidate_name=candidate.name, metadata=candidate.metadata, reserved_controls=[control])
    candidate.score = 0.0
    candidate.rank = candidate.target_rank = candidate.portfolio_rank = candidate.portfolio_score = None
    candidate.chemistry_execution = candidate.docking_execution = candidate.admet_execution = None
    candidate.exclusion_reasons = list(screen.reason_codes)
    candidate.metadata.update({"reserved_control_screen": screen.as_metadata(), "control_only": True, "policy_eligible": False})
    for holder in (metadata["candidate_yield"], metadata["candidate_backfill"]):
        ledger = holder["paper_native_v6_full_pool_control_screening"]
        ledger["rows"][0].update({
            "reserved_control_screen": screen.as_metadata(),
            "reserved_excluded": True, "control_only": True,
        })
        body = dict(ledger); body.pop("attestation_sha256")
        ledger["attestation_sha256"] = _paper_v6_hash(body)
    metadata["candidate_backfill"]["evaluated_identities"][0][
        "reason_codes"
    ] = list(screen.reason_codes)
    assert _paper_v6_reserved_disposition_valid(candidate, [control]) == (True, True)
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3,
        reserved_controls=[control],
    )


def test_v6_primary_source_mutation_fails_when_bound_hashes_are_stale():
    candidate = _valid_group()[0][0]
    candidate.evidence[0].metadata["primary_source"]["pmid"] = "9999"
    assert _paper_v6_candidate_projection(candidate, expected_direction="inhibit") is None


def test_v6_evidence_only_discovery_bounds_mutation_is_rejected():
    candidate = _valid_group()[0][0]
    candidate.evidence[0].metadata["candidate_discovery_bounds"]["target_chembl_id"] = "CHEMBL_OTHER"
    assert _paper_v6_candidate_projection(candidate, expected_direction="inhibit") is None


def test_v6_strict_suffix_unit_rejects_rehashed_wrong_sentence_value():
    candidate = _valid_group()[0][0]
    unit = deepcopy(candidate.metadata["primary_abstract_paper_native_v6"]["candidate_support_unit"])
    unit["sentence"] = unit["sentence"].replace("1.0 nM", "999 nM")
    unit["sentence_sha256"] = sha256(unit["sentence"].encode()).hexdigest()
    body = dict(unit); body.pop("support_unit_sha256")
    unit["support_unit_sha256"] = _paper_v6_hash(body)
    assert _paper_v6_strict_unit(
        unit, expected_target_symbol="FLT3", expected_direction="inhibit"
    ) is None


def test_v6_strict_suffix_unit_rejects_incompatible_observed_relation():
    candidate = _valid_group()[0][0]
    unit = deepcopy(candidate.metadata["primary_abstract_paper_native_v6"]["candidate_support_unit"])
    unit["sentence"] = unit["sentence"].replace("1.0 nM", ">1.0 nM")
    unit["sentence_sha256"] = sha256(unit["sentence"].encode()).hexdigest()
    body = dict(unit); body.pop("support_unit_sha256")
    unit["support_unit_sha256"] = _paper_v6_hash(body)
    assert _paper_v6_strict_unit(
        unit, expected_target_symbol="FLT3", expected_direction="inhibit"
    ) is None


def test_v6_parenthetical_unit_accepts_atomic_measurement_inside_respectively_sentence():
    candidate = _valid_group()[0][0]
    unit = deepcopy(
        candidate.metadata["primary_abstract_paper_native_v6"][
            "candidate_support_unit"
        ]
    )
    sentence = (
        "Compound RSH-7 had IC 50 values of 47 and 12 nM, respectively, "
        "and sorafenib (FLT3 IC 50 = 33 nM)"
    )
    aliases = ["SORAFENIB"]
    activity = ActivityMeasurement(
        endpoint="IC50",
        relation="=",
        value=33,
        unit="nM",
        assay_type="single protein format",
    )
    binding = _primary_parenthetical_binding(
        sentence,
        candidate_aliases=aliases,
        competitor_aliases=[],
        target=TargetCandidate(symbol="FLT3"),
        activity=activity,
    )
    assert binding is not None
    location = {
        "field": "abstract",
        "sentence_index": 0,
        "clause_index": None,
        "normalized_char_start": 0,
        "normalized_char_end": len(sentence),
        "selection_scope": "atomic_parenthetical_with_sentence_context",
    }
    primary = {
        "excerpt_location": location,
        "competitor_alias_provenance": {
            "policy": "all_bounded_pool_candidate_aliases_except_current_v1",
            "truncated": False,
            "competitor_exact_aliases": [],
        },
        "relevance": binding,
    }
    assert _paper_v6_clause_is_unambiguous(
        sentence,
        aliases=aliases,
        primary=primary,
        target_symbol="FLT3",
        endpoint="IC50",
        relation="=",
        value=33,
        unit="nM",
    )

    unit.update({
        "candidate_aliases": aliases,
        "sentence": sentence,
        "sentence_sha256": sha256(sentence.encode()).hexdigest(),
        "source_location": json.dumps(
            location,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "value": 33,
        "unit": "nM",
        "relation": "=",
    })
    body = dict(unit)
    body.pop("support_unit_sha256")
    unit["support_unit_sha256"] = _paper_v6_hash(body)
    assert _paper_v6_strict_unit(
        unit,
        expected_target_symbol="FLT3",
        expected_direction="inhibit",
    ) is not None


@pytest.mark.parametrize("require_primary,backend", [
    (False, ScientificBackendKind.LIVE),
    (True, ScientificBackendKind.FIXTURE),
])
def test_v6_required_token_rejects_nonlive_or_primary_disabled(require_primary, backend):
    brief = SimpleNamespace(
        data_policy=SimpleNamespace(require_primary_sources=require_primary),
        human_priorities=[_PAPER_NATIVE_V6_REQUIRED_PRIORITY],
    )
    packet = SimpleNamespace(backend_kind=backend, packet_id="packet-v6")
    findings = MA3AssuranceAgent._audit_primary_abstract_paper_native_v6(
        object.__new__(MA3AssuranceAgent), brief, packet
    )
    assert [item.code for item in findings] == [
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_EXECUTION_CONTRACT_INVALID"
    ]


def test_v6_packet_quota_cannot_override_plan_quota():
    candidates, metadata = _valid_group()
    brief = SimpleNamespace(
        data_policy=SimpleNamespace(require_primary_sources=True),
        human_priorities=[_PAPER_NATIVE_V6_REQUIRED_PRIORITY],
        constraints=SimpleNamespace(intervention_direction="inhibit", reserved_control_compounds=[]),
    )
    packet = SimpleNamespace(
        backend_kind=ScientificBackendKind.LIVE, packet_id="packet-v6",
        metadata=metadata, candidates=candidates,
    )
    findings = MA3AssuranceAgent._audit_primary_abstract_paper_native_v6(
        object.__new__(MA3AssuranceAgent), brief, packet,
        plan=SimpleNamespace(candidates_per_route=5),
    )
    assert [item.code for item in findings] == [
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_GROUP_PROJECTION_INVALID"
    ]


@pytest.mark.parametrize("excluded_type", [
    "Meta-Analysis", "Retracted-Publication", "Published Erratum",
    "Expression of Concern", "Retraction Notice", "Retraction of Publication",
    "Protocol", "Clinical Trial Protocol",
])
def test_v6_excluded_publication_type_fails_even_when_all_hashes_are_recomputed(excluded_type):
    candidate = _valid_group()[0][0]
    primary = candidate.evidence[0].metadata["primary_source"]
    primary["publication_types"] = ["Journal Article", excluded_type]
    primary["pubmed_type_verification"]["publication_types"] = list(primary["publication_types"])
    # Model a malicious/stale producer claiming the excluded type is clean;
    # MA3 must still reject from the exact publication type list itself.
    primary["pubmed_type_verification"]["excluded_publication_types"] = []
    unit = candidate.metadata["primary_abstract_paper_native_v6"]["candidate_support_unit"]
    unit["publication_types"] = list(primary["publication_types"])
    primary_hash = _paper_v6_hash(primary)
    unit["primary_source_attestation_sha256"] = primary_hash
    unit_body = dict(unit); unit_body.pop("support_unit_sha256")
    unit["support_unit_sha256"] = _paper_v6_hash(unit_body)
    for holder in (candidate.metadata, candidate.evidence[0].metadata):
        envelope = holder["primary_abstract_paper_native_v6"]
        envelope["candidate_support_unit"] = deepcopy(unit)
        envelope["candidate_support_unit_sha256"] = unit["support_unit_sha256"]
        envelope["primary_source_sha256"] = primary_hash
        body = dict(envelope); body.pop("attestation_sha256")
        envelope["attestation_sha256"] = _paper_v6_hash(body)
    assert _paper_v6_candidate_projection(candidate, expected_direction="inhibit") is None


def test_v6_pubmed_exclusion_policy_matches_live_producer_exactly():
    assert (
        _MA3_PUBMED_EXCLUDED_TYPES
        == _LIVE_MA2_PUBMED_EXCLUDED_TYPES
        == PAPER_NATIVE_V6_EXCLUDED_PUBLICATION_TYPES
    )


def test_v6_rejects_evaluation_beyond_earliest_quota_prefix():
    candidates, metadata = _valid_group(
        ids=["CHEMBL101", "CHEMBL102", "CHEMBL103", "CHEMBL104"]
    )
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in rows)
    assert not _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3
    )


def test_v6_success_still_runs_generic_primary_source_audit(monkeypatch):
    candidates, metadata = _valid_group()
    brief = SimpleNamespace(
        data_policy=SimpleNamespace(require_primary_sources=True),
        human_priorities=[_PAPER_NATIVE_V6_REQUIRED_PRIORITY],
        constraints=SimpleNamespace(
            intervention_direction="inhibit", reserved_control_compounds=[]
        ),
    )
    packet = SimpleNamespace(
        backend_kind=ScientificBackendKind.LIVE,
        packet_id="packet-v6",
        metadata=metadata,
        candidates=candidates,
    )
    sentinel = [object()]
    agent = object.__new__(MA3AssuranceAgent)
    monkeypatch.setattr(
        agent,
        "_audit_candidate_primary_sources",
        lambda _brief, _packet: sentinel,
    )
    assert agent._audit_primary_abstract_paper_native_v6(
        brief,
        packet,
        plan=SimpleNamespace(candidates_per_route=3),
    ) is sentinel


def test_v6_packet_projection_copy_and_bounds_mutations_fail():
    candidates, metadata = _valid_group()
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    metadata["candidate_yield"]["paper_native_v6_group_projection"]["fixture"] = True
    assert not _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3)


@pytest.mark.parametrize("mutation", ["identity_other_candidate", "registry_order_swap", "registry_extra_key"])
def test_v6_full_pool_control_identity_binding_rejects_rehashed_mutations(mutation):
    candidates, metadata = _valid_group()
    for holder in (metadata["candidate_yield"], metadata["candidate_backfill"]):
        ledger = holder["paper_native_v6_full_pool_control_screening"]
        if mutation == "identity_other_candidate":
            ledger["rows"][0]["candidate_identity"] = deepcopy(ledger["rows"][1]["candidate_identity"])
        elif mutation == "registry_order_swap":
            left = deepcopy(ledger["rows"][0]["candidate_registry_metadata"])
            ledger["rows"][0]["candidate_registry_metadata"] = deepcopy(ledger["rows"][1]["candidate_registry_metadata"])
            ledger["rows"][1]["candidate_registry_metadata"] = left
        else:
            declared = ledger["rows"][0]["candidate_registry_metadata"]["chembl_id"]
            ledger["rows"][0]["candidate_registry_metadata"] = {
                "chembl_id": declared, "drug_id": declared,
            }
        body = dict(ledger); body.pop("attestation_sha256")
        ledger["attestation_sha256"] = _paper_v6_hash(body)
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert not _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "evaluated_identity",
        "stop_reason",
        "yield_backend",
        "yield_partial",
        "fixture_fill",
        "yield_pool_exhausted",
        "backfill_pool_exhausted",
        "unbounded_search",
        "discovery_pool_limit",
    ],
)
def test_v6_unhashed_execution_provenance_tampering_is_rejected(mutation):
    candidates, metadata = _valid_group()
    rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in rows)
    assert _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3
    )
    if mutation == "evaluated_identity":
        metadata["candidate_backfill"]["evaluated_identities"][0]["outcome"] = "excluded"
    elif mutation == "stop_reason":
        metadata["candidate_backfill"]["stop_reason"] = "FINITE_DISCOVERY_POOL_EXHAUSTED"
    elif mutation == "yield_backend":
        metadata["candidate_yield"]["backend_kind"] = "fixture"
    elif mutation == "yield_partial":
        metadata["candidate_yield"]["partial"] = True
    elif mutation == "fixture_fill":
        metadata["candidate_yield"]["fixture_fill_used"] = True
    elif mutation == "yield_pool_exhausted":
        metadata["candidate_yield"]["pool_exhausted"] = False
    elif mutation == "backfill_pool_exhausted":
        metadata["candidate_backfill"]["pool_exhausted"] = False
    elif mutation == "unbounded_search":
        metadata["candidate_backfill"]["unbounded_search_used"] = True
    else:
        metadata["candidate_backfill"]["discovery_pool_limit"] = 2
    assert not _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3
    )


@pytest.mark.parametrize("mutation", ["respectively", "competitor", "cross_clause"])
def test_v6_ambiguous_primary_clause_mutations_fail_even_when_rehashed(mutation):
    candidate = _valid_group()[0][0]
    primary = candidate.evidence[0].metadata["primary_source"]
    envelope = candidate.metadata["primary_abstract_paper_native_v6"]
    unit = envelope["candidate_support_unit"]
    if mutation == "respectively":
        sentence = unit["sentence"].replace(" with ", " respectively with ")
    elif mutation == "competitor":
        primary["competitor_alias_provenance"]["competitor_exact_aliases"] = ["DRUGX"]
        sentence = unit["sentence"] + " DRUGX"
    else:
        sentence = unit["sentence"].replace(" with ", "; with ")
    primary["excerpt"] = sentence
    primary["excerpt_hash"] = sha256(sentence.encode()).hexdigest()
    unit["sentence"] = sentence
    unit["sentence_sha256"] = primary["excerpt_hash"]
    primary_hash = _paper_v6_hash(primary)
    unit["primary_source_attestation_sha256"] = primary_hash
    unit_body = dict(unit); unit_body.pop("support_unit_sha256")
    unit["support_unit_sha256"] = _paper_v6_hash(unit_body)
    for holder in (candidate.metadata, candidate.evidence[0].metadata):
        copy = holder["primary_abstract_paper_native_v6"]
        copy["candidate_support_unit"] = deepcopy(unit)
        copy["candidate_support_unit_sha256"] = unit["support_unit_sha256"]
        copy["primary_source_sha256"] = primary_hash
        body = dict(copy); body.pop("attestation_sha256")
        copy["attestation_sha256"] = _paper_v6_hash(body)
    assert _paper_v6_candidate_projection(candidate, expected_direction="inhibit") is None


def test_v6_reserved_control_requires_exact_excluded_no_compute_disposition():
    control = ReservedControlCompound(
        name="CHEMBL101",
        registry_ids=[NamespacedRegistryIdentifier(namespace="chembl", identifier="CHEMBL101")],
    )
    candidate = _valid_group()[0][0]
    valid, reserved = _paper_v6_reserved_disposition_valid(candidate, [control])
    assert reserved is True and valid is False
    screen = screen_reserved_control(candidate_name=candidate.name, metadata=candidate.metadata, reserved_controls=[control])
    candidate.status = "excluded"; candidate.score = 0.0
    candidate.rank = candidate.target_rank = candidate.portfolio_rank = candidate.portfolio_score = None
    candidate.chemistry_execution = candidate.docking_execution = candidate.admet_execution = None
    candidate.exclusion_reasons = list(screen.reason_codes)
    candidate.metadata.update({"reserved_control_screen": screen.as_metadata(), "control_only": True, "policy_eligible": False})
    valid, reserved = _paper_v6_reserved_disposition_valid(candidate, [control])
    assert reserved is True and valid is True


def test_v6_accepts_evaluated_prefix_of_larger_verified_upstream_group():
    full_ids = [f"CHEMBL10{index}" for index in range(1, 6)]
    all_candidates, metadata = _valid_group(ids=full_ids)
    suffix_control = ReservedControlCompound(
        name="CHEMBL105",
        registry_ids=[NamespacedRegistryIdentifier(namespace="chembl", identifier="CHEMBL105")],
    )
    candidates = all_candidates[:3]
    projection = candidates[0].metadata["paper_native_v6_group_projection"]
    projection["projected_group_candidate_ids"] = full_ids[:3]
    projection["projected_group_candidate_ids_sha256"] = _paper_v6_hash(full_ids[:3])
    projection["projection_complete"] = True
    projection["complete_declared_group_projected"] = False
    projection["unevaluated_declared_group_candidate_ids"] = full_ids[3:]
    projection["unevaluated_declared_group_candidate_ids_sha256"] = _paper_v6_hash(full_ids[3:])
    projection["eligible_returned_count"] = 3
    projection["evaluated_fixed_pool_prefix_count"] = 3
    body = dict(projection)
    body.pop("projection_sha256")
    projection["projection_sha256"] = _paper_v6_hash(body)
    for item in candidates[1:]:
        item.metadata["paper_native_v6_group_projection"] = deepcopy(projection)
    for holder in (metadata["candidate_yield"], metadata["candidate_backfill"]):
        holder["paper_native_v6_group_projection"] = deepcopy(projection)
    metadata["candidate_backfill"].update({
        "evaluated_count": 3,
        "eligible_count": 3,
        "excluded_count": 0,
        "quota_met": True,
        "pool_exhausted": False,
        "evaluated_identities": deepcopy(
            metadata["candidate_backfill"]["evaluated_identities"][:3]
        ),
        "unevaluated_identities": [
            {
                **metadata["candidate_yield"]["paper_native_v6_full_pool_control_screening"]["rows"][index]["candidate_identity"],
                "pool_index": index,
                "reason_codes": ["ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX"],
            }
            for index in (3, 4)
        ],
    })
    metadata["candidate_yield"].update({
        "evaluated_count": 3, "eligible_count": 3, "quota_met": True,
        "returned_count": 5, "pool_exhausted": False,
    })
    for holder in (metadata["candidate_yield"], metadata["candidate_backfill"]):
        control = holder["paper_native_v6_full_pool_control_screening"]
        for row in control["rows"][3:]:
            row["downstream_evaluated"] = False
        suffix_row = control["rows"][4]
        suffix_screen = screen_reserved_control(
            candidate_name=suffix_row["candidate_name"],
            metadata=suffix_row["candidate_registry_metadata"],
            reserved_controls=[suffix_control],
        )
        suffix_row["reserved_control_screen"] = suffix_screen.as_metadata()
        suffix_row["reserved_excluded"] = True
        suffix_row["control_only"] = True
        body = dict(control); body.pop("attestation_sha256")
        control["attestation_sha256"] = _paper_v6_hash(body)
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert _paper_v6_group_projection_valid(
        candidates, rows, packet_metadata=metadata, requested_eligible_count=3,
        reserved_controls=[suffix_control],
    )
    candidates, metadata = _valid_group()
    candidates[0].metadata["candidate_discovery_bounds"]["primary_abstract_search"]["candidate_support_aggregation"]["candidate_count"] = 99
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert not _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3)


def test_v6_suffix_unit_invalidity_fails_after_unit_and_group_rehash():
    candidates, metadata = _valid_evaluated_prefix_group()
    baseline_rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in baseline_rows)
    assert _paper_v6_group_projection_valid(
        candidates,
        baseline_rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )

    for candidate in candidates:
        expected_group_hash = None
        for holder in (candidate.metadata, candidate.evidence[0].metadata):
            search = holder["candidate_discovery_bounds"]["primary_abstract_search"]
            aggregation = search["candidate_support_aggregation"]
            suffix = aggregation["candidate_support_units"][4]
            suffix["fixture"] = True
            suffix_body = dict(suffix)
            suffix_body.pop("support_unit_sha256")
            suffix["support_unit_sha256"] = _paper_v6_hash(suffix_body)
            group_body = {
                "candidate_ids": aggregation["candidate_ids"],
                "member_support_unit_sha256": [
                    unit["support_unit_sha256"]
                    for unit in aggregation["candidate_support_units"]
                ],
                "minimum_verified_candidates": 3,
                "no_backfill": True,
                "no_reordering": True,
                "policy": _PAPER_NATIVE_V6_POLICY,
                "aggregation_policy": _PAPER_NATIVE_V6_AGGREGATION_POLICY,
            }
            aggregation["group_sha256"] = _paper_v6_hash(group_body)
            expected_group_hash = aggregation["group_sha256"]
            search_body = dict(search)
            search_body.pop("provenance_sha256")
            search["provenance_sha256"] = _paper_v6_hash(search_body)
        for holder in (candidate.metadata, candidate.evidence[0].metadata):
            envelope = holder["primary_abstract_paper_native_v6"]
            envelope["group_sha256"] = expected_group_hash
            envelope_body = dict(envelope)
            envelope_body.pop("attestation_sha256")
            envelope["attestation_sha256"] = _paper_v6_hash(envelope_body)
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert all(row is not None for row in rows)
    assert not _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3)


def test_v6_unevaluated_identity_must_exactly_match_control_row():
    candidates, metadata = _valid_evaluated_prefix_group()
    baseline_rows = [
        _paper_v6_candidate_projection(item, expected_direction="inhibit")
        for item in candidates
    ]
    assert all(row is not None for row in baseline_rows)
    assert _paper_v6_group_projection_valid(
        candidates,
        baseline_rows,
        packet_metadata=metadata,
        requested_eligible_count=3,
    )
    # A foreign identity plus an extra field must not be accepted merely because
    # pool_index and reason are plausible.
    foreign = deepcopy(metadata["candidate_yield"]["paper_native_v6_full_pool_control_screening"]["rows"][4]["candidate_identity"])
    metadata["candidate_backfill"]["unevaluated_identities"][0] = {
        **foreign, "pool_index": 3,
        "reason_codes": ["ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX"],
        "extra": True,
    }
    rows = [_paper_v6_candidate_projection(item, expected_direction="inhibit") for item in candidates]
    assert all(row is not None for row in rows)
    assert not _paper_v6_group_projection_valid(candidates, rows, packet_metadata=metadata, requested_eligible_count=3)
