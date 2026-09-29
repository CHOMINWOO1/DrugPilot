from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import drugpilot.agents.ma3 as ma3_module
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.domain.models import (
    CandidateCard,
    CandidatePacket,
    CandidateRoute,
    DataSourcePolicy,
    GateName,
    GateVerdictCode,
    NamespacedRegistryIdentifier,
    ProjectConstraints,
    ResearchBrief,
    ReservedControlCompound,
    ScientificBackendKind,
    TargetCandidate,
)

from test_live_scientific_tools import (
    _PAPER_NATIVE_V5_GROUP_FOUR,
    _PaperNativeV5PubMedClient,
    _run_paper_native_v5,
)


REQUIRED_TOKEN = "dynamic_pubmed_primary_abstract_paper_native_v5_required"
ENVELOPE_KEY = "primary_abstract_paper_native_v5"


def _canonical_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _rehash(value: dict[str, object], key: str) -> None:
    body = dict(value)
    body.pop(key, None)
    value[key] = _canonical_hash(body)


def _brief(*, priorities: list[str] | None = None) -> ResearchBrief:
    return ResearchBrief(
        research_goal="Audit a bounded paper-native comparative primary group.",
        disease="Example molecular disease",
        constraints=ProjectConstraints(),
        data_policy=DataSourcePolicy(require_primary_sources=True),
        human_priorities=list(priorities or [REQUIRED_TOKEN]),
    )


def _packet(
    *,
    sentence: str = _PAPER_NATIVE_V5_GROUP_FOUR,
    activity_total_count_by_id: dict[str, int] | None = None,
) -> CandidatePacket:
    seeds, _chembl, gateway = _run_paper_native_v5(
        _PaperNativeV5PubMedClient(
            {"91000001": sentence}
        ),
        activity_total_count_by_id=activity_total_count_by_id,
    )
    try:
        target = TargetCandidate(
            symbol="GENE",
            target_name="Example target",
            uniprot_id="P12345",
            identity_status="resolved",
            identity_sources=["ChEMBL"],
            metadata={
                "chembl_target_id": "CHEMBL_TARGET",
                "chembl_target_type": "SINGLE PROTEIN",
                "chembl_organism": "Homo sapiens",
            },
        )
        candidates: list[CandidateCard] = []
        for index, seed in enumerate(seeds, start=1):
            metadata = deepcopy(seed.metadata)
            verified = metadata[ENVELOPE_KEY]["verification_status"] == "verified"
            if not verified:
                metadata["policy_eligible"] = False
            candidates.append(
                CandidateCard(
                    candidate_id=f"cand_paper_v5_{index}",
                    route=CandidateRoute.REPURPOSING,
                    name=seed.name,
                    target=target,
                    mechanism=seed.mechanism,
                    smiles=seed.smiles,
                    activities=list(seed.activities),
                    evidence=list(seed.evidence),
                    metadata=metadata,
                    score=0.0,
                    status="conditional" if verified else "excluded",
                    exclusion_reasons=(
                        [] if verified else ["NOT_IN_PRIMARY_GROUP"]
                    ),
                )
            )
        return CandidatePacket(
            plan_id="plan_paper_native_v5",
            candidates=candidates,
            backend_kind=ScientificBackendKind.LIVE,
        )
    finally:
        gateway.close()


def _with_group_projection(
    packet: CandidatePacket,
    *,
    declared_group_ids: list[str] | None = None,
    evaluated_count: int = 4,
    requested_eligible_count: int = 3,
) -> CandidatePacket:
    packet = packet.model_copy(deep=True)
    packet.candidates = packet.candidates[:evaluated_count]
    envelope = _envelope(packet.candidates[0])
    fixed_ids = envelope["fixed_pool_candidate_ids"]
    assert isinstance(fixed_ids, list)
    if declared_group_ids is None:
        declared_group_ids = list(envelope["group_member_candidate_ids"])
    evaluated_ids = [
        str(candidate.metadata["chembl_id"])
        for candidate in packet.candidates
    ]
    projected = [
        molecule_id
        for molecule_id in fixed_ids
        if molecule_id in set(declared_group_ids)
        and molecule_id in set(evaluated_ids)
    ]
    eligible_count = sum(
        candidate.status != "excluded" for candidate in packet.candidates
    )
    quota_met = eligible_count >= requested_eligible_count
    projection_complete = set(projected) == set(declared_group_ids)
    stop_satisfied = quota_met and projection_complete
    stop_rule = "eligible_quota_and_declared_group_projection_complete_v1"
    projection: dict[str, object] = {
        "schema_version": "paper_native_v5_group_projection_v1",
        "policy": REQUIRED_TOKEN,
        "declared_group_candidate_ids": list(declared_group_ids),
        "declared_group_candidate_ids_sha256": _canonical_hash(
            declared_group_ids
        ),
        "projected_group_candidate_ids": projected,
        "projected_group_candidate_ids_sha256": _canonical_hash(projected),
        "projection_complete": projection_complete,
        "eligible_quota_reached": quota_met,
        "requested_eligible_count": requested_eligible_count,
        "eligible_returned_count": eligible_count,
        "evaluated_fixed_pool_prefix_count": len(packet.candidates),
        "stop_rule": stop_rule,
        "stop_rule_sha256": sha256(stop_rule.encode("utf-8")).hexdigest(),
        "stop_condition_satisfied": stop_satisfied,
        "no_second_discovery": True,
        "no_backfill_beyond_fixed_pool": True,
        "fixture": False,
    }
    _rehash(projection, "projection_sha256")
    for candidate in packet.candidates:
        candidate.metadata["paper_native_v5_group_projection"] = deepcopy(
            projection
        )
    pool_exhausted = len(packet.candidates) == len(fixed_ids)
    stop_reason = (
        "ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"
        if stop_satisfied
        else "FINITE_DISCOVERY_POOL_EXHAUSTED"
    )
    backfill = {
        "requested_eligible_count": requested_eligible_count,
        "discovery_pool_count": len(fixed_ids),
        "evaluated_count": len(packet.candidates),
        "eligible_count": eligible_count,
        "excluded_count": len(packet.candidates) - eligible_count,
        "quota_met": quota_met,
        "pool_exhausted": pool_exhausted,
        "stop_reason": stop_reason,
        "unbounded_search_used": False,
        "paper_native_v5_group_projection": deepcopy(projection),
    }
    candidate_yield = {
        "requested_eligible_count": requested_eligible_count,
        "discovery_pool_count": len(fixed_ids),
        "evaluated_count": len(packet.candidates),
        "eligible_count": eligible_count,
        "quota_met": quota_met,
        "pool_exhausted": pool_exhausted,
        "paper_native_v5_group_projection": deepcopy(projection),
    }
    packet.metadata["candidate_backfill"] = backfill
    packet.metadata["candidate_yield"] = candidate_yield
    return packet


def _audit(
    packet: CandidatePacket,
    *,
    brief: ResearchBrief | None = None,
):
    return MA3AssuranceAgent()._audit_candidate_primary_sources_for_policy(
        brief or _brief(), packet
    )


def _final_packet() -> CandidatePacket:
    """Build the exact-v5 CandidatePacket after MA2's finite projection."""

    return _with_group_projection(_packet())


def _envelope(candidate: CandidateCard) -> dict[str, object]:
    value = candidate.evidence[0].metadata[ENVELOPE_KEY]
    assert isinstance(value, dict)
    return value


def _paper_v5_group_sentence(head: str) -> str:
    return (
        f"{head}(half-maximal inhibitory concentration [IC50] at "
        "physiological [1mM] ATP, 501nM), followed by Betanib "
        "(702nM), Gammanib (903nM), and Controlib (1104nM), and "
        "the most selective."
    )


def _verify_envelope_group_pairs(
    packet: CandidatePacket,
    group: dict[str, object],
) -> list[dict[str, object]] | None:
    envelope = _envelope(packet.candidates[0])
    summary_types = group["summary_publication_types"]
    fetch_types = group["fetch_publication_types"]
    candidate_ids = group["candidate_ids"]
    raw_inventory = envelope["alias_inventory"]
    assert isinstance(summary_types, list)
    assert isinstance(fetch_types, list)
    assert isinstance(candidate_ids, list)
    assert isinstance(raw_inventory, dict)
    inventory = _verify_alias_inventory(packet, raw_inventory)
    assert inventory is not None
    return ma3_module._paper_v5_verify_pairs(
        group["pairs"],
        sentence=str(group["sentence"]),
        section=str(group["section"]),
        publication_types=sorted(set(summary_types).union(fetch_types)),
        inventory=inventory,
        candidate_ids=candidate_ids,
    )


def _tamper_group_sentence_same_length(
    group: dict[str, object],
    *,
    accepted: str,
    replacement: str,
) -> None:
    assert len(accepted) == len(replacement)
    sentence = group["sentence"]
    pairs = group["pairs"]
    assert isinstance(sentence, str)
    assert isinstance(pairs, list)
    assert sentence.count(accepted) == 1
    sentence = sentence.replace(accepted, replacement, 1)
    sentence_sha256 = sha256(sentence.encode("utf-8")).hexdigest()
    group["sentence"] = sentence
    group["sentence_sha256"] = sentence_sha256
    for pair in pairs:
        assert isinstance(pair, dict)
        sentence_span = pair["sentence_span"]
        assert isinstance(sentence_span, dict)
        sentence_span["text"] = sentence
        pair["sentence_sha256"] = sentence_sha256
        evidence_sha256 = ma3_module._paper_v5_pair_evidence_hash(
            pair,
            sentence_sha256=sentence_sha256,
        )
        assert isinstance(evidence_sha256, str)
        pair["evidence_sha256"] = evidence_sha256


def _sync_candidate_copy(candidate: CandidateCard) -> None:
    candidate.metadata[ENVELOPE_KEY] = deepcopy(_envelope(candidate))


def _verify_alias_inventory(
    packet: CandidatePacket,
    raw: dict[str, object],
    *,
    target: object | None = None,
) -> dict[str, object] | None:
    candidate = packet.candidates[0]
    envelope = _envelope(candidate)
    fixed_ids = envelope["fixed_pool_candidate_ids"]
    bounds = candidate.evidence[0].metadata["candidate_discovery_bounds"]
    direct_ids = bounds["paper_native_v5_direct_support_candidate_ids"]
    assert isinstance(fixed_ids, list)
    assert isinstance(direct_ids, list)
    return ma3_module._paper_v5_alias_inventory(
        raw,
        fixed_ids=fixed_ids,
        direct_ids=direct_ids,
        target=target or candidate.target,
    )


def _verify_support_inventory(
    packet: CandidatePacket,
    raw: dict[str, object],
) -> dict[str, object] | None:
    candidate = packet.candidates[0]
    bounds = candidate.evidence[0].metadata["candidate_discovery_bounds"]
    discovery = ma3_module._paper_v5_discovery_bounds(bounds)
    assert discovery is not None
    fixed_ids = discovery["fixed_ids"]
    direct_ids = discovery["direct_ids"]
    assert isinstance(fixed_ids, list)
    assert isinstance(direct_ids, list)
    selected = raw["selected_support_snapshots"]
    assert isinstance(selected, list)
    return ma3_module._paper_v5_support_inventory(
        raw,
        fixed_ids=fixed_ids,
        direct_ids=direct_ids,
        snapshots=selected,
        target_chembl_id=str(discovery["target_chembl_id"]),
        target_uniprot_id=str(candidate.target.uniprot_id),
        resolution_by_id=discovery["resolution_by_id"],
        mechanisms_by_id=discovery["mechanisms_by_id"],
    )


def _mutate_document_field(
    value: dict[str, object],
    *,
    mutation: str,
) -> None:
    if mutation == "missing":
        value.pop("document_chembl_id", None)
    elif mutation == "null":
        value["document_chembl_id"] = None
    elif mutation == "empty":
        value["document_chembl_id"] = ""
    elif mutation == "whitespace":
        value["document_chembl_id"] = "   "
    elif mutation == "padded":
        value["document_chembl_id"] = " CHEMBL12345 "
    elif mutation == "non_string":
        value["document_chembl_id"] = 12345
    elif mutation == "malformed":
        value["document_chembl_id"] = "DOC_CHEMBL12345"
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(mutation)


def _mutate_chembl_retry_query(
    query: dict[str, object],
    *,
    mutation: str,
    value: object,
) -> None:
    if mutation in {"attempt_count", "max_attempts"}:
        query[mutation] = value
        if mutation == "attempt_count":
            attempts = query["attempts"]
            assert isinstance(attempts, list) and len(attempts) == 1
            attempt = attempts[0]
            assert isinstance(attempt, dict)
            attempt["transport_attempt_count"] = value
        return
    if mutation == "attempts_missing":
        query.pop("attempts", None)
        return
    attempts = query["attempts"]
    assert isinstance(attempts, list) and len(attempts) == 1
    attempt = attempts[0]
    assert isinstance(attempt, dict)
    if mutation == "attempts_not_list":
        query["attempts"] = value
    elif mutation == "attempts_extra":
        attempts.append(deepcopy(attempt))
    elif mutation == "attempt_status":
        attempt["status"] = value
    elif mutation == "attempt_transport_count":
        attempt["transport_attempt_count"] = value
    elif mutation == "attempt_order":
        attempt["attempt"] = value
    elif mutation == "attempt_extra_field":
        attempt["unexpected"] = value
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(mutation)


def _verify_mechanism_retry_contract(
    packet: CandidatePacket,
    *,
    field: str,
    value: object,
) -> dict[str, object] | None:
    bounds = deepcopy(
        packet.candidates[0].evidence[0].metadata[
            "candidate_discovery_bounds"
        ]
    )
    assert isinstance(bounds, dict)
    query = bounds["mechanism_query"]
    provenance = bounds["mechanism_provenance_by_id"]
    assert isinstance(query, dict)
    assert isinstance(provenance, dict)
    _mutate_chembl_retry_query(query, mutation=field, value=value)
    for mechanism in provenance.values():
        assert isinstance(mechanism, dict)
        mechanism["query"] = deepcopy(query)
    return ma3_module._paper_v5_discovery_bounds(bounds)


def _verify_candidate_support_retry_contract(
    packet: CandidatePacket,
    *,
    field: str,
    value: object,
) -> dict[str, object] | None:
    packet = packet.model_copy(deep=True)
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]
    envelope = _envelope(candidate)
    raw = deepcopy(envelope["pre_pubmed_chembl_support"])
    fixed_ids = envelope["fixed_pool_candidate_ids"]
    assert isinstance(raw, dict)
    assert isinstance(fixed_ids, list)
    query = raw["chembl_activity_query"]
    assert isinstance(query, dict)
    _mutate_chembl_retry_query(query, mutation=field, value=value)
    raw["chembl_activity_query_sha256"] = _canonical_hash(query)
    assert isinstance(evidence.metadata, dict)
    evidence.metadata["activity_query"] = deepcopy(query)
    _rehash(raw, "support_attestation_sha256")
    return ma3_module._paper_v5_support_snapshot(
        raw,
        candidate=candidate,
        evidence=evidence,
        activity=candidate.activities[0],
        fixed_ids=fixed_ids,
    )


def _mutate_support_inventory_retry_contract(
    raw: dict[str, object],
    *,
    field: str,
    value: object,
) -> None:
    records = raw["records"]
    selected = raw["selected_support_snapshots"]
    assert isinstance(records, list)
    assert isinstance(selected, list)
    selected_indices = {
        str(snapshot["molecule_chembl_id"]): index
        for index, snapshot in enumerate(selected)
        if isinstance(snapshot, dict)
    }
    assert len(selected_indices) == len(selected)
    for record in records:
        assert isinstance(record, dict)
        row_audits = record["row_audits"]
        assert isinstance(row_audits, list)
        for audit in row_audits:
            assert isinstance(audit, dict)
            query = audit["chembl_activity_query"]
            assert isinstance(query, dict)
            _mutate_chembl_retry_query(
                query,
                mutation=field,
                value=value,
            )
            audit["chembl_activity_query_sha256"] = _canonical_hash(query)
            _rehash(audit, "row_audit_sha256")
        snapshot = record["selected_support"]
        if isinstance(snapshot, dict):
            query = snapshot["chembl_activity_query"]
            assert isinstance(query, dict)
            _mutate_chembl_retry_query(
                query,
                mutation=field,
                value=value,
            )
            snapshot["chembl_activity_query_sha256"] = _canonical_hash(query)
            _rehash(snapshot, "support_attestation_sha256")
            molecule_id = str(record["molecule_chembl_id"])
            assert molecule_id in selected_indices
            selected[selected_indices[molecule_id]] = deepcopy(snapshot)
        _rehash(record, "record_sha256")
    raw["selected_support_snapshots_sha256"] = _canonical_hash(selected)
    _rehash(raw, "inventory_sha256")


def _assert_critical(packet: CandidatePacket, code: str) -> None:
    findings = _audit(packet)
    assert [(item.code, item.severity.value) for item in findings] == [
        (code, "CRITICAL")
    ]


def test_v5_chembl_first_attempt_success_uses_frozen_two_attempt_contract(
) -> None:
    packet = _packet()
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    mechanism_query = bounds["mechanism_query"]
    inventory = bounds["paper_native_v5_direct_support_inventory"]
    search = bounds["primary_abstract_search"]
    assert isinstance(mechanism_query, dict)
    assert isinstance(inventory, dict)
    assert isinstance(search, dict)
    chembl_queries = [mechanism_query]
    records = inventory["records"]
    selected = inventory["selected_support_snapshots"]
    assert isinstance(records, list)
    assert isinstance(selected, list)
    for record in records:
        assert isinstance(record, dict)
        row_audits = record["row_audits"]
        assert isinstance(row_audits, list)
        for audit in row_audits:
            assert isinstance(audit, dict)
            query = audit["chembl_activity_query"]
            assert isinstance(query, dict)
            chembl_queries.append(query)
    for snapshot in selected:
        assert isinstance(snapshot, dict)
        query = snapshot["chembl_activity_query"]
        assert isinstance(query, dict)
        chembl_queries.append(query)
    assert {
        (query.get("attempt_count"), query.get("max_attempts"))
        for query in chembl_queries
    } == {(1, 2)}
    for query in chembl_queries:
        attempts = query["attempts"]
        assert isinstance(attempts, list) and len(attempts) == 1
        attempt = dict(attempts[0])
        runtime_seconds = attempt.pop("runtime_seconds")
        assert type(runtime_seconds) in (int, float)
        assert Decimal(str(runtime_seconds)).is_finite()
        assert runtime_seconds >= 0
        assert attempt == {
            "attempt": 1,
            "transport_attempt_count": 1,
            "timeout_seconds": query["timeout_seconds"],
            "status": "completed",
            "status_code": 200,
            "retrying": False,
        }
    assert search["transport"] == {
        "dedicated_from_chembl_client": True,
        "max_attempts": 1,
        "required_max_attempts": 1,
        "logical_call_limit": 3,
    }
    assert _audit(_with_group_projection(packet)) == []


def test_v5_live_candidate_target_uses_deposited_discovery_chembl_binding(
) -> None:
    packet = _packet()
    for candidate in packet.candidates:
        # MA1 owns the canonical target object and need not copy MA2's ChEMBL
        # enrichment into it.  The v5 contract binds ChEMBL identity through
        # the independently verified discovery/support attestations instead.
        candidate.target.metadata.pop("chembl_target_id", None)
        candidate.target.metadata.pop("chembl_target_type", None)
        candidate.target.metadata.pop("chembl_organism", None)

    assert _audit(_with_group_projection(packet)) == []


def test_v5_complete_activity_pagination_accepts_more_than_twenty_rows() -> None:
    packet = _packet(activity_total_count_by_id={"CHEMBL101": 42})
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    verified = ma3_module._paper_v5_discovery_bounds(bounds)

    assert verified is not None
    count = bounds["activity_counts_by_id"]["CHEMBL101"]
    assert count["total_count"] == 42
    assert count["inspected_count"] == 42
    assert count["truncated"] is False
    assert [page["response_count"] for page in count["page_audits"]] == [
        20,
        20,
        2,
    ]
    assert _audit(_with_group_projection(packet)) == []


def test_v5_complete_activity_pagination_rejects_missing_final_page() -> None:
    packet = _packet(activity_total_count_by_id={"CHEMBL101": 42})
    bounds = deepcopy(
        packet.candidates[0].evidence[0].metadata[
            "candidate_discovery_bounds"
        ]
    )
    count = bounds["activity_counts_by_id"]["CHEMBL101"]
    count["page_audits"].pop()
    count["page_count"] = len(count["page_audits"])
    count["page_audits_sha256"] = _canonical_hash(count["page_audits"])
    bounds["activity_page_audits_by_id"]["CHEMBL101"] = deepcopy(
        count["page_audits"]
    )
    bounds["activity_page_audits_by_id_sha256"] = _canonical_hash(
        bounds["activity_page_audits_by_id"]
    )
    bounds["activity_logical_call_count"] -= 1

    assert ma3_module._paper_v5_discovery_bounds(bounds) is None


@pytest.mark.parametrize("attempt_count", [1, 2], ids=["first", "retry"])
def test_v5_chembl_retry_contract_accepts_success_on_attempt_one_or_two(
    attempt_count: int,
) -> None:
    packet = _packet()
    assert _verify_mechanism_retry_contract(
        packet,
        field="attempt_count",
        value=attempt_count,
    ) is not None
    assert _verify_candidate_support_retry_contract(
        packet,
        field="attempt_count",
        value=attempt_count,
    ) is not None
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    _mutate_support_inventory_retry_contract(
        raw,
        field="attempt_count",
        value=attempt_count,
    )
    assert _verify_support_inventory(packet, raw) is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts", 1),
        ("max_attempts", 3),
        ("max_attempts", True),
        ("max_attempts", "2"),
        ("max_attempts", 2.0),
        ("attempt_count", 0),
        ("attempt_count", 3),
        ("attempt_count", True),
        ("attempt_count", "1"),
        ("attempt_count", 1.0),
    ],
    ids=[
        "max_one",
        "max_three",
        "max_bool",
        "max_string",
        "max_float",
        "attempt_zero",
        "attempt_three",
        "attempt_bool",
        "attempt_string",
        "attempt_float",
    ],
)
def test_v5_chembl_retry_contract_rejects_self_consistent_invalid_values(
    field: str,
    value: object,
) -> None:
    packet = _packet()
    assert _verify_mechanism_retry_contract(
        packet,
        field=field,
        value=value,
    ) is None
    assert _verify_candidate_support_retry_contract(
        packet,
        field=field,
        value=value,
    ) is None
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    _mutate_support_inventory_retry_contract(raw, field=field, value=value)
    assert _verify_support_inventory(packet, raw) is None


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("attempts_missing", None),
        ("attempts_not_list", {}),
        ("attempts_extra", None),
        ("attempt_status", "error"),
        ("attempt_transport_count", 2),
        ("attempt_transport_count", True),
        ("attempt_transport_count", "1"),
        ("attempt_order", 2),
        ("attempt_extra_field", "unexpected"),
    ],
    ids=[
        "missing",
        "not_list",
        "extra",
        "error_status",
        "count_mismatch",
        "count_bool",
        "count_string",
        "order",
        "extra_field",
    ],
)
def test_v5_chembl_retry_attempt_ledger_rejects_self_hashed_tamper(
    mutation: str,
    value: object,
) -> None:
    packet = _packet()
    assert _verify_mechanism_retry_contract(
        packet,
        field=mutation,
        value=value,
    ) is None
    assert _verify_candidate_support_retry_contract(
        packet,
        field=mutation,
        value=value,
    ) is None
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    _mutate_support_inventory_retry_contract(
        raw,
        field=mutation,
        value=value,
    )
    assert _verify_support_inventory(packet, raw) is None


def test_indivisible_four_member_group_and_excluded_nonmember_pass() -> None:
    gateway_packet = _packet()
    packet = _with_group_projection(gateway_packet)

    assert _audit(packet) == []
    assert [item.status for item in gateway_packet.candidates] == [
        "conditional",
        "conditional",
        "conditional",
        "conditional",
        "excluded",
    ]


def test_final_packet_requires_ma2_projection_but_gateway_envelope_does_not() -> None:
    packet = _packet()
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]

    assert ma3_module._paper_v5_verify_candidate_envelope(
        candidate=candidate,
        evidence=evidence,
        activity=candidate.activities[0],
        envelope=_envelope(candidate),
    ) is not None
    findings = _audit(packet)
    assert [(item.code, item.severity.value) for item in findings] == [
        ("CANDIDATE_PRIMARY_PAPER_NATIVE_V5_GROUP_PROJECTION_INVALID", "CRITICAL")
    ]


def test_ma2_group4_quota3_projection_is_independently_verified() -> None:
    packet = _with_group_projection(_packet())

    assert _audit(packet) == []


def test_ma2_projection_continues_through_reserved_fourth_group_member() -> None:
    packet = _packet().model_copy(deep=True)
    control = packet.candidates[3]
    control.status = "excluded"
    control.metadata["policy_eligible"] = False
    control.exclusion_reasons = ["RESERVED_ASSAY_CONTROL"]
    control.safety_flags = ["reserved_assay_control"]
    brief = _brief()
    brief.constraints.reserved_control_compounds = [
        ReservedControlCompound(
            name=control.name,
            registry_ids=[
                NamespacedRegistryIdentifier(
                    namespace="ChEMBL",
                    identifier=str(control.metadata["chembl_id"]),
                )
            ],
        )
    ]
    packet = _with_group_projection(packet)

    assert _audit(packet, brief=brief) == []


def test_projection_preserves_sentence_order_but_projects_fixed_pool_order() -> None:
    packet = _packet()
    fixed_ids = _envelope(packet.candidates[0])["fixed_pool_candidate_ids"]
    assert isinstance(fixed_ids, list)
    declared = [fixed_ids[4], fixed_ids[0], fixed_ids[2]]
    packet = _with_group_projection(
        packet,
        declared_group_ids=declared,
        evaluated_count=5,
    )

    assert ma3_module._paper_v5_group_projection(
        packet,
        declared_group_ids=declared,
        fixed_ids=fixed_ids,
    )


def test_projection_rejects_evaluation_beyond_first_satisfied_prefix() -> None:
    packet = _with_group_projection(_packet(), evaluated_count=5)
    fixed_ids = _envelope(packet.candidates[0])["fixed_pool_candidate_ids"]
    group_ids = _envelope(packet.candidates[0])["group_member_candidate_ids"]
    assert isinstance(fixed_ids, list)
    assert isinstance(group_ids, list)

    assert not ma3_module._paper_v5_group_projection(
        packet,
        declared_group_ids=group_ids,
        fixed_ids=fixed_ids,
    )


def test_projection_copy_tamper_is_critical() -> None:
    packet = _with_group_projection(_packet())
    projection = packet.candidates[0].metadata[
        "paper_native_v5_group_projection"
    ]
    assert isinstance(projection, dict)
    projection["no_second_discovery"] = False
    _rehash(projection, "projection_sha256")

    _assert_critical(
        packet,
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_GROUP_PROJECTION_INVALID",
    )


def test_exact_token_is_unique_and_prose_does_not_dispatch_v5() -> None:
    packet = _final_packet()
    duplicate = _audit(
        packet,
        brief=_brief(priorities=[REQUIRED_TOKEN, REQUIRED_TOKEN]),
    )
    prose = _audit(
        packet,
        brief=_brief(priorities=[f"Please use {REQUIRED_TOKEN}."]),
    )

    assert [(item.code, item.severity.value) for item in duplicate] == [
        ("CANDIDATE_PRIMARY_PAPER_NATIVE_V5_TOKEN_INVALID", "HIGH")
    ]
    assert {item.code for item in prose} == {
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    }


def test_missing_verified_evidence_is_deterministic_high() -> None:
    packet = _final_packet()
    candidate = packet.candidates[0]
    candidate.metadata.pop(ENVELOPE_KEY)
    candidate.evidence[0].metadata.pop(ENVELOPE_KEY)

    findings = _audit(packet)

    assert [(item.code, item.severity.value) for item in findings] == [
        ("CANDIDATE_PRIMARY_PAPER_NATIVE_V5_INSUFFICIENT", "HIGH")
    ]
    verdict = MA3AssuranceAgent()._decide(GateName.G1, findings, [])
    assert verdict.verdict is GateVerdictCode.REVISE


@pytest.mark.parametrize(
    "mutation",
    [
        "bracket_span",
        "relation",
        "number",
        "pair_order",
        "member_deletion",
        "support_selected_after_pubmed",
        "query_sort",
        "call_count",
        "alias_collision",
        "group_hash",
        "group_copy",
    ],
)
def test_verified_envelope_tamper_fails_closed(
    mutation: str,
) -> None:
    packet = _final_packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    envelope = _envelope(candidate)
    group = envelope["group"]
    outcome = envelope["search_outcome"]
    assert isinstance(group, dict)
    assert isinstance(outcome, dict)
    pairs = group["pairs"]
    assert isinstance(pairs, list)

    if mutation == "bracket_span":
        wrapper = pairs[0]["endpoint_wrapper_span"]
        assert isinstance(wrapper, dict)
        wrapper["start"] = int(wrapper["start"]) + 1
    elif mutation == "relation":
        pairs[0]["relation_basis"] = "literal_equality"
    elif mutation == "number":
        pairs[0]["value_raw"] = "999"
    elif mutation == "pair_order":
        pairs[0], pairs[1] = pairs[1], pairs[0]
    elif mutation == "member_deletion":
        members = envelope["group_member_candidate_ids"]
        assert isinstance(members, list)
        members.pop()
    elif mutation == "support_selected_after_pubmed":
        support = envelope["pre_pubmed_chembl_support"]
        assert isinstance(support, dict)
        support["selected_before_pubmed_search"] = False
        _rehash(support, "support_attestation_sha256")
    elif mutation == "query_sort":
        outcome["search_sort"] = "relevance"
        _rehash(outcome, "outcome_sha256")
    elif mutation == "call_count":
        calls = outcome["calls"]
        assert isinstance(calls, list)
        calls.pop()
        _rehash(outcome, "outcome_sha256")
    elif mutation == "alias_collision":
        inventory = envelope["alias_inventory"]
        assert isinstance(inventory, dict)
        records = inventory["records"]
        assert isinstance(records, list)
        records[1]["accepted_aliases"][0] = records[0]["accepted_aliases"][0]
        _rehash(inventory, "inventory_sha256")
    elif mutation == "group_hash":
        group["group_sha256"] = "0" * 64
    elif mutation == "group_copy":
        group["sentence_index"] = int(group["sentence_index"]) + 1
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _rehash(envelope, "attestation_sha256")
    _sync_candidate_copy(candidate)

    _assert_critical(
        packet,
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_ATTESTATION_INVALID",
    )


def test_primary_study_method_sentence_is_independently_recomputed() -> None:
    packet = _final_packet()
    group = _envelope(packet.candidates[0])["group"]
    assert isinstance(group, dict)
    primary = deepcopy(group["primary_study_attestation"])
    assert isinstance(primary, dict)
    summary_types = group["summary_publication_types"]
    fetch_types = group["fetch_publication_types"]
    assert isinstance(summary_types, list)
    assert isinstance(fetch_types, list)
    assert ma3_module._paper_v5_verify_primary_study(
        primary,
        document_id=str(group["document_id"]),
        summary_types=summary_types,
        fetch_types=fetch_types,
    ) is not None

    primary["method_sentence"] = (
        "Inhibitory potency was imagined using biochemical assays."
    )
    primary["method_sentence_sha256"] = sha256(
        str(primary["method_sentence"]).encode("utf-8")
    ).hexdigest()
    _rehash(primary, "attestation_sha256")

    assert ma3_module._paper_v5_verify_primary_study(
        primary,
        document_id=str(group["document_id"]),
        summary_types=summary_types,
        fetch_types=fetch_types,
    ) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retraction_safe", False),
        ("correction_safe", False),
        ("linked_status_notice", "linked_notice_detected"),
        ("linked_status_notice_types", ["retractionin"]),
        ("linked_status_notice_pmids", ["202"]),
    ],
)
def test_primary_study_linked_notice_attestation_fails_closed_after_rehash(
    field: str,
    value: object,
) -> None:
    packet = _final_packet()
    group = _envelope(packet.candidates[0])["group"]
    assert isinstance(group, dict)
    primary = deepcopy(group["primary_study_attestation"])
    assert isinstance(primary, dict)
    summary_types = group["summary_publication_types"]
    fetch_types = group["fetch_publication_types"]
    assert isinstance(summary_types, list)
    assert isinstance(fetch_types, list)
    primary[field] = value
    _rehash(primary, "attestation_sha256")

    assert ma3_module._paper_v5_verify_primary_study(
        primary,
        document_id=str(group["document_id"]),
        summary_types=summary_types,
        fetch_types=fetch_types,
    ) is None


@pytest.mark.parametrize(
    "ranking_input",
    [
        {"pubmed_value": 501},
        {"value": 501},
        {"source": "PubMed", "measurement": 501},
    ],
)
def test_pubmed_value_in_candidate_ranking_metadata_is_critical(
    ranking_input: dict[str, object],
) -> None:
    packet = _final_packet().model_copy(deep=True)
    packet.candidates[0].metadata["ranking_input"] = ranking_input

    _assert_critical(
        packet,
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_VALUE_LEAKAGE",
    )


def test_declared_member_removal_from_packet_rejects_indivisible_group() -> None:
    packet = _final_packet().model_copy(deep=True)
    packet.candidates.pop(3)

    _assert_critical(
        packet,
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_GROUP_INCOMPLETE",
    )


def test_pubmed_value_cannot_replace_chembl_activity_measurement() -> None:
    packet = _final_packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    candidate.activities[0] = candidate.activities[0].model_copy(
        update={"value": 501.0}
    )

    _assert_critical(
        packet,
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_ATTESTATION_INVALID",
    )


@pytest.mark.parametrize("mutation", ["legacy_primary_source", "document_mismatch"])
def test_v5_independent_source_separation_is_reverified(mutation: str) -> None:
    packet = _final_packet().model_copy(deep=True)
    evidence = packet.candidates[0].evidence[0]
    contract = evidence.metadata["primary_source_contract"]
    assert isinstance(contract, dict)
    if mutation == "legacy_primary_source":
        evidence.metadata["primary_source"] = {
            "schema_version": "candidate_primary_evidence_v2",
            "reason_code": "CHEMBL_DOCUMENT_NOT_FOUND",
        }
    else:
        contract["chembl_document_chembl_id"] = "CHEMBL9999999"
        _rehash(contract, "attestation_sha256")

    _assert_critical(
        packet,
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_ATTESTATION_INVALID",
    )


def test_reserved_group_member_is_retained_but_excluded_and_does_not_count() -> None:
    packet = _packet().model_copy(deep=True)
    control = packet.candidates[3]
    control.status = "excluded"
    control.metadata["policy_eligible"] = False
    control.exclusion_reasons = ["RESERVED_ASSAY_CONTROL"]
    control.safety_flags = ["reserved_assay_control"]
    brief = _brief()
    brief.constraints.reserved_control_compounds = [
        ReservedControlCompound(
            name=control.name,
            registry_ids=[
                NamespacedRegistryIdentifier(
                    namespace="ChEMBL",
                    identifier=str(control.metadata["chembl_id"]),
                )
            ],
        )
    ]
    packet = _with_group_projection(packet)

    assert _audit(packet, brief=brief) == []


@pytest.mark.parametrize(
    "source_tamper",
    ["missing", "none", "non_list", "non_mapping"],
)
def test_alias_synonym_source_tamper_is_incomplete(
    source_tamper: str,
) -> None:
    packet = _packet()
    raw = deepcopy(_envelope(packet.candidates[0])["alias_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)

    if source_tamper == "missing":
        record.pop("synonym_source_present")
    elif source_tamper == "none":
        record["synonym_source_row_count"] = None
    elif source_tamper == "non_list":
        record["malformed_source_codes"] = ["MOLECULE_SYNONYMS_NOT_LIST"]
    elif source_tamper == "non_mapping":
        record["malformed_source_codes"] = [
            "MOLECULE_SYNONYM_ROW_NOT_OBJECT"
        ]
    else:  # pragma: no cover
        raise AssertionError(source_tamper)
    _rehash(record, "record_sha256")
    _rehash(raw, "inventory_sha256")

    assert _verify_alias_inventory(packet, raw) is None


def test_explicit_empty_synonym_lists_remain_complete() -> None:
    packet = _packet()
    raw = deepcopy(_envelope(packet.candidates[0])["alias_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    assert isinstance(records, list)
    for record in records:
        assert isinstance(record, dict)
        accepted = record["accepted_before_owner_check"]
        assert isinstance(accepted, list)
        preferred_rows = [
            item
            for item in accepted
            if isinstance(item, dict) and item.get("source") == "pref_name"
        ]
        assert len(preferred_rows) == 1
        preferred = str(preferred_rows[0]["alias"])
        source_inventory = record["molecule_synonyms_source_inventory"]
        assert isinstance(source_inventory, dict)
        empty_rows_sha256 = _canonical_hash([])
        source_inventory.update(
            {
                "source_key_present": True,
                "list_typed": True,
                "source_value_type": "list",
                "source_row_count": 0,
                "mapping_row_count": 0,
                "non_mapping_row_indices": [],
                "rows_sha256": empty_rows_sha256,
                "observed_value_sha256": empty_rows_sha256,
                "bounded_row_limit": 64,
                "inspected_row_count": 0,
                "truncated": False,
                "rows": [],
                "reason_codes": [],
                "complete": True,
            }
        )
        _rehash(source_inventory, "inventory_sha256")
        record["synonym_source_present"] = True
        record["synonym_source_row_count"] = 0
        record["molecule_synonyms_source_row_count"] = 0
        record["molecule_synonyms_source_sha256"] = source_inventory[
            "inventory_sha256"
        ]
        record["molecule_synonyms_source_reason_codes"] = []
        record["molecule_synonyms_source_complete"] = True
        record["accepted_before_owner_check"] = preferred_rows
        record["rejected_aliases"] = []
        record["accepted_aliases"] = [preferred]
        record["accepted_alias_count"] = 1
        record["preferred_search_alias"] = preferred
        _rehash(record, "record_sha256")
    _rehash(raw, "inventory_sha256")

    assert _verify_alias_inventory(packet, raw) is not None


def test_alias_rejects_flat_ledger_that_omits_deposited_synonym_rows() -> None:
    packet = _packet()
    raw = deepcopy(_envelope(packet.candidates[0])["alias_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    assert isinstance(records, list)
    record = next(
        item
        for item in records
        if isinstance(item, dict) and item["synonym_source_row_count"] > 0
    )
    accepted = record["accepted_before_owner_check"]
    assert isinstance(accepted, list)
    preferred_rows = [
        item
        for item in accepted
        if isinstance(item, dict) and item.get("source") == "pref_name"
    ]
    assert len(preferred_rows) == 1
    preferred = str(preferred_rows[0]["alias"])
    record["synonym_source_row_count"] = 0
    record["accepted_before_owner_check"] = preferred_rows
    record["rejected_aliases"] = []
    record["accepted_aliases"] = [preferred]
    record["accepted_alias_count"] = 1
    record["preferred_search_alias"] = preferred
    _rehash(record, "record_sha256")
    _rehash(raw, "inventory_sha256")

    assert _verify_alias_inventory(packet, raw) is None


def test_target_alias_equivalence_is_token_exact_and_punctuation_collapsed() -> None:
    assert ma3_module._paper_v5_has_target_token_equivalent(
        "GENE-1 inhibitor IV",
        ["GENE1"],
    )
    assert not ma3_module._paper_v5_has_target_token_equivalent(
        "metformin",
        ["MET"],
    )

    packet = _packet()
    raw = deepcopy(_envelope(packet.candidates[0])["alias_inventory"])
    assert isinstance(raw, dict)
    raw["target_aliases"] = ["GENE1"]
    records = raw["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    aliases = record["accepted_aliases"]
    accepted = record["accepted_before_owner_check"]
    assert isinstance(aliases, list)
    assert isinstance(accepted, list)
    aliases[0] = "GENE-1 inhibitor IV"
    accepted[0]["alias"] = aliases[0]
    accepted[0]["alias_key"] = ma3_module._paper_v5_key(aliases[0])
    record["preferred_search_alias"] = aliases[0]
    _rehash(record, "record_sha256")
    _rehash(raw, "inventory_sha256")

    assert _verify_alias_inventory(
        packet,
        raw,
        target=SimpleNamespace(symbol="GENE1", target_name=""),
    ) is None


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("legacy_primary_document_lookup_executed", True),
        ("legacy_primary_document_lookup_candidate_count", 1),
        ("legacy_primary_document_lookup_candidate_ids", ["CHEMBL_EXTRA"]),
        ("legacy_primary_document_lookup_document_count", 1),
        ("legacy_primary_document_lookup_document_ids", ["DOC_EXTRA"]),
        ("legacy_pubmed_type_lookup_executed", True),
        ("legacy_pubmed_type_lookup_candidate_count", 1),
        ("legacy_pubmed_type_lookup_candidate_ids", ["CHEMBL_EXTRA"]),
        ("legacy_pubmed_type_lookup_pmid_count", 1),
        ("legacy_pubmed_type_lookup_pmid_ids", ["PMID_EXTRA"]),
        ("document_lookup_candidate_identity_count", 1),
        ("document_lookup_candidate_identity_ids", ["CHEMBL_EXTRA"]),
        ("document_lookup_id_count", 1),
        ("pubmed_type_lookup_candidate_identity_count", 1),
        ("pubmed_type_lookup_candidate_identity_ids", ["CHEMBL_EXTRA"]),
        ("pubmed_type_lookup_pmid_count", 1),
    ],
)
def test_discovery_bounds_reject_legacy_pubmed_operation_representation(
    field: str,
    tampered: object,
) -> None:
    packet = _packet()
    raw = deepcopy(
        packet.candidates[0].evidence[0].metadata["candidate_discovery_bounds"]
    )
    assert isinstance(raw, dict)
    raw[field] = tampered

    assert ma3_module._paper_v5_discovery_bounds(raw) is None


def test_discovery_recomputes_identity_resolution_phase_from_fixed_pool() -> None:
    packet = _packet()
    raw = deepcopy(
        packet.candidates[0].evidence[0].metadata["candidate_discovery_bounds"]
    )
    assert isinstance(raw, dict)
    resolution = raw["identity_resolution"]
    assert isinstance(resolution, list)
    resolution[0]["max_phase"] = 99

    assert ma3_module._paper_v5_discovery_bounds(raw) is None


def test_discovery_rejects_direct_support_candidate_above_activity_row_bound() -> None:
    packet = _packet()
    raw = deepcopy(
        packet.candidates[0].evidence[0].metadata["candidate_discovery_bounds"]
    )
    assert isinstance(raw, dict)
    direct_ids = raw["paper_native_v5_direct_support_candidate_ids"]
    activity_counts = raw["activity_counts_by_id"]
    assert isinstance(direct_ids, list) and direct_ids
    assert isinstance(activity_counts, dict)
    count = activity_counts[direct_ids[0]]
    assert isinstance(count, dict)
    count["total_count"] = 21
    count["truncated"] = True

    assert ma3_module._paper_v5_discovery_bounds(raw) is None


def test_discovery_rejects_self_consistent_omission_of_selected_identity() -> None:
    packet = _packet()
    raw = deepcopy(
        packet.candidates[0].evidence[0].metadata["candidate_discovery_bounds"]
    )
    assert isinstance(raw, dict)
    resolution = raw["identity_resolution"]
    eligible_ids = list(raw["eligible_identity_ids"])
    assert isinstance(resolution, list)
    omitted = next(
        str(item["molecule_chembl_id"])
        for item in resolution
        if isinstance(item, dict) and item.get("status") == "selected"
    )
    eligible_ids.remove(omitted)
    fixed_ids = eligible_ids[: ma3_module._PAPER_NATIVE_V5_MAX_CANDIDATES]
    resolution_by_id = {
        str(item["molecule_chembl_id"]): item
        for item in resolution
        if isinstance(item, dict)
    }
    pool = [
        {
            "rank": rank,
            "parent_molecule_chembl_id": candidate_id,
            "max_phase": resolution_by_id[candidate_id]["max_phase"],
        }
        for rank, candidate_id in enumerate(fixed_ids, start=1)
    ]

    assert ma3_module._paper_v5_mechanism_pool(
        raw,
        fixed_ids=fixed_ids,
        eligible_ids=eligible_ids,
        preactivity_pool=pool,
    ) is None


def test_discovery_rejects_fixed_identity_with_excluded_selected_state() -> None:
    packet = _packet()
    raw = deepcopy(
        packet.candidates[0].evidence[0].metadata["candidate_discovery_bounds"]
    )
    assert isinstance(raw, dict)
    fixed_ids = raw["prewarning_pool_ids"]
    resolution = raw["identity_resolution"]
    assert isinstance(fixed_ids, list)
    assert isinstance(resolution, list)
    fixed = next(
        item
        for item in resolution
        if isinstance(item, dict)
        and item.get("molecule_chembl_id") == fixed_ids[0]
    )
    assert "PREACTIVITY_POOL_SELECTED" in fixed["reason_codes"]
    fixed["status"] = "excluded"

    assert ma3_module._paper_v5_discovery_bounds(raw) is None


@pytest.mark.parametrize("mechanism_tamper", ["target", "max_phase"])
def test_discovery_recomputes_mechanism_row_semantics_after_rehash(
    mechanism_tamper: str,
) -> None:
    packet = _packet()
    raw = deepcopy(
        packet.candidates[0].evidence[0].metadata["candidate_discovery_bounds"]
    )
    assert isinstance(raw, dict)
    provenance = raw["mechanism_provenance_by_id"]
    assert isinstance(provenance, dict)
    candidate_id = raw["prewarning_pool_ids"][0]
    mechanism = provenance[candidate_id]
    assert isinstance(mechanism, dict)
    rows = mechanism["rows"]
    assert isinstance(rows, list)
    if mechanism_tamper == "target":
        rows[0]["target_chembl_id"] = "CHEMBL_WRONG"
    else:
        rows[0]["max_phase"] = 99
    mechanism["rows_sha256"] = _canonical_hash(rows)
    raw["mechanism_identity_sha256"] = _canonical_hash(
        {
            molecule_id: provenance[molecule_id]["rows_sha256"]
            for molecule_id in sorted(provenance)
        }
    )

    assert ma3_module._paper_v5_discovery_bounds(raw) is None


def test_support_selection_recomputes_earlier_strict_row_not_status_label() -> None:
    packet = _packet()
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    row_audits = record["row_audits"]
    assert isinstance(row_audits, list)
    original = deepcopy(row_audits[0])
    assert isinstance(original, dict)
    earlier = deepcopy(original)
    earlier["retained_index"] = 0
    earlier["retained_rank"] = 1
    earlier["status"] = "not_supported"
    earlier["reason_codes"] = ["FABRICATED_REJECTION"]
    _rehash(earlier, "row_audit_sha256")
    original["retained_index"] = 1
    original["retained_rank"] = 2
    _rehash(original, "row_audit_sha256")
    record["row_audits"] = [earlier, original]
    record["retained_row_count"] = 2
    record["observed_row_reason_codes"] = ["FABRICATED_REJECTION"]

    snapshot = deepcopy(record["selected_support"])
    assert isinstance(snapshot, dict)
    snapshot["retained_index"] = 1
    snapshot["retained_rank"] = 2
    _rehash(snapshot, "support_attestation_sha256")
    record["selected_support"] = deepcopy(snapshot)
    _rehash(record, "record_sha256")
    selected = raw["selected_support_snapshots"]
    selection_order = raw["selection_order"]
    assert isinstance(selected, list)
    assert isinstance(selection_order, list)
    selected[0] = deepcopy(snapshot)
    selection_order[0]["retained_rank"] = 2
    raw["selected_support_snapshots_sha256"] = _canonical_hash(selected)
    raw["selection_order_sha256"] = _canonical_hash(selection_order)
    _rehash(raw, "inventory_sha256")

    assert _verify_support_inventory(packet, raw) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "null",
        "empty",
        "whitespace",
        "padded",
        "non_string",
        "malformed",
    ],
)
def test_recomputed_strict_support_row_requires_exact_document_id_after_rehash(
    mutation: str,
) -> None:
    packet = _packet()
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    row_audits = record["row_audits"]
    assert isinstance(row_audits, list) and row_audits
    audit = row_audits[0]
    assert isinstance(audit, dict)
    raw_tuple = audit["raw_measurement_tuple"]
    assert isinstance(raw_tuple, dict)

    _mutate_document_field(audit, mutation=mutation)
    _mutate_document_field(raw_tuple, mutation=mutation)
    audit["raw_measurement_tuple_sha256"] = _canonical_hash(raw_tuple)
    _rehash(audit, "row_audit_sha256")
    _rehash(record, "record_sha256")
    _rehash(raw, "inventory_sha256")

    assert _verify_support_inventory(packet, raw) is None


def test_recomputed_strict_support_row_rejects_document_copy_mismatch_after_rehash(
) -> None:
    packet = _packet()
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    row_audits = record["row_audits"]
    assert isinstance(row_audits, list) and row_audits
    audit = row_audits[0]
    assert isinstance(audit, dict)
    raw_tuple = audit["raw_measurement_tuple"]
    assert isinstance(raw_tuple, dict)

    audit["document_chembl_id"] = "CHEMBL999990"
    audit["raw_measurement_tuple_sha256"] = _canonical_hash(raw_tuple)
    _rehash(audit, "row_audit_sha256")
    _rehash(record, "record_sha256")
    _rehash(raw, "inventory_sha256")

    assert _verify_support_inventory(packet, raw) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "null",
        "empty",
        "whitespace",
        "padded",
        "non_string",
        "malformed",
    ],
)
def test_selected_support_requires_exact_document_id_after_rehash(
    mutation: str,
) -> None:
    packet = _packet()
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]
    envelope = _envelope(candidate)
    support = deepcopy(envelope["pre_pubmed_chembl_support"])
    fixed_ids = envelope["fixed_pool_candidate_ids"]
    assert isinstance(support, dict)
    assert isinstance(fixed_ids, list)
    raw_tuple = support["raw_measurement_tuple"]
    assert isinstance(raw_tuple, dict)

    _mutate_document_field(support, mutation=mutation)
    _mutate_document_field(raw_tuple, mutation=mutation)
    _mutate_document_field(evidence.metadata, mutation=mutation)
    support["raw_measurement_tuple_sha256"] = _canonical_hash(raw_tuple)
    _rehash(support, "support_attestation_sha256")

    assert ma3_module._paper_v5_support_snapshot(
        support,
        candidate=candidate,
        evidence=evidence,
        activity=candidate.activities[0],
        fixed_ids=fixed_ids,
    ) is None


def test_selected_support_rejects_document_tuple_mismatch_after_self_rehash(
) -> None:
    packet = _packet()
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]
    envelope = _envelope(candidate)
    support = deepcopy(envelope["pre_pubmed_chembl_support"])
    fixed_ids = envelope["fixed_pool_candidate_ids"]
    assert isinstance(support, dict)
    assert isinstance(fixed_ids, list)

    support["document_chembl_id"] = "CHEMBL999991"
    _rehash(support, "support_attestation_sha256")

    assert ma3_module._paper_v5_support_snapshot(
        support,
        candidate=candidate,
        evidence=evidence,
        activity=candidate.activities[0],
        fixed_ids=fixed_ids,
    ) is None


def test_selected_support_rejects_candidate_evidence_document_copy_mismatch(
) -> None:
    packet = _packet()
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]
    envelope = _envelope(candidate)
    support = deepcopy(envelope["pre_pubmed_chembl_support"])
    fixed_ids = envelope["fixed_pool_candidate_ids"]
    assert isinstance(support, dict)
    assert isinstance(fixed_ids, list)
    assert ma3_module._paper_v5_self_hash(
        support,
        key="support_attestation_sha256",
    )

    evidence.metadata["document_chembl_id"] = "CHEMBL999992"

    assert ma3_module._paper_v5_support_snapshot(
        support,
        candidate=candidate,
        evidence=evidence,
        activity=candidate.activities[0],
        fixed_ids=fixed_ids,
    ) is None


def test_support_rejects_pchembl_below_deposited_query_threshold() -> None:
    packet = _packet()
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    selected = raw["selected_support_snapshots"]
    assert isinstance(records, list)
    assert isinstance(selected, list)
    record = records[0]
    assert isinstance(record, dict)
    row_audits = record["row_audits"]
    assert isinstance(row_audits, list)
    assert len(row_audits) == 1
    audit = row_audits[0]
    snapshot = record["selected_support"]
    assert isinstance(audit, dict)
    assert isinstance(snapshot, dict)
    for source in (audit, snapshot):
        raw_tuple = source["raw_measurement_tuple"]
        assert isinstance(raw_tuple, dict)
        raw_tuple["pchembl_value"] = "1.0"
        source["raw_measurement_tuple_sha256"] = _canonical_hash(raw_tuple)
    _rehash(audit, "row_audit_sha256")
    _rehash(snapshot, "support_attestation_sha256")
    record["selected_support"] = deepcopy(snapshot)
    _rehash(record, "record_sha256")
    selected[0] = deepcopy(snapshot)
    raw["selected_support_snapshots_sha256"] = _canonical_hash(selected)
    _rehash(raw, "inventory_sha256")

    assert _verify_support_inventory(packet, raw) is None


@pytest.mark.parametrize(
    "deposited_threshold",
    [1.0, "5.0", True, float("inf")],
    ids=["unapproved", "numeric_string", "bool", "nonfinite"],
)
def test_support_rejects_self_consistent_invalid_query_threshold(
    deposited_threshold: object,
) -> None:
    packet = _packet()
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["paper_native_v5_direct_support_inventory"])
    assert isinstance(raw, dict)
    records = raw["records"]
    selected = raw["selected_support_snapshots"]
    assert isinstance(records, list)
    assert isinstance(selected, list)
    record = records[0]
    assert isinstance(record, dict)
    row_audits = record["row_audits"]
    assert isinstance(row_audits, list)
    assert len(row_audits) == 1
    audit = row_audits[0]
    snapshot = record["selected_support"]
    assert isinstance(audit, dict)
    assert isinstance(snapshot, dict)
    for source, hash_key in (
        (audit, "row_audit_sha256"),
        (snapshot, "support_attestation_sha256"),
    ):
        params = source["chembl_activity_query_parameters"]
        query = source["chembl_activity_query"]
        binding = source["chembl_activity_query_candidate_binding"]
        assert isinstance(params, dict)
        assert isinstance(query, dict)
        assert isinstance(binding, dict)
        params["pchembl_value__gte"] = deposited_threshold
        parameter_sha256 = ma3_module._paper_v5_gateway_parameter_hash(params)
        source["chembl_activity_query_parameters_sha256"] = parameter_sha256
        query["parameter_hash"] = parameter_sha256
        source["chembl_activity_query_sha256"] = _canonical_hash(query)
        binding["parameters"] = deepcopy(params)
        binding["parameter_sha256"] = parameter_sha256
        source["chembl_activity_query_candidate_binding_sha256"] = (
            _canonical_hash(binding)
        )
        _rehash(source, hash_key)
    record["selected_support"] = deepcopy(snapshot)
    _rehash(record, "record_sha256")
    selected[0] = deepcopy(snapshot)
    raw["selected_support_snapshots_sha256"] = _canonical_hash(selected)
    _rehash(raw, "inventory_sha256")

    assert _verify_support_inventory(packet, raw) is None


@pytest.mark.parametrize("value", [5, 5.0])
def test_support_query_threshold_accepts_only_approved_finite_json_numbers(
    value: object,
) -> None:
    assert ma3_module._paper_v5_query_threshold(value) == Decimal("5.0")


@pytest.mark.parametrize(
    "provenance_tamper",
    ["legacy_primary", "legacy_pubmed", "represented_pmid", "call_limit"],
)
def test_primary_provenance_rejects_extra_or_contradictory_calls(
    provenance_tamper: str,
) -> None:
    packet = _packet()
    envelope = _envelope(packet.candidates[0])
    bounds = packet.candidates[0].evidence[0].metadata[
        "candidate_discovery_bounds"
    ]
    assert isinstance(bounds, dict)
    raw = deepcopy(bounds["primary_abstract_search"])
    assert isinstance(raw, dict)
    if provenance_tamper == "legacy_primary":
        raw["legacy_primary_document_lookup_executed"] = True
    elif provenance_tamper == "legacy_pubmed":
        raw["legacy_pubmed_type_lookup_executed"] = True
    elif provenance_tamper == "represented_pmid":
        raw["legacy_pubmed_type_lookup_pmid_count"] = 1
        raw["legacy_pubmed_type_lookup_pmid_ids"] = ["PMID_EXTRA"]
    elif provenance_tamper == "call_limit":
        transport = raw["transport"]
        assert isinstance(transport, dict)
        transport["logical_call_limit"] = 4
    else:  # pragma: no cover
        raise AssertionError(provenance_tamper)
    _rehash(raw, "provenance_sha256")
    fixed_ids = envelope["fixed_pool_candidate_ids"]
    assert isinstance(fixed_ids, list)

    assert not ma3_module._paper_v5_primary_provenance(
        raw,
        fixed_ids=fixed_ids,
        support_inventory_raw=raw["direct_support_inventory"],
        alias_inventory_raw=raw["alias_inventory"],
        search_raw=raw["search_outcome"],
        promotion_raw=raw["promotion"],
    )


def test_ma3_has_no_producer_parser_or_search_import() -> None:
    source = inspect.getsource(__import__("drugpilot.agents.ma3", fromlist=["*"]))

    assert "from drugpilot.tools.primary_abstract_paper_native_v5 import" not in source
    assert "from drugpilot.tools.primary_abstract_paper_native_search_v5 import" not in source
    assert "parse_paper_native_primary_abstract_v5(" not in source
    assert "search_paper_native_primary_abstract_v5(" not in source


def test_independent_grammar_accounts_for_embedded_numbers_and_predictions() -> None:
    assert [
        match.group(0)
        for match in ma3_module._PAPER_NATIVE_V5_NUMBER.finditer("phase3")
    ] == ["3"]
    assert ma3_module._PAPER_NATIVE_V5_UNSAFE.search(
        "predicted in silico"
    )
    assert ma3_module._PAPER_NATIVE_V5_UNSAFE.search("prospective values")
    assert ma3_module._PAPER_NATIVE_V5_UNSAFE.search(
        "prospectively estimated"
    )


@pytest.mark.parametrize(
    "unsafe_phrase",
    [
        "in-silico",
        "modeling",
        "modelling",
        "simulation",
        "projecting",
        "estimated",
        "forecast",
        "anticipated to be",
        "aimed to be",
        "plans to be",
        "proposes to be",
        "will likely be",
        "would have been",
    ],
)
def test_independent_grammar_rejects_future_and_unsafe_morphologies(
    unsafe_phrase: str,
) -> None:
    assert ma3_module._PAPER_NATIVE_V5_UNSAFE.search(
        f"The activity was {unsafe_phrase} 12 nM."
    )


@pytest.mark.parametrize(
    "head",
    [
        "Alphatinib was measured as the most potent GENE inhibitor ",
        "Alphatinib was determined to inhibit GENE ",
        "Alphatinib directly inhibited GENE ",
    ],
    ids=["measured", "determined", "direct_inhibition"],
)
def test_independent_closed_observed_head_accepts_only_deposited_grammar(
    head: str,
) -> None:
    packet = _packet(sentence=_paper_v5_group_sentence(head))
    group = deepcopy(_envelope(packet.candidates[0])["group"])
    assert isinstance(group, dict)

    verified = _verify_envelope_group_pairs(packet, group)

    assert verified is not None
    assert len(verified) == 4


@pytest.mark.parametrize(
    ("head", "accepted", "replacement"),
    [
        (
            "Alphatinib was measured as the most potent GENE inhibitor ",
            "measured",
            "imagined",
        ),
        (
            "Alphatinib was directly, as expected, the most potent GENE "
            "inhibitor ",
            "directly, as expected,",
            "algorithmically-ranked",
        ),
        (
            "Alphatinib was directly observed to be the most potent GENE "
            "inhibitor ",
            "directly",
            "uniquely",
        ),
    ],
    ids=["imagined", "algorithmically_ranked", "unregistered_modifier"],
)
def test_independent_closed_observed_head_rejects_self_consistent_modifiers(
    head: str,
    accepted: str,
    replacement: str,
) -> None:
    packet = _packet(sentence=_paper_v5_group_sentence(head))
    group = deepcopy(_envelope(packet.candidates[0])["group"])
    assert isinstance(group, dict)
    assert _verify_envelope_group_pairs(packet, group) is not None
    _tamper_group_sentence_same_length(
        group,
        accepted=accepted,
        replacement=replacement,
    )

    assert _verify_envelope_group_pairs(packet, group) is None


def test_protected_v2_v4_science_rules_and_qa_reviewer_authority_are_frozen() -> None:
    path = Path(__file__).resolve().parents[2] / "src/drugpilot/agents/ma3.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    expected = {
        "ReviewerFinding": "f3acc10558b33ca030d8bdb7c3deabfaa5407faf4d8ffb3adf8407a04632daf6",
        "ReviewerAssessment": "8df7087db7c40334577af18810a03585a50bfe1b312fe51298fa211238c36658",
        # 2026-09-29: add only three forbidden-substructure integrity BLOCK
        # codes and one unavailable-screen ABSTAIN code. Existing severities,
        # routing and scientific source validators remain unchanged. Adversarial
        # input binding, canonicalization and legacy coverage is maintained in
        # test_substructure_policy.py; do not remove this freeze assertion.
        "AuditPolicy": "6f3cebdd72cdb59855ee73271364f854d3308fb9131d7daf3d2dd36343b7f0a8",
        # 2026-09-23 QA #4 deliberately changes reviewer control authority:
        # preserve reported opinions, cap unconfirmed effective severity, and
        # derive routing only from independent rules/validations. Behavioral
        # adversarial coverage lives in test_reviewer_authority.py. Scientific
        # source/measurement/policy hashes below remain unchanged.
        "_semantic_review": "005690b808b16f01ed093870f48817397895ec039a4e526c2ceba44ceb759e3e",
        "_decide": "5b39f5ca3c18b67dd5eea47fd8b14bd101911961410ca21919d34e0e27305c90",
        "_audit_activity_comparability": "6c3817f0e96edaab21637c544c984a7a5d62a11365f3a0bdc4ea6e0494624690",
        "_audit_primary_abstract_paired_v4": "b3be8f702047d3d14c18b3a49f3a14171b336010c95dbc997c7a66859f4233e8",
        "_audit_candidate_primary_sources": "dcddc1d9213ad6e852a81c91e7290af3093ca33fef8b0a7bc527b238e45d87eb",
    }
    observed: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in expected:
                continue
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            observed[node.name] = sha256(segment.encode("utf-8")).hexdigest()

    assert observed == expected
