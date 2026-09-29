from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.domain.models import (
    CandidateCard,
    CandidatePacket,
    CandidateRoute,
    DataSourcePolicy,
    NamespacedRegistryIdentifier,
    ProjectConstraints,
    ResearchBrief,
    ReservedControlCompound,
    ScientificBackendKind,
    TargetCandidate,
)
from drugpilot.tools.candidate_policy import screen_reserved_control

from test_live_scientific_tools import (
    _PAIRED_FOUR,
    _PairedPubMedClient,
    _run_paired_v4,
)


REQUIRED_TOKEN = "dynamic_pubmed_primary_abstract_paired_v4_required"
ENVELOPE_KEY = "primary_abstract_paired_v4"


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
        research_goal=(
            "Evaluate a bounded exact-target repurposing cohort with comparative "
            "primary evidence."
        ),
        disease="Example molecular disease",
        constraints=ProjectConstraints(
            reserved_control_compounds=[
                ReservedControlCompound(
                    name="Controlib",
                    registry_ids=[
                        NamespacedRegistryIdentifier(
                            namespace="ChEMBL",
                            identifier="CHEMBL104",
                        )
                    ],
                )
            ]
        ),
        data_policy=DataSourcePolicy(require_primary_sources=True),
        human_priorities=list(priorities or [REQUIRED_TOKEN]),
    )


def _packet(*, sentence: str = _PAIRED_FOUR) -> CandidatePacket:
    seeds, _chembl, gateway = _run_paired_v4(
        _PairedPubMedClient({"42007252": sentence})
    )
    try:
        target = TargetCandidate(
            symbol="GENE",
            uniprot_id="P12345",
            metadata={"chembl_target_id": "CHEMBL_TARGET"},
        )
        brief = _brief()
        candidates: list[CandidateCard] = []
        for index, seed in enumerate(seeds, start=1):
            metadata = deepcopy(seed.metadata)
            is_control = metadata["chembl_id"] == "CHEMBL104"
            if is_control:
                reserved = screen_reserved_control(
                    candidate_name=seed.name,
                    metadata=metadata,
                    reserved_controls=(
                        brief.constraints.reserved_control_compounds
                    ),
                )
                metadata.update(
                    {
                        "reserved_control_screen": reserved.as_metadata(),
                        "control_only": True,
                        "policy_eligible": False,
                    }
                )
            candidates.append(
                CandidateCard(
                    candidate_id=f"cand_paired_{index}",
                    route=CandidateRoute.REPURPOSING,
                    name=seed.name,
                    target=target,
                    mechanism=seed.mechanism,
                    smiles=seed.smiles,
                    activities=list(seed.activities),
                    evidence=list(seed.evidence),
                    metadata=metadata,
                    score=0.0,
                    rank=None,
                    target_rank=None,
                    status="excluded" if is_control else "conditional",
                    safety_flags=(
                        ["reserved_assay_control"] if is_control else []
                    ),
                    exclusion_reasons=(
                        ["RESERVED_ASSAY_CONTROL"] if is_control else []
                    ),
                )
            )
        return CandidatePacket(
            plan_id="plan_paired_v4",
            candidates=candidates,
            backend_kind=ScientificBackendKind.LIVE,
        )
    finally:
        gateway.close()


def _audit(
    packet: CandidatePacket,
    *,
    brief: ResearchBrief | None = None,
):
    return MA3AssuranceAgent()._audit_candidate_primary_sources_for_policy(
        brief or _brief(),
        packet,
    )


def _envelope(candidate: CandidateCard) -> dict[str, object]:
    envelope = candidate.evidence[0].metadata[ENVELOPE_KEY]
    assert isinstance(envelope, dict)
    return envelope


def _sync_candidate_copy(candidate: CandidateCard) -> None:
    candidate.metadata[ENVELOPE_KEY] = deepcopy(_envelope(candidate))


def _assert_verified_claim_is_critical(packet: CandidatePacket) -> None:
    findings = _audit(packet)

    assert [(item.code, item.severity.value) for item in findings] == [
        ("CANDIDATE_PRIMARY_PAIRED_V4_ATTESTATION_INVALID", "CRITICAL")
    ]


def test_three_noncontrol_candidates_and_reserved_common_head_pass() -> None:
    packet = _packet()

    assert _audit(packet) == []
    assert packet.candidates[-1].status == "excluded"
    assert _envelope(packet.candidates[-1])["selected_pair_index"] == 3


def test_source_candidate_target_and_endpoint_case_variation_passes() -> None:
    sentence = (
        _PAIRED_FOUR.replace("Alphatinib", "alphatinib")
        .replace("GENE", "gene")
        .replace("IC50", "ic50")
        .replace("Betanib", "BETANIB")
        .replace("Gammanib", "gammanib")
    )
    packet = _packet(sentence=sentence)

    assert _audit(packet) == []


def test_exact_token_dispatch_cannot_be_downgraded_to_legacy_primary_v2() -> None:
    packet = _packet()
    for candidate in packet.candidates:
        candidate.metadata.pop(ENVELOPE_KEY, None)
        candidate.evidence[0].metadata.pop(ENVELOPE_KEY, None)

    required = _audit(packet)
    prose_only = _audit(
        packet,
        brief=_brief(
            priorities=[
                "Please use dynamic_pubmed_primary_abstract_paired_v4_required "
                "when it is available."
            ]
        ),
    )

    assert [(item.code, item.severity.value) for item in required] == [
        ("CANDIDATE_PRIMARY_PAIRED_V4_INSUFFICIENT", "HIGH")
    ]
    assert {item.code for item in prose_only} == {
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    }


def test_two_noncontrol_candidates_plus_reserved_control_is_insufficient() -> None:
    packet = _packet()
    packet.candidates[2].status = "excluded"

    findings = _audit(packet)

    assert [(item.code, item.severity.value) for item in findings] == [
        ("CANDIDATE_PRIMARY_PAIRED_V4_INSUFFICIENT", "HIGH")
    ]


def test_reserved_control_cannot_be_reintroduced_as_an_eligible_candidate() -> None:
    packet = _packet()
    control = packet.candidates[-1]
    control.status = "conditional"
    control.metadata["policy_eligible"] = True

    findings = _audit(packet)

    assert [(item.code, item.severity.value) for item in findings] == [
        ("RESERVED_CONTROL_CANDIDATE_NOT_EXCLUDED", "CRITICAL")
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "query",
        "uid",
        "group_hash",
        "span",
        "value",
        "unit",
        "assay",
        "group_copy",
        "partial_search",
        "review_type",
    ],
)
def test_verified_envelope_tamper_is_deterministic_critical(
    mutation: str,
) -> None:
    packet = _packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    envelope = _envelope(candidate)
    outcome = envelope["search_outcome"]
    group = envelope["group"]
    assert isinstance(outcome, dict)
    assert isinstance(group, dict)

    if mutation == "query":
        outcome["query"] = str(outcome["query"]) + " AND injected"
        outcome["query_sha256"] = sha256(
            str(outcome["query"]).encode("utf-8")
        ).hexdigest()
        _rehash(outcome, "outcome_sha256")
    elif mutation == "uid":
        outcome["records"][0]["document_id"] = "99999999"
        _rehash(outcome, "outcome_sha256")
    elif mutation == "group_hash":
        group["group_sha256"] = "f" * 64
    elif mutation == "span":
        group["pairs"][0]["candidate_span"]["end"] -= 1
    elif mutation == "value":
        envelope["selected_measurement"]["value"] = 999.0
    elif mutation == "unit":
        envelope["selected_measurement"]["unit"] = "uM"
    elif mutation == "assay":
        signature = envelope["selected_assay_signature"]
        signature["assay_chembl_id"] = "ASSAY_TAMPER"
        _rehash(signature, "signature_sha256")
    elif mutation == "group_copy":
        group["sentence"] = str(group["sentence"]) + " Tampered."
    elif mutation == "partial_search":
        outcome["calls"] = outcome["calls"][:2]
        _rehash(outcome, "outcome_sha256")
    else:
        outcome["records"][0]["publication_types"] = [
            "Journal Article",
            "Review",
        ]
        _rehash(outcome, "outcome_sha256")

    _rehash(envelope, "attestation_sha256")
    _sync_candidate_copy(candidate)

    _assert_verified_claim_is_critical(packet)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "partial",
        "empty_mapping_row",
        "conflict",
        "ambiguous",
        "binding_hash",
        "source_inventory_hash",
        "binding_copy",
        "paper_mismatch",
        "case_only",
    ],
)
def test_condition_binding_tamper_is_deterministic_critical(
    mutation: str,
) -> None:
    packet = _packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]
    envelope = _envelope(candidate)

    if mutation == "missing":
        evidence.metadata.pop("activity_properties")
    elif mutation == "partial":
        evidence.metadata["activity_properties"][0].pop("units")
    elif mutation == "empty_mapping_row":
        evidence.metadata["activity_properties"].append({})
    elif mutation == "conflict":
        evidence.metadata["activity_properties"][0]["standard_value"] = "2"
    elif mutation == "ambiguous":
        evidence.metadata["activity_properties"][0]["value"] = "2"
    elif mutation == "binding_hash":
        envelope["selected_assay_condition_binding"][
            "binding_sha256"
        ] = "f" * 64
        _rehash(envelope, "attestation_sha256")
        _sync_candidate_copy(candidate)
    elif mutation == "source_inventory_hash":
        binding = envelope["selected_assay_condition_binding"]
        binding["source_inventory"][0]["inventory_sha256"] = "f" * 64
        _rehash(binding, "binding_sha256")
        _rehash(envelope, "attestation_sha256")
        _sync_candidate_copy(candidate)
    elif mutation == "binding_copy":
        binding = envelope["selected_assay_condition_binding"]
        binding["source_inventory_count"] = 1
        _rehash(binding, "binding_sha256")
        _rehash(envelope, "attestation_sha256")
        _sync_candidate_copy(candidate)
    else:
        paper_condition = envelope["selected_measurement"][
            "pair_assay_condition"
        ]
        if mutation == "paper_mismatch":
            paper_condition["value_raw"] = "2"
        else:
            paper_condition["subject"] = str(
                paper_condition["subject"]
            ).casefold()
        _rehash(envelope, "attestation_sha256")
        _sync_candidate_copy(candidate)

    _assert_verified_claim_is_critical(packet)


@pytest.mark.parametrize(
    "repair_flag",
    [
        "normalization_applied",
        "case_repair_applied",
        "whitespace_repair_applied",
        "unit_conversion_applied",
        "missing_values_imputed",
    ],
)
def test_condition_binding_repair_or_imputation_is_critical(
    repair_flag: str,
) -> None:
    packet = _packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    envelope = _envelope(candidate)
    binding = envelope["selected_assay_condition_binding"]
    binding[repair_flag] = True
    _rehash(binding, "binding_sha256")
    _rehash(envelope, "attestation_sha256")
    _sync_candidate_copy(candidate)

    _assert_verified_claim_is_critical(packet)


@pytest.mark.parametrize(
    "collision",
    ["candidate_owner_nfkc", "candidate_target_nfkc"],
)
def test_nfkc_casefold_alias_collisions_are_critical(
    collision: str,
) -> None:
    packet = _packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    envelope = _envelope(candidate)
    inventory = envelope["alias_inventory"]
    records = inventory["records"]
    fixed_records = inventory["fixed_pool_alias_records"]
    assert isinstance(records, list)
    assert isinstance(fixed_records, list)
    replacement = (
        "Ａｌｐｈａｔｉｎｉｂ"
        if collision == "candidate_owner_nfkc"
        else str(inventory["target_aliases"][0])
    )
    molecule_id = str(records[1]["molecule_chembl_id"])
    for record in [*records, *fixed_records]:
        if record["molecule_chembl_id"] == molecule_id:
            record["preferred_search_alias"] = replacement
            record["exact_aliases"] = [replacement]
            record["source_alias_count"] = 1
    inventory["fixed_pool_alias_records_sha256"] = _canonical_hash(
        fixed_records
    )
    _rehash(inventory, "inventory_sha256")
    _rehash(envelope, "attestation_sha256")
    _sync_candidate_copy(candidate)

    _assert_verified_claim_is_critical(packet)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("raw_mechanism_row_limit", 79),
        ("prewarning_pool_limit", 4),
        ("fixed_prewarning_pool_limit", 4),
        ("requested_eligible_count", 13),
        ("gateway_return_cap_applied", True),
        ("gateway_return_order_matches_fixed_prewarning_subsequence", False),
        ("returned_candidate_ordering_policy", "primary_reordered"),
    ],
)
def test_fixed_max80_top12_discovery_contract_is_critical(
    field: str,
    tampered_value: object,
) -> None:
    packet = _packet().model_copy(deep=True)
    evidence = packet.candidates[0].evidence[0]
    evidence.metadata["candidate_discovery_bounds"][field] = tampered_value

    _assert_verified_claim_is_critical(packet)


def test_gateway_return_must_equal_all_reason_coded_survivors() -> None:
    packet = _packet().model_copy(deep=True)
    evidence = packet.candidates[0].evidence[0]
    bounds = evidence.metadata["candidate_discovery_bounds"]
    omitted = bounds["gateway_return_candidate_ids"].pop()
    bounds["gateway_return_candidate_count"] -= 1
    bounds["downstream_candidate_identity_ids"].remove(omitted)
    bounds["downstream_candidate_identity_count"] -= 1

    _assert_verified_claim_is_critical(packet)


def test_raw_assay_parameter_inventory_row_hash_mismatch_is_critical() -> None:
    packet = _packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]
    context_record = evidence.metadata["chembl_assay_context"]["record"]
    context_record["assay_parameters"][0]["unattested_extra"] = "tampered"

    _assert_verified_claim_is_critical(packet)


def test_excluded_third_candidate_does_not_count_even_with_valid_envelope() -> None:
    packet = _packet()
    packet.candidates[2].status = "excluded"
    packet.candidates[2].exclusion_reasons.append("independent_policy_exclusion")

    findings = _audit(packet)

    assert findings[0].code == "CANDIDATE_PRIMARY_PAIRED_V4_INSUFFICIENT"
    assert findings[0].severity.value == "HIGH"
