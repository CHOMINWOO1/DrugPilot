from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from typing import Any

from drugpilot.domain.models import ActivityMeasurement, CandidateCard, EvidenceRef


ASSAY_COMPARABILITY_POLICY = "same_exact_chembl_assay_v1"
CHEMBL_ASSAY_CONTEXT_SCHEMA = "chembl_assay_context_v1"
ACTIVITY_RANKING_SCHEMA = "activity_ranking_v1"
ACTIVITY_COMPONENT_SUBWEIGHTS: dict[str, float] = {
    "measurement_evidence_quality": 0.5,
    "comparative_potency": 0.5,
}

_TYPED_ACTIVITY_ENDPOINTS = frozenset({"IC50", "Ki", "Kd", "EC50"})


@dataclass(frozen=True, slots=True)
class ActivityBinding:
    """One value-independent activity binding used to form a comparison cohort.

    The binding deliberately retains the recorded numeric value but never
    converts its unit.  ``numeric_eligible`` means that the measurement can
    participate in a same-assay cohort; it is not a claim that a comparator
    exists or that the value is comparable across assays.
    """

    candidate_id: str
    evidence_id: str | None
    target_key: str
    target_chembl_id: str | None
    assay_chembl_id: str | None
    endpoint: str | None
    relation: str | None
    unit: str | None
    value: float | None
    activity_properties_fingerprint: str
    evidence_quality: float
    numeric_eligible: bool
    reason_codes: tuple[str, ...]

    @property
    def signature(self) -> tuple[str, str, str, str, str, str] | None:
        if not self.numeric_eligible:
            return None
        if not all(
            (
                self.target_key,
                self.target_chembl_id,
                self.assay_chembl_id,
                self.endpoint,
                self.relation,
                self.unit,
            )
        ):
            return None
        return (
            self.target_key,
            str(self.target_chembl_id),
            str(self.assay_chembl_id),
            str(self.endpoint),
            str(self.relation),
            str(self.unit),
        )


def target_key(card: CandidateCard) -> str:
    target = card.target
    if target.uniprot_id:
        return f"uniprot:{target.uniprot_id.strip().upper()}"
    if target.ensembl_id:
        return f"ensembl:{target.ensembl_id.strip().upper()}"
    return f"symbol:{target.symbol.strip().upper()}"


def measurement_evidence_quality_details(card: CandidateCard) -> dict[str, Any]:
    """Score provenance completeness without scoring potency magnitude.

    ``standard_value`` participates only in an equality binding check between
    the typed activity and its source record.  Its magnitude, unit conversion,
    and pChEMBL never enter this score.  Missing/partial assay context is kept
    as provenance but cannot receive the same quality as a completed exact
    ChEMBL assay binding.
    """

    evidence_by_id = {
        evidence.evidence_id: evidence for evidence in card.evidence
    }
    group_weights = {
        "chembl_exact_target_binding": 0.25,
        "standard_activity_record_binding": 0.25,
        "exact_assay_identity_binding": 0.20,
        "completed_exact_assay_context": 0.30,
    }
    candidates: list[dict[str, Any]] = []
    for activity in card.activities:
        evidence = evidence_by_id.get(activity.source_evidence_id or "")
        metadata: Mapping[str, Any] = (
            evidence.metadata if evidence is not None else {}
        )
        assay_id = str(metadata.get("assay_chembl_id") or "").strip()
        target_chembl_id = str(
            metadata.get("target_chembl_id") or ""
        ).strip()
        context = metadata.get("chembl_assay_context")
        context_mapping: Mapping[str, Any] = (
            context if isinstance(context, Mapping) else {}
        )
        record = context_mapping.get("record")
        record_mapping: Mapping[str, Any] = (
            record if isinstance(record, Mapping) else {}
        )
        checks = {
            "chembl_exact_target_binding": bool(
                evidence is not None
                and evidence.source == "ChEMBL"
                and _exact_target_bound(card, evidence)
            ),
            "standard_activity_record_binding": bool(
                evidence is not None
                and activity.endpoint in _TYPED_ACTIVITY_ENDPOINTS
                and str(metadata.get("standard_type") or "")
                == activity.endpoint
                and str(metadata.get("standard_relation") or "")
                == activity.relation
                and str(metadata.get("standard_units") or "") == activity.unit
                and _decimal_equal(
                    metadata.get("standard_value"),
                    activity.value,
                )
            ),
            "exact_assay_identity_binding": bool(
                assay_id
                and target_chembl_id
                and str(context_mapping.get("assay_chembl_id") or "")
                == assay_id
                and str(record_mapping.get("assay_chembl_id") or "")
                == assay_id
                and str(record_mapping.get("target_chembl_id") or "")
                == target_chembl_id
            ),
            "completed_exact_assay_context": bool(
                context_mapping.get("schema_version")
                == CHEMBL_ASSAY_CONTEXT_SCHEMA
                and context_mapping.get("retrieval_status") == "completed"
                and context_mapping.get("binding_status") == "exact"
                and context_mapping.get("missing_values_imputed") is False
                and context_mapping.get("inference_used") is False
            ),
        }
        value = round(
            sum(
                group_weights[name]
                for name, passed in checks.items()
                if passed
            ),
            6,
        )
        candidates.append(
            {
                "source_evidence_id": activity.source_evidence_id,
                "value": value,
                "groups": checks,
            }
        )

    selected = max(
        candidates,
        key=lambda item: (
            float(item["value"]),
            str(item.get("source_evidence_id") or ""),
        ),
        default=None,
    )
    return {
        "method": "measurement_provenance_completeness_v1",
        "value": float(selected["value"]) if selected else 0.0,
        "group_weights": group_weights,
        "selected_source_evidence_id": (
            selected.get("source_evidence_id") if selected else None
        ),
        "groups": dict(selected["groups"]) if selected else {},
        "numeric_magnitude_scored": False,
        "pchembl_scored": False,
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
    }


def measurement_evidence_quality(card: CandidateCard) -> float:
    return float(measurement_evidence_quality_details(card)["value"])


def _decimal_equal(left: Any, right: Any) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return False


def _exact_target_bound(card: CandidateCard, evidence: EvidenceRef) -> bool:
    metadata = evidence.metadata
    target = card.target
    if target.uniprot_id:
        observed = str(
            metadata.get("target_accession")
            or metadata.get("target_uniprot_id")
            or ""
        ).strip()
        if observed.casefold() != target.uniprot_id.strip().casefold():
            return False
    elif target.ensembl_id:
        observed = str(metadata.get("target_ensembl_id") or "").strip()
        if observed.casefold() != target.ensembl_id.strip().casefold():
            return False
    else:
        observed = str(metadata.get("target_gene") or "").strip()
        if observed.casefold() != target.symbol.strip().casefold():
            return False

    expected_chembl = str(target.metadata.get("chembl_target_id") or "").strip()
    observed_chembl = str(metadata.get("target_chembl_id") or "").strip()
    if expected_chembl and observed_chembl != expected_chembl:
        return False
    return bool(observed_chembl)


def _properties_fingerprint(value: Any) -> str:
    properties = value if isinstance(value, list) else []
    encoded = json.dumps(
        properties,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _binding_for_activity(
    card: CandidateCard,
    activity: ActivityMeasurement,
    evidence_by_id: Mapping[str, EvidenceRef],
) -> ActivityBinding:
    reasons: list[str] = []
    evidence_id = activity.source_evidence_id
    evidence = evidence_by_id.get(evidence_id or "")
    metadata: Mapping[str, Any] = evidence.metadata if evidence is not None else {}

    if not evidence_id or evidence is None:
        reasons.append("SOURCE_EVIDENCE_MISSING")
    elif evidence.source != "ChEMBL":
        reasons.append("NON_CHEMBL_ACTIVITY_SOURCE")

    if evidence is None or not _exact_target_bound(card, evidence):
        reasons.append("TARGET_BINDING_UNRESOLVED")

    assay_id = str(metadata.get("assay_chembl_id") or "").strip() or None
    if assay_id is None:
        reasons.append("ASSAY_ID_MISSING")
    if activity.relation != "=":
        reasons.append("RELATION_NOT_EXACT")
    if activity.endpoint not in _TYPED_ACTIVITY_ENDPOINTS:
        reasons.append("ENDPOINT_UNSUPPORTED")

    if evidence is not None:
        if str(metadata.get("standard_relation") or "") != activity.relation:
            reasons.append("ACTIVITY_RELATION_BINDING_MISMATCH")
        if str(metadata.get("standard_units") or "") != activity.unit:
            reasons.append("ACTIVITY_UNIT_BINDING_MISMATCH")
        if str(metadata.get("standard_type") or "") != activity.endpoint:
            reasons.append("ACTIVITY_ENDPOINT_BINDING_MISMATCH")
        if not _decimal_equal(metadata.get("standard_value"), activity.value):
            reasons.append("ACTIVITY_VALUE_BINDING_MISMATCH")

    assay_context = metadata.get("chembl_assay_context")
    if not isinstance(assay_context, Mapping):
        reasons.append("ASSAY_CONTEXT_MISSING")
        assay_context = {}
    elif assay_context.get("retrieval_status") != "completed":
        reasons.append("ASSAY_CONTEXT_NOT_COMPLETED")

    record = assay_context.get("record")
    if not isinstance(record, Mapping):
        reasons.append("ASSAY_RECORD_MISSING")
        record = {}
    if assay_context.get("schema_version") != CHEMBL_ASSAY_CONTEXT_SCHEMA:
        reasons.append("ASSAY_CONTEXT_SCHEMA_MISMATCH")
    if assay_context.get("binding_status") != "exact":
        reasons.append("ASSAY_BINDING_NOT_EXACT")
    if str(assay_context.get("assay_chembl_id") or "") != str(assay_id or ""):
        reasons.append("ASSAY_CONTEXT_ID_MISMATCH")
    if str(record.get("assay_chembl_id") or "") != str(assay_id or ""):
        reasons.append("ASSAY_RECORD_ID_MISMATCH")
    target_chembl_id = str(metadata.get("target_chembl_id") or "").strip() or None
    if str(record.get("target_chembl_id") or "") != str(target_chembl_id or ""):
        reasons.append("ASSAY_TARGET_BINDING_MISMATCH")
    if assay_context.get("missing_values_imputed") is not False:
        reasons.append("ASSAY_CONTEXT_IMPUTATION_STATUS_INVALID")
    if assay_context.get("inference_used") is not False:
        reasons.append("ASSAY_CONTEXT_INFERENCE_STATUS_INVALID")

    return ActivityBinding(
        candidate_id=card.candidate_id,
        evidence_id=evidence_id,
        target_key=target_key(card),
        target_chembl_id=target_chembl_id,
        assay_chembl_id=assay_id,
        endpoint=activity.endpoint,
        relation=activity.relation,
        unit=activity.unit,
        value=float(activity.value),
        activity_properties_fingerprint=_properties_fingerprint(
            metadata.get("activity_properties")
        ),
        evidence_quality=measurement_evidence_quality(card),
        numeric_eligible=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def activity_binding(card: CandidateCard) -> ActivityBinding:
    evidence_by_id = {
        evidence.evidence_id: evidence for evidence in card.evidence
    }
    bindings = [
        _binding_for_activity(card, activity, evidence_by_id)
        for activity in sorted(
            card.activities,
            key=lambda item: (
                str(item.source_evidence_id or ""),
                item.endpoint,
                item.relation,
                item.unit,
            ),
        )
    ]
    if len(bindings) == 1:
        return bindings[0]

    reasons = (
        ["NO_MEASURED_ACTIVITY"]
        if not bindings
        else [
            "MULTIPLE_ACTIVITY_BINDINGS",
            *(reason for binding in bindings for reason in binding.reason_codes),
        ]
    )
    representative = bindings[0] if bindings else None
    return ActivityBinding(
        candidate_id=card.candidate_id,
        evidence_id=representative.evidence_id if representative else None,
        target_key=target_key(card),
        target_chembl_id=(representative.target_chembl_id if representative else None),
        assay_chembl_id=(representative.assay_chembl_id if representative else None),
        endpoint=representative.endpoint if representative else None,
        relation=representative.relation if representative else None,
        unit=representative.unit if representative else None,
        value=representative.value if representative else None,
        activity_properties_fingerprint=(
            representative.activity_properties_fingerprint
            if representative
            else _properties_fingerprint([])
        ),
        evidence_quality=measurement_evidence_quality(card),
        numeric_eligible=False,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _cohort_mismatch_reason(bindings: Sequence[ActivityBinding]) -> str:
    if any(not binding.numeric_eligible for binding in bindings):
        return "COHORT_BINDING_INCOMPLETE"
    if len({binding.target_key for binding in bindings}) != 1 or len(
        {binding.target_chembl_id for binding in bindings}
    ) != 1:
        return "TARGET_SIGNATURE_MISMATCH"
    if len({binding.assay_chembl_id for binding in bindings}) != 1:
        return "DISTINCT_ASSAY_IDS"
    if len({binding.endpoint for binding in bindings}) != 1:
        return "ENDPOINT_MISMATCH"
    if len({binding.relation for binding in bindings}) != 1:
        return "RELATION_MISMATCH"
    if len({binding.unit for binding in bindings}) != 1:
        return "UNIT_MISMATCH"
    if len(
        {binding.activity_properties_fingerprint for binding in bindings}
    ) != 1:
        return "ACTIVITY_PROPERTIES_MISMATCH"
    return "COMPARABILITY_SIGNATURE_MISMATCH"


def _signature_dict(binding: ActivityBinding) -> dict[str, Any] | None:
    signature = binding.signature
    if signature is None:
        return None
    payload = {
        "target_key": signature[0],
        "target_chembl_id": signature[1],
        "assay_chembl_id": signature[2],
        "endpoint": signature[3],
        "relation": signature[4],
        "unit": signature[5],
        "activity_properties_sha256": binding.activity_properties_fingerprint,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **payload,
        "signature_sha256": sha256(encoded.encode("utf-8")).hexdigest(),
    }


def derive_activity_rankings(
    cards: Sequence[CandidateCard],
) -> dict[str, dict[str, Any]]:
    """Derive per-candidate ranking inputs without trusting producer metadata."""

    bindings = {card.candidate_id: activity_binding(card) for card in cards}
    by_target: dict[str, list[CandidateCard]] = {}
    for card in cards:
        if card.status != "excluded":
            by_target.setdefault(target_key(card), []).append(card)

    comparison: dict[str, tuple[bool, float | None, str, int]] = {}
    for target, target_cards in by_target.items():
        target_bindings = [bindings[card.candidate_id] for card in target_cards]
        if len(target_cards) < 2:
            reason = "INSUFFICIENT_COMPARATORS"
            for card in target_cards:
                comparison[card.candidate_id] = (False, None, reason, 1)
            continue

        signatures = [binding.signature for binding in target_bindings]
        properties = {
            binding.activity_properties_fingerprint for binding in target_bindings
        }
        comparable = (
            all(binding.numeric_eligible for binding in target_bindings)
            and len(set(signatures)) == 1
            and None not in signatures
            and len(properties) == 1
        )
        if not comparable:
            reason = _cohort_mismatch_reason(target_bindings)
            for card in target_cards:
                comparison[card.candidate_id] = (
                    False,
                    None,
                    reason,
                    len(target_cards),
                )
            continue

        values = sorted(
            {
                float(binding.value)
                for binding in target_bindings
                if binding.value is not None
            }
        )
        if not values:
            for card in target_cards:
                comparison[card.candidate_id] = (
                    False,
                    None,
                    "COMPARATIVE_VALUE_MISSING",
                    len(target_cards),
                )
            continue
        value_scores = (
            {values[0]: 1.0}
            if len(values) == 1
            else {
                value: round(1.0 - (index / (len(values) - 1)), 6)
                for index, value in enumerate(values)
            }
        )
        for card in target_cards:
            binding = bindings[card.candidate_id]
            comparison[card.candidate_id] = (
                True,
                value_scores[float(binding.value)],
                "SAME_EXACT_ASSAY_COHORT",
                len(target_cards),
            )

    results: dict[str, dict[str, Any]] = {}
    for card in cards:
        binding = bindings[card.candidate_id]
        if card.status == "excluded":
            used, comparative, cohort_reason, cohort_size = (
                False,
                None,
                "EXCLUDED_CANDIDATE",
                0,
            )
        else:
            used, comparative, cohort_reason, cohort_size = comparison.get(
                card.candidate_id,
                (False, None, "TARGET_COHORT_MISSING", 0),
            )
        evidence_quality = binding.evidence_quality
        evidence_quality_details = measurement_evidence_quality_details(card)
        available_activity_inputs = {
            "measurement_evidence_quality": evidence_quality,
            **(
                {"comparative_potency": float(comparative)}
                if used and comparative is not None
                else {}
            ),
        }
        activity_denominator = sum(
            ACTIVITY_COMPONENT_SUBWEIGHTS[name]
            for name in available_activity_inputs
        )
        activity_component = (
            round(
                sum(
                    ACTIVITY_COMPONENT_SUBWEIGHTS[name] * value
                    for name, value in available_activity_inputs.items()
                )
                / activity_denominator,
                6,
            )
            if activity_denominator
            else 0.0
        )
        reason_codes = list(binding.reason_codes)
        if cohort_reason != "SAME_EXACT_ASSAY_COHORT":
            reason_codes.append(cohort_reason)
        results[card.candidate_id] = {
            "schema_version": ACTIVITY_RANKING_SCHEMA,
            "policy": ASSAY_COMPARABILITY_POLICY,
            "status": (
                "comparable_same_exact_assay"
                if used
                else "not_comparable"
            ),
            "measurement_retained": bool(card.activities),
            "measurement_evidence_quality": evidence_quality,
            "measurement_evidence_quality_details": evidence_quality_details,
            "comparative_potency": comparative,
            "comparative_potency_method": (
                "within_assay_dense_rank_percentile_v1" if used else None
            ),
            "comparative_potency_used": used,
            "numeric_value_used_for_ranking": used,
            "activity_component_value": activity_component,
            "activity_component_subweights": dict(
                ACTIVITY_COMPONENT_SUBWEIGHTS
            ),
            "activity_component_available_inputs": sorted(
                available_activity_inputs
            ),
            "activity_component_available_weight": round(
                activity_denominator,
                6,
            ),
            "activity_component_basis": (
                "available_input_weighted_mean_with_same_assay_comparative_potency"
                if used
                else "measurement_evidence_quality_only_no_numeric_potency"
            ),
            "cohort_size": cohort_size,
            "comparison_signature": _signature_dict(binding),
            "source_evidence_id": binding.evidence_id,
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "unit_conversion_applied": False,
            "description_similarity_used": False,
            "missing_value_inference_used": False,
            "pchembl_used_for_ranking": False,
        }
    return results


def comparability_summary(
    rankings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    comparable = sorted(
        candidate_id
        for candidate_id, item in rankings.items()
        if item.get("comparative_potency_used") is True
    )
    return {
        "schema_version": ACTIVITY_RANKING_SCHEMA,
        "policy": ASSAY_COMPARABILITY_POLICY,
        "candidate_count": len(rankings),
        "comparative_potency_candidate_ids": comparable,
        "cross_assay_numeric_comparison": False,
        "unit_conversion_applied": False,
        "description_similarity_used": False,
        "missing_value_inference_used": False,
        "pchembl_used_for_ranking": False,
    }
