"""Pure, fail-closed postprocessing of MA1 context-routing results.

This module deliberately sits outside MA1 and the orchestration graph.  It
takes an immutable-style input pair -- a :class:`HypothesisPacket` and a
:class:`ContextRoutingResult` -- and returns a deep-copied packet containing
validated context records in the matching hypothesis ``evidence`` list only.

Context records are never linked from ``Claim.evidence_ids`` or
``Claim.counter_evidence_ids``.  Explicit ``hypothesis_id`` routing takes
precedence; the ``(track, symbol)`` key is used only when the bundle has no
hypothesis identifier.  Ambiguous or inconsistent target matches raise before
an output is returned.  Unsafe records are quarantined as audit counts and are
not copied into the packet.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from drugpilot.domain.models import (
    EvidenceRef,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
)
from drugpilot.tools.context_router import ContextRoutingResult, TargetContextBundle


CONTEXT_ROUTING_METADATA_KEY = "context_routing"


class MA1ContextEnrichmentError(ValueError):
    """Base error for a context result that cannot be merged safely."""


class ContextTargetMatchError(MA1ContextEnrichmentError):
    """A routed target has no exact, unique hypothesis destination."""


def enrich_hypothesis_packet_with_context(
    packet: HypothesisPacket,
    routing: ContextRoutingResult,
) -> HypothesisPacket:
    """Return a deep-copied packet enriched with context-only evidence.

    The operation is transactional from the caller's perspective: all bundle
    destinations and disease consistency are validated before the copy is
    enriched.  A matching error therefore cannot produce a partially merged
    result.
    """

    destinations = _resolve_destinations(packet, routing)
    enriched = packet.model_copy(deep=True)
    claim_links_before = _claim_link_snapshot(packet)

    status_counts: Counter[str] = Counter()
    source_outcome_counts: Counter[str] = Counter()
    source_record_counts: Counter[str] = Counter()
    source_added_counts: Counter[str] = Counter()
    source_reused_counts: Counter[str] = Counter()
    quarantine_reason_counts: Counter[str] = Counter()
    quarantined_records: list[dict[str, str]] = []
    source_outcomes: list[dict[str, object]] = []
    added_count = 0
    reused_count = 0
    conflict_count = 0
    preexisting_conflict_count = 0
    reused_record_ids: list[str] = []
    conflicting_record_ids: list[str] = []

    for bundle in routing.targets:
        for outcome in bundle.sources:
            status_counts[outcome.status] += 1
            source_outcome_counts[outcome.source] += 1
            # Preserve one bounded, sanitized record per routed source outcome.
            # The previous source/count projection could not tell whether a
            # particular source was empty, skipped, or actually failed.  That
            # ambiguity made the read-only call trace either hide a real error
            # or incorrectly paint every source as failed.
            source_outcomes.append(
                {
                    "target_symbol": bundle.target.symbol,
                    "hypothesis_id": bundle.target.hypothesis_id,
                    "track": bundle.target.track.value,
                    "effective_work_track": _effective_work_track(bundle).value,
                    "work_track_basis": bundle.target.work_track_basis,
                    **outcome.model_dump(mode="json"),
                }
            )

    for bundle, hypothesis_index in zip(routing.targets, destinations, strict=True):
        hypothesis = enriched.hypotheses[hypothesis_index]
        hypothesis.target.metadata = {
            **hypothesis.target.metadata,
            "context_source_outcomes": [
                outcome.model_dump(mode="json") for outcome in bundle.sources
            ],
        }
        (
            existing_by_id,
            preexisting_reused,
            preexisting_conflicts,
        ) = _deduplicate_existing_evidence(hypothesis)
        for record in preexisting_reused:
            reused_count += 1
            reused_record_ids.append(record.evidence_id)
            source_reused_counts[record.source] += 1
        for record in preexisting_conflicts:
            conflict_count += 1
            preexisting_conflict_count += 1
            conflicting_record_ids.append(record.evidence_id)
            _quarantine(
                quarantined_records,
                quarantine_reason_counts,
                hypothesis,
                record,
                "preexisting_duplicate_id_conflict",
                existing=existing_by_id.get(record.evidence_id),
            )
        claim_link_ids = _claim_link_ids(hypothesis)

        for record in bundle.evidence:
            source_record_counts[record.source] += 1
            if not _is_safe_context_record(record):
                _quarantine(
                    quarantined_records,
                    quarantine_reason_counts,
                    hypothesis,
                    record,
                    "non_context_or_unsafe_flags",
                )
                continue
            if record.evidence_id in claim_link_ids:
                # Adding this record would silently satisfy an existing causal
                # or counter-evidence link even though we never edited a Claim.
                _quarantine(
                    quarantined_records,
                    quarantine_reason_counts,
                    hypothesis,
                    record,
                    "claim_link_id_collision",
                )
                continue

            existing = existing_by_id.get(record.evidence_id)
            if existing is not None:
                if _same_immutable_evidence(existing, record):
                    reused_count += 1
                    reused_record_ids.append(record.evidence_id)
                    source_reused_counts[existing.source] += 1
                else:
                    conflict_count += 1
                    conflicting_record_ids.append(record.evidence_id)
                    _quarantine(
                        quarantined_records,
                        quarantine_reason_counts,
                        hypothesis,
                        record,
                        "duplicate_id_conflict",
                        existing=existing,
                    )
                continue

            copied = record.model_copy(deep=True)
            hypothesis.evidence.append(copied)
            existing_by_id[copied.evidence_id] = copied
            source_added_counts[copied.source] += 1
            added_count += 1

    if _claim_link_snapshot(enriched) != claim_links_before:
        # Defensive assertion: no current code path mutates claims.  Retain a
        # runtime guard so a future refactor cannot silently broaden context.
        raise MA1ContextEnrichmentError(
            "context enrichment must not modify claim evidence links"
        )

    routing_status = _routing_status(
        routing,
        status_counts=status_counts,
        quarantined_count=len(quarantined_records),
    )
    sorted_outcomes = _sorted_counts(source_outcome_counts)
    sorted_records = _sorted_counts(source_record_counts)
    sorted_added = _sorted_counts(source_added_counts)
    sorted_reused = _sorted_counts(source_reused_counts)
    source_counts: dict[str, dict[str, int]] = {
        "outcomes": sorted_outcomes,
        "records_observed": sorted_records,
        "records_added": sorted_added,
    }
    if sorted_reused:
        source_counts["records_reused"] = sorted_reused
    audit = {
        "status": routing_status,
        "source_calls_used": routing.source_calls_used,
        "planned_source_calls": routing.planned_source_calls,
        "elapsed_seconds": routing.elapsed_seconds,
        "budget_exhausted": routing.budget_exhausted,
        "stop_reason": routing.stop_reason,
        "safety_scope": routing.safety_scope,
        "status_counts": _sorted_counts(status_counts),
        "source_counts": source_counts,
        "source_outcome_counts": sorted_outcomes,
        "source_outcomes": source_outcomes,
        "source_record_counts": sorted_records,
        "source_added_counts": sorted_added,
        "source_reused_counts": sorted_reused,
        "matched_bundle_count": len(destinations),
        "records_added": added_count,
        "records_reused": reused_count,
        "reused_record_ids": list(dict.fromkeys(reused_record_ids)),
        "records_conflicted": conflict_count,
        "conflicting_record_ids": list(dict.fromkeys(conflicting_record_ids)),
        # Backward-compatible field: count only conflicting instances that
        # physically existed in the input packet and were collapsed. Incoming
        # conflicts are rejected/quarantined, not described as removed.
        "duplicate_ids_removed": preexisting_conflict_count,
        "records_quarantined": len(quarantined_records),
        "quarantine_reason_counts": _sorted_counts(quarantine_reason_counts),
        "quarantined_records": quarantined_records,
        "claim_links_modified": False,
    }
    metadata = dict(enriched.metadata)
    metadata[CONTEXT_ROUTING_METADATA_KEY] = audit
    enriched.metadata = metadata
    return enriched


def _resolve_destinations(
    packet: HypothesisPacket,
    routing: ContextRoutingResult,
) -> list[int]:
    routing_disease = _normalized(routing.disease)
    packet_diseases = {_normalized(item.disease) for item in packet.hypotheses}
    if packet.hypotheses and packet_diseases != {routing_disease}:
        raise ContextTargetMatchError(
            "routing disease must exactly match every hypothesis disease"
        )

    by_id: dict[str, list[int]] = {}
    by_target: dict[tuple[str, str], list[int]] = {}
    for index, hypothesis in enumerate(packet.hypotheses):
        by_id.setdefault(hypothesis.hypothesis_id, []).append(index)
        by_target.setdefault(_hypothesis_key(hypothesis), []).append(index)

    destinations: list[int] = []
    for bundle in routing.targets:
        target = bundle.target
        if target.hypothesis_id:
            matches = by_id.get(target.hypothesis_id, [])
            if len(matches) != 1:
                raise ContextTargetMatchError(
                    "explicit hypothesis_id must match exactly one hypothesis: "
                    f"{target.hypothesis_id!r}"
                )
            index = matches[0]
            hypothesis = packet.hypotheses[index]
            if _hypothesis_key(hypothesis) != _bundle_key(bundle):
                raise ContextTargetMatchError(
                    "explicit hypothesis_id resolved to a different track or symbol: "
                    f"{target.hypothesis_id!r}"
                )
        else:
            key = _bundle_key(bundle)
            matches = by_target.get(key, [])
            if len(matches) != 1:
                raise ContextTargetMatchError(
                    "(track, symbol) must match exactly one hypothesis when "
                    f"hypothesis_id is absent: {key!r}"
                )
            index = matches[0]
        destinations.append(index)
    return destinations


def _deduplicate_existing_evidence(
    hypothesis: Hypothesis,
) -> tuple[dict[str, EvidenceRef], list[EvidenceRef], list[EvidenceRef]]:
    """Collapse existing IDs while separating reuse from actual conflicts."""

    unique: list[EvidenceRef] = []
    by_id: dict[str, EvidenceRef] = {}
    reused: list[EvidenceRef] = []
    conflicts: list[EvidenceRef] = []
    for record in hypothesis.evidence:
        existing = by_id.get(record.evidence_id)
        if existing is not None:
            if _same_immutable_evidence(existing, record):
                reused.append(record)
            else:
                conflicts.append(record)
            continue
        by_id[record.evidence_id] = record
        unique.append(record)
    hypothesis.evidence = unique
    return by_id, reused, conflicts


def _is_safe_context_record(record: EvidenceRef) -> bool:
    metadata = record.metadata
    return (
        record.supports == "context"
        and metadata.get("context_only") is True
        and metadata.get("causal_inference_allowed") is False
        and metadata.get("counter_evidence_allowed") is False
    )


_VOLATILE_EVIDENCE_METADATA = {
    # Deterministic downstream linkage annotations describe how an immutable
    # record is used by MA1; they are not part of the source record itself.
    "claim_linkage_policy",
    "claim_linkage_roles",
    "claim_linkage_scope",
    "claim_linkage_status",
    "human_priority_required",
    "linked_claim_ids",
    # Per-invocation coverage state is audited on the packet/route and kept on
    # the stored record, but it is not source-record content.
    "partial_source_failures",
    "request_origin",
    "retrieved_at",
    "routing_policy",
    # MA1 claim reconciliation annotates every stored record after retrieval.
    # These fields classify whether the record may support a claim; they do
    # not change the immutable source record and are absent on a fresh context
    # adapter result during a later rework pass.
    "direct_support_eligible",
    "direct_support_rejection_codes",
    "verified_direct_primary",
    "verified_direct_primary_rejection_codes",
}

_TRUST_BOUNDARY_METADATA = {
    "cache_hit",
    "demo_fixture",
    "fixture",
    "live",
    "mock",
    "origin_content_hash",
    "origin_provenance_class",
    "provenance_class",
    "retrieved_live",
}

_FIXTURE_PROVENANCE_MARKERS = ("fixture", "mock", "demo")
_RECONCILIATION_POLICIES = {
    "claim-specific-primary-direction-excerpt-v1",
    "atomic-publication-intervention-endpoint-linkage-v2",
}
_SUPPORT_LINKAGE_ROLES = {
    "direction_primary_support",
    "direct_target_direction_primary",
    "primary_target_disease_association",
    "target_disease_association_support",
}
_COUNTER_LINKAGE_ROLES = {
    "direction_primary_refutation",
    "direct_target_direction_refutation",
}
_CONTEXT_LINKAGE_ROLES = {
    "claim_level_counter_context",
    "context_unlinked",
    "counter_context_unlinked",
    "drug_specific_resistance_context_unlinked",
    "resistance_context_linked",
    "resistance_context_unlinked",
    "ranked_association_discovery_context_unlinked",
    "response_modifier_context_unlinked",
}


def _same_immutable_evidence(left: EvidenceRef, right: EvidenceRef) -> bool:
    """Compare source content without crossing an evidence trust boundary."""

    return (
        _immutable_evidence_payload(left) == _immutable_evidence_payload(right)
        and _valid_source_role_projection(left)
        and _valid_source_role_projection(right)
        and _trust_boundaries_compatible(left, right)
    )


def _immutable_evidence_payload(record: EvidenceRef) -> dict[str, object]:
    value = record.model_dump(mode="json", exclude={"retrieved_at"})
    metadata = dict(value.get("metadata") or {})
    value["supports"] = _immutable_source_supports(
        current_supports=str(value.get("supports") or ""),
        metadata=metadata,
    )
    metadata.pop("source_supports", None)
    metadata.pop("pre_reconciliation_supports", None)
    for key in _VOLATILE_EVIDENCE_METADATA | _TRUST_BOUNDARY_METADATA:
        metadata.pop(key, None)
    value["metadata"] = metadata
    return value


def _immutable_source_supports(
    *,
    current_supports: str,
    metadata: dict[str, object],
) -> str:
    """Project a reconciled role back to its immutable source polarity."""

    allowed = {"support", "counter", "context"}
    has_source = "source_supports" in metadata
    has_legacy = "pre_reconciliation_supports" in metadata
    source = metadata.get("source_supports") if has_source else None
    legacy = metadata.get("pre_reconciliation_supports") if has_legacy else None
    if has_source and source not in allowed:
        return f"invalid_source_supports:{source!r}"
    if has_legacy and legacy not in allowed:
        return f"invalid_legacy_source_supports:{legacy!r}"
    if has_source and has_legacy and source != legacy:
        return f"conflicting_source_supports:{source!r}:{legacy!r}"
    if has_source:
        return str(source)
    if has_legacy:
        return str(legacy)
    return current_supports


def _valid_source_role_projection(record: EvidenceRef) -> bool:
    """Accept a changed display role only under the deterministic MA1 contract."""

    metadata = record.metadata
    current = record.supports
    source = _immutable_source_supports(
        current_supports=current,
        metadata=dict(metadata),
    )
    if source not in {"support", "counter", "context"}:
        return False

    status = metadata.get("claim_linkage_status")
    policy = metadata.get("claim_linkage_policy")
    raw_roles = metadata.get("claim_linkage_roles")
    raw_links = metadata.get("linked_claim_ids")
    scope = metadata.get("claim_linkage_scope")
    has_reconciliation_fields = any(
        key in metadata
        for key in (
            "claim_linkage_policy",
            "claim_linkage_roles",
            "linked_claim_ids",
            "claim_linkage_scope",
        )
    )
    if status in {None, "pending_ma1_claim_reconciliation"}:
        return not has_reconciliation_fields and current == source
    if (
        status != "reconciled"
        or policy not in _RECONCILIATION_POLICIES
        or not isinstance(raw_roles, (list, tuple))
        or not raw_roles
        or not isinstance(raw_links, (list, tuple))
    ):
        return False
    roles = {str(role) for role in raw_roles}
    known_roles = (
        _SUPPORT_LINKAGE_ROLES
        | _COUNTER_LINKAGE_ROLES
        | _CONTEXT_LINKAGE_ROLES
    )
    if not roles <= known_roles:
        return False
    support_linked = bool(roles & _SUPPORT_LINKAGE_ROLES)
    counter_linked = bool(roles & _COUNTER_LINKAGE_ROLES)
    context_unlinked = bool(roles & _CONTEXT_LINKAGE_ROLES)
    if sum((support_linked, counter_linked, context_unlinked)) != 1:
        return False
    expected = "counter" if counter_linked else "support" if support_linked else "context"
    links = [str(link) for link in raw_links if str(link)]
    if expected in {"support", "counter"}:
        return (
            source == expected
            and current == expected
            and bool(links)
            and scope == "claim_specific"
        )
    return current == "context" and not links and scope == "unlinked_context"


def _trust_boundaries_compatible(left: EvidenceRef, right: EvidenceRef) -> bool:
    left_class = _provenance_class(left)
    right_class = _provenance_class(right)

    if _is_fixture_boundary(left) != _is_fixture_boundary(right):
        return False
    if left_class == right_class:
        if left_class == "external_live":
            return _valid_live_record(left) and _valid_live_record(right)
        if left_class == "external_cache":
            return _valid_live_origin_cache(left) and _valid_live_origin_cache(right)
        return True

    if {left_class, right_class} != {"external_live", "external_cache"}:
        return False
    live_record = left if left_class == "external_live" else right
    cache_record = left if left_class == "external_cache" else right
    return (
        _valid_live_record(live_record)
        and _valid_live_origin_cache(cache_record)
        and str(cache_record.metadata.get("origin_content_hash") or "")
        == _source_content_hash(live_record)
    )


def _provenance_class(record: EvidenceRef) -> str:
    return str(record.metadata.get("provenance_class") or "").strip().casefold()


def _is_fixture_boundary(record: EvidenceRef) -> bool:
    metadata = record.metadata
    provenance = _provenance_class(record)
    backend_kind = str(metadata.get("backend_kind") or "").strip().casefold()
    return (
        metadata.get("fixture") is True
        or metadata.get("mock") is True
        or metadata.get("demo_fixture") is True
        or any(marker in provenance for marker in _FIXTURE_PROVENANCE_MARKERS)
        or any(marker in backend_kind for marker in _FIXTURE_PROVENANCE_MARKERS)
    )


def _valid_live_record(record: EvidenceRef) -> bool:
    metadata = record.metadata
    origin_class = metadata.get("origin_provenance_class")
    origin_hash = metadata.get("origin_content_hash")
    return (
        _provenance_class(record) == "external_live"
        and metadata.get("live") is not False
        and metadata.get("cache_hit") is not True
        and (
            origin_class is None
            or str(origin_class).strip().casefold() == "external_live"
        )
        and (
            origin_hash is None
            or str(origin_hash) == _source_content_hash(record)
        )
        and not _is_fixture_boundary(record)
    )


def _valid_live_origin_cache(record: EvidenceRef) -> bool:
    metadata = record.metadata
    origin_hash = str(metadata.get("origin_content_hash") or "")
    return (
        _provenance_class(record) == "external_cache"
        and metadata.get("cache_hit") is True
        and metadata.get("retrieved_live") is True
        and metadata.get("live") is not True
        and str(metadata.get("origin_provenance_class") or "")
        .strip()
        .casefold()
        == "external_live"
        and bool(origin_hash)
        and origin_hash == _source_content_hash(record)
        and not _is_fixture_boundary(record)
    )


def _source_content_hash(record: EvidenceRef) -> str:
    declared = record.metadata.get("content_hash") or record.excerpt_hash
    if declared:
        return str(declared)
    payload = json.dumps(
        _immutable_evidence_payload(record),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _trust_boundary_payload(record: EvidenceRef) -> dict[str, object]:
    return {
        key: record.metadata[key]
        for key in sorted(_TRUST_BOUNDARY_METADATA)
        if key in record.metadata
    }



def _immutable_evidence_hash(record: EvidenceRef) -> str:
    payload = json.dumps(
        {
            "scientific_record": _immutable_evidence_payload(record),
            "trust_boundary": _trust_boundary_payload(record),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _effective_work_track(bundle: TargetContextBundle) -> HypothesisTrack:
    return bundle.target.effective_work_track or bundle.target.track


def _quarantine(
    quarantined_records: list[dict[str, str]],
    reason_counts: Counter[str],
    hypothesis: Hypothesis,
    record: EvidenceRef,
    reason: str,
    *,
    existing: EvidenceRef | None = None,
) -> None:
    reason_counts[reason] += 1
    disposition = (
        "incoming_conflicting_record_rejected_existing_record_preserved"
        if reason == "duplicate_id_conflict"
        else "later_preexisting_instance_removed_first_record_preserved"
        if reason == "preexisting_duplicate_id_conflict"
        else "incoming_record_rejected"
    )
    item = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "evidence_id": record.evidence_id,
        "source": record.source,
        "reason": reason,
        "disposition": disposition,
    }
    if existing is not None:
        item.update(
            {
                "existing_content_hash": _immutable_evidence_hash(existing),
                "incoming_content_hash": _immutable_evidence_hash(record),
            }
        )
    quarantined_records.append(item)


def _claim_link_ids(hypothesis: Hypothesis) -> set[str]:
    return {
        evidence_id
        for claim in hypothesis.claims
        for evidence_id in (*claim.evidence_ids, *claim.counter_evidence_ids)
    }


def _claim_link_snapshot(
    packet: HypothesisPacket,
) -> tuple[tuple[str, tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]], ...]:
    return tuple(
        (
            hypothesis.hypothesis_id,
            tuple(
                (
                    claim.claim_id,
                    tuple(claim.evidence_ids),
                    tuple(claim.counter_evidence_ids),
                )
                for claim in hypothesis.claims
            ),
        )
        for hypothesis in packet.hypotheses
    )


def _hypothesis_key(hypothesis: Hypothesis) -> tuple[str, str]:
    return hypothesis.track.value, _normalized(hypothesis.target.symbol)


def _bundle_key(bundle: TargetContextBundle) -> tuple[str, str]:
    return bundle.target.track.value, _normalized(bundle.target.symbol)


def _normalized(value: str) -> str:
    return value.strip().casefold()


def _sorted_counts(values: Counter[str]) -> dict[str, int]:
    return dict(sorted(values.items()))


def _routing_status(
    routing: ContextRoutingResult,
    *,
    status_counts: Counter[str],
    quarantined_count: int,
) -> str:
    if routing.budget_exhausted:
        return "budget_exhausted"
    if status_counts["error"] or status_counts["policy_rejected"]:
        return "completed_with_source_issues"
    if quarantined_count:
        return "completed_with_quarantine"
    return "completed"


__all__ = [
    "CONTEXT_ROUTING_METADATA_KEY",
    "ContextTargetMatchError",
    "MA1ContextEnrichmentError",
    "enrich_hypothesis_packet_with_context",
]
