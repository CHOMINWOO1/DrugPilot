"""Deterministic, opt-in Pharos routing for MA1 target context.

Pharos is deliberately excluded from the default interactive MA1 fan-out.  A
human can request target-development-level or druggability context through the
``ResearchBrief.human_priorities`` field; this router then selects a bounded
set of already-ranked MA1 targets.  The decision is made by orchestration
policy, never by an LLM.

Returned Pharos records remain context-only.  They are appended to hypothesis
evidence for traceability but are never linked to causal or counter-evidence
claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Protocol, Sequence

from drugpilot.agents.ma1 import TargetSeed
from drugpilot.domain.models import EvidenceRef, Hypothesis, HypothesisPacket, ResearchBrief
from drugpilot.domain.goal_intent import has_affirmative_term


_PHAROS_PRIORITY_TERMS: tuple[str, ...] = (
    "pharos",
    "tcrd",
    "tdl",
    "target development level",
    "target-development level",
    "target druggability",
    "druggability",
    "파로스",
    "표적 개발 수준",
    "표적개발수준",
    "타깃 개발 수준",
    "타깃개발수준",
    "약물화 가능성",
    "약물화가능성",
    "드러거빌리티",
)

_CANONICAL_TDL: dict[str, str] = {
    "tclin": "Tclin",
    "tchem": "Tchem",
    "tbio": "Tbio",
    "tdark": "Tdark",
}
_TDL_PROVENANCE_PRIORITY: dict[str, int] = {
    "external_live": 300,
    "external_cache": 200,
    "persisted_verified_metadata": 100,
}
_TDL_RECONCILIATION_POLICY = "pharos-effective-tdl-provenance-v1"
_TRACK_TDL_CONTRACT_POLICY = "separate-work-track-and-scientific-tdl-v1"


def _work_track_for_tdl(
    hypothesis: Hypothesis,
    effective_tdl: str | None,
) -> tuple[str, str]:
    """Resolve work allocation without redefining the immutable MA1 track.

    ``Hypothesis.track`` records how MA1 supplied the hypothesis.  It is not a
    scientific target-development-level assertion.  Once an authoritative TDL
    is available, context work is allocated using that TDL while the original
    hypothesis track remains intact for provenance.
    """

    if effective_tdl in {"Tclin", "Tchem"}:
        return "literature", "effective_tdl"
    if effective_tdl in {"Tbio", "Tdark"}:
        return "tbio", "effective_tdl"
    return hypothesis.track.value, "hypothesis_track_fallback"


def _canonical_tdl(value: object) -> str | None:
    return _CANONICAL_TDL.get(str(value or "").strip().casefold())


def _tdl_source_claims(hypothesis: Hypothesis) -> list[dict[str, object]]:
    """Collect exact-target TDL claims while preserving their source identity."""

    claims: list[dict[str, object]] = []
    target = hypothesis.target
    symbol_key = target.symbol.casefold()
    for record in hypothesis.evidence:
        if (
            record.source.casefold() != "pharos"
            or record.supports != "context"
            or record.external_id.casefold() != symbol_key
        ):
            continue
        value = _canonical_tdl(record.metadata.get("tdl"))
        provenance_class = str(
            record.metadata.get("provenance_class") or ""
        ).strip()
        priority = _TDL_PROVENANCE_PRIORITY.get(provenance_class)
        if value is None or priority is None:
            continue
        claims.append(
            {
                "source": "Pharos",
                "value": value,
                "evidence_id": record.evidence_id,
                "external_id": record.external_id,
                "provenance_class": provenance_class,
                "retrieved_at": record.retrieved_at.isoformat(),
                "priority": priority,
                "eligible_for_effective_tdl": True,
            }
        )

    metadata_source = str(target.metadata.get("tdl_source") or "").strip()
    metadata_value = _canonical_tdl(target.tdl)
    if (
        not claims
        and metadata_value is not None
        and metadata_source.casefold()
        in {"pharos", "pharos api", "target central resource database"}
    ):
        claims.append(
            {
                "source": metadata_source,
                "value": metadata_value,
                "evidence_id": None,
                "external_id": target.symbol,
                "provenance_class": "persisted_verified_metadata",
                "retrieved_at": None,
                "priority": _TDL_PROVENANCE_PRIORITY[
                    "persisted_verified_metadata"
                ],
                "eligible_for_effective_tdl": True,
            }
        )

    claims.sort(
        key=lambda item: (
            -int(item["priority"]),
            str(item["source"]).casefold(),
            str(item["value"]).casefold(),
            str(item.get("evidence_id") or ""),
        )
    )
    return claims


def _reconcile_target_development_level(
    hypothesis: Hypothesis,
    *,
    record_conflicts: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """Resolve one effective TDL without discarding conflicting source claims."""

    target = hypothesis.target
    prior_target_tdl = target.tdl
    prior_tdl_source = target.metadata.get("tdl_source")
    existing_supply_category = target.metadata.get("ma1_supply_category")
    existing_supply = (
        dict(existing_supply_category)
        if isinstance(existing_supply_category, dict)
        else {}
    )
    # Migrate the old ambiguous ``legacy_target_tdl`` field into an explicitly
    # non-scientific namespace.  This preserves audit history without inviting
    # a reviewer or downstream router to compare it with an authoritative TDL.
    legacy_track_label = (
        existing_supply.get("legacy_track_derived_label")
        or existing_supply.get("legacy_target_tdl")
        or prior_target_tdl
    )
    legacy_track_source = (
        existing_supply.get("legacy_track_derived_label_source")
        or existing_supply.get("legacy_tdl_source")
        or prior_tdl_source
    )
    supply_category = {
        "hypothesis_track": hypothesis.track.value,
        "semantics": "original_ma1_evidence_supply_track",
        "legacy_track_derived_label": legacy_track_label,
        "legacy_track_derived_label_source": legacy_track_source,
        "legacy_track_derived_label_is_scientific_tdl": False,
    }
    claims = _tdl_source_claims(hypothesis)
    distinct_values = sorted({str(item["value"]) for item in claims})
    conflicting_values = sorted(
        {
            str(value)
            for item in record_conflicts
            for value in (item.get("existing_tdl"), item.get("incoming_tdl"))
            if _canonical_tdl(value) is not None
        }
    )
    source_values = sorted(set(distinct_values) | set(conflicting_values))
    record_integrity_conflict = bool(record_conflicts)
    conflict_detected = len(source_values) > 1 or record_integrity_conflict

    effective_tdl: str | None = None
    effective_source: str | None = None
    effective_evidence_ids: list[str] = []
    if record_integrity_conflict:
        # A stable evidence ID mapped to changed immutable source content.  Do
        # not choose either version as an authoritative scientific TDL until
        # the collision is resolved, even when both happen to state the same
        # TDL.  Both projections remain in the audit contract below.
        status = "unresolved_record_integrity_conflict"
    elif claims:
        highest_priority = max(int(item["priority"]) for item in claims)
        preferred = [
            item for item in claims if int(item["priority"]) == highest_priority
        ]
        preferred_values = {str(item["value"]) for item in preferred}
        if len(preferred_values) == 1:
            effective_tdl = next(iter(preferred_values))
            effective_sources = sorted({str(item["source"]) for item in preferred})
            effective_source = " + ".join(effective_sources)
            effective_evidence_ids = sorted(
                {
                    str(item["evidence_id"])
                    for item in preferred
                    if item.get("evidence_id")
                }
            )
            status = (
                "resolved_with_source_conflict"
                if conflict_detected
                else "resolved"
            )
        else:
            status = "unresolved_same_priority_conflict"
    else:
        status = "unavailable"

    effective_work_track, work_track_basis = _work_track_for_tdl(
        hypothesis,
        effective_tdl,
    )
    track_tdl_contract = {
        "policy": _TRACK_TDL_CONTRACT_POLICY,
        "hypothesis_track": hypothesis.track.value,
        "hypothesis_track_semantics": "original_ma1_evidence_supply_track",
        "effective_work_track": effective_work_track,
        "effective_work_track_basis": work_track_basis,
        "effective_tdl": effective_tdl,
        "effective_tdl_semantics": (
            "authoritative_scientific_target_development_level"
        ),
        "namespaces_distinct": True,
        "context_source_plan_uses": "effective_work_track",
        "ma2_candidate_route_uses": "effective_tdl",
    }

    target.tdl = effective_tdl
    metadata = {
        **target.metadata,
        "ma1_supply_category": supply_category,
        "track_tdl_contract": track_tdl_contract,
        "effective_tdl": effective_tdl,
        "effective_tdl_source": effective_source,
        "tdl_source_claims": claims,
        "tdl_reconciliation": {
            "policy": _TDL_RECONCILIATION_POLICY,
            "status": status,
            "effective_tdl": effective_tdl,
            "effective_tdl_source": effective_source,
            "effective_evidence_ids": effective_evidence_ids,
            "source_claim_count": len(claims),
            "source_values": source_values,
            "conflict_detected": conflict_detected,
            "conflict_hidden": False,
            "record_integrity_conflict_count": len(record_conflicts),
            "record_integrity_conflicts": [dict(item) for item in record_conflicts],
            "track_tdl_contract_policy": _TRACK_TDL_CONTRACT_POLICY,
            "precedence": [
                "Pharos external_live",
                "Pharos external_cache",
                "persisted verified Pharos/TCRD metadata",
            ],
        },
    }
    if effective_source is None:
        metadata.pop("tdl_source", None)
    else:
        metadata["tdl_source"] = effective_source
    target.metadata = metadata
    return dict(metadata["tdl_reconciliation"])


class PharosEvidenceRetriever(Protocol):
    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]: ...


@dataclass(frozen=True)
class PharosTargetSelection:
    hypothesis_id: str
    symbol: str
    track: str
    rank: int | None
    score: float


@dataclass(frozen=True)
class PharosRoutingResult:
    packet: HypothesisPacket
    decision: dict[str, object]


def requested_pharos_terms(priorities: Sequence[str]) -> tuple[str, ...]:
    """Return deterministic trigger terms found in explicit human priorities."""

    normalized = "\n".join(str(value).casefold() for value in priorities if value)
    return tuple(term for term in _PHAROS_PRIORITY_TERMS if has_affirmative_term(normalized, (term,)))


class OnDemandPharosRouter:
    """Select, query and safely merge bounded Pharos target context."""

    def __init__(
        self,
        retriever: PharosEvidenceRetriever | None,
        *,
        max_targets_per_track: int = 1,
    ) -> None:
        if not 0 <= max_targets_per_track <= 3:
            raise ValueError("max_targets_per_track must be between 0 and 3")
        self.retriever = retriever
        self.max_targets_per_track = max_targets_per_track

    def route(
        self,
        packet: HypothesisPacket,
        brief: ResearchBrief,
    ) -> PharosRoutingResult:
        triggers = requested_pharos_terms(brief.human_priorities)
        required = brief.constraints.druggability_required
        request_fields = (["ResearchBrief.constraints.druggability_required"] if required else []) + (
            ["ResearchBrief.human_priorities"] if triggers else [])
        base_decision: dict[str, object] = {
            "policy_owner": "Orchestrator",
            "request_field": " + ".join(request_fields) or "ResearchBrief.human_priorities",
            "request_fields": request_fields,
            "requested": required or bool(triggers),
            "structured_druggability_required": required,
            "matched_terms": list(triggers),
            "selection_policy": "rank_then_score; bounded_per_track; unique_symbol",
            "max_targets_per_track": self.max_targets_per_track,
            "context_only": True,
            "claim_links_modified": False,
            "selected_targets": [],
            "outcomes": [],
            "records_added": 0,
            "records_reused": 0,
            "records_conflicted": 0,
        }

        if not (required or triggers):
            return self._skip(
                packet,
                base_decision,
                "not requested: no structured druggability requirement or Pharos/TDL/druggability priority",
            )
        if "Pharos" not in brief.data_policy.allowed_sources:
            return self._skip(
                packet,
                base_decision,
                "not allowed: ResearchBrief data policy excludes Pharos",
            )
        if self.max_targets_per_track == 0:
            return self._skip(
                packet,
                base_decision,
                "disabled: on-demand Pharos target limit is zero",
            )
        if self.retriever is None:
            return self._skip(
                packet,
                base_decision,
                "unavailable: public evidence adapters are disabled",
            )

        selected = self._select(packet.hypotheses)
        base_decision["selected_targets"] = [
            {
                "hypothesis_id": item.hypothesis_id,
                "symbol": item.symbol,
                "track": item.track,
                "rank": item.rank,
                "score": item.score,
            }
            for item in selected
        ]
        if not selected:
            return self._skip(
                packet,
                base_decision,
                "not queried: MA1 produced no eligible ranked target",
            )

        enriched = packet.model_copy(deep=True)
        claim_links_before = _claim_links(packet)
        hypothesis_by_id = {
            item.hypothesis_id: item for item in enriched.hypotheses
        }
        outcomes: list[dict[str, object]] = []
        records_added = 0
        records_reused = 0
        records_conflicted = 0

        for selection in selected:
            hypothesis = hypothesis_by_id[selection.hypothesis_id]
            seed = TargetSeed(
                symbol=hypothesis.target.symbol,
                target_name=hypothesis.target.target_name or hypothesis.target.symbol,
                track=hypothesis.track,
                priority=hypothesis.score,
                ensembl_id=hypothesis.target.ensembl_id,
                uniprot_id=hypothesis.target.uniprot_id,
                identity_status=hypothesis.target.identity_status,
                identity_sources=tuple(hypothesis.target.identity_sources),
                identity_metadata=dict(hypothesis.target.metadata),
                discovery_source="ma1_ranked_hypothesis",
            )
            try:
                observed = tuple(
                    self.retriever.retrieve(brief, seed, packet.iterations)
                )
            except Exception as exc:  # isolate one target and preserve the packet
                failures = getattr(self.retriever, "last_failures", {})
                empty_result = bool(failures) and all(
                    str(reason) == "empty_result"
                    for reason in dict(failures).values()
                )
                outcome = {
                    "hypothesis_id": selection.hypothesis_id,
                    "symbol": selection.symbol,
                    "track": selection.track,
                    "queried": True,
                    "status": "empty" if empty_result else "error",
                    "reason": (
                        "Pharos returned no exact target record"
                        if empty_result
                        else f"Pharos query failed: {type(exc).__name__}"
                    ),
                    "records_observed": 0,
                    "records_added": 0,
                    "records_reused": 0,
                    "records_conflicted": 0,
                    "evidence_ids": [],
                    "reused_evidence_ids": [],
                    "conflicting_evidence_ids": [],
                    "provenance_classes": [],
                    "observed_provenance_classes": [],
                    "source_failures": dict(failures),
                }
                outcomes.append(outcome)
                resolution = _reconcile_target_development_level(hypothesis)
                outcome.update(
                    {
                        "effective_tdl": resolution["effective_tdl"],
                        "tdl_resolution_status": resolution["status"],
                        "tdl_conflict_detected": resolution["conflict_detected"],
                    }
                )
                self._attach_target_outcome(hypothesis, outcome)
                continue

            safe_records = [
                item
                for item in observed
                if item.source == "Pharos"
                and item.supports == "context"
                and item.external_id.casefold() == selection.symbol.casefold()
                and item.metadata.get("provenance_class")
                in {"external_live", "external_cache"}
            ]
            existing_by_id = {
                item.evidence_id: item for item in hypothesis.evidence
            }
            added: list[EvidenceRef] = []
            reused: list[EvidenceRef] = []
            conflicts: list[dict[str, object]] = []
            for record in safe_records:
                copied = record.model_copy(
                    update={
                        "metadata": {
                            **record.metadata,
                            "context_only": True,
                            "causal_inference_allowed": False,
                            "counter_evidence_allowed": False,
                            "routing_policy": "explicit_human_priority",
                        }
                    },
                    deep=True,
                )
                existing = existing_by_id.get(copied.evidence_id)
                if existing is not None:
                    if _same_immutable_evidence(existing, copied):
                        reused.append(existing)
                    else:
                        conflicts.append(
                            {
                                "evidence_id": copied.evidence_id,
                                "source": copied.source,
                                "reason": "duplicate_id_conflict",
                                "existing_content_hash": _immutable_evidence_hash(
                                    existing
                                ),
                                "incoming_content_hash": _immutable_evidence_hash(
                                    copied
                                ),
                                "existing_tdl": _canonical_tdl(
                                    existing.metadata.get("tdl")
                                ),
                                "incoming_tdl": _canonical_tdl(
                                    copied.metadata.get("tdl")
                                ),
                                "existing_provenance_class": existing.metadata.get(
                                    "provenance_class"
                                ),
                                "incoming_provenance_class": copied.metadata.get(
                                    "provenance_class"
                                ),
                                "disposition": (
                                    "incoming_record_quarantined_existing_record_preserved"
                                ),
                            }
                        )
                    continue
                hypothesis.evidence.append(copied)
                existing_by_id[copied.evidence_id] = copied
                added.append(copied)
            records_added += len(added)
            records_reused += len(reused)
            records_conflicted += len(conflicts)
            accepted = list(
                {
                    item.evidence_id: item
                    for item in (*added, *reused)
                }.values()
            )
            provenance_classes = sorted(
                {
                    str(item.metadata.get("provenance_class"))
                    for item in accepted
                    if item.metadata.get("provenance_class")
                }
            )
            observed_provenance_classes = sorted(
                {
                    str(item.metadata.get("provenance_class"))
                    for item in safe_records
                    if item.metadata.get("provenance_class")
                }
            )
            if conflicts and accepted:
                outcome_status = "completed_with_conflict"
                outcome_reason = (
                    "exact-symbol context retained; conflicting duplicate ID rejected"
                )
            elif conflicts:
                outcome_status = "conflict"
                outcome_reason = "conflicting duplicate Pharos evidence ID rejected"
            elif added:
                outcome_status = "success"
                outcome_reason = "exact-symbol context records added"
            elif reused:
                outcome_status = "reused_existing"
                outcome_reason = (
                    "exact-symbol context already present and reused without mutation"
                )
            else:
                outcome_status = "empty"
                outcome_reason = "no safe exact-symbol Pharos context record returned"
            outcome = {
                "hypothesis_id": selection.hypothesis_id,
                "symbol": selection.symbol,
                "track": selection.track,
                "queried": True,
                "status": outcome_status,
                "reason": outcome_reason,
                "records_observed": len(observed),
                "records_added": len(added),
                "records_reused": len(reused),
                "records_conflicted": len(conflicts),
                "evidence_ids": [item.evidence_id for item in accepted],
                "reused_evidence_ids": list(
                    dict.fromkeys(item.evidence_id for item in reused)
                ),
                "conflicting_evidence_ids": list(
                    dict.fromkeys(
                        str(item["evidence_id"]) for item in conflicts
                    )
                ),
                "conflicts": conflicts,
                "provenance_classes": provenance_classes,
                "observed_provenance_classes": observed_provenance_classes,
            }
            outcomes.append(outcome)
            resolution = _reconcile_target_development_level(
                hypothesis,
                record_conflicts=conflicts,
            )
            outcome.update(
                {
                    "effective_tdl": resolution["effective_tdl"],
                    "tdl_resolution_status": resolution["status"],
                    "tdl_conflict_detected": resolution["conflict_detected"],
                }
            )
            self._attach_target_outcome(hypothesis, outcome)

        if _claim_links(enriched) != claim_links_before:
            raise RuntimeError("Pharos context routing must not modify claim links")

        base_decision["outcomes"] = outcomes
        base_decision["records_added"] = records_added
        base_decision["records_reused"] = records_reused
        base_decision["records_conflicted"] = records_conflicted
        statuses = {str(item["status"]) for item in outcomes}
        completed_statuses = {"success", "reused_existing"}
        base_decision["status"] = (
            "completed"
            if statuses and statuses <= completed_statuses
            else "failed"
            if statuses and statuses <= {"error", "conflict"}
            else "partial"
        )
        base_decision["reason"] = (
            "configured request fields selected bounded Pharos queries: "
            + ", ".join(base_decision["request_fields"])
        )
        enriched.metadata = {
            **enriched.metadata,
            "pharos_routing": base_decision,
        }
        return PharosRoutingResult(packet=enriched, decision=base_decision)

    def _select(
        self,
        hypotheses: Sequence[Hypothesis],
    ) -> tuple[PharosTargetSelection, ...]:
        ranked = sorted(
            hypotheses,
            key=lambda item: (
                item.track.value,
                item.rank if item.rank is not None else 10_000,
                -item.score,
                item.target.symbol.casefold(),
                item.hypothesis_id,
            ),
        )
        counts: dict[str, int] = {}
        seen_symbols: set[str] = set()
        selected: list[PharosTargetSelection] = []
        for hypothesis in ranked:
            track = hypothesis.track.value
            symbol_key = hypothesis.target.symbol.casefold()
            if counts.get(track, 0) >= self.max_targets_per_track:
                continue
            if not symbol_key or symbol_key in seen_symbols:
                continue
            counts[track] = counts.get(track, 0) + 1
            seen_symbols.add(symbol_key)
            selected.append(
                PharosTargetSelection(
                    hypothesis_id=hypothesis.hypothesis_id,
                    symbol=hypothesis.target.symbol,
                    track=track,
                    rank=hypothesis.rank,
                    score=hypothesis.score,
                )
            )
        return tuple(selected)

    @staticmethod
    def _attach_target_outcome(
        hypothesis: Hypothesis,
        outcome: dict[str, object],
    ) -> None:
        hypothesis.target.metadata = {
            **hypothesis.target.metadata,
            "pharos_source_outcome": dict(outcome),
        }

    @staticmethod
    def _skip(
        packet: HypothesisPacket,
        decision: dict[str, object],
        reason: str,
    ) -> PharosRoutingResult:
        skipped = packet.model_copy(deep=True)
        decision = {
            **decision,
            "status": "skipped",
            "reason": reason,
        }
        skipped.metadata = {
            **skipped.metadata,
            "pharos_routing": decision,
        }
        return PharosRoutingResult(packet=skipped, decision=decision)


def _claim_links(packet: HypothesisPacket) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (
            claim.claim_id,
            tuple(claim.evidence_ids),
            tuple(claim.counter_evidence_ids),
        )
        for hypothesis in packet.hypotheses
        for claim in hypothesis.claims
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
    # Derived MA1 claim-eligibility annotations are invocation state, not
    # Pharos source content.  Ignoring them here makes a rework idempotent
    # without relaxing source identity, TDL, provenance, or fixture checks.
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
