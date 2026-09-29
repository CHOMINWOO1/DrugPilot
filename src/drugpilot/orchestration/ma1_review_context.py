"""Context-only adapters for unapproved MA1 submissions.

Legacy source routers consume ``hypotheses``. Their temporary view must never
escape as an approved packet, nor change a submitted scientific claim.
"""
from __future__ import annotations

from drugpilot.domain.models import ExcludedHypothesis, Hypothesis, HypothesisPacket
from drugpilot.orchestration.goal_focus import GoalFocus, apply_goal_focus

POLICY = "ma3-source-bound-v1"


def review_hypotheses(packet: HypothesisPacket) -> list[Hypothesis]:
    return [*packet.hypotheses, *packet.provisional_hypotheses]


def needs_review_context_view(packet: HypothesisPacket) -> bool:
    return packet.metadata.get("admission_policy") == POLICY and bool(packet.provisional_hypotheses)


def review_context_view(packet: HypothesisPacket) -> HypothesisPacket:
    if not needs_review_context_view(packet):
        raise ValueError("context view requires an explicit MA3 submission policy")
    # Pydantic's model_copy(deep=True, update=...) does not deep-copy values
    # supplied in update. Build the temporary view from the already copied pool.
    view = packet.model_copy(deep=True)
    view.hypotheses = review_hypotheses(view)
    view.provisional_hypotheses = []
    view.metadata["context_view_only_not_scientific_approval"] = True
    return view


def restore_review_context(original: HypothesisPacket, enriched: HypothesisPacket) -> HypothesisPacket:
    original_items = review_hypotheses(original)
    original_ids = [item.hypothesis_id for item in original_items]
    enriched_ids = [item.hypothesis_id for item in enriched.hypotheses]
    if (len(enriched_ids) != len(set(enriched_ids))
            or set(original_ids) != set(enriched_ids) or enriched.provisional_hypotheses):
        raise ValueError("context enrichment changed review submission identities")
    by_id = {item.hypothesis_id: item for item in enriched.hypotheses}
    for before in original_items:
        after = by_id[before.hypothesis_id]
        for field in ("claims", "intervention_direction", "mechanism", "falsification_test", "disease"):
            if getattr(before, field) != getattr(after, field):
                raise ValueError("context enrichment changed a scientific claim")
        for key in ("ma1_admission", "direction_resolution", "molecular_hypothesis_scope"):
            if before.target.metadata.get(key) != after.target.metadata.get(key):
                raise ValueError("context enrichment changed scientific admission authority")
        for field in ("symbol", "ensembl_id", "uniprot_id"):
            if getattr(before.target, field) != getattr(after.target, field):
                raise ValueError("context enrichment changed canonical target identity")
        old_evidence = {item.evidence_id: item for item in before.evidence}
        new_evidence = {item.evidence_id: item for item in after.evidence}
        if not set(old_evidence).issubset(new_evidence):
            raise ValueError("context enrichment removed original source evidence")
        for evidence_id, evidence in old_evidence.items():
            added = new_evidence[evidence_id]
            if (evidence.source, evidence.external_id, evidence.excerpt_hash, evidence.supports,
                evidence.metadata.get("excerpt")) != (
                    added.source, added.external_id, added.excerpt_hash, added.supports, added.metadata.get("excerpt")):
                raise ValueError("context enrichment altered original source identity")
        if any(item.supports != "context" for key, item in new_evidence.items() if key not in old_evidence):
            raise ValueError("context enrichment promoted an added record to primary support")
    metadata = dict(enriched.metadata)
    metadata.pop("context_view_only_not_scientific_approval", None)
    metadata["context_enrichment_submission_scope"] = "review_only_not_executable"
    return enriched.model_copy(update={
        "hypotheses": [by_id[item.hypothesis_id] for item in original.hypotheses],
        "provisional_hypotheses": [by_id[item.hypothesis_id] for item in original.provisional_hypotheses],
        "metadata": metadata,
    }, deep=True)


def apply_ma1_review_goal_focus(packet: HypothesisPacket, focus: GoalFocus):
    """Preserve literal user scope without requiring pre-MA3 scientific support."""
    if packet.metadata.get("admission_policy") != POLICY:
        return apply_goal_focus(packet, focus)
    enriched, updated = apply_goal_focus(packet, focus)
    scope = {item.casefold() for item in focus.executable_target_mentions}
    pending = [item for item in packet.provisional_hypotheses
               if focus.target_scope_mode != "explicit_only" or item.target.symbol.casefold() in scope]
    pending_ids = {item.hypothesis_id for item in pending}
    for item in packet.provisional_hypotheses:
        if item.hypothesis_id not in pending_ids:
            enriched.excluded.append(ExcludedHypothesis(target_symbol=item.target.symbol,
                reason_codes=["OUTSIDE_EXECUTABLE_TARGET_SCOPE"], source_iteration=max(1, item.revision + 1),
                hypothesis_id=item.hypothesis_id))
    symbols = {item.target.symbol.casefold() for item in pending}
    rejected = [{**item, "reason": "source-bound proposal awaiting MA3 scientific review"}
                if item["symbol"].casefold() in symbols else item
                for item in updated.rejected_target_mentions]
    updated = updated.model_copy(update={"rejected_target_mentions": rejected}, deep=True)
    sources = packet.metadata.get("target_sources") or {}
    enriched.provisional_hypotheses = [item.model_copy(deep=True) for item in pending]
    enriched.metadata.update({
        "goal_focus": updated.model_dump(mode="json"),
        "goal_focus_provisional_symbols": sorted(symbols),
        "goal_focus_provisional_semantics": "literal_scope_only_scientific_approval_pending",
        "ma1_provisional_hypothesis_count": len(pending),
        "target_sources_scope": "provisional_review_only",
        "target_sources": {key: value for key, value in sources.items()
                           if str(key).casefold() in symbols},
    })
    return enriched, updated
