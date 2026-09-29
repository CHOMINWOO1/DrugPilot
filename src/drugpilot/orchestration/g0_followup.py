"""Cause-bound G0 follow-up plans. Plans never decide scientific verdicts.

Only existing packet identities and typed findings can select work. Reviewer
prose is not parsed into targets, URLs, queries, measurements or permission.
"""
from __future__ import annotations

from hashlib import sha256
from dataclasses import replace
import json

from drugpilot.domain.models import GateVerdictCode, RunMode
from drugpilot.agents.ma1 import _is_evidence_retry_reason_set
from drugpilot.agents.molecular_scope import SEARCH_KEY, fusion_search_bindings, validated_fusion_search_terms
from drugpilot.agents.variant_search_scope import KEY as VARIANT_KEY, from_screening_entry as variant_search_bindings, validate as validate_variant_search
from drugpilot.orchestration.next_actions import _kind_for
from drugpilot.tools.evidence_gap import diagnose_observation, transient_failure

POLICY = "g0_cause_bound_followup_v2"
SOURCE_ELAPSED_SECONDS = 120
ROLE_REASONS = {
    "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION": "association",
    "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY": "direction",
}


def public_path_retry(history, *, source_task_key, target, role, source):
    """One additional read-only retrieval after an exact bound transient failure.

    Existing task keys are deliberately unchanged: deploying this policy cannot
    make a previously completed query appear unattempted. Legacy/ambiguous
    history grants no retry; paid-model failures never enter this path.
    """
    attempts = {}
    for entry in history:
        if not isinstance(entry, dict) or not isinstance(entry.get("plan_id"), str):
            continue
        keys = entry.get("attempted_source_task_keys")
        if not isinstance(keys, list) or source_task_key not in keys:
            continue
        rows = entry.get("observations", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if (not isinstance(row, dict) or row.get("attempted") is not True
                    or row.get("plan_id") != entry["plan_id"] or row.get("target") != target
                    or row.get("source") != source
                    or row.get("role") not in ({role, "oa_context"} if source == "Europe PMC" else {role})
                    or not isinstance(row.get("source_task_keys"), list)
                    or source_task_key not in row.get("source_task_keys", [])):
                continue
            bound = {k:v for k,v in row.items() if k != "evidence_gap"}
            previous = attempts.get(entry["plan_id"])
            if previous is not None and digest(previous) != digest(bound):
                return {"allowed":False, "attempts":None, "reason":"CONFLICTING_SOURCE_ATTEMPT_HISTORY"}
            attempts[entry["plan_id"]] = bound
    count = len(attempts)
    if count == 0:
        return {"allowed":False, "attempts":None, "reason":"LEGACY_OR_UNBOUND_ATTEMPT_HISTORY"}
    if count >= 2:
        return {"allowed":False, "attempts":count, "reason":"PUBLIC_PATH_RETRY_CAP_REACHED"}
    last = next(reversed(attempts.values()))
    transient = transient_failure(last)
    return {"allowed":transient, "attempts":count,
        "reason":"BOUND_TRANSIENT_PUBLIC_FAILURE" if transient else "COMPLETED_OR_NONTRANSIENT_PATH_NOT_REPLAYED"}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _ma3_followup_entries(packet):
    """Project only typed, hash-bound MA3 needs-evidence decisions into work.

    Historical MA1 diagnostics remain immutable, but cannot overrule a later
    scientific rejection/HOLD. Targets come from original submitted proposals,
    never reviewer prose. Unknown scientific gaps request the two existing
    source roles, not a new query family or permission.
    """
    from drugpilot.agents.ma1 import Critique, _rejected_screening_entry
    from drugpilot.domain.models import Hypothesis
    report = packet.metadata.get("ma3_admission_report")
    if not isinstance(report, dict):
        return [], "MA3_ADMISSION_REVIEW_NOT_COMPLETED", {}
    raw = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("policy") != "ma3-source-bound-v1" or report.get("report_sha256") != digest(raw):
        return [], "MA3_ADMISSION_REPORT_INVALID", {}
    result, dispositions, seen = [], {}, set()
    for row in report.get("entries", []):
        if not isinstance(row, dict) or row.get("status") not in {"approved", "needs_evidence", "hold", "rejected"}:
            return [], "MA3_ADMISSION_REPORT_INVALID", {}
        identifier = row.get("hypothesis_id")
        if not isinstance(identifier, str) or identifier in seen:
            return [], "MA3_ADMISSION_REPORT_INVALID", {}
        seen.add(identifier)
        dispositions[row["status"]] = dispositions.get(row["status"], 0) + 1
        if row["status"] != "needs_evidence":
            continue
        try:
            proposal = Hypothesis.model_validate(row["original_proposal"])
        except (KeyError, TypeError, ValueError):
            return [], "MA3_ADMISSION_REPORT_INVALID", {}
        if proposal.hypothesis_id != identifier or proposal.target.symbol != row.get("target_symbol"):
            return [], "MA3_ADMISSION_REPORT_INVALID", {}
        errors = set(row.get("validation_errors") or [])
        direction_errors = {
            "DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED", "DIRECTION_MISMATCH",
            "INTERVENTION_OUTSIDE_EXPERIMENT", "REQUEST_DIRECTION_NOT_ALLOWED",
        }
        roles = set()
        if errors & direction_errors:
            roles.add("direction")
        if "PRIMARY_ASSOCIATION_NOT_ESTABLISHED" in errors:
            roles.add("association")
        if "REQUIRED_PRIMARY_ROLE_MISSING" in errors:
            claims = (row.get("assessment") or {}).get("claims", [])
            supplied = {claim.get("role") for claim in claims if isinstance(claim, dict)}
            for code, role in ROLE_REASONS.items():
                required = ("direct_target_direction_primary" if role == "direction"
                            else "primary_target_disease_association")
                if required not in supplied:
                    roles.add(role)
        known = direction_errors | {"PRIMARY_ASSOCIATION_NOT_ESTABLISHED", "REQUIRED_PRIMARY_ROLE_MISSING"}
        if not roles or errors - known:
            roles = {"association", "direction"}
        reasons = tuple(code for code, role in ROLE_REASONS.items() if role in roles)
        entry = _rejected_screening_entry(
            Critique(hypothesis=proposal, accepted=False, reason_codes=reasons),
            iteration=max(1, proposal.revision + 1), screening_phase="ma3_source_bound_followup")
        entry["ma3_followup_role_authority"] = "typed_needs_evidence_only"
        entry["disposition"] = "needs_evidence"
        result.append(entry)
    stop = None if result else "MA3_ADMISSION_NO_RETRIEVABLE_EVIDENCE_GAP"
    return result, stop, dispositions


def ma3_followup_revision_context(packet, plan):
    """Build an ephemeral retry input from verified original submissions.

    The persisted MA3 packet is never edited. Exclusion/trace adapters here mean
    a typed missing-role retry request, not fabricated scientific rejection.
    """
    from copy import deepcopy
    from drugpilot.domain.models import ExcludedHypothesis
    if packet.metadata.get("admission_policy") != "ma3-source-bound-v1":
        return packet
    entries, stop, _ = _ma3_followup_entries(packet)
    if stop or not plan.get("eligible"):
        raise ValueError("MA3_FOLLOWUP_NOT_ELIGIBLE")
    selected = {task["target"].casefold() for task in plan["tasks"] if task["status"] == "scheduled"}
    entries = [entry for entry in entries if entry["target"]["symbol"].casefold() in selected]
    if not entries:
        raise ValueError("MA3_FOLLOWUP_HAS_NO_BOUND_TARGET")
    context = packet.model_copy(deep=True)
    existing = {(item.hypothesis_id, tuple(item.reason_codes)) for item in context.excluded}
    for entry in entries:
        key = (entry["hypothesis_id"], tuple(entry["reason_codes"]))
        if key not in existing:
            context.excluded.append(ExcludedHypothesis(target_symbol=entry["target"]["symbol"],
                hypothesis_id=entry["hypothesis_id"], reason_codes=entry["reason_codes"],
                source_iteration=entry["source_iteration"], molecular_scope=entry.get("molecular_hypothesis_scope")))
            existing.add(key)
    # Do not rediscover/reassess a wholly rejected or unresolvable target as a
    # fresh candidate during another target's bounded source followup.
    report = packet.metadata["ma3_admission_report"]
    statuses = {}
    for row in report["entries"]:
        statuses.setdefault(row["target_symbol"].casefold(), set()).add(row["status"])
    for symbol, values in statuses.items():
        if values.issubset({"hold", "rejected"}):
            context.excluded.append(ExcludedHypothesis(target_symbol=symbol,
                reason_codes=["MA3_SCIENTIFIC_REJECTED" if values == {"rejected"} else "MA3_ADMISSION_HOLD"],
                source_iteration=packet.iterations))
    prior = deepcopy(context.metadata.get("ma1_rejected_screening_history")
                     or context.metadata.get("rejected_screening_trace") or {})
    previous_entries = prior.get("entries", [])
    current_ids = {entry["hypothesis_id"] for entry in entries}
    combined = entries + [item for item in previous_entries if item.get("hypothesis_id") not in current_ids]
    prior.update(entries=combined[:24], entry_count_total=max(len(combined), prior.get("entry_count_total", 0)),
                 entry_count_recorded=min(24, len(combined)),
                 scope="ephemeral_ma3_typed_retry_context_original_packet_preserved")
    prior["entry_count_omitted"] = max(0, prior["entry_count_total"] - prior["entry_count_recorded"])
    context.metadata["ma1_rejected_screening_history"] = prior
    context.metadata["ma3_retry_context"] = {"source_packet_id": packet.packet_id,
        "source_packet_sha256": digest(packet.model_dump(mode="json")), "plan_id": plan["plan_id"],
        "targets": sorted(selected), "original_packet_unchanged": True,
        "target_seeds_are_query_scope_not_scientific_support": True}
    return context


def evidence_work_fingerprint(brief, packet):
    """Fingerprint source identities/content/scope, never generated claim links."""
    volatile = {"evidence_id", "claim_id", "hypothesis_id", "evidence_ids", "claim_ids",
                "hypothesis_ids", "linked_claim_ids", "supporting_evidence_ids"}
    def canonical(value):
        if isinstance(value, dict):
            return {k: canonical(v) for k, v in value.items() if k not in volatile}
        if isinstance(value, list):
            unique = {json.dumps(canonical(v), sort_keys=True): canonical(v) for v in value}
            return [unique[key] for key in sorted(unique)]
        return value
    source_fields = ("source", "external_id", "canonical_identity", "pmid", "pmcid", "doi")
    unit_fields = ("bounded_excerpt", "excerpt_hash", "content_hash", "molecular_scope")
    sources = []
    for entry in packet.metadata.get("rejected_screening_trace", {}).get("entries", []):
        sources.append({"target": (entry.get("target") or {}).get("symbol"),
            "scope": entry.get("molecular_hypothesis_scope"),
            "records": [{**{k: r.get(k) for k in source_fields},
                "units": [{k: u.get(k) for k in unit_fields} for u in r.get("evidence_units", [])]}
                for r in entry.get("unique_external_records", [])]})
    for hypothesis in packet.hypotheses:
        sources.append({"target": hypothesis.target.symbol,
            "scope": hypothesis.target.metadata.get("molecular_hypothesis_scope"),
            **({"direction": hypothesis.intervention_direction}
               if packet.metadata.get("admission_policy") != "ma3-source-bound-v1" else {}),
            "records": [{"source": e.source, "external_id": e.external_id,
                         "excerpt_hash": e.excerpt_hash,
                         "content_hash": e.metadata.get("content_hash"),
                         "bounded_excerpt": e.metadata.get("excerpt")} for e in hypothesis.evidence]})
    if packet.metadata.get("admission_policy") == "ma3-source-bound-v1":
        # Status, generated rationale and assessment changes are not new data.
        # Include actual pending/original sources even after the pending list is
        # replaced by MA3's approved subset; do not fingerprint semantic verdicts.
        originals = [h.model_dump(mode="json") for h in packet.provisional_hypotheses]
        report = packet.metadata.get("ma3_admission_report") or {}
        originals.extend(row.get("original_proposal") for row in report.get("entries", [])
                         if isinstance(row, dict) and isinstance(row.get("original_proposal"), dict))
        for hypothesis in originals:
            target = hypothesis.get("target") or {}
            sources.append({"target": target.get("symbol"),
                "scope": (target.get("metadata") or {}).get("molecular_hypothesis_scope"),
                "records": [{"source": e.get("source"), "external_id": e.get("external_id"),
                             "excerpt_hash": e.get("excerpt_hash"),
                             "content_hash": (e.get("metadata") or {}).get("content_hash"),
                             "bounded_excerpt": (e.get("metadata") or {}).get("excerpt")}
                            for e in hypothesis.get("evidence", [])]})
    return digest(canonical({"disease": brief.disease, "goal": brief.research_goal,
        "constraints": brief.constraints.model_dump(mode="json", exclude={
            "max_rework_loops", "max_ma1_iterations", "candidates_per_route"}),
        "sources": brief.data_policy.allowed_sources,
        "source_inputs": sources}))


def enabled(o):
    return (o.settings.effective_mode is RunMode.LIVE
        and o.settings.public_evidence_enabled and o.settings.automatic_resolution_enabled
        and not o.settings.demo_fixture_fallback_enabled
        and o.ma1 is getattr(o, "_configured_ma1", None)
        and o.ma1.target_discovery is getattr(o, "_configured_ma1_target_discovery", None)
        and o.ma1.identity_resolver is getattr(o, "_configured_ma1_identity_resolver", None)
        and o.ma1.proposer.retriever is getattr(o, "_configured_ma1_retriever", None))


def build_plan(brief, packet, verdict, *, rework_loops, target_limit):
    remaining = max(0, brief.constraints.max_rework_loops - rework_loops)
    findings = [f for f in verdict.findings if not f.resolved and f.severity.value != "INFO"
                and getattr(f, "confirmation_status", None) != "unconfirmed"]
    typed = [{"code": f.code, "artifact_id": f.artifact_id,
              "claim_ids": list(f.claim_ids), "kind": _kind_for(f)} for f in findings]
    manual = [f for f in typed if f["kind"] not in {"data_required", "computational_rework"}
              and f["code"] != "NO_HYPOTHESIS"]
    has_data_work = any(f["kind"] == "data_required" or f["code"] == "NO_HYPOTHESIS" for f in typed)
    has_code_work = any(f["kind"] == "computational_rework" for f in typed)
    computational_claim_ids = {
        claim_id
        for finding in typed
        if finding["kind"] == "computational_rework"
        for claim_id in finding["claim_ids"]
    }
    forbidden = {s.casefold() for s in brief.constraints.forbidden_targets}
    entries = packet.metadata.get("rejected_screening_trace", {}).get("entries", [])
    ma3_policy = packet.metadata.get("admission_policy") == "ma3-source-bound-v1"
    ma3_stop_reason, ma3_dispositions = None, {}
    if ma3_policy:
        entries, ma3_stop_reason, ma3_dispositions = _ma3_followup_entries(packet)
        # Scientific admission is not a request to repeat a model judgment.
        # Only concrete new source work for needs_evidence may resume MA1.
        has_code_work = False
    roles_by_target, priority, scope_by_target, variants_by_target = {}, {}, {}, {}
    reclassification_targets = set()
    hard = set()
    for exclusion in packet.excluded:
        if not _is_evidence_retry_reason_set(exclusion.reason_codes):
            hard.add(exclusion.target_symbol.casefold())
    # Old traces remain data, not executable injected seeds. MA1's existing
    # canonical identity/scope restoration and hard-exclusion guards still run.
    for entry in entries:
        target = entry.get("target") or {}
        symbol = target.get("symbol")
        if not isinstance(symbol, str) or symbol.casefold() in forbidden | hard:
            continue
        assignment = entry.get("claim_assignment") or {}
        entry_claim_ids = {
            claim.get("claim_id")
            for claim in assignment.get("claims", [])
            if isinstance(claim, dict) and isinstance(claim.get("claim_id"), str)
        }
        if computational_claim_ids & entry_claim_ids:
            reclassification_targets.add(symbol)
        reasons = set(entry.get("reason_codes", []))
        roles = {role for code, role in ROLE_REASONS.items() if code in reasons}
        for record in entry.get("unique_external_records", []):
            for unit in record.get("evidence_units", []):
                meaning = unit.get("semantic_interpretation") or {}
                if not ma3_policy and meaning.get("role") in {"resistance_context", "context", "unknown"}:
                    roles.add("direction")
        if roles:
            roles_by_target.setdefault(symbol, set()).update(roles)
            priority[symbol] = max(priority.get(symbol, 0), len(roles))
            # A required assay lookup is supporting context, not a new primary
            # hypothesis gate or a replacement for association/direction work.
            from drugpilot.tools.public_biology import PublicBiologyRetriever
            from drugpilot.tools.assay_followup import required_assays
            if ("PubMed" in brief.data_policy.allowed_sources and required_assays(brief)
                    and PublicBiologyRetriever._epmc_oa_assay_trace_required(brief)):
                roles_by_target[symbol].add("assay_context")
                if "ChEMBL" in brief.data_policy.allowed_sources:
                    roles_by_target[symbol].add("assay_identity")
                if "Europe PMC" in brief.data_policy.allowed_sources:
                    roles_by_target[symbol].add("assay_fulltext")
            for binding in fusion_search_bindings(brief, entry, symbol):
                scope_by_target.setdefault(symbol, {})[binding["label"]] = binding
            for binding in variant_search_bindings(brief, entry, symbol):
                variants_by_target.setdefault(symbol,{})[binding["label"]]=binding
    # Retained hypotheses may require reclassification, but cannot be removed
    # from the retained set by this planner. Their ordinary MA1 critic decides.
    history = packet.metadata.get("g0_followup_history", [])
    fingerprint = evidence_work_fingerprint(brief, packet)
    completed_actions = {key for row in history for key in row.get("automatic_action_keys", [])}
    legacy_actions_unknown = any("automatic_action_keys" not in row for row in history)
    def action_key(action):
        return digest({"action": action, "evidence_work_fingerprint": fingerprint})
    attempted = {task for row in history if "attempted_source_task_keys" not in row
                 for task in row.get("attempted_task_keys", [])}
    attempted_paths = {key for row in history for key in row.get("attempted_source_task_keys", [])}
    sources = [s for s in ("Europe PMC", "PubMed", "ChEMBL") if s in brief.data_policy.allowed_sources]
    tasks = []
    for symbol in sorted(roles_by_target, key=lambda s: (-priority[s], s)):
        from drugpilot.orchestration.evidence_carryover import ALTERNATIVE_KEY, alternative_selections
        alternatives = alternative_selections(brief, packet, symbol)
        scopes = scope_by_target.get(symbol, {})
        bindings = [scopes[label] for label in sorted(scopes)[:4]]
        variants=variants_by_target.get(symbol,{})
        variant_bindings=[variants[label] for label in sorted(variants)[:4]]
        for role in sorted(roles_by_target[symbol], key=lambda r:({"association":0,"direction":1,"assay_context":2,"assay_identity":3}.get(r,4),r)):
            key = digest({"policy": POLICY, "disease": brief.disease,
                **({"fusion_scope_labels": sorted(b["label"] for b in bindings)} if bindings else {}),
                **({"variant_scope_labels":sorted(b["label"] for b in variant_bindings)} if variant_bindings else {}),
                "constraints": brief.constraints.model_dump(mode="json"), "target": symbol, "role": role})
            paths = [{"source":source, "source_task_key":digest({"task":key, "source":source}),
                "status":"available"} for source in sources if source in (
                    {"Europe PMC"} if role == "assay_fulltext" else {"ChEMBL"} if role == "assay_identity" else {"PubMed"} if role == "assay_context" else {"Europe PMC", "PubMed"})]
            if role == "direction" and alternatives:
                for path in paths:
                    if path["source"] == "PubMed":
                        path["source_task_key"] = digest({"task":key,"source":"PubMed",ALTERNATIVE_KEY:alternatives})
            for path in paths:
                if key in attempted or path["source_task_key"] in attempted_paths:
                    path["status"] = "already_attempted"
                    retry = public_path_retry(history, source_task_key=path["source_task_key"],
                        target=symbol, role=role, source=path["source"])
                    path["retry_decision"] = retry
                    if retry["allowed"]:
                        path["status"] = "available"
                        path["next_attempt_number"] = retry["attempts"] + 1
                else:
                    path["next_attempt_number"] = 1
            tasks.append({"task_key": key, "target": symbol, "role": role,
                **({ALTERNATIVE_KEY:alternatives} if role == "direction" and alternatives else {}),
                **({SEARCH_KEY: bindings, "fusion_scope_labels_omitted": max(0, len(scopes)-4)} if bindings else {}),
                **({VARIANT_KEY:variant_bindings,"variant_scope_labels_omitted":max(0,len(variants)-4)} if variant_bindings else {}),
                "path": "pmid_bound_oa_assay_methods" if role == "assay_fulltext" else "pmid_bound_chembl_assay_inventory" if role == "assay_identity" else "pubmed_required_assay_context" if role == "assay_context" else "epmc_oa_then_pubmed_precision", "source_paths":paths,
                "priority_basis":"missing_evidence_role_count_then_symbol_not_scientific_merit",
                "priority_value":priority[symbol],
                **({"role_authority":"required_assay_search_context_not_primary_gate_evidence"} if role == "assay_context" else {}),
                **({"role_authority":"database_identity_inventory_not_primary_gate_evidence",
                    "requires":"verified_assay_context_from_this_followup"} if role == "assay_identity" else {}),
                **({"role_authority":"licensed_methods_context_not_primary_gate_evidence",
                    "requires":"verified_assay_context_from_this_followup_and_oa_enabled"} if role == "assay_fulltext" else {}),
                "status": "already_attempted" if paths and all(p["status"] == "already_attempted" for p in paths) else "available"})
    selectable = (remaining > 0 and has_data_work
                  and bool({"PubMed", "Europe PMC"} & set(brief.data_policy.allowed_sources))
                  and verdict.verdict in {GateVerdictCode.REVISE, GateVerdictCode.REPLAN})
    selected_symbols = []
    for task in tasks:
        if not selectable or task["status"] != "available":
            continue
        if task["target"] not in selected_symbols and len(selected_symbols) < max(0, min(4, target_limit)):
            selected_symbols.append(task["target"])
        task["status"] = "scheduled" if task["target"] in selected_symbols else "target_budget_deferred"
    work = []
    if selected_symbols:
        work.append("cause_specific_source_retrieval")
    if (has_code_work and remaining and (reclassification_targets or packet.hypotheses)
            and not legacy_actions_unknown
            and action_key("reclassify_existing_evidence") not in completed_actions):
        work.append("reclassify_existing_evidence")
    # Generic discovery has no task-bound proof of a new query. Admit it only
    # once alongside concrete source work; history cannot grant a blind replay.
    discovery_done = any("existing_bounded_discovery_expansion" in row.get("automatic_actions", [])
                         or "automatic_actions" not in row for row in history)
    if (has_data_work and remaining and selected_symbols and not discovery_done
            and {"PubMed", "Open Targets"} & set(brief.data_policy.allowed_sources)):
        work.append("existing_bounded_discovery_expansion")
    if verdict.verdict not in {GateVerdictCode.REVISE, GateVerdictCode.REPLAN}:
        work = []
    if ma3_stop_reason:
        work = []
    lineage = packet.metadata.get("revision_lineage") or {}
    if lineage.get("scientific_changed") is False:
        work = []
    if not work:
        for task in tasks:
            if task["status"] == "scheduled":
                task["status"] = "not_scheduled"
    for task in tasks:
        task["selection_reason"] = (
            "EXISTING_PATHS_DONE_NO_SAFE_REPLAY" if task["status"] == "already_attempted" else
            "TARGET_BUDGET_DEFERRED_NOT_SCIENTIFIC_REJECTION" if task["status"] == "target_budget_deferred" else
            "BOUND_TRANSIENT_PUBLIC_RETRY_WITHIN_EXISTING_BUDGET" if task["status"] == "scheduled"
                and any(p.get("next_attempt_number", 1) > 1 for p in task["source_paths"]) else
            "UNATTEMPTED_ALLOWED_SOURCE_PATH" if task["status"] == "scheduled" else
            "GATE_LINEAGE_OR_BUDGET_DOES_NOT_ALLOW_EXECUTION")
    reason = (ma3_stop_reason if ma3_stop_reason else
              "AUTOMATIC_G0_WORK_AVAILABLE" if work else
              "G0_REWORK_BUDGET_EXHAUSTED" if not remaining else
              "G0_NO_SCIENTIFIC_DELTA" if lineage.get("scientific_changed") is False else
              "NO_RETAINED_HYPOTHESIS_NO_NEW_EVIDENCE" if not packet.hypotheses else
              "G0_NO_SAFE_AUTOMATIC_ACTION")
    plan = {"schema_version": "1.0", "policy": POLICY,
        "packet_id": packet.packet_id, "packet_sha256": digest(packet.model_dump(mode="json")),
        "verdict_id": verdict.verdict_id, "verdict_sha256": digest(verdict.model_dump(mode="json")),
        "brief_sha256": digest(brief.model_dump(mode="json")), "rework_loops": rework_loops,
        "remaining_rework_loops": remaining, "target_limit": target_limit,
        "source_elapsed_seconds": SOURCE_ELAPSED_SECONDS,
        "elapsed_limit_semantics": "lane_admission_not_exact_wall_clock_deadline",
        "automatic_actions": work, "tasks": tasks, "manual_requirements": manual,
        "evidence_work_fingerprint": fingerprint,
        "automatic_action_keys": [action_key(action) for action in work
                                  if action != "cause_specific_source_retrieval"],
        "retained_hypothesis_count": len(packet.hypotheses),
        "reclassification_targets": sorted(reclassification_targets),
        "reclassification_authority": (
            "typed_computational_finding_claim_id_bound_to_rejected_screening_trace"
        ),
        "public_retry_policy":{"max_attempts_per_source_task":2,
            "successful_empty_query_replayed":False,"paid_model_retry_authorized":False,
            "existing_rework_cost_and_source_limits_unchanged":True},
        "reason_code": reason, "eligible": bool(work),
        **({"ma3_followup_authority": "typed_needs_evidence_only",
            "ma3_admission_disposition_counts": ma3_dispositions,
            "legacy_critic_trace_cannot_override_ma3": True} if ma3_policy else {}),
        "gate_thresholds_unchanged": True, "human_approval_generated": False}
    plan["plan_id"] = "g0plan_" + digest(plan)[:24]
    return plan


class FollowupRetriever:
    """Execute only planned existing targets through existing public adapters."""
    def __init__(self, delegate, plan, prior_evidence=None):
        self.delegate, self.plan = delegate, plan
        self.public = delegate.public
        self.observations = []
        self.prior_evidence = prior_evidence or {}

    def _with_prior(self, records, seed):
        from drugpilot.orchestration.evidence_carryover import merge_sources
        prior = self.prior_evidence.get(seed.symbol.casefold(), ())
        return merge_sources(records, prior) if prior else records

    def retrieve(self, brief, seed, iteration):
        tasks = [t for t in self.plan["tasks"] if t["target"].casefold() == seed.symbol.casefold()
                 and t["status"] == "scheduled"]
        if not tasks:
            if (seed.symbol in self.plan.get("reclassification_targets", [])
                    and self.prior_evidence.get(seed.symbol.casefold())):
                # A typed computational finding requests reclassification of
                # immutable same-run sources. Do not silently replay public
                # network calls merely because the source task already ran.
                return tuple(
                    item.model_copy(deep=True)
                    for item in self.prior_evidence[seed.symbol.casefold()]
                )
            return self._with_prior(self.delegate.retrieve(brief, seed, iteration), seed)
        bindings_by_label = {b["label"]: b for t in tasks for b in t.get(SEARCH_KEY, [])}
        bindings = [bindings_by_label[label] for label in sorted(bindings_by_label)]
        if bindings:
            validated_fusion_search_terms(bindings, seed.symbol)
            seed = replace(seed, discovery_metadata={**seed.discovery_metadata, SEARCH_KEY: bindings})
        variants={b["label"]:b for t in tasks for b in t.get(VARIANT_KEY,[])}
        if variants:
            selected=[variants[label] for label in sorted(variants)]
            validate_variant_search(selected,seed.symbol)
            seed=replace(seed,discovery_metadata={**seed.discovery_metadata,VARIANT_KEY:selected})
        from drugpilot.orchestration.evidence_carryover import ALTERNATIVE_KEY, verify_alternatives
        alternatives = [p for t in tasks if t["role"] == "direction" for p in t.get(ALTERNATIVE_KEY, [])]
        if alternatives:
            pmids = verify_alternatives(brief, seed.symbol, alternatives,
                self.prior_evidence.get(seed.symbol.casefold(), ()))
            seed = replace(seed, discovery_metadata={**seed.discovery_metadata, ALTERNATIVE_KEY:pmids})
        records, observations = self.public.retrieve_g0_followup(
            brief, seed, roles=[t["role"] for t in tasks],
            source_paths=[{"source":p["source"], "role":t["role"]} for t in tasks
                for p in t["source_paths"] if p["status"] == "available"],
            elapsed_seconds=self.plan["source_elapsed_seconds"],
            **({"prior_records":self.prior_evidence[seed.symbol.casefold()]}
               if self.prior_evidence.get(seed.symbol.casefold()) else {}))
        self.observations.extend({**row, "evidence_gap":diagnose_observation(row), "target": seed.symbol,
            "task_keys": [t["task_key"] for t in tasks
                if t["role"] == row.get("role") or (row.get("role") == "oa_context" and t["role"] in {"association", "direction"})],
            "source_task_keys": [p["source_task_key"] for t in tasks for p in t["source_paths"]
                if p["source"] == row.get("source") and p["status"] == "available"
                and (t["role"] == row.get("role") or (row.get("role") == "oa_context" and t["role"] in {"association", "direction"}))],
            "plan_id": self.plan["plan_id"]} for row in observations)
        return self._with_prior(records, seed)

    def history_entry(self):
        attempted = {key for row in self.observations if row.get("attempted")
                     for key in row["source_task_keys"]}
        return {"plan_id": self.plan["plan_id"],
            **({"evidence_work_fingerprint": self.plan["evidence_work_fingerprint"],
                "automatic_actions": self.plan["automatic_actions"],
                "automatic_action_keys": self.plan["automatic_action_keys"]}
               if "evidence_work_fingerprint" in self.plan else {}),
            "attempted_source_task_keys": sorted(attempted),
            "attempted_task_keys": sorted(t["task_key"] for t in self.plan["tasks"]
                if t["status"] == "scheduled" and all(p["source_task_key"] in attempted
                    or p["status"] == "already_attempted" for p in t["source_paths"])),
            "observations": self.observations,
            "scheduled_targets": sorted({t["target"] for t in self.plan["tasks"] if t["status"] == "scheduled"})}
