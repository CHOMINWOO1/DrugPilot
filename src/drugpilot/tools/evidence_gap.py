"""Observed retrieval/selection gaps, never biological absence or gate verdicts.

Diagnostics are recomputed from adapter observations, not model prose. Missing
historical extraction telemetry remains unknown. Counts concern screened query
results, not all literature or all experiments in a paper.
"""
from collections import Counter
from hashlib import sha256
import json
import re

POLICY = "observed_evidence_gap_v1"
_TRANSIENT_TYPES = {"TimeoutError", "ConnectTimeout", "ReadTimeout", "ConnectError", "RemoteProtocolError"}
_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}
_INELIGIBLE = {"review_publication_type", "review_title", "non_primary_synthesis_publication_type"}
_METADATA = {"summary_metadata_missing", "publication_type_missing", "primary_publication_type_not_verified"}
_COUNTER = {"counter_signal", "unfavorable_endpoint_polarity", "non_direction_or_negative_context",
            "target_inhibitor_response_modifier", "other_intervention_response_modifier",
            "multi_target_slash_intervention", "multi_target_co_intervention"}
_CRITERIA = {"association_role_only", "primary_result_context_missing", "exact_target_base_disease_match_missing",
             "requested_direction_missing", "direct_target_intervention_missing", "downstream_only_endpoint",
             "measured_endpoint_missing", "positive_result_language_missing", "favorable_endpoint_polarity_missing",
             "required_subtype_or_variant_not_matched", "primary_association_excerpt_missing"}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode()).hexdigest()


def content_observation(content):
    """Only lengths/status/hashes; no new excerpt, label or primary evidence."""
    context = content.interpretation_context or {}
    content_hash = content.content_hash
    valid_hash = isinstance(content_hash, str) and bool(re.fullmatch(r"[0-9a-f]{64}", content_hash))
    text = context.get("text")
    verified = (context.get("status") == "complete" and isinstance(text, str)
                and sha256(text.encode()).hexdigest() == content_hash)
    return {"policy": POLICY, "abstract_received": bool(valid_hash),
        "abstract_sha256": content_hash if valid_hash else None,
        "excerpt_selected": bool(content.excerpt and content.source_location),
        "excerpt_sha256": sha256(content.excerpt.encode()).hexdigest() if content.excerpt else None,
        "excerpt_char_count": len(content.excerpt or ""),
        "complete_context_verified": verified,
        "context_omitted_for_size": context.get("status") == "omitted_size_limit",
        "selection_status": content.excerpt_selection_status if content.excerpt_selection_status in {
            "not_available", "abstract_not_available", "related_excerpt_not_found", "selected",
            "selected_counter_signal", "selected_compound_linked_document_pair"} else "unknown"}


def _count(value):
    return value if type(value) is int and 0 <= value <= 100000 else None


def _bucket(entry):
    raw_codes = entry.get("rejection_codes", [])
    codes = set(x for x in raw_codes if isinstance(x, str)) if isinstance(raw_codes, list) else set()
    if entry.get("status") == "selected" and not codes:
        return "adapter_selected_pending_downstream_checks"
    if entry.get("assay_context_only") is True and entry.get("status") == "context_found":
        return "assay_context_unassigned_measurement"
    if codes & _INELIGIBLE:
        return "publication_ineligible"
    if codes & _METADATA:
        return "publication_metadata_unverified"
    if entry.get("document_unit_source_preserved") is True:
        return "document_unit_validation_pending"
    obs = entry.get("extraction_observation")
    if isinstance(obs, dict) and obs.get("policy") == POLICY:
        if obs.get("abstract_received") is False:
            return "abstract_not_available"
        if obs.get("abstract_received") is True and obs.get("excerpt_selected") is False:
            return "abstract_received_excerpt_not_selected"
    if "abstract_unavailable" in codes:
        return "abstract_not_available"
    if "abstract_excerpt_unavailable" in codes:
        return "historical_extraction_state_unknown"
    if codes & _COUNTER:
        return "counter_or_mixed_context_not_direct_refutation"
    if codes & _CRITERIA:
        return "selected_excerpt_criteria_unmet"
    return "unclassified_observation"


def transient_failure(row):
    """Only failed public retrieval, never a paid model or successful empty lane."""
    if row.get("source") not in {"PubMed", "Europe PMC"} or row.get("attempted") is not True:
        return False
    if type(row.get("record_count")) is not int or row["record_count"] != 0:
        return False
    if row.get("status") not in {"failed", "error"}:
        return False
    if row.get("reason_code") in _TRANSIENT_TYPES:
        return True
    trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
    if trace.get("error_type") in _TRANSIENT_TYPES:
        return True
    # Only typed per-call HTTP failures; a source summary or free-text "503"
    # cannot grant retry. Protocol/identity errors after HTTP 200 are not transient.
    calls = trace.get("http_call_provenance", [])
    return isinstance(calls, list) and any(isinstance(c, dict)
        and c.get("status") == "error" and type(c.get("status_code")) is int
        and c["status_code"] in _TRANSIENT_HTTP for c in calls)


def diagnose_observation(row):
    trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
    oa = row.get("oa_trace") if isinstance(row.get("oa_trace"), dict) else {}
    source_errors = bool(trace.get("lane_errors") or oa.get("error_reason_codes"))
    entries = trace.get("screening_entries")
    entries = entries if isinstance(entries, list) else []
    bounded = [e for e in entries[:10000] if isinstance(e, dict)]
    counts = dict(sorted(Counter(_bucket(e) for e in bounded).items()))
    reason = row.get("reason_code")
    if row.get("attempted") is False:
        outcome = "budget_not_attempted" if reason == "SOURCE_OR_TIME_BUDGET" else "not_attempted"
    elif transient_failure(row):
        outcome = "transient_retrieval_failure"
    elif source_errors and (_count(row.get("record_count")) or 0) > 0:
        outcome = "records_returned_with_source_errors"
    elif row.get("status") in {"failed", "error"} or source_errors:
        outcome = "retrieval_or_protocol_failure"
    elif row.get("role") == "assay_identity" and row.get("status") == "completed":
        outcome = "assay_identity_inventory_completed"
    elif row.get("role") == "assay_fulltext" and row.get("status") == "completed":
        outcome = "assay_fulltext_context_completed"
    elif counts.get("document_unit_validation_pending"):
        outcome = "document_unit_validation_pending"
    elif counts.get("adapter_selected_pending_downstream_checks") or (_count(row.get("record_count")) or 0) > 0:
        outcome = "records_returned_not_gate_pass"
    elif entries:
        outcome = "screened_without_qualified_excerpt"
    elif _count(trace.get("search_result_count")) == 0 and trace.get("status") in {
            "completed_bounded_without_verified_result", "completed_empty", "empty"}:
        outcome = "no_hits_in_this_query"
    else:
        outcome = "insufficient_diagnostic_telemetry"
    actions = []
    if outcome == "transient_retrieval_failure":
        actions.append("retry_public_path_once_if_budget_and_lineage_allow")
    if counts.get("abstract_not_available") or counts.get("abstract_received_excerpt_not_selected"):
        actions.append("try_unattempted_licensed_fulltext_path")
    if counts.get("historical_extraction_state_unknown"):
        actions.append("do_not_infer_absence_from_legacy_telemetry")
    if counts.get("document_unit_validation_pending"):
        actions.append("validate_preserved_document_unit_with_existing_gates")
    if counts.get("assay_context_unassigned_measurement"):
        actions.append("preserve_assay_context_without_assigning_potency")
    if outcome in {"screened_without_qualified_excerpt", "no_hits_in_this_query"}:
        actions.append("try_other_unattempted_allowed_path_then_hold")
    if outcome in {"budget_not_attempted", "not_attempted", "retrieval_or_protocol_failure", "insufficient_diagnostic_telemetry"}:
        actions.append("hold_with_explicit_unresolved_reason")
    if outcome == "records_returned_with_source_errors":
        actions.append("preserve_partial_records_and_unresolved_source_errors")
    basis = {k:row.get(k) for k in ("source", "role", "attempted", "status", "reason_code", "record_count")}
    basis["trace"] = {k:v for k,v in trace.items() if k != "evidence_gap"}
    basis["oa_trace"] = oa
    result = {"policy":POLICY, "outcome":outcome, "basis_sha256":digest(basis),
        "search_result_count":_count(trace.get("search_result_count")),
        "records_screened":_count(trace.get("records_screened")),
        "unsearched_result_count":_count(trace.get("unsearched_result_count")),
        "classified_entry_count":len(bounded), "omitted_entry_count":max(0,len(entries)-len(bounded)),
        "exclusive_record_buckets":counts, "next_actions":actions,
        "source_error_observed":source_errors or row.get("status") in {"failed", "error"},
        "oa_parent_fulltexts_parsed":_count((oa.get("observed") or {}).get("parent_fulltexts_parsed"))
            if isinstance(oa.get("observed"), dict) else None,
        "oa_child_fulltexts_parsed":_count((oa.get("observed") or {}).get("child_fulltexts_parsed"))
            if isinstance(oa.get("observed"), dict) else None,
        "scope":"recorded_query_window_and_selected_excerpts_only",
        "biological_absence_established":False, "scientific_verdict_changed":False}
    if row.get("role") == "assay_context":
        result.update(matched_context_documents=_count(trace.get("matched_document_count")),
            retained_context_documents=_count(row.get("record_count")),
            omitted_matching_context_documents=_count(trace.get("omitted_matching_document_count")))
    if row.get("role") == "assay_identity":
        result.update(identity_document_count=_count(trace.get("documents_found")),
            identity_activity_rows_screened=_count(trace.get("activity_rows_screened")),
            identity_linked_activity_count=_count(trace.get("linked_activity_count")))
        applicability = trace.get("assay_applicability_summary")
        if isinstance(applicability, dict) and applicability.get("policy") == "deposited_assay_applicability_v1":
            counts = applicability.get("exclusive_annotation_counts")
            result["identity_annotation_counts"] = {k:_count(counts.get(k, 0)) for k in (
                "not_recorded", "unresolved_annotation", "class_annotation_only", "named_annotation_only", "annotation_conflict")
            } if isinstance(counts, dict) else None
    if row.get("role") == "assay_fulltext":
        counts=trace.get("document_status_counts")
        result["assay_fulltext_counts"]={k:_count(counts.get(k,0)) for k in (
            "parsed", "oa_not_admitted", "metadata_not_found", "budget_not_completed", "retrieval_or_validation_error")
        } if isinstance(counts,dict) else None
        result["assay_method_paragraphs"]=_count(trace.get("method_paragraphs_retained"))
        result["assay_table_locators"]=_count(trace.get("table_locators_retained"))
    return result
