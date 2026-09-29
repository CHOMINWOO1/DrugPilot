"""Reuse immutable same-run source bundles, never reviewer prose or new evidence.

Retrieval progress is cumulative. Reusing a source is not another independent
publication, new network call, or permission to resample a disputed model answer.
"""
from hashlib import sha256
from pathlib import Path
import json
import re

from drugpilot.agents.evidence_semantics import (
    KEY, POLICY, EvidenceMeaning, context_digest, digest, source_binding, source_records,
)
from drugpilot.domain.models import EvidenceRef

POLICY_NAME = "same_run_verified_evidence_carryover_v1"
ALTERNATIVE_KEY = "g0_disputed_selection_exclusions"
_PRIOR_LOADER_ONLY_SHA256 = (
    "78087bc5969400b1106cf14669dc549d4c65ec80f28f86922bb9acda490bbef5",
    "4609ce6c3981a28dfcd2141f813c796e6d607f8f9d8d31fed3e28f4f09c32e6a",
)


def metadata_omission_compatible_contract():
    """One audited loader-only migration; every interpretation file stays exact.

    The old loader rejected valid metadata-only search hits. Its source-bearing
    records used the identical model/prompt/scope checks. Do not resample those
    answers just because we now omit non-evidence. No run/disease exceptions.
    Changing any other interpretation file invalidates this compatibility ID.
    """
    root = Path(__file__).parents[1]
    files = (Path(__file__).with_name("semantic_interpretation.py"),
             root / "agents/evidence_semantics.py", root / "agents/molecular_scope.py")
    return digest({**{p.name: sha256(p.read_bytes()).hexdigest() for p in files},
        "evidence_carryover.py": _PRIOR_LOADER_ONLY_SHA256[0]})


def metadata_omission_compatible_contracts():
    """Exact audited contracts for prior loader-only revisions."""
    root = Path(__file__).parents[1]
    files = (Path(__file__).with_name("semantic_interpretation.py"),
             root / "agents/evidence_semantics.py", root / "agents/molecular_scope.py")
    current_interpretation_files = {
        p.name: sha256(p.read_bytes()).hexdigest() for p in files
    }
    return frozenset(
        digest({**current_interpretation_files, "evidence_carryover.py": old_sha})
        for old_sha in _PRIOR_LOADER_ONLY_SHA256
    )


def _non_evidentiary_source_record(e, text, annotation):
    """Recognize source-bound records that contain no reusable excerpt."""
    if text is not None:
        return False
    hash_carriers = [e.title]
    detail = e.metadata.get("detail")
    if isinstance(detail, str) and detail:
        hash_carriers.append(detail)
    hash_is_explainable = e.excerpt_hash is None or any(
        sha256(value.encode()).hexdigest() == e.excerpt_hash
        for value in hash_carriers
    )
    return (
        hash_is_explainable
        and e.metadata.get("content_availability") in {None, "metadata_only"}
        and e.metadata.get("direct_support_eligible") in {None, False}
        and e.metadata.get("provenance_class") in {"external_live", "external_cache"}
        and not e.metadata.get("fixture")
        and not e.metadata.get("mock")
        and annotation.get("policy") == POLICY
        and annotation.get("status") == "not_selected_or_source_unverifiable"
        and annotation.get("extractor") is None
        and annotation.get("verifier") is None
    )


async def load_prior_bundles(store, run_id, brief, *, contract, models):
    snapshot = await store.get_snapshot(run_id)
    if snapshot is None or snapshot.run_id != run_id:
        raise ValueError("CARRYOVER_RUN_NOT_FOUND")
    records, lineage, seen = {}, [], set()
    for event in snapshot.events:
        if event.event_type != "ma1_semantic_interpretation_completed":
            continue
        m = event.metadata
        bid = m.get("evidence_bundle_id")
        if not bid or bid in seen:
            continue
        seen.add(bid)
        bundle = await store.read_artifact(run_id, bid, "MA1EvidenceBundle")
        if (bundle is None or bid != "ma1ev_" + digest(bundle)[:32]
                or m.get("evidence_bundle_sha256") != sha256(json.dumps(
                    bundle, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
                or bundle.get("run_id") != run_id or bundle.get("target") != m.get("target")
                or bundle.get("policy") != POLICY):
            raise ValueError("CARRYOVER_BUNDLE_BINDING_MISMATCH")
        target = bundle["target"]
        # Legacy contracts and different contexts cannot authorize reuse.
        bundle_contract = bundle.get("interpretation_contract_sha256")
        if (bundle_contract not in ({contract} | set(metadata_omission_compatible_contracts()))
                or bundle.get("semantic_models") != models
                or bundle.get("context_sha256") != context_digest(brief, target)):
            continue
        es, omitted = [], []
        for raw in bundle["evidence"]:
            e = EvidenceRef.model_validate(raw)
            if e.source not in {"PubMed", "Europe PMC"}:
                continue
            text = e.metadata.get("excerpt")
            a = e.metadata.get(KEY) or {}
            # A source-bound, explicitly non-evidentiary search hit is not a
            # corrupt excerpt. Keep it in its immutable bundle, not in the
            # reusable evidence pool. A missing hash on *present* text still
            # fails below, as do synthetic/fixture and altered annotations.
            if (_non_evidentiary_source_record(e, text, a)
                    and a.get("context_sha256") == context_digest(brief, target)
                    and a.get("source_sha256") == source_binding(e)
                    and e.source in brief.data_policy.allowed_sources):
                omitted.append({"evidence_id":e.evidence_id,
                    "reason":"source_record_without_bound_excerpt", "original_record_preserved":True})
                continue
            if (e.metadata.get("provenance_class") not in {"external_live", "external_cache"}
                    or e.metadata.get("fixture") or e.metadata.get("mock")
                    or not isinstance(text, str) or not text
                    or sha256(text.encode()).hexdigest() != e.excerpt_hash
                    or e.source not in brief.data_policy.allowed_sources):
                raise ValueError("CARRYOVER_SOURCE_UNVERIFIABLE")
            es.append(e)
        records[target.casefold()] = merge_sources(es, records.get(target.casefold(), ()))
        lineage.append({"bundle_id":bid,"bundle_digest":digest(bundle),"target":target,
            "source_record_count":len(es),"new_retrieval":False,
            "omitted_non_evidence":omitted,
            "interpretation_contract_sha256":bundle_contract,
            "loader_only_compatibility_used":bundle_contract != contract})
    return records, lineage


def merge_sources(current, prior):
    merged = {e.evidence_id:e.model_copy(deep=True) for e in current}
    if len(merged) != len(current):
        raise ValueError("CARRYOVER_DUPLICATE_CURRENT_ID")
    for e in prior:
        existing = merged.get(e.evidence_id)
        if existing is not None:
            if source_binding(existing) != source_binding(e):
                raise ValueError("CARRYOVER_SOURCE_ID_CONFLICT")
            # A current retrieval keeps its own provenance; the historical
            # artifact remains immutable and linked in the carryover event.
            continue
        merged[e.evidence_id] = e.model_copy(deep=True)
    return tuple(merged.values())


def preserved_meanings(brief, target, records, prior):
    """Keep the first checked pair, including disagreement/empty outputs.

Only load_prior_bundles' verified same-contract records may enter this function.
Current public metadata alone is never authority to skip model interpretation.
"""
    fingerprints = {r["record_key"]:digest({k:v for k,v in r.items() if k != "sources"}) for r in records}
    saved = {}
    for e in prior:
        a = e.metadata.get(KEY)
        old = source_records([e])
        if not isinstance(a, dict) or not old:
            continue
        key = old[0]["record_key"]
        if (a.get("status") == "not_selected_or_source_unverifiable"
                or fingerprints.get(key) != digest({k:v for k,v in old[0].items() if k != "sources"})):
            continue
        if (a.get("policy") != POLICY or a.get("record_key") != key
                or a.get("context_sha256") != context_digest(brief, target)
                or a.get("source_sha256") != source_binding(e)):
            raise ValueError("CARRYOVER_MEANING_BINDING_MISMATCH")
        pair = tuple(EvidenceMeaning.model_validate(a[k]) if a.get(k) else None
                     for k in ("extractor", "verifier"))
        if any(m is not None and m.record_key != key for m in pair):
            raise ValueError("CARRYOVER_MEANING_RECORD_MISMATCH")
        if key in saved and saved[key] != pair:
            raise ValueError("CARRYOVER_MEANING_CONFLICT")
        saved[key] = pair
    return saved


def alternative_selections(brief, packet, target):
    """Skip an already selected, source-bound disputed result, not its evidence.

The next lookup examines other publications. This is not a biological rejection
and the disputed source remains in the immutable/cumulative evidence bundle.
"""
    selected = set()
    for h in packet.metadata.get("g0_followup_history", []):
        for o in h.get("observations", []):
            if (o.get("target") != target or o.get("source") != "PubMed"
                    or o.get("role") != "direction" or o.get("attempted") is not True
                    or o.get("status") != "completed_with_results"
                    or o.get("plan_id") != h.get("plan_id")
                    or not set(o.get("source_task_keys", [])) & set(h.get("attempted_source_task_keys", []))):
                continue
            for s in o.get("trace", {}).get("screening_entries", []):
                x = s.get("extraction_observation", {})
                if (s.get("status") == "selected" and not s.get("rejection_codes")
                        and x.get("excerpt_selected") is True):
                    selected.add((s.get("external_id"), x.get("excerpt_sha256")))
    proofs = {}
    for entry in packet.metadata.get("rejected_screening_trace", {}).get("entries", []):
        if ((entry.get("target") or {}).get("symbol") != target
                or "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY" not in entry.get("reason_codes", [])):
            continue
        for record in entry.get("unique_external_records", []):
            if (record.get("source") != "PubMed" or not record.get("provenance_classes")
                    or not set(record["provenance_classes"]).issubset({"external_live", "external_cache"})):
                continue
            pmid = record.get("external_id")
            if not isinstance(pmid, str) or not re.fullmatch(r"[0-9]{1,12}", pmid):
                continue
            for unit in record.get("evidence_units", []):
                a, text = unit.get(KEY) or {}, unit.get("bounded_excerpt")
                if ((pmid, unit.get("excerpt_hash")) not in selected
                        or a.get("policy") != POLICY or a.get("status") != "disagreement"
                        or a.get("context_sha256") != context_digest(brief, target)
                        or a.get("publication_key") != "PMID:" + pmid
                        or not isinstance(text, str) or sha256(text.encode()).hexdigest() != unit.get("excerpt_hash")):
                    continue
                proofs[pmid] = {"pmid":pmid,"evidence_id":unit["evidence_id"],
                    "excerpt_sha256":unit["excerpt_hash"],"record_key":a["record_key"],
                    "context_sha256":a["context_sha256"],"target":target,
                    "authority":"alternative_search_only_prior_disagreement_preserved"}
    return [proofs[k] for k in sorted(proofs)]


def verify_alternatives(brief, target, proofs, prior):
    """The bounded packet's hints must match full verified same-run sources."""
    by_id = {e.evidence_id:e for e in prior}
    pmids = set()
    for p in proofs:
        e = by_id.get(p.get("evidence_id"))
        a = e.metadata.get(KEY, {}) if e else {}
        if (e is None or e.source != "PubMed" or e.external_id != p.get("pmid")
                or e.excerpt_hash != p.get("excerpt_sha256") or p.get("target") != target
                or p.get("authority") != "alternative_search_only_prior_disagreement_preserved"
                or a.get("status") != "disagreement" or a.get("record_key") != p.get("record_key")
                or a.get("source_sha256") != source_binding(e)
                or a.get("context_sha256") != p.get("context_sha256")
                or p.get("context_sha256") != context_digest(brief, target)):
            raise ValueError("ALTERNATIVE_SELECTION_NOT_BOUND_TO_PRIOR_SOURCE")
        pmids.add(e.external_id)
    return sorted(pmids)
