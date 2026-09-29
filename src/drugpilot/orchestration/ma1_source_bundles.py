"""Preserve raw MA1 sources without calling or impersonating an interpreter."""
from hashlib import sha256
import json

from drugpilot.agents.evidence_semantics import (
    ABSTRACT_CONTEXT_KEY, context_digest, digest, source_review_text,
)
from drugpilot.domain.models import EvidenceRef
from drugpilot.orchestration.evidence_carryover import merge_sources
from drugpilot.orchestration.semantic_interpretation import _persist_bound

POLICY = "ma1-source-bound-collection-v1"


async def persist_source_bundle(o, run_id, brief, seed, iteration, evidence):
    """Return unchanged sources after durable source-only persistence, no LLM."""
    payload = {"policy": POLICY, "run_id": run_id, "target": seed.symbol,
        "iteration": iteration, "brief_sha256": digest(brief.model_dump(mode="json")),
        "context_sha256": context_digest(brief, seed.symbol), "semantic_models": [],
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "scope": "Complete retrieved source records; scientific role pending MA3; no interpretation performed"}
    artifact_id = "ma1raw_" + digest(payload)[:32]
    artifact_sha = await _persist_bound(o, run_id, artifact_id, "MA1EvidenceBundle", payload)
    await o._trace_event(run_id, "ma1_source_bundle_persisted",
        f"{seed.symbol}: 원문 {len(evidence)}개 보존 · 의미 심사는 MA3 담당.",
        node="ma1.source_collection", metadata={"policy": POLICY, "target": seed.symbol,
            "iteration": iteration, "evidence_bundle_id": artifact_id,
            "evidence_bundle_sha256": artifact_sha, "new_model_calls": 0,
            "source_record_count": len(evidence), "scientific_approval": False})
    return [item.model_copy(deep=True) for item in evidence]


async def load_source_bundles(store, run_id, brief):
    """Restore only same-run, same-request, hash-verified original literature."""
    snapshot = await store.get_snapshot(run_id)
    if snapshot is None or snapshot.run_id != run_id:
        raise ValueError("RAW_SOURCE_CARRYOVER_RUN_NOT_FOUND")
    records, lineage, seen = {}, [], set()
    for event in snapshot.events:
        if event.event_type != "ma1_source_bundle_persisted":
            continue
        metadata = event.metadata
        artifact_id = metadata.get("evidence_bundle_id")
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        bundle = await store.read_artifact(run_id, artifact_id, "MA1EvidenceBundle")
        if (not isinstance(bundle, dict) or bundle.get("policy") != POLICY
                or artifact_id != "ma1raw_" + digest(bundle)[:32]
                or bundle.get("run_id") != run_id or bundle.get("target") != metadata.get("target")
                or metadata.get("evidence_bundle_sha256") != sha256(json.dumps(
                    bundle, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
                or bundle.get("semantic_models") != []):
            raise ValueError("RAW_SOURCE_CARRYOVER_BUNDLE_INVALID")
        target = bundle["target"]
        if (bundle.get("brief_sha256") != digest(brief.model_dump(mode="json"))
                or bundle.get("context_sha256") != context_digest(brief, target)):
            continue
        reusable, omitted = [], []
        for raw in bundle["evidence"]:
            evidence = EvidenceRef.model_validate(raw)
            if evidence.source not in {"PubMed", "Europe PMC"}:
                continue
            context = evidence.metadata.get(ABSTRACT_CONTEXT_KEY)
            if (evidence.metadata.get("excerpt") is None and (
                context is None or (isinstance(context, dict)
                    and context.get("status") == "omitted_size_limit"
                    and context.get("text") is None)
            )):
                omitted.append(evidence.evidence_id)
                continue
            if evidence.source not in brief.data_policy.allowed_sources or source_review_text(evidence) is None:
                raise ValueError("RAW_SOURCE_CARRYOVER_SOURCE_INVALID")
            reusable.append(evidence)
        records[target.casefold()] = merge_sources(reusable, records.get(target.casefold(), ()))
        lineage.append({"bundle_id": artifact_id, "target": target, "source_record_count": len(reusable),
            "omitted_metadata_only_ids": omitted, "new_retrieval": False,
            "new_model_calls": 0, "source_only_no_semantic_approval": True})
    return records, lineage
