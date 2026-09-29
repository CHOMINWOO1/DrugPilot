"""Bounded, metered semantic interpretation before MA1 claim assignment."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from hashlib import sha256
from drugpilot.domain.models import ExecutionEvent

from drugpilot.agents.evidence_semantics import (
    KEY, POLICY, EvidenceMeaningBatch, attach_interpretations,
    interpretation_prompt, source_records, digest, context_digest, record_key,
    _checked,
)


async def _persist_bound(o, run_id, artifact_id, artifact_type, payload):
    snapshot = await o.store.get_snapshot(run_id)
    if snapshot is None:
        raise KeyError(run_id)
    event = ExecutionEvent(run_id=run_id, stage=snapshot.stage,
        event_type="ma1_intermediate_artifact_committed", node="ma1.evidence_interpretation",
        message=f"{artifact_type} saved with its immutable audit binding.",
        artifact_ids=[artifact_id], metadata={"artifact_type":artifact_type})
    digest = await o.store.commit_artifact_event(snapshot, artifact_id, artifact_type, payload, event)
    # Persisted event is authoritative; broker delivery is only a UI notification.
    if getattr(o, "broker", None) is not None:
        event.metadata["artifact_sha256"] = {artifact_id:digest}
        await o.broker.publish(event)
    return digest


def interpretation_contract():
    return digest({str(p.name): sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(__file__).with_name("evidence_carryover.py"),
        Path(__file__).parents[1] / "agents/evidence_semantics.py",
        Path(__file__).parents[1] / "agents/molecular_scope.py")})


async def interpret_target(o, run_id, brief, seed, iteration, evidence, *, preserved_evidence=()):
    from drugpilot.orchestration.evidence_carryover import preserved_meanings
    records = source_records(evidence)
    contract = interpretation_contract()
    models = [o.settings.terra_model, o.settings.sol_model]
    preserved = preserved_meanings(brief, seed.symbol, records, preserved_evidence)
    fresh_records = [r for r in records if r["record_key"] not in preserved]
    new_model_calls = 0
    recovery_calls = 0
    recovery_record_keys: dict[str, list[str]] = {
        "extractor": [],
        "verifier": [],
    }
    empty = EvidenceMeaningBatch(records=[])
    first = second = empty
    store = getattr(o, "store", None)
    cached = False
    if records:
        prompt = interpretation_prompt(brief, seed.symbol, records)
        binding = {"run_id": run_id, "context": context_digest(brief, seed.symbol),
            "prompt": sha256(prompt.encode()).hexdigest(), "contract": contract,
            "models": models}
        if preserved:
            binding["preserved_meanings_sha256"] = digest({k:[m.model_dump() if m else None for m in pair]
                for k,pair in preserved.items()})
        cache_key = digest(binding)
        attempt_id, result_id = "sem_attempt_" + cache_key[:32], "sem_result_" + cache_key[:32]
        prior = await store.read_artifact(run_id, result_id, "MA1SemanticResult") if store else None
        if prior is not None:
            if prior.get("binding") != binding:
                raise ValueError("SEMANTIC_CACHE_BINDING_MISMATCH")
            first = EvidenceMeaningBatch.model_validate(prior["extractor"])
            second = EvidenceMeaningBatch.model_validate(prior["verifier"])
            cached = True
        else:
            if store and fresh_records:
                if await store.read_artifact(run_id, attempt_id, "MA1SemanticAttempt") is not None:
                    raise ValueError("SEMANTIC_ATTEMPT_INCOMPLETE_NO_AUTOMATIC_REPLAY")
                # Durable intent before dispatch; an interrupted paid operation
                # is not silently replayed after resume or process replacement.
                await _persist_bound(o, run_id, attempt_id, "MA1SemanticAttempt", {"binding": binding})
            prompt = interpretation_prompt(brief, seed.symbol, fresh_records)
            if fresh_records:
                first = await o._llm_enhance(
                    run_id, component="MA1-Evidence-Interpreter", model=o.settings.terra_model,
                    response_model=EvidenceMeaningBatch, fallback=empty, prompt=prompt,
                    # Provider limit includes reasoning and visible JSON.
                    max_output_tokens=min(32768, 8192 + 768 * len(fresh_records)), failure_is_fatal=False,
                )
                # Blinded second pass: no first-pass output or prior role labels.
                second = await o._llm_enhance(
                    run_id, component="MA1-Evidence-Verifier", model=o.settings.sol_model,
                    response_model=EvidenceMeaningBatch, fallback=empty,
                    prompt="Independently verify each source excerpt for the requested claim context.\n" + prompt,
                    max_output_tokens=min(32768, 8192 + 768 * len(fresh_records)), failure_is_fatal=False,
                )
                new_model_calls = 2
                # A deterministic primary-association selector may find a
                # valid source record while one structured model omits it or
                # emits an ungrounded quote.  Retry only those already-fetched
                # role-specific association records once per model.  Both
                # independent models must still pass the unchanged semantic
                # and primary-source gates; this never promotes a record by
                # deterministic metadata alone.
                association_keys = {
                    record_key(item)
                    for item in evidence
                    if item.metadata.get("role_specific_record")
                    == "primary_target_disease_association_excerpt"
                }

                async def recover_missing(
                    batch: EvidenceMeaningBatch,
                    *,
                    label: str,
                    component: str,
                    model: str,
                ) -> EvidenceMeaningBatch:
                    nonlocal new_model_calls, recovery_calls
                    grounded = set(_checked(batch, fresh_records))
                    missing = {
                        item["record_key"]
                        for item in fresh_records
                        if item["record_key"] in association_keys
                        and item["record_key"] not in grounded
                    }
                    if not missing:
                        return batch
                    retry_records = [
                        item
                        for item in fresh_records
                        if item["record_key"] in missing
                    ]
                    retry_prompt = (
                        "Bounded recovery: the prior structured response omitted or "
                        "failed to ground these already-selected primary-association "
                        "records. Return every supplied record_key exactly once using "
                        "only exact contiguous quotes. A neutral measured mutation "
                        "prevalence is association_primary with intervention=none and "
                        "outcome=associated; quote the prevalence as result_quote. "
                        "Do not infer treatment direction from association evidence.\n"
                        + interpretation_prompt(brief, seed.symbol, retry_records)
                    )
                    retry = await o._llm_enhance(
                        run_id,
                        component=component,
                        model=model,
                        response_model=EvidenceMeaningBatch,
                        fallback=empty,
                        prompt=retry_prompt,
                        max_output_tokens=min(
                            32768,
                            8192 + 768 * len(retry_records),
                        ),
                        failure_is_fatal=False,
                    )
                    new_model_calls += 1
                    recovery_calls += 1
                    recovery_record_keys[label] = sorted(missing)
                    retained = [
                        item for item in batch.records
                        if item.record_key not in missing
                    ]
                    recovered = [
                        item for item in retry.records
                        if item.record_key in missing
                    ]
                    return EvidenceMeaningBatch(records=retained + recovered)

                first = await recover_missing(
                    first,
                    label="extractor",
                    component="MA1-Evidence-Interpreter-Recovery",
                    model=o.settings.terra_model,
                )
                second = await recover_missing(
                    second,
                    label="verifier",
                    component="MA1-Evidence-Verifier-Recovery",
                    model=o.settings.sol_model,
                )
            fresh_keys = {r["record_key"] for r in fresh_records}
            first = EvidenceMeaningBatch(records=[m for m in first.records if m.record_key in fresh_keys]
                + [pair[0] for pair in preserved.values() if pair[0] is not None])
            second = EvidenceMeaningBatch(records=[m for m in second.records if m.record_key in fresh_keys]
                + [pair[1] for pair in preserved.values() if pair[1] is not None])
            if store:
                await _persist_bound(o, run_id, result_id, "MA1SemanticResult", {"binding": binding,
                    "extractor": first.model_dump(), "verifier": second.model_dump()})
    result = attach_interpretations(brief, seed.symbol, evidence, records, first, second)
    bundle_id = None
    bundle_sha = None
    if store:
        bundle = {"run_id": run_id, "target": seed.symbol, "iteration": iteration,
            "context_sha256": context_digest(brief, seed.symbol), "policy": POLICY,
            "interpretation_contract_sha256":contract, "semantic_models":models,
            "evidence": [e.model_dump(mode="json") for e in result],
            "scope": "Complete retrieved source records and derived annotations; not a gate verdict"}
        bundle_id = "ma1ev_" + digest(bundle)[:32]
        bundle_sha = await _persist_bound(o, run_id, bundle_id, "MA1EvidenceBundle", bundle)
    statuses = Counter(e.metadata[KEY]["status"] for e in result if KEY in e.metadata)
    keys = {r["record_key"] for r in records}
    complete = (len(first.records) == len(records) == len(second.records)
        and {r.record_key for r in first.records} == keys == {r.record_key for r in second.records})
    completion = "완료" if complete else "일부 미완료 · 미확인 근거는 채택하지 않음"
    await o._trace_event(run_id, "ma1_semantic_interpretation_completed",
        f"{seed.symbol}: 원문 {len(records)}개 의미 해석·독립 대조 {completion} (과학적 통과와 별도).",
        node="ma1.evidence_interpretation", metadata={"policy": POLICY,
            "target": seed.symbol, "iteration": iteration,
            "unique_excerpt_count": len(records), "statuses": dict(statuses),
            "extractor_record_count": len(first.records), "verifier_record_count": len(second.records),
            "model_outputs_complete": complete,
            "semantic_cache_hit": cached, "new_model_calls": new_model_calls,
            "bounded_association_recovery_calls": recovery_calls,
            "bounded_association_recovery_record_keys": recovery_record_keys,
            "association_recovery_does_not_bypass_model_agreement": True,
            "preserved_interpretation_count":len(preserved),
            "fresh_interpretation_count":len(fresh_records) if not cached else 0,
            "prior_disagreement_resampled":False,
            "evidence_bundle_id": bundle_id, "evidence_bundle_sha256": bundle_sha,
            "model_agreement_is_not_scientific_truth": True,
            "existing_primary_gates_required": True})
    return result
