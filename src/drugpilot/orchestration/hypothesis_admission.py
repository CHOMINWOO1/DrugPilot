"""Durable, run-local boundary for paid source-bound MA3 admission.

An unfinished dispatch is not replayed automatically. Original MA1 packets and
every model usage record remain immutable even when assessment is interrupted.
"""
from __future__ import annotations

from hashlib import sha256
import json

from drugpilot.domain.models import HypothesisPacket, RunStage


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


async def review_submission(o, brief, packet, run_id):
    from drugpilot.agents.hypothesis_admission import (
        assess_provisional, AdmissionAssessment, SYSTEM_PROMPT, PROMPT_VERSION,
    )
    from drugpilot.orchestration.budget import BudgetGuard
    from drugpilot.providers.llm import conservative_structured_input_tokens

    async def preflight(model, prompt, max_output):
        await o._cancellation_checkpoint(run_id)
        snapshot = await o.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        BudgetGuard(snapshot.brief.budget).preflight(
            snapshot.token_usage, model=model,
            input_tokens=conservative_structured_input_tokens(
                system=SYSTEM_PROMPT, prompt=prompt, response_model=AdmissionAssessment),
            max_output_tokens=max_output,
        )

    binding = {
        "policy": "ma3-source-bound-v1",
        "prompt_version": PROMPT_VERSION,
        "schema_sha256": digest(AdmissionAssessment.model_json_schema()),
        "run_id": run_id,
        "brief_sha256": digest(brief.model_dump(mode="json")),
        "submission_sha256": digest(packet.model_dump(mode="json")),
        "model": o.ma3.reviewer_model,
        "enabled": o.settings.ma1_provisional_admission_enabled,
    }
    key = digest(binding)
    attempt_id, result_id = "ma3_attempt_" + key[:24], "ma3_result_" + key[:24]
    stored = await o.store.read_artifact(run_id, result_id, "MA3AdmissionResult")
    if stored is not None:
        if stored.get("binding") != binding or stored.get("packet_sha256") != digest(stored["packet"]):
            raise ValueError("MA3_ADMISSION_CACHE_BINDING_MISMATCH")
        approved = HypothesisPacket.model_validate(stored["packet"])
    else:
        prior = await o.store.read_artifact(run_id, attempt_id, "MA3AdmissionAttempt")
        if prior is not None and prior.get("binding") != binding:
            raise ValueError("MA3_ADMISSION_ATTEMPT_BINDING_MISMATCH")
        if prior is None:
            await o._artifact_transition(
                run_id, RunStage.G0,
                "MA3 가설별 원문 심사 요청 기록", "ma3_admission_started", "ma3.admission",
                attempt_id, "MA3AdmissionAttempt", {"binding": binding},
            )
        async def load_review(review_key):
            artifact_key = digest({"run_id": run_id, "review_key": review_key})
            saved = await o.store.read_artifact(run_id, "ma3_review_" + artifact_key,
                                               "MA3HypothesisReviewResult")
            if saved is not None:
                if saved.get("review_key") != review_key or saved.get("payload_sha256") != digest(saved["payload"]):
                    raise ValueError("MA3_HYPOTHESIS_REVIEW_BINDING_MISMATCH")
                return saved["payload"]
            intent = await o.store.read_artifact(run_id, "ma3_dispatch_" + artifact_key,
                                                "MA3HypothesisReviewAttempt")
            if intent is not None:
                if intent.get("review_key") != review_key:
                    raise ValueError("MA3_HYPOTHESIS_DISPATCH_BINDING_MISMATCH")
                return {"ambiguous": True}
            return None

        async def before_dispatch(review_key):
            artifact_key = digest({"run_id": run_id, "review_key": review_key})
            await o._artifact_transition(
                run_id, RunStage.G0, "MA3 개별 가설 유료 심사 요청 기록",
                "ma3_hypothesis_review_started", "ma3.admission",
                "ma3_dispatch_" + artifact_key, "MA3HypothesisReviewAttempt",
                {"review_key": review_key},
            )

        async def save_review(review_key, payload):
            artifact_key = digest({"run_id": run_id, "review_key": review_key})
            await o._artifact_transition(
                run_id, RunStage.G0, "MA3 개별 가설 심사 결과 보존",
                "ma3_hypothesis_review_completed", "ma3.admission",
                "ma3_review_" + artifact_key, "MA3HypothesisReviewResult",
                {"review_key": review_key, "payload": payload, "payload_sha256": digest(payload)},
            )

        provider = o.ma3.reviewer_provider if o.settings.ma1_provisional_admission_enabled else None
        approved, report = await assess_provisional(
            brief, packet, provider=provider, model=o.ma3.reviewer_model,
            on_usage=lambda usage: o._record_llm_usage(run_id, usage),
            before_review=preflight,
            load_review=load_review, before_dispatch=before_dispatch, save_review=save_review,
        )
        payload = {"binding": binding, "packet": approved.model_dump(mode="json")}
        payload["packet_sha256"] = digest(payload["packet"])
        await o._artifact_transition(
            run_id, RunStage.G0, "MA3 가설별 원문 심사 결과 보존",
            "ma3_admission_completed", "ma3.admission", result_id, "MA3AdmissionResult", payload,
        )
    report = approved.metadata.get("ma3_admission_report", {})
    await o._artifact_transition(
        run_id, RunStage.G0,
        f"MA3 가설 심사 · 제안 {len(packet.provisional_hypotheses)}개 / 진행 가능 {len(approved.hypotheses)}개",
        "hypothesis_admission_decided", "ma3.admission", approved.packet_id,
        "HypothesisPacket", approved,
        metadata={"submission_packet_id": packet.packet_id,
                  "approved_count": len(approved.hypotheses),
                  "submitted_count": report.get("submitted_count", len(packet.provisional_hypotheses))},
    )
    return approved
