from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from drugpilot.domain.models import (
    CandidateCard,
    CandidatePacket,
    GateVerdict,
    GateVerdictCode,
    Hypothesis,
    HypothesisPacket,
    PlanSpec,
)


def interaction(
    sender: str,
    receiver: str,
    kind: str,
    summary: str,
    *,
    basis: Iterable[str] = (),
    decision: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    evidence_ids: Iterable[str] = (),
    artifact_ids: Iterable[str] = (),
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, UI-safe explanation of an agent hand-off.

    This is an evidence and decision trace, not a transcript of private model
    reasoning.  It deliberately stores short summaries and stable identifiers
    instead of prompts or free-form hidden deliberation.
    """

    payload: dict[str, Any] = {
        "sender": sender,
        "receiver": receiver,
        "kind": kind,
        "summary": summary,
        "basis": [item for item in basis if item][:8],
        "evidence_ids": list(dict.fromkeys(evidence_ids))[:20],
        "artifact_ids": list(dict.fromkeys(artifact_ids))[:10],
        "explanation_scope": "evidence_and_decision_summary",
    }
    if decision:
        payload["decision"] = decision
    if backend:
        payload["backend"] = backend
    if model:
        payload["model"] = model
    if details:
        payload["details"] = details
    return payload


def hypothesis_handoff(packet: HypothesisPacket) -> dict[str, Any]:
    ranked = sorted(
        packet.hypotheses,
        key=lambda item: (item.track.value, item.rank or 999, -item.score),
    )
    basis = [_hypothesis_basis(item) for item in ranked[:6]]
    evidence_ids = [
        evidence.evidence_id
        for hypothesis in ranked
        for evidence in hypothesis.evidence
    ]
    tracks = sorted({item.track.value for item in ranked})
    return interaction(
        "MA1",
        "MA3 · G0",
        "artifact_handoff",
        f"{len(ranked)}개 가설을 {', '.join(tracks) or '단일'} 트랙으로 구성해 독립 감사에 전달",
        basis=basis,
        evidence_ids=evidence_ids,
        artifact_ids=[packet.packet_id],
        details={
            "hypothesis_count": len(ranked),
            "excluded_count": len(packet.excluded),
            "iterations": packet.iterations,
        },
    )


def plan_handoff(plan: PlanSpec) -> dict[str, Any]:
    route_tasks: dict[str, int] = {}
    for task in plan.tasks:
        route = task.route.value if task.route else "shared"
        route_tasks[route] = route_tasks.get(route, 0) + 1
    tools = sorted(
        {
            assignment.tool_name
            for task in plan.tasks
            for assignment in task.tools
        }
    )
    basis = [
        f"{len(plan.tasks)}개 작업 · 승인 도구 {len(tools)}종",
        f"선택 경로 후보 {plan.candidates_per_route}건 · 토큰 상한 {plan.total_token_budget:,}",
        f"중단 조건 {len(plan.stop_conditions)}개 · 계획 개정 {plan.revision}",
    ]
    return interaction(
        "Planner",
        "MA2",
        "plan_dispatch",
        "G0 통과 가설을 근거 기반 단일 경로 선택과 제한된 실행 계획으로 변환",
        basis=basis,
        artifact_ids=[plan.plan_id, plan.hypothesis_packet_id],
        details={
            "route_task_counts": route_tasks,
            "approved_tools": tools,
            "stop_conditions": plan.stop_conditions[:8],
        },
    )


def candidate_handoff(packet: CandidatePacket) -> dict[str, Any]:
    eligible = [item for item in packet.candidates if item.status != "excluded"]
    route_counts: dict[str, int] = {}
    for candidate in packet.candidates:
        route_counts[candidate.route.value] = route_counts.get(candidate.route.value, 0) + 1
    top_by_route: list[CandidateCard] = []
    for route in sorted(route_counts):
        route_items = [item for item in eligible if item.route.value == route]
        if route_items:
            top_by_route.append(max(route_items, key=lambda item: item.score))
    basis = [_candidate_basis(item) for item in top_by_route]
    basis.append(
        f"도구 추적 {len(packet.tool_calls)}건 · 제외 후보 {len(packet.candidates) - len(eligible)}건"
    )
    evidence_ids = [
        evidence.evidence_id
        for candidate in packet.candidates
        for evidence in candidate.evidence
    ]
    route_decision = packet.metadata.get("route_decision", {})
    selected_route = (
        route_decision.get("selected_route")
        if isinstance(route_decision, dict)
        else None
    )
    return interaction(
        "MA2 · Judge",
        "MA3 · G1",
        "artifact_handoff",
        f"근거로 선택한 {selected_route or '단일'} 경로에서 {len(packet.candidates)}개 후보를 계산·순위화해 과정 감사에 전달",
        basis=basis,
        evidence_ids=evidence_ids,
        artifact_ids=[packet.packet_id, packet.plan_id],
        details={
            "route_counts": route_counts,
            "route_decision": route_decision,
            "eligible_count": len(eligible),
            "generation_iterations": packet.generation_iterations,
        },
    )


def audit_handoff(verdict: GateVerdict) -> dict[str, Any]:
    destination = {
        GateVerdictCode.PASS: "Planner" if verdict.gate.value == "G0" else "Human Gate",
        GateVerdictCode.REVISE: "MA1" if verdict.gate.value == "G0" else "MA2",
        GateVerdictCode.REPLAN: "Planner",
        GateVerdictCode.ABSTAIN: "Human Review",
        GateVerdictCode.BLOCK: "Stop",
    }[verdict.verdict]
    failed = [item for item in verdict.validations if not item.passed]
    high_findings = [
        item for item in verdict.findings if item.severity.value in {"CRITICAL", "HIGH"}
    ]
    basis = [
        verdict.rationale,
        f"검증 {len(verdict.validations)}개 중 실패 {len(failed)}개",
        f"발견사항 {len(verdict.findings)}개 중 High 이상 {len(high_findings)}개",
    ]
    if verdict.rework_order:
        basis.append(
            f"{verdict.rework_order.target_stage}로 제한적 환류 · "
            + "; ".join(verdict.rework_order.required_actions[:3])
        )
    return interaction(
        f"MA3 · {verdict.gate.value}",
        destination,
        "audit_decision",
        f"독립 감사 결과 {verdict.verdict.value} 판정",
        basis=basis,
        decision=verdict.verdict.value,
        model=verdict.reviewer_model,
        artifact_ids=[verdict.verdict_id],
        details={
            "reviewer_invoked": verdict.reviewer_invoked,
            "meta_critic_invoked": verdict.meta_critic_invoked,
            "failed_validations": [item.name for item in failed[:8]],
            "finding_codes": [item.code for item in verdict.findings[:12]],
        },
    )


def _hypothesis_basis(hypothesis: Hypothesis) -> str:
    support = sum(item.supports == "support" for item in hypothesis.evidence)
    counter = sum(item.supports == "counter" for item in hypothesis.evidence)
    context = sum(item.supports == "context" for item in hypothesis.evidence)
    return (
        f"{hypothesis.target.symbol} ({hypothesis.track.value}) · 점수 {hypothesis.score:.2f} · "
        f"지지 {support}/반증 {counter}/맥락 {context} · 주장 {len(hypothesis.claims)}개"
    )


def _candidate_basis(candidate: CandidateCard) -> str:
    metric = candidate.metrics
    pieces = [
        f"{candidate.name} ({candidate.route.value})",
        f"점수 {candidate.score:.2f}",
        f"상태 {candidate.status}",
    ]
    if candidate.activities:
        activity = candidate.activities[0]
        pieces.append(
            f"{activity.endpoint} {activity.relation} {activity.value:g} {activity.unit}"
        )
    if metric.docking_score_kcal_mol is not None:
        pieces.append(f"도킹 {metric.docking_score_kcal_mol:g} kcal/mol")
    pieces.append(f"안전성 경고 {len(candidate.safety_flags)}개")
    return " · ".join(pieces)
