from __future__ import annotations

import re
from collections import defaultdict

from drugpilot.domain.models import (
    AuditFinding,
    CandidateCard,
    ExperimentRequest,
    GateName,
    GateVerdict,
    GateVerdictCode,
    NextActionItem,
    NextActionPacket,
    ResearchBrief,
    RunStage,
    Severity,
    WorkflowCheckpoint,
)


_EXPERIMENT_TERMS = (
    "experiment",
    "experimental",
    "confirmatory",
    "assay",
    "herg",
    "dili",
    "ames",
    "hepatocyte",
    "viability",
    "pstat",
)
_COMPUTATIONAL_TERMS = (
    "unauthorized",
    "schema",
    "rank",
    "ranking",
    "score",
    "proxy",
    "calibrat",
    "contract",
    "classification",
    "implementation",
)
_DATA_TERMS = (
    "evidence",
    "source",
    "publication",
    "full text",
    "fulltext",
    "direct evidence",
    "missing data",
)
_TOOL_TERMS = ("not configured", "tool unavailable", "backend unavailable", "api unavailable")
_RESOURCE_TERMS = ("budget", "token", "cost", "quota", "rate limit", "timeout")
_DECISION_TERMS = ("researcher", "policy decision", "threshold", "ambiguous")
_INTEGRITY_TERMS = (
    "fabricat",
    "synthetic result",
    "hash mismatch",
    "tamper",
    "provenance corruption",
    "immutable artifact",
)


def _text(finding: AuditFinding) -> str:
    return f"{finding.code} {finding.message}".casefold()


def _kind_for(finding: AuditFinding) -> str:
    text = _text(finding)
    # Integrity findings cannot be downgraded through a model's routing hint.
    if any(term in text for term in _INTEGRITY_TERMS):
        return "integrity_block"
    if finding.remediation_kind is not None:
        return finding.remediation_kind
    # These are verification tasks.  An unspecified assay variant is first a
    # source-context gap; do not prescribe a new experiment from that fact alone.
    if finding.code in {"ASSAY_VARIANT_CONTEXT_UNKNOWN", "ASSAY_VARIANT_CLASS_ONLY",
                         "ASSAY_VARIANT_PARTIAL_OVERLAP", "ASSAY_VARIANT_ANNOTATION_ONLY"}:
        return "data_required"
    if finding.code in {
        "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_ATTESTATION_INVALID",
        "REVIEWER_KNOWN_DRUG_COUNT_INCLUDES_NON_ATOMIC_INTERVENTION_LABELS",
    }:
        return "computational_rework"
    text = _text(finding)
    if any(term in text for term in _INTEGRITY_TERMS):
        return "integrity_block"
    # Contract/schema/ranking defects stay computational even when their
    # explanatory message mentions an assay or exposure.  Otherwise a code
    # defect could incorrectly ask a laboratory to solve a software problem.
    if any(term in text for term in _COMPUTATIONAL_TERMS):
        return "computational_rework"
    if any(term in text for term in _TOOL_TERMS):
        return "tool_configuration"
    if any(term in text for term in _RESOURCE_TERMS):
        return "resource_required"
    # Legacy untyped reviewer prose is not authorization for a new experiment.
    # Treat source/model interpretation concerns as reviewable data gaps, even
    # if the prose mentions the original study's experiment or viability assay.
    if not finding.deterministic or finding.code.startswith("REVIEWER_"):
        if finding.category in {"schema", "model_appropriateness"}:
            return "computational_rework"
        if finding.category in {"scientific", "evidence_provenance"}:
            return "data_required"
    if any(term in text for term in _EXPERIMENT_TERMS):
        return "experiment_required"
    if any(term in text for term in _DECISION_TERMS):
        return "researcher_decision"
    if any(term in text for term in _DATA_TERMS):
        return "data_required"
    if finding.category in {"schema", "model_appropriateness"}:
        return "computational_rework"
    if finding.category == "evidence_provenance":
        return "data_required"
    if finding.category == "safety":
        return "experiment_required"
    return "operator_review"


def _owner_for(kind: str) -> str:
    return {
        "computational_rework": "software_team",
        "experiment_required": "experimental_team",
        "data_required": "data_curator",
        "tool_configuration": "platform_operator",
        "researcher_decision": "research_lead",
        "resource_required": "platform_operator",
        "integrity_block": "integrity_officer",
        "operator_review": "research_lead",
    }[kind]


def _candidate_ids_for(
    finding: AuditFinding,
    candidates: list[CandidateCard],
) -> list[str]:
    if finding.artifact_id:
        exact = [item.candidate_id for item in candidates if item.candidate_id == finding.artifact_id]
        if exact:
            return exact
    text = _text(finding)
    named = [item.candidate_id for item in candidates if item.name.casefold() in text]
    return named


def _assays_for(findings: list[AuditFinding]) -> list[str]:
    text = " ".join(_text(item) for item in findings)
    assays: list[str] = []
    if "dili" in text or "hepat" in text:
        assays.append("Human-relevant hepatotoxicity confirmation assay")
    if "herg" in text:
        assays.append("hERG concentration-response electrophysiology assay")
    if "ames" in text:
        assays.append("Ames mutagenicity assay with appropriate controls")
    if "pstat" in text:
        assays.append("Target-pathway pSTAT pharmacodynamic assay")
    if "viability" in text:
        assays.append("Disease-relevant cell viability concentration-response assay")
    if not assays:
        assays.append("Independent assay selected to resolve the cited audit finding")
    return list(dict.fromkeys(assays))


def _action_contract(kind: str, finding: AuditFinding) -> tuple[list[str], list[str]]:
    contracts = {
        "computational_rework": (
            ["Affected code/configuration", "Existing immutable evidence artifacts"],
            [
                "Generic fix is covered by regression tests",
                "Re-audit passes without changing or weakening scientific gates",
            ],
        ),
        "experiment_required": (
            ["Real assay results with controls, replicates, units, uncertainty, and source hash"],
            [
                "Results are independently traceable and schema-valid",
                "G1 is rerun using the new measurements without synthetic substitution",
            ],
        ),
        "data_required": (
            ["Source record or full-text evidence with stable identifier and provenance"],
            ["Required claim is supported by a directly traceable source and re-audited"],
        ),
        "tool_configuration": (
            ["Configured executable/API and versioned readiness evidence"],
            ["Readiness check passes and the required operation completes without fixtures"],
        ),
        "researcher_decision": (
            ["Researcher-authored policy or acceptance criterion with rationale"],
            ["Decision is signed, persisted, and applied prospectively"],
        ),
        "resource_required": (
            ["Explicitly approved budget/quota/time-limit change"],
            ["Preflight succeeds under the approved limit and usage remains auditable"],
        ),
        "integrity_block": (
            ["Independent provenance and artifact-integrity investigation"],
            ["All affected hashes and lineage are verified before any scientific resume"],
        ),
        "operator_review": (
            ["Operator classification and documented remediation plan"],
            ["The finding is resolved or escalated to a typed action before resume"],
        ),
    }
    required_inputs, completion = contracts[kind]
    if finding.artifact_id:
        required_inputs = [*required_inputs, f"Affected artifact: {finding.artifact_id}"]
    return required_inputs, completion


def _title_for(kind: str, finding: AuditFinding) -> str:
    label = {
        "computational_rework": "코드·계약 보정",
        "experiment_required": "추가 실험",
        "data_required": "근거 보강",
        "tool_configuration": "도구 설정",
        "researcher_decision": "연구자 결정",
        "resource_required": "자원 승인",
        "integrity_block": "무결성 조사",
        "operator_review": "운영자 분류",
    }[kind]
    concise_code = re.sub(r"_+", " ", finding.code).strip()
    return f"{label}: {concise_code}"[:300]


def _disposition(kinds: set[str]) -> str:
    if len(kinds) > 1:
        return "MIXED_ACTIONS_REQUIRED"
    kind = next(iter(kinds))
    return {
        "computational_rework": "COMPUTATIONAL_REWORK_REQUIRED",
        "experiment_required": "EXPERIMENT_REQUIRED",
        "data_required": "DATA_OR_TOOL_REQUIRED",
        "tool_configuration": "DATA_OR_TOOL_REQUIRED",
        "researcher_decision": "RESEARCHER_DECISION_REQUIRED",
        "resource_required": "RESEARCHER_DECISION_REQUIRED",
        "integrity_block": "INTEGRITY_BLOCK",
        "operator_review": "RESEARCHER_DECISION_REQUIRED",
    }[kind]


def build_next_action_packet(
    *,
    run_id: str,
    trigger_stage: RunStage,
    brief: ResearchBrief,
    verdict: GateVerdict | None,
    candidates: list[CandidateCard],
    checkpoint: WorkflowCheckpoint | None,
    terminal_message: str,
) -> NextActionPacket | None:
    """Create a deterministic, evidence-preserving handoff from audit findings."""

    findings = [item for item in (verdict.findings if verdict else []) if not item.resolved
                and (item.deterministic or item.confirmation_status is None)]
    if not findings and verdict is not None and verdict.verdict is GateVerdictCode.PASS:
        # Advisory reviewer text is displayed with the audit, never converted
        # into mandatory experiment requests or a new automatic rework loop.
        return None
    if not findings:
        findings = [
            AuditFinding(
                gate=verdict.gate if verdict else GateName.G1,
                category="execution",
                severity=Severity.HIGH,
                code="UNCLASSIFIED_WORKFLOW_STOP",
                message=terminal_message,
            )
        ]

    actions: list[NextActionItem] = []
    experiment_findings: dict[str | None, list[AuditFinding]] = defaultdict(list)
    candidate_by_id = {item.candidate_id: item for item in candidates}
    kinds: set[str] = set()

    for finding in findings:
        kind = _kind_for(finding)
        kinds.add(kind)
        candidate_ids = _candidate_ids_for(finding, candidates)
        required_inputs, completion = _action_contract(kind, finding)
        actions.append(
            NextActionItem(
                kind=kind,
                title=_title_for(kind, finding),
                rationale=finding.message,
                owner=_owner_for(kind),
                priority=finding.severity,
                finding_ids=[finding.finding_id],
                candidate_ids=candidate_ids,
                required_inputs=required_inputs,
                completion_criteria=completion,
            )
        )
        if kind == "experiment_required":
            for candidate_id in candidate_ids or [None]:
                experiment_findings[candidate_id].append(finding)

    experiment_requests: list[ExperimentRequest] = []
    for candidate_id, linked_findings in experiment_findings.items():
        candidate = candidate_by_id.get(candidate_id) if candidate_id else None
        candidate_label = candidate.name if candidate else "해당 후보"
        experiment_requests.append(
            ExperimentRequest(
                candidate_id=candidate_id,
                candidate_name=candidate.name if candidate else None,
                title=f"{candidate_label} 감사 해소 실험",
                objective="; ".join(item.message for item in linked_findings),
                source_finding_ids=[item.finding_id for item in linked_findings],
                required_assays=_assays_for(linked_findings),
                researcher_authored_criteria=[
                    "Acceptance thresholds must be authored and approved by the researcher before results are interpreted.",
                    f"Research goal context: {brief.research_goal}",
                ],
            )
        )

    preserve_ids = [
        item
        for item in (
            verdict.verdict_id if verdict else None,
            checkpoint.checkpoint_id if checkpoint else None,
        )
        if item
    ]
    reentry = [
        "Partial re-audit is allowed after evidence or code changes; unresolved blocking findings remain open.",
        "Any experimental results are real, independently sourced, schema-valid, and hash-addressed.",
        "The unchanged scientific gates are rerun from the recorded checkpoint or an explicitly newer checkpoint.",
        "A fresh G1 PASS is required before candidate APPROVE becomes available.",
    ]
    return NextActionPacket(
        classifier_version="next-actions-r3",
        run_id=run_id,
        trigger_stage=trigger_stage,
        disposition=_disposition(kinds),
        summary=(
            f"자동 감사에서 미해결 항목 {len(findings)}건이 확인되었습니다. "
            "항목별 자료를 제출하거나 현재 자료를 재감사할 수 있습니다. 후보 승인은 G1 PASS 후 가능합니다."
        ),
        actions=actions,
        experiment_requests=experiment_requests,
        resume_checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
        resume_from_node="g1" if checkpoint and checkpoint.completed_node == "g1" else None,
        resume_eligible=False,
        reentry_conditions=reentry,
        preserve_artifact_ids=preserve_ids,
    )
