"""Read-only result readiness and explicit, non-final research selection.

This policy does not change MA3, HumanGateDecision, action completion, or any
scientific artifact. Only narrowly named research limitations are acceptable.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

from drugpilot.domain.models import (
    ExecutionEvent, GateVerdictCode, ResearchCandidateSelection,
    ResearchSelectionLimitation, RunStage, Severity, utc_now,
)
from drugpilot.orchestration.human_review import review_mutation
from drugpilot.storage import CheckpointIntegrityError
from drugpilot.tools.candidate_policy import screen_forbidden_drug, screen_reserved_control

POLICY_VERSION = "research-selection-v1"
ALLOWED_LIMITATIONS = {
    "ASSAY_VARIANT_CONTEXT_UNKNOWN": Severity.LOW,
    "ASSAY_VARIANT_CLASS_ONLY": Severity.LOW,
    "ASSAY_VARIANT_PARTIAL_OVERLAP": Severity.LOW,
    "ASSAY_VARIANT_ANNOTATION_ONLY": Severity.LOW,
    "ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION": Severity.MEDIUM,
}


def _context_digest(snapshot):
    return sha256(json.dumps({name: getattr(snapshot, name).model_dump(mode="json")
        if getattr(snapshot, name) else None for name in ("brief", "hypothesis_packet",
            "g0_verdict", "plan_spec", "candidate_packet", "g1_verdict")},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def eligible_candidate_ids(snapshot) -> set[str]:
    """The original APPROVE screening policy, shared without weakening it."""
    packet = snapshot.candidate_packet
    route_decision = packet.metadata.get("route_decision", {}) if packet else {}
    selected_route = route_decision.get("selected_route") if isinstance(route_decision, dict) else None
    portfolio_routes = {
        summary.target_key: summary.route.value
        for summary in (packet.target_executions if packet else [])
        if summary.status == "completed" and summary.route is not None
    }
    eligible = set()
    for candidate in packet.candidates if packet else []:
        reserved = screen_reserved_control(candidate_name=candidate.name,
            metadata=candidate.metadata,
            reserved_controls=snapshot.brief.constraints.reserved_control_compounds)
        forbidden = screen_forbidden_drug(candidate_name=candidate.name,
            metadata=candidate.metadata,
            forbidden_names=snapshot.brief.constraints.forbidden_drugs,
            forbidden_ids=snapshot.brief.constraints.forbidden_drug_ids,
            forbidden_classes=snapshot.brief.constraints.forbidden_drug_classes)
        restricted = candidate.metadata.get("restricted_safety_screen")
        disposition = restricted.get("disposition") if isinstance(restricted, dict) else None
        target_key = str(candidate.metadata.get("portfolio_target_key") or "")
        authorized = (candidate.route.value == portfolio_routes.get(target_key)
            if packet.portfolio_enabled else candidate.route.value == selected_route)
        rank = candidate.portfolio_rank if packet.portfolio_enabled else candidate.rank
        if (candidate.status != "excluded" and rank is not None
                and candidate.metadata.get("policy_eligible") is not False
                and not reserved.excluded and not forbidden.excluded
                and disposition not in {"excluded", "unresolved"} and authorized):
            eligible.add(candidate.candidate_id)
    return eligible


def _completed_counter_search(snapshot, finding):
    if (finding.code != "COUNTER_EVIDENCE_NOT_FOUND" or not finding.deterministic
            or finding.severity is not Severity.MEDIUM or finding.gate.value != "G0"
            or finding.category != "evidence_provenance" or snapshot.hypothesis_packet is None):
        return False
    hypothesis = next((h for h in snapshot.hypothesis_packet.hypotheses
        if h.hypothesis_id == finding.artifact_id), None)
    summary = hypothesis.target.metadata.get("counter_evidence_search") if hypothesis else None
    if (not isinstance(summary, dict) or summary.get("status") not in {"completed_empty", "completed_with_results"}
            or summary.get("separate_query") is not True or summary.get("fixture_used") is not False):
        return False
    if summary.get("raw_search_status", summary["status"]) != summary["status"]:
        return False
    queries, hashes = summary.get("queries"), summary.get("query_sha256s")
    if (not isinstance(queries, list) or not queries or not all(isinstance(q, str) and q.strip() for q in queries)
            or not isinstance(hashes, list) or not all(isinstance(h, str) for h in hashes)
            or sorted(set(hashes)) != sorted({sha256(q.encode("utf-8")).hexdigest() for q in queries})):
        return False
    if summary.get("query") is not None and (summary["query"] not in queries
            or summary.get("query_sha256") != sha256(summary["query"].encode("utf-8")).hexdigest()):
        return False
    provenance = summary.get("http_call_provenance")
    supported_endpoints = {
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi": "PubMed",
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search": "Europe PMC",
    }
    sources = summary.get("sources")
    return bool(isinstance(provenance, dict) and provenance.get("status") == "completed"
        and type(provenance.get("status_code")) is int and provenance["status_code"] == 200
        and provenance.get("fixture") is False and provenance.get("method") == "GET"
        and isinstance(provenance.get("endpoint"), str) and provenance["endpoint"] in supported_endpoints
        and isinstance(sources, list) and supported_endpoints[provenance["endpoint"]] in sources
        and all(source in {"PubMed", "Europe PMC"} for source in sources)
        and all(isinstance(provenance.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", provenance[key])
            for key in ("response_sha256", "parameter_sha256")))


def _finding_view(finding, *, hypothesis_scoped=False):
    return ResearchSelectionLimitation(finding_id=finding.finding_id,
        code=finding.code, severity=finding.severity, message=finding.message,
        candidate_id=None if hypothesis_scoped else finding.artifact_id,
        hypothesis_id=finding.artifact_id if hypothesis_scoped else None).model_dump(mode="json")


def project_readiness(snapshot, *, busy=False, ledger_valid=False, binding_valid=False):
    reasons, limitations, advisories = [], [], []
    if snapshot.stage in {RunStage.CANCEL_REQUESTED, RunStage.CANCELLED}:
        reasons.append("RUN_CANCELLED")
    if busy:
        reasons.append("RUN_ACTIVE")
    if snapshot.stage is not RunStage.HUMAN_GATE or snapshot.human_decision is not None:
        reasons.append("HUMAN_GATE_CLOSED")
    if any(item.status == "running" for item in snapshot.g1_reaudits):
        reasons.append("REAUDIT_RECOVERY_REQUIRED")
    if not ledger_valid:
        reasons.append("LEDGER_INTEGRITY_FAILED")
    if not binding_valid:
        reasons.append("CURRENT_ARTIFACT_BINDING_INVALID")
    if not snapshot.g0_verdict or snapshot.g0_verdict.verdict is not GateVerdictCode.PASS:
        reasons.append("G0_PASS_REQUIRED")
    if not snapshot.g1_verdict or snapshot.g1_verdict.verdict is not GateVerdictCode.PASS:
        reasons.append("G1_PASS_REQUIRED")
    if not (snapshot.hypothesis_packet and snapshot.plan_spec and snapshot.candidate_packet):
        reasons.append("ANALYSIS_ARTIFACTS_REQUIRED")
    if snapshot.error:
        reasons.append("RUN_ERROR_UNRESOLVED")
    candidate_ids = {c.candidate_id for c in snapshot.candidate_packet.candidates} if snapshot.candidate_packet else set()
    for verdict in (snapshot.g0_verdict, snapshot.g1_verdict):
        if verdict is None:
            continue
        if any(not validation.passed for validation in verdict.validations):
            reasons.append("AUDIT_VALIDATION_FAILED")
        for finding in verdict.findings:
            if finding.resolved:
                continue
            if (finding.severity in {Severity.HIGH, Severity.CRITICAL}
                    or finding.remediation_kind == "integrity_block"):
                reasons.append("BLOCKING_FINDING:" + finding.code)
            elif not finding.deterministic and finding.confirmation_status == "unconfirmed":
                advisories.append(_finding_view(finding))
            elif finding.severity is Severity.INFO:
                advisories.append(_finding_view(finding))
            elif verdict is snapshot.g0_verdict and _completed_counter_search(snapshot, finding):
                limitations.append(_finding_view(finding, hypothesis_scoped=True))
            elif (verdict is snapshot.g1_verdict and finding.deterministic
                    and ALLOWED_LIMITATIONS.get(finding.code) is finding.severity
                    and finding.artifact_id in candidate_ids):
                limitations.append(_finding_view(finding))
            else:
                reasons.append("UNSUPPORTED_FINDING:" + finding.code)
    eligible = eligible_candidate_ids(snapshot)
    if not eligible:
        reasons.append("NO_ELIGIBLE_CANDIDATES")
    record = snapshot.research_candidate_selection
    state = "not_selected"
    if record:
        current = (snapshot.g1_verdict and snapshot.candidate_packet
            and record.expected_g1_verdict_id == snapshot.g1_verdict.verdict_id
            and record.g1_verdict_sha256 == sha256(snapshot.g1_verdict.model_dump_json().encode()).hexdigest()
            and record.candidate_packet_sha256 == sha256(snapshot.candidate_packet.model_dump_json().encode()).hexdigest()
            and record.context_sha256 == _context_digest(snapshot))
        state = "selected" if current and binding_valid and ledger_valid else "stale"
        reasons.append("RESEARCH_SELECTION_ALREADY_RECORDED")
    complete = bool(snapshot.g1_verdict and snapshot.candidate_packet and not busy
        and snapshot.stage in {RunStage.HUMAN_GATE, RunStage.COMPLETED, RunStage.ABSTAINED, RunStage.BLOCKED})
    return {"run_id": snapshot.run_id, "analysis_complete": complete,
        "analysis_state": "complete" if complete else "incomplete", "ledger_valid": ledger_valid,
        "research_selection": {"allowed": not reasons, "state": state,
            "policy_version": POLICY_VERSION,
            "expected_g1_verdict_id": snapshot.g1_verdict.verdict_id if snapshot.g1_verdict else None,
            "expected_candidate_packet_sha256": sha256(snapshot.candidate_packet.model_dump_json().encode()).hexdigest() if snapshot.candidate_packet else None,
            "eligible_candidate_ids": sorted(eligible), "blocking_reasons": list(dict.fromkeys(reasons)),
            "limitations": limitations, "advisories": advisories,
            "record": record.model_dump(mode="json") if record else None},
        "experimental_validation": {"state": "not_verified", "verified": False,
            "reason": "Research selection, audit PASS, uploaded results and historical APPROVE are not typed experimental-validation proof."}}


async def _binding_valid(store, snapshot):
    """Verify current snapshot artifacts, not only the historical ledger chain."""
    for name, id_field in (("brief", "brief_id"), ("hypothesis_packet", "packet_id"),
            ("g0_verdict", "verdict_id"), ("plan_spec", "plan_id"),
            ("candidate_packet", "packet_id"), ("g1_verdict", "verdict_id"),
            ("research_candidate_selection", "selection_id")):
        artifact = getattr(snapshot, name)
        if artifact is None:
            if name == "research_candidate_selection":
                continue
            return False
        stored = await store.read_artifact(snapshot.run_id, getattr(artifact, id_field), type(artifact).__name__)
        if stored is None or type(artifact).model_validate(stored) != artifact:
            return False
    # A snapshot cannot revive an older PASS while a newer G0/G1 exists.
    latest, positions = {}, {}
    for index, artifact in enumerate(await store.artifact_manifest(snapshot.run_id)):
        positions[artifact["artifact_id"]] = index
        latest[artifact["artifact_type"]] = artifact["artifact_id"]
        if artifact["artifact_type"] == "GateVerdict":
            verdict = await store.read_artifact(snapshot.run_id, artifact["artifact_id"], "GateVerdict")
            latest[verdict["gate"]] = verdict["verdict_id"]
    return (latest.get("G0") == snapshot.g0_verdict.verdict_id
        and latest.get("G1") == snapshot.g1_verdict.verdict_id
        and latest.get("ResearchBrief") == snapshot.brief.brief_id
        and latest.get("HypothesisPacket") == snapshot.hypothesis_packet.packet_id
        and latest.get("PlanSpec") == snapshot.plan_spec.plan_id
        and latest.get("CandidatePacket") == snapshot.candidate_packet.packet_id
        and positions[snapshot.candidate_packet.packet_id] < positions[snapshot.g1_verdict.verdict_id])


class ResearchSelectionService:
    def __init__(self, orchestrator):
        self.o = orchestrator

    async def readiness(self, run_id):
        snapshot = await self.o.store.get_snapshot(run_id)
        if snapshot is None:
            raise KeyError(run_id)
        task = self.o._tasks.get(run_id)
        lease = await self.o.store.live_execution_lease_status(run_id, now=utc_now())
        busy = bool((task and not task.done()) or (lease and lease["state"] == "active"))
        return await self._project(snapshot, busy=busy)

    async def _project(self, snapshot, *, busy=False):
        valid = await self.o.store.verify_ledger(snapshot.run_id)
        try:
            bound = valid and await _binding_valid(self.o.store, snapshot)
        except (CheckpointIntegrityError, ValueError, KeyError, TypeError):
            bound = False
        return project_readiness(snapshot, busy=busy, ledger_valid=valid, binding_valid=bound)

    async def select(self, run_id, *, researcher, rationale, selected_candidate_ids,
            expected_g1_verdict_id, expected_candidate_packet_sha256,
            limitations_acknowledged, policy_version):
        if (not researcher.strip() or not rationale.strip() or not selected_candidate_ids
                or len(selected_candidate_ids) != len(set(selected_candidate_ids))):
            raise ValueError("RESEARCH_SELECTION_INPUT_REQUIRED")
        if limitations_acknowledged is not True or policy_version != POLICY_VERSION:
            raise ValueError("RESEARCH_LIMITATIONS_ACKNOWLEDGEMENT_REQUIRED")
        async with review_mutation(self.o, run_id) as token:
            snapshot = await self.o.store.get_snapshot(run_id)
            if not snapshot.g1_verdict or snapshot.g1_verdict.verdict_id != expected_g1_verdict_id:
                raise ValueError("STALE_G1_VERDICT")
            if (not snapshot.candidate_packet or expected_candidate_packet_sha256
                    != sha256(snapshot.candidate_packet.model_dump_json().encode()).hexdigest()):
                raise ValueError("STALE_CANDIDATE_PACKET")
            readiness = (await self._project(snapshot))["research_selection"]
            if not readiness["allowed"]:
                raise ValueError("; ".join(readiness["blocking_reasons"]))
            if not set(selected_candidate_ids) <= set(readiness["eligible_candidate_ids"]):
                raise ValueError("INELIGIBLE_OR_UNKNOWN_CANDIDATE")
            record = ResearchCandidateSelection(run_id=run_id, researcher=researcher.strip(),
                rationale=rationale.strip(), selected_candidate_ids=tuple(selected_candidate_ids),
                expected_g1_verdict_id=expected_g1_verdict_id,
                g1_verdict_sha256=sha256(snapshot.g1_verdict.model_dump_json().encode()).hexdigest(),
                candidate_packet_id=snapshot.candidate_packet.packet_id,
                candidate_packet_sha256=sha256(snapshot.candidate_packet.model_dump_json().encode()).hexdigest(),
                context_sha256=_context_digest(snapshot),
                limitations_acknowledged=True, limitations=tuple(readiness["limitations"]),
                advisories=tuple(readiness["advisories"]))
            snapshot.research_candidate_selection = record
            snapshot.status_message = "연구 후보 채택 기록 완료 · 실험 검증 미완료 · 기존 한계와 후속 검토 유지"
            event = ExecutionEvent(run_id=run_id, stage=snapshot.stage,
                event_type="research_candidate_selected", node="human_gate.research_selection",
                message=snapshot.status_message, artifact_ids=[record.selection_id],
                metadata={"artifact_sha256": {record.selection_id: sha256(record.model_dump_json().encode()).hexdigest()},
                    "g1_verdict_id": expected_g1_verdict_id, "policy_version": POLICY_VERSION,
                    "experimental_validation": "not_verified"})
            snapshot.events.append(event)
            await self.o.store.commit_review_update(snapshot,
                artifacts=[(record.selection_id, "ResearchCandidateSelection", record)],
                events=[event], lease_token=token)
            await self.o.broker.publish(event)
            return snapshot
