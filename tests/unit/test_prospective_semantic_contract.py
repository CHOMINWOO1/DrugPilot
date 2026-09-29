from copy import deepcopy
from hashlib import sha256
import json

import pytest

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.domain.models import GateName, ReworkOrder, ScientificBackendKind, Severity
from drugpilot.orchestration.next_actions import _kind_for
from drugpilot.tools.review_source_enrichment import DOCUMENT_CODES
from drugpilot.tools.assay_variant_alignment import assay_variant_alignment
from test_assay_variant_alignment import _context
from test_ma2_live_execution_contract import _pipeline
from test_ma2_rework_persistence import _live_portfolio, RunLocalLiveGateway, PersistedStructureChemistry, REWORKED_ADMET


RULE = ("각 후보는 개별적으로 평가한다. 최소 3개의 독립 생물학적 반복에서 후보별로 "
        "급성 생존율 감소가 20% 이하인 노출에서 2시간 pSTAT5가 vehicle 대비 50% 이상 "
        "감소하고 72시간 viable-cell count가 vehicle 대비 30% 이상 감소하며 각 방향성 "
        "효과의 95% 신뢰구간이 0을 제외할 때만 지지한다.")


def _setup(tmp_path, brief, rule=RULE):
    hypotheses, plan = _pipeline(tmp_path, brief)
    h = hypotheses.hypotheses[0].model_copy(update={"falsification_test": rule}, deep=True)
    hypotheses = hypotheses.model_copy(update={"hypotheses": [h]}, deep=True)
    packet = MA2Agent().run(plan, hypotheses)
    assert packet.candidates
    return hypotheses, packet


def _rehash(record):
    record.pop("attestation_sha256", None)
    record["attestation_sha256"] = sha256(json.dumps(record, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("rule", [RULE, "A novel unstructured prospective rule."])
def test_unobserved_is_neither_support_nor_refutation_and_source_is_unchanged(tmp_path, brief, rule):
    hypotheses, packet = _setup(tmp_path, brief, rule)
    before = packet.model_dump_json()
    audit = MA3AssuranceAgent()
    findings = audit._audit_prospective_falsification(hypotheses, packet)
    assert findings == [], [f.message for f in findings]
    after = MA2Agent._attest_prospective_falsification_status(packet.candidates, hypotheses.hypotheses[0])
    assert packet.model_dump_json() == before
    for old, new in zip(packet.candidates, after):
        assert old.activities == new.activities and old.evidence == new.evidence
        assert old.score == new.score and old.metrics == new.metrics
        state = new.metadata["prospective_falsification_status"]
        assert state["candidate_id"] == new.candidate_id
        assert state["status"] == "not_evaluated"
        assert state["default_falsification_applied"] is False
        assert state["biological_refutation_established"] is False


@pytest.mark.parametrize("mutation", [
    "status", "experimentally_observed", "default_falsification_applied",
    "biological_refutation_established", "prospective_follow_up_support",
    "used_for_source_eligibility", "used_for_candidate_ranking", "legacy",
    "unobserved_default", "observation", "threshold", "candidate_id", "hypothesis_id",
    "source_text", "source_hash", "payload_hash", "missing", "wrong_link", "wrong_target",
    "combined_legacy_default", "missing_link", "missing_selection", "conflicting_portfolio_link", "swapped_same_target_hypothesis",
])
def test_independent_audit_rejects_semantic_tampering_even_with_recomputed_hash(tmp_path, brief, mutation):
    hypotheses, packet = _setup(tmp_path, brief)
    packet = packet.model_copy(update={"backend_kind": ScientificBackendKind.LIVE}, deep=True)
    candidate = packet.candidates[0]
    record = candidate.metadata["prospective_falsification_status"]
    if mutation == "status": record["status"] = "falsified"
    elif mutation == "legacy": record["schema_version"] = "candidate_prospective_falsification_v2"
    elif mutation == "unobserved_default": record["machine_readable_contract"]["when_unobserved"] = "falsified"
    elif mutation == "observation": record["machine_readable_contract"]["observations"]["early_target_modulation"]["effect_percent_vs_vehicle"] = 60
    elif mutation == "threshold": record["machine_readable_contract"]["criteria"]["directional_endpoints"][0]["effect_threshold_percent"] = 0
    elif mutation in {"candidate_id", "hypothesis_id"}: record[mutation] = "foreign_record"
    elif mutation == "source_text": record["falsification_test"] = "Replacement rule"
    elif mutation == "source_hash": record["falsification_test_sha256"] = "0" * 64
    elif mutation == "wrong_link": candidate.metadata["assay_variant_alignment_hypothesis_id"] = "foreign_hypothesis"
    elif mutation == "wrong_target": candidate.target = candidate.target.model_copy(update={"symbol": "NRAS"})
    elif mutation == "missing_link": candidate.metadata.pop("assay_variant_alignment_hypothesis_id")
    elif mutation == "missing_selection": packet.metadata.pop("primary_hypothesis_selection")
    elif mutation == "conflicting_portfolio_link": candidate.metadata["portfolio_hypothesis_id"] = "foreign_hypothesis"
    elif mutation == "swapped_same_target_hypothesis":
        other = hypotheses.hypotheses[0].model_copy(update={"hypothesis_id": "other_same_target"}, deep=True)
        hypotheses.hypotheses.append(other)
        candidate.metadata["assay_variant_alignment_hypothesis_id"] = other.hypothesis_id
        record["hypothesis_id"] = other.hypothesis_id
    elif mutation == "combined_legacy_default": record["machine_readable_contract"]["default_when_unobserved_or_any_criterion_unmet"] = "falsified_or_partial"
    elif mutation not in {"payload_hash", "missing"}: record[mutation] = True
    _rehash(record)
    if mutation == "payload_hash": record["attestation_sha256"] = "0" * 64
    if mutation == "missing": candidate.metadata.pop("prospective_falsification_status")
    findings = MA3AssuranceAgent()._audit_prospective_falsification(hypotheses, packet)
    assert any(f.artifact_id == candidate.candidate_id and f.severity is Severity.HIGH
               and f.code == "PROSPECTIVE_FALSIFICATION_ATTESTATION_INVALID" for f in findings)


@pytest.mark.parametrize("node", ["ma2.assay_variant_alignment", "ma2.prospective_falsification"])
def test_legacy_semantic_rework_uses_new_packet_without_paid_recomputation(brief, node, monkeypatch):
    hypotheses, plan, packet = _live_portfolio(brief)
    candidate = next(c for c in packet.candidates if c.status != "excluded")
    candidate.metadata["assay_variant_alignment"]["schema_version"] = "assay_hypothesis_variant_alignment_v1"
    candidate.metadata["prospective_falsification_status"]["schema_version"] = "candidate_prospective_falsification_v2"
    before = packet.model_dump_json()
    original = deepcopy(candidate)
    order = ReworkOrder(gate=GateName.G1, target_stage="ma2", failed_node=node,
        affected_candidate_ids=[candidate.candidate_id], required_actions=["Regenerate semantic attestation from preserved source."],
        preserve_artifact_ids=["checkpoint_required_to_preserve", "verdict_required_to_preserve"])
    gateway = RunLocalLiveGateway(REWORKED_ADMET)
    agent = MA2Agent(gateway=gateway, chemistry=PersistedStructureChemistry(),
        ma2_portfolio_enabled=True, max_refinement_iterations=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Copy-only semantic repair must not screen, rank, enrich or run gateways")
    for name in ("_rescreen_candidate", "_rank", "_enrich_ranked_candidates", "_rework_single", "_rework_portfolio"):
        monkeypatch.setattr(agent, name, forbidden)
    result = agent.rework(plan, hypotheses, packet, order)
    assert result.packet_id != packet.packet_id
    assert set(order.preserve_artifact_ids).issubset(result.reused_artifact_ids)
    assert packet.model_dump_json() == before
    updated = next(c for c in result.candidates if c.candidate_id == candidate.candidate_id)
    assert updated.metadata["assay_variant_alignment"]["schema_version"] == "assay_hypothesis_variant_alignment_v2"
    assert updated.metadata["prospective_falsification_status"]["schema_version"] == "candidate_prospective_falsification_v3"
    assert updated.evidence == original.evidence and updated.activities == original.activities
    assert updated.metrics == original.metrics and updated.score == original.score
    assert gateway.predict_calls == gateway.docking_calls == gateway.restore_structure_calls == 0
    assert result.tool_calls == packet.tool_calls
    assert result.target_executions == packet.target_executions
    for old, new in zip(packet.candidates, result.candidates):
        old_value, new_value = old.model_dump(mode="json"), new.model_dump(mode="json")
        for value in (old_value, new_value):
            for key in ("assay_variant_alignment", "prospective_falsification_status"):
                value["metadata"].pop(key)
        assert old_value == new_value
    assert MA3AssuranceAgent()._audit_prospective_falsification(hypotheses, result) == []


@pytest.mark.parametrize("code", ["ASSAY_VARIANT_CLASS_ONLY", "ASSAY_VARIANT_PARTIAL_OVERLAP", "ASSAY_VARIANT_ANNOTATION_ONLY"])
def test_variant_limitations_are_source_enrichment_tasks_not_code_or_lab_requests(code):
    finding = MA3AssuranceAgent._finding(GateName.G1, "model_appropriateness", Severity.LOW,
        code, "Exact assay construct is unverified", "candidate", "ma2.assay_variant_alignment")
    assert code in DOCUMENT_CODES
    assert _kind_for(finding) == "data_required"


@pytest.mark.parametrize("legacy", [False, True])
def test_variant_audit_retains_class_limitation_and_rejects_legacy_exact(tmp_path, brief, legacy):
    hypotheses, packet = _setup(tmp_path, brief)
    h = hypotheses.hypotheses[0]
    h.mechanism = "FLT3-ITD inhibition"
    c = packet.candidates[0]
    context = _context("FLT3 ITD inhibition", "UNDEFINED MUTATION")
    c.metadata["measured_target"] = "CHEMBL1974"
    c.metadata["chembl_assay_context"] = context
    c.metadata["assay_variant_alignment"] = assay_variant_alignment(
        target_symbol="FLT3", hypothesis_texts=(h.mechanism, h.falsification_test),
        chembl_assay_context=context, expected_target_chembl_id="CHEMBL1974")
    if legacy:
        c.metadata["assay_variant_alignment"].update(schema_version="assay_hypothesis_variant_alignment_v1", status="exact_match")
    findings = MA3AssuranceAgent()._audit_assay_variant_alignment(hypotheses, packet)
    relevant = [f for f in findings if f.artifact_id == c.candidate_id]
    assert len(relevant) == 1
    assert relevant[0].code == ("ASSAY_VARIANT_ALIGNMENT_ATTESTATION_MISMATCH" if legacy else "ASSAY_VARIANT_CLASS_ONLY")
    assert relevant[0].severity is (Severity.HIGH if legacy else Severity.LOW)


@pytest.mark.parametrize("mutation", ["binding_status", "measured_target", "context_missing"])
def test_candidate_context_downgrade_does_not_bypass_preserved_source_binding(tmp_path, brief, mutation):
    hypotheses, packet = _setup(tmp_path, brief)
    packet = packet.model_copy(update={"backend_kind": ScientificBackendKind.LIVE}, deep=True)
    c, h = packet.candidates[0], hypotheses.hypotheses[0]
    h.mechanism = "FLT3 ITD inhibition"
    context = _context("FLT3 ITD inhibition", "UNDEFINED MUTATION")
    c.metadata["measured_target"] = "CHEMBL1974"
    c.metadata["chembl_assay_context"] = deepcopy(context)
    evidence = c.evidence[0].model_copy(update={"source": "ChEMBL", "metadata": {
        "target_chembl_id": "CHEMBL1974", "chembl_assay_context": deepcopy(context)}}, deep=True)
    c.evidence = [evidence]
    c.activities = [c.activities[0].model_copy(update={"source_evidence_id": evidence.evidence_id})]
    if mutation == "binding_status": c.metadata["chembl_assay_context"].pop("binding_status")
    elif mutation == "measured_target": c.metadata.pop("measured_target")
    else: c.metadata.pop("chembl_assay_context")
    c.metadata["assay_variant_alignment"] = assay_variant_alignment(target_symbol="FLT3",
        hypothesis_texts=(h.mechanism, h.falsification_test),
        chembl_assay_context=c.metadata.get("chembl_assay_context"),
        expected_target_chembl_id=c.metadata.get("measured_target"))
    findings = MA3AssuranceAgent()._audit_assay_variant_alignment(hypotheses, packet)
    assert any(f.artifact_id == c.candidate_id and f.code == "ASSAY_VARIANT_SOURCE_BINDING_MISMATCH"
               and f.severity is Severity.HIGH for f in findings)
