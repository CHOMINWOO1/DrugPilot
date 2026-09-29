from drugpilot.domain.models import (
    CandidateCard,
    CandidatePacket,
    CandidateRoute,
    Claim,
    EvidenceRef,
    GateName,
    GateVerdict,
    GateVerdictCode,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    PlanSpec,
    PlanTask,
    TargetCandidate,
    ToolAssignment,
)
from drugpilot.orchestration.explanations import (
    audit_handoff,
    candidate_handoff,
    hypothesis_handoff,
    plan_handoff,
)


def test_hypothesis_handoff_is_evidence_bound() -> None:
    evidence = EvidenceRef(
        source="PubMed",
        external_id="123",
        title="Target evidence",
        supports="support",
    )
    hypothesis = Hypothesis(
        track=HypothesisTrack.LITERATURE,
        disease="Disease",
        target=TargetCandidate(symbol="GENE1"),
        mechanism="GENE1 may modulate disease biology.",
        intervention_direction="inhibit",
        falsification_test="No response after target inhibition.",
        claims=[Claim(text="Test claim", evidence_ids=[evidence.evidence_id])],
        evidence=[evidence],
        score=0.8,
        rank=1,
    )
    packet = HypothesisPacket(brief_id="brief", hypotheses=[hypothesis])

    trace = hypothesis_handoff(packet)

    assert trace["sender"] == "MA1"
    assert trace["receiver"] == "MA3 · G0"
    assert evidence.evidence_id in trace["evidence_ids"]
    assert "GENE1" in trace["basis"][0]
    assert trace["explanation_scope"] == "evidence_and_decision_summary"


def test_plan_candidate_and_audit_handoffs_summarize_decisions() -> None:
    plan = PlanSpec(
        hypothesis_packet_id="hp_1",
        tasks=[
            PlanTask(
                name="Screen",
                route=CandidateRoute.REPURPOSING,
                tools=[ToolAssignment(tool_name="chembl_search", purpose="retrieve")],
            )
        ],
        approved_tools=["chembl_search"],
        stop_conditions=["budget"],
        candidates_per_route=1,
        total_token_budget=10_000,
    )
    plan_trace = plan_handoff(plan)
    assert plan_trace["receiver"] == "MA2"
    assert plan_trace["details"]["approved_tools"] == ["chembl_search"]

    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="Compound A",
        target=TargetCandidate(symbol="GENE1"),
        mechanism="inhibit",
        smiles="CCO",
        score=0.9,
        rank=1,
        status="priority",
    )
    packet = CandidatePacket(plan_id=plan.plan_id, candidates=[candidate])
    candidate_trace = candidate_handoff(packet)
    assert candidate_trace["receiver"] == "MA3 · G1"
    assert "Compound A" in candidate_trace["basis"][0]

    verdict = GateVerdict(
        gate=GateName.G1,
        verdict=GateVerdictCode.PASS,
        rationale="All mandatory controls passed.",
        reviewer_model="deterministic-rules-v1",
    )
    audit_trace = audit_handoff(verdict)
    assert audit_trace["receiver"] == "Human Gate"
    assert audit_trace["decision"] == "PASS"
