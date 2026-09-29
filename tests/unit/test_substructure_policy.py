"""Offline chemical-policy controls; all synthetic molecules stay in tests."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent, _TraceRecorder, _fingerprint
from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import CandidateCard, CandidateRoute, GateVerdictCode, ProjectConstraints
from drugpilot.tools import substructure_policy as policy
from drugpilot.tools.chemistry import ChemistryValidator
from drugpilot.tools.protocols import CandidateSeed


def pipeline(brief, patterns):
    constrained = brief.model_copy(deep=True)
    constrained.constraints.forbidden_substructures = list(patterns)
    settings = Settings(_env_file=None, ma1_provisional_admission_enabled=False,
        dynamic_target_discovery_enabled=False, dynamic_target_fallback_enabled=True,
        demo_fixture_fallback_enabled=True)
    hypotheses = MA1Agent(settings=settings).run(constrained)
    plan = PlannerAgent(settings=settings).run(hypotheses, constrained)
    return constrained, hypotheses, plan


@pytest.mark.parametrize("smiles", ["CC(N)=O", "CC(=O)N", "NC(C)=O"])
def test_equivalent_smiles_match_identical_forbidden_subgraph(smiles):
    result = policy.screen_forbidden_substructures(smiles, ["C(=O)N"])
    assert result.status == "matched" and result.blocked
    assert result.matched_patterns == ("C(=O)N",)
    assert result.as_metadata()["string_matching_fallback_used"] is False


@pytest.mark.parametrize("smiles", ["c1ccccc1", "C1=CC=CC=C1"])
def test_aromatic_representation_is_normalized(smiles):
    assert policy.screen_forbidden_substructures(smiles, ["c1ccccc1"]).matched


def test_submitted_salt_and_standardized_parent_are_both_screened():
    # Neutralising the actual downstream parent must not introduce an unchecked
    # forbidden neutral amine; an ionic input-specific query still matches too.
    assert policy.screen_forbidden_substructures("C[NH3+].[Cl-]", ["[N;+0]"]).matched
    assert policy.screen_forbidden_substructures("C[NH3+].[Cl-]", ["[N;+1]"]).matched
    assert not policy.screen_forbidden_substructures("CCO", ["C(=O)N"]).blocked


@pytest.mark.parametrize("patterns", [[""], ["["], ["CC a molecule name"], ["C" * (policy.MAX_PATTERN_CHARS + 1)]])
def test_invalid_query_is_not_silently_treated_as_absent(patterns):
    checked = policy.validate_forbidden_substructures(patterns)
    assert checked.status == "invalid_policy"
    assert checked.blocked and checked.reason_codes


def test_unparseable_structure_is_unresolved():
    result = policy.screen_forbidden_substructures("not-a-smiles", ["CN"])
    assert result.status == "invalid_structure"
    assert result.blocked and not result.matched


def test_explicit_stereochemistry_is_respected():
    assert policy.screen_forbidden_substructures("N[C@@H](C)C(=O)O", ["N[C@@H](C)C(=O)O"]).matched
    assert not policy.screen_forbidden_substructures("N[C@H](C)C(=O)O", ["N[C@@H](C)C(=O)O"]).matched


def test_missing_backend_fails_closed_but_empty_policy_is_legacy_noop(monkeypatch):
    def missing():
        raise ImportError("No RDKit")
    monkeypatch.setattr(policy, "_load_rdkit", missing)
    assert policy.screen_forbidden_substructures("CCO", []).status == "not_requested"
    result = policy.screen_forbidden_substructures("CCO", ["CN"])
    assert result.blocked and result.status == "unavailable" and not result.matched


def test_planner_propagates_policy_and_old_empty_plan_serialization_is_preserved(brief):
    _, _, old = pipeline(brief, [])
    assert "forbidden_substructures" not in old.model_dump(mode="json")
    # Historical briefs are still readable without chemistry or new validators.
    assert ProjectConstraints().model_dump()["forbidden_substructures"] == []
    assert ProjectConstraints(forbidden_substructures=["["]).forbidden_substructures == ["["]
    _, _, new = pipeline(brief, ["C(=O)N"])
    assert new.forbidden_substructures == ["C(=O)N"]
    operations = {op for task in new.tasks for tool in task.tools for op in tool.allowed_operations}
    assert {"screen_forbidden_substructure", "rescreen_forbidden_substructure"} <= operations
    with pytest.raises(ValueError, match="Forbidden substructure policy cannot be executed"):
        pipeline(brief, ["["])


@pytest.mark.parametrize("missing_backend", [False, True])
def test_initial_screen_prevents_descriptor_docking_and_admet_calls(brief, monkeypatch, missing_backend):
    _, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent()
    def forbidden_call(*args, **kwargs):
        pytest.fail("No property/docking/ADMET call is allowed before clear policy screening")
    monkeypatch.setattr(agent, "_chemistry_result", forbidden_call)
    monkeypatch.setattr(agent.gateway, "dock", forbidden_call)
    monkeypatch.setattr(agent, "_admet_result", forbidden_call)
    if missing_backend:
        monkeypatch.setattr(policy, "_load_rdkit", lambda: (_ for _ in ()).throw(ImportError()))
    recorder = _TraceRecorder("test-only")
    seed = CandidateSeed("Synthetic test acetamide", "CC(N)=O", "inhibit FLT3")
    card, _ = agent._evaluate_once(plan=plan, recorder=recorder,
        target=hypotheses.hypotheses[0].target, route=CandidateRoute.REPURPOSING,
        seed=seed, trace_ids=[])
    assert card.status == "excluded" and card.rank is None and card.score == 0
    assert card.smiles == seed.smiles and card.activities == []
    assert card.chemistry_execution is None and card.admet_execution is None and card.docking_execution is None
    assert not card.metrics.admet_scores
    assert [call.operation for call in recorder.calls] == ["screen_forbidden_substructure"]
    assert card.metadata["forbidden_substructure_screen"]["matched"] is not missing_backend


def test_full_g1_accepts_correct_exclusion_and_blocks_tampered_promotion(brief):
    brief, hypotheses, plan = pipeline(brief, ["S(=O)(=O)N"])
    packet = MA2Agent().run(plan, hypotheses)
    excluded = next(card for card in packet.candidates if "forbidden_substructure" in card.safety_flags)
    assert excluded.chemistry_execution is None and excluded.admet_execution is None
    assert excluded.metadata["policy_eligible"] is False
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.PASS, [(f.code, f.message) for f in verdict.findings]
    tampered = packet.model_copy(deep=True)
    promoted = next(card for card in tampered.candidates if card.candidate_id == excluded.candidate_id)
    promoted.status = "priority"
    promoted.rank = 1
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, tampered))
    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "HARD_CONSTRAINT_VIOLATION" in {f.code for f in verdict.findings}


def test_full_g1_checks_excluded_cards_and_detects_computation_after_exclusion(brief):
    brief, hypotheses, plan = pipeline(brief, ["S(=O)(=O)N"])
    packet = MA2Agent().run(plan, hypotheses)
    excluded = next(card for card in packet.candidates if "forbidden_substructure" in card.safety_flags)
    # Add an actual already-recorded later calculation's lineage to the excluded
    # card. G1 must inspect excluded cards too, not skip policy-order auditing.
    later_compute = next(call for call in packet.tool_calls if call.tool_name == "admet_predictor")
    excluded.tool_trace_ids.append(later_compute.call_id)
    excluded.safety_flags.append("invalid_structure")  # exercises old early-continue path
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "FORBIDDEN_SUBSTRUCTURE_COMPUTED_WITHOUT_CLEARANCE" in {f.code for f in verdict.findings}


def test_full_g1_missing_backend_is_hold_not_a_scientific_negative(brief, monkeypatch):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    monkeypatch.setattr(policy, "_load_rdkit", lambda: (_ for _ in ()).throw(ImportError()))
    packet = MA2Agent().run(plan, hypotheses)
    assert all(card.status == "excluded" for card in packet.candidates)
    assert not any(call.tool_name in {"chemistry_validator", "docking", "admet_predictor"} for call in packet.tool_calls)
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.ABSTAIN, [f.code for f in verdict.findings]
    assert "FORBIDDEN_SUBSTRUCTURE_CHECK_UNAVAILABLE" in {f.code for f in verdict.findings}


def test_g1_rejects_missing_or_tampered_trace_and_policy_mismatch(brief):
    brief, hypotheses, plan = pipeline(brief, ["[U]"])
    packet = MA2Agent().run(plan, hypotheses)
    card = packet.candidates[0]
    card.metadata["forbidden_substructure_screens"][0]["result"]["patterns"] = ["[He]"]
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert "FORBIDDEN_SUBSTRUCTURE_SCREEN_INVALID" in {f.code for f in verdict.findings}
    assert verdict.verdict is GateVerdictCode.BLOCK
    plan.forbidden_substructures = []
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert "CANDIDATE_POLICY_MISMATCH" in {f.code for f in verdict.findings}


def test_repair_rescreen_excludes_graph_match_without_rewriting_existing_measurements(brief):
    brief, hypotheses, plan = pipeline(brief, ["[U]"])
    agent = MA2Agent()
    packet = agent.run(plan, hypotheses)
    previous = next(card for card in packet.candidates if card.status != "excluded")
    previous.smiles = "CC(N)=O"  # isolated hypothetical persisted repair input
    plan.forbidden_substructures = ["C(=O)N"]
    recorder = _TraceRecorder("test-only", existing=packet.tool_calls)
    card = agent._rescreen_candidate(plan, recorder, previous)
    assert card.status == "excluded" and card.rank is None and card.score == 0.0
    assert card.metadata["policy_eligible"] is False
    assert card.metrics == previous.metrics and card.activities == previous.activities
    assert recorder.calls[-1].operation == "rescreen_forbidden_substructure"


def test_repair_rescreens_exact_restored_seed_before_expensive_tools(brief, monkeypatch):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent()
    packet = agent.run(plan, hypotheses)
    card = next(c for c in packet.candidates if c.status != "excluded")
    recorder = _TraceRecorder("test-only", existing=packet.tool_calls)
    restored = CandidateSeed(card.name, "CC(N)=O", card.mechanism,
        tuple(card.activities), tuple(card.evidence), card.metadata)
    monkeypatch.setattr(agent, "_restore_persisted_candidate_seed", lambda *a, **kw: restored)
    def fail(*a, **kw):
        pytest.fail("Repair must screen actual restored structure before expensive tools")
    monkeypatch.setattr(agent.gateway, "dock", fail)
    monkeypatch.setattr(agent, "_admet_result", fail)
    result, _ = agent._repair_candidate(plan=plan, recorder=recorder,
        hypothesis=hypotheses.hypotheses[0], target=card.target, direction="inhibit",
        previous=card, repair_intent=frozenset({"docking", "admet"}))
    assert result.status == "excluded" and result.rank is None
    assert result.metadata["forbidden_substructure_screen"]["source_smiles"] == "CC(N)=O"
    assert result.activities == card.activities


def test_refined_seed_is_screened_again_and_prior_clear_computation_is_not_false_violation(brief, monkeypatch):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent()
    recorder = _TraceRecorder("test-only")
    target = hypotheses.hypotheses[0].target
    original = CandidateSeed("Synthetic test parent", "CCO", "inhibit FLT3")
    monkeypatch.setattr(agent.gateway, "refine", lambda seed, **kw: replace(seed, smiles="CC(N)=O"))
    monkeypatch.setattr(agent, "_risk_flags", lambda metrics: (["hepatotoxicity_high"], []))
    card, iterations = agent._evaluate_seed(plan=plan, recorder=recorder,
        hypothesis=hypotheses.hypotheses[0], target=target, route=CandidateRoute.DE_NOVO,
        seed=original, base_trace_ids=[], allow_refinement=True)
    assert iterations == 2 and card.status == "excluded"
    assert [row["result"]["status"] for row in card.metadata["forbidden_substructure_screens"]] == ["clear", "matched"]
    assert sum(call.tool_name == "chemistry_validator" for call in recorder.calls) == 1
    assert sum(call.tool_name == "admet_predictor" for call in recorder.calls) == 1
    findings = MA3AssuranceAgent()._audit_forbidden_substructures(brief, card, recorder.calls)
    assert not findings, [(f.code, f.message) for f in findings]


def test_invalid_initial_structure_can_be_refined_only_after_new_clear_screen(brief, monkeypatch):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent()
    recorder = _TraceRecorder("test-only")
    seed = CandidateSeed("Synthetic invalid test parent", "invalid-smiles", "inhibit FLT3")
    monkeypatch.setattr(agent.gateway, "refine", lambda seed, **kw: replace(seed, smiles="CCO"))
    card, iterations = agent._evaluate_seed(plan=plan, recorder=recorder,
        hypothesis=hypotheses.hypotheses[0], target=hypotheses.hypotheses[0].target,
        route=CandidateRoute.DE_NOVO, seed=seed, base_trace_ids=[], allow_refinement=True)
    assert iterations == 2 and card.status != "excluded"
    assert [row["result"]["status"] for row in card.metadata["forbidden_substructure_screens"]] == ["invalid_structure", "clear"]
    assert sum(call.tool_name == "chemistry_validator" for call in recorder.calls) == 1
    assert not MA3AssuranceAgent()._audit_forbidden_substructures(brief, card, recorder.calls)


@pytest.mark.parametrize("actual_smiles", ["CC(N)=O", "CCCC"])
def test_g1_binds_actual_chemistry_input_not_merely_prior_clear_screen(brief, actual_smiles):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent()
    recorder = _TraceRecorder("test-only")
    metadata, screen_id, _ = agent._screen_substructure(plan, recorder, "CCO", {})
    _, compute_id = recorder.invoke(task_id=agent._task_id(plan, CandidateRoute.REPURPOSING, "chemistry"),
        tool_name="chemistry_validator", operation="validate_and_describe",
        inputs={"name": "isolated audit test", "smiles": actual_smiles},
        chemical_policy_screen_id=screen_id, function=lambda: {"test_only": True})
    candidate = CandidateCard(route=CandidateRoute.REPURPOSING, name="isolated audit test",
        target=hypotheses.hypotheses[0].target, mechanism="inhibit FLT3", smiles="CCO",
        metadata=metadata, tool_trace_ids=[screen_id, compute_id])
    findings = MA3AssuranceAgent()._audit_forbidden_substructures(brief, candidate, recorder.calls)
    assert "FORBIDDEN_SUBSTRUCTURE_COMPUTE_INPUT_MISMATCH" in {finding.code for finding in findings}


@pytest.mark.parametrize("tool", ["chemistry_validator", "docking", "admet_predictor"])
def test_full_g1_detects_changed_actual_compute_smiles_even_with_rehashed_binding(brief, tool):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    packet = MA2Agent().run(plan, hypotheses)
    candidate = next(card for card in packet.candidates if card.status != "excluded")
    call = next(call for call in packet.tool_calls if call.call_id in candidate.tool_trace_ids and call.tool_name == tool)
    binding = call.provenance["molecular_input_binding"]
    binding["arguments"]["smiles"] = "CC(N)=O"
    payload = dict(binding["arguments"])
    if tool == "docking":
        payload["target"] = candidate.target.model_dump(mode="json")
    call.input_hash = _fingerprint(payload)
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "FORBIDDEN_SUBSTRUCTURE_COMPUTE_INPUT_MISMATCH" in {finding.code for finding in verdict.findings}


@pytest.mark.parametrize("source_smiles", ["C(C)O", "CC[NH3+].[Cl-]"])
def test_standardized_compute_parent_remains_bound_to_raw_screen(brief, source_smiles):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent(chemistry=ChemistryValidator(prefer_rdkit=True))
    recorder = _TraceRecorder("test-only")
    candidate, _ = agent._evaluate_once(plan=plan, recorder=recorder,
        target=hypotheses.hypotheses[0].target, route=CandidateRoute.REPURPOSING,
        seed=CandidateSeed("isolated canonical test", source_smiles, "inhibit FLT3"), trace_ids=[])
    assert candidate.smiles != source_smiles
    assert not MA3AssuranceAgent()._audit_forbidden_substructures(brief, candidate, recorder.calls)
    for call in recorder.calls:
        if call.tool_name in {"docking", "admet_predictor"}:
            assert call.provenance["molecular_input_binding"]["arguments"]["smiles"] == candidate.smiles


def test_refinement_may_rename_seed_without_breaking_historical_input_binding(brief, monkeypatch):
    brief, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent()
    recorder = _TraceRecorder("test-only")
    monkeypatch.setattr(agent.gateway, "refine", lambda seed, **kw: replace(seed, name="renamed refined test", smiles="CCCO"))
    monkeypatch.setattr(agent, "_risk_flags", lambda metrics: (["hepatotoxicity_high"], []))
    candidate, _ = agent._evaluate_seed(plan=plan, recorder=recorder,
        hypothesis=hypotheses.hypotheses[0], target=hypotheses.hypotheses[0].target,
        route=CandidateRoute.DE_NOVO, seed=CandidateSeed("initial test name", "CCO", "inhibit FLT3"),
        base_trace_ids=[], allow_refinement=True)
    assert candidate.name == "renamed refined test"
    assert not MA3AssuranceAgent()._audit_forbidden_substructures(brief, candidate, recorder.calls)


def test_unchanged_repair_and_legacy_empty_policy_keep_valid_lineage(brief):
    constrained, hypotheses, plan = pipeline(brief, ["C(=O)N"])
    agent = MA2Agent()
    packet = agent.run(plan, hypotheses)
    previous = next(card for card in packet.candidates if card.status != "excluded")
    recorder = _TraceRecorder("test-only", existing=packet.tool_calls)
    screened = agent._rescreen_candidate(plan, recorder, previous)
    repaired, _ = agent._repair_candidate(plan=plan, recorder=recorder,
        hypothesis=hypotheses.hypotheses[0], target=previous.target, direction="inhibit",
        previous=screened, repair_intent=frozenset({"docking", "admet"}))
    calls = [call for call in recorder.calls if call.call_id in repaired.tool_trace_ids]
    assert not MA3AssuranceAgent()._audit_forbidden_substructures(constrained, repaired, calls)
    brief, hypotheses, plan = pipeline(brief, [])
    legacy = agent.run(plan, hypotheses)
    assert all("molecular_input_binding" not in call.provenance for call in legacy.tool_calls)
    assert asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, legacy)).verdict is GateVerdictCode.PASS


def test_g1_rejects_binding_with_changed_arguments_without_matching_input_hash(brief):
    brief, hypotheses, plan = pipeline(brief, ["[U]"])
    packet = MA2Agent().run(plan, hypotheses)
    call = next(call for call in packet.tool_calls if call.tool_name == "admet_predictor")
    call.provenance["molecular_input_binding"]["arguments"]["smiles"] = "CCO"
    verdict = asyncio.run(MA3AssuranceAgent().audit_g1(brief, hypotheses, plan, packet))
    assert verdict.verdict is GateVerdictCode.BLOCK
    assert "FORBIDDEN_SUBSTRUCTURE_COMPUTE_INPUT_MISMATCH" in {finding.code for finding in verdict.findings}
