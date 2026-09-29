"""Synthetic isolated admission tests: never inserted into live evidence or DBs."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from drugpilot.agents.evidence_semantics import abstract_context, interpretation_text, source_binding, source_review_text
from drugpilot.agents.molecular_scope import hypothesis_scope
from drugpilot.agents.hypothesis_admission import (
    AdmissionAssessment, BoundClaim, COMPONENT, POLICY, ROLES, MAX_HYPOTHESES,
    CONFIDENCE_SEMANTICS, admitted_role_is_linked, admitted_support_ids, assess_provisional,
)
from drugpilot.domain.models import Claim, EvidenceRef, Hypothesis, HypothesisPacket, TargetCandidate, TokenUsage
from drugpilot.providers import LLMProviderCallError, LLMResult

MODEL = "gpt-5.6-sol"
DIRECTION_TEXT = (
    "Compound A directly bound FLT3 and inhibited its activity in FLT3-ITD acute myeloid leukemia cells. "
    "In these same cells, Compound A reduced cell survival and increased apoptosis."
)
ASSOCIATION_TEXT = "FLT3-ITD was measured in acute myeloid leukemia patient samples and was associated with leukemic cell expansion."


def evidence(text, evidence_id="ev_dir"):
    return EvidenceRef(evidence_id=evidence_id, source="PubMed", external_id="synthetic-unit-test",
        title="Synthetic test ONLY", excerpt_hash=sha256(text.encode()).hexdigest(),
        metadata={"excerpt": text, "provenance_class": "external_live", "publication_types": ["Journal Article"],
                  "content_hash": sha256(text.encode()).hexdigest(), "source_location": "abstract:RESULTS"})


def proposal(brief, hypothesis_id="hyp_test"):
    return Hypothesis(hypothesis_id=hypothesis_id, track="literature", disease=brief.disease,
        target=TargetCandidate(symbol="FLT3", identity_status="resolved"),
        intervention_direction="inhibit", mechanism="Unsupported original expansive mechanism proposal",
        falsification_test="Compare target perturbation in a controlled assay.",
        claims=[Claim(claim_id="original_unsupported", text="Unsupported original broad claim.", evidence_ids=[])],
        evidence=[evidence(DIRECTION_TEXT), evidence(ASSOCIATION_TEXT, "ev_assoc")], score=.7)


def packet(brief, *proposals):
    return HypothesisPacket(brief_id=brief.brief_id, hypotheses=[], provisional_hypotheses=list(proposals),
                            metadata={"admission_policy": POLICY})


def bound_claim(hypothesis, role="direct_target_direction_primary", **changes):
    direct = role == "direct_target_direction_primary"
    e = hypothesis.evidence[0 if direct else 1]
    result = dict(role=role, evidence_id=e.evidence_id, source_binding_sha256=source_binding(e),
        source_text_sha256=sha256(e.metadata["excerpt"].encode()).hexdigest(), experiment_quote=e.metadata["excerpt"],
        target_quote="FLT3", model_quote="FLT3-ITD acute myeloid leukemia cells" if direct else "acute myeloid leukemia patient samples",
        disease_quote="acute myeloid leukemia", intervention_quote="Compound A directly bound FLT3 and inhibited its activity" if direct else "",
        result_quote="Compound A reduced cell survival and increased apoptosis" if direct else "was associated with leukemic cell expansion",
        canonical_target="FLT3", model_genotype_quote="FLT3-ITD", observed_direction="inhibit" if direct else "none",
        study_type="primary_experiment" if direct else "primary_association", target_relation="direct", disease_context="actual_model",
        attribution="isolated_target_intervention" if direct else "target_disease_association", outcome="favorable" if direct else "associated",
        certainty="reported_result", experiment_link="same_experiment", claim_scope="tested_model_only", unresolved_counterevidence=False)
    result.update(changes)
    return BoundClaim(**result)


def assessment(hypothesis, **changes):
    result = dict(hypothesis_id=hypothesis.hypothesis_id, status="approved", reason="Both primary roles are source-bound.",
        limitations=["No clinical efficacy or variant-selectivity claim."],
        claims=[bound_claim(hypothesis), bound_claim(hypothesis, "primary_target_disease_association")])
    result.update(changes)
    return AdmissionAssessment(**result)


class Provider:
    def __init__(self, results=None, error=None, usage_model=MODEL):
        self.results = list(results or [])
        self.error = error
        self.calls = []
        self.usage_model = usage_model

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return LLMResult(data=self.results.pop(0), usage=TokenUsage(component=COMPONENT,
            model=self.usage_model, input_tokens=100, output_tokens=50, estimated=False))


async def test_adjacent_experiment_survival_is_admitted_without_mutating_source(brief):
    p = proposal(brief)
    submitted = packet(brief, p)
    original = submitted.model_dump_json()
    provider = Provider([assessment(p)])
    approved, report = await assess_provisional(brief, submitted, provider, MODEL)
    assert submitted.model_dump_json() == original
    assert len(approved.hypotheses) == 1 and not approved.provisional_hypotheses
    h = approved.hypotheses[0]
    assert all(admitted_role_is_linked(h, role, brief) for role in ROLES)
    assert h.rank == 1
    assert h.claims[0].evidence_ids == ["ev_dir"]
    assert "Unsupported original" not in h.claims[0].text
    assert "clinical efficacy" in h.claims[0].text
    assert "ma3_admission" not in p.target.metadata
    assert h.evidence == p.evidence
    assert report["entries"][0]["original_proposal"]["claims"][0]["claim_id"] == "original_unsupported"
    assert report["approved_count"] == 1
    assert approved.metadata["ma3_admission_report"] == report
    assert report["report_sha256"] and report["entries"][0]["certificate_sha256"]


async def test_admission_preserves_tested_scope_and_removes_stale_template_status(brief):
    from drugpilot.agents.hypothesis_admission import admitted_tested_model_context
    from drugpilot.tools.assay_variant_alignment import assay_variant_alignment
    p = proposal(brief)
    p.falsification_test = (
        "Reject the hypothesis if inhibit perturbation of FLT3 in at least two disease-relevant models "
        "fails to improve the pre-specified primary phenotype by at least 30% versus matched controls, "
        "or produces the opposite direction in either model.\n"
        "Proposed intervention direction for MA3 review: inhibit (FLT3). This is a research question, "
        "not a source-resolved or scientifically approved direction."
    )
    original = p.model_dump_json()
    approved, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    h = approved.hypotheses[0]
    context = admitted_tested_model_context(h)
    assert context["contexts"][0]["model_genotype_quote"] == "FLT3-ITD"
    assert context["contexts"][0]["intervention_quote"] == bound_claim(p).intervention_quote
    assert context["planned_model_equivalence_established"] is False
    assert context["chemical_intervention_equivalence_established"] is False
    assert context["variant_selectivity_established"] is False
    assert "FLT3-ITD" in h.mechanism
    assert "unapproved" not in h.falsification_test and "30%" not in h.falsification_test
    assert "Experimental protocol pending" in h.falsification_test
    assert h.target.metadata["ma1_original_falsification_test"] == p.falsification_test
    alignment = assay_variant_alignment(target_symbol="FLT3", hypothesis_texts=(h.mechanism, h.falsification_test), chembl_assay_context={})
    assert alignment["declared_hypothesis_variants"] == ["FLT3-ITD"]
    assert alignment["status"] == "unknown_not_assumed_match"
    assert p.model_dump_json() == original


async def test_custom_falsification_protocol_is_not_silently_removed(brief):
    p = proposal(brief)
    custom = "Measure viable cell count at 72 hours; reject if reduction is under 40% with matched controls."
    p.falsification_test = custom + (
        "\nProposed intervention direction for MA3 review: inhibit (FLT3). This is a research question, "
        "not a source-resolved or scientifically approved direction."
    )
    approved, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert approved.hypotheses[0].falsification_test == custom


async def test_custom_criteria_appended_after_old_default_are_preserved(brief):
    p = proposal(brief)
    custom = "Use measured viable cell count at 72h and a researcher-defined 40% threshold."
    p.falsification_test = (
        "Reject the hypothesis if inhibit perturbation of FLT3 in at least two disease-relevant models "
        "fails to improve the pre-specified primary phenotype by at least 30% versus matched controls, "
        "or produces the opposite direction in either model.\n" + custom)
    approved, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert approved.hypotheses[0].falsification_test == custom


async def test_modified_model_context_cannot_become_a_valid_scope_even_when_rehashed(brief):
    from drugpilot.agents.hypothesis_admission import admitted_tested_model_context, _digest, _hypothesis_binding
    p = proposal(brief)
    approved, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    h = approved.hypotheses[0]
    h.target.metadata["ma3_tested_model_context"]["contexts"][0]["model_genotype_quote"] = "FLT3-WT"
    certificate = h.target.metadata["ma3_admission"]
    certificate["hypothesis_sha256"] = _hypothesis_binding(h)
    certificate.pop("certificate_sha256")
    certificate["certificate_sha256"] = _digest(certificate)
    assert admitted_tested_model_context(h) is None
    assert not admitted_role_is_linked(h, "direct_target_direction_primary", brief)


@pytest.mark.parametrize("text,expected", [
    ("IDH1R132H-mutated MOLM-13 cells", ("IDH1-R132H",)),
    ("IDH1 R132H-mutated cells", ("IDH1-R132H",)),
    ("IDH2R172K-mutated cells", ()),
    ("MYIDH1R132Habc cells", ()),
    ("IDH1 mutant cells, R132H unspecified ownership", ()),
])
def test_explicit_concatenated_genotype_not_discarded_or_inferred(text, expected):
    from drugpilot.tools.assay_variant_alignment import extract_target_variants
    assert extract_target_variants("IDH1", (text,)) == expected


async def test_tested_scope_survives_ma2_and_g1_detects_omission(brief):
    from drugpilot.agents.ma2 import MA2Agent
    from drugpilot.agents.ma3 import MA3AssuranceAgent
    from drugpilot.domain.models import CandidateCard, CandidatePacket
    from drugpilot.tools.assay_variant_alignment import assay_variant_alignment
    p = proposal(brief)
    approved, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    h = approved.hypotheses[0]
    candidate = CandidateCard(route="repurposing", name="synthetic unit-test compound",
        target=h.target.model_copy(deep=True), mechanism="Test only", smiles="CCO",
        metadata={"assay_variant_alignment_hypothesis_id": h.hypothesis_id,
            "assay_variant_alignment": assay_variant_alignment(target_symbol="FLT3",
                hypothesis_texts=(h.mechanism, h.falsification_test), chembl_assay_context={})})
    carried = MA2Agent._attest_prospective_falsification_status([candidate], h)[0]
    assert "ma3_tested_model_context" not in candidate.metadata
    assert carried.metadata["ma3_tested_model_context"] == h.target.metadata["ma3_tested_model_context"]
    cp = CandidatePacket(plan_id="unit-test-plan", candidates=[carried],
        metadata={"primary_hypothesis_selection": {"selected_hypothesis_id": h.hypothesis_id}})
    auditor = MA3AssuranceAgent()
    assert "MA3_TESTED_MODEL_CONTEXT_NOT_PRESERVED" not in {
        f.code for f in auditor._audit_assay_variant_alignment(approved, cp)}
    carried.metadata.pop("ma3_tested_model_context")
    assert "MA3_TESTED_MODEL_CONTEXT_NOT_PRESERVED" in {
        f.code for f in auditor._audit_assay_variant_alignment(approved, cp)}


@pytest.mark.parametrize("changes,code", [
    ({"source_binding_sha256": "f" * 64}, "SOURCE_HASH_MISMATCH"),
    ({"source_text_sha256": "f" * 64}, "SOURCE_HASH_MISMATCH"),
    ({"evidence_id": "invented_pmid_record"}, "SOURCE_BINDING_INVALID"),
    ({"experiment_quote": "Fabricated experimental text."}, "EXPERIMENT_QUOTE_NOT_IN_SOURCE"),
    ({"result_quote": "cured AML patients"}, "COMPONENT_OUTSIDE_EXPERIMENT"),
    ({"intervention_quote": "selectively inhibited FLT3-ITD"}, "INTERVENTION_OUTSIDE_EXPERIMENT"),
    ({"canonical_target": "IDH1"}, "TARGET_IDENTITY_MISMATCH"),
    ({"target_quote": "Compound A"}, "TARGET_QUOTE_IDENTITY_MISMATCH"),
    ({"disease_context": "different_model"}, "TARGET_DISEASE_NOT_DIRECTLY_LINKED"),
    ({"disease_context": "background_only"}, "TARGET_DISEASE_NOT_DIRECTLY_LINKED"),
    ({"target_relation": "indirect"}, "TARGET_DISEASE_NOT_DIRECTLY_LINKED"),
    ({"certainty": "uncertain"}, "UNESTABLISHED_OR_CROSS_EXPERIMENT_RESULT"),
    ({"certainty": "planned"}, "UNESTABLISHED_OR_CROSS_EXPERIMENT_RESULT"),
    ({"experiment_link": "cross_experiment"}, "UNESTABLISHED_OR_CROSS_EXPERIMENT_RESULT"),
    ({"claim_scope": "variant_selectivity"}, "CLAIM_SCOPE_EXCEEDS_TESTED_MODEL"),
    ({"claim_scope": "clinical_efficacy"}, "CLAIM_SCOPE_EXCEEDS_TESTED_MODEL"),
    ({"unresolved_counterevidence": True}, "COUNTEREVIDENCE_UNRESOLVED"),
    ({"observed_direction": "activate"}, "DIRECTION_MISMATCH"),
    ({"outcome": "unfavorable"}, "DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED"),
    ({"outcome": "not_evaluated"}, "DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED"),
    ({"attribution": "combination_only"}, "DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED"),
    ({"attribution": "response_modifier"}, "DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED"),
    ({"study_type": "review"}, "DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED"),
])
async def test_malformed_or_non_supporting_assessment_cannot_approve(brief, changes, code):
    p = proposal(brief)
    a = assessment(p, claims=[bound_claim(p, **changes), bound_claim(p, "primary_target_disease_association")])
    output, report = await assess_provisional(brief, packet(brief, p), Provider([a]), MODEL)
    assert not output.hypotheses
    assert report["entries"][0]["status"] == "needs_evidence"
    assert code in report["entries"][0]["validation_errors"]


@pytest.mark.parametrize("role", list(ROLES))
async def test_both_primary_roles_required(brief, role):
    p = proposal(brief)
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p, claims=[bound_claim(p, role)])]), MODEL)
    assert not out.hypotheses
    assert "REQUIRED_PRIMARY_ROLE_MISSING" in report["entries"][0]["validation_errors"]


async def test_scoped_model_not_required_as_intervention_object(brief):
    p = proposal(brief)
    p.target.metadata["molecular_hypothesis_scope"] = hypothesis_scope(brief, "FLT3", p.evidence, "FLT3:ITD")
    out, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert len(out.hypotheses) == 1
    assert admitted_role_is_linked(out.hypotheses[0], "direct_target_direction_primary", brief)
    wrong = assessment(p, claims=[bound_claim(p, model_genotype_quote="FLT3"), bound_claim(p, "primary_target_disease_association")])
    out, report = await assess_provisional(brief, packet(brief, p), Provider([wrong]), MODEL)
    assert not out.hypotheses
    assert "SCOPED_MODEL_QUOTE_MISSING" in report["entries"][0]["validation_errors"]


async def test_bound_full_abstract_can_supply_adjacent_source_result(brief):
    p = proposal(brief)
    e = p.evidence[0]
    full = "This study concerns acute myeloid leukemia. " + DIRECTION_TEXT
    e.metadata.update(abstract_interpretation_context=abstract_context(full), content_hash_scope="normalized_full_abstract",
                      content_hash=sha256(full.encode()).hexdigest())
    a = assessment(p)
    a.claims[0].source_text_sha256 = sha256(full.encode()).hexdigest()
    out, _ = await assess_provisional(brief, packet(brief, p), Provider([a]), MODEL)
    assert len(out.hypotheses) == 1
    out.hypotheses[0].evidence[0].metadata["abstract_interpretation_context"]["text"] += " altered"
    assert not admitted_role_is_linked(out.hypotheses[0], "direct_target_direction_primary", brief)


@pytest.mark.parametrize("alteration", ["claim", "source", "certificate", "target", "brief", "direction", "claim_link", "role", "offset"])
async def test_certificate_revalidated_after_every_mutation(brief, alteration):
    p = proposal(brief)
    out, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    h = out.hypotheses[0]
    if alteration == "claim": h.claims[0].text += " cures patients"
    elif alteration == "source": h.evidence[0].external_id = "different-paper"
    elif alteration == "certificate": h.target.metadata["ma3_admission"]["status"] = "hold"
    elif alteration == "target": h.target.symbol = "FLT30"
    elif alteration == "brief": brief.constraints.target_cell_type = "new-model"
    elif alteration == "direction": h.intervention_direction = "activate"
    elif alteration == "claim_link": h.claims[0].evidence_ids = ["ev_assoc"]
    elif alteration == "role": h.target.metadata["ma3_admission"]["claims"][0]["role"] = "primary_target_disease_association"
    elif alteration == "offset": h.target.metadata["ma3_admission"]["claims"][0]["experiment_start"] = 3
    assert not admitted_role_is_linked(h, "direct_target_direction_primary", brief)


async def test_plain_verified_boolean_is_never_a_certificate(brief):
    p = proposal(brief)
    p.target.metadata["ma3_admission"] = {"status": "approved", "verified": True}
    assert not admitted_role_is_linked(p, "direct_target_direction_primary", brief)


async def test_all_non_approved_statuses_preserve_originals(brief):
    proposals = [proposal(brief, "h" + str(i)) for i in range(4)]
    statuses = ["approved", "needs_evidence", "hold", "rejected"]
    out, report = await assess_provisional(brief, packet(brief, *proposals),
        Provider([assessment(p, status=s) for p, s in zip(proposals, statuses)]), MODEL)
    assert len(out.hypotheses) == 1
    assert [e["status"] for e in report["entries"]] == statuses
    assert all(entry["original_proposal"] for entry in report["entries"])
    assert not out.provisional_hypotheses


async def test_unavailable_reviewer_no_fallback_to_ma1_acceptance(brief):
    p = proposal(brief)
    out, report = await assess_provisional(brief, packet(brief, p), None, MODEL)
    assert not out.hypotheses
    assert report["entries"][0]["status"] == "hold"
    assert report["entries"][0]["reason"] == "REVIEWER_UNAVAILABLE"


async def test_failed_call_usage_persisted_and_budget_errors_propagate(brief):
    p = proposal(brief)
    usage = TokenUsage(component=COMPONENT, model=MODEL, input_tokens=888, output_tokens=222)
    provider = Provider(error=LLMProviderCallError("private upstream error", usage=usage))
    seen = []
    async def on_usage(item): seen.append(item)
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL, on_usage=on_usage)
    assert seen == [usage]
    assert not out.hypotheses
    assert "private upstream error" not in json.dumps(report)
    async def exhausted(_): raise RuntimeError("budget")
    with pytest.raises(RuntimeError, match="budget"):
        await assess_provisional(brief, packet(brief, p), provider, MODEL, on_usage=exhausted)


async def test_unknown_failed_usage_gets_nonzero_conservative_bound(brief):
    p = proposal(brief)
    seen = []
    async def on_usage(item): seen.append(item)
    await assess_provisional(brief, packet(brief, p), Provider(error=LLMProviderCallError("no usage", usage=None)), MODEL, on_usage=on_usage)
    assert len(seen) == 1 and seen[0].estimated and seen[0].input_tokens > 0


async def test_preflight_runs_before_call_and_success_usage_before_validation(brief):
    p = proposal(brief)
    provider = Provider([assessment(p, hypothesis_id="wrong")])
    seen = []
    async def preflight(model, prompt, output):
        seen.append("preflight")
        assert not provider.calls and model == MODEL and output == 5000
    async def usage(_): seen.append("usage")
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL, usage, preflight)
    assert seen == ["preflight", "usage"] and not out.hypotheses
    assert report["entries"][0]["reason"] == "REVIEW_HYPOTHESIS_MISMATCH"
    async def denied(*_): raise RuntimeError("preflight budget")
    provider = Provider([])
    with pytest.raises(RuntimeError, match="preflight budget"):
        await assess_provisional(brief, packet(brief, p), provider, MODEL, before_review=denied)
    assert not provider.calls


async def test_mock_usage_cannot_authorize_real_admission(brief):
    p = proposal(brief)
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)], usage_model="mock:" + MODEL), MODEL)
    assert not out.hypotheses
    assert report["entries"][0]["reason"] == "REVIEW_PROVIDER_MODEL_MISMATCH"


async def test_invalid_source_records_never_sent_as_evidence(brief):
    p = proposal(brief)
    p.evidence[0].excerpt_hash = "corrupt"
    p.evidence[1].metadata["fixture"] = True
    provider = Provider([])
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL)
    assert not provider.calls and not out.hypotheses
    assert report["entries"][0]["reason"] == "NO_VALID_SOURCE_BOUND_RECORDS"
    assert len(report["entries"][0]["omissions"]["omitted_records"]) == 2


async def test_hypothesis_capacity_is_disclosed_not_silent(brief):
    proposals = [proposal(brief, "h" + str(i)) for i in range(MAX_HYPOTHESES + 1)]
    provider = Provider([assessment(p, status="hold") for p in proposals[:MAX_HYPOTHESES]])
    out, report = await assess_provisional(brief, packet(brief, *proposals), provider, MODEL)
    assert len(provider.calls) == MAX_HYPOTHESES
    assert len(report["entries"]) == MAX_HYPOTHESES + 1
    assert report["entries"][-1]["reason"] == "REVIEW_CAPACITY_REACHED"


async def test_empty_policy_packet_returns_explicit_no_approved_report(brief):
    out, report = await assess_provisional(brief, packet(brief), None, MODEL)
    assert not out.hypotheses and report["submitted_count"] == 0 and report["approved_count"] == 0


async def test_legacy_packet_cannot_silently_change_admission_policy(brief):
    old = HypothesisPacket(brief_id=brief.brief_id, hypotheses=[proposal(brief)])
    with pytest.raises(ValueError, match="explicit packet policy"):
        await assess_provisional(brief, old, None, MODEL)


async def test_unresolved_broad_direction_resolved_only_from_direct_claim(brief):
    brief.constraints.intervention_direction = "unspecified"
    p = proposal(brief)
    p.intervention_direction = "modulate"
    p.target.metadata["direction_resolution"] = {"status": "direction_unresolved", "resolved_direction": None}
    a = assessment(p)
    out, report = await assess_provisional(brief, packet(brief, p), Provider([a]), MODEL)
    assert len(out.hypotheses) == 1
    assert out.hypotheses[0].intervention_direction == "inhibit"
    assert p.intervention_direction == "modulate"
    assert report["entries"][0]["original_proposal"]["intervention_direction"] == "modulate"
    assert report["entries"][0]["resolved_direction"] == "inhibit"
    assert admitted_support_ids(out.hypotheses[0]) == {"ev_dir", "ev_assoc"}
    brief.constraints.allowed_hypothesis_directions = ["activate"]
    out, _ = await assess_provisional(brief, packet(brief, p), Provider([a]), MODEL)
    assert not out.hypotheses


async def test_standalone_support_ids_revalidates_source_request_and_claim(brief):
    p = proposal(brief)
    for e in p.evidence:
        e.supports = "context"
    out, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    h = out.hypotheses[0]
    assert admitted_support_ids(h) == {"ev_dir", "ev_assoc"}
    assert all(e.supports == "context" for e in h.evidence)
    h.target.metadata["ma3_admission"]["request"]["disease"] = "unrelated disease"
    assert admitted_support_ids(h) == set()


async def test_review_publication_cannot_be_relabelled_primary(brief):
    p = proposal(brief)
    p.evidence[0].metadata["publication_types"] = ["Journal Article", "Review"]
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert not out.hypotheses
    assert "SOURCE_PUBLICATION_NOT_PRIMARY" in report["entries"][0]["validation_errors"]


async def test_duplicate_proposal_ids_all_hold_without_model_calls(brief):
    p = proposal(brief)
    provider = Provider([])
    out, report = await assess_provisional(brief, packet(brief, p, p.model_copy(deep=True)), provider, MODEL)
    assert not provider.calls and not out.hypotheses
    assert all(e["reason"] == "DUPLICATE_PROPOSAL_ID" for e in report["entries"])


async def test_true_negative_result_never_promoted_to_favorable(brief):
    p = proposal(brief)
    text = DIRECTION_TEXT.replace("reduced cell survival", "did not reduce cell survival")
    p.evidence[0] = evidence(text)
    negative = bound_claim(p, result_quote="Compound A did not reduce cell survival", outcome="unfavorable")
    a = assessment(p, claims=[negative, bound_claim(p, "primary_target_disease_association")])
    out, report = await assess_provisional(brief, packet(brief, p), Provider([a]), MODEL)
    assert not out.hypotheses
    assert "DIRECT_FAVORABLE_INTERVENTION_NOT_ESTABLISHED" in report["entries"][0]["validation_errors"]


async def test_noncontiguous_legacy_excerpt_not_certified_as_contiguous_source(brief):
    p = proposal(brief)
    p.evidence[0].metadata["source_location"] = "abstract-paired:0-30+200-270"
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert not out.hypotheses
    assert "SOURCE_BINDING_INVALID" in report["entries"][0]["validation_errors"]


async def test_request_disallowed_source_not_admitted(brief):
    p = proposal(brief)
    brief.data_policy.allowed_sources = ["Europe PMC"]
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert not out.hypotheses
    assert "SOURCE_NOT_ALLOWED_BY_REQUEST" in report["entries"][0]["validation_errors"]


class ReviewCache:
    def __init__(self):
        self.rows = {}
        self.dispatches = []
        self.saves = []

    async def load(self, key):
        return deepcopy(self.rows.get(key))

    async def dispatch(self, key):
        assert key not in self.rows, "Ambiguous/completed calls must not be redispatched"
        self.rows[key] = {"ambiguous": True}
        self.dispatches.append(key)

    async def save(self, key, payload):
        assert self.rows.get(key) == {"ambiguous": True}
        self.rows[key] = deepcopy(payload)
        self.saves.append(key)

    def callbacks(self):
        return {"load_review": self.load, "before_dispatch": self.dispatch, "save_review": self.save}


async def test_midbatch_budget_denial_preserves_first_paid_result_for_resume(brief):
    a, b = proposal(brief, "hyp_a"), proposal(brief, "hyp_b")
    submitted = packet(brief, a, b)
    provider = Provider([assessment(a), assessment(b)])
    cache = ReviewCache()
    billed = []
    checked = []
    async def usage(item): billed.append(item)
    async def limited_preflight(*_):
        checked.append(True)
        if len(checked) == 2:
            raise RuntimeError("budget denied before second dispatch")
    with pytest.raises(RuntimeError, match="budget denied"):
        await assess_provisional(brief, submitted, provider, MODEL, usage, limited_preflight, **cache.callbacks())
    assert len(provider.calls) == len(billed) == len(cache.dispatches) == len(cache.saves) == 1
    assert all(not payload.get("ambiguous") for payload in cache.rows.values())
    resumed_preflights = []
    async def preflight(*_): resumed_preflights.append(True)
    out, report = await assess_provisional(brief, submitted, provider, MODEL, usage, preflight, **cache.callbacks())
    assert len(out.hypotheses) == 2
    assert len(provider.calls) == len(billed) == len(cache.dispatches) == len(cache.saves) == 2
    assert len(resumed_preflights) == 1
    assert [entry["review_cache"] for entry in report["entries"]] == ["reused", "fresh"]


async def test_interrupted_dispatch_is_ambiguous_and_never_automatically_replayed(brief):
    import asyncio
    p = proposal(brief)
    cache = ReviewCache()
    provider = Provider(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await assess_provisional(brief, packet(brief, p), provider, MODEL, **cache.callbacks())
    assert len(provider.calls) == len(cache.dispatches) == 1 and not cache.saves
    provider.error = None
    provider.results = [assessment(p)]
    async def forbidden_callback(*_): raise AssertionError("No preflight/billing for ambiguous dispatch")
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL,
        forbidden_callback, forbidden_callback, **cache.callbacks())
    assert not out.hypotheses and len(provider.calls) == 1
    assert report["entries"][0]["reason"] == "AMBIGUOUS_PRIOR_REVIEW_NOT_REPLAYED"


async def test_complete_cached_result_can_be_used_without_provider_or_rebilling(brief):
    p = proposal(brief)
    submitted = packet(brief, p)
    cache = ReviewCache()
    provider = Provider([assessment(p)])
    out, _ = await assess_provisional(brief, submitted, provider, MODEL, **cache.callbacks())
    async def forbidden(*_): raise AssertionError("Already completed")
    resumed, report = await assess_provisional(brief, submitted, None, MODEL, forbidden, forbidden, **cache.callbacks())
    assert resumed.hypotheses == out.hypotheses
    assert len(provider.calls) == 1 and report["entries"][0]["review_cache"] == "reused"


async def test_failed_provider_result_saved_once_and_not_rebilled_on_retry(brief):
    p = proposal(brief)
    cache = ReviewCache()
    usage = TokenUsage(component=COMPONENT, model=MODEL, input_tokens=12, output_tokens=8)
    provider = Provider(error=LLMProviderCallError("private failure", usage=usage))
    billed = []
    async def on_usage(item): billed.append(item)
    for _ in range(2):
        out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL, on_usage, **cache.callbacks())
        assert not out.hypotheses and report["entries"][0]["reason"] == "REVIEWER_UNAVAILABLE"
    assert len(provider.calls) == len(billed) == len(cache.saves) == 1


async def test_invalid_structured_result_saved_as_failure_without_replay(brief):
    p = proposal(brief)
    cache = ReviewCache()
    provider = Provider([p])
    for _ in range(2):
        out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL, **cache.callbacks())
        assert not out.hypotheses and report["entries"][0]["reason"] == "REVIEW_OUTPUT_INVALID"
    assert len(provider.calls) == len(cache.saves) == 1


async def test_accounting_callback_failure_preserves_raw_response_but_cannot_authorize(brief):
    p = proposal(brief)
    cache = ReviewCache()
    provider = Provider([assessment(p)])
    async def fail(_): raise RuntimeError("usage persistence uncertain")
    with pytest.raises(RuntimeError, match="usage persistence uncertain"):
        await assess_provisional(brief, packet(brief, p), provider, MODEL, fail, **cache.callbacks())
    saved = next(iter(cache.rows.values()))
    assert saved["assessment"]["status"] == "approved"
    assert saved["failure"] == "USAGE_ACCOUNTING_NOT_CONFIRMED"
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL, fail, **cache.callbacks())
    assert not out.hypotheses and len(provider.calls) == 1
    assert report["entries"][0]["reason"] == "USAGE_ACCOUNTING_NOT_CONFIRMED"


async def test_invalid_scope_anchor_cannot_use_certificate_route(brief):
    p = proposal(brief)
    p.target.metadata["molecular_hypothesis_scope"] = {"label": "FLT3:ITD"}
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert not out.hypotheses
    assert "MOLECULAR_SCOPE_BINDING_INVALID" in report["entries"][0]["validation_errors"]


async def test_partial_durable_callback_configuration_rejected(brief):
    p = proposal(brief)
    async def load(_): return None
    with pytest.raises(ValueError, match="callbacks together"):
        await assess_provisional(brief, packet(brief, p), None, MODEL, load_review=load)


async def test_unread_known_counterevidence_requires_review_not_scientific_rejection(brief):
    p = proposal(brief)
    p.evidence.extend(evidence(f"Independent record {i} describes FLT3 research in acute myeloid leukemia.", f"context_{i}") for i in range(22))
    p.evidence.append(evidence("FLT3 inhibition increased leukemic expansion in acute myeloid leukemia cells.", "unread_counter"))
    for index, e in enumerate(p.evidence):
        e.external_id = f"synthetic-test-paper-{index}"
        e.supports = "counter"
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    row = report["entries"][0]
    assert not out.hypotheses
    assert row["status"] == "hold"
    assert row["reason"] == "KNOWN_COUNTEREVIDENCE_REVIEW_PENDING"
    assert row["pending_evidence_ids"] == ["unread_counter"]
    assert row["assessment"]["status"] == "approved"  # Original model judgment preserved.
    assert row["omissions"]["omitted_records"][-1]["recorded_support_label"] == "counter"
    assert "no negative conclusion" in row["limitations"][-1]


async def test_duplicate_counter_record_is_not_unread_capacity_counter(brief):
    p = proposal(brief)
    p.evidence[0].supports = "counter"
    duplicate = p.evidence[0].model_copy(deep=True)
    duplicate.evidence_id = "duplicate_counter"
    p.evidence.append(duplicate)
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert len(out.hypotheses) == 1
    assert report["entries"][0]["omissions"]["omitted_records"][-1]["reason"] == "duplicate_source_text"


@pytest.mark.parametrize("signal,category", [(True, None), ("increased leukemic expansion", "opposite_or_null_result")])
async def test_unread_source_only_context_counter_warning_requires_review(brief, signal, category):
    p = proposal(brief)
    p.evidence.extend(evidence(f"Independent record {i} describes FLT3 research in acute myeloid leukemia.", f"context_{i}") for i in range(22))
    for index, item in enumerate(p.evidence):
        item.external_id = f"synthetic-counter-paper-{index}"
        item.supports = "counter"
    text = "FLT3 inhibition increased leukemic expansion in acute myeloid leukemia cells."
    warning = evidence(text, "unread_context_warning")
    warning.external_id = "synthetic-unread-source-only"
    warning.supports = "context"
    warning.metadata.pop("excerpt")
    warning.metadata.pop("source_location")
    warning.excerpt_hash = None
    warning.metadata.update(abstract_interpretation_context=abstract_context(text),
        content_hash_scope="normalized_full_abstract", counter_signal=signal, counter_category=category)
    p.evidence.append(warning)
    original = p.model_dump_json()
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    row = report["entries"][0]
    assert not out.hypotheses and row["reason"] == "KNOWN_COUNTEREVIDENCE_REVIEW_PENDING"
    assert row["pending_evidence_ids"] == ["unread_context_warning"]
    omission = row["omissions"]["omitted_records"][-1]
    assert omission["recorded_support_label"] == "context" and omission["known_counter_warning"] is True
    assert p.model_dump_json() == original


def test_context_warning_is_prioritized_without_treating_rejection_codes_as_counter(brief):
    from drugpilot.agents.hypothesis_admission import _records
    p = proposal(brief)
    p.evidence.extend(evidence(f"Independent record {i} describes FLT3 research.", f"context_{i}") for i in range(23))
    for index, item in enumerate(p.evidence):
        item.external_id = f"synthetic-test-priority-{index}"
        item.supports = "context"
    p.evidence[-1].metadata.update(counter_signal="increased growth", counter_category="opposite_or_null_result")
    p.evidence[-2].metadata.update(rejection_codes=["counter_signal"], counter_signal=False)
    selected, omissions = _records(p)
    assert selected[0]["evidence_id"] == p.evidence[-1].evidence_id
    assert selected[0]["known_counter_warning"] is True
    assert selected[0]["recorded_support_label"] == "context"
    assert omissions["omitted_records"][-1]["evidence_id"] == p.evidence[-2].evidence_id
    assert omissions["omitted_records"][-1]["known_counter_warning"] is False


async def test_invalid_source_counter_flag_is_not_capacity_hold(brief):
    p = proposal(brief)
    invalid = evidence("An invalid source is not evidence.", "invalid_counter_source")
    invalid.metadata["counter_signal"] = True
    invalid.excerpt_hash = "f" * 64
    p.evidence.append(invalid)
    out, report = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert len(out.hypotheses) == 1
    assert report["entries"][0]["omissions"]["omitted_records"][0]["reason"] == "invalid_or_nonprimary_source_binding"


def source_only_proposal(brief):
    p = proposal(brief)
    a = assessment(p)
    for e, bound in zip(p.evidence, a.claims):
        text = e.metadata.pop("excerpt")
        e.excerpt_hash = None
        e.metadata.pop("source_location", None)
        e.metadata.update(abstract_interpretation_context=abstract_context(text),
                          content_hash_scope="normalized_full_abstract", content_hash=sha256(text.encode()).hexdigest())
        bound.source_binding_sha256 = source_binding(e)
    return p, a


async def test_source_only_full_abstract_reviewed_without_inventing_excerpt_or_primary_flags(brief):
    p, a = source_only_proposal(brief)
    original = p.model_dump_json()
    assert all(interpretation_text(e) is None for e in p.evidence)  # Legacy contract unchanged.
    assert all(source_review_text(e) is not None for e in p.evidence)
    provider = Provider([a])
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL)
    assert len(out.hypotheses) == 1
    assert all(admitted_role_is_linked(out.hypotheses[0], role, brief) for role in ROLES)
    assert p.model_dump_json() == original and out.hypotheses[0].evidence == p.evidence
    assert all(e.excerpt_hash is None and "excerpt" not in e.metadata for e in out.hypotheses[0].evidence)
    assert all("verified_direct_primary" not in e.metadata for e in out.hypotheses[0].evidence)
    records = json.loads(provider.calls[0]["prompt"])["source_records"]
    assert all(record["source_scope"] == "complete_normalized_abstract" for record in records)
    assert all(record["source_location"] == "abstract:complete" for record in records)


@pytest.mark.parametrize("change", ["wrong_hash", "incomplete", "present_corrupt_excerpt", "wrong_size", "mock"])
async def test_bad_full_abstract_never_becomes_fallback_primary_source(brief, change):
    p, a = source_only_proposal(brief)
    for e in p.evidence:
        if change == "wrong_hash": e.metadata["content_hash"] = "f" * 64
        elif change == "incomplete": e.metadata["abstract_interpretation_context"]["status"] = "omitted_size_limit"
        elif change == "present_corrupt_excerpt":
            e.metadata["excerpt"] = "Unverifiable excerpt"
            e.excerpt_hash = "f" * 64
        elif change == "wrong_size": e.metadata["abstract_interpretation_context"]["char_count"] += 1
        elif change == "mock": e.metadata["mock"] = True
    provider = Provider([a])
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL)
    assert not out.hypotheses and not provider.calls
    assert report["entries"][0]["reason"] == "NO_VALID_SOURCE_BOUND_RECORDS"


async def test_proposed_broad_direction_is_question_until_supported_and_not_silently_flipped(brief):
    brief.constraints.intervention_direction = "unspecified"
    p = proposal(brief)
    p.target.metadata["direction_resolution"] = {"status": "direction_proposed_for_ma3",
        "requested_direction": "unspecified", "resolved_direction": None, "proposed_direction": "inhibit"}
    out, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    assert len(out.hypotheses) == 1
    assert out.hypotheses[0].target.metadata["direction_resolution"]["status"] == "resolved_from_source"
    assert out.hypotheses[0].target.metadata["direction_resolution"]["resolved_direction"] == "inhibit"
    assert p.target.metadata["direction_resolution"]["resolved_direction"] is None
    wrong = assessment(p)
    wrong.claims[0].observed_direction = "activate"
    out, report = await assess_provisional(brief, packet(brief, p), Provider([wrong]), MODEL)
    assert not out.hypotheses
    assert "DIRECTION_MISMATCH" in report["entries"][0]["validation_errors"]


async def test_full_request_context_reaches_review_without_lexical_scope_becoming_requirement(brief):
    brief.rationale = "FLT3-ITD is background context, not an exclusive inclusion condition."
    brief.human_priorities = ["Compare with FLT3 wild type; do not require either genotype."]
    p, a = source_only_proposal(brief)
    for claim in a.claims:
        claim.model_genotype_quote = ""
    provider = Provider([a])
    out, _ = await assess_provisional(brief, packet(brief, p), provider, MODEL)
    request = json.loads(provider.calls[0]["prompt"])["request"]
    assert request["rationale"] == brief.rationale
    assert request["human_priorities"] == brief.human_priorities
    assert request["constraints"] == brief.constraints.model_dump(mode="json")
    assert request["mentioned_scope_labels"] == ["FLT3:ITD", "FLT3:WT"]
    assert request["scope_mention_authority"] == "lexical_context_not_exclusive_requirements"
    assert len(out.hypotheses) == 1
    assert "molecular_hypothesis_scope" not in out.hypotheses[0].target.metadata


async def test_excluded_variant_mention_does_not_create_typed_scope_requirement(brief):
    brief.rationale = "Do not limit the question to FLT3-ITD; this genotype is excluded from the requested scope."
    p = proposal(brief)
    provider = Provider([assessment(p, status="needs_evidence", reason="Current models do not meet the stated exclusion.")])
    out, report = await assess_provisional(brief, packet(brief, p), provider, MODEL)
    assert not out.hypotheses
    request = json.loads(provider.calls[0]["prompt"])["request"]
    assert request["rationale"] == brief.rationale and request["mentioned_scope_labels"] == ["FLT3:ITD"]
    assert "molecular_hypothesis_scope" not in p.target.metadata
    assert report["entries"][0]["status"] == "needs_evidence"


async def test_claim_placeholder_confidence_is_explicit_and_certificate_bound(brief):
    p = proposal(brief)
    out, _ = await assess_provisional(brief, packet(brief, p), Provider([assessment(p)]), MODEL)
    h = out.hypotheses[0]
    assert all(claim.confidence == .5 for claim in h.claims)
    assert h.target.metadata["ma3_admission"]["confidence_semantics"] == CONFIDENCE_SEMANTICS
    assert admitted_support_ids(h)
    h.target.metadata["ma3_admission"]["confidence_semantics"] = "calibrated_probability"
    assert not admitted_support_ids(h)
