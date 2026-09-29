"""Isolated synthetic contracts. Never used as live research evidence."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

from drugpilot.agents.evidence_semantics import EvidenceMeaningBatch, attach_interpretations, source_records, permits, source_binding
from drugpilot.agents.ma1 import Proposer, TargetSeed, Critic, _expand_molecular_hypotheses, _primary_direction_excerpt
from drugpilot.agents.ma3 import _retained_primary_role_is_linked
from drugpilot.agents.molecular_scope import valid_hypothesis_scope, functional_association_sentence
from drugpilot.domain.models import HypothesisTrack
from drugpilot.tools.public_biology import _bounded_primary_association_excerpt
from drugpilot.tools.assay_variant_alignment import assay_variant_alignment, extract_target_variants
from test_evidence_semantics import evidence, meaning


def _scoped(brief, variant="ITD"):
    e=evidence(f"FLT3-{variant} inhibition significantly reduced viability in acute myeloid leukemia cells.",external_id=f"synthetic-{variant}")
    m=meaning(e,target_quote=f"FLT3-{variant}",intervention_quote=f"FLT3-{variant} inhibition",molecular_scope_label=f"FLT3:{variant}")
    b=EvidenceMeaningBatch(records=[m])
    return attach_interpretations(brief,"FLT3",[e],source_records([e]),b,b)[0]


def _proposal(brief, items):
    retriever=SimpleNamespace(calls=0)
    def retrieve(*args):
        retriever.calls+=1
        return deepcopy(items)
    retriever.retrieve=retrieve
    p=Proposer(retriever)
    seed=TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,0.5)
    return p._build_hypothesis(brief,seed,1),retriever


def test_scoped_agreement_admits_only_that_scope_not_general_or_other_variant(brief):
    e=_scoped(brief)
    assert permits(e,brief,"FLT3","direction","FLT3:ITD")
    assert not permits(e,brief,"FLT3","direction")
    assert not permits(e,brief,"FLT3","direction","FLT3:D835Y")
    assert _primary_direction_excerpt(brief,"FLT3",e,"FLT3:ITD")
    e.metadata["publication_types"]=["Review"]
    assert _primary_direction_excerpt(brief,"FLT3",e,"FLT3:ITD") is None


def test_independent_scoped_proposals_share_no_positive_claims_and_preserve_source(brief):
    items=[_scoped(brief),_scoped(brief,"D835Y")]
    before=brief.model_dump_json(),[e.model_dump_json() for e in items]
    base,retriever=_proposal(brief,items)
    branches=_expand_molecular_hypotheses(brief,base)
    assert len(branches)==3 and retriever.calls==1
    assert len({h.hypothesis_id for h in branches})==3
    scoped=branches[1:]
    for h in scoped:
        label=h.target.metadata["molecular_hypothesis_scope"]["label"]
        assert valid_hypothesis_scope(brief,h)
        assert h.target.symbol=="FLT3"
        units=h.target.metadata["claim_evidence_assignment"]["claim_units"]
        positive=[u for u in units if u["role"]=="direct_target_direction_primary"]
        assert len(positive)==1 and positive[0]["molecular_scope_label"]==label
        assert positive[0]["evidence_id"]==f"test-PubMed-synthetic-{label.split(':')[1]}"
        assert _retained_primary_role_is_linked(h,"direct_target_direction_primary",brief)
        assert not Critic().review(brief,h).accepted  # association remains missing
        assert [source_binding(e) for e in h.evidence]==[source_binding(e) for e in items]
    assert before==(brief.model_dump_json(),[e.model_dump_json() for e in items])


@pytest.mark.parametrize("mutation",["label","anchor","remove","claim_scope"])
def test_scoped_metadata_cannot_be_forged_or_removed_to_pass_g0(brief,mutation):
    base,_=_proposal(brief,[_scoped(brief)])
    h=_expand_molecular_hypotheses(brief,base)[1]
    if mutation=="label": h.target.metadata["molecular_hypothesis_scope"]["label"]="FLT3:D835Y"
    if mutation=="anchor": h.target.metadata["molecular_hypothesis_scope"]["anchors"][0]["start"]+=1
    if mutation=="remove": del h.target.metadata["molecular_hypothesis_scope"]
    if mutation=="claim_scope":
        h.target.metadata["claim_evidence_assignment"]["claim_units"][0]["molecular_scope_label"]="FLT3:D835Y"
    assert not _retained_primary_role_is_linked(h,"direct_target_direction_primary",brief)


def test_declared_scope_excludes_other_observed_variants_without_rewriting_request(brief):
    brief.human_priorities=["Evaluate FLT3-ITD"]
    base,_=_proposal(brief,[_scoped(brief),_scoped(brief,"D835Y")])
    branches=_expand_molecular_hypotheses(brief,base)
    assert [b.target.metadata["molecular_hypothesis_scope"]["label"] for b in branches[1:]]==["FLT3:ITD"]


@pytest.mark.parametrize("text,expected",[
    ("Results: ABC1-XYZ2 fusion protein has been demonstrated to transform lung cancer cells in vitro.",True),
    ("Results: ABC1-XYZ2 fusion protein did not transform lung cancer cells in vitro.",False),
    ("ABC1-XYZ2 fusion protein may transform lung cancer cells in vitro.",False),
    ("Results: XYZ2 was measured; ABC1 promotes proliferation in lung cancer cells.",False),
    ("Results: It is not true that ABC1-XYZ2 fusion protein transformed lung cancer cells.",False),
    ("Results: We proposed that ABC1-XYZ2 fusion protein promotes growth in lung cancer cells.",False),
])
def test_functional_relation_is_target_subject_bound_not_a_drug_effect(text,expected):
    assert functional_association_sentence(text,"XYZ2")==expected


def test_functional_association_is_distinct_from_cohort_prevalence_and_direction():
    text="The ABC1-XYZ2 fusion protein has been demonstrated to transform lung cancer cell lines in vitro."
    selected=_bounded_primary_association_excerpt([("RESULTS",text)],target_symbol="XYZ2",disease="lung cancer")
    assert selected and selected[3]=="functional_target_disease_association"
    assert _bounded_primary_association_excerpt([("BACKGROUND",text)],target_symbol="XYZ2",disease="lung cancer") is None
    assert _bounded_primary_association_excerpt([("ABSTRACT",text)],target_symbol="XYZ2",disease="lung cancer") is None


def test_fusion_assay_scope_is_not_lost_or_assumed_to_match():
    assert extract_target_variants("ALK",["EML4::ALK fusion"])==("EML4::ALK",)
    scoped={"policy":"source_bound_molecular_hypothesis_v1","label":"EML4::ALK",
        "anchors":[{"quote":"EML4::ALK"}]}
    result=assay_variant_alignment(target_symbol="ALK",hypothesis_texts=["ALK inhibition"],
        molecular_scope=scoped,chembl_assay_context={})
    assert result["declared_hypothesis_variants"]==["EML4::ALK"]
    assert result["status"]=="unknown_not_assumed_match"
    assert not result["construct_identity_verified"]
    assert result["molecular_hypothesis_scope"]==scoped


def test_scope_branch_overflow_is_explicit_not_an_invisible_drop(brief):
    base,_=_proposal(brief,[_scoped(brief,f"D{i}Y") for i in range(1,11)])
    result=_expand_molecular_hypotheses(brief,base)
    assert len(result)==9
    trace=result[0].target.metadata["molecular_scope_branching"]
    assert trace["observed_scope_count"]==10 and len(trace["unassessed_labels"])==2


def test_scoped_branch_preserves_original_prospective_protocol(brief):
    base,_ = _proposal(brief,[_scoped(brief)])
    protocol = (
        "Measure differentiation at 48 hours in patient-derived cultures with "
        "non-targeting controls and three independent replicates; reject below the "
        "pre-registered 12% threshold and confirm construct-specific rescue."
    )
    base.falsification_test = protocol
    branch = _expand_molecular_hypotheses(brief,base)[1]
    assert branch.falsification_test.startswith(protocol + "\n")
    assert "Molecular scope for this hypothesis: FLT3:ITD" in branch.falsification_test
    assert "30%" not in branch.falsification_test
    assert "two models" not in branch.falsification_test
    assert base.falsification_test == protocol
