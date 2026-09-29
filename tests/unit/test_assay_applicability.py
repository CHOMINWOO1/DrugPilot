"""Synthetic contracts only. Never primary evidence or live candidate fixtures."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma1 import Proposer, Critic, _expand_molecular_hypotheses
from drugpilot.agents.molecular_scope import HYPOTHESIS_SCOPE_KEY
from drugpilot.tools.assay_applicability import deposited_review, scope_comparison, annotate_hypothesis, KEY
from drugpilot.tools.assay_identity import digest
from test_assay_identity import Chemistry, run
from test_assay_followup import seed


def inventory(brief, *, mutation=None, variant=None, description=None, accession=None):
    client=Chemistry()
    client.data['activity'][2][0].update(assay_variant_mutation=mutation, assay_variant_accession=accession)
    client.data['assay'][2][0].update(variant_sequence=variant, description=description)
    result,_=run(brief,client)
    assert result.status=='completed'
    return result.records[0],result.trace


@pytest.mark.parametrize('mutation,status,labels',[
    (None,'not_recorded',[]),('', 'not_recorded',[]),
    ('UNDEFINED MUTATION','unresolved_annotation',[]),('unknown','unresolved_annotation',[]),
    ('D835Y not determined','unresolved_annotation',[]),('N/A','unresolved_annotation',[]),
    ('D835Y','named_annotation_only',['FLT3:D835Y']),
    ('p.D835Y / F691L','named_annotation_only',['FLT3:D835Y','FLT3:F691L']),
    ('D835Y resistant?','unresolved_annotation',[]),
    ('WT','named_annotation_only',['FLT3:WT']),('ITD','class_annotation_only',['FLT3:ITD']),
    ('D0Y','unresolved_annotation',[]),('O835Y','unresolved_annotation',[]),
    ({'mutation':'D835Y'},'unresolved_annotation',[]),
])
def test_typed_annotations_not_inferred_from_missing_or_arbitrary_prose(brief,mutation,status,labels):
    e,trace=inventory(brief,mutation=mutation)
    a=e.metadata['assay_identity_inventory']; row=a['linked_activities'][0]
    review=row['deposited_applicability_review']
    assert review['status']==status and review['deposited_labels']==labels
    assert review['wild_type_inferred'] is False and review['exact_construct_verified'] is False
    assert review['full_genotype_recorded'] is False
    assert trace['assay_applicability_summary']['exclusive_annotation_counts']=={status:1}
    assert a['annotation_sha256']==digest({k:v for k,v in a.items() if k!='annotation_sha256'})


def test_itd_description_cannot_resolve_undefined_structured_mutation(brief):
    e,_=inventory(brief,mutation='UNDEFINED MUTATION',variant={'mutation':'UNDEFINED MUTATION','sequence':''},
        description='Inhibition of recombinant FLT3 ITD mutant expressed in insect cells.')
    row=e.metadata['assay_identity_inventory']['linked_activities'][0]
    before=deepcopy(row)
    r=deposited_review(row)
    assert r['status']=='unresolved_annotation' and r['deposited_labels']==[]
    assert r['description_mentions'][0]['label']=='FLT3:ITD'
    assert r['description_mentions'][0]['assertion_status']=='mention_not_presence_or_effect'
    assert scope_comparison(r,'FLT3:ITD')=='source_annotation_unresolved'
    assert r['condition_field_coverage']['assay_cell_type']=='not_recorded'
    assert r['condition_field_coverage']['assay_parameters']=='not_recorded'
    assert row==before


@pytest.mark.parametrize('variant,accession,reason',[
    ({'mutation':'F691L'},None,'DEPOSITED_MUTATION_FIELDS_DISAGREE'),
    ({'mutation':'D835Y','accession':'P99999'},None,'VARIANT_ACCESSION_DIFFERS_FROM_TARGET_COMPONENT'),
    ({'mutation':'D835Y','organism':'Mus musculus'},None,'VARIANT_ORGANISM_DIFFERS_FROM_TARGET_COMPONENT'),
    ({'mutation':'D835Y'},'P99999','VARIANT_ACCESSION_DIFFERS_FROM_TARGET_COMPONENT'),
])
def test_conflicts_are_preserved_and_not_used_as_positive_alignment(brief,variant,accession,reason):
    e,_=inventory(brief,mutation='D835Y',variant=variant,accession=accession)
    row=e.metadata['assay_identity_inventory']['linked_activities'][0]
    r=row['deposited_applicability_review']
    assert r['status']=='annotation_conflict' and reason in r['conflict_reasons']
    assert scope_comparison(r,'FLT3:D835Y')=='source_annotation_conflict'
    assert row['deposited_assay_conditions']['variant_sequence']==variant


@pytest.mark.parametrize('mutation,label,expected',[
    ('D835Y','FLT3:D835Y','same_label_not_complete_genotype'),
    ('D835Y,F691L','FLT3:D835Y','partial_label_overlap'),
    ('D835Y','FLT3:ITD','different_labels_equivalence_unestablished'),
    ('WT','FLT3:D835Y','different_labels_equivalence_unestablished'),
    ('ITD','FLT3:ITD','same_variant_class_not_exact_construct'),
    ('D835Y',None,'no_bound_molecular_scope'),
    ('D835Y','ABC1::FLT3','fusion_construct_not_established_by_component_assay'),
])
def test_label_comparison_never_claims_equivalence_or_mutual_exclusion(brief,mutation,label,expected):
    e,_=inventory(brief,mutation=mutation)
    review=e.metadata['assay_identity_inventory']['linked_activities'][0]['deposited_applicability_review']
    assert scope_comparison(review,label)==expected


def test_sequence_and_parameters_presence_is_not_validation(brief):
    e,_=inventory(brief,mutation='D835Y',variant={'mutation':'D835Y','accession':'P00001','sequence':'MABC'})
    row=e.metadata['assay_identity_inventory']['linked_activities'][0]
    row['deposited_assay_conditions']['assay_parameters']=[{'type':'TIME','value':'1','units':'hr'}]
    review=deposited_review(row)
    assert review['sequence_recorded'] and not review['sequence_equivalence_verified']
    assert review['condition_field_coverage']['assay_parameters']=='recorded_not_validated'
    assert not review['disease_model_match_verified']


def proposal(brief,e):
    return Proposer(SimpleNamespace(retrieve=lambda *args:[e]))._build_hypothesis(brief,seed(),1)


def test_request_prose_is_not_a_bound_hypothesis_and_no_gate_changes(brief):
    e,_=inventory(brief,mutation='D835Y')
    brief.research_goal='Do not assume FLT3-D835Y; compare FLT3-ITD contexts.'
    h=proposal(brief,e); before=h.model_dump_json()
    result=annotate_hypothesis(brief,h); a=result.target.metadata[KEY]
    assert h.model_dump_json()==before
    assert a['scope_status']=='not_bound' and a['scope_label'] is None
    assert a['activities'][0]['scope_comparison']=='no_bound_molecular_scope'
    assert result.evidence==h.evidence and result.claims==h.claims
    assert Critic().review(brief,result).reason_codes==Critic().review(brief,h).reason_codes
    assert not a['primary_support_eligible'] and not a['candidate_excluded']


@pytest.mark.parametrize('tamper',['digest','parent','fixture','pmid','scope'])
def test_invalid_inventory_or_scope_cannot_be_used_for_alignment(brief,tamper):
    e,_=inventory(brief,mutation='D835Y')
    h=proposal(brief,e)
    a=h.evidence[0].metadata['assay_identity_inventory']
    if tamper=='digest': a['linked_activities'][0]['measurement']['standard_value']='999'
    if tamper=='parent': a['parent_evidence_id']='different'
    if tamper=='fixture': a['fixture']=True
    if tamper=='pmid': a['pmid']='9999'
    if tamper=='scope': h.target.metadata[HYPOTHESIS_SCOPE_KEY]={'policy':'source_bound_molecular_hypothesis_v1','label':'FLT3:D835Y'}
    if tamper!='digest': a['annotation_sha256']=digest({k:v for k,v in a.items() if k!='annotation_sha256'})
    result=annotate_hypothesis(brief,h).target.metadata[KEY]
    if tamper=='scope':
        assert result['scope_status']=='invalid_scope' and result['scope_label'] is None
    else:
        assert result['reviewed_activity_count']==0 and result['invalid_inventory_evidence_ids']


def test_production_branching_applies_review_without_new_retrieval_or_primary_claim(brief):
    e,_=inventory(brief,mutation='D835Y')
    h=proposal(brief,e)
    # The assay context contains no bound molecular-scope claim.
    result=_expand_molecular_hypotheses(brief,h)
    assert len(result)==1 and result[0].target.metadata[KEY]['reviewed_activity_count']==1
    assert result[0].evidence==h.evidence and result[0].claims==h.claims
    assert not Critic().review(brief,result[0]).accepted


def test_valid_source_bound_scope_is_compared_after_branching(brief):
    from test_molecular_hypothesis_branching import _scoped
    e,_=inventory(brief,mutation='D835Y')
    items=[e,_scoped(brief,'D835Y'),_scoped(brief,'ITD')]
    h=Proposer(SimpleNamespace(retrieve=lambda *args:deepcopy(items)))._build_hypothesis(brief,seed(),1)
    result=_expand_molecular_hypotheses(brief,h)
    by_scope={r.target.metadata[KEY]['scope_label']:r for r in result}
    assert by_scope['FLT3:D835Y'].target.metadata[KEY]['activities'][0]['scope_comparison']=='same_label_not_complete_genotype'
    assert by_scope['FLT3:ITD'].target.metadata[KEY]['activities'][0]['scope_comparison']=='different_labels_equivalence_unestablished'
    for r in result:
        a=r.target.metadata[KEY]
        assert a['annotation_sha256']==digest({k:v for k,v in a.items() if k!='annotation_sha256'})
        without=r.model_copy(deep=True); del without.target.metadata[KEY]
        assert Critic().review(brief,without).reason_codes==Critic().review(brief,r).reason_codes


def test_rejected_and_screened_audits_keep_bounded_diagnostic_summary(brief):
    from drugpilot.agents.ma1 import _rejected_screening_entry, _screened_target_source
    e,_=inventory(brief,mutation='D835Y')
    h=_expand_molecular_hypotheses(brief,proposal(brief,e))[0]
    critique=Critic().review(brief,h)
    for f in (_rejected_screening_entry,_screened_target_source):
        record=f(critique,iteration=1,screening_phase='test')
        a=record['assay_applicability_summary']
        assert a['reviewed_activity_count']==1 and a['scope_comparison_counts']=={'no_bound_molecular_scope':1}
        assert 'activities' not in a and not a['scientific_verdict_changed']
