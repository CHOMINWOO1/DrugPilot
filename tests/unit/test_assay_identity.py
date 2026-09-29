"""Synthetic transport/identity contracts. No live evidence or candidates."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from drugpilot.tools.assay_identity import BASE, digest, resolve_assay_identities
from drugpilot.agents.ma1 import _assign_claim_evidence
from drugpilot.agents.evidence_semantics import source_records
from drugpilot.orchestration.g0_followup import build_plan, FollowupRetriever
from test_assay_followup import Client, public, seed, requested, packet, verdict


class Chemistry:
    def __init__(self,count=1):
        self.calls=[]
        self.failure=None
        self.data={
            'document':('documents','document_chembl_id',[{'document_chembl_id':'CHEMBL1','pubmed_id':1000}]),
            'activity':('activities','activity_id',[{'activity_id':i+1,'record_id':1,'document_chembl_id':'CHEMBL1',
                'assay_chembl_id':'CHEMBL2','molecule_chembl_id':'CHEMBL3','target_chembl_id':'CHEMBL4',
                'standard_type':'IC50','standard_value':'12.0','standard_units':'nM','standard_relation':'=',
                'assay_type':'B','data_validity_comment':None,'potential_duplicate':0} for i in range(count)]),
            'compound_record':('compound_records','record_id',[{'record_id':1,'document_chembl_id':'CHEMBL1',
                'molecule_chembl_id':'CHEMBL3','compound_key':'4','compound_name':'Deposited name'}]),
            'target':('targets','target_chembl_id',[{'target_chembl_id':'CHEMBL4','target_type':'SINGLE PROTEIN',
                'organism':'Homo sapiens','target_components':[{'component_type':'PROTEIN','accession':'P00001',
                    'target_component_synonyms':[{'component_synonym':'FLT3','syn_type':'GENE_SYMBOL'}]}]}]),
            'assay':('assays','assay_chembl_id',[{'assay_chembl_id':'CHEMBL2','document_chembl_id':'CHEMBL1',
                'target_chembl_id':'CHEMBL4','assay_type':'B','assay_parameters':[],'variant_sequence':None}]),
            'molecule':('molecules','molecule_chembl_id',[{'molecule_chembl_id':'CHEMBL3',
                'molecule_structures':{'canonical_smiles':'CCO','standard_inchi_key':'TEST_ONLY'}}]),
        }

    def get(self,url,**kwargs):
        assert url.startswith(BASE)
        params=kwargs['params']; self.calls.append((url,params))
        if self.failure=='timeout': raise httpx.ReadTimeout('private data must not be logged')
        if self.failure=='503': return httpx.Response(503,request=httpx.Request('GET',url))
        kind=url.removeprefix(BASE).removesuffix('.json')
        key,identity,all_rows=self.data[kind]
        offset,limit=params['offset'],params['limit']
        rows=deepcopy(all_rows[offset:offset+limit])
        total=len(all_rows)
        if self.failure=='changed_total' and offset: total+=1
        if self.failure=='duplicate_page' and offset and rows: rows[0][identity]=all_rows[0][identity]
        if self.failure=='empty_page' and offset: rows=[]
        meta={'offset':offset,'limit':limit,'total_count':total,'next':'https://untrusted.invalid/do-not-fetch' if offset+len(rows)<total else None}
        return httpx.Response(200,json={key:rows,'page_meta':meta},request=httpx.Request('GET',url))


def run(brief,client,*,target=None,seconds=120):
    p=public(Client(['Compound 4 inhibited FLT3 with IC50 of 12 nM.']))
    original,_=p.retrieve_g0_followup(brief,seed(),roles=['assay_context'])
    p.client=client
    before=[e.model_dump_json() for e in original]
    outcome=resolve_assay_identities(p,brief,target or seed(),original,elapsed_seconds=seconds)
    assert before==[e.model_dump_json() for e in original]
    return outcome,original


def test_exact_document_record_assay_molecule_join_is_context_only(brief):
    client=Chemistry()
    result,old=run(brief,client)
    assert result.status=='completed' and len(client.calls)==6
    assert result.trace['linked_activity_count']==1
    e=result.records[0]; a=e.metadata['assay_identity_inventory']; row=a['linked_activities'][0]
    assert e.evidence_id==old[0].evidence_id and e.excerpt_hash==old[0].excerpt_hash
    assert row['database_identity_chain_verified'] and row['document_compound_key']=='4'
    assert row['compound_key_mentions_in_excerpt'][0]['quote']=='Compound 4'
    assert row['measurement']['standard_value']=='12.0' and row['measurement']['standard_relation']=='='
    assert row['deposited_assay_conditions']['variant_sequence'] is None
    assert not row['construct_identity_verified'] and not row['excerpt_measurement_assignment_verified']
    assert row['molecule_chembl_id']=='CHEMBL3' and row['target_component']['canonical_gene']=='FLT3'
    assert a['annotation_sha256']==digest({k:v for k,v in a.items() if k!='annotation_sha256'})
    assert not source_records(result.records)
    claims=_assign_claim_evidence(brief,target_symbol='FLT3',evidence=result.records)
    assert not claims.direction_primary_ids and not claims.association_primary_ids


def test_all_41_rows_paginated_without_top20_cutoff_or_following_next_url(brief,monkeypatch):
    monkeypatch.setattr('drugpilot.tools.assay_identity.PAGE_SIZE',20)
    client=Chemistry(41); result,_=run(brief,client)
    assert result.status=='completed' and result.trace['linked_activity_count']==41
    assert result.trace['activity_rows_screened']==41 and result.trace['result_row_limit'] is None
    calls=[params for url,params in client.calls if url.endswith('/activity.json')]
    assert [c['offset'] for c in calls]==[0,20,40]
    assert all(c['order_by']=='activity_id' for c in calls)


@pytest.mark.parametrize('failure',['timeout','503','changed_total','duplicate_page','empty_page'])
def test_failed_or_inconsistent_pages_keep_original_records_only(brief,monkeypatch,failure):
    monkeypatch.setattr('drugpilot.tools.assay_identity.PAGE_SIZE',20)
    client=Chemistry(41); client.failure=failure
    result,old=run(brief,client)
    assert result.status=='error' and result.records==old
    assert not any('assay_identity_inventory' in e.metadata for e in result.records)
    assert 'private data' not in str(result.trace)


@pytest.mark.parametrize('resource,field,value',[
    ('document','pubmed_id',9999),('activity','document_chembl_id','CHEMBL9'),
    ('activity','standard_type','EC50'),('activity','activity_id',True),('activity','record_id',0),
    ('activity','target_organism','Mus musculus'),
    ('compound_record','molecule_chembl_id','CHEMBL9'),('compound_record','document_chembl_id','CHEMBL9'),
    ('assay','document_chembl_id','CHEMBL9'),('assay','target_chembl_id','CHEMBL9'),
    ('assay','assay_type','F'),('molecule','molecule_chembl_id','CHEMBL9'),
])
def test_cross_document_target_or_identity_mismatch_never_joins(brief,resource,field,value):
    client=Chemistry(); client.data[resource][2][0][field]=value
    result,old=run(brief,client)
    assert result.status=='error' and result.records==old


@pytest.mark.parametrize('case',['different_gene','alias_only','mouse','complex','accession_mismatch'])
def test_other_gene_species_complex_and_accession_are_not_claimed_as_target(brief,case):
    client=Chemistry(); target=seed(); t=client.data['target'][2][0]
    part=t['target_components'][0]
    if case=='different_gene': part['target_component_synonyms'][0]['component_synonym']='ABL1'
    if case=='alias_only': part['target_component_synonyms'][0]['syn_type']='GENE_SYMBOL_OTHER'
    if case=='mouse': t['organism']='Mus musculus'
    if case=='complex': t['target_type']='PROTEIN COMPLEX'
    if case=='accession_mismatch': target=replace(target,uniprot_id='P99999')
    result,_=run(brief,client,target=target)
    assert result.status=='completed' and result.trace['linked_activity_count']==0
    assert len(client.calls)==4  # no assay/molecule lookups for other targets


def test_no_document_match_is_not_biological_absence(brief):
    client=Chemistry(); client.data['document'][2].clear()
    result,_=run(brief,client)
    assert result.status=='completed' and len(client.calls)==1
    assert result.trace['documents_found']==result.trace['linked_activity_count']==0
    a=result.records[0].metadata['assay_identity_inventory']
    assert a['documents']==[] and not a['primary_support_eligible']


def test_variant_annotations_are_preserved_but_not_exact_construct_identity(brief):
    client=Chemistry()
    client.data['activity'][2][0].update(assay_variant_accession='P00001',assay_variant_mutation='D835Y')
    result,_=run(brief,client)
    row=result.records[0].metadata['assay_identity_inventory']['linked_activities'][0]
    assert row['deposited_activity_context']['assay_variant_mutation']=='D835Y'
    assert row['deposited_assay_conditions']['variant_sequence'] is None
    assert not row['construct_identity_verified'] and not row['disease_model_match_verified']


@pytest.mark.parametrize('resource',['compound_record','assay'])
def test_conflicting_source_ids_are_not_silently_joined(brief,resource):
    client=Chemistry()
    client.data['activity'][2][0]['src_id']=1
    client.data[resource][2][0]['src_id']=2
    result,old=run(brief,client)
    assert result.status=='error' and result.records==old


@pytest.mark.parametrize('kind',['fixture','mock','excerpt','pmid'])
def test_invalid_parent_cannot_drive_lookup(brief,kind):
    p=public(Client()); parents,_=p.retrieve_g0_followup(brief,seed(),roles=['assay_context'])
    bad=parents[0].model_copy(deep=True)
    bad.metadata[kind]=True if kind in {'fixture','mock'} else 'changed'
    client=Chemistry(); p.client=client
    result=resolve_assay_identities(p,brief,seed(),[bad])
    assert result.status=='not_attempted' and not client.calls


def test_outer_source_budget_defers_identity_join_without_extra_calls(brief):
    client=Client()
    records,rows=public(client,max_source_calls=1).retrieve_g0_followup(brief,seed(),roles=['assay_context','assay_identity'])
    assert len(records)==1 and len(client.calls)==3
    assert rows[-1]['reason_code']=='SOURCE_OR_TIME_BUDGET' and rows[-1]['attempted'] is False


def test_allowlist_and_deadline_prevent_external_calls(brief):
    client=Chemistry(); b=brief.model_copy(deep=True)
    b.data_policy.allowed_sources.remove('ChEMBL')
    result,_=run(b,client)
    assert result.status=='not_attempted' and not client.calls
    result,_=run(brief,client,seconds=0)
    assert result.status=='error' and not client.calls


def test_identity_only_request_never_refetches_consumed_pubmed_context(brief):
    client=Chemistry(); p=public(client)
    records,rows=p.retrieve_g0_followup(brief,seed(),roles=['assay_identity'])
    assert records==() and not client.calls
    assert rows[0]['attempted'] is False and rows[0]['reason_code']=='VERIFIED_ASSAY_CONTEXT_REQUIRED'


def test_production_wrapper_plans_separate_source_and_preserves_ids(brief):
    bio=Client(['Compound 4 inhibited FLT3 with IC50 of 12 nM.']); chem=Chemistry()
    class Combined:
        def get(self,url,**kwargs):
            return chem.get(url,**kwargs) if url.startswith(BASE) else bio.get(url,**kwargs)
    b=requested(brief); p=packet(b,('FLT3',)); plan=build_plan(b,p,verdict(),rework_loops=0,target_limit=1)
    p.metadata['g0_followup_history']=[{'attempted_task_keys':[t['task_key'] for t in plan['tasks'] if t['role'] in {'direction','association'}]}]
    plan=build_plan(b,p,verdict(),rework_loops=1,target_limit=1)
    wrapper=FollowupRetriever(SimpleNamespace(public=public(Combined())),plan)
    records=wrapper.retrieve(b,seed(),1)
    assert len(records)==1 and records[0].metadata['assay_identity_inventory']['linked_activity_count']==1
    assert [r['source'] for r in wrapper.observations]==['PubMed','ChEMBL','Europe PMC']
    assert wrapper.observations[-1]['attempted'] is False  # Not enabled in this isolated identity test.
    row=wrapper.observations[1]
    assert row['record_count']==0 and row['annotated_context_count']==1
    assert row['evidence_gap']['outcome']=='assay_identity_inventory_completed'
    assert len(row['source_task_keys'])==1
    later=build_plan(b,p.model_copy(update={'metadata':{**p.metadata,'g0_followup_history':[*p.metadata['g0_followup_history'],wrapper.history_entry()]}}),verdict(),rework_loops=1,target_limit=1)
    assert all(t['status']=='already_attempted' for t in later['tasks'] if t['role']!='assay_fulltext')
    assert next(t for t in later['tasks'] if t['role']=='assay_fulltext')['status']=='scheduled'
