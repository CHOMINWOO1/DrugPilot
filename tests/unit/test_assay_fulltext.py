"""Synthetic source transport contracts, never live evidence fixtures."""
from copy import deepcopy
from hashlib import sha256

import httpx
import pytest

from drugpilot.tools.assay_fulltext import parse_methods, retrieve_assay_fulltext, KEY
from drugpilot.tools.assay_identity import digest
from drugpilot.tools.europe_pmc_oa_fulltext import OAArticleEligibility, EuropePMCOAValidationError
from drugpilot.tools.public_biology import EUROPE_PMC_SEARCH_URL, _EPMC_OA_ALLOWED_LICENSE_URIS
from drugpilot.agents.ma1 import _assign_claim_evidence
from drugpilot.agents.evidence_semantics import source_records
from drugpilot.orchestration.g0_followup import FollowupRetriever, build_plan
from test_assay_followup import Client,public,seed,requested,packet,verdict
from test_europe_pmc_oa_fulltext_integration import _jats,_core_item,LICENSE_URI


def xml(body=None, **kwargs):
    raw=_jats(pmcid='PMC1000',pmid='1000',sentence='Results mention FLT3 but are not methods.',**kwargs)
    if body is None:
        body='<sec sec-type="methods"><title>Materials and methods</title><sec><title>Kinase assay</title><p>Recombinant FLT3 was incubated for 60 min with ATP at 10 µM.</p></sec></sec>'
    return raw.replace(b'</body>',body.encode()+b'</body>')


def eligibility():
    return OAArticleEligibility('PMC1000','1000',('Journal Article',),True,'CC BY')


def parse(raw):
    return parse_methods(raw,eligibility=eligibility(),allowed_license_uris=_EPMC_OA_ALLOWED_LICENSE_URIS,target='FLT3')


def test_methods_location_and_hash_without_result_promotion():
    raw=xml(); result=parse(raw)
    assert result['document_sha256']==sha256(raw).hexdigest()
    assert result['pmid']=='1000' and result['license_uri']==LICENSE_URI
    assert len(result['units'])==1
    unit=result['units'][0]
    assert unit['source_location']=='body/sec[2]/sec[1]/p[1]'
    assert unit['excerpt_sha256']==sha256(unit['text'].encode()).hexdigest()
    assert '60 min' in unit['text'] and not unit['measurement_assignment_verified']
    assert not result['primary_support_eligible'] and not result['raw_fulltext_retained']


@pytest.mark.parametrize('kind',['pmid','pmcid','doi','license','entity','internal_dtd','depth'])
def test_existing_jats_security_and_identity_guards_are_not_bypassed(kind):
    raw=xml()
    if kind=='pmid':raw=raw.replace(b"type='pmid'>1000",b"type='pmid'>2000")
    if kind=='pmcid':raw=raw.replace(b'PMC1000',b'PMC2000')
    if kind=='doi':
        # Conflicting DOI is checked when the caller independently supplied it.
        from dataclasses import replace
        with pytest.raises(EuropePMCOAValidationError):
            parse_methods(xml(doi='10.1234/wrong'),eligibility=replace(eligibility(),doi='10.1234/right'),allowed_license_uris=_EPMC_OA_ALLOWED_LICENSE_URIS,target='FLT3')
        return
    if kind=='license':raw=raw.replace(LICENSE_URI.encode(),b'https://untrusted.invalid/license')
    if kind=='entity':raw=b'<!DOCTYPE article [<!ENTITY x SYSTEM "file:///private">]>'+raw
    if kind=='internal_dtd':raw=b'<!DOCTYPE article []>'+raw
    if kind=='depth':raw=raw.replace(b'</body>',b'<sec>'*130+b'</sec>'*130+b'</body>')
    with pytest.raises(EuropePMCOAValidationError):parse(raw)


def test_excluded_sections_tables_and_footnotes_are_not_methods():
    body='<sec><title>Discussion</title><sec><title>Methods</title><p>FLT3 assay must not be selected.</p></sec></sec>'
    body+='<sec><title>Methods</title><fig><caption><p>FLT3 assay figure.</p></caption></fig><table-wrap id="t1"><caption><p>IC50 assay values</p></caption><table><tr><th>Compound</th><th>IC50</th></tr><tr><td rowspan="2">X</td><td>12</td></tr></table></table-wrap></sec>'
    result=parse(xml(body))
    assert result['units']==[] and len(result['table_locators'])==1
    assert result['table_locators'][0]['xml_id']=='t1'
    assert not result['table_locators'][0]['measurement_rows_parsed']


def test_shared_cell_culture_protocol_is_not_forced_to_one_target():
    result=parse(xml('<sec><title>Methods</title><sec><title>Cell culture</title><p>Cells were maintained in RPMI at 37 degrees.</p></sec></sec>'))
    assert len(result['units'])==1 and not result['units'][0]['target_mentioned']
    assert not result['units'][0]['experimental_equivalence_verified']


def test_bounds_are_explicit_and_late_method_paragraph_is_not_hidden():
    body='<sec><title>Methods</title>'+''.join('<p>Unrelated chemistry paragraph.</p>' for _ in range(20))
    body+=''.join('<p>FLT3 assay '+('x'*1500)+'</p>' for _ in range(15))+'</sec>'
    result=parse(xml(body))
    assert result['matching_method_paragraphs']==15 and result['retained_method_paragraphs']==12
    assert result['omitted_method_paragraphs']==3
    assert all(u['truncated'] and len(u['text'])==1200 for u in result['units'])


class FulltextClient:
    def __init__(self,*,case='ok'):
        self.calls=[];self.case=case
    def get(self,url,**kwargs):
        self.calls.append((url,kwargs))
        if self.case=='timeout':raise httpx.ReadTimeout('private must not appear')
        if self.case=='503':return httpx.Response(503,request=httpx.Request('GET',url))
        if url==EUROPE_PMC_SEARCH_URL:
            assert kwargs['params']['query']=='EXT_ID:1000 AND SRC:MED'
            item=_core_item(pmcid='PMC1000',pmid='1000');item.update(id='1000',source='MED')
            if self.case=='closed':item.update(pmcid=None,isOpenAccess='N',license=None)
            if self.case=='license':item['license']='unknown'
            if self.case=='review':item['pubTypeList']={'pubType':['Review']}
            if self.case=='retracted':item['isRetracted']='Y'
            if self.case=='wrong_id':item['pmid']='9999'
            if self.case=='wrong_source':item['source']='PMC'
            items=[] if self.case=='empty' else [item]
            count=True if self.case=='invalid_count' else len(items)
            return httpx.Response(200,json={'hitCount':count,'resultList':{'result':items}},request=httpx.Request('GET',url))
        assert url.endswith('/PMC1000/fullTextXML') and kwargs['follow_redirects'] is False
        raw=xml()
        if self.case=='wrong_xml':raw=raw.replace(b"type='pmid'>1000",b"type='pmid'>9999")
        return httpx.Response(200,content=raw,request=httpx.Request('GET',url))


def perform(brief,client,**kwargs):
    p=public(Client()); records,_=p.retrieve_g0_followup(brief,seed(),roles=['assay_context'])
    before=[e.model_dump_json() for e in records]
    p.epmc_oa_fulltext_one_hop_v1=True;p.client=client
    result=retrieve_assay_fulltext(p,brief,seed(),records,**kwargs)
    assert [e.model_dump_json() for e in records]==before
    return result,records


def test_live_boundary_preserves_original_record_and_context_authority(brief):
    client=FulltextClient();result,old=perform(brief,client)
    assert result.status=='completed' and len(client.calls)==2
    assert result.trace['fulltexts_parsed']==1
    e=result.records[0];a=e.metadata[KEY]
    assert a['annotation_sha256']==digest({k:v for k,v in a.items() if k!='annotation_sha256'})
    assert e.excerpt_hash==old[0].excerpt_hash and e.evidence_id==old[0].evidence_id
    assert not a['primary_support_eligible'] and not a['chembl_conditions_overwritten']
    assert not source_records(result.records)
    claims=_assign_claim_evidence(brief,target_symbol='FLT3',evidence=result.records)
    assert not claims.direction_primary_ids and not claims.association_primary_ids


@pytest.mark.parametrize('case,status,calls',[
    ('closed','oa_not_admitted',1),('license','oa_not_admitted',1),('review','oa_not_admitted',1),
    ('retracted','oa_not_admitted',1),('empty','metadata_not_found',1),
    ('wrong_id','retrieval_or_validation_error',1),('wrong_source','retrieval_or_validation_error',1),
    ('invalid_count','retrieval_or_validation_error',1),('wrong_xml','retrieval_or_validation_error',2),
    ('timeout','retrieval_or_validation_error',1),('503','retrieval_or_validation_error',1),
])
def test_closed_missing_invalid_and_failed_paths_never_return_fulltext(brief,case,status,calls):
    client=FulltextClient(case=case);result,_=perform(brief,client)
    a=result.records[0].metadata[KEY]
    assert a['document_status']['status']==status and a['content'] is None
    assert len(client.calls)==calls and not a['measurement_assignment_verified']
    assert 'private' not in str(result.trace)


@pytest.mark.parametrize('limit',['http','bytes','time','documents'])
def test_each_budget_remains_bounded_and_observable(brief,monkeypatch,limit):
    client=FulltextClient()
    if limit=='http':monkeypatch.setattr('drugpilot.tools.public_biology._EPMC_OA_HTTP_CALL_LIMIT',1)
    if limit=='bytes':monkeypatch.setattr('drugpilot.tools.public_biology._EPMC_OA_AGGREGATE_BYTE_LIMIT',1)
    if limit=='documents':monkeypatch.setattr('drugpilot.tools.public_biology._EPMC_OA_PARENT_LIMIT',0)
    result,_=perform(brief,client,elapsed_seconds=0 if limit=='time' else 120)
    assert result.records[0].metadata[KEY]['document_status']['status']=='budget_not_completed'
    assert len(client.calls)<=1 and not result.trace['fulltexts_parsed']


def test_disabled_unallowed_or_invalid_parent_never_calls_source(brief):
    p=public(Client());records,_=p.retrieve_g0_followup(brief,seed(),roles=['assay_context'])
    p.client=FulltextClient()
    assert retrieve_assay_fulltext(p,brief,seed(),records).reason=='OA_FULLTEXT_DISABLED'
    p.epmc_oa_fulltext_one_hop_v1=True
    for field in ['fixture','mock','excerpt']:
        e=records[0].model_copy(deep=True);e.metadata[field]=True
        assert retrieve_assay_fulltext(p,brief,seed(),[e]).reason=='VERIFIED_ASSAY_CONTEXT_REQUIRED'
    b=brief.model_copy(deep=True);b.data_policy.allowed_sources.remove('Europe PMC')
    assert retrieve_assay_fulltext(p,b,seed(),records).reason=='SOURCE_NOT_ALLOWED'
    assert not p.client.calls


def test_wrapper_after_context_and_generic_oa_history_does_not_consume_methods(brief):
    from types import SimpleNamespace
    bio=Client();oa_client=FulltextClient()
    class Combined:
        def get(self,url,**kwargs):
            return oa_client.get(url,**kwargs) if 'europepmc' in url else bio.get(url,**kwargs)
    b=requested(brief);p=packet(b,('FLT3',));plan=build_plan(b,p,verdict(),rework_loops=0,target_limit=1)
    p.metadata['g0_followup_history']=[{'attempted_task_keys':[t['task_key'] for t in plan['tasks'] if t['role'] in {'association','direction','assay_identity'}]}]
    plan=build_plan(b,p,verdict(),rework_loops=1,target_limit=1)
    retriever=public(Combined());retriever.enabled_sources=('PubMed','Europe PMC');retriever.epmc_oa_fulltext_one_hop_v1=True
    wrapper=FollowupRetriever(SimpleNamespace(public=retriever),plan)
    records=wrapper.retrieve(b,seed(),1)
    assert len(records)==1 and records[0].metadata[KEY]['content']
    assert [r['role'] for r in wrapper.observations]==['assay_context','assay_fulltext']
    assert wrapper.observations[1]['record_count']==0 and wrapper.observations[1]['annotated_context_count']==1
    assert wrapper.observations[1]['evidence_gap']['assay_fulltext_counts']['parsed']==1
    assert len(wrapper.observations[1]['source_task_keys'])==1
    p.metadata['g0_followup_history'].append(wrapper.history_entry())
    later=build_plan(b,p,verdict(),rework_loops=1,target_limit=1)
    assert all(t['status']=='already_attempted' for t in later['tasks'])


def test_methods_only_never_replays_a_consumed_pubmed_search(brief):
    p=public(FulltextClient());p.enabled_sources=('Europe PMC',);p.epmc_oa_fulltext_one_hop_v1=True
    records,rows=p.retrieve_g0_followup(brief,seed(),roles=['assay_fulltext'],source_paths=[{'source':'Europe PMC','role':'assay_fulltext'}])
    assert records==() and not p.client.calls
    assert rows[0]['reason_code']=='VERIFIED_ASSAY_CONTEXT_REQUIRED' and rows[0]['attempted'] is False


def test_generic_oa_observation_does_not_mark_exact_parent_methods_as_attempted(brief):
    from types import SimpleNamespace
    b=requested(brief);plan=build_plan(b,packet(b,('FLT3',)),verdict(),rework_loops=0,target_limit=1)
    def retrieve(*args,**kwargs):
        return (),[{'source':'Europe PMC','role':'oa_context','status':'empty','attempted':True,'record_count':0}]
    wrapper=FollowupRetriever(SimpleNamespace(public=SimpleNamespace(retrieve_g0_followup=retrieve)),plan)
    wrapper.retrieve(b,seed(),1)
    fulltext=next(t for t in plan['tasks'] if t['role']=='assay_fulltext')
    assert fulltext['task_key'] not in wrapper.observations[0]['task_keys']
    assert fulltext['source_paths'][0]['source_task_key'] not in wrapper.history_entry()['attempted_source_task_keys']
