"""Synthetic isolated tests. No examples are injected into live research runs."""
from hashlib import sha256

import pytest

from drugpilot.agents.document_result_units import (
    KEY, POLICY, candidate_units, derive_result_units, parent_is_bound, validated_unit_excerpt,
)
from drugpilot.agents.evidence_semantics import (
    EvidenceMeaning, EvidenceMeaningBatch, attach_interpretations, record_key,
    source_records, source_binding,
)
from drugpilot.agents.ma1 import _primary_direction_excerpt, _assign_claim_evidence
from test_evidence_semantics import evidence
from test_abstract_interpretation_context import with_context


def source(brief,target="FLT3",entity="FLT3",result=None):
    first = f"We assessed {entity} inhibition in {brief.disease} cells."
    second = result or f"{entity} inhibition significantly reduced viability in these cells."
    original = with_context(evidence(first),first+" "+second)
    original.metadata["source_location"] = f"abstract:METHODS:chars:0-{len(first)}"
    return original


def interpreted(brief,es,target="FLT3",entity="FLT3",label=""):
    meanings = []
    for e in es:
        meanings.append(EvidenceMeaning(record_key=record_key(e),study_context="measured_primary",
            target_relation="direct",disease_context="actual_model",intervention="direct_requested",
            outcome="favorable",target_quote=entity,model_quote=f"{brief.disease} cells",
            intervention_quote=f"{entity} inhibition",result_quote="reduced viability",molecular_scope_label=label))
    batch = EvidenceMeaningBatch(records=meanings)
    return attach_interpretations(brief,target,es,source_records(es),batch,batch)


def test_exact_adjacent_unit_preserves_original_and_needs_two_semantic_passes(brief):
    e = source(brief)
    before = e.model_dump_json()
    result = derive_result_units(brief,"FLT3",[e])
    assert len(result)==2 and result[0].model_dump_json()==before
    unit = result[1]
    assert unit.evidence_id != e.evidence_id
    assert unit.metadata["excerpt"] == e.metadata["abstract_interpretation_context"]["text"]
    assert unit.excerpt_hash == sha256(unit.metadata["excerpt"].encode()).hexdigest()
    assert unit.metadata[KEY]["unit"]["anchor"]["start"] == 0
    assert parent_is_bound(unit,result)
    assert _primary_direction_excerpt(brief,"FLT3",unit) is None
    annotated = interpreted(brief,result)
    assert _primary_direction_excerpt(brief,"FLT3",annotated[0]) is None
    assert _primary_direction_excerpt(brief,"FLT3",annotated[1]) == unit.metadata["excerpt"]
    assignment = _assign_claim_evidence(brief,target_symbol="FLT3",evidence=annotated)
    assert assignment.direction_primary_ids == (unit.evidence_id,)
    assert not assignment.association_primary_ids
    assert not _assign_claim_evidence(brief,target_symbol="FLT3",evidence=[annotated[1]]).direction_primary_ids


@pytest.mark.parametrize("intake", ["modulate", "unspecified"])
def test_broad_intake_retains_verified_inhibition_unit_without_relabeling_it_activation(brief, intake):
    b = brief.model_copy(deep=True)
    b.constraints.intervention_direction = intake
    e = source(b)
    e.metadata["query_intervention_direction"] = intake
    before = e.model_dump_json()
    derived = derive_result_units(b, "FLT3", [e])
    assert len(derived) == 2
    assert derived[0].model_dump_json() == before
    annotated = interpreted(b, derived)
    assert _primary_direction_excerpt(b, "FLT3", annotated[1], intervention_direction="inhibit")
    assert _primary_direction_excerpt(b, "FLT3", annotated[1], intervention_direction="activate") is None
    assert _primary_direction_excerpt(b, "FLT3", annotated[1], intervention_direction="modulate") is None
    altered = annotated[1].model_copy(deep=True)
    altered.metadata[KEY]["unit"]["anchor"]["start"] += 1
    assert _primary_direction_excerpt(b, "FLT3", altered, intervention_direction="inhibit") is None


@pytest.mark.parametrize("disease,target,entity,label",[
    ("Chronic myeloid leukemia","ABL1","BCR::ABL1","BCR::ABL1"),
    ("Non-small cell lung cancer","ALK","EML4::ALK","EML4::ALK"),
    ("Acute myeloid leukemia","FLT3","FLT3",""),
])
def test_generic_diseases_and_fusions_remain_scoped(brief,disease,target,entity,label):
    b = brief.model_copy(update={"disease":disease,"research_goal":"Investigate source-supported targets"})
    es = derive_result_units(b,target,[source(b,target,entity)])
    assert len(es)==2
    result = interpreted(b,es,target,entity,label)
    assert _primary_direction_excerpt(b,target,result[1],label or None)
    if label:
        assert _primary_direction_excerpt(b,target,result[1]) is None


@pytest.mark.parametrize("second",[
    "FLT3 inhibition reduced viability in lung carcinoma cells.",
    "BCL2 inhibition significantly reduced viability in these cells.",
    "FLT3 inhibition did not reduce viability in these cells.",
    "FLT3 inhibition may reduce viability in these cells.",
    "FLT3 inhibition increased viability in these cells.",
    "FLT3 inhibition significantly reduced ERK phosphorylation in these cells.",
    "FLT3 inhibition significantly reduced viability in these organoids.",
    "FLT3 inhibition significantly reduced viability in these cells and other lung cells.",
    "FLT3 inhibition combined with BCL2 inhibition significantly reduced viability in these cells.",
    "FLT3 inhibition significantly reduced viability in these cells in a different experiment.",
    "We found resistance. FLT3 inhibition significantly reduced viability in these cells.",
    "This treatment significantly reduced viability in these cells.",
])
def test_different_model_entity_polarity_combination_or_distant_reference_is_rejected(brief,second):
    e=source(brief,result=second)
    assert derive_result_units(brief,"FLT3",[e]) == [e]


def test_background_future_and_nonexperimental_models_do_not_form_units(brief):
    positive = source(brief).metadata["abstract_interpretation_context"]["text"]
    for prefix in ("It is known that", "We will assess", "We predicted", "We simulated"):
        assert not candidate_units(brief,"FLT3",positive.replace("We assessed",prefix))


@pytest.mark.parametrize("mutation",[
    lambda u:u.metadata[KEY]["unit"]["anchor"].update(start=5),
    lambda u:u.metadata[KEY]["unit"].update(target_symbol="BCL2"),
    lambda u:u.metadata[KEY]["unit"].update(scope_label="ETV6::FLT3"),
    lambda u:u.metadata[KEY].update(abstract_sha256="0"*64),
    lambda u:u.metadata[KEY].update(candidate_unit_count=5),
    lambda u:u.metadata.update(source_location="abstract-unit:chars:0-99"),
    lambda u:u.metadata.update(excerpt=u.metadata["excerpt"]+" invented"),
    lambda u:u.metadata.update(publication_types=["Review"]),
    lambda u:u.metadata.update(fixture=True),
])
def test_tampered_unit_never_admitted(brief,mutation):
    es=interpreted(brief,derive_result_units(brief,"FLT3",[source(brief)]))
    assert _primary_direction_excerpt(brief,"FLT3",es[1])
    mutation(es[1])
    assert _primary_direction_excerpt(brief,"FLT3",es[1]) is None


def test_wrong_parent_and_duplicate_parent_fail_closed(brief):
    es=interpreted(brief,derive_result_units(brief,"FLT3",[source(brief)]))
    assert parent_is_bound(es[1],es)
    assert not parent_is_bound(es[1],[es[0],es[0],es[1]])
    es[0].metadata["excerpt"] += " changed"
    assert not parent_is_bound(es[1],es)
    assert not _assign_claim_evidence(brief,target_symbol="FLT3",evidence=es).direction_primary_ids


def test_meaning_of_another_result_cannot_authorize_this_unit(brief):
    e=source(brief)
    full=e.metadata["abstract_interpretation_context"]["text"]+" Other cells showed increased apoptosis."
    es=interpreted(brief,derive_result_units(brief,"FLT3",[with_context(e,full)]))
    for actor in ("extractor","verifier"):
        es[1].metadata["semantic_interpretation"][actor]["result_quote"]="increased apoptosis"
    assert _primary_direction_excerpt(brief,"FLT3",es[1]) is None


def test_derivative_count_capped_and_all_originals_preserved(brief):
    e=source(brief)
    full=e.metadata["abstract_interpretation_context"]["text"]
    e=with_context(e,full+" "+full)
    es=derive_result_units(brief,"FLT3",[e])
    assert len(es)==2 and es[0]==e
    assert es[1].metadata[KEY]["candidate_unit_count"]==2
    assert es[1].metadata[KEY]["omitted_unit_count"]==1


def test_review_fixture_oversized_or_invalid_source_cannot_create_unit(brief):
    e=source(brief)
    for field,value in (("publication_types",["Review"]),("fixture",True),("content_hash","0"*64)):
        bad=e.model_copy(deep=True)
        bad.metadata[field]=value
        assert derive_result_units(brief,"FLT3",[bad]) == [bad]


def test_new_module_is_bound_by_deployment_source_digest(monkeypatch):
    from pathlib import Path
    from drugpilot.orchestration.human_review import source_digest
    before=source_digest()
    original=Path.read_bytes
    monkeypatch.setattr(Path,"read_bytes",lambda p:original(p)+b"\n# test" if p.name=="document_result_units.py" else original(p))
    assert source_digest()!=before


def test_repeated_derivation_is_idempotent_and_does_not_duplicate_evidence(brief):
    es=derive_result_units(brief,"FLT3",[source(brief)])
    assert len(es)==2
    assert derive_result_units(brief,"FLT3",es)==es


def test_question_is_not_an_executed_experiment(brief):
    e=source(brief)
    full=e.metadata["abstract_interpretation_context"]["text"].replace("cells.","cells?",1)
    assert not candidate_units(brief,"FLT3",full)


def test_real_adapter_boundary_keeps_pending_source_instead_of_discarding_before_new_unit_check(brief):
    from drugpilot.tools.public_biology import PublicBiologyRetriever
    from drugpilot.agents.ma1 import TargetSeed
    from drugpilot.domain.models import HypothesisTrack
    from test_public_biology import _DirectionPrecisionLaneClient
    text=source(brief).metadata["abstract_interpretation_context"]["text"]
    client=_DirectionPrecisionLaneClient(direction_sentence=text)
    public=PublicBiologyRetriever(client=client,enabled_sources=("PubMed",))
    seed=TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1)
    result=public._pubmed_primary_direction_supplement(brief,seed,
        query=public._pubmed_primary_direction_query(brief,seed),common_params={},excluded_ids=set(),direction_model_context=None)
    assert result.status=="completed_with_context_pending_document_unit_validation"
    assert len(result.records)==1
    parent=result.records[0]
    assert parent.supports=="context" and parent.metadata["verified_direct_primary"] is False
    assert parent.metadata["direct_support_rejection_codes"]
    assert result.trace["records_screened"]==len(client.direction_ids)
    es=derive_result_units(brief,"FLT3",result.records)
    assert len(es)==2 and parent_is_bound(es[1],es)
    assert not _primary_direction_excerpt(brief,"FLT3",es[1])


def test_pending_document_context_does_not_preempt_later_verified_primary(brief):
    from drugpilot.tools.public_biology import PublicBiologyRetriever, PUBMED_FETCH_URL
    from drugpilot.agents.ma1 import TargetSeed
    from drugpilot.domain.models import HypothesisTrack
    from test_public_biology import _DirectionPrecisionLaneClient, FakeResponse
    from xml.etree import ElementTree as ET
    class Client(_DirectionPrecisionLaneClient):
        def get(self,url,**kwargs):
            response=super().get(url,**kwargs)
            if url==PUBMED_FETCH_URL:
                root=ET.fromstring(response.text)
                for article in root.findall('PubmedArticle'):
                    if article.findtext('.//PMID')==self.direction_ids[-1]:
                        article.find('.//AbstractText').text='FLT3 inhibition significantly reduced viability in acute myeloid leukemia cells.'
                return FakeResponse(text=ET.tostring(root,encoding='unicode'))
            return response
    client=Client(direction_sentence=source(brief).metadata['abstract_interpretation_context']['text'])
    public=PublicBiologyRetriever(client=client,enabled_sources=('PubMed',))
    seed=TargetSeed('FLT3','FLT3',HypothesisTrack.LITERATURE,1)
    outcome=public._pubmed_primary_direction_supplement(brief,seed,
        query=public._pubmed_primary_direction_query(brief,seed),common_params={},excluded_ids=set(),direction_model_context=None)
    assert outcome.status=='completed_with_results'
    assert outcome.records[0].external_id==client.direction_ids[-1]
    assert outcome.records[0].metadata['verified_direct_primary'] is True


def test_proposer_wires_source_units_before_interpretation_without_retrieval_side_effects(brief):
    from drugpilot.agents.ma1 import Proposer, TargetSeed
    from drugpilot.domain.models import HypothesisTrack
    from types import SimpleNamespace
    original=source(brief)
    before=original.model_dump_json()
    proposer=Proposer(retriever=SimpleNamespace(retrieve=lambda *args:[original]))
    seen=[]
    def inspect(b,seed,iteration,evidence):
        seen.extend(evidence)
        return interpreted(b,evidence)
    proposer.evidence_interpreter=inspect
    proposer._build_hypothesis(brief,TargetSeed("FLT3","FLT3",HypothesisTrack.LITERATURE,1),1)
    assert len(seen)==2 and KEY in seen[1].metadata
    assert original.model_dump_json()==before
