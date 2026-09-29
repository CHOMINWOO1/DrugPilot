"""Synthetic parser cases only; never records supplied to a live run."""
from copy import deepcopy
from hashlib import sha256

import pytest

from drugpilot.tools.assay_followup import assay_spans
from drugpilot.tools.assay_relations import quantities, quantity_shape, text_relation
from drugpilot.agents.evidence_semantics import source_records
from drugpilot.agents.ma1 import _assign_claim_evidence
from test_assay_followup import Client, public, seed


def bind(text,target="FLT3"):
    spans=assay_spans(text,target,("IC50","EC50","Ki","Kd"))
    assert len(spans)==1
    return text_relation(text,spans[0],target,sha256(text.encode()).hexdigest())


@pytest.mark.parametrize("quote,representation,numbers",[
    ("13.12 ± 1.71 μM","value_with_unspecified_uncertainty",["13.12","1.71"]),
    ("12 +/- 2 nM","value_with_unspecified_uncertainty",["12","2"]),
    ("<10 nM","bound_not_exact_value",["10"]),
    ("≤ 10 nM","bound_not_exact_value",["10"]),
    ("approximately 2.5 µM","approximate_value",["2.5"]),
    ("1.1 and 0.82μM","shared_unit_expression_unassigned",["1.1","0.82"]),
    ("0.719/0.770 nM","shared_unit_expression_unassigned",["0.719","0.770"]),
    ("1–10 nM","shared_unit_expression_unassigned",["1","10"]),
    ("1 to 10 nM","shared_unit_expression_unassigned",["1","10"]),
    ("1e-3 mM","reported_scalar",["1e-3"]),
    (".5 nM","reported_scalar",[".5"]),
    ("12 pM","reported_scalar",["12"]),
])
def test_whole_numeric_expression_preserved(quote,representation,numbers):
    text=f"An IC50 of {quote} was reported."
    row=quantities(text,10)
    assert len(row)==1 and row[0]["quote"]==quote
    assert text[row[0]["start"]-10:row[0]["end"]-10]==quote
    shape=quantity_shape(quote)
    assert shape["representation"]==representation and shape["number_quotes"]==numbers
    assert not shape["unit_conversion_applied"] and not shape["uncertainty_type_verified"]


@pytest.mark.parametrize("target,subject,verb,endpoint,value",[
    ("FLT3","Compound 9x","showed significant inhibition against","IC50","7.2 nM"),
    ("ABL1","TEST12","inhibited","IC50","12.3 ± 2.1 μM"),
    ("RET","Compound 7","had in vitro inhibitory activity against","Ki",".5 µM"),
    ("ALK","The most active compound, 8k,","showed strong activity against","Kd","<10 nM"),
    ("EGFR","EX5","potently inhibited","EC50","1e-3 mM"),
])
def test_explicit_quote_link_is_not_scientific_assignment(target,subject,verb,endpoint,value):
    text=f"Background. {subject} {verb} {target} with {endpoint} value of {value}."
    row=bind(text,target)
    assert row["status"]=="explicit_text_relation" and row["textual_relation_verified"]
    assert row["source_binding_verified"]
    for anchor in row["relation"].values():
        assert text[anchor["start"]:anchor["end"]]==anchor["quote"]
    assert row["relation"]["quantity"]["quote"]==value
    assert not any(row[k] for k in ("measurement_assignment_verified","compound_identity_verified",
        "construct_identity_verified","disease_model_match_verified","assay_protocol_verified","primary_support_eligible"))
    assert row["bioactivity_records_generated"]==0
    assert row["unresolved_fields"]


@pytest.mark.parametrize("text",[
    "Compound 4 inhibited FLT3 with IC50 of 2 nM and ALK with IC50 of 7 nM.",
    "Compound 4 inhibited FLT3 with IC50 of 2 nM and 7 nM respectively.",
    "Compound 4 inhibited FLT3 with IC50 of 2/7 nM.",
    "Compound 4 did not inhibit FLT3 with IC50 of 2 nM.",
    "Compound 4 inhibited FLT3 with IC50 of 2 nM, but not in independent tests.",
    "Compound 4 inhibited FLT3 with IC50 of 2 nM, which was predicted by docking.",
    "Compound 4 inhibited FLT3 with IC50 of 2 nM in combination with Compound 5.",
    "Compound 4 inhibited FLT3 with IC50 of 2 nM, lower than Compound 5.",
    "It inhibited FLT3 with IC50 of 2 nM.",
    "FLT3 inhibition had an IC50 of 2 nM.",
    "Compound 4 inhibited FLT3-positive cells with IC50 of 2 nM.",
    "Compound 4 suppressed proliferation of FLT3-WT cells with IC50 of 2 nM.",
    "Compound 4 inhibited BCR-ABL1 with IC50 of 2 nM; FLT3 was also tested.",
    "Compound 4 inhibited FLT3-ITD with IC50 of 2 nM.",
    "Compound 4 inhibited FLT3 with Ki of 2 nM (IC50 was also tested).",
    "Compound 4 inhibited FLT3 with IC50 of -2 nM.",
    "Compound 4 inhibited FLT3 with IC50 of −2 nM.",
])
def test_ambiguous_or_qualified_context_never_assigns(text):
    row=bind(text)
    assert row["status"]=="unresolved" and row["reason_codes"] and row["relation"] is None
    assert not row["textual_relation_verified"] and not row["measurement_assignment_verified"]


@pytest.mark.parametrize("field,value",[
    ("start",False),("end",-1),("end",10000),("excerpt","tampered"),
    ("excerpt_sha256","0"*64),("quantity_mentions",[]),
])
def test_tampered_source_or_anchor_rejected(field,value):
    text="Compound 4 inhibited FLT3 with IC50 of 2 nM."
    span=deepcopy(assay_spans(text,"FLT3",("IC50",))[0])
    span[field]=value
    row=text_relation(text,span,"FLT3",sha256(text.encode()).hexdigest())
    assert not row["source_binding_verified"] and not row["textual_relation_verified"]


def test_wrong_parent_hash_cannot_be_bound():
    text="Compound 4 inhibited FLT3 with IC50 of 2 nM."
    row=text_relation(text,assay_spans(text,"FLT3",("IC50",))[0],"FLT3","0"*64)
    assert row["reason_codes"]==["ASSAY_SOURCE_BINDING_INVALID"]


def test_selected_clause_cannot_hide_negation_with_self_consistent_hashes():
    text="Not supported: Compound 4 inhibited FLT3 with IC50 of 2 nM."
    start=text.index("Compound")
    excerpt=text[start:]
    span={"start":start,"end":len(text),"excerpt":excerpt,
        "excerpt_sha256":sha256(excerpt.encode()).hexdigest(),"quantity_mentions":quantities(excerpt,start)}
    row=text_relation(text,span,"FLT3",sha256(text.encode()).hexdigest())
    assert row["reason_codes"]==["ASSAY_COMPLETE_SENTENCE_REQUIRED"]


def test_expanded_target_label_does_not_infer_construct_identity():
    text="Compound 3 had in vitro inhibitory activity against Abelson kinase 1 (ABL1) (IC50=8.2 µM), the kinase component of BCR-ABL."
    row=bind(text,"ABL1")
    assert row["textual_relation_verified"]
    assert row["relation"]["target_label"]["quote"]=="Abelson kinase 1 (ABL1)"
    assert not row["construct_identity_verified"]


def test_decimal_and_variant_sentence_offsets_do_not_merge_neighboring_claims():
    text="Compound 4 inhibited FLT3 with IC50 of .5 nM. FLT3 p.D835Y was assessed."
    spans=assay_spans(text,"FLT3",("IC50",))
    assert len(spans)==1 and spans[0]["excerpt"].endswith(".5 nM.")


def test_coded_subject_is_not_assumed_to_be_a_compound():
    row=bind("BCL2 inhibited FLT3 with IC50 of 2 nM.")
    assert row["textual_relation_verified"] and row["subject_kind"]=="untyped_source_subject"
    assert row["relation"]["subject_label"]["quote"]=="BCL2"
    assert not row["compound_identity_verified"]


def test_hidden_other_endpoint_is_recomputed_even_if_not_requested():
    text="Compound 4 inhibited FLT3 with IC50 of 2 nM (Kd was also measured)."
    span=assay_spans(text,"FLT3",("IC50",))[0]
    row=text_relation(text,span,"FLT3",sha256(text.encode()).hexdigest())
    assert "MULTIPLE_ASSAY_ENDPOINTS_UNRESOLVED" in row["reason_codes"]


def test_production_retrieval_retains_mapping_without_promoting_evidence(brief):
    client=Client(["Compound 4 inhibited FLT3 with IC50 of 12 ± 2 nM."])
    records,observations=public(client).retrieve_g0_followup(brief,seed(),roles=["assay_context"])
    row=records[0].metadata["assay_text_relation"]
    assert row["textual_relation_verified"] and row["relation"]["quantity"]["number_quotes"]==["12","2"]
    assert records[0].supports=="context" and not source_records(records)
    claims=_assign_claim_evidence(brief,target_symbol="FLT3",evidence=records)
    assert not claims.direction_primary_ids and not claims.association_primary_ids
    assert len(client.calls)==3 and observations[0]["trace"]["bioactivity_records_generated"]==0
