"""Deposited assay coverage and source-bound scope comparison, not gate evidence.

Descriptions are quoted mentions, not assertions of a tested construct. A named
mutation field is not a complete genotype. No missing field is inferred as WT,
and disjoint mutation labels are not treated as biological incompatibility.
"""
from collections import Counter
import re

from drugpilot.agents.molecular_scope import mentions, HYPOTHESIS_SCOPE_KEY, valid_hypothesis_scope
from drugpilot.tools.assay_identity import digest, source_context, POLICY as IDENTITY_POLICY

POLICY = "deposited_assay_applicability_v1"
KEY = "assay_applicability_review"
_UNKNOWN = re.compile(r"\b(?:undefined|unknown|unspecified|not reported|not available|not determined)\b", re.I)
_VARIANT = re.compile(r"(?:p\.)?([ACDEFGHIKLMNPQRSTVWY][1-9][0-9]{0,4}[ACDEFGHIKLMNPQRSTVWY*])", re.I)


def _mutation(value, target):
    if value is None or value == "":
        return {"status":"not_recorded", "labels":[]}
    if not isinstance(value, str) or not value.strip() or _UNKNOWN.search(value):
        return {"status":"unresolved", "labels":[]}
    value = value.strip()
    if value.upper() in {"WT", "WILD TYPE", "WILD-TYPE"}:
        return {"status":"named_annotation", "labels":[target+":WT"]}
    if value.upper() in {"ITD", "INTERNAL TANDEM DUPLICATION"}:
        return {"status":"class_annotation", "labels":[target+":ITD"]}
    tokens = re.split(r"\s*[,;/+]\s*", value)
    if not tokens or not all(_VARIANT.fullmatch(t) for t in tokens):
        return {"status":"unresolved", "labels":[]}
    return {"status":"named_annotation", "labels":sorted({target+":"+_VARIANT.fullmatch(t).group(1).upper() for t in tokens})}


def deposited_review(row):
    """Called only after the exact database identity join; original data untouched."""
    activity = row["deposited_activity_context"]
    assay = row["deposited_assay_conditions"]
    component = row["target_component"]
    target = component["canonical_gene"]
    raw_variant = assay.get("variant_sequence")
    variant = raw_variant if isinstance(raw_variant, dict) else {}
    fields = {"activity.assay_variant_mutation":_mutation(activity.get("assay_variant_mutation"), target),
              "assay.variant_sequence.mutation":_mutation(variant.get("mutation"), target)}
    accessions = [v for v in (activity.get("assay_variant_accession"), variant.get("accession")) if v is not None and v != ""]
    conflicts = []
    if any(a != component["accession"] for a in accessions):
        conflicts.append("VARIANT_ACCESSION_DIFFERS_FROM_TARGET_COMPONENT")
    if variant.get("organism") not in (None, "", component["organism"]):
        conflicts.append("VARIANT_ORGANISM_DIFFERS_FROM_TARGET_COMPONENT")
    labels = [set(v["labels"]) for v in fields.values() if v["labels"]]
    if len(labels) == 2 and labels[0] != labels[1]:
        conflicts.append("DEPOSITED_MUTATION_FIELDS_DISAGREE")
    states = [v["status"] for v in fields.values()]
    if conflicts:
        status = "annotation_conflict"
    elif "unresolved" in states or (raw_variant is not None and not isinstance(raw_variant, dict)):
        status = "unresolved_annotation"
    elif "class_annotation" in states:
        status = "class_annotation_only"
    elif "named_annotation" in states:
        status = "named_annotation_only"
    else:
        status = "not_recorded"
    # Keep exact description spans, but never let prose fill a missing typed field.
    description_mentions = []
    for name, text in (("activity.assay_description", activity.get("assay_description")), ("assay.description", assay.get("description"))):
        if isinstance(text, str):
            description_mentions.extend({**m,"field":name} for m in mentions(text, target))
    coverage = {}
    for key in ("assay_type", "assay_organism", "assay_cell_type", "assay_tissue", "assay_strain", "assay_subcellular_fraction"):
        value = assay.get(key)
        coverage[key] = ("not_recorded" if value is None or value == "" else
            "unresolved" if not isinstance(value, str) or _UNKNOWN.search(value) else "recorded_not_validated")
    parameters = assay.get("assay_parameters")
    coverage["assay_parameters"] = ("not_recorded" if parameters is None or parameters == [] else
        "recorded_not_validated" if isinstance(parameters, list) else "unresolved")
    return {"policy":POLICY, "status":status, "mutation_fields":fields,
        "deposited_labels":sorted(set().union(*labels)) if labels else [],
        "description_mentions":description_mentions, "conflict_reasons":conflicts,
        "condition_field_coverage":coverage,
        "sequence_recorded":isinstance(variant.get("sequence"), str) and bool(variant["sequence"].strip()),
        "sequence_equivalence_verified":False, "full_genotype_recorded":False,
        "exact_construct_verified":False, "disease_model_match_verified":False,
        "wild_type_inferred":False, "missing_values_imputed":False,
        "source_basis_sha256":digest({k:row[k] for k in ("activity_id", "assay_chembl_id", "target_component",
            "deposited_activity_context", "deposited_assay_conditions", "source_record_hashes")}),
        "next_action":"resolve_deposited_annotation_conflict" if conflicts else "verify_exact_construct_and_experimental_conditions",
        "authority":"source_annotation_coverage_not_experimental_equivalence_or_gate_verdict"}


def summary(rows):
    reviews = [deposited_review(r) for r in rows]
    return {"policy":POLICY, "reviewed_activity_count":len(rows),
        "exclusive_annotation_counts":dict(sorted(Counter(r["status"] for r in reviews).items())),
        "exact_construct_verified_count":0, "disease_model_match_verified_count":0,
        "primary_evidence_generated":False, "gate_verdict_changed":False}


def scope_comparison(review, label):
    """Compare only an authenticated research-scope label, never request prose."""
    observed = set(review["deposited_labels"])
    if label is None:
        return "no_bound_molecular_scope"
    if review["status"] == "annotation_conflict":
        return "source_annotation_conflict"
    if review["status"] in {"not_recorded", "unresolved_annotation"}:
        return "source_annotation_unresolved"
    if "::" in label:
        return "fusion_construct_not_established_by_component_assay"
    if label in observed:
        if label.endswith(":ITD"):
            return "same_variant_class_not_exact_construct"
        return "same_label_not_complete_genotype" if observed == {label} else "partial_label_overlap"
    # A D835Y field, for example, does not establish presence OR absence of ITD.
    return "different_labels_equivalence_unestablished"


def annotate_hypothesis(brief, hypothesis):
    """Add a diagnostic after hypothesis branching. No claims, scores or gates change."""
    inventories = [(e,e.metadata.get("assay_identity_inventory")) for e in hypothesis.evidence
                   if "assay_identity_inventory" in e.metadata]
    if not inventories:
        return hypothesis
    output = hypothesis.model_copy(deep=True)
    raw_scope = hypothesis.target.metadata.get(HYPOTHESIS_SCOPE_KEY)
    valid_scope = valid_hypothesis_scope(brief, hypothesis) if raw_scope is not None else False
    label = raw_scope["label"] if valid_scope else None
    rows, invalid = [], []
    for e, a in inventories:
        valid = (source_context(e) and isinstance(a, dict) and a.get("policy") == IDENTITY_POLICY
            and a.get("source") == "ChEMBL" and a.get("provenance_class") == "external_live"
            and a.get("fixture") is False and a.get("primary_support_eligible") is False
            and a.get("complete_query_window") is True
            and a.get("parent_evidence_id") == e.evidence_id and a.get("parent_excerpt_sha256") == e.excerpt_hash
            and a.get("parent_abstract_sha256") == e.metadata.get("content_hash") and a.get("pmid") == e.external_id
            and a.get("annotation_sha256") == digest({k:v for k,v in a.items() if k != "annotation_sha256"})
            and isinstance(a.get("linked_activities"), list))
        if not valid:
            invalid.append(e.evidence_id)
            continue
        for row in a["linked_activities"]:
            component = row.get("target_component", {})
            if (row.get("database_identity_chain_verified") is not True or row.get("pmid") != e.external_id
                    or component.get("canonical_gene") != hypothesis.target.symbol
                    or (hypothesis.target.uniprot_id and component.get("accession") != hypothesis.target.uniprot_id)):
                invalid.append(e.evidence_id)
                continue
            review = deposited_review(row)
            rows.append({"parent_evidence_id":e.evidence_id, "inventory_sha256":a["annotation_sha256"],
                "activity_id":row["activity_id"], "assay_chembl_id":row["assay_chembl_id"],
                "molecule_chembl_id":row["molecule_chembl_id"],
                "review":{k:review[k] for k in ("policy", "status", "deposited_labels", "conflict_reasons", "source_basis_sha256")},
                "scope_comparison":scope_comparison(review, label)})
    annotation = {"policy":POLICY, "hypothesis_id":hypothesis.hypothesis_id,
        "target_symbol":hypothesis.target.symbol, "expected_uniprot_id":hypothesis.target.uniprot_id,
        "scope_status":"source_bound_research_scope" if valid_scope else "invalid_scope" if raw_scope is not None else "not_bound",
        "scope_label":label, "scope_sha256":digest(raw_scope) if valid_scope else None,
        "requested_cell_context":brief.constraints.target_cell_type,
        "cell_context_comparison":"requires_model_and_comparator_validation_not_free_text_equality",
        "activities":rows, "reviewed_activity_count":len(rows), "invalid_inventory_evidence_ids":sorted(set(invalid)),
        "scope_comparison_counts":dict(sorted(Counter(r["scope_comparison"] for r in rows).items())),
        "primary_support_eligible":False, "candidate_excluded":False, "scientific_verdict_changed":False,
        "scope":"Diagnostic only. A research-scope label and deposited annotation are not proof of experimental equivalence."}
    annotation["annotation_sha256"] = digest(annotation)
    output.target.metadata[KEY] = annotation
    return output


def screening_metadata(hypothesis):
    """Compact rejection audit: no duplicated sequences, descriptions or row arrays."""
    a = hypothesis.target.metadata.get(KEY)
    if not isinstance(a, dict) or a.get("policy") != POLICY:
        return {}
    return {"assay_applicability_summary":{k:a[k] for k in (
        "policy", "hypothesis_id", "scope_status", "scope_label", "scope_sha256", "reviewed_activity_count",
        "scope_comparison_counts", "invalid_inventory_evidence_ids", "primary_support_eligible",
        "candidate_excluded", "scientific_verdict_changed", "annotation_sha256")}}
