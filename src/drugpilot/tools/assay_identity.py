"""Read-only PMID -> ChEMBL document/activity/record/assay/molecule joins.

Database identities and deposited conditions are not a new primary-support
claim. No names, nearest potency values, parent compounds or missing assay
conditions are substituted. The original PubMed evidence stays context-only.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import re

POLICY = "pmid_bound_chembl_assay_inventory_v1"
BASE = "https://www.ebi.ac.uk/chembl/api/data/"
PAGE_SIZE = 1000
BATCH_SIZE = 100


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def chembl_id(value):
    return isinstance(value,str) and re.fullmatch(r"CHEMBL[0-9]+",value) is not None


def exact_int(value):
    return type(value) is int and value>=0


def source_context(evidence):
    meta=evidence.metadata
    text=meta.get("excerpt")
    return (evidence.source=="PubMed" and re.fullmatch(r"[0-9]+",str(evidence.external_id or ""))
        and str(meta.get("pmid"))==evidence.external_id and evidence.supports=="context"
        and meta.get("role_specific_record")=="assay_context_followup"
        and meta.get("provenance_class")=="external_live" and not meta.get("fixture") and not meta.get("mock")
        and meta.get("verified_direct_primary") is False and meta.get("direct_support_eligible") is False
        and isinstance(text,str) and 0<len(text)<=1200 and sha256(text.encode()).hexdigest()==evidence.excerpt_hash
        and re.fullmatch(r"[0-9a-f]{64}",str(meta.get("content_hash") or ""))
        and meta.get("content_hash_scope")=="normalized_full_abstract")


class IdentityBudgetExhausted(ValueError):
    """A local admission deadline is not an external-source outage."""


@dataclass(frozen=True)
class IdentityOutcome:
    records: tuple
    status: str
    reason: str
    trace: dict


class _Reader:
    def __init__(self, public, seconds, trace):
        self.public,self.deadline,self.trace=public,public._clock()+seconds,trace

    def all(self, resource, key, id_field, params, *, ordered=False):
        rows,seen=[],set()
        total,offset,previous=None,0,None
        while total is None or offset<total:
            remaining=self.deadline-self.public._clock()
            if remaining<=0:
                raise IdentityBudgetExhausted("ASSAY_IDENTITY_ADMISSION_BUDGET")
            endpoint=BASE+resource+".json"
            query={**params,"limit":PAGE_SIZE,"offset":offset}
            if ordered: query["order_by"]=id_field
            call={"endpoint":endpoint,"query_parameters":query,"source":"ChEMBL","fixture":False}
            self.trace["http_call_provenance"].append(call)
            try:
                response=self.public.client.get(endpoint,params=query,
                    timeout=min(self.public.timeout_seconds,remaining),follow_redirects=True)
                call.update(status_code=response.status_code,response_sha256=sha256(response.content).hexdigest())
                response.raise_for_status()
                payload=response.json()
                call["status"]="ok"
            except Exception as exc:
                call.update(status="error",error_type=type(exc).__name__)
                raise
            meta=payload.get("page_meta") if isinstance(payload,dict) else None
            page=payload.get(key) if isinstance(payload,dict) else None
            if (not isinstance(meta,dict) or not exact_int(meta.get("total_count"))
                    or type(meta.get("offset")) is not int or meta["offset"]!=offset
                    or not isinstance(page,list) or len(page)>PAGE_SIZE):
                raise ValueError("CHEMBL_PAGE_SCHEMA_INVALID")
            if total is None: total=meta["total_count"]
            if total!=meta["total_count"] or offset+len(page)>total or (not page and offset<total):
                raise ValueError("CHEMBL_PAGE_TOTAL_INCONSISTENT")
            for item in page:
                if not isinstance(item,dict): raise ValueError("CHEMBL_ROW_SCHEMA_INVALID")
                identity=item.get(id_field)
                numeric=id_field in {"record_id","activity_id"}
                if not ((exact_int(identity) and identity>0) if numeric else chembl_id(identity)) or identity in seen:
                    raise ValueError("CHEMBL_ROW_IDENTITY_INVALID")
                if ordered and previous is not None and identity<=previous:
                    raise ValueError("CHEMBL_PAGE_ORDER_INVALID")
                seen.add(identity)
                previous=identity
            rows.extend(page)
            offset+=len(page)
            call.update(record_count=len(page),reported_total_count=total,offset=meta["offset"])
            # Never follow a response-provided URL. Every page uses BASE.
            if (offset==total)!=(meta.get("next") is None):
                raise ValueError("CHEMBL_PAGE_COMPLETENESS_INVALID")
        return rows

    def by_ids(self, resource, key, id_field, ids):
        found={}
        for start in range(0,len(ids),BATCH_SIZE):
            batch=ids[start:start+BATCH_SIZE]
            rows=self.all(resource,key,id_field,{id_field+"__in":",".join(batch)})
            if {r[id_field] for r in rows}!=set(batch):
                raise ValueError("CHEMBL_EXACT_ID_SET_MISMATCH")
            found.update({r[id_field]:r for r in rows})
        return found


def _target_component(row, seed):
    if row.get("target_type")!="SINGLE PROTEIN" or row.get("organism")!="Homo sapiens": return None
    components=row.get("target_components")
    if not isinstance(components,list) or len(components)!=1 or not isinstance(components[0],dict): return None
    part=components[0]
    names=part.get("target_component_synonyms")
    if (part.get("component_type")!="PROTEIN" or not isinstance(names,list)
            or not isinstance(part.get("accession"),str) or not part["accession"]): return None
    genes={x.get("component_synonym") for x in names if isinstance(x,dict) and x.get("syn_type")=="GENE_SYMBOL"}
    if genes!={seed.symbol} or (seed.uniprot_id and part["accession"]!=seed.uniprot_id): return None
    return {"canonical_gene":seed.symbol,"accession":part["accession"],"organism":row["organism"],
        "source_record_sha256":digest(row),"construct_identity_verified":False}


def _key_mentions(text,key):
    if not isinstance(key,str) or not 0<len(key)<=80: return []
    prefix=r"\b(?:compound|inhibitor)\s+" if key.isdigit() else r"(?<!\w)"
    return [{"quote":m.group(),"start":m.start(),"end":m.end()} for m in re.finditer(prefix+re.escape(key)+r"(?!\w)",text,re.I)]


def resolve_assay_identities(public, brief, seed, records, *, elapsed_seconds=120):
    from drugpilot.tools.assay_followup import required_assays
    from drugpilot.tools.assay_applicability import deposited_review, summary
    original=tuple(records)
    parents=[e for e in original if source_context(e)]
    trace={"policy":POLICY,"http_call_provenance":[],"parent_context_count":len(parents),
        "documents_found":0,"activity_rows_screened":0,"linked_activity_count":0,
        "annotations_created":0,"primary_evidence_generated":False,"bioactivity_records_generated":0,
        "measurement_assignment_verified":False,"fixture":False,"result_row_limit":None,
        "page_size":PAGE_SIZE,"elapsed_limit_semantics":"admission_before_each_get_not_inflight_cancellation"}
    def finish(status,reason,updated=original):
        trace.update(status=status,http_requests_used=len(trace["http_call_provenance"]))
        return IdentityOutcome(tuple(updated),status,reason,trace)
    if "ChEMBL" not in brief.data_policy.allowed_sources:
        return finish("not_attempted","SOURCE_NOT_ALLOWED")
    if not parents or not required_assays(brief):
        return finish("not_attempted","VERIFIED_ASSAY_CONTEXT_REQUIRED")
    if len(parents)>20 or len({e.external_id for e in parents})!=len(parents):
        return finish("not_attempted","ASSAY_PARENT_COHORT_INVALID")
    reader=_Reader(public,elapsed_seconds,trace)
    stage="documents"
    try:
        pmids={e.external_id for e in parents}
        documents=reader.all("document","documents","document_chembl_id",{"pubmed_id__in":",".join(sorted(pmids))})
        if any(str(d.get("pubmed_id")) not in pmids for d in documents):
            raise ValueError("CHEMBL_DOCUMENT_PMID_MISMATCH")
        by_doc={d["document_chembl_id"]:d for d in documents}
        trace["documents_found"]=len(documents)
        activities,compound_records=[],[]
        if documents:
            stage="activities"
            filters={"document_chembl_id__in":",".join(sorted(by_doc))}
            activities=reader.all("activity","activities","activity_id",{**filters,"standard_type__in":",".join(required_assays(brief))},ordered=True)
            if any(a.get("document_chembl_id") not in by_doc or a.get("standard_type") not in required_assays(brief)
                    or not all(chembl_id(a.get(k)) for k in ("target_chembl_id","assay_chembl_id","molecule_chembl_id"))
                    or not exact_int(a.get("record_id")) or a["record_id"]<=0 for a in activities):
                raise ValueError("CHEMBL_ACTIVITY_SCOPE_MISMATCH")
            stage="compound_records"
            compound_records=reader.all("compound_record","compound_records","record_id",filters)
            if any(c.get("document_chembl_id") not in by_doc or not chembl_id(c.get("molecule_chembl_id")) for c in compound_records):
                raise ValueError("CHEMBL_COMPOUND_RECORD_SCOPE_MISMATCH")
        trace["activity_rows_screened"]=len(activities)
        stage="targets"
        targets=reader.by_ids("target","targets","target_chembl_id",sorted({a["target_chembl_id"] for a in activities}))
        matched={k:c for k,v in targets.items() if (c:=_target_component(v,seed))}
        selected=[a for a in activities if a["target_chembl_id"] in matched]
        trace["nonmatching_target_activity_count"]=len(activities)-len(selected)
        trace["screened_activity_ids_sha256"]=digest([a["activity_id"] for a in activities])
        stage="assays"
        assays=reader.by_ids("assay","assays","assay_chembl_id",sorted({a["assay_chembl_id"] for a in selected}))
        stage="molecules"
        molecules=reader.by_ids("molecule","molecules","molecule_chembl_id",sorted({a["molecule_chembl_id"] for a in selected}))
        stage="identity_join"
        compounds={c["record_id"]:c for c in compound_records}
        inventories={p:[] for p in pmids}
        for activity in selected:
            compound=compounds.get(activity["record_id"])
            assay=assays[activity["assay_chembl_id"]]
            if (compound is None or compound["document_chembl_id"]!=activity["document_chembl_id"]
                    or compound["molecule_chembl_id"]!=activity["molecule_chembl_id"]
                    or assay.get("document_chembl_id")!=activity["document_chembl_id"]
                    or assay.get("target_chembl_id")!=activity["target_chembl_id"]
                    or assay.get("assay_type")!=activity.get("assay_type")
                    or (activity.get("target_organism") is not None and activity["target_organism"]!=matched[activity["target_chembl_id"]]["organism"])
                    or any(activity.get("src_id") is not None and source.get("src_id") is not None
                        and activity["src_id"]!=source["src_id"] for source in (compound,assay))):
                raise ValueError("CHEMBL_IDENTITY_CHAIN_MISMATCH")
            document=by_doc[activity["document_chembl_id"]]
            molecule=molecules[activity["molecule_chembl_id"]]
            structures=molecule.get("molecule_structures") or {}
            if not isinstance(structures,dict): raise ValueError("CHEMBL_STRUCTURE_SCHEMA_INVALID")
            inventory={"activity_id":activity["activity_id"],"record_id":activity["record_id"],
                "document_chembl_id":activity["document_chembl_id"],"pmid":str(document["pubmed_id"]),
                "assay_chembl_id":activity["assay_chembl_id"],"molecule_chembl_id":activity["molecule_chembl_id"],
                "target_chembl_id":activity["target_chembl_id"],"target_component":matched[activity["target_chembl_id"]],
                "document_compound_key":compound.get("compound_key"),"document_compound_name":compound.get("compound_name"),
                "canonical_smiles":structures.get("canonical_smiles"),"standard_inchi_key":structures.get("standard_inchi_key"),
                "measurement":{k:activity.get(k) for k in ("type","relation","value","units","upper_value","standard_type",
                    "standard_relation","standard_value","standard_units","standard_upper_value","data_validity_comment","potential_duplicate")},
                "deposited_activity_context":{k:activity.get(k) for k in ("assay_variant_accession","assay_variant_mutation",
                    "target_organism","target_tax_id","assay_type","assay_description","bao_format","bao_label","src_id")},
                "deposited_assay_conditions":{k:assay.get(k) for k in ("description","assay_type","assay_organism","assay_cell_type",
                    "assay_tissue","assay_strain","assay_subcellular_fraction","variant_sequence","assay_parameters","confidence_score","relationship_type")},
                "source_record_hashes":{k:digest(v) for k,v in (("activity",activity),("compound_record",compound),("document",document),("assay",assay),("molecule",molecule))},
                "database_identity_chain_verified":True,"excerpt_measurement_assignment_verified":False,
                "construct_identity_verified":False,"disease_model_match_verified":False,"missing_values_imputed":False}
            inventory["deposited_applicability_review"]=deposited_review(inventory)
            inventories[str(document["pubmed_id"])].append(inventory)
        trace["linked_activity_count"]=sum(map(len,inventories.values()))
        trace["assay_applicability_summary"]=summary([r for rows in inventories.values() for r in rows])
        updated=[]
        for e in original:
            if not source_context(e):
                updated.append(e)
                continue
            rows=inventories[e.external_id]
            for row in rows:
                row["compound_key_mentions_in_excerpt"]=_key_mentions(e.metadata["excerpt"],row["document_compound_key"])
            annotation={"policy":POLICY,"source":"ChEMBL","provenance_class":"external_live","fixture":False,
                "parent_evidence_id":e.evidence_id,"parent_excerpt_sha256":e.excerpt_hash,
                "parent_abstract_sha256":e.metadata["content_hash"],"pmid":e.external_id,
                "documents":[{"document_chembl_id":d["document_chembl_id"],"pmid":str(d["pubmed_id"]),"record_sha256":digest(d)} for d in documents if str(d["pubmed_id"])==e.external_id],
                "linked_activities":rows,"linked_activity_count":len(rows),"complete_query_window":True,
                "assay_applicability_summary":summary(rows),
                "query_scope":{"pmids":sorted(pmids),"standard_types":list(required_assays(brief)),
                    "target_symbol":seed.symbol,"expected_uniprot_id":seed.uniprot_id,"target_organism":"Homo sapiens"},
                "primary_support_eligible":False,"bioactivity_records_generated":0,
                "measurement_assignment_verified":False,"http_call_provenance":trace["http_call_provenance"],
                "scope":"Deposited database identities and conditions, not confirmation of an abstract value, exact construct or efficacy."}
            annotation["annotation_sha256"]=digest(annotation)
            updated.append(e.model_copy(update={"metadata":{**e.metadata,"assay_identity_inventory":annotation}},deep=True))
        trace["annotations_created"]=len(parents)
        return finish("completed","ASSAY_IDENTITY_INVENTORY_NOT_PRIMARY_SUPPORT",updated)
    except Exception as exc:
        trace.update(error_stage=stage,error_type=type(exc).__name__)
        # No half-joined inventory survives an incomplete or inconsistent query.
        return finish("error","ASSAY_IDENTITY_SOURCE_OR_PROTOCOL_FAILURE")
