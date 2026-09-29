"""PMID-bound licensed OA methods retrieval, separate from primary evidence.

Never substitutes another paper, follows publisher links, assigns table values,
or overwrites deposited ChEMBL conditions. Failed access is not biological absence.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import re
from xml.etree import ElementTree as ET

from drugpilot.tools.assay_identity import digest, source_context
from drugpilot.tools import europe_pmc_oa_fulltext as oa

POLICY = "pmid_bound_oa_assay_methods_v1"
KEY = "assay_fulltext_context"
MAX_UNITS = 12
MAX_UNIT_CHARS = 1200
_METHODS = re.compile(r"\b(?:methods?|materials?|experimental(?: procedures| section)?|biological evaluation)\b", re.I)
_EXCLUDED = re.compile(r"\b(?:discussion|introduction|background|references?|acknowledg\w*|supplement\w*)\b", re.I)
_ASSAY = re.compile(r"\b(?:IC\s*50|EC\s*50|Ki|Kd|assays?|kinases?|cell\s+(?:culture|lines?|viability)|incubat\w*|ATP|recombinant|cytotoxic\w*|enzym\w*)\b", re.I)


def parse_methods(raw, *, eligibility, allowed_license_uris, target):
    # Reuse the established identity/license/XXE/size/depth checks unchanged.
    admitted = oa.parse_oa_jats(raw, eligibility=eligibility,
        allowed_license_uris=allowed_license_uris, max_spans=1)
    parseable,_ = oa._prepare_xml(raw)
    root = ET.fromstring(parseable)
    body = next((e for e in root if oa._local_name(e.tag) == "body"), None)
    units, table_index = [], []
    eligible_paragraphs = 0
    paragraph_index = 0
    def visit(element, *, methods=False, title="", path="body"):
        nonlocal eligible_paragraphs, paragraph_index
        tag = oa._local_name(element.tag)
        if tag in oa._BLOCKED_CONTENT_TAGS or tag in {"back", "abstract"}:
            return
        if tag == "sec":
            heading = next((c for c in element if oa._local_name(c.tag) == "title"), None)
            title = oa._normalize_text("".join(heading.itertext())) if heading is not None else title
            descriptor = str(element.get("sec-type") or "")+" "+title
            if _EXCLUDED.search(descriptor):
                return
            methods = bool(_METHODS.search(descriptor)) or methods
        if tag == "p":
            paragraph_index += 1
            text = oa._visible_text(element)
            if methods and _ASSAY.search(title+" "+text):
                eligible_paragraphs += 1
                if len(units) < MAX_UNITS:
                    # Prefix with explicit truncation; no reconstructed or merged sentence.
                    excerpt = text[:MAX_UNIT_CHARS]
                    units.append({"kind":"methods_paragraph", "source_location":path,
                        "section_title":title[:160], "paragraph_index":paragraph_index,
                        "text":excerpt, "char_start":0, "char_end":len(excerpt),
                        "excerpt_sha256":sha256(excerpt.encode()).hexdigest(),
                        "normalized_paragraph_sha256":sha256(text.encode()).hexdigest(),
                        "original_char_count":len(text), "truncated":len(text)>len(excerpt),
                        "target_mentioned":bool(re.search(r"(?<!\w)"+re.escape(target)+r"(?!\w)", excerpt, re.I)),
                        "measurement_assignment_verified":False, "experimental_equivalence_verified":False})
            return
        counts = Counter()
        for child in element:
            name = oa._local_name(child.tag)
            counts[name] += 1
            if name != "title":
                visit(child, methods=methods, title=title, path=f"{path}/{name}[{counts[name]}]")
    if body is not None:
        visit(body)
        # Table locator only. Complex headers, merged cells and footnotes are NOT
        # flattened into measurements or silently attributed to one compound.
        for index, table in enumerate((e for e in body.iter() if oa._local_name(e.tag)=="table-wrap"), 1):
            description = " ".join(oa._visible_text(e) for e in table
                if oa._local_name(e.tag) in {"label", "caption"})
            headers = " ".join(oa._visible_text(e) for e in table.iter() if oa._local_name(e.tag)=="th")
            if _ASSAY.search(description+" "+headers):
                table_index.append({"source_location":f"body//table-wrap[{index}]", "xml_id":table.get("id"),
                    "caption":description[:MAX_UNIT_CHARS], "caption_truncated":len(description)>MAX_UNIT_CHARS,
                    "table_serialization_sha256":sha256(ET.tostring(table,encoding="utf-8")).hexdigest(),
                    "measurement_rows_parsed":False, "compound_or_target_assignment_verified":False})
    return {"policy":POLICY, "document_sha256":admitted.audit.document_sha256,
        "pmid":admitted.audit.pmid, "pmcid":admitted.audit.pmcid, "doi":admitted.audit.doi,
        "license_uri":admitted.audit.license_uri, "input_bytes":len(raw),
        "xml_parser_policy":oa.XML_PARSER_POLICY, "units":units,
        "matching_method_paragraphs":eligible_paragraphs, "retained_method_paragraphs":len(units),
        "omitted_method_paragraphs":eligible_paragraphs-len(units),
        "table_locators":table_index[:MAX_UNITS], "matching_table_count":len(table_index),
        "omitted_table_locators":max(0,len(table_index)-MAX_UNITS),
        "raw_fulltext_retained":False, "primary_support_eligible":False,
        "scope":"Methods context and table locations, not primary results or assay/compound assignment"}


class BudgetStop(ValueError):
    pass


@dataclass(frozen=True)
class FulltextOutcome:
    records: tuple
    status: str
    reason: str
    trace: dict


def retrieve_assay_fulltext(public, brief, seed, records, *, elapsed_seconds=120):
    from drugpilot.tools.public_biology import (EUROPE_PMC_SEARCH_URL, _europe_pmc_fulltext_url,
        _europe_pmc_oa_eligibility, _EPMC_OA_ALLOWED_LICENSE_URIS, _EPMC_OA_PARENT_LIMIT,
        _EPMC_OA_HTTP_CALL_LIMIT, _EPMC_OA_DOCUMENT_BYTE_LIMIT, _EPMC_OA_AGGREGATE_BYTE_LIMIT,
        _EPMC_OA_ELAPSED_SECONDS)
    original = tuple(records)
    parents = [e for e in original if source_context(e)]
    trace = {"policy":POLICY, "http_call_provenance":[], "parent_context_count":len(parents),
        "documents":[], "http_requests_used":0, "annotations_created":0, "fulltexts_parsed":0,
        "method_paragraphs_retained":0, "table_locators_retained":0, "lane_errors":[],
        "fixture":False, "primary_evidence_generated":False, "paid_calls":0,
        "limits":{"http_calls":_EPMC_OA_HTTP_CALL_LIMIT,"fulltexts":_EPMC_OA_PARENT_LIMIT,
            "document_bytes":_EPMC_OA_DOCUMENT_BYTE_LIMIT,"aggregate_bytes":_EPMC_OA_AGGREGATE_BYTE_LIMIT,
            "elapsed_seconds":min(elapsed_seconds,_EPMC_OA_ELAPSED_SECONDS),"method_units_per_document":MAX_UNITS},
        "elapsed_limit_semantics":"admission_before_each_get_not_inflight_cancellation"}
    def finish(status,reason,updated=original):
        trace["http_requests_used"] = len(trace["http_call_provenance"])
        trace["document_status_counts"] = dict(Counter(d["status"] for d in trace["documents"]))
        return FulltextOutcome(tuple(updated),status,reason,trace)
    if "Europe PMC" not in brief.data_policy.allowed_sources:
        return finish("not_attempted","SOURCE_NOT_ALLOWED")
    if not public.epmc_oa_fulltext_one_hop_v1:
        return finish("not_attempted","OA_FULLTEXT_DISABLED")
    if not parents:
        return finish("not_attempted","VERIFIED_ASSAY_CONTEXT_REQUIRED")
    if len(parents)>20 or len({e.external_id for e in parents})!=len(parents):
        return finish("not_attempted","ASSAY_PARENT_COHORT_INVALID")
    deadline = public._clock()+min(elapsed_seconds,_EPMC_OA_ELAPSED_SECONDS)
    total_bytes,fulltexts = 0,0
    def get(endpoint,params=None):
        nonlocal total_bytes
        remaining = deadline-public._clock()
        if remaining<=0 or total_bytes>=_EPMC_OA_AGGREGATE_BYTE_LIMIT or len(trace["http_call_provenance"])>=_EPMC_OA_HTTP_CALL_LIMIT:
            raise BudgetStop("ASSAY_FULLTEXT_ADMISSION_BUDGET")
        call = {"endpoint":endpoint,"query_parameters":params or {},"source":"Europe PMC","fixture":False}
        trace["http_call_provenance"].append(call)
        try:
            response = public.client.get(endpoint,params=params,timeout=min(public.timeout_seconds,remaining),follow_redirects=False)
            call.update(status_code=response.status_code,response_sha256=sha256(response.content).hexdigest())
            response.raise_for_status()
            total_bytes += len(response.content)
            if len(response.content)>_EPMC_OA_DOCUMENT_BYTE_LIMIT or total_bytes>_EPMC_OA_AGGREGATE_BYTE_LIMIT:
                raise BudgetStop("ASSAY_FULLTEXT_BYTE_BUDGET")
            call["status"]="ok"
            return response
        except Exception as exc:
            call.update(status="error",error_type=type(exc).__name__)
            raise
    updated=[]
    for e in original:
        if not source_context(e):
            updated.append(e)
            continue
        doc={"pmid":e.external_id,"status":"not_attempted","reason_code":None}
        content=None
        try:
            response=get(EUROPE_PMC_SEARCH_URL,{"query":f"EXT_ID:{e.external_id} AND SRC:MED",
                "format":"json","resultType":"core","pageSize":2})
            payload=response.json()
            items=payload.get("resultList",{}).get("result",[])
            total=payload.get("hitCount")
            if type(total) is not int or total not in {0,1} or not isinstance(items,list) or len(items)!=total:
                raise oa.EuropePMCOAValidationError("EPMC_EXACT_PMID_RESULT_INVALID", "Exact PMID query must return zero or one consistent row")
            if total==0:
                doc.update(status="metadata_not_found",reason_code="EXACT_PMID_NOT_FOUND")
            else:
                item=items[0]
                if (not isinstance(item,dict) or item.get("source")!="MED" or item.get("id")!=e.external_id or item.get("pmid")!=e.external_id):
                    raise oa.EuropePMCOAValidationError("EPMC_EXACT_PMID_IDENTITY_MISMATCH", "Metadata IDs must match the verified parent PMID")
                eligibility,reason=_europe_pmc_oa_eligibility(item)
                doc.update(metadata_sha256=digest(item),pmcid=item.get("pmcid"),doi=item.get("doi"),
                    is_open_access=item.get("isOpenAccess"),license=item.get("license"))
                if eligibility is None:
                    doc.update(status="oa_not_admitted",reason_code=reason)
                elif fulltexts>=_EPMC_OA_PARENT_LIMIT:
                    raise BudgetStop("ASSAY_FULLTEXT_DOCUMENT_BUDGET")
                else:
                    fulltexts+=1
                    endpoint=_europe_pmc_fulltext_url(eligibility.pmcid)
                    raw=get(endpoint).content
                    content=parse_methods(raw,eligibility=eligibility,allowed_license_uris=_EPMC_OA_ALLOWED_LICENSE_URIS,target=seed.symbol)
                    doc.update(status="parsed",reason_code="METHODS_CONTEXT_NOT_EXPERIMENTAL_EQUIVALENCE",endpoint=endpoint,
                        document_sha256=content["document_sha256"],retained_method_paragraphs=content["retained_method_paragraphs"],
                        retained_table_locators=len(content["table_locators"]))
                    trace["fulltexts_parsed"]+=1
                    trace["method_paragraphs_retained"]+=content["retained_method_paragraphs"]
                    trace["table_locators_retained"]+=len(content["table_locators"])
        except BudgetStop as exc:
            doc.update(status="budget_not_completed",reason_code=str(exc))
        except Exception as exc:
            doc.update(status="retrieval_or_validation_error",reason_code=getattr(exc,"reason_code",type(exc).__name__))
            trace["lane_errors"].append({"pmid":e.external_id,"reason_code":doc["reason_code"]})
        trace["documents"].append(doc)
        annotation={"policy":POLICY,"source":"Europe PMC","provenance_class":"external_live","fixture":False,
            "parent_evidence_id":e.evidence_id,"parent_excerpt_sha256":e.excerpt_hash,
            "parent_abstract_sha256":e.metadata["content_hash"],"pmid":e.external_id,
            "document_status":doc,"content":content,"primary_support_eligible":False,
            "measurement_assignment_verified":False,"chembl_conditions_overwritten":False,
            "scope":"Exact-parent OA retrieval only; inaccessible or unparsed content is not biological absence"}
        annotation["annotation_sha256"]=digest(annotation)
        updated.append(e.model_copy(update={"metadata":{**e.metadata,KEY:annotation}},deep=True))
    trace["annotations_created"]=len(parents)
    trace["response_bytes"]=total_bytes
    return finish("completed","ASSAY_FULLTEXT_CONTEXT_NOT_PRIMARY_SUPPORT",updated)
