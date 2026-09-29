"""Source-bound assay-context retrieval, never potency or efficacy assignment.

An IC50/Ki/Kd mention beside a concentration does not tell us which compound,
construct, model or endpoint was measured. Keep the exact sentence as context;
do not create a BioactivityRecord or a primary hypothesis-support record.
"""
from dataclasses import dataclass
from hashlib import sha256
import re
from drugpilot.tools.assay_relations import quantities, sentence_spans, text_relation

POLICY = "pubmed_assay_context_followup_v1"
ROLE = "assay_context_followup"
MAX_SENTENCE_CHARS = 1200
_ASSAYS = {"ic50":"IC50", "ki":"Ki", "kd":"Kd", "ec50":"EC50"}


def required_assays(brief):
    return tuple(dict.fromkeys(_ASSAYS[value.strip().casefold()] for value in brief.constraints.required_assays
        if isinstance(value, str) and value.strip().casefold() in _ASSAYS))


def query_for(brief, seed):
    from drugpilot.tools import public_biology as pb
    assays = required_assays(brief)
    if not assays:
        raise ValueError("NO_SUPPORTED_ASSAY_CONTEXT_ENDPOINT")
    targets = tuple(dict.fromkeys((seed.symbol, *pb._fusion_query_terms(seed))))
    target = " OR ".join(f'"{pb._quote_term(value)}"[Title/Abstract]' for value in targets)
    disease = pb._pubmed_disease_query_clause(brief.disease, target_symbol=seed.symbol,
        terms=pb._disease_query_terms(brief.disease, target_symbol=seed.symbol), field="Title/Abstract")
    endpoints = " OR ".join(f'"{value}"[Title/Abstract]' for value in assays)
    return f"({target}) AND ({disease}) AND ({endpoints})"


def assay_spans(text, target, assays):
    """Exact spans only; nearby numbers are NOT assigned to a target/compound."""
    from drugpilot.tools import public_biology as pb
    if not isinstance(text, str):
        return []
    endpoints = tuple(value for value in assays if value in _ASSAYS.values())
    if not endpoints:
        return []
    # KD commonly denotes kinase domain; KI can denote potassium iodide.
    # Bare short uppercase acronyms are not assigned an assay interpretation.
    terms = [f"(?i:{re.escape(value)})" if value in {"IC50","EC50"} else re.escape(value) for value in endpoints]
    pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(terms) + r")(?![A-Za-z0-9])")
    rows = []
    for start,end,sentence in sentence_spans(text):
        if not 0 < len(sentence) <= MAX_SENTENCE_CHARS or not re.search(pb._target_token_pattern(target), sentence):
            continue
        mentions, values = list(pattern.finditer(sentence)), quantities(sentence,start)
        if not mentions or not values:
            continue
        rows.append({"start":start,"end":end,"excerpt":sentence,"excerpt_sha256":sha256(sentence.encode()).hexdigest(),
            "assay_mentions":[{"quote":v.group(),"start":start+v.start(),"end":start+v.end()} for v in mentions],
            "quantity_mentions":values,
            "measurement_assignment_verified":False,"disease_model_match_verified":False,
            "authority":"assay_search_context_not_measured_target_potency_or_primary_support"})
    return rows


@dataclass(frozen=True)
class AssayOutcome:
    records: tuple
    status: str
    stop_reason: str
    trace: dict


class AssayAdmissionBudgetExhausted(ValueError):
    """An exhausted local budget is not a retryable external outage."""


def retrieve_assay_context(public, brief, seed, *, elapsed_seconds=120):
    from drugpilot.tools import public_biology as pb
    query = query_for(brief, seed)
    common = {"db":"pubmed","tool":public.ncbi_tool}
    if public.ncbi_email:
        common["email"] = public.ncbi_email
    if public.ncbi_api_key:
        common["api_key"] = public.ncbi_api_key
    trace = {"policy":POLICY,"query":query,"query_sha256":sha256(query.encode()).hexdigest(),
        "required_assays":list(required_assays(brief)),"search_result_count":None,"returned_id_count":0,
        "records_screened":0,"unsearched_result_count":None,"screening_entries":[],"http_call_provenance":[],
        "raw_document_budget":public.primary_direction_raw_document_budget,
        "record_retention_budget":public.max_records_per_source,
        "matched_document_count":0,"omitted_matching_document_count":0,
        "elapsed_limit_semantics":"admission_before_each_http_request_not_inflight_cancellation",
        "primary_evidence_generated":False,"bioactivity_records_generated":0,"fixture":False}
    started = public._clock()
    def fetch(endpoint, **params):
        if public._clock() - started >= elapsed_seconds:
            raise AssayAdmissionBudgetExhausted("Assay-context request admission budget exhausted")
        response = None
        params = {**common, **params}
        try:
            response = public._ncbi_get(endpoint, params=params, timeout=public.timeout_seconds, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:
            trace["http_call_provenance"].append(pb._bounded_http_failure_provenance(
                endpoint=endpoint, params=params, error=exc, response=response))
            raise
        trace["http_call_provenance"].append(pb._bounded_http_call_provenance(response, endpoint=endpoint, params=params))
        return response
    records = []
    stage = "search"
    try:
        response = fetch(pb.PUBMED_SEARCH_URL, term=query, retmode="json", sort="relevance",
            retmax=public.primary_direction_raw_document_budget)
        stage = "search_schema"
        total, ids = pb._validate_pubmed_search_payload(response.json(), raw_document_budget=public.primary_direction_raw_document_budget)
        trace.update(search_result_count=total,returned_id_count=len(ids),unsearched_result_count=max(0,total-len(ids)))
        summaries, contents = {}, {}
        for start in range(0,len(ids),200):
            stage = "summary"
            batch = ids[start:start+200]
            raw = fetch(pb.PUBMED_SUMMARY_URL,id=",".join(batch),retmode="json").json()["result"]
            stage = "summary_identity"
            pb._validate_pubmed_batch_ids(batch, raw["uids"])
            if any(not isinstance(raw.get(i),dict) or str(raw[i].get("uid")) != i for i in batch):
                raise ValueError("ASSAY_SUMMARY_RECORD_IDENTITY_MISMATCH")
            summaries.update({i:raw[i] for i in batch})
        for start in range(0,len(ids),20):
            stage = "abstract"
            batch = ids[start:start+20]
            xml = pb._response_text(fetch(pb.PUBMED_FETCH_URL,id=",".join(batch),retmode="xml",rettype="abstract"))
            stage = "abstract_identity"
            pb._validate_pubmed_batch_ids(batch,pb._pubmed_xml_record_ids(xml))
            contents.update(pb._parse_pubmed_content(xml,target_symbol=seed.symbol,disease=brief.disease,
                intervention_direction=brief.constraints.intervention_direction))
        stage = "screening"
        for rank,identifier in enumerate(ids,1):
            entry = {"external_id":identifier,"rank":rank,"status":"rejected","rejection_codes":[]}
            trace["screening_entries"].append(entry)
            trace["records_screened"] += 1
            item, content = summaries[identifier], contents[identifier]
            title = str(item.get("title") or f"PubMed record {identifier}").strip()
            types = [str(v) for v in item.get("pubtype",[])]
            codes = pb._primary_publication_metadata_rejection_codes(title,types)
            if codes:
                entry["rejection_codes"] = list(codes)
                continue
            context = content.interpretation_context or {}
            text = context.get("text")
            if (context.get("status") != "complete" or not isinstance(text,str)
                    or sha256(text.encode()).hexdigest() != content.content_hash):
                entry["rejection_codes"] = ["COMPLETE_ABSTRACT_NOT_VERIFIED"]
                continue
            spans = assay_spans(text,seed.symbol,required_assays(brief))
            if not spans:
                entry["rejection_codes"] = ["NO_TARGET_ASSAY_QUANTITY_SENTENCE"]
                continue
            trace["matched_document_count"] += 1
            entry.update(status="context_found",assay_context_only=True,matched_span_count=len(spans))
            if len(records) >= public.max_records_per_source:
                trace["omitted_matching_document_count"] += 1
                entry["rejection_codes"] = ["ASSAY_CONTEXT_RETENTION_BUDGET_NOT_SCIENTIFIC_REJECTION"]
                continue
            span = spans[0]
            relation = text_relation(text,span,seed.symbol,content.content_hash)
            entry["assay_relation_status"] = relation["status"]
            records.append(public._record(source="PubMed",external_id=identifier,title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/",supports="context",query=query,
                endpoint=pb.PUBMED_FETCH_URL,excerpt_text=span["excerpt"],hash_title_when_no_excerpt=False,
                extra={"pmid":identifier,"pmcid":content.pmcid,"doi":content.doi,"publication_types":types,
                    "content_hash":content.content_hash,"content_hash_scope":"normalized_full_abstract",
                    "content_hash_algorithm":"sha256","content_availability":"abstract_excerpt","full_text_stored":False,
                    "source_location":f"abstract-assay:chars:{span['start']}-{span['end']}",
                    "excerpt":span["excerpt"],"excerpt_truncated":False,"excerpt_max_chars":MAX_SENTENCE_CHARS,
                    "excerpt_selection_policy":POLICY,"query_strategy":POLICY,
                    "role_specific_record":ROLE,"direct_support_eligible":False,"verified_direct_primary":False,
                    "direct_support_rejection_codes":["ASSAY_CONTEXT_NOT_PRIMARY_DIRECTION"],
                    "review_status":"assay_context_requires_measurement_mapping",
                    "assay_context":{k:v for k,v in span.items() if k != "excerpt"},
                    "assay_text_relation":relation,
                    "candidate_span_count":len(spans),"omitted_span_count":len(spans)-1,
                    "evidence_scope":"Quoted assay context only. Compound, construct, model and measurement linkage remain unverified."}))
        status = "completed_with_context" if records else "completed_empty" if total == 0 else "completed_bounded_without_verified_result"
        reason = "ASSAY_CONTEXT_RETRIEVED_NOT_PRIMARY_SUPPORT" if records else "ASSAY_CONTEXT_NOT_FOUND_IN_QUERY_WINDOW"
    except Exception as exc:
        # Partial transport/protocol responses cannot become retained evidence.
        records = []
        status, reason = "error", "ASSAY_CONTEXT_SOURCE_OR_PROTOCOL_FAILURE"
        trace.update(error_stage=stage,error_type=type(exc).__name__)
    trace.update(status=status,stop_reason=reason,http_requests_used=len(trace["http_call_provenance"]))
    return AssayOutcome(tuple(records),status,reason,trace)
