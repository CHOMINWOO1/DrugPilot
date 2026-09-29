"""Exact-source OA retrieval for Human Gate review, never scientific promotion.

Only official keyless ChEMBL and Europe PMC endpoints are requested. There is
no publisher scraping, URL supplied by a user, reference hopping, PDF guessing,
LLM call, candidate rewrite, or invented measurement in this path.
"""
from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import re
import time
from xml.etree import ElementTree

import httpx

from drugpilot.domain.models import (
    PublicSourceCall, SourceAcquisitionReport, SourceDocumentReview,
    SourceReviewPassage, utc_now,
)
from drugpilot.tools.europe_pmc_oa_fulltext import (
    EuropePMCOAValidationError, OAArticleEligibility, _prepare_xml, _visible_text, parse_oa_jats,
)
from drugpilot.tools.http import _build_verified_ssl_context
from drugpilot.tools.public_biology import (
    _EPMC_OA_ALLOWED_LICENSE_URIS, _europe_pmc_oa_eligibility, _europe_pmc_true,
    _europe_pmc_publication_types, _primary_publication_metadata_rejection_codes,
)

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMC_ID = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
PMC_OAI = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
LIMITS = {"http_calls": 24, "bytes_per_response": 4 * 1024 * 1024,
          "aggregate_bytes": 12 * 1024 * 1024, "elapsed_seconds": 180,
          "seconds_per_request": 20, "passages_per_document": 8,
          "chars_per_passage": 600, "refresh_cooldown_seconds": 900}
DOCUMENT_CODES = frozenset({"ASSAY_VARIANT_CONTEXT_UNKNOWN", "ASSAY_VARIANT_EXPLICIT_MISMATCH",
                           "ASSAY_VARIANT_CLASS_ONLY", "ASSAY_VARIANT_PARTIAL_OVERLAP",
                           "ASSAY_VARIANT_ANNOTATION_ONLY",
                           "ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION"})


class SourceUnavailable(ValueError):
    pass


def _id(value, pattern):
    text = str(value or "").strip()
    return text if re.fullmatch(pattern, text) else None


def _contains(text, term):
    return bool(term and re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", text, re.I))


def review_passages(raw, *, eligibility, candidate):
    """Keep bounded Methods/Results/table context separate from evidence rules.

    The existing primary-evidence parser still excludes Methods/tables. This
    independent *review-only* view does not change that promotion policy.
    """
    admitted = parse_oa_jats(raw, eligibility=eligibility,
                            allowed_license_uris=_EPMC_OA_ALLOWED_LICENSE_URIS, max_spans=1)
    parseable, _ = _prepare_xml(raw)
    root = ElementTree.fromstring(parseable)
    parents = {child: parent for parent in root.iter() for child in parent}
    aliases = [candidate.name, *candidate.metadata.get("chembl_exact_aliases", [])]
    aliases = [a for a in aliases if isinstance(a, str) and len(a) >= 3]
    passages = []
    seen = set()
    total_matching = 0
    for index, element in enumerate(root.iter()):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in {"p", "tr"}:
            continue
        lineage = []
        parent = parents.get(element)
        while parent is not None:
            lineage.append(parent)
            parent = parents.get(parent)
        tags = {x.tag.rsplit("}", 1)[-1] for x in lineage}
        if "body" not in tags or tags & {"ref-list", "ack", "fn-group", "supplementary-material"}:
            continue
        titles = [" ".join(next((_visible_text(ch) for ch in x if ch.tag.rsplit("}", 1)[-1] == "title"), "").split()) for x in lineage
                  if x.tag.rsplit("}", 1)[-1] == "sec"]
        section = " / ".join(reversed([t for t in titles if t]))
        if re.search(r"\b(introduction|background|discussion|conclusions?|references?)\b", section, re.I):
            continue
        if not re.search(r"\b(methods?|materials?|experimental|results?|findings?)\b", section, re.I):
            continue
        text = " ".join(_visible_text(element).split())
        if not text or text in seen:
            continue
        seen.add(text)
        # Screening is for review relevance, not causal/assay alignment.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        for sentence_index, sentence in enumerate(sentences):
            candidate_present = any(_contains(sentence, alias) for alias in aliases)
            target_present = _contains(sentence, candidate.target.symbol)
            topics = []
            if re.search(r"\b(ITD|wild[- ]type|mutant|mutation|variant|construct)\b", sentence, re.I):
                topics.append("variant_context")
            if re.search(r"\b(hERG|KCNH2|hepatotoxic|hepatotoxicity|hepatocyte|DILI|Ames|mutagenicity)\b", sentence, re.I):
                topics.append("safety_context")
            if not (candidate_present and (target_present or topics) or target_present and topics):
                continue
            excerpt = sentence[:LIMITS["chars_per_passage"]]
            # Flags describe the stored excerpt, never text omitted by the cap.
            candidate_present = any(_contains(excerpt, alias) for alias in aliases)
            target_present = _contains(excerpt, candidate.target.symbol)
            topics = [topic for topic, pattern in (
                ("variant_context", r"\b(ITD|wild[- ]type|mutant|mutation|variant|construct)\b"),
                ("safety_context", r"\b(hERG|KCNH2|hepatotoxic|hepatotoxicity|hepatocyte|DILI|Ames|mutagenicity)\b"))
                if re.search(pattern, excerpt, re.I)]
            if not (candidate_present and (target_present or topics) or target_present and topics):
                continue
            total_matching += 1
            if len(passages) >= LIMITS["passages_per_document"]:
                continue
            passages.append(SourceReviewPassage(text=excerpt, sha256=sha256(excerpt.encode()).hexdigest(),
                source_location=f"jats:element:{index}:{tag}:normalized_sentence:{sentence_index}",
                section=section[:500], truncated=len(sentence) > len(excerpt),
                candidate_alias_present=candidate_present, target_present=target_present, topics=topics))
    return admitted.audit, passages, total_matching > len(passages)


class PublicSourceEnricher:
    def __init__(self, *, client=None, clock=time.monotonic, ncbi_fallback=False):
        self.client = client
        self.clock = clock
        self.ncbi_fallback = ncbi_fallback
        self.last_ncbi_call = None
        self.calls = 0
        self.bytes = 0

    async def collect(self, snapshot):
        if not snapshot.candidate_packet or not snapshot.g1_verdict:
            raise ValueError("CANDIDATE_AND_G1_REQUIRED")
        owned = self.client is None
        if owned:
            verify, _ = _build_verified_ssl_context(trust_env=True)
            self.client = httpx.AsyncClient(verify=verify, follow_redirects=False, trust_env=True,
                headers={"User-Agent":"DrugPilot/0.1 exact-source-human-review"})
        self.started = self.clock()
        self.calls = self.bytes = 0
        report = SourceAcquisitionReport(run_id=snapshot.run_id,
            candidate_packet_id=snapshot.candidate_packet.packet_id,
            candidate_packet_sha256=sha256(snapshot.candidate_packet.model_dump_json().encode()).hexdigest(),
            limits=dict(LIMITS))
        if self.ncbi_fallback:
            report.policy = "exact_assay_document_epmc_ncbi_oa_review_v2"
        try:
            findings = [f for f in snapshot.g1_verdict.findings if not f.resolved]
            report.non_document_finding_codes = sorted({f.code for f in findings if f.code not in DOCUMENT_CODES})
            for candidate in snapshot.candidate_packet.candidates:
                codes = sorted({f.code for f in findings if f.artifact_id == candidate.candidate_id and f.code in DOCUMENT_CODES})
                if not codes:
                    continue
                item = SourceDocumentReview(candidate_id=candidate.candidate_id, candidate_name=candidate.name,
                                            finding_codes=codes)
                report.documents.append(item)
                try:
                    await self._candidate(candidate, item)
                except (SourceUnavailable, EuropePMCOAValidationError) as exc:
                    item.status = "unavailable" if item.calls else "not_attempted"
                    item.reason_code = exc.reason_code if isinstance(exc, EuropePMCOAValidationError) else str(exc)
                except (ValueError, TypeError, KeyError, AttributeError, ElementTree.ParseError):
                    item.status = "unavailable"
                    item.reason_code = "SOURCE_PAYLOAD_INVALID"
            return report
        finally:
            if owned:
                await self.client.aclose()
                self.client = None

    async def _get(self, item, source, endpoint, params=None):
        allowed = (re.fullmatch(re.escape(CHEMBL) + r"/(?:assay|document)/CHEMBL[0-9]+\.json", endpoint)
                   or re.fullmatch(re.escape(CHEMBL) + r"/activity/[0-9]+\.json", endpoint)
                   or endpoint == EPMC + "/search"
                   or endpoint in {PMC_ID, PMC_OAI}
                   or re.fullmatch(re.escape(EPMC) + r"/PMC[0-9]+/fullTextXML", endpoint))
        if not allowed:
            raise SourceUnavailable("NON_OFFICIAL_ENDPOINT_REJECTED")
        if endpoint in {PMC_ID, PMC_OAI}:
            if self.last_ncbi_call is not None:
                await asyncio.sleep(max(0, 0.35 - (self.clock() - self.last_ncbi_call)))
            self.last_ncbi_call = self.clock()
        remaining = LIMITS["elapsed_seconds"] - (self.clock() - self.started)
        if self.calls >= LIMITS["http_calls"]:
            raise SourceUnavailable("HTTP_CALL_BUDGET_REACHED")
        if remaining <= 0:
            raise SourceUnavailable("ELAPSED_BUDGET_REACHED")
        capacity = min(LIMITS["bytes_per_response"], LIMITS["aggregate_bytes"] - self.bytes)
        if capacity <= 0:
            raise SourceUnavailable("BYTE_BUDGET_REACHED")
        call = PublicSourceCall(source_database=source, endpoint=endpoint, parameters=params or {})
        item.calls.append(call)
        self.calls += 1
        chunks = bytearray()
        try:
            async with asyncio.timeout(min(remaining, LIMITS["seconds_per_request"])):
                async with self.client.stream("GET", endpoint, params=params,
                    headers={"Accept-Encoding":"gzip, deflate"},
                    timeout=min(remaining, LIMITS["seconds_per_request"]), follow_redirects=False) as response:
                    call.status_code = response.status_code
                    if response.status_code != 200:
                        raise SourceUnavailable(f"HTTP_{response.status_code}")
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        available = capacity - len(chunks)
                        chunks.extend(chunk[:available])
                        self.bytes += min(len(chunk), available)
                        if len(chunk) > available:
                            raise SourceUnavailable("RESPONSE_BYTE_LIMIT")
            call.complete = True
            return bytes(chunks)
        except (httpx.HTTPError, TimeoutError) as exc:
            call.reason_code = "TRANSPORT_" + type(exc).__name__
            raise SourceUnavailable(call.reason_code) from exc
        except SourceUnavailable as exc:
            call.reason_code = str(exc)
            raise
        finally:
            call.response_bytes = len(chunks)
            call.response_sha256 = sha256(chunks).hexdigest() if chunks else None
            call.finished_at = utc_now()

    async def _candidate(self, candidate, item):
        meta = candidate.metadata
        context = meta.get("chembl_assay_context") or {}
        original = context.get("record") or {}
        assay_id = _id(context.get("assay_chembl_id"), r"CHEMBL[0-9]+")
        doc_id = _id(original.get("document_chembl_id"), r"CHEMBL[0-9]+")
        target_id = _id(original.get("target_chembl_id"), r"CHEMBL[0-9]+")
        item.assay_chembl_id, item.document_chembl_id = assay_id, doc_id
        if not all((assay_id, doc_id, target_id)) or context.get("binding_status") != "exact":
            raise SourceUnavailable("EXACT_ASSAY_DOCUMENT_LINK_MISSING")
        if meta.get("fixture") is not False or meta.get("live") is not True:
            raise SourceUnavailable("LIVE_SOURCE_PROVENANCE_REQUIRED")
        # Bind the copied assay context to the candidate's original measurement
        # and EvidenceRef before spending even one public request.
        sources = [e for e in candidate.evidence if e.source == "ChEMBL"
                   and e.metadata.get("assay_chembl_id") == assay_id
                   and any(a.source_evidence_id == e.evidence_id for a in candidate.activities)]
        if len(sources) != 1:
            raise SourceUnavailable("ORIGINAL_ACTIVITY_BINDING_NOT_UNIQUE")
        evidence = sources[0]
        em = evidence.metadata
        activity_id = _id(evidence.external_id, r"[0-9]+")
        molecule_id = _id(meta.get("chembl_id"), r"CHEMBL[0-9]+")
        bound = {"activity_id": activity_id, "assay_chembl_id": assay_id,
                 "document_chembl_id": doc_id, "target_chembl_id": target_id,
                 "molecule_chembl_id": molecule_id}
        if (not all(bound.values()) or any(str(em.get(k) or "") != v for k, v in bound.items())
                or em.get("fixture") is not False or em.get("live") is not True
                or em.get("target_gene") != candidate.target.symbol
                or meta.get("measured_target") != target_id
                or not candidate.target.uniprot_id
                or meta.get("measured_target_uniprot_id") != candidate.target.uniprot_id
                or em.get("target_uniprot_id") != candidate.target.uniprot_id
                or candidate.target.metadata.get("chembl_target_id") not in (None, target_id)
                or em.get("chembl_assay_context") != context):
            raise SourceUnavailable("ORIGINAL_ACTIVITY_BINDING_CONFLICT")
        item.source_evidence_id = evidence.evidence_id
        item.source_evidence_sha256 = sha256(evidence.model_dump_json().encode()).hexdigest()
        item.source_activity_tuple = bound
        activity = json.loads(await self._get(item, "ChEMBL", f"{CHEMBL}/activity/{activity_id}.json"))
        if any(str(activity.get(k) or "") != v for k, v in bound.items()):
            raise SourceUnavailable("ACTIVITY_IDENTITY_CHANGED_OR_CONFLICTED")
        assay = json.loads(await self._get(item, "ChEMBL", f"{CHEMBL}/assay/{assay_id}.json"))
        if (assay.get("assay_chembl_id") != assay_id or assay.get("document_chembl_id") != doc_id
                or assay.get("target_chembl_id") != target_id):
            raise SourceUnavailable("ASSAY_IDENTITY_CHANGED_OR_CONFLICTED")
        document = json.loads(await self._get(item, "ChEMBL", f"{CHEMBL}/document/{doc_id}.json"))
        if document.get("document_chembl_id") != doc_id:
            raise SourceUnavailable("DOCUMENT_IDENTITY_CONFLICT")
        pmid = _id(document.get("pubmed_id"), r"[0-9]+")
        doi = _id(document.get("doi"), r"10\.[0-9]{4,9}/[^\s\"<>]+")
        if ((str(document.get("pubmed_id") or "").strip() and not pmid)
                or (str(document.get("doi") or "").strip() and not doi)):
            raise SourceUnavailable("DOCUMENT_IDENTIFIER_INVALID")
        item.pmid, item.doi = pmid, doi
        if not pmid and not doi:
            raise SourceUnavailable("DOCUMENT_HAS_NO_PUBMED_OR_DOI")
        query = f"EXT_ID:{pmid} AND SRC:MED" if pmid else f'DOI:"{doi}"'
        payload = json.loads(await self._get(item, "Europe PMC", EPMC + "/search",
            {"query":query, "format":"json", "resultType":"core", "pageSize":"2"}))
        rows = payload.get("resultList", {}).get("result", [])
        if payload.get("hitCount") != 1 or not isinstance(rows, list) or len(rows) != 1:
            raise SourceUnavailable("PUBLICATION_IDENTITY_NOT_UNIQUE")
        row = rows[0]
        if (pmid and (str(row.get("pmid") or row.get("id")) != pmid or row.get("source") != "MED"
                      or row.get("id") and str(row["id"]) != pmid)
                or doi and str(row.get("doi") or "").casefold() != doi.casefold()):
            raise SourceUnavailable("PUBLICATION_IDENTITY_CONFLICT")
        if row.get("source") == "MED":
            row = {**row, "pmid": row.get("pmid") or row.get("id")}
        eligibility, reason = _europe_pmc_oa_eligibility(row)
        item.is_open_access = _europe_pmc_true(row["isOpenAccess"]) if "isOpenAccess" in row else None
        retraction_fields = [row[k] for k in ("isRetracted", "retracted", "is_retracted") if k in row]
        item.is_retracted = any(_europe_pmc_true(v) for v in retraction_fields) if retraction_fields else None
        item.pmcid = _id(row.get("pmcid"), r"PMC[0-9]+")
        if item.pmcid:
            item.source_url = f"https://europepmc.org/articles/{item.pmcid}"
        if eligibility is None:
            item.attempt_outcomes.append("EUROPE_PMC_OA_" + str(reason).upper())
            if self.ncbi_fallback and reason in {"pmcid_missing", "not_open_access", "license_missing", "license_not_allowed"}:
                await self._ncbi_candidate(candidate, item, row)
                return
            raise SourceUnavailable("OA_" + str(reason).upper())
        # Identity and license are checked again against the full JATS document.
        try:
            raw = await self._get(item, "Europe PMC", f"{EPMC}/{eligibility.pmcid}/fullTextXML")
        except SourceUnavailable as exc:
            item.attempt_outcomes.append("EUROPE_PMC_" + str(exc))
            if self.ncbi_fallback and (str(exc).startswith("HTTP_") or str(exc).startswith("TRANSPORT_")):
                await self._ncbi_candidate(candidate, item, row)
                return
            raise
        audit, passages, truncated = review_passages(raw, eligibility=eligibility, candidate=candidate)
        item.document_sha256 = audit.document_sha256
        item.pmid, item.pmcid, item.doi = audit.pmid, audit.pmcid, audit.doi
        item.license = audit.license_uri
        item.publication_types = list(audit.publication_types)
        item.is_open_access = eligibility.is_open_access
        item.is_retracted = eligibility.is_retracted
        item.identity_verified = True
        item.passages, item.passage_limit_reached = passages, truncated
        item.status = "available_for_review"
        item.reason_code = "REVIEW_CONTEXT_FOUND" if passages else "NO_RELEVANT_CONTEXT_FOUND"
        item.fulltext_source = "Europe PMC"
        item.attempt_outcomes.append("EUROPE_PMC_" + item.reason_code)

    async def _ncbi_candidate(self, candidate, item, row):
        """Independent official fallback, with front-matter/full-text agreement.

        Identity conflicts/retractions are never bypassed by another provider.
        PMC-OAI permits automated reuse retrieval; no article web scraping.
        """
        types = _europe_pmc_publication_types(row)
        if not types or _primary_publication_metadata_rejection_codes(str(row.get("title") or ""), types):
            raise SourceUnavailable("NCBI_PRIMARY_PUBLICATION_STATUS_REQUIRED")
        if any(_europe_pmc_true(row.get(k)) for k in ("isRetracted", "retracted", "is_retracted")):
            raise SourceUnavailable("NCBI_RETRACTED_PUBLICATION")
        identity = item.pmid or item.doi
        payload = json.loads(await self._get(item, "NCBI PMC", PMC_ID,
            {"ids":identity, "idtype":"pmid" if item.pmid else "doi", "format":"json", "tool":"DrugPilot"}))
        records = payload.get("records", [])
        if payload.get("status") != "ok" or not isinstance(records, list) or len(records) != 1:
            raise SourceUnavailable("NCBI_IDENTITY_NOT_UNIQUE")
        record = records[0]
        if str(record.get("requested-id") or "") != identity:
            raise SourceUnavailable("NCBI_REQUEST_IDENTITY_CONFLICT")
        if record.get("status") == "error" or record.get("errmsg") or record.get("error"):
            raise SourceUnavailable("NCBI_PMC_RECORD_UNAVAILABLE")
        pmcid = _id(record.get("pmcid"), r"PMC[0-9]+")
        if not pmcid:
            raise SourceUnavailable("NCBI_PMCID_MISSING")
        if (item.pmid and str(record.get("pmid") or "") != item.pmid
                or item.doi and str(record.get("doi") or "").casefold() != item.doi.casefold()
                or item.pmcid and item.pmcid != pmcid):
            raise SourceUnavailable("NCBI_PUBLICATION_IDENTITY_CONFLICT")
        if record.get("live") in (False, "false") or record.get("release-date"):
            raise SourceUnavailable("NCBI_EMBARGO_OR_RELEASE_CONDITION")
        item.pmcid = pmcid
        item.source_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        identifier = "oai:pubmedcentral.nih.gov:" + pmcid[3:]
        metadata_raw = await self._get(item, "NCBI PMC", PMC_OAI,
            {"verb":"GetRecord", "identifier":identifier, "metadataPrefix":"pmc_fm"})
        metadata, header = _oai_article(metadata_raw, identifier)
        if "pmc-open" not in [e.text for e in header if e.tag.rsplit("}",1)[-1] == "setSpec"]:
            raise SourceUnavailable("NCBI_REUSE_SET_NOT_CONFIRMED")
        licenses = {v for e in metadata.iter() if e.tag.rsplit("}",1)[-1] == "license"
                    for k, v in e.attrib.items() if k.rsplit("}",1)[-1] == "href"}
        if len(licenses) != 1 or not licenses.issubset(_EPMC_OA_ALLOWED_LICENSE_URIS):
            raise SourceUnavailable("NCBI_LICENSE_NOT_ALLOWED_OR_AMBIGUOUS")
        eligibility = OAArticleEligibility(pmcid=pmcid, pmid=item.pmid, doi=item.doi,
            publication_types=tuple(types), license_uri=next(iter(licenses)), is_open_access=True, is_retracted=False)
        parse_oa_jats(ElementTree.tostring(metadata), eligibility=eligibility,
            allowed_license_uris=_EPMC_OA_ALLOWED_LICENSE_URIS, max_spans=1)
        full_raw = await self._get(item, "NCBI PMC", PMC_OAI,
            {"verb":"GetRecord", "identifier":identifier, "metadataPrefix":"pmc"})
        article, full_header = _oai_article(full_raw, identifier)
        if "pmc-open" not in [e.text for e in full_header if e.tag.rsplit("}",1)[-1] == "setSpec"]:
            raise SourceUnavailable("NCBI_REUSE_SET_NOT_CONFIRMED")
        audit, passages, truncated = review_passages(ElementTree.tostring(article), eligibility=eligibility, candidate=candidate)
        item.document_sha256, item.license = audit.document_sha256, audit.license_uri
        item.publication_types, item.is_open_access, item.is_retracted = list(audit.publication_types), True, False
        item.identity_verified, item.passages, item.passage_limit_reached = True, passages, truncated
        item.fulltext_source, item.document_hash_scope = "NCBI PMC", "extracted_jats_serialization"
        item.status = "available_for_review"
        item.reason_code = "REVIEW_CONTEXT_FOUND" if passages else "NO_RELEVANT_CONTEXT_FOUND"
        item.attempt_outcomes.append("NCBI_PMC_" + item.reason_code)


def _oai_article(raw, identifier):
    safe, _ = _prepare_xml(raw)
    root = ElementTree.fromstring(safe)
    ns = {"o":"http://www.openarchives.org/OAI/2.0/"}
    records = root.findall("o:GetRecord/o:record", ns)
    if root.findall("o:error", ns) or len(records) != 1:
        raise SourceUnavailable("NCBI_OAI_RECORD_UNAVAILABLE")
    record = records[0]
    header = record.find("o:header", ns)
    if header is None or header.get("status") == "deleted" or header.findtext("o:identifier", namespaces=ns) != identifier:
        raise SourceUnavailable("NCBI_OAI_IDENTITY_CONFLICT")
    metadata = record.find("o:metadata", ns)
    articles = [e for e in (metadata if metadata is not None else []) if e.tag.rsplit("}",1)[-1] == "article"]
    if len(articles) != 1:
        raise SourceUnavailable("NCBI_OAI_ARTICLE_NOT_UNIQUE")
    return articles[0], header


__all__ = ["PublicSourceEnricher", "review_passages", "LIMITS"]
