"""Isolated protocol fixtures only; no production science or network calls."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace as NS
import json

import httpx
import pytest

from drugpilot.domain.models import EvidenceRef, SourceDocumentReview
from drugpilot.tools.review_source_enrichment import PublicSourceEnricher, SourceUnavailable, LIMITS, CHEMBL

LICENSE = "https://creativecommons.org/licenses/by/4.0/"
BODY = "<sec><title>Methods</title><p>Testdrug inhibition of TEST1 wild-type was measured.</p></sec>"


def oai(jats, *, reuse=True, deleted=False):
    return ('<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record>'
        + ('<header status="deleted">' if deleted else '<header>')
        + '<identifier>oai:pubmedcentral.nih.gov:10001</identifier>'
        + ('<setSpec>pmc-open</setSpec>' if reuse else '')
        + '</header><metadata>' + jats.replace('<article ', '<article xmlns="" ', 1)
        + '</metadata></record></GetRecord></OAI-PMH>')


async def retrieve_ncbi(*, mutation=None):
    c, rs = setup_case()
    row = rs["/search"]["resultList"]["result"][0]
    row["pmcid"] = None
    record = {"requested-id":"10001", "pmcid":"PMC10001", "pmid":10001, "doi":"10.1234/test"}
    full = rs["/PMC10001/fullTextXML"]
    front = full.replace(f"<body>{BODY}</body>", "")
    reuse, deleted = True, False
    if mutation == "identity": record["doi"] = "10.1234/other"
    if mutation == "unavailable": record.update(status="error", errmsg="not in PMC")
    if mutation == "embargo": record["release-date"] = "2099-01-01"
    if mutation == "retracted": row["isRetracted"] = "Y"
    if mutation == "review": row["pubTypeList"] = {"pubType":["Review"]}
    if mutation == "reuse": reuse = False
    if mutation == "deleted": deleted = True
    if mutation == "license": front = front.replace(LICENSE, "https://invalid.example/license")
    if mutation == "full_identity": full = full.replace("10.1234/test", "10.1234/other")
    if mutation == "full_license": full = full.replace(LICENSE, "https://creativecommons.org/licenses/by-nc/4.0/")
    requests = []
    def handler(request):
        requests.append(request)
        assert request.method == "GET" and "authorization" not in request.headers
        if request.url.host == "www.ebi.ac.uk":
            value = next(v for k,v in rs.items() if request.url.path.endswith(k))
            return httpx.Response(200, json=value)
        assert request.url.host == "pmc.ncbi.nlm.nih.gov"
        assert request.headers["accept-encoding"] == "gzip, deflate"
        if "idconv" in request.url.path:
            assert request.url.params["ids"] == "10001"
            return httpx.Response(200, json={"status":"ok", "records":[record]})
        assert request.url.params["identifier"] == "oai:pubmedcentral.nih.gov:10001"
        value = oai(front if request.url.params["metadataPrefix"] == "pmc_fm" else full, reuse=reuse, deleted=deleted)
        if mutation == "malformed": value = "<unclosed"
        if mutation == "entity": value = '<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///secrets">]>' + value
        return httpx.Response(200, text=value)
    s = NS(run_id="test_run", candidate_packet=NS(packet_id="test_packet", candidates=[c], model_dump_json=lambda:"test-only"),
        g1_verdict=NS(findings=[NS(code="ASSAY_VARIANT_CONTEXT_UNKNOWN", resolved=False, artifact_id=c.candidate_id)]))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await PublicSourceEnricher(client=client, ncbi_fallback=True).collect(s)
    return report, requests


@pytest.mark.asyncio
async def test_ncbi_official_fallback_and_two_level_hashes():
    report, requests = await retrieve_ncbi()
    d = report.documents[0]
    assert d.status == "available_for_review", d.reason_code
    assert len(requests) == 7 and len(d.passages) == 1 and d.identity_verified
    assert d.fulltext_source == "NCBI PMC" and d.document_hash_scope == "extracted_jats_serialization"
    assert d.document_sha256 != d.calls[-1].response_sha256
    assert d.license == LICENSE and report.model_calls == 0 and not report.candidate_modified
    assert d.attempt_outcomes[0] == "EUROPE_PMC_OA_PMCID_MISSING"
    assert report.policy == "exact_assay_document_epmc_ncbi_oa_review_v2"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["identity", "unavailable", "embargo", "retracted", "review", "reuse",
    "deleted", "license", "full_identity", "full_license", "malformed", "entity"])
async def test_ncbi_fallback_fails_closed(mutation):
    report, requests = await retrieve_ncbi(mutation=mutation)
    d = report.documents[0]
    assert d.status == "unavailable" and not d.passages and not d.identity_verified, d
    if mutation in {"retracted", "review"}:
        assert not any(r.url.host == "pmc.ncbi.nlm.nih.gov" for r in requests)
    if mutation == "malformed": assert d.reason_code == "SOURCE_PAYLOAD_INVALID"


def setup_case():
    bound = dict(activity_id="123", assay_chembl_id="CHEMBL12", document_chembl_id="CHEMBL13",
                 target_chembl_id="CHEMBL14", molecule_chembl_id="CHEMBL15")
    context = {"binding_status":"exact", "assay_chembl_id":"CHEMBL12", "record":bound.copy()}
    evidence = EvidenceRef(source="ChEMBL", external_id="123", title="Test-only measurement",
        metadata={**bound, "fixture":False, "live":True, "target_gene":"TEST1", "target_uniprot_id":"P12345", "chembl_assay_context":context})
    c = NS(candidate_id="test_candidate", name="Testdrug", metadata={"fixture":False, "live":True,
        "chembl_id":"CHEMBL15", "measured_target":"CHEMBL14", "measured_target_uniprot_id":"P12345", "chembl_assay_context":context},
        target=NS(symbol="TEST1", uniprot_id="P12345", metadata={"chembl_target_id":"CHEMBL14"}), evidence=[evidence],
        activities=[NS(source_evidence_id=evidence.evidence_id)])
    row = dict(id="10001", source="MED", pmid="10001", pmcid="PMC10001", doi="10.1234/test",
        isOpenAccess="Y", isRetracted="N", license="CC BY", pubTypeList={"pubType":["Journal Article"]})
    jats = ('<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>'
        '<article-id pub-id-type="pmc">PMC10001</article-id><article-id pub-id-type="pmid">10001</article-id>'
        '<article-id pub-id-type="doi">10.1234/test</article-id>'
        f'<permissions><license xlink:href="{LICENSE}"/></permissions></article-meta></front>'
        f'<body>{BODY}</body></article>')
    responses = {"/activity/123.json":bound.copy(), "/assay/CHEMBL12.json":bound.copy(),
        "/document/CHEMBL13.json":{"document_chembl_id":"CHEMBL13", "pubmed_id":10001, "doi":"10.1234/test"},
        "/search":{"hitCount":1, "resultList":{"result":[row]}}, "/PMC10001/fullTextXML":jats}
    return c, responses


async def retrieve(c, responses, *, status=200):
    requests = []
    def handler(request):
        requests.append(request)
        assert request.method == "GET" and request.url.host == "www.ebi.ac.uk"
        assert "authorization" not in request.headers
        value = next(v for k, v in responses.items() if request.url.path.endswith(k))
        return httpx.Response(status, json=value) if isinstance(value, dict) else httpx.Response(status, text=value)
    packet = NS(packet_id="test_packet", candidates=[c], model_dump_json=lambda: "test-only-packet")
    s = NS(run_id="test_run", candidate_packet=packet,
        g1_verdict=NS(findings=[NS(code="ASSAY_VARIANT_CONTEXT_UNKNOWN", resolved=False, artifact_id=c.candidate_id)]))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await PublicSourceEnricher(client=client).collect(s)
    return report, requests


@pytest.mark.asyncio
async def test_exact_source_full_chain_is_context_only():
    c, responses = setup_case()
    original = deepcopy(c.__dict__)
    report, requests = await retrieve(c, responses)
    d = report.documents[0]
    assert d.status == "available_for_review", d.reason_code
    assert len(requests) == 5 and all(call.complete for call in d.calls)
    assert d.identity_verified and d.license == LICENSE and d.is_retracted is False
    assert d.source_activity_tuple["molecule_chembl_id"] == "CHEMBL15"
    assert d.source_evidence_sha256 == sha256(c.evidence[0].model_dump_json().encode()).hexdigest()
    assert d.document_sha256 == sha256(responses["/PMC10001/fullTextXML"].encode()).hexdigest()
    p = d.passages[0]
    assert p.section == "Methods" and p.evidence_role == "review_context_not_verified_assay_claim"
    assert p.candidate_alias_present and p.target_present and p.topics == ["variant_context"]
    assert p.sha256 == sha256(p.text.encode()).hexdigest()
    assert d.scientific_resolution == "not_automatically_resolved"
    assert report.model_calls == 0 and not report.candidate_modified and not report.fixture_used
    assert c.__dict__ == original
    assert requests[3].url.params["query"] == "EXT_ID:10001 AND SRC:MED"


@pytest.mark.asyncio
@pytest.mark.parametrize("which,key,value,reason", [
    ("activity", "molecule_chembl_id", "CHEMBL999", "ACTIVITY_IDENTITY_CHANGED_OR_CONFLICTED"),
    ("assay", "target_chembl_id", "CHEMBL999", "ASSAY_IDENTITY_CHANGED_OR_CONFLICTED"),
    ("document", "document_chembl_id", "CHEMBL999", "DOCUMENT_IDENTITY_CONFLICT"),
    ("document", "pubmed_id", "not-a-pmid", "DOCUMENT_IDENTIFIER_INVALID"),
    ("document", "doi", "not-a-doi", "DOCUMENT_IDENTIFIER_INVALID"),
    ("row", "pmid", "999", "PUBLICATION_IDENTITY_CONFLICT"),
    ("row", "id", "999", "PUBLICATION_IDENTITY_CONFLICT"),
    ("row", "doi", "10.1234/other", "PUBLICATION_IDENTITY_CONFLICT"),
    ("row", "isOpenAccess", "N", "OA_NOT_OPEN_ACCESS"),
    ("row", "isRetracted", "Y", "OA_RETRACTED_PUBLICATION"),
    ("row", "license", None, "OA_LICENSE_MISSING"),
    ("row", "license", "All rights reserved", "OA_LICENSE_NOT_ALLOWED"),
    ("row", "pubTypeList", {"pubType":["Review"]}, "OA_METADATA"),
])
async def test_source_identity_status_and_license_fail_closed(which, key, value, reason):
    c, rs = setup_case()
    payload = rs["/search"]["resultList"]["result"][0] if which == "row" else next(v for k, v in rs.items() if k.startswith("/"+which+"/"))
    payload[key] = value
    report, requests = await retrieve(c, rs)
    d = report.documents[0]
    assert d.reason_code.startswith(reason), d.reason_code
    assert not d.passages and not d.identity_verified
    assert not any(r.url.path.endswith("fullTextXML") for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["evidence", "measurement", "candidate", "target", "fixture"])
async def test_original_candidate_binding_checked_before_network(mutation):
    c, rs = setup_case()
    if mutation == "evidence": c.evidence[0].metadata["molecule_chembl_id"] = "CHEMBL999"
    if mutation == "measurement": c.activities = []
    if mutation == "candidate": c.metadata["chembl_id"] = "CHEMBL999"
    if mutation == "target": c.target.metadata["chembl_target_id"] = "CHEMBL999"
    if mutation == "fixture": c.metadata["fixture"] = True
    report, requests = await retrieve(c, rs)
    assert not requests and not report.documents[0].passages


@pytest.mark.asyncio
@pytest.mark.parametrize("old,new", [("PMC10001", "PMC999"), (">10001<", ">999<"),
    ("10.1234/test", "10.1234/other"), (LICENSE, "https://invalid.example/license"),
    ("<article ", '<!DOCTYPE article [<!ENTITY secret SYSTEM "file:///secrets">]><article ')])
async def test_jats_rechecks_ids_license_and_entities(old, new):
    c, rs = setup_case()
    rs["/PMC10001/fullTextXML"] = rs["/PMC10001/fullTextXML"].replace(old,new)
    report, _ = await retrieve(c, rs)
    assert report.documents[0].status == "unavailable" and not report.documents[0].passages


@pytest.mark.asyncio
@pytest.mark.parametrize("body,expected", [
    ("<sec><title>Methods</title><p>Otherdrug inhibition of OTHER1 was measured.</p></sec>", 0),
    ("<sec><title>Introduction</title><p>Testdrug inhibition of TEST1 wild-type was measured.</p></sec>", 0),
    ("<sec><title>Results</title><table-wrap><table><tr><td>Testdrug TEST1</td><td>wild-type inhibition</td></tr></table></table-wrap></sec>", 1),
    ("<sec><title>Methods</title><p>" + "padding "*90 + "Testdrug TEST1 wild-type.</p></sec>", 0),
])
async def test_context_scope_and_truncation_flags(body, expected):
    c, rs = setup_case()
    rs["/PMC10001/fullTextXML"] = rs["/PMC10001/fullTextXML"].replace(BODY, body)
    report, _ = await retrieve(c, rs)
    assert report.documents[0].status == "available_for_review"
    assert len(report.documents[0].passages) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 429, 500])
async def test_http_errors_not_retried_or_redirected(status):
    c, rs = setup_case()
    report, requests = await retrieve(c, rs, status=status)
    assert len(requests) == 1 and report.documents[0].reason_code == f"HTTP_{status}"


@pytest.mark.asyncio
async def test_ambiguous_identity_and_missing_identifiers():
    c, rs = setup_case()
    rs["/search"]["hitCount"] = 2
    report, _ = await retrieve(c, rs)
    assert report.documents[0].reason_code == "PUBLICATION_IDENTITY_NOT_UNIQUE"
    rs["/document/CHEMBL13.json"].update(pubmed_id=None, doi=None)
    report, requests = await retrieve(c, rs)
    assert len(requests) == 3 and report.documents[0].reason_code == "DOCUMENT_HAS_NO_PUBMED_OR_DOI"


@pytest.mark.asyncio
async def test_caps_and_nonofficial_url(monkeypatch):
    c, rs = setup_case()
    monkeypatch.setitem(LIMITS, "bytes_per_response", 10)
    report, requests = await retrieve(c, rs)
    d = report.documents[0]
    assert len(requests) == 1 and d.reason_code == "RESPONSE_BYTE_LIMIT"
    assert d.calls[0].response_bytes == 10 and d.calls[0].complete is False
    enricher = PublicSourceEnricher()
    item = SourceDocumentReview(candidate_id="test", candidate_name="test")
    with pytest.raises(SourceUnavailable, match="NON_OFFICIAL"):
        await enricher._get(item, "ChEMBL", "http://127.0.0.1/private")
    enricher.started = enricher.clock()
    enricher.calls = LIMITS["http_calls"]
    with pytest.raises(SourceUnavailable, match="HTTP_CALL_BUDGET"):
        await enricher._get(item, "ChEMBL", CHEMBL + "/activity/123.json")


@pytest.mark.asyncio
async def test_normalized_candidate_identity_does_not_require_duplicate_target_metadata():
    c, rs = setup_case()
    c.target.metadata = {}
    report, _ = await retrieve(c, rs)
    assert report.documents[0].status == "available_for_review"
    c.target.uniprot_id = "P99999"
    report, requests = await retrieve(c, rs)
    assert not requests and report.documents[0].reason_code == "ORIGINAL_ACTIVITY_BINDING_CONFLICT"


@pytest.mark.asyncio
async def test_namespace_and_boolean_oa_metadata():
    c, rs = setup_case()
    rs["/search"]["resultList"]["result"][0]["isOpenAccess"] = True
    rs["/PMC10001/fullTextXML"] = rs["/PMC10001/fullTextXML"].replace('<article ', '<article xmlns="urn:jats" ')
    report, _ = await retrieve(c, rs)
    d = report.documents[0]
    assert d.status == "available_for_review" and d.is_open_access is True
    assert len(d.passages) == 1 and d.passages[0].section == "Methods"
    rs["/search"]["resultList"]["result"][0]["isRetracted"] = False
    rs["/PMC10001/fullTextXML"] = "not-xml"
    report, _ = await retrieve(c, rs)
    d = report.documents[0]
    assert d.status == "unavailable" and d.is_open_access is True and d.is_retracted is False
