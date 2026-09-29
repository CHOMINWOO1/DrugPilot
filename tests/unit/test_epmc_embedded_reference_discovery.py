"""Synthetic isolated source fixtures only; no live run/evidence writes."""
from hashlib import sha256

import pytest

from test_europe_pmc_oa_fulltext_integration import (
    _OneHopClient, _Response, _brief, _seed, _retriever,
    EUROPE_PMC_SEARCH_URL, _primary_direction_excerpt,
)


def _refs(ids):
    return ("<back><ref-list>" + "".join(
        f"<ref id='R{i}'><element-citation><pub-id pub-id-type='pmid'>{pmid}</pub-id>"
        + ("<pub-id pub-id-type='doi'>10.1000/child</pub-id>" if pmid == "200" else "")
        + "</element-citation></ref>" for i, pmid in enumerate(ids, 1))
        + "</ref-list></back>").encode()


class EmbeddedClient(_OneHopClient):
    def __init__(self, *, ids=("200",), bad_parent_license=False, **kwargs):
        super().__init__(**kwargs)
        self.fulltext = self.fulltext.replace(b"</article>", _refs(ids) + b"</article>")
        if bad_parent_license:
            self.fulltext = self.fulltext.replace(b"https://creativecommons.org/licenses/by/4.0/", b"https://example.test/restricted")

    def get(self, url, **kwargs):
        if url.endswith("/references"):
            self.calls.append((url, kwargs))
            return _Response(status_code=503)
        query = str(kwargs.get("params", {}).get("query", ""))
        if url == EUROPE_PMC_SEARCH_URL and query.startswith("EXT_ID:") and query != "EXT_ID:200 AND SRC:MED":
            self.calls.append((url, kwargs))
            return _Response(payload={"resultList": {"result": []}})
        response = super().get(url, **kwargs)
        if url.endswith("/PMC200/fullTextXML"):
            return _Response(content=response.content.replace(b"</article>", _refs(("300",)) + b"</article>"))
        return response


def _run(client, enabled=True):
    return _retriever(client, epmc_oa_fulltext_one_hop_v1=enabled).retrieve(_brief(), _seed(), 1)


def _children(records):
    return [r for r in records if r.metadata.get("citation_hop") == 1]


def test_validated_parent_jats_references_work_without_reference_api():
    client = EmbeddedClient()
    children = _children(_run(client))
    assert children
    child = children[0]
    trace = child.metadata["epmc_oa_fulltext_one_hop_trace"]
    audit = trace["reference_audits"][0]
    assert audit["reference_source"] == "validated_parent_jats"
    assert audit["parent_document_sha256"] == sha256(client.fulltext).hexdigest()
    assert child.metadata["citation_lineage_sha256"] in audit["citation_lineage_sha256s"]
    assert audit["evidence_role_inherited"] is False
    assert child.metadata["independent_primary_verification"] is True
    assert _primary_direction_excerpt(_brief(), "FLT3", child)
    assert not any(url.endswith("/references") for url, _ in client.calls)
    assert [c["role"] for c in trace["http_calls"]] == ["parent_fulltext", "child_metadata", "child_fulltext"]
    assert not any("300" in str(kwargs.get("params", {})) for _, kwargs in client.calls)
    assert trace["forward_citations_followed"] is False
    assert trace["publisher_or_pdf_calls"] == 0


@pytest.mark.parametrize("options", [{"child_review": True}, {"child_identity_conflict": True}, {"bad_parent_license": True}])
def test_embedded_references_cannot_bypass_child_or_parent_admission(options):
    client = EmbeddedClient(**options)
    assert not _children(_run(client))


def test_embedded_reference_self_edges_duplicates_and_cap():
    client = EmbeddedClient(ids=("100", "200", "200", "201", "202", "203", "204", "205", "206", "207"))
    child = _children(_run(client))[0]
    trace = child.metadata["epmc_oa_fulltext_one_hop_trace"]
    audit = trace["reference_audits"][0]
    assert audit["emitted_reference_count"] == 6
    assert audit["references_truncated_by_limit"] is True
    assert trace["observed"]["references_retained"] == 6
    queries = [str(kwargs.get("params", {}).get("query", "")) for _, kwargs in client.calls]
    assert queries.count("EXT_ID:200 AND SRC:MED") == 1
    assert "EXT_ID:100 AND SRC:MED" not in queries
    assert not any("EXT_ID:206 " in q or "EXT_ID:207 " in q for q in queries)


def test_empty_embedded_references_keep_existing_api_lane_and_failure():
    client = EmbeddedClient(ids=())
    assert not _children(_run(client))
    assert any(url.endswith("/references") for url, _ in client.calls)


def test_embedded_references_respect_disabled_oa_lane():
    client = EmbeddedClient()
    assert not _children(_run(client, enabled=False))
    assert not any(url.endswith("/fullTextXML") or url.endswith("/references") for url, _ in client.calls)
