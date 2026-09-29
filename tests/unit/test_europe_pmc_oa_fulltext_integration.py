from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping

import httpx
import pytest

from drugpilot.agents.ma1 import (
    TargetSeed,
    _assign_claim_evidence,
    _primary_association_excerpt,
    _primary_direction_excerpt,
)
from drugpilot.config import Settings
from drugpilot.domain.models import (
    DataSourcePolicy,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
)
from drugpilot.orchestration.graph import DrugPilotOrchestrator
from drugpilot.tools.http import ScientificHttpClient
from drugpilot.tools.public_biology import (
    EUROPE_PMC_REST_URL,
    EUROPE_PMC_SEARCH_URL,
    PUBMED_FETCH_URL,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
    PublicBiologyRetriever,
    _EuropePMCOABudgetStop,
    _EuropePMCOAEscalationState,
    _deduplicate_references_preserving_first_lineage,
    _ordered_europe_pmc_parent_results,
)


LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/"


def _jats(
    *,
    pmcid: str,
    pmid: str,
    sentence: str,
    license_uri: str = LICENSE_URI,
    doi: str | None = None,
) -> bytes:
    doi_element = (
        f"<article-id pub-id-type='doi'>{doi}</article-id>" if doi else ""
    )
    return (
        "<article xmlns:xlink='http://www.w3.org/1999/xlink'>"
        "<front><article-meta>"
        f"<article-id pub-id-type='pmc'>{pmcid}</article-id>"
        f"<article-id pub-id-type='pmid'>{pmid}</article-id>"
        f"{doi_element}"
        f"<permissions><license xlink:href='{license_uri}'/></permissions>"
        "</article-meta></front><body><sec><title>Results</title>"
        f"<p>{sentence}</p>"
        "</sec></body></article>"
    ).encode("utf-8")


def _core_item(
    *,
    pmcid: str,
    pmid: str,
    abstract: str = "FLT3 was evaluated in acute myeloid leukemia samples.",
    title: str = "Primary FLT3 study in acute myeloid leukemia",
    publication_types: list[str] | None = None,
    license_value: str | None = "CC BY",
    open_access: str = "Y",
    doi: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "pmcid": pmcid,
        "pmid": pmid,
        "title": title,
        "abstractText": abstract,
        "pubYear": "2025",
        "journalTitle": "Example Primary Research",
        "pubTypeList": {
            "pubType": publication_types
            if publication_types is not None
            else ["Research Article", "Journal Article"]
        },
        "isOpenAccess": open_access,
    }
    if license_value is not None:
        item["license"] = license_value
    if doi is not None:
        item["doi"] = doi
    return item


class _Response:
    def __init__(
        self,
        *,
        payload: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        status_code: int = 200,
    ) -> None:
        self._payload = dict(payload or {})
        self.content = (
            content
            if content is not None
            else json.dumps(
                self._payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.text = self.content.decode("utf-8", errors="replace")
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return dict(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                "failure",
                request=request,
                response=response,
            )


class _NeverReadStream(httpx.SyncByteStream):
    def __iter__(self):
        raise AssertionError("oversized response body must not be read")

    def close(self) -> None:
        return None


class _DeadlineCrossingStream(httpx.SyncByteStream):
    def __init__(self, clock: "_Clock") -> None:
        self.clock = clock
        self.yield_count = 0

    def __iter__(self):
        self.clock.advance(21.0)
        self.yield_count += 1
        yield b"x" * (16 * 1024)
        raise AssertionError("deadline crossing must close the stream")

    def close(self) -> None:
        return None


class _ParentFullTextClient:
    def __init__(
        self,
        *,
        item: Mapping[str, Any] | None = None,
        fulltext: bytes | None = None,
    ) -> None:
        self.item = dict(item or _core_item(pmcid="PMC100", pmid="100"))
        self.fulltext = fulltext or _jats(
            pmcid="PMC100",
            pmid="100",
            sentence=(
                "FLT3 inhibition reduced FLT3 phosphorylation and viability "
                "in acute myeloid leukemia cells."
            ),
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if url == EUROPE_PMC_SEARCH_URL:
            return _Response(
                payload={"resultList": {"result": [self.item]}}
            )
        if url.endswith("/PMC100/fullTextXML"):
            return _Response(content=self.fulltext)
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, **kwargs: Any) -> _Response:
        raise AssertionError(f"unexpected POST {url}")


class _PubMedToEuropePMCClient(_ParentFullTextClient):
    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if url == PUBMED_SEARCH_URL:
            return _Response(payload={"esearchresult": {"idlist": ["100"]}})
        if url == PUBMED_SUMMARY_URL:
            return _Response(
                payload={
                    "result": {
                        "uids": ["100"],
                        "100": {
                            "uid": "100",
                            "title": "FLT3 study in acute myeloid leukemia",
                            "pubtype": ["Journal Article"],
                            "articleids": [
                                {"idtype": "pmc", "value": "PMC100"}
                            ],
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL:
            return _Response(
                content=(
                    b"<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    b"<PMID>100</PMID><Article><Abstract>"
                    b"<AbstractText>FLT3 was evaluated in acute myeloid "
                    b"leukemia samples.</AbstractText></Abstract></Article>"
                    b"</MedlineCitation><PubmedData><ArticleIdList>"
                    b"<ArticleId IdType='pmc'>PMC100</ArticleId>"
                    b"</ArticleIdList></PubmedData></PubmedArticle>"
                    b"</PubmedArticleSet>"
                )
            )
        if url == EUROPE_PMC_SEARCH_URL:
            assert kwargs["params"]["query"] == "PMCID:PMC100"
            return _Response(payload={"resultList": {"result": [self.item]}})
        if url.endswith("/PMC100/fullTextXML"):
            return _Response(content=self.fulltext)
        raise AssertionError(f"unexpected GET {url}")


class _OneHopClient(_ParentFullTextClient):
    def __init__(
        self,
        *,
        child_review: bool = False,
        child_identity_conflict: bool = False,
        parent_review: bool = False,
        mixed_case_doi: bool = False,
        include_reference_doi: bool = True,
    ) -> None:
        super().__init__(
            fulltext=_jats(
                pmcid="PMC100",
                pmid="100",
                sentence="EGFR signaling increased in lung carcinoma cells.",
            )
        )
        self.child_review = child_review
        self.child_identity_conflict = child_identity_conflict
        self.reference_doi = (
            "10.1000/CHILD" if mixed_case_doi else "10.1000/child"
        )
        self.metadata_doi = (
            "10.1000/Child" if mixed_case_doi else "10.1000/child"
        )
        self.jats_doi = "10.1000/child"
        self.include_reference_doi = include_reference_doi
        if parent_review:
            self.item = _core_item(
                pmcid="PMC100",
                pmid="100",
                publication_types=["Review", "Meta-Analysis"],
                title="Systematic review of FLT3 studies",
                license_value=None,
                open_access="N",
            )

    def get(self, url: str, **kwargs: Any) -> _Response:
        params = kwargs.get("params", {})
        query = str(params.get("query") or "")
        self.calls.append((url, kwargs))
        if url == EUROPE_PMC_SEARCH_URL and not query.startswith("EXT_ID:"):
            return _Response(
                payload={"resultList": {"result": [self.item]}}
            )
        if url.endswith("/PMC100/fullTextXML"):
            return _Response(content=self.fulltext)
        if url.endswith("/PMC/PMC100/references"):
            reference = {
                "source": "MED",
                "id": "200",
                "pmid": "200",
                "refId": "R1",
                "citedOrder": 1,
            }
            if self.include_reference_doi:
                reference["doi"] = self.reference_doi
            return _Response(
                payload={
                    "referenceList": {
                        "reference": [reference]
                    }
                }
            )
        if url == EUROPE_PMC_SEARCH_URL and query == "EXT_ID:200 AND SRC:MED":
            types = ["Review"] if self.child_review else ["Research Article"]
            return _Response(
                payload={
                    "resultList": {
                        "result": [
                            _core_item(
                                pmcid="PMC200",
                                pmid="200",
                                publication_types=types,
                                title=(
                                    "Review of FLT3 studies"
                                    if self.child_review
                                    else "Primary referenced FLT3 experiment"
                                ),
                                doi=(
                                    "10.1000/conflict"
                                    if self.child_identity_conflict
                                    else self.metadata_doi
                                ),
                            )
                        ]
                    }
                }
            )
        if url.endswith("/PMC200/fullTextXML"):
            return _Response(
                content=_jats(
                    pmcid="PMC200",
                    pmid="200",
                    doi=(
                        "10.1000/conflict"
                        if self.child_identity_conflict
                        else self.jats_doi
                    ),
                    sentence=(
                        "FLT3 inhibition reduced FLT3 phosphorylation and "
                        "viability in acute myeloid leukemia cells."
                    ),
                )
            )
        raise AssertionError(f"unexpected GET {url}")


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _TimedClient(_ParentFullTextClient):
    def __init__(self, clock: _Clock) -> None:
        super().__init__()
        self.clock = clock

    def get(self, url: str, **kwargs: Any) -> _Response:
        response = super().get(url, **kwargs)
        if url.endswith("/fullTextXML"):
            self.clock.advance(21.0)
        return response


class _DuplicateChildParentsClient:
    def __init__(self) -> None:
        self.parents = [
            _core_item(pmcid="PMC101", pmid="101"),
            _core_item(pmcid="PMC102", pmid="102"),
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        params = kwargs.get("params", {})
        query = str(params.get("query") or "")
        self.calls.append((url, kwargs))
        if url == EUROPE_PMC_SEARCH_URL and not query.startswith("EXT_ID:"):
            return _Response(payload={"resultList": {"result": self.parents}})
        for identifier in ("101", "102"):
            if url.endswith(f"/PMC{identifier}/fullTextXML"):
                return _Response(
                    content=_jats(
                        pmcid=f"PMC{identifier}",
                        pmid=identifier,
                        sentence="EGFR signaling increased in lung carcinoma cells.",
                    )
                )
            if url.endswith(f"/PMC/PMC{identifier}/references"):
                return _Response(
                    payload={
                        "referenceList": {
                            "reference": [
                                {
                                    "source": "MED",
                                    "id": "200",
                                    "pmid": "200",
                                    "doi": "10.1000/child",
                                    "refId": f"R{identifier}",
                                    "citedOrder": 1,
                                }
                            ]
                        }
                    }
                )
        if url == EUROPE_PMC_SEARCH_URL and query == "EXT_ID:200 AND SRC:MED":
            return _Response(
                payload={
                    "resultList": {
                        "result": [
                            _core_item(
                                pmcid="PMC200",
                                pmid="200",
                                doi="10.1000/child",
                            )
                        ]
                    }
                }
            )
        if url.endswith("/PMC200/fullTextXML"):
            return _Response(
                content=_jats(
                    pmcid="PMC200",
                    pmid="200",
                    doi="10.1000/child",
                    sentence=(
                        "FLT3 inhibition reduced FLT3 phosphorylation and "
                        "viability in acute myeloid leukemia cells."
                    ),
                )
            )
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, **kwargs: Any) -> _Response:
        raise AssertionError(f"unexpected POST {url}")


class _TwelveCallClient:
    def __init__(self) -> None:
        self.parents = [
            _core_item(pmcid="PMC101", pmid="101"),
            _core_item(pmcid="PMC102", pmid="102"),
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        params = kwargs.get("params", {})
        query = str(params.get("query") or "")
        self.calls.append((url, kwargs))
        if url == EUROPE_PMC_SEARCH_URL and not query.startswith("EXT_ID:"):
            return _Response(
                payload={"resultList": {"result": self.parents}}
            )
        for identifier in ("101", "102"):
            if url.endswith(f"/PMC{identifier}/fullTextXML"):
                return _Response(
                    content=_jats(
                        pmcid=f"PMC{identifier}",
                        pmid=identifier,
                        sentence="EGFR signaling increased in lung carcinoma cells.",
                    )
                )
        if url.endswith("/PMC/PMC101/references"):
            return self._references(201, 203)
        if url.endswith("/PMC/PMC102/references"):
            return self._references(204, 206)
        if url == EUROPE_PMC_SEARCH_URL and query.startswith("EXT_ID:"):
            pmid = query.split(":", 1)[1].split(" ", 1)[0]
            index = int(pmid)
            if index <= 204:
                item = _core_item(
                    pmcid=f"PMC{pmid}",
                    pmid=pmid,
                    license_value=None,
                )
            else:
                item = _core_item(pmcid=f"PMC{pmid}", pmid=pmid)
            return _Response(payload={"resultList": {"result": [item]}})
        for identifier in ("205", "206"):
            if url.endswith(f"/PMC{identifier}/fullTextXML"):
                return _Response(
                    content=_jats(
                        pmcid=f"PMC{identifier}",
                        pmid=identifier,
                        sentence="EGFR signaling increased in lung carcinoma cells.",
                    )
                )
        raise AssertionError(f"unexpected GET {url}")

    @staticmethod
    def _references(first: int, last: int) -> _Response:
        return _Response(
            payload={
                "referenceList": {
                    "reference": [
                        {
                            "source": "MED",
                            "id": str(value),
                            "pmid": str(value),
                            "refId": f"R{value}",
                            "citedOrder": value,
                        }
                        for value in range(first, last + 1)
                    ]
                }
            }
        )

    def post(self, url: str, **kwargs: Any) -> _Response:
        raise AssertionError(f"unexpected POST {url}")


def _brief(*, depth: str = "comprehensive") -> ResearchBrief:
    return ResearchBrief(
        research_goal=(
            "Find evidence-linked FLT3 candidates using bounded live research."
        ),
        disease="Acute myeloid leukemia",
        research_depth=depth,
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(allowed_sources=["Europe PMC"]),
    )


def _seed() -> TargetSeed:
    return TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
    )


def _retriever(client: Any, **kwargs: Any) -> PublicBiologyRetriever:
    max_records_per_source = int(kwargs.pop("max_records_per_source", 8))
    return PublicBiologyRetriever(
        client=client,
        timeout_seconds=30,
        max_records_per_source=max_records_per_source,
        max_attempts=1,
        retry_backoff_seconds=0,
        counter_evidence_enabled=False,
        sleeper=lambda _seconds: None,
        **kwargs,
    )


def test_flag_false_preserves_abstract_only_behavior_exactly() -> None:
    implicit_client = _ParentFullTextClient()
    explicit_client = _ParentFullTextClient()
    implicit = _retriever(implicit_client)
    explicit = _retriever(
        explicit_client,
        epmc_oa_fulltext_one_hop_v1=False,
    )
    brief = _brief()

    implicit_records = implicit.retrieve(brief, _seed(), 1)
    explicit_records = explicit.retrieve(brief, _seed(), 1)

    assert [
        item.model_dump(exclude={"retrieved_at"}) for item in implicit_records
    ] == [
        item.model_dump(exclude={"retrieved_at"}) for item in explicit_records
    ]
    assert [url for url, _ in implicit_client.calls] == [EUROPE_PMC_SEARCH_URL]
    assert [url for url, _ in explicit_client.calls] == [EUROPE_PMC_SEARCH_URL]
    assert all(
        "epmc_oa_fulltext_one_hop_trace" not in item.metadata
        for item in implicit_records
    )


def test_parent_oa_fulltext_runs_only_after_abstract_is_insufficient() -> None:
    brief = _brief().model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["Europe PMC"],
                require_primary_sources=False,
            )
        },
        deep=True,
    )
    sufficient_item = _core_item(
        pmcid="PMC100",
        pmid="100",
        abstract=(
            "FLT3 inhibition reduced FLT3 phosphorylation and viability in "
            "acute myeloid leukemia cells. FLT3 mutations were detected in "
            "42 of 200 patients (21%) with acute myeloid leukemia."
        ),
    )
    sufficient_client = _ParentFullTextClient(item=sufficient_item)
    sufficient = _retriever(
        sufficient_client,
        epmc_oa_fulltext_one_hop_v1=True,
    )

    sufficient_records = sufficient.retrieve(brief, _seed(), 1)

    assert [url for url, _ in sufficient_client.calls] == [
        EUROPE_PMC_SEARCH_URL
    ]
    assert sufficient_records[0].metadata[
        "epmc_oa_fulltext_one_hop_trace"
    ]["status"] == "not_needed"

    client = _ParentFullTextClient()
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(brief, _seed(), 1)
    fulltext = next(
        item
        for item in records
        if item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )
    abstract = next(
        item
        for item in records
        if item.external_id == "100"
        and item.metadata.get("content_availability") != "oa_fulltext_excerpt"
        and not item.metadata.get("role_specific_record")
    )
    trace = fulltext.metadata["epmc_oa_fulltext_one_hop_trace"]

    assert [url for url, _ in client.calls] == [
        EUROPE_PMC_SEARCH_URL,
        f"{EUROPE_PMC_REST_URL}/PMC100/fullTextXML",
    ]
    assert fulltext.supports == "support"
    assert fulltext.metadata["verified_direct_primary"] is True
    assert fulltext.metadata["content_retrieval_policy"] == (
        "epmc_oa_fulltext_one_hop_v1"
    )
    assert fulltext.metadata["excerpt_selection_policy"] == (
        "europe_pmc_oa_jats_results_v1"
    )
    assert fulltext.metadata["source_location"].startswith("full_text:results:")
    assert fulltext.metadata["full_text_stored"] is False
    assert fulltext.metadata["raw_xml_stored"] is False
    assert fulltext.evidence_id != abstract.evidence_id
    assert fulltext.metadata["abstract_parent_evidence_id"] == abstract.evidence_id
    assert fulltext.metadata["abstract_parent_evidence_id"] != fulltext.evidence_id
    assert fulltext.metadata["abstract_parent_content_hash"] == (
        abstract.metadata["content_hash"]
    )
    assert trace["status"] == "completed_with_evidence"
    assert trace["trigger"]["trigger_scope"] == (
        "europe_pmc_source_local_abstract_pass"
    )
    assert trace["trigger"][
        "global_cross_source_sufficiency_not_evaluated"
    ] is True
    assert trace["observed"]["added_http_call_count"] == 1
    assert trace["http_calls"][0]["role"] == "parent_fulltext"
    assert "FLT3 inhibition reduced" not in str(trace)


def test_assay_trace_priority_escalates_even_when_abstract_direction_is_sufficient() -> None:
    brief = _brief().model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["Europe PMC"],
                require_primary_sources=False,
            ),
            "human_priorities": [
                "preserve exact source and assay provenance"
            ],
        },
        deep=True,
    )
    item = _core_item(
        pmcid="PMC100",
        pmid="100",
        abstract=(
            "FLT3 inhibition reduced FLT3 phosphorylation and viability in "
            "acute myeloid leukemia cells. FLT3 mutations were detected in "
            "42 of 200 patients (21%) with acute myeloid leukemia."
        ),
    )
    fulltext = _jats(
        pmcid="PMC100",
        pmid="100",
        sentence=(
            "FLT3 inhibition was measured in an IC50 binding assay at 12 nM "
            "and reduced FLT3 phosphorylation in acute myeloid leukemia cells."
        ),
    )
    client = _ParentFullTextClient(item=item, fulltext=fulltext)

    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(brief, _seed(), 1)

    oa = next(
        record
        for record in records
        if record.metadata.get("content_availability")
        == "oa_fulltext_excerpt"
    )
    trace = oa.metadata["epmc_oa_fulltext_one_hop_trace"]
    assert trace["trigger"]["required_assay_trace_requested"] is True
    assert trace["trigger"]["abstract_required_assay_trace_available"] is False
    assert "required_assay_trace" in trace["trigger"]["needed_roles"]
    assert [url for url, _ in client.calls] == [
        EUROPE_PMC_SEARCH_URL,
        f"{EUROPE_PMC_REST_URL}/PMC100/fullTextXML",
    ]


def test_assay_trace_priority_does_not_escalate_when_abstract_is_sufficient() -> None:
    brief = _brief().model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["Europe PMC"],
                require_primary_sources=False,
            ),
            "human_priorities": [
                "preserve exact source and assay provenance"
            ],
        },
        deep=True,
    )
    item = _core_item(
        pmcid="PMC100",
        pmid="100",
        abstract=(
            "FLT3 inhibition was measured in an IC50 binding assay at 12 nM "
            "and reduced FLT3 phosphorylation in acute myeloid leukemia cells. "
            "FLT3 mutations were detected in 42 of 200 patients (21%) with "
            "acute myeloid leukemia."
        ),
    )
    client = _ParentFullTextClient(item=item)

    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(brief, _seed(), 1)

    assert [url for url, _ in client.calls] == [EUROPE_PMC_SEARCH_URL]
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]
    assert trace["status"] == "not_needed"
    assert trace["trigger"]["required_assay_trace_requested"] is True
    assert trace["trigger"]["abstract_required_assay_trace_available"] is True


def test_pubmed_pmcid_automatically_escalates_to_epmc_oa_in_focused_mode() -> None:
    client = _PubMedToEuropePMCClient()
    brief = ResearchBrief(
        research_goal="Find evidence-linked FLT3 candidates.",
        disease="Acute myeloid leukemia",
        research_depth="focused",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=False,
        ),
    )

    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(brief, _seed(), 1)

    fulltext = next(
        item
        for item in records
        if item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )
    trace = fulltext.metadata["epmc_oa_fulltext_one_hop_trace"]
    assert trace["trigger"]["trigger_scope"] == "pubmed_pmcid_cross_source"
    assert trace["cross_source_escalation"] is True
    assert trace["requested_pmcids"] == ["PMC100"]
    assert trace["resolved_exact_pmcids"] == ["PMC100"]
    assert fulltext.metadata["verified_direct_primary"] is True
    assert [url for url, _ in client.calls] == [
        PUBMED_SEARCH_URL,
        PUBMED_SUMMARY_URL,
        PUBMED_FETCH_URL,
        EUROPE_PMC_SEARCH_URL,
        f"{EUROPE_PMC_REST_URL}/PMC100/fullTextXML",
    ]


def test_official_external_jats_doctype_flows_through_retriever_to_ma1() -> None:
    fulltext = (
        b'<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal '
        b'Archiving and Interchange DTD with MathML3 v1.4 20241031//EN" '
        b'"JATS-archivearticle1-4-mathml3.dtd">\n'
        + _jats(
            pmcid="PMC100",
            pmid="100",
            sentence=(
                "FLT3 inhibition reduced FLT3 phosphorylation and viability "
                "in acute myeloid leukemia cells."
            ),
        )
    )
    client = _ParentFullTextClient(fulltext=fulltext)

    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)

    evidence = next(
        item
        for item in records
        if item.metadata.get("content_availability") == "oa_fulltext_excerpt"
        and item.metadata.get("role_specific_record") is None
    )
    assert evidence.metadata["external_doctype_removed"] is True
    assert evidence.metadata["xml_parser_policy"] == (
        "external_jats_doctype_stripped_no_dtd_no_entities_v1"
    )
    trace = evidence.metadata["epmc_oa_fulltext_one_hop_trace"]
    assert trace["observed"]["external_jats_doctypes_removed"] == 1
    assert _primary_direction_excerpt(_brief(), "FLT3", evidence) is not None
    assert evidence.metadata["full_text_stored"] is False
    assert evidence.metadata["raw_xml_stored"] is False


def test_reference_one_hop_researches_child_and_independently_verifies_it() -> None:
    client = _OneHopClient()
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    child = next(
        item
        for item in records
        if item.external_id == "200"
        and item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )
    trace = child.metadata["epmc_oa_fulltext_one_hop_trace"]

    assert child.supports == "support"
    assert child.metadata["citation_hop"] == 1
    assert child.metadata["citation_relation"] == "reference"
    assert child.metadata["citation_parent_pmcid"] == "PMC100"
    assert child.metadata["lineage_verified"] is True
    assert child.metadata["evidence_role_inherited"] is False
    assert child.metadata["independent_primary_verification"] is True
    assert [item["role"] for item in trace["http_calls"]] == [
        "parent_fulltext",
        "parent_references",
        "child_metadata",
        "child_fulltext",
    ]
    reference_call = trace["http_calls"][1]
    assert reference_call["endpoint_contract_template"] == (
        "/{source}/{id}/references"
    )
    assert reference_call["endpoint_source"] == "PMC"
    assert reference_call["source_and_accession_prefixes_are_distinct"] is True
    assert trace["observed"]["added_http_call_count"] == 4
    assert trace["forward_citations_followed"] is False
    assert trace["publisher_or_pdf_calls"] == 0
    assert _primary_direction_excerpt(_brief(), "FLT3", child) == (
        "FLT3 inhibition reduced FLT3 phosphorylation and viability in "
        "acute myeloid leukemia cells."
    )


def test_pmid_only_reference_edge_resolves_pmc_child_and_flows_into_ma1() -> None:
    records = _retriever(
        _OneHopClient(include_reference_doi=False),
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    child = next(
        item
        for item in records
        if item.external_id == "200"
        and item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )

    assert child.metadata["pmcid"] == "PMC200"
    assert child.metadata["citation_child_pmcid"] is None
    assert child.metadata["citation_child_identity_keys"] == ["MED:200"]
    assert _primary_direction_excerpt(_brief(), "FLT3", child) == (
        "FLT3 inhibition reduced FLT3 phosphorylation and viability in "
        "acute myeloid leukemia cells."
    )


def test_review_parent_is_discovery_only_and_primary_child_is_admitted() -> None:
    client = _OneHopClient(parent_review=True)
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    child = next(
        item
        for item in records
        if item.external_id == "200"
        and item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )
    trace = child.metadata["epmc_oa_fulltext_one_hop_trace"]
    review_parent = trace["discovery_parents"][0]

    assert not any(url.endswith("/PMC100/fullTextXML") for url, _ in client.calls)
    assert any(url.endswith("/PMC/PMC100/references") for url, _ in client.calls)
    assert child.supports == "support"
    assert child.metadata["citation_hop"] == 1
    assert review_parent["discovery_parent_publication_type"] == "review"
    assert review_parent["discovery_only"] is True
    assert review_parent["inherited_role"] is False
    assert review_parent["support_emitted"] is False
    assert trace["reference_audits"][0][
        "discovery_parent_publication_type"
    ] == "review"
    assert trace["reference_audits"][0]["inherited_role"] is False
    assert not any(
        item.external_id == "100" and item.supports == "support"
        for item in records
    )


def test_mixed_case_child_doi_is_canonicalized_across_lineage() -> None:
    records = _retriever(
        _OneHopClient(mixed_case_doi=True),
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    child = next(
        item
        for item in records
        if item.external_id == "200"
        and item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )

    assert child.metadata["doi"] == "10.1000/child"
    assert child.metadata["citation_child_doi"] == "10.1000/child"
    assert "DOI:10.1000/child" in child.metadata[
        "citation_child_identity_keys"
    ]


def test_jats_doi_is_preserved_when_search_metadata_omits_it() -> None:
    client = _ParentFullTextClient(
        fulltext=_jats(
            pmcid="PMC100",
            pmid="100",
            doi="10.1000/PARENT",
            sentence=(
                "FLT3 inhibition reduced FLT3 phosphorylation and viability "
                "in acute myeloid leukemia cells."
            ),
        )
    )
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    fulltext = next(
        item
        for item in records
        if item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )

    assert client.item.get("doi") is None
    assert fulltext.metadata["doi"] == "10.1000/parent"
    assert fulltext.metadata["citation_child_doi"] == "10.1000/parent"
    assert _primary_direction_excerpt(_brief(), "FLT3", fulltext) == (
        "FLT3 inhibition reduced FLT3 phosphorylation and viability in "
        "acute myeloid leukemia cells."
    )


def test_fulltext_association_role_points_to_distinct_oa_parent() -> None:
    client = _ParentFullTextClient(
        fulltext=_jats(
            pmcid="PMC100",
            pmid="100",
            sentence=(
                "In a cohort of 100 acute myeloid leukemia patients, FLT3 "
                "mutations were detected in 30% of cases."
            ),
        )
    )
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    oa_parent = next(
        item
        for item in records
        if item.metadata.get("content_availability") == "oa_fulltext_excerpt"
        and not item.metadata.get("role_specific_record")
    )
    association = next(
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    abstract = next(
        item
        for item in records
        if item.external_id == "100"
        and item.metadata.get("content_availability") != "oa_fulltext_excerpt"
        and not item.metadata.get("role_specific_record")
    )

    assert len({abstract.evidence_id, oa_parent.evidence_id, association.evidence_id}) == 3
    assert oa_parent.metadata["abstract_parent_evidence_id"] == abstract.evidence_id
    assert association.metadata["parent_evidence_id"] == oa_parent.evidence_id
    assert association.metadata["parent_evidence_id"] != association.evidence_id
    assert _primary_association_excerpt(_brief(), "FLT3", association) == (
        "In a cohort of 100 acute myeloid leukemia patients, FLT3 mutations "
        "were detected in 30% of cases."
    )


def test_one_record_budget_drops_oa_association_instead_of_dangling_parent() -> None:
    client = _ParentFullTextClient(
        fulltext=_jats(
            pmcid="PMC100",
            pmid="100",
            sentence=(
                "In a cohort of 100 acute myeloid leukemia patients, FLT3 "
                "mutations were detected in 30% of cases."
            ),
        )
    )

    records = _retriever(
        client,
        max_records_per_source=1,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)

    assert len(records) == 1
    assert not records[0].metadata.get("role_specific_record")
    assert records[0].metadata["source_record_budget_semantics"] == (
        "physical_evidence_record_hard_cap"
    )
    assert records[0].metadata[
        "source_record_atomic_association_selected_count"
    ] == 0
    assert records[0].metadata[
        "source_record_atomic_association_drop_reasons"
    ] == {"insufficient_physical_record_budget": 1}
    assert records[0].metadata[
        "source_record_dangling_parent_pointer_count"
    ] == 0


def test_two_record_budget_keeps_oa_association_and_parent_atomically() -> None:
    item = _core_item(
        pmcid="PMC100",
        pmid="100",
        abstract=(
            "FLT3 inhibition reduced FLT3 phosphorylation and viability in "
            "acute myeloid leukemia cells."
        ),
    )
    client = _ParentFullTextClient(
        item=item,
        fulltext=_jats(
            pmcid="PMC100",
            pmid="100",
            sentence=(
                "In a cohort of 100 acute myeloid leukemia patients, FLT3 "
                "mutations were detected in 30% of cases."
            ),
        ),
    )
    brief = _brief().model_copy(
        update={
            "research_goal": (
                "Establish FLT3 inhibition evidence and the primary association "
                "of FLT3 mutations in acute myeloid leukemia."
            )
        },
        deep=True,
    )

    records = _retriever(
        client,
        max_records_per_source=2,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(brief, _seed(), 1)

    assert len(records) == 2
    association = next(
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    parent = next(
        item
        for item in records
        if item.evidence_id == association.metadata["parent_evidence_id"]
    )
    assert parent.metadata["content_availability"] == "oa_fulltext_excerpt"
    assert association.metadata["parent_record_in_packet"] is True
    assert association.metadata["parent_provenance_pointer_resolved"] is True
    assert association.metadata["parent_excerpt_hash"] == parent.excerpt_hash
    assert association.metadata[
        "source_record_atomic_association_selected_count"
    ] == 1
    assert association.metadata[
        "source_record_atomic_association_dropped_count"
    ] == 0
    assert association.metadata[
        "source_record_dangling_parent_pointer_count"
    ] == 0
    assert _primary_association_excerpt(brief, "FLT3", association) == (
        association.metadata["excerpt"]
    )
    assignment = _assign_claim_evidence(
        brief,
        target_symbol="FLT3",
        evidence=records,
    )
    assert assignment.association_primary_ids == (association.evidence_id,)


def test_separate_direct_and_association_spans_share_selected_parent_hash() -> None:
    fulltext = (
        "<article xmlns:xlink='http://www.w3.org/1999/xlink'>"
        "<front><article-meta>"
        "<article-id pub-id-type='pmc'>PMC100</article-id>"
        "<article-id pub-id-type='pmid'>100</article-id>"
        f"<permissions><license xlink:href='{LICENSE_URI}'/></permissions>"
        "</article-meta></front><body><sec><title>Results</title>"
        "<p>FLT3 inhibition reduced FLT3 phosphorylation and viability in "
        "acute myeloid leukemia cells.</p>"
        "<p>In a cohort of 100 acute myeloid leukemia patients, FLT3 "
        "mutations were detected in 30% of cases.</p>"
        "</sec></body></article>"
    ).encode("utf-8")
    records = _retriever(
        _ParentFullTextClient(fulltext=fulltext),
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    oa_parent = next(
        item
        for item in records
        if item.metadata.get("content_availability") == "oa_fulltext_excerpt"
        and not item.metadata.get("role_specific_record")
    )
    association = next(
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )

    assert oa_parent.excerpt_hash != association.excerpt_hash
    assert association.metadata["parent_evidence_id"] == oa_parent.evidence_id
    assert association.metadata["parent_excerpt_hash"] == oa_parent.excerpt_hash
    assert association.metadata["association_source_excerpt_hash"] == (
        association.excerpt_hash
    )
    assert _primary_association_excerpt(_brief(), "FLT3", association) == (
        association.metadata["excerpt"]
    )
    assignment = _assign_claim_evidence(
        _brief(),
        target_symbol="FLT3",
        evidence=records,
    )
    assert assignment.direction_primary_ids == (oa_parent.evidence_id,)
    assert assignment.association_primary_ids == (association.evidence_id,)


def test_review_child_is_rejected_before_child_fulltext_call() -> None:
    client = _OneHopClient(child_review=True)
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]

    assert not any(url.endswith("/PMC200/fullTextXML") for url, _ in client.calls)
    assert trace["observed"]["child_metadata_checked"] == 1
    assert trace["observed"]["child_fulltexts_attempted"] == 0
    assert trace["eligibility_rejections"][
        "child_metadata_review_publication_type"
    ] == 1
    assert all(item.metadata.get("mock") is False for item in records)


def test_conflicting_child_identity_is_rejected_before_fulltext_call() -> None:
    client = _OneHopClient(child_identity_conflict=True)
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]

    assert not any(url.endswith("/PMC200/fullTextXML") for url, _ in client.calls)
    assert trace["observed"]["child_metadata_checked"] == 1
    assert trace["observed"]["child_fulltexts_attempted"] == 0
    assert "child_metadata_not_found" in trace["error_reason_codes"]
    assert all(item.metadata.get("mock") is False for item in records)


def test_fulltext_byte_failure_preserves_abstract_and_records_skip_reason() -> None:
    marker = b"DO_NOT_RETAIN_RAW_XML_MARKER"
    oversized = marker + b"x" * (4 * 1024 * 1024)
    client = _ParentFullTextClient(fulltext=oversized)
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]

    assert records
    assert records[0].metadata["content_availability"] != "oa_fulltext_excerpt"
    assert trace["status"] == "skipped"
    assert trace["stop_reason"] == "skipped_document_byte_budget"
    assert trace["observed"]["added_http_call_count"] == 1
    assert trace["raw_xml_stored"] is False
    assert marker.decode() not in str(records)
    assert all(item.metadata.get("mock") is False for item in records)


def test_elapsed_budget_failure_preserves_abstract_and_is_audited() -> None:
    clock = _Clock()
    client = _TimedClient(clock)
    records = _retriever(
        client,
        clock=clock,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]

    assert trace["status"] == "skipped"
    assert trace["stop_reason"] == "skipped_elapsed_budget"
    assert trace["observed"]["elapsed_seconds"] == 21.0
    assert trace["elapsed_limit_semantics"] == (
        "request_admission_and_response_acceptance_deadline"
    )
    assert trace["elapsed_budget_semantics"] == "admission_deadline"
    assert trace["exact_wall_clock_cap_claimed"] is False
    assert trace["exact_transport_wall_clock_deadline_claimed"] is False
    assert all(item.metadata.get("mock") is False for item in records)


def test_scientific_client_retry_policy_is_bypassed_for_one_wire_attempt() -> None:
    call_counts = {"search": 0, "fulltext": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            call_counts["search"] += 1
            return httpx.Response(
                200,
                json={
                    "resultList": {
                        "result": [_core_item(pmcid="PMC100", pmid="100")]
                    }
                },
            )
        if request.url.path.endswith("/PMC100/fullTextXML"):
            call_counts["fulltext"] += 1
            return httpx.Response(503, content=b"retryable upstream failure")
        raise AssertionError(f"unexpected GET {request.url}")

    client = ScientificHttpClient(
        transport=httpx.MockTransport(handler),
        max_attempts=4,
        retry_backoff_seconds=0,
        sleeper=lambda _seconds: None,
    )
    try:
        records = _retriever(
            client,
            epmc_oa_fulltext_one_hop_v1=True,
        ).retrieve(_brief(), _seed(), 1)
    finally:
        client.close()
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]
    call = trace["http_calls"][0]

    assert call_counts == {"search": 1, "fulltext": 1}
    assert trace["observed"]["added_http_call_count"] == 1
    assert call["attempt_count"] == 1
    assert call["wire_attempt_count"] == 1
    assert call["retry_count"] == 0
    assert call["retries_allowed"] is False
    assert call["transport_mode"] == "scientific_httpx_one_shot_stream"
    assert call["transport_body_cap_enforced"] is True
    assert call["status"] == "http_error"


def test_scientific_stream_rejects_declared_oversize_without_reading_body() -> None:
    call_counts = {"search": 0, "fulltext": 0}
    declared = 4 * 1024 * 1024 + 1

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            call_counts["search"] += 1
            return httpx.Response(
                200,
                json={
                    "resultList": {
                        "result": [_core_item(pmcid="PMC100", pmid="100")]
                    }
                },
            )
        if request.url.path.endswith("/PMC100/fullTextXML"):
            call_counts["fulltext"] += 1
            return httpx.Response(
                200,
                headers={"Content-Length": str(declared)},
                stream=_NeverReadStream(),
            )
        raise AssertionError(f"unexpected GET {request.url}")

    client = ScientificHttpClient(
        transport=httpx.MockTransport(handler),
        max_attempts=4,
        retry_backoff_seconds=0,
        sleeper=lambda _seconds: None,
    )
    try:
        records = _retriever(
            client,
            epmc_oa_fulltext_one_hop_v1=True,
        ).retrieve(_brief(), _seed(), 1)
    finally:
        client.close()
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]
    call = trace["http_calls"][0]

    assert call_counts == {"search": 1, "fulltext": 1}
    assert trace["status"] == "skipped"
    assert trace["stop_reason"] == "skipped_document_byte_budget"
    assert trace["observed"]["response_bytes_total"] == 0
    assert call["declared_content_length"] == declared
    assert call["response_bytes_read"] == 0
    assert call["response_body_complete"] is False
    assert call["transport_body_cap_enforced"] is True
    assert call["wire_attempt_count"] == 1


def test_scientific_stream_stops_after_chunk_crosses_elapsed_deadline() -> None:
    clock = _Clock()
    stream = _DeadlineCrossingStream(clock)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "resultList": {
                        "result": [_core_item(pmcid="PMC100", pmid="100")]
                    }
                },
            )
        if request.url.path.endswith("/PMC100/fullTextXML"):
            return httpx.Response(200, stream=stream)
        raise AssertionError(f"unexpected GET {request.url}")

    client = ScientificHttpClient(
        transport=httpx.MockTransport(handler),
        max_attempts=4,
        retry_backoff_seconds=0,
        sleeper=lambda _seconds: None,
    )
    try:
        records = _retriever(
            client,
            clock=clock,
            epmc_oa_fulltext_one_hop_v1=True,
        ).retrieve(_brief(), _seed(), 1)
    finally:
        client.close()
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]
    call = trace["http_calls"][0]

    assert stream.yield_count == 1
    assert trace["status"] == "skipped"
    assert trace["stop_reason"] == "skipped_elapsed_budget"
    assert trace["observed"]["elapsed_seconds"] == 21.0
    assert call["wire_attempt_count"] == 1
    assert call["response_body_complete"] is False
    assert call["status"] == "skipped_elapsed_budget"


def test_duplicate_child_across_parents_keeps_first_lineage_once() -> None:
    client = _DuplicateChildParentsClient()
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    child = next(
        item
        for item in records
        if item.external_id == "200"
        and item.metadata.get("content_availability") == "oa_fulltext_excerpt"
    )
    trace = child.metadata["epmc_oa_fulltext_one_hop_trace"]

    assert child.metadata["citation_parent_pmcid"] == "PMC101"
    assert trace["observed"]["cross_parent_duplicate_references_dropped"] == 1
    assert trace["observed"]["references_retained"] == 1
    assert trace["observed"]["child_metadata_checked"] == 1
    assert trace["observed"]["child_fulltexts_attempted"] == 1
    assert sum(
        url.endswith("/PMC200/fullTextXML") for url, _ in client.calls
    ) == 1
    assert all(item.metadata.get("mock") is False for item in records)


def test_cross_parent_dedupe_collapses_transitive_identifier_bridge() -> None:
    first = SimpleNamespace(identity_keys=("MED:200",), label="first")
    second = SimpleNamespace(identity_keys=("DOI:10.1000/child",), label="second")
    bridge = SimpleNamespace(
        identity_keys=("MED:200", "DOI:10.1000/child"),
        label="bridge",
    )

    retained, dropped = _deduplicate_references_preserving_first_lineage(
        (first, second, bridge)
    )

    assert [item.label for item in retained] == ["first"]
    assert dropped == 2


def test_parent_schedule_is_invariant_to_response_permutation() -> None:
    parents = [
        _core_item(pmcid="PMC300", pmid="300"),
        _core_item(pmcid="PMC100", pmid="100"),
        _core_item(pmcid="PMC200", pmid="200"),
    ]

    forward = _ordered_europe_pmc_parent_results(parents)
    reverse = _ordered_europe_pmc_parent_results(list(reversed(parents)))
    preferred = _ordered_europe_pmc_parent_results(
        list(reversed(parents)),
        preferred_pmcid_order=["PMC300", "PMC100"],
    )

    assert [item["pmcid"] for item in forward] == ["PMC100", "PMC200", "PMC300"]
    assert [item["pmcid"] for item in reverse] == ["PMC100", "PMC200", "PMC300"]
    assert [item["pmcid"] for item in preferred] == ["PMC300", "PMC100", "PMC200"]


def test_reference_phase_reserves_budget_for_one_child_lookup_and_fulltext() -> None:
    retriever = _retriever(_ParentFullTextClient())
    state = _EuropePMCOAEscalationState(
        started_at=0.0,
        http_calls=[{} for _ in range(10)],
    )

    with pytest.raises(_EuropePMCOABudgetStop) as caught:
        retriever._epmc_oa_get(
            state,
            f"{EUROPE_PMC_REST_URL}/PMC100/references",
            params={"format": "json"},
            role="parent_references",
            document_body=False,
        )

    assert caught.value.reason_code == "reference_phase_budget_reserved_for_child"


def test_unexpected_optional_lane_error_is_contained_and_preserves_abstract() -> None:
    malformed = _core_item(pmcid="not-a-pmcid", pmid="100")
    client = _ParentFullTextClient(item=malformed)
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]

    assert records
    assert [url for url, _ in client.calls] == [EUROPE_PMC_SEARCH_URL]
    assert trace["status"] == "completed_with_errors_without_evidence"
    assert trace["stop_reason"] == "unexpected_optional_lane_error_contained"
    assert trace["outer_error_contained"] is True
    assert trace["error_reason_codes"] == [
        "outer_EuropePMCOAValidationError"
    ]
    assert trace["http_calls"] == []
    assert all(item.metadata.get("mock") is False for item in records)


def test_one_hop_lane_never_exceeds_wire_and_acceptance_bounds() -> None:
    client = _TwelveCallClient()
    records = _retriever(
        client,
        epmc_oa_fulltext_one_hop_v1=True,
    ).retrieve(_brief(), _seed(), 1)
    trace = records[0].metadata["epmc_oa_fulltext_one_hop_trace"]

    assert trace["observed"]["parent_fulltexts_attempted"] == 2
    assert trace["observed"]["references_retained"] == 6
    assert trace["observed"]["child_metadata_checked"] == 6
    assert trace["observed"]["child_fulltexts_attempted"] == 2
    assert trace["observed"]["added_http_call_count"] == 12
    assert trace["observed"]["added_wire_attempt_count"] == 12
    assert len(trace["http_calls"]) == 12
    assert all(call["method"] == "GET" for call in trace["http_calls"])
    assert all(call["fixture"] is False for call in trace["http_calls"])
    assert trace["bounds"] == {
        "parent_fulltexts": 8,
        "references": 6,
        "child_fulltexts": 2,
        "added_http_calls": 12,
        "added_wire_attempts": 12,
        "bytes_per_document": 4 * 1024 * 1024,
        "aggregate_response_bytes": 12 * 1024 * 1024,
        "elapsed_seconds": 20.0,
        "elapsed_acceptance_deadline_seconds": 20.0,
        "citation_hops": 1,
    }


def test_graph_enables_feature_only_for_comprehensive_depth(tmp_path) -> None:
    assert Settings(_env_file=None).epmc_oa_fulltext_one_hop_v1 is False
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "epmc-wiring.sqlite3",
        public_evidence_enabled=True,
        epmc_oa_fulltext_one_hop_v1=True,
        evidence_research_depth="comprehensive",
    )
    orchestrator = DrugPilotOrchestrator(settings=settings)
    try:
        configured = orchestrator.ma1.proposer.retriever.public
        focused = orchestrator._ma1_for_brief(_brief(depth="focused"))
        standard = orchestrator._ma1_for_brief(_brief(depth="standard"))
        comprehensive = orchestrator._ma1_for_brief(
            _brief(depth="comprehensive")
        )

        assert configured.epmc_oa_fulltext_one_hop_v1 is True
        assert focused.proposer.retriever.public.epmc_oa_fulltext_one_hop_v1 is True
        assert standard.proposer.retriever.public.epmc_oa_fulltext_one_hop_v1 is True
        assert (
            comprehensive.proposer.retriever.public.epmc_oa_fulltext_one_hop_v1
            is True
        )
    finally:
        orchestrator.close()
