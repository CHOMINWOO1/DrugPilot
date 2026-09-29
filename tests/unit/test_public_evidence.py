from __future__ import annotations

import httpx

from drugpilot.agents.ma1 import MockEvidenceRetriever, TargetSeed
from drugpilot.domain.models import HypothesisTrack, ResearchBrief
from drugpilot.tools.evidence import PublicEvidenceRetriever, ResilientEvidenceRetriever


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failure", request=request, response=response)

    def json(self) -> dict:
        return self.payload


class FakePublicClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.get_calls = 0
        self.post_calls = 0

    def get(self, url: str, **kwargs) -> FakeResponse:
        del url, kwargs
        self.get_calls += 1
        if self.fail:
            raise httpx.ConnectTimeout("offline")
        return FakeResponse(
            {
                "resultList": {
                    "result": [
                        {
                            "pmid": "12345678",
                            "title": "FLT3 inhibition in acute myeloid leukemia",
                            "pubYear": "2025",
                            "journalTitle": "Example Journal",
                            "citedByCount": 12,
                        }
                    ]
                }
            }
        )

    def post(self, url: str, **kwargs) -> FakeResponse:
        del url, kwargs
        self.post_calls += 1
        if self.fail:
            raise httpx.ConnectTimeout("offline")
        return FakeResponse(
            {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": "ENSG00000122025",
                                "name": "FLT3",
                                "entity": "target",
                                "description": "Receptor tyrosine kinase",
                            }
                        ]
                    }
                }
            }
        )


def _brief() -> ResearchBrief:
    return ResearchBrief(
        research_goal="Find evidence-linked FLT3 candidates for audit testing.",
        disease="Acute myeloid leukemia",
    )


def _seed() -> TargetSeed:
    return TargetSeed(
        "FLT3",
        "Fms related receptor tyrosine kinase 3",
        HypothesisTrack.LITERATURE,
        0.95,
    )


def test_public_retriever_returns_traceable_records_and_reuses_cache() -> None:
    client = FakePublicClient()
    retriever = PublicEvidenceRetriever(client=client, timeout_seconds=1)

    first = list(retriever.retrieve(_brief(), _seed(), 1))
    second = list(retriever.retrieve(_brief(), _seed(), 2))

    assert {item.source for item in first} == {"Europe PMC", "Open Targets"}
    assert all(item.metadata["provenance_class"] == "external_live" for item in first)
    assert all(item.metadata["provenance_class"] == "external_cache" for item in second)
    assert first[0].url == "https://europepmc.org/article/MED/12345678"
    assert client.get_calls == 1
    assert client.post_calls == 1


def test_resilient_retriever_adds_explicit_fixture_counter() -> None:
    public = PublicEvidenceRetriever(client=FakePublicClient(), timeout_seconds=1)
    retriever = ResilientEvidenceRetriever(public, MockEvidenceRetriever())

    records = list(retriever.retrieve(_brief(), _seed(), 1))

    assert any(item.metadata.get("live") for item in records)
    counter = [item for item in records if item.supports == "counter"]
    assert len(counter) == 1
    assert counter[0].metadata["provenance_class"] == "fixture_fallback"
    assert counter[0].metadata["fallback_role"] == "counter_evidence_placeholder"


def test_resilient_retriever_falls_back_without_failing_the_workflow() -> None:
    public = PublicEvidenceRetriever(client=FakePublicClient(fail=True), timeout_seconds=1)
    retriever = ResilientEvidenceRetriever(public, MockEvidenceRetriever())

    records = list(retriever.retrieve(_brief(), _seed(), 1))

    assert {item.supports for item in records} == {"support", "counter"}
    assert all(item.metadata["provenance_class"] == "fixture_fallback" for item in records)
    assert all(item.metadata.get("fallback_reason") == "RuntimeError" for item in records)
