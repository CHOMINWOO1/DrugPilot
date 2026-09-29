from __future__ import annotations

from typing import Any, Callable, Mapping

import pytest
from pydantic import ValidationError

from drugpilot.tools.disease_targets import (
    DiseaseTargetDiscovery,
    OPEN_TARGETS_GRAPHQL_URL,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
    SourceStatus,
)


class FakeResponse:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Mapping[str, Any]:
        return self.payload


class FakeClient:
    def __init__(
        self,
        *,
        post_handler: Callable[[str, dict[str, Any]], Mapping[str, Any]],
        get_handler: Callable[[str, dict[str, Any]], Mapping[str, Any]],
    ) -> None:
        self.post_handler = post_handler
        self.get_handler = get_handler
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return FakeResponse(self.post_handler(url, kwargs))

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return FakeResponse(self.get_handler(url, kwargs))


def _disease_search_payload() -> Mapping[str, Any]:
    return {
        "data": {
            "search": {
                "hits": [
                    {
                        "id": "MONDO_0008903",
                        "name": "lung adenocarcinoma",
                        "entity": "disease",
                        "description": "A lung carcinoma subtype.",
                    }
                ]
            }
        }
    }


def _association_payload(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "data": {
            "disease": {
                "id": "MONDO_0008903",
                "name": "lung adenocarcinoma",
                "associatedTargets": {"count": len(rows), "rows": rows},
            }
        }
    }


def _row(
    ensembl_id: str,
    symbol: str,
    score: float,
    *,
    datatype_scores: list[Mapping[str, Any]] | None = None,
    tractability: list[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    return {
        "score": score,
        "datatypeScores": datatype_scores or [],
        "target": {
            "id": ensembl_id,
            "approvedSymbol": symbol,
            "approvedName": f"{symbol} protein",
            "biotype": "protein_coding",
            "tractability": tractability or [],
        },
    }


def _empty_pubmed_get(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
    del kwargs
    if url == PUBMED_SEARCH_URL:
        return {"esearchresult": {"idlist": []}}
    raise AssertionError(f"unexpected GET {url}")


def test_associated_target_operation_does_not_count_embedded_disease_as_a_row() -> None:
    row = _row("ENSG00000146648", "EGFR", 0.93)
    client = FakeClient(
        post_handler=lambda _url, kwargs: (
            _disease_search_payload()
            if "SearchDisease" in kwargs["json"]["query"]
            else _association_payload([row])
        ),
        get_handler=_empty_pubmed_get,
    )

    result = DiseaseTargetDiscovery(client=client, max_attempts=1).discover(
        "lung adenocarcinoma", allowed_sources=["Open Targets"]
    )

    status = next(item for item in result.source_statuses if item.source == "Open Targets")
    metric = next(item for item in status.operation_metrics if item.operation == "associated_targets")
    assert metric.operation_records_observed == 1
    assert metric.unique_records_observed == 1
    assert status.unique_records_observed <= status.operation_records_observed


def test_discovers_ranked_targets_and_traceable_pubmed_supplement() -> None:
    rows = [
        _row(
            "ENSG00000146648",
            "EGFR",
            0.93,
            datatype_scores=[
                {"id": "genetic_association", "score": 0.81},
                {"id": "known_drug", "score": 0.76},
            ],
            tractability=[
                {"label": "Approved Drug", "modality": "SM", "value": True}
            ],
        ),
        _row(
            "ENSG00000141510",
            "TP53",
            0.71,
            datatype_scores=[{"id": "literature", "score": 0.64}],
        ),
    ]

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        assert url == OPEN_TARGETS_GRAPHQL_URL
        assert kwargs["timeout"] == 2.5
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            return _disease_search_payload()
        if "DiseaseAssociatedTargets" in query:
            assert kwargs["json"]["variables"]["size"] == 3
            return _association_payload(rows)
        if "SearchTarget" in query:
            assert kwargs["json"]["variables"]["query"] == "ABCD1"
            return {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": "ENSG00000101986",
                                "name": "ABCD1",
                                "entity": "target",
                                "description": "ATP binding cassette subfamily D member 1",
                            }
                        ]
                    }
                }
            }
        raise AssertionError("unexpected GraphQL operation")

    def get_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        assert kwargs["timeout"] == 2.5
        if url == PUBMED_SEARCH_URL:
            assert "lung adenocarcinoma" in kwargs["params"]["term"]
            return {"esearchresult": {"idlist": ["101", "102"]}}
        if url == PUBMED_SUMMARY_URL:
            return {
                "result": {
                    "uids": ["101", "102"],
                    "101": {
                        "uid": "101",
                        "title": "EGFR signaling as a target in lung adenocarcinoma",
                        "pubdate": "2025",
                        "fulljournalname": "Example Journal",
                        "authors": [{"name": "A. Author"}],
                        "pubtype": ["Journal Article"],
                    },
                    "102": {
                        "uid": "102",
                        "title": "ABCD1 is a novel gene in lung adenocarcinoma",
                        "pubdate": "2024",
                    },
                }
            }
        raise AssertionError(f"unexpected GET {url}")

    client = FakeClient(post_handler=post_handler, get_handler=get_handler)
    discovery = DiseaseTargetDiscovery(
        client=client,
        timeout_seconds=2.5,
        max_targets=3,
        max_attempts=1,
    )

    result = discovery.discover("lung adenocarcinoma")

    assert result.resolved_disease is not None
    assert result.resolved_disease.disease_id == "MONDO_0008903"
    assert [candidate.gene_symbol for candidate in result.candidates] == [
        "EGFR",
        "TP53",
        "ABCD1",
    ]
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]

    egfr, tp53, abcd1 = result.candidates
    assert egfr.association_score == 0.93
    assert egfr.evidence_summary.known_drug_score == 0.76
    assert egfr.evidence_summary.tractability[0].value is True
    assert egfr.evidence_summary.pubmed_record_count == 1
    assert egfr.track_hint == "tbio"
    assert "genetic_association" in (egfr.track_hint_basis or "")

    assert tp53.track_hint == "literature"
    assert abcd1.association_score is None
    assert abcd1.track_hint == "literature"
    assert abcd1.provenance.discovery_method == "pubmed_explicit_symbol_co_mention"
    assert abcd1.provenance.causality_interpretation == (
        "co_mention_only_no_causality_inferred"
    )
    assert abcd1.source_records[0].relationship == "identity_mapping_only"
    assert abcd1.source_records[1].relationship == "co_mention_context_only"
    assert result.fallback_used is False
    assert {status.source: status.status for status in result.source_statuses} == {
        "Open Targets": "live",
        "PubMed": "live",
    }
    statuses = {status.source: status for status in result.source_statuses}
    pubmed_status = statuses["PubMed"]
    assert pubmed_status.records_observed == 4
    assert pubmed_status.operation_records_observed == 4
    assert pubmed_status.unique_records_observed == 2
    assert pubmed_status.records_screened == 2
    assert [
        (
            metric.operation,
            metric.operation_records_observed,
            metric.unique_records_observed,
            metric.records_screened,
        )
        for metric in pubmed_status.operation_metrics
    ] == [
        ("search_publications", 2, 2, 0),
        ("publication_metadata", 2, 2, 2),
    ]
    open_targets_status = statuses["Open Targets"]
    assert open_targets_status.records_observed == 4
    assert open_targets_status.operation_records_observed == 4
    assert open_targets_status.unique_records_observed == 4
    assert open_targets_status.records_screened == 4


def test_provided_disease_id_is_validated_without_name_search() -> None:
    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        assert url == OPEN_TARGETS_GRAPHQL_URL
        query = kwargs["json"]["query"]
        if "ResolveDiseaseId" in query:
            assert kwargs["json"]["variables"] == {"diseaseId": "EFO_0000305"}
            return {
                "data": {
                    "disease": {
                        "id": "EFO_0000305",
                        "name": "breast carcinoma",
                        "description": None,
                    }
                }
            }
        if "DiseaseAssociatedTargets" in query:
            return {
                "data": {
                    "disease": {
                        "id": "EFO_0000305",
                        "name": "breast carcinoma",
                        "associatedTargets": {"count": 0, "rows": []},
                    }
                }
            }
        raise AssertionError("disease-name search should not be called")

    client = FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get)
    result = DiseaseTargetDiscovery(client=client, max_attempts=1).discover(
        "breast cancer", " EFO_0000305 "
    )

    assert result.requested_disease_id == "EFO_0000305"
    assert result.resolved_disease is not None
    assert result.resolved_disease.match_method == "provided_id"
    assert result.resolved_disease.disease_name == "breast carcinoma"
    assert result.association_total == 0
    assert result.candidates == ()
    assert all(
        "SearchDisease" not in call[2].get("json", {}).get("query", "")
        for call in client.calls
    )


def test_exact_open_targets_synonym_resolves_tnbc_and_is_cached() -> None:
    query_disease = "triple-negative breast cancer"
    disease_id = "MONDO_0005494"

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        assert url == OPEN_TARGETS_GRAPHQL_URL
        query = kwargs["json"]["query"]
        variables = kwargs["json"]["variables"]
        if "SearchDisease" in query:
            assert variables == {"query": query_disease, "size": 5}
            return {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": disease_id,
                                "name": "triple-negative breast carcinoma",
                                "entity": "disease",
                                "description": "A breast carcinoma subtype.",
                            },
                            {
                                "id": "MONDO_0007254",
                                "name": "breast carcinoma",
                                "entity": "disease",
                                "description": "A broader breast carcinoma term.",
                            },
                        ]
                    }
                }
            }
        if "ResolveDiseaseSynonyms" in query:
            assert variables == {
                "diseaseIds": [disease_id, "MONDO_0007254"]
            }
            assert "synonyms { relation terms }" in query
            return {
                "data": {
                    "diseases": [
                        {
                            "id": disease_id,
                            "name": "triple-negative breast carcinoma",
                            "description": "A breast carcinoma subtype.",
                            "synonyms": [
                                {
                                    "relation": "hasExactSynonym",
                                    "terms": [query_disease, "TNBC"],
                                }
                            ],
                        },
                        {
                            "id": "MONDO_0007254",
                            "name": "breast carcinoma",
                            "description": "A broader breast carcinoma term.",
                            "synonyms": [
                                {
                                    "relation": "hasRelatedSynonym",
                                    "terms": [query_disease],
                                }
                            ],
                        },
                    ]
                }
            }
        if "DiseaseAssociatedTargets" in query:
            assert variables["diseaseId"] == disease_id
            return {
                "data": {
                    "disease": {
                        "id": disease_id,
                        "name": "triple-negative breast carcinoma",
                        "associatedTargets": {"count": 0, "rows": []},
                    }
                }
            }
        raise AssertionError("unexpected GraphQL operation")

    client = FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get)
    discovery = DiseaseTargetDiscovery(client=client, max_attempts=1)

    first = discovery.discover(query_disease)
    call_count = len(client.calls)
    second = discovery.discover(query_disease)

    assert first.resolved_disease is not None
    assert first.resolved_disease.disease_id == disease_id
    assert first.resolved_disease.disease_name == "triple-negative breast carcinoma"
    assert first.resolved_disease.match_method == "exact_synonym"
    assert second.resolved_disease is not None
    assert second.resolved_disease.match_method == "exact_synonym"
    assert second.resolved_disease.retrieval_mode == "cache"
    assert len(client.calls) == call_count
    open_targets = next(
        status for status in second.source_statuses if status.source == "Open Targets"
    )
    assert open_targets.status == "cache"
    assert open_targets.operations == (
        "search_disease",
        "verify_exact_disease_synonyms",
        "associated_targets",
    )


def test_case_and_hyphen_variation_resolves_against_normalized_exact_synonym() -> None:
    query_disease = "Triple Negative Breast Cancer"
    disease_id = "MONDO_0005494"

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        assert url == OPEN_TARGETS_GRAPHQL_URL
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            assert kwargs["json"]["variables"] == {"query": query_disease, "size": 5}
            return {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": disease_id,
                                "name": "triple-negative breast carcinoma",
                                "entity": "disease",
                                "description": "A breast carcinoma subtype.",
                            }
                        ]
                    }
                }
            }
        if "ResolveDiseaseSynonyms" in query:
            return {
                "data": {
                    "diseases": [
                        {
                            "id": disease_id,
                            "name": "triple-negative breast carcinoma",
                            "description": "A breast carcinoma subtype.",
                            "synonyms": [
                                {
                                    "relation": "hasExactSynonym",
                                    "terms": ["triple-negative breast cancer", "TNBC"],
                                }
                            ],
                        }
                    ]
                }
            }
        if "DiseaseAssociatedTargets" in query:
            return {
                "data": {
                    "disease": {
                        "id": disease_id,
                        "name": "triple-negative breast carcinoma",
                        "associatedTargets": {"count": 0, "rows": []},
                    }
                }
            }
        raise AssertionError("unexpected GraphQL operation")

    result = DiseaseTargetDiscovery(
        client=FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get),
        max_attempts=1,
    ).discover(query_disease)

    assert result.resolved_disease is not None
    assert result.resolved_disease.match_method == "exact_synonym"
    provenance = result.resolved_disease.match_provenance
    assert provenance is not None
    assert provenance.normalized_query == "triple negative breast cancer"
    assert provenance.normalized_matched_label == "triple negative breast cancer"
    assert provenance.similarity_method == "normalized_exact"
    assert provenance.similarity_score == 1.0
    assert result.resolved_disease.warnings == ()


def test_unique_high_similarity_official_synonym_corrects_bounded_typo() -> None:
    query_disease = "Tirple Negetive Breast Cancer"
    disease_id = "MONDO_0005494"

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        assert url == OPEN_TARGETS_GRAPHQL_URL
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            search_query = kwargs["json"]["variables"]["query"]
            if search_query == query_disease:
                assert kwargs["json"]["variables"]["size"] == 5
                return {"data": {"search": {"hits": []}}}
            assert search_query in {
                "tirple negetive",
                "negetive breast",
                "breast cancer",
            }
            assert kwargs["json"]["variables"]["size"] == 20
            if search_query != "breast cancer":
                return {"data": {"search": {"hits": []}}}
            return {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": disease_id,
                                "name": "triple-negative breast carcinoma",
                                "entity": "disease",
                                "description": "A breast carcinoma subtype.",
                            },
                            {
                                "id": "MONDO_0007254",
                                "name": "breast carcinoma",
                                "entity": "disease",
                                "description": "A broader breast carcinoma term.",
                            },
                        ]
                    }
                }
            }
        if "ResolveDiseaseSynonyms" in query:
            return {
                "data": {
                    "diseases": [
                        {
                            "id": disease_id,
                            "name": "triple-negative breast carcinoma",
                            "description": "A breast carcinoma subtype.",
                            "synonyms": [
                                {
                                    "relation": "hasExactSynonym",
                                    "terms": ["triple-negative breast cancer", "TNBC"],
                                }
                            ],
                        },
                        {
                            "id": "MONDO_0007254",
                            "name": "breast carcinoma",
                            "description": "A broader breast carcinoma term.",
                            "synonyms": [
                                {
                                    "relation": "hasExactSynonym",
                                    "terms": ["breast cancer"],
                                }
                            ],
                        },
                    ]
                }
            }
        if "DiseaseAssociatedTargets" in query:
            assert kwargs["json"]["variables"]["diseaseId"] == disease_id
            return {
                "data": {
                    "disease": {
                        "id": disease_id,
                        "name": "triple-negative breast carcinoma",
                        "associatedTargets": {"count": 0, "rows": []},
                    }
                }
            }
        raise AssertionError("unexpected GraphQL operation")

    result = DiseaseTargetDiscovery(
        client=FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get),
        max_attempts=1,
    ).discover(query_disease)

    assert result.resolved_disease is not None
    assert result.resolved_disease.disease_id == disease_id
    assert result.resolved_disease.match_method == "unique_high_similarity"
    provenance = result.resolved_disease.match_provenance
    assert provenance is not None
    assert provenance.matched_label == "triple-negative breast cancer"
    assert provenance.matched_label_kind == "exact_synonym"
    assert provenance.similarity_method == "sequence_matcher_token_guard_v1"
    assert provenance.similarity_score >= 0.88
    assert provenance.runner_up_similarity_score is not None
    assert (
        provenance.similarity_score - provenance.runner_up_similarity_score
        >= 0.08
    )
    assert result.resolved_disease.warnings


def test_high_similarity_typo_fails_closed_when_official_results_are_ambiguous() -> None:
    query_disease = "Tirple Negetive Breast Cancer"

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        assert url == OPEN_TARGETS_GRAPHQL_URL
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            return {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": "MONDO_A",
                                "name": "triple-negative breast carcinoma",
                                "entity": "disease",
                            },
                            {
                                "id": "MONDO_B",
                                "name": "triple-negative breast cancer syndrome",
                                "entity": "disease",
                            },
                        ]
                    }
                }
            }
        if "ResolveDiseaseSynonyms" in query:
            return {
                "data": {
                    "diseases": [
                        {
                            "id": "MONDO_A",
                            "name": "triple-negative breast carcinoma",
                            "synonyms": [
                                {
                                    "relation": "hasExactSynonym",
                                    "terms": ["triple-negative breast cancer"],
                                }
                            ],
                        },
                        {
                            "id": "MONDO_B",
                            "name": "triple-negative breast cancer syndrome",
                            "synonyms": [
                                {
                                    "relation": "hasExactSynonym",
                                    "terms": ["triple negative breast cancers"],
                                }
                            ],
                        },
                    ]
                }
            }
        raise AssertionError("candidate retrieval must not run for an ambiguous disease")

    result = DiseaseTargetDiscovery(
        client=FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get),
        max_attempts=1,
    ).discover(query_disease)

    assert result.resolved_disease is None
    assert result.candidates == ()
    assert result.association_total is None


def test_arbitrary_top_search_hit_does_not_resolve_without_exact_synonym() -> None:
    association_requested = False

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        nonlocal association_requested
        assert url == OPEN_TARGETS_GRAPHQL_URL
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            assert kwargs["json"]["variables"]["query"] == "asdf"
            return {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": "EFO_0000222",
                                "name": "acute myeloid leukemia",
                                "entity": "disease",
                                "description": "A real disease but not the query.",
                            }
                        ]
                    }
                }
            }
        if "ResolveDiseaseSynonyms" in query:
            return {
                "data": {
                    "diseases": [
                        {
                            "id": "EFO_0000222",
                            "name": "acute myeloid leukemia",
                            "description": "A real disease but not the query.",
                            "synonyms": [
                                {
                                    "relation": "hasRelatedSynonym",
                                    "terms": ["asdf"],
                                },
                                {
                                    "relation": "hasExactSynonym",
                                    "terms": ["acute myelogenous leukemia"],
                                },
                            ],
                        }
                    ]
                }
            }
        if "DiseaseAssociatedTargets" in query:
            association_requested = True
        raise AssertionError("unexpected GraphQL operation")

    client = FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get)
    result = DiseaseTargetDiscovery(client=client, max_attempts=1).discover("asdf")

    assert result.resolved_disease is None
    assert result.candidates == ()
    assert result.association_total is None
    assert association_requested is False
    open_targets = next(
        status for status in result.source_statuses if status.source == "Open Targets"
    )
    assert open_targets.status == "live"
    assert open_targets.operations == (
        "search_disease",
        "verify_exact_disease_synonyms",
    )


def test_pubmed_failure_is_isolated_from_open_targets_results() -> None:
    class PubMedFailingClient(FakeClient):
        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            self.calls.append(("GET", url, kwargs))
            raise TimeoutError("PubMed unavailable")

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        del url
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            return _disease_search_payload()
        if "DiseaseAssociatedTargets" in query:
            return _association_payload(
                [_row("ENSG00000146648", "EGFR", 0.9)]
            )
        raise AssertionError("unexpected GraphQL operation")

    client = PubMedFailingClient(
        post_handler=post_handler,
        get_handler=_empty_pubmed_get,
    )
    result = DiseaseTargetDiscovery(client=client, max_attempts=1).discover(
        "lung adenocarcinoma"
    )

    assert [candidate.gene_symbol for candidate in result.candidates] == ["EGFR"]
    statuses = {item.source: item for item in result.source_statuses}
    assert statuses["Open Targets"].status == "live"
    assert statuses["PubMed"].status == "failed"
    assert statuses["PubMed"].failure_codes == ("TimeoutError",)
    assert result.fallback_used is False


def test_pubmed_co_mention_can_survive_association_failure_but_stays_unscored() -> None:
    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        del url
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            return _disease_search_payload()
        if "DiseaseAssociatedTargets" in query:
            raise TimeoutError("association endpoint timeout")
        if "SearchTarget" in query:
            return {
                "data": {
                    "search": {
                        "hits": [
                            {
                                "id": "ENSG00000101986",
                                "name": "ABCD1",
                                "entity": "target",
                                "description": "ABCD1 protein",
                            }
                        ]
                    }
                }
            }
        raise AssertionError("unexpected GraphQL operation")

    def get_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        del kwargs
        if url == PUBMED_SEARCH_URL:
            return {"esearchresult": {"idlist": ["201"]}}
        if url == PUBMED_SUMMARY_URL:
            return {
                "result": {
                    "uids": ["201"],
                    "201": {
                        "uid": "201",
                        "title": "ABCD1 as a target in lung adenocarcinoma",
                    },
                }
            }
        raise AssertionError(f"unexpected GET {url}")

    client = FakeClient(post_handler=post_handler, get_handler=get_handler)
    result = DiseaseTargetDiscovery(client=client, max_targets=3, max_attempts=1).discover(
        "lung adenocarcinoma"
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.gene_symbol == "ABCD1"
    assert candidate.ensembl_target_id == "ENSG00000101986"
    assert candidate.association_score is None
    assert candidate.evidence_summary.known_drug_score is None
    assert candidate.provenance.primary_source == "PubMed"
    assert candidate.track_hint == "literature"
    statuses = {item.source: item for item in result.source_statuses}
    assert statuses["Open Targets"].status == "partial"
    assert statuses["Open Targets"].failure_codes == ("TimeoutError",)
    assert statuses["PubMed"].status == "live"


def test_operation_cache_is_reused_and_reported() -> None:
    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        del url
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            return _disease_search_payload()
        if "DiseaseAssociatedTargets" in query:
            return _association_payload(
                [_row("ENSG00000146648", "EGFR", 0.9)]
            )
        raise AssertionError("unexpected GraphQL operation")

    client = FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get)
    discovery = DiseaseTargetDiscovery(client=client, max_attempts=1)

    first = discovery.discover("lung adenocarcinoma")
    call_count = len(client.calls)
    second = discovery.discover("lung adenocarcinoma")

    assert len(client.calls) == call_count
    assert first.candidates[0].source_records[0].retrieval_mode == "live"
    assert second.resolved_disease is not None
    assert second.resolved_disease.retrieval_mode == "cache"
    assert second.candidates[0].source_records[0].retrieval_mode == "cache"
    statuses = {item.source: item for item in second.source_statuses}
    assert statuses["Open Targets"].status == "cache"
    assert statuses["PubMed"].status == "cache"


def test_ranking_is_deterministic_deduplicated_and_bounded() -> None:
    rows = [
        _row("ENSG00000000002", "BETA2", 0.7),
        _row("ENSG00000000001", "ALPHA1", 0.7),
        _row("ENSG00000000003", "GAMMA3", 0.4),
        _row("ENSG00000000003", "GAMMA3", 0.8),
    ]

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        del url
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            return _disease_search_payload()
        if "DiseaseAssociatedTargets" in query:
            return _association_payload(rows)
        raise AssertionError("unexpected GraphQL operation")

    client = FakeClient(post_handler=post_handler, get_handler=_empty_pubmed_get)
    result = DiseaseTargetDiscovery(client=client, max_attempts=1).discover(
        "lung adenocarcinoma", max_targets=2
    )

    assert [(item.gene_symbol, item.association_score) for item in result.candidates] == [
        ("GAMMA3", 0.8),
        ("ALPHA1", 0.7),
    ]
    assert len({item.ensembl_target_id for item in result.candidates}) == 2
    with pytest.raises(ValueError, match="max_targets"):
        DiseaseTargetDiscovery(client=client).discover("lung adenocarcinoma", max_targets=0)


def test_generic_acronyms_do_not_create_untraceable_candidates() -> None:
    target_search_calls = 0

    def post_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        nonlocal target_search_calls
        del url
        query = kwargs["json"]["query"]
        if "SearchDisease" in query:
            return _disease_search_payload()
        if "DiseaseAssociatedTargets" in query:
            return _association_payload([])
        if "SearchTarget" in query:
            target_search_calls += 1
            return {"data": {"search": {"hits": []}}}
        raise AssertionError("unexpected GraphQL operation")

    def get_handler(url: str, kwargs: dict[str, Any]) -> Mapping[str, Any]:
        del kwargs
        if url == PUBMED_SEARCH_URL:
            return {"esearchresult": {"idlist": ["301"]}}
        if url == PUBMED_SUMMARY_URL:
            return {
                "result": {
                    "uids": ["301"],
                    "301": {
                        "uid": "301",
                        "title": "DNA and RNA analysis of human cancer targets",
                    },
                }
            }
        raise AssertionError(f"unexpected GET {url}")

    client = FakeClient(post_handler=post_handler, get_handler=get_handler)
    result = DiseaseTargetDiscovery(client=client, max_attempts=1).discover(
        "lung adenocarcinoma"
    )

    assert result.candidates == ()
    assert target_search_calls == 0


def test_output_models_forbid_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        SourceStatus.model_validate(
            {
                "source": "PubMed",
                "status": "live",
                "operations": (),
                "records_observed": 0,
                "cache_hits": 0,
                "unexpected": True,
            }
        )
