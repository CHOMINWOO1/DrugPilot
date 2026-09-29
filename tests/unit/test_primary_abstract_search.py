from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
from typing import Any

import pytest

from drugpilot.tools.primary_abstract_search import (
    PUBMED_FETCH_URL,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
    CandidateSearchAliases,
    search_primary_abstract_pairs,
)


ENDPOINT = "half-maximal inhibitory concentration [IC50]"
PAIRED_SENTENCE = (
    "Alphatinib was the most potent KIN2 inhibitor "
    "(half-maximal inhibitory concentration [IC50] at physiological [1mM] ATP, 2.9nM), "
    "followed by Betanib (17nM), Gammanib (29nM), and Deltanib (39nM), "
    "and the most selective."
)
TWO_CANDIDATE_SENTENCE = (
    "Alphatinib was the most potent KIN2 inhibitor "
    "(half-maximal inhibitory concentration [IC50] at physiological [1mM] ATP, 2.9nM), "
    "followed by Betanib (17nM)."
)
CANDIDATES = {
    "candidate-a": CandidateSearchAliases(
        preferred_search_alias="Alpha lead",
        exact_aliases=("Alpha lead", "Alphatinib"),
    ),
    "candidate-b": CandidateSearchAliases(
        preferred_search_alias="Beta lead",
        exact_aliases=("Beta lead", "Betanib"),
    ),
    "candidate-c": CandidateSearchAliases(
        preferred_search_alias="Gamma lead",
        exact_aliases=("Gamma lead", "Gammanib"),
    ),
    "candidate-d": CandidateSearchAliases(
        preferred_search_alias="Delta lead",
        exact_aliases=("Delta lead", "Deltanib"),
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


class FakeResponse:
    def __init__(
        self,
        *,
        payload: Any | None = None,
        text: str | None = None,
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        if text is not None:
            self.text = text
            self.content = text.encode("utf-8")
        elif payload is not None:
            self.content = _canonical_json(payload)
            self.text = self.content.decode("utf-8")
        else:
            self.content = b""
            self.text = ""

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeClient:
    def __init__(
        self,
        responses: dict[str, FakeResponse | Exception],
        *,
        max_attempts: int = 1,
    ) -> None:
        self.responses = responses
        self.max_attempts = max_attempts
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def _search(ids: list[str]) -> FakeResponse:
    return FakeResponse(payload={"esearchresult": {"idlist": ids}})


def _summary(
    ids: list[str],
    *,
    publication_types: list[str] | None = None,
    uids: list[str] | None = None,
) -> FakeResponse:
    pubtypes = publication_types or ["Journal Article"]
    result: dict[str, Any] = {"uids": list(ids if uids is None else uids)}
    for document_id in ids:
        result[document_id] = {
            "uid": document_id,
            "pubtype": list(pubtypes),
            "title": "This title must not be retained",
        }
    return FakeResponse(payload={"result": result})


def _article(document_id: str, sentence: str = PAIRED_SENTENCE) -> str:
    return (
        "<PubmedArticle><MedlineCitation>"
        f"<PMID>{document_id}</PMID><Article><Abstract>"
        "<AbstractText Label='Methods'>DO_NOT_STORE_FULL_ABSTRACT_MARKER</AbstractText>"
        f"<AbstractText Label='Results'>{sentence}</AbstractText>"
        "</Abstract></Article></MedlineCitation></PubmedArticle>"
    )


def _fetch(*articles: str) -> FakeResponse:
    return FakeResponse(text=f"<PubmedArticleSet>{''.join(articles)}</PubmedArticleSet>")


def _positive_client(
    *,
    ids: list[str] | None = None,
    publication_types: list[str] | None = None,
    sentence: str = PAIRED_SENTENCE,
) -> FakeClient:
    document_ids = ids or ["101"]
    unique_ids = list(dict.fromkeys(document_ids))
    return FakeClient(
        {
            PUBMED_SEARCH_URL: _search(document_ids),
            PUBMED_SUMMARY_URL: _summary(
                unique_ids, publication_types=publication_types
            ),
            PUBMED_FETCH_URL: _fetch(
                *(_article(document_id, sentence) for document_id in unique_ids)
            ),
        }
    )


def _run(client: FakeClient, **overrides: Any):
    arguments: dict[str, Any] = {
        "target_aliases": {"KIN2"},
        "endpoint": ENDPOINT,
        "candidates": CANDIDATES,
    }
    arguments.update(overrides)
    return search_primary_abstract_pairs(client, **arguments)


def test_generic_jak2_shaped_fixture_is_verified_with_three_bounded_calls() -> None:
    client = _positive_client()

    outcome = _run(client)

    assert outcome.status == "verified"
    assert outcome.reason_code is None
    assert outcome.schema_version == "primary_abstract_search/v1"
    assert outcome.requested_candidate_count == 4
    assert outcome.queried_candidate_count == 4
    assert outcome.requested_document_count == 8
    assert outcome.returned_document_count == 1
    assert outcome.unique_document_count == 1
    assert outcome.screened_document_count == 1
    assert outcome.screened_sentence_count == 1
    assert outcome.verified_candidate_ids == (
        "candidate-a",
        "candidate-b",
        "candidate-c",
        "candidate-d",
    )
    assert len(outcome.calls) == 3
    assert [call.stage for call in outcome.calls] == ["esearch", "esummary", "efetch"]
    assert all(call.status == "completed" for call in outcome.calls)
    assert all(call.status_code == 200 for call in outcome.calls)
    assert all(call.error_type is None for call in outcome.calls)

    expected_query = (
        '("KIN2"[Title/Abstract]) AND '
        '"half-maximal inhibitory concentration [IC50]"[Title/Abstract] AND '
        '("Alpha lead"[Title/Abstract] OR "Beta lead"[Title/Abstract] OR '
        '"Gamma lead"[Title/Abstract] OR "Delta lead"[Title/Abstract]) '
        'NOT Review[Publication Type]'
    )
    assert outcome.query == expected_query
    assert outcome.query_sha256 == hashlib.sha256(expected_query.encode()).hexdigest()
    assert "Alphatinib" not in expected_query
    assert all(
        preferred in expected_query
        for preferred in ("Alpha lead", "Beta lead", "Gamma lead", "Delta lead")
    )

    group = outcome.groups[0]
    assert group.document_id == "101"
    assert group.section == "Results"
    assert group.sentence == PAIRED_SENTENCE
    assert group.sentence_sha256 == hashlib.sha256(PAIRED_SENTENCE.encode()).hexdigest()
    assert [pair.value_raw for pair in group.pairs] == ["2.9", "17", "29", "39"]
    assert group.candidate_ids == outcome.verified_candidate_ids
    assert outcome.records[0].status == "verified"
    assert outcome.records[0].rejection_codes == ()
    assert outcome.records[0].publication_types == ("journal article",)

    with pytest.raises(FrozenInstanceError):
        outcome.status = "failed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        group.document_id = "999"  # type: ignore[misc]


def test_nlm_results_category_overrides_nonstandard_findings_label() -> None:
    article = _article("101").replace(
        "Label='Results'", "Label='FINDINGS' NlmCategory='RESULTS'"
    )
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"]),
            PUBMED_FETCH_URL: _fetch(article),
        }
    )

    outcome = _run(client)

    assert outcome.status == "verified"
    assert len(outcome.groups) == 1
    assert outcome.groups[0].section == "Results"


def test_call_parameter_and_response_hashes_are_exact_and_body_free() -> None:
    client = _positive_client()
    outcome = _run(client)

    for provenance, (endpoint, kwargs) in zip(outcome.calls, client.calls, strict=True):
        assert provenance.endpoint == endpoint
        expected_parameter_bytes = _canonical_json({"params": kwargs["params"]})
        assert provenance.parameter_sha256 == hashlib.sha256(
            expected_parameter_bytes
        ).hexdigest()
        response = client.responses[endpoint]
        assert isinstance(response, FakeResponse)
        assert provenance.response_sha256 == hashlib.sha256(response.content).hexdigest()
        assert provenance.parameter_keys == tuple(
            sorted(f"params.{key}" for key in kwargs["params"])
        )

    serialized = json.dumps(asdict(outcome), ensure_ascii=False, default=str)
    assert "DO_NOT_STORE_FULL_ABSTRACT_MARKER" not in serialized
    assert "This title must not be retained" not in serialized
    assert "PubmedArticleSet" not in serialized
    assert not hasattr(outcome, "raw_body")
    assert not hasattr(outcome.records[0], "abstract")


def test_query_and_hash_are_canonical_across_mapping_and_target_order() -> None:
    first = _run(
        FakeClient({PUBMED_SEARCH_URL: _search([])}),
        target_aliases=("KIN2", "Kinase Two"),
    )
    second = _run(
        FakeClient({PUBMED_SEARCH_URL: _search([])}),
        target_aliases=("Kinase Two", "KIN2"),
        candidates=dict(reversed(tuple(CANDIDATES.items()))),
    )

    assert first.status == second.status == "not_verified"
    assert first.query == second.query
    assert first.query_sha256 == second.query_sha256


def test_candidate_query_is_hard_capped_at_twelve_canonical_discovered_names() -> None:
    candidates = {
        f"candidate-{index:02d}": CandidateSearchAliases(
            preferred_search_alias=f"Lead {index:02d}",
            exact_aliases=(f"Lead {index:02d}",),
        )
        for index in range(14)
    }
    client = FakeClient({PUBMED_SEARCH_URL: _search([])})

    outcome = _run(client, candidates=candidates)

    assert outcome.status == "not_verified"
    assert outcome.requested_candidate_count == 14
    assert outcome.queried_candidate_count == 12
    assert outcome.query is not None
    for index in range(12):
        assert f'"Lead {index:02d}"[Title/Abstract]' in outcome.query
    assert "Lead 12" not in outcome.query
    assert "Lead 13" not in outcome.query
    assert len(client.calls) == 1
    assert client.calls[0][1]["params"]["retmax"] == 8


@pytest.mark.parametrize(
    "candidates,target_aliases,expected_reason",
    [
        (
            {
                "one": CandidateSearchAliases("Same", ("Same",)),
                "two": CandidateSearchAliases("same", ("same",)),
            },
            ("KIN2",),
            "alias_conflict",
        ),
        (
            {
                "one": CandidateSearchAliases("Preferred", ("Different",)),
            },
            ("KIN2",),
            "preferred_alias_not_exact",
        ),
        (
            {
                "one": CandidateSearchAliases("KIN2", ("KIN2",)),
            },
            ("KIN2",),
            "target_candidate_alias_conflict",
        ),
        (
            {
                "one": CandidateSearchAliases("Lead", ("Lead", "lead")),
            },
            ("KIN2",),
            "alias_conflict",
        ),
    ],
)
def test_alias_conflicts_fail_before_http(
    candidates: dict[str, CandidateSearchAliases],
    target_aliases: tuple[str, ...],
    expected_reason: str,
) -> None:
    client = FakeClient({})

    outcome = _run(client, candidates=candidates, target_aliases=target_aliases)

    assert outcome.status == "failed"
    assert outcome.reason_code == expected_reason
    assert outcome.query is None
    assert outcome.query_sha256 is None
    assert outcome.groups == outcome.records == outcome.calls == ()
    assert client.calls == []


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"target_aliases": ("KIN2", "kin2")}, "alias_conflict"),
        ({"endpoint": 'IC50" OR Review[pt]'}, "unsafe_query_term"),
        ({"max_candidates": 13}, "invalid_candidate_limit"),
        ({"max_documents": 9}, "invalid_document_limit"),
        ({"max_documents": 0}, "invalid_document_limit"),
        (
            {
                "candidates": {
                    "one": CandidateSearchAliases("Lead", ("Lead",)),
                    "two": CandidateSearchAliases("Bad\"Alias", ("Bad\"Alias",)),
                }
            },
            "unsafe_query_term",
        ),
    ],
)
def test_query_injection_and_limit_inputs_fail_without_http(
    overrides: dict[str, Any], reason: str
) -> None:
    client = FakeClient({})
    outcome = _run(client, **overrides)
    assert outcome.status == "failed"
    assert outcome.reason_code == reason
    assert client.calls == []


@pytest.mark.parametrize("failed_stage", ["esearch", "esummary", "efetch"])
def test_any_partial_http_status_failure_returns_no_partial_science(
    failed_stage: str,
) -> None:
    responses: dict[str, FakeResponse | Exception] = {
        PUBMED_SEARCH_URL: _search(["101"]),
        PUBMED_SUMMARY_URL: _summary(["101"]),
        PUBMED_FETCH_URL: _fetch(_article("101")),
    }
    endpoint_by_stage = {
        "esearch": PUBMED_SEARCH_URL,
        "esummary": PUBMED_SUMMARY_URL,
        "efetch": PUBMED_FETCH_URL,
    }
    responses[endpoint_by_stage[failed_stage]] = FakeResponse(
        payload={"error": "failure"}, status_code=503
    )
    client = FakeClient(responses)

    outcome = _run(client)

    expected_calls = {"esearch": 1, "esummary": 2, "efetch": 3}[failed_stage]
    assert outcome.status == "failed"
    assert outcome.reason_code == f"{failed_stage}_http_failure"
    assert len(outcome.calls) == expected_calls
    assert outcome.calls[-1].status == "failed"
    assert outcome.calls[-1].status_code == 503
    assert outcome.records == ()
    assert outcome.groups == ()
    assert outcome.screened_document_count == 0
    assert outcome.screened_sentence_count == 0


def test_transport_failure_has_parameter_provenance_but_no_response_hash() -> None:
    client = FakeClient({PUBMED_SEARCH_URL: TimeoutError("offline")})

    outcome = _run(client)

    assert outcome.status == "failed"
    assert outcome.reason_code == "esearch_transport_failure"
    assert len(outcome.calls) == 1
    call = outcome.calls[0]
    assert call.status == "failed"
    assert call.status_code is None
    assert call.response_sha256 is None
    assert call.error_type == "TimeoutError"
    assert len(call.parameter_sha256) == 64


def test_retrying_client_is_rejected_so_three_calls_mean_three_wire_attempts() -> None:
    client = FakeClient({}, max_attempts=2)

    outcome = _run(client)

    assert outcome.status == "failed"
    assert outcome.reason_code == "client_retry_policy_not_single_attempt"
    assert outcome.query is None
    assert outcome.calls == ()
    assert client.calls == []


def test_invalid_utf8_content_only_fetch_fails_closed_as_an_immutable_outcome() -> None:
    class ContentOnlyResponse:
        status_code = 200
        content = b"\xff\xfe"

    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"]),
            PUBMED_FETCH_URL: ContentOnlyResponse(),  # type: ignore[dict-item]
        }
    )

    outcome = _run(client)

    assert outcome.status == "failed"
    assert outcome.reason_code == "efetch_text_invalid"
    assert len(outcome.calls) == 3
    assert outcome.calls[-1].response_sha256 == hashlib.sha256(b"\xff\xfe").hexdigest()
    assert outcome.records == outcome.groups == ()


@pytest.mark.parametrize(
    "publication_types",
    [
        ["Review"],
        ["Journal Article", "Systematic Review"],
        ["Editorial"],
        ["Dataset"],
        ["Journal Article", "Retraction of Publication"],
        ["Journal Article", "Retraction Notice"],
        ["Journal Article", "Clinical Trial Protocol"],
        ["Journal Article", "Expression of Concern"],
    ],
)
def test_only_primary_non_review_publication_types_reach_efetch(
    publication_types: list[str],
) -> None:
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(
                ["101"], publication_types=publication_types
            ),
        }
    )

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "no_verified_group"
    assert len(client.calls) == 2
    assert outcome.groups == ()
    assert outcome.records[0].status == "rejected"
    assert outcome.records[0].rejection_codes == ("non_primary_or_review",)


def test_duplicate_esearch_ids_are_counted_then_deduplicated_before_batch_calls() -> None:
    client = _positive_client(ids=["101", "101"])

    outcome = _run(client)

    assert outcome.status == "verified"
    assert outcome.returned_document_count == 2
    assert outcome.unique_document_count == 1
    assert outcome.screened_document_count == 1
    assert client.calls[1][1]["params"]["id"] == "101"
    assert client.calls[2][1]["params"]["id"] == "101"
    assert len(outcome.groups) == 1


def test_duplicate_esummary_uids_are_payload_ambiguity_and_fail_closed() -> None:
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"], uids=["101", "101"]),
        }
    )

    outcome = _run(client)

    assert outcome.status == "failed"
    assert outcome.reason_code == "esummary_uid_duplicate"
    assert len(outcome.calls) == 2
    assert outcome.records == outcome.groups == ()


def test_raw_esearch_count_above_eight_is_not_truncated_or_paged() -> None:
    client = FakeClient(
        {PUBMED_SEARCH_URL: _search([str(index) for index in range(100, 109)])}
    )

    outcome = _run(client)

    assert outcome.status == "failed"
    assert outcome.reason_code == "esearch_document_bound_invalid"
    assert len(client.calls) == 1
    assert outcome.groups == ()


def test_unknown_efetch_identifier_fails_the_completed_batch() -> None:
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"]),
            PUBMED_FETCH_URL: _fetch(_article("999")),
        }
    )

    outcome = _run(client)

    assert outcome.status == "failed"
    assert outcome.reason_code == "efetch_id_unexpected"
    assert len(outcome.calls) == 3
    assert outcome.groups == outcome.records == ()


def test_duplicate_efetch_record_is_rejected_without_selecting_one_body() -> None:
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"]),
            PUBMED_FETCH_URL: _fetch(_article("101"), _article("101")),
        }
    )

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.groups == ()
    assert outcome.records[0].rejection_codes == ("duplicate_fetch_record",)


def test_two_distinct_candidates_in_one_valid_pair_sentence_are_not_verified() -> None:
    client = _positive_client(sentence=TWO_CANDIDATE_SENTENCE)

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "no_verified_group"
    assert outcome.groups == ()
    assert outcome.verified_candidate_ids == ()
    assert outcome.records[0].rejection_codes == (
        "insufficient_distinct_candidates",
    )


def test_missing_summary_record_has_per_record_rejection_while_other_record_verifies() -> None:
    summary = _summary(["101"])
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101", "102"]),
            PUBMED_SUMMARY_URL: summary,
            PUBMED_FETCH_URL: _fetch(_article("101")),
        }
    )

    outcome = _run(client)

    assert outcome.status == "verified"
    assert outcome.returned_document_count == 2
    assert outcome.unique_document_count == 2
    assert outcome.screened_document_count == 2
    records = {record.document_id: record for record in outcome.records}
    assert records["101"].status == "verified"
    assert records["102"].status == "rejected"
    assert records["102"].rejection_codes == ("summary_missing",)
    assert client.calls[2][1]["params"]["id"] == "101"


def test_long_sentences_are_counted_but_never_sent_to_parser_or_returned() -> None:
    long_sentence = "Alphatinib " + ("x" * 370) + "."
    client = _positive_client(sentence=long_sentence)

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.screened_sentence_count == 0
    assert outcome.groups == ()
    assert outcome.records[0].sentence_count == 1
    assert outcome.records[0].bounded_sentence_count == 0
    assert outcome.records[0].rejection_codes == ("no_bounded_sentence",)
    assert long_sentence not in repr(outcome)
