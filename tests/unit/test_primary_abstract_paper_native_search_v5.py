from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from drugpilot.tools.primary_abstract_paper_native_search_v5 import (
    PAPER_NATIVE_V5_POLICY_TOKEN,
    PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION,
    PUBMED_FETCH_URL,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
    CandidateSearchAliases,
    search_paper_native_primary_abstract_v5,
)


SENTENCE = (
    "Alpha was the most potent GENE inhibitor "
    "(half-maximal inhibitory concentration [IC50] at physiological [1mM] ATP, 2.9nM), "
    "followed by Beta (17nM), Gamma (29nM), and Delta (39nM), "
    "and the most selective."
)
CANDIDATES = {
    "candidate-alpha": CandidateSearchAliases("Alpha", ("Alpha", "Alpha code")),
    "candidate-beta": CandidateSearchAliases("Beta", ("Beta", "Beta code")),
    "candidate-gamma": CandidateSearchAliases("Gamma", ("Gamma", "Gamma code")),
    "candidate-delta": CandidateSearchAliases("Delta", ("Delta", "Delta code")),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


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
            self.content = text.encode()
        elif payload is not None:
            self.content = _canonical_json(payload)
            self.text = self.content.decode()
        else:
            self.text = ""
            self.content = b""

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not json")
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


def _search(
    ids: list[str],
    *,
    total: int | None = None,
    warninglist: Any | None = None,
) -> FakeResponse:
    result: dict[str, Any] = {
        "count": str(len(ids) if total is None else total),
        "idlist": ids,
        "querytranslation": "translated bounded query",
    }
    if warninglist is not None:
        result["warninglist"] = warninglist
    return FakeResponse(payload={"esearchresult": result})


def _summary(
    ids: list[str], *, publication_types: list[str] | None = None
) -> FakeResponse:
    result: dict[str, Any] = {"uids": ids}
    for document_id in ids:
        result[document_id] = {
            "uid": document_id,
            "pubtype": publication_types or ["Journal Article", "Comparative Study"],
            "title": "NEVER_RETAIN_THIS_TITLE",
        }
    return FakeResponse(payload={"result": result})


def _article(
    document_id: str,
    sentence: str = SENTENCE,
    *,
    publication_type: str = "Journal Article",
    prefix_sentence: str | None = None,
) -> str:
    text = f"{prefix_sentence} {sentence}" if prefix_sentence else sentence
    return (
        "<PubmedArticle><MedlineCitation>"
        f"<PMID>{document_id}</PMID><Article>"
        "<ArticleTitle>Experimental inhibitor comparison.</ArticleTitle>"
        "<PublicationTypeList>"
        f"<PublicationType>{publication_type}</PublicationType>"
        "</PublicationTypeList><Abstract>"
        "<AbstractText Label='Methods'>Inhibitory potency was measured using biochemical assays.</AbstractText>"
        "<AbstractText Label='Methods'>NEVER_RETAIN_FULL_ABSTRACT</AbstractText>"
        f"<AbstractText Label='Results'>{text}</AbstractText>"
        "</Abstract></Article></MedlineCitation></PubmedArticle>"
    )


def _fetch(*articles: str) -> FakeResponse:
    return FakeResponse(text=f"<PubmedArticleSet>{''.join(articles)}</PubmedArticleSet>")


def _client(
    *,
    ids: list[str] | None = None,
    total: int | None = None,
    sentence: str = SENTENCE,
) -> FakeClient:
    document_ids = ids or ["101"]
    unique = list(dict.fromkeys(document_ids))
    return FakeClient(
        {
            PUBMED_SEARCH_URL: _search(document_ids, total=total),
            PUBMED_SUMMARY_URL: _summary(unique),
            PUBMED_FETCH_URL: _fetch(
                *(_article(document_id, sentence) for document_id in unique)
            ),
        }
    )


def _run(client: FakeClient, **overrides: Any):
    arguments: dict[str, Any] = {
        "target_aliases": ("GENE",),
        "endpoint": "IC50",
        "candidates": CANDIDATES,
    }
    arguments.update(overrides)
    return search_paper_native_primary_abstract_v5(client, **arguments)


def test_verified_joint_search_is_three_single_attempt_calls_and_fully_hashed() -> None:
    client = _client(total=29)
    outcome = _run(client)

    assert outcome.status == "verified"
    assert outcome.reason_code is None
    assert outcome.schema_version == PAPER_NATIVE_V5_SEARCH_SCHEMA_VERSION
    assert outcome.policy_token == PAPER_NATIVE_V5_POLICY_TOKEN
    assert outcome.search_sort == "pub date"
    assert outcome.total_result_count == 29
    assert outcome.returned_document_count == 1
    assert outcome.unsearched_result_count == 28
    assert outcome.returned_document_ids == ("101",)
    assert len(outcome.returned_document_ids_sha256 or "") == 64
    assert outcome.query_translation == "translated bounded query"
    assert len(outcome.query_translation_sha256 or "") == 64
    assert [call.stage for call in outcome.calls] == ["esearch", "esummary", "efetch"]
    assert all(call.status == "completed" and len(call.call_sha256) == 64 for call in outcome.calls)
    assert client.calls[0][1]["params"]["sort"] == "pub date"
    assert client.calls[0][1]["params"]["retmax"] == 8

    expected_query = (
        '("GENE"[Title/Abstract]) AND "IC50"[Title/Abstract] AND '
        '("Alpha"[Title/Abstract] OR "Beta"[Title/Abstract] OR '
        '"Delta"[Title/Abstract] OR "Gamma"[Title/Abstract])'
    )
    assert outcome.query == expected_query
    assert "Publication Type" not in outcome.query
    assert outcome.query_sha256 == hashlib.sha256(expected_query.encode()).hexdigest()
    for forbidden in (
        "Alpha code",
        "Beta code",
        "Gamma code",
        "Delta code",
        "2.9",
        "17",
        "29",
        "39",
        "nM",
        "mM",
        "PMID",
    ):
        assert forbidden not in outcome.query

    assert len(outcome.groups) == 1
    group = outcome.groups[0]
    assert outcome.selected_group_sha256 == group.group_sha256
    assert group.document_rank == 0
    assert group.sentence_index == 0
    assert group.candidate_ids == (
        "candidate-alpha",
        "candidate-beta",
        "candidate-gamma",
        "candidate-delta",
    )
    assert len(group.pairs) == 4
    assert group.summary_publication_types == ("comparative study", "journal article")
    assert group.fetch_publication_types == ("journal article",)
    primary_study = group.primary_study_attestation
    assert primary_study.status == "verified"
    assert primary_study.article_title == "Experimental inhibitor comparison."
    assert primary_study.method_sentence == (
        "Inhibitory potency was measured using biochemical assays."
    )
    assert primary_study.method_indicator == (
        "passive_experimental_measurement_method"
    )
    assert primary_study.schema_version == "pubmed_primary_study_attestation/v2"
    assert primary_study.retraction_safety_policy == (
        "pubmed_types_and_comments_corrections_fail_closed_v1"
    )
    assert primary_study.retraction_safe is True
    assert primary_study.correction_safe is True
    assert primary_study.linked_status_notice == "none_detected"
    assert primary_study.linked_status_notice_types == ()
    assert primary_study.linked_status_notice_pmids == ()
    assert len(primary_study.attestation_sha256) == 64
    assert len(group.group_sha256) == 64
    assert len(outcome.records[0].record_sha256) == 64

    payload = asdict(outcome)
    actual_hash = payload.pop("outcome_sha256")
    assert actual_hash == hashlib.sha256(_canonical_json(payload)).hexdigest()
    serialized = json.dumps(asdict(outcome), default=str)
    assert "NEVER_RETAIN_FULL_ABSTRACT" not in serialized
    assert "NEVER_RETAIN_THIS_TITLE" not in serialized

    with pytest.raises(FrozenInstanceError):
        outcome.total_result_count = 0  # type: ignore[misc]


def test_quantitative_group_without_positive_method_sentence_is_not_verified() -> None:
    article = _article("101").replace(
        "Inhibitory potency was measured using biochemical assays.",
        "Published evidence was summarized without new experiments.",
    )
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"]),
            PUBMED_FETCH_URL: _fetch(article),
        }
    )

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.groups == ()
    assert "primary_study_attestation_missing" in outcome.records[0].rejection_codes


@pytest.mark.parametrize(
    "notice_type",
    ["RetractionIn", "ExpressionOfConcernIn", "ErratumIn"],
)
def test_linked_retraction_expression_or_correction_fails_closed(
    notice_type: str,
) -> None:
    article = _article("101").replace(
        "</PubmedArticle>",
        (
            "<PubmedData><CommentsCorrectionsList>"
            f"<CommentsCorrections RefType='{notice_type}'><PMID>202</PMID>"
            "</CommentsCorrections></CommentsCorrectionsList></PubmedData>"
            "</PubmedArticle>"
        ),
    )
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"]),
            PUBMED_FETCH_URL: _fetch(article),
        }
    )

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.groups == ()
    assert "linked_status_notice_not_safe" in outcome.records[0].rejection_codes


def test_returned_document_and_sentence_order_define_first_verified_selection() -> None:
    invalid = "No bounded measurement was reported."
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["202", "101"], total=10),
            PUBMED_SUMMARY_URL: _summary(["202", "101"]),
            PUBMED_FETCH_URL: _fetch(
                _article("202", prefix_sentence=invalid),
                _article("101"),
            ),
        }
    )

    outcome = _run(client)

    assert outcome.status == "verified"
    assert outcome.returned_document_ids == ("202", "101")
    assert [(group.document_id, group.document_rank, group.sentence_index) for group in outcome.groups] == [
        ("202", 0, 1),
        ("101", 1, 0),
    ]
    assert outcome.selected_group_sha256 == outcome.groups[0].group_sha256


@pytest.mark.parametrize(
    "warning",
    [
        {"phrasesignored": ["ignored phrase"]},
        {"quotedphrasesnotfound": ["missing phrase"]},
    ],
)
def test_esearch_warning_or_ignored_phrase_fails_closed(warning: Any) -> None:
    client = FakeClient({PUBMED_SEARCH_URL: _search(["101"], warninglist=warning)})

    outcome = _run(client)

    assert outcome.status == "failed"
    assert outcome.reason_code == "esearch_warning"
    assert outcome.groups == outcome.records == ()
    assert len(client.calls) == 1


def test_not_found_within_bound_is_explicit_and_preserves_search_ledger() -> None:
    client = FakeClient({PUBMED_SEARCH_URL: _search([], total=29)})

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "NOT_FOUND_WITHIN_BOUND"
    assert outcome.total_result_count == 29
    assert outcome.unsearched_result_count == 29
    assert outcome.returned_document_ids == ()
    assert outcome.returned_document_ids_sha256 is not None
    assert len(outcome.calls) == 1
    assert outcome.groups == outcome.records == ()


def test_fetch_publication_type_is_independently_retraction_safe() -> None:
    client = FakeClient(
        {
            PUBMED_SEARCH_URL: _search(["101"]),
            PUBMED_SUMMARY_URL: _summary(["101"]),
            PUBMED_FETCH_URL: _fetch(_article("101", publication_type="Review")),
        }
    )

    outcome = _run(client)

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "NOT_FOUND_WITHIN_BOUND"
    assert outcome.groups == ()
    assert outcome.records[0].rejection_codes == ("fetch_non_primary_or_review",)
    assert outcome.records[0].fetch_publication_types == ("review",)


@pytest.mark.parametrize(
    "sentence",
    [
        SENTENCE.replace("most potent GENE", "most potent phase3 GENE"),
        SENTENCE.replace(
            "Alpha was the most potent",
            "Alpha was predicted in silico the most potent",
        ),
    ],
    ids=["unbound-contextual-numeral", "nonexperimental-prediction"],
)
def test_contextual_numeral_or_nonexperimental_language_cannot_form_a_group(
    sentence: str,
) -> None:
    outcome = _run(_client(sentence=sentence))

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "NOT_FOUND_WITHIN_BOUND"
    assert outcome.groups == ()
    assert outcome.verified_candidate_ids == ()


@pytest.mark.parametrize(
    "phrase",
    [
        "projected",
        "hypothetical",
        "putative",
        "planned to be",
        "expected to be",
        "would be",
        "will be",
        "proposed to be",
        "prospective",
        "prospectively",
    ],
)
def test_forward_or_nonobservational_wording_cannot_form_a_search_group(
    phrase: str,
) -> None:
    sentence = SENTENCE.replace(
        "Alpha was the most potent",
        f"Alpha was {phrase} the most potent",
    )
    outcome = _run(_client(sentence=sentence))

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "NOT_FOUND_WITHIN_BOUND"
    assert outcome.groups == ()


def test_retrospective_experimental_search_wording_remains_verified() -> None:
    sentence = SENTENCE.replace(
        "Alpha was the most potent",
        "Alpha was experimentally, as expected, the most potent",
    )
    outcome = _run(_client(sentence=sentence))

    assert outcome.status == "verified"
    assert len(outcome.groups) == 1


@pytest.mark.parametrize(
    "head",
    [
        "Alpha was measured as the most potent GENE inhibitor",
        "Alpha was determined to be the most potent GENE inhibitor",
        "Alpha was determined to inhibit GENE",
        "Alpha inhibited GENE",
    ],
    ids=[
        "measured-potency",
        "determined-potency",
        "determined-inhibition",
        "observed-active-inhibition",
    ],
)
def test_closed_observed_or_measured_search_heads_remain_verified(head: str) -> None:
    sentence = SENTENCE.replace("Alpha was the most potent GENE inhibitor", head)
    outcome = _run(_client(sentence=sentence))

    assert outcome.status == "verified"
    assert len(outcome.groups) == 1


@pytest.mark.parametrize(
    "head",
    [
        "Alpha was in-silico the most potent",
        "Alpha was modeling the most potent",
        "Alpha was modelling the most potent",
        "Alpha was simulation ranked the most potent",
        "Alpha was projecting the most potent",
        "Alpha was estimated to be the most potent",
        "Alpha was forecast as the most potent",
        "Alpha was anticipated to be the most potent",
        "Alpha was aimed to be the most potent",
        "Alpha was plans to be the most potent",
        "Alpha was proposes to be the most potent",
        "Alpha was will likely be the most potent",
        "Alpha was would have been the most potent",
    ],
    ids=[
        "hyphenated-in-silico",
        "modeling",
        "modelling",
        "simulation",
        "projecting",
        "estimated",
        "forecast",
        "anticipated",
        "aimed",
        "plans",
        "proposes",
        "will-likely",
        "would-have-been",
    ],
)
def test_adversarial_nonobservational_morphology_cannot_form_search_group(
    head: str,
) -> None:
    sentence = SENTENCE.replace("Alpha was the most potent", head)
    outcome = _run(_client(sentence=sentence))

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "NOT_FOUND_WITHIN_BOUND"
    assert outcome.groups == ()


def test_unrecognized_search_head_modifier_fails_closed() -> None:
    sentence = SENTENCE.replace(
        "Alpha was the most potent",
        "Alpha was algorithmically ranked the most potent",
    )
    outcome = _run(_client(sentence=sentence))

    assert outcome.status == "not_verified"
    assert outcome.reason_code == "NOT_FOUND_WITHIN_BOUND"
    assert outcome.groups == ()


@pytest.mark.parametrize("failed_stage", ["esearch", "esummary", "efetch"])
def test_partial_network_or_http_failure_never_returns_partial_science(failed_stage: str) -> None:
    responses: dict[str, FakeResponse | Exception] = {
        PUBMED_SEARCH_URL: _search(["101"]),
        PUBMED_SUMMARY_URL: _summary(["101"]),
        PUBMED_FETCH_URL: _fetch(_article("101")),
    }
    endpoint = {
        "esearch": PUBMED_SEARCH_URL,
        "esummary": PUBMED_SUMMARY_URL,
        "efetch": PUBMED_FETCH_URL,
    }[failed_stage]
    responses[endpoint] = FakeResponse(payload={"error": "down"}, status_code=503)

    outcome = _run(FakeClient(responses))

    assert outcome.status == "failed"
    assert outcome.reason_code == f"{failed_stage}_http_failure"
    assert outcome.groups == outcome.records == ()
    assert outcome.verified_candidate_ids == ()


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"endpoint": "[IC50]"}, "endpoint_must_be_base_token"),
        (
            {
                "candidates": {
                    key: value for index, (key, value) in enumerate(CANDIDATES.items()) if index < 2
                }
            },
            "insufficient_candidate_pool",
        ),
        (
            {
                "candidates": {
                    **CANDIDATES,
                    **{
                        f"candidate-extra-{index}": CandidateSearchAliases(
                            f"Extra {index}", (f"Extra {index}",)
                        )
                        for index in range(9)
                    },
                }
            },
            "candidate_pool_bound_exceeded",
        ),
    ],
)
def test_invalid_or_out_of_bound_fixed_pool_fails_before_http(
    overrides: dict[str, Any], reason: str
) -> None:
    client = FakeClient({})

    outcome = _run(client, **overrides)

    assert outcome.status == "failed"
    assert outcome.reason_code == reason
    assert outcome.query is None
    assert outcome.calls == outcome.groups == outcome.records == ()
    assert client.calls == []


def test_retrying_client_is_rejected_before_any_wire_attempt() -> None:
    client = FakeClient({}, max_attempts=2)
    outcome = _run(client)
    assert outcome.status == "failed"
    assert outcome.reason_code == "client_retry_policy_not_single_attempt"
    assert client.calls == []


def test_production_modules_contain_no_target_candidate_pmid_or_expected_value_literals() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        (root / "src/drugpilot/tools/primary_abstract_paper_native_v5.py").read_text(
            encoding="utf-8"
        )
        + (root / "src/drugpilot/tools/primary_abstract_paper_native_search_v5.py").read_text(
            encoding="utf-8"
        )
    ).casefold()
    for forbidden in (
        "jak2",
        "ruxolitinib",
        "fedratinib",
        "momelotinib",
        "pacritinib",
        "42007252",
        "2.9nm",
        "17nm",
        "29nm",
        "39nm",
    ):
        assert forbidden not in source
