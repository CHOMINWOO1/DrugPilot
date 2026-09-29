from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pytest

from drugpilot.tools.europe_pmc_oa_fulltext import (
    EXCERPT_SELECTION_POLICY,
    MAX_EXCERPT_CHARS,
    MAX_OA_JATS_BYTES,
    RETRIEVAL_POLICY,
    XML_PARSER_POLICY,
    EuropePMCOAValidationError,
    OAArticleEligibility,
    deduplicate_one_hop_references,
    document_identity_keys,
    exclude_visited_references,
    parse_europe_pmc_references,
    parse_oa_jats,
)


LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/"


def _eligibility(
    *,
    pmcid: str = "PMC10001",
    pmid: str | None = "10001",
    publication_types: tuple[str, ...] = ("research article", "journal article"),
    is_open_access: bool = True,
    license_uri: str = "CC BY",
    is_retracted: bool = False,
    doi: str | None = None,
) -> OAArticleEligibility:
    return OAArticleEligibility(
        pmcid=pmcid,
        pmid=pmid,
        publication_types=publication_types,
        is_open_access=is_open_access,
        license_uri=license_uri,
        is_retracted=is_retracted,
        doi=doi,
    )


def _jats(
    body: str,
    *,
    pmcid: str = "PMC10001",
    pmid: str | None = "10001",
    doi: str | None = None,
    license_uri: str = LICENSE_URI,
    references: str = "",
) -> str:
    pmid_node = (
        f'<article-id pub-id-type="pmid">{pmid}</article-id>' if pmid else ""
    )
    doi_node = f'<article-id pub-id-type="doi">{doi}</article-id>' if doi else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <article-id pub-id-type="pmc">{pmcid}</article-id>
    {pmid_node}
    {doi_node}
    <permissions><license xlink:href="{license_uri}">
      <license-p>Open access article licence.</license-p>
    </license></permissions>
  </article-meta></front>
  <body>{body}</body>
  <back>{references}</back>
</article>"""


def _parse(xml: str, **kwargs: object):
    return parse_oa_jats(
        xml,
        eligibility=kwargs.pop("eligibility", _eligibility()),
        allowed_license_uris=kwargs.pop("allowed_license_uris", {"CC BY"}),
        **kwargs,
    )


def test_positive_results_paragraph_yields_bounded_traceable_span() -> None:
    sentence = (
        "Direct KIN1 inhibition reduced measured viability in condition alpha "
        "cells by 45 percent."
    )
    result = _parse(
        _jats(
            f"""
            <sec sec-type="results"><title>Results</title>
              <p>{sentence}</p>
            </sec>
            """
        )
    )

    assert [span.text for span in result.spans] == [sentence]
    span = result.spans[0]
    assert span.section_kind == "results"
    assert span.source_location == (
        f"full_text:results:sec:1:p:1:sentence:1:chars:0-{len(sentence)}"
    )
    assert len(span.text) <= MAX_EXCERPT_CHARS
    metadata = result.evidence_metadata(span)
    assert metadata["source_database"] == "Europe PMC"
    assert metadata["evidence_origin"] == "oa_results_fulltext_direct"
    assert metadata["discovery_origin"] == "direct_search"
    assert metadata["content_availability"] == "oa_fulltext_excerpt"
    assert metadata["source_location_scope"] == (
        "normalized_admitted_paragraph_text"
    )
    assert metadata["source_location_char_end_exclusive"] is True
    assert metadata["content_retrieval_policy"] == RETRIEVAL_POLICY
    assert metadata["excerpt_selection_policy"] == EXCERPT_SELECTION_POLICY
    assert metadata["full_text_license"] == LICENSE_URI
    assert metadata["full_text_license_attestation_source"] == (
        "europe_pmc_metadata_and_jats"
    )
    assert metadata["jats_license_present"] is True
    assert metadata["full_text_stored"] is False
    assert metadata["citation_hop"] == 0
    assert metadata["citation_depth"] == 0
    assert metadata["citation_relation"] == "direct_search"
    assert metadata["independent_primary_verification"] is True
    assert metadata["independent_primary_verification_required"] is True
    assert metadata["content_hash"] == metadata["full_text_content_hash"]
    assert metadata["eligible_content_hash"] != metadata["content_hash"]


def test_allowed_europe_pmc_metadata_license_admits_jats_without_license_node() -> None:
    xml = _jats(
        '<sec sec-type="results"><p>Measured result.</p></sec>'
    ).replace(
        f'<permissions><license xlink:href="{LICENSE_URI}">\n'
        "      <license-p>Open access article licence.</license-p>\n"
        "    </license></permissions>",
        "",
    )

    result = _parse(xml)
    metadata = result.to_audit_metadata()

    assert metadata["full_text_license"] == "CC BY"
    assert metadata["full_text_license_attestation_source"] == (
        "europe_pmc_metadata"
    )
    assert metadata["jats_license_present"] is False


def test_only_results_findings_and_case_sections_are_admitted() -> None:
    body = """
      <sec sec-type="intro"><title>Introduction</title>
        <p>INTRODUCTION_SENTINEL should never be admitted.</p>
      </sec>
      <sec sec-type="results"><title>Results</title>
        <p>RESULTS_SENTINEL was measured in the admitted experiment.
          <fig><caption><p>FIGURE_SENTINEL must be excluded.</p></caption></fig>
          <table-wrap><table><tr><td>TABLE_SENTINEL</td></tr></table></table-wrap>
          <supplementary-material>SUPPLEMENT_SENTINEL</supplementary-material>
        </p>
        <sec sec-type="discussion"><title>Discussion</title>
          <p>NESTED_DISCUSSION_SENTINEL must be excluded.</p>
        </sec>
      </sec>
      <sec sec-type="findings"><title>Findings</title>
        <p>FINDINGS_SENTINEL was independently observed.</p>
      </sec>
      <sec sec-type="case-report"><title>Case presentation</title>
        <p>CASE_SENTINEL was recorded in the primary report.</p>
      </sec>
      <sec sec-type="discussion"><title>Discussion</title>
        <p>DISCUSSION_SENTINEL should never be admitted.</p>
      </sec>
    """
    result = _parse(_jats(body))
    serialized_spans = " ".join(span.text for span in result.spans)

    assert "RESULTS_SENTINEL" in serialized_spans
    assert "FINDINGS_SENTINEL" in serialized_spans
    assert "CASE_SENTINEL" in serialized_spans
    for forbidden in (
        "INTRODUCTION_SENTINEL",
        "FIGURE_SENTINEL",
        "TABLE_SENTINEL",
        "SUPPLEMENT_SENTINEL",
        "NESTED_DISCUSSION_SENTINEL",
        "DISCUSSION_SENTINEL",
    ):
        assert forbidden not in serialized_spans
    assert {span.section_kind for span in result.spans} == {
        "results",
        "findings",
        "case",
    }


def test_long_sentence_is_deterministically_bounded() -> None:
    long_sentence = "Observed response " + "boundedword " * 60 + "finished."
    result = _parse(
        _jats(
            f'<sec sec-type="results"><title>Results</title><p>{long_sentence}</p></sec>'
        )
    )

    span = result.spans[0]
    assert len(span.text) <= 320
    assert span.excerpt_truncated is True
    assert span.original_sentence_char_count == len(long_sentence)
    assert span.char_end - span.char_start == len(span.text)


@pytest.mark.parametrize(
    ("eligibility", "reason_code"),
    [
        (_eligibility(publication_types=("review article",)), "non_primary_publication_type"),
        (_eligibility(is_retracted=True), "retracted_publication"),
        (_eligibility(is_open_access=False), "not_open_access"),
    ],
)
def test_non_primary_or_non_oa_metadata_fails_closed(
    eligibility: OAArticleEligibility,
    reason_code: str,
) -> None:
    with pytest.raises(EuropePMCOAValidationError) as captured:
        _parse(
            _jats('<sec sec-type="results"><p>Measured result.</p></sec>'),
            eligibility=eligibility,
        )
    assert captured.value.reason_code == reason_code


@pytest.mark.parametrize(
    ("eligibility", "xml", "allowed", "reason_code"),
    [
        (
            _eligibility(pmcid="PMC10002"),
            _jats('<sec sec-type="results"><p>Measured result.</p></sec>'),
            {"CC BY"},
            "pmcid_mismatch",
        ),
        (
            _eligibility(pmid="10002"),
            _jats('<sec sec-type="results"><p>Measured result.</p></sec>'),
            {"CC BY"},
            "pmid_mismatch",
        ),
        (
            _eligibility(doi="10.1000/expected"),
            _jats(
                '<sec sec-type="results"><p>Measured result.</p></sec>',
                doi="10.1000/actual",
            ),
            {"CC BY"},
            "doi_mismatch",
        ),
        (
            _eligibility(license_uri="CC BY-NC"),
            _jats('<sec sec-type="results"><p>Measured result.</p></sec>'),
            {"CC BY-NC"},
            "jats_license_mismatch",
        ),
        (
            _eligibility(license_uri="CC BY"),
            _jats(
                '<sec sec-type="results"><p>Measured result.</p></sec>',
                license_uri="https://example.invalid/licenses/by/4.0/",
            ),
            {"CC BY"},
            "jats_license_mismatch",
        ),
        (
            _eligibility(license_uri="CC BY"),
            _jats('<sec sec-type="results"><p>Measured result.</p></sec>'),
            {"CC BY-NC"},
            "license_not_allowed",
        ),
    ],
)
def test_identifier_and_license_mismatches_fail_closed(
    eligibility: OAArticleEligibility,
    xml: str,
    allowed: set[str],
    reason_code: str,
) -> None:
    with pytest.raises(EuropePMCOAValidationError) as captured:
        _parse(xml, eligibility=eligibility, allowed_license_uris=allowed)
    assert captured.value.reason_code == reason_code


def test_oversize_doctype_and_entity_payloads_are_rejected_before_parse() -> None:
    with pytest.raises(EuropePMCOAValidationError) as oversized:
        _parse("x" * (MAX_OA_JATS_BYTES + 1))
    assert oversized.value.reason_code == "jats_payload_too_large"

    with pytest.raises(EuropePMCOAValidationError) as doctype:
        _parse(
            '<!DOCTYPE article SYSTEM "remote.dtd">'
            + _jats('<sec sec-type="results"><p>Measured result.</p></sec>')
        )
    assert doctype.value.reason_code == "jats_doctype_forbidden"

    with pytest.raises(EuropePMCOAValidationError) as entity:
        _parse(
            '<!ENTITY unsafe "expanded">'
            + _jats('<sec sec-type="results"><p>Measured result.</p></sec>')
        )
    assert entity.value.reason_code == "jats_entity_forbidden"


def test_official_external_jats_doctype_is_removed_without_loading_the_dtd() -> None:
    xml = _jats(
        '<sec sec-type="results"><p>Measured result.</p></sec>'
    ).replace(
        "?>",
        "?>\n<!DOCTYPE article\n"
        '  PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and '
        'Interchange DTD with MathML3 v1.4 20241031//EN" '
        '"JATS-archivearticle1-4-mathml3.dtd">',
        1,
    )

    result = _parse(xml)

    assert result.audit.external_doctype_removed is True
    metadata = result.to_audit_metadata()
    assert metadata["xml_parser_policy"] == XML_PARSER_POLICY
    assert metadata["external_doctype_removed"] is True
    assert metadata["content_hash"] == hashlib.sha256(
        xml.encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "declaration",
    [
        '<!DOCTYPE article [<!ENTITY unsafe "expanded">]>',
        '<!DOCTYPE book SYSTEM "JATS-archivearticle1-4-mathml3.dtd">',
        '<!DOCTYPE article SYSTEM "one.dtd"><!DOCTYPE article SYSTEM "two.dtd">',
    ],
)
def test_non_external_article_doctype_shapes_remain_fail_closed(
    declaration: str,
) -> None:
    xml = _jats(
        '<sec sec-type="results"><p>Measured result.</p></sec>'
    ).replace("?>", f"?>\n{declaration}", 1)
    with pytest.raises(EuropePMCOAValidationError) as captured:
        _parse(xml)
    assert captured.value.reason_code in {
        "jats_doctype_forbidden",
        "jats_entity_forbidden",
    }


def test_excessively_deep_jats_tree_fails_closed() -> None:
    nested = "<sec>" * 130 + "<p>Measured result.</p>" + "</sec>" * 130
    with pytest.raises(EuropePMCOAValidationError) as captured:
        _parse(_jats(nested))
    assert captured.value.reason_code == "jats_depth_limit_exceeded"


def test_jats_references_are_identifier_only_deduplicated_and_acyclic() -> None:
    references = """
      <ref-list>
        <ref id="r1"><element-citation>
          <pub-id pub-id-type="pmid">20001</pub-id>
          <pub-id pub-id-type="doi">10.1000/child.one</pub-id>
        </element-citation></ref>
        <ref id="r2"><element-citation>
          <pub-id pub-id-type="doi">10.1000/CHILD.ONE</pub-id>
        </element-citation></ref>
        <ref id="self"><element-citation>
          <pub-id pub-id-type="pmid">10001</pub-id>
        </element-citation></ref>
        <ref id="r4"><element-citation>
          <pub-id pub-id-type="pmc">PMC30001</pub-id>
        </element-citation></ref>
      </ref-list>
    """
    result = _parse(
        _jats(
            '<sec sec-type="results"><p>Measured result.</p></sec>',
            references=references,
        )
    )

    assert [(item.reference_position, item.canonical_id) for item in result.references] == [
        (1, "MED:20001"),
        (4, "PMC:PMC30001"),
    ]
    first = result.references[0]
    assert first.identity_keys == ("DOI:10.1000/child.one", "MED:20001")
    assert first.citation_hop == 1
    lineage = first.to_lineage_metadata()
    assert lineage["source_database"] == "Europe PMC"
    assert lineage["discovery_origin"] == "reference_one_hop"
    assert lineage["citation_depth"] == 1
    assert lineage["evidence_role_inherited"] is False


def test_reference_api_parser_deduplicates_and_excludes_visited_cycles() -> None:
    payload = {
        "referenceList": {
            "reference": [
                {"citedOrder": 1, "source": "MED", "id": "21001"},
                {
                    "citedOrder": 2,
                    "pmid": "21001",
                    "doi": "10.1000/shared.child",
                },
                {"citedOrder": 3, "source": "MED", "id": "10001"},
                {"citedOrder": 4, "pmcid": "PMC31001"},
            ]
        }
    }
    parsed = parse_europe_pmc_references(
        payload,
        parent_pmcid="PMC10001",
        parent_pmid="10001",
    )

    assert [item.canonical_id for item in parsed.references] == [
        "MED:21001",
        "PMC:PMC31001",
    ]
    assert parsed.references[0].doi == "10.1000/shared.child"
    assert parsed.audit.raw_reference_count == 4
    assert parsed.audit.emitted_reference_count == 2

    visited = document_identity_keys(pmcid="PMC31001")
    remaining = exclude_visited_references(parsed.references, visited)
    assert [item.canonical_id for item in remaining] == ["MED:21001"]
    assert deduplicate_one_hop_references(parsed.references) == parsed.references


def test_reference_deduplication_merges_transitive_identifier_aliases() -> None:
    payload = {
        "referenceList": {
            "reference": [
                {"citedOrder": 1, "pmid": "23001"},
                {"citedOrder": 2, "doi": "10.1000/transitive.child"},
                {
                    "citedOrder": 3,
                    "pmid": "23001",
                    "doi": "10.1000/transitive.child",
                },
            ]
        }
    }

    parsed = parse_europe_pmc_references(
        payload,
        parent_pmcid="PMC10001",
        parent_pmid="10001",
    )

    assert len(parsed.references) == 1
    assert parsed.references[0].reference_position == 1
    assert parsed.references[0].identity_keys == (
        "DOI:10.1000/transitive.child",
        "MED:23001",
    )


def test_one_hop_lineage_must_match_independently_verified_child() -> None:
    reference_payload = {
        "referenceList": {
            "reference": [
                {
                    "citedOrder": 1,
                    "source": "MED",
                    "id": "22001",
                    "pmcid": "PMC22001",
                    "doi": "10.1000/verified.child",
                }
            ]
        }
    }
    lineage = parse_europe_pmc_references(
        reference_payload,
        parent_pmcid="PMC10001",
        parent_pmid="10001",
    ).references[0]
    child = parse_oa_jats(
        _jats(
            '<sec sec-type="findings"><p>Independent child finding was measured.</p></sec>',
            pmcid="PMC22001",
            pmid="22001",
            doi="10.1000/verified.child",
        ),
        eligibility=_eligibility(
            pmcid="PMC22001",
            pmid="22001",
            doi="10.1000/verified.child",
        ),
        allowed_license_uris={"CC BY"},
    )
    metadata = child.evidence_metadata(child.spans[0], lineage=lineage)

    assert metadata["citation_hop"] == 1
    assert metadata["citation_depth"] == 1
    assert metadata["citation_relation"] == "reference"
    assert metadata["source_database"] == "Europe PMC"
    assert metadata["evidence_origin"] == (
        "oa_results_fulltext_reference_one_hop"
    )
    assert metadata["discovery_origin"] == "reference_one_hop"
    assert metadata["citation_parent_external_id"] == "10001"
    assert metadata["citation_parent_pmcid"] == "PMC10001"
    assert metadata["citation_parent_pmid"] == "10001"
    assert metadata["citation_reference_position"] == 1
    assert metadata["citation_child_canonical_id"] == "MED:22001"
    assert metadata["citation_child_identity_keys"] == [
        "DOI:10.1000/verified.child",
        "MED:22001",
        "PMC:PMC22001",
    ]
    assert metadata["citation_root_external_id"] == "10001"
    assert metadata["lineage_verified"] is True
    assert metadata["evidence_role_inherited"] is False

    unrelated = parse_oa_jats(
        _jats(
            '<sec sec-type="results"><p>Unrelated measured result.</p></sec>',
            pmcid="PMC22002",
            pmid="22002",
        ),
        eligibility=_eligibility(pmcid="PMC22002", pmid="22002"),
        allowed_license_uris={"CC BY"},
    )
    with pytest.raises(EuropePMCOAValidationError) as mismatch:
        unrelated.evidence_metadata(unrelated.spans[0], lineage=lineage)
    assert mismatch.value.reason_code == "citation_child_identity_mismatch"


def test_audit_serialization_never_contains_raw_full_text_or_unsafe_body() -> None:
    unsafe_text = "UNSAFE_FULL_BODY_SENTINEL " * 30
    xml = _jats(
        f"""
        <sec sec-type="introduction"><title>Introduction</title><p>{unsafe_text}</p></sec>
        <sec sec-type="results"><title>Results</title><p>One bounded result.</p></sec>
        """
    )
    result = _parse(xml)

    audit_json = json.dumps(result.to_audit_metadata(), sort_keys=True)
    dataclass_json = json.dumps(asdict(result), sort_keys=True)
    evidence_json = json.dumps(
        result.evidence_metadata(result.spans[0]),
        sort_keys=True,
    )
    assert "UNSAFE_FULL_BODY_SENTINEL" not in audit_json
    assert "UNSAFE_FULL_BODY_SENTINEL" not in dataclass_json
    assert "UNSAFE_FULL_BODY_SENTINEL" not in evidence_json
    assert "<article" not in dataclass_json
    assert "<body" not in dataclass_json
    assert "raw_xml" not in dataclass_json
    assert "full_text_stored" in audit_json
    assert result.audit.full_text_stored is False
