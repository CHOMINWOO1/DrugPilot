from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any, Mapping

import httpx
import pytest

from drugpilot.agents.ma1 import MockEvidenceRetriever, TargetSeed
from drugpilot.domain.models import (
    DataSourcePolicy,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
)
from drugpilot.tools.evidence import ResilientEvidenceRetriever
from drugpilot.tools.public_biology import (
    DGIDB_GRAPHQL_URL,
    EUROPE_PMC_SEARCH_URL,
    OPEN_TARGETS_GRAPHQL_URL,
    PHAROS_GRAPHQL_URL,
    PUBMED_FETCH_URL,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
    UNIPROT_SEARCH_URL,
    PublicBiologyRetriever,
    _PRIMARY_ASSOCIATION_NEGATIVE_RELATION_PATTERN,
    _association_predicate_for_target,
    _classify_counter_signal,
    _direct_support_rejection_codes,
    _direction_endpoint_polarity_status,
    _disease_query_terms,
    _excerpt_matches_variant_clause,
    _has_direct_target_disease_match,
    _has_explicit_primary_disease_cohort,
    _has_admissible_direction_endpoint,
    _has_positive_prevalence_quantitation,
    _has_strict_association_target_disease_match,
    _has_target_rearrangement_or_fusion,
    _has_target_subtype_or_variant,
    _primary_direction_content_rejection_codes,
    _primary_direction_context_match_basis,
    _primary_direction_model_context,
    _primary_association_requirement,
    _primary_publication_metadata_rejection_codes,
    _parse_pubmed_content,
    _pubmed_xml_record_ids,
    _publication_content_from_sections,
    _target_prevalence_quantitations,
    _variant_clause_is_association_subject,
)


def test_retriever_rejects_downstream_only_signaling_as_direct_endpoint() -> None:
    assert _has_admissible_direction_endpoint(
        "FLT3 inhibitors suppressed RSK1 phosphorylation in FLT3-ITD AML cells.",
        "FLT3",
    ) is False
    assert _has_admissible_direction_endpoint(
        "FLT3 inhibitors suppressed FLT3 phosphorylation in FLT3-ITD AML cells.",
        "FLT3",
    ) is True


def test_evidence_id_is_bound_to_query_and_bounded_excerpt_context() -> None:
    retriever = PublicBiologyRetriever(enabled_sources=[])
    first = retriever._record(
        source="Europe PMC",
        external_id="42490065",
        title="Shared publication",
        url="https://example.test/article",
        supports="context",
        query='TITLE_ABS:"IDH1" AND TITLE_ABS:"AML"',
        endpoint=EUROPE_PMC_SEARCH_URL,
        extra={
            "content_availability": "abstract_excerpt",
            "source_location": "abstract:RESULTS:chars:0-200",
        },
        excerpt_text="IDH1 inhibition reduced 2-HG in AML models.",
    )
    second = retriever._record(
        source="Europe PMC",
        external_id="42490065",
        title="Shared publication",
        url="https://example.test/article",
        supports="context",
        query='TITLE_ABS:"CEBPA" AND TITLE_ABS:"AML"',
        endpoint=EUROPE_PMC_SEARCH_URL,
        extra={
            "content_availability": "abstract_excerpt",
            "source_location": "abstract:BACKGROUND:chars:201-400",
        },
        excerpt_text="CEBPA was discussed in AML models.",
    )

    assert first.evidence_id != second.evidence_id
    assert first.metadata["evidence_id_origin_policy"] == (
        "source-external-query-location-excerpt-v2"
    )


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        status_code: int = 200,
        *,
        text: str = "",
    ) -> None:
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failure", request=request, response=response)

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeBiologyClient:
    def __init__(self, *, failing_sources: set[str] | None = None) -> None:
        self.failing_sources = set(failing_sources or set())
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        source = {
            PUBMED_SEARCH_URL: "PubMed",
            PUBMED_SUMMARY_URL: "PubMed",
            PUBMED_FETCH_URL: "PubMed",
            EUROPE_PMC_SEARCH_URL: "Europe PMC",
            UNIPROT_SEARCH_URL: "UniProt",
        }[url]
        if source in self.failing_sources:
            request = httpx.Request("GET", url)
            raise httpx.ConnectTimeout("offline", request=request)

        if url == PUBMED_SEARCH_URL:
            return FakeResponse({"esearchresult": {"idlist": ["12345678"]}})
        if url == PUBMED_SUMMARY_URL:
            return FakeResponse(
                {
                    "result": {
                        "uids": ["12345678"],
                        "12345678": {
                            "uid": "12345678",
                            "title": "FLT3 inhibition in acute myeloid leukemia",
                            "pubdate": "2025",
                            "fulljournalname": "Example Journal",
                            "authors": [{"name": "Kim J"}],
                            "pubtype": ["Journal Article"],
                            "articleids": [
                                {"idtype": "doi", "value": "10.1000/example.12345678"}
                            ],
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL:
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='BACKGROUND'>Unrelated background text.</AbstractText>"
                    "<AbstractText Label='RESULTS'>FLT3 inhibition reduced leukemia "
                    "burden in acute myeloid leukemia models.</AbstractText>"
                    "</Abstract></Article></MedlineCitation><PubmedData><ArticleIdList>"
                    "<ArticleId IdType='pubmed'>12345678</ArticleId>"
                    "<ArticleId IdType='doi'>10.1000/example.12345678</ArticleId>"
                    "<ArticleId IdType='pmc'>PMC12345678</ArticleId>"
                    "</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"
                )
            )
        if url == EUROPE_PMC_SEARCH_URL:
            return FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "pmcid": "PMC9999999",
                                "title": "A functional study of FLT3 in AML",
                                "abstractText": (
                                    "FLT3 inhibition reduced FLT3 phosphorylation and "
                                    "viability in acute myeloid leukemia models."
                                ),
                                "doi": "10.1000/epmc.9999999",
                                "pubYear": "2024",
                                "journalTitle": "Example Biology",
                                "pubTypeList": {
                                    "pubType": ["research article", "journal article"]
                                },
                                "citedByCount": 7,
                                "isOpenAccess": "Y",
                            }
                        ]
                    }
                }
            )
        if url == UNIPROT_SEARCH_URL:
            return FakeResponse(
                {
                    "results": [
                        {
                            "primaryAccession": "P36888",
                            "uniProtkbId": "FLT3_HUMAN",
                            "entryType": "UniProtKB reviewed (Swiss-Prot)",
                            "proteinDescription": {
                                "recommendedName": {
                                    "fullName": {"value": "Receptor-type tyrosine-protein kinase FLT3"}
                                }
                            },
                            "genes": [
                                {
                                    "geneName": {"value": "FLT3"},
                                    "synonyms": [{"value": "CD135"}],
                                }
                            ],
                            "organism": {"scientificName": "Homo sapiens"},
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected GET: {url}")

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        source = {
            OPEN_TARGETS_GRAPHQL_URL: "Open Targets",
            DGIDB_GRAPHQL_URL: "DGIdb",
            PHAROS_GRAPHQL_URL: "Pharos",
        }[url]
        if source in self.failing_sources:
            request = httpx.Request("POST", url)
            raise httpx.ConnectTimeout("offline", request=request)
        if url == OPEN_TARGETS_GRAPHQL_URL:
            return FakeResponse(
                {
                    "data": {
                        "search": {
                            "hits": [
                                {
                                    "id": "ENSG00000122025",
                                    "name": "FLT3",
                                    "entity": "target",
                                    "description": "Fms related receptor tyrosine kinase 3",
                                }
                            ]
                        }
                    }
                }
            )
        if url == DGIDB_GRAPHQL_URL:
            return FakeResponse(
                {
                    "data": {
                        "genes": {
                            "nodes": [
                                {
                                    "name": "FLT3",
                                    "longName": "fms related receptor tyrosine kinase 3",
                                    "conceptId": "hgnc:3765",
                                    "geneCategoriesWithSources": [
                                        {"name": "KINASE", "sourceNames": ["GuideToPharmacology"]}
                                    ],
                                    "interactions": [
                                        {
                                            "drug": {
                                                "name": "QUIZARTINIB",
                                                "conceptId": "chembl:576982",
                                            },
                                            "interactionScore": 8.5,
                                            "interactionTypes": [
                                                {"type": "inhibitor", "directionality": "inhibitory"}
                                            ],
                                            "publications": [{"pmid": 12345678}],
                                            "sources": [{"sourceDbName": "ChEMBL"}],
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                }
            )
        if url == PHAROS_GRAPHQL_URL:
            return FakeResponse(
                {
                    "data": {
                        "targets": {
                            "count": 1,
                            "targets": [
                                {
                                    "name": "Tyrosine-protein kinase receptor FLT3",
                                    "sym": "FLT3",
                                    "description": "A receptor tyrosine kinase",
                                    "tdl": "Tclin",
                                    "fam": "Kinase",
                                    "novelty": 0.12,
                                }
                            ],
                        }
                    }
                }
            )
        raise AssertionError(f"unexpected POST: {url}")


class CounterEvidenceBiologyClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        query = str(kwargs.get("params", {}).get("query", ""))
        if url == EUROPE_PMC_SEARCH_URL and "TITLE_ABS:resistance" in query:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "pmid": "99887766",
                                "title": "Resistance to FLT3 inhibition in acute myeloid leukemia",
                                "abstractText": (
                                    "Acquired resistance to FLT3 inhibition limited durable "
                                    "response in acute myeloid leukemia models."
                                ),
                                "pubYear": "2025",
                                "journalTitle": "Example Resistance Journal",
                                "isOpenAccess": "Y",
                                "pubTypeList": {
                                    "pubType": ["journal article"]
                                },
                            },
                            {
                                "pmid": "11223344",
                                "title": "FLT3 expression in AML",
                                "abstractText": "FLT3 expression was measured in AML samples.",
                                "pubYear": "2024",
                            },
                            {
                                "pmid": "55667788",
                                "title": "Contrasting AML biomarkers",
                                "abstractText": (
                                    "DNMT3A mutations were linked to resistance whereas "
                                    "FLT3 expression was linked to increased sensitivity."
                                ),
                                "pubYear": "2024",
                            },
                        ]
                    }
                }
            )
        return super().get(url, **kwargs)


class PubMedNoRelatedExcerptClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText>EGFR signaling was evaluated in lung carcinoma cells."
                    "</AbstractText></Abstract></Article></MedlineCitation>"
                    "</PubmedArticle></PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class MechanisticRankingBiologyClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='BACKGROUND'>FLT3 mutations are frequent "
                    "and predict poor survival in acute myeloid leukemia.</AbstractText>"
                    "<AbstractText Label='RESULTS'>Pharmacologic FLT3 inhibition "
                    "reduced FLT3 phosphorylation and viability in acute myeloid "
                    "leukemia cells.</AbstractText>"
                    "<AbstractText Label='RESULTS'>Enforced FLT3 activation "
                    "increased phosphorylation and proliferation in acute myeloid "
                    "leukemia cells.</AbstractText>"
                    "</Abstract></Article></MedlineCitation><PubmedData><ArticleIdList>"
                    "<ArticleId IdType='pubmed'>12345678</ArticleId>"
                    "</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class AcronymMechanisticRankingBiologyClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='BACKGROUND'>FLT3 mutations are frequent "
                    "and predict poor survival in acute myeloid leukemia.</AbstractText>"
                    "<AbstractText Label='RESULTS'>Inhibition of NOX2 increased the "
                    "apoptosis of FLT3-mutant AML cells in response to FLT3 "
                    "inhibitors.</AbstractText>"
                    "</Abstract></Article></MedlineCitation><PubmedData><ArticleIdList>"
                    "<ArticleId IdType='pubmed'>12345678</ArticleId>"
                    "</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class PubMedAbstractUnavailableClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            request = httpx.Request("GET", url)
            raise httpx.ConnectTimeout("abstract endpoint offline", request=request)
        return super().get(url, **kwargs)


class PubMedCounterSignalClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText>Acquired resistance to FLT3 inhibition limited "
                    "durable response in acute myeloid leukemia models.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class PrimaryAssociationSupplementClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        params = kwargs.get("params", {})
        term = str(params.get("term", ""))
        identifier = str(params.get("id", ""))
        if url == PUBMED_SEARCH_URL and "NOT (Review[Publication Type])" in term:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse({"esearchresult": {"idlist": ["20515557"]}})
        if url == PUBMED_SUMMARY_URL and identifier == "20515557":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                {
                    "result": {
                        "uids": ["20515557"],
                        "20515557": {
                            "uid": "20515557",
                            "title": "Primary FLT3-ITD cohort study in AML",
                            "pubdate": "2010",
                            "fulljournalname": "Example Hematology",
                            "pubtype": ["Journal Article"],
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL and identifier == "20515557":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>20515557</PMID><Article><Abstract>"
                    "<AbstractText>Among the AML patients, a FLT3 internal "
                    "tandem duplication (FLT3/ITD) mutation was present in 59 "
                    "cases (15.8%).</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class PagedPrimaryAssociationSupplementClient(FakeBiologyClient):
    def __init__(self, *, include_required_result: bool) -> None:
        super().__init__()
        self.include_required_result = include_required_result
        self.supplement_ids = [str(20_000_000 + index) for index in range(1, 25)]

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        params = kwargs.get("params", {})
        term = str(params.get("term", ""))
        identifier = str(params.get("id", ""))
        if url == PUBMED_SEARCH_URL and "NOT (Review[Publication Type])" in term:
            self.calls.append(("GET", url, kwargs))
            second_lane = 'NOT "prevalence"[Title]' in term
            returned = (
                self.supplement_ids[8:]
                if second_lane
                else self.supplement_ids[:8]
            )
            return FakeResponse(
                {
                    "esearchresult": {
                        "count": "16" if second_lane else "15",
                        "idlist": returned,
                    }
                }
            )
        if url == PUBMED_SUMMARY_URL and identifier.startswith("200000"):
            self.calls.append(("GET", url, kwargs))
            ids = identifier.split(",")
            return FakeResponse(
                {
                    "result": {
                        "uids": ids,
                        **{
                            pmid: {
                                "uid": pmid,
                                "title": f"Bounded primary study {pmid}",
                                "pubdate": "2020",
                                "fulljournalname": "Example Hematology",
                                "pubtype": ["Journal Article"],
                            }
                            for pmid in ids
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL and identifier.startswith("200000"):
            self.calls.append(("GET", url, kwargs))
            ids = identifier.split(",")
            articles: list[str] = []
            for pmid in ids:
                if self.include_required_result and pmid == self.supplement_ids[16]:
                    abstract = (
                        "In our cohort of 123 acute myeloid leukemia patients, "
                        "FLT3-ITD mutations were detected in 24% of cases."
                    )
                else:
                    abstract = (
                        "Pharmacologic FLT3 inhibition reduced viability in "
                        "acute myeloid leukemia cells."
                    )
                articles.append(
                    "<PubmedArticle><MedlineCitation>"
                    f"<PMID>{pmid}</PMID><Article><Abstract>"
                    f"<AbstractText Label='RESULTS'>{abstract}</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                )
            return FakeResponse(
                text="<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"
            )
        if url == PUBMED_FETCH_URL and identifier == "12345678":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>In our cohort of 100 acute "
                    "myeloid leukemia patients, FLT3 mutations were detected "
                    "in 30% of cases.</AbstractText>"
                    "<AbstractText Label='RESULTS'>Ningetinib inhibited FLT3 "
                    "phosphorylation and reduced viability in acute myeloid "
                    "leukemia cells.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class OverlappingPrimaryAssociationLanesClient(
    PagedPrimaryAssociationSupplementClient
):
    def __init__(self) -> None:
        super().__init__(include_required_result=False)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        params = kwargs.get("params", {})
        term = str(params.get("term", ""))
        if url == PUBMED_SEARCH_URL and "NOT (Review[Publication Type])" in term:
            self.calls.append(("GET", url, kwargs))
            second_lane = 'NOT "prevalence"[Title]' in term
            returned = (
                self.supplement_ids[4:20]
                if second_lane
                else self.supplement_ids[:8]
            )
            return FakeResponse(
                {
                    "esearchresult": {
                        "count": str(len(returned)),
                        "idlist": returned,
                    }
                }
            )
        return super().get(url, **kwargs)


class CombinedResponseModifierClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>Combined inhibition of "
                    "FLT3/AXL potentiates venetoclax response in FLT3 wild-type "
                    "acute myeloid leukemia by inducing MCL-1 degradation."
                    "</AbstractText></Abstract></Article></MedlineCitation>"
                    "</PubmedArticle></PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class SensitizationResponseModifierClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>Ruxolitinib inhibited JAK2 "
                    "and was used to increase the sensitivity of acute myeloid "
                    "leukemia cells to arsenic trioxide.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class DirectRoleBoundaryEuropePMCClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == EUROPE_PMC_SEARCH_URL:
            self.calls.append(("GET", url, kwargs))
            excerpts = {
                "direct": (
                    "Ningetinib bound directly to FLT3, inhibited FLT3 "
                    "phosphorylation, and reduced viability in acute myeloid "
                    "leukemia cells."
                ),
                "downstream": (
                    "FLC-8 suppressed FLT3-mediated STAT5, AKT, and ERK "
                    "signaling and induced apoptosis in acute myeloid leukemia "
                    "cells."
                ),
                "other-target": (
                    "An ATR inhibitor more effectively inhibited cell "
                    "proliferation in FLT3-mutated acute myeloid leukemia cells."
                ),
                "other-drug": (
                    "Arsenic trioxide induced apoptosis in FLT3-ITD acute "
                    "myeloid leukemia cells through ROS/ERK signaling."
                ),
                "modifier": (
                    "PP2A-activating drugs enhanced FLT3 inhibitor growth "
                    "suppression and apoptosis in FLT3-ITD acute myeloid "
                    "leukemia cells."
                ),
            }
            return FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "pmid": f"boundary-{name}",
                                "title": f"Primary boundary study {name}",
                                "abstractText": excerpt,
                                "pubYear": "2026",
                                "pubTypeList": {
                                    "pubType": ["journal article"]
                                },
                            }
                            for name, excerpt in excerpts.items()
                        ]
                    }
                }
            )
        return super().get(url, **kwargs)


class SameDocumentDualRoleClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>In our cohort of 123 acute "
                    "myeloid leukemia patients, FLT3-ITD mutations were detected "
                    "in 24% of cases.</AbstractText>"
                    "<AbstractText Label='RESULTS'>Ningetinib inhibited FLT3 "
                    "phosphorylation and reduced viability in acute myeloid "
                    "leukemia cells.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class AssociationOnlyMainQueryClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText>Among the AML patients, a FLT3 internal "
                    "tandem duplication (FLT3/ITD) mutation was present in 59 "
                    "cases (15.8%).</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class InvalidMainAssociationMetadataClient(
    PrimaryAssociationSupplementClient
):
    def __init__(self, *, title: str, publication_types: list[str]) -> None:
        super().__init__()
        self.title = title
        self.publication_types = publication_types

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        identifier = str(kwargs.get("params", {}).get("id", ""))
        if url == PUBMED_SUMMARY_URL and identifier == "12345678":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                {
                    "result": {
                        "uids": ["12345678"],
                        "12345678": {
                            "uid": "12345678",
                            "title": self.title,
                            "pubdate": "2020",
                            "fulljournalname": "Example Hematology",
                            "pubtype": self.publication_types,
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL and identifier == "12345678":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>In our cohort of 100 acute "
                    "myeloid leukemia patients, FLT3-ITD mutations were "
                    "detected in 30% of cases.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class MainSubtypeMismatchClient(PrimaryAssociationSupplementClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        identifier = str(kwargs.get("params", {}).get("id", ""))
        if url == PUBMED_FETCH_URL and identifier == "12345678":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>In our cohort of 100 acute "
                    "myeloid leukemia patients, FLT3 D835Y mutations were "
                    "present in 20 cases (20.0%), consistent with an "
                    "activating FLT3 mutation; FLT3-ITD status was documented "
                    "separately.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class MainTitleOnlyRecordClient(PrimaryAssociationSupplementClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        identifier = str(kwargs.get("params", {}).get("id", ""))
        if url == PUBMED_SUMMARY_URL and identifier == "12345678":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                {
                    "result": {
                        "uids": ["12345678"],
                        "12345678": {
                            "uid": "12345678",
                            "title": "Prevalence of FLT3-ITD in AML",
                            "pubdate": "2020",
                            "fulljournalname": "Example Hematology",
                            "pubtype": ["Journal Article"],
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL and identifier == "12345678":
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article />"
                    "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class EuropePMCReviewAssociationClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == EUROPE_PMC_SEARCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "pmid": "87654321",
                                "title": "Systematic review of FLT3 prevalence in AML",
                                "abstractText": (
                                    "Among AML patients, FLT3-ITD mutations "
                                    "were present in 59 of 200 cases (29.5%)."
                                ),
                                "pubYear": "2024",
                                "pubTypeList": {"pubType": ["Review"]},
                            }
                        ]
                    }
                }
            )
        return super().get(url, **kwargs)


class TwoAssociationSentenceMainQueryClient(FakeBiologyClient):
    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == PUBMED_FETCH_URL:
            self.calls.append(("GET", url, kwargs))
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345678</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>Among AML patients, "
                    "FLT3-ITD mutations were present in 59 of 200 cases "
                    "(29.5%). FLT3 expression was significantly higher and "
                    "correlated with proliferation in AML patients.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        return super().get(url, **kwargs)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TimedBiologyClient(FakeBiologyClient):
    def __init__(self, clock: FakeClock, *, seconds_per_http_call: float) -> None:
        super().__init__()
        self.clock = clock
        self.seconds_per_http_call = seconds_per_http_call

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        response = super().get(url, **kwargs)
        self.clock.advance(self.seconds_per_http_call)
        return response

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        response = super().post(url, **kwargs)
        self.clock.advance(self.seconds_per_http_call)
        return response


def _brief(
    sources: list[str] | None = None,
    *,
    direction: str = "inhibit",
) -> ResearchBrief:
    return ResearchBrief(
        research_goal="Find evidence-linked FLT3 candidates for audit testing.",
        disease="Acute myeloid leukemia",
        disease_id="EFO_0000222",
        constraints=ProjectConstraints(intervention_direction=direction),
        data_policy=DataSourcePolicy(
            allowed_sources=sources
            or ["PubMed", "Europe PMC", "Open Targets", "UniProt", "DGIdb", "Pharos"]
        ),
    )


def _seed() -> TargetSeed:
    return TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        uniprot_id="P36888",
    )


def _retriever(client: FakeBiologyClient, **kwargs: Any) -> PublicBiologyRetriever:
    return PublicBiologyRetriever(
        client=client,
        timeout_seconds=1,
        max_attempts=kwargs.pop("max_attempts", 1),
        retry_backoff_seconds=0,
        sleeper=lambda _: None,
        **kwargs,
    )


def test_all_public_sources_return_traceable_records() -> None:
    client = FakeBiologyClient()
    records = list(_retriever(client).retrieve(_brief(), _seed(), 1))

    by_source = {record.source: record for record in records}
    assert set(by_source) == {
        "PubMed",
        "Europe PMC",
        "Open Targets",
        "UniProt",
        "DGIdb",
        "Pharos",
    }

    pubmed = by_source["PubMed"]
    assert pubmed.external_id == "12345678"
    assert pubmed.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert "FLT3" in pubmed.metadata["query"]
    assert pubmed.supports == "context"
    assert pubmed.metadata["direct_support_eligible"] is False
    assert pubmed.metadata["verified_direct_primary"] is False
    assert "measured_endpoint_missing" in pubmed.metadata[
        "direct_support_rejection_codes"
    ]
    assert pubmed.metadata["pmid"] == "12345678"
    assert pubmed.metadata["pmcid"] == "PMC12345678"
    assert pubmed.metadata["doi"] == "10.1000/example.12345678"
    assert pubmed.metadata["content_availability"] == "abstract_excerpt"
    assert pubmed.metadata["abstract_retrieval_status"] == "available"
    assert pubmed.metadata["excerpt_selection_status"] == "selected"
    assert pubmed.metadata["source_location"].startswith("abstract:RESULTS:chars:")
    assert pubmed.metadata["excerpt"].startswith("FLT3 inhibition")
    assert pubmed.metadata["content_hash"]
    assert pubmed.excerpt_hash
    assert pubmed.metadata["full_text_stored"] is False
    assert len(pubmed.metadata["excerpt"]) <= 320

    assert by_source["Europe PMC"].external_id == "PMC9999999"
    assert by_source["Europe PMC"].url == "https://europepmc.org/article/PMC/PMC9999999"
    assert by_source["Europe PMC"].metadata["doi"] == "10.1000/epmc.9999999"
    assert by_source["Europe PMC"].metadata["content_hash"]
    assert by_source["Europe PMC"].metadata["excerpt_selection_status"] == "selected"
    assert by_source["Open Targets"].external_id == "ENSG00000122025"
    assert by_source["Open Targets"].url.endswith("/target/ENSG00000122025")
    assert by_source["UniProt"].external_id == "P36888"
    assert by_source["UniProt"].url == "https://www.uniprot.org/uniprotkb/P36888/entry"
    assert by_source["UniProt"].metadata["gene_names"] == ["FLT3", "CD135"]
    assert by_source["UniProt"].metadata["canonical_target_accession"] == "P36888"
    assert by_source["UniProt"].metadata["canonical_only_scope"] is True
    uniprot_call = next(call for call in client.calls if call[1] == UNIPROT_SEARCH_URL)
    assert uniprot_call[2]["params"]["query"] == "accession:P36888"
    assert uniprot_call[2]["params"]["size"] == 1
    assert by_source["DGIdb"].external_id == "hgnc:3765"
    assert by_source["DGIdb"].metadata["interactions"][0]["drug_name"] == "QUIZARTINIB"
    assert by_source["Pharos"].external_id == "FLT3"
    assert by_source["Pharos"].metadata["tdl"] == "Tclin"

    for record in records:
        assert record.metadata["provenance_class"] == "external_live"
        assert record.metadata["live"] is True
        assert record.metadata["query"]
        assert record.metadata["cache_hit"] is False


def test_pubmed_is_queried_directly_with_search_summary_and_abstract_fetch() -> None:
    client = FakeBiologyClient()
    records = list(
        _retriever(
            client,
            ncbi_tool="DrugPilotTest",
            ncbi_email="team@example.test",
        ).retrieve(_brief(["PubMed"]), _seed(), 1)
    )

    assert [url for _, url, _ in client.calls] == [
        PUBMED_SEARCH_URL,
        PUBMED_SUMMARY_URL,
        PUBMED_FETCH_URL,
        PUBMED_SEARCH_URL,
        PUBMED_SEARCH_URL,
        PUBMED_SEARCH_URL,
    ]
    search_params = client.calls[0][2]["params"]
    summary_params = client.calls[1][2]["params"]
    fetch_params = client.calls[2][2]["params"]
    assert search_params["db"] == "pubmed"
    assert search_params["retmode"] == "json"
    assert search_params["term"].startswith(
        '"FLT3"[Title/Abstract] AND ("Acute myeloid leukemia"[Title/Abstract]'
    )
    assert '"AML"[Title/Abstract] AND (' in search_params["term"]
    assert '"myeloid"[Title/Abstract]' in search_params["term"]
    assert '"leukemia"[Title/Abstract]' in search_params["term"]
    assert '"inhibition"[Title/Abstract]' in search_params["term"]
    assert '"phosphorylation"[Title/Abstract]' in search_params["term"]
    assert '"viability"[Title/Abstract]' in search_params["term"]
    assert '"IC50"[Title/Abstract]' in search_params["term"]
    assert '"activation"[Title/Abstract]' not in search_params["term"]
    assert summary_params["id"] == "12345678"
    assert fetch_params["id"] == "12345678"
    assert fetch_params["rettype"] == "abstract"
    assert fetch_params["retmode"] == "xml"
    assert all(call[2]["params"]["tool"] == "DrugPilotTest" for call in client.calls)
    assert all(call[2]["params"]["email"] == "team@example.test" for call in client.calls)
    assert records[0].metadata["ncbi_email_configured"] is True
    assert "team@example.test" not in str(records[0].metadata)
    assert records[0].source == "PubMed"


def test_modulate_query_searches_both_inhibitory_and_activating_vocabulary() -> None:
    client = FakeBiologyClient()

    list(
        _retriever(client).retrieve(
            _brief(["PubMed"], direction="modulate"),
            _seed(),
            1,
        )
    )

    query = client.calls[0][2]["params"]["term"]
    assert '"modulation"[Title/Abstract]' in query
    assert '"inhibition"[Title/Abstract]' in query
    assert '"blockade"[Title/Abstract]' in query
    assert '"activation"[Title/Abstract]' in query
    assert '"agonist"[Title/Abstract]' in query


def test_open_targets_lookup_rejects_fuzzy_neighbour_target_identity() -> None:
    class AmbiguousOpenTargetsClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, Any]]] = []

        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            self.calls.append(("POST", url, kwargs))
            assert url == OPEN_TARGETS_GRAPHQL_URL
            return FakeResponse(
                {
                    "data": {
                        "search": {
                            "hits": [
                                {
                                    "id": "ENSG00000999999",
                                    "name": "FLT3LG",
                                    "entity": "target",
                                    "description": "Fuzzy neighbour",
                                },
                                {
                                    "id": "ENSG00000122025",
                                    "name": "FLT3",
                                    "entity": "target",
                                    "description": "Canonical target",
                                },
                            ]
                        }
                    }
                }
            )

    client = AmbiguousOpenTargetsClient()
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        ensembl_id="ENSG00000122025",
    )

    records = list(
        _retriever(client).retrieve(
            _brief(["Open Targets"]),
            seed,
            1,
        )
    )

    assert [record.external_id for record in records] == ["ENSG00000122025"]
    assert records[0].metadata["canonical_identity_match"] is True
    assert records[0].metadata["canonical_target_symbol"] == "FLT3"


def test_europe_pmc_query_uses_requested_activation_direction_and_endpoints() -> None:
    client = FakeBiologyClient()
    records = list(
        _retriever(client).retrieve(
            _brief(["Europe PMC"], direction="activate"),
            _seed(),
            1,
        )
    )

    query = client.calls[0][2]["params"]["query"]
    assert query.startswith(
        'TITLE_ABS:"FLT3" AND (TITLE_ABS:"Acute myeloid leukemia"'
    )
    assert 'TITLE_ABS:"AML" AND (' in query
    assert 'TITLE_ABS:"myeloid"' in query
    assert 'TITLE_ABS:"leukemia"' in query
    assert 'TITLE_ABS:"activation"' in query
    assert 'TITLE_ABS:"agonist"' in query
    assert 'TITLE_ABS:"phosphorylation"' in query
    assert 'TITLE_ABS:"IC50"' in query
    assert 'TITLE_ABS:"inhibition"' not in query
    assert records[0].metadata["query_intervention_direction"] == "activate"
    assert records[0].metadata["query_strategy"] == (
        "target-disease-direction-mechanistic-v2"
    )


def test_pubmed_prefers_direct_mechanistic_result_over_prevalence_sentence() -> None:
    client = MechanisticRankingBiologyClient()
    record = list(
        _retriever(client).retrieve(
            _brief(["PubMed"], direction="inhibit"),
            _seed(),
            1,
        )
    )[0]

    assert record.supports == "support"
    assert record.metadata["excerpt"].startswith("Pharmacologic FLT3 inhibition")
    assert "phosphorylation and viability" in record.metadata["excerpt"]
    assert record.metadata["source_location"].startswith(
        "abstract:RESULTS:chars:"
    )
    assert record.metadata["excerpt_selection_policy"] == (
        "same-sentence-target-disease-direction-aware-mechanistic-ranking-v2"
    )
    assert record.metadata["excerpt_relevance_basis"] == [
        "target_symbol",
        "disease",
        "intervention_direction",
        "mechanistic_endpoint",
        "experimental_result_language",
    ]
    assert record.metadata["full_text_stored"] is False
    assert record.excerpt_hash
    assert record.metadata["content_hash"]
    # Primary selection is unchanged; complete abstract context is separately
    # retained for interpretation, including unfavorable/background clauses.
    assert "predict poor survival" not in record.metadata["excerpt"]
    assert "predict poor survival" in record.metadata["abstract_interpretation_context"]["text"]
    assert record.metadata["abstract_interpretation_context"]["authority"] == "interpretation_context_only_not_primary_admission"
    assert len(client.calls) == 5
    assert record.metadata["primary_direction_supplement_status"] == (
        "not_requested"
    )
    assert record.metadata["primary_direction_supplement_stop_reason"] == (
        "main_query_verified_direct_primary_available"
    )
    assert record.metadata["primary_direction_supplement_trace"][
        "http_requests_used"
    ] == 0


def test_multi_target_potentiation_is_context_not_direct_support() -> None:
    record = list(
        _retriever(CombinedResponseModifierClient()).retrieve(
            _brief(["PubMed"], direction="inhibit"),
            _seed(),
            1,
        )
    )[0]

    assert record.supports == "context"
    assert record.metadata["direct_support_eligible"] is False
    assert set(record.metadata["direct_support_rejection_codes"]) >= {
        "multi_target_slash_intervention",
        "combination_or_co_treatment",
        "response_modifier",
        "other_drug_response",
    }
    assert "not direct target-direction support" in record.metadata[
        "evidence_scope"
    ]
    assert "multi_target_slash_intervention" in (
        _direct_support_rejection_codes(
            "AXL/FLT3 inhibition reduced viability in acute myeloid leukemia.",
            "FLT3",
        )
    )


def test_coordinated_combination_noun_phrase_is_not_direct_support() -> None:
    codes = _direct_support_rejection_codes(
        "Primary AML cells showed significantly better responses to combined "
        "menin and FLT3 inhibition than to single-drug or vehicle control "
        "treatment.",
        "FLT3",
    )

    assert "combination_or_co_treatment" in codes


def test_direct_engagement_with_apoptosis_is_literal_positive_direction() -> None:
    excerpt = (
        "XY0206 directly engages FLT3 and suppresses downstream signaling, "
        "resulting in apoptosis in FLT3-ITD AML cells."
    )

    assert "direct_target_intervention_missing" not in (
        _primary_direction_content_rejection_codes(
            _publication_content_from_sections(
                [("RESULTS", excerpt)],
                target_symbol="FLT3",
                disease="Acute myeloid leukemia",
                intervention_direction="inhibit",
            ),
            title="A primary FLT3 AML experiment",
            publication_types=["Journal Article"],
            target_symbol="FLT3",
            disease="Acute myeloid leukemia",
            intervention_direction="inhibit",
        )
    )
    assert _direction_endpoint_polarity_status(excerpt, "inhibit") == "favorable"


def test_pubmed_sensitization_record_is_context_with_explicit_reason() -> None:
    seed = replace(
        _seed(),
        symbol="JAK2",
        target_name="Janus kinase 2",
        ensembl_id="ENSG00000096968",
    )
    record = list(
        _retriever(SensitizationResponseModifierClient()).retrieve(
            _brief(["PubMed"], direction="inhibit"),
            seed,
            1,
        )
    )[0]

    assert record.supports == "context"
    assert record.metadata["direct_support_eligible"] is False
    assert set(record.metadata["direct_support_rejection_codes"]) >= {
        "response_modifier",
        "sensitization_to_other_intervention",
    }


def test_europe_pmc_direct_flags_require_explicit_target_action() -> None:
    records = list(
        _retriever(
            DirectRoleBoundaryEuropePMCClient(),
            enabled_sources=["Europe PMC"],
            max_records_per_source=8,
        ).retrieve(_brief(["Europe PMC"]), _seed(), 1)
    )

    by_id = {record.external_id: record for record in records}
    direct = by_id["boundary-direct"]
    assert direct.supports == "support"
    assert direct.metadata["direct_support_eligible"] is True
    assert direct.metadata["verified_direct_primary"] is True
    assert direct.metadata["direct_support_rejection_codes"] == []

    expected_rejections = {
        "boundary-downstream": "direct_target_intervention_missing",
        "boundary-other-target": "direct_target_intervention_missing",
        "boundary-other-drug": "direct_target_intervention_missing",
        "boundary-modifier": "target_inhibitor_response_modifier",
    }
    for external_id, reason_code in expected_rejections.items():
        record = by_id[external_id]
        assert record.supports == "context"
        assert record.metadata["direct_support_eligible"] is False
        assert record.metadata["verified_direct_primary"] is False
        assert reason_code in record.metadata[
            "direct_support_rejection_codes"
        ]
        assert record.metadata["verified_direct_primary_rejection_codes"] == (
            record.metadata["direct_support_rejection_codes"]
        )


def test_pubmed_accepts_exact_disease_initialism_for_later_mechanistic_result() -> None:
    client = AcronymMechanisticRankingBiologyClient()
    brief = _brief(["PubMed"], direction="inhibit").model_copy(
        update={"disease": "Acute myeloid leukemia (AML)"},
        deep=True,
    )
    record = list(
        _retriever(client).retrieve(
            brief,
            _seed(),
            1,
        )
    )[0]

    query = client.calls[0][2]["params"]["term"]
    assert '"Acute myeloid leukemia (AML)"[Title/Abstract]' in query
    assert '"AML"[Title/Abstract]' in query
    assert record.metadata["excerpt"].startswith("Inhibition of NOX2")
    assert "FLT3-mutant AML cells" in record.metadata["excerpt"]
    assert record.metadata["source_location"].startswith("abstract:RESULTS:chars:")
    assert "predict poor survival" not in record.metadata["excerpt"]
    assert "predict poor survival" in record.metadata["abstract_interpretation_context"]["text"]
    assert record.metadata["abstract_interpretation_context"]["sha256"] == record.metadata["content_hash"]
    assert record.metadata["full_text_stored"] is False
    assert record.excerpt_hash
    assert record.metadata["content_hash"]
    assert record.supports == "context"
    assert "intervention_subject_not_target" in record.metadata[
        "direct_support_rejection_codes"
    ]


def test_derived_disease_acronym_is_anchored_for_literature_query() -> None:
    client = FakeBiologyClient()
    brief = _brief(["PubMed"], direction="inhibit").model_copy(
        update={"disease": "Erdheim-Chester disease"},
        deep=True,
    )

    list(_retriever(client).retrieve(brief, _seed(), 1))

    query = client.calls[0][2]["params"]["term"]
    assert '"Erdheim-Chester disease"[Title/Abstract]' in query
    assert '"ECD"[Title/Abstract] AND (' in query
    assert '"Erdheim-Chester"[Title/Abstract]' in query


def test_mixed_case_clinical_abbreviation_is_not_a_gene_symbol_match() -> None:
    content = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "Lower eGFR was associated with disease severity and renal outcome.",
            )
        ],
        target_symbol="EGFR",
        disease="Erdheim-Chester disease",
        intervention_direction="inhibit",
    )

    assert content.target_lexical_match is False
    assert content.excerpt is None


def test_pubmed_excerpt_ranking_prefers_requested_intervention_direction() -> None:
    record = list(
        _retriever(MechanisticRankingBiologyClient()).retrieve(
            _brief(["PubMed"], direction="activate"),
            _seed(),
            1,
        )
    )[0]

    assert record.supports == "support"
    assert record.metadata["excerpt"].startswith("Enforced FLT3 activation")
    assert "phosphorylation and proliferation" in record.metadata["excerpt"]
    assert "FLT3 inhibition" not in record.metadata["excerpt"]
    assert record.metadata["query_intervention_direction"] == "activate"


def test_bounded_primary_association_supplement_emits_separate_excerpt_record() -> None:
    client = PrimaryAssociationSupplementClient()
    retriever = _retriever(client, max_records_per_source=2)
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association "
                "and direct mechanistic evidence."
            )
        },
        deep=True,
    )

    first = list(retriever.retrieve(brief, _seed(), 1))
    first_call_count = len(client.calls)
    second = list(retriever.retrieve(brief, _seed(), 2))

    assert len(first) == 2
    assert first_call_count == 7
    assert len(client.calls) == first_call_count
    parent = next(
        item for item in first if not item.metadata.get("role_specific_record")
    )
    association = next(
        item
        for item in first
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    assert parent.external_id == "20515557"
    assert association.external_id == "20515557"
    assert association.evidence_id != parent.evidence_id
    assert association.metadata["parent_evidence_id"] == parent.evidence_id
    assert association.metadata["parent_excerpt_hash"] == parent.excerpt_hash
    assert association.metadata["parent_record_in_packet"] is True
    assert association.metadata["parent_provenance_pointer_resolved"] is True
    assert association.metadata["parent_provenance_pointer_type"] == (
        "selected_publication_record_id_not_claim_link"
    )
    assert association.metadata["same_document_external_id"] == "20515557"
    assert association.metadata["association_predicate"] == "mutation_prevalence"
    assert association.metadata["association_scope"] == (
        "activating_or_itd_target_subtype"
    )
    assert association.metadata["source_location"].startswith(
        "abstract:section-1:chars:"
    )
    assert association.metadata["content_hash"]
    assert association.excerpt_hash
    assert association.metadata["additional_network_call"] is True
    assert association.metadata["full_text_stored"] is False
    assert association.metadata["primary_association_supplement_status"] == (
        "completed_with_results"
    )
    trace = association.metadata["primary_association_supplement_trace"]
    assert trace["schema_version"] == "2.0"
    assert trace["query_strategy"] == (
        "brief-derived-precision-first-primary-association-v4"
    )
    assert trace["query_budget"] == 2
    assert trace["query_count_executed"] == 1
    assert trace["http_request_budget"] == 6
    assert trace["http_requests_used"] == 3
    assert trace["raw_document_budget"] == 36
    assert trace["records_screened"] == 1
    assert len(trace["lanes"]) == 1
    assert trace["lanes"][0]["lane_id"] == "title_prevalence"
    assert trace["lanes"][0]["raw_document_budget"] == 20
    assert trace["lanes"][0]["http_request_budget"] == 3
    assert trace["stop_reason"] == (
        "first_verified_required_association_selected"
    )
    assert association.metadata["source_record_candidate_count"] == 3
    assert association.metadata["source_record_budget"] == 2
    assert association.metadata["source_record_dropped_count"] == 1
    assert [item.evidence_id for item in second] == [
        item.evidence_id for item in first
    ]
    assert all(
        item.metadata["origin_provenance_class"] == "external_live"
        and item.metadata["origin_content_hash"]
        for item in second
    )


def test_required_subtype_supplement_pages_past_general_main_association() -> None:
    client = PagedPrimaryAssociationSupplementClient(
        include_required_result=True
    )
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association "
                "and direct mechanistic evidence."
            )
        },
        deep=True,
    )

    records = list(
        _retriever(client, max_records_per_source=3).retrieve(
            brief,
            _seed(),
            1,
        )
    )

    assert len(client.calls) == 9
    supplement_searches = [
        call
        for call in client.calls
        if call[1] == PUBMED_SEARCH_URL
        and "NOT (Review[Publication Type])" in call[2]["params"]["term"]
    ]
    assert [call[2]["params"]["retmax"] for call in supplement_searches] == [
        20,
        16,
    ]
    assert all(
        '"FLT3-ITD"[Title/Abstract]' in call[2]["params"]["term"]
        for call in supplement_searches
    )
    assert '"prevalence"[Title]' in supplement_searches[0][2]["params"]["term"]
    assert (
        'NOT "prevalence"[Title]'
        in supplement_searches[1][2]["params"]["term"]
    )
    association = next(
        item
        for item in records
        if item.external_id == client.supplement_ids[16]
        and item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    trace = association.metadata["primary_association_supplement_trace"]
    assert trace["query_count_executed"] == 2
    assert trace["http_requests_used"] == 6
    assert trace["raw_documents_fetched"] == 24
    assert trace["records_screened"] == 17
    assert trace["variant_clauses"] == ["FLT3-ITD"]
    assert trace["unsearched_result_count"] == 7
    assert [lane["lane_id"] for lane in trace["lanes"]] == [
        "title_prevalence",
        "abstract_primary_association",
    ]
    assert trace["lanes"][0]["raw_documents_fetched"] == 8
    assert trace["lanes"][1]["raw_documents_fetched"] == 16
    assert trace["lanes"][1]["status"] == "completed_with_results"
    assert trace["rejection_counts"] == {
        "primary_association_excerpt_not_found": 16
    }
    assert trace["screening_entries"][-1]["status"] == "selected"


def test_pubmed_cache_is_isolated_by_brief_association_requirement() -> None:
    client = PrimaryAssociationSupplementClient()
    retriever = _retriever(client, max_records_per_source=2)
    general_brief = _brief(["PubMed"])
    subtype_brief = general_brief.model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association."
            )
        },
        deep=True,
    )

    retriever.retrieve(general_brief, _seed(), 1)
    subtype_records = list(retriever.retrieve(subtype_brief, _seed(), 2))
    call_count = len(client.calls)
    cached_subtype_records = list(
        retriever.retrieve(subtype_brief, _seed(), 3)
    )

    assert call_count == 14
    assert len(client.calls) == call_count
    assert any(
        item.external_id == "20515557"
        and item.metadata.get("association_scope")
        == "activating_or_itd_target_subtype"
        for item in subtype_records
    )
    assert [item.evidence_id for item in cached_subtype_records] == [
        item.evidence_id for item in subtype_records
    ]


def test_europe_pmc_cache_is_isolated_by_requirement_in_both_orders() -> None:
    general_brief = _brief(["Europe PMC"])
    subtype_brief = general_brief.model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association."
            )
        },
        deep=True,
    )

    for first_brief, second_brief in (
        (general_brief, subtype_brief),
        (subtype_brief, general_brief),
    ):
        client = FakeBiologyClient()
        retriever = _retriever(
            client,
            enabled_sources=["Europe PMC"],
        )
        first = list(retriever.retrieve(first_brief, _seed(), 1))
        second = list(retriever.retrieve(second_brief, _seed(), 2))
        cached_second = list(
            retriever.retrieve(second_brief, _seed(), 3)
        )

        europe_pmc_calls = [
            call for call in client.calls if call[1] == EUROPE_PMC_SEARCH_URL
        ]
        assert first
        assert second
        assert len(europe_pmc_calls) == 2
        assert [item.evidence_id for item in cached_second] == [
            item.evidence_id for item in second
        ]
        assert all(item.metadata["cache_hit"] for item in cached_second)


def test_main_pubmed_non_primary_association_cannot_suppress_precision_lanes() -> None:
    cases = [
        (
            "Review of FLT3 prevalence in AML",
            ["Review"],
            "review_publication_type",
        ),
        (
            "Systematic review of FLT3 prevalence in AML",
            ["Journal Article"],
            "review_title",
        ),
        (
            "FLT3 prevalence in AML",
            [],
            "publication_type_missing",
        ),
    ]
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association."
            )
        },
        deep=True,
    )

    for title, publication_types, expected_code in cases:
        client = InvalidMainAssociationMetadataClient(
            title=title,
            publication_types=publication_types,
        )
        records = list(
            _retriever(client, max_records_per_source=2).retrieve(
                brief,
                _seed(),
                1,
            )
        )
        association_ids = {
            item.external_id
            for item in records
            if item.metadata.get("role_specific_record")
            == "primary_target_disease_association_excerpt"
        }
        assert association_ids == {"20515557"}
        parent = next(
            item
            for item in records
            if item.external_id == "20515557"
            and not item.metadata.get("role_specific_record")
        )
        association = next(
            item
            for item in records
            if item.metadata.get("role_specific_record")
            == "primary_target_disease_association_excerpt"
        )
        assert association.metadata["parent_evidence_id"] == parent.evidence_id
        assert association.metadata["parent_record_in_packet"] is True
        assert association.metadata[
            "source_record_dropped_primary_association_rejection_counts"
        ][expected_code] == 1
        supplement_searches = [
            call
            for call in client.calls
            if call[1] == PUBMED_SEARCH_URL
            and "NOT (Review[Publication Type])" in call[2]["params"]["term"]
        ]
        assert len(supplement_searches) == 1


def test_main_pubmed_title_only_record_cannot_suppress_precision_lanes() -> None:
    client = MainTitleOnlyRecordClient()
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association."
            )
        },
        deep=True,
    )

    records = list(
        _retriever(client, max_records_per_source=2).retrieve(
            brief,
            _seed(),
            1,
        )
    )

    association_ids = {
        item.external_id
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    }
    assert association_ids == {"20515557"}
    parent = next(
        item
        for item in records
        if item.external_id == "20515557"
        and not item.metadata.get("role_specific_record")
    )
    association = next(
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    assert parent.metadata["content_availability"] == "abstract_excerpt"
    assert association.metadata["parent_evidence_id"] == parent.evidence_id
    supplement_searches = [
        call
        for call in client.calls
        if call[1] == PUBMED_SEARCH_URL
        and "NOT (Review[Publication Type])" in call[2]["params"]["term"]
    ]
    assert len(supplement_searches) == 1


def test_main_europe_pmc_review_association_child_is_not_support() -> None:
    records = list(
        _retriever(
            EuropePMCReviewAssociationClient(),
            max_records_per_source=2,
        ).retrieve(_brief(["Europe PMC"]), _seed(), 1)
    )

    assert len(records) == 1
    assert not records[0].metadata.get("role_specific_record")
    assert records[0].supports == "context"
    assert records[0].metadata["primary_association_candidate_status"] == (
        "rejected"
    )
    assert records[0].metadata[
        "primary_association_candidate_rejection_codes"
    ] == ["review_publication_type", "review_title"]


def test_main_subtype_mismatch_cannot_displace_required_supplement_result() -> None:
    client = MainSubtypeMismatchClient()
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Establish the primary cohort association of FLT3-ITD and an "
                "FLT3 activating mutation in AML."
            )
        },
        deep=True,
    )

    requirement = _primary_association_requirement(brief, "FLT3")
    assert {clause.kind for clause in requirement.variant_clauses} >= {
        "internal_tandem_duplication",
        "activating_variant",
    }

    records = list(
        _retriever(client, max_records_per_source=2).retrieve(
            brief,
            _seed(),
            1,
        )
    )

    association_records = [
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    ]
    assert [item.external_id for item in association_records] == ["20515557"]
    parent = next(
        item
        for item in records
        if item.external_id == "20515557"
        and not item.metadata.get("role_specific_record")
    )
    assert association_records[0].metadata["parent_evidence_id"] == (
        parent.evidence_id
    )
    assert association_records[0].metadata["parent_record_in_packet"] is True
    assert association_records[0].metadata[
        "primary_association_supplement_status"
    ] == "completed_with_results"


def test_primary_association_budget_exhaustion_is_bounded_and_traceable() -> None:
    client = PagedPrimaryAssociationSupplementClient(
        include_required_result=False
    )
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association."
            )
        },
        deep=True,
    )

    first = list(
        _retriever(client, max_records_per_source=3).retrieve(
            brief,
            _seed(),
            1,
        )
    )

    assert len(client.calls) == 9
    assert not any(
        item.metadata.get("association_scope")
        == "activating_or_itd_target_subtype"
        for item in first
    )
    trace = first[0].metadata["primary_association_supplement_trace"]
    assert trace["status"] == "completed_bounded_without_verified_result"
    assert trace["stop_reason"] == (
        "raw_document_budget_exhausted_without_required_association"
    )
    assert trace["query_count_executed"] == 2
    assert trace["http_requests_used"] == 6
    assert trace["raw_documents_fetched"] == 24
    assert trace["records_screened"] == 24
    assert len(trace["screening_entries"]) == 24
    assert trace["rejection_counts"] == {
        "primary_association_excerpt_not_found": 24
    }
    assert [lane["raw_document_budget"] for lane in trace["lanes"]] == [
        20,
        16,
    ]
    assert all(
        lane["http_requests_used"] == 3
        for lane in trace["lanes"]
    )
    assert trace["full_text_stored"] is False


def test_primary_association_lanes_deduplicate_before_fetch_and_screen() -> None:
    client = OverlappingPrimaryAssociationLanesClient()
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association."
            )
        },
        deep=True,
    )

    records = list(
        _retriever(client, max_records_per_source=3).retrieve(
            brief,
            _seed(),
            1,
        )
    )

    trace = records[0].metadata["primary_association_supplement_trace"]
    assert trace["http_requests_used"] == 6
    assert trace["returned_id_count"] == 24
    assert trace["excluded_prior_lane_id_count"] == 4
    assert trace["raw_documents_fetched"] == 20
    assert trace["records_screened"] == 20
    screened_ids = [
        entry["external_id"] for entry in trace["screening_entries"]
    ]
    assert len(screened_ids) == len(set(screened_ids)) == 20
    assert trace["lanes"][1]["excluded_prior_lane_id_count"] == 4
    assert trace["lanes"][1]["raw_documents_fetched"] == 12
    assert trace["stop_reason"] == (
        "returned_candidates_exhausted_without_required_association"
    )


def test_variant_query_is_derived_from_non_fixture_target_brief() -> None:
    brief = ResearchBrief(
        research_goal=(
            "Establish the primary cohort association of BRAF V600E in melanoma."
        ),
        disease="melanoma",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(allowed_sources=["PubMed"]),
    )
    seed = TargetSeed(
        symbol="BRAF",
        target_name="B-Raf proto-oncogene",
        track=HypothesisTrack.LITERATURE,
        priority=0.9,
    )

    query = PublicBiologyRetriever._pubmed_primary_association_query(
        brief,
        seed,
    )
    lanes = PublicBiologyRetriever._pubmed_primary_association_query_lanes(
        brief,
        seed,
    )

    assert '"BRAF V600E"[Title/Abstract]' in query
    assert '"BRAF-V600E"[Title/Abstract]' in query
    assert '"melanoma"[Title/Abstract]' in query
    assert '"prevalence"[Title]' in query
    assert [lane.raw_document_budget for lane in lanes] == [20, 16]
    assert [lane.lane_id for lane in lanes] == [
        "title_prevalence",
        "abstract_primary_association",
    ]
    assert 'NOT "prevalence"[Title]' in lanes[1].query
    assert all("AND NOT" not in lane.query for lane in lanes)
    prevalence_exclusion = " NOT (Review[Publication Type])"
    prevalence_positive = lanes[0].query[: -len(prevalence_exclusion)]
    assert lanes[0].query.endswith(prevalence_exclusion)
    assert prevalence_positive.startswith("(")
    assert prevalence_positive.endswith(")")
    other_exclusions = (
        ' NOT "prevalence"[Title] NOT (Review[Publication Type])'
    )
    other_positive = lanes[1].query[: -len(other_exclusions)]
    assert lanes[1].query.endswith(other_exclusions)
    assert other_positive.startswith("(")
    assert other_positive.endswith(")")
    assert '"mutations"[Title/Abstract]' not in lanes[0].query
    assert '"detected"[Title/Abstract]' not in lanes[0].query
    assert "PMID" not in query.upper()


def test_same_document_mechanism_and_association_use_distinct_immutable_records() -> None:
    client = SameDocumentDualRoleClient()
    records = list(
        _retriever(client, max_records_per_source=2).retrieve(
            _brief(["PubMed"]),
            _seed(),
            1,
        )
    )

    assert len(records) == 2
    assert len(client.calls) == 3
    parent = next(
        item for item in records if not item.metadata.get("role_specific_record")
    )
    association = next(
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    assert parent.external_id == association.external_id == "12345678"
    assert parent.evidence_id != association.evidence_id
    assert association.metadata["parent_evidence_id"] == parent.evidence_id
    assert association.metadata["same_document_external_id"] == parent.external_id
    assert association.metadata["content_hash"] == parent.metadata["content_hash"]
    assert association.excerpt_hash != parent.excerpt_hash
    assert association.metadata["excerpt"] != parent.metadata["excerpt"]
    assert association.metadata["source_location"] != parent.metadata["source_location"]
    assert association.metadata["additional_network_call"] is False
    assert association.metadata["full_text_stored"] is False
    assert parent.metadata["source_record_candidate_count"] == 2
    assert association.metadata["source_record_candidate_count"] == 2


def test_primary_association_selector_rejects_background_resistance_and_other_targets() -> None:
    positive = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "In our cohort of 123 acute myeloid leukemia patients, "
                "FLT3-ITD mutations were detected in 24% of cases.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    amino_variant = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "In our cohort of 200 melanoma patients, BRAF V600E was "
                "detected in 45% of cases.",
            )
        ],
        target_symbol="BRAF",
        disease="melanoma",
        intervention_direction="inhibit",
    )
    background = _publication_content_from_sections(
        [
            (
                "BACKGROUND",
                "FLT3-ITD mutations occur in approximately 30% of acute "
                "myeloid leukemia cases.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    resistance = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "FLT3-ITD acute myeloid leukemia cell lines harbored acquired "
                "resistance mutations after gilteritinib treatment.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    rsk1 = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "We identified RSK1 dependency in FLT3-ITD acute myeloid "
                "leukemia cell lines.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    atr = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "We found ATR dependency in FLT3-ITD acute myeloid leukemia "
                "samples.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    target_first_rsk1 = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "FLT3-ITD acute myeloid leukemia cell lines showed RSK1 "
                "dependency in this study.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    target_first_atr = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "FLT3-mutant acute myeloid leukemia samples showed an ATR "
                "dependency in this study.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    target_first_other_expression = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "FLT3-mutant acute myeloid leukemia samples showed increased "
                "RSK1 expression compared with controls.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    target_first_other_prevalence = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "FLT3-ITD acute myeloid leukemia samples showed RSK1 mutation "
                "prevalence of 20% in this cohort.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    target_first_other_association = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "FLT3-ITD acute myeloid leukemia samples showed RSK1 was "
                "associated with apoptosis in this cohort.",
            )
        ],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )

    assert positive.association_predicate == "mutation_prevalence"
    assert positive.association_scope == "activating_or_itd_target_subtype"
    assert positive.association_excerpt
    assert amino_variant.association_predicate == "mutation_prevalence"
    assert amino_variant.association_scope == "activating_or_itd_target_subtype"
    assert amino_variant.association_excerpt
    assert background.association_excerpt is None
    assert resistance.association_excerpt is None
    assert rsk1.association_excerpt is None
    assert atr.association_excerpt is None
    assert target_first_rsk1.association_excerpt is None
    assert target_first_atr.association_excerpt is None
    assert target_first_other_expression.association_excerpt is None
    assert target_first_other_prevalence.association_excerpt is None
    assert target_first_other_association.association_excerpt is None


def test_primary_association_accepts_strict_positive_quantified_cohort_forms() -> None:
    cases = [
        (
            "section-1",
            "Among the AML patients, a FLT3 internal tandem duplication "
            "(FLT3/ITD) mutation was present in 59 cases (15.8%).",
        ),
        (
            "RESULTS",
            "Among acute myeloid leukemia patients, FLT3-ITD mutations were "
            "present in 59 of 200 cases (29.5%).",
        ),
        (
            "RESULTS",
            "In acute myeloid leukemia, FLT3-ITD mutations were present in "
            "59 of 373 (15.8%).",
        ),
        (
            "RESULTS",
            "In our cohort of 200 acute myeloid leukemia patients, FLT3-ITD "
            "mutations were present in 59 cases (29.5%).",
        ),
        (
            "RESULTS",
            "In 200 acute myeloid leukemia patients, FLT3-ITD mutations were "
            "present in 59 cases (29.5%).",
        ),
        (
            "RESULTS",
            "In our acute myeloid leukemia cohort (n=200), FLT3-ITD mutations "
            "were present in 59 cases (29.5%).",
        ),
        (
            "RESULTS",
            "In 200 patients with acute myeloid leukemia, FLT3-ITD mutations "
            "were present in 59 cases (29.5%).",
        ),
        (
            "RESULTS",
            "Of 200 cases with acute myeloid leukemia, FLT3-ITD mutations were "
            "present in 59 cases (29.5%).",
        ),
        (
            "RESULTS",
            "FLT3-ITD mutations were present in 30 cases (40.0%) among acute "
            "myeloid leukemia patients.",
        ),
    ]

    for label, sentence in cases:
        content = _publication_content_from_sections(
            [(label, sentence)],
            target_symbol="FLT3",
            disease="Acute myeloid leukemia",
            intervention_direction="inhibit",
        )
        assert content.association_predicate == "mutation_prevalence"
        assert content.association_scope == "activating_or_itd_target_subtype"
        assert content.association_excerpt == sentence


def test_primary_association_binds_subtype_subject_in_association_relation() -> None:
    sentence = (
        "The association between FLT3-ITD and acute myeloid leukemia was "
        "significant in our cohort of 100 patients."
    )

    content = _publication_content_from_sections(
        [("RESULTS", sentence)],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )

    assert content.association_predicate == "disease_target_association"
    assert content.association_scope == "activating_or_itd_target_subtype"
    assert content.association_excerpt == sentence


def test_exact_subtype_expression_requires_the_variant_as_grammar_subject() -> None:
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Establish the primary cohort association of FLT3-ITD in AML."
            )
        },
        deep=True,
    )
    requirement = _primary_association_requirement(brief, "FLT3")
    itd_clause = next(
        clause
        for clause in requirement.variant_clauses
        if clause.kind == "internal_tandem_duplication"
    )

    for token in ("FLT3-ITD", "FLT3/ITD", "FLT3–ITD", "FLT3−ITD"):
        sentence = (
            f"In acute myeloid leukemia patients, {token} expression was "
            "significantly higher than in controls."
        )
        content = _publication_content_from_sections(
            [("RESULTS", sentence)],
            target_symbol="FLT3",
            disease="Acute myeloid leukemia",
            intervention_direction="inhibit",
        )
        assert content.association_predicate == "disease_target_expression", token
        assert content.association_scope == "activating_or_itd_target_subtype", token
        assert _variant_clause_is_association_subject(
            content.association_excerpt or "",
            target_symbol="FLT3",
            clause=itd_clause,
            predicate="disease_target_expression",
        )

    unrelated = (
        "In acute myeloid leukemia patients, FLT3-ITD was infrequent whereas "
        "FLT3 expression was significantly higher than in controls."
    )
    unrelated_content = _publication_content_from_sections(
        [("RESULTS", unrelated)],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    assert unrelated_content.association_predicate == "disease_target_expression"
    assert unrelated_content.association_scope == "target_general"
    assert not _variant_clause_is_association_subject(
        unrelated_content.association_excerpt or "",
        target_symbol="FLT3",
        clause=itd_clause,
        predicate="disease_target_expression",
    )

    other_gene = (
        "Acute myeloid leukemia FLT3-ITD samples showed higher NPM1 expression."
    )
    assert not _variant_clause_is_association_subject(
        other_gene,
        target_symbol="FLT3",
        clause=itd_clause,
        predicate="disease_target_expression",
    )


def test_non_target_itd_prefix_is_not_a_positive_exact_subtype() -> None:
    for token in ("non-FLT3-ITD", "non FLT3 ITD"):
        sentence = (
            f"Among acute myeloid leukemia patients, {token} mutations were "
            "present in 20 cases (20.0%)."
        )
        content = _publication_content_from_sections(
            [("RESULTS", sentence)],
            target_symbol="FLT3",
            disease="Acute myeloid leukemia",
            intervention_direction="inhibit",
        )
        assert content.association_excerpt is None, token


def test_numeric_continuation_stops_at_a_new_gene_or_variant_subject() -> None:
    sentences = [
        (
            "Among 100 acute myeloid leukemia patients, FLT3-ITD mutations "
            "were present in 20 cases (20.0%), and NPM1 mutations were present "
            "in 10 of 50 cases (80.0%)."
        ),
        (
            "Among 100 acute myeloid leukemia patients, FLT3-ITD mutations "
            "were present in 20 cases (20.0%), and 10 cases (80.0%) in the "
            "NPM1-mutant subgroup."
        ),
        (
            "Among 100 acute myeloid leukemia patients, FLT3-ITD mutations "
            "were present in 20 cases (20.0%), and 10 cases (80.0%) in the "
            "D835Y subgroup."
        ),
    ]

    for sentence in sentences:
        content = _publication_content_from_sections(
            [("RESULTS", sentence)],
            target_symbol="FLT3",
            disease="Acute myeloid leukemia",
            intervention_direction="inhibit",
        )
        assert content.association_predicate == "mutation_prevalence", sentence
        assert content.association_excerpt == sentence, sentence


def test_primary_association_quantified_cohort_adversarial_matrix_fails_closed() -> None:
    cases = [
        (
            "RESULTS",
            "Among AML patients, FLT3-ITD mutations were present in 0 of 200 "
            "cases (0.0%).",
        ),
        (
            "RESULTS",
            "Among AML patients, FLT3-ITD mutations were present in 00 cases "
            "(1.0%).",
        ),
        (
            "RESULTS",
            "Among AML patients, FLT3-ITD mutations were present in 1 of 0 "
            "(1.0%).",
        ),
        (
            "RESULTS",
            "Among AML patients, FLT3-ITD mutations were present in 201 of "
            "200 cases (99.0%).",
        ),
        (
            "RESULTS",
            "Among AML patients, FLT3-ITD mutations were present in 59 cases "
            "(115.8%).",
        ),
        (
            "RESULTS",
            "Among AML patients, FLT3-ITD mutations were present in 1 of 10 "
            "cases (80.0%).",
        ),
        (
            "RESULTS",
            "No FLT3-ITD mutations were present in 59 of 200 AML cases "
            "(29.5%).",
        ),
        (
            "BACKGROUND",
            "Among AML patients, FLT3-ITD mutations were present in 59 of 200 "
            "cases (29.5%).",
        ),
        (
            "RESULTS",
            "Following treatment, FLT3-ITD mutations were present in 59 of "
            "200 acute myeloid leukemia cases (29.5%).",
        ),
        (
            "RESULTS",
            "Among AML patients with FLT3-ITD, NPM1 mutations were present in "
            "59 of 200 cases (29.5%).",
        ),
        (
            "RESULTS",
            "FLT3-ITD mutations were present in 59 of 200 cases (29.5%). "
            "The cohort had acute myeloid leukemia.",
        ),
        (
            "RESULTS",
            "FLT3-ITD mutations were present in 3 of 10 acute myeloid leukemia "
            "cell lines (30.0%).",
        ),
        (
            "RESULTS",
            "An FLT3-ITD assay was positive in 59 of 200 acute myeloid leukemia "
            "samples (29.5%).",
        ),
        (
            "RESULTS",
            "In our cohort of 100 AML patients, there was no association "
            "between FLT3 and acute myeloid leukemia.",
        ),
        (
            "RESULTS",
            "FLT3 was not associated with acute myeloid leukemia in 100 "
            "patients.",
        ),
        (
            "RESULTS",
            "FLT3 was not correlated with acute myeloid leukemia in 100 "
            "patients.",
        ),
        (
            "RESULTS",
            "FLT3 did not correlate with acute myeloid leukemia in 100 "
            "patients.",
        ),
        (
            "RESULTS",
            "FLT3 expression was not higher in acute myeloid leukemia "
            "patients.",
        ),
        (
            "RESULTS",
            "In our cohort of 100 acute lymphoblastic leukemia patients, "
            "FLT3-ITD mutations were present in 20 cases (20.0%).",
        ),
        (
            "RESULTS",
            "In our cohort of 200 acute myeloid leukemia patients, FLT3-ITD "
            "mutations were present in 59 cases (80.0%).",
        ),
        (
            "RESULTS",
            "Among 100 acute myeloid leukemia patients, NPM1 mutations were "
            "present in 20 cases (20.0%), whereas FLT3-ITD mutations were "
            "present in 10 of 100 cases (80.0%).",
        ),
        (
            "RESULTS",
            "Among 100 acute myeloid leukemia patients, FLT3-ITD mutations "
            "were present in 20 cases (20.0%), and FLT3-ITD mutations were "
            "present in 10 of 100 cases (80.0%).",
        ),
        (
            "RESULTS",
            "Among 100 acute myeloid leukemia patients, FLT3-ITD mutations "
            "were present in 20 cases (20.0%) and 10 of 100 cases (80.0%).",
        ),
        (
            "RESULTS",
            "Among 200 acute myeloid leukemia patients, FLT3-ITD mutations "
            "were present in 59 of 100 cases (59.0%).",
        ),
        (
            "RESULTS",
            "In 200 acute myeloid leukemia patients, FLT3-ITD mutations were "
            "present in 59 cases (80.0%).",
        ),
        (
            "RESULTS",
            "Of 200 acute myeloid leukemia patients, FLT3-ITD mutations were "
            "present in 59 cases (80.0%).",
        ),
        (
            "RESULTS",
            "In our acute myeloid leukemia cohort (n=200), FLT3-ITD mutations "
            "were present in 59 cases (80.0%).",
        ),
        (
            "RESULTS",
            "In 200 patients with acute myeloid leukemia, FLT3-ITD mutations "
            "were present in 59 cases (80.0%).",
        ),
        (
            "RESULTS",
            "Of 200 samples with acute myeloid leukemia, FLT3-ITD mutations "
            "were present in 59 cases (80.0%).",
        ),
        (
            "RESULTS",
            "In our cohort of 100 acute myeloid leukemia patients, FLT3 D835Y "
            "mutations were present in 20 cases (20.0%), whereas "
            "FLT3-ITD-negative status was documented.",
        ),
        (
            "RESULTS",
            "In acute myeloid leukemia patients, the association between FLT3 "
            "and disease was non-significant.",
        ),
        (
            "RESULTS",
            "In acute myeloid leukemia patients, the FLT3 association was "
            "nonsignificant.",
        ),
        (
            "RESULTS",
            "In acute myeloid leukemia patients, an insignificant association "
            "between FLT3 and disease was observed.",
        ),
        (
            "RESULTS",
            "In acute myeloid leukemia patients, there was an absence of "
            "association between FLT3 and disease.",
        ),
        (
            "RESULTS",
            "In acute myeloid leukemia patients, the association between FLT3 "
            "and disease was absent.",
        ),
        (
            "RESULTS",
            "In acute myeloid leukemia patients, we cannot establish an "
            "association between FLT3 and disease.",
        ),
        (
            "RESULTS",
            "Among acute myeloid leukemia patients, FLT3-ITD(-) mutations were "
            "present in 20 cases (20.0%).",
        ),
        (
            "RESULTS",
            "Among acute myeloid leukemia patients, FLT3-ITD(−) mutations were "
            "present in 20 cases (20.0%).",
        ),
        (
            "RESULTS",
            "Among acute myeloid leukemia patients, FLT3 non-ITD mutations were "
            "present in 20 cases (20.0%).",
        ),
        (
            "RESULTS",
            "Among acute myeloid leukemia patients, FLT3 non–ITD mutations were "
            "present in 20 cases (20.0%).",
        ),
        (
            "RESULTS",
            "In our cohort of 100 acute myeloid leukemia patients, there was "
            "no evidence of an association between FLT3 and acute myeloid "
            "leukemia.",
        ),
        (
            "RESULTS",
            "In our cohort of 100 acute myeloid leukemia patients, the "
            "association between FLT3 and acute myeloid leukemia was not "
            "statistically significant.",
        ),
        (
            "RESULTS",
            "In our cohort of 100 acute myeloid leukemia patients, FLT3 did "
            "not show an association with acute myeloid leukemia.",
        ),
        (
            "RESULTS",
            "In our cohort of 100 acute myeloid leukemia patients, FLT3 did "
            "not demonstrate an association with acute myeloid leukemia.",
        ),
    ]

    for label, sentence in cases:
        content = _publication_content_from_sections(
            [(label, sentence)],
            target_symbol="FLT3",
            disease="Acute myeloid leukemia",
            intervention_direction="inhibit",
        )
        assert content.association_excerpt is None, sentence


def test_primary_association_null_relation_phrases_are_explicitly_blocked() -> None:
    phrases = [
        "In our cohort of 100 acute myeloid leukemia patients, there was no "
        "evidence of an association between FLT3 and acute myeloid leukemia.",
        "In our cohort of 100 acute myeloid leukemia patients, the association "
        "between FLT3 and acute myeloid leukemia was not statistically significant.",
        "In our cohort of 100 acute myeloid leukemia patients, FLT3 did not show "
        "an association with acute myeloid leukemia.",
        "In our cohort of 100 acute myeloid leukemia patients, FLT3 did not "
        "demonstrate an association with acute myeloid leukemia.",
        "In our cohort of 100 acute myeloid leukemia patients, evidence for the "
        "correlation between FLT3 and acute myeloid leukemia was insufficient.",
        "In our cohort of 100 acute myeloid leukemia patients, the association "
        "between FLT3 and acute myeloid leukemia lacked statistical significance.",
        "In our cohort of 100 acute myeloid leukemia patients, the association "
        "between FLT3 and acute myeloid leukemia failed to demonstrate statistical "
        "significance.",
        "In our cohort of 100 acute myeloid leukemia patients, the association "
        "between FLT3 and acute myeloid leukemia did not reach statistical "
        "significance.",
        "In acute myeloid leukemia, the FLT3 association was non-significant.",
        "In acute myeloid leukemia, the FLT3 association was nonsignificant.",
        "In acute myeloid leukemia, an insignificant association involved FLT3.",
        "In acute myeloid leukemia, there was an absence of association with FLT3.",
        "In acute myeloid leukemia, the association with FLT3 was absent.",
        "In acute myeloid leukemia, we cannot establish an association with FLT3.",
    ]

    for phrase in phrases:
        assert _PRIMARY_ASSOCIATION_NEGATIVE_RELATION_PATTERN.search(phrase), phrase
        content = _publication_content_from_sections(
            [("RESULTS", phrase)],
            target_symbol="FLT3",
            disease="Acute myeloid leukemia",
            intervention_direction="inhibit",
        )
        assert content.association_excerpt is None, phrase


def test_null_guard_is_bound_to_the_target_predicate_clause() -> None:
    separators = [";", "whereas", "while", "but"]
    positive = (
        "FLT3-ITD mutations were present in 59 of 200 cases (29.5%)"
    )
    unrelated_null = "age was not different between groups"
    lead = "Among 200 acute myeloid leukemia patients,"

    for separator in separators:
        for sentence in (
            f"{lead} {unrelated_null} {separator} {positive}.",
            f"{lead} {positive} {separator} {unrelated_null}.",
        ):
            content = _publication_content_from_sections(
                [("RESULTS", sentence)],
                target_symbol="FLT3",
                disease="Acute myeloid leukemia",
                intervention_direction="inhibit",
            )
            assert content.association_predicate == "mutation_prevalence", sentence
            assert content.association_excerpt == sentence, sentence

    other_gene_sentence = (
        "Among 200 acute myeloid leukemia patients, NPM1 was not associated "
        "with disease; FLT3-ITD mutations were present in 59 cases (29.5%)."
    )
    other_gene = _publication_content_from_sections(
        [("RESULTS", other_gene_sentence)],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    assert other_gene.association_predicate == "mutation_prevalence"


def test_required_variant_review_and_metadata_boundaries_remain_strict() -> None:
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Establish the primary cohort association of FLT3-ITD in AML."
            )
        },
        deep=True,
    )
    requirement = _primary_association_requirement(brief, "FLT3")
    assert requirement.variant_clauses
    itd_clause = requirement.variant_clauses[0]
    mismatch = (
        "Among AML patients, FLT3 D835Y mutations were present in 59 of 200 "
        "cases (29.5%)."
    )
    assert not _excerpt_matches_variant_clause(
        mismatch,
        target_symbol="FLT3",
        clause=itd_clause,
    )
    assert not _excerpt_matches_variant_clause(
        "FLT3-ITD-negative status was documented in the AML cohort.",
        target_symbol="FLT3",
        clause=itd_clause,
    )
    for negative_token in (
        "FLT3-ITD(-)",
        "FLT3-ITD(−)",
        "FLT3 non-ITD",
        "FLT3 non–ITD",
        "FLT3−ITD-negative",
    ):
        assert not _excerpt_matches_variant_clause(
            f"{negative_token} status was documented in the AML cohort.",
            target_symbol="FLT3",
            clause=itd_clause,
        ), negative_token
    for positive_token in ("FLT3/ITD", "FLT3–ITD", "FLT3−ITD"):
        assert _excerpt_matches_variant_clause(
            f"{positive_token} was documented in the AML cohort.",
            target_symbol="FLT3",
            clause=itd_clause,
        ), positive_token
    assert _primary_publication_metadata_rejection_codes(
        "FLT3 variants in AML",
        ["Review"],
    ) == ("review_publication_type",)
    assert _primary_publication_metadata_rejection_codes(
        "Systematic review of FLT3 variants in AML",
        ["Journal Article"],
    ) == ("review_title",)
    assert _primary_publication_metadata_rejection_codes(
        "FLT3 variants in AML",
        ["Journal Article", "Meta-Analysis"],
    ) == ("non_primary_synthesis_publication_type",)
    assert _primary_publication_metadata_rejection_codes(
        "FLT3 variants in AML",
        ["Journal Article", "Practice Guideline"],
    ) == ("non_primary_synthesis_publication_type",)
    assert _primary_publication_metadata_rejection_codes(
        "FLT3 variants in AML",
        ["Journal Article", "Consensus Development Conference"],
    ) == ("non_primary_synthesis_publication_type",)
    metadata_only = _publication_content_from_sections(
        [],
        target_symbol="FLT3",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    assert metadata_only.association_excerpt is None


def test_quantified_prevalence_is_emitted_as_association_role_only() -> None:
    records = list(
        _retriever(
            AssociationOnlyMainQueryClient(),
            max_records_per_source=2,
        ).retrieve(_brief(["PubMed"]), _seed(), 1)
    )

    parent = next(
        item for item in records if not item.metadata.get("role_specific_record")
    )
    association = next(
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    assert parent.supports == "context"
    assert parent.metadata["direct_support_eligible"] is False
    assert parent.metadata["classification_policy"] == "association-role-only-v1"
    assert association.supports == "support"
    assert association.metadata["direct_support_eligible"] is False
    assert "not intervention-direction proof" in association.metadata[
        "evidence_scope"
    ]


def test_distinct_parent_association_sentence_is_never_direct_support() -> None:
    records = list(
        _retriever(
            TwoAssociationSentenceMainQueryClient(),
            max_records_per_source=2,
        ).retrieve(_brief(["PubMed"]), _seed(), 1)
    )

    parent = next(
        item for item in records if not item.metadata.get("role_specific_record")
    )
    association = next(
        item
        for item in records
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    assert parent.metadata["excerpt"] != association.metadata["excerpt"]
    assert "expression was significantly higher" in parent.metadata["excerpt"]
    assert "present in 59 of 200 cases" in association.metadata["excerpt"]
    assert parent.supports == "context"
    assert parent.metadata["direct_support_eligible"] is False
    assert parent.metadata["classification_policy"] == (
        "primary-direction-validation-failed-v1"
    )
    assert association.supports == "support"
    assert association.metadata["direct_support_eligible"] is False


def test_publication_cache_isolated_by_intervention_direction() -> None:
    client = FakeBiologyClient()
    retriever = _retriever(client)
    inhibit_brief = _brief(["PubMed"], direction="inhibit").model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["PubMed"],
                require_primary_sources=False,
            )
        },
        deep=True,
    )
    activate_brief = _brief(["PubMed"], direction="activate").model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["PubMed"],
                require_primary_sources=False,
            )
        },
        deep=True,
    )

    retriever.retrieve(inhibit_brief, _seed(), 1)
    retriever.retrieve(activate_brief, _seed(), 2)

    assert len(client.calls) == 6
    search_terms = [
        call[2]["params"]["term"]
        for call in client.calls
        if call[1] == PUBMED_SEARCH_URL
        and not str(call[2]["params"]["term"]).endswith(
            "NOT review[Publication Type]"
        )
    ]
    assert len(search_terms) == 2
    assert '"inhibition"[Title/Abstract]' in search_terms[0]
    assert '"activation"[Title/Abstract]' in search_terms[1]


def test_pubmed_abstract_without_direct_target_disease_sentence_is_metadata_only() -> None:
    records = list(
        _retriever(PubMedNoRelatedExcerptClient()).retrieve(
            _brief(["PubMed"]),
            _seed(),
            1,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record.supports == "context"
    assert record.excerpt_hash is None
    assert record.metadata["content_availability"] == "metadata_only"
    assert record.metadata["abstract_retrieval_status"] == "available"
    assert record.metadata["excerpt_selection_status"] == "related_excerpt_not_found"
    assert record.metadata["excerpt"] is None
    assert record.metadata["content_hash"]
    assert "EGFR signaling" not in str(record.metadata)


def test_pubmed_abstract_failure_keeps_metadata_but_never_claims_content_was_read() -> None:
    records = list(
        _retriever(PubMedAbstractUnavailableClient()).retrieve(
            _brief(["PubMed"]),
            _seed(),
            1,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record.supports == "context"
    assert record.excerpt_hash is None
    assert record.metadata["content_availability"] == "metadata_only"
    assert record.metadata["abstract_retrieval_status"] == "unavailable"
    assert record.metadata["abstract_retrieval_error_type"] == "ConnectTimeout"
    assert record.metadata["content_hash"] is None


def test_pubmed_counter_signal_is_never_promoted_to_support() -> None:
    records = list(
        _retriever(
            PubMedCounterSignalClient(),
            counter_evidence_enabled=True,
        ).retrieve(_brief(["PubMed"]), _seed(), 1)
    )

    assert len(records) == 1
    record = records[0]
    assert record.supports == "counter"
    assert record.metadata["excerpt_selection_status"] == "selected_counter_signal"
    assert record.metadata["counter_category"] == "resistance"
    assert record.metadata["review_status"] == "researcher_review_required"
    assert "not support evidence" in record.metadata["evidence_scope"]

    context_only = list(
        _retriever(
            PubMedCounterSignalClient(),
            counter_evidence_enabled=False,
        ).retrieve(_brief(["PubMed"]), _seed(), 1)
    )[0]
    assert context_only.supports == "context"


def test_counter_signal_must_be_linked_to_the_requested_target_intervention() -> None:
    assert _classify_counter_signal(
        "Hydroxyurea resistance in JAK2-positive disease",
        "Patients with JAK2 V617F were resistant to hydroxyurea.",
        "JAK2",
        intervention_direction="inhibit",
    ) is None

    classified = _classify_counter_signal(
        "Resistance to JAK2 inhibition",
        "Acquired resistance to JAK2 inhibition limited durable response.",
        "JAK2",
        intervention_direction="inhibit",
    )
    assert classified is not None
    assert classified[0] == "resistance"

    assert _classify_counter_signal(
        "BRAF inhibition in refractory Erdheim-Chester disease",
        (
            "Therapy-refractory Erdheim-Chester disease responded to BRAF "
            "inhibition with reduced lesion burden."
        ),
        "BRAF",
        intervention_direction="inhibit",
    ) is None

    inhibitor_resistance = _classify_counter_signal(
        "Acquired resistance after BRAF inhibition",
        "Resistance to BRAF inhibition emerged after six months.",
        "BRAF",
        intervention_direction="inhibit",
    )
    assert inhibitor_resistance is not None
    assert inhibitor_resistance[0] == "resistance"


def test_counter_signal_does_not_invert_negated_adverse_outcome() -> None:
    assert _classify_counter_signal(
        "Prior FLT3 inhibitor exposure and outcome",
        (
            "Prior FLT3 inhibitor exposure was not associated with inferior "
            "outcome after FLT3 inhibition."
        ),
        "FLT3",
        intervention_direction="inhibit",
    ) is None

    genuine_null_benefit = _classify_counter_signal(
        "FLT3 inhibition and survival",
        "FLT3 inhibition was not associated with improved survival.",
        "FLT3",
        intervention_direction="inhibit",
    )
    assert genuine_null_benefit is not None
    assert genuine_null_benefit[0] == "opposite_or_null_result"


def test_rare_disease_single_patient_variant_is_association_not_prevalence() -> None:
    content = _publication_content_from_sections(
        (
            (
                "RESULTS",
                "In a patient with Erdheim-Chester disease, an activating "
                "mutation in MAP2K1 gene was identified, confirming the "
                "molecular diagnosis.",
            ),
        ),
        target_symbol="MAP2K1",
        disease="Erdheim-Chester disease",
        intervention_direction="inhibit",
    )

    assert content.association_excerpt is not None
    assert content.association_predicate == "disease_target_association"
    assert content.association_scope == "activating_or_itd_target_subtype"
    assert "identified" in content.association_excerpt


def test_explicit_enabled_sources_can_run_pharos_on_demand() -> None:
    client = FakeBiologyClient()
    records = list(
        _retriever(client, enabled_sources=["Pharos"]).retrieve(
            _brief(),
            _seed(),
            1,
        )
    )

    assert [url for _, url, _ in client.calls] == [PHAROS_GRAPHQL_URL]
    assert [record.source for record in records] == ["Pharos"]
    assert records[0].metadata["tdl"] == "Tclin"


def test_source_level_cache_prevents_duplicate_calls_and_marks_provenance() -> None:
    client = FakeBiologyClient()
    retriever = _retriever(client)

    first = list(retriever.retrieve(_brief(), _seed(), 1))
    first_call_count = len(client.calls)
    second = list(retriever.retrieve(_brief(), _seed(), 2))

    assert first_call_count == 11  # PubMed also runs two primary-association lanes.
    assert len(client.calls) == first_call_count
    assert all(item.metadata["provenance_class"] == "external_live" for item in first)
    assert all(item.metadata["provenance_class"] == "external_cache" for item in second)
    assert all(item.metadata["live"] is False for item in second)
    assert all(item.metadata["retrieved_live"] is True for item in second)
    assert all(
        item.metadata["origin_provenance_class"] == "external_live"
        for item in second
    )
    assert all(
        item.metadata["origin_content_hash"]
        == (item.metadata.get("content_hash") or item.excerpt_hash)
        for item in second
    )


def test_prior_revision_evidence_retry_bypasses_source_cache() -> None:
    client = FakeBiologyClient()
    retriever = _retriever(client, enabled_sources=["PubMed"])
    brief = _brief(["PubMed"])

    first = list(retriever.retrieve(brief, _seed(), 1))
    first_call_count = len(client.calls)
    retry_seed = replace(
        _seed(),
        discovery_source="prior_revision_evidence_retry",
    )
    retried = list(retriever.retrieve(brief, retry_seed, 2))

    assert first
    assert len(client.calls) == first_call_count * 2
    assert all(
        item.metadata["evidence_retry_cache_bypassed"] is True
        for item in retried
    )
    assert all(
        item.metadata["evidence_retry_cache_policy"]
        == "fresh_public_lookup_required"
        for item in retried
    )


def test_cache_rejects_fixture_origin_and_refetches_live_source() -> None:
    client = FakeBiologyClient()
    retriever = _retriever(client, enabled_sources=["PubMed"])
    brief = _brief(["PubMed"])
    retriever.retrieve(brief, _seed(), 1)
    assert len(client.calls) == 6

    key = next(iter(retriever._cache))
    entry = retriever._cache[key]
    contaminated = tuple(
        item.model_copy(
            update={
                "metadata": {
                    **item.metadata,
                    "provenance_class": "fixture_fallback",
                    "origin_provenance_class": "fixture_fallback",
                    "mock": True,
                }
            },
            deep=True,
        )
        for item in entry.records
    )
    retriever._cache[key] = entry.__class__(
        stored_at=entry.stored_at,
        records=contaminated,
    )

    records = list(retriever.retrieve(brief, _seed(), 2))

    assert len(client.calls) == 12
    assert all(item.metadata["provenance_class"] == "external_live" for item in records)
    assert all(item.metadata["origin_provenance_class"] == "external_live" for item in records)
    assert all(item.metadata["mock"] is False for item in records)


def test_one_failed_source_does_not_hide_other_live_records() -> None:
    client = FakeBiologyClient(failing_sources={"Europe PMC"})
    retriever = _retriever(client, max_attempts=2)

    records = list(retriever.retrieve(_brief(), _seed(), 1))

    assert {record.source for record in records} == {
        "PubMed",
        "Open Targets",
        "UniProt",
        "DGIdb",
        "Pharos",
    }
    assert retriever.last_failures == {"Europe PMC": "ConnectTimeout"}
    assert all(
        record.metadata["partial_source_failures"]
        == {"Europe PMC": "ConnectTimeout"}
        for record in records
    )
    europe_calls = [call for call in client.calls if call[1] == EUROPE_PMC_SEARCH_URL]
    assert len(europe_calls) == 2


def test_source_call_budget_stops_at_adapter_boundary_and_audits_skips() -> None:
    client = FakeBiologyClient()
    retriever = _retriever(
        client,
        enabled_sources=["PubMed", "Europe PMC", "Open Targets"],
        max_source_calls=2,
        max_elapsed_seconds=30,
    )

    records = list(retriever.retrieve(_brief(), _seed(), 1))

    assert {item.source for item in records} == {"PubMed", "Europe PMC"}
    assert retriever.last_failures == {"Open Targets": "skipped_call_budget"}
    assert OPEN_TARGETS_GRAPHQL_URL not in [url for _, url, _ in client.calls]
    assert all(
        item.metadata["partial_source_failures"]
        == {"Open Targets": "skipped_call_budget"}
        for item in records
    )


def test_elapsed_budget_allows_in_flight_source_then_skips_remaining_sources() -> None:
    clock = FakeClock()
    client = TimedBiologyClient(clock, seconds_per_http_call=0.4)
    retriever = _retriever(
        client,
        enabled_sources=["PubMed", "Europe PMC", "Open Targets"],
        max_source_calls=3,
        max_elapsed_seconds=1.0,
        clock=clock,
    )

    records = list(retriever.retrieve(_brief(), _seed(), 1))

    assert {item.source for item in records} == {"PubMed"}
    assert retriever.last_failures == {
        "Europe PMC": "skipped_time_budget",
        "Open Targets": "skipped_time_budget",
    }
    assert [url for _, url, _ in client.calls] == [
        PUBMED_SEARCH_URL,
        PUBMED_SUMMARY_URL,
        PUBMED_FETCH_URL,
        PUBMED_SEARCH_URL,
        PUBMED_SEARCH_URL,
        PUBMED_SEARCH_URL,
    ]
    assert all(
        item.metadata["partial_source_failures"] == retriever.last_failures
        for item in records
    )


def test_all_sources_unavailable_use_explicit_fixture_fallback() -> None:
    client = FakeBiologyClient(failing_sources={"PubMed"})
    retriever = _retriever(client, fallback=MockEvidenceRetriever())

    records = list(retriever.retrieve(_brief(["PubMed"]), _seed(), 1))

    assert {record.supports for record in records} == {"support", "counter"}
    assert all(record.metadata["provenance_class"] == "fixture_fallback" for record in records)
    assert all(record.metadata["live"] is False for record in records)
    assert all(record.metadata["query"] for record in records)
    assert all(
        record.metadata["fallback_reason"] == {"PubMed": "ConnectTimeout"}
        for record in records
    )


def test_resilient_wrapper_keeps_live_keyword_hits_out_of_counter_evidence() -> None:
    public = _retriever(FakeBiologyClient())
    retriever = ResilientEvidenceRetriever(
        public=public,
        fallback=MockEvidenceRetriever(),
    )

    records = list(retriever.retrieve(_brief(), _seed(), 1))

    live_records = [item for item in records if item.metadata.get("live") is True]
    counter_records = [item for item in records if item.supports == "counter"]
    assert live_records
    assert all(item.supports != "counter" for item in live_records)
    assert len(counter_records) == 1
    assert counter_records[0].metadata["provenance_class"] == "fixture_fallback"
    assert (
        counter_records[0].metadata["fallback_role"]
        == "counter_evidence_placeholder"
    )


def test_real_counter_search_accepts_only_explicit_literature_signal() -> None:
    client = CounterEvidenceBiologyClient()
    records = list(
        _retriever(
            client,
            enabled_sources=["Europe PMC"],
            counter_evidence_enabled=True,
        ).retrieve(_brief(["Europe PMC"]), _seed(), 1)
    )

    support = [item for item in records if item.supports == "support"]
    counter = [item for item in records if item.supports == "counter"]
    assert len(support) == 1
    assert len(counter) == 1
    assert counter[0].external_id == "99887766"
    assert counter[0].metadata["provenance_class"] == "external_live"
    assert counter[0].metadata["counter_category"] == "resistance"
    assert counter[0].metadata["counter_search_separate_query"] is True
    assert counter[0].metadata["counter_search_query"]
    assert len(counter[0].metadata["counter_search_query_sha256"]) == 64
    assert counter[0].metadata["counter_search_raw_hit_count"] == 3
    assert counter[0].metadata["source_location"] == "title"
    assert counter[0].metadata["review_status"] == "researcher_review_required"
    assert counter[0].metadata["detail"].startswith("Resistance to FLT3 inhibition")
    assert counter[0].excerpt_hash
    assert all(item.metadata.get("mock") is False for item in records)
    assert all("11223344" != item.external_id for item in records)
    assert all("55667788" != item.external_id for item in records)


def test_research_mode_wrapper_never_adds_fixture_counter() -> None:
    public = _retriever(
        FakeBiologyClient(),
        enabled_sources=["Europe PMC"],
        counter_evidence_enabled=True,
    )
    retriever = ResilientEvidenceRetriever(
        public=public,
        fallback=MockEvidenceRetriever(),
        fixture_fallback_enabled=False,
    )

    records = list(retriever.retrieve(_brief(["Europe PMC"]), _seed(), 1))

    assert records
    assert all(item.metadata.get("provenance_class") != "fixture_fallback" for item in records)
    assert all(item.metadata.get("mock") is False for item in records)
    assert not [item for item in records if item.supports == "counter"]
    assert records[0].metadata["counter_search_status"] == "completed_empty"


class _DistinctPubMedCounterLaneClient:
    def __init__(self, *, counter_hit: bool) -> None:
        self.counter_hit = counter_hit
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        params = kwargs.get("params", {})
        identifier = str(params.get("id", ""))
        if url == PUBMED_SEARCH_URL:
            term = str(params.get("term", ""))
            is_counter = '"resistance"[Title/Abstract]' in term
            ids = ["87654321"] if is_counter and self.counter_hit else []
            if not is_counter:
                ids = ["12345678"]
            return FakeResponse({"esearchresult": {"idlist": ids}})
        if url == PUBMED_SUMMARY_URL:
            title = (
                "Acquired resistance to FLT3 inhibition in AML"
                if identifier == "87654321"
                else "Direct FLT3 inhibition in AML"
            )
            return FakeResponse(
                {
                    "result": {
                        "uids": [identifier],
                        identifier: {
                            "uid": identifier,
                            "title": title,
                            "pubtype": ["Journal Article"],
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL:
            sentence = (
                "Acquired resistance to FLT3 inhibition limited response in "
                "acute myeloid leukemia."
                if identifier == "87654321"
                else "Direct FLT3 inhibition reduced phosphorylation and "
                "viability in acute myeloid leukemia cells."
            )
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    f"<PMID>{identifier}</PMID><Article><Abstract>"
                    f"<AbstractText Label='RESULTS'>{sentence}</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        raise AssertionError(url)


class _DirectionPrecisionLaneClient:
    def __init__(
        self,
        *,
        empty_main: bool = False,
        metadata_only_direction: bool = False,
        direction_sentence: str | None = None,
    ) -> None:
        self.empty_main = empty_main
        self.metadata_only_direction = metadata_only_direction
        self.direction_sentence = direction_sentence
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.main_id = str(10**12)
        self.target_id = str(10**12 + 1)
        self.direction_ids = [
            self.main_id,
            self.target_id,
            *(str(10**12 + index) for index in range(2, 10)),
        ]

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        params = kwargs.get("params", {})
        term = str(params.get("term", ""))
        identifier = str(params.get("id", ""))
        is_direction = term.endswith("NOT review[Publication Type]")
        if url == PUBMED_SEARCH_URL:
            if is_direction:
                return FakeResponse(
                    {
                        "esearchresult": {
                            "count": str(len(self.direction_ids)),
                            "idlist": list(self.direction_ids),
                        }
                    }
                )
            return FakeResponse(
                {
                    "esearchresult": {
                        "idlist": [] if self.empty_main else [self.main_id]
                    }
                }
            )
        if url == PUBMED_SUMMARY_URL:
            ids = [value for value in identifier.split(",") if value]
            result: dict[str, Any] = {"uids": ids}
            for value in ids:
                result[value] = {
                    "uid": value,
                    "title": (
                        "Direct inhibitor response in molecular disease"
                        if value == self.target_id
                        else "Unrelated target context"
                    ),
                    "pubtype": ["Journal Article"],
                }
            return FakeResponse({"result": result})
        if url == PUBMED_FETCH_URL:
            ids = [value for value in identifier.split(",") if value]
            articles = []
            for value in ids:
                if value == self.target_id and self.metadata_only_direction:
                    articles.append(
                        "<PubmedArticle><MedlineCitation>"
                        f"<PMID>{value}</PMID><Article />"
                        "</MedlineCitation></PubmedArticle>"
                    )
                    continue
                if value == self.target_id:
                    sentence = self.direction_sentence or (
                        "A patient with ALK-rearranged inflammatory "
                        "myofibroblastic tumor showed a drastic response to "
                        "two ALK tyrosine kinase inhibitors."
                    )
                else:
                    sentence = (
                        "ALK inhibition reduced phosphorylation and viability "
                        "in ALK-rearranged lung cancer cells."
                    )
                articles.append(
                    "<PubmedArticle><MedlineCitation>"
                    f"<PMID>{value}</PMID><Article><Abstract>"
                    f"<AbstractText Label='RESULTS'>{sentence}</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                )
            return FakeResponse(
                text="<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"
            )
        raise AssertionError(url)


class _BatchedDirectionLaneClient:
    """Synthetic transport double for large, ordered PubMed UID windows."""

    def __init__(
        self,
        *,
        id_count: int,
        summary_failure: str | None = None,
        summary_mutation: str | None = None,
        fetch_failure: str | None = None,
        fetch_mutation: str | None = None,
        search_failure: str | None = None,
        search_payload: Any | None = None,
    ) -> None:
        self.ids = [str(10**13 + index) for index in range(id_count)]
        self.summary_failure = summary_failure
        self.summary_mutation = summary_mutation
        self.fetch_failure = fetch_failure
        self.fetch_mutation = fetch_mutation
        self.search_failure = search_failure
        self.search_payload = search_payload
        self.summary_calls = 0
        self.fetch_calls = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _raise_timeout(url: str) -> None:
        request = httpx.Request("GET", url)
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        params = kwargs.get("params", {})
        if url == PUBMED_SEARCH_URL:
            if self.search_failure == "http_429":
                return FakeResponse(status_code=429)
            if self.search_failure == "timeout":
                self._raise_timeout(url)
            if self.search_payload is not None:
                return FakeResponse(self.search_payload)
            return FakeResponse(
                {
                    "esearchresult": {
                        "count": str(len(self.ids)),
                        "idlist": list(self.ids),
                    }
                }
            )
        identifiers = [
            value for value in str(params.get("id", "")).split(",") if value
        ]
        if url == PUBMED_SUMMARY_URL:
            self.summary_calls += 1
            if self.summary_calls == 2:
                if self.summary_failure == "http_429":
                    return FakeResponse(status_code=429)
                if self.summary_failure == "timeout":
                    self._raise_timeout(url)
            returned = list(identifiers)
            if self.summary_calls == 2 and self.summary_mutation:
                if self.summary_mutation == "missing":
                    returned.pop()
                elif self.summary_mutation == "duplicate":
                    returned[-1] = returned[0]
                elif self.summary_mutation == "unexpected":
                    returned[-1] = str(10**14)
                elif self.summary_mutation == "reordered":
                    returned.reverse()
                else:  # pragma: no cover - protects the test double itself
                    raise AssertionError(self.summary_mutation)
            result: dict[str, Any] = {"uids": returned}
            for identifier in returned:
                result[identifier] = {
                    "uid": identifier,
                    "title": "Synthetic context-only molecular report",
                    "pubtype": ["Journal Article"],
                }
            return FakeResponse({"result": result})
        if url == PUBMED_FETCH_URL:
            self.fetch_calls += 1
            if self.fetch_calls == 2:
                if self.fetch_failure == "http_429":
                    return FakeResponse(status_code=429)
                if self.fetch_failure == "timeout":
                    self._raise_timeout(url)
            returned = list(identifiers)
            if self.fetch_calls == 1 and self.fetch_mutation == "reordered":
                returned.reverse()
            articles = "".join(
                "<PubmedArticle><MedlineCitation>"
                f"<PMID>{identifier}</PMID><Article />"
                "</MedlineCitation></PubmedArticle>"
                for identifier in returned
            )
            return FakeResponse(
                text=f"<PubmedArticleSet>{articles}</PubmedArticleSet>"
            )
        raise AssertionError(url)


def _run_batched_direction_lane(
    client: _BatchedDirectionLaneClient,
    *,
    raw_document_budget: int = 10_000,
    common_params: Mapping[str, str] | None = None,
) -> Any:
    retriever = _retriever(
        client,
        enabled_sources=["PubMed"],
        primary_direction_raw_document_budget=raw_document_budget,
    )
    brief = _brief(["PubMed"])
    return retriever._pubmed_primary_direction_supplement(
        brief,
        _seed(),
        query='"synthetic target-direction transport audit"',
        common_params=(
            dict(common_params)
            if common_params is not None
            else {"tool": "DrugPilotTest"}
        ),
        excluded_ids=set(),
        direction_model_context=None,
    )


@pytest.mark.parametrize(
    ("payload", "reason_code"),
    [
        (["not-a-mapping"], "response_not_mapping"),
        ({}, "esearchresult_not_mapping"),
        ({"esearchresult": []}, "esearchresult_not_mapping"),
        (
            {"esearchresult": {"count": "0", "idlist": "100"}},
            "idlist_not_sequence",
        ),
        (
            {"esearchresult": {"count": "1", "idlist": [""]}},
            "pmid_not_numeric_nonempty",
        ),
        (
            {"esearchresult": {"count": "1", "idlist": ["not-numeric"]}},
            "pmid_not_numeric_nonempty",
        ),
        (
            {"esearchresult": {"count": "1", "idlist": [123]}},
            "pmid_not_numeric_nonempty",
        ),
        (
            {
                "esearchresult": {
                    "count": "2",
                    "idlist": [str(10**15), str(10**15)],
                }
            },
            "duplicate_pmid",
        ),
        (
            {"esearchresult": {"idlist": []}},
            "count_not_strict_nonnegative_int",
        ),
        (
            {"esearchresult": {"count": -1, "idlist": []}},
            "count_not_strict_nonnegative_int",
        ),
        (
            {"esearchresult": {"count": "01", "idlist": []}},
            "count_not_strict_nonnegative_int",
        ),
        (
            {"esearchresult": {"count": True, "idlist": []}},
            "count_not_strict_nonnegative_int",
        ),
        (
            {
                "esearchresult": {
                    "count": "0",
                    "idlist": [str(10**15)],
                }
            },
            "count_zero_idlist_consistency",
        ),
        (
            {"esearchresult": {"count": "1", "idlist": []}},
            "count_zero_idlist_consistency",
        ),
        (
            {
                "esearchresult": {
                    "count": "1",
                    "idlist": [str(10**15), str(10**15 + 1)],
                }
            },
            "count_less_than_idlist",
        ),
    ],
)
def test_direction_lane_rejects_malformed_esearch_shape_fail_closed(
    payload: Any,
    reason_code: str,
) -> None:
    client = _BatchedDirectionLaneClient(
        id_count=0,
        search_payload=payload,
    )

    outcome = _run_batched_direction_lane(
        client,
        common_params={
            "tool": "DrugPilotTest",
            "api_key": "synthetic-secret-must-not-be-retained",
        },
    )

    trace = outcome.trace
    assert outcome.records == ()
    assert outcome.status == "error"
    assert outcome.stop_reason == f"search_schema_{reason_code}:fail_closed"
    assert trace["error_stage"] == "search_schema"
    assert trace["http_requests_used"] == 1
    assert len(trace["http_call_provenance"]) == 1
    assert trace["http_call_provenance"][0]["status"] == "completed"
    assert trace["http_call_provenance_complete"] is True
    assert trace["summary_batch_count"] == 0
    assert trace["abstract_batch_count"] == 0
    assert "synthetic-secret-must-not-be-retained" not in repr(trace)


def test_direction_lane_rejects_esearch_window_larger_than_registered_budget() -> None:
    client = _BatchedDirectionLaneClient(id_count=2)

    outcome = _run_batched_direction_lane(
        client,
        raw_document_budget=1,
    )

    assert outcome.status == "error"
    assert outcome.stop_reason == (
        "search_schema_idlist_exceeds_raw_document_budget:fail_closed"
    )
    assert outcome.trace["http_requests_used"] == len(
        outcome.trace["http_call_provenance"]
    )


def test_direction_lane_rejects_incomplete_esearch_window_below_budget() -> None:
    identifiers = [str(10**17 + index) for index in range(2)]
    client = _BatchedDirectionLaneClient(
        id_count=0,
        search_payload={
            "esearchresult": {
                "count": "3",
                "idlist": identifiers,
            }
        },
    )

    outcome = _run_batched_direction_lane(
        client,
        raw_document_budget=4,
    )

    assert outcome.status == "error"
    assert outcome.stop_reason == (
        "search_schema_returned_id_window_incomplete:fail_closed"
    )
    assert outcome.trace["error_stage"] == "search_schema"
    assert outcome.trace["http_requests_used"] == 1
    assert outcome.trace["http_requests_used"] == len(
        outcome.trace["http_call_provenance"]
    )


@pytest.mark.parametrize("reported_count", ["2", "5"])
def test_direction_lane_accepts_exact_registered_esearch_window(
    reported_count: str,
) -> None:
    identifiers = [str(10**18 + index) for index in range(2)]
    client = _BatchedDirectionLaneClient(
        id_count=0,
        search_payload={
            "esearchresult": {
                "count": reported_count,
                "idlist": identifiers,
            }
        },
    )

    outcome = _run_batched_direction_lane(
        client,
        raw_document_budget=2,
    )

    assert outcome.status == "completed_bounded_without_verified_result"
    assert outcome.trace["search_result_count"] == int(reported_count)
    assert outcome.trace["returned_id_count"] == 2
    assert outcome.trace["summary_batches_completed"] == 1
    assert outcome.trace["abstract_batches_completed"] == 1


def test_direction_lane_accepts_only_exact_zero_hit_esearch_shape() -> None:
    client = _BatchedDirectionLaneClient(
        id_count=0,
        search_payload={"esearchresult": {"count": "0", "idlist": []}},
    )

    outcome = _run_batched_direction_lane(
        client,
        common_params={
            "tool": "DrugPilotTest",
            "api_key": "synthetic-secret-must-not-be-retained",
        },
    )

    assert outcome.status == "completed_empty"
    assert outcome.stop_reason == "search_returned_zero_hits"
    assert outcome.trace["http_requests_used"] == 1
    assert outcome.trace["http_requests_used"] == len(
        outcome.trace["http_call_provenance"]
    )
    assert client.summary_calls == 0
    assert client.fetch_calls == 0


@pytest.mark.parametrize("failure", ["http_429", "timeout"])
def test_direction_lane_records_bounded_failed_search_provenance(
    failure: str,
) -> None:
    client = _BatchedDirectionLaneClient(
        id_count=1,
        search_failure=failure,
    )

    outcome = _run_batched_direction_lane(
        client,
        common_params={
            "tool": "DrugPilotTest",
            "api_key": "synthetic-secret-must-not-be-retained",
        },
    )

    trace = outcome.trace
    assert outcome.status == "error"
    assert trace["error_stage"] == "search"
    assert trace["http_requests_used"] == 1
    assert len(trace["http_call_provenance"]) == 1
    failed = trace["http_call_provenance"][0]
    assert failed["endpoint"] == PUBMED_SEARCH_URL
    assert failed["method"] == "GET"
    assert failed["status"] == "error"
    assert failed["parameter_sha256"]
    assert failed["error_type"] in {"HTTPStatusError", "ReadTimeout"}
    assert "response_sha256" not in failed
    assert "synthetic timeout" not in repr(trace)
    assert "synthetic-secret-must-not-be-retained" not in repr(trace)
    assert trace["http_call_provenance_complete"] is True


def test_pubmed_xml_preserves_namespaced_article_book_interleaving() -> None:
    identifiers = tuple(str(10**16 + index) for index in range(3))
    xml = (
        '<PubmedArticleSet xmlns="urn:synthetic-pubmed">'
        "<PubmedBookArticle><BookDocument><PMID>"
        f"{identifiers[0]}</PMID><Abstract><AbstractText Label='RESULTS'>"
        "GENE1 inhibition reduced viability in synthetic disease."
        "</AbstractText></Abstract></BookDocument></PubmedBookArticle>"
        "<PubmedArticle><MedlineCitation><PMID>"
        f"{identifiers[1]}</PMID><Article><Abstract><AbstractText Label='RESULTS'>"
        "GENE1 inhibition reduced viability in synthetic disease."
        "</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>"
        "<Envelope><PubmedArticle><MedlineCitation><PMID>999</PMID>"
        "</MedlineCitation></PubmedArticle></Envelope>"
        "<PubmedBookArticle><BookDocument><PMID>"
        f"{identifiers[2]}</PMID><Abstract><AbstractText Label='RESULTS'>"
        "GENE1 inhibition reduced viability in synthetic disease."
        "</AbstractText></Abstract></BookDocument></PubmedBookArticle>"
        "</PubmedArticleSet>"
    )

    returned = _pubmed_xml_record_ids(xml)
    content = _parse_pubmed_content(
        xml,
        target_symbol="GENE1",
        disease="synthetic disease",
        intervention_direction="inhibit",
    )

    assert returned == identifiers
    assert tuple(content) == identifiers
    assert all(item.content_hash for item in content.values())


def test_pubmed_content_never_attaches_reference_doi_or_pmcid() -> None:
    xml = (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
        "<PMID>38978049</PMID><Article><Abstract>"
        "<AbstractText Label='RESULTS'>"
        "FLT3 inhibition reduced viability in acute myeloid leukemia."
        "</AbstractText></Abstract></Article></MedlineCitation>"
        "<PubmedData><ArticleIdList>"
        "<ArticleId IdType='pubmed'>38978049</ArticleId>"
        "<ArticleId IdType='doi'>10.1186/current-article</ArticleId>"
        "<ArticleId IdType='pmc'>PMC11111111</ArticleId>"
        "</ArticleIdList><ReferenceList><Reference><ArticleIdList>"
        "<ArticleId IdType='doi'>10.1038/s41586-020-2864-x</ArticleId>"
        "<ArticleId IdType='pmc'>PMC7677169</ArticleId>"
        "</ArticleIdList></Reference></ReferenceList></PubmedData>"
        "</PubmedArticle></PubmedArticleSet>"
    )

    content = _parse_pubmed_content(
        xml,
        target_symbol="FLT3",
        disease="acute myeloid leukemia",
        intervention_direction="inhibit",
    )["38978049"]

    assert content.doi == "10.1186/current-article"
    assert content.pmcid == "PMC11111111"
    assert content.doi != "10.1038/s41586-020-2864-x"
    assert content.pmcid != "PMC7677169"


class _AlkCoordinatedAssociationLaneClient:
    """Offline rank-six analogue of the audited ALK/IMT PubMed result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.target_id = "association-target"
        self.ids = [
            "association-decoy-1",
            "association-decoy-2",
            "association-decoy-3",
            "association-decoy-4",
            "association-decoy-5",
            self.target_id,
            "association-decoy-7",
            "association-decoy-8",
            "association-decoy-9",
            "association-decoy-10",
            "association-decoy-11",
        ]

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        params = kwargs.get("params", {})
        term = str(params.get("term", ""))
        identifier = str(params.get("id", ""))
        if url == PUBMED_SEARCH_URL:
            if 'NOT "prevalence"[Title]' not in term:
                return FakeResponse(
                    {"esearchresult": {"count": "0", "idlist": []}}
                )
            return FakeResponse(
                {
                    "esearchresult": {
                        "count": str(len(self.ids)),
                        "idlist": list(self.ids),
                    }
                }
            )
        if url == PUBMED_SUMMARY_URL:
            ids = [value for value in identifier.split(",") if value]
            return FakeResponse(
                {
                    "result": {
                        "uids": ids,
                        **{
                            value: {
                                "uid": value,
                                "title": (
                                    "Molecular characterization of inflammatory "
                                    "myofibroblastic tumors with ALK and other "
                                    "gene fusions"
                                    if value == self.target_id
                                    else "Context-only molecular IMT report"
                                ),
                                "pubtype": ["Journal Article"],
                            }
                            for value in ids
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL:
            ids = [value for value in identifier.split(",") if value]
            articles = []
            for value in ids:
                sentence = (
                    "Of the 62 IMTs studied, 35 (56%) showed ALK gene "
                    "rearrangement."
                    if value == self.target_id
                    else "Inflammatory myofibroblastic tumors expressed ALK "
                    "protein in archival material."
                )
                articles.append(
                    "<PubmedArticle><MedlineCitation>"
                    f"<PMID>{value}</PMID><Article><Abstract>"
                    f"<AbstractText Label='RESULTS'>{sentence}</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                )
            return FakeResponse(
                text=(
                    "<PubmedArticleSet>"
                    + "".join(articles)
                    + "</PubmedArticleSet>"
                )
            )
        raise AssertionError(url)


def _alk_direction_brief() -> ResearchBrief:
    return ResearchBrief(
        research_goal="Evaluate direct ALK inhibition in a molecular IMT subtype.",
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=True,
        ),
    )


def _alk_seed() -> TargetSeed:
    return TargetSeed(
        symbol="ALK",
        target_name="Anaplastic lymphoma kinase",
        track=HypothesisTrack.LITERATURE,
        priority=0.9,
    )


@pytest.mark.parametrize(
    ("target", "disease", "expected_disease_terms"),
    [
        (
            "FLT3",
            "Acute myeloid leukemia",
            (
                '"Acute myeloid leukemia"[Title]',
                '"AML"[Title]',
            ),
        ),
        (
            "EGFR",
            "triple-negative breast cancer",
            (
                '"triple-negative breast cancer"[Title]',
                '"TNBC"[Title]',
            ),
        ),
        (
            "EGFR",
            "non-small cell lung cancer",
            (
                '"non-small cell lung cancer"[Title]',
                '"NSCLC"[Title]',
            ),
        ),
    ],
)
def test_direction_title_precision_query_is_generic_for_common_disease_aliases(
    target: str,
    disease: str,
    expected_disease_terms: tuple[str, ...],
) -> None:
    brief = ResearchBrief(
        research_goal=f"Evaluate direct {target} inhibition in {disease}.",
        disease=disease,
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(allowed_sources=["PubMed"]),
    )
    seed = TargetSeed(
        symbol=target,
        target_name=target,
        track=HypothesisTrack.LITERATURE,
        priority=0.9,
    )

    query = PublicBiologyRetriever._pubmed_primary_direction_query(
        brief,
        seed,
    )

    assert all(term in query for term in expected_disease_terms)
    assert f'"{target} inhibitor"[Title]' in query
    assert f'"{target} inhibitors"[Title]' in query
    assert '"response"[Title]' in query
    assert '"responses"[Title]' in query
    assert f'"{target} inhibitor"[Title/Abstract]' in query
    assert f'"inhibition of {target}"[Title/Abstract]' in query
    assert '"apoptosis"[Title/Abstract]' in query
    assert "PMID" not in query.upper()
    assert "CRIZOTINIB" not in query.upper()
    assert query.endswith("NOT review[Publication Type]")


@pytest.mark.parametrize(
    (
        "target",
        "disease",
        "target_cell_type",
        "model_alias",
        "variant_phrase",
    ),
    [
        (
            "JAK2",
            "essential thrombocythemia",
            "SET-2 (molecularly confirmed JAK2-V617F)",
            "SET-2",
            "JAK2 V617F",
        ),
        (
            "FLT3",
            "acute myeloid leukemia",
            "MV4-11 (molecularly confirmed FLT3-ITD)",
            "MV4-11",
            "FLT3-ITD",
        ),
        (
            "RET",
            "non-small cell lung cancer",
            "LC-2/ad (RET fusion-positive)",
            "LC-2/ad",
            "RET fusion",
        ),
    ],
)
def test_direction_query_adds_only_brief_declared_molecular_model_context(
    target: str,
    disease: str,
    target_cell_type: str,
    model_alias: str,
    variant_phrase: str,
) -> None:
    brief = ResearchBrief(
        research_goal=(
            f"Evaluate direct {target} inhibition in the exact molecular model."
        ),
        disease=disease,
        constraints=ProjectConstraints(
            target_cell_type=target_cell_type,
            intervention_direction="inhibit",
        ),
        data_policy=DataSourcePolicy(allowed_sources=["PubMed"]),
    )
    seed = TargetSeed(
        symbol=target,
        target_name=target,
        track=HypothesisTrack.LITERATURE,
        priority=0.9,
    )

    query = PublicBiologyRetriever._pubmed_primary_direction_query(
        brief,
        seed,
    )

    assert f'"{model_alias}"[Title]' in query
    assert f'"{variant_phrase}"[Title]' in query
    assert f'"{target} inhibitor"[Title]' in query
    assert f'"{target} inhibitor"[Title/Abstract]' in query
    assert f'"inhibition of {target}"[Title/Abstract]' in query
    assert "PMID" not in query.upper()
    assert query.endswith("NOT review[Publication Type]")


def test_official_set2_abstract_uses_predeclared_model_result_not_context_sentence(
) -> None:
    """Regression from the official primary abstract; no live ID is embedded."""

    brief = ResearchBrief(
        research_goal=(
            "Evaluate direct JAK2 V617F inhibition in essential "
            "thrombocythemia."
        ),
        disease="essential thrombocythemia",
        constraints=ProjectConstraints(
            target_cell_type=(
                "SET-2 (molecularly confirmed JAK2-V617F)"
            ),
            intervention_direction="inhibit",
        ),
    )
    abstract = (
        "The Janus-associated kinase 2 (JAK2) V617F mutation is believed to "
        "play a critical role in the pathogenesis of polycythemia vera, "
        "essential thrombocythemia, and idiopathic myelofibrosis. "
        "We have characterized a novel small molecule JAK2 inhibitor, AZ960, "
        "and used it as a tool to investigate the consequences of JAK2 V617F "
        "inhibition in the SET-2 cell line. In the SET-2 human "
        "megakaryoblastic cell line, heterozygous for the JAK2 V617F allele, "
        "inhibition of JAK2 resulted in decreased STAT3/5 phosphorylation and "
        "inhibition of cell proliferation (GI(50)=0.033 microm) predominately "
        "through the induction of mitochondrial-mediated apoptosis."
    )
    title = (
        "Effects of the JAK2 inhibitor, AZ960, on survival signaling in the "
        "human JAK2 V617F cell line SET-2."
    )

    literal_only = _publication_content_from_sections(
        [("Abstract", abstract)],
        target_symbol="JAK2",
        disease=brief.disease,
        intervention_direction="inhibit",
    )
    literal_codes = _primary_direction_content_rejection_codes(
        literal_only,
        title=title,
        publication_types=["Journal Article"],
        target_symbol="JAK2",
        disease=brief.disease,
        intervention_direction="inhibit",
    )
    context = _primary_direction_model_context(brief, "JAK2")
    assert context is not None
    selected = _publication_content_from_sections(
        [("Abstract", abstract)],
        target_symbol="JAK2",
        disease=brief.disease,
        intervention_direction="inhibit",
        direction_model_context=context,
    )
    selected_codes = _primary_direction_content_rejection_codes(
        selected,
        title=title,
        publication_types=["Journal Article"],
        target_symbol="JAK2",
        disease=brief.disease,
        intervention_direction="inhibit",
        direction_model_context=context,
    )

    assert literal_only.excerpt is not None
    assert "essential thrombocythemia" in literal_only.excerpt
    assert "direct_target_intervention_missing" in literal_codes
    assert selected.excerpt is not None
    assert "SET-2 human megakaryoblastic cell line" in selected.excerpt
    assert "GI(50)=0.033 microm" in selected.excerpt
    assert selected_codes == ()
    assert _primary_direction_context_match_basis(
        selected.excerpt,
        target_symbol="JAK2",
        disease=brief.disease,
        context=context,
    ) == "predeclared_molecular_model"


@pytest.mark.parametrize(
    ("target", "disease", "target_cell_type", "sentence"),
    [
        (
            "FLT3",
            "acute myeloid leukemia",
            "MV4-11 (FLT3-ITD)",
            "In MV4-11 cells harboring FLT3-ITD, inhibition of FLT3 reduced "
            "phosphorylation and cell proliferation (IC50=12 nM).",
        ),
        (
            "FLT3",
            "acute myeloid leukemia",
            "MOLM-13 (FLT3 p.D835Y)",
            "In MOLM-13 cells harboring FLT3 p.D835Y, inhibition of FLT3 "
            "reduced phosphorylation and cell proliferation (IC50=9 nM).",
        ),
        (
            "ALK",
            "inflammatory myofibroblastic tumor",
            "H3122 (ALK-rearranged)",
            "In ALK-rearranged H3122 cells, ALK inhibition reduced "
            "phosphorylation and viability (IC50=18 nM).",
        ),
        (
            "RET",
            "non-small cell lung cancer",
            "LC-2/ad (RET fusion-positive)",
            "In RET fusion-positive LC-2/ad cells, RET inhibition reduced "
            "phosphorylation and proliferation (IC50=22 nM).",
        ),
    ],
)
def test_predeclared_model_context_is_generic_beyond_one_target(
    target: str,
    disease: str,
    target_cell_type: str,
    sentence: str,
) -> None:
    brief = ResearchBrief(
        research_goal=f"Evaluate direct {target} inhibition in a fixed model.",
        disease=disease,
        constraints=ProjectConstraints(
            target_cell_type=target_cell_type,
            intervention_direction="inhibit",
        ),
    )
    context = _primary_direction_model_context(brief, target)
    assert context is not None
    assert context.model_aliases == (target_cell_type.split(" ", 1)[0],)
    assert _primary_direction_context_match_basis(
        sentence,
        target_symbol=target,
        disease=disease,
        context=context,
    ) == "predeclared_molecular_model"
    content = _publication_content_from_sections(
        [("RESULTS", sentence)],
        target_symbol=target,
        disease=disease,
        intervention_direction="inhibit",
        direction_model_context=context,
    )

    assert content.excerpt == sentence
    assert _primary_direction_content_rejection_codes(
        content,
        title=f"Direct {target} molecular model study",
        publication_types=["Journal Article"],
        target_symbol=target,
        disease=disease,
        intervention_direction="inhibit",
        direction_model_context=context,
    ) == ()


@pytest.mark.parametrize(
    "abstract",
    [
        (
            "In MV4-11 cells, FLT3 inhibition reduced phosphorylation and "
            "viability (IC50=12 nM)."
        ),
        (
            "In FLT3-ITD leukemia cells, FLT3 inhibition reduced "
            "phosphorylation and viability (IC50=12 nM)."
        ),
        (
            "In MV4-11 cells harboring KIT D816V, KIT inhibition reduced "
            "phosphorylation and viability (IC50=12 nM)."
        ),
        (
            "MV4-11 cells harbor FLT3-ITD. FLT3 inhibition reduced "
            "phosphorylation and viability (IC50=12 nM)."
        ),
    ],
)
def test_model_context_fails_closed_without_atomic_target_model_variant_result(
    abstract: str,
) -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct FLT3 inhibition in a fixed model.",
        disease="acute myeloid leukemia",
        constraints=ProjectConstraints(
            target_cell_type="MV4-11 (FLT3-ITD)",
            intervention_direction="inhibit",
        ),
    )
    context = _primary_direction_model_context(brief, "FLT3")
    assert context is not None
    content = _publication_content_from_sections(
        [("RESULTS", abstract)],
        target_symbol="FLT3",
        disease=brief.disease,
        intervention_direction="inhibit",
        direction_model_context=context,
    )
    codes = _primary_direction_content_rejection_codes(
        content,
        title="Molecular model study",
        publication_types=["Journal Article"],
        target_symbol="FLT3",
        disease=brief.disease,
        intervention_direction="inhibit",
        direction_model_context=context,
    )

    assert codes
    assert content.excerpt is None or (
        "direct_target_intervention_missing" in codes
        or "requested_direction_missing" in codes
    )


def test_direction_precision_lane_uses_complete_pubmed_result_window_and_is_attested() -> None:
    client = _DirectionPrecisionLaneClient()
    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=8,
        ).retrieve(_alk_direction_brief(), _alk_seed(), 1)
    )

    selected = next(
        item for item in records if item.external_id == client.target_id
    )
    trace = selected.metadata["primary_direction_supplement_trace"]
    direction_search = client.calls[3]
    assert len(client.calls) == 8
    assert direction_search[0] == PUBMED_SEARCH_URL
    assert direction_search[1]["params"]["retmax"] == 10_000
    direction_query = str(direction_search[1]["params"]["term"])
    assert (
        '"inflammatory myofibroblastic tumor"[Title]'
        in direction_query
    )
    assert '"ALK inhibitor"[Title]' in direction_query
    assert '"ALK inhibitors"[Title]' in direction_query
    assert '"response"[Title]' in direction_query
    assert '"responses"[Title]' in direction_query
    assert '"inflammatory myofibroblastic tumor"[Title/Abstract]' in direction_query
    assert '"ALK inhibitor"[Title/Abstract]' in direction_query
    assert '"inhibition of ALK"[Title/Abstract]' in direction_query
    assert '"apoptosis"[Title/Abstract]' in direction_query
    assert "CRIZOTINIB" not in direction_query.upper()
    assert "PMID" not in direction_query.upper()
    assert direction_query.endswith("NOT review[Publication Type]")
    assert client.direction_ids.index(client.target_id) + 1 == 2
    assert trace["status"] == "completed_with_results"
    assert trace["http_request_budget"] == 3
    assert trace["http_requests_used"] == 3
    assert trace["raw_document_budget"] == 10_000
    assert trace["raw_hit_count"] == 10
    assert trace["returned_id_count"] == 10
    assert trace["duplicate_id_count"] == 0
    assert trace["excluded_main_query_id_count"] == 1
    assert trace["raw_documents_fetched"] == 9
    assert trace["records_screened"] == 1
    screened_entry = dict(trace["screening_entries"][0])
    extraction = screened_entry.pop("extraction_observation")
    assert extraction["policy"] == "observed_evidence_gap_v1"
    assert extraction["abstract_received"] is True
    assert extraction["abstract_sha256"] == selected.metadata["content_hash"]
    assert extraction["excerpt_sha256"] == selected.excerpt_hash
    assert extraction["excerpt_selected"] is True
    assert screened_entry == {
        "external_id": client.target_id,
        "rank": 1,
        "status": "selected",
        "rejection_codes": [],
        "disease_context_match_basis": "literal_disease",
    }
    assert trace["query_sha256"] == sha256(
        trace["query"].encode("utf-8")
    ).hexdigest()
    assert [call["endpoint"] for call in trace["http_call_provenance"]] == [
        PUBMED_SEARCH_URL,
        PUBMED_SUMMARY_URL,
        PUBMED_FETCH_URL,
    ]
    assert all(
        call["parameter_sha256"]
        and call["response_sha256"]
        and call["status_code"] == 200
        and call["fixture"] is False
        for call in trace["http_call_provenance"]
    )
    assert selected.metadata["query_strategy"] == (
        "brief-derived-target-action-experimental-direction-primary-v3"
    )
    assert selected.metadata["query_construction_policy"] == (
        "brief-derived-target-action-experimental-direction-primary-v3"
    )
    assert selected.metadata["primary_direction_supplement"] is True
    search_terms = [
        str(kwargs["params"]["term"])
        for url, kwargs in client.calls
        if url == PUBMED_SEARCH_URL
    ]
    assert len(search_terms) == 4
    assert search_terms[0] != search_terms[1]
    assert "PMID" not in search_terms[1].upper()


def test_direction_lane_batches_large_summary_windows_with_exact_call_accounting() -> None:
    client = _BatchedDirectionLaneClient(id_count=401)

    outcome = _run_batched_direction_lane(client)

    trace = outcome.trace
    summary_calls = [
        kwargs["params"]["id"].split(",")
        for url, kwargs in client.calls
        if url == PUBMED_SUMMARY_URL
    ]
    fetch_calls = [
        kwargs["params"]["id"].split(",")
        for url, kwargs in client.calls
        if url == PUBMED_FETCH_URL
    ]
    assert outcome.status == "completed_bounded_without_verified_result"
    assert outcome.records == ()
    assert [len(batch) for batch in summary_calls] == [200, 200, 1]
    assert [identifier for batch in summary_calls for identifier in batch] == client.ids
    assert [len(batch) for batch in fetch_calls] == [*([20] * 20), 1]
    assert [identifier for batch in fetch_calls for identifier in batch] == client.ids
    assert trace["http_request_budget"] == 25
    assert trace["http_requests_used"] == 25
    assert len(client.calls) == 25
    assert trace["summary_batch_size"] == 200
    assert trace["summary_batch_count"] == 3
    assert trace["summary_batches_completed"] == 3
    assert trace["abstract_batch_size"] == 20
    assert trace["abstract_batch_count"] == 21
    assert trace["abstract_batches_completed"] == 21
    assert trace["raw_documents_fetched"] == 401
    assert trace["transport_attempt_semantics"] == (
        "retry_attempts_owned_by_injected_http_client;"
        "one_adapter_request_counted_per_ncbi_get"
    )
    assert all(
        audit["identity_validation"] == "exact_ordered_match"
        and audit["response_body_stored"] is False
        for audit in [
            *trace["summary_batch_audits"],
            *trace["abstract_batch_audits"],
        ]
    )
    assert "PubmedArticleSet" not in repr(trace)
    assert all(
        set(call) == {
            "method",
            "endpoint",
            "parameter_keys",
            "parameter_sha256",
            "status",
            "status_code",
            "response_sha256",
            "fixture",
        }
        for call in trace["http_call_provenance"]
    )


@pytest.mark.parametrize("failure", ["http_429", "timeout"])
def test_direction_lane_fails_closed_on_partial_summary_transport(
    failure: str,
) -> None:
    client = _BatchedDirectionLaneClient(
        id_count=401,
        summary_failure=failure,
    )

    outcome = _run_batched_direction_lane(client)

    trace = outcome.trace
    assert outcome.status == "error"
    assert outcome.records == ()
    assert outcome.stop_reason.endswith(":fail_closed")
    assert trace["error_stage"] == "summary_batch"
    assert trace["summary_batch_count"] == 3
    assert trace["summary_batches_completed"] == 1
    assert [audit["status"] for audit in trace["summary_batch_audits"]] == [
        "completed",
        "error",
    ]
    assert trace["abstract_batches_completed"] == 0
    assert trace["raw_documents_fetched"] == 0
    assert trace["http_requests_used"] == 3
    assert trace["http_requests_used"] <= trace["http_request_budget"]
    assert client.summary_calls == 2
    assert client.fetch_calls == 0
    assert [call["endpoint"] for call in trace["http_call_provenance"]] == [
        PUBMED_SEARCH_URL,
        PUBMED_SUMMARY_URL,
        PUBMED_SUMMARY_URL,
    ]
    assert trace["http_call_provenance"][-1]["status"] == "error"


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("missing", "returned_id_count_mismatch"),
        ("duplicate", "duplicate_returned_id"),
        ("unexpected", "returned_id_membership_mismatch"),
        ("reordered", "returned_id_order_mismatch"),
    ],
)
def test_direction_lane_rejects_inconsistent_summary_batch_identity(
    mutation: str,
    reason_code: str,
) -> None:
    client = _BatchedDirectionLaneClient(
        id_count=401,
        summary_mutation=mutation,
    )

    outcome = _run_batched_direction_lane(client)

    trace = outcome.trace
    assert outcome.status == "error"
    assert outcome.records == ()
    assert outcome.stop_reason == f"summary_batch_{reason_code}:fail_closed"
    assert trace["summary_batches_completed"] == 1
    assert trace["summary_batch_audits"][-1]["error_reason_code"] == reason_code
    assert trace["abstract_batches_completed"] == 0
    assert trace["raw_documents_fetched"] == 0
    assert client.fetch_calls == 0
    assert trace["http_requests_used"] == len(trace["http_call_provenance"])
    assert trace["http_call_provenance"][-1]["status"] == "completed"


@pytest.mark.parametrize("failure", ["http_429", "timeout"])
def test_direction_lane_discards_partial_fetch_batches_on_transport_error(
    failure: str,
) -> None:
    client = _BatchedDirectionLaneClient(
        id_count=41,
        fetch_failure=failure,
    )

    outcome = _run_batched_direction_lane(client)

    trace = outcome.trace
    assert outcome.status == "error"
    assert outcome.records == ()
    assert outcome.stop_reason.endswith(":fail_closed")
    assert trace["error_stage"] == "abstract_batch"
    assert trace["summary_batches_completed"] == 1
    assert trace["abstract_batch_count"] == 3
    assert trace["abstract_batches_completed"] == 1
    assert trace["raw_documents_fetched"] == 20
    assert [audit["status"] for audit in trace["abstract_batch_audits"]] == [
        "completed",
        "error",
    ]
    assert trace["http_requests_used"] == 4
    assert client.fetch_calls == 2
    assert trace["http_requests_used"] == len(trace["http_call_provenance"])
    assert trace["http_call_provenance"][-1]["status"] == "error"


def test_direction_lane_rejects_reordered_fetch_batch_without_partial_records() -> None:
    client = _BatchedDirectionLaneClient(
        id_count=21,
        fetch_mutation="reordered",
    )

    outcome = _run_batched_direction_lane(client)

    trace = outcome.trace
    assert outcome.status == "error"
    assert outcome.records == ()
    assert outcome.stop_reason == (
        "abstract_batch_returned_id_order_mismatch:fail_closed"
    )
    assert trace["abstract_batches_completed"] == 0
    assert trace["raw_documents_fetched"] == 0
    assert trace["abstract_batch_audits"][0]["error_reason_code"] == (
        "returned_id_order_mismatch"
    )
    assert trace["http_requests_used"] == len(trace["http_call_provenance"])
    assert trace["http_call_provenance"][-1]["status"] == "completed"


def test_direction_precision_lane_can_select_evidence_beyond_rank_twenty() -> None:
    client = _DirectionPrecisionLaneClient(empty_main=True)
    client.direction_ids = [
        *(str(10**12 + index) for index in range(100, 154)),
        client.target_id,
        *(str(10**12 + index) for index in range(155, 180)),
    ]
    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=8,
        ).retrieve(_alk_direction_brief(), _alk_seed(), 1)
    )

    selected = next(
        item for item in records if item.external_id == client.target_id
    )
    trace = selected.metadata["primary_direction_supplement_trace"]

    assert trace["returned_id_count"] == 80
    assert trace["records_screened"] == 55
    assert trace["http_request_budget"] == 6
    assert trace["http_requests_used"] == 6
    assert trace["screening_entries"][-1]["rank"] == 55
    assert trace["stop_reason"] == "first_verified_direct_primary_selected"


def test_direction_precision_lane_runs_after_empty_main_search() -> None:
    client = _DirectionPrecisionLaneClient(empty_main=True)
    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=8,
        ).retrieve(_alk_direction_brief(), _alk_seed(), 1)
    )

    assert len(client.calls) == 6
    selected = next(
        item for item in records if item.external_id == client.target_id
    )
    trace = selected.metadata["primary_direction_supplement_trace"]
    assert trace["status"] == "completed_with_results"
    assert trace["http_requests_used"] == 3
    assert trace["records_screened"] == 2


def test_direction_precision_lane_never_promotes_metadata_only_record() -> None:
    client = _DirectionPrecisionLaneClient(metadata_only_direction=True)
    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=8,
        ).retrieve(_alk_direction_brief(), _alk_seed(), 1)
    )

    assert len(client.calls) == 8
    assert records
    assert all(item.supports == "context" for item in records)
    assert all(item.external_id != client.target_id for item in records)
    trace = records[0].metadata["primary_direction_supplement_trace"]
    assert trace["status"] == "completed_bounded_without_verified_result"
    assert trace["http_requests_used"] == 3
    assert trace["raw_documents_fetched"] == 9
    assert trace["rejection_counts"]["abstract_excerpt_unavailable"] == 9


@pytest.mark.parametrize(
    ("sentence", "expected_code"),
    [
        (
            "ALK inhibition increased viability and tumor growth in "
            "ALK-rearranged inflammatory myofibroblastic tumor cells.",
            "unfavorable_endpoint_polarity",
        ),
        (
            "Viability was measured after ALK inhibition in ALK-rearranged "
            "inflammatory myofibroblastic tumor cells.",
            "positive_result_language_missing",
        ),
    ],
)
def test_direction_precision_lane_rejects_harmful_or_methods_only_results(
    sentence: str,
    expected_code: str,
) -> None:
    client = _DirectionPrecisionLaneClient(direction_sentence=sentence)
    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=8,
        ).retrieve(_alk_direction_brief(), _alk_seed(), 1)
    )

    assert len(client.calls) == 8
    assert all(item.external_id != client.target_id for item in records)
    trace = records[0].metadata["primary_direction_supplement_trace"]
    assert trace["status"] == "completed_bounded_without_verified_result"
    assert trace["rejection_counts"][expected_code] == 1


def test_direction_endpoint_polarity_is_explicit_and_fail_closed() -> None:
    assert _direction_endpoint_polarity_status(
        "ALK inhibition reduced phosphorylation and viability in "
        "ALK-rearranged IMT cells.",
        "inhibit",
    ) == "favorable"
    assert _direction_endpoint_polarity_status(
        "FLC-8 suppressed FLT3-mediated signaling and induced apoptosis in "
        "FLT3-ITD AML cells.",
        "inhibit",
    ) == "favorable"
    assert _direction_endpoint_polarity_status(
        "ALK inhibition increased viability and tumor growth in "
        "ALK-rearranged IMT cells.",
        "inhibit",
    ) == "unfavorable"
    assert _direction_endpoint_polarity_status(
        "Viability was measured after ALK inhibition in ALK-rearranged IMT.",
        "inhibit",
    ) == "ambiguous"
    assert _direction_endpoint_polarity_status(
        "JAK2 inhibitors produced similar antiproliferative effects in MPN cells.",
        "inhibit",
    ) == "favorable"
    assert _direction_endpoint_polarity_status(
        "JAK2 inhibitors were potent inducers of apoptosis in primary cells.",
        "inhibit",
    ) == "favorable"
    assert _direction_endpoint_polarity_status(
        "JAK2 inhibitors did not produce antiproliferative effects in MPN cells.",
        "inhibit",
    ) == "ambiguous"
    assert _direction_endpoint_polarity_status(
        "EGFR activation increased EGFR phosphorylation in EGFR-mutant "
        "NSCLC cells.",
        "activate",
    ) == "favorable"
    assert _direction_endpoint_polarity_status(
        "EGFR activation decreased EGFR phosphorylation in EGFR-mutant "
        "NSCLC cells.",
        "activate",
    ) == "unfavorable"


def test_compound_linked_abstract_pair_admits_same_drug_target_and_outcome() -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct JAK2 inhibition only in myelofibrosis.",
        disease="myelofibrosis",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
    )
    content = _publication_content_from_sections(
        [
            ("BACKGROUND", "Myelofibrosis (MF) is a myeloproliferative neoplasm."),
            (
                "BACKGROUND",
                "Fedratinib, a JAK2-selective inhibitor, was evaluated in MF cells.",
            ),
            (
                "RESULTS",
                "Fedratinib significantly reduced viability in MF cells.",
            ),
        ],
        target_symbol="JAK2",
        disease=brief.disease,
        intervention_direction="inhibit",
    )

    assert content.excerpt_selection_status == (
        "selected_compound_linked_document_pair"
    )
    assert content.excerpt is not None
    assert content.excerpt.count("Fedratinib") == 2
    assert "Myelofibrosis (MF)" in content.excerpt
    assert "neoplasm" not in content.excerpt
    assert content.source_location is not None
    assert content.source_location.startswith("abstract-paired:")
    assert _primary_direction_content_rejection_codes(
        content,
        title="Fedratinib response in myelofibrosis cells",
        publication_types=["Journal Article"],
        target_symbol="JAK2",
        disease=brief.disease,
        intervention_direction="inhibit",
    ) == ()


def test_compound_linked_abstract_pair_rejects_different_outcome_drug() -> None:
    content = _publication_content_from_sections(
        [
            ("BACKGROUND", "Myelofibrosis (MF) is a myeloproliferative neoplasm."),
            (
                "BACKGROUND",
                "Fedratinib, a JAK2-selective inhibitor, was studied in patients "
                "with MF.",
            ),
            (
                "CONCLUSIONS",
                "Pacritinib significantly reduced viability in MF cells.",
            ),
        ],
        target_symbol="JAK2",
        disease="myelofibrosis",
        intervention_direction="inhibit",
    )

    assert content.excerpt is None
    assert content.excerpt_selection_status == "related_excerpt_not_found"


def test_compound_linked_clinical_outcome_is_context_not_mechanistic_support() -> None:
    content = _publication_content_from_sections(
        [
            ("BACKGROUND", "Myelofibrosis (MF) is a myeloproliferative neoplasm."),
            (
                "BACKGROUND",
                "Fedratinib, a JAK2-selective inhibitor, was studied in patients "
                "with MF.",
            ),
            (
                "CONCLUSIONS",
                "Fedratinib therapy significantly reduced splenomegaly and symptom "
                "burden in patients with MF.",
            ),
        ],
        target_symbol="JAK2",
        disease="myelofibrosis",
        intervention_direction="inhibit",
    )

    assert content.excerpt is None
    assert content.excerpt_selection_status == "related_excerpt_not_found"


def test_response_modifier_sentence_does_not_hide_direct_model_result() -> None:
    content = _publication_content_from_sections(
        [
            (
                "RESULTS",
                "ABT-737 enhanced the apoptosis induced by JAK2 inhibition in "
                "JAK2 V617F-positive HEL and SET-2 cells.",
            ),
            (
                "RESULTS",
                "JAK2 inhibition reduced viability and increased apoptosis in "
                "JAK2 V617F-positive SET-2 cells.",
            ),
        ],
        target_symbol="JAK2",
        disease="myelofibrosis",
        intervention_direction="inhibit",
        direction_model_context=_primary_direction_model_context(
            ResearchBrief(
                research_goal="Evaluate JAK2 V617F inhibition in the fixed model.",
                disease="myelofibrosis",
                constraints=ProjectConstraints(
                    target_cell_type="SET-2 (JAK2 V617F)",
                    intervention_direction="inhibit",
                ),
            ),
            "JAK2",
        ),
    )

    assert content.excerpt is not None
    assert content.excerpt.startswith("JAK2 inhibition reduced viability")
    assert "ABT-737" not in content.excerpt
    assert _direct_support_rejection_codes(content.excerpt, "JAK2") == ()


def test_other_agent_enhancement_is_response_modifier_context() -> None:
    codes = _direct_support_rejection_codes(
        "ABT-737 enhanced the apoptosis induced by JAK2 inhibition in "
        "JAK2 V617F-positive HEL and SET-2 cells.",
        "JAK2",
    )

    assert "other_intervention_response_modifier" in codes


def test_sensitization_to_another_agent_is_not_standalone_target_support() -> None:
    excerpt = (
        "Ruxolitinib inhibited JAK2 and was used to increase the sensitivity "
        "of acute myeloid leukemia cells to arsenic trioxide."
    )
    content = _publication_content_from_sections(
        (("RESULTS", excerpt),),
        target_symbol="JAK2",
        disease="acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    codes = _primary_direction_content_rejection_codes(
        content,
        title="JAK2 inhibition in acute myeloid leukemia",
        publication_types=("Journal Article",),
        target_symbol="JAK2",
        disease="acute myeloid leukemia",
        intervention_direction="inhibit",
        require_primary_result_context=True,
    )

    assert content.excerpt == excerpt
    assert "response_modifier" in codes
    assert "sensitization_to_other_intervention" in codes


def test_standalone_direct_target_result_remains_eligible() -> None:
    excerpt = (
        "Ruxolitinib inhibited JAK2 phosphorylation and reduced viability "
        "in acute myeloid leukemia cells."
    )
    content = _publication_content_from_sections(
        (("RESULTS", excerpt),),
        target_symbol="JAK2",
        disease="acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    codes = _primary_direction_content_rejection_codes(
        content,
        title="Direct JAK2 inhibition in acute myeloid leukemia",
        publication_types=("Journal Article",),
        target_symbol="JAK2",
        disease="acute myeloid leukemia",
        intervention_direction="inhibit",
        require_primary_result_context=True,
    )

    assert content.excerpt == excerpt
    assert codes == ()


def test_unstructured_background_claim_is_not_a_primary_result() -> None:
    content = _publication_content_from_sections(
        [
            (
                "section-1",
                "Monotherapy with JAK2 inhibitors reduces splenomegaly and symptom "
                "burden but has limited effect on progression of myelofibrosis.",
            )
        ],
        target_symbol="JAK2",
        disease="myelofibrosis",
        intervention_direction="inhibit",
    )

    codes = _primary_direction_content_rejection_codes(
        content,
        title="Combination treatment with azacitidine and ruxolitinib",
        publication_types=["Journal Article"],
        target_symbol="JAK2",
        disease="myelofibrosis",
        intervention_direction="inhibit",
        require_primary_result_context=True,
    )

    assert "primary_result_context_missing" in codes


class _RetInitialismCollisionClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        params = kwargs.get("params", {})
        term = str(params.get("term", ""))
        if url == PUBMED_SEARCH_URL:
            if term.endswith("NOT review[Publication Type]"):
                return FakeResponse({"esearchresult": {"count": "0", "idlist": []}})
            return FakeResponse({"esearchresult": {"idlist": ["ret-main"]}})
        if url == PUBMED_SUMMARY_URL:
            return FakeResponse(
                {
                    "result": {
                        "uids": ["ret-main"],
                        "ret-main": {
                            "uid": "ret-main",
                            "title": "RET inhibition in rearranged lung cancer",
                            "pubtype": ["Journal Article"],
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL:
            return FakeResponse(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>ret-main</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>RET inhibition reduced "
                    "phosphorylation and viability in RET-rearranged lung "
                    "cancer cells.</AbstractText></Abstract></Article>"
                    "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
                )
            )
        raise AssertionError(url)


def test_target_equal_base_initialism_cannot_suppress_direction_fallback() -> None:
    disease = "RET-rearranged renal epithelial tumor"
    brief = ResearchBrief(
        research_goal="Evaluate direct RET inhibition in renal epithelial tumor.",
        disease=disease,
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=True,
        ),
    )
    seed = TargetSeed(
        symbol="RET",
        target_name="RET proto-oncogene",
        track=HypothesisTrack.LITERATURE,
        priority=0.9,
    )
    client = _RetInitialismCollisionClient()

    records = list(
        _retriever(client, enabled_sources=["PubMed"]).retrieve(
            brief,
            seed,
            1,
        )
    )

    assert len(client.calls) == 6
    assert records[0].metadata["primary_direction_supplement_status"] == (
        "completed_empty"
    )
    trace = records[0].metadata["primary_direction_supplement_trace"]
    assert trace["main_query_verified_direct_primary"] is False
    assert _disease_query_terms(disease, target_symbol="RET") == (
        disease,
        "RRRET",
        "renal epithelial tumor",
        "renal epithelial tumors",
    )


class _VerifiedPrimarySmallCapClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        params = kwargs.get("params", {})
        term = str(params.get("term", ""))
        identifier = str(params.get("id", ""))
        if url == PUBMED_SEARCH_URL:
            ids = (
                ["association-id"]
                if '"prevalence"[Title]' in term
                else ["verified-id", "weak-id"]
            )
            return FakeResponse({"esearchresult": {"idlist": ids}})
        if url == PUBMED_SUMMARY_URL:
            ids = [value for value in identifier.split(",") if value]
            result: dict[str, Any] = {"uids": ids}
            for value in ids:
                result[value] = {
                    "uid": value,
                    "title": (
                        "Primary prevalence of FLT3-ITD in AML"
                        if value == "association-id"
                        else "Direct FLT3 study in AML"
                    ),
                    "pubtype": ["Journal Article"],
                }
            return FakeResponse({"result": result})
        if url == PUBMED_FETCH_URL:
            ids = [value for value in identifier.split(",") if value]
            sentences = {
                "verified-id": (
                    "FLT3 inhibition reduced viability in acute myeloid "
                    "leukemia cells."
                ),
                "weak-id": (
                    "FLT3 phosphorylation, viability, proliferation, "
                    "apoptosis, and tumor growth were measured in acute "
                    "myeloid leukemia cells."
                ),
                "association-id": (
                    "Among 100 acute myeloid leukemia patients, FLT3-ITD "
                    "mutations were detected in 20 cases (20%)."
                ),
            }
            articles = "".join(
                "<PubmedArticle><MedlineCitation>"
                f"<PMID>{value}</PMID><Article><Abstract>"
                f"<AbstractText Label='RESULTS'>{sentences[value]}</AbstractText>"
                "</Abstract></Article></MedlineCitation></PubmedArticle>"
                for value in ids
            )
            return FakeResponse(text=f"<PubmedArticleSet>{articles}</PubmedArticleSet>")
        raise AssertionError(url)


def test_verified_main_primary_survives_small_final_role_budget() -> None:
    client = _VerifiedPrimarySmallCapClient()
    brief = _brief(["PubMed"]).model_copy(
        update={
            "research_goal": (
                "Evaluate FLT3-ITD inhibition with primary cohort association."
            )
        },
        deep=True,
    )

    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=2,
        ).retrieve(brief, _seed(), 1)
    )

    assert len(client.calls) == 6
    assert {item.external_id for item in records} == {"verified-id", "weak-id"}
    assert all(
        item.metadata.get("role_specific_record")
        != "primary_target_disease_association_excerpt"
        for item in records
    )
    verified = next(item for item in records if item.external_id == "verified-id")
    assert verified.metadata["verified_direct_primary"] is True
    assert verified.metadata["primary_direction_supplement_status"] == (
        "not_requested"
    )
    assert all(
        not str(kwargs.get("params", {}).get("term", "")).endswith(
            "NOT review[Publication Type]"
        )
        for url, kwargs in client.calls
        if url == PUBMED_SEARCH_URL
    )


def test_pubmed_runs_one_distinct_bounded_counter_lane_with_primary_result() -> None:
    client = _DistinctPubMedCounterLaneClient(counter_hit=True)
    brief = _brief(["PubMed"]).model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["PubMed"],
                require_primary_sources=False,
            )
        },
        deep=True,
    )
    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=8,
            counter_evidence_enabled=True,
        ).retrieve(brief, _seed(), 1)
    )

    counter = next(item for item in records if item.supports == "counter")
    assert counter.external_id == "87654321"
    assert counter.metadata["counter_search_source"] == "PubMed"
    assert counter.metadata["counter_search_status"] == "completed_with_results"
    assert counter.metadata["counter_search_separate_query"] is True
    assert counter.metadata["counter_search_raw_document_budget"] == 20
    assert counter.metadata["counter_category"] == "resistance"
    assert counter.metadata["full_text_stored"] is False
    search_terms = [
        str(kwargs["params"]["term"])
        for url, kwargs in client.calls
        if url == PUBMED_SEARCH_URL
    ]
    assert len(search_terms) == 2
    assert search_terms[0] != search_terms[1]
    assert len(client.calls) == 6


def test_pubmed_zero_hit_counter_lane_is_completed_empty_not_main_hit() -> None:
    client = _DistinctPubMedCounterLaneClient(counter_hit=False)
    brief = _brief(["PubMed"]).model_copy(
        update={
            "data_policy": DataSourcePolicy(
                allowed_sources=["PubMed"],
                require_primary_sources=False,
            )
        },
        deep=True,
    )
    records = list(
        _retriever(
            client,
            enabled_sources=["PubMed"],
            max_records_per_source=8,
            counter_evidence_enabled=True,
        ).retrieve(brief, _seed(), 1)
    )

    assert len(records) == 1
    assert records[0].supports == "support"
    assert records[0].metadata["counter_search_status"] == "completed_empty"
    assert records[0].metadata["counter_search_raw_hit_count"] == 0
    assert records[0].metadata["counter_search_separate_query"] is True
    assert [url for url, _ in client.calls].count(PUBMED_SEARCH_URL) == 2
    assert len(client.calls) == 4


def test_target_inhibitor_response_determiner_is_not_a_drug_name() -> None:
    excerpt = (
        "A patient with ALK-rearranged IMT showed a drastic response to two "
        "ALK inhibitors."
    )
    content = _publication_content_from_sections(
        (("RESULTS", excerpt),),
        target_symbol="ALK",
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        intervention_direction="inhibit",
    )

    assert _direct_support_rejection_codes(excerpt, "ALK") == ()
    assert content.excerpt == excerpt
    assert content.direct_support_rejection_codes == ()
    assert "other_drug_response" in _direct_support_rejection_codes(
        "ALK-rearranged IMT cells showed response to venetoclax while ALK "
        "signaling was measured.",
        "ALK",
    )
    assert "combination_or_co_treatment" in _direct_support_rejection_codes(
        "Combined inhibition of ALK and MET reduced viability in "
        "ALK-rearranged IMT cells.",
        "ALK",
    )


def test_target_molecular_base_disease_aliases_are_exact_and_bounded() -> None:
    disease = "ALK-rearranged inflammatory myofibroblastic tumor"
    assert _disease_query_terms(disease, target_symbol="ALK") == (
        disease,
        "ARIMT",
        "inflammatory myofibroblastic tumor",
        "inflammatory myofibroblastic tumors",
        "IMT",
    )
    assert _disease_query_terms(
        "Acute myeloid leukemia",
        target_symbol="FLT3",
    ) == ("Acute myeloid leukemia", "AML")
    assert _disease_query_terms(
        "triple-negative breast cancer",
        target_symbol="BRCA1",
    ) == ("triple-negative breast cancer", "TNBC")
    ret_disease = "RET-rearranged renal epithelial tumor"
    assert _disease_query_terms(ret_disease, target_symbol="RET") == (
        ret_disease,
        "RRRET",
        "renal epithelial tumor",
        "renal epithelial tumors",
    )
    assert _has_strict_association_target_disease_match(
        "Among inflammatory myofibroblastic tumors, ALK rearrangements were "
        "identified in 25 cases.",
        target_symbol="ALK",
        disease=disease,
    )
    assert not _has_strict_association_target_disease_match(
        "Inflammatory myofibroblastic tumors expressed ALK protein.",
        target_symbol="ALK",
        disease=disease,
    )
    assert not _has_strict_association_target_disease_match(
        "ROS1 rearrangements were identified in inflammatory "
        "myofibroblastic tumors while ALK expression was unchanged.",
        target_symbol="ALK",
        disease=disease,
    )
    assert _has_direct_target_disease_match(
        "ALK-rearranged IMT showed a response to ALK inhibitors.",
        target_symbol="ALK",
        disease=disease,
    )
    assert not _has_direct_target_disease_match(
        "ALK-rearranged lung cancer showed a response to ALK inhibitors.",
        target_symbol="ALK",
        disease=disease,
    )
    assert not _has_direct_target_disease_match(
        "ROS1-rearranged IMT showed a response while ALK expression was "
        "measured.",
        target_symbol="ALK",
        disease=disease,
    )
    assert _has_direct_target_disease_match(
        "EGFR-mutant NSCLC responded after EGFR inhibition.",
        target_symbol="EGFR",
        disease="EGFR-mutant non-small cell lung cancer",
    )
    assert not _has_direct_target_disease_match(
        "EGFR-mutant SCLC responded after EGFR inhibition.",
        target_symbol="EGFR",
        disease="EGFR-mutant non-small cell lung cancer",
    )
    assert _has_direct_target_disease_match(
        "RET-rearranged renal epithelial tumor cells responded after RET "
        "inhibition.",
        target_symbol="RET",
        disease=ret_disease,
    )
    assert not _has_direct_target_disease_match(
        "RET-rearranged lung cancer cells responded after RET inhibition.",
        target_symbol="RET",
        disease=ret_disease,
    )
    assert not _has_direct_target_disease_match(
        "ALK rearrangement-negative IMT showed a response to ALK inhibitors.",
        target_symbol="ALK",
        disease=disease,
    )
    assert not _has_direct_target_disease_match(
        "ALK fusion-negative IMT showed a response to ALK inhibitors.",
        target_symbol="ALK",
        disease=disease,
    )
    assert not _has_direct_target_disease_match(
        "EGFR-mutant-negative NSCLC responded after EGFR inhibition.",
        target_symbol="EGFR",
        disease="EGFR-mutant non-small cell lung cancer",
    )

    lung_decoy = _publication_content_from_sections(
        (("RESULTS", "ALK inhibition reduced phosphorylation and viability "
          "in ALK-rearranged lung cancer cells."),),
        target_symbol="ALK",
        disease=disease,
        intervention_direction="inhibit",
    )
    assert lung_decoy.excerpt is None
    assert lung_decoy.excerpt_selection_status == "related_excerpt_not_found"


def test_rearrangement_primary_cohort_is_quantitative_not_background() -> None:
    sentence = (
        "Among 50 patients with inflammatory myofibroblastic tumors, ALK "
        "rearrangements were identified in 25 cases (50%)."
    )
    positive = _publication_content_from_sections(
        (("RESULTS", sentence),),
        target_symbol="ALK",
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        intervention_direction="inhibit",
    )
    background = _publication_content_from_sections(
        (("BACKGROUND", "Inflammatory myofibroblastic tumors frequently "
          "harbor ALK rearrangements."),),
        target_symbol="ALK",
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        intervention_direction="inhibit",
    )
    gene_rearrangement_sentence = (
        "Of the 62 IMTs (25 children and 37 adults), 35 (56%) showed ALK "
        "gene rearrangement."
    )
    assert _association_predicate_for_target(
        gene_rearrangement_sentence,
        "ALK",
    ) == "mutation_prevalence"
    assert _target_prevalence_quantitations(
        gene_rearrangement_sentence,
        target_symbol="ALK",
    )
    assert _has_strict_association_target_disease_match(
        gene_rearrangement_sentence,
        target_symbol="ALK",
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
    )
    assert _has_explicit_primary_disease_cohort(
        gene_rearrangement_sentence,
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        target_symbol="ALK",
    )
    assert _has_positive_prevalence_quantitation(
        gene_rearrangement_sentence,
        target_symbol="ALK",
    )
    assert _has_target_subtype_or_variant(gene_rearrangement_sentence, "ALK")
    assert _has_target_rearrangement_or_fusion(
        gene_rearrangement_sentence,
        "ALK",
    )
    gene_rearrangement = _publication_content_from_sections(
        (("section-1", gene_rearrangement_sentence),),
        target_symbol="ALK",
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        intervention_direction="inhibit",
    )

    assert positive.association_excerpt == sentence
    assert positive.association_predicate == "mutation_prevalence"
    assert positive.association_scope == "rearranged_or_fusion_target_subtype"
    assert background.association_excerpt is None
    assert gene_rearrangement.association_excerpt == gene_rearrangement_sentence
    assert gene_rearrangement.association_predicate == "mutation_prevalence"
    assert gene_rearrangement.association_scope == (
        "rearranged_or_fusion_target_subtype"
    )

    rejected = (
        "We reviewed 62 IMTs. ALK rearrangements were identified in 35 "
        "cases (56%).",
        "Of 62 lung cancers studied, 35 (56%) showed ALK gene rearrangement.",
        "Of 62 IMTs studied, 35 (90%) showed ALK gene rearrangement.",
        "Of 62 IMTs studied, 35 (56%) showed ALK gene "
        "rearrangement-negative status.",
        "Of 62 IMTs studied, 35 (56%) showed ROS1 gene rearrangement while "
        "ALK expression was measured.",
    )
    for text in rejected:
        content = _publication_content_from_sections(
            (("section-1", text),),
            target_symbol="ALK",
            disease="ALK-rearranged inflammatory myofibroblastic tumor",
            intervention_direction="inhibit",
        )
        assert content.association_excerpt is None


def test_multi_gene_parenthetical_prevalence_is_target_bound_primary_association() -> None:
    sentence = (
        "We analyzed peripheral-blood samples from 324 patients with "
        "myelofibrosis (73% with JAK2 mutations, 23% with CALR mutations, "
        "and 4% with MPL mutations)."
    )

    expected = {"JAK2": 73.0, "CALR": 23.0, "MPL": 4.0}
    for target, percent in expected.items():
        values = _target_prevalence_quantitations(
            sentence,
            target_symbol=target,
        )
        assert [value.percent for value in values] == [percent]
        assert _has_explicit_primary_disease_cohort(
            sentence,
            disease="Myelofibrosis",
            target_symbol=target,
        )
        content = _publication_content_from_sections(
            (("METHODS", sentence),),
            target_symbol=target,
            disease="Myelofibrosis",
            intervention_direction="inhibit",
        )
        assert content.association_excerpt == sentence
        assert content.association_predicate == "mutation_prevalence"


def test_multi_gene_parenthetical_prevalence_rejects_background_and_wrong_disease() -> None:
    sentence = (
        "We analyzed samples from 324 patients with myelofibrosis "
        "(73% with JAK2 mutations and 23% with CALR mutations)."
    )

    background = _publication_content_from_sections(
        (("BACKGROUND", sentence),),
        target_symbol="JAK2",
        disease="Myelofibrosis",
        intervention_direction="inhibit",
    )
    wrong_disease = _publication_content_from_sections(
        (("METHODS", sentence),),
        target_symbol="JAK2",
        disease="Acute myeloid leukemia",
        intervention_direction="inhibit",
    )
    assert background.association_excerpt is None
    assert wrong_disease.association_excerpt is None


def test_rearrangement_p2_query_uses_coordinated_title_gene_event_policy() -> None:
    brief = ResearchBrief(
        research_goal=(
            "Establish primary cohort evidence for ALK gene rearrangement."
        ),
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(allowed_sources=["PubMed"]),
    )
    seed = TargetSeed(
        symbol="ALK",
        target_name="Anaplastic lymphoma kinase",
        track=HypothesisTrack.LITERATURE,
        priority=0.9,
    )

    lanes = PublicBiologyRetriever._pubmed_primary_association_query_lanes(
        brief,
        seed,
    )

    assert lanes[1].lane_id == "title_target_variant_cohort_result_p2"
    assert '"ALK"[Title]' in lanes[1].query
    assert '"gene fusion"[Title]' in lanes[1].query
    assert '"gene fusions"[Title]' in lanes[1].query
    assert '"gene rearrangement"[Title]' in lanes[1].query
    assert '"gene rearrangements"[Title]' in lanes[1].query
    assert '"inflammatory myofibroblastic tumors"[Title]' in lanes[1].query
    assert '"ALK gene rearrangement"[Title]' not in lanes[1].query
    assert "[Title/Abstract]" not in lanes[1].query
    assert 'NOT "prevalence"[Title]' in lanes[1].query
    assert lanes[1].query.endswith("NOT (Review[Publication Type])")
    assert "PMID" not in lanes[1].query.upper()


def test_coordinated_alk_association_lane_selects_raw_rank_six_fixture() -> None:
    client = _AlkCoordinatedAssociationLaneClient()
    retriever = _retriever(
        client,
        enabled_sources=["PubMed"],
        max_records_per_source=8,
    )
    brief = ResearchBrief(
        research_goal=(
            "Establish primary cohort evidence for ALK gene rearrangement in "
            "ALK-rearranged inflammatory myofibroblastic tumor."
        ),
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=True,
        ),
    )
    requirement = _primary_association_requirement(brief, "ALK")
    lanes = PublicBiologyRetriever._pubmed_primary_association_query_lanes(
        brief,
        _alk_seed(),
        requirement=requirement,
    )

    outcome = retriever._pubmed_primary_association_supplement(
        brief,
        _alk_seed(),
        query_lanes=lanes,
        excluded_ids=set(),
        requirement=requirement,
    )

    assert len(client.calls) == 4
    assert client.ids.index(client.target_id) + 1 == 6
    assert outcome.status == "completed_with_results"
    assert outcome.stop_reason == "first_verified_required_association_selected"
    assert len(outcome.records) == 2
    parent, association = outcome.records
    assert parent.external_id == client.target_id
    assert association.external_id == client.target_id
    assert association.metadata["parent_evidence_id"] == parent.evidence_id
    assert association.metadata["association_predicate"] == (
        "mutation_prevalence"
    )
    assert association.metadata["association_scope"] == (
        "rearranged_or_fusion_target_subtype"
    )
    assert outcome.trace["http_request_budget"] == 6
    assert outcome.trace["http_requests_used"] == 4
    assert outcome.trace["raw_document_budget"] == 36
    assert outcome.trace["raw_documents_fetched"] == 11
    assert outcome.trace["records_screened"] == 6
    screened_entry = dict(outcome.trace["screening_entries"][-1])
    extraction = screened_entry.pop("extraction_observation")
    assert extraction["policy"] == "observed_evidence_gap_v1"
    assert extraction["abstract_received"] is True
    assert extraction["abstract_sha256"] == parent.metadata["content_hash"]
    assert extraction["complete_context_verified"] is True
    assert screened_entry == {
        "external_id": client.target_id,
        "lane_id": "title_target_variant_cohort_result_p2",
        "lane_rank": 6,
        "screening_rank": 6,
        "status": "selected",
        "rejection_codes": [],
    }
    assert outcome.trace["lanes"][0]["raw_document_budget"] == 20
    assert outcome.trace["lanes"][0]["http_requests_used"] == 1
    assert outcome.trace["lanes"][1]["raw_document_budget"] == 16
    assert outcome.trace["lanes"][1]["http_requests_used"] == 3


def test_coordinated_rearrangement_query_is_generic_for_ret_disease() -> None:
    brief = ResearchBrief(
        research_goal=(
            "Establish primary cohort evidence for RET gene rearrangement."
        ),
        disease="RET-rearranged renal epithelial tumor",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(allowed_sources=["PubMed"]),
    )
    seed = TargetSeed(
        symbol="RET",
        target_name="RET proto-oncogene",
        track=HypothesisTrack.LITERATURE,
        priority=0.9,
    )

    query = PublicBiologyRetriever._pubmed_primary_association_query_lanes(
        brief,
        seed,
    )[1].query

    assert '"RET"[Title]' in query
    assert '"renal epithelial tumor"[Title]' in query
    assert '"renal epithelial tumors"[Title]' in query
    assert '"gene fusion"[Title]' in query
    assert '"gene rearrangements"[Title]' in query
    assert "ALK" not in query
    assert "ROS1" not in query
    assert "PMID" not in query.upper()
    assert "[Title/Abstract]" not in query
    assert query.endswith(
        'NOT "prevalence"[Title] NOT (Review[Publication Type])'
    )
