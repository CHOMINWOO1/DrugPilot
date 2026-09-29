from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from typing import Any

import pytest

from drugpilot.agents.ma1 import (
    Critic,
    MA1Agent,
    Proposer,
    TargetCatalogResult,
    TargetSeed,
    _assign_claim_evidence,
    _direct_intervention_basis,
    _has_admissible_direction_endpoint,
    _has_verified_compound_linked_direction_pair,
    _has_target_disease_excerpt,
    _is_response_modifier_context,
    _primary_association_excerpt,
    _primary_direction_excerpt,
    _reclassify_prior_claim_evidence,
    _scope_direct_intervention_present,
    _screening_direction_triage,
    _screening_external_records,
)
from drugpilot.agents.evidence_semantics import (
    EvidenceMeaning,
    EvidenceMeaningBatch,
    attach_interpretations,
    source_records,
)
from drugpilot.agents.molecular_scope import annotate_scopes
from drugpilot.config import Settings
from drugpilot.domain.models import (
    DataSourcePolicy,
    EvidenceRef,
    HypothesisPacket,
    HypothesisTrack,
    ProjectConstraints,
    ResearchBrief,
)
from drugpilot.tools.context_router import (
    ContextRoutingBudget,
    ContextRoutingResult,
    ContextSourceOutcome,
    ContextTarget,
    ContextTargetSelection,
    TargetContextBundle,
)
from drugpilot.tools.europe_pmc_oa_fulltext import (
    OAArticleEligibility,
    parse_europe_pmc_references,
    parse_oa_jats,
)
from drugpilot.tools.ma1_context_enrichment import (
    enrich_hypothesis_packet_with_context,
)
from drugpilot.tools.public_biology import (
    PUBMED_FETCH_URL,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
    PublicBiologyRetriever,
)


def test_reported_binding_without_direct_word_is_not_overstated() -> None:
    assert _direct_intervention_basis(
        "Ningetinib bound FLT3 and reduced viability in FLT3-ITD AML cells.",
        "FLT3",
    ) == "reported_compound_target_binding_with_correlative_phenotype"
    assert _direct_intervention_basis(
        "Ningetinib bound directly to FLT3 and reduced viability.",
        "FLT3",
    ) == "direct_target_binding_and_perturbation"


_EXCERPT_POLICY = (
    "same-sentence-target-disease-direction-aware-mechanistic-ranking-v2"
)
_COMPOUND_LINKED_EXCERPT_POLICY = (
    "same-compound-target-selective-intervention-primary-outcome-pair-v1"
)
_QUERY_POLICY = "target-disease-direction-mechanistic-v2"
_ASSOCIATION_EXCERPT_POLICY = (
    "same-sentence-target-disease-primary-association-ranking-v2"
)
_OA_FULLTEXT_RETRIEVAL_POLICY = "epmc_oa_fulltext_one_hop_v1"
_OA_FULLTEXT_EXCERPT_POLICY = "europe_pmc_oa_jats_results_v1"


@pytest.mark.parametrize(
    ("target", "disease", "excerpt", "expected"),
    [
        (
            "FLT3",
            "Acute myeloid leukemia",
            "FLT3 inhibition reduced viability in AML cells.",
            True,
        ),
        (
            "BRCA1",
            "triple-negative breast cancer",
            "BRCA1 loss reduced viability in TNBC cells.",
            True,
        ),
        (
            "EGFR",
            "non-small cell lung cancer",
            "EGFR inhibition reduced viability in NSCLC cells.",
            True,
        ),
        (
            "ALK",
            "ALK-rearranged inflammatory myofibroblastic tumor",
            "ALK-rearranged IMT showed a response to ALK inhibitors.",
            True,
        ),
        (
            "ALK",
            "ALK-rearranged inflammatory myofibroblastic tumor",
            "ALK-rearranged lung cancer showed a response to ALK inhibitors.",
            False,
        ),
        (
            "ALK",
            "ALK-rearranged inflammatory myofibroblastic tumor",
            "ROS1-rearranged IMT responded while ALK expression was measured.",
            False,
        ),
        (
            "RET",
            "RET-rearranged renal epithelial tumor",
            "RET-rearranged renal epithelial tumor cells responded after "
            "RET inhibition.",
            True,
        ),
        (
            "RET",
            "RET-rearranged renal epithelial tumor",
            "RET-rearranged lung cancer cells responded after RET inhibition.",
            False,
        ),
        (
            "ALK",
            "ALK-rearranged inflammatory myofibroblastic tumor",
            "ALK rearrangement-negative IMT responded to ALK inhibitors.",
            False,
        ),
        (
            "ALK",
            "ALK-rearranged inflammatory myofibroblastic tumor",
            "ALK fusion-negative IMT responded to ALK inhibitors.",
            False,
        ),
        (
            "EGFR",
            "EGFR-mutant non-small cell lung cancer",
            "EGFR-mutant-negative NSCLC responded after EGFR inhibition.",
            False,
        ),
    ],
)
def test_target_disease_excerpt_requires_independent_exact_base_alias(
    target: str,
    disease: str,
    excerpt: str,
    expected: bool,
) -> None:
    assert _has_target_disease_excerpt(
        excerpt,
        target_symbol=target,
        disease=disease,
    ) is expected


class _StaticRetriever:
    def __init__(self, records: Sequence[EvidenceRef]) -> None:
        self.records = tuple(record.model_copy(deep=True) for record in records)

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        return tuple(record.model_copy(deep=True) for record in self.records)


class _Response:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        text: str = "",
    ) -> None:
        self.payload = payload or {}
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _PrimaryAssociationPubMedClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if url == PUBMED_SEARCH_URL:
            return _Response(
                {"esearchresult": {"count": "1", "idlist": ["20515557"]}}
            )
        if url == PUBMED_SUMMARY_URL:
            return _Response(
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
        if url == PUBMED_FETCH_URL:
            return _Response(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>20515557</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>Among 200 acute myeloid "
                    "leukemia patients, FLT3-ITD mutations were present in "
                    "59 cases (29.5%).</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                )
            )
        raise AssertionError(f"unexpected URL: {url}")


class _MolecularModelDirectionPubMedClient:
    """Offline transport for a real abstract-shaped model-context regression."""

    def __init__(self, abstract_text: str, *, pmid: str) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.abstract_text = abstract_text
        self.pmid = pmid

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        params = kwargs.get("params", {})
        if url == PUBMED_SEARCH_URL:
            query = str(params.get("term") or "")
            ids = (
                [self.pmid]
                if query.endswith("NOT review[Publication Type]")
                else []
            )
            return _Response(
                {"esearchresult": {"count": str(len(ids)), "idlist": ids}}
            )
        if url == PUBMED_SUMMARY_URL:
            return _Response(
                {
                    "result": {
                        "uids": [self.pmid],
                        self.pmid: {
                            "uid": self.pmid,
                            "title": (
                                "Effects of a target inhibitor on signaling "
                                "in a molecular cell model"
                            ),
                            "pubtype": ["Journal Article"],
                        },
                    }
                }
            )
        if url == PUBMED_FETCH_URL:
            return _Response(
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    f"<PMID>{self.pmid}</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>"
                    f"{self.abstract_text}"
                    "</AbstractText></Abstract></Article></MedlineCitation>"
                    "</PubmedArticle></PubmedArticleSet>"
                )
            )
        raise AssertionError(f"unexpected URL: {url}")


def _brief() -> ResearchBrief:
    return ResearchBrief(
        brief_id="brief_atomic_claim_test",
        research_goal=(
            "Evaluate FLT3-ITD inhibition with claim-specific primary evidence."
        ),
        disease="Acute myeloid leukemia",
        disease_id="EFO_0000222",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed", "Europe PMC", "Open Targets", "UniProt"]
        ),
    )


def _publication(
    evidence_id: str,
    excerpt: str,
    *,
    source: str = "PubMed",
    supports: str = "support",
    publication_types: Sequence[str] = ("Journal Article",),
    title: str | None = None,
    counter_category: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source=source,
        external_id=evidence_id.removeprefix("ev_"),
        title=title or f"Primary experiment {evidence_id}",
        supports=supports,
        excerpt_hash=f"excerpt-hash-{evidence_id}",
        metadata={
            "excerpt": excerpt,
            "content_availability": "abstract_excerpt",
            "source_location": "abstract:RESULTS:chars:0-180",
            "content_hash": f"content-hash-{evidence_id}",
            "full_text_stored": False,
            "excerpt_selection_policy": _EXCERPT_POLICY,
            "query_strategy": _QUERY_POLICY,
            "query_intervention_direction": "inhibit",
            "provenance_class": "external_live",
            "publication_types": list(publication_types),
            "counter_category": counter_category,
            "review_status": (
                "researcher_review_required" if counter_category else None
            ),
            "claim_linkage_status": "pending_ma1_claim_reconciliation",
            "mock": False,
        },
    )


def test_producer_sensitization_rejection_is_preserved_as_modifier_context() -> None:
    evidence = _publication(
        "ev_sensitization_context",
        "Ruxolitinib inhibited JAK2 and increased the sensitivity of acute "
        "myeloid leukemia cells to arsenic trioxide.",
    )
    evidence = evidence.model_copy(
        update={
            "metadata": {
                **evidence.metadata,
                "direct_support_eligible": False,
                "direct_support_rejection_codes": [
                    "response_modifier",
                    "sensitization_to_other_intervention",
                ],
            }
        },
        deep=True,
    )

    assert _is_response_modifier_context("JAK2", evidence) is True


def test_other_agent_treated_cells_are_not_direct_target_direction_support() -> None:
    evidence = _publication(
        "ev_combination_without_keyword",
        "An ATM kinase inhibitor also significantly reduced the viability of "
        "ETO-treated breast cancer cells.",
    )

    assert _is_response_modifier_context("ATM", evidence) is True


def test_coordinated_target_inhibition_is_combination_context() -> None:
    evidence = _publication(
        "ev_coordinated_combination",
        "Primary AML cells showed significantly better responses to combined "
        "menin and FLT3 inhibition than to single-drug or vehicle control "
        "treatment.",
    )

    assert _is_response_modifier_context("FLT3", evidence) is True


def test_canonical_engagement_in_scoped_model_is_not_scoped_intervention() -> None:
    excerpt = (
        "XY0206 directly engages FLT3 and suppresses downstream signaling, "
        "resulting in apoptosis in FLT3-ITD AML cells."
    )

    assert _scope_direct_intervention_present(
        excerpt,
        "FLT3",
        "inhibit",
        "FLT3:ITD",
    ) is False


def test_canonical_engagement_is_not_assigned_to_scoped_direction_claim() -> None:
    brief = _brief()
    excerpt = (
        "XY0206 directly engages FLT3 and suppresses downstream signaling, "
        "resulting in apoptosis in FLT3-ITD AML cells."
    )
    evidence = _publication("ev_scoped_engagement", excerpt).model_copy(
        update={"excerpt_hash": sha256(excerpt.encode("utf-8")).hexdigest()},
        deep=True,
    )
    annotated = annotate_scopes(brief, "FLT3", [evidence])
    records = source_records(annotated)
    assert len(records) == 1
    meaning = EvidenceMeaning(
        record_key=records[0]["record_key"],
        study_context="measured_primary",
        target_relation="direct",
        disease_context="actual_model",
        intervention="direct_requested",
        outcome="favorable",
        target_quote="FLT3-ITD",
        model_quote="FLT3-ITD AML cells",
        intervention_quote="XY0206 directly engages FLT3",
        result_quote="resulting in apoptosis in FLT3-ITD AML cells",
        molecular_scope_label="FLT3:ITD",
    )
    interpreted = attach_interpretations(
        brief,
        "FLT3",
        annotated,
        records,
        EvidenceMeaningBatch(records=[meaning]),
        EvidenceMeaningBatch(records=[meaning]),
    )

    assignment = _assign_claim_evidence(
        brief,
        target_symbol="FLT3",
        evidence=interpreted,
        scope_label="FLT3:ITD",
    )

    assert assignment.direction_primary_ids == ()


def test_same_doi_with_distinct_erratum_pmid_counts_as_one_publication() -> None:
    article = _publication("ev_11111111", "FLT3 inhibition reduced viability in AML cells.")
    erratum = _publication(
        "ev_22222222",
        "Correction to the reported FLT3 experiment.",
        publication_types=("Published Erratum",),
    )
    shared_doi = "10.1000/shared.article"
    article.metadata.update({"pmid": "11111111", "doi": shared_doi})
    erratum.metadata.update({"pmid": "22222222", "doi": shared_doi.upper()})
    hypothesis = _hypothesis().model_copy(update={"evidence": [article, erratum]}, deep=True)

    records, total = _screening_external_records(hypothesis)

    assert total == 1
    assert len(records) == 1
    assert records[0]["canonical_identity"] == "DOI:10.1000/shared.article"
    assert set(records[0]["external_ids"]) == {"11111111", "22222222"}


def _canonical_hash(payload: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_compound_linked_preclinical_result_is_reverified_by_ma1() -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct JAK2 inhibition only in myelofibrosis.",
        disease="myelofibrosis",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
    )
    excerpt = (
        "Myelofibrosis (MF) Fedratinib, a JAK2-selective inhibitor, was "
        "evaluated in MF cells. Fedratinib significantly reduced viability "
        "in MF cells."
    )
    record = _publication("ev_fedratinib", excerpt).model_copy(
        update={
            "metadata": {
                **_publication("ev_fedratinib", excerpt).metadata,
                "source_location": (
                    "abstract-paired:abstract:IMPORTANCE:chars:0-18|"
                    "abstract:BACKGROUND:chars:0-88|"
                    "abstract:RESULTS:chars:0-59"
                ),
                "excerpt_selection_status": (
                    "selected_compound_linked_document_pair"
                ),
                "excerpt_selection_policy": _COMPOUND_LINKED_EXCERPT_POLICY,
                "query_strategy": (
                    "brief-derived-target-action-experimental-direction-primary-v3"
                ),
                "verified_direct_primary": True,
                "verified_direct_primary_rejection_codes": [],
            }
        },
        deep=True,
    )

    assert _has_verified_compound_linked_direction_pair(
        brief, "JAK2", record, excerpt
    ) is True
    assert _primary_direction_excerpt(brief, "JAK2", record) == excerpt
    assert _assign_claim_evidence(
        brief,
        target_symbol="JAK2",
        evidence=[record],
    ).direction_primary_ids == (record.evidence_id,)

    hypothesis = Proposer(_StaticRetriever([record]))._build_hypothesis(
        brief,
        TargetSeed(
            symbol="JAK2",
            target_name="Janus kinase 2",
            track=HypothesisTrack.LITERATURE,
            priority=1.0,
        ),
        1,
    )
    mechanism_unit = hypothesis.target.metadata[
        "claim_evidence_assignment"
    ]["mechanism_claims"][0]
    assert mechanism_unit["measured_endpoints"] == ["viability"]


def test_compound_linked_clinical_result_rejects_different_outcome_drug() -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct JAK2 inhibition only in myelofibrosis.",
        disease="myelofibrosis",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
    )
    excerpt = (
        "Myelofibrosis (MF) Fedratinib, a JAK2-selective inhibitor, was "
        "evaluated in MF cells. Pacritinib significantly reduced viability "
        "in MF cells."
    )
    base = _publication("ev_mismatched_drug", excerpt)
    record = base.model_copy(
        update={
            "metadata": {
                **base.metadata,
                "source_location": (
                    "abstract-paired:abstract:BACKGROUND:chars:0-100|"
                    "abstract:RESULTS:chars:0-80"
                ),
                "excerpt_selection_status": (
                    "selected_compound_linked_document_pair"
                ),
                "excerpt_selection_policy": _COMPOUND_LINKED_EXCERPT_POLICY,
                "query_strategy": (
                    "brief-derived-target-action-experimental-direction-primary-v3"
                ),
                "verified_direct_primary": True,
                "verified_direct_primary_rejection_codes": [],
            }
        },
        deep=True,
    )

    assert _primary_direction_excerpt(brief, "JAK2", record) is None


def test_compound_linked_clinical_outcome_is_not_direct_mechanistic_evidence() -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct JAK2 inhibition only in myelofibrosis.",
        disease="myelofibrosis",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
    )
    excerpt = (
        "Myelofibrosis (MF) Fedratinib, a JAK2-selective inhibitor, was studied "
        "in patients with MF. Fedratinib therapy significantly reduced "
        "splenomegaly and symptom burden in patients with MF."
    )
    base = _publication("ev_clinical_context", excerpt)
    record = base.model_copy(
        update={
            "metadata": {
                **base.metadata,
                "source_location": (
                    "abstract-paired:abstract:BACKGROUND:chars:0-99|"
                    "abstract:CONCLUSIONS:chars:0-101"
                ),
                "excerpt_selection_status": (
                    "selected_compound_linked_document_pair"
                ),
                "excerpt_selection_policy": _COMPOUND_LINKED_EXCERPT_POLICY,
                "query_strategy": (
                    "brief-derived-target-action-experimental-direction-primary-v3"
                ),
                "verified_direct_primary": True,
                "verified_direct_primary_rejection_codes": [],
            }
        },
        deep=True,
    )

    assert _has_verified_compound_linked_direction_pair(
        brief, "JAK2", record, excerpt
    ) is False
    assert _primary_direction_excerpt(brief, "JAK2", record) is None


def _oa_fulltext_publication(
    evidence_id: str | None = None,
    excerpt: str = (
        "Ningetinib directly inhibited FLT3 phosphorylation and reduced cell "
        "viability in acute myeloid leukemia models."
    ),
    *,
    hop: int = 0,
    title: str = "Primary OA experiment of FLT3 inhibition in AML",
    publication_types: Sequence[str] = ("Journal Article",),
    metadata_updates: dict[str, Any] | None = None,
) -> EvidenceRef:
    pmid = "12345678"
    pmcid = "PMC12345678"
    child = f"MED:{pmid}"
    identity_keys = [f"MED:{pmid}", f"PMC:{pmcid}"]
    if hop == 0:
        parent_pmcid = None
        parent_pmid = None
        parent_external_id = None
        reference_position = None
        relation = "direct_search"
        root = child
        lineage_payload = {
            "policy": _OA_FULLTEXT_RETRIEVAL_POLICY,
            "hop": 0,
            "relation": relation,
            "child": child,
        }
    else:
        parent_pmcid = "PMC11111111"
        parent_pmid = "11111111"
        parent_external_id = parent_pmid
        reference_position = 3
        relation = "reference"
        root = parent_external_id
        lineage_payload = {
            "policy": "europe_pmc_reference_one_hop_v1",
            "hop": 1,
            "relation": relation,
            "parent_pmcid": parent_pmcid,
            "parent_pmid": parent_pmid,
            "position": reference_position,
            "child": child,
            "identity_keys": sorted(identity_keys),
        }
    excerpt_hash = sha256(excerpt.encode("utf-8")).hexdigest()
    full_text_hash = sha256(b"bounded-jats-document").hexdigest()
    origin_evidence_id = "ev_" + sha256(
        (
            f"Europe PMC|{pmid}|oa_fulltext_excerpt|{hop}|{relation}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    metadata = {
        "source_database": "Europe PMC",
        "discovery_origin": (
            "reference_one_hop" if hop == 1 else "direct_search"
        ),
        "evidence_origin": (
            "oa_results_fulltext_reference_one_hop"
            if hop == 1
            else "oa_results_fulltext_direct"
        ),
        "excerpt": excerpt,
        "excerpt_hash": excerpt_hash,
        "full_text_span_excerpt_hash": excerpt_hash,
        "excerpt_char_count": len(excerpt),
        "excerpt_max_chars": 320,
        "excerpt_truncated": False,
        "original_sentence_char_count": len(excerpt),
        "content_hash": full_text_hash,
        "full_text_content_hash": full_text_hash,
        "eligible_content_hash": sha256(b"eligible-results").hexdigest(),
        "content_hash_algorithm": "sha256",
        "content_hash_scope": "europe_pmc_oa_jats_xml",
        "content_availability": "oa_fulltext_excerpt",
        "source_location": (
            f"full_text:results:sec:1:p:1:sentence:1:chars:0-{len(excerpt)}"
        ),
        "source_location_scope": "normalized_admitted_paragraph_text",
        "source_location_char_end_exclusive": True,
        "full_text_stored": False,
        "raw_xml_stored": False,
        "content_storage_policy": "bounded_oa_excerpt_only_no_full_text",
        "content_retrieval_policy": _OA_FULLTEXT_RETRIEVAL_POLICY,
        "excerpt_selection_policy": _OA_FULLTEXT_EXCERPT_POLICY,
        "evidence_id_origin_policy": (
            "source-external-content-origin-hop-relation-v1"
        ),
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": None,
        "is_open_access": True,
        "full_text_license": "https://creativecommons.org/licenses/by/4.0/",
        "publication_status": "primary_research_article",
        "publication_types": list(publication_types),
        "input_byte_count": len(b"bounded-jats-document"),
        "eligible_section_count": 1,
        "eligible_paragraph_count": 1,
        "emitted_span_count": 1,
        "spans_truncated_by_limit": False,
        "citation_hop": hop,
        "citation_depth": hop,
        "citation_relation": relation,
        "citation_parent_external_id": parent_external_id,
        "citation_parent_pmcid": parent_pmcid,
        "citation_parent_pmid": parent_pmid,
        "citation_reference_position": reference_position,
        "citation_child_canonical_id": child,
        "citation_child_identity_keys": sorted(identity_keys),
        "citation_child_pmcid": pmcid,
        "citation_child_pmid": pmid,
        "citation_child_doi": None,
        "citation_root_external_id": root,
        "citation_lineage_sha256": _canonical_hash(lineage_payload),
        "lineage_verified": True,
        "evidence_role_inherited": False,
        "independent_primary_verification_required": True,
        "independent_primary_verification": True,
        "provenance_class": "external_live",
        "mock": False,
        "counter_category": None,
        "review_status": None,
        "claim_linkage_status": "pending_ma1_claim_reconciliation",
        "section_kind": "results",
        "section_title": "Results",
    }
    metadata.update(metadata_updates or {})
    return EvidenceRef(
        evidence_id=evidence_id or origin_evidence_id,
        source="Europe PMC",
        external_id=pmid,
        title=title,
        supports="support",
        excerpt_hash=excerpt_hash,
        metadata=metadata,
    )


def test_oa_downstream_only_phosphorylation_is_not_direct_target_support() -> None:
    excerpt = (
        "Treatment with FLT3 inhibitors quizartinib and midostaurin suppressed "
        "RSK1 phosphorylation in FLT3-ITD mutant acute myeloid leukemia cells."
    )
    record = _oa_fulltext_publication(excerpt=excerpt)

    assert _has_admissible_direction_endpoint(excerpt, "FLT3") is False
    assert _primary_direction_excerpt(_brief(), "FLT3", record) is None


def test_target_anchored_or_phenotypic_endpoints_remain_admissible() -> None:
    assert _has_admissible_direction_endpoint(
        "Quizartinib suppressed FLT3 phosphorylation in AML cells.",
        "FLT3",
    ) is True
    assert _has_admissible_direction_endpoint(
        "An FLT3 inhibitor reduced viability in AML cells.",
        "FLT3",
    ) is True


def _oa_fulltext_association(
    *,
    hop: int = 0,
    metadata_updates: dict[str, Any] | None = None,
    title: str = "Primary OA cohort analysis of FLT3-ITD AML",
) -> EvidenceRef:
    excerpt = (
        "In our cohort of 123 acute myeloid leukemia patients, FLT3-ITD "
        "mutations were detected in 24% of cases."
    )
    record = _oa_fulltext_publication(
        excerpt=excerpt,
        hop=hop,
        title=title,
    )
    parent_evidence_id = "ev_" + sha256(
        (
            f"Europe PMC|{record.external_id}|oa_fulltext_excerpt|"
            f"{hop}|{record.metadata['citation_relation']}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    association_evidence_id = "ev_" + sha256(
        (
            f"{parent_evidence_id}|"
            "primary_target_disease_association_excerpt"
        ).encode("utf-8")
    ).hexdigest()[:12]
    metadata = {
        **record.metadata,
        "excerpt_selection_status": "selected_primary_association",
        "association_excerpt_selection_policy": _ASSOCIATION_EXCERPT_POLICY,
        "excerpt_relevance_basis": [
            "target_symbol",
            "disease",
            "primary_association_result",
            "association_predicate",
        ],
        "role_specific_record": (
            "primary_target_disease_association_excerpt"
        ),
        "association_predicate": "mutation_prevalence",
        "association_scope": "activating_or_itd_target_subtype",
        "association_excerpt_relevance_score": 28,
        "parent_evidence_id": parent_evidence_id,
        "evidence_id_origin_policy": (
            "source-external-content-origin-hop-relation-v1"
        ),
        "parent_excerpt_hash": record.excerpt_hash,
        "association_source_excerpt_hash": record.excerpt_hash,
        "parent_provenance_pointer_type": (
            "selected_oa_document_record_id_not_claim_link"
        ),
        "same_document_external_id": record.external_id,
        "duplicate_document_role": "association_excerpt",
        "additional_network_call": False,
        "direct_support_eligible": False,
        "direct_support_rejection_codes": ["association_role_only"],
        "verified_direct_primary": False,
        "verified_direct_primary_rejection_codes": ["association_role_only"],
        # An association-only parent is not direct support; the association
        # child is independently admitted by its own predicate/scope gates.
        "review_status": "researcher_review_required",
    }
    metadata.update(metadata_updates or {})
    return record.model_copy(
        update={
            "evidence_id": association_evidence_id,
            "metadata": metadata,
        },
        deep=True,
    )


def _oa_fulltext_association_parent(
    association: EvidenceRef,
) -> EvidenceRef:
    parent = _oa_fulltext_publication(
        evidence_id=str(association.metadata["parent_evidence_id"]),
        excerpt=str(association.metadata["excerpt"]),
        hop=int(association.metadata["citation_hop"]),
        title=association.title,
        metadata_updates={
            "evidence_id_origin_policy": (
                "source-external-content-origin-hop-relation-v1"
            ),
            "review_status": "researcher_review_required",
        },
    )
    return parent.model_copy(update={"supports": "context"}, deep=True)


def _association(
    *,
    excerpt_policy: str = _ASSOCIATION_EXCERPT_POLICY,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id="ev_association",
        source="PubMed",
        external_id="20515557",
        title="Primary cohort analysis of FLT3-ITD AML",
        supports="support",
        excerpt_hash="association-hash",
        metadata={
            "excerpt": (
                "In our cohort of 123 acute myeloid leukemia patients, "
                "FLT3-ITD mutations were detected in 24% of cases."
            ),
            "content_availability": "abstract_excerpt",
            "source_location": "abstract:RESULTS:chars:0-112",
            "content_hash": "association-content-hash",
            "full_text_stored": False,
            "excerpt_selection_policy": excerpt_policy,
            "role_specific_record": (
                "primary_target_disease_association_excerpt"
            ),
            "association_predicate": "mutation_prevalence",
            "association_scope": "activating_or_itd_target_subtype",
            "association_excerpt_relevance_score": 28,
            "parent_evidence_id": "ev_association_parent",
            "parent_excerpt_hash": "parent-excerpt-hash",
            "same_document_external_id": "20515557",
            "duplicate_document_role": "association_excerpt",
            "additional_network_call": False,
            "provenance_class": "external_live",
            "publication_types": ["Journal Article"],
            "claim_linkage_status": "pending_ma1_claim_reconciliation",
            "mock": False,
        },
    )


def _ranked_association() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="ev_ot_association",
        source="Open Targets",
        external_id="ENSG00000122025",
        title="Open Targets ranked FLT3 association",
        supports="support",
        excerpt_hash="ot-association-hash",
        metadata={
            "relationship": "ranked_association",
            "record_type": "target_disease_association",
            "provenance_class": "external_live",
            "claim_linkage_status": "pending_ma1_claim_reconciliation",
            "mock": False,
        },
    )


def _records() -> list[EvidenceRef]:
    return [
        _publication(
            "ev_direct",
            "Ningetinib directly bound FLT3, inhibited FLT3 phosphorylation, "
            "and reduced cell viability in acute myeloid leukemia models.",
        ),
        _publication(
            "ev_direct_flc8",
            "FLC-8 suppressed FLT3-mediated signaling and induced apoptosis "
            "in FLT3-ITD acute myeloid leukemia cells.",
        ).model_copy(
            update={
                "metadata": {
                    **_publication(
                        "ev_direct_flc8",
                        "FLC-8 suppressed FLT3-mediated signaling and induced "
                        "apoptosis in FLT3-ITD acute myeloid leukemia cells.",
                    ).metadata,
                    "direct_support_eligible": False,
                    "verified_direct_primary": False,
                    "direct_support_rejection_codes": [
                        "direct_target_intervention_missing"
                    ],
                    "verified_direct_primary_rejection_codes": [
                        "direct_target_intervention_missing"
                    ],
                }
            },
            deep=True,
        ),
        _publication(
            "ev_pp2a_modifier",
            "PP2A inhibition enhanced the anti-leukemia effects of FLT3 "
            "inhibitors and reduced viability in acute myeloid leukemia cells.",
        ),
        _publication(
            "ev_nox2_modifier",
            "Inhibition of NOX2 sensitized FLT3-mutant acute myeloid leukemia "
            "cells and increased apoptosis in response to FLT3 inhibitors.",
        ),
        _publication(
            "ev_atr_other_target",
            "An ATR inhibitor more effectively inhibited cell proliferation "
            "in FLT3-mutated acute myeloid leukemia cells.",
        ).model_copy(
            update={
                "metadata": {
                    **_publication(
                        "ev_atr_other_target",
                        "An ATR inhibitor more effectively inhibited cell "
                        "proliferation in FLT3-mutated acute myeloid leukemia "
                        "cells.",
                    ).metadata,
                    "direct_support_eligible": True,
                    "verified_direct_primary": True,
                }
            },
            deep=True,
        ),
        _publication(
            "ev_rsk1_other_target",
            "RSK1 inhibition decreased FLT3 phosphorylation in acute myeloid "
            "leukemia cells.",
        ),
        _publication(
            "ev_ato_other_intervention",
            "Arsenic trioxide inhibited proliferation in FLT3-ITD acute myeloid "
            "leukemia cells.",
        ),
        _publication(
            "ev_venetoclax_other_intervention",
            "Venetoclax inhibited viability and induced apoptosis in FLT3-mutant "
            "acute myeloid leukemia cells.",
        ),
        _publication(
            "ev_prevalence",
            "FLT3 mutations are frequent and predict poor survival in acute "
            "myeloid leukemia.",
        ),
        _publication(
            "ev_review",
            "FLT3 inhibition reduced phosphorylation and viability in acute "
            "myeloid leukemia models.",
            publication_types=("Review",),
            title="Systematic review of FLT3 biology",
        ),
        _publication(
            "ev_potency_assertion",
            "A potent and selective inhibitor directly targets FLT3 in acute "
            "myeloid leukemia.",
        ),
        _publication(
            "ev_unverified_type",
            "FLT3 inhibition reduced phosphorylation and viability in acute "
            "myeloid leukemia cells.",
            source="Europe PMC",
            publication_types=(),
        ),
        _publication(
            "ev_resistance",
            "Acquired resistance to FLT3 inhibition limited response in acute "
            "myeloid leukemia models.",
            source="Europe PMC",
            supports="counter",
            publication_types=("research article",),
            counter_category="resistance",
        ),
        EvidenceRef(
            evidence_id="ev_context",
            source="UniProt",
            external_id="P36888",
            title="UniProt FLT3 identity",
            supports="context",
            excerpt_hash="context-hash",
            metadata={"provenance_class": "external_live", "mock": False},
        ),
    ]


def _hypothesis():
    association = _association()
    ranked_association = _ranked_association()
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        ensembl_id="ENSG00000122025",
        discovery_source="external_dynamic",
        discovery_evidence=(association, ranked_association),
    )
    hypothesis = Proposer(_StaticRetriever(_records()))._build_hypothesis(
        _brief(),
        seed,
        1,
    )
    return hypothesis


def test_screening_direction_triage_uses_only_persisted_primary_linkage() -> None:
    aligned = _hypothesis()
    aligned_triage = _screening_direction_triage(aligned)
    assert aligned_triage["status"] == "aligned_direct_primary"
    assert aligned_triage["direct_primary_evidence_ids"]
    assert aligned_triage["association_primary_evidence_ids"]
    assert aligned_triage["inference_forbidden"] is True

    unknown = aligned.model_copy(deep=True)
    assignment = unknown.target.metadata["claim_evidence_assignment"]
    assignment["claim_units"] = [
        unit
        for unit in assignment["claim_units"]
        if unit["role"] != "direct_target_direction_primary"
    ]
    assignment["direction_refutation_evidence_ids"] = []
    assert _screening_direction_triage(unknown)["status"] == (
        "unknown_no_direct_primary"
    )

    discordant = unknown.model_copy(deep=True)
    discordant.target.metadata["claim_evidence_assignment"][
        "direction_refutation_evidence_ids"
    ] = [aligned_triage["direct_primary_evidence_ids"][0]]
    discordant_triage = _screening_direction_triage(discordant)
    assert discordant_triage["status"] == "discordant_direct_primary"
    assert discordant_triage["basis"] == (
        "linked_direct_target_direction_refutation"
    )

    unresolved = aligned.model_copy(deep=True)
    direct_id = aligned_triage["direct_primary_evidence_ids"][0]
    direct_record = next(
        item for item in unresolved.evidence if item.evidence_id == direct_id
    )
    direct_record.metadata["association_scope"] = (
        "activating_or_itd_target_subtype"
    )
    direct_record.metadata.pop("predeclared_target_variants", None)
    unresolved_triage = _screening_direction_triage(unresolved)
    assert unresolved_triage["status"] == "variant_scope_unresolved"
    assert unresolved_triage["variant_scope_evidence_ids"] == [direct_id]
    assert unresolved_triage["predeclared_target_variants"] == []


@pytest.mark.parametrize(
    ("policy", "accepted"),
    [
        (
            "same-sentence-target-disease-primary-association-ranking-v1",
            False,
        ),
        (_ASSOCIATION_EXCERPT_POLICY, True),
    ],
)
def test_primary_association_policy_is_exact_and_fail_closed(
    policy: str,
    accepted: bool,
) -> None:
    association = _association(excerpt_policy=policy)
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        discovery_source="external_dynamic",
        discovery_evidence=(association,),
    )

    hypothesis = Proposer(_StaticRetriever([]))._build_hypothesis(
        _brief(),
        seed,
        1,
    )

    linked_claims = [
        claim
        for claim in hypothesis.claims
        if association.evidence_id in claim.evidence_ids
    ]
    assert bool(linked_claims) is accepted
    record = next(
        item
        for item in hypothesis.evidence
        if item.evidence_id == association.evidence_id
    )
    assert record.metadata["excerpt_selection_policy"] == policy
    assert record.supports == ("support" if accepted else "context")
    assert record.metadata["linked_claim_ids"] == (
        [linked_claims[0].claim_id] if accepted else []
    )


@pytest.mark.parametrize("hop", [0, 1])
def test_epmc_oa_fulltext_direction_contract_accepts_only_verified_hops(
    hop: int,
) -> None:
    record = _oa_fulltext_publication(hop=hop)

    assert _primary_direction_excerpt(_brief(), "FLT3", record) == record.metadata[
        "excerpt"
    ]


def test_epmc_oa_parser_metadata_is_accepted_by_ma1_without_contract_translation() -> None:
    license_uri = "https://creativecommons.org/licenses/by/4.0/"
    excerpt = (
        "FLT3 inhibition reduced phosphorylation and viability in acute "
        "myeloid leukemia cells."
    )
    jats = f"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <article-id pub-id-type="pmc">PMC12345678</article-id>
    <article-id pub-id-type="pmid">12345678</article-id>
    <article-id pub-id-type="doi">10.1000/direct-child</article-id>
    <permissions><license xlink:href="{license_uri}" /></permissions>
  </article-meta></front>
  <body><sec sec-type="results"><title>Results</title><p>{excerpt}</p></sec></body>
</article>"""
    result = parse_oa_jats(
        jats,
        eligibility=OAArticleEligibility(
            pmcid="PMC12345678",
            pmid="12345678",
            publication_types=("Journal Article",),
            is_open_access=True,
            license_uri=license_uri,
            doi="10.1000/direct-child",
        ),
        allowed_license_uris={license_uri},
    )
    metadata = result.evidence_metadata(result.spans[0])
    metadata.update(
        {
            "full_text_span_excerpt_hash": result.spans[0].excerpt_sha256,
            "evidence_id_origin_policy": (
                "source-external-content-origin-hop-relation-v1"
            ),
            "raw_xml_stored": False,
            "provenance_class": "external_live",
            "mock": False,
            "counter_category": None,
            "review_status": None,
            "claim_linkage_status": "pending_ma1_claim_reconciliation",
        }
    )
    evidence_id = "ev_" + sha256(
        b"Europe PMC|12345678|oa_fulltext_excerpt|0|direct_search"
    ).hexdigest()[:12]
    record = EvidenceRef(
        evidence_id=evidence_id,
        source="Europe PMC",
        external_id="12345678",
        title="Primary OA experiment of FLT3 inhibition in AML",
        supports="support",
        excerpt_hash=result.spans[0].excerpt_sha256,
        metadata=metadata,
    )

    assert _primary_direction_excerpt(_brief(), "FLT3", record) == excerpt


def test_epmc_oa_direct_doi_is_exactly_bound_to_lineage_identity() -> None:
    doi = "10.1000/direct-lineage"
    record = _oa_fulltext_publication(
        metadata_updates={
            "doi": doi,
            "citation_child_doi": doi,
            "citation_child_identity_keys": [
                "DOI:10.1000/direct-lineage",
                "MED:12345678",
                "PMC:PMC12345678",
            ],
        }
    )

    assert _primary_direction_excerpt(_brief(), "FLT3", record) == record.metadata[
        "excerpt"
    ]
    record.metadata["citation_child_doi"] = None
    assert _primary_direction_excerpt(_brief(), "FLT3", record) is None


def test_epmc_oa_pmid_only_reference_edge_accepts_resolved_pmc_child() -> None:
    license_uri = "https://creativecommons.org/licenses/by/4.0/"
    excerpt = (
        "FLT3 inhibition reduced phosphorylation and viability in acute "
        "myeloid leukemia cells."
    )
    lineage = parse_europe_pmc_references(
        {
            "referenceList": {
                "reference": [
                    {"citedOrder": 1, "source": "MED", "id": "12345678"}
                ]
            }
        },
        parent_pmcid="PMC11111111",
        parent_pmid="11111111",
    ).references[0]
    result = parse_oa_jats(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <article-id pub-id-type="pmc">PMC12345678</article-id>
    <article-id pub-id-type="pmid">12345678</article-id>
    <permissions><license xlink:href="{license_uri}" /></permissions>
  </article-meta></front>
  <body><sec sec-type="results"><p>{excerpt}</p></sec></body>
</article>""",
        eligibility=OAArticleEligibility(
            pmcid="PMC12345678",
            pmid="12345678",
            publication_types=("Journal Article",),
            is_open_access=True,
            license_uri=license_uri,
        ),
        allowed_license_uris={license_uri},
    )
    metadata = result.evidence_metadata(result.spans[0], lineage=lineage)
    metadata.update(
        {
            "full_text_span_excerpt_hash": result.spans[0].excerpt_sha256,
            "evidence_id_origin_policy": (
                "source-external-content-origin-hop-relation-v1"
            ),
            "raw_xml_stored": False,
            "provenance_class": "external_live",
            "mock": False,
            "counter_category": None,
            "review_status": None,
        }
    )
    evidence_id = "ev_" + sha256(
        b"Europe PMC|12345678|oa_fulltext_excerpt|1|reference"
    ).hexdigest()[:12]
    record = EvidenceRef(
        evidence_id=evidence_id,
        source="Europe PMC",
        external_id="12345678",
        title="Primary OA experiment of FLT3 inhibition in AML",
        supports="support",
        excerpt_hash=result.spans[0].excerpt_sha256,
        metadata=metadata,
    )

    assert metadata["citation_child_pmcid"] is None
    assert metadata["citation_child_identity_keys"] == ["MED:12345678"]
    assert _primary_direction_excerpt(_brief(), "FLT3", record) == excerpt


@pytest.mark.parametrize(
    "required_provenance_field",
    [
        "source_database",
        "discovery_origin",
        "evidence_origin",
        "evidence_id_origin_policy",
        "citation_depth",
        "original_sentence_char_count",
        "section_kind",
        "section_title",
        "source_location_scope",
        "source_location_char_end_exclusive",
    ],
)
def test_epmc_oa_required_provenance_metadata_fails_closed_when_missing(
    required_provenance_field: str,
) -> None:
    record = _oa_fulltext_publication()
    record.metadata.pop(required_provenance_field)

    assert _primary_direction_excerpt(_brief(), "FLT3", record) is None


def test_epmc_oa_titleless_results_section_preserves_exact_provenance() -> None:
    record = _oa_fulltext_publication(metadata_updates={"section_title": ""})

    assert _primary_direction_excerpt(_brief(), "FLT3", record) == record.metadata[
        "excerpt"
    ]


@pytest.mark.parametrize("hop", [0, 1])
def test_epmc_oa_fulltext_primary_association_is_admitted_for_verified_hops(
    hop: int,
) -> None:
    association = _oa_fulltext_association(hop=hop)
    parent = _oa_fulltext_association_parent(association)

    assert _primary_association_excerpt(
        _brief(),
        "FLT3",
        association,
    ) == association.metadata["excerpt"]

    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        discovery_source="external_dynamic",
        discovery_evidence=(association,),
    )
    hypothesis = Proposer(
        _StaticRetriever(
            [
                _publication(
                    "ev_oa_mechanism",
                    "Ningetinib directly inhibited FLT3 phosphorylation and "
                    "reduced cell viability in acute myeloid leukemia models.",
                ),
                parent,
            ]
        )
    )._build_hypothesis(_brief(), seed, 1)
    association_claims = [
        claim
        for claim in hypothesis.claims
        if association.evidence_id in claim.evidence_ids
    ]

    assert len(association_claims) == 1
    linked = next(
        item
        for item in hypothesis.evidence
        if item.evidence_id == association.evidence_id
    )
    assert linked.metadata["claim_linkage_roles"] == [
        "primary_target_disease_association"
    ]
    assert linked.metadata["evidence_role_inherited"] is False


def test_epmc_oa_fulltext_association_assignment_requires_bound_parent() -> None:
    association = _oa_fulltext_association()
    parent = _oa_fulltext_association_parent(association)

    without_parent = _assign_claim_evidence(
        _brief(),
        target_symbol="FLT3",
        evidence=(association,),
    )
    with_parent = _assign_claim_evidence(
        _brief(),
        target_symbol="FLT3",
        evidence=(parent, association),
    )

    assert without_parent.association_primary_ids == ()
    assert with_parent.association_primary_ids == (association.evidence_id,)


def test_epmc_oa_fulltext_association_assignment_rejects_wrong_parent_hash() -> None:
    association = _oa_fulltext_association(
        metadata_updates={"parent_excerpt_hash": "0" * 64},
    )
    parent = _oa_fulltext_association_parent(association)

    assert _primary_association_excerpt(
        _brief(),
        "FLT3",
        association,
    ) == association.metadata["excerpt"]
    assignment = _assign_claim_evidence(
        _brief(),
        target_symbol="FLT3",
        evidence=(parent, association),
    )

    assert assignment.association_primary_ids == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("association_excerpt_selection_policy", "association-policy-v0"),
        ("excerpt_selection_status", "selected"),
        ("role_specific_record", "direct_target_direction_primary"),
        ("association_predicate", "title_keyword_match"),
        ("association_scope", "target_general"),
        ("association_excerpt_relevance_score", 29),
        ("evidence_id_origin_policy", "source-external-v0"),
        ("parent_evidence_id", "ev_unrelated_parent"),
        ("parent_excerpt_hash", "not-a-sha256"),
        ("association_source_excerpt_hash", "0" * 64),
        ("parent_provenance_pointer_type", "claim_link"),
        ("same_document_external_id", "different-document"),
        ("duplicate_document_role", "direction_excerpt"),
        ("additional_network_call", True),
        ("direct_support_eligible", True),
        ("direct_support_rejection_codes", []),
        ("verified_direct_primary", True),
        ("verified_direct_primary_rejection_codes", []),
        ("full_text_span_excerpt_hash", "0" * 64),
        ("evidence_role_inherited", True),
        ("review_status", "peer_review_pending"),
    ],
)
def test_epmc_oa_fulltext_association_role_contract_fails_closed(
    field: str,
    value: Any,
) -> None:
    association = _oa_fulltext_association(
        hop=1,
        metadata_updates={field: value},
    )

    assert _primary_association_excerpt(_brief(), "FLT3", association) is None


def test_epmc_oa_fulltext_association_rejects_review_publication() -> None:
    association = _oa_fulltext_association(
        title="Review of FLT3-ITD prevalence in acute myeloid leukemia",
    )

    assert _primary_association_excerpt(_brief(), "FLT3", association) is None


def test_epmc_oa_fulltext_association_rejects_noncanonical_evidence_id() -> None:
    association = _oa_fulltext_association().model_copy(
        update={"evidence_id": "ev_unbound_association"},
        deep=True,
    )

    assert _primary_association_excerpt(_brief(), "FLT3", association) is None


def test_epmc_oa_fulltext_accepts_verified_primary_case_report_section() -> None:
    excerpt = (
        "FLT3 inhibition reduced phosphorylation and viability in acute "
        "myeloid leukemia cells."
    )
    record = _oa_fulltext_publication(
        excerpt=excerpt,
        publication_types=("Case Report",),
        metadata_updates={
            "source_location": (
                f"full_text:case:sec:2:p:1:sentence:1:chars:0-{len(excerpt)}"
            ),
            "section_kind": "case",
            "section_title": "Case presentation",
        },
    )

    assert _primary_direction_excerpt(_brief(), "FLT3", record) == excerpt


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_availability", "abstract_excerpt"),
        ("source_database", "PubMed"),
        ("discovery_origin", "reference_one_hop"),
        ("evidence_origin", "oa_results_fulltext_reference_one_hop"),
        ("content_retrieval_policy", "epmc_oa_fulltext_one_hop_v0"),
        ("excerpt_selection_policy", "europe_pmc_oa_jats_results_v0"),
        ("evidence_id_origin_policy", "source-external-v0"),
        ("source_location", "full_text:discussion:sec:1:p:1:sentence:1:chars:0-110"),
        ("source_location", "full_text:results:sec:0:p:1:sentence:1:chars:0-110"),
        ("full_text_stored", True),
        ("raw_xml_stored", True),
        ("raw_xml", "<article>forbidden retained body</article>"),
        ("raw_full_text_retained", True),
        ("content_storage_policy", "full_text_retained"),
        ("content_hash_algorithm", "md5"),
        ("content_hash_scope", "normalized_text"),
        ("full_text_content_hash", "0" * 64),
        ("eligible_content_hash", "not-a-sha256"),
        ("excerpt_hash", "0" * 64),
        ("full_text_span_excerpt_hash", "0" * 64),
        ("excerpt_char_count", 1),
        ("excerpt_max_chars", 321),
        ("original_sentence_char_count", 1),
        ("input_byte_count", 4 * 1024 * 1024 + 1),
        ("eligible_section_count", 0),
        ("eligible_paragraph_count", 0),
        ("emitted_span_count", 65),
        ("is_open_access", False),
        ("full_text_license", ""),
        ("full_text_license", "all-rights-reserved"),
        ("publication_status", "review_article"),
        ("lineage_verified", False),
        ("evidence_role_inherited", True),
        ("independent_primary_verification_required", False),
        ("independent_primary_verification", False),
        ("citation_hop", 2),
        ("citation_depth", 1),
        ("citation_lineage_sha256", "f" * 64),
        ("citation_child_canonical_id", "PMC:PMC12345678"),
        ("citation_child_identity_keys", ["MED:12345678"]),
        ("section_kind", "discussion"),
        ("section_title", 123),
        ("source_location_scope", "raw_jats_xml"),
        ("source_location_char_end_exclusive", False),
    ],
)
def test_epmc_oa_fulltext_metadata_contract_fails_closed(
    field: str,
    value: Any,
) -> None:
    record = _oa_fulltext_publication(metadata_updates={field: value})

    assert _primary_direction_excerpt(_brief(), "FLT3", record) is None


@pytest.mark.parametrize("bad_id_kind", ["arbitrary", "valid_other_hop"])
def test_epmc_oa_fulltext_direction_rejects_unbound_or_colliding_id(
    bad_id_kind: str,
) -> None:
    record = _oa_fulltext_publication(hop=0)
    bad_id = (
        "ev_unbound_fulltext"
        if bad_id_kind == "arbitrary"
        else _oa_fulltext_publication(hop=1).evidence_id
    )
    tampered = record.model_copy(
        update={"evidence_id": bad_id},
        deep=True,
    )

    assert _primary_direction_excerpt(_brief(), "FLT3", tampered) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("citation_child_pmcid", None),
        ("citation_child_pmid", None),
        (
            "citation_child_identity_keys",
            ["PMC:PMC12345678", "MED:12345678"],
        ),
        (
            "citation_child_identity_keys",
            ["MED:12345678", "MED:12345678", "PMC:PMC12345678"],
        ),
        ("citation_parent_external_id", "PMC11111111"),
        ("citation_child_canonical_id", "PMC:PMC12345678"),
    ],
)
def test_epmc_oa_reference_edge_subset_tamper_fails_closed(
    field: str,
    value: Any,
) -> None:
    record = _oa_fulltext_publication(
        hop=1,
        metadata_updates={field: value},
    )

    assert _primary_direction_excerpt(_brief(), "FLT3", record) is None


@pytest.mark.parametrize(
    ("publication_types", "title"),
    [
        (("Review",), "Review of FLT3 inhibition"),
        (("Journal Article", "Retracted Publication"), "FLT3 experiment"),
        (("Journal Article",), "Expression of concern: FLT3 experiment"),
        (("Editorial",), "FLT3 inhibition in AML"),
    ],
)
def test_epmc_oa_fulltext_rejects_review_nonprimary_and_retracted_records(
    publication_types: Sequence[str],
    title: str,
) -> None:
    record = _oa_fulltext_publication(
        publication_types=publication_types,
        title=title,
    )

    assert _primary_direction_excerpt(_brief(), "FLT3", record) is None


@pytest.mark.parametrize(
    "excerpt",
    [
        "FLT3 inhibition reduced phosphorylation and viability in lymphoma cells.",
        "ATR inhibition reduced FLT3 phosphorylation and viability in acute myeloid leukemia cells.",
        "FLT3 was measured in acute myeloid leukemia cells with reduced viability.",
        "FLT3 inhibition was performed in acute myeloid leukemia cells.",
        "FLT3 inhibition increased viability in acute myeloid leukemia cells.",
        "FLT3 inhibition reduced phosphorylation in acute myeloid leukemia cells. A second sentence reported viability.",
    ],
)
def test_epmc_oa_fulltext_child_is_independently_reverified_scientifically(
    excerpt: str,
) -> None:
    record = _oa_fulltext_publication(excerpt=excerpt, hop=1)

    assert _primary_direction_excerpt(_brief(), "FLT3", record) is None


def test_epmc_oa_reference_child_is_linked_without_inheriting_parent_role() -> None:
    record = _oa_fulltext_publication(hop=1)
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        discovery_source="external_dynamic",
        discovery_evidence=(_association(),),
    )

    hypothesis = Proposer(_StaticRetriever([record]))._build_hypothesis(
        _brief(),
        seed,
        1,
    )

    assert hypothesis.claims[0].evidence_ids == [record.evidence_id]
    linked = next(
        item for item in hypothesis.evidence if item.evidence_id == record.evidence_id
    )
    assert linked.metadata["citation_hop"] == 1
    assert linked.metadata["citation_relation"] == "reference"
    assert linked.metadata["evidence_role_inherited"] is False
    assert linked.metadata["independent_primary_verification"] is True
    assert linked.metadata["claim_linkage_roles"] == [
        "direct_target_direction_primary"
    ]
    assert Critic().review(_brief(), hypothesis).accepted is True


def test_epmc_oa_fulltext_branch_preserves_existing_abstract_contract() -> None:
    abstract = _publication(
        "ev_abstract_unchanged",
        "FLT3 inhibition reduced phosphorylation and viability in acute "
        "myeloid leukemia cells.",
    )

    assert _primary_direction_excerpt(_brief(), "FLT3", abstract) == abstract.metadata[
        "excerpt"
    ]


def test_mutated_target_and_oncometabolite_result_is_direct_primary() -> None:
    excerpt = (
        "LY3410738 covalently inhibited mutated-IDH1/2, reduced 2-HG levels "
        "at low nanomolar concentrations in human AML models, and induced "
        "myeloid differentiation."
    )
    record = _publication(
        "ev_idh1_mutant_oncometabolite",
        excerpt,
        source="Europe PMC",
    )
    brief = _brief().model_copy(
        update={"disease": "Acute myeloid leukemia"},
        deep=True,
    )

    assert _primary_direction_excerpt(brief, "IDH1", record) == excerpt
    assignment = _assign_claim_evidence(
        brief,
        target_symbol="IDH1",
        evidence=[record],
    )
    assert assignment.direction_primary_ids == (record.evidence_id,)


def test_mutated_non_target_intervention_is_not_reassigned() -> None:
    excerpt = (
        "LY3410738 inhibited mutated-IDH2, reduced 2-HG levels in human AML "
        "models, and induced differentiation while IDH1 was measured."
    )
    record = _publication(
        "ev_idh2_not_idh1",
        excerpt,
        source="Europe PMC",
    )
    brief = _brief().model_copy(
        update={"disease": "Acute myeloid leukemia"},
        deep=True,
    )

    assert _primary_direction_excerpt(brief, "IDH1", record) is None


def test_public_biology_v2_association_child_reaches_ma1_claim_assignment() -> None:
    client = _PrimaryAssociationPubMedClient()
    retriever = PublicBiologyRetriever(
        client=client,
        enabled_sources=["PubMed"],
        max_records_per_source=2,
        timeout_seconds=1,
        max_attempts=1,
        retry_backoff_seconds=0,
        sleeper=lambda _: None,
    )
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        discovery_source="external_dynamic",
    )

    hypothesis = Proposer(retriever)._build_hypothesis(_brief(), seed, 1)

    association = next(
        item
        for item in hypothesis.evidence
        if item.metadata.get("role_specific_record")
        == "primary_target_disease_association_excerpt"
    )
    assert association.external_id == "20515557"
    assert association.metadata["excerpt_selection_policy"] == (
        _ASSOCIATION_EXCERPT_POLICY
    )
    linked_claims = [
        claim
        for claim in hypothesis.claims
        if association.evidence_id in claim.evidence_ids
    ]
    assert len(linked_claims) == 1
    assert association.metadata["linked_claim_ids"] == [
        linked_claims[0].claim_id
    ]
    assert retriever.last_failures == {}
    assert [url for url, _ in client.calls] == [
        PUBMED_SEARCH_URL,
        PUBMED_SUMMARY_URL,
        PUBMED_FETCH_URL,
        PUBMED_SEARCH_URL,
    ]


def test_only_explicit_target_action_receives_direct_primary_role() -> None:
    hypothesis = _hypothesis()
    mechanism_claim = hypothesis.claims[0]
    association_claim = hypothesis.claims[1]

    assert mechanism_claim.evidence_ids == ["ev_direct"]
    assert association_claim.evidence_ids == ["ev_association"]
    all_linked_ids = [
        evidence_id
        for claim in hypothesis.claims
        for evidence_id in claim.evidence_ids
    ]
    assert len(all_linked_ids) == len(set(all_linked_ids)) == 2
    assert mechanism_claim.counter_evidence_ids == ["ev_resistance"]
    assert association_claim.counter_evidence_ids == []

    by_id = {record.evidence_id: record for record in hypothesis.evidence}
    assert by_id["ev_direct"].metadata["linked_claim_ids"] == [
        mechanism_claim.claim_id
    ]
    assert by_id["ev_association"].metadata["linked_claim_ids"] == [
        association_claim.claim_id
    ]
    for evidence_id in {
        "ev_direct_flc8",
        "ev_prevalence",
        "ev_review",
        "ev_potency_assertion",
        "ev_unverified_type",
        "ev_pp2a_modifier",
        "ev_nox2_modifier",
        "ev_atr_other_target",
        "ev_rsk1_other_target",
        "ev_ato_other_intervention",
        "ev_venetoclax_other_intervention",
        "ev_context",
        "ev_ot_association",
    }:
        assert by_id[evidence_id].metadata["linked_claim_ids"] == []
        assert by_id[evidence_id].metadata["claim_linkage_scope"] == (
            "unlinked_context"
        )
        assert by_id[evidence_id].metadata["direct_support_eligible"] is False
        assert by_id[evidence_id].metadata["verified_direct_primary"] is False
    assert by_id["ev_direct_flc8"].metadata[
        "direct_support_rejection_codes"
    ] == ["direct_target_intervention_missing"]
    assert by_id["ev_atr_other_target"].metadata[
        "direct_support_rejection_codes"
    ] == ["claim_reconciliation_not_direct_target_direction_primary"]
    assert by_id["ev_resistance"].metadata["linked_claim_ids"] == [
        mechanism_claim.claim_id
    ]
    assert by_id["ev_resistance"].metadata["claim_linkage_roles"] == [
        "claim_level_counter_context",
        "resistance_context_linked",
    ]
    assert by_id["ev_resistance"].supports == "context"
    assert by_id["ev_pp2a_modifier"].metadata["claim_linkage_roles"] == [
        "response_modifier_context_unlinked"
    ]
    assert by_id["ev_nox2_modifier"].metadata["claim_linkage_roles"] == [
        "response_modifier_context_unlinked"
    ]
    assert by_id["ev_ot_association"].metadata["claim_linkage_roles"] == [
        "ranked_association_discovery_context_unlinked"
    ]
    assert by_id["ev_ot_association"].supports == "context"
    assert all(
        record.metadata["claim_linkage_status"] == "reconciled"
        for record in hypothesis.evidence
    )

    assignment = hypothesis.target.metadata["claim_evidence_assignment"]
    assert assignment["resistance_context"]["evidence_ids"] == [
        "ev_resistance"
    ]
    assert assignment["resistance_context"][
        "linked_as_direction_refutation"
    ] is False
    assert assignment["resistance_context"][
        "linked_as_claim_limitation"
    ] is True
    assert all(
        unit["evidence_cardinality"] == 1
        for unit in assignment["claim_units"]
    )
    counter_search = hypothesis.target.metadata["counter_evidence_search"]
    assert counter_search["status"] == "not_recorded"
    assert counter_search["real_counter_count"] == 0
    assert counter_search["raw_search_status"] == "not_recorded"
    assert counter_search["retrieved_resistance_context_count"] == 1
    assert Critic().review(_brief(), hypothesis).accepted is True


def test_bounded_rejected_audit_keeps_every_claim_linked_evidence_unit() -> None:
    hypothesis = _hypothesis()
    records, total = _screening_external_records(hypothesis)
    visible_evidence_ids = {
        unit["evidence_id"]
        for record in records
        for unit in record["evidence_units"]
    }
    linked_evidence_ids = {
        evidence_id
        for claim in hypothesis.claims
        for evidence_id in (
            *claim.evidence_ids,
            *claim.counter_evidence_ids,
        )
    }

    assert total > 12
    assert linked_evidence_ids <= visible_evidence_ids


def _predeclared_model_direction_record(
    evidence_id: str,
    excerpt: str,
    *,
    model_aliases: Sequence[str],
    target_variants: Sequence[str],
) -> EvidenceRef:
    record = _publication(evidence_id, excerpt)
    return record.model_copy(
        update={
            "metadata": {
                **record.metadata,
                "query_strategy": (
                    "brief-derived-disease-or-molecular-model-direction-primary-v2"
                ),
                "primary_direction_supplement": True,
                "direction_disease_context_match_basis": (
                    "predeclared_molecular_model"
                ),
                "direction_disease_context_match_policy": (
                    "literal-disease-or-explicit-model-plus-target-variant-"
                    "same-sentence-v1"
                ),
                "predeclared_model_aliases": list(model_aliases),
                "predeclared_target_variants": list(target_variants),
            }
        },
        deep=True,
    )


@pytest.mark.parametrize(
    (
        "target",
        "disease",
        "target_cell_type",
        "variant",
        "excerpt",
    ),
    [
        (
            "JAK2",
            "essential thrombocythemia",
            "SET-2 (molecularly confirmed JAK2-V617F)",
            "JAK2 V617F",
            "In SET-2 cells heterozygous for JAK2 V617F, inhibition of JAK2 "
            "reduced phosphorylation and proliferation (IC50=33 nM).",
        ),
        (
            "FLT3",
            "acute myeloid leukemia",
            "MV4-11 (molecularly confirmed FLT3-ITD)",
            "FLT3-ITD",
            "In MV4-11 cells harboring FLT3-ITD, inhibition of FLT3 reduced "
            "phosphorylation and viability (IC50=12 nM).",
        ),
        (
            "FLT3",
            "acute myeloid leukemia",
            "MOLM-13 (molecularly confirmed FLT3 p.D835Y)",
            "FLT3 D835Y",
            "In MOLM-13 cells harboring FLT3 p.D835Y, inhibition of FLT3 "
            "reduced phosphorylation and viability (IC50=9 nM).",
        ),
        (
            "ALK",
            "inflammatory myofibroblastic tumor",
            "H3122 (ALK-rearranged)",
            "ALK rearrangement/fusion",
            "In ALK-rearranged H3122 cells, ALK inhibition reduced "
            "phosphorylation and viability (IC50=18 nM).",
        ),
    ],
)
def test_predeclared_molecular_model_direction_is_independently_recomputed(
    target: str,
    disease: str,
    target_cell_type: str,
    variant: str,
    excerpt: str,
) -> None:
    brief = ResearchBrief(
        research_goal=f"Evaluate direct {target} inhibition in the fixed model.",
        disease=disease,
        constraints=ProjectConstraints(
            target_cell_type=target_cell_type,
            intervention_direction="inhibit",
        ),
    )
    model_alias = target_cell_type.split(" ", 1)[0]
    record = _predeclared_model_direction_record(
        f"ev_{target.casefold()}_model",
        excerpt,
        model_aliases=[model_alias],
        target_variants=[variant],
    )

    assert _primary_direction_excerpt(brief, target, record) == excerpt


@pytest.mark.parametrize(
    "excerpt",
    [
        (
            "FLT3 inhibition reduced phosphorylation and viability in "
            "FLT3-ITD cells (IC50=12 nM)."
        ),
        (
            "FLT3 inhibition reduced phosphorylation and viability in "
            "MV4-11 cells (IC50=12 nM)."
        ),
        (
            "KIT inhibition reduced phosphorylation and viability in MV4-11 "
            "cells harboring KIT D816V (IC50=12 nM)."
        ),
        (
            "MV4-11 cells harbor FLT3-ITD. FLT3 inhibition reduced "
            "phosphorylation and viability (IC50=12 nM)."
        ),
    ],
)
def test_predeclared_model_direction_rejects_non_atomic_or_wrong_context(
    excerpt: str,
) -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct FLT3 inhibition in the fixed model.",
        disease="acute myeloid leukemia",
        constraints=ProjectConstraints(
            target_cell_type="MV4-11 (FLT3-ITD)",
            intervention_direction="inhibit",
        ),
    )
    record = _predeclared_model_direction_record(
        "ev_flt3_model_decoy",
        excerpt,
        model_aliases=["MV4-11"],
        target_variants=["FLT3-ITD"],
    )

    assert _primary_direction_excerpt(brief, "FLT3", record) is None


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("direction_disease_context_match_basis", "literal_disease"),
        ("direction_disease_context_match_policy", "untrusted-policy"),
        ("predeclared_model_aliases", ["UNDECLARED-9"]),
        ("predeclared_target_variants", ["FLT3 D835Y"]),
    ],
)
def test_predeclared_model_direction_rejects_forged_producer_attestation(
    metadata_key: str,
    metadata_value: Any,
) -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct FLT3 inhibition in the fixed model.",
        disease="acute myeloid leukemia",
        constraints=ProjectConstraints(
            target_cell_type="MV4-11 (FLT3-ITD)",
            intervention_direction="inhibit",
        ),
    )
    excerpt = (
        "In MV4-11 cells harboring FLT3-ITD, inhibition of FLT3 reduced "
        "phosphorylation and viability (IC50=12 nM)."
    )
    record = _predeclared_model_direction_record(
        "ev_flt3_model_forged",
        excerpt,
        model_aliases=["MV4-11"],
        target_variants=["FLT3-ITD"],
    ).model_copy(
        update={
            "metadata": {
                **_predeclared_model_direction_record(
                    "ev_flt3_model_forged",
                    excerpt,
                    model_aliases=["MV4-11"],
                    target_variants=["FLT3-ITD"],
                ).metadata,
                metadata_key: metadata_value,
            }
        },
        deep=True,
    )

    assert _primary_direction_excerpt(brief, "FLT3", record) is None


def test_predeclared_model_direction_is_atomically_linked_by_proposer() -> None:
    brief = ResearchBrief(
        research_goal="Evaluate direct JAK2 V617F inhibition in a fixed model.",
        disease="essential thrombocythemia",
        constraints=ProjectConstraints(
            target_cell_type="SET-2 (molecularly confirmed JAK2-V617F)",
            intervention_direction="inhibit",
        ),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=False,
        ),
    )
    excerpt = (
        "In SET-2 cells heterozygous for JAK2 V617F, inhibition of JAK2 "
        "reduced phosphorylation and proliferation (IC50=33 nM)."
    )
    record = _predeclared_model_direction_record(
        "ev_jak2_model_link",
        excerpt,
        model_aliases=["SET-2"],
        target_variants=["JAK2 V617F"],
    )

    hypothesis = Proposer(_StaticRetriever([record]))._build_hypothesis(
        brief,
        TargetSeed(
            symbol="JAK2",
            target_name="Janus kinase 2",
            track=HypothesisTrack.LITERATURE,
            priority=0.5,
        ),
        1,
    )

    assert hypothesis.claims[0].evidence_ids == [record.evidence_id]
    linked = next(
        item
        for item in hypothesis.evidence
        if item.evidence_id == record.evidence_id
    )
    assert linked.supports == "support"
    assert linked.metadata["claim_linkage_roles"] == [
        "direct_target_direction_primary"
    ]


@pytest.mark.parametrize(
    (
        "target",
        "disease",
        "target_cell_type",
        "pmid",
        "abstract_text",
        "expected_aliases",
        "expected_variants",
    ),
    [
        (
            "JAK2",
            "essential thrombocythemia",
            "SET-2 (molecularly confirmed JAK2-V617F)",
            "18775810",
            "The target variant is associated with essential thrombocythemia. "
            "In the SET-2 human cell line, heterozygous for the JAK2 V617F "
            "allele, inhibition of JAK2 resulted in decreased phosphorylation "
            "and inhibition of cell proliferation (GI50=0.033 microm).",
            ["SET-2"],
            ["JAK2 V617F"],
        ),
        (
            "FLT3",
            "acute myeloid leukemia",
            "MOLM-13 (molecularly confirmed FLT3 p.D835Y)",
            "37535845",
            "In MOLM-13 cells harboring FLT3 p.D835Y, inhibition of FLT3 "
            "resulted in decreased phosphorylation and inhibition of cell "
            "proliferation (GI50=0.004 microm).",
            ["MOLM-13"],
            ["FLT3 D835Y"],
        ),
    ],
)
def test_public_retriever_model_context_is_consumed_without_fixture_bridge(
    target: str,
    disease: str,
    target_cell_type: str,
    pmid: str,
    abstract_text: str,
    expected_aliases: list[str],
    expected_variants: list[str],
) -> None:
    brief = ResearchBrief(
        research_goal=f"Evaluate direct {target} inhibition in a fixed cell model.",
        disease=disease,
        constraints=ProjectConstraints(
            target_cell_type=target_cell_type,
            intervention_direction="inhibit",
        ),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=True,
        ),
    )
    client = _MolecularModelDirectionPubMedClient(abstract_text, pmid=pmid)
    retriever = PublicBiologyRetriever(
        client=client,
        enabled_sources=["PubMed"],
        fallback=None,
        max_attempts=1,
        retry_backoff_seconds=0,
        max_records_per_source=8,
    )

    hypothesis = Proposer(retriever)._build_hypothesis(
        brief,
        TargetSeed(
            symbol=target,
            target_name=f"{target} target",
            track=HypothesisTrack.LITERATURE,
            priority=0.5,
        ),
        1,
    )

    assert len(client.calls) == 6
    assert hypothesis.claims[0].evidence_ids
    linked = next(
        item
        for item in hypothesis.evidence
        if item.evidence_id == hypothesis.claims[0].evidence_ids[0]
    )
    assert linked.external_id == pmid
    assert linked.metadata["direction_disease_context_match_basis"] == (
        "predeclared_molecular_model"
    )
    assert linked.metadata["query_strategy"] == (
        "brief-derived-target-action-experimental-direction-primary-v3"
    )
    assert linked.metadata["predeclared_model_aliases"] == expected_aliases
    assert linked.metadata["predeclared_target_variants"] == expected_variants
    assert linked.metadata["claim_linkage_roles"] == [
        "direct_target_direction_primary"
    ]
    assert linked.metadata.get("fixture") is not True
    assert linked.metadata.get("mock") is False


def test_target_specific_inhibitor_positive_response_is_atomically_linked() -> None:
    brief = ResearchBrief(
        brief_id="brief_target_inhibitor_response",
        research_goal="Evaluate ALK inhibition in molecularly defined IMT.",
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=False,
        ),
    )
    excerpt = (
        "A patient with ALK-rearranged IMT showed a drastic response to two "
        "ALK tyrosine kinase inhibitors."
    )
    record = _publication("ev_alk_direction", excerpt).model_copy(
        update={
            "metadata": {
                **_publication("ev_alk_direction", excerpt).metadata,
                "query_strategy": (
                    "brief-derived-target-inhibitor-positive-response-primary-v1"
                ),
                "primary_direction_supplement": True,
            }
        },
        deep=True,
    )
    lung_decoy = _publication(
        "ev_alk_lung_decoy",
        "ALK inhibition reduced phosphorylation and viability in "
        "ALK-rearranged lung cancer cells.",
    ).model_copy(
        update={
            "metadata": {
                **_publication(
                    "ev_alk_lung_decoy",
                    "ALK inhibition reduced phosphorylation and viability in "
                    "ALK-rearranged lung cancer cells.",
                ).metadata,
                "query_strategy": (
                    "brief-derived-target-inhibitor-positive-response-primary-v1"
                ),
                "primary_direction_supplement": True,
            }
        },
        deep=True,
    )
    harmful_decoy = _publication(
        "ev_alk_harmful_decoy",
        "ALK inhibition increased viability and tumor growth in "
        "ALK-rearranged IMT cells.",
    ).model_copy(
        update={
            "metadata": {
                **_publication(
                    "ev_alk_harmful_decoy",
                    "ALK inhibition increased viability and tumor growth in "
                    "ALK-rearranged IMT cells.",
                ).metadata,
                "query_strategy": (
                    "brief-derived-target-inhibitor-positive-response-primary-v1"
                ),
                "primary_direction_supplement": True,
            }
        },
        deep=True,
    )
    methods_only = _publication(
        "ev_alk_methods_only",
        "Viability was measured after ALK inhibition in ALK-rearranged IMT "
        "cells.",
    ).model_copy(
        update={
            "metadata": {
                **_publication(
                    "ev_alk_methods_only",
                    "Viability was measured after ALK inhibition in "
                    "ALK-rearranged IMT cells.",
                ).metadata,
                "query_strategy": (
                    "brief-derived-target-inhibitor-positive-response-primary-v1"
                ),
                "primary_direction_supplement": True,
            }
        },
        deep=True,
    )
    negative_qualifier = _publication(
        "ev_alk_negative_qualifier",
        "ALK rearrangement-negative IMT showed a response to ALK inhibitors.",
    ).model_copy(
        update={
            "metadata": {
                **_publication(
                    "ev_alk_negative_qualifier",
                    "ALK rearrangement-negative IMT showed a response to "
                    "ALK inhibitors.",
                ).metadata,
                "query_strategy": (
                    "brief-derived-target-inhibitor-positive-response-primary-v1"
                ),
                "primary_direction_supplement": True,
            }
        },
        deep=True,
    )
    hypothesis = Proposer(
        _StaticRetriever(
            [
                record,
                lung_decoy,
                harmful_decoy,
                methods_only,
                negative_qualifier,
            ]
        )
    )._build_hypothesis(
        brief,
        TargetSeed(
            symbol="ALK",
            target_name="Anaplastic lymphoma kinase",
            track=HypothesisTrack.LITERATURE,
            priority=0.5,
        ),
        1,
    )

    assert len(hypothesis.claims) == 1
    assert hypothesis.claims[0].evidence_ids == ["ev_alk_direction"]
    linked = next(item for item in hypothesis.evidence if item.evidence_id == record.evidence_id)
    assert linked.supports == "support"
    assert linked.metadata["claim_linkage_roles"] == [
        "direct_target_direction_primary"
    ]
    by_id = {item.evidence_id: item for item in hypothesis.evidence}
    for decoy in (lung_decoy, harmful_decoy, methods_only, negative_qualifier):
        assert by_id[decoy.evidence_id].supports == "context"
        assert by_id[decoy.evidence_id].metadata["linked_claim_ids"] == []
    assert Critic().review(brief, hypothesis).accepted is True


def test_activation_direction_requires_favorable_endpoint_polarity() -> None:
    brief = ResearchBrief(
        brief_id="brief_egfr_activation_response",
        research_goal="Evaluate EGFR activation in molecularly defined NSCLC.",
        disease="EGFR-mutant non-small cell lung cancer",
        constraints=ProjectConstraints(intervention_direction="activate"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=False,
        ),
    )

    def activation_record(evidence_id: str, excerpt: str) -> EvidenceRef:
        record = _publication(evidence_id, excerpt)
        return record.model_copy(
            update={
                "metadata": {
                    **record.metadata,
                    "query_intervention_direction": "activate",
                }
            },
            deep=True,
        )

    favorable = activation_record(
        "ev_egfr_activation_favorable",
        "EGFR activation increased EGFR phosphorylation and induced apoptosis "
        "in EGFR-mutant NSCLC cells.",
    )
    opposite = activation_record(
        "ev_egfr_activation_opposite",
        "EGFR activation decreased EGFR phosphorylation in EGFR-mutant NSCLC "
        "cells.",
    )
    hypothesis = Proposer(
        _StaticRetriever([favorable, opposite])
    )._build_hypothesis(
        brief,
        TargetSeed(
            symbol="EGFR",
            target_name="Epidermal growth factor receptor",
            track=HypothesisTrack.LITERATURE,
            priority=0.5,
        ),
        1,
    )

    assert len(hypothesis.claims) == 1
    assert hypothesis.claims[0].evidence_ids == [favorable.evidence_id]
    by_id = {item.evidence_id: item for item in hypothesis.evidence}
    assert by_id[favorable.evidence_id].supports == "support"
    assert by_id[opposite.evidence_id].supports == "context"
    assert by_id[opposite.evidence_id].metadata["linked_claim_ids"] == []


def test_counter_summary_preserves_actual_pubmed_query_provenance() -> None:
    brief = ResearchBrief(
        brief_id="brief_pubmed_counter_summary",
        research_goal="Evaluate direct FLT3 inhibition evidence.",
        disease="Acute myeloid leukemia",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=False,
        ),
    )
    record = _publication(
        "ev_pubmed_counter_attested",
        "FLT3 inhibition reduced phosphorylation and viability in acute "
        "myeloid leukemia cells.",
    ).model_copy(
        update={
            "metadata": {
                **_publication(
                    "ev_pubmed_counter_attested",
                    "FLT3 inhibition reduced phosphorylation and viability "
                    "in acute myeloid leukemia cells.",
                ).metadata,
                "counter_search_status": "completed_empty",
                "counter_search_source": "PubMed",
                "counter_search_query": "distinct bounded counter query",
                "counter_search_query_sha256": "a" * 64,
                "counter_search_separate_query": True,
                "counter_search_raw_hit_count": 0,
                "counter_search_http_call_provenance": {
                    "endpoint": "https://example.test/esearch.fcgi",
                    "parameter_sha256": "b" * 64,
                    "response_sha256": "c" * 64,
                    "status_code": 200,
                },
            }
        },
        deep=True,
    )
    hypothesis = Proposer(_StaticRetriever([record]))._build_hypothesis(
        brief,
        TargetSeed(
            symbol="FLT3",
            target_name="FLT3",
            track=HypothesisTrack.LITERATURE,
            priority=0.5,
        ),
        1,
    )

    summary = hypothesis.target.metadata["counter_evidence_search"]
    assert summary["source"] == "PubMed"
    assert summary["query"] == "distinct bounded counter query"
    assert summary["query_sha256"] == "a" * 64
    assert summary["separate_query"] is True
    assert summary["raw_search_status"] == "completed_empty"
    assert summary["http_call_provenance"]["parameter_sha256"] == "b" * 64


def test_counter_summary_does_not_call_retrieved_context_an_empty_search() -> None:
    brief = ResearchBrief(
        brief_id="brief_multi_source_counter_summary",
        research_goal="Evaluate direct FLT3 inhibition evidence.",
        disease="Acute myeloid leukemia",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed", "Europe PMC"],
            require_primary_sources=False,
        ),
    )
    favorable = _publication(
        "ev_multi_source_direction",
        "FLT3 inhibition reduced phosphorylation and viability in acute "
        "myeloid leukemia cells.",
    ).model_copy(
        update={
            "metadata": {
                **_publication(
                    "ev_multi_source_direction",
                    "FLT3 inhibition reduced phosphorylation and viability in "
                    "acute myeloid leukemia cells.",
                ).metadata,
                "counter_search_status": "completed_with_results",
                "counter_search_source": "PubMed",
                "counter_search_query": "pubmed bounded resistance query",
                "counter_search_query_sha256": "a" * 64,
                "counter_search_separate_query": True,
                "counter_search_raw_hit_count": 20,
            }
        },
        deep=True,
    )
    resistance = _publication(
        "ev_multi_source_resistance",
        "FLT3 inhibitor resistance emerged in acute myeloid leukemia cells.",
        source="Europe PMC",
        supports="counter",
        counter_category="resistance",
    ).model_copy(
        update={
            "metadata": {
                **_publication(
                    "ev_multi_source_resistance",
                    "FLT3 inhibitor resistance emerged in acute myeloid "
                    "leukemia cells.",
                    source="Europe PMC",
                    supports="counter",
                    counter_category="resistance",
                ).metadata,
                "counter_search_status": "completed_with_results",
                "counter_search_source": "Europe PMC",
                "counter_search_query": "europe pmc bounded resistance query",
                "counter_search_query_sha256": "b" * 64,
                "counter_search_separate_query": True,
                "counter_search_raw_hit_count": 1,
                "counter_signal": "resistance",
            }
        },
        deep=True,
    )
    hypothesis = Proposer(
        _StaticRetriever([favorable, resistance])
    )._build_hypothesis(
        brief,
        TargetSeed(
            symbol="FLT3",
            target_name="FLT3",
            track=HypothesisTrack.LITERATURE,
            priority=0.5,
        ),
        1,
    )

    summary = hypothesis.target.metadata["counter_evidence_search"]
    assert summary["status"] == "completed_with_results"
    assert summary["raw_search_status"] == "completed_with_results"
    assert summary["claim_linked_refutation_status"] == "completed_empty"
    assert summary["real_counter_count"] == 0
    assert summary["retrieved_resistance_context_count"] == 1
    assert summary["source"] == "Europe PMC; PubMed"
    assert summary["sources"] == ["Europe PMC", "PubMed"]
    assert summary["query"] is None
    assert summary["queries"] == [
        "europe pmc bounded resistance query",
        "pubmed bounded resistance query",
    ]
    assert summary["separate_query"] is True


def test_rearrangement_association_scope_is_independently_enforced() -> None:
    brief = ResearchBrief(
        brief_id="brief_rearrangement_association",
        research_goal=(
            "Evaluate ALK gene rearrangement in inflammatory "
            "myofibroblastic tumor with primary cohort association evidence."
        ),
        disease="ALK-rearranged inflammatory myofibroblastic tumor",
        constraints=ProjectConstraints(intervention_direction="inhibit"),
        data_policy=DataSourcePolicy(
            allowed_sources=["PubMed"],
            require_primary_sources=True,
        ),
    )
    excerpt = (
        "Among 50 patients with inflammatory myofibroblastic tumors, ALK "
        "rearrangements were identified in 25 cases (50%)."
    )
    association = EvidenceRef(
        evidence_id="ev_alk_association",
        source="PubMed",
        external_id="association-document",
        title="Primary molecular cohort",
        supports="support",
        excerpt_hash="association-excerpt-hash",
        metadata={
            "excerpt": excerpt,
            "content_availability": "abstract_excerpt",
            "source_location": "abstract:RESULTS:chars:0-150",
            "content_hash": "association-content-hash",
            "full_text_stored": False,
            "excerpt_selection_policy": _ASSOCIATION_EXCERPT_POLICY,
            "role_specific_record": "primary_target_disease_association_excerpt",
            "association_predicate": "mutation_prevalence",
            "association_scope": "rearranged_or_fusion_target_subtype",
            "association_excerpt_relevance_score": 28,
            "parent_evidence_id": "ev_alk_parent",
            "same_document_external_id": "association-document",
            "duplicate_document_role": "association_excerpt",
            "provenance_class": "external_live",
            "publication_types": ["Journal Article"],
            "claim_linkage_status": "pending_ma1_claim_reconciliation",
            "mock": False,
        },
    )
    general_scope = association.model_copy(
        update={
            "evidence_id": "ev_alk_association_general",
            "external_id": "association-general-document",
            "metadata": {
                **association.metadata,
                "association_scope": "target_general",
                "parent_evidence_id": "ev_alk_general_parent",
                "same_document_external_id": "association-general-document",
            },
        },
        deep=True,
    )
    split_sentence = association.model_copy(
        update={
            "evidence_id": "ev_alk_association_split",
            "external_id": "association-split-document",
            "metadata": {
                **association.metadata,
                "excerpt": (
                    "We reviewed 50 inflammatory myofibroblastic tumors. "
                    "ALK rearrangements were identified in 25 cases (50%)."
                ),
                "parent_evidence_id": "ev_alk_split_parent",
                "same_document_external_id": "association-split-document",
            },
        },
        deep=True,
    )
    negative_qualifier = association.model_copy(
        update={
            "evidence_id": "ev_alk_association_negative_qualifier",
            "external_id": "association-negative-qualifier-document",
            "metadata": {
                **association.metadata,
                "excerpt": (
                    "Among 50 inflammatory myofibroblastic tumors, ALK "
                    "rearrangement-negative cases were identified in 25 "
                    "cases (50%)."
                ),
                "parent_evidence_id": "ev_alk_negative_qualifier_parent",
                "same_document_external_id": (
                    "association-negative-qualifier-document"
                ),
            },
        },
        deep=True,
    )
    hypothesis = Proposer(
        _StaticRetriever(
            [association, general_scope, split_sentence, negative_qualifier]
        )
    )._build_hypothesis(
        brief,
        TargetSeed(
            symbol="ALK",
            target_name="Anaplastic lymphoma kinase",
            track=HypothesisTrack.LITERATURE,
            priority=0.5,
        ),
        1,
    )

    linked_claims = [
        claim
        for claim in hypothesis.claims
        if association.evidence_id in claim.evidence_ids
    ]
    assert len(linked_claims) == 1
    assert hypothesis.target.metadata["claim_evidence_assignment"][
        "association_claims"
    ]
    assert next(
        item for item in hypothesis.evidence if item.evidence_id == association.evidence_id
    ).metadata["claim_linkage_roles"] == ["primary_target_disease_association"]
    assert all(
        not item.metadata["linked_claim_ids"]
        for item in hypothesis.evidence
        if item.evidence_id
        in {
            general_scope.evidence_id,
            split_sentence.evidence_id,
            negative_qualifier.evidence_id,
        }
    )


def test_claim_reconciliation_preserves_immutable_evidence_ids_and_hashes() -> None:
    hypothesis = _hypothesis()
    before = {
        record.evidence_id: (
            record.excerpt_hash,
            record.metadata.get("content_hash"),
        )
        for record in (*(_association(), _ranked_association()), *_records())
    }
    after = {
        record.evidence_id: (
            record.excerpt_hash,
            record.metadata.get("content_hash"),
        )
        for record in hypothesis.evidence
    }

    assert after == before


def test_post_context_enrichment_preserves_reconciled_claim_specific_links() -> None:
    hypothesis = _hypothesis()
    packet = HypothesisPacket(
        packet_id="hp_atomic_claim_test",
        brief_id=_brief().brief_id,
        hypotheses=[hypothesis],
    )
    target = ContextTarget(
        symbol="FLT3",
        track=HypothesisTrack.LITERATURE,
        hypothesis_id=hypothesis.hypothesis_id,
        rank=1,
    )
    context = EvidenceRef(
        evidence_id="ctx_biorxiv",
        source="bioRxiv",
        external_id="2026.01.01.123456",
        title="Context-only FLT3 preprint",
        supports="context",
        metadata={
            "context_only": True,
            "causal_inference_allowed": False,
            "counter_evidence_allowed": False,
            "provenance_class": "external_live",
            "retrieval_adapter": "test",
            "retrieval_endpoint": "https://example.test/context",
        },
    )
    routing = ContextRoutingResult(
        disease=_brief().disease,
        budget=ContextRoutingBudget(max_source_calls=1),
        target_selection=[
            ContextTargetSelection(
                target=target,
                status="selected",
                detail="focused claim-linkage regression",
            )
        ],
        targets=[
            TargetContextBundle(
                target=target,
                sources=[
                    ContextSourceOutcome(
                        source="bioRxiv",
                        family="preprint",
                        status="ok",
                        queried=True,
                        record_count=1,
                    )
                ],
                evidence=[context],
            )
        ],
        source_calls_used=1,
        planned_source_calls=1,
        elapsed_seconds=0.1,
        budget_exhausted=False,
    )

    enriched = enrich_hypothesis_packet_with_context(packet, routing)
    final_hypothesis = enriched.hypotheses[0]
    by_id = {record.evidence_id: record for record in final_hypothesis.evidence}

    assert final_hypothesis.claims == hypothesis.claims
    assert by_id["ev_direct"].metadata["claim_linkage_status"] == "reconciled"
    assert by_id["ev_direct"].metadata["linked_claim_ids"] == [
        hypothesis.claims[0].claim_id
    ]
    assert by_id["ev_association"].metadata["linked_claim_ids"] == [
        hypothesis.claims[1].claim_id
    ]
    assert by_id["ctx_biorxiv"].supports == "context"
    assert by_id["ctx_biorxiv"].metadata["context_only"] is True
    assert "ctx_biorxiv" not in {
        evidence_id
        for claim in final_hypothesis.claims
        for evidence_id in (*claim.evidence_ids, *claim.counter_evidence_ids)
    }


def test_no_primary_result_fails_closed_without_promoting_association() -> None:
    records = [
        _publication(
            "ev_prevalence_only",
            "FLT3 mutations are frequent and predict poor survival in acute "
            "myeloid leukemia.",
        )
    ]
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        discovery_source="external_dynamic",
        discovery_evidence=(_association(),),
    )
    hypothesis = Proposer(_StaticRetriever(records))._build_hypothesis(
        _brief(),
        seed,
        1,
    )

    mechanism_claim, association_claim = hypothesis.claims
    assert mechanism_claim.evidence_ids == []
    assert mechanism_claim.confidence == 0.25
    assert association_claim.evidence_ids == ["ev_association"]
    critique = Critic().review(_brief(), hypothesis)
    assert critique.accepted is False
    assert "UNSUPPORTED_CLAIM" in critique.reason_codes
    assert "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY" in critique.reason_codes
    assert (
        "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION"
        not in critique.reason_codes
    )


@pytest.mark.parametrize(
    ("publication_types", "title"),
    [
        ([], "Primary-looking FLT3 cohort"),
        (["Review"], "Review article about FLT3 cohorts"),
        (["Journal Article"], "Systematic review of FLT3-ITD prevalence"),
    ],
)
def test_association_claim_rejects_unverified_or_review_publications(
    publication_types: list[str],
    title: str,
) -> None:
    association = _association().model_copy(
        update={
            "title": title,
            "metadata": {
                **_association().metadata,
                "publication_types": publication_types,
            },
        },
        deep=True,
    )
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        discovery_source="external_dynamic",
        discovery_evidence=(association, _ranked_association()),
    )
    hypothesis = Proposer(
        _StaticRetriever([_records()[0]])
    )._build_hypothesis(_brief(), seed, 1)

    association_claim = hypothesis.claims[-1]
    assert association_claim.evidence_ids == []
    assert "No verified primary-publication" in association_claim.text
    by_id = {item.evidence_id: item for item in hypothesis.evidence}
    assert by_id["ev_association"].supports == "context"
    assert by_id["ev_association"].metadata["linked_claim_ids"] == []
    critique = Critic().review(_brief(), hypothesis)
    assert critique.accepted is False
    assert "UNSUPPORTED_CLAIM" in critique.reason_codes
    assert "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION" in critique.reason_codes


def test_invalid_canonical_source_role_is_quarantined_not_overwritten() -> None:
    invalid_direct = _records()[0].model_copy(
        update={
            "metadata": {
                **_records()[0].metadata,
                "source_supports": "invented_role",
            }
        },
        deep=True,
    )
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        discovery_source="external_dynamic",
        discovery_evidence=(_association(),),
    )

    hypothesis = Proposer(
        _StaticRetriever([invalid_direct])
    )._build_hypothesis(_brief(), seed, 1)

    direct_claim = hypothesis.claims[0]
    assert direct_claim.evidence_ids == []
    record = next(
        item for item in hypothesis.evidence if item.evidence_id == "ev_direct"
    )
    assert record.supports == "context"
    assert record.metadata["source_supports"] == "invented_role"
    assert record.metadata["claim_linkage_roles"] == [
        "invalid_source_supports_context_unlinked"
    ]
    assert Critic().review(_brief(), hypothesis).accepted is False


def test_legacy_omnibus_prior_is_reclassified_idempotently_on_rework() -> None:
    hypothesis = _hypothesis()
    all_support = [
        record.evidence_id
        for record in hypothesis.evidence
        if record.supports == "support"
    ]
    all_counter = [
        record.evidence_id
        for record in hypothesis.evidence
        if record.supports == "counter"
    ]
    legacy_evidence = [
        record.model_copy(
            update={
                "metadata": {
                    key: value
                    for key, value in record.metadata.items()
                    if not key.startswith("claim_linkage_")
                    and key != "linked_claim_ids"
                }
                | {"claim_linkage_status": "pending_ma1_claim_reconciliation"}
            },
            deep=True,
        )
        for record in hypothesis.evidence
    ]
    legacy = hypothesis.model_copy(
        update={
            "claims": [
                claim.model_copy(
                    update={
                        "text": "Omnibus legacy assertion",
                        "evidence_ids": all_support,
                        "counter_evidence_ids": all_counter,
                    },
                    deep=True,
                )
                for claim in hypothesis.claims
            ],
            "evidence": legacy_evidence,
        },
        deep=True,
    )
    assert legacy.claims[0].evidence_ids == legacy.claims[1].evidence_ids

    first = _reclassify_prior_claim_evidence(_brief(), legacy)
    second = _reclassify_prior_claim_evidence(_brief(), first)

    assert first.claims == second.claims
    assert first.evidence == second.evidence
    assert first.claims[0].evidence_ids == ["ev_direct"]
    assert first.claims[1].evidence_ids == ["ev_association"]
    assert first.claims[0].counter_evidence_ids == ["ev_resistance"]
    assert first.claims[1].counter_evidence_ids == []
    assert all(
        record.metadata["claim_linkage_status"] == "reconciled"
        for record in first.evidence
    )
    by_id = {record.evidence_id: record for record in first.evidence}
    assert by_id["ev_resistance"].metadata["source_supports"] == "counter"
    assert by_id["ev_resistance"].supports == "context"
    assert by_id["ev_resistance"].metadata["claim_linkage_roles"] == [
        "claim_level_counter_context",
        "resistance_context_linked",
    ]
    assert first.target.metadata["counter_evidence_search"] == (
        second.target.metadata["counter_evidence_search"]
    )
    assert Critic().review(_brief(), first).accepted is True


def test_unsupported_legacy_prior_is_retrieved_again_instead_of_burning_rework() -> None:
    brief = _brief()
    seed = TargetSeed(
        symbol="FLT3",
        target_name="Fms related receptor tyrosine kinase 3",
        track=HypothesisTrack.LITERATURE,
        priority=0.95,
        ensembl_id="ENSG00000122025",
        discovery_source="external_dynamic",
        discovery_evidence=(_association(),),
    )
    prevalence_only = Proposer(
        _StaticRetriever(
            [
                _publication(
                    "ev_prevalence_only",
                    "FLT3 mutations are frequent and predict poor survival in "
                    "acute myeloid leukemia.",
                )
            ]
        )
    )._build_hypothesis(brief, seed, 1)
    all_support = [
        record.evidence_id
        for record in prevalence_only.evidence
        if record.supports == "support"
    ]
    legacy = prevalence_only.model_copy(
        update={
            "claims": [
                claim.model_copy(
                    update={
                        "text": "Legacy omnibus claim",
                        "evidence_ids": all_support,
                    },
                    deep=True,
                )
                for claim in prevalence_only.claims
            ]
        },
        deep=True,
    )
    settings = Settings(
        _env_file=None,
        dynamic_target_discovery_enabled=False,
        target_identity_resolution_enabled=False,
        max_ma1_iterations=1,
    )
    agent = MA1Agent(
        settings=settings,
        evidence_retriever=_StaticRetriever(_records()),
    )
    agent._prepare_target_catalog = lambda brief, per_track_limit: TargetCatalogResult(  # type: ignore[method-assign]
        catalog={
            HypothesisTrack.LITERATURE: (seed,),
            HypothesisTrack.TBIO: (),
        },
        disease_resolution_status="resolved",
        external_seed_count=1,
    )

    packet = agent.run(brief, revision_context={"hypotheses": [legacy]})

    assert packet.metadata["prior_claim_evidence_reclassified_count"] == 1
    assert packet.metadata["prior_claim_evidence_retry_symbols"] == ["flt3"]
    assert [item.target.symbol for item in packet.hypotheses] == ["FLT3"]
    assert not [item for item in packet.excluded if item.target_symbol == "FLT3"]
    assert packet.hypotheses[0].claims[0].evidence_ids == ["ev_direct"]
    assert packet.hypotheses[0].claims[1].evidence_ids == ["ev_association"]


def test_tbio_supply_track_does_not_override_authoritative_tclin_tdl() -> None:
    hypothesis = _hypothesis()
    target = hypothesis.target.model_copy(
        update={
            "tdl": "Tclin",
            "metadata": {
                **hypothesis.target.metadata,
                "effective_tdl": "Tclin",
                "effective_work_track": "tbio",
            },
        },
        deep=True,
    )
    tbio_supplied = hypothesis.model_copy(
        update={"track": HypothesisTrack.TBIO, "target": target},
        deep=True,
    )

    critique = Critic().review(_brief(), tbio_supplied)

    assert critique.accepted is True
    assert "TBIO_CLASSIFICATION_MISSING" not in critique.reason_codes
