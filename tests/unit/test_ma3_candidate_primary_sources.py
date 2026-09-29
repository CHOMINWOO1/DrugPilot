from __future__ import annotations

from hashlib import sha256
import json

import pytest

from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateCard,
    CandidatePacket,
    CandidateRoute,
    DataSourcePolicy,
    EvidenceRef,
    ResearchBrief,
    ScientificBackendKind,
    TargetCandidate,
)


def _brief(*, require_primary_sources: bool) -> ResearchBrief:
    return ResearchBrief(
        research_goal="Evaluate an exact target-linked candidate with primary evidence.",
        disease="Example disease",
        data_policy=DataSourcePolicy(
            require_primary_sources=require_primary_sources
        ),
    )


def _live_packet() -> CandidatePacket:
    excerpt = "Drug one inhibited GENE with an IC50 of 4.2 nM."
    excerpt_hash = sha256(excerpt.encode("utf-8")).hexdigest()
    aliases = ["Drug one", "Research code-1"]
    alias_records = [
        {"molecule_chembl_id": "CHEMBL1", "aliases": aliases}
    ]
    alias_inventory_hash = sha256(
        json.dumps(
            alias_records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    pubmed_params = {
        "db": "pubmed",
        "id": "12345678",
        "retmode": "json",
        "version": "2.0",
    }
    pubmed_parameter_hash = sha256(
        json.dumps(
            pubmed_params,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    document_params = {
        "document_chembl_id__in": "CHEMBL900001",
        "limit": 1,
    }
    document_parameter_hash = sha256(
        json.dumps(
            document_params,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    document_batch = {
        "schema_version": "chembl_document_query_batch_v1",
        "document_chembl_ids": ["CHEMBL900001"],
        "document_count": 1,
        "batch_cap": 80,
        "selected_document_chembl_id": "CHEMBL900001",
        "selected_document_in_batch": True,
        "query_parameters": document_params,
        "query_parameter_hash": document_parameter_hash,
        "call_parameter_hash": document_parameter_hash,
        "truncated": False,
        "fixture": False,
    }
    document_batch["attestation_sha256"] = sha256(
        json.dumps(
            document_batch,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    candidate_start = excerpt.index("Drug one")
    target_start = excerpt.index("GENE")
    endpoint_start = excerpt.index("IC50")
    value_start = excerpt.index("4.2 nM")
    evidence_id = "ev_chembl_activity_primary"
    activity = ActivityMeasurement(
        endpoint="IC50",
        relation="=",
        value=4.2,
        unit="nM",
        assay_type="single protein format",
        source_evidence_id=evidence_id,
    )
    evidence = EvidenceRef(
        evidence_id=evidence_id,
        source="ChEMBL",
        external_id="11",
        title="Measured IC50 for CHEMBL1 against CHEMBL_TARGET",
        supports="support",
        excerpt_hash=excerpt_hash,
        metadata={
            "fixture": False,
            "activity_id": "11",
            "molecule_chembl_id": "CHEMBL1",
            "target_accession": "P12345",
            "target_uniprot_id": "P12345",
            "target_gene": "GENE",
            "target_chembl_id": "CHEMBL_TARGET",
            "document_chembl_id": "CHEMBL900001",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": "4.2",
            "standard_units": "nM",
            "primary_source": {
                "schema_version": "candidate_primary_evidence_v2",
                "verification_status": "verified",
                "retrieval_status": "completed",
                "is_primary_support": True,
                "reason_code": None,
                "document_chembl_id": "CHEMBL900001",
                "doc_type": "PUBLICATION",
                "pmid": "12345678",
                "doi": "10.1000/example.primary",
                "title": "Primary activity study",
                "publication_types": ["Journal Article"],
                "chembl_reported_publication_types": [],
                "pubmed_type_verification": {
                    "schema_version": "pubmed_esummary_publication_type_v1",
                    "verification_status": "verified",
                    "retrieval_status": "completed",
                    "reason_code": None,
                    "pmid": "12345678",
                    "publication_types": ["Journal Article"],
                    "required_publication_type": "Journal Article",
                    "excluded_publication_types": [],
                    "type_policy": (
                        "journal_article_without_excluded_publication_type_v1"
                    ),
                    "batch": {
                        "requested_pmids": ["12345678"],
                        "requested_count": 1,
                        "eligible_pmid_count": 1,
                        "batch_cap": 300,
                        "truncated": False,
                        "single_batch": True,
                        "batch_index": 1,
                        "batch_count": 1,
                        "global_eligible_pmid_count": 1,
                        "observed_result_uids": ["12345678"],
                    },
                    "query": {
                        "source": "PubMed",
                        "endpoint": (
                            "https://eutils.ncbi.nlm.nih.gov/entrez/"
                            "eutils/esummary.fcgi"
                        ),
                        "parameter_keys": ["db", "id", "retmode", "version"],
                        "parameter_hash": pubmed_parameter_hash,
                        "status_code": 200,
                        "response_sha256": "c" * 64,
                    },
                    "fixture": False,
                },
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "retrieved_field": "abstract",
                "excerpt": excerpt,
                "excerpt_location": {
                    "field": "abstract",
                    "sentence_index": 1,
                    "clause_index": 0,
                    "normalized_char_start": 25,
                    "normalized_char_end": 25 + len(excerpt),
                },
                "excerpt_hash": excerpt_hash,
                "content_hash": "b" * 64,
                "content_length": 200,
                "full_text_stored": False,
                "hash_algorithm": "sha256",
                "activity_binding": {
                    "activity_id": "11",
                    "molecule_chembl_id": "CHEMBL1",
                    "target_chembl_id": "CHEMBL_TARGET",
                    "assay_chembl_id": "CHEMBL_ASSAY_1",
                    "document_chembl_id": "CHEMBL900001",
                    "endpoint": "IC50",
                    "relation": "=",
                    "value": 4.2,
                    "unit": "nM",
                    "deposited_activity": {
                        "schema_version": "chembl_deposited_activity_exact_v1",
                        "standard_type": "IC50",
                        "standard_relation": "=",
                        "standard_value": "4.2",
                        "standard_units": "nM",
                        "typed_exact_match": True,
                        "missing_values_imputed": False,
                        "normalization_applied": False,
                        "unit_conversion_applied": False,
                    },
                },
                "candidate_exact_aliases": aliases,
                "candidate_alias_source": (
                    "chembl_pref_name_and_molecule_synonyms_exact_v1"
                ),
                "candidate_alias_conflicts": [],
                "competitor_alias_provenance": {
                    "policy": (
                        "all_bounded_pool_candidate_aliases_except_current_v1"
                    ),
                    "current_molecule_chembl_id": "CHEMBL1",
                    "competitor_exact_aliases": [],
                    "alias_owners": [],
                    "inventory": {
                        "schema_version": "candidate_pool_alias_inventory_v1",
                        "pool_candidate_cap": 100,
                        "per_candidate_alias_cap": 32,
                        "competitor_alias_cap": 3200,
                        "candidate_count": 1,
                        "retained_candidate_count": 1,
                        "alias_count": 2,
                        "source_candidate_ids": ["CHEMBL1"],
                        "missing_molecule_ids": [],
                        "aliasless_molecule_ids": [],
                        "complete": True,
                        "truncated": False,
                        "records": alias_records,
                        "inventory_sha256": alias_inventory_hash,
                        "fixture": False,
                    },
                    "inventory_sha256": alias_inventory_hash,
                    "truncated": False,
                    "fixture": False,
                },
                "relevance": {
                    "matched_groups": ["candidate", "target", "activity"],
                    "matched_terms": {
                        "candidate": ["drug one"],
                        "target": ["gene"],
                        "activity": ["ic50"],
                    },
                    "matched_exact_value_unit": ["4.2 nM"],
                    "binding_pattern": "candidate_target_endpoint_value",
                    "bound_spans": {
                        "candidate": {
                            "normalized_phrase": "drug one",
                            "text": "Drug one",
                            "start": candidate_start,
                            "end": candidate_start + len("Drug one"),
                        },
                        "endpoint": {
                            "normalized_phrase": "ic50",
                            "text": "IC50",
                            "start": endpoint_start,
                            "end": endpoint_start + len("IC50"),
                        },
                        "target": {
                            "normalized_phrase": "gene",
                            "text": "GENE",
                            "start": target_start,
                            "end": target_start + len("GENE"),
                        },
                        "value": {
                            "text": "4.2 nM",
                            "start": value_start,
                            "end": value_start + len("4.2 nM"),
                            "relation": "",
                            "number": "4.2",
                            "unit": "nM",
                        },
                    },
                    "competitor_alias_matches": [],
                    "excerpt_truncated": False,
                    "policy": (
                        "exact_candidate_target_endpoint_value_same_clause_v3"
                    ),
                },
                "activity_document_selection": {
                    "policy": "bounded_top_k_activity_document_rows_v1",
                    "top_k": 3,
                    "eligible_row_count": 1,
                    "inspected_row_count": 1,
                    "truncated": False,
                    "selected_rank": 1,
                    "selected_activity_id": "11",
                    "rows": [
                        {
                            "rank": 1,
                            "activity_id": "11",
                            "document_chembl_id": "CHEMBL900001",
                            "endpoint": "IC50",
                            "pchembl_value": 8.3,
                            "verification_status": "verified",
                            "reason_code": None,
                        }
                    ],
                    "fixture": False,
                    "missing_values_imputed": False,
                },
                "document_query": {
                    "source": "ChEMBL",
                    "endpoint": (
                        "https://www.ebi.ac.uk/chembl/api/data/document.json"
                    ),
                    "parameter_keys": ["document_chembl_id__in", "limit"],
                    "parameter_hash": document_parameter_hash,
                    "status_code": 200,
                    "response_sha256": "a" * 64,
                },
                "document_query_batch": document_batch,
            },
        },
    )
    target = TargetCandidate(
        symbol="GENE",
        uniprot_id="P12345",
        metadata={"chembl_target_id": "CHEMBL_TARGET"},
    )
    candidate = CandidateCard(
        candidate_id="cand_primary_test",
        route=CandidateRoute.REPURPOSING,
        name="Drug one",
        target=target,
        mechanism="Measured inhibitory potency against GENE",
        smiles="CCOc1ccccc1",
        inchikey="TEST1",
        activities=[activity],
        evidence=[evidence],
        metadata={
            "fixture": False,
            "chembl_id": "CHEMBL1",
            "identifiers": ["CHEMBL1"],
            "chembl_exact_aliases": ["Drug one", "Research code-1"],
            "chembl_alias_source": (
                "chembl_pref_name_and_molecule_synonyms_exact_v1"
            ),
            "chembl_alias_conflicts": [],
        },
        status="conditional",
    )
    return CandidatePacket(
        plan_id="plan_primary_test",
        candidates=[candidate],
        backend_kind=ScientificBackendKind.LIVE,
    )


def _primary_findings(
    packet: CandidatePacket,
    *,
    require_primary_sources: bool = True,
):
    return MA3AssuranceAgent()._audit_candidate_primary_sources(
        _brief(require_primary_sources=require_primary_sources),
        packet,
    )


def _replace_primary_excerpt(packet: CandidatePacket, excerpt: str) -> None:
    evidence = packet.candidates[0].evidence[0]
    primary = evidence.metadata["primary_source"]
    excerpt_hash = sha256(excerpt.encode("utf-8")).hexdigest()
    primary["excerpt"] = excerpt
    primary["excerpt_hash"] = excerpt_hash
    primary["excerpt_location"]["normalized_char_end"] = (
        primary["excerpt_location"]["normalized_char_start"] + len(excerpt)
    )
    evidence.excerpt_hash = excerpt_hash


def _rehash_document_batch(primary: dict) -> None:
    batch = primary["document_query_batch"]
    parameters = batch["query_parameters"]
    parameter_hash = sha256(
        json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    batch["query_parameter_hash"] = parameter_hash
    batch["call_parameter_hash"] = parameter_hash
    primary["document_query"]["parameter_hash"] = parameter_hash
    body = dict(batch)
    body.pop("attestation_sha256", None)
    batch["attestation_sha256"] = sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _two_row_selection_packet() -> CandidatePacket:
    packet = _live_packet().model_copy(deep=True)
    primary = packet.candidates[0].evidence[0].metadata["primary_source"]
    selection = primary["activity_document_selection"]
    selected = dict(selection["rows"][0])
    selected["rank"] = 2
    selection.update(
        {
            "eligible_row_count": 2,
            "inspected_row_count": 2,
            "selected_rank": 2,
            "rows": [
                {
                    "rank": 1,
                    "activity_id": "10",
                    "document_chembl_id": "CHEMBL900002",
                    "endpoint": "IC50",
                    "pchembl_value": 9.0,
                    "verification_status": "unverified",
                    "reason_code": "PRIMARY_ABSTRACT_RELEVANCE_UNRESOLVED",
                },
                selected,
            ],
        }
    )
    batch = primary["document_query_batch"]
    batch.update(
        {
            "document_chembl_ids": [
                "CHEMBL900001",
                "CHEMBL900002",
            ],
            "document_count": 2,
            "query_parameters": {
                "document_chembl_id__in": (
                    "CHEMBL900001,CHEMBL900002"
                ),
                "limit": 2,
            },
        }
    )
    _rehash_document_batch(primary)
    return packet


def test_ma3_accepts_exact_hash_verified_primary_candidate_evidence() -> None:
    assert _primary_findings(_live_packet()) == []


def test_ma3_accepts_first_verified_row_after_auditing_a_larger_exact_batch(
) -> None:
    assert _primary_findings(_two_row_selection_packet()) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_key",
        "below_floor",
        "nonfinite",
        "wrong_order",
        "selected_not_first_verified",
        "invalid_activity_id",
        "leading_zero_activity_id",
        "document_whitespace",
    ],
)
def test_ma3_rejects_activity_document_selection_row_or_order_tamper(
    mutation: str,
) -> None:
    packet = _two_row_selection_packet()
    selection = packet.candidates[0].evidence[0].metadata["primary_source"][
        "activity_document_selection"
    ]
    rows = selection["rows"]
    if mutation == "extra_key":
        rows[0]["untrusted"] = True
    elif mutation == "below_floor":
        rows[0]["pchembl_value"] = 4.9
    elif mutation == "nonfinite":
        rows[0]["pchembl_value"] = float("nan")
    elif mutation == "wrong_order":
        rows[0]["pchembl_value"] = 8.0
        rows[1]["pchembl_value"] = 9.0
    elif mutation == "selected_not_first_verified":
        rows[0]["verification_status"] = "verified"
        rows[0]["reason_code"] = None
    elif mutation == "invalid_activity_id":
        rows[0]["activity_id"] = "not-an-id"
    elif mutation == "leading_zero_activity_id":
        rows[0]["activity_id"] = "010"
    else:
        rows[0]["document_chembl_id"] = " CHEMBL900002"

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


def test_ma3_accepts_finite_pchembl_above_fourteen_without_inventing_a_cap(
) -> None:
    packet = _two_row_selection_packet()
    selection = packet.candidates[0].evidence[0].metadata["primary_source"][
        "activity_document_selection"
    ]
    selection["rows"][0]["pchembl_value"] = 15.0

    assert _primary_findings(packet) == []


@pytest.mark.parametrize(
    "mutation",
    ["wrong_selected_id", "unsorted_ids", "call_hash", "truncated"],
)
def test_ma3_rejects_rehashed_document_batch_binding_tamper(
    mutation: str,
) -> None:
    packet = _two_row_selection_packet()
    primary = packet.candidates[0].evidence[0].metadata["primary_source"]
    batch = primary["document_query_batch"]
    if mutation == "wrong_selected_id":
        batch["selected_document_chembl_id"] = "CHEMBL900002"
    elif mutation == "unsorted_ids":
        batch["document_chembl_ids"] = list(
            reversed(batch["document_chembl_ids"])
        )
        batch["query_parameters"]["document_chembl_id__in"] = ",".join(
            batch["document_chembl_ids"]
        )
    elif mutation == "call_hash":
        batch["call_parameter_hash"] = "f" * 64
    else:
        batch["truncated"] = True
    if mutation != "call_hash":
        _rehash_document_batch(primary)
    else:
        body = dict(batch)
        body.pop("attestation_sha256")
        batch["attestation_sha256"] = sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("standard_type", "Ki"),
        ("standard_relation", "<"),
        ("standard_value", "5.0"),
        ("standard_units", "nm"),
        ("standard_units", "µM"),
    ],
)
def test_ma3_rejects_tampered_raw_activity_even_when_attestation_matches_tamper(
    field: str,
    value: object,
) -> None:
    packet = _live_packet().model_copy(deep=True)
    evidence = packet.candidates[0].evidence[0]
    evidence.metadata[field] = value
    deposited = evidence.metadata["primary_source"]["activity_binding"][
        "deposited_activity"
    ]
    deposited[field] = value

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "excluded_type",
        "parameter_hash",
        "missing_exact_pmid",
        "duplicate_result_uid",
        "unrequested_result_uid",
        "doi_only",
    ],
)
def test_ma3_independently_rejects_pubmed_type_or_query_tamper(
    mutation: str,
) -> None:
    packet = _live_packet().model_copy(deep=True)
    primary = packet.candidates[0].evidence[0].metadata["primary_source"]
    pubmed = primary["pubmed_type_verification"]
    if mutation == "excluded_type":
        primary["publication_types"] = ["Journal Article", "Review"]
        pubmed["publication_types"] = ["Journal Article", "Review"]
        pubmed["excluded_publication_types"] = []
    elif mutation == "parameter_hash":
        pubmed["query"]["parameter_hash"] = "d" * 64
    elif mutation == "missing_exact_pmid":
        pubmed["batch"]["observed_result_uids"] = []
    elif mutation == "duplicate_result_uid":
        pubmed["batch"]["observed_result_uids"] = ["12345678", "12345678"]
    elif mutation == "unrequested_result_uid":
        pubmed["batch"]["observed_result_uids"].append("99999999")
    elif mutation == "doi_only":
        primary["pmid"] = None

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


def test_ma3_rejects_competitor_alias_inventory_hash_tamper() -> None:
    packet = _live_packet().model_copy(deep=True)
    provenance = packet.candidates[0].evidence[0].metadata[
        "primary_source"
    ]["competitor_alias_provenance"]
    provenance["inventory"]["records"][0]["aliases"].append("Injected alias")

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


@pytest.mark.parametrize(
    "mutation",
    ["incomplete", "missing_molecule", "aliasless_molecule"],
)
def test_ma3_rejects_incomplete_pool_alias_inventory(mutation: str) -> None:
    packet = _live_packet().model_copy(deep=True)
    inventory = packet.candidates[0].evidence[0].metadata[
        "primary_source"
    ]["competitor_alias_provenance"]["inventory"]
    if mutation == "incomplete":
        inventory["complete"] = False
    elif mutation == "missing_molecule":
        inventory["missing_molecule_ids"] = ["CHEMBL_MISSING"]
    else:
        inventory["aliasless_molecule_ids"] = ["CHEMBL_ALIASLESS"]

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "excerpt_hash",
        "candidate_id",
        "candidate_id_conflict",
        "candidate_source_id_conflict",
        "document_id",
        "target_id",
        "full_text",
    ],
)
def test_ma3_rejects_tampered_or_overstored_primary_evidence(
    mutation: str,
) -> None:
    packet = _live_packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    evidence = candidate.evidence[0]
    primary = evidence.metadata["primary_source"]
    if mutation == "excerpt_hash":
        primary["excerpt"] = "Tampered primary excerpt."
    elif mutation == "candidate_id":
        candidate.metadata["chembl_id"] = "CHEMBL_OTHER"
        candidate.metadata["identifiers"] = ["CHEMBL_OTHER"]
    elif mutation == "candidate_id_conflict":
        candidate.metadata["chembl_id"] = "CHEMBL999"
    elif mutation == "candidate_source_id_conflict":
        candidate.metadata["source_metadata"] = [
            {
                "backend": "ChEMBLLiveAPI",
                "metadata": {"chembl_id": "CHEMBL999"},
            }
        ]
    elif mutation == "document_id":
        primary["document_chembl_id"] = "CHEMBL_OTHER_DOC"
    elif mutation == "target_id":
        primary["activity_binding"]["target_chembl_id"] = "CHEMBL_OTHER_TARGET"
    elif mutation == "full_text":
        primary["abstract"] = "A complete abstract must never be retained here."

    findings = _primary_findings(packet)

    assert [item.code for item in findings] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]
    assert findings[0].artifact_id == candidate.candidate_id


def test_ma3_does_not_promote_title_only_metadata_to_primary_support() -> None:
    packet = _live_packet().model_copy(deep=True)
    evidence = packet.candidates[0].evidence[0]
    primary = evidence.metadata["primary_source"]
    primary.update(
        {
            "verification_status": "metadata_only",
            "is_primary_support": False,
            "reason_code": "PRIMARY_ABSTRACT_NOT_AVAILABLE",
            "retrieved_field": None,
            "excerpt": None,
            "excerpt_location": None,
            "excerpt_hash": None,
            "content_hash": None,
        }
    )
    evidence.excerpt_hash = None

    findings = _primary_findings(packet)

    assert [item.code for item in findings] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


@pytest.mark.parametrize(
    "excerpt",
    [
        "GENE had an IC50 of 4.2 nM in the reported assay.",
        "Drug one caused GENE activation at 4.2 nM.",
        "Drug one inhibited GENE with a low nanomolar IC50.",
        "Drug one inhibited GENE with an IC50 of 0.42 nM.",
        "Drug one inhibited GENE with an IC50 of 0.0042 uM.",
        "Drug one inhibited GENE; IC50 was 4.2 nM.",
        (
            "Drug one inhibited EGFR with an IC50 of 4.2 nM, and GENE "
            "was separately measured."
        ),
        "Drug one inhibited EGFR and GENE with an IC50 of 4.2 nM.",
        (
            "Drug one inhibited GENE with IC50 values of "
            "4.2 nM and 4.2 nM."
        ),
    ],
)
def test_ma3_independently_rejects_missing_exact_relevance_or_bound_value(
    excerpt: str,
) -> None:
    packet = _live_packet().model_copy(deep=True)
    _replace_primary_excerpt(packet, excerpt)

    findings = _primary_findings(packet)

    assert [item.code for item in findings] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


def test_ma3_rejects_non_equal_activity_without_bound_relation_symbol() -> None:
    packet = _live_packet().model_copy(deep=True)
    activity = packet.candidates[0].activities[0]
    evidence = packet.candidates[0].evidence[0]
    activity.relation = "<"
    evidence.metadata["primary_source"]["activity_binding"]["relation"] = "<"

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("doc_type", "PATENT"),
        ("title", "A systematic review of primary activity studies"),
    ],
)
def test_ma3_rejects_patent_or_review_claims(field: str, value: str) -> None:
    packet = _live_packet().model_copy(deep=True)
    packet.candidates[0].evidence[0].metadata["primary_source"][field] = value

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


def test_ma3_rejects_verified_flag_when_exact_alias_has_pool_conflict() -> None:
    packet = _live_packet().model_copy(deep=True)
    candidate = packet.candidates[0]
    primary = candidate.evidence[0].metadata["primary_source"]
    primary["candidate_alias_conflicts"] = ["Research code-1"]
    candidate.metadata["chembl_alias_conflicts"] = ["Research code-1"]

    assert [item.code for item in _primary_findings(packet)] == [
        "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED"
    ]


def test_ma3_primary_policy_false_preserves_existing_live_path() -> None:
    packet = _live_packet().model_copy(deep=True)
    packet.candidates[0].evidence[0].metadata.pop("primary_source")
    packet.candidates[0].evidence[0].excerpt_hash = None

    assert _primary_findings(
        packet,
        require_primary_sources=False,
    ) == []
