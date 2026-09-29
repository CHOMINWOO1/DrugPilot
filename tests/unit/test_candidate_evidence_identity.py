from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.domain.models import (
    CandidateCard,
    CandidateRoute,
    EvidenceRef,
    Hypothesis,
    HypothesisTrack,
    TargetCandidate,
)
from drugpilot.tools.candidate_evidence_identity import (
    attest_candidate_evidence,
    candidate_linked_evidence,
    evidence_scoring_cardinality,
)
from drugpilot.tools.public_chemistry import SourceEnrichment
from drugpilot.tools.restricted_candidate_enrichment import (
    RoutedRestrictedCandidateEnricher,
    _candidate_registry_ids,
)
from drugpilot.tools.restricted_sources import (
    RestrictedSourceResult,
    RestrictedSourceSummary,
)


TARGET = TargetCandidate(symbol="FLT3", uniprot_id="P36888")


def test_restricted_registry_helper_import_constructs_no_chemistry_backend() -> None:
    code = """
import sys
from drugpilot.tools.chemistry import ChemistryValidator
from drugpilot.domain.models import CandidateCard, CandidateRoute, TargetCandidate
constructed = []
original_init = ChemistryValidator.__init__
def observed_init(self, *args, **kwargs):
    constructed.append((args, kwargs))
    return original_init(self, *args, **kwargs)
ChemistryValidator.__init__ = observed_init
from drugpilot.tools.restricted_candidate_enrichment import _candidate_registry_ids
candidate = CandidateCard(
    route=CandidateRoute.REPURPOSING,
    name='candidate-a',
    target=TargetCandidate(symbol='ALK', uniprot_id='Q9UM73'),
    mechanism='inhibit ALK',
    smiles='CCO',
    metadata={'chembl_id': 'CHEMBL42'},
)
assert _candidate_registry_ids(candidate)['ChEMBL'] == ('CHEMBL42',)
loaded = sorted(
    name for name in sys.modules
    if name in {
        'drugpilot.tools.candidate_evidence_identity',
        'rdkit',
    } or name.startswith('rdkit.')
)
assert not loaded, loaded
assert not constructed, constructed
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        disease="acute myeloid leukemia",
        track=HypothesisTrack.LITERATURE,
        target=TARGET,
        mechanism="FLT3 inhibition may suppress AML growth",
        intervention_direction="inhibit",
        falsification_test="No target-linked response in an orthogonal assay",
        claims=[],
        evidence=[],
        score=0.8,
    )


def _candidate(
    name: str,
    smiles: str,
    chembl_id: str,
    drugcentral_id: str,
) -> CandidateCard:
    return CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name=name,
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles=smiles,
        metadata={
            "canonical_smiles": smiles,
            "chembl_id": chembl_id,
            "drugcentral_id": drugcentral_id,
        },
    )


def _record(
    source: str,
    external_id: str,
    *,
    smiles: str | None = None,
    drug_name: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        source=source,
        external_id=external_id,
        title=f"{source} {external_id}",
        supports="context",
        metadata={
            **({"smiles": smiles} if smiles else {}),
            **({"drug_name": drug_name} if drug_name else {}),
            "provenance_class": "external_live",
            "live": True,
            "mock": False,
        },
    )


def _typed_registry_identity_record(
    source: str,
    external_id: str,
    *,
    drug_name: str,
) -> EvidenceRef:
    field = {
        "DrugCentral": "drugcentral_id",
        "DrugBank": "drugbank_id",
        "ChEMBL": "molecule_chembl_id",
        "PubChem": "pubchem_cid",
    }[source]
    return _record(source, external_id, drug_name=drug_name).model_copy(
        update={
            "metadata": {
                "drug_name": drug_name,
                field: external_id,
                "candidate_identity_link": {"status": "exact_candidate"},
            }
        },
        deep=True,
    )


def test_identity_precedence_rejects_a_conflicting_stable_source_id() -> None:
    candidate = _candidate("candidate-a", "CCO", "CHEMBL1", "DC-A")
    conflict = _record(
        "DrugCentral",
        "DC-B",
        smiles="CCO",
        drug_name="candidate-a",
    )

    decision = attest_candidate_evidence(candidate, conflict)

    assert decision.status == "identity_conflict"
    assert decision.matched_by == "canonical_smiles"
    assert "drugcentral_id" in decision.conflicts
    assert candidate_linked_evidence(candidate, conflict) is None


def test_exact_structure_link_is_attested_in_candidate_provenance() -> None:
    candidate = _candidate("candidate-a", "CCO", "CHEMBL1", "DC-A")
    bindingdb = _record("BindingDB", "BDBM-A", smiles="C(C)O")

    linked = candidate_linked_evidence(candidate, bindingdb)

    assert linked is not None
    attestation = linked.metadata["candidate_identity_link"]
    assert attestation["status"] == "exact_candidate"
    assert attestation["matched_by"] == "canonical_smiles"
    assert attestation["conflict_rejected"] is False


def test_scoring_cardinality_keeps_raw_duplicate_rows_without_score_inflation() -> None:
    original = EvidenceRef(
        evidence_id="ev_row_1",
        source="DrugCentral",
        external_id="DC-42",
        title="One candidate record",
        supports="context",
        excerpt_hash="a" * 64,
        metadata={"source_row": 12},
    )
    duplicate = original.model_copy(
        update={
            "evidence_id": "ev_row_2",
            "metadata": {"source_row": 91},
        },
        deep=True,
    )

    cardinality = evidence_scoring_cardinality([original, duplicate])

    assert cardinality["raw_record_count"] == 2
    assert cardinality["unique_scientific_record_count"] == 1
    assert cardinality["duplicate_physical_row_count"] == 1
    assert cardinality["raw_records_preserved"] is True
    assert cardinality["duplicates_contribute_to_score"] is False


def test_preexisting_misattachment_cannot_become_an_identity_source() -> None:
    unrelated = _record(
        "DrugCentral", "DC-B", drug_name="candidate-b"
    )
    candidate = _candidate("candidate-a", "CCO", "CHEMBL1", "DC-A").model_copy(
        update={"evidence": [unrelated]},
        deep=True,
    )

    decision = attest_candidate_evidence(candidate, unrelated)

    assert decision.status == "identity_conflict"
    assert "drugcentral_id" in decision.conflicts


class _RepeatedTargetEnricher:
    def __init__(self, evidence: list[EvidenceRef]) -> None:
        self.evidence = evidence

    def enrich(self, request, *, allowed_sources=None):
        return SimpleNamespace(
            evidence=[item.model_copy(deep=True) for item in self.evidence],
            sources=[
                SourceEnrichment(
                    source="BindingDB",
                    status="ok",
                    queried=True,
                    record_count=len(self.evidence),
                )
            ],
        )


def test_same_target_enrichment_never_cross_contaminates_candidates() -> None:
    candidates = [
        _candidate("candidate-a", "CCO", "CHEMBL1", "DC-A"),
        _candidate("candidate-b", "CCN", "CHEMBL2", "DC-B"),
        _candidate("candidate-c", "CCC", "CHEMBL3", "DC-C"),
    ]
    binding_a = _record("BindingDB", "BDBM-A", smiles="CCO")
    drugcentral_b = _record(
        "DrugCentral", "DC-B", drug_name="candidate-b"
    )
    target_structure = _record("RCSB PDB", "4RT7")
    agent = MA2Agent(
        evidence_enricher=_RepeatedTargetEnricher(
            [binding_a, drugcentral_b, target_structure]
        ),
        enrichment_per_route=3,
    )
    hypothesis = Hypothesis(
        disease="acute myeloid leukemia",
        track=HypothesisTrack.LITERATURE,
        target=TARGET,
        mechanism="FLT3 inhibition may suppress AML growth",
        intervention_direction="inhibit",
        falsification_test="No target-linked response in an orthogonal assay",
        claims=[],
        evidence=[],
        score=0.8,
    )

    enriched, queries = agent._enrich_ranked_candidates(
        candidates,
        hypothesis=hypothesis,
    )

    by_name = {item.name: item for item in enriched}
    assert {item.external_id for item in by_name["candidate-a"].evidence} == {
        "BDBM-A"
    }
    assert {item.external_id for item in by_name["candidate-b"].evidence} == {
        "DC-B"
    }
    assert by_name["candidate-c"].evidence == []
    assert all(
        item.metadata["candidate_identity_link"]["status"] == "exact_candidate"
        for candidate in enriched
        for item in candidate.evidence
    )
    target_context = [
        item
        for item in queries
        if item.get("record_type") == "target_context_evidence"
    ]
    assert {item["external_id"] for item in target_context} == {
        "BDBM-A",
        "DC-B",
        "4RT7",
    }
    assert all("candidate_id" not in item for item in target_context)
    assert all(item["target_key"] == "uniprot:P36888" for item in target_context)


class _RestrictedAdapter:
    documentation_url = "https://example.test/source"
    required_configuration = ()
    limitations = ()

    def __init__(self, source: str, target_records: list[EvidenceRef]) -> None:
        self.source = source
        self.target_records = target_records
        self.requests = []

    def query(self, request):
        self.requests.append(request)
        evidence = self.target_records if request.operation == "repurposing" else []
        return RestrictedSourceResult(
            query=request,
            evidence=evidence,
            summary=RestrictedSourceSummary(
                source=self.source,
                operation=request.operation,
                status="ok" if evidence else "empty",
                queried=True,
                record_count=len(evidence),
                documentation_url=self.documentation_url,
            ),
        )


class _UnusedPatents:
    source = "Google Patents"
    documentation_url = "https://example.test/patents"
    required_configuration = ()
    limitations = ()

    def query(self, request):  # pragma: no cover - not selected for repurposing
        raise AssertionError(request)


def test_restricted_target_search_separates_candidate_and_target_records() -> None:
    candidate = _candidate("candidate-a", "CCO", "CHEMBL1", "DC-A")
    matching = _record("DrugCentral", "DC-A", drug_name="candidate-a")
    unrelated = _record("DrugCentral", "DC-B", drug_name="candidate-b")
    enricher = RoutedRestrictedCandidateEnricher(
        drugcentral=_RestrictedAdapter(
            "DrugCentral", [matching, unrelated]
        ),
        drugbank=_RestrictedAdapter("DrugBank", []),
        google_patents=_UnusedPatents(),
    )
    result = enricher.enrich_candidate(candidate, _hypothesis())

    assert [item.external_id for item in result.evidence] == ["DC-A"]
    assert [item.external_id for item in result.target_context] == ["DC-B"]
    route = next(
        item
        for item in result.routing
        if item.get("source") == "DrugCentral"
        and item.get("operation") == "repurposing"
    )
    assert route["candidate_record_count"] == 1
    assert route["target_context_record_count"] == 1


def test_restricted_registry_ids_are_namespaced_and_require_exact_attestation() -> None:
    trusted_drugcentral = _typed_registry_identity_record(
        "DrugCentral", "5306", drug_name="candidate-a"
    ).model_copy(
        update={
            "metadata": {
                "drug_name": "candidate-a",
                "drugcentral_id": "5306",
                "candidate_identity_link": {"status": "exact_candidate"},
                "identifiers": {"pubchem_cid": "CID2244"},
            }
        },
        deep=True,
    )
    untrusted_target_context = _record(
        "DrugCentral", "9999", drug_name="different-target-ligand"
    ).model_copy(
        update={
            "metadata": {
                "drug_name": "different-target-ligand",
                "drugcentral_id": "9999",
                "candidate_identity_link": {"status": "target_context"},
            }
        },
        deep=True,
    )
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        evidence=[trusted_drugcentral, untrusted_target_context],
        metadata={
            "canonical_smiles": "CCO",
            "drug_id": "CHEMBL42",
            "chembl_id": "CHEMBL42",
            "identifiers": {
                "drugbank_id": "DB01234",
            },
        },
    )

    identities = _candidate_registry_ids(candidate)

    assert identities == {
        "DrugCentral": ("5306",),
        "DrugBank": ("DB01234",),
        "ChEMBL": ("CHEMBL42",),
        "PubChem": ("2244",),
    }

    drugcentral = _RestrictedAdapter("DrugCentral", [])
    drugbank = _RestrictedAdapter("DrugBank", [])
    enricher = RoutedRestrictedCandidateEnricher(
        drugcentral=drugcentral,
        drugbank=drugbank,
        google_patents=_UnusedPatents(),
    )
    enricher.enrich_candidate(candidate, _hypothesis())

    dc_compound_queries = [
        request
        for request in drugcentral.requests
        if request.operation in {"identity_regulatory", "contraindication"}
    ]
    db_compound_queries = [
        request
        for request in drugbank.requests
        if request.operation in {"identity_regulatory", "contraindication"}
    ]
    assert {request.drug_id for request in dc_compound_queries} == {"5306"}
    assert {request.drug_id for request in db_compound_queries} == {"DB01234"}
    assert all(
        request.drug_id not in {"CHEMBL42", "2244"}
        for request in [*dc_compound_queries, *db_compound_queries]
    )


def test_chembl_activity_external_id_is_not_promoted_to_candidate_compound_id() -> None:
    activity = EvidenceRef(
        source="ChEMBL",
        external_id="1234567",
        title="Measured IC50 for CHEMBL42 against CHEMBL1974",
        supports="support",
        metadata={
            "candidate_identity_link": {"status": "exact_candidate"},
            "molecule_chembl_id": "CHEMBL42",
            "activity_id": "1234567",
            "assay_chembl_id": "CHEMBL999001",
            "document_chembl_id": "CHEMBL999002",
            "target_chembl_id": "CHEMBL1974",
            "provenance_class": "external_live",
        },
    )
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        evidence=[activity],
        metadata={"canonical_smiles": "CCO"},
    )

    identities = _candidate_registry_ids(candidate)

    assert identities["ChEMBL"] == ("CHEMBL42",)
    assert "1234567" not in identities["ChEMBL"]
    assert "CHEMBL999001" not in identities["ChEMBL"]
    assert "CHEMBL999002" not in identities["ChEMBL"]
    assert "CHEMBL1974" not in identities["ChEMBL"]


def test_nested_or_generic_chembl_record_ids_need_compound_identity_semantics() -> None:
    generic_records = [
        EvidenceRef(
            source="ChEMBL",
            external_id=external_id,
            title=f"ChEMBL {record_type} record",
            supports="context",
            metadata={
                "candidate_identity_link": {"status": "exact_candidate"},
                "record_type": record_type,
                "primary_source": {"molecule_chembl_id": "CHEMBL777"},
                "provenance_class": "external_live",
            },
        )
        for record_type, external_id in (
            ("activity", "CHEMBL111"),
            ("assay", "CHEMBL222"),
            ("document", "CHEMBL333"),
            ("target_context", "CHEMBL444"),
        )
    ]
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        evidence=generic_records,
        metadata={"canonical_smiles": "CCO"},
    )

    identities = _candidate_registry_ids(candidate)

    assert identities["ChEMBL"] == ()


def test_generic_external_id_requires_explicit_namespace_and_identity_record_contract() -> None:
    contracted = EvidenceRef(
        source="DrugCentral",
        external_id="5306",
        title="Contracted DrugCentral candidate identity",
        supports="context",
        metadata={
            "candidate_identity_link": {"status": "exact_candidate"},
            "registry_identity_namespace": "DrugCentral",
            "registry_identity_record_type": "candidate_compound_identity",
            "provenance_class": "external_local_dataset",
        },
    )
    wrong_namespace = contracted.model_copy(
        update={
            "external_id": "9999",
            "metadata": {
                **contracted.metadata,
                "registry_identity_namespace": "DrugBank",
            },
        },
        deep=True,
    )
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        evidence=[contracted, wrong_namespace],
        metadata={"canonical_smiles": "CCO"},
    )

    identities = _candidate_registry_ids(candidate)

    assert identities["DrugCentral"] == ("5306",)
    assert identities["DrugBank"] == ()


def test_public_compound_identity_adapter_scope_can_supply_generic_external_id() -> None:
    chembl_identity = EvidenceRef(
        source="ChEMBL",
        external_id="CHEMBL42",
        title="ChEMBL compound CHEMBL42",
        supports="context",
        metadata={
            "candidate_identity_link": {"status": "exact_candidate"},
            "retrieval_adapter": "ChEMBLAdapter",
            "evidence_scope": "compound identity and development metadata",
            "provenance_class": "external_live",
        },
    )
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        evidence=[chembl_identity],
        metadata={"canonical_smiles": "CCO"},
    )

    assert _candidate_registry_ids(candidate)["ChEMBL"] == ("CHEMBL42",)


def test_generic_chembl_id_never_leaks_and_name_fallback_is_exact_normalized() -> None:
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate   alpha",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        metadata={
            "canonical_smiles": "CCO",
            "drug_id": "CHEMBL42",
            "chembl_id": "CHEMBL42",
            "identifiers": ["CHEMBL42", "CID2244"],
        },
    )
    drugcentral = _RestrictedAdapter("DrugCentral", [])
    drugbank = _RestrictedAdapter("DrugBank", [])
    enricher = RoutedRestrictedCandidateEnricher(
        drugcentral=drugcentral,
        drugbank=drugbank,
        google_patents=_UnusedPatents(),
    )

    enricher.enrich_candidate(candidate, _hypothesis())

    compound_queries = [
        request
        for adapter in (drugcentral, drugbank)
        for request in adapter.requests
        if request.operation in {"identity_regulatory", "contraindication"}
    ]
    assert compound_queries
    assert all(request.drug_id is None for request in compound_queries)
    assert all(request.candidate_name == "candidate alpha" for request in compound_queries)


def test_candidate_metadata_generic_source_drug_id_is_not_a_typed_registry_id() -> None:
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        metadata={
            "canonical_smiles": "CCO",
            "source": "DrugCentral",
            "drug_id": "5306",
        },
    )

    assert _candidate_registry_ids(candidate)["DrugCentral"] == ()


def test_conflicting_attested_registry_ids_fail_closed_without_name_fallback() -> None:
    supplied_ids = [
        _typed_registry_identity_record(
            "DrugCentral", value, drug_name="candidate-a"
        )
        for value in ("5306", "9999")
    ]
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        evidence=supplied_ids,
        metadata={
            "canonical_smiles": "CCO",
        },
    )
    drugcentral = _RestrictedAdapter("DrugCentral", [])
    drugbank = _RestrictedAdapter("DrugBank", [])
    enricher = RoutedRestrictedCandidateEnricher(
        drugcentral=drugcentral,
        drugbank=drugbank,
        google_patents=_UnusedPatents(),
    )

    result = enricher.enrich_candidate(candidate, _hypothesis())

    assert [request.operation for request in drugcentral.requests] == ["repurposing"]
    conflict_summaries = [
        summary
        for summary in result.sources
        if summary.source == "DrugCentral"
        and summary.operation in {"identity_regulatory", "contraindication"}
    ]
    assert len(conflict_summaries) == 2
    assert all(summary.status == "error" for summary in conflict_summaries)
    assert all(summary.queried is False for summary in conflict_summaries)
    assert all(
        summary.error_type == "candidate_registry_identity_conflict"
        for summary in conflict_summaries
    )
    conflict_routes = [
        route
        for route in result.routing
        if route.get("source") == "DrugCentral"
        and route.get("operation") in {"identity_regulatory", "contraindication"}
    ]
    assert len(conflict_routes) == 2
    assert all(route["status"] == "error" for route in conflict_routes)
    assert all(route["queried"] is False for route in conflict_routes)
    assert all(
        route["policy_decision"] == "candidate_registry_identity_conflict"
        for route in conflict_routes
    )


def test_policy_rejected_registry_remains_skipped_even_with_identity_conflict() -> None:
    candidate = CandidateCard(
        route=CandidateRoute.REPURPOSING,
        name="candidate-a",
        target=TARGET,
        mechanism="inhibit FLT3",
        smiles="CCO",
        evidence=[
            _typed_registry_identity_record(
                "DrugCentral", value, drug_name="candidate-a"
            )
            for value in ("5306", "9999")
        ],
        metadata={"canonical_smiles": "CCO"},
    )
    drugcentral = _RestrictedAdapter("DrugCentral", [])
    enricher = RoutedRestrictedCandidateEnricher(
        drugcentral=drugcentral,
        drugbank=_RestrictedAdapter("DrugBank", []),
        google_patents=_UnusedPatents(),
    )

    result = enricher.enrich_candidate(
        candidate,
        _hypothesis(),
        allowed_sources=["DrugBank"],
    )

    rejected = [
        summary for summary in result.sources if summary.source == "DrugCentral"
    ]
    assert len(rejected) == 3
    assert all(summary.status == "skipped" for summary in rejected)
    assert all(summary.queried is False for summary in rejected)
    assert all(summary.error_type == "policy_rejected" for summary in rejected)
    rejected_routes = [
        route for route in result.routing if route.get("source") == "DrugCentral"
    ]
    assert len(rejected_routes) == 3
    assert all(route["status"] == "skipped" for route in rejected_routes)
    assert all(route["policy_decision"] == "policy_rejected" for route in rejected_routes)
