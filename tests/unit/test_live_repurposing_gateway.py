from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateRoute,
    EvidenceRef,
    TargetCandidate,
)
from drugpilot.tools.live_repurposing_gateway import (
    LiveScientificToolNotConfigured,
    NoRealRepurposingCandidates,
    PAIRED_V4_REQUIRED_POLICY,
    PairedV4RepurposingContractUnavailable,
    PublicRepurposingGateway,
    RepurposingSourceUnavailable,
)
from drugpilot.tools.public_chemistry import SourceEnrichment
from drugpilot.tools.protocols import CandidateSeed


HEADER = (
    "DRUG_NAME\tSTRUCT_ID\tTARGET_NAME\tACCESSION\tGENE\tACT_VALUE\t"
    "ACT_UNIT\tACT_TYPE\tACT_SOURCE\tRELATION\tMOA\tACTION_TYPE\t"
    "ORGANISM\tSMILES\tINCHIKEY\n"
)


class FakeLookupAdapter:
    def __init__(self, records_by_query: dict[str, tuple[EvidenceRef, ...]] | None = None) -> None:
        self.records_by_query = records_by_query or {}
        self.requests: list[Any] = []

    def lookup(self, request):
        self.requests.append(request)
        key = request.candidate_name or request.uniprot_id or ""
        records = self.records_by_query.get(key, ())
        source = records[0].source if records else "PubChem"
        return records, SourceEnrichment(
            source=source,
            status="ok" if records else "empty",
            queried=True,
            record_count=len(records),
        )


class FakeScientificDelegate:
    source_version = "chembl-live-test-v1"
    backend_kind = "live"
    fixture = False

    def __init__(
        self,
        seeds: tuple[CandidateSeed, ...] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.seeds = seeds
        self.error = error
        self.requests: list[tuple[str, str, int]] = []
        self.last_operation_provenance: dict[str, object] = {}

    def retrieve_repurposing(self, target, *, direction: str, limit: int):
        self.requests.append((target.symbol, direction, limit))
        if self.error is not None:
            raise self.error
        result = self.seeds[:limit]
        self.last_operation_provenance = {
            "operation": "retrieve_repurposing",
            "source": "ChEMBL live API",
            "candidate_count": len(result),
            "fixture": False,
        }
        return result


class FakePairedV4Delegate(FakeScientificDelegate):
    def __init__(
        self,
        seeds: tuple[CandidateSeed, ...],
        provenance: dict[str, object],
        *,
        error: Exception | None = None,
    ) -> None:
        super().__init__(seeds, error=error)
        self.provenance = provenance

    def retrieve_repurposing(self, target, *, direction: str, limit: int):
        self.requests.append((target.symbol, direction, limit))
        if self.error is not None:
            raise self.error
        self.last_operation_provenance = deepcopy(self.provenance)
        self.last_operation_provenance.setdefault(
            "requested_eligible_count", limit
        )
        # LiveScientificGateway returns its complete bounded surviving pool;
        # ``limit`` belongs to MA2's later eligible-quota evaluation.
        return self.seeds


def _pubchem(name: str, *, cid: str, smiles: str, inchikey: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"ev_pubchem_{cid}",
        source="PubChem",
        external_id=cid,
        title=name,
        supports="context",
        metadata={
            "live": True,
            "mock": False,
            "provenance_class": "external_live",
            "canonical_smiles": smiles,
            "inchikey": inchikey,
        },
    )


def _chembl(name: str, *, chembl_id: str, smiles: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"ev_chembl_{chembl_id}",
        source="ChEMBL",
        external_id=chembl_id,
        title=name,
        supports="context",
        metadata={
            "live": True,
            "mock": False,
            "provenance_class": "external_live",
            "canonical_smiles": smiles,
            "standard_inchi_key": f"KEY-{chembl_id}",
        },
    )


def _bindingdb(*, record_id: str, smiles: str, value: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"ev_bindingdb_{record_id}",
        source="BindingDB",
        external_id=record_id,
        title=f"BindingDB {record_id}",
        supports="context",
        metadata={
            "live": True,
            "mock": False,
            "provenance_class": "external_live",
            "smiles": smiles,
            "affinity_type": "Ki",
            "affinity_value": value,
        },
    )


def _measured_delegate_seed(
    *,
    name: str,
    chembl_id: str,
    smiles: str,
    inchikey: str,
    endpoint: str = "IC50",
    regulatory_status: str = "not_withdrawn_in_queried_chembl_warnings",
    warnings: list[dict[str, object]] | None = None,
) -> CandidateSeed:
    evidence_id = f"ev_chembl_activity_{chembl_id}"
    evidence = EvidenceRef(
        evidence_id=evidence_id,
        source="ChEMBL",
        external_id=f"ACT-{chembl_id}",
        title=f"Measured {endpoint} for {chembl_id}",
        supports="support",
        metadata={
            "fixture": False,
            "live": True,
            "provenance_class": "external_live",
            "target_chembl_id": "CHEMBL_TARGET",
            "molecule_chembl_id": chembl_id,
            "pchembl_value": 8.1,
        },
    )
    activity = ActivityMeasurement(
        endpoint=endpoint,
        relation="=",
        value=7.0,
        unit="nM",
        assay_type="ChEMBL single protein assay",
        source_evidence_id=evidence_id,
    )
    return CandidateSeed(
        name=name,
        smiles=smiles,
        mechanism="measured target potency; intervention direction evaluated separately",
        activities=(activity,),
        evidence=(evidence,),
        metadata={
            "fixture": False,
            "live": True,
            "chembl_id": chembl_id,
            "drug_id": chembl_id,
            "measured_target": "CHEMBL_TARGET",
            "standard_inchi_key": inchikey,
            "max_phase": 4,
            "regulatory_status": regulatory_status,
            "drug_warnings": warnings or [],
        },
    )


def _with_pchembl(seed: CandidateSeed, value: float) -> CandidateSeed:
    evidence = seed.evidence[0].model_copy(
        update={
            "metadata": {
                **seed.evidence[0].metadata,
                "pchembl_value": value,
            }
        },
        deep=True,
    )
    return replace(seed, evidence=(evidence,))


def _paired_v4_delegate_contract(
    seeds: tuple[CandidateSeed, ...],
    *,
    promoted_ids: tuple[str, ...],
    prewarning_ids: tuple[str, ...] | None = None,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    candidate_ids = tuple(str(seed.metadata["chembl_id"]) for seed in seeds)
    fixed_ids = prewarning_ids or candidate_ids
    bounds: dict[str, object] = {
        "schema_version": "bounded_mechanism_first_candidate_identity_pool_v1",
        "discovery_lane": "chembl_mechanism_first_single_default_v1",
        "target_type": "SINGLE PROTEIN",
        "target_organism": "Homo sapiens",
        "mechanism_action_type": "INHIBITOR",
        "raw_mechanism_row_limit": 80,
        "mechanism_complete_within_bound": True,
        "prewarning_pool_limit": 12,
        "fixed_prewarning_pool_limit": 12,
        "prewarning_pool_count": len(fixed_ids),
        "prewarning_pool_ids": list(fixed_ids),
        "preactivity_pool_ids": list(fixed_ids),
        "activity_query_attempted_ids": list(fixed_ids),
        "prewarning_selection_policy": (
            "clinical_max_phase_desc_then_parent_molecule_chembl_id_v1"
        ),
        "selection_inputs": ["max_phase", "parent_molecule_chembl_id"],
        "pchembl_used_for_prewarning_selection": False,
        "primary_document_used_for_prewarning_selection": False,
        "warning_used_for_prewarning_selection": False,
        "activity_used_for_prewarning_selection": False,
        "activity_candidate_query_limit": 12,
        "no_activity_first_fallback": True,
        "returned_candidate_ordering_policy": (
            "surviving_fixed_prewarning_clinical_phase_desc_then_"
            "stable_parent_molecule_chembl_id_subsequence_v1"
        ),
        "requested_eligible_count": 3,
        "gateway_return_candidate_count": len(candidate_ids),
        "gateway_return_candidate_ids": list(candidate_ids),
        "downstream_candidate_identity_count": len(candidate_ids),
        "downstream_candidate_identity_ids": list(candidate_ids),
        "gateway_returned_from_fixed_prewarning_pool": True,
        "gateway_return_order_matches_fixed_prewarning_subsequence": True,
        "gateway_return_cap_applied": False,
        "eligible_quota_evaluation_owner": "MA2",
        "no_backfill_beyond_prewarning_pool": True,
        "fixture": False,
        "missing_values_imputed": False,
        "primary_abstract_search": {
            "policy": (
                "same_sentence_primary_pairs_exact_chembl_activity_and_assay_v4"
            ),
            "no_backfill": True,
            "fixture": False,
            "promotion": {
                "status": "verified",
                "promoted_candidate_ids": list(promoted_ids),
            },
        },
    }
    contracted: list[CandidateSeed] = []
    for seed, candidate_id in zip(seeds, candidate_ids, strict=True):
        envelope = {
            "schema_version": "candidate_primary_evidence_paired_v4",
            "verification_status": "verified",
            "policy": (
                "same_sentence_primary_pairs_exact_chembl_activity_and_assay_v4"
            ),
            "molecule_chembl_id": candidate_id,
            "fixture": False,
            "missing_values_imputed": False,
        }
        seed_metadata = {
            **seed.metadata,
            "candidate_discovery_bounds": deepcopy(bounds),
        }
        if candidate_id in promoted_ids:
            seed_metadata["primary_abstract_paired_v4"] = deepcopy(envelope)
        evidence = []
        for item in seed.evidence:
            metadata = {
                **item.metadata,
                "candidate_discovery_bounds": deepcopy(bounds),
            }
            if candidate_id in promoted_ids:
                metadata["primary_abstract_paired_v4"] = deepcopy(envelope)
            evidence.append(
                item.model_copy(update={"metadata": metadata}, deep=True)
            )
        contracted.append(
            replace(
                seed,
                evidence=tuple(evidence),
                metadata=seed_metadata,
            )
        )
    provenance: dict[str, object] = {
        "operation": "retrieve_repurposing",
        "status": "completed",
        "source": "ChEMBL live API",
        "candidate_count": len(contracted),
        "requested_limit": 3,
        "requested_eligible_count": 3,
        "fixed_prewarning_pool_limit": 12,
        "gateway_return_candidate_ids": list(candidate_ids),
        "gateway_return_cap_applied": False,
        "gateway_return_order_matches_fixed_prewarning_subsequence": True,
        "eligible_quota_evaluation_owner": "MA2",
        "candidate_discovery_bounds": deepcopy(bounds),
        "fixture": False,
    }
    return tuple(contracted), provenance


def test_public_candidate_priority_never_uses_pchembl_as_tiebreak() -> None:
    gateway = object.__new__(PublicRepurposingGateway)
    seed_a = _measured_delegate_seed(
        name="A",
        chembl_id="CHEMBL1",
        smiles="CCO",
        inchikey="KEY-A",
    )
    seed_b = _measured_delegate_seed(
        name="B",
        chembl_id="CHEMBL2",
        smiles="CCN",
        inchikey="KEY-B",
    )
    first = [_with_pchembl(seed_a, 5.0), _with_pchembl(seed_b, 12.0)]
    swapped = [_with_pchembl(seed_a, 12.0), _with_pchembl(seed_b, 5.0)]

    first_order = [
        seed.metadata["chembl_id"]
        for seed in sorted(first, key=gateway._candidate_priority)
    ]
    swapped_order = [
        seed.metadata["chembl_id"]
        for seed in sorted(swapped, key=gateway._candidate_priority)
    ]

    assert first_order == swapped_order == ["CHEMBL1", "CHEMBL2"]


def _write_dataset(path: Path, *rows: str) -> Path:
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def _gateway(
    dataset: Path,
    *,
    pubchem: FakeLookupAdapter | None = None,
    chembl: FakeLookupAdapter | None = None,
    bindingdb: FakeLookupAdapter | None = None,
    delegate: FakeScientificDelegate | None = None,
    repurposing_evidence_policy: str | None = None,
) -> PublicRepurposingGateway:
    return PublicRepurposingGateway(
        drugcentral_dataset_path=dataset,
        drugcentral_dataset_version="test-snapshot",
        pubchem=pubchem or FakeLookupAdapter(),
        chembl=chembl or FakeLookupAdapter(),
        bindingdb=bindingdb or FakeLookupAdapter(),
        delegate=delegate,
        repurposing_evidence_policy=repurposing_evidence_policy,
    )


def test_retrieves_only_exact_target_and_direction_compatible_real_drugs(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "quizartinib\tDC001\tFLT3\tP36888\tFLT3\t4.2\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCOC1=CC=CC=C1\tKEY-QUIZ\n",
        "agonist-x\tDC002\tFLT3\tP36888\tFLT3\t12\tnM\tEC50\tCHEMBL\t=\t1\tAGONIST\tHomo sapiens\tCCN\tKEY-AGO\n",
        "other-target\tDC003\tEGFR\tP00533\tEGFR\t8\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCCC\tKEY-OTHER\n",
    )
    gateway = _gateway(dataset)

    seeds = gateway.retrieve_repurposing(
        TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
        direction="inhibit",
        limit=5,
    )

    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.name == "quizartinib"
    assert seed.smiles == "COC1=CC=CC=C1"
    assert seed.metadata["drug_id"] == "DC001"
    assert seed.metadata["fixture"] is False
    assert seed.metadata["structure_source"] == "DrugCentralLocalTSV"
    assert seed.activities[0].endpoint == "IC50"
    assert seed.activities[0].value == 4.2
    assert seed.activities[0].unit == "nM"
    drugcentral = next(item for item in seed.evidence if item.source == "DrugCentral")
    assert drugcentral.external_id == "DC001"
    assert drugcentral.supports == "support"
    assert drugcentral.metadata["dataset_row_number"] == 2
    assert drugcentral.metadata["direction_evidence_rule"] == "drugcentral_action_type"
    assert drugcentral.metadata["raw_activity"] == {
        "ACT_TYPE": "IC50",
        "ACT_VALUE": "4.2",
        "ACT_UNIT": "nM",
        "RELATION": "=",
        "ACT_SOURCE": "CHEMBL",
    }


def test_blank_activity_unit_is_preserved_but_never_guessed(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "ivosidenib\t5292\tIDH1\tO75874\tIDH1\t7.149\t\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\t\t\n",
    )
    pubchem = FakeLookupAdapter(
        {
            "ivosidenib": (
                _pubchem(
                    "ivosidenib",
                    cid="71657455",
                    smiles="CC1=NC(=NO1)C2=CC=CC=C2",
                    inchikey="KEY-IVO",
                ),
            )
        }
    )
    gateway = _gateway(dataset, pubchem=pubchem)

    seed = gateway.retrieve_repurposing(
        TargetCandidate(symbol="IDH1", uniprot_id="O75874"),
        direction="inhibit",
        limit=5,
    )[0]

    assert seed.activities == ()
    assert seed.metadata["activity_units_inferred"] is False
    assert seed.metadata["raw_activity_records"][0]["ACT_VALUE"] == "7.149"
    assert seed.metadata["raw_activity_records"][0]["ACT_UNIT"] is None
    assert seed.metadata["structure_source"] == "PubChem"
    assert {item.source for item in seed.evidence} == {"DrugCentral", "PubChem"}


def test_chembl_full_text_result_requires_exact_preferred_name(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "named-drug\tDC010\tFLT3\tP36888\tFLT3\t10\tnM\tIC50\tCHEMBL\t=\t\tINHIBITOR\tHomo sapiens\t\t\n",
    )
    chembl = FakeLookupAdapter(
        {
            "named-drug": (
                _chembl("similar named drug", chembl_id="CHEMBL-WRONG", smiles="CCC"),
                _chembl("named-drug", chembl_id="CHEMBL-RIGHT", smiles="CCO"),
            )
        }
    )
    gateway = _gateway(dataset, chembl=chembl)

    seed = gateway.retrieve_repurposing(
        TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
        direction="inhibit",
        limit=5,
    )[0]

    assert seed.smiles == "CCO"
    assert seed.metadata["structure_source"] == "ChEMBL"
    assert any(item.external_id == "CHEMBL-RIGHT" for item in seed.evidence)
    assert not any(item.external_id == "CHEMBL-WRONG" for item in seed.evidence)


def test_bindingdb_is_attached_only_on_exact_structure_and_declared_unit(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "quizartinib\tDC001\tFLT3\tP36888\tFLT3\t4.2\t\tIC50\tCHEMBL\t=\t\tINHIBITOR\tHomo sapiens\tCCO\tKEY-QUIZ\n",
    )
    bindingdb = FakeLookupAdapter(
        {
            "P36888": (
                _bindingdb(record_id="501", smiles="CCO", value="< 2.5 nM"),
                _bindingdb(record_id="502", smiles="CCC", value="1.0 nM"),
                _bindingdb(record_id="503", smiles="CCO", value="6.2"),
            )
        }
    )
    gateway = _gateway(dataset, bindingdb=bindingdb)

    seed = gateway.retrieve_repurposing(
        TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
        direction="inhibit",
        limit=5,
    )[0]

    binding_evidence = [item for item in seed.evidence if item.source == "BindingDB"]
    assert {item.external_id for item in binding_evidence} == {"501", "503"}
    assert len(seed.activities) == 1
    assert seed.activities[0].endpoint == "Ki"
    assert seed.activities[0].relation == "<"
    assert seed.activities[0].value == 2.5
    assert seed.activities[0].unit == "nM"


def test_gene_symbol_is_exact_fallback_when_uniprot_is_missing(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "gene-drug\tDC020\tDNMT3A\tQ9Y6K1\tDNMT3A\t3\tuM\tIC50\tSOURCE\t=\t\tINHIBITOR\tHomo sapiens\tCCN\tKEY-GENE\n",
    )
    gateway = _gateway(dataset)

    seed = gateway.retrieve_repurposing(
        TargetCandidate(symbol="DNMT3A"),
        direction="inhibit",
        limit=5,
    )[0]

    assert seed.name == "gene-drug"
    assert seed.metadata["target_gene"] == "DNMT3A"


def test_zero_results_raise_typed_error_without_candidate_fill(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "agonist-x\tDC002\tFLT3\tP36888\tFLT3\t12\tnM\tEC50\tCHEMBL\t=\t1\tAGONIST\tHomo sapiens\tCCN\tKEY-AGO\n",
    )
    gateway = _gateway(dataset)

    with pytest.raises(NoRealRepurposingCandidates) as captured:
        gateway.retrieve_repurposing(
            TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
            direction="inhibit",
            limit=5,
        )

    assert captured.value.matching_rows == 1
    assert captured.value.direction_compatible_rows == 0
    assert "No fixture candidates were substituted" in str(captured.value)


def test_unresolved_structure_is_not_replaced_with_a_fixture(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "unresolved-drug\tDC100\tFLT3\tP36888\tFLT3\t12\tnM\tIC50\tCHEMBL\t=\t\tINHIBITOR\tHomo sapiens\t\t\n",
    )
    gateway = _gateway(dataset)

    with pytest.raises(NoRealRepurposingCandidates) as captured:
        gateway.retrieve_repurposing(
            TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
            direction="inhibit",
            limit=5,
        )

    assert captured.value.unresolved_structures == 1


def test_missing_dataset_and_unconfigured_calculations_fail_closed(tmp_path) -> None:
    gateway = _gateway(tmp_path / "missing.tsv")
    target = TargetCandidate(symbol="FLT3", uniprot_id="P36888")

    with pytest.raises(RepurposingSourceUnavailable):
        gateway.retrieve_repurposing(target, direction="inhibit", limit=5)
    with pytest.raises(LiveScientificToolNotConfigured, match="fixture fallback is disabled"):
        gateway.dock(smiles="CCO", target=target)
    with pytest.raises(LiveScientificToolNotConfigured):
        gateway.restore_structure(
            candidate_name="missing",
            route=CandidateRoute.REPURPOSING,
            target=target,
            direction="inhibit",
        )


def test_retrieved_structure_and_activity_can_be_restored_from_live_cache(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "quizartinib\tDC001\tFLT3\tP36888\tFLT3\t4.2\tnM\tIC50\tCHEMBL\t=\t\tINHIBITOR\tHomo sapiens\tCCO\tKEY-QUIZ\n",
    )
    gateway = _gateway(dataset)
    target = TargetCandidate(symbol="FLT3", uniprot_id="P36888")
    seed = gateway.retrieve_repurposing(target, direction="inhibit", limit=5)[0]

    restored = gateway.restore_structure(
        candidate_name="QUIZARTINIB",
        route=CandidateRoute.REPURPOSING,
        target=target,
        direction="inhibit",
    )
    activities = gateway.restore_activity(candidate_name="quizartinib", target=target)

    assert restored == seed
    assert activities == seed.activities


def test_drugcentral_and_delegate_are_both_queried_and_exact_structure_is_merged(
    tmp_path,
) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "ivosidenib\t5292\tIDH1\tO75874\tIDH1\t7.149\t\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCCO\tKEY-IVO\n",
    )
    warning = {
        "warning_type": "Black Box Warning",
        "warning_description": "Differentiation syndrome risk",
    }
    delegate = FakeScientificDelegate(
        (
            _measured_delegate_seed(
                name="Ivosidenib",
                chembl_id="CHEMBL3989910",
                smiles="CCO",
                inchikey="KEY-IVO",
                warnings=[warning],
            ),
        )
    )
    gateway = _gateway(dataset, delegate=delegate)

    seeds = gateway.retrieve_repurposing(
        TargetCandidate(symbol="IDH1", uniprot_id="O75874"),
        direction="inhibit",
        limit=1,
    )

    assert len(seeds) == 1
    seed = seeds[0]
    assert delegate.requests and delegate.requests[0][:2] == ("IDH1", "inhibit")
    assert seed.name == "Ivosidenib"
    assert seed.activities[0].endpoint == "IC50"
    assert seed.activities[0].source_evidence_id == "ev_chembl_activity_CHEMBL3989910"
    assert {item.source for item in seed.evidence} == {"DrugCentral", "ChEMBL"}
    assert set(seed.metadata["retrieval_backends"]) == {
        "ChEMBLLiveAPI",
        "DrugCentralLocalTSV",
    }
    assert set(seed.metadata["identifiers"]) == {"5292", "CHEMBL3989910"}
    assert seed.metadata["drug_warnings"] == [warning]
    assert seed.metadata["fixture"] is False
    assert gateway.last_operation_provenance["drugcentral_candidates"] == 1
    assert gateway.last_operation_provenance["delegate_candidates_accepted"] == 1
    assert gateway.last_operation_provenance["returned_candidates"] == 1


def test_measured_exact_target_candidate_is_preferred_over_unitless_local_row(
    tmp_path,
) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "local-drug\tDC010\tIDH1\tO75874\tIDH1\t5.0\t\tIC50\tSOURCE\t=\t1\tINHIBITOR\tHomo sapiens\tCCC\tKEY-LOCAL\n",
    )
    delegate = FakeScientificDelegate(
        (
            _measured_delegate_seed(
                name="Measured drug",
                chembl_id="CHEMBL-MEASURED",
                smiles="CCN",
                inchikey="KEY-MEASURED",
            ),
        )
    )
    gateway = _gateway(dataset, delegate=delegate)

    seeds = gateway.retrieve_repurposing(
        TargetCandidate(symbol="IDH1", uniprot_id="O75874"),
        direction="inhibit",
        limit=1,
    )

    assert [seed.name for seed in seeds] == ["Measured drug", "local-drug"]
    assert seeds[0].metadata["target_linked_activity_count"] == 1
    assert len(seeds) <= 10
    assert gateway.last_operation_provenance["requested_eligible_count"] == 1
    assert gateway.last_operation_provenance["discovery_pool_limit"] == 10
    assert gateway.last_operation_provenance["discovery_pool_count"] == 2
    assert gateway.last_operation_provenance["bounded_discovery"] is True


def test_delegate_binding_only_candidate_cannot_claim_inhibitory_direction(
    tmp_path,
) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "agonist-x\tDC002\tIDH1\tO75874\tIDH1\t12\tnM\tEC50\tCHEMBL\t=\t1\tAGONIST\tHomo sapiens\tCCN\tKEY-AGO\n",
    )
    delegate = FakeScientificDelegate(
        (
            _measured_delegate_seed(
                name="Binding-only drug",
                chembl_id="CHEMBL-KI",
                smiles="CCO",
                inchikey="KEY-KI",
                endpoint="Ki",
            ),
        )
    )
    gateway = _gateway(dataset, delegate=delegate)

    with pytest.raises(NoRealRepurposingCandidates) as captured:
        gateway.retrieve_repurposing(
            TargetCandidate(symbol="IDH1", uniprot_id="O75874"),
            direction="inhibit",
            limit=3,
        )

    assert captured.value.direction_compatible_rows == 0
    assert any(
        "all_candidates_rejected" in item
        for item in captured.value.criteria_rejections
    )
    assert captured.value.source_failures == ()


def test_real_local_result_survives_delegate_outage_with_partial_provenance(
    tmp_path,
) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "quizartinib\tDC001\tFLT3\tP36888\tFLT3\t4.2\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCOC\tKEY-QUIZ\n",
    )
    delegate = FakeScientificDelegate(error=RuntimeError("upstream unavailable"))
    gateway = _gateway(dataset, delegate=delegate)

    seeds = gateway.retrieve_repurposing(
        TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
        direction="inhibit",
        limit=2,
    )

    assert [seed.name for seed in seeds] == ["quizartinib"]
    assert gateway.last_operation_provenance["partial_source_result"] is True
    assert gateway.last_operation_provenance["fixture"] is False
    assert any(
        "upstream unavailable" in item
        for item in gateway.last_operation_provenance["source_failures"]
    )


def test_nonwithdrawn_measured_candidate_ranks_before_withdrawn_candidate(
    tmp_path,
) -> None:
    dataset = _write_dataset(tmp_path / "drug.target.interaction.tsv")
    delegate = FakeScientificDelegate(
        (
            _measured_delegate_seed(
                name="Withdrawn drug",
                chembl_id="CHEMBL-W",
                smiles="CCO",
                inchikey="KEY-W",
                regulatory_status="withdrawn",
                warnings=[{"warning_type": "Withdrawn"}],
            ),
            _measured_delegate_seed(
                name="Eligible drug",
                chembl_id="CHEMBL-E",
                smiles="CCN",
                inchikey="KEY-E",
            ),
        )
    )
    gateway = _gateway(dataset, delegate=delegate)

    seeds = gateway.retrieve_repurposing(
        TargetCandidate(symbol="IDH1", uniprot_id="O75874"),
        direction="inhibit",
        limit=1,
    )

    assert [seed.name for seed in seeds] == ["Eligible drug", "Withdrawn drug"]
    assert seeds[1].metadata["regulatory_status"] == "withdrawn"
    assert gateway.last_operation_provenance["requested_eligible_count"] == 1
    assert gateway.last_operation_provenance["discovery_pool_count"] == 2
    assert seeds[0].metadata["regulatory_status"].startswith("not_withdrawn")


def test_exact_paired_v4_policy_preserves_complete_delegate_pool_and_order(
    tmp_path,
) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "local-favorite\tDC900\tJAK2\tO60674\tJAK2\t1.0\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCNC\tKEY-LOCAL\n",
    )
    raw = tuple(
        _measured_delegate_seed(
            name=f"Delegate {candidate_id}",
            chembl_id=candidate_id,
            smiles=f"CC{'N' * index}",
            inchikey=f"KEY-{candidate_id}",
        )
        for index, candidate_id in enumerate(
            ("CHEMBL11", "CHEMBL12", "CHEMBL13", "CHEMBL14"),
            start=1,
        )
    )
    contracted, provenance = _paired_v4_delegate_contract(
        raw,
        promoted_ids=("CHEMBL12", "CHEMBL13", "CHEMBL14"),
        prewarning_ids=(
            "CHEMBL11",
            "CHEMBL999",
            "CHEMBL12",
            "CHEMBL13",
            "CHEMBL14",
        ),
    )
    delegate = FakePairedV4Delegate(contracted, provenance)
    pubchem = FakeLookupAdapter()
    bindingdb = FakeLookupAdapter()
    gateway = _gateway(
        dataset,
        pubchem=pubchem,
        bindingdb=bindingdb,
        delegate=delegate,
        repurposing_evidence_policy=PAIRED_V4_REQUIRED_POLICY,
    )

    result = gateway.retrieve_repurposing(
        TargetCandidate(symbol="JAK2", uniprot_id="O60674"),
        direction="inhibit",
        limit=3,
    )

    assert result == contracted
    assert all(actual is expected for actual, expected in zip(result, contracted))
    assert delegate.requests == [("JAK2", "inhibit", 3)]
    assert pubchem.requests == []
    assert bindingdb.requests == []
    assert gateway.last_operation_provenance[
        "gateway_return_candidate_ids"
    ] == ["CHEMBL11", "CHEMBL12", "CHEMBL13", "CHEMBL14"]
    assert gateway.last_operation_provenance[
        "paired_v4_envelope_candidate_ids"
    ] == ["CHEMBL12", "CHEMBL13", "CHEMBL14"]
    assert gateway.last_operation_provenance[
        "delegate_pool_preserved_exactly"
    ] is True
    assert gateway.last_operation_provenance[
        "local_candidate_discovery_executed"
    ] is False
    assert gateway.last_operation_provenance["candidate_merge_used"] is False
    assert gateway.last_operation_provenance["candidate_reranking_used"] is False


def test_exact_paired_v4_policy_fails_closed_without_local_fallback(
    tmp_path,
) -> None:
    dataset = _write_dataset(
        tmp_path / "drug.target.interaction.tsv",
        "local-favorite\tDC900\tJAK2\tO60674\tJAK2\t1.0\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCNC\tKEY-LOCAL\n",
    )
    delegate = FakeScientificDelegate(error=RuntimeError("delegate outage"))
    gateway = _gateway(
        dataset,
        delegate=delegate,
        repurposing_evidence_policy=PAIRED_V4_REQUIRED_POLICY,
    )

    with pytest.raises(PairedV4RepurposingContractUnavailable) as captured:
        gateway.retrieve_repurposing(
            TargetCandidate(symbol="JAK2", uniprot_id="O60674"),
            direction="inhibit",
            limit=3,
        )

    assert captured.value.reason_code == (
        "PAIRED_V4_DELEGATE_RETRIEVAL_UNAVAILABLE"
    )
    assert captured.value.provenance["fixture_fallback_used"] is False
    assert gateway.last_operation_provenance == {}


@pytest.mark.parametrize("contract_failure", ["missing_bounds", "missing_envelope"])
def test_exact_paired_v4_policy_rejects_absent_delegate_contract(
    tmp_path,
    contract_failure: str,
) -> None:
    dataset = _write_dataset(tmp_path / "drug.target.interaction.tsv")
    raw = tuple(
        _measured_delegate_seed(
            name=f"Delegate {candidate_id}",
            chembl_id=candidate_id,
            smiles=f"CN{'C' * index}",
            inchikey=f"KEY-{candidate_id}",
        )
        for index, candidate_id in enumerate(
            ("CHEMBL21", "CHEMBL22", "CHEMBL23"),
            start=1,
        )
    )
    contracted, provenance = _paired_v4_delegate_contract(
        raw,
        promoted_ids=("CHEMBL21", "CHEMBL22", "CHEMBL23"),
    )
    if contract_failure == "missing_bounds":
        provenance.pop("candidate_discovery_bounds")
        expected_reason = "PAIRED_V4_DISCOVERY_BOUNDS_MISSING"
    else:
        first = contracted[0]
        seed_metadata = dict(first.metadata)
        seed_metadata.pop("primary_abstract_paired_v4")
        evidence = []
        for item in first.evidence:
            metadata = dict(item.metadata)
            metadata.pop("primary_abstract_paired_v4")
            evidence.append(item.model_copy(update={"metadata": metadata}, deep=True))
        contracted = (
            replace(first, metadata=seed_metadata, evidence=tuple(evidence)),
            *contracted[1:],
        )
        expected_reason = "PAIRED_V4_ENVELOPE_MEMBERSHIP_MISMATCH"
    delegate = FakePairedV4Delegate(contracted, provenance)
    gateway = _gateway(
        dataset,
        delegate=delegate,
        repurposing_evidence_policy=PAIRED_V4_REQUIRED_POLICY,
    )

    with pytest.raises(PairedV4RepurposingContractUnavailable) as captured:
        gateway.retrieve_repurposing(
            TargetCandidate(symbol="JAK2", uniprot_id="O60674"),
            direction="inhibit",
            limit=3,
        )

    assert captured.value.reason_code == expected_reason


def test_nonexact_paired_v4_token_preserves_generic_merge_path(tmp_path) -> None:
    dataset = _write_dataset(tmp_path / "drug.target.interaction.tsv")
    delegate = FakeScientificDelegate(
        (
            _measured_delegate_seed(
                name="Generic delegate",
                chembl_id="CHEMBL31",
                smiles="CCN",
                inchikey="KEY-CHEMBL31",
            ),
        )
    )
    gateway = _gateway(
        dataset,
        delegate=delegate,
        repurposing_evidence_policy=f"{PAIRED_V4_REQUIRED_POLICY}_extra",
    )

    result = gateway.retrieve_repurposing(
        TargetCandidate(symbol="JAK2", uniprot_id="O60674"),
        direction="inhibit",
        limit=2,
    )

    assert [seed.metadata["chembl_id"] for seed in result] == ["CHEMBL31"]
    assert delegate.requests == [("JAK2", "inhibit", 10)]
    assert gateway.last_operation_provenance["method"] == (
        "exact-target DrugCentral + measured ChEMBL evidence merge"
    )
    assert "delegate_pool_preserved_exactly" not in (
        gateway.last_operation_provenance
    )
