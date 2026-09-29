"""Offline structure-identity regressions; no scientific result is fabricated."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from drugpilot.agents.ma2 import MA2Agent, _TraceRecorder
from drugpilot.domain.models import (
    CandidateRoute, EvidenceRef, PlanSpec, PlanTask, ScientificBackendKind, TargetCandidate,
)
from drugpilot.tools.chemistry import ChemistryValidator
from drugpilot.tools.live_repurposing_gateway import PublicRepurposingGateway, _DrugCentralRow
from drugpilot.tools.public_chemistry import ChemistryLookup, PubChemAdapter, SourceEnrichment


# Reported PubChem structure pairs, not candidate/gate success fixtures.
STRUCTURES = [
    ("ivosidenib", "71657455",
     "C1CC(=O)N(C1C(=O)N(C2=CC(=CN=C2)F)C(C3=CC=CC=C3Cl)C(=O)NC4CC(C4)(F)F)C5=NC=CC(=C5)C#N",
     "C1CC(=O)N([C@@H]1C(=O)N(C2=CC(=CN=C2)F)[C@@H](C3=CC=CC=C3Cl)C(=O)NC4CC(C4)(F)F)C5=NC=CC(=C5)C#N",
     "WIJZXSAJMHAVGX-DHLKQENFSA-N", 2),
    ("azacitidine", "9444", "C1=NC(=NC(=O)N1C2C(C(C(O2)CO)O)O)N",
     "C1=NC(=NC(=O)N1[C@H]2[C@@H]([C@@H]([C@H](O2)CO)O)O)N",
     "NMUSYJAQQFHJEW-KVTDHHQDSA-N", 4),
    ("decitabine", "451668", "C1C(C(OC1N2C=NC(=NC2=O)N)CO)O",
     "C1[C@@H]([C@H](O[C@H]1N2C=NC(=NC2=O)N)CO)O",
     "XAUDJQYHKZQPEU-KVQBGUIXSA-N", 3),
]


class Lookup:
    def __init__(self, records=()):
        self.records = tuple(records)

    def lookup(self, request):
        return self.records, SourceEnrichment(source="PubChem", status="ok", queried=True,
                                             record_count=len(self.records))


def record(name, cid, canonical, isomeric, key):
    return EvidenceRef(source="PubChem", external_id=cid, title=name, supports="context",
                       metadata={"canonical_smiles": canonical, "isomeric_smiles": isomeric,
                                 "inchikey": key, "live": True, "mock": False,
                                 "provenance_class": "external_live",
                                 "isomeric_smiles_source_field": "SMILES"})


def row(name="source-compound"):
    return _DrugCentralRow(row_number=2, drug_name=name, drug_id="DC1",
        target_accession="O75874", target_gene="IDH1", target_name="IDH1",
        organism="Homo sapiens", action_type="INHIBITOR", moa="1",
        activity_type="IC50", activity_value="25", activity_unit="nM", relation="=",
        activity_source="CHEMBL", activity_source_url=None, smiles=None, inchikey=None)


def gateway(tmp_path, records):
    return PublicRepurposingGateway(drugcentral_dataset_path=tmp_path / "unused.tsv",
        pubchem=Lookup(records), chembl=Lookup(), bindingdb=Lookup())


@pytest.mark.parametrize("name,cid,canonical,isomeric,key,centers", STRUCTURES)
def test_reported_isomeric_structure_selected_and_unchanged(tmp_path, name, cid, canonical, isomeric, key, centers):
    Chem = pytest.importorskip("rdkit.Chem")
    source = record(name, cid, canonical, isomeric, key)
    original = source.model_dump(mode="json")
    result = gateway(tmp_path, [source])._resolve_structure(row(name))
    assert result is not None
    selected, selected_key, evidence, origin = result
    assert (selected, selected_key, origin) == (isomeric, key, "PubChem")
    assert evidence[0].model_dump(mode="json") == original
    assert source.model_dump(mode="json") == original
    normalized = ChemistryValidator().inspect(selected)
    assert normalized.backend == "rdkit" and normalized.valid
    assert normalized.inchikey == key
    assert sum(label == "?" for _, label in Chem.FindMolChiralCenters(
        Chem.MolFromSmiles(normalized.canonical_smiles), includeUnassigned=True)) == 0
    assert sum(label == "?" for _, label in Chem.FindMolChiralCenters(
        Chem.MolFromSmiles(canonical), includeUnassigned=True)) == centers
    trace = PublicRepurposingGateway._structure_selection_trace(*result)
    assert trace["metadata_field"] == "isomeric_smiles"
    assert trace["selected_smiles_sha256"] == sha256(isomeric.encode()).hexdigest()
    assert trace["source_property_field"] == "SMILES"
    assert trace["stereochemistry_invented"] is False


@pytest.mark.parametrize("bad_isomeric", ["not-valid-smiles", "C[C@@H](O)C(=O)O", ""])
def test_invalid_or_opposite_stereochemistry_not_silently_downgraded(tmp_path, bad_isomeric):
    pytest.importorskip("rdkit.Chem")
    good = "C[C@H](O)C(=O)O"
    key = ChemistryValidator().inspect(good).inchikey
    source = record("lactic acid", "x", "CC(O)C(=O)O", bad_isomeric, key)
    result = gateway(tmp_path, [source])._resolve_structure(row())
    assert result is None


def test_no_reported_isomeric_or_key_does_not_invent_stereochemistry(tmp_path):
    source = record("unknown stereo", "x", "CC(O)C(=O)O", None, None)
    result = gateway(tmp_path, [source])._resolve_structure(row())
    assert result[0] == "CC(O)C(=O)O"
    assert "@" not in result[0] and result[1] is None


def test_partial_stereochemistry_rejected_against_full_reported_key(tmp_path):
    pytest.importorskip("rdkit.Chem")
    name, cid, canonical, isomeric, key, _ = STRUCTURES[0]
    partial = isomeric.replace("[C@@H]", "C", 1)
    source = record(name, cid, canonical, partial, key)
    assert gateway(tmp_path, [source])._resolve_structure(row(name)) is None


def test_bad_reported_full_inchikey_rejected_even_when_connectivity_same(tmp_path):
    pytest.importorskip("rdkit.Chem")
    source = record("test", "x", "CC(O)C(=O)O", "C[C@H](O)C(=O)O", "BAD-FULL-KEY")
    assert gateway(tmp_path, [source])._resolve_structure(row()) is None


@pytest.mark.parametrize("conflict", ["opposite", "partial", "canonical_only"])
def test_explicit_identity_conflict_cannot_fall_through_to_keyless_chembl(tmp_path, conflict):
    pytest.importorskip("rdkit.Chem")
    if conflict == "opposite":
        canonical = "CC(O)C(=O)O"
        isomeric = "C[C@@H](O)C(=O)O"
        key = ChemistryValidator().inspect("C[C@H](O)C(=O)O").inchikey
    else:
        _, _, canonical, original, key, _ = STRUCTURES[0]
        isomeric = original.replace("[C@@H]", "C", 1) if conflict == "partial" else None
    compound = row("exact-name")
    pubchem = record(compound.drug_name, "x", canonical, isomeric, key)
    fallback = EvidenceRef(source="ChEMBL", external_id="CHEMBL1", title=compound.drug_name,
        supports="context", metadata={"canonical_smiles": canonical, "live": True,
                                      "provenance_class": "external_live"})
    resolver = PublicRepurposingGateway(drugcentral_dataset_path=tmp_path / "unused.tsv",
        pubchem=Lookup([pubchem]), chembl=Lookup([fallback]), bindingdb=Lookup())
    assert resolver._resolve_structure(compound) is None


def test_achiral_reported_isomeric_does_not_create_stereochemistry(tmp_path):
    pytest.importorskip("rdkit.Chem")
    key = ChemistryValidator().inspect("CCO").inchikey
    result = gateway(tmp_path, [record("ethanol", "702", "CCO", "CCO", key)])._resolve_structure(row())
    assert result[0] == "CCO" and result[1] == key


def test_unparseable_reported_isomeric_cannot_discard_declared_key_via_fallback(tmp_path):
    pytest.importorskip("rdkit.Chem")
    compound = row("exact-name")
    key = ChemistryValidator().inspect("C[C@H](O)C(=O)O").inchikey
    pubchem = record(compound.drug_name, "x", "CC(O)C(=O)O", "not-valid-smiles", key)
    fallback = EvidenceRef(source="ChEMBL", external_id="CHEMBL1", title=compound.drug_name,
        supports="context", metadata={"canonical_smiles": "CC(O)C(=O)O", "live": True,
                                      "provenance_class": "external_live"})
    resolver = PublicRepurposingGateway(drugcentral_dataset_path=tmp_path / "unused.tsv",
        pubchem=Lookup([pubchem]), chembl=Lookup([fallback]), bindingdb=Lookup())
    assert resolver._resolve_structure(compound) is None


@pytest.mark.parametrize("canonical_field,isomeric_field", [
    ("CanonicalSMILES", "IsomericSMILES"), ("ConnectivitySMILES", "SMILES")])
def test_pubchem_property_aliases_preserve_original_field_names(canonical_field, isomeric_field):
    name, cid, canonical, isomeric, key, _ = STRUCTURES[1]
    payload = {"PropertyTable": {"Properties": [{"CID": int(cid), "IUPACName": name,
        canonical_field: canonical, isomeric_field: isomeric, "InChIKey": key}]}}
    original = deepcopy(payload)

    class Response:
        def raise_for_status(self): pass
        def json(self): return payload

    class Client:
        def get(self, *args, **kwargs): return Response()

    records, _ = PubChemAdapter(client=Client()).lookup(ChemistryLookup(candidate_name=name))
    assert payload == original
    assert records[0].metadata["canonical_smiles"] == canonical
    assert records[0].metadata["isomeric_smiles"] == isomeric
    assert records[0].metadata["canonical_smiles_source_field"] == canonical_field
    assert records[0].metadata["isomeric_smiles_source_field"] == isomeric_field


@pytest.mark.parametrize("name,cid,canonical,isomeric,key,centers", STRUCTURES)
def test_ma2_actual_admet_argument_preserves_selected_structure(tmp_path, name, cid, canonical, isomeric, key, centers):
    """Exercise actual RDKit + MA2 argument propagation with offline predictor spy."""
    pytest.importorskip("rdkit.Chem")
    path = tmp_path / "drugcentral.tsv"
    path.write_text("DRUG_NAME\tSTRUCT_ID\tTARGET_NAME\tACCESSION\tGENE\tACT_VALUE\tACT_UNIT\tACT_TYPE\tACT_SOURCE\tRELATION\tMOA\tACTION_TYPE\tORGANISM\tSMILES\tINCHIKEY\n"
                    f"{name}\tDC1\tIDH1\tO75874\tIDH1\t25\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\t\t\n", encoding="utf-8")
    target = TargetCandidate(symbol="IDH1", uniprot_id="O75874")
    source_gateway = PublicRepurposingGateway(drugcentral_dataset_path=path,
        pubchem=Lookup([record(name, cid, canonical, isomeric, key)]), chembl=Lookup(), bindingdb=Lookup())
    seed = source_gateway.retrieve_repurposing(target, direction="inhibit", limit=1)[0]
    assert seed.smiles == isomeric
    assert seed.metadata["structure_selection"]["metadata_field"] == "isomeric_smiles"

    class PredictorSpy:
        backend_kind = "live"
        fixture = False
        source_version = "offline-call-argument-test"
        last_operation_provenance = {}
        inputs = []
        def predict_admet(self, *, smiles):
            self.inputs.append(smiles)
            self.last_operation_provenance = {"fixture": False, "model_version": "test-only"}
            return {name: 0.1 for name in ["hepatotoxicity", "herg_risk", "ames_risk",
                    "cyp3a4_inhibition", "clearance_risk", "oral_bioavailability"]}

    spy = PredictorSpy()
    agent = MA2Agent(gateway=spy, chemistry=ChemistryValidator())
    plan = PlanSpec(hypothesis_packet_id="test-only", tasks=[PlanTask(
        name="chemistry and ADMET input propagation", route=CandidateRoute.REPURPOSING)], approved_tools=[],
                    stop_conditions=[], candidates_per_route=1, total_token_budget=10000)
    recorder = _TraceRecorder("test-only", backend_kind=ScientificBackendKind.LIVE, fixture=False)
    candidate, _ = agent._evaluate_once(plan=plan, recorder=recorder, target=target,
                                        route=CandidateRoute.REPURPOSING, seed=seed, trace_ids=[])
    assert candidate.chemistry_execution.valid
    assert candidate.inchikey == key
    assert spy.inputs == [candidate.smiles]
    assert ChemistryValidator().inspect(spy.inputs[0]).inchikey == key
    assert candidate.metadata["structure_selection"]["selected_smiles_sha256"] == sha256(isomeric.encode()).hexdigest()
    expected_admet_hash = sha256(json.dumps({"smiles": spy.inputs[0]}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert candidate.admet_execution.input_hash == expected_admet_hash
