from __future__ import annotations

from hashlib import sha256
import re
from typing import Mapping, Sequence

from drugpilot.domain.models import (
    ActivityMeasurement,
    CandidateRoute,
    EvidenceRef,
    GatewayReadiness,
    ReadinessStatus,
    ScientificBackendKind,
    TargetCandidate,
    ToolReadiness,
)
from drugpilot.tools.protocols import CandidateSeed


_TARGETS: dict[str, tuple[str, str]] = {
    "FLT3": ("P36888", "Receptor-type tyrosine-protein kinase FLT3"),
    "MOLM-13": ("P36888", "Receptor-type tyrosine-protein kinase FLT3"),
    "MOLM13": ("P36888", "Receptor-type tyrosine-protein kinase FLT3"),
    "KRAS": ("P01116", "GTPase KRas"),
    "EGFR": ("P00533", "Epidermal growth factor receptor"),
    "MOLM1": ("P36888", "Receptor-type tyrosine-protein kinase FLT3"),
}

_REPURPOSING_SMILES = (
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "CCN(CC)CCOc1ccc2nc3ccccc3sc2c1",
    "COc1ccc(CCN(C)C)cc1OC",
    "CC(C)NCC(O)c1ccc(O)cc1",
    "CCOC(=O)N1CCC(CC1)c2ccccc2",
    "CC1=CC(=O)NC(=O)N1",
    "CN1CCC(Cc2ccccc2)CC1",
    "CCOC(=O)c1ccc(N)cc1",
    "CC(C)c1ccc(cc1)C(C)C(=O)O",
    "COc1ccccc1OCCN(C)C",
    "CCN1C(=O)NC(=O)C1c2ccccc2",
    "CC(C)OC(=O)c1ccccc1C(=O)O",
    "CN(C)CCc1c[nH]c2ccccc12",
    "CCOc1ccc(NC(=O)C)cc1",
    "CCN(CC)c1ccc(N)cc1",
    "COc1ccc(CN2CCC(O)CC2)cc1",
    "CC(C)NCCc1ccc(O)cc1",
    "CCOC(=O)C1CCN(CC1)C",
    "CN1CCN(CC1)c2ncccn2",
    "CC(C)c1nc(N)nc(N)n1",
)

_DE_NOVO_SMILES = (
    "CCOc1ncc(NC(=O)C2CC2)cc1",
    "COc1ccc(NC(=O)N2CCOCC2)cc1",
    "CC1CCN(C(=O)c2nccs2)CC1",
    "CN1CCC(c2nc3ccccc3n2C)CC1",
    "CCOc1ccc(C2=NN(C)C(=O)C2)cc1",
    "CCCl",
    "COc1ncc(CN2CCOCC2)cc1N",
    "CC1(C)CCN(C(=O)c2ccno2)CC1",
    "CN(C)c1ncnc2cc(OC)ccc12",
    "CCOc1cc(NC(=O)C2CCNCC2)ncn1",
    "CCBr",
    "COc1ccc2ncc(C(=O)N3CCOCC3)cc2c1",
    "CCN1CCN(C(=O)c2cnc(N)nc2)CC1",
    "COc1cc(C2CC2)nc(NC(=O)C)n1",
    "CCOc1ccc(N2CCN(C)CC2)nc1",
    "CCCCl",
    "CN1CCN(Cc2ncc(C3CC3)o2)CC1",
    "CC1CN(C(=O)c2ccc(F)cc2)CCO1",
    "COc1ncc(N2CCC(C)CC2)nc1",
    "CCCBr",
)


def _index_from_name(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 1


def _fraction(*parts: str) -> float:
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class DeterministicFixtureGateway:
    """Offline scientific-tool fixture used for repeatable integration tests."""

    source_version = "drugpilot-fixture-2026.08"
    backend_kind = ScientificBackendKind.FIXTURE.value
    fixture = True

    def readiness(self, target: TargetCandidate | None = None) -> GatewayReadiness:
        operations = (
            ("target_normalizer", "normalize_target", True),
            ("repurposing_search", "retrieve_candidates", True),
            ("de_novo_generator", "generate_candidates", True),
            ("docking", "score_binding_pose", False),
            ("admet_predictor", "predict_six_endpoint_panel", True),
        )
        return GatewayReadiness(
            gateway_name=type(self).__name__,
            source_version=self.source_version,
            backend_kind=ScientificBackendKind.FIXTURE,
            fixture=True,
            capabilities=[
                ToolReadiness(
                    tool_name=tool_name,
                    operation=operation,
                    status=ReadinessStatus.READY,
                    required=required,
                    backend_kind=ScientificBackendKind.FIXTURE,
                    backend_name="deterministic-offline-fixture",
                    backend_version=self.source_version,
                    fixture=True,
                    provenance={
                        "target": target.symbol if target is not None else None,
                        "scientific_use": False,
                    },
                )
                for tool_name, operation, required in operations
            ],
        )

    def normalize_target(self, target: TargetCandidate) -> TargetCandidate:
        raw = re.sub(r"\s+", "", target.symbol).upper()
        symbol = "FLT3" if raw in {"MOLM-13", "MOLM13", "MOLM1"} else raw
        uniprot, name = _TARGETS.get(symbol, (target.uniprot_id, target.target_name))
        return target.model_copy(
            update={
                "symbol": symbol,
                "uniprot_id": uniprot,
                "target_name": name,
            }
        )

    def retrieve_repurposing(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]:
        return [self._repurposing_seed(i, target, direction) for i in range(1, limit + 1)]

    def generate_de_novo(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]:
        seeds: list[CandidateSeed] = []
        for i in range(1, limit + 1):
            smiles = _DE_NOVO_SMILES[(i - 1) % len(_DE_NOVO_SMILES)]
            # The optional 20-candidate load test intentionally includes one
            # traceable structural exclusion.  Normal 5+5 demonstrations are
            # unaffected, while G1 still exercises its policy of accepting an
            # explicitly excluded/traceable candidate without treating it as
            # a missing-safety failure.
            structural_stress_fixture = i == 20
            if structural_stress_fixture:
                smiles = "C1(CC"
            cycle = (i - 1) // len(_DE_NOVO_SMILES)
            if cycle and not structural_stress_fixture:
                smiles = f"C{'C' * cycle}{smiles}"
            provenance = EvidenceRef(
                evidence_id=f"ev_fixture_denovo_{target.symbol.lower()}_{i:03d}",
                source="DE-NOVO-GENERATOR-FIXTURE",
                external_id=f"GEN-FIXTURE-{target.symbol}-{i:03d}",
                title=f"Offline scaffold-generation provenance for {target.symbol} candidate {i}",
                supports="context",
                metadata={"fixture": True, "generator": "fixture-scaffold-hopping"},
            )
            seeds.append(
                CandidateSeed(
                    name=f"DN-{target.symbol}-{i:03d}",
                    smiles=smiles,
                    mechanism=f"{direction} {target.symbol}",
                    evidence=(provenance,),
                    metadata={
                        "generator": "fixture-scaffold-hopping",
                        "iteration": 0,
                        "structural_stress_fixture": structural_stress_fixture,
                    },
                )
            )
        return seeds

    def dock(self, *, smiles: str, target: TargetCandidate) -> float:
        return round(-5.8 - 4.2 * _fraction(smiles, target.symbol, "dock"), 3)

    def predict_admet(self, *, smiles: str) -> Mapping[str, float]:
        halogen_alert = 0.52 if "Cl" in smiles or "Br" in smiles or "I" in smiles else 0.0
        base = 0.08 + 0.18 * _fraction(smiles, "admet")
        return {
            "hepatotoxicity": round(min(0.98, base + halogen_alert), 3),
            "herg_risk": round(min(0.98, 0.07 + 0.30 * _fraction(smiles, "herg") + halogen_alert / 2), 3),
            "ames_risk": round(0.05 + 0.22 * _fraction(smiles, "ames"), 3),
            "cyp3a4_inhibition": round(0.08 + 0.36 * _fraction(smiles, "cyp3a4"), 3),
            "clearance_risk": round(0.10 + 0.30 * _fraction(smiles, "clearance"), 3),
            "oral_bioavailability": round(0.55 + 0.38 * _fraction(smiles, "bioavailability"), 3),
        }

    def refine(
        self,
        seed: CandidateSeed,
        *,
        reasons: Sequence[str],
        iteration: int,
    ) -> CandidateSeed:
        smiles = seed.smiles.replace("Cl", "O").replace("Br", "N").replace("I", "O")
        if smiles == seed.smiles:
            smiles = f"CO{smiles}"
        metadata = dict(seed.metadata)
        metadata.update({"iteration": iteration, "refinement_reasons": list(reasons)})
        return CandidateSeed(
            name=seed.name,
            smiles=smiles,
            mechanism=seed.mechanism,
            activities=seed.activities,
            evidence=seed.evidence,
            metadata=metadata,
        )

    def restore_activity(
        self,
        *,
        candidate_name: str,
        target: TargetCandidate,
    ) -> Sequence[ActivityMeasurement]:
        index = _index_from_name(candidate_name)
        return (self._activity(index, target),)

    def restore_structure(
        self,
        *,
        candidate_name: str,
        route: CandidateRoute,
        target: TargetCandidate,
        direction: str,
    ) -> CandidateSeed:
        index = _index_from_name(candidate_name)
        if route is CandidateRoute.REPURPOSING:
            return self._repurposing_seed(index, target, direction)
        generated = self.generate_de_novo(target, direction=direction, limit=index)
        return generated[-1]

    def _repurposing_seed(self, index: int, target: TargetCandidate, direction: str) -> CandidateSeed:
        smiles = _REPURPOSING_SMILES[(index - 1) % len(_REPURPOSING_SMILES)]
        cycle = (index - 1) // len(_REPURPOSING_SMILES)
        if cycle:
            smiles = f"C{'C' * cycle}{smiles}"
        evidence = EvidenceRef(
            evidence_id=f"ev_fixture_rep_{target.symbol.lower()}_{index:03d}",
            source="ChEMBL-FIXTURE",
            external_id=f"CHEMBL-FIXTURE-{index:05d}",
            title=f"Offline activity fixture {index} for {target.symbol}",
            supports="support",
            metadata={"fixture": True, "approval_status": "approved_or_clinical"},
        )
        return CandidateSeed(
            name=f"RP-{target.symbol}-{index:03d}",
            smiles=smiles,
            mechanism=f"{direction} {target.symbol}",
            activities=(self._activity(index, target, evidence.evidence_id),),
            evidence=(evidence,),
            metadata={
                "fixture": True,
                "drug_id": evidence.external_id,
                "aliases": [f"Fixture {target.symbol} compound {index}"],
                "drug_classes": ["fixture_repurposing_control"],
                "development_stage": "approved_or_clinical",
            },
        )

    def _activity(
        self,
        index: int,
        target: TargetCandidate,
        evidence_id: str | None = None,
    ) -> ActivityMeasurement:
        endpoints = ("IC50", "Ki", "Kd")
        return ActivityMeasurement(
            endpoint=endpoints[(index - 1) % len(endpoints)],
            relation="=",
            value=round(8.0 + index * 6.75 + 12.0 * _fraction(str(index), target.symbol, "activity"), 3),
            unit="nM",
            assay_type="biochemical binding fixture",
            source_evidence_id=evidence_id or f"ev_fixture_rep_{target.symbol.lower()}_{index:03d}",
        )
