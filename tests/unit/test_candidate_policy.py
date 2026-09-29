from __future__ import annotations

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.domain.models import (
    EvidenceRef,
    NamespacedRegistryIdentifier,
    ReservedControlCompound,
)
from drugpilot.tools.candidate_policy import (
    candidate_identity,
    screen_forbidden_drug,
    screen_reserved_control,
)
from drugpilot.tools.protocols import CandidateSeed


def test_forbidden_screen_matches_name_alias_id_and_class_exactly() -> None:
    result = screen_forbidden_drug(
        candidate_name="Allowed primary name",
        metadata={
            "aliases": ["Drug Alpha"],
            "identifiers": {"chembl": "CHEMBL-123"},
            "drug_classes": ["FLT3 inhibitor"],
        },
        forbidden_names=["drug alpha"],
        forbidden_ids=["CHEMBL123"],
        forbidden_classes=["flt3 inhibitor"],
    )

    assert result.excluded is True
    assert result.matched_names == ("drug alpha",)
    assert result.matched_ids == ("CHEMBL123",)
    assert result.matched_classes == ("flt3 inhibitor",)


def test_forbidden_screen_does_not_use_name_or_class_substrings() -> None:
    result = screen_forbidden_drug(
        candidate_name="Drug Alpha Extended",
        metadata={"drug_classes": ["selective FLT3 inhibitor"]},
        forbidden_names=["Drug Alpha"],
        forbidden_classes=["FLT3 inhibitor"],
    )

    assert result.excluded is False


def test_withdrawn_or_banned_status_is_automatically_ineligible() -> None:
    withdrawn = screen_forbidden_drug(
        candidate_name="Historical medicine",
        metadata={"regulatory_status": "Withdrawn"},
    )
    banned = screen_forbidden_drug(
        candidate_name="Prohibited medicine",
        metadata={"marketing_status": "BANNED"},
    )

    assert withdrawn.prohibited_statuses == ("Withdrawn",)
    assert banned.prohibited_statuses == ("BANNED",)
    assert withdrawn.reason_codes == ("prohibited_regulatory_status:Withdrawn",)


def _reserved_control() -> ReservedControlCompound:
    return ReservedControlCompound(
        name="Quizartinib assay control",
        registry_ids=[
            NamespacedRegistryIdentifier(
                namespace="ChEMBL",
                identifier="CHEMBL576982",
            )
        ],
    )


def test_reserved_control_matches_exact_namespaced_registry_identity() -> None:
    result = screen_reserved_control(
        candidate_name="Different display name",
        metadata={"chembl_id": "CHEMBL576982"},
        reserved_controls=[_reserved_control()],
    )

    assert result.matched is True
    assert result.excluded is True
    assert result.reason_codes == ("RESERVED_ASSAY_CONTROL",)
    assert result.matched_registry_ids == ("chembl:chembl576982",)
    assert result.as_metadata()["control_only"] is True


def test_reserved_control_name_fallback_is_not_used_for_stable_id_conflict() -> None:
    result = screen_reserved_control(
        candidate_name="Quizartinib assay control",
        metadata={"chembl_id": "CHEMBL-DIFFERENT"},
        reserved_controls=[_reserved_control()],
    )

    assert result.matched is False
    assert result.identity_conflict is True
    assert result.excluded is True
    assert result.reason_codes == ("RESERVED_CONTROL_IDENTITY_CONFLICT",)
    assert result.as_metadata()["control_only"] is False


def test_reserved_control_does_not_fuzzy_match_name_or_registry_namespace() -> None:
    result = screen_reserved_control(
        candidate_name="Quizartinib assay control extended",
        metadata={"identifiers": {"DrugBank": "CHEMBL576982"}},
        reserved_controls=[_reserved_control()],
    )

    assert result.excluded is False
    assert candidate_identity(
        candidate_name="Quizartinib assay control extended",
        metadata={"identifiers": {"DrugBank": "CHEMBL576982"}},
    ) == {
        "name": "Quizartinib assay control extended",
        "registry_ids": [
            {"namespace": "drugbank", "identifier": "chembl576982"}
        ],
    }


def _control(identifier: str) -> ReservedControlCompound:
    return ReservedControlCompound(
        name=f"reserved control {identifier}",
        registry_ids=[
            NamespacedRegistryIdentifier(
                namespace="ChEMBL",
                identifier=identifier,
            )
        ],
    )


def test_seed_activity_external_id_is_not_treated_as_candidate_compound_id() -> None:
    activity_id = "CHEMBL-ACTIVITY-RESERVED"
    molecule_id = "CHEMBL-MOLECULE-REAL"
    evidence = EvidenceRef(
        source="ChEMBL",
        external_id=activity_id,
        title="Measured activity record",
        supports="support",
        metadata={
            "molecule_chembl_id": molecule_id,
            "activity_id": activity_id,
            "assay_chembl_id": "CHEMBL-ASSAY-RESERVED",
            "document_chembl_id": "CHEMBL-DOCUMENT-RESERVED",
            "target_chembl_id": "CHEMBL-TARGET-RESERVED",
        },
    )
    seed = CandidateSeed(
        name="real candidate",
        smiles="CCO",
        mechanism="measured target inhibition",
        evidence=(evidence,),
        metadata={"chembl_id": molecule_id},
    )
    policy_metadata = MA2Agent._seed_policy_metadata(seed)

    activity_screen = screen_reserved_control(
        candidate_name=seed.name,
        metadata=policy_metadata,
        reserved_controls=[_control(activity_id)],
    )
    molecule_screen = screen_reserved_control(
        candidate_name=seed.name,
        metadata=policy_metadata,
        reserved_controls=[_control(molecule_id)],
    )

    assert activity_screen.excluded is False
    assert molecule_screen.reason_codes == ("RESERVED_ASSAY_CONTROL",)
    assert candidate_identity(
        candidate_name=seed.name,
        metadata=policy_metadata,
    )["registry_ids"] == [
        {"namespace": "chembl", "identifier": molecule_id.casefold()}
    ]


def test_enriched_rescreen_ignores_activity_assay_document_target_and_context_ids() -> None:
    real_molecule_id = "CHEMBL-MOLECULE-REAL"
    context_ids = (
        "CHEMBL-ACTIVITY-RESERVED",
        "CHEMBL-ASSAY-RESERVED",
        "CHEMBL-DOCUMENT-RESERVED",
        "CHEMBL-TARGET-RESERVED",
        "CHEMBL-ENRICHMENT-RESERVED",
    )
    enriched_metadata = {
        "chembl_id": real_molecule_id,
        "source_evidence": [
            {
                "source": "ChEMBL",
                # MA2 copies EvidenceRef.external_id here. It is not a drug ID.
                "drug_id": context_ids[0],
                "metadata": {
                    "activity_id": context_ids[0],
                    "assay_chembl_id": context_ids[1],
                    "document_chembl_id": context_ids[2],
                    "target_chembl_id": context_ids[3],
                },
            }
        ],
        "restricted_source_records": [
            {"source": "ChEMBL", "drug_id": context_ids[4]}
        ],
        "public_enrichment": {
            "source": "ChEMBL",
            "external_id": context_ids[4],
        },
    }

    for context_id in context_ids:
        screen = screen_reserved_control(
            candidate_name="real candidate",
            metadata=enriched_metadata,
            reserved_controls=[_control(context_id)],
        )
        assert screen.excluded is False, context_id


def test_enriched_rescreen_accepts_only_explicit_nested_molecule_identity() -> None:
    molecule_id = "CHEMBL-MOLECULE-RESERVED"
    metadata = {
        "source_evidence": [
            {
                "source": "ChEMBL",
                "drug_id": "CHEMBL-ACTIVITY-NOT-A-MOLECULE",
                "metadata": {
                    "molecule_chembl_id": molecule_id,
                    "activity_id": "CHEMBL-ACTIVITY-NOT-A-MOLECULE",
                },
            }
        ]
    }

    screen = screen_reserved_control(
        candidate_name="different display name",
        metadata=metadata,
        reserved_controls=[_control(molecule_id)],
    )

    assert screen.reason_codes == ("RESERVED_ASSAY_CONTROL",)
