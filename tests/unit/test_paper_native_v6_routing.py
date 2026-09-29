from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.ma3 import _paper_v6_assay_engagement
from drugpilot.api.app import CreateRunRequest
from drugpilot.domain.models import (
    NamespacedRegistryIdentifier,
    ReservedControlCompound,
    ScientificExecutionStatus,
)
from drugpilot.orchestration.graph import (
    _repurposing_evidence_policy_for_priorities,
)
from drugpilot.tools.live_ma2_gateway import (
    _COMPLETE_ACTIVITY_PAGINATION_POLICIES,
)
from drugpilot.tools.live_repurposing_gateway import (
    PAIRED_V4_REQUIRED_POLICY,
    PAPER_NATIVE_V5_REQUIRED_POLICY,
    PAPER_NATIVE_V6_REQUIRED_POLICY,
    _paper_native_v6_assay_engagement,
)
from test_live_repurposing_gateway_paper_native_v6 import (
    IDS,
    _gateway as _v6_wrapper,
    _retrieve as _v6_retrieve,
    _seed as _v6_seed,
)
from test_ma2_live_execution_contract import (
    PartialLiveGateway,
    StubRDKitChemistry,
    _pipeline,
)


ROOT = Path(__file__).resolve().parents[2]
REQUEST = (
    ROOT
    / "docs/validation/FLT3_ITD_AML_R18_LIVE_E2E_REQUEST_R2_2026-09-01.json"
)


def test_v6_wrapper_and_ma3_independently_accept_exact_recombinant_assay() -> None:
    description = (
        "Inhibition of recombinant N-terminal GST-tagged FLT3 kinase using "
        "poly(Glu,Tyr) as substrate in presence of ATP"
    )
    context = {
        "schema_version": "chembl_assay_context_v1",
        "retrieval_status": "completed",
        "binding_status": "exact",
        "assay_chembl_id": "CHEMBL5708845",
        "missing_values_imputed": False,
        "record": {
            "assay_chembl_id": "CHEMBL5708845",
            "target_chembl_id": "CHEMBL1974",
            "assay_type": "B",
            "assay_organism": "Homo sapiens",
            "relationship_type": "D",
            "confidence_score": 9,
            "assay_cell_type": None,
            "assay_tissue": None,
            "assay_strain": None,
            "cell_chembl_id": None,
            "tissue_chembl_id": None,
            "description": {
                "excerpt": description,
                "excerpt_truncated": False,
                "content_length": len(description),
                "content_sha256": sha256(description.encode()).hexdigest(),
            },
        },
    }
    kwargs = {
        "assay_type": "assay format",
        "chembl_assay_context": context,
        "assay_chembl_id": "CHEMBL5708845",
        "target_chembl_id": "CHEMBL1974",
        "target_symbol": "FLT3",
    }

    assert _paper_native_v6_assay_engagement(**kwargs) == (
        "direct_single_protein_context",
        True,
    )
    assert _paper_v6_assay_engagement(**kwargs) == (
        "direct_single_protein_context",
        True,
    )

    wrong_target = {**kwargs, "target_chembl_id": "CHEMBL_OTHER"}
    assert _paper_native_v6_assay_engagement(**wrong_target)[1] is False
    assert _paper_v6_assay_engagement(**wrong_target)[1] is False


def test_v6_policy_dispatch_is_exact_and_conflicts_fail_closed() -> None:
    assert (
        _repurposing_evidence_policy_for_priorities(
            ["unrelated", PAPER_NATIVE_V6_REQUIRED_POLICY]
        )
        == PAPER_NATIVE_V6_REQUIRED_POLICY
    )
    assert (
        _repurposing_evidence_policy_for_priorities(
            [f"prefix:{PAPER_NATIVE_V6_REQUIRED_POLICY}"]
        )
        is None
    )
    for conflicting in (
        PAPER_NATIVE_V5_REQUIRED_POLICY,
        PAIRED_V4_REQUIRED_POLICY,
    ):
        with pytest.raises(ValueError, match="Conflicting exact"):
            _repurposing_evidence_policy_for_priorities(
                [PAPER_NATIVE_V6_REQUIRED_POLICY, conflicting]
            )


def test_v6_uses_complete_activity_pagination_contract() -> None:
    assert PAPER_NATIVE_V6_REQUIRED_POLICY in _COMPLETE_ACTIVITY_PAGINATION_POLICIES
    assert PAPER_NATIVE_V5_REQUIRED_POLICY in _COMPLETE_ACTIVITY_PAGINATION_POLICIES


def test_r2_request_is_schema_valid_and_selects_v6_once() -> None:
    raw = json.loads(REQUEST.read_text(encoding="utf-8"))
    parsed = CreateRunRequest.model_validate(raw)
    assert parsed.candidates_per_route == 3
    assert raw["human_priorities"].count(PAPER_NATIVE_V6_REQUIRED_POLICY) == 1
    assert PAPER_NATIVE_V5_REQUIRED_POLICY not in raw["human_priorities"]
    assert PAIRED_V4_REQUIRED_POLICY not in raw["human_priorities"]
    assert (
        _repurposing_evidence_policy_for_priorities(raw["human_priorities"])
        == PAPER_NATIVE_V6_REQUIRED_POLICY
    )


class _V6MA2Gateway(PartialLiveGateway):
    repurposing_evidence_policy = PAPER_NATIVE_V6_REQUIRED_POLICY

    def __init__(self, seeds) -> None:
        super().__init__(target_linked_activity=True)
        self.seeds = tuple(seeds)
        self.admet_calls = 0

    def retrieve_repurposing(self, target, *, direction, limit):
        self.last_operation_provenance = {
            "requested_eligible_count": limit,
            "discovery_pool_count": len(self.seeds),
            "bounded_discovery": True,
            "fixture": False,
        }
        return self.seeds

    def predict_admet(self, *, smiles):
        self.admet_calls += 1
        return super().predict_admet(smiles=smiles)


def _contracted_v6_seeds(tmp_path):
    rows = _v6_retrieve(
        _v6_wrapper(
            tmp_path,
            [_v6_seed(candidate_id, index) for index, candidate_id in enumerate(IDS)],
        )
    )
    return tuple(
        replace(
            seed,
            evidence=tuple(
                item.model_copy(
                    update={
                        "metadata": {
                            **item.metadata,
                            "target_gene": "FLT3",
                            "target_type": "SINGLE PROTEIN",
                            "target_organism": "Homo sapiens",
                            "direct_single_protein_engagement": True,
                        }
                    },
                    deep=True,
                )
                for item in seed.evidence
            ),
        )
        for seed in rows
    )


def test_ma2_v6_stops_at_requested_eligible_prefix_not_full_upstream_group(
    tmp_path,
    brief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(update={"candidates_per_route": 3}, deep=True)
    gateway = _V6MA2Gateway(_contracted_v6_seeds(tmp_path))

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    assert packet.execution_status is ScientificExecutionStatus.COMPLETED
    assert len(packet.candidates) == 3
    assert gateway.admet_calls == 3
    projection = packet.metadata["candidate_backfill"][
        "paper_native_v6_group_projection"
    ]
    assert projection["declared_group_candidate_ids"] == list(IDS)
    assert projection["projected_group_candidate_ids"] == list(IDS[:3])
    assert projection["unevaluated_declared_group_candidate_ids"] == [IDS[3]]
    assert projection["projected_is_exact_declared_prefix"] is True
    assert projection["complete_declared_group_projected"] is False
    assert projection["projection_complete"] is True
    assert projection["downstream_evaluation_cap_enforced"] is True
    assert packet.metadata["candidate_backfill"]["stop_reason"] == (
        "ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX"
    )


def test_ma2_v6_reserved_control_is_projected_without_downstream_compute(
    tmp_path,
    brief,
) -> None:
    hypotheses, plan = _pipeline(tmp_path, brief)
    plan = plan.model_copy(
        update={
            "candidates_per_route": 3,
            "reserved_control_compounds": [
                ReservedControlCompound(
                    name="Drug 0",
                    registry_ids=[
                        NamespacedRegistryIdentifier(
                            namespace="ChEMBL",
                            identifier=IDS[0],
                        )
                    ],
                )
            ],
        },
        deep=True,
    )
    gateway = _V6MA2Gateway(_contracted_v6_seeds(tmp_path))

    packet = MA2Agent(
        gateway=gateway,
        chemistry=StubRDKitChemistry(),
    ).run(plan, hypotheses)

    assert len(packet.candidates) == 4
    assert len([item for item in packet.candidates if item.status != "excluded"]) == 3
    assert gateway.admet_calls == 3
    reserved = next(item for item in packet.candidates if item.metadata["chembl_id"] == IDS[0])
    assert reserved.status == "excluded"
    assert reserved.chemistry_execution is None
    projection = packet.metadata["candidate_backfill"][
        "paper_native_v6_group_projection"
    ]
    assert projection["projected_group_candidate_ids"] == list(IDS)
    assert projection["eligible_returned_count"] == 3
    assert projection["complete_declared_group_projected"] is True
