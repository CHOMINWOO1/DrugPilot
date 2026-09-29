"""Isolated synthetic HTTP/candidate responses, never live source fixtures."""
from __future__ import annotations

import sys

import httpx
import pytest

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma2 import MA2Agent
from drugpilot.agents.planner import PlannerAgent
from drugpilot.config import Settings
from drugpilot.domain.models import ScientificExecutionStatus, TargetCandidate
from drugpilot.tools.executable import ScientificDataUnavailable
from drugpilot.tools.fixture_gateway import DeterministicFixtureGateway
from drugpilot.tools.live_ma2_gateway import LiveScientificGateway, NoMeasuredLigandCandidates
from drugpilot.tools.live_repurposing_gateway import PAIRED_V4_REQUIRED_POLICY, NoRealRepurposingCandidates
from tests.unit.test_live_scientific_tools import _ADMET, _Docking, _LargeBoundedRepurposingClient, _Response
from tests.unit.test_live_repurposing_gateway import FakeScientificDelegate, _gateway, _write_dataset


class PagedMechanisms(_LargeBoundedRepurposingClient):
    def __init__(self, count, *, broken=None):
        super().__init__(raw_count=count)
        self.mechanism_requests = []
        self.broken = broken

    def get(self, url, **kwargs):
        if not url.endswith("mechanism.json"):
            return super().get(url, **kwargs)
        params = dict(kwargs["params"])
        self.mechanism_requests.append(params)
        offset = params.get("offset", 0)
        limit = params["limit"]
        payload = super().get(url, **kwargs).json()
        rows = payload["mechanisms"][offset:offset + limit]
        total = self.raw_count
        if offset and self.broken == "timeout":
            raise httpx.ReadTimeout("isolated later-page timeout")
        if offset and self.broken == "duplicate":
            rows[0] = payload["mechanisms"][0]
        if offset and self.broken == "changed_total":
            total += 1
        return _Response({"mechanisms": rows, "page_meta": {
            "offset": offset, "total_count": total,
            "next": "/mechanism.json?offset=ignored" if offset + len(rows) < total else None,
        }})


def scientific(client, *, policy=None):
    return LiveScientificGateway(docking=_Docking(), admet=_ADMET(), client=client,
        python_executable=sys.executable, max_http_attempts=1, retry_backoff_seconds=0,
        repurposing_evidence_policy=policy)


@pytest.mark.parametrize("pool_limit", [20, 100])
def test_generic_delegate_honors_full_bounded_pool_not_hidden_top12(pool_limit):
    client = PagedMechanisms(pool_limit)
    gateway = scientific(client)
    target = gateway.normalize_target(TargetCandidate(symbol="GENE", uniprot_id="P12345"))
    seeds = gateway.retrieve_repurposing(target, direction="inhibit", limit=pool_limit)
    assert len(seeds) == pool_limit
    assert len(client.activity_parents) == pool_limit and len(set(client.activity_parents)) == pool_limit
    assert all(request["limit"] <= 80 for request in client.mechanism_requests)
    assert [request.get("offset", 0) for request in client.mechanism_requests] == ([0] if pool_limit == 20 else [0, 80])
    provenance = gateway.last_operation_provenance
    assert provenance["fixed_prewarning_pool_limit"] == pool_limit
    assert provenance["gateway_return_candidate_ids"] == client.activity_parents
    assert provenance["gateway_return_cap_applied"] is False
    assert provenance["source_search_complete"] is True
    assert provenance["exhaustive_discovery"] is False
    if pool_limit == 100:
        bounds = provenance["candidate_discovery_bounds"]
        assert bounds["mechanism_page_count"] == 2
        assert bounds["mechanism_provenance_by_id"]["CHEMBL100100"]["query"] == bounds["mechanism_queries"][1]


@pytest.mark.parametrize("failure", ["timeout", "duplicate", "changed_total"])
def test_later_page_failure_is_not_partial_pool_or_exhaustion(failure):
    client = PagedMechanisms(100, broken=failure)
    gateway = scientific(client)
    target = gateway.normalize_target(TargetCandidate(symbol="GENE", uniprot_id="P12345"))
    with pytest.raises(ScientificDataUnavailable):
        gateway.retrieve_repurposing(target, direction="inhibit", limit=100)
    assert not client.activity_parents
    assert gateway.last_operation_provenance["source_search_complete"] is False
    assert gateway.last_operation_provenance["result_returned"] is False


def test_over_bound_mechanism_set_is_explicit_incomplete_not_zero_results():
    client = PagedMechanisms(101)
    gateway = scientific(client)
    target = gateway.normalize_target(TargetCandidate(symbol="GENE", uniprot_id="P12345"))
    with pytest.raises(ScientificDataUnavailable) as error:
        gateway.retrieve_repurposing(target, direction="inhibit", limit=100)
    assert error.value.provenance["reason_code"] == "MECHANISM_IDENTITY_SET_INCOMPLETE_OR_OVER_CAP"
    assert gateway.last_operation_provenance["source_search_complete"] is False
    assert not client.activity_parents


def test_fixed_paper_contract_is_not_silently_expanded():
    gateway = scientific(PagedMechanisms(20), policy=PAIRED_V4_REQUIRED_POLICY)
    with pytest.raises(ValueError, match="must not exceed 12"):
        gateway.retrieve_repurposing(TargetCandidate(symbol="GENE", uniprot_id="P12345"), direction="inhibit", limit=20)


def test_local_survivor_keeps_delegate_failure_incomplete(tmp_path):
    dataset = _write_dataset(tmp_path / "isolated.tsv",
        "test-drug\tDC001\tFLT3\tP36888\tFLT3\t4.2\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCCO\tTEST-KEY\n")
    delegate = FakeScientificDelegate(error=ValueError("repurposing candidate limit must not exceed 12"))
    gateway = _gateway(dataset, delegate=delegate)
    seeds = gateway.retrieve_repurposing(TargetCandidate(symbol="FLT3", uniprot_id="P36888"), direction="inhibit", limit=5)
    assert len(seeds) == 1 and delegate.requests[0][2] == 20
    assert gateway.last_operation_provenance["partial_source_result"] is True
    assert gateway.last_operation_provenance["source_search_complete"] is False
    assert gateway.last_operation_provenance["search_completed"] is False


def test_nested_successful_but_incomplete_delegate_is_not_relabelled_complete(tmp_path):
    dataset = _write_dataset(tmp_path / "isolated.tsv",
        "test-drug\tDC001\tFLT3\tP36888\tFLT3\t4.2\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCCO\tTEST-KEY\n")
    class PartialDelegate(FakeScientificDelegate):
        def retrieve_repurposing(self, target, *, direction, limit):
            rows = super().retrieve_repurposing(target, direction=direction, limit=limit)
            self.last_operation_provenance.update(source_search_complete=False, partial_source_result=True)
            return rows
    gateway = _gateway(dataset, delegate=PartialDelegate())
    gateway.retrieve_repurposing(TargetCandidate(symbol="FLT3", uniprot_id="P36888"), direction="inhibit", limit=5)
    assert not gateway.last_operation_provenance["source_search_complete"]
    assert not gateway.last_operation_provenance["source_failures"]


@pytest.mark.parametrize("local_survivor", [False, True])
def test_completed_chembl_zero_is_not_an_incomplete_source(tmp_path, local_survivor):
    delegate = scientific(PagedMechanisms(0))
    target = delegate.normalize_target(TargetCandidate(symbol="GENE", uniprot_id="P12345"))
    dataset = _write_dataset(tmp_path / "isolated.tsv",
        "test-drug\tDC001\t"
        + ("GENE\tP12345\tGENE" if local_survivor else "OTHER\tP99999\tOTHER")
        + "\t4.2\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCCO\tTEST-KEY\n")
    gateway = _gateway(dataset, delegate=delegate)
    if local_survivor:
        rows = gateway.retrieve_repurposing(target, direction="inhibit", limit=5)
        assert len(rows) == 1
        provenance = gateway.last_operation_provenance
    else:
        with pytest.raises(NoRealRepurposingCandidates) as error:
            gateway.retrieve_repurposing(target, direction="inhibit", limit=5)
        assert error.value.conclusive_empty_search is True
        provenance = error.value.provenance
    assert delegate.last_operation_provenance["status"] == "completed_empty"
    assert delegate.last_operation_provenance["candidate_count"] == 0
    assert provenance["source_search_complete"] is True
    assert provenance["search_completed"] is True
    assert provenance["partial_source_result"] is False
    assert not provenance["source_failures"]
    assert provenance["criteria_rejections"]


def test_nonconclusive_measured_candidate_error_remains_incomplete(monkeypatch):
    gateway = scientific(PagedMechanisms(0))
    error = NoMeasuredLigandCandidates(target_symbol="GENE", target_uniprot_id="P12345",
        direction="inhibit", eligible_count=0, minimum_required=1,
        reason_code="ISOLATED_NONCONCLUSIVE_QUERY")
    error.conclusive = False
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(gateway, "_fetch_measured_ligands", fail)
    with pytest.raises(NoMeasuredLigandCandidates):
        gateway.retrieve_repurposing(TargetCandidate(symbol="GENE", uniprot_id="P12345"),
            direction="inhibit", limit=20)
    assert gateway.last_operation_provenance["source_search_complete"] is False
    assert gateway.last_operation_provenance["partial_source_result"] is True
    assert gateway.last_operation_provenance["status"] == "error"


def test_empty_partial_delegate_never_becomes_conclusive_zero(tmp_path):
    dataset = _write_dataset(tmp_path / "isolated.tsv",
        "test-drug\tDC001\tOTHER\tP99999\tOTHER\t4.2\tnM\tIC50\tCHEMBL\t=\t1\tINHIBITOR\tHomo sapiens\tCCO\tTEST-KEY\n")
    class PartialDelegate(FakeScientificDelegate):
        def retrieve_repurposing(self, target, *, direction, limit):
            rows = super().retrieve_repurposing(target, direction=direction, limit=limit)
            self.last_operation_provenance.update(source_search_complete=False,
                search_completed=False, partial_source_result=True)
            return rows
    gateway = _gateway(dataset, delegate=PartialDelegate())
    with pytest.raises(NoRealRepurposingCandidates) as error:
        gateway.retrieve_repurposing(TargetCandidate(symbol="FLT3", uniprot_id="P36888"),
            direction="inhibit", limit=5)
    assert error.value.conclusive_empty_search is False
    assert error.value.provenance["search_completed"] is False
    assert error.value.provenance["partial_source_result"] is True
    assert not error.value.source_failures
    assert gateway.last_operation_provenance == error.value.provenance


@pytest.mark.parametrize("excluded", [False, True])
def test_ma2_returned_pool_evaluated_does_not_mean_source_exhausted(brief, excluded):
    settings = Settings(_env_file=None, ma1_provisional_admission_enabled=False,
        dynamic_target_discovery_enabled=False, dynamic_target_fallback_enabled=True,
        demo_fixture_fallback_enabled=True)
    class PartialGateway(DeterministicFixtureGateway):
        def retrieve_repurposing(self, target, *, direction, limit):
            rows = super().retrieve_repurposing(target, direction=direction, limit=limit)
            self.last_operation_provenance = {
                "partial_source_result": True, "source_search_complete": False,
                "source_failures": ["delegate:isolated contract failure"],
            }
            return rows
    hypotheses = MA1Agent(settings=settings).run(brief)
    plan = PlannerAgent(settings=settings).run(hypotheses, brief)
    plan.forbidden_drugs = ["RP-FLT3-002", "RP-FLT3-003", *(["RP-FLT3-001"] if excluded else [])]
    packet = MA2Agent(gateway=PartialGateway()).run(plan, hypotheses)
    backfill = packet.metadata["candidate_backfill"]
    assert backfill["returned_pool_evaluated"] is True
    assert backfill["pool_exhausted"] is False
    assert backfill["source_search_complete"] is False
    assert backfill["stop_reason"] == "SOURCE_DISCOVERY_INCOMPLETE"
    assert packet.execution_status is ScientificExecutionStatus.UNAVAILABLE
    assert "CANDIDATE_DISCOVERY_SOURCE_INCOMPLETE" in packet.abstain_reason_codes
    assert packet.metadata["candidate_search_outcome"]["status"] == "incomplete_search"
    assert packet.metadata["candidate_search_outcome"]["search_completed"] is False
