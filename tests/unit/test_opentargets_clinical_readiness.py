"""Schema-drift tests: clinical metadata schedules research, never proves efficacy."""
import pytest

from drugpilot.agents.ma1 import _discovery_intervention_readiness
from drugpilot.tools.disease_targets import DiseaseTargetDiscovery
from test_disease_targets import FakeClient, _row, _association_payload, _disease_search_payload


def discover(scores):
    row=_row("ENSG00000000001","TESTGENE",0.7,datatype_scores=scores)
    client=FakeClient(post_handler=lambda url,kwargs:
        _disease_search_payload() if "SearchDisease" in kwargs["json"]["query"] else _association_payload([row]),
        get_handler=lambda *a,**k:pytest.fail("No PubMed calls in a schema-only test"))
    return DiseaseTargetDiscovery(client=client,max_attempts=1).discover(
        "lung adenocarcinoma",allowed_sources=["Open Targets"]).candidates[0]


def test_current_clinical_datatype_is_retained_separately_and_schedules_only():
    c=discover([{"id":"clinical","score":0.8}])
    assert c.evidence_summary.clinical_evidence_score==0.8
    assert c.evidence_summary.known_drug_score is None
    assert _discovery_intervention_readiness(c,"inhibit")== (1,["open_targets_clinical_signal"])
    assert _discovery_intervention_readiness(c,"activate")[0]==0
    assert c.provenance.causality_interpretation=="association_score_is_not_causal_proof"
    assert c.source_records[0].relationship=="ranked_association"


@pytest.mark.parametrize("scores,expected", [([],0),([{"id":"clinical","score":0.0}],0),
    ([{"id":"unknown_new_type","score":1.0}],0),([{"id":"known_drug","score":0.6}],1),
    ([{"id":"known_drug","score":0.6},{"id":"clinical","score":0.8}],1)])
def test_missing_zero_unknown_and_legacy_types_do_not_inflate_readiness(scores,expected):
    c=discover(scores)
    assert _discovery_intervention_readiness(c,"inhibit")[0]==expected


def test_discovery_adapter_is_in_deployment_source_fingerprint(monkeypatch):
    from pathlib import Path
    from drugpilot.orchestration.human_review import source_digest
    original=Path.read_bytes
    before=source_digest()
    monkeypatch.setattr(Path,"read_bytes",lambda p:
        original(p)+b"\n# isolated checksum probe" if p.name=="disease_targets.py" else original(p))
    assert source_digest()!=before
