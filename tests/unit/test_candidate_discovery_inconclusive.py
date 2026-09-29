"""Capability/partial errors must never become completed scientific negatives."""
from types import SimpleNamespace

import pytest

from drugpilot.agents.ma2 import MA2Agent
from drugpilot.domain.models import TargetCandidate
from drugpilot.tools.executable import ScientificToolUnavailable
from drugpilot.tools.live_ma2_gateway import NoMeasuredLigandCandidates


@pytest.mark.parametrize("method", ["_retrieve_repurposing_outcome", "_generate_de_novo_outcome"])
@pytest.mark.parametrize("kind", ["capability", "partial"])
def test_inconclusive_discovery_errors_remain_errors(method, kind):
    if kind == "capability":
        error = ScientificToolUnavailable("Unsupported direction; no query performed")
    else:
        error = NoMeasuredLigandCandidates(target_symbol="FLT3", target_uniprot_id=None,
            direction="activate", eligible_count=0, minimum_required=1,
            reason_code="PARTIAL_QUERY_NOT_CONCLUSIVE")
        error.conclusive = False

    def fail(*args, **kwargs):
        raise error

    agent = MA2Agent()
    agent.gateway = SimpleNamespace(retrieve_repurposing=fail, generate_de_novo=fail)
    with pytest.raises(type(error)) as captured:
        getattr(agent, method)(TargetCandidate(symbol="FLT3"), direction="activate", limit=2)
    assert captured.value is error


@pytest.mark.parametrize("method", ["_retrieve_repurposing_outcome", "_generate_de_novo_outcome"])
def test_completed_negative_query_remains_distinct(method):
    def fail(*args, **kwargs):
        raise NoMeasuredLigandCandidates(target_symbol="FLT3", target_uniprot_id=None,
            direction="inhibit", eligible_count=0, minimum_required=1,
            reason_code="NO_DIRECTION_COMPATIBLE_MEASURED_LIGANDS",
            provenance={"query_completed": True})
    agent = MA2Agent()
    agent.gateway = SimpleNamespace(retrieve_repurposing=fail, generate_de_novo=fail)
    outcome = getattr(agent, method)(TargetCandidate(symbol="FLT3"), direction="inhibit", limit=2)
    assert outcome.outcome == "no_candidates"
    assert outcome.provenance["search_completed"] is True
