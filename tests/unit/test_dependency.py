from drugpilot.domain.models import GateName, ReworkOrder
from drugpilot.orchestration.dependency import WorkflowDependencyGraph


def test_scoped_invalidation_preserves_unrelated_retrieval():
    graph = WorkflowDependencyGraph()
    order = ReworkOrder(
        gate=GateName.G1,
        target_stage="ma2",
        failed_node="ma2.activity_normalization",
        required_actions=["Correct IC50 unit"],
    )

    invalidated = graph.invalidated_nodes(order)
    reusable = graph.reusable_nodes(order)

    assert "ma2.activity_normalization" in invalidated
    assert "ma2.evaluate" in invalidated
    assert "ma2.repurpose" in reusable
    assert "ma1.proposer" in reusable

