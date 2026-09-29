from __future__ import annotations

import networkx as nx

from drugpilot.domain.models import ReworkOrder


class WorkflowDependencyGraph:
    """Claim/artifact dependency skeleton used for scoped invalidation."""

    def __init__(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("research_brief", "ma1.intake"),
                ("ma1.intake", "ma1.proposer"),
                ("ma1.proposer", "ma1.critic"),
                ("ma1.critic", "ma1.router"),
                ("ma1.router", "ma1.selector"),
                ("ma1.selector", "g0.audit"),
                ("g0.audit", "planner"),
                ("planner", "ma2.standardize"),
                ("ma2.standardize", "ma2.repurpose"),
                ("ma2.standardize", "ma2.de_novo"),
                ("ma2.repurpose", "ma2.activity_normalization"),
                ("ma2.de_novo", "ma2.evaluate"),
                ("ma2.activity_normalization", "ma2.evaluate"),
                ("ma2.evaluate", "ma2.admet"),
                ("ma2.admet", "ma2.judge"),
                ("ma2.judge", "g1.audit"),
                ("g1.audit", "human_gate"),
            ]
        )
        self.graph = graph

    def invalidated_nodes(self, order: ReworkOrder) -> list[str]:
        failed = self._canonical_node(order.failed_node)
        if failed not in self.graph:
            return [failed]
        affected = {failed, *nx.descendants(self.graph, failed)}
        boundary = {
            "ma1": {node for node in affected if node.startswith("ma1.") or node == "g0.audit"},
            "planner": {node for node in affected if node in {"planner"} or node.startswith("ma2.") or node == "g1.audit"},
            "ma2": {node for node in affected if node.startswith("ma2.") or node == "g1.audit"},
        }[order.target_stage]
        return list(nx.topological_sort(self.graph.subgraph(boundary)))

    def reusable_nodes(self, order: ReworkOrder) -> list[str]:
        invalidated = set(self.invalidated_nodes(order))
        return [node for node in nx.topological_sort(self.graph) if node not in invalidated]

    def is_valid_transition(self, source: str, target: str) -> bool:
        return self.graph.has_edge(self._canonical_node(source), self._canonical_node(target))

    @staticmethod
    def _canonical_node(node: str) -> str:
        aliases = {
            "ma1": "ma1.proposer",
            "ma2": "ma2.evaluate",
            "g0": "g0.audit",
            "g1": "g1.audit",
        }
        return aliases.get(node, node)

