from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from drugpilot.domain.models import (
    ADMETExecutionResult,
    ActivityMeasurement,
    CandidateRoute,
    DockingExecutionResult,
    EvidenceRef,
    GatewayReadiness,
    TargetCandidate,
)


@dataclass(frozen=True, slots=True)
class CandidateSeed:
    """Tool-neutral candidate returned by retrieval or generation backends."""

    name: str
    smiles: str
    mechanism: str
    activities: tuple[ActivityMeasurement, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class CandidateToolGateway(Protocol):
    """Boundary between MA2 control logic and scientific data/model tools.

    A live implementation may wrap REST APIs, MCP servers, local models, or
    command-line tools.  The mock implementation deliberately provides the
    same contract without making network calls.
    """

    source_version: str
    backend_kind: str
    fixture: bool

    def readiness(self, target: TargetCandidate | None = None) -> GatewayReadiness: ...

    def normalize_target(self, target: TargetCandidate) -> TargetCandidate: ...

    def retrieve_repurposing(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]:
        """Return an ordered, finite discovery pool.

        ``limit`` is the requested eligible-candidate quota.  An
        implementation may return a bounded reserve pool larger than that
        quota so MA2 can replace explicit exclusions, but must never perform
        an unbounded follow-up search.  MA2 stops evaluation as soon as the
        requested eligible quota is met.
        """
        ...

    def generate_de_novo(
        self,
        target: TargetCandidate,
        *,
        direction: str,
        limit: int,
    ) -> Sequence[CandidateSeed]: ...

    def dock(
        self,
        *,
        smiles: str,
        target: TargetCandidate,
    ) -> float | DockingExecutionResult: ...

    def predict_admet(
        self,
        *,
        smiles: str,
    ) -> Mapping[str, float] | ADMETExecutionResult: ...

    def refine(
        self,
        seed: CandidateSeed,
        *,
        reasons: Sequence[str],
        iteration: int,
    ) -> CandidateSeed: ...

    def restore_activity(
        self,
        *,
        candidate_name: str,
        target: TargetCandidate,
    ) -> Sequence[ActivityMeasurement]: ...

    def restore_structure(
        self,
        *,
        candidate_name: str,
        route: CandidateRoute,
        target: TargetCandidate,
        direction: str,
    ) -> CandidateSeed: ...


class JsonTransport(Protocol):
    """Small HTTP seam used by future live database adapters."""

    def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str | int | float] | None = None,
    ) -> Any: ...
