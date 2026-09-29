"""Bounded, context-only routing for disease/target enrichment.

The router is an integration boundary around
:class:`drugpilot.tools.public_context.MA1PublicContextRetriever`.  It does not
modify an MA1 hypothesis packet or attach records to causal claims.  Instead it
returns a separate context bundle whose records must retain the adapter's
``supports="context"`` and context-only provenance flags.

Routing is track- and brief-aware:

* literature targets use bioRxiv, medRxiv, and ClinicalTrials.gov;
* Tbio targets use expression and network context in addition to GTEx,
  STRING, Reactome, KEGG, and QuickGO when a UniProt accession is available;
* Human Cell Atlas and CELLxGENE Census are admitted only for an explicit
  cell/single-cell context in the research brief;
* ARCHS4 and BioGRID are admitted for Tbio targets or an explicit expression
  or network priority respectively.

Only the highest-ranked targets in each track are selected.  Each source is
looked up separately so call and elapsed-time budgets can stop additional
remote work at a source boundary and report which planned calls were skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import Field, model_validator

from drugpilot.domain.models import (
    EvidenceRef,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    StrictModel,
)
from drugpilot.domain.cell_context import classify_cell_context
from drugpilot.domain.goal_intent import has_affirmative_term, has_negated_term
from drugpilot.tools.public_context import (
    ContextSource,
    ContextSourceStatus,
    MA1ContextLookup,
    MA1ContextResult,
    MA1PublicContextRetriever,
)


ContextFamily = Literal[
    "preprint",
    "clinical",
    "expression",
    "network",
    "pathway",
    "ontology",
    "single_cell",
]
RoutedSourceStatus = Literal[
    "ok",
    "empty",
    "error",
    "skipped",
    "skipped_missing_identifier",
    "skipped_not_applicable",
    "skipped_call_budget",
    "skipped_time_budget",
    "policy_rejected",
    "not_configured",
]
TargetSelectionStatus = Literal["selected", "outside_top_target_limit", "duplicate"]
BudgetStopReason = Literal["call_budget", "time_budget"]


LITERATURE_SOURCE_PLAN: tuple[tuple[ContextSource, ContextFamily], ...] = (
    ("bioRxiv", "preprint"),
    ("Human Cell Atlas", "single_cell"),
    ("CELLxGENE Census", "single_cell"),
    ("medRxiv", "preprint"),
    ("ClinicalTrials.gov", "clinical"),
    ("ARCHS4", "expression"),
    ("BioGRID", "network"),
)
TBIO_SOURCE_PLAN: tuple[tuple[ContextSource, ContextFamily], ...] = (
    ("GTEx", "expression"),
    ("ARCHS4", "expression"),
    ("STRING", "network"),
    ("BioGRID", "network"),
    ("Reactome", "pathway"),
    ("KEGG", "pathway"),
    ("QuickGO", "ontology"),
    ("Human Cell Atlas", "single_cell"),
    ("CELLxGENE Census", "single_cell"),
)


class ContextTarget(StrictModel):
    """Minimal target identity required by the on-demand context adapters.

    ``track`` is the immutable MA1 hypothesis/evidence-supply track.
    ``effective_work_track`` is a separate deterministic allocation used only
    to choose the context source plan after authoritative TDL reconciliation.
    """

    symbol: str = Field(min_length=1, max_length=80)
    track: HypothesisTrack
    effective_work_track: HypothesisTrack | None = None
    work_track_basis: Literal[
        "effective_tdl", "hypothesis_track_fallback"
    ] = "hypothesis_track_fallback"
    hypothesis_id: str | None = Field(default=None, max_length=100)
    rank: int | None = Field(default=None, ge=1)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    uniprot_id: str | None = Field(default=None, max_length=30)
    gencode_id: str | None = Field(default=None, max_length=50)

    @model_validator(mode="after")
    def normalize_identifiers(self) -> "ContextTarget":
        object.__setattr__(self, "symbol", self.symbol.strip())
        for field_name in ("hypothesis_id", "uniprot_id", "gencode_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, value.strip() or None)
        return self


class ContextRoutingBudget(StrictModel):
    """Hard admission bounds for one routing operation.

    ``max_elapsed_seconds`` is enforced between individual source calls.  The
    in-flight synchronous call is allowed to finish (and remains subject to the
    retriever's own HTTP timeout); no additional call is admitted after the
    deadline.
    """

    max_targets_per_track: int = Field(default=1, ge=1, le=10)
    max_source_calls: int = Field(default=8, ge=1, le=100)
    max_elapsed_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    limit_per_source: int = Field(default=2, ge=1, le=10)


class ContextRoutingRequest(StrictModel):
    disease: str = Field(min_length=2, max_length=300)
    targets: list[ContextTarget]
    target_cell_type: str | None = Field(default=None, max_length=200)
    human_priorities: list[str] = Field(default_factory=list)
    # ``None`` preserves the standalone router's historical all-source
    # behavior.  An orchestrated ResearchBrief always supplies its immutable
    # allow-list here.
    allowed_sources: list[str] | None = None
    budget: ContextRoutingBudget = Field(default_factory=ContextRoutingBudget)

    @model_validator(mode="after")
    def normalize_request(self) -> "ContextRoutingRequest":
        object.__setattr__(self, "disease", self.disease.strip())
        if self.target_cell_type is not None:
            object.__setattr__(
                self,
                "target_cell_type",
                self.target_cell_type.strip() or None,
            )
        object.__setattr__(
            self,
            "human_priorities",
            [value.strip() for value in self.human_priorities if value.strip()],
        )
        if self.allowed_sources is not None:
            object.__setattr__(
                self,
                "allowed_sources",
                list(
                    dict.fromkeys(
                        value.strip()
                        for value in self.allowed_sources
                        if value.strip()
                    )
                ),
            )
        return self


class ContextTargetSelection(StrictModel):
    target: ContextTarget
    status: TargetSelectionStatus
    detail: str


class ContextSourceOutcome(StrictModel):
    """Source status for one selected target and one routed source."""

    source: ContextSource
    family: ContextFamily
    status: RoutedSourceStatus
    queried: bool
    adapter_status: ContextSourceStatus | None = None
    call_index: int | None = Field(default=None, ge=1)
    cache_hit: bool = False
    record_count: int = Field(default=0, ge=0)
    rejected_record_count: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    error_type: str | None = None
    detail: str | None = None


class ContextEvidenceProvenance(StrictModel):
    """Small audit projection of an accepted context record's provenance."""

    evidence_id: str
    source: str
    external_id: str
    url: str | None = None
    provenance_class: str
    retrieval_adapter: str
    retrieval_endpoint: str
    cache_hit: bool
    supports: Literal["context"] = "context"
    context_only: Literal[True] = True
    causal_inference_allowed: Literal[False] = False
    counter_evidence_allowed: Literal[False] = False


class TargetContextBundle(StrictModel):
    target: ContextTarget
    sources: list[ContextSourceOutcome] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    provenance: list[ContextEvidenceProvenance] = Field(default_factory=list)


class ContextRoutingResult(StrictModel):
    disease: str
    budget: ContextRoutingBudget
    target_selection: list[ContextTargetSelection]
    targets: list[TargetContextBundle]
    source_calls_used: int = Field(ge=0)
    planned_source_calls: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    budget_exhausted: bool
    stop_reason: BudgetStopReason | None = None
    safety_scope: Literal["context_only"] = "context_only"
    safety_note: str = (
        "Returned records are contextual annotations only and must not be linked "
        "as causal support or counter-evidence."
    )


@runtime_checkable
class ContextLookup(Protocol):
    """Structural seam implemented by ``MA1PublicContextRetriever``."""

    def lookup(self, request: MA1ContextLookup) -> MA1ContextResult: ...


@runtime_checkable
class DiseaseContextRoutingGateway(Protocol):
    """Integration interface for a future graph or agent wiring step."""

    def route(self, request: ContextRoutingRequest) -> ContextRoutingResult: ...


@dataclass(frozen=True, slots=True)
class _SelectedTarget:
    request_index: int
    target: ContextTarget


@dataclass(frozen=True, slots=True)
class _PlannedCall:
    selected: _SelectedTarget
    source: ContextSource
    family: ContextFamily
    selection_reason: str


class DiseaseTargetContextRouter:
    """Select top targets and retrieve bounded, non-causal context for them."""

    def __init__(
        self,
        retriever: ContextLookup | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.retriever = retriever or MA1PublicContextRetriever()
        self._clock = clock

    def route(self, request: ContextRoutingRequest) -> ContextRoutingResult:
        selected, selections = _select_top_targets(
            request.targets,
            request.budget.max_targets_per_track,
        )
        outcome_lists: dict[int, list[ContextSourceOutcome]] = {
            item.request_index: [] for item in selected
        }
        evidence_lists: dict[int, list[EvidenceRef]] = {
            item.request_index: [] for item in selected
        }
        provenance_lists: dict[int, list[ContextEvidenceProvenance]] = {
            item.request_index: [] for item in selected
        }

        planned_by_target: dict[int, list[_PlannedCall]] = {}
        for item in selected:
            calls, immediate = _source_plan(item, request)
            planned_by_target[item.request_index] = calls
            outcome_lists[item.request_index].extend(immediate)

        plan = _round_robin_plan(selected, planned_by_target)
        started_at = self._clock()
        calls_used = 0
        stop_reason: BudgetStopReason | None = None

        for plan_index, planned in enumerate(plan):
            elapsed_before = _elapsed(self._clock(), started_at)
            if calls_used >= request.budget.max_source_calls:
                stop_reason = "call_budget"
                _record_budget_skips(
                    plan[plan_index:], outcome_lists, "skipped_call_budget"
                )
                break
            if elapsed_before >= request.budget.max_elapsed_seconds:
                stop_reason = "time_budget"
                _record_budget_skips(
                    plan[plan_index:], outcome_lists, "skipped_time_budget"
                )
                break

            calls_used += 1
            call_started = self._clock()
            lookup = MA1ContextLookup(
                disease=request.disease,
                gene_symbol=planned.selected.target.symbol,
                uniprot_id=planned.selected.target.uniprot_id,
                gencode_id=planned.selected.target.gencode_id,
                cell_type=request.target_cell_type,
                limit_per_source=request.budget.limit_per_source,
                sources=[planned.source],
            )
            try:
                result = self.retriever.lookup(lookup)
            except Exception as exc:  # defensive boundary for alternate retrievers
                outcome_lists[planned.selected.request_index].append(
                    ContextSourceOutcome(
                        source=planned.source,
                        family=planned.family,
                        status="error",
                        queried=True,
                        call_index=calls_used,
                        elapsed_seconds=_elapsed(self._clock(), call_started),
                        error_type=type(exc).__name__,
                        detail=_join_details(
                            planned.selection_reason,
                            "retriever lookup failed before returning a source result",
                        ),
                    )
                )
                continue

            summary = next(
                (item for item in result.sources if item.source == planned.source),
                None,
            )
            if summary is None:
                outcome_lists[planned.selected.request_index].append(
                    ContextSourceOutcome(
                        source=planned.source,
                        family=planned.family,
                        status="error",
                        queried=True,
                        call_index=calls_used,
                        elapsed_seconds=_elapsed(self._clock(), call_started),
                        error_type="MissingSourceSummary",
                        detail=_join_details(
                            planned.selection_reason,
                            "retriever did not return the requested source status",
                        ),
                    )
                )
                continue

            source_records = [
                record for record in result.evidence if record.source == planned.source
            ]
            accepted_records = [
                record.model_copy(deep=True)
                for record in source_records
                if _is_context_only(record)
            ]
            rejected_count = len(source_records) - len(accepted_records)
            status: RoutedSourceStatus = summary.status
            detail = _join_details(planned.selection_reason, summary.detail)
            if rejected_count:
                status = "policy_rejected"
                detail = _join_details(
                    planned.selection_reason,
                    (
                        f"{rejected_count} record(s) quarantined because context-only "
                        "provenance was incomplete or unsafe"
                    ),
                )

            target_index = planned.selected.request_index
            evidence_lists[target_index].extend(accepted_records)
            provenance_lists[target_index].extend(
                _provenance(record) for record in accepted_records
            )
            outcome_lists[target_index].append(
                ContextSourceOutcome(
                    source=planned.source,
                    family=planned.family,
                    status=status,
                    queried=summary.queried,
                    adapter_status=summary.status,
                    call_index=calls_used,
                    cache_hit=summary.cache_hit,
                    record_count=len(accepted_records),
                    rejected_record_count=rejected_count,
                    elapsed_seconds=_elapsed(self._clock(), call_started),
                    error_type=summary.error_type,
                    detail=detail,
                )
            )

        finished_elapsed = _elapsed(self._clock(), started_at)
        if stop_reason is None and finished_elapsed >= request.budget.max_elapsed_seconds:
            # The final admitted synchronous call may finish at or just after the
            # deadline.  There are no remaining calls to skip, but the audit
            # result still records that the time budget was consumed.
            stop_reason = "time_budget"

        targets = [
            TargetContextBundle(
                target=item.target.model_copy(deep=True),
                sources=_ordered_outcomes(
                    _routing_track(item.target),
                    outcome_lists[item.request_index],
                ),
                evidence=evidence_lists[item.request_index],
                provenance=provenance_lists[item.request_index],
            )
            for item in selected
        ]
        planned_count = len(plan)
        return ContextRoutingResult(
            disease=request.disease,
            budget=request.budget.model_copy(deep=True),
            target_selection=selections,
            targets=targets,
            source_calls_used=calls_used,
            planned_source_calls=planned_count,
            elapsed_seconds=finished_elapsed,
            budget_exhausted=stop_reason is not None,
            stop_reason=stop_reason,
        )

    def route_hypothesis_packet(
        self,
        packet: HypothesisPacket,
        *,
        budget: ContextRoutingBudget | None = None,
        gencode_ids_by_symbol: Mapping[str, str] | None = None,
        target_cell_type: str | None = None,
        human_priorities: Sequence[str] | None = None,
        allowed_sources: Sequence[str] | None = None,
    ) -> ContextRoutingResult:
        """Convenience adapter for MA1 output without mutating the packet."""

        request = context_request_from_hypothesis_packet(
            packet,
            budget=budget,
            gencode_ids_by_symbol=gencode_ids_by_symbol,
            target_cell_type=target_cell_type,
            human_priorities=human_priorities,
            allowed_sources=allowed_sources,
        )
        return self.route(request)


def context_request_from_hypothesis_packet(
    packet: HypothesisPacket,
    *,
    budget: ContextRoutingBudget | None = None,
    gencode_ids_by_symbol: Mapping[str, str] | None = None,
    target_cell_type: str | None = None,
    human_priorities: Sequence[str] | None = None,
    allowed_sources: Sequence[str] | None = None,
) -> ContextRoutingRequest:
    """Project a hypothesis packet onto the router's explicit input contract."""

    if not packet.hypotheses:
        raise ValueError("cannot infer disease from an empty hypothesis packet")
    diseases = {item.disease.strip() for item in packet.hypotheses}
    if len(diseases) != 1:
        raise ValueError("all hypotheses in a context-routing packet must share a disease")
    normalized_gencode = {
        symbol.casefold(): identifier
        for symbol, identifier in (gencode_ids_by_symbol or {}).items()
    }
    targets = [
        _target_from_hypothesis(item, normalized_gencode)
        for item in packet.hypotheses
    ]
    return ContextRoutingRequest(
        disease=next(iter(diseases)),
        targets=targets,
        target_cell_type=target_cell_type,
        human_priorities=list(human_priorities or []),
        allowed_sources=(
            list(allowed_sources) if allowed_sources is not None else None
        ),
        budget=budget or ContextRoutingBudget(),
    )


def _target_from_hypothesis(
    hypothesis: Hypothesis,
    gencode_ids_by_symbol: Mapping[str, str],
) -> ContextTarget:
    effective_work_track, work_track_basis = _context_work_track(hypothesis)
    return ContextTarget(
        symbol=hypothesis.target.symbol,
        track=hypothesis.track,
        effective_work_track=effective_work_track,
        work_track_basis=work_track_basis,
        hypothesis_id=hypothesis.hypothesis_id,
        rank=hypothesis.rank,
        score=hypothesis.score,
        uniprot_id=hypothesis.target.uniprot_id,
        gencode_id=gencode_ids_by_symbol.get(hypothesis.target.symbol.casefold()),
    )


def _context_work_track(
    hypothesis: Hypothesis,
) -> tuple[HypothesisTrack, Literal["effective_tdl", "hypothesis_track_fallback"]]:
    """Project authoritative scientific TDL into a separate work namespace."""

    metadata = hypothesis.target.metadata
    contract = metadata.get("track_tdl_contract")
    if isinstance(contract, Mapping):
        value = str(contract.get("effective_work_track") or "").strip()
        basis = str(contract.get("effective_work_track_basis") or "").strip()
        if (
            contract.get("namespaces_distinct") is True
            and basis == "effective_tdl"
            and value in {item.value for item in HypothesisTrack}
        ):
            return HypothesisTrack(value), "effective_tdl"

    reconciliation = metadata.get("tdl_reconciliation")
    if isinstance(reconciliation, Mapping) and str(
        reconciliation.get("status") or ""
    ) in {"resolved", "resolved_with_source_conflict"}:
        effective_tdl = str(
            reconciliation.get("effective_tdl")
            or metadata.get("effective_tdl")
            or ""
        ).strip().casefold()
        if effective_tdl in {"tclin", "tchem"}:
            return HypothesisTrack.LITERATURE, "effective_tdl"
        if effective_tdl in {"tbio", "tdark"}:
            return HypothesisTrack.TBIO, "effective_tdl"

    return hypothesis.track, "hypothesis_track_fallback"


def _routing_track(target: ContextTarget) -> HypothesisTrack:
    return target.effective_work_track or target.track


def _select_top_targets(
    targets: Sequence[ContextTarget],
    max_per_track: int,
) -> tuple[list[_SelectedTarget], list[ContextTargetSelection]]:
    selected_indices: set[int] = set()
    duplicate_indices: set[int] = set()

    for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO):
        track_items = [
            (index, target)
            for index, target in enumerate(targets)
            if _routing_track(target) is track
        ]
        track_items.sort(key=lambda item: _target_sort_key(item[0], item[1]))
        seen_symbols: set[str] = set()
        unique_items: list[tuple[int, ContextTarget]] = []
        for index, target in track_items:
            symbol = target.symbol.casefold()
            if symbol in seen_symbols:
                duplicate_indices.add(index)
                continue
            seen_symbols.add(symbol)
            unique_items.append((index, target))
        selected_indices.update(index for index, _ in unique_items[:max_per_track])

    selected = [
        _SelectedTarget(request_index=index, target=target.model_copy(deep=True))
        for index, target in enumerate(targets)
        if index in selected_indices
    ]
    selected.sort(
        key=lambda item: (
            0 if _routing_track(item.target) is HypothesisTrack.LITERATURE else 1,
            _target_sort_key(item.request_index, item.target),
        )
    )
    selections: list[ContextTargetSelection] = []
    for index, target in enumerate(targets):
        if index in selected_indices:
            status: TargetSelectionStatus = "selected"
            detail = (
                "selected as a top-ranked target in its effective work-allocation "
                "track; the original hypothesis track remains provenance only"
            )
        elif index in duplicate_indices:
            status = "duplicate"
            detail = "a higher-priority target with the same symbol and track was selected"
        else:
            status = "outside_top_target_limit"
            detail = "not queried because it falls outside max_targets_per_track"
        selections.append(
            ContextTargetSelection(
                target=target.model_copy(deep=True), status=status, detail=detail
            )
        )
    return selected, selections


def _target_sort_key(index: int, target: ContextTarget) -> tuple[object, ...]:
    return (
        target.rank is None,
        target.rank if target.rank is not None else 10**9,
        -target.score,
        target.symbol.casefold(),
        index,
    )


def _source_plan(
    selected: _SelectedTarget,
    request: ContextRoutingRequest,
) -> tuple[list[_PlannedCall], list[ContextSourceOutcome]]:
    source_plan = (
        LITERATURE_SOURCE_PLAN
        if _routing_track(selected.target) is HypothesisTrack.LITERATURE
        else TBIO_SOURCE_PLAN
    )
    calls: list[_PlannedCall] = []
    immediate: list[ContextSourceOutcome] = []
    for source, family in source_plan:
        if (
            request.allowed_sources is not None
            and source not in request.allowed_sources
        ):
            immediate.append(
                ContextSourceOutcome(
                    source=source,
                    family=family,
                    status="policy_rejected",
                    queried=False,
                    detail=(
                        "not queried: ResearchBrief.data_policy.allowed_sources "
                        "excludes this source"
                    ),
                )
            )
            continue
        applicable, reason = _source_applicability(source, selected.target, request)
        if not applicable:
            immediate.append(
                ContextSourceOutcome(
                    source=source,
                    family=family,
                    status="skipped_not_applicable",
                    queried=False,
                    detail=reason,
                )
            )
            continue
        if source == "QuickGO" and not selected.target.uniprot_id:
            immediate.append(
                ContextSourceOutcome(
                    source=source,
                    family=family,
                    status="skipped_missing_identifier",
                    queried=False,
                    detail="uniprot_id is required for QuickGO",
                )
            )
            continue
        calls.append(
            _PlannedCall(
                selected=selected,
                source=source,
                family=family,
                selection_reason=reason,
            )
        )
    return calls, immediate


def _source_applicability(
    source: ContextSource,
    target: ContextTarget,
    request: ContextRoutingRequest,
) -> tuple[bool, str]:
    """Return an auditable deterministic admission decision for one source."""

    brief_text = "\n".join(
        [request.target_cell_type or "", *request.human_priorities]
    ).casefold()
    optional_terms = {
        "Human Cell Atlas": ("single-cell", "single cell", "singlecell", "scrna", "단일세포", "단일 세포", "Human Cell Atlas"),
        "CELLxGENE Census": ("single-cell", "single cell", "singlecell", "scrna", "단일세포", "단일 세포", "CELLxGENE Census"),
        "ARCHS4": ("expression", "transcriptom", "rna-seq", "rna seq", "발현", "전사체", "ARCHS4"),
        "BioGRID": ("network", "interaction", "interactom", "ppi", "네트워크", "상호작용", "BioGRID"),
    }
    if has_negated_term(brief_text, optional_terms.get(source, ())):
        return False, "not selected: an explicit negative goal cue vetoes this optional context source"
    cell_context_kind = classify_cell_context(
        request.target_cell_type,
        request.human_priorities,
    )
    has_cell_resolved_context = cell_context_kind in {"single_cell", "cell_resolved"}
    has_expression_context = cell_context_kind != "none" or _mentions_any(
        brief_text,
        ("expression", "transcriptom", "rna-seq", "rna seq", "발현", "전사체"),
    )
    has_network_context = _mentions_any(
        brief_text,
        ("network", "interaction", "interactom", "ppi", "네트워크", "상호작용"),
    )

    if source in {"Human Cell Atlas", "CELLxGENE Census"}:
        if has_cell_resolved_context:
            return (
                True,
                "selected because the research brief specifies a cell-resolved or single-cell context",
            )
        if cell_context_kind == "cell_line":
            return (
                False,
                "not selected: a cell-line model is not a cell-resolved atlas request",
            )
        return (
            False,
            "not selected: an explicit cell-resolved or single-cell context is required",
        )
    if source == "ARCHS4":
        if _routing_track(target) is HypothesisTrack.TBIO:
            return True, "selected for effective Tbio work-track expression context"
        if has_expression_context:
            return True, "selected because the research brief requests expression context"
        return (
            False,
            "not selected: ARCHS4 requires a Tbio target or explicit expression/cell context",
        )
    if source == "BioGRID":
        if _routing_track(target) is HypothesisTrack.TBIO:
            return True, "selected for effective Tbio work-track interaction context"
        if has_network_context:
            return True, "selected because the research brief requests network context"
        return (
            False,
            "not selected: BioGRID requires a Tbio target or explicit network/interaction context",
        )
    return True, f"selected by the {_routing_track(target).value} work-track source plan"


def _mentions_any(text: str, terms: Sequence[str]) -> bool:
    return has_affirmative_term(text, terms)


def _round_robin_plan(
    selected: Sequence[_SelectedTarget],
    planned_by_target: Mapping[int, Sequence[_PlannedCall]],
) -> list[_PlannedCall]:
    """Interleave tracks/targets so a tight call budget does not starve Tbio."""

    ordered_targets: list[_SelectedTarget] = []
    for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO):
        ordered_targets.extend(
            item for item in selected if _routing_track(item.target) is track
        )
    max_depth = max(
        (len(planned_by_target[item.request_index]) for item in ordered_targets),
        default=0,
    )
    return [
        planned_by_target[item.request_index][depth]
        for depth in range(max_depth)
        for item in ordered_targets
        if depth < len(planned_by_target[item.request_index])
    ]


def _record_budget_skips(
    remaining: Sequence[_PlannedCall],
    outcome_lists: dict[int, list[ContextSourceOutcome]],
    status: Literal["skipped_call_budget", "skipped_time_budget"],
) -> None:
    detail = (
        "not queried because max_source_calls was reached"
        if status == "skipped_call_budget"
        else "not queried because max_elapsed_seconds was reached"
    )
    for planned in remaining:
        outcome_lists[planned.selected.request_index].append(
            ContextSourceOutcome(
                source=planned.source,
                family=planned.family,
                status=status,
                queried=False,
                detail=_join_details(planned.selection_reason, detail),
            )
        )


def _is_context_only(record: EvidenceRef) -> bool:
    metadata = record.metadata
    return (
        record.supports == "context"
        and metadata.get("context_only") is True
        and metadata.get("causal_inference_allowed") is False
        and metadata.get("counter_evidence_allowed") is False
        and bool(metadata.get("provenance_class"))
        and bool(metadata.get("retrieval_adapter"))
        and bool(metadata.get("retrieval_endpoint"))
    )


def _provenance(record: EvidenceRef) -> ContextEvidenceProvenance:
    metadata = record.metadata
    return ContextEvidenceProvenance(
        evidence_id=record.evidence_id,
        source=record.source,
        external_id=record.external_id,
        url=record.url,
        provenance_class=str(metadata["provenance_class"]),
        retrieval_adapter=str(metadata["retrieval_adapter"]),
        retrieval_endpoint=str(metadata["retrieval_endpoint"]),
        cache_hit=bool(metadata.get("cache_hit", False)),
    )


def _ordered_outcomes(
    track: HypothesisTrack,
    outcomes: Sequence[ContextSourceOutcome],
) -> list[ContextSourceOutcome]:
    source_plan = (
        LITERATURE_SOURCE_PLAN
        if track is HypothesisTrack.LITERATURE
        else TBIO_SOURCE_PLAN
    )
    priority = {source: index for index, (source, _) in enumerate(source_plan)}
    return sorted(
        (item.model_copy(deep=True) for item in outcomes),
        key=lambda item: priority[item.source],
    )


def _elapsed(now: float, start: float) -> float:
    # Custom clocks used in tests/integrations should be monotonic, but clamp a
    # bad implementation rather than emitting an invalid negative duration.
    return round(max(0.0, now - start), 6)


def _join_details(*values: str | None) -> str | None:
    normalized = [value.strip() for value in values if value and value.strip()]
    return "; ".join(dict.fromkeys(normalized)) or None


__all__ = [
    "ContextEvidenceProvenance",
    "ContextLookup",
    "ContextRoutingBudget",
    "ContextRoutingRequest",
    "ContextRoutingResult",
    "ContextSourceOutcome",
    "ContextTarget",
    "ContextTargetSelection",
    "DiseaseContextRoutingGateway",
    "DiseaseTargetContextRouter",
    "LITERATURE_SOURCE_PLAN",
    "TBIO_SOURCE_PLAN",
    "TargetContextBundle",
    "context_request_from_hypothesis_packet",
]
