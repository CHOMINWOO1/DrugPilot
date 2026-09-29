from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from .models import (
    BenchmarkCase,
    CandidatePrediction,
    CasePrediction,
    FrozenModel,
    KnownCandidateHit,
)


DEFAULT_K_VALUES = (1, 3, 5, 10)
FAILED_STATUSES = frozenset({"failed", "timeout", "unavailable"})


class MetricValue(FrozenModel):
    metric: str = Field(min_length=1, max_length=120)
    k: int | None = Field(default=None, ge=1)
    value: float | None
    numerator: float = Field(ge=0)
    denominator: float = Field(ge=0)
    reason_unavailable: str | None = Field(default=None, max_length=500)


class CaseScore(FrozenModel):
    case_id: str
    metrics: tuple[MetricValue, ...]

    def metric(self, name: str, k: int | None = None) -> MetricValue:
        return next(item for item in self.metrics if item.metric == name and item.k == k)


class AggregateMetric(FrozenModel):
    metric: str
    k: int | None = Field(default=None, ge=1)
    value: float | None
    cases_scored: int = Field(ge=0)
    cases_total: int = Field(ge=1)
    reason_unavailable: str | None = None


def score_case(
    case: BenchmarkCase,
    prediction: CasePrediction,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> CaseScore:
    if case.case_id != prediction.case_id:
        raise ValueError("case and prediction identifiers differ")
    ks = tuple(sorted(set(k_values)))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("k values must contain positive integers")

    labels = case.labels
    relevant_targets = {_norm(item) for item in labels.relevant_target_ids}
    ranked_targets = _ranked_target_identifiers(prediction)
    known_hits = labels.known_candidate_hits
    ranked_candidates = sorted(
        (item for item in prediction.candidates if item.status != "excluded"),
        key=lambda item: (item.rank or 10**9, _norm(item.candidate_id)),
    )

    metrics: list[MetricValue] = []
    for k in ks:
        top_target_identifiers = ranked_targets[:k]
        target_recall_hits = len(
            {
                label
                for label in relevant_targets
                if any(label in identifiers for identifiers in top_target_identifiers)
            }
        )
        target_precision_hits = sum(
            bool(relevant_targets.intersection(identifiers))
            for identifiers in top_target_identifiers
        )
        metrics.append(
            _ratio_metric(
                "target_recall_at_k",
                k,
                target_recall_hits,
                len(relevant_targets),
                empty_reason="no relevant target labels",
            )
        )
        candidate_hits = _recovered_hit_count(known_hits, ranked_candidates[:k])
        candidate_recall = _ratio_metric(
            "candidate_recall_at_k",
            k,
            candidate_hits,
            len(known_hits),
            empty_reason="no known candidate-hit labels",
        )
        metrics.extend(
            (
                candidate_recall,
                candidate_recall.model_copy(update={"metric": "known_hit_recovery_at_k"}),
            )
        )

        metrics.append(
            _precision_metric(
                "target_precision_at_k",
                k,
                target_precision_hits,
                labels_complete=labels.target_labels_complete,
                label_kind="target",
            )
        )
        metrics.append(
            _precision_metric(
                "candidate_precision_at_k",
                k,
                candidate_hits,
                labels_complete=labels.candidate_labels_complete,
                label_kind="candidate",
            )
        )
        metrics.append(
            _enrichment_metric(
                "target_enrichment_at_k",
                k,
                target_precision_hits,
                relevant_count=len(relevant_targets),
                universe_size=labels.target_universe_size,
                labels_complete=labels.target_labels_complete,
                label_kind="target",
            )
        )
        metrics.append(
            _enrichment_metric(
                "candidate_enrichment_at_k",
                k,
                candidate_hits,
                relevant_count=len(known_hits),
                universe_size=labels.candidate_universe_size,
                labels_complete=labels.candidate_labels_complete,
                label_kind="candidate",
            )
        )

    excluded_targets = {
        _norm(value)
        for item in prediction.targets
        if item.status in {"excluded", "ineligible"}
        for value in (item.target_id, *item.identifiers)
    }
    excluded_targets.update(
        _norm(item.identifier)
        for item in prediction.exclusions
        if item.item_type == "target"
    )
    metrics.append(
        _ratio_metric(
            "target_false_exclusion_rate",
            None,
            len(relevant_targets.intersection(excluded_targets)),
            len(relevant_targets),
            empty_reason="no relevant target labels",
        )
    )

    excluded_candidates = {
        (_norm(value), _norm(item.target_id))
        for item in prediction.candidates
        if item.status == "excluded"
        for value in (item.candidate_id, *item.identifiers)
    }
    explicit_exclusions = {
        (_norm(item.identifier), _norm(item.target_id) if item.target_id else None)
        for item in prediction.exclusions
        if item.item_type == "candidate"
    }
    falsely_excluded_hits = sum(
        _hit_was_excluded(hit, excluded_candidates, explicit_exclusions)
        for hit in known_hits
    )
    metrics.append(
        _ratio_metric(
            "candidate_false_exclusion_rate",
            None,
            falsely_excluded_hits,
            len(known_hits),
            empty_reason="no known candidate-hit labels",
        )
    )

    expected_branches = {_norm(item) for item in labels.expected_target_branches}
    attempted = {
        _norm(value)
        for item in prediction.targets
        if item.attempted
        for value in (item.target_id, *item.identifiers)
    }
    metrics.append(
        _ratio_metric(
            "target_branch_coverage",
            None,
            len(expected_branches.intersection(attempted)),
            len(expected_branches),
            empty_reason="no expected target branches",
        )
    )

    metrics.extend(
        (
            _failure_rate(prediction, kind=None, name="source_tool_failure_rate"),
            _failure_rate(prediction, kind="source", name="source_failure_rate"),
            _failure_rate(prediction, kind="tool", name="tool_failure_rate"),
            MetricValue(
                metric="runtime_ms",
                value=round(prediction.runtime_ms, 6),
                numerator=prediction.runtime_ms,
                denominator=1,
            ),
        )
    )
    return CaseScore(case_id=case.case_id, metrics=tuple(metrics))


def aggregate_scores(scores: Sequence[CaseScore]) -> tuple[AggregateMetric, ...]:
    if not scores:
        raise ValueError("at least one case score is required")
    keys: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for score in scores:
        for metric in score.metrics:
            key = (metric.metric, metric.k)
            if key not in seen:
                seen.add(key)
                keys.append(key)

    aggregates: list[AggregateMetric] = []
    for name, k in keys:
        matching = [
            score.metric(name, k)
            for score in scores
            if any(item.metric == name and item.k == k for item in score.metrics)
        ]
        values = [item.value for item in matching if item.value is not None]
        aggregates.append(
            AggregateMetric(
                metric=name,
                k=k,
                value=(round(sum(values) / len(values), 6) if values else None),
                cases_scored=len(values),
                cases_total=len(scores),
                reason_unavailable=(
                    None if values else "metric was not label-permitted for any case"
                ),
            )
        )

    runtimes = [score.metric("runtime_ms").value or 0.0 for score in scores]
    aggregates.append(
        AggregateMetric(
            metric="runtime_ms_total",
            value=round(sum(runtimes), 6),
            cases_scored=len(scores),
            cases_total=len(scores),
        )
    )
    return tuple(aggregates)


def _ranked_target_identifiers(prediction: CasePrediction) -> list[set[str]]:
    ranked = sorted(
        (item for item in prediction.targets if item.rank is not None and item.status != "excluded"),
        key=lambda item: (item.rank or 10**9, _norm(item.target_id)),
    )
    result: list[set[str]] = []
    seen_primary: set[str] = set()
    for item in ranked:
        primary = _norm(item.target_id)
        if primary in seen_primary:
            continue
        seen_primary.add(primary)
        result.append({_norm(value) for value in (item.target_id, *item.identifiers)})
    return result


def _recovered_hit_count(
    hits: Sequence[KnownCandidateHit],
    predictions: Sequence[CandidatePrediction],
) -> int:
    predicted_identifiers: list[tuple[set[str], str]] = []
    for item in predictions:
        predicted_identifiers.append(
            (
                {_norm(value) for value in (item.candidate_id, *item.identifiers)},
                _norm(item.target_id),
            )
        )
    return sum(
        any(
            _norm(hit.target_id) == target_id
            and bool(_hit_identifiers(hit).intersection(values))
            for values, target_id in predicted_identifiers
        )
        for hit in hits
    )


def _hit_identifiers(hit: KnownCandidateHit) -> set[str]:
    return {_norm(value) for value in (hit.hit_id, *hit.identifiers)}


def _hit_was_excluded(
    hit: KnownCandidateHit,
    candidate_exclusions: set[tuple[str, str]],
    explicit_exclusions: set[tuple[str, str | None]],
) -> bool:
    target_id = _norm(hit.target_id)
    for identifier in _hit_identifiers(hit):
        if (identifier, target_id) in candidate_exclusions:
            return True
        if (identifier, target_id) in explicit_exclusions:
            return True
        if (identifier, None) in explicit_exclusions:
            return True
    return False


def _ratio_metric(
    name: str,
    k: int | None,
    numerator: int,
    denominator: int,
    *,
    empty_reason: str,
) -> MetricValue:
    return MetricValue(
        metric=name,
        k=k,
        value=(round(numerator / denominator, 6) if denominator else None),
        numerator=numerator,
        denominator=denominator,
        reason_unavailable=(None if denominator else empty_reason),
    )


def _precision_metric(
    name: str,
    k: int,
    recovered: int,
    *,
    labels_complete: bool,
    label_kind: str,
) -> MetricValue:
    if not labels_complete:
        return MetricValue(
            metric=name,
            k=k,
            value=None,
            numerator=recovered,
            denominator=k,
            reason_unavailable=f"{label_kind} labels are not exhaustive",
        )
    return MetricValue(
        metric=name,
        k=k,
        value=round(recovered / k, 6),
        numerator=recovered,
        denominator=k,
    )


def _enrichment_metric(
    name: str,
    k: int,
    recovered: int,
    *,
    relevant_count: int,
    universe_size: int | None,
    labels_complete: bool,
    label_kind: str,
) -> MetricValue:
    if not labels_complete or universe_size is None or relevant_count == 0:
        return MetricValue(
            metric=name,
            k=k,
            value=None,
            numerator=recovered,
            denominator=k,
            reason_unavailable=(
                f"{label_kind} enrichment requires exhaustive labels, a universe size, and positives"
            ),
        )
    prevalence = relevant_count / universe_size
    return MetricValue(
        metric=name,
        k=k,
        value=round((recovered / k) / prevalence, 6),
        numerator=recovered,
        denominator=k,
    )


def _failure_rate(
    prediction: CasePrediction,
    *,
    kind: str | None,
    name: str,
) -> MetricValue:
    attempted = [
        item
        for item in prediction.execution_attempts
        if item.counts_toward_failure_rate
        and item.status != "skipped"
        and (kind is None or item.kind == kind)
    ]
    failures = sum(item.status in FAILED_STATUSES for item in attempted)
    return _ratio_metric(
        name,
        None,
        failures,
        len(attempted),
        empty_reason=f"no attempted {kind or 'source/tool'} operations",
    )


def _norm(value: str) -> str:
    return " ".join(value.strip().casefold().split())


__all__ = [
    "AggregateMetric",
    "CaseScore",
    "DEFAULT_K_VALUES",
    "MetricValue",
    "aggregate_scores",
    "score_case",
]
