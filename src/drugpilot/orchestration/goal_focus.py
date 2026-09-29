"""Conservative, deterministic interpretation of explicit research-goal cues.

This module does not infer biology.  It recognizes a small, closed vocabulary
for source routing and acceptance emphasis, and exact gene-like tokens only
when they occur next to an explicit target/gene label.  A mentioned target can
affect ordering only after MA1 has independently produced an evidence-linked
hypothesis for that exact symbol.
"""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Literal, Sequence

from pydantic import Field

from drugpilot.domain.models import (
    ExcludedHypothesis,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    ResearchBrief,
    StrictModel,
)
from drugpilot.domain.cell_context import classify_cell_context
from drugpilot.domain.goal_intent import cue_rejection_reason, literal_matches


GoalCueCategory = Literal[
    "single_cell",
    "cell_context",
    "cell_line",
    "expression",
    "network",
    "pharos",
    "repurposing",
    "novel_target",
    "safety",
    "explicit_target",
]


class GoalCue(StrictModel):
    category: GoalCueCategory
    source_field: Literal[
        "research_goal", "rationale", "target_cell_type", "human_priorities"
    ]
    matched_text: str
    reason: str


class GoalFocus(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    focus_id: str
    brief_id: str
    cues: list[GoalCue] = Field(default_factory=list)
    suppressed_cues: list[GoalCue] = Field(default_factory=list)
    source_families: list[str] = Field(default_factory=list)
    routing_priorities: list[str] = Field(default_factory=list)
    explicit_target_mentions: list[str] = Field(default_factory=list)
    target_scope_mode: Literal["portfolio", "explicit_only"] = "portfolio"
    executable_target_mentions: list[str] = Field(default_factory=list)
    resolved_target_mentions: list[str] = Field(default_factory=list)
    rejected_target_mentions: list[dict[str, str]] = Field(default_factory=list)
    candidate_strategy_focus: Literal[
        "repurposing", "novel_target", "conflicting", "unspecified"
    ] = "unspecified"
    acceptance_focus: list[str] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)
    evidence_guard: str = (
        "Goal cues may select sources or reorder existing evidence-linked hypotheses; "
        "a closed explicit executable scope may also exclude non-matching hypotheses. "
        "They cannot create targets, citations, activities, or safety conclusions."
    )


_CUE_TERMS: dict[GoalCueCategory, tuple[str, ...]] = {
    "single_cell": (
        "single-cell", "single cell", "singlecell", "scrna", "단일세포",
    ),
    "expression": (
        "gene expression", "expression profile", "transcriptom", "rna-seq",
        "rna seq", "유전자 발현", "발현 분석", "전사체",
    ),
    "network": (
        "interaction network", "protein interaction", "gene network",
        "interactome", "ppi network", "상호작용 네트워크", "단백질 상호작용",
        "유전자 네트워크",
    ),
    "pharos": (
        "pharos", "tcrd", "target development level", "target-development level",
        "tdl", "druggability", "표적 개발 수준", "약물화 가능성",
    ),
    "repurposing": (
        "drug repurposing", "drug repositioning", "repurpos", "reposition",
        "약물 재창출", "약물재창출", "재창출", "리퍼포징",
    ),
    "novel_target": (
        "novel target", "new target", "underexplored target", "unexplored target",
        "신규 표적", "새로운 표적", "미개척 표적",
    ),
    "safety": (
        "safety", "toxicity", "toxicology", "admet", "contraindication",
        "안전성", "독성", "금기",
    ),
    "explicit_target": (),
}

_ROUTING_HINT = {
    "single_cell": "single-cell context",
    "cell_context": "cell-resolved context",
    "cell_line": "cell-line model context",
    "expression": "expression context",
    "network": "interaction network context",
    "pharos": "Pharos target development level",
}

_TARGET_PATTERNS = (
    re.compile(
        r"(?:(?:target|gene|protein)\b|(?:표적|유전자|타깃)(?!들)(?:은|는|이|가)?)"
        r"\s*(?:symbol)?\s*[:=]?\s*"
        r"([A-Za-z][A-Za-z0-9-]{1,14})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Z0-9-]{1,14})\b\s*"
        r"(?:(?:target|gene|protein)\b|(?:표적|유전자|타깃)(?!들)(?:은|는|이|가|을|를)?)"
        r"(?![가-힣])",
    ),
)
_TARGET_STOPWORDS = {
    "candidate", "disease", "drug", "novel", "new", "safety", "specific",
    "therapeutic", "unknown", "cell", "network", "protein", "gene", "target",
    "inhibitor", "inhibitors", "activator", "activators", "agonist", "agonists",
}

_GENE_SYMBOL = r"[A-Z][A-Z0-9-]{1,14}"
_GENE_SYMBOL_LIST = (
    rf"{_GENE_SYMBOL}(?:(?:\s*[,/+&]\s*|\s+(?i:and|or)\s+)"
    rf"{_GENE_SYMBOL}){{0,7}}"
)
_EXECUTABLE_TARGET_SCOPE_PATTERNS = (
    re.compile(
        rf"(?i:\b(?:direct\s+)?)(?P<targets>{_GENE_SYMBOL_LIST})\s+"
        rf"(?i:inhibition|activation|agonism|antagonism|blockade|suppression)\s+"
        rf"(?i:only|exclusively)\b"
    ),
    re.compile(
        rf"(?i:\b(?:direct\s+)?(?:inhibition|activation|agonism|antagonism|"
        rf"blockade|suppression)\s+of\s+)"
        rf"(?P<targets>{_GENE_SYMBOL_LIST})\s+(?i:only|exclusively)\b"
    ),
    re.compile(
        rf"(?i:\b(?:executable|execution[- ]eligible)\s+"
        rf"(?:targets?|genes?|proteins?)\s*(?:is|are|:|=)?\s*)"
        rf"(?P<targets>{_GENE_SYMBOL_LIST})(?:\s*(?i:only|exclusively))?"
    ),
    re.compile(
        rf"(?i:\b(?:restrict|restricted|restricting|limit|limited|limiting)\s+"
        rf"(?:the\s+)?(?:executable\s+)?(?:targets?|genes?|proteins?)\s+"
        rf"(?:to|as)\s*)(?P<targets>{_GENE_SYMBOL_LIST})"
    ),
    re.compile(
        rf"(?i:\b(?:targets?|genes?|proteins?)\s*(?:is|are|:|=)?\s*)"
        rf"(?P<targets>{_GENE_SYMBOL_LIST})\s+(?i:only|exclusively)\b"
    ),
    re.compile(
        rf"(?P<targets>{_GENE_SYMBOL_LIST})\s+"
        rf"(?i:is\s+the\s+only\s+(?:executable\s+)?(?:target|gene|protein))\b"
    ),
    re.compile(
        rf"(?:실행\s*가능|실행)\s*(?:표적|유전자|단백질)\s*"
        rf"(?:은|는|:|=)?\s*(?P<targets>{_GENE_SYMBOL_LIST})"
        rf"(?:\s*(?:만|으로\s*제한))?"
    ),
    re.compile(
        rf"(?:표적|유전자|단백질)(?:을|를)?\s*"
        rf"(?P<targets>{_GENE_SYMBOL_LIST})(?:로|으로)?\s*제한"
    ),
)


def _explicit_executable_target_scope(text: str) -> tuple[list[str], list[str]]:
    """Return exact upper-case symbols only from closed restrictive clauses."""

    symbols: list[str] = []
    matched_clauses: list[str] = []
    for pattern in _EXECUTABLE_TARGET_SCOPE_PATTERNS:
        for match in pattern.finditer(text):
            if cue_rejection_reason(text, match.start(), match.end()):
                continue
            clause = " ".join(match.group(0).split())
            if clause not in matched_clauses:
                matched_clauses.append(clause)
            for raw_symbol in re.findall(_GENE_SYMBOL, match.group("targets")):
                symbol = raw_symbol.upper()
                if (
                    symbol.casefold() in _TARGET_STOPWORDS
                    or symbol in symbols
                ):
                    continue
                symbols.append(symbol)
                if len(symbols) >= 8:
                    return symbols, matched_clauses
    return symbols, matched_clauses


def derive_goal_focus(brief: ResearchBrief) -> GoalFocus:
    """Parse only exact, bounded cues from user-controlled brief fields."""

    fields: list[tuple[str, str]] = [
        ("research_goal", brief.research_goal),
        ("rationale", brief.rationale or ""),
        ("target_cell_type", brief.constraints.target_cell_type or ""),
        ("human_priorities", "\n".join(brief.human_priorities)),
    ]
    cues: list[GoalCue] = []
    suppressed_cues: list[GoalCue] = []
    seen_cues: set[tuple[str, str, str]] = set()
    categories: set[str] = set()
    target_mentions: list[str] = []
    executable_target_mentions: list[str] = []
    executable_scope_matches: list[tuple[str, str]] = []

    for field_name, raw_text in fields:
        for category, terms in _CUE_TERMS.items():
            if category == "explicit_target":
                continue
            for match in literal_matches(raw_text, terms):
                term = match.group()
                rejection = cue_rejection_reason(raw_text, match.start(), match.end())
                if rejection:
                    suppressed_cues.append(GoalCue(category=category, source_field=field_name,
                        matched_text=term, reason=rejection))
                    continue
                key = (category, field_name, term.casefold())
                if key in seen_cues:
                    continue
                seen_cues.add(key)
                categories.add(category)
                cues.append(
                    GoalCue(
                        category=category,
                        source_field=field_name,
                        matched_text=term,
                        reason=f"explicit bounded '{category}' cue matched in {field_name}",
                    )
                )
        for pattern in _TARGET_PATTERNS:
            for match in pattern.finditer(raw_text):
                raw_symbol = match.group(1)
                # Gene symbols are conservatively accepted only when the user
                # wrote an upper-case identifier next to the explicit label.
                # This avoids treating prose such as "novel target discovery"
                # as a target named DISCOVERY.
                if raw_symbol != raw_symbol.upper() or not any(
                    character.isalpha() and character.isupper()
                    for character in raw_symbol
                ):
                    continue
                symbol = raw_symbol.upper()
                if symbol.casefold() in _TARGET_STOPWORDS or symbol in target_mentions:
                    continue
                rejection = cue_rejection_reason(raw_text, match.start(), match.end())
                if rejection:
                    suppressed_cues.append(GoalCue(category="explicit_target", source_field=field_name,
                        matched_text=match.group(), reason=rejection))
                    continue
                target_mentions.append(symbol)
                cues.append(
                    GoalCue(
                        category="explicit_target",
                        source_field=field_name,
                        matched_text=match.group(0),
                        reason="gene-like token occurs next to an explicit target/gene label",
                    )
                )
                if len(target_mentions) >= 8:
                    break

        if field_name in {"research_goal", "rationale", "human_priorities"}:
            for pattern in _EXECUTABLE_TARGET_SCOPE_PATTERNS:
                for match in pattern.finditer(raw_text):
                    rejection = cue_rejection_reason(raw_text, match.start(), match.end())
                    if rejection:
                        suppressed_cues.append(GoalCue(category="explicit_target", source_field=field_name,
                            matched_text=match.group(), reason=rejection))
            scoped_symbols, matched_clauses = _explicit_executable_target_scope(
                raw_text
            )
            for symbol in scoped_symbols:
                if symbol not in executable_target_mentions:
                    executable_target_mentions.append(symbol)
                if symbol not in target_mentions:
                    target_mentions.append(symbol)
            for clause in matched_clauses:
                scope_key = (field_name, clause)
                if scope_key in executable_scope_matches:
                    continue
                executable_scope_matches.append(scope_key)
                cues.append(
                    GoalCue(
                        category="explicit_target",
                        source_field=field_name,
                        matched_text=clause,
                        reason=(
                            "closed executable/restrictive target-scope clause "
                            "matched"
                        ),
                    )
                )

    cell_context_kind = classify_cell_context(
        brief.constraints.target_cell_type,
    )
    if brief.constraints.target_cell_type and cell_context_kind != "none":
        cell_category = {
            "single_cell": "single_cell",
            "cell_resolved": "cell_context",
            "cell_line": "cell_line",
        }.get(cell_context_kind)
        assert cell_category is not None
        categories.add(cell_category)
        cues.append(
            GoalCue(
                category=cell_category,
                source_field="target_cell_type",
                matched_text=brief.constraints.target_cell_type,
                reason=(
                    "explicit target cell type was deterministically classified as "
                    f"{cell_context_kind}"
                ),
            )
        )

    if brief.constraints.target_cell_type and cell_context_kind == "none":
        suppressed_cues.append(GoalCue(category="cell_context", source_field="target_cell_type",
            matched_text=brief.constraints.target_cell_type,
            reason="NO_AFFIRMATIVE_SPECIFIC_CELL_CONTEXT"))

    # A priority can request a cell context, but must not be attributed to an
    # unrelated/generic value in the structured target-cell field.
    priority_cell_kind = classify_cell_context(None, brief.human_priorities)
    # Long protocol priorities often repeat the named cell line and its
    # endpoints.  That generic ``cells`` wording must not override an explicit
    # structured cell-line/isogenic-comparator classification and route the run
    # to single-cell atlases.  An explicit single-cell priority still wins.
    if (
        cell_context_kind == "cell_line"
        and priority_cell_kind == "cell_resolved"
    ):
        priority_cell_kind = "none"
    priority_cell_category = {"single_cell": "single_cell", "cell_resolved": "cell_context",
                              "cell_line": "cell_line"}.get(priority_cell_kind)
    if priority_cell_category:
        categories.add(priority_cell_category)
        if not any(c.category == priority_cell_category and c.source_field == "human_priorities" for c in cues):
            cues.append(GoalCue(category=priority_cell_category, source_field="human_priorities",
                matched_text="\n".join(brief.human_priorities),
                reason=f"affirmative human-priority cell context classified as {priority_cell_kind}"))

    repurposing = "repurposing" in categories
    novel = "novel_target" in categories
    strategy = (
        "conflicting" if repurposing and novel else
        "repurposing" if repurposing else
        "novel_target" if novel else
        "unspecified"
    )
    source_families: list[str] = []
    if "single_cell" in categories or "cell_context" in categories:
        source_families.append("single_cell")
    source_families.extend(
        name for name in ("expression", "network", "pharos")
        if name in categories
    )
    routing_categories = [
        name for name in ("single_cell", "cell_context", "cell_line", "expression", "network", "pharos")
        if name in categories
    ]
    routing_priorities = [_ROUTING_HINT[name] for name in routing_categories]
    acceptance_focus: list[str] = []
    if strategy == "repurposing":
        acceptance_focus.append("require evidence-qualified known-drug and disease relevance")
    elif strategy == "novel_target":
        acceptance_focus.append("require evidence-backed underexplored-target qualification")
    elif strategy == "conflicting":
        acceptance_focus.append("do not choose a candidate strategy until the conflict is resolved")
    if "safety" in categories:
        acceptance_focus.append("retain explicit toxicity, contraindication and uncertainty checks")

    reasons = [
        "only closed-vocabulary literal cues were admitted",
        "unknown prose leaves current deterministic behavior unchanged",
    ]
    if target_mentions:
        reasons.append("target mentions remain pending until exact match to MA1 evidence")
    if executable_target_mentions:
        reasons.append(
            "explicit executable target scope excludes every non-matching hypothesis"
        )
    if suppressed_cues:
        reasons.append("negative, conditional, question or generic-context cues did not create positive routing/scope")
    focus_payload = "|".join(
        [
            brief.brief_id,
            *(f"{c.category}:{c.source_field}:{c.matched_text}" for c in cues),
            "target_scope_mode:"
            + ("explicit_only" if executable_target_mentions else "portfolio"),
            *(f"executable_target:{symbol}" for symbol in executable_target_mentions),
            *(f"suppressed:{c.category}:{c.source_field}:{c.matched_text}:{c.reason}" for c in suppressed_cues),
        ]
    )
    focus_id = f"focus_{sha256(focus_payload.encode('utf-8')).hexdigest()[:12]}"
    return GoalFocus(
        focus_id=focus_id,
        brief_id=brief.brief_id,
        cues=cues,
        suppressed_cues=suppressed_cues,
        source_families=source_families,
        routing_priorities=routing_priorities,
        explicit_target_mentions=target_mentions,
        target_scope_mode=(
            "explicit_only" if executable_target_mentions else "portfolio"
        ),
        executable_target_mentions=executable_target_mentions,
        candidate_strategy_focus=strategy,
        acceptance_focus=acceptance_focus,
        decision_reasons=reasons,
    )


def routing_priorities_for_focus(
    human_priorities: Sequence[str], focus: GoalFocus
) -> list[str]:
    """Merge canonical routing hints without mutating the ResearchBrief."""

    return list(dict.fromkeys([*human_priorities, *focus.routing_priorities]))


def effective_discovery_priority(
    brief: ResearchBrief,
    focus: GoalFocus,
) -> tuple[Literal["auto", "novel_target", "repurposing_target"], str]:
    """Resolve an executable priority without allowing prose to force a route.

    The structured constraint always wins.  A single, unambiguous literal goal
    cue is used only when that constraint is ``auto``; downstream selection
    still requires an independently evidence-qualified hypothesis and falls
    back/abstains under the existing route policy.
    """

    requested = brief.constraints.discovery_priority
    if requested != "auto":
        return requested, "research_brief_structured_constraint"
    if focus.candidate_strategy_focus == "repurposing":
        return "repurposing_target", "explicit_goal_repurposing_cue_when_auto"
    if focus.candidate_strategy_focus == "novel_target":
        return "novel_target", "explicit_goal_novel_target_cue_when_auto"
    if focus.candidate_strategy_focus == "conflicting":
        return "auto", "conflicting_goal_strategy_cues_preserve_auto"
    return "auto", "no_explicit_goal_strategy_cue_preserve_auto"


def apply_goal_focus(
    packet: HypothesisPacket,
    focus: GoalFocus,
) -> tuple[HypothesisPacket, GoalFocus]:
    """Apply evidence-backed ranking and any closed explicit-only target scope."""

    enriched = packet.model_copy(deep=True)
    focus_mentions = (
        focus.executable_target_mentions
        if focus.target_scope_mode == "explicit_only"
        else focus.explicit_target_mentions
    )
    mention_keys = {value.casefold(): value for value in focus_mentions}
    resolved: list[str] = []
    rejected: list[dict[str, str]] = []
    matched_ids: set[str] = set()
    safe_match_ids_by_symbol: dict[str, set[str]] = {}
    hypothesis_symbols = {
        item.target.symbol.casefold() for item in enriched.hypotheses
    }
    raw_discovery_trace = enriched.metadata.get("ma1_discovery_candidate_trace", [])
    discovered_symbols = {
        str(item.get("target_symbol") or "").casefold()
        for item in raw_discovery_trace
        if isinstance(item, dict) and str(item.get("target_symbol") or "").strip()
    }
    raw_screened_sources = enriched.metadata.get("screened_target_sources", [])
    raw_screening_history = enriched.metadata.get(
        "ma1_screening_history_sources", []
    )
    current_screened_symbols = {
        str(item.get("target_symbol") or "").casefold()
        for item in raw_screened_sources
        if isinstance(item, dict) and str(item.get("target_symbol") or "").strip()
    }
    screening_history_symbols = {
        str(item.get("target_symbol") or "").casefold()
        for item in raw_screening_history
        if isinstance(item, dict) and str(item.get("target_symbol") or "").strip()
    }
    screened_symbols = current_screened_symbols | screening_history_symbols
    observed_symbols = hypothesis_symbols | discovered_symbols | screened_symbols

    for mention_key, original in mention_keys.items():
        matches = [
            item for item in enriched.hypotheses
            if item.target.symbol.casefold() == mention_key
        ]
        if not matches:
            reason = (
                "screened by MA1 but no claim-linked retained hypothesis"
                if mention_key in screened_symbols
                else "discovered by MA1 but not screened or retained"
                if mention_key in discovered_symbols
                else "not produced by MA1"
            )
            rejected.append({"symbol": original, "reason": reason})
            continue
        safe_matches = [item for item in matches if _is_evidence_backed(item)]
        if not safe_matches:
            rejected.append({"symbol": original, "reason": "no claim-linked support evidence"})
            continue
        canonical = safe_matches[0].target.symbol
        resolved.append(canonical)
        safe_match_ids_by_symbol[mention_key] = {
            item.hypothesis_id for item in safe_matches
        }
        for hypothesis in safe_matches:
            matched_ids.add(hypothesis.hypothesis_id)
            hypothesis.target.metadata = {
                **hypothesis.target.metadata,
                "goal_focus_primary_candidate": True,
                "goal_focus_id": focus.focus_id,
                "goal_focus_reason": "exact explicit target mention matched claim-linked support evidence",
            }

    scope_excluded_symbols: list[str] = []
    if focus.target_scope_mode == "explicit_only":
        scope_keys = {
            value.casefold() for value in focus.executable_target_mentions
        }
        executable_match_ids = {
            hypothesis_id
            for symbol in scope_keys
            for hypothesis_id in safe_match_ids_by_symbol.get(symbol, set())
        }
        retained: list[Hypothesis] = []
        removed: list[Hypothesis] = []
        for hypothesis in enriched.hypotheses:
            if hypothesis.hypothesis_id in executable_match_ids:
                retained.append(hypothesis)
            else:
                removed.append(hypothesis)
                scope_excluded_symbols.append(hypothesis.target.symbol)
        enriched.hypotheses = retained

        existing_scope_exclusions = {
            (
                item.target_symbol.casefold(),
                reason,
            )
            for item in enriched.excluded
            for reason in item.reason_codes
        }
        for hypothesis in removed:
            reason = (
                "EXECUTABLE_TARGET_SCOPE_EVIDENCE_UNQUALIFIED"
                if hypothesis.target.symbol.casefold() in scope_keys
                else "OUTSIDE_EXECUTABLE_TARGET_SCOPE"
            )
            exclusion_key = (hypothesis.target.symbol.casefold(), reason)
            if exclusion_key in existing_scope_exclusions:
                continue
            enriched.excluded.append(
                ExcludedHypothesis(
                    target_symbol=hypothesis.target.symbol,
                    reason_codes=[reason],
                    source_iteration=max(1, hypothesis.revision + 1),
                )
            )
            existing_scope_exclusions.add(exclusion_key)

    # Preserve MA3's contiguous per-track rank invariant while moving a safe
    # exact match ahead of peers only inside its existing supply track.
    if matched_ids or focus.target_scope_mode == "explicit_only":
        for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO):
            track_items = [item for item in enriched.hypotheses if item.track is track]
            track_items.sort(
                key=lambda item: (
                    item.hypothesis_id not in matched_ids,
                    item.rank if item.rank is not None else 1_000_000,
                    -item.score,
                    item.target.symbol.casefold(),
                )
            )
            for rank, item in enumerate(track_items, start=1):
                item.rank = rank

    updated_focus = focus.model_copy(
        update={
            "resolved_target_mentions": list(dict.fromkeys(resolved)),
            "rejected_target_mentions": rejected,
            "decision_reasons": [
                *focus.decision_reasons,
                (
                    "explicit executable target scope enforced"
                    if focus.target_scope_mode == "explicit_only"
                    else "exact evidence-backed target focus applied"
                    if resolved
                    else "no target focus was applied"
                ),
            ],
        },
        deep=True,
    )
    retained_symbols = {
        item.target.symbol.casefold() for item in enriched.hypotheses
    }
    target_sources = enriched.metadata.get("target_sources")
    scoped_target_sources = (
        {
            symbol: value
            for symbol, value in target_sources.items()
            if str(symbol).casefold() in retained_symbols
        }
        if focus.target_scope_mode == "explicit_only"
        and isinstance(target_sources, dict)
        else target_sources
    )
    enriched.metadata = {
        **enriched.metadata,
        **(
            {"target_sources": scoped_target_sources}
            if isinstance(target_sources, dict)
            else {}
        ),
        "goal_focus": updated_focus.model_dump(mode="json"),
        "goal_focus_observed_symbols": sorted(observed_symbols),
        "goal_focus_discovered_symbols": sorted(discovered_symbols),
        # The unqualified field is current-revision scope and therefore agrees
        # with ma1_screened_candidate_count.  Cumulative history remains
        # separately available for portfolio coverage and exclusion audit.
        "goal_focus_screened_symbols": sorted(current_screened_symbols),
        "goal_focus_screening_history_symbols": sorted(screened_symbols),
        "goal_focus_screened_symbols_semantics": "current_revision_unique_targets",
        "goal_focus_screening_history_symbols_semantics": (
            "cumulative_revision_unique_targets"
        ),
        "goal_focus_scope_enforced": (
            focus.target_scope_mode == "explicit_only"
        ),
        "goal_focus_executable_targets": list(
            focus.executable_target_mentions
        ),
        "goal_focus_retained_symbols": sorted(retained_symbols),
        "goal_focus_scope_excluded_symbols": sorted(
            set(scope_excluded_symbols),
            key=str.casefold,
        ),
    }
    return enriched, updated_focus


def _is_evidence_backed(hypothesis: Hypothesis) -> bool:
    support_ids = {
        item.evidence_id for item in hypothesis.evidence if item.supports == "support"
    }
    linked_ids = {
        evidence_id for claim in hypothesis.claims for evidence_id in claim.evidence_ids
    }
    return bool(support_ids & linked_ids)


__all__ = [
    "GoalCue",
    "GoalFocus",
    "apply_goal_focus",
    "derive_goal_focus",
    "effective_discovery_priority",
    "routing_priorities_for_focus",
]
