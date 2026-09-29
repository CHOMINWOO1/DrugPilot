"""Deterministic classification of user-supplied cellular context.

The classifier deliberately distinguishes immortalized/cultured cell-line
models from cell-resolved or explicitly single-cell requests.  A non-empty
``target_cell_type`` must not, by itself, imply that single-cell atlases are
applicable.
"""

from __future__ import annotations

import re
from typing import Literal, Sequence
from drugpilot.domain.goal_intent import has_affirmative_pattern


CellContextKind = Literal["none", "cell_line", "cell_resolved", "single_cell"]

_SINGLE_CELL_PATTERN = re.compile(
    r"(?:single[- ]?cell|single[- ]?nucle(?:us|i)|scRNA(?:[- ]?seq)?|snRNA(?:[- ]?seq)?|"
    r"cell[- ]resolved|단일\s*세포|단일\s*핵)",
    re.IGNORECASE,
)
_CELL_LINE_PATTERN = re.compile(
    r"(?:cell[- ]?lines?|cell[- ]?culture|cultured\s+cells?|immortali[sz]ed|"
    r"organoids?|isogenic(?:[- ]\w+){0,4}\s+comparator|parental\s+background|"
    r"세포\s*주|배양\s*세포|오가노이드)",
    re.IGNORECASE,
)
_CELL_CONTEXT_PATTERN = re.compile(
    r"(?:\b(?:cells?|blasts?|fibroblasts?|hepatocytes?|microglia|macrophages?|lymphocytes?|"
    r"monocytes?|neutrophils?|astrocytes?|endothelial|epithelial|stromal)\b|세포)",
    re.IGNORECASE,
)
_GENERIC_CONTEXT = re.compile(
    r"^\s*(?:(?:human|primary|disease[- ]relevant|experimental|generic|in[- ]vitro|"
    r"relevant|cancer|tumou?r|일반|질환\s*관련|실험|사람|인간)\s+)*"
    r"(?:cells?|cell[- ]?models?|models?|cell\s*/\s*model|세포(?:\s*모델)?|모델|"
    r"unspecified|unknown|none|n/?a|미정|없음|미지정)\s*$", re.I,
)
_GENERIC_MODEL = re.compile(r"\bcell[- ]?models?\b|세포\s*모델", re.I)


def classify_cell_context(
    target_cell_type: str | None,
    human_priorities: Sequence[str] = (),
) -> CellContextKind:
    """Classify explicit cellular context without inferring new biology."""

    target_text = (target_cell_type or "").strip()
    texts = [value for value in [target_text, *map(str, human_priorities)] if value]
    if any(has_affirmative_pattern(text, _SINGLE_CELL_PATTERN) for text in texts):
        return "single_cell"
    if any(has_affirmative_pattern(text, _CELL_LINE_PATTERN) for text in texts):
        return "cell_line"
    # An unspecified model or arbitrary non-empty field is not an atlas request.
    # Keep source-declared named cell contexts; do not infer a cell type from a
    # disease name, model placeholder, or unknown identifier.
    if any(not _GENERIC_CONTEXT.fullmatch(text) and not _GENERIC_MODEL.search(text)
           and has_affirmative_pattern(text, _CELL_CONTEXT_PATTERN) for text in texts):
        return "cell_resolved"
    return "none"


__all__ = ["CellContextKind", "classify_cell_context"]
