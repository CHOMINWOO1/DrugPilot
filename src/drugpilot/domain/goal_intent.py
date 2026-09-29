"""Bounded literal intent checks for optional goal routing, not scientific evidence.

This is not a general language parser or an authorization boundary. Ambiguous
negative/question clauses do not create affirmative cues. Structured source,
target and safety constraints remain authoritative and are never relaxed here.
"""
from __future__ import annotations

import re
from typing import Sequence


_CLAUSE_BREAK = re.compile(
    r"[.;!?\n]|\b(?:but|however|whereas|instead)\b|"
    r"\band\s+(?=(?:use|include|prioritize|evaluate|analy[sz]e|compare|focus|require)\b)|"
    r"하지만|그러나|대신|(?<=말고)|(?<=제외하고)|(?<=아니라)", re.I,
)
_NEGATIVE_PREFIX = re.compile(
    r"\b(?:not(?!\s+only\b)|no|never|without|avoid\w*|exclud\w*|omit\w*|"
    r"skip\w*|don['’]t|doesn['’]t|shouldn['’]t|mustn['’]t)\b", re.I,
)
_NEGATIVE_SUFFIX = re.compile(
    r"^\s*(?:[-,:]\s*)?(?:(?:analysis|analyses|profiling|sequencing|context|"
    r"data|models?|studies|study|sources?|atlases|atlas|search|network|only|exclusively)\s*){0,3}"
    r"(?:(?:is|are|was|were|should|must|will|would|can)\s+)?"
    r"(?:not\b|never\b|excluded\b|unnecessary\b|isn['’]t\b|aren['’]t\b)|"
    r"^\s*(?:은|는|을|를|이|가|도|만|으로|로)?\s*"
    r"(?:(?:분석|검색|조회|검토|사용|활용|대상|표적|유전자|제한|네트워크)(?:은|는|을|를|이|가|으로|로)?\s*){0,3}"
    r"(?:하지\s*(?:말|않)|제외|불필요|필요\s*없|아닌|아니라|말고|안\s*(?:함|한다|해))",
    re.I,
)
_CONDITIONAL_PREFIX = re.compile(r"^\s*(?:if\b|whether\b|만약)", re.I)
_SHARED_NEGATIVE_PREDICATE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:not\s+(?:required|needed|requested|desired|wanted)|unnecessary|excluded)\b|"
    r"(?:하지\s*(?:말|않)|필요\s*없|불필요|제외)", re.I,
)
_COORDINATION = re.compile(r"\b(?:and|or|nor)\b|(?:및|또는)\s|(?:와|과)\s", re.I)
_NEW_AFFIRMATIVE_ACTION = re.compile(
    r"\b(?:use|include|prioritize|evaluate|analy[sz]e|compare|focus|require)\b", re.I,
)


def cue_rejection_reason(text: str, start: int, end: int) -> str | None:
    """Classify one literal match in its local clause, keeping original offsets."""
    left, right, terminator = 0, len(text), ""
    for boundary in _CLAUSE_BREAK.finditer(text):
        if boundary.end() <= start:
            left = boundary.end()
        elif boundary.start() >= end:
            right, terminator = boundary.start(), boundary.group()
            break
    prefix, suffix = text[left:start], text[end:right]
    if terminator == "?":
        return "QUESTION_NOT_AFFIRMATIVE_DIRECTIVE"
    if _CONDITIONAL_PREFIX.search(text[left:right]):
        return "CONDITIONAL_NOT_AFFIRMATIVE_DIRECTIVE"
    if _NEGATIVE_PREFIX.search(prefix) or _NEGATIVE_SUFFIX.search(suffix):
        return "NEGATED_GOAL_CUE"
    # Shared predicate: "single-cell and expression analyses are not required".
    # An intervening affirmative action must not inherit this subject's negation.
    predicate = _SHARED_NEGATIVE_PREDICATE.search(suffix)
    if predicate:
        subject_tail = suffix[:predicate.start()]
        if _COORDINATION.search(subject_tail) and not _NEW_AFFIRMATIVE_ACTION.search(subject_tail):
            return "NEGATED_GOAL_CUE"
    return None


def literal_matches(text: str, terms: Sequence[str]):
    """Yield original-text matches; tolerate configured stems, not inside words."""
    for term in terms:
        # Some legacy terms deliberately are stems (transcriptom, repurpos).
        # Keep that compatibility while rejecting e.g. 'notsingle-cell'.
        suffix = (r"[A-Za-z]*" if term.casefold() in {"transcriptom", "repurpos", "reposition", "interactom"}
                  else r"s?(?![A-Za-z0-9])" if term.endswith("target")
                  else r"(?![A-Za-z0-9])" if term[-1:].isascii() else "")
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + suffix, re.I)
        yield from pattern.finditer(text)


def has_affirmative_term(text: str, terms: Sequence[str]) -> bool:
    return any(cue_rejection_reason(text, m.start(), m.end()) is None
               for m in literal_matches(text, terms))


def has_affirmative_pattern(text: str, pattern: re.Pattern) -> bool:
    return any(cue_rejection_reason(text, m.start(), m.end()) is None
               for m in pattern.finditer(text))


def has_negated_term(text: str, terms: Sequence[str]) -> bool:
    """Explicit optional-routing veto; questions are not negative instructions."""
    return any(cue_rejection_reason(text, m.start(), m.end()) == "NEGATED_GOAL_CUE"
               for m in literal_matches(text, terms))
