from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Awaitable, Callable, Iterable, Literal

import networkx as nx
from pydantic import BaseModel, Field

from drugpilot.config import Settings, get_settings

from drugpilot.domain.models import (
    ActivityMeasurement,
    AuditFinding,
    CandidateCard,
    CandidatePacket,
    GateName,
    GateVerdict,
    GateVerdictCode,
    HypothesisPacket,
    PlanSpec,
    ResearchBrief,
    ReworkOrder,
    ScientificBackendKind,
    ScientificExecutionStatus,
    Severity,
    TargetCandidate,
    TokenUsage,
    ToolCallRecord,
    ValidationResult,
)
from drugpilot.providers import LLMProvider, usage_for_failed_call
from drugpilot.providers.structured_outputs import ReviewerAssessmentOutput, ReviewerRoutingAssessmentOutput
from drugpilot.orchestration.goal_focus import GoalFocus, effective_discovery_priority
from drugpilot.orchestration.tool_coverage import validate_source_coverage_manifest
from drugpilot.tools.activity_comparability import (
    ACTIVITY_RANKING_SCHEMA,
    ASSAY_COMPARABILITY_POLICY,
    comparability_summary,
    derive_activity_rankings,
)
from drugpilot.tools.assay_variant_alignment import assay_variant_alignment
from drugpilot.tools.candidate_policy import (
    candidate_identity,
    screen_forbidden_drug,
    screen_reserved_control,
)
from drugpilot.tools.substructure_policy import screen_forbidden_substructures, substructure_input_hash
from drugpilot.tools.portfolio_ranking import (
    PORTFOLIO_RANKING_VERSION,
    portfolio_components,
    portfolio_score,
)
from drugpilot.tools.route_policy import (
    select_hypothesis_for_priority,
    select_route_from_evidence,
)


MA3_REVIEWER_SYSTEM_PROMPT = (
    "You are DrugPilot's independent, read-only scientific reviewer. "
    "Inspect the supplied immutable artifacts, deterministic findings, "
    "and validation results. Report only additional, artifact-linked "
    "concerns. Never rewrite an artifact, invent a tool result, change a "
    "threshold, or choose PASS/REVISE/REPLAN/ABSTAIN/BLOCK; the "
    "deterministic MA3 decision table alone selects the verdict. "
    "Your reported severity is advisory until an independent deterministic "
    "check confirms the same artifact-linked violation. Evidence roles belong "
    "to the tuple (evidence_id, hypothesis_id, claim_id, molecular_scope), not "
    "to a publication globally. The same paper may legitimately be unlinked "
    "context for general JAK2 and primary association evidence for JAK2:V617F. "
    "A role difference across hypotheses or molecular scopes alone is not an "
    "inconsistency. Evaluate each claim's provenance and scope separately. "
    "A discovery-trace record marked search_only_not_executable or "
    "not_assessed_search_stage is a search candidate, not an executable "
    "hypothesis; direction uncertainty in that search-only pool is not "
    "by itself a defect. Report a direction concern only when a retained "
    "hypothesis lacks or misclassifies its linked primary direction "
    "evidence, or when an artifact falsely asserts direction alignment. "
    "Set remediation_kind by the actual corrective task, not incidental words. "
    "Wrong source interpretation, disease/model linkage, duplicate counting and "
    "software/schema defects require computational_rework or data_required. "
    "Use experiment_required ONLY when genuinely new physical measurements are "
    "necessary; a sentence mentioning an experiment does not imply this. "
    "Unavailable sources are tool_configuration; exhausted budget is resource_required. "
    "Choose operator_review or null if the corrective task cannot be determined. "
    "ma3_admission_report.entries[*].original_proposal and ma1_original_* annotations "
    "are immutable pre-review history, not retained or executable claims. Evaluate "
    "current packet.hypotheses and their source-bound ma3_admission claim bindings; "
    "non-approved entries are excluded from downstream execution. A difference "
    "from an original unsupported proposal is not itself inconsistency. Raw source "
    "support labels may remain unchanged while the current claim roles are bound "
    "separately in the admission record. Independently challenge current claims "
    "using actual quoted source content; certificates and hashes alone do not "
    "establish scientific correctness."
)


_SMILES_ALLOWED = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#.$\\/%]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHEMBL_STABLE_ID = re.compile(r"^CHEMBL[1-9][0-9]*$")
_CANDIDATE_PRIMARY_EVIDENCE_SCHEMA = "candidate_primary_evidence_v2"
_PRIMARY_EXCERPT_MAX_CHARS = 360
_MAX_EXACT_CANDIDATE_ALIASES = 32
_MAX_PRIMARY_ALIAS_POOL_CANDIDATES = 100
_MAX_PRIMARY_COMPETITOR_ALIASES = (
    _MAX_PRIMARY_ALIAS_POOL_CANDIDATES * _MAX_EXACT_CANDIDATE_ALIASES
)
_PRIMARY_PUBMED_BATCH_MAX_IDS = (
    _MAX_PRIMARY_ALIAS_POOL_CANDIDATES * 3
)
_PRIMARY_RELEVANCE_POLICY = (
    "exact_candidate_target_endpoint_value_same_clause_v3"
)
_CANDIDATE_ALIAS_SOURCE = "chembl_pref_name_and_molecule_synonyms_exact_v1"
_PRIMARY_ACTIVITY_DOCUMENT_TOP_K = 3
_PRIMARY_DOCUMENT_BATCH_SIZE = 80
_PRIMARY_ACTIVITY_DOCUMENT_COMPLETE_V6_POLICY = (
    "complete_exact_parent_activity_document_rows_v1"
)
_PRIMARY_ALIAS_INVENTORY_SCHEMA = "candidate_pool_alias_inventory_v1"
_PUBMED_TYPE_SCHEMA = "pubmed_esummary_publication_type_v1"
_PUBMED_TYPE_POLICY = "journal_article_without_excluded_publication_type_v1"
_PUBMED_SUMMARY_ENDPOINT = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
)
_PRIMARY_CHEMBL_DOCUMENT_ENDPOINT = (
    "https://www.ebi.ac.uk/chembl/api/data/document.json"
)
_PAPER_V6_SOURCE_STAGE_ORDER = (
    "mechanism",
    "molecule",
    "activity",
    "assay",
    "document",
    "warning",
    "pubmed_type",
)
_PAPER_V6_SOURCE_STAGE_ENDPOINTS = {
    "mechanism": "https://www.ebi.ac.uk/chembl/api/data/mechanism.json",
    "molecule": "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
    "activity": "https://www.ebi.ac.uk/chembl/api/data/activity.json",
    "assay": "https://www.ebi.ac.uk/chembl/api/data/assay.json",
    "document": _PRIMARY_CHEMBL_DOCUMENT_ENDPOINT,
    "warning": "https://www.ebi.ac.uk/chembl/api/data/drug_warning.json",
    "pubmed_type": _PUBMED_SUMMARY_ENDPOINT,
}
_PAPER_V6_SOURCE_STAGE_POLICIES = {
    "mechanism": {"fail_operation_no_recovery_v1"},
    "molecule": {"fail_operation_no_recovery_v1"},
    "activity": {"candidate_isolation_without_backfill_v1"},
    "assay": {"explicit_unknown_context_no_imputation_v1"},
    "document": {"unverified_primary_no_substitution_v1"},
    "warning": {
        "candidate_exclusion_on_unresolved_warning_v1",
        "retryable_batch_then_candidate_isolation_v1",
    },
    "pubmed_type": {"unverified_type_no_substitution_v1"},
}
_PUBMED_REQUIRED_TYPE = "journal article"
_PUBMED_EXCLUDED_TYPES = frozenset(
    {
        "review",
        "systematic review",
        "meta analysis",
        "guideline",
        "practice guideline",
        "editorial",
        "letter",
        "comment",
        "news",
        "interview",
        "biography",
        "historical article",
        "patient education handout",
        "consensus development conference",
        "consensus development conference nih",
        "retracted publication",
        "published erratum",
        "expression of concern",
        "retraction notice",
        "retraction of publication",
        "protocol",
        "clinical trial protocol",
    }
)
_PAIRED_V4_REQUIRED_PRIORITY = (
    "dynamic_pubmed_primary_abstract_paired_v4_required"
)
_PAIRED_V4_ENVELOPE_KEY = "primary_abstract_paired_v4"
_PAIRED_V4_ENVELOPE_SCHEMA = "candidate_primary_evidence_paired_v4"
_PAIRED_V4_SEARCH_SCHEMA = "primary_abstract_search/v1"
_PAIRED_V4_PAIR_SCHEMA = "primary_abstract_pairs/v1"
_PAIRED_V4_POLICY = (
    "same_sentence_primary_pairs_exact_chembl_activity_and_assay_v4"
)
_PAIRED_V4_CONDITION_BINDING_SCHEMA = "exact_assay_condition_binding_v1"
_PAIRED_V4_MIN_NONCONTROL_CANDIDATES = 3
_PAIRED_V4_MAX_CANDIDATES = 12
_PAIRED_V4_MAX_DOCUMENTS = 8
_PAIRED_V4_MAX_ALIASES = 16
_PAIRED_V4_SEARCH_ENDPOINT = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
_PAIRED_V4_FETCH_ENDPOINT = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)
_PAIRED_V4_CHEMBL_MECHANISM_ENDPOINT = (
    "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
)
_PAIRED_V4_PRIMARY_TYPES = frozenset(
    {
        "journal article",
        "research article",
        "original article",
        "clinical study",
        "clinical trial",
        "comparative study",
        "evaluation study",
        "observational study",
        "multicenter study",
        "randomized controlled trial",
    }
)
_PAIRED_V4_EXCLUDED_TYPE = re.compile(
    r"(?:review|meta-analysis|editorial|letter|comment|guideline|news|"
    r"retracted publication|retraction (?:notice|of publication)|"
    r"published erratum|expression of concern|protocol)",
    flags=re.IGNORECASE,
)
_PAIRED_V4_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?|\.\d+)"
)
_PAIRED_V4_ENDPOINT_MARKER = re.compile(
    r"(?<![\w-])(?:p?IC50|p?EC50|GI50|LC50|CC50|AC50|ED50|"
    r"p?K[di]|pSTAT\d+)(?![\w-])",
    flags=re.IGNORECASE,
)
_PAIRED_V4_UNSAFE_SENTENCE = re.compile(
    r"(?<!\w)respectively(?!\w)|"
    r"(?:\[\s*\d+(?:\s*[-,–]\s*\d+)*\s*\]|"
    r"\b(?:PMID|PMCID|doi)\s*:?\s*\S+)|"
    r"\b(?:previously|formerly|earlier|prior)\s+"
    r"(?:was\s+|were\s+)?reported\b|"
    r"\breported\s+(?:previously|earlier)\b|"
    r"\b(?:according\s+to|as\s+reported|previous\s+stud(?:y|ies)|"
    r"prior\s+work)\b|"
    r"\b(?:about|approximately|approx\.?|roughly|around|circa)\s*"
    r"(?=\d|\.\d)|"
    r"\b(?:may|might|could|possibly|potentially|apparently|appears?|"
    r"suggests?|suggested)\b|"
    r"\b(?:no|not|neither|without|failed\s+to|fails\s+to|did\s+not)\b|"
    r"(?:±|\+/-|~|≈|≤|≥|<|>)",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_REQUIRED_PRIORITY = (
    "dynamic_pubmed_primary_abstract_paper_native_v5_required"
)
_PAPER_NATIVE_V6_REQUIRED_PRIORITY = (
    "dynamic_pubmed_primary_abstract_paper_native_v6_required"
)
_PAPER_NATIVE_V6_ENVELOPE_KEY = "primary_abstract_paper_native_v6"
_PAPER_NATIVE_V6_POLICY = _PAPER_NATIVE_V6_REQUIRED_PRIORITY
_PAPER_NATIVE_V6_AGGREGATION_POLICY = (
    "candidate_per_primary_record_atomic_support_then_fixed_order_aggregation_v1"
)
_PAPER_NATIVE_V6_GATEWAY_SCHEMA = "primary_abstract_paper_native_gateway_v6"
_PAPER_NATIVE_V6_AGGREGATION_SCHEMA = "primary_abstract_paper_native_search/v6"
_PAPER_NATIVE_V6_WRAPPER_SCREEN_SCHEMA = (
    "paper_native_v6_wrapper_atomic_screen_projection/v1"
)
_PAPER_NATIVE_V6_WRAPPER_SCREEN_REASONS = frozenset(
    {
        "SEED_CARDINALITY_NOT_EXACT",
        "PRIMARY_SOURCE_NOT_MAPPING",
        "PAPER_NATIVE_V6_EXACT_BINDING_INVALID",
        "DELEGATE_NOT_RETURNED",
    }
)
_PAPER_NATIVE_V6_PROJECTION_STOP_RULE = (
    "eligible_quota_reached_in_exact_v6_verified_pool_prefix_v1"
)
_PAPER_NATIVE_V6_MAX_COMPETITOR_ALIASES = 100 * 32
_PAPER_NATIVE_V6_EXCLUDED_PUBLICATION_TYPES = _PUBMED_EXCLUDED_TYPES
_PRIMARY_UNSAFE_CLAUSE = re.compile(
    r"\b(?:previously|formerly|earlier|prior)\s+"
    r"(?:was\s+|were\s+)?reported\b|"
    r"\b(?:was|were)\s+(?:previously|formerly|earlier)\s+reported\b|"
    r"\b(?:according\s+to|as\s+reported|previous\s+stud(?:y|ies)|prior\s+work)\b|"
    r"\b(?:may|might|could|possibly|potentially|apparently|appears?|suggests?|suggested)\b|"
    # Negation of potency remains unsafe. A terminal safety observation such
    # as "without cytotoxicity" does not negate a preceding exact IC50 result.
    r"\b(?:no|not|neither|without(?!\s+(?:(?:detectable|significant|measurable|overt)\s+)?"
    r"(?:cytotoxicity|toxicity)\s*(?:[.,;:!?)]|$))|failed\s+to|fails\s+to|did\s+not)\b|"
    r"\b(?:predict(?:ed|ing|ion|ions|ive|ively)?|"
    # Explicit experimental model nouns are not computational predictions.
    # Unqualified models/modeling and all prediction/computation terms remain.
    r"(?<!cell )(?<!cellular )(?<!animal )(?<!murine )(?<!xenograft )(?<!organoid )(?<!vitro )(?<!vivo )models?|"
    r"model(?:ed|ing|led|ling)|"
    r"simulat(?:e|es|ed|ing|ion|ions|ive|ively)|"
    r"calculat(?:e|es|ed|ing|ion|ions)|"
    r"computational(?:ly)?|in[\s-]+silico|prospective(?:ly)?|"
    r"hypothetical(?:ly)?|putative(?:ly)?|project(?:ed|ing|ion|ions)|"
    r"estimat(?:e|es|ed|ing|ion|ions)|forecast(?:s|ed|ing)?|anticipated)\b|"
    r"\b(?:planned|expected|proposed|anticipated|aimed)\s+to\s+be\b|"
    r"\b(?:plans?|proposes?|aims?)\s+to\s+be\b|"
    r"\b(?:will|would)(?:\s+\w+){0,2}\s+(?:be|been)\b|"
    r"\b(?:about|approximate(?:ly)?|approx\.?|roughly|around|circa)\s*(?=\d|\.\d)|"
    r"(?:±|\+/-|~|≈|≤|≥)|"
    r"(?<![\w.])[+-]\s*(?=\d)|"
    r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
    r"(?:[-–]|\bto\b)\s*\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])|"
    r"\b(?:range[sd]?|ranging)\s+from\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b|"
    r"\bbetween\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?\b|"
    r"\b(?:versus|vs\.?|compared\s+(?:with|to)|relative\s+to|"
    r"greater\s+than|less\s+than|higher\s+than|lower\s+than)\b",
    flags=re.IGNORECASE,
)
_PRIMARY_UNSAFE_NUMERIC_WORD_PREFIX = re.compile(
    r"(?<!\w)(?:about|approximate(?:ly)?|approx\.?|roughly|around|circa|"
    r"up\s+to|at\s+(?:most|least)|more\s+than|less\s+than|"
    r"greater\s+than|higher\s+than|lower\s+than|above|below|under|over)\s*$",
    flags=re.IGNORECASE,
)
_PRIMARY_UNSAFE_NUMERIC_SYMBOL_PREFIX = re.compile(
    r"(?:±|\+/-|~|≈|≤|≥|(?<![\w.])[+-])\s*$"
)


def _primary_clause_semantically_unsafe(text: str) -> bool:
    return _PRIMARY_UNSAFE_CLAUSE.search(
        unicodedata.normalize("NFKC", text)
    ) is not None


def _primary_numeric_match_has_unsafe_prefix(
    text: str, match: re.Match[str], *, number_group: str,
) -> bool:
    before_match = unicodedata.normalize("NFKC", text[:match.start()])
    before_number = unicodedata.normalize(
        "NFKC", text[:match.start(number_group)]
    )
    return bool(
        _PRIMARY_UNSAFE_CLAUSE.search(unicodedata.normalize("NFKC", text))
        or _PRIMARY_UNSAFE_NUMERIC_WORD_PREFIX.search(before_match)
        or _PRIMARY_UNSAFE_NUMERIC_SYMBOL_PREFIX.search(before_number)
    )


def _paper_v6_publication_type_key(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _paper_v6_publication_types_valid(value: object) -> bool:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        return False
    normalized_items = [" ".join(item.split()) for item in value]
    normalized_keys = [_normalize_primary_phrase(item) for item in normalized_items]
    if (
        normalized_items != value
        or any(not key for key in normalized_keys)
        or len(normalized_keys) != len(set(normalized_keys))
    ):
        return False
    verified, excluded = _primary_pubmed_type_decision(value)
    return verified and not excluded


def _paper_v6_exact_value_unit_present(
    sentence: str, value: object, unit: object, relation: object = "="
) -> bool:
    if isinstance(value, bool) or not isinstance(unit, str):
        return False
    for match in re.finditer(
        r"(?<![A-Za-z0-9.])(?P<relation><=|>=|<|>|=)?\s*(?P<value>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?P<unit>pM|nM|uM|µM|μM|mM|M)(?![A-Za-z0-9])",
        sentence,
    ):
        if _primary_numeric_match_has_unsafe_prefix(
            sentence, match, number_group="value"
        ):
            continue
        try:
            same_value = Decimal(match.group("value")) == Decimal(str(value))
        except (InvalidOperation, ValueError):
            same_value = False
        same_unit = match.group("unit") == unit
        observed_relation = match.group("relation") or "="
        relation_ok = observed_relation == relation
        if same_value and same_unit and relation_ok:
            return True
    return False
_PAPER_NATIVE_V5_ENVELOPE_KEY = "primary_abstract_paper_native_v5"
_PAPER_NATIVE_V5_ENVELOPE_SCHEMA = (
    "candidate_primary_evidence_paper_native_v5"
)
_PAPER_NATIVE_V5_SOURCE_CONTRACT = "primary_abstract_paper_native_v5"
_PAPER_NATIVE_V5_SEARCH_SCHEMA = "primary_abstract_paper_native_search/v5"
_PAPER_NATIVE_V5_PARSER_SCHEMA = "primary_abstract_paper_native/v5"
_PAPER_NATIVE_V5_GRAMMAR = (
    "head_measurement_followed_by_candidate_value_list_v1"
)
_PAPER_NATIVE_V5_MAX_CANDIDATES = 12
_PAPER_NATIVE_V5_MAX_DOCUMENTS = 8
_PAPER_NATIVE_V5_MAX_ALIASES = 16
_PAPER_NATIVE_V5_APPROVED_PCHEMBL_THRESHOLD = Decimal("5.0")
_PAPER_NATIVE_V5_SEARCH_ENDPOINT = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
_PAPER_NATIVE_V5_SUMMARY_ENDPOINT = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
)
_PAPER_NATIVE_V5_FETCH_ENDPOINT = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)
_PAPER_NATIVE_V5_CHEMBL_MECHANISM_ENDPOINT = (
    "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
)
_PAPER_NATIVE_V5_PRIMARY_TYPES = frozenset(
    {
        "journal article",
        "research article",
        "original article",
        "clinical study",
        "clinical trial",
        "comparative study",
        "evaluation study",
        "observational study",
        "multicenter study",
        "randomized controlled trial",
    }
)
_PAPER_NATIVE_V5_EXCLUDED_TYPE = re.compile(
    r"(?:review|meta-analysis|editorial|letter|comment|guideline|news|"
    r"retracted publication|retraction (?:notice|of publication)|"
    r"published erratum|expression of concern|protocol)",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_NUMBER = re.compile(
    r"(?:\d+(?:\.\d+)?|\.\d+)"
)
_PAPER_NATIVE_V5_ENDPOINT = re.compile(
    r"(?<![\w-])(?:p?IC50|p?EC50|GI50|LC50|CC50|AC50|ED50|"
    r"p?K[di]|pSTAT\d+)(?![\w-])",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_UNSAFE = re.compile(
    r"(?<!\w)respectively(?!\w)|"
    r"(?:\[\s*\d+(?:\s*[-,–]\s*\d+)*\s*\]|"
    r"\b(?:PMID|PMCID|doi)\s*:?\s*\S+)|"
    r"\b(?:previously|formerly|earlier|prior)\s+"
    r"(?:was\s+|were\s+)?reported\b|"
    r"\b(?:about|approximately|approx\.?|roughly|around|circa)\s*"
    r"(?=\d|\.\d)|"
    r"\b(?:may|might|could|possibly|potentially|apparently|appears?|"
    r"suggests?|suggested)\b|"
    r"\b(?:no|not|neither|without|failed\s+to|fails\s+to|did\s+not)\b|"
    r"\b(?:predict(?:ed|ing|ion|ions|ive|ively)?|"
    r"model(?:s|ed|ing|led|ling)?|"
    r"simulat(?:e|es|ed|ing|ion|ions|ive|ively)|"
    r"computational(?:ly)?|in[\s-]+silico|prospective(?:ly)?|"
    r"hypothetical(?:ly)?|putative(?:ly)?|"
    r"project(?:ed|ing|ion|ions)|"
    r"estimat(?:e|es|ed|ing|ion|ions)|"
    r"forecast(?:s|ed|ing)?|anticipated)\b|"
    r"\b(?:planned|expected|proposed|anticipated|aimed)\s+to\s+be\b|"
    r"\b(?:plans?|proposes?|aims?)\s+to\s+be\b|"
    r"\b(?:will|would)(?:\s+\w+){0,2}\s+(?:be|been)\b|"
    r"(?:±|\+/-|~|≈|≤|≥|<|>)|"
    r"\b(?:versus|vs\.?|compared\s+(?:with|to)|relative\s+to|"
    r"greater\s+than|less\s+than|higher\s+than|lower\s+than)\b",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_OBSERVED_POTENCY_PREFIX = re.compile(
    r"^\s+(?:was|is)\s+(?:"
    r"(?:(?:experimentally|directly)\s+)?|"
    r"(?:experimentally|directly),\s+as\s+expected,\s+|"
    r"(?:(?:experimentally|directly)\s+)?"
    r"(?:observed|measured|determined)\s+(?:to\s+be|as)\s+"
    r")(?:(?:the|a|an)\s+)?(?:most\s+)?potent\s+$",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_OBSERVED_TO_INHIBIT_PREFIX = re.compile(
    r"^\s+(?:was|is)\s+(?:(?:experimentally|directly)\s+)?"
    r"(?:observed|measured|determined)\s+to\s+inhibit\s+$",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_OBSERVED_ACTIVE_INHIBITION_PREFIX = re.compile(
    r"^\s+(?:(?:experimentally|directly)\s+)?"
    r"(?:inhibited|inhibits)\s+$",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_METHOD_PASSIVE = re.compile(
    r"^(?:inhibitory\s+)?(?:potenc(?:y|ies)|selectivit(?:y|ies)|effects?|"
    r"activit(?:y|ies)|inhibition|signaling|cell\s+growth|growth|viability|"
    r"responses?|profiles?).{0,220}\b(?:was|were)\s+"
    r"(?:(?:directly|experimentally)\s+)?"
    r"(?:measured|determined|assessed|evaluated|tested|quantified)\s+"
    r"(?:by|using|with|in)\b",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_METHOD_ACTIVE = re.compile(
    r"^(?:we|this\s+study)\s+"
    r"(?:(?:directly|experimentally)\s+)?"
    r"(?:measured|determined|assessed|evaluated|tested|quantified)\b"
    r".{0,220}\b(?:by|using|with|in)\b",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_METHOD_UNSAFE = re.compile(
    r"\b(?:review|meta-analysis|previously|formerly|earlier|prior\s+(?:work|study)|"
    r"as\s+reported|according\s+to|predicted|in(?:\s+|-)silico|computational|"
    r"modeled|modelled|simulation|hypothetical|putative)\b",
    flags=re.IGNORECASE,
)
_PAPER_NATIVE_V5_INHIBITOR_SUFFIX = re.compile(
    r"^\s+inhibitor\s*$", flags=re.IGNORECASE
)
_PAPER_NATIVE_V5_EMPTY_TARGET_SUFFIX = re.compile(r"^\s*$")
_ENDPOINT_SPECIFIC_TERMS: dict[str, frozenset[str]] = {
    "ic50": frozenset(
        {
            "ic50",
            "ic 50",
            "half maximal inhibitory concentration",
            "50 percent inhibitory concentration",
        }
    ),
    "ki": frozenset({"ki", "inhibition constant", "inhibitory constant"}),
    "kd": frozenset({"kd", "dissociation constant"}),
    "ec50": frozenset(
        {
            "ec50",
            "ec 50",
            "half maximal effective concentration",
            "50 percent effective concentration",
        }
    ),
}


def _normalize_primary_phrase(value: object) -> str:
    return " ".join(
        re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE)
    )


def _primary_exact_phrase_matches(
    text: str,
    phrases: Iterable[object],
) -> list[str]:
    normalized_text = _normalize_primary_phrase(text)
    normalized_phrases = {
        _normalize_primary_phrase(phrase)
        for phrase in phrases
        if _normalize_primary_phrase(phrase)
    }
    return [
        phrase
        for phrase in sorted(normalized_phrases)
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_text)
    ]


def _primary_endpoint_phrases(endpoint: str) -> list[str]:
    normalized = _normalize_primary_phrase(endpoint)
    if not normalized:
        return []
    return sorted(_ENDPOINT_SPECIFIC_TERMS.get(normalized, frozenset({normalized})))


def _primary_exact_activity_value_matches(text: str, activity) -> list[str]:
    """Independently require the bound value/unit without unit conversion."""

    return [
        str(item["text"])
        for item in _primary_exact_activity_value_spans(text, activity)
    ]


def _primary_exact_phrase_spans(
    text: str,
    phrases: Iterable[object],
) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    for phrase in phrases:
        normalized = _normalize_primary_phrase(phrase)
        tokens = re.findall(r"[\w]+", str(phrase or ""), flags=re.UNICODE)
        if not normalized or not tokens:
            continue
        pattern = re.compile(
            r"(?<!\w)"
            + r"[^\w]+".join(re.escape(token) for token in tokens)
            + r"(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        )
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), normalized)
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                {
                    "normalized_phrase": normalized,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    spans.sort(
        key=lambda item: (
            int(item["start"]),
            int(item["end"]),
            str(item["normalized_phrase"]),
        )
    )
    return spans


def _primary_exact_activity_value_spans(
    text: str,
    activity,
) -> list[dict[str, object]]:
    """Return exact deposited relation/value/unit spans without conversion."""

    try:
        expected = Decimal(str(activity.value))
    except InvalidOperation:
        return []
    relation_compatibility = {
        "=": {"", "="},
        "<": {"<"},
        "<=": {"<="},
        ">": {">"},
        ">=": {">="},
    }
    allowed_relations = relation_compatibility.get(activity.relation, set())
    if not allowed_relations:
        return []
    pattern = re.compile(
        rf"(?<![\w.])(?P<relation><=|>=|<|>|=)?\s*"
        rf"(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
        rf"(?P<unit>{re.escape(activity.unit)})(?!\w)",
    )
    matches: list[dict[str, object]] = []
    for match in pattern.finditer(text):
        if _primary_numeric_match_has_unsafe_prefix(
            text, match, number_group="number"
        ):
            continue
        relation = match.group("relation") or ""
        if relation not in allowed_relations:
            continue
        try:
            reported = Decimal(match.group("number"))
        except InvalidOperation:
            continue
        if reported == expected:
            leading = len(match.group(0)) - len(match.group(0).lstrip())
            trailing = len(match.group(0)) - len(match.group(0).rstrip())
            matches.append(
                {
                    "text": match.group(0).strip(),
                    "start": match.start() + leading,
                    "end": match.end() - trailing,
                    "relation": relation,
                    "number": match.group("number"),
                    "unit": match.group("unit"),
                }
            )
    return matches


def _primary_top_level_parenthetical_spans(
    text: str,
) -> list[tuple[int, int]]:
    """Return balanced top-level parenthetical spans, including delimiters."""

    spans: list[tuple[int, int]] = []
    depth = 0
    start: int | None = None
    for index, character in enumerate(text):
        if character == "(":
            if depth == 0:
                start = index
            depth += 1
        elif character == ")":
            if depth == 0:
                return []
            depth -= 1
            if depth == 0:
                assert start is not None
                spans.append((start, index + 1))
                start = None
    return spans if depth == 0 else []


def _primary_parenthetical_binding(
    sentence: str,
    *,
    candidate_aliases: Iterable[object],
    competitor_aliases: Iterable[object],
    target: TargetCandidate,
    activity: ActivityMeasurement,
) -> dict[str, object] | None:
    """Independently bind one exact candidate measurement parenthetical."""

    body = sentence.rstrip(".!?").rstrip()
    if (
        not body
        or len(body) > _PRIMARY_EXCERPT_MAX_CHARS
        or _primary_clause_semantically_unsafe(body)
        or re.search(r"[!?]|(?<!\d)\.(?!\d)", body)
    ):
        return None
    candidate_spans = _primary_exact_phrase_spans(body, candidate_aliases)
    competitor_spans = _primary_exact_phrase_spans(body, competitor_aliases)
    target_spans = _primary_exact_phrase_spans(
        body,
        (
            target.symbol,
            target.uniprot_id,
            target.target_name,
            target.metadata.get("chembl_target_id"),
        ),
    )
    endpoint_spans = _primary_exact_phrase_spans(
        body,
        _primary_endpoint_phrases(activity.endpoint),
    )
    value_spans = _primary_exact_activity_value_spans(body, activity)
    if (
        not candidate_spans
        or competitor_spans
        or not target_spans
        or not endpoint_spans
        or not value_spans
    ):
        return None
    candidate_coordinates = {
        (int(item["start"]), int(item["end"])) for item in candidate_spans
    }
    target_coordinates = {
        (int(item["start"]), int(item["end"])) for item in target_spans
    }
    if len(candidate_coordinates) != 1:
        return None
    candidate_span = candidate_spans[0]
    numeric_spans = list(
        re.finditer(
            r"(?<![\w.])(?:<=|>=|<|>|=)?\s*"
            r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
            r"(?:pM|nM|uM|µM|μM|mM|M)(?!\w)",
            body,
        )
    )
    for group_start, group_end in _primary_top_level_parenthetical_spans(body):
        group_text = body[group_start + 1:group_end - 1]
        if re.search(
            r"(?<!\w)(?:and|or|respectively)(?!\w)",
            group_text,
            flags=re.IGNORECASE,
        ):
            continue
        group_endpoint_spans = [
            item for item in endpoint_spans
            if group_start < int(item["start"]) and int(item["end"]) < group_end
        ]
        group_value_spans = [
            item for item in value_spans
            if group_start < int(item["start"]) and int(item["end"]) < group_end
        ]
        group_numeric_spans = [
            item for item in numeric_spans
            if group_start < item.start() and item.end() < group_end
        ]
        if not (
            len({(int(item["start"]), int(item["end"])) for item in group_endpoint_spans}) == 1
            and len({(int(item["start"]), int(item["end"])) for item in group_value_spans}) == 1
            and len(group_numeric_spans) == 1
        ):
            continue
        group_target_spans = [
            item for item in target_spans
            if group_start < int(item["start"]) and int(item["end"]) < group_end
        ]
        candidate_immediately_before = bool(
            int(candidate_span["end"]) <= group_start
            and not body[int(candidate_span["end"]):group_start].strip()
        )
        immediately_preceding_targets = [
            item for item in target_spans
            if int(item["end"]) <= group_start
            and not body[int(item["end"]):group_start].strip()
        ]
        target_span: dict[str, object] | None = None
        pattern: str | None = None
        if (
            candidate_immediately_before
            and len({(int(item["start"]), int(item["end"])) for item in group_target_spans}) == 1
        ):
            pattern = "candidate_parenthetical_target_endpoint_value"
            target_span = group_target_spans[0]
        elif (
            len({(int(item["start"]), int(item["end"])) for item in group_target_spans}) == 1
            and int(candidate_span["end"]) <= group_start
            and re.search(
                r"(?<!\w)(?:inhibit\w*|potent|activity)(?!\w)",
                body[int(candidate_span["end"]):int(group_target_spans[0]["start"])],
                flags=re.IGNORECASE,
            )
        ):
            target_span = group_target_spans[0]
            pattern = "candidate_subject_parenthetical_target_endpoint_value"
        elif (
            len({(int(item["start"]), int(item["end"])) for item in immediately_preceding_targets}) == 1
            and int(candidate_span["end"]) <= int(immediately_preceding_targets[0]["start"])
            and re.search(
                r"(?<!\w)(?:inhibit\w*|potent|activity)(?!\w)",
                body[int(candidate_span["end"]):int(immediately_preceding_targets[0]["start"])],
                flags=re.IGNORECASE,
            )
        ):
            target_span = immediately_preceding_targets[0]
            pattern = "candidate_subject_target_parenthetical_endpoint_value"
        elif (
            candidate_immediately_before
            and len(target_coordinates) == 1
            and (target_span := target_spans[0]) is not None
            and int(target_span["end"]) <= int(candidate_span["start"])
            and re.search(
                r"(?<!\w)(?:positive\s+control|reference|control|"
                r"comparable\s+to(?:\s+that)?\s+of)(?!\w)",
                body[int(target_span["end"]):int(candidate_span["start"])],
                flags=re.IGNORECASE,
            )
        ):
            pattern = "single_target_context_candidate_parenthetical_endpoint_value"
        if pattern is None or target_span is None:
            continue
        endpoint_span = group_endpoint_spans[0]
        value_span = group_value_spans[0]
        return {
            "binding_pattern": pattern,
            "matched_groups": ["candidate", "target", "activity"],
            "matched_terms": {
                "candidate": sorted({str(item["normalized_phrase"]) for item in candidate_spans}),
                "target": sorted({str(item["normalized_phrase"]) for item in target_spans}),
                "activity": sorted({str(item["normalized_phrase"]) for item in endpoint_spans}),
            },
            "matched_exact_value_unit": [str(value_span["text"])],
            "competitor_alias_matches": [],
            "bound_spans": {
                "candidate": candidate_span,
                "target": target_span,
                "endpoint": endpoint_span,
                "value": value_span,
                "atomic_parenthetical": {
                    "start": group_start,
                    "end": group_end,
                    "text": body[group_start:group_end],
                },
            },
        }
    return None


def _primary_clause_binding(
    clause: str,
    *,
    candidate_aliases: Iterable[object],
    competitor_aliases: Iterable[object],
    target,
    activity,
) -> dict[str, object] | None:
    """Independently recompute deterministic candidate/value ownership."""

    clause_body = clause.rstrip(".!?").rstrip()
    if (
        _primary_clause_semantically_unsafe(clause)
        or
        re.search(
            r";|:|\u2014|\u2013|(?<!\w)(?:whereas|while|but)(?!\w)",
            clause_body,
            flags=re.IGNORECASE,
        )
        or re.search(r"[!?]|(?<!\d)\.(?!\d)", clause_body)
        or re.search(
            r"(?<!\w)respectively(?!\w)",
            clause,
            flags=re.IGNORECASE,
        )
    ):
        return None
    candidate_spans = _primary_exact_phrase_spans(clause, candidate_aliases)
    competitor_spans = _primary_exact_phrase_spans(clause, competitor_aliases)
    target_spans = _primary_exact_phrase_spans(
        clause,
        (
            target.symbol,
            target.uniprot_id,
            target.target_name,
            target.metadata.get("chembl_target_id"),
        ),
    )
    endpoint_spans = _primary_exact_phrase_spans(
        clause,
        _primary_endpoint_phrases(activity.endpoint),
    )
    value_spans = _primary_exact_activity_value_spans(clause, activity)
    if (
        not candidate_spans
        or competitor_spans
        or not target_spans
        or not endpoint_spans
        or not value_spans
    ):
        return None
    if (
        len(
            {(int(item["start"]), int(item["end"])) for item in target_spans}
        )
        != 1
        or len(
            {(int(item["start"]), int(item["end"])) for item in endpoint_spans}
        )
        != 1
        or len(
            {(int(item["start"]), int(item["end"])) for item in value_spans}
        )
        != 1
    ):
        return None
    binding_pattern: str | None = None
    bound_spans: dict[str, dict[str, object]] | None = None
    for candidate_span in candidate_spans:
        for target_span in target_spans:
            for endpoint_span in endpoint_spans:
                for value_span in value_spans:
                    if (
                        int(candidate_span["end"])
                        <= int(target_span["start"])
                        and int(target_span["end"])
                        <= int(endpoint_span["start"])
                        and int(endpoint_span["end"])
                        <= int(value_span["start"])
                    ):
                        binding_pattern = "candidate_target_endpoint_value"
                    elif (
                        int(candidate_span["end"])
                        <= int(endpoint_span["start"])
                        and int(endpoint_span["end"])
                        <= int(value_span["start"])
                        and int(value_span["end"])
                        <= int(target_span["start"])
                        and re.search(
                            r"(?<!\w)(?:against|for)(?!\w)",
                            clause[
                                int(value_span["end"]):int(target_span["start"])
                            ],
                            flags=re.IGNORECASE,
                        )
                    ):
                        binding_pattern = (
                            "candidate_endpoint_value_against_target"
                        )
                    elif (
                        int(endpoint_span["end"])
                        <= int(candidate_span["start"])
                        and int(candidate_span["end"])
                        <= int(target_span["start"])
                        and int(target_span["end"])
                        <= int(value_span["start"])
                        and re.search(
                            r"(?<!\w)(?:of|for)(?!\w)",
                            clause[
                                int(endpoint_span["end"]):int(candidate_span["start"])
                            ],
                            flags=re.IGNORECASE,
                        )
                        and re.search(
                            r"(?<!\w)(?:against|for)(?!\w)",
                            clause[
                                int(candidate_span["end"]):int(target_span["start"])
                            ],
                            flags=re.IGNORECASE,
                        )
                    ):
                        binding_pattern = (
                            "endpoint_of_candidate_against_target_value"
                        )
                    else:
                        continue
                    earliest = min(
                        int(candidate_span["start"]),
                        int(target_span["start"]),
                        int(endpoint_span["start"]),
                        int(value_span["start"]),
                    )
                    latest = max(
                        int(candidate_span["end"]),
                        int(target_span["end"]),
                        int(endpoint_span["end"]),
                        int(value_span["end"]),
                    )
                    if re.search(
                        r"(?<!\w)(?:and|or)(?!\w)",
                        clause[earliest:latest],
                        flags=re.IGNORECASE,
                    ):
                        binding_pattern = None
                        continue
                    bound_spans = {
                        "candidate": candidate_span,
                        "target": target_span,
                        "endpoint": endpoint_span,
                        "value": value_span,
                    }
                    break
                if binding_pattern is not None:
                    break
            if binding_pattern is not None:
                break
        if binding_pattern is not None:
            break
    if binding_pattern is None or bound_spans is None:
        return None
    return {
        "binding_pattern": binding_pattern,
        "matched_groups": ["candidate", "target", "activity"],
        "matched_terms": {
            "candidate": sorted(
                {str(item["normalized_phrase"]) for item in candidate_spans}
            ),
            "target": sorted(
                {str(item["normalized_phrase"]) for item in target_spans}
            ),
            "activity": sorted(
                {str(item["normalized_phrase"]) for item in endpoint_spans}
            ),
        },
        "matched_exact_value_unit": [
            str(item["text"]) for item in value_spans
        ],
        "competitor_alias_matches": [],
        "bound_spans": bound_spans,
    }


def _primary_pubmed_type_decision(
    publication_types: Iterable[object],
) -> tuple[bool, list[str]]:
    normalized = {
        _normalize_primary_phrase(value)
        for value in publication_types
        if _normalize_primary_phrase(value)
    }
    excluded = sorted(normalized.intersection(_PUBMED_EXCLUDED_TYPES))
    return _PUBMED_REQUIRED_TYPE in normalized and not excluded, excluded


def _validated_primary_competitor_aliases(
    provenance: object,
    *,
    current_molecule_id: str,
    current_aliases: list[object],
    pool_candidates: Iterable[object],
    target_chembl_id: str,
) -> list[str] | None:
    """Validate the complete bounded alias inventory and recompute competitors."""

    if not isinstance(provenance, dict) or (
        provenance.get("policy")
        != "all_bounded_pool_candidate_aliases_except_current_v1"
        or provenance.get("current_molecule_chembl_id")
        != current_molecule_id
        or provenance.get("truncated") is not False
    ):
        return None
    inventory = provenance.get("inventory")
    if not isinstance(inventory, dict):
        return None
    records = inventory.get("records")
    if (
        inventory.get("schema_version") != _PRIMARY_ALIAS_INVENTORY_SCHEMA
        or inventory.get("pool_candidate_cap")
        != _MAX_PRIMARY_ALIAS_POOL_CANDIDATES
        or inventory.get("per_candidate_alias_cap")
        != _MAX_EXACT_CANDIDATE_ALIASES
        or inventory.get("competitor_alias_cap")
        != _MAX_PRIMARY_COMPETITOR_ALIASES
        or inventory.get("truncated") is not False
        or inventory.get("complete") is not True
        or inventory.get("missing_molecule_ids") != []
        or inventory.get("aliasless_molecule_ids") != []
        or inventory.get("fixture") is not False
        or not isinstance(records, list)
        or len(records) > _MAX_PRIMARY_ALIAS_POOL_CANDIDATES
        or inventory.get("candidate_count") != len(records)
        or inventory.get("retained_candidate_count") != len(records)
        or inventory.get("source_candidate_ids")
        != [
            str(record.get("molecule_chembl_id") or "")
            for record in records
            if isinstance(record, dict)
        ]
    ):
        return None
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    inventory_hash = sha256(canonical).hexdigest()
    if (
        inventory.get("inventory_sha256") != inventory_hash
        or provenance.get("inventory_sha256") != inventory_hash
    ):
        return None

    records_by_id: dict[str, list[str]] = {}
    alias_count = 0
    for record in records:
        if not isinstance(record, dict):
            return None
        molecule_id = str(record.get("molecule_chembl_id") or "")
        aliases = record.get("aliases")
        if (
            not molecule_id
            or molecule_id in records_by_id
            or not isinstance(aliases, list)
            or not aliases
            or len(aliases) > _MAX_EXACT_CANDIDATE_ALIASES
            or any(not isinstance(alias, str) for alias in aliases)
        ):
            return None
        normalized = [_normalize_primary_phrase(alias) for alias in aliases]
        if any(not item for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            return None
        records_by_id[molecule_id] = list(aliases)
        alias_count += len(aliases)
    if inventory.get("alias_count") != alias_count:
        return None
    if records_by_id.get(current_molecule_id) != current_aliases:
        return None

    # Every candidate visible in the same target packet must be represented;
    # the producer may also retain fetched candidates later screened out.
    for candidate in pool_candidates:
        metadata = getattr(candidate, "metadata", {})
        target = getattr(candidate, "target", None)
        if not isinstance(metadata, dict) or target is None:
            continue
        candidate_target_id = str(
            getattr(target, "metadata", {}).get("chembl_target_id") or ""
        )
        if candidate_target_id != target_chembl_id:
            continue
        molecule_id = str(
            metadata.get("chembl_id") or metadata.get("drug_id") or ""
        )
        aliases = metadata.get("chembl_exact_aliases")
        if molecule_id and isinstance(aliases, list):
            if records_by_id.get(molecule_id) != aliases:
                return None

    alias_owners: dict[str, dict[str, object]] = {}
    for molecule_id in sorted(records_by_id):
        if molecule_id == current_molecule_id:
            continue
        for alias in records_by_id[molecule_id]:
            normalized = _normalize_primary_phrase(alias)
            owner = alias_owners.setdefault(
                normalized,
                {"alias": alias, "molecule_chembl_ids": []},
            )
            raw_ids = owner["molecule_chembl_ids"]
            if not isinstance(raw_ids, list):
                return None
            raw_ids.append(molecule_id)
    owner_rows = [
        {
            "alias": item["alias"],
            "molecule_chembl_ids": sorted(
                set(item["molecule_chembl_ids"])
            ),
        }
        for _normalized, item in sorted(alias_owners.items())
    ]
    competitor_aliases = [str(item["alias"]) for item in owner_rows]
    if (
        len(competitor_aliases) > _MAX_PRIMARY_COMPETITOR_ALIASES
        or provenance.get("alias_owners") != owner_rows
        or provenance.get("competitor_exact_aliases")
        != competitor_aliases
    ):
        return None
    return competitor_aliases


def _paired_v4_canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _paired_v4_hash(value: object) -> str:
    return sha256(_paired_v4_canonical_bytes(value)).hexdigest()


def _paired_v4_alias_key(value: str) -> str:
    """Canonical identity key used only by the paired-v4 contract."""

    return unicodedata.normalize("NFKC", value).casefold()


def _paired_v4_unique_identity_strings(values: list[str]) -> bool:
    return bool(
        all(
            value
            and value == value.strip()
            and len(value) <= 160
            and "\r" not in value
            and "\n" not in value
            for value in values
        )
        and len({_paired_v4_alias_key(value) for value in values})
        == len(values)
    )


def _paired_v4_self_hash(
    value: object,
    *,
    hash_key: str,
) -> bool:
    if not isinstance(value, dict):
        return False
    reported = value.get(hash_key)
    if not isinstance(reported, str) or not _SHA256.fullmatch(reported):
        return False
    body = dict(value)
    body.pop(hash_key, None)
    return reported == _paired_v4_hash(body)


_PAIRED_V4_CONDITION_SOURCES = frozenset(
    {
        "activity_properties",
        "chembl_assay_context.record.assay_parameters",
    }
)
_PAIRED_V4_CONDITION_KEYS = frozenset(
    {
        "type",
        "standard_type",
        "value",
        "standard_value",
        "units",
        "standard_units",
    }
)


def _paired_v4_condition_component(
    record: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    string_only: bool,
) -> tuple[str | None, list[str], str | None]:
    """Read one lexical condition field without normalizing or repairing it."""

    observed: list[tuple[str, str]] = []
    for key in keys:
        if key not in record or record.get(key) is None:
            continue
        raw = record.get(key)
        if isinstance(raw, bool):
            return None, [], "BOOLEAN_CONDITION_FIELD"
        if string_only and not isinstance(raw, str):
            return None, [], "NON_STRING_CONDITION_FIELD"
        value = str(raw)
        if (
            not value
            or value != value.strip()
            or "\r" in value
            or "\n" in value
        ):
            return None, [], "CONDITION_FIELD_REQUIRES_REPAIR"
        observed.append((key, value))
    if not observed:
        return None, [], "CONDITION_FIELD_MISSING"
    values = {value for _key, value in observed}
    if len(values) != 1:
        return None, [key for key, _value in observed], "CONDITION_FIELD_CONFLICT"
    if not string_only:
        try:
            numeric = Decimal(observed[0][1])
        except InvalidOperation:
            return None, [key for key, _value in observed], "CONDITION_VALUE_INVALID"
        if not numeric.is_finite() or numeric <= 0:
            return None, [key for key, _value in observed], "CONDITION_VALUE_INVALID"
    return observed[0][1], [key for key, _value in observed], None


def _paired_v4_list_source_inventory(
    *,
    source: str,
    container: Mapping[str, object] | None,
    field: str,
) -> tuple[dict[str, object], list[object]]:
    """Attest an unprojected source list without trusting its producer."""

    field_present = isinstance(container, Mapping) and field in container
    observed = container.get(field) if field_present else None
    list_typed = isinstance(observed, list)
    rows = list(observed) if list_typed else []
    non_mapping_indices = [
        index
        for index, item in enumerate(rows)
        if not isinstance(item, Mapping)
    ]
    body: dict[str, object] = {
        "schema_version": "exact_assay_condition_list_source_v1",
        "source": source,
        "field": field,
        "field_present": field_present,
        "list_typed": list_typed,
        "observed_type": type(observed).__name__ if field_present else None,
        "row_count": len(rows) if list_typed else None,
        "mapping_row_count": (
            sum(isinstance(item, Mapping) for item in rows)
            if list_typed
            else None
        ),
        "non_mapping_row_indices": non_mapping_indices,
        "rows_sha256": _paired_v4_hash(rows) if list_typed else None,
        "observed_value_sha256": (
            _paired_v4_hash(observed) if field_present else None
        ),
        "complete": bool(
            field_present and list_typed and not non_mapping_indices
        ),
    }
    return {
        **body,
        "inventory_sha256": _paired_v4_hash(body),
    }, rows


def _paired_v4_valid_list_source_inventory(
    raw: object,
    *,
    source: str,
    field: str,
) -> dict[str, object] | None:
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {
            "schema_version",
            "source",
            "field",
            "field_present",
            "list_typed",
            "observed_type",
            "row_count",
            "mapping_row_count",
            "non_mapping_row_indices",
            "rows_sha256",
            "observed_value_sha256",
            "complete",
            "inventory_sha256",
        }
        or not _paired_v4_self_hash(raw, hash_key="inventory_sha256")
        or raw.get("schema_version")
        != "exact_assay_condition_list_source_v1"
        or raw.get("source") != source
        or raw.get("field") != field
    ):
        return None
    field_present = raw.get("field_present")
    list_typed = raw.get("list_typed")
    row_count = raw.get("row_count")
    mapping_count = raw.get("mapping_row_count")
    non_mapping = raw.get("non_mapping_row_indices")
    rows_hash = raw.get("rows_sha256")
    observed_hash = raw.get("observed_value_sha256")
    complete = raw.get("complete")
    if (
        type(field_present) is not bool
        or type(list_typed) is not bool
        or type(complete) is not bool
        or not isinstance(non_mapping, list)
        or any(type(index) is not int or index < 0 for index in non_mapping)
        or non_mapping != sorted(set(non_mapping))
    ):
        return None
    if list_typed:
        if (
            type(row_count) is not int
            or row_count < 0
            or type(mapping_count) is not int
            or mapping_count < 0
            or mapping_count > row_count
            or mapping_count != row_count - len(non_mapping)
            or any(index >= row_count for index in non_mapping)
            or not isinstance(rows_hash, str)
            or not _SHA256.fullmatch(rows_hash)
            or observed_hash != rows_hash
            or raw.get("observed_type") != "list"
        ):
            return None
    elif (
        row_count is not None
        or mapping_count is not None
        or non_mapping
        or rows_hash is not None
        or (
            field_present
            and (
                not isinstance(raw.get("observed_type"), str)
                or not raw.get("observed_type")
                or not isinstance(observed_hash, str)
                or not _SHA256.fullmatch(observed_hash)
            )
        )
        or (
            not field_present
            and (
                raw.get("observed_type") is not None
                or observed_hash is not None
            )
        )
    ):
        return None
    if complete is not bool(field_present and list_typed and not non_mapping):
        return None
    return dict(raw)


def _paired_v4_recompute_condition_binding(
    *,
    evidence_metadata: object,
    assay_record: object,
) -> dict[str, object] | None:
    """Rebuild the exact two-source condition union from deposited records."""

    metadata = evidence_metadata if isinstance(evidence_metadata, dict) else None
    context_record = assay_record if isinstance(assay_record, dict) else None
    invalid_records: list[dict[str, object]] = []
    activity_inventory, property_rows = _paired_v4_list_source_inventory(
        source="activity_properties",
        container=metadata,
        field="activity_properties",
    )
    parameter_source = "chembl_assay_context.record.assay_parameters"
    parameter_attestation = (
        context_record.get("assay_parameters_source_inventory")
        if context_record is not None
        else None
    )
    parameter_inventory = _paired_v4_valid_list_source_inventory(
        parameter_attestation,
        source=parameter_source,
        field="assay_parameters",
    )
    parameter_attestation_valid = parameter_inventory is not None
    if parameter_inventory is None:
        parameter_inventory, _unused_rows = _paired_v4_list_source_inventory(
            source=parameter_source,
            container=None,
            field="assay_parameters",
        )
        invalid_records.append(
            {
                "source": parameter_source,
                "source_index": None,
                "reason_codes": [
                    "CONDITION_SOURCE_INVENTORY_MISSING_OR_INVALID"
                ],
                "present_keys": [],
            }
        )
    parameter_rows = (
        list(context_record.get("assay_parameters"))
        if context_record is not None
        and isinstance(context_record.get("assay_parameters"), list)
        else []
    )
    source_inventory = [activity_inventory, parameter_inventory]
    sources: list[tuple[str, list[object]]] = [
        ("activity_properties", property_rows),
        (parameter_source, parameter_rows),
    ]
    for inventory_record in source_inventory:
        source = str(inventory_record.get("source") or "")
        field = str(inventory_record.get("field") or "")
        if inventory_record.get("field_present") is not True:
            invalid_records.append(
                {
                    "source": source,
                    "source_index": None,
                    "reason_codes": ["CONDITION_SOURCE_FIELD_MISSING"],
                    "present_keys": [],
                }
            )
        elif inventory_record.get("list_typed") is not True:
            invalid_records.append(
                {
                    "source": source,
                    "source_index": None,
                    "reason_codes": ["CONDITION_SOURCE_FIELD_NOT_LIST"],
                    "present_keys": [field],
                }
            )
        else:
            invalid_records.extend(
                {
                    "source": source,
                    "source_index": index,
                    "reason_codes": ["CONDITION_SOURCE_ROW_NOT_OBJECT"],
                    "present_keys": [],
                }
                for index in inventory_record.get(
                    "non_mapping_row_indices", []
                )
            )
    if (
        parameter_attestation_valid
        and parameter_inventory.get("list_typed") is True
        and (
            parameter_inventory.get("row_count") != len(parameter_rows)
            or parameter_inventory.get("rows_sha256")
            != _paired_v4_hash(parameter_rows)
            or parameter_inventory.get("observed_value_sha256")
            != _paired_v4_hash(parameter_rows)
        )
    ):
        invalid_records.append(
            {
                "source": parameter_source,
                "source_index": None,
                "reason_codes": [
                    "CONDITION_SOURCE_PROJECTION_MISMATCH"
                ],
                "present_keys": ["assay_parameters"],
            }
        )
    if context_record is not None:
        parameters = context_record.get("assay_parameters")
        parameter_count = context_record.get("assay_parameter_count")
        if (
            isinstance(parameters, list)
            and (
                type(parameter_count) is not int
                or parameter_count != len(parameters)
            )
        ):
            return None
    exact_records: list[dict[str, object]] = []
    if (
        context_record is not None
        and context_record.get("assay_parameters_truncated") is not False
    ):
        invalid_records.append(
            {
                "source": "chembl_assay_context.record.assay_parameters",
                "source_index": None,
                "reason_codes": [
                    "ASSAY_PARAMETERS_COMPLETENESS_UNATTESTED"
                ],
                "present_keys": [],
            }
        )
    for source, rows in sources:
        for index, item in enumerate(rows):
            if not isinstance(item, Mapping):
                continue
            subject, subject_fields, subject_error = (
                _paired_v4_condition_component(
                    item,
                    ("standard_type", "type"),
                    string_only=True,
                )
            )
            value, value_fields, value_error = (
                _paired_v4_condition_component(
                    item,
                    ("standard_value", "value"),
                    string_only=False,
                )
            )
            unit, unit_fields, unit_error = _paired_v4_condition_component(
                item,
                ("standard_units", "units"),
                string_only=True,
            )
            errors = [
                error
                for error in (subject_error, value_error, unit_error)
                if error is not None
            ]
            if errors:
                invalid_records.append(
                    {
                        "source": source,
                        "source_index": index,
                        "reason_codes": errors,
                        "present_keys": sorted(
                            _PAIRED_V4_CONDITION_KEYS.intersection(item)
                        ),
                    }
                )
                continue
            if subject is None or value is None or unit is None:
                return None
            exact_records.append(
                {
                    "source": source,
                    "source_index": index,
                    "subject": subject,
                    "value_raw": value,
                    "unit_raw": unit,
                    "subject_fields": subject_fields,
                    "value_fields": value_fields,
                    "unit_fields": unit_fields,
                }
            )
    unique_conditions = sorted(
        {
            (
                str(record["subject"]),
                str(record["value_raw"]),
                str(record["unit_raw"]),
            )
            for record in exact_records
        }
    )
    if invalid_records:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_RECORD_INCOMPLETE_OR_CONFLICTING"
    elif not unique_conditions:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_MISSING"
    elif len(unique_conditions) != 1:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_AMBIGUOUS_OR_CONFLICTING"
    else:
        status = "exact"
        reason_code = None
    selected = unique_conditions[0] if status == "exact" else None
    body: dict[str, object] = {
        "schema_version": _PAIRED_V4_CONDITION_BINDING_SCHEMA,
        "status": status,
        "reason_code": reason_code,
        "source_inventory_schema_version": (
            "exact_assay_condition_source_inventory_v1"
        ),
        "source_inventory_count": len(source_inventory),
        "source_inventory": source_inventory,
        "source_inventory_complete": all(
            item["complete"] is True for item in source_inventory
        ),
        "source_inventory_sha256": _paired_v4_hash(source_inventory),
        "subject": selected[0] if selected is not None else None,
        "value_raw": selected[1] if selected is not None else None,
        "unit_raw": selected[2] if selected is not None else None,
        "unique_condition_count": len(unique_conditions),
        "unique_conditions": [
            {"subject": subject, "value_raw": value, "unit_raw": unit}
            for subject, value, unit in unique_conditions
        ],
        "source_record_count": len(exact_records),
        "source_records": exact_records,
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records,
        "normalization_applied": False,
        "case_repair_applied": False,
        "whitespace_repair_applied": False,
        "unit_conversion_applied": False,
        "missing_values_imputed": False,
    }
    body["binding_sha256"] = _paired_v4_hash(body)
    return body


def _paired_v4_valid_condition_binding(
    raw: object,
    *,
    exact_required: bool,
) -> dict[str, object] | None:
    """Validate a self-contained binding and all of its summary inventories."""

    if (
        not isinstance(raw, dict)
        or not _paired_v4_self_hash(raw, hash_key="binding_sha256")
        or raw.get("schema_version")
        != _PAIRED_V4_CONDITION_BINDING_SCHEMA
        or raw.get("normalization_applied") is not False
        or raw.get("case_repair_applied") is not False
        or raw.get("whitespace_repair_applied") is not False
        or raw.get("unit_conversion_applied") is not False
        or raw.get("missing_values_imputed") is not False
    ):
        return None
    source_inventory = raw.get("source_inventory")
    if (
        raw.get("source_inventory_schema_version")
        != "exact_assay_condition_source_inventory_v1"
        or not isinstance(source_inventory, list)
        or len(source_inventory) != 2
        or type(raw.get("source_inventory_count")) is not int
        or raw.get("source_inventory_count") != len(source_inventory)
        or raw.get("source_inventory_sha256")
        != _paired_v4_hash(source_inventory)
        or type(raw.get("source_inventory_complete")) is not bool
    ):
        return None
    expected_inventory_sources = (
        ("activity_properties", "activity_properties"),
        (
            "chembl_assay_context.record.assay_parameters",
            "assay_parameters",
        ),
    )
    recomputed_inventory_complete = True
    source_row_counts: dict[str, int] = {}
    for item, (expected_source, expected_field) in zip(
        source_inventory,
        expected_inventory_sources,
        strict=True,
    ):
        verified_inventory = _paired_v4_valid_list_source_inventory(
            item,
            source=expected_source,
            field=expected_field,
        )
        if verified_inventory is None:
            return None
        recomputed_inventory_complete = (
            recomputed_inventory_complete
            and verified_inventory.get("complete") is True
        )
        row_count = verified_inventory.get("row_count")
        source_row_counts[expected_source] = (
            row_count if type(row_count) is int else 0
        )
    if (
        raw.get("source_inventory_complete")
        is not recomputed_inventory_complete
        or (exact_required and not recomputed_inventory_complete)
    ):
        return None
    source_records = raw.get("source_records")
    invalid_records = raw.get("invalid_records")
    unique_conditions = raw.get("unique_conditions")
    if (
        not isinstance(source_records, list)
        or not isinstance(invalid_records, list)
        or not isinstance(unique_conditions, list)
        or type(raw.get("source_record_count")) is not int
        or raw.get("source_record_count") != len(source_records)
        or type(raw.get("invalid_record_count")) is not int
        or raw.get("invalid_record_count") != len(invalid_records)
        or type(raw.get("unique_condition_count")) is not int
        or raw.get("unique_condition_count") != len(unique_conditions)
    ):
        return None
    observed_conditions: set[tuple[str, str, str]] = set()
    observed_sources: set[tuple[str, int]] = set()
    for record in source_records:
        if not isinstance(record, dict) or set(record) != {
            "source",
            "source_index",
            "subject",
            "value_raw",
            "unit_raw",
            "subject_fields",
            "value_fields",
            "unit_fields",
        }:
            return None
        source = record.get("source")
        source_index = record.get("source_index")
        subject = record.get("subject")
        value = record.get("value_raw")
        unit = record.get("unit_raw")
        subject_fields = record.get("subject_fields")
        value_fields = record.get("value_fields")
        unit_fields = record.get("unit_fields")
        if (
            not isinstance(source, str)
            or source not in _PAIRED_V4_CONDITION_SOURCES
            or type(source_index) is not int
            or source_index < 0
            or source_index >= source_row_counts.get(source, 0)
            or (source, source_index) in observed_sources
            or not all(
                isinstance(item, str)
                and item
                and item == item.strip()
                and "\r" not in item
                and "\n" not in item
                for item in (subject, value, unit)
            )
            or not isinstance(subject_fields, list)
            or not subject_fields
            or any(
                not isinstance(item, str)
                or item not in {"standard_type", "type"}
                for item in subject_fields
            )
            or not isinstance(value_fields, list)
            or not value_fields
            or any(
                not isinstance(item, str)
                or item not in {"standard_value", "value"}
                for item in value_fields
            )
            or not isinstance(unit_fields, list)
            or not unit_fields
            or any(
                not isinstance(item, str)
                or item not in {"standard_units", "units"}
                for item in unit_fields
            )
        ):
            return None
        try:
            numeric = Decimal(str(value))
        except InvalidOperation:
            return None
        if not numeric.is_finite() or numeric <= 0:
            return None
        observed_sources.add((str(source), source_index))
        observed_conditions.add((str(subject), str(value), str(unit)))
    invalid_sources: set[tuple[str, int | None, tuple[str, ...]]] = set()
    for record in invalid_records:
        if not isinstance(record, dict) or set(record) != {
            "source",
            "source_index",
            "reason_codes",
            "present_keys",
        }:
            return None
        source = record.get("source")
        source_index = record.get("source_index")
        reasons = record.get("reason_codes")
        present_keys = record.get("present_keys")
        if (
            not isinstance(source, str)
            or source not in _PAIRED_V4_CONDITION_SOURCES
            or (
                source_index is not None
                and (
                    type(source_index) is not int
                    or source_index < 0
                    or source_index >= source_row_counts.get(source, 0)
                )
            )
            or not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(item, str) or not item for item in reasons)
            or not isinstance(present_keys, list)
            or any(
                not isinstance(item, str)
                or item not in _PAIRED_V4_CONDITION_KEYS
                for item in present_keys
            )
        ):
            return None
        source_key = (source, source_index, tuple(reasons))
        if (
            source_key in invalid_sources
            or present_keys != sorted(set(present_keys))
            or (
                source_index is None
                and reasons
                not in (
                    ["ASSAY_PARAMETERS_COMPLETENESS_UNATTESTED"],
                    ["CONDITION_SOURCE_FIELD_MISSING"],
                    ["CONDITION_SOURCE_FIELD_NOT_LIST"],
                    ["CONDITION_SOURCE_INVENTORY_MISSING_OR_INVALID"],
                    ["CONDITION_SOURCE_PROJECTION_MISMATCH"],
                )
            )
        ):
            return None
        invalid_sources.add(source_key)
    expected_unique = [
        {"subject": subject, "value_raw": value, "unit_raw": unit}
        for subject, value, unit in sorted(observed_conditions)
    ]
    if unique_conditions != expected_unique:
        return None
    if invalid_records:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_RECORD_INCOMPLETE_OR_CONFLICTING"
    elif not expected_unique:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_MISSING"
    elif len(expected_unique) != 1:
        status = "unverified"
        reason_code = "ASSAY_CONDITION_AMBIGUOUS_OR_CONFLICTING"
    else:
        status = "exact"
        reason_code = None
    selected = expected_unique[0] if status == "exact" else None
    if (
        raw.get("status") != status
        or raw.get("reason_code") != reason_code
        or raw.get("subject")
        != (selected["subject"] if selected is not None else None)
        or raw.get("value_raw")
        != (selected["value_raw"] if selected is not None else None)
        or raw.get("unit_raw")
        != (selected["unit_raw"] if selected is not None else None)
        or (exact_required and status != "exact")
    ):
        return None
    return dict(raw)


def _paired_v4_values(value: object) -> list[object] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _paired_v4_strings(
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[str] | None:
    raw = _paired_v4_values(value)
    if raw is None or any(not isinstance(item, str) for item in raw):
        return None
    result = [str(item) for item in raw]
    if len(result) < minimum or (
        maximum is not None and len(result) > maximum
    ):
        return None
    return result


def _paired_v4_normalized_publication_types(
    value: object,
) -> tuple[str, ...] | None:
    raw = _paired_v4_strings(value, minimum=1)
    if raw is None:
        return None
    normalized = tuple(
        sorted(
            {
                _normalize_primary_phrase(item)
                for item in raw
                if _normalize_primary_phrase(item)
            }
        )
    )
    if (
        not normalized
        or any(_PAIRED_V4_EXCLUDED_TYPE.search(item) for item in normalized)
        or not set(normalized).intersection(_PAIRED_V4_PRIMARY_TYPES)
    ):
        return None
    return normalized


def _paired_v4_span(
    raw: object,
    *,
    sentence: str,
) -> tuple[int, int, str] | None:
    if not isinstance(raw, dict):
        return None
    start = raw.get("start")
    end = raw.get("end")
    text = raw.get("text")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not isinstance(text, str)
        or start < 0
        or end <= start
        or end > len(sentence)
        or sentence[start:end] != text
    ):
        return None
    return start, end, text


def _paired_v4_exact_occurrences(
    sentence: str,
    value: str,
) -> list[tuple[int, int]]:
    if not value:
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(
            rf"(?<![\w-]){re.escape(value)}(?![\w-])",
            sentence,
            flags=re.IGNORECASE,
        )
    ]


def _paired_v4_parameter_hash(params: Mapping[str, object]) -> str:
    return _paired_v4_hash({"params": dict(params)})


def _paired_v4_gateway_parameter_hash(
    params: Mapping[str, object],
) -> str:
    encoded = json.dumps(
        dict(params),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _paired_v4_verify_discovery_bounds(
    raw: object,
) -> dict[str, object] | None:
    """Recompute the pre-activity phase-descending fixed top-12 pool."""

    if (
        not isinstance(raw, dict)
        or raw.get("schema_version")
        != "bounded_mechanism_first_candidate_identity_pool_v1"
        or raw.get("discovery_lane")
        != "chembl_mechanism_first_single_default_v1"
        or raw.get("target_type") != "SINGLE PROTEIN"
        or raw.get("target_organism") != "Homo sapiens"
        or raw.get("mechanism_action_type") != "INHIBITOR"
        or raw.get("raw_mechanism_row_limit") != 80
        or raw.get("mechanism_complete_within_bound") is not True
        or raw.get("prewarning_pool_limit")
        != _PAIRED_V4_MAX_CANDIDATES
        or raw.get("prewarning_selection_policy")
        != "clinical_max_phase_desc_then_parent_molecule_chembl_id_v1"
        or raw.get("selection_inputs")
        != ["max_phase", "parent_molecule_chembl_id"]
        or raw.get("pchembl_used_for_prewarning_selection") is not False
        or raw.get("primary_document_used_for_prewarning_selection")
        is not False
        or raw.get("warning_used_for_prewarning_selection") is not False
        or raw.get("activity_used_for_prewarning_selection") is not False
        or raw.get("activity_candidate_query_limit")
        != _PAIRED_V4_MAX_CANDIDATES
        or raw.get("no_activity_first_fallback") is not True
        or raw.get("no_backfill_beyond_prewarning_pool") is not True
        or raw.get("returned_candidate_ordering_policy")
        != (
            "surviving_fixed_prewarning_clinical_phase_desc_then_"
            "stable_parent_molecule_chembl_id_subsequence_v1"
        )
        or raw.get("fixture") is not False
        or raw.get("missing_values_imputed") is not False
    ):
        return None
    target_chembl_id = str(raw.get("target_chembl_id") or "").strip()
    mechanism_query = raw.get("mechanism_query")
    mechanism_params: dict[str, object] = {
        "target_chembl_id": target_chembl_id,
        "action_type": "INHIBITOR",
        "max_phase__gte": 1,
        "direct_interaction": 1,
        "molecular_mechanism": 1,
        "order_by": "molecule_chembl_id",
        "limit": 80,
    }
    if (
        not target_chembl_id
        or not isinstance(mechanism_query, dict)
        or mechanism_query.get("source") != "ChEMBL"
        or mechanism_query.get("endpoint")
        != _PAPER_NATIVE_V5_CHEMBL_MECHANISM_ENDPOINT
        or mechanism_query.get("parameter_keys")
        != sorted(mechanism_params)
        or mechanism_query.get("parameter_hash")
        != _paired_v4_gateway_parameter_hash(mechanism_params)
        or mechanism_query.get("status_code") != 200
        or not _SHA256.fullmatch(
            str(mechanism_query.get("response_sha256") or "")
        )
    ):
        return None
    response_count = raw.get("raw_mechanism_response_count")
    total_count = raw.get("mechanism_total_count")
    if (
        type(response_count) is not int
        or not 0 <= response_count <= 80
        or type(total_count) is not int
        or total_count != response_count
        or not _SHA256.fullmatch(
            str(raw.get("raw_mechanism_rows_sha256") or "")
        )
    ):
        return None
    eligible_ids = _paired_v4_strings(raw.get("eligible_identity_ids"))
    prewarning_ids = _paired_v4_strings(raw.get("prewarning_pool_ids"))
    preactivity_ids = _paired_v4_strings(raw.get("preactivity_pool_ids"))
    activity_attempted_ids = _paired_v4_strings(
        raw.get("activity_query_attempted_ids")
    )
    gateway_return_ids = _paired_v4_strings(
        raw.get("gateway_return_candidate_ids"),
        minimum=1,
    )
    downstream_ids = _paired_v4_strings(
        raw.get("downstream_candidate_identity_ids"),
        minimum=1,
    )
    resolution = raw.get("identity_resolution")
    preactivity_pool = raw.get("preactivity_pool")
    if (
        eligible_ids is None
        or prewarning_ids is None
        or preactivity_ids is None
        or activity_attempted_ids is None
        or gateway_return_ids is None
        or downstream_ids is None
        or not _paired_v4_unique_identity_strings(eligible_ids)
        or not _paired_v4_unique_identity_strings(prewarning_ids)
        or not _paired_v4_unique_identity_strings(gateway_return_ids)
        or not _paired_v4_unique_identity_strings(downstream_ids)
        or not isinstance(resolution, list)
        or not isinstance(preactivity_pool, list)
        or raw.get("eligible_identity_count") != len(eligible_ids)
        or raw.get("prewarning_pool_count") != len(prewarning_ids)
        or raw.get("preactivity_pool_count") != len(prewarning_ids)
        or raw.get("activity_query_attempted_count")
        != len(prewarning_ids)
        or len(prewarning_ids) > _PAIRED_V4_MAX_CANDIDATES
        or preactivity_ids != prewarning_ids
        or activity_attempted_ids != prewarning_ids
        or len(preactivity_pool) != len(prewarning_ids)
        or type(raw.get("requested_eligible_count")) is not int
        or not (
            1
            <= raw.get("requested_eligible_count")
            <= _PAIRED_V4_MAX_CANDIDATES
        )
        or raw.get("fixed_prewarning_pool_limit")
        != _PAIRED_V4_MAX_CANDIDATES
        or raw.get("gateway_return_candidate_count")
        != len(gateway_return_ids)
        or raw.get("downstream_candidate_identity_count")
        != len(downstream_ids)
        or gateway_return_ids != downstream_ids
        or raw.get("gateway_return_cap_applied") is not False
        or raw.get("eligible_quota_evaluation_owner") != "MA2"
        or raw.get("gateway_returned_from_fixed_prewarning_pool")
        is not True
        or raw.get(
            "gateway_return_order_matches_fixed_prewarning_subsequence"
        )
        is not True
    ):
        return None
    phase_by_id: dict[str, int] = {}
    resolution_by_id: dict[str, dict[str, object]] = {}
    for item in resolution:
        if not isinstance(item, dict):
            return None
        molecule_id = str(
            item.get("parent_molecule_chembl_id") or ""
        ).strip()
        phase = item.get("max_phase")
        if molecule_id in eligible_ids:
            if (
                not molecule_id
                or molecule_id in phase_by_id
                or type(phase) is not int
                or not 1 <= phase <= 4
            ):
                return None
            phase_by_id[molecule_id] = phase
        if molecule_id in prewarning_ids:
            if molecule_id in resolution_by_id:
                return None
            resolution_by_id[molecule_id] = item
    if set(phase_by_id) != set(eligible_ids):
        return None
    expected_eligible = sorted(
        eligible_ids,
        key=lambda molecule_id: (-phase_by_id[molecule_id], molecule_id),
    )
    expected_prewarning = expected_eligible[:_PAIRED_V4_MAX_CANDIDATES]
    expected_pool = [
        {
            "rank": rank,
            "parent_molecule_chembl_id": molecule_id,
            "max_phase": phase_by_id[molecule_id],
        }
        for rank, molecule_id in enumerate(expected_prewarning, start=1)
    ]
    expected_gateway_return = [
        molecule_id
        for molecule_id in expected_prewarning
        if resolution_by_id.get(molecule_id, {}).get("downstream_status")
        == "evidence_resolution_eligible"
    ]
    if (
        eligible_ids != expected_eligible
        or prewarning_ids != expected_prewarning
        or preactivity_pool != expected_pool
        or set(resolution_by_id) != set(expected_prewarning)
        or any(
            resolution_by_id[molecule_id].get("warning_screen_status")
            != "resolved"
            or "WARNING_AND_STRUCTURE_STATUS_RESOLVED"
            not in resolution_by_id[molecule_id].get("reason_codes", [])
            for molecule_id in expected_gateway_return
        )
        or gateway_return_ids != expected_gateway_return
    ):
        return None
    return {
        "target_chembl_id": target_chembl_id,
        "prewarning_ids": prewarning_ids,
        "gateway_return_ids": gateway_return_ids,
    }


def _paired_v4_inventory(
    raw: object,
    *,
    discovery_bounds: object,
) -> dict[str, object] | None:
    verified_bounds = _paired_v4_verify_discovery_bounds(discovery_bounds)
    if (
        not isinstance(raw, dict)
        or verified_bounds is None
        or not _paired_v4_self_hash(raw, hash_key="inventory_sha256")
        or raw.get("schema_version")
        != "primary_abstract_paired_v4_query_inventory_v1"
        or raw.get("complete") is not True
        or raw.get("truncated") is not False
        or raw.get("fixture") is not False
        or raw.get("max_candidates") != _PAIRED_V4_MAX_CANDIDATES
        or raw.get("max_documents") != _PAIRED_V4_MAX_DOCUMENTS
    ):
        return None
    target_aliases = _paired_v4_strings(
        raw.get("target_aliases"), minimum=1, maximum=8
    )
    endpoint = raw.get("endpoint")
    records = raw.get("records")
    if (
        target_aliases is None
        or not isinstance(endpoint, str)
        or not endpoint
        or endpoint != endpoint.strip()
        or not isinstance(records, list)
        or not records
        or len(records) > _PAIRED_V4_MAX_CANDIDATES
    ):
        return None
    if any(
        not item
        or item != item.strip()
        or len(item) > 160
        or "\r" in item
        or "\n" in item
        for item in target_aliases
    ):
        return None
    target_keys = {_paired_v4_alias_key(item) for item in target_aliases}
    if len(target_keys) != len(target_aliases):
        return None
    record_ids: list[str] = []
    record_id_keys: set[str] = set()
    aliases_by_id: dict[str, list[str]] = {}
    preferred_by_id: dict[str, str] = {}
    global_alias_owners: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            return None
        molecule_id = str(record.get("molecule_chembl_id") or "").strip()
        preferred = record.get("preferred_search_alias")
        aliases = _paired_v4_strings(
            record.get("exact_aliases"),
            minimum=1,
            maximum=_PAIRED_V4_MAX_ALIASES,
        )
        molecule_key = _paired_v4_alias_key(molecule_id)
        alias_keys = (
            [_paired_v4_alias_key(item) for item in aliases]
            if aliases is not None
            else []
        )
        if (
            not molecule_id
            or molecule_id in aliases_by_id
            or molecule_key in record_id_keys
            or not isinstance(preferred, str)
            or not preferred
            or aliases is None
            or any(
                not item
                or item != item.strip()
                or len(item) > 160
                or "\r" in item
                or "\n" in item
                for item in aliases
            )
            or len(set(alias_keys)) != len(aliases)
            or _paired_v4_alias_key(preferred) not in set(alias_keys)
        ):
            return None
        for alias in aliases:
            key = _paired_v4_alias_key(alias)
            if not key or key in target_keys or key in global_alias_owners:
                return None
            global_alias_owners[key] = molecule_id
        record_ids.append(molecule_id)
        record_id_keys.add(molecule_key)
        aliases_by_id[molecule_id] = aliases
        preferred_by_id[molecule_id] = preferred
    if raw.get("candidate_alias_conflict_ids") != []:
        return None
    if not isinstance(discovery_bounds, dict):
        return None
    direct_ids = _paired_v4_strings(
        discovery_bounds.get("paired_v4_direct_exact_pool_ids")
    )
    prewarning_ids = _paired_v4_strings(
        discovery_bounds.get("prewarning_pool_ids")
    )
    inventory_direct_ids = _paired_v4_strings(
        raw.get("paired_v4_direct_exact_pool_ids")
    )
    inventory_prewarning_ids = _paired_v4_strings(
        raw.get("prewarning_pool_ids")
    )
    cohort = raw.get("pre_pubmed_frozen_cohort")
    selected_ids = (
        _paired_v4_strings(cohort.get("selected_candidate_ids"))
        if isinstance(cohort, dict)
        else None
    )
    queried_ids = _paired_v4_strings(raw.get("queried_candidate_ids"))
    fixed_alias_records = raw.get("fixed_pool_alias_records")
    projected_ids = _paired_v4_strings(raw.get("bounded_alias_projection_ids"))
    if (
        direct_ids is None
        or prewarning_ids is None
        or inventory_direct_ids is None
        or inventory_prewarning_ids is None
        or selected_ids is None
        or queried_ids is None
        or projected_ids is None
        or not _paired_v4_unique_identity_strings(direct_ids)
        or not _paired_v4_unique_identity_strings(prewarning_ids)
        or not _paired_v4_unique_identity_strings(inventory_direct_ids)
        or not _paired_v4_unique_identity_strings(inventory_prewarning_ids)
        or not _paired_v4_unique_identity_strings(selected_ids)
        or not _paired_v4_unique_identity_strings(queried_ids)
        or not _paired_v4_unique_identity_strings(projected_ids)
        or prewarning_ids != verified_bounds.get("prewarning_ids")
        or not isinstance(fixed_alias_records, list)
        or raw.get("fixed_pool_alias_inventory_complete") is not True
        or raw.get("fixed_pool_alias_record_count")
        != len(fixed_alias_records)
        or raw.get("fixed_pool_alias_records_sha256")
        != _paired_v4_hash(fixed_alias_records)
        or raw.get("records_scope")
        != "pre_pubmed_frozen_cohort_query_subset"
        or raw.get("alias_projection_policy")
        != "first_16_exact_chembl_aliases_in_source_order_v1"
        or discovery_bounds.get("paired_v4_direct_exact_pool_count")
        != len(direct_ids)
        or discovery_bounds.get("prewarning_pool_count")
        != len(prewarning_ids)
        or raw.get("paired_v4_direct_exact_pool_count") != len(direct_ids)
        or raw.get("prewarning_pool_count") != len(prewarning_ids)
        or inventory_direct_ids != direct_ids
        or inventory_prewarning_ids != prewarning_ids
        or record_ids != selected_ids
        or queried_ids
        != sorted(
            selected_ids,
            key=lambda item: (_paired_v4_alias_key(item), item),
        )
        or raw.get("queried_candidate_count") != len(queried_ids)
        or not set(direct_ids).issubset(set(prewarning_ids))
        or not set(direct_ids).issubset(
            set(verified_bounds.get("gateway_return_ids", []))
        )
        or not set(selected_ids).issubset(set(direct_ids))
    ):
        return None
    fixed_ids: list[str] = []
    fixed_id_keys: set[str] = set()
    fixed_by_id: dict[str, dict[str, object]] = {}
    full_alias_owners: dict[str, str] = {}
    recomputed_projected_ids: list[str] = []
    prewarning_rank = {
        molecule_id: rank
        for rank, molecule_id in enumerate(prewarning_ids, start=1)
    }
    for record in fixed_alias_records:
        if not isinstance(record, dict):
            return None
        molecule_id = str(record.get("molecule_chembl_id") or "")
        aliases = _paired_v4_strings(
            record.get("exact_aliases"),
            minimum=1,
            maximum=_PAIRED_V4_MAX_ALIASES,
        )
        preferred = record.get("preferred_search_alias")
        source_count = record.get("source_alias_count")
        retained_ranks = record.get("direct_exact_ic50_retained_ranks")
        molecule_key = _paired_v4_alias_key(molecule_id)
        alias_keys = (
            [_paired_v4_alias_key(item) for item in aliases]
            if aliases is not None
            else []
        )
        if (
            not molecule_id
            or molecule_id in fixed_by_id
            or molecule_key in fixed_id_keys
            or aliases is None
            or not isinstance(preferred, str)
            or any(
                not item
                or item != item.strip()
                or len(item) > 160
                or "\r" in item
                or "\n" in item
                for item in aliases
            )
            or len(set(alias_keys)) != len(aliases)
            or _paired_v4_alias_key(preferred) not in set(alias_keys)
            or isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or not len(aliases) <= source_count <= _MAX_EXACT_CANDIDATE_ALIASES
            or not isinstance(retained_ranks, list)
            or not retained_ranks
            or any(
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 1 <= rank <= _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
                for rank in retained_ranks
            )
            or retained_ranks != sorted(set(retained_ranks))
            or record.get("candidate_alias_conflicts") != []
            or record.get("fixed_pool_rank")
            != prewarning_rank.get(molecule_id)
        ):
            return None
        if source_count > len(aliases):
            recomputed_projected_ids.append(molecule_id)
        for alias in aliases:
            key = _paired_v4_alias_key(alias)
            if not key or key in target_keys or key in full_alias_owners:
                return None
            full_alias_owners[key] = molecule_id
        fixed_ids.append(molecule_id)
        fixed_id_keys.add(molecule_key)
        fixed_by_id[molecule_id] = record
    if (
        fixed_ids != direct_ids
        or recomputed_projected_ids != projected_ids
        or any(
            fixed_by_id.get(molecule_id) != record
            for molecule_id, record in zip(record_ids, records, strict=True)
        )
    ):
        return None
    verified_cohort = _paired_v4_verify_frozen_cohort(
        cohort,
        prewarning_ids=prewarning_ids,
    )
    if (
        verified_cohort is None
        or verified_cohort.get("selected_ids") != selected_ids
        or not set(verified_cohort.get("direct_ids", [])).issubset(
            set(direct_ids)
        )
    ):
        return None
    return {
        "target_aliases": target_aliases,
        "endpoint": endpoint,
        "record_ids": record_ids,
        "aliases_by_id": aliases_by_id,
        "preferred_by_id": preferred_by_id,
        "inventory_sha256": raw["inventory_sha256"],
        "cohort": verified_cohort,
        "direct_ids": direct_ids,
        "prewarning_ids": prewarning_ids,
        "queried_ids": queried_ids,
        "target_chembl_id": verified_bounds["target_chembl_id"],
        "gateway_return_ids": verified_bounds["gateway_return_ids"],
    }


def _paired_v4_expected_query(inventory: Mapping[str, object]) -> str | None:
    target_aliases = inventory.get("target_aliases")
    endpoint = inventory.get("endpoint")
    record_ids = inventory.get("record_ids")
    preferred_by_id = inventory.get("preferred_by_id")
    if (
        not isinstance(target_aliases, list)
        or not isinstance(endpoint, str)
        or not isinstance(record_ids, list)
        or not isinstance(preferred_by_id, dict)
    ):
        return None
    ordered_targets = sorted(
        target_aliases,
        key=lambda item: (_paired_v4_alias_key(item), item),
    )
    ordered_ids = sorted(
        record_ids,
        key=lambda item: (_paired_v4_alias_key(item), item),
    )
    if any(
        '"' in item or any(ord(character) < 32 for character in item)
        for item in [*ordered_targets, endpoint]
    ):
        return None
    preferred: list[str] = []
    for molecule_id in ordered_ids:
        alias = preferred_by_id.get(molecule_id)
        if not isinstance(alias, str) or '"' in alias:
            return None
        preferred.append(alias)
    target_clause = " OR ".join(
        f'"{item}"[Title/Abstract]' for item in ordered_targets
    )
    candidate_clause = " OR ".join(
        f'"{item}"[Title/Abstract]' for item in preferred
    )
    return (
        f"({target_clause}) AND \"{endpoint}\"[Title/Abstract] AND "
        f"({candidate_clause}) NOT Review[Publication Type]"
    )


def _paired_v4_valid_signature(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    condition_binding = _paired_v4_valid_condition_binding(
        raw.get("assay_condition_binding"),
        exact_required=True,
    )
    signature_hash = raw.get("signature_sha256")
    body = dict(raw)
    body.pop("signature_sha256", None)
    if (
        condition_binding is None
        or not isinstance(signature_hash, str)
        or signature_hash != _paired_v4_hash(body)
        or not str(raw.get("assay_chembl_id") or "").strip()
        or not str(raw.get("target_chembl_id") or "").strip()
        or raw.get("endpoint") != "IC50"
        or raw.get("relation") != "="
        or raw.get("unit") not in _KNOWN_ACTIVITY_UNITS
        or not _SHA256.fullmatch(
            str(raw.get("activity_properties_sha256") or "")
        )
        or not isinstance(raw.get("assay_type"), str)
        or not raw.get("assay_type")
    ):
        return None
    return dict(raw)


def _paired_v4_verify_frozen_cohort(
    raw: object,
    *,
    prewarning_ids: list[str],
) -> dict[str, object] | None:
    if (
        not isinstance(raw, dict)
        or not _paired_v4_self_hash(
            raw, hash_key="cohort_attestation_sha256"
        )
        or raw.get("schema_version") != "pre_pubmed_exact_assay_cohort_v1"
        or raw.get("selection_status") != "selected"
        or raw.get("reason_code") is not None
        or raw.get("selection_policy")
        != (
            "max_distinct_fixed_pool_candidate_coverage_then_"
            "signature_sha256_then_assay_id_v1"
        )
        or raw.get("within_candidate_row_policy")
        != "retained_rank_then_activity_id_ascending_v1"
        or raw.get("minimum_distinct_candidate_coverage")
        != _PAIRED_V4_MIN_NONCONTROL_CANDIDATES
        or raw.get("pubmed_or_result_used_for_cohort_selection") is not False
        or raw.get("no_backfill") is not True
        or raw.get("missing_values_imputed") is not False
        or raw.get("fixture") is not False
    ):
        return None
    retained_rows = raw.get("retained_top_k_rows")
    considered = raw.get("considered_signatures")
    selected_rows = raw.get("selected_candidate_rows")
    selected_ids = _paired_v4_strings(
        raw.get("selected_candidate_ids"),
        minimum=_PAIRED_V4_MIN_NONCONTROL_CANDIDATES,
    )
    if (
        not isinstance(retained_rows, list)
        or not retained_rows
        or not isinstance(considered, list)
        or not considered
        or not isinstance(selected_rows, list)
        or selected_ids is None
        or len(selected_ids) != len(set(selected_ids))
        or raw.get("selected_candidate_count") != len(selected_ids)
    ):
        return None
    fixed_rank_by_id = {
        molecule_id: rank
        for rank, molecule_id in enumerate(prewarning_ids, start=1)
    }
    rows_by_signature: dict[str, dict[str, dict[str, object]]] = {}
    direct_ids: set[str] = set()
    for row in retained_rows:
        if not isinstance(row, dict):
            return None
        molecule_id = str(row.get("molecule_chembl_id") or "")
        retained_rank = row.get("retained_rank")
        activity_id = str(row.get("activity_id") or "")
        row_binding = _paired_v4_valid_condition_binding(
            row.get("assay_condition_binding"),
            exact_required=False,
        )
        raw_signature = row.get("exact_assay_signature")
        signature = _paired_v4_valid_signature(raw_signature)
        direct_raw = row.get("direct_exact_gate")
        if row_binding is None or type(direct_raw) is not bool:
            return None
        direct = direct_raw
        if (
            molecule_id not in fixed_rank_by_id
            or row.get("fixed_pool_rank") != fixed_rank_by_id[molecule_id]
            or isinstance(retained_rank, bool)
            or not isinstance(retained_rank, int)
            or not 1 <= retained_rank <= _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
            or not activity_id
        ):
            return None
        typed_value = row.get("typed_value")
        deposited_value = row.get("standard_value")
        exact_tuple = bool(
            row.get("standard_type") == row.get("typed_endpoint") == "IC50"
            and row.get("standard_relation")
            == row.get("typed_relation")
            == "="
            and row.get("standard_units") == row.get("typed_unit")
            and row.get("typed_unit") in _KNOWN_ACTIVITY_UNITS
        )
        try:
            exact_tuple = exact_tuple and Decimal(
                str(typed_value)
            ) == Decimal(str(deposited_value)) > 0
        except (InvalidOperation, TypeError, ValueError):
            exact_tuple = False
        if direct != exact_tuple:
            return None
        if raw_signature is not None and signature is None:
            return None
        if signature is None:
            continue
        if (
            not direct
            or signature.get("assay_condition_binding") != row_binding
            or signature.get("assay_chembl_id")
            != row.get("assay_chembl_id")
            or signature.get("endpoint") != row.get("typed_endpoint")
            or signature.get("relation") != row.get("typed_relation")
            or signature.get("unit") != row.get("typed_unit")
        ):
            return None
        direct_ids.add(molecule_id)
        signature_hash = str(signature["signature_sha256"])
        candidates = rows_by_signature.setdefault(signature_hash, {})
        prior = candidates.get(molecule_id)
        key = (retained_rank, activity_id)
        prior_key = (
            (int(prior["retained_rank"]), str(prior["activity_id"]))
            if prior is not None
            else None
        )
        if prior_key is None or key < prior_key:
            candidates[molecule_id] = {
                "molecule_chembl_id": molecule_id,
                "fixed_pool_rank": fixed_rank_by_id[molecule_id],
                "retained_rank": retained_rank,
                "retained_index": retained_rank - 1,
                "activity_id": activity_id,
                "source_evidence_id": row.get("source_evidence_id"),
                "signature": signature,
            }
    expected_considered: list[dict[str, object]] = []
    for signature_hash, rows_by_id in rows_by_signature.items():
        ordered_rows = [
            rows_by_id[molecule_id]
            for molecule_id in prewarning_ids
            if molecule_id in rows_by_id
        ]
        signature = ordered_rows[0]["signature"]
        expected_considered.append(
            {
                "signature_sha256": signature_hash,
                "assay_chembl_id": signature.get("assay_chembl_id"),
                "selected_assay_signature": signature,
                "distinct_candidate_coverage": len(ordered_rows),
                "candidate_ids": [
                    item["molecule_chembl_id"] for item in ordered_rows
                ],
                "candidate_rows": [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "signature"
                    }
                    for item in ordered_rows
                ],
            }
        )
    expected_considered.sort(
        key=lambda item: (
            -int(item["distinct_candidate_coverage"]),
            str(item["signature_sha256"]),
            str(item.get("assay_chembl_id") or ""),
        )
    )
    if (
        considered != expected_considered
        or raw.get("considered_signature_count") != len(expected_considered)
        or not expected_considered
    ):
        return None
    winner = expected_considered[0]
    if (
        int(winner["distinct_candidate_coverage"])
        < _PAIRED_V4_MIN_NONCONTROL_CANDIDATES
        or raw.get("selected_signature")
        != winner["selected_assay_signature"]
        or raw.get("selected_signature_sha256")
        != winner["signature_sha256"]
        or selected_ids != winner["candidate_ids"]
        or selected_rows != winner["candidate_rows"]
    ):
        return None
    return {
        "selected_signature": winner["selected_assay_signature"],
        "selected_signature_sha256": winner["signature_sha256"],
        "selected_ids": selected_ids,
        "selected_rows": selected_rows,
        "direct_ids": [
            molecule_id
            for molecule_id in prewarning_ids
            if molecule_id in direct_ids
        ],
        "canonical_sha256": _paired_v4_hash(raw),
    }


def _paired_v4_call_matches(
    raw: object,
    *,
    stage: str,
    endpoint: str,
    params: Mapping[str, object],
) -> bool:
    if not isinstance(raw, dict):
        return False
    expected_keys = sorted(f"params.{key}" for key in params)
    keys = _paired_v4_strings(raw.get("parameter_keys"))
    return bool(
        raw.get("stage") == stage
        and raw.get("method") == "GET"
        and raw.get("endpoint") == endpoint
        and keys == expected_keys
        and raw.get("parameter_sha256")
        == _paired_v4_parameter_hash(params)
        and raw.get("status") == "completed"
        and raw.get("status_code") == 200
        and raw.get("error_type") is None
        and _SHA256.fullmatch(str(raw.get("response_sha256") or ""))
    )


def _paired_v4_pair_hash(
    pair: Mapping[str, object],
    *,
    sentence_sha256: str,
) -> str | None:
    candidate_span = pair.get("candidate_span")
    target_span = pair.get("target_span")
    endpoint_span = pair.get("endpoint_span")
    condition = pair.get("assay_condition")
    measurement_span = pair.get("measurement_span")
    if not all(
        isinstance(value, dict)
        for value in (
            candidate_span,
            target_span,
            endpoint_span,
            condition,
            measurement_span,
        )
    ):
        return None
    condition_span = condition.get("span")
    if not isinstance(condition_span, dict):
        return None

    def encoded_span(raw: Mapping[str, object]) -> list[object]:
        return [raw.get("start"), raw.get("end"), raw.get("text")]

    payload = {
        "candidate_id": pair.get("candidate_id"),
        "candidate_span": encoded_span(candidate_span),
        "condition_span": encoded_span(condition_span),
        "condition_subject": condition.get("subject"),
        "condition_unit_raw": condition.get("unit_raw"),
        "condition_value_raw": condition.get("value_raw"),
        "endpoint_span": encoded_span(endpoint_span),
        "measurement_span": encoded_span(measurement_span),
        "pair_index": pair.get("pair_index"),
        "publication_types": tuple(pair.get("publication_types") or ()),
        "relation_basis": "direct_assertive_measurement",
        "relation_normalized": "=",
        "relation_raw": pair.get("relation_raw"),
        "schema": _PAIRED_V4_PAIR_SCHEMA,
        "section": pair.get("section"),
        "sentence_sha256": sentence_sha256,
        "target_span": encoded_span(target_span),
        "unit_raw": pair.get("unit_raw"),
        "value_raw": pair.get("value_raw"),
    }
    return _paired_v4_hash(payload)


def _paired_v4_verify_pair_spans(
    pair: object,
    *,
    sentence: str,
    sentence_sha256: str,
    pair_index: int,
    aliases_by_id: Mapping[str, list[str]],
    target_aliases: list[str],
    endpoint: str,
    publication_types: tuple[str, ...],
    section: str,
) -> dict[str, object] | None:
    if not isinstance(pair, dict):
        return None
    sentence_span = _paired_v4_span(
        pair.get("sentence_span"), sentence=sentence
    )
    candidate_span = _paired_v4_span(
        pair.get("candidate_span"), sentence=sentence
    )
    target_span = _paired_v4_span(
        pair.get("target_span"), sentence=sentence
    )
    endpoint_span = _paired_v4_span(
        pair.get("endpoint_span"), sentence=sentence
    )
    measurement_span = _paired_v4_span(
        pair.get("measurement_span"), sentence=sentence
    )
    value_span = _paired_v4_span(pair.get("value_span"), sentence=sentence)
    unit_span = _paired_v4_span(pair.get("unit_span"), sentence=sentence)
    condition = pair.get("assay_condition")
    if not isinstance(condition, dict):
        return None
    condition_span = _paired_v4_span(
        condition.get("span"), sentence=sentence
    )
    condition_value_span = _paired_v4_span(
        condition.get("value_span"), sentence=sentence
    )
    condition_unit_span = _paired_v4_span(
        condition.get("unit_span"), sentence=sentence
    )
    condition_subject_span = _paired_v4_span(
        condition.get("subject_span"), sentence=sentence
    )
    if any(
        value is None
        for value in (
            sentence_span,
            candidate_span,
            target_span,
            endpoint_span,
            condition_span,
            condition_value_span,
            condition_unit_span,
            condition_subject_span,
            measurement_span,
            value_span,
            unit_span,
        )
    ):
        return None
    assert sentence_span is not None
    assert candidate_span is not None
    assert target_span is not None
    assert endpoint_span is not None
    assert condition_span is not None
    assert condition_value_span is not None
    assert condition_unit_span is not None
    assert condition_subject_span is not None
    assert measurement_span is not None
    assert value_span is not None
    assert unit_span is not None
    candidate_id = str(pair.get("candidate_id") or "")
    candidate_aliases = aliases_by_id.get(candidate_id)
    if candidate_aliases is None:
        return None
    candidate_alias_key = _paired_v4_alias_key(candidate_span[2])
    if (
        sentence_span != (0, len(sentence), sentence)
        or pair.get("sentence_sha256") != sentence_sha256
        or pair.get("pair_index") != pair_index
        or pair.get("pair_role")
        != ("common_head" if pair_index == 0 else "direct_follower")
        or pair.get("candidate_alias") != candidate_span[2]
        or candidate_alias_key
        not in {_paired_v4_alias_key(item) for item in candidate_aliases}
        or _paired_v4_exact_occurrences(sentence, candidate_span[2])
        != [(candidate_span[0], candidate_span[1])]
        or pair.get("target_alias") != target_span[2]
        or _paired_v4_alias_key(target_span[2])
        not in {_paired_v4_alias_key(item) for item in target_aliases}
        or pair.get("endpoint") != endpoint_span[2]
        or _paired_v4_alias_key(endpoint_span[2])
        != _paired_v4_alias_key(endpoint)
        or pair.get("section") != section
        or _paired_v4_normalized_publication_types(
            pair.get("publication_types")
        )
        != publication_types
        or pair.get("relation_normalized") != "="
        or pair.get("relation_basis") != "direct_assertive_measurement"
        or pair.get("value_raw") != value_span[2]
        or pair.get("unit_raw") != unit_span[2]
        or condition.get("value_raw") != condition_value_span[2]
        or condition.get("unit_raw") != condition_unit_span[2]
        or condition.get("subject") != condition_subject_span[2]
        or not (
            condition_span[0]
            <= condition_value_span[0]
            < condition_value_span[1]
            <= condition_span[1]
            and condition_span[0]
            <= condition_unit_span[0]
            < condition_unit_span[1]
            <= condition_span[1]
            and condition_span[0]
            <= condition_subject_span[0]
            < condition_subject_span[1]
            <= condition_span[1]
        )
        or not (
            measurement_span[0]
            <= value_span[0]
            < value_span[1]
            <= unit_span[0]
            < unit_span[1]
            <= measurement_span[1]
        )
    ):
        return None
    relation_raw = pair.get("relation_raw")
    relation_span_raw = pair.get("relation_span")
    if relation_raw is None:
        if relation_span_raw is not None or measurement_span[0] != value_span[0]:
            return None
    elif relation_raw == "=":
        relation_span = _paired_v4_span(
            relation_span_raw, sentence=sentence
        )
        if (
            relation_span is None
            or relation_span[2] != "="
            or relation_span[0] != measurement_span[0]
            or relation_span[1] > value_span[0]
        ):
            return None
    else:
        return None
    try:
        if Decimal(value_span[2]) <= 0 or Decimal(condition_value_span[2]) <= 0:
            return None
    except (InvalidOperation, ValueError):
        return None
    expected_pair_hash = _paired_v4_pair_hash(
        pair, sentence_sha256=sentence_sha256
    )
    if (
        expected_pair_hash is None
        or pair.get("evidence_sha256") != expected_pair_hash
    ):
        return None
    return {
        "candidate_id": candidate_id,
        "candidate_span": candidate_span,
        "target_span": target_span,
        "endpoint_span": endpoint_span,
        "condition": condition,
        "condition_span": condition_span,
        "condition_value_span": condition_value_span,
        "measurement_span": measurement_span,
        "value_span": value_span,
        "unit_span": unit_span,
        "relation_raw": relation_raw,
        "pair": pair,
    }


def _paired_v4_verify_group(
    raw: object,
    *,
    inventory: Mapping[str, object],
    query_sha256: str,
    publication_types: tuple[str, ...],
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    document_id = str(raw.get("document_id") or "")
    sentence = raw.get("sentence")
    section = raw.get("section")
    pairs = raw.get("pairs")
    candidate_ids = _paired_v4_strings(raw.get("candidate_ids"), minimum=3)
    aliases_by_id = inventory.get("aliases_by_id")
    target_aliases = inventory.get("target_aliases")
    endpoint = inventory.get("endpoint")
    if (
        not document_id.isdigit()
        or not isinstance(sentence, str)
        or not sentence
        or sentence != sentence.strip()
        or len(sentence) > _PRIMARY_EXCERPT_MAX_CHARS
        or "\n" in sentence
        or "\r" in sentence
        or section not in {"Abstract", "Results"}
        or not isinstance(pairs, (list, tuple))
        or candidate_ids is None
        or len(candidate_ids) != len(set(candidate_ids))
        or len(pairs) != len(candidate_ids)
        or not isinstance(aliases_by_id, dict)
        or not isinstance(target_aliases, list)
        or not isinstance(endpoint, str)
        or _PAIRED_V4_UNSAFE_SENTENCE.search(sentence)
        or "?" in sentence
        or "!" in sentence
        or ";" in sentence
    ):
        return None
    sentence_core = sentence[:-1] if sentence.endswith(".") else sentence
    if re.search(r"(?<!\d)\.(?!\d)", sentence_core):
        return None
    sentence_sha256 = sha256(sentence.encode("utf-8")).hexdigest()
    if raw.get("sentence_sha256") != sentence_sha256:
        return None
    verified_pairs: list[dict[str, object]] = []
    for index, pair in enumerate(pairs):
        verified = _paired_v4_verify_pair_spans(
            pair,
            sentence=sentence,
            sentence_sha256=sentence_sha256,
            pair_index=index,
            aliases_by_id=aliases_by_id,
            target_aliases=target_aliases,
            endpoint=endpoint,
            publication_types=publication_types,
            section=section,
        )
        if verified is None:
            return None
        verified_pairs.append(verified)
    if [item["candidate_id"] for item in verified_pairs] != candidate_ids:
        return None
    candidate_spans = [item["candidate_span"] for item in verified_pairs]
    if candidate_spans != sorted(candidate_spans, key=lambda item: item[0]):
        return None
    first = verified_pairs[0]
    target_span = first["target_span"]
    endpoint_span = first["endpoint_span"]
    condition = first["condition"]
    if any(
        item["target_span"] != target_span
        or item["endpoint_span"] != endpoint_span
        or item["condition"] != condition
        for item in verified_pairs[1:]
    ):
        return None
    assert isinstance(target_span, tuple)
    assert isinstance(endpoint_span, tuple)
    first_candidate_span = first["candidate_span"]
    condition_span = first["condition_span"]
    first_measurement_span = first["measurement_span"]
    if not all(
        isinstance(value, tuple)
        for value in (
            first_candidate_span,
            condition_span,
            first_measurement_span,
        )
    ):
        return None
    if (
        first_candidate_span[0] != 0
        or not (
            first_candidate_span[1]
            < target_span[0]
            < target_span[1]
            < endpoint_span[0]
            < endpoint_span[1]
            <= condition_span[0]
            < condition_span[1]
            < first_measurement_span[0]
            < first_measurement_span[1]
        )
        or re.match(
            r"^\s+(?:was|is)\s+",
            sentence[first_candidate_span[1] : target_span[0]],
            flags=re.IGNORECASE,
        )
        is None
        or re.search(
            r"\b(?:inhibitor|inhibitory|inhibition)\b",
            sentence[first_candidate_span[1] : endpoint_span[0]],
            flags=re.IGNORECASE,
        )
        is None
    ):
        return None
    connectors = list(
        re.finditer(r",\s+followed\s+by\s+", sentence, flags=re.IGNORECASE)
    )
    if len(connectors) != 1:
        return None
    connector = connectors[0]
    if (
        first_measurement_span[1] >= connector.start()
        or re.fullmatch(
            r"\s*\)",
            sentence[first_measurement_span[1] : connector.start()],
        )
        is None
    ):
        return None
    previous_close = connector.end()
    followers = verified_pairs[1:]
    for follower_index, item in enumerate(followers):
        candidate_span = item["candidate_span"]
        measurement_span = item["measurement_span"]
        if not isinstance(candidate_span, tuple) or not isinstance(
            measurement_span, tuple
        ):
            return None
        gap = sentence[previous_close : candidate_span[0]]
        if follower_index == 0:
            if gap:
                return None
        elif follower_index == len(followers) - 1:
            if re.fullmatch(r",\s+and\s+", gap, flags=re.IGNORECASE) is None:
                return None
        elif re.fullmatch(r",\s+", gap) is None:
            return None
        if re.fullmatch(
            r"\s*\(\s*",
            sentence[candidate_span[1] : measurement_span[0]],
        ) is None:
            return None
        close_match = re.match(r"\s*\)", sentence[measurement_span[1] :])
        if close_match is None:
            return None
        previous_close = measurement_span[1] + close_match.end()
    if re.fullmatch(
        r"(?:\.|,\s+and\s+the\s+most\s+selective\.)?",
        sentence[previous_close:],
        flags=re.IGNORECASE,
    ) is None:
        return None
    alias_ranges: list[tuple[int, int]] = []
    for aliases in aliases_by_id.values():
        if not isinstance(aliases, list):
            return None
        for alias in aliases:
            alias_ranges.extend(_paired_v4_exact_occurrences(sentence, alias))
    alias_ranges.extend(_paired_v4_exact_occurrences(sentence, target_span[2]))
    alias_ranges.append((endpoint_span[0], endpoint_span[1]))
    expected_numbers = {
        (
            first["condition_value_span"][0],
            first["condition_value_span"][1],
        ),
        *{
            (item["value_span"][0], item["value_span"][1])
            for item in verified_pairs
        },
    }
    observed_numbers = {
        (match.start(), match.end())
        for match in _PAIRED_V4_NUMBER.finditer(sentence)
        if not any(
            match.start() < end and start < match.end()
            for start, end in alias_ranges
        )
    }
    if observed_numbers != expected_numbers:
        return None
    for match in _PAIRED_V4_ENDPOINT_MARKER.finditer(sentence):
        if not (
            endpoint_span[0] <= match.start()
            and match.end() <= endpoint_span[1]
        ):
            return None
    expected_group_hash = _paired_v4_hash(
        {
            "candidate_ids": tuple(candidate_ids),
            "document_id": document_id,
            "pair_evidence_sha256": [
                item["pair"]["evidence_sha256"] for item in verified_pairs
            ],
            "query_sha256": query_sha256,
            "schema": _PAIRED_V4_SEARCH_SCHEMA,
            "section": section,
            "sentence_sha256": sentence_sha256,
        }
    )
    if raw.get("group_sha256") != expected_group_hash:
        return None
    return {
        "document_id": document_id,
        "sentence": sentence,
        "sentence_sha256": sentence_sha256,
        "group_sha256": expected_group_hash,
        "candidate_ids": candidate_ids,
        "pairs": verified_pairs,
        "canonical_sha256": _paired_v4_hash(raw),
    }


def _paired_v4_verify_search_outcome(
    raw: object,
    *,
    selected_group: object,
    inventory: Mapping[str, object],
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    expected_query = _paired_v4_expected_query(inventory)
    query = raw.get("query")
    query_sha256 = raw.get("query_sha256")
    record_ids = inventory.get("record_ids")
    if (
        expected_query is None
        or query != expected_query
        or query_sha256 != sha256(expected_query.encode("utf-8")).hexdigest()
        or raw.get("schema_version") != _PAIRED_V4_SEARCH_SCHEMA
        or raw.get("status") != "verified"
        or raw.get("reason_code") is not None
        or raw.get("max_candidates") != _PAIRED_V4_MAX_CANDIDATES
        or raw.get("max_documents") != _PAIRED_V4_MAX_DOCUMENTS
        or raw.get("requested_document_count") != _PAIRED_V4_MAX_DOCUMENTS
        or not isinstance(record_ids, list)
        or raw.get("requested_candidate_count") != len(record_ids)
        or raw.get("queried_candidate_count") != len(record_ids)
    ):
        return None
    records = raw.get("records")
    groups = raw.get("groups")
    calls = raw.get("calls")
    verified_candidate_ids = _paired_v4_strings(
        raw.get("verified_candidate_ids"), minimum=3
    )
    if (
        not isinstance(records, (list, tuple))
        or not records
        or len(records) > _PAIRED_V4_MAX_DOCUMENTS
        or not isinstance(groups, (list, tuple))
        or not groups
        or not isinstance(calls, (list, tuple))
        or len(calls) != 3
        or verified_candidate_ids is None
    ):
        return None
    document_ids: list[str] = []
    eligible_ids: list[str] = []
    publication_types_by_id: dict[str, tuple[str, ...]] = {}
    verified_group_counts: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            return None
        document_id = str(record.get("document_id") or "")
        publication_types = _paired_v4_normalized_publication_types(
            record.get("publication_types")
        )
        rejection_codes = _paired_v4_strings(record.get("rejection_codes"))
        if (
            not document_id.isdigit()
            or document_id in publication_types_by_id
            or rejection_codes is None
            or isinstance(record.get("eligible_section_count"), bool)
            or not isinstance(record.get("eligible_section_count"), int)
            or isinstance(record.get("sentence_count"), bool)
            or not isinstance(record.get("sentence_count"), int)
            or isinstance(record.get("bounded_sentence_count"), bool)
            or not isinstance(record.get("bounded_sentence_count"), int)
            or isinstance(record.get("verified_group_count"), bool)
            or not isinstance(record.get("verified_group_count"), int)
        ):
            return None
        document_ids.append(document_id)
        if publication_types is not None:
            eligible_ids.append(document_id)
            publication_types_by_id[document_id] = publication_types
        if record.get("status") == "verified":
            if publication_types is None or rejection_codes:
                return None
            verified_group_counts[document_id] = int(
                record["verified_group_count"]
            )
        elif record.get("status") != "rejected":
            return None
    returned_count = raw.get("returned_document_count")
    unique_count = raw.get("unique_document_count")
    if (
        isinstance(returned_count, bool)
        or not isinstance(returned_count, int)
        or isinstance(unique_count, bool)
        or not isinstance(unique_count, int)
        or returned_count < unique_count
        or unique_count != len(document_ids)
        or raw.get("screened_document_count") != len(records)
        or len(document_ids) != len(set(document_ids))
    ):
        return None
    search_params: dict[str, object] = {
        "db": "pubmed",
        "term": expected_query,
        "retmode": "json",
        "retmax": _PAIRED_V4_MAX_DOCUMENTS,
        "sort": "relevance",
    }
    summary_params: dict[str, object] = {
        "db": "pubmed",
        "id": ",".join(document_ids),
        "retmode": "json",
    }
    fetch_params: dict[str, object] = {
        "db": "pubmed",
        "id": ",".join(eligible_ids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if not (
        _paired_v4_call_matches(
            calls[0],
            stage="esearch",
            endpoint=_PAIRED_V4_SEARCH_ENDPOINT,
            params=search_params,
        )
        and _paired_v4_call_matches(
            calls[1],
            stage="esummary",
            endpoint=_PUBMED_SUMMARY_ENDPOINT,
            params=summary_params,
        )
        and _paired_v4_call_matches(
            calls[2],
            stage="efetch",
            endpoint=_PAIRED_V4_FETCH_ENDPOINT,
            params=fetch_params,
        )
    ):
        return None
    verified_groups: list[dict[str, object]] = []
    group_counts: dict[str, int] = {}
    union_ids: set[str] = set()
    selected_group_hash = (
        selected_group.get("group_sha256")
        if isinstance(selected_group, dict)
        else None
    )
    selected_match_count = 0
    pair_measurements_by_candidate: dict[
        str, set[tuple[str, str, str, str]]
    ] = {}
    for group in groups:
        if not isinstance(group, dict):
            return None
        document_id = str(group.get("document_id") or "")
        publication_types = publication_types_by_id.get(document_id)
        if publication_types is None:
            return None
        verified = _paired_v4_verify_group(
            group,
            inventory=inventory,
            query_sha256=str(query_sha256),
            publication_types=publication_types,
        )
        if verified is None:
            return None
        group_counts[document_id] = group_counts.get(document_id, 0) + 1
        union_ids.update(verified["candidate_ids"])
        verified_groups.append(verified)
        for item in verified["pairs"]:
            pair = item["pair"]
            pair_measurements_by_candidate.setdefault(
                str(item["candidate_id"]), set()
            ).add(
                (
                    str(pair.get("endpoint") or ""),
                    str(pair.get("value_raw") or ""),
                    str(pair.get("unit_raw") or ""),
                    str(pair.get("relation_normalized") or ""),
                )
            )
        if (
            verified["group_sha256"] == selected_group_hash
            and _paired_v4_hash(group) == _paired_v4_hash(selected_group)
        ):
            selected_match_count += 1
    if (
        selected_match_count != 1
        or group_counts != verified_group_counts
        or verified_candidate_ids
        != sorted(
            union_ids,
            key=lambda item: (_paired_v4_alias_key(item), item),
        )
        or any(
            len(values) != 1
            for values in pair_measurements_by_candidate.values()
        )
    ):
        return None
    selected_verified = next(
        (
            item
            for item in verified_groups
            if item["group_sha256"] == selected_group_hash
        ),
        None,
    )
    if selected_verified is None:
        return None
    return {
        "query": expected_query,
        "query_sha256": query_sha256,
        "groups": verified_groups,
        "selected_group": selected_verified,
        "canonical_sha256": _paired_v4_hash(raw),
        "call_response_sha256": [
            str(call.get("response_sha256"))
            for call in calls
            if isinstance(call, dict)
        ],
    }


def _paired_v4_candidate_chembl_id(candidate: object) -> str | None:
    metadata = getattr(candidate, "metadata", None)
    name = getattr(candidate, "name", "")
    if not isinstance(metadata, dict):
        return None
    identity = candidate_identity(candidate_name=str(name), metadata=metadata)
    chembl_ids = {
        str(item.get("identifier") or "").strip()
        for item in identity.get("registry_ids", [])
        if isinstance(item, dict)
        and str(item.get("namespace") or "").casefold() == "chembl"
        and str(item.get("identifier") or "").strip()
    }
    deposited_id = str(
        metadata.get("chembl_id") or metadata.get("drug_id") or ""
    ).strip()
    if (
        len(chembl_ids) != 1
        or not deposited_id
        or chembl_ids != {deposited_id.casefold()}
    ):
        return None
    return deposited_id


def _paired_v4_exact_decimal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return False


def _paired_v4_verify_tool_link(
    raw: object,
    *,
    inventory: Mapping[str, object],
    search_outcome_raw: Mapping[str, object],
    evidence_metadata: Mapping[str, object],
) -> bool:
    if not isinstance(raw, dict):
        return False
    calls = search_outcome_raw.get("calls")
    assay_context = evidence_metadata.get("chembl_assay_context")
    assay_query = (
        assay_context.get("assay_query")
        if isinstance(assay_context, dict)
        else None
    )
    return bool(
        raw.get("operation")
        == "retrieve_repurposing_primary_abstract_paired_v4"
        and raw.get("tool_call_record_linkage_status")
        == "pending_orchestrator_projection_gateway_has_no_packet_call_id"
        and raw.get("primary_search_input_sha256")
        == inventory.get("inventory_sha256")
        and raw.get("primary_search_output_sha256")
        == search_outcome_raw.get("outcome_sha256")
        and raw.get("primary_search_http_call_count") == 3
        and isinstance(calls, (list, tuple))
        and raw.get("primary_search_http_calls") == calls
        and raw.get("chembl_activity_query")
        == evidence_metadata.get("activity_query")
        and raw.get("chembl_assay_query") == assay_query
        and raw.get("invented_tool_call_id") is False
    )


def _paired_v4_verify_candidate_envelope(
    *,
    candidate: object,
    evidence: object,
    activity: object,
    envelope: object,
) -> dict[str, object] | None:
    if (
        not isinstance(envelope, dict)
        or not _paired_v4_self_hash(
            envelope, hash_key="attestation_sha256"
        )
        or envelope.get("schema_version") != _PAIRED_V4_ENVELOPE_SCHEMA
        or envelope.get("verification_status") != "verified"
        or envelope.get("is_primary_support") is not True
        or envelope.get("reason_code") is not None
        or envelope.get("policy") != _PAIRED_V4_POLICY
        or envelope.get("source") != "PubMed"
        or envelope.get("full_text_stored") is not False
        or envelope.get("hash_algorithm") != "sha256"
        or envelope.get("unit_conversion_applied") is not False
        or envelope.get("missing_values_imputed") is not False
        or envelope.get("pubmed_or_result_used_for_cohort_selection")
        is not False
        or envelope.get("fixture") is not False
    ):
        return None
    evidence_source = getattr(evidence, "source", None)
    evidence_supports = getattr(evidence, "supports", None)
    evidence_id = getattr(evidence, "evidence_id", None)
    evidence_external_id = str(getattr(evidence, "external_id", "") or "")
    evidence_metadata = getattr(evidence, "metadata", None)
    candidate_metadata = getattr(candidate, "metadata", None)
    target = getattr(candidate, "target", None)
    if (
        evidence_source != "ChEMBL"
        or evidence_supports != "support"
        or not isinstance(evidence_metadata, dict)
        or not isinstance(candidate_metadata, dict)
        or target is None
        or candidate_metadata.get(_PAIRED_V4_ENVELOPE_KEY) != envelope
    ):
        return None
    molecule_id = _paired_v4_candidate_chembl_id(candidate)
    if (
        molecule_id is None
        or envelope.get("molecule_chembl_id") != molecule_id
        or str(evidence_metadata.get("molecule_chembl_id") or "")
        != molecule_id
        or str(evidence_metadata.get("activity_id") or "")
        != evidence_external_id
    ):
        return None
    discovery_bounds = evidence_metadata.get("candidate_discovery_bounds")
    inventory_raw = envelope.get("alias_inventory")
    inventory = _paired_v4_inventory(
        inventory_raw,
        discovery_bounds=discovery_bounds,
    )
    if inventory is None:
        return None
    aliases_by_id = inventory["aliases_by_id"]
    candidate_aliases = candidate_metadata.get("chembl_exact_aliases")
    inventory_aliases = aliases_by_id.get(molecule_id)
    if (
        not isinstance(candidate_aliases, list)
        or not isinstance(inventory_aliases, list)
        or molecule_id not in inventory.get("gateway_return_ids", [])
        or candidate_aliases[:_PAIRED_V4_MAX_ALIASES] != inventory_aliases
        or candidate_metadata.get("chembl_alias_conflicts") != []
    ):
        return None
    search_outcome_raw = envelope.get("search_outcome")
    if (
        not isinstance(search_outcome_raw, dict)
        or not _paired_v4_self_hash(
            search_outcome_raw, hash_key="outcome_sha256"
        )
    ):
        return None
    group_raw = envelope.get("group")
    search = _paired_v4_verify_search_outcome(
        search_outcome_raw,
        selected_group=group_raw,
        inventory=inventory,
    )
    if search is None:
        return None
    selected_group = search["selected_group"]
    selected_index = envelope.get("selected_pair_index")
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 0 <= selected_index < len(selected_group["pairs"])
    ):
        return None
    selected_pair = selected_group["pairs"][selected_index]
    pair = selected_pair["pair"]
    if (
        selected_pair["candidate_id"] != molecule_id
        or pair.get("pair_index") != selected_index
        or envelope.get("selected_pair_evidence_sha256")
        != pair.get("evidence_sha256")
    ):
        return None
    pmid = str(envelope.get("pmid") or "")
    if (
        pmid != selected_group["document_id"]
        or envelope.get("source_url")
        != f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        or envelope.get("section") != pair.get("section")
        or _paired_v4_normalized_publication_types(
            envelope.get("publication_types")
        )
        != _paired_v4_normalized_publication_types(
            pair.get("publication_types")
        )
    ):
        return None
    target_binding = envelope.get("target_binding")
    target_chembl_id = str(
        getattr(target, "metadata", {}).get("chembl_target_id") or ""
    )
    if (
        not isinstance(target_binding, dict)
        or target_binding.get("target_symbol") != getattr(target, "symbol", None)
        or target_binding.get("target_accession")
        != getattr(target, "uniprot_id", None)
        or target_binding.get("target_chembl_id") != target_chembl_id
        or inventory.get("target_chembl_id") != target_chembl_id
        or target_binding.get("target_alias") != pair.get("target_alias")
        or target_binding.get("target_span") != pair.get("target_span")
        or str(evidence_metadata.get("target_chembl_id") or "")
        != target_chembl_id
        or str(evidence_metadata.get("target_accession") or "")
        != str(getattr(target, "uniprot_id", None) or "")
    ):
        return None
    selected = envelope.get("selected_measurement")
    if not isinstance(selected, dict):
        return None
    pair_condition = pair.get("assay_condition")
    selected_pair_condition = selected.get("pair_assay_condition")
    retained_rank = selected.get("retained_rank")
    cohort = inventory["cohort"]
    cohort_rows = cohort.get("selected_rows")
    cohort_row = next(
        (
            row
            for row in cohort_rows
            if isinstance(row, dict)
            and row.get("molecule_chembl_id") == molecule_id
        ),
        None,
    )
    if (
        not isinstance(cohort_row, dict)
        or selected.get("activity_id") != evidence_external_id
        or selected.get("source_evidence_id") != evidence_id
        or selected.get("retained_rank") != cohort_row.get("retained_rank")
        or selected.get("activity_id") != cohort_row.get("activity_id")
        or selected.get("molecule_chembl_id") != molecule_id
        or selected.get("target_chembl_id") != target_chembl_id
        or selected.get("endpoint") != getattr(activity, "endpoint", None)
        or selected.get("relation") != getattr(activity, "relation", None)
        or selected.get("unit") != getattr(activity, "unit", None)
        or not _paired_v4_exact_decimal(
            selected.get("value"), getattr(activity, "value", None)
        )
        or selected.get("assay_type") != getattr(activity, "assay_type", None)
        or getattr(activity, "source_evidence_id", None) != evidence_id
        or selected.get("deposited_standard_type")
        != evidence_metadata.get("standard_type")
        or selected.get("deposited_standard_relation")
        != evidence_metadata.get("standard_relation")
        or selected.get("deposited_standard_units")
        != evidence_metadata.get("standard_units")
        or not _paired_v4_exact_decimal(
            selected.get("deposited_standard_value"),
            evidence_metadata.get("standard_value"),
        )
        or selected.get("pair_value_raw") != pair.get("value_raw")
        or selected.get("pair_unit_raw") != pair.get("unit_raw")
        or selected.get("pair_relation_normalized")
        != pair.get("relation_normalized")
        or not isinstance(pair_condition, dict)
        or not isinstance(selected_pair_condition, dict)
        or _paired_v4_canonical_bytes(selected_pair_condition)
        != _paired_v4_canonical_bytes(pair_condition)
        or selected.get("exact_decimal_value_match") is not True
        or selected.get("exact_unit_match") is not True
        or selected.get("unit_conversion_applied") is not False
        or selected.get("missing_values_imputed") is not False
        or selected.get("endpoint")
        != selected.get("deposited_standard_type")
        or selected.get("relation")
        != selected.get("deposited_standard_relation")
        or selected.get("unit") != selected.get("deposited_standard_units")
        or not _paired_v4_exact_decimal(
            selected.get("value"), selected.get("deposited_standard_value")
        )
        or not _paired_v4_exact_decimal(
            selected.get("value"), pair.get("value_raw")
        )
        or selected.get("unit") != pair.get("unit_raw")
        or selected.get("relation") != pair.get("relation_normalized")
        or selected.get("endpoint") != "IC50"
        or selected.get("relation") != "="
        or not isinstance(retained_rank, int)
        or isinstance(retained_rank, bool)
        or not 1 <= retained_rank <= _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
        or evidence_metadata.get("direct_single_protein_engagement") is not True
        or not MA3AssuranceAgent._assay_type_supports_direct_engagement(
            str(selected.get("assay_type") or "")
        )
    ):
        return None
    signature = _paired_v4_valid_signature(
        envelope.get("selected_assay_signature")
    )
    activity_properties = evidence_metadata.get("activity_properties")
    assay_context = evidence_metadata.get("chembl_assay_context")
    assay_record = (
        assay_context.get("record")
        if isinstance(assay_context, dict)
        else None
    )
    expected_condition_binding = _paired_v4_recompute_condition_binding(
        evidence_metadata=evidence_metadata,
        assay_record=assay_record,
    )
    signature_condition_binding = (
        signature.get("assay_condition_binding")
        if isinstance(signature, dict)
        else None
    )
    top_level_condition_binding = envelope.get(
        "selected_assay_condition_binding"
    )
    if (
        signature is None
        or expected_condition_binding is None
        or _paired_v4_valid_condition_binding(
            expected_condition_binding,
            exact_required=True,
        )
        is None
        or signature_condition_binding != expected_condition_binding
        or top_level_condition_binding != signature_condition_binding
        or _paired_v4_valid_condition_binding(
            top_level_condition_binding,
            exact_required=True,
        )
        is None
        or not isinstance(selected_pair_condition, dict)
        or any(
            selected_pair_condition.get(key)
            != expected_condition_binding.get(key)
            for key in ("subject", "value_raw", "unit_raw")
        )
        or signature != cohort.get("selected_signature")
        or signature.get("target_chembl_id") != target_chembl_id
        or signature.get("target_accession")
        != getattr(target, "uniprot_id", None)
        or signature.get("assay_chembl_id")
        != evidence_metadata.get("assay_chembl_id")
        or signature.get("assay_type") != getattr(activity, "assay_type", None)
        or signature.get("endpoint") != getattr(activity, "endpoint", None)
        or signature.get("relation") != getattr(activity, "relation", None)
        or signature.get("unit") != getattr(activity, "unit", None)
        or not isinstance(activity_properties, list)
        or signature.get("activity_properties_sha256")
        != _paired_v4_hash(activity_properties)
        or not isinstance(assay_context, dict)
        or assay_context.get("retrieval_status") != "completed"
        or assay_context.get("binding_status") != "exact"
        or assay_context.get("assay_chembl_id")
        != signature.get("assay_chembl_id")
        or assay_context.get("missing_values_imputed") is not False
        or assay_context.get("inference_used") is not False
        or not isinstance(assay_record, dict)
        or assay_record.get("assay_chembl_id")
        != signature.get("assay_chembl_id")
        or assay_record.get("target_chembl_id") != target_chembl_id
    ):
        return None
    same_assay_ids = _paired_v4_strings(
        envelope.get("same_assay_cohort_candidate_ids"),
        minimum=_PAIRED_V4_MIN_NONCONTROL_CANDIDATES,
    )
    if (
        same_assay_ids is None
        or molecule_id not in same_assay_ids
        or not set(same_assay_ids).issubset(
            set(selected_group["candidate_ids"])
        )
        or not set(same_assay_ids).issubset(set(cohort["selected_ids"]))
    ):
        return None
    promotion = envelope.get("promotion_summary")
    if (
        not isinstance(promotion, dict)
        or not _paired_v4_self_hash(
            promotion, hash_key="promotion_sha256"
        )
        or promotion.get("policy") != _PAIRED_V4_POLICY
        or promotion.get("status") != "verified"
        or promotion.get("reason_code") is not None
        or molecule_id not in (promotion.get("promoted_candidate_ids") or [])
        or promotion.get("cross_group_conflict_candidate_ids") != []
        or promotion.get("unit_conversion_applied") is not False
        or promotion.get("missing_values_imputed") is not False
        or promotion.get("no_backfill") is not True
        or promotion.get("fixture") is not False
    ):
        return None
    if not _paired_v4_verify_tool_link(
        envelope.get("tool_call_link"),
        inventory=inventory_raw,
        search_outcome_raw=search_outcome_raw,
        evidence_metadata=evidence_metadata,
    ):
        return None
    return {
        "molecule_id": molecule_id,
        "group_sha256": selected_group["group_sha256"],
        "group_canonical_sha256": selected_group["canonical_sha256"],
        "search_canonical_sha256": search["canonical_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "signature_sha256": signature["signature_sha256"],
        "cohort_ids": same_assay_ids,
        "gateway_return_ids": inventory["gateway_return_ids"],
        "candidate_id": getattr(candidate, "candidate_id", None),
    }


def _paper_v5_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _paper_v5_hash(value: object) -> str:
    return sha256(_paper_v5_bytes(value)).hexdigest()


def _paper_v6_hash(value: object) -> str:
    """MA3-owned canonical hash; deliberately independent of the v6 producer."""

    return sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _paper_v6_exact_decimal(left: object, right: object) -> bool:
    """Compare deposited and typed activity values without type coercion loss."""

    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return False


def _paper_v6_phrase_present(text: str, phrase: str) -> bool:
    if re.sub(r"[^a-z0-9]+", "", phrase.casefold()) == "ic50":
        return re.search(
            r"(?<![A-Za-z0-9])ic[\s\-_]*50(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        ) is not None
    words = re.findall(r"[A-Za-z0-9]+", phrase.casefold())
    return bool(words) and re.search(
        r"(?<![A-Za-z0-9])"
        + r"[\s\-_]+".join(map(re.escape, words))
        + r"(?![A-Za-z0-9])",
        text.casefold(),
    ) is not None


def _paper_v6_strict_unit(
    unit: object, *, expected_target_symbol: str, expected_direction: str,
) -> str | None:
    """Validate the complete persisted V6 support-unit projection."""

    if not isinstance(unit, Mapping):
        return None
    expected_keys = {
        "candidate_id", "candidate_aliases", "direction", "document_id_sha256",
        "fixture", "is_primary_support", "is_review", "predicate", "endpoint",
        "relation", "value", "unit", "primary_source_attestation_sha256",
        "publication_types", "publication_type_screen_passed", "sentence",
        "sentence_sha256", "source_location", "target_symbol",
        "verification_status", "support_unit_sha256",
    }
    aliases = unit.get("candidate_aliases")
    sentence = unit.get("sentence")
    body = dict(unit)
    reported = body.pop("support_unit_sha256", None)
    source_location = unit.get("source_location")
    try:
        decoded_location = json.loads(source_location) if isinstance(source_location, str) else None
    except (TypeError, ValueError):
        decoded_location = None
    parenthetical_scope = bool(
        isinstance(decoded_location, Mapping)
        and decoded_location.get("selection_scope")
        == "atomic_parenthetical_with_sentence_context"
        and decoded_location.get("clause_index") is None
    )
    exact_value_unit = bool(isinstance(sentence, str) and _paper_v6_exact_value_unit_present(
        sentence, unit.get("value"), unit.get("unit"), unit.get("relation")
    ))
    if (
        set(unit) != expected_keys
        or not isinstance(unit.get("candidate_id"), str) or not unit["candidate_id"]
        or not isinstance(aliases, list) or not aliases or len(aliases) != len(set(aliases))
        or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
        or unit.get("direction") != expected_direction or expected_direction != "inhibit"
        or unit.get("target_symbol") != expected_target_symbol
        or unit.get("predicate") not in {"inhibits", "inhibition"}
        or unit.get("endpoint") != "IC50"
        or unit.get("relation") not in {"=", "<", ">", "<=", ">="}
        or type(unit.get("value")) not in {int, float} or not 0 < unit["value"] < 1_000_000_000
        or unit.get("unit") not in {"pM", "nM", "uM", "µM", "μM", "mM", "M"}
        or unit.get("fixture") is not False
        or unit.get("is_primary_support") is not True
        or unit.get("is_review") is not False
        or unit.get("verification_status") != "verified"
        or unit.get("publication_type_screen_passed") is not True
        or not _paper_v6_publication_types_valid(unit.get("publication_types"))
        or not isinstance(sentence, str) or not sentence or sentence != sentence.strip() or len(sentence) > 360
        or unit.get("sentence_sha256") != sha256(sentence.encode("utf-8")).hexdigest()
        or not any(_paper_v6_phrase_present(sentence, alias) for alias in aliases)
        or not _paper_v6_phrase_present(sentence, expected_target_symbol)
        or not _paper_v6_phrase_present(sentence, "IC50")
        or (
            not parenthetical_scope
            and _primary_clause_semantically_unsafe(sentence)
        )
        or not isinstance(decoded_location, Mapping)
        or source_location != json.dumps(decoded_location, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        or not exact_value_unit
        or (
            not parenthetical_scope
            and re.search(
                r"(?<!\w)respectively(?!\w)|;|:|—|–",
                sentence,
                re.I,
            )
        )
        or any(not isinstance(unit.get(key), str) or _SHA256.fullmatch(str(unit[key])) is None for key in (
            "document_id_sha256", "primary_source_attestation_sha256",
        ))
        or reported != _paper_v6_hash(body)
    ):
        return None
    return str(reported)


def _paper_v6_clause_is_unambiguous(
    sentence: str, *, aliases: list[str], primary: Mapping[str, object],
    target_symbol: str, endpoint: str, relation: object, value: object, unit: str,
) -> bool:
    """MA3-local bounded rejection of cross-owner/cross-clause attribution."""

    competitor = primary.get("competitor_alias_provenance")
    competitor_aliases = (
        competitor.get("competitor_exact_aliases")
        if isinstance(competitor, Mapping)
        else None
    )
    location = primary.get("excerpt_location")
    if (
        isinstance(location, Mapping)
        and location.get("selection_scope")
        == "atomic_parenthetical_with_sentence_context"
    ):
        if not (
            isinstance(competitor, Mapping)
            and competitor.get("policy")
                == "all_bounded_pool_candidate_aliases_except_current_v1"
            and competitor.get("truncated") is False
            and isinstance(competitor_aliases, list)
            and len(competitor_aliases) <= _PAPER_NATIVE_V6_MAX_COMPETITOR_ALIASES
            and all(isinstance(item, str) and item for item in competitor_aliases)
        ):
            return False
        try:
            audit_target = TargetCandidate(symbol=target_symbol)
            audit_activity = ActivityMeasurement(
                endpoint=endpoint,
                relation=relation,
                value=value,
                unit=unit,
                assay_type="independent primary-source audit",
            )
        except (TypeError, ValueError):
            return False
        recomputed = _primary_parenthetical_binding(
            sentence,
            candidate_aliases=aliases,
            competitor_aliases=competitor_aliases,
            target=audit_target,
            activity=audit_activity,
        )
        relevance = primary.get("relevance")
        return bool(
            recomputed is not None
            and isinstance(relevance, Mapping)
            and relevance.get("matched_groups")
                == recomputed.get("matched_groups")
            and relevance.get("matched_terms")
                == recomputed.get("matched_terms")
            and relevance.get("matched_exact_value_unit")
                == recomputed.get("matched_exact_value_unit")
            and relevance.get("binding_pattern")
                == recomputed.get("binding_pattern")
            and relevance.get("bound_spans")
                == recomputed.get("bound_spans")
            and relevance.get("competitor_alias_matches") == []
        )
    if (
        not isinstance(competitor, Mapping)
        or competitor.get("policy") != "all_bounded_pool_candidate_aliases_except_current_v1"
        or competitor.get("truncated") is not False
        or not isinstance(competitor_aliases, list)
        or len(competitor_aliases) > _PAPER_NATIVE_V6_MAX_COMPETITOR_ALIASES
        or any(not isinstance(item, str) or not item for item in competitor_aliases)
        or re.search(r"(?<!\w)respectively(?!\w)", sentence, re.I)
        or re.search(r";|:|—|–|(?<!\w)(?:whereas|while|but)(?!\w)", sentence, re.I)
        or re.search(r"[!?]|(?<!\d)\.(?!\d)(?=.+)", sentence.rstrip(". "))
        or any(_paper_v6_phrase_present(sentence, item) for item in competitor_aliases)
        or _primary_clause_semantically_unsafe(sentence)
        or not any(_paper_v6_phrase_present(sentence, item) for item in aliases)
        or not _paper_v6_phrase_present(sentence, target_symbol)
        or not _paper_v6_phrase_present(sentence, endpoint)
    ):
        return False
    return _paper_v6_exact_value_unit_present(sentence, value, unit, relation)


def _paper_v6_reserved_disposition_valid(
    candidate: object, reserved_controls: object
) -> tuple[bool, bool]:
    reserved = screen_reserved_control(
        candidate_name=str(getattr(candidate, "name", "")),
        metadata=getattr(candidate, "metadata", {}),
        reserved_controls=reserved_controls,
    )
    if not reserved.excluded:
        metadata = getattr(candidate, "metadata", {})
        return bool(
            isinstance(metadata, Mapping)
            and metadata.get("reserved_control_screen") == reserved.as_metadata()
            and metadata.get("control_only") is False
        ), False
    metadata = getattr(candidate, "metadata", {})
    valid = bool(
        getattr(candidate, "status", None) == "excluded"
        and getattr(candidate, "score", None) == 0.0
        and all(getattr(candidate, key, None) is None for key in (
            "rank", "target_rank", "portfolio_rank", "portfolio_score",
            "chemistry_execution", "docking_execution", "admet_execution",
        ))
        and isinstance(metadata, Mapping)
        and metadata.get("policy_eligible") is False
        and metadata.get("reserved_control_screen") == reserved.as_metadata()
        and metadata.get("control_only") is reserved.matched
        and set(reserved.reason_codes).issubset(set(getattr(candidate, "exclusion_reasons", [])))
    )
    return valid, True


def _paper_v6_assay_context(value: object) -> str | None:
    """Mirror the producer's non-strengthening assay-context classifier."""

    if not isinstance(value, str) or not value:
        return None
    normalized = " ".join(value.casefold().replace("_", " ").split())
    if any(
        term in normalized
        for term in ("cell", "tissue", "organism", "phenotyp")
    ):
        return "cellular_or_system_context"
    if any(
        term in normalized
        for term in (
            "single protein",
            "biochemical",
            "protein binding",
            "purified protein",
            "enzyme assay",
        )
    ):
        return "direct_single_protein_context"
    return "assay_context_unspecified"


def _paper_v6_assay_engagement(
    *,
    assay_type: str,
    chembl_assay_context: object,
    assay_chembl_id: str,
    target_chembl_id: str,
    target_symbol: str,
) -> tuple[str, bool]:
    """Independently reproduce the exact-record direct-assay decision."""

    label_context = _paper_v6_assay_context(assay_type)
    if label_context == "direct_single_protein_context":
        return label_context, True
    if label_context == "cellular_or_system_context":
        return label_context, False
    context = (
        chembl_assay_context
        if isinstance(chembl_assay_context, Mapping)
        else {}
    )
    record = context.get("record")
    if not isinstance(record, Mapping):
        return "assay_context_unspecified", False
    description = record.get("description")
    excerpt = (
        description.get("excerpt")
        if isinstance(description, Mapping)
        else None
    )
    normalized = " ".join(excerpt.split()) if isinstance(excerpt, str) else ""
    description_valid = bool(
        isinstance(description, Mapping)
        and excerpt == normalized
        and "<" not in normalized
        and ">" not in normalized
        and description.get("excerpt_truncated") is False
        and description.get("content_length") == len(normalized)
        and description.get("content_sha256")
        == sha256(normalized.encode("utf-8")).hexdigest()
        and target_symbol.strip()
        and re.search(
            rf"(?<![A-Za-z0-9]){re.escape(target_symbol.strip())}(?![A-Za-z0-9])",
            normalized,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"(?<!\w)(?:recombinant|purified)(?!\w)",
            normalized,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"(?<!\w)(?:inhibit\w*|kinase|enzyme)(?!\w)",
            normalized,
            flags=re.IGNORECASE,
        )
        and re.search(r"(?<!\w)atp(?!\w)", normalized, flags=re.IGNORECASE)
        and re.search(
            r"(?<!\w)substrate(?!\w)",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    direct = bool(
        context.get("schema_version") == "chembl_assay_context_v1"
        and context.get("retrieval_status") == "completed"
        and context.get("binding_status") == "exact"
        and context.get("assay_chembl_id") == assay_chembl_id
        and context.get("missing_values_imputed") is False
        and record.get("assay_chembl_id") == assay_chembl_id
        and record.get("target_chembl_id") == target_chembl_id
        and record.get("assay_type") == "B"
        and record.get("assay_organism") == "Homo sapiens"
        and record.get("relationship_type") == "D"
        and record.get("confidence_score") == 9
        and all(
            record.get(key) in (None, "")
            for key in (
                "assay_cell_type",
                "assay_tissue",
                "assay_strain",
                "cell_chembl_id",
                "tissue_chembl_id",
            )
        )
        and description_valid
    )
    return (
        "direct_single_protein_context"
        if direct
        else "assay_context_unspecified",
        direct,
    )


def _paper_v6_mechanism_claim(
    *,
    target_symbol: str,
    assay_type: str,
    direction_rule: str,
    mechanism_identity_present: bool,
    assay_context: str | None = None,
) -> str:
    context = assay_context or _paper_v6_assay_context(assay_type)
    if mechanism_identity_present:
        if context == "direct_single_protein_context":
            return (
                "ChEMBL-curated inhibitor mechanism and measured direct "
                f"IC50 activity against {target_symbol}"
            )
        if context == "cellular_or_system_context":
            return (
                "ChEMBL-curated inhibitor mechanism with measured cellular "
                f"IC50 activity for {target_symbol}; direct single-protein "
                "engagement not established by the selected activity row"
            )
        return (
            "ChEMBL-curated inhibitor mechanism with measured IC50 activity "
            f"for {target_symbol}; assay context unspecified; direct "
            "engagement not established"
        )
    if direction_rule == "measured_ic50_endpoint":
        if context == "direct_single_protein_context":
            return f"Measured inhibitory potency against {target_symbol}"
        if context == "cellular_or_system_context":
            return (
                f"Measured cellular inhibitory potency in a {target_symbol}-linked "
                "assay; direct single-protein engagement not established"
            )
        return (
            f"Measured inhibitory potency for {target_symbol}; assay context "
            "unspecified; direct engagement not established"
        )
    return (
        f"Measured IC50 activity for {target_symbol}; "
        "agonist/antagonist direction unresolved"
    )


def _paper_v6_mechanism_identity_valid(
    value: object,
    *,
    bounds: Mapping[str, object],
    molecule_id: str,
) -> bool:
    provenance = bounds.get("mechanism_provenance_by_id")
    mechanism_query = bounds.get("mechanism_query")
    if (
        not isinstance(value, Mapping)
        or not isinstance(provenance, Mapping)
        or provenance.get(molecule_id) != value
        or value.get("schema_version") != "chembl_mechanism_identity_v1"
        or value.get("parent_molecule_chembl_id") != molecule_id
        or value.get("query") != mechanism_query
        or value.get("fixture") is not False
        or value.get("missing_values_imputed") is not False
    ):
        return False
    rows = value.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or value.get("row_count") != len(rows)
        or value.get("rows_sha256") != _paper_v6_hash(rows)
    ):
        return False
    target_id = bounds.get("target_chembl_id")
    return all(
        isinstance(row, Mapping)
        and row.get("canonical_parent_molecule_chembl_id") == molecule_id
        and row.get("target_chembl_id") == target_id
        and row.get("action_type") == "INHIBITOR"
        and row.get("direct_interaction") in (1, "1", True)
        and row.get("molecular_mechanism") in (1, "1", True)
        for row in rows
    )


def _paper_v6_delegate_seed_projection_valid(
    candidates: Sequence[object],
    *,
    bounds: Mapping[str, object],
    fixed_pool_ids: Sequence[str],
    dispositions: Sequence[object],
) -> bool:
    projection = bounds.get("delegate_seed_source_binding_projection")
    if not isinstance(projection, Mapping):
        return False
    body = dict(projection)
    supplied_hash = body.pop("attestation_sha256", None)
    rows = projection.get("rows")
    returned_ids = projection.get("delegate_return_candidate_ids")
    fixed_ids = list(fixed_pool_ids)
    if (
        supplied_hash != _paper_v6_hash(body)
        or projection.get("schema_version")
        != "paper_native_v6_delegate_seed_source_binding_projection/v1"
        or projection.get("fixed_pool_candidate_ids") != fixed_ids
        or projection.get("fixed_pool_candidate_ids_sha256")
        != _paper_v6_hash(fixed_ids)
        or not isinstance(returned_ids, list)
        or returned_ids
        != [item for item in fixed_ids if item in set(returned_ids)]
        or len(returned_ids) != len(set(returned_ids))
        or projection.get("delegate_return_candidate_ids_sha256")
        != _paper_v6_hash(returned_ids)
        or bounds.get("delegate_gateway_return_candidate_ids")
        != returned_ids
        or not isinstance(rows, list)
        or len(rows) != len(fixed_ids)
        or projection.get("row_count") != len(rows)
        or projection.get("rows_sha256") != _paper_v6_hash(rows)
        or projection.get("complete") is not True
        or projection.get("fixture") is not False
        or len(dispositions) != len(rows)
    ):
        return False
    wrapper_projection = bounds.get("wrapper_atomic_screen_projection")
    if not isinstance(wrapper_projection, Mapping):
        return False
    wrapper_body = dict(wrapper_projection)
    wrapper_hash = wrapper_body.pop("attestation_sha256", None)
    wrapper_rows = wrapper_projection.get("rows")
    if (
        set(wrapper_projection)
        != {
            "schema_version",
            "fixed_pool_candidate_ids",
            "fixed_pool_candidate_ids_sha256",
            "row_count",
            "rows",
            "rows_sha256",
            "complete",
            "fixture",
            "attestation_sha256",
        }
        or wrapper_hash != _paper_v6_hash(wrapper_body)
        or wrapper_projection.get("schema_version")
        != _PAPER_NATIVE_V6_WRAPPER_SCREEN_SCHEMA
        or wrapper_projection.get("fixed_pool_candidate_ids") != fixed_ids
        or wrapper_projection.get("fixed_pool_candidate_ids_sha256")
        != _paper_v6_hash(fixed_ids)
        or not isinstance(wrapper_rows, list)
        or len(wrapper_rows) != len(fixed_ids)
        or wrapper_projection.get("row_count") != len(wrapper_rows)
        or wrapper_projection.get("rows_sha256")
        != _paper_v6_hash(wrapper_rows)
        or wrapper_projection.get("complete") is not True
        or wrapper_projection.get("fixture") is not False
    ):
        return False
    candidate_by_id = {
        _paired_v4_candidate_chembl_id(candidate): candidate
        for candidate in candidates
    }
    exact_keys = {
        "pool_index",
        "fixed_pool_rank",
        "molecule_chembl_id",
        "delegate_returned",
        "activity_count",
        "evidence_count",
        "activity_id",
        "document_chembl_id",
        "pmid",
        "primary_verification_status",
        "primary_reason_code",
        "primary_attestation_sha256",
        "fixture",
        "row_sha256",
    }
    screen_exact_keys = {
        "pool_index",
        "fixed_pool_rank",
        "molecule_chembl_id",
        "delegate_returned",
        "wrapper_screen_status",
        "wrapper_screen_reason_code",
        "wrapper_screen_checks",
        "wrapper_screen_checks_sha256",
        "fixture",
        "row_sha256",
    }
    returned_set = set(returned_ids)
    verified_screen_ids: list[str] = []
    for pool_index, (
        candidate_id,
        raw_row,
        raw_screen,
        raw_disposition,
    ) in enumerate(
        zip(fixed_ids, rows, wrapper_rows, dispositions, strict=True)
    ):
        if (
            not isinstance(raw_row, Mapping)
            or not isinstance(raw_screen, Mapping)
            or not isinstance(raw_disposition, Mapping)
        ):
            return False
        row = dict(raw_row)
        row_hash = row.pop("row_sha256", None)
        screen = dict(raw_screen)
        screen_hash = screen.pop("row_sha256", None)
        checks = raw_screen.get("wrapper_screen_checks")
        returned = candidate_id in returned_set
        activity_id = raw_row.get("activity_id")
        document_id = raw_row.get("document_chembl_id")
        pmid = raw_row.get("pmid")
        primary_hash = raw_row.get("primary_attestation_sha256")
        if (
            set(raw_row) != exact_keys
            or row_hash != _paper_v6_hash(row)
            or raw_row.get("pool_index") != pool_index
            or raw_row.get("fixed_pool_rank") != pool_index + 1
            or raw_row.get("molecule_chembl_id") != candidate_id
            or raw_row.get("delegate_returned") is not returned
            or raw_row.get("fixture") is not False
            or raw_disposition.get("delegate_returned") is not returned
            or raw_disposition.get("activity_id") != activity_id
            or raw_disposition.get("document_chembl_id") != document_id
            or raw_disposition.get("pmid") != pmid
            or set(raw_screen) != screen_exact_keys
            or screen_hash != _paper_v6_hash(screen)
            or raw_screen.get("pool_index") != pool_index
            or raw_screen.get("fixed_pool_rank") != pool_index + 1
            or raw_screen.get("molecule_chembl_id") != candidate_id
            or raw_screen.get("delegate_returned") is not returned
            or raw_screen.get("fixture") is not False
            or not isinstance(checks, Mapping)
            or set(checks)
            != {"cardinality_exact", "primary_mapping", "exact_binding"}
            or any(type(value) is not bool for value in checks.values())
            or raw_screen.get("wrapper_screen_checks_sha256")
            != _paper_v6_hash(dict(checks))
            or raw_disposition.get("source_primary_verification_status")
            != raw_row.get("primary_verification_status")
            or raw_disposition.get(
                "source_primary_verification_reason_code"
            )
            != raw_row.get("primary_reason_code")
            or raw_disposition.get("source_primary_attestation_sha256")
            != primary_hash
            or raw_disposition.get("wrapper_screen_status")
            != raw_screen.get("wrapper_screen_status")
            or raw_disposition.get("wrapper_screen_reason_code")
            != raw_screen.get("wrapper_screen_reason_code")
            or raw_disposition.get("wrapper_screen_checks") != checks
            or raw_disposition.get("wrapper_screen_checks_sha256")
            != raw_screen.get("wrapper_screen_checks_sha256")
            or raw_disposition.get("wrapper_screen_row_sha256")
            != screen_hash
        ):
            return False
        status = raw_screen.get("wrapper_screen_status")
        screen_reason = raw_screen.get("wrapper_screen_reason_code")
        cardinality_exact = (
            type(raw_row.get("activity_count")) is int
            and type(raw_row.get("evidence_count")) is int
            and raw_row.get("activity_count") == 1
            and raw_row.get("evidence_count") == 1
        )
        source_primary_mapping = (
            isinstance(primary_hash, str)
            and _SHA256.fullmatch(primary_hash) is not None
        )
        expected_status: str
        expected_reason: str | None
        if not returned:
            expected_status = "not_returned"
            expected_reason = "DELEGATE_NOT_RETURNED"
        elif not cardinality_exact:
            expected_status = "rejected"
            expected_reason = "SEED_CARDINALITY_NOT_EXACT"
        elif not source_primary_mapping:
            expected_status = "rejected"
            expected_reason = "PRIMARY_SOURCE_NOT_MAPPING"
        elif checks.get("exact_binding") is not True:
            expected_status = "rejected"
            expected_reason = "PAPER_NATIVE_V6_EXACT_BINDING_INVALID"
        else:
            expected_status = "verified"
            expected_reason = None
        if (
            checks.get("cardinality_exact") is not cardinality_exact
            or checks.get("primary_mapping") is not source_primary_mapping
            or (
                checks.get("exact_binding") is True
                and (not cardinality_exact or not source_primary_mapping)
            )
            or (
                checks.get("exact_binding") is True
                and (
                    raw_row.get("primary_verification_status")
                    != "verified"
                    or raw_row.get("primary_reason_code") is not None
                )
            )
            or status != expected_status
            or screen_reason != expected_reason
            or (
                screen_reason is not None
                and screen_reason not in _PAPER_NATIVE_V6_WRAPPER_SCREEN_REASONS
            )
            or raw_disposition.get("verification_status")
            != ("verified" if status == "verified" else "unverified")
        ):
            return False
        if status == "verified":
            verified_screen_ids.append(candidate_id)
        if not returned:
            if any(
                raw_row.get(key) is not None
                for key in (
                    "activity_id",
                    "document_chembl_id",
                    "pmid",
                    "primary_verification_status",
                    "primary_reason_code",
                    "primary_attestation_sha256",
                )
            ) or raw_row.get("activity_count") != 0 or raw_row.get(
                "evidence_count"
            ) != 0 or any(checks.values()):
                return False
            continue
        if (
            type(raw_row.get("activity_count")) is not int
            or raw_row.get("activity_count") < 0
            or type(raw_row.get("evidence_count")) is not int
            or raw_row.get("evidence_count") < 0
            or (
                raw_row.get("evidence_count") == 1
                and (
                    not isinstance(activity_id, str)
                    or re.fullmatch(r"[1-9][0-9]*", activity_id) is None
                )
            )
            or (
                raw_row.get("evidence_count") != 1
                and activity_id is not None
            )
            or (
                document_id is not None
                and (
                    not isinstance(document_id, str)
                    or _CHEMBL_STABLE_ID.fullmatch(document_id) is None
                )
            )
            or (
                pmid is not None
                and (
                    not isinstance(pmid, str)
                    or re.fullmatch(r"[1-9][0-9]*", pmid) is None
                )
            )
            or (
                source_primary_mapping
                and raw_row.get("primary_verification_status")
                not in {"verified", "unverified", "metadata_only"}
            )
            or (
                source_primary_mapping
                and raw_row.get("primary_verification_status") == "verified"
                and raw_row.get("primary_reason_code") is not None
            )
            or (
                source_primary_mapping
                and raw_row.get("primary_verification_status") != "verified"
                and (
                    not isinstance(raw_row.get("primary_reason_code"), str)
                    or not raw_row.get("primary_reason_code")
                )
            )
            or (
                not source_primary_mapping
                and any(
                    raw_row.get(key) is not None
                    for key in (
                        "document_chembl_id",
                        "pmid",
                        "primary_verification_status",
                        "primary_reason_code",
                        "primary_attestation_sha256",
                    )
                )
            )
        ):
            return False
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        evidence = getattr(candidate, "evidence", None)
        activities = getattr(candidate, "activities", None)
        v6_evidence = [
            item
            for item in evidence or []
            if isinstance(getattr(item, "metadata", None), Mapping)
            and isinstance(
                getattr(item, "metadata", {}).get(
                    _PAPER_NATIVE_V6_ENVELOPE_KEY
                ),
                Mapping,
            )
        ]
        if (
            not isinstance(evidence, list)
            or len(v6_evidence) != 1
            or not isinstance(activities, list)
            or len(activities) != 1
        ):
            return False
        item = v6_evidence[0]
        item_metadata = getattr(item, "metadata", None)
        primary = (
            item_metadata.get("primary_source")
            if isinstance(item_metadata, Mapping)
            else None
        )
        if (
            not isinstance(primary, Mapping)
            or activity_id != str(getattr(item, "external_id", ""))
            or document_id != primary.get("document_chembl_id")
            or pmid != primary.get("pmid")
            or raw_row.get("primary_verification_status")
            != primary.get("verification_status")
            or raw_row.get("primary_reason_code")
            != primary.get("reason_code")
            or primary_hash != _paper_v6_hash(dict(primary))
        ):
            return False
    return verified_screen_ids == [
        fixed_ids[index]
        for index, disposition in enumerate(dispositions)
        if isinstance(disposition, Mapping)
        and disposition.get("verification_status") == "verified"
    ]


def _paper_v6_retained_activity_ids(
    activity_count: object,
) -> list[str] | None:
    """Bind the V6 retained primary-audit rows to paginated ChEMBL IDs."""

    if not isinstance(activity_count, Mapping):
        return None
    eligible_ids = activity_count.get("eligible_directional_activity_ids")
    retained_ids = activity_count.get("retained_document_activity_ids")
    page_audits = activity_count.get("page_audits")
    if (
        not isinstance(eligible_ids, list)
        or not isinstance(retained_ids, list)
        or eligible_ids != retained_ids
        or not retained_ids
        or len(retained_ids) != len(set(retained_ids))
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[1-9][0-9]*", value) is None
            for value in retained_ids
        )
        or type(activity_count.get("eligible_directional_count")) is not int
        or activity_count.get("eligible_directional_count")
        != len(eligible_ids)
        or type(activity_count.get("retained_document_rows")) is not int
        or activity_count.get("retained_document_rows") != len(retained_ids)
        or activity_count.get("eligible_directional_activity_ids_sha256")
        != _paper_v6_hash(eligible_ids)
        or activity_count.get("retained_document_activity_ids_sha256")
        != _paper_v6_hash(retained_ids)
        or activity_count.get("primary_document_audit_result_cap") is not None
        or activity_count.get(
            "all_eligible_rows_retained_for_primary_audit"
        )
        is not True
        or not isinstance(page_audits, list)
        or not page_audits
    ):
        return None
    observed_page_ids: list[str] = []
    for page in page_audits:
        page_ids = (
            page.get("activity_ids") if isinstance(page, Mapping) else None
        )
        if (
            not isinstance(page_ids, list)
            or len(page_ids) != len(set(page_ids))
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[1-9][0-9]*", value) is None
                for value in page_ids
            )
        ):
            return None
        observed_page_ids.extend(page_ids)
    if (
        len(observed_page_ids) != len(set(observed_page_ids))
        or not set(retained_ids).issubset(observed_page_ids)
    ):
        return None
    return list(retained_ids)


def _paper_v6_candidate_projection(
    candidate: object, *, expected_direction: str
) -> dict[str, object] | None:
    """Rebuild one v6 candidate's atomic support from the immutable packet."""

    metadata = getattr(candidate, "metadata", None)
    evidence = getattr(candidate, "evidence", None)
    activities = getattr(candidate, "activities", None)
    if not isinstance(metadata, Mapping) or not isinstance(evidence, list) or not isinstance(activities, list):
        return None
    molecule_id = _paired_v4_candidate_chembl_id(candidate)
    v6_evidence = [
        item
        for item in evidence
        if isinstance(getattr(item, "metadata", None), Mapping)
        and isinstance(
            getattr(item, "metadata", {}).get(_PAPER_NATIVE_V6_ENVELOPE_KEY),
            Mapping,
        )
    ]
    # Public enrichment may append context evidence after candidate creation.
    # The v6 atomic unit is the single evidence row carrying the v6 envelope,
    # not the candidate's total evidence cardinality.  Duplicate v6 rows stay
    # fail-closed; all supplemental rows are audited by the generic source
    # audit after this projection succeeds.
    if molecule_id is None or len(v6_evidence) != 1 or len(activities) != 1:
        return None
    item, activity = v6_evidence[0], activities[0]
    item_metadata = getattr(item, "metadata", None)
    if not isinstance(item_metadata, Mapping):
        return None
    copies = (metadata.get(_PAPER_NATIVE_V6_ENVELOPE_KEY), item_metadata.get(_PAPER_NATIVE_V6_ENVELOPE_KEY))
    if not isinstance(copies[0], Mapping) or copies[0] != copies[1]:
        return None
    envelope = dict(copies[0])
    target = getattr(candidate, "target", None)
    target_symbol = getattr(target, "symbol", None)
    target_metadata = getattr(target, "metadata", {})
    primary = item_metadata.get("primary_source")
    if not isinstance(primary, Mapping) or not isinstance(target_symbol, str):
        return None
    primary_hash = _paper_v6_hash(dict(primary))
    binding = primary.get("activity_binding")
    deposited = binding.get("deposited_activity") if isinstance(binding, Mapping) else None
    excerpt = primary.get("excerpt")
    pmid = primary.get("pmid")
    document_id = primary.get("document_chembl_id")
    location = primary.get("excerpt_location")
    publication_types = primary.get("publication_types")
    pubmed_type = primary.get("pubmed_type_verification")
    deposited_target_id = next(
        (target_metadata.get(key) for key in ("target_chembl_id", "chembl_target_id") if target_metadata.get(key)),
        None,
    ) if isinstance(target_metadata, Mapping) else None
    discovery_bounds = metadata.get("candidate_discovery_bounds")
    if item_metadata.get("candidate_discovery_bounds") != discovery_bounds:
        return None
    target_id = discovery_bounds.get("target_chembl_id") if isinstance(discovery_bounds, Mapping) else None
    assay_type = getattr(activity, "assay_type", None)
    assay_id = str(
        binding.get("assay_chembl_id")
        if isinstance(binding, Mapping)
        else ""
    ).strip()
    assay_context, direct_engagement = _paper_v6_assay_engagement(
        assay_type=str(assay_type or ""),
        chembl_assay_context=item_metadata.get("chembl_assay_context"),
        assay_chembl_id=assay_id,
        target_chembl_id=str(target_id or ""),
        target_symbol=target_symbol,
    )
    mechanism_identity = metadata.get("mechanism_identity")
    evidence_mechanism_identity = item_metadata.get("mechanism_identity")
    direction_rule = item_metadata.get("direction_evidence_rule")
    expected_mechanism = (
        _paper_v6_mechanism_claim(
            target_symbol=target_symbol,
            assay_type=assay_type,
            direction_rule=str(direction_rule or ""),
            mechanism_identity_present=True,
            assay_context=assay_context,
        )
        if isinstance(assay_type, str)
        else None
    )
    activity_counts = (
        discovery_bounds.get("activity_counts_by_id")
        if isinstance(discovery_bounds, Mapping)
        else None
    )
    activity_count = (
        activity_counts.get(molecule_id)
        if isinstance(activity_counts, Mapping)
        else None
    )
    selection = primary.get("activity_document_selection")
    selection_rows = (
        selection.get("rows") if isinstance(selection, Mapping) else None
    )
    selection_activity_ids = (
        [
            str(row.get("activity_id") or "")
            for row in selection_rows
            if isinstance(row, Mapping)
        ]
        if isinstance(selection_rows, list)
        else None
    )
    retained_activity_ids = _paper_v6_retained_activity_ids(activity_count)
    if (
        primary.get("schema_version") != "candidate_primary_evidence_v2"
        or primary.get("verification_status") != "verified"
        or primary.get("retrieval_status") != "completed"
        or primary.get("is_primary_support") is not True
        or primary.get("reason_code") is not None
        or not isinstance(binding, Mapping)
        or binding.get("molecule_chembl_id") != molecule_id
        or not str(binding.get("activity_id") or "")
        or str(binding.get("activity_id"))
        != str(getattr(item, "external_id", ""))
        or binding.get("target_chembl_id") != target_id
        or (deposited_target_id is not None and deposited_target_id != target_id)
        or binding.get("endpoint") != getattr(activity, "endpoint", None)
        or binding.get("relation") != getattr(activity, "relation", None)
        or binding.get("value") != getattr(activity, "value", None)
        or binding.get("unit") != getattr(activity, "unit", None)
        or not isinstance(assay_type, str)
        or not assay_type
        or assay_context is None
        or binding.get("assay_type") != assay_type
        or binding.get("assay_context") != assay_context
        or binding.get("direct_single_protein_engagement")
        is not direct_engagement
        or binding.get("direction_evidence_rule") != "measured_ic50_endpoint"
        or direction_rule != "measured_ic50_endpoint"
        or metadata.get("direction_requested") != expected_direction
        or metadata.get("direction_evidence_rules")
        != ["measured_ic50_endpoint"]
        or item_metadata.get("intervention_direction_requested")
        != expected_direction
        or item_metadata.get("assay_type") != assay_type
        or item_metadata.get("assay_context") != assay_context
        or item_metadata.get("direct_single_protein_engagement")
        is not direct_engagement
        or metadata.get("activity_assay_type") != assay_type
        or metadata.get("activity_assay_context") != assay_context
        or metadata.get("direct_single_protein_engagement")
        is not direct_engagement
        or evidence_mechanism_identity != mechanism_identity
        or not isinstance(discovery_bounds, Mapping)
        or not _paper_v6_mechanism_identity_valid(
            mechanism_identity,
            bounds=discovery_bounds,
            molecule_id=molecule_id,
        )
        or binding.get("mechanism_identity_sha256")
        != _paper_v6_hash(dict(mechanism_identity))
        or not isinstance(expected_mechanism, str)
        or binding.get("mechanism_claim") != expected_mechanism
        or binding.get("mechanism_claim_sha256")
        != _paper_v6_hash(expected_mechanism)
        or getattr(candidate, "mechanism", None) != expected_mechanism
        or not isinstance(activity_count, Mapping)
        or retained_activity_ids is None
        or not isinstance(selection, Mapping)
        or not isinstance(selection_rows, list)
        or any(not isinstance(row, Mapping) for row in selection_rows)
        or selection_activity_ids
        != retained_activity_ids
        or selection.get("eligible_row_count")
        != activity_count.get("eligible_directional_count")
        or selection.get("inspected_row_count")
        != activity_count.get("retained_document_rows")
        or selection.get("inspected_row_count") != len(selection_rows)
        or selection.get("selected_activity_id")
        != str(getattr(item, "external_id", ""))
        or type(selection.get("selected_rank")) is not int
        or not 1 <= selection.get("selected_rank") <= len(selection_rows)
        or selection_activity_ids[selection.get("selected_rank") - 1]
        != str(getattr(item, "external_id", ""))
        or not isinstance(deposited, Mapping)
        or deposited.get("schema_version") != "chembl_deposited_activity_exact_v1"
        or deposited.get("standard_type") != getattr(activity, "endpoint", None)
        or deposited.get("standard_relation") != getattr(activity, "relation", None)
        or not _paper_v6_exact_decimal(
            deposited.get("standard_value"),
            getattr(activity, "value", None),
        )
        or deposited.get("standard_units") != getattr(activity, "unit", None)
        or deposited.get("typed_exact_match") is not True
        or deposited.get("missing_values_imputed") is not False
        or deposited.get("normalization_applied") is not False
        or deposited.get("unit_conversion_applied") is not False
        or not isinstance(excerpt, str) or not excerpt
        or primary.get("excerpt_hash") != sha256(excerpt.encode("utf-8")).hexdigest()
        or getattr(item, "excerpt_hash", None) not in {None, primary.get("excerpt_hash")}
        or not isinstance(pmid, str) or not pmid.isdigit()
        or not isinstance(document_id, str)
        or _CHEMBL_STABLE_ID.fullmatch(document_id) is None
        or binding.get("document_chembl_id") != document_id
        or not isinstance(location, Mapping)
        or not isinstance(pubmed_type, Mapping)
        or pubmed_type.get("schema_version") != "pubmed_esummary_publication_type_v1"
        or pubmed_type.get("type_policy") != "journal_article_without_excluded_publication_type_v1"
        or pubmed_type.get("verification_status") != "verified"
        or pubmed_type.get("retrieval_status") != "completed"
        or pubmed_type.get("pmid") != pmid
        or pubmed_type.get("reason_code") is not None
        or pubmed_type.get("required_publication_type") != "Journal Article"
        or pubmed_type.get("publication_types") != publication_types
        or pubmed_type.get("excluded_publication_types") != []
        or pubmed_type.get("fixture") is not False
        or not isinstance(publication_types, list) or not publication_types
        or not _paper_v6_publication_types_valid(publication_types)
    ):
        return None
    body = dict(envelope)
    attestation = body.pop("attestation_sha256", None)
    unit = envelope.get("candidate_support_unit")
    if (
        envelope.get("schema_version") != "candidate_primary_evidence_paper_native_v6"
        or envelope.get("source_contract") != "primary_abstract_paper_native_v6"
        or envelope.get("policy") != _PAPER_NATIVE_V6_POLICY
        or envelope.get("aggregation_policy") != _PAPER_NATIVE_V6_AGGREGATION_POLICY
        or envelope.get("verification_status") != "verified"
        or envelope.get("molecule_chembl_id") != molecule_id
        or envelope.get("fixture") is not False
        or envelope.get("missing_values_imputed") is not False
        or attestation != _paper_v6_hash(body)
        or not isinstance(unit, Mapping)
    ):
        return None
    aliases = unit.get("candidate_aliases")
    sentence = unit.get("sentence")
    publications = unit.get("publication_types")
    if (
        unit.get("candidate_id") != molecule_id
        or not isinstance(aliases, list) or not aliases or len(aliases) != len(set(aliases))
        or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
        or not isinstance(sentence, str) or not sentence or sentence != sentence.strip() or len(sentence) > 360
        or unit.get("sentence_sha256") != sha256(sentence.encode("utf-8")).hexdigest()
        or not any(re.search(r"(?<![A-Za-z0-9])" + r"[\s\-_]+".join(map(re.escape, re.findall(r"[A-Za-z0-9]+", alias.casefold()))) + r"(?![A-Za-z0-9])", sentence.casefold()) for alias in aliases if re.findall(r"[A-Za-z0-9]+", alias.casefold()))
        or unit.get("verification_status") != "verified"
        or unit.get("is_primary_support") is not True or unit.get("is_review") is not False
        or unit.get("fixture") is not False
        or unit.get("publication_type_screen_passed") is not True
        or expected_direction != "inhibit"
        or unit.get("predicate") not in {"inhibits", "inhibition"}
        or unit.get("direction") != expected_direction or unit.get("target_symbol") != target_symbol
        or unit.get("endpoint") != "IC50" or unit.get("relation") not in {"=", "<", ">", "<=", ">="}
        or type(unit.get("value")) not in {int, float} or not 0 < unit["value"] < 1_000_000_000
        or unit.get("unit") not in {"pM", "nM", "uM", "µM", "μM", "mM", "M"}
        or not isinstance(publications, list) or not publications
        or not _paper_v6_publication_types_valid(publications)
        or any(not isinstance(unit.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", unit[key]) is None for key in ("document_id_sha256", "primary_source_attestation_sha256"))
        or not isinstance(unit.get("source_location"), str) or not unit["source_location"]
        or unit.get("source_location") != json.dumps(location, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        or unit.get("document_id_sha256") != sha256(pmid.encode("utf-8")).hexdigest()
        or unit.get("primary_source_attestation_sha256") != primary_hash
        or unit.get("publication_types") != publication_types
        or primary.get("candidate_exact_aliases") != aliases
        or not _paper_v6_clause_is_unambiguous(
            sentence, aliases=aliases, primary=primary,
            target_symbol=target_symbol, endpoint=str(unit.get("endpoint")),
            relation=unit.get("relation"), value=unit.get("value"),
            unit=str(unit.get("unit")),
        )
    ):
        return None
    strict_unit_hash = _paper_v6_strict_unit(
        unit, expected_target_symbol=target_symbol,
        expected_direction=expected_direction,
    )
    if strict_unit_hash is None:
        return None
    unit_body = dict(unit)
    unit_hash = unit_body.pop("support_unit_sha256", None)
    unit_body["support_unit_sha256"] = _paper_v6_hash(unit_body)
    if unit_hash != unit_body["support_unit_sha256"] or envelope.get("candidate_support_unit_sha256") != unit_hash:
        return None
    if (
        getattr(activity, "source_evidence_id", None) != getattr(item, "evidence_id", None)
        or getattr(activity, "endpoint", None) != unit.get("endpoint")
        or getattr(activity, "relation", None) != unit.get("relation")
        or getattr(activity, "value", None) != unit.get("value")
        or getattr(activity, "unit", None) != unit.get("unit")
    ):
        return None
    group_ids = envelope.get("group_candidate_ids")
    fixed_ids = envelope.get("fixed_pool_candidate_ids")
    rank = envelope.get("fixed_pool_rank")
    if (
        envelope.get("primary_source_sha256") != primary_hash
        or envelope.get("no_reordering") is not True
        or envelope.get("no_backfill") is not True
        or not isinstance(fixed_ids, list) or len(fixed_ids) != len(set(fixed_ids))
        or not isinstance(group_ids, list) or len(group_ids) < 3 or len(group_ids) != len(set(group_ids))
        or group_ids != [value for value in fixed_ids if value in set(group_ids)]
        or molecule_id not in group_ids or type(rank) is not int
        or rank != fixed_ids.index(molecule_id) + 1
    ):
        return None
    return {
        "candidate_id": molecule_id, "unit_hash": unit_hash,
        "group_ids": list(group_ids), "group_sha256": envelope.get("group_sha256"),
        "rank": rank, "target_symbol": target_symbol,
        "direction": expected_direction, "unit": dict(unit),
        "assay_type": assay_type, "assay_context": assay_context,
        "direct_single_protein_engagement": direct_engagement,
        "mechanism_claim_sha256": _paper_v6_hash(expected_mechanism),
        "fixed_ids": list(fixed_ids),
    }


def _paper_v6_http_call_valid(
    raw: object,
    *,
    expected_source: str,
    expected_endpoint: str,
) -> bool:
    """Independently validate one successful source-call attestation."""

    if not isinstance(raw, Mapping):
        return False
    expected_keys = {
        "source",
        "endpoint",
        "parameter_keys",
        "parameter_hash",
        "timeout_seconds",
        "status_code",
        "response_sha256",
        "started_at",
        "finished_at",
        "runtime_seconds",
        "attempt_count",
        "max_attempts",
        "exhausted",
        "cache_hit",
        "operation",
        "operation_budget_seconds",
        "attempts",
    }
    parameter_keys = raw.get("parameter_keys")
    attempts = raw.get("attempts")
    attempt_count = raw.get("attempt_count")
    max_attempts = raw.get("max_attempts")
    timeout_seconds = raw.get("timeout_seconds")
    runtime_seconds = raw.get("runtime_seconds")
    operation_budget = raw.get("operation_budget_seconds")
    cache_hit = raw.get("cache_hit")
    if (
        set(raw) != expected_keys
        or raw.get("source") != expected_source
        or raw.get("endpoint") != expected_endpoint
        or raw.get("status_code") != 200
        or raw.get("exhausted") is not False
        or type(cache_hit) is not bool
        or not isinstance(parameter_keys, list)
        or parameter_keys != sorted(parameter_keys)
        or len(parameter_keys) != len(set(parameter_keys))
        or any(
            not isinstance(item, str) or not item
            for item in parameter_keys
        )
        or not _SHA256.fullmatch(str(raw.get("parameter_hash") or ""))
        or not _SHA256.fullmatch(str(raw.get("response_sha256") or ""))
        or type(attempt_count) is not int
        or type(max_attempts) is not int
        or not 1 <= max_attempts <= 3
        or not isinstance(attempts, list)
        or any(not isinstance(item, Mapping) for item in attempts)
        or type(timeout_seconds) not in {int, float}
        or float(timeout_seconds) < 0
        or type(runtime_seconds) not in {int, float}
        or float(runtime_seconds) < 0
        or type(operation_budget) not in {int, float}
        or float(operation_budget) <= 0
        or raw.get("operation") != "retrieve_repurposing"
        or not isinstance(raw.get("started_at"), str)
        or not isinstance(raw.get("finished_at"), str)
        or not raw.get("started_at")
        or not raw.get("finished_at")
        or (
            cache_hit is True
            and (attempt_count != 0 or attempts != [])
        )
        or (
            cache_hit is False
            and (not 1 <= attempt_count <= max_attempts or not attempts)
        )
        or (
            expected_source == "PubMed"
            and (cache_hit is not False or max_attempts != 1)
        )
    ):
        return False
    return True


def _paper_v6_deduplicated_calls(
    calls: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    observed: set[tuple[str, str, str, str]] = set()
    for raw in calls:
        key = (
            str(raw.get("source") or ""),
            str(raw.get("endpoint") or ""),
            str(raw.get("parameter_hash") or ""),
            str(raw.get("response_sha256") or ""),
        )
        if key in observed:
            continue
        observed.add(key)
        output.append(dict(raw))
    return output


def _paper_v6_parameter_hash(params: Mapping[str, object]) -> str:
    return sha256(
        json.dumps(
            dict(params),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _paper_v6_call_matches_params(
    call: object,
    params: Mapping[str, object],
) -> bool:
    return bool(
        isinstance(call, Mapping)
        and call.get("parameter_keys") == sorted(params)
        and call.get("parameter_hash") == _paper_v6_parameter_hash(params)
    )


def _paper_v6_projected_exact_aliases(
    pref_name: object,
    synonym_rows: Sequence[Mapping[str, object]],
) -> list[str] | None:
    """Independently rebuild the exact aliases promoted by live MA2."""

    values: list[object] = [pref_name]
    values.extend(row.get("molecule_synonym") for row in synonym_rows)
    aliases: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = re.sub(r"<[^>]+>", " ", str(raw or ""))
        alias = " ".join(text.split())
        normalized = " ".join(
            re.findall(r"[\w]+", alias.casefold(), flags=re.UNICODE)
        )
        if not normalized or normalized in seen:
            continue
        aliases.append(alias)
        seen.add(normalized)
        if len(aliases) >= _MAX_EXACT_CANDIDATE_ALIASES:
            break
    return aliases or None


def _paper_v6_fixed_pool_molecule_projection_valid(
    candidates: Sequence[object],
    *,
    bounds: Mapping[str, object],
    fixed_pool_ids: Sequence[str],
) -> bool:
    """Recompute the fixed-pool molecule rows and all promoted card copies."""

    projection = bounds.get("fixed_pool_molecule_projection")
    if not isinstance(projection, Mapping):
        return False
    body = dict(projection)
    attestation_hash = body.pop("attestation_sha256", None)
    rows = projection.get("rows")
    query = projection.get("molecule_query")
    if (
        set(projection)
        != {
            "schema_version",
            "fixed_pool_candidate_ids",
            "fixed_pool_candidate_ids_sha256",
            "molecule_query",
            "molecule_query_sha256",
            "row_count",
            "rows",
            "rows_sha256",
            "complete",
            "fixture",
            "attestation_sha256",
        }
        or attestation_hash != _paper_v6_hash(body)
        or projection.get("schema_version")
        != "chembl_fixed_pool_molecule_projection_v1"
        or projection.get("fixed_pool_candidate_ids") != list(fixed_pool_ids)
        or projection.get("fixed_pool_candidate_ids_sha256")
        != _paper_v6_hash(list(fixed_pool_ids))
        or not isinstance(query, Mapping)
        or query != bounds.get("molecule_query")
        or projection.get("molecule_query_sha256")
        != _paper_v6_hash(dict(query))
        or not isinstance(rows, list)
        or projection.get("row_count") != len(rows)
        or len(rows) != len(fixed_pool_ids)
        or projection.get("rows_sha256") != _paper_v6_hash(rows)
        or projection.get("complete") is not True
        or projection.get("fixture") is not False
    ):
        return False

    row_by_id: dict[str, Mapping[str, object]] = {}
    source_field_keys = {
        "molecule_chembl_id",
        "canonical_parent_molecule_chembl_id",
        "pref_name",
        "max_phase",
        "canonical_smiles",
        "molecule_synonyms_source_key_present",
        "molecule_synonyms_source_type",
        "molecule_synonyms_source_row_count",
        "molecule_synonyms_source_sha256",
        "molecule_synonym_row_limit",
        "molecule_synonym_retained_count",
        "molecule_synonyms_truncated",
        "molecule_synonyms",
        "exact_aliases",
        "molecule_query_parameter_sha256",
        "molecule_response_sha256",
        "source_molecule_row_sha256",
        "fixture",
    }
    expected_row_keys = {
        "schema_version",
        "pool_index",
        *source_field_keys,
        "projection_complete",
        "source_field_projection_sha256",
        "row_sha256",
    }
    for pool_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            return False
        row_body = dict(row)
        row_hash = row_body.pop("row_sha256", None)
        synonym_rows = row.get("molecule_synonyms")
        molecule_id = fixed_pool_ids[pool_index]
        source_fields = {key: row.get(key) for key in source_field_keys}
        if (
            set(row) != expected_row_keys
            or row_hash != _paper_v6_hash(row_body)
            or row.get("source_field_projection_sha256")
            != _paper_v6_hash(source_fields)
            or row.get("schema_version")
            != "chembl_fixed_pool_molecule_projection_row_v1"
            or row.get("pool_index") != pool_index
            or row.get("molecule_chembl_id") != molecule_id
            or row.get("canonical_parent_molecule_chembl_id") != molecule_id
            or _CHEMBL_STABLE_ID.fullmatch(molecule_id) is None
            or row.get("projection_complete") is not True
            or row.get("fixture") is not False
            or type(row.get("max_phase")) is not int
            or not 1 <= int(row["max_phase"]) <= 4
            or not isinstance(row.get("canonical_smiles"), str)
            or not str(row["canonical_smiles"])
            or str(row["canonical_smiles"]) != str(row["canonical_smiles"]).strip()
            or len(str(row["canonical_smiles"])) > 10_000
            or not isinstance(row.get("exact_aliases"), list)
            or not row.get("exact_aliases")
            or len(row["exact_aliases"]) > _MAX_EXACT_CANDIDATE_ALIASES
            or len(row["exact_aliases"]) != len(set(row["exact_aliases"]))
            or any(
                not isinstance(alias, str) or not alias
                for alias in row["exact_aliases"]
            )
            or row.get("molecule_query_parameter_sha256")
            != query.get("parameter_hash")
            or row.get("molecule_response_sha256")
            != query.get("response_sha256")
            or not _SHA256.fullmatch(
                str(row.get("source_molecule_row_sha256") or "")
            )
            or type(row.get("molecule_synonyms_source_key_present")) is not bool
            or not isinstance(row.get("molecule_synonyms_source_type"), str)
            or not isinstance(synonym_rows, list)
            or type(row.get("molecule_synonym_row_limit")) is not int
            or row.get("molecule_synonym_row_limit")
            != _MAX_EXACT_CANDIDATE_ALIASES * 2
            or row.get("molecule_synonym_retained_count") != len(synonym_rows)
            or len(synonym_rows) > _MAX_EXACT_CANDIDATE_ALIASES * 2
            or type(row.get("molecule_synonyms_truncated")) is not bool
        ):
            return False
        pref_name = row.get("pref_name")
        if pref_name is not None and (
            not isinstance(pref_name, str)
            or pref_name != pref_name.strip()
            or "\r" in pref_name
            or "\n" in pref_name
            or len(pref_name) > 300
        ):
            return False
        for source_index, synonym in enumerate(synonym_rows):
            if (
                not isinstance(synonym, Mapping)
                or set(synonym)
                != {
                    "source_index",
                    "row_type",
                    "molecule_synonym",
                    "syn_type",
                    "fields_valid",
                    "source_row_sha256",
                }
                or synonym.get("source_index") != source_index
                or synonym.get("row_type") != "mapping"
                or synonym.get("fields_valid") is not True
                or not _SHA256.fullmatch(
                    str(synonym.get("source_row_sha256") or "")
                )
            ):
                return False
            for key, limit in (("molecule_synonym", 300), ("syn_type", 80)):
                value = synonym.get(key)
                if value is not None and (
                    not isinstance(value, str)
                    or value != value.strip()
                    or "\r" in value
                    or "\n" in value
                    or len(value) > limit
                ):
                    return False
        source_count = row.get("molecule_synonyms_source_row_count")
        source_present = row.get("molecule_synonyms_source_key_present")
        source_type = row.get("molecule_synonyms_source_type")
        if source_type == "list":
            if (
                source_present is not True
                or type(source_count) is not int
                or source_count < len(synonym_rows)
                or len(synonym_rows)
                != min(source_count, _MAX_EXACT_CANDIDATE_ALIASES * 2)
                or row.get("molecule_synonyms_truncated")
                is not (source_count > _MAX_EXACT_CANDIDATE_ALIASES * 2)
                or not _SHA256.fullmatch(
                    str(row.get("molecule_synonyms_source_sha256") or "")
                )
            ):
                return False
        elif source_type in {"missing", "NoneType"}:
            if (
                source_count is not None
                or synonym_rows
                or row.get("molecule_synonyms_truncated") is not False
                or source_present != (source_type == "NoneType")
                or (
                    source_type == "missing"
                    and row.get("molecule_synonyms_source_sha256") is not None
                )
                or (
                    source_type == "NoneType"
                    and row.get("molecule_synonyms_source_sha256")
                    != _paper_v6_hash(None)
                )
            ):
                return False
        else:
            return False
        recomputed_aliases = _paper_v6_projected_exact_aliases(
            pref_name, synonym_rows
        )
        if row.get("exact_aliases") != recomputed_aliases:
            return False
        row_by_id[molecule_id] = row

    for candidate in candidates:
        molecule_id = _paired_v4_candidate_chembl_id(candidate)
        metadata = getattr(candidate, "metadata", None)
        evidence = getattr(candidate, "evidence", None)
        v6_evidence = [
            item
            for item in evidence or []
            if isinstance(getattr(item, "metadata", None), Mapping)
            and isinstance(
                getattr(item, "metadata", {}).get(
                    _PAPER_NATIVE_V6_ENVELOPE_KEY
                ),
                Mapping,
            )
        ]
        row = row_by_id.get(str(molecule_id))
        if (
            row is None
            or not isinstance(metadata, Mapping)
            or not isinstance(evidence, list)
            or len(v6_evidence) != 1
            or getattr(candidate, "smiles", None) != row.get("canonical_smiles")
            or metadata.get("canonical_smiles") != row.get("canonical_smiles")
            or metadata.get("max_phase") != row.get("max_phase")
            or metadata.get("chembl_exact_aliases") != row.get("exact_aliases")
            or metadata.get("aliases") != row.get("exact_aliases")
            or getattr(candidate, "name", None)
            != (row.get("pref_name") or molecule_id)
        ):
            return False
        evidence_metadata = getattr(v6_evidence[0], "metadata", None)
        primary = (
            evidence_metadata.get("primary_source")
            if isinstance(evidence_metadata, Mapping)
            else None
        )
        envelope = (
            evidence_metadata.get(_PAPER_NATIVE_V6_ENVELOPE_KEY)
            if isinstance(evidence_metadata, Mapping)
            else None
        )
        unit = (
            envelope.get("candidate_support_unit")
            if isinstance(envelope, Mapping)
            else None
        )
        if (
            not isinstance(primary, Mapping)
            or not isinstance(unit, Mapping)
            or primary.get("candidate_exact_aliases") != row.get("exact_aliases")
            or unit.get("candidate_aliases") != row.get("exact_aliases")
        ):
            return False
    return True


def _paper_v6_source_inventory_valid(bounds: object) -> bool:
    """Recompute the V6 source-call inventory and recovery contract."""

    if not isinstance(bounds, Mapping):
        return False
    inventory = bounds.get("source_call_inventory")
    recovery = bounds.get("source_recovery_attestation")
    if not isinstance(inventory, Mapping) or not isinstance(recovery, Mapping):
        return False
    inventory_body = dict(inventory)
    inventory_hash = inventory_body.pop("inventory_sha256", None)
    recovery_body = dict(recovery)
    recovery_hash = recovery_body.pop("attestation_sha256", None)
    stages = inventory.get("stages")
    if (
        set(inventory)
        != {
            "schema_version",
            "required_stages",
            "stages",
            "successful_call_count",
            "failure_count",
            "all_failure_records_represented",
            "fixture",
            "inventory_sha256",
        }
        or inventory_hash != _paper_v6_hash(inventory_body)
        or inventory.get("schema_version")
        != "scientific_source_call_outcome_inventory_v1"
        or inventory.get("required_stages")
        != list(_PAPER_V6_SOURCE_STAGE_ORDER)
        or not isinstance(stages, Mapping)
        or list(stages) != list(_PAPER_V6_SOURCE_STAGE_ORDER)
        or inventory.get("all_failure_records_represented") is not True
        or inventory.get("fixture") is not False
    ):
        return False

    expected_scopes: dict[str, object] = {
        "mechanism": [bounds.get("target_chembl_id")],
        "molecule": bounds.get("valid_mechanism_identity_ids"),
        "activity": bounds.get("prewarning_pool_ids"),
        "assay": None,
        "document": None,
        "warning": bounds.get("warning_screen_identity_ids"),
        "pubmed_type": None,
    }
    expected_scope_counts = {
        "assay": bounds.get("assay_lookup_id_count"),
        "document": bounds.get("document_lookup_id_count"),
        "pubmed_type": bounds.get("pubmed_type_lookup_pmid_count"),
    }
    all_failures: list[dict[str, object]] = []
    total_success = 0
    total_failure = 0
    for stage_name in _PAPER_V6_SOURCE_STAGE_ORDER:
        stage = stages.get(stage_name)
        if not isinstance(stage, Mapping):
            return False
        stage_body = dict(stage)
        stage_hash = stage_body.pop("stage_sha256", None)
        scope_ids = stage.get("scope_ids")
        successful_calls = stage.get("successful_calls")
        failures = stage.get("failures")
        policy = stage.get("recovery_policy")
        if (
            set(stage)
            != {
                "scope_ids",
                "scope_count",
                "successful_calls",
                "successful_call_count",
                "failures",
                "failure_count",
                "recovery_policy",
                "stage_sha256",
            }
            or stage_hash != _paper_v6_hash(stage_body)
            or not isinstance(scope_ids, list)
            or len(scope_ids) != len(set(scope_ids))
            or any(
                not isinstance(item, str)
                or not item
                or item != item.strip()
                for item in scope_ids
            )
            or stage.get("scope_count") != len(scope_ids)
            or not isinstance(successful_calls, list)
            or stage.get("successful_call_count") != len(successful_calls)
            or not isinstance(failures, list)
            or stage.get("failure_count") != len(failures)
            or policy not in _PAPER_V6_SOURCE_STAGE_POLICIES[stage_name]
        ):
            return False
        expected_scope = expected_scopes[stage_name]
        if expected_scope is not None and scope_ids != expected_scope:
            return False
        if stage_name in expected_scope_counts:
            expected_count = expected_scope_counts[stage_name]
            if type(expected_count) is not int or expected_count != len(scope_ids):
                return False
        expected_source = "PubMed" if stage_name == "pubmed_type" else "ChEMBL"
        endpoint = _PAPER_V6_SOURCE_STAGE_ENDPOINTS[stage_name]
        if any(
            not _paper_v6_http_call_valid(
                call,
                expected_source=expected_source,
                expected_endpoint=endpoint,
            )
            for call in successful_calls
        ):
            return False
        if successful_calls != _paper_v6_deduplicated_calls(successful_calls):
            return False
        failure_scope_ids: set[str] = set()
        for failure in failures:
            if not isinstance(failure, Mapping) or set(failure) != {
                "scope_id",
                "failure",
            }:
                return False
            scope_id = failure.get("scope_id")
            failure_payload = failure.get("failure")
            if (
                not isinstance(scope_id, str)
                or not scope_id
                or scope_id in failure_scope_ids
                or not isinstance(failure_payload, Mapping)
                or not failure_payload
                or (
                    scope_id not in scope_ids
                    and not (
                        stage_name == "warning" and scope_id == "__batch__"
                    )
                )
            ):
                return False
            failure_scope_ids.add(scope_id)
            all_failures.append(
                {
                    "stage": stage_name,
                    "scope_id": scope_id,
                    "policy": policy,
                }
            )
        total_success += len(successful_calls)
        total_failure += len(failures)

    mechanism_call = bounds.get("mechanism_query")
    molecule_call = bounds.get("molecule_query")
    mechanism_params = {
        "target_chembl_id": bounds.get("target_chembl_id"),
        "action_type": "INHIBITOR",
        "max_phase__gte": 1,
        "direct_interaction": 1,
        "molecular_mechanism": 1,
        "order_by": "molecule_chembl_id",
        "limit": 80,
    }
    molecule_scope = stages["molecule"].get("scope_ids")
    molecule_params = {
        "molecule_chembl_id__in": (
            ",".join(molecule_scope)
            if isinstance(molecule_scope, list)
            else ""
        ),
        "limit": len(molecule_scope) if isinstance(molecule_scope, list) else 0,
        "order_by": "molecule_chembl_id",
    }
    if (
        not isinstance(mechanism_call, Mapping)
        or not isinstance(molecule_call, Mapping)
        or stages["mechanism"].get("successful_calls")
        != [dict(mechanism_call)]
        or stages["molecule"].get("successful_calls")
        != [dict(molecule_call)]
        or not _paper_v6_call_matches_params(
            mechanism_call, mechanism_params
        )
        or not _paper_v6_call_matches_params(molecule_call, molecule_params)
    ):
        return False

    activity_counts = bounds.get("activity_counts_by_id")
    activity_scope = stages["activity"].get("scope_ids")
    activity_failures = stages["activity"].get("failures")
    if (
        not isinstance(activity_counts, Mapping)
        or not isinstance(activity_scope, list)
        or not isinstance(activity_failures, list)
        or not set(activity_counts).issubset(set(activity_scope))
    ):
        return False
    expected_activity_calls: list[Mapping[str, object]] = []
    for candidate_id in activity_scope:
        count = activity_counts.get(candidate_id)
        if count is None:
            continue
        if not isinstance(count, Mapping):
            return False
        page_audits = count.get("page_audits")
        if not isinstance(page_audits, list) or not page_audits:
            return False
        for page in page_audits:
            query = page.get("query") if isinstance(page, Mapping) else None
            params = (
                page.get("query_parameters")
                if isinstance(page, Mapping)
                else None
            )
            expected_params = {
                "target_chembl_id": bounds.get("target_chembl_id"),
                "parent_molecule_chembl_id": candidate_id,
                "target_organism": "Homo sapiens",
                "assay_type": "B",
                "standard_type": "IC50",
                "pchembl_value__gte": 5.0,
                "order_by": "activity_id",
                "limit": 20,
            }
            page_offset = page.get("offset") if isinstance(page, Mapping) else None
            if page_offset not in (None, 0):
                expected_params["offset"] = page_offset
            if (
                not isinstance(query, Mapping)
                or params != expected_params
                or not _paper_v6_call_matches_params(query, expected_params)
            ):
                return False
            expected_activity_calls.append(query)
    if stages["activity"].get("successful_calls") != (
        _paper_v6_deduplicated_calls(expected_activity_calls)
    ):
        return False
    activity_failure_ids = {
        str(item.get("scope_id"))
        for item in activity_failures
        if isinstance(item, Mapping)
    }
    if set(activity_scope) - set(activity_counts) != activity_failure_ids:
        return False

    def stage_batch_parameters(
        stage_name: str,
        *,
        batch_size: int,
        parameter_name: str,
        extra: Mapping[str, object] | None = None,
    ) -> list[tuple[list[str], dict[str, object]]]:
        scope = stages[stage_name].get("scope_ids")
        if not isinstance(scope, list):
            return []
        output: list[tuple[list[str], dict[str, object]]] = []
        for start in range(0, len(scope), batch_size):
            batch = scope[start : start + batch_size]
            params: dict[str, object] = {
                parameter_name: ",".join(batch),
                "limit": len(batch),
            }
            params.update(dict(extra or {}))
            output.append((batch, params))
        return output

    for stage_name, batch_size, parameter_name in (
        ("assay", 80, "assay_chembl_id__in"),
        ("document", 80, "document_chembl_id__in"),
        ("pubmed_type", 300, "id"),
    ):
        batches = stage_batch_parameters(
            stage_name,
            batch_size=batch_size,
            parameter_name=parameter_name,
            extra=(
                {"db": "pubmed", "retmode": "json", "version": "2.0"}
                if stage_name == "pubmed_type"
                else None
            ),
        )
        if stage_name == "pubmed_type":
            batches = [
                (
                    batch,
                    {
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "retmode": "json",
                        "version": "2.0",
                    },
                )
                for batch, _params in batches
            ]
        calls = stages[stage_name].get("successful_calls")
        failures = stages[stage_name].get("failures")
        if not isinstance(calls, list) or not isinstance(failures, list):
            return False
        expected_hashes = {
            _paper_v6_parameter_hash(params) for _batch, params in batches
        }
        observed_hashes = {
            str(call.get("parameter_hash") or "")
            for call in calls
            if isinstance(call, Mapping)
        }
        if not observed_hashes.issubset(expected_hashes):
            return False
        failure_ids = {
            str(item.get("scope_id") or "")
            for item in failures
            if isinstance(item, Mapping)
        }
        for batch, params in batches:
            parameter_hash = _paper_v6_parameter_hash(params)
            if parameter_hash not in observed_hashes and not set(batch).issubset(
                failure_ids
            ):
                return False

    warning_screen = bounds.get("warning_screen")
    if not isinstance(warning_screen, Mapping):
        return False
    expected_warning_policy = (
        "retryable_batch_then_candidate_isolation_v1"
        if warning_screen.get("batch_fallback_used") is True
        else "candidate_exclusion_on_unresolved_warning_v1"
    )
    if stages["warning"].get("recovery_policy") != expected_warning_policy:
        return False
    warning_scope = stages["warning"].get("scope_ids")
    warning_calls = stages["warning"].get("successful_calls")
    warning_failures = stages["warning"].get("failures")
    if (
        not isinstance(warning_scope, list)
        or not isinstance(warning_calls, list)
        or not isinstance(warning_failures, list)
    ):
        return False
    if warning_screen.get("batch_fallback_used") is False:
        warning_params = {
            "molecule_chembl_id__in": ",".join(warning_scope),
            "limit": max(100, len(warning_scope) * 10),
        }
        if (
            len(warning_calls) != 1
            or not _paper_v6_call_matches_params(
                warning_calls[0], warning_params
            )
            or warning_failures
        ):
            return False
    elif warning_screen.get("batch_fallback_used") is True:
        if not any(
            isinstance(item, Mapping) and item.get("scope_id") == "__batch__"
            for item in warning_failures
        ):
            return False
        individual_hashes = {
            _paper_v6_parameter_hash(
                {"molecule_chembl_id__in": candidate_id, "limit": 100}
            )
            for candidate_id in warning_scope
        }
        if any(
            not isinstance(call, Mapping)
            or call.get("parameter_hash") not in individual_hashes
            for call in warning_calls
        ):
            return False
    else:
        return False

    expected_policies = sorted(
        str(stages[name].get("recovery_policy"))
        for name in _PAPER_V6_SOURCE_STAGE_ORDER
    )
    expected_policies = sorted(set(expected_policies))
    if (
        set(recovery)
        != {
            "schema_version",
            "allowlisted_policies",
            "events",
            "event_count",
            "failure_count",
            "no_unlisted_recovery",
            "fixture",
            "attestation_sha256",
        }
        or recovery_hash != _paper_v6_hash(recovery_body)
        or recovery.get("schema_version")
        != "scientific_source_recovery_attestation_v1"
        or recovery.get("allowlisted_policies") != expected_policies
        or recovery.get("events") != all_failures
        or recovery.get("event_count") != len(all_failures)
        or recovery.get("failure_count") != total_failure
        or recovery.get("no_unlisted_recovery") is not True
        or recovery.get("fixture") is not False
        or inventory.get("successful_call_count") != total_success
        or inventory.get("failure_count") != total_failure
    ):
        return False
    return True


def _paper_v6_group_projection_valid(
    candidates: list[object], rows: list[dict[str, object]], *,
    packet_metadata: Mapping[str, object], requested_eligible_count: int,
    reserved_controls: object = (),
) -> bool:
    if (
        len(rows) < 3
        or len(rows) != len(candidates)
        or any(not isinstance(row, Mapping) for row in rows)
        or any(type(row.get("rank")) is not int for row in rows)
    ):
        return False
    ordered_rows = sorted(rows, key=lambda row: int(row["rank"]))
    ordered_ranks = [int(row["rank"]) for row in ordered_rows]
    # Reserved controls or source-level exclusions may leave gaps in the
    # surviving fixed-pool ranks.  Preserve their original order without
    # falsely requiring the eligible subset to start at rank 1.
    if (
        any(rank < 1 for rank in ordered_ranks)
        or len(ordered_ranks) != len(set(ordered_ranks))
    ):
        return False
    ids = [str(row["candidate_id"]) for row in ordered_rows]
    candidate_by_chembl_id: dict[str, object] = {}
    for candidate in candidates:
        chembl_id = _paired_v4_candidate_chembl_id(candidate)
        if (
            not chembl_id
            or chembl_id in candidate_by_chembl_id
        ):
            return False
        candidate_by_chembl_id[chembl_id] = candidate
    if set(candidate_by_chembl_id) != set(ids):
        return False
    ordered_candidates = [candidate_by_chembl_id[candidate_id] for candidate_id in ids]
    declared_ids = ordered_rows[0]["group_ids"]
    if not isinstance(declared_ids, list) or any(row["group_ids"] != declared_ids for row in ordered_rows):
        return False
    bounds = getattr(candidates[0], "metadata", {}).get("candidate_discovery_bounds")
    if not isinstance(bounds, Mapping) or any(getattr(item, "metadata", {}).get("candidate_discovery_bounds") != bounds for item in candidates[1:]):
        return False
    source_inventory = bounds.get("source_call_inventory")
    if (
        not _paper_v6_source_inventory_valid(bounds)
        or not isinstance(source_inventory, Mapping)
        or source_inventory.get("failure_count") != 0
    ):
        return False
    search = bounds.get("primary_abstract_search")
    aggregation = search.get("candidate_support_aggregation") if isinstance(search, Mapping) else None
    full_units = aggregation.get("candidate_support_units") if isinstance(aggregation, Mapping) else None
    if not isinstance(full_units, list) or not isinstance(aggregation, Mapping):
        return False
    search_body = dict(search) if isinstance(search, Mapping) else {}
    search_hash = search_body.pop("provenance_sha256", None)
    fixed_pool_ids = search.get("fixed_pool_candidate_ids") if isinstance(search, Mapping) else None
    if (
        search_hash != _paper_v6_hash(search_body)
        or search.get("schema_version") != _PAPER_NATIVE_V6_GATEWAY_SCHEMA
        or search.get("policy") != _PAPER_NATIVE_V6_POLICY
        or search.get("aggregation_policy") != _PAPER_NATIVE_V6_AGGREGATION_POLICY
        or search.get("status") != "verified"
        or not isinstance(fixed_pool_ids, list)
        or search.get("fixed_pool_candidate_ids_sha256") != _paper_v6_hash(fixed_pool_ids)
        or search.get("verified_candidate_ids") != declared_ids
        or search.get("verified_candidate_count") != len(declared_ids)
        or search.get("no_reordering") is not True
        or search.get("no_backfill") is not True
        or search.get("fixture") is not False
        or declared_ids != [item for item in fixed_pool_ids if item in set(declared_ids)]
        or any(row.get("fixed_ids") != fixed_pool_ids for row in ordered_rows)
    ):
        return False
    if not _paper_v6_fixed_pool_molecule_projection_valid(
        candidates,
        bounds=bounds,
        fixed_pool_ids=fixed_pool_ids,
    ):
        return False
    full_ids: list[str] = []
    full_hashes: list[str] = []
    expected_target_symbol = ordered_rows[0].get("target_symbol")
    expected_direction = ordered_rows[0].get("direction")
    if not isinstance(expected_target_symbol, str) or not isinstance(expected_direction, str):
        return False
    for unit in full_units:
        reported = _paper_v6_strict_unit(
            unit, expected_target_symbol=expected_target_symbol,
            expected_direction=expected_direction,
        )
        if reported is None or not isinstance(unit, Mapping):
            return False
        full_ids.append(str(unit.get("candidate_id")))
        full_hashes.append(str(reported))
    if full_ids != declared_ids or ids != declared_ids[:len(ids)]:
        return False
    dispositions = search.get("candidate_primary_dispositions")
    if (
        not isinstance(dispositions, list)
        or len(dispositions) != len(fixed_pool_ids)
        or search.get("candidate_primary_disposition_count")
        != len(dispositions)
        or search.get("candidate_primary_dispositions_sha256")
        != _paper_v6_hash(dispositions)
    ):
        return False
    if not _paper_v6_delegate_seed_projection_valid(
        candidates,
        bounds=bounds,
        fixed_pool_ids=fixed_pool_ids,
        dispositions=dispositions,
    ):
        return False
    verified_disposition_ids: list[str] = []
    unverified_dispositions: list[Mapping[str, object]] = []
    support_hash_by_id = dict(zip(full_ids, full_hashes, strict=True))
    for pool_index, disposition in enumerate(dispositions):
        if not isinstance(disposition, Mapping):
            return False
        disposition_body = dict(disposition)
        disposition_hash = disposition_body.pop(
            "disposition_sha256", None
        )
        candidate_id = fixed_pool_ids[pool_index]
        verified = candidate_id in support_hash_by_id
        if (
            set(disposition)
            != {
                "pool_index",
                "fixed_pool_rank",
                "molecule_chembl_id",
                "delegate_returned",
                "verification_status",
                "reason_code",
                "delegate_reason_codes",
                "activity_id",
                "document_chembl_id",
                "pmid",
                "source_primary_verification_status",
                "source_primary_verification_reason_code",
                "source_primary_attestation_sha256",
                "wrapper_screen_status",
                "wrapper_screen_reason_code",
                "wrapper_screen_checks",
                "wrapper_screen_checks_sha256",
                "wrapper_screen_row_sha256",
                "support_unit_sha256",
                "fixture",
                "disposition_sha256",
            }
            or disposition_hash != _paper_v6_hash(disposition_body)
            or disposition.get("pool_index") != pool_index
            or disposition.get("fixed_pool_rank") != pool_index + 1
            or disposition.get("molecule_chembl_id") != candidate_id
            or type(disposition.get("delegate_returned")) is not bool
            or not isinstance(disposition.get("delegate_reason_codes"), list)
            or any(
                not isinstance(item, str) or not item
                for item in disposition.get("delegate_reason_codes", [])
            )
            or disposition.get("fixture") is not False
            or (
                disposition.get("document_chembl_id") is not None
                and (
                    not isinstance(
                        disposition.get("document_chembl_id"), str
                    )
                    or _CHEMBL_STABLE_ID.fullmatch(
                        str(disposition.get("document_chembl_id"))
                    )
                    is None
                )
            )
            or disposition.get("verification_status")
            != ("verified" if verified else "unverified")
            or (
                verified
                and (
                    disposition.get("delegate_returned") is not True
                    or disposition.get("reason_code") is not None
                    or disposition.get("support_unit_sha256")
                    != support_hash_by_id[candidate_id]
                )
            )
            or (
                not verified
                and (
                    not isinstance(disposition.get("reason_code"), str)
                    or not disposition.get("reason_code")
                    or disposition.get("support_unit_sha256") is not None
                )
            )
        ):
            return False
        if verified:
            verified_disposition_ids.append(candidate_id)
        else:
            unverified_dispositions.append(disposition)
    if (
        verified_disposition_ids != declared_ids
        or search.get("rejected_candidate_count")
        != len(unverified_dispositions)
        or search.get("rejected_candidate_projection_sha256")
        != _paper_v6_hash(unverified_dispositions)
    ):
        return False
    if any(
        row.get("unit_hash") != full_hashes[index]
        or row.get("unit") != full_units[index]
        or row.get("target_symbol") != expected_target_symbol
        or row.get("direction") != expected_direction
        for index, row in enumerate(ordered_rows)
    ):
        return False
    ranks = [row["rank"] for row in ordered_rows]
    group_body = {
        "candidate_ids": declared_ids, "member_support_unit_sha256": full_hashes,
        "minimum_verified_candidates": 3, "no_backfill": True, "no_reordering": True,
        "policy": _PAPER_NATIVE_V6_POLICY, "aggregation_policy": _PAPER_NATIVE_V6_AGGREGATION_POLICY,
    }
    expected_group_hash = _paper_v6_hash(group_body)
    if (
        aggregation.get("schema_version") != _PAPER_NATIVE_V6_AGGREGATION_SCHEMA
        or aggregation.get("policy") != _PAPER_NATIVE_V6_POLICY
        or aggregation.get("aggregation_policy") != _PAPER_NATIVE_V6_AGGREGATION_POLICY
        or aggregation.get("status") != "verified"
        or aggregation.get("candidate_ids") != declared_ids
        or aggregation.get("candidate_count") != len(declared_ids)
        or aggregation.get("candidate_ids_order_sha256") != _paper_v6_hash(declared_ids)
        or aggregation.get("candidate_ids_set_sha256") != _paper_v6_hash(sorted(declared_ids))
        or aggregation.get("group_sha256") != expected_group_hash
        or aggregation.get("no_reordering") is not True
        or aggregation.get("no_backfill") is not True
        or ranks != sorted(ranks) or len(set(ranks)) != len(ranks)
        or any(
            row["group_sha256"] != expected_group_hash
            for row in ordered_rows
        )
    ):
        return False
    projection = getattr(candidates[0], "metadata", {}).get("paper_native_v6_group_projection")
    if not isinstance(projection, Mapping) or any(getattr(item, "metadata", {}).get("paper_native_v6_group_projection") != projection for item in candidates[1:]):
        return False
    candidate_yield = packet_metadata.get("candidate_yield")
    candidate_backfill = packet_metadata.get("candidate_backfill")
    if (
        not isinstance(candidate_yield, Mapping)
        or not isinstance(candidate_backfill, Mapping)
        or candidate_yield.get("paper_native_v6_group_projection") != projection
        or candidate_backfill.get("paper_native_v6_group_projection") != projection
    ):
        return False
    control_attestation = candidate_yield.get(
        "paper_native_v6_full_pool_control_screening"
    )
    if (
        not isinstance(control_attestation, Mapping)
        or candidate_backfill.get("paper_native_v6_full_pool_control_screening") != control_attestation
    ):
        return False
    control_body = dict(control_attestation)
    control_hash = control_body.pop("attestation_sha256", None)
    control_rows = control_attestation.get("rows")
    if (
        control_hash != _paper_v6_hash(control_body)
        or control_attestation.get("schema_version") != "paper_native_v6_full_pool_control_screening/v1"
        or control_attestation.get("policy") != _PAPER_NATIVE_V6_POLICY
        or control_attestation.get("fixture") is not False
        or not isinstance(control_rows, list)
        or control_attestation.get("row_count") != len(declared_ids)
        or len(control_rows) != len(declared_ids)
    ):
        return False
    for index, control_row in enumerate(control_rows):
        if not isinstance(control_row, Mapping):
            return False
        identity = control_row.get("candidate_identity")
        if not isinstance(identity, Mapping):
            return False
        identity_metadata = control_row.get("candidate_registry_metadata")
        expected_registry_metadata = {"chembl_id": declared_ids[index]}
        if identity_metadata != expected_registry_metadata:
            return False
        candidate_name = str(control_row.get("candidate_name") or "")
        recomputed_identity = candidate_identity(
            candidate_name=candidate_name, metadata=identity_metadata
        )
        expected_registry_ids = [{
            "namespace": "chembl",
            "identifier": declared_ids[index].casefold(),
        }]
        if (
            identity != recomputed_identity
            or identity.get("name") != candidate_name
            or identity.get("registry_ids") != expected_registry_ids
        ):
            return False
        screened = screen_reserved_control(
            candidate_name=candidate_name,
            metadata=identity_metadata,
            reserved_controls=reserved_controls,
        )
        if (
            control_row.get("pool_index") != index
            or control_row.get("reserved_control_screen") != screened.as_metadata()
            or control_row.get("reserved_excluded") is not screened.excluded
            or control_row.get("control_only") is not screened.matched
            or control_row.get("downstream_evaluated") is not (index < len(ids))
        ):
            return False
    eligible_count = sum(
        getattr(item, "status", None) != "excluded"
        for item in ordered_candidates
    )
    cumulative = 0
    minimum_prefix: int | None = None
    for index, candidate in enumerate(ordered_candidates):
        if getattr(candidate, "status", None) != "excluded":
            cumulative += 1
        if cumulative >= requested_eligible_count:
            minimum_prefix = index + 1
            break
    if minimum_prefix != len(ordered_candidates):
        return False
    unevaluated = candidate_backfill.get("unevaluated_identities")
    if not isinstance(unevaluated, list):
        return False
    expected_evaluated: list[dict[str, object]] = []
    for index, candidate in enumerate(ordered_candidates):
        reason_codes = getattr(candidate, "exclusion_reasons", [])
        if not isinstance(reason_codes, list) or any(
            not isinstance(item, str) for item in reason_codes
        ):
            return False
        expected_evaluated.append({
            **dict(control_rows[index]["candidate_identity"]),
            "pool_index": index,
            "candidate_id": getattr(candidate, "candidate_id", None),
            "outcome": (
                "eligible"
                if getattr(candidate, "status", None) != "excluded"
                else "excluded"
            ),
            "reason_codes": list(reason_codes),
        })
    for offset, item in enumerate(unevaluated):
        pool_index = len(ids) + offset
        if pool_index >= len(control_rows) or not isinstance(item, Mapping):
            return False
        expected_item = {
            **dict(control_rows[pool_index]["candidate_identity"]),
            "pool_index": pool_index,
            "reason_codes": [
                "ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX"
            ],
        }
        if dict(item) != expected_item:
            return False
    body = dict(projection)
    reported = body.pop("projection_sha256", None)
    discovery_pool_limit = candidate_backfill.get("discovery_pool_limit")
    expected_pool_exhausted = len(ids) == len(declared_ids)
    return bool(
        reported == _paper_v6_hash(body)
        and projection.get("schema_version") == "paper_native_v6_group_projection_v1"
        and projection.get("policy") == _PAPER_NATIVE_V6_POLICY
        and projection.get("declared_group_candidate_ids") == declared_ids
        and projection.get("declared_group_candidate_ids_sha256") == _paper_v6_hash(declared_ids)
        and projection.get("projected_group_candidate_ids") == ids
        and projection.get("projected_group_candidate_ids_sha256") == _paper_v6_hash(ids)
        and projection.get("projection_complete") is True
        and projection.get("projection_scope") == "exact_evaluated_fixed_pool_prefix"
        and projection.get("projected_is_exact_declared_prefix") is True
        and projection.get("complete_declared_group_projected") is (ids == declared_ids)
        and projection.get("unevaluated_declared_group_candidate_ids") == declared_ids[len(ids):]
        and projection.get("unevaluated_declared_group_candidate_ids_sha256") == _paper_v6_hash(declared_ids[len(ids):])
        and projection.get("eligible_quota_reached") is True
        and projection.get("requested_eligible_count") == requested_eligible_count
        and projection.get("eligible_returned_count") == eligible_count
        and eligible_count >= requested_eligible_count
        and projection.get("evaluated_fixed_pool_prefix_count") == len(ids)
        and projection.get("stop_rule") == _PAPER_NATIVE_V6_PROJECTION_STOP_RULE
        and projection.get("stop_rule_sha256") == sha256(_PAPER_NATIVE_V6_PROJECTION_STOP_RULE.encode("utf-8")).hexdigest()
        and projection.get("stop_condition_satisfied") is True
        and projection.get("no_second_discovery") is True
        and projection.get("no_backfill_beyond_fixed_pool") is True
        and projection.get("downstream_evaluation_cap_enforced") is True
        and projection.get("fixture") is False
        and candidate_backfill.get("evaluated_count") == len(ids)
        and candidate_backfill.get("evaluated_identities") == expected_evaluated
        and candidate_backfill.get("eligible_count") == eligible_count
        and candidate_backfill.get("excluded_count") == len(ids) - eligible_count
        and candidate_backfill.get("quota_met") is True
        and candidate_backfill.get("stop_reason") == "ELIGIBLE_QUOTA_REACHED_IN_EXACT_V6_VERIFIED_POOL_PREFIX"
        and candidate_backfill.get("pool_exhausted") is expected_pool_exhausted
        and candidate_backfill.get("unbounded_search_used") is False
        and type(discovery_pool_limit) is int
        and len(declared_ids) <= discovery_pool_limit <= 100
        and candidate_yield.get("evaluated_count") == len(ids)
        and candidate_yield.get("eligible_count") == eligible_count
        and candidate_yield.get("quota_met") is True
        and candidate_yield.get("requested_limit") == requested_eligible_count
        and candidate_yield.get("requested_eligible_count") == requested_eligible_count
        and candidate_yield.get("backend_kind") == ScientificBackendKind.LIVE.value
        and candidate_yield.get("partial") is False
        and candidate_yield.get("fixture_fill_used") is False
        and candidate_yield.get("pool_exhausted") is expected_pool_exhausted
        and candidate_yield.get("returned_count") == len(declared_ids)
        and candidate_yield.get("discovery_pool_count") == len(declared_ids)
        and candidate_backfill.get("discovery_pool_count") == len(declared_ids)
        and len(unevaluated) == len(declared_ids) - len(ids)
    )


def _paper_v5_self_hash(
    value: object,
    *,
    key: str,
) -> bool:
    if not isinstance(value, dict):
        return False
    body = dict(value)
    reported = body.pop(key, None)
    return isinstance(reported, str) and reported == _paper_v5_hash(body)


def _paper_v5_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _paper_v5_strings(
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if len(value) < minimum or (maximum is not None and len(value) > maximum):
        return None
    if any(
        not isinstance(item, str)
        or not item
        or item != item.strip()
        or "\r" in item
        or "\n" in item
        for item in value
    ):
        return None
    keys = [_paper_v5_key(item) for item in value]
    if len(set(keys)) != len(keys):
        return None
    return list(value)


def _paper_v5_exact_document_id(value: object) -> str | None:
    """Accept only an exact source-shaped document identifier."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CHEMBL_STABLE_ID.fullmatch(value) is None
    ):
        return None
    return value


def _paper_v5_ordered_subsequence(
    values: list[str],
    reference: list[str],
) -> bool:
    positions = {value: index for index, value in enumerate(reference)}
    try:
        observed = [positions[value] for value in values]
    except KeyError:
        return False
    return observed == sorted(observed) and len(set(values)) == len(values)


def _paper_v5_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() and result > 0 else None


def _paper_v5_query_threshold(value: object) -> Decimal | None:
    """Accept only a deposited JSON number for the approved query floor."""

    if type(value) not in (int, float):
        return None
    return _paper_v5_decimal(value)


def _paper_v5_chembl_retry_contract(query: Mapping[str, object]) -> bool:
    """Validate the frozen two-attempt ChEMBL transport contract."""

    attempt_count = query.get("attempt_count")
    max_attempts = query.get("max_attempts")
    attempts = query.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        return False
    attempt = attempts[0]
    expected_attempt_keys = {
        "attempt",
        "transport_attempt_count",
        "timeout_seconds",
        "status",
        "status_code",
        "runtime_seconds",
        "retrying",
    }
    if not isinstance(attempt, dict) or set(attempt) != expected_attempt_keys:
        return False
    timeout_seconds = _paper_v5_decimal(attempt.get("timeout_seconds"))
    query_timeout_seconds = _paper_v5_decimal(query.get("timeout_seconds"))
    runtime_seconds = attempt.get("runtime_seconds")
    runtime_decimal = (
        Decimal(str(runtime_seconds))
        if type(runtime_seconds) in (int, float)
        else None
    )
    return bool(
        type(max_attempts) is int
        and max_attempts == 2
        and type(attempt_count) is int
        and 1 <= attempt_count <= max_attempts
        and type(attempt.get("attempt")) is int
        and attempt.get("attempt") == 1
        and type(attempt.get("transport_attempt_count")) is int
        and attempt.get("transport_attempt_count") == attempt_count
        and timeout_seconds is not None
        and timeout_seconds == query_timeout_seconds
        and attempt.get("status") == "completed"
        and type(attempt.get("status_code")) is int
        and attempt.get("status_code") == 200
        and attempt.get("status_code") == query.get("status_code")
        and runtime_decimal is not None
        and runtime_decimal.is_finite()
        and runtime_decimal >= 0
        and attempt.get("retrying") is False
    )


def _paper_v5_identity_tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _paper_v5_has_target_token_equivalent(
    value: object,
    target_aliases: Iterable[object],
) -> bool:
    """Match a whole target token, allowing punctuation-only token splits.

    Consecutive alphanumeric tokens may be joined, so ``GENE1`` is present in
    ``GENE-1 inhibitor IV``.  A target is never matched as a substring of one
    larger token (for example, ``MET`` does not match ``metformin``).
    """

    value_tokens = _paper_v5_identity_tokens(value)
    if not value_tokens:
        return False
    for raw_target in target_aliases:
        target_tokens = _paper_v5_identity_tokens(raw_target)
        target_key = "".join(target_tokens)
        if not target_key:
            continue
        for start in range(len(value_tokens)):
            candidate_key = ""
            for token in value_tokens[start:]:
                candidate_key += token
                if candidate_key == target_key:
                    return True
                if len(candidate_key) >= len(target_key):
                    break
    return False


def _paper_v5_gateway_parameter_hash(params: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(params),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _paper_v5_mechanism_pool(
    raw: Mapping[str, object],
    *,
    fixed_ids: list[str],
    eligible_ids: list[str],
    preactivity_pool: list[object],
) -> dict[str, object] | None:
    """Recompute the fixed pool from deposited mechanism identities."""

    discovered_ids = _paper_v5_strings(
        raw.get("discovered_identity_ids"), minimum=len(fixed_ids)
    )
    valid_ids = _paper_v5_strings(
        raw.get("valid_mechanism_identity_ids"), minimum=len(fixed_ids)
    )
    invalid_ids = _paper_v5_strings(raw.get("invalid_mechanism_identity_ids"))
    resolution = raw.get("identity_resolution")
    provenance = raw.get("mechanism_provenance_by_id")
    target_id = raw.get("target_chembl_id")
    mechanism_query = raw.get("mechanism_query")
    mechanism_params: dict[str, object] = {
        "target_chembl_id": target_id,
        "action_type": "INHIBITOR",
        "max_phase__gte": 1,
        "direct_interaction": 1,
        "molecular_mechanism": 1,
        "order_by": "molecule_chembl_id",
        "limit": 80,
    }
    response_count = raw.get("raw_mechanism_response_count")
    total_count = raw.get("mechanism_total_count")
    if (
        discovered_ids is None
        or valid_ids is None
        or invalid_ids is None
        or not isinstance(resolution, list)
        or not isinstance(provenance, dict)
        or not isinstance(target_id, str)
        or not target_id
        or not isinstance(mechanism_query, dict)
        or raw.get("raw_mechanism_row_limit") != 80
        or type(response_count) is not int
        or not 0 <= response_count <= 80
        or type(total_count) is not int
        or total_count != response_count
        or raw.get("mechanism_complete_within_bound") is not True
        or not _SHA256.fullmatch(
            str(raw.get("raw_mechanism_rows_sha256") or "")
        )
        or raw.get("discovered_identity_count") != len(discovered_ids)
        or raw.get("valid_mechanism_identity_count") != len(valid_ids)
        or set(valid_ids).intersection(invalid_ids)
        or set(valid_ids).union(invalid_ids) != set(discovered_ids)
        or set(provenance) != set(valid_ids)
        or not set(fixed_ids).issubset(valid_ids)
        or mechanism_query.get("source") != "ChEMBL"
        or mechanism_query.get("endpoint")
        != _PAIRED_V4_CHEMBL_MECHANISM_ENDPOINT
        or mechanism_query.get("parameter_keys") != sorted(mechanism_params)
        or mechanism_query.get("parameter_hash")
        != _paper_v5_gateway_parameter_hash(mechanism_params)
        or mechanism_query.get("status_code") != 200
        or not _paper_v5_chembl_retry_contract(mechanism_query)
        or mechanism_query.get("exhausted") is not False
        or mechanism_query.get("cache_hit") is not False
        or mechanism_query.get("operation") != "retrieve_repurposing"
        or not isinstance(mechanism_query.get("started_at"), str)
        or not isinstance(mechanism_query.get("finished_at"), str)
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
            str(mechanism_query.get("started_at") or ""),
        )
        is None
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
            str(mechanism_query.get("finished_at") or ""),
        )
        is None
        or str(mechanism_query.get("finished_at"))
        < str(mechanism_query.get("started_at"))
        or not _SHA256.fullmatch(
            str(mechanism_query.get("response_sha256") or "")
        )
    ):
        return None
    resolution_by_id: dict[str, dict[str, object]] = {}
    status_selected_ids: set[str] = set()
    preactivity_selected_ids: set[str] = set()
    preactivity_ranked_ids: set[str] = set()
    for item in resolution:
        if not isinstance(item, dict):
            return None
        molecule_id = item.get("molecule_chembl_id")
        parent_id = item.get("parent_molecule_chembl_id")
        status = item.get("status")
        reason_codes = item.get("reason_codes")
        if (
            not isinstance(molecule_id, str)
            or molecule_id not in discovered_ids
            or molecule_id in resolution_by_id
            or parent_id != molecule_id
            or not isinstance(status, str)
            or status not in {"selected", "excluded"}
            or not isinstance(reason_codes, list)
            or any(not isinstance(code, str) or not code for code in reason_codes)
            or len(reason_codes) != len(set(reason_codes))
        ):
            return None
        selected_marker = "PREACTIVITY_POOL_SELECTED" in reason_codes
        limit_marker = "PREACTIVITY_POOL_LIMIT_REACHED" in reason_codes
        expected_selected = molecule_id in fixed_ids
        expected_limited = (
            molecule_id in eligible_ids and molecule_id not in fixed_ids
        )
        if (
            selected_marker != expected_selected
            or limit_marker != expected_limited
            or status != ("selected" if expected_selected else "excluded")
        ):
            return None
        if selected_marker:
            preactivity_selected_ids.add(molecule_id)
            preactivity_ranked_ids.add(molecule_id)
        elif limit_marker:
            preactivity_ranked_ids.add(molecule_id)
        if status == "selected":
            status_selected_ids.add(molecule_id)
        resolution_by_id[molecule_id] = item
    if set(resolution_by_id) != set(discovered_ids):
        return None
    if (
        preactivity_ranked_ids != set(eligible_ids)
        or preactivity_selected_ids != set(fixed_ids)
        or status_selected_ids != set(fixed_ids)
    ):
        return None

    phase_by_id: dict[str, int] = {}
    mechanisms_by_id: dict[str, dict[str, object]] = {}
    for molecule_id in valid_ids:
        mechanism = provenance.get(molecule_id)
        if not isinstance(mechanism, dict):
            return None
        rows = mechanism.get("rows")
        if (
            not isinstance(rows, list)
            or not rows
            or mechanism.get("schema_version")
            != "chembl_mechanism_identity_v1"
            or mechanism.get("parent_molecule_chembl_id") != molecule_id
            or mechanism.get("row_count") != len(rows)
            or mechanism.get("rows_sha256") != _paper_v5_hash(rows)
            or mechanism.get("query") != mechanism_query
            or mechanism.get("fixture") is not False
            or mechanism.get("missing_values_imputed") is not False
        ):
            return None
        row_phases: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                return None
            phase_decimal = _paper_v5_decimal(row.get("max_phase"))
            phase = int(phase_decimal) if phase_decimal is not None else 0
            refs = row.get("mechanism_refs")
            if (
                phase_decimal is None
                or phase_decimal != Decimal(phase)
                or not 1 <= phase <= 4
                or row.get("canonical_parent_molecule_chembl_id")
                != molecule_id
                or row.get("parent_molecule_chembl_id") not in (
                    None,
                    "",
                    molecule_id,
                )
                or not isinstance(row.get("molecule_chembl_id"), str)
                or not row.get("molecule_chembl_id")
                or row.get("target_chembl_id") != target_id
                or row.get("action_type") != "INHIBITOR"
                or row.get("direct_interaction") not in (1, "1", True)
                or row.get("molecular_mechanism") not in (1, "1", True)
                or not isinstance(refs, list)
                or not refs
                or not any(
                    isinstance(ref, dict)
                    and isinstance(ref.get("ref_type"), str)
                    and bool(ref.get("ref_type"))
                    and bool(ref.get("ref_id") or ref.get("ref_url"))
                    for ref in refs
                )
            ):
                return None
            row_phases.add(phase)
        if len(row_phases) != 1:
            return None
        phase_by_id[molecule_id] = next(iter(row_phases))
        mechanisms_by_id[molecule_id] = mechanism
    expected_identity_hash = _paper_v5_hash(
        {
            molecule_id: mechanisms_by_id[molecule_id]["rows_sha256"]
            for molecule_id in sorted(mechanisms_by_id)
        }
    )
    if raw.get("mechanism_identity_sha256") != expected_identity_hash:
        return None

    eligible_phase_by_id: dict[str, int] = {}
    for molecule_id in eligible_ids:
        item = resolution_by_id.get(molecule_id)
        phase = item.get("max_phase") if isinstance(item, dict) else None
        if (
            item is None
            or molecule_id not in phase_by_id
            or type(phase) is not int
            or not 1 <= phase <= 4
            or phase != phase_by_id[molecule_id]
        ):
            return None
        eligible_phase_by_id[molecule_id] = phase
    expected_eligible = sorted(
        eligible_ids,
        key=lambda molecule_id: (-eligible_phase_by_id[molecule_id], molecule_id),
    )
    expected_fixed = expected_eligible[:_PAPER_NATIVE_V5_MAX_CANDIDATES]
    expected_pool = [
        {
            "rank": rank,
            "parent_molecule_chembl_id": molecule_id,
            "max_phase": eligible_phase_by_id[molecule_id],
        }
        for rank, molecule_id in enumerate(expected_fixed, start=1)
    ]
    if (
        eligible_ids != expected_eligible
        or fixed_ids != expected_fixed
        or preactivity_pool != expected_pool
    ):
        return None
    return {
        "phase_by_id": eligible_phase_by_id,
        "mechanisms_by_id": mechanisms_by_id,
        "resolution_by_id": resolution_by_id,
    }


def _paper_v5_discovery_bounds(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    fixed_ids = _paper_v5_strings(
        raw.get("prewarning_pool_ids"),
        minimum=1,
        maximum=_PAPER_NATIVE_V5_MAX_CANDIDATES,
    )
    eligible_ids = _paper_v5_strings(raw.get("eligible_identity_ids"), minimum=1)
    preactivity_ids = _paper_v5_strings(
        raw.get("preactivity_pool_ids"),
        minimum=1,
        maximum=_PAPER_NATIVE_V5_MAX_CANDIDATES,
    )
    downstream_ids = _paper_v5_strings(
        raw.get("downstream_candidate_identity_ids")
    )
    attempted_ids = _paper_v5_strings(raw.get("activity_query_attempted_ids"))
    direct_ids = _paper_v5_strings(
        raw.get("paper_native_v5_direct_support_candidate_ids"),
        minimum=3,
        maximum=_PAPER_NATIVE_V5_MAX_CANDIDATES,
    )
    snapshots = raw.get("paper_native_v5_direct_support_snapshots")
    preactivity_pool = raw.get("preactivity_pool")
    if (
        fixed_ids is None
        or eligible_ids is None
        or preactivity_ids is None
        or downstream_ids is None
        or attempted_ids is None
        or direct_ids is None
        or not isinstance(snapshots, list)
        or not isinstance(preactivity_pool, list)
        or len(preactivity_pool) != len(fixed_ids)
        or raw.get("schema_version")
        != "bounded_mechanism_first_candidate_identity_pool_v1"
        or raw.get("discovery_lane")
        != "chembl_mechanism_first_single_default_v1"
        or raw.get("target_type") != "SINGLE PROTEIN"
        or raw.get("target_organism") != "Homo sapiens"
        or raw.get("mechanism_action_type") != "INHIBITOR"
        or raw.get("prewarning_pool_limit")
        != _PAPER_NATIVE_V5_MAX_CANDIDATES
        or raw.get("prewarning_pool_count") != len(fixed_ids)
        or raw.get("preactivity_pool_count") != len(fixed_ids)
        or preactivity_ids != fixed_ids
        or fixed_ids != eligible_ids[:_PAPER_NATIVE_V5_MAX_CANDIDATES]
        or raw.get("prewarning_selection_policy")
        != "clinical_max_phase_desc_then_parent_molecule_chembl_id_v1"
        or raw.get("selection_inputs")
        != ["max_phase", "parent_molecule_chembl_id"]
        or raw.get("pchembl_used_for_prewarning_selection") is not False
        or raw.get("primary_document_used_for_prewarning_selection") is not False
        or raw.get("warning_used_for_prewarning_selection") is not False
        or raw.get("activity_used_for_prewarning_selection") is not False
        or raw.get("no_activity_first_fallback") is not True
        or raw.get("no_backfill_beyond_prewarning_pool") is not True
        or raw.get("fixture") is not False
        or raw.get("missing_values_imputed") is not False
        or raw.get("paper_native_v5_fixed_pool_candidate_ids_sha256")
        != _paper_v5_hash(fixed_ids)
        or raw.get("paper_native_v5_direct_support_snapshots_sha256")
        != _paper_v5_hash(snapshots)
        or raw.get("paper_native_v5_pubmed_used_for_support_selection") is not False
        or raw.get("paper_native_v5_pubmed_value_used_for_ordering") is not False
        or raw.get("paper_native_v5_no_reordering") is not True
        or raw.get("paper_native_v5_no_backfill") is not True
        or raw.get("legacy_primary_document_lookup_executed") is not False
        or raw.get("legacy_primary_document_lookup_candidate_count") != 0
        or raw.get("legacy_primary_document_lookup_candidate_ids") != []
        or raw.get("legacy_primary_document_lookup_document_count") != 0
        or raw.get("legacy_primary_document_lookup_document_ids") != []
        or raw.get("legacy_pubmed_type_lookup_executed") is not False
        or raw.get("legacy_pubmed_type_lookup_candidate_count") != 0
        or raw.get("legacy_pubmed_type_lookup_candidate_ids") != []
        or raw.get("legacy_pubmed_type_lookup_pmid_count") != 0
        or raw.get("legacy_pubmed_type_lookup_pmid_ids") != []
        or raw.get("document_lookup_candidate_identity_count") != 0
        or raw.get("document_lookup_candidate_identity_ids") != []
        or raw.get("document_lookup_id_count") != 0
        or raw.get("pubmed_type_lookup_candidate_identity_count") != 0
        or raw.get("pubmed_type_lookup_candidate_identity_ids") != []
        or raw.get("pubmed_type_lookup_pmid_count") != 0
        or not _paper_v5_ordered_subsequence(downstream_ids, fixed_ids)
        or not _paper_v5_ordered_subsequence(direct_ids, fixed_ids)
        or attempted_ids != fixed_ids
        or len(snapshots) != len(direct_ids)
    ):
        return None
    mechanism_pool = _paper_v5_mechanism_pool(
        raw,
        fixed_ids=fixed_ids,
        eligible_ids=eligible_ids,
        preactivity_pool=preactivity_pool,
    )
    if mechanism_pool is None:
        return None
    activity_counts = raw.get("activity_counts_by_id")
    resolution_by_id = mechanism_pool["resolution_by_id"]
    complete_pagination = (
        raw.get("activity_complete_pagination_required") is True
    )
    if (
        raw.get("activity_rows_per_candidate_limit") != 20
        or not isinstance(activity_counts, dict)
        or set(activity_counts) != set(fixed_ids)
    ):
        return None
    page_audits_by_id = raw.get("activity_page_audits_by_id")
    if complete_pagination and (
        raw.get("activity_query_policy")
        != (
            "complete_exact_parent_activity_pagination_v2; 20-row wire "
            "pages; all reported rows inspected; IC50 direction filter; "
            "no identity replacement; partial pages discarded"
        )
        or raw.get("activity_rows_per_candidate_limit_semantics")
        != "wire_page_size_not_result_cap"
        or raw.get("activity_result_row_limit") is not None
        or not isinstance(page_audits_by_id, dict)
        or set(page_audits_by_id) != set(fixed_ids)
        or raw.get("activity_page_audits_by_id_sha256")
        != _paper_v5_hash(page_audits_by_id)
    ):
        return None
    observed_page_count = 0
    for candidate_id in fixed_ids:
        count = activity_counts.get(candidate_id)
        resolution = resolution_by_id.get(candidate_id)
        if (
            not isinstance(count, dict)
            or not isinstance(resolution, dict)
            or isinstance(count.get("total_count"), bool)
            or not isinstance(count.get("total_count"), int)
            or count.get("total_count") < 0
        ):
            return None
        if complete_pagination:
            total_count = count["total_count"]
            pages = count.get("page_audits")
            if (
                count.get("response_count") != total_count
                or count.get("inspected_count") != total_count
                or count.get("inspected_cap") is not None
                or count.get("truncated") is not False
                or count.get("complete_result_set") is not True
                or count.get("page_size") != 20
                or not isinstance(pages, list)
                or count.get("page_count") != len(pages)
                or count.get("page_audits_sha256") != _paper_v5_hash(pages)
                or pages != page_audits_by_id.get(candidate_id)
                or "ACTIVITY_TOTAL_COUNT_EXCEEDS_BOUND"
                in resolution.get("reason_codes", [])
            ):
                return None
            expected_page_count = max(1, (total_count + 19) // 20)
            if len(pages) != expected_page_count:
                return None
            observed_page_count += len(pages)
            previous_activity_id: int | None = None
            observed_ids: set[str] = set()
            observed_rows = 0
            for page_index, page in enumerate(pages):
                if not isinstance(page, dict) or not _paper_v5_self_hash(
                    page, key="page_audit_sha256"
                ):
                    return None
                offset = page_index * 20
                expected_count = min(20, max(0, total_count - offset))
                activity_ids = page.get("activity_ids")
                query = page.get("query")
                params = page.get("query_parameters")
                expected_params: dict[str, object] = {
                    "target_chembl_id": raw.get("target_chembl_id"),
                    "parent_molecule_chembl_id": candidate_id,
                    "target_organism": "Homo sapiens",
                    "assay_type": "B",
                    "standard_type": "IC50",
                    "pchembl_value__gte": 5.0,
                    "order_by": "activity_id",
                    "limit": 20,
                }
                if page_index:
                    expected_params["offset"] = offset
                if (
                    page.get("page_index") != page_index
                    or page.get("offset") != offset
                    or page.get("limit") != 20
                    or page.get("response_count") != expected_count
                    or page.get("reported_total_count") != total_count
                    or not isinstance(activity_ids, list)
                    or len(activity_ids) != expected_count
                    or page.get("activity_ids_sha256")
                    != _paper_v5_hash(activity_ids)
                    or not _SHA256.fullmatch(
                        str(page.get("response_rows_sha256") or "")
                    )
                    or params != expected_params
                    or not isinstance(query, dict)
                    or query.get("source") != "ChEMBL"
                    or query.get("endpoint")
                    != "https://www.ebi.ac.uk/chembl/api/data/activity.json"
                    or query.get("parameter_keys") != sorted(expected_params)
                    or query.get("parameter_hash")
                    != _paper_v5_gateway_parameter_hash(expected_params)
                    or query.get("status_code") != 200
                    or query.get("cache_hit") is not False
                    or query.get("exhausted") is not False
                    or query.get("operation") != "retrieve_repurposing"
                    or not _paper_v5_chembl_retry_contract(query)
                    or not _SHA256.fullmatch(
                        str(query.get("response_sha256") or "")
                    )
                ):
                    return None
                for activity_id in activity_ids:
                    if not isinstance(activity_id, str):
                        return None
                    try:
                        numeric_activity_id = int(activity_id)
                    except ValueError:
                        return None
                    if (
                        numeric_activity_id < 1
                        or activity_id != str(numeric_activity_id)
                        or activity_id in observed_ids
                        or (
                            previous_activity_id is not None
                            and numeric_activity_id <= previous_activity_id
                        )
                    ):
                        return None
                    observed_ids.add(activity_id)
                    previous_activity_id = numeric_activity_id
                observed_rows += expected_count
            if observed_rows != total_count:
                return None
        else:
            if count.get("inspected_cap") != 20:
                return None
            if count["total_count"] > 20 and (
                candidate_id in direct_ids
                or count.get("truncated") is not True
                or count.get("retained_document_rows") != 0
                or count.get("eligible_directional_count") != 0
                or resolution.get("status") != "excluded"
                or "ACTIVITY_TOTAL_COUNT_EXCEEDS_BOUND"
                not in resolution.get("reason_codes", [])
            ):
                return None
    if complete_pagination and raw.get(
        "activity_logical_call_count"
    ) != observed_page_count:
        return None
    expected_pool: list[dict[str, object]] = []
    previous_phase: int | None = None
    previous_id: str | None = None
    for rank, (candidate_id, row) in enumerate(
        zip(fixed_ids, preactivity_pool, strict=False), start=1
    ):
        if (
            not isinstance(row, dict)
            or row.get("rank") != rank
            or row.get("parent_molecule_chembl_id") != candidate_id
            or isinstance(row.get("max_phase"), bool)
            or not isinstance(row.get("max_phase"), int)
        ):
            return None
        phase = int(row["max_phase"])
        if previous_phase is not None and (
            phase > previous_phase
            or (phase == previous_phase and str(previous_id) >= candidate_id)
        ):
            return None
        previous_phase = phase
        previous_id = candidate_id
        expected_pool.append(row)
    if len(expected_pool) != len(fixed_ids):
        return None
    return {
        "fixed_ids": fixed_ids,
        "direct_ids": direct_ids,
        "snapshots": snapshots,
        "downstream_ids": downstream_ids,
        "target_chembl_id": raw.get("target_chembl_id"),
        "support_inventory": raw.get(
            "paper_native_v5_direct_support_inventory"
        ),
        "primary_search": raw.get("primary_abstract_search"),
        "phase_by_id": mechanism_pool["phase_by_id"],
        "mechanisms_by_id": mechanism_pool["mechanisms_by_id"],
        "resolution_by_id": mechanism_pool["resolution_by_id"],
        "complete_activity_pagination": complete_pagination,
        "activity_pages_by_id": (
            page_audits_by_id if complete_pagination else None
        ),
        "canonical_sha256": _paper_v5_hash(raw),
    }


def _paper_v5_synonym_source_inventory(
    raw: object,
    *,
    expected_count: int,
) -> dict[int, tuple[str, str]] | None:
    """Verify the deposited ChEMBL synonym source independently of its ledger."""

    if not isinstance(raw, dict) or not _paper_v5_self_hash(
        raw, key="inventory_sha256"
    ):
        return None
    rows = raw.get("rows")
    rows_sha256 = raw.get("rows_sha256")
    observed_sha256 = raw.get("observed_value_sha256")
    row_limit = 2 * _MAX_EXACT_CANDIDATE_ALIASES
    if (
        raw.get("schema_version")
        != "paper_native_v5_molecule_synonyms_source_v1"
        or raw.get("source") != "ChEMBL.molecule.molecule_synonyms"
        or raw.get("field") != "molecule_synonyms"
        or raw.get("source_key_present") is not True
        or raw.get("list_typed") is not True
        or raw.get("source_value_type") != "list"
        or type(raw.get("source_row_count")) is not int
        or raw.get("source_row_count") != expected_count
        or not 0 <= expected_count <= row_limit
        or raw.get("mapping_row_count") != expected_count
        or raw.get("non_mapping_row_indices") != []
        or not isinstance(rows_sha256, str)
        or _SHA256.fullmatch(rows_sha256) is None
        or observed_sha256 != rows_sha256
        or raw.get("bounded_row_limit") != row_limit
        or raw.get("inspected_row_count") != expected_count
        or raw.get("truncated") is not False
        or not isinstance(rows, list)
        or len(rows) != expected_count
        or raw.get("reason_codes") != []
        or raw.get("complete") is not True
    ):
        return None
    source_by_index: dict[int, tuple[str, str]] = {}
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        row_keys = row.get("row_keys")
        syn_type = row.get("syn_type")
        synonym = row.get("molecule_synonym")
        if (
            row.get("source_index") != expected_index
            or row.get("row_type") != "mapping"
            or not isinstance(row_keys, list)
            or any(not isinstance(key, str) or not key for key in row_keys)
            or row_keys != sorted(set(row_keys))
            or not {"molecule_synonym", "syn_type"}.issubset(row_keys)
            or not isinstance(syn_type, str)
            or not syn_type
            or syn_type != syn_type.strip()
            or not isinstance(synonym, str)
            or not synonym
            or synonym != synonym.strip()
            or row.get("reason_codes") != []
        ):
            return None
        source_by_index[expected_index] = (syn_type, synonym)
    return source_by_index


def _paper_v5_alias_inventory(
    raw: object,
    *,
    fixed_ids: list[str],
    direct_ids: list[str],
    target: object,
) -> dict[str, object] | None:
    if not isinstance(raw, dict) or not _paper_v5_self_hash(
        raw, key="inventory_sha256"
    ):
        return None
    target_aliases = _paper_v5_strings(
        raw.get("target_aliases"), minimum=1, maximum=8
    )
    queried_ids = _paper_v5_strings(
        raw.get("queried_candidate_ids"),
        minimum=3,
        maximum=_PAPER_NATIVE_V5_MAX_CANDIDATES,
    )
    records = raw.get("records")
    owner_conflicts = raw.get("owner_conflicts")
    expected_target_aliases = sorted(
        {
            value
            for value in (
                str(getattr(target, "symbol", "") or "").strip(),
                str(getattr(target, "target_name", "") or "").strip(),
            )
            if value
        },
        key=lambda value: (_paper_v5_key(value), value),
    )[:8]
    if (
        target_aliases is None
        or queried_ids is None
        or not isinstance(records, list)
        or not isinstance(owner_conflicts, list)
        or raw.get("schema_version") != "paper_native_v5_alias_inventory_v1"
        or raw.get("policy") != _PAPER_NATIVE_V5_REQUIRED_PRIORITY
        or raw.get("source_policy")
        != "chembl_pref_name_plus_explicit_trusted_syn_type_v1"
        or raw.get("endpoint") != "IC50"
        or raw.get("fixed_pool_candidate_ids") != fixed_ids
        or raw.get("direct_support_candidate_ids") != direct_ids
        or queried_ids != direct_ids
        or target_aliases != expected_target_aliases
        or raw.get("per_candidate_alias_limit")
        != _PAPER_NATIVE_V5_MAX_ALIASES
        or raw.get("trusted_synonym_types")
        != ["BAN", "DCF", "INN", "JAN", "RESEARCH_CODE", "TRADE_NAME", "USAN"]
        or raw.get("complete") is not True
        or raw.get("normalization_applied_to_source_aliases") is not False
        or raw.get("pubmed_or_result_used_for_alias_selection") is not False
        or raw.get("fixture") is not False
        or len(records) != len(direct_ids)
    ):
        return None
    target_keys = {_paper_v5_key(item) for item in target_aliases}
    aliases_by_id: dict[str, list[str]] = {}
    global_owners: dict[str, str] = {}
    for candidate_id, record in zip(direct_ids, records, strict=True):
        if (
            not isinstance(record, dict)
            or not _paper_v5_self_hash(record, key="record_sha256")
            or record.get("molecule_chembl_id") != candidate_id
            or record.get("fixed_pool_rank") != fixed_ids.index(candidate_id) + 1
            or record.get("malformed_source_codes") != []
            or record.get("owner_conflict_rejections") != []
            or record.get("source_attestation_complete") is not True
            or record.get("query_eligible") is not True
            or record.get("synonym_source_present") is not True
            or isinstance(record.get("synonym_source_row_count"), bool)
            or not isinstance(record.get("synonym_source_row_count"), int)
            or record.get("synonym_source_row_count") < 0
        ):
            return None
        aliases = _paper_v5_strings(
            record.get("accepted_aliases"),
            minimum=1,
            maximum=_PAPER_NATIVE_V5_MAX_ALIASES,
        )
        accepted = record.get("accepted_before_owner_check")
        rejected = record.get("rejected_aliases")
        synonym_count = record.get("synonym_source_row_count")
        synonym_source_inventory = record.get(
            "molecule_synonyms_source_inventory"
        )
        if (
            aliases is None
            or not isinstance(accepted, list)
            or not isinstance(rejected, list)
            or not isinstance(synonym_count, int)
            or isinstance(synonym_count, bool)
            or synonym_count > 2 * _MAX_EXACT_CANDIDATE_ALIASES
            or record.get("accepted_alias_count") != len(aliases)
            or record.get("preferred_search_alias") != aliases[0]
            or record.get("trusted_synonym_types")
            != raw["trusted_synonym_types"]
            or [
                item.get("alias")
                for item in accepted
                if isinstance(item, dict)
            ]
            != aliases
        ):
            return None
        source_inventory = _paper_v5_synonym_source_inventory(
            synonym_source_inventory,
            expected_count=synonym_count,
        )
        if (
            source_inventory is None
            or record.get("molecule_synonyms_source_row_count")
            != synonym_count
            or not isinstance(synonym_source_inventory, dict)
            or record.get("molecule_synonyms_source_sha256")
            != synonym_source_inventory.get("inventory_sha256")
            or record.get("molecule_synonyms_source_reason_codes") != []
            or record.get("molecule_synonyms_source_complete") is not True
        ):
            return None
        source_rows = [*accepted, *rejected]
        if (
            any(not isinstance(item, dict) for item in source_rows)
            or sum(item.get("source") == "pref_name" for item in source_rows) != 1
            or any(
                item.get("syn_type") is not None
                or item.get("source_index") is not None
                for item in source_rows
                if item.get("source") == "pref_name"
            )
        ):
            return None
        synonym_indices = [
            item.get("source_index")
            for item in source_rows
            if item.get("source") == "molecule_synonyms"
        ]
        if (
            any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in synonym_indices
            )
            or sorted(synonym_indices) != list(range(synonym_count))
        ):
            return None
        source_ledger: dict[int, tuple[str, str]] = {}
        for source_row in source_rows:
            if source_row.get("source") != "molecule_synonyms":
                continue
            source_index = source_row.get("source_index")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index in source_ledger
                or not isinstance(source_row.get("syn_type"), str)
                or not isinstance(source_row.get("alias"), str)
            ):
                return None
            source_ledger[source_index] = (
                source_row["syn_type"],
                source_row["alias"],
            )
        if source_ledger != source_inventory:
            return None
        for alias, source_row in zip(aliases, accepted, strict=True):
            source = source_row.get("source") if isinstance(source_row, dict) else None
            syn_type = source_row.get("syn_type") if isinstance(source_row, dict) else None
            target_containing = _paper_v5_has_target_token_equivalent(
                alias,
                target_aliases,
            )
            if (
                not isinstance(source_row, dict)
                or source_row.get("alias_key") != _paper_v5_key(alias)
                or _paper_v5_key(alias) in target_keys
                or _paper_v5_key(alias) in global_owners
                or target_containing
                or source not in {"pref_name", "molecule_synonyms"}
                or alias.isdigit()
                or re.search(
                    r"(?<!\w)(?:\d+(?:\.\d+)?|\.\d+)\s*"
                    r"(?:pM|nM|uM|µM|μM|mM|M)(?!\w)",
                    alias,
                )
                is not None
                or re.fullmatch(r"PMID(?:[\s:_-]*\d+)?", alias, re.IGNORECASE)
                or (
                    source == "pref_name"
                    and (
                        syn_type is not None
                        or source_row.get("source_index") is not None
                    )
                )
                or (
                    source == "molecule_synonyms"
                    and (
                        syn_type not in set(raw["trusted_synonym_types"])
                        or isinstance(source_row.get("source_index"), bool)
                        or not isinstance(source_row.get("source_index"), int)
                        or source_row.get("source_index") < 0
                    )
                )
                or re.fullmatch(
                    r"(?:CHEMBL|PUBCHEM(?:\s+CID)?|CID|DRUGBANK|DB|UNII|CAS)"
                    r"(?:[\s:_-]*[A-Z0-9][A-Z0-9._-]*)",
                    alias,
                    flags=re.IGNORECASE,
                )
                or re.fullmatch(r"\d{2,7}-\d{2}-\d", alias)
            ):
                return None
            global_owners[_paper_v5_key(alias)] = candidate_id
        aliases_by_id[candidate_id] = aliases
    if owner_conflicts != []:
        return None
    return {
        "target_aliases": target_aliases,
        "aliases_by_id": aliases_by_id,
        "preferred_by_id": {
            item: aliases_by_id[item][0] for item in direct_ids
        },
        "inventory_sha256": raw["inventory_sha256"],
        "canonical_sha256": _paper_v5_hash(raw),
    }


def _paper_v5_support_snapshot(
    raw: object,
    *,
    candidate: object,
    evidence: object,
    activity: object,
    fixed_ids: list[str],
) -> dict[str, object] | None:
    if not isinstance(raw, dict) or not _paper_v5_self_hash(
        raw, key="support_attestation_sha256"
    ):
        return None
    candidate_metadata = getattr(candidate, "metadata", None)
    evidence_metadata = getattr(evidence, "metadata", None)
    target = getattr(candidate, "target", None)
    if (
        not isinstance(candidate_metadata, dict)
        or not isinstance(evidence_metadata, dict)
        or target is None
    ):
        return None
    molecule_id = _paired_v4_candidate_chembl_id(candidate)
    raw_tuple = raw.get("raw_measurement_tuple")
    typed_tuple = raw.get("typed_measurement_tuple")
    assay_context = raw.get("assay_context")
    activity_query = raw.get("chembl_activity_query")
    activity_params = raw.get("chembl_activity_query_parameters")
    assay_record = (
        assay_context.get("record") if isinstance(assay_context, dict) else None
    )
    document_id = _paper_v5_exact_document_id(raw.get("document_chembl_id"))
    value = _paper_v5_decimal(getattr(activity, "value", None))
    raw_value = (
        _paper_v5_decimal(raw_tuple.get("standard_value"))
        if isinstance(raw_tuple, dict)
        else None
    )
    raw_pchembl = (
        _paper_v5_decimal(raw_tuple.get("pchembl_value"))
        if isinstance(raw_tuple, dict)
        else None
    )
    query_threshold = (
        _paper_v5_query_threshold(activity_params.get("pchembl_value__gte"))
        if isinstance(activity_params, dict)
        else None
    )
    activity_parameter_keys = {
        "target_chembl_id",
        "parent_molecule_chembl_id",
        "target_organism",
        "assay_type",
        "standard_type",
        "pchembl_value__gte",
        "order_by",
        "limit",
    }
    if isinstance(activity_params, dict) and "offset" in activity_params:
        activity_parameter_keys.add("offset")
    if (
        molecule_id is None
        or molecule_id not in fixed_ids
        or raw.get("schema_version")
        != "paper_native_v5_pre_pubmed_chembl_support_v1"
        or raw.get("verification_status") != "verified"
        or raw.get("reason_code") is not None
        or raw.get("molecule_chembl_id") != molecule_id
        or raw.get("fixed_pool_rank") != fixed_ids.index(molecule_id) + 1
        or isinstance(raw.get("retained_index"), bool)
        or not isinstance(raw.get("retained_index"), int)
        or raw.get("retained_rank") != raw.get("retained_index") + 1
        or not isinstance(raw_tuple, dict)
        or not isinstance(typed_tuple, dict)
        or raw.get("raw_measurement_tuple_sha256")
        != _paper_v5_hash(raw_tuple)
        or raw.get("typed_measurement_tuple_sha256")
        != _paper_v5_hash(typed_tuple)
        or not isinstance(assay_context, dict)
        or raw.get("assay_context_sha256") != _paper_v5_hash(assay_context)
        or not isinstance(activity_query, dict)
        or raw.get("chembl_activity_query_sha256")
        != _paper_v5_hash(activity_query)
        or not isinstance(activity_params, dict)
        or raw.get("chembl_activity_query_parameters_sha256")
        != _paper_v5_gateway_parameter_hash(activity_params)
        or activity_query.get("parameter_hash")
        != raw.get("chembl_activity_query_parameters_sha256")
        or query_threshold != _PAPER_NATIVE_V5_APPROVED_PCHEMBL_THRESHOLD
        or raw_pchembl is None
        or raw_pchembl < query_threshold
        or raw.get("selected_before_pubmed_search") is not True
        or raw.get("selection_policy")
        != "first_strict_direct_support_in_retained_order_v1"
        or raw.get("pubmed_value_used_for_selection") is not False
        or raw.get("cross_source_numeric_equality_required") is not False
        or raw.get("unit_conversion_applied") is not False
        or raw.get("missing_values_imputed") is not False
        or raw.get("fixture") is not False
        or raw.get("activity_id") != getattr(evidence, "external_id", None)
        or raw.get("source_evidence_id") != getattr(evidence, "evidence_id", None)
        or getattr(activity, "source_evidence_id", None)
        != getattr(evidence, "evidence_id", None)
        or raw.get("target_uniprot_id") != getattr(target, "uniprot_id", None)
        or raw.get("target_organism") != "Homo sapiens"
        or raw.get("target_type") != "SINGLE PROTEIN"
        or evidence_metadata.get("molecule_chembl_id") != molecule_id
        or evidence_metadata.get("target_chembl_id")
        != raw.get("target_chembl_id")
        or evidence_metadata.get("target_uniprot_id")
        != raw.get("target_uniprot_id")
        or document_id is None
        or raw_tuple.get("document_chembl_id") != document_id
        or evidence_metadata.get("document_chembl_id") != document_id
        or evidence_metadata.get("activity_query") != activity_query
        or raw_tuple.get("activity_id") != raw.get("activity_id")
        or raw_tuple.get("molecule_chembl_id") != molecule_id
        or raw_tuple.get("parent_molecule_chembl_id") != molecule_id
        or raw_tuple.get("target_chembl_id") != raw.get("target_chembl_id")
        or raw_tuple.get("assay_chembl_id") != raw.get("assay_chembl_id")
        or raw_tuple.get("assay_type") != "B"
        or not MA3AssuranceAgent._assay_type_supports_direct_engagement(
            str(raw_tuple.get("bao_label") or "")
        )
        or raw_tuple.get("standard_type") != "IC50"
        or raw_tuple.get("standard_relation") != "="
        or raw_tuple.get("standard_units") not in _KNOWN_ACTIVITY_UNITS
        or raw_tuple.get("data_validity_comment")
        not in (None, "", "Manually validated")
        or value is None
        or raw_value is None
        or raw_value != value
        or typed_tuple.get("endpoint") != getattr(activity, "endpoint", None)
        or typed_tuple.get("relation") != getattr(activity, "relation", None)
        or _paper_v5_decimal(typed_tuple.get("value")) != value
        or typed_tuple.get("unit") != getattr(activity, "unit", None)
        or typed_tuple.get("assay_type") != getattr(activity, "assay_type", None)
        or typed_tuple.get("source_evidence_id")
        != getattr(activity, "source_evidence_id", None)
        or typed_tuple.get("endpoint") != "IC50"
        or typed_tuple.get("relation") != "="
        or typed_tuple.get("unit") != raw_tuple.get("standard_units")
        or not MA3AssuranceAgent._assay_type_supports_direct_engagement(
            str(typed_tuple.get("assay_type") or "")
        )
        or assay_context.get("schema_version") != "chembl_assay_context_v1"
        or assay_context.get("retrieval_status") != "completed"
        or assay_context.get("binding_status") != "exact"
        or assay_context.get("assay_chembl_id") != raw.get("assay_chembl_id")
        or assay_context.get("missing_values_imputed") is not False
        or assay_context.get("inference_used") is not False
        or not isinstance(assay_record, dict)
        or assay_record.get("assay_chembl_id") != raw.get("assay_chembl_id")
        or assay_record.get("target_chembl_id") != raw.get("target_chembl_id")
        or assay_record.get("assay_type") != "B"
        or assay_record.get("assay_organism") != "Homo sapiens"
        or assay_record.get("bao_format") != "BAO_0000357"
        or assay_record.get("bao_label") != raw_tuple.get("bao_label")
        or not MA3AssuranceAgent._assay_type_supports_direct_engagement(
            str(assay_record.get("bao_label") or "")
        )
        or activity_query.get("source") != "ChEMBL"
        or activity_query.get("endpoint")
        != "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        or activity_query.get("parameter_keys")
        != sorted(activity_parameter_keys)
        or set(activity_params) != activity_parameter_keys
        or activity_params.get("target_chembl_id")
        != raw.get("target_chembl_id")
        or activity_params.get("parent_molecule_chembl_id") != molecule_id
        or activity_params.get("target_organism") != "Homo sapiens"
        or activity_params.get("assay_type") != "B"
        or activity_params.get("standard_type") != "IC50"
        or activity_params.get("order_by") != "activity_id"
        or activity_params.get("limit") != 20
        or (
            "offset" in activity_params
            and (
                type(activity_params.get("offset")) is not int
                or int(activity_params["offset"]) < 20
                or int(activity_params["offset"]) % 20 != 0
            )
        )
        or activity_query.get("status_code") != 200
        or not _paper_v5_chembl_retry_contract(activity_query)
        or activity_query.get("exhausted") is not False
        or activity_query.get("operation") != "retrieve_repurposing"
        or not isinstance(activity_query.get("started_at"), str)
        or not isinstance(activity_query.get("finished_at"), str)
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
            str(activity_query.get("started_at") or ""),
        )
        is None
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
            str(activity_query.get("finished_at") or ""),
        )
        is None
        or str(activity_query.get("finished_at"))
        < str(activity_query.get("started_at"))
        or isinstance(activity_query.get("runtime_seconds"), bool)
        or not isinstance(activity_query.get("runtime_seconds"), (int, float))
        or float(activity_query.get("runtime_seconds")) < 0
        or not _SHA256.fullmatch(str(activity_query.get("parameter_hash") or ""))
        or not _SHA256.fullmatch(str(activity_query.get("response_sha256") or ""))
    ):
        return None
    return {
        "molecule_id": molecule_id,
        "support_sha256": raw["support_attestation_sha256"],
        "activity_id": raw["activity_id"],
        "retained_rank": raw["retained_rank"],
        "canonical_sha256": _paper_v5_hash(raw),
    }


def _paper_v5_strict_support_row(
    audit: Mapping[str, object],
    *,
    candidate_id: str,
    target_chembl_id: str,
    pchembl_threshold: Decimal,
    fallback_support: Mapping[str, object] | None,
) -> bool:
    """Recompute support eligibility without consulting status/reason flags."""

    raw_tuple = audit.get("raw_measurement_tuple")
    typed_tuple = audit.get("typed_measurement_tuple")
    assay_context = audit.get("assay_context")
    if (
        not isinstance(assay_context, dict)
        and isinstance(fallback_support, Mapping)
        and audit.get("assay_context_sha256")
        == fallback_support.get("assay_context_sha256")
    ):
        assay_context = fallback_support.get("assay_context")
    assay_record = (
        assay_context.get("record") if isinstance(assay_context, dict) else None
    )
    raw_value = (
        _paper_v5_decimal(raw_tuple.get("standard_value"))
        if isinstance(raw_tuple, dict)
        else None
    )
    typed_value = (
        _paper_v5_decimal(typed_tuple.get("value"))
        if isinstance(typed_tuple, dict)
        else None
    )
    pchembl = (
        _paper_v5_decimal(raw_tuple.get("pchembl_value"))
        if isinstance(raw_tuple, dict)
        else None
    )
    activity_id = audit.get("activity_id")
    assay_id = audit.get("assay_chembl_id")
    document_id = _paper_v5_exact_document_id(
        audit.get("document_chembl_id")
    )
    return bool(
        isinstance(raw_tuple, dict)
        and isinstance(typed_tuple, dict)
        and isinstance(assay_context, dict)
        and isinstance(assay_record, dict)
        and isinstance(activity_id, str)
        and activity_id
        and isinstance(assay_id, str)
        and assay_id
        and document_id is not None
        and audit.get("raw_measurement_tuple_sha256")
        == _paper_v5_hash(raw_tuple)
        and audit.get("typed_measurement_tuple_sha256")
        == _paper_v5_hash(typed_tuple)
        and audit.get("assay_context_sha256")
        == _paper_v5_hash(assay_context)
        and raw_tuple.get("activity_id") == activity_id
        and raw_tuple.get("molecule_chembl_id") == candidate_id
        and raw_tuple.get("parent_molecule_chembl_id") == candidate_id
        and raw_tuple.get("target_chembl_id") == target_chembl_id
        and raw_tuple.get("assay_chembl_id") == assay_id
        and raw_tuple.get("document_chembl_id") == document_id
        and raw_tuple.get("assay_type") == "B"
        and MA3AssuranceAgent._assay_type_supports_direct_engagement(
            str(raw_tuple.get("bao_label") or "")
        )
        and raw_tuple.get("standard_type") == "IC50"
        and raw_tuple.get("standard_relation") == "="
        and raw_tuple.get("standard_units") in _KNOWN_ACTIVITY_UNITS
        and "data_validity_comment" in raw_tuple
        and raw_tuple.get("data_validity_comment")
        in (None, "", "Manually validated")
        and raw_value is not None
        and pchembl is not None
        and pchembl >= pchembl_threshold
        and typed_value == raw_value
        and typed_tuple.get("endpoint") == "IC50"
        and typed_tuple.get("relation") == "="
        and typed_tuple.get("unit") == raw_tuple.get("standard_units")
        and MA3AssuranceAgent._assay_type_supports_direct_engagement(
            str(typed_tuple.get("assay_type") or "")
        )
        and isinstance(typed_tuple.get("source_evidence_id"), str)
        and bool(typed_tuple.get("source_evidence_id"))
        and assay_context.get("schema_version") == "chembl_assay_context_v1"
        and assay_context.get("retrieval_status") == "completed"
        and assay_context.get("binding_status") == "exact"
        and assay_context.get("reason_code") is None
        and assay_context.get("assay_chembl_id") == assay_id
        and assay_context.get("missing_values_imputed") is False
        and assay_context.get("inference_used") is False
        and assay_context.get("unit_conversion_applied") is False
        and assay_context.get("description_similarity_used") is False
        and assay_record.get("assay_chembl_id") == assay_id
        and assay_record.get("target_chembl_id") == target_chembl_id
        and assay_record.get("assay_type") == "B"
        and assay_record.get("assay_organism") == "Homo sapiens"
        and assay_record.get("bao_format") == "BAO_0000357"
        and assay_record.get("bao_label") == raw_tuple.get("bao_label")
        and MA3AssuranceAgent._assay_type_supports_direct_engagement(
            str(assay_record.get("bao_label") or "")
        )
    )


def _paper_v5_support_inventory(
    raw: object,
    *,
    fixed_ids: list[str],
    direct_ids: list[str],
    snapshots: list[object],
    target_chembl_id: str,
    target_uniprot_id: str,
    resolution_by_id: Mapping[str, object],
    mechanisms_by_id: Mapping[str, object],
    complete_activity_pagination: bool = False,
    activity_pages_by_id: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    if not isinstance(raw, dict) or not _paper_v5_self_hash(
        raw, key="inventory_sha256"
    ):
        return None
    records = raw.get("records")
    selected = raw.get("selected_support_snapshots")
    selection_order = raw.get("selection_order")
    identity_bindings = raw.get("fixed_pool_identity_bindings")
    if (
        not isinstance(records, list)
        or not isinstance(selected, list)
        or not isinstance(selection_order, list)
        or not isinstance(identity_bindings, list)
        or not target_chembl_id
        or not target_uniprot_id
        or raw.get("schema_version")
        != "paper_native_v5_direct_support_inventory_v1"
        or raw.get("policy") != _PAPER_NATIVE_V5_REQUIRED_PRIORITY
        or raw.get("fixed_pool_candidate_ids") != fixed_ids
        or raw.get("fixed_pool_candidate_ids_sha256")
        != _paper_v5_hash(fixed_ids)
        or raw.get("selected_candidate_ids") != direct_ids
        or raw.get("selected_candidate_count") != len(direct_ids)
        or selected != snapshots
        or raw.get("selected_support_snapshots_sha256")
        != _paper_v5_hash(selected)
        or raw.get("selection_order_sha256")
        != _paper_v5_hash(selection_order)
        or raw.get("fixed_pool_identity_bindings_sha256")
        != _paper_v5_hash(identity_bindings)
        or raw.get("selection_completed_before_pubmed_search") is not True
        or raw.get("pubmed_or_result_used_for_selection") is not False
        or raw.get("no_reordering") is not True
        or raw.get("no_backfill") is not True
        or raw.get("fixture") is not False
        or len(records) != len(fixed_ids)
        or len(identity_bindings) != len(fixed_ids)
        or len(selected) != len(direct_ids)
        or len(selection_order) != len(direct_ids)
    ):
        return None
    selected_by_id: dict[str, dict[str, object]] = {}
    for candidate_id, snapshot, order in zip(
        direct_ids, selected, selection_order, strict=True
    ):
        snapshot_document_id = (
            _paper_v5_exact_document_id(snapshot.get("document_chembl_id"))
            if isinstance(snapshot, dict)
            else None
        )
        snapshot_raw_tuple = (
            snapshot.get("raw_measurement_tuple")
            if isinstance(snapshot, dict)
            else None
        )
        if (
            not isinstance(snapshot, dict)
            or not _paper_v5_self_hash(
                snapshot, key="support_attestation_sha256"
            )
            or snapshot.get("molecule_chembl_id") != candidate_id
            or snapshot.get("fixed_pool_rank") != fixed_ids.index(candidate_id) + 1
            or snapshot_document_id is None
            or not isinstance(snapshot_raw_tuple, dict)
            or snapshot_raw_tuple.get("document_chembl_id")
            != snapshot_document_id
            or not isinstance(order, dict)
            or order
            != {
                "molecule_chembl_id": candidate_id,
                "fixed_pool_rank": snapshot.get("fixed_pool_rank"),
                "retained_rank": snapshot.get("retained_rank"),
                "activity_id": snapshot.get("activity_id"),
            }
        ):
            return None
        selected_by_id[candidate_id] = snapshot
    recomputed_direct_ids: list[str] = []
    for candidate_id, record, identity_binding in zip(
        fixed_ids, records, identity_bindings, strict=True
    ):
        fixed_rank = fixed_ids.index(candidate_id) + 1
        if (
            not isinstance(record, dict)
            or not _paper_v5_self_hash(record, key="record_sha256")
            or record.get("molecule_chembl_id") != candidate_id
            or record.get("fixed_pool_rank") != fixed_rank
            or not isinstance(record.get("row_audits"), list)
            or record.get("retained_row_count")
            != len(record.get("row_audits"))
            or not 0 <= len(record.get("row_audits")) <= 3
            or not isinstance(identity_binding, dict)
            or record.get("candidate_identity_binding") != identity_binding
            or not _paper_v5_self_hash(
                identity_binding, key="binding_sha256"
            )
            or record.get("candidate_identity_binding_sha256")
            != identity_binding.get("binding_sha256")
            or identity_binding.get("schema_version")
            != "paper_native_v5_fixed_pool_identity_binding_v1"
            or identity_binding.get("molecule_chembl_id") != candidate_id
            or identity_binding.get("fixed_pool_rank") != fixed_rank
            or identity_binding.get("target_chembl_id") != target_chembl_id
            or identity_binding.get("target_uniprot_id")
            != target_uniprot_id
            or identity_binding.get("target_type") != "SINGLE PROTEIN"
            or identity_binding.get("target_organism") != "Homo sapiens"
            or identity_binding.get("identity_resolution")
            != resolution_by_id.get(candidate_id)
            or identity_binding.get("mechanism_provenance")
            != mechanisms_by_id.get(candidate_id)
            or identity_binding.get("binding_status") != "exact"
            or identity_binding.get("reason_codes") != []
        ):
            return None
        molecule_metadata = identity_binding.get("molecule_metadata")
        identity_resolution = resolution_by_id.get(candidate_id)
        if (
            not isinstance(molecule_metadata, dict)
            or not isinstance(identity_resolution, dict)
            or molecule_metadata
            != {
                "molecule_chembl_id": candidate_id,
                "max_phase": identity_resolution.get("max_phase"),
                "molecule_type": "Small molecule",
            }
        ):
            return None
        row_audits = record["row_audits"]
        snapshot = selected_by_id.get(candidate_id)
        valid_indices: list[int] = []
        activity_ids: list[str] = []
        retained_order: list[tuple[Decimal, str]] = []
        candidate_activity_query: dict[str, object] | None = None
        candidate_activity_params: dict[str, object] | None = None
        for retained_index, audit in enumerate(row_audits):
            if (
                not isinstance(audit, dict)
                or not _paper_v5_self_hash(audit, key="row_audit_sha256")
                or audit.get("schema_version")
                != "paper_native_v5_direct_support_row_audit_v1"
                or audit.get("molecule_chembl_id") != candidate_id
                or audit.get("fixed_pool_rank")
                != fixed_rank
                or audit.get("retained_index") != retained_index
                or audit.get("retained_rank") != retained_index + 1
                or audit.get("unit_conversion_applied") is not False
                or audit.get("missing_values_imputed") is not False
                or audit.get("fixture") is not False
                or not isinstance(audit.get("reason_codes"), list)
                or not isinstance(audit.get("raw_measurement_tuple"), dict)
                or not isinstance(audit.get("typed_measurement_tuple"), dict)
                or audit.get("raw_measurement_tuple_sha256")
                != _paper_v5_hash(audit["raw_measurement_tuple"])
                or audit.get("typed_measurement_tuple_sha256")
                != _paper_v5_hash(audit["typed_measurement_tuple"])
            ):
                return None
            raw_tuple = audit["raw_measurement_tuple"]
            activity_id = audit.get("activity_id")
            pchembl = _paper_v5_decimal(raw_tuple.get("pchembl_value"))
            activity_query = audit.get("chembl_activity_query")
            activity_params = audit.get("chembl_activity_query_parameters")
            query_threshold = (
                _paper_v5_query_threshold(
                    activity_params.get("pchembl_value__gte")
                )
                if isinstance(activity_params, dict)
                else None
            )
            query_binding = audit.get(
                "chembl_activity_query_candidate_binding"
            )
            expected_parameter_keys = {
                "target_chembl_id",
                "parent_molecule_chembl_id",
                "target_organism",
                "assay_type",
                "standard_type",
                "pchembl_value__gte",
                "order_by",
                "limit",
            }
            if isinstance(activity_params, dict) and "offset" in activity_params:
                expected_parameter_keys.add("offset")
            if (
                not isinstance(activity_id, str)
                or not activity_id
                or activity_id in activity_ids
                or pchembl is None
                or not isinstance(activity_query, dict)
                or audit.get("chembl_activity_query_sha256")
                != _paper_v5_hash(activity_query)
                or activity_query.get("source") != "ChEMBL"
                or activity_query.get("endpoint")
                != "https://www.ebi.ac.uk/chembl/api/data/activity.json"
                or activity_query.get("status_code") != 200
                or not _paper_v5_chembl_retry_contract(activity_query)
                or activity_query.get("exhausted") is not False
                or activity_query.get("cache_hit") is not False
                or activity_query.get("operation") != "retrieve_repurposing"
                or not isinstance(activity_query.get("started_at"), str)
                or not isinstance(activity_query.get("finished_at"), str)
                or re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
                    str(activity_query.get("started_at") or ""),
                )
                is None
                or re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
                    str(activity_query.get("finished_at") or ""),
                )
                is None
                or str(activity_query.get("finished_at"))
                < str(activity_query.get("started_at"))
                or not _SHA256.fullmatch(
                    str(activity_query.get("response_sha256") or "")
                )
                or not isinstance(activity_params, dict)
                or set(activity_params)
                != expected_parameter_keys
                or audit.get("chembl_activity_query_parameters_sha256")
                != _paper_v5_gateway_parameter_hash(activity_params)
                or activity_query.get("parameter_hash")
                != audit.get("chembl_activity_query_parameters_sha256")
                or activity_query.get("parameter_keys")
                != sorted(activity_params)
                or activity_params.get("target_chembl_id")
                != target_chembl_id
                or activity_params.get("parent_molecule_chembl_id")
                != candidate_id
                or activity_params.get("target_organism")
                != "Homo sapiens"
                or activity_params.get("assay_type") != "B"
                or activity_params.get("standard_type") != "IC50"
                or activity_params.get("order_by") != "activity_id"
                or activity_params.get("limit") != 20
                or (
                    "offset" in activity_params
                    and (
                        type(activity_params.get("offset")) is not int
                        or int(activity_params["offset"]) < 20
                        or int(activity_params["offset"]) % 20 != 0
                    )
                )
                or query_threshold
                != _PAPER_NATIVE_V5_APPROVED_PCHEMBL_THRESHOLD
                or pchembl < query_threshold
                or not isinstance(query_binding, dict)
                or audit.get(
                    "chembl_activity_query_candidate_binding_sha256"
                )
                != _paper_v5_hash(query_binding)
                or query_binding
                != {
                    "molecule_chembl_id": candidate_id,
                    "target_chembl_id": target_chembl_id,
                    "endpoint": (
                        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
                    ),
                    "parameters": activity_params,
                    "parameter_sha256": audit.get(
                        "chembl_activity_query_parameters_sha256"
                    ),
                }
            ):
                return None
            if complete_activity_pagination:
                candidate_pages = (
                    activity_pages_by_id.get(candidate_id)
                    if isinstance(activity_pages_by_id, Mapping)
                    else None
                )
                matching_pages = [
                    page
                    for page in candidate_pages
                    if isinstance(page, dict)
                    and page.get("query") == activity_query
                    and page.get("query_parameters") == activity_params
                    and activity_id in page.get("activity_ids", [])
                ] if isinstance(candidate_pages, list) else []
                if len(matching_pages) != 1:
                    return None
            else:
                if candidate_activity_query is None:
                    candidate_activity_query = activity_query
                    candidate_activity_params = activity_params
                elif (
                    activity_query != candidate_activity_query
                    or activity_params != candidate_activity_params
                ):
                    return None
            activity_ids.append(activity_id)
            retained_order.append((-pchembl, activity_id))
            is_strict = _paper_v5_strict_support_row(
                audit,
                candidate_id=candidate_id,
                target_chembl_id=target_chembl_id,
                pchembl_threshold=query_threshold,
                fallback_support=snapshot,
            )
            if is_strict:
                valid_indices.append(retained_index)
                if (
                    audit.get("status") != "supported"
                    or audit.get("reason_codes") != []
                ):
                    return None
            elif (
                audit.get("status") != "not_supported"
                or not audit.get("reason_codes")
            ):
                return None
        if retained_order != sorted(retained_order):
            return None
        observed_codes = sorted(
            {
                code
                for audit in row_audits
                for code in audit.get("reason_codes", [])
                if isinstance(code, str)
            }
        )
        if record.get("observed_row_reason_codes") != observed_codes:
            return None
        first_valid_index = valid_indices[0] if valid_indices else None
        if first_valid_index is None:
            if (
                snapshot is not None
                or record.get("selection_status") != "not_selected"
                or record.get("selected_support") is not None
                or record.get("reason_code")
                != "NO_STRICT_DIRECT_CHEMBL_SUPPORT_IN_RETAINED_ROWS"
            ):
                return None
            continue
        recomputed_direct_ids.append(candidate_id)
        if snapshot is None:
            return None
        retained_index = snapshot.get("retained_index")
        if (
            isinstance(retained_index, bool)
            or not isinstance(retained_index, int)
            or retained_index != first_valid_index
            or record.get("selection_status") != "selected"
            or record.get("reason_code") is not None
            or record.get("selected_support") != snapshot
            or row_audits[retained_index].get("activity_id")
            != snapshot.get("activity_id")
            or row_audits[retained_index].get("assay_chembl_id")
            != snapshot.get("assay_chembl_id")
            or row_audits[retained_index].get("document_chembl_id")
            != snapshot.get("document_chembl_id")
            or row_audits[retained_index].get("raw_measurement_tuple")
            != snapshot.get("raw_measurement_tuple")
            or row_audits[retained_index].get("typed_measurement_tuple")
            != snapshot.get("typed_measurement_tuple")
            or row_audits[retained_index].get("assay_context_sha256")
            != snapshot.get("assay_context_sha256")
            or row_audits[retained_index].get("assay_context")
            != snapshot.get("assay_context")
            or row_audits[retained_index].get("chembl_activity_query")
            != snapshot.get("chembl_activity_query")
            or row_audits[retained_index].get(
                "chembl_activity_query_parameters"
            )
            != snapshot.get("chembl_activity_query_parameters")
            or row_audits[retained_index].get(
                "chembl_activity_query_candidate_binding"
            )
            != snapshot.get("chembl_activity_query_candidate_binding")
        ):
            return None
    if recomputed_direct_ids != direct_ids:
        return None
    return {
        "selected_by_id": selected_by_id,
        "selection_order_sha256": raw["selection_order_sha256"],
        "inventory_sha256": raw["inventory_sha256"],
        "canonical_sha256": _paper_v5_hash(raw),
    }


def _paper_v5_expected_query(inventory: Mapping[str, object]) -> str | None:
    target_aliases = inventory.get("target_aliases")
    preferred_by_id = inventory.get("preferred_by_id")
    aliases_by_id = inventory.get("aliases_by_id")
    if (
        not isinstance(target_aliases, list)
        or not isinstance(preferred_by_id, dict)
        or not isinstance(aliases_by_id, dict)
    ):
        return None
    target_values = sorted(
        target_aliases, key=lambda value: (_paper_v5_key(value), value)
    )
    candidate_ids = sorted(
        preferred_by_id,
        key=lambda value: (_paper_v5_key(value), value),
    )
    preferred = [preferred_by_id[item] for item in candidate_ids]
    if any(
        not isinstance(item, str)
        or '"' in item
        or any(ord(character) < 32 for character in item)
        for item in [*target_values, *preferred]
    ):
        return None
    pubmed_term = lambda value: f'"{value}"[Title/Abstract]'
    return (
        f"({' OR '.join(pubmed_term(item) for item in target_values)}) "
        f"AND {pubmed_term('IC50')} AND "
        f"({' OR '.join(pubmed_term(item) for item in preferred)})"
    )


def _paper_v5_call(
    raw: object,
    *,
    stage: str,
    endpoint: str,
    params: Mapping[str, object],
) -> bool:
    if not isinstance(raw, dict):
        return False
    parameter_keys = sorted(f"params.{key}" for key in params)
    payload = {
        "endpoint": endpoint,
        "error_type": None,
        "method": "GET",
        "parameter_keys": parameter_keys,
        "parameter_sha256": _paper_v5_hash({"params": dict(params)}),
        "response_sha256": raw.get("response_sha256"),
        "schema": _PAPER_NATIVE_V5_SEARCH_SCHEMA,
        "stage": stage,
        "status": "completed",
        "status_code": 200,
    }
    return bool(
        raw.get("stage") == stage
        and raw.get("method") == "GET"
        and raw.get("endpoint") == endpoint
        and raw.get("parameter_keys") == parameter_keys
        and raw.get("parameter_sha256") == payload["parameter_sha256"]
        and raw.get("status") == "completed"
        and raw.get("status_code") == 200
        and raw.get("error_type") is None
        and _SHA256.fullmatch(str(raw.get("response_sha256") or ""))
        and raw.get("call_sha256") == _paper_v5_hash(payload)
    )


def _paper_v5_publication_types(value: object) -> list[str] | None:
    values = _paper_v5_strings(value, minimum=1)
    if values is None:
        return None
    normalized = sorted(
        {
            unicodedata.normalize("NFKC", item).strip().casefold()
            for item in values
            if item.strip()
        }
    )
    if (
        values != normalized
        or any(_PAPER_NATIVE_V5_EXCLUDED_TYPE.search(item) for item in normalized)
        or not set(normalized).intersection(_PAPER_NATIVE_V5_PRIMARY_TYPES)
    ):
        return None
    return normalized


def _paper_v5_span(
    raw: object,
    *,
    sentence: str,
    allow_none: bool = False,
) -> tuple[int, int, str] | None:
    if raw is None and allow_none:
        return (-1, -1, "")
    if not isinstance(raw, dict) or set(raw) != {"start", "end", "text"}:
        return None
    start = raw.get("start")
    end = raw.get("end")
    text = raw.get("text")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(text, str)
        or not 0 <= start <= end <= len(sentence)
        or sentence[start:end] != text
    ):
        return None
    return (start, end, text)


def _paper_v5_occurrences(sentence: str, alias: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in re.finditer(
            rf"(?<![\w-]){re.escape(alias)}(?![\w-])",
            sentence,
            flags=re.IGNORECASE,
        )
    ]


def _paper_v5_closed_observed_head(
    sentence: str,
    *,
    candidate_end: int,
    target_start: int,
    target_end: int,
    open_parenthesis: int,
) -> bool:
    """Independently consume the complete observed-inhibition head clause."""

    prefix = sentence[candidate_end:target_start]
    suffix = sentence[target_end:open_parenthesis]
    potency_observation = (
        _PAPER_NATIVE_V5_OBSERVED_POTENCY_PREFIX.fullmatch(prefix) is not None
        and _PAPER_NATIVE_V5_INHIBITOR_SUFFIX.fullmatch(suffix) is not None
    )
    direct_observation = (
        (
            _PAPER_NATIVE_V5_OBSERVED_TO_INHIBIT_PREFIX.fullmatch(prefix)
            is not None
            or _PAPER_NATIVE_V5_OBSERVED_ACTIVE_INHIBITION_PREFIX.fullmatch(
                prefix
            )
            is not None
        )
        and _PAPER_NATIVE_V5_EMPTY_TARGET_SUFFIX.fullmatch(suffix) is not None
    )
    return potency_observation or direct_observation


def _paper_v5_pair_evidence_hash(
    pair: Mapping[str, object],
    *,
    sentence_sha256: str,
) -> str | None:
    candidate_span = pair.get("candidate_span")
    target_span = pair.get("target_span")
    endpoint_span = pair.get("endpoint_span")
    wrapper_span = pair.get("endpoint_wrapper_span")
    condition = pair.get("assay_condition")
    measurement_span = pair.get("measurement_span")
    if not all(
        isinstance(item, dict)
        for item in (
            candidate_span,
            target_span,
            endpoint_span,
            condition,
            measurement_span,
        )
    ):
        return None
    assert isinstance(candidate_span, dict)
    assert isinstance(target_span, dict)
    assert isinstance(endpoint_span, dict)
    assert isinstance(condition, dict)
    assert isinstance(measurement_span, dict)
    condition_span = condition.get("span")
    if not isinstance(condition_span, dict):
        return None

    def span_list(value: Mapping[str, object]) -> list[object]:
        return [value.get("start"), value.get("end"), value.get("text")]

    payload = {
        "candidate_id": pair.get("candidate_id"),
        "candidate_span": span_list(candidate_span),
        "condition_span": span_list(condition_span),
        "condition_subject": condition.get("subject"),
        "condition_unit_raw": condition.get("unit_raw"),
        "condition_value_raw": condition.get("value_raw"),
        "endpoint_span": span_list(endpoint_span),
        "endpoint_wrapper_raw": pair.get("endpoint_wrapper_raw"),
        "endpoint_wrapper_span": (
            span_list(wrapper_span)
            if isinstance(wrapper_span, dict)
            else None
        ),
        "grammar": _PAPER_NATIVE_V5_GRAMMAR,
        "measurement_span": span_list(measurement_span),
        "pair_index": pair.get("pair_index"),
        "scope_basis": (
            "explicit_common_head"
            if pair.get("pair_index") == 0
            else "inherited_from_common_head"
        ),
        "publication_types": pair.get("publication_types"),
        "relation_basis": pair.get("relation_basis"),
        "relation_normalized": pair.get("relation_normalized"),
        "relation_raw": pair.get("relation_raw"),
        "literal_equality": pair.get("literal_equality"),
        "schema": _PAPER_NATIVE_V5_PARSER_SCHEMA,
        "section": pair.get("section"),
        "sentence_sha256": sentence_sha256,
        "target_span": span_list(target_span),
        "unit_raw": pair.get("unit_raw"),
        "value_decimal": pair.get("value_decimal"),
        "value_raw": pair.get("value_raw"),
    }
    return _paper_v5_hash(payload)


def _paper_v5_verify_pairs(
    pairs: object,
    *,
    sentence: str,
    section: str,
    publication_types: list[str],
    inventory: Mapping[str, object],
    candidate_ids: list[str],
) -> list[dict[str, object]] | None:
    if (
        not isinstance(pairs, list)
        or len(pairs) < 3
        or len(pairs) != len(candidate_ids)
        or not sentence
        or len(sentence) > _PRIMARY_EXCERPT_MAX_CHARS
        or sentence != sentence.strip()
        or "\r" in sentence
        or "\n" in sentence
        or any(item in sentence for item in ("?", "!", ";"))
        or any(item in sentence for item in (":", "\u2014", "\u2013"))
        or _PAPER_NATIVE_V5_UNSAFE.search(sentence)
        or re.search(
            r"(?:\d+(?:\.\d+)?|\.\d+)\s*(?:-|\u2013|\u2014|to)\s*"
            r"(?:\d+(?:\.\d+)?|\.\d+)|\b(?:range|ranged)\b",
            sentence,
            flags=re.IGNORECASE,
        )
        or "/" in sentence
    ):
        return None
    core = sentence[:-1] if sentence.endswith(".") else sentence
    if re.search(r"(?<!\d)\.(?!\d)", core):
        return None
    stack: list[str] = []
    expected_closer = {"(": ")", "[": "]"}
    for character in sentence:
        if character in expected_closer:
            stack.append(character)
            if len(stack) > 4:
                return None
        elif character in ")]":
            if not stack or expected_closer[stack.pop()] != character:
                return None
    if stack:
        return None

    aliases_by_id = inventory.get("aliases_by_id")
    target_aliases = inventory.get("target_aliases")
    if not isinstance(aliases_by_id, dict) or not isinstance(target_aliases, list):
        return None
    all_candidate_occurrences: list[tuple[int, int, str]] = []
    for owner, aliases in aliases_by_id.items():
        if not isinstance(owner, str) or not isinstance(aliases, list):
            return None
        for alias in aliases:
            if not isinstance(alias, str):
                return None
            all_candidate_occurrences.extend(
                (start, end, owner)
                for start, end in _paper_v5_occurrences(sentence, alias)
            )
    all_candidate_occurrences.sort()
    if (
        len(all_candidate_occurrences) != len(candidate_ids)
        or [item[2] for item in all_candidate_occurrences] != candidate_ids
        or len({item[2] for item in all_candidate_occurrences})
        != len(candidate_ids)
        or any(
            left[1] > right[0]
            for left, right in zip(
                all_candidate_occurrences,
                all_candidate_occurrences[1:],
            )
        )
    ):
        return None
    target_occurrences = [
        (start, end)
        for alias in target_aliases
        for start, end in _paper_v5_occurrences(sentence, alias)
    ]
    endpoint_occurrences = list(_PAPER_NATIVE_V5_ENDPOINT.finditer(sentence))
    connectors = list(
        re.finditer(r",\s+followed\s+by\s+", sentence, flags=re.IGNORECASE)
    )
    if (
        len(target_occurrences) != 1
        or len(endpoint_occurrences) != 1
        or len(connectors) != 1
        or len(re.findall(r"\bfollowed\s+by\b", sentence, re.IGNORECASE)) != 1
    ):
        return None
    sentence_sha256 = sha256(sentence.encode("utf-8")).hexdigest()
    target_occurrence = target_occurrences[0]
    endpoint_occurrence = (
        endpoint_occurrences[0].start(),
        endpoint_occurrences[0].end(),
    )
    common_condition: dict[str, object] | None = None
    closing_parentheses: list[int] = []
    verified: list[dict[str, object]] = []
    numeric_spans: set[tuple[int, int]] = set()
    ignored_numeric_ranges = [
        *(item[:2] for item in all_candidate_occurrences),
        target_occurrence,
        endpoint_occurrence,
    ]
    for index, raw_pair in enumerate(pairs):
        if not isinstance(raw_pair, dict):
            return None
        candidate_span = _paper_v5_span(
            raw_pair.get("candidate_span"), sentence=sentence
        )
        sentence_span = _paper_v5_span(
            raw_pair.get("sentence_span"), sentence=sentence
        )
        target_span = _paper_v5_span(
            raw_pair.get("target_span"), sentence=sentence
        )
        endpoint_span = _paper_v5_span(
            raw_pair.get("endpoint_span"), sentence=sentence
        )
        measurement_span = _paper_v5_span(
            raw_pair.get("measurement_span"), sentence=sentence
        )
        value_span = _paper_v5_span(
            raw_pair.get("value_span"), sentence=sentence
        )
        unit_span = _paper_v5_span(
            raw_pair.get("unit_span"), sentence=sentence
        )
        condition = raw_pair.get("assay_condition")
        if not isinstance(condition, dict):
            return None
        condition_span = _paper_v5_span(condition.get("span"), sentence=sentence)
        condition_value_span = _paper_v5_span(
            condition.get("value_span"), sentence=sentence
        )
        condition_unit_span = _paper_v5_span(
            condition.get("unit_span"), sentence=sentence
        )
        condition_subject_span = _paper_v5_span(
            condition.get("subject_span"), sentence=sentence
        )
        if any(
            item is None
            for item in (
                candidate_span,
                sentence_span,
                target_span,
                endpoint_span,
                measurement_span,
                value_span,
                unit_span,
                condition_span,
                condition_value_span,
                condition_unit_span,
                condition_subject_span,
            )
        ):
            return None
        assert candidate_span is not None
        assert sentence_span is not None
        assert target_span is not None
        assert endpoint_span is not None
        assert measurement_span is not None
        assert value_span is not None
        assert unit_span is not None
        assert condition_span is not None
        assert condition_value_span is not None
        assert condition_unit_span is not None
        assert condition_subject_span is not None
        relation_raw = raw_pair.get("relation_raw")
        relation_span_raw = raw_pair.get("relation_span")
        relation_span = (
            _paper_v5_span(relation_span_raw, sentence=sentence)
            if relation_span_raw is not None
            else None
        )
        wrapper_raw = raw_pair.get("endpoint_wrapper_span")
        wrapper_span = (
            _paper_v5_span(wrapper_raw, sentence=sentence)
            if wrapper_raw is not None
            else None
        )
        expected_role = "common_head" if index == 0 else "direct_follower"
        expected_scope = (
            "explicit_common_head" if index == 0 else "inherited_from_common_head"
        )
        value_decimal = _paper_v5_decimal(raw_pair.get("value_raw"))
        condition_decimal = _paper_v5_decimal(condition.get("value_raw"))
        if (
            raw_pair.get("grammar") != _PAPER_NATIVE_V5_GRAMMAR
            or raw_pair.get("pair_index") != index
            or raw_pair.get("pair_role") != expected_role
            or raw_pair.get("candidate_id") != candidate_ids[index]
            or candidate_span[:2] != all_candidate_occurrences[index][:2]
            or raw_pair.get("candidate_alias") != candidate_span[2]
            or sentence_span != (0, len(sentence), sentence)
            or raw_pair.get("sentence_sha256") != sentence_sha256
            or target_span[:2] != target_occurrence
            or raw_pair.get("target_alias") != target_span[2]
            or endpoint_span[:2] != endpoint_occurrence
            or _paper_v5_key(str(raw_pair.get("endpoint") or "")) != "ic50"
            or raw_pair.get("endpoint") != endpoint_span[2]
            or raw_pair.get("section") != section
            or raw_pair.get("publication_types") != publication_types
            or raw_pair.get("target_scope_basis") != expected_scope
            or raw_pair.get("endpoint_scope_basis") != expected_scope
            or raw_pair.get("assay_condition_scope_basis") != expected_scope
            or value_decimal is None
            or str(value_decimal) != str(raw_pair.get("value_decimal"))
            or raw_pair.get("value_raw") != value_span[2]
            or raw_pair.get("unit_raw") != unit_span[2]
            or unit_span[2] not in {"pM", "nM", "uM", "µM", "μM", "mM", "M"}
            or not (
                measurement_span[0]
                == (relation_span[0] if relation_span is not None else value_span[0])
                <= value_span[0]
                < value_span[1]
                <= unit_span[0]
                < unit_span[1]
                == measurement_span[1]
            )
            or re.fullmatch(
                (
                    rf"=\s*{re.escape(str(raw_pair.get('value_raw')))}\s*"
                    rf"{re.escape(str(raw_pair.get('unit_raw')))}"
                    if relation_raw == "="
                    else rf"{re.escape(str(raw_pair.get('value_raw')))}\s*"
                    rf"{re.escape(str(raw_pair.get('unit_raw')))}"
                ),
                measurement_span[2],
            )
            is None
            or condition_decimal is None
            or str(condition_decimal) != str(condition.get("value_decimal"))
            or condition.get("value_raw") != condition_value_span[2]
            or condition.get("unit_raw") != condition_unit_span[2]
            or condition.get("subject") != condition_subject_span[2]
            or not (
                condition_span[0]
                <= condition_value_span[0]
                < condition_value_span[1]
                <= condition_unit_span[0]
                < condition_unit_span[1]
                <= condition_subject_span[0]
                < condition_subject_span[1]
                <= condition_span[1]
            )
            or re.fullmatch(
                (
                    r"at\s+(?:(?:[A-Za-z][A-Za-z-]*)\s+){0,4}"
                    rf"(?:\[\s*{re.escape(str(condition.get('value_raw')))}\s*"
                    rf"{re.escape(str(condition.get('unit_raw')))}\s*\]|"
                    rf"{re.escape(str(condition.get('value_raw')))}\s*"
                    rf"{re.escape(str(condition.get('unit_raw')))})\s+"
                    rf"{re.escape(str(condition.get('subject')))}"
                ),
                condition_span[2],
                flags=re.IGNORECASE,
            )
            is None
        ):
            return None
        if relation_raw is None:
            if (
                relation_span_raw is not None
                or raw_pair.get("relation_normalized") is not None
                or raw_pair.get("relation_basis") != "reported_point_estimate"
                or raw_pair.get("literal_equality") is not False
            ):
                return None
        elif (
            relation_raw != "="
            or relation_span is None
            or relation_span[2] != "="
            or raw_pair.get("relation_normalized") != "="
            or raw_pair.get("relation_basis") != "literal_equality"
            or raw_pair.get("literal_equality") is not True
        ):
            return None
        if wrapper_span is None:
            if (
                raw_pair.get("endpoint_wrapper_raw") is not None
                or (
                    endpoint_span[0] > 0
                    and sentence[endpoint_span[0] - 1] == "["
                )
                or (
                    endpoint_span[1] < len(sentence)
                    and sentence[endpoint_span[1]] == "]"
                )
            ):
                return None
            lexical_endpoint = endpoint_span
        else:
            if (
                wrapper_span[2] != f"[{endpoint_span[2]}]"
                or wrapper_span[0] != endpoint_span[0] - 1
                or wrapper_span[1] != endpoint_span[1] + 1
                or raw_pair.get("endpoint_wrapper_raw") != wrapper_span[2]
            ):
                return None
            lexical_endpoint = wrapper_span
        if common_condition is None:
            common_condition = condition
        elif _paper_v5_bytes(condition) != _paper_v5_bytes(common_condition):
            return None
        if index == 0:
            head_text = sentence[:connectors[0].start()]
            open_parenthesis = head_text.find("(")
            if (
                candidate_span[0] != 0
                or head_text.count("(") != 1
                or head_text.count(")") != 1
                or not (
                    candidate_span[1]
                    < target_span[0]
                    < target_span[1]
                    < open_parenthesis
                    < lexical_endpoint[0]
                    < lexical_endpoint[1]
                    <= condition_span[0]
                    < condition_span[1]
                    <= measurement_span[0]
                )
                or not _paper_v5_closed_observed_head(
                    sentence,
                    candidate_end=candidate_span[1],
                    target_start=target_span[0],
                    target_end=target_span[1],
                    open_parenthesis=open_parenthesis,
                )
                or sentence[lexical_endpoint[1]:condition_span[0]].strip()
                or re.fullmatch(
                    r",\s*",
                    sentence[condition_span[1]:measurement_span[0]],
                )
                is None
                or re.fullmatch(
                    r"\s*\)",
                    sentence[measurement_span[1]:connectors[0].start()],
                )
                is None
            ):
                return None
            closing_parentheses.append(connectors[0].start())
        else:
            opening = sentence[candidate_span[1]:measurement_span[0]]
            if re.fullmatch(r"\s*\(\s*", opening) is None:
                return None
            close_match = re.match(r"\s*\)", sentence[measurement_span[1]:])
            if close_match is None:
                return None
            closing_parentheses.append(measurement_span[1] + close_match.end())
        numeric_spans.add(value_span[:2])
        numeric_spans.add(condition_value_span[:2])
        expected_evidence_hash = _paper_v5_pair_evidence_hash(
            raw_pair,
            sentence_sha256=sentence_sha256,
        )
        if raw_pair.get("evidence_sha256") != expected_evidence_hash:
            return None
        verified.append(raw_pair)

    connector = connectors[0]
    if connector.end() != all_candidate_occurrences[1][0]:
        return None
    for index in range(1, len(pairs) - 1):
        gap = sentence[
            closing_parentheses[index]:all_candidate_occurrences[index + 1][0]
        ]
        expected = (
            r",\s+and\s+" if index + 1 == len(pairs) - 1 else r",\s+"
        )
        if re.fullmatch(expected, gap, flags=re.IGNORECASE) is None:
            return None
    tail = sentence[closing_parentheses[-1]:]
    if re.fullmatch(
        r"(?:\.|,\s+and\s+the\s+most\s+selective\.)?",
        tail,
        flags=re.IGNORECASE,
    ) is None:
        return None
    observed_numbers = {
        (match.start(), match.end())
        for match in _PAPER_NATIVE_V5_NUMBER.finditer(sentence)
        if not any(
            start <= match.start() and match.end() <= end
            for start, end in ignored_numeric_ranges
        )
    }
    if observed_numbers != numeric_spans:
        return None
    return verified


def _paper_v5_method_indicator(sentence: str) -> str | None:
    if (
        not sentence
        or sentence != sentence.strip()
        or len(sentence) > 360
        or "\r" in sentence
        or "\n" in sentence
        or _PAPER_NATIVE_V5_METHOD_UNSAFE.search(sentence)
    ):
        return None
    if _PAPER_NATIVE_V5_METHOD_PASSIVE.search(sentence):
        return "passive_experimental_measurement_method"
    if _PAPER_NATIVE_V5_METHOD_ACTIVE.search(sentence):
        return "active_experimental_measurement_method"
    return None


def _paper_v5_verify_primary_study(
    raw: object,
    *,
    document_id: str,
    summary_types: list[str],
    fetch_types: list[str],
) -> dict[str, object] | None:
    expected_keys = {
        "article_title",
        "article_title_sha256",
        "attestation_sha256",
        "document_id",
        "fetch_publication_types",
        "fixture",
        "method_indicator",
        "method_sentence",
        "method_sentence_index",
        "method_sentence_sha256",
        "policy",
        "retraction_safety_policy",
        "retraction_safe",
        "correction_safe",
        "linked_status_notice",
        "linked_status_notice_types",
        "linked_status_notice_pmids",
        "schema_version",
        "status",
        "summary_publication_types",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        return None
    title = raw.get("article_title")
    sentence = raw.get("method_sentence")
    sentence_index = raw.get("method_sentence_index")
    indicator = raw.get("method_indicator")
    if (
        raw.get("schema_version") != "pubmed_primary_study_attestation/v2"
        or raw.get("policy")
        != "same_abstract_experimental_method_and_linked_notice_check_v2"
        or raw.get("status") != "verified"
        or raw.get("document_id") != document_id
        or raw.get("summary_publication_types") != summary_types
        or raw.get("fetch_publication_types") != fetch_types
        or raw.get("retraction_safety_policy")
        != "pubmed_types_and_comments_corrections_fail_closed_v1"
        or raw.get("retraction_safe") is not True
        or raw.get("correction_safe") is not True
        or raw.get("linked_status_notice") != "none_detected"
        or raw.get("linked_status_notice_types") != []
        or raw.get("linked_status_notice_pmids") != []
        or raw.get("fixture") is not False
        or not isinstance(title, str)
        or not title
        or title != title.strip()
        or len(title) > 500
        or "\r" in title
        or "\n" in title
        or raw.get("article_title_sha256")
        != sha256(title.encode("utf-8")).hexdigest()
        or type(sentence_index) is not int
        or sentence_index < 0
        or not isinstance(sentence, str)
        or raw.get("method_sentence_sha256")
        != sha256(sentence.encode("utf-8")).hexdigest()
        or _paper_v5_method_indicator(sentence) != indicator
    ):
        return None
    payload = {key: value for key, value in raw.items() if key != "attestation_sha256"}
    if raw.get("attestation_sha256") != _paper_v5_hash(payload):
        return None
    return raw


def _paper_v5_verify_group(
    raw: object,
    *,
    query_sha256: str,
    inventory: Mapping[str, object],
    fixed_ids: list[str],
    direct_ids: list[str],
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    document_id = raw.get("document_id")
    document_rank = raw.get("document_rank")
    sentence_index = raw.get("sentence_index")
    section = raw.get("section")
    sentence = raw.get("sentence")
    candidate_ids = _paper_v5_strings(raw.get("candidate_ids"), minimum=3)
    summary_types = _paper_v5_publication_types(
        raw.get("summary_publication_types")
    )
    fetch_types = _paper_v5_publication_types(
        raw.get("fetch_publication_types")
    )
    if (
        not isinstance(document_id, str)
        or not document_id.isdigit()
        or isinstance(document_rank, bool)
        or not isinstance(document_rank, int)
        or document_rank < 0
        or isinstance(sentence_index, bool)
        or not isinstance(sentence_index, int)
        or sentence_index < 0
        or section not in {"Abstract", "Results"}
        or not isinstance(sentence, str)
        or candidate_ids is None
        or summary_types is None
        or fetch_types is None
        or not set(candidate_ids).issubset(set(direct_ids))
        or not set(candidate_ids).issubset(set(fixed_ids))
        or raw.get("sentence_sha256")
        != sha256(sentence.encode("utf-8")).hexdigest()
    ):
        return None
    primary_study = _paper_v5_verify_primary_study(
        raw.get("primary_study_attestation"),
        document_id=document_id,
        summary_types=summary_types,
        fetch_types=fetch_types,
    )
    if primary_study is None:
        return None
    publication_types = sorted(set(summary_types).union(fetch_types))
    pairs = _paper_v5_verify_pairs(
        raw.get("pairs"),
        sentence=sentence,
        section=section,
        publication_types=publication_types,
        inventory=inventory,
        candidate_ids=candidate_ids,
    )
    if pairs is None:
        return None
    group_payload = {
        "candidate_ids": candidate_ids,
        "document_id": document_id,
        "document_rank": document_rank,
        "grammar": _PAPER_NATIVE_V5_GRAMMAR,
        "fetch_publication_types": fetch_types,
        "pair_evidence_sha256": [item["evidence_sha256"] for item in pairs],
        "primary_study_attestation_sha256": primary_study[
            "attestation_sha256"
        ],
        "query_sha256": query_sha256,
        "schema": _PAPER_NATIVE_V5_SEARCH_SCHEMA,
        "section": section,
        "sentence_index": sentence_index,
        "sentence_sha256": raw["sentence_sha256"],
        "summary_publication_types": summary_types,
    }
    if raw.get("group_sha256") != _paper_v5_hash(group_payload):
        return None
    return {
        "group": raw,
        "group_sha256": raw["group_sha256"],
        "candidate_ids": candidate_ids,
        "pairs": pairs,
        "document_id": document_id,
        "document_rank": document_rank,
        "sentence_index": sentence_index,
        "canonical_sha256": _paper_v5_hash(raw),
    }


def _paper_v5_normalized_type_list(value: object) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    normalized = sorted(
        {
            unicodedata.normalize("NFKC", item).strip().casefold()
            for item in value
            if item.strip()
        }
    )
    return normalized if value == normalized else None


def _paper_v5_is_primary(types: list[str]) -> bool:
    return bool(types) and not any(
        _PAPER_NATIVE_V5_EXCLUDED_TYPE.search(item) for item in types
    ) and bool(set(types).intersection(_PAPER_NATIVE_V5_PRIMARY_TYPES))


def _paper_v5_verify_record(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    publication_types = _paper_v5_normalized_type_list(
        raw.get("publication_types")
    )
    fetch_types = _paper_v5_normalized_type_list(
        raw.get("fetch_publication_types")
    )
    rejection_codes = _paper_v5_strings(raw.get("rejection_codes"))
    document_id = raw.get("document_id")
    status = raw.get("status")
    numeric_fields = (
        "eligible_section_count",
        "sentence_count",
        "bounded_sentence_count",
        "verified_group_count",
    )
    if (
        publication_types is None
        or fetch_types is None
        or rejection_codes is None
        or not isinstance(document_id, str)
        or not document_id.isdigit()
        or status not in {"verified", "rejected"}
        or any(
            isinstance(raw.get(key), bool)
            or not isinstance(raw.get(key), int)
            or int(raw[key]) < 0
            for key in numeric_fields
        )
        or (status == "verified")
        != (raw.get("verified_group_count", 0) > 0)
        or (status == "verified" and rejection_codes != [])
        or (status == "verified" and not _paper_v5_is_primary(publication_types))
        or (status == "verified" and not _paper_v5_is_primary(fetch_types))
    ):
        return None
    payload = {
        "bounded_sentence_count": raw["bounded_sentence_count"],
        "document_id": document_id,
        "eligible_section_count": raw["eligible_section_count"],
        "fetch_publication_types": fetch_types,
        "publication_types": publication_types,
        "rejection_codes": rejection_codes,
        "schema": _PAPER_NATIVE_V5_SEARCH_SCHEMA,
        "sentence_count": raw["sentence_count"],
        "status": status,
        "verified_group_count": raw["verified_group_count"],
    }
    if raw.get("record_sha256") != _paper_v5_hash(payload):
        return None
    return {
        "document_id": document_id,
        "status": status,
        "publication_types": publication_types,
        "fetch_publication_types": fetch_types,
        "bounded_sentence_count": raw["bounded_sentence_count"],
        "verified_group_count": raw["verified_group_count"],
    }


def _paper_v5_verify_search(
    raw: object,
    *,
    selected_group_raw: object,
    inventory: Mapping[str, object],
    fixed_ids: list[str],
    direct_ids: list[str],
) -> dict[str, object] | None:
    if not isinstance(raw, dict) or not _paper_v5_self_hash(
        raw, key="outcome_sha256"
    ):
        return None
    query = _paper_v5_expected_query(inventory)
    returned_ids = _paper_v5_strings(
        raw.get("returned_document_ids"),
        minimum=1,
        maximum=_PAPER_NATIVE_V5_MAX_DOCUMENTS,
    )
    verified_ids = _paper_v5_strings(
        raw.get("verified_candidate_ids"), minimum=3
    )
    calls = raw.get("calls")
    records_raw = raw.get("records")
    groups_raw = raw.get("groups")
    translation = raw.get("query_translation")
    if (
        query is None
        or returned_ids is None
        or verified_ids is None
        or not isinstance(calls, list)
        or not isinstance(records_raw, list)
        or not isinstance(groups_raw, list)
        or not isinstance(translation, str)
        or not translation
        or translation != translation.strip()
        or len(translation) > 20_000
        or raw.get("schema_version") != _PAPER_NATIVE_V5_SEARCH_SCHEMA
        or raw.get("policy_token") != _PAPER_NATIVE_V5_REQUIRED_PRIORITY
        or raw.get("grammar") != _PAPER_NATIVE_V5_GRAMMAR
        or raw.get("status") != "verified"
        or raw.get("reason_code") is not None
        or raw.get("query") != query
        or raw.get("query_sha256")
        != sha256(query.encode("utf-8")).hexdigest()
        or raw.get("query_translation_sha256")
        != sha256(translation.encode("utf-8")).hexdigest()
        or raw.get("search_sort") != "pub date"
        or raw.get("max_candidates") != _PAPER_NATIVE_V5_MAX_CANDIDATES
        or raw.get("max_documents") != _PAPER_NATIVE_V5_MAX_DOCUMENTS
        or raw.get("requested_candidate_count") != len(direct_ids)
        or raw.get("queried_candidate_count") != len(direct_ids)
        or raw.get("requested_document_count")
        != _PAPER_NATIVE_V5_MAX_DOCUMENTS
        or raw.get("returned_document_count") != len(returned_ids)
        or raw.get("unique_document_count") != len(returned_ids)
        or raw.get("returned_document_ids_sha256")
        != _paper_v5_hash({"ordered_values": returned_ids})
        or isinstance(raw.get("total_result_count"), bool)
        or not isinstance(raw.get("total_result_count"), int)
        or raw.get("total_result_count") < len(returned_ids)
        or raw.get("unsearched_result_count")
        != raw.get("total_result_count") - len(returned_ids)
        or len(calls) != 3
        or len(records_raw) != len(returned_ids)
        or not groups_raw
    ):
        return None
    records: list[dict[str, object]] = []
    for document_id, raw_record in zip(returned_ids, records_raw, strict=True):
        record = _paper_v5_verify_record(raw_record)
        if record is None or record["document_id"] != document_id:
            return None
        records.append(record)
    eligible_ids = [
        str(item["document_id"])
        for item in records
        if _paper_v5_is_primary(item["publication_types"])  # type: ignore[arg-type]
    ]
    query_sha256 = str(raw["query_sha256"])
    groups: list[dict[str, object]] = []
    for raw_group in groups_raw:
        group = _paper_v5_verify_group(
            raw_group,
            query_sha256=query_sha256,
            inventory=inventory,
            fixed_ids=fixed_ids,
            direct_ids=direct_ids,
        )
        if group is None:
            return None
        document_rank = int(group["document_rank"])
        if (
            document_rank >= len(returned_ids)
            or group["document_id"] != returned_ids[document_rank]
        ):
            return None
        record = records[document_rank]
        group_raw = group["group"]
        if (
            record["status"] != "verified"
            or group_raw.get("summary_publication_types")
            != record["publication_types"]
            or group_raw.get("fetch_publication_types")
            != record["fetch_publication_types"]
        ):
            return None
        groups.append(group)
    expected_order = sorted(
        groups,
        key=lambda item: (
            int(item["document_rank"]),
            int(item["sentence_index"]),
            str(item["group_sha256"]),
        ),
    )
    flattened_verified = list(
        dict.fromkeys(
            candidate_id
            for group in groups
            for candidate_id in group["candidate_ids"]  # type: ignore[union-attr]
        )
    )
    selected = groups[0]
    if (
        groups != expected_order
        or verified_ids != flattened_verified
        or raw.get("selected_group_sha256") != selected["group_sha256"]
        or not isinstance(selected_group_raw, dict)
        or _paper_v5_bytes(selected_group_raw)
        != _paper_v5_bytes(selected["group"])
        or raw.get("screened_document_count") != len(records)
        or raw.get("screened_sentence_count")
        != sum(int(item["bounded_sentence_count"]) for item in records)
        or any(
            int(record["verified_group_count"])
            != sum(
                group["document_id"] == record["document_id"]
                for group in groups
            )
            for record in records
        )
        or not _paper_v5_call(
            calls[0],
            stage="esearch",
            endpoint=_PAPER_NATIVE_V5_SEARCH_ENDPOINT,
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": _PAPER_NATIVE_V5_MAX_DOCUMENTS,
                "sort": "pub date",
            },
        )
        or not _paper_v5_call(
            calls[1],
            stage="esummary",
            endpoint=_PAPER_NATIVE_V5_SUMMARY_ENDPOINT,
            params={
                "db": "pubmed",
                "id": ",".join(returned_ids),
                "retmode": "json",
            },
        )
        or not _paper_v5_call(
            calls[2],
            stage="efetch",
            endpoint=_PAPER_NATIVE_V5_FETCH_ENDPOINT,
            params={
                "db": "pubmed",
                "id": ",".join(eligible_ids),
                "retmode": "xml",
                "rettype": "abstract",
            },
        )
    ):
        return None
    return {
        "selected_group": selected,
        "outcome_sha256": raw["outcome_sha256"],
        "canonical_sha256": _paper_v5_hash(raw),
        "returned_ids": returned_ids,
        "calls": calls,
    }


def _paper_v5_promotion(
    raw: object,
    *,
    selected_group: Mapping[str, object],
    fixed_ids: list[str],
) -> dict[str, object] | None:
    if not isinstance(raw, dict) or not _paper_v5_self_hash(
        raw, key="promotion_sha256"
    ):
        return None
    source_order = selected_group.get("candidate_ids")
    if not isinstance(source_order, list):
        return None
    fixed_order = [item for item in fixed_ids if item in set(source_order)]
    if (
        raw.get("schema_version") != "paper_native_v5_promotion_v1"
        or raw.get("policy") != _PAPER_NATIVE_V5_REQUIRED_PRIORITY
        or raw.get("status") != "verified"
        or raw.get("reason_code") is not None
        or raw.get("selected_group_sha256")
        != selected_group.get("group_sha256")
        or raw.get("selected_group_candidate_ids") != source_order
        or raw.get("verified_member_candidate_ids") != fixed_order
        or raw.get("verified_member_count") != len(source_order)
        or raw.get("minimum_group_candidate_count") != 3
        or raw.get("group_member_failure_rejects_entire_group") is not True
        or raw.get("cross_source_numeric_equality_required") is not False
        or raw.get("pubmed_value_used_for_activity_selection") is not False
        or raw.get("pubmed_value_used_for_candidate_rank") is not False
        or raw.get("pubmed_value_used_for_candidate_ordering") is not False
        or raw.get("no_partial_group_promotion") is not True
        or raw.get("no_reordering") is not True
        or raw.get("no_backfill") is not True
        or raw.get("fixture") is not False
    ):
        return None
    return {
        "member_ids_source_order": source_order,
        "member_ids_fixed_order": fixed_order,
        "promotion_sha256": raw["promotion_sha256"],
        "canonical_sha256": _paper_v5_hash(raw),
    }


def _paper_v5_primary_provenance(
    raw: object,
    *,
    fixed_ids: list[str],
    support_inventory_raw: object,
    alias_inventory_raw: object,
    search_raw: object,
    promotion_raw: object,
) -> bool:
    if not isinstance(raw, dict) or not _paper_v5_self_hash(
        raw, key="provenance_sha256"
    ):
        return False
    transport = raw.get("transport")
    calls = search_raw.get("calls") if isinstance(search_raw, dict) else None
    return bool(
        isinstance(transport, dict)
        and isinstance(calls, list)
        and raw.get("schema_version")
        == "primary_abstract_paper_native_gateway_v5"
        and raw.get("policy") == _PAPER_NATIVE_V5_REQUIRED_PRIORITY
        and transport.get("max_attempts") == 1
        and transport.get("required_max_attempts") == 1
        and transport.get("logical_call_limit") == 3
        and len(calls) == transport.get("logical_call_limit")
        and raw.get("legacy_primary_document_lookup_executed") is False
        and raw.get("legacy_primary_document_lookup_candidate_count") == 0
        and raw.get("legacy_primary_document_lookup_candidate_ids") == []
        and raw.get("legacy_primary_document_lookup_document_count") == 0
        and raw.get("legacy_primary_document_lookup_document_ids") == []
        and raw.get("legacy_pubmed_type_lookup_executed") is False
        and raw.get("legacy_pubmed_type_lookup_candidate_count") == 0
        and raw.get("legacy_pubmed_type_lookup_candidate_ids") == []
        and raw.get("legacy_pubmed_type_lookup_pmid_count") == 0
        and raw.get("legacy_pubmed_type_lookup_pmid_ids") == []
        and raw.get("fixed_pool_candidate_ids") == fixed_ids
        and raw.get("fixed_pool_candidate_ids_sha256")
        == _paper_v5_hash(fixed_ids)
        and raw.get("direct_support_inventory") == support_inventory_raw
        and raw.get("alias_inventory") == alias_inventory_raw
        and raw.get("search_outcome") == search_raw
        and raw.get("promotion") == promotion_raw
        and raw.get("pubmed_or_result_used_for_support_selection") is False
        and raw.get("pubmed_value_used_for_activity_selection") is False
        and raw.get("cross_source_numeric_equality_required") is False
        and raw.get("no_paired_v4_search_or_fallback") is True
        and raw.get("no_reordering") is True
        and raw.get("no_backfill") is True
        and raw.get("fixture") is False
    )


def _paper_v5_candidate_uses_only_support_activity(
    candidate: object,
    support: Mapping[str, object],
) -> bool:
    activities = getattr(candidate, "activities", None)
    typed = support.get("typed_measurement_tuple")
    if not isinstance(activities, list) or len(activities) != 1 or not isinstance(typed, dict):
        return False
    activity = activities[0]
    return bool(
        getattr(activity, "endpoint", None) == typed.get("endpoint") == "IC50"
        and getattr(activity, "relation", None) == typed.get("relation") == "="
        and _paper_v5_decimal(getattr(activity, "value", None))
        == _paper_v5_decimal(typed.get("value"))
        and getattr(activity, "unit", None) == typed.get("unit")
        and getattr(activity, "assay_type", None) == typed.get("assay_type")
        and getattr(activity, "source_evidence_id", None)
        == typed.get("source_evidence_id")
        and support.get("pubmed_value_used_for_selection") is False
        and support.get("cross_source_numeric_equality_required") is False
    )


def _paper_v5_verify_candidate_envelope(
    *,
    candidate: object,
    evidence: object,
    activity: object,
    envelope: object,
) -> dict[str, object] | None:
    if (
        not isinstance(envelope, dict)
        or not _paper_v5_self_hash(envelope, key="attestation_sha256")
        or envelope.get("schema_version")
        != _PAPER_NATIVE_V5_ENVELOPE_SCHEMA
        or envelope.get("source_contract")
        != _PAPER_NATIVE_V5_SOURCE_CONTRACT
        or envelope.get("policy") != _PAPER_NATIVE_V5_REQUIRED_PRIORITY
        or envelope.get("verification_status") != "verified"
        or envelope.get("is_primary_support") is not True
        or envelope.get("reason_code") is not None
        or envelope.get("source") != "PubMed"
        or envelope.get("cross_source_numeric_equality_required") is not False
        or envelope.get("cross_source_numeric_comparison_performed") is not False
        or envelope.get("pubmed_value_used_for_activity_selection") is not False
        or envelope.get("pubmed_value_used_for_candidate_rank") is not False
        or envelope.get("pubmed_value_used_for_candidate_ordering") is not False
        or envelope.get("candidate_seed_activity_is_pre_pubmed_chembl_support")
        is not True
        or envelope.get("no_partial_group_promotion") is not True
        or envelope.get("no_reordering") is not True
        or envelope.get("no_backfill") is not True
        or envelope.get("unit_conversion_applied") is not False
        or envelope.get("missing_values_imputed") is not False
        or envelope.get("fixture") is not False
    ):
        return None
    candidate_metadata = getattr(candidate, "metadata", None)
    evidence_metadata = getattr(evidence, "metadata", None)
    target = getattr(candidate, "target", None)
    if (
        not isinstance(candidate_metadata, dict)
        or not isinstance(evidence_metadata, dict)
        or target is None
        or candidate_metadata.get(_PAPER_NATIVE_V5_ENVELOPE_KEY) != envelope
        or evidence_metadata.get(_PAPER_NATIVE_V5_ENVELOPE_KEY) != envelope
        or getattr(evidence, "source", None) != "ChEMBL"
        or getattr(evidence, "supports", None) != "support"
        or getattr(activity, "source_evidence_id", None)
        != getattr(evidence, "evidence_id", None)
    ):
        return None
    molecule_id = _paired_v4_candidate_chembl_id(candidate)
    fixed_ids = _paper_v5_strings(
        envelope.get("fixed_pool_candidate_ids"),
        minimum=1,
        maximum=_PAPER_NATIVE_V5_MAX_CANDIDATES,
    )
    discovery_raw = evidence_metadata.get("candidate_discovery_bounds")
    discovery = _paper_v5_discovery_bounds(discovery_raw)
    phase_by_id = discovery.get("phase_by_id") if discovery is not None else None
    mechanisms_by_id = (
        discovery.get("mechanisms_by_id") if discovery is not None else None
    )
    if (
        molecule_id is None
        or fixed_ids is None
        or discovery is None
        or envelope.get("molecule_chembl_id") != molecule_id
        or molecule_id not in fixed_ids
        or envelope.get("fixed_pool_rank") != fixed_ids.index(molecule_id) + 1
        or envelope.get("fixed_pool_candidate_ids_sha256")
        != _paper_v5_hash(fixed_ids)
        or fixed_ids != discovery["fixed_ids"]
        or candidate_metadata.get("candidate_discovery_bounds") != discovery_raw
        or not isinstance(phase_by_id, dict)
        or candidate_metadata.get("max_phase") != phase_by_id.get(molecule_id)
        or not isinstance(mechanisms_by_id, dict)
        or candidate_metadata.get("mechanism_identity")
        != mechanisms_by_id.get(molecule_id)
        or evidence_metadata.get("mechanism_identity")
        != mechanisms_by_id.get(molecule_id)
        or candidate_metadata.get("direction_requested") != "inhibit"
        or candidate_metadata.get("direction_evidence_rules")
        != ["measured_ic50_endpoint"]
        or evidence_metadata.get("intervention_direction_requested")
        != "inhibit"
        or evidence_metadata.get("direction_evidence_rule")
        != "measured_ic50_endpoint"
    ):
        return None
    direct_ids = discovery["direct_ids"]
    snapshots = discovery["snapshots"]
    assert isinstance(direct_ids, list)
    assert isinstance(snapshots, list)
    support_inventory_raw = discovery["support_inventory"]
    support_inventory = _paper_v5_support_inventory(
        support_inventory_raw,
        fixed_ids=fixed_ids,
        direct_ids=direct_ids,
        snapshots=snapshots,
        target_chembl_id=str(discovery["target_chembl_id"]),
        target_uniprot_id=str(getattr(target, "uniprot_id", "") or ""),
        resolution_by_id=discovery["resolution_by_id"],  # type: ignore[arg-type]
        mechanisms_by_id=discovery["mechanisms_by_id"],  # type: ignore[arg-type]
        complete_activity_pagination=bool(
            discovery.get("complete_activity_pagination")
        ),
        activity_pages_by_id=discovery.get("activity_pages_by_id"),  # type: ignore[arg-type]
    )
    if support_inventory is None:
        return None
    selected_by_id = support_inventory["selected_by_id"]
    if not isinstance(selected_by_id, dict):
        return None
    support_raw = envelope.get("pre_pubmed_chembl_support")
    if support_raw != selected_by_id.get(molecule_id):
        return None
    support = _paper_v5_support_snapshot(
        support_raw,
        candidate=candidate,
        evidence=evidence,
        activity=activity,
        fixed_ids=fixed_ids,
    )
    if support is None or not _paper_v5_candidate_uses_only_support_activity(
        candidate, support_raw
    ):
        return None
    source_separation = evidence_metadata.get("primary_source_contract")
    if (
        "primary_source" in evidence_metadata
        or not isinstance(source_separation, dict)
        or not _paper_v5_self_hash(
            source_separation, key="attestation_sha256"
        )
        or source_separation.get("schema_version")
        != "paper_native_v5_independent_source_separation_v1"
        or source_separation.get("policy")
        != _PAPER_NATIVE_V5_REQUIRED_PRIORITY
        or source_separation.get(
            "legacy_candidate_primary_evidence_v2_applicable"
        )
        is not False
        or source_separation.get(
            "legacy_chembl_document_metadata_lookup_executed"
        )
        is not False
        or source_separation.get("chembl_document_chembl_id")
        != support_raw.get("document_chembl_id")
        or source_separation.get("pubmed_document_id")
        != envelope.get("pmid")
        or source_separation.get("same_document_binding_required") is not False
        or source_separation.get("independent_measurements_not_equated") is not True
        or source_separation.get("fixture") is not False
    ):
        return None
    alias_inventory_raw = envelope.get("alias_inventory")
    alias_inventory = _paper_v5_alias_inventory(
        alias_inventory_raw,
        fixed_ids=fixed_ids,
        direct_ids=direct_ids,
        target=target,
    )
    if alias_inventory is None:
        return None
    group_raw = envelope.get("group")
    search_raw = envelope.get("search_outcome")
    search = _paper_v5_verify_search(
        search_raw,
        selected_group_raw=group_raw,
        inventory=alias_inventory,
        fixed_ids=fixed_ids,
        direct_ids=direct_ids,
    )
    if search is None:
        return None
    selected_group = search["selected_group"]
    if not isinstance(selected_group, dict):
        return None
    group = selected_group["group"]
    pairs = selected_group["pairs"]
    group_ids = selected_group["candidate_ids"]
    if (
        not isinstance(group, dict)
        or not isinstance(pairs, list)
        or not isinstance(group_ids, list)
        or envelope.get("group_member_candidate_ids") != group_ids
        or envelope.get("selected_group_sha256")
        != selected_group.get("group_sha256")
        or molecule_id not in group_ids
    ):
        return None
    matching_indices = [
        index
        for index, pair in enumerate(pairs)
        if pair.get("candidate_id") == molecule_id
    ]
    selected_index = envelope.get("selected_pair_index")
    if (
        len(matching_indices) != 1
        or selected_index != matching_indices[0]
        or envelope.get("selected_pair") != pairs[matching_indices[0]]
        or envelope.get("selected_pair_evidence_sha256")
        != pairs[matching_indices[0]].get("evidence_sha256")
    ):
        return None
    pair = pairs[matching_indices[0]]
    target_binding = envelope.get("target_binding")
    if (
        not isinstance(target_binding, dict)
        or target_binding.get("target_symbol") != getattr(target, "symbol", None)
        or target_binding.get("target_accession")
        != getattr(target, "uniprot_id", None)
        or target_binding.get("target_chembl_id")
        != discovery["target_chembl_id"]
        or target_binding.get("target_type") != "SINGLE PROTEIN"
        or target_binding.get("target_organism") != "Homo sapiens"
        or target_binding.get("target_alias") != pair.get("target_alias")
        or target_binding.get("target_span") != pair.get("target_span")
        or envelope.get("pmid") != selected_group.get("document_id")
        or envelope.get("source_url")
        != f"https://pubmed.ncbi.nlm.nih.gov/{selected_group.get('document_id')}/"
    ):
        return None
    promotion_raw = envelope.get("promotion_summary")
    promotion = _paper_v5_promotion(
        promotion_raw,
        selected_group=selected_group,
        fixed_ids=fixed_ids,
    )
    primary_search_raw = discovery["primary_search"]
    if (
        promotion is None
        or not _paper_v5_primary_provenance(
            primary_search_raw,
            fixed_ids=fixed_ids,
            support_inventory_raw=support_inventory_raw,
            alias_inventory_raw=alias_inventory_raw,
            search_raw=search_raw,
            promotion_raw=promotion_raw,
        )
    ):
        return None
    return {
        "molecule_id": molecule_id,
        "candidate_id": getattr(candidate, "candidate_id", None),
        "fixed_ids": fixed_ids,
        "direct_ids": direct_ids,
        "group_ids": group_ids,
        "group_sha256": selected_group["group_sha256"],
        "group_canonical_sha256": selected_group["canonical_sha256"],
        "group_canonical_json": _paper_v5_bytes(group).decode("utf-8"),
        "search_outcome_sha256": search["outcome_sha256"],
        "search_canonical_sha256": search["canonical_sha256"],
        "alias_inventory_sha256": alias_inventory["inventory_sha256"],
        "support_inventory_sha256": support_inventory["inventory_sha256"],
        "selection_order_sha256": support_inventory["selection_order_sha256"],
        "promotion_sha256": promotion["promotion_sha256"],
        "envelope_canonical_sha256": _paper_v5_hash(envelope),
    }


def _paper_v5_no_pubmed_numeric_projection(candidate: object) -> bool:
    activities = getattr(candidate, "activities", None)
    if not isinstance(activities, list):
        return False
    allowed_activity_fields = {
        "endpoint",
        "relation",
        "value",
        "unit",
        "assay_type",
        "source_evidence_id",
    }
    if any(set(vars(activity)) != allowed_activity_fields for activity in activities):
        return False

    metadata = getattr(candidate, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    envelope = metadata.get(_PAPER_NATIVE_V5_ENVELOPE_KEY)
    group = envelope.get("group") if isinstance(envelope, dict) else None
    pairs = group.get("pairs") if isinstance(group, dict) else None
    pubmed_values = {
        parsed
        for pair in pairs or []
        if isinstance(pair, dict)
        for parsed in (_paper_v5_decimal(pair.get("value_decimal")),)
        if parsed is not None
    }

    def walk(
        value: object,
        *,
        inside_envelope: bool = False,
        ranking_context: bool = False,
    ) -> bool:
        if isinstance(value, dict):
            if (
                ranking_context
                and str(value.get("source") or "").casefold() == "pubmed"
            ):
                return False
            for key, item in value.items():
                key_text = str(key).casefold()
                child_inside = inside_envelope or key in {
                    _PAPER_NATIVE_V5_ENVELOPE_KEY,
                    "primary_abstract_search",
                }
                child_ranking = ranking_context or any(
                    token in key_text
                    for token in ("rank", "score", "order", "selection")
                )
                if not child_inside and "pubmed" in key_text and "value" in key_text:
                    if item is not False:
                        return False
                if not child_inside and key_text in {
                    "paper_native_v5_value",
                    "paper_native_value",
                    "primary_abstract_value",
                }:
                    return False
                if not walk(
                    item,
                    inside_envelope=child_inside,
                    ranking_context=child_ranking,
                ):
                    return False
        elif isinstance(value, (list, tuple)):
            return all(
                walk(
                    item,
                    inside_envelope=inside_envelope,
                    ranking_context=ranking_context,
                )
                for item in value
            )
        elif (
            ranking_context
            and not inside_envelope
            and _paper_v5_decimal(value) in pubmed_values
        ):
            return False
        return True

    return walk(metadata)


def _paper_v5_group_projection(
    packet: object,
    *,
    declared_group_ids: list[str],
    fixed_ids: list[str],
) -> bool:
    """Verify MA2 evaluated the finite prefix through quota and full group."""

    metadata = getattr(packet, "metadata", None)
    candidates = getattr(packet, "candidates", None)
    if not isinstance(metadata, dict) or not isinstance(candidates, list):
        return False
    backfill = metadata.get("candidate_backfill")
    candidate_yield = metadata.get("candidate_yield")
    if not isinstance(backfill, dict) or not isinstance(candidate_yield, dict):
        return False
    projection = backfill.get("paper_native_v5_group_projection")
    if (
        not isinstance(projection, dict)
        or candidate_yield.get("paper_native_v5_group_projection") != projection
        or not _paper_v5_self_hash(projection, key="projection_sha256")
        or projection.get("schema_version")
        != "paper_native_v5_group_projection_v1"
        or projection.get("policy") != _PAPER_NATIVE_V5_REQUIRED_PRIORITY
    ):
        return False
    declared = _paper_v5_strings(
        projection.get("declared_group_candidate_ids"), minimum=3
    )
    projected = _paper_v5_strings(
        projection.get("projected_group_candidate_ids"), minimum=3
    )
    if declared is None or projected is None:
        return False
    expected_projected = [
        molecule_id
        for molecule_id in fixed_ids
        if molecule_id in set(declared_group_ids)
    ]
    stop_rule = "eligible_quota_and_declared_group_projection_complete_v1"
    requested = projection.get("requested_eligible_count")
    returned = projection.get("eligible_returned_count")
    evaluated = projection.get("evaluated_fixed_pool_prefix_count")
    if (
        declared != declared_group_ids
        or projected != expected_projected
        or set(projected) != set(declared)
        or projection.get("declared_group_candidate_ids_sha256")
        != _paper_v5_hash(declared)
        or projection.get("projected_group_candidate_ids_sha256")
        != _paper_v5_hash(projected)
        or projection.get("projection_complete") is not True
        or type(requested) is not int
        or not 1 <= requested <= _PAPER_NATIVE_V5_MAX_CANDIDATES
        or type(returned) is not int
        or returned < 0
        or type(evaluated) is not int
        or not 1 <= evaluated <= len(fixed_ids)
        or projection.get("stop_rule") != stop_rule
        or projection.get("stop_rule_sha256")
        != sha256(stop_rule.encode("utf-8")).hexdigest()
        or projection.get("no_second_discovery") is not True
        or projection.get("no_backfill_beyond_fixed_pool") is not True
        or projection.get("fixture") is not False
        or len(candidates) != evaluated
    ):
        return False
    candidate_ids: list[str] = []
    status_by_id: dict[str, str] = {}
    for candidate in candidates:
        candidate_metadata = getattr(candidate, "metadata", None)
        molecule_id = _paired_v4_candidate_chembl_id(candidate)
        status = getattr(candidate, "status", None)
        if (
            not isinstance(candidate_metadata, dict)
            or candidate_metadata.get("paper_native_v5_group_projection")
            != projection
            or molecule_id is None
            or molecule_id in status_by_id
            or not isinstance(status, str)
        ):
            return False
        candidate_ids.append(molecule_id)
        status_by_id[molecule_id] = status
    evaluated_fixed_ids = fixed_ids[:evaluated]
    if (
        set(candidate_ids) != set(evaluated_fixed_ids)
        or len(candidate_ids) != len(evaluated_fixed_ids)
    ):
        return False
    recomputed_eligible = sum(
        status_by_id[molecule_id] != "excluded"
        for molecule_id in evaluated_fixed_ids
    )
    quota_reached = recomputed_eligible >= requested
    stop_satisfied = quota_reached and set(declared).issubset(
        evaluated_fixed_ids
    )
    if (
        returned != recomputed_eligible
        or projection.get("eligible_quota_reached") is not quota_reached
        or projection.get("stop_condition_satisfied") is not stop_satisfied
        or backfill.get("requested_eligible_count") != requested
        or backfill.get("discovery_pool_count") != len(fixed_ids)
        or backfill.get("evaluated_count") != evaluated
        or backfill.get("eligible_count") != recomputed_eligible
        or backfill.get("excluded_count")
        != evaluated - recomputed_eligible
        or backfill.get("quota_met") is not quota_reached
        or backfill.get("pool_exhausted") is not (evaluated == len(fixed_ids))
        or backfill.get("unbounded_search_used") is not False
        or candidate_yield.get("requested_eligible_count") != requested
        or candidate_yield.get("discovery_pool_count") != len(fixed_ids)
        or candidate_yield.get("evaluated_count") != evaluated
        or candidate_yield.get("eligible_count") != recomputed_eligible
        or candidate_yield.get("quota_met") is not quota_reached
        or candidate_yield.get("pool_exhausted")
        is not (evaluated == len(fixed_ids))
    ):
        return False
    minimal_stop: int | None = None
    for prefix_count in range(1, evaluated + 1):
        prefix = fixed_ids[:prefix_count]
        prefix_eligible = sum(
            status_by_id[molecule_id] != "excluded"
            for molecule_id in prefix
            if molecule_id in status_by_id
        )
        if prefix_eligible >= requested and set(declared).issubset(prefix):
            minimal_stop = prefix_count
            break
    if quota_reached:
        return bool(
            stop_satisfied
            and minimal_stop == evaluated
            and backfill.get("stop_reason")
            == "ELIGIBLE_QUOTA_AND_DECLARED_GROUP_PROJECTION_COMPLETE"
        )
    return bool(
        minimal_stop is None
        and evaluated == len(fixed_ids)
        and backfill.get("stop_reason") == "FINITE_DISCOVERY_POOL_EXHAUSTED"
    )


_KNOWN_ACTIVITY_UNITS = {"pM", "nM", "uM", "mM"}
_LIVE_DIRECTION_RULES = {
    "modulate": frozenset(
        {
            "measured_exact_target_activity",
            "modulation_accepts_exact_target_interaction",
            "de_novo_generation_intended_direction",
        }
    ),
    "inhibit": frozenset(
        {
            "measured_ic50_endpoint",
            "explicit_inhibitory_action",
            "drugcentral_action_type",
            "drugcentral_ic50_endpoint",
            "de_novo_generation_intended_direction",
        }
    ),
    "activate": frozenset(
        {
            "explicit_activating_action",
            "drugcentral_action_type",
            "de_novo_generation_intended_direction",
        }
    ),
}

# ADMET-AI classification outputs and the approved-DrugBank clearance
# percentile are screening signals, not confirmatory experimental findings.
# MA2 keeps candidates carrying these flags conditional; MA3 records the
# endpoint-specific work needed before a researcher may interpret the signal as
# a safety liability.  Keeping the mapping here makes the assurance record
# useful without converting a model prediction into a second exclusion rule.
_ADMET_CONFIRMATION_REQUIREMENTS = {
    "hepatotoxicity_high": (
        "hepatotoxicity",
        "DILI probability",
        "orthogonal human-hepatocyte and mechanism-based DILI assays",
    ),
    "herg_risk_high": (
        "herg_risk",
        "hERG-blocking probability",
        "concentration-response hERG patch-clamp testing",
    ),
    "ames_risk_high": (
        "ames_risk",
        "Ames-mutagenicity probability",
        "bacterial reverse-mutation (Ames) testing",
    ),
    "cyp3a4_inhibition_high": (
        "cyp3a4_inhibition",
        "CYP3A4-inhibition probability",
        "human-microsome or recombinant-enzyme CYP3A4 inhibition testing",
    ),
    "clearance_risk_high": (
        "clearance_risk",
        "approved-drug hepatocyte-clearance percentile",
        "human-hepatocyte intrinsic-clearance measurement and exposure modelling",
    ),
}


class ReviewerFinding(BaseModel):
    """A semantic concern proposed by the external reviewer.

    It deliberately cannot set the final gate verdict.  MA3 converts these
    proposals to immutable AuditFinding records and applies its own decision
    table afterwards.
    """

    category: Literal[
        "scientific",
        "model_appropriateness",
        "execution",
        "evidence_provenance",
        "safety",
        "schema",
    ]
    severity: Severity = Severity.MEDIUM
    code: str = Field(min_length=2, max_length=80)
    message: str = Field(min_length=5, max_length=1_500)
    artifact_id: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    failed_node: str | None = None


class ReviewerAssessment(BaseModel):
    summary: str = Field(min_length=1, max_length=2_000)
    additional_findings: list[ReviewerFinding] = Field(
        default_factory=list, max_length=20
    )
    uncertainties: list[str] = Field(default_factory=list, max_length=20)


UsageSink = Callable[[TokenUsage], Awaitable[None]]
PreflightSink = Callable[[str, str, int], Awaitable[None]]


class ReviewerRoutingFinding(ReviewerFinding):
    """Versioned routing suggestion, never a gate or evidence override."""
    remediation_kind: Literal["computational_rework", "data_required", "experiment_required",
        "tool_configuration", "resource_required", "researcher_decision", "integrity_block",
        "operator_review"] | None = None


class ReviewerRoutingAssessment(ReviewerAssessment):
    additional_findings: list[ReviewerRoutingFinding] = Field(default_factory=list, max_length=20)


def _reviewer_authority(findings: list[AuditFinding]) -> list[AuditFinding]:
    """Resolve gate authority from independently produced findings, not prose.

    A matching code alone is insufficient: the deterministic finding must name
    the same gate, category, artifact and claim set. The original model message
    and reported severity remain available in the newly created audit record.
    No persisted finding is mutated or retrospectively reclassified on load.
    """
    severity_rank = {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
                     Severity.HIGH: 3, Severity.CRITICAL: 4}
    deterministic = [f for f in findings if f.deterministic and not f.resolved]
    result = []
    for finding in findings:
        if finding.deterministic:
            result.append(finding)
            continue
        reported = finding.reported_severity or finding.severity
        confirmed = [f for f in deterministic if finding.artifact_id
            and f.gate == finding.gate and f.category == finding.category
            and f.artifact_id == finding.artifact_id
            and set(f.claim_ids) == set(finding.claim_ids)
            and f.code == finding.code.removeprefix("REVIEWER_")]
        ceiling = max((f.severity for f in confirmed), key=severity_rank.get) if confirmed else Severity.MEDIUM
        effective = min((reported, ceiling), key=severity_rank.get)
        result.append(finding.model_copy(update={
            "severity": effective,
            "reported_severity": reported,
            "effective_severity": effective,
            "confirmation_status": "confirmed" if confirmed else "unconfirmed",
            "confirmation_finding_ids": [f.finding_id for f in confirmed],
        }))
    return result


def _scoped_evidence_roles(artifacts: dict[str, BaseModel]) -> list[dict[str, Any]]:
    """Read-only review index; never merge roles across hypothesis boundaries."""
    packet = artifacts.get("hypothesis_packet")
    if getattr(packet, "schema_version", None) == "lossless_hypothesis_input_v1":
        from drugpilot.agents.hypothesis_projection import expand_hypothesis_projection
        packet = HypothesisPacket.model_validate(expand_hypothesis_projection(packet))
    result = []
    for hypothesis in getattr(packet, "hypotheses", []):
        scope = hypothesis.target.metadata.get("molecular_hypothesis_scope")
        scope_label = scope.get("label") if isinstance(scope, Mapping) else None
        assignment = hypothesis.target.metadata.get("claim_evidence_assignment")
        units = assignment.get("claim_units", []) if isinstance(assignment, Mapping) else []
        for record in hypothesis.evidence:
            matching_units = [unit for unit in units if isinstance(unit, Mapping)
                              and unit.get("evidence_id") == record.evidence_id]
            result.append({
                "evidence_id": record.evidence_id,
                "source": record.source,
                "external_id": record.external_id,
                "hypothesis_id": hypothesis.hypothesis_id,
                "target_symbol": hypothesis.target.symbol,
                "molecular_scope": scope_label,
                "claim_roles": [{"claim_id": unit.get("claim_id"), "role": unit.get("role"),
                                 "molecular_scope": unit.get("molecular_scope_label") or scope_label}
                                for unit in matching_units],
                "record_roles": record.metadata.get("claim_linkage_roles", []),
                "linked_claim_ids": record.metadata.get("linked_claim_ids", []),
            })
    # Only repeated publications need an extra cross-scope index. Full source
    # and claim records already exist in the immutable artifact payload.
    scopes: dict[tuple[str, str], set[str]] = {}
    for row in result:
        key = (row["source"], row["external_id"] or row["evidence_id"])
        scopes.setdefault(key, set()).add(row["hypothesis_id"])
    return [row for row in result if len(scopes[(row["source"], row["external_id"] or row["evidence_id"])]) > 1]


def _required_candidate_direction(brief: ResearchBrief, candidate: Any,
                                  hypotheses: HypothesisPacket | None) -> str | None:
    """Bind broad intake to a G0-retained hypothesis, never to MA2's assertion."""
    requested = brief.constraints.intervention_direction
    if requested not in {"modulate", "unspecified"}:
        return requested
    if hypotheses is None:
        return None
    linked_id = candidate.metadata.get("portfolio_hypothesis_id") or candidate.metadata.get("assay_variant_alignment_hypothesis_id")
    matching = [h for h in hypotheses.hypotheses
                if h.target.symbol.casefold() == candidate.target.symbol.casefold()
                and (not linked_id or h.hypothesis_id == linked_id)]
    if any(not brief.constraints.allows_hypothesis_direction(h.intervention_direction) for h in matching):
        return None
    directions = {h.intervention_direction for h in matching}
    return next(iter(directions)) if len(directions) == 1 else None


def _retained_primary_role_is_linked(hypothesis: Any, role: str, brief: ResearchBrief | None = None) -> bool:
    """Recompute retained claim/evidence linkage from immutable packet fields."""

    # New admission packets carry a source-bound MA3 assessment, not an MA1
    # regex approval. Never fall back to the legacy path for a broken certificate.
    if "ma3_admission" in hypothesis.target.metadata:
        from drugpilot.agents.hypothesis_admission import admitted_role_is_linked
        return brief is not None and admitted_role_is_linked(hypothesis, role, brief)
    if brief is not None and not brief.constraints.allows_hypothesis_direction(hypothesis.intervention_direction):
        return False
    assignment = hypothesis.target.metadata.get("claim_evidence_assignment")
    if not isinstance(assignment, Mapping):
        return False
    raw_units = assignment.get("claim_units")
    if not isinstance(raw_units, list):
        return False
    claims = {claim.claim_id: claim for claim in hypothesis.claims}
    evidence = {item.evidence_id: item for item in hypothesis.evidence}
    for unit in raw_units:
        if not isinstance(unit, Mapping) or unit.get("role") != role:
            continue
        claim_id = str(unit.get("claim_id") or "")
        evidence_id = str(unit.get("evidence_id") or "")
        claim = claims.get(claim_id)
        record = evidence.get(evidence_id)
        if (
            unit.get("evidence_cardinality") != 1
            or claim is None
            or record is None
            or evidence_id not in claim.evidence_ids
        ):
            continue
        molecular_scope = hypothesis.target.metadata.get("molecular_hypothesis_scope")
        if molecular_scope is not None or unit.get("molecular_scope_label"):
            from drugpilot.agents.molecular_scope import valid_hypothesis_scope
            from drugpilot.agents.ma1 import _primary_direction_excerpt, _primary_association_excerpt
            if brief is None or not valid_hypothesis_scope(brief,hypothesis):
                continue
            label = molecular_scope["label"]
            if unit.get("molecular_scope_label") != label:
                continue
            if role == "direct_target_direction_primary":
                excerpt = _primary_direction_excerpt(brief, hypothesis.target.symbol, record, label,
                    intervention_direction=hypothesis.intervention_direction)
            else:
                excerpt = _primary_association_excerpt(brief, hypothesis.target.symbol, record, label)
            if excerpt is None:
                continue
        elif (role == "direct_target_direction_primary" and brief is not None
              and brief.constraints.intervention_direction in {"modulate", "unspecified"}):
            # Broad intake makes the retained direction a derived scientific
            # claim. Recheck that direction against source text, rather than
            # trusting otherwise-consistent linkage metadata from MA1.
            from drugpilot.agents.ma1 import _primary_direction_excerpt
            if _primary_direction_excerpt(brief, hypothesis.target.symbol, record,
                    intervention_direction=hypothesis.intervention_direction) is None:
                continue
        linked_claim_ids = record.metadata.get("linked_claim_ids")
        linked_roles = record.metadata.get("claim_linkage_roles")
        if (
            not isinstance(linked_claim_ids, list)
            or claim_id not in linked_claim_ids
            or not isinstance(linked_roles, list)
            or role not in linked_roles
        ):
            continue
        if role == "direct_target_direction_primary" and (
            record.metadata.get("direct_support_eligible") is not True
            or record.metadata.get("verified_direct_primary") is not True
        ):
            continue
        return True
    return False


@dataclass(frozen=True)
class AuditPolicy:
    block_codes: frozenset[str] = frozenset(
        {
            "MISSING_PROVENANCE",
            "NONEXISTENT_ID",
            "HARD_CONSTRAINT_VIOLATION",
            "UNAPPROVED_MODEL",
            "UNAUTHORIZED_THRESHOLD_CHANGE",
            "MISSING_SAFETY",
            "FORBIDDEN_DRUG_NOT_EXCLUDED",
            "ROUTE_POLICY_VIOLATION",
            "UNAPPROVED_TOOL_OPERATION",
            "PLAN_TASK_TOOL_MISMATCH",
            "INACTIVE_ROUTE_TOOL_EXECUTED",
            "FORBIDDEN_DRUG_COMPUTED_AFTER_EXCLUSION",
            "FORBIDDEN_SUBSTRUCTURE_SCREEN_INVALID",
            "FORBIDDEN_SUBSTRUCTURE_COMPUTED_WITHOUT_CLEARANCE",
            "FORBIDDEN_SUBSTRUCTURE_COMPUTE_INPUT_MISMATCH",
            "MISSING_PLAN_TASK_TRACE",
            "BACKEND_BOUNDARY_VIOLATION",
            "EXECUTION_INPUT_HASH_MISMATCH",
            "FIXTURE_RESULT_IN_LIVE_RUN",
            "TOOL_RESULT_STATUS_MISMATCH",
            "PORTFOLIO_ROUTE_POLICY_VIOLATION",
            "CROSS_TARGET_RAW_SCORE_COMPARISON",
            "PORTFOLIO_PROVENANCE_MISMATCH",
            "PORTFOLIO_COMPLETENESS_MISMATCH",
            "DOCKING_SKIP_BASIS_INVALID",
            "SOURCE_COVERAGE_MANIFEST_MISSING",
            "SOURCE_COVERAGE_MANIFEST_INVALID",
            "SOURCE_COVERAGE_REGISTRY_MISMATCH",
            "SOURCE_COVERAGE_POLICY_MISMATCH",
            "SOURCE_COVERAGE_GAP",
            "SOURCE_COVERAGE_COUNT_MISMATCH",
            "SOURCE_COVERAGE_BUDGET_ESCALATION",
            "SOURCE_COVERAGE_DEPTH_MISMATCH",
            "SOURCE_COVERAGE_BUDGET_MISMATCH",
            "UNGUARDED_LLM_SOURCE_SUGGESTION",
        }
    )
    abstain_codes: frozenset[str] = frozenset(
        {
            "UNRESOLVED_DISEASE",
            "EVIDENCE_SOURCE_UNAVAILABLE",
            "DISEASE_ID_LABEL_MISMATCH",
            "BRIEF_FIELD_MISMATCH",
            "FIXTURE_ONLY_DISEASE_SUPPORT",
            "ROUTE_ABSTAIN_RECOMMENDATION_IGNORED",
            "SCIENTIFIC_TOOL_UNAVAILABLE",
            "FORBIDDEN_SUBSTRUCTURE_CHECK_UNAVAILABLE",
            "GATEWAY_NOT_READY",
            "GATEWAY_READINESS_MISSING",
            "LIVE_TOOL_VERSION_MISSING",
            "LIVE_RESULT_PROVENANCE_MISSING",
            "PORTFOLIO_PARTIAL_EXECUTION",
            "NO_QUALIFIED_CANDIDATES",
        }
    )
    abstain_ood_threshold: float = 0.85


class MA3AssuranceAgent:
    """Read-only assurance engine shared by G0 and G1.

    It never mutates upstream artifacts.  Audit and isolated validation produce
    new findings, then a deterministic decision table derives the verdict.
    """

    def __init__(
        self,
        reviewer_model: str = "deterministic-rules-v1",
        policy: AuditPolicy | None = None,
        reviewer_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.reviewer_model = reviewer_model
        self.policy = policy or AuditPolicy()
        self.reviewer_provider = reviewer_provider
        self.settings = settings or get_settings()

    async def audit_g0(
        self,
        brief: ResearchBrief,
        packet: HypothesisPacket,
        *,
        on_usage: UsageSink | None = None,
        before_review: PreflightSink | None = None,
    ) -> GateVerdict:
        findings = self._audit_g0(brief, packet)
        if packet.provisional_hypotheses:
            findings.append(self._finding(
                GateName.G0, "evidence_provenance", Severity.HIGH,
                "PROVISIONAL_HYPOTHESES_NOT_REVIEWED",
                "Provisional proposals cannot pass G0 before source-bound MA3 review.",
                packet.packet_id, "ma3.admission",
            ))
        if packet.metadata.get("admission_policy") == "ma3-source-bound-v1":
            from drugpilot.agents.hypothesis_admission import admitted_role_is_linked
            if len(packet.hypotheses) > 1 and len({h.target.symbol.casefold() for h in packet.hypotheses}) == 1:
                findings.append(self._finding(
                    GateName.G0, "scientific", Severity.MEDIUM,
                    "LIMITED_TARGET_DIVERSITY",
                    "Approved hypotheses cover one canonical target. Portfolio breadth is limited, "
                    "but distinct tracks or molecular scopes are not invalid solely for sharing a target.",
                    packet.packet_id, "ma3.portfolio",
                ))
            for hypothesis in packet.hypotheses:
                if not all(admitted_role_is_linked(hypothesis, role, brief) for role in (
                    "direct_target_direction_primary", "primary_target_disease_association",
                )):
                    findings.append(self._finding(
                        GateName.G0, "evidence_provenance", Severity.HIGH,
                        "MA3_ADMISSION_CERTIFICATE_INVALID",
                        "New-policy retained hypothesis is missing a valid source-bound MA3 assessment.",
                        hypothesis.hypothesis_id, "ma3.admission",
                    ))
        validations = self._validate_g0(packet)
        semantic, invoked, reviewer_note = await self._semantic_review_r18(
            gate=GateName.G0,
            artifacts={"research_brief": brief, "hypothesis_packet": packet},
            findings=findings,
            validations=validations,
            on_usage=on_usage,
            before_review=before_review,
        )
        return self._decide(
            GateName.G0,
            findings + semantic,
            validations,
            reviewer_invoked=invoked,
            reviewer_note=reviewer_note,
        )

    async def audit_g1(
        self,
        brief: ResearchBrief,
        hypotheses: HypothesisPacket,
        plan: PlanSpec,
        candidates: CandidatePacket,
        *,
        on_usage: UsageSink | None = None,
        before_review: PreflightSink | None = None,
        supplemental_evidence: BaseModel | None = None,
    ) -> GateVerdict:
        findings = self._audit_g1(brief, hypotheses, plan, candidates)
        validations = self._validate_g1(candidates)
        semantic, invoked, reviewer_note = await self._semantic_review_r18(
            gate=GateName.G1,
            artifacts={
                "research_brief": brief,
                "hypothesis_packet": hypotheses,
                "plan_spec": plan,
                "candidate_packet": candidates,
                **({"supplemental_evidence": supplemental_evidence} if supplemental_evidence is not None else {}),
            },
            findings=findings,
            validations=validations,
            on_usage=on_usage,
            before_review=before_review,
        )
        return self._decide(
            GateName.G1,
            findings + semantic,
            validations,
            reviewer_invoked=invoked,
            reviewer_note=reviewer_note,
        )

    async def _semantic_review(
        self,
        *,
        gate: GateName,
        artifacts: dict[str, BaseModel],
        findings: list[AuditFinding],
        validations: list[ValidationResult],
        on_usage: UsageSink | None,
        before_review: PreflightSink | None,
    ) -> tuple[list[AuditFinding], bool, str | None]:
        """Run an optional independent review without granting write authority."""

        if self.reviewer_provider is None:
            return [], False, None

        payload = {
            "gate": gate.value,
            "artifacts": {
                name: artifact.model_dump(mode="json")
                for name, artifact in artifacts.items()
            },
            "deterministic_findings": [
                finding.model_dump(mode="json") for finding in findings
            ],
            "validation_results": [
                validation.model_dump(mode="json") for validation in validations
            ],
        }
        fallback = ReviewerAssessmentOutput(
            summary="External review was not invoked; deterministic audit retained.",
            additional_findings=[],
            uncertainties=[],
        )
        prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        # Budget/policy preflight is authoritative and must propagate to the
        # Orchestrator's ABSTAIN path.  Only provider/network failures are an
        # optional-review fallback.
        if before_review is not None:
            await before_review(self.reviewer_model, prompt, 4_000)

        def reviewer_unavailable() -> tuple[list[AuditFinding], bool, str]:
            unavailable = AuditFinding(
                gate=gate,
                category="model_appropriateness",
                severity=Severity.MEDIUM,
                code="REVIEWER_UNAVAILABLE",
                message=(
                    "The optional external semantic reviewer was unavailable; "
                    "deterministic MA3 checks remained authoritative."
                ),
                failed_node=f"{gate.value.lower()}.reviewer",
                deterministic=False,
            )
            return (
                [unavailable],
                False,
                "External reviewer unavailable; deterministic fallback used.",
            )

        try:
            result = await self.reviewer_provider.structured(
                component=f"MA3-{gate.value}-Reviewer",
                model=self.reviewer_model,
                system=(
                    "You are DrugPilot's independent, read-only scientific reviewer. "
                    "Inspect the supplied immutable artifacts, deterministic findings, "
                    "and validation results. Report only additional, artifact-linked "
                    "concerns. Never rewrite an artifact, invent a tool result, change a "
                    "threshold, or choose PASS/REVISE/REPLAN/ABSTAIN/BLOCK; the "
                    "deterministic MA3 decision table alone selects the verdict."
                ),
                prompt=prompt,
                response_model=ReviewerAssessmentOutput,
                fallback=fallback,
                max_output_tokens=4_000,
            )
        except Exception as exc:
            failed_usage = usage_for_failed_call(
                exc,
                component=f"MA3-{gate.value}-Reviewer",
                model=self.reviewer_model,
            )
            if failed_usage is not None and on_usage is not None:
                # Usage accounting and its budget postflight are authoritative.
                # If the callback raises BudgetExceeded, do not convert it into
                # the optional-review fallback below.
                await on_usage(failed_usage)
            return reviewer_unavailable()

        if on_usage is not None:
            await on_usage(result.usage)
        try:
            # Conversion is deliberately inside the optional-review boundary,
            # but only after successful-call usage is durably accounted.  A
            # DTO/domain mismatch cannot erase a paid call or block the
            # deterministic audit.
            assessment = ReviewerAssessment.model_validate(result.data.model_dump())
        except Exception:
            return reviewer_unavailable()
        converted = [
            AuditFinding(
                gate=gate,
                category=finding.category,
                severity=finding.severity,
                code=f"REVIEWER_{finding.code.removeprefix('REVIEWER_')}",
                message=finding.message,
                artifact_id=finding.artifact_id,
                claim_ids=finding.claim_ids,
                failed_node=finding.failed_node,
                deterministic=False,
            )
            for finding in assessment.additional_findings
        ]
        note = assessment.summary
        if assessment.uncertainties:
            note += f" Reviewer reported {len(assessment.uncertainties)} uncertainty item(s)."
        return _reviewer_authority([*findings, *converted])[len(findings):], True, note

    async def _semantic_review_r18(
        self,
        *,
        gate: GateName,
        artifacts: dict[str, BaseModel],
        findings: list[AuditFinding],
        validations: list[ValidationResult],
        on_usage: UsageSink | None,
        before_review: PreflightSink | None,
    ) -> tuple[list[AuditFinding], bool, str | None]:
        """Run the current versioned reviewer policy without changing v2-v4."""

        if self.reviewer_provider is None:
            return [], False, None

        payload = {
            "gate": gate.value,
            "scoped_evidence_roles": _scoped_evidence_roles(artifacts),
            "artifacts": {
                name: artifact.model_dump(mode="json")
                for name, artifact in artifacts.items()
            },
            "deterministic_findings": [
                finding.model_dump(mode="json") for finding in findings
            ],
            "validation_results": [
                validation.model_dump(mode="json") for validation in validations
            ],
        }
        fallback = ReviewerAssessmentOutput(
            summary="External review was not invoked; deterministic audit retained.",
            additional_findings=[],
            uncertainties=[],
        )
        prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        # Budget/policy preflight is authoritative and must propagate to the
        # Orchestrator's ABSTAIN path. Only provider/network failures are an
        # optional-review fallback.
        if before_review is not None:
            await before_review(self.reviewer_model, prompt, 4_000)

        def reviewer_unavailable() -> tuple[list[AuditFinding], bool, str]:
            unavailable = AuditFinding(
                gate=gate,
                category="model_appropriateness",
                severity=Severity.MEDIUM,
                code="REVIEWER_UNAVAILABLE",
                message=(
                    "The optional external semantic reviewer was unavailable; "
                    "deterministic MA3 checks remained authoritative."
                ),
                failed_node=f"{gate.value.lower()}.reviewer",
                deterministic=False,
            )
            return (
                [unavailable],
                False,
                "External reviewer unavailable; deterministic fallback used.",
            )

        try:
            result = await self.reviewer_provider.structured(
                component=f"MA3-{gate.value}-Reviewer",
                model=self.reviewer_model,
                system=MA3_REVIEWER_SYSTEM_PROMPT,
                prompt=prompt,
                response_model=ReviewerRoutingAssessmentOutput,
                fallback=fallback,
                max_output_tokens=4_000,
            )
        except Exception as exc:
            failed_usage = usage_for_failed_call(
                exc,
                component=f"MA3-{gate.value}-Reviewer",
                model=self.reviewer_model,
            )
            if failed_usage is not None and on_usage is not None:
                # Usage accounting and its budget postflight are authoritative.
                # If the callback raises BudgetExceeded, do not convert it into
                # the optional-review fallback below.
                await on_usage(failed_usage)
            return reviewer_unavailable()

        if on_usage is not None:
            await on_usage(result.usage)
        try:
            # Conversion is deliberately inside the optional-review boundary,
            # but only after successful-call usage is durably accounted. A
            # DTO/domain mismatch cannot erase a paid call or block the
            # deterministic audit.
            assessment = ReviewerRoutingAssessment.model_validate(result.data.model_dump())
        except Exception:
            return reviewer_unavailable()
        converted = [
            AuditFinding(
                gate=gate,
                category=finding.category,
                severity=finding.severity,
                code=f"REVIEWER_{finding.code.removeprefix('REVIEWER_')}",
                message=finding.message,
                artifact_id=finding.artifact_id,
                claim_ids=finding.claim_ids,
                failed_node=finding.failed_node,
                remediation_kind=finding.remediation_kind,
                deterministic=False,
            )
            for finding in assessment.additional_findings
        ]
        note = assessment.summary
        if assessment.uncertainties:
            note += f" Reviewer reported {len(assessment.uncertainties)} uncertainty item(s)."
        return _reviewer_authority([*findings, *converted])[len(findings):], True, note

    def _audit_g0(
        self, brief: ResearchBrief, packet: HypothesisPacket
    ) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        if packet.brief_id != brief.brief_id:
            findings.append(
                self._finding(
                    GateName.G0,
                    "schema",
                    Severity.CRITICAL,
                    "ID_CONTINUITY_FAILURE",
                    "HypothesisPacket brief_id does not match ResearchBrief.",
                    packet.packet_id,
                    "ma1.selector",
                )
            )
        intake = packet.metadata.get("intake_validation")
        if isinstance(intake, dict):
            if intake.get("abstain_recommended") is True:
                reason_codes = intake.get("reason_codes")
                code = (
                    str(reason_codes[0])
                    if isinstance(reason_codes, list) and reason_codes
                    else "UNRESOLVED_DISEASE"
                )
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.HIGH,
                        code,
                        "Disease intake did not pass authoritative external verification.",
                        packet.packet_id,
                        "ma1.intake_guard",
                    )
                )
            elif (
                intake.get("disease_resolved") is not True
                and intake.get("external_verification_status") != "demo_fixture"
            ):
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.HIGH,
                        "UNRESOLVED_DISEASE",
                        "Disease resolution is absent even though intake did not abstain.",
                        packet.packet_id,
                        "ma1.intake_guard",
                    )
                )
        else:
            resolution = packet.metadata.get("resolved_disease")
            resolution_status = packet.metadata.get("disease_resolution_status")
            match_method = (
                resolution.get("match_method")
                if isinstance(resolution, dict)
                else None
            )
            component_demo = (
                packet.metadata.get("target_discovery_error")
                == "target_discovery_not_configured"
            )
            explicit_demo = bool(
                packet.metadata.get("demo_fixture_fallback_authorized")
            )
            if match_method == "top_search_hit" or resolution_status == "unresolved":
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.HIGH,
                        "UNRESOLVED_DISEASE",
                        "A non-exact or unresolved disease identity cannot pass G0.",
                        packet.packet_id,
                        "ma1.intake_guard",
                    )
                )
            elif (
                resolution_status == "source_unavailable"
                and not component_demo
                and not explicit_demo
            ):
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.HIGH,
                        "EVIDENCE_SOURCE_UNAVAILABLE",
                        "The authoritative disease resolver was unavailable.",
                        packet.packet_id,
                        "ma1.intake_guard",
                    )
                )
            if (
                packet.metadata.get("target_fallback_used")
                and int(packet.metadata.get("external_target_seed_count", 0) or 0) == 0
                and not component_demo
                and not explicit_demo
            ):
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.HIGH,
                        "FIXTURE_ONLY_DISEASE_SUPPORT",
                        "Fixture-only targets cannot establish disease-specific support.",
                        packet.packet_id,
                        "ma1.intake_guard",
                    )
                )
        if not packet.hypotheses:
            findings.append(
                self._finding(
                    GateName.G0,
                    "scientific",
                    Severity.HIGH,
                    "NO_HYPOTHESIS",
                    "MA1 did not return a testable hypothesis.",
                    packet.packet_id,
                    "ma1.proposer",
                )
            )
            return findings

        for hypothesis in packet.hypotheses:
            evidence_ids = {e.evidence_id for e in hypothesis.evidence}
            counter_records = [
                evidence
                for evidence in hypothesis.evidence
                if evidence.supports == "counter"
            ]
            counter_search = hypothesis.target.metadata.get(
                "counter_evidence_search",
                {},
            )
            if not counter_records:
                counter_status = (
                    str(counter_search.get("status"))
                    if isinstance(counter_search, dict)
                    else "not_recorded"
                )
                completed = counter_status in {
                    "completed_empty",
                    "completed_with_results",
                }
                context_only = counter_status == "completed_with_results"
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.MEDIUM,
                        (
                            "COUNTER_EVIDENCE_NOT_FOUND"
                            if completed
                            else "COUNTER_EVIDENCE_SEARCH_UNAVAILABLE"
                        ),
                        (
                            "A real counter-evidence search completed and returned "
                            "resistance or response-modifier context, but no record "
                            "was eligible as a claim-linked direct refutation."
                            if context_only
                            else "A real counter-evidence search completed but "
                            "returned no record with an explicit limitation signal."
                            if completed
                            else "No verified counter-evidence search outcome was available."
                        ),
                        hypothesis.hypothesis_id,
                        "ma1.evidence_retrieval",
                    )
                )
            if not hypothesis.target.symbol.strip():
                findings.append(
                    self._finding(
                        GateName.G0,
                        "schema",
                        Severity.CRITICAL,
                        "NONEXISTENT_ID",
                        "Hypothesis target identifier is missing.",
                        hypothesis.hypothesis_id,
                        "ma1.proposer",
                    )
                )
            if not hypothesis.falsification_test.strip():
                findings.append(
                    self._finding(
                        GateName.G0,
                        "scientific",
                        Severity.HIGH,
                        "NOT_FALSIFIABLE",
                        "Hypothesis has no falsification test.",
                        hypothesis.hypothesis_id,
                        "ma1.critic",
                    )
                )
            if not brief.constraints.allows_hypothesis_direction(hypothesis.intervention_direction):
                findings.append(
                    self._finding(
                        GateName.G0,
                        "scientific",
                        Severity.HIGH,
                        "RETAINED_INTERVENTION_DIRECTION_MISMATCH",
                        "A retained hypothesis does not match the ResearchBrief intervention direction or allowed hypothesis directions.",
                        hypothesis.hypothesis_id,
                        "ma1.selector",
                    )
                )
            retained_primary_roles = (
                (
                    "direct_target_direction_primary",
                    "RETAINED_DIRECT_TARGET_DIRECTION_PRIMARY_MISSING",
                    "A retained hypothesis lacks an independently traceable direct target-direction primary claim linkage.",
                ),
                (
                    "primary_target_disease_association",
                    "RETAINED_PRIMARY_TARGET_DISEASE_ASSOCIATION_MISSING",
                    "A retained hypothesis lacks an independently traceable primary target-disease association claim linkage.",
                ),
            )
            for role, code, message in retained_primary_roles:
                if _retained_primary_role_is_linked(hypothesis, role, brief):
                    continue
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.HIGH,
                        code,
                        message,
                        hypothesis.hypothesis_id,
                        "ma1.claim_evidence_assignment",
                    )
                )
            if not hypothesis.claims or not hypothesis.evidence:
                findings.append(
                    self._finding(
                        GateName.G0,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "MISSING_PROVENANCE",
                        "Hypothesis has no evidence-linked claim.",
                        hypothesis.hypothesis_id,
                        "ma1.proposer",
                    )
                )
            for claim in hypothesis.claims:
                missing = set(claim.evidence_ids) - evidence_ids
                if not claim.evidence_ids or missing:
                    findings.append(
                        AuditFinding(
                            gate=GateName.G0,
                            category="evidence_provenance",
                            severity=Severity.HIGH,
                            code="CLAIM_EVIDENCE_GAP",
                            message=f"Claim evidence is missing or unresolved: {sorted(missing)}",
                            artifact_id=hypothesis.hypothesis_id,
                            claim_ids=[claim.claim_id],
                            failed_node="ma1.proposer",
                        )
                    )
            if hypothesis.target.symbol in brief.constraints.forbidden_targets:
                findings.append(
                    self._finding(
                        GateName.G0,
                        "safety",
                        Severity.CRITICAL,
                        "HARD_CONSTRAINT_VIOLATION",
                        "Selected target is explicitly forbidden by the brief.",
                        hypothesis.hypothesis_id,
                        "ma1.selector",
                    )
                )
        return findings

    def _validate_g0(self, packet: HypothesisPacket) -> list[ValidationResult]:
        track_names = {hypothesis.track.value for hypothesis in packet.hypotheses}
        raw_ranks_by_track = {
            track: [
                hypothesis.rank
                for hypothesis in packet.hypotheses
                if hypothesis.track.value == track
            ]
            for track in track_names
        }
        ranks_by_track = {
            track: sorted(rank for rank in ranks if rank is not None)
            for track, ranks in raw_ranks_by_track.items()
        }
        rank_integrity = bool(packet.hypotheses) and bool(track_names) and all(
            all(rank is not None for rank in raw_ranks_by_track[track])
            and ranks == list(range(1, len(raw_ranks_by_track[track]) + 1))
            for track, ranks in ranks_by_track.items()
        )
        unique_targets = len({h.target.symbol.casefold() for h in packet.hypotheses})
        source_bound_policy = packet.metadata.get("admission_policy") == "ma3-source-bound-v1"
        # Breadth is a portfolio property, not evidence that a source-bound
        # hypothesis is false. Never require a second target to make a valid
        # single-target / multi-track research question executable. An empty
        # approved pool still fails, and rank/certificate checks remain hard.
        minimum_targets = 1 if source_bound_policy or not packet.hypotheses else min(2, len(packet.hypotheses))
        return [
            ValidationResult(
                name="rank_integrity_control",
                passed=rank_integrity,
                control_type="positive",
                details={"ranks_by_track": ranks_by_track},
            ),
            ValidationResult(
                name="target_diversity_control",
                passed=unique_targets >= minimum_targets,
                control_type="negative",
                details={
                    "unique_targets": unique_targets,
                    "hypothesis_count": len(packet.hypotheses),
                    "minimum_required": minimum_targets,
                    "policy": "nonempty_approved_pool_breadth_advisory" if source_bound_policy else "legacy_minimum_diversity",
                },
            ),
        ]

    def _audit_g1(
        self,
        brief: ResearchBrief,
        hypotheses: HypothesisPacket,
        plan: PlanSpec,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        if plan.hypothesis_packet_id != hypotheses.packet_id:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "PLAN_HYPOTHESIS_MISMATCH",
                    "PlanSpec does not reference the audited HypothesisPacket.",
                    plan.plan_id,
                    "planner",
                )
            )
        if brief.constraints.allowed_hypothesis_directions is not None:
            for hypothesis in hypotheses.hypotheses:
                if brief.constraints.allows_hypothesis_direction(hypothesis.intervention_direction):
                    continue
                findings.append(self._finding(
                    GateName.G1, "scientific", Severity.HIGH,
                    "RETAINED_INTERVENTION_DIRECTION_MISMATCH",
                    "The candidate workflow references a hypothesis outside allowed_hypothesis_directions.",
                    hypothesis.hypothesis_id, "ma2.judge",
                ))
        if packet.plan_id != plan.plan_id:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.HIGH,
                    "PLAN_TRACE_MISMATCH",
                    "CandidatePacket does not reference the active PlanSpec.",
                    packet.packet_id,
                    "ma2.judge",
                )
            )

        for code, message in validate_source_coverage_manifest(
            brief,
            plan.source_coverage,
            self.settings,
        ):
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    code,
                    message,
                    plan.plan_id,
                    "planner.source_coverage",
                )
            )

        effective_depth = self.settings.evidence_budget_for(
            brief.research_depth
        ).depth
        if plan.research_depth != effective_depth:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "SOURCE_COVERAGE_DEPTH_MISMATCH",
                    "PlanSpec research depth does not match ResearchBrief.",
                    plan.plan_id,
                    "planner.source_coverage",
                )
            )

        findings.extend(self._audit_ma2_execution_contract(packet))
        findings.extend(
            self._audit_assay_variant_alignment(hypotheses, packet)
        )
        findings.extend(self._audit_prospective_falsification(hypotheses, packet))
        if packet.portfolio_enabled:
            findings.extend(
                self._audit_target_portfolio(hypotheses, plan, packet)
            )
        if packet.execution_status in {
            ScientificExecutionStatus.UNAVAILABLE,
            ScientificExecutionStatus.FAILED,
        }:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.HIGH,
                    "SCIENTIFIC_TOOL_UNAVAILABLE",
                    (
                        "MA2 failed closed because required live scientific data or "
                        "tooling was unavailable; no fixture or neutral value was substituted. "
                        f"Reasons: {packet.abstain_reason_codes}."
                    ),
                    packet.packet_id,
                    "ma2.tool_dispatch",
                )
            )
            return findings
        if packet.execution_status is ScientificExecutionStatus.PARTIAL:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.HIGH,
                    "PORTFOLIO_PARTIAL_EXECUTION",
                    (
                        "MA2 retained valid target results but did not complete every "
                        "eligible target; the packet cannot receive a complete PASS. "
                        f"Reasons: {packet.abstain_reason_codes}."
                    ),
                    packet.packet_id,
                    "ma2.portfolio",
                )
            )

        findings.extend(self._audit_evidence_scoring_cardinality(packet))
        findings.extend(self._audit_activity_comparability(packet))
        findings.extend(
            self._audit_candidate_primary_sources_for_policy(
                brief, packet, plan=plan, hypotheses=hypotheses
            )
        )

        findings.extend(self._audit_route_policy(hypotheses, plan, packet))
        raw_focus = hypotheses.metadata.get("goal_focus")
        focus = (
            GoalFocus.model_validate(raw_focus)
            if isinstance(raw_focus, dict)
            else None
        )
        expected_priority, expected_derivation = (
            effective_discovery_priority(brief, focus)
            if focus is not None
            else (
                brief.constraints.discovery_priority,
                "research_brief_structured_constraint",
            )
        )
        if (
            plan.discovery_priority != expected_priority
            or plan.requested_discovery_priority
            != brief.constraints.discovery_priority
            or plan.discovery_priority_derivation != expected_derivation
        ):
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "DISCOVERY_PRIORITY_MISMATCH",
                    "PlanSpec effective discovery priority or its derivation does not "
                    "match the approved ResearchBrief and persisted GoalFocus.",
                    plan.plan_id,
                    "planner.policy",
                )
            )
        if (
            plan.forbidden_drugs != brief.constraints.forbidden_drugs
            or plan.forbidden_drug_ids != brief.constraints.forbidden_drug_ids
            or plan.forbidden_drug_classes
            != brief.constraints.forbidden_drug_classes
            or plan.forbidden_substructures != brief.constraints.forbidden_substructures
            or plan.reserved_control_compounds
            != brief.constraints.reserved_control_compounds
        ):
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "CANDIDATE_POLICY_MISMATCH",
                    (
                        "PlanSpec drug/substructure exclusions or reserved assay-control "
                        "identities do not match the approved ResearchBrief."
                    ),
                    plan.plan_id,
                    "planner.policy",
                )
            )

        approved_tools = set(plan.approved_tools)
        task_ids = {task.task_id for task in plan.tasks}
        task_contracts: dict[str, dict[str, set[str]]] = {}
        for task in plan.tasks:
            task_contracts[task.task_id] = {}
            for assignment in task.tools:
                task_contracts[task.task_id][assignment.tool_name] = set(
                    assignment.allowed_operations
                )
        route_decision = packet.metadata.get("route_decision", {})
        selected_route = (
            route_decision.get("selected_route")
            if isinstance(route_decision, dict)
            else None
        )
        inactive_discovery_tool = None if packet.portfolio_enabled else {
            "repurposing": "de_novo_generator",
            "de_novo": "repurposing_search",
        }.get(selected_route)
        for call in packet.tool_calls:
            if call.tool_name not in approved_tools:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "UNAPPROVED_TOOL",
                        f"Tool {call.tool_name} is outside the PlanSpec allow-list.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if call.task_id is None:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.HIGH,
                        "MISSING_PLAN_TASK_TRACE",
                        "Tool call has no PlanTask identifier.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            elif call.task_id in task_contracts and call.tool_name not in task_contracts[call.task_id]:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "PLAN_TASK_TOOL_MISMATCH",
                        (
                            f"Tool {call.tool_name} is not assigned to PlanTask "
                            f"{call.task_id}."
                        ),
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            elif (
                call.task_id in task_contracts
                and call.tool_name in task_contracts[call.task_id]
                and call.operation
                not in task_contracts[call.task_id][call.tool_name]
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "UNAPPROVED_TOOL_OPERATION",
                        (
                            f"Operation {call.operation} is outside the PlanSpec allow-list "
                            f"for {call.tool_name}."
                        ),
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if call.task_id and call.task_id not in task_ids:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.HIGH,
                        "UNKNOWN_PLAN_TASK",
                        f"Tool call references unknown task {call.task_id}.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if not call.success:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.HIGH,
                        "TOOL_CALL_FAILED",
                        f"Scientific tool call failed: {call.tool_name}/{call.operation}.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if inactive_discovery_tool and call.tool_name == inactive_discovery_tool:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "INACTIVE_ROUTE_TOOL_EXECUTED",
                        (
                            f"The {inactive_discovery_tool} tool ran despite the audited "
                            f"{selected_route} route decision."
                        ),
                        call.call_id,
                        "ma2.route_selection",
                    )
                )

        observed_tools = {call.tool_name for call in packet.tool_calls if call.success}
        if packet.portfolio_enabled:
            selected_discovery_tools = {
                {
                    "repurposing": "repurposing_search",
                    "de_novo": "de_novo_generator",
                }[item.route.value]
                for item in packet.target_executions
                if item.attempted and item.route is not None
            }
        else:
            selected_tool = {
                "repurposing": "repurposing_search",
                "de_novo": "de_novo_generator",
            }.get(selected_route)
            selected_discovery_tools = {selected_tool} if selected_tool else set()
        required_tools = {"target_normalizer", "candidate_judge"}
        required_tools.update(selected_discovery_tools)
        if any(candidate.status != "excluded" for candidate in packet.candidates):
            required_tools.update({"chemistry_validator", "docking", "admet_predictor"})
        missing_tools = required_tools - observed_tools
        if missing_tools:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "MISSING_REQUIRED_TOOL_EXECUTION",
                    f"Required selected-route tool executions are missing: {sorted(missing_tools)}.",
                    packet.packet_id,
                    "ma2.tool_dispatch",
                )
            )

        eligible_candidates = [
            candidate
            for candidate in packet.candidates
            if candidate.status != "excluded"
        ]
        if not eligible_candidates:
            findings.append(
                self._finding(
                    GateName.G1,
                    "scientific",
                    Severity.HIGH,
                    "NO_QUALIFIED_CANDIDATES",
                    (
                        "MA2 completed its available target searches but returned no "
                        "candidate eligible for researcher review."
                    ),
                    packet.packet_id,
                    "ma2.judge",
                )
            )
        if not packet.candidates:
            return findings

        trace_ids = {call.call_id for call in packet.tool_calls}
        calls_by_id = {call.call_id: call for call in packet.tool_calls}
        docking_required = any(
            assignment.tool_name == "docking" and assignment.required
            for task in plan.tasks
            for assignment in task.tools
        )
        for candidate in packet.candidates:
            # MA2 may deliberately retain a structurally invalid seed as an
            # excluded, traceable negative result.  That is evidence that the
            # upstream safety filter worked, not a candidate proposed for
            # progression.  G1 still audits its lineage and exclusion reason,
            # while completeness checks apply only to evaluable candidates.
            structurally_excluded = (
                candidate.status == "excluded"
                and "invalid_structure" in candidate.safety_flags
            )
            reserved_screen = screen_reserved_control(
                candidate_name=candidate.name,
                metadata=candidate.metadata,
                reserved_controls=brief.constraints.reserved_control_compounds,
            )
            expected_reserved_metadata = reserved_screen.as_metadata()
            stored_reserved_metadata = candidate.metadata.get(
                "reserved_control_screen"
            )
            if (
                stored_reserved_metadata != expected_reserved_metadata
                or candidate.metadata.get("control_only")
                is not reserved_screen.matched
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_SCREEN_MISMATCH",
                        (
                            "Stored assay-control screening metadata does not "
                            "match MA3's independent exact identity calculation."
                        ),
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            expected_reserved_flags = {
                *(
                    ["reserved_assay_control"]
                    if reserved_screen.matched
                    else []
                ),
                *(
                    ["reserved_control_identity_conflict"]
                    if reserved_screen.identity_conflict
                    else []
                ),
            }
            reserved_ineligible = (
                reserved_screen.excluded
                and candidate.status == "excluded"
                and candidate.score == 0.0
                and candidate.rank is None
                and candidate.target_rank is None
                and candidate.portfolio_rank is None
                and candidate.portfolio_score is None
                and candidate.metadata.get("policy_eligible") is False
                and expected_reserved_flags.issubset(candidate.safety_flags)
                and set(reserved_screen.reason_codes).issubset(
                    candidate.exclusion_reasons
                )
            )
            if reserved_screen.excluded and not reserved_ineligible:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "safety",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_CANDIDATE_NOT_EXCLUDED",
                        (
                            "An exact reserved assay control or stable-ID "
                            "identity conflict retained eligibility, score, or rank."
                        ),
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            forbidden_screen = screen_forbidden_drug(
                candidate_name=candidate.name,
                metadata=candidate.metadata,
                forbidden_names=brief.constraints.forbidden_drugs,
                forbidden_ids=brief.constraints.forbidden_drug_ids,
                forbidden_classes=brief.constraints.forbidden_drug_classes,
            )
            policy_excluded = (
                forbidden_screen.excluded
                and candidate.status == "excluded"
                and candidate.rank is None
                and candidate.score == 0.0
                and candidate.metadata.get("policy_eligible") is False
            )
            restricted_screen = candidate.metadata.get("restricted_safety_screen")
            restricted_disposition = (
                str(restricted_screen.get("disposition"))
                if isinstance(restricted_screen, dict)
                else None
            )
            restricted_sources = (
                restricted_screen.get("sources", [])
                if isinstance(restricted_screen, dict)
                else []
            )
            restricted_errors = [
                item
                for item in restricted_sources
                if isinstance(item, dict) and item.get("status") == "error"
            ]
            restricted_unresolved = (
                restricted_disposition == "unresolved" or bool(restricted_errors)
            )
            restricted_excluded = restricted_disposition == "excluded"
            restricted_ineligible = (
                (restricted_unresolved or restricted_excluded)
                and candidate.status == "excluded"
                and candidate.rank is None
                and candidate.score == 0.0
                and candidate.metadata.get("policy_eligible") is False
            )
            if restricted_unresolved:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "safety",
                        Severity.HIGH,
                        "RESTRICTED_SAFETY_LOOKUP_UNRESOLVED",
                        (
                            "A configured identity/regulatory or contraindication "
                            "lookup failed; unknown safety status cannot be treated as clean."
                        ),
                        candidate.candidate_id,
                        "ma2.restricted_safety_screen",
                    )
                )
            if (restricted_unresolved or restricted_excluded) and not restricted_ineligible:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "safety",
                        Severity.CRITICAL,
                        "RESTRICTED_SAFETY_CANDIDATE_NOT_EXCLUDED",
                        "A restricted-source safety exclusion or unresolved lookup remained eligible.",
                        candidate.candidate_id,
                        "ma2.restricted_safety_screen",
                    )
                )
            if candidate.status == "excluded" and candidate.rank is not None:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.HIGH,
                        "EXCLUDED_CANDIDATE_RANKED",
                        "An excluded candidate was assigned an eligible portfolio rank.",
                        candidate.candidate_id,
                        "ma2.judge",
                    )
                )
            if forbidden_screen.excluded and not policy_excluded:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "safety",
                        Severity.CRITICAL,
                        "FORBIDDEN_DRUG_NOT_EXCLUDED",
                        (
                            "A forbidden/withdrawn repurposing candidate remained eligible "
                            "or retained a rank."
                        ),
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            if (
                not structurally_excluded
                and not policy_excluded
                and (not candidate.smiles or not _SMILES_ALLOWED.fullmatch(candidate.smiles))
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "schema",
                        Severity.HIGH,
                        "INVALID_SMILES",
                        "Candidate SMILES failed the independent syntax check.",
                        candidate.candidate_id,
                        "ma2.standardize",
                    )
                )
            unresolved_traces = set(candidate.tool_trace_ids) - trace_ids
            if not candidate.tool_trace_ids or unresolved_traces:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "MISSING_PROVENANCE",
                        f"Candidate tool lineage is incomplete: {sorted(unresolved_traces)}",
                        candidate.candidate_id,
                        "ma2.evaluate",
                    )
                )
            candidate_calls = [
                calls_by_id[call_id]
                for call_id in candidate.tool_trace_ids
                if call_id in calls_by_id
            ]
            forbidden_screen_calls = [
                call
                for call in candidate_calls
                if call.operation
                in {
                    "screen_forbidden_drug",
                    "rescreen_forbidden_drug",
                }
            ]
            reserved_screen_calls = [
                call
                for call in candidate_calls
                if call.operation
                in {
                    "screen_reserved_assay_control",
                    "rescreen_reserved_assay_control",
                }
                and call.tool_name == "candidate_judge"
                and call.success
            ]
            expected_reserved_output_hash = sha256(
                json.dumps(
                    expected_reserved_metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            matching_reserved_screen_calls = [
                call
                for call in reserved_screen_calls
                if call.output_hash == expected_reserved_output_hash
            ]
            restricted_calls = [
                call
                for call in candidate_calls
                if call.operation
                == "precompute_identity_regulatory_contraindication"
            ]
            compute_calls = [
                call
                for call in candidate_calls
                if call.tool_name
                in {"chemistry_validator", "docking", "admet_predictor"}
            ]
            if not reserved_screen_calls:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_TRACE_MISSING",
                        (
                            "Candidate assay-control identity metadata has no "
                            "successful reserved-control screening trace."
                        ),
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            elif not matching_reserved_screen_calls:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_TRACE_MISMATCH",
                        (
                            "Reserved-control trace output does not hash to MA3's "
                            "independently recomputed exact identity result."
                        ),
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            elif (
                compute_calls
                and min(call.started_at for call in compute_calls)
                < min(
                    call.finished_at
                    for call in matching_reserved_screen_calls
                )
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_ORDER_VIOLATION",
                        (
                            "Candidate computation began before exact reserved-control "
                            "screening completed."
                        ),
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            candidate_compute_tools = {call.tool_name for call in compute_calls}
            if candidate.status != "excluded":
                missing_candidate_tools = {
                    "chemistry_validator",
                    "docking",
                    "admet_predictor",
                } - candidate_compute_tools
                if missing_candidate_tools:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "execution",
                            Severity.CRITICAL,
                            "CANDIDATE_COMPUTE_TRACE_INCOMPLETE",
                            (
                                "Eligible candidate lacks its own required compute traces: "
                                f"{sorted(missing_candidate_tools)}."
                            ),
                            candidate.candidate_id,
                            "ma2.evaluate",
                        )
                    )
            docking = candidate.docking_execution
            if (
                docking is not None
                and docking.status is ScientificExecutionStatus.SKIPPED
            ):
                skip_reason = docking.reason_code
                provenance = docking.provenance
                independently_linked_activity = (
                    self._has_target_linked_measured_activity(candidate)
                )
                basis_valid = not docking_required and (
                    provenance.get("plan_tool_required") is False
                )
                if skip_reason == "DIRECT_MEASURED_ACTIVITY_AVAILABLE":
                    basis_valid = basis_valid and (
                        independently_linked_activity
                        and provenance.get("target_linked_activity_verified") is True
                        and provenance.get("measured_activity_count")
                        == len(candidate.activities)
                    )
                elif skip_reason == "OPTIONAL_DOCKING_NOT_ASSESSED":
                    basis_valid = basis_valid and (
                        not independently_linked_activity
                        and provenance.get("target_linked_activity_verified") is False
                        and provenance.get("capability_status")
                        in {"not_configured", "unavailable"}
                        and bool(provenance.get("capability_reason_code"))
                        and provenance.get(
                            "candidate_eligible_without_target_engagement"
                        )
                        is False
                        and candidate.status == "excluded"
                        and "TARGET_ENGAGEMENT_EVIDENCE_UNAVAILABLE"
                        in candidate.exclusion_reasons
                    )
                else:
                    basis_valid = False
                if not basis_valid:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "evidence_provenance",
                            Severity.CRITICAL,
                            "DOCKING_SKIP_BASIS_INVALID",
                            (
                                "A skipped docking result is not supported by an optional "
                                "plan assignment and independently verifiable target evidence "
                                "or an explicit unavailable target-profile exclusion."
                            ),
                            candidate.candidate_id,
                            "ma2.docking",
                        )
                    )
            if policy_excluded and compute_calls:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "FORBIDDEN_DRUG_COMPUTED_AFTER_EXCLUSION",
                        "Forbidden/withdrawn candidate consumed chemistry, docking, or ADMET after screening.",
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            if reserved_screen.excluded and compute_calls:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_COMPUTED_AFTER_EXCLUSION",
                        (
                            "A reserved assay control or identity-conflict card "
                            "consumed chemistry, docking, or ADMET computation."
                        ),
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            if restricted_ineligible and compute_calls:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "RESTRICTED_SAFETY_COMPUTED_AFTER_STOP",
                        (
                            "Chemistry, docking, or ADMET ran after a restricted-source "
                            "safety exclusion or unresolved lookup."
                        ),
                        candidate.candidate_id,
                        "ma2.restricted_safety_screen",
                    )
                )
            if (
                isinstance(restricted_screen, dict)
                and candidate.route.value == "repurposing"
                and not restricted_calls
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "RESTRICTED_SAFETY_TRACE_MISSING",
                        "Repurposing safety-screen metadata has no corresponding tool trace.",
                        candidate.candidate_id,
                        "ma2.restricted_safety_screen",
                    )
                )
            if (
                restricted_calls
                and compute_calls
                and min(call.started_at for call in compute_calls)
                < min(call.finished_at for call in restricted_calls)
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "RESTRICTED_SAFETY_ORDER_VIOLATION",
                        "Restricted identity/regulatory checks occurred after expensive computation.",
                        candidate.candidate_id,
                        "ma2.restricted_safety_screen",
                    )
                )
            if not reserved_screen.excluded and (
                not forbidden_screen_calls
                or (
                compute_calls
                and min(call.started_at for call in compute_calls)
                < min(call.finished_at for call in forbidden_screen_calls)
                )
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "POLICY_SCREEN_ORDER_VIOLATION",
                        "Forbidden-drug screening was missing or occurred after docking/ADMET.",
                        candidate.candidate_id,
                        "ma2.policy_screen",
                    )
                )
            if not candidate.evidence:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.HIGH,
                        "MISSING_CANDIDATE_EVIDENCE",
                        "Candidate has no source evidence.",
                        candidate.candidate_id,
                        "ma2.repurpose",
                    )
                )
            # This check must also cover excluded cards, before the early
            # continue for structural/drug exclusions below.
            findings.extend(self._audit_forbidden_substructures(brief, candidate, candidate_calls))
            chemical_policy_result = candidate.metadata.get("forbidden_substructure_screen", {})
            chemical_ineligible = (
                bool(brief.constraints.forbidden_substructures)
                and isinstance(chemical_policy_result, dict)
                and chemical_policy_result.get("blocked") is True
                and candidate.status == "excluded"
                and candidate.metadata.get("policy_eligible") is False
            )
            if (
                structurally_excluded
                or reserved_ineligible
                or policy_excluded
                or restricted_ineligible
                or chemical_ineligible
            ):
                if not candidate.exclusion_reasons:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "execution",
                            Severity.HIGH,
                            "MISSING_EXCLUSION_REASON",
                            "Excluded candidate has no recorded reason.",
                            candidate.candidate_id,
                            "ma2.standardize",
                        )
                    )
                continue
            for activity in candidate.activities:
                if activity.unit not in _KNOWN_ACTIVITY_UNITS:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "scientific",
                            Severity.HIGH,
                            "ACTIVITY_UNIT_ERROR",
                            f"Unsupported activity unit {activity.unit}.",
                            candidate.candidate_id,
                            "ma2.activity_normalization",
                        )
                    )
                if activity.value > 1_000_000:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "scientific",
                            Severity.HIGH,
                            "ACTIVITY_SCALE_OUTLIER",
                            "Activity value is inconsistent with its declared unit.",
                            candidate.candidate_id,
                            "ma2.activity_normalization",
                        )
                    )
            required_direction = _required_candidate_direction(brief, candidate, hypotheses)
            if packet.backend_kind is ScientificBackendKind.FIXTURE:
                direction_mismatch = required_direction is None or bool(
                    candidate.mechanism
                    and required_direction not in candidate.mechanism.lower()
                )
            else:
                direction_mismatch = required_direction is None or not self._has_live_direction_attestation(
                    candidate.metadata,
                    required_direction,
                )
            if direction_mismatch:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "scientific",
                        Severity.HIGH,
                        "DIRECTION_MISMATCH",
                        "Candidate mechanism conflicts with the required intervention direction.",
                        candidate.candidate_id,
                        "ma2.judge",
                    )
                )
            applicability = candidate.metadata.get("applicability_assessment")
            applicability = applicability if isinstance(applicability, dict) else {}
            applicability_calibrated = applicability.get("calibrated") is True
            if (
                candidate.metrics.ood_score is not None
                and candidate.metrics.ood_score >= self.policy.abstain_ood_threshold
                and applicability_calibrated
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.HIGH,
                        "OOD_HIGH",
                        "Candidate is outside the configured model applicability domain.",
                        candidate.candidate_id,
                        "ma2.admet",
                    )
                )
            elif (
                candidate.metrics.ood_score is not None
                and candidate.metrics.ood_score >= self.policy.abstain_ood_threshold
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.MEDIUM,
                        "OOD_PROXY_HIGH_REQUIRES_REVIEW",
                        (
                            "The uncalibrated descriptor-distance proxy is high; "
                            "this is a prioritization warning, not proof that the "
                            "candidate is outside a model applicability domain."
                        ),
                        candidate.candidate_id,
                        "ma2.admet",
                    )
                )
            required_admet = {
                "hepatotoxicity",
                "herg_risk",
                "ames_risk",
                "cyp3a4_inhibition",
                "clearance_risk",
                "oral_bioavailability",
            }
            admet = candidate.metrics.admet_scores
            invalid_admet_values = [
                key
                for key, value in admet.items()
                if isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ]
            if (
                candidate.metadata.get("policy_eligible") is not False
                and (set(admet) != required_admet or invalid_admet_values)
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "safety",
                        Severity.CRITICAL,
                        "MISSING_SAFETY",
                        (
                            "Candidate does not have the exact six-endpoint 0..1 "
                            "ADMET/safety panel."
                        ),
                        candidate.candidate_id,
                        "ma2.admet",
                    )
                )
            if candidate.status == "conditional":
                confirmation_items: list[str] = []
                for flag, (
                    endpoint,
                    endpoint_label,
                    follow_up,
                ) in _ADMET_CONFIRMATION_REQUIREMENTS.items():
                    if flag not in candidate.safety_flags or endpoint not in admet:
                        continue
                    confirmation_items.append(
                        f"{endpoint_label} ({endpoint})={float(admet[endpoint]):.3f}; "
                        f"follow-up: {follow_up}"
                    )
                if confirmation_items:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "safety",
                            Severity.MEDIUM,
                            "ADMET_RISK_REQUIRES_EXPERIMENTAL_CONFIRMATION",
                            (
                                "Candidate remains conditional because the following "
                                "model-predicted ADMET signals require experimental "
                                "confirmation: "
                                + " | ".join(confirmation_items)
                                + ". These predictions are prioritization signals, not "
                                "confirmatory safety exclusions."
                            ),
                            candidate.candidate_id,
                            "ma2.admet",
                        )
                    )
        return findings

    def _audit_forbidden_substructures(
        self,
        brief: ResearchBrief,
        candidate: CandidateCard,
        candidate_calls: Sequence[ToolCallRecord],
    ) -> list[AuditFinding]:
        """Independently check molecular graphs and precompute policy lineage.

        A correctly excluded motif is not itself an audit failure. Missing
        chemistry is a HOLD, never evidence that the forbidden motif is absent.
        Earlier calculations on a clear seed remain valid if a later refined
        structure is excluded before any further calculation.
        """
        patterns = brief.constraints.forbidden_substructures
        if not patterns:
            return []
        findings: list[AuditFinding] = []

        def add(code: str, message: str, *, unavailable: bool = False) -> None:
            findings.append(self._finding(
                GateName.G1, "safety" if unavailable else "execution",
                Severity.HIGH if unavailable else Severity.CRITICAL,
                code, message, candidate.candidate_id, "ma2.policy_screen",
            ))

        current = screen_forbidden_substructures(candidate.smiles, patterns)
        invalid_structure_excluded = (
            current.status == "invalid_structure" and candidate.status == "excluded"
            and "invalid_structure" in candidate.safety_flags and candidate.rank is None
        )
        if current.blocked and not current.matched and not invalid_structure_excluded:
            add("FORBIDDEN_SUBSTRUCTURE_CHECK_UNAVAILABLE",
                "User chemical exclusion cannot be verified: " + ", ".join(current.reason_codes), unavailable=True)
        if current.matched and (
            candidate.status != "excluded" or candidate.rank is not None
            or candidate.target_rank is not None or candidate.portfolio_rank is not None
            or candidate.portfolio_score is not None or candidate.score != 0.0
            or candidate.metadata.get("policy_eligible") is True
        ):
            add("HARD_CONSTRAINT_VIOLATION",
                f"Eligible candidate contains a forbidden molecular subgraph: {list(current.matched_patterns)}")

        compute = [call for call in candidate_calls
                   if call.tool_name in {"chemistry_validator", "docking", "admet_predictor"}]
        calls_by_id = {call.call_id: call for call in candidate_calls}
        call_order = {call.call_id: index for index, call in enumerate(candidate_calls)}
        records = candidate.metadata.get("forbidden_substructure_screens", [])
        verified: list[tuple[ToolCallRecord, dict]] = []
        invalid = not isinstance(records, list)
        for entry in records if isinstance(records, list) else []:
            if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
                invalid = True
                continue
            result = entry["result"]
            call = calls_by_id.get(entry.get("call_id")) if isinstance(entry.get("call_id"), str) else None
            source_smiles = result.get("source_smiles")
            expected_output = sha256(json.dumps(result, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
            if (call is None or call.tool_name != "candidate_judge" or not call.success
                or call.operation not in {"screen_forbidden_substructure", "rescreen_forbidden_substructure"}
                or not isinstance(result.get("patterns"), (list, tuple))
                or list(result["patterns"]) != patterns
                or not isinstance(source_smiles, str)
                or call.input_hash != substructure_input_hash(source_smiles, patterns)
                or result.get("input_sha256") != call.input_hash or call.output_hash != expected_output):
                invalid = True
                continue
            if result.get("status") == "invalid_structure":
                # A parse-invalid graph is independently an invalid candidate,
                # not a claimed absence of the forbidden motif. A later
                # refinement may be valid, but cannot compute until re-screened.
                checked = screen_forbidden_substructures(source_smiles, patterns)
                if checked.status != "invalid_structure":
                    invalid = True
                    continue
            elif result.get("status") not in {"clear", "matched"}:
                add("FORBIDDEN_SUBSTRUCTURE_CHECK_UNAVAILABLE",
                    "The recorded chemical exclusion check was unresolved; no absence claim is permitted.", unavailable=True)
            else:
                checked = screen_forbidden_substructures(source_smiles, patterns)
                if checked.blocked and not checked.matched:
                    add("FORBIDDEN_SUBSTRUCTURE_CHECK_UNAVAILABLE",
                        "The recorded chemical exclusion cannot currently be independently rechecked.", unavailable=True)
                elif (checked.status != result.get("status")
                      or not isinstance(result.get("matched_patterns"), (list, tuple))
                      or list(checked.matched_patterns) != list(result["matched_patterns"])
                      or checked.standardized_parent_smiles != result.get("standardized_parent_smiles")
                      or checked.blocked != result.get("blocked")):
                    invalid = True
                    continue
            verified.append((call, result))
        if invalid or ((compute or candidate.status != "excluded") and not verified):
            add("FORBIDDEN_SUBSTRUCTURE_SCREEN_INVALID",
                "Candidate lacks an authentic source- and policy-bound chemical exclusion screen.")
        # Windows timestamps can tie for adjacent fast local operations; use
        # immutable execution order too, so a later refined screen cannot be
        # incorrectly applied to an earlier clear parent's calculation.
        verified.sort(key=lambda item: call_order[item[0].call_id])
        if verified and candidate.metadata.get("forbidden_substructure_screen") != verified[-1][1]:
            add("FORBIDDEN_SUBSTRUCTURE_SCREEN_INVALID",
                "Displayed chemical-policy result does not match its latest recorded execution.")
        for call in compute:
            preceding = [item for item in verified
                         if call_order[item[0].call_id] < call_order[call.call_id]
                         and item[0].finished_at <= call.started_at]
            if not preceding or preceding[-1][1].get("status") != "clear":
                add("FORBIDDEN_SUBSTRUCTURE_COMPUTED_WITHOUT_CLEARANCE",
                    "Molecular property, docking, or ADMET computation ran without prior clear chemical-policy screening.")
                break
            bound_call, bound_result = preceding[-1]
            binding = call.provenance.get("molecular_input_binding")
            arguments = binding.get("arguments") if isinstance(binding, dict) else None
            expected_argument_keys = {
                ("chemistry_validator", "validate_and_describe"): {"name", "smiles"},
                ("docking", "score_binding_pose"): {"smiles"},
                ("docking", "rescore_binding_pose"): {"candidate_id", "smiles"},
                ("admet_predictor", "predict_six_endpoint_panel"): {"smiles"},
                ("admet_predictor", "repeat_six_endpoint_panel"): {"smiles"},
            }.get((call.tool_name, call.operation))
            binding_valid = (
                isinstance(binding, dict)
                and binding.get("schema_version") == "chemical-compute-input-v1"
                and binding.get("screen_call_id") == bound_call.call_id
                and isinstance(arguments, dict) and set(arguments) == expected_argument_keys
                and isinstance(arguments.get("smiles"), str)
            )
            if binding_valid:
                payload = dict(arguments)
                if call.tool_name == "docking":
                    target_payload = candidate.target.model_dump(mode="json")
                    target_digest = sha256(json.dumps(target_payload, ensure_ascii=False,
                        sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
                    binding_valid = binding.get("target_input_sha256") == target_digest
                    payload["target"] = target_payload
                else:
                    binding_valid = binding.get("target_input_sha256") is None
                payload_hash = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
                binding_valid = binding_valid and payload_hash == call.input_hash
                if "candidate_id" in payload:
                    binding_valid = binding_valid and payload["candidate_id"] == candidate.candidate_id
            if not binding_valid:
                add("FORBIDDEN_SUBSTRUCTURE_COMPUTE_INPUT_MISMATCH",
                    "Actual molecular tool arguments are not hash-bound to the preceding chemical-policy screen.")
                continue
            computed = screen_forbidden_substructures(arguments["smiles"], patterns)
            if computed.status != "clear":
                if computed.matched or computed.status == "invalid_structure":
                    add("FORBIDDEN_SUBSTRUCTURE_COMPUTE_INPUT_MISMATCH",
                        "Actual computation used a forbidden or invalid molecular graph despite an earlier clear screen.")
                else:
                    add("FORBIDDEN_SUBSTRUCTURE_CHECK_UNAVAILABLE",
                        "Actual computed molecular input cannot currently be independently screened.", unavailable=True)
                continue
            if computed.standardized_parent_smiles != bound_result.get("standardized_parent_smiles"):
                add("FORBIDDEN_SUBSTRUCTURE_COMPUTE_INPUT_MISMATCH",
                    "Actual computation used a different molecular parent than the source-bound clear screen.")
        return findings

    def _audit_evidence_scoring_cardinality(
        self,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Recompute score cardinality while preserving every source row."""

        findings: list[AuditFinding] = []
        if packet.backend_kind is not ScientificBackendKind.LIVE:
            return findings
        for candidate in packet.candidates:
            if candidate.status == "excluded":
                continue
            identities = [
                (
                    item.source.strip().casefold(),
                    item.external_id.strip().casefold(),
                    " ".join(item.title.split()).casefold(),
                    str(item.excerpt_hash or "").strip().casefold(),
                    item.supports,
                )
                for item in candidate.evidence
            ]
            final_hashes = [
                sha256(
                    json.dumps(
                        identity,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for identity in identities
            ]
            final_unique_hashes = sorted(set(final_hashes))
            breakdown = candidate.metadata.get("activity_score_breakdown")
            recorded = (
                breakdown.get("evidence_cardinality")
                if isinstance(breakdown, Mapping)
                else None
            )
            components = (
                breakdown.get("components")
                if isinstance(breakdown, Mapping)
                else None
            )
            evidence_components = [
                item
                for item in components or []
                if isinstance(item, Mapping)
                and item.get("name") == "evidence_count"
            ]
            recorded_raw_hashes = (
                recorded.get("raw_record_identity_sha256s")
                if isinstance(recorded, Mapping)
                else None
            )
            recorded_unique_hashes = (
                recorded.get("unique_scientific_record_identity_sha256s")
                if isinstance(recorded, Mapping)
                else None
            )
            recorded_unique_count = (
                recorded.get("unique_scientific_record_count")
                if isinstance(recorded, Mapping)
                else None
            )
            recorded_count_contract_valid = bool(
                isinstance(recorded_raw_hashes, list)
                and all(
                    isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in recorded_raw_hashes
                )
                and isinstance(recorded_unique_hashes, list)
                and recorded_unique_hashes
                == sorted(set(recorded_raw_hashes))
                and recorded_unique_count == len(recorded_unique_hashes)
                and recorded.get("raw_record_count")
                == len(recorded_raw_hashes)
                and recorded.get("duplicate_physical_row_count")
                == len(recorded_raw_hashes) - len(recorded_unique_hashes)
                and recorded.get("policy_version")
                == "source-external-id-title-excerpt-support-v1"
                and recorded.get("raw_records_preserved") is True
                and recorded.get("duplicates_contribute_to_score") is False
                and not (
                    Counter(recorded_raw_hashes) - Counter(final_hashes)
                )
            )
            expected_value = min(1.0, float(recorded_unique_count or 0) / 2.0)
            component_valid = (
                len(evidence_components) == 1
                and evidence_components[0].get("value") == expected_value
                if recorded_unique_count
                else not evidence_components
            )
            portfolio_record = candidate.metadata.get("portfolio_ranking")
            portfolio_cardinality = (
                portfolio_record.get("evidence_cardinality")
                if isinstance(portfolio_record, Mapping)
                else None
            )
            portfolio_expected = {
                "policy_version": (
                    "source-external-id-title-excerpt-support-v1"
                ),
                "raw_record_count": len(final_hashes),
                "unique_scientific_record_count": len(final_unique_hashes),
                "duplicate_physical_row_count": (
                    len(final_hashes) - len(final_unique_hashes)
                ),
                "raw_record_identity_sha256s": final_hashes,
                "unique_scientific_record_identity_sha256s": (
                    final_unique_hashes
                ),
                "raw_records_preserved": True,
                "duplicates_contribute_to_score": False,
            }
            portfolio_valid = (
                not packet.portfolio_enabled
                or portfolio_cardinality == portfolio_expected
            )
            if (
                not recorded_count_contract_valid
                or not component_valid
                or not portfolio_valid
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.HIGH,
                        "EVIDENCE_SCORE_DUPLICATE_INFLATION",
                        (
                            "Evidence scoring does not match the independently "
                            "recomputed unique scientific-record cardinality; raw "
                            "physical rows must remain auditable but duplicates "
                            "must not multiply candidate or portfolio scores."
                        ),
                        candidate.candidate_id,
                        "ma2.ranking",
                    )
                )
        return findings

    @staticmethod
    def _semantic_hypothesis(hypotheses, packet, candidate):
        from drugpilot.tools.assay_variant_alignment import bound_semantic_hypothesis
        return bound_semantic_hypothesis(hypotheses, packet, candidate)

    def _audit_assay_variant_alignment(
        self,
        hypotheses: HypothesisPacket,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Recompute variant alignment and surface uncertainty without inference."""

        findings: list[AuditFinding] = []
        for candidate in packet.candidates:
            hypothesis = self._semantic_hypothesis(hypotheses, packet, candidate)
            if hypothesis is None:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.HIGH,
                        "ASSAY_VARIANT_HYPOTHESIS_LINK_MISSING",
                        "Candidate assay context cannot be linked to one audited hypothesis.",
                        candidate.candidate_id,
                        "ma2.assay_variant_alignment",
                    )
                )
                continue
            from drugpilot.agents.hypothesis_admission import admitted_tested_model_context
            expected_model_context = admitted_tested_model_context(hypothesis)
            if (expected_model_context is not None
                    and candidate.metadata.get("ma3_tested_model_context") != expected_model_context):
                findings.append(self._finding(
                    GateName.G1, "model_appropriateness", Severity.HIGH,
                    "MA3_TESTED_MODEL_CONTEXT_NOT_PRESERVED",
                    "The candidate did not preserve its source-bound tested-model intervention scope; "
                    "general-target or planned-model efficacy cannot be inferred.",
                    candidate.candidate_id, "ma2.assay_variant_alignment",
                ))
            expected = assay_variant_alignment(
                target_symbol=candidate.target.symbol,
                molecular_scope=hypothesis.target.metadata.get("molecular_hypothesis_scope"),
                hypothesis_texts=(
                    hypothesis.mechanism,
                    hypothesis.falsification_test,
                ),
                chembl_assay_context=candidate.metadata.get(
                    "chembl_assay_context"
                ),
                expected_target_chembl_id=candidate.metadata.get("measured_target"),
            )
            recorded = candidate.metadata.get("assay_variant_alignment")
            if packet.backend_kind is ScientificBackendKind.LIVE:
                activity_sources = {a.source_evidence_id for a in candidate.activities}
                source_contexts = [e for e in candidate.evidence if e.source == "ChEMBL"
                                   and e.evidence_id in activity_sources and e.metadata.get("chembl_assay_context")]
                if (source_contexts or expected["assay_target_binding_verified"]) and not any(
                           e.evidence_id in activity_sources
                           and e.metadata.get("target_chembl_id") == candidate.metadata.get("measured_target")
                           and e.metadata.get("chembl_assay_context") == candidate.metadata.get("chembl_assay_context")
                           for e in source_contexts):
                    findings.append(self._finding(
                        GateName.G1, "evidence_provenance", Severity.HIGH,
                        "ASSAY_VARIANT_SOURCE_BINDING_MISMATCH",
                        "Assay variant context is not bound to the candidate's preserved ChEMBL activity evidence.",
                        candidate.candidate_id, "ma2.assay_variant_alignment"))
                    continue
            if recorded != expected:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.HIGH,
                        "ASSAY_VARIANT_ALIGNMENT_ATTESTATION_MISMATCH",
                        "Candidate assay-variant alignment cannot be independently reproduced.",
                        candidate.candidate_id,
                        "ma2.assay_variant_alignment",
                    )
                )
                continue
            if expected["status"] == "explicit_mismatch":
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.MEDIUM,
                        "ASSAY_VARIANT_EXPLICIT_MISMATCH",
                        (
                            "The retained activity assay reports "
                            f"{expected['observed_assay_variants']}, while the "
                            "hypothesis context declares "
                            f"{expected['declared_hypothesis_variants']}; the "
                            "measurement is retained as a limitation and is not "
                            "treated as exact variant support."
                        ),
                        candidate.candidate_id,
                        "ma2.assay_variant_alignment",
                    )
                )
            elif expected["status"] == "unknown_not_assumed_match":
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.LOW,
                        "ASSAY_VARIANT_CONTEXT_UNKNOWN",
                        (
                            "The hypothesis declares a target variant, but the "
                            "activity assay lacks verified target/variant detail. "
                            "Unknown was not interpreted as wild type or as a match."
                        ),
                        candidate.candidate_id,
                        "ma2.assay_variant_alignment",
                    )
                )
            elif expected["status"] in {"variant_class_match_only", "partial_variant_overlap", "variant_annotation_match"}:
                findings.append(self._finding(
                    GateName.G1, "model_appropriateness", Severity.LOW,
                    {"variant_class_match_only": "ASSAY_VARIANT_CLASS_ONLY",
                     "partial_variant_overlap": "ASSAY_VARIANT_PARTIAL_OVERLAP",
                     "variant_annotation_match": "ASSAY_VARIANT_ANNOTATION_ONLY"}[expected["status"]],
                    "Reported variant labels overlap, but they do not establish the same exact variant sequence, insertion, compound-mutant context or experimental construct.",
                    candidate.candidate_id, "ma2.assay_variant_alignment"))
        return findings

    def _audit_prospective_falsification(
        self, hypotheses: HypothesisPacket, packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """An unperformed experiment is neither support nor a refutation.

        Independently check semantic invariants before reproducing the producer's
        closed parse. A recomputed hash alone cannot legitimise an invented
        observation, changed threshold, or legacy default-falsified state.
        This checks MA2's pre-experiment contract, not later laboratory results.
        """
        from drugpilot.agents.ma2 import MA2Agent

        findings: list[AuditFinding] = []
        for candidate in packet.candidates:
            record = candidate.metadata.get("prospective_falsification_status")
            if record is None and packet.backend_kind is not ScientificBackendKind.LIVE:
                continue
            hypothesis = self._semantic_hypothesis(hypotheses, packet, candidate)
            issues: list[str] = []
            if (hypothesis is None or hypothesis.target.symbol != candidate.target.symbol
                    or (hypothesis.target.uniprot_id is not None
                        and hypothesis.target.uniprot_id != candidate.target.uniprot_id)):
                issues.append("hypothesis_binding")
            if not isinstance(record, Mapping):
                issues.append("missing_attestation")
            else:
                if record.get("schema_version") != "candidate_prospective_falsification_v3":
                    issues.append("legacy_or_unknown_schema")
                for key in ("experimentally_observed", "default_falsification_applied",
                            "biological_refutation_established", "prospective_follow_up_support",
                            "used_for_source_eligibility", "used_for_candidate_ranking"):
                    if record.get(key) is not False:
                        issues.append(key)
                if record.get("status") != "not_evaluated":
                    issues.append("status")
                if record.get("prospective_evidence_disposition") != "unobserved_no_outcome":
                    issues.append("disposition")
                contract = record.get("machine_readable_contract")
                if not isinstance(contract, Mapping):
                    issues.append("missing_contract")
                else:
                    if (contract.get("schema_version") != "prospective_joint_criteria_contract/v2"
                            or contract.get("status") not in {"machine_readable", "unstructured"}
                            or contract.get("evaluation_status") != "not_evaluated"
                            or contract.get("when_unobserved") != "not_evaluated"
                            or "default_when_unobserved_or_any_criterion_unmet" in contract):
                        issues.append("unobserved_contract")

                    def unobserved(value: Any) -> bool:
                        if isinstance(value, Mapping):
                            return all(unobserved(v) for v in value.values())
                        return value is None

                    if not unobserved(contract.get("observations")):
                        issues.append("invented_observation")
                if hypothesis is not None:
                    expected = MA2Agent._attest_prospective_falsification_status(
                        [candidate], hypothesis,
                    )[0].metadata["prospective_falsification_status"]
                    # Includes candidate/hypothesis binding, exact source text,
                    # source/criteria hashes, thresholds and canonical payload hash.
                    if record != expected:
                        issues.append("source_or_attestation_reproduction")
            if issues:
                findings.append(self._finding(
                    GateName.G1, "model_appropriateness", Severity.HIGH,
                    "PROSPECTIVE_FALSIFICATION_ATTESTATION_INVALID",
                    "MA2 pre-experiment contract is invalid: " + ", ".join(issues)
                    + ". No wet-lab support or biological refutation may be inferred; regenerate the contract without altering source evidence.",
                    candidate.candidate_id, "ma2.prospective_falsification",
                ))
        return findings

    def _audit_activity_comparability(
        self,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Recompute live activity eligibility without trusting MA2 metadata.

        The audit deliberately uses the raw candidate target, activity and
        evidence records.  A measured value remains valid provenance when its
        exact assay context is incomplete, but it cannot enter a numeric
        potency comparison until every ``same_exact_chembl_assay_v1`` binding
        condition is independently reproduced here.
        """

        if (
            packet.backend_kind is not ScientificBackendKind.LIVE
            or not packet.candidates
        ):
            return []

        expected_rankings = derive_activity_rankings(packet.candidates)
        expected_summary = comparability_summary(expected_rankings)
        findings: list[AuditFinding] = []

        recorded_summary = packet.metadata.get("activity_comparability")
        summary_mismatches = [
            key
            for key, expected in expected_summary.items()
            if not isinstance(recorded_summary, Mapping)
            or recorded_summary.get(key) != expected
        ]
        if packet.portfolio_enabled:
            portfolio_attestation = packet.metadata.get("portfolio_ranking")
            expected_portfolio_attestation = {
                "activity_comparability_policy": ASSAY_COMPARABILITY_POLICY,
                "measured_activity_quality_scope": (
                    "value_independent_measurement_evidence_quality"
                ),
                "comparative_potency_directly_scored": False,
                "cross_assay_numeric_comparison": False,
                "unit_conversion_applied": False,
                "pchembl_used_for_ranking": False,
            }
            summary_mismatches.extend(
                f"portfolio_ranking.{key}"
                for key, expected in expected_portfolio_attestation.items()
                if not isinstance(portfolio_attestation, Mapping)
                or portfolio_attestation.get(key) != expected
            )
        if summary_mismatches:
            findings.append(
                self._finding(
                    GateName.G1,
                    "model_appropriateness",
                    Severity.HIGH,
                    "CROSS_ASSAY_ACTIVITY_COMPARISON_POLICY_VIOLATION",
                    (
                        "The packet-level activity-comparability attestation "
                        "cannot be reproduced under "
                        f"{ASSAY_COMPARABILITY_POLICY}; mismatched fields: "
                        f"{summary_mismatches}."
                    ),
                    packet.packet_id,
                    "ma2.ranking",
                )
            )

        expected_component_weights = {
            "activity": 0.28,
            "docking": 0.25,
            "predicted_safety": 0.25,
            "developability": 0.12,
            "evidence_count": 0.10,
        }
        for candidate in packet.candidates:
            expected = expected_rankings[candidate.candidate_id]
            recorded = candidate.metadata.get("activity_ranking")
            mismatches = [
                key
                for key, expected_value in expected.items()
                if not isinstance(recorded, Mapping)
                or recorded.get(key) != expected_value
            ]
            if packet.portfolio_enabled:
                portfolio_record = candidate.metadata.get("portfolio_ranking")
                portfolio_expected = {
                    "activity_comparability_policy": (
                        ASSAY_COMPARABILITY_POLICY
                    ),
                    "measured_activity_quality_scope": (
                        "value_independent_measurement_evidence_quality"
                    ),
                    "comparative_potency_directly_scored": False,
                    "comparative_potency_used_within_target_rank": (
                        expected["comparative_potency_used"] is True
                    ),
                    "cross_assay_numeric_comparison": False,
                    "pchembl_used_for_ranking": False,
                    "activity_comparability": expected_summary,
                }
                mismatches.extend(
                    f"portfolio_ranking.{key}"
                    for key, expected_value in portfolio_expected.items()
                    if not isinstance(portfolio_record, Mapping)
                    or portfolio_record.get(key) != expected_value
                )
                if isinstance(portfolio_record, Mapping):
                    factors = portfolio_record.get("factors")
                    if (
                        not isinstance(factors, Mapping)
                        or factors.get("measured_activity_quality")
                        != expected["measurement_evidence_quality"]
                    ):
                        mismatches.append(
                            "portfolio_ranking.measured_activity_quality"
                        )

            breakdown = candidate.metadata.get("activity_score_breakdown")
            if not isinstance(breakdown, Mapping):
                mismatches.append("activity_score_breakdown")
            else:
                embedded = breakdown.get("activity_ranking")
                if candidate.status != "excluded":
                    if not isinstance(embedded, Mapping):
                        mismatches.append("score_breakdown.activity_ranking")
                    else:
                        mismatches.extend(
                            f"score_breakdown.activity_ranking.{key}"
                            for key, expected_value in expected.items()
                            if embedded.get(key) != expected_value
                        )

                components = breakdown.get("components")
                component_list = (
                    list(components)
                    if isinstance(components, list)
                    and all(isinstance(item, Mapping) for item in components)
                    else []
                )
                if not isinstance(components, list) or len(component_list) != len(
                    components
                ):
                    mismatches.append("score_breakdown.components")
                names = [str(item.get("name") or "") for item in component_list]
                if len(names) != len(set(names)) or any(
                    name not in expected_component_weights for name in names
                ):
                    mismatches.append("score_breakdown.component_names")
                for item in component_list:
                    name = str(item.get("name") or "")
                    if (
                        name in expected_component_weights
                        and item.get("weight") != expected_component_weights[name]
                    ):
                        mismatches.append(f"score_breakdown.{name}.weight")

                activity_components = [
                    item for item in component_list if item.get("name") == "activity"
                ]
                if candidate.status == "excluded":
                    if (
                        component_list
                        or breakdown.get("excluded") is not True
                        or float(breakdown.get("final_score") or 0.0) != 0.0
                        or candidate.score != 0.0
                    ):
                        mismatches.append("score_breakdown.excluded_score")
                elif candidate.activities:
                    if len(activity_components) != 1 or activity_components[0].get(
                        "value"
                    ) != expected["activity_component_value"]:
                        mismatches.append("score_breakdown.activity.value")
                elif activity_components:
                    mismatches.append("score_breakdown.activity.unmeasured")

                if candidate.status != "excluded" and component_list:
                    try:
                        numerator = sum(
                            float(item["weight"]) * float(item["value"])
                            for item in component_list
                        )
                        denominator = sum(
                            float(item["weight"]) for item in component_list
                        )
                        warning_penalty = min(
                            0.24,
                            0.04 * len(candidate.safety_flags),
                        )
                        final_score = round(
                            min(
                                1.0,
                                max(
                                    0.0,
                                    numerator / denominator - warning_penalty,
                                ),
                            ),
                            6,
                        )
                        if (
                            abs(float(breakdown.get("numerator")) - numerator)
                            > 1e-6
                            or abs(
                                float(breakdown.get("denominator")) - denominator
                            )
                            > 1e-6
                            or abs(
                                float(breakdown.get("warning_penalty"))
                                - warning_penalty
                            )
                            > 1e-6
                            or float(breakdown.get("final_score")) != final_score
                            or candidate.score != final_score
                        ):
                            mismatches.append("score_breakdown.recalculation")
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        mismatches.append("score_breakdown.recalculation")

            if mismatches:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.HIGH,
                        "CROSS_ASSAY_ACTIVITY_COMPARISON_POLICY_VIOLATION",
                        (
                            "MA3 independently recomputed the activity ranking "
                            f"under {ASSAY_COMPARABILITY_POLICY} "
                            f"({ACTIVITY_RANKING_SCHEMA}) and found mismatched "
                            f"fields for {candidate.name}: "
                            f"{list(dict.fromkeys(mismatches))}."
                        ),
                        candidate.candidate_id,
                        "ma2.ranking",
                    )
                )
        return findings

    def _audit_candidate_primary_sources_for_policy(
        self,
        brief: ResearchBrief,
        packet: CandidatePacket,
        *,
        plan: PlanSpec | None = None,
        hypotheses: HypothesisPacket | None = None,
    ) -> list[AuditFinding]:
        """Dispatch only on an exact pre-registered source-policy token."""

        if _PAPER_NATIVE_V6_REQUIRED_PRIORITY in brief.human_priorities:
            return self._audit_primary_abstract_paper_native_v6(
                brief, packet, plan=plan, hypotheses=hypotheses
            )
        if _PAPER_NATIVE_V5_REQUIRED_PRIORITY in brief.human_priorities:
            return self._audit_primary_abstract_paper_native_v5(brief, packet)
        if _PAIRED_V4_REQUIRED_PRIORITY in brief.human_priorities:
            return self._audit_primary_abstract_paired_v4(brief, packet)
        return self._audit_candidate_primary_sources(brief, packet)

    def _audit_primary_abstract_paper_native_v6(
        self,
        brief: ResearchBrief,
        packet: CandidatePacket,
        *,
        plan: PlanSpec | None = None,
        hypotheses: HypothesisPacket | None = None,
    ) -> list[AuditFinding]:
        """Independently recompute the complete paper-native-v6 contract."""

        if not brief.data_policy.require_primary_sources or packet.backend_kind is not ScientificBackendKind.LIVE:
            return [self._finding(
                GateName.G1, "evidence_provenance", Severity.CRITICAL,
                "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_EXECUTION_CONTRACT_INVALID",
                "The required V6 policy needs primary-source enforcement on the LIVE backend.",
                packet.packet_id, "ma2.primary_evidence",
            )]
        priorities = brief.human_priorities
        if (
            priorities.count(_PAPER_NATIVE_V6_REQUIRED_PRIORITY) != 1
            or _PAPER_NATIVE_V5_REQUIRED_PRIORITY in priorities
            or _PAIRED_V4_REQUIRED_PRIORITY in priorities
        ):
            return [
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.CRITICAL,
                    "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_TOKEN_INVALID",
                    "The v6 source policy must occur exactly once and cannot be combined with another paper-native policy.",
                    packet.packet_id,
                    "ma2.primary_evidence",
                )
            ]
        rows: list[dict[str, object]] = []
        candidate_yield = packet.metadata.get("candidate_yield")
        candidate_backfill = packet.metadata.get("candidate_backfill")
        requested = (
            candidate_yield.get("requested_eligible_count")
            if isinstance(candidate_yield, Mapping)
            else None
        )
        expected_requested = (
            plan.candidates_per_route if plan is not None else None
        )
        if (
            type(expected_requested) is not int
            or requested != expected_requested
            or type(requested) is not int
            or requested < 1
            or not isinstance(candidate_backfill, Mapping)
            or candidate_backfill.get("requested_eligible_count") != requested
        ):
            return [
                self._finding(
                    GateName.G1, "evidence_provenance", Severity.CRITICAL,
                    "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_GROUP_PROJECTION_INVALID",
                    "The packet does not carry one consistent requested eligible quota.",
                    packet.packet_id, "ma2.primary_evidence",
                )
            ]
        for candidate in packet.candidates:
            disposition_valid, is_reserved = _paper_v6_reserved_disposition_valid(
                candidate, brief.constraints.reserved_control_compounds
            )
            if not disposition_valid:
                return [
                    self._finding(
                        GateName.G1, "safety", Severity.CRITICAL,
                        "RESERVED_CONTROL_CANDIDATE_NOT_EXCLUDED",
                        "MA3 independently matched a reserved control whose exclusion, rank, compute, or policy projection is inconsistent.",
                        candidate.candidate_id, "ma2.primary_evidence",
                    )
                ]
            if is_reserved and candidate.status != "excluded":
                return [
                    self._finding(
                        GateName.G1, "safety", Severity.CRITICAL,
                        "RESERVED_CONTROL_CANDIDATE_NOT_EXCLUDED",
                        "A reserved control was counted as an eligible v6 candidate.",
                        candidate.candidate_id, "ma2.primary_evidence",
                    )
                ]
            row = _paper_v6_candidate_projection(
                candidate, expected_direction=_required_candidate_direction(brief, candidate, hypotheses)
            )
            if row is None or row["candidate_id"] in {item["candidate_id"] for item in rows}:
                return [
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_ATTESTATION_INVALID",
                        "MA3 could not independently reproduce a candidate's v6 evidence, activity, envelope, identity, or canonical hashes.",
                        candidate.candidate_id,
                        "ma2.primary_evidence",
                    )
                ]
            rows.append(row)
        if not _paper_v6_group_projection_valid(
            list(packet.candidates), rows,
            packet_metadata=packet.metadata,
            requested_eligible_count=requested,
            reserved_controls=brief.constraints.reserved_control_compounds,
        ):
            return [
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.CRITICAL,
                    "CANDIDATE_PRIMARY_PAPER_NATIVE_V6_GROUP_PROJECTION_INVALID",
                    "MA3 could not independently reproduce the v6 group order, group hash, or terminal projection.",
                    packet.packet_id,
                    "ma2.primary_evidence",
                )
            ]
        # V6 adds group semantics; it does not replace the generic primary
        # evidence audit (batch/query/source/support/full-text/relevance).
        return self._audit_candidate_primary_sources(brief, packet)

    def _audit_primary_abstract_paper_native_v5(
        self,
        brief: ResearchBrief,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Independently verify the paper-native v5 comparative group.

        No producer parser or search helper is imported or called.  MA3
        rebuilds the frozen pool, ChEMBL support snapshot, PubMed query/call
        ledger, source grammar, spans, hashes and indivisible group membership
        from the final immutable CandidatePacket.
        """

        if (
            not brief.data_policy.require_primary_sources
            or packet.backend_kind is not ScientificBackendKind.LIVE
        ):
            return []
        if (
            brief.human_priorities.count(_PAPER_NATIVE_V5_REQUIRED_PRIORITY) != 1
            or _PAIRED_V4_REQUIRED_PRIORITY in brief.human_priorities
        ):
            return [
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.HIGH,
                    "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_TOKEN_INVALID",
                    (
                        "The paper-native-v5 source policy must occur exactly "
                        "once and cannot be combined with the paired-v4 policy."
                    ),
                    packet.packet_id,
                    "ma2.primary_evidence",
                )
            ]

        reserved_ids = {
            item.identifier.casefold()
            for control in brief.constraints.reserved_control_compounds
            for item in control.registry_ids
            if item.namespace.casefold() == "chembl"
        }
        all_by_molecule: dict[str, object] = {}
        valid_by_molecule: dict[str, dict[str, object]] = {}
        eligible_missing: list[str] = []
        malformed_artifact_id: str | None = None
        projection_artifact_id: str | None = None

        for candidate in packet.candidates:
            molecule_id = _paired_v4_candidate_chembl_id(candidate)
            if molecule_id is not None:
                if molecule_id in all_by_molecule:
                    malformed_artifact_id = candidate.candidate_id
                all_by_molecule[molecule_id] = candidate
            reserved = screen_reserved_control(
                candidate_name=candidate.name,
                metadata=candidate.metadata,
                reserved_controls=brief.constraints.reserved_control_compounds,
            )
            is_reserved = reserved.excluded or bool(
                molecule_id and molecule_id.casefold() in reserved_ids
            )
            if is_reserved and not (
                candidate.status == "excluded"
                and candidate.score == 0.0
                and candidate.rank is None
                and candidate.target_rank is None
                and candidate.portfolio_rank is None
                and candidate.portfolio_score is None
                and candidate.metadata.get("policy_eligible") is False
            ):
                return [
                    self._finding(
                        GateName.G1,
                        "safety",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_CANDIDATE_NOT_EXCLUDED",
                        (
                            "A reserved assay control in the paper-native-v5 "
                            "group retained candidate eligibility or rank."
                        ),
                        candidate.candidate_id,
                        "ma2.primary_evidence",
                    )
                ]
            if not _paper_v5_no_pubmed_numeric_projection(candidate):
                projection_artifact_id = candidate.candidate_id

            candidate_copy = candidate.metadata.get(
                _PAPER_NATIVE_V5_ENVELOPE_KEY
            )
            evidence_entries: list[tuple[object, object, object]] = []
            for evidence in candidate.evidence:
                envelope = evidence.metadata.get(
                    _PAPER_NATIVE_V5_ENVELOPE_KEY
                )
                if envelope is None:
                    continue
                linked = [
                    activity
                    for activity in candidate.activities
                    if activity.source_evidence_id == evidence.evidence_id
                ]
                if len(linked) != 1:
                    if (
                        isinstance(envelope, dict)
                        and envelope.get("verification_status") == "verified"
                    ):
                        malformed_artifact_id = candidate.candidate_id
                    continue
                evidence_entries.append((evidence, linked[0], envelope))
            claimed_verified = bool(
                isinstance(candidate_copy, dict)
                and candidate_copy.get("verification_status") == "verified"
            ) or any(
                isinstance(envelope, dict)
                and envelope.get("verification_status") == "verified"
                for _evidence, _activity, envelope in evidence_entries
            )
            if len(evidence_entries) != 1:
                if claimed_verified:
                    malformed_artifact_id = candidate.candidate_id
                elif not is_reserved and candidate.status != "excluded":
                    eligible_missing.append(candidate.candidate_id)
                continue
            evidence, activity, envelope = evidence_entries[0]
            if not claimed_verified:
                if not is_reserved and candidate.status != "excluded":
                    eligible_missing.append(candidate.candidate_id)
                continue
            if candidate_copy != envelope:
                malformed_artifact_id = candidate.candidate_id
                continue
            verified = _paper_v5_verify_candidate_envelope(
                candidate=candidate,
                evidence=evidence,
                activity=activity,
                envelope=envelope,
            )
            if verified is None:
                malformed_artifact_id = candidate.candidate_id
                continue
            verified_id = str(verified["molecule_id"])
            if verified_id in valid_by_molecule:
                malformed_artifact_id = candidate.candidate_id
                continue
            valid_by_molecule[verified_id] = verified

        if projection_artifact_id is not None:
            return [
                self._finding(
                    GateName.G1,
                    "model_appropriateness",
                    Severity.CRITICAL,
                    "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_VALUE_LEAKAGE",
                    (
                        "A PubMed paper-native measurement was projected into "
                        "ActivityMeasurement or candidate ranking metadata."
                    ),
                    projection_artifact_id,
                    "ma2.ranking",
                )
            ]
        if malformed_artifact_id is not None:
            return [
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.CRITICAL,
                    "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_ATTESTATION_INVALID",
                    (
                        "A producer-verified paper-native-v5 envelope failed "
                        "MA3's independent fixed-pool, ChEMBL support, query "
                        "ledger, publication-type, grammar, span or hash audit. "
                        "The claimed source group is indivisible, so this "
                        "finding invalidates the entire group for G1 rather "
                        "than permitting partial member salvage."
                    ),
                    malformed_artifact_id,
                    "ma2.primary_evidence",
                )
            ]

        if valid_by_molecule:
            shared_fields = (
                "fixed_ids",
                "direct_ids",
                "group_ids",
                "group_sha256",
                "group_canonical_sha256",
                "search_outcome_sha256",
                "search_canonical_sha256",
                "alias_inventory_sha256",
                "support_inventory_sha256",
                "selection_order_sha256",
                "promotion_sha256",
            )
            if any(
                len(
                    {
                        _paper_v5_hash(value[field])
                        for value in valid_by_molecule.values()
                    }
                )
                != 1
                for field in shared_fields
            ) or len(
                {
                    str(value["group_canonical_json"])
                    for value in valid_by_molecule.values()
                }
            ) != 1 or len(
                {
                    tuple(value["group_ids"])  # type: ignore[arg-type]
                    for value in valid_by_molecule.values()
                }
            ) != 1:
                return [
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_GROUP_COPY_MISMATCH",
                        (
                            "Candidate envelopes do not preserve one identical "
                            "paper-native group, search outcome, fixed pool and "
                            "pre-PubMed support selection snapshot."
                        ),
                        packet.packet_id,
                        "ma2.primary_evidence",
                    )
                ]
            exemplar = next(iter(valid_by_molecule.values()))
            group_ids = list(exemplar["group_ids"])
            if (
                len(group_ids) < 3
                or not set(valid_by_molecule).issubset(set(group_ids))
                or not set(group_ids).issubset(set(all_by_molecule))
            ):
                return [
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_GROUP_INCOMPLETE",
                        (
                            "The selected paper-native group is indivisible: "
                            "every declared member must be returned with the "
                            "same independently verified group envelope."
                        ),
                        packet.packet_id,
                        "ma2.primary_evidence",
                    )
                ]
            for molecule_id in set(group_ids).difference(valid_by_molecule):
                missing_candidate = all_by_molecule.get(molecule_id)
                missing_id = getattr(missing_candidate, "candidate_id", None)
                if isinstance(missing_id, str):
                    eligible_missing.append(missing_id)
            fixed_ids = list(exemplar["fixed_ids"])
            if not _paper_v5_group_projection(
                packet,
                declared_group_ids=group_ids,
                fixed_ids=fixed_ids,
            ):
                return [
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_GROUP_PROJECTION_INVALID",
                        (
                            "MA2 did not preserve the exact finite fixed-pool "
                            "prefix through both the eligible quota and every "
                            "declared paper-native group member."
                        ),
                        packet.packet_id,
                        "ma2.candidate_projection",
                    )
                ]

        noncontrol_verified = {
            molecule_id
            for molecule_id in valid_by_molecule
            if molecule_id.casefold() not in reserved_ids
        }
        if eligible_missing or len(noncontrol_verified) < 3:
            return [
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.HIGH,
                    "CANDIDATE_PRIMARY_PAPER_NATIVE_V5_INSUFFICIENT",
                    (
                        "The exact paper-native-v5 policy requires an "
                        "indivisible group of at least three noncontrol "
                        "candidates, each with independently verified pre-"
                        "PubMed ChEMBL direct support."
                    ),
                    packet.packet_id,
                    "ma2.primary_evidence",
                )
            ]
        return []

    def _audit_primary_abstract_paired_v4(
        self,
        brief: ResearchBrief,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Independently verify the bounded comparative primary envelope.

        The producer parser/search modules are deliberately not called here.
        Every source span, hash, PubMed call, candidate/value binding, stable
        identity, exact ChEMBL tuple and pre-PubMed cohort decision is rebuilt
        from the immutable packet.
        """

        if (
            not brief.data_policy.require_primary_sources
            or packet.backend_kind is not ScientificBackendKind.LIVE
        ):
            return []

        malformed_artifact_id: str | None = None
        missing_candidate_ids: list[str] = []
        valid_by_molecule: dict[str, dict[str, object]] = {}
        eligible_by_molecule: dict[str, object] = {}
        reserved_ids = {
            item.identifier.casefold()
            for control in brief.constraints.reserved_control_compounds
            for item in control.registry_ids
            if item.namespace.casefold() == "chembl"
        }

        for candidate in packet.candidates:
            molecule_id = _paired_v4_candidate_chembl_id(candidate)
            reserved = screen_reserved_control(
                candidate_name=candidate.name,
                metadata=candidate.metadata,
                reserved_controls=brief.constraints.reserved_control_compounds,
            )
            is_reserved = reserved.excluded or bool(
                molecule_id and molecule_id.casefold() in reserved_ids
            )
            if is_reserved and not (
                candidate.status == "excluded"
                and candidate.score == 0.0
                and candidate.rank is None
                and candidate.target_rank is None
                and candidate.portfolio_rank is None
                and candidate.portfolio_score is None
                and candidate.metadata.get("policy_eligible") is False
            ):
                return [
                    self._finding(
                        GateName.G1,
                        "safety",
                        Severity.CRITICAL,
                        "RESERVED_CONTROL_CANDIDATE_NOT_EXCLUDED",
                        (
                            "A reserved assay control in the paired-v4 source "
                            "group retained candidate eligibility or rank."
                        ),
                        candidate.candidate_id,
                        "ma2.primary_evidence",
                    )
                ]
            if (
                not is_reserved
                and candidate.status != "excluded"
                and molecule_id is not None
            ):
                if molecule_id in eligible_by_molecule:
                    malformed_artifact_id = candidate.candidate_id
                eligible_by_molecule[molecule_id] = candidate

            candidate_copy = candidate.metadata.get(
                _PAIRED_V4_ENVELOPE_KEY
            )
            evidence_entries: list[tuple[object, object, object]] = []
            for evidence in candidate.evidence:
                envelope = evidence.metadata.get(_PAIRED_V4_ENVELOPE_KEY)
                if envelope is None:
                    continue
                linked = [
                    activity
                    for activity in candidate.activities
                    if activity.source_evidence_id == evidence.evidence_id
                ]
                if len(linked) != 1:
                    if (
                        isinstance(envelope, dict)
                        and envelope.get("verification_status") == "verified"
                    ):
                        malformed_artifact_id = candidate.candidate_id
                    continue
                evidence_entries.append((evidence, linked[0], envelope))

            claimed_verified = bool(
                isinstance(candidate_copy, dict)
                and candidate_copy.get("verification_status") == "verified"
            ) or any(
                isinstance(envelope, dict)
                and envelope.get("verification_status") == "verified"
                for _evidence, _activity, envelope in evidence_entries
            )
            if len(evidence_entries) > 1:
                if claimed_verified:
                    malformed_artifact_id = candidate.candidate_id
                continue
            if not evidence_entries:
                if claimed_verified:
                    malformed_artifact_id = candidate.candidate_id
                elif not is_reserved and candidate.status != "excluded":
                    missing_candidate_ids.append(candidate.candidate_id)
                continue
            evidence, activity, envelope = evidence_entries[0]
            if not (
                isinstance(envelope, dict)
                and envelope.get("verification_status") == "verified"
            ):
                if claimed_verified:
                    malformed_artifact_id = candidate.candidate_id
                elif not is_reserved and candidate.status != "excluded":
                    missing_candidate_ids.append(candidate.candidate_id)
                continue
            verified = _paired_v4_verify_candidate_envelope(
                candidate=candidate,
                evidence=evidence,
                activity=activity,
                envelope=envelope,
            )
            if verified is None:
                malformed_artifact_id = candidate.candidate_id
                continue
            verified_molecule_id = str(verified["molecule_id"])
            if verified_molecule_id in valid_by_molecule:
                malformed_artifact_id = candidate.candidate_id
                continue
            valid_by_molecule[verified_molecule_id] = verified

        if malformed_artifact_id is not None:
            return [
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.CRITICAL,
                    "CANDIDATE_PRIMARY_PAIRED_V4_ATTESTATION_INVALID",
                    (
                        "A producer-verified paired-v4 envelope failed MA3's "
                        "independent canonical hash, source-span, query-ledger, "
                        "candidate/value, exact-activity, or assay-cohort audit."
                    ),
                    malformed_artifact_id,
                    "ma2.primary_evidence",
                )
            ]

        eligible_verified = {
            molecule_id: valid_by_molecule[molecule_id]
            for molecule_id in eligible_by_molecule
            if molecule_id in valid_by_molecule
        }
        if valid_by_molecule:
            group_hashes = {
                str(item["group_sha256"])
                for item in valid_by_molecule.values()
            }
            group_copies = {
                str(item["group_canonical_sha256"])
                for item in valid_by_molecule.values()
            }
            search_copies = {
                str(item["search_canonical_sha256"])
                for item in valid_by_molecule.values()
            }
            inventory_hashes = {
                str(item["inventory_sha256"])
                for item in valid_by_molecule.values()
            }
            signature_hashes = {
                str(item["signature_sha256"])
                for item in valid_by_molecule.values()
            }
            cohort_lists = {
                tuple(str(value) for value in item["cohort_ids"])
                for item in valid_by_molecule.values()
            }
            gateway_return_lists = {
                tuple(str(value) for value in item["gateway_return_ids"])
                for item in valid_by_molecule.values()
            }
            if any(
                len(values) != 1
                for values in (
                    group_hashes,
                    group_copies,
                    search_copies,
                    inventory_hashes,
                    signature_hashes,
                    cohort_lists,
                    gateway_return_lists,
                )
            ):
                return [
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "CANDIDATE_PRIMARY_PAIRED_V4_GROUP_COPY_MISMATCH",
                        (
                            "Candidate envelopes do not preserve one identical "
                            "comparative group, search result, frozen cohort and "
                            "exact ChEMBL assay signature."
                        ),
                        packet.packet_id,
                        "ma2.primary_evidence",
                    )
                ]

        noncontrol_verified = {
            molecule_id
            for molecule_id, verified in eligible_verified.items()
            if molecule_id.casefold() not in reserved_ids
            and molecule_id in set(verified["cohort_ids"])
        }
        if (
            missing_candidate_ids
            or len(noncontrol_verified)
            < _PAIRED_V4_MIN_NONCONTROL_CANDIDATES
        ):
            return [
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.HIGH,
                    "CANDIDATE_PRIMARY_PAIRED_V4_INSUFFICIENT",
                    (
                        "The exact paired-v4 policy requires at least three "
                        "distinct, nonexcluded, noncontrol candidates in one "
                        "independently verified primary sentence and exact "
                        "pre-PubMed ChEMBL assay cohort. Reserved controls and "
                        "missing or honest not-verified envelopes do not count."
                    ),
                    packet.packet_id,
                    "ma2.primary_evidence",
                )
            ]
        return []

    def _audit_candidate_primary_sources(
        self,
        brief: ResearchBrief,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Require one exact, locally verifiable primary record per live candidate.

        Fixture packets retain their established demo/test contract.  This
        policy check applies to live evidence only, where an eligible candidate
        could otherwise be promoted from a ChEMBL title or document identifier
        without an auditable primary passage.
        """

        if (
            not brief.data_policy.require_primary_sources
            or packet.backend_kind is not ScientificBackendKind.LIVE
        ):
            return []
        findings: list[AuditFinding] = []
        for candidate in packet.candidates:
            if candidate.status == "excluded":
                continue
            verified_ids = self._verified_candidate_primary_evidence_ids(
                candidate,
                pool_candidates=packet.candidates,
            )
            if verified_ids:
                continue
            findings.append(
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.HIGH,
                    "CANDIDATE_PRIMARY_SOURCE_UNVERIFIED",
                    (
                        "Eligible live candidate lacks a verified primary-publication "
                        "excerpt bound to its exact candidate, target, activity, and "
                        "ChEMBL document identifiers."
                    ),
                    candidate.candidate_id,
                    "ma2.primary_evidence",
                )
            )
        return findings

    @classmethod
    def _verified_candidate_primary_evidence_ids(
        cls,
        candidate,
        *,
        pool_candidates: Iterable[object] = (),
    ) -> list[str]:
        evidence_by_id = {
            evidence.evidence_id: evidence
            for evidence in candidate.evidence
        }
        verified: list[str] = []
        for activity in candidate.activities:
            evidence_id = activity.source_evidence_id
            if not evidence_id:
                continue
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            if cls._is_verified_candidate_primary_evidence(
                candidate,
                activity,
                evidence,
                pool_candidates=pool_candidates,
            ):
                verified.append(evidence_id)
        return list(dict.fromkeys(verified))

    @classmethod
    def _is_verified_candidate_primary_evidence(
        cls,
        candidate,
        activity,
        evidence,
        *,
        pool_candidates: Iterable[object] = (),
    ) -> bool:
        if evidence.source != "ChEMBL" or evidence.supports != "support":
            return False
        metadata = evidence.metadata
        primary = metadata.get("primary_source")
        if not isinstance(primary, dict):
            return False
        if (
            primary.get("schema_version")
            != _CANDIDATE_PRIMARY_EVIDENCE_SCHEMA
            or primary.get("verification_status") != "verified"
            or primary.get("retrieval_status") != "completed"
            or primary.get("is_primary_support") is not True
            or primary.get("full_text_stored") is not False
            or primary.get("retrieved_field") != "abstract"
            or str(primary.get("doc_type") or "").casefold()
            != "publication"
        ):
            return False
        if cls._contains_full_primary_text(primary):
            return False
        publication_types = primary.get("publication_types")
        chembl_publication_types = primary.get(
            "chembl_reported_publication_types"
        )
        if (
            not isinstance(publication_types, list)
            or not isinstance(chembl_publication_types, list)
            or any(not isinstance(item, str) for item in publication_types)
            or any(
                not isinstance(item, str)
                for item in chembl_publication_types
            )
        ):
            return False
        review_markers = [
            primary.get("title") or "",
            *publication_types,
            *chembl_publication_types,
        ]
        if any(
            re.search(
                r"(?<!\w)(?:review|systematic review|meta analysis|guideline|editorial|letter|comment)(?!\w)",
                _normalize_primary_phrase(value),
            )
            for value in review_markers
        ):
            return False
        pmid = str(primary.get("pmid") or "").strip()
        if not pmid.isdigit():
            return False
        pubmed_type = primary.get("pubmed_type_verification")
        if not isinstance(pubmed_type, dict):
            return False
        independently_verified_type, excluded_types = (
            _primary_pubmed_type_decision(publication_types)
        )
        batch = pubmed_type.get("batch")
        query = pubmed_type.get("query")
        if not isinstance(batch, dict) or not isinstance(query, dict):
            return False
        requested_pmids = batch.get("requested_pmids")
        observed_uids = batch.get("observed_result_uids")
        batch_index = batch.get("batch_index")
        batch_count = batch.get("batch_count")
        global_eligible_count = batch.get("global_eligible_pmid_count")
        if (
            pubmed_type.get("schema_version") != _PUBMED_TYPE_SCHEMA
            or pubmed_type.get("verification_status") != "verified"
            or pubmed_type.get("retrieval_status") != "completed"
            or pubmed_type.get("reason_code") is not None
            or pubmed_type.get("pmid") != pmid
            or pubmed_type.get("publication_types") != publication_types
            or pubmed_type.get("required_publication_type")
            != "Journal Article"
            or pubmed_type.get("excluded_publication_types")
            != excluded_types
            or pubmed_type.get("type_policy") != _PUBMED_TYPE_POLICY
            or pubmed_type.get("fixture") is not False
            or not independently_verified_type
            or batch.get("batch_cap") != _PRIMARY_PUBMED_BATCH_MAX_IDS
            or type(batch_index) is not int
            or type(batch_count) is not int
            or type(global_eligible_count) is not int
            or not 1 <= batch_index <= batch_count
            or global_eligible_count < 1
            or batch_count
            != (
                global_eligible_count + _PRIMARY_PUBMED_BATCH_MAX_IDS - 1
            )
            // _PRIMARY_PUBMED_BATCH_MAX_IDS
            or batch.get("single_batch") is not (batch_count == 1)
            or batch.get("truncated") is not False
            or not isinstance(requested_pmids, list)
            or not requested_pmids
            or len(requested_pmids) > _PRIMARY_PUBMED_BATCH_MAX_IDS
            or batch.get("requested_count") != len(requested_pmids)
            or batch.get("eligible_pmid_count") != len(requested_pmids)
            or len(requested_pmids) != len(set(requested_pmids))
            or any(
                not isinstance(item, str) or not item.isdigit()
                for item in requested_pmids
            )
            or requested_pmids
            != sorted(
                requested_pmids,
                key=lambda value: (int(value), value),
            )
            or pmid not in requested_pmids
            or not isinstance(observed_uids, list)
            or len(observed_uids) != len(set(observed_uids))
            or any(
                not isinstance(item, str) or not item.isdigit()
                for item in observed_uids
            )
            or not set(observed_uids).issubset(set(requested_pmids))
            or pmid not in observed_uids
            or query.get("source") != "PubMed"
            or query.get("endpoint") != _PUBMED_SUMMARY_ENDPOINT
            or query.get("status_code") != 200
            or query.get("parameter_keys")
            != ["db", "id", "retmode", "version"]
            or not _SHA256.fullmatch(
                str(query.get("response_sha256") or "")
            )
            or not _SHA256.fullmatch(
                str(query.get("parameter_hash") or "")
            )
        ):
            return False
        expected_batch_size = min(
            _PRIMARY_PUBMED_BATCH_MAX_IDS,
            global_eligible_count
            - (batch_index - 1) * _PRIMARY_PUBMED_BATCH_MAX_IDS,
        )
        if len(requested_pmids) != expected_batch_size:
            return False
        expected_pubmed_params = {
            "db": "pubmed",
            "id": ",".join(requested_pmids),
            "retmode": "json",
            "version": "2.0",
        }
        expected_parameter_hash = sha256(
            json.dumps(
                expected_pubmed_params,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if query.get("parameter_hash") != expected_parameter_hash:
            return False
        document_id = str(primary.get("document_chembl_id") or "").strip()
        if (
            not document_id
            or _CHEMBL_STABLE_ID.fullmatch(document_id) is None
            or document_id
            != str(metadata.get("document_chembl_id") or "").strip()
        ):
            return False

        excerpt = primary.get("excerpt")
        excerpt_hash = str(primary.get("excerpt_hash") or "")
        content_hash = str(primary.get("content_hash") or "")
        if (
            not isinstance(excerpt, str)
            or not excerpt.strip()
            or excerpt != excerpt.strip()
            or len(excerpt) > _PRIMARY_EXCERPT_MAX_CHARS
            or not _SHA256.fullmatch(excerpt_hash)
            or not _SHA256.fullmatch(content_hash)
            or sha256(excerpt.encode("utf-8")).hexdigest() != excerpt_hash
            or evidence.excerpt_hash != excerpt_hash
        ):
            return False
        location = primary.get("excerpt_location")
        if not isinstance(location, dict):
            return False
        start = location.get("normalized_char_start")
        end = location.get("normalized_char_end")
        sentence_index = location.get("sentence_index")
        clause_index = location.get("clause_index")
        parenthetical_scope = location.get("selection_scope") == (
            "atomic_parenthetical_with_sentence_context"
        )
        if (
            location.get("field") != "abstract"
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or isinstance(sentence_index, bool)
            or not isinstance(sentence_index, int)
            or start < 0
            or sentence_index < 0
            or end - start != len(excerpt)
            or (
                parenthetical_scope
                and (
                    set(location) != {
                        "field", "sentence_index", "clause_index",
                        "normalized_char_start", "normalized_char_end",
                        "selection_scope",
                    }
                    or clause_index is not None
                )
            )
            or (
                not parenthetical_scope
                and (
                    set(location) != {
                        "field", "sentence_index", "clause_index",
                        "normalized_char_start", "normalized_char_end",
                    }
                    or isinstance(clause_index, bool)
                    or not isinstance(clause_index, int)
                    or clause_index < 0
                )
            )
        ):
            return False
        relevance = primary.get("relevance")
        if not isinstance(relevance, dict):
            return False
        groups = relevance.get("matched_groups")
        reported_terms = relevance.get("matched_terms")
        reported_value_units = relevance.get("matched_exact_value_unit")
        if (
            relevance.get("policy") != _PRIMARY_RELEVANCE_POLICY
            or not isinstance(groups, list)
            or len(groups) != 3
            or set(groups) != {"candidate", "target", "activity"}
            or not isinstance(reported_terms, dict)
            or not isinstance(reported_value_units, list)
            or not reported_value_units
            or relevance.get("competitor_alias_matches") != []
            or relevance.get("excerpt_truncated") is not False
        ):
            return False

        primary_aliases = primary.get("candidate_exact_aliases")
        candidate_aliases = candidate.metadata.get("chembl_exact_aliases")
        if (
            primary.get("candidate_alias_source") != _CANDIDATE_ALIAS_SOURCE
            or candidate.metadata.get("chembl_alias_source")
            != _CANDIDATE_ALIAS_SOURCE
            or not isinstance(primary_aliases, list)
            or not isinstance(candidate_aliases, list)
            or not primary_aliases
            or len(primary_aliases) > _MAX_EXACT_CANDIDATE_ALIASES
            or len(candidate_aliases) > _MAX_EXACT_CANDIDATE_ALIASES
        ):
            return False
        normalized_primary_aliases = {
            _normalize_primary_phrase(alias)
            for alias in primary_aliases
            if _normalize_primary_phrase(alias)
        }
        normalized_candidate_aliases = {
            _normalize_primary_phrase(alias)
            for alias in candidate_aliases
            if _normalize_primary_phrase(alias)
        }
        if (
            len(normalized_primary_aliases) != len(primary_aliases)
            or len(normalized_candidate_aliases) != len(candidate_aliases)
            or normalized_primary_aliases != normalized_candidate_aliases
            or _normalize_primary_phrase(candidate.name)
            not in normalized_candidate_aliases
        ):
            return False
        primary_conflicts = primary.get("candidate_alias_conflicts")
        candidate_conflicts = candidate.metadata.get("chembl_alias_conflicts")
        if primary_conflicts != [] or candidate_conflicts != []:
            return False
        molecule_id = str(metadata.get("molecule_chembl_id") or "").strip()
        competitor_aliases = _validated_primary_competitor_aliases(
            primary.get("competitor_alias_provenance"),
            current_molecule_id=molecule_id,
            current_aliases=primary_aliases,
            pool_candidates=pool_candidates,
            target_chembl_id=str(
                candidate.target.metadata.get("chembl_target_id") or ""
            ),
        )
        if competitor_aliases is None:
            return False
        recomputed_binding = None
        source_location = primary.get("excerpt_location")
        if (
            isinstance(source_location, Mapping)
            and source_location.get("selection_scope")
            == "atomic_parenthetical_with_sentence_context"
        ):
            recomputed_binding = _primary_parenthetical_binding(
                excerpt,
                candidate_aliases=primary_aliases,
                competitor_aliases=competitor_aliases,
                target=candidate.target,
                activity=activity,
            )
        if recomputed_binding is None:
            recomputed_binding = _primary_clause_binding(
                excerpt,
                candidate_aliases=primary_aliases,
                competitor_aliases=competitor_aliases,
                target=candidate.target,
                activity=activity,
            )
        if (
            recomputed_binding is None
            or groups != recomputed_binding.get("matched_groups")
            or reported_terms != recomputed_binding.get("matched_terms")
            or reported_value_units
            != recomputed_binding.get("matched_exact_value_unit")
            or relevance.get("binding_pattern")
            != recomputed_binding.get("binding_pattern")
            or relevance.get("bound_spans")
            != recomputed_binding.get("bound_spans")
            or relevance.get("competitor_alias_matches")
            != recomputed_binding.get("competitor_alias_matches")
        ):
            return False

        query = primary.get("document_query")
        document_batch = primary.get("document_query_batch")
        batch_body = (
            dict(document_batch)
            if isinstance(document_batch, Mapping)
            else {}
        )
        batch_attestation = batch_body.pop("attestation_sha256", None)
        batch_ids = batch_body.get("document_chembl_ids")
        batch_parameters = batch_body.get("query_parameters")
        expected_document_parameter_hash = (
            sha256(
                json.dumps(
                    batch_parameters,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if isinstance(batch_parameters, Mapping)
            else None
        )
        if (
            not isinstance(query, dict)
            or query.get("source") != "ChEMBL"
            or query.get("endpoint") != _PRIMARY_CHEMBL_DOCUMENT_ENDPOINT
            or query.get("status_code") != 200
            or not _SHA256.fullmatch(str(query.get("response_sha256") or ""))
            or query.get("parameter_keys")
            != ["document_chembl_id__in", "limit"]
            or not isinstance(document_batch, Mapping)
            or batch_body.get("schema_version")
            != "chembl_document_query_batch_v1"
            or not isinstance(batch_ids, list)
            or not batch_ids
            or len(batch_ids) > _PRIMARY_DOCUMENT_BATCH_SIZE
            or batch_ids != sorted(batch_ids)
            or len(batch_ids) != len(set(batch_ids))
            or any(not isinstance(item, str) or not item for item in batch_ids)
            or any(_CHEMBL_STABLE_ID.fullmatch(item) is None for item in batch_ids)
            or batch_body.get("document_count") != len(batch_ids)
            or batch_body.get("batch_cap") != _PRIMARY_DOCUMENT_BATCH_SIZE
            or batch_body.get("selected_document_chembl_id") != document_id
            or batch_body.get("selected_document_in_batch") is not True
            or document_id not in batch_ids
            or batch_parameters
            != {
                "document_chembl_id__in": ",".join(batch_ids),
                "limit": len(batch_ids),
            }
            or batch_body.get("query_parameter_hash")
            != expected_document_parameter_hash
            or batch_body.get("call_parameter_hash")
            != expected_document_parameter_hash
            or query.get("parameter_hash")
            != expected_document_parameter_hash
            or batch_body.get("truncated") is not False
            or batch_body.get("fixture") is not False
            or batch_attestation != _paper_v6_hash(batch_body)
        ):
            return False
        binding = primary.get("activity_binding")
        if not isinstance(binding, dict):
            return False
        deposited = binding.get("deposited_activity")
        raw_standard_type = metadata.get("standard_type")
        raw_standard_relation = metadata.get("standard_relation")
        raw_standard_value = metadata.get("standard_value")
        raw_standard_units = metadata.get("standard_units")
        if (
            not isinstance(deposited, dict)
            or raw_standard_type != activity.endpoint
            or raw_standard_relation != activity.relation
            or raw_standard_units != activity.unit
            or raw_standard_type not in {"IC50", "Ki", "Kd", "EC50"}
            or raw_standard_relation not in {"=", "<", ">", "<=", ">="}
            or raw_standard_units not in _KNOWN_ACTIVITY_UNITS
            or isinstance(raw_standard_value, bool)
            or deposited
            != {
                "schema_version": "chembl_deposited_activity_exact_v1",
                "standard_type": raw_standard_type,
                "standard_relation": raw_standard_relation,
                "standard_value": raw_standard_value,
                "standard_units": raw_standard_units,
                "typed_exact_match": True,
                "missing_values_imputed": False,
                "normalization_applied": False,
                "unit_conversion_applied": False,
            }
        ):
            return False
        activity_id = str(evidence.external_id)
        if (
            str(metadata.get("activity_id") or "") != activity_id
            or str(binding.get("activity_id") or "") != activity_id
            or str(binding.get("document_chembl_id") or "") != document_id
            or binding.get("endpoint") != activity.endpoint
            or binding.get("relation") != activity.relation
            or binding.get("unit") != activity.unit
        ):
            return False
        try:
            typed_value = Decimal(str(activity.value))
            if (
                Decimal(str(binding.get("value"))) != typed_value
                or Decimal(str(raw_standard_value)) != typed_value
            ):
                return False
        except (InvalidOperation, TypeError, ValueError):
            return False

        v6_envelope = candidate.metadata.get(_PAPER_NATIVE_V6_ENVELOPE_KEY)
        if isinstance(v6_envelope, Mapping):
            assay_type = getattr(activity, "assay_type", None)
            assay_id = str(binding.get("assay_chembl_id") or "").strip()
            discovery_bounds = candidate.metadata.get(
                "candidate_discovery_bounds"
            )
            target_id = (
                discovery_bounds.get("target_chembl_id")
                if isinstance(discovery_bounds, Mapping)
                else None
            )
            assay_context, direct_engagement = _paper_v6_assay_engagement(
                assay_type=str(assay_type or ""),
                chembl_assay_context=metadata.get(
                    "chembl_assay_context"
                ),
                assay_chembl_id=assay_id,
                target_chembl_id=str(target_id or ""),
                target_symbol=str(candidate.target.symbol or ""),
            )
            mechanism_identity = candidate.metadata.get(
                "mechanism_identity"
            )
            direction_rule = metadata.get("direction_evidence_rule")
            target_symbol = str(candidate.target.symbol or "")
            expected_mechanism = (
                _paper_v6_mechanism_claim(
                    target_symbol=target_symbol,
                    assay_type=assay_type,
                    direction_rule=str(direction_rule or ""),
                    mechanism_identity_present=True,
                    assay_context=assay_context,
                )
                if isinstance(assay_type, str)
                else None
            )
            if (
                not isinstance(assay_type, str)
                or not assay_type
                or assay_context is None
                or binding.get("assay_type") != assay_type
                or binding.get("assay_context") != assay_context
                or binding.get("direct_single_protein_engagement")
                is not direct_engagement
                or binding.get("direction_evidence_rule")
                != "measured_ic50_endpoint"
                or direction_rule != "measured_ic50_endpoint"
                or metadata.get("assay_type") != assay_type
                or metadata.get("assay_context") != assay_context
                or metadata.get("direct_single_protein_engagement")
                is not direct_engagement
                or candidate.metadata.get("activity_assay_type")
                != assay_type
                or candidate.metadata.get("activity_assay_context")
                != assay_context
                or candidate.metadata.get(
                    "direct_single_protein_engagement"
                )
                is not direct_engagement
                or candidate.metadata.get("direction_requested")
                != "inhibit"
                or candidate.metadata.get("direction_evidence_rules")
                != ["measured_ic50_endpoint"]
                or metadata.get("intervention_direction_requested")
                != "inhibit"
                or metadata.get("mechanism_identity")
                != mechanism_identity
                or not isinstance(discovery_bounds, Mapping)
                or not _paper_v6_mechanism_identity_valid(
                    mechanism_identity,
                    bounds=discovery_bounds,
                    molecule_id=molecule_id,
                )
                or binding.get("mechanism_identity_sha256")
                != _paper_v6_hash(dict(mechanism_identity))
                or not isinstance(expected_mechanism, str)
                or binding.get("mechanism_claim") != expected_mechanism
                or binding.get("mechanism_claim_sha256")
                != _paper_v6_hash(expected_mechanism)
                or candidate.mechanism != expected_mechanism
            ):
                return False

        selection = primary.get("activity_document_selection")
        if not isinstance(selection, dict):
            return False
        selection_rows = selection.get("rows")
        inspected_count = selection.get("inspected_row_count")
        eligible_count = selection.get("eligible_row_count")
        selected_rank = selection.get("selected_rank")
        complete_v6_selection = (
            selection.get("selected_by_candidate_primary_record_v6") is True
        )
        retained_v6_activity_ids: list[str] | None = None
        if complete_v6_selection:
            selection_bounds = candidate.metadata.get(
                "candidate_discovery_bounds"
            )
            selection_counts = (
                selection_bounds.get("activity_counts_by_id")
                if isinstance(selection_bounds, Mapping)
                else None
            )
            selection_count = (
                selection_counts.get(molecule_id)
                if isinstance(selection_counts, Mapping)
                else None
            )
            retained_v6_activity_ids = _paper_v6_retained_activity_ids(
                selection_count
            )
        selection_rows_valid = isinstance(selection_rows, list)
        first_verified_rank: int | None = None
        selection_order: list[tuple[Decimal, int]] = []
        if isinstance(selection_rows, list):
            expected_row_keys = {
                "rank",
                "activity_id",
                "document_chembl_id",
                "endpoint",
                "pchembl_value",
                "verification_status",
                "reason_code",
            }
            for row_index, row in enumerate(selection_rows, start=1):
                if not isinstance(row, dict) or set(row) != expected_row_keys:
                    selection_rows_valid = False
                    break
                activity_id_text = str(row.get("activity_id") or "").strip()
                try:
                    activity_id_number = int(activity_id_text)
                except (TypeError, ValueError):
                    activity_id_number = -1
                raw_pchembl = row.get("pchembl_value")
                try:
                    pchembl = (
                        Decimal(str(raw_pchembl))
                        if type(raw_pchembl) in {int, float}
                        else Decimal("NaN")
                    )
                except (InvalidOperation, ValueError):
                    pchembl = Decimal("NaN")
                row_status = row.get("verification_status")
                row_reason = row.get("reason_code")
                row_document = row.get("document_chembl_id")
                if (
                    row.get("rank") != row_index
                    or activity_id_number < 1
                    or activity_id_text != str(activity_id_number)
                    or not pchembl.is_finite()
                    or pchembl < Decimal("5")
                    or row.get("endpoint") != activity.endpoint
                    or (
                        row_document is not None
                        and (
                            not isinstance(row_document, str)
                            or not row_document
                            or row_document != row_document.strip()
                            or _CHEMBL_STABLE_ID.fullmatch(row_document) is None
                        )
                    )
                    or row_status
                    not in {"verified", "unverified", "metadata_only"}
                    or (
                        row_status == "verified"
                        and row_reason is not None
                    )
                    or (
                        row_status != "verified"
                        and (
                            not isinstance(row_reason, str)
                            or not row_reason
                        )
                    )
                ):
                    selection_rows_valid = False
                    break
                if row_status == "verified" and first_verified_rank is None:
                    first_verified_rank = row_index
                selection_order.append((-pchembl, activity_id_number))
            if selection_order != sorted(selection_order):
                selection_rows_valid = False
        if (
            (
                selection.get("policy")
                != (
                    _PRIMARY_ACTIVITY_DOCUMENT_COMPLETE_V6_POLICY
                    if complete_v6_selection
                    else "bounded_top_k_activity_document_rows_v1"
                )
            )
            or (
                selection.get("top_k")
                != (
                    None
                    if complete_v6_selection
                    else _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
                )
            )
            or not isinstance(selection_rows, list)
            or not selection_rows
            or not selection_rows_valid
            or (
                not complete_v6_selection
                and len(selection_rows) > _PRIMARY_ACTIVITY_DOCUMENT_TOP_K
            )
            or isinstance(inspected_count, bool)
            or not isinstance(inspected_count, int)
            or inspected_count != len(selection_rows)
            or isinstance(eligible_count, bool)
            or not isinstance(eligible_count, int)
            or eligible_count < inspected_count
            or isinstance(selected_rank, bool)
            or not isinstance(selected_rank, int)
            or not 1 <= selected_rank <= inspected_count
            or selected_rank != first_verified_rank
            or selection.get("truncated") is not (eligible_count > inspected_count)
            or (
                complete_v6_selection
                and (
                    retained_v6_activity_ids is None
                    or [
                        str(row.get("activity_id") or "")
                        for row in selection_rows
                        if isinstance(row, Mapping)
                    ]
                    != retained_v6_activity_ids
                    or eligible_count != len(retained_v6_activity_ids)
                    or inspected_count != len(retained_v6_activity_ids)
                    or selection.get("result_row_cap") is not None
                    or inspected_count != eligible_count
                    or selection.get("all_eligible_rows_inspected") is not True
                    or selection.get(
                        "paper_native_v6_paired_search_or_selection_used"
                    )
                    is not False
                    or selection.get(
                        "selected_by_pre_pubmed_paired_v4_cohort"
                    )
                    is not False
                    or selection.get(
                        "paired_v4_selection_precedes_pubmed_search"
                    )
                    is not False
                )
            )
            or str(selection.get("selected_activity_id") or "") != activity_id
        ):
            return False
        for row_index, row in enumerate(selection_rows, start=1):
            if not isinstance(row, dict) or row.get("rank") != row_index:
                return False
        selected_row = selection_rows[selected_rank - 1]
        if (
            str(selected_row.get("activity_id") or "") != activity_id
            or str(selected_row.get("document_chembl_id") or "") != document_id
            or selected_row.get("endpoint") != activity.endpoint
            or selected_row.get("verification_status") != "verified"
            or selected_row.get("reason_code") is not None
        ):
            return False

        if (
            not molecule_id
            or str(binding.get("molecule_chembl_id") or "").strip()
            != molecule_id
        ):
            return False
        independent_identity = candidate_identity(
            candidate_name=candidate.name,
            metadata=candidate.metadata,
        )
        candidate_chembl_ids = {
            str(item.get("identifier") or "").casefold()
            for item in independent_identity.get("registry_ids", [])
            if isinstance(item, dict)
            and str(item.get("namespace") or "").casefold() == "chembl"
            and str(item.get("identifier") or "")
        }
        if candidate_chembl_ids != {molecule_id.casefold()}:
            return False
        if not cls._primary_evidence_matches_target(
            candidate.target,
            metadata,
            binding,
        ):
            return False
        return True

    @staticmethod
    def _primary_evidence_matches_target(
        target,
        metadata: dict,
        binding: dict,
    ) -> bool:
        target_uniprot = str(target.uniprot_id or "").strip().casefold()
        evidence_uniprot = {
            str(metadata.get(key) or "").strip().casefold()
            for key in (
                "target_accession",
                "target_uniprot_id",
                "uniprot_accession",
                "uniprot_id",
            )
            if str(metadata.get(key) or "").strip()
        }
        target_chembl = str(
            target.metadata.get("chembl_target_id") or ""
        ).strip().casefold()
        evidence_chembl = {
            str(metadata.get(key) or "").strip().casefold()
            for key in ("target_chembl_id", "chembl_target_id")
            if str(metadata.get(key) or "").strip()
        }
        bound_chembl = str(
            binding.get("target_chembl_id") or ""
        ).strip().casefold()
        if bound_chembl:
            evidence_chembl.add(bound_chembl)
        stable_match = False
        if target_uniprot:
            if evidence_uniprot != {target_uniprot}:
                return False
            stable_match = True
        elif evidence_uniprot:
            return False
        if target_chembl:
            if evidence_chembl != {target_chembl}:
                return False
            stable_match = True
        elif target_uniprot and evidence_chembl:
            # The target has no ChEMBL identifier to compare, but the exact
            # UniProt link above remains authoritative for the same record.
            stable_match = True
        if stable_match:
            return True
        target_symbol = str(target.symbol or "").strip().casefold()
        evidence_symbols = {
            str(metadata.get(key) or "").strip().casefold()
            for key in ("target_gene", "gene", "gene_symbol")
            if str(metadata.get(key) or "").strip()
        }
        return bool(target_symbol and evidence_symbols == {target_symbol})

    @staticmethod
    def _contains_full_primary_text(value: object) -> bool:
        forbidden_keys = {
            "abstract",
            "abstract_text",
            "full_abstract",
            "full_text",
            "fulltext",
        }
        if isinstance(value, dict):
            if any(str(key).casefold() in forbidden_keys for key in value):
                return True
            return any(
                MA3AssuranceAgent._contains_full_primary_text(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                MA3AssuranceAgent._contains_full_primary_text(item)
                for item in value
            )
        return False

    def _audit_ma2_execution_contract(
        self,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Verify typed readiness, backend boundary, versions and input lineage."""

        findings: list[AuditFinding] = []
        live = packet.backend_kind is ScientificBackendKind.LIVE
        readiness = packet.gateway_readiness
        if readiness is None:
            if live:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.HIGH,
                        "GATEWAY_READINESS_MISSING",
                        "Live MA2 did not persist its gateway capability snapshot.",
                        packet.packet_id,
                        "ma2.tool_dispatch",
                    )
                )
        else:
            if (
                readiness.backend_kind is not packet.backend_kind
                or readiness.fixture != (
                    packet.backend_kind is ScientificBackendKind.FIXTURE
                )
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "BACKEND_BOUNDARY_VIOLATION",
                        "Gateway readiness does not match CandidatePacket backend mode.",
                        packet.packet_id,
                        "ma2.tool_dispatch",
                    )
                )
            route_decision = packet.metadata.get("route_decision", {})
            selected_route = (
                route_decision.get("selected_route")
                if isinstance(route_decision, dict)
                else None
            )
            required_tools = {"target_normalizer"}
            if any(
                candidate.admet_execution is not None
                or candidate.status != "excluded"
                for candidate in packet.candidates
            ):
                required_tools.add("admet_predictor")
            selected_tool = {
                "repurposing": "repurposing_search",
                "de_novo": "de_novo_generator",
            }.get(selected_route)
            if selected_tool and not packet.portfolio_enabled:
                required_tools.add(selected_tool)
            unready = [
                capability
                for capability in readiness.capabilities
                if capability.tool_name in required_tools
                and capability.status.value != "ready"
            ]
            if unready:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.HIGH,
                        "GATEWAY_NOT_READY",
                        (
                            "Required MA2 capabilities were not ready: "
                            + ", ".join(
                                f"{item.tool_name}/{item.reason_code}"
                                for item in unready
                            )
                        ),
                        packet.packet_id,
                        "ma2.tool_dispatch",
                    )
                )

        calls_by_id = {call.call_id: call for call in packet.tool_calls}
        for call in packet.tool_calls:
            if (
                call.backend_kind is not packet.backend_kind
                or call.fixture
                != (packet.backend_kind is ScientificBackendKind.FIXTURE)
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "BACKEND_BOUNDARY_VIOLATION",
                        f"Tool trace {call.call_id} crosses the declared backend boundary.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if not _SHA256.fullmatch(call.input_hash) or not _SHA256.fullmatch(
                call.output_hash
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "EXECUTION_INPUT_HASH_MISMATCH",
                        "Tool trace input/output digest is not a valid SHA-256 value.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if call.provenance.get("input_hash") != call.input_hash:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "EXECUTION_INPUT_HASH_MISMATCH",
                        "Persisted tool provenance does not match the dispatched input hash.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if live and not call.source_version:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "model_appropriateness",
                        Severity.HIGH,
                        "LIVE_TOOL_VERSION_MISSING",
                        "A live tool trace has no source/model version.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )
            if (
                packet.execution_status is ScientificExecutionStatus.COMPLETED
                and call.execution_status
                in {
                    ScientificExecutionStatus.UNAVAILABLE,
                    ScientificExecutionStatus.FAILED,
                }
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "TOOL_RESULT_STATUS_MISMATCH",
                        "CandidatePacket is completed despite an unavailable/failed tool call.",
                        call.call_id,
                        "ma2.tool_dispatch",
                    )
                )

        for candidate in packet.candidates:
            if live and (
                candidate.metadata.get("fixture") is True
                or any(item.metadata.get("fixture") is True for item in candidate.evidence)
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "FIXTURE_RESULT_IN_LIVE_RUN",
                        "A live CandidatePacket contains fixture candidate/evidence provenance.",
                        candidate.candidate_id,
                        "ma2.evaluate",
                    )
                )
            for result, tool_name in (
                (candidate.chemistry_execution, "chemistry_validator"),
                (candidate.docking_execution, "docking"),
                (candidate.admet_execution, "admet_predictor"),
            ):
                if result is None:
                    if candidate.status != "excluded":
                        findings.append(
                            self._finding(
                                GateName.G1,
                                "execution",
                                Severity.HIGH,
                                "TYPED_EXECUTION_RESULT_MISSING",
                                f"Eligible candidate has no typed {tool_name} result.",
                                candidate.candidate_id,
                                "ma2.evaluate",
                            )
                        )
                    continue
                if (
                    result.backend_kind is not packet.backend_kind
                    or result.fixture
                    != (packet.backend_kind is ScientificBackendKind.FIXTURE)
                ):
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "execution",
                            Severity.CRITICAL,
                            "BACKEND_BOUNDARY_VIOLATION",
                            f"Typed {tool_name} result has the wrong backend boundary.",
                            candidate.candidate_id,
                            "ma2.evaluate",
                        )
                    )
                candidate_calls = [
                    calls_by_id[call_id]
                    for call_id in candidate.tool_trace_ids
                    if call_id in calls_by_id
                    and calls_by_id[call_id].tool_name == tool_name
                ]
                if candidate_calls and result.input_hash not in {
                    call.input_hash for call in candidate_calls
                }:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "evidence_provenance",
                            Severity.CRITICAL,
                            "EXECUTION_INPUT_HASH_MISMATCH",
                            f"Typed {tool_name} result is not linked to its dispatched input.",
                            candidate.candidate_id,
                            "ma2.evaluate",
                        )
                    )
                if live and result.status is ScientificExecutionStatus.COMPLETED:
                    if not result.backend_version:
                        findings.append(
                            self._finding(
                                GateName.G1,
                                "model_appropriateness",
                                Severity.HIGH,
                                "LIVE_TOOL_VERSION_MISSING",
                                f"Completed live {tool_name} result has no backend version.",
                                candidate.candidate_id,
                                "ma2.evaluate",
                            )
                        )
                    if not result.provenance:
                        findings.append(
                            self._finding(
                                GateName.G1,
                                "evidence_provenance",
                                Severity.HIGH,
                                "LIVE_RESULT_PROVENANCE_MISSING",
                                f"Completed live {tool_name} result has no provenance.",
                                candidate.candidate_id,
                                "ma2.evaluate",
                            )
                        )
        return findings

    @staticmethod
    def _has_live_direction_attestation(
        metadata: dict,
        required_direction: str,
    ) -> bool:
        """Require a target-scoped structured rule, never mechanism prose."""

        requested = str(metadata.get("direction_requested") or "").strip().casefold()
        required = required_direction.strip().casefold()
        raw_rules = metadata.get("direction_evidence_rules")
        if requested != required or isinstance(raw_rules, (str, bytes)):
            return False
        if not isinstance(raw_rules, (list, tuple, set, frozenset)):
            return False
        rules = {
            str(item).strip()
            for item in raw_rules
            if isinstance(item, str) and item.strip()
        }
        return bool(rules.intersection(_LIVE_DIRECTION_RULES.get(required, frozenset())))

    @staticmethod
    def _has_target_linked_measured_activity(candidate) -> bool:
        """Independently reproduce the strict target link for a dock waiver.

        Stable identifiers are authoritative.  A matching gene symbol cannot
        override a conflicting UniProt accession in the same evidence record.
        """

        evidence_by_id = {
            evidence.evidence_id: evidence
            for evidence in candidate.evidence
        }
        target = candidate.target
        target_uniprot = (target.uniprot_id or "").casefold()
        target_symbol = target.symbol.casefold()
        target_chembl = str(
            target.metadata.get("chembl_target_id") or ""
        ).casefold()
        for activity in candidate.activities:
            if not activity.source_evidence_id:
                continue
            evidence = evidence_by_id.get(activity.source_evidence_id)
            if evidence is None:
                continue
            metadata = evidence.metadata
            direct_attestation = metadata.get(
                "direct_single_protein_engagement"
            )
            if direct_attestation is not True and not (
                direct_attestation is None
                and MA3AssuranceAgent._assay_type_supports_direct_engagement(
                    activity.assay_type
                )
            ):
                continue
            accessions = {
                str(metadata.get(key) or "").casefold()
                for key in (
                    "target_accession",
                    "target_uniprot_id",
                    "uniprot_accession",
                    "uniprot_id",
                )
            }
            genes = {
                str(metadata.get(key) or "").casefold()
                for key in ("target_gene", "gene", "gene_symbol")
            }
            chembl_targets = {
                str(metadata.get(key) or "").casefold()
                for key in ("target_chembl_id", "chembl_target_id")
            }
            accessions.discard("")
            genes.discard("")
            chembl_targets.discard("")
            if target_uniprot:
                linked = (
                    accessions == {target_uniprot}
                    if accessions
                    else bool(
                        target_chembl
                        and chembl_targets == {target_chembl}
                    )
                )
            elif target_chembl:
                linked = chembl_targets == {target_chembl}
            else:
                linked = bool(
                    target_symbol
                    and not accessions
                    and not chembl_targets
                    and genes == {target_symbol}
                )
            if linked:
                return True
        return False

    @staticmethod
    def _assay_type_supports_direct_engagement(assay_type: str) -> bool:
        normalized = " ".join(
            assay_type.casefold().replace("_", " ").split()
        )
        if any(
            term in normalized
            for term in ("cell", "tissue", "organism", "phenotyp")
        ):
            return False
        return any(
            term in normalized
            for term in (
                "single protein",
                "biochemical",
                "protein binding",
                "purified protein",
                "enzyme assay",
            )
        )

    @staticmethod
    def _portfolio_target_key(target) -> str:
        if target.uniprot_id:
            return f"uniprot:{target.uniprot_id.strip().upper()}"
        if target.ensembl_id:
            return f"ensembl:{target.ensembl_id.strip().upper()}"
        return f"symbol:{target.symbol.strip().upper()}"

    def _audit_target_portfolio(
        self,
        hypotheses: HypothesisPacket,
        plan: PlanSpec,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Independently verify target coverage, routing, provenance and ranking."""

        findings: list[AuditFinding] = []
        summary_by_hypothesis = {
            item.hypothesis_id: item for item in packet.target_executions
        }
        if len(summary_by_hypothesis) != len(packet.target_executions):
            findings.append(
                self._finding(
                    GateName.G1,
                    "schema",
                    Severity.CRITICAL,
                    "PORTFOLIO_PROVENANCE_MISMATCH",
                    "Target execution summaries contain duplicate hypothesis identifiers.",
                    packet.packet_id,
                    "ma2.portfolio",
                )
            )
        if set(summary_by_hypothesis) != {
            item.hypothesis_id for item in hypotheses.hypotheses
        }:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "PORTFOLIO_COMPLETENESS_MISMATCH",
                    "The target portfolio does not retain every G0-approved hypothesis.",
                    packet.packet_id,
                    "ma2.portfolio",
                )
            )

        seen_keys: set[str] = set()
        eligible_keys: set[str] = set()
        expected_route_by_key: dict[str, str] = {}
        hypothesis_score_by_key: dict[str, float] = {}
        for hypothesis in hypotheses.hypotheses:
            summary = summary_by_hypothesis.get(hypothesis.hypothesis_id)
            if summary is None:
                continue
            key = self._portfolio_target_key(hypothesis.target)
            identity_resolved = hypothesis.target.identity_status == "resolved" or (
                packet.backend_kind is ScientificBackendKind.FIXTURE
                and hypothesis.target.identity_status == "fixture"
            )
            if summary.target_key != key or summary.identity_resolved != identity_resolved:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "PORTFOLIO_PROVENANCE_MISMATCH",
                        "A target summary does not match its immutable MA1 target identity.",
                        hypothesis.hypothesis_id,
                        "ma2.portfolio",
                    )
                )
            if key in seen_keys:
                if summary.status != "duplicate" or summary.attempted:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "execution",
                            Severity.HIGH,
                            "PORTFOLIO_DUPLICATE_TARGET_EXECUTED",
                            "A duplicate stable target identity was executed more than once.",
                            hypothesis.hypothesis_id,
                            "ma2.portfolio",
                        )
                    )
                continue
            seen_keys.add(key)
            if not identity_resolved:
                if summary.execution_eligible or summary.attempted:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "execution",
                            Severity.CRITICAL,
                            "PORTFOLIO_COMPLETENESS_MISMATCH",
                            "An unresolved target identity was marked executable.",
                            hypothesis.hypothesis_id,
                            "ma2.portfolio",
                        )
                    )
                continue
            expected = select_route_from_evidence(hypothesis, hypothesis.target)
            expected_route = str(expected["selected_route"])
            expected_route_by_key[key] = expected_route
            declared = summary.route_decision
            route_attestation_invalid = (
                summary.route is None
                or summary.route.value != expected_route
                or declared.get("selected_route") != expected_route
                or declared.get("selected_routes") != [expected_route]
                or declared.get("policy_version") != expected.get("policy_version")
                or declared.get("hypothesis_id") != hypothesis.hypothesis_id
                or declared.get("thresholds") != expected.get("thresholds")
            )
            if route_attestation_invalid:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "PORTFOLIO_ROUTE_POLICY_VIOLATION",
                        f"Target {key} did not follow its independently recomputed MA1-evidence route.",
                        hypothesis.hypothesis_id,
                        "ma2.route_selection",
                    )
                )
            fallback = expected.get("fallback", {})
            route_evidence_insufficient = bool(
                isinstance(fallback, dict)
                and fallback.get("abstain_recommended") is True
            )
            if route_evidence_insufficient:
                if (
                    summary.execution_eligible
                    or summary.attempted
                    or summary.status != "ineligible"
                    or "ROUTE_EVIDENCE_INSUFFICIENT" not in summary.reason_codes
                ):
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "execution",
                            Severity.CRITICAL,
                            "PORTFOLIO_ROUTE_POLICY_VIOLATION",
                            (
                                "A target whose deterministic route required more "
                                "evidence was executed or was not retained as ineligible."
                            ),
                            hypothesis.hypothesis_id,
                            "ma2.route_selection",
                        )
                    )
                continue

            eligible_keys.add(key)
            hypothesis_score_by_key[key] = hypothesis.score
            if not summary.execution_eligible or not summary.attempted:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "PORTFOLIO_COMPLETENESS_MISMATCH",
                        "An evidence-qualified, identity-resolved target was not attempted by MA2.",
                        hypothesis.hypothesis_id,
                        "ma2.portfolio",
                    )
                )
                continue

        candidates_by_key: dict[str, list] = {}
        for candidate in packet.candidates:
            key = str(candidate.metadata.get("portfolio_target_key", ""))
            candidates_by_key.setdefault(key, []).append(candidate)
            if key not in eligible_keys:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "PORTFOLIO_PROVENANCE_MISMATCH",
                        "A candidate is not linked to an eligible portfolio target.",
                        candidate.candidate_id,
                        "ma2.portfolio",
                    )
                )
            expected_route = expected_route_by_key.get(key)
            if expected_route and candidate.route.value != expected_route:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "PORTFOLIO_ROUTE_POLICY_VIOLATION",
                        "A candidate route conflicts with its target-scoped route decision.",
                        candidate.candidate_id,
                        "ma2.route_selection",
                    )
                )

        summary_call_ids: set[str] = set()
        for summary in packet.target_executions:
            if not summary.attempted:
                continue
            current_summary_call_ids = set(summary.tool_call_ids)
            summary_call_ids.update(current_summary_call_ids)
            if summary.readiness is None:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.HIGH,
                        "GATEWAY_READINESS_MISSING",
                        "An attempted portfolio target has no capability snapshot.",
                        summary.hypothesis_id,
                        "ma2.tool_dispatch",
                    )
                )
            else:
                required = {"target_normalizer"}
                branch_candidates = candidates_by_key.get(
                    summary.target_key, []
                )
                if any(
                    candidate.admet_execution is not None
                    or candidate.status != "excluded"
                    for candidate in branch_candidates
                ):
                    required.add("admet_predictor")
                if summary.route is not None:
                    required.add(
                        "repurposing_search"
                        if summary.route.value == "repurposing"
                        else "de_novo_generator"
                    )
                unready = [
                    item
                    for item in summary.readiness.capabilities
                    if item.tool_name in required and item.status.value != "ready"
                ]
                if unready:
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "execution",
                            Severity.HIGH,
                            "GATEWAY_NOT_READY",
                            "Target-scoped capabilities were not ready: "
                            + ", ".join(
                                f"{item.tool_name}/{item.reason_code}"
                                for item in unready
                            ),
                            summary.hypothesis_id,
                            "ma2.tool_dispatch",
                        )
                    )
            actual_ids = {
                item.candidate_id
                for item in candidates_by_key.get(summary.target_key, [])
            }
            if actual_ids != set(summary.candidate_ids):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "PORTFOLIO_PROVENANCE_MISMATCH",
                        "A target summary candidate manifest does not match CandidatePacket.",
                        summary.hypothesis_id,
                        "ma2.portfolio",
                    )
                )
            for candidate in candidates_by_key.get(summary.target_key, []):
                if not set(candidate.tool_trace_ids).issubset(current_summary_call_ids):
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "evidence_provenance",
                            Severity.CRITICAL,
                            "PORTFOLIO_PROVENANCE_MISMATCH",
                            "Candidate tool lineage escapes its target execution scope.",
                            candidate.candidate_id,
                            "ma2.portfolio",
                        )
                    )
        plan_level_call_ids = {
            item.call_id
            for item in packet.tool_calls
            if item.operation == "repair_plan_trace"
        }
        if summary_call_ids | plan_level_call_ids != {
            item.call_id for item in packet.tool_calls
        }:
            findings.append(
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.CRITICAL,
                    "PORTFOLIO_PROVENANCE_MISMATCH",
                    "Target execution manifests do not exactly cover the packet tool trace.",
                    packet.packet_id,
                    "ma2.portfolio",
                )
            )

        count_attestation = {
            "g0_approved_target_count": packet.g0_approved_target_count,
            "eligible_target_count": packet.eligible_target_count,
            "attempted_target_count": packet.attempted_target_count,
            "completed_target_count": packet.completed_target_count,
            "unavailable_target_count": packet.unavailable_target_count,
            "concluded_target_count": sum(
                item.status in {"completed", "no_candidates"}
                for item in packet.target_executions
            ),
            "no_candidate_target_count": sum(
                item.status == "no_candidates"
                for item in packet.target_executions
            ),
        }
        if any(
            packet.metadata.get(name) != value
            for name, value in count_attestation.items()
        ):
            findings.append(
                self._finding(
                    GateName.G1,
                    "schema",
                    Severity.CRITICAL,
                    "PORTFOLIO_PROVENANCE_MISMATCH",
                    "Portfolio metadata counts do not match the typed packet fields.",
                    packet.packet_id,
                    "ma2.portfolio",
                )
            )

        eligible_portfolio_candidates = [
            candidate
            for candidate in packet.candidates
            if candidate.status != "excluded"
        ]
        declared_has_candidate = packet.metadata.get(
            "portfolio_has_eligible_candidate"
        )
        if declared_has_candidate is not bool(eligible_portfolio_candidates):
            findings.append(
                self._finding(
                    GateName.G1,
                    "schema",
                    Severity.CRITICAL,
                    "PORTFOLIO_PROVENANCE_MISMATCH",
                    "Portfolio candidate-availability metadata does not match the packet.",
                    packet.packet_id,
                    "ma2.portfolio",
                )
            )

        for summary in packet.target_executions:
            if summary.status != "no_candidates":
                continue
            branch_candidates = candidates_by_key.get(summary.target_key, [])
            if any(item.status != "excluded" for item in branch_candidates):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "execution",
                        Severity.CRITICAL,
                        "PORTFOLIO_COMPLETENESS_MISMATCH",
                        "A no-candidate branch contains an eligible candidate.",
                        summary.hypothesis_id,
                        "ma2.portfolio",
                    )
                )
            branch_calls = [
                item
                for item in packet.tool_calls
                if item.call_id in set(summary.tool_call_ids)
            ]
            discovery_tool = (
                "repurposing_search"
                if summary.route is not None
                and summary.route.value == "repurposing"
                else "de_novo_generator"
            )
            discovery_calls = [
                item for item in branch_calls if item.tool_name == discovery_tool
            ]
            if (
                not summary.reason_codes
                or not discovery_calls
                or any(
                    not item.success
                    or item.execution_status
                    is not ScientificExecutionStatus.COMPLETED
                    for item in discovery_calls
                )
            ):
                findings.append(
                    self._finding(
                        GateName.G1,
                        "evidence_provenance",
                        Severity.CRITICAL,
                        "PORTFOLIO_COMPLETENESS_MISMATCH",
                        (
                            "A no-candidate branch lacks a successful, completed "
                            "target-scoped discovery trace and reason."
                        ),
                        summary.hypothesis_id,
                        "ma2.portfolio",
                    )
                )

        attestation = packet.metadata.get("portfolio_ranking", {})
        if (
            not isinstance(attestation, dict)
            or attestation.get("version") != PORTFOLIO_RANKING_VERSION
            or attestation.get("cross_target_raw_docking_comparison") is not False
        ):
            findings.append(
                self._finding(
                    GateName.G1,
                    "model_appropriateness",
                    Severity.CRITICAL,
                    "CROSS_TARGET_RAW_SCORE_COMPARISON",
                    "Portfolio ranking lacks the audited within-target normalization attestation.",
                    packet.packet_id,
                    "ma2.portfolio_ranking",
                )
            )

        expected_order: list[tuple[float, str, int, str, str]] = []
        for key, cards in candidates_by_key.items():
            eligible_cards = [item for item in cards if item.status != "excluded"]
            target_best_score = max(
                (float(item.score) for item in eligible_cards),
                default=0.0,
            )
            expected_target_ranks = list(range(1, len(eligible_cards) + 1))
            observed_target_ranks = sorted(
                item.target_rank for item in eligible_cards if item.target_rank is not None
            )
            if observed_target_ranks != expected_target_ranks:
                findings.append(
                    self._finding(
                        GateName.G1,
                        "schema",
                        Severity.HIGH,
                        "TARGET_RANK_INTEGRITY_FAILURE",
                        f"Target ranks are not contiguous for {key}.",
                        packet.packet_id,
                        "ma2.portfolio_ranking",
                    )
                )
            for candidate in eligible_cards:
                factors = portfolio_components(
                    candidate,
                    hypothesis_score=hypothesis_score_by_key.get(key, 0.0),
                    eligible_count_for_target=len(eligible_cards),
                    target_best_score=target_best_score,
                )
                expected_score = portfolio_score(factors)
                recorded = candidate.metadata.get("portfolio_ranking", {})
                if (
                    candidate.portfolio_score is None
                    or abs(candidate.portfolio_score - expected_score) > 1e-6
                    or not isinstance(recorded, dict)
                    or recorded.get("factors") != factors
                    or recorded.get("raw_docking_compared_across_targets") is not False
                    or recorded.get("within_target_normalization")
                    != "continuous_score_ratio_to_target_best"
                    or recorded.get("target_best_score")
                    != round(target_best_score, 6)
                ):
                    findings.append(
                        self._finding(
                            GateName.G1,
                            "model_appropriateness",
                            Severity.CRITICAL,
                            "CROSS_TARGET_RAW_SCORE_COMPARISON",
                            "A portfolio score cannot be reproduced from audited normalized factors.",
                            candidate.candidate_id,
                            "ma2.portfolio_ranking",
                        )
                    )
                expected_order.append(
                    (
                        expected_score,
                        key,
                        candidate.target_rank or 10**9,
                        candidate.name.casefold(),
                        candidate.candidate_id,
                    )
                )
        expected_order.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
        expected_ranks = {
            item[-1]: index for index, item in enumerate(expected_order, start=1)
        }
        if any(
            candidate.portfolio_rank != expected_ranks.get(candidate.candidate_id)
            for candidate in packet.candidates
            if candidate.status != "excluded"
        ):
            findings.append(
                self._finding(
                    GateName.G1,
                    "schema",
                    Severity.HIGH,
                    "PORTFOLIO_RANK_INTEGRITY_FAILURE",
                    "Portfolio ranks do not match the independently recomputed ordering.",
                    packet.packet_id,
                    "ma2.portfolio_ranking",
                )
            )

        incomplete = [
            item
            for item in packet.target_executions
            if item.execution_eligible
            if item.status not in {"completed", "no_candidates", "duplicate"}
        ]
        if incomplete and packet.execution_status is ScientificExecutionStatus.COMPLETED:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "PORTFOLIO_COMPLETENESS_MISMATCH",
                    "Incomplete target coverage was mislabeled as completed.",
                    packet.packet_id,
                    "ma2.portfolio",
                )
            )
        return findings

    def _audit_route_policy(
        self,
        hypotheses: HypothesisPacket,
        plan: PlanSpec,
        packet: CandidatePacket,
    ) -> list[AuditFinding]:
        """Recompute MA2's one-route rule from immutable MA1 inputs."""

        findings: list[AuditFinding] = []
        if packet.portfolio_enabled:
            # Every target-scoped route has already been independently
            # recomputed by ``_audit_target_portfolio``.  Applying the legacy
            # one-primary-route assertion here would incorrectly reject a
            # valid mixed-route portfolio.
            return findings
        decision = packet.metadata.get("route_decision")
        if not isinstance(decision, dict):
            return [
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "ROUTE_POLICY_VIOLATION",
                    "CandidatePacket has no auditable MA1-evidence route decision.",
                    packet.packet_id,
                    "ma2.route_selection",
                )
            ]
        if not hypotheses.hypotheses:
            return findings
        primary, expected_selection = select_hypothesis_for_priority(
            hypotheses.hypotheses,
            plan.discovery_priority,
        )
        target = primary.target
        expected_decision = select_route_from_evidence(primary, target)
        expected_fallback = expected_decision.get("fallback", {})
        if (
            isinstance(expected_fallback, dict)
            and expected_fallback.get("abstain_recommended") is True
        ) or expected_selection.get("abstain_recommended_in_live") is True:
            findings.append(
                self._finding(
                    GateName.G1,
                    "evidence_provenance",
                    Severity.HIGH,
                    "ROUTE_ABSTAIN_RECOMMENDATION_IGNORED",
                    "MA2 executed even though the deterministic route policy required abstention.",
                    packet.packet_id,
                    "planner.route_guard",
                )
            )
        expected = expected_decision["selected_route"]
        selected = decision.get("selected_route")
        selected_many = decision.get("selected_routes")
        thresholds = decision.get("thresholds")
        expected_thresholds = expected_decision["thresholds"]
        structured_fields = (
            "known_drug_count",
            "metadata_known_drug_count",
            "structured_known_drug_count",
            "structured_known_drug_count_basis",
            "known_drug_count_basis",
            "known_drug_sources",
            "known_drug_examples",
            "known_drug_evidence_ids",
            "dgidb_interaction_count",
            "dgidb_interactions_observed",
            "dgidb_record_observed",
            "dgidb_overlap_verifiable",
            "clinical_trial_drug_count",
            "disease_relevant_trial_count",
            "qualifying_trial_count",
            "late_stage_trial_count",
            "active_trial_count",
            "active_or_completed_trial_count",
            "trial_summaries",
            "disease_relevant_trial_drugs",
            "qualifying_trial_drugs",
            "cross_source_drug_overlap",
            "direction_aligned_overlap",
            "direction_aligned_drugs",
            "direction_conflicting_drugs",
            "approved_drug_tractability",
            "clinical_stage_tractability",
            "drug_evidence_tier",
            "tier_reason_codes",
            "repurposing_signal_score",
            "evidence_completeness_score",
            "evidence_completeness_components",
            "evidence_mode",
            "known_drug_source_complete",
            "known_drug_coverage_status",
            "source_coverage",
            "coverage_requirements",
            "drug_name_match_method",
            "drug_name_match_limitations",
            "target_status",
            "fallback",
        )
        valid_attestation = (
            selected == expected
            and selected_many == [expected]
            and decision.get("policy_version")
            == expected_decision.get("policy_version")
            and decision.get("target_symbol") == target.symbol
            and decision.get("hypothesis_id") == primary.hypothesis_id
            and thresholds == expected_thresholds
            and decision.get("policy_eligible") is True
            and all(
                decision.get(field) == expected_decision.get(field)
                for field in structured_fields
            )
            and packet.metadata.get("primary_hypothesis_selection")
            == expected_selection
        )
        candidate_routes = {candidate.route.value for candidate in packet.candidates}
        search_outcome = packet.metadata.get("candidate_search_outcome", {})
        completed_empty_search = (
            isinstance(search_outcome, dict)
            and search_outcome.get("status") == "no_candidates"
            and search_outcome.get("search_completed") is True
            and not packet.candidates
        )
        if candidate_routes != {expected} and not completed_empty_search:
            valid_attestation = False
        if not valid_attestation:
            findings.append(
                self._finding(
                    GateName.G1,
                    "execution",
                    Severity.CRITICAL,
                    "ROUTE_POLICY_VIOLATION",
                    (
                        f"Expected exactly the {expected} route from audited MA1 evidence; "
                        f"declared={selected!r}, candidate_routes={sorted(candidate_routes)}."
                    ),
                    packet.packet_id,
                    "ma2.route_selection",
                )
            )
        return findings

    def _validate_g1(self, packet: CandidatePacket) -> list[ValidationResult]:
        repurposing = [
            candidate
            for candidate in packet.candidates
            if candidate.route.value == "repurposing" and candidate.status != "excluded"
        ]
        novel = [
            candidate
            for candidate in packet.candidates
            if candidate.route.value == "de_novo" and candidate.status != "excluded"
        ]
        decision = packet.metadata.get("route_decision", {})
        selected = decision.get("selected_route") if isinstance(decision, dict) else None
        search_outcome = packet.metadata.get("candidate_search_outcome", {})
        completed_empty_search = (
            isinstance(search_outcome, dict)
            and search_outcome.get("status") == "no_candidates"
            and search_outcome.get("search_completed") is True
            and not any(
                candidate.status != "excluded"
                for candidate in packet.candidates
            )
        )
        if packet.portfolio_enabled:
            evaluable_summaries = [
                item
                for item in packet.target_executions
                if item.status == "completed"
                and item.eligible_candidate_count > 0
            ]
            selected_routes = {
                item.route.value
                for item in evaluable_summaries
                if item.route is not None
            }
            no_candidate_count = sum(
                item.status == "no_candidates"
                for item in packet.target_executions
            )
            return [
                ValidationResult(
                    name="repurposing_positive_control",
                    passed=(
                        "repurposing" not in selected_routes
                        or (
                            bool(repurposing)
                            and any(candidate.activities for candidate in repurposing)
                        )
                    ),
                    control_type="repurposing",
                    details={
                        "candidate_count": len(repurposing),
                        "selected_target_count": sum(
                            item.route is not None
                            and item.route.value == "repurposing"
                            for item in evaluable_summaries
                        ),
                        "completed_no_candidate_target_count": no_candidate_count,
                        "portfolio_enabled": True,
                    },
                ),
                ValidationResult(
                    name="de_novo_novelty_control",
                    passed=(
                        "de_novo" not in selected_routes
                        or (
                            bool(novel)
                            and all(
                                len({candidate.smiles for candidate in group})
                                == len(group)
                                for key in {
                                    str(candidate.metadata.get("portfolio_target_key"))
                                    for candidate in novel
                                }
                                for group in [[
                                    candidate
                                    for candidate in novel
                                    if str(candidate.metadata.get("portfolio_target_key")) == key
                                ]]
                            )
                        )
                    ),
                    control_type="novel",
                    details={
                        "candidate_count": len(novel),
                        "selected_target_count": sum(
                            item.route is not None and item.route.value == "de_novo"
                            for item in evaluable_summaries
                        ),
                        "completed_no_candidate_target_count": no_candidate_count,
                        "portfolio_enabled": True,
                    },
                ),
            ]
        return [
            ValidationResult(
                name="repurposing_positive_control",
                passed=(
                    selected != "repurposing"
                    or completed_empty_search
                    or (bool(repurposing) and any(candidate.activities for candidate in repurposing))
                ),
                control_type="repurposing",
                details={
                    "candidate_count": len(repurposing),
                    "selected_route": selected,
                    "not_applicable": selected != "repurposing" or completed_empty_search,
                    "completed_empty_search": completed_empty_search,
                },
            ),
            ValidationResult(
                name="de_novo_novelty_control",
                passed=(
                    selected != "de_novo"
                    or completed_empty_search
                    or (bool(novel) and len({candidate.smiles for candidate in novel}) == len(novel))
                ),
                control_type="novel",
                details={
                    "candidate_count": len(novel),
                    "selected_route": selected,
                    "not_applicable": selected != "de_novo" or completed_empty_search,
                    "completed_empty_search": completed_empty_search,
                },
            ),
        ]

    def _decide(
        self,
        gate: GateName,
        findings: list[AuditFinding],
        validations: list[ValidationResult],
        *,
        reviewer_invoked: bool = False,
        meta_critic_invoked: bool = False,
        reviewer_note: str | None = None,
    ) -> GateVerdict:
        findings = _reviewer_authority(findings)
        # Unconfirmed reviewer suggestions cannot trigger a control-flow change,
        # including code/failed-node shortcuts to BLOCK, ABSTAIN or REPLAN.
        unresolved = [finding for finding in findings if not finding.resolved and finding.deterministic]
        codes = {finding.code for finding in unresolved}
        if codes & self.policy.block_codes:
            verdict = GateVerdictCode.BLOCK
        elif "UNAPPROVED_TOOL" in codes:
            verdict = GateVerdictCode.BLOCK
        elif codes & self.policy.abstain_codes:
            verdict = GateVerdictCode.ABSTAIN
        elif any(finding.code == "OOD_HIGH" for finding in unresolved):
            verdict = GateVerdictCode.ABSTAIN
        elif any(
            finding.code == "RESTRICTED_SAFETY_LOOKUP_UNRESOLVED"
            for finding in unresolved
        ):
            verdict = GateVerdictCode.ABSTAIN
        elif any(
            finding.failed_node == "planner" or finding.code.startswith("PLAN_")
            for finding in unresolved
        ):
            verdict = GateVerdictCode.REPLAN
        elif any(finding.severity in {Severity.HIGH, Severity.CRITICAL} for finding in unresolved):
            verdict = GateVerdictCode.REVISE
        elif any(not validation.passed for validation in validations):
            verdict = GateVerdictCode.REVISE
        else:
            verdict = GateVerdictCode.PASS

        rework = None
        if verdict in {GateVerdictCode.REVISE, GateVerdictCode.REPLAN}:
            failed_nodes = [finding.failed_node for finding in unresolved if finding.failed_node]
            failed_node = failed_nodes[0] if failed_nodes else ("ma1.critic" if gate is GateName.G0 else "ma2.evaluate")
            target_stage = "planner" if verdict is GateVerdictCode.REPLAN else ("ma1" if gate is GateName.G0 else "ma2")
            rework = ReworkOrder(
                gate=gate,
                target_stage=target_stage,
                failed_node=failed_node,
                affected_claim_ids=self._flatten(finding.claim_ids for finding in unresolved),
                affected_candidate_ids=[
                    finding.artifact_id
                    for finding in unresolved
                    if finding.artifact_id and finding.artifact_id.startswith("cand_")
                ],
                required_actions=[finding.message for finding in unresolved] or ["Repeat failed validation."],
            )

        rationale = (
            "All mandatory audit and validation checks passed."
            if verdict is GateVerdictCode.PASS
            else f"{len(unresolved)} unresolved finding(s); deterministic rule selected {verdict.value}."
        )
        advisory_count = sum(not f.resolved and f.confirmation_status == "unconfirmed" for f in findings)
        if advisory_count:
            rationale += f" {advisory_count} unconfirmed reviewer opinion(s) retained as non-blocking advisory findings."
        if reviewer_note:
            rationale = f"{rationale} {reviewer_note}"
        return GateVerdict(
            gate=gate,
            verdict=verdict,
            findings=findings,
            validations=validations,
            rationale=rationale,
            rework_order=rework,
            reviewer_model=self.reviewer_model,
            reviewer_invoked=reviewer_invoked,
            meta_critic_invoked=meta_critic_invoked,
        )

    @staticmethod
    def _finding(
        gate: GateName,
        category: str,
        severity: Severity,
        code: str,
        message: str,
        artifact_id: str | None,
        failed_node: str,
    ) -> AuditFinding:
        return AuditFinding(
            gate=gate,
            category=category,  # type: ignore[arg-type]
            severity=severity,
            code=code,
            message=message,
            artifact_id=artifact_id,
            failed_node=failed_node,
        )

    @staticmethod
    def _flatten(values: Iterable[list[str]]) -> list[str]:
        return list(dict.fromkeys(item for group in values for item in group))
