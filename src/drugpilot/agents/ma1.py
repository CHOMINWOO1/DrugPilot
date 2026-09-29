"""MA1 hypothesis-discovery workflow.

The production workflow will replace the evidence retriever and the Proposer/Critic
reasoning calls with live providers.  The default implementation is intentionally
deterministic, evidence-linked, and safe to run without API keys so the complete
web demonstration can be exercised in ``mock`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from inspect import signature
import json
import re
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from drugpilot.config import Settings, get_settings
from drugpilot.domain.models import (
    Claim,
    EvidenceRef,
    ExcludedHypothesis,
    Hypothesis,
    HypothesisPacket,
    HypothesisTrack,
    ResearchBrief,
    RunMode,
    TargetCandidate,
)
from drugpilot.orchestration.goal_focus import derive_goal_focus
from drugpilot.tools.disease_targets import (
    DiseaseTargetCandidate,
    DiseaseTargetDiscoveryResult,
    TargetSourceRecord,
)
from drugpilot.tools.target_identity import TargetIdentityResolution


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _target_symbol_key(value: object) -> str:
    return str(value).strip().casefold()


def _strict_explicit_scope_identity_reason(
    seed: "TargetSeed",
    *,
    requested_symbol: str,
) -> str | None:
    """Return a fail-closed reason for a request-scoped target identity.

    Identity resolution only establishes what target the user asked us to
    query.  It is never target--disease or intervention evidence.  The
    request-scoped lane therefore accepts only a complete, exact, non-fixture
    human identity with externally traceable stable identifiers.
    """

    if seed.identity_status == "fixture":
        return "EXPLICIT_TARGET_FIXTURE_IDENTITY_REJECTED"
    if seed.identity_status != "resolved":
        return "EXPLICIT_TARGET_IDENTITY_NOT_RESOLVED"
    if _target_symbol_key(seed.symbol) != _target_symbol_key(requested_symbol):
        return "EXPLICIT_TARGET_CANONICAL_SYMBOL_MISMATCH"
    if not seed.ensembl_id or not seed.uniprot_id:
        return "EXPLICIT_TARGET_STABLE_IDENTIFIERS_INCOMPLETE"
    metadata = (
        seed.identity_metadata
        if isinstance(seed.identity_metadata, Mapping)
        else {}
    )
    if metadata.get("status") != "resolved":
        return "EXPLICIT_TARGET_IDENTITY_NOT_RESOLVED"
    if metadata.get("ready_for_structure_lookup") is not True:
        return "EXPLICIT_TARGET_STRUCTURE_IDENTITY_NOT_READY"
    warnings = metadata.get("warnings")
    if isinstance(warnings, Sequence) and not isinstance(
        warnings, (str, bytes)
    ):
        if any(
            isinstance(item, str)
            and ("disagree" in item.casefold() or "conflict" in item.casefold())
            for item in warnings
        ):
            return "EXPLICIT_TARGET_IDENTITY_SOURCE_CONFLICT"
    raw_sources = metadata.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(
        raw_sources, (str, bytes)
    ):
        return "EXPLICIT_TARGET_IDENTITY_PROVENANCE_MISSING"
    sources = [item for item in raw_sources if isinstance(item, Mapping)]
    if not sources:
        return "EXPLICIT_TARGET_IDENTITY_PROVENANCE_MISSING"
    if any(
        item.get("source") == "fixture"
        or item.get("provenance_class") == "fixture_fallback"
        for item in sources
    ):
        return "EXPLICIT_TARGET_FIXTURE_IDENTITY_REJECTED"
    external_ok = [
        item
        for item in sources
        if item.get("status") == "ok"
        and item.get("provenance_class")
        in {"external_live", "external_cache"}
    ]
    if not external_ok:
        return "EXPLICIT_TARGET_EXTERNAL_IDENTITY_ATTESTATION_MISSING"
    if not any(
        _target_symbol_key(item.get("canonical_symbol"))
        == _target_symbol_key(requested_symbol)
        for item in external_ok
    ):
        return "EXPLICIT_TARGET_SYMBOL_ATTESTATION_MISSING"
    if not any(
        str(item.get("ensembl_id") or "").casefold()
        == seed.ensembl_id.casefold()
        for item in external_ok
    ):
        return "EXPLICIT_TARGET_ENSEMBL_ATTESTATION_MISSING"
    if not any(
        str(item.get("uniprot_accession") or "").casefold()
        == seed.uniprot_id.casefold()
        for item in external_ok
    ):
        return "EXPLICIT_TARGET_UNIPROT_ATTESTATION_MISSING"
    return None


def _explicit_scope_identity_trace_row(
    *,
    requested_symbol: str,
    origin: str,
    query_attempted: bool,
    seed: "TargetSeed" | None,
    reason_code: str | None,
) -> dict[str, Any]:
    metadata = (
        seed.identity_metadata
        if seed is not None and isinstance(seed.identity_metadata, Mapping)
        else {}
    )
    raw_sources = metadata.get("sources")
    source_rows: list[dict[str, Any]] = []
    if isinstance(raw_sources, Sequence) and not isinstance(
        raw_sources, (str, bytes)
    ):
        for item in raw_sources:
            if not isinstance(item, Mapping):
                continue
            source_rows.append(
                {
                    key: item.get(key)
                    for key in (
                        "source",
                        "status",
                        "queried",
                        "cache_hit",
                        "provenance_class",
                        "matched_by",
                        "canonical_symbol",
                        "ensembl_id",
                        "uniprot_accession",
                        "error_type",
                        "endpoint",
                    )
                }
            )
    row = {
        "requested_symbol": requested_symbol,
        "origin": origin,
        "query_attempted": query_attempted,
        "identity_status": seed.identity_status if seed is not None else None,
        "canonical_symbol": seed.symbol if seed is not None else None,
        "ensembl_id": seed.ensembl_id if seed is not None else None,
        "uniprot_id": seed.uniprot_id if seed is not None else None,
        "disposition": "resolved" if reason_code is None else "rejected",
        "reason_code": reason_code,
        "seed_created": reason_code is None,
        "scientific_support_created": False,
        "identity_source_outcomes": source_rows,
        "identity_source_outcomes_sha256": _canonical_sha256(source_rows),
        "fixture": False,
    }
    row["row_sha256"] = _canonical_sha256(row)
    return row


@dataclass(frozen=True)
class TargetSeed:
    symbol: str
    target_name: str
    track: HypothesisTrack
    priority: float
    ensembl_id: str | None = None
    uniprot_id: str | None = None
    identity_status: str | None = None
    identity_sources: tuple[str, ...] = ()
    identity_metadata: Mapping[str, Any] = field(default_factory=dict)
    discovery_source: str = "fixture_fallback"
    discovery_evidence: tuple[EvidenceRef, ...] = ()
    discovery_statuses: tuple[Mapping[str, Any], ...] = ()
    discovery_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntakeResult:
    brief: ResearchBrief
    max_iterations: int
    prior_hypotheses: tuple[Hypothesis, ...]
    prior_exclusions: tuple[ExcludedHypothesis, ...]


@dataclass(frozen=True)
class Critique:
    hypothesis: Hypothesis
    accepted: bool
    reason_codes: tuple[str, ...] = ()


_MA3_ADMISSION_POLICY = "ma3-source-bound-v1"
_MA3_REVIEWABLE_SEMANTIC_REASONS = frozenset({
    "UNSUPPORTED_CLAIM",
    "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY",
    "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION",
    "INTERVENTION_DIRECTION_UNRESOLVED",
    "VARIANT_SCOPE_UNRESOLVED",
    "DIRECT_TARGET_DIRECTION_DISCORDANT",
})


def _prepare_ma3_submission(brief: ResearchBrief, critique: Critique) -> Critique:
    """Separate source/constraint admission from scientific acceptance.

    Legacy Critic reasons, claim links, evidence flags, and direction stay intact.
    A pending submission is never a scientific approval. Unknown reason codes
    fail closed; only the explicit semantic diagnostic set is delegated.
    """
    from drugpilot.agents.evidence_semantics import ABSTRACT_CONTEXT_KEY, source_review_text

    hypothesis = critique.hypothesis.model_copy(deep=True)
    resolution = hypothesis.target.metadata.get("direction_resolution")
    unresolved_broad_placeholder = (
        isinstance(resolution, Mapping)
        and resolution.get("status") == "direction_unresolved"
        and resolution.get("resolved_direction") is None
        and hypothesis.intervention_direction == "modulate"
        and brief.constraints.intervention_direction in {"modulate", "unspecified"}
    )
    hard_reasons = [reason for reason in critique.reason_codes
                    if reason not in _MA3_REVIEWABLE_SEMANTIC_REASONS
                    and not (reason == "HYPOTHESIS_DIRECTION_NOT_ALLOWED"
                             and unresolved_broad_placeholder)]
    if not hypothesis.claims:
        hard_reasons.append("MISSING_CLAIMS")
    if not hypothesis.evidence:
        hard_reasons.append("MISSING_EVIDENCE")
    ids = [item.evidence_id for item in hypothesis.evidence]
    if len(ids) != len(set(ids)) or any(not value.strip() for value in ids):
        hard_reasons.append("EVIDENCE_IDENTITY_INVALID")
    valid_literature = 0
    complete_abstract_only_ids: list[str] = []
    for evidence in hypothesis.evidence:
        metadata = evidence.metadata
        if evidence.source not in brief.data_policy.allowed_sources:
            hard_reasons.append("UNAPPROVED_EVIDENCE_SOURCE")
        if not evidence.external_id.strip():
            hard_reasons.append("EVIDENCE_IDENTITY_INVALID")
        if (metadata.get("mock") or metadata.get("fixture")
                or metadata.get("provenance_class") not in {
                    "external_live", "external_cache", "external_mixed"}):
            hard_reasons.append("EVIDENCE_PROVENANCE_UNVERIFIED")
        excerpt = metadata.get("excerpt")
        if excerpt is not None and (
            not isinstance(excerpt, str) or not excerpt.strip()
            or sha256(excerpt.encode("utf-8")).hexdigest() != evidence.excerpt_hash
        ):
            hard_reasons.append("EVIDENCE_EXCERPT_HASH_INVALID")
        # A verified complete abstract may reach MA3 even if the legacy
        # sentence selector found no result. Titles/PMIDs alone remain out.
        if evidence.source in {"PubMed", "Europe PMC"} and (
            excerpt or ABSTRACT_CONTEXT_KEY in metadata
        ):
            context = metadata.get(ABSTRACT_CONTEXT_KEY)
            if (excerpt is None and isinstance(context, Mapping)
                    and context.get("status") == "omitted_size_limit"
                    and context.get("text") is None):
                # Availability/capacity is not source corruption. This record
                # cannot supply any source text or satisfy admission on its own.
                continue
            if source_review_text(evidence) is None:
                hard_reasons.append("EVIDENCE_SOURCE_CONTEXT_INVALID")
            else:
                valid_literature += 1
                if excerpt is None:
                    complete_abstract_only_ids.append(evidence.evidence_id)
    if valid_literature == 0:
        hard_reasons.append("MISSING_SOURCE_BOUND_LITERATURE")
    hard_reasons = list(dict.fromkeys(hard_reasons))
    hypothesis.target.metadata["ma1_admission"] = {
        "policy": _MA3_ADMISSION_POLICY,
        "status": "excluded_hard_constraint" if hard_reasons else "provisional_pending_ma3",
        "legacy_critic_accepted": critique.accepted,
        "legacy_reason_codes": list(critique.reason_codes),
        "hard_reason_codes": hard_reasons,
        "direction_filter_deferred_until_source_resolution": unresolved_broad_placeholder,
        "scientific_approval": False,
        "source_bound_literature_count": valid_literature,
        "complete_abstract_without_selected_excerpt_ids": complete_abstract_only_ids,
        "molecular_scope_enumeration": (
            "legacy_verified_excerpt_scopes_only; complete abstract remains available to MA3; "
            "not a claim that every abstract-only scope was separately enumerated"
            if complete_abstract_only_ids else "legacy_verified_excerpt_scopes"
        ),
        "claim_links_unchanged": True,
    }
    return Critique(
        hypothesis=hypothesis,
        accepted=critique.accepted and not hard_reasons,
        reason_codes=tuple(dict.fromkeys((*critique.reason_codes, *hard_reasons))),
    )


def _pending_ma3_submission(critique: Critique) -> bool:
    admission = critique.hypothesis.target.metadata.get("ma1_admission")
    return (isinstance(admission, Mapping)
            and admission.get("policy") == _MA3_ADMISSION_POLICY
            and admission.get("status") == "provisional_pending_ma3")


_EVIDENCE_RETRY_REASON_CODES = frozenset(
    {
        "MISSING_EVIDENCE",
        "UNSUPPORTED_CLAIM",
        "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY",
        "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION",
        "BROKEN_EVIDENCE_LINK",
        "BROKEN_COUNTER_EVIDENCE_LINK",
    }
)
_SCREENING_TRACE_ENTRY_LIMIT = 24
# A legacy packet can bootstrap at most 24 distinct attempts from the bounded
# screen history.  Five allowed reworks with the maximum four-slot supplement
# can add at most 20 more, so 64 preserves every reachable entry with margin.
_EVIDENCE_RETRY_ATTEMPT_LEDGER_ENTRY_LIMIT = 64
_SCREENING_TRACE_EXTERNAL_RECORD_LIMIT = 12
_SCREENING_TRACE_EVIDENCE_UNITS_PER_RECORD_LIMIT = 4
_SCREENING_TRACE_PROVENANCE_CLASS_LIMIT = 8
_SCREENING_TRACE_OPERATION_LIMIT = 24
_SCREENING_TRACE_CLAIM_LIMIT = 32
_SCREENING_TRACE_CLAIM_UNIT_LIMIT = 32
_SCREENING_TRACE_EXCERPT_MAX_CHARS = 320
_SCREENING_TRACE_QUERY_MAX_CHARS = 500
_SCREENING_TRACE_TEXT_MAX_CHARS = 300
_SCREENING_TRACE_SCHEMA_VERSION = "1.0"
_SENSITIVE_AUDIT_VALUE = re.compile(
    r"(?i)(?:(?:api[_-]?key|authorization|access[_-]?token|secret)\s*[:=]|"
    r"bearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,})"
)


@dataclass(frozen=True)
class TargetCatalogResult:
    catalog: Mapping[HypothesisTrack, tuple[TargetSeed, ...]]
    resolved_disease: Mapping[str, Any] | None = None
    source_statuses: tuple[Mapping[str, Any], ...] = ()
    discovery_error: str | None = None
    disease_resolution_status: str = "not_attempted"
    demo_fixture_authorized: bool = False
    external_seed_count: int = 0
    fallback_seed_count: int = 0
    executable_scope_applied: bool = False
    executable_scope_symbols: tuple[str, ...] = ()
    scope_candidate_pool_count: int = 0
    scope_candidate_match_count: int = 0
    scope_identity_match_count: int = 0
    discovery_candidate_trace: tuple[Mapping[str, Any], ...] = ()
    discovery_candidate_trace_total: int = 0
    discovery_candidate_trace_omitted: int = 0
    discovery_requested_max_candidates: int = 0
    request_scope_identity_lane_used: bool = False
    request_scope_identity_query_count: int = 0
    request_scope_identity_seed_count: int = 0
    request_scope_identity_trace: tuple[Mapping[str, Any], ...] = ()
    scope_dynamic_match_symbols: tuple[str, ...] = ()
    discovery_rework_expansion: int = 0


@dataclass(frozen=True)
class _ClaimEvidenceAssignment:
    direction_primary_ids: tuple[str, ...] = ()
    direction_refutation_ids: tuple[str, ...] = ()
    association_primary_ids: tuple[str, ...] = ()
    ranked_association_context_ids: tuple[str, ...] = ()
    response_modifier_context_ids: tuple[str, ...] = ()
    resistance_context_ids: tuple[str, ...] = ()
    other_counter_context_ids: tuple[str, ...] = ()
    fixture_compatibility: bool = False


@dataclass(frozen=True)
class _TrackAssignment:
    candidate: DiseaseTargetCandidate
    track: HypothesisTrack
    assignment_basis: str


class EvidenceRetriever(Protocol):
    """Minimal boundary for a future REST, MCP, or RAG evidence adapter."""

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]: ...


class TargetDiscovery(Protocol):
    """Disease-to-target boundary implemented by ``DiseaseTargetDiscovery``."""

    def discover(
        self,
        disease: str,
        disease_id: str | None = None,
        *,
        max_targets: int | None = None,
        allowed_sources: Sequence[str] | None = None,
    ) -> DiseaseTargetDiscoveryResult: ...


class IdentityResolver(Protocol):
    """Canonical target identity boundary implemented by ``TargetIdentityResolver``."""

    def resolve(
        self,
        gene_symbol: str,
        ensembl_id: str | None = None,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> TargetIdentityResolution: ...


class MockEvidenceRetriever:
    """Build stable evidence fixtures without presenting them as real citations."""

    @staticmethod
    def _pick_source(brief: ResearchBrief, preferences: Sequence[str]) -> str:
        allowed = brief.data_policy.allowed_sources
        for preference in preferences:
            if preference in allowed:
                return preference
        return allowed[0]

    def retrieve(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Sequence[EvidenceRef]:
        if seed.track is HypothesisTrack.LITERATURE:
            support_source = self._pick_source(brief, ("PubMed", "Europe PMC"))
            counter_source = self._pick_source(brief, ("Europe PMC", "PubMed"))
        else:
            support_source = self._pick_source(brief, ("Pharos", "Open Targets", "UniProt"))
            counter_source = self._pick_source(brief, ("Open Targets", "UniProt", "Pharos"))

        common = (brief.brief_id, seed.track.value, seed.symbol, iteration)
        support_title = (
            f"Mock support record: {seed.symbol} and {brief.disease} "
            f"({seed.track.value} track)"
        )
        counter_title = (
            f"Mock counter-evidence record: uncertainty for {seed.symbol} in "
            f"{brief.disease}"
        )
        return (
            EvidenceRef(
                evidence_id=_stable_id("ev", *common, "support"),
                source=support_source,
                external_id=f"MOCK-{seed.symbol}-SUPPORT",
                title=support_title,
                supports="support",
                excerpt_hash=sha256(support_title.encode("utf-8")).hexdigest(),
                retrieved_at=brief.created_at,
                metadata={
                    "mock": True,
                    "provenance_class": "fixture_fallback",
                    "track": seed.track.value,
                    "quality_score": seed.priority,
                    "target_discovery_source": seed.discovery_source,
                    "requires_live_verification": True,
                },
            ),
            EvidenceRef(
                evidence_id=_stable_id("ev", *common, "counter"),
                source=counter_source,
                external_id=f"MOCK-{seed.symbol}-COUNTER",
                title=counter_title,
                supports="counter",
                excerpt_hash=sha256(counter_title.encode("utf-8")).hexdigest(),
                retrieved_at=brief.created_at,
                metadata={
                    "mock": True,
                    "provenance_class": "fixture_fallback",
                    "track": seed.track.value,
                    "quality_score": seed.priority,
                    "target_discovery_source": seed.discovery_source,
                    "requires_live_verification": True,
                },
            ),
        )


class Intake:
    """Validate the brief and normalize bounded iteration/revision inputs."""

    def __init__(self, max_iterations_cap: int = 3) -> None:
        self.max_iterations_cap = max(1, min(3, max_iterations_cap))

    def review(self, brief: ResearchBrief, revision_context: object | None = None) -> IntakeResult:
        if not brief.data_policy.allowed_sources:
            raise ValueError("MA1 requires at least one approved evidence source")

        # The proposal specifies a bounded 2--3 round workflow.  A caller may
        # explicitly lower the budget to one round, which remains respected.
        max_iterations = min(
            3,
            self.max_iterations_cap,
            brief.constraints.max_ma1_iterations,
        )
        hypotheses = tuple(
            item.model_copy(deep=True)
            for item in _read_hypotheses(revision_context)
        )
        if any(item.disease != brief.disease for item in hypotheses):
            raise ValueError("Revision hypotheses do not match the current disease")
        prior = tuple(item.model_copy(deep=True) for item in _read_exclusions(revision_context))
        return IntakeResult(
            brief=brief,
            max_iterations=max_iterations,
            prior_hypotheses=hypotheses,
            prior_exclusions=prior,
        )


class Proposer:
    """Produce one candidate per evidence-supply track in each bounded round."""

    _DEFAULT_SEEDS: Mapping[HypothesisTrack, tuple[TargetSeed, ...]] = {
        HypothesisTrack.LITERATURE: (
            TargetSeed("STAT3", "Signal transducer and activator of transcription 3", HypothesisTrack.LITERATURE, 0.93),
            TargetSeed("MAPK1", "Mitogen-activated protein kinase 1", HypothesisTrack.LITERATURE, 0.87),
            TargetSeed("PIK3CA", "Phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic alpha", HypothesisTrack.LITERATURE, 0.82),
        ),
        HypothesisTrack.TBIO: (
            TargetSeed("GPR132", "G protein-coupled receptor 132", HypothesisTrack.TBIO, 0.89),
            TargetSeed("SLC15A4", "Solute carrier family 15 member 4", HypothesisTrack.TBIO, 0.84),
            TargetSeed("TMEM176B", "Transmembrane protein 176B", HypothesisTrack.TBIO, 0.79),
        ),
    }
    _AML_SEEDS: Mapping[HypothesisTrack, tuple[TargetSeed, ...]] = {
        HypothesisTrack.LITERATURE: (
            TargetSeed("FLT3", "Fms related receptor tyrosine kinase 3", HypothesisTrack.LITERATURE, 0.95),
            TargetSeed("BCL2", "BCL2 apoptosis regulator", HypothesisTrack.LITERATURE, 0.91),
            TargetSeed("IDH2", "Isocitrate dehydrogenase 2", HypothesisTrack.LITERATURE, 0.86),
        ),
        HypothesisTrack.TBIO: (
            TargetSeed("LILRB4", "Leukocyte immunoglobulin like receptor B4", HypothesisTrack.TBIO, 0.90),
            TargetSeed("CD300A", "CD300a molecule", HypothesisTrack.TBIO, 0.84),
            TargetSeed("GPR132", "G protein-coupled receptor 132", HypothesisTrack.TBIO, 0.79),
        ),
    }

    def __init__(self, retriever: EvidenceRetriever | None = None) -> None:
        self.retriever = retriever or MockEvidenceRetriever()
        # Run-local orchestrator callbacks; no provider or event loop in MA1.
        self.evidence_interpreter = None
        self.on_target_started = None

    @staticmethod
    def _catalog(brief: ResearchBrief) -> Mapping[HypothesisTrack, tuple[TargetSeed, ...]]:
        text = f"{brief.disease} {brief.research_goal}".casefold()
        if "acute myeloid" in text or "aml" in text or "급성 골수" in text:
            return Proposer._AML_SEEDS
        return Proposer._DEFAULT_SEEDS

    def propose(
        self,
        brief: ResearchBrief,
        iteration: int,
        excluded_symbols: set[str],
        accepted_symbols: set[str],
        catalog: Mapping[HypothesisTrack, tuple[TargetSeed, ...]] | None = None,
        allowed_target_keys: frozenset[str] | None = None,
        proposals_per_track: int = 1,
        provisional_admission: bool = False,
    ) -> list[Hypothesis]:
        proposals: list[Hypothesis] = []
        proposed_symbols: set[str] = set()
        bounded_track_quota = max(1, int(proposals_per_track))
        selected_catalog = catalog or self._catalog(brief)
        for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO):
            track_seeds = selected_catalog.get(track, ())
            track_proposal_count = 0
            for seed in track_seeds:
                normalized_symbol = _target_symbol_key(seed.symbol)
                if (
                    (
                        allowed_target_keys is not None
                        and normalized_symbol not in allowed_target_keys
                    )
                    or normalized_symbol in excluded_symbols
                    or normalized_symbol in accepted_symbols
                    or normalized_symbol in proposed_symbols
                ):
                    continue
                for scoped in _expand_molecular_hypotheses(
                    brief, self._build_hypothesis(brief, seed, iteration)
                ):
                    proposals.extend(_expand_directional_hypotheses(
                        brief, scoped, provisional_admission=provisional_admission))
                proposed_symbols.add(normalized_symbol)
                track_proposal_count += 1
                if track_proposal_count >= bounded_track_quota:
                    break
        return proposals

    def _build_hypothesis(
        self,
        brief: ResearchBrief,
        seed: TargetSeed,
        iteration: int,
    ) -> Hypothesis:
        if self.on_target_started is not None:
            self.on_target_started(seed, iteration)
        retrieved = list(self.retriever.retrieve(brief, seed, iteration))
        evidence = _merge_evidence((*seed.discovery_evidence, *retrieved))
        from drugpilot.agents.document_result_units import derive_result_units
        evidence = derive_result_units(brief, seed.symbol, evidence)
        from drugpilot.agents.molecular_scope import annotate_scopes
        evidence = annotate_scopes(brief, seed.symbol, evidence)
        if self.evidence_interpreter is not None:
            evidence = self.evidence_interpreter(brief, seed, iteration, evidence)
        assignment = _assign_claim_evidence(
            brief,
            target_symbol=seed.symbol,
            evidence=evidence,
        )
        direction = brief.constraints.intervention_direction
        if direction == "unspecified":
            direction = "modulate"  # A non-executable placeholder until source resolution.
        mechanism = _prospective_mechanism(brief, seed.symbol, direction)
        claims, claim_links, claim_units = _atomic_claims_and_links(
            brief,
            target_symbol=seed.symbol,
            track=seed.track,
            assignment=assignment,
            evidence=evidence,
            intervention_direction=direction,
        )
        reconciled_evidence = _reconcile_evidence_claim_linkage(
            evidence,
            assignment,
            claim_links=claim_links,
        )
        falsification = (
            f"Reject the hypothesis if {direction} perturbation of {seed.symbol} in at "
            "least two disease-relevant models fails to improve the pre-specified primary "
            "phenotype by at least 30% versus matched controls, or produces the opposite "
            "direction in either model."
        )
        return Hypothesis(
            hypothesis_id=_stable_id("hyp", brief.brief_id, seed.track.value, seed.symbol),
            track=seed.track,
            disease=brief.disease,
            target=TargetCandidate(
                symbol=seed.symbol,
                ensembl_id=seed.ensembl_id,
                uniprot_id=seed.uniprot_id,
                target_name=seed.target_name,
                # A request-scoped identity seed has no discovery evidence
                # from which a scientific TDL classification could be made.
                # Its track is scheduling only and must not become evidence.
                tdl=(
                    None
                    if seed.discovery_source
                    == "explicit_request_scope_identity"
                    else "Tbio"
                    if seed.track is HypothesisTrack.TBIO
                    else "literature-supported"
                ),
                target_cell_type=brief.constraints.target_cell_type,
                identity_status=_target_identity_status(seed),
                identity_sources=list(seed.identity_sources),
                metadata={
                    "target_discovery_source": seed.discovery_source,
                    "target_discovery_statuses": [
                        dict(item) for item in seed.discovery_statuses
                    ],
                    "association_score": seed.discovery_metadata.get(
                        "association_score"
                    ),
                    "discovery_rank": seed.discovery_metadata.get("rank"),
                    "discovery_track_hint": seed.discovery_metadata.get(
                        "track_hint"
                    ),
                    "discovery_track_hint_basis": seed.discovery_metadata.get(
                        "track_hint_basis"
                    ),
                    "identity": dict(seed.identity_metadata),
                    "counter_evidence_search": _counter_search_summary(
                        assignment,
                        evidence,
                    ),
                    "claim_evidence_assignment": _claim_assignment_metadata(
                        assignment,
                        claim_units=claim_units,
                    ),
                    **{
                        key: value
                        for key, value in seed.discovery_metadata.items()
                        if key
                        not in {
                            "association_score",
                            "rank",
                            "track_hint",
                            "track_hint_basis",
                        }
                    },
                },
            ),
            mechanism=mechanism,
            intervention_direction=direction,
            falsification_test=falsification,
            claims=claims,
            evidence=reconciled_evidence,
            revision=max(0, iteration - 1),
        )


def _expand_directional_hypotheses(
    brief: ResearchBrief, base: Hypothesis, *, provisional_admission: bool = False,
) -> list[Hypothesis]:
    """Resolve broad intake from the existing sources, without another model/search call.

    The request and source query metadata remain immutable. Each resolved question
    must independently satisfy the same direction, scope and source contracts.
    """
    requested = brief.constraints.intervention_direction
    if requested not in {"modulate", "unspecified"}:
        return [base]
    scope = (base.target.metadata.get("molecular_hypothesis_scope") or {}).get("label")
    result: list[Hypothesis] = []
    if provisional_admission:
        # Enumerate allowed research questions, not source-supported conclusions.
        # A regex that recognizes one direction must not suppress another before
        # MA3 sees the same original sources. At most three paid reviews remain
        # subject to the unchanged run call/cost preflight.
        directions = [direction for direction in ("inhibit", "activate", "modulate")
                      if brief.constraints.allows_hypothesis_direction(direction)]
        for direction in directions:
            branch = base.model_copy(deep=True)
            branch.hypothesis_id = _stable_id("hyp", base.hypothesis_id, "ma3-question", direction)
            branch.intervention_direction = direction
            branch.target.metadata["direction_resolution"] = {
                "policy": "ma3_allowed_direction_question_enumeration_v1",
                "requested_direction": requested, "proposed_direction": direction,
                "resolved_direction": None, "status": "direction_proposed_for_ma3",
                "evidence_ids": [], "scientific_approval": False,
                "question_count": len(directions), "maximum_questions": 3,
                "potential_ma3_review_calls": len(directions),
                "additional_network_calls": 0, "additional_model_calls": 0,
                "scope": "Allowed research question, not a verified intervention direction",
            }
            branch.falsification_test = (
                base.falsification_test
                + f"\nProposed intervention direction for MA3 review: {direction} "
                f"({scope or base.target.symbol}). This is a research question, "
                "not a source-resolved or scientifically approved direction."
            )
            diagnostic = _reclassify_prior_claim_evidence(brief, branch)
            # The legacy assignment remains proposal-level diagnostics only.
            # Do not rewrite source roles/flags when enumerating a question.
            diagnostic.evidence = [item.model_copy(deep=True) for item in base.evidence]
            result.append(diagnostic)
        return result
    for direction in ("inhibit", "activate", "modulate"):
        if not brief.constraints.allows_hypothesis_direction(direction):
            continue
        assignment = _assign_claim_evidence(
            brief, target_symbol=base.target.symbol, evidence=base.evidence,
            scope_label=scope, intervention_direction=direction,
        )
        if not assignment.direction_primary_ids or assignment.fixture_compatibility:
            continue
        branch = base.model_copy(deep=True)
        branch.hypothesis_id = _stable_id("hyp", base.hypothesis_id, direction)
        branch.intervention_direction = direction
        branch.target.metadata["direction_resolution"] = {
            "policy": "source_bound_direction_resolution_v1",
            "requested_direction": requested, "resolved_direction": direction,
            "status": "resolved_direct_primary",
            "evidence_ids": list(assignment.direction_primary_ids),
            "additional_network_calls": 0, "additional_model_calls": 0,
        }
        branch.falsification_test = (
            base.falsification_test
            + f"\nResolved intervention direction for this hypothesis: {direction} "
            f"({scope or base.target.symbol}). Preserve the protocol, controls, "
            "endpoints and decision thresholds specified above."
        )
        result.append(_reclassify_prior_claim_evidence(brief, branch))
    if result:
        return result
    unresolved = base.model_copy(deep=True)
    unresolved.target.metadata["direction_resolution"] = {
        "policy": "source_bound_direction_resolution_v1", "requested_direction": requested,
        "resolved_direction": None, "status": "direction_unresolved", "evidence_ids": [],
        "additional_network_calls": 0, "additional_model_calls": 0,
    }
    return [unresolved]


def _expand_molecular_hypotheses(brief: ResearchBrief, base: Hypothesis) -> list[Hypothesis]:
    """Reuse one retrieval/interpretation; independently screen source-scoped questions.

    A source label never becomes a user requirement or a verified construct ID.
    Full evidence is retained for audit; only same-scope eligible records can link.
    """
    from drugpilot.agents.molecular_scope import observed_scope_labels, hypothesis_scope
    from drugpilot.tools.assay_applicability import annotate_hypothesis
    labels = observed_scope_labels(brief,base.target.symbol,base.evidence)
    if not labels:
        return [annotate_hypothesis(brief, base)]
    selected = labels[:8]
    base = base.model_copy(deep=True)
    base.target.metadata["molecular_scope_branching"] = {
        "policy":"source_bound_molecular_hypothesis_v1","observed_scope_count":len(labels),
        "selected_labels":selected,"unassessed_labels":labels[8:],
        "max_scope_questions_per_target":8,"additional_retrieval_calls":0,
        "additional_model_calls":0,"general_target_question_preserved":True}
    result = [base]
    for label in selected:
        branch = base.model_copy(deep=True)
        branch.hypothesis_id = _stable_id("hyp",base.hypothesis_id,label)
        branch.target.metadata["molecular_hypothesis_scope"] = hypothesis_scope(
            brief,base.target.symbol,base.evidence,label)
        branch.target.metadata["scope_parent_hypothesis_id"] = base.hypothesis_id
        branch.falsification_test = (
            base.falsification_test
            + f"\nMolecular scope for this hypothesis: {label}. Document the exact "
            "construct while preserving the protocol, controls, endpoints and decision "
            "thresholds specified above. No such experiment is asserted to have been performed.")
        result.append(_reclassify_prior_claim_evidence(brief,branch))
    return [annotate_hypothesis(brief, h) for h in result]


class Critic:
    """Apply reproducible evidence, provenance, constraint, and falsifiability checks."""

    def review(self, brief: ResearchBrief, hypothesis: Hypothesis) -> Critique:
        reasons: list[str] = []
        if (hypothesis.target.metadata.get("direction_resolution") or {}).get("status") == "direction_unresolved":
            reasons.append("INTERVENTION_DIRECTION_UNRESOLVED")
        if (brief.constraints.allowed_hypothesis_directions is not None
                and not brief.constraints.allows_hypothesis_direction(hypothesis.intervention_direction)):
            reasons.append("HYPOTHESIS_DIRECTION_NOT_ALLOWED")
        from drugpilot.agents.molecular_scope import HYPOTHESIS_SCOPE_KEY, valid_hypothesis_scope
        if HYPOTHESIS_SCOPE_KEY in hypothesis.target.metadata and not valid_hypothesis_scope(brief,hypothesis):
            reasons.append("MOLECULAR_HYPOTHESIS_SCOPE_INVALID")
        forbidden = {
            _target_symbol_key(symbol)
            for symbol in brief.constraints.forbidden_targets
        }
        evidence_by_id = {item.evidence_id: item for item in hypothesis.evidence}
        allowed_sources = set(brief.data_policy.allowed_sources)

        if _target_symbol_key(hypothesis.target.symbol) in forbidden:
            reasons.append("FORBIDDEN_TARGET")
        if not hypothesis.falsification_test.strip():
            reasons.append("MISSING_FALSIFICATION_TEST")
        if not hypothesis.evidence:
            reasons.append("MISSING_EVIDENCE")
        if any(item.source not in allowed_sources for item in hypothesis.evidence):
            reasons.append("UNAPPROVED_EVIDENCE_SOURCE")
        direction_triage = _screening_direction_triage(hypothesis)
        if direction_triage["status"] == "variant_scope_unresolved":
            reasons.append("VARIANT_SCOPE_UNRESOLVED")
        elif direction_triage["status"] == "discordant_direct_primary":
            reasons.append("DIRECT_TARGET_DIRECTION_DISCORDANT")
        # Absence of a qualifying counter record after a real search is an
        # uncertainty, not a reason to invent a fixture or discard the
        # hypothesis. MA3 records this transparently as a non-blocking finding.
        # ``hypothesis.track`` records the candidate supply/work track; it is
        # not a scientific TDL assertion.  Authoritative target TDL (for
        # example Pharos ``Tclin``) is reconciled separately and must not make
        # an otherwise valid Tbio-supplied hypothesis fail this gate.

        for claim in hypothesis.claims:
            if not claim.evidence_ids:
                reasons.append("UNSUPPORTED_CLAIM")
                assignment = hypothesis.target.metadata.get(
                    "claim_evidence_assignment"
                )
                if isinstance(assignment, Mapping):
                    units = assignment.get("claim_units")
                    if isinstance(units, Sequence) and not isinstance(
                        units, (str, bytes)
                    ):
                        missing_roles = {
                            str(unit.get("role"))
                            for unit in units
                            if isinstance(unit, Mapping)
                            and unit.get("claim_id") == claim.claim_id
                            and unit.get("evidence_cardinality") == 0
                        }
                        if "missing_direct_target_direction_primary" in missing_roles:
                            reasons.append(
                                "MISSING_DIRECT_TARGET_DIRECTION_PRIMARY"
                            )
                        if (
                            "missing_primary_target_disease_association"
                            in missing_roles
                        ):
                            reasons.append(
                                "MISSING_PRIMARY_TARGET_DISEASE_ASSOCIATION"
                            )
            if any(evidence_id not in evidence_by_id for evidence_id in claim.evidence_ids):
                reasons.append("BROKEN_EVIDENCE_LINK")
            if any(evidence_id not in evidence_by_id for evidence_id in claim.counter_evidence_ids):
                reasons.append("BROKEN_COUNTER_EVIDENCE_LINK")

        reason_codes = tuple(dict.fromkeys(reasons))
        if reason_codes:
            return Critique(hypothesis=hypothesis.model_copy(deep=True), accepted=False, reason_codes=reason_codes)

        claim_confidence = sum(claim.confidence for claim in hypothesis.claims) / max(1, len(hypothesis.claims))
        linked = sum(bool(claim.evidence_ids) for claim in hypothesis.claims) / max(1, len(hypothesis.claims))
        quality = max(
            (float(item.metadata.get("quality_score", 0.5)) for item in hypothesis.evidence),
            default=0.5,
        )
        score = min(0.99, 0.55 * claim_confidence + 0.25 * linked + 0.10 + 0.10 * quality)
        scored = hypothesis.model_copy(update={"score": round(score, 4)}, deep=True)
        return Critique(hypothesis=scored, accepted=True)


def _is_evidence_retry_reason_set(reason_codes: Sequence[str]) -> bool:
    """Return true only for a non-empty, evidence-link-only failure set."""

    codes = tuple(str(code) for code in reason_codes)
    return bool(codes) and set(codes).issubset(_EVIDENCE_RETRY_REASON_CODES)


def _bounded_audit_text(value: object, *, limit: int) -> str | None:
    """Project text into a compact audit field without retaining credentials."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if _SENSITIVE_AUDIT_VALUE.search(text):
        return "[redacted]"
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _bounded_audit_endpoint(value: object) -> str | None:
    """Retain an endpoint path while dropping query strings and fragments."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    try:
        parsed = urlsplit(text)
    except ValueError:
        return _bounded_audit_text(text, limit=_SCREENING_TRACE_QUERY_MAX_CHARS)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _bounded_audit_text(text, limit=_SCREENING_TRACE_QUERY_MAX_CHARS)
    host = parsed.hostname
    if not host:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = f"{host}:{port}" if port is not None else host
    return _bounded_audit_text(
        urlunsplit((parsed.scheme, netloc, parsed.path, "", "")),
        limit=_SCREENING_TRACE_QUERY_MAX_CHARS,
    )


def _bounded_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _bounded_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _bounded_audit_values(
    values: object,
    *,
    limit: int,
    text_limit: int = 120,
) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result: list[str] = []
    for value in values[:limit]:
        bounded = _bounded_audit_text(value, limit=text_limit)
        if bounded is not None:
            result.append(bounded)
    return result


def _screening_audit_evidence_order(
    hypothesis: Hypothesis,
) -> tuple[EvidenceRef, ...]:
    """Place claim-linked support and limitations before audit-only context."""

    linked_ids = {
        evidence_id
        for claim in hypothesis.claims
        for evidence_id in (
            *claim.evidence_ids,
            *claim.counter_evidence_ids,
        )
    }
    return tuple(
        evidence
        for _, evidence in sorted(
            enumerate(hypothesis.evidence),
            key=lambda item: (
                item[1].evidence_id not in linked_ids,
                item[0],
            ),
        )
    )


def _screening_operation_observations(
    hypothesis: Hypothesis,
) -> tuple[list[dict[str, Any]], int]:
    """Separate source-operation status from unique scientific records."""

    observations: list[dict[str, Any]] = []
    observation_hashes: set[str] = set()
    observation_count_total = 0

    def append(observation: Mapping[str, Any]) -> None:
        nonlocal observation_count_total
        compact = {
            key: value
            for key, value in observation.items()
            if value not in (None, "", [], ())
        }
        if not compact:
            return
        encoded = json.dumps(
            compact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = sha256(encoded.encode("utf-8")).hexdigest()
        if digest in observation_hashes:
            return
        observation_hashes.add(digest)
        observation_count_total += 1
        if len(observations) < _SCREENING_TRACE_OPERATION_LIMIT:
            observations.append(compact)

    # The external-record table is bounded to the first twelve canonical
    # records. Prioritize a retrieval observation for the corresponding first
    # twelve evidence units before adding aggregate source telemetry, so the
    # visible record IDs never appear detached from the operation ledger.
    for evidence in _screening_audit_evidence_order(hypothesis)[
        :_SCREENING_TRACE_EXTERNAL_RECORD_LIMIT
    ]:
        metadata = evidence.metadata
        append(
            {
                "source": _bounded_audit_text(
                    evidence.source, limit=_SCREENING_TRACE_TEXT_MAX_CHARS
                ),
                "external_id": _bounded_audit_text(
                    evidence.external_id,
                    limit=_SCREENING_TRACE_TEXT_MAX_CHARS,
                ),
                "evidence_id": _bounded_audit_text(
                    evidence.evidence_id, limit=200
                ),
                "pmid": _bounded_audit_text(metadata.get("pmid"), limit=80),
                "doi": _bounded_audit_text(metadata.get("doi"), limit=200),
                "status": _bounded_audit_text(
                    metadata.get("abstract_retrieval_status") or "recorded",
                    limit=80,
                ),
                "content_availability": _bounded_audit_text(
                    metadata.get("content_availability"), limit=80
                ),
                "query": _bounded_audit_text(
                    metadata.get("query"),
                    limit=_SCREENING_TRACE_QUERY_MAX_CHARS,
                ),
                "endpoint": _bounded_audit_endpoint(
                    metadata.get("retrieval_endpoint")
                ),
                "observation_kind": "evidence_record_retrieval",
                "visible_record_priority": True,
            }
        )

    statuses = hypothesis.target.metadata.get("target_discovery_statuses")
    if isinstance(statuses, Sequence) and not isinstance(statuses, (str, bytes)):
        for status in statuses:
            if not isinstance(status, Mapping):
                continue
            append(
                {
                    "source": _bounded_audit_text(
                        status.get("source"), limit=_SCREENING_TRACE_TEXT_MAX_CHARS
                    ),
                    "status": _bounded_audit_text(status.get("status"), limit=80),
                    "operations": _bounded_audit_values(
                        status.get("operations"), limit=12
                    ),
                    "operation_records_observed": _bounded_nonnegative_int(
                        status.get(
                            "operation_records_observed",
                            status.get("records_observed"),
                        )
                    ),
                    "unique_records_observed": _bounded_nonnegative_int(
                        status.get("unique_records_observed")
                    ),
                    "records_screened": _bounded_nonnegative_int(
                        status.get("records_screened")
                    ),
                    "cache_hits": _bounded_nonnegative_int(
                        status.get("cache_hits")
                    ),
                    "failure_codes": _bounded_audit_values(
                        status.get("failure_codes"), limit=12
                    ),
                    "observation_kind": "target_discovery_operation",
                }
            )
            metrics = status.get("operation_metrics")
            if isinstance(metrics, Sequence) and not isinstance(
                metrics, (str, bytes)
            ):
                for metric in metrics[:12]:
                    if not isinstance(metric, Mapping):
                        continue
                    append(
                        {
                            "source": _bounded_audit_text(
                                status.get("source"),
                                limit=_SCREENING_TRACE_TEXT_MAX_CHARS,
                            ),
                            "operation": _bounded_audit_text(
                                metric.get("operation"), limit=120
                            ),
                            "status": _bounded_audit_text(
                                metric.get("status"), limit=80
                            ),
                            "operation_records_observed": _bounded_nonnegative_int(
                                metric.get("operation_records_observed")
                            ),
                            "unique_records_observed": _bounded_nonnegative_int(
                                metric.get("unique_records_observed")
                            ),
                            "records_screened": _bounded_nonnegative_int(
                                metric.get("records_screened")
                            ),
                            "success": _bounded_bool(metric.get("success")),
                            "cache_hit": _bounded_bool(metric.get("cache_hit")),
                            "failure_code": _bounded_audit_text(
                                metric.get("failure_code"), limit=120
                            ),
                            "observation_kind": "target_discovery_operation_metric",
                        }
                    )

    for evidence in _screening_audit_evidence_order(hypothesis):
        metadata = evidence.metadata
        failures = metadata.get("partial_source_failures")
        if not isinstance(failures, Mapping):
            failures = metadata.get("fallback_reason")
        if isinstance(failures, Mapping):
            for source, failure in sorted(
                failures.items(), key=lambda item: str(item[0])
            ):
                append(
                    {
                        "source": _bounded_audit_text(
                            source, limit=_SCREENING_TRACE_TEXT_MAX_CHARS
                        ),
                        "status": "failed",
                        "failure_code": _bounded_audit_text(failure, limit=120),
                        # ``partial_source_failures`` is aggregate telemetry
                        # copied onto every successfully returned evidence
                        # record. Its record-level query/endpoint therefore
                        # belongs to that successful record's source, not to
                        # the failed source named above. Never manufacture a
                        # false source-endpoint pairing.
                        "endpoint_attribution": (
                            "not_available_from_aggregate_failure"
                        ),
                        "observation_kind": "evidence_source_failure",
                    }
                )
    return observations, observation_count_total


def _screening_external_records(
    hypothesis: Hypothesis,
) -> tuple[list[dict[str, Any]], int]:
    """Group evidence units under canonical publication/source identities.

    PubMed and Europe PMC may expose the same article through different source
    records.  Preserve both provenance units, but count the shared PMID (then
    DOI or PMCID) once when reporting independent external records.
    """

    grouped: dict[str, dict[str, Any]] = {}
    unique_record_keys: set[str] = set()
    provenance_by_key: dict[str, set[str]] = {}
    sources_by_key: dict[str, set[str]] = {}
    external_ids_by_key: dict[str, set[str]] = {}
    for evidence in _screening_audit_evidence_order(hypothesis):
        raw_source = str(evidence.source)
        raw_external_id = str(evidence.external_id)
        source = _bounded_audit_text(
            raw_source, limit=_SCREENING_TRACE_TEXT_MAX_CHARS
        ) or "unknown"
        external_id = _bounded_audit_text(
            raw_external_id, limit=_SCREENING_TRACE_TEXT_MAX_CHARS
        ) or "unknown"
        metadata = evidence.metadata
        raw_pmid = str(metadata.get("pmid") or "").strip()
        if not raw_pmid and raw_source in {"PubMed", "Europe PMC"}:
            if re.fullmatch(r"[0-9]+", raw_external_id):
                raw_pmid = raw_external_id
        raw_doi = str(metadata.get("doi") or "").strip().casefold()
        raw_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw_doi)
        raw_pmcid = str(metadata.get("pmcid") or "").strip().upper()
        # DOI is publication-level identity. An erratum can have a distinct
        # PMID while repeating the original DOI, so PMID-first grouping would
        # overstate the number of independent publications.
        if raw_doi:
            canonical_identity = f"DOI:{raw_doi}"
        elif raw_pmid and re.fullmatch(r"[0-9]+", raw_pmid):
            canonical_identity = f"MED:{raw_pmid}"
        elif re.fullmatch(r"PMC[0-9]+", raw_pmcid):
            canonical_identity = raw_pmcid
        else:
            canonical_identity = f"SOURCE:{raw_source}\0{raw_external_id}"
        # Hash the exact canonical identity. Persisted display fields remain
        # bounded/redacted, so no oversized identity leaks into the packet.
        key = sha256(canonical_identity.encode("utf-8")).hexdigest()
        unique_record_keys.add(key)
        if (
            key not in grouped
            and len(grouped) >= _SCREENING_TRACE_EXTERNAL_RECORD_LIMIT
        ):
            continue
        if key not in grouped:
            grouped[key] = {
                "source": source,
                "external_id": external_id,
                "canonical_identity": _bounded_audit_text(
                    canonical_identity, limit=240
                ),
                "pmid": _bounded_audit_text(raw_pmid, limit=80),
                "doi": _bounded_audit_text(raw_doi, limit=200),
                "pmcid": _bounded_audit_text(raw_pmcid, limit=80),
                "sources": [],
                "external_ids": [],
                "provenance_classes": [],
                "provenance_class_count_total": 0,
                "evidence_units": [],
                "evidence_unit_count_total": 0,
            }
            provenance_by_key[key] = set()
            sources_by_key[key] = set()
            external_ids_by_key[key] = set()
        record = grouped[key]
        if source not in sources_by_key[key]:
            sources_by_key[key].add(source)
            record["sources"].append(source)
        if external_id not in external_ids_by_key[key]:
            external_ids_by_key[key].add(external_id)
            record["external_ids"].append(external_id)
        record["source"] = " + ".join(sorted(sources_by_key[key]))
        provenance = _bounded_audit_text(
            metadata.get("provenance_class"), limit=80
        )
        provenance_seen = provenance_by_key[key]
        if provenance and provenance not in provenance_seen:
            provenance_seen.add(provenance)
            record["provenance_class_count_total"] = len(provenance_seen)
            if len(record["provenance_classes"]) < _SCREENING_TRACE_PROVENANCE_CLASS_LIMIT:
                record["provenance_classes"].append(provenance)
        roles = _bounded_audit_values(
            metadata.get("claim_linkage_roles"), limit=12
        )
        if not roles:
            roles = [evidence.supports]
        oa_trace = _screening_epmc_oa_trace(metadata)
        unit = {
            "evidence_id": _bounded_audit_text(evidence.evidence_id, limit=200),
            "supports": evidence.supports,
            "query": _bounded_audit_text(
                metadata.get("query"), limit=_SCREENING_TRACE_QUERY_MAX_CHARS
            ),
            "endpoint": _bounded_audit_endpoint(
                metadata.get("retrieval_endpoint")
            ),
            # Only the pre-bounded excerpt projection is retained. Full
            # abstract/full-text fields are deliberately never inspected.
            "bounded_excerpt": _bounded_audit_text(
                metadata.get("excerpt"),
                limit=_SCREENING_TRACE_EXCERPT_MAX_CHARS,
            ),
            "excerpt_hash": _bounded_audit_text(evidence.excerpt_hash, limit=128),
            "content_hash": _bounded_audit_text(
                metadata.get("content_hash"), limit=128
            ),
            "roles": roles,
            "linked_claim_ids": _bounded_audit_values(
                metadata.get("linked_claim_ids"), limit=_SCREENING_TRACE_CLAIM_LIMIT
            ),
            "direct_support_eligible": metadata.get("direct_support_eligible"),
            "claim_linkage_status": _bounded_audit_text(
                metadata.get("claim_linkage_status"), limit=100
            ),
            "direct_support_rejection_codes": _bounded_audit_values(
                metadata.get("direct_support_rejection_codes"), limit=12
            ),
            "review_status": _bounded_audit_text(
                metadata.get("review_status"), limit=100
            ),
            "publication_types": _bounded_audit_values(
                metadata.get("publication_types"), limit=8
            ),
            "oa_fulltext_trace": oa_trace,
            "semantic_interpretation": {
                k: metadata["semantic_interpretation"].get(k)
                for k in ("policy", "status", "role", "record_key", "publication_key",
                          "context_sha256", "source_sha256", "authority")
            } | ({
                "disagreement_fields": sorted(
                    key
                    for key in (
                        "study_context", "target_relation", "disease_context",
                        "intervention", "outcome", "molecular_scope_label",
                    )
                    if (
                        isinstance(metadata["semantic_interpretation"].get("extractor"), Mapping)
                        and isinstance(metadata["semantic_interpretation"].get("verifier"), Mapping)
                        and metadata["semantic_interpretation"]["extractor"].get(key)
                        != metadata["semantic_interpretation"]["verifier"].get(key)
                    )
                )
            } if metadata["semantic_interpretation"].get("status") == "disagreement" else {})
            if isinstance(metadata.get("semantic_interpretation"), Mapping) else None,
            "molecular_scope": metadata.get("molecular_scope"),
        }
        record["evidence_unit_count_total"] += 1
        if (
            len(record["evidence_units"])
            < _SCREENING_TRACE_EVIDENCE_UNITS_PER_RECORD_LIMIT
        ):
            record["evidence_units"].append(
                {
                    key: value
                    for key, value in unit.items()
                    if value not in (None, [], "")
                }
            )

    records: list[dict[str, Any]] = []
    for key in grouped:
        record = grouped[key]
        record["evidence_unit_count_recorded"] = len(record["evidence_units"])
        record["evidence_unit_count_omitted"] = max(
            0,
            record["evidence_unit_count_total"]
            - record["evidence_unit_count_recorded"],
        )
        record["provenance_class_count_recorded"] = len(
            record["provenance_classes"]
        )
        record["provenance_class_count_omitted"] = max(
            0,
            record["provenance_class_count_total"]
            - record["provenance_class_count_recorded"],
        )
        records.append(
            {
                field: value
                for field, value in record.items()
                if value not in (None, [], "")
            }
        )
    return records, len(unique_record_keys)


def _screening_epmc_oa_trace(
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project OA-attempt provenance without retaining queries or documents."""

    raw = metadata.get("epmc_oa_fulltext_one_hop_trace")
    if not isinstance(raw, Mapping):
        return None
    trigger = raw.get("trigger")
    trigger = trigger if isinstance(trigger, Mapping) else {}
    observed = raw.get("observed")
    observed = observed if isinstance(observed, Mapping) else {}
    return {
        key: value
        for key, value in {
            "policy": _bounded_audit_text(raw.get("policy"), limit=120),
            "enabled": _bounded_bool(raw.get("enabled")),
            "status": _bounded_audit_text(raw.get("status"), limit=120),
            "stop_reason": _bounded_audit_text(
                raw.get("stop_reason"), limit=200
            ),
            "trigger_scope": _bounded_audit_text(
                trigger.get("trigger_scope"), limit=160
            ),
            "cross_source_escalation": _bounded_bool(
                raw.get("cross_source_escalation")
            ),
            "requested_pmcids": _bounded_audit_values(
                raw.get("requested_pmcids"), limit=16
            ),
            "resolved_exact_pmcids": _bounded_audit_values(
                raw.get("resolved_exact_pmcids"), limit=16
            ),
            "http_calls": _bounded_nonnegative_int(
                observed.get("added_http_call_count")
            ),
            "parent_fulltexts_attempted": _bounded_nonnegative_int(
                observed.get("parent_fulltexts_attempted")
            ),
            "parent_fulltexts_parsed": _bounded_nonnegative_int(
                observed.get("parent_fulltexts_parsed")
            ),
            "result_record_count": _bounded_nonnegative_int(
                observed.get("result_record_count")
            ),
            "error_reason_codes": _bounded_audit_values(
                raw.get("error_reason_codes"), limit=16
            ),
        }.items()
        if value not in (None, [], "")
    }


def _screening_claim_assignment(hypothesis: Hypothesis) -> dict[str, Any]:
    assignment = hypothesis.target.metadata.get("claim_evidence_assignment")
    assignment = assignment if isinstance(assignment, Mapping) else {}
    raw_units = assignment.get("claim_units")
    units: list[dict[str, Any]] = []
    if isinstance(raw_units, Sequence) and not isinstance(raw_units, (str, bytes)):
        for item in raw_units[:_SCREENING_TRACE_CLAIM_UNIT_LIMIT]:
            if not isinstance(item, Mapping):
                continue
            publication = item.get("publication")
            publication = publication if isinstance(publication, Mapping) else {}
            units.append(
                {
                    key: value
                    for key, value in {
                        "claim_id": _bounded_audit_text(item.get("claim_id"), limit=200),
                        "evidence_id": _bounded_audit_text(
                            item.get("evidence_id"), limit=200
                        ),
                        "role": _bounded_audit_text(item.get("role"), limit=120),
                        "predicate": _bounded_audit_text(
                            item.get("predicate"),
                            limit=_SCREENING_TRACE_TEXT_MAX_CHARS,
                        ),
                        "publication_source": _bounded_audit_text(
                            publication.get("source"), limit=120
                        ),
                        "publication_external_id": _bounded_audit_text(
                            publication.get("external_id"), limit=200
                        ),
                        "intervention_basis": _bounded_audit_text(
                            item.get("intervention_basis"),
                            limit=_SCREENING_TRACE_TEXT_MAX_CHARS,
                        ),
                        "measured_endpoints": _bounded_audit_values(
                            item.get("measured_endpoints"), limit=8
                        ),
                        "excerpt_hash": _bounded_audit_text(
                            item.get("excerpt_hash"), limit=128
                        ),
                        "source_location": _bounded_audit_text(
                            item.get("source_location"), limit=200
                        ),
                        "same_document_external_id": _bounded_audit_text(
                            item.get("same_document_external_id"), limit=200
                        ),
                        "parent_evidence_id": _bounded_audit_text(
                            item.get("parent_evidence_id"), limit=200
                        ),
                        "evidence_cardinality": _bounded_nonnegative_int(
                            item.get("evidence_cardinality")
                        ),
                    }.items()
                    if value not in (None, [], "")
                }
            )
    claims = [
        {
            "claim_id": _bounded_audit_text(claim.claim_id, limit=200),
            "evidence_ids": _bounded_audit_values(
                claim.evidence_ids, limit=_SCREENING_TRACE_EXTERNAL_RECORD_LIMIT
            ),
            "counter_evidence_ids": _bounded_audit_values(
                claim.counter_evidence_ids,
                limit=_SCREENING_TRACE_EXTERNAL_RECORD_LIMIT,
            ),
        }
        for claim in hypothesis.claims[:_SCREENING_TRACE_CLAIM_LIMIT]
    ]
    return {
        "policy": _bounded_audit_text(assignment.get("policy"), limit=200),
        "atomic_claim_linkage": _bounded_bool(
            assignment.get("atomic_claim_linkage")
        ),
        "atomicity_contract": _bounded_audit_text(
            assignment.get("atomicity_contract"), limit=200
        ),
        "claim_count_total": len(hypothesis.claims),
        "claim_count_recorded": len(claims),
        "claim_count_omitted": max(0, len(hypothesis.claims) - len(claims)),
        "claims": claims,
        "claim_unit_count_total": (
            len(raw_units)
            if isinstance(raw_units, Sequence) and not isinstance(raw_units, (str, bytes))
            else 0
        ),
        "claim_unit_count_recorded": len(units),
        "claim_unit_count_omitted": max(
            0,
            (
                len(raw_units)
                if isinstance(raw_units, Sequence)
                and not isinstance(raw_units, (str, bytes))
                else 0
            )
            - len(units),
        ),
        "claim_units": units,
    }


def _screened_target_source(
    critique: Critique,
    *,
    iteration: int,
    screening_phase: str,
) -> dict[str, Any]:
    from drugpilot.tools.assay_applicability import screening_metadata
    hypothesis = critique.hypothesis
    canonical_record_count = _screening_external_records(hypothesis)[1]
    return {
        "target_symbol": _bounded_audit_text(
            hypothesis.target.symbol, limit=_SCREENING_TRACE_TEXT_MAX_CHARS
        ),
        "hypothesis_id": _bounded_audit_text(hypothesis.hypothesis_id, limit=200),
        "molecular_hypothesis_scope": hypothesis.target.metadata.get("molecular_hypothesis_scope"),
        "track": hypothesis.track.value,
        "source_iteration": iteration,
        "screening_phase": screening_phase,
        "disposition": ("provisional" if _pending_ma3_submission(critique)
                        else "retained" if critique.accepted else "rejected"),
        **screening_metadata(hypothesis),
        "reason_codes": list(critique.reason_codes),
        "target_source": _bounded_audit_text(
            hypothesis.target.metadata.get(
                "target_discovery_source", "prior_revision"
            ),
            limit=_SCREENING_TRACE_TEXT_MAX_CHARS,
        ),
        "identity_status": hypothesis.target.identity_status,
        "ensembl_id": _bounded_audit_text(
            hypothesis.target.ensembl_id, limit=200
        ),
        "uniprot_id": _bounded_audit_text(
            hypothesis.target.uniprot_id, limit=200
        ),
        "evidence_count": canonical_record_count,
        "evidence_unit_count": len(hypothesis.evidence),
        "evidence_count_semantics": "canonical_publication_or_source_record_v2",
        "direction_triage": _screening_direction_triage(hypothesis),
        "direction_resolution": hypothesis.target.metadata.get("direction_resolution"),
    }


def _evidence_linkage_entries(brief: ResearchBrief, critique: Critique) -> list[dict[str, Any]]:
    """Bounded diagnostic projection; it never changes source or admission results."""
    from drugpilot.agents.evidence_semantics import permits
    h = critique.hypothesis
    scope = (h.target.metadata.get("molecular_hypothesis_scope") or {}).get("label")
    assignment = h.target.metadata.get("claim_evidence_assignment") or {}
    units = assignment.get("claim_units") or []
    result = []
    for item in h.evidence:
        roles = sorted({str(unit.get("role")) for unit in units
                        if isinstance(unit, Mapping) and unit.get("evidence_id") == item.evidence_id})
        semantic = item.metadata.get("semantic_interpretation") or {}
        if not isinstance(semantic, Mapping):
            semantic = {}
        reasons: list[str] = []
        if "direct_target_direction_primary" not in roles:
            for key in ("direct_support_rejection_codes", "verified_direct_primary_rejection_codes"):
                reasons.extend(_bounded_audit_values(item.metadata.get(key), limit=12))
            if not permits(item, brief, h.target.symbol, "direction", scope):
                reasons.append("SEMANTIC_ROLE_OR_MOLECULAR_SCOPE_NOT_ADMITTED")
            excerpt = str(item.metadata.get("excerpt") or "")
            if not excerpt:
                reasons.append("SOURCE_EXCERPT_UNAVAILABLE")
            else:
                if not _scope_direct_intervention_present(excerpt, h.target.symbol, h.intervention_direction, scope):
                    reasons.append("DIRECT_TARGET_INTERVENTION_MISSING")
                if not _has_admissible_direction_endpoint(excerpt, h.target.symbol):
                    reasons.append("DIRECT_OR_PHENOTYPIC_ENDPOINT_MISSING")
                if _direction_endpoint_polarity_status(excerpt, h.intervention_direction) != "favorable":
                    reasons.append("FAVORABLE_ENDPOINT_POLARITY_NOT_ESTABLISHED")
            if not reasons:
                reasons.append("PRIMARY_SOURCE_CONTRACT_OR_CLAIM_ASSIGNMENT_NOT_ADMITTED")
        result.append({
            "hypothesis_id": h.hypothesis_id, "target_symbol": h.target.symbol,
            "molecular_scope": scope, "requested_direction": brief.constraints.intervention_direction,
            "resolved_direction": (h.target.metadata.get("direction_resolution") or {}).get(
                "resolved_direction", h.intervention_direction),
            "hypothesis_disposition": ("provisional" if _pending_ma3_submission(critique)
                                       else "retained" if critique.accepted else "excluded"),
            "hypothesis_reason_codes": list(critique.reason_codes),
            "evidence_id": item.evidence_id, "source": item.source, "external_id": item.external_id,
            "excerpt_hash": item.excerpt_hash, "semantic_status": semantic.get("status"),
            "semantic_role": semantic.get("role"), "linked_roles": roles,
            "direction_rejection_codes": list(dict.fromkeys(reasons)),
            "diagnostic_only": True,
        })
    return result


def _rejected_screening_entry(
    critique: Critique,
    *,
    iteration: int,
    screening_phase: str,
) -> dict[str, Any]:
    from drugpilot.tools.assay_applicability import screening_metadata
    hypothesis = critique.hypothesis
    records, unique_record_total = _screening_external_records(hypothesis)
    observations, observation_total = _screening_operation_observations(hypothesis)
    return {
        "target": {
            "symbol": _bounded_audit_text(
                hypothesis.target.symbol, limit=_SCREENING_TRACE_TEXT_MAX_CHARS
            ),
            "target_name": _bounded_audit_text(
                hypothesis.target.target_name, limit=_SCREENING_TRACE_TEXT_MAX_CHARS
            ),
            "track": hypothesis.track.value,
            "ensembl_id": _bounded_audit_text(
                hypothesis.target.ensembl_id, limit=200
            ),
            "uniprot_id": _bounded_audit_text(
                hypothesis.target.uniprot_id, limit=200
            ),
            "identity_status": hypothesis.target.identity_status,
            "identity_sources": _bounded_audit_values(
                hypothesis.target.identity_sources, limit=12
            ),
        },
        "hypothesis_id": _bounded_audit_text(hypothesis.hypothesis_id, limit=200),
        "molecular_hypothesis_scope": hypothesis.target.metadata.get("molecular_hypothesis_scope"),
        "source_iteration": iteration,
        "screening_phase": screening_phase,
        "reason_codes": list(critique.reason_codes),
        **({"disposition": "provisional", "diagnostic_only": True,
            "final_scientific_rejection": False}
           if _pending_ma3_submission(critique) else {}),
        "evidence_count_total": unique_record_total,
        "evidence_unit_count_total": len(hypothesis.evidence),
        "evidence_count_semantics": "canonical_publication_or_source_record_v2",
        "unique_external_record_count_total": unique_record_total,
        "unique_external_record_count_recorded": len(records),
        "unique_external_record_count_omitted": max(
            0, unique_record_total - len(records)
        ),
        "unique_external_records": records,
        "operation_observation_count_total": observation_total,
        "operation_observation_count_recorded": len(observations),
        "operation_observation_count_omitted": max(
            0, observation_total - len(observations)
        ),
        "operation_observation_linkage_policy": (
            "claim-linked-visible-external-record-retrieval-first-v2; "
            "aggregate-source-telemetry-secondary"
        ),
        "operation_observations": observations,
        "claim_assignment": _screening_claim_assignment(hypothesis),
        **screening_metadata(hypothesis),
        "direction_triage": _screening_direction_triage(hypothesis),
    }


def _bounded_rejected_screening_trace(
    entries: Sequence[Mapping[str, Any]],
    *,
    entry_count_total: int | None = None,
) -> dict[str, Any]:
    recorded = [dict(entry) for entry in entries[:_SCREENING_TRACE_ENTRY_LIMIT]]
    return {
        "schema_version": _SCREENING_TRACE_SCHEMA_VERSION,
        "scope": "router_rejected_only",
        "immutable_projection": True,
        "content_policy": "bounded_excerpt_only_no_full_abstract_no_secrets",
        "entry_limit": _SCREENING_TRACE_ENTRY_LIMIT,
        "external_record_limit_per_entry": _SCREENING_TRACE_EXTERNAL_RECORD_LIMIT,
        "evidence_units_per_record_limit": (
            _SCREENING_TRACE_EVIDENCE_UNITS_PER_RECORD_LIMIT
        ),
        "operation_observation_limit_per_entry": _SCREENING_TRACE_OPERATION_LIMIT,
        "excerpt_max_chars": _SCREENING_TRACE_EXCERPT_MAX_CHARS,
        "entry_count_total": (
            max(len(entries), entry_count_total)
            if entry_count_total is not None
            else len(entries)
        ),
        "entry_count_recorded": len(recorded),
        "entry_count_omitted": max(
            0,
            (
                max(len(entries), entry_count_total)
                if entry_count_total is not None
                else len(entries)
            )
            - len(recorded),
        ),
        "entries": recorded,
    }


def _screening_trace_sha256(trace: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        trace,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _select_supplementary_evidence_retry_catalog(
    catalog: Mapping[HypothesisTrack, tuple[TargetSeed, ...]],
    retry_symbols: set[str],
    *,
    prior_attempt_counts: Mapping[str, int],
    retry_limit: int,
    preferred_symbols: frozenset[str] = frozenset(),
) -> dict[HypothesisTrack, tuple[TargetSeed, ...]]:
    """Select a small retry supplement without consuming fresh-target slots.

    Least-attempted targets age to the front.  Discovery priority and symbol
    provide deterministic tie-breaks, so repeated rework cannot silently
    starve every retry except the first catalog item.
    """

    bounded_limit = max(0, int(retry_limit))
    selected: dict[HypothesisTrack, list[TargetSeed]] = {
        track: [] for track in HypothesisTrack
    }
    if bounded_limit == 0:
        return {track: () for track in HypothesisTrack}
    candidates: list[tuple[HypothesisTrack, TargetSeed]] = []
    for track in HypothesisTrack:
        for item in catalog.get(track, ()):
            if _target_symbol_key(item.symbol) not in retry_symbols:
                continue
            candidates.append(
                (
                    track,
                    replace(
                        item,
                        discovery_source="prior_revision_evidence_retry",
                        discovery_metadata={
                            **dict(item.discovery_metadata),
                            "retry_source_before_reclassification": (
                                item.discovery_source
                            ),
                            "evidence_retry_cache_policy": (
                                "fresh_public_lookup_required"
                            ),
                        },
                    ),
                )
            )
    candidates.sort(
        key=lambda pair: (
            0 if _target_symbol_key(pair[1].symbol) in preferred_symbols else 1,
            int(prior_attempt_counts.get(_target_symbol_key(pair[1].symbol), 0)),
            (
                1
                if pair[1].discovery_metadata.get(
                    "restored_from_bounded_screening_history"
                )
                else 0
            ),
            (
                0.0
                if pair[1].discovery_metadata.get(
                    "restored_from_bounded_screening_history"
                )
                else -float(pair[1].priority)
            ),
            _target_symbol_key(pair[1].symbol),
            pair[0].value,
        )
    )
    for track, seed in candidates[:bounded_limit]:
        selected[track].append(seed)
    return {track: tuple(items) for track, items in selected.items()}


def _cap_proposal_catalog(
    catalog: Mapping[HypothesisTrack, tuple[TargetSeed, ...]],
    *,
    total_limit: int,
    per_track_limit: int,
) -> dict[HypothesisTrack, tuple[TargetSeed, ...]]:
    """Apply one global proposal cap while preserving per-track order."""

    bounded_total = max(1, int(total_limit))
    bounded_per_track = max(1, int(per_track_limit))
    selected: dict[HypothesisTrack, list[TargetSeed]] = {
        track: [] for track in HypothesisTrack
    }
    positions = {track: 0 for track in HypothesisTrack}
    while sum(len(items) for items in selected.values()) < bounded_total:
        progressed = False
        for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO):
            items = catalog.get(track, ())
            position = positions[track]
            if (
                len(selected[track]) >= bounded_per_track
                or position >= len(items)
            ):
                continue
            selected[track].append(items[position])
            positions[track] += 1
            progressed = True
            if sum(len(items) for items in selected.values()) >= bounded_total:
                break
        if not progressed:
            break
    return {track: tuple(items) for track, items in selected.items()}


def _reconcile_discovery_trace_with_proposal_budget(
    trace: Sequence[Mapping[str, Any]],
    proposal_catalog: Mapping[HypothesisTrack, tuple[TargetSeed, ...]],
    *,
    max_iterations: int,
    retry_symbols: set[str],
) -> tuple[Mapping[str, Any], ...]:
    """Report the post-retry proposal schedule, not the pre-retry draft."""

    scheduled_keys = {
        _target_symbol_key(seed.symbol)
        for track in HypothesisTrack
        for seed in proposal_catalog.get(track, ())[:max_iterations]
    }
    reconciled: list[Mapping[str, Any]] = []
    for raw in trace:
        item = dict(raw)
        key = _target_symbol_key(str(item.get("target_symbol") or ""))
        if item.get("scope_disposition") == "outside_explicit_scope":
            reconciled.append(item)
            continue
        if key in scheduled_keys:
            item["screening_allocation"] = (
                "scheduled_for_ma1_evidence_retry"
                if key in retry_symbols
                else "scheduled_for_ma1"
            )
            item["reason_codes"] = []
        elif item.get("screening_allocation") == "scheduled_for_ma1":
            item["screening_allocation"] = (
                "not_scheduled_evidence_retry_budget"
            )
            item["reason_codes"] = [
                "MA1_SCREENING_BUDGET_REALLOCATED_TO_EVIDENCE_RETRY"
            ]
        reconciled.append(item)
    return tuple(reconciled)


def _restore_retry_seeds_from_screening_history(
    entries: Sequence[Mapping[str, Any]],
    retry_symbols: set[str],
) -> tuple[TargetSeed, ...]:
    """Recover canonical target identity for an evidence-only retry.

    Router exclusions intentionally keep only a symbol and reason codes. The
    bounded rejected-screening history retains the canonical identity needed
    to perform a real retry when a fresh disease-discovery window no longer
    contains that target. No evidence or claim is restored or promoted.
    """

    restored: list[TargetSeed] = []
    seen: set[str] = set()
    for entry in entries:
        target = entry.get("target")
        if not isinstance(target, Mapping):
            continue
        symbol = str(target.get("symbol") or "").strip()
        symbol_key = _target_symbol_key(symbol)
        if not symbol_key or symbol_key not in retry_symbols or symbol_key in seen:
            continue
        try:
            track = HypothesisTrack(str(target.get("track")))
        except ValueError:
            continue
        raw_sources = target.get("identity_sources")
        identity_sources = (
            tuple(str(value) for value in raw_sources if str(value).strip())
            if isinstance(raw_sources, Sequence)
            and not isinstance(raw_sources, (str, bytes))
            else ()
        )
        restored.append(
            TargetSeed(
                symbol=symbol,
                target_name=str(target.get("target_name") or symbol),
                track=track,
                priority=1.0,
                ensembl_id=(
                    str(target.get("ensembl_id"))
                    if target.get("ensembl_id")
                    else None
                ),
                uniprot_id=(
                    str(target.get("uniprot_id"))
                    if target.get("uniprot_id")
                    else None
                ),
                identity_status=(
                    str(target.get("identity_status"))
                    if target.get("identity_status")
                    else None
                ),
                identity_sources=identity_sources,
                discovery_source="prior_revision_evidence_retry",
                discovery_metadata={
                    "restored_from_bounded_screening_history": True,
                    "prior_hypothesis_id": entry.get("hypothesis_id"),
                },
            )
        )
        seen.add(symbol_key)
    return tuple(restored)


def _append_outside_scope_exclusions(
    excluded: list[ExcludedHypothesis],
    hypotheses: Sequence[Hypothesis],
) -> int:
    """Record deterministic scope removals once without promoting any target."""

    reason = "OUTSIDE_EXECUTABLE_TARGET_SCOPE"
    existing = {
        _target_symbol_key(item.target_symbol)
        for item in excluded
        if reason in item.reason_codes
    }
    added = 0
    for hypothesis in hypotheses:
        symbol_key = _target_symbol_key(hypothesis.target.symbol)
        if not symbol_key or symbol_key in existing:
            continue
        excluded.append(
            ExcludedHypothesis(
                target_symbol=hypothesis.target.symbol,
                reason_codes=[reason],
                source_iteration=max(1, hypothesis.revision + 1),
            )
        )
        existing.add(symbol_key)
        added += 1
    return added


class Router:
    """Route accepted proposals onward and preserve rejected candidates as exclusions."""

    def route(
        self,
        critiques: Sequence[Critique],
        iteration: int,
        *,
        provisional_admission: bool = False,
    ) -> tuple[list[Hypothesis], list[ExcludedHypothesis]]:
        accepted: list[Hypothesis] = []
        excluded: list[ExcludedHypothesis] = []
        for critique in critiques:
            if (_pending_ma3_submission(critique) if provisional_admission else critique.accepted):
                accepted.append(critique.hypothesis.model_copy(deep=True))
            else:
                excluded.append(
                    ExcludedHypothesis(
                        target_symbol=critique.hypothesis.target.symbol,
                        reason_codes=list(critique.reason_codes),
                        source_iteration=iteration,
                        hypothesis_id=critique.hypothesis.hypothesis_id,
                        molecular_scope=critique.hypothesis.target.metadata.get("molecular_hypothesis_scope"),
                    )
                )
        return accepted, excluded


class Selector:
    """Rank literature and Tbio candidates independently to avoid supply bias."""

    def select(
        self,
        hypotheses: Sequence[Hypothesis],
        *,
        allowed_target_keys: frozenset[str] | None = None,
    ) -> list[Hypothesis]:
        ranked: list[Hypothesis] = []
        for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO):
            track_items = [
                item
                for item in hypotheses
                if item.track is track
                and (
                    allowed_target_keys is None
                    or _target_symbol_key(item.target.symbol) in allowed_target_keys
                )
            ]
            track_items.sort(key=lambda item: (-item.score, item.target.symbol))
            ranked.extend(
                item.model_copy(update={"rank": rank}, deep=True)
                for rank, item in enumerate(track_items, start=1)
            )
        return ranked


class MA1Agent:
    """Bounded Intake -> Proposer -> Critic -> Router -> Selector workflow."""

    def __init__(
        self,
        settings: Settings | None = None,
        evidence_retriever: EvidenceRetriever | None = None,
        target_discovery: TargetDiscovery | None = None,
        identity_resolver: IdentityResolver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.intake = Intake(self.settings.max_ma1_iterations)
        self.proposer = Proposer(evidence_retriever)
        self.target_discovery = target_discovery
        self.identity_resolver = identity_resolver
        self.critic = Critic()
        self.router = Router()
        self.selector = Selector()

    def run(
        self,
        brief: ResearchBrief,
        revision_context: object | None = None,
    ) -> HypothesisPacket:
        provisional_admission = self.settings.ma1_provisional_admission_enabled
        intake = self.intake.review(brief, revision_context)
        goal_focus = derive_goal_focus(brief)
        executable_scope_symbols = (
            tuple(dict.fromkeys(goal_focus.executable_target_mentions))
            if goal_focus.target_scope_mode == "explicit_only"
            else ()
        )
        executable_scope_keys = frozenset(
            _target_symbol_key(symbol)
            for symbol in executable_scope_symbols
            if _target_symbol_key(symbol)
        )
        # Target discovery is intentionally executed once per MA1 run.  Every
        # bounded proposal round consumes the immutable catalog produced here.
        prior_attempt_symbols = tuple(
            sorted(
                {
                _target_symbol_key(item.target.symbol)
                for item in intake.prior_hypotheses
                }
                | {
                    _target_symbol_key(item.target_symbol)
                    for item in intake.prior_exclusions
                }
            )
        )
        # Iterations are scientific proposal/critique rounds, not a hidden
        # target-count limit.  Size the per-track catalog from the explicit
        # discovery budget so a comprehensive eight-target pool can be fully
        # screened across the still-bounded 1--3 MA1 rounds.
        screening_candidate_budget = max(
            1,
            int(self.settings.dynamic_target_max_candidates),
        )
        screening_per_track_limit = max(
            1,
            (screening_candidate_budget + len(HypothesisTrack) - 1)
            // len(HypothesisTrack),
        )
        prepare_kwargs: dict[str, Any] = {
            "per_track_limit": screening_per_track_limit,
        }
        # Preserve the long-standing component-test seam where callers replace
        # this private method with a two-argument deterministic catalog.
        if "prior_attempt_count" in signature(
            self._prepare_target_catalog
        ).parameters:
            prepare_kwargs["prior_attempt_count"] = len(prior_attempt_symbols)
        if "prior_attempt_symbols" in signature(
            self._prepare_target_catalog
        ).parameters:
            prepare_kwargs["prior_attempt_symbols"] = prior_attempt_symbols
        if "effective_max_iterations" in signature(
            self._prepare_target_catalog
        ).parameters:
            prepare_kwargs["effective_max_iterations"] = intake.max_iterations
        target_catalog = self._prepare_target_catalog(brief, **prepare_kwargs)
        # Previous candidates remain immutable inputs and compete with new
        # proposals; they are copied rather than revised in place.
        # A prior iteration can never carry hypotheses across an unresolved
        # disease boundary.  Explicit demo-fixture mode is the sole exception.
        scope_evidence_missing = bool(
            executable_scope_keys
            and target_catalog.scope_identity_match_count == 0
        )
        fail_closed = (
            (
                target_catalog.disease_resolution_status
                in {"unresolved", "source_unavailable"}
                and not target_catalog.demo_fixture_authorized
                and target_catalog.fallback_seed_count == 0
            )
            or scope_evidence_missing
        )
        eligible_prior_hypotheses = tuple(intake.prior_hypotheses)
        outside_scope_prior_hypotheses: tuple[Hypothesis, ...] = ()
        if executable_scope_keys:
            outside_scope_prior_hypotheses = tuple(
                item
                for item in intake.prior_hypotheses
                if _target_symbol_key(item.target.symbol)
                not in executable_scope_keys
            )
            eligible_prior_hypotheses = tuple(
                item
                for item in intake.prior_hypotheses
                if _target_symbol_key(item.target.symbol)
                in executable_scope_keys
            )
        prior_exclusions = (
            []
            if fail_closed
            else [item.model_copy(deep=True) for item in intake.prior_exclusions]
        )
        prior_scope_exclusion_count = 0
        if not fail_closed:
            prior_scope_exclusion_count = _append_outside_scope_exclusions(
                prior_exclusions,
                outside_scope_prior_hypotheses,
            )
        prior_exclusion_groups: dict[str, list[ExcludedHypothesis]] = {}
        for item in prior_exclusions:
            prior_exclusion_groups.setdefault(
                _target_symbol_key(item.target_symbol), []
            ).append(item)
        retry_candidates = {
            symbol
            for symbol, items in prior_exclusion_groups.items()
            if items
            and all(
                _is_evidence_retry_reason_set(item.reason_codes)
                for item in items
            )
        }
        hard_retry_exclusions = {
            symbol
            for symbol, items in prior_exclusion_groups.items()
            if any(
                not _is_evidence_retry_reason_set(item.reason_codes)
                for item in items
            )
        }
        # A current human scope constraint always wins over a stale evidence
        # retry code, even if an old packet did not yet record FORBIDDEN_TARGET.
        hard_retry_exclusions.update(
            _target_symbol_key(symbol)
            for symbol in brief.constraints.forbidden_targets
        )
        reclassified_prior_count = 0
        prior_evidence_retry_symbols: set[str] = set(retry_candidates)
        prior_metadata = (
            revision_context.metadata
            if isinstance(revision_context, HypothesisPacket)
            else {}
        )
        raw_prior_screening_history = prior_metadata.get(
            "ma1_screening_history_sources",
            prior_metadata.get("screened_target_sources", []),
        )
        prior_screening_history = [
            {
                **dict(item),
                "record_origin": "prior_revision_screening",
            }
            for item in raw_prior_screening_history
            if isinstance(item, Mapping)
        ][:_SCREENING_TRACE_ENTRY_LIMIT]
        prior_screening_history_total = int(
            prior_metadata.get(
                "ma1_screening_history_source_count_total",
                prior_metadata.get(
                    "screened_target_source_count_total",
                    len(prior_screening_history),
                ),
            )
            or 0
        )
        prior_evidence_retry_attempt_counts: dict[str, int] = {}
        raw_retry_attempt_ledger = prior_metadata.get(
            "ma1_evidence_retry_attempt_ledger"
        )
        retry_attempt_ledger_source = "initialized_empty"
        retry_attempt_ledger_carried_forward = False
        if "ma1_evidence_retry_attempt_ledger" in prior_metadata:
            if (
                not isinstance(raw_retry_attempt_ledger, Mapping)
                or raw_retry_attempt_ledger.get("schema_version")
                != "ma1_evidence_retry_attempt_ledger/v1"
                or raw_retry_attempt_ledger.get("authoritative") is not True
            ):
                raise ValueError(
                    "invalid persisted MA1 evidence-retry attempt ledger"
                )
            raw_retry_attempt_entries = raw_retry_attempt_ledger.get(
                "entries", []
            )
            integer_fields = {
                name: raw_retry_attempt_ledger.get(name)
                for name in (
                    "entry_limit",
                    "entry_count_total",
                    "entry_count_recorded",
                    "entry_count_omitted",
                )
            }
            if (
                not isinstance(raw_retry_attempt_entries, Sequence)
                or isinstance(raw_retry_attempt_entries, (str, bytes))
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in integer_fields.values()
                )
                or integer_fields["entry_limit"]
                != _EVIDENCE_RETRY_ATTEMPT_LEDGER_ENTRY_LIMIT
                or integer_fields["entry_count_total"]
                != len(raw_retry_attempt_entries)
                or integer_fields["entry_count_recorded"]
                != len(raw_retry_attempt_entries)
                or integer_fields["entry_count_omitted"] != 0
                or len(raw_retry_attempt_entries)
                > _EVIDENCE_RETRY_ATTEMPT_LEDGER_ENTRY_LIMIT
            ):
                raise ValueError(
                    "invalid persisted MA1 evidence-retry attempt ledger"
                )
            seen_retry_ledger_symbols: set[str] = set()
            for item in raw_retry_attempt_entries:
                if not isinstance(item, Mapping):
                    raise ValueError(
                        "invalid persisted MA1 evidence-retry attempt ledger"
                    )
                raw_symbol = item.get("target_symbol")
                symbol = _target_symbol_key(raw_symbol)
                attempt_count = item.get("attempt_count")
                if (
                    not isinstance(raw_symbol, str)
                    or not symbol
                    or raw_symbol != symbol
                    or symbol in seen_retry_ledger_symbols
                    or not isinstance(attempt_count, int)
                    or isinstance(attempt_count, bool)
                    or attempt_count <= 0
                ):
                    raise ValueError(
                        "invalid persisted MA1 evidence-retry attempt ledger"
                    )
                seen_retry_ledger_symbols.add(symbol)
                prior_evidence_retry_attempt_counts[symbol] = attempt_count
            retry_attempt_ledger_source = "carried_forward_authoritative"
            retry_attempt_ledger_carried_forward = True
        if not retry_attempt_ledger_carried_forward:
            for item in prior_screening_history:
                if item.get("target_source") != "prior_revision_evidence_retry":
                    continue
                symbol = _target_symbol_key(item.get("target_symbol"))
                if symbol:
                    prior_evidence_retry_attempt_counts[symbol] = (
                        prior_evidence_retry_attempt_counts.get(symbol, 0) + 1
                    )
            if prior_evidence_retry_attempt_counts:
                retry_attempt_ledger_source = "legacy_history_bootstrap"
        raw_prior_rejected_history = prior_metadata.get(
            "ma1_rejected_screening_history"
        )
        if not isinstance(raw_prior_rejected_history, Mapping):
            raw_prior_rejected_history = prior_metadata.get(
                "rejected_screening_trace", {}
            )
        raw_prior_rejected_entries = raw_prior_rejected_history.get(
            "entries", []
        )
        prior_rejected_history = [
            {
                **dict(item),
                "record_origin": "prior_revision_screening",
            }
            for item in raw_prior_rejected_entries
            if isinstance(item, Mapping)
        ][:_SCREENING_TRACE_ENTRY_LIMIT]
        prior_rejected_history_total = int(
            raw_prior_rejected_history.get(
                "entry_count_total", len(prior_rejected_history)
            )
            or 0
        )
        revision_attempt = int(
            prior_metadata.get("ma1_revision_attempt", -1)
        ) + 1
        screened_target_sources: list[dict[str, Any]] = []
        screened_target_source_count_total = 0
        rejected_screening_entries: list[dict[str, Any]] = []
        rejected_screening_entry_count_total = 0
        evidence_linkage_entries: list[dict[str, Any]] = []
        evidence_linkage_entry_count = 0
        evidence_linkage_reason_counts: dict[str, int] = {}

        def capture_screening(
            critiques: Sequence[Critique],
            *,
            iteration: int,
            screening_phase: str,
        ) -> None:
            nonlocal screened_target_source_count_total
            nonlocal rejected_screening_entry_count_total
            nonlocal evidence_linkage_entry_count
            for critique in critiques:
                entries = _evidence_linkage_entries(brief, critique)
                evidence_linkage_entry_count += len(entries)
                for entry in entries:
                    for reason in entry["direction_rejection_codes"]:
                        evidence_linkage_reason_counts[reason] = evidence_linkage_reason_counts.get(reason, 0) + 1
                evidence_linkage_entries.extend(entries[:max(0, 240 - len(evidence_linkage_entries))])
                screened_target_source_count_total += 1
                if len(screened_target_sources) < _SCREENING_TRACE_ENTRY_LIMIT:
                    screened_target_sources.append(
                        _screened_target_source(
                            critique,
                            iteration=iteration,
                            screening_phase=screening_phase,
                        )
                    )
                if critique.accepted:
                    continue
                rejected_screening_entry_count_total += 1
                if len(rejected_screening_entries) < _SCREENING_TRACE_ENTRY_LIMIT:
                    rejected_screening_entries.append(
                        _rejected_screening_entry(
                            critique,
                            iteration=iteration,
                            screening_phase=screening_phase,
                        )
                    )

        hypotheses: list[Hypothesis] = []
        excluded_prior: list[ExcludedHypothesis] = []
        if not fail_closed and eligible_prior_hypotheses:
            if self.proposer.evidence_interpreter is not None:
                eligible_prior_hypotheses = tuple(h.model_copy(update={"evidence":
                    self.proposer.evidence_interpreter(brief, TargetSeed(
                        h.target.symbol, h.target.target_name or h.target.symbol,
                        h.track, h.score), max(1, h.revision + 1), h.evidence)}, deep=True)
                    for h in eligible_prior_hypotheses)
            reclassified_prior = [
                _reclassify_prior_claim_evidence(brief, item)
                for item in eligible_prior_hypotheses
            ]
            reclassified_prior_count = len(reclassified_prior)
            prior_critiques = [
                self.critic.review(brief, item) for item in reclassified_prior
            ]
            if provisional_admission:
                prior_critiques = [_prepare_ma3_submission(brief, item) for item in prior_critiques]
            prior_iteration = max(
                1,
                max(item.revision for item in reclassified_prior) + 1,
            )
            # Capture the bounded audit projection before Router intentionally
            # reduces rejected hypotheses to ExcludedHypothesis reason codes.
            capture_screening(
                prior_critiques,
                iteration=prior_iteration,
                screening_phase="prior_reclassification",
            )
            accepted_prior, excluded_prior = self.router.route(
                prior_critiques,
                prior_iteration,
                provisional_admission=provisional_admission,
            )
            # A legacy omnibus packet can become unsupported when the current
            # atomic policy is applied.  That stale evidence-link failure is
            # retryable: let the normal proposer re-retrieve the same target
            # under the current bounded query policy instead of preserving an
            # exclusion that would make recovery impossible.  Any scientific,
            # source-policy, or scope failure remains a hard exclusion.
            retry_candidates.update(
                _target_symbol_key(critique.hypothesis.target.symbol)
                for critique in prior_critiques
                if not critique.accepted
                and _is_evidence_retry_reason_set(critique.reason_codes)
            )
            hard_retry_exclusions.update(
                _target_symbol_key(critique.hypothesis.target.symbol)
                for critique in prior_critiques
                if not critique.accepted
                and not _is_evidence_retry_reason_set(critique.reason_codes)
            )
            hypotheses.extend(accepted_prior)
        requested_prior_evidence_retry_symbols = (
            retry_candidates - hard_retry_exclusions
        )
        requested_prior_evidence_retry_symbols.difference_update(
            _target_symbol_key(item.target.symbol) for item in hypotheses
        )
        retry_catalog: dict[HypothesisTrack, tuple[TargetSeed, ...]] = {
            track: tuple(target_catalog.catalog.get(track, ()))
            for track in HypothesisTrack
        }
        existing_catalog_symbols = {
            _target_symbol_key(seed.symbol)
            for items in retry_catalog.values()
            for seed in items
        }
        restored_retry_seeds = _restore_retry_seeds_from_screening_history(
            prior_rejected_history,
            requested_prior_evidence_retry_symbols - existing_catalog_symbols,
        )
        for seed in restored_retry_seeds:
            retry_catalog[seed.track] = (
                *retry_catalog.get(seed.track, ()),
                seed,
            )
        restored_prior_evidence_retry_symbols = {
            _target_symbol_key(seed.symbol) for seed in restored_retry_seeds
        }
        catalog_symbols = {
            _target_symbol_key(seed.symbol)
            for items in retry_catalog.values()
            for seed in items
        }
        catalog_prior_evidence_retry_symbols = (
            requested_prior_evidence_retry_symbols & catalog_symbols
        )
        catalog_missing_prior_evidence_retry_symbols = (
            requested_prior_evidence_retry_symbols - catalog_symbols
        )
        fresh_catalog = {
            track: tuple(
                seed
                for seed in retry_catalog.get(track, ())
                if _target_symbol_key(seed.symbol)
                not in catalog_prior_evidence_retry_symbols
            )
            for track in HypothesisTrack
        }
        fresh_proposal_catalog = _cap_proposal_catalog(
            fresh_catalog,
            total_limit=screening_candidate_budget,
            per_track_limit=screening_per_track_limit,
        )
        configured_supplementary_retry_budget = max(
            0,
            int(self.settings.ma1_supplementary_evidence_retry_budget),
        )
        effective_supplementary_retry_budget = min(
            configured_supplementary_retry_budget,
            len(catalog_prior_evidence_retry_symbols),
        )
        supplementary_retry_catalog = (
            _select_supplementary_evidence_retry_catalog(
                retry_catalog,
                (catalog_prior_evidence_retry_symbols & set(self.g0_followup_targets)
                 if hasattr(self, "g0_followup_targets") else catalog_prior_evidence_retry_symbols),
                prior_attempt_counts=prior_evidence_retry_attempt_counts,
                retry_limit=effective_supplementary_retry_budget,
                preferred_symbols=frozenset(getattr(self, "g0_followup_targets", ())),
            )
        )
        proposal_catalog = {
            track: (
                *supplementary_retry_catalog.get(track, ()),
                *fresh_proposal_catalog.get(track, ()),
            )
            for track in HypothesisTrack
        }
        proposal_per_track_limit = max(
            screening_per_track_limit,
            max(
                (len(items) for items in proposal_catalog.values()),
                default=0,
            ),
        )
        effective_discovery_candidate_trace = (
            _reconcile_discovery_trace_with_proposal_budget(
                target_catalog.discovery_candidate_trace,
                proposal_catalog,
                max_iterations=proposal_per_track_limit,
                retry_symbols=catalog_prior_evidence_retry_symbols,
            )
        )
        prior_evidence_retry_symbols = {
            _target_symbol_key(seed.symbol)
            for items in proposal_catalog.values()
            for seed in items[:proposal_per_track_limit]
            if _target_symbol_key(seed.symbol)
            in catalog_prior_evidence_retry_symbols
        }
        budget_unscheduled_prior_evidence_retry_symbols = (
            catalog_prior_evidence_retry_symbols
            - prior_evidence_retry_symbols
        )
        excluded = [
            item
            for item in prior_exclusions
            if _target_symbol_key(item.target_symbol)
            not in prior_evidence_retry_symbols
        ]
        excluded.extend(
            item
            for item in excluded_prior
            if _target_symbol_key(item.target_symbol)
            not in prior_evidence_retry_symbols
        )
        excluded_symbols = {
            _target_symbol_key(item.target_symbol) for item in excluded
        }
        accepted_symbols = {
            _target_symbol_key(item.target.symbol) for item in hypotheses
        }
        attempted_prior_evidence_retry_symbols: set[str] = set()
        scheduled_proposal_symbols = {
            _target_symbol_key(seed.symbol)
            for items in proposal_catalog.values()
            for seed in items[:proposal_per_track_limit]
        }
        iterations = 0

        for iteration in range(1, intake.max_iterations + 1):
            iterations = iteration
            remaining_rounds = intake.max_iterations - iteration + 1
            remaining_by_track = [
                sum(
                    1
                    for seed in proposal_catalog.get(track, ())[
                        :proposal_per_track_limit
                    ]
                    if _target_symbol_key(seed.symbol) not in excluded_symbols
                    and _target_symbol_key(seed.symbol) not in accepted_symbols
                    and (
                        not executable_scope_keys
                        or _target_symbol_key(seed.symbol)
                        in executable_scope_keys
                    )
                )
                for track in HypothesisTrack
            ]
            proposals_per_track = max(
                1,
                (
                    max(remaining_by_track, default=0)
                    + remaining_rounds
                    - 1
                )
                // remaining_rounds,
            )
            proposals = self.proposer.propose(
                brief,
                iteration,
                excluded_symbols=excluded_symbols,
                accepted_symbols=accepted_symbols,
                catalog=proposal_catalog,
                allowed_target_keys=(
                    executable_scope_keys if executable_scope_keys else None
                ),
                proposals_per_track=proposals_per_track,
                **({"provisional_admission": True} if provisional_admission else {}),
            )
            attempted_prior_evidence_retry_symbols.update(
                _target_symbol_key(proposal.target.symbol)
                for proposal in proposals
                if _target_symbol_key(proposal.target.symbol)
                in prior_evidence_retry_symbols
            )
            critiques = [self.critic.review(brief, proposal) for proposal in proposals]
            if provisional_admission:
                critiques = [_prepare_ma3_submission(brief, item) for item in critiques]
            capture_screening(
                critiques,
                iteration=iteration,
                screening_phase="proposal",
            )
            newly_accepted, newly_excluded = self.router.route(
                critiques, iteration, provisional_admission=provisional_admission
            )
            hypotheses.extend(newly_accepted)
            excluded.extend(newly_excluded)
            accepted_symbols.update(
                _target_symbol_key(item.target.symbol)
                for item in newly_accepted
            )
            excluded_symbols.update(
                _target_symbol_key(item.target_symbol)
                for item in newly_excluded
            )

            # Two candidates in each supply track are enough for a useful mock
            # portfolio; a third round is used only when constraints removed one.
            per_track = {
                track: sum(item.track is track for item in hypotheses)
                for track in HypothesisTrack
            }
            if iteration >= min(2, intake.max_iterations) and all(
                count >= 2 for count in per_track.values()
            ) and prior_evidence_retry_symbols.issubset(
                attempted_prior_evidence_retry_symbols
            ) and scheduled_proposal_symbols.issubset(
                accepted_symbols | excluded_symbols
            ):
                break

        unattempted_prior_evidence_retry_symbols = (
            prior_evidence_retry_symbols
            - attempted_prior_evidence_retry_symbols
        )
        cumulative_evidence_retry_attempt_counts = dict(
            prior_evidence_retry_attempt_counts
        )
        for symbol in attempted_prior_evidence_retry_symbols:
            cumulative_evidence_retry_attempt_counts[symbol] = (
                cumulative_evidence_retry_attempt_counts.get(symbol, 0) + 1
            )
        if (
            len(cumulative_evidence_retry_attempt_counts)
            > _EVIDENCE_RETRY_ATTEMPT_LEDGER_ENTRY_LIMIT
        ):
            raise ValueError(
                "MA1 evidence-retry attempt ledger exceeded its bounded capacity"
            )
        retry_attempt_ledger_entries = [
            {
                "target_symbol": symbol,
                "attempt_count": attempt_count,
            }
            for symbol, attempt_count in sorted(
                cumulative_evidence_retry_attempt_counts.items()
            )
        ]
        if unattempted_prior_evidence_retry_symbols:
            existing_exclusions = {
                (
                    _target_symbol_key(item.target_symbol),
                    tuple(item.reason_codes),
                    item.source_iteration,
                )
                for item in excluded
            }
            for item in (*prior_exclusions, *excluded_prior):
                key = (
                    _target_symbol_key(item.target_symbol),
                    tuple(item.reason_codes),
                    item.source_iteration,
                )
                if (
                    _target_symbol_key(item.target_symbol)
                    in unattempted_prior_evidence_retry_symbols
                    and key not in existing_exclusions
                ):
                    excluded.append(item.model_copy(deep=True))
                    existing_exclusions.add(key)

        outside_scope_selected = [
            item
            for item in hypotheses
            if executable_scope_keys
            and _target_symbol_key(item.target.symbol) not in executable_scope_keys
        ]
        selector_scope_exclusion_count = _append_outside_scope_exclusions(
            excluded,
            outside_scope_selected,
        )
        ranked = self.selector.select(
            hypotheses,
            allowed_target_keys=(
                executable_scope_keys if executable_scope_keys else None
            ),
        )
        if provisional_admission:
            # Stable bounded proposal rounds already constrain breadth. Avoid
            # duplicate review work without collapsing distinct molecular scopes.
            unique: dict[str, Hypothesis] = {}
            for item in ranked:
                unique.setdefault(item.hypothesis_id, item)
            ranked = list(unique.values())
        packet_key = ",".join(item.hypothesis_id for item in ranked)
        exclusion_key = ",".join(
            f"{item.target_symbol}:{'/'.join(item.reason_codes)}:{item.source_iteration}"
            for item in excluded
        )
        target_sources = {
            item.target.symbol: {
                "track": item.track.value,
                "target_source": item.target.metadata.get(
                    "target_discovery_source", "prior_revision"
                ),
                "ensembl_id": item.target.ensembl_id,
                "uniprot_id": item.target.uniprot_id,
                "identity_status": item.target.identity_status,
                "original_track_hint": item.target.metadata.get(
                    "original_track_hint"
                ),
                "assignment_basis": item.target.metadata.get("assignment_basis"),
            }
            for item in ranked
        }
        fallback_used = target_catalog.fallback_seed_count > 0 or any(
            source["target_source"] == "fixture_fallback"
            for source in target_sources.values()
        )
        screening_trace = _bounded_rejected_screening_trace(
            rejected_screening_entries,
            entry_count_total=rejected_screening_entry_count_total,
        )
        if provisional_admission:
            screening_trace["scope"] = "legacy_critic_diagnostics_not_final_disposition"
            screening_trace["contains_provisional_submissions"] = True
        screening_trace_digest = _screening_trace_sha256(screening_trace)
        recorded_screened_sources = list(screened_target_sources)
        current_history_sources = [
            {
                **dict(item),
                "record_origin": "current_revision_screening",
                "revision_attempt": revision_attempt,
            }
            for item in recorded_screened_sources
        ]
        screening_history_sources = (
            prior_screening_history + current_history_sources
        )[:_SCREENING_TRACE_ENTRY_LIMIT]
        screening_history_source_count_total = (
            prior_screening_history_total + screened_target_source_count_total
        )
        current_rejected_history = [
            {
                **dict(item),
                "record_origin": "current_revision_screening",
                "revision_attempt": revision_attempt,
            }
            for item in rejected_screening_entries
        ]
        rejected_screening_history_entries = (
            prior_rejected_history + current_rejected_history
        )[:_SCREENING_TRACE_ENTRY_LIMIT]
        rejected_screening_history_total = (
            prior_rejected_history_total
            + rejected_screening_entry_count_total
        )
        rejected_screening_history = {
            **screening_trace,
            "scope": "cumulative_revision_rejected_screening",
            "entry_count_total": rejected_screening_history_total,
            "entry_count_recorded": len(rejected_screening_history_entries),
            "entry_count_omitted": max(
                0,
                rejected_screening_history_total
                - len(rejected_screening_history_entries),
            ),
            "entries": rejected_screening_history_entries,
        }
        current_screened_keys = {
            _target_symbol_key(str(item.get("target_symbol") or ""))
            for item in recorded_screened_sources
        }
        prior_screened_keys = {
            _target_symbol_key(str(item.get("target_symbol") or ""))
            for item in prior_screening_history
        }
        prior_input_exclusion_keys = {
            _target_symbol_key(item.target_symbol)
            for item in intake.prior_exclusions
        }
        exclusion_provenance_total = len(excluded)
        exclusion_provenance = [
            {
                "target_symbol": item.target_symbol,
                "reason_codes": list(item.reason_codes),
                "source_iteration": item.source_iteration,
                **({"hypothesis_id": item.hypothesis_id} if item.hypothesis_id else {}),
                **({"molecular_scope": item.molecular_scope} if item.molecular_scope else {}),
                "record_origin": (
                    "current_revision_screening"
                    if _target_symbol_key(item.target_symbol)
                    in current_screened_keys
                    else "prior_packet_screened_preserved_not_rescreened"
                    if _target_symbol_key(item.target_symbol)
                    in prior_screened_keys
                    else "prior_packet_preserved_not_rescreened"
                    if _target_symbol_key(item.target_symbol)
                    in prior_input_exclusion_keys
                    else "current_revision_constraint_or_selector"
                ),
                "scientific_rejection_replayed": False,
            }
            for item in excluded[:_SCREENING_TRACE_ENTRY_LIMIT]
        ]
        recorded_retry_status_symbols = sorted(
            requested_prior_evidence_retry_symbols
        )[:_SCREENING_TRACE_ENTRY_LIMIT]
        current_catalog_prior_evidence_retry_symbols = (
            catalog_prior_evidence_retry_symbols
            - restored_prior_evidence_retry_symbols
        )
        current_discovery_trace_symbols = {
            _target_symbol_key(item.get("target_symbol"))
            for item in target_catalog.discovery_candidate_trace
            if isinstance(item, Mapping) and item.get("target_symbol")
        }
        current_catalog_priority_by_symbol = {
            _target_symbol_key(seed.symbol): float(seed.priority)
            for items in target_catalog.catalog.values()
            for seed in items
            if _target_symbol_key(seed.symbol)
            in current_catalog_prior_evidence_retry_symbols
        }
        prior_evidence_retry_state_records = [
            {
                "target_symbol": symbol,
                "prior_retry_attempt_count": int(
                    prior_evidence_retry_attempt_counts.get(symbol, 0)
                ),
                "current_revision_attempted": (
                    symbol in attempted_prior_evidence_retry_symbols
                ),
                "cumulative_retry_attempt_count": int(
                    cumulative_evidence_retry_attempt_counts.get(symbol, 0)
                ),
                "present_in_current_discovery_trace": (
                    symbol in current_discovery_trace_symbols
                ),
                "current_catalog_priority_available": (
                    symbol in current_catalog_priority_by_symbol
                ),
                "current_catalog_priority": (
                    current_catalog_priority_by_symbol.get(symbol)
                ),
                "catalog_seed_status": (
                    "restored_from_screening_history"
                    if symbol in restored_prior_evidence_retry_symbols
                    else "available_from_current_catalog"
                    if symbol in current_catalog_prior_evidence_retry_symbols
                    else "catalog_missing"
                ),
                "execution_status": (
                    "attempted"
                    if symbol in attempted_prior_evidence_retry_symbols
                    else "catalog_missing"
                    if symbol in catalog_missing_prior_evidence_retry_symbols
                    else "budget_not_scheduled"
                    if symbol in budget_unscheduled_prior_evidence_retry_symbols
                    else "scheduled_not_attempted"
                ),
            }
            for symbol in recorded_retry_status_symbols
        ]
        prior_evidence_retry_execution_summary = {
            "semantics": (
                "catalog seed restoration only reconstructs a bounded target "
                "identity; evidence retrieval occurs only for attempted symbols; "
                "supplementary retry slots never replace fresh-target slots"
            ),
            "selection_policy": (
                "bounded_supplementary_least_prior_attempts_then_available_"
                "catalog_priority_then_stable_symbol_v2"
            ),
            "configured_supplementary_budget": (
                configured_supplementary_retry_budget
            ),
            "effective_supplementary_budget": (
                effective_supplementary_retry_budget
            ),
            "requested_count": len(requested_prior_evidence_retry_symbols),
            "catalog_available_count": len(
                catalog_prior_evidence_retry_symbols
            ),
            "catalog_seed_restored_count": len(
                restored_prior_evidence_retry_symbols
            ),
            "catalog_missing_count": len(
                catalog_missing_prior_evidence_retry_symbols
            ),
            "scheduled_count": len(prior_evidence_retry_symbols),
            "attempted_count": len(attempted_prior_evidence_retry_symbols),
            "budget_not_scheduled_count": len(
                budget_unscheduled_prior_evidence_retry_symbols
            ),
            "scheduled_not_attempted_count": len(
                unattempted_prior_evidence_retry_symbols
            ),
        }
        identity_trace_by_key = {
            _target_symbol_key(item.get("requested_symbol")): item
            for item in target_catalog.request_scope_identity_trace
            if isinstance(item, Mapping) and item.get("requested_symbol")
        }
        ordered_scope_collections = {
            "requested": list(executable_scope_symbols),
            "dynamic_exact_match": [
                symbol
                for symbol in executable_scope_symbols
                if _target_symbol_key(symbol)
                in {
                    _target_symbol_key(item)
                    for item in target_catalog.scope_dynamic_match_symbols
                }
            ],
            "identity_queried": [
                symbol
                for symbol in executable_scope_symbols
                if bool(
                    identity_trace_by_key.get(
                        _target_symbol_key(symbol), {}
                    ).get("query_attempted")
                )
            ],
            "identity_resolved": [
                symbol
                for symbol in executable_scope_symbols
                if identity_trace_by_key.get(
                    _target_symbol_key(symbol), {}
                ).get("disposition")
                == "resolved"
            ],
            "scheduled": [
                symbol
                for symbol in executable_scope_symbols
                if _target_symbol_key(symbol) in scheduled_proposal_symbols
            ],
            "screened": [
                symbol
                for symbol in executable_scope_symbols
                if _target_symbol_key(symbol)
                in (accepted_symbols | excluded_symbols)
            ],
            "retained": [
                symbol
                for symbol in executable_scope_symbols
                if not provisional_admission and _target_symbol_key(symbol)
                in {
                    _target_symbol_key(item.target.symbol) for item in ranked
                }
            ],
            "excluded": [
                symbol
                for symbol in executable_scope_symbols
                if _target_symbol_key(symbol)
                in {
                    _target_symbol_key(item.target_symbol)
                    for item in excluded
                }
            ],
        }
        if provisional_admission:
            ordered_scope_collections["provisional"] = [
                symbol for symbol in executable_scope_symbols
                if _target_symbol_key(symbol) in {
                    _target_symbol_key(item.target.symbol) for item in ranked
                }
            ]
        requested_scope_keys = {
            _target_symbol_key(symbol) for symbol in executable_scope_symbols
        }
        observed_scope_keys = {
            _target_symbol_key(symbol)
            for collection in ordered_scope_collections.values()
            for symbol in collection
        }
        unexpected_scope_symbols = sorted(
            observed_scope_keys - requested_scope_keys
        )
        explicit_scope_conservation_ledger = {
            "schema_version": "ma1_explicit_scope_conservation_ledger/v1",
            "authoritative": True,
            "target_scope_mode": (
                "explicit_only" if executable_scope_symbols else "portfolio"
            ),
            "request_identity_is_query_scope_not_scientific_support": True,
            "target_fallback_used": False if executable_scope_symbols else fallback_used,
            "unexpected_generated_symbols": unexpected_scope_symbols,
            "conservation_valid": not unexpected_scope_symbols,
            "collections": {
                name: {
                    "count": len(values),
                    "symbols": values,
                    "order_sha256": _canonical_sha256(values),
                    "set_sha256": _canonical_sha256(
                        sorted(_target_symbol_key(item) for item in values)
                    ),
                }
                for name, values in ordered_scope_collections.items()
            },
            "identity_rejections": [
                {
                    "requested_symbol": item.get("requested_symbol"),
                    "reason_code": item.get("reason_code"),
                    "origin": item.get("origin"),
                }
                for item in target_catalog.request_scope_identity_trace
                if item.get("disposition") == "rejected"
            ],
        }
        explicit_scope_conservation_ledger["ledger_sha256"] = (
            _canonical_sha256(explicit_scope_conservation_ledger)
        )
        return HypothesisPacket(
            packet_id=_stable_id(
                "hp", brief.brief_id, packet_key, exclusion_key, iterations
            ),
            brief_id=brief.brief_id,
            hypotheses=[] if provisional_admission else ranked,
            provisional_hypotheses=ranked if provisional_admission else [],
            excluded=excluded,
            iterations=max(1, iterations),
            metadata={
                **({
                    "admission_policy": _MA3_ADMISSION_POLICY,
                    "ma1_provisional_hypothesis_count": len(ranked),
                    "ma1_submitted_hypothesis_ids": [item.hypothesis_id for item in ranked],
                    "ma1_scientific_approval_authority": "MA3",
                    "broad_direction_question_policy": "per_allowed_direction_unapproved_questions_v1",
                    "broad_direction_max_questions_per_target_scope": 3,
                    "broad_direction_cost_scope": "up to three MA3 reviews instead of one; existing run budgets unchanged",
                } if provisional_admission else {}),
                "target_discovery_enabled": bool(
                    self.settings.dynamic_target_discovery_enabled
                ),
                "resolved_disease": (
                    dict(target_catalog.resolved_disease)
                    if target_catalog.resolved_disease is not None
                    else None
                ),
                "target_source_statuses": [
                    dict(item) for item in target_catalog.source_statuses
                ],
                "target_discovery_error": target_catalog.discovery_error,
                "disease_resolution_status": (
                    target_catalog.disease_resolution_status
                ),
                "demo_fixture_fallback_authorized": (
                    target_catalog.demo_fixture_authorized
                ),
                "target_fallback_used": fallback_used,
                "external_target_seed_count": target_catalog.external_seed_count,
                "fallback_target_seed_count": target_catalog.fallback_seed_count,
                "ma1_executable_target_scope_applied": bool(
                    executable_scope_keys
                ),
                "ma1_executable_target_scope_symbols": list(
                    executable_scope_symbols
                ),
                "ma1_scope_candidate_pool_count": (
                    target_catalog.scope_candidate_pool_count
                ),
                "ma1_scope_candidate_match_count": (
                    target_catalog.scope_candidate_match_count
                ),
                "ma1_scope_identity_match_count": (
                    target_catalog.scope_identity_match_count
                ),
                "ma1_request_scope_identity_lane_used": (
                    target_catalog.request_scope_identity_lane_used
                ),
                "ma1_request_scope_identity_query_count": (
                    target_catalog.request_scope_identity_query_count
                ),
                "ma1_request_scope_identity_seed_count": (
                    target_catalog.request_scope_identity_seed_count
                ),
                "ma1_request_scope_identity_trace": [
                    dict(item)
                    for item in target_catalog.request_scope_identity_trace
                ],
                "ma1_request_scope_identity_trace_sha256": (
                    _canonical_sha256(
                        target_catalog.request_scope_identity_trace
                    )
                ),
                "ma1_explicit_scope_conservation_ledger": (
                    explicit_scope_conservation_ledger
                ),
                "ma1_discovery_requested_max_candidates": (
                    target_catalog.discovery_requested_max_candidates
                ),
                "ma1_screening_candidate_budget": screening_candidate_budget,
                "ma1_screening_per_track_capacity": screening_per_track_limit,
                "ma1_supplementary_evidence_retry_budget_configured": (
                    configured_supplementary_retry_budget
                ),
                "ma1_supplementary_evidence_retry_budget_effective": (
                    effective_supplementary_retry_budget
                ),
                "ma1_total_screening_candidate_budget": (
                    screening_candidate_budget
                    + effective_supplementary_retry_budget
                ),
                "ma1_supplementary_evidence_retry_policy": (
                    "fresh_window_preserved_plus_bounded_least_attempted_retry_v2"
                ),
                "ma1_scheduled_target_count": len(scheduled_proposal_symbols),
                "ma1_screened_target_count": len(
                    scheduled_proposal_symbols
                    & (accepted_symbols | excluded_symbols)
                ),
                "ma1_search_candidate_count": (
                    target_catalog.discovery_candidate_trace_total
                ),
                "ma1_screened_candidate_count": len(
                    scheduled_proposal_symbols
                    & (accepted_symbols | excluded_symbols)
                ),
                "ma1_executable_hypothesis_count": 0 if provisional_admission else len(ranked),
                "ma1_executable_admission_policy": (
                    _MA3_ADMISSION_POLICY if provisional_admission else
                    "critic_acceptance_requires_linked_direct_target_direction_primary_"
                    "and_primary_target_disease_association_v1"
                ),
                "ma1_pool_contract": {
                    "schema_version": "search-screen-executable-pools-v1",
                    "search_pool_role": "search_only_not_executable",
                    "search_stage_direction_semantics": (
                        "not_assessed_search_stage"
                    ),
                    "screening_direction_triage_policy": (
                        "persisted_claim_linkage_only_no_biological_inference_v1"
                    ),
                    "executable_pool_role": ("ma3_approved_hypotheses_only"
                                             if provisional_admission else "retained_hypotheses_only"),
                    **({"provisional_hypothesis_count": len(ranked),
                        "provisional_pool_role": "review_only_not_executable"}
                       if provisional_admission else {}),
                    "search_candidate_count": (
                        target_catalog.discovery_candidate_trace_total
                    ),
                    "screened_candidate_count": len(
                        scheduled_proposal_symbols
                        & (accepted_symbols | excluded_symbols)
                    ),
                    "executable_hypothesis_count": 0 if provisional_admission else len(ranked),
                },
                "ma1_discovery_rework_expansion": (
                    target_catalog.discovery_rework_expansion
                ),
                "ma1_discovery_candidate_trace_total": (
                    target_catalog.discovery_candidate_trace_total
                ),
                "ma1_discovery_candidate_trace_recorded": len(
                    effective_discovery_candidate_trace
                ),
                "ma1_discovery_candidate_trace_omitted": (
                    target_catalog.discovery_candidate_trace_omitted
                ),
                "ma1_discovery_candidate_trace": [
                    dict(item)
                    for item in effective_discovery_candidate_trace
                ],
                "ma1_scope_evidence_missing_fail_closed": (
                    scope_evidence_missing
                ),
                "ma1_scope_prior_exclusion_count": (
                    prior_scope_exclusion_count
                ),
                "ma1_scope_selector_exclusion_count": (
                    selector_scope_exclusion_count
                ),
                "prior_claim_evidence_reclassified_count": (
                    reclassified_prior_count
                ),
                "prior_claim_evidence_retry_symbols": sorted(
                    attempted_prior_evidence_retry_symbols
                ),
                "prior_evidence_retry_requested_count": len(
                    requested_prior_evidence_retry_symbols
                ),
                "prior_evidence_retry_requested_symbols": sorted(
                    requested_prior_evidence_retry_symbols
                )[:_SCREENING_TRACE_ENTRY_LIMIT],
                "prior_evidence_retry_scheduled_symbols": sorted(
                    prior_evidence_retry_symbols
                ),
                "prior_evidence_retry_catalog_seed_restored_symbols": sorted(
                    restored_prior_evidence_retry_symbols
                )[:_SCREENING_TRACE_ENTRY_LIMIT],
                "prior_evidence_retry_catalog_seed_restored_count": len(
                    restored_prior_evidence_retry_symbols
                ),
                "prior_evidence_retry_restored_symbols": sorted(
                    restored_prior_evidence_retry_symbols
                )[:_SCREENING_TRACE_ENTRY_LIMIT],
                "prior_evidence_retry_restored_symbols_semantics": (
                    "deprecated alias for "
                    "prior_evidence_retry_catalog_seed_restored_symbols; "
                    "restoration does not mean scheduled or attempted"
                ),
                "prior_evidence_retry_catalog_missing_symbols": sorted(
                    catalog_missing_prior_evidence_retry_symbols
                )[:_SCREENING_TRACE_ENTRY_LIMIT],
                "prior_evidence_retry_budget_unscheduled_symbols": sorted(
                    budget_unscheduled_prior_evidence_retry_symbols
                )[:_SCREENING_TRACE_ENTRY_LIMIT],
                "prior_evidence_retry_unattempted_symbols": sorted(
                    unattempted_prior_evidence_retry_symbols
                ),
                "prior_evidence_retry_status_count_total": len(
                    requested_prior_evidence_retry_symbols
                ),
                "prior_evidence_retry_statuses": [
                    {
                        "target_symbol": symbol,
                        "status": (
                            "attempted"
                            if symbol in attempted_prior_evidence_retry_symbols
                            else "catalog_missing"
                            if symbol
                            in catalog_missing_prior_evidence_retry_symbols
                            else "budget_not_scheduled"
                            if symbol
                            in budget_unscheduled_prior_evidence_retry_symbols
                            else "not_attempted"
                        ),
                    }
                    for symbol in recorded_retry_status_symbols
                ],
                "prior_evidence_retry_state_records": (
                    prior_evidence_retry_state_records
                ),
                "prior_evidence_retry_execution_summary": (
                    prior_evidence_retry_execution_summary
                ),
                "ma1_evidence_retry_attempt_ledger": {
                    "schema_version": "ma1_evidence_retry_attempt_ledger/v1",
                    "authoritative": True,
                    "carry_forward_source": retry_attempt_ledger_source,
                    "legacy_history_bootstrap_used": (
                        retry_attempt_ledger_source
                        == "legacy_history_bootstrap"
                    ),
                    "entry_limit": (
                        _EVIDENCE_RETRY_ATTEMPT_LEDGER_ENTRY_LIMIT
                    ),
                    "entry_count_total": len(
                        cumulative_evidence_retry_attempt_counts
                    ),
                    "entry_count_recorded": len(retry_attempt_ledger_entries),
                    "entry_count_omitted": 0,
                    "current_revision_incremented_symbols": sorted(
                        attempted_prior_evidence_retry_symbols
                    ),
                    "attempt_count_semantics": (
                        "incremented once per target only after an actual MA1 "
                        "evidence-retry proposal was attempted"
                    ),
                    "entries": retry_attempt_ledger_entries,
                },
                "prior_evidence_retry_allow_list": sorted(
                    _EVIDENCE_RETRY_REASON_CODES
                ),
                "prior_evidence_retry_exclusion_count_total": sum(
                    _target_symbol_key(item.target_symbol)
                    in attempted_prior_evidence_retry_symbols
                    for item in prior_exclusions
                ),
                "prior_evidence_retry_exclusions": [
                    {
                        "target_symbol": item.target_symbol,
                        "reason_codes": list(item.reason_codes),
                        "source_iteration": item.source_iteration,
                    }
                    for item in prior_exclusions
                    if _target_symbol_key(item.target_symbol)
                    in attempted_prior_evidence_retry_symbols
                ][:_SCREENING_TRACE_ENTRY_LIMIT],
                "target_sources_scope": ("provisional_review_only"
                                         if provisional_admission else "retained_only"),
                "target_sources": target_sources,
                "screened_target_source_count_total": (
                    screened_target_source_count_total
                ),
                "screened_target_source_count_recorded": len(
                    recorded_screened_sources
                ),
                "screened_target_source_count_omitted": max(
                    0,
                    screened_target_source_count_total
                    - len(recorded_screened_sources),
                ),
                "screened_target_sources": recorded_screened_sources,
                "ma1_revision_attempt": revision_attempt,
                "ma1_screening_history_source_count_total": (
                    screening_history_source_count_total
                ),
                "ma1_screening_history_source_count_recorded": len(
                    screening_history_sources
                ),
                "ma1_screening_history_source_count_omitted": max(
                    0,
                    screening_history_source_count_total
                    - len(screening_history_sources),
                ),
                "ma1_screening_history_sources": screening_history_sources,
                "ma1_rejected_screening_history": rejected_screening_history,
                "ma1_exclusion_provenance_total": exclusion_provenance_total,
                "ma1_exclusion_provenance_recorded": len(exclusion_provenance),
                "ma1_exclusion_provenance_omitted": max(
                    0,
                    exclusion_provenance_total - len(exclusion_provenance),
                ),
                "ma1_exclusion_provenance": exclusion_provenance,
                "rejected_screening_trace": screening_trace,
                "screening_trace_sha256": screening_trace_digest,
                "evidence_linkage_trace": {
                    "policy": "hypothesis_scoped_evidence_linkage_trace_v1",
                    "entry_count_total": evidence_linkage_entry_count,
                    "entry_count_recorded": len(evidence_linkage_entries),
                    "entry_count_omitted": max(0, evidence_linkage_entry_count - len(evidence_linkage_entries)),
                    "direction_rejection_counts": evidence_linkage_reason_counts,
                    "entries": evidence_linkage_entries,
                    "diagnostic_only": True,
                },
            },
        )

    def _prepare_target_catalog(
        self,
        brief: ResearchBrief,
        *,
        per_track_limit: int,
        prior_attempt_count: int = 0,
        prior_attempt_symbols: Sequence[str] = (),
        effective_max_iterations: int | None = None,
    ) -> TargetCatalogResult:
        goal_focus = derive_goal_focus(brief)
        scope_symbols = (
            tuple(dict.fromkeys(goal_focus.executable_target_mentions))
            if goal_focus.target_scope_mode == "explicit_only"
            else ()
        )
        scope_keys = frozenset(
            _target_symbol_key(symbol)
            for symbol in scope_symbols
            if _target_symbol_key(symbol)
        )
        forbidden_scope_keys = frozenset(
            _target_symbol_key(symbol)
            for symbol in brief.constraints.forbidden_targets
            if _target_symbol_key(symbol) in scope_keys
        )
        dynamic_enabled = bool(self.settings.dynamic_target_discovery_enabled)
        explicit_demo_fallback = bool(
            self.settings.demo_fixture_fallback_enabled
        )
        # Direct MA1 component demos historically omit a discovery adapter or
        # disable dynamic discovery altogether.  Preserve those isolated
        # fixtures, while the orchestrator independently refuses to advance
        # them unless the explicit demo flag is enabled.
        component_fixture_mode = not dynamic_enabled or self.target_discovery is None
        fallback_enabled = bool(
            (
                self.settings.mode is RunMode.MOCK
                and component_fixture_mode
            )
            or (
                self.settings.dynamic_target_fallback_enabled
                and (explicit_demo_fallback or component_fixture_mode)
            )
        ) and not scope_keys
        legacy_catalog = self.proposer._catalog(brief)
        resolved_disease: Mapping[str, Any] | None = None
        source_statuses: tuple[Mapping[str, Any], ...] = ()
        discovery_error: str | None = None
        disease_resolution_status = "not_attempted"
        dynamic_seeds: list[TargetSeed] = []
        scope_candidate_pool_count = 0
        scope_candidate_match_count = 0
        scope_dynamic_match_symbols: tuple[str, ...] = ()
        discovery_candidate_trace: list[Mapping[str, Any]] = []
        discovery_candidate_trace_total = 0
        request_scope_identity_trace: list[dict[str, Any]] = []
        request_scope_identity_query_count = 0
        rework_expansion = (
            min(max(0, prior_attempt_count), 30 - self.settings.dynamic_target_max_candidates)
            if prior_attempt_count and not scope_keys
            else 0
        )
        requested_max_candidates = min(
            30,
            self.settings.dynamic_target_max_candidates + rework_expansion,
        )

        if dynamic_enabled and self.target_discovery is not None:
            try:
                discovery = self.target_discovery.discover(
                    brief.disease,
                    brief.disease_id,
                    max_targets=requested_max_candidates,
                    allowed_sources=brief.data_policy.allowed_sources,
                )
                source_statuses = tuple(
                    item.model_dump(mode="json") for item in discovery.source_statuses
                )
                resolution = discovery.resolved_disease
                if (
                    resolution is not None
                    and resolution.match_method
                    in {
                        "provided_id",
                        "exact_name",
                        "exact_synonym",
                        "unique_high_similarity",
                    }
                ):
                    resolved_disease = resolution.model_dump(mode="json")
                    disease_resolution_status = "resolved"
                    discovered_candidates = tuple(discovery.candidates)
                    scope_candidate_pool_count = len(discovered_candidates)
                    discovery_candidate_trace_total = len(discovered_candidates)
                    for candidate in discovered_candidates[:_SCREENING_TRACE_ENTRY_LIMIT]:
                        scope_match = bool(
                            scope_keys
                            and _target_symbol_key(candidate.gene_symbol) in scope_keys
                        )
                        readiness_score, readiness_basis = (
                            _discovery_intervention_readiness(
                                candidate,
                                brief.constraints.intervention_direction,
                            )
                        )
                        discovery_candidate_trace.append(
                            {
                                "rank": candidate.rank,
                                "target_symbol": candidate.gene_symbol,
                                "target_name": candidate.target_name,
                                "ensembl_id": candidate.ensembl_target_id,
                                "association_score": candidate.association_score,
                                "track_hint": candidate.track_hint,
                                "primary_source": candidate.provenance.primary_source,
                                "contributing_sources": list(
                                    candidate.provenance.contributing_sources
                                ),
                                "intervention_readiness_score": readiness_score,
                                "intervention_readiness_basis": readiness_basis,
                                "direction_alignment_status": (
                                    "not_established_at_discovery"
                                ),
                                "direction_evaluation_status": (
                                    "not_assessed_search_stage"
                                ),
                                "execution_disposition": (
                                    "search_only_not_executable"
                                ),
                                "scope_disposition": (
                                    "scope_match"
                                    if scope_match
                                    else "outside_explicit_scope"
                                    if scope_keys
                                    else "portfolio_candidate"
                                ),
                                "reason_codes": (
                                    []
                                    if scope_match or not scope_keys
                                    else ["OUTSIDE_EXECUTABLE_TARGET_SCOPE"]
                                ),
                                "scientific_rejection": False,
                            }
                        )
                    scoped_candidates = (
                        tuple(
                            candidate
                            for candidate in discovered_candidates
                            if _target_symbol_key(candidate.gene_symbol)
                            in scope_keys
                        )
                        if scope_keys
                        else discovered_candidates
                    )
                    scope_candidate_match_count = len(scoped_candidates)
                    scoped_candidate_keys = {
                        _target_symbol_key(candidate.gene_symbol)
                        for candidate in scoped_candidates
                    }
                    scope_dynamic_match_symbols = tuple(
                        symbol
                        for symbol in scope_symbols
                        if _target_symbol_key(symbol) in scoped_candidate_keys
                    )
                    prior_attempt_keys = {
                        _target_symbol_key(symbol)
                        for symbol in prior_attempt_symbols
                        if _target_symbol_key(symbol)
                    }
                    fresh_priority_keys: frozenset[str] = frozenset()
                    if prior_attempt_keys and not scope_keys:
                        # A G0 rework must explore fresh disease-ranked targets
                        # before spending every bounded slot on the identical
                        # unsupported portfolio. Previous targets remain in the
                        # catalog after fresh candidates for an auditable retry.
                        scoped_candidates = tuple(
                            candidate
                            for candidate in scoped_candidates
                            if _target_symbol_key(candidate.gene_symbol)
                            not in prior_attempt_keys
                        ) + tuple(
                            candidate
                            for candidate in scoped_candidates
                            if _target_symbol_key(candidate.gene_symbol)
                            in prior_attempt_keys
                        )
                        fresh_priority_keys = frozenset(
                            _target_symbol_key(candidate.gene_symbol)
                            for candidate in scoped_candidates
                            if _target_symbol_key(candidate.gene_symbol)
                            not in prior_attempt_keys
                        )
                    assignments = _balanced_candidate_assignments(
                        scoped_candidates,
                        per_track_limit=per_track_limit,
                        total_limit=self.settings.dynamic_target_max_candidates,
                        priority_symbols=fresh_priority_keys,
                        intervention_direction=(
                            brief.constraints.intervention_direction
                        ),
                    )
                    dynamic_seeds = [
                        _seed_from_discovery_candidate(
                            brief,
                            assignment.candidate,
                            assigned_track=assignment.track,
                            assignment_basis=assignment.assignment_basis,
                            source_statuses=source_statuses,
                        )
                        for assignment in assignments
                    ]
                    scheduled_keys = {
                        _target_symbol_key(assignment.candidate.gene_symbol)
                        for assignment in assignments
                    }
                    discovery_candidate_trace = [
                        {
                            **dict(item),
                            "screening_allocation": (
                                "outside_explicit_scope"
                                if item["scope_disposition"]
                                == "outside_explicit_scope"
                                else "scheduled_for_ma1"
                                if _target_symbol_key(
                                    str(item["target_symbol"])
                                )
                                in scheduled_keys
                                else "previous_revision_screened_result_preserved"
                                if _target_symbol_key(
                                    str(item["target_symbol"])
                                )
                                in prior_attempt_keys
                                else "not_scheduled_iteration_budget"
                            ),
                            "reason_codes": (
                                item["reason_codes"]
                                if item["reason_codes"]
                                else []
                                if _target_symbol_key(
                                    str(item["target_symbol"])
                                )
                                in scheduled_keys
                                else ["MA1_PREVIOUS_REVISION_SCREENING_PRESERVED"]
                                if _target_symbol_key(
                                    str(item["target_symbol"])
                                )
                                in prior_attempt_keys
                                else ["MA1_SCREENING_BUDGET_NOT_SCHEDULED"]
                            ),
                        }
                        for item in discovery_candidate_trace
                    ]
                else:
                    open_targets = next(
                        (
                            item
                            for item in source_statuses
                            if item.get("source") == "Open Targets"
                        ),
                        None,
                    )
                    source_state = (
                        str(open_targets.get("status"))
                        if isinstance(open_targets, Mapping)
                        else "failed"
                    )
                    disease_resolution_status = (
                        "source_unavailable"
                        if source_state in {"failed", "partial", "skipped"}
                        else "unresolved"
                    )
            except Exception as exc:
                discovery_error = type(exc).__name__
                disease_resolution_status = "source_unavailable"
        elif dynamic_enabled:
            discovery_error = "target_discovery_not_configured"
            disease_resolution_status = "source_unavailable"

        identity_resolution_available = bool(
            self.settings.target_identity_resolution_enabled
            and self.identity_resolver is not None
        )
        if scope_keys:
            strict_dynamic_seeds: list[TargetSeed] = []
            for seed in dynamic_seeds:
                requested_symbol = next(
                    (
                        symbol
                        for symbol in scope_symbols
                        if _target_symbol_key(symbol)
                        == _target_symbol_key(seed.symbol)
                    ),
                    seed.symbol,
                )
                if _target_symbol_key(requested_symbol) in forbidden_scope_keys:
                    request_scope_identity_trace.append(
                        _explicit_scope_identity_trace_row(
                            requested_symbol=requested_symbol,
                            origin="disease_ranked_exact_match",
                            query_attempted=False,
                            seed=None,
                            reason_code="EXPLICIT_TARGET_FORBIDDEN",
                        )
                    )
                    continue
                resolved_seed = (
                    self._resolve_seed_identity(
                        seed,
                        allowed_sources=brief.data_policy.allowed_sources,
                    )
                    if identity_resolution_available
                    else seed
                )
                reason_code = _strict_explicit_scope_identity_reason(
                    resolved_seed,
                    requested_symbol=requested_symbol,
                )
                request_scope_identity_trace.append(
                    _explicit_scope_identity_trace_row(
                        requested_symbol=requested_symbol,
                        origin="disease_ranked_exact_match",
                        query_attempted=identity_resolution_available,
                        seed=resolved_seed,
                        reason_code=reason_code,
                    )
                )
                if reason_code is None:
                    strict_dynamic_seeds.append(resolved_seed)
            dynamic_seeds = strict_dynamic_seeds

            discovered_scope_keys = {
                _target_symbol_key(symbol)
                for symbol in scope_dynamic_match_symbols
            }
            missing_scope_symbols = tuple(
                symbol
                for symbol in scope_symbols
                if _target_symbol_key(symbol) not in discovered_scope_keys
            )
            track_counts = {
                track: sum(seed.track is track for seed in dynamic_seeds)
                for track in HypothesisTrack
            }
            for requested_symbol in missing_scope_symbols:
                symbol_key = _target_symbol_key(requested_symbol)
                if symbol_key in forbidden_scope_keys:
                    request_scope_identity_trace.append(
                        _explicit_scope_identity_trace_row(
                            requested_symbol=requested_symbol,
                            origin="explicit_request_scope_identity",
                            query_attempted=False,
                            seed=None,
                            reason_code="EXPLICIT_TARGET_FORBIDDEN",
                        )
                    )
                    continue
                # An explicitly requested target is an independent identity
                # lookup lane.  Disease resolution controls disease-ranked
                # discovery, but must not suppress canonical target identity
                # resolution.  The resulting seed still carries no disease
                # association or direction support; those claims remain
                # subject to the ordinary evidence screens below.
                assigned_track = min(
                    HypothesisTrack,
                    key=lambda track: (
                        track_counts[track],
                        0 if track is HypothesisTrack.LITERATURE else 1,
                    ),
                )
                request_seed = TargetSeed(
                    symbol=requested_symbol,
                    target_name=requested_symbol,
                    track=assigned_track,
                    priority=0.0,
                    identity_status="unresolved",
                    discovery_source="explicit_request_scope_identity",
                    discovery_evidence=(),
                    discovery_statuses=source_statuses,
                    discovery_metadata={
                        "request_scope_identity_lane": True,
                        "request_scope_symbol": requested_symbol,
                        "assignment_basis": (
                            "request_order_balanced_scheduling_only"
                        ),
                        "tdl_interpretation": "unknown_not_inferred",
                        "association_evidence_created": False,
                        "target_direction_evidence_created": False,
                    },
                )
                if identity_resolution_available:
                    request_scope_identity_query_count += 1
                    request_seed = self._resolve_seed_identity(
                        request_seed,
                        allowed_sources=brief.data_policy.allowed_sources,
                    )
                    reason_code = _strict_explicit_scope_identity_reason(
                        request_seed,
                        requested_symbol=requested_symbol,
                    )
                else:
                    reason_code = "EXPLICIT_TARGET_IDENTITY_RESOLVER_UNAVAILABLE"
                request_scope_identity_trace.append(
                    _explicit_scope_identity_trace_row(
                        requested_symbol=requested_symbol,
                        origin="explicit_request_scope_identity",
                        query_attempted=identity_resolution_available,
                        seed=request_seed,
                        reason_code=reason_code,
                    )
                )
                if reason_code is None:
                    dynamic_seeds.append(request_seed)
                    track_counts[assigned_track] += 1
        elif identity_resolution_available:
            dynamic_seeds = [
                self._resolve_seed_identity(
                    seed,
                    allowed_sources=brief.data_policy.allowed_sources,
                )
                for seed in dynamic_seeds
            ]

        scope_identity_match_count = len(dynamic_seeds) if scope_keys else 0

        # A canonical target can appear in only one supply track.  Preserve the
        # discovery ranking and prefer the first evidence-supported assignment.
        seen_symbols: set[str] = set()
        seen_ensembl: set[str] = set()
        seen_uniprot: set[str] = set()
        by_track: dict[HypothesisTrack, list[TargetSeed]] = {
            HypothesisTrack.LITERATURE: [],
            HypothesisTrack.TBIO: [],
        }
        for seed in dynamic_seeds:
            symbol_key = _target_symbol_key(seed.symbol)
            ensembl_key = (seed.ensembl_id or "").casefold()
            uniprot_key = (seed.uniprot_id or "").casefold()
            if (
                symbol_key in seen_symbols
                or (ensembl_key and ensembl_key in seen_ensembl)
                or (uniprot_key and uniprot_key in seen_uniprot)
            ):
                continue
            if sum(len(items) for items in by_track.values()) >= int(
                self.settings.dynamic_target_max_candidates
            ):
                continue
            if len(by_track[seed.track]) >= per_track_limit:
                continue
            by_track[seed.track].append(seed)
            seen_symbols.add(symbol_key)
            if ensembl_key:
                seen_ensembl.add(ensembl_key)
            if uniprot_key:
                seen_uniprot.add(uniprot_key)

        if fallback_enabled:
            # Fixture/demo fillers preserve their historical per-round bound.
            # The larger live screening capacity must never manufacture extra
            # fallback targets merely to fill a public-discovery portfolio.
            fallback_per_track_limit = min(
                per_track_limit,
                max(
                    1,
                    int(
                        effective_max_iterations
                        if effective_max_iterations is not None
                        else min(
                            3,
                            self.settings.max_ma1_iterations,
                            brief.constraints.max_ma1_iterations,
                        )
                    ),
                ),
            )
            for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO):
                for fallback in legacy_catalog[track]:
                    if len(by_track[track]) >= fallback_per_track_limit:
                        break
                    symbol_key = _target_symbol_key(fallback.symbol)
                    if symbol_key in seen_symbols:
                        continue
                    seed = replace(
                        fallback,
                        identity_status="fixture",
                        identity_sources=("fixture",),
                        discovery_source="fixture_fallback",
                        discovery_statuses=source_statuses,
                        discovery_metadata={
                            "fallback_reason": discovery_error
                            or "insufficient_track_specific_dynamic_candidates",
                            "requires_live_verification": True,
                        },
                    )
                    if (
                        self.settings.target_identity_resolution_enabled
                        and self.identity_resolver
                    ):
                        seed = self._resolve_seed_identity(
                            seed,
                            allowed_sources=brief.data_policy.allowed_sources,
                        )
                    by_track[track].append(seed)
                    seen_symbols.add(_target_symbol_key(seed.symbol))
                    if seed.ensembl_id:
                        seen_ensembl.add(seed.ensembl_id.casefold())

        scheduled_identity_keys = {
            _target_symbol_key(seed.symbol)
            for items in by_track.values()
            for seed in items
        }
        scheduled_identity_ids = {
            (seed.ensembl_id or "").casefold()
            for items in by_track.values()
            for seed in items
            if seed.ensembl_id
        }
        finalized_identity_trace: list[dict[str, Any]] = []
        for source_row in request_scope_identity_trace:
            row = dict(source_row)
            row.pop("row_sha256", None)
            resolved_key = _target_symbol_key(row.get("canonical_symbol"))
            resolved_ensembl = str(row.get("ensembl_id") or "").casefold()
            if not row.get("seed_created"):
                scheduling_disposition = "not_eligible"
            elif (
                resolved_key in scheduled_identity_keys
                and resolved_ensembl in scheduled_identity_ids
            ):
                scheduling_disposition = "scheduled_for_ma1"
            else:
                scheduling_disposition = "unscheduled_capacity_or_duplicate"
            row["scheduling_disposition"] = scheduling_disposition
            row["row_sha256"] = _canonical_sha256(row)
            finalized_identity_trace.append(row)

        return TargetCatalogResult(
            catalog={track: tuple(items) for track, items in by_track.items()},
            resolved_disease=resolved_disease,
            source_statuses=source_statuses,
            discovery_error=discovery_error,
            disease_resolution_status=disease_resolution_status,
            demo_fixture_authorized=explicit_demo_fallback,
            external_seed_count=sum(
                seed.discovery_source == "external_dynamic"
                for items in by_track.values()
                for seed in items
            ),
            fallback_seed_count=sum(
                seed.discovery_source == "fixture_fallback"
                for items in by_track.values()
                for seed in items
            ),
            executable_scope_applied=bool(scope_keys),
            executable_scope_symbols=scope_symbols,
            scope_candidate_pool_count=scope_candidate_pool_count,
            scope_candidate_match_count=scope_candidate_match_count,
            scope_identity_match_count=scope_identity_match_count,
            discovery_candidate_trace=tuple(discovery_candidate_trace),
            discovery_candidate_trace_total=discovery_candidate_trace_total,
            discovery_candidate_trace_omitted=max(
                0,
                discovery_candidate_trace_total - len(discovery_candidate_trace),
            ),
            discovery_requested_max_candidates=requested_max_candidates,
            request_scope_identity_lane_used=(
                request_scope_identity_query_count > 0
            ),
            request_scope_identity_query_count=(
                request_scope_identity_query_count
            ),
            request_scope_identity_seed_count=sum(
                seed.discovery_source
                == "explicit_request_scope_identity"
                for items in by_track.values()
                for seed in items
            ),
            request_scope_identity_trace=tuple(finalized_identity_trace),
            scope_dynamic_match_symbols=scope_dynamic_match_symbols,
            discovery_rework_expansion=rework_expansion,
        )

    def _resolve_seed_identity(
        self,
        seed: TargetSeed,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> TargetSeed:
        assert self.identity_resolver is not None
        try:
            resolution = self.identity_resolver.resolve(
                seed.symbol,
                seed.ensembl_id,
                allowed_sources=allowed_sources,
            )
        except Exception as exc:  # discovery identity remains usable on resolver failure
            status = "partial" if seed.ensembl_id else "unresolved"
            return replace(
                seed,
                identity_status=status,
                identity_sources=seed.identity_sources,
                identity_metadata={
                    **dict(seed.identity_metadata),
                    "resolver_error": type(exc).__name__,
                },
            )

        identity = resolution.identity
        canonical_symbol = (
            identity.canonical_symbol if identity and identity.canonical_symbol else seed.symbol
        )
        ensembl_id = (
            identity.ensembl_id if identity and identity.ensembl_id else seed.ensembl_id
        )
        uniprot_id = (
            identity.uniprot_accession
            if identity and identity.uniprot_accession
            else seed.uniprot_id
        )
        target_name = (
            identity.protein_name if identity and identity.protein_name else seed.target_name
        )
        source_names = tuple(dict.fromkeys(item.source for item in resolution.sources))
        status = resolution.status
        if status == "unresolved" and ensembl_id:
            # The discovery association still supplies a traceable Ensembl ID,
            # even though the secondary resolver could not enrich it.
            status = "partial"
            source_names = tuple(
                dict.fromkeys((*seed.identity_sources, "Open Targets (discovery)"))
            )
        return replace(
            seed,
            symbol=canonical_symbol,
            target_name=target_name,
            ensembl_id=ensembl_id,
            uniprot_id=uniprot_id,
            identity_status=status,
            identity_sources=source_names or seed.identity_sources,
            identity_metadata={
                "status": resolution.status,
                "ready_for_structure_lookup": resolution.ready_for_structure_lookup,
                "canonical_symbol": (
                    identity.canonical_symbol if identity is not None else None
                ),
                "ensembl_id": (
                    identity.ensembl_id if identity is not None else None
                ),
                "uniprot_accession": (
                    identity.uniprot_accession
                    if identity is not None
                    else None
                ),
                "organism": (
                    identity.organism if identity is not None else None
                ),
                "taxon_id": (
                    identity.taxon_id if identity is not None else None
                ),
                "warnings": list(resolution.warnings),
                "sources": [item.model_dump(mode="json") for item in resolution.sources],
            },
        )

    async def arun(
        self,
        brief: ResearchBrief,
        revision_context: object | None = None,
    ) -> HypothesisPacket:
        return self.run(brief, revision_context=revision_context)


def _balanced_candidate_assignments(
    candidates: Sequence[DiseaseTargetCandidate],
    *,
    per_track_limit: int,
    total_limit: int | None = None,
    priority_symbols: frozenset[str] = frozenset(),
    intervention_direction: str = "modulate",
) -> list[_TrackAssignment]:
    """Assign unique dynamic targets while preserving explicit hints first.

    Open Targets associations commonly receive a Tbio hint because they carry
    genetics/pathway evidence.  Candidates beyond an explicit track's bounded
    capacity therefore remain available to the other, under-supplied track.
    Reallocation changes only the workflow supply track; it does not change the
    underlying evidence relationship or assert a new biological conclusion.
    """

    ordered = sorted(
        candidates,
        key=lambda item: (
            0
            if _target_symbol_key(item.gene_symbol) in priority_symbols
            else 1,
            -_discovery_intervention_readiness(
                item,
                intervention_direction,
            )[0],
            item.rank,
            -(item.association_score or 0.0),
            _target_symbol_key(item.gene_symbol),
            item.ensembl_target_id,
        ),
    )
    unique: list[DiseaseTargetCandidate] = []
    seen_symbols: set[str] = set()
    seen_ensembl: set[str] = set()
    for candidate in ordered:
        symbol_key = _target_symbol_key(candidate.gene_symbol)
        ensembl_key = candidate.ensembl_target_id.casefold()
        if symbol_key in seen_symbols or ensembl_key in seen_ensembl:
            continue
        unique.append(candidate)
        seen_symbols.add(symbol_key)
        seen_ensembl.add(ensembl_key)

    by_track: dict[HypothesisTrack, list[_TrackAssignment]] = {
        HypothesisTrack.LITERATURE: [],
        HypothesisTrack.TBIO: [],
    }
    balanced_track_limit = min(
        per_track_limit,
        max(1, (len(unique) + len(HypothesisTrack) - 1) // len(HypothesisTrack)),
    )
    remaining: list[DiseaseTargetCandidate] = []
    for candidate in unique:
        explicit_track = {
            "literature": HypothesisTrack.LITERATURE,
            "tbio": HypothesisTrack.TBIO,
        }.get(candidate.track_hint)
        if (
            explicit_track is not None
            and len(by_track[explicit_track]) < balanced_track_limit
        ):
            by_track[explicit_track].append(
                _TrackAssignment(candidate, explicit_track, "explicit_track_hint")
            )
        else:
            remaining.append(candidate)

    for candidate in remaining:
        deficits = {
            track: balanced_track_limit - len(items)
            for track, items in by_track.items()
        }
        available = [track for track, deficit in deficits.items() if deficit > 0]
        if not available:
            break
        # Largest deficit first; Literature wins a tie to prevent a Tbio-heavy
        # Open Targets result from silently retaining generic Literature seeds.
        assigned_track = min(
            available,
            key=lambda track: (
                -deficits[track],
                0 if track is HypothesisTrack.LITERATURE else 1,
            ),
        )
        by_track[assigned_track].append(
            _TrackAssignment(candidate, assigned_track, "balanced_reallocation")
        )

    # Explicit assignments remain ahead of reallocations within each track;
    # each group itself preserves discovery rank.
    assignments = [
        assignment
        for track in (HypothesisTrack.LITERATURE, HypothesisTrack.TBIO)
        for assignment in by_track[track]
    ]
    if total_limit is None or len(assignments) <= max(1, int(total_limit)):
        return assignments
    retained_keys = {
        (
            _target_symbol_key(candidate.gene_symbol),
            candidate.ensembl_target_id.casefold(),
        )
        for candidate in unique[: max(1, int(total_limit))]
    }
    return [
        assignment
        for assignment in assignments
        if (
            _target_symbol_key(assignment.candidate.gene_symbol),
            assignment.candidate.ensembl_target_id.casefold(),
        )
        in retained_keys
    ]


def _discovery_intervention_readiness(
    candidate: DiseaseTargetCandidate,
    intervention_direction: str,
) -> tuple[int, list[str]]:
    """Rank retrievable intervention evidence without asserting direction.

    Association rank alone can spend a bounded MA1 window on targets that have
    no recorded intervention feasibility.  Open Targets clinical/legacy
    known-drug and tractability annotations are useful scheduling signals, but they do not
    prove that the requested action is beneficial.  They therefore affect only
    search order for inhibition requests; the downstream primary-publication
    direction gate remains authoritative.
    """

    if intervention_direction != "inhibit":
        return 0, ["direction_specific_readiness_not_available"]
    summary = candidate.evidence_summary
    clinical = (
        summary.clinical_evidence_score is not None
        and summary.clinical_evidence_score > 0
    )
    known_drug = (
        summary.known_drug_score is not None
        and summary.known_drug_score > 0
    )
    # ``clinical`` replaced ``known_drug`` in the current upstream schema.
    # They represent the same scheduling category and therefore count once.
    intervention_signal = clinical or known_drug
    tractable = any(item.value is True for item in summary.tractability)
    basis: list[str] = []
    if clinical:
        basis.append("open_targets_clinical_signal")
    elif known_drug:
        basis.append("open_targets_known_drug_signal")
    if tractable:
        basis.append("open_targets_tractability_signal")
    return int(intervention_signal) + int(tractable), basis or [
        "no_intervention_readiness_signal"
    ]


def _seed_from_discovery_candidate(
    brief: ResearchBrief,
    candidate: DiseaseTargetCandidate,
    *,
    assigned_track: HypothesisTrack,
    assignment_basis: str,
    source_statuses: tuple[Mapping[str, Any], ...],
) -> TargetSeed:
    evidence = tuple(
        _discovery_record_to_evidence(brief, candidate, record)
        for record in candidate.source_records
    )
    association_score = candidate.association_score
    priority = association_score if association_score is not None else 0.5
    return TargetSeed(
        symbol=candidate.gene_symbol,
        target_name=candidate.target_name or candidate.gene_symbol,
        track=assigned_track,
        priority=max(0.0, min(1.0, priority)),
        ensembl_id=candidate.ensembl_target_id,
        identity_status="partial",
        identity_sources=("Open Targets",),
        identity_metadata={
            "source": "disease_target_discovery",
            "ensembl_id_traceable": True,
        },
        discovery_source="external_dynamic",
        discovery_evidence=evidence,
        discovery_statuses=source_statuses,
        discovery_metadata={
            "rank": candidate.rank,
            "association_score": candidate.association_score,
            "track_hint": candidate.track_hint,
            "original_track_hint": candidate.track_hint,
            "track_hint_basis": candidate.track_hint_basis,
            "assigned_track": assigned_track.value,
            "assignment_basis": assignment_basis,
            "biotype": candidate.biotype,
            "evidence_summary": candidate.evidence_summary.model_dump(mode="json"),
            "provenance": candidate.provenance.model_dump(mode="json"),
            "causality_interpretation": candidate.provenance.causality_interpretation,
        },
    )


def _discovery_record_to_evidence(
    brief: ResearchBrief,
    candidate: DiseaseTargetCandidate,
    record: TargetSourceRecord,
) -> EvidenceRef:
    # Only Open Targets ranked associations are support.  PubMed title
    # co-mentions and target-identity records remain context and therefore can
    # never be promoted into a mechanism claim by this conversion.
    supports = "support" if record.relationship == "ranked_association" else "context"
    provenance_class = {
        "live": "external_live",
        "cache": "external_cache",
        "mixed": "external_mixed",
    }[record.retrieval_mode]
    return EvidenceRef(
        evidence_id=_stable_id(
            "ev",
            brief.brief_id,
            candidate.ensembl_target_id,
            record.source,
            record.record_id,
            record.relationship,
        ),
        source=record.source,
        external_id=record.record_id,
        title=record.title,
        url=record.url,
        retrieved_at=brief.created_at,
        supports=supports,
        excerpt_hash=sha256(record.title.encode("utf-8")).hexdigest(),
        metadata={
            **dict(record.metadata),
            "mock": False,
            "provenance_class": provenance_class,
            "relationship": record.relationship,
            "record_type": record.record_type,
            "query": record.query,
            "retrieval_endpoint": record.retrieval_endpoint,
            "retrieval_mode": record.retrieval_mode,
            "quality_score": record.score if record.score is not None else 0.5,
            "causality_scope": (
                "ranked association; not causal proof"
                if record.relationship == "ranked_association"
                else "context only; no causality inferred"
            ),
        },
    )


def _merge_evidence(items: Sequence[EvidenceRef]) -> list[EvidenceRef]:
    merged: list[EvidenceRef] = []
    seen_ids: set[str] = set()
    for item in items:
        if item.evidence_id in seen_ids:
            continue
        seen_ids.add(item.evidence_id)
        merged.append(item.model_copy(deep=True))
    return merged


_DIRECTION_PRIMARY_POLICY = (
    "atomic-publication-intervention-endpoint-linkage-v2"
)
_PRIMARY_EXCERPT_POLICY = (
    "same-sentence-target-disease-direction-aware-mechanistic-ranking-v2"
)
_COMPOUND_LINKED_DIRECTION_EXCERPT_POLICY = (
    "same-compound-target-selective-intervention-primary-outcome-pair-v1"
)
_PRIMARY_ASSOCIATION_EXCERPT_POLICY = (
    "same-sentence-target-disease-primary-association-ranking-v2"
)
_PRIMARY_QUERY_POLICY = "target-disease-direction-mechanistic-v2"
_PRIMARY_DIRECTION_SUPPLEMENT_QUERY_POLICY = (
    "brief-derived-target-inhibitor-positive-response-primary-v1"
)
_PRIMARY_DIRECTION_MODEL_QUERY_POLICY = (
    "brief-derived-disease-or-molecular-model-direction-primary-v2"
)
_PRIMARY_DIRECTION_EXPERIMENTAL_QUERY_POLICY = (
    "brief-derived-target-action-experimental-direction-primary-v3"
)
_PRIMARY_DIRECTION_MODEL_CONTEXT_BASIS = "predeclared_molecular_model"
_PRIMARY_DIRECTION_MODEL_CONTEXT_POLICY = (
    "literal-disease-or-explicit-model-plus-target-variant-same-sentence-v1"
)
_PRIMARY_QUERY_POLICIES = {
    _PRIMARY_QUERY_POLICY,
    _PRIMARY_DIRECTION_SUPPLEMENT_QUERY_POLICY,
    _PRIMARY_DIRECTION_MODEL_QUERY_POLICY,
    _PRIMARY_DIRECTION_EXPERIMENTAL_QUERY_POLICY,
}
_EPMC_OA_FULLTEXT_CONTENT_AVAILABILITY = "oa_fulltext_excerpt"
_EPMC_OA_FULLTEXT_RETRIEVAL_POLICY = "epmc_oa_fulltext_one_hop_v1"
_EPMC_OA_FULLTEXT_EXCERPT_POLICY = "europe_pmc_oa_jats_results_v1"
_EPMC_OA_FULLTEXT_CONTENT_HASH_SCOPE = "europe_pmc_oa_jats_xml"
_EPMC_OA_FULLTEXT_STORAGE_POLICY = "bounded_oa_excerpt_only_no_full_text"
_EPMC_OA_FULLTEXT_PUBLICATION_STATUS = "primary_research_article"
_EPMC_OA_FULLTEXT_MAX_EXCERPT_CHARS = 320
_EPMC_OA_FULLTEXT_MAX_INPUT_BYTES = 4 * 1024 * 1024
_EPMC_OA_FULLTEXT_MAX_JATS_ELEMENTS = 100_000
_EPMC_OA_FULLTEXT_MAX_EMITTED_SPANS = 64
_EPMC_OA_FULLTEXT_ALLOWED_LICENSES = frozenset(
    {
        "CC BY",
        "CC BY-SA",
        "CC BY-ND",
        "CC BY-NC",
        "CC BY-NC-SA",
        "CC BY-NC-ND",
        "CC0",
        "https://creativecommons.org/licenses/by/4.0/",
        "http://creativecommons.org/licenses/by/4.0/",
        "https://creativecommons.org/licenses/by-nc/4.0/",
        "http://creativecommons.org/licenses/by-nc/4.0/",
        "https://creativecommons.org/licenses/by-nd/4.0/",
        "http://creativecommons.org/licenses/by-nd/4.0/",
        "https://creativecommons.org/licenses/by-nc-nd/4.0/",
        "http://creativecommons.org/licenses/by-nc-nd/4.0/",
        "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "http://creativecommons.org/licenses/by-nc-sa/4.0/",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "http://creativecommons.org/licenses/by-sa/4.0/",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "http://creativecommons.org/publicdomain/zero/1.0/",
    }
)
_EPMC_OA_FORBIDDEN_RAW_BODY_KEYS = frozenset(
    {
        "full_text",
        "full_text_body",
        "full_text_content",
        "full_text_xml",
        "jats_xml",
        "raw_body",
        "raw_full_text",
        "raw_jats_xml",
        "raw_xml",
    }
)
_EPMC_OA_FULLTEXT_LOCATION_PATTERN = re.compile(
    r"^full_text:(results|findings|case):sec:([1-9]\d*):p:([1-9]\d*):"
    r"sentence:([1-9]\d*):chars:(\d+)-(\d+)$"
)
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.I)
_EPMC_OA_PRIMARY_PUBLICATION_TYPE_PATTERN = re.compile(
    r"\b(?:journal article|research[- ]article|clinical trial|"
    r"randomized controlled|comparative study|evaluation study|"
    r"validation study|multicenter study|observational study|"
    r"preclinical study|case reports?)\b",
    re.I,
)
_UNSAFE_PRIMARY_PUBLICATION_PATTERN = re.compile(
    r"\b(?:review|meta[- ]?analysis|guideline|consensus|editorial|commentary|"
    r"letter|correction|retracted publication|retraction of publication|"
    r"retract(?:ed|ion)|withdrawn|expression of concern|published erratum|"
    r"protocol)\b",
    re.I,
)
_MOLECULAR_NEGATIVE_SUFFIX_GUARD = (
    r"(?!\s*(?:(?:[-\u2212\u2013\u2014]\s*)?negative\b|"
    r"\(\s*[-\u2212\u2013\u2014]\s*\)|"
    r"[-\u2212\u2013\u2014](?=\s*(?:[,;.)]|$))))"
)
_EXTERNAL_PROVENANCE_CLASSES = {
    "external_live",
    "external_cache",
    "external_mixed",
}
_DIRECTION_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "inhibit": re.compile(
        r"\b(?:inhibit(?:ed|ing|ion|or|ors|ory|s)?|block(?:ade|ed|ing|s)?|"
        r"suppress(?:ed|es|ing|ion)?|knock[- ]?down|silenc(?:e|ed|ing))\b",
        re.I,
    ),
    "activate": re.compile(
        r"\b(?:activat(?:e|ed|es|ing|ion|or|ors)|agonis(?:t|ts|m)|"
        r"stimulat(?:e|ed|es|ing|ion)|overexpress(?:ed|es|ing|ion)?)\b",
        re.I,
    ),
    "modulate": re.compile(
        r"\b(?:modulat(?:e|ed|es|ing|ion|or|ors)|regulat(?:e|ed|es|ing|ion)|"
        r"perturb(?:ed|ing|ation|ations)?)\b",
        re.I,
    ),
}
_MEASURED_ENDPOINT_PATTERN = re.compile(
    r"\b(?:target[- ]engagement|kinase activity|enzymatic activity|"
    r"phosphorylat\w*|viabilit\w*|proliferat\w*|apopto\w*|cytotoxi\w*|"
    r"sensitiv\w*|respons(?:e|es|ive)|IC50|EC50|Ki|Kd|tumou?r growth|"
    r"tumou?r burden|cell survival|differentiation|oncometabolite(?: levels?)?|"
    r"metabolite levels?|2[- ]?(?:HG|hydroxyglutarate))\b",
    re.I,
)
_EXPERIMENTAL_RESULT_PATTERN = re.compile(
    r"\b(?:reduc(?:e|ed|es|ing|tion)|decreas(?:e|ed|es|ing)|"
    r"increas(?:e|ed|es|ing)|induc(?:e|ed|es|ing|tion)|"
    r"suppress(?:ed|es|ing|ion)|impair(?:ed|s|ing|ment)|"
    r"enhanc(?:e|ed|es|ing|ement)|attenuat(?:e|ed|es|ing|ion)|"
    r"abolish(?:ed|es|ing)|inhibit(?:ed|s|ing)|block(?:ed|s|ing)|"
    r"show(?:ed|s|ing)|achiev(?:e|ed|es|ing)|"
    r"demonstrat(?:e|ed|es|ing|ion)|"
    r"led to|resulted in|"
    r"dose[- ]dependent|significant(?:ly)?)\b",
    re.I,
)
_NON_DIRECTION_PRIMARY_PATTERN = re.compile(
    r"\b(?:prevalen\w*|frequen\w*|common(?:ly)?|prognos\w*|survival|"
    r"associated with|association|resistan\w*|refractor\w*|relaps\w*|"
    r"remains? unclear|unknown role|failed to|did not|no response|"
    r"limited efficacy)\b",
    re.I,
)
_REVIEW_TITLE_PATTERN = re.compile(
    r"\b(?:systematic review|review article|meta-analysis|consensus|guideline)\b",
    re.I,
)
_PRIMARY_PUBLICATION_TYPE_PATTERN = re.compile(
    r"\b(?:journal article|research[- ]article|clinical trial|randomized controlled|"
    r"comparative study|evaluation study|validation study|multicenter study|"
    r"observational study|preclinical study)\b",
    re.I,
)
_RESPONSE_MODIFIER_PATTERN = re.compile(
    r"\b(?:enhanc(?:e|ed|es|ing|ement)|potentiat(?:e|ed|es|ing|ion)|"
    r"sensiti[sz](?:e|ed|es|ing|ation)|synerg(?:y|istic|istically)|"
    r"(?:increas\w*|enhanc\w*|elevat\w*|improv\w*)\s+"
    r"(?:the\s+)?(?:drug\s+)?sensitiv\w*|"
    r"in response to|combination|combined with|"
    r"combined\s+[^.!?]{1,48}\band\b[^.!?]{0,48}\b"
    r"(?:inhibition|treatment|therapy|blockade)|"
    r"co[- ]treat(?:ed|ment)|"
    r"resensiti[sz](?:e|ed|es|ing|ation))\b",
    re.I,
)
_OTHER_AGENT_TREATED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?P<agent>[A-Z][A-Z0-9]{1,10})[- ]treated\b"
)
_ASSOCIATION_ROLE_RECORD = "primary_target_disease_association_excerpt"
_ASSOCIATION_PREDICATES = {
    "functional_target_disease_association",
    "mutation_prevalence",
    "disease_target_association",
    "disease_target_expression",
}
_ASSOCIATION_SCOPES = {
    "target_general",
    "activating_or_itd_target_subtype",
    "rearranged_or_fusion_target_subtype",
}
_ENDPOINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("target_engagement", re.compile(r"\b(?:target[- ]engagement|bound to)\b", re.I)),
    ("kinase_activity", re.compile(r"\b(?:kinase|enzymatic) activity\b", re.I)),
    ("phosphorylation", re.compile(r"\bphosphorylat\w*\b", re.I)),
    ("viability", re.compile(r"\bviabilit\w*\b", re.I)),
    ("proliferation", re.compile(r"\bproliferat\w*\b", re.I)),
    ("apoptosis", re.compile(r"\bapopto\w*\b", re.I)),
    ("sensitivity_or_response", re.compile(r"\b(?:sensitiv\w*|respons(?:e|es|ive))\b", re.I)),
    ("potency", re.compile(r"\b(?:IC50|EC50|Ki|Kd|potenc\w*)\b", re.I)),
    ("tumor_growth", re.compile(r"\btumou?r (?:growth|burden)\b", re.I)),
)


@dataclass(frozen=True)
class _DeclaredDirectionVariant:
    canonical: str
    sentence_pattern: re.Pattern[str]


def _disease_initialisms(disease: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z0-9]+", disease)
    if len(tokens) < 2:
        return ()
    initialism = "".join(token[0] for token in tokens if token).upper()
    return (initialism,) if 3 <= len(initialism) <= 8 else ()


def _target_molecular_disease_base(
    disease: str,
    *,
    target_symbol: str,
) -> tuple[str, str] | None:
    """Return the base only for an exact target-prefixed molecular disease."""

    match = re.match(
        rf"^\s*{re.escape(target_symbol.strip())}\s*"
        r"(?:[-\u2010-\u2015]\s*|\s+)"
        r"(?:rearranged|rearrangement[- ]positive|fusion[- ]positive|"
        r"fusion|mutant|mutated)\s+(?P<base>.+?)\s*$",
        disease,
        re.I,
    )
    if match is None:
        return None
    base = " ".join(match.group("base").split()).strip()
    qualifier_match = re.match(
        rf"^\s*{re.escape(target_symbol.strip())}\s*"
        r"(?:[-\u2010-\u2015]\s*|\s+)"
        r"(?P<qualifier>rearranged|rearrangement[- ]positive|"
        r"fusion[- ]positive|fusion|mutant|mutated)\s+",
        disease,
        re.I,
    )
    if not base or qualifier_match is None:
        return None
    return base, qualifier_match.group("qualifier").casefold()


def _terminal_plural_aliases(value: str) -> tuple[str, ...]:
    normalized = " ".join(value.split()).strip()
    if re.search(r"\btumou?r$", normalized, re.I):
        return (normalized, normalized + "s")
    return (normalized,)


def _has_target_disease_excerpt(
    excerpt: str,
    *,
    target_symbol: str,
    disease: str,
) -> bool:
    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])",
        excerpt,
        re.I,
    ) is None:
        return False
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", excerpt.casefold()))
    normalized_disease = " ".join(
        re.findall(r"[a-z0-9]+", disease.casefold())
    )
    molecular = _target_molecular_disease_base(
        disease,
        target_symbol=target_symbol,
    )
    if molecular is not None:
        molecular_base, qualifier = molecular
        normalized_aliases = {
            " ".join(re.findall(r"[a-z0-9]+", alias.casefold()))
            for alias in _terminal_plural_aliases(molecular_base)
        }
        base_match = any(
            alias and f" {alias} " in f" {normalized_text} "
            for alias in normalized_aliases
        ) or any(
            re.search(
                rf"(?<![A-Z0-9]){re.escape(value)}(?![A-Z0-9])",
                excerpt,
            )
            for value in _disease_initialisms(molecular_base)
            if value.casefold() != target_symbol.casefold()
        )
        if not base_match:
            return False
        target = rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
        if "rearrang" in qualifier or "fusion" in qualifier:
            return bool(
                re.search(
                    rf"(?:{target}\s*(?:[-\u2010-\u2015]\s*|\s+)"
                    r"(?:gene\s+)?(?:rearrang(?:ed|ement|ements)|"
                    r"fusion(?:[- ]positive|s)?)\b|"
                    r"\b(?:rearrang(?:ement|ements)|fusion|fusions)\s+"
                    rf"(?:of|in|involving)\s+{target}\b)"
                    rf"{_MOLECULAR_NEGATIVE_SUFFIX_GUARD}",
                    excerpt,
                    re.I,
                )
            )
        elif "mut" in qualifier:
            return bool(
                re.search(
                    rf"(?:{target}\s*(?:[-\u2010-\u2015]\s*|\s+)"
                    r"(?:mutant|mutated|mutation(?:s)?)\b|"
                    r"\b(?:mutation|mutations)\s+(?:of|in)\s+"
                    rf"{target}\b){_MOLECULAR_NEGATIVE_SUFFIX_GUARD}",
                    excerpt,
                    re.I,
                )
            )
        else:
            return False
    if normalized_disease and normalized_disease in normalized_text:
        return True
    if any(
        re.search(rf"(?<![A-Z0-9]){re.escape(value)}(?![A-Z0-9])", excerpt)
        for value in _disease_initialisms(disease)
    ):
        return True
    disease_tokens = {
        token
        for token in normalized_disease.split()
        if len(token) >= 3
        and token not in {"disease", "cancer", "syndrome", "disorder"}
    }
    required = 1 if len(disease_tokens) <= 1 else 2
    matched = sum(
        1
        for token in disease_tokens
        if re.search(rf"\b{re.escape(token)}\b", normalized_text)
    )
    return bool(disease_tokens) and matched >= required


def _has_atomic_target_disease_excerpt(
    excerpt: str,
    *,
    target_symbol: str,
    disease: str,
) -> bool:
    """Require target and exact disease/base alias in one stored sentence."""

    sentences = _stored_excerpt_sentences(excerpt)
    return any(
        _has_target_disease_excerpt(
            sentence,
            target_symbol=target_symbol,
            disease=disease,
        )
        for sentence in sentences
    )


def _stored_excerpt_sentences(excerpt: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", excerpt)
        if sentence.strip()
    ) or (excerpt,)


def _declared_direction_variants(
    brief: ResearchBrief,
    target_symbol: str,
) -> tuple[_DeclaredDirectionVariant, ...]:
    """Independently derive exact target variants declared in the Brief."""

    target = rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
    separator = r"[-/\u2010-\u2015\u2212]"
    fields = (
        brief.disease,
        brief.research_goal,
        brief.rationale or "",
        brief.constraints.target_cell_type or "",
        *brief.human_priorities,
    )
    variants: dict[str, _DeclaredDirectionVariant] = {}
    for text in fields:
        itd = re.compile(
            rf"(?:{target}\s*(?:{separator}\s*|\s+)"
            rf"(?:ITD(?![A-Za-z0-9])|internal tandem duplication\b)|"
            rf"internal tandem duplication(?:\s+(?:of|in))?\s+{target})"
            rf"{_MOLECULAR_NEGATIVE_SUFFIX_GUARD}",
            re.I,
        )
        if itd.search(text):
            variants.setdefault(
                f"{target_symbol}-ITD".casefold(),
                _DeclaredDirectionVariant(
                    canonical=f"{target_symbol}-ITD",
                    sentence_pattern=itd,
                ),
            )

        amino = re.compile(
            rf"{target}\s*(?:(?:{separator}|:)\s*|\s+)"
            r"(?:p\.)?([A-Za-z][0-9]{1,5}[A-Za-z*])"
            rf"(?![A-Za-z0-9]){_MOLECULAR_NEGATIVE_SUFFIX_GUARD}",
            re.I,
        )
        for match in amino.finditer(text):
            residue = match.group(1).upper()
            canonical = f"{target_symbol} {residue}"
            variants.setdefault(
                canonical.casefold(),
                _DeclaredDirectionVariant(
                    canonical=canonical,
                    sentence_pattern=re.compile(
                        rf"{target}\s*(?:(?:{separator}|:)\s*|\s+)"
                        rf"(?:p\.)?{re.escape(residue)}"
                        rf"(?![A-Za-z0-9])"
                        rf"{_MOLECULAR_NEGATIVE_SUFFIX_GUARD}",
                        re.I,
                    ),
                ),
            )

        rearranged_or_fusion = re.compile(
            rf"(?:{target}\s*(?:[-\u2010-\u2015]\s*|\s+)"
            r"(?:gene\s+)?(?:rearrang(?:ed|ement|ements)|"
            r"fusion(?:[- ]positive|s)?)\b|"
            r"\b(?:rearrang(?:ement|ements)|fusion|fusions)\s+"
            rf"(?:of|in|involving)\s+{target}\b)"
            rf"{_MOLECULAR_NEGATIVE_SUFFIX_GUARD}",
            re.I,
        )
        if rearranged_or_fusion.search(text):
            canonical = f"{target_symbol} rearrangement/fusion"
            variants.setdefault(
                canonical.casefold(),
                _DeclaredDirectionVariant(
                    canonical=canonical,
                    sentence_pattern=rearranged_or_fusion,
                ),
            )
    return tuple(variants.values())[:4]


def _declared_direction_model_aliases(
    brief: ResearchBrief,
    target_symbol: str,
    variants: Sequence[_DeclaredDirectionVariant],
) -> tuple[str, ...]:
    """Extract bounded cell-model identifiers without inferring biology."""

    configured = brief.constraints.target_cell_type
    if not isinstance(configured, str) or not configured.strip():
        return ()
    primary_model_text = re.split(
        r"\b(?:with|versus|vs\.?)\b",
        configured,
        maxsplit=1,
        flags=re.I,
    )[0]
    blocked = {
        re.sub(r"[^a-z0-9]+", "", value.casefold())
        for value in (
            target_symbol,
            *(variant.canonical for variant in variants),
            *(
                token
                for variant in variants
                for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9]*",
                    variant.canonical,
                )
            ),
        )
    }
    target_key = re.sub(
        r"[^a-z0-9]+",
        "",
        target_symbol.casefold(),
    )
    aliases: list[str] = []
    for match in re.finditer(
        r"(?<![A-Za-z0-9])"
        r"([A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)"
        r"(?![A-Za-z0-9])",
        primary_model_text,
    ):
        alias = match.group(1)
        if not re.search(r"\d", alias):
            continue
        normalized = re.sub(r"[^a-z0-9]+", "", alias.casefold())
        if (
            not normalized
            or normalized in blocked
            or normalized.startswith(target_key)
            or re.match(r"^(?:p?stat|ic|ec)\d", normalized)
        ):
            continue
        aliases.append(alias)
        if len(aliases) == 4:
            break
    return tuple(dict.fromkeys(aliases))


def _metadata_text_sequence(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    normalized = tuple(str(item).strip() for item in value)
    if not normalized or any(not item for item in normalized):
        return None
    return normalized


def _has_verified_predeclared_direction_model_context(
    brief: ResearchBrief,
    target_symbol: str,
    excerpt: str,
    metadata: Mapping[str, Any],
) -> bool:
    """Recompute the producer's exact model+variant contract fail-closed."""

    if (
        metadata.get("direction_disease_context_match_basis")
        != _PRIMARY_DIRECTION_MODEL_CONTEXT_BASIS
        or metadata.get("direction_disease_context_match_policy")
        != _PRIMARY_DIRECTION_MODEL_CONTEXT_POLICY
    ):
        return False
    variants = _declared_direction_variants(brief, target_symbol)
    aliases = _declared_direction_model_aliases(
        brief,
        target_symbol,
        variants,
    )
    if not variants or not aliases:
        return False
    if _metadata_text_sequence(
        metadata.get("predeclared_model_aliases")
    ) != aliases:
        return False
    if _metadata_text_sequence(
        metadata.get("predeclared_target_variants")
    ) != tuple(variant.canonical for variant in variants):
        return False
    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])",
        excerpt,
        re.I,
    ) is None:
        return False
    if not any(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            excerpt,
            re.I,
        )
        for alias in aliases
    ):
        return False
    return any(
        variant.sentence_pattern.search(excerpt)
        for variant in variants
    )


def _publication_types(evidence: EvidenceRef) -> tuple[str, ...]:
    raw = evidence.metadata.get("publication_types", ())
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(str(item) for item in raw)
    return ()


def _source_supports(evidence: EvidenceRef) -> str:
    if "source_supports" in evidence.metadata:
        return str(evidence.metadata.get("source_supports"))
    if "pre_reconciliation_supports" in evidence.metadata:
        return str(evidence.metadata.get("pre_reconciliation_supports"))
    return evidence.supports


def _has_verified_primary_publication_type(evidence: EvidenceRef) -> bool:
    publication_types = _publication_types(evidence)
    return bool(
        publication_types
        and not any("review" in value.casefold() for value in publication_types)
        and any(
            _PRIMARY_PUBLICATION_TYPE_PATTERN.search(value)
            for value in publication_types
        )
        and not _REVIEW_TITLE_PATTERN.search(evidence.title)
    )


def _has_verified_epmc_oa_primary_publication_type(
    evidence: EvidenceRef,
) -> bool:
    publication_types = _publication_types(evidence)
    publication_text = " ".join((evidence.title, *publication_types))
    return bool(
        publication_types
        and any(
            _EPMC_OA_PRIMARY_PUBLICATION_TYPE_PATTERN.search(value)
            for value in publication_types
        )
        and not _UNSAFE_PRIMARY_PUBLICATION_PATTERN.search(publication_text)
    )


def _is_sha256_hex(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and _SHA256_HEX_PATTERN.fullmatch(value)
    )


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _epmc_oa_fulltext_lineage_is_valid(evidence: EvidenceRef) -> bool:
    """Validate bounded direct-search or one-hop reference provenance.

    Citation lineage is discovery provenance only.  It never transfers a
    parent's evidence role, publication status, or scientific conclusion to
    the child document.  A one-hop child must carry the same independent
    primary-publication verification required for a directly found document.
    """

    metadata = evidence.metadata
    required_fields = {
        "citation_hop",
        "citation_depth",
        "citation_relation",
        "citation_parent_external_id",
        "citation_parent_pmcid",
        "citation_parent_pmid",
        "citation_reference_position",
        "citation_child_canonical_id",
        "citation_child_identity_keys",
        "citation_child_pmcid",
        "citation_child_pmid",
        "citation_child_doi",
        "citation_root_external_id",
        "citation_lineage_sha256",
        "lineage_verified",
        "evidence_role_inherited",
        "independent_primary_verification_required",
        "independent_primary_verification",
    }
    if not required_fields.issubset(metadata):
        return False
    hop = metadata.get("citation_hop")
    if not _is_nonnegative_int(hop) or hop not in {0, 1}:
        return False
    if (
        metadata.get("citation_depth") != hop
        or metadata.get("lineage_verified") is not True
        or metadata.get("evidence_role_inherited") is not False
        or metadata.get("independent_primary_verification_required") is not True
        or metadata.get("independent_primary_verification") is not True
        or not _is_sha256_hex(metadata.get("citation_lineage_sha256"))
    ):
        return False

    pmcid = metadata.get("pmcid")
    pmid = metadata.get("pmid")
    if not isinstance(pmcid, str) or re.fullmatch(r"PMC[0-9]+", pmcid) is None:
        return False
    if pmid is not None and (
        not isinstance(pmid, str) or re.fullmatch(r"[0-9]+", pmid) is None
    ):
        return False
    core_identity_keys = [f"PMC:{pmcid}"]
    if pmid:
        core_identity_keys.append(f"MED:{pmid}")
    doi = metadata.get("doi")
    if doi is not None:
        if (
            not isinstance(doi, str)
            or doi != doi.strip().casefold()
            or _DOI_PATTERN.fullmatch(doi) is None
        ):
            return False
    resolved_identity_set = set(core_identity_keys)
    resolved_external_ids = {pmcid, f"PMC:{pmcid}"}
    if pmid:
        resolved_external_ids.update({pmid, f"MED:{pmid}"})
    if doi:
        resolved_identity_set.add(f"DOI:{doi}")
        resolved_external_ids.update({doi, f"DOI:{doi}"})
    reported_identity_keys = _metadata_text_sequence(
        metadata.get("citation_child_identity_keys")
    )
    if (
        reported_identity_keys is None
        or reported_identity_keys != tuple(sorted(reported_identity_keys))
        or len(set(reported_identity_keys)) != len(reported_identity_keys)
    ):
        return False

    parent_external_id = metadata.get("citation_parent_external_id")
    parent_pmcid = metadata.get("citation_parent_pmcid")
    parent_pmid = metadata.get("citation_parent_pmid")
    reference_position = metadata.get("citation_reference_position")
    root_external_id = metadata.get("citation_root_external_id")
    if hop == 0:
        child_doi = metadata.get("citation_child_doi")
        if child_doi is not None and (
            not isinstance(child_doi, str)
            or child_doi != child_doi.strip().casefold()
            or _DOI_PATTERN.fullmatch(child_doi) is None
            or child_doi != doi
        ):
            return False
        if child_doi != doi:
            return False
        lineage_identity_keys = list(core_identity_keys)
        if doi:
            lineage_identity_keys.append(f"DOI:{doi}")
            resolved_identity_set.add(f"DOI:{doi}")
            resolved_external_ids.update(
                {doi, f"DOI:{doi}"}
            )
        if evidence.external_id not in resolved_external_ids:
            return False
        child_canonical_id = f"MED:{pmid}" if pmid else f"PMC:{pmcid}"
        lineage_payload = {
            "policy": _EPMC_OA_FULLTEXT_RETRIEVAL_POLICY,
            "hop": 0,
            "relation": "direct_search",
            "child": child_canonical_id,
        }
        return bool(
            metadata.get("citation_child_pmcid") == pmcid
            and metadata.get("citation_child_pmid") == pmid
            and metadata.get("citation_child_doi") == doi
            and reported_identity_keys
            == tuple(sorted(lineage_identity_keys))
            and metadata.get("citation_child_canonical_id")
            == child_canonical_id
            and metadata.get("citation_relation") == "direct_search"
            and parent_external_id is None
            and parent_pmcid is None
            and parent_pmid is None
            and reference_position is None
            and root_external_id == child_canonical_id
            and metadata.get("citation_lineage_sha256")
            == _canonical_json_sha256(lineage_payload)
        )

    child_pmcid = metadata.get("citation_child_pmcid")
    child_pmid = metadata.get("citation_child_pmid")
    child_doi = metadata.get("citation_child_doi")
    if child_pmcid is not None and (
        not isinstance(child_pmcid, str)
        or re.fullmatch(r"PMC[0-9]+", child_pmcid) is None
        or child_pmcid != pmcid
    ):
        return False
    if child_pmid is not None and (
        not isinstance(child_pmid, str)
        or re.fullmatch(r"[0-9]+", child_pmid) is None
        or child_pmid != pmid
    ):
        return False
    if child_doi is not None:
        if (
            not isinstance(child_doi, str)
            or child_doi != child_doi.strip().casefold()
            or _DOI_PATTERN.fullmatch(child_doi) is None
            or child_doi != doi
        ):
            return False
    edge_identity_keys: list[str] = []
    if child_pmcid:
        edge_identity_keys.append(f"PMC:{child_pmcid}")
    if child_pmid:
        edge_identity_keys.append(f"MED:{child_pmid}")
    if child_doi:
        edge_identity_keys.append(f"DOI:{child_doi}")
    if not edge_identity_keys:
        return False
    if child_doi:
        resolved_identity_set.add(f"DOI:{child_doi}")
        resolved_external_ids.update({child_doi, f"DOI:{child_doi}"})
    if evidence.external_id not in resolved_external_ids:
        return False
    child_canonical_id = (
        f"MED:{child_pmid}"
        if child_pmid
        else f"PMC:{child_pmcid}"
        if child_pmcid
        else f"DOI:{child_doi}"
    )
    if (
        reported_identity_keys != tuple(sorted(edge_identity_keys))
        or not set(reported_identity_keys).issubset(resolved_identity_set)
        or child_canonical_id not in reported_identity_keys
        or metadata.get("citation_child_canonical_id")
        != child_canonical_id
    ):
        return False
    if (
        metadata.get("citation_relation") != "reference"
        or not isinstance(parent_external_id, str)
        or not parent_external_id.strip()
        or not _is_nonnegative_int(reference_position)
        or reference_position < 1
        or root_external_id != parent_external_id
    ):
        return False
    if parent_pmcid is not None and (
        not isinstance(parent_pmcid, str)
        or re.fullmatch(r"PMC[0-9]+", parent_pmcid) is None
    ):
        return False
    if parent_pmid is not None and (
        not isinstance(parent_pmid, str)
        or re.fullmatch(r"[0-9]+", parent_pmid) is None
    ):
        return False
    if parent_pmcid is None and parent_pmid is None:
        return False
    if parent_external_id != (parent_pmid or parent_pmcid):
        return False
    if parent_external_id in resolved_external_ids:
        return False
    lineage_payload = {
        "policy": "europe_pmc_reference_one_hop_v1",
        "hop": 1,
        "relation": "reference",
        "parent_pmcid": parent_pmcid,
        "parent_pmid": parent_pmid,
        "position": reference_position,
        "child": child_canonical_id,
        "identity_keys": list(reported_identity_keys),
    }
    return metadata.get("citation_lineage_sha256") == _canonical_json_sha256(
        lineage_payload
    )


def _epmc_oa_fulltext_contract_excerpt(
    evidence: EvidenceRef,
    *,
    allow_context_support: bool = False,
) -> str | None:
    """Return a copyright-bounded OA Results excerpt only if provenance is exact."""

    metadata = evidence.metadata
    required_fields = {
        "excerpt",
        "excerpt_hash",
        "full_text_span_excerpt_hash",
        "excerpt_char_count",
        "excerpt_max_chars",
        "excerpt_truncated",
        "original_sentence_char_count",
        "content_hash",
        "full_text_content_hash",
        "eligible_content_hash",
        "content_hash_algorithm",
        "content_hash_scope",
        "content_availability",
        "source_location",
        "full_text_stored",
        "raw_xml_stored",
        "content_storage_policy",
        "content_retrieval_policy",
        "excerpt_selection_policy",
        "evidence_id_origin_policy",
        "source_database",
        "discovery_origin",
        "evidence_origin",
        "pmcid",
        "pmid",
        "doi",
        "is_open_access",
        "full_text_license",
        "publication_status",
        "publication_types",
        "input_byte_count",
        "eligible_section_count",
        "eligible_paragraph_count",
        "emitted_span_count",
        "spans_truncated_by_limit",
        "section_kind",
        "section_title",
        "source_location_scope",
        "source_location_char_end_exclusive",
    }
    if not required_fields.issubset(metadata):
        return None
    excerpt = metadata.get("excerpt")
    allowed_supports = (
        {"support", "context"}
        if allow_context_support
        else {"support"}
    )
    if (
        evidence.source != "Europe PMC"
        or evidence.supports not in allowed_supports
        or _source_supports(evidence) not in allowed_supports
        or not isinstance(excerpt, str)
        or not excerpt
        or excerpt != excerpt.strip()
        or len(excerpt) > _EPMC_OA_FULLTEXT_MAX_EXCERPT_CHARS
        or metadata.get("excerpt_char_count") != len(excerpt)
        or metadata.get("excerpt_max_chars")
        != _EPMC_OA_FULLTEXT_MAX_EXCERPT_CHARS
        or not isinstance(metadata.get("excerpt_truncated"), bool)
        or metadata.get("content_availability")
        != _EPMC_OA_FULLTEXT_CONTENT_AVAILABILITY
        or metadata.get("source_database") != "Europe PMC"
        or metadata.get("full_text_stored") is not False
        or metadata.get("raw_xml_stored") is not False
        or metadata.get("content_storage_policy")
        != _EPMC_OA_FULLTEXT_STORAGE_POLICY
        or metadata.get("content_retrieval_policy")
        != _EPMC_OA_FULLTEXT_RETRIEVAL_POLICY
        or metadata.get("excerpt_selection_policy")
        != _EPMC_OA_FULLTEXT_EXCERPT_POLICY
        or metadata.get("content_hash_algorithm") != "sha256"
        or metadata.get("content_hash_scope")
        != _EPMC_OA_FULLTEXT_CONTENT_HASH_SCOPE
        or metadata.get("is_open_access") is not True
        or metadata.get("full_text_license")
        not in _EPMC_OA_FULLTEXT_ALLOWED_LICENSES
        or metadata.get("publication_status")
        != _EPMC_OA_FULLTEXT_PUBLICATION_STATUS
        or not _is_nonnegative_int(metadata.get("input_byte_count"))
        or not 1
        <= metadata.get("input_byte_count")
        <= _EPMC_OA_FULLTEXT_MAX_INPUT_BYTES
        or not _is_nonnegative_int(metadata.get("eligible_section_count"))
        or not 1
        <= metadata.get("eligible_section_count")
        <= _EPMC_OA_FULLTEXT_MAX_JATS_ELEMENTS
        or not _is_nonnegative_int(metadata.get("eligible_paragraph_count"))
        or not 1
        <= metadata.get("eligible_paragraph_count")
        <= _EPMC_OA_FULLTEXT_MAX_JATS_ELEMENTS
        or not _is_nonnegative_int(metadata.get("emitted_span_count"))
        or not 1
        <= metadata.get("emitted_span_count")
        <= _EPMC_OA_FULLTEXT_MAX_EMITTED_SPANS
        or not isinstance(metadata.get("spans_truncated_by_limit"), bool)
        or not _is_nonnegative_int(metadata.get("original_sentence_char_count"))
        or not isinstance(metadata.get("section_title"), str)
        or metadata.get("source_location_scope")
        != "normalized_admitted_paragraph_text"
        or metadata.get("source_location_char_end_exclusive") is not True
        or metadata.get("provenance_class") not in _EXTERNAL_PROVENANCE_CLASSES
        or metadata.get("mock") is True
    ):
        return None
    if any(key in metadata for key in _EPMC_OA_FORBIDDEN_RAW_BODY_KEYS):
        return None
    for no_retention_flag in (
        "raw_full_text_retained",
        "raw_reference_payload_retained",
        "raw_reference_payload_stored",
    ):
        if (
            no_retention_flag in metadata
            and metadata.get(no_retention_flag) is not False
        ):
            return None

    location_match = _EPMC_OA_FULLTEXT_LOCATION_PATTERN.fullmatch(
        str(metadata.get("source_location") or "")
    )
    if location_match is None:
        return None
    section_kind, section_index, paragraph_index, sentence_index, char_start, char_end = (
        location_match.groups()
    )
    section_index, paragraph_index, sentence_index, char_start, char_end = (
        int(value)
        for value in (
            section_index,
            paragraph_index,
            sentence_index,
            char_start,
            char_end,
        )
    )
    if char_end <= char_start:
        return None
    span_length = char_end - char_start
    if (
        metadata.get("section_kind") != section_kind
        or paragraph_index > metadata.get("eligible_paragraph_count")
        or span_length != len(excerpt)
        or (
            metadata.get("excerpt_truncated") is False
            and metadata.get("original_sentence_char_count") != len(excerpt)
        )
        or (
            metadata.get("excerpt_truncated") is True
            and metadata.get("original_sentence_char_count") <= len(excerpt)
        )
    ):
        return None
    expected_discovery_origin = (
        "reference_one_hop"
        if metadata.get("citation_hop") == 1
        else "direct_search"
    )
    expected_evidence_origin = (
        "oa_results_fulltext_reference_one_hop"
        if metadata.get("citation_hop") == 1
        else "oa_results_fulltext_direct"
    )
    if (
        metadata.get("discovery_origin") != expected_discovery_origin
        or metadata.get("evidence_origin") != expected_evidence_origin
    ):
        return None

    content_hash = metadata.get("content_hash")
    if (
        not _is_sha256_hex(content_hash)
        or metadata.get("full_text_content_hash") != content_hash
        or not _is_sha256_hex(metadata.get("eligible_content_hash"))
        or metadata.get("excerpt_hash") != evidence.excerpt_hash
        or metadata.get("full_text_span_excerpt_hash")
        != evidence.excerpt_hash
        or evidence.excerpt_hash
        != sha256(excerpt.encode("utf-8")).hexdigest()
    ):
        return None
    if not _has_verified_epmc_oa_primary_publication_type(evidence):
        return None
    if not _epmc_oa_fulltext_lineage_is_valid(evidence):
        return None
    parent_evidence_id = "ev_" + sha256(
        (
            f"Europe PMC|{evidence.external_id}|oa_fulltext_excerpt|"
            f"{metadata.get('citation_hop')}|"
            f"{metadata.get('citation_relation')}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    role_specific_record = metadata.get("role_specific_record")
    if role_specific_record is None:
        expected_evidence_id = parent_evidence_id
    elif role_specific_record == _ASSOCIATION_ROLE_RECORD:
        expected_evidence_id = "ev_" + sha256(
            f"{parent_evidence_id}|{_ASSOCIATION_ROLE_RECORD}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
    else:
        return None
    if (
        metadata.get("evidence_id_origin_policy")
        != "source-external-content-origin-hop-relation-v1"
        or evidence.evidence_id != expected_evidence_id
    ):
        return None
    return excerpt


def _epmc_oa_fulltext_association_parent_is_bound(
    association: EvidenceRef,
    evidence: Sequence[EvidenceRef],
) -> bool:
    """Bind an OA association role to its selected document record.

    The association excerpt may be a different Results span from the selected
    parent excerpt.  Its own hash is therefore checked by the role contract,
    while this reconciliation check authenticates the separately stored parent
    pointer and document/citation lineage against the actual evidence set.
    """

    metadata = association.metadata
    parent_id = metadata.get("parent_evidence_id")
    parents = [item for item in evidence if item.evidence_id == parent_id]
    if len(parents) != 1:
        return False
    parent = parents[0]
    if (
        parent.evidence_id == association.evidence_id
        or parent.source != association.source
        or parent.external_id != association.external_id
        or parent.metadata.get("role_specific_record") is not None
        or parent.metadata.get("evidence_id_origin_policy")
        != "source-external-content-origin-hop-relation-v1"
        or parent.excerpt_hash != metadata.get("parent_excerpt_hash")
        or _epmc_oa_fulltext_contract_excerpt(
            parent,
            allow_context_support=True,
        )
        is None
    ):
        return False
    shared_document_and_lineage_fields = (
        "source_database",
        "content_hash",
        "full_text_content_hash",
        "eligible_content_hash",
        "content_hash_algorithm",
        "content_hash_scope",
        "content_retrieval_policy",
        "excerpt_selection_policy",
        "pmcid",
        "pmid",
        "doi",
        "citation_hop",
        "citation_depth",
        "citation_relation",
        "citation_parent_external_id",
        "citation_parent_pmcid",
        "citation_parent_pmid",
        "citation_reference_position",
        "citation_child_canonical_id",
        "citation_child_identity_keys",
        "citation_child_pmcid",
        "citation_child_pmid",
        "citation_child_doi",
        "citation_root_external_id",
        "citation_lineage_sha256",
    )
    return all(
        parent.metadata.get(field) == metadata.get(field)
        for field in shared_document_and_lineage_fields
    )


def _epmc_oa_fulltext_primary_direction_excerpt(
    brief: ResearchBrief,
    target_symbol: str,
    evidence: EvidenceRef,
    intervention_direction: str | None = None,
) -> str | None:
    """Independently verify an OA full-text child; never inherit parent support."""

    if (
        evidence.metadata.get("role_specific_record") is not None
        or evidence.metadata.get("counter_category")
        or evidence.metadata.get("review_status")
    ):
        return None
    excerpt = _epmc_oa_fulltext_contract_excerpt(evidence)
    if excerpt is None or len(_stored_excerpt_sentences(excerpt)) != 1:
        return None
    if not _has_atomic_target_disease_excerpt(
        excerpt,
        target_symbol=target_symbol,
        disease=brief.disease,
    ):
        return None
    direction = intervention_direction or brief.constraints.intervention_direction
    if _DIRECTION_PATTERNS[direction].search(excerpt) is None:
        return None
    if _is_response_modifier_context(target_symbol, evidence):
        return None
    if _direct_target_intervention_pattern(
        target_symbol,
        direction,
    ).search(excerpt) is None:
        return None
    if not _has_admissible_direction_endpoint(excerpt, target_symbol):
        return None
    if _MEASURED_ENDPOINT_PATTERN.search(excerpt) is None:
        return None
    if _EXPERIMENTAL_RESULT_PATTERN.search(excerpt) is None:
        return None
    if _direction_endpoint_polarity_status(
        excerpt,
        direction,
    ) != "favorable":
        return None
    if _NON_DIRECTION_PRIMARY_PATTERN.search(excerpt):
        return None
    return excerpt


def _direct_target_intervention_pattern(
    target_symbol: str,
    direction: str,
) -> re.Pattern[str]:
    # Blocking a target's pathway is not proof of perturbing that target.
    target = (rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
              rf"(?![- ](?:signall?ing|pathways?|cascades?|mediated)\b)")
    if direction == "inhibit":
        # Closed noun-phrase grammar: no arbitrary text between action and
        # entity, and no pathway/cascade substitution for the named target.
        descriptor = r"(?:(?:(?:oncogenic|mutant|mutated|wild[- ]type)\s+)?(?:(?:tyrosine\s+)?kinase|protein|receptor|fusion(?:\s+protein)?)\s+)?"
        target_state = r"(?:(?:mutant|mutated|wild[- ]type)[- ]*)?"
        target_inhibitor = (
            rf"{target}(?:[- ](?:selective|specific))?[- ]"
            rf"(?:(?:tyrosine[- ]kinase|kinase)[- ])?inhibitors?"
        )
        relationship = (
            rf"(?:\b(?:inhibit(?:ed|s|ing)|suppress(?:ed|es|ing)|"
            rf"block(?:ed|s|ing))\s+(?:the\s+)?{descriptor}{target_state}{target}"
            rf"(?![- ]mediated)"
            rf"(?:[- ]kinase|\s+(?:kinase(?:\s+activity)?|activity|"
            rf"phosphorylation))?|"
            rf"\b(?:inhibition|suppression|blockade)\s+of\s+(?:the\s+)?{descriptor}{target}|"
            rf"\b(?:knock[- ]?down|silencing)\s+of\s+(?:the\s+)?{target}|"
            rf"{target}[- ](?:knock[- ]?down|silencing)\b|"
            rf"\b(?:silenc(?:e|ed|ing)|knocked\s+down)\s+{target}|"
            rf"{target}\s+(?:was\s+)?(?:silenced|knocked\s+down)\b|"
            rf"{target}\s+(?:kinase\s+)?(?:inhibition|suppression|blockade)\b|"
            rf"{target}\s+(?:kinase(?:\s+activity)?|activity|phosphorylation)\s+"
            rf"(?:was\s+)?(?:inhibit(?:ed)?|suppress(?:ed)?|block(?:ed)?)|"
            rf"\b(?:directly\s+)?engag(?:e|ed|es|ing)\s+(?:with\s+)?{target}|"
            rf"{target}\s+(?:was\s+)?(?:directly\s+)?engag(?:e|ed)|"
            rf"\bbound(?:\s+directly)?\s+(?:to\s+)?[^.!?]{{0,48}}{target}|"
            rf"{target_inhibitor}\b)"
        )
    elif direction == "activate":
        relationship = (
            rf"(?:\b(?:activat(?:e|ed|es|ing)|stimulat(?:e|ed|es|ing))\s+"
            rf"(?:the\s+)?{target}\b|"
            rf"\b(?:activation|stimulation|overexpression)\s+of\s+{target}\b|"
            rf"\boverexpress(?:ed|ing)?\s+{target}\b|"
            rf"{target}\s+(?:was\s+)?overexpressed\b|"
            rf"{target}\s+(?:activation|stimulation|overexpression)\b|"
            rf"{target}(?:[- ](?:selective|specific))?[- ]agonists?\b)"
        )
    else:
        relationship = (
            rf"(?:\b(?:modulat(?:e|ed|es|ing)|regulat(?:e|ed|es|ing))\s+"
            rf"(?:the\s+)?{target}\b|"
            rf"\b(?:modulation|regulation)\s+of\s+{target}\b|"
            rf"{target}\s+(?:modulation|regulation)\b)"
        )
    return re.compile(relationship, re.I)


def _has_admissible_direction_endpoint(
    excerpt: str,
    target_symbol: str,
) -> bool:
    """Reject downstream-only molecular readouts as direct target evidence.

    A target-named inhibitor plus suppression of an unrelated phosphoprotein
    is pathway context, not a direct measurement of the requested target.
    Phenotypic outcomes, explicit potency/binding, or a molecular readout that
    names the requested target remain admissible.
    """

    if re.search(
        r"\b(?:viabilit\w*|proliferat\w*|apopto\w*|cytotoxi\w*|"
        r"sensitiv\w*|respons(?:e|es|ive)|IC50|EC50|Ki|Kd|potenc\w*|"
        r"tumou?r (?:growth|burden)|cell survival|differentiation|"
        r"oncometabolite(?: levels?)?|metabolite levels?|"
        r"2[- ]?(?:HG|hydroxyglutarate)|target[- ]engagement|bound to)\b",
        excerpt,
        re.I,
    ):
        return True

    target = rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
    return re.search(
        rf"(?:{target}\s+(?:(?:kinase|enzymatic)\s+)?"
        rf"(?:activity|signaling|phosphorylat\w*)\b|"
        rf"{target}\s+(?:was\s+)?phosphorylat\w*\b|"
        rf"\b(?:phosphorylat\w*|(?:kinase|enzymatic) activity|signaling)"
        rf"\s+of\s+(?:the\s+)?{target})",
        excerpt,
        re.I,
    ) is not None


def _direction_endpoint_polarity_status(
    excerpt: str,
    intervention_direction: str,
) -> str:
    """Mirror the retriever's fail-closed positive outcome predicate."""

    decrease = (
        r"(?:reduc\w*|decreas\w*|suppress\w*|impair\w*|"
        r"inhibit(?:ed|s|ing)?|"
        r"attenuat\w*|abolish\w*|block\w*|lower\w*)"
    )
    increase = (
        r"(?:increas\w*|enhanc\w*|elevat\w*|induc\w*|promot\w*|"
        r"stimulat\w*)"
    )
    adverse_growth = (
        r"(?:viabilit\w*|proliferat\w*|tumou?r\s+(?:growth|burden)|"
        r"cell\s+survival)"
    )
    favorable_gain = r"(?:apopto\w*|cytotoxi\w*|sensitiv\w*|differentiation)"
    disease_metabolite = (
        r"(?:oncometabolite(?:\s+levels?)?|metabolite\s+levels?|"
        r"2[- ]?(?:HG|hydroxyglutarate)(?:\s+levels?)?)"
    )
    signal = (
        r"(?:phosphorylat\w*|(?:kinase|enzymatic)\s+activity|"
        r"target[- ]engagement|signaling)"
    )

    def related(effect: str, endpoint: str) -> bool:
        # Mirror the retriever's local-clause binding so a decrease in one
        # endpoint cannot be reassigned across a conjunction to another.
        clauses = re.split(
            r"[;,.]|\b(?:and|but|whereas|while)\b",
            excerpt,
            flags=re.I,
        )
        return any(
            re.search(
                rf"(?:\b{effect}\b[^.!?]{{0,48}}\b{endpoint}\b|"
                rf"\b{endpoint}\b[^.!?]{{0,48}}\b{effect}\b)",
                clause,
                re.I,
            )
            is not None
            for clause in clauses
        )

    unfavorable = related(increase, adverse_growth) or related(
        decrease,
        favorable_gain,
    )
    if intervention_direction == "inhibit":
        unfavorable = unfavorable or related(increase, signal)
    elif intervention_direction == "activate":
        unfavorable = unfavorable or related(decrease, signal)
    if unfavorable:
        return "unfavorable"

    favorable = related(decrease, adverse_growth) or related(
        increase,
        favorable_gain,
    )
    favorable = favorable or related(decrease, disease_metabolite)
    if intervention_direction == "inhibit":
        favorable = favorable or related(decrease, signal)
    elif intervention_direction == "activate":
        favorable = favorable or related(increase, signal)
    favorable = favorable or bool(
        re.search(
            r"\b(?:show(?:ed|s|ing)|achiev(?:e|ed|es|ing)|"
            r"demonstrat(?:e|ed|es|ing)|had)\b[^.!?]{0,96}"
            r"\b(?:objective|complete|partial|durable|drastic|clinical\s+)?"
            r"respons(?:e|es)\b|"
            r"\b(?:tumou?r\s+regression|dose[- ]dependent\s+"
            r"(?:response|cytotoxicity|growth\s+inhibition))\b|"
            r"\b(?:IC50|EC50|Ki|Kd)\s*(?:of|=|was|were|:)?\s*"
            r"[<>\u2264\u2265]?\s*\d",
            excerpt,
            re.I,
        )
    )
    favorable = favorable or bool(
        re.search(
            r"\b(?:result(?:ed|ing)\s+in|caus(?:e|ed|es|ing))\s+"
            r"(?:apopto\w*|cytotoxi\w*|differentiation)\b",
            excerpt,
            re.I,
        )
    )
    return "favorable" if favorable else "ambiguous"


def _is_response_modifier_context(
    target_symbol: str,
    evidence: EvidenceRef,
) -> bool:
    excerpt = str(evidence.metadata.get("excerpt") or "")
    rejection_codes = {
        str(code)
        for key in (
            "direct_support_rejection_codes",
            "verified_direct_primary_rejection_codes",
        )
        for code in (_metadata_text_sequence(evidence.metadata.get(key)) or ())
    }
    if rejection_codes.intersection(
        {
            "combination_or_co_treatment",
            "response_modifier",
            "sensitization_to_other_intervention",
            "target_inhibitor_response_modifier",
            "other_intervention_response_modifier",
        }
    ):
        return True
    if not excerpt:
        return False
    # A coordinated intervention such as ``combined menin and FLT3
    # inhibition`` is not an isolated FLT3-direction result.  The earlier
    # grammar only caught the noun ``FLT3 inhibitor`` and therefore missed
    # the equally common ``FLT3 inhibition`` form.  Keep this conservative:
    # when a bounded admission excerpt contains both a combination cue and a
    # literal action on the requested target, retain it as context only.
    target_action = re.search(
        rf"(?:\b(?:inhibition|suppression|blockade|activation|stimulation)\s+"
        rf"of\s+(?:the\s+)?(?<![A-Za-z0-9]){re.escape(target_symbol)}"
        rf"(?![A-Za-z0-9])|(?<![A-Za-z0-9]){re.escape(target_symbol)}"
        rf"(?![A-Za-z0-9])\s+(?:inhibition|suppression|blockade|activation|"
        rf"stimulation)\b)",
        excerpt,
        re.I,
    )
    if target_action is not None and _RESPONSE_MODIFIER_PATTERN.search(excerpt):
        return True
    target_inhibitor = re.search(
        rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
        r"[^.!?]{0,32}\binhibitors?\b|\binhibitors?\b[^.!?]{0,32}"
        rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])",
        excerpt,
        re.I,
    )
    if target_inhibitor is None:
        return False
    if _RESPONSE_MODIFIER_PATTERN.search(excerpt):
        return True
    # A target inhibitor tested only in cells already treated with a distinct
    # named agent is combination context even if the abstract omits the words
    # "combination" and "co-treatment".
    return any(
        match.group("agent").casefold() != target_symbol.casefold()
        for match in _OTHER_AGENT_TREATED_PATTERN.finditer(excerpt)
    )


def _required_target_association_scope(
    brief: ResearchBrief,
    target_symbol: str,
) -> str | None:
    text = " ".join(
        value
        for value in (
            brief.research_goal,
            brief.rationale or "",
            *brief.human_priorities,
        )
        if value
    )
    if re.search(
        rf"(?:{re.escape(target_symbol)}\s*(?:[-\u2010-\u2015]\s*|\s+)"
        r"(?:gene\s+)?(?:rearrang(?:ed|ement|ements)|"
        rf"fusion(?:[- ]positive|s)?)\b{_MOLECULAR_NEGATIVE_SUFFIX_GUARD}|"
        rf"{re.escape(target_symbol)}\s*(?:재배열|융합))",
        text,
        re.I,
    ):
        return "rearranged_or_fusion_target_subtype"
    if (
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?:[- ]?ITD|\s+"
            r"internal tandem duplication)(?![A-Za-z0-9])",
            text,
            re.I,
        )
        or re.search(
            rf"(?<![A-Za-z0-9]){re.escape(target_symbol)}(?![A-Za-z0-9])"
            r"[^.!?]{0,40}\bactivating (?:mutation|variant)\b",
            text,
            re.I,
        )
    ):
        return "activating_or_itd_target_subtype"
    return None


def _epmc_oa_fulltext_primary_association_excerpt(
    brief: ResearchBrief,
    target_symbol: str,
    evidence: EvidenceRef,
) -> str | None:
    """Verify an OA association child without inheriting its parent role."""

    metadata = evidence.metadata
    excerpt = _epmc_oa_fulltext_contract_excerpt(evidence)
    if excerpt is None:
        return None
    expected_parent_id = "ev_" + sha256(
        (
            f"Europe PMC|{evidence.external_id}|oa_fulltext_excerpt|"
            f"{metadata.get('citation_hop')}|"
            f"{metadata.get('citation_relation')}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    expected_evidence_id = "ev_" + sha256(
        f"{expected_parent_id}|{_ASSOCIATION_ROLE_RECORD}".encode("utf-8")
    ).hexdigest()[:12]
    expected_relevance_basis = (
        "target_symbol",
        "disease",
        "primary_association_result",
        "association_predicate",
    )
    role_only_rejections = ("association_role_only",)
    association_score = metadata.get(
        "association_excerpt_relevance_score"
    )
    if (
        metadata.get("role_specific_record") != _ASSOCIATION_ROLE_RECORD
        or metadata.get("association_excerpt_selection_policy")
        != _PRIMARY_ASSOCIATION_EXCERPT_POLICY
        or metadata.get("excerpt_selection_status")
        != "selected_primary_association"
        or _metadata_text_sequence(metadata.get("excerpt_relevance_basis"))
        != expected_relevance_basis
        or metadata.get("association_predicate")
        not in _ASSOCIATION_PREDICATES
        or metadata.get("association_scope") not in _ASSOCIATION_SCOPES
        or not _is_nonnegative_int(association_score)
        or not 1 <= association_score <= 28
        or metadata.get("evidence_id_origin_policy")
        != "source-external-content-origin-hop-relation-v1"
        or evidence.evidence_id != expected_evidence_id
        or metadata.get("parent_evidence_id") != expected_parent_id
        or metadata.get("parent_evidence_id") == evidence.evidence_id
        or not _is_sha256_hex(metadata.get("parent_excerpt_hash"))
        or metadata.get("association_source_excerpt_hash")
        != evidence.excerpt_hash
        or metadata.get("parent_provenance_pointer_type")
        != "selected_oa_document_record_id_not_claim_link"
        or metadata.get("same_document_external_id") != evidence.external_id
        or metadata.get("duplicate_document_role") != "association_excerpt"
        or metadata.get("additional_network_call") is not False
        or metadata.get("direct_support_eligible") is not False
        or _metadata_text_sequence(
            metadata.get("direct_support_rejection_codes")
        )
        != role_only_rejections
        or metadata.get("verified_direct_primary") is not False
        or _metadata_text_sequence(
            metadata.get("verified_direct_primary_rejection_codes")
        )
        != role_only_rejections
        or metadata.get("full_text_span_excerpt_hash")
        != evidence.excerpt_hash
        or metadata.get("counter_category")
        or metadata.get("review_status")
        not in {None, "researcher_review_required"}
        or len(_stored_excerpt_sentences(excerpt)) != 1
        or not _has_atomic_target_disease_excerpt(
            excerpt,
            target_symbol=target_symbol,
            disease=brief.disease,
        )
    ):
        return None
    required_scope = _required_target_association_scope(
        brief,
        target_symbol,
    )
    if required_scope and metadata.get("association_scope") != required_scope:
        return None
    return excerpt


def _primary_association_excerpt(
    brief: ResearchBrief,
    target_symbol: str,
    evidence: EvidenceRef,
    scope_label: str | None = None,
) -> str | None:
    from drugpilot.agents.evidence_semantics import permits
    if not permits(evidence, brief, target_symbol, "association", scope_label):
        return None
    metadata = evidence.metadata
    if metadata.get("association_predicate") == "functional_target_disease_association":
        from drugpilot.agents.molecular_scope import functional_association_sentence
        if not functional_association_sentence(str(metadata.get("excerpt") or ""),target_symbol):
            return None
    if (
        metadata.get("content_availability")
        == _EPMC_OA_FULLTEXT_CONTENT_AVAILABILITY
    ):
        return _epmc_oa_fulltext_primary_association_excerpt(
            brief,
            target_symbol,
            evidence,
        )
    excerpt = metadata.get("excerpt")
    if (
        _source_supports(evidence) != "support"
        or evidence.source not in {"PubMed", "Europe PMC"}
        or metadata.get("role_specific_record") != _ASSOCIATION_ROLE_RECORD
        or not isinstance(excerpt, str)
        or not excerpt.strip()
        or not evidence.excerpt_hash
        or metadata.get("content_availability") != "abstract_excerpt"
        or not str(metadata.get("source_location") or "").startswith("abstract:")
        or not metadata.get("content_hash")
        or metadata.get("full_text_stored") is not False
        or metadata.get("excerpt_selection_policy")
        != _PRIMARY_ASSOCIATION_EXCERPT_POLICY
        or metadata.get("association_predicate") not in _ASSOCIATION_PREDICATES
        or not metadata.get("parent_evidence_id")
        or metadata.get("parent_evidence_id") == evidence.evidence_id
        or metadata.get("same_document_external_id") != evidence.external_id
        or metadata.get("duplicate_document_role") != "association_excerpt"
        or metadata.get("provenance_class") not in _EXTERNAL_PROVENANCE_CLASSES
        or metadata.get("counter_category")
        or metadata.get("review_status")
    ):
        return None
    if not _has_verified_primary_publication_type(evidence):
        return None
    if len(_stored_excerpt_sentences(excerpt)) != 1:
        return None
    if not _has_atomic_target_disease_excerpt(
        excerpt,
        target_symbol=target_symbol,
        disease=brief.disease,
    ):
        return None
    required_scope = _required_target_association_scope(
        brief,
        target_symbol,
    )
    if required_scope and metadata.get("association_scope") != required_scope:
        return None
    return excerpt


def _has_verified_compound_linked_direction_pair(
    brief: ResearchBrief,
    target_symbol: str,
    evidence: EvidenceRef,
    excerpt: str,
    intervention_direction: str | None = None,
) -> bool:
    """Re-verify a document-local drug/target anchor and outcome pair.

    A structured abstract may identify a target-selective drug in its
    background and report the same drug's outcome in Results or Conclusions.
    This is accepted only when the identical compound token links both stored
    sentences; metadata alone cannot turn a multi-sentence excerpt into proof.
    """

    direction = intervention_direction or brief.constraints.intervention_direction
    metadata = evidence.metadata
    rejection_codes = metadata.get("verified_direct_primary_rejection_codes", ())
    if isinstance(rejection_codes, str):
        rejection_codes = (rejection_codes,)
    if (
        metadata.get("verified_direct_primary") is not True
        or not isinstance(rejection_codes, Sequence)
        or isinstance(rejection_codes, (str, bytes))
        or any(str(code).strip() for code in rejection_codes)
    ):
        return False

    sentences = _stored_excerpt_sentences(excerpt)
    if len(sentences) < 2:
        return False
    target = re.escape(target_symbol)
    compound_before = re.compile(
        rf"\b(?P<compound>[A-Za-z][A-Za-z0-9-]{{2,30}})\s*,?\s+"
        rf"(?:(?:is|was)\s+)?(?:an?\s+)?(?:oral\s+)?{target}"
        rf"(?:[- ](?:selective|specific))?[- ]"
        rf"(?:(?:tyrosine[- ]kinase|kinase)[- ])?inhibitor\b",
        re.I,
    )
    compound_after = re.compile(
        rf"\b{target}(?:[- ](?:selective|specific))?[- ]"
        rf"(?:(?:tyrosine[- ]kinase|kinase)[- ])?inhibitor\s+"
        rf"(?P<compound>[A-Za-z][A-Za-z0-9-]{{2,30}})\b",
        re.I,
    )
    location = str(metadata.get("source_location") or "")
    result_section = re.search(
        r"abstract:[^|]*(?:RESULT|CONCLUSION|FINDING)",
        location,
        re.I,
    ) is not None
    normalized_disease = " ".join(brief.disease.casefold().split())
    disease_aliases = tuple(
        dict.fromkeys(
            (
                *_disease_initialisms(brief.disease),
                *(
                    match.group(1).upper()
                    for match in re.finditer(
                        rf"{re.escape(brief.disease)}\s*\(\s*"
                        r"([A-Z][A-Z0-9-]{1,8})\s*\)",
                        excerpt,
                        re.I,
                    )
                ),
            )
        )
    )

    for anchor in sentences:
        if _direct_target_intervention_pattern(
            target_symbol,
            direction,
        ).search(anchor) is None:
            continue
        match = compound_before.search(anchor) or compound_after.search(anchor)
        if match is None:
            continue
        compound = match.group("compound")
        compound_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(compound)}(?![A-Za-z0-9])",
            re.I,
        )
        for result in sentences:
            if result == anchor or compound_pattern.search(result) is None:
                continue
            normalized_result = " ".join(result.casefold().split())
            has_disease = normalized_disease in normalized_result or any(
                re.search(
                    rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])",
                    result,
                )
                for alias in disease_aliases
            )
            if not has_disease:
                continue
            if _MEASURED_ENDPOINT_PATTERN.search(result) is None:
                continue
            if _EXPERIMENTAL_RESULT_PATTERN.search(result) is None:
                continue
            if _direction_endpoint_polarity_status(
                result,
                direction,
            ) != "favorable":
                continue
            if not result_section and re.search(
                r"\b(?:significant(?:ly)?|P\s*[<=>]|\d+(?:\.\d+)?%)\b",
                result,
                re.I,
            ) is None:
                continue
            return True
    return False


def _primary_direction_excerpt(
    brief: ResearchBrief,
    target_symbol: str,
    evidence: EvidenceRef,
    scope_label: str | None = None,
    *,
    intervention_direction: str | None = None,
) -> str | None:
    requested = brief.constraints.intervention_direction
    direction = intervention_direction or requested
    if direction == "unspecified" or (direction == "modulate" and intervention_direction is None):
        # Broad retrieval is never a therapeutic direction. Check each concrete
        # source-supported direction; the proposer later separates hypotheses.
        for resolved in ("inhibit", "activate", "modulate"):
            excerpt = _primary_direction_excerpt(
                brief, target_symbol, evidence, scope_label,
                intervention_direction=resolved,
            )
            if excerpt is not None:
                return excerpt
        return None
    if direction not in _DIRECTION_PATTERNS or (
        requested not in {"modulate", "unspecified"} and direction != requested
    ):
        return None
    from drugpilot.agents.evidence_semantics import permits
    if not permits(evidence, brief, target_symbol, "direction", scope_label):
        return None
    metadata = evidence.metadata
    if "document_result_unit" in metadata:
        if direction != "inhibit":
            return None  # The current adjacent-result source grammar proves inhibition only.
        from drugpilot.agents.document_result_units import validated_unit_excerpt
        return validated_unit_excerpt(brief, target_symbol, evidence, scope_label)
    if (
        metadata.get("content_availability")
        == _EPMC_OA_FULLTEXT_CONTENT_AVAILABILITY
    ):
        return _epmc_oa_fulltext_primary_direction_excerpt(
            brief,
            target_symbol,
            evidence,
            direction,
        )
    excerpt = metadata.get("excerpt")
    source_location = str(metadata.get("source_location") or "")
    paired_compound_result = bool(
        metadata.get("excerpt_selection_status")
        == "selected_compound_linked_document_pair"
        and source_location.startswith("abstract-paired:")
        and metadata.get("excerpt_selection_policy")
        == _COMPOUND_LINKED_DIRECTION_EXCERPT_POLICY
    )
    single_sentence_result = bool(
        source_location.startswith("abstract:")
        and metadata.get("excerpt_selection_policy") == _PRIMARY_EXCERPT_POLICY
    )
    if (
        _source_supports(evidence) != "support"
        or evidence.source not in {"PubMed", "Europe PMC"}
        or not isinstance(excerpt, str)
        or not excerpt.strip()
        or not evidence.excerpt_hash
        or metadata.get("content_availability") != "abstract_excerpt"
        or not (single_sentence_result or paired_compound_result)
        or not metadata.get("content_hash")
        or metadata.get("full_text_stored") is not False
        or metadata.get("query_strategy") not in _PRIMARY_QUERY_POLICIES
        or metadata.get("query_intervention_direction")
        != requested
        or metadata.get("provenance_class") not in _EXTERNAL_PROVENANCE_CLASSES
        or metadata.get("counter_category")
        or metadata.get("review_status")
    ):
        return None
    if not _has_verified_primary_publication_type(evidence):
        return None
    if paired_compound_result:
        if not _has_verified_compound_linked_direction_pair(
            brief,
            target_symbol,
            evidence,
            excerpt,
            direction,
        ):
            return None
    elif len(_stored_excerpt_sentences(excerpt)) != 1:
        return None
    literal_disease_context = _has_atomic_target_disease_excerpt(
        excerpt,
        target_symbol=target_symbol,
        disease=brief.disease,
    )
    predeclared_model_context = (
        _has_verified_predeclared_direction_model_context(
            brief,
            target_symbol,
            excerpt,
            metadata,
        )
    )
    if not (literal_disease_context or predeclared_model_context):
        return None
    direction_pattern = _DIRECTION_PATTERNS[direction]
    if direction_pattern.search(excerpt) is None:
        return None
    if _is_response_modifier_context(target_symbol, evidence):
        return None
    if not _scope_direct_intervention_present(excerpt,target_symbol,
            direction,scope_label):
        return None
    if not _has_admissible_direction_endpoint(excerpt, target_symbol):
        return None
    if _MEASURED_ENDPOINT_PATTERN.search(excerpt) is None:
        return None
    if _EXPERIMENTAL_RESULT_PATTERN.search(excerpt) is None:
        return None
    if _direction_endpoint_polarity_status(
        excerpt,
        direction,
    ) != "favorable":
        return None
    if _NON_DIRECTION_PRIMARY_PATTERN.search(excerpt):
        return None
    return excerpt


def _scope_direct_intervention_present(excerpt, target, direction, scope_label=None):
    if scope_label is None:
        return _direct_target_intervention_pattern(target,direction).search(excerpt) is not None
    from drugpilot.agents.molecular_scope import mentions
    # Canonical-target action in a mutant-cell sentence is not enough to
    # claim action on the requested molecular scope.  The scoped construct
    # itself must own the intervention predicate (for example,
    # ``FLT3-ITD inhibition``).  This deliberately fails closed even when a
    # model interprets the broader sentence as variant relevant.
    for m in mentions(excerpt,target):
        if m["label"] != scope_label:
            continue
        for entity in (m["quote"],m["quote"]+" fusion",m["quote"]+" fusion protein"):
            if _direct_target_intervention_pattern(entity,direction).search(excerpt):
                return True
    return False


def _primary_direction_score(evidence: EvidenceRef, excerpt: str) -> tuple[int, str]:
    endpoint_hits = len(_MEASURED_ENDPOINT_PATTERN.findall(excerpt))
    result_hits = len(_EXPERIMENTAL_RESULT_PATTERN.findall(excerpt))
    location = str(evidence.metadata.get("source_location") or "")
    score = (
        endpoint_hits * 10
        + result_hits * 6
        + (4 if evidence.source == "PubMed" else 3)
        + (2 if ":RESULT" in location.upper() else 0)
        + (1 if evidence.metadata.get("provenance_class") == "external_live" else 0)
    )
    return score, evidence.evidence_id


def _is_ranked_association_evidence(evidence: EvidenceRef) -> bool:
    return bool(
        _source_supports(evidence) == "support"
        and evidence.source == "Open Targets"
        and evidence.metadata.get("relationship") == "ranked_association"
        and evidence.metadata.get("record_type") == "target_disease_association"
        and evidence.metadata.get("provenance_class")
        in _EXTERNAL_PROVENANCE_CLASSES
        and evidence.metadata.get("mock") is not True
    )


def _assign_claim_evidence(
    brief: ResearchBrief,
    *,
    target_symbol: str,
    evidence: Sequence[EvidenceRef],
    scope_label: str | None = None,
    intervention_direction: str | None = None,
) -> _ClaimEvidenceAssignment:
    primary: list[tuple[int, str]] = []
    association_primary: list[tuple[int, str]] = []
    ranked_association_context_ids: list[str] = []
    response_modifier_context_ids: list[str] = []
    resistance_context_ids: list[str] = []
    other_counter_context_ids: list[str] = []
    fixture_support_ids: list[str] = []

    for item in evidence:
        if "document_result_unit" in item.metadata:
            from drugpilot.agents.document_result_units import parent_is_bound
            if not parent_is_bound(item, evidence):
                continue
        if _is_response_modifier_context(target_symbol, item):
            response_modifier_context_ids.append(item.evidence_id)
        else:
            excerpt = _primary_direction_excerpt(
                brief, target_symbol, item, scope_label,
                intervention_direction=intervention_direction,
            )
            if excerpt is not None:
                primary.append(_primary_direction_score(item, excerpt))
        association_excerpt = _primary_association_excerpt(
            brief,
            target_symbol,
            item,
            scope_label,
        )
        if (
            association_excerpt is not None
            and item.metadata.get("content_availability")
            == _EPMC_OA_FULLTEXT_CONTENT_AVAILABILITY
            and not _epmc_oa_fulltext_association_parent_is_bound(
                item,
                evidence,
            )
        ):
            association_excerpt = None
        if association_excerpt is not None:
            association_primary.append(
                (
                    int(
                        item.metadata.get(
                            "association_excerpt_relevance_score"
                        )
                        or 0
                    ),
                    item.evidence_id,
                )
            )
        if _is_ranked_association_evidence(item):
            ranked_association_context_ids.append(item.evidence_id)
        source_supports = _source_supports(item)
        if scope_label is None and source_supports == "support" and item.metadata.get("mock") is True:
            fixture_support_ids.append(item.evidence_id)
        if source_supports != "counter":
            continue
        if item.metadata.get("counter_category") == "resistance":
            resistance_context_ids.append(item.evidence_id)
        else:
            other_counter_context_ids.append(item.evidence_id)

    primary_ids = [
        evidence_id
        for _, evidence_id in sorted(
            primary,
            key=lambda item: (-item[0], item[1]),
        )[:24]
    ]
    association_primary_ids = [
        evidence_id
        for _, evidence_id in sorted(
            association_primary,
            key=lambda item: (-item[0], item[1]),
        )[:8]
    ]
    # Keep both database records, but do not turn one source excerpt into two
    # independent support units after semantic cross-source interpretation.
    def deduplicate_interpreted(ids: list[str]) -> list[str]:
        by_id = {item.evidence_id: item for item in evidence}
        seen: set[str] = set()
        unique: list[str] = []
        for evidence_id in ids:
            annotation = by_id[evidence_id].metadata.get("semantic_interpretation")
            key = annotation.get("record_key") if isinstance(annotation, Mapping) else evidence_id
            if key not in seen:
                unique.append(evidence_id)
                seen.add(key)
        return unique
    primary_ids = deduplicate_interpreted(primary_ids)
    association_primary_ids = deduplicate_interpreted(association_primary_ids)
    fixture_compatibility = bool(fixture_support_ids)
    if not primary_ids and fixture_support_ids:
        primary_ids = fixture_support_ids[:1]
    if not association_primary_ids and fixture_support_ids:
        association_primary_ids = fixture_support_ids[:1]

    # Resistance to a particular inhibitor is an important limitation, but it
    # is not a direct falsification of basic target-direction biology.  Only a
    # future explicit target-direction null-result classifier may populate the
    # refutation set; it is intentionally empty under this v1 policy.
    return _ClaimEvidenceAssignment(
        direction_primary_ids=tuple(primary_ids),
        direction_refutation_ids=(),
        association_primary_ids=tuple(
            dict.fromkeys(association_primary_ids)
        ),
        ranked_association_context_ids=tuple(
            dict.fromkeys(ranked_association_context_ids)
        ),
        response_modifier_context_ids=tuple(
            dict.fromkeys(response_modifier_context_ids)
        ),
        resistance_context_ids=tuple(dict.fromkeys(resistance_context_ids)),
        other_counter_context_ids=tuple(
            dict.fromkeys(other_counter_context_ids)
        ),
        fixture_compatibility=fixture_compatibility,
    )


def _direction_noun(direction: str) -> str:
    return {
        "inhibit": "inhibition",
        "activate": "activation",
        "modulate": "modulation",
        "unspecified": "direction-unresolved perturbation",
    }[direction]


def _prospective_mechanism(brief: ResearchBrief, target_symbol: str, intervention_direction: str | None = None) -> str:
    direction = _direction_noun(intervention_direction or brief.constraints.intervention_direction)
    return (
        f"{target_symbol} {direction} in {brief.disease} is a prospective, "
        "experimentally falsifiable mechanism; linked evidence does not by itself "
        "establish candidate-specific target engagement or efficacy."
    )


def _measured_endpoints(excerpt: str) -> tuple[str, ...]:
    return tuple(
        name for name, pattern in _ENDPOINT_PATTERNS if pattern.search(excerpt)
    )


def _direct_intervention_basis(
    excerpt: str,
    target_symbol: str,
) -> str:
    target = re.escape(target_symbol)
    if re.search(rf"\bbound(?:\s+directly)?\s+to\s+{target}\b", excerpt, re.I):
        return "direct_target_binding_and_perturbation"
    if re.search(rf"\bbound\s+{target}\b", excerpt, re.I):
        return "reported_compound_target_binding_with_correlative_phenotype"
    if re.search(rf"\b{target}(?:[- ](?:selective|specific))?[- ]inhibitors?\b", excerpt, re.I):
        return "target_inhibitor_intervention"
    return "direct_target_perturbation"


def _atomic_claims_and_links(
    brief: ResearchBrief,
    *,
    target_symbol: str,
    track: HypothesisTrack,
    assignment: _ClaimEvidenceAssignment,
    evidence: Sequence[EvidenceRef],
    scope_label: str | None = None,
    intervention_direction: str | None = None,
) -> tuple[
    list[Claim],
    dict[str, list[tuple[str, str]]],
    list[dict[str, Any]],
]:
    """Build one executable claim per immutable evidence excerpt."""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    claims: list[Claim] = []
    links: dict[str, list[tuple[str, str]]] = {}
    units: list[dict[str, Any]] = []
    direction = _direction_noun(intervention_direction or brief.constraints.intervention_direction)
    claim_target = scope_label or target_symbol
    target_key = target_symbol if scope_label is None else target_symbol + "|" + scope_label
    if brief.constraints.intervention_direction in {"modulate", "unspecified"}:
        target_key += "|" + (intervention_direction or "unresolved")

    def add_unit(
        *,
        evidence_id: str,
        role: str,
        text: str,
        confidence: float,
        predicate: str,
        intervention_basis: str | None,
        endpoints: Sequence[str],
        counter_context_ids: Sequence[str] = (),
    ) -> None:
        item = evidence_by_id[evidence_id]
        claim_id = _stable_id(
            "claim",
            brief.brief_id,
            target_key,
            role,
            evidence_id,
        )
        claims.append(
            Claim(
                claim_id=claim_id,
                text=text,
                evidence_ids=[evidence_id],
                counter_evidence_ids=list(counter_context_ids),
                confidence=confidence,
            )
        )
        links.setdefault(evidence_id, []).append((claim_id, role))
        for counter_id in counter_context_ids:
            links.setdefault(counter_id, []).append(
                (claim_id, "claim_level_counter_context")
            )
        units.append(
            {
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "role": role,
                "molecular_scope_label": scope_label,
                "predicate": predicate,
                "publication": {
                    "source": item.source,
                    "external_id": item.external_id,
                },
                "intervention_basis": intervention_basis,
                "measured_endpoints": list(endpoints),
                "excerpt_hash": item.excerpt_hash,
                "source_location": item.metadata.get("source_location"),
                "same_document_external_id": item.metadata.get(
                    "same_document_external_id"
                ),
                "parent_evidence_id": item.metadata.get("parent_evidence_id"),
                "evidence_cardinality": 1,
                "counter_context_ids": list(counter_context_ids),
                "counter_context_cardinality": len(counter_context_ids),
            }
        )

    limitation_ids = tuple(
        dict.fromkeys(
            (
                *assignment.resistance_context_ids,
                *assignment.other_counter_context_ids,
            )
        )
    )
    for evidence_id in assignment.direction_primary_ids:
        item = evidence_by_id[evidence_id]
        excerpt = str(item.metadata.get("excerpt") or "")
        endpoints = _measured_endpoints(excerpt)
        endpoint_text = ", ".join(endpoints) or "a measured endpoint"
        add_unit(
            evidence_id=evidence_id,
            role="direct_target_direction_primary",
            text=(
                f"{item.source} {item.external_id} reports a bounded primary "
                f"result for direct {claim_target} {direction} in "
                f"{brief.disease}, with {endpoint_text}; this claim is limited "
                "to that publication and excerpt."
            ),
            confidence=0.5 if assignment.fixture_compatibility else 0.76,
            predicate="direct target-direction measured result",
            intervention_basis=_direct_intervention_basis(
                excerpt,
                target_symbol,
            ),
            endpoints=endpoints,
            counter_context_ids=limitation_ids,
        )

    if not assignment.direction_primary_ids:
        claim_id = _stable_id(
            "claim",
            brief.brief_id,
            target_key,
            "missing_direct_target_direction_primary",
        )
        claims.append(
            Claim(
                claim_id=claim_id,
                text=(
                    f"No verified primary abstract excerpt directly supports "
                    f"{claim_target} {direction} with a measured result in "
                    f"{brief.disease}."
                ),
                evidence_ids=[],
                counter_evidence_ids=[],
                confidence=0.25,
            )
        )
        units.append(
            {
                "claim_id": claim_id,
                "evidence_id": None,
                "role": "missing_direct_target_direction_primary",
                "predicate": "direct target-direction measured result",
                "status": "fail_closed_missing_primary_evidence",
                "evidence_cardinality": 0,
            }
        )

    for evidence_id in assignment.association_primary_ids:
        item = evidence_by_id[evidence_id]
        predicate = str(item.metadata.get("association_predicate") or "association")
        scope = str(item.metadata.get("association_scope") or "target_general")
        add_unit(
            evidence_id=evidence_id,
            role="primary_target_disease_association",
            text=(
                f"{item.source} {item.external_id} reports a bounded primary "
                f"{predicate.replace('_', ' ')} result linking {claim_target} "
                f"to {brief.disease} at {scope.replace('_', ' ')} scope; this "
                "is not intervention-direction proof."
            ),
            confidence=0.5 if assignment.fixture_compatibility else 0.74,
            predicate=predicate,
            intervention_basis=None,
            endpoints=(),
        )

    if (
        brief.data_policy.require_primary_sources
        and not assignment.association_primary_ids
    ):
        claim_id = _stable_id(
            "claim",
            brief.brief_id,
            target_key,
            "missing_primary_target_disease_association",
        )
        claims.append(
            Claim(
                claim_id=claim_id,
                text=(
                    f"No verified primary-publication association or prevalence "
                    f"excerpt links {claim_target} to {brief.disease}; ranked "
                    "aggregator records remain discovery context only."
                ),
                evidence_ids=[],
                counter_evidence_ids=[],
                confidence=0.25,
            )
        )
        units.append(
            {
                "claim_id": claim_id,
                "evidence_id": None,
                "role": "missing_primary_target_disease_association",
                "predicate": "primary target-disease association",
                "status": "fail_closed_missing_primary_evidence",
                "evidence_cardinality": 0,
            }
        )
    return claims, links, units


def _claim_assignment_metadata(
    assignment: _ClaimEvidenceAssignment,
    *,
    claim_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mechanism_units = [
        dict(item)
        for item in claim_units
        if "direction" in str(item.get("role"))
    ]
    association_units = [
        dict(item)
        for item in claim_units
        if "association" in str(item.get("role"))
    ]
    return {
        "policy": _DIRECTION_PRIMARY_POLICY,
        "atomic_claim_linkage": not assignment.fixture_compatibility,
        "atomicity_contract": "one executable claim to one bounded excerpt",
        "claim_units": [dict(item) for item in claim_units],
        "mechanism_claims": mechanism_units,
        "association_claims": association_units,
        "ranked_association_discovery_context_ids": list(
            assignment.ranked_association_context_ids
        ),
        "direction_refutation_evidence_ids": list(
            assignment.direction_refutation_ids
        ),
        "response_modifier_context": {
            "evidence_ids": list(assignment.response_modifier_context_ids),
            "interpretation": (
                "combination or sensitizer context; not direct target-direction "
                "support"
            ),
        },
        "resistance_context": {
            "evidence_ids": list(assignment.resistance_context_ids),
            "interpretation": (
                "resistance limitation context; candidate specificity is not "
                "assumed and this is not basic target-direction refutation"
            ),
            "linked_as_direction_refutation": False,
            "linked_as_claim_limitation": bool(
                assignment.direction_primary_ids
                and assignment.resistance_context_ids
            ),
        },
        "other_counter_context_ids": list(
            assignment.other_counter_context_ids
        ),
        "fixture_compatibility": assignment.fixture_compatibility,
    }


def _screening_direction_triage(hypothesis: Hypothesis) -> dict[str, Any]:
    """Classify only direction states supported by persisted claim linkage.

    Discovery rank, tractability, loss-of-function biology, and target names
    are deliberately ignored.  A variant-scope limitation is recorded only
    when the evidence metadata explicitly says that a linked primary record is
    subtype-specific and no predeclared matching target variant was retained.
    """

    assignment = hypothesis.target.metadata.get("claim_evidence_assignment")
    assignment = assignment if isinstance(assignment, Mapping) else {}
    raw_units = assignment.get("claim_units")
    units = (
        [item for item in raw_units if isinstance(item, Mapping)]
        if isinstance(raw_units, Sequence) and not isinstance(raw_units, (str, bytes))
        else []
    )
    evidence_by_id = {item.evidence_id: item for item in hypothesis.evidence}

    def linked_ids(role: str) -> tuple[str, ...]:
        ids: list[str] = []
        for unit in units:
            evidence_id = str(unit.get("evidence_id") or "").strip()
            if (
                unit.get("role") == role
                and unit.get("evidence_cardinality") == 1
                and evidence_id in evidence_by_id
            ):
                ids.append(evidence_id)
        return tuple(dict.fromkeys(ids))

    direct_ids = linked_ids("direct_target_direction_primary")
    association_ids = linked_ids("primary_target_disease_association")
    raw_refutations = assignment.get("direction_refutation_evidence_ids", ())
    refutation_ids = tuple(
        evidence_id
        for evidence_id in (
            str(item).strip()
            for item in raw_refutations
        )
        if evidence_id in evidence_by_id
    ) if isinstance(raw_refutations, Sequence) and not isinstance(
        raw_refutations, (str, bytes)
    ) else ()

    subtype_scopes = {
        "activating_or_itd_target_subtype",
        "rearranged_or_fusion_target_subtype",
    }
    scope_evidence_ids: list[str] = []
    declared_variants: list[str] = []
    # Variant scope affects direction executability only when the *direct
    # target-direction* primary record is explicitly subtype-specific.  A
    # subtype-specific disease-association record alone cannot establish or
    # refute the intervention direction and must not create a false direction
    # conflict.
    for evidence_id in direct_ids:
        metadata = evidence_by_id[evidence_id].metadata
        if str(metadata.get("association_scope") or "") in subtype_scopes:
            scope_evidence_ids.append(evidence_id)
        raw_declared = metadata.get("predeclared_target_variants", ())
        if isinstance(raw_declared, Sequence) and not isinstance(
            raw_declared, (str, bytes)
        ):
            declared_variants.extend(
                str(item).strip() for item in raw_declared if str(item).strip()
            )

    observed_scopes = []
    unresolved_direction_scopes = []
    for item in hypothesis.evidence:
        annotation = item.metadata.get("molecular_scope")
        if not isinstance(annotation, Mapping):
            continue
        scope = annotation.get("source_scope") or {}
        if scope.get("source_verified") and scope.get("mentions"):
            observed_scopes.append({"evidence_id":item.evidence_id,
                "source":item.source,"external_id":item.external_id,
                "observed_labels":annotation.get("observed_labels", []),
                "declared_labels":annotation.get("declared_labels", []),
                "status":annotation.get("status"),
                "claim_roles":[u.get("role") for u in units if u.get("evidence_id") == item.evidence_id],
                "excerpt_sha256":item.excerpt_hash})
        # This new span annotation is not a replacement for historical primary
        # validators. Only verified source spans can establish a scope conflict.
        branch = hypothesis.target.metadata.get("molecular_hypothesis_scope") or {}
        branch_matches = bool(branch.get("label") and
            ((item.metadata.get("semantic_interpretation") or {}).get("extractor") or {}).get("molecular_scope_label") == branch.get("label"))
        if not branch_matches and scope.get("source_verified") and item.evidence_id in direct_ids and annotation.get("status") in {
            "scoped_source_not_general_target", "declared_observed_scope_mismatch",
            "declared_scope_present_with_additional_scopes", "scope_unverifiable"}:
            unresolved_direction_scopes.append(item.evidence_id)
    if unresolved_direction_scopes:
        status = "variant_scope_unresolved"
        basis = "source_scope_cannot_be_generalized_to_requested_target"
    elif scope_evidence_ids and not declared_variants and not hypothesis.target.metadata.get("molecular_hypothesis_scope"):
        status = "variant_scope_unresolved"
        basis = "explicit_subtype_scope_without_predeclared_variant"
    elif direct_ids:
        status = "aligned_direct_primary"
        basis = "linked_direct_target_direction_primary"
    elif refutation_ids:
        status = "discordant_direct_primary"
        basis = "linked_direct_target_direction_refutation"
    else:
        status = "unknown_no_direct_primary"
        basis = "no_linked_direct_target_direction_primary"

    return {
        "status": status,
        "basis": basis,
        "intervention_direction": hypothesis.intervention_direction,
        "direct_primary_evidence_ids": list(direct_ids),
        "association_primary_evidence_ids": list(association_ids),
        "direction_refutation_evidence_ids": list(dict.fromkeys(refutation_ids)),
        "variant_scope_evidence_ids": list(dict.fromkeys(scope_evidence_ids)),
        "predeclared_target_variants": list(dict.fromkeys(declared_variants)),
        "molecular_hypothesis_scope": hypothesis.target.metadata.get("molecular_hypothesis_scope"),
        "observed_molecular_scopes": observed_scopes[:24],
        "observed_molecular_scope_count": len(observed_scopes),
        "observed_scope_is_not_predeclared_requirement": True,
        "unresolved_direction_scope_evidence_ids": unresolved_direction_scopes,
        "inference_forbidden": True,
    }


def _reconcile_evidence_claim_linkage(
    evidence: Sequence[EvidenceRef],
    assignment: _ClaimEvidenceAssignment,
    *,
    claim_links: Mapping[str, Sequence[tuple[str, str]]],
) -> list[EvidenceRef]:
    """Copy evidence with semantic role, support polarity and claim linkage."""

    refutation_ids = set(assignment.direction_refutation_ids)
    ranked_association_ids = set(assignment.ranked_association_context_ids)
    modifier_ids = set(assignment.response_modifier_context_ids)
    resistance_ids = set(assignment.resistance_context_ids)
    other_counter_ids = set(assignment.other_counter_context_ids)
    reconciled: list[EvidenceRef] = []
    for item in evidence:
        source_supports = _source_supports(item)
        links = list(claim_links.get(item.evidence_id, ()))
        linked_claim_ids = list(dict.fromkeys(claim_id for claim_id, _ in links))
        roles = list(dict.fromkeys(role for _, role in links))
        if item.evidence_id in modifier_ids:
            roles.append("response_modifier_context_unlinked")
        if item.evidence_id in ranked_association_ids:
            roles.append("ranked_association_discovery_context_unlinked")
        if item.evidence_id in resistance_ids:
            roles.append(
                "resistance_context_linked"
                if linked_claim_ids
                else "resistance_context_unlinked"
            )
        elif item.evidence_id in other_counter_ids:
            roles.append(
                "counter_context_linked"
                if linked_claim_ids
                else "counter_context_unlinked"
            )
        if source_supports not in {"support", "counter", "context"}:
            roles = ["invalid_source_supports_context_unlinked"]
            linked_claim_ids = []
        if not roles:
            roles.append("context_unlinked")
        support_role = any(
            role in {
                "direct_target_direction_primary",
                "primary_target_disease_association",
            }
            for role in roles
        )
        direct_support_role = bool(
            linked_claim_ids
            and "direct_target_direction_primary" in roles
        )
        direct_rejection_codes: list[str] = []
        for metadata_key in (
            "direct_support_rejection_codes",
            "verified_direct_primary_rejection_codes",
        ):
            raw_codes = item.metadata.get(metadata_key, ())
            if isinstance(raw_codes, str):
                raw_codes = (raw_codes,)
            if isinstance(raw_codes, Sequence) and not isinstance(
                raw_codes,
                (str, bytes),
            ):
                direct_rejection_codes.extend(
                    str(code)
                    for code in raw_codes
                    if str(code).strip()
                )
        direct_rejection_codes = list(dict.fromkeys(direct_rejection_codes))
        if (
            not direct_support_role
            and not direct_rejection_codes
            and (
                item.metadata.get("direct_support_eligible") is True
                or item.metadata.get("verified_direct_primary") is True
            )
        ):
            direct_rejection_codes.append(
                "claim_reconciliation_not_direct_target_direction_primary"
            )
        reconciled_supports = (
            "counter"
            if item.evidence_id in refutation_ids
            else "counter"
            if assignment.fixture_compatibility and source_supports == "counter"
            else "support"
            if support_role
            else "context"
        )
        reconciled.append(
            item.model_copy(
                update={
                    "supports": reconciled_supports,
                    "metadata": {
                        **item.metadata,
                        "claim_linkage_status": "reconciled",
                        "claim_linkage_policy": _DIRECTION_PRIMARY_POLICY,
                        "linked_claim_ids": linked_claim_ids,
                        "claim_linkage_roles": list(dict.fromkeys(roles)),
                        "claim_linkage_scope": (
                            "claim_specific" if linked_claim_ids else "unlinked_context"
                        ),
                        "source_supports": source_supports,
                        "direct_support_eligible": direct_support_role,
                        "verified_direct_primary": direct_support_role,
                        "direct_support_rejection_codes": (
                            [] if direct_support_role else direct_rejection_codes
                        ),
                        "verified_direct_primary_rejection_codes": (
                            [] if direct_support_role else direct_rejection_codes
                        ),
                    },
                },
                deep=True,
            )
        )
    return reconciled


def _counter_search_summary(
    assignment: _ClaimEvidenceAssignment,
    evidence: Sequence[EvidenceRef],
) -> dict[str, Any]:
    counter_metadata = [
        item.metadata
        for item in evidence
        if item.metadata.get("counter_search_status")
    ]
    statuses = {
        str(metadata.get("counter_search_status"))
        for metadata in counter_metadata
    }
    sources = {
        str(metadata.get("counter_search_source"))
        for metadata in counter_metadata
        if metadata.get("counter_search_source")
    }
    queries = {
        str(metadata.get("counter_search_query"))
        for metadata in counter_metadata
        if metadata.get("counter_search_query")
    }
    query_hashes = {
        str(metadata.get("counter_search_query_sha256"))
        for metadata in counter_metadata
        if metadata.get("counter_search_query_sha256")
    }
    http_provenance = [
        dict(value)
        for metadata in counter_metadata
        if isinstance(
            (value := metadata.get("counter_search_http_call_provenance")),
            Mapping,
        )
        and value
    ]
    real_count = len(assignment.direction_refutation_ids)
    resistance_count = len(assignment.resistance_context_ids)
    other_context_count = len(assignment.other_counter_context_ids)
    modifier_count = len(assignment.response_modifier_context_ids)
    recorded_raw_hit_counts = [
        int(value)
        for metadata in counter_metadata
        if isinstance(
            (value := metadata.get("counter_search_raw_hit_count")),
            int,
        )
        and value >= 0
    ]
    raw_hit_count = (
        max(recorded_raw_hit_counts)
        if recorded_raw_hit_counts
        else resistance_count + other_context_count
    )
    raw_search_status = (
        "error"
        if "error" in statuses
        else "completed_with_results"
        if "completed_with_results" in statuses
        else "completed_empty"
        if "completed_empty" in statuses
        else "not_recorded"
    )
    claim_linked_refutation_status = (
        "completed_with_results"
        if real_count
        else "completed_empty"
        if raw_search_status in {"completed_empty", "completed_with_results"}
        else "error"
        if raw_search_status == "error"
        else "not_recorded"
    )
    return {
        # ``status`` describes the actual bounded source search, not merely
        # the narrower subset that can be linked as a direct refutation.  A
        # resistance/context result must never be summarized as an empty
        # search just because it is intentionally excluded from claim-level
        # refutation.
        "status": raw_search_status,
        "source": (
            next(iter(sources))
            if len(sources) == 1
            else "; ".join(sorted(sources))
            if sources
            else None
        ),
        "sources": sorted(sources),
        "query": next(iter(queries)) if len(queries) == 1 else None,
        "queries": sorted(queries),
        "query_sha256": (
            next(iter(query_hashes)) if len(query_hashes) == 1 else None
        ),
        "query_sha256s": sorted(query_hashes),
        "http_call_provenance": (
            http_provenance[0]
            if http_provenance
            and all(item == http_provenance[0] for item in http_provenance)
            else {}
        ),
        "separate_query": bool(
            counter_metadata
            and all(
                metadata.get("counter_search_separate_query") is True
                for metadata in counter_metadata
            )
        ),
        "real_counter_count": real_count,
        "claim_linked_refutation_status": claim_linked_refutation_status,
        "raw_search_status": raw_search_status,
        "raw_hit_count": raw_hit_count,
        "retrieved_resistance_context_count": resistance_count,
        "retrieved_other_counter_context_count": other_context_count,
        "response_modifier_context_count": modifier_count,
        "fixture_used": assignment.fixture_compatibility,
        "classification_policy": (
            "claim-linked-direct-target-refutation-v2; resistance-context-excluded"
        ),
    }


def _reclassify_prior_claim_evidence(
    brief: ResearchBrief,
    hypothesis: Hypothesis,
) -> Hypothesis:
    """Reapply atomic linkage to a persisted MA1 hypothesis during G0 rework."""
    scope_label = (hypothesis.target.metadata.get("molecular_hypothesis_scope") or {}).get("label")
    assignment = _assign_claim_evidence(
        brief,
        target_symbol=hypothesis.target.symbol,
        evidence=hypothesis.evidence,
        scope_label=scope_label,
        intervention_direction=hypothesis.intervention_direction,
    )
    claims, claim_links, claim_units = _atomic_claims_and_links(
        brief,
        target_symbol=hypothesis.target.symbol,
        track=hypothesis.track,
        assignment=assignment,
        evidence=hypothesis.evidence,
        scope_label=scope_label,
        intervention_direction=hypothesis.intervention_direction,
    )
    target = hypothesis.target.model_copy(
        update={
            "metadata": {
                **hypothesis.target.metadata,
                "counter_evidence_search": _counter_search_summary(
                    assignment,
                    hypothesis.evidence,
                ),
                "claim_evidence_assignment": _claim_assignment_metadata(
                    assignment,
                    claim_units=claim_units,
                ),
            }
        },
        deep=True,
    )
    reconciled_evidence = _reconcile_evidence_claim_linkage(
        hypothesis.evidence,
        assignment,
        claim_links=claim_links,
    )
    return hypothesis.model_copy(
        update={
            "mechanism": _prospective_mechanism(
                brief,
                scope_label or hypothesis.target.symbol,
                hypothesis.intervention_direction,
            ),
            "target": target,
            "evidence": reconciled_evidence,
            "claims": claims,
        },
        deep=True,
    )


def _target_identity_status(
    seed: TargetSeed,
) -> str:
    allowed = {"resolved", "partial", "unresolved", "fixture"}
    if seed.identity_status in allowed:
        return seed.identity_status
    if seed.uniprot_id and seed.ensembl_id:
        return "resolved"
    if seed.ensembl_id or seed.uniprot_id:
        return "partial"
    return "fixture" if seed.discovery_source == "fixture_fallback" else "unresolved"


def _read_exclusions(context: object | None) -> list[ExcludedHypothesis]:
    if context is None:
        return []
    if isinstance(context, HypothesisPacket):
        return list(context.excluded)
    if isinstance(context, Mapping):
        raw_items = context.get("excluded", context.get("prior_exclusions", []))
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            return []
        result: list[ExcludedHypothesis] = []
        for item in raw_items:
            if isinstance(item, ExcludedHypothesis):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(ExcludedHypothesis.model_validate(dict(item)))
        return result
    return []


def _read_hypotheses(context: object | None) -> list[Hypothesis]:
    if context is None:
        return []
    if isinstance(context, HypothesisPacket):
        return list(context.hypotheses)
    if isinstance(context, Mapping):
        raw_items = context.get("hypotheses", context.get("prior_hypotheses", []))
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            return []
        result: list[Hypothesis] = []
        for item in raw_items:
            if isinstance(item, Hypothesis):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(Hypothesis.model_validate(dict(item)))
        return result
    return []


__all__ = [
    "Critic",
    "EvidenceRetriever",
    "IdentityResolver",
    "Intake",
    "MA1Agent",
    "MockEvidenceRetriever",
    "Proposer",
    "Router",
    "Selector",
    "TargetDiscovery",
    "TargetSeed",
]
