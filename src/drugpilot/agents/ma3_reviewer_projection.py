from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from drugpilot.agents.ma3 import MA3AssuranceAgent
from drugpilot.domain.models import (
    CandidateCard,
    CandidatePacket,
    GateName,
    GateVerdict,
    HypothesisPacket,
    PlanSpec,
    ResearchBrief,
)
from drugpilot.tools.live_repurposing_gateway import (
    PAIRED_V4_REQUIRED_POLICY,
    PAPER_NATIVE_V5_REQUIRED_POLICY,
)


PAPER_NATIVE_V5_G1_REVIEW_PROJECTION_SCHEMA = (
    "paper_native_v5_g1_semantic_reviewer_projection_v1"
)
PAPER_NATIVE_V5_G1_REVIEW_CANONICALIZATION = (
    "json_sort_keys_compact_utf8_sha256_v1"
)
GENERIC_G1_REVIEW_PROJECTION_SCHEMA = (
    "lossless_g1_candidate_reviewer_projection_v1"
)
GENERIC_G1_REVIEW_PROJECTION_POLICY = (
    "lossless_candidate_packet_canonical_dag_v1"
)
_REFERENCE_KEY = "$drugpilot_canonical_ref_v1"
_LITERAL_OBJECT_KEY = "$drugpilot_literal_object_v1"
_SHARED_VALUE_MIN_BYTES = 512
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_INVALID_V5_POLICY_REVIEWER_NOTE = (
    "External semantic review was not invoked because the paper-native-v5 "
    "policy token was duplicated or combined with the paired-v4 policy."
)


class ReviewerProjectionIntegrityError(ValueError):
    """The compact reviewer artifact cannot reconstruct its source packet."""


class _ProjectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanonicalSharedBlock(_ProjectionModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    canonical_size_bytes: int = Field(ge=1)
    json_type: Literal["object", "array", "string"]
    value: Any


class CandidateReviewerIndex(_ProjectionModel):
    candidate_id: str
    name: str
    status: Literal["priority", "conditional", "excluded"]
    score: float
    rank: int | None
    target_rank: int | None
    portfolio_rank: int | None
    portfolio_score: float | None
    exclusion_reasons: list[str]
    safety_flags: list[str]
    evidence_count: int = Field(ge=0)
    activity_count: int = Field(ge=0)
    tool_count: int = Field(ge=0)
    tool_trace_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    distinct_source_count: int = Field(ge=0)
    source_counts: dict[str, int]


class ReviewerProjectionIndex(_ProjectionModel):
    candidate_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    target_execution_count: int = Field(ge=0)
    candidates: list[CandidateReviewerIndex]


class PaperNativeV5G1ReviewerProjection(_ProjectionModel):
    """Lossless CandidatePacket projection used only by the external reviewer.

    The deterministic G1 audit and the persisted CandidatePacket continue to
    use the original full object.  Repeated JSON subtrees are represented once
    in ``shared_canonical_blocks`` and referenced by their hash.  The expanded
    value is byte-canonically identical to ``CandidatePacket.model_dump``.
    """

    schema_version: Literal[
        "paper_native_v5_g1_semantic_reviewer_projection_v1"
    ] = PAPER_NATIVE_V5_G1_REVIEW_PROJECTION_SCHEMA
    policy: Literal[
        "dynamic_pubmed_primary_abstract_paper_native_v5_required"
    ] = PAPER_NATIVE_V5_REQUIRED_POLICY
    source_artifact_type: Literal["CandidatePacket"] = "CandidatePacket"
    source_packet_id: str
    source_packet_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonicalization: Literal[
        "json_sort_keys_compact_utf8_sha256_v1"
    ] = PAPER_NATIVE_V5_G1_REVIEW_CANONICALIZATION
    reference_semantics: str
    reviewer_index: ReviewerProjectionIndex
    reviewer_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    encoded_candidate_packet: dict[str, Any]
    shared_canonical_blocks: list[CanonicalSharedBlock]
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)


class GenericG1ReviewerProjection(_ProjectionModel):
    """Lossless CandidatePacket DAG for ordinary G1 semantic review."""

    schema_version: Literal[
        "lossless_g1_candidate_reviewer_projection_v1"
    ] = GENERIC_G1_REVIEW_PROJECTION_SCHEMA
    policy: Literal[
        "lossless_candidate_packet_canonical_dag_v1"
    ] = GENERIC_G1_REVIEW_PROJECTION_POLICY
    source_artifact_type: Literal["CandidatePacket"] = "CandidatePacket"
    source_packet_id: str
    source_packet_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonicalization: Literal[
        "json_sort_keys_compact_utf8_sha256_v1"
    ] = PAPER_NATIVE_V5_G1_REVIEW_CANONICALIZATION
    reference_semantics: str
    reviewer_index: ReviewerProjectionIndex
    reviewer_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    encoded_candidate_packet: dict[str, Any]
    shared_canonical_blocks: list[CanonicalSharedBlock]
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass
class _CanonicalNode:
    canonical: bytes
    value: Any
    count: int = 0


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReviewerProjectionIntegrityError(
            "review projection input is not canonical JSON"
        ) from exc
    return encoded.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _json_type(value: Any) -> Literal["object", "array", "string"]:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    raise ReviewerProjectionIntegrityError("unsupported shared JSON value type")


def _is_shareable(value: Any, canonical_size: int) -> bool:
    return (
        isinstance(value, (dict, list, str))
        and canonical_size >= _SHARED_VALUE_MIN_BYTES
    )


def _collect_canonical_nodes(
    value: Any,
    nodes: dict[str, _CanonicalNode],
) -> None:
    if isinstance(value, (dict, list, str)):
        canonical = _canonical_bytes(value)
        digest = sha256(canonical).hexdigest()
        existing = nodes.get(digest)
        if existing is None:
            nodes[digest] = _CanonicalNode(
                canonical=canonical,
                value=value,
                count=1,
            )
        else:
            if existing.canonical != canonical:
                raise ReviewerProjectionIntegrityError(
                    "canonical SHA-256 collision while building reviewer projection"
                )
            existing.count += 1
    if isinstance(value, dict):
        for child in value.values():
            _collect_canonical_nodes(child, nodes)
    elif isinstance(value, list):
        for child in value:
            _collect_canonical_nodes(child, nodes)


def _shared_digests(nodes: Mapping[str, _CanonicalNode]) -> set[str]:
    reference_size = len(_canonical_bytes({_REFERENCE_KEY: "0" * 64}))
    selected: set[str] = set()
    for digest, node in nodes.items():
        size = len(node.canonical)
        if node.count < 2 or not _is_shareable(node.value, size):
            continue
        # The estimate deliberately includes a conservative block-manifest
        # overhead.  Sharing is therefore an input-size reduction, not merely
        # a structural rewrite with more bytes.
        saved = size * (node.count - 1)
        overhead = reference_size * node.count + 192
        if saved > overhead:
            selected.add(digest)
    return selected


def _is_reserved_source_object(value: Mapping[str, Any]) -> bool:
    return _REFERENCE_KEY in value or _LITERAL_OBJECT_KEY in value


def _encode_value(
    value: Any,
    *,
    shared: set[str],
    root_digest: str | None = None,
) -> Any:
    if isinstance(value, (dict, list, str)):
        digest = _canonical_sha256(value)
        if digest in shared and digest != root_digest:
            return {_REFERENCE_KEY: digest}
    if isinstance(value, dict):
        if _is_reserved_source_object(value):
            return {
                _LITERAL_OBJECT_KEY: [
                    [
                        key,
                        _encode_value(child, shared=shared),
                    ]
                    for key, child in value.items()
                ]
            }
        return {
            key: _encode_value(child, shared=shared)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_encode_value(child, shared=shared) for child in value]
    return value


def _source_counts(candidate: CandidateCard) -> dict[str, int]:
    counts = Counter(str(item.source) for item in candidate.evidence)
    return dict(sorted(counts.items()))


def _reviewer_index(packet: CandidatePacket) -> ReviewerProjectionIndex:
    candidates: list[CandidateReviewerIndex] = []
    for candidate in packet.candidates:
        source_counts = _source_counts(candidate)
        candidates.append(
            CandidateReviewerIndex(
                candidate_id=candidate.candidate_id,
                name=candidate.name,
                status=candidate.status,
                score=candidate.score,
                rank=candidate.rank,
                target_rank=candidate.target_rank,
                portfolio_rank=candidate.portfolio_rank,
                portfolio_score=candidate.portfolio_score,
                exclusion_reasons=list(candidate.exclusion_reasons),
                safety_flags=list(candidate.safety_flags),
                evidence_count=len(candidate.evidence),
                activity_count=len(candidate.activities),
                tool_count=len(candidate.tool_trace_ids),
                tool_trace_count=len(candidate.tool_trace_ids),
                source_count=sum(source_counts.values()),
                distinct_source_count=len(source_counts),
                source_counts=source_counts,
            )
        )
    return ReviewerProjectionIndex(
        candidate_count=len(packet.candidates),
        tool_call_count=len(packet.tool_calls),
        target_execution_count=len(packet.target_executions),
        candidates=candidates,
    )


def _projection_body_sha256(raw: Mapping[str, Any]) -> str:
    body = dict(raw)
    body.pop("projection_sha256", None)
    return _canonical_sha256(body)


def build_paper_native_v5_g1_reviewer_projection(
    packet: CandidatePacket,
) -> PaperNativeV5G1ReviewerProjection:
    """Build a deterministic, lossless and compact reviewer-only artifact."""

    source = packet.model_dump(mode="json")
    nodes: dict[str, _CanonicalNode] = {}
    _collect_canonical_nodes(source, nodes)
    shared = _shared_digests(nodes)
    blocks = [
        CanonicalSharedBlock(
            sha256=digest,
            canonical_size_bytes=len(nodes[digest].canonical),
            json_type=_json_type(nodes[digest].value),
            value=_encode_value(
                nodes[digest].value,
                shared=shared,
                root_digest=digest,
            ),
        )
        for digest in sorted(shared)
    ]
    index = _reviewer_index(packet)
    raw: dict[str, Any] = {
        "schema_version": PAPER_NATIVE_V5_G1_REVIEW_PROJECTION_SCHEMA,
        "policy": PAPER_NATIVE_V5_REQUIRED_POLICY,
        "source_artifact_type": "CandidatePacket",
        "source_packet_id": packet.packet_id,
        "source_packet_sha256": _canonical_sha256(source),
        "canonicalization": PAPER_NATIVE_V5_G1_REVIEW_CANONICALIZATION,
        "reference_semantics": (
            f"A JSON object with the sole key {_REFERENCE_KEY!r} resolves to "
            "the shared block carrying that SHA-256. Block hashes and the "
            "source packet hash cover fully expanded canonical JSON. No "
            "CandidatePacket field or value is omitted."
        ),
        "reviewer_index": index.model_dump(mode="json"),
        "reviewer_index_sha256": _canonical_sha256(
            index.model_dump(mode="json")
        ),
        "encoded_candidate_packet": _encode_value(source, shared=shared),
        "shared_canonical_blocks": [
            block.model_dump(mode="json") for block in blocks
        ],
    }
    raw["projection_sha256"] = _projection_body_sha256(raw)
    projection = PaperNativeV5G1ReviewerProjection.model_validate(raw)
    # Construction is fail-closed: never send a compact artifact that cannot
    # reproduce the exact immutable CandidatePacket.
    expanded = expand_paper_native_v5_g1_reviewer_projection(projection)
    if expanded != source:
        raise ReviewerProjectionIntegrityError(
            "review projection reconstruction differs from CandidatePacket"
        )
    return projection


def build_generic_g1_reviewer_projection(
    packet: CandidatePacket,
) -> GenericG1ReviewerProjection:
    """Reuse the validated lossless DAG for non-v5 G1 reviewer input."""

    validated = build_paper_native_v5_g1_reviewer_projection(packet)
    raw = validated.model_dump(mode="json")
    raw["schema_version"] = GENERIC_G1_REVIEW_PROJECTION_SCHEMA
    raw["policy"] = GENERIC_G1_REVIEW_PROJECTION_POLICY
    raw["projection_sha256"] = _projection_body_sha256(raw)
    return GenericG1ReviewerProjection.model_validate(raw)


def expand_paper_native_v5_g1_reviewer_projection(
    projection: PaperNativeV5G1ReviewerProjection | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and losslessly reconstruct a compact reviewer projection."""

    if isinstance(projection, PaperNativeV5G1ReviewerProjection):
        raw = projection.model_dump(mode="json")
    else:
        raw = deepcopy(dict(projection))
        try:
            PaperNativeV5G1ReviewerProjection.model_validate(raw)
        except Exception as exc:
            raise ReviewerProjectionIntegrityError(
                "review projection schema is invalid"
            ) from exc

    if raw.get("projection_sha256") != _projection_body_sha256(raw):
        raise ReviewerProjectionIntegrityError(
            "review projection self hash is invalid"
        )
    reviewer_index = raw.get("reviewer_index")
    if raw.get("reviewer_index_sha256") != _canonical_sha256(reviewer_index):
        raise ReviewerProjectionIntegrityError("reviewer index hash is invalid")

    entries = raw.get("shared_canonical_blocks")
    if not isinstance(entries, list):
        raise ReviewerProjectionIntegrityError("shared block inventory is invalid")
    blocks: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ReviewerProjectionIntegrityError("shared block entry is invalid")
        digest = entry["sha256"]
        if digest in blocks:
            raise ReviewerProjectionIntegrityError(
                "duplicate shared block hash collision"
            )
        blocks[digest] = entry

    expanded_blocks: dict[str, Any] = {}
    visiting: set[str] = set()
    used: set[str] = set()

    def resolve_block(digest: str) -> Any:
        if digest in expanded_blocks:
            return deepcopy(expanded_blocks[digest])
        entry = blocks.get(digest)
        if entry is None:
            raise ReviewerProjectionIntegrityError(
                f"dangling shared block reference: {digest}"
            )
        if digest in visiting:
            raise ReviewerProjectionIntegrityError(
                f"cyclic shared block reference: {digest}"
            )
        visiting.add(digest)
        expanded = decode(entry.get("value"))
        visiting.remove(digest)
        canonical = _canonical_bytes(expanded)
        if sha256(canonical).hexdigest() != digest:
            raise ReviewerProjectionIntegrityError(
                f"shared block hash mismatch: {digest}"
            )
        if entry.get("canonical_size_bytes") != len(canonical):
            raise ReviewerProjectionIntegrityError(
                f"shared block size mismatch: {digest}"
            )
        if entry.get("json_type") != _json_type(expanded):
            raise ReviewerProjectionIntegrityError(
                f"shared block JSON type mismatch: {digest}"
            )
        expanded_blocks[digest] = expanded
        return deepcopy(expanded)

    def decode(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {_REFERENCE_KEY}:
                digest = value[_REFERENCE_KEY]
                if not isinstance(digest, str):
                    raise ReviewerProjectionIntegrityError(
                        "shared block reference hash is not a string"
                    )
                used.add(digest)
                return resolve_block(digest)
            if set(value) == {_LITERAL_OBJECT_KEY}:
                pairs = value[_LITERAL_OBJECT_KEY]
                if not isinstance(pairs, list):
                    raise ReviewerProjectionIntegrityError(
                        "escaped literal object is invalid"
                    )
                restored: dict[str, Any] = {}
                for pair in pairs:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or not isinstance(pair[0], str)
                        or pair[0] in restored
                    ):
                        raise ReviewerProjectionIntegrityError(
                            "escaped literal object contains an invalid key pair"
                        )
                    restored[pair[0]] = decode(pair[1])
                return restored
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value

    encoded_packet = raw.get("encoded_candidate_packet")
    if not isinstance(encoded_packet, dict):
        raise ReviewerProjectionIntegrityError(
            "encoded CandidatePacket root is invalid"
        )
    expanded_packet = decode(encoded_packet)
    if set(blocks) != used:
        unused = sorted(set(blocks) - used)
        raise ReviewerProjectionIntegrityError(
            f"unreferenced shared block inventory entries: {unused}"
        )
    if raw.get("source_packet_sha256") != _canonical_sha256(expanded_packet):
        raise ReviewerProjectionIntegrityError(
            "expanded CandidatePacket hash differs from source_packet_sha256"
        )
    if raw.get("source_packet_id") != expanded_packet.get("packet_id"):
        raise ReviewerProjectionIntegrityError(
            "expanded CandidatePacket ID differs from source_packet_id"
        )
    try:
        parsed_packet = CandidatePacket.model_validate(expanded_packet)
    except Exception as exc:
        raise ReviewerProjectionIntegrityError(
            "expanded CandidatePacket schema is invalid"
        ) from exc
    parsed_source = parsed_packet.model_dump(mode="json")
    if _canonical_bytes(parsed_source) != _canonical_bytes(expanded_packet):
        raise ReviewerProjectionIntegrityError(
            "expanded CandidatePacket is not its canonical typed projection"
        )
    recomputed_index = _reviewer_index(parsed_packet).model_dump(mode="json")
    if _canonical_bytes(recomputed_index) != _canonical_bytes(reviewer_index):
        raise ReviewerProjectionIntegrityError(
            "reviewer index differs from expanded CandidatePacket"
        )
    return expanded_packet


class ReviewerProjectionMA3AssuranceAgent(MA3AssuranceAgent):
    """MA3 that losslessly compacts every G1 external-reviewer input."""

    async def audit_g0(self, brief, packet, *, on_usage=None, before_review=None):
        from drugpilot.agents.hypothesis_projection import compact_hypothesis_input
        # The full packet still goes through every deterministic rule.
        findings = self._audit_g0(brief, packet)
        validations = self._validate_g0(packet)
        projected = compact_hypothesis_input(packet) if self.reviewer_provider else packet
        semantic, invoked, note = await self._semantic_review_r18(
            gate=GateName.G0,
            artifacts={"research_brief":brief, "hypothesis_packet":projected},
            findings=findings, validations=validations,
            on_usage=on_usage, before_review=before_review)
        return self._decide(GateName.G0, findings + semantic, validations,
            reviewer_invoked=invoked, reviewer_note=note)

    async def audit_g1(
        self,
        brief: ResearchBrief,
        hypotheses: HypothesisPacket,
        plan: PlanSpec,
        candidates: CandidatePacket,
        *,
        on_usage=None,
        before_review=None,
        supplemental_evidence: BaseModel | None = None,
    ) -> GateVerdict:
        v5_policy_count = brief.human_priorities.count(
            PAPER_NATIVE_V5_REQUIRED_POLICY
        )
        paired_v4_present = PAIRED_V4_REQUIRED_POLICY in brief.human_priorities
        if v5_policy_count > 0 and (
            v5_policy_count != 1 or paired_v4_present
        ):
            # Invalid exact-v5 routing is already a deterministic MA3 finding.
            # Do not spend external-review budget before that structural
            # rejection, and do not build a reviewer-only projection for an
            # invalid policy request.
            findings = self._audit_g1(brief, hypotheses, plan, candidates)
            validations = self._validate_g1(candidates)
            return self._decide(
                GateName.G1,
                findings,
                validations,
                reviewer_invoked=False,
                reviewer_note=_INVALID_V5_POLICY_REVIEWER_NOTE,
            )

        if self.reviewer_provider is None:
            return await super().audit_g1(
                brief,
                hypotheses,
                plan,
                candidates,
                on_usage=on_usage,
                before_review=before_review,
                **({"supplemental_evidence": supplemental_evidence} if supplemental_evidence is not None else {}),
            )

        # Full immutable artifacts remain authoritative for deterministic G1.
        # Only the optional semantic reviewer sees the lossless DAG projection.
        findings = self._audit_g1(brief, hypotheses, plan, candidates)
        validations = self._validate_g1(candidates)
        reviewer_candidate_artifact = (
            build_paper_native_v5_g1_reviewer_projection(candidates)
            if v5_policy_count == 1
            else build_generic_g1_reviewer_projection(candidates)
        )
        from drugpilot.agents.hypothesis_projection import compact_hypothesis_input
        reviewer_hypothesis_artifact = compact_hypothesis_input(hypotheses)
        semantic, invoked, reviewer_note = await self._semantic_review_r18(
            gate=GateName.G1,
            artifacts={
                "research_brief": brief,
                "hypothesis_packet": reviewer_hypothesis_artifact,
                "plan_spec": plan,
                "candidate_packet": reviewer_candidate_artifact,
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
