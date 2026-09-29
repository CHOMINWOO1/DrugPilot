"""Fail-closed candidate-to-evidence identity attestation.

Candidate enrichment often queries a target and a compound at the same time.
Target-scoped records (for example, BindingDB ligands or DrugCentral target
interactions) are useful context, but they are not evidence for every
candidate returned for that target.  This module provides the single identity
boundary used before an enrichment record is appended to ``CandidateCard``.

The comparison is deliberately conservative:

* persisted InChIKey and canonical SMILES are preferred;
* ChEMBL and DrugCentral stable compound identifiers are accepted when their
  namespace is explicit;
* an exact candidate name is only a last-resort identity key;
* any shared identifier conflict rejects the link, even if a lower-priority
  key happens to match;
* records without a candidate identity remain target context.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Literal

from drugpilot.domain.models import CandidateCard, EvidenceRef
from drugpilot.tools.chemistry import ChemistryValidator


IDENTITY_LINK_POLICY_VERSION = "candidate-evidence-exact-identity-v1"
EVIDENCE_SCORING_IDENTITY_POLICY_VERSION = (
    "source-external-id-title-excerpt-support-v1"
)
_CHEMBL_ID = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
_IDENTITY_PRECEDENCE = (
    "inchikey",
    "canonical_smiles",
    "chembl_id",
    "drugcentral_id",
    "exact_name",
)


@dataclass(frozen=True, slots=True)
class CandidateEvidenceLinkDecision:
    status: Literal["exact_candidate", "target_context", "identity_conflict"]
    matched_by: str | None
    matched_value: str | None
    shared_identifiers: tuple[str, ...]
    conflicts: tuple[str, ...]
    policy_version: str = IDENTITY_LINK_POLICY_VERSION

    @property
    def candidate_specific(self) -> bool:
        return self.status == "exact_candidate"

    def as_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "matched_by": self.matched_by,
            "matched_value": self.matched_value,
            "shared_identifiers": list(self.shared_identifiers),
            "conflicts": list(self.conflicts),
            "policy_version": self.policy_version,
            "conflict_rejected": self.status == "identity_conflict",
        }


_CHEMISTRY = ChemistryValidator(prefer_rdkit=True)


def attest_candidate_evidence(
    candidate: CandidateCard,
    evidence: EvidenceRef,
) -> CandidateEvidenceLinkDecision:
    """Return an exact candidate link or a non-candidate context decision."""

    candidate_ids = _candidate_identifiers(candidate)
    evidence_ids = _evidence_identifiers(evidence)
    shared: list[str] = []
    matches: list[tuple[str, str]] = []
    conflicts: list[str] = []

    for namespace in _IDENTITY_PRECEDENCE:
        left = candidate_ids[namespace]
        right = evidence_ids[namespace]
        if not left or not right:
            continue
        shared.append(namespace)
        overlap = left.intersection(right)
        if overlap:
            matches.append((namespace, sorted(overlap)[0]))
        else:
            conflicts.append(namespace)

    # A single contradictory namespace is sufficient to reject the record.
    # This prevents a reused target ligand from being rescued by a coincidental
    # name or by an unrelated database identifier.
    if conflicts:
        return CandidateEvidenceLinkDecision(
            status="identity_conflict",
            matched_by=matches[0][0] if matches else None,
            matched_value=matches[0][1] if matches else None,
            shared_identifiers=tuple(shared),
            conflicts=tuple(conflicts),
        )
    if matches:
        matched_by, matched_value = matches[0]
        return CandidateEvidenceLinkDecision(
            status="exact_candidate",
            matched_by=matched_by,
            matched_value=matched_value,
            shared_identifiers=tuple(shared),
            conflicts=(),
        )
    return CandidateEvidenceLinkDecision(
        status="target_context",
        matched_by=None,
        matched_value=None,
        shared_identifiers=(),
        conflicts=(),
    )


def candidate_linked_evidence(
    candidate: CandidateCard,
    evidence: EvidenceRef,
) -> EvidenceRef | None:
    """Copy an exact record with its attestation; never mutate source data."""

    decision = attest_candidate_evidence(candidate, evidence)
    if not decision.candidate_specific:
        return None
    return evidence.model_copy(
        update={
            "metadata": {
                **evidence.metadata,
                "candidate_identity_link": decision.as_metadata(),
            }
        },
        deep=True,
    )


def evidence_scoring_identity(evidence: EvidenceRef) -> tuple[str, ...]:
    """Return one stable scientific-record key for score cardinality.

    Source adapters may preserve several physical rows for the same scientific
    record so that their row-level provenance remains auditable.  Those rows
    must stay in ``CandidateCard.evidence``, but they must not multiply a
    candidate's score.  A changed excerpt or title remains a distinct record;
    volatile row numbers, retrieval timestamps and generated evidence IDs do
    not.
    """

    return (
        evidence.source.strip().casefold(),
        evidence.external_id.strip().casefold(),
        " ".join(evidence.title.split()).casefold(),
        str(evidence.excerpt_hash or "").strip().casefold(),
        evidence.supports,
    )


def evidence_scoring_cardinality(
    evidence: Iterable[EvidenceRef],
) -> dict[str, Any]:
    """Describe raw and unique evidence counts without deleting any rows."""

    rows = list(evidence)
    raw_identities = [evidence_scoring_identity(item) for item in rows]
    raw_hashes = [
        sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for identity in raw_identities
    ]
    unique_hashes = sorted(set(raw_hashes))
    unique_count = len(unique_hashes)
    raw_count = len(rows)
    return {
        "policy_version": EVIDENCE_SCORING_IDENTITY_POLICY_VERSION,
        "raw_record_count": raw_count,
        "unique_scientific_record_count": unique_count,
        "duplicate_physical_row_count": raw_count - unique_count,
        "raw_record_identity_sha256s": raw_hashes,
        "unique_scientific_record_identity_sha256s": unique_hashes,
        "raw_records_preserved": True,
        "duplicates_contribute_to_score": False,
    }


def _candidate_identifiers(candidate: CandidateCard) -> dict[str, set[str]]:
    identifiers = _empty_identifiers()
    _add(identifiers["exact_name"], candidate.name, casefold=True)
    _add(identifiers["inchikey"], candidate.inchikey, casefold=True)
    _add_smiles(identifiers["canonical_smiles"], candidate.smiles)
    chemistry = candidate.chemistry_execution
    if chemistry is not None:
        _add(identifiers["inchikey"], chemistry.inchikey, casefold=True)
        _add_smiles(identifiers["canonical_smiles"], chemistry.canonical_smiles)

    metadata = candidate.metadata
    _collect_common_identity(metadata, identifiers)
    for nested in _nested_source_metadata(metadata):
        _collect_common_identity(nested, identifiers)

    # Do not learn identity from the mutable enrichment collection itself.
    # That would make a previously mis-attached target record self-validating
    # on the next pass.  Only the candidate fields, chemistry attestation and
    # immutable discovery ``source_metadata``/``source_evidence`` envelopes are
    # authoritative inputs here.
    return identifiers


def _evidence_identifiers(evidence: EvidenceRef) -> dict[str, set[str]]:
    identifiers = _empty_identifiers()
    metadata = evidence.metadata
    _collect_common_identity(metadata, identifiers)
    source = evidence.source.casefold()
    if source == "drugcentral":
        _add(identifiers["drugcentral_id"], evidence.external_id, casefold=True)
    elif source == "chembl" and _CHEMBL_ID.fullmatch(evidence.external_id.strip()):
        _add(identifiers["chembl_id"], evidence.external_id, casefold=True)

    for key in ("drug_name", "candidate_name", "pref_name", "preferred_name"):
        _add(identifiers["exact_name"], metadata.get(key), casefold=True)
    return identifiers


def _empty_identifiers() -> dict[str, set[str]]:
    return {namespace: set() for namespace in _IDENTITY_PRECEDENCE}


def _collect_common_identity(
    metadata: Mapping[str, Any],
    identifiers: dict[str, set[str]],
) -> None:
    for key in (
        "standard_inchi_key",
        "inchikey",
        "inchi_key",
        "resolved_inchikey",
    ):
        _add(identifiers["inchikey"], metadata.get(key), casefold=True)
    for key in ("canonical_smiles", "smiles", "identity_parent_smiles"):
        _add_smiles(identifiers["canonical_smiles"], metadata.get(key))

    for key in ("chembl_id", "molecule_chembl_id"):
        _add_chembl(identifiers["chembl_id"], metadata.get(key))
    drug_id = metadata.get("drug_id")
    if drug_id is not None and _CHEMBL_ID.fullmatch(str(drug_id).strip()):
        _add_chembl(identifiers["chembl_id"], drug_id)

    for key in ("drugcentral_id", "struct_id"):
        _add(identifiers["drugcentral_id"], metadata.get(key), casefold=True)
    backend = str(
        metadata.get("backend")
        or metadata.get("retrieval_backend")
        or metadata.get("source")
        or ""
    ).casefold()
    if "drugcentral" in backend and drug_id is not None:
        _add(identifiers["drugcentral_id"], drug_id, casefold=True)

    for key in ("candidate_name", "drug_name", "pref_name", "preferred_name"):
        _add(identifiers["exact_name"], metadata.get(key), casefold=True)


def _nested_source_metadata(metadata: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("source_metadata", "source_evidence"):
        raw = metadata.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            yield item
            nested = item.get("metadata")
            if isinstance(nested, Mapping):
                yield nested


def _add_chembl(target: set[str], value: Any) -> None:
    if value is None:
        return
    normalized = str(value).strip()
    if _CHEMBL_ID.fullmatch(normalized):
        target.add(normalized.casefold())


def _add(target: set[str], value: Any, *, casefold: bool) -> None:
    if value is None:
        return
    normalized = str(value).strip()
    if not normalized:
        return
    target.add(normalized.casefold() if casefold else normalized)


def _add_smiles(target: set[str], value: Any) -> None:
    if value is None or not str(value).strip():
        return
    raw = str(value).strip()
    report = _CHEMISTRY.inspect(raw)
    normalized = report.canonical_smiles if report.valid else "".join(raw.split())
    if normalized:
        target.add(normalized)


__all__ = [
    "CandidateEvidenceLinkDecision",
    "EVIDENCE_SCORING_IDENTITY_POLICY_VERSION",
    "IDENTITY_LINK_POLICY_VERSION",
    "attest_candidate_evidence",
    "candidate_linked_evidence",
    "evidence_scoring_cardinality",
    "evidence_scoring_identity",
]
