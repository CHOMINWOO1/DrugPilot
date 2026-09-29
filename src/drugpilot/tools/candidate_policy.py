"""Deterministic candidate exclusion helpers shared by MA2 and MA3.

The screen intentionally avoids fuzzy name matching.  Drug names, stable
identifiers and declared classes are matched independently and exactly after
conservative normalization.  This makes every exclusion reproducible and
prevents a short blacklist token from removing an unrelated compound.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


_NAME_KEYS = frozenset(
    {
        "aliases",
        "brand_names",
        "drug_name",
        "generic_name",
        "names",
        "synonyms",
    }
)
_ID_KEYS = frozenset(
    {
        "chembl_id",
        "compound_id",
        "drug_id",
        "drug_ids",
        "drugbank_id",
        "identifiers",
        "pubchem_cid",
    }
)
_CLASS_KEYS = frozenset(
    {
        "atc_class",
        "classes",
        "drug_class",
        "drug_classes",
        "pharmacologic_class",
        "therapeutic_class",
    }
)
_STATUS_KEYS = frozenset(
    {
        "approval_status",
        "development_status",
        "development_stage",
        "marketing_status",
        "regulatory_status",
        "status",
    }
)
_PROHIBITED_STATUSES = frozenset(
    {
        "banned",
        "prohibited",
        "recalled",
        "suspended",
        "withdrawn",
        "withdrawn from market",
    }
)

_REGISTRY_FIELD_NAMES = {
    "chembl_id": "chembl",
    "molecule_chembl_id": "chembl",
    "drugbank_id": "drugbank",
    "drugcentral_id": "drugcentral",
    "pubchem_cid": "pubchem",
}
_REGISTRY_NAMESPACE_ALIASES = {
    "chembl": "chembl",
    "chembl api": "chembl",
    "drugbank": "drugbank",
    "drug central": "drugcentral",
    "drugcentral": "drugcentral",
    "pubchem": "pubchem",
    "pubchem cid": "pubchem",
}
_CANDIDATE_BACKEND_NAMESPACES = {
    "chemblliveapi": "chembl",
    "drugcentrallocaltsv": "drugcentral",
}


def _name_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _id_key(value: object) -> str:
    # Registry identifiers are often rendered with or without hyphens.
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _registry_namespace_key(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).strip()).casefold()
    return _REGISTRY_NAMESPACE_ALIASES.get(text, text)


def _registry_identifier_key(value: object) -> str:
    # Reserved controls use exact stable identifiers.  Unlike the legacy
    # blacklist, punctuation is material and is never stripped.
    return str(value).strip().casefold()


def _candidate_registry_identifiers(
    metadata: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    """Extract candidate IDs only from immutable, identity-bearing fields.

    This is deliberately not a recursive metadata search.  Activity, assay,
    document and target records frequently use the same registry namespace as
    a compound (notably ChEMBL), so a generic ``external_id`` or ``drug_id``
    found in an evidence/enrichment envelope is not candidate identity.
    """

    found: set[tuple[str, str]] = set()

    def add(namespace: object, identifier: object) -> None:
        namespace_key = _registry_namespace_key(namespace)
        identifier_key = _registry_identifier_key(identifier)
        if namespace_key and identifier_key:
            found.add((namespace_key, identifier_key))

    def lowered(value: Mapping[str, Any]) -> dict[str, Any]:
        return {str(key).casefold(): nested for key, nested in value.items()}

    def backend_namespace(value: object) -> str | None:
        key = re.sub(r"[^a-z0-9]", "", str(value).casefold())
        return _CANDIDATE_BACKEND_NAMESPACES.get(key)

    def candidate_level(
        value: Mapping[str, Any],
        *,
        backend_hint: object | None = None,
    ) -> None:
        payload = lowered(value)
        for field, namespace in _REGISTRY_FIELD_NAMES.items():
            if field in payload:
                for identifier in _iter_values(payload[field]):
                    add(namespace, identifier)

        # A mapping is typed (namespace -> ID); a flat legacy list is not and
        # must not be prefix-inferred because it may contain activity IDs.
        identifiers = payload.get("identifiers")
        if isinstance(identifiers, Mapping):
            for namespace, nested in identifiers.items():
                for identifier in _iter_values(nested):
                    add(namespace, identifier)

        explicit_registry = payload.get("registry")
        namespace = (
            _registry_namespace_key(explicit_registry)
            if explicit_registry is not None
            else backend_namespace(
                backend_hint or payload.get("retrieval_backend") or ""
            )
        )
        if namespace not in set(_REGISTRY_NAMESPACE_ALIASES.values()):
            return
        for field in ("drug_id", "compound_id"):
            if field in payload:
                for identifier in _iter_values(payload[field]):
                    add(namespace, identifier)

    source = metadata or {}
    candidate_level(source)

    # Merged live candidates retain the immutable candidate-level metadata of
    # each contributing backend here.  Process only that immediate layer.
    for container_key in ("source_metadata", "candidate_source_metadata"):
        records = source.get(container_key)
        if not isinstance(records, (list, tuple)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            nested = record.get("metadata")
            if isinstance(nested, Mapping):
                candidate_level(nested, backend_hint=record.get("backend"))

    # MA2's source_evidence outer ``drug_id`` is a legacy copy of
    # EvidenceRef.external_id and may be an activity/document ID.  Ignore it.
    # Only explicit molecule/drug fields inside the immutable evidence
    # metadata are eligible.
    source_evidence = source.get("source_evidence")
    if isinstance(source_evidence, (list, tuple)):
        for record in source_evidence:
            if not isinstance(record, Mapping):
                continue
            evidence_metadata = record.get("metadata")
            if not isinstance(evidence_metadata, Mapping):
                continue
            payload = lowered(evidence_metadata)
            source_namespace = _registry_namespace_key(record.get("source") or "")
            for field in (
                "molecule_chembl_id",
                "drugbank_id",
                "drugcentral_id",
                "pubchem_cid",
            ):
                if field in payload:
                    for identifier in _iter_values(payload[field]):
                        add(_REGISTRY_FIELD_NAMES[field], identifier)
            if source_namespace == "drugcentral" and "drug_id" in payload:
                for identifier in _iter_values(payload["drug_id"]):
                    add("drugcentral", identifier)

    return found


def candidate_identity(
    *,
    candidate_name: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a stable, JSON-ready identity for bounded-pool audit records."""

    identifiers = sorted(_candidate_registry_identifiers(metadata))
    return {
        "name": candidate_name,
        "registry_ids": [
            {"namespace": namespace, "identifier": identifier}
            for namespace, identifier in identifiers
        ],
    }


def _iter_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_values(nested)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            yield from _iter_values(nested)
        return
    text = str(value).strip()
    if text:
        yield text


def _metadata_values(metadata: Mapping[str, Any], keys: frozenset[str]) -> set[str]:
    values: set[str] = set()
    for key, value in metadata.items():
        if str(key).casefold() in keys:
            values.update(_iter_values(value))
        if isinstance(value, Mapping):
            values.update(_metadata_values(value, keys))
        elif isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                if isinstance(nested, Mapping):
                    values.update(_metadata_values(nested, keys))
    return values


@dataclass(frozen=True, slots=True)
class ForbiddenDrugScreen:
    matched_names: tuple[str, ...] = ()
    matched_ids: tuple[str, ...] = ()
    matched_classes: tuple[str, ...] = ()
    prohibited_statuses: tuple[str, ...] = ()

    @property
    def excluded(self) -> bool:
        return bool(
            self.matched_names
            or self.matched_ids
            or self.matched_classes
            or self.prohibited_statuses
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        reasons = [f"forbidden_drug_name:{item}" for item in self.matched_names]
        reasons.extend(f"forbidden_drug_id:{item}" for item in self.matched_ids)
        reasons.extend(f"forbidden_drug_class:{item}" for item in self.matched_classes)
        reasons.extend(f"prohibited_regulatory_status:{item}" for item in self.prohibited_statuses)
        return tuple(reasons)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "screened": True,
            "matched": self.excluded,
            "matched_names": list(self.matched_names),
            "matched_ids": list(self.matched_ids),
            "matched_classes": list(self.matched_classes),
            "prohibited_statuses": list(self.prohibited_statuses),
            "matching_policy": "exact_case_insensitive_v1",
        }


def screen_forbidden_drug(
    *,
    candidate_name: str,
    metadata: Mapping[str, Any] | None,
    forbidden_names: Iterable[str] = (),
    forbidden_ids: Iterable[str] = (),
    forbidden_classes: Iterable[str] = (),
) -> ForbiddenDrugScreen:
    """Return exact matches while retaining the human-entered policy values."""

    source = metadata or {}
    candidate_names = {candidate_name, *_metadata_values(source, _NAME_KEYS)}
    candidate_ids = _metadata_values(source, _ID_KEYS)
    candidate_classes = _metadata_values(source, _CLASS_KEYS)
    candidate_statuses = _metadata_values(source, _STATUS_KEYS)

    names_by_key = {
        _name_key(item): str(item).strip()
        for item in forbidden_names
        if str(item).strip()
    }
    ids_by_key = {
        _id_key(item): str(item).strip()
        for item in forbidden_ids
        if _id_key(item)
    }
    classes_by_key = {
        _name_key(item): str(item).strip()
        for item in forbidden_classes
        if str(item).strip()
    }

    matched_names = {
        names_by_key[key]
        for value in candidate_names
        if (key := _name_key(value)) in names_by_key
    }
    matched_ids = {
        ids_by_key[key]
        for value in candidate_ids
        if (key := _id_key(value)) in ids_by_key
    }
    matched_classes = {
        classes_by_key[key]
        for value in candidate_classes
        if (key := _name_key(value)) in classes_by_key
    }
    prohibited_statuses = {
        value
        for value in candidate_statuses
        if _name_key(value) in _PROHIBITED_STATUSES
    }
    return ForbiddenDrugScreen(
        matched_names=tuple(sorted(matched_names, key=str.casefold)),
        matched_ids=tuple(sorted(matched_ids, key=str.casefold)),
        matched_classes=tuple(sorted(matched_classes, key=str.casefold)),
        prohibited_statuses=tuple(sorted(prohibited_statuses, key=str.casefold)),
    )


@dataclass(frozen=True, slots=True)
class ReservedControlScreen:
    """Exact assay-control match kept separate from drug prohibitions."""

    matched_control_names: tuple[str, ...] = ()
    matched_registry_ids: tuple[str, ...] = ()
    name_fallback_control_names: tuple[str, ...] = ()
    identity_conflict_control_names: tuple[str, ...] = ()
    candidate_registry_ids: tuple[str, ...] = ()

    @property
    def matched(self) -> bool:
        return bool(self.matched_control_names)

    @property
    def identity_conflict(self) -> bool:
        return bool(self.identity_conflict_control_names)

    @property
    def excluded(self) -> bool:
        # A same-name stable-ID conflict is not relabelled as a control, but
        # it is fail-closed so the reserved compound cannot slip into ranking.
        return self.matched or self.identity_conflict

    @property
    def reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.matched:
            reasons.append("RESERVED_ASSAY_CONTROL")
        if self.identity_conflict:
            reasons.append("RESERVED_CONTROL_IDENTITY_CONFLICT")
        return tuple(reasons)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "screened": True,
            "matched": self.matched,
            "excluded": self.excluded,
            "control_only": self.matched,
            "identity_conflict": self.identity_conflict,
            "matched_control_names": list(self.matched_control_names),
            "matched_registry_ids": list(self.matched_registry_ids),
            "name_fallback_control_names": list(
                self.name_fallback_control_names
            ),
            "identity_conflict_control_names": list(
                self.identity_conflict_control_names
            ),
            "candidate_registry_ids": list(self.candidate_registry_ids),
            "matching_policy": (
                "exact_namespaced_registry_id_then_name_only_without_candidate_"
                "stable_id_v1"
            ),
        }


def _reserved_control_payload(control: Any) -> tuple[str, set[tuple[str, str]]]:
    if isinstance(control, Mapping):
        name = str(control.get("name") or "").strip()
        raw_ids = control.get("registry_ids") or ()
    else:
        name = str(getattr(control, "name", "") or "").strip()
        raw_ids = getattr(control, "registry_ids", ())
    identifiers: set[tuple[str, str]] = set()
    for item in raw_ids:
        if isinstance(item, Mapping):
            namespace = item.get("namespace")
            identifier = item.get("identifier")
        else:
            namespace = getattr(item, "namespace", None)
            identifier = getattr(item, "identifier", None)
        if namespace is None or identifier is None:
            continue
        namespace_key = _registry_namespace_key(namespace)
        identifier_key = _registry_identifier_key(identifier)
        if namespace_key and identifier_key:
            identifiers.add((namespace_key, identifier_key))
    return name, identifiers


def screen_reserved_control(
    *,
    candidate_name: str,
    metadata: Mapping[str, Any] | None,
    reserved_controls: Iterable[Any] = (),
) -> ReservedControlScreen:
    """Screen one candidate against structured, exact control identities.

    An exact namespaced registry-ID match is authoritative.  Exact name is a
    fallback only when the candidate exposes no stable registry ID.  If the
    name matches while stable IDs disagree, the result is an explicit
    fail-closed identity conflict rather than a name-based control match.
    """

    candidate_ids = _candidate_registry_identifiers(metadata)
    candidate_name_key = _name_key(candidate_name)
    matched_names: set[str] = set()
    matched_ids: set[str] = set()
    fallback_names: set[str] = set()
    conflicts: set[str] = set()

    for control in reserved_controls:
        control_name, control_ids = _reserved_control_payload(control)
        if not control_name:
            continue
        overlap = candidate_ids.intersection(control_ids)
        if overlap:
            matched_names.add(control_name)
            matched_ids.update(
                f"{namespace}:{identifier}"
                for namespace, identifier in overlap
            )
            continue
        if _name_key(control_name) != candidate_name_key:
            continue
        if candidate_ids:
            conflicts.add(control_name)
            continue
        matched_names.add(control_name)
        fallback_names.add(control_name)

    return ReservedControlScreen(
        matched_control_names=tuple(sorted(matched_names, key=str.casefold)),
        matched_registry_ids=tuple(sorted(matched_ids, key=str.casefold)),
        name_fallback_control_names=tuple(
            sorted(fallback_names, key=str.casefold)
        ),
        identity_conflict_control_names=tuple(
            sorted(conflicts, key=str.casefold)
        ),
        candidate_registry_ids=tuple(
            sorted(
                f"{namespace}:{identifier}"
                for namespace, identifier in candidate_ids
            )
        ),
    )
