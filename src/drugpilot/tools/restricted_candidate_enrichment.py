"""Route configured restricted sources for a ranked MA2 candidate.

This module is deliberately separate from the keyless chemistry retriever.
DrugCentral, DrugBank, and Google Patents have different licences, credentials,
costs, and scientific meanings.  Every source is considered for every selected
candidate, but only route-appropriate operations are admitted:

* repurposing: target-drug, identity/regulatory, and contraindication context;
* de novo: existing-drug identity/regulatory and patent-keyword prior-art context.

Missing configuration is returned as a typed source summary and never treated
as an empty/negative result.  Patent matches are context only and are not a
legal novelty opinion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any
import unicodedata

from pydantic import Field

from drugpilot.domain.models import (
    CandidateCard,
    CandidateRoute,
    EvidenceRef,
    Hypothesis,
    StrictModel,
)
from drugpilot.tools.restricted_sources import (
    DrugBankLicensedAdapter,
    DrugCentralRestrictedAdapter,
    GooglePatentsBigQueryAdapter,
    RestrictedSourceQuery,
    RestrictedSourceSummary,
)


class RestrictedCandidateEnrichment(StrictModel):
    """Evidence plus an exact per-operation audit summary."""

    evidence: list[EvidenceRef] = Field(default_factory=list)
    target_context: list[EvidenceRef] = Field(default_factory=list)
    sources: list[RestrictedSourceSummary] = Field(default_factory=list)
    routing: list[dict[str, Any]] = Field(default_factory=list)


class RoutedRestrictedCandidateEnricher:
    """Deterministic MA2 router for licensed/local/metered sources."""

    def __init__(
        self,
        *,
        drugcentral: DrugCentralRestrictedAdapter | None = None,
        drugbank: DrugBankLicensedAdapter | None = None,
        google_patents: GooglePatentsBigQueryAdapter | None = None,
    ) -> None:
        self.drugcentral = drugcentral or DrugCentralRestrictedAdapter()
        self.drugbank = drugbank or DrugBankLicensedAdapter()
        self.google_patents = google_patents or GooglePatentsBigQueryAdapter()

    def enrich_candidate(
        self,
        candidate: CandidateCard,
        hypothesis: Hypothesis,
        *,
        allowed_sources: Sequence[str] | None = None,
    ) -> RestrictedCandidateEnrichment:
        """Run only operations applicable to the selected scientific route."""

        # Keep the existing full candidate-evidence/SMILES identity policy for
        # real restricted enrichment, but import it only when enrichment is
        # actually executed.  Read-only registry-ID inspection must not import
        # or construct an RDKit-backed ChemistryValidator.
        from drugpilot.tools.candidate_evidence_identity import (
            IDENTITY_LINK_POLICY_VERSION,
            attest_candidate_evidence,
            candidate_linked_evidence,
        )

        allowed = None if allowed_sources is None else frozenset(allowed_sources)

        registry_ids = _candidate_registry_ids(candidate)
        registry_id_by_source = {
            source: values[0] if len(values) == 1 else None
            for source, values in registry_ids.items()
        }
        registry_identity_conflicts = {
            source
            for source in ("DrugCentral", "DrugBank")
            if len(registry_ids[source]) > 1
        }
        candidate_name = _normalized_exact_name(candidate.name)
        inchikey = candidate.inchikey
        if inchikey and inchikey.upper().startswith("MOCK"):
            inchikey = None
        common = {
            "candidate_name": candidate_name,
            "inchikey": inchikey,
            "target_uniprot": candidate.target.uniprot_id,
            "target_name": candidate.target.symbol,
            "condition": hypothesis.disease,
            "limit": 5,
        }

        planned: list[tuple[object, RestrictedSourceQuery, str]] = []
        if candidate.route is CandidateRoute.REPURPOSING:
            repurposing = RestrictedSourceQuery(
                operation="repurposing",
                target_uniprot=common["target_uniprot"],
                target_name=common["target_name"],
                limit=5,
            )
            drugcentral_identity = RestrictedSourceQuery(
                operation="identity_regulatory",
                candidate_name=candidate_name,
                drug_id=registry_id_by_source["DrugCentral"],
                inchikey=inchikey,
                limit=5,
            )
            drugbank_identity = RestrictedSourceQuery(
                operation="identity_regulatory",
                candidate_name=candidate_name,
                drug_id=registry_id_by_source["DrugBank"],
                inchikey=inchikey,
                limit=5,
            )
            drugcentral_contraindication = RestrictedSourceQuery(
                operation="contraindication",
                candidate_name=candidate_name,
                drug_id=registry_id_by_source["DrugCentral"],
                condition=hypothesis.disease,
                limit=5,
            )
            drugbank_contraindication = RestrictedSourceQuery(
                operation="contraindication",
                candidate_name=candidate_name,
                drug_id=registry_id_by_source["DrugBank"],
                condition=hypothesis.disease,
                limit=5,
            )
            planned.extend(
                [
                    (self.drugcentral, repurposing, "target_drug_context"),
                    (self.drugbank, repurposing, "target_drug_context"),
                    (
                        self.drugcentral,
                        drugcentral_identity,
                        "identity_and_regulatory_context",
                    ),
                    (
                        self.drugbank,
                        drugbank_identity,
                        "identity_and_regulatory_context",
                    ),
                    (
                        self.drugcentral,
                        drugcentral_contraindication,
                        "contraindication_context",
                    ),
                    (
                        self.drugbank,
                        drugbank_contraindication,
                        "contraindication_context",
                    ),
                ]
            )
        else:
            drugcentral_identity = RestrictedSourceQuery(
                operation="identity_regulatory",
                candidate_name=candidate_name,
                drug_id=registry_id_by_source["DrugCentral"],
                inchikey=inchikey,
                limit=5,
            )
            drugbank_identity = RestrictedSourceQuery(
                operation="identity_regulatory",
                candidate_name=candidate_name,
                drug_id=registry_id_by_source["DrugBank"],
                inchikey=inchikey,
                limit=5,
            )
            patents = RestrictedSourceQuery(
                operation="patent_novelty",
                novelty_terms=tuple(
                    dict.fromkeys(
                        item
                        for item in (
                            candidate.name,
                            candidate.target.symbol,
                            hypothesis.disease,
                        )
                        if item and str(item).strip()
                    )
                ),
                limit=5,
            )
            planned.extend(
                [
                    (
                        self.drugcentral,
                        drugcentral_identity,
                        "known_drug_collision_context",
                    ),
                    (
                        self.drugbank,
                        drugbank_identity,
                        "known_drug_collision_context",
                    ),
                    (self.google_patents, patents, "prior_art_keyword_context"),
                ]
            )

        evidence: dict[str, EvidenceRef] = {}
        target_context: dict[str, EvidenceRef] = {}
        summaries: list[RestrictedSourceSummary] = []
        routing: list[dict[str, Any]] = []
        for adapter, query, reason in planned:
            registry_identity = _registry_identity_routing(
                source=adapter.source,  # type: ignore[attr-defined]
                operation=query.operation,
                query=query,
                registry_ids=registry_ids,
            )
            if allowed is not None and adapter.source not in allowed:  # type: ignore[attr-defined]
                summary = RestrictedSourceSummary(
                    source=adapter.source,  # type: ignore[attr-defined]
                    operation=query.operation,
                    status="skipped",
                    queried=False,
                    error_type="policy_rejected",
                    documentation_url=adapter.documentation_url,  # type: ignore[attr-defined]
                    required_configuration=adapter.required_configuration,  # type: ignore[attr-defined]
                    limitations=adapter.limitations,  # type: ignore[attr-defined]
                )
                summaries.append(summary)
                routing.append(
                    {
                        "source": summary.source,
                        "operation": query.operation,
                        "selected": False,
                        "selection_reason": (
                            "policy_rejected: ResearchBrief.data_policy."
                            "allowed_sources excludes this source"
                        ),
                        "status": "skipped",
                        "queried": False,
                        "record_count": 0,
                        "policy_decision": "policy_rejected",
                        **registry_identity,
                        "configuration_required": list(
                            summary.required_configuration
                        ),
                    }
                )
                continue
            if (
                query.operation in {"identity_regulatory", "contraindication"}
                and adapter.source in registry_identity_conflicts  # type: ignore[attr-defined]
            ):
                summary = RestrictedSourceSummary(
                    source=adapter.source,  # type: ignore[attr-defined]
                    operation=query.operation,
                    # The source was not queried because the candidate itself
                    # carries contradictory stable identities for this
                    # registry.  This is an unresolved validation error, not a
                    # policy/configuration skip.  Recording ``error`` lets the
                    # existing pre-compute safety disposition fail closed
                    # before chemistry, docking, or ADMET can run.
                    status="error",
                    queried=False,
                    error_type="candidate_registry_identity_conflict",
                    documentation_url=adapter.documentation_url,  # type: ignore[attr-defined]
                    required_configuration=adapter.required_configuration,  # type: ignore[attr-defined]
                    limitations=adapter.limitations,  # type: ignore[attr-defined]
                )
                summaries.append(summary)
                routing.append(
                    {
                        "source": summary.source,
                        "operation": query.operation,
                        "selected": False,
                        "selection_reason": (
                            "fail_closed: multiple distinct candidate IDs were "
                            "supplied for this registry namespace"
                        ),
                        "status": "error",
                        "queried": False,
                        "record_count": 0,
                        "policy_decision": "candidate_registry_identity_conflict",
                        **registry_identity,
                        "configuration_required": list(
                            summary.required_configuration
                        ),
                    }
                )
                continue
            result = adapter.query(query)  # type: ignore[attr-defined]
            summaries.append(result.summary.model_copy(deep=True))
            candidate_record_count = 0
            context_record_count = 0
            for record in result.evidence:
                linked = candidate_linked_evidence(candidate, record)
                if linked is not None:
                    evidence[linked.evidence_id] = linked
                    candidate_record_count += 1
                    continue
                decision = attest_candidate_evidence(candidate, record)
                target_context[record.evidence_id] = record.model_copy(
                    update={
                        "metadata": {
                            **record.metadata,
                            "candidate_identity_link": decision.as_metadata(),
                            "context_scope": (
                                "target_level"
                                if query.operation == "repurposing"
                                else "non_candidate_context"
                            ),
                        }
                    },
                    deep=True,
                )
                context_record_count += 1
            routing.append(
                {
                    "source": result.summary.source,
                    "operation": query.operation,
                    "selected": True,
                    "selection_reason": reason,
                    "status": result.summary.status,
                    "queried": result.summary.queried,
                    "record_count": result.summary.record_count,
                    "candidate_record_count": candidate_record_count,
                    "target_context_record_count": context_record_count,
                    "candidate_identity_policy": IDENTITY_LINK_POLICY_VERSION,
                    **registry_identity,
                    "configuration_required": list(
                        result.summary.required_configuration
                    ),
                }
            )

        considered_sources = {"DrugCentral", "DrugBank", "Google Patents"}
        selected_sources = {item.source for item in summaries}
        for source in sorted(considered_sources - selected_sources):
            policy_rejected = allowed is not None and source not in allowed
            routing.append(
                {
                    "source": source,
                    "selected": False,
                    "selection_reason": (
                        "policy_rejected: ResearchBrief.data_policy."
                        "allowed_sources excludes this source"
                        if policy_rejected
                        else "not_applicable_to_repurposing_route"
                        if candidate.route is CandidateRoute.REPURPOSING
                        else "not_applicable_to_de_novo_route"
                    ),
                    "status": "skipped",
                    "queried": False,
                    "record_count": 0,
                    **(
                        {"policy_decision": "policy_rejected"}
                        if policy_rejected
                        else {}
                    ),
                }
            )

        return RestrictedCandidateEnrichment(
            evidence=list(evidence.values()),
            target_context=list(target_context.values()),
            sources=summaries,
            routing=routing,
        )


_REGISTRY_NAMESPACES = ("DrugCentral", "DrugBank", "ChEMBL", "PubChem")
_CHEMBL_ID = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
_DRUGBANK_ID = re.compile(r"^DB\d{5,}$", re.IGNORECASE)
_PUBCHEM_CID = re.compile(r"^(?:CID)?(\d+)$", re.IGNORECASE)
_REGISTRY_KEY_ALIASES = {
    "drugcentral": "DrugCentral",
    "drugcentralid": "DrugCentral",
    "structid": "DrugCentral",
    "drugbank": "DrugBank",
    "drugbankid": "DrugBank",
    "chembl": "ChEMBL",
    "chemblid": "ChEMBL",
    "moleculechemblid": "ChEMBL",
    "pubchem": "PubChem",
    "pubchemid": "PubChem",
    "pubchemcid": "PubChem",
}
_COMPOUND_IDENTITY_ADAPTER_SCOPES = {
    ("ChEMBL", "ChEMBLAdapter", "compound identity and development metadata"),
    ("PubChem", "PubChemAdapter", "compound identity and physicochemical metadata"),
}
_GENERIC_CANDIDATE_IDENTITY_RECORD_TYPES = frozenset(
    {"candidate_compound_identity", "candidate_registry_identity"}
)
_EXACT_IDENTITY_NAME_FIELDS = (
    "candidate_name",
    "drug_name",
    "pref_name",
    "preferred_name",
)
_INCHIKEY_FIELDS = (
    "standard_inchi_key",
    "inchikey",
    "inchi_key",
    "resolved_inchikey",
)


def _mapping_registry_ids(metadata: Mapping[str, Any]) -> dict[str, set[str]]:
    collected: dict[str, dict[str, str]] = {
        namespace: {} for namespace in _REGISTRY_NAMESPACES
    }
    _collect_explicit_registry_fields(metadata, collected)
    _collect_identifier_container(metadata.get("identifiers"), collected)
    return {
        namespace: set(values)
        for namespace, values in collected.items()
    }


def _exact_values(
    metadata: Mapping[str, Any],
    fields: Sequence[str],
) -> set[str]:
    return {
        normalized
        for field in fields
        if (
            normalized := _normalized_exact_name(
                unicodedata.normalize("NFKC", str(metadata.get(field) or ""))
            )
        )
    }


def _candidate_authoritative_registry_ids(
    candidate: CandidateCard,
) -> dict[str, set[str]]:
    combined = _mapping_registry_ids(candidate.metadata)
    for key in ("source_metadata", "source_evidence"):
        raw = candidate.metadata.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            nested = item.get("metadata")
            for mapping in (
                item,
                nested if isinstance(nested, Mapping) else {},
            ):
                for namespace, values in _mapping_registry_ids(mapping).items():
                    combined[namespace].update(values)
    return combined


def _stable_identity_decision(
    *,
    candidate: CandidateCard,
    evidence_mappings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attest registry evidence without SMILES normalization or RDKit.

    Stable namespace conflicts reject immediately.  Only when no supplied
    stable namespace conflicts may exact InChIKey, then exact normalized name,
    establish the identity.  This deliberately has no fuzzy/SMILES fallback.
    """

    candidate_ids = _candidate_authoritative_registry_ids(candidate)
    evidence_ids = {
        namespace: set() for namespace in _REGISTRY_NAMESPACES
    }
    for mapping in evidence_mappings:
        for namespace, values in _mapping_registry_ids(mapping).items():
            evidence_ids[namespace].update(values)

    shared_namespaces: list[str] = []
    matched_namespaces: list[str] = []
    conflicts: list[str] = []
    for namespace in _REGISTRY_NAMESPACES:
        left = candidate_ids[namespace]
        right = evidence_ids[namespace]
        if not left or not right:
            continue
        shared_namespaces.append(namespace)
        if left.intersection(right):
            matched_namespaces.append(namespace)
        else:
            conflicts.append(namespace)
    if conflicts:
        return {
            "status": "identity_conflict",
            "matched_by": matched_namespaces[0] if matched_namespaces else None,
            "shared_identifiers": shared_namespaces,
            "conflicts": conflicts,
            "smiles_or_fuzzy_fallback_used": False,
        }
    if matched_namespaces:
        return {
            "status": "exact_candidate",
            "matched_by": matched_namespaces[0],
            "shared_identifiers": shared_namespaces,
            "conflicts": [],
            "smiles_or_fuzzy_fallback_used": False,
        }

    candidate_inchikeys = _exact_values(candidate.metadata, _INCHIKEY_FIELDS)
    if candidate.inchikey:
        candidate_inchikeys.add(_normalized_exact_name(candidate.inchikey))
    evidence_inchikeys: set[str] = set()
    for mapping in evidence_mappings:
        evidence_inchikeys.update(_exact_values(mapping, _INCHIKEY_FIELDS))
    if candidate_inchikeys and evidence_inchikeys:
        if candidate_inchikeys.isdisjoint(evidence_inchikeys):
            return {
                "status": "identity_conflict",
                "matched_by": None,
                "shared_identifiers": ["inchikey"],
                "conflicts": ["inchikey"],
                "smiles_or_fuzzy_fallback_used": False,
            }
        return {
            "status": "exact_candidate",
            "matched_by": "inchikey",
            "shared_identifiers": ["inchikey"],
            "conflicts": [],
            "smiles_or_fuzzy_fallback_used": False,
        }

    candidate_names = {_normalized_exact_name(candidate.name)}
    candidate_names.update(
        _exact_values(candidate.metadata, _EXACT_IDENTITY_NAME_FIELDS)
    )
    evidence_names: set[str] = set()
    for mapping in evidence_mappings:
        evidence_names.update(_exact_values(mapping, _EXACT_IDENTITY_NAME_FIELDS))
    if candidate_names.intersection(evidence_names):
        return {
            "status": "exact_candidate",
            "matched_by": "exact_name",
            "shared_identifiers": ["exact_name"],
            "conflicts": [],
            "smiles_or_fuzzy_fallback_used": False,
        }
    if any(
        _has_exact_candidate_attestation(mapping)
        for mapping in evidence_mappings
    ):
        return {
            "status": "exact_candidate",
            "matched_by": "persisted_exact_attestation",
            "shared_identifiers": ["persisted_exact_attestation"],
            "conflicts": [],
            "smiles_or_fuzzy_fallback_used": False,
        }
    return {
        "status": "target_context",
        "matched_by": None,
        "shared_identifiers": [],
        "conflicts": [],
        "smiles_or_fuzzy_fallback_used": False,
    }


def _candidate_evidence_registry_attestation(
    candidate: CandidateCard,
    evidence: EvidenceRef,
) -> dict[str, Any]:
    """Lightweight exact attestation for candidate registry-ID extraction."""

    return _stable_identity_decision(
        candidate=candidate,
        evidence_mappings=(evidence.metadata,),
    )


def _attested_source_record_has_no_stable_conflict(
    candidate: CandidateCard,
    record: Mapping[str, Any],
    nested: Mapping[str, Any],
) -> bool:
    decision = _stable_identity_decision(
        candidate=candidate,
        evidence_mappings=(record, nested),
    )
    return decision["status"] != "identity_conflict"


def _candidate_registry_ids(candidate: CandidateCard) -> dict[str, tuple[str, ...]]:
    """Resolve candidate registry IDs without crossing registry namespaces.

    Explicit candidate metadata is authoritative.  Evidence-derived IDs are
    admitted only after an exact-candidate identity attestation (evaluated now
    or already persisted without a current conflict).  Bare values in a mixed
    ``identifiers`` list are used only when their namespace is self-describing;
    a numeric DrugCentral ID is therefore accepted from an explicit
    DrugCentral field/source, never guessed globally.
    """

    collected: dict[str, dict[str, str]] = {
        namespace: {} for namespace in _REGISTRY_NAMESPACES
    }
    metadata = candidate.metadata
    _collect_explicit_registry_fields(metadata, collected)
    _collect_identifier_container(metadata.get("identifiers"), collected)

    for key in ("source_metadata", "source_evidence", "restricted_source_records"):
        raw = metadata.get(key)
        items = raw if isinstance(raw, (list, tuple)) else ()
        for item in items:
            if not isinstance(item, Mapping):
                continue
            nested = item.get("metadata")
            nested_metadata = nested if isinstance(nested, Mapping) else {}
            if not _has_exact_candidate_attestation(item, nested_metadata):
                continue
            if not _attested_source_record_has_no_stable_conflict(
                candidate, item, nested_metadata
            ):
                continue
            _collect_attested_source_record(item, nested_metadata, collected)

    for evidence in candidate.evidence:
        decision = _candidate_evidence_registry_attestation(
            candidate, evidence
        )
        if decision["status"] != "exact_candidate":
            continue
        namespace = _generic_candidate_identity_namespace(
            {"source": evidence.source, "external_id": evidence.external_id},
            evidence.metadata,
        )
        if namespace is not None:
            _add_registry_id(collected, namespace, evidence.external_id)
        _collect_explicit_registry_fields(evidence.metadata, collected)
        _collect_identifier_container(
            evidence.metadata.get("identifiers"), collected
        )

    return {
        namespace: tuple(
            canonical
            for _, canonical in sorted(
                values.items(), key=lambda item: item[0]
            )
        )
        for namespace, values in collected.items()
    }


def _collect_attested_source_record(
    record: Mapping[str, Any],
    nested: Mapping[str, Any],
    collected: dict[str, dict[str, str]],
) -> None:
    namespace = _generic_candidate_identity_namespace(record, nested)
    if namespace is not None:
        for value in (
            record.get("external_id"),
            record.get("drug_id"),
            nested.get("external_id"),
            nested.get("drug_id"),
        ):
            _add_registry_id(collected, namespace, value)
    _collect_explicit_registry_fields(record, collected)
    _collect_explicit_registry_fields(nested, collected)
    _collect_identifier_container(record.get("identifiers"), collected)
    _collect_identifier_container(nested.get("identifiers"), collected)


def _collect_explicit_registry_fields(
    metadata: Mapping[str, Any],
    collected: dict[str, dict[str, str]],
) -> None:
    for key, value in metadata.items():
        normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
        namespace = _REGISTRY_KEY_ALIASES.get(normalized_key)
        if namespace is None:
            continue
        for item in _identifier_values(value):
            _add_registry_id(collected, namespace, item)


def _collect_identifier_container(
    raw: Any,
    collected: dict[str, dict[str, str]],
) -> None:
    if isinstance(raw, Mapping):
        _collect_explicit_registry_fields(raw, collected)
        return
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return
    for value in raw:
        _collect_self_describing_identifier(value, collected)


def _generic_candidate_identity_namespace(
    record: Mapping[str, Any],
    nested: Mapping[str, Any],
) -> str | None:
    """Admit a generic record ID only under an explicit compound-ID contract.

    Exact candidate attestation is checked by the caller, but is not enough to
    establish identifier semantics.  ChEMBL activity, assay and document
    records can all be exact-candidate evidence while their ``external_id`` is
    not a molecule identifier.  The same applies to target-context and
    restricted-query records in other registries.

    Existing public compound identity adapters have an exact adapter/scope
    contract.  Other producers must explicitly provide both
    ``registry_identity_namespace`` and either a candidate-compound record type
    or the ``identity_regulatory`` operation.  Explicit typed fields such as
    ``molecule_chembl_id`` and ``drugcentral_id`` are collected separately and
    do not depend on this generic-ID path.
    """

    source_namespace = _registry_namespace_from_source(
        record.get("source") or nested.get("source")
    )
    explicit_namespace_values = {
        namespace
        for value in (
            record.get("registry_identity_namespace"),
            nested.get("registry_identity_namespace"),
        )
        if (namespace := _registry_namespace_from_source(value)) is not None
    }
    if len(explicit_namespace_values) > 1:
        return None
    explicit_namespace = next(iter(explicit_namespace_values), None)
    if (
        explicit_namespace is not None
        and source_namespace is not None
        and explicit_namespace != source_namespace
    ):
        return None

    record_types = {
        str(value).strip().casefold()
        for value in (
            record.get("record_type"),
            nested.get("record_type"),
            record.get("registry_identity_record_type"),
            nested.get("registry_identity_record_type"),
        )
        if str(value or "").strip()
    }
    operations = {
        str(value).strip().casefold()
        for value in (
            record.get("operation"),
            nested.get("operation"),
            record.get("restricted_operation"),
            nested.get("restricted_operation"),
        )
        if str(value or "").strip()
    }
    if explicit_namespace is not None and (
        record_types.intersection(_GENERIC_CANDIDATE_IDENTITY_RECORD_TYPES)
        or "identity_regulatory" in operations
    ):
        return explicit_namespace

    adapter = str(
        nested.get("retrieval_adapter") or record.get("retrieval_adapter") or ""
    ).strip()
    scope = str(
        nested.get("evidence_scope") or record.get("evidence_scope") or ""
    ).strip()
    adapter_contract = (source_namespace, adapter, scope)
    if adapter_contract in _COMPOUND_IDENTITY_ADAPTER_SCOPES:
        return source_namespace
    return None


def _collect_self_describing_identifier(
    value: Any,
    collected: dict[str, dict[str, str]],
) -> bool:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not normalized:
        return False
    if ":" in normalized:
        raw_namespace, raw_value = normalized.split(":", 1)
        namespace = _REGISTRY_KEY_ALIASES.get(
            re.sub(r"[^a-z0-9]", "", raw_namespace.casefold())
        )
        if namespace is not None:
            _add_registry_id(collected, namespace, raw_value)
            return True
    if _CHEMBL_ID.fullmatch(normalized):
        _add_registry_id(collected, "ChEMBL", normalized)
        return True
    if _DRUGBANK_ID.fullmatch(normalized):
        _add_registry_id(collected, "DrugBank", normalized)
        return True
    if normalized.upper().startswith("CID") and _PUBCHEM_CID.fullmatch(normalized):
        _add_registry_id(collected, "PubChem", normalized)
        return True
    return False


def _identifier_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


def _add_registry_id(
    collected: dict[str, dict[str, str]],
    namespace: str,
    value: Any,
) -> None:
    normalized = _normalize_registry_id(namespace, value)
    if normalized is None:
        return
    collected[namespace].setdefault(normalized.casefold(), normalized)


def _normalize_registry_id(namespace: str, value: Any) -> str | None:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not normalized:
        return None
    aliases = {
        "DrugCentral": ("drugcentral:",),
        "DrugBank": ("drugbank:",),
        "ChEMBL": ("chembl:",),
        "PubChem": ("pubchem:", "cid:"),
    }
    for prefix in aliases[namespace]:
        if normalized.casefold().startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    if not normalized:
        return None
    if namespace == "ChEMBL":
        return normalized.upper() if _CHEMBL_ID.fullmatch(normalized) else None
    if namespace == "DrugBank":
        return normalized.upper() if _DRUGBANK_ID.fullmatch(normalized) else None
    if namespace == "PubChem":
        match = _PUBCHEM_CID.fullmatch(normalized)
        return match.group(1) if match else None
    return normalized


def _has_exact_candidate_attestation(*metadata_items: Mapping[str, Any]) -> bool:
    for metadata in metadata_items:
        raw = metadata.get("candidate_identity_link")
        if isinstance(raw, Mapping) and raw.get("status") == "exact_candidate":
            return True
    return False


def _registry_namespace_from_source(value: Any) -> str | None:
    normalized = re.sub(
        r"[^a-z0-9]", "", unicodedata.normalize("NFKC", str(value or "")).casefold()
    )
    if normalized == "drugcentral" or normalized.startswith("drugcentral"):
        return "DrugCentral"
    if normalized == "drugbank" or normalized.startswith("drugbank"):
        return "DrugBank"
    if normalized == "chembl" or normalized.startswith("chembl"):
        return "ChEMBL"
    if normalized == "pubchem" or normalized.startswith("pubchem"):
        return "PubChem"
    return None


def _normalized_exact_name(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _registry_identity_routing(
    *,
    source: str,
    operation: str,
    query: RestrictedSourceQuery,
    registry_ids: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    if source not in {"DrugCentral", "DrugBank"} or operation not in {
        "identity_regulatory",
        "contraindication",
    }:
        return {}
    values = registry_ids[source]
    return {
        "registry_identity_namespace": source,
        "registry_identity_ids": list(values),
        "registry_identity_query_mode": (
            "conflict_unresolved"
            if len(values) > 1
            else "exact_registry_id"
            if query.drug_id
            else "inchikey_then_exact_normalized_name"
            if query.inchikey
            else "exact_normalized_name"
        ),
        "registry_identity_conflict": len(values) > 1,
    }


__all__ = [
    "RestrictedCandidateEnrichment",
    "RoutedRestrictedCandidateEnricher",
]
