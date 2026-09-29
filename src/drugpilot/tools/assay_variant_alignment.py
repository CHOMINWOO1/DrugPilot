from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


ASSAY_VARIANT_ALIGNMENT_SCHEMA = "assay_hypothesis_variant_alignment_v2"
ASSAY_VARIANT_ALIGNMENT_POLICY = (
    "target-linked-variant-annotation-not-construct-identity-v2"
)


def _bounded_text(value: Any) -> str:
    if isinstance(value, Mapping):
        excerpt = value.get("excerpt")
        return str(excerpt or "")[:4_000]
    return str(value or "")[:4_000]


def bound_semantic_hypothesis(hypotheses, packet, candidate):
    """Resolve an explicit candidate link against the recorded execution owner."""
    link = candidate.metadata.get("assay_variant_alignment_hypothesis_id")
    portfolio_link = candidate.metadata.get("portfolio_hypothesis_id")
    if not link or (portfolio_link is not None and portfolio_link != link):
        return None
    if packet.portfolio_enabled:
        executions = [s for s in packet.target_executions if candidate.candidate_id in s.candidate_ids]
        if portfolio_link != link or len(executions) != 1 or executions[0].hypothesis_id != link:
            return None
    else:
        selection = packet.metadata.get("primary_hypothesis_selection", {})
        selected = selection.get("selected_hypothesis_id") if isinstance(selection, Mapping) else None
        if selected != link:
            return None
    hypothesis = next((h for h in hypotheses.hypotheses if h.hypothesis_id == link), None)
    if (hypothesis is None or hypothesis.target.symbol != candidate.target.symbol
            or (hypothesis.target.uniprot_id is not None
                and hypothesis.target.uniprot_id != candidate.target.uniprot_id)):
        return None
    return hypothesis


def extract_target_variants(
    target_symbol: str,
    texts: Sequence[Any],
    *,
    allow_unqualified: bool = False,
) -> tuple[str, ...]:
    """Extract only explicit target-linked ITD or amino-acid variants.

    This deliberately does not infer wild type from absence and does not
    interpret broad words such as ``mutant`` as a specific construct.
    """

    symbol = target_symbol.strip().upper()
    if not symbol:
        return ()
    target = rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])"
    separator = r"[-/\u2010-\u2015\u2212:]"
    variants: dict[str, str] = {}
    for raw in texts:
        text = _bounded_text(raw)
        if not text:
            continue
        from drugpilot.agents.molecular_scope import mentions
        for scoped in mentions(text,target_symbol):
            if scoped["kind"] in {"fusion_label","exon_event","wild_type_label"}:
                variants.setdefault(scoped["label"],scoped["label"])
        if re.search(
            rf"(?:{target}\s*(?:{separator}\s*|\s+)ITD(?![A-Za-z0-9])|"
            rf"{target}[^.!?]{{0,80}}internal tandem duplication\b|"
            rf"internal tandem duplication[^.!?]{{0,80}}{target})",
            text,
            re.I,
        ):
            variants.setdefault(f"{symbol}-ITD", f"{symbol}-ITD")
        for match in re.finditer(
            rf"{target}\s*(?:(?:{separator})\s*|\s+)"
            r"(?:p\.)?([A-Za-z][0-9]{1,5}[A-Za-z*])"
            r"(?![A-Za-z0-9])",
            text,
            re.I,
        ):
            residue = match.group(1).upper()
            variants.setdefault(f"{symbol}-{residue}", f"{symbol}-{residue}")

        # Published model names also use concatenated target-residue notation
        # (e.g. IDH1R132H-MOLM-13). This is an explicit lexical annotation, not
        # evidence of genotype presence, selectivity, or construct identity.
        for match in re.finditer(
            rf"(?<![A-Za-z0-9]){re.escape(symbol)}([A-Za-z][0-9]{{1,5}}[A-Za-z*])(?![A-Za-z0-9])",
            text, re.I,
        ):
            residue = match.group(1).upper()
            variants.setdefault(f"{symbol}-{residue}", f"{symbol}-{residue}")

        # Unqualified tokens are legal only in an already target-bound
        # variant_sequence field, never in free hypothesis/assay prose.
        if allow_unqualified:
            # Do not strip an embedded object's ownership/accession fields:
            # a foreign gene's mutation could otherwise become this target's.
            mutation = text
            # A bare structured token/list is not a licence to reinterpret
            # another gene's residue mentioned in a prose sentence.
            if not re.fullmatch(r"\s*(?:p\.)?[A-Za-z][0-9]{1,5}[A-Za-z*](?:\s*[,;/+]\s*(?:p\.)?[A-Za-z][0-9]{1,5}[A-Za-z*])*\s*", mutation):
                continue
            for match in re.finditer(
                r"(?<![A-Za-z0-9])(?:p\.)?([A-Za-z][0-9]{1,5}[A-Za-z*])"
                r"(?![A-Za-z0-9])", mutation, re.I,
            ):
                residue = match.group(1).upper()
                variants.setdefault(f"{symbol}-{residue}", f"{symbol}-{residue}")
    return tuple(variants.values())[:8]


def assay_variant_alignment(
    *,
    target_symbol: str,
    hypothesis_texts: Sequence[Any],
    chembl_assay_context: Any,
    expected_target_chembl_id: str | None = None,
    molecular_scope: Any = None,
) -> dict[str, Any]:
    """Attest assay/hypothesis variant alignment without changing eligibility."""

    if isinstance(molecular_scope,Mapping) and molecular_scope.get("policy") == "source_bound_molecular_hypothesis_v1":
        scoped_texts = [a.get("quote","") for a in molecular_scope.get("anchors",[])
            if isinstance(a,Mapping)]
        declared = extract_target_variants(target_symbol,scoped_texts)
    else:
        declared = extract_target_variants(target_symbol, hypothesis_texts)
    context = (
        chembl_assay_context
        if isinstance(chembl_assay_context, Mapping)
        else {}
    )
    record = context.get("record")
    record = record if isinstance(record, Mapping) else {}
    bound = bool(re.fullmatch(r"CHEMBL[0-9]+", str(expected_target_chembl_id or ""))
        and re.fullmatch(r"CHEMBL[0-9]+", str(context.get("assay_chembl_id") or ""))
        and context.get("schema_version") == "chembl_assay_context_v1"
        and context.get("retrieval_status") == "completed"
        and context.get("missing_values_imputed") is False
        and context.get("inference_used") is False
        and context.get("binding_status") == "exact"
        and context.get("assay_chembl_id") == record.get("assay_chembl_id")
        and record.get("target_chembl_id") == expected_target_chembl_id)
    variant_text = _bounded_text(record.get("variant_sequence"))
    sequence_unresolved = bool(re.search(r"\b(?:UNDEFINED\s+MUTATION|UNKNOWN|UNSPECIFIED|NOT\s+REPORTED)\b", variant_text, re.I))
    assay_variants = tuple(dict.fromkeys((
        *extract_target_variants(target_symbol, (record.get("description"),)),
        *extract_target_variants(target_symbol, (record.get("variant_sequence"),), allow_unqualified=bound),
    )))
    condition = context.get("condition_coverage")
    condition = condition if isinstance(condition, Mapping) else {}
    mutation_state = str(
        condition.get("mutation_state")
        or "unknown_not_assumed_wild_type"
    )

    if not declared:
        status = "not_declared"
        reasons = ["HYPOTHESIS_VARIANT_NOT_DECLARED"]
    elif not bound:
        status = "unknown_not_assumed_match"
        reasons = ["ASSAY_TARGET_BINDING_UNVERIFIED"]
    elif set(declared).intersection(assay_variants):
        if set(declared) != set(assay_variants) or len(declared) > 1:
            status = "partial_variant_overlap"
            reasons = ["ASSAY_VARIANT_PARTIAL_OR_COMPOUND_CONTEXT_UNVERIFIED"]
        elif any(v.endswith("-ITD") for v in declared):
            status = "variant_class_match_only"
            reasons = ["ASSAY_VARIANT_CLASS_ONLY_NOT_EXACT_CONSTRUCT"]
        elif sequence_unresolved:
            status = "unknown_not_assumed_match"
            reasons = ["ASSAY_VARIANT_SEQUENCE_UNDEFINED"]
        else:
            status = "variant_annotation_match"
            reasons = ["NAMED_VARIANT_ANNOTATION_ONLY_NOT_EXACT_CONSTRUCT"]
    elif assay_variants:
        status = "explicit_mismatch"
        reasons = ["ASSAY_VARIANT_EXPLICIT_MISMATCH"]
    else:
        status = "unknown_not_assumed_match"
        reasons = ["ASSAY_VARIANT_NOT_REPORTED"]

    return {
        "schema_version": ASSAY_VARIANT_ALIGNMENT_SCHEMA,
        "policy": ASSAY_VARIANT_ALIGNMENT_POLICY,
        "target_symbol": target_symbol.strip().upper(),
        **({"molecular_hypothesis_scope":dict(molecular_scope),
            "hypothesis_scope_authority":"G0-bound_source_scope_not_exact_construct_identity"}
            if isinstance(molecular_scope,Mapping) else {}),
        "declared_hypothesis_variants": list(declared),
        "observed_assay_variants": list(assay_variants),
        "assay_mutation_state": mutation_state,
        "assay_chembl_id": context.get("assay_chembl_id"),
        "status": status,
        "reason_codes": reasons,
        "comparison_scope": "reported_variant_labels_not_sequence_or_construct_identity",
        "exact_variant_match": False,
        "construct_identity_verified": False,
        "assay_target_binding_verified": bound,
        "assay_variant_sequence_unresolved": sequence_unresolved or not variant_text,
        "unknown_assumed_wild_type": False,
        "unknown_assumed_match": False,
        "score_effect_applied": False,
        "eligibility_effect_applied": False,
        "hypothesis_text_authority": "narrative_context_only",
        "missing_values_imputed": False,
    }


__all__ = [
    "ASSAY_VARIANT_ALIGNMENT_POLICY",
    "ASSAY_VARIANT_ALIGNMENT_SCHEMA",
    "assay_variant_alignment",
    "extract_target_variants",
    "bound_semantic_hypothesis",
]
