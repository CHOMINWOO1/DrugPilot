"""Source-bound variant vocabulary for follow-up search, never claim authority."""
from hashlib import sha256

from drugpilot.agents.molecular_scope import BRANCH_POLICY, declared_labels, mentions

KEY = "g0_variant_search_bindings"
POLICY = "source_bound_variant_search_v1"
KINDS = {"amino_acid_variant", "itd_class", "exon_event"}


def validate(bindings, target):
    if not isinstance(bindings,list) or len(bindings)>4:
        raise ValueError("VARIANT_SEARCH_BINDING_INVALID")
    terms=[]
    for b in bindings:
        if not isinstance(b,dict):
            raise ValueError("VARIANT_SEARCH_BINDING_INVALID")
        text,anchor=b.get("excerpt"),b.get("anchor")
        if (b.get("policy")!=POLICY or b.get("target_symbol")!=target
                or b.get("source") not in {"PubMed","Europe PMC"}
                or b.get("provenance_class") not in {"external_live","external_cache"}
                or not b.get("external_id") or not b.get("evidence_id")
                or b.get("authority")!="search_scope_only_not_support_or_identity"
                or not isinstance(text,str) or not 0<len(text)<=320
                or sha256(text.encode()).hexdigest()!=b.get("excerpt_sha256")
                or not isinstance(anchor,dict)):
            raise ValueError("VARIANT_SEARCH_BINDING_INVALID")
        matched=[m for m in mentions(text,target) if m["kind"] in KINDS
            and m["label"]==b.get("label")
            and {k:m[k] for k in ("start","end","quote","kind")}==anchor]
        if len(matched)!=1:
            raise ValueError("VARIANT_SEARCH_BINDING_INVALID")
        if anchor["quote"] not in terms:
            terms.append(anchor["quote"])
    return tuple(terms)


def query_terms(bindings, target):
    """Return bounded aliases for a validated, source-bound variant label.

    The observed quote remains the authority for opening this search lane.
    Aliases only broaden retrieval vocabulary; they never establish construct
    identity, evidence support, or a gate result.  Every retrieved excerpt is
    still required to pass the exact scoped-intervention predicate.
    """
    observed_terms = validate(bindings, target)
    terms = list(observed_terms)
    for binding in bindings:
        label = binding["label"]
        kind = binding["anchor"]["kind"]
        suffix = label.split(":", 1)[1] if ":" in label else ""
        aliases = ()
        if kind == "itd_class" and suffix == "ITD":
            aliases = (
                f"{target}-ITD",
                f"{target}/ITD",
                f"{target} ITD",
                f"{target} internal tandem duplication",
                f"internal tandem duplication of {target}",
            )
        elif kind == "amino_acid_variant" and suffix:
            aliases = (
                f"{target}-{suffix}",
                f"{target} {suffix}",
                f"{target} p.{suffix}",
            )
        elif kind == "exon_event" and suffix.startswith("EXON "):
            event = suffix.casefold()
            aliases = (f"{target} {event}",)
        for alias in aliases:
            if alias not in terms:
                terms.append(alias)
    # Four source bindings times five aliases is the largest supported input.
    # Keep the result independently bounded even if more kinds are added later.
    return tuple(terms[:20])


def from_screening_entry(brief,entry,target):
    scope=entry.get("molecular_hypothesis_scope")
    if (not isinstance(scope,dict) or scope.get("policy")!=BRANCH_POLICY
            or scope.get("origin")!="observed_source_span"
            or scope.get("target_symbol")!=target):
        return []
    declared=declared_labels(brief,target)
    if declared and scope.get("label") not in declared:
        return []
    for anchor in scope.get("anchors",[]):
        if not isinstance(anchor,dict) or anchor.get("kind") not in KINDS:
            continue
        for record in entry.get("unique_external_records",[]):
            provenance=record.get("provenance_classes",[])
            if (not provenance or not set(provenance).issubset({"external_live","external_cache"})
                    or anchor.get("source") not in record.get("sources",[])
                    or anchor.get("external_id") not in record.get("external_ids",[])):
                continue
            for unit in record.get("evidence_units",[]):
                if (unit.get("evidence_id")!=anchor.get("evidence_id")
                        or unit.get("excerpt_hash")!=anchor.get("excerpt_sha256")):
                    continue
                bound={"policy":POLICY,"target_symbol":target,"label":scope.get("label"),
                    **{k:anchor.get(k) for k in ("source","external_id","evidence_id","excerpt_sha256")},
                    "provenance_class":sorted(provenance)[0],"excerpt":unit.get("bounded_excerpt"),
                    "anchor":{k:anchor.get(k) for k in ("start","end","quote","kind")},
                    "authority":"search_scope_only_not_support_or_identity"}
                try:
                    validate([bound],target)
                except (ValueError,TypeError):
                    continue
                return [bound]
    return []
