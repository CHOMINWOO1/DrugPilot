"""Exact source-span scope mentions, never inferred construct identity or efficacy.

Observed labels and user-declared labels are separate. A fusion mention cannot
silently become evidence about an isolated partner or a wild-type protein.
This conservative lexical layer does not resolve accessions, sequence, causality,
negation, or the biologically actionable partner; the original audits still do.
"""
from hashlib import sha256
import re

POLICY = "source_bound_scope_mentions_v1"
KEY = "molecular_scope"
HYPOTHESIS_SCOPE_KEY = "molecular_hypothesis_scope"
BRANCH_POLICY = "source_bound_molecular_hypothesis_v1"
SEARCH_POLICY = "source_bound_fusion_search_v1"
SEARCH_KEY = "g0_fusion_search_bindings"
_SEP = r"(?:\s*::\s*|\s*[-\u2010-\u2015\u2212]\s*)"
_GENE = r"[A-Z][A-Z0-9]{1,14}"
_FUSION = re.compile(rf"(?<![A-Za-z0-9])(?P<a>{_GENE}){_SEP}(?P<b>{_GENE})(?![A-Za-z0-9])")


def mentions(text, target):
    """Return quoted label occurrences, not assertions that they are present."""
    if not isinstance(text, str) or not isinstance(target, str) or not target.strip():
        return []
    target = target.strip().upper()
    token = rf"(?<![A-Za-z0-9]){re.escape(target)}(?![A-Za-z0-9])"
    found = []
    def add(match, kind, label, participants):
        found.append({"kind":kind,"label":label,"participants":participants,
            "start":match.start(),"end":match.end(),"quote":match.group(),
            "identity_verified":False,"assertion_status":"mention_not_presence_or_effect"})
    for m in _FUSION.finditer(text):
        a, b = m.group("a"), m.group("b")
        if target not in (a,b) or a == b or any(
            re.fullmatch(r"(?:[A-Z][0-9]{1,5}[A-Z]|ITD|WT)", p) for p in (a,b)):
            continue
        # Hyphens can denote a complex or pathway. Require an explicit fusion
        # cue in the same sentence; double-colon is an explicit label separator.
        start = max(text.rfind(c, 0, m.start()) for c in ".!?\n") + 1
        ends = [text.find(c, m.end()) for c in ".!?\n"]
        end = min((i for i in ends if i >= 0), default=len(text))
        if "::" not in m.group() and not re.search(r"\b(?:fusion|fusions|fused|rearrangement)\b", text[start:end], re.I):
            continue
        add(m, "fusion_label", a + "::" + b, [a,b])
    patterns = (
        ("amino_acid_variant", rf"{token}\s*(?:[-\u2010-\u2015:]\s*|\s+)(?:p\.)?(?P<v>[A-Z][0-9]{{1,5}}[A-Z*])(?![A-Za-z0-9])"),
        ("itd_class", rf"{token}\s*(?:[-/\u2010-\u2015]\s*|\s+)(?P<v>ITD|internal tandem duplication)(?![A-Za-z0-9])"),
        ("exon_event", rf"{token}\s+(?P<v>exon\s+[0-9]{{1,3}}\s+(?:skipping|deletion|insertion)s?)(?![A-Za-z0-9])"),
        ("wild_type_label", rf"(?:{token}\s+(?P<v>wild[- ]type|WT)|wild[- ]type\s+{token})(?![A-Za-z0-9])"),
    )
    for kind, pattern in patterns:
        for m in re.finditer(pattern, text, re.I):
            value = "WT" if kind == "wild_type_label" else "ITD" if kind == "itd_class" else " ".join(m.group("v").upper().split())
            add(m, kind, target + ":" + value, [target])
    return sorted(found, key=lambda r:(r["start"],r["end"],r["kind"]))


def declared_labels(brief, target):
    fields = (brief.disease, brief.research_goal, brief.rationale or "",
        brief.constraints.target_cell_type or "", *brief.human_priorities)
    return sorted({m["label"] for text in fields for m in mentions(text, target)})


def source_scope(evidence, target):
    text = evidence.metadata.get("excerpt")
    verified = (evidence.source in {"PubMed", "Europe PMC"}
        and evidence.metadata.get("provenance_class") in {"external_live", "external_cache"}
        and not evidence.metadata.get("fixture") and not evidence.metadata.get("mock")
        and isinstance(text, str) and 0 < len(text) <= 12000
        and sha256(text.encode()).hexdigest() == evidence.excerpt_hash)
    all_mentions = mentions(text, target) if verified else []
    target_hits = list(re.finditer(rf"(?<![A-Za-z0-9]){re.escape(target)}(?![A-Za-z0-9])", text, re.I)) if verified else []
    only_scoped = bool(target_hits) and all(any(m["start"] <= t.start() and t.end() <= m["end"]
        for m in all_mentions) for t in target_hits)
    return {"policy":POLICY,"target_symbol":target,"source":evidence.source,
        "external_id":evidence.external_id,"evidence_id":evidence.evidence_id,
        "excerpt_sha256":evidence.excerpt_hash,"source_verified":verified,
        "mentions":all_mentions[:12],"mention_count":len(all_mentions),
        "mentions_omitted":max(0,len(all_mentions)-12),
        "target_mentions_only_scoped":only_scoped,
        "wild_type_inferred":False,"construct_identity_verified":False,
        "authority":"source_span_annotation_not_primary_claim_or_gate_verdict"}


def scope_check(brief, target, evidence):
    scope = source_scope(evidence, target)
    declared = declared_labels(brief, target)
    observed = sorted({m["label"] for m in scope["mentions"]})
    if not scope["source_verified"] or scope["mentions_omitted"]:
        status = "scope_unverifiable"
    elif observed and declared and set(observed).isdisjoint(declared):
        status = "declared_observed_scope_mismatch"
    elif observed and declared and not set(observed).issubset(declared):
        # A source may report separately measured results for the declared
        # variant and additional variants in one bounded passage.  Presence of
        # the additional labels is not itself a contradiction.  Unscoped
        # claims still cannot use this record; a scoped claim must separately
        # bind both semantic model target quotes to its one declared label.
        status = "declared_scope_present_with_additional_scopes"
    elif observed and not declared and scope["target_mentions_only_scoped"]:
        status = "scoped_source_not_general_target"
    else:
        status = "no_detected_scope_conflict"  # Not proof of identity/alignment.
    return {"status":status,"declared_labels":declared,"observed_labels":observed,
        "source_scope":scope,"scope_match_is_not_construct_identity":True}


def annotate_scopes(brief, target, evidence):
    return [e.model_copy(update={"metadata":{**e.metadata,KEY:scope_check(brief,target,e)}},deep=True)
        if e.source in {"PubMed","Europe PMC"} else e.model_copy(deep=True) for e in evidence]


def observed_scope_labels(brief, target, evidence):
    """Candidate research scopes only; no label is itself biological support."""
    declared = set(declared_labels(brief, target))
    labels = {m["label"] for e in evidence for m in source_scope(e, target)["mentions"]}
    return sorted(labels & declared if declared else labels)


def hypothesis_scope(brief, target, evidence, label):
    if label not in observed_scope_labels(brief, target, evidence):
        return None
    anchors = []
    for e in evidence:
        s = source_scope(e, target)
        for m in s["mentions"]:
            if m["label"] == label:
                anchors.append({"evidence_id":e.evidence_id,"source":e.source,
                    "external_id":e.external_id,"excerpt_sha256":e.excerpt_hash,
                    "start":m["start"],"end":m["end"],"quote":m["quote"],"kind":m["kind"]})
    return {"policy":BRANCH_POLICY,"target_symbol":target,"label":label,
        "origin":"observed_source_span","anchors":anchors,
        "user_declared":label in declared_labels(brief,target),
        "canonical_identity_is_component_only":"::" in label,
        "construct_identity_verified":False,
        "authority":"research_scope_not_support_or_experimental_identity"}


def valid_hypothesis_scope(brief, hypothesis):
    stored = hypothesis.target.metadata.get(HYPOTHESIS_SCOPE_KEY)
    if not isinstance(stored, dict) or not isinstance(stored.get("label"), str):
        return False
    expected = hypothesis_scope(brief,hypothesis.target.symbol,hypothesis.evidence,stored["label"])
    return expected is not None and stored == expected


def validated_fusion_search_terms(bindings, target):
    """Recheck every observed span before it can influence a source query.

    This is query vocabulary only: even a negative/review mention may guide a
    search, but it never asserts construct identity, disease relevance or effect.
    """
    if not isinstance(bindings, list) or len(bindings) > 4:
        raise ValueError("FUSION_SEARCH_BINDING_INVALID")
    terms = []
    for b in bindings:
        if not isinstance(b, dict):
            raise ValueError("FUSION_SEARCH_BINDING_INVALID")
        text, anchor = b.get("excerpt"), b.get("anchor")
        if (b.get("policy") != SEARCH_POLICY or b.get("target_symbol") != target
                or b.get("source") not in {"PubMed", "Europe PMC"}
                or b.get("provenance_class") not in {"external_live", "external_cache"}
                or not b.get("external_id") or not b.get("evidence_id")
                or b.get("authority") != "search_scope_only_not_support_or_identity"
                or not isinstance(text, str) or not 0 < len(text) <= 320
                or sha256(text.encode()).hexdigest() != b.get("excerpt_sha256")
                or not isinstance(anchor, dict)):
            raise ValueError("FUSION_SEARCH_BINDING_INVALID")
        matched = [m for m in mentions(text, target) if m["kind"] == "fusion_label"
            and m["label"] == b.get("label")
            and {k:m[k] for k in ("start", "end", "quote", "kind")} == anchor]
        if len(matched) != 1:
            raise ValueError("FUSION_SEARCH_BINDING_INVALID")
        for term in (b["label"], anchor["quote"]):
            if term not in terms:
                terms.append(term)
    return tuple(terms)


def fusion_search_bindings(brief, entry, target):
    """Extract search anchors only from exact, untruncated rejected source units."""
    scope = entry.get(HYPOTHESIS_SCOPE_KEY)
    if (not isinstance(scope, dict) or scope.get("policy") != BRANCH_POLICY
            or scope.get("origin") != "observed_source_span"
            or scope.get("target_symbol") != target or "::" not in str(scope.get("label", ""))):
        return []
    declared = declared_labels(brief, target)
    if declared and scope["label"] not in declared:
        return []
    for anchor in scope.get("anchors", []):
        if not isinstance(anchor, dict):
            continue
        for record in entry.get("unique_external_records", []):
            provenance = record.get("provenance_classes", [])
            if (not provenance or not set(provenance).issubset({"external_live", "external_cache"})
                    or anchor.get("source") not in record.get("sources", [])
                    or anchor.get("external_id") not in record.get("external_ids", [])):
                continue
            for unit in record.get("evidence_units", []):
                if (unit.get("evidence_id") != anchor.get("evidence_id")
                        or unit.get("excerpt_hash") != anchor.get("excerpt_sha256")):
                    continue
                bound = {"policy": SEARCH_POLICY, "target_symbol": target, "label": scope["label"],
                    **{k:anchor.get(k) for k in ("source", "external_id", "evidence_id", "excerpt_sha256")},
                    "provenance_class": sorted(provenance)[0], "excerpt": unit.get("bounded_excerpt"),
                    "anchor": {k:anchor.get(k) for k in ("start", "end", "quote", "kind")},
                    "authority": "search_scope_only_not_support_or_identity"}
                try:
                    validated_fusion_search_terms([bound], target)
                except (ValueError, TypeError):
                    continue
                # One exact source is enough to authenticate search vocabulary,
                # not enough to admit any scientific hypothesis.
                return [bound]
    return []


def functional_association_sentence(text, target):
    """A narrow positive functional relation, never requested-direction proof.

    The caller must separately verify primary publication, result context and
    disease/model linkage. Do not infer a positive result from a negated clause.
    """
    if re.search(r"\b(?:no|not|never|cannot|may|might|could|hypothesi[sz]ed|proposed)\b",text,re.I):
        return False  # Mixed/negated clauses remain for review, never positive by this selector.
    if not re.search(r"\b(?:cell(?:s| line)?|in vitro|in vivo|xenografts?|organoids?)\b",text,re.I):
        return False
    entities = [m["quote"] for m in mentions(text,target)] or [target]
    for entity in entities:
        pattern = (rf"(?<![A-Za-z0-9]){re.escape(entity)}(?![A-Za-z0-9])"
            r"(?:\s+(?:fusion|mutant))?(?:\s+protein)?\s+"
            r"(?:(?:has been|was|is)\s+(?:demonstrated|shown)\s+to\s+)?"
            r"(?:significantly\s+)?(?:transform(?:ed|s)?\b|"
            r"promot(?:e|ed|es)\s+(?:the\s+)?(?:proliferation|growth|tumori?genesis)\b)")
        if re.search(pattern,text,re.I):
            return True
    return False
