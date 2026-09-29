"""Exact adjacent-sentence result units; no paraphrases or inferred experiments.

V1 handles only explicit inhibition repeated in two adjacent sentences with an
unambiguous same-model reference. It does not resolve distant anaphora, mixed
constructs, different models, unnamed drugs or clinical generalization. Every
unit retains its original parent and requires semantic agreement and G0 audit.
"""
from copy import deepcopy
from hashlib import sha256
import re

from drugpilot.agents.evidence_semantics import (
    ABSTRACT_CONTEXT_KEY, KEY as SEMANTIC_KEY, digest, interpretation_text,
    permits, source_binding,
)
from drugpilot.agents.molecular_scope import KEY as SCOPE_KEY, mentions

POLICY = "source_bound_adjacent_result_unit_v1"
KEY = "document_result_unit"
MAX_UNIT_CHARS = 1200
# One derived result per publication is input compression, not independent
# corroboration. Other originals remain present and any omitted units are counted.
MAX_UNITS_PER_PUBLICATION = 1
_MODELS = r"(?:cell lines|cells|organoids|xenografts)"
_MODEL_REF = re.compile(rf"\bin\s+(?:these|the same)\s+(?P<model>{_MODELS})\b", re.I)
_ORIGINAL_WORK = re.compile(r"\b(?:we\s+(?:assessed|evaluated|examined|tested|investigated|studied)|this study\s+(?:assessed|evaluated|examined|tested|investigated))\b", re.I)
_UNSAFE = re.compile(r"\b(?:not|no|never|neither|without|failed|may|might|could|will|would|hypothes\w*|propos\w*|predict\w*|simulat\w*|resistan\w*|however|whereas|combination|combined|co[- ]treat\w*|synerg\w*)\b", re.I)


def _sentences(text):
    # Do not claim to solve arbitrary scientific sentence segmentation. Spans
    # include the original punctuation, and every returned unit is contiguous.
    for m in re.finditer(r"[^.!?]+[.!?]+|[^.!?]+$", text):
        start = m.start() + len(m.group()) - len(m.group().lstrip())
        end = m.end() - (len(m.group()) - len(m.group().rstrip()))
        if start < end:
            yield {"start":start, "end":end, "text":text[start:end]}


def candidate_units(brief, target, text):
    """Return source-span contracts, not biological verdicts."""
    from drugpilot.agents.ma1 import (
        _scope_direct_intervention_present, _has_admissible_direction_endpoint,
        _MEASURED_ENDPOINT_PATTERN, _EXPERIMENTAL_RESULT_PATTERN,
        _NON_DIRECTION_PRIMARY_PATTERN, _direction_endpoint_polarity_status,
    )
    from drugpilot.tools.public_biology import _direct_support_rejection_codes
    # This source-span grammar specifically proves inhibition. Broad intake can
    # retain that proof; it cannot turn the same unit into activation evidence.
    if brief.constraints.intervention_direction not in {"inhibit", "modulate", "unspecified"} or not isinstance(text,str):
        return []
    disease = " ".join(brief.disease.split())
    # No memory-based disease aliases; only an exact user disease phrase and
    # optional source-declared abbreviation directly preceding the model noun.
    model_pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(disease)}\s*(?:\([A-Z0-9-]{{2,10}}\)\s*)?(?P<model>{_MODELS})\b",re.I)
    sentences, results = list(_sentences(text)), []
    for first, second in zip(sentences,sentences[1:]):
        unit = text[first["start"]:second["end"]]
        anchor, result = first["text"], second["text"]
        if len(unit) > MAX_UNIT_CHARS or "?" in unit or _UNSAFE.search(unit) or _NON_DIRECTION_PRIMARY_PATTERN.search(unit):
            continue
        models, refs = list(model_pattern.finditer(anchor)), list(_MODEL_REF.finditer(result))
        if (len(models) != 1 or len(refs) != 1 or not _ORIGINAL_WORK.search(anchor)
                or models[0]["model"].casefold() != refs[0]["model"].casefold()
                or len(re.findall(rf"\b{_MODELS}\b",anchor,re.I)) != 1
                or len(re.findall(rf"\b{_MODELS}\b",result,re.I)) != 1):
            continue
        scopes = [{m["label"] for m in mentions(s,target)} for s in (anchor,result)]
        if scopes[0] != scopes[1] or len(scopes[0]) > 1:
            continue
        label = next(iter(scopes[0]),None)
        if not all(_scope_direct_intervention_present(s,target,"inhibit",label) for s in (anchor,result)):
            continue
        if (_direct_support_rejection_codes(unit,target)
                or not _has_admissible_direction_endpoint(result,target)
                or not _MEASURED_ENDPOINT_PATTERN.search(result)
                or not _EXPERIMENTAL_RESULT_PATTERN.search(result)
                or _direction_endpoint_polarity_status(result,"inhibit") != "favorable"):
            continue
        # Named alternatives in the result invalidate "these cells" linkage;
        # only the closed same-model prepositional phrase may introduce a model.
        if any(m.start() != refs[0].start() for m in re.finditer(r"\bin\s+",result,re.I)):
            continue
        results.append({"policy":POLICY,"target_symbol":target,"scope_label":label,
            "disease":brief.disease,"direction":"inhibit","model_noun":models[0]["model"].casefold(),
            "anchor":first,"result":second,"start":first["start"],"end":second["end"],
            "text_sha256":sha256(unit.encode()).hexdigest(),
            "link":"adjacent_repeated_entity_and_inhibition_explicit_same_model",
            "authority":"source_span_unit_requires_semantic_primary_and_gate_checks"})
    return results


def _source_text(e):
    context = e.metadata.get(ABSTRACT_CONTEXT_KEY)
    text = interpretation_text(e)
    if (not isinstance(context,dict) or context.get("status") != "complete"
            or text != context.get("text")
            or e.metadata.get("content_availability") != "abstract_excerpt"
            or e.metadata.get("full_text_stored") is not False):
        return None
    return text


def derive_result_units(brief,target,evidence):
    """Keep every original verbatim; append separately identified source slices."""
    from drugpilot.agents.ma1 import _has_verified_primary_publication_type, _PRIMARY_QUERY_POLICIES
    originals, result = list(evidence), []
    seen = {(e.source,e.external_id,e.excerpt_hash) for e in originals if KEY in e.metadata}
    for e in originals:
        result.append(e)
        if (KEY in e.metadata or not _has_verified_primary_publication_type(e)
                or e.metadata.get("query_strategy") not in _PRIMARY_QUERY_POLICIES
                or e.metadata.get("query_intervention_direction") != brief.constraints.intervention_direction):
            continue
        text = _source_text(e)
        if text is None:
            continue
        candidates = candidate_units(brief,target,text)
        if not candidates:
            continue
        for candidate in candidates[:MAX_UNITS_PER_PUBLICATION]:
            excerpt = text[candidate["start"]:candidate["end"]]
            unique = (e.source,e.external_id,candidate["text_sha256"])
            if unique in seen:
                continue
            seen.add(unique)
            binding = {"unit":candidate,"parent_evidence_id":e.evidence_id,
                "parent_excerpt_sha256":e.excerpt_hash,"parent_source_sha256":source_binding(e),
                "abstract_sha256":e.metadata["content_hash"],
                "candidate_unit_count":len(candidates),"omitted_unit_count":max(0,len(candidates)-1)}
            # Copy only document/retrieval provenance. The parent's previous
            # excerpt-specific role/rejection/claim fields remain on the parent.
            keys = ("pmid","pmcid","doi","publication_types","provenance_class","query",
                "query_strategy","query_intervention_direction","retrieval_endpoint","content_hash",
                "content_hash_algorithm","content_hash_scope","full_text_stored",ABSTRACT_CONTEXT_KEY)
            metadata = {k:deepcopy(e.metadata[k]) for k in keys if k in e.metadata}
            metadata.update({KEY:binding,"excerpt":excerpt,"excerpt_char_count":len(excerpt),
                "excerpt_max_chars":MAX_UNIT_CHARS,"excerpt_truncated":False,
                "excerpt_selection_policy":POLICY,"content_availability":"abstract_excerpt",
                "source_location":f"abstract-unit:chars:{candidate['start']}-{candidate['end']}",
                "role_specific_record":"direct_direction_adjacent_result_unit",
                "duplicate_document_role":"same_publication_not_independent_corroboration",
                "content_storage_policy":"exact_contiguous_source_unit_with_complete_abstract_context",
                "abstract_context_stored":True,"additional_network_call":False,
                "direct_support_eligible":True,"direct_support_rejection_codes":[],
                "claim_linkage_status":"pending_ma1_claim_reconciliation"})
            result.append(e.model_copy(update={"evidence_id":"ev_du_"+digest(binding)[:20],
                "supports":"support","excerpt_hash":candidate["text_sha256"],"metadata":metadata},deep=True))
    return result


def validated_unit_excerpt(brief,target,e,scope_label):
    from drugpilot.agents.ma1 import _has_verified_primary_publication_type, _PRIMARY_QUERY_POLICIES, _source_supports
    binding = e.metadata.get(KEY)
    if (not isinstance(binding,dict) or SEMANTIC_KEY not in e.metadata
            or not permits(e,brief,target,"direction",scope_label)
            or not _has_verified_primary_publication_type(e) or _source_supports(e) != "support"
            or e.metadata.get("query_strategy") not in _PRIMARY_QUERY_POLICIES
            or e.metadata.get("query_intervention_direction") != brief.constraints.intervention_direction
            or e.metadata.get("counter_category") or e.metadata.get("review_status")):
        return None
    text = _source_text(e)
    if text is None:
        return None
    candidates = candidate_units(brief,target,text)
    unit = binding.get("unit")
    if (not isinstance(unit,dict) or unit not in candidates[:MAX_UNITS_PER_PUBLICATION]
            or unit["scope_label"] != scope_label or binding.get("abstract_sha256") != e.metadata["content_hash"]
            or binding.get("candidate_unit_count") != len(candidates)
            or binding.get("omitted_unit_count") != max(0,len(candidates)-1)
            or e.metadata.get("excerpt_selection_policy") != POLICY
            or e.metadata.get("source_location") != f"abstract-unit:chars:{unit['start']}-{unit['end']}"
            or e.metadata.get("excerpt") != text[unit["start"]:unit["end"]]
            or e.excerpt_hash != unit["text_sha256"]):
        return None
    # Both independent interpretations must quote THIS result unit, not a
    # different favorable experiment elsewhere in the complete abstract.
    annotation = e.metadata[SEMANTIC_KEY]
    for key in ("extractor","verifier"):
        meaning = annotation.get(key) or {}
        if (not meaning.get("result_quote") or meaning["result_quote"] not in unit["result"]["text"]
                or not meaning.get("model_quote") or meaning["model_quote"] not in unit["anchor"]["text"]
                or not meaning.get("intervention_quote") or meaning["intervention_quote"] not in e.metadata["excerpt"]
                or not meaning.get("target_quote") or meaning["target_quote"] not in e.metadata["excerpt"]):
            return None
    return e.metadata["excerpt"]


def parent_is_bound(e,evidence):
    binding = e.metadata.get(KEY)
    if not isinstance(binding,dict):
        return False
    parents = [p for p in evidence if p.evidence_id == binding.get("parent_evidence_id")]
    if len(parents) != 1:
        return False
    p = parents[0]
    return (KEY not in p.metadata and p.source == e.source and p.external_id == e.external_id
        and p.excerpt_hash == binding.get("parent_excerpt_sha256")
        and source_binding(p) == binding.get("parent_source_sha256")
        and p.metadata.get("content_hash") == binding.get("abstract_sha256")
        and _source_text(p) is not None)
