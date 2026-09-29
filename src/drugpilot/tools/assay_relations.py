"""Exact abstract relations, not chemical identity or validated bioactivity.

Only explicit local subject/action/target/endpoint statements are accepted.
Unsupported prose stays unresolved. No nearest-number assignment, cross-sentence
coreference, numeric conversion or database identity inference is performed.
"""
from hashlib import sha256
import re

POLICY = "source_bound_assay_text_relation_v1"
_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_SEPARATOR = r"(?:\s*(?:±|\+/-|/|,|–|—|-)\s*|\s+(?:and|to)\s+)"
_QUANTITY = re.compile(
    rf"(?<![\w.])(?:(?:<=|>=|<|>|≤|≥|~|≈)\s*|(?:about|approximately|less than|more than)\s+)?"
    rf"(?:[-−]\s*)?{_NUMBER}(?:{_SEPARATOR}{_NUMBER})*\s*(?:pM|nM|uM|µM|μM|mM)(?![\w])")
_SENTENCES = re.compile(r"(?:\d+\.\d+|(?<![\w.])\.\d+|[pP]\.[A-Za-z][0-9]{1,5}[A-Za-z*](?![A-Za-z0-9])|[^.!?])+(?:[.!?]+|$)")
_ENDPOINT = re.compile(r"(?<![A-Za-z0-9])(?:(?i:IC50|EC50)|Ki|Kd)(?![A-Za-z0-9])")
_UNSAFE = re.compile(r"\b(?:not|no|neither|without|failed|failure|hypothetical|predicted|prediction|"
    r"docking|simulation|simulated|estimated|estimate|may|might|could|would|proposed|"
    r"synergy|synergistic|synergistically|combination|combined|respectively|than|compared|"
    r"equal|sixfold|twofold|threefold)\b", re.I)
_CELL = re.compile(r"\b(?:cell|cells|cellular|proliferation|cytotoxic|cytotoxicity|growth)\b", re.I)


def quantities(text, offset=0):
    """Preserve whole values, uncertainty, bounds and shared-unit lists."""
    if not isinstance(text, str):
        return []
    return [{"quote":m.group(),"start":offset+m.start(),"end":offset+m.end()}
        for m in _QUANTITY.finditer(text)]


def sentence_spans(text):
    """Assay-only sentence tokenizer; leading decimals are not full stops."""
    for match in _SENTENCES.finditer(text):
        start=match.start()+len(match.group())-len(match.group().lstrip())
        end=match.end()-(len(match.group())-len(match.group().rstrip()))
        if end>start:
            yield start,end,text[start:end]


def quantity_shape(quote):
    """Lexical representation only. ± is not assumed to be SD/SEM/CI."""
    if not isinstance(quote, str) or not _QUANTITY.fullmatch(quote):
        return None
    unit = re.search(r"(?:pM|nM|uM|µM|μM|mM)$", quote).group()
    body = quote[:-len(unit)].strip()
    values = [m.group() for m in re.finditer(_NUMBER, body)]
    if re.match(r"(?:[<>=≤≥~≈]+\s*)?[-−]",body):
        representation = "invalid_negative_concentration"
    elif "±" in body or "+/-" in body:
        representation = "value_with_unspecified_uncertainty"
    elif len(values) > 1:
        representation = "shared_unit_expression_unassigned"
    elif re.match(r"(?:<=|>=|<|>|≤|≥|less than|more than)", body):
        representation = "bound_not_exact_value"
    elif re.match(r"(?:~|≈|about|approximately)", body):
        representation = "approximate_value"
    else:
        representation = "reported_scalar"
    return {"quote":quote,"unit_quote":unit,"number_quotes":values,
        "representation":representation,"unit_conversion_applied":False,
        "uncertainty_type_verified":False}


def text_relation(text, span, target, abstract_sha256):
    """Bind quotes to a hashed complete abstract; never grant gate authority."""
    result = {"policy":POLICY,"status":"unresolved","textual_relation_verified":False,
        "measurement_assignment_verified":False,"compound_identity_verified":False,
        "construct_identity_verified":False,"disease_model_match_verified":False,
        "assay_protocol_verified":False,"primary_support_eligible":False,
        "bioactivity_records_generated":0,"missing_values_imputed":False,
        "source_binding_verified":False,"reason_codes":[],"relation":None,
        "unresolved_fields":["compound_chemical_identity","target_construct_identity",
            "disease_model_match","assay_protocol_and_conditions","replicates_and_controls"],
        "authority":"explicit_source_wording_only_not_validated_target_potency"}
    if not isinstance(text,str) or not isinstance(span,dict) or not isinstance(target,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,30}",target):
        result["reason_codes"]=["ASSAY_SOURCE_BINDING_INVALID"]
        return result
    start,end = span.get("start"),span.get("end")
    excerpt = span.get("excerpt")
    if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(text)
            or not isinstance(excerpt,str) or len(excerpt)>1200 or text[start:end]!=excerpt
            or sha256(text.encode()).hexdigest()!=abstract_sha256
            or sha256(excerpt.encode()).hexdigest()!=span.get("excerpt_sha256")):
        result["reason_codes"]=["ASSAY_SOURCE_BINDING_INVALID"]
        return result
    if not any(a==start and b==end for a,b,_ in sentence_spans(text)):
        result["reason_codes"]=["ASSAY_COMPLETE_SENTENCE_REQUIRED"]
        return result
    endpoints=[{"quote":m.group(),"start":start+m.start(),"end":start+m.end()} for m in _ENDPOINT.finditer(excerpt)]
    values=quantities(excerpt,start)
    # Recompute anchors; caller-supplied selections cannot hide ambiguity.
    if not values or not endpoints or values!=span.get("quantity_mentions"):
        result["reason_codes"]=["ASSAY_SOURCE_ANCHOR_MISMATCH"]
        return result
    result.update(source_binding_verified=True,
        source_binding={"abstract_sha256":abstract_sha256,"excerpt_sha256":span["excerpt_sha256"],
            "start":start,"end":end},quantity_expressions=[quantity_shape(q["quote"]) for q in values])
    reasons=[]
    if len(endpoints)!=1:
        reasons.append("MULTIPLE_ASSAY_ENDPOINTS_UNRESOLVED")
    if len(values)!=1 or any(quantity_shape(q["quote"])["representation"]=="shared_unit_expression_unassigned" for q in values):
        reasons.append("MULTIPLE_QUANTITIES_UNRESOLVED")
    if any(quantity_shape(q["quote"])["representation"]=="invalid_negative_concentration" for q in values):
        reasons.append("INVALID_REPORTED_CONCENTRATION")
    if _UNSAFE.search(excerpt):
        reasons.append("QUALIFIED_OR_COMPARATIVE_CONTEXT_UNRESOLVED")
    # Closed local grammar, not loose co-occurrence in a search hit.
    label=r"[A-Za-z0-9][A-Za-z0-9-]{0,39}"
    code=r"(?=[A-Z][A-Z0-9-]*\d)[A-Z][A-Z0-9-]{1,39}"
    subject=rf"(?P<compound>(?:(?:Compound|compound)\s+{label}|{code}))"
    subject=rf"(?:{subject}|The most active compound,\s*(?P<label>{label}),)"
    symbol=re.escape(target)
    target_expression=rf"(?P<target>(?:[A-Za-z][A-Za-z0-9 -]{{1,70}}\s+\({symbol}\)|{symbol})(?:\s+kinase)?)"
    verb=r"(?:had (?:in vitro )?inhibitory activity against|(?:potently )?inhibited|showed (?:(?:significant|strong|potent) )?(?:inhibition|activity) against)"
    pattern=rf"^{subject}\s+{verb}\s+{target_expression}\s*(?:\(\s*|with\s+(?:an?\s+)?)?(?P<endpoint>IC50|EC50|Ki|Kd)(?:\s+value)?\s*(?:of\s+|[=:]\s*)?(?P<quantity>{_QUANTITY.pattern})"
    match=re.match(pattern,excerpt)
    if match:
        tail=excerpt[match.end():]
        if not re.fullmatch(r"\)?(?:\.|, (?:indicating that|the kinase component of) [^!?]+\.)?",tail):
            reasons.append("UNRESOLVED_CLAUSE_AFTER_MEASUREMENT")
        if _CELL.search(excerpt[:match.end()]):
            reasons.append("CELLULAR_RESPONSE_NOT_TARGET_POTENCY")
        if len(values)==1 and (match.start("quantity")+start!=values[0]["start"] or match.end("quantity")+start!=values[0]["end"]):
            reasons.append("QUANTITY_EXPRESSION_PARTIAL_MATCH")
        if quantity_shape(match.group("quantity"))["representation"]=="value_with_unspecified_uncertainty":
            result["unresolved_fields"].append("reported_uncertainty_type")
    else:
        reasons.append("EXPLICIT_LOCAL_RELATION_NOT_ESTABLISHED")
        if _CELL.search(excerpt):
            reasons.append("CELLULAR_RESPONSE_NOT_TARGET_POTENCY")
    result["reason_codes"]=list(dict.fromkeys(reasons))
    if match and not reasons:
        def anchor(group):
            return {"quote":match.group(group),"start":start+match.start(group),"end":start+match.end(group)}
        # A coded subject could be a gene or another entity, not a chemical.
        # Only the literal word "compound" licenses the noun label, not identity.
        result["subject_kind"]="explicit_compound_label" if excerpt.startswith(("Compound ","compound ","The most active compound,")) else "untyped_source_subject"
        relation={"subject_label":anchor("compound" if match.group("compound") else "label"),
            "target_label":anchor("target"),"endpoint":anchor("endpoint"),
            "quantity":{**anchor("quantity"),**quantity_shape(match.group("quantity"))}}
        result.update(status="explicit_text_relation",textual_relation_verified=True,relation=relation)
    result["next_action"]="resolve_structured_assay_identity_and_conditions" if result["textual_relation_verified"] else "retrieve_richer_source_context_without_guessing_assignment"
    return result
