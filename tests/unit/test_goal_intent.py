from __future__ import annotations

import pytest

from drugpilot.domain.cell_context import classify_cell_context
from drugpilot.domain.goal_intent import cue_rejection_reason, literal_matches
from drugpilot.domain.models import ResearchBrief, ProjectConstraints, HypothesisTrack
from drugpilot.orchestration.goal_focus import derive_goal_focus, effective_discovery_priority
from drugpilot.tools.context_router import (
    ContextRoutingRequest, ContextRoutingBudget, ContextTarget, DiseaseTargetContextRouter,
)
from drugpilot.tools.public_context import MA1ContextResult, ContextSourceResult
from drugpilot.tools.pharos_on_demand import requested_pharos_terms, OnDemandPharosRouter


def brief(text, **kwargs):
    return ResearchBrief(brief_id="goal-intent-test", research_goal=text,
                         disease="test disease", **kwargs)


@pytest.mark.parametrize("text,category", [
    ("Do not use single-cell analysis.", "single_cell"),
    ("Single-cell analysis is not required.", "single_cell"),
    ("Avoid gene expression profiling.", "expression"),
    ("No interaction network analysis is needed.", "network"),
    ("Skip Pharos for this request.", "pharos"),
    ("Drug repurposing is not required.", "repurposing"),
    ("Do not prioritize novel targets.", "novel_target"),
    ("단일세포 분석은 하지 말 것.", "single_cell"),
    ("유전자 발현 분석은 필요 없음.", "expression"),
    ("상호작용 네트워크는 제외한다.", "network"),
    ("약물 재창출은 하지 않는다.", "repurposing"),
    ("신규 표적은 검토하지 말 것.", "novel_target"),
    ("Should we use single-cell analysis?", "single_cell"),
    ("If needed, use single-cell analysis.", "single_cell"),
])
def test_nonaffirmative_cue_is_recorded_not_admitted(text, category):
    b = brief(text)
    original = b.model_dump(mode="json")
    focus = derive_goal_focus(b)
    assert not any(c.category == category for c in focus.cues)
    assert any(c.category == category for c in focus.suppressed_cues)
    assert b.model_dump(mode="json") == original


@pytest.mark.parametrize("text", [
    "Do not restrict targets to FLT3.",
    "Never use executable target FLT3 only.",
    "Target FLT3 only is not required.",
    "Should we restrict targets to FLT3?",
    "If necessary, restrict targets to FLT3.",
    "표적을 FLT3로 제한하지 말 것.",
    "실행 표적 FLT3만 사용하지 말 것.",
])
def test_negative_or_question_restriction_does_not_close_portfolio(text):
    focus = derive_goal_focus(brief(text))
    assert focus.target_scope_mode == "portfolio"
    assert focus.executable_target_mentions == []
    assert focus.explicit_target_mentions == []
    assert any(c.category == "explicit_target" for c in focus.suppressed_cues)


@pytest.mark.parametrize("text,strategy", [
    ("Do not use drug repurposing, but prioritize novel targets.", "novel_target"),
    ("약물 재창출은 하지 말고 신규 표적을 검토한다.", "novel_target"),
    ("신규 표적이 아니라 약물 재창출을 검토한다.", "repurposing"),
    ("Avoid novel targets and use drug repurposing.", "repurposing"),
    ("Drug repurposing is unnecessary. Prioritize novel targets.", "novel_target"),
])
def test_contrast_clause_preserves_affirmative_strategy(text, strategy):
    f = derive_goal_focus(brief(text))
    assert f.candidate_strategy_focus == strategy


@pytest.mark.parametrize("text", [
    "Use not only single-cell analysis but also gene expression profiling.",
    "Use single-cell analysis without fabricating any evidence.",
    "Use single-cell analysis; do not fabricate evidence.",
])
def test_negated_unrelated_action_does_not_remove_positive_cue(text):
    assert "single_cell" in derive_goal_focus(brief(text)).source_families


def test_separate_priority_items_do_not_share_negation_scope():
    b = brief("Evaluate disease-relevant mechanisms.", human_priorities=[
        "Do not use gene expression profiling", "Use interaction network analysis"])
    f = derive_goal_focus(b)
    assert "expression" not in f.source_families
    assert "network" in f.source_families


def test_structured_priority_and_safety_are_not_disabled_by_free_text():
    b = brief("Do not use drug repurposing. Skip safety and ADMET checks.",
        constraints=ProjectConstraints(discovery_priority="repurposing_target",
            chronic_safety_required=True, druggability_required=True))
    assert effective_discovery_priority(b, derive_goal_focus(b)) == (
        "repurposing_target", "research_brief_structured_constraint")
    assert b.constraints.chronic_safety_required
    assert b.constraints.druggability_required


@pytest.mark.parametrize("context", [
    "cells", "cell models", "disease-relevant cell models", "CML cell models",
    "experimental models", "primary cells", "세포 모델", "일반 세포", "미정", "unspecified", "unknown identifier",
])
def test_generic_or_unknown_context_is_not_cell_atlas_request(context):
    assert classify_cell_context(context) == "none"
    f = derive_goal_focus(brief("Evaluate disease-relevant mechanisms.",
        constraints=ProjectConstraints(target_cell_type=context)))
    assert "single_cell" not in f.source_families
    assert any(c.reason == "NO_AFFIRMATIVE_SPECIFIC_CELL_CONTEXT" for c in f.suppressed_cues)


@pytest.mark.parametrize("context,expected", [
    ("synovial fibroblasts", "cell_resolved"), ("AML blasts", "cell_resolved"),
    ("CD8 T cells", "cell_resolved"), ("AML cell lines", "cell_line"),
    ("patient-derived organoids", "cell_line"), ("single-cell RNA-seq", "single_cell"),
    ("단일세포 분석", "single_cell"), ("배양 세포", "cell_line"),
])
def test_specific_positive_cell_context_preserved(context, expected):
    assert classify_cell_context(context) == expected


def test_negative_single_cell_priority_does_not_reclassify_cell_line():
    assert classify_cell_context("AML cell lines", ["Do not use single-cell analysis"]) == "cell_line"
    assert classify_cell_context(None, ["single-cell analysis is not required"]) == "none"
    assert classify_cell_context(None, [
        "OCI-AML3 비교는 cell-line selectivity만 평가하며 FLT3-ITD 인과성을 주장하지 않고 후속 검증을 요구한다."]
    ) == "cell_line"


@pytest.mark.parametrize("text", [
    "Single-cell analysis and gene expression profiling are not required.",
    "단일세포 및 유전자 발현 분석은 하지 말 것.",
])
def test_shared_negative_predicate_applies_to_coordinated_cues(text):
    f = derive_goal_focus(brief(text))
    assert "single_cell" not in f.source_families
    assert "expression" not in f.source_families


def test_cell_priority_is_not_attributed_to_generic_target_field():
    f = derive_goal_focus(brief("Evaluate disease-relevant mechanisms.",
        constraints=ProjectConstraints(target_cell_type="cell models"),
        human_priorities=["Use single-cell analysis"]))
    assert "single_cell" in f.source_families
    assert not any(c.source_field == "target_cell_type" for c in f.cues)
    assert any(c.source_field == "human_priorities" for c in f.cues)


def test_negative_pharos_preference_does_not_disable_structured_requirement():
    from drugpilot.domain.models import HypothesisPacket
    b = brief("Evaluate supported mechanisms.", human_priorities=["Do not use Pharos"],
              constraints=ProjectConstraints(druggability_required=True))
    result = OnDemandPharosRouter(None).route(HypothesisPacket(brief_id=b.brief_id, hypotheses=[]), b)
    assert result.decision["requested"] is True
    assert "ResearchBrief.constraints.druggability_required" in result.decision["request_fields"]


class RecordingRetriever:
    def __init__(self):
        self.sources = []

    def lookup(self, request):
        source = request.sources[0]
        self.sources.append(source)
        return MA1ContextResult(lookup=request, evidence=[], sources=[
            ContextSourceResult(source=source, status="empty", queried=True, record_count=0)])


@pytest.mark.parametrize("priority", [
    "Do not use single-cell analysis, gene expression profiling or interaction network analysis.",
    "single-cell analysis is not required; gene expression is not required; interaction network is not required.",
    "단일세포 분석은 하지 말 것; 유전자 발현 분석은 하지 말 것; 상호작용 네트워크는 제외한다.",
])
def test_raw_human_priority_cannot_bypass_intent_in_actual_router(priority):
    r = RecordingRetriever()
    request = ContextRoutingRequest(disease="test disease", targets=[
        ContextTarget(symbol="TP53", track=HypothesisTrack.LITERATURE)],
        human_priorities=[priority], allowed_sources=[
            "Human Cell Atlas", "CELLxGENE Census", "ARCHS4", "BioGRID"],
        budget=ContextRoutingBudget(max_source_calls=10))
    result = DiseaseTargetContextRouter(r).route(request)
    assert r.sources == []
    assert result.source_calls_used == 0


def test_positive_network_after_negative_single_cell_routes_only_network():
    r = RecordingRetriever()
    request = ContextRoutingRequest(disease="test disease", targets=[
        ContextTarget(symbol="TP53", track=HypothesisTrack.LITERATURE)],
        human_priorities=["Do not use single-cell analysis, but use interaction network analysis."],
        allowed_sources=["Human Cell Atlas", "CELLxGENE Census", "ARCHS4", "BioGRID"],
        budget=ContextRoutingBudget(max_source_calls=10))
    DiseaseTargetContextRouter(r).route(request)
    assert r.sources == ["BioGRID"]


def test_optional_veto_overrides_track_default_and_implicit_cell_atlas_route():
    r = RecordingRetriever()
    request = ContextRoutingRequest(disease="test disease", targets=[
        ContextTarget(symbol="TP53", track=HypothesisTrack.TBIO)], target_cell_type="AML blasts",
        human_priorities=["Do not use single-cell analysis, expression profiling or interaction network analysis."],
        allowed_sources=["Human Cell Atlas", "CELLxGENE Census", "ARCHS4", "BioGRID"],
        budget=ContextRoutingBudget(max_source_calls=10))
    result = DiseaseTargetContextRouter(r).route(request)
    assert r.sources == []
    outcomes = [o for target in result.targets for o in target.sources if o.source in request.allowed_sources]
    assert all(o.status == "skipped_not_applicable" and "negative goal cue" in o.detail for o in outcomes)


@pytest.mark.parametrize("text", ["Do not use Pharos", "Pharos is not required", "Pharos는 사용하지 말 것"])
def test_pharos_router_trigger_honors_negation(text):
    assert requested_pharos_terms([text]) == ()


def test_stem_matches_cover_full_word_for_suffix_negation():
    text = "Drug repurposing is not required."
    matches = list(literal_matches(text, ["repurpos"]))
    assert len(matches) == 1 and matches[0].group() == "repurposing"
    assert cue_rejection_reason(text, matches[0].start(), matches[0].end()) == "NEGATED_GOAL_CUE"


def test_fingerprint_includes_all_intent_consumers():
    import inspect
    from drugpilot.orchestration.human_review import source_digest
    source = inspect.getsource(source_digest)
    for name in ["domain/goal_intent.py", "domain/cell_context.py", "orchestration/goal_focus.py",
                 "tools/context_router.py", "tools/pharos_on_demand.py"]:
        assert name in source
