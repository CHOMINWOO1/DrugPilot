"""Strict, deterministic PlanSpec builder for the MA2 route-aware workflow."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

from drugpilot.config import Settings, get_settings
from drugpilot.domain.models import (
    CandidateRoute,
    HypothesisPacket,
    PlanSpec,
    PlanTask,
    ResearchBrief,
    ToolAssignment,
)
from drugpilot.orchestration.goal_focus import GoalFocus, effective_discovery_priority
from drugpilot.orchestration.tool_coverage import build_broad_source_coverage


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{sha256(payload.encode('utf-8')).hexdigest()[:12]}"


APPROVED_TOOLS: tuple[str, ...] = (
    "target_normalizer",
    "repurposing_search",
    "restricted_regulatory_registry",
    "de_novo_generator",
    "chemistry_validator",
    "docking",
    "admet_predictor",
    "candidate_judge",
)


class PlannerAgent:
    """Translate a G0-approved hypothesis packet into an auditable MA2 plan.

    The planner selects tasks only from a closed registry.  It does not execute
    tools, expand the budget, or silently discard prior revision instructions.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(
        self,
        hypothesis_packet: HypothesisPacket,
        brief: ResearchBrief,
        revision_context: object | None = None,
    ) -> PlanSpec:
        if hypothesis_packet.brief_id != brief.brief_id:
            raise ValueError("HypothesisPacket does not belong to the supplied ResearchBrief")
        if brief.constraints.forbidden_substructures:
            from drugpilot.tools.substructure_policy import validate_forbidden_substructures
            chemical_policy = validate_forbidden_substructures(brief.constraints.forbidden_substructures)
            if chemical_policy.status != "valid_policy":
                raise ValueError("Forbidden substructure policy cannot be executed: " + ", ".join(chemical_policy.reason_codes))
        if hypothesis_packet.provisional_hypotheses:
            raise ValueError("Planner cannot execute provisional hypotheses awaiting MA3")
        if hypothesis_packet.metadata.get("admission_policy") == "ma3-source-bound-v1":
            from drugpilot.agents.hypothesis_admission import admitted_role_is_linked
            if any(not admitted_role_is_linked(h, role, brief)
                   for h in hypothesis_packet.hypotheses
                   for role in ("direct_target_direction_primary", "primary_target_disease_association")):
                raise ValueError("Planner requires source-bound MA3 approval for every hypothesis")
        if not hypothesis_packet.hypotheses:
            raise ValueError("Planner requires at least one G0-approved hypothesis")
        if (brief.constraints.allowed_hypothesis_directions is not None
                and any(not brief.constraints.allows_hypothesis_direction(h.intervention_direction)
                        for h in hypothesis_packet.hypotheses)):
            raise ValueError("Planner cannot use a hypothesis outside allowed_hypothesis_directions")

        context = _normalize_revision_context(revision_context)
        previous = context.get("previous_plan")
        revision = _next_revision(previous, context)
        # The API has already applied ``settings.candidates_per_route`` when
        # the caller omits a value.  Once the requested count is captured in
        # the immutable ResearchBrief, the Planner must preserve it instead of
        # silently applying the process default as a second hard cap.  This
        # keeps the UI's normal five-candidate run and an explicit 20-candidate load test
        # semantically distinct and reproducible.
        candidates_per_route = brief.constraints.candidates_per_route
        total_budget = min(brief.budget.max_run_tokens, self.settings.max_run_tokens)
        if total_budget < 10_000:
            raise ValueError("The effective planner budget must be at least 10,000 tokens")

        raw_focus = hypothesis_packet.metadata.get("goal_focus")
        focus = (
            GoalFocus.model_validate(raw_focus)
            if isinstance(raw_focus, dict)
            else None
        )
        effective_priority, priority_derivation = (
            effective_discovery_priority(brief, focus)
            if focus is not None
            else (
                brief.constraints.discovery_priority,
                "research_brief_structured_constraint",
            )
        )
        target_symbols = sorted(
            {hypothesis.target.symbol for hypothesis in hypothesis_packet.hypotheses}
        )
        context_note = _context_note(context)
        tasks = self._tasks(
            packet=hypothesis_packet,
            target_symbols=target_symbols,
            candidates_per_route=candidates_per_route,
            total_budget=total_budget,
            brief=brief,
            context_note=context_note,
        )
        _validate_task_graph(tasks)
        _validate_closed_tool_registry(tasks)
        allocated_tokens = sum(task.max_tokens for task in tasks)
        if allocated_tokens > total_budget:
            raise ValueError("Planned task token allocation exceeds total_token_budget")

        plan_key = ",".join(task.task_id for task in tasks)
        source_coverage = build_broad_source_coverage(
            brief,
            hypothesis_packet,
            self.settings,
        )
        effective_depth = self.settings.evidence_budget_for(
            brief.research_depth
        ).depth
        return PlanSpec(
            plan_id=_stable_id(
                "plan", hypothesis_packet.packet_id, revision, plan_key, context_note
            ),
            hypothesis_packet_id=hypothesis_packet.packet_id,
            revision=revision,
            tasks=tasks,
            approved_tools=list(APPROVED_TOOLS),
            stop_conditions=self._stop_conditions(brief, context),
            candidates_per_route=candidates_per_route,
            total_token_budget=total_budget,
            research_depth=effective_depth,
            discovery_priority=effective_priority,
            requested_discovery_priority=brief.constraints.discovery_priority,
            discovery_priority_derivation=priority_derivation,
            forbidden_drugs=list(brief.constraints.forbidden_drugs),
            forbidden_drug_ids=list(brief.constraints.forbidden_drug_ids),
            forbidden_drug_classes=list(brief.constraints.forbidden_drug_classes),
            forbidden_substructures=list(brief.constraints.forbidden_substructures),
            reserved_control_compounds=[
                item.model_copy(deep=True)
                for item in brief.constraints.reserved_control_compounds
            ],
            source_coverage=source_coverage,
            created_at=brief.created_at,
        )

    async def arun(
        self,
        hypothesis_packet: HypothesisPacket,
        brief: ResearchBrief,
        revision_context: object | None = None,
    ) -> PlanSpec:
        return self.run(hypothesis_packet, brief, revision_context=revision_context)

    def _tasks(
        self,
        *,
        packet: HypothesisPacket,
        target_symbols: list[str],
        candidates_per_route: int,
        total_budget: int,
        brief: ResearchBrief,
        context_note: str,
    ) -> list[PlanTask]:
        prefix = packet.packet_id
        task_ids = {
            name: _stable_id("task", prefix, name)
            for name in (
                "normalize_targets",
                "route_selection",
                "repurposing_discovery",
                "de_novo_design",
                "chemistry_validation",
                "docking",
                "admet",
                "candidate_ranking",
            )
        }
        allocations = _allocate_tokens(total_budget, (8, 6, 17, 21, 9, 13, 12, 14))
        target_text = ", ".join(target_symbols)
        constraints = brief.constraints
        direction_criterion = f"Intervention direction equals '{constraints.intervention_direction}'"
        if constraints.intervention_direction in {"modulate", "unspecified"}:
            resolved = sorted({
                f"{h.target.symbol}: {h.intervention_direction}"
                for h in packet.hypotheses if h.target.symbol in target_symbols
            })
            direction_criterion = (
                "Every candidate's intervention direction equals its linked approved hypothesis direction; "
                "resolved hypotheses: " + "; ".join(resolved)
            )
        common_criteria = [
            f"Every candidate references one of the approved targets: {target_text}",
            direction_criterion,
            "Every computed value retains its source tool call and version",
        ]
        if constraints.allowed_hypothesis_directions is not None:
            common_criteria.append(
                "Retained hypothesis directions must stay within the user's selected set: "
                + ", ".join(constraints.allowed_hypothesis_directions)
                + "; broad retrieval does not override this selection"
            )
        if context_note:
            common_criteria.append(f"Revision directive retained: {context_note}")

        return [
            PlanTask(
                task_id=task_ids["normalize_targets"],
                name="Normalize and resolve approved targets",
                model_name=self.settings.terra_model,
                tools=[
                    ToolAssignment(
                        tool_name="target_normalizer",
                        purpose="Normalize target symbols and stable protein identifiers",
                        allowed_operations=["normalize_target"],
                    )
                ],
                acceptance_criteria=[
                    "All selected hypotheses resolve to a canonical target symbol",
                    "Unresolved or conflicting identifiers stop dependent route tasks",
                ],
                max_tokens=allocations[0],
            ),
            PlanTask(
                task_id=task_ids["route_selection"],
                name="Select one evidence-appropriate candidate route",
                depends_on=[task_ids["normalize_targets"]],
                model_name=self.settings.terra_model,
                tools=[
                    ToolAssignment(
                        tool_name="candidate_judge",
                        purpose=(
                            "Select repurposing for well-characterized/drug-evidenced "
                            "targets or de novo for evidence-supported underexplored targets"
                        ),
                        allowed_operations=["assign_route_from_ma1_evidence"],
                    )
                ],
                acceptance_criteria=[
                    "Exactly one route is selected for the primary hypothesis",
                    "The decision cites persisted MA1 target/evidence factors and fixed thresholds",
                    "Missing or ambiguous metadata triggers a named conservative fallback",
                ],
                max_tokens=allocations[1],
            ),
            PlanTask(
                task_id=task_ids["repurposing_discovery"],
                name="Retrieve and score repurposing candidates",
                route=CandidateRoute.REPURPOSING,
                depends_on=[task_ids["route_selection"]],
                model_name=self.settings.terra_model,
                tools=[
                    ToolAssignment(
                        tool_name="repurposing_search",
                        purpose="Retrieve known compounds, activities, mechanism, and provenance",
                        allowed_operations=[
                            "retrieve_candidates",
                            "restore_candidate_structure",
                            "restore_activity_units",
                        ],
                    ),
                    ToolAssignment(
                        tool_name="restricted_regulatory_registry",
                        purpose=(
                            "Resolve repurposing-candidate identity, regulatory status, "
                            "and disease-specific contraindication evidence before "
                            "chemistry, docking, or ADMET computation"
                        ),
                        allowed_operations=[
                            "precompute_identity_regulatory_contraindication"
                        ],
                    ),
                ],
                acceptance_criteria=common_criteria
                + [
                    f"Return no more than {candidates_per_route} repurposing candidates",
                    "IC50, Ki, and Kd preserve relation, value, unit, assay type, and source",
                    "Approval or clinical status is not inferred when absent",
                    "Configured restricted-source failures remain unresolved and stop candidate computation",
                ],
                max_tokens=allocations[2],
            ),
            PlanTask(
                task_id=task_ids["de_novo_design"],
                name="Generate constrained de novo candidates",
                route=CandidateRoute.DE_NOVO,
                depends_on=[task_ids["route_selection"]],
                model_name=self.settings.terra_model,
                tools=[
                    ToolAssignment(
                        tool_name="de_novo_generator",
                        purpose="Generate novel molecules under target and structure constraints",
                        allowed_operations=[
                            "generate_candidates",
                            "refine_candidate",
                            "restore_candidate_structure",
                        ],
                    )
                ],
                acceptance_criteria=common_criteria
                + [
                    f"Return no more than {candidates_per_route} de novo candidates",
                    "Forbidden substructures are excluded before downstream calculation",
                    "Generated structures retain parent/scaffold provenance when applicable",
                ],
                max_tokens=allocations[3],
            ),
            PlanTask(
                task_id=task_ids["chemistry_validation"],
                name="Validate and standardize molecular structures",
                depends_on=[
                    task_ids["repurposing_discovery"],
                    task_ids["de_novo_design"],
                ],
                dependency_policy="any",
                tools=[
                    ToolAssignment(
                        tool_name="chemistry_validator",
                        purpose="Canonicalize SMILES, detect invalid chemistry, and deduplicate",
                        allowed_operations=["validate_and_describe"],
                    )
                ],
                acceptance_criteria=[
                    "Exactly one selected route predecessor is sufficient; the unselected route is not a failed mandatory dependency",
                    "Every retained candidate has a valid canonical SMILES",
                    "Duplicates are resolved without merging evidence provenance",
                    "Forbidden substructures are reported as explicit exclusions",
                ],
                max_tokens=allocations[4],
            ),
            PlanTask(
                task_id=task_ids["docking"],
                name="Run selective structural compatibility checks",
                depends_on=[task_ids["chemistry_validation"]],
                tools=[
                    ToolAssignment(
                        tool_name="docking",
                        purpose="Dock only candidates lacking sufficient direct activity evidence",
                        required=False,
                        allowed_operations=[
                            "score_binding_pose",
                            "rescore_binding_pose",
                        ],
                    )
                ],
                acceptance_criteria=[
                    "Docking is an orthogonal compatibility check and never replaces direct activity evidence",
                    "Structure source and confidence are recorded with every docking score",
                    "Docking score is treated as supporting evidence, not proof of efficacy",
                ],
                max_tokens=allocations[5],
            ),
            PlanTask(
                task_id=task_ids["admet"],
                name="Estimate safety and developability",
                depends_on=[task_ids["chemistry_validation"]],
                tools=[
                    ToolAssignment(
                        tool_name="admet_predictor",
                        purpose="Calculate physicochemical and ADMET risk indicators",
                        allowed_operations=[
                            "predict_six_endpoint_panel",
                            "repeat_six_endpoint_panel",
                        ],
                    )
                ],
                acceptance_criteria=[
                    "All configured safety endpoints return a value or an explicit missing reason",
                    "Prediction applicability/OOD is recorded",
                    "Chronic-safety risks are surfaced when chronic safety is required",
                ],
                max_tokens=allocations[6],
            ),
            PlanTask(
                task_id=task_ids["candidate_ranking"],
                name="Rank policy-selected candidate portfolio",
                depends_on=[task_ids["docking"], task_ids["admet"]],
                model_name=self.settings.terra_model,
                tools=[
                    ToolAssignment(
                        tool_name="candidate_judge",
                        purpose="Apply the declared multi-criteria decision policy per route",
                        allowed_operations=[
                            "screen_reserved_assay_control",
                            "screen_paper_native_v5_primary_source",
                            "screen_paper_native_v6_primary_source",
                            "screen_forbidden_drug",
                            "screen_forbidden_substructure",
                            "rescreen_reserved_assay_control",
                            "rescreen_forbidden_drug",
                            "rescreen_forbidden_substructure",
                            "validate_persisted_candidate_identity",
                            "deterministic_multi_objective_ranking",
                            "repair_plan_trace",
                        ],
                    )
                ],
                acceptance_criteria=[
                    "Only candidates from the policy-selected route are eligible for ranking",
                    "Reserved assay controls remain control-only, excluded, and unranked",
                    "Forbidden or withdrawn candidates remain unranked exclusions",
                    "Priority, conditional, and excluded status include traceable reasons",
                    "Candidate packet contains no tool call outside the approved registry",
                    "Every material claim is linked to evidence or a recorded calculation",
                ],
                max_tokens=allocations[7],
            ),
        ]

    def _stop_conditions(
        self,
        brief: ResearchBrief,
        context: Mapping[str, Any],
    ) -> list[str]:
        conditions = [
            f"Stop before exceeding {min(brief.budget.max_run_tokens, self.settings.max_run_tokens)} total run tokens",
            f"Stop before exceeding {min(brief.budget.max_llm_calls, self.settings.max_llm_calls)} LLM calls",
            f"Stop before exceeding USD {min(brief.budget.max_run_cost_usd, self.settings.max_run_cost_usd):.2f}",
            "Stop a branch after two consecutive tool failures for the same operation",
            "Stop and ABSTAIN when mandatory evidence provenance cannot be established",
            "Stop and BLOCK on policy violation, corrupted artifact, or unapproved tool use",
            "Do not advance to the Human Gate without a G1 PASS verdict",
        ]
        required_assays = ", ".join(brief.constraints.required_assays)
        if required_assays:
            conditions.append(
                f"Mark activity evidence incomplete when none of the required endpoints are available: {required_assays}"
            )
        additional_conditions = context.get("additional_stop_conditions", [])
        if isinstance(additional_conditions, str):
            additional_conditions = [additional_conditions]
        for condition in additional_conditions:
            text = str(condition).strip()
            if text and text not in conditions:
                conditions.append(text)
        return conditions


def _allocate_tokens(total: int, weights: Sequence[int]) -> list[int]:
    if not weights or sum(weights) <= 0:
        raise ValueError("Planner task weights must be positive")
    raw = [total * weight // sum(weights) for weight in weights]
    raw[-1] += total - sum(raw)
    return raw


def _normalize_revision_context(context: object | None) -> dict[str, Any]:
    if context is None:
        return {}
    if isinstance(context, PlanSpec):
        return {"previous_plan": context}
    if isinstance(context, Mapping):
        normalized = dict(context)
        previous = normalized.get("previous_plan")
        if isinstance(previous, Mapping):
            normalized["previous_plan"] = PlanSpec.model_validate(dict(previous))
        return normalized
    # ReworkOrder and similar Pydantic artifacts expose model_dump.
    if hasattr(context, "model_dump"):
        return dict(context.model_dump(mode="python"))
    return {"reason": str(context)}


def _next_revision(previous: object, context: Mapping[str, Any]) -> int:
    if isinstance(previous, PlanSpec):
        return previous.revision + 1
    explicit = context.get("revision")
    if explicit is not None:
        return max(1, int(explicit))
    return 1 if context else 0


def _context_note(context: Mapping[str, Any]) -> str:
    actions = context.get("required_actions", [])
    if isinstance(actions, str):
        actions = [actions]
    if isinstance(actions, Sequence):
        cleaned = [str(item).strip() for item in actions if str(item).strip()]
        if cleaned:
            return "; ".join(cleaned)
    reason = context.get("reason") or context.get("rationale")
    return str(reason).strip() if reason else ""


def _validate_task_graph(tasks: Sequence[PlanTask]) -> None:
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("PlanSpec contains duplicate task IDs")
    known: set[str] = set()
    for task in tasks:
        missing = set(task.depends_on) - known
        if missing:
            raise ValueError(
                f"Task {task.task_id} depends on unknown or forward tasks: {sorted(missing)}"
            )
        known.add(task.task_id)


def _validate_closed_tool_registry(tasks: Sequence[PlanTask]) -> None:
    used = {tool.tool_name for task in tasks for tool in task.tools}
    unapproved = used - set(APPROVED_TOOLS)
    if unapproved:
        raise ValueError(f"Plan requests unapproved tools: {sorted(unapproved)}")


__all__ = ["APPROVED_TOOLS", "PlannerAgent"]
