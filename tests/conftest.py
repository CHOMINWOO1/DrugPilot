from __future__ import annotations

import pytest

from drugpilot.domain.models import ProjectConstraints, ResearchBrief


@pytest.fixture
def brief() -> ResearchBrief:
    return ResearchBrief(
        brief_id="brief_test_001",
        research_goal=(
            "Identify evidence-linked FLT3 inhibition hypotheses and compare "
            "repurposing with de novo candidates for acute myeloid leukemia."
        ),
        disease="Acute myeloid leukemia",
        constraints=ProjectConstraints(
            target_cell_type="MOLM-13",
            intervention_direction="inhibit",
            candidates_per_route=3,
            max_ma1_iterations=2,
            max_rework_loops=2,
        ),
    )

