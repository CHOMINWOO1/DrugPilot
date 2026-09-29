from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from drugpilot.agents.ma1 import MA1Agent
from drugpilot.agents.ma3 import (
    MA3AssuranceAgent,
    ReviewerAssessment,
)
from drugpilot.config import Settings
from drugpilot.domain.models import GateVerdictCode, RunMode
from drugpilot.providers import AnthropicProvider
from drugpilot.providers.structured_outputs import (
    ReviewerAssessmentOutput,
    ReviewerRoutingAssessmentOutput,
    ReviewerRoutingFindingOutput,
)


def _fixture_ma1(brief):
    return MA1Agent(
        Settings(
            mode=RunMode.MOCK,
            public_evidence_enabled=False,
            dynamic_target_discovery_enabled=False,
            dynamic_target_fallback_enabled=True,
            demo_fixture_fallback_enabled=True,
        )
    ).run(brief)


def test_reviewer_wire_schema_cannot_choose_a_gate_verdict() -> None:
    with pytest.raises(ValidationError):
        ReviewerAssessmentOutput.model_validate(
            {
                "summary": "Attempted authority escalation.",
                "additional_findings": [],
                "uncertainties": [],
                "verdict": "PASS",
                "rewrite_artifacts": True,
            }
        )


class FakeMessages:
    def __init__(self, assessment: BaseModel) -> None:
        self.assessment = assessment
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(
                    type="text",
                    text=self.assessment.model_dump_json(),
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                cache_creation_input_tokens=20,
                cache_read_input_tokens=30,
                output_tokens=50,
                output_tokens_details=SimpleNamespace(reasoning_tokens=10),
            ),
        )


class FakeAnthropicClient:
    def __init__(self, assessment: BaseModel) -> None:
        self.messages = FakeMessages(assessment)


@pytest.mark.asyncio
async def test_anthropic_adapter_uses_strict_create_without_network():
    assessment = ReviewerAssessment(summary="No additional issue.")
    client = FakeAnthropicClient(assessment)
    provider = AnthropicProvider(client=client)

    result = await provider.structured(
        component="MA3-G0-Reviewer",
        model="reviewer-test-model",
        system="Read only.",
        prompt="{}",
        response_model=ReviewerAssessment,
        fallback=assessment,
        max_output_tokens=4000,
    )

    assert result.data == assessment
    assert result.usage.input_tokens == 150
    assert result.usage.output_tokens == 40
    assert result.usage.reasoning_tokens == 10
    output_format = client.messages.kwargs["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["type"] == "object"
    assert client.messages.kwargs["messages"] == [
        {"role": "user", "content": "{}"}
    ]


@pytest.mark.asyncio
async def test_ma3_external_reviewer_is_read_only_and_cannot_set_verdict_directly(brief):
    packet = _fixture_ma1(brief)
    before = packet.model_dump_json()
    assessment = ReviewerRoutingAssessmentOutput(
        summary="A mechanism caveat needs revision.",
        additional_findings=[
            ReviewerRoutingFindingOutput(
                category="scientific",
                severity="HIGH",
                code="MECHANISM_CAVEAT",
                message="The proposed mechanism needs an orthogonal validation assay.",
                artifact_id=packet.hypotheses[0].hypothesis_id,
                claim_ids=[],
                failed_node="ma1.critic",
                remediation_kind="experiment_required",
            )
        ],
        uncertainties=[],
    )
    client = FakeAnthropicClient(assessment)
    usage = []
    assurance = MA3AssuranceAgent(
        reviewer_model="reviewer-test-model",
        reviewer_provider=AnthropicProvider(client=client),
    )

    async def record(item):
        usage.append(item)

    verdict = await assurance.audit_g0(brief, packet, on_usage=record)

    assert verdict.verdict is GateVerdictCode.PASS
    assert verdict.reviewer_invoked is True
    assert verdict.meta_critic_invoked is False
    assert "REVIEWER_MECHANISM_CAVEAT" in {
        finding.code for finding in verdict.findings
    }
    assert next(
        finding for finding in verdict.findings
        if finding.code == "REVIEWER_MECHANISM_CAVEAT"
    ).remediation_kind == "experiment_required"
    advisory = next(f for f in verdict.findings if f.code == "REVIEWER_MECHANISM_CAVEAT")
    assert advisory.severity.value == "MEDIUM"
    assert advisory.reported_severity.value == "HIGH"
    assert advisory.effective_severity.value == "MEDIUM"
    assert advisory.confirmation_status == "unconfirmed"
    assert verdict.rework_order is None
    assert packet.model_dump_json() == before
    assert len(usage) == 1
    output_format = client.messages.kwargs["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["type"] == "object"
    reviewer_system = client.messages.kwargs["system"]
    assert "search_only_not_executable" in reviewer_system
    assert "not_assessed_search_stage" in reviewer_system
    assert "retained hypothesis" in reviewer_system
    assert "not by itself a defect" in reviewer_system


@pytest.mark.asyncio
async def test_ma3_without_reviewer_key_retains_deterministic_fallback(brief):
    packet = _fixture_ma1(brief)

    verdict = await MA3AssuranceAgent().audit_g0(brief, packet)

    assert verdict.verdict is GateVerdictCode.PASS
    assert verdict.meta_critic_invoked is False
    assert verdict.reviewer_invoked is False
    assert verdict.reviewer_model == "deterministic-rules-v1"


@pytest.mark.asyncio
async def test_ma3_does_not_swallow_authoritative_reviewer_preflight(brief):
    packet = _fixture_ma1(brief)
    assurance = MA3AssuranceAgent(
        reviewer_model="reviewer-test-model",
        reviewer_provider=AnthropicProvider(
            client=FakeAnthropicClient(
                ReviewerAssessmentOutput(
                    summary="unused",
                    additional_findings=[],
                    uncertainties=[],
                )
            )
        ),
    )

    async def deny(*_args):
        raise RuntimeError("budget preflight denied")

    with pytest.raises(RuntimeError, match="budget preflight denied"):
        await assurance.audit_g0(brief, packet, before_review=deny)
