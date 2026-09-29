from __future__ import annotations

from dataclasses import dataclass

from drugpilot.domain.models import BudgetPolicy, TokenUsage


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0


# Planning-date defaults (2026-08-13). Environment/model-registry updates can
# replace these without changing workflow code.
DEFAULT_MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5.6-sol": ModelPrice(5.0, 30.0, 0.5),
    "gpt-5.6-terra": ModelPrice(2.0, 12.0, 0.2),
    "gpt-5.6-luna": ModelPrice(0.2, 1.2, 0.02),
    # Anthropic public list price for Claude Opus 4.1.  Kept as an
    # explicit approved entry so the independent Reviewer cannot bypass
    # DrugPilot's cost preflight.
    "claude-opus-4-1": ModelPrice(15.0, 75.0, 1.5),
}


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    def __init__(
        self,
        policy: BudgetPolicy,
        prices: dict[str, ModelPrice] | None = None,
    ) -> None:
        self.policy = policy
        self.prices = prices or DEFAULT_MODEL_PRICES

    def price_usage(self, usage: TokenUsage) -> TokenUsage:
        model = usage.model.removeprefix("mock:")
        price = self.prices.get(model)
        if price is None or usage.model.startswith("mock:"):
            return usage
        output_and_reasoning = usage.output_tokens + usage.reasoning_tokens
        estimated_cost = (
            usage.input_tokens * price.input_per_million
            + output_and_reasoning * price.output_per_million
        ) / 1_000_000
        usage.estimated_cost_usd = round(estimated_cost, 6)
        return usage

    def estimate_cost(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        price = self.prices.get(model.removeprefix("mock:"))
        if price is None or model.startswith("mock:"):
            return 0.0
        return (
            input_tokens * price.input_per_million
            + output_tokens * price.output_per_million
        ) / 1_000_000

    def preflight(
        self,
        usages: list[TokenUsage],
        *,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
    ) -> None:
        """Reject an LLM request before it can exceed any run policy limit."""

        if not model.startswith("mock:") and any(
            usage.estimated and not usage.model.startswith("mock:")
            for usage in usages
        ):
            raise BudgetExceeded(
                "prior live LLM usage is estimated; further live calls are blocked"
            )
        if input_tokens > self.policy.max_single_input_tokens:
            raise BudgetExceeded(
                "single input cap would be exceeded "
                f"({input_tokens:,}/{self.policy.max_single_input_tokens:,})"
            )
        calls = len(usages)
        if calls + 1 > self.policy.max_llm_calls:
            raise BudgetExceeded(
                f"LLM call cap would be exceeded ({calls + 1}/{self.policy.max_llm_calls})"
            )
        total = sum(usage.total_tokens for usage in usages)
        projected = total + input_tokens + max_output_tokens
        if projected > self.policy.hard_max_run_tokens:
            raise BudgetExceeded(
                "hard token cap would be exceeded "
                f"({projected:,}/{self.policy.hard_max_run_tokens:,})"
            )
        if projected > self.policy.max_run_tokens:
            raise BudgetExceeded(
                "run token cap would be exceeded "
                f"({projected:,}/{self.policy.max_run_tokens:,})"
            )
        cost = sum(usage.estimated_cost_usd for usage in usages)
        if (
            model not in self.prices
            and not model.startswith("mock:")
        ):
            raise BudgetExceeded(
                f"no approved price is configured for live model {model}"
            )
        projected_cost = cost + self.estimate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=max_output_tokens,
        )
        if projected_cost > self.policy.max_run_cost_usd:
            raise BudgetExceeded(
                "run cost cap would be exceeded "
                f"(${projected_cost:.2f}/${self.policy.max_run_cost_usd:.2f})"
            )

    def check(self, usages: list[TokenUsage]) -> None:
        total_tokens = sum(usage.total_tokens for usage in usages)
        calls = len(usages)
        cost = sum(usage.estimated_cost_usd for usage in usages)
        if total_tokens > self.policy.hard_max_run_tokens:
            raise BudgetExceeded(
                f"hard token cap reached ({total_tokens:,}/{self.policy.hard_max_run_tokens:,})"
            )
        if total_tokens > self.policy.max_run_tokens:
            raise BudgetExceeded(
                f"run token cap reached ({total_tokens:,}/{self.policy.max_run_tokens:,})"
            )
        if calls > self.policy.max_llm_calls:
            raise BudgetExceeded(
                f"LLM call cap reached ({calls}/{self.policy.max_llm_calls})"
            )
        if cost > self.policy.max_run_cost_usd:
            raise BudgetExceeded(
                f"run cost cap reached (${cost:.2f}/${self.policy.max_run_cost_usd:.2f})"
            )
        if any(
            usage.estimated and not usage.model.startswith("mock:")
            for usage in usages
        ):
            raise BudgetExceeded(
                "live LLM usage is estimated; conservative usage was recorded and the run stopped"
            )

    @staticmethod
    def summary(usages: list[TokenUsage]) -> dict[str, int | float]:
        return {
            "calls": len(usages),
            "input_tokens": sum(usage.input_tokens for usage in usages),
            "output_tokens": sum(usage.output_tokens for usage in usages),
            "reasoning_tokens": sum(usage.reasoning_tokens for usage in usages),
            "total_tokens": sum(usage.total_tokens for usage in usages),
            "estimated_cost_usd": round(
                sum(usage.estimated_cost_usd for usage in usages), 6
            ),
        }
