from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from drugpilot.config import Settings
from drugpilot.domain.models import RunMode, TokenUsage


ResponseT = TypeVar("ResponseT", bound=BaseModel)


@dataclass(frozen=True)
class LLMResult:
    data: BaseModel
    usage: TokenUsage


class LLMProviderCallError(RuntimeError):
    """A provider call failed after an external request was attempted.

    ``usage`` is populated whenever the provider returned a response carrying
    billable token counts, even when refusal, incomplete output, or local
    validation prevents that response from becoming an ``LLMResult``.  A
    request/HTTP failure from a built-in live provider carries a conservative
    usage bound when exact provider usage is unavailable.  This prevents a
    paid attempt from becoming a zero-token accounting marker.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: TokenUsage | None,
        reason_code: str = "PROVIDER_CALL_FAILED",
    ) -> None:
        super().__init__(message)
        self.usage = usage
        # Never persist raw provider errors, response bodies, or credentials.
        allowed = {"PROVIDER_CALL_FAILED", "OUTPUT_LIMIT", "PROVIDER_INCOMPLETE",
            "PROVIDER_ERROR", "PROVIDER_REFUSAL", "EMPTY_STRUCTURED_OUTPUT",
            "STRUCTURED_OUTPUT_INVALID", "TRANSPORT_FAILURE", "DOMAIN_AUTHORITY_REJECTED"}
        self.reason_code = reason_code if reason_code in allowed else "PROVIDER_CALL_FAILED"


class LLMProvider(Protocol):
    async def structured(
        self,
        *,
        component: str,
        model: str,
        system: str,
        prompt: str,
        response_model: type[ResponseT],
        fallback: ResponseT,
        max_output_tokens: int,
    ) -> LLMResult: ...


def rough_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def _openai_text_format(response_model: type[ResponseT]) -> dict[str, object]:
    from openai.lib._pydantic import to_strict_json_schema

    return {
        "type": "json_schema",
        "name": response_model.__name__[:64],
        "schema": to_strict_json_schema(response_model),
        "strict": True,
    }


def _anthropic_output_format(response_model: type[ResponseT]) -> dict[str, object]:
    from anthropic.lib._parse._transform import transform_schema
    from pydantic import TypeAdapter

    return {
        "type": "json_schema",
        "schema": transform_schema(TypeAdapter(response_model).json_schema()),
    }


def _canonical_utf8_size(payload: object) -> int:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return len(encoded)


def conservative_structured_input_tokens(
    *,
    system: str,
    prompt: str,
    response_model: type[ResponseT],
) -> int:
    """Return a provider-neutral upper bound for structured input accounting.

    Strict structured-output schemas are part of the request and can be
    billable input.  Use the larger canonical OpenAI/Anthropic request-like
    UTF-8 byte count as a deliberately conservative token bound (one token per
    byte), rather than the ordinary four-byte rough estimate.
    """

    openai_payload = {
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "text": {"format": _openai_text_format(response_model)},
    }
    anthropic_payload = {
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "format": _anthropic_output_format(response_model),
        },
    }
    return max(
        1,
        _canonical_utf8_size(openai_payload),
        _canonical_utf8_size(anthropic_payload),
    )


def _value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


_MISSING = object()


def _strict_required_usage_int(value: object, name: str) -> int | None:
    """Read a required provider counter without coercing malformed values."""

    raw = _value(value, name, _MISSING)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return raw


def _strict_optional_usage_int(value: object, name: str) -> int | None:
    """Read an optional provider counter; absent/None means an exact zero."""

    raw = _value(value, name, _MISSING)
    if raw is _MISSING or raw is None:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return raw


def _conservative_usage(
    *,
    component: str,
    model: str,
    input_tokens: int,
    max_output_tokens: int,
) -> TokenUsage:
    """Bound an attempted call when the provider supplies no trustworthy usage."""

    return TokenUsage(
        component=component,
        model=model,
        input_tokens=max(1, input_tokens),
        output_tokens=max(0, max_output_tokens),
        estimated=True,
    )


def _openai_usage_unchecked(
    response: object,
    *,
    component: str,
    model: str,
) -> TokenUsage | None:
    usage_obj = _value(response, "usage")
    if usage_obj is None:
        return None
    input_tokens = _strict_required_usage_int(usage_obj, "input_tokens")
    output_tokens = _strict_required_usage_int(usage_obj, "output_tokens")
    total_tokens = _strict_required_usage_int(usage_obj, "total_tokens")
    if (
        input_tokens is None
        or input_tokens <= 0
        or output_tokens is None
        or total_tokens is None
        or total_tokens != input_tokens + output_tokens
    ):
        return None
    details = _value(usage_obj, "output_tokens_details", None)
    if details is not None and isinstance(
        details,
        (str, bytes, bool, int, float, list, tuple, set),
    ):
        return None
    reasoning = _strict_optional_usage_int(details, "reasoning_tokens")
    if reasoning is None or reasoning > output_tokens:
        return None
    return TokenUsage(
        component=component,
        model=model,
        input_tokens=input_tokens,
        output_tokens=max(0, output_tokens - reasoning),
        reasoning_tokens=reasoning,
    )


def _openai_usage(
    response: object,
    *,
    component: str,
    model: str,
) -> TokenUsage | None:
    """Parse exact OpenAI usage or return None for any malformed object."""

    try:
        return _openai_usage_unchecked(
            response,
            component=component,
            model=model,
        )
    except Exception:
        return None


def _anthropic_usage_unchecked(
    response: object,
    *,
    component: str,
    model: str,
) -> TokenUsage | None:
    usage_obj = _value(response, "usage")
    if usage_obj is None:
        return None
    input_tokens = _strict_required_usage_int(usage_obj, "input_tokens")
    output_tokens = _strict_required_usage_int(usage_obj, "output_tokens")
    cache_creation = _strict_optional_usage_int(
        usage_obj,
        "cache_creation_input_tokens",
    )
    cache_read = _strict_optional_usage_int(usage_obj, "cache_read_input_tokens")
    if (
        None in {input_tokens, output_tokens, cache_creation, cache_read}
        or input_tokens <= 0
    ):
        return None
    details = _value(usage_obj, "output_tokens_details", None)
    if details is not None and isinstance(
        details,
        (str, bytes, bool, int, float, list, tuple, set),
    ):
        return None
    reasoning = _strict_optional_usage_int(details, "reasoning_tokens")
    if reasoning is None or reasoning > output_tokens:
        return None
    input_tokens += cache_creation + cache_read
    return TokenUsage(
        component=component,
        model=model,
        input_tokens=input_tokens,
        output_tokens=max(0, output_tokens - reasoning),
        reasoning_tokens=reasoning,
    )


def _anthropic_usage(
    response: object,
    *,
    component: str,
    model: str,
) -> TokenUsage | None:
    """Parse exact Anthropic usage or return None for any malformed object."""

    try:
        return _anthropic_usage_unchecked(
            response,
            component=component,
            model=model,
        )
    except Exception:
        return None


def usage_for_failed_call(
    exc: BaseException,
    *,
    component: str,
    model: str,
    conservative_input_tokens: int | None = None,
    conservative_max_output_tokens: int | None = None,
) -> TokenUsage | None:
    """Return known usage or a conservative bound for an attempted API call.

    ``estimated=True`` makes the absence of exact provider usage explicit.
    The caller must price and persist this bound before stopping.  Exceptions
    raised before an external request attempt are not ``LLMProviderCallError``
    and do not consume the call ceiling.
    """

    if not isinstance(exc, LLMProviderCallError):
        return None
    if exc.usage is not None:
        return exc.usage.model_copy(deep=True)
    if (
        conservative_input_tokens is None
        or conservative_max_output_tokens is None
    ):
        # Built-in live providers always attach their own exact-prompt bound.
        # A third-party provider that supplies neither usage nor a caller bound
        # cannot be converted into a misleading zero-token marker.
        return None
    return _conservative_usage(
        component=component,
        model=model,
        input_tokens=conservative_input_tokens,
        max_output_tokens=conservative_max_output_tokens,
    )


def _has_refusal(response: object) -> bool:
    output = _value(response, "output", [])
    if not isinstance(output, list):
        return False
    for item in output:
        content = _value(item, "content", [])
        if not isinstance(content, list):
            continue
        if any(_value(part, "type") == "refusal" for part in content):
            return True
    return False


def _validated_openai_response(
    response: Any,
    response_model: type[ResponseT],
    *,
    usage: TokenUsage | None,
) -> ResponseT:
    """Fail closed while retaining usage from every returned response."""

    status = getattr(response, "status", None)
    if status is not None and status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
        raise LLMProviderCallError(
            f"provider returned non-completed status: {status}",
            usage=usage,
            reason_code="OUTPUT_LIMIT" if status == "incomplete" and reason == "max_output_tokens" else "PROVIDER_INCOMPLETE",
        )
    if getattr(response, "error", None) is not None:
        raise LLMProviderCallError("provider returned an error response", usage=usage, reason_code="PROVIDER_ERROR")
    if _has_refusal(response):
        raise LLMProviderCallError("provider refused structured output", usage=usage, reason_code="PROVIDER_REFUSAL")
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise LLMProviderCallError("provider returned no JSON output", usage=usage, reason_code="EMPTY_STRUCTURED_OUTPUT")
    try:
        return response_model.model_validate_json(output_text)
    except Exception as exc:
        raise LLMProviderCallError(
            "provider returned structured output that failed local validation",
            usage=usage,
            reason_code="STRUCTURED_OUTPUT_INVALID",
        ) from exc


def _validated_anthropic_response(
    response: Any,
    response_model: type[ResponseT],
    *,
    usage: TokenUsage | None,
) -> ResponseT:
    """Validate Anthropic JSON locally after preserving response usage."""

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason not in {None, "end_turn", "stop_sequence"}:
        raise LLMProviderCallError(
            f"Anthropic reviewer returned non-completed stop reason: {stop_reason}",
            usage=usage,
        )
    content = getattr(response, "content", [])
    text_parts = [
        getattr(item, "text", "")
        for item in content
        if getattr(item, "type", None) == "text"
    ]
    output_text = "".join(text_parts)
    if not output_text.strip():
        raise LLMProviderCallError(
            "Anthropic reviewer returned no structured output",
            usage=usage,
        )
    try:
        return response_model.model_validate_json(output_text)
    except Exception as exc:
        raise LLMProviderCallError(
            "Anthropic reviewer output failed local validation",
            usage=usage,
        ) from exc


class MockLLMProvider:
    """Deterministic provider that preserves the live provider contract."""

    async def structured(
        self,
        *,
        component: str,
        model: str,
        system: str,
        prompt: str,
        response_model: type[ResponseT],
        fallback: ResponseT,
        max_output_tokens: int,
    ) -> LLMResult:
        encoded = fallback.model_dump_json()
        input_tokens = rough_tokens(system + prompt)
        output_tokens = min(rough_tokens(encoded), max_output_tokens)
        return LLMResult(
            data=response_model.model_validate_json(encoded),
            usage=TokenUsage(
                component=component,
                model=f"mock:{model}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=0,
                estimated=True,
            ),
        )


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, *, client: Any | None = None) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no OpenAI client is injected")
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, max_retries=0)
        # One accounted request must not hide additional SDK-level paid retries.
        self.client = client.with_options(max_retries=0) if callable(getattr(client, "with_options", None)) else client

    async def structured(
        self,
        *,
        component: str,
        model: str,
        system: str,
        prompt: str,
        response_model: type[ResponseT],
        fallback: ResponseT,
        max_output_tokens: int,
    ) -> LLMResult:
        del fallback  # Live validation failures must never be masked by fixture data.
        # ``responses.parse`` performs Pydantic parsing inside the SDK's
        # post-parser.  If that parser raises, the caller never receives the
        # response object and its paid usage is lost.  Request the identical
        # strict schema through ``responses.create`` and validate locally after
        # extracting usage instead.
        text_format = _openai_text_format(response_model)
        conservative_usage = _conservative_usage(
            component=component,
            model=model,
            input_tokens=conservative_structured_input_tokens(
                system=system,
                prompt=prompt,
                response_model=response_model,
            ),
            max_output_tokens=max_output_tokens,
        )
        try:
            response = await self.client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                text={"format": text_format},
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise LLMProviderCallError(
                "OpenAI request failed before a usable response was returned",
                usage=conservative_usage,
                reason_code="TRANSPORT_FAILURE",
            ) from exc
        usage = _openai_usage(response, component=component, model=model)
        if usage is None:
            usage = conservative_usage
        data = _validated_openai_response(
            response,
            response_model,
            usage=usage,
        )
        return LLMResult(data=data, usage=usage)


class AnthropicProvider:
    """Structured-output adapter reserved for the independent MA3 reviewer.

    The injected ``client`` hook keeps the adapter unit-testable without a key or
    network access.  Production construction uses ``AsyncAnthropic`` from the
    installed SDK.  The strict schema is sent through ``messages.create`` and
    validated locally only after paid response usage has been extracted.
    """

    def __init__(self, api_key: str | None = None, *, client: Any | None = None) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no Anthropic client is injected")
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=api_key, max_retries=0)
        self.client = client.with_options(max_retries=0) if callable(getattr(client, "with_options", None)) else client

    async def structured(
        self,
        *,
        component: str,
        model: str,
        system: str,
        prompt: str,
        response_model: type[ResponseT],
        fallback: ResponseT,
        max_output_tokens: int,
    ) -> LLMResult:
        del fallback  # Live structured output must validate; never mask it with fixture data.
        output_format = _anthropic_output_format(response_model)
        conservative_usage = _conservative_usage(
            component=component,
            model=model,
            input_tokens=conservative_structured_input_tokens(
                system=system,
                prompt=prompt,
                response_model=response_model,
            ),
            max_output_tokens=max_output_tokens,
        )
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": output_format},
            )
        except Exception as exc:
            raise LLMProviderCallError(
                "Anthropic request failed before a usable response was returned",
                usage=conservative_usage,
            ) from exc
        usage = _anthropic_usage(response, component=component, model=model)
        if usage is None:
            usage = conservative_usage
        data = _validated_anthropic_response(
            response,
            response_model,
            usage=usage,
        )
        return LLMResult(data=data, usage=usage)


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mock = MockLLMProvider()
        self.openai: OpenAIProvider | None = None
        self.anthropic_reviewer: AnthropicProvider | None = None
        if settings.openai_api_key:
            self.openai = OpenAIProvider(settings.openai_api_key)
        if settings.anthropic_api_key:
            self.anthropic_reviewer = AnthropicProvider(settings.anthropic_api_key)

    def for_component(self, component: str) -> LLMProvider:
        if self.settings.effective_mode is RunMode.MOCK:
            return self.mock
        if self.openai is not None:
            return self.openai
        if self.settings.effective_mode is RunMode.HYBRID:
            return self.mock
        raise RuntimeError(f"no live LLM provider configured for {component}")

    @property
    def component_backend(self) -> str:
        """Public, secret-free description of the MA1/Planner LLM backend."""

        if self.settings.effective_mode is not RunMode.MOCK and self.openai is not None:
            return "openai"
        return "mock"

    @property
    def reviewer_backend(self) -> str:
        """Public, secret-free description of the read-only MA3 reviewer."""

        if (
            self.settings.effective_mode is not RunMode.MOCK
            and self.anthropic_reviewer is not None
        ):
            return "anthropic"
        if (
            self.settings.effective_mode is not RunMode.MOCK
            and self.settings.openai_ma3_reviewer_enabled
            and self.openai is not None
        ):
            return "openai-read-only"
        return "deterministic-rules"

    def for_reviewer(self) -> LLMProvider | None:
        """Return a read-only semantic reviewer only when live calls are enabled.

        A separately configured Anthropic provider is preferred.  In the test
        build, an OpenAI key may also supply a role-separated Sol reviewer so a
        single key demonstrates semantic G0/G1 review.  Regardless of provider,
        the reviewer never selects the verdict: MA3's deterministic decision
        table remains authoritative.
        """

        if self.settings.effective_mode is RunMode.MOCK:
            return None
        if self.anthropic_reviewer is not None:
            return self.anthropic_reviewer
        if self.settings.openai_ma3_reviewer_enabled:
            return self.openai
        return None


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
