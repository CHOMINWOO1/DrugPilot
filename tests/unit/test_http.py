from __future__ import annotations

import httpx
import pytest

from drugpilot.tools.http import ScientificHttpClient, classify_transport_error


def test_shared_client_retries_a_transient_connection_error_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary connection reset", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    client = ScientificHttpClient(
        transport=httpx.MockTransport(handler),
        max_attempts=2,
        retry_backoff_seconds=0,
    )
    try:
        response = client.get("https://example.test/query?secret=must-not-be-stored")
    finally:
        client.close()

    assert response.json() == {"ok": True}
    assert calls == 2
    failures = client.recent_failures()
    assert len(failures) == 1
    assert failures[0].category == "connection"
    assert failures[0].retrying is True
    assert failures[0].endpoint == "https://example.test/query"


def test_certificate_verification_failure_is_diagnostic_and_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate in certificate chain",
            request=request,
        )

    client = ScientificHttpClient(
        transport=httpx.MockTransport(handler),
        max_attempts=3,
        retry_backoff_seconds=0,
    )
    with pytest.raises(httpx.ConnectError):
        client.get("https://api.example.test/private?q=not-retained")
    client.close()

    assert calls == 1
    failure = client.recent_failures()[-1]
    assert failure.category == "tls_certificate_verification"
    assert failure.retrying is False
    assert "?" not in failure.endpoint


def test_retryable_http_status_respects_bound() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, headers={"Retry-After": "0"}, request=request)

    client = ScientificHttpClient(
        transport=httpx.MockTransport(handler),
        max_attempts=2,
        retry_backoff_seconds=0,
    )
    try:
        response = client.post("https://example.test/graphql", json={"query": "{ ping }"})
    finally:
        client.close()

    assert response.status_code == 503
    assert calls == 2
    assert [item.retrying for item in client.recent_failures()] == [True, False]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectTimeout("slow"), "connect_timeout"),
        (httpx.ReadTimeout("slow"), "read_timeout"),
        (httpx.ProxyError("proxy refused"), "proxy_connection"),
        (httpx.ConnectError("getaddrinfo failed"), "dns_resolution"),
    ],
)
def test_transport_error_categories_are_stable(
    error: httpx.TransportError,
    expected: str,
) -> None:
    assert classify_transport_error(error) == expected
