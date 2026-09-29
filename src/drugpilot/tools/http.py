"""Shared HTTP transport for read-only scientific database adapters.

The Windows build intentionally uses the operating-system certificate store.
Corporate and institutional networks commonly terminate TLS with a locally
trusted root certificate that is absent from Python's bundled ``certifi``
bundle.  PowerShell then succeeds while a plain ``httpx`` call fails with
``CERTIFICATE_VERIFY_FAILED``.  ``truststore`` closes that gap without turning
certificate verification off.

The client keeps environment proxy support enabled, applies bounded retries to
read-only queries, and retains sanitized failure diagnostics.  Individual
adapters remain injectable: unit tests can continue to pass their fake clients
without constructing this transport or touching the network.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import os
import ssl
from threading import Lock
from time import sleep
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import httpx


_RETRYABLE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "POST"})
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HttpFailureDiagnostic:
    """Sanitized description of one failed or retried public HTTP request."""

    method: str
    endpoint: str
    category: str
    attempt: int
    max_attempts: int
    detail: str
    status_code: int | None = None
    retrying: bool = False


def _safe_endpoint(url: str) -> str:
    """Remove query strings and credentials before retaining diagnostics."""

    parsed = urlsplit(url)
    host = parsed.hostname or "unknown-host"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    return f"{parsed.scheme or 'https'}://{host}{path}"


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text and text not in parts:
            parts.append(text)
        current = current.__cause__ or current.__context__
    return " | ".join(parts)[:600]


def classify_transport_error(exc: BaseException) -> str:
    """Map transport exceptions to an operator-facing, stable category."""

    text = _exception_chain_text(exc).casefold()
    if "certificate_verify_failed" in text or "certificate verify failed" in text:
        return "tls_certificate_verification"
    if isinstance(exc, httpx.ProxyError) or "proxy" in text:
        return "proxy_connection"
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "connection_pool_timeout"
    if "getaddrinfo failed" in text or "name or service not known" in text:
        return "dns_resolution"
    if "ssl" in text or "tls" in text:
        return "tls_connection"
    if isinstance(exc, httpx.ConnectError):
        return "connection"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "remote_protocol"
    return "transport"


def _build_verified_ssl_context(*, trust_env: bool) -> tuple[ssl.SSLContext, str]:
    """Build a verified context, preferring explicit CA env then OS trust."""

    cert_file = os.getenv("SSL_CERT_FILE") if trust_env else None
    cert_dir = os.getenv("SSL_CERT_DIR") if trust_env else None
    if cert_file or cert_dir:
        # Respect the standard Python/httpx environment contract when an
        # administrator intentionally supplied a custom CA bundle or directory.
        context = ssl.create_default_context(cafile=cert_file, capath=cert_dir)
        return context, "environment_ca"

    try:
        import truststore

        # Native trust is essential on managed Windows networks.  This still
        # performs normal hostname and certificate-chain verification.
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), "system_truststore"
    except ImportError:
        # ``httpx`` depends on certifi, but keep a stdlib-only final fallback so
        # importing this module never weakens or disables verification.
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where()), "certifi"
        except ImportError:
            return ssl.create_default_context(), "python_default"


class ScientificHttpClient:
    """Synchronous pooled client for keyless, read-only scientific APIs.

    POST is included in the retryable set because Open Targets, DGIdb, Pharos
    and RCSB use POST only for side-effect-free GraphQL/search requests.  The
    class must not be reused for a mutating API without disabling retries.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.20,
        max_retry_delay_seconds: float = 2.0,
        trust_env: bool = True,
        proxy: str | None = None,
        user_agent: str = "DrugPilot/0.1 scientific-public-data-client",
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= max_attempts <= 4:
            raise ValueError("max_attempts must be between 1 and 4")
        if retry_backoff_seconds < 0 or max_retry_delay_seconds < 0:
            raise ValueError("retry delays must not be negative")

        verify, trust_backend = _build_verified_ssl_context(trust_env=trust_env)
        client_kwargs: dict[str, Any] = {
            "verify": verify,
            "timeout": httpx.Timeout(
                timeout_seconds,
                connect=min(timeout_seconds, 5.0),
                pool=min(timeout_seconds, 5.0),
            ),
            "follow_redirects": True,
            "trust_env": trust_env,
            "headers": {
                "User-Agent": user_agent,
                "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
            },
            "limits": httpx.Limits(
                max_connections=30,
                max_keepalive_connections=15,
                keepalive_expiry=30.0,
            ),
        }
        if proxy:
            client_kwargs["proxy"] = proxy
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.trust_env = trust_env
        self.trust_backend = trust_backend
        self.proxy_mode = "explicit" if proxy else ("environment" if trust_env else "disabled")
        self._sleeper = sleeper
        self._diagnostics: deque[HttpFailureDiagnostic] = deque(maxlen=200)
        self._diagnostics_lock = Lock()

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        normalized_method = method.upper()
        attempts = self.max_attempts if normalized_method in _RETRYABLE_METHODS else 1
        endpoint = _safe_endpoint(url)
        last_error: httpx.TransportError | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.request(normalized_method, url, **kwargs)
            except httpx.TransportError as exc:
                last_error = exc
                category = classify_transport_error(exc)
                # A certificate/configuration error is deterministic; retrying
                # it only delays the user and obscures the actual cause.
                retrying = attempt < attempts and category not in {
                    "tls_certificate_verification",
                }
                self._record_failure(
                    HttpFailureDiagnostic(
                        method=normalized_method,
                        endpoint=endpoint,
                        category=category,
                        attempt=attempt,
                        max_attempts=attempts,
                        detail=_exception_chain_text(exc) or type(exc).__name__,
                        retrying=retrying,
                    )
                )
                if not retrying:
                    raise
                self._sleeper(self._retry_delay(attempt, None))
                continue

            if response.status_code not in _RETRYABLE_STATUS_CODES or attempt >= attempts:
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    self._record_failure(
                        HttpFailureDiagnostic(
                            method=normalized_method,
                            endpoint=endpoint,
                            category="retryable_http_status",
                            attempt=attempt,
                            max_attempts=attempts,
                            status_code=response.status_code,
                            detail=f"HTTP {response.status_code}",
                            retrying=False,
                        )
                    )
                return response

            retry_after = response.headers.get("Retry-After")
            self._record_failure(
                HttpFailureDiagnostic(
                    method=normalized_method,
                    endpoint=endpoint,
                    category="retryable_http_status",
                    attempt=attempt,
                    max_attempts=attempts,
                    status_code=response.status_code,
                    detail=f"HTTP {response.status_code}",
                    retrying=True,
                )
            )
            response.close()
            self._sleeper(self._retry_delay(attempt, retry_after))

        # The loop either returns or raises.  Keep a defensive branch for type
        # checkers and future changes to the retry policy.
        assert last_error is not None
        raise last_error

    def recent_failures(self) -> tuple[HttpFailureDiagnostic, ...]:
        with self._diagnostics_lock:
            return tuple(self._diagnostics)

    def diagnostic_summary(self) -> dict[str, Any]:
        failures = self.recent_failures()
        return {
            "trust_backend": self.trust_backend,
            "proxy_mode": self.proxy_mode,
            "trust_env": self.trust_env,
            "max_attempts": self.max_attempts,
            "recent_failures": [asdict(item) for item in failures],
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ScientificHttpClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _record_failure(self, item: HttpFailureDiagnostic) -> None:
        with self._diagnostics_lock:
            self._diagnostics.append(item)

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(max(0.0, float(retry_after)), self.max_retry_delay_seconds)
            except ValueError:
                pass
        return min(
            self.retry_backoff_seconds * (2 ** (attempt - 1)),
            self.max_retry_delay_seconds,
        )


@lru_cache(maxsize=1)
def get_scientific_http_client() -> ScientificHttpClient:
    """Return the process-wide connection pool used by public adapters."""

    return ScientificHttpClient()


class HttpJsonTransport:
    """Small base-URL JSON adapter retained for external extension points."""

    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        verify, _ = _build_verified_ssl_context(trust_env=True)
        self._client = httpx.Client(
            base_url=base_url,
            headers=dict(headers or {}),
            timeout=timeout_seconds,
            follow_redirects=True,
            trust_env=True,
            verify=verify,
        )

    def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str | int | float] | None = None,
    ) -> Any:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpJsonTransport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "HttpFailureDiagnostic",
    "HttpJsonTransport",
    "ScientificHttpClient",
    "classify_transport_error",
    "get_scientific_http_client",
]
