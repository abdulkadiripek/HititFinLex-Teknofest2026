from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from fastapi import HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader


ADMIN_API_KEY_ENV = "HITITFINLEX_ADMIN_API_KEY"
ADMIN_API_KEY_HEADER = "X-API-Key"
MIN_ADMIN_API_KEY_LENGTH = 32
REJECTED_ADMIN_API_KEYS = frozenset(
    {
        "admin",
        "changeme",
        "change_me_local_only",
        "change_me_to_a_long_random_value",
        "default",
        "password",
        "replace_me",
        "secret",
        "test",
    }
)
DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_RATE_LIMIT_PER_MINUTE = 120
DEFAULT_ADMIN_RATE_LIMIT_PER_MINUTE = 30
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


admin_api_key_header = APIKeyHeader(
    name=ADMIN_API_KEY_HEADER,
    scheme_name="AdminApiKey",
    description=(
        "Required for review administration and for /intake requests with "
        "write=true. Configure HITITFINLEX_ADMIN_API_KEY on the server."
    ),
    auto_error=False,
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value >= minimum else default


def get_max_body_bytes() -> int:
    return _env_int(
        "HITITFINLEX_MAX_BODY_BYTES",
        DEFAULT_MAX_BODY_BYTES,
        minimum=1024,
    )


def get_rate_limits() -> tuple[int, int]:
    return (
        _env_int(
            "HITITFINLEX_RATE_LIMIT_PER_MINUTE",
            DEFAULT_RATE_LIMIT_PER_MINUTE,
        ),
        _env_int(
            "HITITFINLEX_ADMIN_RATE_LIMIT_PER_MINUTE",
            DEFAULT_ADMIN_RATE_LIMIT_PER_MINUTE,
        ),
    )


def get_cors_settings() -> dict[str, Any]:
    configured = os.getenv("HITITFINLEX_CORS_ORIGINS")
    if configured is None:
        origins = list(DEFAULT_CORS_ORIGINS)
    else:
        origins = [
            value.strip()
            for value in configured.split(",")
            if value.strip()
        ]

    allow_credentials = _env_bool(
        "HITITFINLEX_CORS_ALLOW_CREDENTIALS",
        default=False,
    )
    if "*" in origins and allow_credentials:
        raise RuntimeError(
            "HITITFINLEX_CORS_ALLOW_CREDENTIALS cannot be enabled when "
            "HITITFINLEX_CORS_ORIGINS contains '*'."
        )

    origin_regex = os.getenv("HITITFINLEX_CORS_ORIGIN_REGEX")
    return {
        "allow_origins": origins,
        "allow_origin_regex": origin_regex.strip() if origin_regex else None,
        "allow_credentials": allow_credentials,
        "allow_methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Accept",
            "Authorization",
            "Content-Type",
            ADMIN_API_KEY_HEADER,
            "X-RAG-Client-Id",
            "X-RAG-Session-Id",
        ],
    }


def ensure_admin_api_key(api_key: str | None) -> None:
    expected = os.getenv(ADMIN_API_KEY_ENV, "").strip()
    normalized = expected.casefold()
    insecure = (
        len(expected) < MIN_ADMIN_API_KEY_LENGTH
        or normalized in REJECTED_ADMIN_API_KEYS
        or normalized.startswith(("change_me", "changeme", "replace_me"))
        or len(set(expected)) < 8
    )
    if insecure:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Admin API is disabled because "
                f"{ADMIN_API_KEY_ENV} is missing or insecure. Configure a "
                f"unique value of at least {MIN_ADMIN_API_KEY_LENGTH} characters."
            ),
        )
    if not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid admin API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def require_admin_api_key(
    api_key: str | None = Security(admin_api_key_header),
) -> None:
    ensure_admin_api_key(api_key)


class BodySizeLimitMiddleware:
    """Reject oversized request bodies before FastAPI parses them."""

    def __init__(self, app, max_body_bytes: int | None = None):
        self.app = app
        self.max_body_bytes = max_body_bytes or get_max_body_bytes()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                response = JSONResponse(
                    {"detail": "Invalid Content-Length header."},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
                await response(scope, receive, send)
                return
            if declared_length < 0:
                response = JSONResponse(
                    {"detail": "Invalid Content-Length header."},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
                await response(scope, receive, send)
                return
            if declared_length > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return

        if scope.get("method", "GET").upper() not in {
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        }:
            await self.app(scope, receive, send)
            return

        buffered_messages = deque()
        received_bytes = 0
        while True:
            message = await receive()
            buffered_messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            received_bytes += len(message.get("body", b""))
            if received_bytes > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive():
            if buffered_messages:
                return buffered_messages.popleft()
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, scope, receive, send):
        response = JSONResponse(
            {
                "detail": (
                    "Request body exceeds the configured limit of "
                    f"{self.max_body_bytes} bytes."
                )
            },
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )
        await response(scope, receive, send)


class SlidingWindowRateLimitMiddleware:
    """Small in-process limiter; deploy a shared edge limiter when scaling out."""

    def __init__(
        self,
        app,
        requests_per_minute: int | None = None,
        admin_requests_per_minute: int | None = None,
    ):
        self.app = app
        default_public, default_admin = get_rate_limits()
        self.requests_per_minute = (
            requests_per_minute or default_public
        )
        self.admin_requests_per_minute = (
            admin_requests_per_minute or default_admin
        )
        self._requests: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._operations = 0
        self._trust_proxy_headers = _env_bool(
            "HITITFINLEX_TRUST_PROXY_HEADERS",
            default=False,
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        is_admin_path = path == "/intake" or path.startswith("/reviews")
        bucket = "admin" if is_admin_path else "public"
        limit = (
            self.admin_requests_per_minute
            if bucket == "admin"
            else self.requests_per_minute
        )
        client = self._client_identifier(scope)
        now = time.monotonic()
        retry_after = 0

        with self._lock:
            timestamps = self._requests[(bucket, client)]
            cutoff = now - 60.0
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= limit:
                retry_after = max(1, int(60.0 - (now - timestamps[0])))
            else:
                timestamps.append(now)
            self._operations += 1
            if self._operations % 256 == 0:
                self._sweep(cutoff)

        if retry_after:
            response = JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _client_identifier(self, scope) -> str:
        if self._trust_proxy_headers:
            headers = {
                key.lower(): value
                for key, value in scope.get("headers", [])
            }
            forwarded = headers.get(b"x-forwarded-for", b"").decode(
                "latin-1"
            )
            if forwarded:
                return forwarded.split(",", 1)[0].strip()
        client = scope.get("client")
        return str(client[0]) if client else "unknown"

    def _sweep(self, cutoff: float) -> None:
        stale_keys = []
        for key, timestamps in self._requests.items():
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if not timestamps:
                stale_keys.append(key)
        for key in stale_keys:
            self._requests.pop(key, None)
