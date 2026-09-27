"""Auth for outbound A2A calls: every auth type as a single httpx.Auth interface.

`build_auth(auth_config)` is the one construction point — it turns any A2AAuthConfig
into an `httpx.Auth` (or None) that both call sites (the runtime client and the config
service) attach via httpx's `auth=`. Routing static and refreshing auth through the SAME
mechanism is deliberate: a token baked into a client's static `headers=` freezes, and the
SDK client is cached for its lifetime, so a 300s Keycloak access token would die on it.

Auth strategies:
  - StaticHeaderAuth — one fixed header (bearer token / API key); the static analogue of
    OAuth2BearerAuth, mirroring httpx's own BasicAuth.
  - OAuth2BearerAuth + OAuth2TokenProvider — fetch a short-lived bearer from any RFC 6749
    token endpoint, cache it until just before expiry, refresh on demand and on a 401.

The OAuth2 side is provider-agnostic: Keycloak / Auth0 / Okta / Azure AD / Google — any
standard token endpoint — given a token URL + client_id + grant. Two headless grants:
  - client_credentials: a confidential client's own identity (preferred for M2M).
  - refresh_token: a previously-obtained (offline) refresh token.

NOT covered (add here if a future agent needs it): mTLS / private_key_jwt client
authentication, token exchange (RFC 8693), DPoP / sender-constrained tokens.
"""
import asyncio
import logging
import os
import time
from typing import Awaitable, Callable, Mapping, Optional

import httpx

from schemas.configuration_schemas import A2AAuthConfig, A2AAuthType

logger = logging.getLogger(__name__)


class OAuth2TokenError(Exception):
    """Token endpoint was unreachable or rejected the grant (bad creds, expired refresh, ...)."""


class OAuth2TokenProvider:
    """Caches a bearer access token and refreshes it before it expires."""

    # Refresh this many seconds BEFORE the real expiry so an in-flight request never
    # carries an already-dead token (covers clock skew + network latency).
    _EXPIRY_SKEW_SECONDS = 30
    # Assumed cache lifetime when the server OMITS expires_in entirely (RFC 6749 makes it
    # optional). A present-but-low value is honored as-is and can floor to 0 → re-fetch
    # (see test_short_ttl_refetches_each_call); it is not raised to this.
    _MIN_TTL_SECONDS = 30

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        grant_type: str,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        scope: Optional[str] = None,
        timeout_seconds: float = 20.0,
        on_refresh_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._token_url = token_url
        self._client_id = client_id
        self._grant_type = grant_type
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._scope = scope
        self._timeout = timeout_seconds
        self._on_refresh_token = on_refresh_token
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0  # monotonic() deadline, already skew-adjusted
        self._lock = asyncio.Lock()

    async def get_access_token(self, *, force: bool = False) -> str:
        # monotonic() is immune to wall-clock jumps (NTP / DST) that could otherwise
        # mark a fresh token expired or keep a stale one alive.
        if not force and self._access_token and time.monotonic() < self._expires_at:
            return self._access_token
        async with self._lock:
            # Re-check inside the lock: a concurrent caller may have refreshed while we
            # waited. Without this, N racing requests cause N redundant token fetches.
            if not force and self._access_token and time.monotonic() < self._expires_at:
                return self._access_token
            await self._fetch()
            return self._access_token  # set by _fetch or it raised

    async def _fetch(self) -> None:
        data = {"client_id": self._client_id, "grant_type": self._grant_type}
        if self._client_secret:
            data["client_secret"] = self._client_secret
        if self._scope:
            data["scope"] = self._scope
        if self._grant_type == "refresh_token":
            if not self._refresh_token:
                raise OAuth2TokenError("refresh_token grant configured without a refresh token")
            data["refresh_token"] = self._refresh_token

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._token_url, data=data)
        except httpx.HTTPError as exc:
            raise OAuth2TokenError(f"token endpoint unreachable: {exc}") from exc

        if resp.status_code != 200:
            # Surface Keycloak's error/description (e.g. invalid_grant = expired refresh
            # token) without echoing the request body.
            detail = f": HTTP {resp.status_code}"
            try:
                body = resp.json()
                detail = f": {body.get('error')} {body.get('error_description', '')}".rstrip()
            except Exception:
                pass
            raise OAuth2TokenError(f"token request failed{detail}")

        tok = resp.json()
        access = tok.get("access_token")
        if not access:
            raise OAuth2TokenError("token response contained no access_token")
        self._access_token = access
        # Cache until the skew window before real expiry. A present expires_in is honored (a
        # sub-skew value floors to 0 → re-fetch); an omitted one falls back to _MIN_TTL_SECONDS
        # AFTER the skew. The old `or _MIN_TTL` sat BEFORE the subtraction, so 30-30 cancelled
        # to 0 and the omit-case default cached nothing — a fetch on every request.
        raw_ttl = tok.get("expires_in")
        if raw_ttl:
            self._expires_at = time.monotonic() + max(int(raw_ttl) - self._EXPIRY_SKEW_SECONDS, 0)
        else:
            self._expires_at = time.monotonic() + self._MIN_TTL_SECONDS

        # Keycloak rotates the refresh token on each use when "Revoke Refresh Token" is
        # enabled; we must keep the newest or the next refresh fails with invalid_grant.
        # Persist it (callback) so a process restart doesn't fall back to a dead token.
        new_rt = tok.get("refresh_token")
        if new_rt and new_rt != self._refresh_token:
            self._refresh_token = new_rt
            if self._on_refresh_token:
                try:
                    await self._on_refresh_token(new_rt)
                except Exception:
                    logger.exception("Failed to persist rotated refresh token")

        logger.info(
            "OAuth2 token refreshed (grant=%s, ttl=%ss) from %s",
            self._grant_type, tok.get("expires_in"), self._token_url,
        )


class StaticHeaderAuth(httpx.Auth):
    """Injects one fixed header on every request — a bearer token or an API key.

    The static analogue of OAuth2BearerAuth: same ``auth=`` mechanism, no refresh. Mirrors
    httpx's own BasicAuth — override the sync ``auth_flow`` (setting a header does no I/O), so
    it works on sync and async clients alike and is safe to share across concurrent requests
    (immutable; httpx builds a fresh flow generator per request).
    """

    def __init__(self, header_name: str, header_value: str):
        self._header_name = header_name
        self._header_value = header_value

    def auth_flow(self, request):
        request.headers[self._header_name] = self._header_value
        yield request


class OAuth2BearerAuth(httpx.Auth):
    """Injects a fresh bearer token per request; refreshes once on a 401 and retries.

    Attached to the httpx client via ``auth=`` (NOT baked into static headers), so a
    rotated token always takes effect. This is the fix for the frozen-cached-header
    problem: a client built once otherwise keeps a dead Authorization header forever.
    """

    # No requires_response_body flag: httpx only honors it inside the base async_auth_flow,
    # which we override here, and we read response.status_code only (never the body). httpx
    # itself reads the 401 body before replaying the retry, so nothing is left unread.

    def __init__(self, provider: OAuth2TokenProvider, extra_headers: Optional[Mapping[str, str]] = None):
        self._provider = provider
        self._extra_headers = dict(extra_headers or {})

    async def async_auth_flow(self, request):
        request.headers.update(self._extra_headers)
        request.headers["Authorization"] = f"Bearer {await self._provider.get_access_token()}"
        response = yield request
        if response.status_code == 401:
            # Token may have been revoked before its TTL (admin logout, key rotation).
            # Force a single refresh and replay once; a second 401 propagates as-is.
            request.headers["Authorization"] = (
                f"Bearer {await self._provider.get_access_token(force=True)}"
            )
            yield request


def _resolve_secret(direct: Optional[str], env_name: Optional[str]) -> Optional[str]:
    """Prefer a directly-stored secret, else read the named backend env var."""
    if direct:
        return direct
    if env_name:
        return os.getenv(env_name)
    return None


def validate_auth_env_vars(auth_config: A2AAuthConfig) -> None:
    """Reject a config whose secret env var is unset — at SAVE time, not mid-run.

    oauth2 with client_secret_env would otherwise construct fine and only fail 5 minutes
    later when the token endpoint is first called. Checked only when the env var is the
    actual source: a directly-stored token/secret takes precedence in build_auth and makes
    the env var irrelevant. Raises ValueError; callers translate (HTTP 400 in the config
    APIs). Shared by the A2A config service and the external-MCP save/discover paths.
    """
    if (auth_config.type in [A2AAuthType.BEARER, A2AAuthType.API_KEY]
            and auth_config.token_env and not auth_config.token
            and not os.getenv(auth_config.token_env)):
        raise ValueError(
            f"auth env var {auth_config.token_env} is not set on the backend process"
        )
    if auth_config.type == A2AAuthType.OAUTH2:
        for env_name, direct in (
            (auth_config.client_secret_env, auth_config.client_secret),
            (auth_config.refresh_token_env, auth_config.refresh_token),
            (auth_config.token_env, auth_config.token),
        ):
            if env_name and not direct and not os.getenv(env_name):
                raise ValueError(
                    f"auth env var {env_name} is not set on the backend process"
                )


def build_oauth2_provider(
    auth_config: A2AAuthConfig,
    *,
    on_refresh_token: Optional[Callable[[str], Awaitable[None]]] = None,
) -> OAuth2TokenProvider:
    """Construct a provider from an ``oauth2`` A2AAuthConfig (validated upstream)."""
    grant = getattr(auth_config.grant_type, "value", auth_config.grant_type)
    return OAuth2TokenProvider(
        token_url=auth_config.token_url,
        client_id=auth_config.client_id,
        grant_type=grant,
        client_secret=_resolve_secret(auth_config.client_secret, auth_config.client_secret_env),
        refresh_token=_resolve_secret(auth_config.refresh_token, auth_config.refresh_token_env),
        scope=auth_config.scope,
        on_refresh_token=on_refresh_token,
    )


def build_auth(
    auth_config: A2AAuthConfig,
    *,
    provider_factory: Optional[Callable[[], OAuth2TokenProvider]] = None,
) -> Optional[httpx.Auth]:
    """Turn any A2AAuthConfig into an httpx.Auth (or None for 'none') — the single auth
    construction point shared by the runtime client and the config service.

    Raises ValueError on missing/invalid credentials; callers translate it to their own
    error type (A2AAuthError in the runtime client, HTTP 400 in the config API), so this
    helper stays free of either dependency.

    provider_factory lets the runtime client inject its per-server *cached* OAuth2 provider;
    the config service omits it and gets a fresh provider (admin calls are rare).
    """
    auth_type = auth_config.type
    if auth_type == "none":
        return None
    if auth_type == "bearer":
        token = _require_token(auth_config, "Bearer auth")
        return StaticHeaderAuth("Authorization", f"Bearer {token}")
    if auth_type == "api_key":
        token = _require_token(auth_config, "API key auth")
        return StaticHeaderAuth(auth_config.header_name or "X-API-Key", token)
    if auth_type == "oauth2":
        extra_headers = {}
        if auth_config.header_name:
            extra_headers[auth_config.header_name] = _require_token(auth_config, "oauth2 header_name")
        provider = provider_factory() if provider_factory else build_oauth2_provider(auth_config)
        return OAuth2BearerAuth(provider, extra_headers=extra_headers)
    return None


def _require_token(auth_config: A2AAuthConfig, label: str) -> str:
    token = _resolve_secret(auth_config.token, auth_config.token_env)
    if not token:
        raise ValueError(
            f"auth env var {auth_config.token_env} is not set on the backend process"
            if auth_config.token_env else f"{label} requires a token or token_env"
        )
    return token
