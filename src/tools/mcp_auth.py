"""Outbound auth for external MCP servers — reuses the A2A token provider (AppFactory-313).

External MCP configs carry the same ``auth`` block as A2A (``A2AAuthConfig``). This module
is the single point that turns that block into an ``httpx.Auth`` for the MCP HTTP transports,
caching ONE OAuth2 token provider per ``(tenant, server, auth-revision)`` so that discovery,
health-check and every tool call to a server share one short-lived access token — one request
to the token endpoint per TTL, not per call.

The provider itself is ``a2a_auth.OAuth2TokenProvider`` built by ``a2a_auth.build_oauth2_provider``
— NOT a copy — so A2A and MCP fetch, cache, refresh, and retry-on-401 identically. MCP has no
long-lived client object (each discover/health/call builds a fresh ``ExternalMCPClient``), so the
cache lives here at module scope instead of on a client instance the way A2A's does.

Scope note: the token-persistence callback (``on_refresh_token``) is intentionally omitted. The
urban/IDU use case is ``client_credentials``, which has no refresh token to rotate. A
``refresh_token`` grant still works in-process (the provider keeps the newest rotated token for
its own lifetime) but MCP does not write it back to the config, so after a restart it would fall
back to the stored one — acceptable because refresh_token is not the MCP use case here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional, Union

import httpx

from integrations.a2a_auth import build_auth, build_oauth2_provider, OAuth2TokenProvider
from schemas.configuration_schemas import A2AAuthConfig, A2AAuthType

# One provider per "tenant:server:auth-revision". The revision (a hash of the auth block)
# means an edit to the credentials starts a fresh provider instead of serving a token minted
# for the old ones; the stale entry is harmless and cleared on process restart.
_providers: Dict[str, OAuth2TokenProvider] = {}


def _cache_key(tenant_id: str, server_id: str, auth_config: A2AAuthConfig) -> str:
    data = auth_config.model_dump(mode="json", exclude_none=True)
    # refresh_token rotates in-process (see module note); excluding it keeps rotation from
    # recreating the provider and discarding its live access token — mirrors a2a_client.
    data.pop("refresh_token", None)
    revision = hashlib.sha256(
        json.dumps(data, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{tenant_id}:{server_id}:{revision}"


def _cached_provider(cache_key: str, auth_config: A2AAuthConfig) -> OAuth2TokenProvider:
    # Synchronous get-or-create: no await between the check and the insert, so concurrent
    # coroutines on one event loop cannot each mint a provider (matches a2a_client's unlocked
    # cache). The provider's own asyncio.Lock then collapses the token-fetch stampede.
    provider = _providers.get(cache_key)
    if provider is None:
        provider = build_oauth2_provider(auth_config)
        _providers[cache_key] = provider
    return provider


def build_mcp_auth(
    auth: Optional[Union[A2AAuthConfig, dict]],
    *,
    tenant_id: str,
    server_id: str,
) -> Optional[httpx.Auth]:
    """Turn an MCP server's ``auth`` block into an ``httpx.Auth`` (or None) for the transport.

    Accepts the parsed ``A2AAuthConfig`` (from a discover request) or the raw dict (from a
    stored ``metadata.external_mcp.auth``). Returns None for a missing/``none`` auth block so
    no token is ever fetched (contract 7). For ``oauth2`` the returned auth carries this
    server's *cached* provider so discovery, health and calls share one token; static
    ``bearer``/``api_key`` build a header directly with no provider.

    Raises ``pydantic.ValidationError`` if a raw dict is malformed, or ``ValueError`` if
    credentials are missing — callers translate to their own error (HTTP 400 / tool error).
    """
    if not auth:
        return None
    auth_config = auth if isinstance(auth, A2AAuthConfig) else A2AAuthConfig(**auth)
    if auth_config.type == A2AAuthType.NONE:
        return None
    cache_key = _cache_key(tenant_id, server_id, auth_config)
    return build_auth(
        auth_config,
        provider_factory=lambda: _cached_provider(cache_key, auth_config),
    )
