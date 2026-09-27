import base64
import copy
import hashlib
import json
import logging
from dataclasses import replace
from datetime import datetime
from typing import Dict, Any, AsyncIterator, List
from urllib.parse import urlparse
from uuid import uuid4

from a2a.client import ClientCallContext, ClientConfig, ClientFactory
from a2a.client.card_resolver import parse_agent_card
from a2a.compat.v0_3.jsonrpc_transport import CompatJsonRpcTransport
from a2a.utils.constants import TransportProtocol
from a2a.utils.errors import MethodNotFoundError
from a2a.types import (
    Message,
    Part,
    SendMessageRequest,
    Task,
    GetTaskRequest,
    CancelTaskRequest,
    StreamResponse,
    Role,
    TaskState,
)
import httpx
from google.protobuf.struct_pb2 import DESCRIPTOR
from google.protobuf.message_factory import GetMessageClass
from google.protobuf.json_format import Parse, ParseDict, ParseError, MessageToDict
from schemas.configuration_schemas import A2AAuthConfig
from integrations.a2a_auth import build_auth, build_oauth2_provider
from integrations.a2a_tracing import build_call_context
from integrations.a2a_params import (
    EXTENSIONS_ACTIVATION_HEADER,
    extension_activation_header_value,
)
from integrations.a2a_contracts import (
    A2AContractError,
    A2AContractSnapshot,
    url_origin,
    validate_agent_card,
)
from integrations.a2a_output_modes import (
    A2AOutputModeError,
    resolve_declared_output_modes,
    supported_declared_output_modes,
    validate_event_output_modes,
)
from storage.mongo_backend import MongoStorageBackend

from typing import TypeVar

T = TypeVar('T', bound=Exception)

logger = logging.getLogger(__name__)


class A2AAuthError(Exception):
    pass


class A2AServerNotFoundError(Exception):
    pass


class A2AServerDisabledError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# BEGIN A2A NON-CONFORMANCE WORKAROUND  —  isolated concession, safe to remove
# ═══════════════════════════════════════════════════════════════════════════════
# WHAT: some A2A agents return responses that violate the spec and crash the SDK's
#   strict v0.3 validator/converter. To keep runs working we repair the raw response
#   in a custom transport BEFORE the SDK parses it. This is a concession to MALFORMED
#   agents, not desired behavior — the correct fix is upstream (the agent).
# NOTIFICATION: every repair is logged at WARNING ("[A2A NON-CONFORMANCE] ...") naming
#   the agent and the violation, so operators know an agent is malformed.
# UPSTREAM: the requested agent-side fixes are tracked in a2a_idu_message_ru.md.
# TO DISABLE: set A2A_LENIENT_COMPAT_ENABLED = False — malformed agents then fail loudly
#   at SDK validation (strict mode), surfacing the non-conformance instead of hiding it.
# TO REMOVE ENTIRELY (once agents are conformant): delete this whole block AND the
#   matching "A2A NON-CONFORMANCE WORKAROUND" block in _get_sdk_client.
# KNOWN VIOLATIONS handled (as of 2026-06-26, IDU restriction-creation-agent):
#   1. Message objects in history / status.message missing required `messageId` (A2A §Message)
#   2. status.timestamp without a timezone offset (breaks protobuf Timestamp parsing)
# SAFETY: repairs are best-effort and never raise — a workaround must not turn a
#   response the SDK could handle into a failure (see _send_request below).
# ═══════════════════════════════════════════════════════════════════════════════

A2A_LENIENT_COMPAT_ENABLED = True


def _ensure_timezone(ts: str) -> str:
    """Append 'Z' (UTC) to an ISO timestamp that carries no timezone; leave others as-is.

    protobuf's Timestamp.FromJsonString (used by the SDK to convert status.timestamp)
    requires an RFC3339 offset. Given a TZ-less value its parser does rfind('-'), which
    hits the date's own hyphen and truncates the string, raising. Assuming UTC for a
    naive timestamp is harmless here — we don't act on this value — and avoids failing
    the whole run over status metadata.
    """
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    return ts if parsed.tzinfo is not None else ts + "Z"


_JS_SAFE_INT = 2 ** 53


def _preserve_large_ints(obj: Any) -> Any:
    """Stringify integers beyond ±2^53 so they survive a DataPart round-trip losslessly.

    A protobuf ``Value`` holds every number as a float64, so an int past 2^53 would be
    silently rounded on the wire. JSON/JS can't represent such an id as a number either,
    so stringifying is the sane lossless form. Smaller ints are left untouched — they map
    to float64 exactly and stay JSON numbers, so this changes nothing for the common case.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int) and abs(obj) > _JS_SAFE_INT:
        return str(obj)
    if isinstance(obj, dict):
        return {k: _preserve_large_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_preserve_large_ints(v) for v in obj]
    return obj


def _normalize_a2a_response(result: Any) -> List[str]:
    """Repair spec-noncompliant fields in an agent's response, in place.

    Returns a list of human-readable descriptions of the violations repaired (empty when
    the response was already conformant), so the caller can log the agent's non-conformance.

    Two real-world A2A v0.3 violations break the SDK's strict validation/conversion of an
    otherwise-valid response:
      1. Message objects missing the required `messageId` — notably the user message echoed
         back into `history`. `Task.model_validate` rejects these.
      2. `status.timestamp` without a timezone — see _ensure_timezone.
    We consume neither field (results arrive as artifacts), so repair rather than fail.
    """
    violations: List[str] = []
    if not isinstance(result, dict):
        return violations

    candidates: List[Any] = []
    if isinstance(result.get("history"), list):
        candidates.extend(result["history"])
    status = result.get("status")
    if isinstance(status, dict) and isinstance(status.get("message"), dict):
        candidates.append(status["message"])
    # The result itself may be a bare Message rather than a Task.
    if result.get("role") and isinstance(result.get("parts"), list):
        candidates.append(result)

    missing_ids = 0
    for msg in candidates:
        if isinstance(msg, dict) and msg.get("role") and not msg.get("messageId"):
            msg["messageId"] = str(uuid4())
            missing_ids += 1
    if missing_ids:
        violations.append(f"{missing_ids} Message object(s) missing required 'messageId'")

    if isinstance(status, dict) and isinstance(status.get("timestamp"), str):
        fixed = _ensure_timezone(status["timestamp"])
        if fixed != status["timestamp"]:
            violations.append(f"status.timestamp {status['timestamp']!r} has no timezone (not RFC3339)")
            status["timestamp"] = fixed

    return violations


class _LenientCompatJsonRpcTransport(CompatJsonRpcTransport):
    """v0.3 compat transport that repairs noncompliant responses before validation.

    The SDK validates the raw JSON-RPC `result` against strict Pydantic models; agents
    that drop the required `messageId` (see _normalize_a2a_response) would otherwise fail
    every call. We hook the single unary choke point (`_send_request`) shared by
    send_message/get_task/etc., normalize the payload, and log any repair so the agent's
    non-conformance is visible rather than silently absorbed.
    """

    async def _send_request(self, json_data, context=None):
        response_data = await super()._send_request(json_data, context)
        # A repair must NEVER turn a working response into a failure: swallow any error
        # and fall back to the raw response (the SDK then validates it as it normally would).
        try:
            if isinstance(response_data, dict):
                violations = _normalize_a2a_response(response_data.get("result"))
                if violations:
                    logger.warning(
                        "[A2A NON-CONFORMANCE] agent at %s returned a spec-violating response; "
                        "applied compatibility repairs: %s. Fix the agent (see a2a_idu_message_ru.md); "
                        "set A2A_LENIENT_COMPAT_ENABLED=False to enforce strict mode.",
                        self.url, "; ".join(violations),
                    )
        except Exception:
            logger.exception("[A2A NON-CONFORMANCE] response normalization failed for %s", self.url)
        return response_data

# ═══════════════════════════════════════════════════════════════════════════════
# END A2A NON-CONFORMANCE WORKAROUND
# ═══════════════════════════════════════════════════════════════════════════════


# Signals meaning "this agent's card advertised message/stream but the
# endpoint does not implement it" — raised by the SDK before the first stream event,
# so no task was created and re-sending as message/send cannot double-submit. The
# compat transport maps JSON-RPC -32601 to MethodNotFoundError and wraps a streaming
# HTTP 404/405/501 in A2AClientError chained (`raise ... from`) off the httpx error.
_STREAM_UNSUPPORTED_HTTP_STATUS = (404, 405, 501)


def _stream_method_unsupported(exc: BaseException) -> bool:
    if isinstance(exc, MethodNotFoundError):
        return True
    cause = exc.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        return cause.response.status_code in _STREAM_UNSUPPORTED_HTTP_STATUS
    return False


def _describe_stream_unsupported(exc: BaseException) -> str:
    cause = exc.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        return f"HTTP {cause.response.status_code}"
    return type(exc).__name__


class A2AClient:
    _CANCEL_FIXED_TIMEOUT: float = 5.0
    _AGENT_CARD_ROUTE: str = "/.well-known/agent-card.json"

    def __init__(self, storage: MongoStorageBackend):
        self.storage = storage
        self._http_client: httpx.AsyncClient | None = None
        self._sdk_clients: Dict[str, Any] = {}
        # One OAuth2 token provider per "tenant:server" — see _get_token_provider.
        self._token_providers: Dict[str, Any] = {}

    def _get_token_provider(
        self,
        server_id: str,
        tenant_id: str,
        auth_config: A2AAuthConfig,
        revision: str = "",
        *,
        expected_updated_at: datetime | None = None,
    ):
        """Cache one OAuth2 token provider per server so the access-token cache and any
        rotated refresh token are shared across every call path (SDK, batch, card fetch)."""
        cache_key = f"{tenant_id}:{server_id}:{revision}"
        provider = self._token_providers.get(cache_key)
        if provider is None:
            async def _persist_rotated_refresh_token(new_rt: str) -> None:
                # Keycloak may rotate the refresh token on each use; persist the newest so a
                # restart doesn't fall back to a revoked one. This must not advance the
                # configuration revision: preflight uses that revision for its CAS write.
                try:
                    updated = await self.storage.update_a2a_server_rotated_refresh_token(
                        server_id,
                        tenant_id,
                        new_rt,
                        expected_updated_at=expected_updated_at,
                    )
                    if not updated:
                        logger.warning(
                            "[A2A_AUTH] server_id=%s — skipped rotated refresh token "
                            "because configuration changed concurrently",
                            server_id,
                        )
                except Exception:
                    logger.exception("Failed to persist rotated refresh token for '%s'", server_id)

            provider = build_oauth2_provider(
                auth_config, on_refresh_token=_persist_rotated_refresh_token
            )
            self._token_providers[cache_key] = provider
        return provider

    @staticmethod
    def _server_connection_revision(server: Dict[str, Any]) -> str:
        """Stable cache revision for fields that affect outbound connectivity."""
        auth = server.get("auth")
        if isinstance(auth, dict):
            auth = dict(auth)
            # OAuth providers update this runtime credential automatically. Including it
            # would recreate the provider and discard its cached access token on every
            # rotation; explicit auth edits still invalidate through the remaining fields.
            auth.pop("refresh_token", None)
        connection = {
            key: server.get(key)
            for key in (
                "endpoint_url",
                "url",
                "agent_card_url",
                "rpc_endpoint",
                "request_timeout_seconds",
                "enabled",
                "use_a2a_streaming",
            )
        }
        connection["auth"] = auth
        encoded = json.dumps(
            connection, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _sdk_client_revision(connection_revision: str, card_hash: str) -> str:
        """Bind an SDK client to both connection settings and one Agent Card."""
        encoded = f"{connection_revision}:{card_hash}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _streaming_gate(config: Dict[str, Any], agent_card: Dict[str, Any]) -> bool:
        """Whether to open message/stream for this server: the per-server opt-in
        (or the coscientist transition literal) AND the card actually advertising
        streaming.
        """
        capabilities = agent_card.get("capabilities") or {}
        if not capabilities.get("streaming"):
            return False
        opt_in = config.get("use_a2a_streaming")
        if opt_in is not None:
            return bool(opt_in)
        # TRANSITION: a legacy server that never saved the flag keeps streaming only
        # when its card is coscientist, until re-discovery/backfill sets
        # use_a2a_streaming; then delete this and the literal disappears. An explicit
        # false above wins, so an operator can always force coscientist to message/send.
        return agent_card.get("name") == "coscientist"

    async def _get_server_config(
        self,
        server_id: str,
        tenant_id: str,
        *,
        server: Dict[str, Any] | None = None,
        revision: str = "",
    ) -> Dict[str, Any]:
        """Get server configuration from registry."""
        if server is None:
            server = await self.storage.get_a2a_server(server_id, tenant_id)
        if not server:
            raise A2AServerNotFoundError(
                f"A2A server '{server_id}' not found for tenant '{tenant_id}'"
            )
        # Gate every operational path (send / tasks / node) on the registry toggle.
        # Admin validate/refresh use the service's own fetch, not this client, so a
        # disabled server can still be re-validated before being re-enabled.
        if not server.get("enabled", True):
            raise A2AServerDisabledError(
                f"A2A server '{server_id}' is disabled; enable it before sending requests"
            )
        auth_data = server.get("auth")
        revision = revision or self._server_connection_revision(server)
        headers = {}
        auth_obj = None

        # Activate any required Profile Extension the agent advertises (e.g. IDU's
        # scenario-context). Per the A2A spec the client must list the extension URIs in
        # the A2A-Extensions header; without it the agent may ignore our structured
        # DataPart and fall back to parsing the intent text. Driven by the cached card so
        # it stays in lockstep with the DataPart we build (a2a_params, same gate).
        activation = extension_activation_header_value(
            (server.get("cached_agent_card_summary") or {}).get("capabilities")
        )
        if activation:
            headers[EXTENSIONS_ACTIVATION_HEADER] = activation
        if auth_data:
            try:
                auth_config = A2AAuthConfig(**auth_data)
                # Every auth type becomes an httpx.Auth attached via auth= (never frozen into
                # the cached client's static headers) — see a2a_auth.build_auth. OAuth2 reuses
                # this server's cached provider so the access + rotated refresh token are shared.
                auth_obj = build_auth(
                    auth_config,
                    provider_factory=lambda: self._get_token_provider(
                        server_id,
                        tenant_id,
                        auth_config,
                        revision,
                        expected_updated_at=server.get("updated_at"),
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to resolve auth for server '{server_id}': {e}")
                raise A2AAuthError(f"Invalid auth configuration: {e}")

        endpoint_url = server.get("endpoint_url") or server.get("url")
        if not endpoint_url:
            raise A2AServerNotFoundError(
                f"A2A server '{server_id}' has no endpoint_url or url field"
            )

        rpc_endpoint = server.get("rpc_endpoint", "/")
        if not rpc_endpoint.startswith('/'):
            rpc_endpoint = '/' + rpc_endpoint

        base_url = endpoint_url.rstrip('/')

        return {
            "url": base_url,
            "headers": headers,
            "auth": auth_obj,
            "request_timeout_seconds": server.get("request_timeout_seconds", 60),
            "rpc_endpoint": rpc_endpoint,
            "agent_card_url": server.get("agent_card_url"),
            # AppFactory-280: opt-in per-server switch from "hold one message/send call
            # open for the task's whole lifetime" to "submit once, then poll tasks/get"
            # — needed for multi-hour external MAS runs (a single long-held HTTP call
            # is exactly the fragile shape the reconnect bug comes from) and a
            # prerequisite for surviving a backend restart mid-task. Defaults False so
            # every existing a2a server keeps today's behavior unchanged.
            "long_running": bool(server.get("long_running", False)),
            "poll_interval_seconds": server.get("poll_interval_seconds", 15),
            # Per-server opt-in read by _streaming_gate. Absent (None) is kept
            # distinct from an explicit false: only a missing flag falls through to the
            # coscientist transition literal, so an operator's explicit opt-out is
            # honored even for coscientist.
            "use_a2a_streaming": server.get("use_a2a_streaming"),
            "checkpoints": server.get("checkpoints") or {
                "enabled": False,
                "callback_nonce_env": None,
            },
        }

    async def get_server_config(self, server_id: str, tenant_id: str) -> Dict[str, Any]:
        """Public accessor for a server's operational config (timeouts, polling mode).

        For callers that need to branch on it BEFORE deciding how to talk to the
        server at all (e.g. the workflow engine choosing send_message vs. the
        submit-and-poll path) — send_message/tasks_get resolve this internally
        for their own use, but had no external accessor until now.
        """
        return await self._get_server_config(server_id, tenant_id)

    async def _get_output_contract(
        self, server_id: str, tenant_id: str
    ) -> A2AContractSnapshot:
        """Load and validate the exact Agent Card governing a runtime response."""
        server = await self.storage.get_a2a_server(server_id, tenant_id)
        if not server:
            raise A2AServerNotFoundError(
                f"A2A server '{server_id}' not found for tenant '{tenant_id}'"
            )
        card = server.get("cached_agent_card")
        if not isinstance(card, dict):
            raise A2AContractError(
                "a2a_contract_unavailable",
                f"A2A server '{server_id}' has no validated Agent Card",
            )
        return validate_agent_card(card)

    async def _get_runtime_snapshot(
        self,
        server_id: str,
        tenant_id: str,
        *,
        skill_id: str | None = None,
        force_non_streaming: bool = False,
    ) -> tuple[Dict[str, Any], A2AContractSnapshot, Any]:
        """Bind config, card contract, auth provider, and SDK client to one DB revision.

        ``force_non_streaming`` (Tier-1 fallback): build/reuse a
        message/send-only sibling of the SDK client — cached under a distinct
        revision so it never collides with the streaming client for the same server.
        """
        server = await self.storage.get_a2a_server(server_id, tenant_id)
        connection_revision = self._server_connection_revision(server or {})
        config = await self._get_server_config(
            server_id,
            tenant_id,
            server=server,
            revision=connection_revision,
        )
        card = server.get("cached_agent_card")
        if not isinstance(card, dict):
            raise A2AContractError(
                "a2a_contract_unavailable",
                f"A2A server '{server_id}' has no validated Agent Card",
            )
        contract = validate_agent_card(card)
        if skill_id:
            if not any(
                skill.get("id") == skill_id
                for skill in card.get("skills") or ()
                if isinstance(skill, dict)
            ):
                raise A2AContractError(
                    "a2a_skill_not_found",
                    f"A2A skill '{skill_id}' is not declared by server '{server_id}'",
                    location="agentCard.skills",
                    protocol_version=contract.protocol_version,
                )
            declared_modes = resolve_declared_output_modes(card, skill_id=skill_id)
            contract = replace(
                contract,
                declared_output_modes=declared_modes,
                accepted_output_modes=supported_declared_output_modes(declared_modes),
            )
        # Resolve the streaming decision once, off the same (config, card) the SDK
        # client is built from, and stash it so send_message knows whether a Tier-1
        # fallback even applies without re-reading the card.
        config["a2a_streaming_enabled"] = (
            not force_non_streaming and self._streaming_gate(config, card)
        )
        sdk_revision = self._sdk_client_revision(
            connection_revision, contract.card_hash
        )
        if force_non_streaming:
            sdk_revision = f"{sdk_revision}:nonstream"
        sdk_client = await self._get_sdk_client(
            server_id,
            tenant_id,
            config=config,
            revision=sdk_revision,
            agent_card=card,
            card_hash=contract.card_hash,
            protocol_version=contract.protocol_version,
            force_non_streaming=force_non_streaming,
        )
        return config, contract, sdk_client

    @staticmethod
    def _pin_rpc_to_endpoint(server_id: str, card: Any, config: Dict[str, Any]) -> None:
        """Send JSON-RPC to ``url + rpc_endpoint`` when the card names another scheme, host or port.

        An agent behind a gateway (e.g. a LiteLLM pass-through) still advertises its internal
        address, which we can't reach and which must not receive our credentials.
        """
        endpoint_origin = url_origin(urlparse(config["url"]))
        rpc_url = config["url"].rstrip("/") + (config.get("rpc_endpoint") or "/")
        for interface in card.supported_interfaces:
            advertised_origin = url_origin(urlparse(interface.url))
            if (
                interface.protocol_binding == TransportProtocol.JSONRPC
                and advertised_origin
                and advertised_origin != endpoint_origin
            ):
                logger.info(
                    "[A2A_SDK_CACHE] server_id=%s card advertises %s; calling configured %s",
                    server_id,
                    interface.url,
                    rpc_url,
                )
                interface.url = rpc_url

    async def _get_sdk_client(
        self,
        server_id: str,
        tenant_id: str,
        *,
        config: Dict[str, Any],
        revision: str,
        agent_card: Dict[str, Any],
        card_hash: str,
        protocol_version: str,
        force_non_streaming: bool = False,
    ) -> Any:
        """Get or create SDK client using ClientFactory."""
        cache_key = f"{tenant_id}:{server_id}:{revision}"

        if cache_key not in self._sdk_clients:
            logger.info(
                "[A2A_SDK_CACHE] server_id=%s protocol=%s card_hash=%s revision=%s "
                "— creating SDK client",
                server_id,
                protocol_version or "unknown",
                card_hash[:12] or "unknown",
                revision[:12] or "unversioned",
            )
            http_client = httpx.AsyncClient(
                base_url=config["url"],
                headers=config["headers"],
                auth=config.get("auth"),
                timeout=httpx.Timeout(config["request_timeout_seconds"]),
                follow_redirects=True,
            )

            # Stream when the per-server opt-in is set and the card
            # advertises it (see _streaming_gate); force_non_streaming builds the
            # message/send sibling used by send_message's Tier-1 fallback.
            streaming = (
                False if force_non_streaming
                else self._streaming_gate(config, agent_card)
            )
            client_config = ClientConfig(
                streaming=streaming,
                polling=False,
                httpx_client=http_client,
            )

            factory = ClientFactory(client_config)

            # --- BEGIN A2A NON-CONFORMANCE WORKAROUND (see module-level block; safe to remove) ---
            # Swap the v0.3 compat JSON-RPC transport for our lenient subclass so malformed
            # agents' responses are repaired (and logged) before the SDK validates them.
            # Non-legacy agents keep the default transport. Gated by the module toggle so
            # strict mode can be restored in one line.
            if A2A_LENIENT_COMPAT_ENABLED:
                _default_jsonrpc_producer = factory._registry[TransportProtocol.JSONRPC]

                def _lenient_jsonrpc_producer(card, url, cfg):
                    transport = _default_jsonrpc_producer(card, url, cfg)
                    if isinstance(transport, CompatJsonRpcTransport):
                        return _LenientCompatJsonRpcTransport(
                            transport.httpx_client, transport.agent_card, transport.url
                        )
                    return transport

                factory.register(TransportProtocol.JSONRPC, _lenient_jsonrpc_producer)
            # --- END A2A NON-CONFORMANCE WORKAROUND ---

            try:
                sdk_card = parse_agent_card(copy.deepcopy(agent_card))
                self._pin_rpc_to_endpoint(server_id, sdk_card, config)
                sdk_client = factory.create(sdk_card, interceptors=None)
            except Exception:
                await http_client.aclose()
                raise

            self._sdk_clients[cache_key] = {
                "client": sdk_client,
                "http_client": http_client
            }

        return self._sdk_clients[cache_key]["client"]

    @staticmethod
    def _coerce_message_to_parts(message) -> List[Part]:
        """Build proto Parts from a message.

        Accepts either a plain string (one TextPart — the common case) or a list of
        part dicts: ``{"kind":"text","text":...}`` or ``{"kind":"data","data":{...}}``.
        The caller (workflow node / future delegate tool) decides text vs data per part;
        the client stays agent-agnostic. Unknown kinds are skipped with a warning rather
        than guessed.
        """
        if isinstance(message, str):
            specs = [{"kind": "text", "text": message}]
        else:
            specs = message or []

        parts: List[Part] = []
        for spec in specs:
            # A caller can hand us a list with a bare string element (the /send-message
            # endpoint body is untyped Any); skip it instead of raising AttributeError on
            # .get, which would abort the whole message.
            if not isinstance(spec, dict):
                logger.warning("Skipping non-dict A2A message part: %r", spec)
                continue
            kind = spec.get("kind", "text")
            part = Part()
            if kind == "text":
                part.text = spec.get("text", "")
            elif kind == "data":
                # part.data is a google.protobuf.Value; ParseDict sets the oneof to
                # `data` and round-trips via MessageToDict in _serialize_parts. But
                # shared_context can hold datetimes/ObjectIds (Mongo round-trip), which
                # ParseDict rejects — so first coerce to JSON-native (default=str
                # stringifies the stragglers). If even that fails, skip this part with a
                # warning rather than letting one bad value fail the whole node.
                # protobuf Value stores every number as float64, so ints past 2^53 would
                # lose precision silently; stringify only those (JSON/JS can't hold them as
                # numbers either) so the exact value survives. Smaller ints ride as-is —
                # they round-trip through float64 exactly and stay numbers on the wire.
                try:
                    safe = _preserve_large_ints(
                        json.loads(json.dumps(spec.get("data", {}), default=str))
                    )
                    ParseDict(safe, part.data)
                except (TypeError, ValueError, ParseError) as exc:
                    logger.warning("Skipping un-serializable A2A data part: %s", exc)
                    continue
            else:
                logger.warning("Skipping A2A message part with unsupported kind '%s'", kind)
                continue
            parts.append(part)
        return parts

    async def _iterate_send(
        self,
        sdk_client,
        request: SendMessageRequest,
        call_context: ClientCallContext,
        *,
        contract: A2AContractSnapshot,
        tenant_id: str,
        server_id: str,
        skill_id: str | None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Drive one SDK send_message (streaming or not): validate output modes and
        persist the task contract for each event before yielding it. Shared by the
        streaming attempt and the message/send fallback so both honor the contract."""
        async for stream_response in sdk_client.send_message(request, context=call_context):
            event = self._serialize_stream_response(stream_response)
            try:
                validate_event_output_modes(
                    event,
                    contract.declared_output_modes,
                    protocol_version=contract.protocol_version,
                )
            except A2AOutputModeError as exc:
                logger.error(
                    "[A2A_CONTRACT] server_id=%s error_type=%s location=%s — %s",
                    server_id,
                    exc.error_type,
                    exc.location,
                    exc,
                )
                raise
            await self._persist_task_contract(
                event=event,
                tenant_id=tenant_id,
                server_id=server_id,
                contract=contract,
                skill_id=skill_id,
            )
            yield event

    async def send_message(
            self,
            server_id: str,
            tenant_id: str,
            message,
            context_id: str | None = None,
            task_id: str | None = None,
            metadata: Dict[str, Any] | None = None,
            skill_id: str | None = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Send a message via the A2A SDK (message/send).

        ``message`` is a plain string (one TextPart) or a list of part dicts (see
        _coerce_message_to_parts). Per-server timeout from config.
        """
        config, contract, sdk_client = await self._get_runtime_snapshot(
            server_id, tenant_id, skill_id=skill_id
        )

        # task_id rides on proto STRING fields (message_id, task_id); a numeric task_id
        # from a raw request body would raise TypeError. Coerce truthy non-strings while
        # preserving the falsy -> generated-id behavior (None/0/"" still fall back).
        if task_id and not isinstance(task_id, str):
            task_id = str(task_id)
        if task_id:
            contract = await self._get_task_contract(
                tenant_id=tenant_id,
                server_id=server_id,
                task_id=task_id,
                fallback=contract,
                expected_skill_id=skill_id,
            )

        request = self._build_send_message_request(
            tenant_id=tenant_id,
            message=message,
            context_id=context_id,
            task_id=task_id,
            metadata=metadata,
            accepted_output_modes=contract.accepted_output_modes,
            return_immediately=False,
        )

        call_context: ClientCallContext = build_call_context(
            timeout=float(config["request_timeout_seconds"])
        )

        if not config.get("a2a_streaming_enabled"):
            async for event in self._iterate_send(
                sdk_client, request, call_context,
                contract=contract, tenant_id=tenant_id,
                server_id=server_id, skill_id=skill_id,
            ):
                yield event
            return

        # Streaming is on. Try message/stream; if the card advertised a stream the
        # agent does not actually implement, the SDK raises a method-unsupported
        # signal BEFORE the first event (no task created) — only then re-send the
        # SAME request (same message_id) as message/send. An error after the first
        # event, or any non-method-unsupported error, propagates: a blind retry there
        # could double-submit an already-created task.
        yielded_any = False
        try:
            async for event in self._iterate_send(
                sdk_client, request, call_context,
                contract=contract, tenant_id=tenant_id,
                server_id=server_id, skill_id=skill_id,
            ):
                yielded_any = True
                yield event
            return
        except Exception as exc:
            if yielded_any or not _stream_method_unsupported(exc):
                raise
            logger.warning(
                "[A2A_STREAM_FALLBACK] server_id=%s — agent advertised streaming but "
                "message/stream is unsupported (%s); re-sending via message/send",
                server_id,
                _describe_stream_unsupported(exc),
            )

        _, _, nonstream_client = await self._get_runtime_snapshot(
            server_id, tenant_id, skill_id=skill_id, force_non_streaming=True
        )
        async for event in self._iterate_send(
            nonstream_client, request, call_context,
            contract=contract, tenant_id=tenant_id,
            server_id=server_id, skill_id=skill_id,
        ):
            yield event

    async def _send_once_and_track(
            self,
            *,
            request: SendMessageRequest,
            config: Dict[str, Any],
            contract,
            sdk_client,
            server_id: str,
            tenant_id: str,
            skill_id: str | None,
            context_id: str | None,
            timeout_seconds: int | None,
    ) -> Dict[str, Any]:
        """One non-streaming message/send round trip (the request must already have
        configuration.return_immediately=True) — shared by submit_and_track (new
        task) and continue_task (AppFactory-281 P1 review fix: answering
        input_required without blocking until the task reaches a terminal state).

        Validates output modes, persists the task contract, and returns
        {"task_id": str | None, "context_id": str | None, "event": <serialized
        event>}. task_id is None only for a bare Message reply — nothing to poll,
        the caller should treat `event` as already terminal.
        """
        timeout = timeout_seconds or config["request_timeout_seconds"]
        call_context = build_call_context(timeout=float(timeout))

        event: Dict[str, Any] | None = None
        async for stream_response in sdk_client.send_message(request, context=call_context):
            event = self._serialize_stream_response(stream_response)
            try:
                validate_event_output_modes(
                    event,
                    contract.declared_output_modes,
                    protocol_version=contract.protocol_version,
                )
            except A2AOutputModeError as exc:
                logger.error(
                    "[A2A_CONTRACT] server_id=%s error_type=%s location=%s — %s",
                    server_id,
                    exc.error_type,
                    exc.location,
                    exc,
                )
                raise
            break  # non-streaming send_message always yields exactly one item

        if event is None:
            raise A2AContractError(
                "a2a_submit_no_response",
                f"A2A server '{server_id}' returned no response to message/send",
                location="a2aTaskSubmit",
                protocol_version=contract.protocol_version,
                declared_modes=contract.declared_output_modes,
            )

        task_id = self._event_task_id(event)
        if not task_id:
            logger.info(
                "[A2A_RECOVER] server_id=%s send returned a bare message "
                "(no task id) — nothing to poll, treat as already complete",
                server_id,
            )
            return {"task_id": None, "context_id": context_id, "event": event}

        await self._persist_task_contract(
            event=event,
            tenant_id=tenant_id,
            server_id=server_id,
            contract=contract,
            skill_id=skill_id,
        )

        response_context_id = (event.get("data") or {}).get("context_id") or context_id
        return {"task_id": task_id, "context_id": response_context_id, "event": event}

    async def submit_and_track(
            self,
            server_id: str,
            tenant_id: str,
            message,
            context_id: str | None = None,
            metadata: Dict[str, Any] | None = None,
            skill_id: str | None = None,
            timeout_seconds: int | None = None,
            message_id: str | None = None,
    ) -> Dict[str, Any]:
        """Submit a NEW task and return as soon as the server acknowledges it —
        does NOT wait for a terminal state. Entry point for the submit-once,
        then-poll model (AppFactory-280 plan §3/§4): the caller persists the
        returned task_id BEFORE it starts polling via tasks_get, so a restart
        or network drop mid-poll has something durable to reconcile against.

        Only for STARTING a new task — never pass a task_id (send_message's
        task_id parameter is for continuing/answering an existing task, e.g.
        input-required; continue_task below is now the recommended way to do
        that without blocking on the answer's own reply). Sets
        configuration.return_immediately=True so the underlying non-streaming
        transport call returns with the task's initial state (typically
        SUBMITTED/WORKING) instead of blocking until terminal — verified
        against the installed a2a-sdk source (BaseClient.send_message always
        does exactly one request/response round-trip when streaming=False,
        which is how this client is configured; return_immediately controls
        how long the SERVER holds that one call open, not how many items the
        client iterates).

        `message_id` (AppFactory-280 finding #2): caller-supplied correlation id,
        persisted BEFORE this call is made. When the caller is retrying a
        submission whose outcome was never durably confirmed (process died
        between the adapter accepting message/send and the task_id being
        recorded), passing the SAME message_id here gives an adapter that
        dedups on it a chance to recognize the retry as the original request
        instead of starting a second task. Not a protocol guarantee — core A2A
        JSON-RPC does not mandate message_id dedup — but strictly better than
        today's behavior of minting a brand-new id on every retry.

        Returns {"task_id": str | None, "context_id": str | None, "event": <serialized event>}.
        task_id is None only if the server answered with a bare Message (no
        Task) — i.e. it finished so fast under return_immediately=True that
        there is nothing to poll; the caller should treat `event` as already
        terminal instead of persisting a cursor.
        """
        config, contract, sdk_client = await self._get_runtime_snapshot(
            server_id, tenant_id, skill_id=skill_id
        )

        request = self._build_send_message_request(
            tenant_id=tenant_id,
            message=message,
            context_id=context_id,
            task_id=None,
            metadata=metadata,
            accepted_output_modes=contract.accepted_output_modes,
            return_immediately=True,
            message_id=message_id,
        )

        result = await self._send_once_and_track(
            request=request,
            config=config,
            contract=contract,
            sdk_client=sdk_client,
            server_id=server_id,
            tenant_id=tenant_id,
            skill_id=skill_id,
            context_id=context_id,
            timeout_seconds=timeout_seconds,
        )
        if result["task_id"]:
            logger.info(
                "[A2A_RECOVER] server_id=%s task_id=%s context_id=%s — submitted, tracking begins",
                server_id,
                result["task_id"],
                result["context_id"],
            )
        return result

    async def continue_task(
            self,
            server_id: str,
            tenant_id: str,
            message,
            task_id: str,
            context_id: str | None = None,
            metadata: Dict[str, Any] | None = None,
            skill_id: str | None = None,
            timeout_seconds: int | None = None,
            message_id: str | None = None,
    ) -> Dict[str, Any]:
        """Continue an EXISTING task (e.g. answering input_required) without
        blocking until it reaches a terminal state (AppFactory-281 P1 review fix).

        Mirrors submit_and_track's return_immediately=True / one-round-trip
        shape, but REQUIRES task_id — where submit_and_track forbids it
        (that one is for starting a NEW task only; this one is for the
        opposite case).

        Before this method existed, answering input_required went through
        send_message(...), which hardcodes return_immediately=False and blocks
        the whole call on request_timeout_seconds (default 60s) — a
        legitimate "still working" reply from the agent after accepting the
        answer would exceed that budget and get misread by the caller as the
        adapter rejecting the answer. This method decouples "did the answer
        get delivered" (this one non-blocking round trip) from "how long did
        the agent take to finish" (the caller's own tasks_get poll loop
        afterward) — the same separation submit_and_track already gives the
        original submission.

        `message_id` lets a caller retrying after a crash mid-dispatch reuse
        the same correlation id (same not-a-protocol-guarantee-but-better-
        than-nothing rationale as submit_and_track's).

        Returns {"task_id": str | None, "context_id": str | None, "event":
        <serialized event>} — task_id is None only for a bare Message reply
        (task already done, nothing left to poll).
        """
        config, contract, sdk_client = await self._get_runtime_snapshot(
            server_id, tenant_id, skill_id=skill_id
        )
        # Same as send_message's continuing-a-task branch: prefer the contract
        # actually bound to this task_id over a fresh card-derived default, so
        # accepted_output_modes stays consistent across the task's lifetime.
        contract = await self._get_task_contract(
            tenant_id=tenant_id,
            server_id=server_id,
            task_id=task_id,
            fallback=contract,
            expected_skill_id=skill_id,
        )

        request = self._build_send_message_request(
            tenant_id=tenant_id,
            message=message,
            context_id=context_id,
            task_id=task_id,
            metadata=metadata,
            accepted_output_modes=contract.accepted_output_modes,
            return_immediately=True,
            message_id=message_id,
        )

        result = await self._send_once_and_track(
            request=request,
            config=config,
            contract=contract,
            sdk_client=sdk_client,
            server_id=server_id,
            tenant_id=tenant_id,
            skill_id=skill_id,
            context_id=context_id,
            timeout_seconds=timeout_seconds,
        )
        logger.info(
            "[A2A_RECOVER] server_id=%s task_id=%s — continuation delivered, tracking resumes",
            server_id,
            task_id,
        )
        return result

    @staticmethod
    def _build_send_message_request(
        *,
        tenant_id: str,
        message,
        context_id: str | None,
        task_id: str | None,
        metadata: Dict[str, Any] | None,
        accepted_output_modes,
        return_immediately: bool,
        message_id: str | None = None,
    ) -> SendMessageRequest:
        """Build a message/send request. Shared by send_message (server holds
        the call open until it decides to respond) and submit_and_track (server
        responds immediately, before the task reaches a terminal state) — the
        only behavioral difference between the two is this one flag.

        message_id priority: an explicit correlation id (submit_and_track's
        idempotent-retry case, AppFactory-280 finding #2) wins over task_id
        (send_message's continuing-an-existing-task case), which wins over a
        freshly generated one — unchanged fallback for every existing caller.
        """
        msg = Message()
        msg.message_id = message_id or task_id or f"msg_{int(datetime.now().timestamp() * 1000)}"
        msg.role = Role.ROLE_USER
        for part in A2AClient._coerce_message_to_parts(message):
            msg.parts.append(part)

        if context_id:
            msg.context_id = context_id
        if task_id:
            msg.task_id = task_id

        request = SendMessageRequest()
        request.tenant = tenant_id
        request.message.CopyFrom(msg)

        if metadata:
            try:
                json_str = json.dumps(metadata)
                StructClass = GetMessageClass(DESCRIPTOR.message_types_by_name['Struct'])
                struct = Parse(json_str, StructClass())
                request.metadata.CopyFrom(struct)
            except Exception as exc:
                logger.warning(f"Failed to set metadata: {exc}")

        request.configuration.return_immediately = return_immediately
        request.configuration.accepted_output_modes.extend(accepted_output_modes)
        return request

    @staticmethod
    def _event_task_id(event: Dict[str, Any]) -> str | None:
        event_type = event.get("type")
        data = event.get("data") or {}
        if event_type == "task":
            task_id = data.get("id")
        elif event_type in ("message", "status_update", "artifact_update"):
            task_id = data.get("task_id")
        else:
            task_id = None
        return str(task_id) if task_id else None

    async def _persist_task_contract(
        self,
        *,
        event: Dict[str, Any],
        tenant_id: str,
        server_id: str,
        contract: A2AContractSnapshot,
        skill_id: str | None,
    ) -> None:
        """Persist a task's first validated contract before exposing its ID."""
        task_id = self._event_task_id(event)
        if not task_id:
            return
        try:
            stored = await self.storage.save_a2a_task_contract(
                tenant_id=tenant_id,
                server_id=server_id,
                task_id=task_id,
                contract=contract.as_dict(),
                skill_id=skill_id,
            )
            stored_contract = A2AContractSnapshot.from_dict(stored.get("contract"))
        except A2AContractError:
            raise
        except Exception as exc:
            raise A2AContractError(
                "a2a_task_contract_persist_failed",
                f"Could not persist A2A contract for task '{task_id}': {exc}",
                location="a2aTaskContract",
                protocol_version=contract.protocol_version,
                declared_modes=contract.declared_output_modes,
            ) from exc
        if stored_contract != contract or (
            skill_id is not None and stored.get("skill_id") != skill_id
        ):
            raise A2AContractError(
                "a2a_task_contract_conflict",
                f"A2A task '{task_id}' is already bound to a different contract",
                location="a2aTaskContract",
                protocol_version=contract.protocol_version,
                declared_modes=contract.declared_output_modes,
            )

    async def _get_task_contract(
        self,
        *,
        tenant_id: str,
        server_id: str,
        task_id: str,
        fallback: A2AContractSnapshot,
        expected_skill_id: str | None = None,
    ) -> A2AContractSnapshot:
        """Load a task-bound contract, with an explicit legacy fallback."""
        stored = await self.storage.get_a2a_task_contract(
            tenant_id, server_id, task_id
        )
        if not stored:
            logger.warning(
                "[A2A_CONTRACT] server_id=%s task_id=%s — no persisted task "
                "contract; using current Agent Card defaults",
                server_id,
                task_id,
            )
            return fallback
        if (
            stored.get("tenant_id") != tenant_id
            or stored.get("server_id") != server_id
            or stored.get("task_id") != task_id
        ):
            raise A2AContractError(
                "a2a_task_contract_invalid",
                f"Persisted A2A task contract identity does not match task '{task_id}'",
                location="a2aTaskContract",
            )
        if (
            expected_skill_id is not None
            and stored.get("skill_id") != expected_skill_id
        ):
            raise A2AContractError(
                "a2a_task_contract_conflict",
                f"A2A task '{task_id}' is bound to a different skill",
                location="a2aTaskContract.skill_id",
            )
        return A2AContractSnapshot.from_dict(stored.get("contract"))

    async def send_batch(
            self,
            server_id: str,
            tenant_id: str,
            requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Send batch JSON-RPC requests."""
        config = await self._get_server_config(server_id, tenant_id)

        batch = []
        for i, req in enumerate(requests):
            batch.append({
                "jsonrpc": "2.0",
                "id": i,
                **req
            })

        async with httpx.AsyncClient(
                base_url=config["url"],
                headers=config["headers"],
                auth=config.get("auth"),
                timeout=httpx.Timeout(config["request_timeout_seconds"]),
        ) as client:
            response = await client.post(config["rpc_endpoint"], json=batch)
            response.raise_for_status()
            results = response.json()

            if not isinstance(results, list):
                raise ValueError("Expected batch response to be an array")

            return sorted(results, key=lambda x: x.get("id", 0))


    def _parse_custom_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse custom JSON-RPC response to standard format."""
        result = response.get("result", {})

        if "task" in result:
            task = result["task"]
            return {
                "type": "task",
                "data": {
                    "id": task.get("id"),
                    "status": task.get("status"),
                    "artifacts": task.get("artifacts", []),
                }
            }

        if "streamingUrl" in result:
            return {
                "type": "streaming_url",
                "data": {"url": result["streamingUrl"]}
            }

        return {
            "type": "response",
            "data": result
        }

    async def tasks_get(
            self,
            server_id: str,
            tenant_id: str,
            task_id: str,
            timeout_seconds: int | None = None,
    ) -> Dict[str, Any]:
        """Get task status. Encapsulates JSON-RPC tasks/get."""
        config, contract, sdk_client = await self._get_runtime_snapshot(
            server_id, tenant_id
        )
        contract = await self._get_task_contract(
            tenant_id=tenant_id,
            server_id=server_id,
            task_id=task_id,
            fallback=contract,
        )

        request = GetTaskRequest()
        request.tenant = tenant_id
        request.id = task_id

        timeout = timeout_seconds or config["request_timeout_seconds"]
        call_context = build_call_context(timeout=float(timeout))

        task = await sdk_client.get_task(request, context=call_context)
        if not task:
            return {}
        serialized = self._serialize_task(task)
        validate_event_output_modes(
            {"type": "task", "data": serialized},
            contract.declared_output_modes,
            protocol_version=contract.protocol_version,
        )
        return serialized

    async def tasks_cancel(
            self,
            server_id: str,
            tenant_id: str,
            task_id: str,
    ) -> Dict[str, Any]:
        """Cancel a running task."""
        _, contract, sdk_client = await self._get_runtime_snapshot(
            server_id, tenant_id
        )
        contract = await self._get_task_contract(
            tenant_id=tenant_id,
            server_id=server_id,
            task_id=task_id,
            fallback=contract,
        )

        cancel_timeout = self._CANCEL_FIXED_TIMEOUT

        logger.debug(
            f"Cancelling task '{task_id}' on server '{server_id}' "
            f"with fixed {cancel_timeout}s timeout (ignoring server config)"
        )

        request = CancelTaskRequest()
        request.tenant = tenant_id
        request.id = task_id

        call_context = build_call_context(timeout=cancel_timeout)

        task = await sdk_client.cancel_task(request, context=call_context)
        if not task:
            return {}
        serialized = self._serialize_task(task)
        validate_event_output_modes(
            {"type": "task", "data": serialized},
            contract.declared_output_modes,
            protocol_version=contract.protocol_version,
        )
        return serialized

    @staticmethod
    def _serialize_parts(parts) -> List[Dict[str, Any]]:
        """Convert protobuf Part oneof values to JSON-serializable dicts.

        `data` is guarded on the real oneof member and run through MessageToDict:
        an unset singular message field is always truthy (so a plain `if part.data`
        fires for text/empty/contentless parts too), and the raw protobuf Value is
        not JSON-serializable — encoding it 500s the tasks endpoints.
        """
        serialized = []
        for part in parts:
            content_kind = part.WhichOneof("content")
            if content_kind == "text":
                item = {"type": "text", "text": part.text}
            elif content_kind == "url":
                item = {"type": "url", "url": part.url}
            elif content_kind == "raw":
                item = {
                    "type": "raw",
                    "raw": base64.b64encode(part.raw).decode("ascii"),
                }
            elif content_kind == "data":
                item = {"type": "data", "data": MessageToDict(part.data)}
            else:
                raise A2AOutputModeError(
                    "a2a_output_part_invalid",
                    "A2A response contains a Part without content",
                    part_type=None,
                )
            if part.media_type:
                item["media_type"] = part.media_type
            if part.filename:
                item["filename"] = part.filename
            if part.metadata.ListFields():
                item["metadata"] = MessageToDict(part.metadata)
            serialized.append(item)
        return serialized

    def _serialize_stream_response(self, stream_response: StreamResponse) -> Dict[str, Any]:
        """Convert protobuf StreamResponse to JSON-serializable dict."""
        result = {"type": None, "data": None}
        if stream_response.HasField("task"):
            result["type"] = "task"
            result["data"] = self._serialize_task(stream_response.task)
        elif stream_response.HasField("message"):
            result["type"] = "message"
            result["data"] = self._serialize_message(stream_response.message)
        elif stream_response.HasField("status_update"):
            result["type"] = "status_update"
            result["data"] = {
                "task_id": stream_response.status_update.task_id,
                "context_id": stream_response.status_update.context_id,
                "state": TaskState.Name(stream_response.status_update.status.state),
                "timestamp": (
                    stream_response.status_update.status.timestamp.ToDatetime().isoformat()
                    if stream_response.status_update.status.timestamp
                    else None
                ),
            }
            if stream_response.status_update.status.HasField("message"):
                result["data"]["message"] = self._serialize_message(
                    stream_response.status_update.status.message
                )
        elif stream_response.HasField("artifact_update"):
            result["type"] = "artifact_update"
            artifact = stream_response.artifact_update.artifact
            artifact_data = {
                "task_id": stream_response.artifact_update.task_id,
                "context_id": stream_response.artifact_update.context_id,
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "append": stream_response.artifact_update.append,
                "last_chunk": stream_response.artifact_update.last_chunk,
            }

            if artifact.parts:
                artifact_data["parts"] = self._serialize_parts(artifact.parts)

            result["data"] = artifact_data

        if result["type"] is None:
            raise A2AOutputModeError(
                "a2a_output_event_invalid",
                "A2A response contains no recognized event payload",
            )
        return result

    def _serialize_task(self, task: Task) -> Dict[str, Any]:
        """Convert protobuf Task to JSON-serializable dict."""
        if not task:
            return {}

        result = {
            "id": task.id,
            "context_id": task.context_id,
        }
        if task.metadata.ListFields():
            result["metadata"] = MessageToDict(task.metadata)

        if task.status:
            result["status"] = {
                "state": TaskState.Name(task.status.state),
                "timestamp": (
                    task.status.timestamp.ToDatetime().isoformat()
                    if task.status.timestamp
                    else None
                ),
            }
            if task.status.HasField("message"):
                result["status"]["message"] = self._serialize_message(task.status.message)

        result["artifacts"] = []
        for artifact in task.artifacts:
            artifact_dict = {
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "description": artifact.description,
            }

            if artifact.parts:
                artifact_dict["parts"] = self._serialize_parts(artifact.parts)

            result["artifacts"].append(artifact_dict)

        result["history"] = [self._serialize_message(m) for m in task.history]

        return result

    def _serialize_message(self, msg: Message) -> Dict[str, Any]:
        """Convert protobuf Message to JSON-serializable dict."""
        if not msg:
            return {}

        parts = self._serialize_parts(msg.parts)

        result = {
            "message_id": msg.message_id,
            "role": Role.Name(msg.role),
            "parts": parts,
            "context_id": msg.context_id,
            "task_id": msg.task_id,
        }
        # Message.metadata is a supported, open A2A object (contract doc line 127) —
        # e.g. an agent's role/step on an input_required question (AppFactory-281). Mirrors
        # _serialize_parts' capture of Part.metadata just above; omitted when unset so
        # callers relying on the key's absence (existing snapshot-style tests) are unaffected.
        if msg.metadata.ListFields():
            result["metadata"] = MessageToDict(msg.metadata)
        return result

    async def get_agent_card(
            self,
            server_id: str,
            tenant_id: str,
            force_refresh: bool = False,
            persist_endpoint_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get agent card, honoring a per-server ``agent_card_url`` override.

        The override may be a full URL (even on a different host) or a path; it is
        passed through as the fetch endpoint so the card is read from where the server
        actually publishes it, not the hardcoded default route. Typed A2A errors
        (not-found / disabled / auth) propagate unwrapped so callers can classify them.
        ``persist_endpoint_cache=False`` is for callers that perform their own guarded
        cache write after the fetch and must keep the current revision unchanged.
        """
        config = await self._get_server_config(server_id, tenant_id)
        endpoint = config.get("agent_card_url") or self._AGENT_CARD_ROUTE
        try:
            card = await self._get_agent_card_from_endpoint(
                server_id=server_id,
                tenant_id=tenant_id,
                endpoint=endpoint,
                force_refresh=force_refresh,
                persist_endpoint_cache=persist_endpoint_cache,
            )
            validate_agent_card(card)
            return card
        except (A2AServerNotFoundError, A2AServerDisabledError, A2AAuthError):
            raise
        except Exception as exc:
            logger.warning(f"Failed to get agent card from {endpoint} for '{server_id}': {exc}")
            raise

    async def _get_agent_card_from_endpoint(
            self,
            server_id: str,
            tenant_id: str,
            endpoint: str,
            force_refresh: bool = False,
            persist_endpoint_cache: bool = True,
    ) -> Dict[str, Any]:
        """Internal method to get agent card from specific endpoint."""
        if not force_refresh:
            server = await self.storage.get_a2a_server(server_id, tenant_id)
            if server:
                def sanitize_key(key: str) -> str:
                    return key.replace('.', '_').replace('$', '_').replace('/', '_')

                endpoint_safe = sanitize_key(endpoint)
                cache_key = f"cached_agent_card_{endpoint_safe}"
                cached_at_key = f"cached_at_{endpoint_safe}"
                cached_card = server.get(cache_key)
                cached_at = server.get(cached_at_key)

                if cached_card and cached_at:
                    if isinstance(cached_at, datetime):
                        cache_age = (datetime.utcnow() - cached_at).total_seconds()
                        if cache_age < 300:  # 5 minutes
                            logger.debug(
                                f"Using cached agent card for '{server_id}' from {endpoint} "
                                f"(age: {cache_age:.0f}s)"
                            )
                            return cached_card

        config = await self._get_server_config(server_id, tenant_id)
        timeout = config["request_timeout_seconds"]

        async with httpx.AsyncClient(
                base_url=config["url"],
                headers=config["headers"],
                auth=config.get("auth"),
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
        ) as http_client:
            try:
                response = await http_client.get(endpoint)
                response.raise_for_status()
                agent_card = response.json()

                if persist_endpoint_cache:
                    await self.storage.update_a2a_server_cache_with_endpoint(
                        server_id=server_id,
                        tenant_id=tenant_id,
                        agent_card=agent_card,
                        validated_at=datetime.utcnow(),
                        endpoint=endpoint,
                    )

                logger.info(
                    f"Fetched agent card from {endpoint} for server '{server_id}': "
                    f"{agent_card.get('name')} v{agent_card.get('version')}"
                )

                return agent_card

            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch agent card from {endpoint} for '{server_id}': {e}")
                raise

    async def invalidate_sdk_client(self, server_id: str, tenant_id: str) -> None:
        """Invalidate cached SDK client for a specific server."""
        cache_prefix = f"{tenant_id}:{server_id}:"

        for cache_key in [
            key for key in self._sdk_clients if key.startswith(cache_prefix)
        ]:
            logger.info(f"Invalidating SDK client for {cache_key}")
            cache_entry = self._sdk_clients[cache_key]
            try:
                await cache_entry["client"].close()
                await cache_entry["http_client"].aclose()
            except Exception as e:
                logger.warning(f"Error closing client for {cache_key}: {e}")

            del self._sdk_clients[cache_key]
            logger.debug(f"SDK client for {cache_key} invalidated")

        # Drop the token provider too (unconditionally — a server used only for a card
        # fetch has a provider but no SDK client), so a config change re-reads credentials.
        for token_key in [
            key for key in self._token_providers if key.startswith(cache_prefix)
        ]:
            self._token_providers.pop(token_key, None)

    async def close(self):
        """Close all SDK clients and HTTP clients."""
        for cache_entry in self._sdk_clients.values():
            await cache_entry["client"].close()
            await cache_entry["http_client"].aclose()

        self._sdk_clients.clear()
        self._token_providers.clear()

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


class A2AClientFactory:
    """Factory for creating and reusing A2A clients."""

    _instances: Dict[int, A2AClient] = {}

    @classmethod
    def get_client(cls, storage: MongoStorageBackend) -> A2AClient:
        """Get or create A2A client for storage."""
        storage_id = id(storage)
        if storage_id not in cls._instances:
            cls._instances[storage_id] = A2AClient(storage)
        return cls._instances[storage_id]
    @classmethod
    async def invalidate_sdk_client(cls, storage: MongoStorageBackend, server_id: str, tenant_id: str) -> None:
        storage_id = id(storage)
        if storage_id in cls._instances:
            await cls._instances[storage_id].invalidate_sdk_client(server_id, tenant_id)

    @classmethod
    async def close_all(cls):
        """Close all A2A client."""
        for client in cls._instances.values():
            await client.close()
        cls._instances.clear()
