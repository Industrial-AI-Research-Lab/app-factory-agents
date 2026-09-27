"""
Shared Context System

Provides project-wide context that all agents can access.
Context is stored in database and includes:
- User prompt
- Requirements (Q&A history)
- Plan/task hierarchy
- Agent decisions
- Files created
- Critical insights
"""

from typing import Any, Callable, Dict, List, Optional
from copy import deepcopy
from datetime import datetime
import json
import logging
from storage.message_store import (
    TOOL_LEDGER_TYPES,
    pair_tool_records,
    tool_payload_preview,
)
from llm.agent_model_params import resolve_effective_agent_model_params
from utils.run_config import normalize_run_config


logger = logging.getLogger(__name__)


def _bare_agent_id(agent_id: str) -> str:
    """Strip the '@<project-suffix>' runtime suffix (mirrors agent_selection._agent_id_prefix)."""
    return str(agent_id).split("@")[0]


# Transport bound on the deliberate journal fetch — NOT a prompt window.
# What enters the prompt is governed by the model-window char valve in
# agents/base.py (interim until the compaction engine, AppFactory-149). This
# bounds the Mongo read per context build because tool_result bodies are
# stored inline up to the 512KB spill threshold; write-time previews would
# let it go away entirely. The journal itself keeps everything.
TOOL_JOURNAL_CONTEXT_RECORDS = 1000

# Page width for the compaction fold-forward read (get_conversation_since). Same
# width as the get_messages default, but the fold pages from the summary frontier
# to the newest turn with NO overall cap — so no message is skipped however large
# the un-summarized tail (ADR-0014 Decision 3).
CONVERSATION_FOLD_PAGE = 1000


def _tool_pair_entry(pair: Dict[str, Any]) -> tuple:
    """One conversation entry per call/result pair, anchored at the call's
    sequence (the result's for orphans) so it interleaves with NL turns in
    stream order. A hanging call renders as "awaiting result" — the visible
    pause point, not a completed action."""
    call, result = pair["call"], pair["result"]
    anchor = call or result
    call_data = (call or {}).get("data") or {}
    result_data = (result or {}).get("data") or {}
    name = call_data.get("name") or result_data.get("name") or "unknown_tool"

    lines = []
    if call and result:
        lines.append(f"[tool] {name} → {result.get('status') or 'ok'}")
    elif call:
        lines.append(f"[tool] {name} — awaiting result")
    else:
        lines.append(
            f"[tool] {name} → {result.get('status') or 'ok'} (call outside context window)"
        )
    if call and call_data.get("arguments"):
        lines.append(f"arguments: {tool_payload_preview(call_data['arguments'])}")
    if result:
        lines.append(f"result: {tool_payload_preview(result_data.get('result'))}")

    entry = {
        "role": "tool",
        "content": "\n".join(lines),
        # The stream sequence, kept on the entry (not just the sort tuple) so the
        # compaction seeder can tell folded turns from fresh ones (AppFactory-149).
        "sequence": anchor.get("sequence", 0),
        # Where the pair's span ENDS — the result's sequence once it has one. The
        # entry sits at the call's sequence, but the fold's frontier must clear
        # the result too, or the next run re-reads it as an orphan of a call that
        # is already in the summary.
        "end_sequence": max(
            int((call or {}).get("sequence", 0) or 0),
            int((result or {}).get("sequence", 0) or 0),
        ),
        "phase": anchor.get("phase", "general"),
        "timestamp": anchor.get("created_at", ""),
        "metadata": {
            "tool_call_id": call_data.get("tool_call_id") or result_data.get("tool_call_id"),
        },
    }
    return anchor.get("sequence", 0), entry


def _initial_context_field_factories(
    project_id: str,
    normalized_run_config: Optional[Any],
) -> Dict[str, Callable[[], Any]]:
    """Describe the persisted SharedContext cache shape in one place.

    Factories ensure each SharedContext instance receives fresh mutable
    containers while constants can still derive the key set without duplicating
    the cache schema.
    """
    return {
        "project_id": lambda: project_id,
        "created_at": lambda: datetime.utcnow().isoformat(),
        "user_prompt": lambda: None,
        "requirements": dict,
        "plan": dict,
        "decisions": list,
        "insights": list,
        "custom_context": dict,
        "workflow_approval": lambda: None,
        "run_config": lambda: normalized_run_config,
        # Deploy dimension (see docs/state_models/deploy_phase.md)
        "deploy_status": lambda: "not_started",
        "deploy_error": lambda: None,
        "deployments": list,
    }


def _build_initial_context_cache(
    project_id: str,
    normalized_run_config: Optional[Any],
) -> Dict[str, Any]:
    """Create a fresh initial cache from the SharedContext cache shape."""
    return {
        key: factory()
        for key, factory in _initial_context_field_factories(
            project_id,
            normalized_run_config,
        ).items()
    }


def _base_logical_context_field_factories(
    conversation_history: List[Dict[str, Any]],
    legacy_artifacts: List[Dict[str, Any]],
) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """Describe the base logical context shape exposed to agents."""
    return {
        "user_prompt": lambda cache: cache.get("user_prompt", ""),
        "conversation_history": lambda cache: conversation_history,
        "requirements": lambda cache: cache.get("requirements", {}),
        "plan": lambda cache: cache.get("plan", {}),
        "decisions": lambda cache: cache.get("decisions", []),
        "insights": lambda cache: cache.get("insights", []),
        # DEPRECATED compatibility only. Real file artifacts live in ArtifactStore.
        "artifacts": lambda cache: legacy_artifacts,
        # Deploy-related fields
        "deploy_status": lambda cache: cache.get("deploy_status", "not_started"),
        "deploy_error": lambda cache: cache.get("deploy_error"),
        "deployments": lambda cache: cache.get("deployments", []),
        "project_attachments": lambda cache: cache.get("project_attachments") or [],
        "tenant_artifacts": lambda cache: cache.get("tenant_artifacts") or [],
        "workflow_approval": lambda cache: cache.get("workflow_approval"),
    }


def _build_base_logical_context(
    cache: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
    legacy_artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create the base logical context from the readable context shape."""
    return {
        key: factory(cache)
        for key, factory in _base_logical_context_field_factories(
            conversation_history,
            legacy_artifacts,
        ).items()
    }


# Contract key taxonomy:
# - READABLE_CONTEXT_KEYS are derived from the base logical context shape.
# - WRITABLE_CONTEXT_KEYS have specialized write paths and may be declared in writes.
# - INTERNAL_CONTEXT_KEYS are derived from SharedContext cache/storage shape and
#   must not be shadowed by custom_context outputs. API validation imports this
#   taxonomy instead of re-declaring its own reserved-key list.
READABLE_CONTEXT_KEYS = frozenset(
    _base_logical_context_field_factories([], []).keys()
)
WRITABLE_CONTEXT_KEYS = frozenset({"requirements", "plan"})
INITIAL_CONTEXT_KEYS = frozenset(_initial_context_field_factories("", None).keys())
LIFECYCLE_CONTEXT_KEYS = frozenset({
    "status",
})
AUDIT_CONTEXT_KEYS = frozenset({
    "updated_at",
})
INTERNAL_CONTEXT_KEYS = INITIAL_CONTEXT_KEYS | LIFECYCLE_CONTEXT_KEYS | AUDIT_CONTEXT_KEYS
READ_ONLY_CONTEXT_KEYS = (READABLE_CONTEXT_KEYS | INTERNAL_CONTEXT_KEYS) - WRITABLE_CONTEXT_KEYS
FULL_CONTEXT_READS_TOKEN = "*"
WORKFLOW_DEFAULT_READS_TOKEN = "$workflow_defaults"
SYSTEM_BASE_READ_KEYS = ("user_prompt", "conversation_history")

# Internal sentinel routed only by SharedContext.runless(); __init__ rejects
# every other falsy run_id so a None can never slip in through a fallback.
_RUNLESS: Any = object()


class SharedContext:
    """
    Project-wide shared context accessible by all agents.
    
    Agents can:
    - Add critical information
    - Search semantically
    - Query by phase/category
    - Update plan/requirements
    """

    @classmethod
    def runless(
            cls,
            project_id: str,
            storage_backend,
            *,
            message_store=None,
            run_config: Optional[Any] = None
    ) -> "SharedContext":
        """The ONLY way to build a context without a run (run_id=None to
        __init__ raises). For throwaway tooling that loads/imports/syncs
        state and is discarded without ever reaching an agent runner —
        today that is snapshot restore alone. Named constructor on purpose:
        a run-less context must be a visible decision at the call site,
        never the result of a None fallback."""
        return cls(
            project_id,
            storage_backend,
            run_id=_RUNLESS,
            message_store=message_store,
            run_config=run_config,
        )

    def __init__(
            self,
            project_id: str,
            storage_backend,
            *,
            run_id: str,
            message_store=None,
            run_config: Optional[Any] = None
    ):
        """
        Args:
            project_id: Unique project identifier
            storage_backend: Database storage implementation
            run_id: Run identifier — REQUIRED and non-empty. A missing
                run_id silently disables the tool ledger and ask_human
                parking for every consumer of this context, so None/empty
                raises here instead of degrading later. The lazy fallback
                `active_run.get("run_id") if active_run else None` is
                exactly the bug this guards against — resolve a real run
                id first, or fail the operation. The single run-less
                consumer (snapshot restore) must use `runless()`.
            message_store: Optional MessageStore for dual-write (unified message system)
        """
        if run_id is not _RUNLESS and not run_id:
            raise ValueError(
                f"SharedContext({project_id}) requires a non-empty run_id, got {run_id!r}: "
                "a run-less context silently disables the tool ledger and ask_human "
                "parking. Resolve the active run first, or use SharedContext.runless() "
                "for tooling that never reaches an agent runner."
            )
        self.project_id = project_id
        self.run_id = None if run_id is _RUNLESS else run_id
        self.storage = storage_backend
        self.message_store = message_store
        normalized_run_config = normalize_run_config(run_config)

        # Read-through cache state (Bulletproof Persistence 2.1.4)
        self._loaded: bool = False  # True after initialize() or load_from_db()

        # In-memory cache (synced with DB)
        # NOTE: conversation_history removed - messages collection is single source of truth
        # Use get_full_context_async() to read conversation from message_store
        self._cache: Dict[str, Any] = _build_initial_context_cache(
            project_id,
            normalized_run_config,
        )
    
    async def initialize(self, user_prompt: str):
        """Initialize context with user's prompt"""
        self._cache["user_prompt"] = user_prompt
        self._cache["status"] = "initialized"
        
        await self._sync_to_db()
        self._loaded = True  # Mark as loaded (Bulletproof Persistence)
        
        # SINGLE WRITE PATH: message_store is the only writer for messages
        if self.message_store:
            await self.message_store.append_user_message(
                self.project_id, user_prompt, run_id=self.run_id,
                metadata={"phase": "initial"}
            )
    
    async def add_conversation_message(
            self,
            role: str,
            content: str,
            phase: str = "general",
            metadata: Optional[Dict] = None
    ):
        """
        Add a message to conversation history.
        
        SINGLE WRITE PATH: message_store is the only writer for messages.
        conversation_history in cache is deprecated - use get_full_context_async() to read from messages.
        
        Args:
            role: "user" or "assistant" or "system"
            content: Message content
            phase: Current phase (requirements, planning, execution, etc.)
            metadata: Optional metadata (agent_id, etc.)
        """
        # SINGLE WRITE PATH: message_store is the only writer for messages
        if self.message_store:
            msg_metadata = {**(metadata or {}), "phase": phase}
            
            if role == "user":
                await self.message_store.append_user_message(
                    self.project_id, content, run_id=self.run_id, metadata=msg_metadata
                )
            elif role == "assistant":
                await self.message_store.append_assistant_message(
                    self.project_id, content, run_id=self.run_id,
                    phase=phase, metadata=metadata
                )
            else:
                await self.message_store.append_system_message(
                    self.project_id, subtype="conversation", content=content,
                    run_id=self.run_id, metadata=msg_metadata
                )
    
    async def add(self, key: str, value: Any, category: str = "general"):
        """
        Add information to shared context.
        
        Args:
            key: Identifier for this information
            value: Data to store
            category: Category for organization (requirements, planning, coding, etc.)
        
        Example:
            await context.add(
                "caching_requirement", 
                "User wants Redis for caching",
                category="requirements"
            )
        """
        entry = {
            "key": key,
            "value": value,
            "category": category,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._cache["insights"].append(entry)
        
        # Incremental update to MongoDB (if available)
        if hasattr(self.storage, 'append_to_context_array'):
            await self.storage.append_to_context_array(
                self.project_id,
                "insights",
                entry
            )
        else:
            await self._sync_to_db()

    def _custom_context(self) -> Dict[str, Any]:
        custom = self._cache.get("custom_context")
        if not isinstance(custom, dict):
            custom = {}
            self._cache["custom_context"] = custom
        return custom

    def _logical_context_view(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Return context with custom keys addressable as top-level logical keys."""
        custom_context = self._custom_context()
        view = {**custom_context, **base_context}
        view["custom_context"] = dict(custom_context)
        return view

    def read_context_key(self, key: str, default: Any = None) -> Any:
        """Read a logical context key from well-known fields or custom_context."""
        key = str(key or "").strip()
        if not key:
            return default
        if key in READABLE_CONTEXT_KEYS:
            return self.get_full_context().get(key, default)
        return self._custom_context().get(key, default)

    async def read_context_key_async(self, key: str, default: Any = None) -> Any:
        """Async logical context read, including conversation history when available."""
        key = str(key or "").strip()
        if not key:
            return default
        if key in READABLE_CONTEXT_KEYS:
            return (await self.get_full_context_async()).get(key, default)
        await self._ensure_loaded()
        return self._custom_context().get(key, default)

    async def write_context_key(self, key: str, value: Any) -> None:
        """Write a logical contract key to the correct SharedContext location."""
        key = str(key or "").strip()
        if not key:
            raise ValueError("Context key is required")
        if key in READ_ONLY_CONTEXT_KEYS:
            raise ValueError(f"Context key '{key}' is read-only")
        if key == "requirements":
            await self.update_requirements(value if isinstance(value, dict) else {"raw": value})
            return
        if key == "plan":
            await self.update_plan(value if isinstance(value, dict) else {"raw": value})
            return

        # context_write's `value` arg is untyped, so a model may return a structured
        # key's value as a JSON string; recover a list/dict before a json_schema
        # validator rejects it. Scalars and plain text stay the strings they were.
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, (list, dict)):
                value = parsed

        custom_context = self._custom_context()
        custom_context[key] = value
        await self._sync_to_db()

    async def restore_context_keys_exact(self, values: Dict[str, Any]) -> None:
        """Restore durable phase outputs without applying normal write transforms."""
        if not isinstance(values, dict) or not values:
            raise ValueError("Context restore values must be a non-empty object")

        normalized: Dict[str, Any] = {}
        for raw_key, value in values.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ValueError("Context restore key is invalid")
            key = raw_key.strip()
            if key in READ_ONLY_CONTEXT_KEYS:
                raise ValueError(f"Context key '{key}' is read-only")
            normalized[key] = deepcopy(value)

        await self._ensure_loaded()
        previous_requirements = deepcopy(self._cache.get("requirements", {}))
        previous_plan = deepcopy(self._cache.get("plan", {}))
        previous_custom = deepcopy(self._custom_context())
        try:
            for key, value in normalized.items():
                if key in {"requirements", "plan"}:
                    self._cache[key] = value
                else:
                    self._custom_context()[key] = value
            await self._sync_to_db()
        except BaseException:
            self._cache["requirements"] = previous_requirements
            self._cache["plan"] = previous_plan
            self._cache["custom_context"] = previous_custom
            raise

    async def record_workflow_approval(self, approval: Optional[Dict]) -> None:
        self._cache["workflow_approval"] = deepcopy(approval)
        await self._sync_to_db()

    async def update_requirements(self, requirements: Dict):
        """Update requirements after requirements phase"""
        # Store FULL requirements context, including original prompt
        self._cache["requirements"] = {
            "user_prompt": self._cache.get("user_prompt", ""),  # Include original prompt
            **requirements,
            "updated_at": datetime.utcnow().isoformat()
        }
        await self._sync_to_db()

    async def record_deployment_result(self, deployment: Dict, deploy_error: Optional[Any] = None):
        """Append a DeploymentSummary result and update deploy_status/deploy_error."""
        deployments = list(self._cache.get("deployments", []))
        dep_id = deployment.get("deployment_id")
        if dep_id:
            updated = False
            for idx, existing in enumerate(deployments):
                if isinstance(existing, dict) and existing.get("deployment_id") == dep_id:
                    deployments[idx] = deployment
                    updated = True
                    break
            if not updated:
                deployments.append(deployment)
        else:
            deployments.append(deployment)
        self._cache["deployments"] = deployments
        status = deployment.get("status") or "failed"
        self._cache["deploy_status"] = status
        self._cache["deploy_error"] = deploy_error
        await self._sync_to_db()
    
    def get_full_context(self) -> Dict:
        """
        Get context for agents (sync version - NO conversation history).
        
        WARNING: Use get_full_context_async() instead - it reads conversation from message_store.
        This sync version returns empty conversation_history.
        """
        return self._logical_context_view(
            _build_base_logical_context(
                self._cache,
                [],  # Empty - use get_full_context_async() for messages
                self._get_legacy_artifacts(),
            )
        )

    async def get_full_context_async(
        self, exclude_journal_task_id: Optional[str] = None
    ) -> Dict:
        """
        Get COMPLETE context for agents with conversation from message store (DB).

        This is the proper method - message store is the single source of truth
        for conversation history. Other fields come from in-memory cache.

        ``exclude_journal_task_id`` skips that attempt's tool pairs in the
        rendered history: a restart-resumed attempt already re-enters the
        prompt as structured transcript items (ADR-0009), so rendering the
        same pairs as text turns would double-feed the model.
        """
        # Fetch conversation from message store (single source of truth)
        conversation = []
        if self.message_store:
            try:
                # Excluded server-side so the read window (limit=1000,
                # oldest-first) is spent on conversation, not journal records
                # two-per-tool-call; the in-loop skip below stays as the belt
                # for records arriving via any other fetch path.
                messages = await self.message_store.get_messages(
                    self.project_id,
                    after_sequence=0,
                    exclude_types=list(TOOL_LEDGER_TYPES),
                )
                # (sequence, entry) so journal entries can interleave with NL
                # in stream order before the sequence is dropped.
                sequenced_entries = []
                # Convert DB messages to conversation format for agents
                for msg in messages:
                    entry = self._nl_entry(msg)
                    # Journal records render as pairs below, not as turns;
                    # _nl_entry returns None for them — the belt for records
                    # arriving via any other fetch path (they'd fall through to
                    # role:"system" turns with None content).
                    if entry is None:
                        continue
                    sequenced_entries.append((entry["sequence"], entry))

                # The journal, fetched deliberately (ADR-0008): the newest
                # pairs render as role:"tool" turns so the next phase sees
                # what was already done and where the agent stopped. Its own
                # try/except — losing the action log must not cost the NL
                # conversation.
                try:
                    journal = await self.message_store.get_messages(
                        self.project_id,
                        only_types=list(TOOL_LEDGER_TYPES),
                        tail=True,
                        limit=TOOL_JOURNAL_CONTEXT_RECORDS,
                    )
                    for pair in pair_tool_records(journal):
                        if exclude_journal_task_id and (
                            ((pair.get("call") or {}).get("data") or {})
                            .get("task_id") == exclude_journal_task_id
                        ):
                            continue
                        sequenced_entries.append(_tool_pair_entry(pair))
                except Exception:
                    logger.warning(
                        "Tool journal fetch failed for %s; context is NL-only",
                        self.project_id,
                    )

                # Sequences are unique per project, so this is a total order.
                sequenced_entries.sort(key=lambda pair: pair[0])
                conversation = [entry for _, entry in sequenced_entries]
            except Exception:
                # No fallback - messages collection is single source of truth
                conversation = []
        else:
            # No message_store - return empty (messages is single source of truth)
            conversation = []
        project_attachments = await self._project_attachments_manifest()
        tenant_artifacts = await self._tenant_artifacts_manifest()
        return self._logical_context_view(
            {
                **_build_base_logical_context(
                    self._cache,
                    conversation,
                    self._get_legacy_artifacts(),
                ),
                "project_attachments": project_attachments,
                "tenant_artifacts": tenant_artifacts,
            }
        )
    
    def _nl_entry(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map one stored message to a natural-language conversation entry, or
        None for tool-ledger records (they render as paired journal turns, not NL
        turns). The single mapping shared by the full-context read and the
        fold-forward read, so a folded turn reads to the compactor exactly as it
        would on screen."""
        msg_type = msg.get("type", "")
        if msg_type in TOOL_LEDGER_TYPES:
            return None
        content = msg.get("content", "")
        if msg_type == "user":
            role = "user"
        elif msg_type == "assistant":
            role = "assistant"
        elif msg_type == "approval":
            role = "assistant"
            # Include approval data in content for context
            data = msg.get("data", {})
            if data:
                content = f"{content}\n[Approval data available]"
        else:
            role = "system"
        return {
            "role": role,
            "content": content,
            # Kept on the entry (not just the sort tuple) so the compaction seeder
            # can distinguish folded from fresh turns by sequence (AppFactory-149).
            "sequence": msg.get("sequence", 0),
            "phase": msg.get("phase", msg.get("metadata", {}).get("phase", "general")),
            "timestamp": msg.get("created_at", ""),
            "metadata": msg.get("metadata", {}),
        }

    async def get_conversation_since(
        self, after_sequence: int
    ) -> List[Dict[str, Any]]:
        """Every NL conversation entry with sequence > ``after_sequence``, ascending,
        read forward to completion — the compaction fold-forward read (ADR-0014
        Decision 3).

        Unlike get_full_context_async's single oldest-first page (capped at 1000),
        this pages forward with no overall cap, so the fold always sees the whole
        un-summarized tail — including the newest turns past the 1000th. That is
        what lets the fold advance the summary frontier only over turns it actually
        read: the oldest-first window silently drops the newest turns once a project
        passes 1000 messages, and if the seeder is toggled off then on (or messages
        are bulk-imported) the gap is arbitrarily large. Tool records are excluded
        here; the journal is a separate bounded recent read.
        """
        if not self.message_store:
            return []
        entries: List[Dict[str, Any]] = []
        cursor = int(after_sequence or 0)
        while True:
            page = await self.message_store.get_messages(
                self.project_id,
                after_sequence=cursor,
                exclude_types=list(TOOL_LEDGER_TYPES),
                limit=CONVERSATION_FOLD_PAGE,
            )
            if not page:
                break
            for msg in page:
                entry = self._nl_entry(msg)
                if entry is not None:
                    entries.append(entry)
            last_seq = int(page[-1].get("sequence", 0))
            # Last partial page ends the walk; the sequence guard prevents a stall
            # if a duck-typed store ever returns a non-advancing page.
            if len(page) < CONVERSATION_FOLD_PAGE or last_seq <= cursor:
                break
            cursor = last_seq
        return entries

    async def get_journal_since(
        self, after_sequence: int, exclude_journal_task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Every tool-ledger pair anchored past ``after_sequence``, ascending, read
        forward to completion — the fold's tool-side counterpart to
        get_conversation_since (ADR-0014 Decision 1: one unified stream).

        get_full_context_async loads only the newest TOOL_JOURNAL_CONTEXT_RECORDS
        raw records for the verbatim journal render — a deliberately bounded recent
        view (ADR-0008). The fold needs the WHOLE tool tail past the frontier
        instead: it advances conversation_covers_to_sequence to the folded cut, and
        that frontier gates tool records too, so folding off the bounded window would
        mark old pairs (below the cut but dropped from the newest-1000 tail) as
        covered without ever summarising them — silent loss on the next run.

        ``exclude_journal_task_id`` mirrors get_full_context_async: a restart-resumed
        attempt re-enters the prompt as structured items, so its pairs are skipped
        here too rather than double-fed.
        """
        if not self.message_store:
            return []
        raw: List[Dict[str, Any]] = []
        cursor = int(after_sequence or 0)
        while True:
            page = await self.message_store.get_messages(
                self.project_id,
                after_sequence=cursor,
                only_types=list(TOOL_LEDGER_TYPES),
                limit=CONVERSATION_FOLD_PAGE,
            )
            if not page:
                break
            raw.extend(page)
            last_seq = int(page[-1].get("sequence", 0))
            if len(page) < CONVERSATION_FOLD_PAGE or last_seq <= cursor:
                break
            cursor = last_seq
        # Pair over the whole accumulated list so a call/result straddling a page
        # boundary still pairs; each entry's anchor sequence is > after_sequence
        # because every raw record read is.
        entries: List[Dict[str, Any]] = []
        for pair in pair_tool_records(raw):
            if exclude_journal_task_id and (
                ((pair.get("call") or {}).get("data") or {})
                .get("task_id") == exclude_journal_task_id
            ):
                continue
            entries.append(_tool_pair_entry(pair)[1])
        return entries

    def export_state(self) -> Dict:
        """Export snapshot-ready state."""
        return self.get_full_context()
    
    async def import_state(self, state: Dict):
        """Restore state from snapshot and persist to DB."""
        if not isinstance(state, dict):
            return
        self._cache["workflow_approval"] = None
        for k in (
            "user_prompt",
            "conversation_history",
            "requirements",
            "plan",
            "decisions",
            "insights",
            "custom_context",
            # DEPRECATED compatibility for legacy snapshots only.
            "artifacts",
            "deploy_status",
            "deploy_error",
            "deployments",
        ):
            if k in state:
                self._cache[k] = state[k]
        await self._sync_to_db()
        self._loaded = True  # Mark as loaded (Bulletproof Persistence)
    
    async def update_plan(self, plan: Dict):
        """Update plan after planning phase"""
        self._cache["plan"] = {
            **plan,
            "updated_at": datetime.utcnow().isoformat()
        }
        await self._sync_to_db()
    
    async def add_decision(
            self,
            agent_id: str,
            decision: str,
            reasoning: str,
            metadata: Optional[Dict] = None
    ):
        """
        Record an agent's decision.
        
        Useful for tracking why certain paths were taken.
        """
        entry = {
            "agent_id": agent_id,
            "decision": decision,
            "reasoning": reasoning,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._cache["decisions"].append(entry)
        
        # Incremental update to MongoDB (if available)
        if hasattr(self.storage, 'append_to_context_array'):
            await self.storage.append_to_context_array(
                self.project_id,
                "decisions",
                entry
            )
        else:
            await self._sync_to_db()
    
    async def add_artifact(
            self,
            artifact_type: str,
            path: str,
            content: Optional[str] = None,
            metadata: Optional[Dict] = None
    ):
        """
        DEPRECATED: persist file artifacts via ArtifactStore, not SharedContext.

        Args:
            artifact_type: Type (file, api_endpoint, config, etc.)
            path: Location or identifier
            content: Actual content of the artifact (code, config data, etc.)
            metadata: Additional info (size, language, etc.)
        """
        logger.warning(
            "[SHARED_CONTEXT] project_id=%s path=%s artifact_type=%s - add_artifact is deprecated; use ArtifactStore.save_file()",
            self.project_id,
            path,
            artifact_type,
        )

        if not path or content is None:
            logger.warning(
                "[SHARED_CONTEXT] project_id=%s path=%s - deprecated add_artifact skipped because content/path is missing",
                self.project_id,
                path,
            )
            return

        try:
            from storage.artifact_store import ArtifactStore

            artifact_store = ArtifactStore(self.storage, message_store=self.message_store)
            await artifact_store.initialize()
            if artifact_store.collection is None:
                logger.warning(
                    "[SHARED_CONTEXT] project_id=%s path=%s - ArtifactStore unavailable during deprecated add_artifact",
                    self.project_id,
                    path,
                )
                return
            await artifact_store.save_file(self.project_id, path, content, run_id=self.run_id)
        except Exception as exc:
            logger.warning(
                "[SHARED_CONTEXT] project_id=%s path=%s error=%s - deprecated add_artifact failed",
                self.project_id,
                path,
                exc,
            )
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete context"""
        return self._cache.copy()

    def _get_legacy_artifacts(self) -> List[Dict[str, Any]]:
        """Return legacy context artifacts for backward compatibility only."""
        return self._cache.get("artifacts", [])

    def get(self, key: str, default: Any = None) -> Any:
        """Get specific key from context (sync - uses cached state)"""
        return self._cache.get(key, default)
    
    async def get_async(self, key: str, default: Any = None) -> Any:
        """
        Get specific key from context with read-through behavior.
        
        If context hasn't been loaded yet, loads from DB first.
        Use this in async contexts where you need guaranteed fresh data.
        """
        await self._ensure_loaded()
        return self._cache.get(key, default)
    
    async def _ensure_loaded(self) -> None:
        """
        Ensure context is loaded from DB (read-through cache).
        
        Called automatically by get_async(). If already loaded, this is a no-op.
        """
        if not self._loaded:
            await self.load_from_db()
    
    def invalidate(self) -> None:
        """
        Invalidate the cache, forcing next get_async() to reload from DB.
        
        Use this when you know the DB state may have changed externally.
        """
        self._loaded = False
    
    def get_by_phase(self, phase: str) -> Dict:
        """
        Get context for a specific phase.
        
        Args:
            phase: Phase name (requirements, planning, coding, testing, etc.)
        
        Returns:
            Filtered context relevant to that phase
        """
        insights = [
            item for item in self._cache.get("insights", [])
            if item.get("category") == phase
        ]
        
        decisions = [
            item for item in self._cache.get("decisions", [])
            if phase in item.get("decision", "").lower()
        ]
        
        return {
            "phase": phase,
            "insights": insights,
            "decisions": decisions,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search across context.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of relevant context entries
        
        Note: This would use embeddings in production.
        For POC, we do simple keyword matching.
        """
        query_lower = query.lower()
        results = []
        
        # Search insights
        for insight in self._cache.get("insights", []):
            value_str = str(insight.get("value", "")).lower()
            if query_lower in value_str:
                results.append({
                    "type": "insight",
                    "content": insight,
                    "relevance": 0.9  # Simplified scoring
                })
        
        # Search decisions
        for decision in self._cache.get("decisions", []):
            decision_str = str(decision.get("decision", "")).lower()
            reasoning_str = str(decision.get("reasoning", "")).lower()
            
            if query_lower in decision_str or query_lower in reasoning_str:
                results.append({
                    "type": "decision",
                    "content": decision,
                    "relevance": 0.8
                })
        
        # Search requirements
        req_str = json.dumps(self._cache.get("requirements", {})).lower()
        if query_lower in req_str:
            results.append({
                "type": "requirements",
                "content": self._cache.get("requirements"),
                "relevance": 1.0
            })
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    async def _sync_to_db(self):
        """Persist context to database (per-run if run_id is set)"""
        await self.storage.save_context(self.project_id, self._cache, run_id=self.run_id)
    
    async def load_from_db(self):
        """Load context from database (per-run if run_id is set)"""
        data = await self.storage.load_context(self.project_id, run_id=self.run_id)
        if data:
            self._cache = {**self._cache, **data}
            self._custom_context()
            approval = self._cache.get("workflow_approval")
            if approval is not None and (
                not isinstance(approval, dict)
                or approval.get("project_id") != self.project_id
                or approval.get("run_id") != self.run_id
            ):
                logger.warning(
                    "[APPROVAL] project_id=%s run_id=%s — discarded approval from another context",
                    self.project_id, self.run_id,
                )
                self._cache["workflow_approval"] = None
        self._loaded = True  # Mark as loaded (Bulletproof Persistence)
    
    async def truncate_conversation(self, before_index: int) -> None:
        """Truncate conversation history - delegates to message_store.

        NOTE: before_index maps to sequence number in message_store.
        This is used on revert-to-user-action so that the reverted-to message is not in history yet.
        """
        if self.message_store and isinstance(before_index, int) and before_index >= 0:
            # Delete messages after the target sequence
            await self.message_store.delete_messages_after_sequence(
                self.project_id, before_index, run_id=self.run_id
            )

    def get_model(self, subsystem: str, fallback_model: Optional[str] = None, agent_id: Optional[str] = None) -> str:
        """Get model for a subsystem from run configuration.

        Fallback chain:
        force_model_override -> agent_configs[agent_id].model -> models[subsystem]
        -> models["default"] -> project base model -> fallback_model -> "gpt-5-mini"
        """
        rc = normalize_run_config(self._cache.get("run_config"))
        return resolve_effective_agent_model_params(
            agent_id=agent_id or "",
            subsystem=subsystem,
            agent_model=fallback_model,
            agent_temperature=None,
            agent_reasoning_effort=None,
            run_config=rc,
            project_model=getattr(self, "_model_override", None),
            force_project_model=bool(
                getattr(self, "_force_model_override", False)
            ),
            project_reasoning_effort=getattr(
                self,
                "_reasoning_effort_override",
                None,
            ),
        ).model

    def get_reasoning_effort(
        self,
        default: Optional[str] = "medium",
        agent_id: Optional[str] = None,
    ) -> Optional[str]:
        """Return the project-level reasoning effort, or `default` if none was set.

        The override is populated from the create-project payload
        (`reasoning.effort`) and lives only on the in-memory SharedContext;
        agents read it via `BaseAgent._resolve_reasoning_effort` so a per-project
        choice beats their static config without mutating the agent record.
        """
        validated = getattr(self, "_effective_agent_model_params", None)
        if isinstance(validated, dict) and agent_id:
            agent_params = validated.get(_bare_agent_id(agent_id))
            if agent_params is not None:
                return agent_params.reasoning_effort

        return resolve_effective_agent_model_params(
            agent_id=agent_id or "",
            subsystem="agent_default",
            agent_model=None,
            agent_temperature=None,
            agent_reasoning_effort=default,
            run_config=normalize_run_config(self._cache.get("run_config")),
            project_model=getattr(self, "_model_override", None),
            force_project_model=bool(
                getattr(self, "_force_model_override", False)
            ),
            project_reasoning_effort=getattr(
                self,
                "_reasoning_effort_override",
                None,
            ),
        ).reasoning_effort
    
    def get_temperature(
        self,
        default: Optional[float] = 1.0,
        agent_id: Optional[str] = None,
    ) -> Optional[float]:
        return resolve_effective_agent_model_params(
            agent_id=agent_id or "",
            subsystem="agent_default",
            agent_model=None,
            agent_temperature=default,
            agent_reasoning_effort=None,
            run_config=normalize_run_config(self._cache.get("run_config")),
            project_model=getattr(self, "_model_override", None),
            force_project_model=bool(
                getattr(self, "_force_model_override", False)
            ),
            project_reasoning_effort=getattr(
                self,
                "_reasoning_effort_override",
                None,
            ),
            project_temperature=getattr(
                self,
                "_temperature_override",
                None,
            ),
        ).temperature

    def get_step_limit(self, default: int = 30, agent_id: Optional[str] = None) -> int:
        rc = normalize_run_config(self._cache.get("run_config"))
        if agent_id:
            bare_id = _bare_agent_id(agent_id)
            agent_cfg = (rc.get("agent_configs") or {}).get(bare_id) or {}
            if isinstance(agent_cfg, dict) and isinstance(agent_cfg.get("step_limit"), (int, float)):
                return int(agent_cfg["step_limit"])
        return default

    def to_dict(self) -> Dict:
        """Export context as dictionary"""
        return self._cache.copy()
    
    def __repr__(self) -> str:
        return f"<SharedContext(project={self.project_id}, insights={len(self._cache.get('insights', []))})>"

    @property
    def cache(self):
        return self._cache

    @property
    def tenant_id(self) -> Optional[str]:
        """Tenant for tool config scoping; set by ProjectManager / approvals via ``_tenant_id``."""
        return getattr(self, "_tenant_id", None)

    async def _project_attachments_manifest(self) -> list:
        """id/filename/size/mime (+ text_content for small text files). Empty on failure."""
        tenant_id = str(self.tenant_id or "")
        if not tenant_id or self.storage is None:
            if (
                self.storage is not None
                and not tenant_id
                and getattr(self.storage, "user_attachments", None) is not None
            ):
                logger.warning("[ATTACH] manifest skipped project=%s — no tenant_id", self.project_id)
            return []
        try:
            from storage.file_attachment_store import FileAttachmentStore
            from storage.file_blob_store import FileBlobStore
            from tools.attachment_tools import (
                inline_attachment_text,
                inline_text_total_max_bytes,
                public_attachment,
            )

            blob = FileBlobStore.from_env()
            rows, truncated = await FileAttachmentStore(self.storage, blob).list_user_attachments(
                tenant_id=tenant_id, project_id=self.project_id
            )
        except Exception as exc:
            logger.warning("[ATTACH] manifest failed project=%s: %s", self.project_id, exc)
            return []
        if truncated:
            logger.warning(
                "[ATTACH] manifest truncated project=%s tenant=%s",
                self.project_id,
                tenant_id,
            )
        manifest = []
        inline_budget = inline_text_total_max_bytes()
        inline_used = 0
        for row in rows:
            if not row.get("_id"):
                continue
            entry = public_attachment(row)
            try:
                text = await inline_attachment_text(blob, row)
            except Exception as exc:
                logger.warning(
                    "[ATTACH] inline text skipped attachment=%s project=%s: %s",
                    row.get("_id"),
                    self.project_id,
                    exc,
                )
                text = None
            if text:
                nbytes = len(text.encode("utf-8"))
                if inline_used + nbytes <= inline_budget:
                    entry["text_content"] = text
                    inline_used += nbytes
                else:
                    entry["inline_skipped"] = "aggregate_budget; use attachment_view"
            manifest.append(entry)
        return manifest

    async def _tenant_artifacts_manifest(self) -> list:
        """id/filename/size/mime/title for this tenant. Empty on any failure."""
        tenant_id = str(self.tenant_id or "")
        if not tenant_id or self.storage is None:
            if (
                self.storage is not None
                and not tenant_id
                and getattr(self.storage, "tenant_artifacts", None) is not None
            ):
                logger.warning(
                    "[TENANT_ARTIFACT] manifest skipped project=%s — no tenant_id",
                    self.project_id,
                )
            return []
        try:
            from storage.file_attachment_store import FileAttachmentStore
            from storage.file_blob_store import FileBlobStore
            from tools.tenant_artifact_tools import public_tenant_artifact

            rows, truncated = await FileAttachmentStore(self.storage, FileBlobStore()).list_tenant_artifacts(
                tenant_id=tenant_id
            )
        except Exception as exc:
            logger.warning(
                "[TENANT_ARTIFACT] manifest failed project=%s tenant=%s: %s",
                self.project_id,
                tenant_id,
                exc,
            )
            return []
        if truncated:
            logger.warning(
                "[TENANT_ARTIFACT] manifest truncated project=%s tenant=%s",
                self.project_id,
                tenant_id,
            )
        return [public_tenant_artifact(row) for row in rows if row.get("_id")]
