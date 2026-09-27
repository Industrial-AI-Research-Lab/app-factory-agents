"""Pydantic models for Configuration API (F4).

Provides Create / Update / Response schemas for:
- agent_configurations
- workflow_definitions
- tool_configurations

Used by FastAPI routes for automatic request validation and
OpenAPI documentation generation.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, HttpUrl
from pydantic_core.core_schema import ValidationInfo

from schemas.checkpoints import A2ACheckpointConfig
from llm.agent_model_params import DEFAULT_AGENT_MODEL

AGENT_ID_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"
AGENT_ID_DESCRIPTION = "Lowercase letters, digits, underscores. Must start with a letter. Length 2-64. Example: coding_agent"
SHORT_DESCRIPTION_MAX_LEN = 256
WORKFLOW_PROMPT_NAME_MAX_LEN = 200
WORKFLOW_PROMPT_TEXT_MAX_LEN = 20000


def _clamp_short_description(text: Any) -> str:
    return str(text or "")[:SHORT_DESCRIPTION_MAX_LEN]


def entity_short_description_from_doc(data: Any) -> str:
    """Short card/catalog text with legacy fallback (matches UI entityShortDescription)."""
    if not isinstance(data, dict):
        return ""
    short = str(data.get("short_description") or "").strip()
    if short:
        return _clamp_short_description(short)
    return _clamp_short_description(str(data.get("description") or "").strip())


def _coerce_entity_descriptions(data: Any) -> Any:
    """Legacy docs: short_description falls back to description; long only when key absent."""
    if not isinstance(data, dict):
        return data
    legacy_full = str(data.get("description") or "").strip()
    legacy_short = _clamp_short_description(legacy_full)
    if not str(data.get("short_description") or "").strip():
        data = {**data, "short_description": legacy_short}
    if "long_description" not in data:
        data = {**data, "long_description": legacy_full or None}
    elif data.get("long_description") is None and legacy_full:
        data = {**data, "long_description": legacy_full}
    return data


ENTITY_DESCRIPTION_FIELDS = frozenset({"short_description", "long_description", "description"})


def preserve_custom_entity_descriptions(
    existing: Any,
    doc: dict[str, Any],
) -> None:
    """Keep admin-set short/long when rebuilding a doc (e.g. seed)."""
    if not isinstance(existing, dict):
        return
    for field in ("short_description", "long_description"):
        if field not in existing:
            continue
        custom = str(existing.get(field) or "").strip()
        if field == "short_description" and not custom:
            continue
        doc[field] = custom


def sync_entity_descriptions_for_save(
    doc: dict[str, Any],
    *,
    touched: frozenset[str] | None = None,
    prior: dict[str, Any] | None = None,
) -> None:
    """Align short/long/legacy description on write (mirrors UI entityDescriptionsForSave)."""
    # Optional create fields often arrive as None; treat that as "field omitted".
    for field in ("short_description", "long_description"):
        if field in doc and doc.get(field) is None:
            doc.pop(field, None)

    explicit_short = bool(str(doc.get("short_description") or "").strip())
    explicit_long = bool(str(doc.get("long_description") or "").strip())
    legacy = str(doc.get("description") or "").strip()
    long = str(doc.get("long_description") or "").strip()
    short = _clamp_short_description(str(doc.get("short_description") or "").strip())
    prior_doc = prior or {}
    prior_doc_long = str(prior_doc.get("long_description") or "").strip()
    prior_had_explicit_long = bool(prior_doc_long)
    prior_legacy = str(prior_doc.get("description") or "").strip()
    prior_has_short = bool(str(prior_doc.get("short_description") or "").strip())
    prior_legacy_only = (
        bool(prior_legacy)
        and not prior_had_explicit_long
        and not prior_has_short
        and "long_description" not in prior_doc
    )
    prior_migrated = prior is not None and (
        "long_description" in prior_doc or prior_has_short
    )
    prior_cleared_long = (
        prior is not None
        and "long_description" in prior_doc
        and not prior_doc_long
    )
    clearing_explicit_long = (
        touched is not None
        and "long_description" in touched
        and not long
        and prior_had_explicit_long
    )
    # Do not revive an intentional empty long from legacy on short/long-only PUT.
    allow_legacy_long_fallback = bool(
        legacy and not clearing_explicit_long and not prior_cleared_long
    )
    unmigrated_full_legacy = prior_legacy if prior_legacy_only else legacy
    description_touched = touched is not None and "description" in touched

    if touched is None:
        # Fill long from legacy when omitted. Explicit "" is a clear unless the
        # whole short/long pair was left empty alongside a non-empty description
        # (common API create shape that would otherwise wipe description).
        # Skip that create-shape backfill when prior already had long_description
        # (incl. ""), so MCP rediscover does not undo an admin clear.
        if not long and legacy and "long_description" not in doc:
            long = legacy
        elif (
            not long
            and not short
            and legacy
            and doc.get("long_description") == ""
            and "long_description" not in prior_doc
        ):
            long = legacy
        # Omitted short (e.g. still matched prior upstream) → take new legacy card text
        # even when long stays explicitly cleared.
        if not short and legacy and "short_description" not in doc:
            short = _clamp_short_description(legacy)
    else:
        if (
            "description" in touched
            and "long_description" not in touched
            and "short_description" not in touched
        ):
            if prior_doc_long and prior_doc_long != legacy and prior_doc_long != short:
                long = prior_doc_long
            elif prior_migrated and not prior_doc_long:
                long = ""
            else:
                long = legacy
                short = _clamp_short_description(legacy)
        elif (
            "long_description" in touched
            and not long
            and prior_legacy_only
            and not clearing_explicit_long
        ):
            # First migration from description-only Mongo. When the blob is longer
            # than a card, always keep it in long — even if UI edited short away
            # from a prefix (read-time coerce makes forms look migrated).
            # Short legacy (<=256): allow explicit clear when description is touched.
            if len(prior_legacy) > SHORT_DESCRIPTION_MAX_LEN or not description_touched:
                long = prior_legacy
        elif (
            "short_description" in touched
            and not long
            and allow_legacy_long_fallback
            and not description_touched
        ):
            long = unmigrated_full_legacy
        elif (
            "long_description" in touched
            and not long
            and allow_legacy_long_fallback
            and "description" not in touched
            and "short_description" not in touched
        ):
            long = unmigrated_full_legacy

    if not short and long:
        short = _clamp_short_description(long)
    doc["short_description"] = short
    doc["long_description"] = long
    description_only_put = (
        touched is not None
        and "description" in touched
        and "long_description" not in touched
        and "short_description" not in touched
    )
    if (
        touched is None
        and legacy
        and not explicit_short
        and not explicit_long
        and long == legacy
        and len(legacy) > SHORT_DESCRIPTION_MAX_LEN
    ):
        doc["description"] = legacy
    elif description_touched:
        doc["description"] = legacy
    else:
        doc["description"] = short or long


WORKFLOW_EXECUTION_MODES = frozenset({"static", "dynamic"})


def normalize_workflow_execution_mode(value: object, *, default: str = "dynamic") -> str:
    """Canonical lowercase execution_mode for storage, validation, and runtime guards."""
    if value is None or value == "":
        return default
    mode = str(value).strip().lower()
    if mode not in WORKFLOW_EXECUTION_MODES:
        raise ValueError(f"execution_mode must be 'static' or 'dynamic', got {value!r}")
    return mode

def _normalize_optional_string_list(value: Optional[List[str]]) -> Optional[List[str]]:
    """Trim string lists while preserving None and explicit empty lists."""
    if value is None:
        return None

    normalized: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


# =====================================================================
# Agent Configurations
# =====================================================================

class AgentConfigurationCreate(BaseModel):
    """Request body for creating a new agent configuration."""
    id: str = Field(..., pattern=AGENT_ID_PATTERN, description=AGENT_ID_DESCRIPTION)
    type: str = Field(..., description="Agent type for auction matching")
    name: str = Field(
        ...,
        description=(
            "Wire name or legacy UI label. A trimmed, nonblank name differing "
            "from id overrides display_name."
        ),
    )
    display_name: Optional[str] = Field(
        None,
        description="UI label; preserved when name is blank or its trimmed value equals id",
    )
    description: str = Field("", description="Legacy description; kept for compatibility")
    short_description: Optional[str] = Field(
        None,
        max_length=SHORT_DESCRIPTION_MAX_LEN,
        description="Card/table text; falls back to description when omitted on read",
    )
    long_description: Optional[str] = Field(
        None,
        description="Optional detailed description for forms",
    )
    agent_class: str = Field("GenericAgent", description="Python class name")
    model: str = Field(DEFAULT_AGENT_MODEL, description="LLM model identifier")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt: str = Field(..., description="System prompt for the agent")
    allowed_phases: List[str] = Field(default_factory=list)
    eval_keywords: List[str] = Field(
        default_factory=list,
        description="DEPRECATED, ignored by the runtime — bids are LLM-evaluated. Kept so old bundles/exports still import.",
    )
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="Built-in / non-MCP tool ids the agent may invoke",
    )
    allowed_mcp_tools: List[str] = Field(
        default_factory=list,
        description="MCP tool public ids (server.tool) the agent may invoke",
    )
    allowed_delegation_targets: Optional[List[str]] = Field(
        None,
        description="Allowed target agent ids for delegate_to_agent. None/[] disables delegation; ['*'] allows any non-self same-tenant or __system__ project agent.",
    )
    output_save_key: Optional[str] = Field(None, description="Key to save output in shared context")
    use_streaming: Optional[bool] = Field(None)
    exact_tool_values: bool = Field(False, strict=True)
    reasoning_effort: Optional[str] = Field(None)
    step_limit: Optional[int] = Field(None, ge=1, le=100)
    enabled: bool = Field(True)
    plugins: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Per-plugin partial config overrides: plugin name -> settings (agent level of the agent > run > tenant chain, ADR-0004)",
    )

    @field_validator("allowed_tools", "allowed_mcp_tools")
    @classmethod
    def normalize_allowed_tool_lists(cls, value: Optional[List[str]]) -> List[str]:
        return _normalize_optional_string_list(value) or []

    @field_validator("allowed_delegation_targets")
    @classmethod
    def normalize_allowed_delegation_targets(
        cls,
        value: Optional[List[str]],
    ) -> Optional[List[str]]:
        return _normalize_optional_string_list(value)

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("model must be a non-empty string")
        return text

    @field_validator("temperature", mode="before")
    @classmethod
    def reject_boolean_temperature(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("temperature must be a number, not a boolean")
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def normalize_reasoning_effort(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str):
            return value.strip() or None
        return value


class AgentConfigurationUpdate(BaseModel):
    """Request body for updating an existing agent configuration (partial)."""
    type: Optional[str] = Field(None, description="Agent type for auction matching")
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = Field(None, max_length=SHORT_DESCRIPTION_MAX_LEN)
    long_description: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None
    allowed_phases: Optional[List[str]] = None
    eval_keywords: Optional[List[str]] = Field(
        None, description="DEPRECATED, ignored by the runtime"
    )
    allowed_tools: Optional[List[str]] = None
    allowed_mcp_tools: Optional[List[str]] = None
    allowed_delegation_targets: Optional[List[str]] = None
    output_save_key: Optional[str] = None
    use_streaming: Optional[bool] = None
    exact_tool_values: Optional[bool] = Field(None, strict=True)
    reasoning_effort: Optional[str] = None
    step_limit: Optional[int] = Field(None, ge=1, le=100)
    enabled: Optional[bool] = None
    plugins: Optional[Dict[str, Dict[str, Any]]] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("type must be a non-empty string")
        return text

    @field_validator("allowed_tools", "allowed_mcp_tools")
    @classmethod
    def normalize_allowed_tool_lists(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        return _normalize_optional_string_list(value)

    @field_validator("allowed_delegation_targets")
    @classmethod
    def normalize_allowed_delegation_targets(
        cls,
        value: Optional[List[str]],
    ) -> Optional[List[str]]:
        return _normalize_optional_string_list(value)

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("model cannot be null")
        text = value.strip()
        if not text:
            raise ValueError("model must be a non-empty string")
        return text

    @field_validator("temperature", mode="before")
    @classmethod
    def reject_boolean_temperature(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("temperature must be a number, not a boolean")
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def normalize_reasoning_effort(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str):
            return value.strip() or None
        return value


class AgentBulkEnabledItem(BaseModel):
    """Single agent enabled flag update within a bulk request."""
    agent_id: str = Field(..., pattern=AGENT_ID_PATTERN, description=AGENT_ID_DESCRIPTION)
    tenant_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Root-only tenant selector for agent ids shared across tenants",
    )
    enabled: bool

    @field_validator("tenant_id")
    @classmethod
    def normalize_tenant_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("tenant_id must not be empty")
        return normalized


class AgentBulkEnabledRequest(BaseModel):
    """Request body for PATCH /configurations/agents/bulk-enabled."""
    updates: List[AgentBulkEnabledItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Agent enabled updates (max 100 per request)",
    )

    @model_validator(mode="after")
    def validate_unique_agent_ids(self) -> "AgentBulkEnabledRequest":
        selectors_by_agent: dict[str, set[str | None]] = {}
        for item in self.updates:
            existing = selectors_by_agent.setdefault(item.agent_id, set())
            if item.tenant_id in existing or (
                existing and (item.tenant_id is None or None in existing)
            ):
                raise ValueError(
                    "duplicate agent_id values require distinct explicit tenant_id values"
                )
            existing.add(item.tenant_id)
        return self


class AgentBulkEnabledFailure(BaseModel):
    """Per-item failure in a bulk enabled update."""
    agent_id: str
    tenant_id: Optional[str] = None
    error: str


class AgentBulkEnabledResponse(BaseModel):
    """Response for PATCH /configurations/agents/bulk-enabled.

    HTTP status is always 200 for a valid request. When ``updated`` is empty and
    ``failed`` is non-empty, every item in the batch failed.
    """
    updated: List[str] = Field(
        default_factory=list,
        description="Agent ids successfully updated in this batch",
    )
    failed: List[AgentBulkEnabledFailure] = Field(
        default_factory=list,
        description="Per-item failures; non-empty with empty updated means full batch failure",
    )


class AgentWorkflowUsageResponse(BaseModel):
    """Workflow definition that directly references an agent."""
    id: str
    name: str


class AgentConfigurationResponse(BaseModel):
    """Response body for a single agent configuration."""
    id: str = Field(..., validation_alias="_id")
    tenant_id: Optional[str] = None
    type: str
    name: str
    display_name: Optional[str] = None
    description: str = ""
    short_description: str = ""
    long_description: Optional[str] = None
    agent_class: str = "GenericAgent"
    model: str = DEFAULT_AGENT_MODEL
    temperature: Optional[float] = None
    system_prompt: str = ""
    allowed_phases: List[str] = Field(default_factory=list)
    eval_keywords: List[str] = Field(
        default_factory=list,
        description="DEPRECATED, ignored by the runtime",
    )
    allowed_tools: List[str] = Field(default_factory=list)
    allowed_mcp_tools: List[str] = Field(default_factory=list)
    allowed_delegation_targets: Optional[List[str]] = None
    output_save_key: Optional[str] = None
    use_streaming: Optional[bool] = None
    exact_tool_values: bool = Field(False, strict=True)
    reasoning_effort: Optional[str] = None
    step_limit: Optional[int] = None
    enabled: bool = True
    plugins: Optional[Dict[str, Dict[str, Any]]] = None
    dangling_references: List[str] = Field(
        default_factory=list,
        description="Allow-list entries that do not resolve in this tenant (warn-only)",
    )
    workflow_usage_count: int = Field(
        0,
        description="Number of visible workflow definitions that directly reference this agent",
    )
    workflow_usage: List[AgentWorkflowUsageResponse] = Field(
        default_factory=list,
        description="Visible workflow definitions that directly reference this agent",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the agent was created"
    )
    created_by: str = Field(
        "system",
        description="ID of the user who created the agent"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the last update"
    )
    updated_by: str = Field(
        "system",
        description="ID of the user who last updated the agent"
    )
    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def coerce_objectids(cls, data):
        """Convert any ObjectId fields to str before validation."""
        if isinstance(data, dict):
            for key in ("created_by", "updated_by", "_id", "id"):
                if key in data and type(data[key]).__name__ == "ObjectId":
                    data[key] = str(data[key])
            data.setdefault("allowed_mcp_tools", [])
            data = _coerce_entity_descriptions(data)
        return data

    @field_validator("allowed_delegation_targets")
    @classmethod
    def normalize_allowed_delegation_targets(
        cls,
        value: Optional[List[str]],
    ) -> Optional[List[str]]:
        return _normalize_optional_string_list(value)


class PhaseDictionaryItem(BaseModel):
    """Single phase entry in GET /configurations/phases."""
    id: str
    label: str


# =====================================================================
# Workflow Definitions
# =====================================================================

def _normalize_context_key_list(value: Optional[List[str]]) -> Optional[List[str]]:
    """Trim context key lists while preserving explicit empty lists."""
    return _normalize_optional_string_list(value)


def _normalize_contract_entry_list(value: Optional[List[Any]]) -> Optional[List[Any]]:
    """Normalize a node reads/writes list that may mix plain string keys with
    structured A2A contract entries.

    Phase nodes use plain string keys; a2a_agent nodes use dict entries
    ({"key":.., "context_key":.., "part":..} for reads,
    {"artifact_name":.., "context_key":.., "required":..} for writes) that the A2A
    engine consumes as objects. Strings are trimmed + de-duplicated exactly as the
    string-only normalizer did; dicts pass through verbatim. An all-string list is
    returned unchanged, so every existing config validates and stores identically.
    """
    if value is None:
        return None
    normalized: List[Any] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            normalized.append(item)
            continue
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


class WorkflowNode(BaseModel):
    """A single node in the workflow DAG."""
    model_config = ConfigDict(extra="forbid")
    id: str
    type: str = Field(
        ...,
        description="start | end | phase | approval_gate | execution | deploy | a2a_agent | validator | tool | map",
    )
    operation: Optional[str] = Field(
        None, description="tool only: MCP operation the engine calls, e.g. snapshot_web_sources"
    )
    server: Optional[str] = Field(
        None, description="tool only: MCP server id that provides the operation"
    )
    binding: Optional[str] = Field(
        None,
        description="tool: task type whose argument and output bindings fill the call;"
        " map (optional): binding that picks the items and merges the answers"
        " into the value the node writes",
    )
    items_from: Optional[str] = Field(
        None, description="map only: dotted context path to the list whose items become tasks"
    )
    item_key: Optional[str] = Field(
        None,
        description="map only: field of a dict item that names it in progress events"
        " and the node result; short id strings and ints name themselves",
    )
    batch_size: Optional[int] = Field(
        None, strict=True, ge=1, le=50, description="map only: items per task (default 1)"
    )
    concurrency: Optional[int] = Field(
        None, strict=True, ge=1, le=16, description="map only: tasks running at once (default 4)"
    )
    max_item_attempts: Optional[int] = Field(
        None, strict=True, ge=1, le=5, description="map only: attempts per item (default 2)"
    )
    item_checks: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="map only: json_schema / value_relation checks on each answer, "
        "keyed 'item' or 'item.<path>'; 'map_item' is the input item",
    )
    task_type: Optional[str] = None
    description: Optional[str] = None
    agent_selection: Optional[str] = None
    agent_type: Optional[str] = None
    phase_label: Optional[str] = None
    label: Optional[str] = None
    html_from: Optional[str] = Field(
        None,
        min_length=1,
        description="deploy nodes: dotted context path to an HTML attachment id to publish",
    )
    server_id: Optional[str] = Field(
        None,
        description="a2a_agent only: id of the registered A2A server this node calls",
    )
    a2a_poll_interval_seconds: Optional[float] = Field(
        None,
        gt=0,
        description="a2a_agent only: seconds between tasks/get requests for a working task",
    )
    a2a_task_timeout_seconds: Optional[float] = Field(
        None,
        gt=0,
        description="a2a_agent only: maximum seconds to wait for a working task",
    )
    # Phase nodes: plain string context keys. a2a_agent nodes: structured dict entries
    # (reads {"key","context_key","part"}, writes {"artifact_name","context_key","required"}).
    reads: Optional[List[Union[str, Dict[str, Any]]]] = None
    writes: Optional[List[Union[str, Dict[str, Any]]]] = None
    show_keys: Optional[List[str]] = Field(
        None,
        description="approval_gate only: context keys to display in the approval card",
    )
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    max_tool_failures: Optional[int] = Field(
        None, strict=True, ge=1, le=10,
        description="Phase attempt stops when any tool reaches this many failed calls. "
        "Successful calls do not reset the count. Absent disables the limit.",
    )
    max_output_repairs: Optional[int] = Field(
        None,
        strict=True,
        ge=0,
        le=5,
        description="phase and map (per item task): extra model turns, in the same "
        "session, that an exact-value agent gets when its final JSON names a "
        "reference the run cannot resolve. Absent or 0 fails the task on the first "
        "such answer.",
    )
    output_schema: Optional[Dict[str, Any]] = None
    interaction_schema: Optional[Dict[str, Any]] = None
    can_delegate: bool = False
    reviewers: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "phase only: per-target delegation reviewers wrapped around "
            "delegate_to_agent, e.g. {'targets': {'coscientist_planner': "
            "{'human': 'post'}, 'research_worker': {'critic': ['pre','post']}}}. "
            "Consumed by DelegationReviewers (AppFactory-154)."
        ),
    )
    checks: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "validator only: deterministic format checks run against "
            "shared_context. Each check: {key, kind='structural'|'json_schema'|'id_set_equals'|'value_relation', "
            "...params}."
        ),
    )
    refine_checks: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "validator only: value-relation checks added when a pending gate "
            "re-runs its upstream phase"
        ),
    )
    max_reject_retries: Optional[int] = Field(
        None,
        ge=0,
        le=10,
        description=(
            "validator only: cap on how many times this validator may reject-and-retry "
            "before the run fails loudly (validator_retry_limit_exhausted). Bounds LLM "
            "spend on the validator→rejected→phase loop below the global max_iterations. "
            "Absent = no per-validator cap (old behavior). AppFactory-77 F3, ADR-0011."
        ),
    )

    @field_validator("show_keys")
    @classmethod
    def normalize_context_keys(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _normalize_context_key_list(value)

    @field_validator("reads", "writes")
    @classmethod
    def normalize_contract_entries(cls, value: Optional[List[Any]]) -> Optional[List[Any]]:
        return _normalize_contract_entry_list(value)


class WorkflowEdge(BaseModel):
    """A directed edge in the workflow DAG."""
    source: str = Field(..., alias="from")
    target: str = Field(..., alias="to")
    condition: Optional[str] = None

    model_config = {"populate_by_name": True}


class WorkflowPrompt(BaseModel):
    """A named example prompt offered when launching a project with this workflow.

    Picking one on the launch screen pre-fills the request field with ``text``;
    the user can still edit it before starting.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(..., min_length=1, max_length=WORKFLOW_PROMPT_NAME_MAX_LEN)
    text: str = Field(..., min_length=1, max_length=WORKFLOW_PROMPT_TEXT_MAX_LEN)


class WorkflowDefinitionCreate(BaseModel):
    """Request body for creating a new workflow definition."""
    id: str = Field(
        ...,
        pattern=AGENT_ID_PATTERN,
        description=(
            "Unique workflow wire id — lowercase letters, digits, underscores; must "
            "start with a letter; length 2-64. Example: requirements_build"
        ),
    )
    name: str
    description: str = ""
    short_description: Optional[str] = Field(None, max_length=SHORT_DESCRIPTION_MAX_LEN)
    long_description: Optional[str] = None
    execution_mode: Literal["static", "dynamic"] = "dynamic"
    default_reads: Optional[List[str]] = None
    run_timeout_seconds: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Wall-clock budget for one engine run of this workflow, in seconds. "
            "None disables the guard (default). Enforced by cancelling the "
            "in-flight node on expiry (AppFactory-77, ADR-0012)."
        ),
    )
    max_iterations: Optional[int] = Field(
        None,
        ge=1,
        description="Max DAG-walk iterations per run. None = engine default (50).",
    )
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    is_default: bool = False
    prompts: List[WorkflowPrompt] = Field(
        default_factory=list,
        description=(
            "Example prompts (name + text) offered on the launch screen; picking one "
            "pre-fills the request field. Empty by default."
        ),
    )

    @field_validator("execution_mode", mode="before")
    @classmethod
    def normalize_execution_mode(cls, value: object) -> object:
        if value is None:
            return "dynamic"
        return normalize_workflow_execution_mode(value)

    @field_validator("default_reads")
    @classmethod
    def normalize_default_reads(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _normalize_context_key_list(value)


class WorkflowDefinitionUpdate(BaseModel):
    """Request body for updating an existing workflow definition (partial)."""
    name: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = Field(None, max_length=SHORT_DESCRIPTION_MAX_LEN)
    long_description: Optional[str] = None
    execution_mode: Optional[Literal["static", "dynamic"]] = None
    default_reads: Optional[List[str]] = None
    run_timeout_seconds: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Wall-clock budget for one engine run of this workflow, in seconds. "
            "None disables the guard (default). Enforced by cancelling the "
            "in-flight node on expiry (AppFactory-77, ADR-0012)."
        ),
    )
    max_iterations: Optional[int] = Field(
        None,
        ge=1,
        description="Max DAG-walk iterations per run. None = engine default (50).",
    )
    nodes: Optional[List[WorkflowNode]] = None
    edges: Optional[List[WorkflowEdge]] = None
    is_default: Optional[bool] = None
    prompts: Optional[List[WorkflowPrompt]] = None

    @field_validator("execution_mode", mode="before")
    @classmethod
    def normalize_execution_mode(cls, value: object) -> object:
        if value is None:
            return None
        return normalize_workflow_execution_mode(value)

    @field_validator("default_reads")
    @classmethod
    def normalize_default_reads(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _normalize_context_key_list(value)


class WorkflowDefinitionResponse(BaseModel):
    """Response body for a single workflow definition."""
    id: str = Field(..., validation_alias="_id")
    name: str
    display_name: Optional[str] = None
    description: str = ""
    short_description: str = ""
    long_description: Optional[str] = None
    execution_mode: Literal["static", "dynamic"] = "dynamic"
    default_reads: Optional[List[str]] = None
    # No ge=1 here (unlike Create/Update): a read must tolerate whatever a
    # non-API write path (direct Mongo edit, migration) may have stored — the
    # engine's _resolve_max_iterations/_resolve_run_deadline already fall back
    # on invalid values at runtime, so GET must not 500 on the same doc.
    run_timeout_seconds: Optional[int] = Field(
        None,
        description=(
            "Wall-clock budget for one engine run of this workflow, in seconds. "
            "None disables the guard (default). Enforced by cancelling the "
            "in-flight node on expiry (AppFactory-77, ADR-0012)."
        ),
    )
    max_iterations: Optional[int] = Field(
        None,
        description="Max DAG-walk iterations per run. None = engine default (50).",
    )
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    is_default: bool = False
    # Lenient (dicts, not WorkflowPrompt) for the same reason as nodes/edges above:
    # a GET must not 500 on a row a non-API write path may have stored malformed.
    prompts: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def coerce_descriptions(cls, data):
        return _coerce_entity_descriptions(data)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def coerce_execution_mode(cls, value: object) -> object:
        if value is None:
            return "dynamic"
        return normalize_workflow_execution_mode(value)


class DAGValidateRequest(BaseModel):
    """Request body for standalone DAG validation (no workflow_id needed)."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    default_reads: Optional[List[str]] = None
    execution_mode: Literal["static", "dynamic"] = "dynamic"

    @field_validator("execution_mode", mode="before")
    @classmethod
    def normalize_execution_mode(cls, value: object) -> object:
        if value is None:
            return "dynamic"
        return normalize_workflow_execution_mode(value)

    @field_validator("default_reads")
    @classmethod
    def normalize_default_reads(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _normalize_context_key_list(value)


class DAGValidationResult(BaseModel):
    """Result of DAG validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =====================================================================
# Tool Configurations
# =====================================================================

WIRE_TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"
MCP_RPC_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"
TOOL_CATEGORY_UNASSIGNED = "__unassigned__"


def _normalize_stored_tool_category(value: object) -> str:
    """Map wire unassigned bucket back to empty stored category."""
    text = str(value or "").strip()
    if text == TOOL_CATEGORY_UNASSIGNED:
        return ""
    return text


def _validate_wire_tool_name(value: str, *, field_label: str = "name") -> str:
    raw = str(value or "").strip()
    if not raw or not re.fullmatch(WIRE_TOOL_NAME_PATTERN, raw):
        raise ValueError(
            f"{field_label} must start with a lowercase letter, be 2-64 chars, "
            "and use only lowercase letters, digits, underscores"
        )
    return raw


def _validate_mcp_rpc_name(value: str, *, field_label: str = "name") -> str:
    raw = str(value or "").strip()
    if not raw or not re.fullmatch(MCP_RPC_NAME_PATTERN, raw):
        raise ValueError(
            f"{field_label} must match [a-zA-Z0-9_-]+ "
            "(letters, numbers, hyphen, underscore; no dots)"
        )
    return raw


def _normalize_optional_rpc_name(value: object) -> Optional[str]:
    """Blank ``rpc_name`` reads as "not submitted": the edit modal always sends
    the field, and an empty box must not clear the tools/call contract name."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return _validate_mcp_rpc_name(raw, field_label="rpc_name")


class ToolConfigurationCreate(BaseModel):
    """Request body for creating a new tool configuration."""
    id: str = Field(..., description="Unique tool identifier (e.g. 'create')")
    name: str = Field(..., max_length=128)
    rpc_name: Optional[str] = Field(None, max_length=128)
    description: str = ""
    short_description: Optional[str] = Field(None, max_length=SHORT_DESCRIPTION_MAX_LEN)
    long_description: Optional[str] = None
    category: str = Field("general", description="Tool category: file_ops, search, shell, web, task_mgmt, mcp")
    source: str = Field("builtin", description="builtin | mcp_server")
    tool_schema: Optional[Dict[str, Any]] = Field(None, alias="schema", description="OpenAI function-calling schema")
    mcp_server: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    version: str = Field(default="", max_length=32)

    model_config = {"populate_by_name": True}

    @field_validator("rpc_name", mode="before")
    @classmethod
    def normalize_rpc_name_create(cls, value: object) -> Optional[str]:
        return _normalize_optional_rpc_name(value)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category_create(cls, value: object) -> str:
        if value is None:
            return "general"
        return _normalize_stored_tool_category(value)

    @field_validator("version", mode="before")
    @classmethod
    def normalize_version_create(cls, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if "\n" in text or "\r" in text:
            raise ValueError("version must not contain newlines")
        return text

    @model_validator(mode="after")
    def validate_name_for_source(self) -> "ToolConfigurationCreate":
        if self.source == "mcp_server":
            self.name = _validate_mcp_rpc_name(self.name)
        else:
            self.name = _validate_wire_tool_name(self.name)
        return self


class ToolConfigurationUpdate(BaseModel):
    """Request body for updating an existing tool configuration (partial)."""
    name: Optional[str] = Field(None, max_length=128)
    rpc_name: Optional[str] = Field(None, max_length=128)
    description: Optional[str] = None
    short_description: Optional[str] = Field(None, max_length=SHORT_DESCRIPTION_MAX_LEN)
    long_description: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None
    tool_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")
    mcp_server: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    version: Optional[str] = Field(None, max_length=32)

    model_config = {"populate_by_name": True}

    @field_validator("rpc_name", mode="before")
    @classmethod
    def normalize_rpc_name_update(cls, value: object) -> Optional[str]:
        return _normalize_optional_rpc_name(value)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category_update(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        return _normalize_stored_tool_category(value)

    @field_validator("version", mode="before")
    @classmethod
    def normalize_version_update(cls, value: object) -> Optional[str]:
        if value is None:
            return ""
        text = str(value).strip()
        if "\n" in text or "\r" in text:
            raise ValueError("version must not contain newlines")
        return text

    @field_validator("name")
    @classmethod
    def validate_optional_name(cls, value: Optional[str]) -> Optional[str]:
        # Accepts wire OR MCP-RPC shape so MCP PUT can rename via rpc_name path.
        # Builtin tools: strict wire invariant (<=64, snake_case) is enforced in
        # update_tool_configuration — source lives on the stored doc, not this body.
        if value is None:
            return value
        raw = str(value).strip()
        if re.fullmatch(WIRE_TOOL_NAME_PATTERN, raw):
            return raw
        if re.fullmatch(MCP_RPC_NAME_PATTERN, raw):
            return raw
        raise ValueError(
            "name must be a wire tool id (lowercase snake_case) or an MCP RPC name "
            "([a-zA-Z0-9_-]+)"
        )


class ToolConfigurationResponse(BaseModel):
    """Response body for a single tool configuration."""
    id: str = Field(..., validation_alias="_id")
    name: str = Field(..., max_length=64)
    rpc_name: Optional[str] = Field(
        None,
        description="MCP RPC contract name (tools/call); path A storage field",
    )
    description: str = ""
    short_description: str = ""
    long_description: Optional[str] = None
    category: str = "general"
    source: str = "builtin"
    tool_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")
    mcp_server: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    tenant_id: Optional[str] = None
    version: str = ""

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def coerce_descriptions(cls, data):
        data = _coerce_entity_descriptions(data)
        if not isinstance(data, dict):
            return data
        if data.get("version") is None:
            data = {**data, "version": ""}
        # Match list/dictionary bucket: missing/blank category → __unassigned__
        # (default "general" only for callers that never set the key after this coerce).
        raw_category = data.get("category")
        if raw_category is None or not str(raw_category).strip():
            data = {**data, "category": TOOL_CATEGORY_UNASSIGNED}
        else:
            data = {**data, "category": str(raw_category).strip()}
        return data


class MCPToolBatchItem(BaseModel):
    """One tool in a batch create (per-tool fields only; server/metadata are top-level)."""

    id: str = Field(..., description="Public tool id (e.g. github.search_issues)")
    name: str = Field(..., max_length=128, description="MCP RPC / tools/call name")
    description: str = ""
    tool_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_mcp_batch_name(self) -> "MCPToolBatchItem":
        self.name = _validate_mcp_rpc_name(self.name)
        return self


class MCPToolsBatchCreate(BaseModel):
    """All-or-nothing create of several MCP tools for one server."""

    mcp_server: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tools: List[MCPToolBatchItem] = Field(..., min_length=1)


class MCPToolsBatchCreateResponse(BaseModel):
    """Successful batch create payload."""

    created_count: int
    tools: List[ToolConfigurationResponse]


class McpToolGroupResponse(BaseModel):
    """Grouped MCP tools catalog entry (one MCP server)."""
    id: str
    label: str
    tools_count: int
    tools: List[ToolConfigurationResponse]


class ToolCategoriesResponse(BaseModel):
    """Distinct builtin tool categories for the caller's tenant scope."""
    items: List[str] = Field(default_factory=list)


# =====================================================================
# Outbound auth (shared by A2A servers and external MCP servers — AppFactory-313)
# =====================================================================

class A2AAuthType(str, Enum):
    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"


class A2AOAuth2GrantType(str, Enum):
    # Both grants run headless (no browser). client_credentials uses a confidential
    # client's own identity (preferred for agent-to-agent); refresh_token exchanges a
    # previously-obtained (offline) refresh token for fresh access tokens.
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"


class A2AAuthConfig(BaseModel):
    # Named for A2A historically; now the single outbound-auth block for A2A AND external
    # MCP servers (same shape, same validator, same provider) so the two cannot drift.
    type: A2AAuthType = Field(default=A2AAuthType.NONE)
    token: str | None = Field(None, description="Direct token/key — stored in the DB and masked in responses (alternative to token_env)")
    token_env: str | None = Field(None, description="Name of a backend env var holding the token (alternative to token); used only when token is blank")
    header_name: str | None = Field(None, description="Custom header name for API key (default: X-API-Key); with oauth2, an extra header carrying token/token_env next to the bearer, e.g. a gateway key")

    # OAuth2 fields (type == "oauth2"): the client fetches a short-lived bearer token
    # from token_url and refreshes it automatically, instead of storing a static token.
    token_url: str | None = Field(None, description="OAuth2 token endpoint, e.g. Keycloak .../protocol/openid-connect/token")
    client_id: str | None = Field(None, description="OAuth2 client_id")
    client_secret: str | None = Field(None, description="OAuth2 client_secret (confidential client) — stored in the DB and masked in responses (alternative to client_secret_env)")
    client_secret_env: str | None = Field(None, description="Backend env var holding the client_secret (used only when client_secret is blank)")
    grant_type: A2AOAuth2GrantType | None = Field(None, description="client_credentials (service account) or refresh_token (offline token)")
    refresh_token: str | None = Field(None, description="Refresh/offline token for the refresh_token grant — stored in the DB and masked in responses (alternative to refresh_token_env)")
    refresh_token_env: str | None = Field(None, description="Backend env var holding the refresh token (used only when refresh_token is blank)")
    scope: str | None = Field(None, description="Space-separated OAuth2 scopes, e.g. 'openid profile email offline_access'")

    @model_validator(mode='after')
    def validate_auth_config(self) -> 'A2AAuthConfig':
        if self.type in [A2AAuthType.BEARER, A2AAuthType.API_KEY] and not (self.token or self.token_env):
            raise ValueError(f'{self.type.value} auth requires either a token or token_env')
        if self.type == A2AAuthType.OAUTH2:
            if not self.token_url or not self.client_id:
                raise ValueError('oauth2 auth requires token_url and client_id')
            if not self.grant_type:
                raise ValueError('oauth2 auth requires grant_type (client_credentials or refresh_token)')
            if self.grant_type == A2AOAuth2GrantType.CLIENT_CREDENTIALS and not (self.client_secret or self.client_secret_env):
                raise ValueError('client_credentials grant requires client_secret or client_secret_env')
            if self.grant_type == A2AOAuth2GrantType.REFRESH_TOKEN and not (self.refresh_token or self.refresh_token_env):
                raise ValueError('refresh_token grant requires refresh_token or refresh_token_env')
            has_header_value = bool(self.token or self.token_env)
            if self.header_name and not has_header_value:
                raise ValueError('oauth2 header_name requires a token or token_env')
            if has_header_value and not self.header_name:
                raise ValueError('oauth2 token/token_env is only sent under header_name; set header_name')
            if self.header_name and self.header_name.strip().lower() == 'authorization':
                raise ValueError('oauth2 header_name cannot be Authorization: the bearer token uses it')
        return self


# Directly-stored auth secrets to mask in API responses; the matching *_env fields are
# variable NAMES (not secrets). Shared by A2A and external-MCP response masking.
AUTH_SECRET_FIELDS = ("token", "client_secret", "refresh_token")


# =====================================================================
# MCP Server Discovery
# =====================================================================

class MCPHeaderPair(BaseModel):
    """A single HTTP header name/value pair."""
    name: str
    value: str


class MCPServerDiscoverRequest(BaseModel):
    """Request to discover tools from an external MCP server."""
    server_id: str = Field(..., description="Unique server identifier (e.g. chembl)")
    endpoint: str = Field("", description="MCP endpoint URL; required for http/streamable-http mode")
    mode: str = Field("http", description="Transport mode: http | streamable-http | stdio")
    # Upper bound is enforced at the API layer against the owner tenant's ceiling
    # (mcp_call_timeout_max_seconds, default 300) — not hardwired here — so a tenant
    # that raised its ceiling can carry a higher timeout (AppFactory-314).
    timeout_seconds: float = Field(30.0, ge=1.0)
    # HTTP / streamable-http
    headers: Optional[List[MCPHeaderPair]] = Field(None, description="Static HTTP headers (e.g. X-User-Id); sent alongside `auth`")
    auth: Optional[A2AAuthConfig] = Field(
        None,
        description="Outbound auth for http/streamable-http (same block as A2A): oauth2 "
                    "client_credentials fetches and refreshes a short-lived token itself, "
                    "instead of a static header that expires. Omit or type:none for no auth.",
    )
    # stdio — docker image
    image: Optional[str] = Field(
        None,
        description="Docker image for local MCP runtime; when empty backend may resolve system default",
    )
    docker_env_vars: Optional[Dict[str, str]] = Field(None, description="Extra env vars for docker run (-e KEY=VALUE)")
    docker_cmd_args: Optional[List[str]] = Field(None, description="Positional args appended after image name")
    container_port: Optional[int] = Field(
        None,
        ge=1,
        le=65535,
        description="TCP port the MCP process listens on inside the container (from EXPOSE); host port is auto-assigned",
    )
    endpoint_path: Optional[str] = Field(
        None,
        description="HTTP path on the container endpoint (default /mcp)",
    )
    # stdio — local command
    command: Optional[str] = Field(None, description="Local command to run as stdio MCP server")
    command_args: Optional[List[str]] = Field(None, description="Arguments for the local command")
    command_env: Optional[Dict[str, str]] = Field(None, description="Environment variables for the local command")


class MCPDiscoveredTool(BaseModel):
    """A tool discovered from an MCP server."""
    name: str
    description: str = ""
    tool_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")
    mcp_server: str

    model_config = {"populate_by_name": True}


class MCPServerDiscoverResponse(BaseModel):
    """Response payload for MCP discovery."""
    server_id: str
    endpoint: str
    mode: str
    tools: List[MCPDiscoveredTool] = Field(default_factory=list)


class MCPServerHealthItem(BaseModel):
    """One MCP server result from batch health-check (saved tool configs)."""
    server_id: str
    status: str = Field(
        ...,
        description="ok | error | skipped | policy_blocked | not_found",
    )
    mode: str = ""
    endpoint_host: str = Field("", description="HTTP host or label stdio/docker-local")
    # Same stable key as discovery dedupe — UI merge must not collapse by server_id alone.
    runtime_key: str = ""
    tools_count: Optional[int] = None
    message: str = ""
    duration_ms: Optional[float] = None


class MCPServersHealthCheckRequest(BaseModel):
    """Optional filter for batch MCP health-check."""

    server_ids: Optional[List[str]] = Field(
        None,
        description=(
            "If set and non-empty: probe only these server_ids (UI mcp_server names). "
            "Omit/null: all tenant MCP servers. Empty list: no probes, empty servers[]."
        ),
    )
    tenant_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Root-only tenant selector for a targeted health check",
    )


class MCPServersHealthResponse(BaseModel):
    """Batch health-check for all distinct MCP servers referenced by tenant tools."""
    checked_at: str
    servers: List[MCPServerHealthItem] = Field(default_factory=list)
    summary_ok: int = 0
    summary_error: int = 0
    summary_skipped: int = 0
    summary_policy_blocked: int = 0
    summary_not_found: int = 0


class McpServerLifecycleRequest(BaseModel):
    """Target tenant for a server-level MCP lifecycle action."""

    tenant_id: Optional[str] = Field(None, max_length=128)

    @field_validator("tenant_id")
    @classmethod
    def normalize_tenant_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("tenant_id must not be empty")
        return value


class McpServerLifecycleResponse(BaseModel):
    status: str
    tenant_id: str
    server_id: str
    tools_count: int = 0
    containers_removed: int = 0
    agent_tool_links_removed: int = 0


class McpServerConnectionUpdate(BaseModel):
    """Partial connection metadata fan-out to all tenant-owned tools on a server."""

    tenant_id: Optional[str] = Field(None, max_length=128)
    endpoint: Optional[str] = None
    mode: Optional[str] = None
    timeout_seconds: Optional[float] = Field(None, ge=1.0)
    headers: Optional[List[MCPHeaderPair]] = None
    auth: Optional[A2AAuthConfig] = None

    @field_validator("tenant_id")
    @classmethod
    def normalize_connection_tenant_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("tenant_id must not be empty")
        return value

    @field_validator("endpoint", "mode", mode="before")
    @classmethod
    def strip_optional_str(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("must not be empty")
        return text


class McpServerConnectionUpdateResponse(BaseModel):
    server_id: str
    tenant_id: str
    tools_updated: int


# =====================================================================
# Run Configurations
# =====================================================================

class RunAgentConfigOverride(BaseModel):
    """Typed per-agent runtime overrides; omitted values inherit defaults."""

    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    reasoning_effort: Optional[str] = None
    step_limit: Optional[int] = Field(None, ge=1)

    model_config = ConfigDict(extra="forbid")


class RunConfigurationCreate(BaseModel):
    """Request body for creating a new run configuration."""
    id: str = Field(..., alias="_id", description="Unique run configuration identifier")
    name: str = Field(..., description="Human-readable run configuration name")
    description: str = Field("", description="Legacy description; kept for compatibility")
    short_description: Optional[str] = Field(None, max_length=SHORT_DESCRIPTION_MAX_LEN)
    long_description: Optional[str] = None
    models: Dict[str, str] = Field(default_factory=dict, description="Subsystem -> model mapping (legacy)")
    agent_configs: Optional[Dict[str, RunAgentConfigOverride]] = Field(
        default=None,
        description="Per-agent config overrides: agent_id -> {model, reasoning_effort, temperature, step_limit}",
    )
    plugins: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Per-plugin config: plugin name -> settings (run level of the agent > run > tenant chain, ADR-0004)",
    )
    is_default: bool = Field(False, description="Whether this is the default run configuration")
    approval_mode: Optional[Literal["human", "auto"]] = Field(
        None,
        description="Gate approval mode a launch inherits when it doesn't pick its own: 'human' pauses at each gate, 'auto' skips them. None leaves the decision to the launch (which defaults to human).",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional opaque metadata (e.g. seed_description_baseline)",
    )

    model_config = ConfigDict(populate_by_name=True)


class RunConfigurationUpdate(BaseModel):
    """Request body for updating an existing run configuration (partial)."""
    name: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = Field(None, max_length=SHORT_DESCRIPTION_MAX_LEN)
    long_description: Optional[str] = None
    models: Optional[Dict[str, str]] = None
    agent_configs: Optional[Dict[str, RunAgentConfigOverride]] = None
    plugins: Optional[Dict[str, Dict[str, Any]]] = None
    is_default: Optional[bool] = None
    approval_mode: Optional[Literal["human", "auto"]] = None
    metadata: Optional[Dict[str, Any]] = None


class RunConfigurationResponse(BaseModel):
    """Response body for a single run configuration."""
    id: str = Field(..., alias="_id")
    tenant_id: Optional[str] = None
    name: str
    description: str = ""
    short_description: str = ""
    long_description: Optional[str] = None
    models: Dict[str, str] = Field(default_factory=dict)
    agent_configs: Optional[Dict[str, RunAgentConfigOverride]] = None
    plugins: Optional[Dict[str, Dict[str, Any]]] = None
    is_default: bool = False
    approval_mode: Optional[Literal["human", "auto"]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def coerce_descriptions(cls, data):
        return _coerce_entity_descriptions(data)


class A2AServerConfigurationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Unique name within tenant")
    endpoint_url: HttpUrl = Field(..., description="Base URL of A2A agent")
    agent_card_url: HttpUrl | None = Field(None, description="Override URL for agent card (default: {endpoint_url}/.well-known/agent-card.json)")
    auth: A2AAuthConfig = Field(default_factory=A2AAuthConfig)
    rpc_endpoint: str = Field(
        default="/a2a",
        description="RPC endpoint path (default: /a2a)"
    )
    request_timeout_seconds: int = Field(default=60, ge=1, le=300)
    enabled: bool = Field(default=True)
    long_running: bool = Field(
        default=False,
        description="AppFactory-280: submit once (message/send, return_immediately=true) then "
                     "poll tasks/get on an interval, instead of holding one message/send call "
                     "open for the task's whole lifetime. Needed for multi-hour external MAS "
                     "runs and required for surviving a backend restart mid-task."
    )
    poll_interval_seconds: int = Field(
        default=15, ge=1, le=3600,
        description="Only used when long_running=true — interval between tasks/get polls."
    )
    use_a2a_streaming: bool | None = Field(
        default=None,
        description="Open message/stream (SSE) for this server instead of "
                    "message/send. Omit to derive from the agent card's "
                    "capabilities.streaming (forced off when long_running=true); set it "
                    "explicitly to override the card. Mutually exclusive with long_running.",
    )
    checkpoints: A2ACheckpointConfig = Field(default_factory=A2ACheckpointConfig)

    @field_validator('agent_card_url')
    @classmethod
    def validate_agent_card_url(cls, v: HttpUrl | None, info: ValidationInfo) -> HttpUrl | None:
        if v and 'endpoint_url' in info.data:
            endpoint_url = info.data['endpoint_url']
            if hasattr(endpoint_url, 'host') and hasattr(v, 'host'):
                if v.host != endpoint_url.host:
                    logging.warning(f"Agent card URL domain {v.host} differs from endpoint domain {endpoint_url.host}")
        return v


class A2AServerConfigurationUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    endpoint_url: HttpUrl | None = None
    agent_card_url: HttpUrl | None = None
    rpc_endpoint: str | None = Field(None, description="RPC endpoint path")
    auth: A2AAuthConfig | None = None
    request_timeout_seconds: int | None = Field(None, ge=1, le=300)
    enabled: bool | None = None
    long_running: bool | None = None
    poll_interval_seconds: int | None = Field(None, ge=1, le=3600)
    use_a2a_streaming: bool | None = None
    checkpoints: A2ACheckpointConfig | None = None

    @field_validator("checkpoints", mode="before")
    @classmethod
    def reject_null_checkpoints(cls, value):
        if value is None:
            raise ValueError(
                "checkpoints cannot be null; omit it for no change or set enabled=false"
            )
        return value

    @field_validator("enabled", "long_running", "use_a2a_streaming", mode="before")
    @classmethod
    def reject_null_bool_flags(cls, value, info):
        # An explicit null is $set into Mongo, then fails the response schema, which
        # types these as a required bool (500 on the next read/list). Absent is the
        # only "leave unchanged"; a null is a client error, not a no-op.
        if value is None:
            raise ValueError(
                f"{info.field_name} cannot be null; omit it to leave the setting unchanged"
            )
        return value


class A2AServerConfigurationResponse(BaseModel):
    model_config = {"populate_by_name": True}

    id: str = Field(..., alias="_id")
    tenant_id: str
    name: str
    endpoint_url: str
    rpc_endpoint: str = "/a2a"
    agent_card_url: str | None = None
    auth: A2AAuthConfig
    request_timeout_seconds: int
    enabled: bool
    long_running: bool = False
    poll_interval_seconds: int = 15
    use_a2a_streaming: bool = False
    checkpoints: A2ACheckpointConfig = Field(default_factory=A2ACheckpointConfig)
    cached_agent_card_summary: Dict[str, Any] | None = Field(None, description="Summary of agent card (skills, name, version)")
    cached_at: datetime | None = None
    last_validated_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class A2ASkillInfo(BaseModel):
    id: str
    name: str
    tags: List[str] | None = Field(default_factory=list)


class A2AAgentCardSummary(BaseModel):
    name: str
    version: str
    url: str
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[A2ASkillInfo]
    capabilities: Dict[str, Any]


class A2AValidateResponse(BaseModel):
    id: str
    name: str
    validated_at: datetime
    skills_count: int
    default_output_modes: List[str]
    agent_card_summary: A2AAgentCardSummary


class A2AValidationError(BaseModel):
    detail: str
    error_type: str
    missing_env_var: str | None = None


class A2APreviewRequest(BaseModel):
    """Discover an agent card without persisting a server (the Add-form Discover button)."""
    endpoint_url: HttpUrl = Field(..., description="Base/origin URL of the A2A agent")
    agent_card_url: HttpUrl | None = Field(None, description="Override URL for the agent card")
    auth: A2AAuthConfig = Field(default_factory=A2AAuthConfig)
    request_timeout_seconds: int = Field(default=30, ge=1, le=300)


class A2APreviewSuggestion(BaseModel):
    """Field values derived from a discovered card, offered to the form as auto-fill."""
    name: str | None = None
    endpoint_url: str | None = None
    rpc_endpoint: str | None = None
    agent_card_url: str | None = None


class A2APreviewResponse(BaseModel):
    agent_card_summary: A2AAgentCardSummary
    suggested: A2APreviewSuggestion
    discovered_card_url: str
    protocol_version: str | None = None
    preferred_transport: str | None = None
