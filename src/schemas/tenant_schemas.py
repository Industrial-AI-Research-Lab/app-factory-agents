"""
Pydantic schemas for tenant management and tenant settings.

Covers: tenant CRUD, tenant_settings CRUD.
"""

from datetime import datetime
import re
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator

_TENANT_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

# Default ceiling (seconds) for external-MCP tool-call timeouts when a tenant has
# not set ``mcp_call_timeout_max_seconds``. This is today's hardwired limit; the
# absence of the setting must behave exactly as before (AppFactory-314).
DEFAULT_MCP_CALL_TIMEOUT_MAX_SECONDS = 300


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------

class TenantCreate(BaseModel):
    """POST /api/tenants — root only."""
    id: str = Field(
        ...,
        description=(
            "Unique tenant identifier: starts with lowercase letter, "
            "2-64 chars, lowercase letters/digits/underscore"
        ),
    )
    name: str = Field(..., min_length=1)
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def validate_tenant_slug(cls, value: str) -> str:
        tid = str(value or "").strip()
        if not _TENANT_SLUG_RE.match(tid):
            raise ValueError(
                "Tenant ID must start with a lowercase letter, be 2-64 characters long, "
                "and contain only lowercase letters, digits, or underscores."
            )
        return tid


class TenantUpdate(BaseModel):
    """PUT /api/tenants/{tenant_id} — partial update."""
    name: Optional[str] = None
    enabled: Optional[bool] = None


class TenantResponse(BaseModel):
    """Response body for a tenant."""
    id: str = Field(..., validation_alias="_id")
    name: str
    enabled: bool = True
    created_at: Optional[datetime] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Tenant Settings (per-tenant LLM keys, billing, limits)
# ---------------------------------------------------------------------------

class TenantSettingsUpdate(BaseModel):
    """PUT /api/tenants/{tenant_id}/settings — tenant_admin or root."""
    llm_provider: Optional[str] = Field(None, description="bifrost | openai")
    bifrost_vk: Optional[str] = Field(None, description="Per-tenant Bifrost virtual key")
    openai_api_key: Optional[str] = Field(None, description="Direct OpenAI API key")
    bifrost_url: Optional[str] = Field(None, description="Override Bifrost URL (or use global)")
    bifrost_provider: Optional[str] = Field(None, description="Provider prefix (e.g. openrouter)")
    fallback_models: Optional[List[str]] = Field(None, description="Per-tenant fallback models")
    default_model: Optional[str] = Field(None, description="Tenant default model")
    plugins: Optional[dict[str, dict]] = Field(
        None,
        description="Per-plugin tenant defaults: plugin name -> settings (tenant level of the agent > run > tenant chain, ADR-0004)",
    )
    max_concurrent_projects: Optional[int] = Field(None, description="Resource limit")
    mcp_call_timeout_max_seconds: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Per-tenant ceiling (seconds) for external MCP tool-call timeouts; "
            "absence means the default of 300s"
        ),
    )
    spill_threshold_bytes: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Per-tenant byte threshold above which tool results and file artifacts "
            "spill to object storage instead of inlining; absence means the "
            "ARCHIVE_SPILL_THRESHOLD_BYTES env value or the built-in default"
        ),
    )

    @field_validator("bifrost_vk", "openai_api_key")
    @classmethod
    def reject_masked_secret_values(cls, value: Optional[str]) -> Optional[str]:
        """Block round-trip writes of masked secrets (e.g. sk-l***CDEF)."""
        if isinstance(value, str) and "***" in value:
            raise ValueError(
                "Looks like a masked value; send the real secret or omit the field"
            )
        return value


class TenantMcpExportKeyInSettings(BaseModel):
    """Export key row in GET settings (key_hash masked)."""
    id: str
    name: str
    key_hash: Optional[str] = Field(None, description="Masked bcrypt hash")
    scopes: List[str] = []
    enabled: bool = True
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    last_used_at: Optional[str] = None


class TenantSettingsResponse(BaseModel):
    """GET /api/tenants/{tenant_id}/settings — sensitive fields masked."""
    id: str = Field(..., validation_alias="_id")
    llm_provider: Optional[str] = None
    bifrost_vk: Optional[str] = Field(None, description="Masked in response")
    openai_api_key: Optional[str] = Field(None, description="Masked in response")
    bifrost_url: Optional[str] = None
    bifrost_provider: Optional[str] = None
    fallback_models: List[str] = []
    default_model: Optional[str] = None
    max_concurrent_projects: Optional[int] = None
    spill_threshold_bytes: Optional[int] = None
    mcp_call_timeout_max_seconds: int = Field(
        DEFAULT_MCP_CALL_TIMEOUT_MAX_SECONDS,
        description=(
            "Effective external-MCP call-timeout ceiling (seconds); "
            "default 300 when the tenant has not set it"
        ),
    )
    plugins: Optional[dict[str, dict]] = None
    mcp_export_api_keys: List[TenantMcpExportKeyInSettings] = Field(
        default_factory=list,
        description="Tenant MCP export keys (key_hash masked)",
    )

    model_config = {"populate_by_name": True}

    @field_validator("mcp_call_timeout_max_seconds", mode="before")
    @classmethod
    def _null_ceiling_means_default(cls, v):
        # A stored/explicit null means "unset" → default, like an absent field. The update
        # schema accepts null and it can be persisted, so without this a persisted null
        # fails int validation and 500s the settings GET/PUT until a corrective write.
        return DEFAULT_MCP_CALL_TIMEOUT_MAX_SECONDS if v is None else v


def resolve_mcp_call_timeout_ceiling(settings: Optional[dict]) -> int:
    """The owner tenant's external-MCP call-timeout ceiling in seconds.

    Reads ``mcp_call_timeout_max_seconds`` from a tenant-settings dict; a missing,
    non-integer, or < 1 value falls back to the default 300 — today's behavior for
    tenants that never set it (AppFactory-314 contracts 1 & 6).
    """
    if isinstance(settings, dict):
        raw = settings.get("mcp_call_timeout_max_seconds")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
        if value is not None and value >= 1:
            return value
    return DEFAULT_MCP_CALL_TIMEOUT_MAX_SECONDS


class TenantEffectiveDefaultModelResponse(BaseModel):
    """GET /api/tenants/{tenant_id}/default-model — resolved default for pre-fill."""

    model_id: str
    source: Literal["tenant_settings", "system_info", "hardcoded_fallback"]
