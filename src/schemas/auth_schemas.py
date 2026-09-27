"""
Pydantic schemas for F5 Auth module.

Covers: login, token response, user CRUD, tenant CRUD.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Auth / Login
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """POST /api/auth/login"""
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Response after successful login or token refresh."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """POST /api/auth/refresh"""
    refresh_token: str


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

ROLE_HIERARCHY = ["viewer", "developer", "tenant_admin", "root"]


def _normalize_optional_tenant_id(value: Optional[str]) -> Optional[str]:
    """Normalize tenant_id when provided and reject empty/whitespace values."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("tenant_id must not be empty")
    return normalized


class UserCreate(BaseModel):
    """POST /api/auth/users — create a new user (admin or root only)."""
    email: str = Field(..., description="Unique email")
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=8, description="Initial password (min 8 chars)")
    role: str = Field("developer", description="viewer | developer | tenant_admin | root")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (optional; inherits creator tenant when omitted)")
    enabled: bool = True

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_optional_tenant_id(value)


class UserUpdate(BaseModel):
    """PUT /api/auth/users/{id} — partial update."""
    name: Optional[str] = None
    role: Optional[str] = None
    tenant_id: Optional[str] = None
    enabled: Optional[bool] = None

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: Optional[str]) -> Optional[str]:
        normalized = _normalize_optional_tenant_id(value)
        if normalized is None:
            raise ValueError("tenant_id must not be empty")
        return normalized



class UserResponse(BaseModel):
    """Response body for a single user (password_hash never exposed)."""
    id: str = Field(..., validation_alias="_id")
    email: str
    name: str
    role: str
    tenant_id: Optional[str] = None
    enabled: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[str] = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def coerce_objectid(cls, data):
        """Convert legacy BSON ObjectId user IDs to strings."""
        if isinstance(data, Mapping):
            data = dict(data)
            for key in ("_id", "id"):
                if key in data and isinstance(data[key], ObjectId):
                    data[key] = str(data[key])
        return data


class PasswordResetRequest(BaseModel):
    """POST /api/auth/users/{id}/reset-password"""
    new_password: str = Field(..., min_length=8)


class MeResponse(BaseModel):
    """GET /api/auth/me — current user info from JWT."""
    user_id: str
    email: str
    name: str
    role: str
    tenant_id: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    email: str = Field(description="User email")


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=32, max_length=36, description="Reset token from email")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")


# ---------------------------------------------------------------------------
# MCP cursor-json-http export keys (tenant_settings.mcp_export_api_keys[])
# ---------------------------------------------------------------------------

class McpExportApiKeyCreate(BaseModel):
    """POST /api/tenants/{tenant_id}/mcp-export-keys"""
    name: str = Field(default="mcp-export", min_length=1, max_length=128)


class McpExportApiKeyCreatedResponse(BaseModel):
    """Plaintext api_key returned once on create."""
    id: str
    name: str
    api_key: str
    tenant_id: str
    scopes: List[str] = []
    created_at: Optional[str] = None


class McpExportApiKeyListItem(BaseModel):
    id: str
    name: str
    scopes: List[str] = []
    enabled: bool = True
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    last_used_at: Optional[str] = None
