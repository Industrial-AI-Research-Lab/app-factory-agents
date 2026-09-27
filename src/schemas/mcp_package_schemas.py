"""API models for MCP ZIP upload, Docker build, and built-image gallery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PreflightIssueResponse(BaseModel):
    code: str = ""
    severity: str = "warning"
    message: str = ""
    path: str = ""


class DockerfileCandidateResponse(BaseModel):
    relative_path: str
    context_dir: str
    score: int = 0
    expose_ports: List[int] = Field(default_factory=list)
    suggested_mode: str = "http"
    suggested_container_port: int = 8080
    suggested_path: str = "/mcp"
    hints: List[str] = Field(default_factory=list)
    preflight_status: str = "ready"
    preflight_issues: List[PreflightIssueResponse] = Field(default_factory=list)


class McpPackageAutofill(BaseModel):
    """Post-analyze hints only; mode/port/path must be set by the operator in UI."""

    server_id: str
    image: Optional[str] = None
    container_port: Optional[int] = None
    path: Optional[str] = None
    mode: Optional[str] = None
    docker_cmd_args: Optional[List[str]] = None
    runtime_scope: str = "tenant"


class McpPackageUploadResponse(BaseModel):
    upload_id: str
    server_id: str
    status: str = "uploaded"


class McpPackageAnalyzeResponse(BaseModel):
    upload_id: str
    server_id: str
    status: str = "analyzed"
    candidates: List[DockerfileCandidateResponse] = Field(default_factory=list)
    autofill: McpPackageAutofill
    archive_warnings: List[PreflightIssueResponse] = Field(default_factory=list)


class McpPackageBuildRequest(BaseModel):
    server_id: str
    dockerfile_relative_path: str
    mode: str = Field(..., description="stdio | streamable-http (required)")
    container_port: Optional[int] = Field(
        None,
        ge=1,
        le=65535,
        description="Required when mode is streamable-http",
    )
    endpoint_path: str = Field("/mcp", description="HTTP MCP path inside container")
    force_rebuild: bool = Field(
        False,
        description="Set true to run build+smoke again when package status is already ready",
    )


class McpPackageBuildResponse(BaseModel):
    upload_id: str
    job_id: str
    image_tag: str
    status: str = "building"


class McpPackageBuildStatusResponse(BaseModel):
    upload_id: str
    job_id: str
    status: str
    phase: str = ""
    log_tail: str = ""
    image_tag: Optional[str] = None
    discover_tool_count: Optional[int] = None
    container_port: Optional[int] = None
    host_port: Optional[int] = None
    error: Optional[str] = None


class McpBuiltImageItem(BaseModel):
    id: str = Field(..., description="Mongo _id of __image__ doc")
    image_tag: str
    server_id: str
    tenant_id: str
    upload_id: Optional[str] = None
    status: str = ""
    created_at: Optional[str] = None
    in_use: bool = False
    discover_tool_count: Optional[int] = None


class McpBuiltImageListResponse(BaseModel):
    images: List[McpBuiltImageItem] = Field(default_factory=list)


class McpPackageMetadata(BaseModel):
    """Stored under tool doc ``metadata.mcp_package`` (name ``__package__``)."""

    record_type: str = "mcp_package"
    upload_id: str = ""
    status: str = "uploaded"
    zip_filename: str = ""
    extract_path: str = ""
    dockerfile_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    selected_dockerfile: Optional[str] = None
    build_log: str = ""
    job_id: Optional[str] = None
    image_tag: Optional[str] = None
    discover_tool_count: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    created_by: Optional[str] = None
