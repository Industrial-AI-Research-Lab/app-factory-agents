"""Reserved ``name`` values for ZIP/build rows in ``tool_mcp_configurations``."""

from __future__ import annotations

import re

MCP_PACKAGE_NAME = "__package__"
MCP_IMAGE_NAME_PREFIX = "__image__"
MCP_SERVER_INTERNAL_SOURCE = "mcp_server_internal"

def is_mcp_internal_tool_name(name: str | None) -> bool:
    """True only for ZIP wizard/gallery rows (``__package__`` and ``__image__*``)."""
    return is_mcp_package_name(name) or is_mcp_built_image_name(name)


def is_mcp_package_name(name: str | None) -> bool:
    return str(name or "").strip() == MCP_PACKAGE_NAME


def is_mcp_built_image_name(name: str | None) -> bool:
    n = str(name or "").strip()
    return n.startswith(MCP_IMAGE_NAME_PREFIX) and n != MCP_PACKAGE_NAME


def mcp_user_tool_name_filter() -> dict:
    """Mongo match: rows exposed as MCP tools (exclude ZIP internal docs only)."""
    return {
        "$and": [
            {"name": {"$ne": MCP_PACKAGE_NAME}},
            {"name": {"$not": re.compile(f"^{re.escape(MCP_IMAGE_NAME_PREFIX)}")}},
        ]
    }


def mcp_built_image_name_filter() -> dict:
    return {"name": {"$regex": f"^{re.escape(MCP_IMAGE_NAME_PREFIX)}"}}


def mcp_package_name_filter() -> dict:
    return {"name": MCP_PACKAGE_NAME}
