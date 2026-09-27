"""Full execution-trace projection entry point.

The semantic projector emits compact semantic nodes and technical correlation
layers in one deterministic graph. This module provides a single entry point
so future source adapters can reuse the same graph assembly without
duplicating projection logic.
"""
from __future__ import annotations

from orchestration.semantic_trace_projector import project_semantic_trace


def project_full_trace(**kwargs):
    """Project all persisted trace layers into one read-only graph."""
    return project_semantic_trace(**kwargs)
