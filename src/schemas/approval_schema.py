"""
Approval Schema Definition

Defines the format for approval gates.
If you need to change field names or structure, change it HERE only.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ApprovalStatus(str, Enum):
    """Approval status - centralized enum"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalType(str, Enum):
    """Approval types"""
    REQUIREMENTS = "requirements"
    PLAN = "plan"
    OUTPUT = "output"
    DEPLOY = "deploy"


@dataclass
class ApprovalSchema:
    """
    Schema for approval objects.
    
    DO NOT access approval fields directly in business logic!
    Use ApprovalSchema.get_*() methods instead.
    
    Milestone 4: Per-instance approvals with UUID ids and run_id scoping.
    Each approval is now a unique instance with its own UUID, scoped to a run.
    """
    
    # Approval field names
    APPROVAL_ID = "approval_id"
    PROJECT_ID = "project_id"
    RUN_ID = "run_id"
    GATE_TYPE = "gate_type"
    DATA = "data"
    STATUS = "status"
    CREATED_AT = "created_at"
    RESOLVED_AT = "resolved_at"
    FEEDBACK = "feedback"
    SUPERSEDES_APPROVAL_ID = "supersedes_approval_id"
    
    # Legacy field names (for backward compatibility during migration)
    TYPE = "type"
    REQUESTED_AT = "requested_at"
    APPROVED_AT = "approved_at"
    REJECTED_AT = "rejected_at"
    REASON = "reason"
    
    @staticmethod
    def create(
        approval_id: str,
        project_id: str,
        approval_type: ApprovalType,
        data: Dict[str, Any],
        status: ApprovalStatus = ApprovalStatus.PENDING,
        run_id: Optional[str] = None,
        created_at: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create approval object with correct format (Milestone 4: per-instance UUID-based)."""
        approval = {
            ApprovalSchema.APPROVAL_ID: approval_id,
            ApprovalSchema.PROJECT_ID: project_id,
            ApprovalSchema.GATE_TYPE: approval_type,
            ApprovalSchema.DATA: data,
            ApprovalSchema.STATUS: status,
            ApprovalSchema.CREATED_AT: created_at,
        }
        if run_id:
            approval[ApprovalSchema.RUN_ID] = run_id
        approval.update(kwargs)
        return approval
    
    @staticmethod
    def get_id(approval: Dict[str, Any]) -> str:
        """Get approval ID."""
        return approval.get(ApprovalSchema.APPROVAL_ID, "")
    
    @staticmethod
    def get_project_id(approval: Dict[str, Any]) -> str:
        """Get project ID."""
        return approval.get(ApprovalSchema.PROJECT_ID, "")
    
    @staticmethod
    def get_type(approval: Dict[str, Any]) -> str:
        """Get approval type (gate_type for Milestone 4, fallback to type for legacy)."""
        return approval.get(ApprovalSchema.GATE_TYPE) or approval.get(ApprovalSchema.TYPE, "")
    
    @staticmethod
    def get_run_id(approval: Dict[str, Any]) -> Optional[str]:
        """Get run ID (Milestone 4: per-instance approvals are scoped to runs)."""
        return approval.get(ApprovalSchema.RUN_ID)
    
    @staticmethod
    def get_data(approval: Dict[str, Any]) -> Dict[str, Any]:
        """Get approval data."""
        return approval.get(ApprovalSchema.DATA, {})
    
    @staticmethod
    def get_status(approval: Dict[str, Any]) -> str:
        """Get approval status."""
        return approval.get(ApprovalSchema.STATUS, ApprovalStatus.PENDING)
    
    @staticmethod
    def is_pending(approval: Dict[str, Any]) -> bool:
        """Check if approval is pending."""
        return ApprovalSchema.get_status(approval) == ApprovalStatus.PENDING
    
    @staticmethod
    def is_approved(approval: Dict[str, Any]) -> bool:
        """Check if approval is approved."""
        return ApprovalSchema.get_status(approval) == ApprovalStatus.APPROVED
    
    @staticmethod
    def is_rejected(approval: Dict[str, Any]) -> bool:
        """Check if approval is rejected."""
        return ApprovalSchema.get_status(approval) == ApprovalStatus.REJECTED
