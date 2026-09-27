"""
Result Schema Definition

Defines the format for agent execution results.
If you need to change field names or structure, change it HERE only.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status - centralized enum"""
    PENDING = "pending"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class ResultSchema:
    """
    Schema for agent execution results.
    
    DO NOT access result fields directly in business logic!
    Use ResultSchema.get_*() methods instead.
    """
    
    # Result field names
    STATUS = "status"
    OUTPUT = "output"
    ARTIFACTS = "artifacts"
    REASONING = "reasoning"
    ATTEMPTS = "attempts"
    # terminal-contract envelope; domain payload stays in OUTPUT
    GENERATION_OUTCOME = "generation_outcome"
    TASK_FAILURE = "task_failure"
    
    @staticmethod
    def create(
        status: TaskStatus,
        output: Any = None,
        artifacts: List[Dict] = None,
        reasoning: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Create result object with correct format."""
        result = {
            ResultSchema.STATUS: status,
            ResultSchema.OUTPUT: output,
            ResultSchema.ARTIFACTS: artifacts or [],
            ResultSchema.REASONING: reasoning,
        }
        result.update(kwargs)
        return result
    
    @staticmethod
    def get_status(result: Dict[str, Any]) -> str:
        """Get status from result."""
        return result.get(ResultSchema.STATUS, TaskStatus.PENDING)
    
    @staticmethod
    def get_output(result: Dict[str, Any]) -> Any:
        """Get output from result."""
        return result.get(ResultSchema.OUTPUT)
    
    @staticmethod
    def get_artifacts(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get artifacts from result."""
        return result.get(ResultSchema.ARTIFACTS, [])
    
    @staticmethod
    def get_reasoning(result: Dict[str, Any]) -> str:
        """Get reasoning from result."""
        return result.get(ResultSchema.REASONING, "")
    
    @staticmethod
    def is_completed(result: Dict[str, Any]) -> bool:
        """Check if result indicates completion."""
        return ResultSchema.get_status(result) == TaskStatus.COMPLETED
    
    @staticmethod
    def is_failed(result: Dict[str, Any]) -> bool:
        """Check if result indicates failure."""
        return ResultSchema.get_status(result) == TaskStatus.FAILED
