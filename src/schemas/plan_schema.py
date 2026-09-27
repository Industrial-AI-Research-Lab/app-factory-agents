"""
Plan Schema Definition

Defines the format for project plans (task hierarchies).
If you need to change field names or structure, change it HERE only.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PlanSchema:
    """
    Schema for plan objects.
    
    DO NOT access plan fields directly in business logic!
    Use PlanSchema.get_*() methods instead.
    """
    
    # Top-level plan fields
    MAIN_GOAL = "main_goal"
    TASKS = "tasks"
    UPDATED_AT = "updated_at"
    
    # Task fields
    TASK_ID = "task_id"
    DESCRIPTION = "description"
    TYPE = "type"
    SUBTASKS = "subtasks"
    DEPENDENCIES = "dependencies"
    INPUT_SPEC = "input_spec"
    OUTPUT_SPEC = "output_spec"
    COMPLEXITY = "complexity"
    
    @staticmethod
    def create(
        main_goal: str = "",
        tasks: List[Dict] = None,
        updated_at: str = None
    ) -> Dict[str, Any]:
        """Create plan object with correct format."""
        return {
            PlanSchema.MAIN_GOAL: main_goal,
            PlanSchema.TASKS: tasks or [],
            PlanSchema.UPDATED_AT: updated_at,
        }
    
    @staticmethod
    def get_main_goal(plan: Dict[str, Any]) -> str:
        """Get main goal from plan."""
        return plan.get(PlanSchema.MAIN_GOAL, "")
    
    @staticmethod
    def get_tasks(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all tasks from plan."""
        return plan.get(PlanSchema.TASKS, [])
    
    @staticmethod
    def get_updated_at(plan: Dict[str, Any]) -> Optional[str]:
        """Get last update timestamp."""
        return plan.get(PlanSchema.UPDATED_AT)
    
    # Task accessors
    @staticmethod
    def get_task_id(task: Dict[str, Any]) -> str:
        """Get task ID from task dict."""
        return task.get(PlanSchema.TASK_ID, "")
    
    @staticmethod
    def get_task_description(task: Dict[str, Any]) -> str:
        """Get task description."""
        return task.get(PlanSchema.DESCRIPTION, "")
    
    @staticmethod
    def get_task_type(task: Dict[str, Any]) -> str:
        """Get task type (coding, testing, etc.)."""
        return task.get(PlanSchema.TYPE, "")
    
    @staticmethod
    def get_subtasks(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get subtasks from task."""
        return task.get(PlanSchema.SUBTASKS, [])
    
    @staticmethod
    def get_dependencies(task: Dict[str, Any]) -> List[str]:
        """Get task dependencies (list of task IDs)."""
        return task.get(PlanSchema.DEPENDENCIES, [])
    
    @staticmethod
    def get_complexity(task: Dict[str, Any]) -> int:
        """Get task complexity (1-5)."""
        return task.get(PlanSchema.COMPLEXITY, 3)
    
    @staticmethod
    def create_task(
        task_id: str,
        description: str,
        task_type: str,
        subtasks: List[Dict] = None,
        dependencies: List[str] = None,
        complexity: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a task dict with correct format."""
        task = {
            PlanSchema.TASK_ID: task_id,
            PlanSchema.DESCRIPTION: description,
            PlanSchema.TYPE: task_type,
            PlanSchema.SUBTASKS: subtasks or [],
            PlanSchema.DEPENDENCIES: dependencies or [],
            PlanSchema.COMPLEXITY: complexity,
        }
        task.update(kwargs)
        return task
