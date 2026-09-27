"""
Task Schema Definition

Defines the format for tasks passed to agents.
If you need to change field names or structure, change it HERE only.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TaskSchema:
    """
    Schema for task objects passed to agents.
    
    DO NOT access task fields directly in business logic!
    Use TaskSchema.get_*() methods instead.
    """
    
    # Field name definitions - change these to change the format
    TASK_ID = "task_id"
    PROJECT_ID = "project_id"
    TYPE = "type"
    DESCRIPTION = "description"
    IS_FIRST_TASK = "is_first_task"
    USER_PROMPT = "user_prompt"
    CONTEXT = "context"
    QUESTIONS = "questions"
    ANALYSIS = "analysis"
    READS = "reads"
    WRITES = "writes"
    RETRY_FEEDBACK = "retry_feedback"
    SUPPRESS_ASSISTANT_MESSAGE = "suppress_assistant_message"
    TRACE_ATTEMPT = "_trace_attempt"
    WORKFLOW_NODE_ID = "workflow_node_id"
    WORKFLOW_NODE_LABEL = "workflow_node_label"
    NEXT_STEP = "next_step"
    WORKFLOW_STAGE_FIELDS = (WORKFLOW_NODE_ID, WORKFLOW_NODE_LABEL, NEXT_STEP)
    
    @staticmethod
    def create(
        task_id: str,
        project_id: str,
        task_type: str,
        description: str,
        is_first_task: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a task with the correct format.
        
        Args:
            task_id: Unique task identifier
            project_id: Project this task belongs to
            task_type: Type of task (requirements_gathering, planning, coding, etc.)
            description: Human-readable task description
            is_first_task: True if this is the first task in the project
            **kwargs: Additional fields (user_prompt, context, etc.)
        
        Returns:
            Task dict with correct format
        """
        task = {
            TaskSchema.TASK_ID: task_id,
            TaskSchema.PROJECT_ID: project_id,
            TaskSchema.TYPE: task_type,
            TaskSchema.DESCRIPTION: description,
            TaskSchema.IS_FIRST_TASK: is_first_task,
        }
        
        # Add optional fields
        task.update(kwargs)
        
        return task
    
    @staticmethod
    def get_id(task: Dict[str, Any]) -> str:
        """Get task ID from task dict."""
        return task.get(TaskSchema.TASK_ID, "")
    
    @staticmethod
    def get_project_id(task: Dict[str, Any]) -> str:
        """Get project ID from task dict."""
        return task.get(TaskSchema.PROJECT_ID, "")
    
    @staticmethod
    def get_type(task: Dict[str, Any]) -> str:
        """Get task type from task dict."""
        return task.get(TaskSchema.TYPE, "")
    
    @staticmethod
    def get_description(task: Dict[str, Any]) -> str:
        """Get task description from task dict."""
        return task.get(TaskSchema.DESCRIPTION, "")
    
    @staticmethod
    def is_first_task(task: Dict[str, Any]) -> bool:
        """Check if this is the first task in the project."""
        return task.get(TaskSchema.IS_FIRST_TASK, False)
    
    @staticmethod
    def get_user_prompt(task: Dict[str, Any]) -> Optional[str]:
        """Get user prompt if present in task."""
        return task.get(TaskSchema.USER_PROMPT)
    
    @staticmethod
    def get_context(task: Dict[str, Any]) -> Dict[str, Any]:
        """Get task context."""
        return task.get(TaskSchema.CONTEXT, {})
    
    @staticmethod
    def get_questions(task: Dict[str, Any]) -> list:
        """Get questions if present (for HumanExpert tasks)."""
        return task.get(TaskSchema.QUESTIONS, [])
    
    @staticmethod
    def get_analysis(task: Dict[str, Any]) -> Dict[str, Any]:
        """Get analysis if present (for HumanExpert tasks)."""
        return task.get(TaskSchema.ANALYSIS, {})

    @staticmethod
    def has_reads(task: Dict[str, Any]) -> bool:
        """Return True when a task explicitly declares a reads contract."""
        return TaskSchema.READS in task and task.get(TaskSchema.READS) is not None

    @staticmethod
    def get_reads(task: Dict[str, Any]) -> Optional[list]:
        """Get the explicit reads contract for the task, if present."""
        if not TaskSchema.has_reads(task):
            return None
        return task.get(TaskSchema.READS) or []

    @staticmethod
    def has_writes(task: Dict[str, Any]) -> bool:
        """Return True when a task explicitly declares a writes contract."""
        return TaskSchema.WRITES in task and task.get(TaskSchema.WRITES) is not None

    @staticmethod
    def get_writes(task: Dict[str, Any]) -> Optional[list]:
        """Get the explicit writes contract for the task, if present."""
        if not TaskSchema.has_writes(task):
            return None
        return task.get(TaskSchema.WRITES) or []

    @staticmethod
    def get_retry_feedback(task: Dict[str, Any]) -> Optional[str]:
        """Get workflow/runtime retry feedback, if present."""
        return task.get(TaskSchema.RETRY_FEEDBACK)

    @staticmethod
    def get_workflow_stage(task: Dict[str, Any]) -> Dict[str, Any]:
        """Stage fields present on the task; an absent field was never recorded."""
        return {
            key: task[key] for key in TaskSchema.WORKFLOW_STAGE_FIELDS if key in task
        }

    @staticmethod
    def should_suppress_assistant_message(task: Dict[str, Any]) -> bool:
        """Return whether the result is rendered by a separate persistent card."""
        return task.get(TaskSchema.SUPPRESS_ASSISTANT_MESSAGE) is True
