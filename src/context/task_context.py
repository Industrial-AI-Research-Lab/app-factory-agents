"""
Task Context System

Provides task-specific context for agents working on subtasks.
Helps agents understand:
- What's the parent task?
- What are sibling tasks?
- What input/output is expected?
- How does this integrate with other subtasks?
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class TaskContext:
    """
    Context specific to a task/subtask.
    
    Particularly useful for large tasks with parallel subtasks
    where agents need to understand integration points.
    """
    
    def __init__(
        self,
        task_id: str,
        parent_task_id: Optional[str] = None,
        shared_context = None
    ):
        """
        Args:
            task_id: Unique task identifier
            parent_task_id: Parent task (if this is a subtask)
            shared_context: Reference to project's SharedContext
        """
        self.task_id = task_id
        self.parent_task_id = parent_task_id
        self.shared_context = shared_context
        
        self._data: Dict[str, Any] = {
            "task_id": task_id,
            "parent_task_id": parent_task_id,
            "created_at": datetime.utcnow().isoformat(),
            "description": None,
            "input_spec": {},
            "output_spec": {},
            "dependencies": [],
            "sibling_tasks": [],
            "status": "pending"
        }
    
    def set_description(self, description: str):
        """Set task description"""
        self._data["description"] = description
    
    def set_input_spec(self, spec: Dict):
        """
        Define expected inputs for this task.
        
        Args:
            spec: Input specification
                {
                    "type": "API_response",
                    "format": "JSON",
                    "fields": ["user_id", "message"],
                    "source": "task_123"  # Which task provides this
                }
        """
        self._data["input_spec"] = spec
    
    def set_output_spec(self, spec: Dict):
        """
        Define expected outputs from this task.
        
        Args:
            spec: Output specification
                {
                    "type": "Python_function",
                    "format": "function",
                    "signature": "send_message(user_id: str, text: str)",
                    "consumers": ["task_125", "task_126"]  # Which tasks need this
                }
        """
        self._data["output_spec"] = spec
    
    def add_dependency(self, task_id: str, reason: str):
        """
        Declare dependency on another task.
        
        Args:
            task_id: Task this depends on
            reason: Why this dependency exists
        """
        self._data["dependencies"].append({
            "task_id": task_id,
            "reason": reason,
            "added_at": datetime.utcnow().isoformat()
        })
    
    def set_sibling_tasks(self, sibling_ids: List[str]):
        """Set IDs of parallel subtasks under same parent"""
        self._data["sibling_tasks"] = sibling_ids
    
    def get_parent_task(self) -> Optional[Dict]:
        """
        Get parent task information.
        
        Returns:
            Parent task data or None if this is top-level
        """
        if not self.parent_task_id or not self.shared_context:
            return None
        
        # Look up parent in shared context
        plan = self.shared_context.get("plan", {})
        tasks = plan.get("tasks", [])
        
        for task in tasks:
            if task.get("task_id") == self.parent_task_id:
                return task
        
        return None
    
    def get_sibling_tasks(self) -> List[Dict]:
        """
        Get information about sibling tasks.
        
        Returns:
            List of sibling task data
        """
        if not self.shared_context:
            return []
        
        plan = self.shared_context.get("plan", {})
        tasks = plan.get("tasks", [])
        
        return [
            task for task in tasks
            if task.get("task_id") in self._data["sibling_tasks"]
        ]
    
    def get_input_spec(self) -> Dict:
        """Get expected input specification"""
        return self._data["input_spec"]
    
    def get_output_spec(self) -> Dict:
        """Get expected output specification"""
        return self._data["output_spec"]
    
    def get_dependencies(self) -> List[Dict]:
        """Get list of task dependencies"""
        return self._data["dependencies"]
    
    def update_status(self, status: str):
        """Update task status"""
        self._data["status"] = status
        self._data["updated_at"] = datetime.utcnow().isoformat()
    
    def add_note(self, note: str, agent_id: Optional[str] = None):
        """
        Add note/comment to task context.
        
        Useful for agents to communicate about the task.
        """
        if "notes" not in self._data:
            self._data["notes"] = []
        
        self._data["notes"].append({
            "content": note,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_integration_context(self) -> Dict:
        """
        Get full integration context for this task.
        
        Returns consolidated view of:
        - Parent task goal
        - Sibling tasks (what they're doing)
        - Input expectations
        - Output requirements
        - Dependencies
        """
        parent = self.get_parent_task()
        siblings = self.get_sibling_tasks()
        
        return {
            "task_id": self.task_id,
            "parent_task": parent.get("description") if parent else "Root task",
            "sibling_tasks": [
                {
                    "id": s.get("task_id"),
                    "description": s.get("description"),
                    "status": s.get("status")
                }
                for s in siblings
            ],
            "input_spec": self.get_input_spec(),
            "output_spec": self.get_output_spec(),
            "dependencies": self.get_dependencies(),
            "notes": self._data.get("notes", [])
        }
    
    def to_dict(self) -> Dict:
        """Export as dictionary"""
        return self._data.copy()
    
    def __repr__(self) -> str:
        return f"<TaskContext(id={self.task_id}, parent={self.parent_task_id})>"
