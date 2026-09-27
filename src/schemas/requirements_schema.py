"""
Requirements Schema Definition

Defines the format for requirements documents.
If you need to change field names or structure, change it HERE only.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class RequirementsSchema:
    """
    Schema for requirements objects.
    
    DO NOT access requirements fields directly in business logic!
    Use RequirementsSchema.get_*() methods instead.
    """
    
    # Top-level field names
    PROJECT_TYPE = "project_type"
    CLARITY_SCORE = "clarity_score"
    MISSING_INFO = "missing_info"
    TECHNICAL_STACK = "technical_stack"
    QUESTIONS_ANSWERED_BY_AI = "questions_answered_by_ai"
    QUESTIONS_ANSWERED_BY_HUMAN = "questions_answered_by_human"
    QUESTIONS_NEEDING_HUMAN = "questions_needing_human"
    INFERRED_DECISIONS = "inferred_decisions"
    TIMESTAMP = "timestamp"
    
    # Technical stack nested fields
    TECH_LANGUAGES = "languages"
    TECH_FRAMEWORKS = "frameworks"
    TECH_DATABASES = "databases"
    TECH_APIS = "apis"
    TECH_ARCHITECTURE = "architecture"
    TECH_DEPLOYMENT = "deployment_target"
    
    # Question fields
    QUESTION_TEXT = "question"
    QUESTION_ANSWER = "answer"
    QUESTION_CONFIDENCE = "confidence"
    QUESTION_REASONING = "reasoning"
    QUESTION_CATEGORY = "category"
    QUESTION_PRIORITY = "priority"
    
    @staticmethod
    def create(
        project_type: str = "unknown",
        clarity_score: float = 0.5,
        missing_info: List[str] = None,
        technical_stack: Dict = None,
        questions_answered_by_ai: List[Dict] = None,
        questions_answered_by_human: Dict = None,
        questions_needing_human: List[Dict] = None,
        inferred_decisions: Dict = None,
        timestamp: str = None
    ) -> Dict[str, Any]:
        """Create requirements object with correct format."""
        return {
            RequirementsSchema.PROJECT_TYPE: project_type,
            RequirementsSchema.CLARITY_SCORE: clarity_score,
            RequirementsSchema.MISSING_INFO: missing_info or [],
            RequirementsSchema.TECHNICAL_STACK: technical_stack or {},
            RequirementsSchema.QUESTIONS_ANSWERED_BY_AI: questions_answered_by_ai or [],
            RequirementsSchema.QUESTIONS_ANSWERED_BY_HUMAN: questions_answered_by_human or {},
            RequirementsSchema.QUESTIONS_NEEDING_HUMAN: questions_needing_human or [],
            RequirementsSchema.INFERRED_DECISIONS: inferred_decisions or {},
            RequirementsSchema.TIMESTAMP: timestamp,
        }
    
    @staticmethod
    def get_project_type(requirements: Dict[str, Any]) -> str:
        """Get project type."""
        return requirements.get(RequirementsSchema.PROJECT_TYPE, "unknown")
    
    @staticmethod
    def get_clarity_score(requirements: Dict[str, Any]) -> float:
        """Get clarity score (0-1)."""
        return requirements.get(RequirementsSchema.CLARITY_SCORE, 0.5)
    
    @staticmethod
    def get_missing_info(requirements: Dict[str, Any]) -> List[str]:
        """Get list of missing information."""
        return requirements.get(RequirementsSchema.MISSING_INFO, [])
    
    @staticmethod
    def get_technical_stack(requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get technical stack dict."""
        return requirements.get(RequirementsSchema.TECHNICAL_STACK, {})
    
    @staticmethod
    def get_tech_languages(requirements: Dict[str, Any]) -> List[str]:
        """Get programming languages from technical stack."""
        tech_stack = RequirementsSchema.get_technical_stack(requirements)
        return tech_stack.get(RequirementsSchema.TECH_LANGUAGES, [])
    
    @staticmethod
    def get_tech_architecture(requirements: Dict[str, Any]) -> str:
        """Get architecture from technical stack."""
        tech_stack = RequirementsSchema.get_technical_stack(requirements)
        return tech_stack.get(RequirementsSchema.TECH_ARCHITECTURE, "")
    
    @staticmethod
    def get_tech_deployment(requirements: Dict[str, Any]) -> str:
        """Get deployment target from technical stack."""
        tech_stack = RequirementsSchema.get_technical_stack(requirements)
        return tech_stack.get(RequirementsSchema.TECH_DEPLOYMENT, "")
    
    @staticmethod
    def get_ai_questions(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get ALL AI-answered questions with full context.
        
        Returns list of dicts with: question, answer, confidence, reasoning
        """
        return requirements.get(RequirementsSchema.QUESTIONS_ANSWERED_BY_AI, [])
    
    @staticmethod
    def get_human_questions(requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get human-answered questions."""
        return requirements.get(RequirementsSchema.QUESTIONS_ANSWERED_BY_HUMAN, {})
    
    @staticmethod
    def get_questions_needing_human(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get questions that still need human input."""
        return requirements.get(RequirementsSchema.QUESTIONS_NEEDING_HUMAN, [])
    
    @staticmethod
    def get_inferred_decisions(requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get inferred decisions."""
        return requirements.get(RequirementsSchema.INFERRED_DECISIONS, {})
    
    # Question accessors
    @staticmethod
    def get_question_text(question: Dict[str, Any]) -> str:
        """Get question text from question dict."""
        return question.get(RequirementsSchema.QUESTION_TEXT, "")
    
    @staticmethod
    def get_question_answer(question: Dict[str, Any]) -> str:
        """Get answer from question dict."""
        return question.get(RequirementsSchema.QUESTION_ANSWER, "")
    
    @staticmethod
    def get_question_confidence(question: Dict[str, Any]) -> float:
        """Get confidence score from question dict."""
        return question.get(RequirementsSchema.QUESTION_CONFIDENCE, 0.0)
    
    @staticmethod
    def get_question_reasoning(question: Dict[str, Any]) -> str:
        """Get reasoning from question dict."""
        return question.get(RequirementsSchema.QUESTION_REASONING, "")
    
    @staticmethod
    def create_question(
        question: str,
        answer: str = "",
        confidence: float = 0.0,
        reasoning: str = "",
        category: str = "",
        priority: str = ""
    ) -> Dict[str, Any]:
        """Create a question dict with correct format."""
        return {
            RequirementsSchema.QUESTION_TEXT: question,
            RequirementsSchema.QUESTION_ANSWER: answer,
            RequirementsSchema.QUESTION_CONFIDENCE: confidence,
            RequirementsSchema.QUESTION_REASONING: reasoning,
            RequirementsSchema.QUESTION_CATEGORY: category,
            RequirementsSchema.QUESTION_PRIORITY: priority,
        }
