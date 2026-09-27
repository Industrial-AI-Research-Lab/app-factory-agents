"""
Artifact Schema Definition

Defines the format for artifacts (code files, configs, etc.).
If you need to change field names or structure, change it HERE only.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ArtifactSchema:
    """
    Schema for artifact objects.
    
    DO NOT access artifact fields directly in business logic!
    Use ArtifactSchema.get_*() methods instead.
    """
    
    # Artifact field names
    TYPE = "type"
    PATH = "path"
    CONTENT = "content"
    METADATA = "metadata"
    TIMESTAMP = "timestamp"
    
    # Common artifact types
    TYPE_CODE_FILE = "code_file"
    TYPE_CONFIG = "config"
    TYPE_API_ENDPOINT = "api_endpoint"
    TYPE_DOCUMENTATION = "documentation"
    
    @staticmethod
    def create(
        artifact_type: str,
        path: str,
        content: Optional[str] = None,
        metadata: Dict = None,
        timestamp: str = None
    ) -> Dict[str, Any]:
        """Create artifact object with correct format."""
        return {
            ArtifactSchema.TYPE: artifact_type,
            ArtifactSchema.PATH: path,
            ArtifactSchema.CONTENT: content,
            ArtifactSchema.METADATA: metadata or {},
            ArtifactSchema.TIMESTAMP: timestamp,
        }
    
    @staticmethod
    def get_type(artifact: Dict[str, Any]) -> str:
        """Get artifact type."""
        return artifact.get(ArtifactSchema.TYPE, "")
    
    @staticmethod
    def get_path(artifact: Dict[str, Any]) -> str:
        """Get artifact path."""
        return artifact.get(ArtifactSchema.PATH, "")
    
    @staticmethod
    def get_content(artifact: Dict[str, Any]) -> Optional[str]:
        """Get artifact content."""
        return artifact.get(ArtifactSchema.CONTENT)
    
    @staticmethod
    def get_metadata(artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Get artifact metadata."""
        return artifact.get(ArtifactSchema.METADATA, {})
