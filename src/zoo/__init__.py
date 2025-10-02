"""
NeuronMap Analysis Zoo

A collaborative platform for sharing and discovering ML interpretability artifacts
including SAE models, circuits, configurations, and analysis results.
"""

from .artifact_schema import (
    ArtifactSchema,
    ArtifactType,
    LicenseType,
    ModelCompatibility,
    FileInfo,
    AuthorInfo,
    CitationInfo,
    PerformanceMetrics,
    ArtifactDependency,
    ArtifactSearchFilter,
    ArtifactSearchResult,
    create_sae_artifact_template,
    create_circuit_artifact_template,
    create_config_artifact_template
)

from .artifact_manager import (
    ArtifactManager,
    ArtifactValidationError,
    ArtifactStorageError
)

__version__ = "1.0.0"
__all__ = [
    # Schema classes
    "ArtifactSchema",
    "ArtifactType", 
    "LicenseType",
    "ModelCompatibility",
    "FileInfo",
    "AuthorInfo",
    "CitationInfo",
    "PerformanceMetrics",
    "ArtifactDependency",
    "ArtifactSearchFilter",
    "ArtifactSearchResult",
    
    # Template functions
    "create_sae_artifact_template",
    "create_circuit_artifact_template", 
    "create_config_artifact_template",
    
    # Manager classes
    "ArtifactManager",
    "ArtifactValidationError",
    "ArtifactStorageError"
]
