"""
Artifact Metadata Schema for NeuronMap Analysis Zoo

This module defines the schema for sharing and discovering analysis artifacts
including SAE models, circuits, configurations, and analysis results.

Based on aufgabenliste_b.md Task B1: Definition des "Artefakt"-Metadaten-Schemas
"""

from pydantic import BaseModel, Field, HttpUrl, validator, ConfigDict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
import uuid


class ArtifactType(str, Enum):
    """Types of artifacts that can be shared in the Analysis Zoo."""
    SAE_MODEL = "sae_model"
    CIRCUIT = "circuit"
    INTERVENTION_CONFIG = "intervention_config"
    ANALYSIS_RESULT = "analysis_result"
    DATASET = "dataset"
    VISUALIZATION = "visualization"


class LicenseType(str, Enum):
    """Supported licenses for artifacts."""
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    GPL_3_0 = "GPL-3.0"
    PROPRIETARY = "Proprietary"
    CUSTOM = "Custom"


class ModelCompatibility(BaseModel):
    """Model compatibility information."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(..., description="Base model name (e.g., 'gpt2', 'llama-2-7b')")
    model_family: Optional[str] = Field(None, description="Model family (e.g., 'gpt', 'llama')")
    architecture: Optional[str] = Field(None, description="Architecture type (e.g., 'transformer')")
    min_parameters: Optional[int] = Field(None, description="Minimum model parameters")
    max_parameters: Optional[int] = Field(None, description="Maximum model parameters")
    layers: Optional[List[int]] = Field(None, description="Compatible layer indices")


class FileInfo(BaseModel):
    """Information about files in the artifact."""
    path: str = Field(..., description="Relative path within the artifact")
    size_bytes: int = Field(..., description="File size in bytes")
    checksum_sha256: str = Field(..., description="SHA256 checksum")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    description: Optional[str] = Field(None, description="Human-readable description")


class AuthorInfo(BaseModel):
    """Author/contributor information."""
    name: str = Field(..., description="Author name")
    email: Optional[str] = Field(None, description="Contact email")
    affiliation: Optional[str] = Field(None, description="Organization/institution")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    github: Optional[str] = Field(None, description="GitHub username")


class CitationInfo(BaseModel):
    """Citation information for the artifact."""
    title: str = Field(..., description="Title for citation")
    authors: List[AuthorInfo] = Field(..., description="List of authors")
    year: int = Field(..., description="Publication/creation year")
    doi: Optional[str] = Field(None, description="DOI if published")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier")
    bibtex: Optional[str] = Field(None, description="BibTeX citation")


class PerformanceMetrics(BaseModel):
    """Performance metrics for the artifact."""
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Accuracy metric")
    loss: Optional[float] = Field(None, ge=0, description="Loss value")
    sparsity: Optional[float] = Field(None, ge=0, le=1, description="Sparsity for SAE models")
    reconstruction_loss: Optional[float] = Field(None, ge=0, description="Reconstruction loss")
    custom_metrics: Optional[Dict[str, float]] = Field(None, description="Custom metrics")


class ArtifactDependency(BaseModel):
    """Dependency on another artifact."""
    artifact_id: str = Field(..., description="UUID of the dependent artifact")
    version: Optional[str] = Field(None, description="Required version")
    optional: bool = Field(False, description="Whether dependency is optional")
    purpose: Optional[str] = Field(None, description="Why this dependency is needed")


class ArtifactSchema(BaseModel):
    """
    Complete metadata schema for Analysis Zoo artifacts.
    
    This schema defines all metadata required for sharing artifacts in the
    NeuronMap Analysis Zoo, enabling discovery, compatibility checking,
    and proper attribution.
    """
    model_config = ConfigDict(protected_namespaces=())
    
    # Core identification
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+(-\w+)?$", description="Semantic version")
    
    # Classification
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    tags: List[str] = Field(default=[], description="Searchable tags")
    category: Optional[str] = Field(None, description="Category within type")
    
    # Description and documentation
    description: str = Field(..., min_length=10, max_length=500, description="Brief description")
    long_description: Optional[str] = Field(None, max_length=5000, description="Detailed description")
    readme_path: Optional[str] = Field(None, description="Path to README file")
    documentation_url: Optional[HttpUrl] = Field(None, description="External documentation URL")
    
    # Author and attribution
    authors: List[AuthorInfo] = Field(..., min_items=1, description="Artifact authors")
    citation: Optional[CitationInfo] = Field(None, description="Citation information")
    license: LicenseType = Field(..., description="License type")
    license_text: Optional[str] = Field(None, description="Custom license text")
    
    # Technical details
    model_compatibility: List[ModelCompatibility] = Field(..., description="Compatible models")
    dependencies: List[str] = Field(default=[], description="Python package dependencies")
    artifact_dependencies: List[ArtifactDependency] = Field(default=[], description="Other artifact dependencies")
    python_version: Optional[str] = Field(None, description="Required Python version")
    frameworks: List[str] = Field(default=[], description="Required frameworks (torch, transformers, etc.)")
    
    # Files and storage
    files: List[FileInfo] = Field(default=[], description="Files in the artifact")
    total_size_bytes: int = Field(default=0, ge=0, description="Total size of all files")
    storage_path: Optional[str] = Field(None, description="Storage backend path")
    
    # Performance and quality
    metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Quality assessment score")
    verification_status: Optional[Literal["pending", "verified", "failed"]] = Field(None)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    downloads: int = Field(default=0, ge=0, description="Download count")
    stars: int = Field(default=0, ge=0, description="Star rating count")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Average user rating")
    
    # Publishing info
    is_public: bool = Field(True, description="Whether artifact is publicly accessible")
    is_official: bool = Field(False, description="Whether this is an official NeuronMap artifact")
    featured: bool = Field(False, description="Whether to feature this artifact")
    
    # Ownership and stars
    owner_id: Optional[str] = Field(None, description="User ID of the artifact owner")
    starred_by: List[str] = Field(default_factory=list, description="List of user IDs who starred this artifact")
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format."""
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        for tag in v:
            if not tag.replace('-', '').replace('_', '').isalnum():
                raise ValueError(f"Invalid tag format: {tag}")
        return v
    
    @validator('total_size_bytes')
    def validate_size(cls, v):
        """Validate artifact size limits."""
        max_size = 5 * 1024 * 1024 * 1024  # 5GB
        if v > max_size:
            raise ValueError(f"Artifact size {v} exceeds maximum of {max_size} bytes")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.dict(exclude_none=True)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Get JSON schema for validation."""
        return self.schema()


class ArtifactSearchFilter(BaseModel):
    """Search filters for artifact discovery."""
    model_config = ConfigDict(protected_namespaces=())
    
    query: Optional[str] = Field(None, description="Text search query")
    artifact_type: Optional[ArtifactType] = Field(None, description="Filter by type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (OR)")
    model_name: Optional[str] = Field(None, description="Filter by compatible model")
    license: Optional[LicenseType] = Field(None, description="Filter by license")
    author: Optional[str] = Field(None, description="Filter by author name")
    min_rating: Optional[float] = Field(None, ge=1, le=5, description="Minimum rating")
    verified_only: bool = Field(False, description="Only verified artifacts")
    official_only: bool = Field(False, description="Only official artifacts")
    featured_only: bool = Field(False, description="Only featured artifacts")
    sort_by: Literal["created_at", "updated_at", "downloads", "rating", "stars"] = Field("created_at")
    sort_order: Literal["asc", "desc"] = Field("desc")
    limit: int = Field(50, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Results offset")


class ArtifactSearchResult(BaseModel):
    """Search result for artifact discovery."""
    artifacts: List[ArtifactSchema] = Field(..., description="Found artifacts")
    total_count: int = Field(..., description="Total matching artifacts")
    has_more: bool = Field(..., description="Whether more results available")
    search_time_ms: float = Field(..., description="Search time in milliseconds")


# Template generators for common artifact types
def create_sae_artifact_template(
    name: str,
    model_name: str,
    layer: int,
    dict_size: int,
    authors: List[AuthorInfo]
) -> ArtifactSchema:
    """Create a template for SAE model artifacts."""
    return ArtifactSchema(
        name=name,
        version="1.0.0",
        artifact_type=ArtifactType.SAE_MODEL,
        description=f"Sparse Autoencoder trained on {model_name} layer {layer}",
        tags=["sae", "sparse-autoencoder", model_name.lower(), f"layer-{layer}"],
        authors=authors,
        license=LicenseType.MIT,
        model_compatibility=[
            ModelCompatibility(
                model_name=model_name,
                layers=[layer]
            )
        ],
        dependencies=["torch", "transformers", "numpy"],
        files=[],  # To be filled with actual files
        total_size_bytes=0,  # To be calculated
        metrics=PerformanceMetrics(
            sparsity=0.0,  # To be filled
            reconstruction_loss=0.0  # To be filled
        )
    )


def create_circuit_artifact_template(
    name: str,
    model_name: str,
    circuit_type: str,
    authors: List[AuthorInfo]
) -> ArtifactSchema:
    """Create a template for circuit artifacts."""
    return ArtifactSchema(
        name=name,
        version="1.0.0",
        artifact_type=ArtifactType.CIRCUIT,
        description=f"{circuit_type} circuit discovered in {model_name}",
        tags=["circuit", circuit_type.lower(), model_name.lower()],
        authors=authors,
        license=LicenseType.CC_BY_4_0,
        model_compatibility=[
            ModelCompatibility(
                model_name=model_name
            )
        ],
        dependencies=["networkx", "matplotlib"],
        files=[],  # To be filled
        total_size_bytes=0  # To be calculated
    )


def create_config_artifact_template(
    name: str,
    config_type: str,
    authors: List[AuthorInfo]
) -> ArtifactSchema:
    """Create a template for configuration artifacts."""
    return ArtifactSchema(
        name=name,
        version="1.0.0",
        artifact_type=ArtifactType.INTERVENTION_CONFIG,
        description=f"Configuration for {config_type} experiments",
        tags=["config", config_type.lower(), "experiment"],
        authors=authors,
        license=LicenseType.MIT,
        model_compatibility=[],  # To be filled based on config
        dependencies=["pydantic", "yaml"],
        files=[],  # To be filled
        total_size_bytes=0
    )
