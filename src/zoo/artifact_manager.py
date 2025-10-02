"""
Artifact Manager for NeuronMap Analysis Zoo

This module provides the core functionality for managing artifacts in the
Analysis Zoo, including validation, storage, and retrieval operations.

Based on aufgabenliste_b.md Tasks B1-B4
"""

import json
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, IO
from datetime import datetime
import tempfile
import logging

from .artifact_schema import (
    ArtifactSchema, 
    ArtifactType, 
    FileInfo, 
    ArtifactSearchFilter,
    ArtifactSearchResult
)

logger = logging.getLogger(__name__)


class ArtifactValidationError(Exception):
    """Raised when artifact validation fails."""
    pass


class ArtifactStorageError(Exception):
    """Raised when artifact storage operations fail."""
    pass


class ArtifactManager:
    """
    Core manager for Analysis Zoo artifacts.
    
    Handles validation, packaging, storage, and retrieval of artifacts
    with support for local file storage and future cloud storage backends.
    """
    
    def __init__(self, storage_root: Path, cache_dir: Optional[Path] = None):
        """
        Initialize the artifact manager.
        
        Args:
            storage_root: Root directory for artifact storage
            cache_dir: Directory for caching downloaded artifacts
        """
        self.storage_root = Path(storage_root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.storage_root / "cache"
        
        # Create directories if they don't exist
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Local metadata database (JSON files for now)
        self.metadata_dir = self.storage_root / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
    def validate_artifact(self, artifact_path: Path) -> ArtifactSchema:
        """
        Validate an artifact directory and return its metadata.
        
        Args:
            artifact_path: Path to artifact directory
            
        Returns:
            Validated artifact schema
            
        Raises:
            ArtifactValidationError: If validation fails
        """
        try:
            # Check if artifact.json exists
            manifest_path = artifact_path / "artifact.json"
            if not manifest_path.exists():
                raise ArtifactValidationError("Missing artifact.json manifest")
            
            # Load and validate schema
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            artifact = ArtifactSchema(**manifest_data)
            
            # Validate files exist and match checksums
            for file_info in artifact.files:
                file_path = artifact_path / file_info.path
                
                if not file_path.exists():
                    raise ArtifactValidationError(f"Missing file: {file_info.path}")
                
                # Check file size
                actual_size = file_path.stat().st_size
                if actual_size != file_info.size_bytes:
                    raise ArtifactValidationError(
                        f"File size mismatch for {file_info.path}: "
                        f"expected {file_info.size_bytes}, got {actual_size}"
                    )
                
                # Check checksum
                actual_checksum = self._calculate_file_checksum(file_path)
                if actual_checksum != file_info.checksum_sha256:
                    raise ArtifactValidationError(
                        f"Checksum mismatch for {file_info.path}: "
                        f"expected {file_info.checksum_sha256}, got {actual_checksum}"
                    )
            
            # Validate total size
            total_size = sum(f.size_bytes for f in artifact.files)
            if total_size != artifact.total_size_bytes:
                raise ArtifactValidationError(
                    f"Total size mismatch: expected {artifact.total_size_bytes}, got {total_size}"
                )
            
            logger.info(f"Artifact validation successful: {artifact.name} v{artifact.version}")
            return artifact
            
        except Exception as e:
            if isinstance(e, ArtifactValidationError):
                raise
            raise ArtifactValidationError(f"Validation failed: {str(e)}")
    
    def prepare_artifact(self, source_path: Path, manifest: Optional[ArtifactSchema] = None) -> ArtifactSchema:
        """
        Prepare an artifact directory by generating metadata and validating files.
        
        Args:
            source_path: Path to source files
            manifest: Optional pre-existing manifest to update
            
        Returns:
            Complete artifact schema with file information
        """
        if not source_path.exists():
            raise ArtifactValidationError(f"Source path does not exist: {source_path}")
        
        # Load existing manifest or create new one
        manifest_path = source_path / "artifact.json"
        if manifest is None and manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            manifest = ArtifactSchema(**manifest_data)
        elif manifest is None:
            raise ArtifactValidationError("No manifest provided and artifact.json not found")
        
        # Scan files and generate FileInfo objects
        files = []
        total_size = 0
        
        for file_path in source_path.rglob('*'):
            if file_path.is_file() and file_path.name != 'artifact.json':
                relative_path = file_path.relative_to(source_path)
                
                # Calculate file info
                size = file_path.stat().st_size
                checksum = self._calculate_file_checksum(file_path)
                mime_type = mimetypes.guess_type(str(file_path))[0]
                
                file_info = FileInfo(
                    path=str(relative_path),
                    size_bytes=size,
                    checksum_sha256=checksum,
                    mime_type=mime_type
                )
                
                files.append(file_info)
                total_size += size
        
        # Update manifest with file information
        manifest.files = files
        manifest.total_size_bytes = total_size
        manifest.updated_at = datetime.utcnow()
        
        # Save updated manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest.dict(exclude_none=True), f, indent=2, default=str)
        
        logger.info(f"Prepared artifact: {manifest.name} v{manifest.version} ({total_size} bytes)")
        return manifest
    
    def store_artifact(self, artifact_path: Path, overwrite: bool = False) -> str:
        """
        Store an artifact in the local storage.
        
        Args:
            artifact_path: Path to validated artifact directory
            overwrite: Whether to overwrite existing artifacts
            
        Returns:
            Artifact UUID
            
        Raises:
            ArtifactStorageError: If storage operation fails
        """
        try:
            # Validate artifact first
            artifact = self.validate_artifact(artifact_path)
            
            # Check if artifact already exists
            storage_path = self.storage_root / "artifacts" / artifact.uuid
            if storage_path.exists() and not overwrite:
                raise ArtifactStorageError(f"Artifact {artifact.uuid} already exists")
            
            # Create storage directory
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Copy artifact files
            for file_info in artifact.files:
                source_file = artifact_path / file_info.path
                dest_file = storage_path / file_info.path
                
                # Create parent directories
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_file, dest_file)
            
            # Copy manifest
            shutil.copy2(artifact_path / "artifact.json", storage_path / "artifact.json")
            
            # Store metadata for search
            metadata_file = self.metadata_dir / f"{artifact.uuid}.json"
            with open(metadata_file, 'w') as f:
                json.dump(artifact.dict(exclude_none=True), f, indent=2, default=str)
            
            # Update storage path in artifact
            artifact.storage_path = str(storage_path)
            
            logger.info(f"Stored artifact: {artifact.uuid} at {storage_path}")
            return artifact.uuid
            
        except Exception as e:
            if isinstance(e, (ArtifactValidationError, ArtifactStorageError)):
                raise
            raise ArtifactStorageError(f"Storage failed: {str(e)}")
    
    def get_artifact(self, artifact_id: str) -> Optional[ArtifactSchema]:
        """
        Retrieve artifact metadata by ID.
        
        Args:
            artifact_id: Artifact UUID
            
        Returns:
            Artifact schema or None if not found
        """
        metadata_file = self.metadata_dir / f"{artifact_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            return ArtifactSchema(**data)
        except Exception as e:
            logger.error(f"Failed to load artifact {artifact_id}: {e}")
            return None
    
    def list_artifacts(self, filter_params: Optional[ArtifactSearchFilter] = None) -> ArtifactSearchResult:
        """
        List artifacts with optional filtering.
        
        Args:
            filter_params: Search filters
            
        Returns:
            Search results
        """
        if filter_params is None:
            filter_params = ArtifactSearchFilter()
        
        start_time = datetime.utcnow()
        artifacts = []
        
        # Load all metadata files
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                artifact = ArtifactSchema(**data)
                
                # Apply filters
                if self._matches_filter(artifact, filter_params):
                    artifacts.append(artifact)
                    
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
                continue
        
        # Sort results
        artifacts.sort(
            key=lambda x: getattr(x, filter_params.sort_by),
            reverse=(filter_params.sort_order == "desc")
        )
        
        # Apply pagination
        total_count = len(artifacts)
        start_idx = filter_params.offset
        end_idx = start_idx + filter_params.limit
        page_artifacts = artifacts[start_idx:end_idx]
        
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ArtifactSearchResult(
            artifacts=page_artifacts,
            total_count=total_count,
            has_more=end_idx < total_count,
            search_time_ms=search_time
        )
    
    def download_artifact(self, artifact_id: str, target_path: Optional[Path] = None) -> Path:
        """
        Download an artifact to local cache or specified path.
        
        Args:
            artifact_id: Artifact UUID
            target_path: Optional target path (defaults to cache)
            
        Returns:
            Path to downloaded artifact
            
        Raises:
            ArtifactStorageError: If download fails
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise ArtifactStorageError(f"Artifact {artifact_id} not found")
        
        # Determine target path
        if target_path is None:
            target_path = self.cache_dir / artifact_id
        
        target_path = Path(target_path)
        
        # Check if already cached and valid
        if target_path.exists():
            try:
                cached_artifact = self.validate_artifact(target_path)
                if cached_artifact.uuid == artifact_id:
                    logger.info(f"Using cached artifact: {artifact_id}")
                    return target_path
            except ArtifactValidationError:
                # Invalid cache, remove it
                shutil.rmtree(target_path)
        
        # Copy from storage
        storage_path = self.storage_root / "artifacts" / artifact_id
        if not storage_path.exists():
            raise ArtifactStorageError(f"Artifact storage not found: {storage_path}")
        
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        for file_info in artifact.files:
            source_file = storage_path / file_info.path
            dest_file = target_path / file_info.path
            
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, dest_file)
        
        # Copy manifest
        shutil.copy2(storage_path / "artifact.json", target_path / "artifact.json")
        
        logger.info(f"Downloaded artifact {artifact_id} to {target_path}")
        return target_path
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact from storage.
        
        Args:
            artifact_id: Artifact UUID
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from storage
        storage_path = self.storage_root / "artifacts" / artifact_id
        metadata_file = self.metadata_dir / f"{artifact_id}.json"
        
        deleted = False
        
        if storage_path.exists():
            shutil.rmtree(storage_path)
            deleted = True
        
        if metadata_file.exists():
            metadata_file.unlink()
            deleted = True
        
        if deleted:
            logger.info(f"Deleted artifact: {artifact_id}")
        
        return deleted
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _matches_filter(self, artifact: ArtifactSchema, filter_params: ArtifactSearchFilter) -> bool:
        """Check if artifact matches search filters."""
        # Text search in name and description
        if filter_params.query:
            query_lower = filter_params.query.lower()
            if (query_lower not in artifact.name.lower() and 
                query_lower not in artifact.description.lower() and
                not any(query_lower in tag.lower() for tag in artifact.tags)):
                return False
        
        # Type filter
        if filter_params.artifact_type and artifact.artifact_type != filter_params.artifact_type:
            return False
        
        # Tags filter (OR logic)
        if filter_params.tags:
            if not any(tag in artifact.tags for tag in filter_params.tags):
                return False
        
        # Model compatibility filter
        if filter_params.model_name:
            if not any(
                filter_params.model_name.lower() in compat.model_name.lower()
                for compat in artifact.model_compatibility
            ):
                return False
        
        # License filter
        if filter_params.license and artifact.license != filter_params.license:
            return False
        
        # Author filter
        if filter_params.author:
            author_lower = filter_params.author.lower()
            if not any(
                author_lower in author.name.lower()
                for author in artifact.authors
            ):
                return False
        
        # Rating filter
        if filter_params.min_rating and (not artifact.rating or artifact.rating < filter_params.min_rating):
            return False
        
        # Status filters
        if filter_params.verified_only and artifact.verification_status != "verified":
            return False
        
        if filter_params.official_only and not artifact.is_official:
            return False
        
        if filter_params.featured_only and not artifact.featured:
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the artifact collection."""
        all_artifacts = self.list_artifacts().artifacts
        
        stats = {
            "total_artifacts": len(all_artifacts),
            "by_type": {},
            "by_license": {},
            "total_size_bytes": 0,
            "total_downloads": 0,
            "verified_count": 0,
            "official_count": 0,
            "featured_count": 0
        }
        
        for artifact in all_artifacts:
            # Count by type
            type_name = artifact.artifact_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1
            
            # Count by license
            license_name = artifact.license.value
            stats["by_license"][license_name] = stats["by_license"].get(license_name, 0) + 1
            
            # Aggregate stats
            stats["total_size_bytes"] += artifact.total_size_bytes
            stats["total_downloads"] += artifact.downloads
            
            if artifact.verification_status == "verified":
                stats["verified_count"] += 1
            
            if artifact.is_official:
                stats["official_count"] += 1
            
            if artifact.featured:
                stats["featured_count"] += 1
        
        return stats
