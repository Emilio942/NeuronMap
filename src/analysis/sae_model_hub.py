"""
SAE Model Hub and Management System

This module provides a comprehensive system for managing trained SAE models,
including storage, loading, versioning, and metadata management.
"""

import logging
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import torch
import datetime
import shutil

from .sae_training import SparseAutoencoder, SAEConfig, SAETrainingResult
from .sae_feature_analysis import SAEFeatureAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class SAEModelMetadata:
    """Metadata for a stored SAE model."""
    model_id: str
    model_name: str
    base_model_name: str
    layer_index: int
    component: str

    # Training information
    training_date: str
    training_duration: float  # seconds
    config_hash: str

    # Model properties
    input_dim: int
    hidden_dim: int
    num_parameters: int

    # Performance metrics
    final_loss: float
    sparsity_achieved: float
    reconstruction_quality: float

    # File paths
    model_path: str
    config_path: str
    metadata_path: str

    # Optional analysis results
    feature_analysis_path: Optional[str] = None
    training_history_path: Optional[str] = None

    # Tags and description
    tags: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'base_model_name': self.base_model_name,
            'layer_index': self.layer_index,
            'component': self.component,
            'training_date': self.training_date,
            'training_duration': self.training_duration,
            'config_hash': self.config_hash,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_parameters': self.num_parameters,
            'final_loss': self.final_loss,
            'sparsity_achieved': self.sparsity_achieved,
            'reconstruction_quality': self.reconstruction_quality,
            'model_path': self.model_path,
            'config_path': self.config_path,
            'metadata_path': self.metadata_path,
            'feature_analysis_path': self.feature_analysis_path,
            'training_history_path': self.training_history_path,
            'tags': self.tags,
            'description': self.description,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SAEModelMetadata':
        """Create from dictionary."""
        return cls(**data)


class SAEModelHub:
    """Central hub for managing SAE models."""

    def __init__(self, hub_directory: str = "sae_models"):
        self.hub_directory = Path(hub_directory)
        self.hub_directory.mkdir(parents=True, exist_ok=True)

        # Initialize hub structure
        self.models_dir = self.hub_directory / "models"
        self.metadata_dir = self.hub_directory / "metadata"
        self.configs_dir = self.hub_directory / "configs"
        self.analyses_dir = self.hub_directory / "analyses"

        for dir_path in [
                self.models_dir,
                self.metadata_dir,
                self.configs_dir,
                self.analyses_dir]:
            dir_path.mkdir(exist_ok=True)

        # Load existing models registry
        self.registry_path = self.hub_directory / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, SAEModelMetadata]:
        """Load the models registry."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                return {
                    model_id: SAEModelMetadata.from_dict(metadata)
                    for model_id, metadata in data.items()
                }
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

        return {}

    def _save_registry(self):
        """Save the models registry."""
        data = {
            model_id: metadata.to_dict()
            for model_id, metadata in self.registry.items()
        }

        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Registry saved with {len(self.registry)} models")

    def _generate_model_id(
            self,
            base_model_name: str,
            layer_index: int,
            component: str,
            config_hash: str) -> str:
        """Generate a unique model ID."""
        # Create a hash from the key components
        id_string = f"{base_model_name}_{layer_index}_{component}_{config_hash}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]

    def _compute_config_hash(self, config: SAEConfig) -> str:
        """Compute hash of the configuration."""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def register_model(
        self,
        training_result: SAETrainingResult,
        base_model_name: str,
        layer_index: int,
        model_name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a trained SAE model in the hub.

        Args:
            training_result: Result from SAE training
            base_model_name: Name of the base transformer model
            layer_index: Layer index the SAE was trained on
            model_name: Optional custom model name
            description: Model description
            tags: Optional tags for categorization

        Returns:
            Model ID
        """
        logger.info(f"Registering SAE model for {base_model_name} layer {layer_index}")

        # Generate model ID
        config_hash = self._compute_config_hash(training_result.config)
        model_id = self._generate_model_id(base_model_name, layer_index, config_hash)

        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{base_model_name}_layer{layer_index}_{timestamp}"

        # Set up file paths
        model_filename = f"{model_id}.pt"
        config_filename = f"{model_id}_config.json"
        metadata_filename = f"{model_id}_metadata.json"
        history_filename = f"{model_id}_history.pkl"

        model_path = self.models_dir / model_filename
        config_path = self.configs_dir / config_filename
        metadata_path = self.metadata_dir / metadata_filename
        history_path = self.analyses_dir / history_filename

        # Save model
        torch.save({
            'model_state_dict': training_result.model.state_dict(),
            'config': training_result.config.to_dict(),
            'training_info': training_result.to_dict()
        }, model_path)

        # Save config
        with open(config_path, 'w') as f:
            json.dump(training_result.config.to_dict(), f, indent=2)

        # Save training history
        with open(history_path, 'wb') as f:
            pickle.dump(training_result.training_history, f)

        # Create metadata
        metadata = SAEModelMetadata(
            model_id=model_id,
            model_name=model_name,
            base_model_name=base_model_name,
            layer_index=layer_index,
            component=training_result.config.component,
            training_date=datetime.datetime.now().isoformat(),
            training_duration=training_result.training_time,
            config_hash=config_hash,
            input_dim=training_result.config.input_dim,
            hidden_dim=training_result.config.hidden_dim,
            num_parameters=training_result.final_metrics.get('num_parameters', 0),
            final_loss=training_result.final_metrics.get('final_train_loss', 0.0),
            sparsity_achieved=training_result.final_metrics.get(
                'sparsity_achieved', {}).get('effective_sparsity', 0.0),
            reconstruction_quality=1.0 -
            training_result.final_metrics.get('final_train_loss', 1.0),  # Approximation
            model_path=str(model_path),
            config_path=str(config_path),
            metadata_path=str(metadata_path),
            training_history_path=str(history_path),
            tags=tags or [],
            description=description
        )

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Register in hub
        self.registry[model_id] = metadata
        self._save_registry()

        logger.info(f"Model registered with ID: {model_id}")
        return model_id

    def load_model(self,
                   model_id: str,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load a model from the hub.

        Args:
            model_id: Model ID to load
            device: Device to load the model onto

        Returns:
            Dictionary containing model, config, and metadata
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self.registry[model_id]
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(metadata.model_path, map_location=device)
        config = SAEConfig.from_dict(checkpoint['config'])

        model = SparseAutoencoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        logger.info(f"Loaded model {model_id}: {metadata.model_name}")
        return {
            'model': model,
            'config': config,
            'metadata': metadata,
            'device': device
        }

    def list_models(
        self,
        base_model_name: Optional[str] = None,
        layer_index: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> List[SAEModelMetadata]:
        """
        List models in the hub with optional filtering.

        Args:
            base_model_name: Filter by base model name
            layer_index: Filter by layer index
            tags: Filter by tags (any of the provided tags)

        Returns:
            List of model metadata
        """
        models = list(self.registry.values())

        # Apply filters
        if base_model_name:
            models = [m for m in models if m.base_model_name == base_model_name]

        if layer_index is not None:
            models = [m for m in models if m.layer_index == layer_index]

        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]

        # Sort by training date (newest first)
        models.sort(key=lambda m: m.training_date, reverse=True)

        return models

    def delete_model(self, model_id: str, confirm: bool = False):
        """
        Delete a model from the hub.

        Args:
            model_id: Model ID to delete
            confirm: Confirmation flag
        """
        if not confirm:
            raise ValueError("Set confirm=True to delete model")

        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self.registry[model_id]

        # Delete files
        for path_str in [metadata.model_path, metadata.config_path,
                         metadata.metadata_path, metadata.training_history_path]:
            if path_str:
                path = Path(path_str)
                if path.exists():
                    path.unlink()

        # Delete feature analysis if exists
        if metadata.feature_analysis_path:
            path = Path(metadata.feature_analysis_path)
            if path.exists():
                path.unlink()

        # Remove from registry
        del self.registry[model_id]
        self._save_registry()

        logger.info(f"Deleted model {model_id}")

    def attach_feature_analysis(
            self,
            model_id: str,
            analysis_result: SAEFeatureAnalysisResult):
        """
        Attach feature analysis results to a model.

        Args:
            model_id: Model ID
            analysis_result: Feature analysis results
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Save analysis results
        analysis_filename = f"{model_id}_feature_analysis.json"
        analysis_path = self.analyses_dir / analysis_filename

        with open(analysis_path, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2)

        # Update metadata
        self.registry[model_id].feature_analysis_path = str(analysis_path)
        self._save_registry()

        logger.info(f"Feature analysis attached to model {model_id}")

    def search_models(self, query: str) -> List[SAEModelMetadata]:
        """
        Search models by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching model metadata
        """
        query = query.lower()
        matches = []

        for metadata in self.registry.values():
            # Search in name, description, and tags
            search_text = " ".join([
                metadata.model_name.lower(),
                metadata.description.lower(),
                " ".join(metadata.tags).lower()
            ])

            if query in search_text:
                matches.append(metadata)

        return matches

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self.registry[model_id]
        info = metadata.to_dict()

        # Add file sizes
        for path_key in [
            'model_path',
            'config_path',
            'training_history_path',
                'feature_analysis_path']:
            path_str = info.get(path_key)
            if path_str:
                path = Path(path_str)
                if path.exists():
                    info[f"{path_key}_size"] = path.stat().st_size

        return info

    def export_model(
            self,
            model_id: str,
            export_path: str,
            include_analysis: bool = True):
        """
        Export a model and its associated files.

        Args:
            model_id: Model ID to export
            export_path: Directory to export to
            include_analysis: Whether to include analysis results
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self.registry[model_id]
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        files_to_copy = [
            metadata.model_path,
            metadata.config_path,
            metadata.metadata_path,
            metadata.training_history_path
        ]

        if include_analysis and metadata.feature_analysis_path:
            files_to_copy.append(metadata.feature_analysis_path)

        for file_path in files_to_copy:
            if file_path and Path(file_path).exists():
                dest_path = export_dir / Path(file_path).name
                shutil.copy2(file_path, dest_path)

        logger.info(f"Model {model_id} exported to {export_path}")

    def get_hub_statistics(self) -> Dict[str, Any]:
        """Get statistics about the model hub."""
        models = list(self.registry.values())

        if not models:
            return {'num_models': 0}

        # Count by base model
        base_model_counts = {}
        layer_counts = {}
        total_size = 0

        for metadata in models:
            # Count base models
            base_model_counts[metadata.base_model_name] = base_model_counts.get(
                metadata.base_model_name, 0) + 1

            # Count layers
            layer_counts[metadata.layer_index] = layer_counts.get(
                metadata.layer_index, 0) + 1

            # Calculate total size
            if Path(metadata.model_path).exists():
                total_size += Path(metadata.model_path).stat().st_size

        return {
            'num_models': len(models),
            'base_model_counts': base_model_counts,
            'layer_counts': layer_counts,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_sparsity': sum(m.sparsity_achieved for m in models) / len(models),
            'avg_reconstruction_quality': sum(m.reconstruction_quality for m in models) / len(models)
        }


# Global hub instance
_global_hub = None


def get_sae_hub(hub_directory: Optional[str] = None) -> SAEModelHub:
    """Get the global SAE model hub instance."""
    global _global_hub

    if _global_hub is None or (
            hub_directory and hub_directory != str(
            _global_hub.hub_directory)):
        _global_hub = SAEModelHub(hub_directory or "sae_models")

    return _global_hub


# Convenience functions
def register_sae_model(
    training_result: SAETrainingResult,
    **kwargs
) -> str:
    """Register a SAE model in the global hub."""
    hub = get_sae_hub()
    return hub.register_model(
        training_result,
        base_model_name=training_result.config.model_name,
        layer_index=training_result.config.layer,
        component=training_result.config.component,
        **kwargs
    )


def load_sae_from_hub(
        model_id: str) -> Tuple[SparseAutoencoder, SAEConfig, SAEModelMetadata]:
    """Load a SAE model from the global hub."""
    hub = get_sae_hub()
    loaded_info = hub.load_model(model_id)
    return loaded_info['model'], loaded_info['config'], loaded_info['metadata']


def list_sae_models(**kwargs) -> List[SAEModelMetadata]:
    """List SAE models in the global hub."""
    hub = get_sae_hub()
    return hub.list_models(**kwargs)


def search_sae_models(query: str) -> List[SAEModelMetadata]:
    """Search SAE models in the global hub."""
    hub = get_sae_hub()
    return hub.search_models(query)
