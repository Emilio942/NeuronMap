"""Model compatibility checking and capability detection for NeuronMap.

This module provides comprehensive model compatibility validation, resource requirement
estimation, and intelligent fallback suggestions according to roadmap section 2.2.
"""

import torch
import logging
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import importlib
import re
from pathlib import Path

from .error_handling import (
    NeuronMapException, ValidationError, ModelCompatibilityError
)
from .robust_decorators import robust_execution

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    GPT2 = "gpt2"
    BERT = "bert"
    T5 = "t5"
    LLAMA = "llama"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    CUSTOM = "custom"


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    ACTIVATION_EXTRACTION = "activation_extraction"
    LAYER_ANALYSIS = "layer_analysis"
    ATTENTION_VISUALIZATION = "attention_visualization"
    GRADIENT_ANALYSIS = "gradient_analysis"
    CONCEPTUAL_ANALYSIS = "conceptual_analysis"
    FULL_ANALYSIS = "full_analysis"


@dataclass
class ModelInfo:
    """Information about a model's capabilities and requirements."""
    name: str
    model_type: ModelType
    size_gb: float
    min_memory_gb: float
    recommended_memory_gb: float
    min_cuda_version: Optional[str] = None
    supported_analysis: List[AnalysisType] = None
    layer_count: Optional[int] = None
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    max_sequence_length: Optional[int] = None
    requires_internet: bool = False
    license_restrictions: bool = False


@dataclass
class SystemResources:
    """Current system resource availability."""
    cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_available: bool
    gpu_count: int = 0
    gpu_memory_gb: List[float] = None
    gpu_names: List[str] = None
    cuda_version: Optional[str] = None
    storage_available_gb: float = 0.0
    internet_available: bool = True


@dataclass
class CompatibilityResult:
    """Result of compatibility checking."""
    compatible: bool
    confidence_score: float  # 0.0 to 1.0
    warnings: List[str] = None
    errors: List[str] = None
    recommendations: List[str] = None
    estimated_memory_usage_gb: Optional[float] = None
    estimated_processing_time_minutes: Optional[float] = None
    fallback_suggestions: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.recommendations is None:
            self.recommendations = []
        if self.fallback_suggestions is None:
            self.fallback_suggestions = []


class ModelRegistry:
    """Registry of known models and their characteristics."""

    def __init__(self):
        self._models = self._initialize_model_database()

    def _initialize_model_database(self) -> Dict[str, ModelInfo]:
        """Initialize the database of known models."""
        return {
            "gpt2": ModelInfo(
                name="gpt2",
                model_type=ModelType.GPT2,
                size_gb=0.5,
                min_memory_gb=2.0,
                recommended_memory_gb=4.0,
                supported_analysis=[
                    AnalysisType.ACTIVATION_EXTRACTION,
                    AnalysisType.LAYER_ANALYSIS,
                    AnalysisType.ATTENTION_VISUALIZATION,
                    AnalysisType.CONCEPTUAL_ANALYSIS
                ],
                layer_count=12,
                hidden_size=768,
                vocab_size=50257,
                max_sequence_length=1024,
                requires_internet=True
            ),
            "gpt2-medium": ModelInfo(
                name="gpt2-medium",
                model_type=ModelType.GPT2,
                size_gb=1.5,
                min_memory_gb=4.0,
                recommended_memory_gb=8.0,
                supported_analysis=[
                    AnalysisType.ACTIVATION_EXTRACTION,
                    AnalysisType.LAYER_ANALYSIS,
                    AnalysisType.ATTENTION_VISUALIZATION,
                    AnalysisType.CONCEPTUAL_ANALYSIS
                ],
                layer_count=24,
                hidden_size=1024,
                vocab_size=50257,
                max_sequence_length=1024,
                requires_internet=True
            ),
            "gpt2-large": ModelInfo(
                name="gpt2-large",
                model_type=ModelType.GPT2,
                size_gb=3.0,
                min_memory_gb=8.0,
                recommended_memory_gb=16.0,
                supported_analysis=[
                    AnalysisType.ACTIVATION_EXTRACTION,
                    AnalysisType.LAYER_ANALYSIS,
                    AnalysisType.ATTENTION_VISUALIZATION,
                    AnalysisType.CONCEPTUAL_ANALYSIS
                ],
                layer_count=36,
                hidden_size=1280,
                vocab_size=50257,
                max_sequence_length=1024,
                requires_internet=True
            ),
            "bert-base-uncased": ModelInfo(
                name="bert-base-uncased",
                model_type=ModelType.BERT,
                size_gb=0.4,
                min_memory_gb=2.0,
                recommended_memory_gb=4.0,
                supported_analysis=[
                    AnalysisType.ACTIVATION_EXTRACTION,
                    AnalysisType.LAYER_ANALYSIS,
                    AnalysisType.ATTENTION_VISUALIZATION
                ],
                layer_count=12,
                hidden_size=768,
                vocab_size=30522,
                max_sequence_length=512,
                requires_internet=True
            ),
            "t5-small": ModelInfo(
                name="t5-small",
                model_type=ModelType.T5,
                size_gb=0.2,
                min_memory_gb=1.5,
                recommended_memory_gb=3.0,
                supported_analysis=[
                    AnalysisType.ACTIVATION_EXTRACTION,
                    AnalysisType.LAYER_ANALYSIS
                ],
                layer_count=6,
                hidden_size=512,
                vocab_size=32128,
                max_sequence_length=512,
                requires_internet=True
            )
        }

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information for a specific model."""
        # Direct lookup
        if model_name in self._models:
            return self._models[model_name]

        # Fuzzy matching for variations
        normalized_name = model_name.lower().replace("-", "").replace("_", "")
        for name, info in self._models.items():
            if normalized_name in name.lower().replace("-", "").replace("_", ""):
                logger.info(f"Fuzzy matched '{model_name}' to '{name}'")
                return info

        return None

    def list_compatible_models(self, analysis_type: AnalysisType,
                              system_resources: SystemResources) -> List[str]:
        """List models compatible with given analysis type and resources."""
        compatible = []
        for name, info in self._models.items():
            if (analysis_type in info.supported_analysis and
                info.min_memory_gb <= system_resources.available_memory_gb):
                compatible.append(name)
        return compatible


class CapabilityDatabase:
    """Database of analysis requirements and capabilities."""

    def __init__(self):
        self._requirements = self._initialize_requirements()

    def _initialize_requirements(self) -> Dict[AnalysisType, Dict[str, Any]]:
        """Initialize analysis requirements database."""
        return {
            AnalysisType.ACTIVATION_EXTRACTION: {
                "min_memory_multiplier": 1.5,  # Model size * multiplier
                "gpu_preferred": False,
                "requires_gradients": False,
                "batch_size_impact": "high",
                "sequence_length_impact": "medium"
            },
            AnalysisType.LAYER_ANALYSIS: {
                "min_memory_multiplier": 2.0,
                "gpu_preferred": True,
                "requires_gradients": False,
                "batch_size_impact": "medium",
                "sequence_length_impact": "low"
            },
            AnalysisType.ATTENTION_VISUALIZATION: {
                "min_memory_multiplier": 2.5,
                "gpu_preferred": True,
                "requires_gradients": False,
                "batch_size_impact": "high",
                "sequence_length_impact": "high"
            },
            AnalysisType.GRADIENT_ANALYSIS: {
                "min_memory_multiplier": 3.0,
                "gpu_preferred": True,
                "requires_gradients": True,
                "batch_size_impact": "very_high",
                "sequence_length_impact": "high"
            },
            AnalysisType.CONCEPTUAL_ANALYSIS: {
                "min_memory_multiplier": 1.8,
                "gpu_preferred": False,
                "requires_gradients": False,
                "batch_size_impact": "medium",
                "sequence_length_impact": "medium"
            }
        }

    def get_requirements(self, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Get requirements for a specific analysis type."""
        return self._requirements.get(analysis_type, {})


class ModelCompatibilityChecker:
    """Main class for checking model compatibility and providing recommendations."""

    def __init__(self):
        self.model_registry = ModelRegistry()
        self.capability_database = CapabilityDatabase()
        self._system_resources = None

    @robust_execution(max_retries=2, retry_delay=1.0)
    def get_system_resources(self) -> SystemResources:
        """Get current system resource information."""
        if self._system_resources is None:
            self._system_resources = self._detect_system_resources()
        return self._system_resources

    def _detect_system_resources(self) -> SystemResources:
        """Detect current system resources."""
        # CPU and Memory
        cpu_cores = psutil.cpu_count()
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)

        # Storage
        disk = psutil.disk_usage('/')
        storage_available_gb = disk.free / (1024**3)

        # GPU detection
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_memory_gb = []
        gpu_names = []
        cuda_version = None

        if gpu_available:
            cuda_version = torch.version.cuda
            for i in range(gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpu_memory_gb.append(props.total_memory / (1024**3))
                    gpu_names.append(props.name)
                except Exception as e:
                    logger.warning(f"Could not get GPU {i} properties: {e}")

        # Internet connectivity check
        internet_available = self._check_internet_connectivity()

        return SystemResources(
            cpu_cores=cpu_cores,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            gpu_names=gpu_names,
            cuda_version=cuda_version,
            storage_available_gb=storage_available_gb,
            internet_available=internet_available
        )

    def _check_internet_connectivity(self) -> bool:
        """Check if internet connection is available."""
        try:
            import requests
            response = requests.get("https://huggingface.co", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def check_compatibility(self, model_name: str, analysis_type: AnalysisType,
                          batch_size: int = 1, sequence_length: int = 512,
                          system_resources: Optional[SystemResources] = None) -> CompatibilityResult:
        """
        Check compatibility between model, analysis type, and system resources.

        Args:
            model_name: Name of the model to check
            analysis_type: Type of analysis to perform
            batch_size: Batch size for processing
            sequence_length: Maximum sequence length
            system_resources: Override system resource detection

        Returns:
            CompatibilityResult with detailed compatibility information
        """
        if system_resources is None:
            system_resources = self.get_system_resources()

        # Get model information
        model_info = self.model_registry.get_model_info(model_name)
        if model_info is None:
            return CompatibilityResult(
                compatible=False,
                confidence_score=0.0,
                errors=[f"Unknown model: {model_name}"],
                fallback_suggestions=self._suggest_similar_models(model_name)
            )

        # Get analysis requirements
        requirements = self.capability_database.get_requirements(analysis_type)
        if not requirements:
            return CompatibilityResult(
                compatible=False,
                confidence_score=0.0,
                errors=[f"Unknown analysis type: {analysis_type}"]
            )

        # Perform compatibility checks
        result = CompatibilityResult(compatible=True, confidence_score=1.0)

        # Check if analysis is supported by model
        if analysis_type not in model_info.supported_analysis:
            result.compatible = False
            result.errors.append(
                f"Model {model_name} does not support {analysis_type.value} analysis"
            )
            result.fallback_suggestions.extend(
                self._suggest_alternative_analysis(model_info, analysis_type)
            )

        # Memory requirement check
        memory_required = self._estimate_memory_usage(
            model_info, requirements, batch_size, sequence_length
        )
        result.estimated_memory_usage_gb = memory_required

        if memory_required > system_resources.available_memory_gb:
            result.compatible = False
            result.errors.append(
                f"Insufficient memory: {memory_required:.1f}GB required, "
                f"{system_resources.available_memory_gb:.1f}GB available"
            )
            result.fallback_suggestions.extend(
                self._suggest_memory_optimization(memory_required, system_resources)
            )
        elif memory_required > system_resources.available_memory_gb * 0.8:
            result.warnings.append(
                f"Memory usage will be high: {memory_required:.1f}GB of "
                f"{system_resources.available_memory_gb:.1f}GB available"
            )
            result.confidence_score *= 0.7

        # GPU requirements check
        if requirements.get("gpu_preferred") and not system_resources.gpu_available:
            result.warnings.append(
                "GPU recommended for optimal performance but not available"
            )
            result.confidence_score *= 0.8
            result.recommendations.append("Consider using GPU for better performance")

        # Internet connectivity check
        if model_info.requires_internet and not system_resources.internet_available:
            result.compatible = False
            result.errors.append(
                "Internet connection required to download model but not available"
            )

        # CUDA version check
        if (system_resources.gpu_available and model_info.min_cuda_version and
            system_resources.cuda_version and
            self._compare_versions(system_resources.cuda_version, model_info.min_cuda_version) < 0):
            result.warnings.append(
                f"CUDA version {system_resources.cuda_version} may be incompatible. "
                f"Minimum required: {model_info.min_cuda_version}"
            )
            result.confidence_score *= 0.9

        # Storage space check
        if model_info.size_gb > system_resources.storage_available_gb:
            result.compatible = False
            result.errors.append(
                f"Insufficient storage: {model_info.size_gb:.1f}GB required, "
                f"{system_resources.storage_available_gb:.1f}GB available"
            )

        # Estimate processing time
        result.estimated_processing_time_minutes = self._estimate_processing_time(
            model_info, analysis_type, batch_size, sequence_length, system_resources
        )

        # Add optimization recommendations
        if result.compatible:
            result.recommendations.extend(
                self._suggest_optimizations(model_info, analysis_type, system_resources)
            )

        return result

    def _estimate_memory_usage(self, model_info: ModelInfo, requirements: Dict[str, Any],
                             batch_size: int, sequence_length: int) -> float:
        """Estimate memory usage for the analysis."""
        base_memory = model_info.size_gb
        multiplier = requirements.get("min_memory_multiplier", 1.5)

        # Adjust for batch size
        batch_impact = requirements.get("batch_size_impact", "medium")
        if batch_impact == "very_high":
            batch_multiplier = 1 + (batch_size - 1) * 0.8
        elif batch_impact == "high":
            batch_multiplier = 1 + (batch_size - 1) * 0.6
        elif batch_impact == "medium":
            batch_multiplier = 1 + (batch_size - 1) * 0.4
        else:  # low
            batch_multiplier = 1 + (batch_size - 1) * 0.2

        # Adjust for sequence length
        seq_impact = requirements.get("sequence_length_impact", "medium")
        if model_info.max_sequence_length:
            seq_ratio = sequence_length / model_info.max_sequence_length
            if seq_impact == "high":
                seq_multiplier = 1 + (seq_ratio - 1) * 0.5
            elif seq_impact == "medium":
                seq_multiplier = 1 + (seq_ratio - 1) * 0.3
            else:  # low
                seq_multiplier = 1 + (seq_ratio - 1) * 0.1
            seq_multiplier = max(seq_multiplier, 0.5)  # Don't go below 50%
        else:
            seq_multiplier = 1.0

        return base_memory * multiplier * batch_multiplier * seq_multiplier

    def _estimate_processing_time(self, model_info: ModelInfo, analysis_type: AnalysisType,
                                batch_size: int, sequence_length: int,
                                system_resources: SystemResources) -> float:
        """Estimate processing time in minutes."""
        # Base time estimation (very rough)
        base_time_per_sample = {
            AnalysisType.ACTIVATION_EXTRACTION: 0.1,
            AnalysisType.LAYER_ANALYSIS: 0.2,
            AnalysisType.ATTENTION_VISUALIZATION: 0.3,
            AnalysisType.GRADIENT_ANALYSIS: 0.5,
            AnalysisType.CONCEPTUAL_ANALYSIS: 0.15
        }.get(analysis_type, 0.2)

        # Model size factor
        size_factor = model_info.size_gb / 0.5  # Relative to GPT-2 small

        # GPU acceleration factor
        gpu_factor = 0.3 if system_resources.gpu_available else 1.0

        # Sequence length factor
        if model_info.max_sequence_length:
            seq_factor = sequence_length / model_info.max_sequence_length
        else:
            seq_factor = sequence_length / 512  # Default reference

        return base_time_per_sample * size_factor * gpu_factor * seq_factor * batch_size

    def _suggest_similar_models(self, model_name: str) -> List[str]:
        """Suggest similar models for unknown model names."""
        suggestions = []
        model_name_lower = model_name.lower()

        # Check for partial matches
        for name in self.model_registry._models.keys():
            if any(part in name.lower() for part in model_name_lower.split("-")):
                suggestions.append(f"Did you mean '{name}'?")

        if not suggestions:
            suggestions.append("Try 'gpt2' for a general-purpose language model")
            suggestions.append("Try 'bert-base-uncased' for text understanding tasks")

        return suggestions[:3]  # Limit to 3 suggestions

    def _suggest_alternative_analysis(self, model_info: ModelInfo,
                                    analysis_type: AnalysisType) -> List[str]:
        """Suggest alternative analysis types for unsupported combinations."""
        supported = model_info.supported_analysis
        if not supported:
            return ["No alternative analysis types available for this model"]

        suggestions = []
        for supported_type in supported:
            if supported_type != analysis_type:
                suggestions.append(f"Try {supported_type.value} instead")

        return suggestions[:2]  # Limit to 2 suggestions

    def _suggest_memory_optimization(self, required_gb: float,
                                   system_resources: SystemResources) -> List[str]:
        """Suggest memory optimization strategies."""
        suggestions = []
        deficit_gb = required_gb - system_resources.available_memory_gb

        suggestions.append(f"Reduce batch size to lower memory usage by ~{deficit_gb:.1f}GB")
        suggestions.append("Try a smaller model variant (e.g., gpt2 instead of gpt2-large)")
        suggestions.append("Close other applications to free up memory")

        if not system_resources.gpu_available:
            suggestions.append("Consider using GPU to offload computation from CPU memory")

        return suggestions

    def _suggest_optimizations(self, model_info: ModelInfo, analysis_type: AnalysisType,
                             system_resources: SystemResources) -> List[str]:
        """Suggest performance optimizations."""
        suggestions = []

        if system_resources.gpu_available and system_resources.gpu_count > 1:
            suggestions.append("Consider using model parallelization for multi-GPU setup")

        if analysis_type in [AnalysisType.ACTIVATION_EXTRACTION, AnalysisType.LAYER_ANALYSIS]:
            suggestions.append("Use mixed precision (fp16) to reduce memory usage")

        if model_info.model_type == ModelType.GPT2:
            suggestions.append("Enable gradient checkpointing for memory efficiency")

        return suggestions

    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare version strings. Returns -1, 0, or 1."""
        def normalize_version(v):
            return [int(x) for x in re.sub(r'[^\d.]', '', v).split('.') if x.isdigit()]

        v1_parts = normalize_version(version1)
        v2_parts = normalize_version(version2)

        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts += [0] * (max_len - len(v1_parts))
        v2_parts += [0] * (max_len - len(v2_parts))

        for v1_part, v2_part in zip(v1_parts, v2_parts):
            if v1_part < v2_part:
                return -1
            elif v1_part > v2_part:
                return 1
        return 0

    def validate_model_compatibility(self, model_name: str,
                                   analysis_types: List[AnalysisType],
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, CompatibilityResult]:
        """
        Validate compatibility for multiple analysis types.

        Args:
            model_name: Name of the model
            analysis_types: List of analysis types to check
            config: Optional configuration overrides

        Returns:
            Dictionary mapping analysis types to compatibility results
        """
        results = {}
        system_resources = self.get_system_resources()

        batch_size = config.get("batch_size", 1) if config else 1
        sequence_length = config.get("max_length", 512) if config else 512

        for analysis_type in analysis_types:
            try:
                results[analysis_type.value] = self.check_compatibility(
                    model_name, analysis_type, batch_size, sequence_length, system_resources
                )
            except Exception as e:
                logger.error(f"Error checking compatibility for {analysis_type}: {e}")
                results[analysis_type.value] = CompatibilityResult(
                    compatible=False,
                    confidence_score=0.0,
                    errors=[f"Error during compatibility check: {str(e)}"]
                )

        return results


# Convenience functions for easy usage
def check_model_compatibility(model_name: str, analysis_type: str = "activation_extraction",
                            **kwargs) -> CompatibilityResult:
    """Convenience function for checking model compatibility."""
    checker = ModelCompatibilityChecker()
    try:
        analysis_enum = AnalysisType(analysis_type)
    except ValueError:
        return CompatibilityResult(
            compatible=False,
            confidence_score=0.0,
            errors=[f"Invalid analysis type: {analysis_type}"]
        )

    return checker.check_compatibility(model_name, analysis_enum, **kwargs)


def get_compatible_models(analysis_type: str = "activation_extraction") -> List[str]:
    """Get list of models compatible with the current system."""
    checker = ModelCompatibilityChecker()
    try:
        analysis_enum = AnalysisType(analysis_type)
        system_resources = checker.get_system_resources()
        return checker.model_registry.list_compatible_models(analysis_enum, system_resources)
    except Exception as e:
        logger.error(f"Error getting compatible models: {e}")
        return []


if __name__ == "__main__":
    # Demo usage
    checker = ModelCompatibilityChecker()

    # Check system resources
    resources = checker.get_system_resources()
    print("System Resources:")
    print(f"  CPU cores: {resources.cpu_cores}")
    print(f"  Memory: {resources.available_memory_gb:.1f}/{resources.total_memory_gb:.1f} GB")
    print(f"  GPU available: {resources.gpu_available}")
    if resources.gpu_available:
        print(f"  GPU count: {resources.gpu_count}")
        print(f"  CUDA version: {resources.cuda_version}")
    print()

    # Check compatibility for different models
    models_to_test = ["gpt2", "gpt2-large", "bert-base-uncased", "unknown-model"]
    analysis_type = AnalysisType.ACTIVATION_EXTRACTION

    for model in models_to_test:
        print(f"Checking compatibility for {model}:")
        result = checker.check_compatibility(model, analysis_type)
        print(f"  Compatible: {result.compatible}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        if result.estimated_memory_usage_gb:
            print(f"  Est. memory: {result.estimated_memory_usage_gb:.1f}GB")
        if result.estimated_processing_time_minutes:
            print(f"  Est. time: {result.estimated_processing_time_minutes:.1f}min")
        if result.errors:
            print(f"  Errors: {result.errors}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
        if result.recommendations:
            print(f"  Recommendations: {result.recommendations}")
        if result.fallback_suggestions:
            print(f"  Suggestions: {result.fallback_suggestions}")
        print()
