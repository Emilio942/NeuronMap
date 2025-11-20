"""
Enhanced Configuration Management System for NeuronMap
====================================================

This module provides a robust, validated configuration system using Pydantic
for type checking and YAML/JSON support for flexible configuration files.
"""

try:
    from pydantic import BaseModel, field_validator, ConfigDict, Field, ValidationError, PrivateAttr
except ImportError:
    from pydantic import BaseModel, validator as field_validator, ValidationError
    # Fallback für ältere Pydantic Versionen
    ConfigDict = None
    Field = lambda default=None, **kwargs: default
    PrivateAttr = lambda default=None: None

import yaml
import json
import logging
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when configuration validation or loading fails."""
    pass

class Environment(str, Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ConfigValidationError:
    """Configuration validation error."""
    field: str
    message: str
    value: Any
    suggestion: Optional[str] = None


class ModelConfig(BaseModel):
    """Configuration for neural network models"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields

    name: str = Field(default="distilgpt2", description="Model identifier")
    device: str = Field(default="auto", description="Device for model execution")
    max_length: int = Field(default=512, gt=0, le=32768, description="Maximum sequence length")
    batch_size: int = Field(default=1, gt=0, le=512, description="Batch size for processing")
    target_layers: List[str] = Field(default=["transformer.h.5.mlp.c_proj"], description="Target layers for analysis")

    # Extended model configuration
    layer_count: Optional[int] = Field(default=None, gt=0, le=100, description="Number of layers")
    hidden_size: Optional[int] = Field(default=None, gt=0, description="Hidden dimension size")
    attention_heads: Optional[int] = Field(default=None, gt=0, description="Number of attention heads")
    max_memory_gb: float = Field(default=8.0, gt=0, le=80, description="Maximum memory usage in GB")
    requires_gpu: bool = Field(default=True, description="Whether model requires GPU")
    quantization_bits: Optional[int] = Field(default=None, description="Quantization bits (8, 16, None)")

    # Additional fields from YAML configs
    type: Optional[str] = Field(default=None, description="Model type (gpt, bert, t5, llama)")
    layers: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Layer configuration")

    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        valid_devices = ["auto", "cpu", "cuda"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v

    @field_validator('quantization_bits')
    @classmethod
    def validate_quantization(cls, v):
        if v is not None and v not in [8, 16]:
            raise ValueError("Quantization bits must be 8, 16, or None")
        return v

    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()


class DataConfig(BaseModel):
    """Configuration for data processing"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields

    input_file: str = Field(default="generated_questions.jsonl", description="Input file path")
    output_file: str = Field(default="activation_results.csv", description="Output file path")
    data_dir: str = Field(default="data", description="Data directory")
    raw_dir: str = Field(default="data/raw", description="Raw data directory")
    processed_dir: str = Field(default="data/processed", description="Processed data directory")
    outputs_dir: str = Field(default="data/outputs", description="Output data directory")

    # Processing configuration
    batch_size: int = Field(default=32, gt=0, le=512, description="Batch size for processing")
    max_samples: int = Field(default=1000, gt=0, description="Maximum number of samples to process")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_dir: str = Field(default="data/cache", description="Cache directory")
    duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Threshold for duplicate detection")

    @field_validator('data_dir', 'cache_dir', 'raw_dir', 'processed_dir', 'outputs_dir')
    @classmethod
    def validate_directories(cls, v):
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        except Exception as e:
            raise ValueError(f"Cannot create directory {v}: {e}")

# AnalysisConfig and VisualizationConfig are defined below with more complete configurations

class SystemConfig(BaseModel):
    """System-wide configuration"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields

    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Runtime environment")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    max_concurrent_jobs: int = Field(default=4, gt=0, le=32, description="Maximum concurrent jobs")
    job_timeout_minutes: int = Field(default=30, gt=0, le=1440, description="Job timeout in minutes")
    temp_directory: str = Field(default="data/temp", description="Temporary files directory")
    output_directory: str = Field(default="data/outputs", description="Output files directory")
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    gpu_memory_fraction: float = Field(default=0.8, gt=0.0, le=1.0, description="GPU memory fraction")

class WebConfig(BaseModel):
    """Web interface configuration"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields

    host: str = Field(default="127.0.0.1", description="Web server host")
    port: int = Field(default=5000, gt=0, le=65535, description="Web server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    secret_key: str = Field(default="dev-secret-key", description="Flask secret key")
    max_upload_size_mb: int = Field(default=100, gt=0, le=1000, description="Maximum upload size in MB")
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    outputs_dir: str = "data/outputs"


class AnalysisConfig(BaseModel):
    """Configuration for analysis parameters"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields

    # Basic analysis settings
    aggregation_method: str = "mean"  # "mean", "max", "sum"
    normalization: bool = True
    dimensionality_reduction: Optional[str] = None  # "pca", "tsne", "umap"
    clustering_method: Optional[str] = None  # "kmeans", "dbscan", "hierarchical"

    # Processing configuration from YAML files
    batch_size: Optional[int] = Field(default=32, description="Batch size for processing")
    max_sequence_length: Optional[int] = Field(default=512, description="Maximum sequence length")
    device: Optional[str] = Field(default="auto", description="Device for computation")
    precision: Optional[str] = Field(default="float32", description="Numerical precision")
    enable_gradient_checkpointing: Optional[bool] = Field(default=False, description="Enable gradient checkpointing")
    cache_activations: Optional[bool] = Field(default=True, description="Cache activations")

    # Memory optimization settings
    memory_optimization: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Memory optimization settings")

    # Statistical analysis settings
    statistics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Statistical analysis settings")

    # Clustering settings
    clustering: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Clustering settings")

    # Attention analysis settings
    attention: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Attention analysis settings")

    # Performance settings
    performance: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance settings")

    # Output settings
    output: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Output settings")

    @field_validator('aggregation_method')
    @classmethod
    def validate_aggregation(cls, v):
        valid_methods = ["mean", "max", "sum", "first", "last"]
        if v not in valid_methods:
            raise ValueError(f"Aggregation method must be one of {valid_methods}")
        return v


class VisualizationConfig(BaseModel):
    """Configuration for visualization"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields

    # Basic settings
    enabled: bool = True
    backend: str = "matplotlib"  # "matplotlib", "plotly", "seaborn"
    dpi: int = 300
    figure_width: int = 12
    figure_height: int = 8
    save_format: str = "png"  # "png", "pdf", "svg"

    # Extended settings from YAML files
    color_scheme: Optional[str] = Field(default="viridis", description="Color scheme")
    interactive: Optional[bool] = Field(default=True, description="Enable interactive plots")
    export_format: Optional[str] = Field(default="png", description="Export format")

    # Style settings
    style: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Style settings")
    color_schemes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Color schemes")
    plots: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Plot-specific settings")
    interactive_features: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Interactive features")
    export: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Export settings")
    dashboard: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Dashboard settings")
    animation: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Animation settings")


class ExperimentConfig(BaseModel):
    """Configuration for experiments"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields

    name: str = Field(default="default_experiment", description="Experiment name")
    description: str = Field(default="", description="Experiment description")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    logging_level: str = Field(default="INFO", description="Logging level")
    save_intermediate: bool = Field(default=True, description="Save intermediate results")

    @field_validator('logging_level')
    @classmethod
    def validate_logging_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Logging level must be one of {valid_levels}")
        return v.upper()


class GuardianConfig(BaseModel):
    """Configuration for the Guardian Network (Meta-cognitive layer)"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')

    enabled: bool = Field(default=False, description="Enable Guardian Network")
    mode: str = Field(default="monitoring", description="Operation mode: 'monitoring' or 'intervention'")
    intervention_layers: List[int] = Field(default_factory=list, description="Layer indices to hook into")
    guardian_model_path: Optional[str] = Field(default=None, description="Path to the secondary guardian model")
    
    # Flow Control Thresholds
    entropy_min: float = Field(default=0.5, description="Minimum entropy threshold (trigger noise)")
    entropy_max: float = Field(default=2.5, description="Maximum entropy threshold (trigger steering)")
    
    # Intervention Settings
    noise_std: float = Field(default=0.1, description="Standard deviation for noise injection")
    steering_coeff: float = Field(default=1.0, description="Coefficient for steering vectors")

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        valid_modes = ["monitoring", "intervention"]
        if v not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        return v


class NeuronMapConfig(BaseModel):
    """Main configuration class that combines all sub-configurations"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields for flexibility

    # Version and metadata
    version: str = Field(default="1.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now, description="Configuration creation time")
    config_dir: Path = Field(default_factory=lambda: Path("configs"), description="Configuration directory")

    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    guardian: GuardianConfig = Field(default_factory=GuardianConfig)

    # Multiple model support
    models: Dict[str, ModelConfig] = Field(default_factory=dict, description="Multiple model configurations")

    # Plugin configurations
    plugins: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Plugin-specific configurations")

    # Additional fields for config file compatibility
    layer_patterns: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Layer patterns")
    analysis_configs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis configurations")
    extraction_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extraction settings")
    environment: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Environment settings")

    # Top-level experiment configs
    default: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Default experiment config")
    dev: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Development config")
    prod: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Production config")
    experiments: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Experiments config")

    _models_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _experiments_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _layer_patterns_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, config_dir: Optional[Union[str, Path]] = None, **data: Any):
        if config_dir is not None:
            data['config_dir'] = Path(config_dir)
        super().__init__(**data)
        self.config_dir = Path(self.config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _read_yaml(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Configuration file {path} must contain a mapping at the top level")
            return data

    def load_models_config(self) -> Dict[str, Any]:
        if not self._models_cache:
            self._models_cache = self._read_yaml('models.yaml')
            self._layer_patterns_cache = self._models_cache.get('layer_patterns', {})
        return deepcopy(self._models_cache)

    def load_experiments_config(self) -> Dict[str, Any]:
        if not self._experiments_cache:
            self._experiments_cache = self._read_yaml('experiments.yaml')
        return deepcopy(self._experiments_cache)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        models = self.load_models_config().get('models', {})
        if model_name not in models:
            raise KeyError(f"Model '{model_name}' not found in models.yaml")
        return deepcopy(models[model_name])

    def get_experiment_config(self, experiment_name: str = "default") -> Dict[str, Any]:
        """Get experiment configuration by name."""
        experiments = self.load_experiments_config()

        if experiment_name in experiments:
            return deepcopy(experiments[experiment_name])

        nested = experiments.get('experiments', {})
        if experiment_name in nested:
            return deepcopy(nested[experiment_name])

        raise KeyError(f"Experiment '{experiment_name}' not found in experiments.yaml")

    def resolve_layer_name(self, model_config: Dict[str, Any], layer_type: str, layer_index: int) -> str:
        """Resolve a templated layer name for a given model configuration."""
        layers = model_config.get('layers', {})
        template = layers.get(layer_type)
        if isinstance(template, str):
            return template.format(layer=layer_index)

        if layer_type in {'attention', 'mlp'}:
            model_type = model_config.get('type')
            patterns = self._layer_patterns_cache or self.load_models_config().get('layer_patterns', {})
            if model_type in patterns:
                key = 'attention_patterns' if layer_type == 'attention' else 'mlp_patterns'
                options = patterns[model_type].get(key, [])
                if options:
                    return options[0].format(layer=layer_index)

        raise KeyError(f"Layer type '{layer_type}' not defined for model '{model_config.get('name', model_config)}'")

    def load_config(self, config_path: Union[str, Path]) -> 'NeuronMapConfig':
        """Load configuration from file using ConfigManager."""
        manager = ConfigManager()
        loaded_config = manager.load_config(str(config_path))

        # Update this instance with the loaded configuration
        for field_name, field_value in loaded_config.__dict__.items():
            setattr(self, field_name, field_value)

        return self

    # Dict-like access for test compatibility
    def __getitem__(self, key: str):
        """Dict-like access to config attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Config key '{key}' not found")

    def __setitem__(self, key: str, value):
        """Dict-like setting of config attributes."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Dict-like 'in' operator."""
        return hasattr(self, key)

    def get(self, key: str, default=None):
        """Dict-like get method."""
        return getattr(self, key, default)


class ConfigManager:
    """High-level configuration manager with YAML support and caching."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            if config_dir == "configs":
                self.config_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise ConfigurationError("Config directory not found")
        if not self.config_dir.is_dir():
            raise ConfigurationError("Config directory not found")

        self._loaded_files: List[Path] = []
        self._raw_config: Optional[Dict[str, Any]] = None
        self._cached_dict: Optional[Dict[str, Any]] = None
        self._config_model: Optional[NeuronMapConfig] = None
        self._config: Optional[NeuronMapConfig] = None
        self._validation_errors: List[Dict[str, Any]] = []
        self._environment = self._determine_initial_environment()

        self._models_config_cache: Optional[Dict[str, Any]] = None

    def _determine_initial_environment(self) -> str:
        env = os.getenv("NEURONMAP_CONFIG_ENV", "").strip().lower()
        return env or "default"

    @property
    def current_environment(self) -> str:
        """Return the active environment name."""
        return self._environment

    @property
    def config(self) -> Dict[str, Any]:
        """Dictionary view of the cached configuration."""
        return self.get_config()

    def load_config(self, config_path: Optional[str] = None) -> NeuronMapConfig:
        """Load configuration from disk and build a typed representation."""
        if config_path:
            config_path = Path(config_path)
            data = self._load_yaml_file(config_path)
            self._loaded_files = [config_path]
        else:
            data = self._load_config_files()

        env_overrides = self._load_environment_overrides()
        merged = self._merge_config_dict(data, env_overrides) if env_overrides else data

        self._raw_config = deepcopy(merged)
        self._cached_dict = self._raw_config

        try:
            self._config_model = NeuronMapConfig(config_dir=self.config_dir, **self._raw_config)
            self._config = self._config_model
            self._validation_errors = []
        except ValidationError as exc:
            logger.warning("Configuration validation produced warnings: %s", exc)
            self._validation_errors = exc.errors() if hasattr(exc, "errors") else [{"msg": str(exc)}]
            self._config_model = NeuronMapConfig(config_dir=self.config_dir)
            self._config = self._config_model

        # Clear any cached derived data so future access reflects the refreshed configuration
        self._models_config_cache = None

        return self._config_model

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        if self._cached_dict is None or self._config is None:
            self.load_config()
        return self._cached_dict  # type: ignore[return-value]

    def get_config_model(self) -> NeuronMapConfig:
        """Return the typed configuration instance."""
        if self._config is None:
            self.load_config()
        return self._config  # type: ignore[return-value]

    def get_default_config(self) -> NeuronMapConfig:
        """Convenience accessor used by legacy code paths."""
        return self.get_config_model()

    def _load_config_files(self) -> Dict[str, Any]:
        """Load all baseline YAML configuration files from the directory."""
        merged: Dict[str, Any] = {}
        self._loaded_files = []

        yaml_candidates = list(self.config_dir.glob("*.yml")) + list(self.config_dir.glob("*.yaml"))
        seen: set[Path] = set()
        yaml_files: List[Path] = []
        for file_path in yaml_candidates:
            if file_path not in seen:
                yaml_files.append(file_path)
                seen.add(file_path)

        yaml_files.sort(key=self._config_file_priority)
        for file_path in yaml_files:
            if file_path.name.startswith("environment_") or file_path.name.startswith("config."):
                # Environment-specific files are handled separately
                continue
            data = self._load_yaml_file(file_path)
            if data:
                merged = self._merge_config_dict(merged, data)
                self._loaded_files.append(file_path)

        return merged

    def _config_file_priority(self, file_path: Path) -> Tuple[int, str]:
        name = file_path.name.lower()
        base_prefixes = ("config", "models", "analysis", "data", "system", "default")
        priority = 0 if name.startswith(base_prefixes) else 1
        return (priority, name)

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
                if not isinstance(data, dict):
                    raise ValueError("Configuration file must contain a JSON/YAML object")
                return data
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read configuration file %s: %s", path, exc)
            raise ConfigurationError(f"Failed to load config from {path}: {exc}")

    def _resolve_environment_file(self, environment: str) -> Optional[Path]:
        if environment in ("", "default"):
            return None

        candidates = [
            self.config_dir / f"environment_{environment}.yaml",
            self.config_dir / f"environment_{environment}.yml",
            self.config_dir / f"config.{environment}.yaml",
            self.config_dir / f"config.{environment}.yml",
        ]

        aliases = {
            "dev": ["development"],
            "development": ["dev"],
            "test": ["testing"],
            "testing": ["test"],
            "production": ["prod"],
            "prod": ["production"],
        }

        for alias in aliases.get(environment, []):
            candidates.extend([
                self.config_dir / f"environment_{alias}.yaml",
                self.config_dir / f"environment_{alias}.yml",
                self.config_dir / f"config.{alias}.yaml",
                self.config_dir / f"config.{alias}.yml",
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_environment_overrides(self) -> Dict[str, Any]:
        env_file = self._resolve_environment_file(self._environment)
        if not env_file:
            return {}

        overrides = self._load_yaml_file(env_file)
        if overrides:
            self._loaded_files.append(env_file)
        return overrides

    def _merge_config_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config_dict(result[key], value)
            else:
                result[key] = value
        return result

    def set_environment(self, environment: Union[str, Environment]) -> None:
        normalized = self._normalize_environment(environment)
        if not normalized:
            raise ConfigurationError("Environment name cannot be empty")

        if normalized != "default" and not self._resolve_environment_file(normalized):
            raise ConfigurationError(f"Environment config file not found for '{environment}'")

        self._environment = normalized
        self._raw_config = None
        self._cached_dict = None
        self._config_model = None
        self._config = None
        self._validation_errors = []
        self._models_config_cache = None
        self.load_config()

    def _normalize_environment(self, environment: Union[str, Environment]) -> str:
        if isinstance(environment, Environment):
            return environment.value
        if not environment:
            raise ConfigurationError("Environment name cannot be empty")

        env = environment.strip().lower()
        aliases = {
            "dev": "dev",
            "development": "development",
            "test": "test",
            "testing": "testing",
            "prod": "production",
            "production": "production",
            "default": "default",
        }
        return aliases.get(env, env)

    def get_environment(self) -> str:
        return self._environment

    def validate_config(self) -> List[ConfigValidationError]:
        """Run lightweight validation checks returning structured errors."""
        errors: List[ConfigValidationError] = []
        if self._validation_errors:
            for err in self._validation_errors:
                loc = err.get("loc", []) if isinstance(err, dict) else []
                field = ".".join(str(part) for part in loc) if loc else "config"
                message = err.get("msg", str(err)) if isinstance(err, dict) else str(err)
                value = err.get("input") if isinstance(err, dict) else None
                errors.append(ConfigValidationError(field, message, value))

        config_model = self.get_config_model()
        raw_config = self.get_config()

        analysis_section = raw_config.get("analysis", {}) if isinstance(raw_config, dict) else {}
        if "batch_size" in analysis_section:
            batch_size = analysis_section.get("batch_size")
            if not isinstance(batch_size, int):
                errors.append(ConfigValidationError(
                    "analysis.batch_size",
                    "Batch size must be an integer",
                    batch_size,
                    "Provide a positive integer value",
                ))
            elif batch_size <= 0:
                errors.append(ConfigValidationError(
                    "analysis.batch_size",
                    "Batch size must be greater than zero",
                    batch_size,
                    "Use a positive integer",
                ))

        if "max_sequence_length" in analysis_section:
            max_seq = analysis_section.get("max_sequence_length")
            if not isinstance(max_seq, int):
                errors.append(ConfigValidationError(
                    "analysis.max_sequence_length",
                    "Max sequence length must be an integer",
                    max_seq,
                    "Provide a positive integer value",
                ))
            elif max_seq <= 0:
                errors.append(ConfigValidationError(
                    "analysis.max_sequence_length",
                    "Max sequence length must be greater than zero",
                    max_seq,
                    "Use a positive integer",
                ))

        if config_model.system.enable_gpu:
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():  # pragma: no cover - depends on hardware
                    errors.append(ConfigValidationError(
                        "system.enable_gpu",
                        "GPU enabled but CUDA not available",
                        True,
                        "Disable GPU or install CUDA-enabled PyTorch",
                    ))
            except ImportError:
                errors.append(ConfigValidationError(
                    "system.enable_gpu",
                    "GPU enabled but PyTorch is not installed",
                    True,
                    "Install torch with CUDA support",
                ))

        for dir_attr in ("temp_directory", "output_directory"):
            directory = Path(getattr(config_model.system, dir_attr))
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - filesystem dependent
                errors.append(ConfigValidationError(
                    f"system.{dir_attr}",
                    f"Cannot create directory: {exc}",
                    str(directory),
                ))

        return errors

    def validate_all_configs(self) -> Dict[str, Any]:
        """Aggregate validation information for test assertions."""
        errors = self.validate_config()
        if errors:
            return {
                "status": "error",
                "validation_results": [
                    {"name": err.field, "valid": False, "errors": [err.message]} for err in errors
                ],
            }

        return {
            "status": "success",
            "validation_results": [
                {"name": "schema", "valid": True, "errors": []}
            ],
        }

    def load_models_config(self) -> Dict[str, Dict[str, Any]]:
        """Return mapping of model names to configuration dictionaries."""
        if self._models_config_cache is None:
            models: Dict[str, Dict[str, Any]] = {}
            config_model = self.get_config_model()

            # Prefer explicit models stored on the config model
            if config_model.models:
                for name, model in config_model.models.items():
                    if hasattr(model, "model_dump"):
                        models[name] = model.model_dump()
                    else:
                        models[name] = deepcopy(model)  # type: ignore[arg-type]

            # Fallback to dedicated YAML file
            try:
                raw_models = config_model.load_models_config()
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Failed to load models.yaml: %s", exc)
                raw_models = {}

            if isinstance(raw_models, dict):
                models_section = raw_models.get("models", raw_models)
                if isinstance(models_section, dict):
                    for name, model_data in models_section.items():
                        if isinstance(model_data, dict):
                            models.setdefault(name, deepcopy(model_data))

            self._models_config_cache = deepcopy(models)

        return deepcopy(self._models_config_cache)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        models = self.load_models_config()
        if model_name in models:
            return deepcopy(models[model_name])
        raise KeyError(f"Model '{model_name}' not found in configuration")

    def get_analysis_config(self) -> AnalysisConfig:
        """Return a deep copy of the analysis configuration."""
        analysis = self.get_config_model().analysis
        if hasattr(analysis, "model_copy"):
            return analysis.model_copy(deep=True)  # type: ignore[attr-defined]
        return deepcopy(analysis)

    def get_visualization_config(self) -> VisualizationConfig:
        """Return a deep copy of the visualization configuration."""
        visualization = self.get_config_model().visualization
        if hasattr(visualization, "model_copy"):
            return visualization.model_copy(deep=True)  # type: ignore[attr-defined]
        return deepcopy(visualization)

    def get_environment_config(self) -> Dict[str, Any]:
        """Return merged environment/system configuration as a plain dictionary."""
        config_model = self.get_config_model()
        environment_data: Dict[str, Any] = {}

        if isinstance(config_model.environment, dict):
            environment_data.update(deepcopy(config_model.environment))

        system_config = config_model.system
        if hasattr(system_config, "model_dump"):
            environment_data.setdefault("system", system_config.model_dump())
        else:
            environment_data.setdefault("system", deepcopy(system_config))

        environment_data.setdefault("environment", self.current_environment)
        return environment_data

    def resolve_layer_name(self, model_name: str, layer_type: str, layer_index: int) -> str:
        """Resolve a templated layer name for a specific model."""
        model_config = self.get_model_config(model_name)
        return self.get_config_model().resolve_layer_name(model_config, layer_type, layer_index)

    def validate_hardware_compatibility(self) -> List[str]:
        """Check basic hardware compatibility against the active configuration."""
        issues: List[str] = []
        analysis_config = self.get_analysis_config()
        env_config = self.get_environment_config()
        system_config = self.get_config_model().system

        device_pref = getattr(analysis_config, "device", "auto")
        if hasattr(device_pref, "value"):
            device_pref = device_pref.value  # type: ignore[attr-defined]
        if not isinstance(device_pref, str):
            device_pref = str(device_pref)

        try:  # pragma: no cover - torch optional dependency
            import torch  # type: ignore
        except Exception:  # noqa: BLE001
            torch = None  # type: ignore

        cuda_requested = device_pref.startswith("cuda") or device_pref == "cuda" or device_pref == "auto"
        if cuda_requested:
            if torch is None or not torch.cuda.is_available():
                if device_pref != "auto":
                    issues.append("CUDA device requested but CUDA not available")
            else:
                try:
                    gpu_props = torch.cuda.get_device_properties(0)
                    gpu_total = gpu_props.total_memory / (1024 ** 3)
                    memory_opt = getattr(analysis_config, "memory_optimization", {}) or {}
                    if hasattr(memory_opt, "model_dump"):
                        memory_opt = memory_opt.model_dump()  # type: ignore[attr-defined]
                    required_gpu = None
                    if isinstance(memory_opt, dict):
                        required_gpu = memory_opt.get("max_memory_usage_gb") or memory_opt.get("gpu_memory_gb")
                    if isinstance(required_gpu, (int, float)) and required_gpu > gpu_total:
                        issues.append(
                            f"Required GPU memory ({required_gpu}GB) exceeds available ({gpu_total:.1f}GB)"
                        )
                except Exception as exc:  # pragma: no cover - diagnostic only
                    issues.append(f"Could not determine GPU memory: {exc}")

        max_workers = env_config.get("max_workers")
        if max_workers is None:
            system_section = env_config.get("system", {})
            if isinstance(system_section, dict):
                max_workers = system_section.get("max_workers")
        if max_workers is None and hasattr(system_config, "max_concurrent_jobs"):
            max_workers = system_config.max_concurrent_jobs

        try:
            cpu_count = os.cpu_count() or 1
            if isinstance(max_workers, int) and max_workers > cpu_count:
                issues.append(
                    f"Configured max workers ({max_workers}) exceeds available CPU cores ({cpu_count})"
                )
        except Exception:  # pragma: no cover - defensive
            pass

        memory_limit = env_config.get("memory_limit_gb")
        if memory_limit is None:
            system_section = env_config.get("system", {})
            if isinstance(system_section, dict):
                memory_limit = system_section.get("memory_limit_gb")

        memory_opt_general = getattr(analysis_config, "memory_optimization", {}) or {}
        if hasattr(memory_opt_general, "model_dump"):
            memory_opt_general = memory_opt_general.model_dump()  # type: ignore[attr-defined]
        required_memory = None
        if isinstance(memory_opt_general, dict):
            required_memory = memory_opt_general.get("max_memory_usage_gb") or memory_opt_general.get("required_memory_gb")

        if isinstance(required_memory, (int, float)) and isinstance(memory_limit, (int, float)):
            if required_memory > memory_limit:
                issues.append(
                    f"Required memory ({required_memory}GB) exceeds configured limit ({memory_limit}GB)"
                )

        return issues

    def perform_startup_validation(self) -> Tuple[bool, List[str]]:
        """Run comprehensive startup validation returning (is_valid, issues)."""
        issues: List[str] = []
        validation_summary = self.validate_all_configs()
        if validation_summary.get("status") != "success":
            for result in validation_summary.get("validation_results", []):
                if not isinstance(result, dict):
                    continue
                if result.get("valid", True):
                    continue
                errors = result.get("errors", [])
                if isinstance(errors, list):
                    issues.extend(str(err) for err in errors)
                elif errors:
                    issues.append(str(errors))

        issues.extend(self.validate_hardware_compatibility())

        data_config = self.get_config_model().data
        directory_attrs = [
            "data_dir",
            "raw_dir",
            "processed_dir",
            "outputs_dir",
            "cache_dir",
        ]
        for attr in directory_attrs:
            path_value = getattr(data_config, attr, None)
            if not path_value:
                continue
            try:
                Path(path_value).mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - filesystem dependent
                issues.append(f"Directory setup failed for {path_value}: {exc}")

        try:
            output_directory = Path(self.get_config_model().system.output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            test_file = output_directory / "config_write_test.tmp"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
        except Exception as exc:  # pragma: no cover - filesystem dependent
            issues.append(f"Write permission check failed: {exc}")

        return (len(issues) == 0, issues)

    def setup_logging(self) -> None:
        """Configure logging according to environment/system settings."""
        env_config = self.get_environment_config()
        level_name = env_config.get("log_level")
        if level_name is None:
            system_section = env_config.get("system", {})
            if isinstance(system_section, dict):
                level_name = system_section.get("log_level")
        if not isinstance(level_name, str):
            level_name = "INFO"

        log_level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(level=log_level)

        log_to_file = env_config.get("log_to_file")
        if log_to_file is None:
            system_section = env_config.get("system", {})
            if isinstance(system_section, dict):
                log_to_file = system_section.get("log_to_file")

        if log_to_file:
            log_path = env_config.get("log_file_path") or env_config.get("system", {}).get("log_file_path")
            if not log_path:
                log_path = "logs/neuronmap.log"

            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            from logging.handlers import RotatingFileHandler

            max_bytes = env_config.get("max_log_size_mb") or env_config.get("system", {}).get("max_log_size_mb")
            if isinstance(max_bytes, (int, float)):
                max_bytes = int(max_bytes * 1024 * 1024)
            else:
                max_bytes = 10 * 1024 * 1024

            backup_count = env_config.get("backup_count") or env_config.get("system", {}).get("backup_count")
            if not isinstance(backup_count, int):
                backup_count = 5

            handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)

    def get_device(self, device_config: str | None = None):
        """Return a torch.device inferred from configuration preferences."""
        device_pref = device_config or getattr(self.get_analysis_config(), "device", "auto")
        if hasattr(device_pref, "value"):
            device_pref = device_pref.value  # type: ignore[attr-defined]
        if not isinstance(device_pref, str):
            device_pref = str(device_pref)

        try:  # pragma: no cover - torch optional dependency
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyTorch is required to determine devices") from exc

        if device_pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_pref.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device(device_pref)
        if device_pref == "cpu":
            return torch.device("cpu")

        raise ValueError(f"Invalid device configuration: {device_pref}")

    def create_output_paths(self) -> Dict[str, Path]:
        """Ensure standard output paths exist and return them."""
        data_config = self.get_config_model().data
        paths: Dict[str, Path] = {}

        directory_fields = {
            "data_dir": data_config.data_dir,
            "raw_dir": data_config.raw_dir,
            "processed_dir": data_config.processed_dir,
            "outputs_dir": data_config.outputs_dir,
        }

        for key, value in directory_fields.items():
            directory = Path(value)
            directory.mkdir(parents=True, exist_ok=True)
            paths[key] = directory

        return paths

    def switch_environment(self, environment: Union[str, Environment]) -> None:
        """Compatibility wrapper for environment switching."""
        self.set_environment(environment)

    def is_valid(self) -> bool:
        """Return True if configuration validates successfully."""
        return self.validate_all_configs().get("status") == "success"

    def get_validation_errors(self) -> List[str]:
        """Return human-readable validation error messages."""
        errors = []
        for error in self.validate_config():
            errors.append(f"{error.field}: {error.message}")
        return errors


    def add_model_config(self, model_name: str, model_config: Dict[str, Any]):
        config_dict = self.get_config()
        models = config_dict.setdefault("models", {})
        models[model_name] = deepcopy(model_config)
        self._raw_config = config_dict
        self._cached_dict = config_dict
        self._config_model = None
        self._config = None

    def save_config(self, config_path: Optional[Path] = None) -> Path:
        if self._raw_config is None:
            raise ValueError("No configuration to save")

        target_path = config_path or (self.config_dir / "config.yaml")
        with open(target_path, "w", encoding="utf-8") as handle:
            yaml.dump(self._raw_config, handle, default_flow_style=False, sort_keys=False, indent=2)
        logger.info("Configuration saved to %s", target_path)
        return target_path

    def create_default_configs(self):
        """Generate default YAML configuration files in the config directory."""
        default_config = NeuronMapConfig(config_dir=self.config_dir)
        self._config_model = default_config
        self._config = default_config
        self._raw_config = default_config.model_dump()
        self._cached_dict = self._raw_config
        self._validation_errors = []
        self.save_config(self.config_dir / "config.yaml")

    def validate_model_compatibility(self, model_name: str) -> List[ConfigValidationError]:
        errors: List[ConfigValidationError] = []
        try:
            model_config = self.get_model_config(model_name)
        except KeyError:
            errors.append(ConfigValidationError(
                f"models.{model_name}",
                "Model configuration not found",
                None,
                "Add a model entry to models.yaml",
            ))
            return errors

        system_config = self.get_config_model().system
        requires_gpu = model_config.get("requires_gpu", False)
        if requires_gpu and not system_config.enable_gpu:
            errors.append(ConfigValidationError(
                f"models.{model_name}.requires_gpu",
                "Model requires GPU but GPU usage is disabled in system configuration",
                True,
                "Enable GPU in system configuration",
            ))

        return errors

    def get_experiment_config(self, experiment_name: str = "default") -> Dict[str, Any]:
        experiments = NeuronMapConfig(self.config_dir).load_experiments_config()

        if experiment_name in experiments:
            return deepcopy(experiments[experiment_name])

        nested = experiments.get("experiments", {})
        if experiment_name in nested:
            return deepcopy(nested[experiment_name])

        raise KeyError(f"Experiment '{experiment_name}' not found")

    def get_hardware_info(self) -> Dict[str, Any]:
        """Collect lightweight hardware telemetry for diagnostics."""
        import platform

        info = {
            "cpu": platform.processor(),
            "cores": os.cpu_count() or 1,
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
        }

        try:
            import psutil  # type: ignore

            info["total_memory_gb"] = int(psutil.virtual_memory().total / (1024 ** 3))
        except ImportError:
            info["total_memory_gb"] = None

        try:
            import torch  # type: ignore

            info["cuda_available"] = bool(torch.cuda.is_available())
        except ImportError:
            info["cuda_available"] = False

        return info

# Global config manager instance for convenience
_global_config_manager: Optional[ConfigManager] = None

def get_config(config_path: Optional[str] = None) -> NeuronMapConfig:
    """Get global configuration instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        NeuronMapConfig instance
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()

    if config_path is None:
        return _global_config_manager.get_config()
    else:
        return _global_config_manager.load_config(config_path)

def get_config_manager() -> ConfigManager:
    """Get global config manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def reset_config_manager() -> None:
    """Reset the global config manager instance."""
    global _global_config_manager
    _global_config_manager = None


def setup_global_config(
    config_dir: Optional[str] = None,
    environment: Optional[Union[str, Environment]] = None,
) -> ConfigManager:
    """Initialize the global configuration manager with validation and logging.

    Args:
        config_dir: Optional path to the configuration directory.
        environment: Optional environment name to activate.

    Returns:
        Configured ConfigManager instance.

    Raises:
        RuntimeError: If startup validation fails.
    """

    reset_config_manager()

    manager = ConfigManager(config_dir=config_dir or "configs")
    if environment:
        manager.set_environment(environment)

    is_valid, issues = manager.perform_startup_validation()
    if not is_valid:
        raise RuntimeError(
            "Configuration validation failed: " + "; ".join(str(issue) for issue in issues)
        )

    manager.setup_logging()

    global _global_config_manager
    _global_config_manager = manager
    return manager
