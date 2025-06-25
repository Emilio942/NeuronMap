"""
Enhanced Configuration Management System for NeuronMap
====================================================

This module provides a robust, validated configuration system using Pydantic
for type checking and YAML/JSON support for flexible configuration files.
"""

try:
    from pydantic import BaseModel, field_validator, ConfigDict, Field, ValidationError
except ImportError:
    from pydantic import BaseModel, validator as field_validator, ValidationError
    # Fallback für ältere Pydantic Versionen
    ConfigDict = None
    Field = lambda default=None, **kwargs: default

import yaml
import json
import logging
from pathlib import Path
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


class NeuronMapConfig(BaseModel):
    """Main configuration class that combines all sub-configurations"""
    if ConfigDict:
        model_config = ConfigDict(extra='allow')  # Allow extra fields for flexibility

    # Version and metadata
    version: str = Field(default="1.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now, description="Configuration creation time")

    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

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

    def get_experiment_config(self, experiment_name: str = "default") -> Dict[str, Any]:
        """Get experiment configuration by name."""
        return {
            'name': experiment_name,
            'description': getattr(self.experiment, 'description', f'Experiment: {experiment_name}') if self.experiment else f'Default experiment: {experiment_name}',
            'model': self.model.name,
            'data': {
                'input_file': self.data.input_file,
                'output_file': self.data.output_file,
                'batch_size': self.data.batch_size
            },
            'analysis': {
                'target_layers': self.model.target_layers,
                'max_length': self.model.max_length
            }
        }

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
    """Enhanced configuration manager with validation and multiple format support"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[NeuronMapConfig] = None
        self._loaded_files: List[Path] = []

    def load_config(self, config_path: Optional[str] = None) -> NeuronMapConfig:
        """Load configuration from file or create default"""
        if config_path is None:
            config_path = self._find_config_file()
        else:
            config_path = Path(config_path)

        if config_path.exists():
            self._config = self._load_from_file(config_path)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self._config = NeuronMapConfig()

        # Load environment-specific overrides
        self._load_environment_overrides()

        # Validate configuration
        validation_errors = self.validate_config()
        if validation_errors:
            logger.warning(f"Configuration validation found {len(validation_errors)} issues")
            for error in validation_errors:
                logger.warning(f"  {error.field}: {error.message}")
                if error.suggestion:
                    logger.info(f"    Suggestion: {error.suggestion}")

        return self._config

    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations"""
        possible_files = [
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
            self.config_dir / "neuronmap.yaml",
            Path("config.yaml"),
            Path("neuronmap.yaml")
        ]

        for config_file in possible_files:
            if config_file.exists():
                return config_file

        return self.config_dir / "config.yaml"

    def _load_from_file(self, config_path: Path) -> NeuronMapConfig:
        """Load configuration from YAML or JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")

            self._loaded_files.append(config_path)
            logger.info(f"Loaded configuration from {config_path}")

            return NeuronMapConfig(**data)

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise ValueError(f"Failed to load configuration: {e}")

    def _load_environment_overrides(self):
        """Load environment-specific configuration overrides"""
        if not self._config:
            return

        env = self._config.system.environment
        env_file = self.config_dir / f"config.{env.value}.yaml"

        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    env_data = yaml.safe_load(f)

                # Merge environment overrides
                merged_data = self._merge_config_dict(self._config.dict(), env_data)
                self._config = NeuronMapConfig(**merged_data)

                self._loaded_files.append(env_file)
                logger.info(f"Applied environment overrides from {env_file}")

            except Exception as e:
                logger.warning(f"Error loading environment config {env_file}: {e}")

    def _merge_config_dict(self, base: dict, override: dict) -> dict:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config_dict(result[key], value)
            else:
                result[key] = value
        return result

    def validate_config(self) -> List[ConfigValidationError]:
        """Validate current configuration and return errors"""
        errors = []

        if not self._config:
            errors.append(ConfigValidationError("config", "No configuration loaded", None))
            return errors

        # Validate GPU availability if required
        if self._config.system.enable_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append(ConfigValidationError(
                        "system.enable_gpu",
                        "GPU enabled but CUDA not available",
                        True,
                        "Set enable_gpu to false or install CUDA support"
                    ))
            except ImportError:
                errors.append(ConfigValidationError(
                    "system.enable_gpu",
                    "GPU enabled but PyTorch not available",
                    True,
                    "Install PyTorch with CUDA support"
                ))

        # Validate directories exist or can be created
        for dir_attr in ['temp_directory', 'output_directory']:
            try:
                directory = Path(getattr(self._config.system, dir_attr))
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(ConfigValidationError(
                    f"system.{dir_attr}",
                    f"Cannot create directory: {e}",
                    getattr(self._config.system, dir_attr)
                ))

        # Validate model configurations
        for model_name, model_config in self._config.models.items():
            if model_config.attention_heads and model_config.hidden_size:
                if model_config.hidden_size % model_config.attention_heads != 0:
                    errors.append(ConfigValidationError(
                        f"models.{model_name}.attention_heads",
                        f"Hidden size {model_config.hidden_size} not divisible by attention heads {model_config.attention_heads}",
                        model_config.attention_heads,
                        f"Change attention_heads to {model_config.hidden_size // (model_config.hidden_size // model_config.attention_heads)}"
                    ))

        return errors

    def get_config(self) -> NeuronMapConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        config = self.get_config()
        return config.models.get(model_name, config.model)

    def add_model_config(self, model_name: str, model_config: ModelConfig):
        """Add or update model configuration"""
        config = self.get_config()
        config.models[model_name] = model_config

    def save_config(self, config_path: Optional[Path] = None) -> Path:
        """Save current configuration to file"""
        if not self._config:
            raise ValueError("No configuration to save")

        if config_path is None:
            config_path = self.config_dir / "config.yaml"

        try:
            # Convert to dict and clean up
            config_dict = self._config.dict()
            config_dict.pop('created_at', None)  # Remove timestamp for cleaner file

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)

            logger.info(f"Configuration saved to {config_path}")
            return config_path

        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise ValueError(f"Failed to save configuration: {e}")

    def create_default_configs(self):
        """Create default configuration files"""
        # Main configuration with common models
        default_config = NeuronMapConfig()

        # Add common model configurations
        default_config.models["gpt2"] = ModelConfig(
            name="gpt2",
            layer_count=12,
            hidden_size=768,
            attention_heads=12,
            max_length=1024,
            target_layers=["transformer.h.5.mlp.c_proj", "transformer.h.5.attn"]
        )

        default_config.models["bert-base-uncased"] = ModelConfig(
            name="bert-base-uncased",
            layer_count=12,
            hidden_size=768,
            attention_heads=12,
            max_length=512,
            target_layers=["encoder.layer.5.attention.self", "encoder.layer.5.output"]
        )

        default_config.models["distilgpt2"] = ModelConfig(
            name="distilgpt2",
            layer_count=6,
            hidden_size=768,
            attention_heads=12,
            max_length=1024,
            target_layers=["transformer.h.5.mlp.c_proj"]
        )

        self._config = default_config

        # Save main config
        self.save_config(self.config_dir / "config.yaml")

        # Create environment-specific configs
        for env in Environment:
            env_config = {
                "system": {
                    "environment": env.value,
                    "log_level": "DEBUG" if env == Environment.DEVELOPMENT else "INFO"
                }
            }

            if env == Environment.PRODUCTION:
                env_config["web"] = {
                    "debug": False,
                    "secret_key": "CHANGE-THIS-IN-PRODUCTION"
                }
                env_config["system"]["log_level"] = "WARNING"

            env_file = self.config_dir / f"config.{env.value}.yaml"
            with open(env_file, 'w', encoding='utf-8') as f:
                yaml.dump(env_config, f, default_flow_style=False, indent=2)

        logger.info("Created default configuration files")

    def validate_model_compatibility(self, model_name: str) -> List[ConfigValidationError]:
        """Validate model compatibility with current system"""
        errors = []
        model_config = self.get_model_config(model_name)

        if not model_config:
            errors.append(ConfigValidationError(
                f"models.{model_name}",
                "Model configuration not found",
                None,
                "Add model configuration or use a configured model"
            ))
            return errors

        # Check GPU requirements
        if model_config.requires_gpu and not self._config.system.enable_gpu:
            errors.append(ConfigValidationError(
                f"models.{model_name}.requires_gpu",
                "Model requires GPU but system GPU is disabled",
                True,
                "Enable GPU in system configuration or use CPU-compatible model"
            ))

        # Check memory requirements
        if model_config.max_memory_gb > 16:  # Warning for large models
            logger.warning(f"Model {model_name} requires {model_config.max_memory_gb}GB memory")

        return errors

    def get_experiment_config(self, experiment_name: str = "default") -> Dict[str, Any]:
        """Get experiment configuration by name."""
        return {
            'name': experiment_name,
            'description': getattr(self.experiment, 'description', f'Experiment: {experiment_name}') if self.experiment else f'Default experiment: {experiment_name}',
            'model': self.model.name,
            'data': {
                'input_file': self.data.input_file,
                'output_file': self.data.output_file,
                'batch_size': self.data.batch_size
            },
            'analysis': {
                'target_layers': self.model.target_layers,
                'max_length': self.model.max_length
            }
        }

    def set_environment(self, environment: Union[str, Environment]) -> None:
        """Set the runtime environment and reload configuration with environment-specific overrides"""
        if isinstance(environment, str):
            try:
                # Handle test compatibility - accept shortened names
                env_mapping = {
                    'dev': 'development',
                    'test': 'testing',
                    'prod': 'production'
                }

                normalized_env = env_mapping.get(environment.lower(), environment.lower())
                environment = Environment(normalized_env)
            except ValueError:
                raise ValueError(f"Invalid environment: {environment}. Must be one of {list(Environment)}")

        if not self._config:
            self._config = NeuronMapConfig()

        # Update environment in system config
        self._config.system.environment = environment

        # Reload environment-specific overrides
        self._load_environment_overrides()

        logger.info(f"Environment set to: {environment.value}")

    def get_environment(self) -> Environment:
        """Get the current runtime environment"""
        if not self._config:
            return Environment.DEVELOPMENT
        return self._config.system.environment

    @property
    def current_environment(self) -> str:
        """Get current environment for test compatibility."""
        return self.get_environment()

    def validate_all_configs(self) -> Dict[str, Any]:
        """Validate all configuration files - test compatibility method."""
        validation_errors = self.validate_config()
        return {
            'valid': len(validation_errors) == 0,
            'errors': validation_errors,
            'total_files': len(self._loaded_files)
        }

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for test compatibility."""
        import platform
        import psutil
        try:
            return {
                'cpu': platform.processor(),
                'cores': psutil.cpu_count(),
                'memory': psutil.virtual_memory().total,
                'platform': platform.system(),
                'architecture': platform.architecture()[0]
            }
        except ImportError:
            return {
                'cpu': 'unknown',
                'cores': 1,
                'memory': 1000000000,  # 1GB fallback
                'platform': platform.system(),
                'architecture': 'unknown'
            }

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
