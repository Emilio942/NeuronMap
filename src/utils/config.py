"""Configuration management utilities for NeuronMap."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
from pydantic import BaseModel, Field, field_validator, ValidationError
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# Pydantic models for configuration validation
class EnvironmentType(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class DeviceType(str, Enum):
    """Device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ModelType(str, Enum):
    """Supported model types."""
    GPT = "gpt"
    BERT = "bert"
    T5 = "t5"
    LLAMA = "llama"


class LayerConfig(BaseModel):
    """Configuration for model layers."""
    attention: str
    mlp: str
    total_layers: int = Field(gt=0, le=200)


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    name: str
    type: ModelType
    layers: LayerConfig
    hidden_size: Optional[int] = Field(default=None, gt=0)
    attention_heads: Optional[int] = Field(default=None, gt=0)
    max_context_length: Optional[int] = Field(default=None, gt=0)
    supported_tasks: Optional[List[str]] = Field(default_factory=list)

    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class MemoryOptimizationConfig(BaseModel):
    """Memory optimization configuration."""
    use_memory_efficient_attention: bool = True
    max_memory_usage_gb: float = Field(gt=0, le=512)
    offload_to_cpu: bool = False
    clear_cache_between_batches: bool = True


class StatisticsConfig(BaseModel):
    """Statistics analysis configuration."""
    compute_mean: bool = True
    compute_std: bool = True
    compute_skewness: bool = True
    compute_kurtosis: bool = True
    compute_percentiles: List[int] = Field(default_factory=lambda: [5, 25, 50, 75, 95])
    correlation_methods: List[str] = Field(default_factory=lambda: ["pearson", "spearman"])


class ClusteringConfig(BaseModel):
    """Clustering analysis configuration."""
    methods: List[str] = Field(default_factory=lambda: ["kmeans", "hierarchical", "dbscan"])
    n_clusters_range: List[int] = Field(default_factory=lambda: [2, 3, 4, 5, 8, 10])
    dimensionality_reduction: str = "pca"
    max_samples_for_clustering: int = Field(gt=0, default=10000)


class AttentionConfig(BaseModel):
    """Attention analysis configuration."""
    analyze_attention_patterns: bool = True
    extract_attention_weights: bool = True
    compute_attention_entropy: bool = True
    head_importance_analysis: bool = True
    circuit_discovery: bool = True


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    num_workers: int = Field(ge=1, le=32, default=4)
    timeout_seconds: int = Field(gt=0, default=300)
    retry_attempts: int = Field(ge=0, le=10, default=3)
    checkpoint_frequency: int = Field(gt=0, default=100)


class AnalysisConfig(BaseModel):
    """Main analysis configuration."""
    batch_size: int = Field(gt=0, le=1024, default=32)
    max_sequence_length: int = Field(gt=0, le=8192, default=512)
    device: DeviceType = DeviceType.AUTO
    precision: str = Field(default="float32", pattern="^(float16|float32|float64)$")
    enable_gradient_checkpointing: bool = False
    cache_activations: bool = True
    memory_optimization: MemoryOptimizationConfig = Field(default_factory=MemoryOptimizationConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


class StyleConfig(BaseModel):
    """Style configuration for visualizations."""
    theme: str = Field(default="modern", pattern="^(modern|classic|minimal)$")
    font_family: str = "Arial"
    font_size: int = Field(ge=8, le=24, default=12)
    title_font_size: int = Field(ge=10, le=32, default=14)
    axis_label_font_size: int = Field(ge=8, le=20, default=11)
    legend_font_size: int = Field(ge=8, le=16, default=10)
    grid_alpha: float = Field(ge=0.0, le=1.0, default=0.3)
    line_width: float = Field(gt=0.0, le=10.0, default=2.0)


class ColorSchemesConfig(BaseModel):
    """Color schemes configuration."""
    categorical: str = "Set1"
    sequential: str = "viridis"
    diverging: str = "RdBu_r"
    attention: str = "Blues"
    heatmap: str = "coolwarm"


class VisualizationConfig(BaseModel):
    """Visualization configuration."""
    figure_width: int = Field(ge=4, le=30, default=12)
    figure_height: int = Field(ge=4, le=30, default=8)
    dpi: int = Field(ge=72, le=600, default=300)
    color_scheme: str = "viridis"
    interactive: bool = True
    export_format: str = Field(default="png", pattern="^(png|svg|pdf|html)$")
    style: StyleConfig = Field(default_factory=StyleConfig)
    color_schemes: ColorSchemesConfig = Field(default_factory=ColorSchemesConfig)


class MonitoringConfig(BaseModel):
    """System monitoring configuration."""
    enable_system_monitoring: bool = True
    monitor_gpu_usage: bool = True
    monitor_memory_usage: bool = True
    monitor_cpu_usage: bool = True
    monitoring_interval_seconds: int = Field(ge=1, le=300, default=10)
    alert_threshold_memory_percent: int = Field(ge=50, le=100, default=90)


class EnvironmentConfig(BaseModel):
    """Environment configuration."""
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    enable_debugging: bool = True
    log_to_file: bool = True
    log_file_path: str = "logs/neuronmap.log"
    log_rotation: bool = True
    max_log_size_mb: int = Field(gt=0, le=1000, default=100)
    backup_count: int = Field(ge=1, le=20, default=5)
    max_workers: int = Field(ge=1, le=32, default=4)
    timeout_seconds: int = Field(gt=0, default=300)
    memory_limit_gb: int = Field(gt=0, le=1024, default=32)
    cpu_limit_percent: int = Field(ge=10, le=100, default=80)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


class ConfigManager:
    """Enhanced configuration manager with validation and environment support."""

    def __init__(self, config_dir: Optional[str] = None, environment: str = "development"):
        """Initialize configuration manager.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
            environment: Environment type (development, testing, production).
        """
        if config_dir is None:
            # Default to configs directory relative to this file
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        self.environment = environment
        self._logger = logging.getLogger(__name__)

        # Cache for loaded configurations
        self._models_config = None
        self._analysis_config = None
        self._visualization_config = None
        self._environment_config = None
        self._validation_errors = []

    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load and parse a YAML configuration file.

        Args:
            filename: Name of the configuration file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If configuration file not found.
            yaml.YAMLError: If YAML parsing fails.
        """
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            self._logger.error(f"Failed to parse YAML file {config_path}: {e}")
            raise

    def load_models_config(self) -> Dict[str, ModelConfig]:
        """Load and validate model configurations.

        Returns:
            Dictionary of validated model configurations.
        """
        if self._models_config is None:
            raw_config = self._load_yaml_file("models.yaml")
            self._models_config = {}

            for model_name, model_data in raw_config.get('models', {}).items():
                try:
                    self._models_config[model_name] = ModelConfig(**model_data)
                except ValidationError as e:
                    error_msg = f"Invalid model config for '{model_name}': {e}"
                    self._logger.error(error_msg)
                    self._validation_errors.append(error_msg)

        return self._models_config

    def load_analysis_config(self) -> AnalysisConfig:
        """Load and validate analysis configuration.

        Returns:
            Validated analysis configuration.
        """
        if self._analysis_config is None:
            raw_config = self._load_yaml_file("analysis.yaml")
            try:
                self._analysis_config = AnalysisConfig(**raw_config.get('analysis', {}))
            except ValidationError as e:
                error_msg = f"Invalid analysis config: {e}"
                self._logger.error(error_msg)
                self._validation_errors.append(error_msg)
                # Use default config on validation failure
                self._analysis_config = AnalysisConfig()

        return self._analysis_config

    def load_visualization_config(self) -> VisualizationConfig:
        """Load and validate visualization configuration.

        Returns:
            Validated visualization configuration.
        """
        if self._visualization_config is None:
            raw_config = self._load_yaml_file("visualization.yaml")
            try:
                self._visualization_config = VisualizationConfig(**raw_config.get('visualization', {}))
            except ValidationError as e:
                error_msg = f"Invalid visualization config: {e}"
                self._logger.error(error_msg)
                self._validation_errors.append(error_msg)
                # Use default config on validation failure
                self._visualization_config = VisualizationConfig()

        return self._visualization_config

    def load_environment_config(self) -> EnvironmentConfig:
        """Load and validate environment configuration with inheritance.

        Returns:
            Validated environment configuration.
        """
        if self._environment_config is None:
            raw_config = self._load_environment_specific_config("environment.yaml")
            try:
                self._environment_config = EnvironmentConfig(**raw_config.get('environment', {}))
            except ValidationError as e:
                error_msg = f"Invalid environment config: {e}"
                self._logger.error(error_msg)
                self._validation_errors.append(error_msg)
                # Use default config on validation failure
                self._environment_config = EnvironmentConfig()

        return self._environment_config

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model configuration to retrieve.

        Returns:
            Validated model configuration.

        Raises:
            KeyError: If model configuration not found.
            ValidationError: If model configuration is invalid.
        """
        models = self.load_models_config()
        if model_name not in models:
            available_models = list(models.keys())
            raise KeyError(f"Model configuration '{model_name}' not found. "
                         f"Available models: {available_models}")
        return models[model_name]

    def get_analysis_config(self) -> AnalysisConfig:
        """Get validated analysis configuration.

        Returns:
            Validated analysis configuration.
        """
        return self.load_analysis_config()

    def get_visualization_config(self) -> VisualizationConfig:
        """Get validated visualization configuration.

        Returns:
            Validated visualization configuration.
        """
        return self.load_visualization_config()

    def get_environment_config(self) -> EnvironmentConfig:
        """Get validated environment configuration.

        Returns:
            Validated environment configuration.
        """
        return self.load_environment_config()

    def validate_all_configs(self) -> List[str]:
        """Validate all configuration files.

        Returns:
            List of validation error messages. Empty if all configs are valid.
        """
        self._validation_errors = []

        # Load all configurations to trigger validation
        try:
            self.load_models_config()
            self.load_analysis_config()
            self.load_visualization_config()
            self.load_environment_config()
        except Exception as e:
            self._validation_errors.append(f"Critical configuration error: {e}")

        return self._validation_errors

    def is_valid(self) -> bool:
        """Check if all configurations are valid.

        Returns:
            True if all configurations are valid, False otherwise.
        """
        errors = self.validate_all_configs()
        return len(errors) == 0

    def get_validation_errors(self) -> List[str]:
        """Get current validation errors.

        Returns:
            List of validation error messages.
        """
        return self._validation_errors.copy()

    def switch_environment(self, environment: str) -> None:
        """Switch to a different environment configuration.

        Args:
            environment: Environment name (development, testing, production).
        """
        if environment not in ["development", "testing", "production"]:
            raise ValueError(f"Invalid environment: {environment}. "
                           f"Must be one of: development, testing, production")

        self.environment = environment
        # Clear cached configurations to force reload
        self._environment_config = None
        self._logger.info(f"Switched to environment: {environment}")

    def validate_hardware_compatibility(self) -> List[str]:
        """Validate hardware compatibility with current configuration.

        Returns:
            List of hardware compatibility warnings/errors.
        """
        compatibility_issues = []

        try:
            analysis_config = self.get_analysis_config()
            env_config = self.get_environment_config()

            # Check CUDA availability if requested
            if analysis_config.device == DeviceType.CUDA:
                if not torch.cuda.is_available():
                    compatibility_issues.append("CUDA device requested but CUDA not available")
                elif torch.cuda.device_count() == 0:
                    compatibility_issues.append("CUDA available but no CUDA devices found")

            # Check memory requirements
            memory_limit = env_config.memory_limit_gb
            if hasattr(analysis_config, 'memory_optimization'):
                required_memory = analysis_config.memory_optimization.max_memory_usage_gb
                if required_memory > memory_limit:
                    compatibility_issues.append(
                        f"Required memory ({required_memory}GB) exceeds limit ({memory_limit}GB)"
                    )

            # Check GPU memory if using CUDA
            if analysis_config.device in [DeviceType.CUDA, DeviceType.AUTO] and torch.cuda.is_available():
                try:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    required_gpu_memory = getattr(analysis_config.memory_optimization, 'max_memory_usage_gb', 0)
                    if required_gpu_memory > gpu_memory_gb:
                        compatibility_issues.append(
                            f"Required GPU memory ({required_gpu_memory}GB) exceeds available ({gpu_memory_gb:.1f}GB)"
                        )
                except Exception as e:
                    compatibility_issues.append(f"Could not check GPU memory: {e}")

            # Check CPU cores vs max_workers
            import os
            cpu_cores = os.cpu_count() or 1
            max_workers = env_config.max_workers
            if max_workers > cpu_cores:
                compatibility_issues.append(
                    f"max_workers ({max_workers}) exceeds CPU cores ({cpu_cores})"
                )

        except Exception as e:
            compatibility_issues.append(f"Hardware compatibility check failed: {e}")

        return compatibility_issues

    def perform_startup_validation(self) -> Tuple[bool, List[str]]:
        """Perform comprehensive startup validation.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        all_issues = []

        # Validate all configurations
        config_errors = self.validate_all_configs()
        all_issues.extend(config_errors)

        # Validate hardware compatibility
        hardware_issues = self.validate_hardware_compatibility()
        all_issues.extend(hardware_issues)

        # Validate required directories exist
        try:
            env_config = self.get_environment_config()
            log_path = Path(env_config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Check cache directory
            if hasattr(env_config, 'cache') and getattr(env_config.cache, 'enable_caching', False):
                cache_dir = Path(getattr(env_config.cache, 'cache_directory', 'cache'))
                cache_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            all_issues.append(f"Directory setup failed: {e}")

        # Check write permissions
        try:
            test_file = self.config_dir.parent / "test_write_permission.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            all_issues.append(f"Write permission check failed: {e}")

        is_valid = len(all_issues) == 0
        return is_valid, all_issues

    def setup_logging(self) -> None:
        """Setup logging based on environment configuration."""
        env_config = self.get_environment_config()

        # Configure logging level
        log_level = getattr(logging, env_config.log_level.upper())
        logging.basicConfig(level=log_level)

        if env_config.log_to_file:
            # Create log directory if it doesn't exist
            log_path = Path(env_config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Setup file handler
            if env_config.log_rotation:
                from logging.handlers import RotatingFileHandler
                handler = RotatingFileHandler(
                    log_path,
                    maxBytes=env_config.max_log_size_mb * 1024 * 1024,
                    backupCount=env_config.backup_count
                )
            else:
                handler = logging.FileHandler(log_path)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)

    def get_device(self, device_config: str = "auto") -> torch.device:
        """Get torch device based on configuration.

        Args:
            device_config: Device configuration ("auto", "cuda", "cpu").

        Returns:
            PyTorch device object.
        """
        if device_config == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")
        elif device_config == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Invalid device configuration: {device_config}")

    def resolve_layer_name(self, model_config: Dict[str, Any],
                          layer_type: str, layer_num: int) -> str:
        """Resolve layer name from configuration template.

        Args:
            model_config: Model configuration dictionary.
            layer_type: Type of layer ("attention", "mlp").
            layer_num: Layer number.

        Returns:
            Resolved layer name.
        """
        if layer_type not in model_config['layers']:
            raise KeyError(f"Layer type '{layer_type}' not found in model config")

        layer_template = model_config['layers'][layer_type]
        return layer_template.format(layer=layer_num)

    def _load_environment_specific_config(self, base_filename: str) -> Dict[str, Any]:
        """Load environment-specific configuration with inheritance.

        Args:
            base_filename: Base configuration file name.

        Returns:
            Merged configuration dictionary.
        """
        # Load base configuration
        base_config = self._load_yaml_file(base_filename)

        # Load environment-specific configuration if it exists
        env_filename = f"environment_{self.environment}.yaml"
        env_config_path = self.config_dir / env_filename

        if env_config_path.exists():
            try:
                env_config = self._load_yaml_file(env_filename)
                # Merge environment config into base config
                merged_config = self._merge_configs(base_config, env_config)
                self._logger.debug(f"Merged environment config from {env_filename}")
                return merged_config
            except Exception as e:
                self._logger.warning(f"Failed to load environment config {env_filename}: {e}")
                return base_config
        else:
            self._logger.debug(f"No environment-specific config found: {env_filename}")
            return base_config

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries.

        Args:
            base: Base configuration dictionary.
            override: Override configuration dictionary.

        Returns:
            Merged configuration dictionary.
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def create_output_paths(self, base_config: Dict[str, Any]) -> Dict[str, Path]:
        """Create output directory paths from configuration.

        Args:
            base_config: Base configuration containing file paths.

        Returns:
            Dictionary with resolved Path objects.
        """
        paths = {}

        # Create data directories
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        (data_dir / "raw").mkdir(exist_ok=True)
        (data_dir / "processed").mkdir(exist_ok=True)
        (data_dir / "outputs").mkdir(exist_ok=True)

        # Resolve paths from config
        for key, path_str in base_config.items():
            if isinstance(path_str, str) and ('/' in path_str or path_str.endswith('.csv') or path_str.endswith('.jsonl')):
                paths[key] = Path(path_str)
                # Create parent directory if it doesn't exist
                paths[key].parent.mkdir(parents=True, exist_ok=True)

        return paths


# Global configuration instance
_global_config_manager = None


def get_config_manager(config_dir: Optional[str] = None,
                      environment: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance.

    Args:
        config_dir: Path to configuration directory. Only used on first call.
        environment: Environment type. Only used on first call.

    Returns:
        Global ConfigManager instance.
    """
    global _global_config_manager

    if _global_config_manager is None:
        # Default environment from environment variable or 'development'
        if environment is None:
            environment = os.environ.get('NEURONMAP_ENV', 'development')

        _global_config_manager = ConfigManager(
            config_dir=config_dir,
            environment=environment
        )

    return _global_config_manager


def reset_config_manager() -> None:
    """Reset the global configuration manager instance.

    Useful for testing or environment switching.
    """
    global _global_config_manager
    _global_config_manager = None


def setup_global_config(config_dir: Optional[str] = None,
                       environment: Optional[str] = None) -> ConfigManager:
    """Setup and return global configuration manager with validation.

    Args:
        config_dir: Path to configuration directory.
        environment: Environment type.

    Returns:
        Configured and validated ConfigManager instance.

    Raises:
        RuntimeError: If configuration validation fails.
    """
    # Reset any existing instance
    reset_config_manager()

    # Create new instance
    config_manager = get_config_manager(config_dir, environment)

    # Perform startup validation
    is_valid, issues = config_manager.perform_startup_validation()

    if not is_valid:
        raise RuntimeError(f"Configuration validation failed: {'; '.join(issues)}")

    # Setup logging
    config_manager.setup_logging()

    return config_manager


def get_config(config_name: str = "default"):
    """Get configuration using the config manager.

    Args:
        config_name: Name of the configuration to load

    Returns:
        NeuronMapConfig instance
    """
    # Import here to avoid circular imports
    from .config_manager import get_config as config_manager_get_config

    # Handle test compatibility - if config_name is provided, ignore it for now
    if config_name:
        logger.debug(f"Config name '{config_name}' provided but using default config")

    return config_manager_get_config()