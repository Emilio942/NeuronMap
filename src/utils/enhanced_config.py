"""Enhanced configuration management with validation for NeuronMap."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
from pydantic import BaseModel, Field, validator, ValidationError
import logging


logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Validated model configuration."""

    name: str = Field(..., description="Model name or path")
    type: str = Field(..., pattern="^(gpt|bert|t5|llama|other)$", description="Model architecture type")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Model parameters")
    layers: Dict[str, Any] = Field(..., description="Layer configuration")
    memory_requirements: Optional[Dict[str, float]] = Field(default=None, description="Memory requirements")
    supported_tasks: List[str] = Field(default_factory=list, description="Supported analysis tasks")
    max_context_length: int = Field(default=512, gt=0, le=32768, description="Maximum context length")

    @validator('name')
    def validate_model_name(cls, v):
        """Validate model name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @validator('layers')
    def validate_layers_config(cls, v):
        """Validate layer configuration structure."""
        required_keys = ['total_layers']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required layer config key: {key}")

        if not isinstance(v['total_layers'], int) or v['total_layers'] <= 0:
            raise ValueError("total_layers must be a positive integer")

        return v

    @validator('memory_requirements')
    def validate_memory_requirements(cls, v):
        """Validate memory requirements."""
        if v is None:
            return v

        valid_keys = ['min_ram_gb', 'recommended_ram_gb', 'gpu_memory_gb']
        for key, value in v.items():
            if key not in valid_keys:
                logger.warning(f"Unknown memory requirement key: {key}")
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Memory requirement {key} must be a non-negative number")

        return v


class AnalysisConfig(BaseModel):
    """Validated analysis configuration."""

    batch_size: int = Field(default=32, gt=0, le=1024, description="Batch size for processing")
    max_sequence_length: int = Field(default=512, gt=0, le=32768, description="Maximum sequence length")
    device: str = Field(default="auto", pattern="^(auto|cuda|cpu|cuda:[0-9]+)$", description="Device configuration")
    precision: str = Field(default="float32", pattern="^(float16|float32|float64)$", description="Computation precision")
    enable_gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    cache_activations: bool = Field(default=True, description="Cache extracted activations")

    @validator('device')
    def validate_device_availability(cls, v):
        """Validate device availability."""
        if v == "auto":
            return v
        elif v == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif v.startswith("cuda:"):
            device_id = int(v.split(":")[1])
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device_id} not available")
        return v


class VisualizationConfig(BaseModel):
    """Validated visualization configuration."""

    figure_width: int = Field(default=10, gt=0, le=50, description="Figure width in inches")
    figure_height: int = Field(default=8, gt=0, le=50, description="Figure height in inches")
    dpi: int = Field(default=300, gt=50, le=1000, description="Figure DPI")
    color_scheme: str = Field(default="viridis", description="Default color scheme")
    interactive: bool = Field(default=True, description="Enable interactive visualizations")
    export_format: str = Field(default="png", pattern="^(png|svg|pdf|html)$", description="Default export format")

    @validator('color_scheme')
    def validate_color_scheme(cls, v):
        """Validate color scheme availability."""
        import matplotlib.pyplot as plt
        try:
            plt.cm.get_cmap(v)
        except ValueError:
            logger.warning(f"Color scheme '{v}' not found, using default")
            return "viridis"
        return v


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration."""

    environment: str = Field(default="development", pattern="^(development|testing|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    enable_debugging: bool = Field(default=False, description="Enable debugging features")
    max_workers: int = Field(default=4, gt=0, le=32, description="Maximum worker processes")
    timeout_seconds: int = Field(default=300, gt=0, le=3600, description="Operation timeout")

    @validator('max_workers')
    def validate_max_workers(cls, v):
        """Validate maximum workers against system capabilities."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if v > cpu_count * 2:
            logger.warning(f"max_workers {v} exceeds recommended limit, using {cpu_count}")
            return cpu_count
        return v


class ConfigManager:
    """Enhanced configuration manager with validation."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        self._models_config: Optional[Dict[str, ModelConfig]] = None
        self._analysis_config: Optional[AnalysisConfig] = None
        self._visualization_config: Optional[VisualizationConfig] = None
        self._environment_config: Optional[EnvironmentConfig] = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            env_config = self.get_environment_config()
            logging.basicConfig(
                level=getattr(logging, env_config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        except Exception:
            # Fallback if config loading fails
            logging.basicConfig(level=logging.INFO)

    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load and validate YAML configuration file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Loaded configuration dictionary.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            yaml.YAMLError: If YAML syntax is invalid.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.debug(f"Loaded configuration from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML syntax in {config_path}: {e}")
            raise

    def get_models_config(self) -> Dict[str, ModelConfig]:
        """Load and validate model configurations.

        Returns:
            Dictionary of validated model configurations.

        Raises:
            ValidationError: If model configuration is invalid.
        """
        if self._models_config is None:
            config_path = self.config_dir / "models.yaml"
            raw_config = self._load_yaml_config(config_path)

            self._models_config = {}
            for model_name, model_config in raw_config.get('models', {}).items():
                try:
                    self._models_config[model_name] = ModelConfig(**model_config)
                    logger.debug(f"Validated model config: {model_name}")
                except ValidationError as e:
                    logger.error(f"Invalid model config for '{model_name}': {e}")
                    raise

        return self._models_config

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get validated configuration for a specific model.

        Args:
            model_name: Name of the model configuration to retrieve.

        Returns:
            Validated model configuration.

        Raises:
            KeyError: If model configuration not found.
        """
        models = self.get_models_config()
        if model_name not in models:
            available_models = list(models.keys())
            raise KeyError(f"Model configuration '{model_name}' not found. "
                         f"Available models: {available_models}")
        return models[model_name]

    def get_analysis_config(self) -> AnalysisConfig:
        """Load and validate analysis configuration.

        Returns:
            Validated analysis configuration.
        """
        if self._analysis_config is None:
            config_path = self.config_dir / "analysis.yaml"
            if config_path.exists():
                raw_config = self._load_yaml_config(config_path)
                self._analysis_config = AnalysisConfig(**raw_config.get('analysis', {}))
            else:
                # Use defaults if config file doesn't exist
                self._analysis_config = AnalysisConfig()
                logger.warning(f"Analysis config file not found, using defaults")

        return self._analysis_config

    def get_visualization_config(self) -> VisualizationConfig:
        """Load and validate visualization configuration.

        Returns:
            Validated visualization configuration.
        """
        if self._visualization_config is None:
            config_path = self.config_dir / "visualization.yaml"
            if config_path.exists():
                raw_config = self._load_yaml_config(config_path)
                self._visualization_config = VisualizationConfig(**raw_config.get('visualization', {}))
            else:
                self._visualization_config = VisualizationConfig()
                logger.warning(f"Visualization config file not found, using defaults")

        return self._visualization_config

    def get_environment_config(self) -> EnvironmentConfig:
        """Load and validate environment configuration.

        Returns:
            Validated environment configuration.
        """
        if self._environment_config is None:
            config_path = self.config_dir / "environment.yaml"
            if config_path.exists():
                raw_config = self._load_yaml_config(config_path)
                self._environment_config = EnvironmentConfig(**raw_config.get('environment', {}))
            else:
                self._environment_config = EnvironmentConfig()
                logger.warning(f"Environment config file not found, using defaults")

        return self._environment_config

    def validate_all_configs(self) -> List[str]:
        """Validate all configuration files.

        Returns:
            List of validation errors (empty if all valid).
        """
        errors = []

        try:
            self.get_models_config()
            logger.info("Models configuration validated successfully")
        except Exception as e:
            errors.append(f"Models config error: {e}")

        try:
            self.get_analysis_config()
            logger.info("Analysis configuration validated successfully")
        except Exception as e:
            errors.append(f"Analysis config error: {e}")

        try:
            self.get_visualization_config()
            logger.info("Visualization configuration validated successfully")
        except Exception as e:
            errors.append(f"Visualization config error: {e}")

        try:
            self.get_environment_config()
            logger.info("Environment configuration validated successfully")
        except Exception as e:
            errors.append(f"Environment config error: {e}")

        return errors

    def check_hardware_compatibility(self, model_name: str) -> Dict[str, bool]:
        """Check hardware compatibility for a specific model.

        Args:
            model_name: Model name to check compatibility for.

        Returns:
            Dictionary with compatibility status.
        """
        model_config = self.get_model_config(model_name)
        analysis_config = self.get_analysis_config()

        compatibility = {
            "sufficient_ram": True,
            "gpu_available": torch.cuda.is_available(),
            "sufficient_gpu_memory": True,
            "device_compatible": True
        }

        # Check RAM requirements
        if model_config.memory_requirements:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            min_ram = model_config.memory_requirements.get('min_ram_gb', 0)
            compatibility["sufficient_ram"] = available_ram_gb >= min_ram

        # Check GPU memory requirements
        if compatibility["gpu_available"] and model_config.memory_requirements:
            required_gpu_memory = model_config.memory_requirements.get('gpu_memory_gb', 0)
            if required_gpu_memory > 0:
                try:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    compatibility["sufficient_gpu_memory"] = gpu_memory_gb >= required_gpu_memory
                except Exception:
                    compatibility["sufficient_gpu_memory"] = False

        # Check device compatibility
        device_config = analysis_config.device
        if device_config.startswith("cuda") and not compatibility["gpu_available"]:
            compatibility["device_compatible"] = False

        return compatibility

    def get_device(self, model_name: Optional[str] = None) -> torch.device:
        """Get optimal torch device based on configuration and hardware.

        Args:
            model_name: Optional model name for hardware compatibility check.

        Returns:
            PyTorch device object.
        """
        analysis_config = self.get_analysis_config()
        device_config = analysis_config.device

        if device_config == "auto":
            # Check hardware compatibility if model specified
            if model_name:
                compatibility = self.check_hardware_compatibility(model_name)
                if not compatibility["gpu_available"] or not compatibility["sufficient_gpu_memory"]:
                    logger.warning(f"GPU not suitable for {model_name}, using CPU")
                    return torch.device("cpu")

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        elif device_config == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")

        elif device_config.startswith("cuda:"):
            device_id = int(device_config.split(":")[1])
            if device_id >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {device_id} not available")
            return torch.device(device_config)

        elif device_config == "cpu":
            return torch.device("cpu")

        else:
            raise ValueError(f"Invalid device configuration: {device_config}")

    def create_output_paths(self, experiment_name: str = "default") -> Dict[str, Path]:
        """Create organized output directory structure.

        Args:
            experiment_name: Name of the experiment for organization.

        Returns:
            Dictionary with resolved output paths.
        """
        base_dir = Path("data/outputs") / experiment_name

        paths = {
            "base": base_dir,
            "activations": base_dir / "activations",
            "analysis": base_dir / "analysis",
            "visualizations": base_dir / "visualizations",
            "logs": base_dir / "logs",
            "metadata": base_dir / "metadata"
        }

        # Create all directories
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output structure for experiment: {experiment_name}")
        return paths


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def validate_startup_config() -> bool:
    """Validate configuration at startup.

    Returns:
        True if all configurations are valid, False otherwise.
    """
    try:
        config_manager = get_config_manager()
        errors = config_manager.validate_all_configs()

        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info("All configurations validated successfully")
        return True

    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return False


# Backward compatibility
def get_config() -> ConfigManager:
    """Get configuration manager (backward compatibility)."""
    return get_config_manager()
