"""
Configuration Schema for Model Interventions - B6: Konfigurations-Schema
This module implements robust Pydantic-based configuration schemas for intervention
experiments, enabling YAML-based configuration of ablation and patching experiments.
"""
from pydantic import BaseModel, Field, validator, model_validator
from typing import Dict, List, Optional, Union, Any, Literal
from enum import Enum
import yaml
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
class InterventionTypeConfig(str, Enum):
    """Supported intervention types."""
    ABLATION = "ablation"
    NOISE = "noise"
    MEAN = "mean"
    PATCHING = "patching"
    CUSTOM = "custom"
class MetricConfig(str, Enum):
    """Supported metrics for effect measurement."""
    LOGIT_DIFF = "logit_diff"
    PROBABILITY = "probability"
    ENTROPY = "entropy"
    KL_DIVERGENCE = "kl_divergence"
    COSINE_SIMILARITY = "cosine_similarity"
class LayerSelectionConfig(BaseModel):
    """Configuration for layer selection."""
    names: Optional[List[str]] = Field(None, description="Explicit layer names")
    patterns: Optional[List[str]] = Field(None, description="Regex patterns for layer names")
    indices: Optional[List[int]] = Field(None, description="Layer indices")
    layer_types: Optional[List[str]] = Field(None, description="Layer types (e.g., 'attention', 'mlp')")
    @model_validator(mode='after')
    def validate_selection(cls, values):
        """Ensure at least one selection method is provided."""
        selection_methods = [values.names, values.patterns,
                           values.indices, values.layer_types]
        if not any(selection_methods):
            raise ValueError("At least one layer selection method must be provided")
        return values
class NeuronSelectionConfig(BaseModel):
    """Configuration for neuron/head selection within layers."""
    indices: Optional[List[int]] = Field(None, description="Specific neuron indices")
    range_start: Optional[int] = Field(None, description="Start of neuron range")
    range_end: Optional[int] = Field(None, description="End of neuron range")
    top_k: Optional[int] = Field(None, description="Select top-k most active neurons")
    bottom_k: Optional[int] = Field(None, description="Select bottom-k least active neurons")
    random_k: Optional[int] = Field(None, description="Select k random neurons")
    percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Percentage of neurons to select")
    @validator('range_end')
    def validate_range(cls, v, values):
        """Ensure range_end > range_start if both provided."""
        if v is not None and values.get('range_start') is not None:
            if v <= values['range_start']:
                raise ValueError("range_end must be greater than range_start")
        return v
class InterventionTargetConfig(BaseModel):
    """Configuration for a single intervention target."""
    layer_selection: LayerSelectionConfig = Field(..., description="How to select target layers")
    neuron_selection: Optional[NeuronSelectionConfig] = Field(None, description="How to select neurons within layers")
    intervention_type: InterventionTypeConfig = Field(..., description="Type of intervention to apply")
    intervention_value: Optional[float] = Field(None, description="Value for ablation/noise interventions")
    noise_std: Optional[float] = Field(0.1, ge=0.0, description="Standard deviation for noise intervention")
    custom_function: Optional[str] = Field(None, description="Python function for custom interventions")
    @validator('intervention_value')
    def validate_intervention_value(cls, v, values):
        """Validate intervention_value based on intervention_type."""
        intervention_type = values.get('intervention_type')
        if intervention_type == InterventionTypeConfig.ABLATION and v is not None and v != 0.0:
            raise ValueError("Ablation intervention_value should be 0.0 or None")
        return v
class ExperimentInputConfig(BaseModel):
    """Configuration for experiment inputs."""
    clean_prompts: List[str] = Field(..., description="Clean input prompts")
    corrupted_prompts: Optional[List[str]] = Field(None, description="Corrupted prompts for patching")
    input_file: Optional[Path] = Field(None, description="File containing prompts")
    tokenizer_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for tokenizer")
    @model_validator(mode='after')
    def validate_prompts(cls, values):
        """Ensure prompts are provided either directly or via file."""
        clean_prompts = values.clean_prompts or []
        input_file = values.input_file
        if not clean_prompts and not input_file:
            raise ValueError("Either clean_prompts or input_file must be provided")
        # For patching experiments, corrupted prompts are required
        corrupted_prompts = values.corrupted_prompts
        if corrupted_prompts and len(corrupted_prompts) != len(clean_prompts):
            raise ValueError("Number of corrupted_prompts must match clean_prompts")
        return values
class ModelConfig(BaseModel):
    """Configuration for the target model."""
    name: str = Field(..., description="Model name or path")
    device: str = Field("auto", description="Device to run model on")
    precision: str = Field("float32", description="Model precision")
    max_length: Optional[int] = Field(None, description="Maximum sequence length")
    batch_size: int = Field(1, ge=1, description="Batch size for processing")
    class Config:
        extra = "allow"  # Allow additional model-specific parameters
class CacheConfig(BaseModel):
    """Configuration for intervention cache."""
    enabled: bool = Field(True, description="Whether to use caching")
    cache_dir: Optional[Path] = Field(None, description="Cache directory")
    max_memory_gb: float = Field(2.0, ge=0.1, description="Maximum memory for cache")
    compression_level: int = Field(6, ge=0, le=9, description="Compression level for disk cache")
    experiment_id: Optional[str] = Field(None, description="Experiment identifier for cache")
    auto_cleanup: bool = Field(True, description="Automatically clean up old cache entries")
class OutputConfig(BaseModel):
    """Configuration for experiment outputs."""
    output_dir: Path = Field(..., description="Directory for output files")
    save_activations: bool = Field(False, description="Save intermediate activations")
    save_detailed_results: bool = Field(True, description="Save detailed analysis results")
    plot_results: bool = Field(True, description="Generate visualization plots")
    export_format: List[str] = Field(["json", "csv"], description="Export formats")
    @validator('export_format')
    def validate_export_format(cls, v):
        """Validate export formats."""
        valid_formats = ["json", "csv", "pickle", "yaml"]
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}. Valid: {valid_formats}")
        return v
class AnalysisConfig(BaseModel):
    """Configuration for analysis methods."""
    metrics: List[MetricConfig] = Field([MetricConfig.LOGIT_DIFF], description="Metrics to compute")
    baseline_comparison: bool = Field(True, description="Compare against baseline")
    statistical_tests: bool = Field(False, description="Perform statistical significance tests")
    confidence_level: float = Field(0.95, ge=0.0, le=1.0, description="Confidence level for statistics")
    effect_size_threshold: float = Field(0.1, ge=0.0, description="Minimum effect size to consider significant")
class AblationExperimentConfig(BaseModel):
    """Complete configuration for an ablation experiment."""
    experiment_name: str = Field(..., description="Name of the experiment")
    description: Optional[str] = Field(None, description="Experiment description")
    model: ModelConfig = Field(..., description="Model configuration")
    inputs: ExperimentInputConfig = Field(..., description="Input configuration")
    targets: List[InterventionTargetConfig] = Field(..., description="Intervention targets")
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig, description="Analysis configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    output: OutputConfig = Field(..., description="Output configuration")
    # Metadata
    created_by: Optional[str] = Field(None, description="Experiment creator")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Experiment tags")
    @validator('targets')
    def validate_targets(cls, v):
        """Ensure at least one target is specified."""
        if not v:
            raise ValueError("At least one intervention target must be specified")
        return v
class PathPatchingExperimentConfig(BaseModel):
    """Complete configuration for a path patching experiment."""
    experiment_name: str = Field(..., description="Name of the experiment")
    description: Optional[str] = Field(None, description="Experiment description")
    model: ModelConfig = Field(..., description="Model configuration")
    inputs: ExperimentInputConfig = Field(..., description="Input configuration")
    patch_targets: List[InterventionTargetConfig] = Field(..., description="Layers/neurons to patch")
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig, description="Analysis configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    output: OutputConfig = Field(..., description="Output configuration")
    # Path patching specific
    clean_cache_key: Optional[str] = Field(None, description="Cache key for clean activations")
    patch_order: List[str] = Field(default_factory=list, description="Order of patching operations")
    # Metadata
    created_by: Optional[str] = Field(None, description="Experiment creator")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Experiment tags")
    @model_validator(mode='after')
    def validate_patching_config(cls, values):
        """Validate path patching specific configuration."""
        inputs = values.inputs
        corrupted_prompts = getattr(inputs, 'corrupted_prompts', None)
        if not corrupted_prompts:
            raise ValueError("Path patching experiments require corrupted_prompts")
        return values
# Configuration loading and validation utilities
class ConfigurationManager:
    """Manager for loading and validating intervention configurations."""
    @staticmethod
    def load_ablation_config(config_path: Union[str, Path]) -> AblationExperimentConfig:
        """Load and validate ablation experiment configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return AblationExperimentConfig(**config_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")
    @staticmethod
    def load_patching_config(config_path: Union[str, Path]) -> PathPatchingExperimentConfig:
        """Load and validate path patching experiment configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return PathPatchingExperimentConfig(**config_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")
    @staticmethod
    def save_config(config: Union[AblationExperimentConfig, PathPatchingExperimentConfig],
                   output_path: Union[str, Path]):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}")
    @staticmethod
    def create_example_ablation_config() -> AblationExperimentConfig:
        """Create an example ablation configuration."""
        return AblationExperimentConfig(
            experiment_name="example_ablation",
            description="Example ablation experiment configuration",
            model=ModelConfig(
                name="gpt2",
                device="auto",
                precision="float32"
            ),
            inputs=ExperimentInputConfig(
                clean_prompts=[
                    "The capital of France is",
                    "The largest planet in our solar system is"
                ]
            ),
            targets=[
                InterventionTargetConfig(
                    layer_selection=LayerSelectionConfig(
                        patterns=["transformer.h.8.*"]
                    ),
                    neuron_selection=NeuronSelectionConfig(
                        top_k=100
                    ),
                    intervention_type=InterventionTypeConfig.ABLATION
                )
            ],
            output=OutputConfig(
                output_dir=Path("./outputs/ablation_example"),
                save_detailed_results=True,
                plot_results=True
            )
        )
    @staticmethod
    def create_example_patching_config() -> PathPatchingExperimentConfig:
        """Create an example path patching configuration."""
        return PathPatchingExperimentConfig(
            experiment_name="example_patching",
            description="Example path patching experiment configuration",
            model=ModelConfig(
                name="gpt2",
                device="auto"
            ),
            inputs=ExperimentInputConfig(
                clean_prompts=["The capital of France is Paris"],
                corrupted_prompts=["The capital of Germany is Paris"]
            ),
            patch_targets=[
                InterventionTargetConfig(
                    layer_selection=LayerSelectionConfig(
                        names=["transformer.h.6.mlp", "transformer.h.7.mlp"]
                    ),
                    intervention_type=InterventionTypeConfig.PATCHING
                )
            ],
            output=OutputConfig(
                output_dir=Path("./outputs/patching_example"),
                save_detailed_results=True
            )
        )
# Template generation utilities
def generate_config_template(experiment_type: Literal["ablation", "patching"]) -> str:
    """Generate a YAML template for the specified experiment type."""
    if experiment_type == "ablation":
        config = ConfigurationManager.create_example_ablation_config()
    elif experiment_type == "patching":
        config = ConfigurationManager.create_example_patching_config()
    else:
        raise ValueError("experiment_type must be 'ablation' or 'patching'")
    return yaml.dump(config.dict(), default_flow_style=False, indent=2)
def validate_config_file(config_path: Union[str, Path],
                        experiment_type: Literal["ablation", "patching"]) -> bool:
    """Validate a configuration file without loading it completely."""
    try:
        if experiment_type == "ablation":
            ConfigurationManager.load_ablation_config(config_path)
        else:
            ConfigurationManager.load_patching_config(config_path)
        logger.info(f"Configuration file {config_path} is valid")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
