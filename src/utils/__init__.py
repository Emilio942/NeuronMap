"""Utilities module for NeuronMap."""

# Core utilities
from .config_manager import NeuronMapConfig, ModelConfig, DataConfig, AnalysisConfig
from .validation import (
    validate_analysis_request, validate_experiment_config, validate_config_file,
    validate_model_name, validate_layer_name, ValidationResult
)
from .error_handling import (
    NeuronMapException, ValidationError, ConfigurationError,
    ModelCompatibilityError, ResourceError
)

# Advanced systems
from .model_compatibility import (
    ModelCompatibilityChecker, ModelType, AnalysisType, ModelInfo,
    SystemResources, CompatibilityResult, check_model_compatibility,
    get_compatible_models
)
from .resource_monitor import (
    GPUResourceManager, GPUStatus, MemoryProfile, VRAMOptimizer,
    WorkloadScheduler, WorkloadTask, WorkloadPriority,
    get_gpu_memory_usage, optimize_gpu_memory
)

# Section 2.3: Monitoring & Observability
from .progress_tracker import (
    ProgressTracker, ProgressState, MultiLevelProgressTracker,
    ProgressMetrics, ProgressSnapshot, ETACalculator
)
from .comprehensive_system_monitor import (
    SystemResourceMonitor, ResourceThresholds, AlertLevel,
    CPUMetrics, MemoryMetrics, DiskMetrics, NetworkMetrics,
    SystemMetrics, ResourceOptimizer,
    get_system_info, format_bytes, format_metrics_summary
)
from .performance_metrics import (
    PerformanceCollector, MetricType, AggregationType,
    PerformanceMetric, AggregatedMetric, PerformanceTrend,
    measure_operation, record_metric, get_performance_summary, analyze_trends
)
from .health_monitor import (
    HealthMonitor, ServiceEndpoint, ServiceType, ServiceStatus,
    HealthCheckResult, ServiceStatistics, CircuitBreaker,
    register_ollama_service, get_service_health, check_ollama_connection
)

from .input_validation import InputValidator
from .robust_decorators import (
    robust_execution, model_operation_handler, data_processing_handler,
    file_operation_handler, gpu_operation_handler
)

# Section 3.1: Multi-Model Support Extension
try:
    from .multi_model_support import (
        MultiModelAnalyzer, UniversalModelRegistry, UniversalModelAdapter,
        GPTModelAdapter, BERTModelAdapter, LayerMapping,
        ModelFamily, ModelArchitecture, multi_model_analyzer
    )
except ImportError:
    pass

# Legacy monitoring (for backward compatibility) - SystemResourceMonitor already imported above
try:
    from .system_monitor import start_system_monitoring, stop_system_monitoring
except ImportError:
    pass

__all__ = [
    # Core
    'NeuronMapConfig', 'ModelConfig', 'DataConfig', 'AnalysisConfig',
    'validate_analysis_request', 'validate_experiment_config', 'validate_config_file',
    'validate_model_name', 'validate_layer_name', 'ValidationResult',
    'NeuronMapException', 'ValidationError', 'ConfigurationError', 'ModelCompatibilityError', 'ResourceError',

    # Model compatibility
    'ModelCompatibilityChecker', 'ModelType', 'AnalysisType', 'ModelInfo',
    'SystemResources', 'CompatibilityResult', 'check_model_compatibility', 'get_compatible_models',

    # GPU Resource monitoring
    'GPUResourceManager', 'GPUStatus', 'MemoryProfile', 'VRAMOptimizer',
    'WorkloadScheduler', 'WorkloadTask', 'WorkloadPriority',
    'get_gpu_memory_usage', 'optimize_gpu_memory',

    # Section 2.3: Monitoring & Observability
    'ProgressTracker', 'ProgressState', 'MultiLevelProgressTracker',
    'ProgressMetrics', 'ProgressSnapshot', 'ETACalculator',
    'SystemResourceMonitor', 'ResourceThresholds', 'AlertLevel',
    'CPUMetrics', 'MemoryMetrics', 'DiskMetrics', 'NetworkMetrics',
    'SystemMetrics', 'ResourceOptimizer',
    'get_system_info', 'format_bytes', 'format_metrics_summary',
    'PerformanceCollector', 'MetricType', 'AggregationType',
    'PerformanceMetric', 'AggregatedMetric', 'PerformanceTrend',
    'measure_operation', 'record_metric', 'get_performance_summary', 'analyze_trends',
    'HealthMonitor', 'ServiceEndpoint', 'ServiceType', 'ServiceStatus',
    'HealthCheckResult', 'ServiceStatistics', 'CircuitBreaker',
    'register_ollama_service', 'get_service_health', 'check_ollama_connection',

    # Section 3.1: Multi-Model Support Extension
    'MultiModelAnalyzer', 'UniversalModelRegistry', 'UniversalModelAdapter',
    'GPTModelAdapter', 'BERTModelAdapter', 'LayerMapping',
    'ModelFamily', 'ModelArchitecture', 'multi_model_analyzer',

    # Input validation
    'InputValidator',

    # Decorators
    'robust_execution', 'model_operation_handler', 'data_processing_handler',
    'file_operation_handler', 'gpu_operation_handler',

    # Legacy
    'SystemResourceMonitor', 'start_system_monitoring', 'stop_system_monitoring'
]
