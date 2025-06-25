"""Enhanced error handling and recovery mechanisms for NeuronMap."""

import time
import logging
import functools
from typing import Callable, Any, Optional, List, Dict, Union, Type
from dataclasses import dataclass
import traceback
import json
from pathlib import Path
import torch


logger = logging.getLogger(__name__)


# ===== COMPREHENSIVE EXCEPTION HIERARCHY =====

class NeuronMapException(Exception):
    """Base exception for all NeuronMap-specific errors."""

    def __init__(self, message: str, error_code: str, context: dict = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.format_error_message())

    def format_error_message(self) -> str:
        """Format the error message with context."""
        msg = f"[{self.error_code}] {self.message}"
        if self.context:
            msg += f" | Context: {self.context}"
        return msg


class ModelLoadingError(NeuronMapException):
    """Raised when model loading fails."""

    def __init__(self, model_name: str, reason: str, context: dict = None):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_ERROR",
            context=context
        )
        self.model_name = model_name
        self.reason = reason


class ActivationExtractionError(NeuronMapException):
    """Raised when activation extraction fails."""

    def __init__(self, layer_name: str, reason: str, context: dict = None):
        super().__init__(
            message=f"Failed to extract activations from layer '{layer_name}': {reason}",
            error_code="ACTIVATION_ERROR",
            context=context
        )
        self.layer_name = layer_name
        self.reason = reason


class ConfigurationError(NeuronMapException):
    """Raised when configuration is invalid."""

    def __init__(self, field: str, reason: str, suggestion: str = None, context: dict = None):
        message = f"Configuration error in field '{field}': {reason}"
        if suggestion:
            message += f" | Suggestion: {suggestion}"

        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            context=context
        )
        self.field = field
        self.suggestion = suggestion


class ValidationError(NeuronMapException):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: Any, reason: str, context: dict = None):
        super().__init__(
            message=f"Validation failed for field '{field}' with value '{value}': {reason}",
            error_code="VALIDATION_ERROR",
            context=context
        )
        self.field = field
        self.value = value


class ResourceError(NeuronMapException):
    """Raised when system resources are insufficient."""

    def __init__(self, resource_type: str, required: str, available: str, context: dict = None):
        super().__init__(
            message=f"Insufficient {resource_type}: required {required}, available {available}",
            error_code="RESOURCE_ERROR",
            context=context
        )
        self.resource_type = resource_type
        self.required = required
        self.available = available


class NetworkError(NeuronMapException):
    """Raised when network operations fail."""

    def __init__(self, operation: str, reason: str, context: dict = None):
        super().__init__(
            message=f"Network operation '{operation}' failed: {reason}",
            error_code="NETWORK_ERROR",
            context=context
        )
        self.operation = operation


class PluginError(NeuronMapException):
    """Raised when plugin operations fail."""

    def __init__(self, plugin_name: str, operation: str, reason: str, context: dict = None):
        super().__init__(
            message=f"Plugin '{plugin_name}' failed during '{operation}': {reason}",
            error_code="PLUGIN_ERROR",
            context=context
        )
        self.plugin_name = plugin_name
        self.operation = operation


class VisualizationError(NeuronMapException):
    """Raised when visualization operations fail."""

    def __init__(self, plot_type: str, reason: str, context: dict = None):
        super().__init__(
            message=f"Visualization '{plot_type}' failed: {reason}",
            error_code="VISUALIZATION_ERROR",
            context=context
        )
        self.plot_type = plot_type


class DataProcessingError(NeuronMapException):
    """Raised when data processing fails."""

    def __init__(self, operation: str, reason: str, context: dict = None):
        super().__init__(
            message=f"Data processing operation '{operation}' failed: {reason}",
            error_code="DATA_PROCESSING_ERROR",
            context=context
        )
        self.operation = operation


class ModelCompatibilityError(NeuronMapException):
    """Raised when model compatibility checks fail."""

    def __init__(self, model_name: str, analysis_type: str, reason: str, context: dict = None):
        super().__init__(
            message=f"Model '{model_name}' is not compatible with analysis '{analysis_type}': {reason}",
            error_code="MODEL_COMPATIBILITY_ERROR",
            context=context
        )
        self.model_name = model_name
        self.analysis_type = analysis_type


# ===== ERROR RECOVERY MECHANISMS =====

@dataclass
class PartialResult:
    """Container for partial results when operations partially fail."""
    successful_components: List[str]
    failed_components: List[Dict[str, Any]]
    degradation_level: float
    data: Dict[str, Any]

    def is_usable(self, threshold: float = 0.5) -> bool:
        """Check if the partial result is usable based on degradation threshold."""
        return self.degradation_level < threshold

    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        total = len(self.successful_components) + len(self.failed_components)
        if total == 0:
            return 0.0
        return (len(self.successful_components) / total) * 100


class AutomaticRecovery:
    """Handles automatic recovery from various types of failures."""

    def __init__(self):
        self.recovery_strategies = {}
        self.fallback_models = {
            'gpt2': ['distilgpt2', 'gpt2-medium'],
            'bert-base-uncased': ['distilbert-base-uncased', 'bert-base-cased'],
            'llama': ['gpt2', 'distilgpt2']
        }

    def register_recovery_strategy(self, error_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy

    def attempt_model_fallback(self, original_model: str) -> Optional[str]:
        """Attempt to use a fallback model when the original fails."""
        for model_pattern, fallbacks in self.fallback_models.items():
            if model_pattern in original_model.lower():
                for fallback in fallbacks:
                    try:
                        # Test if fallback model is available
                        from transformers import AutoTokenizer
                        AutoTokenizer.from_pretrained(fallback)
                        logger.info(f"Using fallback model: {fallback} for {original_model}")
                        return fallback
                    except Exception:
                        continue
        return None

    def recover_from_memory_error(self, original_batch_size: int) -> int:
        """Reduce batch size when encountering memory errors."""
        new_batch_size = max(1, original_batch_size // 2)
        logger.warning(f"Reducing batch size from {original_batch_size} to {new_batch_size} due to memory error")
        return new_batch_size

    def recover_from_gpu_error(self) -> str:
        """Fall back to CPU when GPU operations fail."""
        logger.warning("GPU operation failed, falling back to CPU")
        return "cpu"


def robust_execution(func: Callable) -> Callable:
    """Decorator that provides comprehensive error handling and recovery."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        recovery_manager = AutomaticRecovery()

        try:
            return func(*args, **kwargs)

        except ModelLoadingError as e:
            logger.error(f"Model loading failed: {e.message}")

            # Attempt model fallback
            if hasattr(kwargs, 'model_name') or 'model_name' in kwargs:
                model_name = kwargs.get('model_name', getattr(args[0], 'model_name', None))
                if model_name:
                    fallback_model = recovery_manager.attempt_model_fallback(model_name)
                    if fallback_model:
                        kwargs['model_name'] = fallback_model
                        logger.info(f"Retrying with fallback model: {fallback_model}")
                        return func(*args, **kwargs)
            raise

        except ActivationExtractionError as e:
            logger.warning(f"Activation extraction failed: {e.message}")

            # Return partial results with warning
            return PartialResult(
                successful_components=[],
                failed_components=[{'error': e.message, 'layer': e.layer_name}],
                degradation_level=1.0,
                data={'error': e.message}
            )

        except torch.cuda.OutOfMemoryError as e:
            logger.warning("GPU out of memory, attempting recovery")

            # Reduce batch size and retry
            if 'batch_size' in kwargs:
                original_batch_size = kwargs['batch_size']
                new_batch_size = recovery_manager.recover_from_memory_error(original_batch_size)
                kwargs['batch_size'] = new_batch_size
                return func(*args, **kwargs)

            # Fall back to CPU
            if 'device' in kwargs:
                kwargs['device'] = recovery_manager.recover_from_gpu_error()
                return func(*args, **kwargs)

            raise ResourceError(
                resource_type="GPU_MEMORY",
                required="Unknown",
                available="Insufficient",
                context={'original_error': str(e)}
            )

        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise NetworkError(
                operation=func.__name__,
                reason=str(e),
                context={'function': func.__name__, 'args': str(args)[:100]}
            )

        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise NeuronMapException(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={'function': func.__name__, 'error_type': type(e).__name__}
            )

    return wrapper


class GracefulDegradationManager:
    """Enhanced graceful degradation with partial results support."""

    def __init__(self):
        self.component_status = {}
        self.fallback_configs = {}
        self.partial_results = {}

    def handle_component_failure(self, component: str, error: Exception) -> PartialResult:
        """Handle failure of a system component with graceful degradation."""
        self.component_status[component] = {
            'status': 'failed',
            'error': str(error),
            'timestamp': time.time()
        }

        logger.warning(f"Component '{component}' failed: {error}")

        # Check if we can continue with other components
        successful_components = [
            comp for comp, status in self.component_status.items()
            if status.get('status') != 'failed'
        ]

        failed_components = [
            {'component': comp, 'error': status['error']}
            for comp, status in self.component_status.items()
            if status.get('status') == 'failed'
        ]

        degradation_level = len(failed_components) / len(self.component_status) if self.component_status else 1.0

        partial_result = PartialResult(
            successful_components=successful_components,
            failed_components=failed_components,
            degradation_level=degradation_level,
            data=self.partial_results.copy()
        )

        return partial_result

    def is_system_functional(self, minimum_components: int = 1) -> bool:
        """Check if the system is still functional based on active components."""
        active_components = [
            comp for comp, status in self.component_status.items()
            if status.get('status') != 'failed'
        ]
        return len(active_components) >= minimum_components


# ===== INTELLIGENT ERROR ANALYSIS =====

class ErrorAnalyzer:
    """Analyzes error patterns and provides insights."""

    def __init__(self):
        self.error_patterns = {}
        self.solution_database = {}

    def analyze_error_pattern(self, error: Exception) -> Dict[str, Any]:
        """Analyze an error and provide insights."""
        error_type = type(error).__name__
        error_message = str(error)

        analysis = {
            'error_type': error_type,
            'category': self._categorize_error(error),
            'severity': self._assess_severity(error),
            'likely_causes': self._identify_causes(error),
            'suggested_solutions': self._suggest_solutions(error),
            'prevention_tips': self._prevention_tips(error)
        }

        return analysis

    def _categorize_error(self, error: Exception) -> str:
        """Categorize the error type."""
        if isinstance(error, (ModelLoadingError, ConfigurationError)):
            return "configuration"
        elif isinstance(error, (ResourceError, torch.cuda.OutOfMemoryError)):
            return "resource"
        elif isinstance(error, NetworkError):
            return "network"
        elif isinstance(error, ValidationError):
            return "input"
        else:
            return "unknown"

    def _assess_severity(self, error: Exception) -> str:
        """Assess the severity of the error."""
        if isinstance(error, (ConfigurationError, ValidationError)):
            return "high"  # Prevents system from working
        elif isinstance(error, ResourceError):
            return "medium"  # May be recoverable
        elif isinstance(error, NetworkError):
            return "low"  # Usually temporary
        else:
            return "medium"

    def _identify_causes(self, error: Exception) -> List[str]:
        """Identify likely causes of the error."""
        causes = []

        if isinstance(error, ModelLoadingError):
            causes.extend([
                "Model not found in HuggingFace Hub",
                "Insufficient disk space",
                "Network connectivity issues",
                "Invalid model name or path"
            ])
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            causes.extend([
                "Batch size too large",
                "Model too large for available GPU memory",
                "Memory leak in previous operations",
                "Multiple processes using GPU simultaneously"
            ])
        elif isinstance(error, ConfigurationError):
            causes.extend([
                "Invalid configuration values",
                "Missing required configuration fields",
                "Incompatible configuration combinations"
            ])

        return causes

    def _suggest_solutions(self, error: Exception) -> List[str]:
        """Suggest solutions for the error."""
        solutions = []

        if isinstance(error, ModelLoadingError):
            solutions.extend([
                "Check model name spelling and availability",
                "Ensure sufficient disk space (>5GB)",
                "Check internet connection",
                "Try a smaller model if memory is limited"
            ])
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            solutions.extend([
                "Reduce batch size (try half the current size)",
                "Use gradient checkpointing",
                "Clear GPU cache with torch.cuda.empty_cache()",
                "Use a smaller model or quantization",
                "Process data in smaller chunks"
            ])
        elif isinstance(error, ConfigurationError):
            solutions.extend([
                "Check configuration file syntax",
                "Validate all required fields are present",
                "Review configuration documentation",
                "Use default configuration as template"
            ])

        return solutions

    def _prevention_tips(self, error: Exception) -> List[str]:
        """Provide tips to prevent similar errors."""
        tips = []

        if isinstance(error, ResourceError):
            tips.extend([
                "Monitor system resources before operations",
                "Set appropriate resource limits in configuration",
                "Use resource monitoring tools"
            ])
        elif isinstance(error, NetworkError):
            tips.extend([
                "Implement retry logic for network operations",
                "Use connection pooling",
                "Add timeout configurations"
            ])

        return tips


# Global instances
global_error_analyzer = ErrorAnalyzer()
global_degradation_manager = GracefulDegradationManager()


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    backoff_factor: float = 2.0
    jitter: bool = True


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float
    attempt: int
    max_attempts: int
    error_type: str
    error_message: str
    traceback_str: str
    context_data: Dict[str, Any]


class RetryableError(Exception):
    """Base class for errors that should be retried."""
    pass


class PermanentError(Exception):
    """Base class for errors that should not be retried."""
    pass


def with_retry(config: Optional[RetryConfig] = None,
               retry_on: Optional[List[Type[Exception]]] = None,
               stop_on: Optional[List[Type[Exception]]] = None):
    """Decorator to add retry logic to functions.

    Args:
        config: Retry configuration. If None, uses default.
        retry_on: List of exception types to retry on.
        stop_on: List of exception types to never retry on.
    """
    if config is None:
        config = RetryConfig()

    if retry_on is None:
        retry_on = [RetryableError, ConnectionError, TimeoutError]

    if stop_on is None:
        stop_on = [PermanentError, KeyboardInterrupt]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should stop immediately
                    if any(isinstance(e, exc_type) for exc_type in stop_on):
                        logger.error(f"Permanent error in {func.__name__}: {e}")
                        raise

                    # Check if we should retry
                    should_retry = any(isinstance(e, exc_type) for exc_type in retry_on)

                    if not should_retry or attempt >= config.max_retries:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise

                    # Calculate delay
                    delay = config.base_delay
                    if config.exponential_backoff:
                        delay *= (config.backoff_factor ** attempt)

                    # Apply jitter
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)

                    # Cap the delay
                    delay = min(delay, config.max_delay)

                    logger.warning(f"Attempt {attempt + 1}/{config.max_retries + 1} failed for "
                                 f"{func.__name__}: {e}. Retrying in {delay:.1f}s...")

                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper
    return decorator


class ErrorHandler:
    """Centralized error handling and recovery."""

    def __init__(self, log_file: Optional[str] = None):
        """Initialize error handler.

        Args:
            log_file: Path to error log file. If None, logs to default location.
        """
        self.error_log_file = log_file or "neuronmap_errors.jsonl"
        self.error_history: List[ErrorContext] = []

    def handle_error(self, error: Exception,
                    operation: str,
                    context_data: Optional[Dict[str, Any]] = None,
                    attempt: int = 1,
                    max_attempts: int = 1) -> ErrorContext:
        """Handle and log an error with context.

        Args:
            error: The exception that occurred.
            operation: Description of the operation that failed.
            context_data: Additional context information.
            attempt: Current attempt number.
            max_attempts: Maximum number of attempts.

        Returns:
            ErrorContext object with error details.
        """
        error_context = ErrorContext(
            operation=operation,
            timestamp=time.time(),
            attempt=attempt,
            max_attempts=max_attempts,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            context_data=context_data or {}
        )

        # Log error
        logger.error(f"Error in {operation} (attempt {attempt}/{max_attempts}): "
                    f"{error_context.error_type}: {error_context.error_message}")

        # Add to history
        self.error_history.append(error_context)

        # Save to file
        self._save_error_to_file(error_context)

        return error_context

    def _save_error_to_file(self, error_context: ErrorContext):
        """Save error context to log file.

        Args:
            error_context: Error context to save.
        """
        try:
            # Create log file directory if it doesn't exist
            log_path = Path(self.error_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dictionary and save as JSON line
            error_dict = {
                'operation': error_context.operation,
                'timestamp': error_context.timestamp,
                'attempt': error_context.attempt,
                'max_attempts': error_context.max_attempts,
                'error_type': error_context.error_type,
                'error_message': error_context.error_message,
                'traceback': error_context.traceback_str,
                'context_data': error_context.context_data
            }

            with open(log_path, 'a', encoding='utf-8') as f:
                json.dump(error_dict, f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            logger.error(f"Failed to save error to log file: {e}")

    def get_error_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent errors.

        Args:
            last_n: Number of recent errors to include.

        Returns:
            Dictionary with error summary statistics.
        """
        recent_errors = self.error_history[-last_n:] if self.error_history else []

        if not recent_errors:
            return {
                'total_errors': 0,
                'recent_errors': 0,
                'error_types': {},
                'operations': {},
                'recent_errors_list': []
            }

        # Count error types
        error_types = {}
        operations = {}

        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            operations[error.operation] = operations.get(error.operation, 0) + 1

        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_types': error_types,
            'operations': operations,
            'recent_errors_list': [
                {
                    'operation': e.operation,
                    'error_type': e.error_type,
                    'error_message': e.error_message[:100],
                    'timestamp': e.timestamp
                }
                for e in recent_errors
            ]
        }


class GracefulDegradation:
    """Handle graceful degradation when components fail."""

    def __init__(self):
        """Initialize graceful degradation handler."""
        self.fallback_configs = {}
        self.disabled_features = set()

    def register_fallback(self, feature: str, fallback_config: Dict[str, Any]):
        """Register a fallback configuration for a feature.

        Args:
            feature: Name of the feature.
            fallback_config: Fallback configuration to use.
        """
        self.fallback_configs[feature] = fallback_config
        logger.info(f"Registered fallback for feature: {feature}")

    def disable_feature(self, feature: str, reason: str):
        """Disable a feature due to failure.

        Args:
            feature: Name of the feature to disable.
            reason: Reason for disabling the feature.
        """
        self.disabled_features.add(feature)
        logger.warning(f"Disabled feature '{feature}': {reason}")

    def is_feature_available(self, feature: str) -> bool:
        """Check if a feature is available.

        Args:
            feature: Name of the feature to check.

        Returns:
            True if feature is available, False if disabled.
        """
        return feature not in self.disabled_features

    def get_config(self, feature: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration for a feature, using fallback if needed.

        Args:
            feature: Name of the feature.
            default_config: Default configuration.

        Returns:
            Configuration to use (default or fallback).
        """
        if feature in self.disabled_features:
            logger.warning(f"Feature '{feature}' is disabled")
            return {}

        if feature in self.fallback_configs:
            logger.info(f"Using fallback configuration for '{feature}'")
            return self.fallback_configs[feature]

        return default_config


class BatchProcessor:
    """Process items in batches with error recovery."""

    def __init__(self,
                 batch_size: int = 10,
                 max_failures_per_batch: int = 3,
                 error_handler: Optional[ErrorHandler] = None):
        """Initialize batch processor.

        Args:
            batch_size: Number of items per batch.
            max_failures_per_batch: Maximum failures allowed per batch.
            error_handler: Error handler instance.
        """
        self.batch_size = batch_size
        self.max_failures_per_batch = max_failures_per_batch
        self.error_handler = error_handler or ErrorHandler()

    def process_items(self,
                     items: List[Any],
                     process_func: Callable[[Any], Any],
                     context: str = "batch_processing") -> Dict[str, Any]:
        """Process items in batches with error recovery.

        Args:
            items: List of items to process.
            process_func: Function to process each item.
            context: Context description for error logging.

        Returns:
            Dictionary with processing results and statistics.
        """
        results = {
            'successful': [],
            'failed': [],
            'total_items': len(items),
            'processed_items': 0,
            'failed_items': 0,
            'failed_batches': 0,
            'processing_time': 0
        }

        start_time = time.time()

        # Process items in batches
        for batch_start in range(0, len(items), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(items))
            batch = items[batch_start:batch_end]

            batch_failures = 0

            logger.info(f"Processing batch {batch_start//self.batch_size + 1}: "
                       f"items {batch_start}-{batch_end-1}")

            for i, item in enumerate(batch):
                try:
                    result = process_func(item)
                    results['successful'].append({
                        'item_index': batch_start + i,
                        'item': item,
                        'result': result
                    })
                    results['processed_items'] += 1

                except Exception as e:
                    batch_failures += 1
                    results['failed_items'] += 1

                    error_context = self.error_handler.handle_error(
                        error=e,
                        operation=f"{context}_item_{batch_start + i}",
                        context_data={
                            'item_index': batch_start + i,
                            'batch_start': batch_start,
                            'batch_failures': batch_failures
                        }
                    )

                    results['failed'].append({
                        'item_index': batch_start + i,
                        'item': item,
                        'error': error_context.error_message,
                        'error_type': error_context.error_type
                    })

                    # Check if we should abort this batch
                    if batch_failures > self.max_failures_per_batch:
                        logger.error(f"Too many failures in batch ({batch_failures}), "
                                   f"skipping remaining items in this batch")
                        results['failed_batches'] += 1

                        # Mark remaining items in batch as failed
                        for j in range(i + 1, len(batch)):
                            results['failed'].append({
                                'item_index': batch_start + j,
                                'item': batch[j],
                                'error': 'Skipped due to batch failure threshold',
                                'error_type': 'BatchFailure'
                            })
                            results['failed_items'] += 1

                        break

        results['processing_time'] = time.time() - start_time
        results['success_rate'] = (results['processed_items'] / results['total_items']) * 100

        logger.info(f"Batch processing completed: {results['processed_items']}/{results['total_items']} "
                   f"items processed successfully ({results['success_rate']:.1f}%)")

        return results


# Global error handler instance
global_error_handler = ErrorHandler()


def log_error(operation: str, context_data: Optional[Dict[str, Any]] = None):
    """Decorator to automatically log errors with context.

    Args:
        operation: Description of the operation.
        context_data: Additional context information.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                global_error_handler.handle_error(
                    error=e,
                    operation=f"{operation}_{func.__name__}",
                    context_data=context_data
                )
                raise
        return wrapper
    return decorator


def safe_execute(func: Callable,
                default_return: Any = None,
                operation: str = "unknown_operation",
                log_errors: bool = True) -> Any:
    """Safely execute a function, returning default value on error.

    Args:
        func: Function to execute.
        default_return: Value to return on error.
        operation: Operation description for logging.
        log_errors: Whether to log errors.

    Returns:
        Function result or default value on error.
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            global_error_handler.handle_error(
                error=e,
                operation=operation
            )
        return default_return


# Initialize graceful degradation handler
graceful_degradation = GracefulDegradation()


class PerformanceMonitor:
    """Monitor performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            del self.start_times[operation]
            return duration
        return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get all performance metrics."""
        return self.metrics.copy()


class ResourceMonitor:
    """Monitor system resources."""

    def __init__(self):
        """Initialize resource monitor."""
        self.enabled = True
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.enabled = False
            logger.warning("psutil not available, resource monitoring disabled")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not self.enabled:
            return {"available": False}

        memory = self.psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not self.enabled:
            return 0.0
        return self.psutil.cpu_percent(interval=1)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system information (test compatibility)."""
        return self.get_system_info()

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        if not self.enabled:
            return {"available": False}

        return {
            "cpu_count": self.psutil.cpu_count(),
            "memory": self.get_memory_usage(),
            "cpu_percent": self.get_cpu_usage()
        }


class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self):
        """Initialize memory optimizer."""
        self.enabled = True

    def clear_cache(self) -> None:
        """Clear system caches."""
        import gc
        gc.collect()

    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        self.clear_cache()
        return {"optimization_performed": True}


class PerformanceOptimizer:
    """Performance optimization utilities."""

    def __init__(self):
        """Initialize performance optimizer."""
        self.gpu_available = False
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

    def optimize_for_gpu(self) -> Dict[str, Any]:
        """Optimize settings for GPU usage."""
        if self.gpu_available:
            return {"gpu_optimization": True, "device": "cuda"}
        return {"gpu_optimization": False, "device": "cpu"}

    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get recommended optimization settings."""
        return {
            "use_gpu": self.gpu_available,
            "batch_size_recommendation": 32 if self.gpu_available else 8,
            "memory_optimization": True
        }


# Alias for test compatibility
NeuronMapError = NeuronMapException
