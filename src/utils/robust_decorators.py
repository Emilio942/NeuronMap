"""
Production-Ready Exception Handling Decorator System
===================================================

This module provides comprehensive exception handling decorators that can be
applied to any function or method to ensure robust error handling, logging,
and recovery mechanisms.
"""

import functools
import logging
import time
import traceback
from typing import Callable, Any, Optional, Dict, Union, Type
from pathlib import Path

# Import our error handling system
try:
    from .error_handling import (
        NeuronMapException, ModelLoadingError, ActivationExtractionError,
        DataProcessingError, ValidationError, ConfigurationError
    )
except ImportError:
    # Fallback definitions
    class NeuronMapException(Exception):
        pass
    class ModelLoadingError(NeuronMapException):
        pass
    class ActivationExtractionError(NeuronMapException):
        pass
    class DataProcessingError(NeuronMapException):
        pass
    class ValidationError(NeuronMapException):
        pass
    class ConfigurationError(NeuronMapException):
        pass

logger = logging.getLogger(__name__)


def robust_execution(
    fallback_return=None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exceptions_to_retry: tuple = (ConnectionError, TimeoutError),
    exceptions_to_catch: tuple = (Exception,),
    log_errors: bool = True,
    raise_on_failure: bool = True
):
    """
    Comprehensive exception handling decorator with retry logic and fallback options.

    Args:
        fallback_return: Value to return if all retries fail and raise_on_failure=False
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        exceptions_to_retry: Tuple of exception types that should trigger retries
        exceptions_to_catch: Tuple of exception types to catch and handle
        log_errors: Whether to log errors
        raise_on_failure: Whether to raise exception after all retries fail

    Returns:
        Decorated function with robust error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions_to_retry as e:
                    last_exception = e
                    if attempt < max_retries:
                        if log_errors:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {retry_delay}s..."
                            )
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        if log_errors:
                            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
                        break

                except exceptions_to_catch as e:
                    last_exception = e
                    if log_errors:
                        logger.error(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                    break

                except Exception as e:
                    last_exception = e
                    if log_errors:
                        logger.error(
                            f"Unexpected error in {func.__name__}: {e}"
                        )
                    break

            # Handle failure case
            if raise_on_failure:
                if isinstance(last_exception, NeuronMapException):
                    raise last_exception
                else:
                    raise NeuronMapException(
                        f"Function {func.__name__} failed after {max_retries + 1} attempts: {last_exception}",
                        error_code="FUNCTION_EXECUTION_ERROR",
                        context={
                            'function': func.__name__,
                            'attempts': max_retries + 1,
                            'last_error': str(last_exception)
                        }
                    )
            else:
                return fallback_return

        return wrapper
    return decorator


def model_operation_handler(fallback_model: str = "gpt2"):
    """
    Specialized decorator for model-related operations with automatic fallback.

    Args:
        fallback_model: Model to fall back to if the primary model fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except (ModelLoadingError, ImportError, FileNotFoundError) as e:
                logger.warning(f"Model operation failed: {e}. Attempting fallback to {fallback_model}")

                # Try to modify kwargs to use fallback model
                if 'model_name' in kwargs:
                    kwargs['model_name'] = fallback_model
                elif len(args) > 0 and hasattr(args[0], 'model_name'):
                    # Handle class methods where first arg might have model_name
                    args[0].model_name = fallback_model

                try:
                    return func(*args, **kwargs)
                except Exception as fallback_error:
                    raise ModelLoadingError(
                        f"Both primary and fallback model failed: {e}, {fallback_error}",
                        error_code="MODEL_FALLBACK_FAILED",
                        context={'primary_error': str(e), 'fallback_error': str(fallback_error)}
                    )

            except Exception as e:
                raise ModelLoadingError(
                    f"Unexpected model operation error: {e}",
                    error_code="MODEL_OPERATION_ERROR",
                    context={'function': func.__name__, 'error': str(e)}
                )

        return wrapper
    return decorator


def data_processing_handler(partial_results: bool = True):
    """
    Specialized decorator for data processing operations with partial result support.

    Args:
        partial_results: Whether to return partial results on failure
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except (DataProcessingError, ValidationError) as e:
                if partial_results:
                    logger.warning(f"Data processing partially failed: {e}. Returning partial results.")

                    # Try to return whatever partial results we can
                    if hasattr(e, 'context') and 'partial_data' in e.context:
                        return e.context['partial_data']
                    else:
                        return {'status': 'partial_failure', 'error': str(e), 'data': None}
                else:
                    raise e

            except Exception as e:
                logger.error(f"Data processing failed completely: {e}")

                if partial_results:
                    return {'status': 'complete_failure', 'error': str(e), 'data': None}
                else:
                    raise DataProcessingError(
                        f"Data processing error: {e}",
                        error_code="DATA_PROCESSING_ERROR",
                        context={'function': func.__name__, 'error': str(e)}
                    )

        return wrapper
    return decorator


def file_operation_handler(create_missing_dirs: bool = True, backup_on_write: bool = False):
    """
    Specialized decorator for file operations with automatic directory creation and backup.

    Args:
        create_missing_dirs: Whether to create missing directories automatically
        backup_on_write: Whether to create backup files before writing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Extract file path from args/kwargs
                file_path = None
                if 'file_path' in kwargs:
                    file_path = kwargs['file_path']
                elif 'filename' in kwargs:
                    file_path = kwargs['filename']
                elif len(args) > 0:
                    file_path = args[0]

                if file_path and create_missing_dirs:
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

                if file_path and backup_on_write and 'write' in func.__name__.lower():
                    if Path(file_path).exists():
                        backup_path = f"{file_path}.backup"
                        Path(file_path).rename(backup_path)
                        logger.info(f"Created backup: {backup_path}")

                return func(*args, **kwargs)

            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.error(f"File operation failed: {e}")
                raise DataProcessingError(
                    f"File operation error: {e}",
                    error_code="FILE_OPERATION_ERROR",
                    context={'function': func.__name__, 'file_path': file_path, 'error': str(e)}
                )

            except Exception as e:
                logger.error(f"Unexpected file operation error: {e}")
                raise DataProcessingError(
                    f"Unexpected file error: {e}",
                    error_code="FILE_OPERATION_UNEXPECTED_ERROR",
                    context={'function': func.__name__, 'error': str(e)}
                )

        return wrapper
    return decorator


def gpu_operation_handler(fallback_to_cpu: bool = True):
    """
    Specialized decorator for GPU operations with automatic CPU fallback.

    Args:
        fallback_to_cpu: Whether to fall back to CPU if GPU operations fail
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except (RuntimeError, ImportError) as e:
                error_msg = str(e).lower()
                is_gpu_error = any(keyword in error_msg for keyword in
                                 ['cuda', 'gpu', 'out of memory', 'device'])

                if is_gpu_error and fallback_to_cpu:
                    logger.warning(f"GPU operation failed: {e}. Falling back to CPU.")

                    # Modify device setting to CPU
                    if 'device' in kwargs:
                        kwargs['device'] = 'cpu'
                    elif hasattr(args[0], 'device'):
                        args[0].device = 'cpu'

                    try:
                        return func(*args, **kwargs)
                    except Exception as cpu_error:
                        raise ActivationExtractionError(
                            f"Both GPU and CPU execution failed: {e}, {cpu_error}",
                            error_code="GPU_CPU_FALLBACK_FAILED",
                            context={'gpu_error': str(e), 'cpu_error': str(cpu_error)}
                        )
                else:
                    raise ActivationExtractionError(
                        f"GPU operation error: {e}",
                        error_code="GPU_OPERATION_ERROR",
                        context={'function': func.__name__, 'error': str(e)}
                    )

            except Exception as e:
                logger.error(f"Unexpected GPU operation error: {e}")
                raise ActivationExtractionError(
                    f"Unexpected GPU error: {e}",
                    error_code="GPU_OPERATION_UNEXPECTED_ERROR",
                    context={'function': func.__name__, 'error': str(e)}
                )

        return wrapper
    return decorator


# Convenience decorators for common use cases
safe_execution = robust_execution(
    max_retries=0,
    raise_on_failure=False,
    log_errors=True
)

network_operation = robust_execution(
    max_retries=3,
    retry_delay=1.0,
    exceptions_to_retry=(ConnectionError, TimeoutError),
    log_errors=True
)

critical_operation = robust_execution(
    max_retries=0,
    raise_on_failure=True,
    log_errors=True
)
