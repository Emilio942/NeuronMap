#!/usr/bin/env python3
"""
Advanced Structured Logging System for NeuronMap

This module provides comprehensive structured logging capabilities with:
- Multi-level logging strategies (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured logging with JSON format for better analysis
- Performance monitoring and security audit trails
- Log rotation and retention policies
- Automated alerting and monitoring support
"""

import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import uuid
from enum import Enum

class LogLevel(Enum):
    """Enhanced log levels with specific use cases."""
    DEBUG = "DEBUG"      # Detailed execution traces for development
    INFO = "INFO"        # User actions and system events
    WARNING = "WARNING"  # Recoverable errors and performance issues
    ERROR = "ERROR"      # Critical failures requiring intervention
    CRITICAL = "CRITICAL" # System-down scenarios
    AUDIT = "AUDIT"      # Security and compliance events
    PERFORMANCE = "PERFORMANCE"  # Performance metrics and profiling


@dataclass
class LogEvent:
    """Structured log event with metadata."""
    timestamp: str
    level: str
    message: str
    component: str
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    input_size: Optional[int] = None
    duration_ms: Optional[float] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Extract structured data from record
        event_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'component': getattr(record, 'component', 'unknown'),
            'event_type': getattr(record, 'event_type', 'generic'),
            'logger_name': record.name,
            'filename': record.filename,
            'line_number': record.lineno,
            'function_name': record.funcName,
            'process_id': record.process,
            'thread_id': record.thread,
        }

        # Add optional fields if present
        optional_fields = [
            'user_id', 'session_id', 'request_id', 'model_name',
            'input_size', 'duration_ms', 'error_code', 'metadata'
        ]
        
        for field in optional_fields:
            if hasattr(record, field):
                event_data[field] = getattr(record, field)

        # Add exception information if present
        if record.exc_info:
            event_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }

        return json.dumps(event_data, ensure_ascii=False)


class PerformanceLogger:
    """Specialized logger for performance monitoring."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.active_operations = {}
        self.lock = threading.Lock()

    @contextmanager
    def log_operation(self, operation_name: str, **metadata):
        """Context manager for logging operation performance."""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        with self.lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'metadata': metadata
            }

        # Log operation start
        self.logger.info(
            f"Operation started: {operation_name}",
            extra={
                'component': 'performance',
                'event_type': 'operation_start',
                'operation_id': operation_id,
                'operation_name': operation_name,
                'metadata': metadata
            }
        )

        try:
            yield operation_id
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            with self.lock:
                if operation_id in self.active_operations:
                    del self.active_operations[operation_id]

            # Log operation completion
            self.logger.info(
                f"Operation completed: {operation_name} ({duration_ms:.2f}ms)",
                extra={
                    'component': 'performance',
                    'event_type': 'operation_complete',
                    'operation_id': operation_id,
                    'operation_name': operation_name,
                    'duration_ms': duration_ms,
                    'metadata': metadata
                }
            )

    def log_metric(self, metric_name: str, value: Union[int, float], unit: str = "", **metadata):
        """Log a performance metric."""
        self.logger.info(
            f"Metric: {metric_name} = {value} {unit}",
            extra={
                'component': 'performance',
                'event_type': 'metric',
                'metric_name': metric_name,
                'metric_value': value,
                'metric_unit': unit,
                'metadata': metadata
            }
        )


class SecurityAuditLogger:
    """Specialized logger for security and audit events."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_user_action(self, user_id: str, action: str, resource: str = "", **metadata):
        """Log user action for audit trail."""
        self.logger.info(
            f"User action: {action}",
            extra={
                'component': 'security',
                'event_type': 'user_action',
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'metadata': metadata
            }
        )

    def log_authentication(self, user_id: str, success: bool, ip_address: str = "", **metadata):
        """Log authentication attempt."""
        status = "success" if success else "failure"
        self.logger.info(
            f"Authentication {status} for user {user_id}",
            extra={
                'component': 'security',
                'event_type': 'authentication',
                'user_id': user_id,
                'success': success,
                'ip_address': ip_address,
                'metadata': metadata
            }
        )

    def log_security_event(self, event_type: str, severity: str, description: str, **metadata):
        """Log security event."""
        level = getattr(logging, severity.upper(), logging.WARNING)
        self.logger.log(
            level,
            f"Security event: {description}",
            extra={
                'component': 'security',
                'event_type': event_type,
                'security_severity': severity,
                'metadata': metadata
            }
        )


class NeuronMapLogger:
    """Main structured logging system for NeuronMap."""

    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 10,
                 enable_console: bool = True,
                 enable_structured: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger('neuronmap')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set up handlers
        self._setup_file_handlers(max_file_size, backup_count, enable_structured)
        
        if enable_console:
            self._setup_console_handler(enable_structured)

        # Prevent double logging when root logger also receives handlers
        self.logger.propagate = False

        # Mirror handlers to root logger so other components inherit configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.logger.level)
        root_logger.handlers.clear()
        for handler in self.logger.handlers:
            root_logger.addHandler(handler)
        
        # Initialize specialized loggers
        self.performance = PerformanceLogger(self.logger)
        self.security = SecurityAuditLogger(self.logger)
        
        # Log system initialization
        self.log_analysis_event("logging_system_initialized", {
            'log_dir': str(self.log_dir),
            'log_level': log_level,
            'structured_logging': enable_structured
        })

    def _setup_file_handlers(self, max_file_size: int, backup_count: int, structured: bool):
        """Set up rotating file handlers."""
        # Main application log
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'neuronmap.log',
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Error log (WARNING and above)
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'neuronmap_errors.log',
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)
        
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        main_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)

    def _setup_console_handler(self, structured: bool):
        """Set up console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
        
        self.logger.addHandler(console_handler)

    def log_analysis_start(self, model_name: str, input_size: int, user_id: str = "", **metadata):
        """Log the start of an analysis operation."""
        self.logger.info(
            f"Analysis started for model {model_name}",
            extra={
                'component': 'analysis',
                'event_type': 'analysis_start',
                'model_name': model_name,
                'input_size': input_size,
                'user_id': user_id,
                'metadata': metadata
            }
        )

    def log_analysis_complete(self, model_name: str, duration_ms: float, 
                            result_size: int = 0, user_id: str = "", **metadata):
        """Log the completion of an analysis operation."""
        self.logger.info(
            f"Analysis completed for model {model_name} in {duration_ms:.2f}ms",
            extra={
                'component': 'analysis',
                'event_type': 'analysis_complete',
                'model_name': model_name,
                'duration_ms': duration_ms,
                'result_size': result_size,
                'user_id': user_id,
                'metadata': metadata
            }
        )

    def log_analysis_error(self, model_name: str, error: Exception, user_id: str = "", **metadata):
        """Log an analysis error."""
        self.logger.error(
            f"Analysis failed for model {model_name}: {error}",
            extra={
                'component': 'analysis',
                'event_type': 'analysis_error',
                'model_name': model_name,
                'error_code': type(error).__name__,
                'user_id': user_id,
                'metadata': metadata
            },
            exc_info=True
        )

    def log_analysis_event(self, event_type: str, metadata: Dict[str, Any], 
                          level: str = "INFO", user_id: str = ""):
        """Log a generic analysis event."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(
            log_level,
            f"Analysis event: {event_type}",
            extra={
                'component': 'analysis',
                'event_type': event_type,
                'user_id': user_id,
                'metadata': metadata
            }
        )

    def log_system_event(self, event_type: str, message: str, 
                        level: str = "INFO", **metadata):
        """Log a system-level event."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(
            log_level,
            message,
            extra={
                'component': 'system',
                'event_type': event_type,
                'metadata': metadata
            }
        )

    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'log_directory': str(self.log_dir),
            'logger_name': self.logger.name,
            'logger_level': self.logger.level,
            'handler_count': len(self.logger.handlers),
            'log_files': [f.name for f in self.log_dir.glob('*.log')]
        }


# Global logger instance
_logger_instance: Optional[NeuronMapLogger] = None


def get_logger() -> NeuronMapLogger:
    """Get the global NeuronMap logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = NeuronMapLogger()
    return _logger_instance


def configure_logging(log_dir: str = "logs", 
                     log_level: str = "INFO",
                     structured: bool = True,
                     console: bool = True) -> NeuronMapLogger:
    """Configure the global logging system."""
    global _logger_instance
    _logger_instance = NeuronMapLogger(
        log_dir=log_dir,
        log_level=log_level,
        enable_structured=structured,
        enable_console=console
    )
    return _logger_instance


# Convenience functions for common logging patterns
def log_analysis_start(model_name: str, input_size: int, user_id: str = "", **metadata):
    """Convenience function for logging analysis start."""
    get_logger().log_analysis_start(model_name, input_size, user_id, **metadata)


def log_analysis_complete(model_name: str, duration_ms: float, 
                         result_size: int = 0, user_id: str = "", **metadata):
    """Convenience function for logging analysis completion."""
    get_logger().log_analysis_complete(model_name, duration_ms, result_size, user_id, **metadata)


def log_analysis_error(model_name: str, error: Exception, user_id: str = "", **metadata):
    """Convenience function for logging analysis errors."""
    get_logger().log_analysis_error(model_name, error, user_id, **metadata)


if __name__ == "__main__":
    """CLI interface for testing the structured logging system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuronMap Structured Logging System")
    parser.add_argument("--test", action="store_true", help="Run logging system tests")
    parser.add_argument("--log-dir", default="logs", help="Log directory path")
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--structured", action="store_true", default=True, help="Enable structured JSON logging")
    parser.add_argument("--console", action="store_true", default=True, help="Enable console logging")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing NeuronMap Structured Logging System...")
        
        # Configure logging
        logger = configure_logging(
            log_dir=args.log_dir,
            log_level=args.log_level,
            structured=args.structured,
            console=args.console
        )
        
        # Test different log levels
        logger.log_system_event("test_start", "Starting logging system tests", level="INFO")
        
        # Test analysis logging
        with logger.performance.log_operation("test_analysis", model="test-gpt2"):
            logger.log_analysis_start("test-gpt2", 100, "test-user-123")
            time.sleep(0.1)  # Simulate work
            logger.log_analysis_complete("test-gpt2", 100.0, 50, "test-user-123")
        
        # Test performance logging
        logger.performance.log_metric("inference_speed", 25.5, "tokens/sec", model="test-gpt2")
        
        # Test security logging
        logger.security.log_user_action("test-user-123", "model_analysis", "test-gpt2")
        logger.security.log_authentication("test-user-123", True, "127.0.0.1")
        
        # Test error logging
        try:
            raise ValueError("Test error for logging demonstration")
        except Exception as e:
            logger.log_analysis_error("test-gpt2", e, "test-user-123")
        
        # Test different log levels
        logger.log_system_event("debug_event", "Debug level test", level="DEBUG")
        logger.log_system_event("warning_event", "Warning level test", level="WARNING")
        logger.log_system_event("error_event", "Error level test", level="ERROR")
        
        logger.log_system_event("test_complete", "Logging system tests completed", level="INFO")
        
        # Display statistics
        stats = logger.get_log_stats()
        print(f"\nLogging Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nLogs written to: {args.log_dir}")
        print("Test completed successfully!")
    
    else:
        print("NeuronMap Structured Logging System")
        print("Use --test to run system tests")
