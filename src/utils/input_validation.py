"""
Comprehensive Input Validation System for NeuronMap
==================================================

This module implements robust input validation with type checking, range validation,
semantic consistency checks and security validation for all user inputs.
"""

import re
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationError as PydanticValidationError
import logging

# Import our error handling
try:
    from .error_handling import ValidationError, NeuronMapException
except ImportError:
    class ValidationError(Exception):
        pass
    class NeuronMapException(Exception):
        pass

logger = logging.getLogger(__name__)


class AnalysisRequest(BaseModel):
    """Comprehensive validation for analysis requests."""

    model_name: str = Field(..., pattern=r'^[a-zA-Z0-9\-_/]+$',
                           description="Model name (alphanumeric, hyphens, underscores, slashes only)")
    input_texts: List[str] = Field(..., min_length=1, max_length=10000,
                                  description="List of input texts to analyze")
    layers: List[int] = Field(..., min_length=1, max_length=50,
                             description="List of layer indices to analyze")
    batch_size: int = Field(default=32, ge=1, le=512,
                           description="Batch size for processing")
    max_length: Optional[int] = Field(default=512, ge=1, le=8192,
                                     description="Maximum sequence length")
    device: str = Field(default="auto", pattern=r'^(auto|cpu|cuda|cuda:\d+)$',
                       description="Device for computation")
    output_format: str = Field(default="csv", pattern=r'^(csv|json|pickle)$',
                              description="Output format")

    @field_validator('input_texts')
    @classmethod
    def validate_text_content(cls, v):
        """Validate text content for safety and quality."""
        for i, text in enumerate(v):
            # Check for empty text
            if len(text.strip()) == 0:
                raise ValueError(f"Text {i}: Empty text not allowed")

            # Check for excessive length
            if len(text) > 10000:
                raise ValueError(f"Text {i}: Text too long (max 10000 chars)")

            # Check for potential security issues
            dangerous_patterns = [
                r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
                r'javascript:',  # JavaScript URLs
                r'data:.*base64',  # Base64 data URLs
                r'file:\/\/',  # File URLs
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    raise ValueError(f"Text {i}: Potentially unsafe content detected")

        return v

    @field_validator('layers')
    @classmethod
    def validate_layers(cls, v):
        """Validate layer specifications."""
        # Check for negative layers
        if any(layer < 0 for layer in v):
            raise ValueError("Layer indices must be non-negative")

        # Check for excessively high layer numbers
        if any(layer > 100 for layer in v):
            raise ValueError("Layer indices seem too high (max reasonable: 100)")

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate layer indices not allowed")

        return sorted(v)  # Return sorted layers


class ModelValidationRequest(BaseModel):
    """Validation for model-related requests."""

    model_name: str = Field(..., description="Model name to validate")
    model_type: Optional[str] = Field(default=None,
                                     pattern=r'^(gpt|bert|t5|llama|auto)$',
                                     description="Model type hint")
    cache_dir: Optional[str] = Field(default=None, description="Model cache directory")

    @field_validator('model_name')
    @classmethod
    def validate_model_name_security(cls, v):
        """Security validation for model names."""
        # Prevent path traversal
        if '..' in v or '/' in v and not v.startswith('huggingface/'):
            if not re.match(r'^[a-zA-Z0-9\-_./]+$', v):
                raise ValueError("Model name contains invalid characters")

        # Check length
        if len(v) > 200:
            raise ValueError("Model name too long")

        return v


class FileValidationRequest(BaseModel):
    """Validation for file-related operations."""

    file_path: str = Field(..., description="Path to file")
    operation: str = Field(..., pattern=r'^(read|write|append)$',
                          description="File operation type")
    max_size_mb: Optional[float] = Field(default=100.0, ge=0.1, le=1000.0,
                                        description="Maximum file size in MB")
    allowed_extensions: Optional[List[str]] = Field(default=None,
                                                   description="Allowed file extensions")

    @field_validator('file_path')
    @classmethod
    def validate_file_path_security(cls, v):
        """Security validation for file paths."""
        path = Path(v)

        # Prevent path traversal
        try:
            path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            raise ValueError("File path outside working directory not allowed")

        # Check for dangerous paths
        dangerous_paths = ['/etc', '/proc', '/sys', '/dev', '/var/log']
        resolved_path = str(path.resolve())

        for dangerous in dangerous_paths:
            if resolved_path.startswith(dangerous):
                raise ValueError(f"Access to {dangerous} not allowed")

        return v


class InputValidator:
    """Comprehensive input validation system."""

    def __init__(self):
        self.validation_cache = {}
        self.known_models = {
            'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
            'bert-base-uncased', 'bert-large-uncased',
            'distilbert-base-uncased', 'roberta-base',
            't5-small', 't5-base', 't5-large'
        }

    def validate_analysis_request(self, request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a complete analysis request.

        Args:
            request_data: Dictionary containing request parameters

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            # Use Pydantic for comprehensive validation
            validated_request = AnalysisRequest(**request_data)

            # Additional business logic validation
            errors = []

            # Model availability check
            if not self.is_model_available(validated_request.model_name):
                errors.append(f"Model '{validated_request.model_name}' is not available")

            # Resource feasibility check
            estimated_memory = self.estimate_memory_usage(
                validated_request.model_name,
                validated_request.batch_size,
                validated_request.max_length
            )

            if estimated_memory > self.get_available_memory():
                errors.append(f"Estimated memory usage ({estimated_memory:.1f}GB) exceeds available memory")

            # Layer compatibility check
            max_layers = self.get_model_max_layers(validated_request.model_name)
            if max_layers and any(layer >= max_layers for layer in validated_request.layers):
                errors.append(f"Some layer indices exceed model's maximum layers ({max_layers})")

            return len(errors) == 0, errors

        except PydanticValidationError as e:
            error_messages = []
            for error in e.errors():
                field = " -> ".join(str(x) for x in error['loc'])
                message = error['msg']
                error_messages.append(f"{field}: {message}")

            return False, error_messages

        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return False, [f"Validation failed: {str(e)}"]

    def validate_file_operation(self, file_path: str, operation: str,
                               max_size_mb: float = 100.0) -> Tuple[bool, List[str]]:
        """
        Validate file operations for security and feasibility.

        Args:
            file_path: Path to the file
            operation: Type of operation (read/write/append)
            max_size_mb: Maximum allowed file size in MB

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            request = FileValidationRequest(
                file_path=file_path,
                operation=operation,
                max_size_mb=max_size_mb
            )

            errors = []
            path = Path(file_path)

            # Check file existence for read operations
            if operation == 'read' and not path.exists():
                errors.append(f"File does not exist: {file_path}")

            # Check file size
            if path.exists():
                file_size_mb = path.stat().st_size / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    errors.append(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")

            # Check directory permissions
            parent_dir = path.parent
            if operation in ['write', 'append']:
                if not parent_dir.exists():
                    try:
                        parent_dir.mkdir(parents=True, exist_ok=True)
                    except PermissionError:
                        errors.append(f"Cannot create directory: {parent_dir}")
                elif not os.access(parent_dir, os.W_OK):
                    errors.append(f"No write permission for directory: {parent_dir}")

            return len(errors) == 0, errors

        except PydanticValidationError as e:
            return False, [str(e)]
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, [f"File validation failed: {str(e)}"]

    def validate_model_request(self, model_name: str, model_type: str = None) -> Tuple[bool, List[str]]:
        """
        Validate model-related requests.

        Args:
            model_name: Name of the model
            model_type: Optional model type hint

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            request = ModelValidationRequest(
                model_name=model_name,
                model_type=model_type
            )

            errors = []

            # Check if model is in known models or follows known patterns
            if not self.is_model_name_valid(model_name):
                errors.append(f"Model name '{model_name}' does not follow expected patterns")

            # Check model type consistency
            if model_type:
                if not self.is_model_type_consistent(model_name, model_type):
                    errors.append(f"Model name '{model_name}' is not consistent with type '{model_type}'")

            return len(errors) == 0, errors

        except PydanticValidationError as e:
            return False, [str(e)]
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False, [f"Model validation failed: {str(e)}"]

    def sanitize_input(self, input_data: Any) -> Any:
        """
        Sanitize input data to remove potential security risks.

        Args:
            input_data: Input data to sanitize

        Returns:
            Sanitized input data
        """
        if isinstance(input_data, str):
            # Remove null bytes
            input_data = input_data.replace('\x00', '')

            # Remove excessive whitespace
            input_data = re.sub(r'\s+', ' ', input_data).strip()

            # Limit length
            if len(input_data) > 10000:
                input_data = input_data[:10000]
                logger.warning("Input truncated due to excessive length")

        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]

        elif isinstance(input_data, dict):
            return {key: self.sanitize_input(value) for key, value in input_data.items()}

        return input_data

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available for use."""
        # Check known models
        if model_name in self.known_models:
            return True

        # Check common patterns
        patterns = [
            r'^gpt2(-\w+)?$',
            r'^bert-\w+-\w+$',
            r'^distilbert-\w+-\w+$',
            r'^roberta-\w+$',
            r'^t5-\w+$',
            r'^\w+/\w+$'  # HuggingFace format
        ]

        return any(re.match(pattern, model_name) for pattern in patterns)

    def is_model_name_valid(self, model_name: str) -> bool:
        """Validate model name format."""
        return bool(re.match(r'^[a-zA-Z0-9\-_./]+$', model_name))

    def is_model_type_consistent(self, model_name: str, model_type: str) -> bool:
        """Check if model name is consistent with model type."""
        type_patterns = {
            'gpt': r'gpt',
            'bert': r'bert',
            't5': r't5',
            'llama': r'llama'
        }

        if model_type in type_patterns:
            return bool(re.search(type_patterns[model_type], model_name, re.IGNORECASE))

        return True  # Auto type is always consistent

    def estimate_memory_usage(self, model_name: str, batch_size: int, max_length: int) -> float:
        """Estimate memory usage in GB."""
        # Simple heuristic-based estimation
        base_memory = {
            'gpt2': 0.5,
            'gpt2-medium': 1.0,
            'gpt2-large': 2.0,
            'gpt2-xl': 4.0,
            'bert-base-uncased': 0.5,
            'bert-large-uncased': 1.0,
            't5-small': 0.3,
            't5-base': 0.8,
            't5-large': 2.0
        }

        base = base_memory.get(model_name, 1.0)  # Default 1GB
        scaling_factor = (batch_size * max_length) / (32 * 512)  # Normalize to default

        return base * max(1.0, scaling_factor)

    def get_available_memory(self) -> float:
        """Get available system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 8.0  # Conservative default

    def get_model_max_layers(self, model_name: str) -> Optional[int]:
        """Get maximum number of layers for a model."""
        layer_counts = {
            'gpt2': 12,
            'gpt2-medium': 24,
            'gpt2-large': 36,
            'gpt2-xl': 48,
            'bert-base-uncased': 12,
            'bert-large-uncased': 24,
            'distilbert-base-uncased': 6,
            't5-small': 6,
            't5-base': 12,
            't5-large': 24
        }

        return layer_counts.get(model_name)


# Convenience functions for backward compatibility
def validate_analysis_request(request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate analysis request - convenience function."""
    validator = InputValidator()
    return validator.validate_analysis_request(request_data)


def validate_file_path(file_path: str, operation: str = 'read') -> bool:
    """Validate file path - convenience function."""
    validator = InputValidator()
    is_valid, _ = validator.validate_file_operation(file_path, operation)
    return is_valid


def sanitize_user_input(input_data: Any) -> Any:
    """Sanitize user input - convenience function."""
    validator = InputValidator()
    return validator.sanitize_input(input_data)
