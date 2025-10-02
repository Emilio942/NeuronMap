"""
Comprehensive Input Validation System for NeuronMap
=================================================

This module provides robust input validation with type checking, range validation,
semantic consistency checks, and security validation for all user inputs.
"""

import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pydantic import BaseModel, validator, field_validator, Field, ValidationError as PydanticValidationError
from transformers import AutoTokenizer, AutoModel
import json
import yaml
from dataclasses import dataclass
from enum import Enum

from .error_handling import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    STANDARD = "standard"
    LENIENT = "lenient"


@dataclass
class ValidationResult:
    """Result of validation with details."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def add_suggestion(self, message: str):
        """Add a suggestion."""
        self.suggestions.append(message)


class AnalysisRequest(BaseModel):
    """Validated request for neural analysis."""
    model_name: str = Field(..., pattern=r'^[a-zA-Z0-9\-_/]+$')
    input_texts: List[str] = Field(..., min_length=1, max_length=10000)
    layers: List[Union[str, int]] = Field(..., min_length=1, max_length=50)
    batch_size: int = Field(default=32, ge=1, le=512)
    device: str = Field(default="auto", pattern=r'^(auto|cpu|cuda|cuda:\d+)$')
    max_length: int = Field(default=512, ge=1, le=32768)

    @field_validator('input_texts')
    @classmethod
    def validate_text_content(cls, v):
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")
            if len(text.strip()) == 0:
                raise ValueError(f"Text at index {i} cannot be empty")
            if len(text) > 50000:
                raise ValueError(f"Text at index {i} too long (max 50000 chars), got {len(text)}")
            if contains_malicious_content(text):
                raise ValueError(f"Text at index {i} contains potentially malicious content")
        return v

    @field_validator('layers')
    @classmethod
    def validate_layers(cls, v):
        for i, layer in enumerate(v):
            if isinstance(layer, str):
                if not layer.strip():
                    raise ValueError(f"Layer name at index {i} cannot be empty")
                if len(layer) > 200:
                    raise ValueError(f"Layer name at index {i} too long")
            elif isinstance(layer, int):
                if layer < 0 or layer > 100:
                    raise ValueError(f"Layer index at index {i} must be between 0 and 100")
        return v


class ModelValidator:
    """Validates model-related inputs and configurations."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.supported_architectures = {
            'gpt2', 'bert', 'roberta', 'distilbert', 't5', 'llama', 'opt', 'bloom'
        }

    def validate_model_name(self, model_name: str) -> ValidationResult:
        """
        Validate model name with comprehensive checks.

        Args:
            model_name: Model name to validate

        Returns:
            ValidationResult with detailed feedback
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Basic validation
        if not model_name or not isinstance(model_name, str):
            result.add_error("Model name must be a non-empty string")
            return result

        model_name = model_name.strip()

        # Length validation
        if len(model_name) > 200:
            result.add_error(f"Model name too long (max 200 chars): {len(model_name)}")

        # Pattern validation
        if not re.match(r'^[a-zA-Z0-9_/-]+$', model_name):
            result.add_error("Model name contains invalid characters. Only alphanumeric, hyphens, underscores, and slashes allowed")

        # Security validation
        if contains_malicious_content(model_name):
            result.add_error("Model name contains potentially malicious content")

        # Architecture validation
        architecture_detected = False
        for arch in self.supported_architectures:
            if arch in model_name.lower():
                architecture_detected = True
                break

        if not architecture_detected and self.validation_level == ValidationLevel.STRICT:
            result.add_warning(f"Model architecture not recognized. Supported: {', '.join(self.supported_architectures)}")

        # Availability check (if not strict mode to avoid network calls)
        if self.validation_level != ValidationLevel.LENIENT:
            if not self._check_model_availability(model_name):
                result.add_error(f"Model '{model_name}' not found or not accessible")
                result.add_suggestion("Check model name spelling and HuggingFace Hub availability")

        return result

    def _check_model_availability(self, model_name: str) -> bool:
        """Check if model is available without downloading."""
        try:
            # Quick availability check using tokenizer (faster than full model)
            AutoTokenizer.from_pretrained(model_name, use_fast=False)
            return True
        except Exception:
            return False

    def validate_layer_specification(self, layers: List[Union[str, int]], model_name: str) -> ValidationResult:
        """
        Validate layer specifications against model architecture.

        Args:
            layers: List of layer names or indices
            model_name: Model name for architecture checking

        Returns:
            ValidationResult with layer validation details
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        if not layers:
            result.add_error("At least one layer must be specified")
            return result

        # Check for duplicates
        str_layers = [str(layer) for layer in layers]
        if len(str_layers) != len(set(str_layers)):
            result.add_warning("Duplicate layers specified")

        # Validate each layer
        for i, layer in enumerate(layers):
            if isinstance(layer, str):
                self._validate_layer_name(layer, i, result)
            elif isinstance(layer, int):
                self._validate_layer_index(layer, i, result, model_name)
            else:
                result.add_error(f"Layer at index {i} must be string or integer, got {type(layer)}")

        return result

    def _validate_layer_name(self, layer_name: str, index: int, result: ValidationResult):
        """Validate a layer name."""
        if not layer_name.strip():
            result.add_error(f"Layer name at index {index} cannot be empty")
            return

        if len(layer_name) > 200:
            result.add_error(f"Layer name at index {index} too long (max 200 chars)")

        # Check for common layer naming patterns
        common_patterns = [
            r'.*\.attention\..*',
            r'.*\.mlp\..*',
            r'.*\.h\.\d+\..*',  # GPT-style
            r'.*\.layer\.\d+\..*',  # BERT-style
            r'.*\.block\.\d+\..*',  # T5-style
        ]

        if not any(re.match(pattern, layer_name) for pattern in common_patterns):
            result.add_warning(f"Layer name '{layer_name}' doesn't match common patterns")
            result.add_suggestion("Common patterns: transformer.h.0.attn, encoder.layer.0.attention")

    def _validate_layer_index(self, layer_index: int, index: int, result: ValidationResult, model_name: str):
        """Validate a layer index."""
        if layer_index < 0:
            result.add_error(f"Layer index at position {index} cannot be negative: {layer_index}")

        if layer_index > 100:
            result.add_warning(f"Layer index {layer_index} seems very high. Most models have <50 layers")

        # Model-specific validation
        if 'gpt2' in model_name.lower() and layer_index >= 12:
            result.add_warning(f"GPT-2 typically has 12 layers, index {layer_index} may be invalid")
        elif 'bert-base' in model_name.lower() and layer_index >= 12:
            result.add_warning(f"BERT-base typically has 12 layers, index {layer_index} may be invalid")


class TextValidator:
    """Validates text inputs for neural analysis."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.max_single_text_length = 50000
        self.max_total_texts = 10000

    def validate_text_inputs(self, texts: List[str]) -> ValidationResult:
        """
        Validate a list of text inputs.

        Args:
            texts: List of text strings to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        if not texts:
            result.add_error("At least one text input is required")
            return result

        if not isinstance(texts, list):
            result.add_error("Texts must be provided as a list")
            return result

        if len(texts) > self.max_total_texts:
            result.add_error(f"Too many texts: {len(texts)} (max {self.max_total_texts})")

        # Validate each text
        total_length = 0
        empty_texts = 0

        for i, text in enumerate(texts):
            text_result = self._validate_single_text(text, i)

            # Merge results
            result.errors.extend(text_result.errors)
            result.warnings.extend(text_result.warnings)
            result.suggestions.extend(text_result.suggestions)

            if not text_result.is_valid:
                result.is_valid = False

            if isinstance(text, str):
                total_length += len(text)
                if len(text.strip()) == 0:
                    empty_texts += 1

        # Overall statistics
        if empty_texts > 0:
            result.add_warning(f"{empty_texts} empty texts found")

        if total_length > 1000000:  # 1M characters
            result.add_warning(f"Large total text length: {total_length:,} characters")
            result.add_suggestion("Consider processing in smaller batches for better performance")

        return result

    def _validate_single_text(self, text: Any, index: int) -> ValidationResult:
        """Validate a single text input."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Type validation
        if not isinstance(text, str):
            result.add_error(f"Text at index {index} must be a string, got {type(text)}")
            return result

        # Length validation
        if len(text) == 0:
            result.add_error(f"Text at index {index} cannot be empty")
        elif len(text.strip()) == 0:
            result.add_warning(f"Text at index {index} contains only whitespace")

        if len(text) > self.max_single_text_length:
            result.add_error(f"Text at index {index} too long: {len(text)} chars (max {self.max_single_text_length})")

        # Content validation
        if self.validation_level != ValidationLevel.LENIENT:
            # Check for potentially problematic content
            if contains_malicious_content(text):
                result.add_error(f"Text at index {index} contains potentially malicious content")

            # Check encoding
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                result.add_error(f"Text at index {index} contains invalid Unicode characters")

            # Check for extremely long lines (potential formatting issues)
            lines = text.split('\n')
            for line_idx, line in enumerate(lines):
                if len(line) > 10000:
                    result.add_warning(f"Text at index {index}, line {line_idx} is very long ({len(line)} chars)")

        return result


class ParameterValidator:
    """Validates analysis parameters and configuration."""

    def validate_batch_size(self, batch_size: int, available_memory_gb: float = None) -> ValidationResult:
        """Validate batch size parameter."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        if not isinstance(batch_size, int):
            result.add_error(f"Batch size must be an integer, got {type(batch_size)}")
            return result

        if batch_size <= 0:
            result.add_error(f"Batch size must be positive, got {batch_size}")
        elif batch_size > 1024:
            result.add_warning(f"Batch size {batch_size} is very large, may cause memory issues")

        # Memory-based recommendations
        if available_memory_gb:
            recommended_max = max(1, int(available_memory_gb * 4))  # Rough estimate
            if batch_size > recommended_max:
                result.add_suggestion(f"Consider reducing batch size to {recommended_max} based on available memory")

        return result

    def validate_device_config(self, device: str) -> ValidationResult:
        """Validate device configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        if not isinstance(device, str):
            result.add_error(f"Device must be a string, got {type(device)}")
            return result

        device = device.lower().strip()

        # Valid device patterns
        valid_patterns = [
            r'^auto$',
            r'^cpu$',
            r'^cuda$',
            r'^cuda:\d+$'
        ]

        if not any(re.match(pattern, device) for pattern in valid_patterns):
            result.add_error(f"Invalid device format: {device}")
            result.add_suggestion("Valid formats: 'auto', 'cpu', 'cuda', 'cuda:0'")

        # CUDA availability check
        if 'cuda' in device and device != 'auto':
            if not torch.cuda.is_available():
                result.add_error("CUDA device specified but CUDA is not available")
                result.add_suggestion("Use 'cpu' or 'auto' instead")
            else:
                if ':' in device:
                    device_idx = int(device.split(':')[1])
                    if device_idx >= torch.cuda.device_count():
                        result.add_error(f"CUDA device {device_idx} not available (only {torch.cuda.device_count()} devices)")

        return result


def contains_malicious_content(text: str) -> bool:
    """Check for potentially malicious content in text."""
    malicious_patterns = [
        r'<script.*?>.*?</script>',  # JavaScript
        r'javascript:',              # JavaScript protocol
        r'eval\s*\(',               # eval() calls
        r'exec\s*\(',               # exec() calls
        r'import\s+os',             # OS imports
        r'__import__',              # Dynamic imports
        r'file://',                 # File protocol
        r'\.\./',                   # Directory traversal
    ]

    text_lower = text.lower()
    for pattern in malicious_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            return True

    return False


def validate_analysis_request(request_data: Dict[str, Any],
                            validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """
    Comprehensive validation of an analysis request.

    Args:
        request_data: Dictionary containing request parameters
        validation_level: Level of validation strictness

    Returns:
        ValidationResult with comprehensive validation details
    """
    overall_result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

    try:
        # Use Pydantic for initial validation
        validated_request = AnalysisRequest(**request_data)

        # Additional custom validation
        model_validator = ModelValidator(validation_level)
        text_validator = TextValidator(validation_level)
        param_validator = ParameterValidator()

        # Validate model
        model_result = model_validator.validate_model_name(validated_request.model_name)
        overall_result.errors.extend(model_result.errors)
        overall_result.warnings.extend(model_result.warnings)
        overall_result.suggestions.extend(model_result.suggestions)

        if not model_result.is_valid:
            overall_result.is_valid = False

        # Validate layers
        layer_result = model_validator.validate_layer_specification(
            validated_request.layers,
            validated_request.model_name
        )
        overall_result.errors.extend(layer_result.errors)
        overall_result.warnings.extend(layer_result.warnings)
        overall_result.suggestions.extend(layer_result.suggestions)

        if not layer_result.is_valid:
            overall_result.is_valid = False

        # Validate texts
        text_result = text_validator.validate_text_inputs(validated_request.input_texts)
        overall_result.errors.extend(text_result.errors)
        overall_result.warnings.extend(text_result.warnings)
        overall_result.suggestions.extend(text_result.suggestions)

        if not text_result.is_valid:
            overall_result.is_valid = False

        # Validate parameters
        batch_result = param_validator.validate_batch_size(validated_request.batch_size)
        device_result = param_validator.validate_device_config(validated_request.device)

        for result in [batch_result, device_result]:
            overall_result.errors.extend(result.errors)
            overall_result.warnings.extend(result.warnings)
            overall_result.suggestions.extend(result.suggestions)
            if not result.is_valid:
                overall_result.is_valid = False

    except PydanticValidationError as e:
        overall_result.is_valid = False
        for error in e.errors():
            field = error.get('loc', ['unknown'])[0]
            message = error.get('msg', 'Unknown validation error')
            overall_result.add_error(f"Field '{field}': {message}")

    except Exception as e:
        overall_result.is_valid = False
        overall_result.add_error(f"Unexpected validation error: {str(e)}")

    return overall_result


# Legacy functions for backwards compatibility
def validate_model_name(model_name: str) -> bool:
    """Legacy function for backwards compatibility."""
    model_validator = ModelValidator()
    result = model_validator.validate_model_name(model_name)
    return result.is_valid


def validate_layer_name(layer_name: str, model=None) -> bool:
    """Legacy function for backwards compatibility."""
    if not layer_name or not isinstance(layer_name, str):
        return False

    # Check if layer exists in model if provided
    if model:
        for name, _ in model.named_modules():
            if name == layer_name:
                return True
        return False

    # Basic validation without model
    return len(layer_name.strip()) > 0


def validate_config_file(config_path: str) -> bool:
    """Legacy function for backwards compatibility."""
    try:
        path = Path(config_path)
        return path.exists() and path.is_file() and path.suffix in ['.yaml', '.yml', '.json']
    except Exception:
        return False


def validate_output_directory(output_dir: str) -> bool:
    """Legacy function for backwards compatibility."""
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Try to create a test file
        test_file = path / ".test_write"
        test_file.write_text("test")
        test_file.unlink()  # Delete test file

        return True
    except Exception as e:
        logger.error(f"Cannot write to output directory {output_dir}: {e}")
        return False


def validate_device_config(device_config: str) -> bool:
    """Legacy function for backwards compatibility."""
    param_validator = ParameterValidator()
    result = param_validator.validate_device_config(device_config)
    return result.is_valid


def check_system_requirements() -> Dict[str, Dict[str, Any]]:
    """Check if core dependencies are available and provide version info."""
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "yaml": "yaml"
    }

    results: Dict[str, Dict[str, Any]] = {}

    for display_name, module_name in packages.items():
        info = {"available": False, "version": None}
        try:
            module = __import__(module_name)
            info["available"] = True
            info["version"] = getattr(module, "__version__", None)
        except ImportError:
            info["available"] = False
            info["version"] = None

        results[display_name] = info

    return results


class ConfigValidator:
    """Simplified configuration validator used by unit tests."""

    REQUIRED_FIELDS = {'model', 'batch_size', 'layers', 'output_dir'}

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        errors: List[Dict[str, Any]] = []

        if not isinstance(config, dict):
            errors.append({'type': 'invalid_type', 'message': 'Config must be a dictionary'})
            return False, errors

        missing = self.REQUIRED_FIELDS - config.keys()
        if missing:
            errors.append({'type': 'missing_fields', 'missing': sorted(missing)})

        model = config.get('model')
        if not isinstance(model, str) or not model.strip():
            errors.append({'type': 'invalid_model', 'message': 'Model must be a non-empty string'})

        batch_size = config.get('batch_size')
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append({'type': 'invalid_batch_size', 'message': 'Batch size must be a positive integer'})

        layers = config.get('layers')
        if not isinstance(layers, list) or not layers:
            errors.append({'type': 'invalid_layers', 'message': 'Layers must be a non-empty list'})

        output_dir = config.get('output_dir')
        if not isinstance(output_dir, str) or not output_dir.strip():
            errors.append({'type': 'invalid_output_dir', 'message': 'Output directory must be a string'})

        return len(errors) == 0, errors


class DataValidator:
    """Simplified data validator used by unit tests."""

    REQUIRED_KEYS = {'questions', 'activations', 'metadata'}

    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        errors: List[Dict[str, Any]] = []

        if not isinstance(data, dict):
            errors.append({'type': 'invalid_type', 'message': 'Data must be a dictionary'})
            return False, errors

        missing = self.REQUIRED_KEYS - data.keys()
        if missing:
            errors.append({'type': 'missing_keys', 'missing': sorted(missing)})

        questions = data.get('questions', [])
        if not isinstance(questions, list) or not questions:
            errors.append({'type': 'invalid_questions', 'message': 'Questions must be a non-empty list'})
        else:
            for idx, question in enumerate(questions):
                if not isinstance(question, str) or not question.strip():
                    errors.append({'type': 'invalid_question', 'index': idx})

        activations = data.get('activations', {})
        if not isinstance(activations, dict) or not activations:
            errors.append({'type': 'invalid_activations', 'message': 'Activations must be a non-empty dict'})
        else:
            lengths = set()
            for layer, values in activations.items():
                if not isinstance(values, list):
                    errors.append({'type': 'invalid_activation_layer', 'layer': layer})
                    continue
                lengths.add(len(values))
            if len(lengths) > 1:
                errors.append({'type': 'activation_mismatch', 'message': 'Activation lengths differ between layers'})

        metadata = data.get('metadata', {})
        if not isinstance(metadata, dict):
            errors.append({'type': 'invalid_metadata', 'message': 'Metadata must be a dictionary'})
        elif 'model' not in metadata:
            errors.append({'type': 'missing_metadata', 'field': 'model'})

        return len(errors) == 0, errors


# Additional validation functions for main.py compatibility
def validate_experiment_config(config: Dict[str, Any]) -> List[str]:
    """Validate experiment configuration and return list of errors."""
    errors = []

    try:
        # Handle simple config format (common in unit tests)
        simple_keys = {
            'model', 'model_name', 'batch_size', 'num_questions', 'max_length',
            'temperature', 'top_p', 'top_k', 'seed', 'description'
        }

        if set(config.keys()).issubset(simple_keys):
            model_value = config.get('model') or config.get('model_name')
            if not isinstance(model_value, str) or not model_value.strip():
                errors.append("Model name must be a non-empty string")

            if 'batch_size' in config:
                if not isinstance(config['batch_size'], int):
                    errors.append("Batch size must be an integer")
                elif config['batch_size'] <= 0:
                    errors.append("Batch size must be positive")

            if 'max_length' in config:
                if not isinstance(config['max_length'], int):
                    errors.append("Max length must be an integer")
                elif config['max_length'] <= 0:
                    errors.append("Max length must be positive")

            if 'num_questions' in config and (
                not isinstance(config['num_questions'], int) or config['num_questions'] <= 0
            ):
                errors.append("Number of questions must be a positive integer")

            return errors

        # Handle complex config format (full experiments)
        required_fields = ['name', 'description']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field in experiment config: {field}")

        # Validate model configuration
        if 'model' in config:
            if not isinstance(config['model'], str) or not config['model'].strip():
                errors.append("Model name must be a non-empty string")
        else:
            errors.append("Missing model configuration")

        # Validate data configuration
        if 'data' in config:
            data_config = config['data']
            if 'batch_size' in data_config and not isinstance(data_config['batch_size'], int):
                errors.append("Batch size must be an integer")
            if 'batch_size' in data_config and data_config['batch_size'] <= 0:
                errors.append("Batch size must be positive")

        # Validate analysis configuration
        if 'analysis' in config:
            analysis_config = config['analysis']
            if 'target_layers' in analysis_config:
                if not isinstance(analysis_config['target_layers'], list):
                    errors.append("Target layers must be a list")
                elif len(analysis_config['target_layers']) == 0:
                    errors.append("At least one target layer must be specified")

        return errors

    except Exception as e:
        logger.error(f"Error validating experiment config: {e}")
        return [f"Validation error: {str(e)}"]


def validate_questions_file(file_path: str) -> Dict[str, Any]:
    """Validate questions file format and content."""
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'questions_count': 0
    }

    try:
        path = Path(file_path)
        if not path.exists():
            result['errors'].append(f"Questions file does not exist: {file_path}")
            return result

        if not path.suffix == '.jsonl':
            result['warnings'].append(f"Questions file should have .jsonl extension: {file_path}")

        # Try to read and validate a few lines
        questions_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 lines
                    break
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'question' not in data:
                            result['errors'].append(f"Invalid format in questions file line {i+1}: missing 'question' field")
                            return result
                        questions_count += 1
                    except json.JSONDecodeError as e:
                        result['errors'].append(f"Invalid JSON in questions file line {i+1}: {e}")
                        return result

        result['valid'] = True
        result['questions_count'] = questions_count
        return result
    except Exception as e:
        logger.error(f"Error validating questions file: {e}")
        return False


def validate_activation_file(file_path: str) -> bool:
    """Validate activation file format and content."""
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Activation file does not exist: {file_path}")
            return False

        if path.suffix == '.csv':
            import pandas as pd
            try:
                df = pd.read_csv(path, nrows=5)  # Read first 5 rows
                required_columns = ['question', 'activation_vector']
                for col in required_columns:
                    if col not in df.columns:
                        logger.error(f"Missing required column in activation file: {col}")
                        return False
                return True
            except Exception as e:
                logger.error(f"Error reading activation CSV file: {e}")
                return False

        elif path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 lines
                        break
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'question' not in data or 'activation_vector' not in data:
                                logger.error(f"Invalid format in activation file line {i+1}")
                                return False
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in activation file line {i+1}: {e}")
                            return False
            return True

        else:
            logger.error(f"Unsupported activation file format: {path.suffix}")
            return False

    except Exception as e:
        logger.error(f"Error validating activation file: {e}")
        return False


def validate_file_path(file_path: str) -> bool:
    """Validate that a file path exists and is accessible."""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate model configuration parameters."""
    required_fields = ['name', 'type']

    try:
        for field in required_fields:
            if field not in config:
                return False
        return True
    except Exception:
        return False


@dataclass
class ValidationReport:
    """Comprehensive validation report for analysis results."""
    is_valid: bool = True
    format_issues: List[str] = None
    statistical_warnings: List[str] = None
    consistency_errors: List[str] = None
    scientific_warnings: List[str] = None
    
    def __post_init__(self):
        if self.format_issues is None:
            self.format_issues = []
        if self.statistical_warnings is None:
            self.statistical_warnings = []
        if self.consistency_errors is None:
            self.consistency_errors = []
        if self.scientific_warnings is None:
            self.scientific_warnings = []
    
    def add_format_issues(self, issues: List[str]):
        """Add format validation issues."""
        self.format_issues.extend(issues)
        if issues:
            self.is_valid = False
    
    def add_statistical_warnings(self, warnings: List[str]):
        """Add statistical anomaly warnings."""
        self.statistical_warnings.extend(warnings)
    
    def add_consistency_errors(self, errors: List[str]):
        """Add consistency validation errors."""
        self.consistency_errors.extend(errors)
        if errors:
            self.is_valid = False
    
    def add_scientific_warnings(self, warnings: List[str]):
        """Add scientific accuracy warnings."""
        self.scientific_warnings.extend(warnings)


class FormatValidatorSuite:
    """Validates output data formats and structure."""
    
    def check_data_formats(self, results: Dict[str, Any]) -> List[str]:
        """Check data format compliance."""
        issues = []
        
        # Check if required fields are present
        required_fields = ['activations', 'metadata', 'model_info']
        for field in required_fields:
            if field not in results:
                issues.append(f"Missing required field: {field}")
        
        # Validate activation data format
        if 'activations' in results:
            activations = results['activations']
            if not isinstance(activations, (list, np.ndarray, torch.Tensor)):
                issues.append("Activations must be list, numpy array, or torch tensor")
            elif isinstance(activations, list) and len(activations) > 0:
                if not all(isinstance(x, (list, np.ndarray, torch.Tensor)) for x in activations):
                    issues.append("All activation vectors must be numeric arrays")
        
        # Validate metadata structure
        if 'metadata' in results:
            metadata = results['metadata']
            if not isinstance(metadata, dict):
                issues.append("Metadata must be a dictionary")
            else:
                required_meta = ['timestamp', 'model_name', 'layer_name']
                for meta_field in required_meta:
                    if meta_field not in metadata:
                        issues.append(f"Missing metadata field: {meta_field}")
        
        return issues


class StatisticalValidatorSuite:
    """Validates statistical properties of analysis results."""
    
    def detect_anomalies(self, results: Dict[str, Any]) -> List[str]:
        """Detect statistical anomalies in activation patterns."""
        warnings = []
        
        if 'activations' not in results:
            return ["Cannot perform statistical validation: missing activations"]
        
        activations = results['activations']
        
        # Convert to numpy array for analysis
        if isinstance(activations, torch.Tensor):
            act_array = activations.detach().cpu().numpy()
        elif isinstance(activations, list):
            try:
                act_array = np.array(activations)
            except (ValueError, TypeError):
                return ["Cannot convert activations to numpy array for statistical analysis"]
        else:
            act_array = activations
        
        # Check for NaN or infinite values
        if np.any(np.isnan(act_array)):
            warnings.append("Activations contain NaN values")
        if np.any(np.isinf(act_array)):
            warnings.append("Activations contain infinite values")
        
        # Check activation distribution
        if act_array.size > 0:
            std_dev = np.std(act_array)
            mean_val = np.mean(act_array)
            
            # Check for unusually low variance (dead neurons)
            if std_dev < 1e-6:
                warnings.append(f"Very low activation variance detected: {std_dev:.2e}")
            
            # Check for extreme values
            if np.abs(mean_val) > 100:
                warnings.append(f"Extreme mean activation value: {mean_val:.2f}")
            
            # Check for activation saturation
            max_val = np.max(act_array)
            min_val = np.min(act_array)
            if max_val > 1000 or min_val < -1000:
                warnings.append(f"Extreme activation values detected: min={min_val:.2f}, max={max_val:.2f}")
        
        return warnings


class ConsistencyCheckerSuite:
    """Validates consistency across different components."""
    
    def cross_validate(self, results: Dict[str, Any]) -> List[str]:
        """Perform cross-validation checks."""
        errors = []
        
        # Check dimension consistency
        if 'activations' in results and 'metadata' in results:
            activations = results['activations']
            metadata = results['metadata']
            
            # Check if activation dimensions match expected model dimensions
            if 'expected_dimensions' in metadata:
                expected_dims = metadata['expected_dimensions']
                if isinstance(activations, (list, np.ndarray, torch.Tensor)):
                    actual_shape = np.array(activations).shape
                    if len(actual_shape) != len(expected_dims):
                        errors.append(f"Dimension mismatch: expected {len(expected_dims)}, got {len(actual_shape)}")
        
        # Check temporal consistency for sequences
        if 'sequence_data' in results:
            seq_data = results['sequence_data']
            if isinstance(seq_data, list) and len(seq_data) > 1:
                shapes = [np.array(item).shape for item in seq_data]
                if not all(shape == shapes[0] for shape in shapes):
                    errors.append("Inconsistent shapes across sequence elements")
        
        return errors


class ScientificValidatorSuite:
    """Validates scientific accuracy and plausibility."""
    
    def verify_accuracy(self, results: Dict[str, Any]) -> List[str]:
        """Verify scientific accuracy of results."""
        warnings = []
        
        # Check activation patterns for biological plausibility
        if 'activations' in results:
            activations = results['activations']
            act_array = np.array(activations) if not isinstance(activations, np.ndarray) else activations
            
            # Check sparsity (biological neurons are typically sparse)
            if act_array.size > 0:
                sparsity = np.sum(act_array == 0) / act_array.size
                if sparsity < 0.1:
                    warnings.append(f"Low sparsity detected: {sparsity:.2%} (expected >10% for biological plausibility)")
                
                # Check for unrealistic activation patterns
                if np.all(act_array >= 0):
                    # All positive activations might be suspicious for some layer types
                    layer_name = results.get('metadata', {}).get('layer_name', '')
                    if 'attention' in layer_name.lower():
                        warnings.append("All-positive activations unusual for attention layers")
        
        # Check model-architecture consistency
        metadata = results.get('metadata', {})
        model_name = metadata.get('model_name', '')
        layer_name = metadata.get('layer_name', '')
        
        if model_name and layer_name:
            # Basic sanity checks for common architectures
            if 'gpt' in model_name.lower() and 'encoder' in layer_name.lower():
                warnings.append("GPT models don't have encoder layers - possible layer name error")
            elif 'bert' in model_name.lower() and 'decoder' in layer_name.lower():
                warnings.append("BERT models don't have decoder layers - possible layer name error")
        
        return warnings


class OutputValidator:
    """Comprehensive output validation system for analysis results."""
    
    def __init__(self):
        self.format_validators = FormatValidatorSuite()
        self.statistical_validators = StatisticalValidatorSuite()
        self.consistency_checkers = ConsistencyCheckerSuite()
        self.scientific_validators = ScientificValidatorSuite()
    
    def validate_analysis_results(self, results: Dict[str, Any]) -> ValidationReport:
        """
        Perform comprehensive validation of analysis results.
        
        Args:
            results: Dictionary containing analysis results
            
        Returns:
            ValidationReport with detailed validation outcomes
        """
        validation_report = ValidationReport()
        
        # Format validation
        format_issues = self.format_validators.check_data_formats(results)
        validation_report.add_format_issues(format_issues)
        
        # Statistical validation
        statistical_anomalies = self.statistical_validators.detect_anomalies(results)
        validation_report.add_statistical_warnings(statistical_anomalies)
        
        # Consistency checking
        consistency_errors = self.consistency_checkers.cross_validate(results)
        validation_report.add_consistency_errors(consistency_errors)
        
        # Scientific accuracy
        scientific_issues = self.scientific_validators.verify_accuracy(results)
        validation_report.add_scientific_warnings(scientific_issues)
        
        logger.info(f"Output validation completed. Valid: {validation_report.is_valid}")
        
        return validation_report
    
    def validate_with_baseline(self, results: Dict[str, Any], baseline: Dict[str, Any]) -> ValidationReport:
        """
        Validate results against a known baseline.
        
        Args:
            results: Current analysis results
            baseline: Baseline results for comparison
            
        Returns:
            ValidationReport including baseline comparison
        """
        validation_report = self.validate_analysis_results(results)
        
        # Additional baseline comparison
        baseline_warnings = []
        
        if 'activations' in results and 'activations' in baseline:
            current_act = np.array(results['activations'])
            baseline_act = np.array(baseline['activations'])
            
            if current_act.shape == baseline_act.shape:
                correlation = np.corrcoef(current_act.flatten(), baseline_act.flatten())[0, 1]
                if correlation < 0.8:
                    baseline_warnings.append(f"Low correlation with baseline: {correlation:.3f}")
                
                # Check for significant deviation
                mse = np.mean((current_act - baseline_act) ** 2)
                if mse > 1.0:
                    baseline_warnings.append(f"High MSE deviation from baseline: {mse:.3f}")
        
        validation_report.add_scientific_warnings(baseline_warnings)
        
        return validation_report


class DomainValidator:
    """Base class for domain-specific validators."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.task_validators = {}
    
    def get_task_validator(self, task_type: str):
        """Get validator for specific task type."""
        return self.task_validators.get(task_type, self)
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> List[str]:
        """Perform domain-specific validation."""
        return []


class NLPDomainValidator(DomainValidator):
    """Validator for Natural Language Processing tasks."""
    
    def __init__(self):
        super().__init__("nlp")
        self.language_patterns = {
            'english': re.compile(r'[a-zA-Z\s.,!?;:]+'),
            'multilingual': re.compile(r'[\w\s.,!?;:]+', re.UNICODE)
        }
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> List[str]:
        """Validate NLP-specific aspects."""
        warnings = []
        
        # Check for text-related metadata
        metadata = results.get('metadata', {})
        if 'input_texts' in results:
            texts = results['input_texts']
            
            # Language consistency check
            for i, text in enumerate(texts):
                if isinstance(text, str):
                    if len(text) > 10000:
                        warnings.append(f"Text {i} unusually long ({len(text)} chars) - may cause memory issues")
                    
                    # Check for potential encoding issues
                    try:
                        text.encode('utf-8')
                    except UnicodeEncodeError:
                        warnings.append(f"Text {i} contains encoding issues")
        
        # Model-specific validation
        model_name = metadata.get('model_name', '').lower()
        if 'gpt' in model_name:
            # GPT-specific validation
            if 'activations' in results:
                activations = np.array(results['activations'])
                if activations.ndim < 2:
                    warnings.append("GPT models should produce multi-dimensional activations")
        
        elif 'bert' in model_name:
            # BERT-specific validation
            if 'attention_weights' in results:
                attention = results['attention_weights']
                if not self._check_attention_symmetry(attention):
                    warnings.append("BERT attention patterns show unusual asymmetry")
        
        return warnings
    
    def _check_attention_symmetry(self, attention_weights) -> bool:
        """Check if attention weights follow expected patterns."""
        if isinstance(attention_weights, (list, np.ndarray)):
            attention_array = np.array(attention_weights)
            if attention_array.ndim >= 2:
                # Check if attention weights sum approximately to 1
                row_sums = np.sum(attention_array, axis=-1)
                return np.allclose(row_sums, 1.0, rtol=0.1)
        return True


class ComputerVisionDomainValidator(DomainValidator):
    """Validator for Computer Vision tasks."""
    
    def __init__(self):
        super().__init__("computer_vision")
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> List[str]:
        """Validate CV-specific aspects."""
        warnings = []
        
        metadata = results.get('metadata', {})
        
        # Check image-related metadata
        if 'image_shape' in metadata:
            shape = metadata['image_shape']
            if len(shape) < 3:
                warnings.append("Image should have at least 3 dimensions (H, W, C)")
            elif len(shape) == 3 and shape[2] not in [1, 3, 4]:
                warnings.append(f"Unusual number of image channels: {shape[2]}")
        
        # Feature map validation
        if 'feature_maps' in results:
            feature_maps = results['feature_maps']
            if isinstance(feature_maps, (list, np.ndarray)):
                fm_array = np.array(feature_maps)
                
                # Check for spatial consistency
                if fm_array.ndim >= 3:
                    spatial_dims = fm_array.shape[-2:]
                    if spatial_dims[0] * spatial_dims[1] > 1000000:
                        warnings.append("Very large spatial dimensions may indicate issue")
                
                # Check for reasonable activation ranges
                if np.max(fm_array) > 1000 or np.min(fm_array) < -1000:
                    warnings.append("Extreme feature map values detected")
        
        return warnings


class NeuroscienceDomainValidator(DomainValidator):
    """Validator for Neuroscience research tasks."""
    
    def __init__(self):
        super().__init__("neuroscience")
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> List[str]:
        """Validate neuroscience-specific aspects."""
        warnings = []
        
        # Check biological plausibility
        if 'activations' in results:
            activations = np.array(results['activations'])
            
            # Sparsity check (biological neurons are typically sparse)
            sparsity = np.sum(activations == 0) / activations.size
            if sparsity < 0.05:
                warnings.append(f"Low sparsity ({sparsity:.1%}) - unusual for biological systems")
            
            # Check firing rates
            if np.mean(activations) > 100:
                warnings.append("High average activation - may be unbiological")
            
            # Check for temporal consistency in sequential data
            if activations.ndim >= 2 and activations.shape[0] > 1:
                temporal_corr = np.corrcoef(activations[:-1].flatten(), activations[1:].flatten())[0, 1]
                if temporal_corr > 0.99:
                    warnings.append("Extremely high temporal correlation - may indicate static activation")
        
        # EEG/fMRI specific validation
        metadata = results.get('metadata', {})
        data_type = metadata.get('data_type', '').lower()
        
        if 'eeg' in data_type:
            # EEG-specific checks
            if 'sampling_rate' in metadata:
                sr = metadata['sampling_rate']
                if sr < 100 or sr > 10000:
                    warnings.append(f"Unusual EEG sampling rate: {sr} Hz")
        
        elif 'fmri' in data_type:
            # fMRI-specific checks
            if 'tr' in metadata:  # Repetition time
                tr = metadata['tr']
                if tr < 0.5 or tr > 5.0:
                    warnings.append(f"Unusual fMRI TR: {tr} seconds")
        
        return warnings


class CodeAnalysisDomainValidator(DomainValidator):
    """Validator for code analysis tasks."""
    
    def __init__(self):
        super().__init__("code_analysis")
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> List[str]:
        """Validate code analysis specific aspects."""
        warnings = []
        
        metadata = results.get('metadata', {})
        
        # Check programming language consistency
        if 'language' in metadata:
            language = metadata['language'].lower()
            if 'code_samples' in results:
                code_samples = results['code_samples']
                for i, code in enumerate(code_samples):
                    if self._check_language_consistency(code, language):
                        warnings.append(f"Code sample {i} may not match declared language: {language}")
        
        # Check for code-specific activations
        if 'activations' in results:
            activations = np.array(results['activations'])
            
            # Code embeddings often have specific patterns
            if activations.ndim >= 2:
                # Check for reasonable embedding dimensions
                embedding_dim = activations.shape[-1]
                if embedding_dim < 50 or embedding_dim > 4096:
                    warnings.append(f"Unusual embedding dimension for code: {embedding_dim}")
        
        return warnings
    
    def _check_language_consistency(self, code: str, declared_language: str) -> bool:
        """Check if code matches declared programming language."""
        if not isinstance(code, str):
            return False
        
        language_indicators = {
            'python': ['def ', 'import ', 'from ', 'if __name__'],
            'javascript': ['function', 'var ', 'let ', 'const ', '=>'],
            'java': ['public class', 'import java.', 'public static void'],
            'cpp': ['#include', 'using namespace', 'int main()', '::'],
            'c': ['#include', 'int main(', 'printf(', 'malloc(']
        }
        
        if declared_language in language_indicators:
            indicators = language_indicators[declared_language]
            found_indicators = sum(1 for indicator in indicators if indicator in code)
            return found_indicators == 0  # Return True if suspicious (no indicators found)
        
        return False


class DomainValidatorRegistry:
    """Registry for domain-specific validators."""
    
    def __init__(self):
        self.domain_validators = {
            'nlp': NLPDomainValidator(),
            'computer_vision': ComputerVisionDomainValidator(),
            'neuroscience': NeuroscienceDomainValidator(),
            'code_analysis': CodeAnalysisDomainValidator()
        }
    
    def get_validator(self, domain: str, task_type: str = None) -> DomainValidator:
        """Get validator for specific domain and task."""
        if domain not in self.domain_validators:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.domain_validators.keys())}")
        
        validator = self.domain_validators[domain]
        if task_type:
            return validator.get_task_validator(task_type)
        return validator
    
    def validate_with_domain(self, results: Dict[str, Any], domain: str, task_type: str = None) -> List[str]:
        """Perform domain-specific validation."""
        validator = self.get_validator(domain, task_type)
        return validator.validate_domain_specific(results)
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        return list(self.domain_validators.keys())


# Add domain validation to the main OutputValidator
class EnhancedOutputValidator(OutputValidator):
    """Enhanced output validator with domain-specific validation."""
    
    def __init__(self, domain: str = None):
        super().__init__()
        self.domain_registry = DomainValidatorRegistry()
        self.domain = domain
    
    def validate_analysis_results(self, results: Dict[str, Any], domain: str = None) -> ValidationReport:
        """
        Perform comprehensive validation including domain-specific checks.
        
        Args:
            results: Dictionary containing analysis results
            domain: Optional domain for specialized validation
            
        Returns:
            ValidationReport with detailed validation outcomes
        """
        # Perform standard validation
        validation_report = super().validate_analysis_results(results)
        
        # Add domain-specific validation if domain is specified
        target_domain = domain or self.domain
        if target_domain:
            try:
                domain_warnings = self.domain_registry.validate_with_domain(results, target_domain)
                validation_report.add_scientific_warnings(domain_warnings)
            except ValueError as e:
                validation_report.add_scientific_warnings([f"Domain validation error: {str(e)}"])
        
        return validation_report
