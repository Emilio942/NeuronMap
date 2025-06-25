# Utilities Module API Reference

The utils module provides configuration management, file handling, validation, and other utility functions.

## Overview

```{eval-rst}
.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration Management

### ConfigManager

```{eval-rst}
.. autoclass:: src.utils.config.ConfigManager
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: get_model_config
   .. automethod:: get_analysis_config
   .. automethod:: get_visualization_config
   .. automethod:: validate_all_configs
```

### Configuration Functions

```{eval-rst}
.. autofunction:: src.utils.config.get_config_manager
.. autofunction:: src.utils.config.setup_global_config
.. autofunction:: src.utils.config.reset_config_manager
```

## File Handling

### File I/O Functions

```{eval-rst}
.. autofunction:: src.utils.file_handlers.load_questions
.. autofunction:: src.utils.file_handlers.save_results
.. autofunction:: src.utils.file_handlers.load_json
.. autofunction:: src.utils.file_handlers.save_json
```

## Validation

### Validation Classes

```{eval-rst}
.. autoclass:: src.utils.validation.ValidationResult
   :members:
   :undoc-members:
   :show-inheritance:
```

### Validation Functions

```{eval-rst}
.. autofunction:: src.utils.validation.validate_model_config
.. autofunction:: src.utils.validation.validate_analysis_request
.. autofunction:: src.utils.validation.validate_experiment_config
```

## Usage Examples

### Configuration Setup

```python
from src.utils.config import get_config_manager, setup_global_config

# Basic setup
config = get_config_manager()

# Advanced setup with validation
config = setup_global_config(
    environment="production",
    config_dir="custom_configs/"
)

# Get specific configurations
model_config = config.get_model_config("gpt2")
analysis_config = config.get_analysis_config()
```

### File Operations

```python
from src.utils.file_handlers import load_questions, save_results

# Load questions from file
questions = load_questions("data/questions.jsonl")

# Save analysis results
results = {
    'activations': activation_data,
    'metadata': experiment_metadata
}
save_results(results, "outputs/analysis_results.json")
```

### Input Validation

```python
from src.utils.validation import validate_analysis_request

# Validate analysis parameters
validation_result = validate_analysis_request({
    'model_name': 'gpt2',
    'questions': question_list,
    'layers': [3, 6, 9],
    'batch_size': 32
})

if not validation_result.is_valid:
    print("Validation errors:", validation_result.errors)
```

### Error Handling

```python
from src.utils.error_handling import with_retry, safe_execute

# Retry decorator for unstable operations
@with_retry(max_attempts=3, delay=1.0)
def unstable_model_call():
    return model.generate(input_text)

# Safe execution with error handling
result = safe_execute(
    func=risky_operation,
    fallback_value=None,
    log_errors=True
)
```

## Configuration Schema

### Model Configuration

```{eval-rst}
.. autoclass:: src.utils.config.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

### Analysis Configuration

```{eval-rst}
.. autoclass:: src.utils.config.AnalysisConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

### Environment Configuration

```{eval-rst}
.. autoclass:: src.utils.config.EnvironmentConfig
   :members:
   :undoc-members:
   :show-inheritance:
```
