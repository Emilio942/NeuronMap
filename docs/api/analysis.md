# Analysis Module API Reference

The analysis module provides comprehensive neural network activation analysis capabilities.

## Overview

```{eval-rst}
.. automodule:: src.analysis
   :members:
   :undoc-members:
   :show-inheritance:
```

## Core Classes

### ActivationExtractor

```{eval-rst}
.. autoclass:: src.analysis.ActivationExtractor
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: extract_activations
   .. automethod:: extract_from_layer
   .. automethod:: get_model_info
```

### LayerInspector

```{eval-rst}
.. autoclass:: src.analysis.LayerInspector
   :members:
   :undoc-members:
   :show-inheritance:
```

### AdvancedAnalyzer

```{eval-rst}
.. autoclass:: src.analysis.AdvancedAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Basic Activation Extraction

```python
from src.analysis import ActivationExtractor
from src.utils.config import get_config_manager

# Initialize with configuration
config = get_config_manager()
extractor = ActivationExtractor(
    model_name="gpt2",
    config=config.get_analysis_config()
)

# Extract activations
questions = ["What is AI?", "How do neural networks work?"]
results = extractor.extract_activations(questions)

print(f"Extracted activations shape: {results['activations'].shape}")
```

### Multi-Layer Analysis

```python
from src.analysis import LayerInspector

inspector = LayerInspector(model_name="bert-base-uncased")

# Analyze all layers
layer_analysis = inspector.analyze_all_layers(
    inputs=test_inputs,
    metrics=["activation_stats", "attention_patterns"]
)

# Compare specific layers
comparison = inspector.compare_layers(
    layer_indices=[3, 6, 9],
    metric="representation_similarity"
)
```

## Function Reference

```{eval-rst}
.. autofunction:: src.analysis.analyze_model_layers
.. autofunction:: src.analysis.compute_activation_statistics
.. autofunction:: src.analysis.extract_attention_patterns
```
