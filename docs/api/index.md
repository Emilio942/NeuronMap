# API Reference

This section provides comprehensive API documentation for all NeuronMap modules and classes.

## 📚 Module Overview

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: module.rst
   :recursive:

   src.analysis
   src.visualization
   src.data_generation
   src.utils
```

## 🔍 Quick Reference

### Core Classes

```{eval-rst}
.. currentmodule:: src

.. autosummary::
   :toctree: _autosummary

   analysis.ActivationExtractor
   analysis.LayerInspector
   visualization.CoreVisualizer
   visualization.InteractivePlots
   data_generation.QuestionGenerator
   utils.config.ConfigManager
```

### Utility Functions

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   utils.config.get_config_manager
   utils.config.setup_global_config
   utils.file_handlers.load_questions
   utils.file_handlers.save_results
   utils.validation.validate_model_config
```

## 📖 Detailed Documentation

### Analysis Module

The analysis module provides core functionality for extracting and analyzing neural network activations.

```{eval-rst}
.. automodule:: src.analysis
   :members:
   :undoc-members:
   :show-inheritance:
```

### Visualization Module

The visualization module offers comprehensive plotting and visualization capabilities.

```{eval-rst}
.. automodule:: src.visualization
   :members:
   :undoc-members:
   :show-inheritance:
```

### Data Generation Module

The data generation module handles question generation and synthetic data creation.

```{eval-rst}
.. automodule:: src.data_generation
   :members:
   :undoc-members:
   :show-inheritance:
```

### Utils Module

The utils module provides configuration management, file handling, and validation utilities.

```{eval-rst}
.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## 🎯 Usage Examples

### Basic Analysis Workflow

```python
from src.utils.config import get_config_manager
from src.analysis.activation_extractor import ActivationExtractor
from src.visualization.core_visualizer import CoreVisualizer

# Setup configuration
config = get_config_manager()

# Initialize analyzer
extractor = ActivationExtractor(
    model_name="gpt2",
    config=config.get_analysis_config()
)

# Extract activations
questions = ["What is machine learning?", "How do neural networks work?"]
activations = extractor.extract_activations(questions)

# Visualize results
visualizer = CoreVisualizer(config=config.get_visualization_config())
fig = visualizer.plot_activation_heatmap(activations)
fig.show()
```

### Configuration Management

```python
from src.utils.config import get_config_manager, setup_global_config

# Setup with custom environment
config = setup_global_config(environment="production")

# Get model-specific configuration
model_config = config.get_model_config("gpt2")
print(f"Model layers: {model_config.layers.total_layers}")

# Validate hardware compatibility
issues = config.validate_hardware_compatibility()
if issues:
    print("Hardware compatibility issues:", issues)
```

### Advanced Visualization

```python
from src.visualization.interactive_plots import InteractivePlots
from src.utils.config import get_config_manager

config = get_config_manager()
plotter = InteractivePlots(config=config.get_visualization_config())

# Create interactive PCA plot
pca_fig = plotter.create_pca_plot(
    data=activation_data,
    labels=question_labels,
    title="Activation Space Visualization"
)

# Add clustering overlay
clustered_fig = plotter.add_clustering_overlay(
    pca_fig, 
    n_clusters=5,
    method="kmeans"
)

# Export to HTML
plotter.export_html(clustered_fig, "activation_analysis.html")
```

## 🛠 Configuration Schema

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

### Visualization Configuration

```{eval-rst}
.. autoclass:: src.utils.config.VisualizationConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

## 🔧 Extension Points

### Custom Model Support

```python
from src.analysis.activation_extractor import ActivationExtractor
from src.utils.config import ModelConfig

# Define custom model configuration
custom_model_config = ModelConfig(
    name="custom-transformer",
    type="gpt",
    layers={
        "attention": "transformer.h.{layer}.attn",
        "mlp": "transformer.h.{layer}.mlp",
        "total_layers": 24
    },
    hidden_size=1024,
    attention_heads=16
)

# Use with extractor
extractor = ActivationExtractor(
    model_name="custom-transformer",
    model_config=custom_model_config
)
```

### Custom Visualization Themes

```python
from src.visualization.core_visualizer import CoreVisualizer
from src.utils.config import StyleConfig, ColorSchemesConfig

# Define custom theme
custom_style = StyleConfig(
    theme="custom",
    font_family="Roboto",
    font_size=14,
    grid_alpha=0.2
)

custom_colors = ColorSchemesConfig(
    categorical="Set2",
    sequential="plasma",
    diverging="RdYlBu"
)

# Apply to visualizer
visualizer = CoreVisualizer(
    style_config=custom_style,
    color_config=custom_colors
)
```

## 📊 Type Definitions

### Common Types

```python
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import numpy as np

# Activation data types
ActivationTensor = torch.Tensor
ActivationArray = np.ndarray
ActivationData = Union[ActivationTensor, ActivationArray]

# Configuration types
ConfigDict = Dict[str, Any]
ModelName = str
LayerName = str

# Analysis types
QuestionList = List[str]
ActivationList = List[ActivationData]
ResultDict = Dict[str, Any]

# Visualization types
PlotData = Union[np.ndarray, torch.Tensor, List[float]]
ColorScheme = str
FigureSize = Tuple[int, int]
```

## 🧪 Testing Utilities

### Mock Objects for Testing

```python
from src.utils.testing import MockModel, MockTokenizer, MockConfig

# Create mock objects for testing
mock_model = MockModel(hidden_size=768, num_layers=12)
mock_tokenizer = MockTokenizer(vocab_size=50257)
mock_config = MockConfig(environment="testing")

# Use in tests
def test_activation_extraction():
    extractor = ActivationExtractor(model=mock_model)
    activations = extractor.extract_activations(["test question"])
    assert activations.shape[0] == 1
```

### Test Data Generators

```python
from src.utils.testing import generate_test_activations, generate_test_questions

# Generate synthetic test data
test_activations = generate_test_activations(
    n_samples=100,
    hidden_size=768,
    seed=42
)

test_questions = generate_test_questions(
    n_questions=50,
    categories=["science", "history", "philosophy"]
)
```

---

```{note}
This API documentation is automatically generated from docstrings in the source code. 
For the most up-to-date information, please refer to the source code directly or 
build the documentation locally using `sphinx-build`.
```

```{seealso}
- {doc}`../tutorials/index` - Step-by-step tutorials using these APIs
- {doc}`../examples/index` - Complete examples and use cases  
- {doc}`../development/index` - Development guidelines and contribution guide
```
