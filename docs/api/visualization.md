# Visualization Module API Reference

The visualization module provides comprehensive plotting and visualization capabilities for neural network analysis.

## Overview

```{eval-rst}
.. automodule:: src.visualization
   :members:
   :undoc-members:
   :show-inheritance:
```

## Core Classes

### CoreVisualizer

```{eval-rst}
.. autoclass:: src.visualization.CoreVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: plot_pca
   .. automethod:: plot_tsne
   .. automethod:: plot_activation_heatmap
   .. automethod:: save_figure
```

### InteractivePlots

```{eval-rst}
.. autoclass:: src.visualization.InteractivePlots
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Basic Visualization

```python
from src.visualization import CoreVisualizer
from src.utils.config import get_config_manager

# Initialize visualizer
config = get_config_manager()
visualizer = CoreVisualizer(config=config.get_visualization_config())

# Create PCA plot
pca_fig = visualizer.plot_pca(
    data=activation_data,
    labels=question_labels,
    title="Activation Space Visualization"
)

# Save the plot
visualizer.save_figure(pca_fig, "activation_pca.png")
```

### Interactive Visualization

```python
from src.visualization import InteractivePlots

plotter = InteractivePlots()

# Create interactive heatmap
heatmap = plotter.create_interactive_heatmap(
    data=attention_weights,
    x_labels=tokens,
    y_labels=layers,
    title="Attention Pattern Analysis"
)

# Export to HTML
plotter.export_html(heatmap, "attention_analysis.html")
```

### Custom Styling

```python
# Apply custom theme
visualizer.set_style_config({
    'theme': 'dark',
    'color_scheme': 'viridis',
    'font_size': 14
})

# Create styled visualization
styled_fig = visualizer.plot_tsne(
    data=embeddings,
    perplexity=30,
    title="Styled t-SNE Visualization"
)
```

## Function Reference

```{eval-rst}
.. autofunction:: src.visualization.create_interactive_analysis
.. autofunction:: src.visualization.plot_attention_patterns
.. autofunction:: src.visualization.generate_report_figures
```
