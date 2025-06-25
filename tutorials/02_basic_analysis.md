# Basic Analysis Tutorial

Learn how to perform comprehensive neural network activation analysis with NeuronMap.

## Overview

This tutorial covers:
- Loading and configuring models
- Processing different input types
- Extracting activations from specific layers
- Basic visualization techniques
- Interpreting results

## Prerequisites

- Completed [Quick Start Guide](01_quick_start.md)
- Basic understanding of neural networks
- Python familiarity helpful

## Setting Up Your Analysis

### 1. Model Configuration

Create a model configuration file:

```yaml
# configs/my_analysis.yaml
model:
  name: "bert-base-uncased"
  type: "huggingface"
  cache_dir: "./models"
  
analysis:
  layers: [0, 3, 6, 9, 11]  # Which layers to analyze
  batch_size: 16
  max_length: 512
  
visualization:
  plot_types: ["heatmap", "line", "scatter"]
  save_format: "png"
  dpi: 300
```

### 2. Input Data Preparation

#### Text Input
```python
# Single text
text = "The quick brown fox jumps over the lazy dog."

# Multiple texts
texts = [
    "Positive sentiment example.",
    "Negative sentiment example.", 
    "Neutral statement here."
]

# From file
with open('inputs.txt', 'r') as f:
    texts = f.readlines()
```

#### Structured Data
```python
# For code analysis
code_samples = [
    "def hello(): return 'world'",
    "class MyClass: pass",
    "for i in range(10): print(i)"
]

# For mathematical expressions
math_expressions = [
    "x^2 + 2x + 1 = 0",
    "∫ sin(x) dx = -cos(x) + C",
    "lim(x→∞) 1/x = 0"
]
```

## Step-by-Step Analysis

### Step 1: Generate Activations

```bash
# Command line approach
neuronmap generate \
    --config configs/my_analysis.yaml \
    --input-file inputs.txt \
    --output-dir ./analysis_output \
    --verbose
```

```python
# Python API approach
from neuronmap import NeuronMap
from neuronmap.config import load_config

# Load configuration
config = load_config("configs/my_analysis.yaml")

# Initialize analyzer
nm = NeuronMap(config)

# Generate activations
results = nm.generate_activations(
    inputs=texts,
    output_dir="./analysis_output"
)
```

### Step 2: Extract Layer Information

```bash
# Extract specific layers
neuronmap extract \
    --data-path ./analysis_output \
    --layers 0 6 11 \
    --extract-attention \
    --extract-hidden-states \
    --output-format json
```

```python
# Python extraction
layer_data = nm.extract_layers(
    results,
    layers=[0, 6, 11],
    include_attention=True,
    include_hidden_states=True
)
```

### Step 3: Basic Visualizations

#### Activation Heatmaps
```python
import matplotlib.pyplot as plt
from neuronmap.visualization import plot_activation_heatmap

# Plot layer activations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, layer_idx in enumerate([0, 6, 11]):
    plot_activation_heatmap(
        layer_data[layer_idx],
        ax=axes[i],
        title=f"Layer {layer_idx} Activations"
    )

plt.tight_layout()
plt.savefig("activation_heatmaps.png", dpi=300)
```

#### Attention Patterns
```python
from neuronmap.visualization import plot_attention_patterns

# Visualize attention heads
attention_fig = plot_attention_patterns(
    layer_data,
    input_text=text,
    layer=6,
    head=0
)
attention_fig.savefig("attention_patterns.png")
```

### Step 4: Statistical Analysis

```python
from neuronmap.analysis import compute_activation_statistics

# Compute basic statistics
stats = compute_activation_statistics(layer_data)

print(f"Mean activation: {stats['mean']:.4f}")
print(f"Standard deviation: {stats['std']:.4f}")
print(f"Max activation: {stats['max']:.4f}")
print(f"Min activation: {stats['min']:.4f}")

# Layer-wise comparison
layer_stats = {}
for layer_idx in [0, 6, 11]:
    layer_stats[layer_idx] = compute_activation_statistics(
        layer_data[layer_idx]
    )
```

## Advanced Analysis Options

### 1. Neuron-Level Analysis

```python
# Identify most active neurons
from neuronmap.analysis import find_top_neurons

top_neurons = find_top_neurons(
    layer_data[6],  # Layer 6
    top_k=50,
    method="activation_magnitude"
)

# Analyze neuron specialization
specialization = nm.analyze_neuron_specialization(
    layer_data,
    neurons=top_neurons,
    inputs=texts
)
```

### 2. Comparative Analysis

```python
# Compare different inputs
comparison = nm.compare_activations(
    inputs=[
        "Positive sentence.",
        "Negative sentence."
    ],
    layers=[6, 11]
)

# Visualize differences
nm.plot_activation_difference(comparison)
```

### 3. Temporal Analysis (for sequences)

```python
# Analyze how activations change over time/tokens
temporal_analysis = nm.analyze_temporal_patterns(
    layer_data,
    sequence_length=len(text.split())
)

nm.plot_temporal_activations(temporal_analysis)
```

## Interpreting Results

### Activation Patterns
- **High activations**: Indicate strong feature responses
- **Low activations**: Features not relevant for current input
- **Sparse patterns**: Specialized feature detection
- **Dense patterns**: General feature activation

### Layer Differences
- **Early layers**: Basic features (syntax, simple patterns)
- **Middle layers**: Complex patterns, semantic features
- **Late layers**: Task-specific, abstract representations

### Attention Analysis
- **Self-attention**: How tokens relate to each other
- **Cross-attention**: Relationships between different sequences
- **Head specialization**: Different attention heads focus on different aspects

## Saving and Sharing Results

### Export Options

```bash
# Export to various formats
neuronmap export \
    --data-path ./analysis_output \
    --format csv \
    --include-metadata \
    --output-file results.csv

# Export for research papers
neuronmap export \
    --format latex \
    --include-figures \
    --paper-ready
```

### Creating Reports

```python
# Generate comprehensive report
report = nm.generate_report(
    analysis_results=results,
    include_visualizations=True,
    include_statistics=True,
    output_format="html"
)

report.save("analysis_report.html")
```

## Common Analysis Patterns

### Pattern 1: Model Comparison
```python
# Compare two models
model_a_results = nm.analyze_text(model_a, text)
model_b_results = nm.analyze_text(model_b, text)

comparison = nm.compare_models(model_a_results, model_b_results)
nm.plot_model_comparison(comparison)
```

### Pattern 2: Input Sensitivity
```python
# Test model sensitivity to input changes
original = "The cat sat on the mat."
modified = "The dog sat on the mat."

sensitivity = nm.analyze_sensitivity(
    original_text=original,
    modified_text=modified,
    layers=[6, 11]
)
```

### Pattern 3: Feature Attribution
```python
# Identify which input tokens contribute most
attribution = nm.compute_feature_attribution(
    text=text,
    target_layer=11,
    method="integrated_gradients"
)

nm.plot_feature_attribution(attribution)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   neuronmap generate --batch-size 1 --max-length 128
   ```

2. **Slow Processing**
   ```bash
   neuronmap generate --device cuda --num-workers 4
   ```

3. **Visualization Errors**
   ```python
   # Check data shapes
   print(f"Layer data shape: {layer_data[6].shape}")
   
   # Reduce complexity
   nm.plot_activations(layer_data[6][:100, :100])  # First 100x100
   ```

## Next Steps

🎯 **Continue Learning:**
- [Understanding Results](03_understanding_results.md) - Deep dive into interpretation
- [Advanced Visualization](05_advanced_visualization.md) - Publication-ready plots
- [Interpretability Analysis](07_interpretability.md) - Advanced analysis techniques

## Exercise

Try this hands-on exercise:

1. Choose 3 different sentences with varying complexity
2. Analyze them with a BERT model
3. Compare activation patterns across layers 0, 6, and 11
4. Create visualizations showing the differences
5. Write a brief interpretation of the results

Share your results in our [Discord community](https://discord.gg/neuronmap)!

---

*Great work! You're now ready for more advanced analysis techniques! 🚀*
