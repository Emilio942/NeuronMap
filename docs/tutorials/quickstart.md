# Quick Start Tutorial

Get up and running with NeuronMap in under 5 minutes!

## 🚀 Installation

First, let's get NeuronMap installed:

```bash
# Clone the repository
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.utils.config import get_config_manager; print('✅ Installation successful!')"
```

## 🎯 Your First Analysis

Let's analyze activation patterns in a small language model:

### Step 1: Setup Configuration

```python
from src.utils.config import get_config_manager
from src.analysis.activation_extractor import ActivationExtractor
from src.visualization.core_visualizer import CoreVisualizer

# Initialize configuration
config = get_config_manager(environment="development")
print("✅ Configuration loaded")
```

### Step 2: Prepare Questions

```python
# Sample questions for analysis
questions = [
    "What is the capital of France?",
    "How does photosynthesis work?", 
    "Why do we dream?",
    "What causes gravity?",
    "How do computers work?"
]

print(f"📝 Prepared {len(questions)} questions")
```

### Step 3: Extract Activations

```python
# Initialize activation extractor
extractor = ActivationExtractor(
    model_name="distilgpt2",  # Small model for quick demo
    config=config.get_analysis_config()
)

# Extract activations
print("🔍 Extracting activations...")
results = extractor.extract_activations(
    questions=questions,
    layers=[3, 6, 9],  # Three representative layers
    batch_size=2
)

print(f"✅ Extracted activations: {results['activations'].shape}")
```

### Step 4: Visualize Results

```python
# Initialize visualizer
visualizer = CoreVisualizer(config=config.get_visualization_config())

# Create activation heatmap
fig = visualizer.plot_activation_heatmap(
    activations=results['activations'],
    labels=questions,
    title="Activation Patterns Across Questions"
)

# Save the plot
fig.savefig("my_first_analysis.png", dpi=300, bbox_inches='tight')
print("📊 Visualization saved as 'my_first_analysis.png'")
```

### Step 5: Basic Statistics

```python
# Compute basic statistics
stats = extractor.compute_activation_statistics(
    activations=results['activations'],
    metrics=["mean", "std", "sparsity"]
)

print("📈 Activation Statistics:")
for metric, value in stats.items():
    print(f"  {metric}: {value:.3f}")
```

## 🎉 Complete Example

Here's the complete code for your first analysis:

```python
from src.utils.config import get_config_manager
from src.analysis.activation_extractor import ActivationExtractor
from src.visualization.core_visualizer import CoreVisualizer

# Setup
config = get_config_manager(environment="development")
questions = [
    "What is the capital of France?",
    "How does photosynthesis work?", 
    "Why do we dream?",
    "What causes gravity?",
    "How do computers work?"
]

# Extract activations
extractor = ActivationExtractor(
    model_name="distilgpt2",
    config=config.get_analysis_config()
)

results = extractor.extract_activations(
    questions=questions,
    layers=[3, 6, 9],
    batch_size=2
)

# Visualize
visualizer = CoreVisualizer(config=config.get_visualization_config())
fig = visualizer.plot_activation_heatmap(
    activations=results['activations'],
    labels=questions,
    title="My First NeuronMap Analysis"
)

fig.savefig("my_first_analysis.png", dpi=300, bbox_inches='tight')
print("🎉 Analysis complete! Check 'my_first_analysis.png'")
```

## 🎛 Configuration Options

NeuronMap is highly configurable. Here are some key options:

### Model Selection

```python
# Available models (examples)
models = [
    "distilgpt2",      # Small, fast
    "gpt2",            # Medium size
    "bert-base",       # Encoder model
    "t5-small"         # Encoder-decoder
]

# Use different model
extractor = ActivationExtractor(model_name="bert-base")
```

### Analysis Parameters

```python
# Get current configuration
analysis_config = config.get_analysis_config()

# Modify parameters
analysis_config.batch_size = 8      # Larger batches
analysis_config.device = "cuda"     # Use GPU if available
analysis_config.precision = "float16"  # Mixed precision

# Apply modified configuration
extractor = ActivationExtractor(config=analysis_config)
```

### Visualization Options

```python
# Get visualization configuration
viz_config = config.get_visualization_config()

# Customize appearance
viz_config.figure_width = 15
viz_config.figure_height = 10
viz_config.color_scheme = "plasma"
viz_config.style.theme = "modern"

# Apply to visualizer
visualizer = CoreVisualizer(config=viz_config)
```

## 🔧 Troubleshooting

### Common Issues

**Memory Errors**
```python
# Reduce batch size
analysis_config.batch_size = 1
analysis_config.memory_optimization.max_memory_usage_gb = 4
```

**CUDA Errors**
```python
# Force CPU usage
analysis_config.device = "cpu"
```

**Import Errors**
```bash
# Make sure you're in the project directory
cd /path/to/neuronmap
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🎓 Next Steps

Now that you've completed your first analysis, here's what to explore next:

1. **📚 Tutorials**: Learn more advanced techniques
   - {doc}`advanced_analysis` - Multi-model comparisons
   - {doc}`custom_visualization` - Create custom plots
   - {doc}`batch_processing` - Process large datasets

2. **🔬 Research**: Scientific analysis
   - {doc}`../research/index` - Research methodology
   - {doc}`../research/experimental_design` - Design experiments

3. **⚙️ Configuration**: Advanced setup
   - {doc}`../configuration/index` - Configuration system
   - {doc}`../configuration/environments` - Environment management

4. **🎨 Visualization**: Advanced plotting
   - {doc}`../examples/visualization_gallery` - Plot examples
   - {doc}`../api/visualization` - Visualization API

5. **🤝 Community**: Get involved
   - {doc}`../development/contributing` - Contribute to NeuronMap
   - [GitHub Discussions](https://github.com/Emilio942/NeuronMap/discussions)

## 📋 Quick Reference

### Essential Imports

```python
from src.utils.config import get_config_manager
from src.analysis.activation_extractor import ActivationExtractor
from src.visualization.core_visualizer import CoreVisualizer
from src.data_generation.question_generator import QuestionGenerator
```

### Basic Workflow

```python
# 1. Setup
config = get_config_manager()

# 2. Extract
extractor = ActivationExtractor(model_name="your_model")
results = extractor.extract_activations(questions)

# 3. Visualize
visualizer = CoreVisualizer()
fig = visualizer.plot_results(results)

# 4. Save
fig.savefig("results.png")
```

### Configuration Commands

```bash
# Validate configuration
python -m src.utils.config --validate

# Switch environments
python -m src.utils.config --environment production

# Hardware check
python -m src.utils.config --hardware-check
```

---

```{admonition} 🎉 Congratulations!
:class: tip

You've completed your first NeuronMap analysis! You now have the basics to:
- Extract neural network activations
- Visualize activation patterns
- Configure the analysis pipeline
- Troubleshoot common issues

Ready for more advanced features? Check out the {doc}`../tutorials/index` for in-depth tutorials.
```

```{seealso}
- {doc}`../installation/index` - Detailed installation guide
- {doc}`../examples/basic_usage` - More basic examples
- {doc}`../api/index` - Complete API reference
- {doc}`../troubleshooting/index` - Comprehensive troubleshooting
```
