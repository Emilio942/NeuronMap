# Quick Start Guide

Welcome to NeuronMap! This guide will get you up and running with neural network activation analysis in just 5 minutes.

## Installation

### Option 1: pip (Recommended)
```bash
pip install neuronmap
```

### Option 2: From Source
```bash
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
pip install -e .
```

### Option 3: Docker
```bash
docker pull emilio942/neuronmap:latest
docker run -it emilio942/neuronmap:latest
```

## Verify Installation

```bash
neuronmap --version
```

## Your First Analysis

### 1. Basic Model Analysis

Let's analyze a simple BERT model:

```bash
# Generate activations for a sample text
neuronmap generate \
    --model-name bert-base-uncased \
    --input-text "Hello, this is a test sentence." \
    --output-dir ./my_first_analysis

# Extract and visualize activations
neuronmap extract \
    --data-path ./my_first_analysis \
    --layers 0 6 11 \
    --output-format json

# Create visualizations
neuronmap visualize \
    --data-path ./my_first_analysis \
    --plot-type heatmap \
    --save-path ./my_first_analysis/plots
```

### 2. Using Python API

```python
from neuronmap import NeuronMap, ModelConfig

# Initialize NeuronMap
nm = NeuronMap()

# Load a model
config = ModelConfig(
    name="bert-base-uncased",
    model_type="huggingface"
)
model = nm.load_model(config)

# Analyze a text
text = "Hello, this is a test sentence."
results = nm.analyze_text(model, text)

# Visualize results
nm.plot_activations(results, plot_type="heatmap")
```

### 3. Exploring Results

Your analysis will generate:
- **Activation maps**: `activations.json`
- **Visualizations**: `plots/heatmap.png`
- **Statistics**: `stats.json`
- **Metadata**: `metadata.yaml`

## What's Next?

🎯 **Try These Next Steps:**

1. **Explore Different Models**: Try GPT-2, RoBERTa, or your custom model
2. **Analyze Different Inputs**: Use longer texts, different languages, or specific domains
3. **Advanced Visualizations**: Create 3D plots, attention maps, or interactive dashboards
4. **Compare Models**: Analyze multiple models side-by-side

## Quick Examples

### Analyze a Custom Model
```bash
neuronmap generate --model-path ./my_model.pt --input-text "Custom analysis"
```

### Batch Processing
```bash
neuronmap generate --input-file inputs.txt --batch-size 32
```

### Interactive Dashboard
```bash
neuronmap dashboard --data-path ./analysis_results
```

### Export for Research
```bash
neuronmap export --format csv --include-metadata
```

## Common Issues

### Model Download Fails
```bash
# Use local cache
export TRANSFORMERS_CACHE=/path/to/cache
neuronmap generate --model-name bert-base-uncased --use-cache
```

### Memory Issues
```bash
# Reduce batch size
neuronmap generate --batch-size 1 --max-length 128
```

### GPU Not Detected
```bash
# Force CPU mode
neuronmap generate --device cpu
```

## Help and Support

- 📖 **Full Documentation**: [docs/](../docs/)
- 💬 **Community Chat**: [Discord](https://discord.gg/neuronmap)
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/Emilio942/NeuronMap/issues)
- ❓ **Ask Questions**: [GitHub Discussions](https://github.com/Emilio942/NeuronMap/discussions)

## Next Tutorial

👉 Continue with [Basic Analysis](02_basic_analysis.md) to learn more detailed analysis techniques.

---

*Congratulations! You've completed your first NeuronMap analysis! 🎉*
