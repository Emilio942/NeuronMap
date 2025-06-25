# NeuronMap - Neural Network Activation Analysis Toolkit

NeuronMap is a comprehensive tool for analyzing neural network activations through question generation, activation extraction, and advanced analysis. It provides insights into how neural networks process different types of input by examining internal layer activations across multiple model architectures.

## 🚀 Features

### Core Functionality
- **Question Generation**: Generate diverse test questions using Ollama
- **Multi-Model Support**: Extract activations from GPT, BERT, T5, Llama, and other transformers
- **Multi-Layer Extraction**: Simultaneous extraction from multiple layers with memory optimization
- **Advanced Analysis**: Comprehensive statistical analysis, clustering, and correlation analysis
- **Attention Analysis**: Specialized tools for analyzing attention patterns and circuits
- **Visualization**: Create PCA, t-SNE, heatmap, and interactive visualizations

### Advanced Capabilities
- **Cross-Model Comparison**: Compare activation patterns between different model architectures
- **Layer Evolution Tracking**: Analyze how information flows through network layers
- **Neuron-Level Analysis**: Individual neuron statistics, clustering, and importance analysis
- **Attention Pattern Extraction**: Head-wise attention analysis and circuit discovery
- **Memory-Efficient Processing**: HDF5 storage and batch processing for large-scale analysis
- **Real-time Monitoring**: System resource monitoring and health checks

### Technical Features
- **Configurable**: YAML-based configuration system for models and experiments
- **CLI Interface**: Comprehensive command-line interface with subcommands
- **Extensible**: Modular design for easy extension and customization
- **Error Handling**: Robust error handling with retry logic and graceful degradation
- **Validation**: Input validation and system requirement checks

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) (for question generation)
- CUDA-compatible GPU (optional, but recommended for large models)
- 16GB+ RAM recommended for large models

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Ollama Setup

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull a model for question generation:
   ```bash
   ollama pull deepseek-r1:32b
   # or use a smaller model like:
   ollama pull llama2:7b
   ```

## 🏗️ Project Structure

```
neuronmap/
├── src/
│   ├── data_generation/       # Question generation using Ollama
│   ├── analysis/              # Multi-model activation extraction and analysis
│   │   ├── activation_extractor.py      # Original single-layer extractor
│   │   ├── multi_model_extractor.py     # Multi-model, multi-layer extractor
│   │   ├── advanced_analysis.py         # Statistical and correlation analysis
│   │   └── attention_analysis.py        # Attention-specific analysis
│   ├── visualization/         # Visualization tools and interactive plots
│   └── utils/                 # Configuration, monitoring, and error handling
├── configs/                   # YAML configuration files
│   ├── models.yaml           # Model configurations and layer mappings
│   └── experiments.yaml      # Experiment configurations
├── data/                      # Data storage (raw, processed, outputs)
├── tests/                     # Unit tests and integration tests
├── docs/                      # Documentation
├── main.py                    # Main CLI interface
├── requirements.txt           # Python dependencies
├── MULTI_MODEL_GUIDE.md       # Detailed guide for new features
└── setup.py                   # Package setup
```

## 🚀 Quick Start

### 1. Validate Setup

```bash
python main.py validate
```

### 2. Discover Available Models

```bash
# List all configured models
python main.py discover

# Test model availability
python main.py discover --test-availability
```

### 3. Generate Questions

```bash
python main.py generate --config dev
```

### 4. Multi-Layer Activation Extraction

```bash
# Discover model layers
python main.py multi-extract --discover-layers --model gpt2_small

# Extract from multiple layers
python main.py multi-extract --model gpt2_small --layer-range 0 6

# Extract from specific layers
python main.py multi-extract --model bert_base --layers "encoder.layer.0.attention.output.dense" "encoder.layer.6.attention.output.dense"
```

### 5. Advanced Analysis

```bash
# Comprehensive analysis report
python main.py analyze --input-file data/outputs/activations.h5

# Analyze specific layer
python main.py analyze --input-file data/outputs/activations.h5 --layer "transformer.h.5.attn.c_attn"

# Compare multiple layers
python main.py analyze --input-file data/outputs/activations.h5 --compare-layers "transformer.h.2.attn.c_attn" "transformer.h.5.attn.c_attn"
```

### 6. Visualization

```bash
python main.py visualize --methods pca tsne heatmap
```

### 7. Run Full Pipeline

```bash
python main.py pipeline --config dev
```

## 📋 Configuration

NeuronMap uses YAML configuration files for all settings:

### Model Configuration (`configs/models.yaml`)

```yaml
models:
  gpt2_small:
    name: "gpt2"
    type: "gpt"
    layers:
      attention: "transformer.h.{layer}.attn.c_attn"
      mlp: "transformer.h.{layer}.mlp.c_proj"
      total_layers: 12
```

### Experiment Configuration (`configs/experiments.yaml`)

```yaml
default:
  question_generation:
    ollama_host: "http://localhost:11434"
    model_name: "deepseek-r1:32b"
    num_questions: 1000
    batch_size: 20
    
  activation_extraction:
    model_config: "gpt2_small"
    target_layer: "transformer.h.5.mlp.c_proj"
    device: "auto"
    
  visualization:
    methods: ["pca", "tsne", "heatmap"]
```

## 🔧 CLI Commands

### Generate Questions

```bash
# Basic generation
python main.py generate

# With custom configuration
python main.py generate --config dev

# With custom prompt template
python main.py generate --prompt-file custom_prompt.txt
```

### Extract Activations

```bash
# List available layers in model
python main.py extract --list-layers

# Extract with specific parameters
python main.py extract --target-layer "transformer.h.3.mlp.c_proj" --questions-file questions.jsonl

# Use different configuration
python main.py extract --config prod
```

### Create Visualizations

```bash
# All visualization methods
python main.py visualize

# Specific methods only
python main.py visualize --methods pca tsne

# Custom input file
python main.py visualize --input-file custom_activations.csv
```

### Show Configuration

```bash
# Show current configuration
python main.py config

# Show available models
python main.py config --models

# Show available experiments
python main.py config --experiments
```

## 📊 Understanding the Output

### Question Generation
- Creates `data/raw/generated_questions.jsonl` with questions in JSON Lines format
- Each line contains: `{"question": "What is the capital of France?"}`

### Activation Extraction
- Creates `data/processed/activation_results.csv` with:
  - `question_id`: Unique identifier
  - `question`: Original question text
  - `activation_vector`: List of activation values
  - `layer_name`: Target layer name
  - `model_name`: Model used

### Visualizations
- **PCA Scatter**: 2D projection showing activation clusters
- **t-SNE Scatter**: Non-linear dimensionality reduction
- **Heatmap**: Raw activation patterns across questions/neurons
- **Statistics**: Distribution and variance analysis

## 🔬 Research Applications

NeuronMap can be used for:

- **Interpretability Research**: Understanding what different layers learn
- **Model Comparison**: Comparing activation patterns across models
- **Layer Analysis**: Finding optimal layers for specific tasks
- **Activation Clustering**: Identifying similar question types
- **Feature Analysis**: Understanding neuron specialization

## 🛠️ Advanced Usage

### Custom Model Support

Add new models to `configs/models.yaml`:

```yaml
my_custom_model:
  name: "microsoft/DialoGPT-medium"
  type: "gpt"
  layers:
    attention: "transformer.h.{layer}.attn.c_attn"
    mlp: "transformer.h.{layer}.mlp.c_proj"
    total_layers: 24
```

### Batch Processing

Process multiple experiments:

```bash
for config in dev prod experiment1; do
    python main.py pipeline --config $config
done
```

### Custom Visualization

Extend the visualizer class:

```python
from src.visualization.visualizer import ActivationVisualizer

class CustomVisualizer(ActivationVisualizer):
    def custom_plot(self, data):
        # Your custom visualization code
        pass
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_question_generator.py
```

## 📚 Documentation

Full documentation is available in the `docs/` directory:

- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Contributing](docs/contributing.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   ollama list
   # Start Ollama if needed
   ollama serve
   ```

2. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use CPU mode: `--config dev` (which forces CPU)
   - Use smaller model

3. **Import Errors**
   ```bash
   # Install in development mode
   pip install -e .
   ```

4. **Layer Not Found**
   ```bash
   # List available layers
   python main.py extract --list-layers
   ```

## 🎯 Roadmap

- [ ] Support for more model architectures (LLaMA, Claude, etc.)
- [ ] Interactive visualization dashboard
- [ ] Batch experiment management
- [ ] Model comparison tools
- [ ] Export to popular ML platforms
- [ ] Integration with Weights & Biases
- [ ] Docker containers for easy deployment

## 📞 Support

For questions and support:

- Open an [issue](https://github.com/Emilio942/NeuronMap/issues)
- Check the [documentation](docs/)
- Review [troubleshooting guide](docs/troubleshooting.md)
- Visit the [project repository](https://github.com/Emilio942/NeuronMap)

---

Made with ❤️ for the AI research community
