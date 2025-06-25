# NeuronMap - Neural Network Activation Analysis Framework

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

*A comprehensive framework for analyzing neural network activations with multi-model support and advanced visualization capabilities.*

</div>

## 🚀 Quick Start (< 5 minutes)

### Option 1: Basic Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Install dependencies
pip install -r requirements.txt

# Quick test
python -m src.analysis.activation_extractor --help
```

### Option 2: Development Setup
```bash
# Create virtual environment
python -m venv neuronmap_env
source neuronmap_env/bin/activate  # On Windows: neuronmap_env\Scripts\activate

# Install in development mode
pip install -e .

# Run validation
python validate_section_1_1.py
```

### 🎯 Basic Usage Example

```python
from src.analysis.activation_extractor import ActivationExtractor
from src.visualization.core_visualizer import CoreVisualizer

# Initialize analyzer
analyzer = ActivationExtractor(model_name="distilgpt2")

# Extract activations
questions = ["Hello world", "How are you?", "What is AI?"]
results = analyzer.analyze_questions(questions)

# Visualize results
visualizer = CoreVisualizer()
visualizer.create_activation_heatmap(results)
```

**Expected Output:**
- Activation analysis CSV file with extracted neural patterns
- Interactive visualizations (PCA, t-SNE plots)
- Statistical summaries of activation patterns

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

### 🧠 **Multi-Model Support**
- **GPT Family**: GPT-2, GPT-Neo, GPT-J, DistilGPT-2
- **BERT Family**: BERT, DistilBERT, RoBERTa, ELECTRA
- **T5 Family**: T5, UL2, Flan-T5 (Coming Soon)
- **LLaMA Family**: LLaMA-7B, Alpaca, Vicuna (Coming Soon)
- **Domain-Specific**: CodeBERT, SciBERT, BioBERT (Coming Soon)

### 📊 **Advanced Analysis Capabilities**
- **Multi-layer activation extraction** in single pass
- **Attention pattern analysis** with head-level granularity
- **Residual stream tracking** between layers
- **MLP vs. Attention component** separation
- **Token-level activations** with statistical analysis

### 🎨 **Rich Visualization Suite**
- **Interactive plots** with Plotly integration
- **Heatmaps** for activation patterns
- **Dimensionality reduction** (PCA, t-SNE, UMAP)
- **Clustering analysis** with multiple algorithms
- **Statistical summaries** with publication-ready plots

### ⚙️ **Production-Ready Features**
- **Environment-based configuration** (dev/test/prod)
- **Hardware compatibility validation**
- **Memory optimization** for large models
- **Robust error handling** with graceful degradation
- **Comprehensive logging** and monitoring

## 🛠 Installation

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (CUDA-compatible)
- **Storage**: 5GB free space

### Platform-Specific Instructions

<details>
<summary><b>🐧 Linux (Ubuntu/Debian)</b></summary>

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv git -y

# Clone and setup
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "from src.utils.config import ConfigManager; print('✅ Installation successful!')"
```

</details>

<details>
<summary><b>🪟 Windows</b></summary>

```powershell
# Install Python from python.org or Microsoft Store
# Open PowerShell as Administrator

# Clone and setup
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "from src.utils.config import ConfigManager; print('✅ Installation successful!')"
```

</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git

# Clone and setup
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "from src.utils.config import ConfigManager; print('✅ Installation successful!')"
```

</details>

### 🐳 Docker Installation

```bash
# Pull pre-built image (coming soon)
docker pull emilio942/neuronmap:latest

# Or build from source
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
docker build -t neuronmap .

# Run container
docker run -it --gpus all -v $(pwd)/data:/app/data neuronmap
```

### 📦 Conda Installation

```bash
# Create conda environment
conda create -n neuronmap python=3.9 -y
conda activate neuronmap

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Clone and install
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
pip install -r requirements.txt
```

## 💻 Usage

### 🎯 Quick Analysis Workflow

```python
# 1. Configuration Setup
from src.utils.config import get_config_manager

config = get_config_manager()
config.switch_environment("development")  # or "production"

# 2. Generate Questions (Optional)
from src.data_generation.question_generator import QuestionGenerator

generator = QuestionGenerator()
questions = generator.generate_batch(["science", "history", "philosophy"], batch_size=50)

# 3. Extract Activations
from src.analysis.activation_extractor import ActivationExtractor

extractor = ActivationExtractor(model_name="distilgpt2", target_layer="transformer.h.5.mlp.c_proj")
results = extractor.process_questions(questions)

# 4. Analyze and Visualize
from src.visualization.core_visualizer import CoreVisualizer
from src.analysis.layer_inspector import LayerInspector

visualizer = CoreVisualizer()
inspector = LayerInspector()

# Generate comprehensive analysis
visualizer.create_analysis_dashboard(results)
inspector.generate_layer_comparison_report(results)
```

### 📊 Advanced Analysis Examples

<details>
<summary><b>Multi-Model Comparison</b></summary>

```python
from src.utils.multi_model_support import MultiModelAnalyzer

# Initialize multi-model analyzer
analyzer = MultiModelAnalyzer()

# Add models for comparison
models = ["distilgpt2", "gpt2", "microsoft/DialoGPT-medium"]
for model in models:
    analyzer.add_model(model)

# Run comparative analysis
questions = ["What is consciousness?", "Explain quantum mechanics", "How does photosynthesis work?"]
comparison_results = analyzer.compare_models(questions)

# Generate comparison report
analyzer.generate_comparison_report(comparison_results, output_dir="results/model_comparison/")
```

</details>

<details>
<summary><b>Attention Pattern Analysis</b></summary>

```python
from src.analysis.interpretability import AttentionAnalyzer

# Initialize attention analyzer
attention_analyzer = AttentionAnalyzer(model_name="bert-base-uncased")

# Analyze attention patterns
text = "The quick brown fox jumps over the lazy dog"
attention_data = attention_analyzer.extract_attention_patterns(text)

# Visualize attention heads
attention_analyzer.plot_attention_heatmap(attention_data, save_path="attention_analysis.png")

# Generate interpretability report
attention_analyzer.generate_head_analysis_report(attention_data)
```

</details>

<details>
<summary><b>Large-Scale Processing</b></summary>

```python
from src.utils.performance import BatchProcessor

# Setup batch processor for large datasets
processor = BatchProcessor(
    model_name="gpt2",
    batch_size=32,
    max_workers=4,
    memory_limit_gb=16
)

# Process large question dataset
large_dataset = "path/to/large_questions.jsonl"
results = processor.process_large_dataset(
    large_dataset,
    output_dir="results/large_scale/",
    checkpoint_every=1000
)

# Monitor processing progress
processor.get_processing_status()
```

</details>

## ⚙️ Configuration

NeuronMap uses a sophisticated configuration system supporting multiple environments:

### Environment-Based Configuration

```bash
# Development environment (detailed logging, conservative limits)
python -m src.utils.config --environment development --validate

# Production environment (optimized performance, strict limits)
python -m src.utils.config --environment production --startup-check

# Testing environment (fast execution, minimal resources)
python -m src.utils.config --environment testing --hardware-check
```

### Configuration Files

- `configs/models.yaml` - Model-specific parameters and layer mappings
- `configs/analysis.yaml` - Analysis settings and performance tuning
- `configs/visualization.yaml` - Visualization themes and export settings
- `configs/environment.yaml` - Environment-specific configurations

### Hardware Optimization

```python
from src.utils.config import get_config_manager

config = get_config_manager()

# Check hardware compatibility
compatibility_issues = config.validate_hardware_compatibility()
if compatibility_issues:
    print("Hardware issues:", compatibility_issues)

# Optimize for your system
analysis_config = config.get_analysis_config()
analysis_config.device = "cuda" if torch.cuda.is_available() else "cpu"
analysis_config.batch_size = 16 if torch.cuda.is_available() else 4
```

## 📚 Documentation

### 📖 Complete Documentation Suite

- **[Installation Guide](docs/installation/)** - Detailed OS-specific setup instructions
- **[API Reference](docs/api/)** - Complete API documentation with examples
- **[Tutorials](docs/tutorials/)** - Step-by-step guides for common use cases
- **[Research Guide](docs/research_guide.md)** - Scientific methodology and best practices
- **[Troubleshooting](docs/troubleshooting/)** - Common issues and solutions

### 🎓 Tutorials and Examples

- **[Getting Started](docs/tutorials/getting_started.md)** - Your first analysis in 10 minutes
- **[Multi-Model Analysis](docs/tutorials/multi_model.md)** - Comparing different architectures
- **[Attention Visualization](docs/tutorials/attention_analysis.md)** - Understanding attention patterns
- **[Large-Scale Processing](docs/tutorials/large_scale.md)** - Handling big datasets efficiently
- **[Custom Models](docs/tutorials/custom_models.md)** - Adding support for new architectures

### 📊 Example Notebooks

Explore our [Jupyter notebooks](examples/notebooks/) covering:
- Basic activation extraction and visualization
- Advanced statistical analysis of neural patterns
- Comparative studies across model families
- Reproducible research workflows

## 🔧 Troubleshooting

### Common Issues and Solutions

<details>
<summary><b>❌ ImportError: cannot import name 'get_config_manager'</b></summary>

**Cause**: Module path or dependency issues

**Solution**:
```bash
# Verify installation
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run from project root
cd /path/to/NeuronMap
python -c "from src.utils.config import get_config_manager; print('Success!')"
```

</details>

<details>
<summary><b>⚠️ CUDA out of memory</b></summary>

**Cause**: Model too large for available GPU memory

**Solution**:
```python
# Reduce batch size
from src.utils.config import get_config_manager
config = get_config_manager()
analysis_config = config.get_analysis_config()
analysis_config.batch_size = 4  # Reduce from default

# Enable memory optimization
analysis_config.memory_optimization.use_memory_efficient_attention = True
analysis_config.memory_optimization.offload_to_cpu = True
```

</details>

<details>
<summary><b>🐌 Slow processing</b></summary>

**Cause**: Suboptimal configuration for your hardware

**Solution**:
```bash
# Run hardware compatibility check
python -m src.utils.config --hardware-check

# Optimize configuration
python -m src.utils.config --environment production

# Use GPU if available
export CUDA_VISIBLE_DEVICES=0
```

</details>

<details>
<summary><b>📁 File not found errors</b></summary>

**Cause**: Incorrect working directory or file paths

**Solution**:
```bash
# Always run from project root
cd /path/to/NeuronMap

# Check file structure
ls -la configs/
ls -la src/

# Verify configuration
python -m src.utils.config --validate
```

</details>

### 🆘 Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/Emilio942/NeuronMap/issues)
- **Discussions**: [Community Q&A and discussions](https://github.com/Emilio942/NeuronMap/discussions)
- **Documentation**: [Complete guides and API reference](docs/)
- **Email**: [neuronmap.support@example.com](mailto:neuronmap.support@example.com)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Setup development environment
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Run tests
python -m pytest tests/
python validate_section_1_1.py
python validate_section_1_2.py

# Make your changes and submit a PR!
```

## 📈 Project Status

- ✅ **Section 1.1**: Project Structure Reorganization (Complete)
- ✅ **Section 1.2**: Configuration System Implementation (85% Complete)
- 🚧 **Section 1.3**: Documentation Enhancement (In Progress)
- 📋 **Section 2.x**: Multi-Model Support Extensions (Planned)
- 📋 **Section 3.x**: Advanced Analysis Methods (Planned)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** for model implementations
- **PyTorch** for deep learning infrastructure
- **Plotly** for interactive visualizations
- **Scientific Python Ecosystem** for numerical computing

---

<div align="center">

**[⭐ Star this repository](https://github.com/Emilio942/NeuronMap)** if you find it useful!

Made with ❤️ for the neural network analysis community

</div>
