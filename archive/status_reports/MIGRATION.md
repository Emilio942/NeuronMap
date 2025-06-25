# Migration Guide - From Old Structure to New Modular Structure

This guide helps you migrate from the old flat structure to the new modular NeuronMap structure.

## What Changed

### Old Structure
```
NeuronMap/
├── fragenG.py
├── run.py
├── visualizer.py
├── aufgabenliste.md
└── LICENSE
```

### New Structure
```
NeuronMap/
├── src/
│   ├── data_generation/
│   │   └── question_generator.py      # Refactored fragenG.py
│   ├── analysis/
│   │   └── activation_extractor.py    # Refactored run.py
│   ├── visualization/
│   │   └── visualizer.py              # Refactored visualizer.py
│   └── utils/
│       ├── config.py                  # New configuration system
│       ├── file_handlers.py           # File I/O utilities
│       └── validation.py              # Input validation
├── configs/
│   ├── models.yaml                    # Model configurations
│   └── experiments.yaml               # Experiment configurations
├── data/
│   ├── raw/                          # Generated questions
│   ├── processed/                    # Activation results
│   └── outputs/                      # Visualizations
├── tests/                            # Unit tests
├── docs/                             # Documentation
├── main.py                           # New CLI interface
├── setup.py                          # Package installation
├── requirements.txt                  # Dependencies
├── setup.sh                          # Quick setup script
└── README.md                         # Comprehensive documentation
```

## Migration Steps

### 1. Keep Your Old Files (Backup)
Your original files (`fragenG.py`, `run.py`, `visualizer.py`) are still there and functional. The new system doesn't replace them - it provides a better organized alternative.

### 2. Try the New System

#### Install the new system:
```bash
./setup.sh
```

#### Or manually:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Validate setup
python main.py validate
```

### 3. Configuration Migration

#### Old Way (hardcoded in scripts):
```python
MODEL_NAME = "distilgpt2"
TARGET_LAYER_NAME = "transformer.h.5.mlp.c_proj"
OUTPUT_FILE = "activation_results.csv"
```

#### New Way (in configs/experiments.yaml):
```yaml
default:
  activation_extraction:
    model_config: "default"  # Refers to configs/models.yaml
    target_layer: "transformer.h.5.mlp.c_proj"
    output_file: "data/processed/activation_results.csv"
```

### 4. Usage Migration

#### Old Way:
```bash
# Step 1: Generate questions
python fragenG.py

# Step 2: Extract activations  
python run.py

# Step 3: Visualize
python visualizer.py
```

#### New Way:
```bash
# All-in-one pipeline
python main.py pipeline --config dev

# Or step by step
python main.py generate --config dev
python main.py extract --target-layer "transformer.h.5.mlp.c_proj"
python main.py visualize --methods pca tsne heatmap
```

## Key Improvements

### 1. **Better Error Handling**
- Comprehensive validation before execution
- Graceful error recovery
- Detailed logging and debugging

### 2. **Configuration Management**
- YAML-based configuration files
- Environment-specific settings (dev, prod)
- Model-specific layer mappings
- Easy experiment management

### 3. **CLI Interface**
- Unified command-line interface
- Built-in help and validation
- Progress indicators and logging
- Configuration inspection tools

### 4. **Modular Design**
- Separate modules for each functionality
- Easy to extend and customize
- Better testing capabilities
- Cleaner code organization

### 5. **Documentation**
- Comprehensive README
- API documentation
- Configuration guides
- Troubleshooting help

## Gradual Migration Strategy

### Phase 1: Test the New System
```bash
# Quick test with small dataset
python main.py pipeline --config dev
```

### Phase 2: Migrate Your Configurations
1. Copy your settings from old scripts to `configs/experiments.yaml`
2. Add any custom models to `configs/models.yaml`
3. Test with your specific configuration

### Phase 3: Full Migration
1. Use the new CLI for all operations
2. Leverage the new visualization options
3. Take advantage of the validation and error handling

## Backward Compatibility

Your old scripts will still work! The new system is designed to coexist with your existing workflow:

- Old output files are compatible with new visualizer
- Same file formats (JSONL for questions, CSV for activations)
- Same model support (any HuggingFace transformer)
- Same visualization types (PCA, t-SNE, heatmaps)

## Getting Help

### Commands to explore the new system:
```bash
# Show help
python main.py --help

# Validate your setup
python main.py validate

# Show available configurations
python main.py config --models --experiments

# List available layers in a model
python main.py extract --list-layers

# Test question generation only
python main.py generate --config dev
```

### If you encounter issues:
1. Check the validation: `python main.py validate`
2. Look at the logs in `neuronmap.log`
3. Refer to the README.md troubleshooting section
4. Your old scripts are still there as fallback

## Advanced Features You Can Now Use

### 1. Multiple Experiment Configurations
Create different configs for different research questions:
```yaml
layer_comparison:
  description: "Compare activations across layers"
  activation_extraction:
    target_layers: ["transformer.h.2.mlp.c_proj", "transformer.h.4.mlp.c_proj"]

model_comparison:
  description: "Compare different models"
  models: ["gpt2_small", "bert_base"]
```

### 2. Batch Processing
```bash
for config in dev prod layer_comparison; do
    python main.py pipeline --config $config
done
```

### 3. Custom Visualization
Extend the visualizer class for custom plots and analysis.

### 4. Better Model Support
Easy addition of new models through configuration files.

---

**The new system is designed to be a powerful upgrade while keeping everything you love about the original workflow!**
