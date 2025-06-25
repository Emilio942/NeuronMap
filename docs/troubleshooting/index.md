# Troubleshooting Guide

This comprehensive guide helps you resolve common issues when using NeuronMap.

## 🚨 Quick Fixes

### Installation Issues

**Problem**: `ImportError: cannot import name 'get_config_manager'`

```bash
# Solution 1: Verify installation
pip install -r requirements.txt
python -c "from src.utils.config import get_config_manager; print('Success!')"

# Solution 2: Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solution 3: Reinstall in development mode
pip install -e .
```

**Problem**: `ModuleNotFoundError: No module named 'src'`

```bash
# Make sure you're in the project root directory
cd /path/to/NeuronMap
ls -la  # Should show src/ directory

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports
python -m src.analysis.activation_extractor --help
```

### Memory Issues

**Problem**: `CUDA out of memory`

```python
# Solution: Reduce batch size and enable memory optimization
from src.utils.config import get_config_manager

config = get_config_manager()
analysis_config = config.get_analysis_config()

# Reduce batch size
analysis_config.batch_size = 1

# Enable memory optimizations
analysis_config.memory_optimization.use_memory_efficient_attention = True
analysis_config.memory_optimization.clear_cache_between_batches = True
analysis_config.memory_optimization.offload_to_cpu = True

# Force CPU if necessary
analysis_config.device = "cpu"
```

**Problem**: System runs out of RAM

```python
# Enable CPU offloading and reduce memory usage
analysis_config.memory_optimization.max_memory_usage_gb = 4
analysis_config.memory_optimization.offload_to_cpu = True

# Process smaller batches
analysis_config.batch_size = 1
analysis_config.max_sequence_length = 256
```

### Configuration Issues

**Problem**: `Configuration validation failed`

```bash
# Check configuration files
python -m src.utils.config --validate

# Reset to default configuration
python -m src.utils.config --reset-defaults

# Validate specific environment
python -m src.utils.config --environment development --validate
```

## 🔧 Detailed Troubleshooting

### GPU and CUDA Issues

<details>
<summary><b>CUDA not available</b></summary>

**Symptoms**: 
- `torch.cuda.is_available()` returns `False`
- Error: "CUDA device requested but not available"

**Diagnosis**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current device: {torch.cuda.current_device()}")
```

**Solutions**:
```bash
# 1. Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Check NVIDIA drivers
nvidia-smi

# 3. Verify CUDA installation
nvcc --version

# 4. Force CPU usage if CUDA issues persist
export CUDA_VISIBLE_DEVICES=""
```

</details>

<details>
<summary><b>GPU memory management</b></summary>

**Symptoms**:
- "CUDA out of memory" errors
- System freezing during analysis

**Solutions**:
```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Monitor GPU memory
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB")

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

# Enable memory efficient attention
analysis_config.memory_optimization.use_memory_efficient_attention = True
```

</details>

### Model Loading Issues

<details>
<summary><b>Model not found or loading errors</b></summary>

**Symptoms**:
- "Model not found" errors
- Slow model loading
- Network timeouts

**Solutions**:
```python
# 1. Check model availability
from transformers import AutoTokenizer, AutoModel
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")

# 2. Use local model cache
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

# 3. Download models manually
from transformers import AutoModel
model = AutoModel.from_pretrained("gpt2", cache_dir="./models")

# 4. Use smaller models for testing
test_models = ["distilgpt2", "distilbert-base-uncased"]
```

</details>

<details>
<summary><b>Layer name resolution issues</b></summary>

**Symptoms**:
- "Layer not found" errors
- Incorrect activation extraction

**Diagnosis**:
```python
from src.analysis.activation_extractor import ActivationExtractor

extractor = ActivationExtractor(model_name="gpt2")
extractor.print_model_structure()

# Or manually inspect model
model = extractor.model
for name, module in model.named_modules():
    print(f"{name}: {type(module)}")
```

**Solutions**:
```python
# 1. Use correct layer names for your model
layer_mappings = {
    "gpt2": "transformer.h.{layer}",
    "bert-base": "bert.encoder.layer.{layer}",
    "t5-small": "encoder.block.{layer}"
}

# 2. Update model configuration
config = get_config_manager()
model_config = config.get_model_config("gpt2")
print(f"Correct layer pattern: {model_config.layers.attention}")

# 3. Use layer indices instead of names
extractor.extract_activations(questions, layers=[0, 5, 11])  # Layer indices
```

</details>

### Data Processing Issues

<details>
<summary><b>Question loading and format issues</b></summary>

**Symptoms**:
- "No questions loaded" warnings
- JSON parsing errors
- Empty datasets

**Solutions**:
```python
# 1. Validate question file format
import json

def validate_questions_file(filepath):
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                if 'question' not in data:
                    print(f"Line {i+1}: Missing 'question' key")
                    return False
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

# 2. Convert text file to JSONL
def convert_text_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            question = line.strip()
            if question:
                json.dump({"question": question}, outfile)
                outfile.write('\n')

# 3. Create sample questions for testing
sample_questions = [
    {"question": "What is artificial intelligence?"},
    {"question": "How do neural networks learn?"},
    {"question": "What is the purpose of attention mechanisms?"}
]

with open("test_questions.jsonl", 'w') as f:
    for q in sample_questions:
        json.dump(q, f)
        f.write('\n')
```

</details>

<details>
<summary><b>Activation extraction failures</b></summary>

**Symptoms**:
- "No activations captured" warnings
- NaN values in results
- Dimension mismatch errors

**Diagnosis**:
```python
# Debug activation extraction
def debug_activation_extraction(extractor, question):
    print(f"Testing question: {question}")
    
    # Check tokenization
    tokens = extractor.tokenizer.encode(question)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    # Check model forward pass
    with torch.no_grad():
        inputs = extractor.tokenizer(question, return_tensors="pt")
        outputs = extractor.model(**inputs)
        print(f"Model output shape: {outputs.last_hidden_state.shape}")
    
    # Test activation hook
    activations = extractor.extract_activations([question], layers=[0])
    print(f"Extracted activations shape: {activations.shape}")

# Run diagnostics
extractor = ActivationExtractor(model_name="distilgpt2")
debug_activation_extraction(extractor, "Test question")
```

**Solutions**:
```python
# 1. Increase sequence length limit
analysis_config.max_sequence_length = 1024

# 2. Handle tokenization issues
extractor.tokenizer.pad_token = extractor.tokenizer.eos_token

# 3. Validate input questions
def clean_questions(questions):
    cleaned = []
    for q in questions:
        if len(q.strip()) > 0 and len(q) < 512:  # Remove empty and very long questions
            cleaned.append(q.strip())
    return cleaned
```

</details>

### Visualization Issues

<details>
<summary><b>Plotting and figure generation issues</b></summary>

**Symptoms**:
- Blank figures
- Font rendering issues
- Memory errors during plotting

**Solutions**:
```python
# 1. Backend configuration
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# 2. Font issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# 3. Memory management for large plots
def plot_with_memory_management(data):
    plt.figure(figsize=(10, 8))
    plt.plot(data)
    plt.savefig('output.png', dpi=150, bbox_inches='tight')
    plt.close()  # Important: close figure to free memory
    plt.clf()    # Clear figure
    
# 4. Interactive plotting issues
# Use static plotting for headless environments
viz_config.interactive = False
```

</details>

### Configuration and Environment Issues

<details>
<summary><b>Environment configuration problems</b></summary>

**Symptoms**:
- Configuration validation failures
- Environment switching errors
- YAML parsing issues

**Solutions**:
```bash
# 1. Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/models.yaml'))"

# 2. Reset configuration
python -m src.utils.config --reset-defaults

# 3. Create minimal configuration
cat > configs/minimal.yaml << EOF
models:
  distilgpt2:
    name: "distilgpt2"
    type: "gpt"
    layers:
      attention: "transformer.h.{layer}.attn"
      mlp: "transformer.h.{layer}.mlp"
      total_layers: 6
analysis:
  batch_size: 1
  device: "cpu"
EOF

# 4. Use environment variables
export NEURONMAP_ENV=development
export NEURONMAP_CONFIG_DIR=/path/to/configs
```

</details>

## 🔍 Diagnostic Tools

### System Information Script

```python
#!/usr/bin/env python3
"""System diagnostics for NeuronMap troubleshooting."""

import sys
import torch
import platform
import psutil
import subprocess
from pathlib import Path

def print_system_info():
    print("=== SYSTEM INFORMATION ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"\n=== MEMORY INFORMATION ===")
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1e9:.1f} GB")
    print(f"Available RAM: {memory.available / 1e9:.1f} GB")
    print(f"Used RAM: {memory.percent:.1f}%")
    
    print(f"\n=== NEURONMAP INSTALLATION ===")
    try:
        from src.utils.config import get_config_manager
        print("✅ NeuronMap imports successfully")
        
        config = get_config_manager()
        print("✅ Configuration manager works")
        
        # Test module imports
        modules_to_test = [
            "src.analysis.activation_extractor",
            "src.visualization.core_visualizer", 
            "src.data_generation.question_generator"
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"✅ {module}")
            except Exception as e:
                print(f"❌ {module}: {e}")
                
    except Exception as e:
        print(f"❌ NeuronMap installation issue: {e}")

if __name__ == "__main__":
    print_system_info()
```

### Configuration Validator

```python
#!/usr/bin/env python3
"""Configuration validation and repair tool."""

import yaml
from pathlib import Path

def validate_and_repair_config():
    config_dir = Path("configs")
    
    if not config_dir.exists():
        print("Creating configs directory...")
        config_dir.mkdir()
    
    # Check each config file
    config_files = ["models.yaml", "analysis.yaml", "visualization.yaml", "environment.yaml"]
    
    for config_file in config_files:
        config_path = config_dir / config_file
        
        if not config_path.exists():
            print(f"Missing {config_file}, creating default...")
            create_default_config(config_path, config_file)
        else:
            try:
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
                print(f"✅ {config_file} is valid")
            except Exception as e:
                print(f"❌ {config_file} has errors: {e}")
                print("Creating backup and default config...")
                config_path.rename(config_path.with_suffix('.yaml.bak'))
                create_default_config(config_path, config_file)

def create_default_config(path, config_type):
    """Create default configuration files."""
    defaults = {
        "models.yaml": {
            "models": {
                "distilgpt2": {
                    "name": "distilgpt2",
                    "type": "gpt",
                    "layers": {
                        "attention": "transformer.h.{layer}.attn",
                        "mlp": "transformer.h.{layer}.mlp", 
                        "total_layers": 6
                    }
                }
            }
        },
        "analysis.yaml": {
            "analysis": {
                "batch_size": 1,
                "device": "auto",
                "max_sequence_length": 512
            }
        },
        "visualization.yaml": {
            "visualization": {
                "figure_width": 12,
                "figure_height": 8,
                "color_scheme": "viridis"
            }
        },
        "environment.yaml": {
            "environment": {
                "environment": "development",
                "log_level": "INFO"
            }
        }
    }
    
    with open(path, 'w') as f:
        yaml.dump(defaults[config_type], f, default_flow_style=False)

if __name__ == "__main__":
    validate_and_repair_config()
```

## 📞 Getting Help

### Self-Service Resources

1. **Documentation**: [Complete documentation](https://neuronmap.readthedocs.io)
2. **API Reference**: {doc}`../api/index`
3. **Examples**: {doc}`../examples/index`
4. **Configuration Guide**: {doc}`../configuration/index`

### Community Support

1. **GitHub Issues**: [Report bugs](https://github.com/Emilio942/NeuronMap/issues)
2. **Discussions**: [Community Q&A](https://github.com/Emilio942/NeuronMap/discussions)
3. **Stack Overflow**: Tag your questions with `neuronmap`

### Professional Support

- **Email**: support@neuronmap.org
- **Enterprise Support**: Available for commercial users

### When Reporting Issues

Please include:

1. **System Information**: Run the diagnostic script above
2. **Error Messages**: Complete error traceback
3. **Configuration**: Your config files (without sensitive data)
4. **Reproduction Steps**: Minimal example that reproduces the issue
5. **Expected vs Actual Behavior**: What you expected vs what happened

### Emergency Fixes

For critical issues, try these emergency fixes:

```bash
# Nuclear option: Fresh installation
rm -rf neuronmap_env/
git clean -xfd
pip cache purge
python -m venv neuronmap_env
source neuronmap_env/bin/activate
pip install -r requirements.txt

# Reset all configuration
rm -rf configs/
python -m src.utils.config --reset-defaults

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
export NEURONMAP_DEVICE=cpu
```

---

```{admonition} 💡 Pro Tip
:class: tip

Most issues can be resolved by:
1. Checking you're in the correct directory
2. Verifying your Python environment
3. Resetting configuration to defaults
4. Using CPU mode when GPU issues occur

When in doubt, start with the diagnostic script to identify the problem area.
```

```{seealso}
- {doc}`../installation/index` - Installation guide
- {doc}`../configuration/index` - Configuration system
- {doc}`../api/index` - API reference
- [GitHub Issues](https://github.com/Emilio942/NeuronMap/issues) - Report bugs
```
