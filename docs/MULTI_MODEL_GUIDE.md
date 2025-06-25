# Multi-Model and Advanced Analysis Guide

This guide covers the new multi-model support and advanced analysis capabilities added to NeuronMap.

## New Features Overview

### 1. Multi-Model Support

NeuronMap now supports a wide range of transformer models:

- **GPT Family**: GPT-2 (small/medium/large), GPT-Neo, GPT-J, CodeGen
- **BERT Family**: BERT, RoBERTa, DeBERTa, DistilBERT, SciBERT, BioBERT
- **T5 Family**: T5, Flan-T5, CodeT5
- **Llama Family**: Llama-2 (with proper access)
- **Domain-specific**: SciBERT, BioBERT, CodeGen

### 2. Multi-Layer Extraction

Extract activations from multiple layers simultaneously:

- Automatic layer discovery and mapping
- Batch processing for efficiency
- HDF5 output format for large datasets
- Memory-optimized processing

### 3. Advanced Analysis

Comprehensive analysis tools for activation patterns:

- Statistical analysis (mean, std, skewness, kurtosis, sparsity)
- Neuron-level analysis and clustering
- Cross-layer comparisons
- Dimensionality reduction (PCA, t-SNE)
- Correlation analysis

### 4. Attention Analysis

Specialized tools for attention mechanisms:

- Attention pattern extraction
- Head-wise analysis
- Distance-based attention analysis
- Token-type attention patterns
- Circuit discovery

## Command Line Interface

### Multi-Model Extraction

```bash
# Discover available model layers
python main.py multi-extract --discover-layers --model gpt2_small

# Extract from specific layers
python main.py multi-extract --model bert_base --layers "encoder.layer.0.attention.output.dense" "encoder.layer.6.attention.output.dense"

# Extract from layer range
python main.py multi-extract --model gpt2_small --layer-range 0 6

# Custom output format
python main.py multi-extract --model t5_small --layer-range 0 3 --output-format both
```

### Advanced Analysis

```bash
# Comprehensive analysis report
python main.py analyze --input-file data/outputs/activations.h5

# Analyze specific layer
python main.py analyze --input-file data/outputs/activations.h5 --layer "transformer.h.5.attn.c_attn"

# Compare multiple layers
python main.py analyze --input-file data/outputs/activations.h5 --compare-layers "transformer.h.2.attn.c_attn" "transformer.h.5.attn.c_attn" "transformer.h.8.attn.c_attn"
```

### Model Discovery

```bash
# List available models
python main.py discover

# Test model availability
python main.py discover --test-availability
```

## Configuration

### Model Configuration (configs/models.yaml)

Models are configured with:

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

### Extraction Settings

```yaml
extraction_settings:
  gpt:
    preferred_dtype: "float16"
    batch_size: 1
    max_sequence_length: 1024
```

## Output Formats

### HDF5 Format (Recommended)

Efficient storage for large datasets:

```
activations.h5
├── questions/
│   ├── question_0
│   ├── question_1
│   └── ...
├── activations/
│   ├── question_0/
│   │   ├── transformer_h_0_attn_c_attn/
│   │   │   ├── vector (dataset)
│   │   │   └── stats (attributes)
│   │   └── ...
│   └── ...
└── metadata (attributes)
```

### Analysis Reports

JSON format with comprehensive statistics:

```json
{
  "layer_analyses": {
    "layer_name": {
      "overall_statistics": {...},
      "per_question_statistics": [...],
      "neuron_analysis": {...},
      "correlation_analysis": {...}
    }
  },
  "cross_layer_analysis": {...},
  "dimensionality_reduction": {...},
  "clustering": {...}
}
```

## Usage Examples

### Example 1: Multi-Model Comparison

```bash
# Extract activations from different models
python main.py multi-extract --model gpt2_small --layer-range 0 6 --output-format hdf5
python main.py multi-extract --model bert_base --layer-range 0 6 --output-format hdf5

# Analyze and compare
python main.py analyze --input-file data/outputs/gpt2_activations.h5 --output analysis/gpt2/
python main.py analyze --input-file data/outputs/bert_activations.h5 --output analysis/bert/
```

### Example 2: Layer-by-Layer Analysis

```bash
# Extract from all layers
python main.py multi-extract --model gpt2_medium --layer-range 0 24

# Analyze specific layers of interest
python main.py analyze --input-file data/outputs/activations.h5 --layer "transformer.h.6.attn.c_attn"
python main.py analyze --input-file data/outputs/activations.h5 --layer "transformer.h.12.attn.c_attn"
python main.py analyze --input-file data/outputs/activations.h5 --layer "transformer.h.18.attn.c_attn"
```

### Example 3: Attention Analysis

```bash
# Generate attention-focused extraction
python main.py multi-extract --model gpt2_small --layers $(python -c "
for i in range(12):
    print(f'transformer.h.{i}.attn.c_attn', end=' ')
")

# Analyze attention patterns
python main.py attention --input data/outputs/activations.h5
```

## Performance Considerations

### Memory Management

- Use HDF5 format for large datasets
- Adjust batch size based on available memory
- Use float16 for memory efficiency
- Enable memory monitoring

### GPU Usage

- Automatic device selection
- Memory-optimized loading
- Gradient checkpointing for large models
- Multi-GPU support (where available)

### Processing Speed

- Batch processing for multiple questions
- Parallel analysis where possible
- Efficient numpy operations
- Progress tracking with tqdm

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model availability with `discover --test-availability`
   - Verify internet connection for model downloads
   - Check available memory/disk space

2. **Layer Not Found**
   - Use `--discover-layers` to see available layers
   - Check layer naming patterns in configs/models.yaml
   - Verify model type matches configuration

3. **Memory Issues**
   - Reduce batch size
   - Use float16 instead of float32
   - Process fewer layers at once
   - Enable memory monitoring

4. **Analysis Errors**
   - Verify HDF5 file integrity
   - Check input file format
   - Ensure sufficient disk space for outputs

### Getting Help

- Use `--log-level DEBUG` for detailed logs
- Check `neuronmap.log` for error details
- Use `python main.py errors` to see recent issues
- Monitor system resources with `python main.py monitor`

## Advanced Features

### Custom Model Integration

Add new models to `configs/models.yaml`:

```yaml
my_custom_model:
  name: "path/to/model"
  type: "gpt"  # or "bert", "t5", etc.
  layers:
    attention: "custom.layer.{layer}.attention"
    mlp: "custom.layer.{layer}.mlp"
    total_layers: 24
```

### Analysis Customization

Extend analysis classes for custom metrics:

```python
from src.analysis.advanced_analysis import ActivationAnalyzer

class CustomAnalyzer(ActivationAnalyzer):
    def custom_metric(self, activation_matrix):
        # Implement custom analysis
        return results
```

### Integration with Existing Workflows

Use as Python library:

```python
from src.analysis.multi_model_extractor import MultiModelActivationExtractor
from src.analysis.advanced_analysis import ActivationAnalyzer

# Extract activations
extractor = MultiModelActivationExtractor()
extractor.load_model("gpt2_small")
results = extractor.run_multi_layer_extraction(
    questions_file="questions.jsonl",
    layer_range=(0, 6)
)

# Analyze results
analyzer = ActivationAnalyzer()
data = analyzer.load_activations_hdf5("activations.h5")
report = analyzer.generate_analysis_report(data)
```

## Next Steps

1. **Experiment with Different Models**: Test various model families to understand their activation patterns
2. **Layer Comparison**: Compare how information flows through different layers
3. **Cross-Model Analysis**: Analyze how different models represent similar concepts
4. **Custom Analysis**: Develop domain-specific analysis methods
5. **Visualization**: Create interactive visualizations of activation patterns

This enhanced NeuronMap provides a powerful foundation for understanding neural network internals across a wide range of models and architectures.
