# Advanced Conceptual Analysis Tutorial

This tutorial demonstrates how to use NeuronMap's cutting-edge conceptual analysis capabilities for deep neural network interpretability.

## Overview

The conceptual analysis module implements state-of-the-art techniques for understanding neural networks at a conceptual and mechanistic level:

- **Concept Extraction**: Discover meaningful concepts in neural representations
- **Circuit Discovery**: Find functional circuits for specific tasks  
- **Causal Tracing**: Understand causal relationships in model computations
- **World Model Analysis**: Analyze how models represent world knowledge
- **Concept Algebra**: Perform mathematical operations on neural concepts
- **Cross-Model Analysis**: Compare representations across different models

## Prerequisites

Ensure you have the required dependencies:
```bash
pip install torch transformers scikit-learn networkx scipy numpy
```

## Quick Start

### 1. Basic Concept Extraction

Extract conceptual representations from a model:

```bash
# Create sample data
cat > concept_data.json << 'EOF'
{
  "texts": [
    "The red car is fast",
    "The blue car is slow", 
    "Fast vehicles are typically red",
    "Slow vehicles are often blue"
  ],
  "labels": ["fast_red", "slow_blue", "fast_red", "slow_blue"]
}
EOF

# Extract concepts using PCA
python main.py conceptual \
  --analysis-type concepts \
  --model bert-base-uncased \
  --input-file concept_data.json \
  --concept-method pca \
  --concept-threshold 0.7 \
  --output results/concepts \
  --target-layers transformer.encoder.layer.6 transformer.encoder.layer.11
```

### 2. Circuit Discovery

Discover functional circuits for specific tasks:

```bash
# Create task data
cat > circuit_data.json << 'EOF'
{
  "texts": [
    "This movie is amazing!",
    "This movie is terrible.",
    "I love this film",
    "I hate this film",
    "Great acting and plot",
    "Poor acting and plot"
  ],
  "labels": ["positive", "negative", "positive", "negative", "positive", "negative"]
}
EOF

# Discover sentiment classification circuits
python main.py conceptual \
  --analysis-type circuits \
  --model bert-base-uncased \
  --input-file circuit_data.json \
  --task-name sentiment_classification \
  --circuit-threshold 0.5 \
  --output results/circuits \
  --target-layers transformer.encoder.layer.3 transformer.encoder.layer.6 transformer.encoder.layer.9
```

### 3. Causal Tracing

Trace causal effects of interventions:

```bash
# Perform causal intervention analysis
python main.py conceptual \
  --analysis-type causal \
  --model bert-base-uncased \
  --input-file circuit_data.json \
  --intervention-layer transformer.encoder.layer.6 \
  --intervention-neurons 100,150,200,250 \
  --intervention-value 0.0 \
  --causal-threshold 0.6 \
  --output results/causal
```

## Advanced Usage

### Concept Algebra

Perform mathematical operations on extracted concepts:

```bash
# First extract concepts
python main.py conceptual \
  --analysis-type concepts \
  --model gpt2 \
  --input-file concept_data.json \
  --output results/base_concepts

# Then perform concept algebra
python main.py conceptual \
  --analysis-type algebra \
  --model gpt2 \
  --input-file concept_data.json \
  --operation add \
  --output results/concept_algebra
```

### World Model Analysis

Analyze how models represent world knowledge:

```bash
# Create spatial reasoning data
cat > world_model_data.json << 'EOF'
{
  "texts": [
    "The ball is on the table",
    "The book is under the chair",
    "The cat sits beside the window",
    "The car drives on the road",
    "The bird flies above the trees"
  ],
  "metadata": [
    {"object": "ball", "position": [1, 2], "relation": "on"},
    {"object": "book", "position": [3, 4], "relation": "under"},
    {"object": "cat", "position": [5, 6], "relation": "beside"},
    {"object": "car", "position": [7, 8], "relation": "on"},
    {"object": "bird", "position": [9, 10], "relation": "above"}
  ]
}
EOF

# Analyze world model representations
python main.py conceptual \
  --analysis-type world_model \
  --model roberta-base \
  --input-file world_model_data.json \
  --output results/world_model \
  --target-layers roberta.encoder.layer.8 roberta.encoder.layer.11
```

## Understanding Results

### Concept Extraction Results

The concept extraction produces:
- **Concept vectors**: High-dimensional representations of discovered concepts
- **Confidence scores**: Reliability measures for each concept
- **Layer analysis**: How concepts emerge across different layers

```json
{
  "concepts": {
    "layer6_fast_red": {
      "name": "layer6_fast_red",
      "confidence": 0.85,
      "layer": "transformer.encoder.layer.6",
      "metadata": {
        "method": "pca",
        "n_samples": 10,
        "explained_variance": 0.76
      }
    }
  }
}
```

### Circuit Discovery Results

Circuit analysis provides:
- **Circuit components**: Neurons and layers involved in the circuit
- **Connections**: Strength and direction of connections
- **Evidence strength**: Quantitative measure of circuit reliability

```json
{
  "circuits": {
    "sentiment_circuit_0": {
      "name": "sentiment_circuit_0",
      "components": ["layer3_att_0", "layer6_ffn_1", "layer9_att_2"],
      "connections": [
        ["layer3_att_0", "layer6_ffn_1", 0.78],
        ["layer6_ffn_1", "layer9_att_2", 0.65]
      ],
      "function": "Task processing for sentiment_classification",
      "evidence_strength": 0.71
    }
  }
}
```

### Causal Tracing Results

Causal analysis reveals:
- **Layer effects**: How interventions propagate through layers
- **Output effects**: Impact on final model predictions
- **Causal pathways**: Which components are most critical

```json
{
  "causal_effects": {
    "intervention_layer": "transformer.encoder.layer.6",
    "layer_effects": {
      "transformer.encoder.layer.7": 2.45,
      "transformer.encoder.layer.8": 1.87,
      "transformer.encoder.layer.9": 1.23
    },
    "output_effect": 3.21
  }
}
```

## Best Practices

### 1. Data Preparation
- Use diverse, representative samples
- Ensure balanced datasets for bias-free analysis
- Include meaningful labels and metadata

### 2. Model Selection
- Start with well-studied models (BERT, GPT-2)
- Consider model size vs. computational resources
- Use models appropriate for your task domain

### 3. Layer Selection
- Analyze multiple layers to understand concept emergence
- Focus on middle to late layers for high-level concepts
- Include attention layers for transformer models

### 4. Threshold Tuning
- Start with default thresholds and adjust based on results
- Lower thresholds for exploratory analysis
- Higher thresholds for production-ready insights

### 5. Interpretation
- Combine multiple analysis types for comprehensive understanding
- Validate findings with domain knowledge
- Use statistical testing for robust conclusions

## Advanced Configurations

### Custom Analysis Pipeline

Create a configuration file for complex analyses:

```yaml
# conceptual_config.yaml
analysis:
  concept_extraction:
    method: pca
    threshold: 0.75
    components: 50
  
  circuit_discovery:
    threshold: 0.6
    min_circuit_size: 3
    max_circuits: 10
  
  causal_tracing:
    intervention_strength: 0.0
    propagation_steps: 5
    
  world_model:
    spatial_analysis: true
    temporal_analysis: true
    object_analysis: true
```

### Batch Processing

Process multiple models or datasets:

```bash
# Process multiple models
for model in bert-base-uncased roberta-base distilbert-base-uncased; do
  python main.py conceptual \
    --analysis-type concepts \
    --model $model \
    --input-file concept_data.json \
    --output results/${model}_concepts
done
```

## Integration with Other Tools

### TensorBoard Integration

Visualize concept evolution:
```python
from src.integrations.tensorboard import TensorBoardIntegration

tb_integration = TensorBoardIntegration()
tb_integration.log_conceptual_analysis(
    concepts=analyzer.concepts,
    circuits=analyzer.circuits,
    log_dir="tensorboard_logs"
)
```

### Weights & Biases Integration

Track experiments:
```python
from src.integrations.wandb_integration import WandBIntegration

wandb_integration = WandBIntegration()
wandb_integration.log_conceptual_metrics(
    concepts=concepts,
    circuits=circuits,
    experiment_name="conceptual_analysis"
)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or number of target layers
2. **Low Concept Confidence**: Adjust threshold or use more diverse data
3. **No Circuits Found**: Lower circuit threshold or increase sample size
4. **Intervention Effects Too Small**: Try different intervention values or layers

### Performance Optimization

- Use GPU acceleration when available
- Process data in batches for large datasets
- Cache intermediate results for repeated analyses
- Use efficient data formats (HDF5 for large activations)

## Further Reading

- [Concept Activation Vectors (CAVs)](https://arxiv.org/abs/1711.11279)
- [Network Dissection](https://arxiv.org/abs/1704.05796)
- [Causal Tracing](https://arxiv.org/abs/2202.05262)
- [Mechanistic Interpretability](https://arxiv.org/abs/2211.00593)

## Next Steps

1. Explore the API documentation for programmatic access
2. Try the interactive Jupyter notebooks in `examples/`
3. Integrate with your existing ML workflows
4. Contribute new analysis methods to the project

For support and questions, visit our [GitHub Discussions](https://github.com/Emilio942/NeuronMap/discussions) or check the [troubleshooting guide](../docs/troubleshooting.md).
