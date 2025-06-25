# Sentiment Analysis with BERT

This example demonstrates how to use NeuronMap to analyze BERT's internal representations for sentiment classification.

## Overview

We'll explore:
- How BERT processes sentiment information across layers
- Which neurons are most important for sentiment detection
- How attention patterns differ between positive and negative texts
- Layer-wise sentiment information flow

## Files

- `analyze_sentiment.py` - Main analysis script
- `config.yaml` - Configuration settings
- `data/` - Sample sentiment data
- `results/` - Analysis outputs
- `notebook.ipynb` - Interactive Jupyter notebook

## Quick Start

```bash
# Install dependencies
pip install neuronmap torch transformers

# Run basic analysis
python analyze_sentiment.py

# Run with custom data
python analyze_sentiment.py --data-path /path/to/your/sentiment_data.csv

# Interactive analysis
jupyter notebook notebook.ipynb
```

## Data Format

Your sentiment data should be in CSV format:
```csv
text,label
"I love this movie!",positive
"This film is terrible.",negative
"It's an okay movie.",neutral
```

## Analysis Results

The analysis will generate:
1. **Attention visualizations** - How the model attends to sentiment-bearing words
2. **Layer-wise activations** - Sentiment information across BERT layers
3. **Neuron importance scores** - Which neurons are most important for sentiment
4. **Comparative analysis** - Differences between positive/negative processing

## Key Insights

### Layer Analysis
- **Early layers (0-3)**: Focus on syntax and basic word features
- **Middle layers (4-8)**: Develop sentiment-aware representations
- **Late layers (9-11)**: Task-specific sentiment classification

### Attention Patterns
- Positive texts: Attention to positive adjectives and emotional words
- Negative texts: Focus on negation and negative descriptors
- Neutral texts: More distributed attention patterns

### Important Neurons
- Layer 6-8 neurons show highest sentiment specificity
- Certain neurons activate strongly for emotional words
- Some neurons specialize in negation detection

## Usage Examples

### Basic Analysis
```python
from neuronmap import NeuronMap
import pandas as pd

# Load data
data = pd.read_csv('data/sentiment_samples.csv')

# Initialize NeuronMap
nm = NeuronMap()

# Load BERT model
model = nm.load_model('bert-base-uncased')

# Analyze sentiment processing
results = nm.analyze_sentiment(
    model=model,
    texts=data['text'].tolist(),
    labels=data['label'].tolist(),
    layers=[0, 3, 6, 9, 11]
)

# Visualize results
nm.plot_sentiment_analysis(results)
```

### Advanced Analysis
```python
# Compare different sentiment categories
positive_texts = data[data['label'] == 'positive']['text'].tolist()
negative_texts = data[data['label'] == 'negative']['text'].tolist()

# Analyze differences
comparison = nm.compare_sentiment_processing(
    positive_texts=positive_texts,
    negative_texts=negative_texts,
    model=model
)

# Find sentiment-specific neurons
sentiment_neurons = nm.find_sentiment_neurons(
    model=model,
    data=data,
    threshold=0.8
)
```

## Configuration

Edit `config.yaml` to customize the analysis:

```yaml
model:
  name: "bert-base-uncased"
  max_length: 128
  batch_size: 16

analysis:
  layers: [0, 3, 6, 9, 11]
  attention_heads: "all"
  extract_hidden_states: true
  extract_attention: true

sentiment:
  categories: ["positive", "negative", "neutral"]
  threshold: 0.5
  method: "classification"

visualization:
  plot_attention: true
  plot_activations: true
  plot_neurons: true
  save_format: "png"
  dpi: 300
```

## Expected Outputs

### Visualizations
1. `attention_heatmaps.png` - Token attention patterns
2. `layer_activations.png` - Activation strength across layers
3. `sentiment_neurons.png` - Important neurons for sentiment
4. `comparison_plot.png` - Positive vs negative processing

### Data Files
1. `activations.json` - Raw activation data
2. `attention_weights.json` - Attention weight matrices
3. `neuron_scores.csv` - Neuron importance rankings
4. `statistics.json` - Summary statistics

### Reports
1. `analysis_report.html` - Comprehensive analysis report
2. `insights.md` - Key findings and interpretations

## Research Applications

### Academic Studies
- Model interpretability research
- Sentiment processing mechanisms
- Cross-linguistic sentiment analysis
- Bias detection in sentiment models

### Industry Applications
- Product review analysis
- Social media monitoring
- Customer feedback processing
- Brand sentiment tracking

## Extending the Analysis

### Custom Models
```python
# Use a fine-tuned sentiment model
model = nm.load_model('path/to/fine-tuned-bert')

# Or compare multiple models
models = {
    'bert': nm.load_model('bert-base-uncased'),
    'roberta': nm.load_model('roberta-base'),
    'distilbert': nm.load_model('distilbert-base-uncased')
}

comparison = nm.compare_models(models, data)
```

### Custom Metrics
```python
# Define custom sentiment metrics
def sentiment_specificity(activations, labels):
    # Custom metric implementation
    pass

# Add to analysis
nm.add_custom_metric('sentiment_specificity', sentiment_specificity)
results = nm.analyze_sentiment(model, texts, labels)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in config
   - Use shorter text sequences
   - Analyze fewer layers

2. **Slow Processing**
   - Enable GPU acceleration
   - Reduce number of samples
   - Use distributed processing

3. **Poor Visualizations**
   - Check data quality
   - Adjust visualization parameters
   - Filter outliers

### Performance Tips

- Use `batch_size=32` for optimal GPU utilization
- Limit `max_length=128` for faster processing
- Select specific layers instead of analyzing all
- Use caching for repeated analyses

## Further Reading

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Attention Visualization](https://arxiv.org/abs/1906.05714)
- [Sentiment Analysis Survey](https://arxiv.org/abs/2005.11401)
- [Model Interpretability](https://arxiv.org/abs/1702.04896)

## Citation

If you use this example in your research, please cite:

```bibtex
@misc{neuronmap_sentiment_example,
  title={Sentiment Analysis with NeuronMap: Understanding BERT's Internal Representations},
  author={NeuronMap Team},
  year={2024},
  url={https://github.com/Emilio942/NeuronMap/examples/nlp/sentiment_analysis}
}
```

---

*Happy analyzing! 🎭📊*
