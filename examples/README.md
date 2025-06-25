# NeuronMap Examples & Use Cases

This directory contains real-world examples and use cases demonstrating NeuronMap's capabilities across different domains and applications.

## 📁 Directory Structure

```
examples/
├── nlp/                    # Natural Language Processing
│   ├── sentiment_analysis/
│   ├── question_answering/
│   ├── text_classification/
│   └── language_modeling/
├── vision/                 # Computer Vision
│   ├── image_classification/
│   ├── object_detection/
│   └── semantic_segmentation/
├── multimodal/            # Multi-modal Models
│   ├── vision_language/
│   ├── audio_text/
│   └── cross_modal/
├── research/              # Research Applications
│   ├── interpretability_studies/
│   ├── model_comparison/
│   ├── bias_analysis/
│   └── mechanistic_interpretability/
├── domain_specific/       # Domain-Specific Analysis
│   ├── code_analysis/
│   ├── mathematical_reasoning/
│   ├── scientific_text/
│   └── legal_documents/
├── notebooks/             # Interactive Jupyter Notebooks
│   ├── beginner/
│   ├── intermediate/
│   └── advanced/
└── datasets/              # Sample datasets
    ├── text/
    ├── images/
    └── multimodal/
```

## 🚀 Quick Start Examples

### 1. Sentiment Analysis with BERT
```bash
cd examples/nlp/sentiment_analysis/
python analyze_sentiment.py
```

### 2. Vision Model Interpretability
```bash
cd examples/vision/image_classification/
python analyze_resnet.py
```

### 3. Multi-modal Analysis
```bash
cd examples/multimodal/vision_language/
python analyze_clip.py
```

## 📊 Use Case Categories

### Natural Language Processing

#### 🎯 **Sentiment Analysis**
- **Models**: BERT, RoBERTa, DistilBERT
- **Applications**: Social media monitoring, product reviews
- **Techniques**: Attention visualization, layer-wise analysis
- **[View Example →](nlp/sentiment_analysis/)**

#### ❓ **Question Answering**
- **Models**: BERT-QA, RoBERTa-QA, T5
- **Applications**: Chatbots, information retrieval
- **Techniques**: Attention flow, answer extraction analysis
- **[View Example →](nlp/question_answering/)**

#### 📝 **Text Classification**
- **Models**: BERT, XLNet, ELECTRA
- **Applications**: Document categorization, spam detection
- **Techniques**: Feature attribution, class-specific analysis
- **[View Example →](nlp/text_classification/)**

### Computer Vision

#### 🖼️ **Image Classification**
- **Models**: ResNet, VGG, EfficientNet
- **Applications**: Medical imaging, autonomous vehicles
- **Techniques**: Activation maps, feature visualization
- **[View Example →](vision/image_classification/)**

#### 🎯 **Object Detection**
- **Models**: YOLO, R-CNN, SSD
- **Applications**: Security systems, autonomous driving
- **Techniques**: Bounding box analysis, feature maps
- **[View Example →](vision/object_detection/)**

### Research Applications

#### 🔬 **Interpretability Studies**
- **Focus**: Understanding model behavior
- **Methods**: Probing, concept analysis, causal intervention
- **Publications**: Research paper examples
- **[View Example →](research/interpretability_studies/)**

#### ⚖️ **Bias Analysis**
- **Focus**: Fairness and ethical AI
- **Methods**: Demographic parity, equalized odds
- **Applications**: Hiring systems, loan approval
- **[View Example →](research/bias_analysis/)**

#### 🧠 **Mechanistic Interpretability**
- **Focus**: Circuit-level understanding
- **Methods**: Activation patching, causal tracing
- **Applications**: Safety research, alignment
- **[View Example →](research/mechanistic_interpretability/)**

## 🎓 Educational Examples

### Beginner Level
- Basic activation extraction
- Simple visualization
- Model comparison
- Layer analysis

### Intermediate Level
- Attention mechanisms
- Feature attribution
- Cross-model analysis
- Custom metrics

### Advanced Level
- Causal interventions
- Concept algebra
- Model editing
- Research workflows

## 🏭 Industry Applications

### Healthcare
- Medical image analysis
- Clinical text processing
- Drug discovery models
- Diagnostic assistance

### Finance
- Risk assessment models
- Fraud detection
- Algorithmic trading
- Credit scoring

### Technology
- Recommendation systems
- Search algorithms
- Content moderation
- Voice assistants

### Legal & Compliance
- Document analysis
- Regulatory compliance
- Contract review
- Legal research

## 📚 Educational Resources

### Datasets Included
- **Text**: News articles, reviews, social media
- **Images**: CIFAR-10 subset, medical scans
- **Multimodal**: Image-caption pairs, video clips
- **Domain-specific**: Code repos, scientific papers

### Pre-trained Models
- Popular architectures ready to analyze
- Fine-tuned domain-specific models
- Comparison baselines
- Custom model templates

## 🔧 Running Examples

### Prerequisites
```bash
pip install neuronmap[examples]
```

### Basic Usage
```bash
# Run individual example
cd examples/nlp/sentiment_analysis/
python run_analysis.py

# Run with custom data
python run_analysis.py --data-path /path/to/your/data

# Batch processing
python run_analysis.py --batch-mode --output-dir results/
```

### Interactive Notebooks
```bash
# Start Jupyter
jupyter notebook examples/notebooks/

# Or use Google Colab
# Click the "Open in Colab" button in any notebook
```

## 📈 Performance Benchmarks

### Computational Requirements
- **Memory**: 4-16GB RAM depending on model size
- **GPU**: Optional but recommended for large models
- **Storage**: 1-10GB for datasets and models
- **Time**: Minutes to hours depending on analysis depth

### Scalability Examples
- Single sample analysis: Seconds
- Batch processing: Minutes to hours
- Full dataset analysis: Hours to days
- Distributed processing: Available for large-scale studies

## 🤝 Contributing Examples

### How to Add New Examples
1. Choose appropriate category
2. Follow template structure
3. Include documentation
4. Add tests and validation
5. Submit pull request

### Example Template
```
example_name/
├── README.md              # Description and usage
├── run_analysis.py        # Main script
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── data/                 # Sample data
├── outputs/              # Expected outputs
└── notebook.ipynb       # Interactive version
```

### Quality Guidelines
- Clear documentation
- Reproducible results
- Educational value
- Real-world relevance
- Performance considerations

## 📊 Results Gallery

### Visualization Examples
- [Attention Heatmaps](gallery/attention_heatmaps.md)
- [Activation Flows](gallery/activation_flows.md)
- [Feature Maps](gallery/feature_maps.md)
- [Comparison Plots](gallery/comparisons.md)

### Research Findings
- [Model Behavior Studies](gallery/research_findings.md)
- [Bias Detection Results](gallery/bias_analysis.md)
- [Performance Comparisons](gallery/benchmarks.md)
- [Novel Discoveries](gallery/discoveries.md)

## 📞 Support

### Getting Help
- **Documentation**: [Main docs](../docs/)
- **Tutorials**: [Step-by-step guides](../tutorials/)
- **Community**: [Discord chat](https://discord.gg/neuronmap)
- **Issues**: [GitHub issues](https://github.com/Emilio942/NeuronMap/issues)

### Reporting Problems
- Example not working? File an issue
- Found a bug? Report with minimal reproduction
- Need a new example? Request in discussions
- Want to contribute? Check contributing guidelines

---

*Explore, learn, and discover insights with NeuronMap! 🔍✨*
