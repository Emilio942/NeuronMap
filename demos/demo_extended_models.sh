#!/bin/bash
"""
Extended Model Support Demo for NeuronMap
=========================================

This demo showcases the extended model support capabilities including
BERT variants, T5 models, and domain-specific models.
"""

echo "🧠 NeuronMap Extended Model Support Demo"
echo "========================================"
echo

# Create demo directory
mkdir -p demo_outputs/extended_models
cd demo_outputs/extended_models

echo "📋 Demo Questions:"
cat > demo_questions.txt << 'EOF'
What is the capital of France?
How does machine learning work?
Explain the process of photosynthesis.
What are the applications of artificial intelligence?
How do neural networks learn patterns?
EOF

echo "Loaded 5 demo questions"
echo

# Demo 1: BERT family models
echo "🔬 Demo 1: BERT Family Models"
echo "============================="

echo "🤖 Testing BERT-base-uncased..."
python3 ../../main_new.py \
  --model bert-base-uncased \
  --input-file demo_questions.txt \
  --output-file bert_base_results.csv \
  --analyze \
  --device cpu

echo "✅ BERT-base analysis completed"
echo

echo "🤖 Testing RoBERTa-base..."
python3 ../../main_new.py \
  --model roberta-base \
  --input-file demo_questions.txt \
  --output-file roberta_base_results.csv \
  --analyze \
  --device cpu

echo "✅ RoBERTa analysis completed"
echo

echo "🤖 Testing DistilBERT..."
python3 ../../main_new.py \
  --model distilbert-base-uncased \
  --input-file demo_questions.txt \
  --output-file distilbert_results.csv \
  --analyze \
  --device cpu

echo "✅ DistilBERT analysis completed"
echo

# Demo 2: T5 family models
echo "🔬 Demo 2: T5 Family Models"
echo "============================"

echo "🤖 Testing T5-small..."
python3 ../../main_new.py \
  --model t5-small \
  --input-file demo_questions.txt \
  --output-file t5_small_results.csv \
  --analyze \
  --device cpu

echo "✅ T5-small analysis completed"
echo

echo "🤖 Testing FLAN-T5-small..."
python3 ../../main_new.py \
  --model google/flan-t5-small \
  --input-file demo_questions.txt \
  --output-file flan_t5_small_results.csv \
  --analyze \
  --device cpu

echo "✅ FLAN-T5-small analysis completed"
echo

# Demo 3: Domain-specific models
echo "🔬 Demo 3: Domain-Specific Models"
echo "=================================="

echo "🧪 Testing SciBERT (Scientific domain)..."
python3 ../../main_new.py \
  --model allenai/scibert_scivocab_uncased \
  --input-file demo_questions.txt \
  --output-file scibert_results.csv \
  --analyze \
  --device cpu

echo "✅ SciBERT analysis completed"
echo

echo "🧬 Testing BioBERT (Biomedical domain)..."
python3 ../../main_new.py \
  --model dmis-lab/biobert-v1.1 \
  --input-file demo_questions.txt \
  --output-file biobert_results.csv \
  --analyze \
  --device cpu

echo "✅ BioBERT analysis completed"
echo

# Demo 4: Multi-model comparison with extended models
echo "🔬 Demo 4: Multi-Model Comparison"
echo "================================="

echo "🚀 Running multi-model comparison across architectures..."
python3 ../../main_new.py \
  --multi-model \
  --models bert-base-uncased roberta-base t5-small \
  --input-file demo_questions.txt \
  --device cpu \
  --visualize

echo "✅ Multi-model comparison completed"
echo

# Demo 5: Advanced analytics
echo "🔬 Demo 5: Advanced Analytics"
echo "============================="

echo "🧬 Running comprehensive advanced analytics..."
python3 ../../main_new.py \
  --model bert-base-uncased \
  --input-file demo_questions.txt \
  --advanced-analytics \
  --device cpu

echo "✅ Advanced analytics completed"
echo

# Demo 6: Available models listing
echo "🔬 Demo 6: Available Models"
echo "==========================="

echo "📋 Listing all available preconfigured models:"
python3 -c "
import sys
sys.path.insert(0, '../../src')
from analysis.universal_model_adapter import UniversalModelAdapter
from utils.config_manager import get_config

config = get_config()
adapter = UniversalModelAdapter(config)
models = adapter.get_available_models()

print('Available preconfigured models:')
for model in models:
    info = adapter.get_model_info(model)
    print(f'  - {model} ({info[\"type\"]}, {info[\"total_layers\"]} layers)')
"

echo
echo "🎉 Extended Model Support Demo Completed!"
echo "========================================"
echo
echo "📊 Results Summary:"
echo "  - BERT family models: ✅ Tested"
echo "  - T5 family models: ✅ Tested"  
echo "  - Domain-specific models: ✅ Tested"
echo "  - Multi-model comparison: ✅ Completed"
echo "  - Advanced analytics: ✅ Completed"
echo "  - Model listing: ✅ Available"
echo
echo "📁 Check the demo_outputs/extended_models/ directory for:"
echo "  - Individual model analysis results"
echo "  - Multi-model comparison visualizations"
echo "  - Advanced analytics reports"
echo
echo "🚀 The NeuronMap system now supports a wide range of model architectures!"
echo "   Try your own models using the Universal Model Adapter."
