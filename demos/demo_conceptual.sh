#!/bin/bash

# Demo script for advanced conceptual analysis features (Section 14)
# This script demonstrates cutting-edge neural network interpretability techniques

echo "=== NeuronMap Advanced Conceptual Analysis Demo ==="
echo "Demonstrating Section 14: Advanced conceptual analyses"
echo

# Create demo data directory
mkdir -p demo_data/conceptual

# Create sample input data for conceptual analysis
cat > demo_data/conceptual/concepts_input.json << 'EOF'
{
  "texts": [
    "The red car is fast",
    "The blue car is slow", 
    "The fast train is red",
    "The slow train is blue",
    "Red objects move quickly",
    "Blue objects move slowly",
    "Fast vehicles are red",
    "Slow vehicles are blue",
    "The quick red bicycle",
    "The slow blue bicycle",
    "Red animals run fast",
    "Blue animals walk slowly",
    "Fast red machines",
    "Slow blue machines",
    "The red rocket speeds up",
    "The blue boat slows down"
  ],
  "labels": [
    "fast_red", "slow_blue", "fast_red", "slow_blue", 
    "fast_red", "slow_blue", "fast_red", "slow_blue",
    "fast_red", "slow_blue", "fast_red", "slow_blue",
    "fast_red", "slow_blue", "fast_red", "slow_blue"
  ]
}
EOF

# Create causal intervention demo data
cat > demo_data/conceptual/causal_input.json << 'EOF'
{
  "texts": [
    "The cat is happy",
    "The dog is sad", 
    "Happy cats purr loudly",
    "Sad dogs whimper quietly",
    "Emotions affect behavior"
  ],
  "labels": ["positive", "negative", "positive", "negative", "neutral"]
}
EOF

# Create world model demo data
cat > demo_data/conceptual/world_model_input.json << 'EOF'
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

echo "1. Concept Extraction Analysis"
echo "   Extracting conceptual representations from neural network activations..."
python main.py conceptual \
  --analysis-type concepts \
  --model bert-base-uncased \
  --input-file demo_data/conceptual/concepts_input.json \
  --concept-method pca \
  --concept-threshold 0.6 \
  --output demo_outputs/conceptual_concepts \
  --target-layers transformer.encoder.layer.6 transformer.encoder.layer.11

echo
echo "2. Circuit Discovery Analysis"
echo "   Discovering functional circuits for specific tasks..."
python main.py conceptual \
  --analysis-type circuits \
  --model bert-base-uncased \
  --input-file demo_data/conceptual/concepts_input.json \
  --task-name color_speed_classification \
  --circuit-threshold 0.4 \
  --output demo_outputs/conceptual_circuits \
  --target-layers transformer.encoder.layer.3 transformer.encoder.layer.6 transformer.encoder.layer.9

echo
echo "3. Causal Tracing Analysis"
echo "   Tracing causal effects of interventions in the network..."
python main.py conceptual \
  --analysis-type causal \
  --model bert-base-uncased \
  --input-file demo_data/conceptual/causal_input.json \
  --intervention-layer transformer.encoder.layer.6 \
  --intervention-neurons 100,150,200 \
  --intervention-value 0.0 \
  --causal-threshold 0.5 \
  --output demo_outputs/conceptual_causal

echo
echo "4. World Model Analysis"
echo "   Analyzing how the model represents world knowledge..."
python main.py conceptual \
  --analysis-type world_model \
  --model bert-base-uncased \
  --input-file demo_data/conceptual/world_model_input.json \
  --output demo_outputs/conceptual_world_model \
  --target-layers transformer.encoder.layer.8 transformer.encoder.layer.11

echo
echo "5. Concept Algebra"
echo "   Performing algebraic operations on extracted concepts..."
# First extract concepts, then perform algebra
python main.py conceptual \
  --analysis-type concepts \
  --model bert-base-uncased \
  --input-file demo_data/conceptual/concepts_input.json \
  --concept-method pca \
  --output demo_outputs/conceptual_algebra_base

echo "   Running concept algebra operations..."
python main.py conceptual \
  --analysis-type algebra \
  --model bert-base-uncased \
  --input-file demo_data/conceptual/concepts_input.json \
  --operation add \
  --output demo_outputs/conceptual_algebra

echo
echo "=== Analysis Complete ==="
echo "Check the following directories for results:"
echo "  demo_outputs/conceptual_concepts/"
echo "  demo_outputs/conceptual_circuits/"
echo "  demo_outputs/conceptual_causal/"
echo "  demo_outputs/conceptual_world_model/"
echo "  demo_outputs/conceptual_algebra/"
echo
echo "Key features demonstrated:"
echo "  ✓ Concept extraction using PCA/NMF"
echo "  ✓ Functional circuit discovery"
echo "  ✓ Causal tracing and intervention"
echo "  ✓ World model representation analysis"
echo "  ✓ Concept algebra operations"
echo "  ✓ Advanced neural network interpretability"
echo
echo "These techniques enable deep understanding of:"
echo "  - How concepts are represented in neural networks"
echo "  - Which circuits process specific tasks"
echo "  - Causal relationships in model computations"
echo "  - How models represent world knowledge"
echo "  - Mathematical operations on neural concepts"
