"""
Circuit Analysis REST API endpoints.

This module provides REST API endpoints for circuit discovery and analysis,
including induction heads, composition analysis, and neuron-head influence.
"""

from flask import Blueprint, request, jsonify, current_app
import logging
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import traceback
import torch
import transformers

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from analysis.circuits import (
    NeuralCircuit,
    AttentionHeadCompositionAnalyzer,
    InductionHeadScanner,
    CopyingHeadScanner,
    NeuronToHeadAnalyzer,
    CircuitVerifier
)

logger = logging.getLogger(__name__)

# Create Blueprint
circuits_bp = Blueprint('circuits', __name__, url_prefix='/api/circuits')

# Global model cache
_model_cache = {}


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load and cache model and tokenizer."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = transformers.AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            output_attentions=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.to(device)
        model.eval()
        
        _model_cache[model_name] = (model, tokenizer)
        logger.info(f"Loaded model {model_name} on {device}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise


@circuits_bp.route('/models', methods=['GET'])
def list_available_models():
    """List available models for circuit analysis."""
    try:
        available_models = [
            "gpt2",
            "gpt2-medium", 
            "gpt2-large",
            "gpt2-xl",
            "distilgpt2"
        ]
        
        return jsonify({
            'success': True,
            'models': available_models,
            'cached_models': list(_model_cache.keys())
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@circuits_bp.route('/induction-heads', methods=['POST'])
def find_induction_heads():
    """Find induction heads in a model."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'gpt2')
        threshold = data.get('threshold', 0.1)
        device = data.get('device', 'auto')
        
        logger.info(f"Finding induction heads in {model_name} with threshold {threshold}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # Create scanner
        scanner = InductionHeadScanner(model, tokenizer)
        
        # Create test prompt
        test_prompt = scanner.create_induction_test_prompt()
        
        # Run scan
        candidates = scanner.scan_for_induction_heads(test_prompt, threshold=threshold)
        
        # Create circuit
        circuit = scanner.create_induction_circuit(threshold=threshold, input_ids=test_prompt)
        
        results = {
            'success': True,
            'model': model_name,
            'threshold': threshold,
            'candidates': [(layer, head, float(score)) for layer, head, score in candidates],
            'circuit_stats': circuit.get_circuit_statistics(),
            'num_found': len(candidates)
        }
        
        logger.info(f"Found {len(candidates)} induction heads")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error finding induction heads: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@circuits_bp.route('/copying-heads', methods=['POST'])
def find_copying_heads():
    """Find copying/saliency heads in a model."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'gpt2')
        threshold = data.get('threshold', 0.3)
        concentration_threshold = data.get('concentration_threshold', 2.0)
        device = data.get('device', 'auto')
        
        logger.info(f"Finding copying heads in {model_name}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # Create scanner
        scanner = CopyingHeadScanner(model, tokenizer)
        
        # Create test prompt
        test_text = "The quick brown fox jumps over the lazy dog"
        test_prompt = tokenizer(test_text, return_tensors="pt")["input_ids"].to(model.device)
        
        # Run scan
        candidates = scanner.scan_for_copying_heads(
            test_prompt, 
            copying_threshold=threshold,
            concentration_threshold=concentration_threshold
        )
        
        results = {
            'success': True,
            'model': model_name,
            'threshold': threshold,
            'concentration_threshold': concentration_threshold,
            'candidates': [
                {
                    'layer': layer,
                    'head': head,
                    'scores': scores
                } for layer, head, scores in candidates
            ],
            'num_found': len(candidates)
        }
        
        logger.info(f"Found {len(candidates)} copying heads")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error finding copying heads: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@circuits_bp.route('/composition', methods=['POST'])
def analyze_composition():
    """Analyze attention head composition between layers."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'gpt2')
        layer1 = data.get('layer1', 0)
        layer2 = data.get('layer2', 1)
        threshold = data.get('threshold', 0.3)
        device = data.get('device', 'auto')
        
        logger.info(f"Analyzing composition in {model_name}: L{layer1} -> L{layer2}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # Create analyzer
        analyzer = AttentionHeadCompositionAnalyzer(model, tokenizer)
        
        # Create test prompt
        test_text = "The quick brown fox jumps over the lazy dog"
        test_prompt = tokenizer(test_text, return_tensors="pt")["input_ids"].to(model.device)
        
        # Run composition analysis
        compositions = analyzer.analyze_layer_compositions(test_prompt, layer1, layer2, threshold)
        
        # Build circuit
        circuit = analyzer.build_composition_circuit(test_prompt, threshold)
        
        results = {
            'success': True,
            'model': model_name,
            'layer1': layer1,
            'layer2': layer2,
            'threshold': threshold,
            'compositions': [
                {
                    'head1': head1,
                    'head2': head2,
                    'score': float(score)
                } for head1, head2, score in compositions
            ],
            'circuit_stats': circuit.get_circuit_statistics(),
            'num_found': len(compositions)
        }
        
        logger.info(f"Found {len(compositions)} significant compositions")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error analyzing composition: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@circuits_bp.route('/neuron-head', methods=['POST'])
def analyze_neuron_head():
    """Analyze neuron-to-head influence connections."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'gpt2')
        max_layers = data.get('max_layers', 3)
        threshold = data.get('threshold', 0.1)
        device = data.get('device', 'auto')
        
        logger.info(f"Analyzing neuron-head influence in {model_name}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # Create analyzer
        analyzer = NeuronToHeadAnalyzer(model, tokenizer)
        
        # Create test prompt
        test_text = "The quick brown fox jumps over the lazy dog"
        test_prompt = tokenizer(test_text, return_tensors="pt")["input_ids"].to(model.device)
        
        # Build circuit
        circuit = analyzer.build_neuron_head_circuit(test_prompt, max_layers, threshold)
        
        results = {
            'success': True,
            'model': model_name,
            'max_layers': max_layers,
            'threshold': threshold,
            'circuit_stats': circuit.get_circuit_statistics(),
            'num_components': len(circuit._components),
            'num_connections': len(circuit._connections)
        }
        
        logger.info(f"Found {len(circuit._components)} components in neuron-head circuit")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error analyzing neuron-head influence: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@circuits_bp.route('/verify', methods=['POST'])
def verify_circuit():
    """Verify a circuit using causal interventions."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'gpt2')
        circuit_data = data.get('circuit')
        prompt = data.get('prompt', 'The quick brown fox')
        target_position = data.get('target_position', -1)
        device = data.get('device', 'auto')
        
        if not circuit_data:
            return jsonify({
                'success': False,
                'error': 'Circuit data is required'
            }), 400
        
        logger.info(f"Verifying circuit in {model_name}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # Create circuit from data
        circuit = NeuralCircuit.from_dict(circuit_data)
        
        # Create verifier
        verifier = CircuitVerifier(model, tokenizer)
        
        # Run verification
        verification_results = verifier.verify_circuit(circuit, prompt, target_position)
        
        results = {
            'success': True,
            'model': model_name,
            'prompt': prompt,
            'verification_score': verification_results['verification_score'],
            'max_component_effect': verification_results['max_component_effect'],
            'components_tested': verification_results['total_components_tested'],
            'component_effects': verification_results['component_effects']
        }
        
        logger.info(f"Circuit verification score: {verification_results['verification_score']:.4f}")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error verifying circuit: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@circuits_bp.route('/circuit-data/<circuit_type>', methods=['GET'])
def get_circuit_data(circuit_type: str):
    """Get circuit data for visualization."""
    try:
        # This would normally load from a database or cache
        # For now, return mock data structure
        
        if circuit_type == 'induction':
            # Mock induction heads circuit
            circuit_data = {
                'nodes': [
                    {
                        'id': 'layer_0_head_5',
                        'label': 'Attn Head L0H5',
                        'type': 'attention_head',
                        'layer': 0,
                        'position': 5,
                        'score': 0.85
                    },
                    {
                        'id': 'layer_2_head_3',
                        'label': 'Attn Head L2H3',
                        'type': 'attention_head',
                        'layer': 2,
                        'position': 3,
                        'score': 0.73
                    }
                ],
                'edges': [
                    {
                        'source': 'layer_0_head_5',
                        'target': 'layer_2_head_3',
                        'weight': 0.65,
                        'type': 'induction_sequence'
                    }
                ],
                'metadata': {
                    'circuit_type': 'induction',
                    'model': 'gpt2',
                    'threshold': 0.1
                }
            }
        elif circuit_type == 'composition':
            # Mock composition circuit
            circuit_data = {
                'nodes': [
                    {
                        'id': 'layer_0_head_1',
                        'label': 'Attn Head L0H1',
                        'type': 'attention_head',
                        'layer': 0,
                        'position': 1,
                        'score': 0.92
                    },
                    {
                        'id': 'layer_1_head_7',
                        'label': 'Attn Head L1H7',
                        'type': 'attention_head',
                        'layer': 1,
                        'position': 7,
                        'score': 0.78
                    }
                ],
                'edges': [
                    {
                        'source': 'layer_0_head_1',
                        'target': 'layer_1_head_7',
                        'weight': 0.89,
                        'type': 'composition'
                    }
                ],
                'metadata': {
                    'circuit_type': 'composition',
                    'model': 'gpt2',
                    'threshold': 0.3
                }
            }
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown circuit type: {circuit_type}'
            }), 400
        
        return jsonify({
            'success': True,
            'circuit_type': circuit_type,
            'data': circuit_data
        })
        
    except Exception as e:
        logger.error(f"Error getting circuit data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@circuits_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'service': 'circuits',
        'cached_models': list(_model_cache.keys()),
        'gpu_available': torch.cuda.is_available()
    })


# Error handlers
@circuits_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@circuits_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
