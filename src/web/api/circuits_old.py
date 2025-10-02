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

from ...analysis.circuits import (
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
        # This would be replaced with actual model discovery
        models = [
            {
                'name': 'gpt2',
                'type': 'GPT',
                'layers': 12,
                'attention_heads': 12,
                'hidden_size': 768
            },
            {
                'name': 'gpt2-medium',
                'type': 'GPT',
                'layers': 24,
                'attention_heads': 16,
                'hidden_size': 1024
            },
            {
                'name': 'bert-base-uncased',
                'type': 'BERT',
                'layers': 12,
                'attention_heads': 12,
                'hidden_size': 768
            }
        ]
        
        return jsonify({
            'status': 'success',
            'models': models
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@circuits_bp.route('/load-model', methods=['POST'])
def load_model():
    """Load a model for circuit analysis."""
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({
                'status': 'error',
                'message': 'model_name is required'
            }), 400
        
        model_name = data['model_name']
        
        # Load model
        model_manager = get_model_manager()
        model_manager.load_model(model_name)
        
        # Get model info
        model_info = model_manager.get_model_info()
        
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} loaded successfully',
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@circuits_bp.route('/find-induction-heads', methods=['POST'])
def find_induction_heads():
    """Find induction heads in the loaded model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body is required'
            }), 400
        
        # Extract parameters
        threshold = data.get('threshold', 0.3)
        pattern_length = data.get('pattern_length', 10)
        num_examples = data.get('num_examples', 50)
        layers = data.get('layers')  # Optional layer restriction
        
        # Initialize scanner
        model_manager = get_model_manager()
        config = get_config()
        scanner = InductionHeadScanner(model_manager, config)
        
        # Perform analysis
        if layers:
            layer_list = [int(x) for x in layers] if isinstance(layers, list) else [layers]
            result = scanner.find_induction_heads(
                threshold=threshold,
                pattern_length=pattern_length,
                num_examples=num_examples,
                layers=layer_list
            )
        else:
            result = scanner.find_induction_heads(
                threshold=threshold,
                pattern_length=pattern_length,
                num_examples=num_examples
            )
        
        return jsonify({
            'status': 'success',
            'result': result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error finding induction heads: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if current_app.debug else None
        }), 500


@circuits_bp.route('/find-copying-heads', methods=['POST'])
def find_copying_heads():
    """Find copying heads in the loaded model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body is required'
            }), 400
        
        # Extract parameters
        threshold = data.get('threshold', 0.5)
        num_examples = data.get('num_examples', 100)
        layers = data.get('layers')  # Optional layer restriction
        
        # Initialize scanner
        model_manager = get_model_manager()
        config = get_config()
        scanner = CopyingHeadScanner(model_manager, config)
        
        # Perform analysis
        if layers:
            layer_list = [int(x) for x in layers] if isinstance(layers, list) else [layers]
            result = scanner.find_copying_heads(
                threshold=threshold,
                num_examples=num_examples,
                layers=layer_list
            )
        else:
            result = scanner.find_copying_heads(
                threshold=threshold,
                num_examples=num_examples
            )
        
        return jsonify({
            'status': 'success',
            'result': result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error finding copying heads: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if current_app.debug else None
        }), 500


@circuits_bp.route('/analyze-composition', methods=['POST'])
def analyze_composition():
    """Analyze attention head composition patterns."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body is required'
            }), 400
        
        # Extract parameters
        source_layers = data.get('source_layers')
        target_layers = data.get('target_layers')
        composition_threshold = data.get('composition_threshold', 0.1)
        top_k_pairs = data.get('top_k_pairs', 20)
        
        # Initialize analyzer
        model_manager = get_model_manager()
        config = get_config()
        analyzer = AttentionHeadCompositionAnalyzer(model_manager, config)
        
        # Perform analysis
        result = analyzer.analyze_head_composition(
            source_layers=source_layers,
            target_layers=target_layers,
            composition_threshold=composition_threshold,
            top_k_pairs=top_k_pairs
        )
        
        return jsonify({
            'status': 'success',
            'result': result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing composition: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if current_app.debug else None
        }), 500


@circuits_bp.route('/analyze-neuron-heads', methods=['POST'])
def analyze_neuron_heads():
    """Analyze influence of MLP neurons on attention heads."""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'input_text is required'
            }), 400
        
        # Extract parameters
        input_text = data['input_text']
        source_layers = data.get('source_layers')
        target_layers = data.get('target_layers')
        top_k_neurons = data.get('top_k_neurons', 50)
        top_k_heads = data.get('top_k_heads', 20)
        influence_threshold = data.get('influence_threshold', 0.1)
        
        # Initialize analyzer
        model_manager = get_model_manager()
        config = get_config()
        analyzer = NeuronHeadAnalyzer(model_manager, config)
        
        # Perform analysis
        result = analyzer.analyze_neuron_head_influence(
            input_text=input_text,
            source_layers=source_layers,
            target_layers=target_layers,
            top_k_neurons=top_k_neurons,
            top_k_heads=top_k_heads,
            influence_threshold=influence_threshold
        )
        
        return jsonify({
            'status': 'success',
            'result': result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing neuron-head influence: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if current_app.debug else None
        }), 500


@circuits_bp.route('/find-critical-paths', methods=['POST'])
def find_critical_paths():
    """Find critical paths through neuron-head connections."""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'input_text is required'
            }), 400
        
        # Extract parameters
        input_text = data['input_text']
        min_path_length = data.get('min_path_length', 2)
        max_path_length = data.get('max_path_length', 5)
        influence_threshold = data.get('influence_threshold', 0.2)
        
        # Initialize analyzer
        model_manager = get_model_manager()
        config = get_config()
        analyzer = NeuronHeadAnalyzer(model_manager, config)
        
        # Find critical paths
        paths = analyzer.find_critical_neuron_head_paths(
            input_text=input_text,
            min_path_length=min_path_length,
            max_path_length=max_path_length,
            influence_threshold=influence_threshold
        )
        
        return jsonify({
            'status': 'success',
            'critical_paths': paths,
            'num_paths': len(paths),
            'parameters': {
                'min_path_length': min_path_length,
                'max_path_length': max_path_length,
                'influence_threshold': influence_threshold
            }
        })
        
    except Exception as e:
        logger.error(f"Error finding critical paths: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if current_app.debug else None
        }), 500


@circuits_bp.route('/get-circuit/<circuit_id>', methods=['GET'])
def get_circuit(circuit_id: str):
    """Get details of a specific circuit."""
    try:
        # This would typically load from a database or file system
        # For now, return a placeholder response
        return jsonify({
            'status': 'success',
            'circuit': {
                'id': circuit_id,
                'name': f'Circuit {circuit_id}',
                'description': 'Circuit loaded from storage',
                'components': [],
                'connections': []
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting circuit: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@circuits_bp.route('/list-circuits', methods=['GET'])
def list_circuits():
    """List available circuit files."""
    try:
        # Get circuit directory from query params or use default
        circuit_dir = request.args.get('directory', 'outputs/circuits')
        
        circuits_path = Path(circuit_dir)
        if not circuits_path.exists():
            return jsonify({
                'status': 'success',
                'circuits': [],
                'message': f'Circuit directory {circuit_dir} does not exist'
            })
        
        circuits = []
        
        # Find JSON and GraphML files
        for file_path in circuits_path.glob('**/*'):
            if file_path.suffix.lower() in ['.json', '.graphml']:
                try:
                    circuit_info = {
                        'file': str(file_path),
                        'name': file_path.stem,
                        'format': file_path.suffix[1:].lower(),
                        'size': file_path.stat().st_size,
                        'modified': file_path.stat().st_mtime
                    }
                    
                    # Try to get circuit metadata for JSON files
                    if file_path.suffix.lower() == '.json':
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    circuit_info.update({
                                        'circuit_id': data.get('circuit_id', file_path.stem),
                                        'model_name': data.get('model_name', 'unknown'),
                                        'components': len(data.get('components', [])),
                                        'connections': len(data.get('connections', []))
                                    })
                        except (json.JSONDecodeError, KeyError):
                            pass
                    
                    circuits.append(circuit_info)
                    
                except OSError:
                    continue
        
        return jsonify({
            'status': 'success',
            'circuits': circuits,
            'directory': circuit_dir
        })
        
    except Exception as e:
        logger.error(f"Error listing circuits: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@circuits_bp.route('/export-circuit', methods=['POST'])
def export_circuit():
    """Export a circuit to file."""
    try:
        data = request.get_json()
        if not data or 'circuit' not in data:
            return jsonify({
                'status': 'error',
                'message': 'circuit data is required'
            }), 400
        
        circuit_data = data['circuit']
        output_path = data.get('output_path', 'outputs/circuits/exported_circuit')
        export_format = data.get('format', 'json')
        
        # Create circuit object
        circuit = NeuralCircuit.from_dict(circuit_data)
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if export_format == 'json':
            json_path = output_path_obj.with_suffix('.json')
            circuit.save_json(str(json_path))
            file_path = str(json_path)
        elif export_format == 'graphml':
            graphml_path = output_path_obj.with_suffix('.graphml')
            circuit.save_graphml(str(graphml_path))
            file_path = str(graphml_path)
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported export format: {export_format}'
            }), 400
        
        return jsonify({
            'status': 'success',
            'message': f'Circuit exported successfully',
            'file_path': file_path,
            'format': export_format
        })
        
    except Exception as e:
        logger.error(f"Error exporting circuit: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Error handlers
@circuits_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@circuits_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405


@circuits_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500
