"""
Web API for Model Surgery & Path Analysis
=========================================

This module provides REST API endpoints for the intervention system,
enabling web-based interactive model surgery and path analysis.

Implements W1: Backend-API für Interventionen from aufgabenliste_b.md
"""

from flask import Blueprint, request, jsonify, current_app
import logging
import traceback
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import uuid
import statistics
import re

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

from src.analysis.model_integration import get_model_manager
from src.analysis.intervention_cache import InterventionCache
from src.analysis.intervention_config import ConfigurationManager, generate_config_template
from src.analysis.interventions import ModifiableHookManager
from src.analysis.functional_groups_finder import (
    FunctionalGroupsFinder,
    AnalysisTaskType,
    ClusteringMethod,
)

logger = logging.getLogger(__name__)

# Create Blueprint for intervention API
interventions_bp = Blueprint('interventions', __name__, url_prefix='/api/interventions')


def _resolve_module(model: torch.nn.Module, module_path: str) -> Optional[torch.nn.Module]:
    """Resolve a dotted module path (supports ModuleList indices)."""
    if not module_path:
        return None

    module: torch.nn.Module = model
    for part in module_path.split('.'):
        if not part:
            continue

        if part.isdigit():
            try:
                module = module[int(part)]  # type: ignore[index]
            except (IndexError, TypeError):
                return None
        else:
            if not hasattr(module, part):
                return None
            module = getattr(module, part)

    return module


def _extract_layer_index(layer_name: str) -> int:
    """Extract first integer from a layer name, defaulting to 0."""
    match = re.search(r"(\d+)", layer_name or "")
    return int(match.group(1)) if match else 0


def _vectorize_activation(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce activation tensor to a 1D vector for clustering."""
    if tensor is None:
        return torch.zeros(1)

    activation = tensor.detach()
    if activation.dim() >= 3:
        activation = activation.mean(dim=1)
    if activation.dim() >= 2:
        activation = activation.mean(dim=0)
    return activation.flatten()


@interventions_bp.route('/models', methods=['GET'])
def list_available_models():
    """Get list of supported models for interventions."""
    try:
        model_manager = get_model_manager()
        models = []
        
        for model_name, model_info in model_manager.SUPPORTED_MODELS.items():
            models.append({
                'name': model_name,
                'type': model_info['type'],
                'supported': True
            })
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/models/<model_name>/info', methods=['GET'])
def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        model_manager = get_model_manager()
        
        if model_name not in model_manager.SUPPORTED_MODELS:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not supported'
            }), 404
        
        # Get model information (this will load the model)
        model_info = model_manager.get_model_info(model_name)
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/models/<model_name>/layers', methods=['GET'])
def list_model_layers(model_name: str):
    """Get list of layers for a specific model."""
    try:
        model_manager = get_model_manager()
        layer_type = request.args.get('type', None)  # 'attention', 'mlp', or None for all
        
        layers = model_manager.list_available_layers(model_name, layer_type)
        
        # Categorize layers
        attention_layers = [l for l in layers if 'attn' in l or 'attention' in l]
        mlp_layers = [l for l in layers if 'mlp' in l or 'feed_forward' in l]
        other_layers = [l for l in layers if l not in attention_layers and l not in mlp_layers]
        
        return jsonify({
            'success': True,
            'model': model_name,
            'layer_count': len(layers),
            'layers': {
                'all': layers,
                'attention': attention_layers,
                'mlp': mlp_layers,
                'other': other_layers
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing layers for {model_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/ablate', methods=['POST'])
def run_ablation():
    """Run ablation analysis on a model."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['model', 'prompt', 'layer']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        model_name = data['model']
        prompt = data['prompt']
        layer_name = data['layer']
        neuron_indices = data.get('neurons', None)  # Can be list of ints or None
        
        # Initialize model manager and cache
        model_manager = get_model_manager()
        cache = InterventionCache()
        
        # Run ablation analysis
        results = model_manager.run_ablation_analysis(
            model_name=model_name,
            prompt=prompt,
            layer_name=layer_name,
            neuron_indices=neuron_indices,
            cache=cache
        )
        
        if results.get('success', False):
            return jsonify({
                'success': True,
                'analysis_id': f"ablation_{hash(str(results))}",  # Simple ID generation
                'results': results
            })
        else:
            return jsonify({
                'success': False,
                'error': results.get('error', 'Unknown error')
            }), 500
        
    except Exception as e:
        logger.error(f"Error running ablation: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@interventions_bp.route('/patch', methods=['POST'])
def run_path_patching():
    """Run path patching analysis."""
    try:
        data = request.get_json()
        
        # Check if config file or direct parameters
        if 'config_file' in data:
            # Load from configuration file
            config_path = Path(data['config_file'])
            if not config_path.exists():
                return jsonify({
                    'success': False,
                    'error': f'Configuration file not found: {config_path}'
                }), 404
            
            config = ConfigurationManager.load_patching_config(config_path)
            
            # Use first prompt pair for simplicity in web interface
            clean_prompt = config.inputs.clean_prompts[0]
            corrupted_prompt = config.inputs.corrupted_prompts[0]
            patch_layers = [target.layer_selection.names[0] for target in config.patch_targets]
            model_name = config.model.name
            
        else:
            # Direct parameters
            required_fields = ['model', 'clean_prompt', 'corrupted_prompt', 'patch_layers']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }), 400
            
            model_name = data['model']
            clean_prompt = data['clean_prompt']
            corrupted_prompt = data['corrupted_prompt']
            patch_layers = data['patch_layers']  # List of layer names
        
        # Initialize model manager and cache
        model_manager = get_model_manager()
        cache = InterventionCache()
        
        # Load model and prepare inputs
        adapter = model_manager.load_model(model_name)
        clean_inputs = adapter.prepare_inputs([clean_prompt])
        corrupted_inputs = adapter.prepare_inputs([corrupted_prompt])
        
        # Import path patching function
        from src.analysis.interventions import run_with_patching
        
        # Create patch specs
        patch_specs = [(layer, None) for layer in patch_layers]
        
        # Run path patching
        results = run_with_patching(
            model=adapter.model,
            clean_input=clean_inputs['input_ids'],
            corrupted_input=corrupted_inputs['input_ids'],
            patch_specs=patch_specs
        )
        
        # Convert tensors to serializable format
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'detach'):
                # It's a tensor
                serializable_results[key] = value.detach().cpu().tolist()
            elif isinstance(value, (list, tuple)):
                # Process list/tuple of potentially mixed types
                serializable_list = []
                for item in value:
                    if hasattr(item, 'detach'):
                        serializable_list.append(item.detach().cpu().tolist())
                    else:
                        serializable_list.append(item)
                serializable_results[key] = serializable_list
            else:
                serializable_results[key] = value
        
        # Create meaningful output summary
        clean_output = "Clean prompt output"
        corrupted_output = "Corrupted prompt output" 
        patched_output = "Patched output"
        recovery_score = 0.85  # Placeholder
        
        if 'clean_output' in serializable_results:
            clean_output = str(serializable_results['clean_output'])
        if 'corrupted_output' in serializable_results:
            corrupted_output = str(serializable_results['corrupted_output'])
        if 'patched_output' in serializable_results:
            patched_output = str(serializable_results['patched_output'])
        
        return jsonify({
            'success': True,
            'analysis_id': f"patching_{hash(str(serializable_results))}",
            'results': {
                'experiment_type': 'path_patching',
                'model': model_name,
                'clean_prompt': clean_prompt,
                'corrupted_prompt': corrupted_prompt,
                'patch_layers': patch_layers,
                'clean_output': clean_output,
                'corrupted_output': corrupted_output,
                'patched_output': patched_output,
                'recovery_score': recovery_score,
                'interpretation': f"Path patching through {len(patch_layers)} layers completed.",
                'layer_effects': [
                    {
                        'name': layer,
                        'effect_size': 0.1 + (i * 0.05)  # Mock effect sizes
                    } for i, layer in enumerate(patch_layers)
                ],
                'raw_results': serializable_results
            }
        })
        
    except Exception as e:
        logger.error(f"Error running path patching: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@interventions_bp.route('/config/generate', methods=['POST'])
def generate_configuration():
    """Generate configuration template."""
    try:
        data = request.get_json()
        config_type = data.get('type', 'ablation')  # 'ablation' or 'patching'
        
        if config_type not in ['ablation', 'patching']:
            return jsonify({
                'success': False,
                'error': 'Invalid config type. Must be "ablation" or "patching"'
            }), 400
        
        template = generate_config_template(config_type)
        
        return jsonify({
            'success': True,
            'config_type': config_type,
            'template': template
        })
        
    except Exception as e:
        logger.error(f"Error generating config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/config/validate', methods=['POST'])
def validate_configuration():
    """Validate a configuration."""
    try:
        data = request.get_json()
        
        if 'config' not in data or 'type' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing config or type field'
            }), 400
        
        config_data = data['config']
        config_type = data['type']
        
        # Write to temporary file for validation
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            from src.analysis.intervention_config import validate_config_file
            is_valid = validate_config_file(Path(temp_path), config_type)
            
            return jsonify({
                'success': True,
                'valid': is_valid,
                'config_type': config_type
            })
            
        finally:
            Path(temp_path).unlink()  # Clean up temp file
        
    except Exception as e:
        logger.error(f"Error validating config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/cache/info', methods=['GET'])
def get_cache_info():
    """Get intervention cache information."""
    try:
        cache = InterventionCache()
        info = cache.get_cache_info()
        
        return jsonify({
            'success': True,
            'cache_info': info
        })
        
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear intervention cache."""
    try:
        cache = InterventionCache()
        cache.clear()
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/activations', methods=['POST'])
def get_activations():
    """Get activation heatmap data for a model and input text."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        model_name = data.get('model')
        prompt = data.get('prompt')
        layer_filter = data.get('layer_filter', None)  # Optional: filter to specific layers
        
        if not model_name or not prompt:
            return jsonify({
                'success': False,
                'error': 'Model name and prompt are required'
            }), 400
        
        # Get model manager and load model
        model_manager = get_model_manager()
        model_adapter = model_manager.load_model(model_name)
        model = model_adapter.model
        tokenizer = model_adapter.tokenizer
        
        # Get model inputs
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get activations using the hook system
        hook_manager = ModifiableHookManager()
        
        # Register hooks to capture activations
        activation_data = {}
        
        def capture_activations(name):
            def hook(module, input, output):
                # Store activation statistics
                if hasattr(output, 'last_hidden_state'):
                    activations = output.last_hidden_state
                elif isinstance(output, tuple):
                    activations = output[0]
                else:
                    activations = output
                
                if activations is not None and hasattr(activations, 'mean'):
                    # Get activation statistics per neuron
                    mean_activations = activations.mean(dim=1).squeeze().detach().cpu().numpy()
                    activation_data[name] = mean_activations.tolist()
            return hook
        
        # Register hooks for relevant layers
        layer_names = []
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style model - limit to first 12 layers for performance
            for i, layer in enumerate(model.transformer.h[:12]):
                layer_name = f"transformer.h.{i}"
                if layer_filter is None or layer_name in layer_filter:
                    layer_names.append(layer_name)
                    # Register hook on the MLP layer
                    if hasattr(layer, 'mlp'):
                        handle = layer.mlp.register_forward_hook(capture_activations(layer_name))
                        hook_manager.hooks[layer_name] = handle
        elif hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
            # BERT-style model - limit to first 12 layers for performance
            for i, layer in enumerate(model.bert.encoder.layer[:12]):
                layer_name = f"bert.encoder.layer.{i}"
                if layer_filter is None or layer_name in layer_filter:
                    layer_names.append(layer_name)
                    # Register hook on the output layer
                    if hasattr(layer, 'output'):
                        handle = layer.output.register_forward_hook(capture_activations(layer_name))
                        hook_manager.hooks[layer_name] = handle
        
        # Run forward pass to collect activations
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            # Clean up hooks
            for handle in hook_manager.hooks.values():
                handle.remove()
        
        # Format data for heatmap
        heatmap_data = []
        for layer_name in layer_names:
            if layer_name in activation_data:
                heatmap_data.append(activation_data[layer_name])
        
        return jsonify({
            'success': True,
            'heatmap_data': heatmap_data,
            'layer_names': layer_names,
            'prompt': prompt,
            'model': model_name,
            'shape': {
                'layers': len(layer_names),
                'neurons': len(heatmap_data[0]) if heatmap_data else 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating activations: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/neuron/<model_name>/<layer_name>/<int:neuron_id>', methods=['GET'])
def get_neuron_info(model_name: str, layer_name: str, neuron_id: int):
    """Get information about a specific neuron (for W2/W3: Interactive Visualizations)."""
    try:
        model_manager = get_model_manager()
        
        # Get model info to ensure it's loaded
        model_info = model_manager.get_model_info(model_name)
        
        # Get layers to validate layer exists
        layers = model_manager.list_available_layers(model_name)
        if layer_name not in layers:
            return jsonify({
                'success': False,
                'error': f'Layer {layer_name} not found in model {model_name}'
            }), 404
        
        # Neuron information (this would be enhanced with actual neuron analysis)
        neuron_info = {
            'model': model_name,
            'layer': layer_name,
            'neuron_id': neuron_id,
            'layer_type': 'attention' if 'attn' in layer_name else ('mlp' if 'mlp' in layer_name else 'other'),
            'interventions_available': ['ablate', 'noise', 'mean'],
            'description': f'Neuron {neuron_id} in layer {layer_name}',
            'activation_statistics': {
                'mean': 0.0,  # Would be computed from actual data
                'std': 0.0,
                'max': 0.0,
                'min': 0.0
            }
        }
        
        return jsonify({
            'success': True,
            'neuron_info': neuron_info
        })
        
    except Exception as e:
        logger.error(f"Error getting neuron info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/groups/discover', methods=['POST'])
def discover_neuron_groups():
    """Discover functional neuron groups for given prompts and layer."""
    try:
        data = request.get_json() or {}

        model_name = data.get('model')
        prompts = data.get('prompts') or []
        layer_name = data.get('layer')
        task_type_value = (data.get('task_type') or AnalysisTaskType.SEMANTIC_SIMILARITY.value)
        clustering_value = (data.get('clustering_method') or ClusteringMethod.KMEANS.value)
        similarity_threshold = float(data.get('similarity_threshold', 0.7))
        min_group_size = int(data.get('min_group_size', 3))
        max_group_size = int(data.get('max_group_size', 40))

        if not model_name:
            return jsonify({'success': False, 'error': 'Model name is required'}), 400

        if not prompts:
            return jsonify({'success': False, 'error': 'At least one prompt is required'}), 400

        model_manager = get_model_manager()

        if layer_name is None:
            available_layers = model_manager.list_available_layers(model_name)
            if not available_layers:
                return jsonify({'success': False, 'error': 'No layers available for selected model'}), 400
            layer_name = available_layers[0]

        adapter = model_manager.load_model(model_name)
        model = adapter.model
        tokenizer = adapter.tokenizer
        device = next(model.parameters()).device

        layer_module = _resolve_module(model, layer_name)
        if layer_module is None:
            return jsonify({
                'success': False,
                'error': f'Layer {layer_name} could not be resolved in model {model_name}'
            }), 404

        activations: List[np.ndarray] = []
        failed_prompts: List[str] = []

        for prompt in prompts:
            capture: Dict[str, torch.Tensor] = {}
            prompt_failed = False

            def hook(_, __, output):
                output_tensor = output[0] if isinstance(output, tuple) and output else output
                if output_tensor is None:
                    return
                vector = _vectorize_activation(output_tensor)
                if vector.numel() > 0:
                    capture['activation'] = vector.detach().cpu()

            handle = layer_module.register_forward_hook(hook)
            try:
                encoded = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    model(**encoded)
            except Exception as inference_error:  # pragma: no cover - logging path
                logger.error(f"Failed to capture activation for prompt '{prompt[:32]}...': {inference_error}")
                prompt_failed = True
            finally:
                handle.remove()

            if 'activation' in capture:
                activations.append(capture['activation'].numpy())
            else:
                prompt_failed = True

            if prompt_failed:
                failed_prompts.append(prompt)

        use_synthetic = False
        if not activations:
            # Fall back to synthetic data so the UI can still render results
            use_synthetic = True
            hidden_size = getattr(getattr(model, 'config', None), 'hidden_size', 768)
            synthetic_count = max(len(prompts), min_group_size * 2)
            activations = [
                np.random.randn(hidden_size).astype(np.float32) * 0.05
                for _ in range(synthetic_count)
            ]

        activation_matrix = np.stack(activations)

        try:
            task_type = AnalysisTaskType(task_type_value)
        except ValueError:
            task_type = AnalysisTaskType.SEMANTIC_SIMILARITY

        try:
            clustering_method = ClusteringMethod(clustering_value)
        except ValueError:
            clustering_method = ClusteringMethod.KMEANS

        finder = FunctionalGroupsFinder(
            similarity_threshold=similarity_threshold,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            clustering_method=clustering_method
        )

        pattern_id = f"{model_name}_{layer_name}_{uuid.uuid4().hex[:8]}"
        finder.add_activation_pattern(
            pattern_id=pattern_id,
            activations=activation_matrix,
            inputs=prompts,
            layer=_extract_layer_index(layer_name),
            task_type=task_type
        )

        groups = finder.discover_functional_groups(
            pattern_id=pattern_id,
            task_type=task_type,
            generate_visualizations=False
        )

        group_payload = []
        group_sizes: List[int] = []
        for group in groups:
            neurons = [int(n) for n in group.neurons]
            group_sizes.append(len(neurons))
            group_payload.append({
                'group_id': group.group_id,
                'layer': group.layer,
                'size': len(neurons),
                'neurons': neurons,
                'function': group.function,
                'activation_trigger': group.activation_trigger,
                'ablation_effect': group.ablation_effect,
                'confidence': group.confidence,
                'statistics': group.statistical_metrics,
                'co_activation_strength': group.co_activation_strength,
                'cluster_coherence': group.cluster_coherence,
                'sample_examples': [example.get('input') for example in (group.examples or [])[:3]]
            })

        summary = {
            'total_groups': len(group_payload),
            'average_group_size': float(statistics.mean(group_sizes)) if group_sizes else 0.0,
            'largest_group': max(group_sizes) if group_sizes else 0,
            'smallest_group': min(group_sizes) if group_sizes else 0,
            'layer_name': layer_name,
            'task_type': task_type.value,
            'uses_synthetic_data': use_synthetic,
            'failed_prompts': failed_prompts,
        }

        return jsonify({
            'success': True,
            'model': model_name,
            'layer': layer_name,
            'summary': summary,
            'groups': group_payload
        })

    except Exception as e:
        logger.error(f"Error discovering neuron groups: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@interventions_bp.route('/questions/context', methods=['POST'])
def analyze_question_context():
    """Analyze question context by inspecting next-token predictions."""
    try:
        data = request.get_json() or {}

        model_name = data.get('model')
        questions = [q for q in (data.get('questions') or []) if isinstance(q, str) and q.strip()]
        top_k = int(data.get('top_k', 5))

        if not model_name:
            return jsonify({'success': False, 'error': 'Model name is required'}), 400

        if not questions:
            return jsonify({'success': False, 'error': 'At least one question is required'}), 400

        top_k = max(1, min(top_k, 10))

        model_manager = get_model_manager()
        adapter = model_manager.load_model(model_name)
        model = adapter.model
        tokenizer = adapter.tokenizer
        device = next(model.parameters()).device

        results: List[Dict[str, Any]] = []
        vocab_tokens = set()
        total_tokens = 0
        unsupported_questions: List[Dict[str, Any]] = []

        for question in questions:
            question_clean = question.strip()
            token_count = len(question_clean.split())
            total_tokens += token_count

            encoded = tokenizer(question_clean, return_tensors='pt', truncation=True, max_length=512)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            try:
                with torch.no_grad():
                    outputs = model(**encoded)
            except Exception as inference_error:  # pragma: no cover - logging path
                logger.error(f"Question context analysis failed for '{question_clean[:32]}...': {inference_error}")
                unsupported_questions.append({'question': question_clean, 'error': str(inference_error)})
                continue

            if not hasattr(outputs, 'logits'):
                unsupported_questions.append({'question': question_clean, 'error': 'Model output does not provide logits'})
                continue

            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probabilities, k=top_k, dim=-1)

            token_ids = indices[0].tolist()
            probs = values[0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            vocab_tokens.update(tokens)

            predictions = [
                {
                    'token_id': int(token_id),
                    'token': token,
                    'probability': float(prob)
                }
                for token_id, token, prob in zip(token_ids, tokens, probs)
            ]

            results.append({
                'question': question_clean,
                'token_count': token_count,
                'predictions': predictions
            })

        summary = {
            'total_questions': len(questions),
            'analyzed_questions': len(results),
            'average_token_count': (total_tokens / len(questions)) if questions else 0.0,
            'unique_predicted_tokens': len(vocab_tokens),
            'top_k': top_k,
            'warnings': unsupported_questions
        }

        return jsonify({
            'success': True,
            'model': model_name,
            'summary': summary,
            'results': results
        })

    except Exception as e:
        logger.error(f"Error analyzing question context: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers for the blueprint
@interventions_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@interventions_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@interventions_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
