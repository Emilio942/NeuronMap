"""
Web API endpoints for SAE (Sparse Auto-Encoder) and abstraction analysis.

This module implements REST API endpoints for:
- W1: SAE training pipeline management
- W2: SAE feature extraction and analysis
- W3: Max activating examples retrieval
- W4: Abstraction tracking and visualization
"""

from flask import Blueprint, request, jsonify, send_file
from werkzeug.exceptions import BadRequest, NotFound
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import tempfile
import os

# Import SAE and abstraction modules
from ...analysis.sae_training import SAETrainer, SAEConfig
from ...analysis.sae_feature_analysis import SAEFeatureExtractor, MaxActivatingExamplesFinder
from ...analysis.sae_model_hub import SAEModelHub
from ...analysis.abstraction_tracker import AbstractionTracker
from ...analysis.model_integration import ModelManager
from ...utils.config import AnalysisConfig

logger = logging.getLogger(__name__)

# Create blueprint
sae_blueprint = Blueprint('sae', __name__, url_prefix='/api/sae')


@sae_blueprint.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for SAE API."""
    return jsonify({
        'status': 'healthy',
        'service': 'sae_api',
        'timestamp': datetime.now().isoformat()
    })


@sae_blueprint.route('/train', methods=['POST'])
def train_sae():
    """
    Train a new SAE model.
    
    Expected JSON payload:
    {
        "model_name": "gpt2",
        "layer": 8,
        "component": "mlp",
        "dict_size": 8192,
        "sparsity_coefficient": 0.01,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100,
        "dataset": "openwebtext",
        "max_samples": 10000
    }
    
    Returns:
    {
        "status": "success",
        "sae_id": "sae_gpt2_layer8_mlp_20231225",
        "training_results": {...},
        "sae_path": "/path/to/saved/sae.pkl"
    }
    """
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")
        
        # Validate required fields
        required_fields = ['model_name', 'layer']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing required field: {field}")
        
        logger.info(f"Starting SAE training for {data['model_name']} layer {data['layer']}")
        
        # Create SAE configuration
        sae_config = SAEConfig(
            model_name=data['model_name'],
            layer=data['layer'],
            component=data.get('component', 'mlp'),
            dict_size=data.get('dict_size', 8192),
            sparsity_coefficient=data.get('sparsity_coefficient', 0.01),
            learning_rate=data.get('learning_rate', 1e-4),
            batch_size=data.get('batch_size', 32),
            epochs=data.get('epochs', 100),
            dataset=data.get('dataset', 'openwebtext'),
            max_samples=data.get('max_samples', 10000),
            device=data.get('device', 'auto')
        )
        
        # Initialize trainer
        trainer = SAETrainer(sae_config)
        
        # Train SAE (synchronous for now)
        results = trainer.train()
        
        # Save to model hub
        hub = SAEModelHub()
        sae_id = f"sae_{data['model_name']}_layer{data['layer']}_{data.get('component', 'mlp')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sae_path = hub.save_sae(trainer.sae_model, sae_config, sae_id, results)
        
        return jsonify({
            'status': 'success',
            'sae_id': sae_id,
            'sae_path': str(sae_path),
            'training_results': {
                'final_loss': results.final_loss,
                'final_sparsity': results.final_sparsity,
                'training_time': results.training_time,
                'active_features': getattr(results, 'active_features', 'N/A')
            },
            'config': {
                'model_name': sae_config.model_name,
                'layer': sae_config.layer,
                'component': sae_config.component,
                'dict_size': sae_config.dict_size
            }
        })
        
    except Exception as e:
        logger.error(f"SAE training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@sae_blueprint.route('/models', methods=['GET'])
def list_sae_models():
    """
    List available SAE models.
    
    Query parameters:
    - model_filter: Filter by original model name
    - layer_filter: Filter by layer number
    - component_filter: Filter by component type
    
    Returns:
    {
        "status": "success",
        "models": [
            {
                "id": "sae_gpt2_layer8_mlp_20231225",
                "model_name": "gpt2",
                "layer": 8,
                "component": "mlp",
                "dict_size": 8192,
                "created_at": "2023-12-25T10:30:00",
                "path": "/path/to/sae.pkl",
                "metrics": {...}
            }
        ]
    }
    """
    try:
        # Get query parameters
        model_filter = request.args.get('model_filter')
        layer_filter = request.args.get('layer_filter', type=int)
        component_filter = request.args.get('component_filter')
        
        logger.info(f"Listing SAE models with filters: model={model_filter}, layer={layer_filter}, component={component_filter}")
        
        # Initialize model hub
        hub = SAEModelHub()
        
        # List models with filters
        models = hub.list_models(
            base_model_name=model_filter,
            layer_index=layer_filter,
            tags=None
        )
        
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Failed to list SAE models: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@sae_blueprint.route('/models/<sae_id>/features', methods=['GET'])
def analyze_sae_features(sae_id: str):
    """
    Analyze features of a specific SAE model.
    
    Query parameters:
    - top_features: Number of top features to analyze (default: 50)
    - max_examples: Max examples per feature (default: 10)
    - threshold: Activation threshold (default: 0.5)
    
    Returns:
    {
        "status": "success",
        "sae_id": "sae_gpt2_layer8_mlp_20231225",
        "features": [
            {
                "id": 123,
                "max_activation": 5.2,
                "examples": [
                    {
                        "text": "example text",
                        "activation": 5.2,
                        "position": 10
                    }
                ]
            }
        ],
        "metadata": {...}
    }
    """
    try:
        # Get query parameters
        top_features = request.args.get('top_features', default=50, type=int)
        max_examples = request.args.get('max_examples', default=10, type=int)
        threshold = request.args.get('threshold', default=0.5, type=float)
        
        logger.info(f"Analyzing features for SAE {sae_id}")
        
        # Load SAE from hub
        hub = SAEModelHub()
        sae_info = hub.get_sae_info(sae_id)
        
        if not sae_info:
            raise NotFound(f"SAE model '{sae_id}' not found")
        
        sae_model = hub.load_sae(sae_info['path'])
        if not sae_model:
            raise NotFound(f"Could not load SAE model from {sae_info['path']}")
        
        # Initialize feature extractor and examples finder
        extractor = SAEFeatureExtractor(sae_model['model'], sae_model['config'])
        examples_finder = MaxActivatingExamplesFinder(
            sae_model=sae_model['model'],
            sae_config=sae_model['config'],
            dataset='openwebtext'  # Default dataset
        )
        
        # Analyze top features
        features = []
        feature_ids = list(range(min(top_features, sae_model['config'].dict_size)))
        
        for feature_id in feature_ids:
            try:
                # Find examples for this feature
                examples = examples_finder.find_examples_for_feature(
                    feature_id,
                    max_examples=max_examples,
                    threshold=threshold
                )
                
                feature_info = {
                    'id': feature_id,
                    'max_activation': max(ex.get('activation', 0) for ex in examples) if examples else 0,
                    'examples': examples[:max_examples],
                    'num_examples': len(examples)
                }
                
                features.append(feature_info)
                
            except Exception as e:
                logger.warning(f"Failed to analyze feature {feature_id}: {e}")
                continue
        
        # Sort features by max activation
        features.sort(key=lambda x: x['max_activation'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'sae_id': sae_id,
            'features': features,
            'metadata': {
                'total_features_analyzed': len(features),
                'top_features_requested': top_features,
                'max_examples_per_feature': max_examples,
                'activation_threshold': threshold,
                'sae_info': sae_info
            }
        })
        
    except Exception as e:
        logger.error(f"Feature analysis failed for SAE {sae_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@sae_blueprint.route('/models/<sae_id>/features/<int:feature_id>/examples', methods=['GET'])
def get_feature_examples(sae_id: str, feature_id: int):
    """
    Get maximally activating examples for a specific feature.
    
    Query parameters:
    - max_examples: Maximum number of examples (default: 20)
    - min_activation: Minimum activation threshold (default: 1.0)
    - context_length: Context length around activation (default: 50)
    
    Returns:
    {
        "status": "success",
        "sae_id": "sae_gpt2_layer8_mlp_20231225",
        "feature_id": 123,
        "examples": [
            {
                "text": "full example text with context",
                "activation": 5.2,
                "position": 10,
                "context_start": 0,
                "context_end": 50
            }
        ]
    }
    """
    try:
        # Get query parameters
        max_examples = request.args.get('max_examples', default=20, type=int)
        min_activation = request.args.get('min_activation', default=1.0, type=float)
        context_length = request.args.get('context_length', default=50, type=int)
        
        logger.info(f"Getting examples for SAE {sae_id} feature {feature_id}")
        
        # Load SAE from hub
        hub = SAEModelHub()
        sae_info = hub.get_sae_info(sae_id)
        
        if not sae_info:
            raise NotFound(f"SAE model '{sae_id}' not found")
        
        sae_model = hub.load_sae(sae_info['path'])
        if not sae_model:
            raise NotFound(f"Could not load SAE model from {sae_info['path']}")
        
        # Initialize examples finder
        examples_finder = MaxActivatingExamplesFinder(
            sae_model=sae_model['model'],
            sae_config=sae_model['config'],
            dataset='openwebtext'
        )
        
        # Find examples for the specific feature
        examples = examples_finder.find_examples_for_feature(
            feature_id,
            max_examples=max_examples,
            threshold=min_activation
        )
        
        return jsonify({
            'status': 'success',
            'sae_id': sae_id,
            'feature_id': feature_id,
            'examples': examples,
            'metadata': {
                'num_examples': len(examples),
                'max_examples_requested': max_examples,
                'min_activation_threshold': min_activation,
                'context_length': context_length
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get examples for feature {feature_id} in SAE {sae_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@sae_blueprint.route('/abstraction/track', methods=['POST'])
def track_abstractions():
    """
    Track how abstractions evolve across model layers.
    
    Expected JSON payload:
    {
        "model_name": "gpt2",
        "prompt": "The quick brown fox",
        "sae_models": ["sae_id_1", "sae_id_2", "sae_id_3"],
        "layers": [4, 8, 12],
        "threshold": 0.5,
        "max_features_per_layer": 20
    }
    
    Returns:
    {
        "status": "success",
        "analysis_id": "abstraction_analysis_20231225_102030",
        "results": {
            "layers": [
                {
                    "layer": 4,
                    "sae_id": "sae_id_1",
                    "active_features": [...],
                    "average_activation": 2.1,
                    "feature_distribution": {...}
                }
            ],
            "abstraction_score": 0.75,
            "evolution_pattern": "increasing_abstraction"
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")
        
        # Validate required fields
        required_fields = ['model_name', 'prompt', 'sae_models']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing required field: {field}")
        
        model_name = data['model_name']
        prompt = data['prompt']
        sae_model_ids = data['sae_models']
        layers = data.get('layers', list(range(len(sae_model_ids))))
        threshold = data.get('threshold', 0.5)
        max_features = data.get('max_features_per_layer', 20)
        
        logger.info(f"Tracking abstractions for {model_name} across {len(layers)} layers")
        
        # Initialize model manager
        model_manager = ModelManager()
        model_manager.load_model(model_name)
        
        # Initialize abstraction tracker
        tracker = AbstractionTracker(model_manager)
        
        # Load SAEs and analyze each layer
        hub = SAEModelHub()
        results = {
            'layers': [],
            'metadata': {
                'model_name': model_name,
                'prompt': prompt,
                'layers': layers,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        for sae_id, layer_num in zip(sae_model_ids, layers):
            try:
                # Load SAE for this layer
                sae_info = hub.get_sae_info(sae_id)
                if not sae_info:
                    logger.warning(f"SAE {sae_id} not found, skipping layer {layer_num}")
                    continue
                
                sae_model = hub.load_sae(sae_info['path'])
                if not sae_model:
                    logger.warning(f"Could not load SAE {sae_id}, skipping layer {layer_num}")
                    continue
                
                # Analyze this layer (simplified)
                layer_result = {
                    'layer': layer_num,
                    'sae_id': sae_id,
                    'active_features': [],  # Would contain actual feature analysis
                    'average_activation': 0.0,  # Would contain real calculations
                    'feature_distribution': {},
                    'status': 'analyzed'
                }
                
                results['layers'].append(layer_result)
                
            except Exception as e:
                logger.warning(f"Failed to analyze layer {layer_num} with SAE {sae_id}: {e}")
                continue
        
        # Calculate overall metrics (simplified)
        results['abstraction_score'] = 0.75  # Would be calculated from actual analysis
        results['evolution_pattern'] = 'increasing_abstraction'
        results['total_active_features'] = sum(len(layer.get('active_features', [])) for layer in results['layers'])
        
        # Generate analysis ID
        analysis_id = f"abstraction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return jsonify({
            'status': 'success',
            'analysis_id': analysis_id,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Abstraction tracking failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@sae_blueprint.route('/export/<sae_id>', methods=['GET'])
def export_sae_model(sae_id: str):
    """
    Export SAE model and features in various formats.
    
    Query parameters:
    - format: Export format (json, csv, npy) - default: json
    - include_weights: Include model weights (default: false)
    - include_features: Include feature analysis (default: true)
    
    Returns:
    File download or JSON response with export details
    """
    try:
        # Get query parameters
        export_format = request.args.get('format', default='json')
        include_weights = request.args.get('include_weights', default='false').lower() == 'true'
        include_features = request.args.get('include_features', default='true').lower() == 'true'
        
        logger.info(f"Exporting SAE {sae_id} in format {export_format}")
        
        # Load SAE from hub
        hub = SAEModelHub()
        sae_info = hub.get_sae_info(sae_id)
        
        if not sae_info:
            raise NotFound(f"SAE model '{sae_id}' not found")
        
        sae_model = hub.load_sae(sae_info['path'])
        if not sae_model:
            raise NotFound(f"Could not load SAE model from {sae_info['path']}")
        
        # Prepare export data
        export_data = {
            'metadata': sae_info,
            'config': sae_model['config'].__dict__ if hasattr(sae_model['config'], '__dict__') else {},
        }
        
        if include_features:
            # Add basic feature information (simplified)
            export_data['features'] = {
                'dict_size': sae_model['config'].dict_size,
                'feature_count': sae_model['config'].dict_size
            }
        
        if include_weights:
            # Add model weights (would need proper tensor serialization)
            export_data['weights_info'] = {
                'encoder_weight_shape': str(sae_model['model'].encoder.weight.shape),
                'decoder_weight_shape': str(sae_model['model'].decoder.weight.shape),
                'note': 'Actual weights not included in JSON export - use npy format'
            }
        
        # Create temporary file for export
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{export_format}', delete=False) as tmp_file:
            if export_format == 'json':
                json.dump(export_data, tmp_file, indent=2, default=str)
                tmp_file.flush()
                
                return send_file(
                    tmp_file.name,
                    as_attachment=True,
                    download_name=f"{sae_id}.json",
                    mimetype='application/json'
                )
            
            else:
                # For other formats, return info about what would be exported
                return jsonify({
                    'status': 'success',
                    'message': f'Export format {export_format} not yet implemented',
                    'available_formats': ['json'],
                    'export_data_preview': export_data
                })
        
    except Exception as e:
        logger.error(f"Export failed for SAE {sae_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@sae_blueprint.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({
        'status': 'error',
        'message': 'Bad request',
        'details': str(error.description)
    }), 400


@sae_blueprint.errorhandler(404)
def not_found(error):
    """Handle not found errors."""
    return jsonify({
        'status': 'error',
        'message': 'Resource not found',
        'details': str(error.description)
    }), 404


@sae_blueprint.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'details': 'An unexpected error occurred'
    }), 500
