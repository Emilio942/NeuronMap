#!/usr/bin/env python3
"""
Enhanced test server for NeuronMap Web Interface with Analysis Zoo integration
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request

# Import the API blueprints
from src.web.api.interventions import interventions_bp

# Create Flask app
app = Flask(__name__,
            template_folder='web/templates',
            static_folder='web/static')
app.secret_key = 'neuronmap_test_key'

# Register the interventions API blueprint
app.register_blueprint(interventions_bp)

@app.route('/')
def index():
    """Home page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/analysis')
def analysis():
    """Analysis page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/visualization')
def visualization():
    """Visualization page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/multi-model')
def multi_model():
    """Multi-model page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/results')
def results():
    """Results page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/advanced-analytics')
def advanced_analytics():
    """Advanced analytics page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/performance')
def performance():
    """Performance page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/reports')
def reports_page():
    """Reports page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/plugins')
def plugins_page():
    """Plugins page redirect to model surgery."""
    return render_template('model_surgery.html')

@app.route('/model-surgery')
def model_surgery():
    """Interactive Model Surgery & Path Analysis page."""
    return render_template('model_surgery.html')

@app.route('/causal-tracing')
def causal_tracing():
    """Causal Tracing & Path Patching page."""
    return render_template('causal_tracing.html')

@app.route('/causal-path')
def causal_path():
    """Advanced Causal Path Visualization page."""
    return render_template('causal_path.html')

@app.route('/analysis-zoo')
def analysis_zoo():
    """Analysis Zoo artifact gallery and search page."""
    return render_template('analysis_zoo.html')

@app.route('/artifact/<artifact_id>')
def artifact_detail(artifact_id):
    """Individual artifact detail page."""
    return render_template('artifact_detail.html', artifact_id=artifact_id)

# Mock API endpoints for Analysis Zoo
@app.route('/api/zoo/stats')
def zoo_stats():
    """Mock API endpoint for zoo statistics."""
    return jsonify({
        'total_artifacts': 1247,
        'sae_models': 523,
        'circuits': 341,
        'contributors': 89,
        'recent_uploads': 23
    })

@app.route('/api/zoo/artifacts')
def zoo_artifacts():
    """Mock API endpoint for artifact listing."""
    # Mock artifact data
    artifacts = []
    types = ['sae_model', 'circuit', 'intervention_config', 'analysis_result', 'dataset', 'visualization']
    models = ['gpt2', 'llama-2-7b', 'bert-base', 't5-base', 'gpt-4']
    licenses = ['MIT', 'Apache-2.0', 'CC-BY-4.0', 'GPL-3.0']
    authors = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown']
    
    for i in range(48):
        artifact_type = types[i % len(types)]
        artifacts.append({
            'id': f'artifact-{i + 1}',
            'name': f'{artifact_type.replace("_", " ").title()} #{i + 1}',
            'description': f'A high-quality {artifact_type.replace("_", " ")} for neural network analysis.',
            'type': artifact_type,
            'author': {
                'name': authors[i % len(authors)],
                'email': f'{authors[i % len(authors)].lower().replace(" ", ".")}@example.com'
            },
            'created_at': (datetime.now().timestamp() - (i * 86400)),
            'updated_at': (datetime.now().timestamp() - (i * 43200)),
            'model_compatibility': {
                'model_name': models[i % len(models)],
                'model_family': models[i % len(models)].split('-')[0]
            },
            'license': licenses[i % len(licenses)],
            'download_count': 100 + (i * 15),
            'size_bytes': (1024 * 1024) * (50 + (i * 10)),  # Size in bytes
            'tags': ['interpretability', 'analysis', artifact_type.split('_')[0]][:2 + (i % 2)],
            'performance_metrics': {
                'accuracy': 0.8 + (i % 20) * 0.01,
                'loss': 0.1 + (i % 10) * 0.01
            }
        })
    
    # Apply filters
    search = request.args.get('search', '').lower()
    artifact_type = request.args.get('type', '')
    model_family = request.args.get('model', '')
    license_filter = request.args.get('license', '')
    sort_by = request.args.get('sort', 'created_desc')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 12))
    
    # Filter artifacts
    filtered_artifacts = artifacts
    
    if search:
        filtered_artifacts = [a for a in filtered_artifacts 
                            if search in a['name'].lower() or search in a['description'].lower()]
    
    if artifact_type:
        filtered_artifacts = [a for a in filtered_artifacts if a['type'] == artifact_type]
    
    if model_family:
        filtered_artifacts = [a for a in filtered_artifacts 
                            if a['model_compatibility']['model_family'] == model_family]
    
    if license_filter:
        filtered_artifacts = [a for a in filtered_artifacts if a['license'] == license_filter]
    
    # Sort artifacts
    if sort_by == 'created_desc':
        filtered_artifacts.sort(key=lambda x: x['created_at'], reverse=True)
    elif sort_by == 'created_asc':
        filtered_artifacts.sort(key=lambda x: x['created_at'])
    elif sort_by == 'downloads_desc':
        filtered_artifacts.sort(key=lambda x: x['download_count'], reverse=True)
    elif sort_by == 'name_asc':
        filtered_artifacts.sort(key=lambda x: x['name'])
    elif sort_by == 'name_desc':
        filtered_artifacts.sort(key=lambda x: x['name'], reverse=True)
    
    # Paginate
    total = len(filtered_artifacts)
    start = (page - 1) * per_page
    end = start + per_page
    page_artifacts = filtered_artifacts[start:end]
    
    return jsonify({
        'artifacts': page_artifacts,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page
    })

@app.route('/api/zoo/artifacts/<artifact_id>')
def zoo_artifact_detail(artifact_id):
    """Mock API endpoint for individual artifact details."""
    # Mock detailed artifact data
    artifact = {
        'id': artifact_id,
        'name': f'Advanced SAE Model {artifact_id.split("-")[-1]}',
        'description': 'A state-of-the-art Sparse Autoencoder model trained on GPT-2 activations.',
        'type': 'sae_model',
        'author': {
            'name': 'Dr. Alice Johnson',
            'email': 'alice.johnson@university.edu',
            'affiliation': 'AI Research Lab, University of Technology',
            'orcid': '0000-0000-0000-0000'
        },
        'created_at': datetime.now().timestamp() - 86400,
        'updated_at': datetime.now().timestamp() - 43200,
        'version': '1.2.0',
        'model_compatibility': {
            'model_name': 'gpt2',
            'model_family': 'gpt',
            'architecture': 'transformer',
            'layers': [8, 9, 10, 11, 12],
            'min_parameters': 124000000,
            'max_parameters': 1500000000
        },
        'license': 'MIT',
        'download_count': 456,
        'size_bytes': 157286400,  # ~150MB
        'checksum_sha256': 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
        'tags': ['interpretability', 'sae', 'gpt2', 'sparse-autoencoder', 'activation-analysis'],
        'performance_metrics': {
            'accuracy': 0.94,
            'loss': 0.023,
            'sparsity': 0.95,
            'reconstruction_loss': 0.015,
            'custom_metrics': {
                'feature_density': 0.05,
                'dead_features': 0.02
            }
        },
        'files': [
            {
                'path': 'model.pt',
                'size_bytes': 134217728,
                'checksum_sha256': 'abc123...',
                'mime_type': 'application/octet-stream',
                'description': 'PyTorch model file'
            },
            {
                'path': 'config.json',
                'size_bytes': 2048,
                'checksum_sha256': 'def456...',
                'mime_type': 'application/json',
                'description': 'Model configuration'
            },
            {
                'path': 'README.md',
                'size_bytes': 8192,
                'checksum_sha256': 'ghi789...',
                'mime_type': 'text/markdown',
                'description': 'Documentation and usage instructions'
            }
        ],
        'dependencies': [
            {
                'artifact_id': 'artifact-base-123',
                'version': '1.0.0',
                'optional': False,
                'purpose': 'Base model weights'
            }
        ],
        'citation': {
            'title': 'Advanced Sparse Autoencoders for Neural Network Interpretability',
            'authors': [
                {
                    'name': 'Alice Johnson',
                    'affiliation': 'University of Technology'
                }
            ],
            'year': 2024,
            'arxiv_id': '2401.12345',
            'bibtex': '@article{johnson2024advanced,\n  title={Advanced Sparse Autoencoders...},\n  author={Johnson, Alice},\n  year={2024}\n}'
        }
    }
    
    return jsonify(artifact)

@app.route('/test-buttons')
def test_buttons():
    """Test page for button functionality."""
    with open('test_buttons.html', 'r') as f:
        return f.read()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    args = parser.parse_args()
    
    print("Starting Model Surgery Test Server...")
    print(f"Visit: http://localhost:{args.port}/model-surgery")
    app.run(debug=True, port=args.port, host='0.0.0.0')
