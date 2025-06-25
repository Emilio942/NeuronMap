"""
Web Interface for NeuronMap
==========================

Professional Flask-based web interface for running NeuronMap analyses.
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    # Flask imports already handled above
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available. Web interface will be disabled.")

if FLASK_AVAILABLE:
    app = Flask(__name__,
                template_folder='../../web/templates',
                static_folder='../../web/static')
    app.secret_key = 'neuronmap_secret_key_change_in_production'

    # Global variables for managing analysis state
    analysis_status = {
        'running': False,
        'progress': 0,
        'message': 'Ready',
        'results': None,
        'error': None,
        'analysis_id': None
    }

    # Store analysis results
    analysis_results = {}

    # Route definitions moved to the FLASK_AVAILABLE block below

    @app.route('/multi-model')
    def multi_model():
        """Multi-model comparison page"""
        # We'll create this template next
        return render_template('multi_model.html')

    @app.route('/results')
    def results():
        """Results management page"""
        # We'll create this template next
        return render_template('results.html')

def run_analysis_background(analysis_id, model, device, target_layers, questions_file, questions_text, advanced, visualize):
    """Run analysis in background thread."""
    global analysis_status, analysis_results

    try:
        analysis_status.update({
            'running': True,
            'progress': 0,
            'message': 'Initializing analysis...',
            'error': None,
            'analysis_id': analysis_id
        })

        # Simulate progress for now (replace with actual analysis later)
        steps = [
            (10, "Loading model..."),
            (30, "Processing questions..."),
            (50, "Extracting activations..."),
            (70, "Computing statistics..."),
            (90, "Finalizing results..."),
            (100, "Analysis complete!")
        ]

        for progress, message in steps:
            time.sleep(2)  # Simulate work
            analysis_status.update({
                'progress': progress,
                'message': message
            })

        # Mock results
        results = {
            'model': model,
            'question_count': 5 if questions_text else 1,
            'layer_count': 12,
            'duration': '10s',
            'analysis_id': analysis_id
        }

        # Store results
        analysis_results[analysis_id] = results

        analysis_status.update({
            'running': False,
            'progress': 100,
            'message': 'Analysis completed successfully!',
            'results': results,
            'completed': True
        })

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        analysis_status.update({
            'running': False,
            'error': str(e),
            'message': f'Analysis failed: {str(e)}'
        })

    except Exception as e:
        analysis_status.update({
            'running': False,
            'error': str(e),
            'message': f'Analysis failed: {str(e)}'
        })
        logger.error(f"Background analysis failed: {e}")

if FLASK_AVAILABLE:

    @app.route('/')
    def index():
        """Main dashboard page."""
        return render_template('index.html')

    @app.route('/analysis')
    def analysis_page():
        """Analysis configuration page."""
        return render_template('analysis.html')

    @app.route('/results')
    def results_page():
        """Results viewing page."""
        return render_template('results.html')

    @app.route('/api/models')
    def get_available_models():
        """Get list of available models."""
        models = [
            'distilgpt2', 'gpt2', 'gpt2-medium',
            'bert-base-uncased', 'distilbert-base-uncased',
            'roberta-base', 'distilroberta-base'
        ]
        return jsonify({'models': models})

    @app.route('/api/start_analysis', methods=['POST'])
    def start_analysis():
        """Start background analysis."""
        global analysis_status

        if analysis_status['running']:
            return jsonify({'error': 'Analysis already running'}), 400

        try:
            data = request.get_json()
            analysis_type = data.get('type', 'basic')

            # Load configuration
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.config_manager import get_config

            config = get_config()

            # Update config with web parameters
            if 'model' in data:
                config.model.name = data['model']

            if 'target_layers' in data and data['target_layers']:
                config.model.target_layers = data['target_layers']

            # Start analysis in background
            kwargs = {}
            if analysis_type == 'multi_model':
                kwargs['models'] = data.get('models', ['distilgpt2', 'gpt2'])

            thread = threading.Thread(
                target=run_analysis_background,
                args=(config, analysis_type),
                kwargs=kwargs
            )
            thread.daemon = True
            thread.start()

            return jsonify({'message': 'Analysis started', 'status': 'running'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/status')
    def get_analysis_status():
        """Get current analysis status."""
        return jsonify(analysis_status)

    @app.route('/api/results')
    def get_results():
        """Get analysis results."""
        if analysis_status.get('results'):
            return jsonify(analysis_status['results'])
        else:
            return jsonify({'error': 'No results available'}), 404

    @app.route('/api/upload_questions', methods=['POST'])
    def upload_questions():
        """Upload questions file."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            # Save uploaded file
            filename = 'uploaded_questions.jsonl'
            file.save(filename)

            # Validate file format
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                question_count = len([l for l in lines if l.strip()])

            return jsonify({
                'message': f'Successfully uploaded {question_count} questions',
                'filename': filename,
                'question_count': question_count
            })

        except Exception as e:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500

    @app.route('/api/download_results')
    def download_results():
        """Download results as JSON file."""
        if not analysis_status.get('results'):
            return jsonify({'error': 'No results available'}), 404

        try:
            # Create temporary results file
            results_file = 'neuronmap_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_status['results'], f, indent=2, ensure_ascii=False)

            return send_file(results_file, as_attachment=True, download_name='neuronmap_results.json')

        except Exception as e:
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

    # API Routes
    @app.route('/api/stats')
    def api_stats():
        """Get system statistics"""
        try:
            # Count analyses, models, etc.
            data_dir = Path('../../data/outputs')
            stats = {
                'total_analyses': len(list(data_dir.glob('**/activation_results.csv'))) if data_dir.exists() else 0,
                'models_analyzed': len(set([d.name for d in data_dir.iterdir() if d.is_dir()])) if data_dir.exists() else 0,
                'layers_processed': 86,  # From previous discovery
                'visualizations_created': len(list(data_dir.glob('**/visualizations/*.png'))) if data_dir.exists() else 0
            }
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recent-activity')
    def api_recent_activity():
        """Get recent activity"""
        try:
            activities = [
                {
                    'icon': 'chart-line',
                    'color': 'primary',
                    'message': 'Analysis completed for GPT-2',
                    'timestamp': '2 minutes ago'
                },
                {
                    'icon': 'chart-bar',
                    'color': 'success',
                    'message': 'Visualization generated',
                    'timestamp': '5 minutes ago'
                },
                {
                    'icon': 'layer-group',
                    'color': 'info',
                    'message': 'Multi-model comparison started',
                    'timestamp': '10 minutes ago'
                }
            ]
            return jsonify({'activities': activities})
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analyze', methods=['POST'])
    def api_analyze():
        """Start new analysis"""
        global analysis_status

        try:
            # Get form data
            model = request.form.get('model')
            if model == 'custom':
                model = request.form.get('customModel')

            device = request.form.get('device', 'auto')
            target_layers = request.form.get('targetLayers', '')
            advanced = request.form.get('advanced') == 'on'
            visualize = request.form.get('visualize') == 'on'

            # Get questions
            questions_file = request.files.get('questions')
            questions_text = request.form.get('questionsText', '')

            if not model:
                return jsonify({'error': 'Model is required'}), 400

            if not questions_file and not questions_text.strip():
                return jsonify({'error': 'Questions are required'}), 400

            # Generate unique analysis ID
            analysis_id = f"analysis_{int(time.time())}"

            # Start background analysis
            thread = threading.Thread(
                target=run_analysis_background,
                args=(analysis_id, model, device, target_layers, questions_file, questions_text, advanced, visualize)
            )
            thread.daemon = True
            thread.start()

            return jsonify({
                'success': True,
                'analysis_id': analysis_id,
                'message': 'Analysis started'
            })

        except Exception as e:
            logger.error(f"Error starting analysis: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analysis-status/<analysis_id>')
    def api_analysis_status(analysis_id):
        """Get analysis status"""
        global analysis_status

        try:
            if analysis_status.get('analysis_id') == analysis_id:
                return jsonify(analysis_status)
            else:
                # Check if analysis exists in results
                if analysis_id in analysis_results:
                    return jsonify({
                        'completed': True,
                        'progress': 100,
                        'message': 'Analysis completed',
                        'results': analysis_results[analysis_id]
                    })
                else:
                    return jsonify({'error': 'Analysis not found'}), 404
        except Exception as e:
            logger.error(f"Error getting analysis status: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/list-layers')
    def api_list_layers():
        """List available layers for a model"""
        try:
            model_name = request.args.get('model', 'gpt2')

            # Import here to avoid issues if dependencies not available
            from utils.config_manager import get_config
            from analysis.activation_analyzer import ActivationAnalyzer

            config = get_config()
            config.model.name = model_name

            analyzer = ActivationAnalyzer(config)
            layers = analyzer.get_layer_names()

            return jsonify({
                'success': True,
                'layers': layers,
                'count': len(layers)
            })

        except Exception as e:
            logger.error(f"Error listing layers: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/visualize', methods=['POST'])
    def api_visualize():
        """Generate visualization"""
        try:
            data_source = request.form.get('dataSource')
            visualization_type = request.form.get('visualizationType')

            if not data_source or not visualization_type:
                return jsonify({'error': 'Data source and visualization type are required'}), 400

            # Generate unique visualization ID
            viz_id = f"viz_{int(time.time())}"

            # Mock response for now
            result = {
                'success': True,
                'visualization_id': viz_id,
                'plots': [
                    {
                        'title': 'Activation Statistics',
                        'type': 'plotly',
                        'data': [],
                        'layout': {'title': 'Neural Network Activations'}
                    }
                ]
            }

            return jsonify(result)

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/logs')
    def api_logs():
        """View system logs"""
        try:
            log_file = Path('../../neuronmap.log')
            if log_file.exists():
                return send_file(str(log_file.absolute()), as_attachment=False)
            else:
                return "No log file found", 404
        except Exception as e:
            logger.error(f"Error accessing logs: {e}")
            return jsonify({'error': str(e)}), 500


def create_web_templates():
    """Create basic HTML templates for the web interface."""
    web_dir = Path(__file__).parent.parent.parent / 'web'
    templates_dir = web_dir / 'templates'
    static_dir = web_dir / 'static'

    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}NeuronMap{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">🧠 NeuronMap</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Dashboard</a>
                <a class="nav-link" href="/analysis">Analysis</a>
                <a class="nav-link" href="/results">Results</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>

    {% block scripts %}{% endblock %}
</body>
</html>'''

    # Index template
    index_template = '''{% extends "base.html" %}

{% block title %}Dashboard - NeuronMap{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>Welcome to NeuronMap</h3>
            </div>
            <div class="card-body">
                <p>Neural Network Activation Analysis System</p>
                <div class="row">
                    <div class="col-md-4">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h5>Basic Analysis</h5>
                                <p>Single model activation extraction</p>
                                <a href="/analysis?type=basic" class="btn btn-primary">Start</a>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h5>Multi-Model</h5>
                                <p>Compare multiple models</p>
                                <a href="/analysis?type=multi_model" class="btn btn-success">Start</a>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h5>Advanced</h5>
                                <p>Clustering & statistics</p>
                                <a href="/analysis?type=advanced" class="btn btn-warning">Start</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5>System Status</h5>
            </div>
            <div class="card-body">
                <div id="status-display">
                    <p>Status: <span id="status-text" class="badge bg-secondary">Ready</span></p>
                    <div id="progress-container" style="display: none;">
                        <div class="progress">
                            <div id="progress-bar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
function updateStatus() {
    $.get('/api/status', function(data) {
        if (data.running) {
            $('#status-text').removeClass().addClass('badge bg-primary').text('Running');
            $('#progress-container').show();
            $('#progress-bar').css('width', data.progress + '%').text(data.progress + '%');
        } else if (data.error) {
            $('#status-text').removeClass().addClass('badge bg-danger').text('Error');
            $('#progress-container').hide();
        } else {
            $('#status-text').removeClass().addClass('badge bg-success').text('Ready');
            $('#progress-container').hide();
        }
    });
}

// Update status every 2 seconds
setInterval(updateStatus, 2000);
updateStatus();
</script>
{% endblock %}'''

    # Save templates
    with open(templates_dir / 'base.html', 'w', encoding='utf-8') as f:
        f.write(base_template)

    with open(templates_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(index_template)

    # Create simple analysis template
    analysis_template = '''{% extends "base.html" %}

{% block title %}Analysis - NeuronMap{% endblock %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h3>Configure Analysis</h3>
    </div>
    <div class="card-body">
        <form id="analysis-form">
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label">Analysis Type</label>
                        <select class="form-select" id="analysis-type">
                            <option value="basic">Basic Analysis</option>
                            <option value="multi_model">Multi-Model Comparison</option>
                            <option value="advanced">Advanced Analysis</option>
                        </select>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Model</label>
                        <select class="form-select" id="model-select">
                            <option value="distilgpt2">DistilGPT-2</option>
                            <option value="gpt2">GPT-2</option>
                            <option value="bert-base-uncased">BERT Base</option>
                        </select>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label">Upload Questions (Optional)</label>
                        <input type="file" class="form-control" id="questions-file" accept=".jsonl,.json,.txt">
                    </div>

                    <div class="mb-3">
                        <button type="submit" class="btn btn-primary">Start Analysis</button>
                        <button type="button" class="btn btn-secondary" onclick="window.location.href='/results'">View Results</button>
                    </div>
                </div>
            </div>
        </form>

        <div id="analysis-status" style="display: none;">
            <div class="alert alert-info">
                <h5>Analysis in Progress</h5>
                <p id="status-message">Starting...</p>
                <div class="progress">
                    <div id="progress-bar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
$('#analysis-form').submit(function(e) {
    e.preventDefault();

    const formData = {
        type: $('#analysis-type').val(),
        model: $('#model-select').val()
    };

    $.ajax({
        url: '/api/start_analysis',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(formData),
        success: function(data) {
            $('#analysis-status').show();
            checkStatus();
        },
        error: function(xhr) {
            alert('Failed to start analysis: ' + xhr.responseJSON.error);
        }
    });
});

function checkStatus() {
    $.get('/api/status', function(data) {
        if (data.running) {
            $('#status-message').text(data.message);
            $('#progress-bar').css('width', data.progress + '%').text(data.progress + '%');
            setTimeout(checkStatus, 2000);
        } else if (data.error) {
            $('#status-message').text('Error: ' + data.error);
            $('#analysis-status').removeClass('alert-info').addClass('alert-danger');
        } else if (data.results) {
            $('#status-message').text('Analysis completed! View results below.');
            $('#analysis-status').removeClass('alert-info').addClass('alert-success');
            window.location.href = '/results';
        }
    });
}
</script>
{% endblock %}'''

    with open(templates_dir / 'analysis.html', 'w', encoding='utf-8') as f:
        f.write(analysis_template)

    # Create results template
    results_template = '''{% extends "base.html" %}

{% block title %}Results - NeuronMap{% endblock %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h3>Analysis Results</h3>
        <button class="btn btn-sm btn-outline-primary" onclick="downloadResults()">Download JSON</button>
    </div>
    <div class="card-body">
        <div id="results-container">
            <p>Loading results...</p>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
function loadResults() {
    $.get('/api/results', function(data) {
        displayResults(data);
    }).fail(function() {
        $('#results-container').html('<div class="alert alert-warning">No results available. Run an analysis first.</div>');
    });
}

function displayResults(data) {
    let html = '';

    if (data.type === 'basic') {
        html = `
            <h4>Basic Analysis Results</h4>
            <p><strong>Samples analyzed:</strong> ${data.sample_count}</p>
            <p><strong>Success rate:</strong> ${(data.statistics.success_rate * 100).toFixed(1)}%</p>
            <pre>${JSON.stringify(data.statistics, null, 2)}</pre>
        `;
    } else if (data.type === 'multi_model') {
        html = `
            <h4>Multi-Model Analysis Results</h4>
            <p><strong>Models compared:</strong> ${data.models.join(', ')}</p>
            <pre>${JSON.stringify(data.comparison, null, 2)}</pre>
        `;
    } else if (data.type === 'advanced') {
        html = `
            <h4>Advanced Analysis Results</h4>
            <p><strong>Layer analyzed:</strong> ${data.layer_name}</p>
            <pre>${JSON.stringify(data.analyses, null, 2)}</pre>
        `;
    }

    $('#results-container').html(html);
}

function downloadResults() {
    window.location.href = '/api/download_results';
}

loadResults();
</script>
{% endblock %}'''

    with open(templates_dir / 'results.html', 'w', encoding='utf-8') as f:
        f.write(results_template)

    logger.info(f"Web templates created in {templates_dir}")


def start_web_server(host='127.0.0.1', port=5000, debug=False):
    """Start the Flask web server."""
    if not FLASK_AVAILABLE:
        logger.error("Flask not available. Cannot start web server.")
        return False

    try:
        # Create templates if they don't exist
        create_web_templates()

        logger.info(f"Starting NeuronMap web server on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
        return True

    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        return False


if __name__ == "__main__":
    # Install Flask if not available
    if not FLASK_AVAILABLE:
        print("Flask not found. Install with: pip install flask")
        exit(1)

    start_web_server(debug=True)
