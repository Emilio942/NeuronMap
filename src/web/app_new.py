"""
NeuronMap Web Interface
======================

Professional Flask-based web interface for neural network activation analysis.
"""

import os
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Try to import Flask and dependencies
try:
    from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available. Web interface will be disabled.")

# Try to import NeuronMap components
try:
    from utils.config_manager import ConfigManager
    from analysis.activation_analyzer import ActivationAnalyzer
    from analysis.multi_model_analyzer import MultiModelAnalyzer
    from analysis.advanced_analyzer import AdvancedAnalyzer
    from visualization.activation_visualizer import ActivationVisualizer
    from data_processing.question_loader import QuestionLoader
    NEURONMAP_AVAILABLE = True
except ImportError as e:
    NEURONMAP_AVAILABLE = False
    logger.warning(f"NeuronMap components not available: {e}")

if FLASK_AVAILABLE:
    app = Flask(__name__,
                template_folder='../../web/templates',
                static_folder='../../web/static')
    app.secret_key = 'neuronmap_secret_key_change_in_production'

    # Global state management
    analysis_jobs = {}
    visualization_jobs = {}
    system_stats = {
        'total_analyses': 0,
        'models_analyzed': 0,
        'layers_processed': 0,
        'visualizations_created': 0
    }
    recent_activities = []

    def add_activity(message: str, icon: str = "chart-line", color: str = "primary"):
        """Add activity to recent activities list."""
        activity = {
            'message': message,
            'icon': icon,
            'color': color,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        recent_activities.insert(0, activity)
        if len(recent_activities) > 20:
            recent_activities.pop()

    # ============================================================================
    # MAIN PAGE ROUTES
    # ============================================================================

    @app.route('/')
    def index():
        """Home page."""
        return render_template('index.html')

    @app.route('/analysis')
    def analysis():
        """Analysis page."""
        return render_template('analysis.html')

    @app.route('/visualization')
    def visualization():
        """Visualization page."""
        return render_template('visualization.html')

    @app.route('/multi-model')
    def multi_model():
        """Multi-model comparison page."""
        return render_template('multi_model.html')

    @app.route('/results')
    def results():
        """Results browser page."""
        return render_template('results.html')

    # ============================================================================
    # API ROUTES - SYSTEM INFO
    # ============================================================================

    @app.route('/api/stats')
    def api_stats():
        """Get system statistics."""
        try:
            # Update stats from actual data
            data_outputs = Path('data/outputs')
            if data_outputs.exists():
                # Count analysis results
                analysis_dirs = list(data_outputs.glob('**/activations_*.csv'))
                system_stats['total_analyses'] = len(analysis_dirs)

                # Count visualizations
                viz_dirs = list(data_outputs.glob('**/visualizations'))
                system_stats['visualizations_created'] = sum(
                    len(list(viz_dir.glob('*.png'))) for viz_dir in viz_dirs if viz_dir.is_dir()
                )

                # Estimate layers and models
                system_stats['layers_processed'] = system_stats['total_analyses'] * 12  # Avg layers
                system_stats['models_analyzed'] = min(system_stats['total_analyses'], 10)  # Unique models

            return jsonify(system_stats)
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return jsonify(system_stats)

    @app.route('/api/recent-activity')
    def api_recent_activity():
        """Get recent system activity."""
        return jsonify({'activities': recent_activities})

    @app.route('/api/logs')
    def api_logs():
        """View system logs."""
        try:
            log_file = Path('neuronmap.log')
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = f.readlines()[-100:]  # Last 100 lines
                return '<pre>' + ''.join(logs) + '</pre>'
            else:
                return '<pre>No logs available</pre>'
        except Exception as e:
            return f'<pre>Error reading logs: {e}</pre>'

    # ============================================================================
    # API ROUTES - ANALYSIS
    # ============================================================================

    @app.route('/api/analyze', methods=['POST'])
    def api_analyze():
        """Start a new analysis."""
        if not NEURONMAP_AVAILABLE:
            return jsonify({'error': 'NeuronMap components not available'}), 500

        try:
            # Generate unique analysis ID
            analysis_id = str(uuid.uuid4())

            # Get parameters
            model_name = request.form.get('model')
            if model_name == 'custom':
                model_name = request.form.get('customModel')

            device = request.form.get('device', 'auto')
            target_layers = request.form.get('targetLayers', '')
            advanced = request.form.get('advanced') == 'on'
            visualize = request.form.get('visualize') == 'on'

            # Get questions
            questions = []
            questions_file = request.files.get('questions')
            questions_text = request.form.get('questionsText', '').strip()

            if questions_file and questions_file.filename:
                # Handle file upload
                questions = handle_questions_file(questions_file)
            elif questions_text:
                # Handle manual text input
                questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            else:
                return jsonify({'error': 'No questions provided'}), 400

            if not questions:
                return jsonify({'error': 'No valid questions found'}), 400

            # Validate model
            if not model_name:
                return jsonify({'error': 'Model name required'}), 400

            # Create analysis job
            analysis_jobs[analysis_id] = {
                'id': analysis_id,
                'status': 'starting',
                'progress': 0,
                'message': 'Initializing analysis...',
                'model': model_name,
                'questions': questions,
                'device': device,
                'target_layers': target_layers,
                'advanced': advanced,
                'visualize': visualize,
                'start_time': datetime.now(),
                'results': None,
                'error': None
            }

            # Start analysis in background
            thread = threading.Thread(
                target=run_analysis_background,
                args=(analysis_id,),
                daemon=True
            )
            thread.start()

            add_activity(f"Started analysis with {model_name}", "play", "success")

            return jsonify({
                'analysis_id': analysis_id,
                'message': 'Analysis started successfully'
            })

        except Exception as e:
            logger.error(f"Error starting analysis: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analysis-status/<analysis_id>')
    def api_analysis_status(analysis_id):
        """Get analysis status."""
        if analysis_id not in analysis_jobs:
            return jsonify({'error': 'Analysis not found'}), 404

        job = analysis_jobs[analysis_id]

        return jsonify({
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message'],
            'completed': job['status'] in ['completed', 'failed'],
            'error': job['error'],
            'results': job['results'] if job['status'] == 'completed' else None
        })

    @app.route('/api/cancel-analysis/<analysis_id>', methods=['POST'])
    def api_cancel_analysis(analysis_id):
        """Cancel an analysis."""
        if analysis_id in analysis_jobs:
            analysis_jobs[analysis_id]['status'] = 'cancelled'
            add_activity(f"Cancelled analysis {analysis_id[:8]}", "stop", "warning")

        return jsonify({'message': 'Analysis cancelled'})

    @app.route('/api/list-layers')
    def api_list_layers():
        """List available layers for a model."""
        if not NEURONMAP_AVAILABLE:
            return jsonify({'error': 'NeuronMap components not available'}), 500

        model_name = request.args.get('model')
        if not model_name:
            return jsonify({'error': 'Model name required'}), 400

        try:
            # Use a simple approach to get layers
            config = ConfigManager()
            analyzer = ActivationAnalyzer(config)

            # Load model and get layer names
            layers = analyzer.list_available_layers(model_name)

            return jsonify({'layers': layers})

        except Exception as e:
            logger.error(f"Error listing layers: {e}")
            return jsonify({'error': str(e)}), 500

    # ============================================================================
    # API ROUTES - VISUALIZATION
    # ============================================================================

    @app.route('/api/visualize', methods=['POST'])
    def api_visualize():
        """Generate visualizations."""
        if not NEURONMAP_AVAILABLE:
            return jsonify({'error': 'NeuronMap components not available'}), 500

        try:
            visualization_id = str(uuid.uuid4())

            # Get parameters
            data_source = request.form.get('dataSource')
            visualization_type = request.form.get('visualizationType')

            # Create visualization job
            visualization_jobs[visualization_id] = {
                'id': visualization_id,
                'status': 'starting',
                'progress': 0,
                'data_source': data_source,
                'visualization_type': visualization_type
            }

            # Start visualization in background
            thread = threading.Thread(
                target=run_visualization_background,
                args=(visualization_id,),
                daemon=True
            )
            thread.start()

            add_activity(f"Started {visualization_type} visualization", "chart-bar", "info")

            return jsonify({
                'visualization_id': visualization_id,
                'message': 'Visualization started'
            })

        except Exception as e:
            logger.error(f"Error starting visualization: {e}")
            return jsonify({'error': str(e)}), 500

    # ============================================================================
    # HELPER FUNCTIONS
    # ============================================================================

    def handle_questions_file(file):
        """Handle uploaded questions file."""
        try:
            if not NEURONMAP_AVAILABLE:
                # Simple text processing
                content = file.read().decode('utf-8')
                if file.filename.endswith('.txt'):
                    return [line.strip() for line in content.split('\n') if line.strip()]
                elif file.filename.endswith('.json'):
                    data = json.loads(content)
                    if isinstance(data, list):
                        return [str(item) for item in data]
                    elif isinstance(data, dict) and 'questions' in data:
                        return [str(q) for q in data['questions']]
                return []
            else:
                # Use QuestionLoader
                loader = QuestionLoader()
                temp_path = Path(f'/tmp/{file.filename}')
                file.save(temp_path)
                questions = loader.load_questions(temp_path)
                temp_path.unlink()  # Clean up
                return questions
        except Exception as e:
            logger.error(f"Error processing questions file: {e}")
            return []

    def run_analysis_background(analysis_id: str):
        """Run analysis in background thread."""
        job = analysis_jobs[analysis_id]

        try:
            job['status'] = 'running'
            job['progress'] = 10
            job['message'] = 'Loading model...'

            if not NEURONMAP_AVAILABLE:
                # Simulate analysis for demo
                simulate_analysis(job)
                return

            # Real analysis
            config = ConfigManager()
            analyzer = ActivationAnalyzer(config)

            job['progress'] = 30
            job['message'] = 'Extracting activations...'

            # Run analysis
            results = analyzer.analyze_activations(
                model_name=job['model'],
                questions=job['questions'],
                device=job['device'],
                target_layers=job['target_layers'].split(',') if job['target_layers'] else None
            )

            job['progress'] = 80
            job['message'] = 'Processing results...'

            # Generate visualizations if requested
            if job['visualize']:
                job['message'] = 'Generating visualizations...'
                visualizer = ActivationVisualizer(config)
                viz_results = visualizer.create_visualizations(results)
                results['visualizations'] = viz_results

            job['progress'] = 100
            job['status'] = 'completed'
            job['message'] = 'Analysis completed successfully'
            job['results'] = {
                'model': job['model'],
                'question_count': len(job['questions']),
                'layer_count': len(results.get('layers', [])),
                'duration': str(datetime.now() - job['start_time']),
                'output_file': results.get('output_file', ''),
                'visualizations': results.get('visualizations', [])
            }

            # Update stats
            system_stats['total_analyses'] += 1
            add_activity(f"Completed analysis for {job['model']}", "check", "success")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            job['status'] = 'failed'
            job['error'] = str(e)
            job['message'] = f'Analysis failed: {e}'
            add_activity(f"Analysis failed: {str(e)[:50]}", "exclamation-triangle", "danger")

    def simulate_analysis(job):
        """Simulate analysis for demo purposes."""
        import time
        import random

        steps = [
            (20, "Loading model configuration..."),
            (40, "Initializing model weights..."),
            (60, "Processing questions..."),
            (80, "Extracting activations..."),
            (90, "Calculating statistics..."),
            (100, "Saving results...")
        ]

        for progress, message in steps:
            job['progress'] = progress
            job['message'] = message
            time.sleep(random.uniform(0.5, 1.5))

        job['status'] = 'completed'
        job['results'] = {
            'model': job['model'],
            'question_count': len(job['questions']),
            'layer_count': 12,
            'duration': '45 seconds',
            'output_file': 'demo_results.csv'
        }

    def run_visualization_background(visualization_id: str):
        """Run visualization in background thread."""
        job = visualization_jobs[visualization_id]

        try:
            job['status'] = 'running'
            job['progress'] = 20

            # Simulate visualization
            time.sleep(2)
            job['progress'] = 100
            job['status'] = 'completed'

            system_stats['visualizations_created'] += 1
            add_activity(f"Generated {job['visualization_type']} plot", "image", "info")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            job['status'] = 'failed'
            job['error'] = str(e)

    # ============================================================================
    # ERROR HANDLERS
    # ============================================================================

    @app.errorhandler(404)
    def not_found(error):
        return render_template('base.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('base.html'), 500

else:
    # Flask not available
    app = None
    FLASK_AVAILABLE = False

# Export for use in other modules
__all__ = ['app', 'FLASK_AVAILABLE']
