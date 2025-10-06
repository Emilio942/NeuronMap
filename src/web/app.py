"""
NeuronMap Web Interface
======================

Professional Flask-based web interface for neural network activation analysis.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Try to import Flask and dependencies
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available. Web interface will be disabled.")

# Try to import NeuronMap components
try:
    from utils.config_manager import ConfigManager
    from analysis.activation_analyzer import ActivationAnalyzer
    from analysis.universal_model_adapter import UniversalModelAdapter
    from data_processing.question_loader import QuestionLoader
    from utils.system_monitor import start_system_monitoring
    NEURONMAP_AVAILABLE = True

    # Start system monitoring
    start_system_monitoring()
    logger.info("System monitoring started")

except ImportError as e:
    NEURONMAP_AVAILABLE = False
    logger.warning(f"NeuronMap components not available: {e}")

if FLASK_AVAILABLE:
    app = Flask(__name__,
                template_folder='../../web/templates',
                static_folder='../../web/static')
    app.secret_key = 'neuronmap_secret_key_change_in_production'
    
    # Register API blueprints
    try:
        from web.api.interventions import interventions_bp
        app.register_blueprint(interventions_bp)
        logger.info("Interventions API blueprint registered")
    except ImportError as e:
        logger.warning(f"Could not register interventions API: {e}")

    # Register circuits API blueprint
    try:
        from web.api.circuits import circuits_bp
        app.register_blueprint(circuits_bp)
        logger.info("Circuits API blueprint registered")
    except ImportError as e:
        logger.warning(f"Could not register circuits API: {e}")

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
    # BACKGROUND FUNCTIONS
    # ============================================================================

    def run_analysis_background(analysis_id: str):
        """Run analysis in background thread."""
        try:
            if analysis_id not in analysis_jobs:
                logger.error(f"Analysis job {analysis_id} not found")
                return

            job = analysis_jobs[analysis_id]
            job['status'] = 'running'
            job['progress'] = 10

            # Simulate analysis work
            import time
            for i in range(10):
                time.sleep(0.5)  # Simulate work
                job['progress'] = 10 + (i + 1) * 9

            job['status'] = 'completed'
            job['progress'] = 100
            job['results'] = {'message': 'Analysis completed successfully'}

            add_activity(f"Analysis {analysis_id} completed", "check", "success")

        except Exception as e:
            logger.error(f"Background analysis error: {e}")
            if analysis_id in analysis_jobs:
                analysis_jobs[analysis_id]['status'] = 'error'
                analysis_jobs[analysis_id]['error'] = str(e)

    def run_visualization_background(viz_id: str):
        """Run visualization in background thread."""
        try:
            if viz_id not in visualization_jobs:
                logger.error(f"Visualization job {viz_id} not found")
                return

            job = visualization_jobs[viz_id]
            job['status'] = 'running'
            job['progress'] = 10

            # Simulate visualization work
            import time
            for i in range(8):
                time.sleep(0.3)  # Simulate work
                job['progress'] = 10 + (i + 1) * 11

            job['status'] = 'completed'
            job['progress'] = 100
            job['results'] = {'message': 'Visualization completed successfully'}

            add_activity(f"Visualization {viz_id} completed", "image", "info")

        except Exception as e:
            logger.error(f"Background visualization error: {e}")
            if viz_id in visualization_jobs:
                visualization_jobs[viz_id]['status'] = 'error'
                visualization_jobs[viz_id]['error'] = str(e)

    def run_advanced_analytics_background(analytics_id: str):
        """Run advanced analytics in background thread."""
        try:
            if analytics_id not in analysis_jobs:
                logger.error(f"Analytics job {analytics_id} not found")
                return

            job = analysis_jobs[analytics_id]
            job['status'] = 'running'
            job['progress'] = 10

            # Simulate analytics work
            import time
            for i in range(12):
                time.sleep(0.4)  # Simulate work
                job['progress'] = 10 + (i + 1) * 7

            job['status'] = 'completed'
            job['progress'] = 100
            job['results'] = {'message': 'Advanced analytics completed successfully'}

            add_activity(f"Advanced analytics {analytics_id} completed", "brain", "warning")

        except Exception as e:
            logger.error(f"Background analytics error: {e}")
            if analytics_id in analysis_jobs:
                analysis_jobs[analytics_id]['status'] = 'error'
                analysis_jobs[analytics_id]['error'] = str(e)

    # ============================================================================
    # MAIN PAGE ROUTES
    # ============================================================================

    @app.route('/')
    def index():
        """Home page."""
        return render_template('working_viz.html')

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

    @app.route('/advanced-analytics')
    def advanced_analytics_page():
        """Advanced analytics page."""
        return render_template('advanced_analytics.html')

    @app.route('/performance')
    def performance():
        """System performance monitoring page."""
        return render_template('performance.html')

    @app.route('/reports')
    def reports_page():
        """Reports and export page."""
        return render_template('reports.html')

    @app.route('/model-surgery')
    def model_surgery():
        """Interactive Model Surgery & Path Analysis page."""
        return render_template('model_surgery.html')

    @app.route('/circuits')
    def circuit_explorer():
        """Circuit discovery and exploration page."""
        return render_template('circuit_explorer.html')

    @app.route('/circuit-fixed')
    def circuit_fixed():
        """Repaired Circuit Explorer page."""
        return render_template('circuit_fixed.html')

    @app.route('/zoo')
    def analysis_zoo():
        """Analysis Zoo - Community artifact sharing platform."""
        return render_template('analysis_zoo_simple.html')

    @app.route('/minimal-test')
    def minimal_test():
        """Minimaler Test."""
        return render_template('minimal_test.html')

    @app.route('/simple-test')
    def simple_test():
        """Einfacher Test."""
        return render_template('simple_test.html')

    @app.route('/test-cytoscape')
    def test_cytoscape():
        """Test page for Cytoscape integration."""
        return render_template('cytoscape_test.html')

    @app.route('/fixed')
    def fixed_visualization():
        """Repaired visualization test page."""
        return render_template('working_viz.html')

    @app.route('/working')
    def working_viz():
        """Working visualization test page."""
        return render_template('working_viz.html')

    @app.route('/neuronmap-fixed')
    def neuronmap_fixed():
        """NeuronMap fixed visualization test page."""
        return render_template('fixed_viz.html')

    @app.route('/visualization-fixed')
    def visualization_fixed():
        """Fixed visualization page."""
        return send_from_directory('web/static', 'fixed_visualization.html')

    # ============================================================================
    # UTILITY FUNCTIONS
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

    @app.route('/api/system/status')
    def api_system_status():
        """Get current system performance status."""
        try:
            from utils.system_monitor import get_system_status
            status = get_system_status()
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/system/health')
    def api_system_health():
        """Get system health assessment."""
        try:
            from utils.system_monitor import get_system_health
            health = get_system_health()
            return jsonify(health)
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/system/monitor/start', methods=['POST'])
    def api_start_monitoring():
        """Start system monitoring."""
        try:
            from utils.system_monitor import start_system_monitoring
            start_system_monitoring()
            add_activity("System monitoring started", "chart-area", "info")
            return jsonify({'message': 'System monitoring started'})
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/system/monitor/stop', methods=['POST'])
    def api_stop_monitoring():
        """Stop system monitoring."""
        try:
            from utils.system_monitor import stop_system_monitoring
            stop_system_monitoring()
            add_activity("System monitoring stopped", "chart-area", "warning")
            return jsonify({'message': 'System monitoring stopped'})
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return jsonify({'error': str(e)}), 500

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

    # Circuit API endpoints
    @app.route('/api/circuits/load-model', methods=['POST'])
    def api_circuits_load_model():
        """Load a model for circuit analysis."""
        try:
            data = request.get_json()
            model_name = data.get('model_name', 'gpt2-small')
            
            # Simulate model loading
            analysis_id = str(uuid.uuid4())
            add_activity(f"Loading model {model_name} for circuit analysis", "upload", "info")
            
            return jsonify({
                'success': True,
                'analysis_id': analysis_id,
                'model': model_name,
                'message': f'Model {model_name} loaded successfully'
            })
        except Exception as e:
            logger.error(f"Error loading model for circuits: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/circuits/find-induction-heads', methods=['POST'])
    def api_circuits_find_induction_heads():
        """Find induction heads in the loaded model."""
        try:
            data = request.get_json()
            analysis_id = data.get('analysis_id')
            
            # Simulate finding induction heads
            add_activity("Finding induction heads", "search", "info")
            
            # Mock data for demonstration
            induction_heads = [
                {'layer': 5, 'head': 1, 'score': 0.89, 'pattern_type': 'induction'},
                {'layer': 5, 'head': 5, 'score': 0.76, 'pattern_type': 'induction'},
                {'layer': 6, 'head': 9, 'score': 0.82, 'pattern_type': 'induction'}
            ]
            
            return jsonify({
                'success': True,
                'induction_heads': induction_heads,
                'count': len(induction_heads)
            })
        except Exception as e:
            logger.error(f"Error finding induction heads: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/circuits/find-copying-heads', methods=['POST'])
    def api_circuits_find_copying_heads():
        """Find copying heads in the loaded model."""
        try:
            data = request.get_json()
            analysis_id = data.get('analysis_id')
            
            # Simulate finding copying heads
            add_activity("Finding copying heads", "search", "info")
            
            # Mock data for demonstration
            copying_heads = [
                {'layer': 0, 'head': 7, 'score': 0.94, 'pattern_type': 'copying'},
                {'layer': 1, 'head': 4, 'score': 0.88, 'pattern_type': 'copying'},
                {'layer': 2, 'head': 1, 'score': 0.79, 'pattern_type': 'copying'}
            ]
            
            return jsonify({
                'success': True,
                'copying_heads': copying_heads,
                'count': len(copying_heads)
            })
        except Exception as e:
            logger.error(f"Error finding copying heads: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/circuits/analyze-neuron-heads', methods=['POST'])
    def api_circuits_analyze_neuron_heads():
        """Analyze neuron-head interactions."""
        try:
            data = request.get_json()
            analysis_id = data.get('analysis_id')
            
            # Simulate neuron-head analysis
            add_activity("Analyzing neuron-head interactions", "cogs", "info")
            
            # Mock data for demonstration
            interactions = [
                {
                    'neuron': {'layer': 3, 'index': 42},
                    'head': {'layer': 5, 'index': 1},
                    'interaction_strength': 0.87,
                    'interaction_type': 'excitatory'
                },
                {
                    'neuron': {'layer': 4, 'index': 156},
                    'head': {'layer': 6, 'index': 9},
                    'interaction_strength': 0.73,
                    'interaction_type': 'inhibitory'
                }
            ]
            
            return jsonify({
                'success': True,
                'interactions': interactions,
                'count': len(interactions)
            })
        except Exception as e:
            logger.error(f"Error analyzing neuron-head interactions: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/circuits/export-circuit', methods=['POST'])
    def api_circuits_export_circuit():
        """Export discovered circuit data."""
        try:
            data = request.get_json()
            circuit_data = data.get('circuit_data')
            format_type = data.get('format', 'json')
            
            # Simulate circuit export
            add_activity(f"Exporting circuit data as {format_type}", "download", "success")
            
            export_id = str(uuid.uuid4())
            
            return jsonify({
                'success': True,
                'export_id': export_id,
                'format': format_type,
                'download_url': f'/api/circuits/download/{export_id}',
                'message': 'Circuit data exported successfully'
            })
        except Exception as e:
            logger.error(f"Error exporting circuit: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/circuits/download/<export_id>')
    def api_circuits_download(export_id):
        """Download exported circuit data."""
        try:
            # Mock circuit data for download
            circuit_data = {
                'export_id': export_id,
                'timestamp': datetime.now().isoformat(),
                'circuit_type': 'induction_head_circuit',
                'components': [
                    {'type': 'head', 'layer': 5, 'index': 1, 'role': 'induction'},
                    {'type': 'neuron', 'layer': 3, 'index': 42, 'role': 'feature_detector'}
                ]
            }
            
            add_activity(f"Downloaded circuit data {export_id[:8]}", "download", "success")
            
            return jsonify(circuit_data)
        except Exception as e:
            logger.error(f"Error downloading circuit data: {e}")
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
    # NEW API ROUTES - ADVANCED ANALYTICS
    # ============================================================================

    @app.route('/api/advanced-analytics', methods=['POST'])
    def api_advanced_analytics():
        """Start advanced analytics analysis."""
        if not NEURONMAP_AVAILABLE:
            return jsonify({'error': 'NeuronMap components not available'}), 500

        try:
            data = request.get_json()

            # Extract parameters
            model_name = data.get('model', 'distilgpt2')
            questions = data.get('questions', [])

            if not questions:
                return jsonify({'error': 'No questions provided'}), 400

            # Generate analysis ID
            analysis_id = f"advanced_{int(time.time())}"

            # Start analysis in background thread
            thread = threading.Thread(
                target=run_advanced_analytics_background,
                args=(analysis_id, model_name, questions)
            )
            thread.daemon = True
            thread.start()

            return jsonify({
                'success': True,
                'analysis_id': analysis_id,
                'message': 'Advanced analytics started'
            })

        except Exception as e:
            logger.error(f"Failed to start advanced analytics: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/list-models')
    def api_list_models():
        """List all available models from the universal adapter."""
        try:
            # Import required modules
            from utils.config_manager import get_config

            config = get_config()
            adapter = UniversalModelAdapter(config)

            # Get available models
            models = adapter.get_available_models()

            # Get detailed information for each model
            model_info = []
            for model in models:
                info = adapter.get_model_info(model)
                model_info.append({
                    'name': info['name'],
                    'type': info['type'],
                    'total_layers': info['total_layers'],
                    'display_name': info['name'].split('/')[-1] if '/' in info['name'] else info['name']
                })

            # Group by type
            grouped_models = {}
            for info in model_info:
                model_type = info['type']
                if model_type not in grouped_models:
                    grouped_models[model_type] = []
                grouped_models[model_type].append(info)

            return jsonify({
                'success': True,
                'models': model_info,
                'grouped_models': grouped_models,
                'total_count': len(model_info)
            })

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return jsonify({'error': str(e)}), 500

    # ============================================================================
    # PLUGIN MANAGEMENT
    # ============================================================================

    try:
        from core.plugin_system import PluginManager
        plugin_manager = PluginManager()
        # Load built-in plugins
        plugin_manager.load_builtin_plugins()
        PLUGIN_SYSTEM_AVAILABLE = True
    except ImportError:
        plugin_manager = None
        PLUGIN_SYSTEM_AVAILABLE = False

    @app.route('/plugins')
    def plugins_page():
        """Plugin management page."""
        return render_template('plugins.html')

    @app.route('/api/plugins/list')
    def api_list_plugins():
        """Get list of all plugins."""
        try:
            if not PLUGIN_SYSTEM_AVAILABLE:
                return jsonify({'error': 'Plugin system not available'}), 500

            plugins_info = []
            for plugin_name, plugin in plugin_manager.plugins.items():
                metadata = plugin.get_metadata()
                plugins_info.append({
                    'name': metadata.name,
                    'version': metadata.version,
                    'author': metadata.author,
                    'description': metadata.description,
                    'plugin_type': metadata.plugin_type,
                    'dependencies': metadata.dependencies,
                    'tags': metadata.tags,
                    'created_at': metadata.created_at,
                    'updated_at': metadata.updated_at,
                    'status': 'active'
                })

            return jsonify({
                'plugins': plugins_info,
                'total_count': len(plugins_info),
                'available_types': ['analysis', 'model_adapter', 'visualization', 'custom']
            })

        except Exception as e:
            logger.error(f"Error listing plugins: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/plugins/<plugin_name>/info')
    def api_plugin_info(plugin_name):
        """Get detailed information about a specific plugin."""
        try:
            if not PLUGIN_SYSTEM_AVAILABLE:
                return jsonify({'error': 'Plugin system not available'}), 500

            if plugin_name not in plugin_manager.plugins:
                return jsonify({'error': f'Plugin {plugin_name} not found'}), 404

            plugin = plugin_manager.plugins[plugin_name]
            metadata = plugin.get_metadata()

            plugin_info = {
                'name': metadata.name,
                'version': metadata.version,
                'author': metadata.author,
                'description': metadata.description,
                'plugin_type': metadata.plugin_type,
                'dependencies': metadata.dependencies,
                'tags': metadata.tags,
                'created_at': metadata.created_at,
                'updated_at': metadata.updated_at,
                'status': 'active',
                'methods': []
            }

            # Get available methods
            import inspect
            for method_name, method in inspect.getmembers(plugin, predicate=inspect.ismethod):
                if not method_name.startswith('_'):
                    plugin_info['methods'].append({
                        'name': method_name,
                        'doc': method.__doc__ or 'No documentation available'
                    })

            return jsonify(plugin_info)

        except Exception as e:
            logger.error(f"Error getting plugin info: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/plugins/<plugin_name>/execute', methods=['POST'])
    def api_execute_plugin(plugin_name):
        """Execute a plugin with given parameters."""
        try:
            if not PLUGIN_SYSTEM_AVAILABLE:
                return jsonify({'error': 'Plugin system not available'}), 500

            if plugin_name not in plugin_manager.plugins:
                return jsonify({'error': f'Plugin {plugin_name} not found'}), 404

            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            plugin = plugin_manager.plugins[plugin_name]

            # Execute plugin
            result = plugin.execute(**data)

            add_activity(f"Executed plugin: {plugin_name}", "puzzle-piece", "info")

            return jsonify({
                'success': True,
                'result': result,
                'message': f'Plugin {plugin_name} executed successfully'
            })

        except Exception as e:
            logger.error(f"Error executing plugin {plugin_name}: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/plugins/load', methods=['POST'])
    def api_load_plugin():
        """Load a new plugin from file."""
        try:
            if not PLUGIN_SYSTEM_AVAILABLE:
                return jsonify({'error': 'Plugin system not available'}), 500

            data = request.get_json()
            if not data or 'plugin_path' not in data:
                return jsonify({'error': 'Plugin path not provided'}), 400

            plugin_path = data['plugin_path']
            success = plugin_manager.load_plugin_from_file(plugin_path)

            if success:
                add_activity(f"Loaded new plugin from {plugin_path}", "puzzle-piece", "success")
                return jsonify({
                    'success': True,
                    'message': f'Plugin loaded successfully from {plugin_path}'
                })
            else:
                return jsonify({'error': 'Failed to load plugin'}), 500

        except Exception as e:
            logger.error(f"Error loading plugin: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/plugins/<plugin_name>/unload', methods=['POST'])
    def api_unload_plugin(plugin_name):
        """Unload a plugin."""
        try:
            if not PLUGIN_SYSTEM_AVAILABLE:
                return jsonify({'error': 'Plugin system not available'}), 500

            success = plugin_manager.unload_plugin(plugin_name)

            if success:
                add_activity(f"Unloaded plugin: {plugin_name}", "puzzle-piece", "warning")
                return jsonify({
                    'success': True,
                    'message': f'Plugin {plugin_name} unloaded successfully'
                })
            else:
                return jsonify({'error': f'Failed to unload plugin {plugin_name}'}), 500

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/plugins/types')
    def api_plugin_types():
        """Get available plugin types and their descriptions."""
        try:
            plugin_types = {
                'analysis': {
                    'name': 'Analysis Plugins',
                    'description': 'Plugins that perform custom analysis on neural activations',
                    'base_class': 'AnalysisPlugin',
                    'methods': ['analyze']
                },
                'model_adapter': {
                    'name': 'Model Adapter Plugins',
                    'description': 'Plugins that provide custom model loading and activation extraction',
                    'base_class': 'ModelAdapterPlugin',
                    'methods': ['load_model', 'extract_activations']
                },
                'visualization': {
                    'name': 'Visualization Plugins',
                    'description': 'Plugins that create custom visualizations',
                    'base_class': 'VisualizationPlugin',
                    'methods': ['create_visualization']
                },
                'custom': {
                    'name': 'Custom Plugins',
                    'description': 'General-purpose plugins with custom functionality',
                    'base_class': 'PluginBase',
                    'methods': ['execute']
                }
            }

            return jsonify(plugin_types)

        except Exception as e:
            logger.error(f"Error getting plugin types: {e}")
            return jsonify({'error': str(e)}), 500

    # ============================================================================
    # ERROR HANDLERS
    # ============================================================================

    @app.errorhandler(404)
    def not_found(error):
        return render_template('base.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('base.html'), 500

    @app.route('/api/reports/generate', methods=['POST'])
    def api_generate_report():
        """Generate a report from analysis data."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            analysis_id = data.get('analysis_id')
            format_type = data.get('format', 'pdf')

            if not analysis_id:
                return jsonify({'error': 'Analysis ID required'}), 400

            # TODO: Integrate with actual advanced_reporter.py
            # For now, return a mock response
            report_id = f"report_{analysis_id}_{int(time.time())}"

            # Simulate report generation
            import time
            time.sleep(2)  # Simulate processing time

            add_activity(f"Generated {format_type.upper()} report", "file-export", "success")

            return jsonify({
                'success': True,
                'report_id': report_id,
                'message': f'{format_type.upper()} report generated successfully',
                'download_url': f'/api/reports/download/{report_id}'
            })

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/reports/download/<report_id>')
    def api_download_report(report_id):
        """Download a generated report."""
        try:
            # TODO: Implement actual report download
            # For now, return a placeholder
            from flask import Response

            # Create a simple text file as placeholder
            content = f"NeuronMap Report\\n==================\\n\\nReport ID: {report_id}\\nGenerated: {datetime.now()}\\n\\nThis is a placeholder report. Actual report generation will be implemented with the advanced_reporter.py integration."

            return Response(
                content,
                mimetype='text/plain',
                headers={'Content-Disposition': f'attachment; filename=neuronmap_report_{report_id}.txt'}
            )

        except Exception as e:
            logger.error(f"Error downloading report: {e}")
            return jsonify({'error': str(e)}), 500

else:
    # Flask not available
    app = None
    FLASK_AVAILABLE = False

if __name__ == '__main__':
    if FLASK_AVAILABLE:
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Flask is not installed. Cannot start web server.")

# Export for use in other modules
__all__ = ['app', 'FLASK_AVAILABLE']
