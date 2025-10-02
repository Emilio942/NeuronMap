#!/usr/bin/env python3
"""
Standalone Model Surgery Server
Simple server for testing Model Surgery functionality without full NeuronMap dependencies
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template_string, render_template

# Import the interventions API
from src.web.api.interventions import interventions_bp

# Import the circuits API
from src.web.api.circuits import circuits_bp

# Import the SAE API
from src.web.api.sae import sae_blueprint

# Create Flask app
app = Flask(__name__,
            template_folder='web/templates',
            static_folder='web/static')
app.secret_key = 'neuronmap_test_key'

# Register the interventions API blueprint
app.register_blueprint(interventions_bp)
# Register the circuits API blueprint
app.register_blueprint(circuits_bp)
# Register the SAE API blueprint
app.register_blueprint(sae_blueprint)

# Simple standalone template that doesn't need base.html
STANDALONE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Surgery - NeuronMap</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</head>
<body>
    <div class="container-fluid mt-3">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4">
                    <i class="fas fa-brain me-2"></i>Interactive Model Surgery
                </h1>
                <p class="text-muted">Explore neural networks through interactive interventions and real-time analysis.</p>
                <div class="alert alert-info">
                    <strong>Status:</strong> Standalone test server running. 
                    API endpoints available at <code>/api/interventions/*</code>, <code>/api/circuits/*</code>, and <code>/api/sae/*</code>
                </div>
                
                <!-- Navigation -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="btn-group" role="group">
                            <a href="/" class="btn btn-primary">
                                <i class="fas fa-scissors me-1"></i>Model Surgery
                            </a>
                            <a href="/circuits" class="btn btn-outline-primary">
                                <i class="fas fa-project-diagram me-1"></i>Circuit Explorer
                            </a>
                            <a href="/sae" class="btn btn-outline-primary">
                                <i class="fas fa-brain me-1"></i>SAE Explorer
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-cog me-2"></i>Quick Tests</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-grid gap-2">
                            <button class="btn btn-primary" onclick="testModels()">
                                <i class="fas fa-list me-1"></i>Test Models API
                            </button>
                            <button class="btn btn-success" onclick="testActivations()">
                                <i class="fas fa-brain me-1"></i>Test Activations API
                            </button>
                            <button class="btn btn-warning" onclick="testAblation()">
                                <i class="fas fa-scissors me-1"></i>Test Ablation API
                            </button>
                        </div>
                        <div class="mt-3">
                            <label for="testPrompt" class="form-label">Test Prompt:</label>
                            <input type="text" class="form-control" id="testPrompt" value="The capital of France is">
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-terminal me-2"></i>Test Results</h5>
                    </div>
                    <div class="card-body">
                        <pre id="results" class="bg-dark text-light p-3" style="height: 400px; overflow-y: auto;">
Ready for testing...
                        </pre>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function log(message) {
            const results = document.getElementById('results');
            const timestamp = new Date().toLocaleTimeString();
            results.textContent += `[${timestamp}] ${message}\\n`;
            results.scrollTop = results.scrollHeight;
        }

        async function testModels() {
            log('Testing Models API...');
            try {
                const response = await fetch('/api/interventions/models');
                const data = await response.json();
                log(`✅ Models API: ${data.success ? 'SUCCESS' : 'FAILED'}`);
                log(`Found ${data.models?.length || 0} models`);
                if (data.models) {
                    data.models.slice(0, 3).forEach(model => {
                        log(`  - ${model.name} (${model.type})`);
                    });
                }
            } catch (error) {
                log(`❌ Models API Error: ${error.message}`);
            }
        }

        async function testActivations() {
            log('Testing Activations API...');
            const prompt = document.getElementById('testPrompt').value;
            try {
                const response = await fetch('/api/interventions/activations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: 'gpt2', prompt: prompt })
                });
                const data = await response.json();
                log(`✅ Activations API: ${data.success ? 'SUCCESS' : 'FAILED'}`);
                if (data.success) {
                    log(`Generated heatmap: ${data.shape?.layers}x${data.shape?.neurons}`);
                    log(`Layers: ${data.layer_names?.length || 0}`);
                } else {
                    log(`Error: ${data.error}`);
                }
            } catch (error) {
                log(`❌ Activations API Error: ${error.message}`);
            }
        }

        async function testAblation() {
            log('Testing Ablation API...');
            const prompt = document.getElementById('testPrompt').value;
            try {
                const response = await fetch('/api/interventions/ablate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'gpt2',
                        prompt: prompt,
                        layer: 'transformer.h.6.mlp',
                        neurons: [100, 150]
                    })
                });
                const data = await response.json();
                log(`✅ Ablation API: ${data.success ? 'SUCCESS' : 'FAILED'}`);
                if (data.success) {
                    log(`Baseline: "${data.results?.baseline_output}"`);
                    log(`Ablated: "${data.results?.ablated_output}"`);
                    log(`Effect: ${data.results?.effect_size?.toFixed(3)}`);
                } else {
                    log(`Error: ${data.error}`);
                }
            } catch (error) {
                log(`❌ Ablation API Error: ${error.message}`);
            }
        }

        // Auto-test on load
        window.onload = function() {
            log('🚀 Model Surgery Test Server Ready');
            log('Click buttons above to test API functionality');
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Standalone test page."""
    return render_template_string(STANDALONE_TEMPLATE)

@app.route('/standalone')
def standalone():
    """Standalone test page."""
    return render_template_string(STANDALONE_TEMPLATE)

from flask import jsonify

@app.route('/circuits')
def circuits():
    """Circuit exploration page (returns JSON for now)."""
    # This is a placeholder. In a real application, you would fetch circuit data here.
    dummy_circuit_data = {
        "success": True,
        "message": "Circuit data will be available here.",
        "circuits": [
            {"id": "circuit_1", "name": "Induction Head 1", "nodes": 5, "edges": 4},
            {"id": "circuit_2", "name": "Copying Head 2", "nodes": 3, "edges": 2}
        ]
    }
    return jsonify(dummy_circuit_data)

@app.route('/sae')
def sae_explorer():
    """SAE exploration page."""
    return render_template('sae_explorer.html')

if __name__ == '__main__':
    print("Starting Standalone Model Surgery Test Server...")
    print("Visit: http://localhost:5002")
    print("Standalone interface: http://localhost:5002/standalone")
    app.run(debug=True, port=5002, host='0.0.0.0')
