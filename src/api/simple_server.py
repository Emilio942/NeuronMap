from flask import Flask, render_template_string

app = Flask(__name__)

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuronMap - WORKING VISUALIZATION</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .graph-container {
            width: 100%;
            height: 600px;
            border: 8px solid #007bff;
            background: #f8f9fa;
            position: relative;
            border-radius: 10px;
        }
        
        .controls {
            text-align: center;
            margin: 30px 0;
        }
        
        button {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            margin: 10px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        .status {
            background: #e9ecef;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-family: monospace;
        }
        
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 NeuronMap</h1>
            <h2>✅ WORKING VISUALIZATION</h2>
            <p>This is the FIXED version!</p>
        </div>
        
        <div class="controls">
            <button onclick="startDemo()">🚀 Start Demo</button>
            <button onclick="addNode()">➕ Add Node</button>
            <button onclick="clearGraph()">🗑️ Clear</button>
        </div>
        
        <div id="graph" class="graph-container"></div>
        
        <div id="status" class="status">Ready to start...</div>
    </div>

    <script>
        let cy = null;

        function updateStatus(message, type = 'info') {
            const status = document.getElementById('status');
            status.textContent = new Date().toLocaleTimeString() + ': ' + message;
            status.className = 'status ' + type;
            console.log(message);
        }

        function startDemo() {
            updateStatus('Starting demo...', 'info');
            
            const container = document.getElementById('graph');
            
            if (!container) {
                updateStatus('ERROR: Container not found!', 'error');
                return;
            }
            
            if (typeof cytoscape === 'undefined') {
                updateStatus('ERROR: Cytoscape not loaded!', 'error');
                container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: red; font-size: 24px;"><strong>CYTOSCAPE NOT LOADED!</strong></div>';
                return;
            }
            
            try {
                cy = cytoscape({
                    container: container,
                    
                    elements: [
                        { data: { id: 'n1', label: 'Neuron 1' } },
                        { data: { id: 'n2', label: 'Neuron 2' } },
                        { data: { id: 'h1', label: 'Head 1' } },
                        { data: { id: 'e1', source: 'n1', target: 'n2' } },
                        { data: { id: 'e2', source: 'n2', target: 'h1' } }
                    ],
                    
                    style: [
                        {
                            selector: 'node',
                            style: {
                                'label': 'data(label)',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'color': 'white',
                                'background-color': '#007bff',
                                'width': 100,
                                'height': 100,
                                'font-size': '16px',
                                'font-weight': 'bold',
                                'border-width': 4,
                                'border-color': '#0056b3'
                            }
                        },
                        {
                            selector: 'edge',
                            style: {
                                'width': 5,
                                'line-color': '#28a745',
                                'target-arrow-color': '#28a745',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier'
                            }
                        }
                    ],
                    
                    layout: {
                        name: 'circle',
                        fit: true,
                        padding: 50
                    }
                });
                
                cy.on('tap', 'node', function(evt) {
                    const node = evt.target;
                    updateStatus('Clicked: ' + node.id(), 'success');
                });
                
                updateStatus('Demo started successfully! Graph has ' + cy.nodes().length + ' nodes.', 'success');
                
            } catch (error) {
                updateStatus('ERROR: ' + error.message, 'error');
            }
        }

        function addNode() {
            if (!cy) {
                updateStatus('Start demo first!', 'error');
                return;
            }
            
            const nodeId = 'node' + Date.now();
            cy.add({ data: { id: nodeId, label: 'New Node' } });
            cy.layout({ name: 'circle', fit: true }).run();
            updateStatus('Added node: ' + nodeId, 'success');
        }

        function clearGraph() {
            if (!cy) {
                updateStatus('No graph to clear!', 'error');
                return;
            }
            
            cy.elements().remove();
            updateStatus('Graph cleared!', 'success');
        }

        // Auto-start
        document.addEventListener('DOMContentLoaded', function() {
            updateStatus('Page loaded. Auto-starting in 2 seconds...', 'info');
            setTimeout(startDemo, 2000);
        });
    </script>
</body>
</html>'''

@app.route('/demo')
def demo():
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
