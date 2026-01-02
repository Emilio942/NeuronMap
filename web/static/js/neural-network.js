/**
 * Neural Network Visualization
 * Visualizes neuron activations with color-coded points
 */

class NeuralNetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.neurons = [];
        this.activeNeurons = new Set();
        this.touchedNeurons = new Set();
        this.animationId = null;
        
        if (this.canvas) {
            this.resizeCanvas();
            window.addEventListener('resize', () => this.resizeCanvas());
        }
    }
    
    resizeCanvas() {
        if (!this.canvas) return;
        
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = 250;
    }
    
    /**
     * Initialize neurons in a grid pattern
     * @param {number} layerCount - Number of layers to display
     * @param {number} neuronsPerLayer - Neurons per layer
     */
    initializeNeurons(layerCount = 5, neuronsPerLayer = 12) {
        this.neurons = [];
        const padding = 40;
        const width = this.canvas.width - 2 * padding;
        const height = this.canvas.height - 2 * padding;
        
        const layerSpacing = width / (layerCount + 1);
        const neuronSpacing = height / (neuronsPerLayer + 1);
        
        for (let layer = 0; layer < layerCount; layer++) {
            for (let i = 0; i < neuronsPerLayer; i++) {
                const neuron = {
                    id: `neuron_${layer}_${i}`,
                    layer: layer,
                    index: i,
                    x: padding + (layer + 1) * layerSpacing,
                    y: padding + (i + 1) * neuronSpacing,
                    radius: 4,
                    state: 'inactive' // 'inactive', 'touched', 'active'
                };
                this.neurons.push(neuron);
            }
        }
    }
    
    /**
     * Update neuron states
     * @param {Array} activeIds - IDs of active neurons
     * @param {Array} touchedIds - IDs of touched neurons
     */
    updateNeuronStates(activeIds = [], touchedIds = []) {
        this.activeNeurons.clear();
        this.touchedNeurons.clear();
        
        activeIds.forEach(id => this.activeNeurons.add(id));
        touchedIds.forEach(id => this.touchedNeurons.add(id));
        
        this.neurons.forEach(neuron => {
            if (this.activeNeurons.has(neuron.id)) {
                neuron.state = 'active';
            } else if (this.touchedNeurons.has(neuron.id)) {
                neuron.state = 'touched';
            } else {
                neuron.state = 'inactive';
            }
        });
    }
    
    /**
     * Get color for neuron state
     */
    getColor(state, intensity = 1) {
        switch (state) {
            case 'active':
                return `rgba(76, 175, 80, ${0.8 * intensity})`; // Green
            case 'touched':
                return `rgba(255, 193, 7, ${0.6 * intensity})`; // Yellow/Orange
            case 'inactive':
            default:
                return `rgba(158, 158, 158, ${0.2 * intensity})`; // Gray
        }
    }
    
    /**
     * Get glow color for active neurons
     */
    getGlowColor(state) {
        switch (state) {
            case 'active':
                return 'rgba(76, 175, 80, 0.5)'; // Green glow
            case 'touched':
                return 'rgba(255, 193, 7, 0.3)'; // Orange glow
            default:
                return 'rgba(158, 158, 158, 0)';
        }
    }
    
    /**
     * Draw the neural network
     */
    draw() {
        if (!this.ctx || !this.canvas) return;
        
        // Clear canvas
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0)';
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections between layers
        this.drawConnections();
        
        // Draw neurons
        this.drawNeurons();
        
        // Draw labels
        this.drawLabels();
    }
    
    /**
     * Draw connections between neuron layers
     */
    drawConnections() {
        this.ctx.strokeStyle = 'rgba(100, 100, 100, 0.1)';
        this.ctx.lineWidth = 0.5;
        
        for (let i = 0; i < this.neurons.length - 1; i++) {
            const neuron1 = this.neurons[i];
            const neuron2 = this.neurons[i + 1];
            
            // Only connect neurons from consecutive layers
            if (neuron2.layer === neuron1.layer + 1) {
                this.ctx.beginPath();
                this.ctx.moveTo(neuron1.x, neuron1.y);
                this.ctx.lineTo(neuron2.x, neuron2.y);
                this.ctx.stroke();
            }
        }
    }
    
    /**
     * Draw neurons as circles
     */
    drawNeurons() {
        this.neurons.forEach(neuron => {
            // Draw glow for active neurons
            if (neuron.state !== 'inactive') {
                this.ctx.fillStyle = this.getGlowColor(neuron.state);
                this.ctx.beginPath();
                this.ctx.arc(neuron.x, neuron.y, neuron.radius * 3, 0, Math.PI * 2);
                this.ctx.fill();
            }
            
            // Draw neuron circle
            this.ctx.fillStyle = this.getColor(neuron.state);
            this.ctx.beginPath();
            this.ctx.arc(neuron.x, neuron.y, neuron.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw border
            this.ctx.strokeStyle = this.getColor(neuron.state, 1.2);
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        });
    }
    
    /**
     * Draw layer labels
     */
    drawLabels() {
        this.ctx.fillStyle = 'rgba(200, 200, 200, 0.6)';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        
        const layers = new Set(this.neurons.map(n => n.layer));
        const layerPositions = {};
        
        this.neurons.forEach(neuron => {
            if (!layerPositions[neuron.layer]) {
                layerPositions[neuron.layer] = neuron.x;
            }
        });
        
        Object.entries(layerPositions).forEach(([layer, x]) => {
            const label = `Layer ${parseInt(layer) + 1}`;
            this.ctx.fillText(label, x, this.canvas.height - 10);
        });
    }
    
    /**
     * Get statistics
     */
    getStats() {
        return {
            active: this.activeNeurons.size,
            touched: this.touchedNeurons.size,
            inactive: this.neurons.length - this.activeNeurons.size - this.touchedNeurons.size,
            total: this.neurons.length
        };
    }
    
    /**
     * Start continuous rendering
     */
    startAnimation() {
        const animate = () => {
            this.draw();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }
    
    /**
     * Stop animation
     */
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Global visualizer instance
let neuralNetworkViz = null;
let neuralNetworkDataTimer = null;

/**
 * Initialize the neural network visualization
 */
function initializeNeuralNetwork(canvasId = 'analysisNeuralNetworkCanvas') {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        return;
    }

    if (neuralNetworkViz) {
        neuralNetworkViz.stopAnimation();
    }

    neuralNetworkViz = new NeuralNetworkVisualizer(canvasId);
    neuralNetworkViz.initializeNeurons(5, 12);
    neuralNetworkViz.resizeCanvas();
    neuralNetworkViz.startAnimation();

    if (neuralNetworkDataTimer) {
        clearInterval(neuralNetworkDataTimer);
    }

    loadNeuralNetworkData();
    neuralNetworkDataTimer = window.setInterval(loadNeuralNetworkData, 5000);
}

/**
 * Load neural network data from API or simulate data
 */
async function loadNeuralNetworkData() {
    if (!neuralNetworkViz) return;
    
    try {
        // Try to fetch real data from API
        const response = await fetch('/api/neural-network-activity');
        if (response.ok) {
            const data = await response.json();
            neuralNetworkViz.updateNeuronStates(
                data.active_neurons || [],
                data.touched_neurons || []
            );

            setActivationTimestamp();
        } else {
            // Fallback: simulate data for demonstration
            simulateNeuralActivity();
            setActivationTimestamp(true);
        }
    } catch (error) {
        // Fallback: simulate data
        simulateNeuralActivity();
        setActivationTimestamp(true);
    }
    
    // Update statistics
    updateNeuronStats();
}

/**
 * Simulate neural network activity for demonstration
 */
function simulateNeuralActivity() {
    if (!neuralNetworkViz) return;
    
    const totalNeurons = neuralNetworkViz.neurons.length;
    
    // Random selection of active and touched neurons
    const activeCount = Math.floor(Math.random() * (totalNeurons * 0.3));
    const touchedCount = Math.floor(Math.random() * (totalNeurons * 0.2));
    
    const activeIds = [];
    const touchedIds = [];
    
    // Create random active neurons
    for (let i = 0; i < activeCount; i++) {
        const randomNeuron = neuralNetworkViz.neurons[
            Math.floor(Math.random() * totalNeurons)
        ];
        activeIds.push(randomNeuron.id);
    }
    
    // Create random touched neurons (excluding active)
    for (let i = 0; i < touchedCount; i++) {
        let randomNeuron;
        do {
            randomNeuron = neuralNetworkViz.neurons[
                Math.floor(Math.random() * totalNeurons)
            ];
        } while (activeIds.includes(randomNeuron.id));
        
        touchedIds.push(randomNeuron.id);
    }
    
    neuralNetworkViz.updateNeuronStates(activeIds, touchedIds);
}

/**
 * Update neuron statistics display
 */
function updateNeuronStats() {
    if (!neuralNetworkViz) return;
    
    const stats = neuralNetworkViz.getStats();
    const activeEl = document.getElementById('active-count');
    const touchedEl = document.getElementById('touched-count');
    const inactiveEl = document.getElementById('inactive-count');

    if (activeEl) activeEl.textContent = stats.active;
    if (touchedEl) touchedEl.textContent = stats.touched;
    if (inactiveEl) inactiveEl.textContent = stats.inactive;
}

function setActivationTimestamp(simulated = false) {
    const activationTimeEl = document.getElementById('activation-time');
    if (!activationTimeEl) return;

    const now = new Date().toLocaleTimeString('de-DE');
    activationTimeEl.textContent = simulated ? `Simuliert · ${now}` : now;
}

/**
 * Manually trigger neuron activation (for testing)
 */
function activateNeuron(neuronId) {
    if (!neuralNetworkViz) return;
    
    const neuron = neuralNetworkViz.neurons.find(n => n.id === neuronId);
    if (neuron) {
        neuron.state = 'active';
        neuralNetworkViz.activeNeurons.add(neuronId);
        updateNeuronStats();
    }
}

/**
 * Reset all neurons to inactive state
 */
function resetNeurons() {
    if (!neuralNetworkViz) return;
    
    neuralNetworkViz.activeNeurons.clear();
    neuralNetworkViz.touchedNeurons.clear();
    neuralNetworkViz.neurons.forEach(n => n.state = 'inactive');
    updateNeuronStats();
}

function teardownNeuralNetworkVisualization() {
    if (neuralNetworkDataTimer) {
        clearInterval(neuralNetworkDataTimer);
        neuralNetworkDataTimer = null;
    }

    if (neuralNetworkViz) {
        neuralNetworkViz.stopAnimation();
        neuralNetworkViz = null;
    }
}
