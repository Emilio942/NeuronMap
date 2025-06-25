#!/usr/bin/env python3
"""
Demo script for Real-time Activation Visualization
Tests the streaming system with synthetic data
"""

import asyncio
import sys
import os
import time
import numpy as np
from typing import List, AsyncGenerator

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_mock_model_wrapper():
    """Create a mock model wrapper for testing"""
    
    class MockModelWrapper:
        def __init__(self, model_name: str = "gpt2"):
            self.model_name = model_name
            self.hidden_size = 768
            print(f"Mock model wrapper initialized for {model_name}")
        
        def extract_activations(self, text: str, layer_indices: List[int] = [0]):
            """Generate synthetic activations for testing"""
            # Simulate processing time
            time.sleep(0.01)
            
            # Generate synthetic activations based on text characteristics
            text_length = len(text)
            complexity = text.count(' ') + text.count(',') + text.count('.')
            
            # Create synthetic activation pattern
            activations = []
            for layer_idx in layer_indices:
                # Base activation level influenced by text characteristics
                base_level = 0.1 + (complexity / 100.0)
                
                # Add some randomness with text-dependent seed
                np.random.seed(hash(text) % 1000000)
                layer_activations = np.random.normal(
                    base_level, 0.3, (text_length, self.hidden_size)
                )
                
                # Add some interesting patterns
                # Attention-like peaks at beginning and end
                layer_activations[0] *= 1.5
                layer_activations[-1] *= 1.3
                
                # Add some periodic patterns
                for i in range(0, self.hidden_size, 50):
                    layer_activations[:, i] *= 1.2
                
                # Clip to reasonable range
                layer_activations = np.clip(layer_activations, -2.0, 2.0)
                activations.append(layer_activations)
            
            return activations
    
    return MockModelWrapper()

async def demo_text_generator() -> AsyncGenerator[str, None]:
    """Generate demo texts for streaming"""
    
    demo_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence rapidly.",
        "Neural networks can learn complex patterns from large datasets.",
        "Transformers revolutionized natural language processing completely.",
        "Attention mechanisms help models focus on relevant information efficiently.",
        "Deep learning requires substantial amounts of high-quality training data.",
        "GPT models generate coherent and contextually appropriate text responses.",
        "BERT uses bidirectional attention for enhanced understanding capabilities.",
        "Computer vision and natural language processing are converging rapidly.",
        "The future of artificial intelligence looks incredibly promising and exciting.",
        "Reinforcement learning enables agents to learn optimal decision-making strategies.",
        "Convolutional neural networks excel at hierarchical feature extraction tasks.",
        "Recurrent neural networks can model sequential dependencies effectively.",
        "Self-supervised learning reduces the need for labeled training examples.",
        "Transfer learning allows models to leverage pre-trained knowledge efficiently.",
        "Few-shot learning enables rapid adaptation to new tasks and domains.",
        "Multimodal models can process and understand diverse data types simultaneously.",
        "Adversarial training improves model robustness against malicious attacks.",
        "Federated learning enables privacy-preserving distributed model training.",
        "Quantum machine learning may revolutionize computational capabilities."
    ]
    
    for i, text in enumerate(demo_texts):
        print(f"Streaming text {i+1}/{len(demo_texts)}: {text[:50]}...")
        yield text
        await asyncio.sleep(2.0)  # 2 second intervals
    
    # Continue with random variations
    import random
    while True:
        base_text = random.choice(demo_texts)
        variations = [
            f"Consider this: {base_text}",
            f"Analyzing: {base_text}",
            f"Understanding: {base_text}",
            f"Processing: {base_text}"
        ]
        yield random.choice(variations)
        await asyncio.sleep(3.0)

async def run_realtime_demo():
    """Run the real-time visualization demo"""
    
    print("="*80)
    print("NEURONMAP - REAL-TIME ACTIVATION VISUALIZATION DEMO")
    print("="*80)
    
    try:
        # Import required modules with mock fallbacks
        try:
            from src.visualization.realtime_streamer import (
                RealtimeVisualizationEngine, 
                StreamingConfig
            )
            
            # Create mock model wrapper
            mock_model = create_mock_model_wrapper()
            
            # Create visualization engine with mock model
            engine = RealtimeVisualizationEngine.__new__(RealtimeVisualizationEngine)
            engine.model_wrapper = mock_model
            engine.streaming_config = StreamingConfig(
                target_fps=10,  # Lower FPS for demo
                max_buffer_size=1000,
                websocket_port=8765,
                max_neurons_per_frame=1000,
                compression_enabled=True,
                delta_compression=True
            )
            
            # Create streamer with mock model
            from src.visualization.realtime_streamer import RealtimeActivationStreamer
            engine.streamer = RealtimeActivationStreamer(mock_model, engine.streaming_config)
            
            print("✅ Real-time visualization engine initialized")
            print("📡 Starting WebSocket server on port 8765...")
            print("🌐 Open http://localhost:8080/static/realtime_viewer.html to view")
            print("⏱️  Demo will run for 2 minutes with 20 sample texts")
            print()
            
            # Start demo session
            await engine.demo_session(
                [text async for text in demo_text_generator()],
                interval=2.0
            )
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Running fallback demo with synthetic WebSocket server...")
            await run_fallback_demo()
            
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

async def run_fallback_demo():
    """Fallback demo with synthetic WebSocket server"""
    
    print("🔄 Starting fallback demo mode...")
    
    # Create a simple WebSocket server for testing
    try:
        import websockets
        import json
        
        connected_clients = set()
        
        async def handle_client(websocket, path):
            print(f"📱 Client connected: {websocket.remote_address}")
            connected_clients.add(websocket)
            
            try:
                # Send configuration
                config_msg = {
                    'type': 'config',
                    'max_neurons': 1000,
                    'target_fps': 10,
                    'compression_enabled': True
                }
                await websocket.send(json.dumps(config_msg))
                
                # Keep connection alive
                async for message in websocket:
                    data = json.loads(message)
                    print(f"📨 Received: {data.get('type', 'unknown')}")
                    
            except websockets.exceptions.ConnectionClosed:
                print(f"📱 Client disconnected: {websocket.remote_address}")
            finally:
                connected_clients.discard(websocket)
        
        async def broadcast_demo_data():
            """Send demo activation data to clients"""
            frame_id = 0
            
            async for text in demo_text_generator():
                if not connected_clients:
                    continue
                
                # Generate synthetic activations
                mock_model = create_mock_model_wrapper()
                activations = mock_model.extract_activations(text)[0]
                activation_vector = activations.mean(axis=0)  # Average over sequence
                
                # Create frame data
                frame_data = {
                    'type': 'activation_frame',
                    'timestamp': time.time(),
                    'layer_idx': 0,
                    'frame_id': frame_id,
                    'input_text': text,
                    'activations': {
                        'type': 'full',
                        'data': activation_vector.tolist(),
                        'shape': activation_vector.shape
                    },
                    'performance_stats': {
                        'frames_processed': frame_id + 1,
                        'avg_processing_time': 0.01,
                        'current_fps': 10.0,
                        'compression_ratio': 0.7,
                        'connected_clients': len(connected_clients)
                    }
                }
                
                # Broadcast to all clients
                if connected_clients:
                    message = json.dumps(frame_data)
                    disconnected = []
                    
                    for client in connected_clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.append(client)
                    
                    # Remove disconnected clients
                    for client in disconnected:
                        connected_clients.discard(client)
                    
                    print(f"📊 Sent frame {frame_id} to {len(connected_clients)} clients")
                
                frame_id += 1
        
        # Start server and demo
        server = await websockets.serve(handle_client, "localhost", 8765)
        print("🚀 Fallback WebSocket server started on port 8765")
        
        # Start broadcasting demo data
        await broadcast_demo_data()
        
    except ImportError:
        print("❌ websockets module not available")
        print("🔧 Install with: pip install websockets")
        return False

def create_simple_http_server():
    """Create a simple HTTP server to serve the HTML file"""
    import http.server
    import socketserver
    import threading
    import os
    
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    def run_server():
        with socketserver.TCPServer(("", 8080), CustomHandler) as httpd:
            print("🌐 HTTP server started on http://localhost:8080")
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread

if __name__ == "__main__":
    print("🧠 NeuronMap Real-time Visualization Demo")
    print("==========================================")
    
    # Check if HTML file exists
    html_path = os.path.join(os.getcwd(), "static", "realtime_viewer.html")
    if os.path.exists(html_path):
        print(f"✅ HTML viewer found: {html_path}")
        
        # Start simple HTTP server
        create_simple_http_server()
        
        print("\n📋 Setup Instructions:")
        print("1. Open http://localhost:8080/static/realtime_viewer.html in your browser")
        print("2. Click 'Connect' to start receiving live data")
        print("3. Watch real-time neural activations!")
        print("\n⏳ Starting demo in 3 seconds...")
        
        time.sleep(3)
        
        # Run the demo
        asyncio.run(run_realtime_demo())
        
    else:
        print(f"❌ HTML viewer not found at: {html_path}")
        print("Please ensure the static/realtime_viewer.html file exists")
