#!/usr/bin/env python3
"""
Real-time Activation Visualization Engine
High-performance streaming architecture for live neuron activation monitoring
"""

import asyncio
import websockets
import json
import numpy as np
import torch
import time
from typing import Dict, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.analysis.model_wrapper import ModelWrapper
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class ActivationFrame:
    """Single frame of activation data for streaming"""
    timestamp: float
    layer_idx: int
    activations: np.ndarray
    input_text: str
    sequence_position: int
    frame_id: int

@dataclass
class StreamingConfig:
    """Configuration for real-time streaming"""
    target_fps: int = 30
    max_buffer_size: int = 10000
    websocket_port: int = 8765
    max_neurons_per_frame: int = 50000
    compression_enabled: bool = True
    delta_compression: bool = True
    adaptive_sampling: bool = True

class CircularBuffer:
    """High-performance circular buffer for activation storage"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()

    def add(self, frame: ActivationFrame):
        """Add frame to buffer (thread-safe)"""
        with self.lock:
            self.buffer.append(frame)

    def get_latest(self, count: int = 1) -> List[ActivationFrame]:
        """Get latest N frames"""
        with self.lock:
            return list(self.buffer)[-count:]

    def get_range(self, start_time: float, end_time: float) -> List[ActivationFrame]:
        """Get frames within time range"""
        with self.lock:
            return [
                frame for frame in self.buffer
                if start_time <= frame.timestamp <= end_time
            ]

    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()

class FrameRateController:
    """Adaptive frame rate control for optimal performance"""

    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.last_frame_time = time.time()
        self.frame_times = deque(maxlen=100)

        # Adaptive parameters
        self.client_capabilities = {}
        self.network_latency = 0.0

    async def wait_for_next_frame(self):
        """Wait for next frame maintaining target FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time

        if elapsed < self.target_frame_time:
            await asyncio.sleep(self.target_frame_time - elapsed)

        self.last_frame_time = time.time()
        self.frame_times.append(elapsed)

    def get_actual_fps(self) -> float:
        """Get actual achieved FPS"""
        if not self.frame_times:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

    def adjust_target_fps(self, client_performance: Dict[str, float]):
        """Adaptively adjust FPS based on client performance"""
        client_fps = client_performance.get('achieved_fps', self.target_fps)
        network_latency = client_performance.get('network_latency', 0.0)

        # Adjust FPS based on client capabilities
        if client_fps < self.target_fps * 0.8:
            self.target_fps = max(10, int(self.target_fps * 0.9))
        elif client_fps > self.target_fps * 1.1 and network_latency < 50:
            self.target_fps = min(120, int(self.target_fps * 1.1))

        self.target_frame_time = 1.0 / self.target_fps

class DeltaCompressor:
    """Delta compression for efficient data transmission"""

    def __init__(self):
        self.last_frame: Optional[np.ndarray] = None
        self.compression_threshold = 0.001  # Minimum change to transmit

    def compress(self, current_activations: np.ndarray) -> Dict:
        """Compress activations using delta encoding"""
        if self.last_frame is None:
            self.last_frame = current_activations.copy()
            return {
                'type': 'full',
                'data': current_activations.tolist(),
                'shape': current_activations.shape
            }

        # Calculate delta
        delta = current_activations - self.last_frame

        # Find significant changes
        significant_changes = np.abs(delta) > self.compression_threshold

        if np.sum(significant_changes) < len(delta) * 0.1:  # Less than 10% changed
            # Send sparse delta
            indices = np.where(significant_changes)[0]
            values = delta[indices]

            self.last_frame = current_activations.copy()

            return {
                'type': 'delta',
                'indices': indices.tolist(),
                'values': values.tolist(),
                'shape': current_activations.shape
            }
        else:
            # Send full frame if too many changes
            self.last_frame = current_activations.copy()
            return {
                'type': 'full',
                'data': current_activations.tolist(),
                'shape': current_activations.shape
            }

class RealtimeActivationStreamer:
    """Main class for real-time activation streaming"""

    def __init__(self, model_wrapper: ModelWrapper, config: StreamingConfig):
        self.model = model_wrapper
        self.config = config

        # Core components
        self.circular_buffer = CircularBuffer(config.max_buffer_size)
        self.frame_rate_controller = FrameRateController(config.target_fps)
        self.delta_compressor = DeltaCompressor()

        # WebSocket management
        self.connected_clients = set()
        self.websocket_server = None

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.streaming_active = False
        self.frame_counter = 0

        # Performance monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'avg_processing_time': 0.0,
            'current_fps': 0.0,
            'compression_ratio': 0.0,
            'connected_clients': 0
        }

    async def start_streaming_server(self):
        """Start WebSocket server for real-time streaming"""
        logger.info(f"Starting real-time activation streaming server on port {self.config.websocket_port}")

        self.websocket_server = await websockets.serve(
            self.handle_client_connection,
            "localhost",
            self.config.websocket_port
        )

        logger.info("Real-time streaming server started successfully")

    async def handle_client_connection(self, websocket, path):
        """Handle new client connections"""
        logger.info(f"New client connected: {websocket.remote_address}")
        self.connected_clients.add(websocket)

        try:
            # Send initial configuration
            await websocket.send(json.dumps({
                'type': 'config',
                'max_neurons': self.config.max_neurons_per_frame,
                'target_fps': self.config.target_fps,
                'compression_enabled': self.config.compression_enabled
            }))

            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(websocket, json.loads(message))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.connected_clients.discard(websocket)

    async def handle_client_message(self, websocket, message: Dict):
        """Handle messages from clients"""
        message_type = message.get('type')

        if message_type == 'performance_update':
            # Update frame rate based on client performance
            self.frame_rate_controller.adjust_target_fps(message.get('performance', {}))

        elif message_type == 'request_history':
            # Send historical data
            start_time = message.get('start_time', 0)
            end_time = message.get('end_time', time.time())
            frames = self.circular_buffer.get_range(start_time, end_time)

            await websocket.send(json.dumps({
                'type': 'history_data',
                'frames': [self._serialize_frame(frame) for frame in frames]
            }))

        elif message_type == 'change_layer':
            # Change layer being monitored
            layer_idx = message.get('layer_idx', 0)
            # Implementation for layer switching
            pass

    async def stream_activations(self, input_texts: AsyncGenerator[str, None]):
        """Main streaming loop for processing and sending activations"""
        self.streaming_active = True

        try:
            async for text in input_texts:
                if not self.streaming_active:
                    break

                # Process text and extract activations
                start_time = time.time()
                activations = await self._extract_activations_async(text)
                processing_time = time.time() - start_time

                # Create frame
                frame = ActivationFrame(
                    timestamp=time.time(),
                    layer_idx=0,  # TODO: Make configurable
                    activations=activations,
                    input_text=text,
                    sequence_position=0,
                    frame_id=self.frame_counter
                )

                # Add to buffer
                self.circular_buffer.add(frame)

                # Send to connected clients
                if self.connected_clients:
                    await self._broadcast_frame(frame)

                # Update performance stats
                self._update_performance_stats(processing_time)

                # Wait for next frame
                await self.frame_rate_controller.wait_for_next_frame()

                self.frame_counter += 1

        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            self.streaming_active = False

    async def _extract_activations_async(self, text: str) -> np.ndarray:
        """Extract activations asynchronously"""
        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        activations = await loop.run_in_executor(
            self.executor,
            self._extract_activations_sync,
            text
        )

        return activations

    def _extract_activations_sync(self, text: str) -> np.ndarray:
        """Synchronous activation extraction"""
        try:
            # Use model wrapper to extract activations
            with torch.no_grad():
                activations = self.model.extract_activations(text, layer_indices=[0])

                # Get first layer activations and flatten
                layer_activations = activations[0]
                if len(layer_activations.shape) > 1:
                    layer_activations = layer_activations.mean(axis=0)  # Average over sequence

                # Limit neurons for performance
                if len(layer_activations) > self.config.max_neurons_per_frame:
                    layer_activations = layer_activations[:self.config.max_neurons_per_frame]

                return layer_activations.cpu().numpy()

        except Exception as e:
            logger.error(f"Error extracting activations: {e}")
            return np.zeros(1000)  # Return dummy data on error

    async def _broadcast_frame(self, frame: ActivationFrame):
        """Broadcast frame to all connected clients"""
        if not self.connected_clients:
            return

        # Prepare frame data
        frame_data = self._prepare_frame_for_transmission(frame)

        # Send to all clients
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(frame_data))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.discard(client)

    def _prepare_frame_for_transmission(self, frame: ActivationFrame) -> Dict:
        """Prepare frame data for network transmission"""
        if self.config.delta_compression:
            compressed_data = self.delta_compressor.compress(frame.activations)
        else:
            compressed_data = {
                'type': 'full',
                'data': frame.activations.tolist(),
                'shape': frame.activations.shape
            }

        return {
            'type': 'activation_frame',
            'timestamp': frame.timestamp,
            'layer_idx': frame.layer_idx,
            'frame_id': frame.frame_id,
            'input_text': frame.input_text[:100],  # Truncate for transmission
            'activations': compressed_data,
            'performance_stats': self.performance_stats
        }

    def _serialize_frame(self, frame: ActivationFrame) -> Dict:
        """Serialize frame for JSON transmission"""
        return {
            'timestamp': frame.timestamp,
            'layer_idx': frame.layer_idx,
            'frame_id': frame.frame_id,
            'input_text': frame.input_text,
            'activations_shape': frame.activations.shape,
            'activations_summary': {
                'mean': float(np.mean(frame.activations)),
                'std': float(np.std(frame.activations)),
                'max': float(np.max(frame.activations)),
                'min': float(np.min(frame.activations))
            }
        }

    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.performance_stats['frames_processed'] += 1

        # Update average processing time
        alpha = 0.1  # Exponential moving average
        if self.performance_stats['avg_processing_time'] == 0:
            self.performance_stats['avg_processing_time'] = processing_time
        else:
            self.performance_stats['avg_processing_time'] = (
                alpha * processing_time +
                (1 - alpha) * self.performance_stats['avg_processing_time']
            )

        self.performance_stats['current_fps'] = self.frame_rate_controller.get_actual_fps()
        self.performance_stats['connected_clients'] = len(self.connected_clients)

    def stop_streaming(self):
        """Stop the streaming process"""
        self.streaming_active = False
        if self.websocket_server:
            self.websocket_server.close()

class RealtimeVisualizationEngine:
    """High-level interface for real-time visualization"""

    def __init__(self, model_name: str = "gpt2", config_path: str = None):
        # Load configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.load_config(config_path)

        # Initialize components
        self.model_wrapper = ModelWrapper(model_name)
        self.streaming_config = StreamingConfig()
        self.streamer = RealtimeActivationStreamer(self.model_wrapper, self.streaming_config)

        logger.info(f"Real-time visualization engine initialized for {model_name}")

    async def start_realtime_session(self, input_source: AsyncGenerator[str, None]):
        """Start a real-time visualization session"""
        logger.info("Starting real-time visualization session")

        # Start WebSocket server
        await self.streamer.start_streaming_server()

        # Start streaming
        await self.streamer.stream_activations(input_source)

    async def demo_session(self, demo_texts: List[str], interval: float = 1.0):
        """Run a demo session with predefined texts"""

        async def demo_generator():
            for text in demo_texts:
                yield text
                await asyncio.sleep(interval)

        await self.start_realtime_session(demo_generator())

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.streamer.performance_stats.copy()

# Convenience functions
async def create_demo_text_stream(texts: List[str], interval: float = 1.0) -> AsyncGenerator[str, None]:
    """Create a demo text stream"""
    for text in texts:
        yield text
        await asyncio.sleep(interval)

def start_realtime_demo(model_name: str = "gpt2", port: int = 8765):
    """Start a demo real-time visualization session"""

    demo_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Neural networks can learn complex patterns from data.",
        "Transformers revolutionized natural language processing.",
        "Attention mechanisms help models focus on relevant information.",
        "Deep learning requires large amounts of training data.",
        "GPT models generate coherent and contextual text.",
        "BERT uses bidirectional attention for better understanding.",
        "Computer vision and NLP are converging rapidly.",
        "The future of AI looks incredibly promising."
    ]

    async def main():
        # Create visualization engine
        config = StreamingConfig(websocket_port=port)
        model_wrapper = ModelWrapper(model_name)
        engine = RealtimeVisualizationEngine(model_name)
        engine.streamer.config.websocket_port = port

        # Start demo
        await engine.demo_session(demo_texts, interval=2.0)

    # Run the demo
    asyncio.run(main())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Activation Visualization")
    parser.add_argument("--model", default="gpt2", help="Model name to analyze")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")

    args = parser.parse_args()

    if args.demo:
        print(f"Starting real-time demo on port {args.port}")
        print("Open http://localhost:8080/realtime_viewer.html to view the stream")
        start_realtime_demo(args.model, args.port)
    else:
        print("Real-time visualization engine ready")
        print("Use start_realtime_demo() or create your own session")
