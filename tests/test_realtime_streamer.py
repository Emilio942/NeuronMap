import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.realtime_streamer import RealtimeActivationStreamer, StreamingConfig, ActivationFrame

# Mock ModelWrapper
class MockModelWrapper:
    def extract_activations(self, text, layer_indices):
        # Return dummy activations for requested layers
        return {idx: np.zeros((1, 10)) for idx in layer_indices}

@pytest.fixture
def streamer():
    model = MockModelWrapper()
    config = StreamingConfig()
    return RealtimeActivationStreamer(model, config)

@pytest.mark.asyncio
async def test_layer_switching(streamer):
    # Initial state
    assert streamer.current_layer_idx == 0
    
    # Simulate client message to change layer
    message = {'type': 'change_layer', 'layer_idx': 5}
    await streamer.handle_client_message(None, message)
    
    assert streamer.current_layer_idx == 5
    
    # Verify that extraction uses the new layer index
    # We need to mock _extract_activations_async or check if it calls model with correct index
    
    with patch.object(streamer.model, 'extract_activations', return_value={5: np.zeros((1, 10))}) as mock_extract:
        # Call the sync extraction directly to verify arguments
        streamer._extract_activations_sync("test text", layer_idx=5)
        
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[1]['layer_indices'] == [5]

@pytest.mark.asyncio
async def test_stream_activations_uses_current_layer(streamer):
    # Mock dependencies
    streamer.connected_clients.add("client1")
    streamer._broadcast_frame = AsyncMock()
    streamer.frame_rate_controller.wait_for_next_frame = AsyncMock()
    
    # Set a specific layer
    streamer.current_layer_idx = 3
    
    # Mock extraction to return something valid (tuple of activations and guardian data)
    with patch.object(streamer, '_extract_activations_async', return_value=(np.zeros(10), {})) as mock_extract:
        # Create a generator for input texts
        async def input_gen():
            yield "test"
        
        # Run stream for one item
        await streamer.stream_activations(input_gen())
        
        # Verify extraction was called with correct layer
        mock_extract.assert_called_with("test", 3)
        
        # Verify frame has correct layer index
        call_args = streamer._broadcast_frame.call_args
        frame = call_args[0][0]
        assert frame.layer_idx == 3

@pytest.mark.asyncio
async def test_config_update_polling(streamer):
    # Add check_config_update to the mock model
    streamer.model.check_config_update = MagicMock()
    
    # Mock dependencies to run fast
    streamer.connected_clients.add("client1")
    streamer._broadcast_frame = AsyncMock()
    streamer.frame_rate_controller.wait_for_next_frame = AsyncMock()
    streamer._extract_activations_async = AsyncMock(return_value=(np.zeros(10), {}))
    
    # Create a generator that yields enough items to trigger the check (30 frames)
    # We need 31 items to trigger it once (at frame 30, 0-indexed)
    async def input_gen():
        for i in range(35):
            yield f"test {i}"
            
    await streamer.stream_activations(input_gen())
    
    # Verify check_config_update was called
    assert streamer.model.check_config_update.called
