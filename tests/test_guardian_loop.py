
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.guardian.engine import GuardianEngine
from src.guardian.intervention_extractor import InterventionExtractor
from src.guardian.probes import ProbeManager

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

@pytest.fixture
def guardian_config():
    return {
        'enabled': True,
        'mode': 'intervention',
        'intervention_layers': [0], # Hook into first layer (layer1)
        'entropy_min': 10.0, # High min entropy to force noise injection (since simple model has low entropy)
        'noise_std': 0.5
    }

def test_guardian_intervention_loop(guardian_config):
    # 1. Setup
    torch.manual_seed(42)
    model = SimpleModel()
    input_tensor = torch.randn(1, 10)
    
    # 2. Baseline Run (No Guardian)
    # We just run the model normally
    with torch.no_grad():
        baseline_output = model(input_tensor)
        
    # 3. Guardian Run
    # Initialize Engine
    engine = GuardianEngine(guardian_config)
    
    # Initialize Extractor
    # We mock the tokenizer/model loading part of ActivationExtractor since we provide our own model
    extractor = InterventionExtractor(guardian_engine=engine, model_name_or_config="test-model")
    extractor.model = model
    extractor.device = "cpu"
    
    # Register Hook on layer1
    # Note: In a real scenario, ActivationExtractor finds layers by name. 
    # Here we manually register for the test.
    # InterventionExtractor.register_intervention_hook expects a module and an index
    extractor.register_intervention_hook(model.layer1, layer_idx=0)
    
    # Run Forward Pass
    with torch.no_grad():
        # We invoke the model directly, the hook is registered
        guardian_output = model(input_tensor)
        
    # 4. Assertions
    
    # Outputs should be different because noise was injected
    # The policy has min_entropy=10.0. A simple linear layer output will likely have entropy < 10.0.
    # So 'inject_noise' should trigger.
    
    assert not torch.allclose(baseline_output, guardian_output), "Guardian failed to modify output (Intervention did not happen)"
    
    # Verify that the difference is not just random noise but significant
    diff = torch.norm(baseline_output - guardian_output)
    print(f"Difference magnitude: {diff.item()}")
    assert diff.item() > 0.001

def test_guardian_monitoring_mode():
    # Test that monitoring mode does NOT modify output
    config = {
        'enabled': True,
        'mode': 'monitoring',
        'intervention_layers': [0],
        'entropy_min': 10.0 # Would trigger noise if in intervention mode
    }
    
    torch.manual_seed(42)
    model = SimpleModel()
    input_tensor = torch.randn(1, 10)
    
    # Baseline
    with torch.no_grad():
        baseline_output = model(input_tensor)
        
    # Guardian Setup
    engine = GuardianEngine(config)
    extractor = InterventionExtractor(guardian_engine=engine, model_name_or_config="test-model")
    extractor.model = model
    extractor.register_intervention_hook(model.layer1, layer_idx=0)
    
    # Run
    with torch.no_grad():
        monitor_output = model(input_tensor)
        
    # Assertions
    assert torch.allclose(baseline_output, monitor_output), "Monitoring mode modified the output!"

