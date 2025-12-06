
import sys
import os
import torch
import torch.nn as nn
import unittest
from dataclasses import dataclass
from typing import List

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from analysis.interventions import (
    ModifiableHookManager,
    InterventionSpec,
    InterventionType,
    intervention_context,
    create_dimmer_intervention
)

# Mock NeuronGroup for testing
@dataclass
class MockNeuronGroup:
    neuron_indices: List[int]

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        # Initialize weights to identity for easy testing
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(10))
            self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)

class TestDimmerIntervention(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.input_tensor = torch.ones(1, 10)  # Input of all ones

    def test_dimmer_scaling_all(self):
        """Test scaling all neurons in a layer."""
        layer_name = "linear"
        scaling_factor = 0.5
        
        # Create intervention spec manually
        spec = InterventionSpec(
            layer_name=layer_name,
            intervention_type=InterventionType.SCALING,
            intervention_value=scaling_factor
        )
        
        with intervention_context(self.model, [(layer_name, spec)]):
            output = self.model(self.input_tensor)
            
        # Expected output: all 0.5 (since input is 1.0 and weight is identity)
        expected = torch.ones(1, 10) * 0.5
        self.assertTrue(torch.allclose(output, expected), f"Expected {expected}, got {output}")

    def test_dimmer_scaling_specific_indices(self):
        """Test scaling specific neuron indices."""
        layer_name = "linear"
        scaling_factor = 0.0  # Turn off completely
        target_indices = [0, 2, 4]
        
        spec = create_dimmer_intervention(
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            target_indices=target_indices
        )
        
        with intervention_context(self.model, [(layer_name, spec)]):
            output = self.model(self.input_tensor)
            
        # Expected: indices 0, 2, 4 are 0.0, others are 1.0
        expected = torch.ones(1, 10)
        expected[0, [0, 2, 4]] = 0.0
        
        self.assertTrue(torch.allclose(output, expected), f"Expected {expected}, got {output}")

    def test_dimmer_with_neuron_group(self):
        """Test scaling using a NeuronGroup object."""
        layer_name = "linear"
        scaling_factor = 0.1
        group = MockNeuronGroup(neuron_indices=[1, 3, 5])
        
        spec = create_dimmer_intervention(
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            neuron_group=group
        )
        
        with intervention_context(self.model, [(layer_name, spec)]):
            output = self.model(self.input_tensor)
            
        # Expected: indices 1, 3, 5 are 0.1, others are 1.0
        expected = torch.ones(1, 10)
        expected[0, [1, 3, 5]] = 0.1
        
        self.assertTrue(torch.allclose(output, expected), f"Expected {expected}, got {output}")

    def test_amplification(self):
        """Test scaling factor > 1.0 (amplification)."""
        layer_name = "linear"
        scaling_factor = 2.0
        
        spec = create_dimmer_intervention(
            layer_name=layer_name,
            scaling_factor=scaling_factor
        )
        
        with intervention_context(self.model, [(layer_name, spec)]):
            output = self.model(self.input_tensor)
            
        expected = torch.ones(1, 10) * 2.0
        self.assertTrue(torch.allclose(output, expected), f"Expected {expected}, got {output}")

if __name__ == '__main__':
    unittest.main()
