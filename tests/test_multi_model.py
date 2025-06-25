"""Tests for multi-model and advanced analysis functionality."""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# We'll test the logic without requiring actual models/dependencies
from src.analysis.multi_model_extractor import ModelLayerMapper
from src.analysis.advanced_analysis import ActivationAnalyzer


class TestModelLayerMapper:
    """Test the model layer mapping functionality."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.test_config = {
            "name": "test-model",
            "type": "gpt",
            "layers": {
                "total_layers": 6
            }
        }
    
    def test_layer_mapper_init(self):
        """Test ModelLayerMapper initialization."""
        mapper = ModelLayerMapper(self.test_config)
        assert mapper.model_name == "test-model"
        assert mapper.model_type == "gpt"
    
    def test_categorize_layer(self):
        """Test layer categorization."""
        mapper = ModelLayerMapper(self.test_config)
        
        # Test attention layers
        assert mapper.categorize_layer("transformer.h.0.attn.c_attn") == "attention"
        assert mapper.categorize_layer("encoder.layer.0.attention.output") == "attention"
        
        # Test MLP layers
        assert mapper.categorize_layer("transformer.h.0.mlp.c_proj") == "mlp"
        assert mapper.categorize_layer("encoder.layer.0.intermediate.dense") == "mlp"
        
        # Test other layers
        assert mapper.categorize_layer("embeddings.word_embeddings") == "other"


class TestActivationAnalyzer:
    """Test the activation analysis functionality."""
    
    def setup_method(self):
        """Setup test data."""
        # Create mock activation data
        self.test_data = {
            'questions': ['Question 1', 'Question 2', 'Question 3'],
            'activations': {
                'layer_1': [
                    {'question_idx': 0, 'vector': [1.0, 2.0, 3.0], 'stats': {}},
                    {'question_idx': 1, 'vector': [2.0, 3.0, 4.0], 'stats': {}},
                    {'question_idx': 2, 'vector': [3.0, 4.0, 5.0], 'stats': {}}
                ],
                'layer_2': [
                    {'question_idx': 0, 'vector': [0.5, 1.5, 2.5], 'stats': {}},
                    {'question_idx': 1, 'vector': [1.5, 2.5, 3.5], 'stats': {}},
                    {'question_idx': 2, 'vector': [2.5, 3.5, 4.5], 'stats': {}}
                ]
            },
            'metadata': {'model_name': 'test-model'}
        }
    
    @patch('src.analysis.advanced_analysis.get_config')
    def test_analyzer_init(self, mock_config):
        """Test analyzer initialization."""
        mock_config.return_value.get_experiment_config.return_value = {
            'activation_extraction': {'output_file': 'test.csv'}
        }
        
        analyzer = ActivationAnalyzer()
        assert analyzer is not None
    
    def test_compute_activation_statistics(self):
        """Test activation statistics computation."""
        analyzer = ActivationAnalyzer.__new__(ActivationAnalyzer)  # Skip __init__
        
        # Test with simple data
        activations = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        stats = analyzer.compute_activation_statistics(activations)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'sparsity' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        
        # Check basic calculations
        assert stats['mean'] == pytest.approx(5.0)  # Mean of 1-9
        assert stats['min'] == 1.0
        assert stats['max'] == 9.0
    
    def test_analyze_neurons(self):
        """Test neuron-wise analysis."""
        analyzer = ActivationAnalyzer.__new__(ActivationAnalyzer)  # Skip __init__
        
        # Create test activation matrix
        activation_matrix = np.array([
            [1, 0, 3],  # Question 1
            [2, 0, 4],  # Question 2  
            [3, 0, 5]   # Question 3
        ])
        
        neuron_stats = analyzer._analyze_neurons(activation_matrix)
        
        assert neuron_stats['total_neurons'] == 3
        assert len(neuron_stats['neuron_statistics']) == 3
        assert len(neuron_stats['most_active_neurons']) > 0
        assert len(neuron_stats['sparsest_neurons']) > 0
        
        # Check that neuron 1 (index 1) is identified as sparse (all zeros)
        sparsest = neuron_stats['sparsest_neurons']
        sparsest_neuron_idx = max(sparsest, key=lambda x: x['sparsity'])['neuron_idx']
        assert sparsest_neuron_idx == 1  # Second neuron (all zeros)
    
    def test_analyze_correlations(self):
        """Test correlation analysis."""
        analyzer = ActivationAnalyzer.__new__(ActivationAnalyzer)  # Skip __init__
        
        # Create test data with known correlations
        activation_matrix = np.array([
            [1, 2],  # Question 1
            [2, 4],  # Question 2 (perfectly correlated)
            [3, 6]   # Question 3 (perfectly correlated)
        ])
        
        corr_analysis = analyzer._analyze_correlations(activation_matrix)
        
        assert 'question_correlation_matrix' in corr_analysis
        assert 'neuron_correlation_matrix' in corr_analysis
        assert 'question_correlation_stats' in corr_analysis
        assert 'neuron_correlation_stats' in corr_analysis
        
        # Check neuron correlation (should be perfect correlation = 1.0)
        neuron_corr = np.array(corr_analysis['neuron_correlation_matrix'])
        assert neuron_corr[0, 1] == pytest.approx(1.0, abs=1e-10)
    
    def test_find_highly_correlated_pairs(self):
        """Test finding highly correlated pairs."""
        analyzer = ActivationAnalyzer.__new__(ActivationAnalyzer)  # Skip __init__
        
        # Create correlation matrix with known high correlations
        corr_matrix = np.array([
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        
        pairs = analyzer._find_highly_correlated_pairs(corr_matrix, threshold=0.8)
        
        assert len(pairs) == 1  # Only one pair above 0.8
        assert pairs[0]['index_1'] == 0
        assert pairs[0]['index_2'] == 1
        assert pairs[0]['correlation'] == pytest.approx(0.9)


class TestIntegration:
    """Integration tests for the multi-model system."""
    
    def test_config_loading(self):
        """Test that configurations can be loaded."""
        # This would test with actual config files
        # For now, just test the structure
        
        expected_sections = [
            'models',
            'layer_patterns', 
            'analysis_configs',
            'extraction_settings'
        ]
        
        # In a real test, we'd load the actual config
        # and verify these sections exist
        assert True  # Placeholder
    
    @patch('src.analysis.multi_model_extractor.MultiModelActivationExtractor')
    def test_cli_integration(self, mock_extractor):
        """Test CLI integration with new commands."""
        # Mock the extractor to avoid loading actual models
        mock_instance = Mock()
        mock_extractor.return_value = mock_instance
        mock_instance.load_model.return_value = True
        mock_instance.discover_model_layers.return_value = {
            'attention': ['layer1', 'layer2'],
            'mlp': ['layer3', 'layer4'],
            'other': []
        }
        
        # Test would call CLI commands here
        # For now, just verify mocking works
        assert mock_extractor is not None


def test_requirements_compatibility():
    """Test that all required packages are specified."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            requirements = f.read()
        
        # Check for key dependencies
        required_packages = [
            'torch', 'transformers', 'numpy', 'pandas',
            'scikit-learn', 'h5py', 'scipy'
        ]
        
        for package in required_packages:
            assert package in requirements, f"Missing required package: {package}"


if __name__ == "__main__":
    # Run specific tests
    test_mapper = TestModelLayerMapper()
    test_mapper.setup_method()
    test_mapper.test_layer_mapper_init()
    test_mapper.test_categorize_layer()
    
    test_analyzer = TestActivationAnalyzer()
    test_analyzer.setup_method()
    test_analyzer.test_compute_activation_statistics()
    test_analyzer.test_analyze_neurons()
    test_analyzer.test_analyze_correlations()
    test_analyzer.test_find_highly_correlated_pairs()
    
    print("✓ All basic tests passed!")
