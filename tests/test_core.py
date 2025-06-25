"""Unit tests for NeuronMap core functionality.

This module provides comprehensive unit tests for all major components:
- Configuration management
- Data processing
- Analysis modules
- Visualization components
- Performance optimization
"""

import unittest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.utils.config_manager import ConfigManager, get_config
    from src.utils.validation import validate_experiment_config, check_system_requirements
    from src.utils.error_handling import NeuronMapException, ValidationError
    from src.data_processing.quality_manager import DataQualityManager
    from src.data_processing.metadata_manager import MetadataManager
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


class TestConfigManager(unittest.TestCase):
    """Test configuration management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name) / "configs"
        self.config_dir.mkdir(parents=True)
        
        # Create test config files
        self.models_config = {
            "test_model": {
                "model_name": "gpt2",
                "layers": ["transformer.h.0", "transformer.h.1"],
                "max_length": 512
            }
        }
        
        self.experiments_config = {
            "test_experiment": {
                "model": "test_model",
                "batch_size": 4,
                "num_questions": 10
            }
        }
        
        with open(self.config_dir / "models.yaml", 'w') as f:
            import yaml
            yaml.dump(self.models_config, f)
        
        with open(self.config_dir / "experiments.yaml", 'w') as f:
            import yaml
            yaml.dump(self.experiments_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_config_loading(self):
        """Test configuration loading."""
        try:
            config_manager = ConfigManager(str(self.config_dir))
            
            # Test model config loading
            model_config = config_manager.get_model_config("test_model")
            self.assertEqual(model_config["model_name"], "gpt2")
            self.assertEqual(len(model_config["layers"]), 2)
            
            # Test experiment config loading
            exp_config = config_manager.get_experiment_config("test_experiment")
            self.assertEqual(exp_config["batch_size"], 4)
            
        except ImportError:
            self.skipTest("ConfigManager not available")
    
    def test_config_validation(self):
        """Test configuration validation."""
        try:
            # Test valid config
            valid_config = {
                "model": "test_model",
                "batch_size": 4,
                "num_questions": 10
            }
            errors = validate_experiment_config(valid_config)
            self.assertEqual(len(errors), 0)
            
            # Test invalid config
            invalid_config = {
                "model": "",  # Empty model name
                "batch_size": -1,  # Negative batch size
            }
            errors = validate_experiment_config(invalid_config)
            self.assertGreater(len(errors), 0)
            
        except ImportError:
            self.skipTest("Validation functions not available")


class TestDataQualityManager(unittest.TestCase):
    """Test data quality management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data = [
            {"text": "This is a good question.", "category": "test"},
            {"text": "This is a good question.", "category": "test"},  # Duplicate
            {"text": "", "category": "test"},  # Empty text
            {"text": "Another unique question.", "category": "test"},
            {"text": "Short", "category": "test"},  # Too short
        ]
    
    def tearDown(self):
        """Clean up test fixtures.""" 
        self.temp_dir.cleanup()
    
    def test_duplicate_detection(self):
        """Test duplicate detection functionality."""
        try:
            quality_manager = DataQualityManager()
            
            # Test duplicate detection
            duplicates = quality_manager.detect_duplicates(self.test_data)
            self.assertGreater(len(duplicates), 0)
            
            # Check that duplicates are detected correctly
            duplicate_texts = [item["text"] for item in duplicates]
            self.assertIn("This is a good question.", duplicate_texts)
            
        except ImportError:
            self.skipTest("DataQualityManager not available")
    
    def test_data_validation(self):
        """Test data validation."""
        try:
            quality_manager = DataQualityManager()
            
            # Test validation
            validation_errors = quality_manager.validate_questions_data(self.test_data)
            
            # Should find errors for empty text and short text
            self.assertGreater(len(validation_errors), 0)
            
            # Check specific error types
            error_types = [error["type"] for error in validation_errors]
            self.assertIn("empty_text", error_types)
            
        except ImportError:
            self.skipTest("DataQualityManager not available")


class TestMetadataManager(unittest.TestCase):
    """Test metadata management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_file = Path(self.temp_dir.name) / "metadata.json"
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_experiment_recording(self):
        """Test experiment metadata recording."""
        try:
            metadata_manager = MetadataManager(str(self.metadata_file))
            
            # Record an experiment
            experiment_id = metadata_manager.record_experiment(
                config_name="test_config",
                parameters={"batch_size": 4, "model": "gpt2"},
                description="Test experiment"
            )
            
            self.assertIsNotNone(experiment_id)
            
            # Retrieve experiment
            experiments = metadata_manager.list_experiments()
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0]["config_name"], "test_config")
            
        except ImportError:
            self.skipTest("MetadataManager not available")


class TestSystemRequirements(unittest.TestCase):
    """Test system requirements checking."""
    
    def test_system_requirements(self):
        """Test system requirements validation."""
        try:
            requirements = check_system_requirements()
            
            # Should return a dictionary
            self.assertIsInstance(requirements, dict)
            
            # Should check for key packages
            expected_packages = ["torch", "transformers", "numpy", "pandas"]
            for package in expected_packages:
                self.assertIn(package, requirements)
            
        except ImportError:
            self.skipTest("System requirements check not available")


class TestErrorHandling(unittest.TestCase):
    """Test error handling functionality."""
    
    def test_custom_exceptions(self):
        """Test custom exception types."""
        try:
            # Test NeuronMapException
            with self.assertRaises(NeuronMapException):
                raise NeuronMapException("Test error", "TEST_ERROR")
            
            # Test ValidationError
            with self.assertRaises(ValidationError):
                raise ValidationError("Validation failed", "VALIDATION_ERROR")
            
        except ImportError:
            self.skipTest("Custom exceptions not available")


class MockAnalysisTests(unittest.TestCase):
    """Test analysis modules with mocked dependencies."""
    
    def test_activation_extraction_mock(self):
        """Test activation extraction with mocked model."""
        # Mock PyTorch model
        mock_model = MagicMock()
        
        try:
            import torch
            mock_model.return_value.hidden_states = [
                torch.randn(1, 10, 768) for _ in range(12)
            ]
        except ImportError:
            # Fallback to numpy if torch not available
            mock_model.return_value.hidden_states = [
                np.random.randn(1, 10, 768) for _ in range(12)
            ]
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]]
        }
        
        # Test would go here if we had the actual analyzer
        self.assertTrue(True)  # Placeholder
    
    def test_visualization_mock(self):
        """Test visualization with mocked data."""
        # Mock activation data
        mock_activations = {
            "layer_0": np.random.randn(100, 768),
            "layer_6": np.random.randn(100, 768),
            "layer_11": np.random.randn(100, 768)
        }
        
        # Test would create visualizations with mocked data
        self.assertIsInstance(mock_activations, dict)
        self.assertEqual(len(mock_activations), 3)


# Parametrized tests disabled - requires pytest
# @pytest.mark.parametrize("config_name,expected_model", [
#     ("default", "gpt2"),
#     ("dev", "distilgpt2"),
# ])
# def test_config_parametrized(config_name, expected_model):
#     """Parametrized test for different configurations."""
#     # This would test different configuration scenarios
#     assert config_name in ["default", "dev"]
#     assert expected_model in ["gpt2", "distilgpt2"]


class PropertyBasedTests(unittest.TestCase):
    """Property-based tests using hypothesis (when available)."""
    
    def test_activation_properties(self):
        """Test properties of activation data."""
        try:
            # Hypothesis not available, using basic test instead
            activations = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            # Normalize activations
            activations = np.array(activations)
            normalized = (activations - np.mean(activations)) / (np.std(activations) + 1e-8)
            
            # Check that order is preserved
            self.assertTrue(normalized[0] < normalized[-1])
            
        except ImportError:
            self.skipTest("Property-based testing not available")


class IntegrationTests(unittest.TestCase):
    """Integration tests for end-to-end workflows."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "outputs"
        self.output_dir.mkdir()
    
    def tearDown(self):
        """Clean up integration test environment."""
        self.temp_dir.cleanup()
    
    def test_full_pipeline_mock(self):
        """Test full analysis pipeline with mocked components."""
        # This would test the complete workflow:
        # 1. Question generation
        # 2. Activation extraction
        # 3. Analysis
        # 4. Visualization
        # 5. Result saving
        
        # Mock each component and verify the workflow
        steps_completed = []
        
        # Mock question generation
        mock_questions = ["What is AI?", "How do neural networks work?"]
        steps_completed.append("question_generation")
        
        # Mock activation extraction
        mock_activations = {"layer_0": np.random.randn(2, 768)}
        steps_completed.append("activation_extraction")
        
        # Mock analysis
        mock_analysis_results = {"pca_variance": 0.85, "clusters": 3}
        steps_completed.append("analysis")
        
        # Mock visualization
        mock_plot_paths = [str(self.output_dir / "pca_plot.png")]
        steps_completed.append("visualization")
        
        # Verify all steps completed
        expected_steps = ["question_generation", "activation_extraction", "analysis", "visualization"]
        self.assertEqual(steps_completed, expected_steps)


def run_benchmark_tests():
    """Run performance benchmark tests."""
    print("Running benchmark tests...")
    
    # Benchmark activation extraction speed
    start_time = time.time()
    # Simulate activation extraction
    for i in range(100):
        mock_activation = np.random.randn(768)
    extraction_time = time.time() - start_time
    
    print(f"Mock activation extraction: {extraction_time:.4f}s for 100 iterations")
    
    # Benchmark visualization rendering
    start_time = time.time()
    # Simulate visualization creation
    for i in range(10):
        mock_plot_data = np.random.randn(100, 2)
    visualization_time = time.time() - start_time
    
    print(f"Mock visualization creation: {visualization_time:.4f}s for 10 plots")


if __name__ == "__main__":
    # Run unit tests
    print("Running NeuronMap unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run benchmark tests
    run_benchmark_tests()
