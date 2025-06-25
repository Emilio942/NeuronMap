"""
Shared test fixtures and utilities for the NeuronMap test suite.

This module provides common fixtures and utilities that are used across
multiple test modules to ensure consistency and reduce duplication.
"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock
from typing import Dict, Any, List

# Test configuration for fixtures
TEST_CONFIG = {
    "models": {
        "default": {
            "name": "test-model",
            "type": "transformer",
            "parameters": {
                "num_layers": 6,
                "hidden_size": 256,
                "num_attention_heads": 8
            }
        }
    },
    "analysis": {
        "batch_size": 4,
        "max_sequence_length": 64,
        "target_layers": [2, 4, 6]
    },
    "data_generation": {
        "question_count": 10,
        "difficulty_range": [1, 5],
        "domains": ["science", "literature"]
    }
}


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration for all tests."""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_questions():
    """Provide sample questions for testing."""
    return [
        "What is the capital of France?",
        "How do neural networks learn?",
        "Explain the concept of entropy in physics.",
        "What are the main themes in Shakespeare's Hamlet?",
        "How does photosynthesis work in plants?"
    ]


@pytest.fixture
def sample_activations():
    """Provide sample activation data for testing."""
    # Generate realistic activation patterns
    batch_size, seq_len, hidden_size = 4, 16, 256
    activations = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    # Add some structure to make it more realistic
    activations[:, :, :64] *= 2.0  # Simulate higher activation in some neurons
    activations[:, 0, :] *= 1.5   # Simulate higher activation for first token
    
    return activations


@pytest.fixture
def sample_attention_weights():
    """Provide sample attention weight data for testing."""
    batch_size, num_heads, seq_len = 4, 8, 16
    # Create attention weights that sum to 1 across sequence dimension
    weights = np.random.rand(batch_size, num_heads, seq_len, seq_len).astype(np.float32)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights


@pytest.fixture
def mock_model():
    """Provide a mock neural network model for testing."""
    model = MagicMock()
    model.config.num_hidden_layers = 6
    model.config.hidden_size = 256
    model.config.num_attention_heads = 8
    model.config.vocab_size = 50000
    
    # Mock model layers
    model.layers = [MagicMock() for _ in range(6)]
    for i, layer in enumerate(model.layers):
        layer.name = f"layer_{i}"
    
    return model


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 7592, 2003, 1996, 3007, 1997, 2605, 1029, 102]
    tokenizer.decode.return_value = "What is the capital of France?"
    tokenizer.vocab_size = 50000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 102
    tokenizer.bos_token_id = 101
    return tokenizer


@pytest.fixture
def sample_experiment_config():
    """Provide sample experiment configuration."""
    return {
        "experiment_name": "test_experiment",
        "model_name": "test-model",
        "input_file": "test_questions.txt",
        "output_dir": "test_output",
        "layers": [2, 4, 6],
        "batch_size": 4,
        "max_length": 64,
        "analysis_methods": ["pca", "tsne"],
        "metadata": {
            "created_by": "test_user",
            "description": "Test experiment configuration"
        }
    }


@pytest.fixture
def sample_domain_data():
    """Provide sample domain-specific data for testing."""
    return {
        "science": {
            "vocabulary": ["atom", "molecule", "electron", "neutron", "proton"],
            "patterns": [
                "What is {concept}?",
                "How does {process} work?",
                "Explain the relationship between {term1} and {term2}."
            ]
        },
        "literature": {
            "vocabulary": ["metaphor", "symbolism", "narrative", "protagonist", "theme"],
            "patterns": [
                "Analyze the {device} in {work}.",
                "What themes are present in {author}'s work?",
                "How does {character} develop throughout {story}?"
            ]
        }
    }


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_test_questions_file(path: Path, questions: List[str]):
        """Create a test questions file."""
        with open(path, 'w', encoding='utf-8') as f:
            for question in questions:
                f.write(f"{question}\n")
    
    @staticmethod
    def create_test_config_file(path: Path, config: Dict[str, Any]):
        """Create a test configuration file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def create_test_activations_file(path: Path, activations: np.ndarray):
        """Create a test activations file."""
        np.save(path, activations)


class PerformanceTimer:
    """Context manager for measuring test performance."""
    
    def __init__(self, max_duration: float = 1.0):
        self.max_duration = max_duration
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.duration = time.time() - self.start_time
        assert self.duration < self.max_duration, f"Test took {self.duration:.2f}s, expected < {self.max_duration}s"


# Custom assertion helpers
def assert_valid_activation_shape(activations: np.ndarray, expected_batch_size: int = None):
    """Assert that activation tensor has valid shape."""
    assert activations.ndim >= 2, f"Activations should have at least 2 dimensions, got {activations.ndim}"
    assert activations.shape[-1] > 0, "Hidden dimension should be > 0"
    if expected_batch_size is not None:
        assert activations.shape[0] == expected_batch_size, f"Expected batch size {expected_batch_size}, got {activations.shape[0]}"


def assert_valid_attention_weights(weights: np.ndarray):
    """Assert that attention weights are valid (sum to 1, non-negative)."""
    assert weights.ndim == 4, f"Attention weights should have 4 dimensions, got {weights.ndim}"
    assert np.all(weights >= 0), "Attention weights should be non-negative"
    
    # Check that weights sum to approximately 1 across the last dimension
    weight_sums = weights.sum(axis=-1)
    np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-5, 
                              err_msg="Attention weights should sum to 1")


def assert_questions_quality(questions: List[str], min_length: int = 10):
    """Assert that generated questions meet quality standards."""
    assert len(questions) > 0, "Should generate at least one question"
    
    for i, question in enumerate(questions):
        assert isinstance(question, str), f"Question {i} should be a string"
        assert len(question.strip()) >= min_length, f"Question {i} too short: '{question}'"
        assert question.strip().endswith('?'), f"Question {i} should end with '?': '{question}'"
        assert question.strip()[0].isupper(), f"Question {i} should start with capital letter: '{question}'"


def assert_config_validity(config: Dict[str, Any], required_keys: List[str]):
    """Assert that configuration contains required keys and valid values."""
    for key in required_keys:
        assert key in config, f"Missing required config key: {key}"
    
    # Check for common config validity patterns
    if 'batch_size' in config:
        assert isinstance(config['batch_size'], int) and config['batch_size'] > 0
    
    if 'layers' in config:
        assert isinstance(config['layers'], list) and len(config['layers']) > 0
        assert all(isinstance(layer, int) and layer >= 0 for layer in config['layers'])
