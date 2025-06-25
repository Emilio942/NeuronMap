"""Test configuration utilities."""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.utils.config_manager import NeuronMapConfig


class TestConfig:
    """Test configuration management."""
    
    def test_init_default_config_dir(self):
        """Test initialization with default config directory."""
        config = NeuronMapConfig()
        assert config.config_dir.name == "configs"
        
    def test_init_custom_config_dir(self):
        """Test initialization with custom config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = NeuronMapConfig(tmpdir)
            assert str(config.config_dir) == tmpdir
            
    def test_load_models_config(self):
        """Test loading model configurations."""
        config = NeuronMapConfig()
        models_config = config.load_models_config()
        
        assert "models" in models_config
        assert "layer_patterns" in models_config
        assert "default" in models_config["models"]
        
    def test_load_experiments_config(self):
        """Test loading experiment configurations."""
        config = NeuronMapConfig()
        experiments_config = config.load_experiments_config()
        
        assert "default" in experiments_config
        assert "question_generation" in experiments_config["default"]
        
    def test_get_model_config(self):
        """Test getting specific model configuration."""
        config = NeuronMapConfig()
        model_config = config.get_model_config("default")
        
        assert "name" in model_config
        assert "type" in model_config
        assert "layers" in model_config
        
    def test_get_model_config_not_found(self):
        """Test getting non-existent model configuration."""
        config = NeuronMapConfig()
        
        with pytest.raises(KeyError):
            config.get_model_config("nonexistent_model")
            
    def test_get_experiment_config(self):
        """Test getting experiment configuration."""
        config = NeuronMapConfig()
        exp_config = config.get_experiment_config("default")
        
        assert "question_generation" in exp_config
        assert "activation_extraction" in exp_config
        assert "visualization" in exp_config
        
    def test_resolve_layer_name(self):
        """Test resolving layer names from templates."""
        config = NeuronMapConfig()
        model_config = config.get_model_config("default")
        
        layer_name = config.resolve_layer_name(model_config, "mlp", 3)
        expected = "transformer.h.3.mlp.c_proj"
        assert layer_name == expected
        
    def test_resolve_layer_name_invalid_type(self):
        """Test resolving invalid layer type."""
        config = NeuronMapConfig()
        model_config = config.get_model_config("default")
        
        with pytest.raises(KeyError):
            config.resolve_layer_name(model_config, "invalid_type", 3)
