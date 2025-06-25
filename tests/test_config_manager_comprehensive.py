"""
Comprehensive unit tests for Configuration System (Section 1.2).

This module tests the ConfigManager class and configuration validation
with focus on YAML loading, environment handling, and hardware detection.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any

from src.utils.config_manager import ConfigManager, ConfigurationError
from tests.conftest import TestDataGenerator, assert_config_validity


class TestConfigManager:
    """Test the ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            
            # Create test config files
            base_config = {
                "models": {
                    "default": {
                        "name": "test-model",
                        "type": "transformer",
                        "parameters": {"num_layers": 6, "hidden_size": 256}
                    }
                },
                "analysis": {
                    "batch_size": 4,
                    "max_sequence_length": 64
                }
            }
            
            dev_config = {
                "models": {
                    "default": {
                        "name": "test-model-dev",
                        "parameters": {"num_layers": 2, "hidden_size": 128}
                    }
                },
                "analysis": {
                    "batch_size": 2
                }
            }
            
            # Write config files
            with open(config_dir / "models.yaml", 'w') as f:
                yaml.dump(base_config, f)
            
            with open(config_dir / "environment_dev.yaml", 'w') as f:
                yaml.dump(dev_config, f)
            
            yield config_dir
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a ConfigManager with test configuration."""
        return ConfigManager(config_dir=str(temp_config_dir))
    
    def test_initialization(self, config_manager):
        """Test ConfigManager initialization."""
        assert config_manager.current_environment == "default"
        assert config_manager.config_dir is not None
        assert config_manager.config is not None
    
    def test_load_base_config(self, config_manager):
        """Test loading base configuration files."""
        config = config_manager.get_config()
        
        assert "models" in config
        assert "analysis" in config
        assert config["models"]["default"]["name"] == "test-model"
        assert config["analysis"]["batch_size"] == 4
    
    def test_environment_specific_config(self, config_manager):
        """Test loading environment-specific configurations."""
        # Switch to dev environment
        config_manager.set_environment("dev")
        config = config_manager.get_config()
        
        # Should have dev-specific overrides
        assert config["models"]["default"]["name"] == "test-model-dev"
        assert config["models"]["default"]["parameters"]["num_layers"] == 2
        assert config["analysis"]["batch_size"] == 2
        
        # Should still have base config values not overridden
        assert config["analysis"]["max_sequence_length"] == 64
    
    def test_config_merging(self, config_manager):
        """Test proper merging of base and environment configs."""
        config_manager.set_environment("dev")
        config = config_manager.get_config()
        
        # Test deep merging
        assert config["models"]["default"]["parameters"]["hidden_size"] == 128  # from dev
        assert config["models"]["default"]["type"] == "transformer"  # from base
    
    def test_invalid_environment(self, config_manager):
        """Test handling of invalid environment names."""
        with pytest.raises(ConfigurationError, match="Environment config file not found"):
            config_manager.set_environment("nonexistent")
    
    def test_missing_config_dir(self):
        """Test handling of missing configuration directory."""
        with pytest.raises(ConfigurationError, match="Config directory not found"):
            ConfigManager(config_dir="/nonexistent/path")
    
    def test_config_validation_basic(self, config_manager):
        """Test basic configuration validation."""
        config = config_manager.get_config()
        
        # Test required keys
        required_keys = ["models", "analysis"]
        assert_config_validity(config, required_keys)
    
    def test_config_schema_validation(self, config_manager):
        """Test Pydantic schema validation of configuration."""
        # This tests that the configuration matches expected schemas
        result = config_manager.validate_all_configs()
        assert result["status"] == "success"
        assert len(result["validation_results"]) > 0
        
        # All validations should pass
        for validation in result["validation_results"]:
            assert validation["valid"] is True
    
    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_hardware_detection(self, mock_memory, mock_cuda, config_manager):
        """Test hardware detection and validation."""
        # Mock hardware info
        mock_cuda.return_value = True
        mock_memory.return_value = Mock(total=16 * 1024**3)  # 16GB RAM
        
        hardware_info = config_manager.get_hardware_info()
        
        assert "cuda_available" in hardware_info
        assert "total_memory_gb" in hardware_info
        assert hardware_info["cuda_available"] is True
        assert hardware_info["total_memory_gb"] == 16
    
    def test_config_caching(self, config_manager):
        """Test that configuration is properly cached."""
        # First access
        config1 = config_manager.get_config()
        
        # Second access should return the same object
        config2 = config_manager.get_config()
        
        assert config1 is config2  # Should be the same object reference
    
    def test_config_refresh(self, config_manager, temp_config_dir):
        """Test configuration refresh when files change."""
        # Get initial config
        initial_config = config_manager.get_config()
        initial_batch_size = initial_config["analysis"]["batch_size"]
        
        # Modify config file
        new_config = {
            "analysis": {
                "batch_size": 999
            }
        }
        
        with open(temp_config_dir / "models.yaml", 'w') as f:
            yaml.dump(new_config, f)
        
        # Force reload
        config_manager._config = None
        updated_config = config_manager.get_config()
        
        assert updated_config["analysis"]["batch_size"] == 999
        assert updated_config["analysis"]["batch_size"] != initial_batch_size


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_validate_model_config(self, config_manager):
        """Test model configuration validation."""
        config = config_manager.get_config()
        model_config = config["models"]["default"]
        
        # Test required model fields
        required_fields = ["name", "type"]
        for field in required_fields:
            assert field in model_config, f"Missing required model field: {field}"
        
        # Test valid types
        assert model_config["type"] in ["transformer", "bert", "gpt", "t5", "llama"]
    
    def test_validate_analysis_config(self, config_manager):
        """Test analysis configuration validation."""
        config = config_manager.get_config()
        analysis_config = config["analysis"]
        
        # Test numeric constraints
        assert isinstance(analysis_config["batch_size"], int)
        assert analysis_config["batch_size"] > 0
        assert analysis_config["max_sequence_length"] > 0
    
    def test_config_type_validation(self, config_manager):
        """Test that configuration values have correct types."""
        config = config_manager.get_config()
        
        # Models section
        assert isinstance(config["models"], dict)
        assert isinstance(config["models"]["default"], dict)
        assert isinstance(config["models"]["default"]["name"], str)
        
        # Analysis section
        assert isinstance(config["analysis"], dict)
        assert isinstance(config["analysis"]["batch_size"], int)
    
    def test_invalid_yaml_handling(self, temp_config_dir):
        """Test handling of invalid YAML files."""
        # Create invalid YAML file
        with open(temp_config_dir / "invalid.yaml", 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with pytest.raises(ConfigurationError, match="Failed to load config"):
            config_manager = ConfigManager(config_dir=str(temp_config_dir))
            # Try to load the invalid file indirectly by setting up a bad config
            config_manager._load_config_files()


class TestEnvironmentHandling:
    """Test environment-specific configuration handling."""
    
    def test_environment_switching(self, config_manager):
        """Test switching between environments."""
        # Start with default
        assert config_manager.current_environment == "default"
        
        # Switch to dev
        config_manager.set_environment("dev")
        assert config_manager.current_environment == "dev"
        
        # Config should be updated
        config = config_manager.get_config()
        assert config["models"]["default"]["name"] == "test-model-dev"
    
    def test_environment_inheritance(self, config_manager):
        """Test that environments properly inherit from base configuration."""
        config_manager.set_environment("dev")
        config = config_manager.get_config()
        
        # Should have dev overrides
        assert config["models"]["default"]["parameters"]["num_layers"] == 2
        
        # Should inherit from base
        assert config["models"]["default"]["type"] == "transformer"
        assert config["analysis"]["max_sequence_length"] == 64
    
    def test_environment_validation(self, config_manager):
        """Test validation of environment names."""
        # Valid environment
        config_manager.set_environment("dev")  # Should not raise
        
        # Invalid environment
        with pytest.raises(ConfigurationError):
            config_manager.set_environment("invalid_env")
        
        # Empty environment name
        with pytest.raises(ConfigurationError):
            config_manager.set_environment("")


class TestConfigManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config_files(self, temp_config_dir):
        """Test handling of empty configuration files."""
        # Create empty config file
        with open(temp_config_dir / "empty.yaml", 'w') as f:
            f.write("")
        
        # Should handle gracefully
        config_manager = ConfigManager(config_dir=str(temp_config_dir))
        config = config_manager.get_config()
        assert isinstance(config, dict)
    
    def test_partial_config_files(self, temp_config_dir):
        """Test handling of partial configuration files."""
        # Create config with only some sections
        partial_config = {"models": {"test": {"name": "test-partial"}}}
        
        with open(temp_config_dir / "partial.yaml", 'w') as f:
            yaml.dump(partial_config, f)
        
        config_manager = ConfigManager(config_dir=str(temp_config_dir))
        config = config_manager.get_config()
        
        # Should have models section
        assert "models" in config
        assert "test" in config["models"]
    
    def test_config_with_invalid_types(self, temp_config_dir):
        """Test handling of configurations with invalid data types."""
        invalid_config = {
            "analysis": {
                "batch_size": "not_an_integer",  # Should be int
                "max_sequence_length": -1  # Should be positive
            }
        }
        
        with open(temp_config_dir / "invalid_types.yaml", 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(config_dir=str(temp_config_dir))
        
        # Validation should catch the errors
        result = config_manager.validate_all_configs()
        assert result["status"] == "error" or any(
            not v["valid"] for v in result["validation_results"]
        )
    
    @patch.dict(os.environ, {'NEURONMAP_CONFIG_ENV': 'test_env'})
    def test_environment_from_env_var(self, temp_config_dir):
        """Test setting environment from environment variable."""
        # Create environment config
        env_config = {"analysis": {"batch_size": 123}}
        with open(temp_config_dir / "environment_test_env.yaml", 'w') as f:
            yaml.dump(env_config, f)
        
        config_manager = ConfigManager(config_dir=str(temp_config_dir))
        
        # Should automatically use the environment from env var
        config = config_manager.get_config()
        assert config["analysis"]["batch_size"] == 123


class TestPerformanceAndMemory:
    """Test performance and memory characteristics of ConfigManager."""
    
    def test_config_loading_performance(self, config_manager):
        """Test that configuration loading is fast enough."""
        import time
        
        start_time = time.time()
        
        # Load config multiple times
        for _ in range(100):
            config_manager.get_config()
        
        elapsed = time.time() - start_time
        
        # Should be fast due to caching
        assert elapsed < 0.1, f"Config loading too slow: {elapsed:.3f}s for 100 calls"
    
    def test_memory_usage(self, config_manager):
        """Test that configuration doesn't leak memory."""
        import gc
        import sys
        
        # Get initial memory usage
        gc.collect()
        initial_refs = len(gc.get_objects())
        
        # Create and destroy multiple configs
        for _ in range(10):
            config = config_manager.get_config()
            del config
        
        gc.collect()
        final_refs = len(gc.get_objects())
        
        # Memory usage shouldn't grow significantly
        ref_growth = final_refs - initial_refs
        assert ref_growth < 100, f"Too many new objects created: {ref_growth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
