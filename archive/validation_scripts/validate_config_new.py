#!/usr/bin/env python3
"""Configuration validation script for NeuronMap."""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_config_system():
    """Test the new configuration system."""
    print("🧪 Testing NeuronMap Configuration System")
    print("=" * 50)
    
    try:
        from utils.config_manager import ConfigManager, NeuronMapConfig, get_config
        print("✅ Successfully imported configuration system")
    except ImportError as e:
        print(f"❌ Failed to import configuration system: {e}")
        return False
    
    # Test 1: Create default configuration
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        print("✅ Successfully created default configuration")
        print(f"   Model: {config.model.name}")
        print(f"   Device: {config.model.device}")
        print(f"   Input file: {config.data.input_file}")
    except Exception as e:
        print(f"❌ Failed to create default configuration: {e}")
        return False
    
    # Test 2: Validate configuration structure
    try:
        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'analysis')
        assert hasattr(config, 'visualization')
        assert hasattr(config, 'experiment')
        print("✅ Configuration structure is valid")
    except AssertionError:
        print("❌ Configuration structure is invalid")
        return False
    
    # Test 3: Test path validation
    try:
        paths_valid = config_manager.validate_paths()
        if paths_valid:
            print("✅ All configured paths are valid")
        else:
            print("⚠️  Some path validation issues (but not critical)")
    except Exception as e:
        print(f"❌ Path validation failed: {e}")
        return False
    
    # Test 4: Test configuration update
    try:
        config_manager.update_config(
            model={"name": "gpt2", "device": "cpu"},
            experiment={"name": "test_experiment"}
        )
        updated_config = config_manager.get_config()
        assert updated_config.model.name == "gpt2"
        assert updated_config.experiment.name == "test_experiment"
        print("✅ Configuration update works correctly")
    except Exception as e:
        print(f"❌ Configuration update failed: {e}")
        return False
    
    # Test 5: Test configuration save/load
    try:
        test_config_path = "configs/test_config.yaml"
        config_manager.save_config(test_config_path)
        
        # Load it back
        new_manager = ConfigManager()
        loaded_config = new_manager.load_config(test_config_path)
        assert loaded_config.model.name == "gpt2"
        print("✅ Configuration save/load works correctly")
        
        # Cleanup
        if os.path.exists(test_config_path):
            os.remove(test_config_path)
            
    except Exception as e:
        print(f"❌ Configuration save/load failed: {e}")
        return False
    
    return True


def test_directory_structure():
    """Test that the project directory structure is correct."""
    print("\n🏗️  Testing Directory Structure")
    print("=" * 50)
    
    required_dirs = [
        "src",
        "src/utils",
        "src/analysis",
        "src/data_processing",
        "src/visualization",
        "configs",
        "data",
        "tests"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")
            all_good = False
    
    return all_good


def test_dependencies():
    """Test that required dependencies are available."""
    print("\n📦 Testing Dependencies")
    print("=" * 50)
    
    required_packages = [
        "torch",
        "transformers", 
        "pandas",
        "numpy",
        "pydantic",
        "yaml",
        "tqdm"
    ]
    
    all_good = True
    for package in required_packages:
        try:
            if package == "yaml":
                import yaml
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ Missing: {package}")
            all_good = False
    
    return all_good


def main():
    """Main validation function."""
    print("NeuronMap Configuration Validation")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Dependencies
    if test_dependencies():
        tests_passed += 1
    
    # Test 2: Directory structure
    if test_directory_structure():
        tests_passed += 1
    
    # Test 3: Configuration system
    if test_config_system():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
