#!/usr/bin/env python3
"""Configuration validation script for NeuronMap."""

import sys
import logging
from pathlib import Path
import argparse
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from utils.config_manager import (
        ConfigManager, 
        NeuronMapConfig,
        get_config,
        load_config
    )
    ENHANCED_CONFIG = True
except ImportError:
    print("Error: Could not import enhanced configuration. Falling back to basic validation.")
    ENHANCED_CONFIG = False


def setup_logging(verbose: bool = False):
    """Setup logging for validation script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_individual_configs(config_manager: ConfigManager) -> Dict[str, bool]:
    """Validate individual configuration files.
    
    Args:
        config_manager: ConfigManager instance.
        
    Returns:
        Dictionary with validation results for each config type.
    """
    results = {}
    
    # Validate models config
    try:
        models = config_manager.get_models_config()
        results['models'] = True
        print(f"✓ Models config valid - {len(models)} models configured")
        for model_name in models.keys():
            print(f"  - {model_name}")
    except Exception as e:
        results['models'] = False
        print(f"✗ Models config error: {e}")
    
    # Validate analysis config
    try:
        analysis = config_manager.get_analysis_config()
        results['analysis'] = True
        print(f"✓ Analysis config valid - batch_size: {analysis.batch_size}, device: {analysis.device}")
    except Exception as e:
        results['analysis'] = False
        print(f"✗ Analysis config error: {e}")
    
    # Validate visualization config
    try:
        viz = config_manager.get_visualization_config()
        results['visualization'] = True
        print(f"✓ Visualization config valid - {viz.figure_width}x{viz.figure_height}, {viz.color_scheme}")
    except Exception as e:
        results['visualization'] = False
        print(f"✗ Visualization config error: {e}")
    
    # Validate environment config
    try:
        env = config_manager.get_environment_config()
        results['environment'] = True
        print(f"✓ Environment config valid - {env.environment} mode, log level: {env.log_level}")
    except Exception as e:
        results['environment'] = False
        print(f"✗ Environment config error: {e}")
    
    return results


def check_hardware_compatibility(config_manager: ConfigManager) -> None:
    """Check hardware compatibility for configured models."""
    print("\n🔧 Hardware Compatibility Check:")
    
    try:
        models = config_manager.get_models_config()
        
        for model_name in models.keys():
            try:
                compatibility = config_manager.check_hardware_compatibility(model_name)
                
                status_icons = {
                    True: "✓",
                    False: "✗"
                }
                
                print(f"\n  Model: {model_name}")
                print(f"    RAM sufficient: {status_icons[compatibility['sufficient_ram']]}")
                print(f"    GPU available: {status_icons[compatibility['gpu_available']]}")
                print(f"    GPU memory sufficient: {status_icons[compatibility['sufficient_gpu_memory']]}")
                print(f"    Device compatible: {status_icons[compatibility['device_compatible']]}")
                
                if not all(compatibility.values()):
                    print(f"    ⚠️  Warnings exist for {model_name}")
                    
            except Exception as e:
                print(f"  ✗ Error checking {model_name}: {e}")
                
    except Exception as e:
        print(f"✗ Hardware compatibility check failed: {e}")


def test_device_resolution(config_manager: ConfigManager) -> None:
    """Test device resolution for different models."""
    print("\n🖥️  Device Resolution Test:")
    
    try:
        models = config_manager.get_models_config()
        sample_models = list(models.keys())[:3]  # Test first 3 models
        
        for model_name in sample_models:
            try:
                device = config_manager.get_device(model_name)
                print(f"  {model_name}: {device}")
            except Exception as e:
                print(f"  ✗ {model_name}: Error - {e}")
                
    except Exception as e:
        print(f"✗ Device resolution test failed: {e}")


def test_output_paths(config_manager: ConfigManager) -> None:
    """Test output path creation."""
    print("\n📁 Output Path Creation Test:")
    
    try:
        paths = config_manager.create_output_paths("validation_test")
        
        for path_type, path in paths.items():
            if path.exists():
                print(f"  ✓ {path_type}: {path}")
            else:
                print(f"  ✗ {path_type}: {path} (not created)")
                
        # Cleanup test directories
        import shutil
        test_dir = Path("data/outputs/validation_test")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("  🧹 Cleaned up test directories")
            
    except Exception as e:
        print(f"✗ Output path test failed: {e}")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate NeuronMap configuration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--config-dir", type=str, help="Custom configuration directory")
    parser.add_argument("--skip-hardware", action="store_true", help="Skip hardware compatibility check")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("🔍 NeuronMap Configuration Validation")
    print("=" * 40)
    
    # Initialize configuration manager
    try:
        if args.config_dir:
            config_manager = ConfigManager(args.config_dir)
        else:
            config_manager = get_config_manager()
        print("✓ Configuration manager initialized")
    except Exception as e:
        print(f"✗ Failed to initialize configuration manager: {e}")
        sys.exit(1)
    
    # Validate startup configuration
    print("\n📋 Startup Configuration Validation:")
    startup_valid = validate_startup_config()
    if startup_valid:
        print("✓ All startup configurations are valid")
    else:
        print("✗ Startup configuration validation failed")
    
    # Validate individual configurations
    print("\n🔧 Individual Configuration Validation:")
    individual_results = validate_individual_configs(config_manager)
    
    # Hardware compatibility check
    if not args.skip_hardware:
        check_hardware_compatibility(config_manager)
    
    # Device resolution test
    test_device_resolution(config_manager)
    
    # Output paths test
    test_output_paths(config_manager)
    
    # Summary
    print("\n📊 Validation Summary:")
    print("=" * 30)
    
    all_valid = all(individual_results.values()) and startup_valid
    
    if all_valid:
        print("🎉 All configurations are valid!")
        print("✓ NeuronMap is ready to use")
        return_code = 0
    else:
        print("⚠️  Some configuration issues found")
        print("✗ Please fix the errors above")
        return_code = 1
    
    # Configuration info
    print(f"\nConfiguration files location: {config_manager.config_dir}")
    print(f"Available models: {len(config_manager.get_models_config())}")
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()
