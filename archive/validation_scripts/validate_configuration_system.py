#!/usr/bin/env python3
"""
Configuration Validation Script for NeuronMap
============================================

This script validates all configuration files and checks hardware compatibility
as required by aufgabenliste.md Task 1.2.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils.config_manager import ConfigManager, get_config_manager
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def main():
    """Main validation function."""
    print("🔍 NeuronMap Configuration Validation")
    print("=" * 50)
    
    # Validate all configuration files
    print("\n📁 Validating all configuration files...")
    config_results = validate_all_configs()
    
    total_errors = 0
    for config_name, errors in config_results.items():
        if errors:
            total_errors += len(errors)
            print(f"\n❌ {config_name}.yaml:")
            for error in errors:
                print(f"   - {error.field}: {error.message}")
                if error.suggestion:
                    print(f"     💡 Suggestion: {error.suggestion}")
        else:
            print(f"✅ {config_name}.yaml: Valid")
    
    # Hardware compatibility validation
    print(f"\n🖥️  Validating hardware compatibility...")
    try:
        # Ensure configuration is loaded for hardware validation
        if not config_manager._config:
            config_manager.load_config()
        
        hardware_errors = validate_hardware_compatibility()
        if hardware_errors:
            total_errors += len(hardware_errors)
            print("❌ Hardware compatibility issues:")
            for error in hardware_errors:
                print(f"   - {error.field}: {error.message}")
                if error.suggestion:
                    print(f"     💡 Suggestion: {error.suggestion}")
        else:
            print("✅ Hardware compatibility: OK")
    except Exception as e:
        print(f"⚠️  Hardware validation error: {e}")
        total_errors += 1
    
    # Test configuration loading
    print(f"\n⚙️  Testing configuration loading...")
    try:
        config = config_manager.load_config()
        validation_errors = config_manager.validate_config()
        if validation_errors:
            total_errors += len(validation_errors)
            print("❌ Configuration validation errors:")
            for error in validation_errors:
                print(f"   - {error.field}: {error.message}")
        else:
            print("✅ Configuration loading: OK")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        total_errors += 1
    
    # Environment switching test
    print(f"\n🔄 Testing environment switching...")
    environments = ["development", "testing", "production"]
    for env in environments:
        try:
            config_manager.set_environment(env)
            print(f"✅ {env}: OK")
        except Exception as e:
            print(f"❌ {env}: {e}")
            total_errors += 1
    
    # Summary
    print(f"\n📊 Validation Summary")
    print("=" * 30)
    if total_errors == 0:
        print("🎉 All validations passed!")
        print("✅ Configuration system is working correctly")
        return 0
    else:
        print(f"❌ Found {total_errors} validation errors")
        print("🔧 Please fix the issues above before proceeding")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
