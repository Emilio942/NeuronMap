#!/usr/bin/env python3
"""Test script to verify CLI integration for interpretability and experimental analysis."""

import sys
import importlib.util
from pathlib import Path

def test_cli_integration():
    """Test that the CLI commands are properly integrated."""
    
    print("Testing CLI Integration for NeuronMap...")
    
    # Check if main.py can be parsed without import errors
    main_file = Path("main.py")
    if not main_file.exists():
        print("❌ main.py not found")
        return False
    
    # Read and parse the main.py file to check command structure
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Check for required command functions
    required_commands = [
        "cmd_interpretability_analysis",
        "cmd_experimental_analysis", 
        "cmd_create_probing_dataset",
        "cmd_advanced_experimental",
        "cmd_domain_analysis",
        "cmd_ethics_analysis",
        "cmd_conceptual_analysis"
    ]
    
    missing_commands = []
    for cmd in required_commands:
        if f"def {cmd}(" not in content:
            missing_commands.append(cmd)
    
    if missing_commands:
        print(f"❌ Missing command functions: {missing_commands}")
        return False
    else:
        print("✅ All required command functions found")
    
    # Check for argument parsers
    required_parsers = [
        'interpret_parser = subparsers.add_parser("interpret"',
        'experiment_parser = subparsers.add_parser("experiment"',
        'probe_parser = subparsers.add_parser("probe"',
        'advanced_parser = subparsers.add_parser("advanced"',
        'domain_parser = subparsers.add_parser("domain"',
        'ethics_parser = subparsers.add_parser("ethics"',
        'conceptual_parser = subparsers.add_parser("conceptual"'
    ]
    
    missing_parsers = []
    for parser in required_parsers:
        if parser not in content:
            missing_parsers.append(parser)
    
    if missing_parsers:
        print(f"❌ Missing argument parsers: {missing_parsers}")
        return False
    else:
        print("✅ All required argument parsers found")
    
    # Check command dispatch dictionary
    command_mappings = [
        '"interpret": cmd_interpretability_analysis',
        '"experiment": cmd_experimental_analysis',
        '"probe": cmd_create_probing_dataset',
        '"advanced": cmd_advanced_experimental',
        '"domain": cmd_domain_analysis',
        '"ethics": cmd_ethics_analysis',
        '"conceptual": cmd_conceptual_analysis'
    ]
    
    missing_mappings = []
    for mapping in command_mappings:
        if mapping not in content:
            missing_mappings.append(mapping)
    
    if missing_mappings:
        print(f"❌ Missing command mappings: {missing_mappings}")
        return False
    else:
        print("✅ All command mappings found")
    
    # Check if interpretability module exists
    interpretability_file = Path("src/analysis/interpretability.py")
    if not interpretability_file.exists():
        print("❌ Interpretability module not found")
        return False
    else:
        print("✅ Interpretability module exists")
    
    # Check if experimental analysis module exists
    experimental_file = Path("src/analysis/experimental_analysis.py")
    if not experimental_file.exists():
        print("❌ Experimental analysis module not found")
        return False
    else:
        print("✅ Experimental analysis module exists")
    
    # Check if advanced experimental module exists
    advanced_experimental_file = Path("src/analysis/advanced_experimental.py")
    if not advanced_experimental_file.exists():
        print("❌ Advanced experimental module not found")
        return False
    else:
        print("✅ Advanced experimental module exists")
    
    # Check if domain-specific module exists
    domain_specific_file = Path("src/analysis/domain_specific.py")
    if not domain_specific_file.exists():
        print("❌ Domain-specific analysis module not found")
        return False
    else:
        print("✅ Domain-specific analysis module exists")
    
    # Check if conceptual analysis module exists
    conceptual_file = Path("src/analysis/conceptual_analysis.py")
    if not conceptual_file.exists():
        print("❌ Conceptual analysis module not found")
        return False
    else:
        print("✅ Conceptual analysis module exists")
    
    # Check if ethics and bias module exists
    ethics_file = Path("src/analysis/ethics_bias.py")
    if not ethics_file.exists():
        print("❌ Ethics and bias analysis module not found")
        return False
    else:
        print("✅ Ethics and bias analysis module exists")
    
    print("\n🎉 CLI Integration Test PASSED!")
    print("\nNew CLI Commands Available:")
    print("  python main.py interpret --model gpt2 --layer transformer.h.6")
    print("  python main.py experiment --input-file data/activations.h5")
    print("  python main.py probe --input-file data/texts.txt --create-sentiment")
    print("  python main.py advanced --model gpt2 --input-file data/texts.txt --analysis-types adversarial counterfactual")
    print("  python main.py domain --analysis-type code --model gpt2 --input-file code.txt")
    print("  python main.py ethics --model gpt2 --texts-file texts.txt --groups-file groups.txt")
    print("  python main.py conceptual --analysis-type concepts --model gpt2 --input-file data.json")
    
    return True

if __name__ == "__main__":
    success = test_cli_integration()
    sys.exit(0 if success else 1)
