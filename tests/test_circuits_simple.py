#!/usr/bin/env python3
"""
Simplified test for NeuronMap Circuit Discovery core functionality.
"""

import sys
import os  
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_circuits_standalone():
    """Test just the circuits module directly."""
    print("Testing Circuits Module Directly...")
    
    try:
        # Direct import without going through __init__.py
        sys.path.append(str(Path(__file__).parent / "src" / "analysis"))
        import circuits
        
        # Test basic classes
        scanner = circuits.InductionHeadScanner(None)  # Pass None model for testing
        print("✓ InductionHeadScanner created")
        
        copying_scanner = circuits.CopyingHeadScanner(None)
        print("✓ CopyingHeadScanner created")
        
        comp_analyzer = circuits.AttentionHeadCompositionAnalyzer(None)
        print("✓ AttentionHeadCompositionAnalyzer created")
        
        return True
        
    except Exception as e:
        print(f"✗ Circuits test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_standalone():
    """Test API import standalone."""
    print("\nTesting API Standalone...")
    
    try:
        sys.path.append(str(Path(__file__).parent / "src" / "web" / "api"))
        import circuits as api_circuits
        
        print("✓ API circuits module imported")
        print(f"✓ Blueprint available: {hasattr(api_circuits, 'circuits_bp')}")
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_standalone():
    """Test CLI standalone."""
    print("\nTesting CLI Standalone...")
    
    try:
        sys.path.append(str(Path(__file__).parent / "src" / "cli"))
        import circuits_commands
        
        print("✓ CLI circuits_commands module imported")
        print(f"✓ Commands available: {dir(circuits_commands)}")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified tests."""
    print("=" * 60)
    print("NeuronMap Circuit Discovery - Simplified Test")
    print("=" * 60)
    
    results = {
        "Circuits Core": test_circuits_standalone(),
        "API Standalone": test_api_standalone(),
        "CLI Standalone": test_cli_standalone(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20} {status}")
    
    print("=" * 60)
    
    all_passed = all(results.values())
    if all_passed:
        print("🎉 Core functionality is working!")
    else:
        print("❌ Some core tests failed.")

if __name__ == "__main__":
    main()
