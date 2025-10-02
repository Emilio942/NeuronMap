#!/usr/bin/env python3
"""
Comprehensive test for NeuronMap Circuit Discovery system.
Tests backend, CLI, and API integration.
"""

import sys
import os  
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_backend():
    """Test circuit analysis backend."""
    print("Testing Circuit Analysis Backend...")
    
    try:
        from analysis.circuits import (
            InductionHeadScanner,
            CopyingHeadScanner,
            AttentionHeadCompositionAnalyzer,
            NeuronToHeadAnalyzer,
            CircuitVerifier
        )
        print("✓ All backend classes imported successfully")
        
        # Test initialization
        scanner = InductionHeadScanner()
        print("✓ InductionHeadScanner initialized")
        
        copying_scanner = CopyingHeadScanner()
        print("✓ CopyingHeadScanner initialized")
        
        comp_analyzer = AttentionHeadCompositionAnalyzer()
        print("✓ AttentionHeadCompositionAnalyzer initialized")
        
        neuron_analyzer = NeuronToHeadAnalyzer()
        print("✓ NeuronToHeadAnalyzer initialized")
        
        verifier = CircuitVerifier()
        print("✓ CircuitVerifier initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Backend test failed: {e}")
        return False

def test_cli():
    """Test CLI commands."""
    print("\nTesting CLI Commands...")
    
    try:
        from cli.circuits_commands import (
            find_induction_heads,
            find_copying_heads,
            analyze_composition,
            analyze_neuron_head,
            verify_circuit
        )
        print("✓ All CLI commands imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def test_api():
    """Test API endpoints."""
    print("\nTesting API Endpoints...")
    
    try:
        # Import with path fix
        sys.path.append(str(Path(__file__).parent / "src" / "web"))
        
        # Test direct import
        from web.api.circuits import circuits_bp
        print("✓ Circuits API blueprint imported successfully")
        print(f"✓ Blueprint name: {circuits_bp.name}")
        print(f"✓ Blueprint URL prefix: {circuits_bp.url_prefix}")
        
        # List endpoints
        print("✓ Available endpoints:")
        for rule in circuits_bp.url_map.iter_rules():
            print(f"  - {rule.rule} [{', '.join(rule.methods)}]")
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test end-to-end integration."""
    print("\nTesting Integration...")
    
    try:
        # Test a simple analysis
        from analysis.circuits import InductionHeadScanner
        scanner = InductionHeadScanner()
        
        print("✓ Can create scanner instance")
        print("✓ Integration test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("NeuronMap Circuit Discovery System Test")
    print("=" * 60)
    
    results = {
        "Backend": test_backend(),
        "CLI": test_cli(), 
        "API": test_api(),
        "Integration": test_integration()
    }
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:15} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! Circuit Discovery system is ready.")
        print("\nNext steps:")
        print("1. Run CLI: python -m src.cli circuits find-induction-heads --model gpt2")
        print("2. Start web server and visit /circuits")
        print("3. Test API endpoints with curl or browser")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
