"""Test runner for NeuronMap test suite.

This script runs all tests and generates coverage reports.
"""

import sys
import subprocess
import os
from pathlib import Path
import time


def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Duration: {end_time - start_time:.2f}s")
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    else:
        print("❌ FAILED")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    
    return result.returncode == 0


def main():
    """Run all tests."""
    print("🧪 NeuronMap Test Suite Runner")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Import tests
    total_tests += 1
    print("\n🔍 Testing module imports...")
    if run_command("python -c \"from tests.test_comprehensive import run_tests; print('All imports successful')\"", "Module Import Test"):
        success_count += 1
    
    # Test 2: CLI integration test
    total_tests += 1
    if run_command("python test_cli_integration.py", "CLI Integration Test"):
        success_count += 1
    
    # Test 3: Configuration validation
    total_tests += 1
    if run_command("python main.py validate", "System Validation Test"):
        success_count += 1
    
    # Test 4: Config display test
    total_tests += 1
    if run_command("python main.py config --models", "Configuration Display Test"):
        success_count += 1
    
    # Test 5: Domain analysis help test
    total_tests += 1
    if run_command("python main.py domain --help", "Domain Analysis Help Test"):
        success_count += 1
    
    # Track test results
    results = {}
    
    # 1. Run unit tests
    print("\n🔬 Running Unit Tests...")
    results['unit_tests'] = run_command(
        "python -m pytest tests/ -v --tb=short",
        "Unit Tests with pytest"
    )
    
    # Fallback to unittest if pytest fails
    if not results['unit_tests']:
        print("\n🔄 Falling back to unittest...")
        results['unit_tests_fallback'] = run_command(
            "python -m unittest discover tests/ -v",
            "Unit Tests with unittest"
        )
    
    # 2. Run integration tests
    print("\n🔗 Running Integration Tests...")
    results['integration_tests'] = run_command(
        "python tests/test_core.py",
        "Integration Tests"
    )
    
    # 3. Check code quality
    print("\n🎨 Running Code Quality Checks...")
    
    # Check if flake8 is available
    results['flake8'] = run_command(
        "python -m flake8 --version && python -m flake8 src/ main.py --count --select=E9,F63,F7,F82 --show-source --statistics",
        "Flake8 Syntax Check"
    )
    
    # Check if black is available  
    results['black_check'] = run_command(
        "python -m black --version && python -m black --check --diff src/ main.py",
        "Black Code Formatting Check"
    )
    
    # 4. Run CLI integration test
    print("\n🖥️  Running CLI Integration Test...")
    results['cli_test'] = run_command(
        "python test_cli_integration.py",
        "CLI Integration Test"
    )
    
    # 5. Test imports
    print("\n📦 Testing Module Imports...")
    import_tests = [
        ("src.utils.config", "Configuration utilities"),
        ("src.data_generation.question_generator", "Question generator"),
        ("src.analysis.activation_extractor", "Activation extractor"),
        ("src.visualization.visualizer", "Visualizer"),
        ("src.analysis.interpretability", "Interpretability analysis"),
        ("src.analysis.experimental_analysis", "Experimental analysis"),
        ("src.analysis.advanced_experimental", "Advanced experimental analysis"),
        ("src.analysis.domain_specific", "Domain-specific analysis"),
    ]
    
    import_success = 0
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
            import_success += 1
        except ImportError as e:
            print(f"❌ {description}: {module_name} - {e}")
    
    results['imports'] = import_success == len(import_tests)
    
    # 6. Generate summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
        total_tests += 1
        if passed:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
