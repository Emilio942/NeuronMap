#!/usr/bin/env python3
"""
Comprehensive test runner for NeuronMap project.

This script runs the complete test suite according to Section 8.1 requirements:
- >98% code coverage
- Property-based testing
- Performance benchmarks
- Automated test data management
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner for NeuronMap."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.src_dir = project_root / "src"
        self.coverage_target = 85  # Start with 85%, work towards 98%
        
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with coverage reporting."""
        print("🧪 Running unit tests with coverage...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--cov", str(self.src_dir),
            "--cov-report", "term-missing",
            "--cov-report", "html:htmlcov",
            "--cov-report", "xml",
            f"--cov-fail-under={self.coverage_target}",
            "--tb=short",
            "-v",
            "--durations=10",
            "--hypothesis-show-statistics"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test suite timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run tests: {e}"
            }
    
    def run_property_based_tests(self) -> Dict[str, Any]:
        """Run property-based tests specifically."""
        print("🔍 Running property-based tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-m", "hypothesis",
            "--hypothesis-show-statistics",
            "--hypothesis-verbosity=verbose",
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run property-based tests: {e}"
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("⚡ Running performance tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-m", "slow",
            "--tb=short",
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run performance tests: {e}"
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-m", "integration",
            "--tb=short",
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run integration tests: {e}"
            }
    
    def validate_test_environment(self) -> Dict[str, Any]:
        """Validate that test environment is properly set up."""
        print("🔧 Validating test environment...")
        
        issues = []
        
        # Check pytest installation
        try:
            import pytest
        except ImportError:
            issues.append("pytest not installed")
        
        # Check hypothesis installation
        try:
            import hypothesis
        except ImportError:
            issues.append("hypothesis not installed")
        
        # Check coverage installation
        try:
            import coverage
        except ImportError:
            issues.append("coverage not installed")
        
        # Check test files exist
        test_files = list(self.test_dir.glob("test_*.py"))
        if len(test_files) < 5:
            issues.append(f"Only {len(test_files)} test files found, expected at least 5")
        
        # Check source files exist
        src_files = list(self.src_dir.rglob("*.py"))
        if len(src_files) < 10:
            issues.append(f"Only {len(src_files)} source files found, expected at least 10")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "test_files_count": len(test_files) if 'test_files' in locals() else 0,
            "src_files_count": len(src_files) if 'src_files' in locals() else 0
        }
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("# NeuronMap Test Suite Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Environment validation
        env_result = results.get("environment", {})
        if env_result.get("success"):
            report.append("✅ Test environment validation: PASSED")
            report.append(f"   - Test files: {env_result.get('test_files_count', 0)}")
            report.append(f"   - Source files: {env_result.get('src_files_count', 0)}")
        else:
            report.append("❌ Test environment validation: FAILED")
            for issue in env_result.get("issues", []):
                report.append(f"   - {issue}")
        report.append("")
        
        # Unit tests
        unit_result = results.get("unit_tests", {})
        if unit_result.get("success"):
            report.append("✅ Unit tests: PASSED")
        else:
            report.append("❌ Unit tests: FAILED")
            if "error" in unit_result:
                report.append(f"   Error: {unit_result['error']}")
        report.append("")
        
        # Property-based tests
        prop_result = results.get("property_tests", {})
        if prop_result.get("success"):
            report.append("✅ Property-based tests: PASSED")
        else:
            report.append("❌ Property-based tests: FAILED")
            if "error" in prop_result:
                report.append(f"   Error: {prop_result['error']}")
        report.append("")
        
        # Performance tests
        perf_result = results.get("performance_tests", {})
        if perf_result.get("success"):
            report.append("✅ Performance tests: PASSED")
        else:
            report.append("❌ Performance tests: FAILED")
            if "error" in perf_result:
                report.append(f"   Error: {perf_result['error']}")
        report.append("")
        
        # Integration tests
        int_result = results.get("integration_tests", {})
        if int_result.get("success"):
            report.append("✅ Integration tests: PASSED")
        else:
            report.append("❌ Integration tests: FAILED")
            if "error" in int_result:
                report.append(f"   Error: {int_result['error']}")
        report.append("")
        
        # Summary
        total_tests = 4
        passed_tests = sum(1 for key in ["unit_tests", "property_tests", "performance_tests", "integration_tests"] 
                          if results.get(key, {}).get("success", False))
        
        report.append(f"## Summary: {passed_tests}/{total_tests} test suites passed")
        
        if passed_tests == total_tests:
            report.append("🎉 All test suites passed successfully!")
        else:
            report.append("⚠️  Some test suites failed. Check details above.")
        
        return "\n".join(report)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report."""
        print("🚀 Starting comprehensive test suite...")
        start_time = time.time()
        
        results = {}
        
        # 1. Validate environment
        results["environment"] = self.validate_test_environment()
        if not results["environment"]["success"]:
            print("❌ Test environment validation failed. Stopping.")
            return results
        
        # 2. Run unit tests
        results["unit_tests"] = self.run_unit_tests()
        
        # 3. Run property-based tests
        results["property_tests"] = self.run_property_based_tests()
        
        # 4. Run performance tests
        results["performance_tests"] = self.run_performance_tests()
        
        # 5. Run integration tests
        results["integration_tests"] = self.run_integration_tests()
        
        elapsed_time = time.time() - start_time
        results["total_time"] = elapsed_time
        
        # Generate report
        report = self.generate_test_report(results)
        
        # Save report
        report_file = self.project_root / "test_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Test suite completed in {elapsed_time:.2f} seconds")
        print(f"📝 Report saved to: {report_file}")
        print("\n" + "="*60)
        print(report)
        
        return results


def main():
    """Main entry point for test runner."""
    project_root = Path(__file__).parent
    runner = TestRunner(project_root)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run NeuronMap test suite")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--property-only", action="store_true", help="Run only property-based tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage-target", type=int, default=85, help="Coverage target percentage")
    
    args = parser.parse_args()
    
    # Set coverage target
    runner.coverage_target = args.coverage_target
    
    # Run specific test suites if requested
    if args.unit_only:
        result = runner.run_unit_tests()
        print("Unit tests result:", "PASSED" if result["success"] else "FAILED")
        return 0 if result["success"] else 1
    
    if args.property_only:
        result = runner.run_property_based_tests()
        print("Property-based tests result:", "PASSED" if result["success"] else "FAILED")
        return 0 if result["success"] else 1
    
    if args.performance_only:
        result = runner.run_performance_tests()
        print("Performance tests result:", "PASSED" if result["success"] else "FAILED")
        return 0 if result["success"] else 1
    
    if args.integration_only:
        result = runner.run_integration_tests()
        print("Integration tests result:", "PASSED" if result["success"] else "FAILED")
        return 0 if result["success"] else 1
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Return exit code based on results
    all_passed = all(
        results.get(key, {}).get("success", False)
        for key in ["unit_tests", "property_tests", "performance_tests", "integration_tests"]
    )
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
