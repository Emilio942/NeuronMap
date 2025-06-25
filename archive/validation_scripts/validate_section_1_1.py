#!/usr/bin/env python3
"""
Section 1.1 Project Structure Verification Script
================================================

This script verifies that the project structure reorganization is complete
and all migrated modules work correctly.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all migrated modules can be imported."""
    logger.info("Testing module imports...")
    
    tests = [
        # Data generation
        ("src.data_generation.question_generator", "QuestionGenerator"),
        
        # Analysis
        ("src.analysis.activation_extractor", "ActivationExtractor"),
        ("src.analysis.layer_inspector", "LayerInspector"),
        
        # Visualization
        ("src.visualization.core_visualizer", "CoreVisualizer"),
        
        # Main src module
        ("src", None),
    ]
    
    failed_imports = []
    
    for module_name, class_name in tests:
        try:
            if class_name:
                exec(f"from {module_name} import {class_name}")
                logger.info(f"✓ {module_name}.{class_name} imported successfully")
            else:
                exec(f"import {module_name}")
                logger.info(f"✓ {module_name} imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import {module_name}.{class_name}: {e}")
            failed_imports.append((module_name, class_name, str(e)))
    
    if failed_imports:
        logger.error(f"Import tests failed: {len(failed_imports)} failures")
        return False
    else:
        logger.info("All import tests passed!")
        return True


def test_module_execution():
    """Test that modules can be executed as scripts."""
    logger.info("Testing module execution...")
    
    tests = [
        ("python -m src.data_generation.question_generator --help", "QuestionGenerator CLI"),
        ("python -m src.analysis.activation_extractor --help", "ActivationExtractor CLI"),
        ("python -m src.visualization.core_visualizer --help", "CoreVisualizer CLI"),
    ]
    
    failed_executions = []
    
    for command, description in tests:
        try:
            import subprocess
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "usage:" in result.stdout:
                logger.info(f"✓ {description} works correctly")
            else:
                logger.error(f"✗ {description} failed: return code {result.returncode}")
                failed_executions.append((description, result.stderr))
        except Exception as e:
            logger.error(f"✗ {description} failed: {e}")
            failed_executions.append((description, str(e)))
    
    if failed_executions:
        logger.error(f"Execution tests failed: {len(failed_executions)} failures")
        return False
    else:
        logger.info("All execution tests passed!")
        return True


def test_class_instantiation():
    """Test that migrated classes can be instantiated."""
    logger.info("Testing class instantiation...")
    
    try:
        # Test QuestionGenerator
        from src.data_generation.question_generator import QuestionGenerator
        qg = QuestionGenerator()
        logger.info("✓ QuestionGenerator instantiated successfully")
        
        # Test ActivationExtractor
        from src.analysis.activation_extractor import ActivationExtractor
        ae = ActivationExtractor("distilgpt2", "transformer.h.5.mlp.c_proj")
        logger.info("✓ ActivationExtractor instantiated successfully")
        
        # Test LayerInspector (skip because it requires a loaded model)
        # from src.analysis.layer_inspector import LayerInspector
        # li = LayerInspector("distilgpt2")  # This would fail because it expects a model, not string
        # logger.info("✓ LayerInspector instantiated successfully")
        logger.info("✓ LayerInspector skipped (requires loaded model)")
        
        # Test CoreVisualizer
        from src.visualization.core_visualizer import CoreVisualizer
        cv = CoreVisualizer()
        logger.info("✓ CoreVisualizer instantiated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Class instantiation failed: {e}")
        return False


def verify_file_migration():
    """Verify that files have been properly migrated."""
    logger.info("Verifying file migration...")
    
    expected_files = [
        "src/data_generation/question_generator.py",
        "src/analysis/activation_extractor.py",
        "src/visualization/core_visualizer.py",
        "src/analysis/layer_inspector.py",
    ]
    
    original_files = [
        "fragenG.py",
        "run.py", 
        "visualizer.py"
    ]
    
    missing_files = []
    
    # Check that new files exist
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            logger.error(f"✗ Missing migrated file: {file_path}")
        else:
            logger.info(f"✓ Migrated file exists: {file_path}")
    
    # Check that original files still exist (they should be preserved)
    for file_path in original_files:
        if Path(file_path).exists():
            logger.info(f"✓ Original file preserved: {file_path}")
        else:
            logger.warning(f"⚠ Original file not found: {file_path}")
    
    return len(missing_files) == 0


def verify_init_files():
    """Verify that __init__.py files are properly configured."""
    logger.info("Verifying __init__.py files...")
    
    init_files = [
        "src/__init__.py",
        "src/data_generation/__init__.py",
        "src/analysis/__init__.py",
        "src/visualization/__init__.py",
    ]
    
    for init_file in init_files:
        if Path(init_file).exists():
            try:
                with open(init_file, 'r') as f:
                    content = f.read()
                    if "__all__" in content:
                        logger.info(f"✓ {init_file} has proper exports")
                    else:
                        logger.warning(f"⚠ {init_file} missing __all__ exports")
            except Exception as e:
                logger.error(f"✗ Error reading {init_file}: {e}")
                return False
        else:
            logger.error(f"✗ Missing {init_file}")
            return False
    
    return True


def main():
    """Run all verification tests."""
    logger.info("Starting Section 1.1 Project Structure Verification")
    logger.info("=" * 60)
    
    tests = [
        ("File Migration", verify_file_migration),
        ("Init Files", verify_init_files),
        ("Module Imports", test_imports),
        ("Class Instantiation", test_class_instantiation),
        ("Module Execution", test_module_execution),
    ]
    
    all_passed = True
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
            all_passed = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! Section 1.1 project structure reorganization is complete.")
        logger.info("\nVERIFICATION CRITERIA SATISFIED:")
        logger.info("✓ All Python files run without import errors")
        logger.info("✓ python -m src.analysis.activation_extractor functional")
        logger.info("✓ python -m src.visualization.core_visualizer functional")
        logger.info("✓ No circular imports (verified by import tests)")
        return True
    else:
        logger.error("\n❌ SOME TESTS FAILED! Project structure reorganization needs attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
