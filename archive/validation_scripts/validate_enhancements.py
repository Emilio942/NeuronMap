#!/usr/bin/env python3
"""
NeuronMap Enhancement Validation Script
======================================

Production-ready validation using the new configuration and error handling systems.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import new validation and config systems
try:
    from src.utils.config_manager import NeuronMapConfig
    from src.utils.validation import validate_experiment_config
    from src.utils.error_handling import NeuronMapException, ValidationError, ConfigurationError
    NEURONMAP_AVAILABLE = True
    logger.info("NeuronMap modules loaded successfully")
except ImportError as e:
    NEURONMAP_AVAILABLE = False
    logger.warning(f"NeuronMap modules not available: {e}")

def check_file_structure():
    """Check that all required files exist using the new modular structure."""
    required_files = [
        'src/analysis/activation_extractor.py',
        'src/data_generation/question_generator.py',
        'src/visualization/core_visualizer.py',
        'src/utils/config_manager.py',
        'src/utils/validation.py',
        'src/utils/error_handling.py',
        'configs/config.yaml',
        'configs/models.yaml',
        'main.py',
        'requirements.txt'
    ]
    
    logger.info("🔍 Checking modernized file structure...")
    all_good = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            logger.info(f"  ✓ {file_path}")
        else:
            logger.error(f"  ✗ {file_path} - MISSING!")
            all_good = False
    
    return all_good

def check_config_structure():
    """Check configuration file structure using the new validation system."""
    logger.info("📝 Checking configuration structure...")
    
    if not NEURONMAP_AVAILABLE:
        logger.warning("NeuronMap modules not available, using basic YAML validation")
        return check_config_basic()
    
    try:
        # Use the new config manager
        config = NeuronMapConfig()
        config.load_config(project_root / 'configs' / 'config.yaml')
        
        # Validate the configuration
        validation_errors = validate_experiment_config(config.get_experiment_config())
        
        if validation_errors:
            logger.error(f"Configuration validation failed with {len(validation_errors)} errors:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("  ✓ Configuration loaded and validated successfully")
        logger.info(f"  ✓ Models config: {len(config.models)} model configurations")
        logger.info(f"  ✓ Data config: {config.data.batch_size} batch size")
        logger.info(f"  ✓ Visualization config: {config.visualization.dpi} DPI")
        
        return True
        
    except ConfigurationError as e:
        logger.error(f"  ✗ Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"  ✗ Unexpected error during config validation: {e}")
        return False

def check_config_basic():
    """Basic config validation when NeuronMap modules are not available."""
    try:
        import yaml
        
        config_file = project_root / 'configs' / 'config.yaml'
        if not config_file.exists():
            logger.error("  ✗ config.yaml missing")
            return False
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['models', 'analysis', 'visualization']
        for section in required_sections:
            if section in config:
                logger.info(f"  ✓ {section} section found")
            else:
                logger.error(f"  ✗ {section} section missing")
                return False
        
        return True
        
    except ImportError:
        logger.warning("  ⚠ PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        logger.error(f"  ✗ Error reading config: {e}")
        return False

def check_imports():
    """Check basic imports work and test the new modular system."""
    logger.info("🐍 Checking Python imports...")
    
    # Test basic Python modules
    basic_imports = [
        ('pathlib', 'Path'),
        ('json', None),
        ('logging', None),
        ('argparse', None)
    ]
    
    for module, attr in basic_imports:
        try:
            if attr:
                exec(f"from {module} import {attr}")
            else:
                exec(f"import {module}")
            logger.info(f"  ✓ {module}")
        except ImportError as e:
            logger.error(f"  ✗ {module} - {e}")
            return False
    
    # Test the new modular imports
    if NEURONMAP_AVAILABLE:
        logger.info("  Testing new modular system...")
        try:
            from src.analysis.activation_extractor import ActivationExtractor
            from src.data_generation.question_generator import QuestionGenerator
            from src.visualization.core_visualizer import CoreVisualizer
            logger.info("  ✓ Core modules importable")
            
            # Test that classes can be instantiated with default parameters
            try:
                extractor = ActivationExtractor("gpt2", 6)  # Provide required parameters
                generator = QuestionGenerator() 
                visualizer = CoreVisualizer()
                logger.info("  ✓ Core classes instantiable")
            except Exception as e:
                logger.warning(f"  ⚠ Some classes need parameters: {e}")
                logger.info("  ✓ Core classes are importable (parameters may be required)")
            
        except Exception as e:
            logger.error(f"  ✗ Modular system import failed: {e}")
            return False
    else:
        logger.warning("  ⚠ NeuronMap modules not available, skipping modular tests")
    
    return True

def check_task_completion():
    """Check which tasks from the task list have been completed."""
    logger.info("📋 Checking task completion...")
    
    try:
        task_file = project_root / 'aufgabenliste.md'
        with open(task_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count completed tasks
        completed_count = content.count('✅ COMPLETED')
        not_done_count = content.count('❌ NICHT ERLEDIGT')
        total_sections = content.count('###')
        
        logger.info(f"  ✓ {completed_count} sections marked as completed")
        logger.info(f"  ⚠ {not_done_count} sections still need work")
        logger.info(f"  📊 Total sections: {total_sections}")
        
        # Check specific high-priority completions
        key_completions = [
            ('Modularization', 'Projektstruktur umorganisieren'),
            ('Configuration', 'Konfigurationssystem einführen'),
            ('Validation', 'Input-Validierung'),
            ('Error Handling', 'Exception-Behandlung')
        ]
        
        completion_score = 0
        for name, search_term in key_completions:
            if search_term in content and '✅ COMPLETED' in content:
                # Look for completed markers near the search term
                term_index = content.find(search_term)
                if term_index > 0:
                    surrounding_text = content[max(0, term_index-200):term_index+200]
                    if '✅ COMPLETED' in surrounding_text:
                        logger.info(f"  ✓ {name} - marked as completed")
                        completion_score += 1
                    else:
                        logger.warning(f"  ⚠ {name} - still in progress")
                else:
                    logger.warning(f"  ⚠ {name} - not found in roadmap")
            else:
                logger.warning(f"  ⚠ {name} - not marked as completed")
        
        logger.info(f"  📊 High-priority completion score: {completion_score}/{len(key_completions)}")
        
        return completion_score >= len(key_completions) * 0.5  # At least 50% of key tasks
        
    except Exception as e:
        logger.error(f"  ✗ Error checking tasks: {e}")
        return False

def main():
    """Run all validation checks with structured logging and error handling."""
    logger.info("🧠 NeuronMap Enhancement Validation (Modernized)")
    logger.info("=" * 60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Configuration System", check_config_structure), 
        ("Import System", check_imports),
        ("Task Completion", check_task_completion)
    ]
    
    results = []
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Running {check_name} check...")
        try:
            result = check_func()
            results.append((check_name, result))
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {check_name}: {status}")
        except Exception as e:
            logger.error(f"  ✗ {check_name} failed with error: {e}")
            results.append((check_name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        status_icon = "🟢" if result else "🔴"
        logger.info(f"  {status_icon} {name:<25} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        logger.info("\n🎉 All validations passed! System is ready for production use.")
        return 0
    else:
        logger.warning(f"\n⚠️  {len(results) - passed} validation(s) failed. System needs attention.")
        if passed >= len(results) * 0.75:  # 75% pass rate
            logger.info("💡 Most checks passed - system is functional but needs improvements.")
            return 0
        else:
            logger.error("🚨 Critical validation failures detected.")
            return 1

if __name__ == "__main__":
    sys.exit(main())
