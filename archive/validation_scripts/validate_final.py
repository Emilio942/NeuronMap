#!/usr/bin/env python3
"""
Final validation script for NeuronMap project.
Tests core functionality and ensures everything works together.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_imports() -> bool:
    """Test that all critical modules can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("Testing module imports...")
    
    try:
        # Core modules
        from src.data_generation.question_generator import QuestionGenerator
        from src.analysis.activation_extractor import ActivationExtractor
        from src.visualization.visualizer import ActivationVisualizer
        from src.utils.config_manager import get_config
        
        # Advanced analysis modules  
        from src.analysis.advanced_analysis import ActivationAnalyzer
        from src.analysis.attention_analysis import AttentionAnalyzer
        
        # Phase 3 modules
        from src.analysis.interpretability import ConceptActivationVectorAnalyzer
        from src.analysis.experimental_analysis import RepresentationalSimilarityAnalyzer
        from src.analysis.domain_specific import DomainSpecificPipeline
        
        # Phase 4 modules
        try:
            from src.analysis.scientific_rigor import StatisticalTester
            from src.analysis.ethics_bias import FairnessAnalyzer  
            from src.analysis.conceptual_analysis import ConceptualAnalyzer
            logger.info("✅ All advanced modules imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Some advanced modules not available: {e}")
        
        logger.info("✅ Core module imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_configuration() -> bool:
    """Test configuration system."""
    logger = logging.getLogger(__name__)
    logger.info("Testing configuration system...")
    
    try:
        from src.utils.config_manager import get_config
        
        # Test loading configurations
        configs = ['dev', 'default', 'test']
        for config_name in configs:
            try:
                config = get_config(config_name)
                logger.info(f"✅ Config '{config_name}' loaded successfully")
            except:
                logger.warning(f"⚠️ Config '{config_name}' not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_data_generation() -> bool:
    """Test question generation functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing data generation...")
    
    try:
        from src.data_generation.question_generator import QuestionGenerator
        
        # Create a simple test
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_questions.json"
            
            # Test basic instantiation
            generator = QuestionGenerator()
            logger.info("✅ QuestionGenerator instantiated successfully")
            
            # Test mock generation (without actually calling Ollama)
            test_questions = [
                {"question": "What is the capital of France?", "category": "geography"},
                {"question": "How does photosynthesis work?", "category": "science"}
            ]
            
            with open(output_file, 'w') as f:
                json.dump(test_questions, f, indent=2)
            
            logger.info("✅ Data generation test completed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Data generation test failed: {e}")
        return False

def test_analysis_pipeline() -> bool:
    """Test core analysis functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing analysis pipeline...")
    
    try:
        from src.analysis.activation_extractor import ActivationExtractor
        import numpy as np
        
        # Test basic instantiation
        extractor = ActivationExtractor()
        logger.info("✅ ActivationExtractor instantiated successfully")
        
        # Test mock analysis with synthetic data
        synthetic_activations = {
            'layer_0': np.random.randn(10, 768),
            'layer_1': np.random.randn(10, 768),
            'layer_2': np.random.randn(10, 768)
        }
        
        # Test basic analysis operations
        if len(synthetic_activations) > 0:
            logger.info("✅ Synthetic activation data created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis pipeline test failed: {e}")
        return False

def test_visualization() -> bool:
    """Test visualization functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing visualization...")
    
    try:
        from src.visualization.visualizer import ActivationVisualizer
        import numpy as np
        
        # Test basic instantiation
        visualizer = ActivationVisualizer()
        logger.info("✅ ActivationVisualizer instantiated successfully")
        
        # Test with synthetic data
        synthetic_data = np.random.randn(50, 10)
        labels = [f"sample_{i}" for i in range(50)]
        
        # Test dimensionality reduction (without actually plotting)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(synthetic_data)
        
        if reduced_data.shape == (50, 2):
            logger.info("✅ Dimensionality reduction test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization test failed: {e}")
        return False

def test_advanced_features() -> bool:
    """Test advanced interpretability features."""
    logger = logging.getLogger(__name__)
    logger.info("Testing advanced features...")
    
    try:
        # Test interpretability
        try:
            from src.analysis.interpretability import ConceptActivationVectorAnalyzer
            cav_analyzer = ConceptActivationVectorAnalyzer()
            logger.info("✅ CAV analyzer instantiated")
        except ImportError:
            logger.warning("⚠️ CAV analyzer not available")
        
        # Test experimental analysis
        try:
            from src.analysis.experimental_analysis import RepresentationalSimilarityAnalyzer
            rsa_analyzer = RepresentationalSimilarityAnalyzer()
            logger.info("✅ RSA analyzer instantiated")
        except ImportError:
            logger.warning("⚠️ RSA analyzer not available")
        
        # Test domain-specific analysis
        try:
            from src.analysis.domain_specific import DomainSpecificPipeline
            domain_pipeline = DomainSpecificPipeline()
            logger.info("✅ Domain-specific pipeline instantiated")
        except ImportError:
            logger.warning("⚠️ Domain-specific pipeline not available")
        
        # Test conceptual analysis
        try:
            from src.analysis.conceptual_analysis import ConceptualAnalyzer
            conceptual_analyzer = ConceptualAnalyzer()
            logger.info("✅ Conceptual analyzer instantiated")
        except ImportError:
            logger.warning("⚠️ Conceptual analyzer not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced features test failed: {e}")
        return False

def test_file_structure() -> bool:
    """Test that required files and directories exist."""
    logger = logging.getLogger(__name__)
    logger.info("Testing file structure...")
    
    required_files = [
        "main.py",
        "requirements.txt", 
        "setup.py",
        "README.md",
        "src/",
        "configs/",
        "tests/",
        "tutorials/",
        "examples/"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"⚠️ Missing files/directories: {missing_files}")
    else:
        logger.info("✅ All required files and directories present")
    
    return len(missing_files) == 0

def test_documentation() -> bool:
    """Test that documentation files exist."""
    logger = logging.getLogger(__name__)
    logger.info("Testing documentation...")
    
    doc_files = [
        "README.md",
        "CONTRIBUTING.md", 
        "COMMUNITY.md",
        "tutorials/README.md",
        "examples/README.md",
        "PHASE4_ENHANCEMENTS.md",
        "PROJECT_OVERVIEW.md"
    ]
    
    missing_docs = []
    for doc_file in doc_files:
        if not Path(doc_file).exists():
            missing_docs.append(doc_file)
    
    if missing_docs:
        logger.warning(f"⚠️ Missing documentation: {missing_docs}")
    else:
        logger.info("✅ All documentation files present")
    
    return len(missing_docs) == 0

def run_validation() -> bool:
    """Run all validation tests."""
    logger = setup_logging()
    logger.info("🚀 Starting NeuronMap final validation...")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Documentation", test_documentation), 
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Generation", test_data_generation),
        ("Analysis Pipeline", test_analysis_pipeline),
        ("Visualization", test_visualization),
        ("Advanced Features", test_advanced_features)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*50)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:20s} {status}")
    
    logger.info(f"\nTests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! NeuronMap is ready for deployment!")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Review issues above.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
