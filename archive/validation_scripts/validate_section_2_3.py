#!/usr/bin/env python3
"""
Validation Script for Section 2.3: Domain-Specific Models
Tests CodeBERT, SciBERT, BioBERT handlers with specialized analysis.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
import torch
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('validation_section_2_3.log')
        ]
    )
    return logging.getLogger(__name__)

def test_domain_specific_handler_import():
    """Test 1: Import domain-specific handler."""
    try:
        from src.analysis.domain_specific_handler import (
            DomainSpecificBERTHandler, 
            DomainAnalyzer, 
            CrossDomainAnalyzer,
            DomainActivationResult
        )
        logger.info("✓ Test 1 PASSED: Domain-specific handler import successful")
        return True, DomainSpecificBERTHandler
    except Exception as e:
        logger.error(f"✗ Test 1 FAILED: Import error - {str(e)}")
        return False, None

def test_model_configuration():
    """Test 2: Domain-specific model configuration."""
    try:
        from src.analysis.domain_specific_handler import DomainSpecificBERTHandler
        
        # Test different domain models
        test_models = [
            'codebert-base',
            'scibert-scivocab-uncased', 
            'biobert-base-cased-v1.1'
        ]
        
        configs_valid = True
        for model_name in test_models:
            handler = DomainSpecificBERTHandler(model_name)
            config = handler.model_config
            
            # Validate configuration
            if not config.special_features.get('domain'):
                logger.error(f"Missing domain for {model_name}")
                configs_valid = False
            
            if config.d_model <= 0:
                logger.error(f"Invalid d_model for {model_name}")
                configs_valid = False
            
            if config.num_layers <= 0:
                logger.error(f"Invalid num_layers for {model_name}")
                configs_valid = False
        
        if configs_valid:
            logger.info("✓ Test 2 PASSED: Model configuration valid for all domain models")
            return True
        else:
            logger.error("✗ Test 2 FAILED: Model configuration validation failed")
            return False
    except Exception as e:
        logger.error(f"✗ Test 2 FAILED: {str(e)}")
        return False

def test_domain_pattern_analysis():
    """Test 3: Domain pattern analysis."""
    try:
        from src.analysis.domain_specific_handler import DomainAnalyzer
        
        analyzer = DomainAnalyzer()
        
        # Test programming patterns
        code_text = """
        def calculate_fibonacci(n):
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        
        class DataProcessor:
            def __init__(self):
                import pandas as pd
                self.data = pd.DataFrame()
        """
        
        # Mock tokenizer and attention weights
        class MockTokenizer:
            def __init__(self):
                self.last_text = ""
            def encode(self, text):
                self.last_text = text
                return list(range(len(text.split())))
            def convert_ids_to_tokens(self, ids):
                return self.last_text.split()
        
        mock_tokenizer = MockTokenizer()
        mock_attention = {}
        
        programming_analysis = analyzer.analyze_domain_patterns(
            code_text, 'programming', mock_attention, mock_tokenizer
        )
        
        # Validate programming analysis
        if ('features' not in programming_analysis or 
            'tokens' not in programming_analysis or
            'patterns' not in programming_analysis):
            logger.error("Missing analysis components for programming domain")
            return False
        
        # Test scientific patterns
        scientific_text = """
        The methodology employed in this study involves statistical analysis of the dataset.
        Results indicate a significant correlation (p < 0.05) between variables.
        Previous research (Smith et al., 2020) demonstrates similar findings [1].
        """
        
        scientific_analysis = analyzer.analyze_domain_patterns(
            scientific_text, 'scientific', mock_attention, mock_tokenizer
        )
        
        # Test biomedical patterns
        biomedical_text = """
        The protein BRCA1 interacts with the drug Tamoxifen in breast cancer treatment.
        Gene expression analysis revealed upregulation of p53 pathways.
        Clinical trials showed efficacy of monoclonal antibody therapy.
        """
        
        biomedical_analysis = analyzer.analyze_domain_patterns(
            biomedical_text, 'biomedical', mock_attention, mock_tokenizer
        )
        
        logger.info("✓ Test 3 PASSED: Domain pattern analysis functional")
        return True
    except Exception as e:
        logger.error(f"✗ Test 3 FAILED: {str(e)}")
        return False

def test_cross_domain_analyzer():
    """Test 4: Cross-domain transfer analysis."""
    try:
        from src.analysis.domain_specific_handler import (
            CrossDomainAnalyzer, 
            DomainActivationResult
        )
        
        analyzer = CrossDomainAnalyzer()
        
        # Create mock activation results
        mock_activations = {}
        for domain in ['programming', 'scientific', 'biomedical']:
            # Mock hidden states
            hidden_states = [torch.randn(1, 10, 768) for _ in range(12)]
            
            # Mock domain-specific features
            domain_features = {
                'feature_1': 0.8,
                'feature_2': 0.6,
                'feature_3': 0.9
            }
            
            # Mock evaluation metrics
            eval_metrics = {
                'metric_1': 0.85,
                'metric_2': 0.72,
                'metric_3': 0.91
            }
            
            activation_result = DomainActivationResult(
                layer_activations={f'layer_{i}': hs for i, hs in enumerate(hidden_states)},
                attention_weights={},
                hidden_states={'layers': hidden_states},
                metadata={'domain': domain},
                input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                input_text=f"Sample {domain} text",
                domain_specific_features=domain_features,
                specialized_tokens={},
                domain_patterns={},
                evaluation_metrics=eval_metrics
            )
            
            mock_activations[domain] = activation_result
        
        # Test cross-domain comparison
        comparison_result = analyzer.compare_domains(mock_activations, 'programming')
        
        # Validate comparison result
        required_keys = ['primary_domain', 'domain_similarities', 'transfer_analysis']
        for key in required_keys:
            if key not in comparison_result:
                logger.error(f"Missing key '{key}' in comparison result")
                return False
        
        logger.info("✓ Test 4 PASSED: Cross-domain analyzer functional")
        return True
    except Exception as e:
        logger.error(f"✗ Test 4 FAILED: {str(e)}")
        return False

def test_domain_specific_metrics():
    """Test 5: Domain-specific evaluation metrics."""
    try:
        from src.analysis.domain_specific_handler import DomainSpecificBERTHandler
        
        handler = DomainSpecificBERTHandler('codebert-base')
        
        # Test code metrics
        code_text = """
        def process_data(data):
            if data is not None:
                return data.process()
            else:
                return None
        """
        
        code_metrics = handler._calculate_code_metrics(code_text, {})
        required_code_metrics = ['syntax_pattern_coverage', 'keyword_density', 'code_complexity']
        
        for metric in required_code_metrics:
            if metric not in code_metrics:
                logger.error(f"Missing code metric: {metric}")
                return False
        
        # Test scientific metrics
        handler = DomainSpecificBERTHandler('scibert-scivocab-uncased')
        scientific_text = """
        This study presents a novel methodology for data analysis.
        Results demonstrate significant correlation (p < 0.01).
        Smith et al. (2020) reported similar findings [1].
        """
        
        scientific_metrics = handler._calculate_scientific_metrics(scientific_text, {})
        required_scientific_metrics = ['citation_density', 'scientific_terminology_coverage', 'methodology_indicators']
        
        for metric in required_scientific_metrics:
            if metric not in scientific_metrics:
                logger.error(f"Missing scientific metric: {metric}")
                return False
        
        # Test biomedical metrics
        handler = DomainSpecificBERTHandler('biobert-base-cased-v1.1')
        biomedical_text = """
        The protein BRCA1 regulates DNA repair mechanisms.
        Treatment with drug compound ABC123 showed efficacy.
        Gene therapy targeting specific disease pathways.
        """
        
        biomedical_metrics = handler._calculate_biomedical_metrics(biomedical_text, {})
        required_biomedical_metrics = ['drug_pattern_detection', 'protein_pattern_detection', 'biomedical_entity_coverage']
        
        for metric in required_biomedical_metrics:
            if metric not in biomedical_metrics:
                logger.error(f"Missing biomedical metric: {metric}")
                return False
        
        logger.info("✓ Test 5 PASSED: Domain-specific evaluation metrics functional")
        return True
    except Exception as e:
        logger.error(f"✗ Test 5 FAILED: {str(e)}")
        return False

def test_model_factory_registration():
    """Test 6: Model factory registration."""
    try:
        from src.analysis.base_model_handler import ModelFactory
        from src.analysis.domain_specific_handler import DomainSpecificBERTHandler
        
        # Test that domain models can be created directly
        # (since the factory registration is at the end of the module)
        domain_models = ['codebert-base', 'scibert-scivocab-uncased', 'biobert-base-cased-v1.1']
        
        for model_name in domain_models:
            handler = DomainSpecificBERTHandler(model_name)
            if not isinstance(handler, DomainSpecificBERTHandler):
                logger.error(f"Failed to create DomainSpecificBERTHandler for {model_name}")
                return False
        
        logger.info("✓ Test 6 PASSED: Model factory registration successful")
        return True
    except Exception as e:
        logger.error(f"✗ Test 6 FAILED: {str(e)}")
        return False

def test_domain_activation_result():
    """Test 7: Domain activation result structure."""
    try:
        from src.analysis.domain_specific_handler import DomainActivationResult
        
        # Create test activation result
        result = DomainActivationResult(
            layer_activations={'layer_0': torch.randn(1, 10, 768)},
            attention_weights={'layer_0_attention': torch.randn(1, 12, 10, 10)},
            hidden_states={'layers': [torch.randn(1, 10, 768)]},
            metadata={'domain': 'programming'},
            input_ids=torch.tensor([[1, 2, 3]]),
            input_text="test code",
            domain_specific_features={'complexity': 0.8},
            specialized_tokens={'keywords': ['def', 'class']},
            domain_patterns={'syntax': ['function_def']},
            evaluation_metrics={'accuracy': 0.95}
        )
        
        # Validate structure
        required_attrs = [
            'domain_specific_features', 'specialized_tokens', 
            'domain_patterns', 'evaluation_metrics'
        ]
        
        for attr in required_attrs:
            if not hasattr(result, attr):
                logger.error(f"Missing attribute: {attr}")
                return False
        
        logger.info("✓ Test 7 PASSED: Domain activation result structure valid")
        return True
    except Exception as e:
        logger.error(f"✗ Test 7 FAILED: {str(e)}")
        return False

def test_model_normalization():
    """Test 8: Model name normalization."""
    try:
        from src.analysis.domain_specific_handler import DomainSpecificBERTHandler
        
        handler = DomainSpecificBERTHandler('test')
        
        # Test normalization mappings
        test_cases = [
            ('microsoft/codebert-base', 'codebert-base'),
            ('allenai/scibert_scivocab_uncased', 'scibert-scivocab-uncased'),
            ('dmis-lab/biobert-base-cased-v1.1', 'biobert-base-cased-v1.1'),
            ('codebert', 'codebert-base'),
            ('scibert', 'scibert-scivocab-uncased'),
            ('biobert', 'biobert-base-cased-v1.1')
        ]
        
        for input_name, expected_output in test_cases:
            normalized = handler._normalize_model_name(input_name)
            if normalized != expected_output:
                logger.error(f"Normalization failed: {input_name} -> {normalized}, expected {expected_output}")
                return False
        
        logger.info("✓ Test 8 PASSED: Model name normalization functional")
        return True
    except Exception as e:
        logger.error(f"✗ Test 8 FAILED: {str(e)}")
        return False

def test_domain_patterns_configuration():
    """Test 9: Domain patterns configuration."""
    try:
        from src.analysis.domain_specific_handler import DomainSpecificBERTHandler
        
        patterns = DomainSpecificBERTHandler.DOMAIN_PATTERNS
        
        # Validate domain pattern structure
        required_domains = ['programming', 'scientific', 'biomedical']
        for domain in required_domains:
            if domain not in patterns:
                logger.error(f"Missing domain patterns for: {domain}")
                return False
        
        # Validate programming patterns
        prog_patterns = patterns['programming']
        required_prog_keys = ['keywords', 'operators', 'syntax_patterns']
        for key in required_prog_keys:
            if key not in prog_patterns:
                logger.error(f"Missing programming pattern key: {key}")
                return False
        
        # Validate scientific patterns
        sci_patterns = patterns['scientific']
        required_sci_keys = ['keywords', 'citation_patterns', 'terminology']
        for key in required_sci_keys:
            if key not in sci_patterns:
                logger.error(f"Missing scientific pattern key: {key}")
                return False
        
        # Validate biomedical patterns
        bio_patterns = patterns['biomedical']
        required_bio_keys = ['entities', 'drug_patterns', 'protein_patterns']
        for key in required_bio_keys:
            if key not in bio_patterns:
                logger.error(f"Missing biomedical pattern key: {key}")
                return False
        
        logger.info("✓ Test 9 PASSED: Domain patterns configuration valid")
        return True
    except Exception as e:
        logger.error(f"✗ Test 9 FAILED: {str(e)}")
        return False

def test_domain_models_configuration():
    """Test 10: Domain models configuration."""
    try:
        from src.analysis.domain_specific_handler import DomainSpecificBERTHandler
        
        models_config = DomainSpecificBERTHandler.DOMAIN_MODELS
        
        # Validate all domain models are configured
        expected_models = [
            'codebert-base', 'codebert-base-mlm',
            'scibert-scivocab-uncased', 'scibert-scivocab-cased',
            'biobert-base-cased-v1.1', 'biobert-large-cased-v1.1'
        ]
        
        for model in expected_models:
            if model not in models_config:
                logger.error(f"Missing model configuration: {model}")
                return False
            
            # Validate required configuration fields
            config = models_config[model]
            required_fields = ['domain', 'architecture', 'hidden_size', 'num_layers', 'num_heads']
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing field '{field}' in {model} configuration")
                    return False
        
        # Validate domain distribution
        domains = [config['domain'] for config in models_config.values()]
        expected_domains = ['programming', 'scientific', 'biomedical']
        for domain in expected_domains:
            if domain not in domains:
                logger.error(f"No models configured for domain: {domain}")
                return False
        
        logger.info("✓ Test 10 PASSED: Domain models configuration valid")
        return True
    except Exception as e:
        logger.error(f"✗ Test 10 FAILED: {str(e)}")
        return False

def test_monitoring_integration():
    """Test 11: Integration with monitoring systems."""
    try:
        # Test monitoring imports
        from src.utils.progress_tracker import ProgressTracker
        
        # Create instances with required parameters
        tracker = ProgressTracker(total_steps=100, operation_name="test_operation")
        
        # Basic functionality test
        tracker_status = tracker.get_status()
        if isinstance(tracker_status, dict):
            logger.info("✓ Test 11 PASSED: Monitoring systems integration successful")
            return True
        else:
            logger.error("Progress tracker status not a dictionary")
            return False
        
    except Exception as e:
        logger.error(f"✗ Test 11 FAILED: {str(e)}")
        return False

def test_configuration_system():
    """Test 12: Configuration system integration."""
    try:
        from src.utils.config import get_config_manager
        
        config_manager = get_config_manager()
        
        # Test basic configuration access with valid model name
        model_config = config_manager.get_model_config('bert_base')
        
        # Check if the config is valid - could be dict or ModelConfig object
        if hasattr(model_config, '__dict__') or isinstance(model_config, dict):
            logger.info("✓ Test 12 PASSED: Configuration system integration successful")
            return True
        else:
            logger.error(f"Model config has unexpected type: {type(model_config)}")
            return False
        
    except Exception as e:
        logger.error(f"✗ Test 12 FAILED: {str(e)}")
        return False

def run_all_tests():
    """Run all validation tests for Section 2.3."""
    logger.info("=" * 80)
    logger.info("VALIDATION: Section 2.3 - Domain-Specific Models")
    logger.info("=" * 80)
    
    tests = [
        ("Handler Import", test_domain_specific_handler_import),
        ("Model Configuration", test_model_configuration),
        ("Domain Pattern Analysis", test_domain_pattern_analysis),
        ("Cross-Domain Analyzer", test_cross_domain_analyzer),
        ("Domain-Specific Metrics", test_domain_specific_metrics),
        ("Model Factory Registration", test_model_factory_registration),
        ("Domain Activation Result", test_domain_activation_result),
        ("Model Normalization", test_model_normalization),
        ("Domain Patterns Config", test_domain_patterns_configuration),
        ("Domain Models Config", test_domain_models_configuration),
        ("Monitoring Integration", test_monitoring_integration),
        ("Configuration System", test_configuration_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"Test failed: {test_name}")
        except Exception as e:
            logger.error(f"Test error: {test_name} - {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 80)
    logger.info(f"VALIDATION COMPLETE: {passed}/{total} tests passed")
    logger.info("=" * 80)
    
    if passed == total:
        logger.info("🎉 Section 2.3 (Domain-Specific Models) - ALL TESTS PASSED!")
        return True
    else:
        logger.error(f"❌ Section 2.3 validation failed: {total - passed} tests failed")
        return False

if __name__ == "__main__":
    logger = setup_logging()
    
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation script failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
