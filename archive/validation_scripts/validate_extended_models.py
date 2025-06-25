#!/usr/bin/env python3
"""
Validation script for Section 3.1 Phase 2: Extended Multi-Model Support
Validates T5, LLaMA, and Domain-Specific model integration.

Part of the NeuronMap modernization roadmap (aufgabenliste.md).
"""

import os
import sys
import logging
import torch
import importlib
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_imports():
    """Validate that all required modules can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("=== Validating Imports ===")
    
    required_modules = [
        'src.utils.multi_model_support',
        'src.utils.error_handling',
        'src.utils.robust_decorators'
    ]
    
    results = {}
    
    for module_name in required_modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"✓ {module_name} imported successfully")
            results[module_name] = True
        except Exception as e:
            logger.error(f"✗ Failed to import {module_name}: {e}")
            results[module_name] = False
    
    return all(results.values())


def validate_class_structure():
    """Validate that all required classes and methods exist."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Validating Class Structure ===")
    
    try:
        from src.utils.multi_model_support import (
            ModelFamily, ModelArchitecture, ModelConfig, LayerMapping,
            UniversalModelAdapter, GPTModelAdapter, BERTModelAdapter,
            T5ModelAdapter, LLaMAModelAdapter, DomainSpecificModelAdapter,
            UniversalModelRegistry, MultiModelAnalyzer
        )
        
        # Validate enums
        assert hasattr(ModelFamily, 'GPT'), "ModelFamily missing GPT"
        assert hasattr(ModelFamily, 'BERT'), "ModelFamily missing BERT"
        assert hasattr(ModelFamily, 'T5'), "ModelFamily missing T5"
        assert hasattr(ModelFamily, 'LLAMA'), "ModelFamily missing LLAMA"
        assert hasattr(ModelFamily, 'DOMAIN_SPECIFIC'), "ModelFamily missing DOMAIN_SPECIFIC"
        
        # Validate T5 specific architectures
        assert hasattr(ModelArchitecture, 'T5'), "ModelArchitecture missing T5"
        assert hasattr(ModelArchitecture, 'FLAN_T5'), "ModelArchitecture missing FLAN_T5"
        assert hasattr(ModelArchitecture, 'UL2'), "ModelArchitecture missing UL2"
        
        # Validate LLaMA specific architectures
        assert hasattr(ModelArchitecture, 'LLAMA'), "ModelArchitecture missing LLAMA"
        assert hasattr(ModelArchitecture, 'ALPACA'), "ModelArchitecture missing ALPACA"
        assert hasattr(ModelArchitecture, 'VICUNA'), "ModelArchitecture missing VICUNA"
        
        # Validate domain-specific architectures
        assert hasattr(ModelArchitecture, 'CODEBERT'), "ModelArchitecture missing CODEBERT"
        assert hasattr(ModelArchitecture, 'SCIBERT'), "ModelArchitecture missing SCIBERT"
        assert hasattr(ModelArchitecture, 'BIOBERT'), "ModelArchitecture missing BIOBERT"
        
        logger.info("✓ All enum values present")
        
        # Validate adapter classes
        adapters_to_test = [
            ('T5ModelAdapter', T5ModelAdapter),
            ('LLaMAModelAdapter', LLaMAModelAdapter),
            ('DomainSpecificModelAdapter', DomainSpecificModelAdapter)
        ]
        
        for adapter_name, adapter_class in adapters_to_test:
            # Check inheritance
            assert issubclass(adapter_class, UniversalModelAdapter), f"{adapter_name} not subclass of UniversalModelAdapter"
            
            # Check required methods
            required_methods = ['load_model', 'get_layer_mapping', 'extract_activations', 'get_attention_patterns']
            for method in required_methods:
                assert hasattr(adapter_class, method), f"{adapter_name} missing method {method}"
            
            logger.info(f"✓ {adapter_name} structure valid")
        
        # Validate analyzer class
        analyzer = MultiModelAnalyzer()
        required_analyzer_methods = [
            'load_model', 'extract_activations', 'get_attention_patterns',
            'get_model_info', 'list_supported_models', 'unload_model', 'get_memory_usage'
        ]
        
        for method in required_analyzer_methods:
            assert hasattr(analyzer, method), f"MultiModelAnalyzer missing method {method}"
        
        logger.info("✓ MultiModelAnalyzer structure valid")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Class structure validation failed: {e}")
        return False


def validate_model_configs():
    """Validate model configurations for all families."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Validating Model Configurations ===")
    
    try:
        from src.utils.multi_model_support import (
            T5ModelAdapter, LLaMAModelAdapter, DomainSpecificModelAdapter
        )
        
        # Validate T5 configs
        logger.info("Validating T5 configurations...")
        t5_configs = T5ModelAdapter.T5_CONFIGS
        assert len(t5_configs) > 0, "No T5 configurations found"
        
        for model_name, config in t5_configs.items():
            assert config.family.value == 't5', f"T5 config {model_name} has wrong family"
            assert config.is_encoder_decoder == True, f"T5 config {model_name} should be encoder-decoder"
            assert config.has_relative_attention == True, f"T5 config {model_name} should have relative attention"
            logger.info(f"  ✓ {model_name} configuration valid")
        
        # Validate LLaMA configs
        logger.info("Validating LLaMA configurations...")
        llama_configs = LLaMAModelAdapter.LLAMA_CONFIGS
        assert len(llama_configs) > 0, "No LLaMA configurations found"
        
        for model_name, config in llama_configs.items():
            assert config.family.value == 'llama', f"LLaMA config {model_name} has wrong family"
            assert config.uses_rms_norm == True, f"LLaMA config {model_name} should use RMS norm"
            assert config.supports_model_parallel == True, f"LLaMA config {model_name} should support model parallel"
            logger.info(f"  ✓ {model_name} configuration valid")
        
        # Validate Domain-specific configs
        logger.info("Validating Domain-Specific configurations...")
        domain_configs = DomainSpecificModelAdapter.DOMAIN_CONFIGS
        assert len(domain_configs) > 0, "No domain-specific configurations found"
        
        for model_name, config in domain_configs.items():
            assert config.family.value == 'domain_specific', f"Domain config {model_name} has wrong family"
            assert config.domain is not None, f"Domain config {model_name} missing domain"
            logger.info(f"  ✓ {model_name} configuration valid (domain: {config.domain})")
        
        logger.info("✓ All model configurations valid")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model configuration validation failed: {e}")
        return False


def validate_layer_mappings():
    """Validate layer mappings for different architectures."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Validating Layer Mappings ===")
    
    try:
        from src.utils.multi_model_support import (
            T5ModelAdapter, LLaMAModelAdapter, DomainSpecificModelAdapter,
            ModelConfig, ModelFamily, ModelArchitecture
        )
        
        # Test T5 layer mapping
        t5_config = ModelConfig(
            name='test-t5', family=ModelFamily.T5, architecture=ModelArchitecture.T5,
            num_layers=6, hidden_size=512, num_attention_heads=8,
            max_position_embeddings=512, is_encoder_decoder=True
        )
        
        t5_adapter = T5ModelAdapter(t5_config)
        t5_mapping = t5_adapter.get_layer_mapping()
        
        assert t5_mapping.encoder_path is not None, "T5 mapping missing encoder path"
        assert t5_mapping.decoder_path is not None, "T5 mapping missing decoder path"
        assert t5_mapping.cross_attention_path is not None, "T5 mapping missing cross-attention path"
        logger.info("✓ T5 layer mapping valid")
        
        # Test LLaMA layer mapping
        llama_config = ModelConfig(
            name='test-llama', family=ModelFamily.LLAMA, architecture=ModelArchitecture.LLAMA,
            num_layers=32, hidden_size=4096, num_attention_heads=32,
            max_position_embeddings=2048, uses_rms_norm=True
        )
        
        llama_adapter = LLaMAModelAdapter(llama_config)
        llama_mapping = llama_adapter.get_layer_mapping()
        
        assert "model.layers" in llama_mapping.transformer_layers_path, "LLaMA mapping wrong transformer path"
        assert "self_attn" in llama_mapping.attention_path, "LLaMA mapping wrong attention path"
        logger.info("✓ LLaMA layer mapping valid")
        
        # Test domain-specific layer mapping
        domain_config = ModelConfig(
            name='test-domain', family=ModelFamily.DOMAIN_SPECIFIC, architecture=ModelArchitecture.CODEBERT,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, domain="programming"
        )
        
        domain_adapter = DomainSpecificModelAdapter(domain_config)
        domain_mapping = domain_adapter.get_layer_mapping()
        
        assert "encoder.layer" in domain_mapping.transformer_layers_path, "Domain mapping wrong transformer path"
        assert "attention.self" in domain_mapping.attention_path, "Domain mapping wrong attention path"
        logger.info("✓ Domain-specific layer mapping valid")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Layer mapping validation failed: {e}")
        return False


def validate_model_registry():
    """Validate the universal model registry functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Validating Model Registry ===")
    
    try:
        from src.utils.multi_model_support import UniversalModelRegistry, ModelFamily
        
        registry = UniversalModelRegistry()
        
        # Test adapter registration
        assert ModelFamily.T5 in registry.adapters, "T5 adapter not registered"
        assert ModelFamily.LLAMA in registry.adapters, "LLaMA adapter not registered"
        # Note: DOMAIN_SPECIFIC might be handled differently
        
        # Test model family detection
        test_cases = [
            ("t5-small", ModelFamily.T5),
            ("flan-t5-base", ModelFamily.T5),
            ("llama-7b", ModelFamily.LLAMA),
            ("alpaca-7b", ModelFamily.LLAMA),
            ("vicuna-13b", ModelFamily.LLAMA),
            ("codebert-base", ModelFamily.DOMAIN_SPECIFIC),
            ("scibert", ModelFamily.DOMAIN_SPECIFIC),
            ("biobert", ModelFamily.DOMAIN_SPECIFIC),
        ]
        
        for model_name, expected_family in test_cases:
            detected_family = registry.detect_model_family(model_name)
            assert detected_family == expected_family, f"Wrong family detected for {model_name}: {detected_family} != {expected_family}"
            logger.info(f"  ✓ {model_name} -> {detected_family.value}")
        
        # Test adapter creation
        test_models = ["t5-small", "llama-7b"]
        for model_name in test_models:
            try:
                adapter = registry.create_adapter(model_name)
                assert adapter is not None, f"Failed to create adapter for {model_name}"
                logger.info(f"  ✓ Adapter created for {model_name}")
            except Exception as e:
                logger.warning(f"  ⚠ Could not create adapter for {model_name}: {e}")
        
        logger.info("✓ Model registry validation complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model registry validation failed: {e}")
        return False


def validate_memory_management():
    """Validate memory management functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Validating Memory Management ===")
    
    try:
        from src.utils.multi_model_support import MultiModelAnalyzer
        
        analyzer = MultiModelAnalyzer()
        
        # Test memory usage reporting
        memory_info = analyzer.get_memory_usage()
        assert isinstance(memory_info, dict), "Memory info should be a dictionary"
        assert "system_memory_gb" in memory_info, "Missing system memory info"
        assert "system_memory_percent" in memory_info, "Missing system memory percentage"
        
        logger.info("Memory usage info:")
        for key, value in memory_info.items():
            logger.info(f"  {key}: {value:.2f}")
        
        # Test model listing
        supported_models = analyzer.list_supported_models()
        assert isinstance(supported_models, dict), "Supported models should be a dictionary"
        assert len(supported_models) > 0, "No supported models found"
        
        logger.info("Supported model families:")
        for family, models in supported_models.items():
            logger.info(f"  {family}: {len(models)} models")
        
        logger.info("✓ Memory management validation complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory management validation failed: {e}")
        return False


def validate_error_handling():
    """Validate error handling and robust execution."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Validating Error Handling ===")
    
    try:
        from src.utils.multi_model_support import MultiModelAnalyzer
        from src.utils.error_handling import ModelCompatibilityError
        
        analyzer = MultiModelAnalyzer()
        
        # Test loading non-existent model
        try:
            analyzer.load_model("non-existent-model-12345")
            logger.warning("Expected error for non-existent model not raised")
            return False
        except Exception as e:
            logger.info(f"✓ Proper error handling for non-existent model: {type(e).__name__}")
        
        # Test operations on unloaded model
        try:
            analyzer.extract_activations("unloaded-model", ["test"], [0])
            logger.warning("Expected error for unloaded model not raised")
            return False
        except Exception as e:
            logger.info(f"✓ Proper error handling for unloaded model: {type(e).__name__}")
        
        logger.info("✓ Error handling validation complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling validation failed: {e}")
        return False


def run_integration_test():
    """Run a basic integration test with available models."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Running Integration Test ===")
    
    try:
        from src.utils.multi_model_support import MultiModelAnalyzer
        
        analyzer = MultiModelAnalyzer()
        
        # Try with commonly available models
        test_models = ["gpt2", "bert-base-uncased"]
        test_input = "This is a test sentence for neural network analysis."
        
        success_count = 0
        
        for model_name in test_models:
            try:
                logger.info(f"Testing integration with {model_name}...")
                
                # Load model
                analyzer.load_model(model_name, device="cpu")
                
                # Get model info
                config = analyzer.get_model_info(model_name)
                logger.info(f"  Model family: {config.family.value}")
                
                # Extract features
                activations = analyzer.extract_activations(model_name, [test_input], [0])
                attention = analyzer.get_attention_patterns(model_name, [test_input], [0])
                
                logger.info(f"  ✓ Extracted {len(activations)} activations, {len(attention)} attention patterns")
                
                # Cleanup
                analyzer.unload_model(model_name)
                
                success_count += 1
                
            except Exception as e:
                logger.warning(f"  ⚠ Integration test failed for {model_name}: {e}")
        
        logger.info(f"✓ Integration test complete: {success_count}/{len(test_models)} models successful")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


def main():
    """Main validation function."""
    logger = setup_logging()
    logger.info("Starting Extended Multi-Model Support Validation (Section 3.1 Phase 2)")
    
    validation_tests = [
        ("Import Validation", validate_imports),
        ("Class Structure Validation", validate_class_structure),
        ("Model Configuration Validation", validate_model_configs),
        ("Layer Mapping Validation", validate_layer_mappings),
        ("Model Registry Validation", validate_model_registry),
        ("Memory Management Validation", validate_memory_management),
        ("Error Handling Validation", validate_error_handling),
        ("Integration Test", run_integration_test)
    ]
    
    results = {}
    
    for test_name, test_func in validation_tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info('='*60)
        
        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"\n{test_name}: {status}")
            
        except Exception as e:
            logger.error(f"\n{test_name}: FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 ALL VALIDATION TESTS PASSED!")
        logger.info("Section 3.1 Phase 2 implementation is validated and ready.")
        return True
    else:
        logger.error(f"\n❌ {total_tests - passed_tests} validation tests failed.")
        logger.error("Section 3.1 Phase 2 implementation needs fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
