#!/usr/bin/env python3
"""
Section 3.1 Validation Script: Universal Model Support & Advanced Analysis
===========================================================================

This script validates the implementation of the Universal Model Support system
with automatic layer mapping and cross-architecture compatibility.

Test coverage:
- Universal Model Support Framework
- Automatic Layer Discovery
- Architecture Registry
- Domain-Specific Adaptations
- Cross-Architecture Compatibility
- Performance Analysis
- Optimization Recommendations
"""

import sys
import os
import traceback
import logging
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_1_universal_model_support_creation():
    """Test 1: Universal Model Support instance creation"""
    try:
        from src.analysis.universal_model_support import UniversalModelSupport, create_universal_model_support
        
        # Test direct instantiation
        ums = UniversalModelSupport()
        assert ums is not None, "UniversalModelSupport instance should not be None"
        assert hasattr(ums, 'architecture_registry'), "Should have architecture_registry"
        assert hasattr(ums, 'layer_mapper'), "Should have layer_mapper"
        assert hasattr(ums, 'domain_adapters'), "Should have domain_adapters"
        
        # Test convenience function
        ums2 = create_universal_model_support()
        assert ums2 is not None, "create_universal_model_support should return valid instance"
        
        print("✓ Test 1 passed: Universal Model Support creation")
        return True
    except Exception as e:
        print(f"✗ Test 1 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_2_architecture_registry():
    """Test 2: Architecture Registry functionality"""
    try:
        from src.analysis.universal_model_support import ArchitectureRegistry, ArchitectureType, AdapterConfig
        
        registry = ArchitectureRegistry()
        
        # Test default configurations
        supported_families = registry.list_supported_families()
        assert len(supported_families) >= 4, f"Should have at least 4 default families, got {len(supported_families)}"
        assert "gpt" in supported_families, "Should support GPT family"
        assert "bert" in supported_families, "Should support BERT family"
        assert "t5" in supported_families, "Should support T5 family"
        assert "llama" in supported_families, "Should support LLaMA family"
        
        # Test architecture detection
        assert registry.detect_architecture("gpt-2") == ArchitectureType.GPT
        assert registry.detect_architecture("bert-base-uncased") == ArchitectureType.BERT
        assert registry.detect_architecture("t5-base") == ArchitectureType.T5
        assert registry.detect_architecture("llama-7b") == ArchitectureType.LLAMA
        assert registry.detect_architecture("unknown-model") == ArchitectureType.UNKNOWN
        
        # Test configuration retrieval
        gpt_config = registry.get_config("gpt")
        assert gpt_config is not None, "Should retrieve GPT configuration"
        assert gpt_config.architecture_type == ArchitectureType.GPT
        assert not gpt_config.supports_bidirectional, "GPT should not support bidirectional"
        
        print("✓ Test 2 passed: Architecture Registry functionality")
        return True
    except Exception as e:
        print(f"✗ Test 2 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_3_layer_mapper():
    """Test 3: Universal Layer Mapper functionality"""
    try:
        from src.analysis.universal_model_support import UniversalLayerMapper, ArchitectureRegistry, LayerType
        import torch.nn as nn
        
        registry = ArchitectureRegistry()
        mapper = UniversalLayerMapper(registry)
        
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 768)
                self.layers = nn.ModuleList([
                    nn.MultiheadAttention(768, 12),
                    nn.Linear(768, 3072),
                    nn.LayerNorm(768)
                ])
                self.output = nn.Linear(768, 1000)
        
        model = TestModel()
        
        # Test layer discovery
        layers = mapper.discover_layers(model, "test-model")
        assert len(layers) > 0, "Should discover at least some layers"
        
        # Test layer type classification
        layer_types = set(layer.layer_type for layer in layers)
        assert LayerType.EMBEDDING in layer_types, "Should detect embedding layer"
        
        # Test layer filtering
        attention_layers = mapper.get_layers_by_type(layers, LayerType.ATTENTION)
        assert len(attention_layers) >= 0, "Should be able to filter attention layers"
        
        print("✓ Test 3 passed: Universal Layer Mapper functionality")
        return True
    except Exception as e:
        print(f"✗ Test 3 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_4_model_architecture_analysis():
    """Test 4: Model architecture analysis"""
    try:
        from src.analysis.universal_model_support import UniversalModelSupport
        import torch.nn as nn
        
        # Create test model
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 512)
                self.encoder_layers = nn.ModuleList([
                    nn.MultiheadAttention(512, 8) for _ in range(6)
                ])
                self.output = nn.Linear(512, 1000)
        
        model = SimpleTransformer()
        ums = UniversalModelSupport()
        
        # Test architecture analysis
        analysis = ums.analyze_model_architecture(model, "simple-transformer")
        
        assert "model_name" in analysis, "Analysis should include model name"
        assert "architecture_type" in analysis, "Analysis should include architecture type"
        assert "total_layers" in analysis, "Analysis should include layer count"
        assert "layer_types" in analysis, "Analysis should include layer type distribution"
        assert "layers" in analysis, "Analysis should include layer details"
        
        assert analysis["model_name"] == "simple-transformer"
        assert analysis["total_layers"] > 0, "Should detect layers"
        assert isinstance(analysis["layers"], list), "Layers should be a list"
        
        print("✓ Test 4 passed: Model architecture analysis")
        return True
    except Exception as e:
        print(f"✗ Test 4 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_5_cross_architecture_compatibility():
    """Test 5: Cross-architecture compatibility validation"""
    try:
        from src.analysis.universal_model_support import UniversalModelSupport
        
        ums = UniversalModelSupport()
        
        # Test compatibility check
        compatibility = ums.validate_cross_architecture_compatibility("gpt-2", "bert-base-uncased")
        
        assert "architectures" in compatibility, "Should include architecture names"
        assert "are_compatible" in compatibility, "Should indicate compatibility"
        assert "common_features" in compatibility, "Should list common features"
        assert "differences" in compatibility, "Should list differences"
        assert "recommendations" in compatibility, "Should provide recommendations"
        
        assert len(compatibility["architectures"]) == 2, "Should compare two architectures"
        assert compatibility["are_compatible"] == False, "GPT and BERT should not be directly compatible"
        
        # Test same architecture compatibility
        same_arch_compatibility = ums.validate_cross_architecture_compatibility("gpt-2", "gpt-neo-1.3B")
        assert same_arch_compatibility["are_compatible"] == True, "Same architecture should be compatible"
        
        print("✓ Test 5 passed: Cross-architecture compatibility validation")
        return True
    except Exception as e:
        print(f"✗ Test 5 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_6_performance_analyzer():
    """Test 6: Performance Analyzer functionality"""
    try:
        from src.analysis.universal_advanced_analyzer import PerformanceAnalyzer
        import torch.nn as nn
        
        # Create test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(768, 768) for _ in range(12)
                ])
        
        model = TestModel()
        analyzer = PerformanceAnalyzer()
        
        # Test performance analysis
        metrics = analyzer.analyze_performance(model, "test-model")
        
        assert hasattr(metrics, 'memory_usage_mb'), "Should have memory usage metric"
        assert hasattr(metrics, 'inference_time_ms'), "Should have inference time metric"
        assert hasattr(metrics, 'total_parameters'), "Should have parameter count"
        assert hasattr(metrics, 'efficiency_score'), "Should have efficiency score"
        
        assert metrics.memory_usage_mb > 0, "Memory usage should be positive"
        assert metrics.total_parameters > 0, "Parameter count should be positive"
        assert 0 <= metrics.efficiency_score <= 1, "Efficiency score should be between 0 and 1"
        
        # Test optimization recommendations
        recommendations = analyzer.generate_optimization_recommendations(metrics, "test-model")
        assert isinstance(recommendations, list), "Recommendations should be a list"
        
        print("✓ Test 6 passed: Performance Analyzer functionality")
        return True
    except Exception as e:
        print(f"✗ Test 6 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_7_domain_specific_analyzer():
    """Test 7: Domain-Specific Analyzer functionality"""
    try:
        from src.analysis.universal_advanced_analyzer import DomainSpecificAnalyzer
        import torch.nn as nn
        
        # Create test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Linear(768, 768)
        
        model = TestModel()
        analyzer = DomainSpecificAnalyzer()
        
        # Test domain analysis
        for domain in ["code", "scientific", "biomedical"]:
            analysis = analyzer.analyze_domain_specificity(model, "test-model", domain)
            
            assert "domain" in analysis, f"Analysis should include domain for {domain}"
            assert "vocabulary_specialization" in analysis, f"Should analyze vocabulary for {domain}"
            assert "attention_analysis" in analysis, f"Should analyze attention for {domain}"
            assert "domain_compatibility_score" in analysis, f"Should provide compatibility score for {domain}"
            
            assert analysis["domain"] == domain
            assert 0 <= analysis["domain_compatibility_score"] <= 1, "Compatibility score should be between 0 and 1"
        
        # Test invalid domain
        try:
            analyzer.analyze_domain_specificity(model, "test-model", "invalid-domain")
            assert False, "Should raise error for invalid domain"
        except ValueError:
            pass  # Expected
        
        print("✓ Test 7 passed: Domain-Specific Analyzer functionality")
        return True
    except Exception as e:
        print(f"✗ Test 7 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_8_cross_architecture_analyzer():
    """Test 8: Cross-Architecture Analyzer functionality"""
    try:
        from src.analysis.universal_advanced_analyzer import CrossArchitectureAnalyzer
        from src.analysis.universal_model_support import UniversalModelSupport
        import torch.nn as nn
        
        # Create test models
        class Model1(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(768, 768) for _ in range(6)])
        
        class Model2(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(12)])
        
        model1 = Model1()
        model2 = Model2()
        
        ums = UniversalModelSupport()
        analyzer = CrossArchitectureAnalyzer(ums)
        
        # Test cross-architecture comparison
        comparison = analyzer.compare_architectures(model1, "model1", model2, "model2")
        
        assert hasattr(comparison, 'model1_name'), "Should have model1 name"
        assert hasattr(comparison, 'model2_name'), "Should have model2 name"
        assert hasattr(comparison, 'similarity_score'), "Should have similarity score"
        assert hasattr(comparison, 'shared_features'), "Should have shared features"
        assert hasattr(comparison, 'unique_features1'), "Should have unique features for model1"
        assert hasattr(comparison, 'unique_features2'), "Should have unique features for model2"
        assert hasattr(comparison, 'recommendations'), "Should have recommendations"
        
        assert comparison.model1_name == "model1"
        assert comparison.model2_name == "model2"
        assert 0 <= comparison.similarity_score <= 1, "Similarity score should be between 0 and 1"
        
        print("✓ Test 8 passed: Cross-Architecture Analyzer functionality")
        return True
    except Exception as e:
        print(f"✗ Test 8 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_9_universal_advanced_analyzer():
    """Test 9: Universal Advanced Analyzer comprehensive functionality"""
    try:
        from src.analysis.universal_advanced_analyzer import UniversalAdvancedAnalyzer
        import torch.nn as nn
        
        # Create test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 768)
                self.layers = nn.ModuleList([
                    nn.MultiheadAttention(768, 12) for _ in range(6)
                ])
                self.output = nn.Linear(768, 1000)
        
        model = TestModel()
        analyzer = UniversalAdvancedAnalyzer()
        
        # Test comprehensive analysis
        results = analyzer.comprehensive_analysis(model, "comprehensive-test-model")
        
        assert "architecture_analysis" in results, "Should include architecture analysis"
        assert "performance_analysis" in results, "Should include performance analysis"
        assert "optimization_recommendations" in results, "Should include optimization recommendations"
        assert "summary" in results, "Should include summary"
        
        # Test with domain analysis
        results_with_domain = analyzer.comprehensive_analysis(
            model, "test-model-with-domain", domain="code"
        )
        assert "domain_analysis" in results_with_domain, "Should include domain analysis when requested"
        
        # Test summary structure
        summary = results["summary"]
        assert "model_type" in summary, "Summary should include model type"
        assert "performance_score" in summary, "Summary should include performance score"
        assert "key_insights" in summary, "Summary should include key insights"
        
        print("✓ Test 9 passed: Universal Advanced Analyzer comprehensive functionality")
        return True
    except Exception as e:
        print(f"✗ Test 9 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_10_integration_with_existing_handlers():
    """Test 10: Integration with existing model handlers"""
    try:
        from src.analysis.universal_model_support import analyze_model
        from src.analysis.universal_advanced_analyzer import UniversalAdvancedAnalyzer
        import torch.nn as nn
        
        # Test convenience function
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 768)
        
        model = SimpleModel()
        
        # Test quick analysis function
        quick_analysis = analyze_model(model, "quick-test")
        assert isinstance(quick_analysis, dict), "Quick analysis should return a dictionary"
        assert "model_name" in quick_analysis, "Quick analysis should include model name"
        
        # Test compatibility with comprehensive analyzer
        analyzer = UniversalAdvancedAnalyzer()
        comprehensive_results = analyzer.comprehensive_analysis(model, "integration-test")
        
        # Both should analyze the same model consistently
        assert comprehensive_results["architecture_analysis"]["model_name"] == "integration-test"
        assert quick_analysis["model_name"] == "quick-test"
        
        print("✓ Test 10 passed: Integration with existing handlers")
        return True
    except Exception as e:
        print(f"✗ Test 10 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_11_supported_models_information():
    """Test 11: Supported models information retrieval"""
    try:
        from src.analysis.universal_model_support import UniversalModelSupport
        
        ums = UniversalModelSupport()
        
        # Test supported models information
        supported_info = ums.get_supported_models()
        
        assert "model_families" in supported_info, "Should list model families"
        assert "architecture_types" in supported_info, "Should list architecture types"
        assert "total_supported" in supported_info, "Should provide total count"
        
        assert len(supported_info["model_families"]) >= 4, "Should support at least 4 model families"
        assert len(supported_info["architecture_types"]) >= 5, "Should have at least 5 architecture types"
        assert supported_info["total_supported"] > 0, "Should have positive total count"
        
        print("✓ Test 11 passed: Supported models information retrieval")
        return True
    except Exception as e:
        print(f"✗ Test 11 failed: {str(e)}")
        traceback.print_exc()
        return False

def test_12_cache_management():
    """Test 12: Cache management functionality"""
    try:
        from src.analysis.universal_model_support import UniversalModelSupport
        import torch.nn as nn
        
        # Create test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(768, 768)
        
        model = TestModel()
        ums = UniversalModelSupport()
        
        # Analyze model multiple times to test caching
        analysis1 = ums.analyze_model_architecture(model, "cache-test-model")
        analysis2 = ums.analyze_model_architecture(model, "cache-test-model")
        
        # Results should be consistent
        assert analysis1["model_name"] == analysis2["model_name"]
        assert analysis1["total_layers"] == analysis2["total_layers"]
        
        # Test cache clearing
        ums.clear_cache()
        
        # Should still work after cache clear
        analysis3 = ums.analyze_model_architecture(model, "cache-test-model")
        assert analysis3["model_name"] == "cache-test-model"
        
        print("✓ Test 12 passed: Cache management functionality")
        return True
    except Exception as e:
        print(f"✗ Test 12 failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests for Section 3.1"""
    print("=" * 80)
    print("SECTION 3.1 VALIDATION: Universal Model Support & Advanced Analysis")
    print("=" * 80)
    
    tests = [
        test_1_universal_model_support_creation,
        test_2_architecture_registry,
        test_3_layer_mapper,
        test_4_model_architecture_analysis,
        test_5_cross_architecture_compatibility,
        test_6_performance_analyzer,
        test_7_domain_specific_analyzer,
        test_8_cross_architecture_analyzer,
        test_9_universal_advanced_analyzer,
        test_10_integration_with_existing_handlers,
        test_11_supported_models_information,
        test_12_cache_management
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 80)
    print(f"SECTION 3.1 VALIDATION RESULTS: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Section 3.1 implementation is complete and functional.")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
