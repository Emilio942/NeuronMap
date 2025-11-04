#!/usr/bin/env python3
"""
Simple validation test for NeuronMap interpretability tools.
Tests basic functionality without heavy dependencies.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.analysis.interpretability.ig_explainer import IntegratedGradientsExplainer
        print("✅ Integrated Gradients imported")
    except Exception as e:
        print(f"❌ IG import failed: {e}")
    
    try:
        from src.analysis.interpretability.shap_explainer import DeepSHAPExplainer
        print("✅ SHAP explainer imported")
    except Exception as e:
        print(f"❌ SHAP import failed: {e}")
    
    try:
        from src.analysis.interpretability.semantic_labeling import SemanticLabeler
        print("✅ Semantic labeling imported")
    except Exception as e:
        print(f"❌ Semantic labeling import failed: {e}")
    
    try:
        from src.analysis.concepts.ace_extractor import ACEConceptExtractor
        print("✅ ACE concepts imported")
    except Exception as e:
        print(f"❌ ACE import failed: {e}")
    
    try:
        from src.analysis.testing.coverage_tracker import NeuronCoverageTracker
        print("✅ Coverage tracker imported")
    except Exception as e:
        print(f"❌ Coverage tracker import failed: {e}")
    
    try:
        from src.analysis.testing.surprise_tracker import SurpriseCoverageTracker
        print("✅ Surprise tracker imported")
    except Exception as e:
        print(f"❌ Surprise tracker import failed: {e}")
    
    try:
        from src.analysis.metrics.wasserstein_comparator import WassersteinComparator
        print("✅ Wasserstein comparator imported")
    except Exception as e:
        print(f"❌ Wasserstein import failed: {e}")
    
    try:
        from src.analysis.metrics.emd_heatmap import EMDHeatmapComparator
        print("✅ EMD heatmap imported")
    except Exception as e:
        print(f"❌ EMD import failed: {e}")
    
    try:
        from src.analysis.mechanistic.transformerlens_adapter import TransformerLensAdapter
        print("✅ TransformerLens adapter imported")
    except Exception as e:
        print(f"❌ TL adapter import failed: {e}")
    
    try:
        from src.analysis.mechanistic.residual_stream_comparator import ResidualStreamComparator
        print("✅ Residual stream comparator imported")
    except Exception as e:
        print(f"❌ Residual stream import failed: {e}")

def test_basic_functionality():
    """Test basic functionality of a few tools."""
    print("\nTesting basic functionality...")
    
    # Test data
    test_activations = np.random.randn(10, 50)
    test_heatmap = np.random.rand(32, 32)
    
    try:
        # Test ACE extractor (least dependencies)
        from src.analysis.concepts.ace_extractor import ACEConceptExtractor
        
        ace = ACEConceptExtractor({
            'n_clusters': 3,
            'clustering_method': 'kmeans'
        })
        
        if ace.initialize():
            print("✅ ACE extractor initialization successful")
            
            result = ace.execute(
                activations=test_activations,
                concept_descriptions=["concept A", "concept B", "concept C"]
            )
            
            if result.success:
                print("✅ ACE extractor execution successful")
            else:
                print(f"❌ ACE execution failed: {result.errors}")
        else:
            print("❌ ACE initialization failed")
            
    except Exception as e:
        print(f"❌ ACE test failed: {e}")
    
    try:
        # Test Wasserstein comparator
        from src.analysis.metrics.wasserstein_comparator import WassersteinComparator
        
        wass = WassersteinComparator()
        
        if wass.initialize():
            print("✅ Wasserstein comparator initialization successful")
            
            dist_a = np.random.randn(50, 10)
            dist_b = np.random.randn(50, 10) + 0.5
            
            result = wass.execute(dist_a, dist_b)
            
            if result.success:
                print("✅ Wasserstein comparator execution successful")
                print(f"   Distance: {result.outputs.get('wasserstein_distance', 'N/A')}")
            else:
                print(f"❌ Wasserstein execution failed: {result.errors}")
        else:
            print("❌ Wasserstein initialization failed")
            
    except Exception as e:
        print(f"❌ Wasserstein test failed: {e}")

def test_registry_config():
    """Test that the tools registry config is valid."""
    print("\nTesting registry configuration...")
    
    try:
        import yaml
        
        registry_path = project_root / "configs" / "tools_registry.yaml"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = yaml.safe_load(f)
            
            tools = registry.get('tools', {})
            security = registry.get('security', {})
            
            print(f"✅ Registry loaded: {len(tools)} tools defined")
            print(f"✅ Security config loaded: {len(security)} settings")
            
            # Check that all our implemented tools are in the registry
            expected_tools = [
                'integrated_gradients',
                'deepshap_explainer', 
                'semantic_labeling',
                'ace_concepts',
                'neuron_coverage',
                'surprise_coverage',
                'wasserstein_distance',
                'emd_heatmap',
                'transformerlens_adapter',
                'residual_stream_comparator'
            ]
            
            missing_tools = []
            for tool_id in expected_tools:
                if tool_id not in tools:
                    missing_tools.append(tool_id)
            
            if missing_tools:
                print(f"❌ Missing tools in registry: {missing_tools}")
            else:
                print("✅ All expected tools found in registry")
                
        else:
            print(f"❌ Registry file not found: {registry_path}")
            
    except Exception as e:
        print(f"❌ Registry test failed: {e}")

def test_plugin_interface():
    """Test the plugin interface."""
    print("\nTesting plugin interface...")
    
    try:
        from src.core.plugin_interface import InterpretabilityPluginBase, ToolExecutionResult
        
        # Test that we can create a basic plugin
        class TestPlugin(InterpretabilityPluginBase):
            def __init__(self):
                super().__init__("test_plugin")
                
            def initialize(self):
                self.initialized = True
                return True
                
            def execute(self, *args, **kwargs):
                return ToolExecutionResult(
                    tool_id=self.tool_id,
                    success=True,
                    execution_time=0.1,
                    outputs={'test': 'data'},
                    metadata=self.get_metadata(),
                    errors=[],
                    warnings=[],
                    timestamp='2024-01-01 00:00:00'
                )
        
        plugin = TestPlugin()
        
        if plugin.initialize():
            print("✅ Plugin interface initialization successful")
            
            result = plugin.execute()
            if result.success:
                print("✅ Plugin interface execution successful")
            else:
                print("❌ Plugin interface execution failed")
        else:
            print("❌ Plugin interface initialization failed")
            
    except Exception as e:
        print(f"❌ Plugin interface test failed: {e}")

def main():
    """Run all tests."""
    print("🧠 NeuronMap Interpretability Tools - Validation Test")
    print("=" * 60)
    
    test_imports()
    test_plugin_interface()
    test_registry_config()
    test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("🎯 Validation completed!")
    print("\nTo run the full demo with all dependencies installed:")
    print("   pip install transformers torch scikit-learn scipy matplotlib seaborn")
    print("   python demo_final_validation.py")

if __name__ == '__main__':
    main()
