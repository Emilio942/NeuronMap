"""
Simple Test Runner for NeuronMap Tools
====================================

Basic testing without heavy dependencies to verify tool structure.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_tool_imports():
    """Test if tools can be imported without errors."""
    results = {}
    
    tools_to_test = [
        ("integrated_gradients", "src.analysis.interpretability.ig_explainer", "IntegratedGradientsExplainer"),
        ("deep_shap", "src.analysis.interpretability.shap_explainer", "DeepSHAPExplainer"),
        ("semantic_labeling", "src.analysis.interpretability.semantic_labeling", "SemanticLabeler"),
        ("ace_concepts", "src.analysis.concepts.ace_extractor", "ACEConceptExtractor"),
    ]
    
    for tool_id, module_path, class_name in tools_to_test:
        try:
            # Import module
            module = __import__(module_path, fromlist=[class_name])
            tool_class = getattr(module, class_name)
            
            # Try to instantiate
            tool_instance = tool_class()
            
            # Check basic attributes
            has_tool_id = hasattr(tool_instance, 'tool_id')
            has_version = hasattr(tool_instance, 'version')
            has_description = hasattr(tool_instance, 'description')
            
            results[tool_id] = {
                'import_success': True,
                'instantiation_success': True,
                'has_required_attrs': has_tool_id and has_version and has_description,
                'tool_id': getattr(tool_instance, 'tool_id', None),
                'version': getattr(tool_instance, 'version', None)
            }
            
            print(f"✅ {tool_id}: Import and instantiation successful")
            
        except Exception as e:
            results[tool_id] = {
                'import_success': False,
                'error': str(e)
            }
            print(f"❌ {tool_id}: Failed - {e}")
    
    return results

def test_basic_functionality():
    """Test basic functionality without external models."""
    print("\nTesting basic functionality...")
    
    try:
        # Test plugin interface
        from src.core.plugin_interface import InterpretabilityPluginBase
        print("✅ Plugin interface imported successfully")
        
        # Test tools registry
        from pathlib import Path
        registry_path = Path("configs/tools_registry.yaml")
        if registry_path.exists():
            print("✅ Tools registry found")
        else:
            print("❌ Tools registry not found")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("NeuronMap Tools Basic Test")
    print("=" * 60)
    
    # Test imports
    import_results = test_tool_imports()
    
    # Test basic functionality
    basic_test = test_basic_functionality()
    
    # Summary
    successful_imports = sum(1 for r in import_results.values() if r.get('import_success', False))
    total_tools = len(import_results)
    
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✅ Successfully imported: {successful_imports}/{total_tools} tools")
    print(f"🔧 Basic functionality: {'✅ Working' if basic_test else '❌ Failed'}")
    
    if successful_imports == total_tools and basic_test:
        print("🎉 All basic tests passed!")
    else:
        print("⚠️  Some tests failed - check implementation")
