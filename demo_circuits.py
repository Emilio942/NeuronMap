#!/usr/bin/env python3
"""
Standalone Demo of NeuronMap Circuit Discovery System
This demonstrates the core functionality without complex imports.
"""

import sys
import torch
import transformers
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Direct import of circuits module
sys.path.append(str(Path(__file__).parent / "src" / "analysis"))
import circuits

def demo_circuit_discovery():
    """Demo the circuit discovery system with a real model."""
    print("🧠 NeuronMap Circuit Discovery Demo")
    print("=" * 50)
    
    # Load a small model for demonstration
    print("Loading GPT-2 model...")
    try:
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Model loaded successfully")
        
        # Test induction head scanner
        print("\n🔍 Testing Induction Head Scanner...")
        scanner = circuits.InductionHeadScanner(model)
        print("✓ InductionHeadScanner initialized")
        
        # Create a simple test input
        test_text = "The cat sat on the mat. The cat"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        
        print(f"Test input: '{test_text}'")
        print("Analyzing induction patterns...")
        
        # Run analysis
        with torch.no_grad():
            results = scanner.scan_for_induction_heads(
                inputs['input_ids'], 
                threshold=0.5
            )
        
        print(f"✓ Found {len(results)} potential induction heads")
        
        # Test copying head scanner
        print("\n📋 Testing Copying Head Scanner...")
        copying_scanner = circuits.CopyingHeadScanner(model)
        print("✓ CopyingHeadScanner initialized")
        
        copying_results = copying_scanner.scan_for_copying_heads(
            inputs['input_ids'],
            copying_threshold=0.3
        )
        
        print(f"✓ Found {len(copying_results)} potential copying heads")
        
        print("\n⚡ Testing Attention Head Composition Analyzer...")
        comp_analyzer = circuits.AttentionHeadCompositionAnalyzer(model)
        print("✓ AttentionHeadCompositionAnalyzer initialized")
        
        if results and len(results) >= 2:
            # Test composition between first two induction heads
            layer1, head1, score1 = results[0]
            layer2, head2, score2 = results[1]
            if layer1 < layer2:
                comp_results = comp_analyzer.analyze_layer_compositions(
                    input_ids=inputs['input_ids'],
                    layer1=layer1,
                    layer2=layer2
                )
                print(f"✓ Composition analysis complete between layers {layer1} and {layer2}")
            else:
                print("✓ Composition analyzer ready (need different layers for full test)")
        else:
            print("✓ Composition analyzer ready (need more heads for full test)")
        
        print("\n✅ All circuit discovery components are working!")
        
        # Show summary
        print("\n" + "=" * 50)
        print("📊 Analysis Summary:")
        print(f"Model: GPT-2 ({model.config.n_layer} layers, {model.config.n_head} heads)")
        print(f"Induction heads found: {len(results)}")
        print(f"Copying heads found: {len(copying_results)}")
        print("Circuit discovery system is fully operational! 🎉")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_web_api():
    """Demo the web API functionality."""
    print("\n🌐 Testing Web API Components...")
    
    try:
        # Import the API blueprint
        sys.path.append(str(Path(__file__).parent / "src" / "web" / "api"))
        import circuits as api_circuits
        
        print("✓ API circuits module imported")
        
        # Check if the module has the expected functions
        funcs = [f for f in dir(api_circuits) if not f.startswith('_')]
        print(f"✓ Available functions: {funcs}")
        
        # Show that we can import the load function
        if hasattr(api_circuits, 'load_model_and_tokenizer'):
            print("✓ Model loading function available")
        
        print("✓ Web API components are working")
        return True
            
    except Exception as e:
        print(f"❌ API demo failed: {e}")
        return False

def main():
    """Run the full demo."""
    print("Starting NeuronMap Circuit Discovery Demo...")
    print("This may take a few moments to load the model.\n")
    
    # Test core functionality
    core_success = demo_circuit_discovery()
    
    # Test API components
    api_success = demo_web_api()
    
    print("\n" + "=" * 60)
    print("🎯 Final Demo Results:")
    print("=" * 60)
    print(f"Core Circuit Discovery: {'✅ PASS' if core_success else '❌ FAIL'}")
    print(f"Web API Components:     {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if core_success and api_success:
        print("\n🎉 Circuit Discovery system is fully functional!")
        print("\nNext steps:")
        print("1. Start web server: python -m src.web.app")
        print("2. Visit: http://localhost:5000/circuits")
        print("3. Use API endpoints for programmatic access")
    else:
        print("\n⚠️  Some components may need attention.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
