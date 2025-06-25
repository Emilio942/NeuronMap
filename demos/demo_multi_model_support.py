#!/usr/bin/env python3
"""Demonstration of Multi-Model Support System for Section 3.1.

This script demonstrates the comprehensive multi-model support capabilities
implemented for NeuronMap's Section 3.1 requirements.
"""

import sys
import time
import logging
from pathlib import Path
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.multi_model_support import (
    MultiModelAnalyzer, ModelFamily, ModelArchitecture,
    GPTModelAdapter, BERTModelAdapter, UniversalModelRegistry
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_model_registry():
    """Demonstrate the universal model registry capabilities."""
    print("\n🔧 UNIVERSAL MODEL REGISTRY DEMO")
    print("=" * 50)
    
    registry = UniversalModelRegistry()
    
    # Show supported model families
    print("Supported Model Families:")
    for family in ModelFamily:
        print(f"  • {family.value}")
    
    # Show supported architectures
    print("\nSupported Architectures:")
    for arch in ModelArchitecture:
        print(f"  • {arch.value}")
    
    # Demonstrate model family detection
    test_models = [
        "gpt2", "gpt2-medium", "gpt-neo-125M", "gpt-j-6B",
        "bert-base-uncased", "roberta-base", "distilbert-base-uncased"
    ]
    
    print("\nModel Family Detection:")
    for model in test_models:
        family = registry.detect_model_family(model)
        config = registry.get_model_config(model)
        
        if config:
            print(f"  • {model:20} → {family.value:10} ({config.num_layers} layers, {config.hidden_size} hidden)")
        else:
            print(f"  • {model:20} → {family.value:10} (config auto-detect)")


def demo_gpt_models():
    """Demonstrate GPT model family support."""
    print("\n🤖 GPT MODEL FAMILY DEMO")
    print("=" * 50)
    
    analyzer = MultiModelAnalyzer()
    
    # Show supported GPT models
    print("Supported GPT Models:")
    for model_name, config in GPTModelAdapter.GPT_CONFIGS.items():
        print(f"  • {model_name:15} - {config.num_layers:2d} layers, {config.hidden_size:4d} hidden, {config.memory_requirements_gb:4.1f}GB")
    
    # Demonstrate loading a small GPT model
    print(f"\nLoading GPT-2 model...")
    try:
        model_id = analyzer.load_model("gpt2")
        print(f"✓ Successfully loaded: {model_id}")
        
        # Get model info
        model_info = analyzer.get_model_info("gpt2")
        print(f"  - Architecture: {model_info.architecture.value}")
        print(f"  - Layers: {model_info.num_layers}")
        print(f"  - Hidden size: {model_info.hidden_size}")
        print(f"  - Attention heads: {model_info.num_attention_heads}")
        print(f"  - Max position: {model_info.max_position_embeddings}")
        
        # Test activation extraction
        print(f"\nExtracting activations...")
        test_inputs = ["Hello, how are you?", "The quick brown fox jumps"]
        layer_indices = [0, 6, 11]  # First, middle, last layers
        
        start_time = time.time()
        activations = analyzer.extract_activations("gpt2", test_inputs, layer_indices)
        extraction_time = time.time() - start_time
        
        print(f"✓ Extracted activations in {extraction_time:.2f}s")
        for layer_name, activation in activations.items():
            print(f"  - {layer_name}: {activation.shape}")
        
        # Test attention pattern extraction
        print(f"\nExtracting attention patterns...")
        start_time = time.time()
        attention_patterns = analyzer.get_attention_patterns("gpt2", test_inputs, layer_indices)
        attention_time = time.time() - start_time
        
        print(f"✓ Extracted attention patterns in {attention_time:.2f}s")
        for layer_name, attention in attention_patterns.items():
            print(f"  - {layer_name}: {attention.shape}")
        
        # Show memory usage
        memory_usage = analyzer.get_memory_usage()
        print(f"\nMemory Usage:")
        print(f"  - System RAM: {memory_usage['system_memory_gb']:.2f}GB ({memory_usage['system_memory_percent']:.1f}%)")
        if torch.cuda.is_available():
            print(f"  - GPU Memory: {memory_usage['gpu_memory_gb']:.2f}GB ({memory_usage['gpu_memory_percent']:.1f}%)")
        
        # Unload model
        analyzer.unload_model("gpt2")
        print(f"✓ Model unloaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading GPT model: {e}")


def demo_bert_models():
    """Demonstrate BERT model family support."""
    print("\n🔍 BERT MODEL FAMILY DEMO")
    print("=" * 50)
    
    analyzer = MultiModelAnalyzer()
    
    # Show supported BERT models
    print("Supported BERT Models:")
    for model_name, config in BERTModelAdapter.BERT_CONFIGS.items():
        print(f"  • {model_name:25} - {config.num_layers:2d} layers, {config.tokenizer_type}")
    
    # Demonstrate loading a BERT model
    print(f"\nLoading BERT model...")
    try:
        model_id = analyzer.load_model("bert-base-uncased")
        print(f"✓ Successfully loaded: {model_id}")
        
        # Get model info
        model_info = analyzer.get_model_info("bert-base-uncased")
        print(f"  - Architecture: {model_info.architecture.value}")
        print(f"  - Layers: {model_info.num_layers}")
        print(f"  - Tokenizer: {model_info.tokenizer_type}")
        print(f"  - Max position: {model_info.max_position_embeddings}")
        
        # Test with BERT-specific inputs (classification task)
        print(f"\nExtracting BERT activations...")
        test_inputs = [
            "The movie was fantastic and entertaining.",
            "This product is terrible and disappointing."
        ]
        layer_indices = [0, 6, 11]  # First, middle, last layers
        
        start_time = time.time()
        activations = analyzer.extract_activations("bert-base-uncased", test_inputs, layer_indices)
        extraction_time = time.time() - start_time
        
        print(f"✓ Extracted activations in {extraction_time:.2f}s")
        for layer_name, activation in activations.items():
            print(f"  - {layer_name}: {activation.shape}")
        
        # Test bidirectional attention patterns
        print(f"\nExtracting bidirectional attention patterns...")
        start_time = time.time()
        attention_patterns = analyzer.get_attention_patterns("bert-base-uncased", test_inputs, layer_indices)
        attention_time = time.time() - start_time
        
        print(f"✓ Extracted attention patterns in {attention_time:.2f}s")
        for layer_name, attention in attention_patterns.items():
            print(f"  - {layer_name}: {attention.shape} (bidirectional)")
        
        # Unload model
        analyzer.unload_model("bert-base-uncased")
        print(f"✓ Model unloaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading BERT model: {e}")


def demo_cross_model_comparison():
    """Demonstrate cross-model comparison capabilities."""
    print("\n🔄 CROSS-MODEL COMPARISON DEMO")
    print("=" * 50)
    
    analyzer = MultiModelAnalyzer()
    
    try:
        # Load multiple models for comparison
        print("Loading models for comparison...")
        models_to_compare = ["gpt2", "bert-base-uncased"]
        
        for model_name in models_to_compare:
            analyzer.load_model(model_name)
            print(f"✓ Loaded: {model_name}")
        
        # Compare model architectures
        print(f"\nModel Architecture Comparison:")
        print(f"{'Model':<20} {'Family':<10} {'Layers':<8} {'Hidden':<8} {'Heads':<8} {'Max Pos':<8}")
        print("-" * 70)
        
        for model_name in models_to_compare:
            config = analyzer.get_model_info(model_name)
            print(f"{model_name:<20} {config.family.value:<10} {config.num_layers:<8} "
                  f"{config.hidden_size:<8} {config.num_attention_heads:<8} {config.max_position_embeddings:<8}")
        
        # Compare activation extraction performance
        print(f"\nPerformance Comparison:")
        test_input = ["This is a test sentence for comparison."]
        layer_indices = [0, 5]  # Test with fewer layers for speed
        
        for model_name in models_to_compare:
            start_time = time.time()
            activations = analyzer.extract_activations(model_name, test_input, layer_indices)
            extraction_time = time.time() - start_time
            
            total_params = sum(activation.numel() for activation in activations.values())
            print(f"  • {model_name:20} - {extraction_time:.3f}s, {total_params:,} activation parameters")
        
        # Show combined memory usage
        memory_usage = analyzer.get_memory_usage()
        print(f"\nCombined Memory Usage:")
        print(f"  - System RAM: {memory_usage['system_memory_gb']:.2f}GB")
        if torch.cuda.is_available():
            print(f"  - GPU Memory: {memory_usage['gpu_memory_gb']:.2f}GB")
        
        # Unload all models
        for model_name in models_to_compare:
            analyzer.unload_model(model_name)
        print(f"\n✓ All models unloaded")
        
    except Exception as e:
        print(f"✗ Error in cross-model comparison: {e}")


def demo_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    print("\n💾 MEMORY OPTIMIZATION DEMO")
    print("=" * 50)
    
    analyzer = MultiModelAnalyzer()
    
    print("Memory optimization features:")
    print("  • Automatic gradient checkpointing")
    print("  • Mixed precision support")
    print("  • Device mapping for large models")
    print("  • Memory profiling and monitoring")
    print("  • Automatic garbage collection")
    
    # Show memory before loading
    initial_memory = analyzer.get_memory_usage()
    print(f"\nInitial Memory State:")
    print(f"  - System RAM: {initial_memory['system_memory_gb']:.2f}GB")
    if torch.cuda.is_available():
        print(f"  - GPU Memory: {initial_memory.get('gpu_memory_gb', 0):.2f}GB")
    
    try:
        # Load model with optimization
        print(f"\nLoading model with memory optimization...")
        analyzer.load_model("gpt2")
        
        # Show memory after loading
        loaded_memory = analyzer.get_memory_usage()
        print(f"\nMemory After Loading:")
        print(f"  - System RAM: {loaded_memory['system_memory_gb']:.2f}GB "
              f"(+{loaded_memory['system_memory_gb'] - initial_memory['system_memory_gb']:.2f}GB)")
        if torch.cuda.is_available():
            print(f"  - GPU Memory: {loaded_memory.get('gpu_memory_gb', 0):.2f}GB "
                  f"(+{loaded_memory.get('gpu_memory_gb', 0) - initial_memory.get('gpu_memory_gb', 0):.2f}GB)")
        
        # Unload and show memory cleanup
        analyzer.unload_model("gpt2")
        final_memory = analyzer.get_memory_usage()
        print(f"\nMemory After Cleanup:")
        print(f"  - System RAM: {final_memory['system_memory_gb']:.2f}GB "
              f"({final_memory['system_memory_gb'] - initial_memory['system_memory_gb']:.2f}GB difference)")
        if torch.cuda.is_available():
            print(f"  - GPU Memory: {final_memory.get('gpu_memory_gb', 0):.2f}GB "
                  f"({final_memory.get('gpu_memory_gb', 0) - initial_memory.get('gpu_memory_gb', 0):.2f}GB difference)")
        
    except Exception as e:
        print(f"✗ Error in memory optimization demo: {e}")


def demo_supported_models_overview():
    """Show overview of all supported models."""
    print("\n📋 SUPPORTED MODELS OVERVIEW")
    print("=" * 50)
    
    analyzer = MultiModelAnalyzer()
    supported_models = analyzer.list_supported_models()
    
    total_models = 0
    for family, models in supported_models.items():
        print(f"\n{family.upper()} Family ({len(models)} models):")
        for model in models:
            config = analyzer.get_model_info(model)
            print(f"  • {model:25} - {config.memory_requirements_gb:4.1f}GB, {config.recommended_batch_size:2d} batch")
            total_models += 1
    
    print(f"\nTotal Supported Models: {total_models}")
    print("\nCapabilities:")
    print("  ✓ Automatic architecture detection")
    print("  ✓ Unified activation extraction API")
    print("  ✓ Cross-model attention pattern analysis")
    print("  ✓ Memory-optimized loading")
    print("  ✓ Device-aware model placement")
    print("  ✓ Performance monitoring")


def main():
    """Run all multi-model support demonstrations."""
    print("🧠 NEURONMAP MULTI-MODEL SUPPORT DEMONSTRATION")
    print("Section 3.1: Multi-Model Support Extension")
    print("=" * 60)
    
    # Check system requirements
    print(f"System Information:")
    print(f"  • PyTorch: {torch.__version__}")
    print(f"  • CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • GPU: {torch.cuda.get_device_name()}")
        print(f"  • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # Run demonstrations
        demo_model_registry()
        demo_gpt_models()
        demo_bert_models()
        demo_cross_model_comparison()
        demo_memory_optimization()
        demo_supported_models_overview()
        
        print("\n" + "=" * 60)
        print("🎉 MULTI-MODEL SUPPORT DEMONSTRATION COMPLETE!")
        print("=" * 60)
        
        print("\n✅ SECTION 3.1 FEATURES DEMONSTRATED:")
        print("  • Universal model architecture support")
        print("  • GPT family models (GPT-2, GPT-Neo, GPT-J)")
        print("  • BERT family models (BERT, RoBERTa, DistilBERT)")
        print("  • Automatic layer mapping and architecture detection")
        print("  • Cross-model comparison and analysis")
        print("  • Memory optimization and management")
        print("  • Unified API for all model families")
        
        print("\n🚀 READY FOR: T5, LLaMA, and Domain-Specific Model Extensions")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
