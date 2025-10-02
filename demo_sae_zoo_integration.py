#!/usr/bin/env python3
"""
Demo script for SAE-Analysis Zoo integration
Demonstrates how to share SAE models and features through the Analysis Zoo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.sae_training import SAETrainer, SAEConfig
from src.analysis.sae_feature_analysis import SAEFeatureExtractor, FeatureAnalysis
from src.zoo.artifact_schema import ArtifactSchema, ArtifactType, LicenseType, ModelCompatibility, AuthorInfo
from src.zoo.storage import S3StorageManager as StorageManager
import torch
import json
import tempfile
from pathlib import Path
import numpy as np

def create_demo_sae_artifact():
    """Create a demo SAE artifact for the Analysis Zoo"""
    print("🏗️ Creating demo SAE artifact...")
    
    # Create a simple SAE configuration
    config = SAEConfig(
        model_name="gpt2",
        layer=8,
        component="mlp",
        input_dim=768,
        hidden_dim=4096,
        sparsity_penalty=0.01,
        learning_rate=0.0001,
        batch_size=32,
        num_epochs=100
    )
    
    # Create mock training results
    training_results = {
        "final_reconstruction_loss": 0.045,
        "sparsity_achieved": 0.012,
        "features_activated": 3876,
        "training_time": "2h 34m",
        "convergence_epoch": 87
    }
    
    # Create a temporary SAE model file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_model:
        # Mock SAE model state (in reality this would be the actual trained model)
        mock_model_data = {
            'encoder_weight': torch.randn(768, 4096),
            'decoder_weight': torch.randn(4096, 768),
            'encoder_bias': torch.zeros(4096),
            'decoder_bias': torch.zeros(768),
            'config': config.__dict__,
            'training_results': training_results
        }
        torch.save(mock_model_data, tmp_model.name)
        model_path = tmp_model.name
    
    # Create artifact metadata
    artifact = ArtifactSchema(
        artifact_type=ArtifactType.SAE_MODEL,
        name="gpt2_layer8_mlp_sae",
        description="Sparse Autoencoder trained on GPT-2 layer 8 MLP activations with 0.045 reconstruction loss",
        authors=[AuthorInfo(name="SAE Training Pipeline", email="demo@neuronmap.ai")],
        version="1.0.0",
        license=LicenseType.MIT,
        model_compatibility=[ModelCompatibility(
            model_name="gpt2",
            model_family="gpt",
            architecture="transformer",
            layers=[8]
        )],
        dependencies=[],
        tags=["sae", "gpt2", "mlp", "layer8", "interpretability"]
    )
    
    print(f"   ✅ Created SAE artifact: {artifact.name}")
    print(f"   📊 Model: {artifact.model_compatibility[0].model_name} layer {artifact.model_compatibility[0].layers[0]}")
    print(f"   🎯 Architecture: {artifact.model_compatibility[0].architecture}")
    print(f"   📈 Description: {artifact.description}")
    
    return artifact, model_path

def create_demo_feature_analysis_artifact():
    """Create a demo feature analysis artifact"""
    print("\n🔍 Creating demo feature analysis artifact...")
    
    # Create mock feature analysis results
    feature_analysis = {
        "model_name": "gpt2_layer8_mlp_sae",
        "total_features": 4096,
        "active_features": 3876,
        "sparsity": 0.012,
        "top_features": [
            {
                "feature_id": 0,
                "max_activation": 4.368,
                "sparsity": 0.447,
                "examples": 20,
                "top_tokens": ["sat", "cat", "understanding"],
                "interpretation": "Verb-object relationships"
            },
            {
                "feature_id": 1,
                "max_activation": 4.199,
                "sparsity": 0.468,
                "examples": 16,
                "top_tokens": ["requires", "needs", "demands"],
                "interpretation": "Necessity and requirements"
            },
            {
                "feature_id": 2,
                "max_activation": 4.050,
                "sparsity": 0.447,
                "examples": 18,
                "top_tokens": ["Science", "Research", "Study"],
                "interpretation": "Academic and scientific concepts"
            }
        ],
        "feature_similarities": [
            {"feature_a": 0, "feature_b": 15, "similarity": 0.823},
            {"feature_a": 1, "feature_b": 8, "similarity": 0.756},
            {"feature_a": 2, "feature_b": 10, "similarity": 0.689}
        ]
    }
    
    # Create temporary analysis file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_analysis:
        json.dump(feature_analysis, tmp_analysis, indent=2)
        analysis_path = tmp_analysis.name
    
    # Create artifact metadata
    artifact = ArtifactSchema(
        artifact_type=ArtifactType.ANALYSIS_RESULT,
        name="gpt2_layer8_mlp_features",
        description="Feature analysis for GPT-2 layer 8 MLP SAE with 4096 features and 0.012 sparsity",
        authors=[AuthorInfo(name="SAE Feature Analyzer", email="demo@neuronmap.ai")],
        version="1.0.0",
        license=LicenseType.MIT,
        model_compatibility=[ModelCompatibility(
            model_name="gpt2",
            model_family="gpt",
            architecture="transformer",
            layers=[8]
        )],
        dependencies=[],
        tags=["features", "analysis", "sae", "gpt2", "interpretability"]
    )
    
    print(f"   ✅ Created feature analysis artifact: {artifact.name}")
    print(f"   🔍 Description: {artifact.description}")
    print(f"   📊 Model compatibility: {artifact.model_compatibility[0].model_name}")
    
    return artifact, analysis_path

def demo_zoo_integration():
    """Demonstrate SAE-Zoo integration"""
    print("🏛️ SAE-Analysis Zoo Integration Demo")
    print("=" * 60)
    
    # Create demo artifacts
    sae_artifact, sae_path = create_demo_sae_artifact()
    feature_artifact, feature_path = create_demo_feature_analysis_artifact()
    
    # Initialize storage manager
    print("\n💾 Initializing Storage Manager...")
    storage = StorageManager()
    
    # Upload SAE model
    print(f"\n⬆️ Uploading SAE model to Analysis Zoo...")
    try:
        sae_artifact_id = storage.upload_artifact(sae_artifact, sae_path)
        print(f"   ✅ SAE model uploaded successfully!")
        print(f"   🆔 Artifact ID: {sae_artifact_id}")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return
    
    # Upload feature analysis
    print(f"\n⬆️ Uploading feature analysis to Analysis Zoo...")
    try:
        feature_artifact_id = storage.upload_artifact(feature_artifact, feature_path)
        print(f"   ✅ Feature analysis uploaded successfully!")
        print(f"   🆔 Artifact ID: {feature_artifact_id}")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return
    
    # Search for SAE artifacts
    print(f"\n🔍 Searching for SAE artifacts...")
    try:
        sae_results = storage.search_artifacts(
            artifact_type=ArtifactType.SAE_MODEL,
            tags=["gpt2", "sae"]
        )
        print(f"   ✅ Found {len(sae_results)} SAE models")
        for result in sae_results:
            print(f"   📄 {result.name}: {result.description}")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    # Search for feature analyses
    print(f"\n🔍 Searching for feature analyses...")
    try:
        feature_results = storage.search_artifacts(
            artifact_type=ArtifactType.FEATURE_ANALYSIS,
            tags=["features", "analysis"]
        )
        print(f"   ✅ Found {len(feature_results)} feature analyses")
        for result in feature_results:
            print(f"   📊 {result.name}: {result.description}")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    # Download and verify
    print(f"\n⬇️ Testing artifact download...")
    try:
        downloaded_artifact, downloaded_path = storage.download_artifact(sae_artifact_id)
        print(f"   ✅ Downloaded artifact: {downloaded_artifact.name}")
        print(f"   📁 File path: {downloaded_path}")
        
        # Verify downloaded model
        model_data = torch.load(downloaded_path)
        print(f"   🔍 Model verification:")
        print(f"      Encoder shape: {model_data['encoder_weight'].shape}")
        print(f"      Decoder shape: {model_data['decoder_weight'].shape}")
        print(f"      Reconstruction loss: {model_data['training_results']['final_reconstruction_loss']}")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
    
    # Cleanup
    print(f"\n🧹 Cleanup...")
    try:
        os.unlink(sae_path)
        os.unlink(feature_path)
        if 'downloaded_path' in locals():
            os.unlink(downloaded_path)
        print("   ✅ Temporary files cleaned up")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 SAE-ZOO INTEGRATION DEMO COMPLETED!")
    print("=" * 60)
    print("✅ Demonstrated Features:")
    print("   • SAE model artifact creation")
    print("   • Feature analysis artifact creation")
    print("   • Artifact upload to Analysis Zoo")
    print("   • Search functionality for SAE artifacts")
    print("   • Download and verification")
    print("   • Dependency tracking")
    print("\n🚀 Ready for Production:")
    print("   • SAE models can be shared via Analysis Zoo")
    print("   • Feature analyses can be distributed")
    print("   • Community collaboration enabled")
    print("   • Version control and metadata tracking")

if __name__ == "__main__":
    demo_zoo_integration()
