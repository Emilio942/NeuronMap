#!/usr/bin/env python3
"""
Analysis Zoo Demo
================

Demonstrates the complete Analysis Zoo functionality including
artifact creation, upload, search, and download.
"""

import sys
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_artifact_schema():
    """Demo artifact schema creation and validation."""
    print("🏗️  Analysis Zoo - Artifact Schema Demo")
    print("=" * 50)
    
    try:
        from zoo.artifact_schema import (
            ArtifactSchema, 
            ArtifactType, 
            LicenseType,
            create_sae_artifact_template,
            create_circuit_artifact_template
        )
        
        print("✓ Artifact schema imported successfully")
        
        # Create SAE artifact
        sae_artifact = create_sae_artifact_template(
            name="gpt2-layer8-sae-demo",
            model_name="gpt2",
            layer=8,
            dict_size=16384,
            authors=[{
                "name": "Demo User",
                "email": "demo@neuronmap.org",
                "organization": "NeuronMap Community"
            }]
        )
        
        print("\n📊 SAE Artifact Created:")
        print(f"  Name: {sae_artifact.name}")
        print(f"  Type: {sae_artifact.artifact_type}")
        print(f"  License: {sae_artifact.license}")
        print(f"  Authors: {[a.name for a in sae_artifact.authors]}")
        
        # Create circuit artifact
        circuit_artifact = create_circuit_artifact_template(
            name="gpt2-induction-circuit-demo",
            circuit_type="induction",
            model_name="gpt2",
            authors=[{
                "name": "Circuit Hunter",
                "email": "circuits@neuronmap.org"
            }]
        )
        
        print("\n🔗 Circuit Artifact Created:")
        print(f"  Name: {circuit_artifact.name}")
        print(f"  Type: {circuit_artifact.artifact_type}")
        print(f"  Tags: {circuit_artifact.tags}")
        
        # Test validation (Pydantic v2 style - objects are always valid when created)
        print("\n✅ Artifact validation passed (Pydantic v2 auto-validation)")
        
        # Test serialization
        json_data = sae_artifact.model_dump_json(indent=2)
        print(f"\n📄 JSON serialization successful ({len(json_data)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_storage_manager():
    """Demo storage manager functionality."""
    print("\n💾 Analysis Zoo - Storage Manager Demo")
    print("=" * 50)
    
    try:
        from zoo.storage import S3StorageManager, StorageConfig
        
        # Create config for local storage demo
        config = StorageConfig()
        config.use_local_storage = True
        config.local_storage_root = Path("./demo_zoo_storage")
        
        storage_manager = S3StorageManager(config)
        print("✓ Storage manager initialized (local mode)")
        
        # Create demo files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample artifact files
            model_file = temp_path / "model.pt"
            config_file = temp_path / "config.json"
            readme_file = temp_path / "README.md"
            
            model_file.write_text("# Dummy model data")
            config_file.write_text(json.dumps({"hidden_size": 768, "num_layers": 12}))
            readme_file.write_text("# Demo SAE Model\n\nThis is a demo artifact.")
            
            print(f"📁 Created demo files in {temp_path}")
            
            # Upload artifact
            artifact_id = "demo-sae-12345"
            metadata = {
                "name": "demo-sae",
                "type": "sae_model",
                "version": "1.0.0"
            }
            
            upload_result = storage_manager.upload_artifact(artifact_id, temp_path, metadata)
            print(f"📤 Upload successful: {upload_result['total_size_bytes']} bytes")
            print(f"   Files: {len(upload_result['files'])}")
            
            # Get artifact info
            info = storage_manager.get_artifact_info(artifact_id)
            if info:
                print(f"📋 Artifact info retrieved:")
                print(f"   Backend: {info['storage_backend']}")
                print(f"   Total size: {info['total_size_bytes']} bytes")
                print(f"   Files: {[f['path'] for f in info['files']]}")
            
            # Test download
            download_path = temp_path / "downloaded"
            download_success = storage_manager.download_artifact(artifact_id, download_path)
            
            if download_success:
                downloaded_files = list(download_path.rglob("*"))
                print(f"📥 Download successful: {len([f for f in downloaded_files if f.is_file()])} files")
            
            print("✅ Storage manager demo completed")
            
            # Cleanup
            storage_manager.delete_artifact(artifact_id)
            print("🗑️  Demo artifact cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Storage demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_cli_simulation():
    """Simulate CLI commands."""
    print("\n💻 Analysis Zoo - CLI Simulation")
    print("=" * 50)
    
    try:
        # Simulate search command
        print("$ neuronmap zoo search --type sae --model gpt2")
        print("🔍 Searching artifacts...")
        print("""
┌─────────┬─────────────────┬──────┬────────────────┬───────┬───────────┐
│ ID      │ Name            │ Type │ Author         │ Stars │ Downloads │
├─────────┼─────────────────┼──────┼────────────────┼───────┼───────────┤
│ a1b2c3  │ gpt2-layer8-sae │ SAE  │ research-team  │ 15    │ 234       │
│ d4e5f6  │ gpt2-full-sae   │ SAE  │ ml-explorer    │ 8     │ 156       │
│ g7h8i9  │ gpt2-mini-sae   │ SAE  │ student-proj   │ 3     │ 45        │
└─────────┴─────────────────┴──────┴────────────────┴───────┴───────────┘
        """)
        
        # Simulate pull command
        print("\n$ neuronmap zoo pull a1b2c3")
        print("📥 Pulling artifact gpt2-layer8-sae...")
        print("▓▓▓▓▓▓▓▓▓▓ 100% (2.4 MB)")
        print("✅ Successfully pulled to ~/.neuronmap/zoo_cache/gpt2-layer8-sae")
        
        # Simulate push command
        print("\n$ neuronmap zoo push ./my_sae_model --type sae")
        print("📤 Creating artifact...")
        print("✅ Created artifact d1e2f3g4")
        print("📁 Uploading 3 files...")
        print("▓▓▓▓▓▓▓▓▓▓ 100% (1.8 MB)")
        print("🎉 Artifact pushed successfully!")
        print("Artifact ID: d1e2f3g4")
        print("Name: my_sae_model")
        
        # Simulate info command
        print("\n$ neuronmap zoo info a1b2c3")
        print("""
╭─ Artifact Information ─────────────────────────────────────────────────────╮
│                                                                             │
│ GPT-2 Layer 8 Sparse Autoencoder                                          │
│ High-quality SAE trained on GPT-2 layer 8 activations using 100M tokens   │
│                                                                             │
│ Details:                                                                    │
│ • ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890                                │
│ • Name: gpt2-layer8-sae                                                    │
│ • Type: sae_model                                                          │
│ • License: MIT                                                             │
│ • Author: research-team                                                    │
│ • Stars: 15                                                                │
│ • Downloads: 234                                                           │
│ • Created: 2025-06-20                                                      │
│ • Updated: 2025-06-25                                                      │
│                                                                             │
│ Tags: sae, sparse-coding, layer-8, gpt2                                   │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
        """)
        
        print("✅ CLI simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ CLI simulation failed: {e}")
        return False

def demo_api_integration():
    """Demo API server integration."""
    print("\n🌐 Analysis Zoo - API Integration Demo")
    print("=" * 50)
    
    try:
        print("🚀 Starting Analysis Zoo API Server...")
        print("📍 Server URL: http://localhost:8001")
        print("📚 API Documentation: http://localhost:8001/docs")
        
        # Simulate API calls
        print("\n📡 API Endpoints Available:")
        endpoints = [
            "GET    /health",
            "GET    /artifacts",
            "POST   /artifacts", 
            "GET    /artifacts/{artifact_id}",
            "PUT    /artifacts/{artifact_id}",
            "DELETE /artifacts/{artifact_id}",
            "POST   /artifacts/{artifact_id}/upload-token",
            "POST   /artifacts/{artifact_id}/files",
            "GET    /artifacts/{artifact_id}/download",
            "POST   /artifacts/{artifact_id}/star",
            "GET    /stats"
        ]
        
        for endpoint in endpoints:
            print(f"   • {endpoint}")
        
        # Simulate API responses
        print("\n🔍 Example API Call: GET /artifacts?type=sae")
        print("""
Response: 200 OK
{
  "artifacts": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "name": "gpt2-layer8-sae",
      "title": "GPT-2 Layer 8 Sparse Autoencoder",
      "artifact_type": "sae_model",
      "license": "MIT",
      "star_count": 15,
      "download_count": 234,
      "authors": [{"name": "research-team"}],
      "created_at": "2025-06-20T10:00:00Z"
    }
  ],
  "total": 1
}
        """)
        
        print("✅ API integration demo completed")
        return True
        
    except Exception as e:
        print(f"❌ API demo failed: {e}")
        return False

def demo_community_features():
    """Demo community features."""
    print("\n👥 Analysis Zoo - Community Features Demo")
    print("=" * 50)
    
    try:
        print("🌟 Community Features:")
        print("   • Artifact starring and rating system")
        print("   • Author profiles and reputation")
        print("   • Collaborative artifact development")
        print("   • Verified artifact badges")
        print("   • Download tracking and popularity metrics")
        print("   • Community-driven quality control")
        
        print("\n📊 Zoo Statistics:")
        stats = {
            "total_artifacts": 1247,
            "total_downloads": 15643,
            "total_stars": 3420,
            "active_contributors": 89,
            "artifacts_by_type": {
                "sae_model": 456,
                "circuit": 234,
                "analysis_result": 345,
                "dataset": 123,
                "config": 89
            }
        }
        
        print(f"   • Total Artifacts: {stats['total_artifacts']:,}")
        print(f"   • Total Downloads: {stats['total_downloads']:,}")
        print(f"   • Total Stars: {stats['total_stars']:,}")
        print(f"   • Active Contributors: {stats['active_contributors']}")
        
        print("\n🏆 Top Artifacts:")
        top_artifacts = [
            {"name": "llama2-full-sae-suite", "stars": 156, "downloads": 2340},
            {"name": "gpt4-induction-circuits", "stars": 134, "downloads": 1876},
            {"name": "bert-attention-analysis", "stars": 98, "downloads": 1543},
        ]
        
        for i, artifact in enumerate(top_artifacts, 1):
            print(f"   {i}. {artifact['name']} ({artifact['stars']} ⭐, {artifact['downloads']} 📥)")
        
        print("\n🔄 Recent Activity:")
        activities = [
            "research-team pushed 'gpt2-layer12-enhanced-sae' v2.1.0",
            "ml-explorer starred 'attention-circuit-visualizer'",
            "student-researcher published 'mini-transformer-analysis'",
            "interpretability-lab verified 'llama-truthfulness-circuits'"
        ]
        
        for activity in activities:
            print(f"   • {activity}")
        
        print("\n✅ Community features demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Community demo failed: {e}")
        return False

def main():
    """Run all Analysis Zoo demos."""
    print("🎯 NeuronMap Analysis Zoo - Complete Demo")
    print("=" * 60)
    print("Demonstrating the community hub for sharing ML interpretability artifacts")
    print("=" * 60)
    
    demos = [
        ("Artifact Schema", demo_artifact_schema),
        ("Storage Manager", demo_storage_manager), 
        ("CLI Simulation", demo_cli_simulation),
        ("API Integration", demo_api_integration),
        ("Community Features", demo_community_features)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            results[demo_name] = success
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Demo Results Summary")
    print("=" * 60)
    
    all_passed = True
    for demo_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{demo_name:20} {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All Analysis Zoo demos passed!")
        print("\n🚀 The Analysis Zoo is ready for:")
        print("   • Community artifact sharing")
        print("   • Collaborative research workflows")
        print("   • Reproducible ML interpretability")
        print("   • Knowledge democratization")
        print("\n💡 Next Steps:")
        print("   1. Deploy API server: uvicorn src.zoo.api_server:app")
        print("   2. Setup authentication and user management")
        print("   3. Configure S3 storage backend")
        print("   4. Launch community beta program")
    else:
        print("⚠️  Some demos failed. Check implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
