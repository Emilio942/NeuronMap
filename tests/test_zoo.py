#!/usr/bin/env python3
"""
Test script for Analysis Zoo implementation.

This script tests the core functionality of the Analysis Zoo:
- Artifact schema validation
- Artifact manager operations
- API server startup
- CLI command integration

Run with: python test_zoo.py
"""

import os
import sys
import tempfile
import json
from pathlib import Path
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.zoo import (
    ArtifactSchema,
    ArtifactType,
    LicenseType,
    AuthorInfo,
    ArtifactManager,
    create_sae_artifact_template
)


def create_test_artifact(temp_dir: Path) -> Path:
    """Create a test artifact for testing."""
    artifact_dir = temp_dir / "test_sae_artifact"
    artifact_dir.mkdir()
    
    # Create some test files
    (artifact_dir / "model.safetensors").write_text("fake model weights")
    (artifact_dir / "config.json").write_text('{"dict_size": 16384}')
    (artifact_dir / "README.md").write_text("# Test SAE Model\n\nThis is a test.")
    
    # Create artifact manifest
    author = AuthorInfo(
        name="Test Author",
        email="test@example.com"
    )
    
    artifact = create_sae_artifact_template(
        name="test-gpt2-sae",
        model_name="gpt2",
        layer=8,
        dict_size=16384,
        authors=[author]
    )
    
    return artifact_dir


def test_artifact_schema():
    """Test artifact schema validation."""
    print("🧪 Testing artifact schema...")
    
    try:
        # Create a valid artifact
        author = AuthorInfo(
            name="Test Author",
            email="test@example.com"
        )
        
        artifact = ArtifactSchema(
            name="test-artifact",
            version="1.0.0",
            artifact_type=ArtifactType.SAE_MODEL,
            description="A test artifact for validation",
            authors=[author],
            license=LicenseType.MIT,
            model_compatibility=[],
            files=[],
            total_size_bytes=0
        )
        
        # Test serialization
        artifact_dict = artifact.to_dict()
        assert "uuid" in artifact_dict
        assert artifact_dict["name"] == "test-artifact"
        
        # Test that we can create artifacts with empty files for templates
        assert len(artifact.files) == 0  # Templates start with empty files
        
        print("✅ Artifact schema validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Artifact schema validation failed: {e}")
        return False


def test_artifact_manager():
    """Test artifact manager operations."""
    print("🧪 Testing artifact manager...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create manager
            manager = ArtifactManager(temp_path / "storage")
            
            # Create test artifact
            artifact_dir = create_test_artifact(temp_path)
            
            # Prepare artifact (this will scan files and create manifest)
            author = AuthorInfo(name="Test Author", email="test@example.com")
            template = create_sae_artifact_template(
                name="test-gpt2-sae",
                model_name="gpt2", 
                layer=8,
                dict_size=16384,
                authors=[author]
            )
            
            prepared_artifact = manager.prepare_artifact(artifact_dir, template)
            
            # Validate artifact
            validated_artifact = manager.validate_artifact(artifact_dir)
            assert validated_artifact.name == "test-gpt2-sae"
            assert len(validated_artifact.files) > 0
            
            # Store artifact
            artifact_id = manager.store_artifact(artifact_dir)
            assert artifact_id == validated_artifact.uuid
            
            # Retrieve artifact
            retrieved_artifact = manager.get_artifact(artifact_id)
            assert retrieved_artifact is not None
            assert retrieved_artifact.name == "test-gpt2-sae"
            
            # List artifacts
            search_result = manager.list_artifacts()
            assert search_result.total_count == 1
            assert len(search_result.artifacts) == 1
            
            # Download artifact
            downloaded_path = manager.download_artifact(artifact_id)
            assert downloaded_path.exists()
            
            print("✅ Artifact manager tests passed")
            return True
            
    except Exception as e:
        print(f"❌ Artifact manager tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_server_startup():
    """Test that the API server can start up."""
    print("🧪 Testing API server startup...")
    
    try:
        # Import API server components
        from src.zoo.api_server import create_app
        from src.zoo.auth import get_auth_manager
        
        # Create app
        app = create_app()
        assert app is not None
        
        # Test auth manager
        auth_manager = get_auth_manager()
        assert auth_manager is not None
        
        # Check if default admin user exists
        admin_user = auth_manager.get_user_by_username("admin")
        assert admin_user is not None
        assert "admin" in admin_user.roles
        
        print("✅ API server startup test passed")
        return True
        
    except Exception as e:
        print(f"❌ API server startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_commands():
    """Test CLI command imports."""
    print("🧪 Testing CLI command imports...")
    
    try:
        from src.cli.zoo_commands import zoo
        assert zoo is not None
        
        print("✅ CLI command import test passed")
        return True
        
    except Exception as e:
        print(f"❌ CLI command import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Running Analysis Zoo Tests")
    print("=" * 50)
    
    tests = [
        test_artifact_schema,
        test_artifact_manager,
        test_api_server_startup,
        test_cli_commands
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Analysis Zoo is ready to use.")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start API server: python -m src.zoo.api_server")
        print("3. Test CLI: python main.py zoo --help")
        print("4. Create a user: python main.py zoo create-user")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
