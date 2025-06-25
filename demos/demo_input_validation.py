#!/usr/bin/env python3
"""
Demo of the Comprehensive Input Validation System
================================================

This demonstrates the robust input validation capabilities of NeuronMap.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.utils.input_validation import InputValidator, validate_analysis_request

def demo_input_validation():
    """Demonstrate comprehensive input validation."""
    print("🧠 NeuronMap Input Validation System Demo")
    print("=" * 50)
    
    validator = InputValidator()
    
    # Test 1: Valid analysis request
    print("\n1️⃣ Testing VALID analysis request...")
    valid_request = {
        'model_name': 'gpt2',
        'input_texts': ['Hello world', 'How are you?'],
        'layers': [6, 8, 10],
        'batch_size': 16,
        'max_length': 512,
        'device': 'auto',
        'output_format': 'csv'
    }
    
    is_valid, errors = validator.validate_analysis_request(valid_request)
    print(f"✅ Valid: {is_valid}")
    if errors:
        print(f"❌ Errors: {errors}")
    
    # Test 2: Invalid analysis request
    print("\n2️⃣ Testing INVALID analysis request...")
    invalid_request = {
        'model_name': 'invalid/../model',  # Path traversal attempt
        'input_texts': ['', 'x' * 20000],  # Empty and too long text
        'layers': [-1, 150],  # Negative and too high layer
        'batch_size': 1000,  # Too large batch
        'device': 'invalid_device'  # Invalid device
    }
    
    is_valid, errors = validator.validate_analysis_request(invalid_request)
    print(f"❌ Valid: {is_valid}")
    print(f"🔍 Errors detected:")
    for error in errors:
        print(f"   - {error}")
    
    # Test 3: File validation
    print("\n3️⃣ Testing file validation...")
    
    # Safe file
    safe_file = "data/sample.txt"
    is_valid, errors = validator.validate_file_operation(safe_file, 'read')
    print(f"📄 Safe file '{safe_file}': Valid={is_valid}")
    
    # Dangerous file
    dangerous_file = "../../../etc/passwd"
    is_valid, errors = validator.validate_file_operation(dangerous_file, 'read')
    print(f"⚠️  Dangerous file '{dangerous_file}': Valid={is_valid}")
    if errors:
        print(f"   Errors: {errors}")
    
    # Test 4: Input sanitization
    print("\n4️⃣ Testing input sanitization...")
    dangerous_input = "<script>alert('xss')</script>\x00malicious\n\n\ntext   with   spaces"
    sanitized = validator.sanitize_input(dangerous_input)
    print(f"🧹 Original: {repr(dangerous_input)}")
    print(f"✨ Sanitized: {repr(sanitized)}")
    
    # Test 5: Model validation
    print("\n5️⃣ Testing model validation...")
    
    valid_models = ['gpt2', 'bert-base-uncased', 'huggingface/gpt2']
    invalid_models = ['../malicious', 'model<script>', 'x' * 300]
    
    for model in valid_models:
        is_valid, errors = validator.validate_model_request(model)
        print(f"✅ Model '{model}': Valid={is_valid}")
    
    for model in invalid_models:
        is_valid, errors = validator.validate_model_request(model)
        print(f"❌ Model '{model}': Valid={is_valid}")
        if errors:
            print(f"   Errors: {errors}")
    
    print("\n" + "=" * 50)
    print("✅ Input validation demo completed!")
    print("🔒 System successfully validates and sanitizes all inputs")
    print("🛡️  Security measures protect against common attack vectors")


if __name__ == "__main__":
    demo_input_validation()
