#!/usr/bin/env python3
"""
Simple test for difficulty assessment system
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing basic imports...")
    
    from src.data_generation.difficulty_analyzer import DifficultyAssessmentEngine
    print("✓ DifficultyAssessmentEngine imported successfully")
    
    # Test engine creation
    engine = DifficultyAssessmentEngine()
    print("✓ Engine created successfully")
    
    # Test basic assessment
    result = engine.assess_difficulty("What is the capital of France?")
    print(f"✓ Basic assessment works: score = {result.difficulty_score}")
    
    print("\n🎉 Basic functionality is working!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
