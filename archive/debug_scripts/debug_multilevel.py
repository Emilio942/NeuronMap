#!/usr/bin/env python3
"""Debug specific validation sections."""

import sys
sys.path.append('/home/emilio/Documents/ai/NeuronMap')

from src.utils.progress_tracker import MultiLevelProgressTracker, ProgressState

print("Testing MultiLevelProgressTracker...")

try:
    # Test 5: Multi-level progress tracker
    ml_tracker = MultiLevelProgressTracker("test_ml")
    print("✓ Created MultiLevelProgressTracker")
    
    ml_tracker.add_level("level1", 5, "Level 1")
    print("✓ Added level1")
    
    ml_tracker.add_level("level2", 10, "Level 2")
    print("✓ Added level2")
    
    ml_tracker.update_level("level1", 1)
    print("✓ Updated level1")
    
    ml_tracker.update_level("level2", 5)
    print("✓ Updated level2")
    
    print("Getting overall status...")
    status = ml_tracker.get_overall_status()
    print(f"Status type: {type(status)}")
    print(f"Status: {status}")
    
    has_overall_progress = hasattr(status, 'overall_progress_percent')
    print(f"Has overall_progress_percent: {has_overall_progress}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Done testing MultiLevelProgressTracker")
