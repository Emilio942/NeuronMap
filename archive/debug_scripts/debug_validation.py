#!/usr/bin/env python3
"""Quick debug version of monitoring validation."""

import sys
sys.path.append('/home/emilio/Documents/ai/NeuronMap')

from src.utils.progress_tracker import ProgressTracker, MultiLevelProgressTracker

print("Testing progress tracking only...")

# Test 1: Basic progress tracker creation
try:
    tracker = ProgressTracker(100, "test_operation")
    tracker.start()
    print("✓ Progress tracker creation")
except Exception as e:
    print(f"✗ Progress tracker creation: {e}")
    exit(1)

# Test 2: Progress updates
try:
    for i in range(10):
        tracker.update(1, {"step": i})
    status = tracker.get_status()
    success = status['completed_steps'] == 10
    print(f"✓ Progress updates: {success}")
except Exception as e:
    print(f"✗ Progress updates: {e}")

# Test 3: ETA calculation
try:
    status = tracker.get_status()
    has_eta = 'eta_seconds' in status and status['eta_seconds'] >= 0
    print(f"✓ ETA calculation: {has_eta}")
except Exception as e:
    print(f"✗ ETA calculation: {e}")

# Test 4: Progress percentage calculation  
try:
    status = tracker.get_status()
    correct_percentage = abs(status['progress_percent'] - 10.0) < 0.1
    print(f"✓ Progress percentage calculation: {correct_percentage}")
except Exception as e:
    print(f"✗ Progress percentage calculation: {e}")

# Test 5: Multi-level progress tracker
try:
    ml_tracker = MultiLevelProgressTracker("test_ml")
    print("✓ Multi-level progress tracker creation")
except Exception as e:
    print(f"✗ Multi-level progress tracker creation: {e}")

print("Basic tests completed!")
