#!/usr/bin/env python3

import sys
sys.path.append('/home/emilio/Documents/ai/NeuronMap')

from src.utils.progress_tracker import ProgressTracker

# Debug the progress tracker
tracker = ProgressTracker(100, "test_operation")
print("Created tracker")

# Start the tracker
tracker.start()
print("Started tracker")

for i in range(10):
    tracker.update(1, {"step": i})
    print(f"Updated tracker with step {i}")

status = tracker.get_status()
print(f"Status type: {type(status)}")
print(f"Status: {status}")
print(f"completed_steps: {status.get('completed_steps', 'NOT FOUND')}")
print(f"progress_percent: {status.get('progress_percent', 'NOT FOUND')}")
