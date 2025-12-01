
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("."))

from src.core.orchestrator import SystemOrchestrator

def test_task_queue():
    print("Initializing Orchestrator with Task Queue...")
    orchestrator = SystemOrchestrator(base_dir="test_workspace_queue")
    
    print("Creating Project...")
    project_id = orchestrator.create_project("Queue Test Project")
    
    print("Submitting Analysis Pipeline (Async)...")
    experiment_id = orchestrator.submit_analysis_pipeline(
        project_id=project_id,
        model_name="gpt2",
        input_data="Testing the queue system.",
        analysis_types=["statistical"]
    )
    print(f"Experiment Submitted: {experiment_id}")
    
    print("Polling for completion...")
    for i in range(30): # Wait up to 30 seconds
        experiment = orchestrator.get_project(project_id)["experiments"]
        # In real usage we'd get the specific experiment, but here we know it's the only one or we check ID
        # Actually project_manager.get_experiment is better
        exp_details = orchestrator.project_manager.get_experiment(project_id, experiment_id)
        
        status = exp_details["status"]
        print(f"Status: {status}")
        
        if status == "completed":
            print("Analysis Completed Successfully!")
            print(f"Results saved at: {exp_details.get('results_path')}")
            break
        elif status == "failed":
            print("Analysis Failed!")
            break
            
        time.sleep(2)

if __name__ == "__main__":
    test_task_queue()
