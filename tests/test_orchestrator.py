
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("."))

from src.core.orchestrator import SystemOrchestrator

def test_orchestrator():
    print("Initializing Orchestrator...")
    orchestrator = SystemOrchestrator(base_dir="test_workspace")
    
    print("Creating Project...")
    project_id = orchestrator.create_project("Test Project", "A test project")
    print(f"Project Created: {project_id}")
    
    print("Running Analysis Pipeline...")
    try:
        experiment_id = orchestrator.run_analysis_pipeline(
            project_id=project_id,
            model_name="gpt2",
            input_data="Hello world",
            analysis_types=["statistical"]
        )
        print(f"Experiment Created: {experiment_id}")
        
        experiment = orchestrator.get_project(project_id)["experiments"][0]
        print(f"Experiment ID in Project: {experiment}")
        
    except Exception as e:
        print(f"Pipeline failed (expected if plugins/models missing): {e}")

    print("Test Complete")

if __name__ == "__main__":
    test_orchestrator()
