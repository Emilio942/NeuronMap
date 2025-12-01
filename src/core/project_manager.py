"""
Project Manager Module
=====================

Manages projects, experiments, and their associated data.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ProjectManager:
    """
    Manages the lifecycle of projects and experiments.
    
    Structure:
    projects/
      <project_id>/
        metadata.json
        experiments/
          <experiment_id>/
            metadata.json
            results/
            visualizations/
            logs/
    """
    
    def __init__(self, base_dir: str = "projects"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.active_project_id: Optional[str] = None
        
    def create_project(self, name: str, description: str = "") -> str:
        """Create a new project."""
        project_id = str(uuid.uuid4())
        project_dir = self.base_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "experiments": []
        }
        
        self._save_project_metadata(project_id, metadata)
        logger.info(f"Created project: {name} ({project_id})")
        return project_id

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project metadata."""
        metadata_path = self.base_dir / project_id / "metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        projects = []
        for project_dir in self.base_dir.iterdir():
            if project_dir.is_dir():
                metadata = self.get_project(project_dir.name)
                if metadata:
                    projects.append(metadata)
        return projects

    def create_experiment(self, project_id: str, name: str, config: Dict[str, Any]) -> str:
        """Create a new experiment within a project."""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
            
        experiment_id = str(uuid.uuid4())
        experiment_dir = self.base_dir / project_id / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        (experiment_dir / "results").mkdir()
        (experiment_dir / "visualizations").mkdir()
        (experiment_dir / "logs").mkdir()
        
        metadata = {
            "id": experiment_id,
            "project_id": project_id,
            "name": name,
            "config": config,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self._save_experiment_metadata(project_id, experiment_id, metadata)
        
        # Update project metadata
        project["experiments"].append(experiment_id)
        project["updated_at"] = datetime.now().isoformat()
        self._save_project_metadata(project_id, project)
        
        logger.info(f"Created experiment: {name} ({experiment_id}) in project {project_id}")
        return experiment_id

    def get_experiment(self, project_id: str, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment metadata."""
        metadata_path = self.base_dir / project_id / "experiments" / experiment_id / "metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, "r") as f:
            return json.load(f)

    def update_experiment_status(self, project_id: str, experiment_id: str, status: str, results: Optional[Dict] = None):
        """Update experiment status and results."""
        metadata = self.get_experiment(project_id, experiment_id)
        if not metadata:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        metadata["status"] = status
        metadata["updated_at"] = datetime.now().isoformat()
        
        if results:
            # Save results to file instead of metadata to keep it light
            results_path = self.base_dir / project_id / "experiments" / experiment_id / "results" / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            metadata["results_path"] = str(results_path)
            
        self._save_experiment_metadata(project_id, experiment_id, metadata)

    def _save_project_metadata(self, project_id: str, metadata: Dict[str, Any]):
        """Save project metadata to disk."""
        path = self.base_dir / project_id / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_experiment_metadata(self, project_id: str, experiment_id: str, metadata: Dict[str, Any]):
        """Save experiment metadata to disk."""
        path = self.base_dir / project_id / "experiments" / experiment_id / "metadata.json"
        # Atomic write to prevent race conditions/partial writes
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        temp_path.replace(path)

    def get_experiment_dir(self, project_id: str, experiment_id: str) -> Path:
        """Get the directory path for an experiment."""
        return self.base_dir / project_id / "experiments" / experiment_id
