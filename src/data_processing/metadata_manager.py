"""Metadata and experiment management for NeuronMap."""

from __future__ import annotations

import json
import logging
import time
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class MetadataManager:
    """Persist experiment metadata and provenance information."""

    def __init__(self, location: str = "data"):
        path = Path(location)

        if path.suffix:  # explicit metadata file provided
            self.metadata_file = path
            _ensure_parent(self.metadata_file)
            self.metadata_dir = self.metadata_file.parent
        else:
            self.metadata_dir = path / "metadata"
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_file = self.metadata_dir / "experiments.yaml"

        self.experiments_file = self.metadata_file  # backwards-compatible alias
        self.provenance_file = self.metadata_dir / "provenance.jsonl"

    def create_experiment(self, name: str, description: str,
                          config: Dict[str, Any]) -> str:
        """Create a new experiment record.

        Args:
            name: Experiment name.
            description: Experiment description.
            config: Experiment configuration.

        Returns:
            Experiment ID.
        """
        experiment_id = str(uuid.uuid4())

        experiment_data = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'config': config,
            'created_at': time.time(),
            'status': 'created',
            'datasets': [],
            'models': [],
            'outputs': []
        }

        # Load existing experiments
        return self.record_experiment(
            config_name=name,
            model_name=config.get('model_name', 'unknown'),
            parameters=config,
            description=description,
        )

    def update_experiment_status(self, experiment_id: str, status: str,
                                 metadata: Optional[Dict[str, Any]] = None):
        """Update experiment status.

        Args:
            experiment_id: Experiment ID.
            status: New status.
            metadata: Additional metadata.
        """
        experiments = self._load_experiments()

        if experiment_id in experiments:
            experiment = experiments[experiment_id]
            experiment['status'] = status
            experiment['updated_at'] = time.time()

            if metadata:
                experiment.update(metadata)

            self._save_experiments(experiments)
            logger.info(f"Updated experiment {experiment_id} status to {status}")
        else:
            logger.warning(f"Experiment {experiment_id} not found")

    def log_provenance(self, event_type: str, experiment_id: str,
                       data: Dict[str, Any]):
        """Log a provenance event.

        Args:
            event_type: Type of event (e.g., 'question_generated', 'activation_extracted').
            experiment_id: Associated experiment ID.
            data: Event data.
        """
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'experiment_id': experiment_id,
            'data': data
        }

        with open(self.provenance_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def get_experiment_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get provenance history for an experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            List of provenance events.
        """
        history = []

        if self.provenance_file.exists():
            with open(self.provenance_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('experiment_id') == experiment_id:
                            history.append(event)
                    except json.JSONDecodeError:
                        continue

        return sorted(history, key=lambda x: x['timestamp'])

    def get_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Get all experiments.

        Returns:
            Dictionary of all experiments.
        """
        return self._load_experiments()

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get specific experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Experiment data or None if not found.
        """
        experiments = self._load_experiments()
        return experiments.get(experiment_id)

    # ------------------------------------------------------------------
    # Modern test-facing API -------------------------------------------------
    # ------------------------------------------------------------------
    def record_experiment(self, config_name: str, model_name: Optional[str] = None,
                          parameters: Optional[Dict[str, Any]] = None,
                          description: str = "") -> str:
        experiment_id = str(uuid.uuid4())
        experiments = self._load_experiments()

        parameters = parameters or {}
        resolved_model = (model_name
                          or parameters.get('model')
                          or parameters.get('model_name')
                          or 'unknown')

        experiment = {
            'experiment_id': experiment_id,
            'config_name': config_name,
            'model_name': resolved_model,
            'parameters': parameters,
            'description': description,
            'created_at': time.time(),
            'status': 'created',
            'progress': 0,
        }

        experiments[experiment_id] = experiment
        self._save_experiments(experiments)
        return experiment_id

    def list_experiments(self) -> List[Dict[str, Any]]:
        experiments = self._load_experiments()
        return list(experiments.values())

    def filter_experiments(self, **criteria: Any) -> List[Dict[str, Any]]:
        experiments = self.list_experiments()
        if not criteria:
            return experiments

        results: List[Dict[str, Any]] = []
        for experiment in experiments:
            if all(experiment.get(key) == value for key, value in criteria.items()):
                results.append(experiment)
        return results

    def record_results(self, experiment_id: str, results: Dict[str, Any]) -> None:
        experiments = self._load_experiments()
        if experiment_id not in experiments:
            raise KeyError(f"Experiment {experiment_id} not found")

        experiment = experiments[experiment_id]
        experiment.setdefault('results', {}).update(results)
        experiment['updated_at'] = time.time()
        experiment['status'] = results.get('status', experiment.get('status', 'completed'))
        self._save_experiments(experiments)

    def _load_experiments(self) -> Dict[str, Any]:
        """Load experiments from file."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_experiments(self, experiments: Dict[str, Any]):
        """Save experiments to file."""
        with open(self.experiments_file, 'w') as f:
            yaml.dump(experiments, f, default_flow_style=False)


class ExperimentTracker:
    """High-level helper that wraps :class:`MetadataManager` for tests."""

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manager = MetadataManager(self.base_dir / "experiments.yaml")

    def start_experiment(self, name: str, config: Dict[str, Any]) -> str:
        experiment_id = self._manager.record_experiment(
            config_name=name,
            model_name=config.get('model', 'unknown'),
            parameters=config,
            description=config.get('description', ''),
        )
        self._manager.update_experiment_status(experiment_id, 'running', {'progress': 0})
        return experiment_id

    def log_progress(self, experiment_id: str, stage: str, progress: int) -> None:
        metadata = {
            'last_stage': stage,
            'progress': max(0, min(100, progress)),
        }
        self._manager.update_experiment_status(experiment_id, 'running', metadata)

    def finish_experiment(self, experiment_id: str, results: Dict[str, Any]) -> None:
        results = dict(results)
        results.setdefault('status', 'completed')
        results.setdefault('completed_at', time.time())
        self._manager.record_results(experiment_id, results)
        self._manager.update_experiment_status(experiment_id, 'completed', {'progress': 100})

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        experiment = self._manager.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment {experiment_id} not found")
        return experiment


class DatasetVersionManager:
    """Manage dataset versions and checksums."""

    def __init__(self, data_dir: str = "data"):
        """Initialize dataset version manager.

        Args:
            data_dir: Base data directory.
        """
        self.data_dir = Path(data_dir)
        self.versions_file = self.data_dir / "dataset_versions.yaml"

    def create_version(self, dataset_name: str, file_path: str,
                       description: str = "") -> str:
        """Create a new dataset version.

        Args:
            dataset_name: Name of the dataset.
            file_path: Path to dataset file.
            description: Version description.

        Returns:
            Version ID.
        """
        version_id = f"v{int(time.time())}"
        checksum = self._calculate_checksum(file_path)

        version_data = {
            'id': version_id,
            'dataset_name': dataset_name,
            'file_path': str(file_path),
            'description': description,
            'checksum': checksum,
            'created_at': time.time(),
            'size_bytes': Path(file_path).stat().st_size
        }

        # Load existing versions
        versions = self._load_versions()
        if dataset_name not in versions:
            versions[dataset_name] = []

        versions[dataset_name].append(version_data)

        # Save updated versions
        self._save_versions(versions)

        logger.info(f"Created dataset version: {dataset_name} {version_id}")
        return version_id

    def verify_version(self, dataset_name: str, version_id: str) -> bool:
        """Verify dataset version integrity.

        Args:
            dataset_name: Name of the dataset.
            version_id: Version ID to verify.

        Returns:
            True if version is valid.
        """
        versions = self._load_versions()

        if dataset_name not in versions:
            return False

        for version in versions[dataset_name]:
            if version['id'] == version_id:
                file_path = version['file_path']
                if not Path(file_path).exists():
                    return False

                current_checksum = self._calculate_checksum(file_path)
                return current_checksum == version['checksum']

        return False

    def get_latest_version(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Latest version data or None.
        """
        versions = self._load_versions()

        if dataset_name not in versions or not versions[dataset_name]:
            return None

        # Sort by creation time and return latest
        sorted_versions = sorted(versions[dataset_name],
                                 key=lambda x: x['created_at'], reverse=True)
        return sorted_versions[0]

    def list_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """List all versions of a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            List of version data.
        """
        versions = self._load_versions()
        return versions.get(dataset_name, [])

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _load_versions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load versions from file."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_versions(self, versions: Dict[str, List[Dict[str, Any]]]):
        """Save versions to file."""
        with open(self.versions_file, 'w') as f:
            yaml.dump(versions, f, default_flow_style=False)
