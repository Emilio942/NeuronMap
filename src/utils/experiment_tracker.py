"""Experiment tracking system for NeuronMap analysis."""

import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3
import threading
from datetime import datetime, timedelta
import pickle
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    model_name: str
    analysis_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from an experiment."""
    experiment_id: str
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    start_time: datetime
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    duration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class ExperimentTracker:
    """
    Tracks experiments, their configurations, and results.

    This class provides comprehensive experiment management including:
    - Experiment configuration and metadata tracking
    - Real-time progress monitoring
    - Result storage and retrieval
    - Performance metrics collection
    - Artifact management
    """

    def __init__(self, storage_path: Union[str, Path] = "experiments"):
        """
        Initialize ExperimentTracker.

        Args:
            storage_path: Path to store experiment data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Database for experiment metadata
        self.db_path = self.storage_path / "experiments.db"
        self._init_database()

        # In-memory tracking
        self.active_experiments: Dict[str, ExperimentResult] = {}
        self._lock = threading.Lock()

        logger.info(f"ExperimentTracker initialized with storage: {self.storage_path}")

    def _init_database(self):
        """Initialize SQLite database for experiment storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    model_name TEXT,
                    analysis_type TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration REAL,
                    config_json TEXT,
                    results_json TEXT,
                    metrics_json TEXT,
                    artifacts_json TEXT,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_status
                ON experiments(status)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_model
                ON experiments(model_name)
            """)

    def start_experiment(self,
                        config: ExperimentConfig,
                        experiment_id: Optional[str] = None) -> str:
        """
        Start a new experiment.

        Args:
            config: Experiment configuration
            experiment_id: Optional custom experiment ID

        Returns:
            Experiment ID
        """
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())

        # Create experiment result
        result = ExperimentResult(
            experiment_id=experiment_id,
            status='running',
            start_time=datetime.now()
        )

        # Store in memory
        with self._lock:
            self.active_experiments[experiment_id] = result

        # Store in database
        self._save_experiment_to_db(experiment_id, config, result)

        logger.info(f"Started experiment: {experiment_id} ({config.name})")
        return experiment_id

    def update_experiment(self,
                         experiment_id: str,
                         metrics: Optional[Dict[str, float]] = None,
                         results: Optional[Dict[str, Any]] = None,
                         artifacts: Optional[List[str]] = None):
        """
        Update experiment with new metrics, results, or artifacts.

        Args:
            experiment_id: Experiment ID
            metrics: Performance metrics to add
            results: Results data to add
            artifacts: Artifact paths to add
        """
        with self._lock:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            result = self.active_experiments[experiment_id]

            if metrics:
                result.metrics.update(metrics)

            if results:
                result.results.update(results)

            if artifacts:
                result.artifacts.extend(artifacts)

        # Update database
        self._update_experiment_in_db(experiment_id)

        logger.debug(f"Updated experiment: {experiment_id}")

    def finish_experiment(self,
                         experiment_id: str,
                         status: str = 'completed',
                         error_message: Optional[str] = None):
        """
        Finish an experiment.

        Args:
            experiment_id: Experiment ID
            status: Final status ('completed', 'failed', 'cancelled')
            error_message: Error message if status is 'failed'
        """
        with self._lock:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            result = self.active_experiments[experiment_id]
            result.status = status
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()

            if error_message:
                result.error_message = error_message

            # Move from active to completed
            del self.active_experiments[experiment_id]

        # Update database
        self._update_experiment_in_db(experiment_id)

        logger.info(f"Finished experiment: {experiment_id} (status: {status})")

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment by ID."""
        # Check active experiments first
        with self._lock:
            if experiment_id in self.active_experiments:
                return self.active_experiments[experiment_id]

        # Check database
        return self._load_experiment_from_db(experiment_id)

    def list_experiments(self,
                        status: Optional[str] = None,
                        model_name: Optional[str] = None,
                        limit: int = 100) -> List[ExperimentResult]:
        """
        List experiments with optional filtering.

        Args:
            status: Filter by status
            model_name: Filter by model name
            limit: Maximum number of results

        Returns:
            List of experiment results
        """
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        experiments = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor:
                experiment = self._row_to_experiment_result(row)
                experiments.append(experiment)

        return experiments

    def get_experiment_metrics(self,
                              experiment_ids: Optional[List[str]] = None,
                              metric_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get metrics for multiple experiments.

        Args:
            experiment_ids: List of experiment IDs (None for all)
            metric_names: List of metric names to include (None for all)

        Returns:
            Dictionary mapping experiment_id to metrics
        """
        query = "SELECT id, metrics_json FROM experiments WHERE metrics_json IS NOT NULL"
        params = []

        if experiment_ids:
            placeholders = ",".join("?" * len(experiment_ids))
            query += f" AND id IN ({placeholders})"
            params.extend(experiment_ids)

        result = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            for row in cursor:
                experiment_id, metrics_json = row
                try:
                    metrics = json.loads(metrics_json) if metrics_json else {}

                    # Filter metrics if specified
                    if metric_names:
                        metrics = {k: v for k, v in metrics.items() if k in metric_names}

                    result[experiment_id] = metrics

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metrics for experiment {experiment_id}")

        return result

    def compare_experiments(self,
                           experiment_ids: List[str],
                           metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare experiments across specified metrics.

        Args:
            experiment_ids: List of experiment IDs to compare
            metric_names: List of metrics to compare

        Returns:
            Comparison results
        """
        if len(experiment_ids) < 2:
            raise ValueError("Need at least 2 experiments to compare")

        # Get metrics for all experiments
        all_metrics = self.get_experiment_metrics(experiment_ids, metric_names)

        # Find common metrics
        if not all_metrics:
            return {'error': 'No metrics found for specified experiments'}

        all_metric_names = set()
        for metrics in all_metrics.values():
            all_metric_names.update(metrics.keys())

        if metric_names:
            all_metric_names = all_metric_names.intersection(set(metric_names))

        # Build comparison
        comparison = {
            'experiment_ids': experiment_ids,
            'metrics': {},
            'summary': {}
        }

        for metric_name in all_metric_names:
            values = []
            for exp_id in experiment_ids:
                if exp_id in all_metrics and metric_name in all_metrics[exp_id]:
                    values.append(all_metrics[exp_id][metric_name])
                else:
                    values.append(None)

            comparison['metrics'][metric_name] = dict(zip(experiment_ids, values))

            # Calculate summary statistics
            valid_values = [v for v in values if v is not None]
            if valid_values:
                comparison['summary'][metric_name] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'best_experiment': experiment_ids[values.index(max(valid_values))]
                }

        return comparison

    def export_experiment(self,
                         experiment_id: str,
                         export_path: Union[str, Path],
                         include_artifacts: bool = True) -> None:
        """
        Export experiment data to a file.

        Args:
            experiment_id: Experiment ID
            export_path: Path to export file
            include_artifacts: Whether to include artifact files
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        export_path = Path(export_path)
        export_data = {
            'experiment': experiment.to_dict(),
            'export_timestamp': datetime.now().isoformat(),
            'include_artifacts': include_artifacts
        }

        # Include artifact data if requested
        if include_artifacts and experiment.artifacts:
            export_data['artifact_data'] = {}
            for artifact_path in experiment.artifacts:
                full_path = self.storage_path / artifact_path
                if full_path.exists():
                    try:
                        if full_path.suffix == '.json':
                            with open(full_path, 'r') as f:
                                export_data['artifact_data'][artifact_path] = json.load(f)
                        elif full_path.suffix in ['.pkl', '.pickle']:
                            with open(full_path, 'rb') as f:
                                # Store as base64 for JSON compatibility
                                import base64
                                export_data['artifact_data'][artifact_path] = base64.b64encode(f.read()).decode()
                        else:
                            # For other files, just store metadata
                            export_data['artifact_data'][artifact_path] = {
                                'type': 'file',
                                'size': full_path.stat().st_size,
                                'path': str(full_path)
                            }
                    except Exception as e:
                        logger.warning(f"Failed to include artifact {artifact_path}: {e}")

        # Save export data
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported experiment {experiment_id} to {export_path}")

    def save_artifact(self,
                     experiment_id: str,
                     artifact_name: str,
                     data: Any,
                     format: str = 'pickle') -> str:
        """
        Save an artifact for an experiment.

        Args:
            experiment_id: Experiment ID
            artifact_name: Name of the artifact
            data: Data to save
            format: Format ('pickle', 'json', 'numpy')

        Returns:
            Relative path to saved artifact
        """
        # Create experiment artifact directory
        artifact_dir = self.storage_path / experiment_id
        artifact_dir.mkdir(exist_ok=True)

        # Determine file extension
        if format == 'pickle':
            ext = '.pkl'
        elif format == 'json':
            ext = '.json'
        elif format == 'numpy':
            ext = '.npy'
        else:
            ext = '.dat'

        artifact_path = artifact_dir / f"{artifact_name}{ext}"
        relative_path = str(artifact_path.relative_to(self.storage_path))

        # Save data
        try:
            if format == 'pickle':
                with open(artifact_path, 'wb') as f:
                    pickle.dump(data, f)
            elif format == 'json':
                with open(artifact_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == 'numpy':
                np.save(artifact_path, data)
            else:
                # Raw binary
                with open(artifact_path, 'wb') as f:
                    f.write(data)

            # Update experiment
            self.update_experiment(experiment_id, artifacts=[relative_path])

            logger.debug(f"Saved artifact {artifact_name} for experiment {experiment_id}")
            return relative_path

        except Exception as e:
            logger.error(f"Failed to save artifact {artifact_name}: {e}")
            raise

    def load_artifact(self,
                     experiment_id: str,
                     artifact_name: str,
                     format: str = 'pickle') -> Any:
        """
        Load an artifact for an experiment.

        Args:
            experiment_id: Experiment ID
            artifact_name: Name of the artifact
            format: Format ('pickle', 'json', 'numpy')

        Returns:
            Loaded data
        """
        # Determine file extension
        if format == 'pickle':
            ext = '.pkl'
        elif format == 'json':
            ext = '.json'
        elif format == 'numpy':
            ext = '.npy'
        else:
            ext = '.dat'

        artifact_path = self.storage_path / experiment_id / f"{artifact_name}{ext}"

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact {artifact_name} not found for experiment {experiment_id}")

        try:
            if format == 'pickle':
                with open(artifact_path, 'rb') as f:
                    return pickle.load(f)
            elif format == 'json':
                with open(artifact_path, 'r') as f:
                    return json.load(f)
            elif format == 'numpy':
                return np.load(artifact_path)
            else:
                # Raw binary
                with open(artifact_path, 'rb') as f:
                    return f.read()

        except Exception as e:
            logger.error(f"Failed to load artifact {artifact_name}: {e}")
            raise

    def _save_experiment_to_db(self, experiment_id: str, config: ExperimentConfig, result: ExperimentResult):
        """Save experiment to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (id, name, description, model_name, analysis_type, status, start_time,
                 config_json, results_json, metrics_json, artifacts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                config.name,
                config.description,
                config.model_name,
                config.analysis_type,
                result.status,
                result.start_time.isoformat(),
                json.dumps(asdict(config)),
                json.dumps(result.results),
                json.dumps(result.metrics),
                json.dumps(result.artifacts)
            ))

    def _update_experiment_in_db(self, experiment_id: str):
        """Update experiment in database."""
        # Get current result
        result = None
        with self._lock:
            if experiment_id in self.active_experiments:
                result = self.active_experiments[experiment_id]

        if result is None:
            result = self._load_experiment_from_db(experiment_id)
            if result is None:
                return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments
                SET status = ?, end_time = ?, duration = ?, results_json = ?,
                    metrics_json = ?, artifacts_json = ?, error_message = ?
                WHERE id = ?
            """, (
                result.status,
                result.end_time.isoformat() if result.end_time else None,
                result.duration,
                json.dumps(result.results),
                json.dumps(result.metrics),
                json.dumps(result.artifacts),
                result.error_message,
                experiment_id
            ))

    def _load_experiment_from_db(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_experiment_result(row)

        return None

    def _row_to_experiment_result(self, row) -> ExperimentResult:
        """Convert database row to ExperimentResult."""
        return ExperimentResult(
            experiment_id=row['id'],
            status=row['status'],
            start_time=datetime.fromisoformat(row['start_time']),
            end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
            results=json.loads(row['results_json']) if row['results_json'] else {},
            metrics=json.loads(row['metrics_json']) if row['metrics_json'] else {},
            artifacts=json.loads(row['artifacts_json']) if row['artifacts_json'] else [],
            error_message=row['error_message'],
            duration=row['duration']
        )

    def cleanup_old_experiments(self, older_than_days: int = 30):
        """Clean up old experiments and their artifacts."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        with sqlite3.connect(self.db_path) as conn:
            # Get experiments to delete
            cursor = conn.execute(
                "SELECT id, artifacts_json FROM experiments WHERE start_time < ?",
                (cutoff_date.isoformat(),)
            )

            deleted_count = 0
            for row in cursor:
                experiment_id, artifacts_json = row

                # Delete artifacts
                if artifacts_json:
                    try:
                        artifacts = json.loads(artifacts_json)
                        for artifact_path in artifacts:
                            full_path = self.storage_path / artifact_path
                            if full_path.exists():
                                full_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete artifacts for {experiment_id}: {e}")

                # Delete experiment directory if empty
                exp_dir = self.storage_path / experiment_id
                if exp_dir.exists() and not any(exp_dir.iterdir()):
                    exp_dir.rmdir()

                deleted_count += 1

            # Delete from database
            conn.execute(
                "DELETE FROM experiments WHERE start_time < ?",
                (cutoff_date.isoformat(),)
            )

            logger.info(f"Cleaned up {deleted_count} old experiments")


# Convenience functions
def create_experiment_tracker(storage_path: Union[str, Path] = "experiments") -> ExperimentTracker:
    """Create an ExperimentTracker instance."""
    return ExperimentTracker(storage_path)
