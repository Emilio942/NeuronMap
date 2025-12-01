"""
System Orchestrator
==================

Central coordination layer for NeuronMap.
Integrates Project Management, Plugin System, and Caching.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.core.project_manager import ProjectManager
from src.core.plugin_system import PluginManager
from src.storage.real_advanced_cache import RealAdvancedCache
from src.core.task_queue import create_task_queue, TaskStatus

logger = logging.getLogger(__name__)

class SystemOrchestrator:
    """
    The central brain of NeuronMap.
    Coordinates all subsystems to execute user intents.
    """
    
    def __init__(self, base_dir: str = ".", queue_type: str = "local"):
        self.base_dir = Path(base_dir)
        
        # Initialize Subsystems
        self.project_manager = ProjectManager(base_dir=str(self.base_dir / "projects"))
        self.plugin_manager = PluginManager(plugin_dir=str(self.base_dir / "plugins"))
        self.cache = RealAdvancedCache(disk_cache_dir=str(self.base_dir / "cache"))
        
        # Initialize Task Queue
        # If redis is requested but fails, we could fallback, but for now let's be explicit
        try:
            self.task_queue = create_task_queue(queue_type, max_workers=4)
            logger.info(f"Task queue initialized: {queue_type}")
        except Exception as e:
            logger.error(f"Failed to initialize {queue_type} queue: {e}. Falling back to local.")
            self.task_queue = create_task_queue("local", max_workers=4)
        
        # Load Plugins
        self.plugin_manager.discover_plugins()
        
        logger.info("System Orchestrator initialized")

    # ------------------------------------------------------------------
    # Project Management Facade
    # ------------------------------------------------------------------
    
    def create_project(self, name: str, description: str = "") -> str:
        return self.project_manager.create_project(name, description)
        
    def list_projects(self) -> List[Dict[str, Any]]:
        return self.project_manager.list_projects()
        
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self.project_manager.get_project(project_id)

    # ------------------------------------------------------------------
    # Analysis Workflow
    # ------------------------------------------------------------------

    def submit_analysis_pipeline(self, 
                            project_id: str, 
                            model_name: str, 
                            input_data: Any, 
                            analysis_types: List[str]) -> str:
        """
        Submit an analysis pipeline to the task queue.
        Returns the experiment_id (which serves as the task tracking ID).
        """
        # 1. Create Experiment immediately to reserve ID
        config = {
            "model_name": model_name,
            "analysis_types": analysis_types
        }
        experiment_id = self.project_manager.create_experiment(
            project_id, 
            f"Analysis of {model_name}", 
            config
        )
        
        # 2. Submit to Task Queue
        # We pass the experiment_id so the worker knows where to save results
        # We use the standalone function to avoid pickling the entire Orchestrator instance
        self.task_queue.submit(
            run_analysis_task,
            str(self.base_dir), # Pass base_dir to worker
            project_id,
            experiment_id,
            model_name,
            input_data,
            analysis_types
        )
        
        logger.info(f"Analysis submitted: {experiment_id}")
        return experiment_id

    def _run_analysis_worker(self, 
                           project_id: str, 
                           experiment_id: str,
                           model_name: str, 
                           input_data: Any, 
                           analysis_types: List[str]):
        """
        Worker function that executes the analysis pipeline.
        """
        try:
            self.project_manager.update_experiment_status(project_id, experiment_id, "running")
            
            # Check Cache first
            cache_key = f"analysis:{model_name}:{hash(str(input_data))}"
            cached_results = self.cache.get(cache_key)
            
            if cached_results:
                logger.info("Using cached analysis results")
                results = cached_results
            else:
                # Use ModelAdapterPlugin to load model and extract activations
                activations_data = self._extract_activations(model_name, input_data)
                
                # Extract the raw activations list for analysis plugins
                if isinstance(activations_data, dict) and "activations" in activations_data:
                    activations = activations_data["activations"]
                    results = activations_data
                else:
                    activations = activations_data
                    results = {"activations": activations}
                
                # Run requested analysis plugins
                for analysis_type in analysis_types:
                    plugin_name = None
                    if analysis_type == "statistical":
                        plugin_name = "statistical_analysis"
                    elif analysis_type == "performance":
                        plugin_name = "performance_monitor"
                    
                    if plugin_name:
                        plugin = self.plugin_manager.plugins.get(plugin_name)
                        if plugin:
                            logger.info(f"Running analysis plugin: {plugin_name}")
                            analysis_results = plugin.execute(activations)
                            results[analysis_type] = analysis_results
                
                # Cache results
                self.cache.set(cache_key, results)

            # Save to Experiment
            self.project_manager.update_experiment_status(
                project_id, 
                experiment_id, 
                "completed", 
                results
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            self.project_manager.update_experiment_status(
                project_id, 
                experiment_id, 
                "failed", 
                {"error": str(e)}
            )
            raise

    # Deprecated synchronous method (kept for compatibility if needed, or removed)
    def run_analysis_pipeline(self, *args, **kwargs):
        """Deprecated. Use submit_analysis_pipeline instead."""
        return self.submit_analysis_pipeline(*args, **kwargs)

    def _extract_activations(self, model_name: str, input_data: Any) -> Any:
        """
        Internal helper to extract activations using the ModelAdapterPlugin.
        """
        # Find a suitable model adapter
        # In a real system, we might have multiple adapters (HF, ONNX, etc.)
        # We'll look for the HuggingFace adapter first
        
        adapter_name = "hf_adapter_HuggingFaceAdapter" # Name generated by plugin loader
        
        # If exact name not found, search by type
        if adapter_name not in self.plugin_manager.plugins:
            adapters = self.plugin_manager.get_plugins_by_type("model_adapter")
            if adapters:
                adapter_name = adapters[0]
            else:
                logger.warning("No model adapter found. Using mock data.")
                import numpy as np
                return np.random.rand(10, 768).tolist()
        
        adapter = self.plugin_manager.plugins[adapter_name]
        logger.info(f"Using model adapter: {adapter_name}")
        
        # Load model and extract
        model_bundle = adapter.load_model(model_name, {})
        return adapter.extract_activations(model_bundle, input_data)

    # ------------------------------------------------------------------
    # Visualization Workflow
    # ------------------------------------------------------------------
    
    def generate_visualization(self, project_id: str, experiment_id: str, viz_type: str) -> str:
        """
        Generate a visualization for an experiment result.
        """
        experiment = self.project_manager.get_experiment(project_id, experiment_id)
        if not experiment or experiment["status"] != "completed":
            raise ValueError("Experiment not found or not completed")
            
        # Load results
        results_path = experiment.get("results_path")
        if not results_path:
            raise ValueError("No results found for experiment")
            
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        # Use Visualization Plugin
        viz_plugin = self.plugin_manager.plugins.get("advanced_visualization")
        if not viz_plugin:
            raise ValueError("Visualization plugin not available")
            
        # Output path
        exp_dir = self.project_manager.get_experiment_dir(project_id, experiment_id)
        output_dir = exp_dir / "visualizations"
        
        config = {
            "type": viz_type,
            "output_dir": str(output_dir)
        }
        
        # Assuming the plugin takes raw data or specific part of results
        # Here we pass the whole results dict for flexibility
        viz_path = viz_plugin.execute(results.get("activations", []), config)
        return viz_path

# ------------------------------------------------------------------
# Standalone Task Functions (Picklable)
# ------------------------------------------------------------------

def run_analysis_task(base_dir: str,
                     project_id: str, 
                     experiment_id: str,
                     model_name: str, 
                     input_data: Any, 
                     analysis_types: List[str]):
    """
    Standalone task function for async execution.
    Instantiates a fresh Orchestrator to avoid pickling issues with the main instance.
    """
    # Create a fresh orchestrator instance
    # Note: We use default paths. In a complex setup, we might need to pass config.
    orchestrator = SystemOrchestrator(base_dir=base_dir)
    return orchestrator._run_analysis_worker(
        project_id, experiment_id, model_name, input_data, analysis_types
    )

