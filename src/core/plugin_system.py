"""
Advanced Plugin Architecture
===========================

Extensible plugin system for custom analysis modules, model adapters,
and visualization components.
"""

import importlib
import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Union
import json
import sys
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: str
    dependencies: List[str]
    tags: List[str]
    created_at: str
    updated_at: str

class PluginBase(ABC):
    """Base class for all plugins."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the main plugin functionality."""
        pass

    def cleanup(self):
        """Cleanup resources when plugin is unloaded."""
        pass

class AnalysisPlugin(PluginBase):
    """Base class for analysis plugins."""

    @abstractmethod
    def analyze(self, activations: Any, **kwargs) -> Dict[str, Any]:
        """Perform custom analysis on activations."""
        pass

class ModelAdapterPlugin(PluginBase):
    """Base class for model adapter plugins."""

    @abstractmethod
    def load_model(self, model_name: str, config: Dict[str, Any]) -> Any:
        """Load and return the model."""
        pass

    @abstractmethod
    def extract_activations(self, model: Any, inputs: Any, layer_names: List[str]) -> Dict[str, Any]:
        """Extract activations from the model."""
        pass

class VisualizationPlugin(PluginBase):
    """Base class for visualization plugins."""

    @abstractmethod
    def create_visualization(self, data: Any, config: Dict[str, Any]) -> str:
        """Create and return path to visualization."""
        pass

class PluginManager:
    """Advanced plugin management system."""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)

        # Plugin registry
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_types: Dict[str, List[str]] = {
            'analysis': [],
            'model_adapter': [],
            'visualization': [],
            'custom': []
        }

        # Plugin hooks
        self.hooks: Dict[str, List[Callable]] = {
            'before_analysis': [],
            'after_analysis': [],
            'before_visualization': [],
            'after_visualization': [],
            'on_error': []
        }

        # Initialize built-in plugins
        self._register_builtin_plugins()

        # Discover and load external plugins
        self.discover_plugins()

    def _register_builtin_plugins(self):
        """Register built-in plugins."""
        # Statistical Analysis Plugin
        stat_plugin = StatisticalAnalysisPlugin()
        self.register_plugin('statistical_analysis', stat_plugin)

        # Advanced Visualization Plugin
        viz_plugin = AdvancedVisualizationPlugin()
        self.register_plugin('advanced_visualization', viz_plugin)

        # Performance Monitor Plugin
        perf_plugin = PerformanceMonitorPlugin()
        self.register_plugin('performance_monitor', perf_plugin)

    def register_plugin(self, name: str, plugin: PluginBase) -> bool:
        """Register a plugin."""
        try:
            metadata = plugin.get_metadata()

            # Validate plugin
            if not self._validate_plugin(plugin, metadata):
                logger.error(f"Plugin validation failed for {name}")
                return False

            # Initialize plugin
            if not plugin.initialize({}):
                logger.error(f"Plugin initialization failed for {name}")
                return False

            # Register plugin
            self.plugins[name] = plugin
            self.plugin_metadata[name] = metadata

            # Add to type registry
            plugin_type = metadata.plugin_type
            if plugin_type in self.plugin_types:
                self.plugin_types[plugin_type].append(name)
            else:
                self.plugin_types['custom'].append(name)

            logger.info(f"Successfully registered plugin: {name}")
            return True

        except Exception as e:
            logger.error(f"Error registering plugin {name}: {e}")
            return False

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin."""
        try:
            if name in self.plugins:
                plugin = self.plugins[name]
                plugin.cleanup()

                # Remove from registries
                del self.plugins[name]
                del self.plugin_metadata[name]

                # Remove from type registry
                for plugin_type, plugin_list in self.plugin_types.items():
                    if name in plugin_list:
                        plugin_list.remove(name)

                logger.info(f"Successfully unregistered plugin: {name}")
                return True
            else:
                logger.warning(f"Plugin {name} not found")
                return False

        except Exception as e:
            logger.error(f"Error unregistering plugin {name}: {e}")
            return False

    def _validate_plugin(self, plugin: PluginBase, metadata: PluginMetadata) -> bool:
        """Validate plugin implementation."""
        try:
            # Check required methods
            required_methods = ['get_metadata', 'initialize', 'execute']
            for method in required_methods:
                if not hasattr(plugin, method):
                    logger.error(f"Plugin missing required method: {method}")
                    return False

            # Check metadata
            if not metadata.name or not metadata.version:
                logger.error("Plugin metadata incomplete")
                return False

            # Check dependencies
            for dep in metadata.dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    logger.error(f"Plugin dependency not found: {dep}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Plugin validation error: {e}")
            return False

    def discover_plugins(self):
        """Discover plugins in the plugin directory."""
        try:
            # Look for Python files in plugin directory
            for plugin_file in self.plugin_dir.glob("*.py"):
                if plugin_file.name.startswith('__'):
                    continue

                self._load_plugin_from_file(plugin_file)

            # Look for plugin packages
            for plugin_dir in self.plugin_dir.iterdir():
                if plugin_dir.is_dir() and not plugin_dir.name.startswith('__'):
                    plugin_init = plugin_dir / "__init__.py"
                    if plugin_init.exists():
                        self._load_plugin_from_package(plugin_dir)

        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")

    def _load_plugin_from_file(self, plugin_file: Path):
        """Load plugin from a Python file."""
        try:
            module_name = plugin_file.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginBase) and obj != PluginBase:
                    plugin_instance = obj()
                    self.register_plugin(f"{module_name}_{name}", plugin_instance)

        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_file}: {e}")

    def _load_plugin_from_package(self, plugin_dir: Path):
        """Load plugin from a package directory."""
        try:
            module_name = plugin_dir.name
            sys.path.insert(0, str(self.plugin_dir))

            module = importlib.import_module(module_name)

            # Look for plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginBase) and obj != PluginBase:
                    plugin_instance = obj()
                    self.register_plugin(f"{module_name}_{name}", plugin_instance)

        except Exception as e:
            logger.error(f"Error loading plugin package {plugin_dir}: {e}")
        finally:
            if str(self.plugin_dir) in sys.path:
                sys.path.remove(str(self.plugin_dir))

    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a specific plugin."""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not found")

        try:
            plugin = self.plugins[plugin_name]
            return plugin.execute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing plugin {plugin_name}: {e}")
            self._trigger_hook('on_error', plugin_name, e)
            raise

    def get_plugins_by_type(self, plugin_type: str) -> List[str]:
        """Get all plugins of a specific type."""
        return self.plugin_types.get(plugin_type, [])

    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata for a specific plugin."""
        return self.plugin_metadata.get(plugin_name)

    def list_plugins(self) -> Dict[str, PluginMetadata]:
        """List all registered plugins."""
        return self.plugin_metadata.copy()

    def add_hook(self, hook_name: str, callback: Callable):
        """Add a hook callback."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)

    def _trigger_hook(self, hook_name: str, *args, **kwargs):
        """Trigger all callbacks for a specific hook."""
        for callback in self.hooks.get(hook_name, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook {hook_name}: {e}")

    def load_builtin_plugins(self):
        """Load built-in plugins from the plugins/builtin directory."""
        try:
            builtin_dir = Path(__file__).parent.parent.parent / "plugins" / "builtin"
            if not builtin_dir.exists():
                logger.warning("Built-in plugins directory not found")
                return

            # Add to Python path
            import sys
            if str(builtin_dir.parent) not in sys.path:
                sys.path.insert(0, str(builtin_dir.parent))

            # Load each built-in plugin
            for plugin_file in builtin_dir.glob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue

                try:
                    self.load_plugin_from_file(str(plugin_file))
                    logger.info(f"Loaded built-in plugin: {plugin_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load built-in plugin {plugin_file.name}: {e}")

            logger.info("Built-in plugins loading complete")

        except Exception as e:
            logger.error(f"Error loading built-in plugins: {e}")

    def load_plugin_from_file(self, file_path: str) -> bool:
        """Load a plugin from a Python file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Plugin file not found: {file_path}")
                return False

            # Import the module
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for {file_path}")
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path.stem] = module
            spec.loader.exec_module(module)

            # Find plugin classes in the module
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginBase) and
                    obj != PluginBase and
                    obj not in [AnalysisPlugin, ModelAdapterPlugin, VisualizationPlugin]):
                    plugin_classes.append((name, obj))

            if not plugin_classes:
                logger.warning(f"No plugin classes found in {file_path}")
                return False

            # Register all found plugin classes
            success_count = 0
            for class_name, plugin_class in plugin_classes:
                try:
                    # Create instance with default config
                    plugin_instance = plugin_class()
                    if plugin_instance.initialize({}):
                        plugin_name = f"{file_path.stem}_{class_name}"
                        if self.register_plugin(plugin_name, plugin_instance):
                            success_count += 1
                            logger.info(f"Registered plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Error instantiating plugin {class_name}: {e}")

            return success_count > 0

        except Exception as e:
            logger.error(f"Error loading plugin from file {file_path}: {e}")
            return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        return self.unregister_plugin(name)

    # ============================================================================
    # BUILT-IN PLUGIN LOADING
    # ============================================================================

# Built-in plugins

class StatisticalAnalysisPlugin(AnalysisPlugin):
    """Built-in statistical analysis plugin."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Statistical Analysis",
            version="1.0.0",
            author="NeuronMap Team",
            description="Advanced statistical analysis of neural activations",
            plugin_type="analysis",
            dependencies=["numpy", "scipy"],
            tags=["statistics", "analysis"],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            import numpy as np
            import scipy.stats as stats
            self.np = np
            self.stats = stats
            return True
        except ImportError:
            return False

    def execute(self, *args, **kwargs) -> Any:
        return self.analyze(*args, **kwargs)

    def analyze(self, activations: Any, **kwargs) -> Dict[str, Any]:
        """Perform statistical analysis on activations."""
        try:
            activations = self.np.array(activations)

            return {
                'mean': float(self.np.mean(activations)),
                'std': float(self.np.std(activations)),
                'var': float(self.np.var(activations)),
                'min': float(self.np.min(activations)),
                'max': float(self.np.max(activations)),
                'median': float(self.np.median(activations)),
                'skewness': float(self.stats.skew(activations.flatten())),
                'kurtosis': float(self.stats.kurtosis(activations.flatten())),
                'percentiles': {
                    '25': float(self.np.percentile(activations, 25)),
                    '75': float(self.np.percentile(activations, 75)),
                    '95': float(self.np.percentile(activations, 95)),
                    '99': float(self.np.percentile(activations, 99))
                }
            }
        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return {}

class AdvancedVisualizationPlugin(VisualizationPlugin):
    """Built-in advanced visualization plugin."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Advanced Visualization",
            version="1.0.0",
            author="NeuronMap Team",
            description="Create advanced interactive visualizations",
            plugin_type="visualization",
            dependencies=["plotly", "matplotlib"],
            tags=["visualization", "interactive"],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            import plotly.graph_objects as go
            import matplotlib.pyplot as plt
            self.go = go
            self.plt = plt
            return True
        except ImportError:
            return False

    def execute(self, *args, **kwargs) -> Any:
        return self.create_visualization(*args, **kwargs)

    def create_visualization(self, data: Any, config: Dict[str, Any]) -> str:
        """Create advanced visualization."""
        try:
            viz_type = config.get('type', 'heatmap')
            output_dir = Path(config.get('output_dir', 'data/outputs/visualizations'))
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{viz_type}_{timestamp}.html"

            if viz_type == 'heatmap':
                fig = self.go.Figure(data=self.go.Heatmap(z=data))
                fig.update_layout(title="Advanced Activation Heatmap")
                fig.write_html(str(output_file))

            return str(output_file)

        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return ""

class PerformanceMonitorPlugin(AnalysisPlugin):
    """Built-in performance monitoring plugin."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Performance Monitor",
            version="1.0.0",
            author="NeuronMap Team",
            description="Monitor analysis performance and resource usage",
            plugin_type="analysis",
            dependencies=["psutil", "time"],
            tags=["performance", "monitoring"],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            import psutil
            import time
            self.psutil = psutil
            self.time = time
            return True
        except ImportError:
            return False

    def execute(self, *args, **kwargs) -> Any:
        return self.analyze(*args, **kwargs)

    def analyze(self, activations: Any, **kwargs) -> Dict[str, Any]:
        """Monitor performance metrics."""
        try:
            process = self.psutil.Process()

            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_info': process.memory_info()._asdict(),
                'num_threads': process.num_threads(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            return {}

# Global plugin manager instance
plugin_manager = PluginManager()

def register_plugin(name: str, plugin: PluginBase) -> bool:
    """Register a new plugin."""
    return plugin_manager.register_plugin(name, plugin)

def execute_plugin(name: str, *args, **kwargs) -> Any:
    """Execute a plugin."""
    return plugin_manager.execute_plugin(name, *args, **kwargs)

def get_available_plugins() -> Dict[str, PluginMetadata]:
    """Get all available plugins."""
    return plugin_manager.list_plugins()
