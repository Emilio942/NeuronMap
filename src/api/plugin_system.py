"""
NeuronMap Plugin System

Extensible plugin architecture for custom analysis methods,
visualization types, and integrations.
"""

import abc
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by NeuronMap."""
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    MODEL_LOADER = "model_loader"
    DATA_PROCESSOR = "data_processor"
    EXPORTER = "exporter"
    INTEGRATION = "integration"

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    requirements: List[str]
    config_schema: Optional[Dict[str, Any]] = None
    supported_models: Optional[List[str]] = None
    api_version: str = "1.0"


class BasePlugin(abc.ABC):
    """
    Abstract base class for all NeuronMap plugins.

    All plugins must inherit from this class and implement the required methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.

        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self.metadata = self.get_metadata()
        self.is_initialized = False

        logger.info(f"Initializing plugin: {self.metadata.name} v{self.metadata.version}")

    @abc.abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def cleanup(self):
        """Clean up plugin resources."""
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.metadata.config_schema:
            return True

        # Basic schema validation (could use jsonschema for more advanced validation)
        required_fields = self.metadata.config_schema.get('required', [])
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required config field: {field}")
                return False

        return True

    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this plugin provides."""
        return []


class AnalysisPlugin(BasePlugin):
    """Base class for analysis plugins."""

    @abc.abstractmethod
    def analyze(
        self,
        model: Any,
        input_data: Any,
        layers: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform analysis on model activations.

        Args:
            model: The model to analyze
            input_data: Input data for analysis
            layers: Specific layers to analyze
            **kwargs: Additional analysis parameters

        Returns:
            Analysis results dictionary
        """
        pass

    def get_supported_models(self) -> List[str]:
        """Return list of supported model types."""
        return self.metadata.supported_models or []

class VisualizationPlugin(BasePlugin):
    """Base class for visualization plugins."""

    @abc.abstractmethod
    def create_visualization(
        self,
        data: Dict[str, Any],
        plot_type: str,
        config: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create visualization from analysis data.

        Args:
            data: Analysis data to visualize
            plot_type: Type of plot to create
            config: Visualization configuration
            save_path: Path to save visualization

        Returns:
            Path to created visualization
        """
        pass

    def get_supported_plot_types(self) -> List[str]:
        """Return list of supported plot types."""
        return []


class ModelLoaderPlugin(BasePlugin):
    """Base class for model loader plugins."""

    @abc.abstractmethod
    def load_model(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Load a model from the specified path.

        Args:
            model_path: Path to the model
            config: Model loading configuration

        Returns:
            Loaded model object
        """
        pass

    @abc.abstractmethod
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get information about a loaded model.

        Args:
            model: Loaded model object

        Returns:
            Model information dictionary
        """
        pass


class DataProcessorPlugin(BasePlugin):
    """Base class for data processing plugins."""

    @abc.abstractmethod
    def process_data(
        self,
        data: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Process input data.

        Args:
            data: Input data to process
            config: Processing configuration

        Returns:
            Processed data
        """
        pass


class ExporterPlugin(BasePlugin):
    """Base class for export plugins."""

    @abc.abstractmethod
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Export analysis results to specified format.

        Args:
            results: Analysis results to export
            output_path: Output file path
            format_type: Export format type
            config: Export configuration

        Returns:
            True if export successful, False otherwise
        """
        pass

    def get_supported_formats(self) -> List[str]:
        """Return list of supported export formats."""
        return []


class IntegrationPlugin(BasePlugin):
    """Base class for integration plugins."""

    @abc.abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """
        Connect to external service or system.

        Args:
            config: Connection configuration

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def sync_data(self, data: Dict[str, Any]) -> bool:
        """
        Sync data with external system.

        Args:
            data: Data to sync

        Returns:
            True if sync successful, False otherwise
        """
        pass


class PluginManager:
    """
    Manages loading, initialization, and execution of plugins.
    """

    def __init__(self, plugin_directories: Optional[List[str]] = None):
        """
        Initialize the plugin manager.

        Args:
            plugin_directories: Directories to search for plugins
        """
        self.plugin_directories = plugin_directories or [
            "./plugins",
            "~/.neuronmap/plugins",
            "/etc/neuronmap/plugins"
        ]

        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_registry: Dict[PluginType, Dict[str, BasePlugin]] = {
            plugin_type: {} for plugin_type in PluginType
        }

        logger.info("Plugin manager initialized")


    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directories.

        Returns:
            List of discovered plugin paths
        """
        discovered = []

        for directory in self.plugin_directories:
            plugin_dir = Path(directory).expanduser()
            if not plugin_dir.exists():
                continue

            # Look for Python files
            for plugin_file in plugin_dir.glob("*.py"):
                if not plugin_file.name.startswith("__"):
                    discovered.append(str(plugin_file))

            # Look for plugin packages
            for plugin_package in plugin_dir.iterdir():
                if plugin_package.is_dir() and (plugin_package / "__init__.py").exists():
                    discovered.append(str(plugin_package))

        logger.info(f"Discovered {len(discovered)} potential plugins")
        return discovered

    def load_plugin(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin from the specified path.

        Args:
            plugin_path: Path to the plugin file or package
            config: Plugin configuration

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            plugin_path = Path(plugin_path)

            # Add plugin directory to Python path
            if plugin_path.is_file():
                sys.path.insert(0, str(plugin_path.parent))
                module_name = plugin_path.stem
            else:
                sys.path.insert(0, str(plugin_path.parent))
                module_name = plugin_path.name

            # Import the plugin module
            module = importlib.import_module(module_name)

            # Find plugin classes in the module
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, BasePlugin) and
                    obj is not BasePlugin):
                    plugin_classes.append(obj)

            if not plugin_classes:
                logger.warning(f"No plugin classes found in {plugin_path}")
                return False

            # Initialize plugins
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class(config)

                    if plugin_instance.validate_config(config or {}):
                        if plugin_instance.initialize():
                            # Register the plugin
                            plugin_name = plugin_instance.metadata.name
                            plugin_type = plugin_instance.metadata.plugin_type

                            self.plugins[plugin_name] = plugin_instance
                            self.plugin_registry[plugin_type][plugin_name] = plugin_instance

                            logger.info(f"Successfully loaded plugin: {plugin_name}")
                        else:
                            logger.error(f"Failed to initialize plugin: {plugin_class.__name__}")
                    else:
                        logger.error(f"Invalid configuration for plugin: {plugin_class.__name__}")

                except Exception as e:
                    logger.error(f"Error initializing plugin {plugin_class.__name__}: {e}")
                    logger.error(traceback.format_exc())

            return True

        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_path}: {e}")
            logger.error(traceback.format_exc())
            return False

    def load_all_plugins(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        Load all discovered plugins.

        Args:
            config: Global plugin configuration

        Returns:
            Number of successfully loaded plugins
        """
        plugin_paths = self.discover_plugins()
        loaded_count = 0

        for plugin_path in plugin_paths:
            if self.load_plugin(plugin_path, config):
                loaded_count += 1

        logger.info(f"Loaded {loaded_count} plugins out of {len(plugin_paths)} discovered")
        return loaded_count

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, BasePlugin]:
        """Get all plugins of a specific type."""
        return self.plugin_registry[plugin_type].copy()

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins with their metadata."""
        plugin_list = []

        for plugin_name, plugin in self.plugins.items():
            plugin_info = {
                'name': plugin.metadata.name,
                'version': plugin.metadata.version,
                'description': plugin.metadata.description,
                'author': plugin.metadata.author,
                'type': plugin.metadata.plugin_type.value,
                'capabilities': plugin.get_capabilities(),
                'initialized': plugin.is_initialized
            }
            plugin_list.append(plugin_info)

        return plugin_list

    def execute_analysis_plugin(
        self,
        plugin_name: str,
        model: Any,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute an analysis plugin."""
        plugin = self.get_plugin(plugin_name)

        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")

        if not isinstance(plugin, AnalysisPlugin):
            raise ValueError(f"Plugin {plugin_name} is not an analysis plugin")

        return plugin.analyze(model, input_data, **kwargs)

    def execute_visualization_plugin(
        self,
        plugin_name: str,
        data: Dict[str, Any],
        plot_type: str,
        **kwargs
    ) -> str:
        """Execute a visualization plugin."""
        plugin = self.get_plugin(plugin_name)

        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")

        if not isinstance(plugin, VisualizationPlugin):
            raise ValueError(f"Plugin {plugin_name} is not a visualization plugin")

        return plugin.create_visualization(data, plot_type, **kwargs)


    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            return False

        plugin = self.plugins[plugin_name]

        try:
            plugin.cleanup()

            # Remove from registries
            del self.plugins[plugin_name]
            plugin_type = plugin.metadata.plugin_type
            if plugin_name in self.plugin_registry[plugin_type]:
                del self.plugin_registry[plugin_type][plugin_name]

            logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def cleanup_all_plugins(self):
        """Clean up all loaded plugins."""
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)

        logger.info("All plugins cleaned up")

# Example plugin implementations

class ExampleSentimentAnalysisPlugin(AnalysisPlugin):
    """Example sentiment analysis plugin."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_sentiment",
            version="1.0.0",
            description="Example sentiment analysis plugin",
            author="NeuronMap Team",
            plugin_type=PluginType.ANALYSIS,
            requirements=["torch", "transformers"],
            supported_models=["bert-*", "roberta-*"],
            config_schema={
                "required": ["sentiment_threshold"],
                "properties": {
                    "sentiment_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        )

    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            # Check if required packages are available
            import torch
            import transformers
            self.is_initialized = True
            return True
        except ImportError as e:
            logger.error(f"Missing required dependency: {e}")
            return False

    def analyze(
        self,
        model: Any,
        input_data: Any,
        layers: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform sentiment analysis."""
        # Placeholder implementation
        sentiment_scores = {
            "positive": 0.8,
            "negative": 0.1,
            "neutral": 0.1
        }

        return {
            "sentiment_scores": sentiment_scores,
            "predicted_sentiment": "positive",
            "confidence": 0.8,
            "analysis_method": "example_sentiment_plugin"
        }

    def cleanup(self):
        """Clean up resources."""
        pass

    def get_capabilities(self) -> List[str]:
        return ["sentiment_classification", "confidence_scoring"]


class ExampleVisualizationPlugin(VisualizationPlugin):
    """Example visualization plugin."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_viz",
            version="1.0.0",
            description="Example visualization plugin",
            author="NeuronMap Team",
            plugin_type=PluginType.VISUALIZATION,
            requirements=["matplotlib", "seaborn"]
        )

    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            import matplotlib
            import seaborn
            self.is_initialized = True
            return True
        except ImportError as e:
            logger.error(f"Missing required dependency: {e}")
            return False

    def create_visualization(
        self,
        data: Dict[str, Any],
        plot_type: str,
        config: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create visualization."""
        import matplotlib.pyplot as plt

        # Create simple plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Example Visualization")

        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            plt.show()
            return "displayed"

    def cleanup(self):
        """Clean up resources."""
        pass

    def get_supported_plot_types(self) -> List[str]:
        return ["example_plot"]

# Plugin discovery and registration helper
def register_plugin_class(plugin_class: Type[BasePlugin]):
    """Decorator to register a plugin class."""
    # This could be used to automatically register plugins
    # when they're imported
    pass

# Example usage
if __name__ == "__main__":
    # Initialize plugin manager
    manager = PluginManager()

    # Load all plugins
    loaded = manager.load_all_plugins()
    print(f"Loaded {loaded} plugins")

    # List plugins
    plugins = manager.list_plugins()
    for plugin in plugins:
        print(f"Plugin: {plugin['name']} ({plugin['type']}) - {plugin['description']}")

    # Example plugin execution
    try:
        result = manager.execute_analysis_plugin(
            "example_sentiment",
            model=None,  # Placeholder
            input_data="This is a great example!"
        )
        print(f"Analysis result: {result}")
    except Exception as e:
        print(f"Plugin execution failed: {e}")

    # Cleanup
    manager.cleanup_all_plugins()
