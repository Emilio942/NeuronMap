"""
NeuronMap API Module

Provides REST API, Python API, and plugin architecture for NeuronMap.
Enables programmatic access to all NeuronMap functionality.
"""

from .rest_api import NeuronMapAPI, create_app
from .python_api import NeuronMapClient, AsyncNeuronMapClient
from .plugin_system import PluginManager, BasePlugin

__all__ = [
    'NeuronMapAPI',
    'create_app',
    'NeuronMapClient',
    'AsyncNeuronMapClient',
    'PluginManager',
    'BasePlugin'
]

__version__ = '1.0.0'
