"""
NeuronMap API Module

Provides REST API, Python API, and plugin architecture for NeuronMap.
Enables programmatic access to all NeuronMap functionality.
"""

from .rest_api import NeuronMapAPI, create_app
from .python_api import NeuronMapClient, AsyncNeuronMapClient
from .plugin_system import PluginManager, BasePlugin
from .websocket_api import WebSocketHandler
from .graphql_api import GraphQLSchema

__all__ = [
    'NeuronMapAPI',
    'create_app',
    'NeuronMapClient',
    'AsyncNeuronMapClient',
    'PluginManager',
    'BasePlugin',
    'WebSocketHandler',
    'GraphQLSchema'
]

__version__ = '1.0.0'
