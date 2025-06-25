"""Dashboard management system for NeuronMap visualization."""

import logging
import json
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard."""
    title: str = "NeuronMap Dashboard"
    port: int = 8050
    host: str = "127.0.0.1"
    debug: bool = False
    theme: str = "plotly_white"
    auto_refresh: bool = True
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000


@dataclass
class VisualizationPanel:
    """Represents a visualization panel in the dashboard."""
    id: str
    title: str
    type: str  # 'line', 'bar', 'heatmap', 'scatter', '3d', 'network'
    data_source: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {'row': 1, 'col': 1})
    size: Dict[str, int] = field(default_factory=lambda: {'width': 6, 'height': 4})


class DashboardManager:
    """
    Manages interactive dashboards for neural network analysis visualization.

    This class provides:
    - Real-time data visualization
    - Interactive analysis dashboards
    - Multiple visualization backends (Dash, Streamlit)
    - Customizable layouts and themes
    - Live data updates and monitoring
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize DashboardManager.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.panels: Dict[str, VisualizationPanel] = {}
        self.data_sources: Dict[str, Callable] = {}
        self.app = None
        self.server_thread = None
        self.is_running = False
        self._lock = threading.Lock()

        # Check available backends
        self.available_backends = []
        if DASH_AVAILABLE:
            self.available_backends.append('dash')
        if STREAMLIT_AVAILABLE:
            self.available_backends.append('streamlit')

        if not self.available_backends:
            logger.warning("No dashboard backends available. Install dash or streamlit.")

        logger.info(f"DashboardManager initialized with backends: {self.available_backends}")

    def add_panel(self, panel: VisualizationPanel):
        """Add a visualization panel to the dashboard."""
        with self._lock:
            self.panels[panel.id] = panel
            logger.debug(f"Added panel: {panel.id} ({panel.type})")

    def remove_panel(self, panel_id: str):
        """Remove a visualization panel."""
        with self._lock:
            if panel_id in self.panels:
                del self.panels[panel_id]
                logger.debug(f"Removed panel: {panel_id}")

    def add_data_source(self, name: str, data_function: Callable[[], Dict[str, Any]]):
        """
        Add a data source for visualization.

        Args:
            name: Name of the data source
            data_function: Function that returns data when called
        """
        self.data_sources[name] = data_function
        logger.debug(f"Added data source: {name}")

    def create_activation_panel(self,
                               model_name: str,
                               layer_names: List[str],
                               position: Optional[Dict[str, int]] = None) -> str:
        """
        Create a panel for activation visualization.

        Args:
            model_name: Name of the model
            layer_names: List of layer names to visualize
            position: Panel position in grid

        Returns:
            Panel ID
        """
        panel_id = f"activation_{model_name}_{uuid.uuid4().hex[:8]}"

        panel = VisualizationPanel(
            id=panel_id,
            title=f"Activations - {model_name}",
            type="heatmap",
            data_source="activations",
            config={
                "model_name": model_name,
                "layer_names": layer_names,
                "colorscale": "viridis"
            },
            position=position or {'row': 1, 'col': 1}
        )

        self.add_panel(panel)
        return panel_id

    def create_attention_panel(self,
                              model_name: str,
                              head_names: List[str],
                              position: Optional[Dict[str, int]] = None) -> str:
        """
        Create a panel for attention visualization.

        Args:
            model_name: Name of the model
            head_names: List of attention head names
            position: Panel position in grid

        Returns:
            Panel ID
        """
        panel_id = f"attention_{model_name}_{uuid.uuid4().hex[:8]}"

        panel = VisualizationPanel(
            id=panel_id,
            title=f"Attention - {model_name}",
            type="heatmap",
            data_source="attention",
            config={
                "model_name": model_name,
                "head_names": head_names,
                "colorscale": "blues"
            },
            position=position or {'row': 1, 'col': 2}
        )

        self.add_panel(panel)
        return panel_id

    def create_metrics_panel(self,
                           metric_names: List[str],
                           position: Optional[Dict[str, int]] = None) -> str:
        """
        Create a panel for metrics visualization.

        Args:
            metric_names: List of metric names to plot
            position: Panel position in grid

        Returns:
            Panel ID
        """
        panel_id = f"metrics_{uuid.uuid4().hex[:8]}"

        panel = VisualizationPanel(
            id=panel_id,
            title="Performance Metrics",
            type="line",
            data_source="metrics",
            config={
                "metric_names": metric_names,
                "x_axis": "timestamp",
                "y_axis": "value"
            },
            position=position or {'row': 2, 'col': 1}
        )

        self.add_panel(panel)
        return panel_id

    def create_network_panel(self,
                           network_data_source: str,
                           position: Optional[Dict[str, int]] = None) -> str:
        """
        Create a panel for network topology visualization.

        Args:
            network_data_source: Data source for network data
            position: Panel position in grid

        Returns:
            Panel ID
        """
        panel_id = f"network_{uuid.uuid4().hex[:8]}"

        panel = VisualizationPanel(
            id=panel_id,
            title="Network Topology",
            type="network",
            data_source=network_data_source,
            config={
                "layout_algorithm": "force_directed",
                "node_size_metric": "activation_magnitude",
                "edge_width_metric": "connection_strength"
            },
            position=position or {'row': 2, 'col': 2}
        )

        self.add_panel(panel)
        return panel_id

    def start_dash_server(self, external_stylesheets: Optional[List[str]] = None):
        """Start Dash server for dashboard."""
        if not DASH_AVAILABLE:
            raise RuntimeError("Dash is not available. Install with: pip install dash")

        if self.is_running:
            logger.warning("Dashboard server is already running")
            return

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets or []
        )

        # Set up layout
        self._setup_dash_layout()

        # Set up callbacks
        self._setup_dash_callbacks()

        # Start server in thread
        def run_server():
            self.app.run_server(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug,
                use_reloader=False
            )

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True

        logger.info(f"Dash server started at http://{self.config.host}:{self.config.port}")

    def _setup_dash_layout(self):
        """Set up Dash application layout."""
        # Header
        header = html.Div([
            html.H1(self.config.title, className="dashboard-title"),
            html.Div([
                html.Button("Refresh", id="refresh-button", className="btn btn-primary"),
                html.Div(id="last-update", className="last-update")
            ], className="dashboard-controls")
        ], className="dashboard-header")

        # Create panel grid
        panel_grid = self._create_panel_grid()

        # Footer
        footer = html.Div([
            html.P(f"NeuronMap Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ], className="dashboard-footer")

        self.app.layout = html.Div([
            header,
            panel_grid,
            footer,
            dcc.Interval(
                id='interval-component',
                interval=self.config.refresh_interval * 1000,  # in milliseconds
                n_intervals=0,
                disabled=not self.config.auto_refresh
            )
        ], className="dashboard-container")

    def _create_panel_grid(self):
        """Create grid layout for visualization panels."""
        if not self.panels:
            return html.Div([
                html.H3("No panels configured"),
                html.P("Add visualization panels to see data")
            ], className="empty-dashboard")

        # Group panels by row
        rows = {}
        for panel in self.panels.values():
            row = panel.position.get('row', 1)
            if row not in rows:
                rows[row] = []
            rows[row].append(panel)

        # Create row components
        row_components = []
        for row_num in sorted(rows.keys()):
            panel_components = []
            for panel in sorted(rows[row_num], key=lambda p: p.position.get('col', 1)):
                panel_component = html.Div([
                    html.H4(panel.title, className="panel-title"),
                    dcc.Graph(id=f"graph-{panel.id}")
                ], className=f"panel col-{panel.size['width']}")
                panel_components.append(panel_component)

            row_component = html.Div(
                panel_components,
                className="dashboard-row row"
            )
            row_components.append(row_component)

        return html.Div(row_components, className="dashboard-grid")

    def _setup_dash_callbacks(self):
        """Set up Dash callbacks for interactivity."""
        # Get all graph IDs
        graph_ids = [f"graph-{panel.id}" for panel in self.panels.values()]

        if not graph_ids:
            return

        # Create callback for updating graphs
        @self.app.callback(
            [Output(graph_id, 'figure') for graph_id in graph_ids] + [Output('last-update', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_graphs(n_intervals, n_clicks):
            # Update all graphs
            figures = []
            for panel in self.panels.values():
                try:
                    fig = self._create_panel_figure(panel)
                    figures.append(fig)
                except Exception as e:
                    logger.error(f"Error updating panel {panel.id}: {e}")
                    # Create error figure
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                    figures.append(fig)

            # Update timestamp
            timestamp = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

            return figures + [timestamp]

    def _create_panel_figure(self, panel: VisualizationPanel):
        """Create Plotly figure for a panel."""
        # Get data from source
        if panel.data_source not in self.data_sources:
            raise ValueError(f"Data source '{panel.data_source}' not found")

        data = self.data_sources[panel.data_source]()

        # Create figure based on panel type
        if panel.type == "line":
            return self._create_line_figure(panel, data)
        elif panel.type == "bar":
            return self._create_bar_figure(panel, data)
        elif panel.type == "heatmap":
            return self._create_heatmap_figure(panel, data)
        elif panel.type == "scatter":
            return self._create_scatter_figure(panel, data)
        elif panel.type == "3d":
            return self._create_3d_figure(panel, data)
        elif panel.type == "network":
            return self._create_network_figure(panel, data)
        else:
            raise ValueError(f"Unknown panel type: {panel.type}")

    def _create_line_figure(self, panel: VisualizationPanel, data: Dict[str, Any]):
        """Create line plot figure."""
        fig = go.Figure()

        x_data = data.get(panel.config.get('x_axis', 'x'), [])

        for metric in panel.config.get('metric_names', ['y']):
            if metric in data:
                y_data = data[metric]
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    name=metric
                ))

        fig.update_layout(
            title=panel.title,
            xaxis_title=panel.config.get('x_axis', 'X'),
            yaxis_title=panel.config.get('y_axis', 'Y'),
            template=self.config.theme
        )

        return fig

    def _create_heatmap_figure(self, panel: VisualizationPanel, data: Dict[str, Any]):
        """Create heatmap figure."""
        # Extract matrix data
        if 'matrix' in data:
            matrix = np.array(data['matrix'])
        elif 'values' in data:
            matrix = np.array(data['values'])
        else:
            # Try to find any 2D array in data
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    arr = np.array(value)
                    if arr.ndim == 2:
                        matrix = arr
                        break
            else:
                # Create dummy data
                matrix = np.random.rand(10, 10)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale=panel.config.get('colorscale', 'viridis'),
            x=data.get('x_labels', None),
            y=data.get('y_labels', None)
        ))

        fig.update_layout(
            title=panel.title,
            template=self.config.theme
        )

        return fig

    def _create_bar_figure(self, panel: VisualizationPanel, data: Dict[str, Any]):
        """Create bar plot figure."""
        fig = go.Figure()

        x_data = data.get('categories', data.get('x', []))
        y_data = data.get('values', data.get('y', []))

        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data,
            name=panel.title
        ))

        fig.update_layout(
            title=panel.title,
            template=self.config.theme
        )

        return fig

    def _create_scatter_figure(self, panel: VisualizationPanel, data: Dict[str, Any]):
        """Create scatter plot figure."""
        fig = go.Figure()

        x_data = data.get('x', [])
        y_data = data.get('y', [])

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name=panel.title
        ))

        fig.update_layout(
            title=panel.title,
            template=self.config.theme
        )

        return fig

    def _create_3d_figure(self, panel: VisualizationPanel, data: Dict[str, Any]):
        """Create 3D plot figure."""
        fig = go.Figure()

        x_data = data.get('x', [])
        y_data = data.get('y', [])
        z_data = data.get('z', [])

        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            name=panel.title
        ))

        fig.update_layout(
            title=panel.title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            template=self.config.theme
        )

        return fig

    def _create_network_figure(self, panel: VisualizationPanel, data: Dict[str, Any]):
        """Create network visualization figure."""
        # Extract network data
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        if not nodes:
            # Create empty network
            fig = go.Figure()
            fig.add_annotation(
                text="No network data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig

        # Extract node positions (or generate them)
        if 'positions' in data:
            positions = data['positions']
        else:
            # Generate random positions
            positions = {
                node['id']: (np.random.rand(), np.random.rand())
                for node in nodes
            }

        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = positions[edge['source']]
            x1, y1 = positions[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node trace
        node_x = [positions[node['id']][0] for node in nodes]
        node_y = [positions[node['id']][1] for node in nodes]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=[node.get('label', node['id']) for node in nodes],
            marker=dict(
                size=[node.get('size', 10) for node in nodes],
                color=[node.get('color', 'blue') for node in nodes],
                line=dict(width=2)
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=panel.title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Network visualization",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=self.config.theme
        )

        return fig

    def stop_server(self):
        """Stop the dashboard server."""
        if self.is_running:
            self.is_running = False
            logger.info("Dashboard server stopped")

    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.config.host}:{self.config.port}"

    def export_dashboard_config(self, filepath: Union[str, Path]) -> None:
        """Export dashboard configuration to file."""
        config_data = {
            'config': {
                'title': self.config.title,
                'port': self.config.port,
                'host': self.config.host,
                'theme': self.config.theme,
                'auto_refresh': self.config.auto_refresh,
                'refresh_interval': self.config.refresh_interval
            },
            'panels': [
                {
                    'id': panel.id,
                    'title': panel.title,
                    'type': panel.type,
                    'data_source': panel.data_source,
                    'config': panel.config,
                    'position': panel.position,
                    'size': panel.size
                }
                for panel in self.panels.values()
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Dashboard config exported to {filepath}")


# Convenience functions
def create_dashboard_manager(config: Optional[DashboardConfig] = None) -> DashboardManager:
    """Create a DashboardManager instance."""
    return DashboardManager(config)


def create_default_neuron_dashboard(data_sources: Dict[str, Callable]) -> DashboardManager:
    """Create a default dashboard for neural network analysis."""
    config = DashboardConfig(
        title="NeuronMap Analysis Dashboard",
        auto_refresh=True,
        refresh_interval=5
    )

    dashboard = DashboardManager(config)

    # Add data sources
    for name, func in data_sources.items():
        dashboard.add_data_source(name, func)

    # Create default panels
    if 'activations' in data_sources:
        dashboard.create_activation_panel("model", ["layer1", "layer2"], {'row': 1, 'col': 1})

    if 'attention' in data_sources:
        dashboard.create_attention_panel("model", ["head1", "head2"], {'row': 1, 'col': 2})

    if 'metrics' in data_sources:
        dashboard.create_metrics_panel(["accuracy", "loss"], {'row': 2, 'col': 1})

    return dashboard
