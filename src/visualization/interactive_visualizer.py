"""Interactive visualization helpers with optional Plotly and Dash fallbacks."""

import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Plotly imports with graceful fallbacks
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency handling
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - fallback when Plotly is missing
    go_module = types.ModuleType("plotly.graph_objects")

    class _Trace(dict):
        """Minimal trace representation used by the Plotly stub."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class _Figure:
        def __init__(self):
            self.data: List[Any] = []
            self.layout: Dict[str, Any] = {}
            self.frames: List[Any] = []

        def add_trace(self, trace: Any) -> None:
            self.data.append(trace)

        def update_layout(self, **kwargs) -> None:
            self.layout.update(kwargs)

        def update_traces(self, **kwargs) -> None:  # pragma: no cover - simple stub
            for trace in self.data:
                if isinstance(trace, dict):
                    trace.update(kwargs)

        def write_html(self, file_path: str, **_kwargs) -> None:
            Path(file_path).write_text("<html><body><p>Plotly stub output</p></body></html>", encoding="utf-8")

        def add_frame(self, frame: Any) -> None:
            self.frames.append(frame)

    go_module.Figure = _Figure  # type: ignore[attr-defined]
    go_module.Scatter = _Trace  # type: ignore[attr-defined]
    go_module.Scatter3d = _Trace  # type: ignore[attr-defined]
    go_module.Heatmap = _Trace  # type: ignore[attr-defined]
    go_module.Frame = _Trace  # type: ignore[attr-defined]

    sys.modules.setdefault("plotly", types.ModuleType("plotly"))
    sys.modules["plotly"].graph_objects = go_module  # type: ignore[attr-defined]
    sys.modules["plotly.graph_objects"] = go_module
    go = go_module  # type: ignore

# ---------------------------------------------------------------------------
# Optional Dash imports with graceful fallbacks
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency handling
    import dash  # type: ignore
    from dash import dcc, html  # type: ignore
except Exception:  # pragma: no cover - fallback when Dash is missing
    dash_module = types.ModuleType("dash")

    class _DashApp:
        def __init__(self, name: str, **_kwargs):
            self.name = name
            self.layout: Any = None

        def run_server(self, *args, **kwargs) -> None:  # pragma: no cover - stubbed server
            logger.info("Dash server stub invoked with args=%s kwargs=%s", args, kwargs)

    dash_module.Dash = _DashApp  # type: ignore[attr-defined]

    def _component_factory(component_name: str):
        class _Component(dict):  # pragma: no cover - simple stub component
            def __init__(self, *children, **props):
                super().__init__(props)
                self.children = list(children)
                self.component_name = component_name

        return _Component

    html_module = types.ModuleType("dash.html")
    html_module.Div = _component_factory("Div")
    html_module.H1 = _component_factory("H1")
    html_module.H2 = _component_factory("H2")
    html_module.P = _component_factory("P")
    html_module.Button = _component_factory("Button")

    dcc_module = types.ModuleType("dash.dcc")
    dcc_module.Graph = _component_factory("Graph")

    sys.modules["dash"] = dash_module
    sys.modules["dash.html"] = html_module
    sys.modules["dash.dcc"] = dcc_module

    dash = dash_module  # type: ignore
    html = html_module  # type: ignore
    dcc = dcc_module  # type: ignore

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_config(config_name: str) -> Dict[str, Any]:
    try:
        config = get_config()
        if hasattr(config, "get_experiment_config"):
            experiment = config.get_experiment_config(config_name)
            if isinstance(experiment, dict):
                return experiment
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        logger.debug("Interactive visualization config load failed: %s", exc)
    return {}


class InteractiveVisualizer:
    """Create lightweight interactive visualizations using Plotly."""

    def __init__(self, config_name: str = "default", output_dir: Optional[str] = None):
        self._plotly_available = hasattr(go, "Figure")
        self.config = _extract_config(config_name)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._scaler = StandardScaler()
        self._last_embedding: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _ensure_plotly(self) -> None:
        if not self._plotly_available:
            raise ImportError("plotly is required for interactive visualizations but is not available.")

    def _compute_embedding(self, activations: np.ndarray, n_components: int) -> np.ndarray:
        if activations is None:
            raise ValueError("Activation data cannot be None.")

        matrix = np.asarray(activations)
        if matrix.ndim != 2:
            raise ValueError("Activation data must be a 2D array.")

        matrix = self._scaler.fit_transform(matrix)
        components = max(1, min(n_components, min(matrix.shape)))
        embedding = PCA(n_components=components).fit_transform(matrix)
        self._last_embedding = embedding
        return embedding

    def _prepare_plotly_data(self, activations: np.ndarray,
                              labels: Optional[List[str]] = None,
                              n_components: int = 2) -> Dict[str, Any]:
        embedding = self._compute_embedding(activations, n_components)

        prepared: Dict[str, Any] = {
            "x": embedding[:, 0].tolist(),
            "y": embedding[:, 1].tolist() if embedding.shape[1] > 1 else [0.0] * len(embedding),
            "text": labels or [f"Point {idx}" for idx in range(len(embedding))],
        }

        if embedding.shape[1] > 2:
            prepared["z"] = embedding[:, 2].tolist()

        return prepared

    # ------------------------------------------------------------------
    # Public API used by tests
    # ------------------------------------------------------------------
    def create_interactive_scatter(self, activations: np.ndarray,
                                   labels: Optional[List[str]] = None,
                                   title: str = "Interactive Scatter",
                                   save_path: Optional[str] = None) -> Any:
        self._ensure_plotly()
        data = self._prepare_plotly_data(activations, labels, n_components=2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode="markers",
                                 text=data["text"], hoverinfo="text"))
        fig.update_layout(title=title, template="plotly_white")

        if save_path:
            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)

        return fig

    def create_3d_scatter(self, activations: np.ndarray,
                          labels: Optional[List[str]] = None,
                          title: str = "Interactive 3D Scatter",
                          save_path: Optional[str] = None) -> Any:
        self._ensure_plotly()
        data = self._prepare_plotly_data(activations, labels, n_components=3)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=data["x"], y=data["y"], z=data.get("z", [0.0] * len(data["x"])),
                                   mode="markers", text=data["text"], hoverinfo="text"))
        fig.update_layout(title=title, template="plotly_white")

        if save_path:
            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)

        return fig

    def create_activation_animation(self, activations: Dict[str, np.ndarray],
                                    save_path: Optional[str] = None,
                                    frame_axis: int = 0) -> Any:
        self._ensure_plotly()
        if not activations:
            raise ValueError("activations dictionary cannot be empty")

        first_layer = next(iter(activations.values()))
        data = self._prepare_plotly_data(first_layer, labels=None, n_components=2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode="markers"))
        fig.update_layout(title="Activation Animation", updatemenus=[{
            "type": "buttons",
            "buttons": [{"label": "Play", "method": "animate", "args": [None]}]
        }])

        frames = []
        for layer_name, layer_data in activations.items():
            frame_prepared = self._prepare_plotly_data(layer_data, labels=None, n_components=2)
            frames.append(go.Frame(name=layer_name, data=[go.Scatter(x=frame_prepared["x"],
                                                                     y=frame_prepared["y"],
                                                                     mode="markers")]))

        if hasattr(fig, "frames"):
            fig.frames = frames

        if save_path:
            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)

        return fig

    # Expose helper for tests
    def prepare_plotly_data(self, activations: np.ndarray,
                            labels: Optional[List[str]] = None,
                            n_components: int = 2) -> Dict[str, Any]:
        return self._prepare_plotly_data(activations, labels, n_components)

# ---------------------------------------------------------------------------
# Dashboard manager with optional Dash support
# ---------------------------------------------------------------------------


class DashboardManager:
    """Create and manage a lightweight Dash application."""

    DEFAULT_CONFIG = {
        "title": "NeuronMap Dashboard",
        "port": 8050,
        "debug": False,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._app: Optional[Any] = None

    def _ensure_dash(self) -> None:
        if not hasattr(dash, "Dash"):
            raise ImportError("dash is required for the dashboard but is not available.")

    def _create_layout(self) -> Any:
        self._ensure_dash()
        return html.Div([
            html.H1(self.config.get("title", "NeuronMap Dashboard")),
            html.P("Interactive monitoring for neuron analyses."),
            dcc.Graph(id="placeholder-graph"),
        ])

    def create_dashboard(self) -> Any:
        self._ensure_dash()
        app = dash.Dash(__name__)
        app.layout = self._create_layout()
        self._app = app
        return app

    def run_dashboard(self) -> None:
        if self._app is None:
            self.create_dashboard()
        assert self._app is not None
        self._app.run_server(port=self.config.get("port", 8050),
                             debug=self.config.get("debug", False))
