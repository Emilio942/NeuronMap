"""
PSG Visualizer
==============

Visualizes Parameter Sparsity Gates (PSGs) and their reactions.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. PSG visualizations will be disabled.")

from analysis.psg_detector import PSGNode

class PSGVisualizer:
    """Creates visualizations for detected Parameter Sparsity Gates."""

    def __init__(self, output_dir: str = "outputs/psg_viz"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_structure(self, psgs: List[PSGNode], filename: str = "psg_structure.html"):
        """
        Visualizes the distribution and properties of detected PSGs.
        """
        if not PLOTLY_AVAILABLE or not psgs:
            return

        # Prepare data
        layers = [p.layer for p in psgs]
        indices = [p.index for p in psgs]
        sparsities = [p.activation_sparsity for p in psgs]
        strengths = [p.gating_strength for p in psgs]
        ids = [p.id for p in psgs]
        
        # Clean data for plotting
        sizes = [max(0.0, float(s)) * 10 for s in strengths]
        sizes = [0 if np.isnan(s) else s for s in sizes]

        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=layers,
            y=indices,
            mode='markers',
            marker=dict(
                size=sizes,  # Size by gating strength
                color=sparsities,  # Color by sparsity
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Activation Sparsity")
            ),
            text=[f"ID: {i}<br>Sparsity: {s:.2f}<br>Strength: {g:.2f}" 
                  for i, s, g in zip(ids, sparsities, strengths)],
            hoverinfo='text'
        ))

        fig.update_layout(
            title="Parameter Sparsity Gates (PSG) Structure",
            xaxis_title="Layer Index",
            yaxis_title="Neuron Index",
            template="plotly_dark",
            height=None,  # Allow responsive height
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"Saved PSG structure visualization to {output_path}")

    def visualize_reaction(self, psgs: List[PSGNode], input_text: str, filename: str = "psg_reaction.html"):
        """
        Visualizes how PSGs react (activate) to a specific input.
        Note: This assumes psgs have 'reaction_profile' populated or we pass new activations.
        For this simple version, we visualize the 'max_activation' as a proxy for reaction intensity.
        """
        if not PLOTLY_AVAILABLE or not psgs:
            return

        # Prepare data
        layers = [p.layer for p in psgs]
        indices = [p.index for p in psgs]
        activations = [p.max_activation for p in psgs] # Using max activation as "reaction"
        ids = [p.id for p in psgs]
        
        # Clean data for plotting
        sizes = [np.log1p(max(0.0, float(a))) * 15 for a in activations]
        sizes = [0 if np.isnan(s) else s for s in sizes]

        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=layers,
            y=indices,
            mode='markers',
            marker=dict(
                size=sizes,  # Size by activation intensity
                color=activations,
                colorscale='Magma',
                showscale=True,
                colorbar=dict(title="Activation Intensity")
            ),
            text=[f"ID: {i}<br>Activation: {a:.4f}" for i, a in zip(ids, activations)],
            hoverinfo='text'
        ))

        fig.update_layout(
            title=f"PSG Reaction to Input: '{input_text[:50]}...'",
            xaxis_title="Layer Index",
            yaxis_title="Neuron Index",
            template="plotly_dark",
            height=None,  # Allow responsive height
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"Saved PSG reaction visualization to {output_path}")
