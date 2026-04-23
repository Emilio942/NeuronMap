import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from pathlib import Path

class ToposVisualizer:
    """
    Visualizes Topos-theoretic Heyting Algebras and non-Boolean logic states.
    Displays the 'vielleicht' (maybe) truth values as points in a lattice.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("outputs/visualizations/topos")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_heyting_lattice(self, activations: torch.Tensor, labels: List[str], title: str = "Topos-Logic Lattice (Heyting Algebra)") -> go.Figure:
        """
        Plots a 2D lattice representing the truth values of neural features.
        Top (1.0) is Absolute Truth, Bottom (0.0) is Absolute Falsehood.
        Points in between represent intuitionistic 'maybe' states.
        """
        # Normalize activations to [0, 1] for truth value mapping
        v = (activations - activations.min()) / (activations.max() - activations.min() + 1e-8)
        v = v.detach().cpu().numpy()
        
        # We create a diamond-shaped lattice (Distributive Lattice)
        # x-axis represents semantic ambiguity, y-axis represents truth value
        x = np.sin(np.pi * (v - 0.5)) * (1.0 - np.abs(v - 0.5) * 2.0)
        y = v # The vertical axis is the truth value [0, 1]
        
        fig = go.Figure()
        
        # Background Lattice Structure (Edges of the diamond)
        fig.add_trace(go.Scatter(
            x=[0, 0.5, 0, -0.5, 0],
            y=[1, 0.5, 0, 0.5, 1],
            mode='lines',
            line=dict(color='rgba(200, 200, 200, 0.5)', width=2, dash='dash'),
            name='Lattice Boundary',
            showlegend=False
        ))
        
        # Feature States
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers+text',
            marker=dict(
                size=12,
                color=y,
                colorscale='Viridis',
                colorbar=dict(title="Truth Value (nu)"),
                showscale=True,
                line=dict(width=1, color='black')
            ),
            text=labels,
            textposition="top center",
            name='Neural Features'
        ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(title="Semantic Ambiguity (eta)", range=[-0.6, 0.6], showgrid=False, zeroline=False),
            yaxis=dict(title="Truth Value (nu)", range=[-0.1, 1.1], showgrid=False, zeroline=False),
            template="plotly_white",
            width=800,
            height=600,
            annotations=[
                dict(x=0, y=1.05, text="TRUE (1)", showarrow=False, font=dict(size=14, color="green")),
                dict(x=0, y=-0.05, text="FALSE (0)", showarrow=False, font=dict(size=14, color="red")),
                dict(x=0.4, y=0.5, text="MAYBE (Middle)", showarrow=False, font=dict(size=12, color="blue"))
            ]
        )
        
        return fig

    def save_visualization(self, fig: go.Figure, filename: str):
        """Saves the interactive visualization as an HTML file."""
        path = self.output_dir / f"{filename}.html"
        fig.write_html(str(path))
        print(f"Topos visualization saved to {path}")

if __name__ == "__main__":
    # Demo code
    visualizer = ToposVisualizer()
    mock_activations = torch.tensor([0.95, 0.05, 0.5, 0.7, 0.3])
    mock_labels = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    
    fig = visualizer.plot_heyting_lattice(mock_activations, mock_labels)
    # fig.show() # Uncomment for local display
    visualizer.save_visualization(fig, "demo_topos_lattice")
