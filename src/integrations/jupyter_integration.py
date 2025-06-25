"""
Jupyter Integration for NeuronMap

Provides enhanced Jupyter notebook support with interactive widgets and visualizations.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import base64
from io import BytesIO

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, Image, Javascript
    import jupyter_bokeh
    JUPYTER_AVAILABLE = True
except ImportError:
    widgets = None
    display = None
    HTML = None
    Image = None
    Javascript = None
    JUPYTER_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

# Internal imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.error_handling import NeuronMapError
from utils.monitoring import setup_monitoring
from core.neuron_map import NeuronMap

logger = logging.getLogger(__name__)

class JupyterIntegration:
    """
    Enhanced Jupyter notebook integration for NeuronMap.

    Provides:
    - Interactive widgets for analysis configuration
    - Real-time visualization updates
    - Progress bars and status indicators
    - Export capabilities
    - Integration with existing notebook workflows
    """

    def __init__(self, neuron_map: Optional[NeuronMap] = None):
        """
        Initialize Jupyter integration.

        Args:
            neuron_map: NeuronMap instance to use
        """
        if not JUPYTER_AVAILABLE:
            raise ImportError("Jupyter integration requires ipywidgets and IPython")

        self.neuron_map = neuron_map or NeuronMap()
        self.monitor = setup_monitoring()

        # Widget storage
        self.widgets = {}
        self.current_analysis = None
        self.analysis_results = {}

        logger.info("Jupyter integration initialized")

    def create_analysis_widget(self) -> widgets.Widget:
        """
        Create interactive widget for configuring and running analysis.

        Returns:
            Complete analysis widget
        """
        try:
            # Model selection
            model_dropdown = widgets.Dropdown(
                options=[
                    'bert-base-uncased',
                    'gpt2',
                    'roberta-base',
                    'distilbert-base-uncased'
                ],
                value='bert-base-uncased',
                description='Model:',
                style={'description_width': 'initial'}
            )

            # Text input
            text_input = widgets.Textarea(
                value='Enter your text here for analysis...',
                placeholder='Type your text here',
                description='Input Text:',
                layout=widgets.Layout(width='100%', height='100px'),
                style={'description_width': 'initial'}
            )

            # Analysis type
            analysis_type = widgets.Dropdown(
                options=[
                    ('Basic Analysis', 'basic'),
                    ('Sentiment Analysis', 'sentiment'),
                    ('Interpretability Analysis', 'interpretability'),
                    ('Attention Analysis', 'attention')
                ],
                value='basic',
                description='Analysis Type:',
                style={'description_width': 'initial'}
            )

            # Layer selection
            layer_selection = widgets.SelectMultiple(
                options=[('Layer 0', 0), ('Layer 3', 3), ('Layer 6', 6), ('Layer 9', 9), ('Layer 11', 11)],
                value=[0, 6, 11],
                description='Layers:',
                style={'description_width': 'initial'}
            )

            # Advanced options
            include_attention = widgets.Checkbox(
                value=True,
                description='Include Attention Analysis',
                style={'description_width': 'initial'}
            )

            include_gradients = widgets.Checkbox(
                value=False,
                description='Include Gradient Analysis',
                style={'description_width': 'initial'}
            )

            # Progress bar
            progress_bar = widgets.IntProgress(
                value=0,
                min=0,
                max=100,
                description='Progress:',
                bar_style='',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='100%')
            )

            # Status output
            status_output = widgets.Output()

            # Results output
            results_output = widgets.Output()

            # Run button
            run_button = widgets.Button(
                description='Run Analysis',
                button_style='primary',
                tooltip='Start the analysis',
                icon='play'
            )

            # Export button
            export_button = widgets.Button(
                description='Export Results',
                button_style='success',
                tooltip='Export analysis results',
                icon='download',
                disabled=True
            )

            # Store widgets for later access
            self.widgets.update({
                'model_dropdown': model_dropdown,
                'text_input': text_input,
                'analysis_type': analysis_type,
                'layer_selection': layer_selection,
                'include_attention': include_attention,
                'include_gradients': include_gradients,
                'progress_bar': progress_bar,
                'status_output': status_output,
                'results_output': results_output,
                'run_button': run_button,
                'export_button': export_button
            })

            # Define button click handlers
            def on_run_clicked(b):
                self._run_analysis_interactive()

            def on_export_clicked(b):
                self._export_results()

            run_button.on_click(on_run_clicked)
            export_button.on_click(on_export_clicked)

            # Layout the widget
            config_section = widgets.VBox([
                widgets.HTML('<h3>Analysis Configuration</h3>'),
                model_dropdown,
                text_input,
                analysis_type,
                layer_selection,
                widgets.HBox([include_attention, include_gradients]),
                widgets.HBox([run_button, export_button])
            ])

            progress_section = widgets.VBox([
                widgets.HTML('<h3>Progress</h3>'),
                progress_bar,
                status_output
            ])

            results_section = widgets.VBox([
                widgets.HTML('<h3>Results</h3>'),
                results_output
            ])

            main_widget = widgets.VBox([
                config_section,
                progress_section,
                results_section
            ])

            return main_widget

        except Exception as e:
            logger.error(f"Failed to create analysis widget: {e}")
            return widgets.HTML(f"<p style='color: red;'>Error creating widget: {e}</p>")

    def _run_analysis_interactive(self):
        """Run analysis with interactive updates."""
        try:
            # Clear previous outputs
            self.widgets['status_output'].clear_output()
            self.widgets['results_output'].clear_output()

            with self.widgets['status_output']:
                print("Starting analysis...")

            # Update progress
            self._update_progress(10, "Loading model...")

            # Get configuration from widgets
            config = {
                'model_name': self.widgets['model_dropdown'].value,
                'text': self.widgets['text_input'].value,
                'analysis_type': self.widgets['analysis_type'].value,
                'layers': list(self.widgets['layer_selection'].value),
                'include_attention': self.widgets['include_attention'].value,
                'include_gradients': self.widgets['include_gradients'].value
            }

            # Load model
            model_config = {'name': config['model_name'], 'type': 'huggingface'}
            model = self.neuron_map.load_model(model_config)

            self._update_progress(30, "Running analysis...")

            # Run analysis based on type
            if config['analysis_type'] == 'sentiment':
                results = self.neuron_map.analyze_sentiment(
                    model=model,
                    texts=[config['text']],
                    layers=config['layers']
                )
            elif config['analysis_type'] == 'interpretability':
                results = self.neuron_map.analyze_interpretability(
                    model=model,
                    input_text=config['text'],
                    layers=config['layers']
                )
            else:  # basic analysis
                results = self.neuron_map.generate_activations(
                    text=config['text'],
                    layers=config['layers'],
                    include_attention=config['include_attention']
                )

            self._update_progress(80, "Creating visualizations...")

            # Store results
            self.analysis_results = results
            self.current_analysis = config

            # Create visualizations
            self._display_results(results, config)

            self._update_progress(100, "Analysis complete!")

            # Enable export button
            self.widgets['export_button'].disabled = False

            with self.widgets['status_output']:
                print("✅ Analysis completed successfully!")

        except Exception as e:
            self._update_progress(0, f"Error: {str(e)}")
            with self.widgets['status_output']:
                print(f"❌ Analysis failed: {e}")
            logger.error(f"Interactive analysis failed: {e}")

    def _update_progress(self, value: int, status: str):
        """Update progress bar and status."""
        self.widgets['progress_bar'].value = value
        self.widgets['progress_bar'].description = f"Progress: {status}"

    def _display_results(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Display analysis results in the notebook."""
        try:
            with self.widgets['results_output']:
                # Display summary
                print(f"Analysis Type: {config['analysis_type']}")
                print(f"Model: {config['model_name']}")
                print(f"Layers Analyzed: {config['layers']}")
                print("-" * 50)

                # Create interactive visualizations
                if 'activations' in results:
                    self._create_activation_visualization(results['activations'])

                if 'attention_weights' in results:
                    self._create_attention_visualization(results['attention_weights'])

                if 'layer_statistics' in results:
                    self._create_statistics_table(results['layer_statistics'])

                # Display metrics if available
                if 'metrics' in results:
                    self._display_metrics(results['metrics'])

        except Exception as e:
            with self.widgets['results_output']:
                print(f"Error displaying results: {e}")
            logger.error(f"Failed to display results: {e}")

    def _create_activation_visualization(self, activations: Dict[str, Any]):
        """Create interactive activation visualization."""
        try:
            if not PLOTLY_AVAILABLE:
                print("Plotly not available for interactive visualizations")
                return

            # Create subplot for each layer
            layers = list(activations.keys())
            num_layers = len(layers)

            fig = make_subplots(
                rows=1, cols=num_layers,
                subplot_titles=[f"Layer {layer}" for layer in layers],
                horizontal_spacing=0.1
            )

            for i, (layer_idx, activation) in enumerate(activations.items()):
                # Convert to numpy if needed
                if hasattr(activation, 'detach'):
                    activation = activation.detach().cpu().numpy()

                # Take a sample for visualization
                if len(activation.shape) > 1:
                    sample = activation[:50, :50]  # First 50x50
                else:
                    sample = activation[:100].reshape(-1, 1)  # First 100 as column

                # Create heatmap
                heatmap = go.Heatmap(
                    z=sample,
                    colorscale='Viridis',
                    showscale=(i == num_layers - 1)  # Show scale only on last plot
                )

                fig.add_trace(heatmap, row=1, col=i+1)

            fig.update_layout(
                title="Layer Activations",
                height=400,
                showlegend=False
            )

            # Display the figure
            display(fig)

        except Exception as e:
            print(f"Could not create activation visualization: {e}")

    def _create_attention_visualization(self, attention_weights: Dict[str, Any]):
        """Create interactive attention visualization."""
        try:
            if not PLOTLY_AVAILABLE:
                print("Plotly not available for interactive visualizations")
                return

            # Create attention pattern for first layer with attention
            layer_idx = list(attention_weights.keys())[0]
            attention = attention_weights[layer_idx]

            if hasattr(attention, 'detach'):
                attention = attention.detach().cpu().numpy()

            # Average across heads if multi-head
            if len(attention.shape) > 2:
                attention = attention.mean(axis=0)

            # Create interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=attention,
                colorscale='Blues',
                hoverongaps=False
            ))

            fig.update_layout(
                title=f"Attention Pattern - Layer {layer_idx}",
                xaxis_title="Token Position",
                yaxis_title="Token Position",
                height=500,
                width=500
            )

            display(fig)

        except Exception as e:
            print(f"Could not create attention visualization: {e}")

    def _create_statistics_table(self, layer_stats: Dict[str, Any]):
        """Create statistics table."""
        try:
            import pandas as pd

            # Convert statistics to DataFrame
            stats_data = []
            for layer, stats in layer_stats.items():
                row = {'Layer': layer}
                row.update(stats)
                stats_data.append(row)

            df = pd.DataFrame(stats_data)

            # Display as HTML table
            html_table = df.to_html(classes='table table-striped', table_id='stats-table')
            display(HTML(html_table))

        except Exception as e:
            print(f"Could not create statistics table: {e}")

    def _display_metrics(self, metrics: Dict[str, Any]):
        """Display metrics in a formatted way."""
        print("\n📊 Analysis Metrics:")
        print("-" * 30)

        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric_name}: {value:.4f}")
            else:
                print(f"{metric_name}: {value}")

    def _export_results(self):
        """Export analysis results."""
        try:
            if not self.analysis_results:
                print("No results to export!")
                return

            # Create export options widget
            export_format = widgets.Dropdown(
                options=[
                    ('JSON', 'json'),
                    ('CSV (statistics only)', 'csv'),
                    ('HTML Report', 'html')
                ],
                value='json',
                description='Format:'
            )

            export_button = widgets.Button(
                description='Download',
                button_style='success'
            )

            output_area = widgets.Output()

            def do_export(b):
                with output_area:
                    output_area.clear_output()

                    if export_format.value == 'json':
                        self._export_json()
                    elif export_format.value == 'csv':
                        self._export_csv()
                    elif export_format.value == 'html':
                        self._export_html()

            export_button.on_click(do_export)

            export_widget = widgets.VBox([
                widgets.HTML('<h4>Export Results</h4>'),
                export_format,
                export_button,
                output_area
            ])

            display(export_widget)

        except Exception as e:
            print(f"Export failed: {e}")

    def _export_json(self):
        """Export results as JSON."""
        try:
            from datetime import datetime

            # Prepare data for export
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.current_analysis,
                'results': self._make_json_serializable(self.analysis_results)
            }

            # Create download link
            json_str = json.dumps(export_data, indent=2)
            self._create_download_link(json_str, 'neuronmap_results.json', 'application/json')

        except Exception as e:
            print(f"JSON export failed: {e}")

    def _export_csv(self):
        """Export statistics as CSV."""
        try:
            import pandas as pd
            from io import StringIO

            if 'layer_statistics' not in self.analysis_results:
                print("No statistics available for CSV export")
                return

            # Convert to DataFrame
            stats_data = []
            for layer, stats in self.analysis_results['layer_statistics'].items():
                row = {'layer': layer}
                row.update(stats)
                stats_data.append(row)

            df = pd.DataFrame(stats_data)

            # Convert to CSV
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()

            self._create_download_link(csv_str, 'neuronmap_statistics.csv', 'text/csv')

        except Exception as e:
            print(f"CSV export failed: {e}")

    def _export_html(self):
        """Export as HTML report."""
        try:
            from datetime import datetime

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>NeuronMap Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>NeuronMap Analysis Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="section">
                    <h2>Configuration</h2>
                    <p><strong>Model:</strong> {self.current_analysis.get('model_name', 'Unknown')}</p>
                    <p><strong>Analysis Type:</strong> {self.current_analysis.get('analysis_type', 'Unknown')}</p>
                    <p><strong>Layers:</strong> {', '.join(map(str, self.current_analysis.get('layers', [])))}</p>
                </div>

                <div class="section">
                    <h2>Results Summary</h2>
                    <p>Analysis completed successfully with detailed layer-wise activations and statistics.</p>
                </div>
            </body>
            </html>
            """

            self._create_download_link(html_content, 'neuronmap_report.html', 'text/html')

        except Exception as e:
            print(f"HTML export failed: {e}")

    def _create_download_link(self, content: str, filename: str, mime_type: str):
        """Create a download link for content."""
        try:
            # Encode content
            b64_content = base64.b64encode(content.encode()).decode()

            # Create download link
            download_link = f"""
            <a download="{filename}"
               href="data:{mime_type};base64,{b64_content}"
               style="background-color: #4CAF50; color: white; padding: 10px 20px;
                      text-decoration: none; border-radius: 4px;">
                Download {filename}
            </a>
            """

            display(HTML(download_link))
            print(f"✅ {filename} ready for download!")

        except Exception as e:
            print(f"Failed to create download link: {e}")

    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if hasattr(obj, 'detach'):  # PyTorch tensor
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
            except ImportError:
                pass
            return obj

    def create_quick_analysis_widget(self) -> widgets.Widget:
        """Create a simplified widget for quick analysis."""
        text_input = widgets.Text(
            value="This is a test sentence.",
            description="Text:",
            style={'description_width': 'initial'}
        )

        analyze_button = widgets.Button(
            description="Quick Analysis",
            button_style="primary"
        )

        output = widgets.Output()

        def quick_analyze(b):
            with output:
                output.clear_output()
                try:
                    print("Running quick analysis...")
                    # Simple analysis
                    results = self.neuron_map.generate_activations(
                        text=text_input.value,
                        layers=[0, 6, 11]
                    )
                    print("✅ Analysis complete!")
                    print(f"Analyzed text: '{text_input.value}'")
                    print(f"Layers processed: {list(results.get('activations', {}).keys())}")
                except Exception as e:
                    print(f"❌ Error: {e}")

        analyze_button.on_click(quick_analyze)

        return widgets.VBox([
            widgets.HTML("<h3>Quick Analysis</h3>"),
            text_input,
            analyze_button,
            output
        ])

# Utility functions for Jupyter integration
def create_neuronmap_widget(neuron_map: Optional[NeuronMap] = None) -> widgets.Widget:
    """
    Create a complete NeuronMap analysis widget.

    Args:
        neuron_map: NeuronMap instance to use

    Returns:
        Interactive widget for Jupyter notebooks
    """
    if not JUPYTER_AVAILABLE:
        return HTML("<p style='color: red;'>Jupyter integration not available. Install ipywidgets.</p>")

    integration = JupyterIntegration(neuron_map)
    return integration.create_analysis_widget()

def quick_widget(neuron_map: Optional[NeuronMap] = None) -> widgets.Widget:
    """
    Create a simplified analysis widget.

    Args:
        neuron_map: NeuronMap instance to use

    Returns:
        Simple analysis widget
    """
    if not JUPYTER_AVAILABLE:
        return HTML("<p style='color: red;'>Jupyter integration not available.</p>")

    integration = JupyterIntegration(neuron_map)
    return integration.create_quick_analysis_widget()

def display_results_interactive(results: Dict[str, Any]):
    """
    Display analysis results with interactive visualizations.

    Args:
        results: NeuronMap analysis results
    """
    if not JUPYTER_AVAILABLE:
        print("Interactive display not available")
        return

    integration = JupyterIntegration()
    config = {'analysis_type': 'interactive_display'}
    integration._display_results(results, config)

# Auto-setup for Jupyter notebooks
def setup_jupyter():
    """Setup Jupyter environment for NeuronMap."""
    if JUPYTER_AVAILABLE:
        # Load custom CSS for better styling
        css = """
        <style>
        .neuronmap-widget {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #fafafa;
        }
        .neuronmap-header {
            color: #2E7D32;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        """
        display(HTML(css))

        # Display welcome message
        welcome_html = """
        <div class="neuronmap-widget">
            <div class="neuronmap-header">🧠 NeuronMap Jupyter Integration Loaded</div>
            <p>Use <code>create_neuronmap_widget()</code> to start interactive analysis.</p>
        </div>
        """
        display(HTML(welcome_html))

# Example usage
if __name__ == "__main__":
    # This would be used in a Jupyter notebook
    if JUPYTER_AVAILABLE:
        print("Creating example widget...")
        widget = create_neuronmap_widget()
        display(widget)
    else:
        print("Jupyter integration not available")
