"""Unit tests for visualization modules."""

import unittest
import pytest
import tempfile
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.visualization.visualizer import ActivationVisualizer
    from src.visualization.interactive_visualizer import InteractiveVisualizer, DashboardManager
except ImportError as e:
    print(f"Warning: Could not import visualization modules: {e}")


class TestActivationVisualizer(unittest.TestCase):
    """Test basic activation visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "plots"
        self.output_dir.mkdir()
        
        # Create mock activation data
        self.sample_data = {
            'questions': ['Question 1', 'Question 2', 'Question 3'],
            'activations': {
                'layer_0': np.random.randn(3, 768),
                'layer_6': np.random.randn(3, 768), 
                'layer_11': np.random.randn(3, 768)
            },
            'metadata': {
                'model_name': 'gpt2',
                'timestamp': '2024-01-01'
            }
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_visualizer_init(self):
        """Test ActivationVisualizer initialization."""
        try:
            visualizer = ActivationVisualizer(output_dir=str(self.output_dir))
            self.assertEqual(str(visualizer.output_dir), str(self.output_dir))
            
        except ImportError:
            self.skipTest("ActivationVisualizer not available")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_activation_heatmap(self, mock_show, mock_savefig):
        """Test activation heatmap creation."""
        try:
            visualizer = ActivationVisualizer(output_dir=str(self.output_dir))
            
            # Create heatmap with sample data
            result = visualizer.create_activation_heatmap(
                self.sample_data['activations']['layer_0'],
                title="Test Heatmap",
                save_path=str(self.output_dir / "test_heatmap.png")
            )
            
            # Verify plot was attempted to be saved
            mock_savefig.assert_called()
            self.assertIsNotNone(result)
            
        except ImportError:
            self.skipTest("ActivationVisualizer not available")
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_pca_plot(self, mock_savefig):
        """Test PCA visualization creation."""
        try:
            visualizer = ActivationVisualizer(output_dir=str(self.output_dir))
            
            # Create PCA plot
            result = visualizer.create_pca_plot(
                self.sample_data['activations']['layer_0'],
                labels=self.sample_data['questions'],
                save_path=str(self.output_dir / "test_pca.png")
            )
            
            mock_savefig.assert_called()
            self.assertIsNotNone(result)
            
        except ImportError:
            self.skipTest("ActivationVisualizer not available")
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_tsne_plot(self, mock_savefig):
        """Test t-SNE visualization creation.""" 
        try:
            visualizer = ActivationVisualizer(output_dir=str(self.output_dir))
            
            # Create t-SNE plot
            result = visualizer.create_tsne_plot(
                self.sample_data['activations']['layer_0'],
                labels=self.sample_data['questions'],
                save_path=str(self.output_dir / "test_tsne.png")
            )
            
            mock_savefig.assert_called()
            self.assertIsNotNone(result)
            
        except ImportError:
            self.skipTest("ActivationVisualizer not available")
    
    def test_compute_pca_components(self):
        """Test PCA computation."""
        try:
            visualizer = ActivationVisualizer(output_dir=str(self.output_dir))
            
            # Compute PCA
            pca_result = visualizer._compute_pca(
                self.sample_data['activations']['layer_0'],
                n_components=2
            )
            
            self.assertEqual(pca_result.shape[1], 2)  # Should have 2 components
            self.assertEqual(pca_result.shape[0], 3)  # Should have 3 samples
            
        except ImportError:
            self.skipTest("ActivationVisualizer not available")
    
    def test_layer_comparison_plot(self):
        """Test layer comparison visualization."""
        try:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                visualizer = ActivationVisualizer(output_dir=str(self.output_dir))
                
                # Create layer comparison plot
                result = visualizer.create_layer_comparison_plot(
                    self.sample_data['activations'],
                    save_path=str(self.output_dir / "layer_comparison.png")
                )
                
                mock_savefig.assert_called()
                self.assertIsNotNone(result)
                
        except ImportError:
            self.skipTest("ActivationVisualizer not available")


class TestInteractiveVisualizer(unittest.TestCase):
    """Test interactive visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "interactive"
        self.output_dir.mkdir()
        
        # Sample data for interactive plots
        self.sample_data = {
            'questions': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            'activations': {
                'layer_0': np.random.randn(5, 100),
                'layer_3': np.random.randn(5, 100),
                'layer_6': np.random.randn(5, 100)
            },
            'metadata': {'model_name': 'test-model'}
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_interactive_visualizer_init(self):
        """Test InteractiveVisualizer initialization."""
        try:
            visualizer = InteractiveVisualizer(output_dir=str(self.output_dir))
            self.assertEqual(str(visualizer.output_dir), str(self.output_dir))
            
        except ImportError:
            self.skipTest("InteractiveVisualizer not available")
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_interactive_scatter(self, mock_write_html):
        """Test interactive scatter plot creation."""
        try:
            visualizer = InteractiveVisualizer(output_dir=str(self.output_dir))
            
            # Create interactive scatter plot
            result = visualizer.create_interactive_scatter(
                self.sample_data['activations']['layer_0'],
                labels=self.sample_data['questions'],
                title="Interactive Scatter Test",
                save_path=str(self.output_dir / "scatter.html")
            )
            
            mock_write_html.assert_called()
            self.assertIsNotNone(result)
            
        except ImportError:
            self.skipTest("InteractiveVisualizer not available")
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_3d_visualization(self, mock_write_html):
        """Test 3D interactive visualization."""
        try:
            visualizer = InteractiveVisualizer(output_dir=str(self.output_dir))
            
            # Create 3D plot
            result = visualizer.create_3d_scatter(
                self.sample_data['activations']['layer_0'],
                labels=self.sample_data['questions'],
                title="3D Scatter Test",
                save_path=str(self.output_dir / "3d_scatter.html")
            )
            
            mock_write_html.assert_called()
            self.assertIsNotNone(result)
            
        except ImportError:
            self.skipTest("InteractiveVisualizer not available")
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_activation_animation(self, mock_write_html):
        """Test activation animation creation."""
        try:
            visualizer = InteractiveVisualizer(output_dir=str(self.output_dir))
            
            # Create animation
            result = visualizer.create_activation_animation(
                self.sample_data['activations'],
                save_path=str(self.output_dir / "animation.html")
            )
            
            mock_write_html.assert_called()
            self.assertIsNotNone(result)
            
        except ImportError:
            self.skipTest("InteractiveVisualizer not available")
    
    def test_prepare_data_for_plotly(self):
        """Test data preparation for Plotly."""
        try:
            visualizer = InteractiveVisualizer(output_dir=str(self.output_dir))
            
            # Test data preparation
            plot_data = visualizer._prepare_plotly_data(
                self.sample_data['activations']['layer_0'],
                labels=self.sample_data['questions']
            )
            
            self.assertIsInstance(plot_data, dict)
            self.assertIn('x', plot_data)
            self.assertIn('y', plot_data)
            self.assertIn('text', plot_data)
            
        except ImportError:
            self.skipTest("InteractiveVisualizer not available")


class TestDashboardManager(unittest.TestCase):
    """Test dashboard management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dashboard_config = {
            'title': 'NeuronMap Dashboard',
            'port': 8050,
            'debug': False
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_dashboard_manager_init(self):
        """Test DashboardManager initialization."""
        try:
            dashboard = DashboardManager(self.dashboard_config)
            self.assertEqual(dashboard.config['title'], 'NeuronMap Dashboard')
            self.assertEqual(dashboard.config['port'], 8050)
            
        except ImportError:
            self.skipTest("DashboardManager not available")
    
    @patch('dash.Dash.run_server')
    def test_dashboard_startup(self, mock_run_server):
        """Test dashboard startup."""
        try:
            dashboard = DashboardManager(self.dashboard_config)
            
            # Test dashboard creation
            app = dashboard.create_dashboard()
            self.assertIsNotNone(app)
            
            # Test dashboard startup (mocked)
            dashboard.run_dashboard()
            mock_run_server.assert_called()
            
        except ImportError:
            self.skipTest("DashboardManager not available")
    
    def test_dashboard_layout_creation(self):
        """Test dashboard layout generation."""
        try:
            dashboard = DashboardManager(self.dashboard_config)
            
            # Create layout
            layout = dashboard._create_layout()
            
            # Layout should be a valid Dash component
            self.assertIsNotNone(layout)
            
        except ImportError:
            self.skipTest("DashboardManager not available")


class TestVisualizationUtils(unittest.TestCase):
    """Test visualization utility functions."""
    
    def test_color_palette_generation(self):
        """Test color palette generation for plots."""
        try:
            from src.visualization.visualizer import generate_color_palette
            
            # Test palette generation
            colors = generate_color_palette(5)
            self.assertEqual(len(colors), 5)
            
            # Colors should be valid hex codes or color names
            for color in colors:
                self.assertIsInstance(color, str)
                
        except ImportError:
            self.skipTest("Color palette utilities not available")
    
    def test_plot_configuration(self):
        """Test plot configuration utilities."""
        try:
            from src.visualization.visualizer import get_default_plot_config
            
            config = get_default_plot_config()
            
            self.assertIsInstance(config, dict)
            self.assertIn('figure_size', config)
            self.assertIn('dpi', config)
            
        except ImportError:
            self.skipTest("Plot configuration utilities not available")


class MockPlottingTests(unittest.TestCase):
    """Test plotting with all dependencies mocked."""
    
    def test_matplotlib_import_error(self):
        """Test graceful handling of matplotlib import errors."""
        try:
            with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
                # Should handle missing matplotlib gracefully
                from src.visualization.visualizer import ActivationVisualizer
                
                visualizer = ActivationVisualizer(output_dir="/tmp")
                # Visualizer should initialize even without matplotlib
                self.assertIsNotNone(visualizer)
                
        except ImportError:
            self.skipTest("ActivationVisualizer not available")
    
    def test_plotly_import_error(self):
        """Test graceful handling of plotly import errors."""
        try:
            with patch.dict('sys.modules', {'plotly': None}):
                # Should handle missing plotly gracefully
                from src.visualization.interactive_visualizer import InteractiveVisualizer
                
                visualizer = InteractiveVisualizer(output_dir="/tmp")
                # Should initialize with limited functionality
                self.assertIsNotNone(visualizer)
                
        except ImportError:
            self.skipTest("InteractiveVisualizer not available")


class IntegrationVisualizationTests(unittest.TestCase):
    """Integration tests for visualization pipeline."""
    
    def setUp(self):
        """Set up integration test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data_file = Path(self.temp_dir.name) / "test_activations.json"
        
        # Create realistic test data
        test_data = {
            'questions': [f"Question {i}" for i in range(10)],
            'activations': {
                f'layer_{i}': np.random.randn(10, 768).tolist() for i in range(12)
            },
            'metadata': {
                'model_name': 'gpt2',
                'layers_extracted': list(range(12)),
                'timestamp': '2024-01-01'
            }
        }
        
        with open(self.test_data_file, 'w') as f:
            import json
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up integration test data."""
        self.temp_dir.cleanup()
    
    def test_full_visualization_pipeline(self):
        """Test complete visualization pipeline."""
        try:
            # Load test data
            with open(self.test_data_file, 'r') as f:
                import json
                data = json.load(f)
            
            # Convert activations back to numpy arrays
            for layer in data['activations']:
                data['activations'][layer] = np.array(data['activations'][layer])
            
            # Create visualizations
            output_dir = Path(self.temp_dir.name) / "viz_output"
            output_dir.mkdir()
            
            with patch('matplotlib.pyplot.savefig'):
                with patch('plotly.graph_objects.Figure.write_html'):
                    # Test basic visualizer
                    basic_viz = ActivationVisualizer(output_dir=str(output_dir))
                    basic_result = basic_viz.create_activation_heatmap(
                        data['activations']['layer_0'],
                        save_path=str(output_dir / "heatmap.png")
                    )
                    
                    # Test interactive visualizer
                    interactive_viz = InteractiveVisualizer(output_dir=str(output_dir))
                    interactive_result = interactive_viz.create_interactive_scatter(
                        data['activations']['layer_0'],
                        labels=data['questions'],
                        save_path=str(output_dir / "scatter.html")
                    )
                    
                    self.assertIsNotNone(basic_result)
                    self.assertIsNotNone(interactive_result)
            
        except ImportError:
            self.skipTest("Visualization modules not available")


if __name__ == "__main__":
    unittest.main(verbosity=2)