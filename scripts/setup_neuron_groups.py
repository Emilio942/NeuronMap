#!/usr/bin/env python3
"""
Setup Script for Neuron Group Visualization
==========================================

This script sets up the environment for neuron group visualization by
installing required dependencies and configuring the system.
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies for neuron group visualization."""
    dependencies = [
        # Core dependencies
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        
        # Visualization dependencies
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        
        # Analysis dependencies  
        'scikit-learn>=1.0.0',
        'networkx>=2.5',
        
        # Optional interactive dependencies
        'jupyter>=1.0.0',
        'ipywidgets>=7.6.0'
    ]
    
    logger.info("Installing dependencies for neuron group visualization...")
    
    for dep in dependencies:
        try:
            logger.info(f"Installing {dep}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', dep],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"✓ Successfully installed {dep}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠ Failed to install {dep}: {e}")
            logger.warning(f"  stdout: {e.stdout}")
            logger.warning(f"  stderr: {e.stderr}")

def test_imports():
    """Test if all required modules can be imported."""
    test_modules = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('plotly.graph_objects', 'go'),
        ('sklearn.cluster', None),
        ('networkx', 'nx')
    ]
    
    logger.info("Testing module imports...")
    
    failed_imports = []
    
    for module, alias in test_modules:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            logger.info(f"✓ {module} imported successfully")
        except ImportError as e:
            logger.error(f"✗ Failed to import {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        logger.info("All modules imported successfully!")
        return True

def create_demo_data():
    """Create demo data directory and sample files."""
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # Create sample activation data
    try:
        import numpy as np
        import pandas as pd
        
        # Generate sample activation matrix
        np.random.seed(42)
        activation_matrix = np.random.random((50, 30))
        
        # Save as CSV for easy loading
        df = pd.DataFrame(activation_matrix)
        df.to_csv(demo_dir / "sample_activations.csv", index=False)
        
        # Create sample question metadata
        questions_df = pd.DataFrame({
            'question_id': range(50),
            'question': [f'Sample question {i}' for i in range(50)],
            'category': np.random.choice(['math', 'language', 'logic'], 50)
        })
        questions_df.to_csv(demo_dir / "sample_questions.csv", index=False)
        
        logger.info(f"✓ Demo data created in {demo_dir}")
        return True
        
    except ImportError:
        logger.warning("Cannot create demo data - NumPy/Pandas not available")
        return False

def setup_output_directories():
    """Create necessary output directories."""
    directories = [
        "data/outputs/neuron_groups",
        "data/outputs/visualizations", 
        "data/outputs/reports",
        "demo_outputs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {dir_path}")

def run_basic_test():
    """Run a basic test of the neuron group system."""
    try:
        # Test basic import
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from visualization.neuron_group_visualizer import NeuronGroupVisualizer
        import numpy as np
        
        # Create test data
        test_activation_matrix = np.random.random((20, 15))
        
        # Initialize visualizer
        visualizer = NeuronGroupVisualizer(output_dir="test_output")
        
        # Test group identification
        groups = visualizer.identify_neuron_groups(
            test_activation_matrix,
            method='correlation_clustering',
            correlation_threshold=0.5,
            min_group_size=2
        )
        
        logger.info(f"✓ Basic test passed - found {len(groups)} neuron groups")
        
        # Clean up test output
        import shutil
        if Path("test_output").exists():
            shutil.rmtree("test_output")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("="*60)
    logger.info("NeuronMap - Neuron Group Visualization Setup")
    logger.info("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Test imports
    if not test_imports():
        logger.error("Some dependencies failed to install. Please install them manually:")
        logger.error("pip install numpy pandas matplotlib seaborn plotly scikit-learn networkx")
        sys.exit(1)
    
    # Setup directories
    setup_output_directories()
    
    # Create demo data
    create_demo_data()
    
    # Run basic test
    if run_basic_test():
        logger.info("✓ Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run the demo: python scripts/demo_neuron_groups.py")
        logger.info("2. Check the documentation: docs/neuron_group_visualization.md")
        logger.info("3. Integrate with your NeuronMap workflow")
    else:
        logger.warning("Setup completed with warnings. Basic test failed.")
        logger.warning("Please check the error messages above.")

if __name__ == "__main__":
    main()
