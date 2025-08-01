#!/usr/bin/env python3
"""
PyTorch Neuron Group Visualization Demo
=====================================

Demonstrates the PyTorch neuron group visualization capabilities 
with a simple neural network and synthetic data.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run PyTorch neuron group visualization demo."""
    print("🚀 PyTorch Neuron Group Visualization Demo gestartet...")
    
    # Check PyTorch availability
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        print(f"✅ PyTorch {torch.__version__} verfügbar auf {torch.cuda.is_available() and 'CUDA' or 'CPU'}")
    except ImportError:
        print("❌ PyTorch nicht verfügbar. Installation mit: pip install torch")
        return False
    
    # Import our PyTorch visualizer
    try:
        from src.visualization.pytorch_neuron_group_visualizer import (
            PyTorchNeuronGroupVisualizer,
            create_pytorch_neuron_group_analysis
        )
        print("✅ PyTorch Neuron Group Visualizer geladen")
    except ImportError as e:
        print(f"❌ Fehler beim Laden des Visualizers: {e}")
        return False
    
    # Create simple neural network for demo
    class DemoNet(nn.Module):
        def __init__(self, input_size=20, hidden_size=50, output_size=5):
            super(DemoNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    print("\n📊 Erstelle Demo Neural Network...")
    
    # Create model and sample data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DemoNet()
    model.to(device)
    
    # Generate synthetic training data
    n_samples = 200
    input_size = 20
    
    # Create data with some structure (different skill types)
    X_math = torch.randn(n_samples//4, input_size) + torch.tensor([1.0, 0.5] + [0.0]*18)
    X_language = torch.randn(n_samples//4, input_size) + torch.tensor([0.0, 0.0, 1.0, 0.5] + [0.0]*16)
    X_logic = torch.randn(n_samples//4, input_size) + torch.tensor([0.0]*5 + [1.0, 0.8] + [0.0]*13)
    X_memory = torch.randn(n_samples//4, input_size) + torch.tensor([0.0]*10 + [1.0, 0.6] + [0.0]*8)
    
    X = torch.cat([X_math, X_language, X_logic, X_memory], dim=0)
    
    # Create corresponding labels and metadata
    y_math = torch.zeros(n_samples//4, 5); y_math[:, 0] = 1
    y_language = torch.zeros(n_samples//4, 5); y_language[:, 1] = 1
    y_logic = torch.zeros(n_samples//4, 5); y_logic[:, 2] = 1
    y_memory = torch.zeros(n_samples//4, 5); y_memory[:, 3] = 1
    
    y = torch.cat([y_math, y_language, y_logic, y_memory], dim=0)
    
    # Create metadata DataFrame
    categories = ['mathematical'] * (n_samples//4) + ['linguistic'] * (n_samples//4) + \
                ['logical'] * (n_samples//4) + ['memory'] * (n_samples//4)
    
    questions = [f"Math problem {i}" for i in range(n_samples//4)] + \
               [f"Language task {i}" for i in range(n_samples//4)] + \
               [f"Logic puzzle {i}" for i in range(n_samples//4)] + \
               [f"Memory recall {i}" for i in range(n_samples//4)]
    
    metadata = pd.DataFrame({
        'category': categories,
        'question': questions,
        'difficulty': np.random.choice(['easy', 'medium', 'hard'], n_samples)
    })
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Demo-Daten erstellt: {n_samples} Samples, {input_size} Features")
    print(f"📝 Kategorien: {set(categories)}")
    
    # Specify layers to analyze
    layer_names = ['fc1', 'fc2', 'fc3']
    
    print(f"\n🔍 Analysiere Layers: {layer_names}")
    
    # Run complete PyTorch neuron group analysis
    try:
        results = create_pytorch_neuron_group_analysis(
            model=model,
            dataloader=dataloader,
            layer_names=layer_names,
            question_metadata=metadata,
            output_dir="demo_outputs/pytorch_neuron_groups",
            device=device,
            max_batches=10,  # Limit for demo
            model_epoch=1,
            model_info={
                'model_type': 'DemoNet',
                'parameters': sum(p.numel() for p in model.parameters()),
                'layers': len(list(model.named_modules())),
                'input_size': input_size,
                'hidden_size': 50,
                'output_size': 5
            }
        )
        
        if results and results.get('summary', {}).get('analysis_complete'):
            print("\n✅ PyTorch Neuron Group Analysis erfolgreich abgeschlossen!")
            
            # Print summary
            summary = results['summary']
            print(f"📊 Ergebnisse:")
            print(f"   🎯 Layers analysiert: {summary['total_layers']}")
            print(f"   🧩 Neuron Groups gefunden: {summary['total_groups']}")
            print(f"   📚 Learning Events identifiziert: {summary['total_learning_events']}")
            print(f"   💻 Device verwendet: {summary['device_used']}")
            
            # Print layer details
            print(f"\n🔍 Layer Details:")
            for layer_name, groups in results['neuron_groups'].items():
                if groups:
                    avg_cohesion = np.mean([g.cohesion_score for g in groups])
                    total_neurons = sum([g.group_size for g in groups])
                    print(f"   {layer_name}: {len(groups)} Groups, {total_neurons} Neurons, "
                          f"Ø Cohesion: {avg_cohesion:.3f}")
                else:
                    print(f"   {layer_name}: Keine Groups gefunden")
            
            # Print learning patterns
            if results['learning_events']:
                skill_counts = {}
                for event in results['learning_events']:
                    skill_counts[event.skill_type] = skill_counts.get(event.skill_type, 0) + 1
                
                print(f"\n📚 Learning Events nach Skill Type:")
                for skill, count in skill_counts.items():
                    print(f"   {skill}: {count} Events")
            
            # List generated files
            print(f"\n📁 Generierte Dateien:")
            output_dir = Path("demo_outputs/pytorch_neuron_groups")
            if output_dir.exists():
                for file_path in output_dir.iterdir():
                    if file_path.is_file():
                        size_kb = file_path.stat().st_size / 1024
                        print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
            
            # Show visualization paths
            if 'visualizations' in results:
                print(f"\n🎨 Visualisierungen:")
                for vis_name, vis_path in results['visualizations'].items():
                    if vis_path and Path(vis_path).exists():
                        print(f"   🖼️  {vis_name}: {vis_path}")
            
            if 'dashboard' in results and results['dashboard']:
                dashboard_path = results['dashboard']
                if Path(dashboard_path).exists():
                    print(f"   📊 Interactive Dashboard: {dashboard_path}")
            
            if 'report' in results and results['report']:
                report_path = results['report']
                if Path(report_path).exists():
                    print(f"   📋 Analysis Report: {report_path}")
            
            print(f"\n🎉 Demo erfolgreich abgeschlossen!")
            return True
            
        else:
            print("❌ PyTorch Neuron Group Analysis fehlgeschlagen")
            return False
            
    except Exception as e:
        print(f"❌ Fehler während der Analysis: {e}")
        logger.exception("Detailed error:")
        return False

def test_pytorch_basic_functionality():
    """Test basic PyTorch functionality."""
    print("\n🧪 Teste PyTorch Basis-Funktionalität...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test tensor operations
        x = torch.randn(5, 10)
        y = torch.randn(10, 3)
        z = torch.mm(x, y)
        print(f"✅ Tensor Operations: {x.shape} @ {y.shape} = {z.shape}")
        
        # Test neural network layer
        layer = nn.Linear(10, 5)
        output = layer(x)
        print(f"✅ Neural Network Layer: {x.shape} -> {output.shape}")
        
        # Test device availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_device = x.to(device)
        print(f"✅ Device Support: {device} - {x_device.device}")
        
        # Test correlations
        corr_matrix = torch.corrcoef(x.T)
        print(f"✅ Correlation Matrix: {corr_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch Test fehlgeschlagen: {e}")
        return False

def test_visualizer_import():
    """Test importing the PyTorch visualizer."""
    print("\n📦 Teste PyTorch Visualizer Import...")
    
    try:
        from src.visualization.pytorch_neuron_group_visualizer import (
            PyTorchNeuronGroupVisualizer,
            PyTorchNeuronGroup,
            PyTorchLearningEvent,
            create_pytorch_neuron_group_analysis
        )
        print("✅ PyTorchNeuronGroupVisualizer importiert")
        print("✅ PyTorchNeuronGroup importiert")
        print("✅ PyTorchLearningEvent importiert")
        print("✅ create_pytorch_neuron_group_analysis importiert")
        
        # Test initialization
        visualizer = PyTorchNeuronGroupVisualizer(output_dir="test_output")
        print(f"✅ Visualizer initialisiert mit device: {visualizer.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import Test fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔥 PyTorch Neuron Group Visualization Demo")
    print("=" * 60)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_pytorch_basic_functionality():
        tests_passed += 1
    
    if test_visualizer_import():
        tests_passed += 1
    
    if main():
        tests_passed += 1
    
    print(f"\n{'='*60}")
    print(f"🎯 Demo Abschluss: {tests_passed}/{total_tests} Tests erfolgreich")
    
    if tests_passed == total_tests:
        print("🎉 Alle Tests erfolgreich! PyTorch Neuron Group Visualization ist einsatzbereit.")
    else:
        print("⚠️  Einige Tests fehlgeschlagen. Bitte überprüfen Sie die Installation.")
    
    print("="*60)
