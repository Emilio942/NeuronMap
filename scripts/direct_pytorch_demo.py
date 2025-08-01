#!/usr/bin/env python3
"""
Direct PyTorch Neuron Group Demo
===============================

Direkte Implementierung der PyTorch Neuron Group Analyse
ohne komplexe Plugin-Abhängigkeiten.
"""

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
    """Run direct PyTorch neuron group analysis demo."""
    print("🚀 Direct PyTorch Neuron Group Demo gestartet...")
    
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
    
    # Direct import without plugins
    try:
        # Import the visualizer directly without going through the plugin system
        from src.visualization.pytorch_neuron_group_visualizer import PyTorchNeuronGroupVisualizer
        print("✅ PyTorch Neuron Group Visualizer direkt geladen")
    except ImportError as e:
        print(f"❌ Fehler beim direkten Laden: {e}")
        return False
    
    # Create simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 3)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    print("\n📊 Erstelle einfaches Neural Network...")
    
    # Create model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet()
    model.to(device)
    
    # Generate structured test data
    n_samples = 100
    input_size = 10
    
    # Create data with patterns
    X = torch.randn(n_samples, input_size)
    
    # Add structure: first half of features correlate with first output
    X[:n_samples//3, :5] += 1.5  # Pattern for output 1
    X[n_samples//3:2*n_samples//3, 5:] += 1.5  # Pattern for output 2
    X[2*n_samples//3:, :3] += 2.0  # Pattern for output 3
    
    # Create dataset and dataloader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
    
    print(f"✅ Testdaten erstellt: {n_samples} Samples, {input_size} Features")
    
    # Initialize visualizer
    visualizer = PyTorchNeuronGroupVisualizer(
        output_dir="demo_outputs/direct_pytorch",
        device=device
    )
    
    print(f"✅ Visualizer initialisiert mit device: {visualizer.device}")
    
    # Extract activations
    layer_names = ['fc1', 'fc2', 'fc3']
    print(f"\n🔍 Extrahiere Aktivierungen von Layers: {layer_names}")
    
    activations_dict = visualizer.extract_activations_from_model(
        model=model,
        dataloader=dataloader,
        layer_names=layer_names,
        max_batches=5
    )
    
    if not activations_dict:
        print("❌ Keine Aktivierungen extrahiert")
        return False
    
    print(f"✅ Aktivierungen extrahiert für {len(activations_dict)} Layers")
    
    # Identify neuron groups
    neuron_groups_dict = {}
    total_groups = 0
    
    for layer_name, activation_tensor in activations_dict.items():
        print(f"\n🧩 Identifiziere Neuron Groups in {layer_name}...")
        
        groups = visualizer.identify_neuron_groups_pytorch(
            activation_tensor=activation_tensor,
            layer_name=layer_name,
            method='correlation_clustering',
            correlation_threshold=0.4,
            min_group_size=2
        )
        
        neuron_groups_dict[layer_name] = groups
        total_groups += len(groups)
        
        print(f"   ✅ {len(groups)} Groups gefunden")
        
        # Print group details
        for group in groups:
            print(f"      Group {group.group_id}: {group.group_size} neurons, "
                  f"cohesion: {group.cohesion_score:.3f}")
    
    if total_groups == 0:
        print("\n⚠️  Keine Neuron Groups gefunden - adjustiere Parameter")
        return False
    
    # Create metadata
    categories = ['pattern_a'] * (n_samples//3) + ['pattern_b'] * (n_samples//3) + ['pattern_c'] * (n_samples - 2*(n_samples//3))
    metadata = pd.DataFrame({
        'category': categories,
        'pattern_type': categories
    })
    
    # Analyze learning patterns
    print(f"\n📚 Analysiere Learning Patterns...")
    
    learning_events = visualizer.analyze_pytorch_learning_patterns(
        activations_dict=activations_dict,
        neuron_groups_dict=neuron_groups_dict,
        question_metadata=metadata,
        model_epoch=1
    )
    
    print(f"✅ {len(learning_events)} Learning Events identifiziert")
    
    # Create visualizations
    print(f"\n🎨 Erstelle Visualisierungen...")
    
    # Multi-layer visualization
    multi_viz = visualizer.visualize_pytorch_groups(
        activations_dict, neuron_groups_dict, method='heatmap'
    )
    
    if multi_viz:
        print(f"   ✅ Multi-layer Heatmap: {multi_viz}")
    
    # Individual layer scatter plots
    for layer_name in layer_names:
        if layer_name in activations_dict and neuron_groups_dict[layer_name]:
            scatter_viz = visualizer.visualize_pytorch_groups(
                activations_dict, neuron_groups_dict, 
                method='scatter', layer_name=layer_name
            )
            if scatter_viz:
                print(f"   ✅ {layer_name} Scatter: {scatter_viz}")
    
    # Create interactive dashboard
    print(f"\n📊 Erstelle Interactive Dashboard...")
    
    dashboard_path = visualizer.create_pytorch_interactive_dashboard(
        activations_dict, neuron_groups_dict, learning_events, metadata
    )
    
    if dashboard_path:
        print(f"   ✅ Dashboard: {dashboard_path}")
    
    # Generate comprehensive report
    print(f"\n📋 Erstelle Analysis Report...")
    
    report_path = visualizer.generate_pytorch_report(
        activations_dict=activations_dict,
        neuron_groups_dict=neuron_groups_dict,
        learning_events=learning_events,
        model_info={
            'model_type': 'SimpleNet',
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'device': str(device),
            'layers': layer_names
        }
    )
    
    if report_path:
        print(f"   ✅ Report: {report_path}")
    
    # Final summary
    print(f"\n🎯 Demo Zusammenfassung:")
    print(f"   📊 Layers analysiert: {len(activations_dict)}")
    print(f"   🧩 Neuron Groups gefunden: {total_groups}")
    print(f"   📚 Learning Events: {len(learning_events)}")
    print(f"   💻 Device: {device}")
    
    # List all generated files
    output_dir = Path("demo_outputs/direct_pytorch")
    if output_dir.exists():
        print(f"\n📁 Generierte Dateien:")
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
    
    print(f"\n🎉 Direct PyTorch Demo erfolgreich abgeschlossen!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔥 Direct PyTorch Neuron Group Demo")
    print("=" * 60)
    
    success = main()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 Demo erfolgreich! PyTorch Neuron Group Analysis funktioniert.")
    else:
        print("⚠️  Demo unvollständig. Überprüfen Sie die Ausgabe.")
    print("="*60)
