#!/usr/bin/env python3
"""
Standalone PyTorch Neuron Group Demo
===================================

Standalone Implementation ohne src Package Dependencies.
"""

import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

@dataclass
class StandaloneNeuronGroup:
    """Standalone neuron group representation."""
    group_id: int
    neuron_indices: List[int]
    layer_name: str
    group_size: int
    cohesion_score: float
    mean_activation: float

@dataclass 
class StandaloneLearningEvent:
    """Standalone learning event representation."""
    event_id: int
    sample_indices: List[int]
    activated_groups: List[int]
    learning_strength: float
    skill_type: str

class StandalonePyTorchVisualizer:
    """Standalone PyTorch neuron group visualizer."""
    
    def __init__(self, output_dir: str = "demo_outputs/standalone_pytorch"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if TORCH_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
        
        logger.info(f"Standalone visualizer initialized with device: {self.device}")
    
    def extract_activations(self, model, dataloader, layer_names, max_batches=None):
        """Extract activations from PyTorch model."""
        if not TORCH_AVAILABLE:
            return {}
        
        model.eval()
        model.to(self.device)
        
        activations = {name: [] for name in layer_names}
        hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name].append(output.detach().cpu())
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_hook(name))
                hooks.append(hook)
        
        # Extract activations
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                _ = model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate activations
        final_activations = {}
        for name in layer_names:
            if activations[name]:
                concatenated = torch.cat(activations[name], dim=0)
                if concatenated.dim() > 2:
                    batch_size = concatenated.size(0)
                    concatenated = concatenated.view(batch_size, -1)
                final_activations[name] = concatenated
        
        return final_activations
    
    def identify_groups(self, activation_tensor, layer_name, threshold=0.4, min_size=2):
        """Identify neuron groups using correlation clustering."""
        if not isinstance(activation_tensor, torch.Tensor):
            return []
        
        # Convert to numpy for correlation analysis
        activations = activation_tensor.cpu().numpy()
        n_neurons = activations.shape[1]
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(activations.T)
        
        # Find groups
        visited = set()
        groups = []
        group_id = 0
        
        for i in range(n_neurons):
            if i in visited:
                continue
            
            # Find correlated neurons
            correlated = [i]
            for j in range(n_neurons):
                if i != j and abs(correlation_matrix[i, j]) >= threshold:
                    correlated.append(j)
            
            if len(correlated) >= min_size:
                for neuron_idx in correlated:
                    visited.add(neuron_idx)
                
                # Calculate group statistics
                group_activations = activations[:, correlated]
                
                # Calculate cohesion
                if len(correlated) > 1:
                    group_corr_matrix = np.corrcoef(group_activations.T)
                    upper_tri = np.triu_indices_from(group_corr_matrix, k=1)
                    cohesion = np.mean(group_corr_matrix[upper_tri])
                else:
                    cohesion = 1.0
                
                group = StandaloneNeuronGroup(
                    group_id=group_id,
                    neuron_indices=correlated,
                    layer_name=layer_name,
                    group_size=len(correlated),
                    cohesion_score=float(cohesion) if not np.isnan(cohesion) else 0.0,
                    mean_activation=float(np.mean(group_activations))
                )
                
                groups.append(group)
                group_id += 1
        
        return groups
    
    def analyze_learning_patterns(self, activations_dict, groups_dict, metadata=None):
        """Analyze learning patterns from activations and groups."""
        learning_events = []
        event_id = 0
        
        # For simplicity, use first layer for learning analysis
        first_layer = list(activations_dict.keys())[0]
        if first_layer not in groups_dict:
            return learning_events
        
        activations = activations_dict[first_layer]
        groups = groups_dict[first_layer]
        n_samples = activations.size(0)
        
        # Identify learning events every N samples
        for i in range(0, n_samples, 10):
            sample_activations = activations[i].cpu().numpy()
            
            # Determine which groups are active
            activated_groups = []
            for group in groups:
                group_activation = np.mean(sample_activations[group.neuron_indices])
                threshold = np.mean(sample_activations) + 0.5 * np.std(sample_activations)
                
                if group_activation > threshold:
                    activated_groups.append(group.group_id)
            
            if activated_groups:
                # Determine skill type
                skill_type = "general"
                if metadata is not None and len(metadata) > i:
                    if 'category' in metadata.columns:
                        skill_type = metadata.iloc[i]['category']
                
                learning_strength = np.max([
                    np.mean(sample_activations[group.neuron_indices]) 
                    for group in groups if group.group_id in activated_groups
                ])
                
                event = StandaloneLearningEvent(
                    event_id=event_id,
                    sample_indices=[i],
                    activated_groups=activated_groups,
                    learning_strength=float(learning_strength),
                    skill_type=skill_type
                )
                
                learning_events.append(event)
                event_id += 1
        
        return learning_events
    
    def create_visualizations(self, activations_dict, groups_dict):
        """Create simple visualizations."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for visualizations")
            return []
        
        plot_paths = []
        
        # 1. Groups per layer
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layer_names = list(groups_dict.keys())
        group_counts = [len(groups_dict[layer]) for layer in layer_names]
        
        bars = ax.bar(layer_names, group_counts, color=['#EE4C2C', '#FF9500', '#007ACC'])
        ax.set_title('Neuron Groups per Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Groups')
        ax.set_xlabel('Layer')
        
        # Add value labels on bars
        for bar, count in zip(bars, group_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
        
        plot_path = self.output_dir / "groups_per_layer.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # 2. Group cohesion distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_cohesions = []
        layer_labels = []
        
        for layer_name, groups in groups_dict.items():
            for group in groups:
                all_cohesions.append(group.cohesion_score)
                layer_labels.append(f"{layer_name}_G{group.group_id}")
        
        if all_cohesions:
            colors = plt.cm.viridis(np.linspace(0, 1, len(all_cohesions)))
            bars = ax.bar(range(len(all_cohesions)), all_cohesions, color=colors)
            
            ax.set_title('Group Cohesion Scores', fontsize=14, fontweight='bold')
            ax.set_ylabel('Cohesion Score')
            ax.set_xlabel('Groups')
            ax.set_xticks(range(len(layer_labels)))
            ax.set_xticklabels(layer_labels, rotation=45, ha='right')
        
        plot_path = self.output_dir / "group_cohesion.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # 3. Activation heatmap for first layer
        if layer_names and layer_names[0] in activations_dict:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            first_layer = layer_names[0]
            activations = activations_dict[first_layer].cpu().numpy()
            
            # Show first 50 samples and reorganize by groups
            n_show = min(50, activations.shape[0])
            show_activations = activations[:n_show]
            
            # Reorganize columns by groups
            groups = groups_dict[first_layer]
            grouped_indices = []
            
            for group in groups:
                grouped_indices.extend(group.neuron_indices)
            
            # Add ungrouped neurons
            all_neurons = set(range(activations.shape[1]))
            grouped_set = set(grouped_indices)
            ungrouped = list(all_neurons - grouped_set)
            grouped_indices.extend(ungrouped)
            
            reordered = show_activations[:, grouped_indices]
            
            im = ax.imshow(reordered.T, cmap='viridis', aspect='auto')
            ax.set_title(f'Activation Heatmap - {first_layer}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Samples')
            ax.set_ylabel('Neurons (Grouped)')
            
            plt.colorbar(im, ax=ax, label='Activation Strength')
            
            plot_path = self.output_dir / f"activation_heatmap_{first_layer}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def generate_report(self, activations_dict, groups_dict, learning_events, model_info=None):
        """Generate analysis report."""
        report = {
            'standalone_analysis': {
                'device': self.device,
                'torch_available': TORCH_AVAILABLE,
                'total_layers': len(activations_dict),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'model_info': model_info or {},
            'layer_analysis': {},
            'learning_patterns': {},
            'summary': {}
        }
        
        # Layer analysis
        total_groups = 0
        total_neurons = 0
        
        for layer_name, groups in groups_dict.items():
            layer_info = {
                'groups_found': len(groups),
                'groups_detail': []
            }
            
            if layer_name in activations_dict:
                layer_info['activation_shape'] = list(activations_dict[layer_name].shape)
            
            for group in groups:
                group_info = {
                    'group_id': group.group_id,
                    'size': group.group_size,
                    'cohesion': group.cohesion_score,
                    'mean_activation': group.mean_activation
                }
                layer_info['groups_detail'].append(group_info)
                total_neurons += group.group_size
            
            report['layer_analysis'][layer_name] = layer_info
            total_groups += len(groups)
        
        # Learning patterns
        if learning_events:
            skill_counts = {}
            for event in learning_events:
                skill_counts[event.skill_type] = skill_counts.get(event.skill_type, 0) + 1
            
            report['learning_patterns'] = {
                'total_events': len(learning_events),
                'skill_distribution': skill_counts,
                'average_strength': np.mean([e.learning_strength for e in learning_events])
            }
        
        # Summary
        report['summary'] = {
            'total_groups': total_groups,
            'total_neurons_in_groups': total_neurons,
            'average_group_size': total_neurons / total_groups if total_groups > 0 else 0,
            'analysis_complete': True
        }
        
        # Save report
        report_path = self.output_dir / "standalone_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path

def main():
    """Run standalone PyTorch demo."""
    print("🚀 Standalone PyTorch Neuron Group Demo gestartet...")
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch nicht verfügbar")
        return False
    
    print(f"✅ PyTorch {torch.__version__} verfügbar auf {torch.cuda.is_available() and 'CUDA' or 'CPU'}")
    
    # Create simple model
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8) 
            self.fc3 = nn.Linear(8, 4)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    print("\n📊 Erstelle Test Model...")
    
    model = TestNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Create structured test data
    n_samples = 80
    input_size = 8
    
    X = torch.randn(n_samples, input_size)
    
    # Add patterns to create correlations
    X[:n_samples//4, :4] += 1.0  # First pattern
    X[n_samples//4:n_samples//2, 4:] += 1.0  # Second pattern  
    X[n_samples//2:3*n_samples//4, :2] += 1.5  # Third pattern
    X[3*n_samples//4:, 6:] += 1.5  # Fourth pattern
    
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    print(f"✅ Test data: {n_samples} samples, {input_size} features")
    
    # Initialize visualizer
    visualizer = StandalonePyTorchVisualizer()
    
    # Extract activations
    layer_names = ['fc1', 'fc2', 'fc3']
    print(f"\n🔍 Extrahiere Aktivierungen von {layer_names}...")
    
    activations_dict = visualizer.extract_activations(
        model, dataloader, layer_names, max_batches=5
    )
    
    if not activations_dict:
        print("❌ Keine Aktivierungen extrahiert")
        return False
    
    print(f"✅ Aktivierungen extrahiert für {len(activations_dict)} layers")
    
    # Identify groups
    groups_dict = {}
    total_groups = 0
    
    for layer_name, activations in activations_dict.items():
        print(f"\n🧩 Analysiere {layer_name}...")
        
        groups = visualizer.identify_groups(
            activations, layer_name, threshold=0.3, min_size=2
        )
        
        groups_dict[layer_name] = groups
        total_groups += len(groups)
        
        print(f"   ✅ {len(groups)} groups gefunden")
        for group in groups:
            print(f"      Group {group.group_id}: {group.group_size} neurons, "
                  f"cohesion: {group.cohesion_score:.3f}")
    
    # Create metadata
    categories = ['pattern_a'] * (n_samples//4) + ['pattern_b'] * (n_samples//4) + \
                ['pattern_c'] * (n_samples//4) + ['pattern_d'] * (n_samples - 3*(n_samples//4))
    
    metadata = pd.DataFrame({'category': categories})
    
    # Analyze learning patterns
    print(f"\n📚 Analysiere Learning Patterns...")
    
    learning_events = visualizer.analyze_learning_patterns(
        activations_dict, groups_dict, metadata
    )
    
    print(f"✅ {len(learning_events)} learning events gefunden")
    
    # Create visualizations
    print(f"\n🎨 Erstelle Visualisierungen...")
    
    plot_paths = visualizer.create_visualizations(activations_dict, groups_dict)
    
    for plot_path in plot_paths:
        print(f"   ✅ {plot_path.name}")
    
    # Generate report
    print(f"\n📋 Erstelle Report...")
    
    model_info = {
        'model_type': 'TestNet',
        'parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device)
    }
    
    report_path = visualizer.generate_report(
        activations_dict, groups_dict, learning_events, model_info
    )
    
    print(f"✅ Report: {report_path}")
    
    # Summary
    print(f"\n🎯 Demo Zusammenfassung:")
    print(f"   📊 Layers: {len(activations_dict)}")
    print(f"   🧩 Groups: {total_groups}")
    print(f"   📚 Learning Events: {len(learning_events)}")
    print(f"   🎨 Visualizations: {len(plot_paths)}")
    print(f"   💻 Device: {device}")
    
    # List generated files
    print(f"\n📁 Generierte Dateien:")
    for file_path in sorted(visualizer.output_dir.iterdir()):
        if file_path.is_file():
            size_kb = file_path.stat().st_size / 1024
            print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
    
    print(f"\n🎉 Standalone Demo erfolgreich abgeschlossen!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔥 Standalone PyTorch Neuron Group Demo")
    print("=" * 60)
    
    success = main()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 Demo erfolgreich! Standalone PyTorch Analysis funktioniert.")
    else:
        print("⚠️  Demo fehlgeschlagen.")
    print("="*60)
