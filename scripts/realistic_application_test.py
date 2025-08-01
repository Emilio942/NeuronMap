#!/usr/bin/env python3
"""
Real-World Application Test
==========================

Test the PyTorch neuron group system with a realistic scenario:
- Pre-trained model simulation
- Real-world data patterns  
- Complete analysis pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.standalone_pytorch_demo import StandalonePyTorchVisualizer

class RealisticTestNet(nn.Module):
    """A more realistic neural network for testing."""
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights to create more realistic activations
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def create_realistic_dataset(n_samples=1000, input_size=784, num_classes=10):
    """Create a realistic dataset with class-based patterns."""
    
    X = torch.randn(n_samples, input_size) * 0.1
    y = torch.randint(0, num_classes, (n_samples,))
    
    # Create class-specific patterns
    for class_id in range(num_classes):
        class_mask = y == class_id
        class_samples = class_mask.sum()
        
        if class_samples > 0:
            # Each class has specific features that are activated
            feature_start = class_id * (input_size // num_classes)
            feature_end = min((class_id + 1) * (input_size // num_classes), input_size)
            
            # Add class-specific signal
            class_signal = torch.randn(class_samples, feature_end - feature_start) * 2.0 + 1.0
            X[class_mask, feature_start:feature_end] = class_signal
            
            # Add some noise to other features
            noise_strength = 0.5
            X[class_mask] += torch.randn_like(X[class_mask]) * noise_strength
    
    return X, y

def create_metadata(y, num_classes=10):
    """Create realistic metadata."""
    
    class_names = [
        'numerical', 'spatial', 'linguistic', 'logical', 'memory',
        'visual', 'auditory', 'motor', 'abstract', 'temporal'
    ]
    
    skill_types = [
        'mathematical', 'geometric', 'language', 'reasoning', 'recall',
        'perception', 'audio_processing', 'coordination', 'conceptual', 'sequential'
    ]
    
    difficulty_levels = ['easy', 'medium', 'hard']
    
    metadata = []
    for i, label in enumerate(y):
        metadata.append({
            'sample_id': i,
            'class': int(label),
            'class_name': class_names[int(label)],
            'category': skill_types[int(label)],
            'difficulty': np.random.choice(difficulty_levels),
            'question': f"Task of type {class_names[int(label)]} - sample {i}"
        })
    
    return pd.DataFrame(metadata)

def main():
    """Run realistic application test."""
    
    print("🌟 Real-World PyTorch Neuron Group Application Test")
    print("=" * 60)
    
    # Create realistic model and data
    print("\n📊 Creating realistic test scenario...")
    
    model = RealisticTestNet(
        input_size=784,  # MNIST-like
        hidden_sizes=[256, 128, 64],
        num_classes=10
    )
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create realistic dataset
    X, y = create_realistic_dataset(n_samples=500, input_size=784, num_classes=10)
    metadata = create_metadata(y)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    print(f"✅ Dataset created: {X.shape[0]} samples, {X.shape[1]} features, {len(torch.unique(y))} classes")
    
    # Initialize visualizer
    visualizer = StandalonePyTorchVisualizer("test_outputs/realistic_application")
    
    # Extract activations from key layers
    layer_names = ['network.0', 'network.4', 'network.8']  # Linear layers
    print(f"\n🔍 Extracting activations from layers: {layer_names}")
    
    activations = visualizer.extract_activations(
        model, dataloader, layer_names, max_batches=8
    )
    
    if not activations:
        print("❌ No activations extracted")
        return False
    
    print(f"✅ Extracted activations from {len(activations)} layers")
    for name, tensor in activations.items():
        print(f"   {name}: {tensor.shape}")
    
    # Identify neuron groups with multiple methods
    print(f"\n🧩 Identifying neuron groups...")
    
    groups_dict = {}
    total_groups = 0
    
    for layer_name, activation_tensor in activations.items():
        print(f"\n   Analyzing {layer_name}...")
        
        # Try different thresholds to find optimal grouping
        best_groups = []
        best_score = 0
        
        for threshold in [0.2, 0.3, 0.4, 0.5]:
            groups = visualizer.identify_groups(
                activation_tensor, layer_name, 
                threshold=threshold, min_size=3
            )
            
            if groups:
                avg_cohesion = np.mean([g.cohesion_score for g in groups])
                total_neurons = sum(g.group_size for g in groups)
                score = len(groups) * avg_cohesion * (total_neurons / activation_tensor.shape[1])
                
                if score > best_score:
                    best_score = score
                    best_groups = groups
        
        groups_dict[layer_name] = best_groups
        total_groups += len(best_groups)
        
        if best_groups:
            avg_size = np.mean([g.group_size for g in best_groups])
            avg_cohesion = np.mean([g.cohesion_score for g in best_groups])
            print(f"      ✅ {len(best_groups)} groups found")
            print(f"         Average size: {avg_size:.1f} neurons")
            print(f"         Average cohesion: {avg_cohesion:.3f}")
        else:
            print(f"      ⚠️  No groups found")
    
    print(f"\n🎯 Total groups found across all layers: {total_groups}")
    
    # Analyze learning patterns
    print(f"\n📚 Analyzing learning patterns...")
    
    learning_events = visualizer.analyze_learning_patterns(
        activations, groups_dict, metadata
    )
    
    if learning_events:
        print(f"✅ {len(learning_events)} learning events identified")
        
        # Analyze by skill type
        skill_counts = {}
        for event in learning_events:
            skill_counts[event.skill_type] = skill_counts.get(event.skill_type, 0) + 1
        
        print("   Learning events by skill type:")
        for skill, count in sorted(skill_counts.items()):
            print(f"      {skill}: {count} events")
    else:
        print("⚠️  No learning events identified")
    
    # Create comprehensive visualizations
    print(f"\n🎨 Creating visualizations...")
    
    plot_paths = visualizer.create_visualizations(activations, groups_dict)
    
    print(f"✅ {len(plot_paths)} visualizations created:")
    for plot_path in plot_paths:
        file_size = Path(plot_path).stat().st_size / 1024
        print(f"   📊 {Path(plot_path).name} ({file_size:.1f} KB)")
    
    # Generate comprehensive report
    print(f"\n📋 Generating analysis report...")
    
    model_info = {
        'architecture': 'RealisticTestNet',
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'layer_count': len(list(model.named_modules())),
        'input_size': 784,
        'hidden_sizes': [256, 128, 64],
        'output_size': 10,
        'dataset_size': X.shape[0],
        'num_classes': len(torch.unique(y))
    }
    
    report_path = visualizer.generate_report(
        activations, groups_dict, learning_events, model_info
    )
    
    report_size = Path(report_path).stat().st_size / 1024
    print(f"✅ Analysis report generated: {Path(report_path).name} ({report_size:.1f} KB)")
    
    # Performance summary
    print(f"\n⚡ Performance Summary:")
    print(f"   🧩 Total neuron groups: {total_groups}")
    print(f"   📚 Learning events: {len(learning_events)}")
    print(f"   🎨 Visualizations: {len(plot_paths)}")
    print(f"   📊 Layers analyzed: {len(activations)}")
    print(f"   💾 Total activations: {sum(t.numel() for t in activations.values())} values")
    
    # Success criteria
    success_criteria = [
        total_groups > 0,
        len(learning_events) > 0,
        len(plot_paths) >= 3,
        Path(report_path).exists()
    ]
    
    passed_criteria = sum(success_criteria)
    success_rate = passed_criteria / len(success_criteria)
    
    print(f"\n🎯 Success Criteria: {passed_criteria}/{len(success_criteria)} met ({success_rate:.1%})")
    
    if success_rate >= 0.75:
        print("🎉 REALISTIC APPLICATION TEST PASSED!")
        print("   The system successfully handles real-world scenarios.")
    else:
        print("⚠️  REALISTIC APPLICATION TEST PARTIAL")
        print("   Some functionality may need improvement.")
    
    # List all generated files
    output_dir = Path("test_outputs/realistic_application")
    if output_dir.exists():
        print(f"\n📁 Generated files in {output_dir}:")
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
    
    return success_rate >= 0.75

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("✅ Real-world application test SUCCESSFUL!")
    else:
        print("❌ Real-world application test needs attention.")
    print("=" * 60)
