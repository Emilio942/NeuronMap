#!/usr/bin/env python3
"""
Improved PyTorch Neuron Group Visualizer
========================================

This version addresses the critical issues identified in the validation:
1. False positives in random data
2. Better threshold selection
3. Statistical significance testing
4. Noise robustness improvements
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImprovedNeuronGroup:
    """Enhanced neuron group with statistical validation."""
    neuron_indices: List[int]
    group_size: int
    cohesion_score: float
    p_value: float  # Statistical significance
    confidence_interval: Tuple[float, float]  # 95% CI for cohesion
    stability_score: float  # Robustness measure
    layer_name: str
    
    @property
    def is_significant(self) -> bool:
        """Check if group is statistically significant."""
        return self.p_value < 0.05
    
    @property
    def is_stable(self) -> bool:
        """Check if group is stable (robust to noise)."""
        return self.stability_score > 0.7

class ImprovedPyTorchVisualizer:
    """Improved PyTorch neuron group visualizer with statistical rigor."""
    
    def __init__(self, output_dir: str, device: str = None):
        """Initialize with enhanced validation."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Improved visualizer initialized with device: {self.device}")
        
        # Enhanced parameters
        self.min_samples_for_stats = 30  # Minimum samples for statistical tests
        self.bootstrap_iterations = 1000  # For confidence intervals
        self.noise_test_iterations = 10   # For stability testing
        
    def extract_activations(self, model: nn.Module, dataloader, 
                          layer_names: List[str], max_batches: int = None) -> Dict[str, torch.Tensor]:
        """Extract activations with validation."""
        model.eval()
        model = model.to(self.device)
        
        activations = {name: [] for name in layer_names}
        hooks = []
        
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name].append(output.detach().cpu())
            return hook_fn
        
        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        try:
            batch_count = 0
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)
                    
                    _ = model(inputs)
                    batch_count += 1
                    
                    if max_batches and batch_count >= max_batches:
                        break
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        # Concatenate activations
        final_activations = {}
        for name in layer_names:
            if activations[name]:
                tensor = torch.cat(activations[name], dim=0)
                # Flatten if needed (keep batch dimension)
                if tensor.dim() > 2:
                    tensor = tensor.view(tensor.size(0), -1)
                final_activations[name] = tensor
                logger.info(f"Extracted {name}: {tensor.shape}")
        
        return final_activations
    
    def _calculate_statistical_threshold(self, correlations: torch.Tensor, 
                                       alpha: float = 0.05) -> float:
        """Calculate statistically motivated threshold."""
        n_samples = correlations.shape[0]
        
        # Fisher transformation for correlation significance
        # Critical correlation for given sample size and alpha
        t_critical = stats.t.ppf(1 - alpha/2, n_samples - 2)
        r_critical = t_critical / np.sqrt(t_critical**2 + n_samples - 2)
        
        return float(r_critical)
    
    def _test_correlation_significance(self, r: float, n_samples: int) -> float:
        """Test if correlation is statistically significant."""
        if abs(r) >= 0.999:  # Avoid division by zero
            return 0.0
        
        # t-test for correlation significance
        t_stat = r * np.sqrt((n_samples - 2) / (1 - r**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
        return p_value
    
    def _bootstrap_confidence_interval(self, data: torch.Tensor, 
                                     indices: List[int]) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for group cohesion."""
        if len(indices) < 2:
            return (0.0, 0.0)
        
        group_data = data[:, indices]
        n_samples = group_data.shape[0]
        
        bootstrap_scores = []
        for _ in range(self.bootstrap_iterations):
            # Bootstrap sample
            boot_indices = torch.randint(0, n_samples, (n_samples,))
            boot_data = group_data[boot_indices]
            
            # Calculate cohesion for bootstrap sample
            corr_matrix = torch.corrcoef(boot_data.T)
            if corr_matrix.numel() == 1:
                cohesion = float(corr_matrix)
            else:
                upper_tri = corr_matrix[torch.triu(torch.ones_like(corr_matrix), diagonal=1) == 1]
                cohesion = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0
            
            bootstrap_scores.append(cohesion)
        
        # 95% confidence interval
        lower = np.percentile(bootstrap_scores, 2.5)
        upper = np.percentile(bootstrap_scores, 97.5)
        
        return (float(lower), float(upper))
    
    def _test_stability(self, data: torch.Tensor, indices: List[int]) -> float:
        """Test group stability against noise."""
        if len(indices) < 2:
            return 0.0
        
        original_data = data[:, indices]
        original_corr = torch.corrcoef(original_data.T)
        
        if original_corr.numel() == 1:
            original_cohesion = float(original_corr)
        else:
            upper_tri = original_corr[torch.triu(torch.ones_like(original_corr), diagonal=1) == 1]
            original_cohesion = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0
        
        stability_scores = []
        noise_levels = [0.1, 0.2, 0.3]  # Different noise levels
        
        for noise_level in noise_levels:
            for _ in range(self.noise_test_iterations):
                # Add noise
                noise = torch.randn_like(original_data) * noise_level
                noisy_data = original_data + noise
                
                # Calculate cohesion with noise
                noisy_corr = torch.corrcoef(noisy_data.T)
                if noisy_corr.numel() == 1:
                    noisy_cohesion = float(noisy_corr)
                else:
                    upper_tri = noisy_corr[torch.triu(torch.ones_like(noisy_corr), diagonal=1) == 1]
                    noisy_cohesion = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0
                
                # Stability as correlation with original
                if original_cohesion != 0:
                    stability = abs(noisy_cohesion / original_cohesion)
                else:
                    stability = 0.0
                
                stability_scores.append(min(stability, 1.0))
        
        return float(np.mean(stability_scores))
    
    def identify_improved_groups(self, activations: torch.Tensor, layer_name: str,
                               min_size: int = 3, max_groups: int = None,
                               use_adaptive_threshold: bool = True) -> List[ImprovedNeuronGroup]:
        """Identify neuron groups with statistical validation."""
        
        if activations.shape[0] < self.min_samples_for_stats:
            logger.warning(f"Too few samples ({activations.shape[0]}) for reliable statistics")
            return []
        
        n_samples, n_neurons = activations.shape
        
        # Calculate correlation matrix
        corr_matrix = torch.corrcoef(activations.T)
        
        # Handle edge cases
        if corr_matrix.numel() == 1 or n_neurons < 2:
            return []
        
        # Determine threshold
        if use_adaptive_threshold:
            # Statistical threshold based on sample size
            threshold = self._calculate_statistical_threshold(activations)
            logger.info(f"Using adaptive threshold: {threshold:.3f} for {n_samples} samples")
        else:
            threshold = 0.5  # Conservative default
        
        # Apply multiple comparison correction (Bonferroni)
        n_comparisons = n_neurons * (n_neurons - 1) // 2
        corrected_alpha = 0.05 / n_comparisons
        corrected_threshold = self._calculate_statistical_threshold(activations, corrected_alpha)
        
        logger.info(f"Bonferroni corrected threshold: {corrected_threshold:.3f}")
        
        # Use the more conservative threshold
        final_threshold = max(threshold, corrected_threshold)
        
        # Find connected components (neurons with high correlation)
        adjacency = (torch.abs(corr_matrix) >= final_threshold).float()
        adjacency.fill_diagonal_(0)  # Remove self-connections
        
        # Group identification using connected components
        visited = torch.zeros(n_neurons, dtype=torch.bool)
        groups = []
        
        for i in range(n_neurons):
            if visited[i]:
                continue
            
            # BFS to find connected component
            component = []
            queue = [i]
            
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                
                visited[node] = True
                component.append(node)
                
                # Add neighbors
                neighbors = torch.where(adjacency[node] > 0)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        queue.append(neighbor.item())
            
            # Validate group
            if len(component) >= min_size:
                # Calculate group statistics
                group_data = activations[:, component]
                group_corr = torch.corrcoef(group_data.T)
                
                if group_corr.numel() == 1:
                    cohesion = float(group_corr)
                    p_value = self._test_correlation_significance(cohesion, n_samples)
                else:
                    upper_tri = group_corr[torch.triu(torch.ones_like(group_corr), diagonal=1) == 1]
                    cohesion = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0
                    
                    # Average p-value for group correlations
                    p_values = []
                    for j in range(len(component)):
                        for k in range(j+1, len(component)):
                            r = float(group_corr[j, k])
                            p_val = self._test_correlation_significance(r, n_samples)
                            p_values.append(p_val)
                    
                    # Use Bonferroni correction for group
                    p_value = min(1.0, min(p_values) * len(p_values)) if p_values else 1.0
                
                # Additional validations
                confidence_interval = self._bootstrap_confidence_interval(activations, component)
                stability_score = self._test_stability(activations, component)
                
                group = ImprovedNeuronGroup(
                    neuron_indices=component,
                    group_size=len(component),
                    cohesion_score=cohesion,
                    p_value=p_value,
                    confidence_interval=confidence_interval,
                    stability_score=stability_score,
                    layer_name=layer_name
                )
                
                # Only add statistically significant and stable groups
                if group.is_significant and group.is_stable:
                    groups.append(group)
                    logger.info(f"Valid group found: {len(component)} neurons, "
                              f"cohesion={cohesion:.3f}, p={p_value:.4f}, "
                              f"stability={stability_score:.3f}")
        
        # Sort by cohesion score
        groups.sort(key=lambda g: g.cohesion_score, reverse=True)
        
        # Limit number of groups if specified
        if max_groups:
            groups = groups[:max_groups]
        
        logger.info(f"Found {len(groups)} statistically validated groups in {layer_name}")
        return groups
    
    def _perform_random_baseline_test(self, n_samples: int, n_neurons: int) -> int:
        """Test how many groups are found in random data."""
        random_data = torch.randn(n_samples, n_neurons)
        random_groups = self.identify_improved_groups(random_data, 'random_test')
        return len(random_groups)
    
    def create_improved_visualizations(self, activations: Dict[str, torch.Tensor],
                                     groups: Dict[str, List[ImprovedNeuronGroup]]) -> List[str]:
        """Create enhanced visualizations with statistical information."""
        plot_paths = []
        
        # 1. Statistical validation plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Validation of Neuron Groups', fontsize=16)
        
        # Collect data for plots
        all_p_values = []
        all_cohesions = []
        all_stabilities = []
        layer_names = []
        
        for layer_name, layer_groups in groups.items():
            for group in layer_groups:
                all_p_values.append(group.p_value)
                all_cohesions.append(group.cohesion_score)
                all_stabilities.append(group.stability_score)
                layer_names.append(layer_name)
        
        if all_p_values:
            # P-value distribution
            axes[0, 0].hist(all_p_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            axes[0, 0].set_xlabel('P-values')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('P-value Distribution')
            axes[0, 0].legend()
            
            # Cohesion vs P-value
            scatter = axes[0, 1].scatter(all_cohesions, all_p_values, 
                                       c=all_stabilities, cmap='viridis', alpha=0.7)
            axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Cohesion Score')
            axes[0, 1].set_ylabel('P-value')
            axes[0, 1].set_title('Cohesion vs Statistical Significance')
            plt.colorbar(scatter, ax=axes[0, 1], label='Stability Score')
            
            # Stability distribution
            axes[1, 0].hist(all_stabilities, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=0.7, color='red', linestyle='--', label='Stability threshold')
            axes[1, 0].set_xlabel('Stability Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Stability Score Distribution')
            axes[1, 0].legend()
            
            # Groups per layer
            layer_counts = {}
            for layer in set(layer_names):
                layer_counts[layer] = layer_names.count(layer)
            
            axes[1, 1].bar(layer_counts.keys(), layer_counts.values())
            axes[1, 1].set_xlabel('Layer')
            axes[1, 1].set_ylabel('Number of Groups')
            axes[1, 1].set_title('Valid Groups per Layer')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        stat_plot_path = self.output_dir / 'statistical_validation.png'
        plt.savefig(stat_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(stat_plot_path))
        
        # 2. Confidence intervals plot
        if all_cohesions:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            group_labels = []
            lower_bounds = []
            upper_bounds = []
            
            for layer_name, layer_groups in groups.items():
                for i, group in enumerate(layer_groups):
                    group_labels.append(f"{layer_name}_G{i+1}")
                    lower_bounds.append(group.confidence_interval[0])
                    upper_bounds.append(group.confidence_interval[1])
            
            y_pos = np.arange(len(group_labels))
            
            # Plot confidence intervals
            ax.barh(y_pos, [u - l for l, u in zip(lower_bounds, upper_bounds)],
                   left=lower_bounds, alpha=0.7)
            
            # Plot point estimates
            for i, cohesion in enumerate(all_cohesions):
                ax.plot(cohesion, i, 'ro', markersize=8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(group_labels)
            ax.set_xlabel('Cohesion Score')
            ax.set_title('95% Confidence Intervals for Group Cohesion')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            ci_plot_path = self.output_dir / 'confidence_intervals.png'
            plt.savefig(ci_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(str(ci_plot_path))
        
        return plot_paths
    
    def generate_improved_report(self, activations: Dict[str, torch.Tensor],
                               groups: Dict[str, List[ImprovedNeuronGroup]],
                               model_info: Dict = None) -> str:
        """Generate comprehensive statistical report."""
        
        report = {
            'improved_analysis': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_info': model_info or {},
                'statistical_parameters': {
                    'min_samples_for_stats': self.min_samples_for_stats,
                    'bootstrap_iterations': self.bootstrap_iterations,
                    'noise_test_iterations': self.noise_test_iterations,
                    'significance_level': 0.05
                }
            },
            'layer_analysis': {},
            'statistical_summary': {},
            'validation_results': {}
        }
        
        total_groups = 0
        total_significant = 0
        total_stable = 0
        all_p_values = []
        all_cohesions = []
        all_stabilities = []
        
        # Analyze each layer
        for layer_name, layer_groups in groups.items():
            layer_stats = {
                'total_groups': len(layer_groups),
                'significant_groups': sum(1 for g in layer_groups if g.is_significant),
                'stable_groups': sum(1 for g in layer_groups if g.is_stable),
                'groups': []
            }
            
            for i, group in enumerate(layer_groups):
                group_info = {
                    'group_id': i + 1,
                    'neuron_indices': group.neuron_indices,
                    'size': group.group_size,
                    'cohesion_score': group.cohesion_score,
                    'p_value': group.p_value,
                    'is_significant': group.is_significant,
                    'confidence_interval': group.confidence_interval,
                    'stability_score': group.stability_score,
                    'is_stable': group.is_stable
                }
                layer_stats['groups'].append(group_info)
                
                # Collect for overall statistics
                all_p_values.append(group.p_value)
                all_cohesions.append(group.cohesion_score)
                all_stabilities.append(group.stability_score)
            
            report['layer_analysis'][layer_name] = layer_stats
            total_groups += len(layer_groups)
            total_significant += layer_stats['significant_groups']
            total_stable += layer_stats['stable_groups']
        
        # Overall statistical summary
        if all_p_values:
            report['statistical_summary'] = {
                'total_groups_found': total_groups,
                'significant_groups': total_significant,
                'stable_groups': total_stable,
                'significance_rate': total_significant / total_groups if total_groups > 0 else 0,
                'stability_rate': total_stable / total_groups if total_groups > 0 else 0,
                'mean_p_value': float(np.mean(all_p_values)),
                'mean_cohesion': float(np.mean(all_cohesions)),
                'mean_stability': float(np.mean(all_stabilities)),
                'bonferroni_correction_applied': True
            }
        
        # Validation against random data
        sample_size = list(activations.values())[0].shape[0] if activations else 100
        neuron_count = list(activations.values())[0].shape[1] if activations else 50
        
        random_groups_found = self._perform_random_baseline_test(sample_size, neuron_count)
        
        report['validation_results'] = {
            'random_baseline_test': {
                'groups_found_in_random_data': random_groups_found,
                'sample_size': sample_size,
                'neuron_count': neuron_count,
                'passes_validation': random_groups_found == 0
            }
        }
        
        # Save report
        report_path = self.output_dir / 'improved_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Improved analysis report saved to {report_path}")
        return str(report_path)

def main():
    """Demonstration of improved system."""
    print("🔬 Improved PyTorch Neuron Group Visualizer")
    print("=" * 50)
    
    # Create test model and data
    model = nn.Sequential(
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Create data with known structure
    n_samples = 150
    X = torch.randn(n_samples, 20)
    
    from torch.utils.data import DataLoader, TensorDataset
    dataloader = DataLoader(TensorDataset(X), batch_size=32, shuffle=False)
    
    # Initialize improved visualizer
    visualizer = ImprovedPyTorchVisualizer("test_outputs/improved_demo")
    
    # Extract activations
    activations = visualizer.extract_activations(model, dataloader, ['0', '2'])
    
    # Identify groups with statistical validation
    groups = {}
    for layer_name, activation_tensor in activations.items():
        groups[layer_name] = visualizer.identify_improved_groups(
            activation_tensor, layer_name, min_size=3
        )
    
    # Create improved visualizations
    plot_paths = visualizer.create_improved_visualizations(activations, groups)
    
    # Generate comprehensive report
    report_path = visualizer.generate_improved_report(
        activations, groups, 
        model_info={'architecture': 'test_model', 'parameters': sum(p.numel() for p in model.parameters())}
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Groups found: {sum(len(g) for g in groups.values())}")
    print(f"🎨 Visualizations: {len(plot_paths)}")
    print(f"📋 Report: {report_path}")

if __name__ == "__main__":
    main()
