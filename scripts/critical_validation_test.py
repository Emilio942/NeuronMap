#!/usr/bin/env python3
"""
Critical Validation Test
========================

This test challenges the fundamental assumptions of the neuron grouping system
by testing against random data, noise, and known failure cases.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from scipy import stats
from sklearn.metrics import adjusted_rand_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.standalone_pytorch_demo import StandalonePyTorchVisualizer

class CriticalValidationTest:
    """Tests that challenge the system's assumptions."""
    
    def __init__(self):
        self.results = {
            'random_data_test': {},
            'noise_sensitivity': {},
            'stability_test': {},
            'baseline_comparison': {},
            'statistical_significance': {}
        }
    
    def test_random_data_performance(self):
        """Test 1: How does the system perform on pure random data?"""
        print("\n🎲 Critical Test 1: Random Data Performance")
        print("Question: Should we find ANY groups in pure random data?")
        
        visualizer = StandalonePyTorchVisualizer("test_outputs/critical_random")
        
        # Pure random data - should find NO meaningful groups
        n_samples, n_neurons = 200, 50
        random_activations = torch.randn(n_samples, n_neurons)
        
        groups_found = []
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            groups = visualizer.identify_groups(
                random_activations, 'random_layer', 
                threshold=threshold, min_size=3
            )
            groups_found.append(len(groups))
            print(f"   Threshold {threshold}: {len(groups)} groups found")
        
        # Statistical test: Are found correlations significant?
        corr_matrix = torch.corrcoef(random_activations.T)
        upper_tri = corr_matrix[torch.triu(torch.ones_like(corr_matrix), diagonal=1) == 1]
        
        # Test if correlations are significantly different from 0
        t_stat, p_value = stats.ttest_1samp(upper_tri.numpy(), 0)
        
        self.results['random_data_test'] = {
            'groups_found_by_threshold': dict(zip(thresholds, groups_found)),
            'total_groups_random': sum(groups_found),
            'correlation_significance': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'mean_correlation': float(upper_tri.mean()),
                'std_correlation': float(upper_tri.std())
            }
        }
        
        # CRITICAL QUESTION: If we find groups in random data, is our method meaningful?
        if sum(groups_found) > 0:
            print(f"   ⚠️  CONCERN: Found {sum(groups_found)} groups in pure random data!")
            print(f"   ⚠️  This suggests the method may be prone to false positives.")
        else:
            print(f"   ✅ Good: No groups found in random data.")
        
        return sum(groups_found) == 0  # Success = no groups in random data
    
    def test_noise_sensitivity(self):
        """Test 2: How sensitive is grouping to noise levels?"""
        print("\n📊 Critical Test 2: Noise Sensitivity")
        print("Question: How robust are groupings to noise?")
        
        visualizer = StandalonePyTorchVisualizer("test_outputs/critical_noise")
        
        # Create data with known groups + varying noise
        n_samples, group_size = 100, 10
        
        base_pattern_1 = torch.randn(n_samples)
        base_pattern_2 = torch.randn(n_samples) * 2
        
        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        group_stability = []
        
        for noise_level in noise_levels:
            # Group 1 with noise
            group1_data = torch.stack([
                base_pattern_1 + torch.randn(n_samples) * noise_level 
                for _ in range(group_size)
            ], dim=1)
            
            # Group 2 with noise
            group2_data = torch.stack([
                base_pattern_2 + torch.randn(n_samples) * noise_level 
                for _ in range(group_size)
            ], dim=1)
            
            # Random neurons
            random_data = torch.randn(n_samples, 10)
            
            test_data = torch.cat([group1_data, group2_data, random_data], dim=1)
            
            groups = visualizer.identify_groups(
                test_data, f'noise_{noise_level}', 
                threshold=0.3, min_size=3
            )
            
            # Analyze if true groups were recovered
            group_sizes = [g.group_size for g in groups]
            cohesions = [g.cohesion_score for g in groups]
            
            group_stability.append({
                'noise_level': noise_level,
                'groups_found': len(groups),
                'avg_group_size': np.mean(group_sizes) if group_sizes else 0,
                'avg_cohesion': np.mean(cohesions) if cohesions else 0
            })
            
            print(f"   Noise {noise_level}: {len(groups)} groups, "
                  f"avg cohesion: {np.mean(cohesions):.3f}")
        
        self.results['noise_sensitivity'] = group_stability
        
        # Check if method degrades gracefully with noise
        cohesions = [gs['avg_cohesion'] for gs in group_stability]
        degradation_rate = (cohesions[0] - cohesions[-1]) / cohesions[0] if cohesions[0] > 0 else 0
        
        print(f"   📉 Cohesion degradation: {degradation_rate:.1%}")
        
        return degradation_rate < 0.8  # Acceptable if < 80% degradation
    
    def test_stability_across_runs(self):
        """Test 3: Are results stable across multiple runs?"""
        print("\n🔄 Critical Test 3: Stability Across Runs")
        print("Question: Do we get consistent results with same data?")
        
        visualizer = StandalonePyTorchVisualizer("test_outputs/critical_stability")
        
        # Create deterministic test data
        torch.manual_seed(42)
        n_samples, n_neurons = 150, 30
        
        # Two clear groups
        base1 = torch.randn(n_samples)
        base2 = torch.randn(n_samples)
        
        test_data = torch.zeros(n_samples, n_neurons)
        for i in range(10):
            test_data[:, i] = base1 + torch.randn(n_samples) * 0.1
        for i in range(10, 20):
            test_data[:, i] = base2 + torch.randn(n_samples) * 0.1
        for i in range(20, 30):
            test_data[:, i] = torch.randn(n_samples)
        
        # Run multiple times with same data
        runs = []
        num_runs = 10
        
        for run in range(num_runs):
            groups = visualizer.identify_groups(
                test_data, f'stability_run_{run}', 
                threshold=0.5, min_size=3
            )
            
            runs.append({
                'run': run,
                'num_groups': len(groups),
                'group_sizes': [g.group_size for g in groups],
                'group_neurons': [g.neuron_indices for g in groups]
            })
        
        # Analyze consistency
        group_counts = [run['num_groups'] for run in runs]
        count_std = np.std(group_counts)
        count_mean = np.mean(group_counts)
        
        print(f"   Groups found across runs: {group_counts}")
        print(f"   Mean: {count_mean:.1f}, Std: {count_std:.1f}")
        
        # Check if the same neurons are consistently grouped
        consistency_scores = []
        if len(runs) > 1:
            for i in range(1, len(runs)):
                # Simple overlap metric
                groups1 = set(tuple(sorted(g)) for g in runs[0]['group_neurons'])
                groups2 = set(tuple(sorted(g)) for g in runs[i]['group_neurons'])
                
                overlap = len(groups1.intersection(groups2))
                total = len(groups1.union(groups2))
                consistency = overlap / total if total > 0 else 0
                consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        
        self.results['stability_test'] = {
            'group_counts': group_counts,
            'count_std': float(count_std),
            'avg_consistency': float(avg_consistency),
            'runs_detail': runs
        }
        
        print(f"   📊 Group count stability: σ = {count_std:.2f}")
        print(f"   🎯 Group consistency: {avg_consistency:.1%}")
        
        return count_std < 1.0 and avg_consistency > 0.7
    
    def test_baseline_comparison(self):
        """Test 4: Compare against random clustering baseline."""
        print("\n🎲 Critical Test 4: Baseline Comparison")
        print("Question: Are our groups better than random clustering?")
        
        visualizer = StandalonePyTorchVisualizer("test_outputs/critical_baseline")
        
        # Create test data with known structure
        n_samples, n_neurons = 120, 24
        
        # True groups: 3 groups of 8 neurons each
        true_labels = np.repeat([0, 1, 2], 8)
        
        test_data = torch.zeros(n_samples, n_neurons)
        for group_id in range(3):
            group_indices = np.where(true_labels == group_id)[0]
            base_pattern = torch.randn(n_samples)
            
            for idx in group_indices:
                test_data[:, idx] = base_pattern + torch.randn(n_samples) * 0.2
        
        # Our method
        our_groups = visualizer.identify_groups(
            test_data, 'baseline_test', threshold=0.4, min_size=3
        )
        
        # Convert to labels for comparison
        our_labels = np.full(n_neurons, -1)
        for i, group in enumerate(our_groups):
            for neuron_idx in group.neuron_indices:
                our_labels[neuron_idx] = i
        
        # Random baseline: Random grouping with same number of groups
        num_groups = len(our_groups) if our_groups else 3
        random_labels = np.random.randint(0, num_groups, n_neurons)
        
        # Calculate Adjusted Rand Index (measures clustering quality)
        our_ari = adjusted_rand_score(true_labels, our_labels)
        random_ari = adjusted_rand_score(true_labels, random_labels)
        
        # K-means baseline
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(test_data.T.numpy())
        kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
        
        self.results['baseline_comparison'] = {
            'our_ari': float(our_ari),
            'random_ari': float(random_ari),
            'kmeans_ari': float(kmeans_ari),
            'our_groups': len(our_groups),
            'improvement_over_random': float(our_ari - random_ari),
            'vs_kmeans': float(our_ari - kmeans_ari)
        }
        
        print(f"   Our method ARI: {our_ari:.3f}")
        print(f"   Random ARI: {random_ari:.3f}")
        print(f"   K-means ARI: {kmeans_ari:.3f}")
        print(f"   Improvement over random: {our_ari - random_ari:.3f}")
        
        return our_ari > random_ari and our_ari > 0.3
    
    def test_statistical_significance(self):
        """Test 5: Are correlations statistically significant?"""
        print("\n📈 Critical Test 5: Statistical Significance")
        print("Question: Are found correlations statistically meaningful?")
        
        visualizer = StandalonePyTorchVisualizer("test_outputs/critical_stats")
        
        # Test with data that has some real correlations
        n_samples = 200
        
        # Create correlated and uncorrelated neurons
        base1 = torch.randn(n_samples)
        base2 = torch.randn(n_samples)
        
        test_data = torch.zeros(n_samples, 20)
        
        # Strongly correlated group (r ≈ 0.8)
        for i in range(5):
            test_data[:, i] = base1 + torch.randn(n_samples) * 0.3
        
        # Weakly correlated group (r ≈ 0.4)
        for i in range(5, 10):
            test_data[:, i] = base2 + torch.randn(n_samples) * 0.8
        
        # Uncorrelated neurons
        for i in range(10, 20):
            test_data[:, i] = torch.randn(n_samples)
        
        # Calculate correlation matrix
        corr_matrix = torch.corrcoef(test_data.T)
        n_neurons = test_data.shape[1]
        
        # Statistical test for each correlation
        significant_pairs = 0
        total_pairs = 0
        p_values = []
        
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                r = corr_matrix[i, j].item()
                
                # Test significance of correlation
                t_stat = r * np.sqrt((n_samples - 2) / (1 - r**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
                p_values.append(p_val)
                
                total_pairs += 1
                if p_val < 0.05:  # Bonferroni correction would be 0.05/total_pairs
                    significant_pairs += 1
        
        # Apply Bonferroni correction
        bonferroni_threshold = 0.05 / total_pairs
        significant_after_correction = sum(1 for p in p_values if p < bonferroni_threshold)
        
        # Get groups from our method
        groups = visualizer.identify_groups(
            test_data, 'significance_test', threshold=0.3, min_size=2
        )
        
        self.results['statistical_significance'] = {
            'total_pairs': total_pairs,
            'significant_pairs_uncorrected': significant_pairs,
            'significant_pairs_bonferroni': significant_after_correction,
            'bonferroni_threshold': bonferroni_threshold,
            'groups_found': len(groups),
            'mean_p_value': float(np.mean(p_values)),
            'min_p_value': float(np.min(p_values))
        }
        
        print(f"   Total correlation pairs: {total_pairs}")
        print(f"   Significant (p<0.05): {significant_pairs}")
        print(f"   Significant (Bonferroni): {significant_after_correction}")
        print(f"   Groups found by method: {len(groups)}")
        
        # Success if we have some significant correlations
        return significant_after_correction > 0
    
    def run_critical_validation(self):
        """Run all critical validation tests."""
        print("🔍 CRITICAL VALIDATION SUITE")
        print("Challenging fundamental assumptions of the neuron grouping system")
        print("=" * 80)
        
        tests = [
            ("Random Data Performance", self.test_random_data_performance),
            ("Noise Sensitivity", self.test_noise_sensitivity),
            ("Stability Across Runs", self.test_stability_across_runs),
            ("Baseline Comparison", self.test_baseline_comparison),
            ("Statistical Significance", self.test_statistical_significance)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                passed = test_func()
                results.append((test_name, passed))
            except Exception as e:
                print(f"   ❌ {test_name} failed with error: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 CRITICAL VALIDATION RESULTS")
        print("=" * 80)
        
        passed_tests = 0
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")
            if passed:
                passed_tests += 1
        
        success_rate = passed_tests / len(results)
        print(f"\n📊 Critical Tests Passed: {passed_tests}/{len(results)} ({success_rate:.1%})")
        
        # Interpretation
        if success_rate >= 0.8:
            print("🎉 SYSTEM PASSES CRITICAL VALIDATION")
            print("   The method appears scientifically sound.")
        elif success_rate >= 0.6:
            print("⚠️  SYSTEM SHOWS SOME CONCERNS")
            print("   Method may work but has limitations.")
        else:
            print("❌ SYSTEM FAILS CRITICAL VALIDATION")
            print("   Fundamental issues with the approach detected.")
        
        # Save results
        results_path = Path("test_outputs/critical_validation_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results: {results_path}")
        
        return success_rate

def main():
    """Run critical validation."""
    validator = CriticalValidationTest()
    return validator.run_critical_validation()

if __name__ == "__main__":
    success_rate = main()
    print(f"\n{'='*80}")
    if success_rate >= 0.6:
        print("✅ Critical validation suggests the system has merit.")
    else:
        print("❌ Critical validation reveals fundamental issues.")
    print(f"{'='*80}")
