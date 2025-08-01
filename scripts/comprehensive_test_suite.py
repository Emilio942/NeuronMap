#!/usr/bin/env python3
"""
Comprehensive Test Suite for PyTorch Neuron Group Visualization
=============================================================

End-to-end testing following the improved prompt requirements.
"""

import sys
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_models():
    """Create various model architectures for testing."""
    
    class SimpleLinearNet(nn.Module):
        """Basic linear network for testing."""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 15)
            self.fc3 = nn.Linear(15, 5)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    class ConvNet(nn.Module):
        """Convolutional network for testing."""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc1 = nn.Linear(16 * 4 * 4, 32)
            self.fc2 = nn.Linear(32, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    class DeepNet(nn.Module):
        """Deeper network for testing scalability."""
        def __init__(self):
            super().__init__()
            layers = []
            sizes = [50, 100, 80, 60, 40, 20, 10]
            
            for i in range(len(sizes) - 1):
                layers.extend([
                    nn.Linear(sizes[i], sizes[i+1]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
            
            self.network = nn.Sequential(*layers[:-1])  # Remove last dropout
            
        def forward(self, x):
            return self.network(x)
    
    return {
        'simple_linear': SimpleLinearNet(),
        'conv_net': ConvNet(),
        'deep_net': DeepNet()
    }

def create_test_datasets():
    """Create various datasets for testing."""
    
    # Linear data with patterns
    n_samples = 200
    
    # Simple linear data
    X_linear = torch.randn(n_samples, 10)
    # Add correlations
    X_linear[:, 1] = X_linear[:, 0] + torch.randn(n_samples) * 0.1
    X_linear[:, 2] = X_linear[:, 0] * 0.8 + torch.randn(n_samples) * 0.2
    
    # Conv data (images)
    X_conv = torch.randn(n_samples, 1, 8, 8)
    
    # Deep network data
    X_deep = torch.randn(n_samples, 50)
    # Create structured patterns
    X_deep[:n_samples//4, :10] += 2.0  # Pattern 1
    X_deep[n_samples//4:n_samples//2, 10:20] += 1.5  # Pattern 2
    X_deep[n_samples//2:3*n_samples//4, 20:30] += 1.8  # Pattern 3
    X_deep[3*n_samples//4:, 30:40] += 2.2  # Pattern 4
    
    return {
        'linear_data': DataLoader(TensorDataset(X_linear), batch_size=32, shuffle=False),
        'conv_data': DataLoader(TensorDataset(X_conv), batch_size=32, shuffle=False),
        'deep_data': DataLoader(TensorDataset(X_deep), batch_size=32, shuffle=False)
    }

class TestSuite:
    """Comprehensive test suite for PyTorch neuron group system."""
    
    def __init__(self):
        self.results = {
            'activation_extraction': {},
            'group_identification': {},
            'visualization_generation': {},
            'report_creation': {},
            'performance_metrics': {},
            'errors': []
        }
        
        # Import the standalone visualizer
        try:
            from scripts.standalone_pytorch_demo import StandalonePyTorchVisualizer
            self.visualizer_class = StandalonePyTorchVisualizer
            self.standalone_available = True
        except ImportError as e:
            self.standalone_available = False
            self.results['errors'].append(f"Standalone visualizer import failed: {e}")
    
    def test_activation_extraction(self):
        """Test 1: Activation extraction functionality."""
        print("\n🔍 Test 1: Activation Extraction")
        
        if not self.standalone_available:
            print("❌ Skipped - Standalone visualizer not available")
            return False
        
        models = create_test_models()
        datasets = create_test_datasets()
        
        test_cases = [
            ('simple_linear', 'linear_data', ['fc1', 'fc2', 'fc3']),
            ('conv_net', 'conv_data', ['conv1', 'fc1', 'fc2']),
            ('deep_net', 'deep_data', ['network.0', 'network.3', 'network.6'])
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for model_name, data_name, layer_names in test_cases:
            try:
                model = models[model_name]
                dataloader = datasets[data_name]
                
                visualizer = self.visualizer_class(f"test_outputs/extraction_{model_name}")
                
                start_time = time.time()
                activations = visualizer.extract_activations(
                    model, dataloader, layer_names, max_batches=3
                )
                extraction_time = time.time() - start_time
                
                # Validate activations
                success = True
                for layer_name in layer_names:
                    if layer_name not in activations:
                        print(f"   ❌ {model_name}: Layer {layer_name} not extracted")
                        success = False
                        continue
                    
                    tensor = activations[layer_name]
                    if not isinstance(tensor, torch.Tensor):
                        print(f"   ❌ {model_name}: {layer_name} not a tensor")
                        success = False
                        continue
                    
                    if tensor.shape[0] == 0:
                        print(f"   ❌ {model_name}: {layer_name} empty tensor")
                        success = False
                        continue
                    
                    print(f"   ✅ {model_name}.{layer_name}: {tensor.shape}")
                
                if success:
                    passed_tests += 1
                    self.results['activation_extraction'][model_name] = {
                        'success': True,
                        'layers_extracted': len(activations),
                        'extraction_time': extraction_time,
                        'shapes': {name: list(tensor.shape) for name, tensor in activations.items()}
                    }
                else:
                    self.results['activation_extraction'][model_name] = {'success': False}
                    
            except Exception as e:
                print(f"   ❌ {model_name}: Exception - {e}")
                self.results['activation_extraction'][model_name] = {'success': False, 'error': str(e)}
        
        success_rate = passed_tests / total_tests
        print(f"   📊 Activation Extraction: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        
        return success_rate >= 0.8
    
    def test_group_identification(self):
        """Test 2: Group identification algorithms."""
        print("\n🧩 Test 2: Group Identification")
        
        if not self.standalone_available:
            print("❌ Skipped - Standalone visualizer not available")
            return False
        
        # Create test data with known correlations
        n_samples = 100
        n_neurons = 30
        
        # Create synthetic activation data with clear groups
        activations = torch.zeros(n_samples, n_neurons)
        
        # Group 1: neurons 0-9 (highly correlated)
        base_pattern_1 = torch.randn(n_samples)
        for i in range(10):
            noise = torch.randn(n_samples) * 0.1
            activations[:, i] = base_pattern_1 + noise
        
        # Group 2: neurons 10-19 (moderately correlated)
        base_pattern_2 = torch.randn(n_samples)
        for i in range(10, 20):
            noise = torch.randn(n_samples) * 0.3
            activations[:, i] = base_pattern_2 + noise
        
        # Group 3: neurons 20-29 (independent)
        for i in range(20, 30):
            activations[:, i] = torch.randn(n_samples)
        
        visualizer = self.visualizer_class("test_outputs/group_identification")
        
        # Test different thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        threshold_results = {}
        
        for threshold in thresholds:
            try:
                start_time = time.time()
                groups = visualizer.identify_groups(
                    activations, 'test_layer', 
                    threshold=threshold, min_size=3
                )
                identification_time = time.time() - start_time
                
                # Analyze results
                group_sizes = [g.group_size for g in groups]
                cohesion_scores = [g.cohesion_score for g in groups]
                
                threshold_results[threshold] = {
                    'num_groups': len(groups),
                    'total_neurons_grouped': sum(group_sizes),
                    'avg_group_size': np.mean(group_sizes) if group_sizes else 0,
                    'avg_cohesion': np.mean(cohesion_scores) if cohesion_scores else 0,
                    'identification_time': identification_time
                }
                
                print(f"   ✅ Threshold {threshold}: {len(groups)} groups, "
                      f"avg cohesion: {np.mean(cohesion_scores):.3f}")
                
            except Exception as e:
                print(f"   ❌ Threshold {threshold}: {e}")
                threshold_results[threshold] = {'error': str(e)}
        
        self.results['group_identification'] = threshold_results
        
        # Success if we found groups at multiple thresholds
        successful_thresholds = sum(1 for r in threshold_results.values() if 'num_groups' in r and r['num_groups'] > 0)
        success_rate = successful_thresholds / len(thresholds)
        
        print(f"   📊 Group Identification: {successful_thresholds}/{len(thresholds)} thresholds successful ({success_rate:.1%})")
        
        return success_rate >= 0.5
    
    def test_visualization_generation(self):
        """Test 3: Visualization generation."""
        print("\n🎨 Test 3: Visualization Generation")
        
        if not self.standalone_available:
            print("❌ Skipped - Standalone visualizer not available")
            return False
        
        try:
            # Create simple test case
            model = create_test_models()['simple_linear']
            dataloader = create_test_datasets()['linear_data']
            
            visualizer = self.visualizer_class("test_outputs/visualization")
            
            # Extract activations
            activations = visualizer.extract_activations(
                model, dataloader, ['fc1', 'fc2'], max_batches=2
            )
            
            # Identify groups
            groups = {}
            for layer_name, activation_tensor in activations.items():
                groups[layer_name] = visualizer.identify_groups(
                    activation_tensor, layer_name, threshold=0.3
                )
            
            # Generate visualizations
            start_time = time.time()
            plot_paths = visualizer.create_visualizations(activations, groups)
            visualization_time = time.time() - start_time
            
            # Check if files were created
            successful_plots = 0
            for plot_path in plot_paths:
                if Path(plot_path).exists():
                    file_size = Path(plot_path).stat().st_size
                    print(f"   ✅ {Path(plot_path).name}: {file_size} bytes")
                    successful_plots += 1
                else:
                    print(f"   ❌ {Path(plot_path).name}: File not created")
            
            self.results['visualization_generation'] = {
                'total_plots': len(plot_paths),
                'successful_plots': successful_plots,
                'visualization_time': visualization_time,
                'plot_paths': [str(p) for p in plot_paths]
            }
            
            success_rate = successful_plots / len(plot_paths) if plot_paths else 0
            print(f"   📊 Visualization: {successful_plots}/{len(plot_paths)} plots created ({success_rate:.1%})")
            
            return success_rate >= 0.8
            
        except Exception as e:
            print(f"   ❌ Visualization generation failed: {e}")
            self.results['visualization_generation'] = {'error': str(e)}
            return False
    
    def test_report_creation(self):
        """Test 4: Report creation."""
        print("\n📋 Test 4: Report Creation")
        
        if not self.standalone_available:
            print("❌ Skipped - Standalone visualizer not available")
            return False
        
        try:
            # Create test data
            model = create_test_models()['simple_linear']
            dataloader = create_test_datasets()['linear_data']
            
            visualizer = self.visualizer_class("test_outputs/reports")
            
            # Run analysis
            activations = visualizer.extract_activations(
                model, dataloader, ['fc1', 'fc2'], max_batches=2
            )
            
            groups = {}
            for layer_name, activation_tensor in activations.items():
                groups[layer_name] = visualizer.identify_groups(
                    activation_tensor, layer_name, threshold=0.3
                )
            
            learning_events = visualizer.analyze_learning_patterns(
                activations, groups
            )
            
            # Generate report
            start_time = time.time()
            report_path = visualizer.generate_report(
                activations, groups, learning_events,
                model_info={'test': True, 'parameters': 1000}
            )
            report_time = time.time() - start_time
            
            # Validate report
            if Path(report_path).exists():
                file_size = Path(report_path).stat().st_size
                print(f"   ✅ Report created: {Path(report_path).name} ({file_size} bytes)")
                
                # Try to load and validate JSON
                try:
                    import json
                    with open(report_path, 'r') as f:
                        report_data = json.load(f)
                    
                    required_sections = ['standalone_analysis', 'layer_analysis', 'summary']
                    sections_present = sum(1 for section in required_sections if section in report_data)
                    
                    print(f"   ✅ Report structure: {sections_present}/{len(required_sections)} sections present")
                    
                    self.results['report_creation'] = {
                        'success': True,
                        'file_size': file_size,
                        'report_time': report_time,
                        'sections_present': sections_present,
                        'total_sections': len(required_sections)
                    }
                    
                    return sections_present >= len(required_sections) - 1
                    
                except json.JSONDecodeError as e:
                    print(f"   ❌ Report JSON invalid: {e}")
                    return False
                    
            else:
                print(f"   ❌ Report file not created")
                return False
                
        except Exception as e:
            print(f"   ❌ Report creation failed: {e}")
            self.results['report_creation'] = {'error': str(e)}
            return False
    
    def test_performance_metrics(self):
        """Test 5: Performance and scalability."""
        print("\n⚡ Test 5: Performance Metrics")
        
        if not self.standalone_available:
            print("❌ Skipped - Standalone visualizer not available")
            return False
        
        performance_tests = [
            ('small', 50, 10, 2),    # 50 samples, 10 neurons, 2 layers
            ('medium', 200, 50, 3),  # 200 samples, 50 neurons, 3 layers
            ('large', 500, 100, 4),  # 500 samples, 100 neurons, 4 layers
        ]
        
        for test_name, n_samples, n_neurons, n_layers in performance_tests:
            try:
                print(f"   🧪 Testing {test_name} scale: {n_samples} samples, {n_neurons} neurons")
                
                # Create model
                layers = []
                layer_names = []
                prev_size = 20
                
                for i in range(n_layers):
                    layer_size = max(5, n_neurons - i * 15)
                    layers.extend([nn.Linear(prev_size, layer_size), nn.ReLU()])
                    layer_names.append(f'layer_{i*2}')  # Only Linear layers
                    prev_size = layer_size
                
                model = nn.Sequential(*layers)
                
                # Create data
                X = torch.randn(n_samples, 20)
                dataloader = DataLoader(TensorDataset(X), batch_size=32)
                
                # Time the full pipeline
                start_time = time.time()
                
                visualizer = self.visualizer_class(f"test_outputs/performance_{test_name}")
                
                # Extraction
                extract_start = time.time()
                activations = visualizer.extract_activations(
                    model, dataloader, layer_names[:min(3, len(layer_names))]
                )
                extract_time = time.time() - extract_start
                
                # Group identification
                group_start = time.time()
                groups = {}
                total_groups = 0
                for layer_name, activation_tensor in activations.items():
                    layer_groups = visualizer.identify_groups(
                        activation_tensor, layer_name, threshold=0.3
                    )
                    groups[layer_name] = layer_groups
                    total_groups += len(layer_groups)
                group_time = time.time() - group_start
                
                total_time = time.time() - start_time
                
                # Memory usage (approximate)
                total_memory = sum(tensor.numel() * tensor.element_size() 
                                 for tensor in activations.values())
                
                metrics = {
                    'total_time': total_time,
                    'extract_time': extract_time,
                    'group_time': group_time,
                    'total_groups': total_groups,
                    'memory_bytes': total_memory,
                    'samples_per_second': n_samples / total_time
                }
                
                self.results['performance_metrics'][test_name] = metrics
                
                print(f"      ⏱️  Total time: {total_time:.2f}s")
                print(f"      🧩 Groups found: {total_groups}")
                print(f"      💾 Memory used: {total_memory/1024/1024:.1f}MB")
                print(f"      🚀 Speed: {metrics['samples_per_second']:.1f} samples/s")
                
            except Exception as e:
                print(f"   ❌ {test_name} performance test failed: {e}")
                self.results['performance_metrics'][test_name] = {'error': str(e)}
        
        # Success if at least 2/3 performance tests completed
        successful_tests = sum(1 for metrics in self.results['performance_metrics'].values() 
                             if 'total_time' in metrics)
        return successful_tests >= 2
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("🧪 Starting Comprehensive PyTorch Neuron Group Test Suite")
        print("=" * 70)
        
        test_results = []
        
        # Run all tests
        test_results.append(("Activation Extraction", self.test_activation_extraction()))
        test_results.append(("Group Identification", self.test_group_identification()))
        test_results.append(("Visualization Generation", self.test_visualization_generation()))
        test_results.append(("Report Creation", self.test_report_creation()))
        test_results.append(("Performance Metrics", self.test_performance_metrics()))
        
        # Calculate overall results
        passed_tests = sum(1 for _, passed in test_results if passed)
        total_tests = len(test_results)
        overall_success_rate = passed_tests / total_tests
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 70)
        
        for test_name, passed in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed ({overall_success_rate:.1%})")
        
        if overall_success_rate >= 0.8:
            print("🎉 Test suite PASSED - System is ready for production!")
        elif overall_success_rate >= 0.6:
            print("⚠️  Test suite PARTIAL - Some issues need attention")
        else:
            print("❌ Test suite FAILED - Major issues detected")
        
        # Save detailed results
        results_path = Path("test_outputs/comprehensive_test_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_path}")
        
        return overall_success_rate

def main():
    """Run the comprehensive test suite."""
    test_suite = TestSuite()
    return test_suite.run_all_tests()

if __name__ == "__main__":
    success_rate = main()
    exit_code = 0 if success_rate >= 0.8 else 1
    sys.exit(exit_code)
