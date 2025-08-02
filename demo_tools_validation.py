#!/usr/bin/env python3
"""
Demo Tools Validation for NeuronMap Interpretability Tools
=========================================================

Comprehensive validation script that tests all implemented tools
with GPT-2 and random inputs to ensure full functionality.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import traceback
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoToolsValidator:
    """Comprehensive validation of all NeuronMap interpretability tools."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.validation_results = {}
        self.test_model = None
        self.test_data = {}
        
        # Initialize test environment
        self.setup_test_environment()
        
    def setup_test_environment(self):
        """Set up test model and data."""
        logger.info("🔧 Setting up test environment...")
        
        try:
            # Create simple test model (GPT-2 style architecture)
            self.test_model = self._create_test_model()
            
            # Generate test data
            self.test_data = {
                'input_tensor': torch.randn(2, 10, 512),  # Batch, seq_len, hidden_dim
                'target_layer': 'layer_1',
                'concepts': self._generate_test_concepts(),
                'baseline_data': torch.randn(5, 10, 512),
                'comparison_data': torch.randn(2, 10, 512)
            }
            
            logger.info("✅ Test environment ready")
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise
    
    def _create_test_model(self):
        """Create a simple test model."""
        
        class TestTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 512)
                self.layer_1 = torch.nn.Linear(512, 512)
                self.layer_2 = torch.nn.Linear(512, 512)
                self.output = torch.nn.Linear(512, 1000)
                self.dropout = torch.nn.Dropout(0.1)
                
            def forward(self, x):
                if x.dim() == 3:  # Already embedded
                    x = self.dropout(x)
                else:  # Token indices
                    x = self.embedding(x)
                    x = self.dropout(x)
                
                x = torch.relu(self.layer_1(x))
                x = torch.relu(self.layer_2(x))
                return self.output(x)
        
        model = TestTransformer()
        model.eval()
        return model
    
    def _generate_test_concepts(self):
        """Generate test concept data."""
        return {
            'concept_a': {
                'id': 'test_concept_a',
                'description': 'Test concept A for validation',
                'activations': torch.randn(10, 512),
                'examples': ['example_1', 'example_2'],
                'cav_vector': torch.randn(512)
            },
            'concept_b': {
                'id': 'test_concept_b', 
                'description': 'Test concept B for validation',
                'activations': torch.randn(10, 512),
                'examples': ['example_3', 'example_4'],
                'cav_vector': torch.randn(512)
            }
        }
    
    def validate_all_tools(self) -> Dict[str, Any]:
        """Validate all implemented tools."""
        logger.info("🚀 Starting comprehensive tool validation...")
        
        # Define all tools to validate
        tools_to_validate = [
            ('integrated_gradients', self._validate_integrated_gradients),
            ('deepshap_explainer', self._validate_deepshap),
            ('semantic_labeling', self._validate_semantic_labeling),
            ('ace_concepts', self._validate_ace_concepts),
            ('tcav_plus_comparator', self._validate_tcav_plus),
            ('neuron_coverage', self._validate_neuron_coverage),
            ('surprise_coverage', self._validate_surprise_coverage),
            ('wasserstein_distance', self._validate_wasserstein),
            ('emd_heatmap', self._validate_emd_heatmap),
            ('transformerlens_adapter', self._validate_transformerlens),
            ('residual_stream_comparator', self._validate_residual_stream)
        ]
        
        validation_summary = {
            'total_tools': len(tools_to_validate),
            'successful_validations': 0,
            'failed_validations': 0,
            'tool_results': {},
            'validation_time': 0
        }
        
        start_time = time.time()
        
        for tool_name, validation_func in tools_to_validate:
            logger.info(f"\n--- Validating {tool_name} ---")
            
            try:
                result = validation_func()
                
                if result['success']:
                    validation_summary['successful_validations'] += 1
                    logger.info(f"✅ {tool_name} validation passed")
                else:
                    validation_summary['failed_validations'] += 1
                    logger.error(f"❌ {tool_name} validation failed: {result['error']}")
                
                validation_summary['tool_results'][tool_name] = result
                
            except Exception as e:
                logger.error(f"💥 {tool_name} validation crashed: {e}")
                logger.error(traceback.format_exc())
                
                validation_summary['failed_validations'] += 1
                validation_summary['tool_results'][tool_name] = {
                    'success': False,
                    'error': f"Validation crashed: {e}",
                    'traceback': traceback.format_exc()
                }
        
        validation_summary['validation_time'] = time.time() - start_time
        validation_summary['success_rate'] = (
            validation_summary['successful_validations'] / 
            validation_summary['total_tools'] * 100
        )
        
        return validation_summary
    
    def _validate_integrated_gradients(self) -> Dict[str, Any]:
        """Validate Integrated Gradients tool."""
        try:
            # Import and test the tool
            from src.analysis.interpretability.ig_explainer import IntegratedGradientsExplainer
            
            explainer = IntegratedGradientsExplainer()
            
            # Test with sample data
            result = explainer.explain(
                model=self.test_model,
                input_tensor=self.test_data['input_tensor'],
                target_layer=self.test_data['target_layer'],
                baseline=torch.zeros_like(self.test_data['input_tensor'])
            )
            
            # Validate output
            assert 'attributions' in result
            assert isinstance(result['attributions'], torch.Tensor)
            assert result['attributions'].shape == self.test_data['input_tensor'].shape
            
            return {'success': True, 'output_shape': result['attributions'].shape}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_deepshap(self) -> Dict[str, Any]:
        """Validate DeepSHAP tool."""
        try:
            from src.analysis.interpretability.shap_explainer import DeepSHAPExplainer
            
            explainer = DeepSHAPExplainer()
            
            result = explainer.explain(
                model=self.test_model,
                background_data=self.test_data['baseline_data'],
                test_data=self.test_data['input_tensor']
            )
            
            assert 'shap_values' in result
            assert 'feature_importance' in result
            
            return {'success': True, 'features_analyzed': len(result['feature_importance'])}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_semantic_labeling(self) -> Dict[str, Any]:
        """Validate Semantic Labeling tool."""
        try:
            from src.analysis.interpretability.semantic_labeling import SemanticLabeler
            
            labeler = SemanticLabeler()
            
            # Create test cluster data
            test_clusters = {
                'cluster_0': {
                    'activations': torch.randn(5, 512),
                    'examples': ['test example 1', 'test example 2']
                }
            }
            
            result = labeler.label_clusters(test_clusters)
            
            assert 'semantic_labels' in result
            assert isinstance(result['semantic_labels'], dict)
            
            return {'success': True, 'clusters_labeled': len(result['semantic_labels'])}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_ace_concepts(self) -> Dict[str, Any]:
        """Validate ACE Concept Extractor."""
        try:
            from src.analysis.concepts.ace_extractor import ACEConceptExtractor
            
            extractor = ACEConceptExtractor()
            
            result = extractor.extract_concepts(
                model=self.test_model,
                dataset=self.test_data['input_tensor'],
                layer_name=self.test_data['target_layer'],
                num_concepts=5
            )
            
            assert 'extracted_concepts' in result
            assert 'concept_scores' in result
            
            return {'success': True, 'concepts_found': len(result['extracted_concepts'])}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_tcav_plus(self) -> Dict[str, Any]:
        """Validate TCAV++ Comparator."""
        try:
            from src.analysis.concepts.tcav_plus_comparator import TCAVPlusComparator
            
            comparator = TCAVPlusComparator()
            
            result = comparator.compare_concepts(
                concept_a=self.test_data['concepts']['concept_a'],
                concept_b=self.test_data['concepts']['concept_b']
            )
            
            assert 'similarity_metrics' in result
            assert 'compatibility_score' in result
            
            return {'success': True, 'similarity_score': result['compatibility_score']}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_neuron_coverage(self) -> Dict[str, Any]:
        """Validate Neuron Coverage tool."""
        try:
            from src.analysis.testing.coverage_tracker import NeuronCoverageTracker
            
            tracker = NeuronCoverageTracker()
            
            result = tracker.track_coverage(
                model=self.test_model,
                inputs=self.test_data['input_tensor'],
                layer_names=['layer_1', 'layer_2']
            )
            
            assert 'coverage_statistics' in result
            assert 'layer_coverage' in result
            
            return {'success': True, 'layers_tracked': len(result['layer_coverage'])}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_surprise_coverage(self) -> Dict[str, Any]:
        """Validate Surprise Coverage tool."""
        try:
            from src.analysis.testing.surprise_tracker import SurpriseCoverageTracker
            
            tracker = SurpriseCoverageTracker()
            
            result = tracker.track_surprise(
                model=self.test_model,
                baseline_distribution=self.test_data['baseline_data'],
                test_inputs=self.test_data['input_tensor']
            )
            
            assert 'surprise_statistics' in result
            assert 'outlier_indices' in result
            
            return {'success': True, 'outliers_found': len(result['outlier_indices'])}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_wasserstein(self) -> Dict[str, Any]:
        """Validate Wasserstein Distance tool."""
        try:
            from src.analysis.metrics.wasserstein_comparator import WassersteinComparator
            
            comparator = WassersteinComparator()
            
            dist_a = torch.randn(100, 512).numpy()
            dist_b = torch.randn(100, 512).numpy()
            
            result = comparator.compare_distributions(dist_a, dist_b)
            
            assert 'wasserstein_distance' in result
            assert isinstance(result['wasserstein_distance'], (int, float))
            
            return {'success': True, 'distance': result['wasserstein_distance']}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_emd_heatmap(self) -> Dict[str, Any]:
        """Validate EMD Heatmap tool."""
        try:
            from src.analysis.metrics.emd_heatmap import EMDHeatmapComparator
            
            comparator = EMDHeatmapComparator()
            
            cluster_map_a = np.random.rand(10, 10)
            cluster_map_b = np.random.rand(10, 10)
            
            result = comparator.compare_clustermaps(cluster_map_a, cluster_map_b)
            
            assert 'emd_distance' in result
            assert isinstance(result['emd_distance'], (int, float))
            
            return {'success': True, 'emd_distance': result['emd_distance']}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_transformerlens(self) -> Dict[str, Any]:
        """Validate TransformerLens Adapter."""
        try:
            from src.analysis.mechanistic.transformerlens_adapter import TransformerLensAdapter
            
            adapter = TransformerLensAdapter()
            
            result = adapter.adapt_model(
                model_name='gpt2',
                hook_points=['hook_embed', 'hook_pos_embed']
            )
            
            assert 'neuron_activations' in result
            assert 'model_name' in result
            
            return {'success': True, 'model_adapted': result['model_name']}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_residual_stream(self) -> Dict[str, Any]:
        """Validate Residual Stream Comparator."""
        try:
            from src.analysis.mechanistic.residual_stream_comparator import ResidualStreamComparator
            
            comparator = ResidualStreamComparator()
            
            neuronmap_data = {'layer_1': torch.randn(10, 512)}
            transformerlens_data = {'layer_1': torch.randn(10, 512)}
            
            result = comparator.compare_streams(
                neuronmap_data=neuronmap_data,
                transformerlens_data=transformerlens_data
            )
            
            assert 'comparison_summary' in result
            assert 'similarity_scores' in result
            
            return {'success': True, 'layers_compared': len(result['similarity_scores'])}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report."""
        
        report = f"""
🎯 NeuronMap Tools Validation Report
====================================

📊 Summary Statistics:
  Total Tools Tested: {results['total_tools']}
  Successful Validations: {results['successful_validations']}
  Failed Validations: {results['failed_validations']}
  Success Rate: {results['success_rate']:.1f}%
  Total Validation Time: {results['validation_time']:.2f}s

📋 Individual Tool Results:
"""
        
        for tool_name, result in results['tool_results'].items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            report += f"  {tool_name}: {status}"
            
            if result['success']:
                # Add specific success details
                if 'output_shape' in result:
                    report += f" (Output shape: {result['output_shape']})"
                elif 'similarity_score' in result:
                    report += f" (Similarity: {result['similarity_score']:.3f})"
                elif 'layers_tracked' in result:
                    report += f" (Layers: {result['layers_tracked']})"
                elif 'distance' in result:
                    report += f" (Distance: {result['distance']:.3f})"
            else:
                report += f" - Error: {result['error']}"
            
            report += "\n"
        
        if results['success_rate'] == 100:
            report += "\n🎉 ALL TOOLS VALIDATED SUCCESSFULLY!"
        else:
            report += f"\n⚠️ {results['failed_validations']} tools need attention"
        
        return report

def main():
    """Main validation function."""
    print("🧪 NeuronMap Demo Tools Validation")
    print("=" * 50)
    
    validator = DemoToolsValidator()
    
    try:
        # Run comprehensive validation
        results = validator.validate_all_tools()
        
        # Generate and display report
        report = validator.generate_validation_report(results)
        print(report)
        
        # Save results
        results_file = Path(__file__).parent / "demo_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to {results_file}")
        
        # Exit with appropriate code
        if results['success_rate'] == 100:
            print("\n🎉 All validations passed! NeuronMap tools are ready for production.")
            sys.exit(0)
        else:
            print(f"\n⚠️ {results['failed_validations']} validations failed. Check errors above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
