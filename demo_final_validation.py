#!/usr/bin/env python3
"""
NeuronMap Interpretability Tools - Demo and Validation
=====================================================

Comprehensive demo script showcasing all implemented interpretability
tools with validation and example usage patterns.
"""

import torch
import numpy as np
import json
import yaml
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import all implemented tools
try:
    from src.analysis.interpretability.ig_explainer import create_integrated_gradients_explainer
    from src.analysis.interpretability.shap_explainer import create_shap_explainer
    from src.analysis.interpretability.semantic_labeling import create_semantic_labeler
    from src.analysis.concepts.ace_extractor import create_ace_extractor
    from src.analysis.testing.coverage_tracker import create_coverage_tracker
    from src.analysis.testing.surprise_tracker import create_surprise_tracker
    from src.analysis.metrics.wasserstein_comparator import create_wasserstein_comparator
    from src.analysis.metrics.emd_heatmap import create_emd_comparator
    from src.analysis.mechanistic.transformerlens_adapter import create_transformerlens_adapter
    from src.analysis.mechanistic.residual_stream_comparator import create_residual_stream_comparator
    TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some tools not available: {e}")
    TOOLS_AVAILABLE = False

class NeuronMapDemo:
    """Comprehensive demo for NeuronMap interpretability tools."""
    
    def __init__(self):
        self.demo_results = {}
        self.failed_demos = []
        
        # Create synthetic test data
        self.test_data = self._create_test_data()
        
        logger.info("NeuronMap Demo initialized")
    
    def _create_test_data(self) -> Dict[str, Any]:
        """Create synthetic test data for demonstrations."""
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create synthetic model activations
        batch_size = 32
        seq_length = 128
        hidden_dim = 768
        
        test_data = {
            'model_activations': torch.randn(batch_size, seq_length, hidden_dim),
            'baseline_activations': torch.randn(batch_size, seq_length, hidden_dim) * 0.1,
            'input_embeddings': torch.randn(batch_size, seq_length, hidden_dim),
            'target_labels': torch.randint(0, 10, (batch_size,)),
            'attention_weights': torch.softmax(torch.randn(batch_size, 12, seq_length, seq_length), dim=-1),
            'text_inputs': [f"Sample text input {i} for analysis" for i in range(batch_size)],
            'heatmap_a': np.random.rand(64, 64),
            'heatmap_b': np.random.rand(64, 64) + 0.2,
            'cluster_data': np.random.randn(1000, 50),
            'concept_descriptions': [
                "Neural network layers",
                "Attention mechanisms", 
                "Feature representations",
                "Gradient patterns",
                "Activation clusters"
            ]
        }
        
        # Create more realistic heatmaps
        test_data['heatmap_a'] = self._create_realistic_heatmap(64, 64, 'gaussian')
        test_data['heatmap_b'] = self._create_realistic_heatmap(64, 64, 'circular')
        
        return test_data
    
    def _create_realistic_heatmap(self, height: int, width: int, pattern: str) -> np.ndarray:
        """Create realistic synthetic heatmaps."""
        
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        X, Y = np.meshgrid(x, y)
        
        if pattern == 'gaussian':
            # Multiple Gaussian peaks
            heatmap = (np.exp(-(X**2 + Y**2)) + 
                      0.7 * np.exp(-((X-1)**2 + (Y-1)**2)) +
                      0.5 * np.exp(-((X+1)**2 + (Y+0.5)**2)))
        elif pattern == 'circular':
            # Circular patterns
            heatmap = np.sin(np.sqrt(X**2 + Y**2) * np.pi) * np.exp(-0.5 * (X**2 + Y**2))
        else:
            # Random pattern
            heatmap = np.random.rand(height, width)
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    
    def demo_integrated_gradients(self) -> Dict[str, Any]:
        """Demo Integrated Gradients explainer."""
        
        logger.info("🔍 Demo: Integrated Gradients")
        
        try:
            # Create tool
            ig_tool = create_integrated_gradients_explainer({
                'steps': 50,
                'method': 'riemann_trapezoid'
            })
            
            if not ig_tool.initialize():
                raise RuntimeError("Failed to initialize IG tool")
            
            # Simple model for testing
            def simple_model(x):
                return torch.sum(x * torch.randn_like(x), dim=-1)
            
            # Execute
            result = ig_tool.execute(
                model=simple_model,
                inputs=self.test_data['input_embeddings'][:4],  # Small batch
                baselines=self.test_data['baseline_activations'][:4]
            )
            
            if result.success:
                logger.info("✅ Integrated Gradients demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'attribution_shape': result.outputs.get('attributions', {}).get('shape', 'N/A')
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ Integrated Gradients demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_shap_explainer(self) -> Dict[str, Any]:
        """Demo SHAP explainer."""
        
        logger.info("🔍 Demo: DeepSHAP Explainer")
        
        try:
            shap_tool = create_shap_explainer({
                'n_samples': 100,
                'feature_names': [f'feature_{i}' for i in range(50)]
            })
            
            if not shap_tool.initialize():
                raise RuntimeError("Failed to initialize SHAP tool")
            
            # Simple function for SHAP analysis
            def prediction_function(x):
                return np.sum(x * np.random.randn(x.shape[1]), axis=1)
            
            # Execute
            result = shap_tool.execute(
                model=prediction_function,
                data=self.test_data['cluster_data'][:100],  # Use cluster data
                background_data=self.test_data['cluster_data'][100:200]
            )
            
            if result.success:
                logger.info("✅ SHAP explainer demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'shap_values_shape': result.outputs.get('shap_values', {}).get('shape', 'N/A')
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ SHAP explainer demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_semantic_labeling(self) -> Dict[str, Any]:
        """Demo semantic labeling."""
        
        logger.info("🔍 Demo: Semantic Labeling")
        
        try:
            semantic_tool = create_semantic_labeler({
                'method': 'tfidf',  # Use TF-IDF method to avoid API requirements
                'max_features': 100
            })
            
            if not semantic_tool.initialize():
                raise RuntimeError("Failed to initialize semantic labeling tool")
            
            # Execute
            result = semantic_tool.execute(
                activations=self.test_data['model_activations'][:10].numpy(),
                concept_descriptions=self.test_data['concept_descriptions'],
                cluster_assignments=np.random.randint(0, 5, 10)
            )
            
            if result.success:
                logger.info("✅ Semantic labeling demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'num_labels': len(result.outputs.get('semantic_labels', {}))
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ Semantic labeling demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_ace_concepts(self) -> Dict[str, Any]:
        """Demo ACE concept extraction."""
        
        logger.info("🔍 Demo: ACE Concept Extraction")
        
        try:
            ace_tool = create_ace_extractor({
                'n_clusters': 5,
                'clustering_method': 'kmeans',
                'max_features': 100
            })
            
            if not ace_tool.initialize():
                raise RuntimeError("Failed to initialize ACE tool")
            
            # Execute
            result = ace_tool.execute(
                activations=self.test_data['cluster_data'],
                concept_descriptions=self.test_data['concept_descriptions']
            )
            
            if result.success:
                logger.info("✅ ACE concept extraction demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'num_concepts': len(result.outputs.get('extracted_concepts', {}))
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ ACE concept extraction demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_coverage_tracker(self) -> Dict[str, Any]:
        """Demo neuron coverage tracker."""
        
        logger.info("🔍 Demo: Neuron Coverage Tracker")
        
        try:
            coverage_tool = create_coverage_tracker({
                'threshold': 0.1,
                'layer_names': ['layer_0', 'layer_1', 'layer_2']
            })
            
            if not coverage_tool.initialize():
                raise RuntimeError("Failed to initialize coverage tracker")
            
            # Simple model for coverage tracking
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer_0 = torch.nn.Linear(768, 512)
                    self.layer_1 = torch.nn.Linear(512, 256)
                    self.layer_2 = torch.nn.Linear(256, 10)
                
                def forward(self, x):
                    x = torch.relu(self.layer_0(x))
                    x = torch.relu(self.layer_1(x))
                    x = self.layer_2(x)
                    return x
            
            model = SimpleModel()
            
            # Execute
            result = coverage_tool.execute(
                model=model,
                input_data=self.test_data['input_embeddings'][:8],
                track_layers=['layer_0', 'layer_1', 'layer_2']
            )
            
            if result.success:
                logger.info("✅ Coverage tracker demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'layers_tracked': len(result.outputs.get('layer_coverage', {}))
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ Coverage tracker demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_surprise_tracker(self) -> Dict[str, Any]:
        """Demo surprise coverage tracker."""
        
        logger.info("🔍 Demo: Surprise Coverage Tracker")
        
        try:
            surprise_tool = create_surprise_tracker({
                'outlier_method': 'zscore',
                'threshold': 2.0,
                'baseline_size': 100
            })
            
            if not surprise_tool.initialize():
                raise RuntimeError("Failed to initialize surprise tracker")
            
            # Execute
            result = surprise_tool.execute(
                current_activations=self.test_data['model_activations'][:16].numpy(),
                baseline_activations=self.test_data['baseline_activations'][:100].numpy(),
                layer_name='test_layer'
            )
            
            if result.success:
                logger.info("✅ Surprise tracker demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'num_surprises': result.outputs.get('summary_statistics', {}).get('total_surprises', 0)
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ Surprise tracker demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_wasserstein_distance(self) -> Dict[str, Any]:
        """Demo Wasserstein distance comparator."""
        
        logger.info("🔍 Demo: Wasserstein Distance Comparator")
        
        try:
            wass_tool = create_wasserstein_comparator({
                'distance_type': 'euclidean',
                'approximation_samples': 100
            })
            
            if not wass_tool.initialize():
                raise RuntimeError("Failed to initialize Wasserstein tool")
            
            # Execute
            result = wass_tool.execute(
                distribution_a=self.test_data['cluster_data'][:100],
                distribution_b=self.test_data['cluster_data'][100:200] + 0.5  # Shifted distribution
            )
            
            if result.success:
                logger.info("✅ Wasserstein distance demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'distance': result.outputs.get('wasserstein_distance', 'N/A')
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ Wasserstein distance demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_emd_heatmap(self) -> Dict[str, Any]:
        """Demo EMD heatmap comparator."""
        
        logger.info("🔍 Demo: EMD Heatmap Comparator")
        
        try:
            emd_tool = create_emd_comparator({
                'normalization': 'sum',
                'distance_metric': 'euclidean'
            })
            
            if not emd_tool.initialize():
                raise RuntimeError("Failed to initialize EMD tool")
            
            # Execute
            result = emd_tool.execute(
                heatmap_a=self.test_data['heatmap_a'],
                heatmap_b=self.test_data['heatmap_b']
            )
            
            if result.success:
                logger.info("✅ EMD heatmap demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'emd_distance': result.outputs.get('emd_distance', 'N/A')
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ EMD heatmap demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_transformerlens_adapter(self) -> Dict[str, Any]:
        """Demo TransformerLens adapter."""
        
        logger.info("🔍 Demo: TransformerLens Adapter")
        
        try:
            tl_tool = create_transformerlens_adapter({
                'model_name': 'gpt2-small',
                'device': 'cpu'
            })
            
            # Check if TransformerLens is available
            if not hasattr(tl_tool, 'initialized'):
                logger.warning("TransformerLens not available, skipping demo")
                return {'status': 'skipped', 'reason': 'TransformerLens not available'}
            
            if not tl_tool.initialize():
                logger.warning("TransformerLens initialization failed, skipping demo")
                return {'status': 'skipped', 'reason': 'Initialization failed'}
            
            # Execute with simple text
            result = tl_tool.execute(
                input_text="Hello world, this is a test.",
                analysis_type="activations"
            )
            
            if result.success:
                logger.info("✅ TransformerLens adapter demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'model_name': result.outputs.get('model_name', 'N/A')
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ TransformerLens adapter demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def demo_residual_stream_comparator(self) -> Dict[str, Any]:
        """Demo residual stream comparator."""
        
        logger.info("🔍 Demo: Residual Stream Comparator")
        
        try:
            residual_tool = create_residual_stream_comparator({
                'similarity_metrics': ['cosine', 'euclidean'],
                'dimensionality_reduction': 'pca'
            })
            
            if not residual_tool.initialize():
                raise RuntimeError("Failed to initialize residual stream tool")
            
            # Create mock TL and NeuronMap data
            tl_data = {
                'model_name': 'gpt2-small',
                'residual_stream_data': {
                    'blocks.0.hook_resid_post': {
                        'data': self.test_data['model_activations'][:8].numpy().tolist(),
                        'shape': list(self.test_data['model_activations'][:8].shape)
                    }
                }
            }
            
            nm_data = {
                'neuron_activations': {
                    'blocks.0.hook_resid_post': {
                        'activations': self.test_data['baseline_activations'][:8].numpy().tolist(),
                        'shape': list(self.test_data['baseline_activations'][:8].shape)
                    }
                }
            }
            
            # Execute
            result = residual_tool.execute(
                tl_data=tl_data,
                neuronmap_data=nm_data,
                comparison_type="similarity"
            )
            
            if result.success:
                logger.info("✅ Residual stream comparator demo successful")
                return {
                    'status': 'success',
                    'execution_time': result.execution_time,
                    'output_keys': list(result.outputs.keys()),
                    'layers_compared': result.outputs.get('metadata', {}).get('layers_compared', 0)
                }
            else:
                raise RuntimeError(f"Execution failed: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ Residual stream comparator demo failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_all_demos(self) -> Dict[str, Any]:
        """Run all available demos."""
        
        logger.info("🚀 Starting comprehensive NeuronMap tool demos")
        
        if not TOOLS_AVAILABLE:
            logger.error("Tools not available - cannot run demos")
            return {'error': 'Tools not available'}
        
        # List of demo functions
        demos = [
            ('integrated_gradients', self.demo_integrated_gradients),
            ('shap_explainer', self.demo_shap_explainer),
            ('semantic_labeling', self.demo_semantic_labeling),
            ('ace_concepts', self.demo_ace_concepts),
            ('coverage_tracker', self.demo_coverage_tracker),
            ('surprise_tracker', self.demo_surprise_tracker),
            ('wasserstein_distance', self.demo_wasserstein_distance),
            ('emd_heatmap', self.demo_emd_heatmap),
            ('transformerlens_adapter', self.demo_transformerlens_adapter),
            ('residual_stream_comparator', self.demo_residual_stream_comparator)
        ]
        
        results = {}
        start_time = time.time()
        
        for demo_name, demo_func in demos:
            logger.info(f"\n--- Running {demo_name} demo ---")
            
            try:
                demo_result = demo_func()
                results[demo_name] = demo_result
                
                if demo_result['status'] == 'success':
                    logger.info(f"✅ {demo_name} completed successfully")
                elif demo_result['status'] == 'skipped':
                    logger.info(f"⏭️  {demo_name} skipped: {demo_result.get('reason', 'Unknown')}")
                else:
                    logger.error(f"❌ {demo_name} failed")
                    self.failed_demos.append(demo_name)
                    
            except Exception as e:
                logger.error(f"❌ {demo_name} demo crashed: {e}")
                results[demo_name] = {'status': 'crashed', 'error': str(e)}
                self.failed_demos.append(demo_name)
        
        total_time = time.time() - start_time
        
        # Generate summary
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')
        failed = sum(1 for r in results.values() if r.get('status') in ['failed', 'crashed'])
        
        summary = {
            'total_demos': len(demos),
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total_execution_time': total_time,
            'success_rate': successful / len(demos) * 100,
            'failed_demos': self.failed_demos
        }
        
        logger.info(f"\n🎯 Demo Summary:")
        logger.info(f"   Total demos: {summary['total_demos']}")
        logger.info(f"   Successful: {summary['successful']}")
        logger.info(f"   Skipped: {summary['skipped']}")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"   Total time: {summary['total_execution_time']:.2f}s")
        
        if self.failed_demos:
            logger.warning(f"   Failed demos: {', '.join(self.failed_demos)}")
        
        return {
            'summary': summary,
            'detailed_results': results,
            'test_data_info': {
                'batch_size': self.test_data['model_activations'].shape[0],
                'sequence_length': self.test_data['model_activations'].shape[1],
                'hidden_dimension': self.test_data['model_activations'].shape[2],
                'heatmap_size': self.test_data['heatmap_a'].shape
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str = "demo_results.json"):
        """Save demo results to file."""
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📁 Demo results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main function for running demos."""
    
    print("🧠 NeuronMap Interpretability Tools - Comprehensive Demo")
    print("=" * 60)
    
    demo = NeuronMapDemo()
    results = demo.run_all_demos()
    
    # Save results
    demo.save_results(results)
    
    # Exit with appropriate code
    if results['summary']['failed'] > 0:
        print(f"\n⚠️  Some demos failed. Check logs for details.")
        exit(1)
    else:
        print(f"\n🎉 All demos completed successfully!")
        exit(0)

if __name__ == '__main__':
    main()
