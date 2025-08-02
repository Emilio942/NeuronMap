#!/usr/bin/env python3
"""
CLI Validator for NeuronMap Interpretability Tools
=================================================

Automated CLI tester that validates all tools can be executed
with --test-mode and produce valid outputs.
"""

import subprocess
import json
import yaml
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIValidator:
    """Automated CLI validation for all NeuronMap tools."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.cli_script = self.project_root / "src" / "cli_integration.py"
        self.tools_registry = {}
        self.validation_results = {}
        
        # Load tools registry
        self.load_tools_registry()
        
    def load_tools_registry(self):
        """Load the tools registry configuration."""
        try:
            registry_path = self.project_root / "configs" / "tools_registry.yaml"
            
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry_data = yaml.safe_load(f)
                
                # Extract tools from categories structure
                categories = registry_data.get('categories', {})
                tools = {}
                
                for category_name, category_data in categories.items():
                    category_tools = category_data.get('tools', {})
                    tools.update(category_tools)
                
                self.tools_registry = tools
                logger.info(f"Loaded {len(self.tools_registry)} tools from registry")
            else:
                logger.error(f"Tools registry not found: {registry_path}")
                
        except Exception as e:
            logger.error(f"Failed to load tools registry: {e}")
    
    def create_test_config(self, tool_id: str, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create test configuration for a tool."""
        
        # Default test configuration
        test_config = {
            'test_mode': True,
            'batch_size': 4,
            'max_samples': 10
        }
        
        # Tool-specific test configurations
        if tool_id == 'integrated_gradients':
            test_config.update({
                'steps': 10,
                'method': 'riemann_trapezoid'
            })
        elif tool_id == 'deepshap_explainer':
            test_config.update({
                'n_samples': 50,
                'feature_names': [f'feature_{i}' for i in range(20)]
            })
        elif tool_id == 'semantic_labeling':
            test_config.update({
                'method': 'tfidf',
                'max_features': 50
            })
        elif tool_id == 'ace_concepts':
            test_config.update({
                'n_clusters': 3,
                'clustering_method': 'kmeans'
            })
        elif tool_id == 'neuron_coverage':
            test_config.update({
                'threshold': 0.1,
                'layer_names': ['test_layer']
            })
        elif tool_id == 'surprise_coverage':
            test_config.update({
                'outlier_method': 'zscore',
                'threshold': 2.0
            })
        elif tool_id == 'wasserstein_distance':
            test_config.update({
                'distance_type': 'euclidean',
                'approximation_samples': 50
            })
        elif tool_id == 'emd_heatmap':
            test_config.update({
                'normalization': 'sum',
                'distance_metric': 'euclidean'
            })
        elif tool_id == 'transformerlens_adapter':
            test_config.update({
                'model_name': 'gpt2-small',
                'device': 'cpu'
            })
        elif tool_id == 'residual_stream_comparator':
            test_config.update({
                'similarity_metrics': ['cosine'],
                'dimensionality_reduction': 'pca'
            })
        elif tool_id == 'tcav_plus_comparator':
            test_config.update({
                'similarity_threshold': 0.7,
                'normalize_activations': True
            })
        
        return test_config
    
    def create_test_input(self, tool_id: str) -> Dict[str, Any]:
        """Create test input data for a tool."""
        
        import numpy as np
        
        # Common test data
        test_input = {
            'activations': np.random.randn(10, 20).tolist(),
            'heatmap': np.random.rand(16, 16).tolist(),
            'text_input': "This is a test input for analysis.",
            'labels': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'concept_descriptions': ["test concept A", "test concept B", "test concept C"]
        }
        
        # Tool-specific test inputs
        if tool_id in ['integrated_gradients']:
            test_input['model_type'] = 'test_model'
            test_input['input_tensor'] = np.random.randn(4, 10, 20).tolist()
            
        elif tool_id in ['deepshap_explainer']:
            test_input['data'] = np.random.randn(20, 10).tolist()
            test_input['background_data'] = np.random.randn(20, 10).tolist()
            
        elif tool_id in ['semantic_labeling']:
            test_input['cluster_assignments'] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
            
        elif tool_id in ['ace_concepts']:
            test_input['activations'] = np.random.randn(50, 30).tolist()
            
        elif tool_id in ['neuron_coverage', 'surprise_coverage']:
            test_input['model_activations'] = {
                'layer_0': np.random.randn(10, 20).tolist(),
                'layer_1': np.random.randn(10, 15).tolist()
            }
            
        elif tool_id in ['wasserstein_distance']:
            test_input['distribution_a'] = np.random.randn(30, 10).tolist()
            test_input['distribution_b'] = np.random.randn(30, 10).tolist()
            
        elif tool_id in ['emd_heatmap']:
            test_input['heatmap_a'] = np.random.rand(16, 16).tolist()
            test_input['heatmap_b'] = np.random.rand(16, 16).tolist()
            
        elif tool_id in ['transformerlens_adapter']:
            test_input['input_text'] = "Hello world, this is a test."
            test_input['analysis_type'] = "activations"
            
        elif tool_id in ['residual_stream_comparator']:
            test_input['tl_data'] = {
                'model_name': 'test_model',
                'residual_stream_data': {
                    'layer_0': {
                        'data': np.random.randn(5, 10).tolist(),
                        'shape': [5, 10]
                    }
                }
            }
            test_input['neuronmap_data'] = {
                'neuron_activations': {
                    'layer_0': {
                        'activations': np.random.randn(5, 10).tolist(),
                        'shape': [5, 10]
                    }
                }
            }
            
        elif tool_id == 'tcav_plus_comparator':
            test_input['concept_a'] = {
                'id': 'concept_a',
                'activations': np.random.randn(10, 15).tolist(),
                'cav_vector': np.random.randn(15).tolist()
            }
            test_input['concept_b'] = {
                'id': 'concept_b', 
                'activations': np.random.randn(10, 15).tolist(),
                'cav_vector': np.random.randn(15).tolist()
            }
        
        return test_input
    
    def validate_tool_cli(self, tool_id: str, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single tool via CLI."""
        
        logger.info(f"Validating CLI for tool: {tool_id}")
        
        validation_result = {
            'tool_id': tool_id,
            'success': False,
            'execution_time': 0.0,
            'output_valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Create test configuration and input
            test_config = self.create_test_config(tool_id, tool_config)
            test_input = self.create_test_input(tool_id)
            
            # Save test files
            config_file = self.project_root / f"test_config_{tool_id}.json"
            input_file = self.project_root / f"test_input_{tool_id}.json"
            output_file = self.project_root / f"test_output_{tool_id}.json"
            
            with open(config_file, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            with open(input_file, 'w') as f:
                json.dump(test_input, f, indent=2)
            
            # Construct CLI command
            cmd = [
                sys.executable, str(self.cli_script),
                'execute', tool_id,
                '--config', str(config_file),
                '--input', str(input_file),
                '--output', str(output_file)
            ]
            
            # Execute CLI command
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                cwd=str(self.project_root)
            )
            execution_time = time.time() - start_time
            
            validation_result['execution_time'] = execution_time
            
            # Check if command succeeded
            if result.returncode == 0:
                validation_result['success'] = True
                logger.info(f"✅ {tool_id} CLI execution successful")
                
                # Validate output file
                if output_file.exists():
                    try:
                        with open(output_file, 'r') as f:
                            output_data = json.load(f)
                        
                        # Basic output validation
                        if self.validate_tool_output(output_data, tool_id):
                            validation_result['output_valid'] = True
                            logger.info(f"✅ {tool_id} output validation successful")
                        else:
                            validation_result['warnings'].append("Output validation failed")
                            logger.warning(f"⚠️ {tool_id} output validation failed")
                    
                    except Exception as e:
                        validation_result['errors'].append(f"Output parsing failed: {e}")
                        logger.error(f"❌ {tool_id} output parsing failed: {e}")
                else:
                    validation_result['warnings'].append("No output file generated")
                    logger.warning(f"⚠️ {tool_id} no output file generated")
            
            else:
                validation_result['errors'].append(f"CLI execution failed with code {result.returncode}")
                validation_result['errors'].append(f"STDOUT: {result.stdout}")
                validation_result['errors'].append(f"STDERR: {result.stderr}")
                logger.error(f"❌ {tool_id} CLI execution failed")
            
            # Cleanup test files
            for test_file in [config_file, input_file, output_file]:
                if test_file.exists():
                    test_file.unlink()
            
        except subprocess.TimeoutExpired:
            validation_result['errors'].append("CLI execution timed out")
            logger.error(f"❌ {tool_id} CLI execution timed out")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation exception: {e}")
            logger.error(f"❌ {tool_id} validation failed: {e}")
        
        return validation_result
    
    def validate_tool_output(self, output_data: Dict[str, Any], tool_id: str) -> bool:
        """Validate tool output data."""
        
        try:
            # Common validation checks
            if not isinstance(output_data, dict):
                return False
            
            # Check for execution metadata
            if '_execution_metadata' in output_data:
                metadata = output_data['_execution_metadata']
                if metadata.get('tool_id') != tool_id:
                    return False
            
            # Tool-specific validation
            if tool_id == 'integrated_gradients':
                return 'attributions' in output_data or 'attribution_scores' in output_data
                
            elif tool_id == 'deepshap_explainer':
                return 'shap_values' in output_data or 'feature_importance' in output_data
                
            elif tool_id == 'semantic_labeling':
                return 'semantic_labels' in output_data or 'concept_labels' in output_data
                
            elif tool_id == 'ace_concepts':
                return 'extracted_concepts' in output_data or 'concepts' in output_data
                
            elif tool_id in ['neuron_coverage', 'surprise_coverage']:
                return 'coverage_statistics' in output_data or 'layer_coverage' in output_data
                
            elif tool_id == 'wasserstein_distance':
                return 'wasserstein_distance' in output_data
                
            elif tool_id == 'emd_heatmap':
                return 'emd_distance' in output_data
                
            elif tool_id == 'transformerlens_adapter':
                return 'neuron_activations' in output_data or 'model_name' in output_data
                
            elif tool_id == 'residual_stream_comparator':
                return 'comparison_summary' in output_data
                
            elif tool_id == 'tcav_plus_comparator':
                return 'similarity_metrics' in output_data
            
            # Generic validation - check for non-empty output
            return len(output_data) > 0
            
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run full CLI validation for all tools."""
        
        logger.info("🚀 Starting full CLI validation")
        
        start_time = time.time()
        validation_results = {}
        
        # Test CLI list command first
        logger.info("Testing CLI list command...")
        try:
            result = subprocess.run(
                [sys.executable, str(self.cli_script), 'list'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_root)
            )
            
            if result.returncode == 0:
                logger.info("✅ CLI list command successful")
            else:
                logger.error("❌ CLI list command failed")
        except Exception as e:
            logger.error(f"❌ CLI list command exception: {e}")
        
        # Validate each tool
        for tool_id, tool_config in self.tools_registry.items():
            logger.info(f"\n--- Validating {tool_id} ---")
            
            validation_result = self.validate_tool_cli(tool_id, tool_config)
            validation_results[tool_id] = validation_result
        
        total_time = time.time() - start_time
        
        # Generate summary
        successful = sum(1 for r in validation_results.values() if r['success'])
        output_valid = sum(1 for r in validation_results.values() if r['output_valid'])
        total_tools = len(validation_results)
        
        summary = {
            'total_tools_tested': total_tools,
            'successful_executions': successful,
            'valid_outputs': output_valid,
            'success_rate': successful / total_tools * 100 if total_tools > 0 else 0,
            'output_validity_rate': output_valid / total_tools * 100 if total_tools > 0 else 0,
            'total_validation_time': total_time,
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"\n🎯 CLI Validation Summary:")
        logger.info(f"   Total tools tested: {summary['total_tools_tested']}")
        logger.info(f"   Successful executions: {summary['successful_executions']}")
        logger.info(f"   Valid outputs: {summary['valid_outputs']}")
        logger.info(f"   Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"   Output validity rate: {summary['output_validity_rate']:.1f}%")
        logger.info(f"   Total time: {summary['total_validation_time']:.2f}s")
        
        return {
            'summary': summary,
            'detailed_results': validation_results
        }
    
    def save_validation_report(self, results: Dict[str, Any], 
                              filename: str = "cli_validation_report.json"):
        """Save validation report to file."""
        
        try:
            with open(self.project_root / filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📄 Validation report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

def main():
    """Main CLI validation function."""
    
    print("🧪 NeuronMap CLI Validator")
    print("=" * 50)
    
    validator = CLIValidator()
    
    if not validator.tools_registry:
        print("❌ No tools found in registry. Cannot run validation.")
        sys.exit(1)
    
    # Run full validation
    results = validator.run_full_validation()
    
    # Save results
    validator.save_validation_report(results)
    
    # Exit with appropriate code
    summary = results['summary']
    if summary['success_rate'] < 100:
        print(f"\n⚠️ Some CLI validations failed. Success rate: {summary['success_rate']:.1f}%")
        sys.exit(1)
    else:
        print(f"\n🎉 All CLI validations passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
