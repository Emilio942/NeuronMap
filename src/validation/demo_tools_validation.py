"""
Demo Tools Validation for NeuronMap
==================================

Comprehensive validation system for all interpretability tools.
Tests each tool with GPT-2 and random input to ensure functionality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import traceback
from pathlib import Path
import json
import yaml

try:
    from transformers import GPT2Model, GPT2Tokenizer, AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available for validation")

from ..core.plugin_interface import (
    InterpretabilityPluginBase, 
    create_plugin_factory,
    validate_all_plugins
)

logger = logging.getLogger(__name__)

class DemoToolsValidator:
    """
    Comprehensive validation system for interpretability tools.
    
    Tests all registered tools with standard models and inputs to ensure
    they work correctly and produce valid outputs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.test_results = {}
        self.failed_tools = []
        self.successful_tools = []
        
        # Test configuration
        self.test_models = ['gpt2', 'bert-base-uncased'] if TRANSFORMERS_AVAILABLE else []
        self.max_test_time = self.config.get('max_test_time', 300)  # 5 minutes per tool
        self.verbose = self.config.get('verbose', True)
        
        # Load tools registry
        self.tools_registry = self._load_tools_registry()
        self.plugin_factory = create_plugin_factory()
        
        logger.info(f"Initialized demo tools validator with {len(self.plugin_factory)} tools")
    
    def _load_tools_registry(self) -> Dict[str, Any]:
        """Load tools registry configuration."""
        try:
            registry_path = Path("configs/tools_registry.yaml")
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load tools registry: {e}")
            return {}
    
    def validate_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all registered interpretability tools.
        
        Returns:
            Dictionary with validation results for each tool
        """
        logger.info("Starting comprehensive tool validation")
        
        results = {}
        
        # Test each tool
        for tool_id, tool_info in self.plugin_factory.items():
            logger.info(f"Validating tool: {tool_id}")
            
            try:
                tool_result = self._validate_single_tool(tool_id, tool_info)
                results[tool_id] = tool_result
                
                if tool_result['success']:
                    self.successful_tools.append(tool_id)
                    logger.info(f"✅ Tool {tool_id} validation successful")
                else:
                    self.failed_tools.append(tool_id)
                    logger.error(f"❌ Tool {tool_id} validation failed")
                    
            except Exception as e:
                error_msg = f"Critical validation error for {tool_id}: {e}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                results[tool_id] = {
                    'success': False,
                    'error': error_msg,
                    'traceback': traceback.format_exc()
                }
                self.failed_tools.append(tool_id)
        
        # Generate summary
        summary = self._generate_validation_summary(results)
        results['_summary'] = summary
        
        self.test_results = results
        return results
    
    def _validate_single_tool(self, tool_id: str, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single interpretability tool."""
        start_time = time.time()
        
        result = {
            'tool_id': tool_id,
            'success': False,
            'tests_passed': 0,
            'total_tests': 0,
            'execution_time': 0,
            'errors': [],
            'warnings': [],
            'test_details': {}
        }
        
        try:
            # Create tool instance
            tool_instance = self._create_tool_instance(tool_id, tool_info)
            if tool_instance is None:
                result['errors'].append("Failed to create tool instance")
                return result
            
            # Test 1: Initialization
            result['total_tests'] += 1
            if self._test_initialization(tool_instance, result):
                result['tests_passed'] += 1
            
            # Test 2: Basic execution with GPT-2
            if TRANSFORMERS_AVAILABLE and 'gpt2' in self.test_models:
                result['total_tests'] += 1
                if self._test_gpt2_execution(tool_instance, result):
                    result['tests_passed'] += 1
            
            # Test 3: Random input handling
            result['total_tests'] += 1
            if self._test_random_input(tool_instance, result):
                result['tests_passed'] += 1
            
            # Test 4: Output validation
            result['total_tests'] += 1
            if self._test_output_validation(tool_instance, result):
                result['tests_passed'] += 1
            
            # Test 5: Error handling
            result['total_tests'] += 1
            if self._test_error_handling(tool_instance, result):
                result['tests_passed'] += 1
            
            # Test 6: Security validation
            result['total_tests'] += 1
            if self._test_security_validation(tool_instance, result):
                result['tests_passed'] += 1
            
            # Mark as successful if most tests pass
            result['success'] = result['tests_passed'] >= result['total_tests'] * 0.7
            
        except Exception as e:
            result['errors'].append(f"Validation exception: {e}")
            logger.error(f"Validation failed for {tool_id}: {e}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _create_tool_instance(self, tool_id: str, tool_info: Dict[str, Any]) -> Optional[InterpretabilityPluginBase]:
        """Create an instance of the tool for testing."""
        try:
            # Get tool configuration from registry
            tool_config = self._get_tool_config(tool_id)
            
            # Import the tool class dynamically
            # For now, we'll create instances of the implemented tools
            if tool_id == "integrated_gradients":
                from ..analysis.interpretability.ig_explainer import IntegratedGradientsExplainer
                return IntegratedGradientsExplainer(config=tool_config)
            elif tool_id == "deep_shap":
                from ..analysis.interpretability.shap_explainer import DeepSHAPExplainer
                return DeepSHAPExplainer(config=tool_config)
            elif tool_id == "llm_auto_labeling":
                from ..analysis.interpretability.semantic_labeling import SemanticLabeler
                return SemanticLabeler(config=tool_config)
            elif tool_id == "ace_concepts":
                from ..analysis.concepts.ace_extractor import ACEConceptExtractor
                return ACEConceptExtractor(config=tool_config)
            else:
                logger.warning(f"Tool {tool_id} not implemented yet")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create tool instance for {tool_id}: {e}")
            return None
    
    def _get_tool_config(self, tool_id: str) -> Dict[str, Any]:
        """Get tool configuration from registry."""
        categories = self.tools_registry.get('categories', {})
        for category_name, category in categories.items():
            if tool_id in category.get('tools', {}):
                return category['tools'][tool_id].get('test_config', {})
        return {}
    
    def _test_initialization(self, tool: InterpretabilityPluginBase, 
                           result: Dict[str, Any]) -> bool:
        """Test tool initialization."""
        try:
            success = tool.initialize()
            result['test_details']['initialization'] = {
                'success': success,
                'initialized': tool.initialized
            }
            return success
        except Exception as e:
            result['errors'].append(f"Initialization failed: {e}")
            result['test_details']['initialization'] = {'success': False, 'error': str(e)}
            return False
    
    def _test_gpt2_execution(self, tool: InterpretabilityPluginBase,
                            result: Dict[str, Any]) -> bool:
        """Test tool execution with GPT-2 model."""
        try:
            # Load GPT-2 model
            model = GPT2Model.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            # Prepare test input
            test_text = "This is a test sentence for neural network analysis."
            inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=512)
            
            # Execute tool
            if hasattr(tool, 'execute_with_validation'):
                exec_result = tool.execute_with_validation(model=model, inputs=inputs['input_ids'])
            else:
                exec_result = tool.test_mode_execution()
            
            success = exec_result.success if hasattr(exec_result, 'success') else True
            
            result['test_details']['gpt2_execution'] = {
                'success': success,
                'execution_time': getattr(exec_result, 'execution_time', 0),
                'has_outputs': bool(getattr(exec_result, 'outputs', {}))
            }
            
            return success
            
        except Exception as e:
            result['errors'].append(f"GPT-2 execution failed: {e}")
            result['test_details']['gpt2_execution'] = {'success': False, 'error': str(e)}
            return False
    
    def _test_random_input(self, tool: InterpretabilityPluginBase,
                          result: Dict[str, Any]) -> bool:
        """Test tool with random input data."""
        try:
            # Create simple random model and input
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleModel()
            random_input = torch.randn(4, 10)
            
            # Execute tool in test mode
            exec_result = tool.test_mode_execution({'model': model, 'test_input': random_input})
            
            success = exec_result.success if hasattr(exec_result, 'success') else True
            
            result['test_details']['random_input'] = {
                'success': success,
                'execution_time': getattr(exec_result, 'execution_time', 0)
            }
            
            return success
            
        except Exception as e:
            result['errors'].append(f"Random input test failed: {e}")
            result['test_details']['random_input'] = {'success': False, 'error': str(e)}
            return False
    
    def _test_output_validation(self, tool: InterpretabilityPluginBase,
                               result: Dict[str, Any]) -> bool:
        """Test output validation functionality."""
        try:
            # Test with valid output
            valid_output = self._create_mock_valid_output(tool.tool_id)
            valid_test = tool.validate_output(valid_output)
            
            # Test with invalid output
            invalid_output = {'invalid': 'data'}
            invalid_test = not tool.validate_output(invalid_output)
            
            success = valid_test and invalid_test
            
            result['test_details']['output_validation'] = {
                'success': success,
                'valid_output_accepted': valid_test,
                'invalid_output_rejected': invalid_test
            }
            
            return success
            
        except Exception as e:
            result['errors'].append(f"Output validation test failed: {e}")
            result['test_details']['output_validation'] = {'success': False, 'error': str(e)}
            return False
    
    def _test_error_handling(self, tool: InterpretabilityPluginBase,
                            result: Dict[str, Any]) -> bool:
        """Test error handling capabilities."""
        try:
            # Test with invalid inputs
            try:
                exec_result = tool.execute_with_validation(model=None, inputs=None)
                graceful_failure = not exec_result.success
            except Exception:
                graceful_failure = True  # Acceptable to throw exception
            
            result['test_details']['error_handling'] = {
                'success': graceful_failure,
                'handles_invalid_input': graceful_failure
            }
            
            return graceful_failure
            
        except Exception as e:
            result['errors'].append(f"Error handling test failed: {e}")
            result['test_details']['error_handling'] = {'success': False, 'error': str(e)}
            return False
    
    def _test_security_validation(self, tool: InterpretabilityPluginBase,
                                 result: Dict[str, Any]) -> bool:
        """Test security validation functionality."""
        try:
            # Test security validation
            security_valid = tool.validate_security()
            
            # Check security attributes
            has_security_attrs = (
                hasattr(tool, 'allow_defer') and not tool.allow_defer and
                hasattr(tool, 'execution_reason') and tool.execution_reason != "irrelevant" and
                hasattr(tool, 'mandatory_execution') and tool.mandatory_execution
            )
            
            success = security_valid and has_security_attrs
            
            result['test_details']['security_validation'] = {
                'success': success,
                'security_valid': security_valid,
                'has_security_attrs': has_security_attrs
            }
            
            return success
            
        except Exception as e:
            result['errors'].append(f"Security validation test failed: {e}")
            result['test_details']['security_validation'] = {'success': False, 'error': str(e)}
            return False
    
    def _create_mock_valid_output(self, tool_id: str) -> Dict[str, Any]:
        """Create mock valid output for testing."""
        if tool_id == "integrated_gradients":
            return {
                'attributions': torch.randn(1, 10),
                'baseline': torch.zeros(1, 10),
                'method_config': {'n_steps': 50}
            }
        elif tool_id == "deep_shap":
            return {
                'shap_values': np.random.randn(1, 10),
                'expected_values': np.array([0.5]),
                'feature_importance': {'mean_absolute_shap': 0.1}
            }
        elif tool_id == "llm_auto_labeling":
            return {
                'semantic_labels': {'cluster_1': {'label': 'test', 'confidence': 0.8, 'description': 'test'}},
                'confidence_scores': {'cluster_1': 0.8}
            }
        elif tool_id == "ace_concepts":
            return {
                'concepts': {'concept_1': {'name': 'test', 'importance_score': 0.8, 'coherence_score': 0.7}},
                'concept_scores': {'concept_1': 0.8}
            }
        else:
            return {'test_output': 'valid'}
    
    def _generate_validation_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        total_tools = len(results)
        successful_tools = sum(1 for r in results.values() if r.get('success', False))
        failed_tools = total_tools - successful_tools
        
        # Calculate average test success rate
        total_tests = sum(r.get('total_tests', 0) for r in results.values())
        passed_tests = sum(r.get('tests_passed', 0) for r in results.values())
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Collect common error patterns
        all_errors = []
        for r in results.values():
            all_errors.extend(r.get('errors', []))
        
        summary = {
            'total_tools_tested': total_tools,
            'successful_tools': successful_tools,
            'failed_tools': failed_tools,
            'overall_success_rate': f"{success_rate:.1f}%",
            'total_tests_run': total_tests,
            'tests_passed': passed_tests,
            'most_common_errors': self._get_most_common_errors(all_errors),
            'successful_tool_list': self.successful_tools,
            'failed_tool_list': self.failed_tools,
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def _get_most_common_errors(self, errors: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Get most common error patterns."""
        error_counts = {}
        
        for error in errors:
            # Extract error type (before first colon)
            error_type = error.split(':')[0] if ':' in error else error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'error_type': error, 'count': count} for error, count in sorted_errors[:top_k]]
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate comprehensive validation report."""
        if not self.test_results:
            return "No validation results available. Run validate_all_tools() first."
        
        # Create report
        report_lines = [
            "=" * 80,
            "NeuronMap Interpretability Tools Validation Report",
            "=" * 80,
            "",
            f"Validation completed at: {self.test_results.get('_summary', {}).get('validation_timestamp', 'Unknown')}",
            ""
        ]
        
        # Summary section
        summary = self.test_results.get('_summary', {})
        report_lines.extend([
            "SUMMARY:",
            f"- Total tools tested: {summary.get('total_tools_tested', 0)}",
            f"- Successful tools: {summary.get('successful_tools', 0)}",
            f"- Failed tools: {summary.get('failed_tools', 0)}",
            f"- Overall success rate: {summary.get('overall_success_rate', '0%')}",
            f"- Total tests run: {summary.get('total_tests_run', 0)}",
            f"- Tests passed: {summary.get('tests_passed', 0)}",
            ""
        ])
        
        # Successful tools
        successful_tools = summary.get('successful_tool_list', [])
        if successful_tools:
            report_lines.extend([
                "SUCCESSFUL TOOLS:",
                *[f"✅ {tool}" for tool in successful_tools],
                ""
            ])
        
        # Failed tools
        failed_tools = summary.get('failed_tool_list', [])
        if failed_tools:
            report_lines.extend([
                "FAILED TOOLS:",
                *[f"❌ {tool}" for tool in failed_tools],
                ""
            ])
        
        # Detailed results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)
        
        for tool_id, result in self.test_results.items():
            if tool_id == '_summary':
                continue
                
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            report_lines.extend([
                f"\n{tool_id}: {status}",
                f"  Tests passed: {result.get('tests_passed', 0)}/{result.get('total_tests', 0)}",
                f"  Execution time: {result.get('execution_time', 0):.2f}s"
            ])
            
            if result.get('errors'):
                report_lines.append("  Errors:")
                for error in result['errors']:
                    report_lines.append(f"    - {error}")
            
            if result.get('warnings'):
                report_lines.append("  Warnings:")
                for warning in result['warnings']:
                    report_lines.append(f"    - {warning}")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text
    
    def save_results(self, output_path: Path):
        """Save validation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert torch tensors to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.test_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Validation results saved to {output_path}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

def run_validation(config: Optional[Dict[str, Any]] = None, 
                  output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run complete validation of all interpretability tools.
    
    Args:
        config: Validation configuration
        output_dir: Directory to save results and reports
        
    Returns:
        Validation results dictionary
    """
    validator = DemoToolsValidator(config)
    
    # Run validation
    results = validator.validate_all_tools()
    
    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        validator.save_results(output_dir / "validation_results.json")
        
        # Generate and save report
        report = validator.generate_report(output_dir / "validation_report.txt")
        
        logger.info(f"Validation completed. Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    # Run validation when script is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate NeuronMap interpretability tools")
    parser.add_argument("--output-dir", type=str, default="validation_outputs",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    config = {'verbose': args.verbose}
    results = run_validation(config, Path(args.output_dir))
    
    # Print summary
    summary = results.get('_summary', {})
    print(f"\nValidation completed:")
    print(f"✅ Successful tools: {summary.get('successful_tools', 0)}")
    print(f"❌ Failed tools: {summary.get('failed_tools', 0)}")
    print(f"Overall success rate: {summary.get('overall_success_rate', '0%')}")
