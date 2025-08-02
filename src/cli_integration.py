"""
CLI Integration for NeuronMap Interpretability Tools
==================================================

Command-line interface for executing interpretability tools defined
in the tools registry with proper validation and security.
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time

# Import all tool modules
from ..analysis.interpretability.ig_explainer import create_integrated_gradients_explainer
from ..analysis.interpretability.shap_explainer import create_shap_explainer
from ..analysis.interpretability.semantic_labeling import create_semantic_labeler
from ..analysis.concepts.ace_extractor import create_ace_extractor
from ..analysis.concepts.tcav_plus_comparator import create_tcav_plus_comparator
from ..analysis.testing.coverage_tracker import create_coverage_tracker
from ..analysis.testing.surprise_tracker import create_surprise_tracker
from ..analysis.metrics.wasserstein_comparator import create_wasserstein_comparator
from ..analysis.metrics.emd_heatmap import create_emd_comparator
from ..analysis.mechanistic.transformerlens_adapter import create_transformerlens_adapter
from ..analysis.mechanistic.residual_stream_comparator import create_residual_stream_comparator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuronMapCLI:
    """Command-line interface for NeuronMap interpretability tools."""
    
    def __init__(self):
        self.tools_registry = {}
        self.security_config = {}
        self.load_registry()
    
    def load_registry(self):
        """Load tools registry from configuration file."""
        registry_path = Path(__file__).parent.parent / "configs" / "tools_registry.yaml"
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = yaml.safe_load(f)
            
            # Extract tools from categories structure
            categories = registry_data.get('categories', {})
            tools = {}
            
            for category_name, category_data in categories.items():
                category_tools = category_data.get('tools', {})
                tools.update(category_tools)
            
            self.tools_registry = tools
            self.security_config = registry_data.get('security', {})
            
            logger.info(f"Loaded {len(self.tools_registry)} tools from registry")
            
        except Exception as e:
            logger.error(f"Failed to load tools registry: {e}")
            sys.exit(1)
    
    def create_tool(self, tool_id: str, config: Optional[Dict[str, Any]] = None):
        """Create a tool instance based on tool ID."""
        tool_creators = {
            'integrated_gradients': create_integrated_gradients_explainer,
            'deepshap_explainer': create_shap_explainer,
            'semantic_labeling': create_semantic_labeler,
            'ace_concepts': create_ace_extractor,
            'tcav_plus_comparator': create_tcav_plus_comparator,
            'neuron_coverage': create_coverage_tracker,
            'surprise_coverage': create_surprise_tracker,
            'wasserstein_distance': create_wasserstein_comparator,
            'emd_heatmap': create_emd_comparator,
            'transformerlens_adapter': create_transformerlens_adapter,
            'residual_stream_comparator': create_residual_stream_comparator
        }
        
        if tool_id not in tool_creators:
            raise ValueError(f"Unknown tool ID: {tool_id}")
        
        return tool_creators[tool_id](config)
    
    def validate_security(self, tool_id: str, args: Dict[str, Any]) -> bool:
        """Validate security requirements for tool execution."""
        
        # Check if tool is allowed
        allowed_tools = self.security_config.get('allowed_tools', [])
        if allowed_tools and tool_id not in allowed_tools:
            logger.error(f"Tool {tool_id} not in allowed tools list")
            return False
        
        # Check for prompt manipulation protection
        prompt_protection = self.security_config.get('prompt_manipulation_protection', True)
        if prompt_protection:
            # Scan arguments for potential prompt injection
            for key, value in args.items():
                if isinstance(value, str):
                    if self._detect_prompt_manipulation(value):
                        logger.error(f"Potential prompt manipulation detected in {key}")
                        return False
        
        return True
    
    def _detect_prompt_manipulation(self, text: str) -> bool:
        """Detect potential prompt manipulation attempts."""
        
        # Common prompt injection patterns
        suspicious_patterns = [
            "ignore previous instructions",
            "system:",
            "assistant:",
            "human:",
            "###",
            "---",
            "```",
            "forget everything",
            "new instructions",
            "override",
            "jailbreak"
        ]
        
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                return True
        
        return False
    
    def execute_tool(self, tool_id: str, config: Dict[str, Any], 
                    input_data: Any, **kwargs) -> Dict[str, Any]:
        """Execute a tool with given configuration and input."""
        
        # Validate security
        if not self.validate_security(tool_id, {**config, **kwargs}):
            raise SecurityError(f"Security validation failed for tool {tool_id}")
        
        # Create tool instance
        tool = self.create_tool(tool_id, config)
        
        # Initialize tool
        if not tool.initialize():
            raise RuntimeError(f"Failed to initialize tool {tool_id}")
        
        # Execute tool
        if hasattr(tool, 'execute'):
            result = tool.execute(input_data, **kwargs)
        else:
            raise RuntimeError(f"Tool {tool_id} does not support execution")
        
        return result.outputs if result.success else {}
    
    def run_analysis_pipeline(self, pipeline_config: Dict[str, Any], 
                             input_data: Any) -> Dict[str, Any]:
        """Run a pipeline of analysis tools."""
        
        pipeline_results = {}
        
        for step_name, step_config in pipeline_config.items():
            tool_id = step_config.get('tool_id')
            tool_config = step_config.get('config', {})
            tool_kwargs = step_config.get('kwargs', {})
            
            if not tool_id:
                logger.warning(f"Step {step_name} missing tool_id, skipping")
                continue
            
            try:
                logger.info(f"Executing pipeline step: {step_name} ({tool_id})")
                
                # Use output from previous step if specified
                step_input = input_data
                input_from = step_config.get('input_from')
                if input_from and input_from in pipeline_results:
                    step_input = pipeline_results[input_from]
                
                # Execute tool
                result = self.execute_tool(tool_id, tool_config, step_input, **tool_kwargs)
                pipeline_results[step_name] = result
                
                logger.info(f"Completed step: {step_name}")
                
            except Exception as e:
                logger.error(f"Pipeline step {step_name} failed: {e}")
                pipeline_results[step_name] = {'error': str(e)}
        
        return pipeline_results
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        
        parser = argparse.ArgumentParser(
            description="NeuronMap Interpretability Tools CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Main command
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # List tools command
        list_parser = subparsers.add_parser('list', help='List available tools')
        list_parser.add_argument('--category', help='Filter by category')
        
        # Execute tool command
        exec_parser = subparsers.add_parser('execute', help='Execute a single tool')
        exec_parser.add_argument('tool_id', help='Tool ID to execute')
        exec_parser.add_argument('--config', help='Tool configuration (JSON file or string)')
        exec_parser.add_argument('--input', help='Input data (JSON file or string)')
        exec_parser.add_argument('--output', help='Output file path')
        exec_parser.add_argument('--kwargs', help='Additional keyword arguments (JSON)')
        
        # Pipeline command
        pipeline_parser = subparsers.add_parser('pipeline', help='Execute analysis pipeline')
        pipeline_parser.add_argument('pipeline_config', help='Pipeline configuration file')
        pipeline_parser.add_argument('--input', help='Input data (JSON file or string)')
        pipeline_parser.add_argument('--output', help='Output file path')
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate tool configuration')
        validate_parser.add_argument('tool_id', help='Tool ID to validate')
        validate_parser.add_argument('--config', help='Configuration to validate')
        
        return parser
    
    def cmd_list_tools(self, args):
        """List available tools."""
        
        print("Available NeuronMap Interpretability Tools:")
        print("=" * 50)
        
        for tool_id, tool_config in self.tools_registry.items():
            category = tool_config.get('category', 'unknown')
            
            if args.category and args.category.lower() != category.lower():
                continue
            
            name = tool_config.get('name', tool_id)
            description = tool_config.get('description', 'No description')
            
            print(f"\nTool ID: {tool_id}")
            print(f"Name: {name}")
            print(f"Category: {category}")
            print(f"Description: {description}")
            
            # Show parameters if available
            if 'parameters' in tool_config:
                print("Parameters:")
                for param, param_config in tool_config['parameters'].items():
                    param_type = param_config.get('type', 'unknown')
                    required = param_config.get('required', False)
                    default = param_config.get('default', 'None')
                    
                    print(f"  - {param} ({param_type}): required={required}, default={default}")
        
        print("\nUse 'neuronmap execute <tool_id>' to run a specific tool.")
    
    def cmd_execute_tool(self, args):
        """Execute a single tool."""
        
        # Load configuration
        config = {}
        if args.config:
            config = self._load_json_or_file(args.config)
        
        # Load input data
        input_data = None
        if args.input:
            input_data = self._load_json_or_file(args.input)
        
        # Load additional kwargs
        kwargs = {}
        if args.kwargs:
            kwargs = self._load_json_or_file(args.kwargs)
        
        # Execute tool
        try:
            start_time = time.time()
            result = self.execute_tool(args.tool_id, config, input_data, **kwargs)
            execution_time = time.time() - start_time
            
            # Add execution metadata
            result['_execution_metadata'] = {
                'tool_id': args.tool_id,
                'execution_time': execution_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Output result
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            sys.exit(1)
    
    def cmd_run_pipeline(self, args):
        """Run analysis pipeline."""
        
        # Load pipeline configuration
        try:
            with open(args.pipeline_config, 'r') as f:
                if args.pipeline_config.endswith('.yaml') or args.pipeline_config.endswith('.yml'):
                    pipeline_config = yaml.safe_load(f)
                else:
                    pipeline_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pipeline configuration: {e}")
            sys.exit(1)
        
        # Load input data
        input_data = None
        if args.input:
            input_data = self._load_json_or_file(args.input)
        
        # Run pipeline
        try:
            start_time = time.time()
            results = self.run_analysis_pipeline(pipeline_config, input_data)
            execution_time = time.time() - start_time
            
            # Add pipeline metadata
            results['_pipeline_metadata'] = {
                'pipeline_config_file': args.pipeline_config,
                'total_execution_time': execution_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'steps_executed': len(results) - 1  # Subtract metadata
            }
            
            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Pipeline results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            sys.exit(1)
    
    def cmd_validate_tool(self, args):
        """Validate tool configuration."""
        
        if args.tool_id not in self.tools_registry:
            print(f"Error: Unknown tool ID '{args.tool_id}'")
            sys.exit(1)
        
        # Load configuration if provided
        config = {}
        if args.config:
            config = self._load_json_or_file(args.config)
        
        # Validate configuration against tool requirements
        tool_spec = self.tools_registry[args.tool_id]
        
        # Check required parameters
        required_params = []
        for param, param_config in tool_spec.get('parameters', {}).items():
            if param_config.get('required', False):
                required_params.append(param)
        
        missing_params = [p for p in required_params if p not in config]
        
        if missing_params:
            print(f"Validation failed: Missing required parameters: {missing_params}")
            sys.exit(1)
        
        # Check parameter types (basic validation)
        for param, value in config.items():
            if param in tool_spec.get('parameters', {}):
                expected_type = tool_spec['parameters'][param].get('type')
                if expected_type and not self._validate_type(value, expected_type):
                    print(f"Validation warning: Parameter '{param}' type mismatch")
        
        print(f"Configuration for tool '{args.tool_id}' is valid.")
    
    def _load_json_or_file(self, data_or_path: str) -> Any:
        """Load data from JSON string or file."""
        try:
            # Try to parse as JSON string first
            return json.loads(data_or_path)
        except json.JSONDecodeError:
            # Try to load as file
            try:
                with open(data_or_path, 'r') as f:
                    return json.load(f)
            except Exception:
                # Return as string if all else fails
                return data_or_path
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Basic type validation."""
        type_mapping = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown types pass validation
    
    def run(self):
        """Run the CLI application."""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        # Route to appropriate command handler
        if args.command == 'list':
            self.cmd_list_tools(args)
        elif args.command == 'execute':
            self.cmd_execute_tool(args)
        elif args.command == 'pipeline':
            self.cmd_run_pipeline(args)
        elif args.command == 'validate':
            self.cmd_validate_tool(args)

class SecurityError(Exception):
    """Security validation error."""
    pass

def main():
    """Main entry point for CLI."""
    cli = NeuronMapCLI()
    cli.run()

if __name__ == '__main__':
    main()
